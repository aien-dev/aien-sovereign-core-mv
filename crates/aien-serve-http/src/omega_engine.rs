//! The Omega engine behind the door (OWN-30B cut 8): Qwen3-30B-A3B on AIEN's own GPU
//! kernels, FP8 weights resident between requests, prefix KV reuse across turns, and the
//! shared-chip lock taken only around each generation.
//!
//! Chip etiquette: `/tmp/aien-gb10.lock` is the hive's GPU lock. The server never holds it
//! while idle; `generate` takes it (waiting up to [`LOCK_WAIT`]) and releases it when the
//! reply is done. If another job holds it past the wait the request fails with
//! [`EngineError::Busy`], which the server turns into HTTP 503.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen3_moe_omega::{self as omega, LoadReport, OmegaEngine, WeightFormat};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::sampling::sample_next;

pub const GPU_LOCK_PATH: &str = "/tmp/aien-gb10.lock";
/// How long `generate` waits for the chip lock before answering busy.
pub const LOCK_WAIT: Duration = Duration::from_secs(30);
/// Host memory the load needs free (the model is 30.5 GB resident plus the KV pool and
/// loader transients; the rule leaves room for another 30 GB job).
pub const LOAD_MIN_FREE_GB: f64 = 45.0;

/// Prefill path of the door (OWN-30B release candidate, W-RC 2026-10-14). Both switches are ON
/// by default: the whole prompt goes through omega's prefill GEMMs and attention in one pass
/// (`set_whole_prompt`), with the router logits and MoE combine on the chip
/// (`set_whole_prompt_device_host`). Chip gate (sc own30b/whole-prompt 7cc2b66): TTFT 512 tokens
/// 7.6 s -> 0.31 s, 2048 tokens 31.4 s -> 0.84 s; tokens differ from the chunked path only by FP8
/// activation rounding. Off: `--whole-prompt off` / `AIEN_SERVE_WHOLE_PROMPT=0` (the chunked
/// prefill of cut 10), `--prefill-device-host off` / `AIEN_SERVE_WP_DEVICE_HOST=0` (host routing
/// and combine). The cached 30 GB weight copy and speculative decoding stay off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillMode {
    pub whole_prompt: bool,
    pub device_host: bool,
}

impl Default for PrefillMode {
    fn default() -> Self {
        Self { whole_prompt: true, device_host: true }
    }
}

/// Environment switch for [`PrefillMode::whole_prompt`].
pub const WHOLE_PROMPT_ENV: &str = "AIEN_SERVE_WHOLE_PROMPT";
/// Environment switch for [`PrefillMode::device_host`].
pub const WP_DEVICE_HOST_ENV: &str = "AIEN_SERVE_WP_DEVICE_HOST";

/// `on|off|1|0|true|false` (anything else is refused, so a typo never silently flips a path).
pub fn parse_switch(name: &str, v: &str) -> Result<bool, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(format!("{name} must be on or off, got {other:?}")),
    }
}

impl PrefillMode {
    /// Defaults overridden by the environment (`None` = unset).
    pub fn from_env(whole_prompt: Option<&str>, device_host: Option<&str>) -> Result<Self, String> {
        let mut m = Self::default();
        if let Some(v) = whole_prompt {
            m.whole_prompt = parse_switch(WHOLE_PROMPT_ENV, v)?;
        }
        if let Some(v) = device_host {
            m.device_host = parse_switch(WP_DEVICE_HOST_ENV, v)?;
        }
        Ok(m)
    }

    /// What the engine runs: the device router/combine only exists inside the whole-prompt path.
    pub fn effective(self) -> Self {
        Self { whole_prompt: self.whole_prompt, device_host: self.whole_prompt && self.device_host }
    }
}

/// An exclusive hold of the hive's GPU lock (released on drop).
pub struct GpuLock(#[allow(dead_code)] File);

impl GpuLock {
    /// Takes the lock, polling until `wait` has passed. `Err` = still held by someone else.
    pub fn acquire(wait: Duration) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(GPU_LOCK_PATH)
            .map_err(|e| format!("{GPU_LOCK_PATH}: {e}"))?;
        let t0 = Instant::now();
        loop {
            // SAFETY: flock on a valid open descriptor; non-blocking so the wait is bounded.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                return Ok(Self(file));
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(format!("flock {GPU_LOCK_PATH}: {err}"));
            }
            if t0.elapsed() >= wait {
                return Err(format!("GPU lock held by another job for {} s", wait.as_secs()));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// `MemAvailable` from /proc/meminfo in GB (0 when unreadable).
pub fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

pub struct OmegaDoor {
    engine: OmegaEngine,
    lock_wait: Duration,
}

impl OmegaDoor {
    /// Loads the checkpoint onto Omega (under the GPU lock, released before returning) with a
    /// KV for `max_context` tokens. Refuses with less than [`LOAD_MIN_FREE_GB`] available.
    pub fn load(model_dir: &Path, max_context: usize, format: WeightFormat, spin_us: Option<u32>, prefill: PrefillMode) -> Result<Self, String> {
        let avail = mem_available_gb();
        if avail < LOAD_MIN_FREE_GB {
            return Err(format!("{avail:.1} GB available, the load needs {LOAD_MIN_FREE_GB} GB free (another model may be loaded)"));
        }
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        if format.fp8_experts() && !omega::OmegaModelUploader::uses_fp8_api() {
            return Err("this binary has no FP8 GEMV linked; use --weights bf16 or link the integration omega".into());
        }
        if let Some(us) = spin_us {
            aien_omega_gpu::set_spin_us(us).map_err(|e| e.to_string())?;
        }
        let _lock = GpuLock::acquire(LOCK_WAIT)?;
        let t0 = Instant::now();
        let mut n = 0usize;
        let engine = OmegaEngine::load_with(model_dir, max_context, format, &mut |what| {
            n += 1;
            if n.is_multiple_of(24) || what.starts_with("kv") || what == "lm_head" {
                eprintln!("aien-serve-http: omega load [{:.0}s] {what}", t0.elapsed().as_secs_f64());
            }
        })?;
        // Prefill path (W-RC): switched on while the load still holds the chip lock. A refusal is
        // fatal: the operator asked for the path (by default), so say how to turn it off.
        let prefill = prefill.effective();
        let t_wp = Instant::now();
        engine.with_engine(|e| -> Result<(), String> {
            e.set_whole_prompt(prefill.whole_prompt)
                .map_err(|m| format!("{m} (start with --whole-prompt off to use the chunked prefill)"))?;
            if prefill.whole_prompt {
                e.set_whole_prompt_device_host(prefill.device_host)
                    .map_err(|m| format!("{m} (start with --prefill-device-host off to keep routing on the host)"))?;
            }
            Ok(())
        })??;
        let (wp_on, wp_dev) = engine.with_engine(|e| (e.whole_prompt_on(), e.whole_prompt_device_host_on()))?;
        eprintln!(
            "aien-serve-http: whole_prompt={} prefill_device_host={} (set in {:.2} s)",
            if wp_on { "on" } else { "off" },
            if wp_dev { "on" } else { "off" },
            t_wp.elapsed().as_secs_f64()
        );
        let r = &engine.load_report;
        let (attn_path, prefill_chunk, router_path, lm_head_path, prefill_path) = engine.with_engine(|e| {
            (
                e.attention_path(),
                e.prefill_chunk,
                if e.device_routing { "device" } else { "host" },
                if matches!(e.weights().lm_head, omega::LmHead::Gemv(_)) { "gemv" } else { "dense" },
                if e.prefill_gemm { "gemm" } else { "gemv" },
            )
        })?;
        eprintln!(
            "aien-serve-http: omega engine ready in {:.1} s: {:?} weights, {:.2} GB resident, {:.2} GB host, peak rss {:.2} GB, window_api={} fp8_api={} dattn_api={} attention={attn_path} router={router_path} lm_head={lm_head_path} prefill={prefill_path} prefill_chunk={prefill_chunk} spin_us={:?}",
            r.seconds,
            r.format,
            r.resident_bytes as f64 / 1e9,
            r.host_bytes as f64 / 1e9,
            r.peak_rss_bytes.unwrap_or(0) as f64 / 1e9,
            r.window_api,
            omega::OmegaModelUploader::uses_fp8_api(),
            omega::OmegaModelUploader::uses_dattn_api(),
            aien_omega_gpu::spin_us(),
        );
        Ok(Self { engine, lock_wait: LOCK_WAIT })
    }

    pub fn load_report(&self) -> &LoadReport {
        &self.engine.load_report
    }

    /// Tokens whose KV is resident from the last generation (prefix reuse candidates).
    pub fn history_len(&self) -> usize {
        self.engine.with_engine(|e| e.history().len()).unwrap_or(0)
    }
}

impl Engine for OmegaDoor {
    fn kind(&self) -> &str {
        self.engine.kind()
    }

    fn max_context_tokens(&self) -> usize {
        self.engine.max_context_tokens()
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        let max = self.engine.max_context_tokens();
        if prompt.is_empty() {
            return Err(EngineError::Failed("empty prompt".into()));
        }
        if prompt.len() >= max {
            return Err(EngineError::PromptTooLong { len: prompt.len(), max });
        }
        let q = omega::GenerateParams {
            max_new_tokens: p.max_new_tokens,
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
            stop_token_ids: p.stop_token_ids.clone(),
        };
        let t_lock = Instant::now();
        let _lock = GpuLock::acquire(self.lock_wait).map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        // The server's own sampler, so CPU and Omega engines pick tokens the same way.
        let mut sampler = |logits: &[f32], step: usize| sample_next(logits, p, step);
        let o = self.engine.generate_with(prompt, &q, &mut sampler, on_token).map_err(EngineError::Failed)?;
        eprintln!(
            "aien-serve-http: omega: lock wait {lock_wait_ms} ms, prompt {} tok, {} reused from the previous turn, prefill {} ms, decode {} tok in {} ms",
            o.prompt_tokens, o.reused_prompt_tokens, o.prefill_ms, o.emitted_tokens, o.decode_ms
        );
        Ok(GenerateOutcome {
            emitted_tokens: o.emitted_tokens,
            finish: match o.finish {
                omega::FinishReason::Stop => FinishReason::Stop,
                omega::FinishReason::Length => FinishReason::Length,
                omega::FinishReason::Cancelled => FinishReason::Cancelled,
            },
            prefill_ms: o.prefill_ms,
            decode_ms: o.decode_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_lock_is_exclusive_and_released_on_drop() {
        // Use a private path so the test never touches the hive's real lock.
        let dir = std::env::temp_dir().join(format!("own30b-lock-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lock");
        let open = || OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).unwrap();
        let a = open();
        // SAFETY: valid descriptors, non-blocking flock.
        assert_eq!(unsafe { libc::flock(a.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        let b = open();
        assert_ne!(unsafe { libc::flock(b.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0, "second holder must be refused");
        drop(a);
        assert_eq!(unsafe { libc::flock(b.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0, "released on drop");
        drop(b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mem_available_reads_proc() {
        assert!(mem_available_gb() > 0.0);
    }

    #[test]
    fn prefill_mode_defaults_on_and_env_switches_off() {
        assert_eq!(PrefillMode::default(), PrefillMode { whole_prompt: true, device_host: true });
        assert_eq!(PrefillMode::from_env(None, None).unwrap(), PrefillMode::default());
        let m = PrefillMode::from_env(Some("0"), None).unwrap();
        assert!(!m.whole_prompt && m.device_host);
        // The device router/combine lives inside the whole-prompt path: off with it.
        assert_eq!(m.effective(), PrefillMode { whole_prompt: false, device_host: false });
        let m = PrefillMode::from_env(Some("on"), Some("off")).unwrap();
        assert_eq!(m.effective(), PrefillMode { whole_prompt: true, device_host: false });
        assert!(PrefillMode::from_env(Some("of"), None).is_err());
        assert!(parse_switch("--whole-prompt", "maybe").is_err());
        assert!(parse_switch("--whole-prompt", " TRUE ").unwrap());
    }
}
