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

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, PrefillOutcome};
use crate::session::plan_prefill;
use crate::sampling::sample_next;

pub const GPU_LOCK_PATH: &str = "/tmp/aien-gb10.lock";
/// How long `generate` waits for the chip lock before answering busy.
pub const LOCK_WAIT: Duration = Duration::from_secs(30);
/// How long a background prefill chunk waits for the chip lock before the scheduler retries.
pub const PREFILL_LOCK_WAIT: Duration = Duration::from_secs(2);
/// Host memory the load needs free (the model is 30.5 GB resident plus the KV pool and
/// loader transients; the rule leaves room for another 30 GB job).
pub const LOAD_MIN_FREE_GB: f64 = 45.0;

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
    pub fn load(model_dir: &Path, max_context: usize, format: WeightFormat, spin_us: Option<u32>) -> Result<Self, String> {
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

    fn supports_prefill(&self) -> bool {
        true
    }

    fn prefill(&self, tokens: &[u32], max_new: usize) -> Result<PrefillOutcome, EngineError> {
        let max = self.engine.max_context_tokens();
        if tokens.len() >= max {
            return Err(EngineError::PromptTooLong { len: tokens.len(), max });
        }
        // Background work waits only briefly for the chip; the scheduler retries later.
        let _lock = GpuLock::acquire(PREFILL_LOCK_WAIT).map_err(EngineError::Busy)?;
        let t0 = Instant::now();
        self.engine
            .with_engine(|e| -> Result<PrefillOutcome, String> {
                let plan = plan_prefill(e.history(), tokens);
                let mut rewound = false;
                if let Some(n) = plan.rewind_to {
                    if n == 0 {
                        e.reset();
                    } else {
                        e.rewind(n);
                    }
                    rewound = true;
                }
                let end = tokens.len().min(plan.common.saturating_add(max_new));
                let chunk = e.prefill_chunk.max(1);
                for part in tokens[plan.common.min(end)..end].chunks(chunk) {
                    e.step_batch(part)?;
                }
                Ok(PrefillOutcome {
                    cached: end.max(plan.common),
                    reused: plan.common,
                    computed: end.saturating_sub(plan.common),
                    rewound,
                    ms: t0.elapsed().as_millis() as u64,
                })
            })
            .map_err(EngineError::Failed)?
            .map_err(EngineError::Failed)
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
            reused_prompt_tokens: o.reused_prompt_tokens,
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
}
