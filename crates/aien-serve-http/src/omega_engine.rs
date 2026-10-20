//! The Omega engine behind the door (OWN-30B cut 8): Qwen3-30B-A3B on AIEN's own GPU
//! kernels, FP8 weights resident between requests, prefix KV reuse across turns, and the
//! shared-chip lock taken only around each generation.
//!
//! Chip etiquette: `/tmp/aien-gb10.lock` is the hive's GPU lock. The server never holds it
//! while idle; `generate` takes it (waiting up to [`LOCK_WAIT`]) and releases it when the
//! reply is done. On release it hands the lock to any process already queued on it (up to
//! [`GPU_LOCK_YIELD`]; flock has no queue order, L188). If another job holds it past the wait
//! the request fails with [`EngineError::Busy`], which the server turns into HTTP 503.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use aien_inference_abi::qwen3_moe_omega::{self as omega, LoadReport, OmegaEngine, WeightFormat};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::kv_persist::PersistState;
use crate::sampling::SamplingObserver;

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
        Self {
            whole_prompt: true,
            device_host: true,
        }
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
        Self {
            whole_prompt: self.whole_prompt,
            device_host: self.whole_prompt && self.device_host,
        }
    }
}

/// An exclusive hold of the hive's GPU lock (released on drop).
pub struct GpuLock(File);

impl GpuLock {
    /// Takes the lock, polling until `wait` has passed. `Err` = still held by someone else.
    pub fn acquire(wait: Duration) -> Result<Self, String> {
        Self::acquire_at(Path::new(GPU_LOCK_PATH), wait)
    }

    /// [`Self::acquire`] on the lock file at `path` (tests pass a private file).
    pub fn acquire_at(path: &Path, wait: Duration) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let t0 = Instant::now();
        loop {
            // SAFETY: flock on a valid open descriptor; non-blocking so the wait is bounded.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                return Ok(Self(file));
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(format!("flock {}: {err}", path.display()));
            }
            if t0.elapsed() >= wait {
                return Err(format!(
                    "GPU lock held by another job for {} s",
                    wait.as_secs()
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// Dropping the lock hands it to a process already queued on it. flock has no queue order, and
/// the kernel takes a woken waiter off its blocked list before the waiter runs, so a process
/// that unlocks and re-locks at once beats a waiting `flock -x` every time (L188: 18 h of
/// back-to-back requests starved every measurement). So when another process is blocked on the
/// lock at release time, the release waits until that process holds the lock, for at most
/// [`GPU_LOCK_YIELD`]. Nobody queued, or /proc/locks unreadable = no delay.
impl Drop for GpuLock {
    fn drop(&mut self) {
        let waiter = other_process_row(&self.0, true);
        // SAFETY: unlock on a valid open descriptor.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
        if !waiter {
            return;
        }
        let t0 = Instant::now();
        let max = gpu_lock_yield();
        while t0.elapsed() < max && !other_process_row(&self.0, false) {
            std::thread::sleep(YIELD_POLL);
        }
    }
}

/// How long a release waits for a queued process to take the lock.
pub const GPU_LOCK_YIELD: Duration = Duration::from_secs(2);
const YIELD_POLL: Duration = Duration::from_millis(2);

/// [`GPU_LOCK_YIELD`], or `AIEN_GPU_LOCK_YIELD_MS` when set (0 turns yielding off).
fn gpu_lock_yield() -> Duration {
    std::env::var("AIEN_GPU_LOCK_YIELD_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(GPU_LOCK_YIELD)
}

/// True when a process other than this one is blocked on (`blocked`) or holds (`!blocked`) the
/// lock `file`, per /proc/locks (the table `spark-window locks` reads). Unreadable table = false.
fn other_process_row(file: &File, blocked: bool) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(meta) = file.metadata() else {
        return false;
    };
    let Ok(table) = std::fs::read_to_string("/proc/locks") else {
        return false;
    };
    lock_row_of_other(&table, meta.dev(), meta.ino(), std::process::id(), blocked)
}

/// True when a /proc/locks `table` has a `->` (blocked) row, or with `!blocked` a holder row,
/// for (`dev`, `ino`) from a pid other than `own_pid`.
fn lock_row_of_other(table: &str, dev: u64, ino: u64, own_pid: u32, blocked: bool) -> bool {
    // Rows: `1: FLOCK  ADVISORY  WRITE 1234 08:01:131075 0 EOF`, blocked ones `1: -> FLOCK ...`.
    let key = format!("{:02x}:{:02x}:{ino}", libc::major(dev), libc::minor(dev));
    table.lines().any(|line| {
        let mut f = line.split_whitespace().skip(1).peekable();
        let arrow = f.next_if_eq(&"->").is_some();
        // class, mode, rw, then pid and dev:ino.
        let mut rest = f.skip(3);
        let pid = rest.next().and_then(|p| p.parse::<u32>().ok());
        arrow == blocked && pid != Some(own_pid) && rest.next() == Some(key.as_str())
    })
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
    /// Declared first: a running store drains its parks before the engine goes.
    persist: PersistState,
    engine: OmegaEngine,
    lock_wait: Duration,
    lock_path: PathBuf,
    /// Prompt tokens the last generation reused (RAM or restored prefix).
    reused_last: AtomicUsize,
}

impl OmegaDoor {
    /// A door over an already built engine, taking the GPU lock at `lock_path` around each
    /// generation (tests: a private lock file and the host test engine). Persistence off.
    pub fn from_parts(engine: OmegaEngine, lock_path: PathBuf, lock_wait: Duration) -> Self {
        Self {
            persist: PersistState::Off,
            engine,
            lock_wait,
            lock_path,
            reused_last: AtomicUsize::new(0),
        }
    }

    /// Installs the persistence state (`kv_persist::open`), before the door is shared.
    pub fn set_persistence(&mut self, state: PersistState) {
        self.persist = state;
    }

    pub fn persistence(&self) -> &PersistState {
        &self.persist
    }

    /// The engine's run settings that can change a K/V byte (persistent KV identity).
    pub fn kv_switches(&self) -> omega::OmegaKvSwitches {
        self.engine
            .with_engine(|e| e.kv_switches())
            .expect("engine lock")
    }

    /// The loaded model config.
    pub fn config(&self) -> omega::Qwen3MoeConfig {
        self.engine
            .with_engine(|e| e.config().clone())
            .expect("engine lock")
    }

    /// Tokens whose KV the engine holds (the next turn's reuse candidates).
    pub fn history(&self) -> Vec<u32> {
        self.engine
            .with_engine(|e| e.history().to_vec())
            .unwrap_or_default()
    }

    /// Prompt tokens the last generation did not prefill (kept in RAM or restored).
    pub fn reused_last(&self) -> usize {
        self.reused_last.load(Ordering::Relaxed)
    }

    /// The K/V of positions `0..len` as the persistent KV payload, under the GPU lock (the chip
    /// check compares the uninterrupted and the restored run byte for byte).
    pub fn export_kv_for_check(&self, len: usize) -> Result<Vec<u8>, String> {
        let _lock = GpuLock::acquire_at(&self.lock_path, self.lock_wait)?;
        self.engine.with_engine(|e| e.export_kv(0, len))?
    }

    /// Imports `payload` as the K/V of `tokens` (positions `0..tokens.len()`), under the GPU lock,
    /// exactly as a restore does (the chip check times it and round-trips it against
    /// [`Self::export_kv_for_check`]). On error the engine is reset.
    pub fn restore_kv_for_check(&self, tokens: &[u32], payload: &[u8]) -> Result<(), String> {
        let _lock = GpuLock::acquire_at(&self.lock_path, self.lock_wait)?;
        self.engine.with_engine(|e| e.restore_kv(tokens, payload))?
    }

    /// Forgets the KV the engine holds (the next prompt is prefilled, or restored from the store).
    pub fn reset_kv(&self) {
        let _ = self.engine.with_engine(|e| e.reset());
    }
    /// Loads the checkpoint onto Omega (under the GPU lock, released before returning) with a
    /// KV for `max_context` tokens. Refuses with less than [`LOAD_MIN_FREE_GB`] available.
    pub fn load(
        model_dir: &Path,
        max_context: usize,
        format: WeightFormat,
        spin_us: Option<u32>,
        prefill: PrefillMode,
    ) -> Result<Self, String> {
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
                eprintln!(
                    "aien-serve-http: omega load [{:.0}s] {what}",
                    t0.elapsed().as_secs_f64()
                );
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
        let (wp_on, wp_dev) =
            engine.with_engine(|e| (e.whole_prompt_on(), e.whole_prompt_device_host_on()))?;
        eprintln!(
            "aien-serve-http: whole_prompt={} prefill_device_host={} (set in {:.2} s)",
            if wp_on { "on" } else { "off" },
            if wp_dev { "on" } else { "off" },
            t_wp.elapsed().as_secs_f64()
        );
        let r = &engine.load_report;
        let (attn_path, prefill_chunk, router_path, lm_head_path, prefill_path) = engine
            .with_engine(|e| {
                (
                    e.attention_path(),
                    e.prefill_chunk,
                    if e.device_routing { "device" } else { "host" },
                    if matches!(e.weights().lm_head, omega::LmHead::Gemv(_)) {
                        "gemv"
                    } else {
                        "dense"
                    },
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
        Ok(Self::from_parts(
            engine,
            PathBuf::from(GPU_LOCK_PATH),
            LOCK_WAIT,
        ))
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
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max,
            });
        }
        let q = omega::GenerateParams {
            max_new_tokens: p.max_new_tokens,
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
            stop_token_ids: p.stop_token_ids.clone(),
        };
        // Persistent KV (sc#505), off unless AIEN_KV_STORE_DIR asked for it: look up and load
        // the longest stored prefix that beats the RAM prefix, without the chip lock.
        let persist = match &self.persist {
            PersistState::On(pk) => Some(pk.as_ref()),
            _ => None,
        };
        let fetched = persist.filter(|pk| pk.restore_on()).and_then(|pk| {
            let in_ram = self
                .engine
                .with_engine(|e| e.reusable_prefix(prompt))
                .unwrap_or(0);
            pk.fetch(prompt, in_ram)
        });
        let t_lock = Instant::now();
        let lock =
            GpuLock::acquire_at(&self.lock_path, self.lock_wait).map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        if let (Some(pk), Some(f)) = (persist, fetched.as_ref()) {
            // All or nothing: on error the engine is reset and the prompt is prefilled.
            match self
                .engine
                .with_engine(|e| e.restore_kv(&prompt[..f.tokens], &f.payload))
            {
                Ok(Ok(())) => pk.note_restored(f),
                Ok(Err(m)) | Err(m) => {
                    pk.note_import_failed();
                    eprintln!("aien-serve-http: persistent KV: restore of {} tokens refused ({m}); prefilling", f.tokens);
                }
            }
        }
        drop(fetched);
        // The server's own sampler, so CPU and Omega engines pick tokens the same way.
        let mut observer = SamplingObserver::new(p);
        let mut sampler = |logits: &[f32], step: usize| observer.sample(logits, step);
        let o = self
            .engine
            .generate_with(prompt, &q, &mut sampler, on_token)
            .map_err(EngineError::Failed)?;
        let sampling = observer.record();
        self.reused_last
            .store(o.reused_prompt_tokens, Ordering::Relaxed);
        // Park: export under the chip lock, write after it is released.
        let parked = persist.map(|pk| {
            self.engine
                .with_engine(|e| pk.snapshot(e))
                .unwrap_or_default()
        });
        drop(lock);
        if let (Some(pk), Some(items)) = (persist, parked) {
            pk.submit(items);
        }
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
                // A constraint finished the output: complete, which HTTP clients read as "stop".
                omega::FinishReason::Constraint => FinishReason::Stop,
            },
            prefill_ms: o.prefill_ms,
            decode_ms: o.decode_ms,
            sampling: Some(sampling),
        })
    }

    fn persistence_status(&self) -> Option<serde_json::Value> {
        match &self.persist {
            PersistState::Off => None,
            PersistState::Refused(why) => {
                Some(serde_json::json!({ "state": "refused", "reason": why, "refusals": 1 }))
            }
            PersistState::On(pk) => Some(pk.status_json()),
        }
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
        let open = || {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .unwrap()
        };
        let a = open();
        // SAFETY: valid descriptors, non-blocking flock.
        assert_eq!(
            unsafe { libc::flock(a.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let b = open();
        assert_ne!(
            unsafe { libc::flock(b.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0,
            "second holder must be refused"
        );
        drop(a);
        assert_eq!(
            unsafe { libc::flock(b.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0,
            "released on drop"
        );
        drop(b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A second process (forked, so /proc/locks shows a pid other than ours) that blocks on the
    /// lock, then on getting it writes one byte to `done_fd`, holds it 100 ms and exits. Only
    /// async-signal-safe calls run in the child.
    fn fork_waiter(path: &Path, done_fd: libc::c_int) -> libc::pid_t {
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        // SAFETY: the child only calls close/alarm/open/flock/write/nanosleep/_exit.
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0, "fork");
            if pid == 0 {
                // Drop every inherited descriptor but the pipe: a copy of the parent's lock file
                // description would keep the lock held after the parent releases it, and an
                // inherited quiet-lock descriptor would block every quiet-measure on the box.
                for fd in 3..1024 {
                    if fd != done_fd {
                        libc::close(fd);
                    }
                }
                // Never outlive a stuck test.
                libc::alarm(30);
                let fd = libc::open(c_path.as_ptr(), libc::O_RDWR);
                if fd < 0 || libc::flock(fd, libc::LOCK_EX) != 0 {
                    libc::_exit(1);
                }
                libc::write(done_fd, b"x".as_ptr().cast(), 1);
                let hold = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 100_000_000,
                };
                libc::nanosleep(&hold, std::ptr::null_mut());
                libc::_exit(0);
            }
            pid
        }
    }

    fn pipe_pair() -> (libc::c_int, libc::c_int) {
        let mut fds = [0; 2];
        // SAFETY: fds has room for the two descriptors; the read end never blocks (O_NONBLOCK).
        assert_eq!(
            unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_NONBLOCK) },
            0
        );
        (fds[0], fds[1])
    }

    fn pipe_has_byte(rd: libc::c_int) -> bool {
        let mut b = [0u8; 1];
        // SAFETY: one byte into a valid buffer, non-blocking descriptor.
        unsafe { libc::read(rd, b.as_mut_ptr().cast(), 1) == 1 }
    }

    fn private_lock_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gpu-lock-yield-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("lock")
    }

    /// L188: the server releases the lock between requests and the next request re-takes it
    /// at once. Linux flock has no queue order, so a `flock -x` job that was already waiting
    /// never got a turn (spark-window waited 90 min, chip gates 18 h behind aien-translate).
    /// With a waiter queued, the next acquire must let that waiter take the lock first.
    #[test]
    fn gpu_lock_gives_a_queued_waiter_the_next_turn() {
        let path = private_lock_path("turn");
        let held = GpuLock::acquire_at(&path, Duration::from_secs(1)).unwrap();
        let (rd, wr) = pipe_pair();
        let child = fork_waiter(&path, wr);
        let t0 = Instant::now();
        while queued_rows_in_proc(&path) == 0 {
            assert!(t0.elapsed() < Duration::from_secs(5), "waiter never queued");
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(held);
        // The server's next request, straight after the release.
        let again = GpuLock::acquire_at(&path, Duration::from_secs(10)).expect("reacquire");
        assert!(
            pipe_has_byte(rd),
            "the waiting process must have had the lock before the server took it again"
        );
        drop(again);
        let mut status = 0;
        // SAFETY: reaping our own child.
        unsafe { libc::waitpid(child, &mut status, 0) };
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn lock_row_of_other_tells_blocked_from_holder_rows_of_this_lock() {
        let table = "\
1: FLOCK  ADVISORY  WRITE 100 08:01:777 0 EOF
1: -> FLOCK  ADVISORY  WRITE 200 08:01:777 0 EOF
2: FLOCK  ADVISORY  WRITE 300 08:01:999 0 EOF
";
        let dev = libc::makedev(8, 1);
        assert!(lock_row_of_other(table, dev, 777, 100, true));
        assert!(!lock_row_of_other(table, dev, 777, 200, true), "own pid");
        assert!(lock_row_of_other(table, dev, 777, 200, false));
        assert!(
            !lock_row_of_other(table, dev, 999, 100, true),
            "no waiter there"
        );
        assert!(
            !lock_row_of_other(table, dev, 555, 100, true),
            "other inode"
        );
        assert!(
            !lock_row_of_other(table, libc::makedev(8, 2), 777, 100, true),
            "other device"
        );
    }

    /// With nobody waiting, back-to-back requests are not delayed at all.
    #[test]
    fn gpu_lock_does_not_delay_when_nobody_waits() {
        let path = private_lock_path("idle");
        let t0 = Instant::now();
        for _ in 0..20 {
            let lock = GpuLock::acquire_at(&path, Duration::from_secs(1)).unwrap();
            drop(lock);
        }
        assert!(
            t0.elapsed() < Duration::from_millis(500),
            "20 uncontended acquires took {:?}",
            t0.elapsed()
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// Independent of the production parser: how many `->` (blocked) rows /proc/locks shows for
    /// this lock file's inode.
    fn queued_rows_in_proc(path: &Path) -> usize {
        use std::os::unix::fs::MetadataExt;
        let ino = std::fs::metadata(path).unwrap().ino();
        let tail = format!(":{ino} ");
        std::fs::read_to_string("/proc/locks")
            .unwrap_or_default()
            .lines()
            .filter(|l| l.contains(" -> ") && l.contains(&tail))
            .count()
    }

    #[test]
    fn mem_available_reads_proc() {
        assert!(mem_available_gb() > 0.0);
    }

    #[test]
    fn prefill_mode_defaults_on_and_env_switches_off() {
        assert_eq!(
            PrefillMode::default(),
            PrefillMode {
                whole_prompt: true,
                device_host: true
            }
        );
        assert_eq!(
            PrefillMode::from_env(None, None).unwrap(),
            PrefillMode::default()
        );
        let m = PrefillMode::from_env(Some("0"), None).unwrap();
        assert!(!m.whole_prompt && m.device_host);
        // The device router/combine lives inside the whole-prompt path: off with it.
        assert_eq!(
            m.effective(),
            PrefillMode {
                whole_prompt: false,
                device_host: false
            }
        );
        let m = PrefillMode::from_env(Some("on"), Some("off")).unwrap();
        assert_eq!(
            m.effective(),
            PrefillMode {
                whole_prompt: true,
                device_host: false
            }
        );
        assert!(PrefillMode::from_env(Some("of"), None).is_err());
        assert!(parse_switch("--whole-prompt", "maybe").is_err());
        assert!(parse_switch("--whole-prompt", " TRUE ").unwrap());
    }
}
