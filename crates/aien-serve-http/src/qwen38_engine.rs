//! Qwen3.8-27B behind the door (sc #642 serving): the whole model resident on AIEN's own Omega
//! kernels (64 layers, fused FP8 projections, bf16 lm_head), prompts through omega's prefill,
//! replies through the chip-proven decode graph, next turn resumed from the live state or the
//! snapshot taken before the prompt's last turn header (`aien_inference_abi::qwen38_omega::
//! Qwen38Session`).
//!
//! Chip etiquette as the 30B door ([`crate::omega_engine`]): `/tmp/aien-gb10.lock` is taken for
//! the upload and around each generation, never while idle. Persistent KV parks
//! (`AIEN_KV_STORE_DIR`) are not offered: they hold attention KV only, not the DeltaNet state.
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Session, Reuse};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::sampling::SamplingObserver;

/// Model id the door answers to unless `--model-id` says otherwise.
pub const DEFAULT_MODEL_ID: &str = "qwen3.8-27b-fp8";
/// The same for the sparse sibling, Qwen3.6-35B-A3B (config model_type qwen3_5_moe, sc L306).
pub const DEFAULT_MODEL_ID_MOE: &str = "qwen3.6-35b-a3b-fp8";
/// Prefill chunk length (the 2026-10-18 TTFT receipts ran 4096).
pub const PREFILL_CHUNK: usize = 4096;
/// How long a request waits in the chip lock's queue before answering busy (the hive's chip
/// jobs queue with `flock(1)` and run for seconds to minutes each).
pub const REQUEST_LOCK_WAIT: Duration = Duration::from_secs(300);
/// How long the upload waits for the chip lock (other jobs' chip runs take minutes; a request
/// waits [`REQUEST_LOCK_WAIT`]).
pub const LOAD_LOCK_WAIT: Duration = Duration::from_secs(1800);
/// Memory kept free beyond [`need_bytes`] at load (other jobs share the box).
pub const KEEP_FREE: u64 = 8 << 30;

fn model_type(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["model_type"].as_str().map(str::to_owned))
}

/// `config.json` says `model_type: qwen3_5` (Qwen3.8-27B) or `qwen3_5_moe` (Qwen3.6-35B-A3B):
/// the family this door serves.
pub fn is_qwen38_dir(dir: &Path) -> bool {
    matches!(model_type(dir).as_deref(), Some("qwen3_5" | "qwen3_5_moe"))
}

/// The sparse member (`qwen3_5_moe`).
pub fn is_moe_dir(dir: &Path) -> bool {
    model_type(dir).as_deref() == Some("qwen3_5_moe")
}

/// The model id a checkpoint answers to by default.
pub fn default_model_id(dir: &Path) -> &'static str {
    if is_moe_dir(dir) {
        DEFAULT_MODEL_ID_MOE
    } else {
        DEFAULT_MODEL_ID
    }
}

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// This process's resident set in bytes (0 when unreadable).
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

pub struct Qwen38Door {
    session: Mutex<Qwen38Session>,
    max_context: usize,
    lock_path: PathBuf,
    lock_wait: Duration,
}

impl Qwen38Door {
    /// Uploads the checkpoint at `dir` with a KV cache for `max_context` positions (rounded up
    /// to whole 256-position passes), under the GPU lock.
    pub fn load(dir: &Path, max_context: usize) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        // The 35B runs prompts through its decode graph (no prefill entries needed).
        if !is_moe_dir(dir) && !aien_omega_q38::has_prefill() {
            return Err("this binary has no qwen38 prefill entries linked (build against omega qwen38/prefill-graph)".into());
        }
        let c = Qwen38Config::from_checkpoint(dir)?;
        let max_pos = max_context.next_multiple_of(256);
        let need = need_bytes(&c, max_pos, PREFILL_CHUNK);
        let avail = mem_available();
        if avail < need + KEEP_FREE {
            return Err(format!(
                "needs {} MiB plus {} MiB kept free, MemAvailable {} MiB (another model may be loaded); refusing",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let _lock = GpuLock::acquire_queued_at(Path::new(GPU_LOCK_PATH), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen38: uploading {} layers, KV {max_pos} positions (need {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        if is_moe_dir(dir) && !aien_omega_q38::q36::is_native() {
            return Err("this binary has no q36 decode graph linked (build against omega q36/serve)".into());
        }
        let model = Qwen38Omega::open(dir, None, max_pos)?;
        eprintln!(
            "aien-serve-http: qwen38 engine ready in {:.1} s: context {max_context} tokens, weights {:.2} GB resident, KV {:.2} GB, rss {:.2} GB, prefill chunk {PREFILL_CHUNK}, spin_us {:?}",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            model.kv_bytes() as f64 / 1e9,
            rss_bytes() as f64 / 1e9,
            aien_omega_gpu::spin_us(),
        );
        Ok(Self {
            session: Mutex::new(Qwen38Session::new(model, PREFILL_CHUNK)),
            max_context,
            lock_path: PathBuf::from(GPU_LOCK_PATH),
            lock_wait: REQUEST_LOCK_WAIT,
        })
    }
}

impl Engine for Qwen38Door {
    fn kind(&self) -> &str {
        "omega-qwen38"
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        if prompt.is_empty() {
            return Err(EngineError::Failed("empty prompt".into()));
        }
        if prompt.len() >= self.max_context {
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max: self.max_context,
            });
        }
        let t_lock = Instant::now();
        let lock = GpuLock::acquire_queued_at(&self.lock_path, self.lock_wait)
            .map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        let mut s = self
            .session
            .lock()
            .map_err(|_| EngineError::Failed("qwen38 session poisoned".into()))?;
        let t0 = Instant::now();
        let (mut logits, fs) = s.start(prompt).map_err(EngineError::Failed)?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut observer = SamplingObserver::new(p);
        let (mut emitted, mut chip_ns) = (0usize, 0u64);
        let finish = loop {
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let tok = observer.sample(&logits, emitted);
            if p.stop_token_ids.contains(&tok) {
                break FinishReason::Stop;
            }
            emitted += 1;
            if !on_token(tok) {
                break FinishReason::Cancelled;
            }
            // The last allowed token is not fed: nothing would read its logits.
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let (next, ns) = s.advance(tok).map_err(EngineError::Failed)?;
            logits = next;
            chip_ns += ns;
        };
        let decode_ms = t1.elapsed().as_millis() as u64;
        let steps = emitted.saturating_sub(1).max(1) as f64;
        let reused = fs.reuse.map_or(0, Reuse::tokens);
        drop(s);
        drop(lock);
        let pf = &fs.prefill;
        eprintln!(
            "aien-serve-http: qwen38: lock wait {lock_wait_ms} ms, prompt {} tok, reuse {:?} ({reused} tok), fed {} ({} stepped), prefill {prefill_ms} ms (gemm chip {} ms, gdn chip {} ms, attn chip {} ms, host {} ms, restore {} ms, snapshot {} ms), decode {emitted} tok in {decode_ms} ms, graph chip {:.1} ms/token, wall {:.1} ms/token",
            prompt.len(),
            fs.reuse.unwrap_or(Reuse::Cold),
            fs.fed,
            fs.stepped,
            pf.gemm_chip_ns / 1_000_000,
            pf.gdn_chip_ns / 1_000_000,
            pf.attn_chip_ns / 1_000_000,
            pf.host_ns / 1_000_000,
            fs.restore_ns / 1_000_000,
            fs.snapshot_ns / 1_000_000,
            chip_ns as f64 / 1e6 / steps,
            decode_ms as f64 / steps,
        );
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms,
            sampling: Some(observer.record()),
        })
    }
}
