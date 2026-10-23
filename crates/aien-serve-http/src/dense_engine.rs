//! aien-rig: dense Qwen3 (Qwen3-8B FP8 or MXFP4, Qwen3-4B bf16) behind the door on AIEN's own
//! Omega kernels (`aien_inference_abi::qwen3_dense_omega::engine::Qwen3DenseOmegaEngine`), with an
//! optional memory budget (`--mem-budget-gib 12` for the RTX 5070): weights + KV + scratch are
//! planned before the upload ([`aien_inference_abi::qwen3_dense_budget`]), the context is shrunk
//! or the load refused when they would not fit, and after the upload the bytes omega actually
//! asked the driver for are checked against the same budget.
//!
//! Chip etiquette as the other doors: `/tmp/aien-gb10.lock` is taken (queued, blocking) for the
//! upload and around each generation, never while idle.

use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen3_dense_budget::{
    self as budget, BudgetPlan, BudgetPolicy, ProjFormat,
};
use aien_inference_abi::qwen3_dense_omega::engine::{
    OmegaDenseUploader, Qwen3DenseOmegaEngine, W4DenseUploader,
};
use aien_inference_abi::qwen3_dense_omega::{DenseTensor, DenseUploader, Qwen3DenseConfig};
use aien_inference_abi::qwen3_moe_omega as omega;

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::sampling::SamplingObserver;

/// Model id the door answers to unless `--model-id` says otherwise.
pub const DEFAULT_MODEL_ID: &str = "qwen3-dense";
/// How long a request waits for the chip lock before answering busy.
pub const REQUEST_LOCK_WAIT: Duration = Duration::from_secs(300);
/// Opt-in: append each request's prompt and completion token ids (and timings) as one JSON line
/// to this file, for the correctness gate (`tools/rig-http-gate.sh`). Unset: nothing is written.
pub const IDS_LOG_ENV: &str = "AIEN_RIG_IDS_LOG";

/// `config.json` says `model_type: qwen3` (dense Qwen3: 0.6B to 32B, Qwen3-4B-Instruct-2507).
pub fn is_dense_qwen3_dir(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|v| v["model_type"].as_str() == Some("qwen3"))
}

/// How to load: projection format, context asked, optional budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseOptions {
    pub format: ProjFormat,
    pub max_context: usize,
    /// `None` = no budget (the context is what was asked).
    pub budget_bytes: Option<u64>,
    pub policy: BudgetPolicy,
}

/// What the load did, for the startup lines and `/health`.
#[derive(Debug, Clone)]
pub struct DenseLoadFacts {
    pub plan: Option<BudgetPlan>,
    pub context: usize,
    pub load_seconds: f64,
    pub resident_bytes: u64,
    /// Bytes omega asked the driver for during the load (page-rounded, never freed by the load):
    /// an upper bound of what this process holds on the chip after it.
    pub device_alloc_bytes: Option<u64>,
    pub w4_rel_rms: Option<f64>,
}

pub struct DenseDoor {
    engine: Mutex<Qwen3DenseOmegaEngine>,
    facts: DenseLoadFacts,
    lock_path: PathBuf,
    lock_wait: Duration,
}

/// Blocks in the kernel's queue for the chip lock (jobs that wait with `flock(1)` queue the same
/// way), for the one-time upload.
fn lock_queued(path: &Path) -> Result<std::fs::File, String> {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: flock on a valid open descriptor.
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    if rc != 0 {
        return Err(format!(
            "flock {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(f)
}

/// Plans the budget without touching the chip (also used by `--plan-only`).
pub fn plan_for(
    dir: &Path,
    opts: &DenseOptions,
) -> Result<(Qwen3DenseConfig, Option<BudgetPlan>), String> {
    let c = Qwen3DenseConfig::from_checkpoint(dir)?;
    if opts.format == ProjFormat::W4 && !c.is_fp8(DenseTensor::Qkv) {
        return Err(
            "--weights w4 converts Qwen's FP8 checkpoint to MXFP4; this checkpoint is bf16".into(),
        );
    }
    let plan = match opts.budget_bytes {
        Some(b) => Some(budget::plan(
            &c,
            opts.format,
            b,
            opts.max_context,
            opts.policy,
        )?),
        None => None,
    };
    Ok((c, plan))
}

impl DenseDoor {
    pub fn load(dir: &Path, opts: DenseOptions) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        let (_c, plan) = plan_for(dir, &opts)?;
        let context = plan.as_ref().map_or(opts.max_context, |p| p.context);
        if let Some(p) = &plan {
            eprintln!("aien-serve-http: {}", p.line());
        }
        let before = aien_omega_gpu::alloc_stats().map(|s| s.alloc_bytes);
        let lock = lock_queued(Path::new(GPU_LOCK_PATH))?;
        let t0 = Instant::now();
        let base = OmegaDenseUploader;
        let w4 = W4DenseUploader::chip(&base);
        let up: &dyn DenseUploader = match opts.format {
            ProjFormat::W4 => &w4,
            ProjFormat::Checkpoint => &base,
        };
        let (engine, report) = Qwen3DenseOmegaEngine::load_omega(dir, up, context, &mut |_| {})?;
        drop(lock);
        let after = aien_omega_gpu::alloc_stats().map(|s| s.alloc_bytes);
        let device_alloc_bytes = match (before, after) {
            (Some(b), Some(a)) => Some(a.saturating_sub(b)),
            _ => None,
        };
        let w4_rel_rms = (opts.format == ProjFormat::W4).then(|| w4.stats().stats.rel_rms());
        let facts = DenseLoadFacts {
            plan: plan.clone(),
            context: engine.max_context_tokens(),
            load_seconds: t0.elapsed().as_secs_f64(),
            resident_bytes: report.resident_bytes as u64,
            device_alloc_bytes,
            w4_rel_rms,
        };
        eprintln!(
            "aien-serve-http: dense load: {:?} context {} in {:.1} s, weights resident {:.3} GiB, omega device allocations {}, attention {}, kv resident {}{}",
            opts.format,
            facts.context,
            facts.load_seconds,
            facts.resident_bytes as f64 / budget::GIB as f64,
            device_alloc_bytes.map_or("unknown".to_string(), |b| format!("{:.3} GiB", b as f64 / budget::GIB as f64)),
            engine.attention_path(),
            engine.kv_resident(),
            w4_rel_rms.map_or(String::new(), |r| format!(", MXFP4 weight rel RMS error {:.2}%", 100.0 * r)),
        );
        if let (Some(p), Some(b)) = (&plan, device_alloc_bytes) {
            if b > p.budget {
                return Err(format!(
                    "measured device allocations {:.3} GiB exceed the {:.3} GiB budget (plan said {:.3})",
                    b as f64 / budget::GIB as f64,
                    p.budget as f64 / budget::GIB as f64,
                    p.total() as f64 / budget::GIB as f64
                ));
            }
        }
        Ok(Self {
            engine: Mutex::new(engine),
            facts,
            lock_path: PathBuf::from(GPU_LOCK_PATH),
            lock_wait: REQUEST_LOCK_WAIT,
        })
    }

    pub fn facts(&self) -> &DenseLoadFacts {
        &self.facts
    }
}

impl Engine for DenseDoor {
    fn kind(&self) -> &str {
        "omega-dense"
    }

    fn max_context_tokens(&self) -> usize {
        self.facts.context
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        let max = self.facts.context;
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
        let t_lock = Instant::now();
        let lock =
            GpuLock::acquire_at(&self.lock_path, self.lock_wait).map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        let mut observer = SamplingObserver::new(p);
        let mut sampler = |logits: &[f32], step: usize| observer.sample(logits, step);
        let mut e = self
            .engine
            .lock()
            .map_err(|_| EngineError::Failed("dense engine lock poisoned".into()))?;
        let mut ids: Vec<u32> = Vec::new();
        let record = std::env::var_os(IDS_LOG_ENV);
        let mut tap = |t: u32| {
            if record.is_some() {
                ids.push(t);
            }
            on_token(t)
        };
        let o = e
            .generate_with(prompt, &q, &mut sampler, &mut tap)
            .map_err(EngineError::Failed)?;
        drop(e);
        drop(lock);
        let sampling = observer.record();
        let chip_ms_per_token = if o.emitted_tokens > 0 {
            o.decode_timing.chip_ns as f64 / 1e6 / o.emitted_tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "aien-serve-http: omega-dense: lock wait {lock_wait_ms} ms, prompt {} tok, {} reused from the previous turn, prefill {} ms, decode {} tok in {} ms (chip {chip_ms_per_token:.2} ms/token, {} launches)",
            o.prompt_tokens, o.reused_prompt_tokens, o.prefill_ms, o.emitted_tokens, o.decode_ms, o.decode_timing.launches
        );
        if let Some(path) = record {
            // Correctness gates read the exact ids (aien-rig): one JSON line per request.
            let line = serde_json::json!({
                "prompt_ids": prompt, "completion_ids": ids, "temperature": p.temperature,
                "seed": p.seed, "prompt_tokens": o.prompt_tokens, "reused_prompt_tokens": o.reused_prompt_tokens,
                "prefill_ms": o.prefill_ms, "decode_ms": o.decode_ms, "emitted_tokens": o.emitted_tokens,
                "decode_chip_ns": o.decode_timing.chip_ns, "decode_launches": o.decode_timing.launches,
                "lock_wait_ms": lock_wait_ms as u64,
            });
            use std::io::Write;
            let w = OpenOptions::new().create(true).append(true).open(&path);
            if let Err(e) = w.and_then(|mut f| writeln!(f, "{line}")) {
                eprintln!("aien-serve-http: {IDS_LOG_ENV}: {e}");
            }
        }
        Ok(GenerateOutcome {
            emitted_tokens: o.emitted_tokens,
            finish: match o.finish {
                omega::FinishReason::Stop => FinishReason::Stop,
                omega::FinishReason::Length => FinishReason::Length,
                omega::FinishReason::Cancelled => FinishReason::Cancelled,
                omega::FinishReason::Constraint => FinishReason::Stop,
            },
            prefill_ms: o.prefill_ms,
            decode_ms: o.decode_ms,
            sampling: Some(sampling),
        })
    }

    fn memory_status(&self) -> Option<serde_json::Value> {
        // `/health` carries the memory plan (aien-rig), so a client can see the context it got.
        let f = &self.facts;
        Some(serde_json::json!({
            "engine": "omega-dense",
            "context": f.context,
            "budget_bytes": f.plan.as_ref().map(|p| p.budget),
            "planned_bytes": f.plan.as_ref().map(|p| p.total()),
            "weights_bytes": f.plan.as_ref().map(|p| p.weights),
            "kv_bytes": f.plan.as_ref().map(|p| p.kv),
            "scratch_bytes": f.plan.as_ref().map(|p| p.scratch),
            "context_shrunk": f.plan.as_ref().map(|p| p.shrunk()),
            "device_alloc_bytes": f.device_alloc_bytes,
            "w4_weight_rel_rms": f.w4_rel_rms,
        }))
    }
}
