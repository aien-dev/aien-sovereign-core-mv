//! Start-up reservation of the GB10 serving buffers and kernels (sovereign-core#277, a cut
//! toward it).
//!
//! Without it, omega grows the attention staging pool, the matmul activation/result buffers
//! and the matmul kernel cache on demand while serving, and every growth is a driver (RM)
//! allocation that can fail with NV_ERR_NO_MEMORY when MemFree is low (omega#327; omega#332
//! measured 458 allocations and 456 frees over 428 tokens). omega#333 and omega#338 add an
//! reservation: allocate the buffers once for declared bounds, build and pin every
//! matmul kernel the model will run (`prepare`), then `seal`, after which serving makes no
//! driver allocation inside the bounds and an unprepared kernel or a call past the bounds is
//! refused `TOO_LARGE` by name.
//!
//! This module derives the bounds and the shapes from the model and the daemon's declared
//! limits, and runs reserve, prepare, seal in that order through [`ServingOps`] (the real one
//! calls omega; tests record). It applies only to Qwen3 on the GB10, which is enabled by
//! default since #277 (`AIEN_GB10_QWEN3_DECLARED_ATTEMPT=0` refuses it). It
//! does not claim to fix #277: it is not proven on the chip, and omega's header lists what is
//! still not covered (elementwise scratch, the attention kernel).

use crate::omega_backend::omega_model_refusal_with;
use aien_abi_core::ModelConfig;
use aien_omega_gpu::{ServingBounds, ServingBytes};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// omega's contiguous f32 attention kernel refuses `seq_len` past this itself
/// (`OMEGA_GPU_ATTN_MAX_GQA_CTX`, omega `src/omega_gpu_attention_api.h:83`), so a larger
/// declared context cannot be served natively and is not reserved.
pub const GB10_ATTENTION_MAX_CONTEXT: u32 = 4096;

/// Matmul kernel cache slots to reserve: omega's maximum (`CACHE_SLOTS_MAX`, omega
/// `src/omega_gpu_matmul_api.c:30`, 128 at 80c4daa). omega picks `grid_x` from the row count, so
/// one weight shape needs several kernels; omega's own sweep of Qwen3-4B (6 shapes x rows 1..256,
/// CTA budget 256) observed 95 distinct kernels (omega#338). Every one is prepared at start-up,
/// so the slots must cover them; `prepare` refuses with TOO_LARGE if they do not.
pub const GB10_MATMUL_KERNEL_SLOTS: u32 = 128;

/// What the daemon declares at start-up (aien-cli `run_daemon_server`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingLimits {
    /// Context the KV pool is planned for (`KvPoolPlan::context_tokens`).
    pub context_tokens: usize,
    /// Most sequences one decode step batches (`SchedulerConfig::max_batch_size`); the batched
    /// decode path runs every projection with this many rows.
    pub max_batch_rows: usize,
    /// Prompt tokens per prefill chunk (`SchedulerConfig::prefill_chunk_size`).
    pub prefill_chunk_rows: usize,
}

/// The three omega calls the start-up makes. The real one is [`OmegaServingOps`].
pub trait ServingOps {
    fn reserve(&self, bounds: &ServingBounds) -> Result<(), String>;
    /// Build and pin the kernel a call of `m` rows over a `k` x `n` weight will use.
    fn prepare(&self, m: u32, k: u32, n: u32) -> Result<(), String>;
    /// From now on an unprepared kernel is refused, never built.
    fn seal(&self);
}

/// A reservation that was made: the bounds, the driver bytes omega was asked for, and what the
/// kernel preparation cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingReservation {
    pub bounds: ServingBounds,
    pub bytes: ServingBytes,
    /// Number of `prepare` calls made (rows x distinct weight shapes).
    pub prepare_calls: u32,
    /// Wall time of the whole reserve, prepare and seal.
    pub elapsed: Duration,
}

impl ServingReservation {
    /// The start-up log line a chip run reads.
    pub fn log_line(&self) -> String {
        let (b, y) = (&self.bounds, &self.bytes);
        format!(
            "GB10_SERVING_RESERVATION reserved bytes={} (attention pool {} + q/out {}x2 + table {}; matmul activation {} + result {}) \
             bounds: context<={} seqs<={} rows<={} k<={} n<={} logits_n<={} kernel_slots={}; \
             prepared {} matmul kernel calls in {} ms, then sealed; \
             calls past the bounds or unprepared kernels are refused TOO_LARGE; not covered: elementwise scratch, attention kernel",
            y.total(),
            y.attention_pool,
            y.attention_q_out,
            y.attention_table,
            y.matmul_activation,
            y.matmul_result,
            b.max_context,
            b.max_seqs,
            b.max_rows,
            b.max_k,
            b.max_n,
            b.max_n_one_row,
            b.kernel_slots,
            self.prepare_calls,
            self.elapsed.as_millis(),
        )
    }
}

fn dim(what: &str, v: usize) -> Result<u32, String> {
    match u32::try_from(v) {
        Ok(x) if x > 0 => Ok(x),
        _ => Err(format!("{what} = {v} is not a usable serving bound")),
    }
}

/// The distinct `(k, n)` weight shapes the forward pass multiplies, from the model config:
/// q (hidden, q_dim), k and v (hidden, kv_dim), o (q_dim, hidden), gate and up (hidden,
/// intermediate), down (intermediate, hidden), and the logits projection (hidden, vocab).
pub fn serving_matmul_shapes(config: &ModelConfig) -> Result<Vec<(u32, u32)>, String> {
    let h = dim("hidden_dim", config.hidden_dim)?;
    let q = dim("q width", config.num_heads * config.head_dim)?;
    let kv = dim("kv width", config.num_kv_heads * config.head_dim)?;
    let i = dim("intermediate_dim", config.intermediate_dim)?;
    let v = dim("vocab_size", config.vocab_size)?;
    let set: BTreeSet<(u32, u32)> = [(h, q), (h, kv), (q, h), (h, i), (i, h), (h, v)]
        .into_iter()
        .collect();
    Ok(set.into_iter().collect())
}

/// Bounds for `config` under `limits`. Errors name the offending number.
pub fn gb10_serving_bounds(
    config: &ModelConfig,
    limits: &ServingLimits,
) -> Result<ServingBounds, String> {
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    // Widest input: o_proj reads q_dim, q/k/v/gate/up read hidden, down_proj reads intermediate.
    let max_k = config.hidden_dim.max(q_dim).max(config.intermediate_dim);
    // Widest output for a call of up to max_rows rows. The batched decode path projects the
    // logits with every decode row (transformer_backend.rs, step 7: matmul_batch M = D), so the
    // vocabulary is in the many-row bound as well as the one-row bound.
    let max_n = q_dim
        .max(kv_dim)
        .max(config.hidden_dim)
        .max(config.intermediate_dim)
        .max(config.vocab_size);
    let context = limits
        .context_tokens
        .min(GB10_ATTENTION_MAX_CONTEXT as usize);
    let max_rows = dim(
        "max rows (max_batch_size, prefill chunk)",
        limits.max_batch_rows.max(limits.prefill_chunk_rows),
    )?;
    Ok(ServingBounds {
        max_context: dim("declared context", context)?,
        // omega_backend.rs paged_attention_batch runs one chip_paged_one call per sequence.
        max_seqs: 1,
        num_q_heads: dim("num_heads", config.num_heads)?,
        num_kv_heads: dim("num_kv_heads", config.num_kv_heads)?,
        head_dim: dim("head_dim", config.head_dim)?,
        kv_block_size: dim("block_size", config.block_size)?,
        max_rows,
        max_k: dim("widest input dimension", max_k)?,
        max_n: dim("widest output dimension", max_n)?,
        max_n_one_row: dim("vocab_size", config.vocab_size)?,
        kernel_slots: GB10_MATMUL_KERNEL_SLOTS,
    })
}

/// The start-up sequence, with the GPU, the opt-in and the omega calls passed in.
///
/// `Ok(None)`: it does not apply (not a GB10 build, not Qwen3, or the Qwen3 opt-in is off, in
/// which case the default refusal in `check_model` stops the model before any serving).
/// `Ok(Some(_))`: reserved, every (rows, shape) kernel prepared, sealed. `Err(_)`: omega refused
/// the reservation or a prepare (or a bound could not be formed); the caller must refuse to
/// serve. There is no on-demand fallback. The reservation is held for the life of the process
/// (omega ends it when it closes the device); this never releases it.
pub fn reserve_gb10_serving_with(
    config: &ModelConfig,
    limits: &ServingLimits,
    gpu_native: bool,
    qwen3_opted_in: bool,
    ops: &dyn ServingOps,
) -> Result<Option<ServingReservation>, String> {
    if !gpu_native || !config.qk_norm || omega_model_refusal_with(config, qwen3_opted_in).is_some()
    {
        return Ok(None);
    }
    let refuse = |why: String| {
        format!(
            "GB10_SERVING_RESERVATION refused for {:?}: {why}; refusing to serve (sovereign-core#277: \
             no on-demand fallback, because on-demand growth is the driver allocation that fails when MemFree is low)",
            config.model_id
        )
    };
    let started = Instant::now();
    let bounds = gb10_serving_bounds(config, limits).map_err(&refuse)?;
    let shapes = serving_matmul_shapes(config).map_err(&refuse)?;
    ops.reserve(&bounds).map_err(&refuse)?;
    let mut prepare_calls = 0u32;
    for m in 1..=bounds.max_rows {
        for &(k, n) in &shapes {
            ops.prepare(m, k, n)
                .map_err(|e| refuse(format!("matmul kernel prepare m={m} k={k} n={n}: {e}")))?;
            prepare_calls += 1;
        }
    }
    ops.seal();
    Ok(Some(ServingReservation {
        bounds,
        bytes: bounds.bytes(),
        prepare_calls,
        elapsed: started.elapsed(),
    }))
}

/// The real calls: omega's `omega_gpu_reserve_serving`, `omega_gpu_matmul_prepare` and
/// `omega_gpu_serving_seal`, with omega's stage text added to a refusal.
pub struct OmegaServingOps;

fn with_stage(e: aien_omega_gpu::OmegaGpuError) -> String {
    let stage = aien_omega_gpu::last_error();
    if stage.is_empty() {
        e.to_string()
    } else {
        format!("{e} (omega: {stage})")
    }
}

impl ServingOps for OmegaServingOps {
    fn reserve(&self, b: &ServingBounds) -> Result<(), String> {
        aien_omega_gpu::reserve_serving(b).map_err(with_stage)
    }
    fn prepare(&self, m: u32, k: u32, n: u32) -> Result<(), String> {
        aien_omega_gpu::matmul_prepare(m, k, n).map_err(with_stage)
    }
    fn seal(&self) {
        aien_omega_gpu::seal_serving();
    }
}

/// Reads the serving log lines the daemon prints next to `GB10_SERVING_RESERVATION`: the
/// allocation counters once after warm-up, and the change since warm-up after every request.
/// A nonzero change is printed loudly, never hidden.
pub trait ServingProbe: Send + Sync {
    /// Called once, after the declared warm-up and before the first request is accepted.
    fn after_warm_up(&self) -> Option<String>;
    /// Called after each served connection.
    fn after_request(&self) -> Option<String>;
}

/// [`ServingProbe`] over omega's allocation counters (`omega_gpu_session_alloc_stats`).
/// Counters are cumulative for the process, so overlapping requests cannot confuse the delta.
pub struct AllocProbe {
    read: Box<dyn Fn() -> Option<aien_omega_gpu::AllocStats> + Send + Sync>,
    baseline: std::sync::Mutex<Option<aien_omega_gpu::AllocStats>>,
}

impl AllocProbe {
    pub fn new(read: Box<dyn Fn() -> Option<aien_omega_gpu::AllocStats> + Send + Sync>) -> Self {
        Self {
            read,
            baseline: std::sync::Mutex::new(None),
        }
    }

    /// The probe on omega's real counters.
    pub fn omega() -> Self {
        Self::new(Box::new(aien_omega_gpu::alloc_stats))
    }
}

impl ServingProbe for AllocProbe {
    fn after_warm_up(&self) -> Option<String> {
        let s = (self.read)()?;
        *self.baseline.lock().ok()? = Some(s);
        Some(format!(
            "GB10_SERVING_ALLOC after_warmup allocs={} bytes={} failures={} frees={}",
            s.allocs, s.alloc_bytes, s.alloc_failures, s.frees
        ))
    }

    fn after_request(&self) -> Option<String> {
        let now = (self.read)()?;
        let base = (*self.baseline.lock().ok()?)?;
        let d = |a: u64, b: u64| a.saturating_sub(b);
        let (allocs, bytes, failures, frees) = (
            d(now.allocs, base.allocs),
            d(now.alloc_bytes, base.alloc_bytes),
            d(now.alloc_failures, base.alloc_failures),
            d(now.frees, base.frees),
        );
        // Frees alone are not an allocation, but a free after warm-up means a buffer was replaced
        // (or the daemon is tearing down), so it is flagged too, under its own name.
        let flag = if allocs != 0 || failures != 0 {
            " NONZERO: serving asked the driver for memory after warm-up"
        } else if frees != 0 {
            " FREES: driver memory released after warm-up (teardown, or a buffer was replaced)"
        } else {
            ""
        };
        Some(format!(
            "GB10_SERVING_ALLOC request allocs_since_warmup={allocs} bytes_since_warmup={bytes} failures_since_warmup={failures} frees_since_warmup={frees}{flag}"
        ))
    }
}
