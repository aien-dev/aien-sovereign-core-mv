//! Start-up reservation of the GB10 serving buffers (sovereign-core#277, a cut toward it).
//!
//! Without it, omega grows the attention staging pool, the matmul activation/result buffers
//! and the matmul kernel cache on demand while serving, and every growth is a driver (RM)
//! allocation that can fail with NV_ERR_NO_MEMORY when MemFree is low (omega#327; omega#332
//! measured 458 allocations and 456 frees over 428 tokens). omega#333 (b980783) adds an
//! opt-in reservation: allocate once for declared bounds, then serve with zero driver
//! allocations inside them and a named refusal (`TOO_LARGE`) past them.
//!
//! This module derives the bounds from the model and the daemon's declared limits and makes
//! the one call. It applies only on the path that is already opt-in (Qwen3 on the GB10,
//! `AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1`); the default Qwen3 refusal is untouched. It does not
//! claim to fix #277: it is not proven on the chip, and it does not reserve what omega's
//! header lists as not covered (elementwise scratch, the attention kernel cache, the first
//! build of each matmul shape).

use crate::omega_backend::{gb10_qwen3_opted_in, omega_model_refusal_with, OmegaGb10Backend};
use aien_abi_core::ModelConfig;
use aien_omega_gpu::{ServingBounds, ServingBytes};

/// omega's contiguous f32 attention kernel refuses `seq_len` past this itself
/// (`OMEGA_GPU_ATTN_MAX_GQA_CTX`, omega `src/omega_gpu_attention_api.h:83`), so a larger
/// declared context cannot be served natively and is not reserved.
pub const GB10_ATTENTION_MAX_CONTEXT: u32 = 4096;

/// Matmul kernel cache depth: 21 distinct (kp, np, grid_x) shapes for Qwen3-4B at the daemon's
/// CTA budget (docs/design/gb10-weight-ownership.md), omega allows at most 32.
pub const GB10_MATMUL_KERNEL_SLOTS: u32 = 32;

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

/// A reservation that was made: the bounds and the driver bytes omega was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingReservation {
    pub bounds: ServingBounds,
    pub bytes: ServingBytes,
}

impl ServingReservation {
    /// The start-up log line a chip run reads.
    pub fn log_line(&self) -> String {
        let (b, y) = (&self.bounds, &self.bytes);
        format!(
            "GB10_SERVING_RESERVATION reserved bytes={} (attention pool {} + q/out {}x2 + table {}; matmul activation {} + result {}) \
             bounds: context<={} seqs<={} rows<={} k<={} n<={} logits_n<={} kernel_slots={}; \
             calls past the bounds are refused TOO_LARGE; not covered: elementwise scratch, attention kernel, first build of each matmul shape",
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
        )
    }
}

fn dim(what: &str, v: usize) -> Result<u32, String> {
    match u32::try_from(v) {
        Ok(x) if x > 0 => Ok(x),
        _ => Err(format!("{what} = {v} is not a usable serving bound")),
    }
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
    Ok(ServingBounds {
        max_context: dim("declared context", context)?,
        // omega_backend.rs paged_attention_batch runs one chip_paged_one call per sequence.
        max_seqs: 1,
        num_q_heads: dim("num_heads", config.num_heads)?,
        num_kv_heads: dim("num_kv_heads", config.num_kv_heads)?,
        head_dim: dim("head_dim", config.head_dim)?,
        kv_block_size: dim("block_size", config.block_size)?,
        max_rows: dim(
            "max rows (max_batch_size, prefill chunk)",
            limits.max_batch_rows.max(limits.prefill_chunk_rows),
        )?,
        max_k: dim("widest input dimension", max_k)?,
        max_n: dim("widest output dimension", max_n)?,
        max_n_one_row: dim("vocab_size", config.vocab_size)?,
        kernel_slots: GB10_MATMUL_KERNEL_SLOTS,
    })
}

/// [`reserve_gb10_serving`] with the GPU, the opt-in and the reservation call passed in.
///
/// `Ok(None)`: the reservation does not apply (not a GB10 build, not Qwen3, or the Qwen3 opt-in
/// is off, in which case the default refusal in `check_model` stops the model before any
/// serving). `Ok(Some(_))`: reserved. `Err(_)`: omega refused the reservation (or a bound could
/// not be formed); the caller must refuse to serve. There is no on-demand fallback.
pub fn reserve_gb10_serving_with(
    config: &ModelConfig,
    limits: &ServingLimits,
    gpu_native: bool,
    qwen3_opted_in: bool,
    reserve: &dyn Fn(&ServingBounds) -> Result<(), String>,
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
    let bounds = gb10_serving_bounds(config, limits).map_err(&refuse)?;
    reserve(&bounds).map_err(&refuse)?;
    Ok(Some(ServingReservation {
        bounds,
        bytes: bounds.bytes(),
    }))
}

/// Reserve the serving buffers once, after the GPU session and the weights are set up and
/// before the first request. See [`reserve_gb10_serving_with`] for the outcomes.
pub fn reserve_gb10_serving(
    config: &ModelConfig,
    limits: &ServingLimits,
) -> Result<Option<ServingReservation>, String> {
    reserve_gb10_serving_with(
        config,
        limits,
        OmegaGb10Backend::new().is_available(),
        gb10_qwen3_opted_in(),
        &|b| {
            aien_omega_gpu::reserve_serving(b).map_err(|e| {
                let stage = aien_omega_gpu::last_error();
                if stage.is_empty() {
                    e.to_string()
                } else {
                    format!("{e} (omega: {stage})")
                }
            })
        },
    )
}
