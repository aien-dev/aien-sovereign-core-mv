//! aien-rig memory budget for the dense Qwen3 engine: weights + KV cache + scratch must fit one
//! card's memory (the RTX 5070 has 12 GiB). Computed from the checkpoint's shape before anything is
//! uploaded, so the server can shrink the context or refuse instead of failing half way; the load
//! then measures what omega actually allocated (`aien_omega_gpu::alloc_stats`) and checks that too.
//!
//! What is counted on the card:
//! - weights as uploaded: bf16 `2nk`; FP8 `nk` plus the block-128 scale grid (counted at 4 bytes
//!   per scale, the widest a device copy holds); MXFP4 `nk/2` nibbles plus `nk/32` scale bytes. The
//!   lm_head is bf16 in every format (a tied head too: it is uploaded from the embedding).
//! - the paged bf16 KV pool for the context, exactly as `aien_kv_cache::KvPoolConfig` lays it out.
//! - scratch: a fixed allowance (kernel code, launch rings, attention and logits buffers) plus the
//!   prefill row buffers (x and y in f32 for the widest projection, `rows` rows at once).
//!
//! What is not on the card: the bf16 embedding table stays in system RAM (one row is read per
//! token), so it is reported but not counted.

use crate::qwen3_dense_omega::{DenseTensor, Qwen3DenseConfig};
use crate::qwen3_moe::FP8_BLOCK;
use crate::qwen3_moe_omega::KV_BLOCK_SIZE;

pub const GIB: u64 = 1 << 30;
/// The RTX 5070's memory (12 GiB GDDR7).
pub const RIG_BUDGET_BYTES: u64 = 12 * GIB;
/// Kernel code, launch rings, per-API x/y staging, attention scratch and logits. Checked against
/// the measured allocation on every budgeted load.
pub const FIXED_SCRATCH_BYTES: u64 = 256 << 20;
/// Smallest context worth serving; below this the load is refused rather than shrunk.
pub const MIN_CONTEXT: usize = 2048;
/// Context granularity when shrinking (whole 256-token steps).
pub const CONTEXT_STEP: usize = 256;

/// How the four per-layer projections sit on the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjFormat {
    /// As the checkpoint stores them (bf16, or Qwen's FP8 block-128).
    Checkpoint,
    /// Converted from Qwen's FP8 to MXFP4 at load.
    W4,
}

/// What to do when the asked context does not fit the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPolicy {
    /// Serve the largest context that fits (at least [`MIN_CONTEXT`]).
    Shrink,
    /// Refuse to start.
    Refuse,
}

impl BudgetPolicy {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "shrink" => Ok(Self::Shrink),
            "refuse" => Ok(Self::Refuse),
            other => Err(format!(
                "budget policy must be shrink or refuse, got {other}"
            )),
        }
    }
}

/// Bytes of the weights on the card.
pub fn device_weight_bytes(c: &Qwen3DenseConfig, fmt: ProjFormat) -> u64 {
    c.matrices()
        .iter()
        .map(|&(what, n, k)| {
            let (n, k) = (n as u64, k as u64);
            if what == DenseTensor::LmHead || !c.is_fp8(what) {
                2 * n * k
            } else if fmt == ProjFormat::W4 {
                n * k / 2 + n * k / 32
            } else {
                let b = FP8_BLOCK as u64;
                n * k + 4 * n.div_ceil(b) * k.div_ceil(b)
            }
        })
        .enumerate()
        .map(|(i, b)| if i == 4 { b } else { b * c.layers as u64 })
        .sum()
}

/// Bytes of the paged bf16 KV pool for `context` tokens (whole blocks, as the engine allocates it).
pub fn kv_bytes(c: &Qwen3DenseConfig, context: usize) -> Result<u64, String> {
    let blocks = context.div_ceil(KV_BLOCK_SIZE).max(1);
    let cfg = aien_kv_cache::KvPoolConfig::for_model(
        blocks,
        KV_BLOCK_SIZE,
        c.layers,
        c.kv_heads,
        c.head_dim,
        aien_kv_cache::KvDType::Bf16,
    );
    Ok(cfg.layout()?.total_bytes as u64)
}

/// KV bytes one token adds (K and V, every layer, bf16).
pub fn kv_bytes_per_token(c: &Qwen3DenseConfig) -> u64 {
    (c.layers * c.kv_heads * c.head_dim * 2 * 2) as u64
}

/// Scratch on the card: the fixed allowance plus `rows` prefill rows of x and y (f32) for the
/// widest projection, plus the f32 logits.
pub fn scratch_bytes(c: &Qwen3DenseConfig, rows: usize) -> u64 {
    let ms = c.matrices();
    let max_n = ms
        .iter()
        .filter(|m| m.0 != DenseTensor::LmHead)
        .map(|m| m.1)
        .max()
        .unwrap_or(0);
    let max_k = ms.iter().map(|m| m.2).max().unwrap_or(0);
    FIXED_SCRATCH_BYTES + 4 * (rows.max(1) * (max_n + max_k)) as u64 + 4 * c.vocab as u64
}

/// Prefill rows one projection launch takes: FP8 projections use omega's FP8 GEMM (up to 1024
/// rows); bf16 and MXFP4 run the GEMV per row.
pub fn prefill_rows(c: &Qwen3DenseConfig, fmt: ProjFormat) -> usize {
    if fmt == ProjFormat::Checkpoint && c.is_fp8(DenseTensor::Qkv) {
        aien_omega_fp8::GEMM_MAX_ROWS
    } else {
        1
    }
}

/// The plan the server prints and serves with.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetPlan {
    pub budget: u64,
    pub format: ProjFormat,
    pub weights: u64,
    pub scratch: u64,
    pub kv: u64,
    pub kv_per_token: u64,
    pub asked_context: usize,
    pub context: usize,
    pub embed_host: u64,
}

impl BudgetPlan {
    pub fn total(&self) -> u64 {
        self.weights + self.scratch + self.kv
    }
    pub fn shrunk(&self) -> bool {
        self.context < self.asked_context
    }
    /// One line for the log: every term, in GiB with three decimals.
    pub fn line(&self) -> String {
        let g = |b: u64| b as f64 / GIB as f64;
        format!(
            "memory budget {:.3} GiB: weights {:.3} ({:?}) + KV {:.3} ({} tokens x {} B) + scratch {:.3} = {:.3} GiB{}; embedding {:.3} GiB stays in system RAM",
            g(self.budget),
            g(self.weights),
            self.format,
            g(self.kv),
            self.context,
            self.kv_per_token,
            g(self.scratch),
            g(self.total()),
            if self.shrunk() {
                format!(" (context shrunk from {} to fit)", self.asked_context)
            } else {
                String::new()
            },
            g(self.embed_host),
        )
    }
}

/// Plans weights + KV + scratch inside `budget` for `asked_context` tokens (capped at the model's
/// `max_position`). Too big: `Shrink` serves the largest whole [`CONTEXT_STEP`] context that fits,
/// `Refuse` errors; either way a model that cannot hold [`MIN_CONTEXT`] tokens is refused.
pub fn plan(
    c: &Qwen3DenseConfig,
    format: ProjFormat,
    budget: u64,
    asked_context: usize,
    policy: BudgetPolicy,
) -> Result<BudgetPlan, String> {
    if format == ProjFormat::W4 && !c.is_fp8(DenseTensor::Qkv) {
        return Err(
            "MXFP4 is converted from Qwen's FP8 checkpoint; this checkpoint is bf16".into(),
        );
    }
    let weights = device_weight_bytes(c, format);
    let scratch = scratch_bytes(c, prefill_rows(c, format));
    let asked = asked_context.min(c.max_position.max(1));
    let mut p = BudgetPlan {
        budget,
        format,
        weights,
        scratch,
        kv: kv_bytes(c, asked)?,
        kv_per_token: kv_bytes_per_token(c),
        asked_context,
        context: asked,
        embed_host: (2 * c.vocab * c.hidden) as u64,
    };
    let g = |b: u64| b as f64 / GIB as f64;
    let min_kv = kv_bytes(c, MIN_CONTEXT)?;
    if weights + scratch + min_kv > budget {
        return Err(format!(
            "does not fit: weights {:.3} GiB ({format:?}) + scratch {:.3} + KV for {MIN_CONTEXT} tokens {:.3} = {:.3} GiB, budget {:.3} GiB",
            g(weights),
            g(scratch),
            g(min_kv),
            g(weights + scratch + min_kv),
            g(budget)
        ));
    }
    if p.total() <= budget {
        return Ok(p);
    }
    if policy == BudgetPolicy::Refuse {
        return Err(format!(
            "context {asked} does not fit: {:.3} GiB needed, budget {:.3} GiB (policy refuse; ask for less or use shrink)",
            g(p.total()),
            g(budget)
        ));
    }
    let room = budget - weights - scratch;
    let mut ctx = ((room / p.kv_per_token) as usize / CONTEXT_STEP) * CONTEXT_STEP;
    while ctx > MIN_CONTEXT && kv_bytes(c, ctx)? > room {
        ctx -= CONTEXT_STEP;
    }
    p.context = ctx.max(MIN_CONTEXT);
    p.kv = kv_bytes(c, p.context)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-8B-FP8's config.json (shape keys and quantization_config as shipped).
    const Q8B_FP8: &str = r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":4096,
        "intermediate_size":12288,"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
        "head_dim":128,"vocab_size":151936,"max_position_embeddings":40960,"rms_norm_eps":1e-6,
        "rope_theta":1000000,"tie_word_embeddings":false,"eos_token_id":151645,"hidden_act":"silu",
        "torch_dtype":"bfloat16","attention_bias":false,
        "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128]}}"#;

    fn q8b_fp8() -> Qwen3DenseConfig {
        Qwen3DenseConfig::from_json(Q8B_FP8, None).expect("8B FP8 config")
    }

    fn q8b_bf16() -> Qwen3DenseConfig {
        let mut v: serde_json::Value = serde_json::from_str(Q8B_FP8).unwrap();
        v.as_object_mut().unwrap().remove("quantization_config");
        Qwen3DenseConfig::from_json(&v.to_string(), None).expect("8B bf16 config")
    }

    #[test]
    fn weight_bytes_match_the_ceiling_table() {
        // engine/CEILING.md (aien-rig): 8B MXFP4 + bf16 head 4,934,598,656 B (calc:bdf022ee);
        // FP8 counts scales at 4 B here (the table counts the checkpoint's 2 B): + 847,872.
        assert_eq!(
            device_weight_bytes(&q8b_fp8(), ProjFormat::W4),
            4_934_598_656
        );
        assert_eq!(
            device_weight_bytes(&q8b_fp8(), ProjFormat::Checkpoint),
            8_191_275_008 + 847_872
        );
        assert_eq!(
            device_weight_bytes(&q8b_bf16(), ProjFormat::Checkpoint),
            15_136_194_560
        );
        assert_eq!(kv_bytes_per_token(&q8b_fp8()), 147_456);
    }

    #[test]
    fn w4_8b_holds_32k_in_12_gib() {
        let p = plan(
            &q8b_fp8(),
            ProjFormat::W4,
            RIG_BUDGET_BYTES,
            32768,
            BudgetPolicy::Refuse,
        )
        .unwrap();
        assert_eq!(p.context, 32768);
        assert!(p.total() <= RIG_BUDGET_BYTES, "{}", p.line());
    }

    #[test]
    fn fp8_8b_shrinks_or_refuses_at_32k() {
        let c = q8b_fp8();
        let p = plan(
            &c,
            ProjFormat::Checkpoint,
            RIG_BUDGET_BYTES,
            32768,
            BudgetPolicy::Shrink,
        )
        .unwrap();
        assert!(p.shrunk(), "{}", p.line());
        assert!(p.context >= MIN_CONTEXT && p.context % CONTEXT_STEP == 0);
        assert!(p.total() <= RIG_BUDGET_BYTES, "{}", p.line());
        // One more step would not fit: the shrink is the largest.
        let more = kv_bytes(&c, p.context + CONTEXT_STEP).unwrap();
        assert!(p.weights + p.scratch + more > RIG_BUDGET_BYTES);
        let r = plan(
            &c,
            ProjFormat::Checkpoint,
            RIG_BUDGET_BYTES,
            32768,
            BudgetPolicy::Refuse,
        );
        assert!(r.unwrap_err().contains("does not fit"));
    }

    #[test]
    fn bf16_8b_is_refused_at_any_context() {
        let e = plan(
            &q8b_bf16(),
            ProjFormat::Checkpoint,
            RIG_BUDGET_BYTES,
            4096,
            BudgetPolicy::Shrink,
        )
        .unwrap_err();
        assert!(e.starts_with("does not fit"), "{e}");
    }

    #[test]
    fn w4_needs_an_fp8_checkpoint() {
        let e = plan(
            &q8b_bf16(),
            ProjFormat::W4,
            RIG_BUDGET_BYTES,
            4096,
            BudgetPolicy::Shrink,
        )
        .unwrap_err();
        assert!(e.contains("bf16"), "{e}");
    }

    #[test]
    fn context_is_capped_at_the_model_max() {
        let p = plan(
            &q8b_fp8(),
            ProjFormat::W4,
            64 * GIB,
            1 << 20,
            BudgetPolicy::Refuse,
        )
        .unwrap();
        assert_eq!(p.context, 40960);
    }
}
