//! BitNet b1.58 2B-4T on AIEN's own GPU engine (Omega), ternary stage T1: config, loader,
//! ternariser, packer, 8-bit word table builder. Host only: no launch, no chip.
//!
//! Design: `~/handoffs/2026-10-15-ternary-engine-design.md` (section 3.3, stage table row T1).
//! Model spec (the ground truth this file cites as `spec`): `~/.t3/projects/research/ternary-oracle/spec.md`.
//! This file is new beside `qwen3_dense_omega.rs` and `qwen3_moe_omega.rs`; it reuses their
//! tensor sources, RSS probe and FP8 upload rule and edits neither.
//!
//! - [`BitNetConfig`]: the shape from `config.json`, refused by name unless it is the BitNet
//!   family this engine implements: `model_type` bitnet, `hidden_act` relu2, tied embeddings,
//!   `torch_dtype` bfloat16, `quantization_config` exactly {bitnet, autobitlinear, online}.
//! - [`absmean_ternarise`]: the quantisation rule of the bf16 master as Microsoft's reference
//!   converter computes it (spec 2.1, `REF/convert_checkpoint.py:23-27`, read only): every
//!   intermediate a bf16 tensor, so every step rounded to bf16. The 21 scales the spec lists from
//!   the reference int2 checkpoint equal ours (real-checkpoint test). Hugging Face's online mode
//!   differs: `WeightQuant.forward` casts to f32 first (`HF/integrations/bitnet.py:211`), so a
//!   weight near the half-scale boundary can ternarise differently there. One scale per
//!   projection (q, k, v, o, gate, up, down: 7 per layer), never per fused matrix.
//! - [`pack_trits`]: omega's 2-bit trit code (`omega/src/algebra/oma_trit.h`: bit0 plus plane,
//!   bit1 minus plane, 0b11 invalid), 16 trits per little-endian u32, row major
//!   ([`TERN_LAYOUT_ID`]). q|k|v and gate|up are stacked row blocks, each block with its own
//!   [`TernSegment`] scale, so a kernel never indexes scales by thirds (the reference decode
//!   kernel's bug, spec 2.3).
//! - [`build_fp8_table`]: the 8-bit word table (design 3.1 KT4 option A): the tied bf16
//!   embedding quantised to e4m3 in 128x128 blocks, scale = amax / 448 rounded to bf16,
//!   round-to-nearest-even, saturating (the rule of omega's `tests/gpu_lmhead_fp8_study.c`).
//! - The streaming loader: no f32 image of the model. Every projection is read as raw bf16 into
//!   one reused buffer, ternarised and packed into two more reused buffers, and handed to a
//!   [`TernUploader`]. The host keeps the bf16 embedding (the input lookup reads one row) and
//!   the f32 norms. Peak host memory is the embedding plus the buffers:
//!   [`bitnet_loader_transient_bytes`].
//!
//! The Q and K rows stay in the checkpoint (Hugging Face) order: the engine rotates them with
//! half-split RoPE pairs (dim j with j + 64, spec 1.3). Microsoft's int2 checkpoint permutes them
//! for adjacent pairs instead; this loader does not.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::qwen3_dense_omega::{check_fp8_upload, DenseGemv};
use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3fn, Shard, FP8_BLOCK};
use crate::qwen3_moe_omega::{
    peak_rss_bytes, KvGeometry, ShardSource, TensorSource, KV_BLOCK_SIZE,
};

#[cfg(any(test, feature = "host-test-engine"))]
pub mod test_support;

#[cfg(test)]
mod gguf_ref;

#[cfg(test)]
mod tests;

// The loader reads little-endian bf16 words in place and packs little-endian u32 words.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "the BitNet loader needs a little-endian host"
);

/// The packed weight layout every uploader receives, recorded in sealed goldens: omega's 2-bit
/// trit code, 16 trits per u32 word (trit `c % 16` at bits `2*(c % 16)`, plus plane low),
/// `[n][k / 16]` words row major, rows in checkpoint order (no RoPE permutation), q|k|v and
/// gate|up stacked as row blocks with one scale each.
pub const TERN_LAYOUT_ID: &str = "omega-trit2-rm16-hfrows-v1";

/// Trits per packed u32 word.
pub const TRITS_PER_WORD: usize = 16;

/// sha256 of the bf16 master `model.safetensors` (microsoft/bitnet-b1.58-2B-4T-bf16 at revision
/// 276681394656abdadb8e80e5b2c3db5e5d7fcaff, `weights.md` section 1). Checked by the real
/// checkpoint runs, never silently skipped there.
pub const BITNET_2B4T_SAFETENSORS_SHA256: &str =
    "529637ff6dab1f5890767356928693f69ffe61d3b6040a43de9306b37bfd5ae1";

/// sha256 of the same revision's `tokenizer.json`.
pub const BITNET_2B4T_TOKENIZER_SHA256: &str =
    "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7";

// ---------------------------------------------------------------------------
// Model shape, read from the checkpoint.
// ---------------------------------------------------------------------------

/// The four ternary matrices of a layer, as the engine stacks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernTensor {
    /// `[q_dim + 2 * kv_dim][hidden]`: q rows, then k rows, then v rows.
    Qkv,
    /// `[hidden][q_dim]`.
    O,
    /// `[2 * intermediate][hidden]`: gate rows, then up rows.
    GateUp,
    /// `[hidden][intermediate]`.
    Down,
}

impl TernTensor {
    pub const ALL: [TernTensor; 4] = [Self::Qkv, Self::O, Self::GateUp, Self::Down];

    pub fn name(self) -> &'static str {
        match self {
            Self::Qkv => "qkv",
            Self::O => "o",
            Self::GateUp => "gate_up",
            Self::Down => "down",
        }
    }
}

/// One checkpoint projection: the unit that owns a scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernPart {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl TernPart {
    /// The checkpoint tensor name of this projection in `layer`.
    pub fn tensor_name(self, layer: usize) -> String {
        let p = format!("model.layers.{layer}");
        match self {
            Self::Q => format!("{p}.self_attn.q_proj.weight"),
            Self::K => format!("{p}.self_attn.k_proj.weight"),
            Self::V => format!("{p}.self_attn.v_proj.weight"),
            Self::O => format!("{p}.self_attn.o_proj.weight"),
            Self::Gate => format!("{p}.mlp.gate_proj.weight"),
            Self::Up => format!("{p}.mlp.up_proj.weight"),
            Self::Down => format!("{p}.mlp.down_proj.weight"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

/// The BitNet shape, read from `config.json` and held to what this engine implements.
#[derive(Debug, Clone, PartialEq)]
pub struct BitNetConfig {
    pub layers: usize,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub max_position: usize,
    pub bos_token_id: u32,
    /// Ids generation stops on (`generation_config.json` eos list, else `config.json` eos).
    pub eos_token_ids: Vec<u32>,
}

/// `config.json` keys this reader knows. Anything else is refused by name: a key it does not
/// understand could change the maths.
const KNOWN_KEYS: &[&str] = &[
    "architectures",
    "auto_map",
    "bos_token_id",
    "eos_token_id",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "rms_norm_eps",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rope_theta",
    "tie_word_embeddings",
    "torch_dtype",
    "use_cache",
    "vocab_size",
    "quantization_config",
    // Accepted only at their neutral values (checked below).
    "attention_bias",
    "attention_dropout",
    "rope_scaling",
    "head_dim",
    "pad_token_id",
    "transformers_version",
];

fn refused(what: impl std::fmt::Display) -> String {
    format!("config refused: {what}")
}

fn eos_ids(v: &serde_json::Value) -> Option<Vec<u32>> {
    match v {
        serde_json::Value::Number(n) => Some(vec![u32::try_from(n.as_u64()?).ok()?]),
        serde_json::Value::Array(a) => a
            .iter()
            .map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok()))
            .collect(),
        _ => None,
    }
}

impl BitNetConfig {
    /// `config.json` and, when present, `generation_config.json` of a checkpoint directory.
    pub fn from_checkpoint(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let gen_path = dir.join("generation_config.json");
        let gen = match std::fs::read_to_string(&gen_path) {
            Ok(t) => Some(t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(format!("{}: {e}", gen_path.display())),
        };
        Self::from_json(&text, gen.as_deref()).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn from_json(config_json: &str, generation_json: Option<&str>) -> Result<Self, String> {
        let raw: serde_json::Value =
            serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        let obj = raw
            .as_object()
            .ok_or_else(|| refused("config.json is not an object"))?;
        let get = |key: &str| obj.get(key).filter(|v| !v.is_null());

        // The four family rules first, each refused by name with the value found.
        let named = |key: &str, want: serde_json::Value, why: &str| -> Result<(), String> {
            match get(key) {
                Some(v) if *v == want => Ok(()),
                other => Err(refused(format!(
                    "{key} {}, this engine implements {want} ({why})",
                    other.map_or("absent".to_string(), |v| v.to_string())
                ))),
            }
        };
        named(
            "model_type",
            serde_json::json!("bitnet"),
            "BitNet b1.58 only",
        )?;
        named(
            "hidden_act",
            serde_json::json!("relu2"),
            "squared ReLU on the gate",
        )?;
        named(
            "tie_word_embeddings",
            serde_json::json!(true),
            "the embedding is the output head",
        )?;
        named(
            "torch_dtype",
            serde_json::json!("bfloat16"),
            "the loader streams the bf16 master",
        )?;
        named(
            "quantization_config",
            serde_json::json!({
                "quant_method": "bitnet",
                "linear_class": "autobitlinear",
                "quantization_mode": "online"
            }),
            "ternary weights computed from the bf16 master at load",
        )?;
        if let Some(a) = get("architectures") {
            if *a != serde_json::json!(["BitNetForCausalLM"]) {
                return Err(refused(format!(
                    "architectures {a}, this engine implements [\"BitNetForCausalLM\"]"
                )));
            }
        }
        for key in obj.keys() {
            if !KNOWN_KEYS.contains(&key.as_str()) {
                return Err(refused(format!("field {key:?} is not implemented")));
            }
        }
        // Neutral values only.
        if get("attention_bias").is_some_and(|v| *v != serde_json::json!(false)) {
            return Err(refused("attention_bias must be false (no biases)"));
        }
        if get("attention_dropout").is_some_and(|v| v.as_f64() != Some(0.0)) {
            return Err(refused("attention_dropout must be 0"));
        }
        if let Some(r) = get("rope_scaling") {
            return Err(refused(format!(
                "rope_scaling {r} is not implemented (plain RoPE only)"
            )));
        }

        let int = |key: &str| -> Result<usize, String> {
            get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| refused(format!("{key} missing or not a whole number")))
        };
        let float = |key: &str| -> Result<f64, String> {
            get(key)
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| refused(format!("{key} missing or not a positive number")))
        };
        let hidden = int("hidden_size")?;
        let q_heads = int("num_attention_heads")?;
        if q_heads == 0 || !hidden.is_multiple_of(q_heads) {
            return Err(refused(format!(
                "hidden_size {hidden} is not a whole number of {q_heads} heads"
            )));
        }
        let head_dim = hidden / q_heads;
        if let Some(h) = get("head_dim") {
            if h.as_u64() != Some(head_dim as u64) {
                return Err(refused(format!(
                    "head_dim {h}, hidden_size / num_attention_heads is {head_dim}"
                )));
            }
        }
        let bos_token_id = get("bos_token_id")
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| refused("bos_token_id missing"))?;
        let mut eos_token_ids = match get("eos_token_id") {
            None => return Err(refused("eos_token_id missing")),
            Some(v) => eos_ids(v).ok_or_else(|| {
                refused(format!(
                    "eos_token_id {v} is not a token id or a list of them"
                ))
            })?,
        };
        if let Some(g) = generation_json {
            let g: serde_json::Value =
                serde_json::from_str(g).map_err(|e| format!("generation_config.json: {e}"))?;
            if let Some(v) = g.get("eos_token_id") {
                eos_token_ids = eos_ids(v).ok_or_else(|| {
                    refused(format!(
                        "generation_config.json eos_token_id {v} is not a token id or a list of them"
                    ))
                })?;
            }
        }
        let cfg = Self {
            layers: int("num_hidden_layers")?,
            hidden,
            q_heads,
            kv_heads: int("num_key_value_heads")?,
            head_dim,
            intermediate: int("intermediate_size")?,
            vocab: int("vocab_size")?,
            rms_eps: float("rms_norm_eps")? as f32,
            rope_theta: float("rope_theta")? as f32,
            max_position: int("max_position_embeddings")?,
            bos_token_id,
            eos_token_ids,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// What this engine can pack and run: whole GQA groups, an even head width (RoPE pairs),
    /// every `k` a whole number of packed words, the word table on omega's FP8 block grid.
    pub fn validate(&self) -> Result<(), String> {
        if self.layers == 0 || self.vocab == 0 || self.kv_heads == 0 || self.intermediate == 0 {
            return Err(refused("zero layers, vocab, kv heads or intermediate"));
        }
        if self.q_heads * self.head_dim != self.hidden {
            return Err(refused(format!(
                "{} heads of {} is not hidden {}",
                self.q_heads, self.head_dim, self.hidden
            )));
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(refused(format!(
                "{} query heads over {} kv heads is not a whole GQA group",
                self.q_heads, self.kv_heads
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(refused(format!(
                "head_dim {} is odd (RoPE rotates pairs)",
                self.head_dim
            )));
        }
        for (what, k) in [
            ("hidden_size", self.hidden),
            ("intermediate_size", self.intermediate),
        ] {
            if !k.is_multiple_of(TRITS_PER_WORD) {
                return Err(refused(format!(
                    "{what} {k} is not a multiple of {TRITS_PER_WORD} (trits per packed word)"
                )));
            }
        }
        if self.eos_token_ids.is_empty() {
            return Err(refused(
                "eos_token_id is an empty list: generation could never stop",
            ));
        }
        if self.bos_token_id as usize >= self.vocab
            || self.eos_token_ids.iter().any(|&e| e as usize >= self.vocab)
        {
            return Err(refused("bos or eos id outside the vocabulary"));
        }
        aien_omega_fp8::check_shape(1, self.vocab, self.hidden)
            .map_err(|e| refused(format!("word table [{}][{}]: {e}", self.vocab, self.hidden)))
    }

    pub fn q_dim(&self) -> usize {
        self.q_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The projections stacked into `what`, in row order, with their row counts.
    pub fn parts(&self, what: TernTensor) -> Vec<(TernPart, usize)> {
        match what {
            TernTensor::Qkv => vec![
                (TernPart::Q, self.q_dim()),
                (TernPart::K, self.kv_dim()),
                (TernPart::V, self.kv_dim()),
            ],
            TernTensor::O => vec![(TernPart::O, self.hidden)],
            TernTensor::GateUp => vec![
                (TernPart::Gate, self.intermediate),
                (TernPart::Up, self.intermediate),
            ],
            TernTensor::Down => vec![(TernPart::Down, self.hidden)],
        }
    }

    /// `(n, k)` of a stacked matrix.
    pub fn shape(&self, what: TernTensor) -> (usize, usize) {
        let n = self.parts(what).iter().map(|p| p.1).sum();
        let k = match what {
            TernTensor::Qkv | TernTensor::GateUp => self.hidden,
            TernTensor::O => self.q_dim(),
            TernTensor::Down => self.intermediate,
        };
        (n, k)
    }

    /// Packed bytes of one stacked matrix plus its segment scales (bf16 each).
    pub fn ternary_bytes(&self, what: TernTensor) -> usize {
        let (n, k) = self.shape(what);
        n * k / 4 + 2 * self.parts(what).len()
    }

    /// The e4m3 word table plus its bf16 block scales.
    pub fn table_bytes(&self) -> usize {
        self.vocab * self.hidden + 2 * (self.vocab / FP8_BLOCK) * (self.hidden / FP8_BLOCK)
    }

    /// Everything handed to the uploader: every layer's packed matrices and scales, the table.
    pub fn model_resident_bytes(&self) -> usize {
        self.layers
            * TernTensor::ALL
                .iter()
                .map(|&w| self.ternary_bytes(w))
                .sum::<usize>()
            + self.table_bytes()
    }

    /// The bf16 embedding the host keeps.
    pub fn embed_host_bytes(&self) -> usize {
        2 * self.vocab * self.hidden
    }

    /// Every tensor name the checkpoint must hold, and no other (332 for 2B-4T: the embedding,
    /// 11 per layer, the final norm; no `lm_head`, no bias, no stored scale).
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names = vec!["model.embed_tokens.weight".to_string()];
        for l in 0..self.layers {
            let p = format!("model.layers.{l}");
            names.push(format!("{p}.input_layernorm.weight"));
            names.push(format!("{p}.self_attn.attn_sub_norm.weight"));
            names.push(format!("{p}.post_attention_layernorm.weight"));
            names.push(format!("{p}.mlp.ffn_sub_norm.weight"));
            for what in TernTensor::ALL {
                names.extend(self.parts(what).into_iter().map(|(p, _)| p.tensor_name(l)));
            }
        }
        names.push("model.norm.weight".into());
        names
    }

    /// Refuses a checkpoint whose tensor names are not exactly [`Self::tensor_names`], naming
    /// the first few unexpected and missing ones.
    pub fn check_tensor_names(&self, present: &[String]) -> Result<(), String> {
        let names = self.tensor_names();
        let want: std::collections::BTreeSet<&str> = names.iter().map(String::as_str).collect();
        let have: std::collections::BTreeSet<&str> = present.iter().map(String::as_str).collect();
        let extra: Vec<&str> = have.difference(&want).take(5).copied().collect();
        let missing: Vec<&str> = want.difference(&have).take(5).copied().collect();
        if extra.is_empty() && missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "checkpoint tensors refused: {} present, {} expected; unexpected {extra:?}; missing {missing:?}",
            have.len(),
            want.len()
        ))
    }

    /// Largest stacked matrix, in elements.
    pub fn max_matrix_elems(&self) -> usize {
        TernTensor::ALL
            .iter()
            .map(|&w| {
                let (n, k) = self.shape(w);
                n * k
            })
            .max()
            .unwrap_or(0)
    }
}

impl From<&BitNetConfig> for KvGeometry {
    fn from(c: &BitNetConfig) -> Self {
        Self {
            layers: c.layers,
            q_heads: c.q_heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            rope_theta: c.rope_theta,
            block_size: KV_BLOCK_SIZE,
        }
    }
}

/// Host bytes the loader holds beyond the model's own, at its peak: the larger of the two
/// phases, which never overlap. Phase 1 holds the bf16 read buffer, the trit buffer and the
/// packed-word buffer (each sized for the largest stacked matrix); they are dropped before
/// phase 2 builds the e4m3 table and its scales.
pub fn bitnet_loader_transient_bytes(c: &BitNetConfig) -> usize {
    let m = c.max_matrix_elems();
    (2 * m + m + m / 4).max(c.table_bytes())
}

// ---------------------------------------------------------------------------
// The ternariser (spec 2.1).
// ---------------------------------------------------------------------------

/// bf16 round-to-nearest-even of an f32, as PyTorch stores every bf16 op's result.
fn bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// The absmean scale of one projection, every step as PyTorch computes it on a bf16 tensor
/// (`REF/convert_checkpoint.py:23-27`):
/// `mean = bf16(mean |W|)`, `s = bf16(1 / max(mean, 1e-5))`, `w_scale = bf16(1 / s)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbsmeanScale {
    /// `bf16(mean |W|)`.
    pub mean: f32,
    /// `bf16(1 / mean)`: the multiplier the trits are rounded from.
    pub s: f32,
    /// `bf16(1 / s)`: the stored weight scale (`weight_scale` in the reference int2 checkpoint),
    /// what a GEMV multiplies its integer sum by.
    pub w_scale: f32,
}

impl AbsmeanScale {
    /// The stored scale as bf16 bits.
    pub fn w_scale_bf16(&self) -> u16 {
        f32_to_bf16(self.w_scale)
    }
}

/// Elements per partial sum of the mean. Fixed, so the sum's order (and so its rounding) does
/// not depend on the thread count.
const MEAN_CHUNK: usize = 1 << 16;

/// `bf16(mean |W|)` over bf16 words. The partial sums are f64 over fixed chunks, added in order;
/// the mean is rounded to f32 and then to bf16, as PyTorch's f32 reduction of a bf16 tensor
/// returns it. The f64 sum is exact for these weights (bf16 values of one projection span far
/// fewer than 53 bits), so the only rounding is PyTorch's own; spec 7 item 2 is resolved
/// against the reference int2 scales by the real-checkpoint test.
fn absmean(bits: &[u16]) -> f32 {
    let sum: f64 = bits
        .par_chunks(MEAN_CHUNK)
        .map(|c| c.iter().map(|&b| bf16_to_f32(b).abs() as f64).sum::<f64>())
        .collect::<Vec<f64>>()
        .into_iter()
        .sum();
    bf16((sum / bits.len() as f64) as f32)
}

/// The ternary value of one bf16 weight under multiplier `s`: `round(bf16(w * s))` with ties to
/// even (PyTorch `round`), clamped to [-1, 1]. The product of two bf16 values is exact in f32,
/// so `bf16(w * s)` is PyTorch's bf16 multiply.
#[inline]
pub fn trit_of(w_bits: u16, s: f32) -> i8 {
    let p = bf16(bf16_to_f32(w_bits) * s);
    // round-half-even then clamp: |p| <= 0.5 gives 0, above gives the sign.
    if p > 0.5 {
        1
    } else if p < -0.5 {
        -1
    } else {
        0
    }
}

/// Ternarises one projection of bf16 words into `trits` (one i8 per weight, same order) and
/// returns its scale. A NaN or infinite weight is refused with its index; an empty projection
/// is refused.
pub fn absmean_ternarise(bits: &[u16], trits: &mut [i8]) -> Result<AbsmeanScale, String> {
    if bits.is_empty() || bits.len() != trits.len() {
        return Err(format!(
            "ternarise: {} weights into {} trits",
            bits.len(),
            trits.len()
        ));
    }
    if let Some(at) = bits.iter().position(|&b| b & 0x7f80 == 0x7f80) {
        return Err(format!("weight {at} is NaN or infinite"));
    }
    // `clamp_(min=1e-5)` on a bf16 tensor: the bound is bf16(1e-5).
    let mean = absmean(bits).max(bf16(1e-5));
    let s = bf16(1.0 / mean);
    let w_scale = bf16(1.0 / s);
    trits
        .par_chunks_mut(MEAN_CHUNK)
        .zip(bits.par_chunks(MEAN_CHUNK))
        .for_each(|(t, b)| {
            for (t, &b) in t.iter_mut().zip(b) {
                *t = trit_of(b, s);
            }
        });
    Ok(AbsmeanScale { mean, s, w_scale })
}

// ---------------------------------------------------------------------------
// The packer: omega's trit code, 16 per word.
// ---------------------------------------------------------------------------

/// omega's 2-bit code of a trit (`oma_trit.h`): 0 -> 0b00, +1 -> 0b01 (plus plane), -1 -> 0b10
/// (minus plane). 0b11 is never produced.
#[inline]
pub fn trit_code(t: i8) -> u32 {
    match t {
        1 => 0b01,
        -1 => 0b10,
        _ => 0b00,
    }
}

/// Decodes one 2-bit code; 0b11 is not a trit (`None`).
#[inline]
pub fn code_trit(code: u32) -> Option<i8> {
    match code & 3 {
        0b00 => Some(0),
        0b01 => Some(1),
        0b10 => Some(-1),
        _ => None,
    }
}

/// Packs `[n][k]` trits (row major, `k % 16 == 0`) into `[n][k / 16]` words of
/// [`TERN_LAYOUT_ID`].
pub fn pack_trits(trits: &[i8], k: usize, words: &mut [u32]) -> Result<(), String> {
    if !k.is_multiple_of(TRITS_PER_WORD)
        || !trits.len().is_multiple_of(k)
        || words.len() * TRITS_PER_WORD != trits.len()
    {
        return Err(format!(
            "pack: {} trits of row width {k} into {} words",
            trits.len(),
            words.len()
        ));
    }
    words
        .par_chunks_mut(4096)
        .zip(trits.par_chunks(4096 * TRITS_PER_WORD))
        .for_each(|(w, t)| {
            for (word, group) in w.iter_mut().zip(t.chunks_exact(TRITS_PER_WORD)) {
                *word = group
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &v)| acc | (trit_code(v) << (2 * i)));
            }
        });
    Ok(())
}

/// The first packed word holding a 0b11 code, if any, with the trit index inside it.
pub fn find_invalid_code(words: &[u32]) -> Option<(usize, usize)> {
    words.iter().enumerate().find_map(|(i, &w)| {
        let both = w & (w >> 1) & 0x5555_5555;
        (both != 0).then(|| (i, both.trailing_zeros() as usize / 2))
    })
}

/// One row block of a stacked ternary matrix and its scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernSegment {
    pub part: TernPart,
    pub rows: usize,
    /// The stored weight scale, bf16 bits ([`AbsmeanScale::w_scale_bf16`]).
    pub scale_bf16: u16,
}

/// The ternary upload rule, checked on the host before anything is handed to a device: lengths
/// for `[n][k]` in [`TERN_LAYOUT_ID`], segments covering exactly `n` rows with finite positive
/// scales, and no 0b11 code. The first offender is named.
pub fn check_ternary_upload(
    n: usize,
    k: usize,
    words: &[u32],
    segments: &[TernSegment],
) -> Result<(), String> {
    if n == 0 || !k.is_multiple_of(TRITS_PER_WORD) || words.len() != n * k / TRITS_PER_WORD {
        return Err(format!(
            "{} words for [{n}][{k}] in {TERN_LAYOUT_ID}",
            words.len()
        ));
    }
    let rows: usize = segments.iter().map(|s| s.rows).sum();
    if rows != n || segments.iter().any(|s| s.rows == 0) {
        return Err(format!("segments cover {rows} rows of {n}"));
    }
    if let Some(s) = segments.iter().find(|s| {
        let v = bf16_to_f32(s.scale_bf16);
        !(v.is_finite() && v > 0.0)
    }) {
        return Err(format!(
            "segment {} scale {} is not finite and positive",
            s.part.name(),
            bf16_to_f32(s.scale_bf16)
        ));
    }
    if let Some((w, t)) = find_invalid_code(words) {
        let wpr = k / TRITS_PER_WORD;
        return Err(format!(
            "row {} column {}: code 0b11 is not a trit (refused at upload)",
            w / wpr,
            (w % wpr) * TRITS_PER_WORD + t
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The 8-bit word table (design 3.1 KT4 option A).
// ---------------------------------------------------------------------------

/// Quantises the `[n][k]` bf16 table to e4m3 in 128x128 blocks into `w` (`n * k` bytes) and
/// `scales` (`[n/128][k/128]` bf16): per block `scale = bf16(amax / 448)` (1.0 for an all-zero
/// block), each weight `e4m3(w / scale)` round-to-nearest-even and saturating, a negative zero
/// stored as +0 (omega's `gpu_lmhead_fp8_study.c` rule, byte for byte).
pub fn build_fp8_table(
    table: &[u16],
    n: usize,
    k: usize,
    w: &mut [u8],
    scales: &mut [u16],
) -> Result<(), String> {
    let b = FP8_BLOCK;
    if !n.is_multiple_of(b)
        || !k.is_multiple_of(b)
        || table.len() != n * k
        || w.len() != n * k
        || scales.len() != (n / b) * (k / b)
    {
        return Err(format!(
            "fp8 table: [{n}][{k}] with {} words, {} bytes, {} scales",
            table.len(),
            w.len(),
            scales.len()
        ));
    }
    if let Some(at) = table.iter().position(|&x| x & 0x7f80 == 0x7f80) {
        return Err(format!("word table element {at} is NaN or infinite"));
    }
    let kb = k / b;
    // One block row (128 table rows) per task.
    w.par_chunks_mut(b * k)
        .zip(scales.par_chunks_mut(kb))
        .enumerate()
        .for_each(|(bi, (wrows, srow))| {
            let src = &table[bi * b * k..(bi + 1) * b * k];
            for bj in 0..kb {
                let cols = bj * b..(bj + 1) * b;
                let amax = (0..b)
                    .flat_map(|r| src[r * k + cols.start..r * k + cols.end].iter())
                    .map(|&x| bf16_to_f32(x).abs())
                    .fold(0.0f32, f32::max);
                let sb = f32_to_bf16(if amax > 0.0 { amax / 448.0 } else { 1.0 });
                let s = bf16_to_f32(sb);
                srow[bj] = sb;
                for r in 0..b {
                    for c in cols.clone() {
                        let code = f32_to_fp8_e4m3fn(bf16_to_f32(src[r * k + c]) / s);
                        wrows[r * k + c] = if code == 0x80 { 0 } else { code };
                    }
                }
            }
        });
    Ok(())
}

// ---------------------------------------------------------------------------
// The device side, behind a trait so the loader runs without a chip.
// ---------------------------------------------------------------------------

/// One resident stacked ternary matrix. The GEMV contract is omega's Omega-X
/// (`realize_common.h`): `acc[r] = sum_c W[r][c] * x[c]`, W in {-1, 0, 1}, x int8, exact int32.
/// The per-segment scaling is the caller's (it knows the activation scale).
pub trait TernMatrix: Send + Sync + std::fmt::Debug {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    fn segments(&self) -> &[TernSegment];
    fn gemv_i8(&self, x: &[i8], acc: &mut [i32]) -> Result<(), String>;
}

/// Makes packed ternary matrices and the e4m3 word table resident. The real one (omega's KT1
/// `tensor_upload`) comes with stage T3; the loader is the same.
pub trait TernUploader {
    /// `[n][k]` trits in [`TERN_LAYOUT_ID`], `segments` the row blocks in order.
    fn upload_ternary(
        &self,
        layer: usize,
        what: TernTensor,
        n: usize,
        k: usize,
        words: &[u32],
        segments: &[TernSegment],
    ) -> Result<Arc<dyn TernMatrix>, String>;

    /// The `[n][k]` e4m3 word table with its `[n/128][k/128]` bf16 block scales.
    fn upload_table_fp8(
        &self,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String>;
}

/// A packed ternary matrix on the host: the exact int32 GEMV (the oracle contract) and the
/// unpacking the byte tests use.
#[derive(Debug)]
pub struct HostTernMatrix {
    n: usize,
    k: usize,
    words: Vec<u32>,
    segments: Vec<TernSegment>,
}

impl HostTernMatrix {
    /// Checked by [`check_ternary_upload`] first.
    pub fn new(
        n: usize,
        k: usize,
        words: Vec<u32>,
        segments: Vec<TernSegment>,
    ) -> Result<Self, String> {
        check_ternary_upload(n, k, &words, &segments)?;
        Ok(Self {
            n,
            k,
            words,
            segments,
        })
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Row `r` as trits.
    pub fn row(&self, r: usize) -> Vec<i8> {
        let wpr = self.k / TRITS_PER_WORD;
        self.words[r * wpr..(r + 1) * wpr]
            .iter()
            .flat_map(|&w| (0..TRITS_PER_WORD).map(move |i| code_trit(w >> (2 * i)).unwrap_or(0)))
            .collect()
    }
}

impl TernMatrix for HostTernMatrix {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn segments(&self) -> &[TernSegment] {
        &self.segments
    }
    fn gemv_i8(&self, x: &[i8], acc: &mut [i32]) -> Result<(), String> {
        if x.len() != self.k || acc.len() != self.n {
            return Err(format!(
                "ternary gemv [{}][{}] given x {} acc {}",
                self.n,
                self.k,
                x.len(),
                acc.len()
            ));
        }
        let wpr = self.k / TRITS_PER_WORD;
        acc.par_iter_mut().enumerate().for_each(|(r, out)| {
            let row = &self.words[r * wpr..(r + 1) * wpr];
            let mut sum = 0i32;
            for (j, &w) in row.iter().enumerate() {
                let xs = &x[j * TRITS_PER_WORD..(j + 1) * TRITS_PER_WORD];
                for (i, &xv) in xs.iter().enumerate() {
                    // plus plane adds, minus plane subtracts (validated: never both).
                    let code = (w >> (2 * i)) & 3;
                    sum += (code & 1) as i32 * xv as i32 - (code >> 1) as i32 * xv as i32;
                }
            }
            *out = sum;
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint sources.
// ---------------------------------------------------------------------------

/// One `model.safetensors` file (the BitNet master is not sharded), with the same validated
/// reader as the shard loader: dtype, shape and byte range checked per tensor.
pub struct SafetensorsFile(Shard);

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        Shard::open(path).map(Self).map_err(|e| e.to_string())
    }
}

impl TensorSource for SafetensorsFile {
    fn read_into(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        dst: &mut [u8],
    ) -> Result<(), String> {
        self.0
            .read(name, dtype, shape, dst)
            .map_err(|e| e.to_string())
    }

    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        Ok(self.0.dtype(name).map(str::to_string))
    }
}

/// The tensor names a checkpoint directory holds: the `weight_map` of
/// `model.safetensors.index.json` when present, else the header of `model.safetensors`.
pub fn checkpoint_tensor_names(dir: &Path) -> Result<Vec<String>, String> {
    use std::io::Read;
    let header: serde_json::Value = if dir.join("model.safetensors.index.json").exists() {
        let p = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let index: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
        index
            .get("weight_map")
            .cloned()
            .ok_or_else(|| format!("{}: no weight_map", p.display()))?
    } else {
        let p = dir.join("model.safetensors");
        let io = |e: std::io::Error| format!("{}: {e}", p.display());
        let mut f = std::fs::File::open(&p).map_err(io)?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(io)?;
        let len = u64::from_le_bytes(len);
        if len > 100 << 20 {
            return Err(format!("{}: header of {len} bytes refused", p.display()));
        }
        let mut text = vec![0u8; len as usize];
        f.read_exact(&mut text).map_err(io)?;
        serde_json::from_slice(&text).map_err(|e| format!("{}: header: {e}", p.display()))?
    };
    let map = header
        .as_object()
        .ok_or("safetensors header is not an object")?;
    Ok(map
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect())
}

/// The checkpoint's tensors: the shard index when there is one, else `model.safetensors`.
pub fn open_checkpoint(dir: &Path) -> Result<Box<dyn TensorSource>, String> {
    if dir.join("model.safetensors.index.json").exists() {
        Ok(Box::new(ShardSource::open(dir)?))
    } else {
        Ok(Box::new(SafetensorsFile::open(
            &dir.join("model.safetensors"),
        )?))
    }
}

// ---------------------------------------------------------------------------
// The loaded model.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct BitNetLayer {
    /// RMSNorm before attention (`input_layernorm`), `[hidden]`.
    pub input_norm: Vec<f32>,
    /// RMSNorm on the attention output before o (`attn_sub_norm`), `[q_dim]`.
    pub attn_sub_norm: Vec<f32>,
    /// RMSNorm before the feed-forward (`post_attention_layernorm`), `[hidden]`.
    pub post_norm: Vec<f32>,
    /// RMSNorm on `relu(gate)^2 * up` before down (`ffn_sub_norm`), `[intermediate]`.
    pub ffn_sub_norm: Vec<f32>,
    pub qkv: Arc<dyn TernMatrix>,
    pub o: Arc<dyn TernMatrix>,
    pub gate_up: Arc<dyn TernMatrix>,
    pub down: Arc<dyn TernMatrix>,
}

impl BitNetLayer {
    pub fn matrix(&self, what: TernTensor) -> &Arc<dyn TernMatrix> {
        match what {
            TernTensor::Qkv => &self.qkv,
            TernTensor::O => &self.o,
            TernTensor::GateUp => &self.gate_up,
            TernTensor::Down => &self.down,
        }
    }
}

/// The whole model: packed matrices and the 8-bit table on the device, the bf16 embedding and
/// f32 norms on the host.
#[derive(Debug)]
pub struct BitNetOmegaWeights {
    pub config: BitNetConfig,
    /// `[vocab][hidden]` bf16 bits: the input lookup (one row widened to f32) and, being tied,
    /// the 16-bit output head of the host reference.
    pub embed_bf16: Vec<u16>,
    pub layers: Vec<BitNetLayer>,
    pub final_norm: Vec<f32>,
    /// The e4m3 word table (the output head on the chip).
    pub table_fp8: Arc<dyn DenseGemv>,
}

impl BitNetOmegaWeights {
    /// The embedding row of `token` in f32; a token outside the vocabulary is refused.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>, String> {
        let h = self.config.hidden;
        let t = token as usize;
        if t >= self.config.vocab {
            return Err(format!(
                "token {token} outside the vocabulary of {}",
                self.config.vocab
            ));
        }
        Ok(self.embed_bf16[t * h..(t + 1) * h]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect())
    }

    /// Bytes the host keeps after the load (embedding and norms).
    pub fn host_bytes(&self) -> usize {
        self.embed_bf16.len() * 2
            + self.final_norm.len() * 4
            + self
                .layers
                .iter()
                .map(|l| {
                    4 * (l.input_norm.len()
                        + l.attn_sub_norm.len()
                        + l.post_norm.len()
                        + l.ffn_sub_norm.len())
                })
                .sum::<usize>()
    }
}

/// What a load measured.
#[derive(Debug, Clone, PartialEq)]
pub struct BitNetLoadReport {
    /// Bytes handed to the uploader: packed words, segment scales, e4m3 table and its scales.
    pub resident_bytes: usize,
    pub host_bytes: usize,
    /// The loader's reused buffers, bytes ([`bitnet_loader_transient_bytes`]).
    pub transient_bytes: usize,
    pub uploads: usize,
    /// Every projection's scale, `[layer][q, k, v, o, gate, up, down]`.
    pub scales: Vec<[AbsmeanScale; 7]>,
    /// Process peak resident set (VmHWM) before and after the load, if `/proc` is readable.
    pub peak_rss_before_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub seconds: f64,
}

/// A u16 slice as bytes of twice the length.
fn words_as_bytes_mut(words: &mut [u16]) -> &mut [u8] {
    // SAFETY: u8 has no alignment need and every byte pattern is a valid u16 half; the length
    // is exactly the slice's size in bytes.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 2) }
}

/// Reads one bf16 tensor (dtype and shape checked by the source) into `dst` as words.
fn read_bf16(
    source: &mut dyn TensorSource,
    name: &str,
    shape: &[usize],
    dst: &mut [u16],
) -> Result<(), String> {
    if shape.iter().product::<usize>() != dst.len() {
        return Err(format!(
            "{name}: shape {shape:?} does not fill {} words",
            dst.len()
        ));
    }
    source.read_into(name, "BF16", shape, words_as_bytes_mut(dst))
}

/// The streaming loader's state: the source, the uploader, its reused buffers, the totals.
struct Stream<'a> {
    config: &'a BitNetConfig,
    source: &'a mut dyn TensorSource,
    uploader: &'a dyn TernUploader,
    bf16: Vec<u16>,
    trits: Vec<i8>,
    words: Vec<u32>,
    resident: usize,
    uploads: usize,
}

impl Stream<'_> {
    fn norm(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        read_bf16(self.source, name, &[len], &mut self.bf16[..len])?;
        let v: Vec<f32> = self.bf16[..len].iter().map(|&b| bf16_to_f32(b)).collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("{name}: NaN or infinite norm weight"));
        }
        Ok(v)
    }

    /// Reads the parts of `what` back to back (so the stacked `[n][k]` is the parts in order with
    /// no copy), ternarises each part with its own scale, packs the stack and uploads it.
    fn matrix(
        &mut self,
        layer: usize,
        what: TernTensor,
    ) -> Result<(Arc<dyn TernMatrix>, Vec<AbsmeanScale>), String> {
        let (n, k) = self.config.shape(what);
        let at = format!("layer {layer} {} [{n}][{k}]", what.name());
        let mut segments = Vec::new();
        let mut scales = Vec::new();
        let mut off = 0usize;
        for (part, rows) in self.config.parts(what) {
            let name = part.tensor_name(layer);
            let range = off..off + rows * k;
            read_bf16(
                self.source,
                &name,
                &[rows, k],
                &mut self.bf16[range.clone()],
            )?;
            let scale = absmean_ternarise(&self.bf16[range.clone()], &mut self.trits[range])
                .map_err(|e| format!("{name}: {e}"))?;
            segments.push(TernSegment {
                part,
                rows,
                scale_bf16: scale.w_scale_bf16(),
            });
            scales.push(scale);
            off += rows * k;
        }
        let words = &mut self.words[..n * k / TRITS_PER_WORD];
        pack_trits(&self.trits[..n * k], k, words).map_err(|e| format!("{at}: {e}"))?;
        let m = self
            .uploader
            .upload_ternary(layer, what, n, k, words, &segments)
            .map_err(|e| format!("{at} upload: {e}"))?;
        if m.n() != n || m.k() != k {
            return Err(format!("{at} upload returned [{}][{}]", m.n(), m.k()));
        }
        self.resident += self.config.ternary_bytes(what);
        self.uploads += 1;
        Ok((m, scales))
    }
}

/// Loads a checkpoint directory: `config.json` for the shape, `model.safetensors` (or shards)
/// for the weights.
pub fn load_bitnet_omega(
    dir: &Path,
    uploader: &dyn TernUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(BitNetOmegaWeights, BitNetLoadReport), String> {
    let config = BitNetConfig::from_checkpoint(dir)?;
    config.check_tensor_names(&checkpoint_tensor_names(dir)?)?;
    let mut source = open_checkpoint(dir)?;
    load_bitnet_omega_from(&config, source.as_mut(), uploader, progress)
}

/// The streaming load from any [`TensorSource`].
pub fn load_bitnet_omega_from(
    config: &BitNetConfig,
    source: &mut dyn TensorSource,
    uploader: &dyn TernUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(BitNetOmegaWeights, BitNetLoadReport), String> {
    config.validate()?;
    let t0 = Instant::now();
    let peak_before = peak_rss_bytes();
    let (h, i) = (config.hidden, config.intermediate);

    // Every tensor must be stored bf16 (the header decides, not the name).
    let mut embed_bf16 = vec![0u16; config.vocab * h];
    read_bf16(
        source,
        "model.embed_tokens.weight",
        &[config.vocab, h],
        &mut embed_bf16,
    )?;
    if let Some(at) = embed_bf16.iter().position(|&x| x & 0x7f80 == 0x7f80) {
        return Err(format!(
            "model.embed_tokens.weight: element {at} is NaN or infinite"
        ));
    }
    if source.dtype_of("lm_head.weight")?.is_some() {
        return Err(
            "lm_head.weight present in a tied checkpoint: refused (the embedding is the head)"
                .into(),
        );
    }

    let m = config.max_matrix_elems();
    let mut s = Stream {
        config,
        source,
        uploader,
        bf16: vec![0; m],
        trits: vec![0; m],
        words: vec![0; m / TRITS_PER_WORD],
        resident: 0,
        uploads: 0,
    };

    let mut layers = Vec::with_capacity(config.layers);
    let mut all_scales = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        let input_norm = s.norm(&format!("{p}.input_layernorm.weight"), h)?;
        let attn_sub_norm = s.norm(
            &format!("{p}.self_attn.attn_sub_norm.weight"),
            config.q_dim(),
        )?;
        let post_norm = s.norm(&format!("{p}.post_attention_layernorm.weight"), h)?;
        let ffn_sub_norm = s.norm(&format!("{p}.mlp.ffn_sub_norm.weight"), i)?;
        let (qkv, sq) = s.matrix(layer, TernTensor::Qkv)?;
        let (o, so) = s.matrix(layer, TernTensor::O)?;
        let (gate_up, sg) = s.matrix(layer, TernTensor::GateUp)?;
        let (down, sd) = s.matrix(layer, TernTensor::Down)?;
        all_scales.push([sq[0], sq[1], sq[2], so[0], sg[0], sg[1], sd[0]]);
        layers.push(BitNetLayer {
            input_norm,
            attn_sub_norm,
            post_norm,
            ffn_sub_norm,
            qkv,
            o,
            gate_up,
            down,
        });
        progress(&format!("layer {}/{} packed", layer + 1, config.layers));
    }
    let final_norm = s.norm("model.norm.weight", h)?;
    let (mut resident, uploads) = (s.resident, s.uploads + 1);
    drop(s);

    // The 8-bit word table from the embedding the host already holds.
    let (n, b) = (config.vocab, FP8_BLOCK);
    let mut table = vec![0u8; n * h];
    let mut table_scales = vec![0u16; (n / b) * (h / b)];
    build_fp8_table(&embed_bf16, n, h, &mut table, &mut table_scales)?;
    let table_fp8 = uploader
        .upload_table_fp8(n, h, &table, &table_scales)
        .map_err(|e| format!("word table [{n}][{h}] upload: {e}"))?;
    if table_fp8.n() != n || table_fp8.k() != h {
        return Err(format!(
            "word table upload returned [{}][{}]",
            table_fp8.n(),
            table_fp8.k()
        ));
    }
    resident += config.table_bytes();
    drop((table, table_scales));
    progress("word table packed");

    let weights = BitNetOmegaWeights {
        config: config.clone(),
        embed_bf16,
        layers,
        final_norm,
        table_fp8,
    };
    let report = BitNetLoadReport {
        resident_bytes: resident,
        host_bytes: weights.host_bytes(),
        transient_bytes: bitnet_loader_transient_bytes(config),
        uploads,
        scales: all_scales,
        peak_rss_before_bytes: peak_before,
        peak_rss_bytes: peak_rss_bytes(),
        seconds: t0.elapsed().as_secs_f64(),
    };
    Ok((weights, report))
}

/// The upload rule of the word table: omega's FP8 check (lengths, no e4m3 NaN code, finite
/// scales). Re-exported for uploaders.
pub fn check_table_upload(n: usize, k: usize, w: &[u8], scales: &[u16]) -> Result<(), String> {
    check_fp8_upload(n, k, w, scales)
}

/// sha256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let got = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if got == 0 {
            break;
        }
        h.update(&buf[..got]);
    }
    Ok(h.finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>())
}
