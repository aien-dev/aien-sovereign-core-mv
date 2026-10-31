// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Qwen3 dense (Qwen3-8B family, bf16) on AIEN's own GPU engine (Omega), OWN-8B Stage 1.
//!
//! Design: `~/handoffs/2026-10-15-own8b-design.md` (section 2.2, stage table row 1). This stage
//! is host only: no launch, no chip. It is the dense sibling of `qwen3_moe_omega.rs`, which it
//! reuses (shard reader, peak RSS probe, matmul limits) but does not edit.
//!
//! - [`Qwen3DenseConfig`]: the shape, read from `config.json` through the strict Qwen3 reader of
//!   `model_dir.rs`, then held to what the dense Omega engine can run: model_type `qwen3` (the
//!   MoE family is refused by name), tied or untied embeddings, plain rope, head_dim 64 or 128, every
//!   projection inside the bf16 streaming GEMV rule (`k % 256 == 0`, `256 <= k <= 16384`) and
//!   inside the resident matmul limits used for prefill (`K <= 16384`, `N <= 262144`).
//! - The streaming loader: no f32 image of the model. Every matrix is read from the shards as raw
//!   bf16 words into one reused host buffer, in the checkpoint's own `[n][k]` row layout (what the
//!   bf16 GEMV takes, no transpose), and handed to a [`DenseUploader`]. q, k and v are read back to
//!   back, so the fused `[q_dim + 2*kv_dim][hidden]` tensor is the three checkpoint tensors stacked
//!   in that order with no copy; gate and up likewise give `[2*intermediate][hidden]`, gate rows
//!   first. The host keeps the bf16 embedding (widened per row at lookup; the design's section
//!   2.2 says f32, which would add 1.24 GB at the 8B for the same values, so bf16 as in the 30B
//!   engine and the design's section 0) and the f32 norms. Peak
//!   host memory is the embedding plus one buffer the size of the largest matrix (the lm_head):
//!   [`dense_loader_transient_bytes`].
//!
//! Stage 1b (FP8, design revision 2026-10-15): the same loader reads Qwen's FP8 checkpoint
//! (`Qwen/Qwen3-8B-FP8`). [`DenseWeightFormat`] comes from `config.json`'s `quantization_config`
//! (absent: bf16; exactly fp8 / e4m3 / block [128, 128] / dynamic: FP8; anything else refused by
//! name). Per projection the safetensors header decides: an `F8_E4M3` weight must have its bf16
//! `weight_scale_inv` grid `[n/128][k/128]` beside it, and a bf16 projection in an FP8 checkpoint
//! is refused (no mixed loads); lm_head, embedding and norms stay bf16. The e4m3 bytes and the
//! scale rows of q|k|v and of gate|up are stacked the same way as the bf16 rows and handed to
//! [`DenseUploader::upload_fp8`]. [`HostFp8Gemv`] is the host reference
//! (`y[r] = sum_c e4m3(w[r][c]) * x[c] * scale[r/128][c/128]` in f32, the chip's formula).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::model_dir::model_config_from_hf_json;
use crate::qwen3_moe::{bf16_to_f32, fp8_table, FP8_BLOCK};
use crate::qwen3_moe_omega::{
    dequant_fp8_sources_to_bf16_cols, peak_rss_bytes, Fp8Source, KvGeometry, ShardSource,
    TensorSource, KV_BLOCK_SIZE, MAX_RESIDENT_K, MAX_RESIDENT_N,
};

// The loader reads little-endian bf16 words in place (see `read_bf16`).
/// OWN-8B stage 2: the engine (decode, prefill, generation) over the loaded weights.
#[path = "qwen3_dense_omega_engine.rs"]
pub mod engine;

const _: () = assert!(
    cfg!(target_endian = "little"),
    "the dense loader needs a little-endian host"
);

/// head_dim values omega's paged attention takes (`omega_gpu_attention_api.h`, `TOO_LARGE`).
pub const ATTN_HEAD_DIMS: [usize; 2] = [64, 128];
/// Most query heads one paged attention launch takes (`OMEGA_GPU_ATTN_MAX_Q_HEADS`).
pub const ATTN_MAX_Q_HEADS: usize = 64;

// ---------------------------------------------------------------------------
// Model shape, read from the checkpoint.
// ---------------------------------------------------------------------------

/// The five matrices the dense engine streams, each `[n][k]` bf16 as the checkpoint stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseTensor {
    /// `q_proj | k_proj | v_proj` rows: `[q_dim + 2*kv_dim][hidden]`.
    Qkv,
    /// `o_proj`: `[hidden][q_dim]`.
    O,
    /// `gate_proj | up_proj` rows: `[2*intermediate][hidden]`.
    GateUp,
    /// `down_proj`: `[hidden][intermediate]`.
    Down,
    /// `lm_head`: `[vocab][hidden]`.
    LmHead,
}

impl DenseTensor {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qkv => "q|k|v",
            Self::O => "o",
            Self::GateUp => "gate|up",
            Self::Down => "down",
            Self::LmHead => "lm_head",
        }
    }
}

/// How the checkpoint stores the four per-layer projections (lm_head, embedding and norms are
/// bf16 in both).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseWeightFormat {
    /// No `quantization_config`: every tensor bf16.
    #[default]
    Bf16,
    /// Qwen's FP8 recipe: e4m3 weights `[n][k]` with a bf16 `weight_scale_inv` per 128x128
    /// block, `[n/128][k/128]`; activations quantised dynamically (not by this loader).
    Fp8Block128,
}

impl DenseWeightFormat {
    /// The format named by `config.json`'s `quantization_config` (absent or null: bf16). Only the
    /// exact FP8 block-128 recipe is taken; another method, format, block size or activation
    /// scheme, an unknown key, or a `modules_to_not_convert` entry that could match a projection is refused
    /// by name.
    pub fn from_config(raw: &serde_json::Value) -> Result<Self, String> {
        let Some(q) = raw.get("quantization_config").filter(|q| !q.is_null()) else {
            return Ok(Self::Bf16);
        };
        let refuse = |what: String| Err(format!("config refused: quantization_config {what}"));
        let shown =
            |v: Option<&serde_json::Value>| v.map_or("absent".to_string(), |v| v.to_string());
        let Some(obj) = q.as_object() else {
            return refuse(format!("{q} is not an object"));
        };
        // The method first, so an int4 / GPTQ / AWQ config is refused for what it is.
        match obj.get("quant_method") {
            Some(m) if m == "fp8" => {}
            other => {
                return refuse(format!(
                    "quant_method {}: only \"fp8\" (Qwen's e4m3 block-128) is implemented",
                    shown(other)
                ))
            }
        }
        let want = [
            ("fmt", serde_json::json!("e4m3")),
            (
                "weight_block_size",
                serde_json::json!([FP8_BLOCK, FP8_BLOCK]),
            ),
            ("activation_scheme", serde_json::json!("dynamic")),
        ];
        for (key, value) in &want {
            match obj.get(*key) {
                Some(v) if v == value => {}
                other => return refuse(format!("{key} {}, expected {value}", shown(other))),
            }
        }
        for (key, value) in obj {
            match key.as_str() {
                "quant_method" | "fmt" | "weight_block_size" | "activation_scheme" => {}
                "modules_to_not_convert" => {
                    let names = value.as_array().ok_or_else(|| {
                        format!("config refused: quantization_config modules_to_not_convert {value} is not a list")
                    })?;
                    // Hugging Face skips a module when an entry is a substring of its name, so
                    // an entry is safe only if it cannot be a substring of any projection name
                    // (`model.layers.N.{self_attn,mlp}.*_proj`). Those names contain none of
                    // these words, so an entry containing one can only match a bf16 tensor.
                    let bf16_only = |s: &str| {
                        ["lm_head", "norm", "embed_tokens"]
                            .iter()
                            .any(|w| s.contains(w))
                    };
                    if let Some(p) = names
                        .iter()
                        .find(|n| n.as_str().is_none_or(|s| !bf16_only(s)))
                    {
                        return refuse(format!(
                            "modules_to_not_convert names {p}: it could match a projection, and every projection of this engine is FP8 in an FP8 checkpoint"
                        ));
                    }
                }
                other => return refuse(format!("field {other:?} is not implemented")),
            }
        }
        Ok(Self::Fp8Block128)
    }
}

/// The dense Qwen3 shape, read from `config.json` and validated against what the dense Omega
/// engine can run.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3DenseConfig {
    /// How the projections are stored (`quantization_config`).
    pub format: DenseWeightFormat,
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
    /// Ids generation stops on (`generation_config.json` eos list, else `config.json` eos).
    pub eos_token_ids: Vec<u32>,
    /// `tie_word_embeddings`: the lm_head is the embedding matrix (Qwen3-4B, 0.6B, 1.7B). The
    /// loader then uploads the host embedding as the lm_head and reads no `lm_head.weight`.
    pub tie_word_embeddings: bool,
}

impl Qwen3DenseConfig {
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
        match raw.get("model_type").and_then(|m| m.as_str()) {
            Some("qwen3") => {}
            Some("qwen3_moe") => {
                return Err(
                    "config refused: model_type \"qwen3_moe\" is the mixture-of-experts \
                            family; this loader is for dense \"qwen3\" (the MoE engine is \
                            qwen3_moe_omega)"
                        .into(),
                )
            }
            other => {
                return Err(format!(
                    "config refused: model_type {other:?}, this loader is for dense \"qwen3\""
                ))
            }
        }
        // A checkpoint that is not bf16 is refused here, not at its first tensor read. An FP8
        // checkpoint says bfloat16 too: it names the type of its unquantised tensors.
        for key in ["torch_dtype", "dtype"] {
            if let Some(d) = raw.get(key).filter(|d| !d.is_null()) {
                if d.as_str() != Some("bfloat16") {
                    return Err(format!(
                        "config refused: {key} {d}, this loader streams bfloat16 weights"
                    ));
                }
            }
        }
        let format = DenseWeightFormat::from_config(&raw)?;
        // The strict Qwen3ForCausalLM reader: unknown keys, sliding window, biases, activation,
        // rope type, head_dim presence and the attention geometry are refused there by name. It
        // does not know `quantization_config` (the bf16 engines must keep refusing an FP8
        // checkpoint), so the key, read above, is taken out of what it sees.
        let stripped;
        let config_json = if raw.get("quantization_config").is_some() {
            let mut rest = raw.clone();
            if let Some(obj) = rest.as_object_mut() {
                obj.remove("quantization_config");
            }
            stripped = rest.to_string();
            stripped.as_str()
        } else {
            config_json
        };
        let mc = model_config_from_hf_json("qwen3-dense", config_json, generation_json)
            .map_err(|e| format!("config refused: {e}"))?;
        if !mc.qk_norm {
            return Err(
                "config refused: architectures must be [\"Qwen3ForCausalLM\"] (q/k norm)".into(),
            );
        }
        if mc.rope_scaling.is_some() {
            return Err("config refused: rope_scaling is not implemented (plain rope only)".into());
        }
        let cfg = Self {
            format,
            layers: mc.num_layers,
            hidden: mc.hidden_dim,
            q_heads: mc.num_heads,
            kv_heads: mc.num_kv_heads,
            head_dim: mc.head_dim,
            intermediate: mc.intermediate_dim,
            vocab: mc.vocab_size,
            rms_eps: mc.rms_norm_eps,
            rope_theta: mc.rope_theta,
            max_position: mc.max_sequence_length,
            eos_token_ids: mc.eos_token_ids,
            tie_word_embeddings: mc.tie_word_embeddings,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// What the dense engine can run: omega's paged attention, the bf16 streaming GEMV for every
    /// projection (decode) and the resident matmul limits (prefill, `[k][n]`).
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: String| Err(format!("config refused: {what}"));
        if self.layers == 0
            || self.vocab == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || self.intermediate == 0
        {
            return bad("zero layers, vocab, heads or intermediate".into());
        }
        if !ATTN_HEAD_DIMS.contains(&self.head_dim) {
            return bad(format!(
                "head_dim {} (omega paged attention takes {ATTN_HEAD_DIMS:?})",
                self.head_dim
            ));
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) || self.q_heads > ATTN_MAX_Q_HEADS {
            return bad(format!(
                "{} query heads over {} kv heads (need a whole GQA group and <= {ATTN_MAX_Q_HEADS} query heads)",
                self.q_heads, self.kv_heads
            ));
        }
        if self.format == DenseWeightFormat::Fp8Block128 {
            // Each stacked part (q, k, v; gate, up) must start on a scale row, and every k on a
            // scale column.
            for (what, v) in [
                ("hidden_size", self.hidden),
                ("intermediate_size", self.intermediate),
                ("q_dim", self.q_dim()),
                ("kv_dim", self.kv_dim()),
            ] {
                if !v.is_multiple_of(FP8_BLOCK) {
                    return bad(format!(
                        "{what} {v} is not a multiple of the FP8 block {FP8_BLOCK}"
                    ));
                }
            }
        }
        for (what, n, k) in self.matrices() {
            if self.is_fp8(what) {
                if let Err(e) = aien_omega_fp8::check_shape(1, n, k) {
                    return bad(format!("{} [{n}][{k}]: {e}", what.name()));
                }
                continue;
            }
            if let Err(e) = aien_omega_bf16::check_shape(n, k) {
                return bad(format!("{} [{n}][{k}]: {e}", what.name()));
            }
            if k > MAX_RESIDENT_K || n > MAX_RESIDENT_N {
                return bad(format!(
                    "{} [{n}][{k}] exceeds the resident matmul limit K <= {MAX_RESIDENT_K}, N <= {MAX_RESIDENT_N}",
                    what.name()
                ));
            }
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.q_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// Rows of the fused q|k|v tensor.
    pub fn qkv_n(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    /// Rows of the fused gate|up tensor.
    pub fn gate_up_n(&self) -> usize {
        2 * self.intermediate
    }

    /// Every streamed matrix as `(what, n, k)`, `[n][k]` row layout.
    pub fn matrices(&self) -> [(DenseTensor, usize, usize); 5] {
        let h = self.hidden;
        [
            (DenseTensor::Qkv, self.qkv_n(), h),
            (DenseTensor::O, h, self.q_dim()),
            (DenseTensor::GateUp, self.gate_up_n(), h),
            (DenseTensor::Down, h, self.intermediate),
            (DenseTensor::LmHead, self.vocab, h),
        ]
    }

    /// bf16 words of the largest streamed matrix (the loader's one host buffer).
    pub fn max_matrix_words(&self) -> usize {
        self.matrices()
            .iter()
            .map(|&(_, n, k)| n * k)
            .max()
            .unwrap_or(0)
    }

    /// Whether `what` is stored as FP8 (the four projections of an FP8 checkpoint; the lm_head
    /// is bf16 in both formats).
    pub fn is_fp8(&self, what: DenseTensor) -> bool {
        self.format == DenseWeightFormat::Fp8Block128 && what != DenseTensor::LmHead
    }

    /// Bytes of one streamed matrix as uploaded: `2nk` bf16, or `nk` e4m3 plus the bf16 scale
    /// grid `[n/128][k/128]`.
    pub fn matrix_bytes(&self, what: DenseTensor, n: usize, k: usize) -> usize {
        if self.is_fp8(what) {
            n * k + 2 * (n / FP8_BLOCK) * (k / FP8_BLOCK)
        } else {
            2 * n * k
        }
    }

    /// Bytes of one layer's four streamed tensors.
    pub fn layer_resident_bytes(&self) -> usize {
        self.matrices()
            .iter()
            .filter(|m| m.0 != DenseTensor::LmHead)
            .map(|&(what, n, k)| self.matrix_bytes(what, n, k))
            .sum()
    }

    /// Bytes handed to the uploader for the whole model (layers plus lm_head). For FP8 this
    /// counts the bf16 scale grids as the checkpoint stores them; a device that widens the
    /// scales to f32 holds slightly more.
    pub fn model_resident_bytes(&self) -> usize {
        self.layers * self.layer_resident_bytes()
            + self.matrix_bytes(DenseTensor::LmHead, self.vocab, self.hidden)
    }

    /// Words of the loader's one reused buffer: bf16, the largest matrix; FP8, the larger of the
    /// bf16 lm_head and the largest projection's e4m3 bytes.
    pub fn loader_buffer_words(&self) -> usize {
        self.matrices()
            .iter()
            // A tied lm_head is uploaded from the host embedding, not read through the buffer.
            .filter(|m| !(self.tie_word_embeddings && m.0 == DenseTensor::LmHead))
            .map(|&(what, n, k)| {
                if self.is_fp8(what) {
                    (n * k).div_ceil(2)
                } else {
                    n * k
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// bf16 words of the loader's scale buffer: the largest FP8 scale grid (0 for bf16).
    pub fn loader_scale_words(&self) -> usize {
        self.matrices()
            .iter()
            .filter(|m| self.is_fp8(m.0))
            .map(|&(_, n, k)| (n / FP8_BLOCK) * (k / FP8_BLOCK))
            .max()
            .unwrap_or(0)
    }

    /// Bytes of the bf16 embedding the host keeps.
    pub fn embed_host_bytes(&self) -> usize {
        2 * self.vocab * self.hidden
    }
}

/// The 8B's KV shape (ADR 0004 section 2.1): the dense engine will build its cache from this
/// (step 6).
impl From<&Qwen3DenseConfig> for KvGeometry {
    fn from(c: &Qwen3DenseConfig) -> Self {
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

/// Largest host buffer the loader holds beyond the model's own host bytes: one buffer reused for
/// every read (norms included), the size of the largest matrix as stored, plus, for FP8, the
/// largest scale grid. At the 8B the lm_head (bf16 in both formats, 1.24 GB) sets it.
pub fn dense_loader_transient_bytes(config: &Qwen3DenseConfig) -> usize {
    2 * (config.loader_buffer_words() + config.loader_scale_words())
}

// ---------------------------------------------------------------------------
// The device side, behind a trait so the loader runs without a chip.
// ---------------------------------------------------------------------------

/// One resident `[n][k]` bf16 matrix: `y[r] = sum_c W[r][c] * x[c]` in f32. Shared across
/// threads: an implementation over omega's `Bf16Tensor` (not `Sync`) holds it behind a lock, as
/// `OmegaLmHead` does for the 30B.
pub trait DenseGemv: Send + Sync + std::fmt::Debug {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Returns the device time in ns (0 when not measured).
    fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String>;
    /// `m` rows: `x` is `[m][k]`, `y` is `[m][n]`. Returns (device ns, launches). The default
    /// runs the GEMV once per row; a multi-row kernel (omega's FP8 GEMM) must give the same
    /// numbers bit for bit, so prefill and decode share one oracle.
    fn rows_f32(&self, m: usize, x: &[f32], y: &mut [f32]) -> Result<(u64, u64), String> {
        engine::rows_by_gemv(self, m, x, y)
    }
}

/// Makes `[n][k]` bf16 matrices resident, byte for byte as given (the checkpoint row layout).
/// The real one (omega's `Bf16Tensor`, NaN/Inf refused at upload) comes with the engine stage.
pub trait DenseUploader {
    /// `layer` is `None` for the lm_head.
    fn upload_bf16(
        &self,
        layer: Option<usize>,
        what: DenseTensor,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String>;

    /// An `[n][k]` FP8 projection as the checkpoint stores it: `w` is `n * k` e4m3 bytes, row
    /// major, `scales` the `[n/128][k/128]` bf16 `weight_scale_inv` grid (fused tensors: the
    /// parts' rows and scale rows stacked in the same order). Uploaders without FP8 refuse, as
    /// `ModelUploader::upload_fp8` does for the 30B.
    fn upload_fp8(
        &self,
        layer: Option<usize>,
        what: DenseTensor,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        let _ = (layer, what, n, k, w, scales);
        Err("this uploader has no FP8 path".into())
    }
}

/// The FP8 upload rule (omega's `omega_gpu_fp8_tensor_upload`, checked on the host): lengths for
/// `[n][k]` with block-128 scales, no e4m3 NaN code (0x7f, 0xff) in the weights and no NaN or
/// infinite scale. The first offender is named.
pub fn check_fp8_upload(n: usize, k: usize, w: &[u8], scales: &[u16]) -> Result<(), String> {
    if !n.is_multiple_of(FP8_BLOCK) || !k.is_multiple_of(FP8_BLOCK) || n == 0 || k == 0 {
        return Err(format!(
            "fp8 [{n}][{k}]: n and k must be positive multiples of {FP8_BLOCK}"
        ));
    }
    let grid = (n / FP8_BLOCK) * (k / FP8_BLOCK);
    if w.len() != n * k || scales.len() != grid {
        return Err(format!(
            "fp8 [{n}][{k}]: {} weight bytes and {} scales, expected {} and {grid}",
            w.len(),
            scales.len(),
            n * k
        ));
    }
    if let Some(at) = w.iter().position(|&b| b & 0x7f == 0x7f) {
        return Err(format!(
            "weight byte {at} is an e4m3 NaN code (refused at upload)"
        ));
    }
    // bf16 exponent all ones = Inf or NaN.
    if let Some(at) = scales.iter().position(|&s| s & 0x7f80 == 0x7f80) {
        return Err(format!("scale {at} is NaN or infinite (refused at upload)"));
    }
    Ok(())
}

/// The host reference of one FP8 projection (design section 2.2 item 4, the per-layer oracle of
/// section 2.4): the checkpoint's e4m3 bytes and bf16 scales in the chip's formula,
/// `y[r] = sum_c e4m3(w[r][c]) * x[c] * scale[r/128][c/128]`, f32, columns in order, each
/// product scaled. A kernel that sums a block before scaling rounds differently, so a chip
/// comparison against this reference needs a tolerance, not bit equality. The
/// [`HostFp8Gemv::dequantised_bf16`] variant first rounds every dequantised weight to bf16
/// (`dequant_fp8_sources_to_bf16_cols`, the 30B engine's bf16 path) and then multiplies as the
/// bf16 GEMV does.
#[derive(Debug)]
pub struct HostFp8Gemv {
    n: usize,
    k: usize,
    w: Vec<u8>,
    scales: Vec<u16>,
    /// `[n][k]` bf16 words of the dequantised weights, for the bf16 variant.
    dequantised: Option<Vec<u16>>,
}

impl HostFp8Gemv {
    /// Holds `w` and `scales` as given, after [`check_fp8_upload`].
    pub fn new(n: usize, k: usize, w: Vec<u8>, scales: Vec<u16>) -> Result<Self, String> {
        check_fp8_upload(n, k, &w, &scales)?;
        Ok(Self {
            n,
            k,
            w,
            scales,
            dequantised: None,
        })
    }

    /// The same projection, dequantised once to bf16 (`e4m3(w) * scale` rounded to nearest even).
    pub fn dequantised_bf16(
        n: usize,
        k: usize,
        w: Vec<u8>,
        scales: Vec<u16>,
    ) -> Result<Self, String> {
        let mut m = Self::new(n, k, w, scales)?;
        // The 30B helper writes the transposed `[k][n]` layout; read it back by rows.
        let mut cols = vec![0u16; k * n];
        dequant_fp8_sources_to_bf16_cols(
            &[Fp8Source {
                weights: &m.w,
                scales: &m.scales,
                rows: n,
                col0: 0,
            }],
            k,
            &mut cols,
            n,
        )?;
        let mut rows = vec![0u16; n * k];
        for (c, col) in cols.chunks_exact(n).enumerate() {
            for (r, &v) in col.iter().enumerate() {
                rows[r * k + c] = v;
            }
        }
        m.dequantised = Some(rows);
        Ok(m)
    }

    /// The e4m3 bytes, `[n][k]`.
    pub fn weights(&self) -> &[u8] {
        &self.w
    }

    /// The bf16 scale grid, `[n/128][k/128]`.
    pub fn scales(&self) -> &[u16] {
        &self.scales
    }
}

impl DenseGemv for HostFp8Gemv {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String> {
        let (n, k) = (self.n, self.k);
        if x.len() != k || y.len() != n {
            return Err(format!(
                "fp8 gemv [{n}][{k}] given x {} y {}",
                x.len(),
                y.len()
            ));
        }
        // Rows in parallel; each row sums its columns in order, so the result does not depend on
        // the thread count.
        use rayon::prelude::*;
        if let Some(d) = &self.dequantised {
            y.par_iter_mut().enumerate().for_each(|(r, out)| {
                let row = &d[r * k..(r + 1) * k];
                *out = row.iter().zip(x).map(|(&w, &v)| bf16_to_f32(w) * v).sum();
            });
            return Ok(0);
        }
        let lut = fp8_table();
        let kb = k / FP8_BLOCK;
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            let row = &self.w[r * k..(r + 1) * k];
            let srow = &self.scales[(r / FP8_BLOCK) * kb..(r / FP8_BLOCK + 1) * kb];
            *out = row
                .iter()
                .zip(x)
                .enumerate()
                .map(|(c, (&w, &v))| lut[w as usize] * v * bf16_to_f32(srow[c / FP8_BLOCK]))
                .sum();
        });
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// The loaded model.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DenseLayer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub qkv: Arc<dyn DenseGemv>,
    pub o: Arc<dyn DenseGemv>,
    pub gate_up: Arc<dyn DenseGemv>,
    pub down: Arc<dyn DenseGemv>,
}

/// The whole model: streamed matrices on the device, the bf16 embedding and f32 norms on the host.
#[derive(Debug)]
pub struct Qwen3DenseOmegaWeights {
    pub config: Qwen3DenseConfig,
    /// `[vocab][hidden]` bf16 bits; a row is widened to f32 at lookup.
    pub embed_bf16: Vec<u16>,
    pub layers: Vec<DenseLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Arc<dyn DenseGemv>,
}

impl Qwen3DenseOmegaWeights {
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
                    4 * (l.input_norm.len() + l.post_norm.len() + l.q_norm.len() + l.k_norm.len())
                })
                .sum::<usize>()
    }
}

/// What a load measured.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLoadReport {
    /// Bytes handed to the uploader: bf16 words, or e4m3 bytes plus their bf16 scales.
    pub resident_bytes: usize,
    pub host_bytes: usize,
    /// The loader's reused buffers, bytes ([`dense_loader_transient_bytes`]).
    pub transient_bytes: usize,
    pub uploads: usize,
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
    // The source writes little-endian bf16 words, this host's byte order (asserted at compile
    // time).
    source.read_into(name, "BF16", shape, words_as_bytes_mut(dst))
}

/// A norm vector, read through the front of the loader's buffer and widened to f32.
fn read_norm(
    source: &mut dyn TensorSource,
    name: &str,
    len: usize,
    buf: &mut [u16],
) -> Result<Vec<f32>, String> {
    read_bf16(source, name, &[len], &mut buf[..len])?;
    Ok(buf[..len].iter().map(|&b| bf16_to_f32(b)).collect())
}

/// Reads one FP8 projection `name` (`[rows][k]` e4m3) into `w` and its `weight_scale_inv`
/// (`[rows/128][k/128]` bf16) into `scales`. The checkpoint header decides, not the name: the
/// weight must be `F8_E4M3` with a scale grid beside it; a bf16 projection in an FP8 checkpoint
/// and an FP8 weight without its scales are refused by name.
fn read_fp8(
    source: &mut dyn TensorSource,
    name: &str,
    rows: usize,
    k: usize,
    w: &mut [u8],
    scales: &mut [u16],
) -> Result<(), String> {
    let scale_name = format!("{name}_scale_inv");
    let dtype = source
        .dtype_of(name)?
        .ok_or_else(|| format!("{name}: not in the checkpoint"))?;
    let has_scales = source.dtype_of(&scale_name)?.is_some();
    match (dtype.as_str(), has_scales) {
        ("F8_E4M3", true) => {}
        ("F8_E4M3", false) => {
            return Err(format!(
            "{name}: F8_E4M3 without {scale_name}: refused (an FP8 weight needs its block scales)"
        ))
        }
        ("BF16", _) => {
            return Err(format!(
                "{name}: BF16 projection in an FP8 checkpoint: refused (no mixed loads)"
            ))
        }
        (other, _) => {
            return Err(format!(
                "{name}: dtype {other}, an FP8 checkpoint's projection must be F8_E4M3"
            ))
        }
    }
    source.read_into(name, "F8_E4M3", &[rows, k], w)?;
    // dtype BF16 and shape [rows/128][k/128] checked by the source.
    let grid = [rows / FP8_BLOCK, k / FP8_BLOCK];
    if grid[0] * grid[1] != scales.len() {
        return Err(format!(
            "{scale_name}: grid {grid:?} does not fill {} scales",
            scales.len()
        ));
    }
    source.read_into(&scale_name, "BF16", &grid, words_as_bytes_mut(scales))
}

/// The streaming loader's state: the source, the uploader, its one reused buffer (plus the
/// scale buffer for FP8) and the running totals.
struct Stream<'a> {
    config: &'a Qwen3DenseConfig,
    source: &'a mut dyn TensorSource,
    uploader: &'a dyn DenseUploader,
    buf: Vec<u16>,
    scales: Vec<u16>,
    resident: usize,
    uploads: usize,
}

impl Stream<'_> {
    fn norm(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        read_norm(self.source, name, len, &mut self.buf)
    }

    /// `parts` (checkpoint tensor name, rows), each `[rows][k]`, read back to back into the
    /// buffer, so the fused `[sum rows][k]` matrix is the parts stacked in order with no copy,
    /// and uploaded. FP8: the e4m3 bytes and the scale rows are stacked the same way.
    fn matrix(
        &mut self,
        layer: Option<usize>,
        what: DenseTensor,
        parts: &[(String, usize)],
        k: usize,
    ) -> Result<Arc<dyn DenseGemv>, String> {
        let n: usize = parts.iter().map(|p| p.1).sum();
        let at = layer.map_or("lm_head".to_string(), |l| format!("layer {l}"));
        let uploaded = if self.config.is_fp8(what) {
            let kb = k / FP8_BLOCK;
            let bytes = words_as_bytes_mut(&mut self.buf);
            let (mut off, mut s_off) = (0usize, 0usize);
            for (name, rows) in parts {
                let sl = (rows / FP8_BLOCK) * kb;
                read_fp8(
                    self.source,
                    name,
                    *rows,
                    k,
                    &mut bytes[off..off + rows * k],
                    &mut self.scales[s_off..s_off + sl],
                )?;
                off += rows * k;
                s_off += sl;
            }
            debug_assert_eq!(off, n * k);
            self.uploader
                .upload_fp8(layer, what, n, k, &bytes[..off], &self.scales[..s_off])
        } else {
            let mut off = 0usize;
            for (name, rows) in parts {
                read_bf16(
                    self.source,
                    name,
                    &[*rows, k],
                    &mut self.buf[off..off + rows * k],
                )?;
                off += rows * k;
            }
            debug_assert_eq!(off, n * k);
            self.uploader
                .upload_bf16(layer, what, n, k, &self.buf[..off])
        };
        let m = uploaded.map_err(|e| format!("{at} {} [{n}][{k}] upload: {e}", what.name()))?;
        if m.n() != n || m.k() != k {
            return Err(format!(
                "{at} {} upload returned [{}][{}], asked [{n}][{k}]",
                what.name(),
                m.n(),
                m.k()
            ));
        }
        self.resident += self.config.matrix_bytes(what, n, k);
        self.uploads += 1;
        Ok(m)
    }
}

/// Loads a checkpoint directory: `config.json` for the shape and format, the shards for the
/// weights.
pub fn load_qwen3_dense_omega(
    dir: &Path,
    uploader: &dyn DenseUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3DenseOmegaWeights, DenseLoadReport), String> {
    let config = Qwen3DenseConfig::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    load_qwen3_dense_omega_from(&config, &mut source, uploader, progress)
}

/// The streaming load from any [`TensorSource`].
pub fn load_qwen3_dense_omega_from(
    config: &Qwen3DenseConfig,
    source: &mut dyn TensorSource,
    uploader: &dyn DenseUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3DenseOmegaWeights, DenseLoadReport), String> {
    config.validate()?;
    let t0 = Instant::now();
    let peak_before = peak_rss_bytes();
    let (h, i, hd) = (config.hidden, config.intermediate, config.head_dim);
    let (qd, kvd) = (config.q_dim(), config.kv_dim());

    let mut embed_bf16 = vec![0u16; config.vocab * h];
    read_bf16(
        source,
        "model.embed_tokens.weight",
        &[config.vocab, h],
        &mut embed_bf16,
    )?;

    // The one transient: every matrix and norm is read through it, layer after layer.
    let mut s = Stream {
        config,
        source,
        uploader,
        buf: vec![0; config.loader_buffer_words()],
        scales: vec![0; config.loader_scale_words()],
        resident: 0,
        uploads: 0,
    };
    let transient_bytes = 2 * (s.buf.len() + s.scales.len());

    let mut layers = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        let input_norm = s.norm(&format!("{p}.input_layernorm.weight"), h)?;
        let post_norm = s.norm(&format!("{p}.post_attention_layernorm.weight"), h)?;
        let q_norm = s.norm(&format!("{p}.self_attn.q_norm.weight"), hd)?;
        let k_norm = s.norm(&format!("{p}.self_attn.k_norm.weight"), hd)?;
        let a = |n: &str| format!("{p}.self_attn.{n}.weight");
        let m = |n: &str| format!("{p}.mlp.{n}.weight");
        let l = Some(layer);
        let qkv = s.matrix(
            l,
            DenseTensor::Qkv,
            &[(a("q_proj"), qd), (a("k_proj"), kvd), (a("v_proj"), kvd)],
            h,
        )?;
        let o = s.matrix(l, DenseTensor::O, &[(a("o_proj"), h)], qd)?;
        // gate|up: gate rows first, then up rows.
        let gate_up = s.matrix(
            l,
            DenseTensor::GateUp,
            &[(m("gate_proj"), i), (m("up_proj"), i)],
            h,
        )?;
        let down = s.matrix(l, DenseTensor::Down, &[(m("down_proj"), h)], i)?;
        layers.push(DenseLayer {
            input_norm,
            post_norm,
            q_norm,
            k_norm,
            qkv,
            o,
            gate_up,
            down,
        });
        progress(&format!("layer {}/{} uploaded", layer + 1, config.layers));
    }

    let final_norm = s.norm("model.norm.weight", h)?;
    let lm_head = if config.tie_word_embeddings {
        // Tied: the lm_head is the embedding matrix (`[vocab][hidden]`, the GEMV's `[n][k]`),
        // uploaded from the host copy; the checkpoint carries no `lm_head.weight`.
        let m = s
            .uploader
            .upload_bf16(None, DenseTensor::LmHead, config.vocab, h, &embed_bf16)
            .map_err(|e| format!("lm_head (tied to embed_tokens) upload: {e}"))?;
        if m.n() != config.vocab || m.k() != h {
            return Err(format!(
                "lm_head (tied) upload returned [{}][{}], asked [{}][{h}]",
                m.n(),
                m.k(),
                config.vocab
            ));
        }
        s.resident += config.matrix_bytes(DenseTensor::LmHead, config.vocab, h);
        s.uploads += 1;
        m
    } else {
        s.matrix(
            None,
            DenseTensor::LmHead,
            &[("lm_head.weight".to_string(), config.vocab)],
            h,
        )?
    };
    progress("lm_head uploaded");
    let (resident, uploads) = (s.resident, s.uploads);
    drop(s);

    let weights = Qwen3DenseOmegaWeights {
        config: config.clone(),
        embed_bf16,
        layers,
        final_norm,
        lm_head,
    };
    let report = DenseLoadReport {
        resident_bytes: resident,
        host_bytes: weights.host_bytes(),
        transient_bytes,
        uploads,
        peak_rss_before_bytes: peak_before,
        peak_rss_bytes: peak_rss_bytes(),
        seconds: t0.elapsed().as_secs_f64(),
    };
    Ok((weights, report))
}

// ---------------------------------------------------------------------------
// The tiny host-test dense model and the recording uploader.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "host-test-engine"))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod test_support {
    //! A tiny dense Qwen3 checkpoint (in memory or written as two shards, bf16 or FP8) and a
    //! recording uploader with host GEMVs, so the loader runs end to end without a chip.
    use super::*;
    use crate::qwen3_moe::{f32_to_bf16, fp8_e4m3fn_to_f32};
    use crate::qwen3_moe_omega::test_support::{prand, MemorySource};
    use std::sync::Mutex;

    /// 2 layers, hidden 256, 4 query heads over 2 kv heads of 128 (q_dim 512 differs from hidden,
    /// so a transposed o_proj fails), intermediate 512, vocab 64. bf16.
    pub fn tiny_config() -> Qwen3DenseConfig {
        Qwen3DenseConfig {
            format: DenseWeightFormat::Bf16,
            layers: 2,
            hidden: 256,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 512,
            vocab: 64,
            rms_eps: 1e-6,
            rope_theta: 1e6,
            max_position: 4096,
            eos_token_ids: vec![3],
            tie_word_embeddings: false,
        }
    }

    /// The same shape stored as Qwen's FP8 (e4m3 projections with block-128 scales).
    pub fn tiny_fp8_config() -> Qwen3DenseConfig {
        Qwen3DenseConfig {
            format: DenseWeightFormat::Fp8Block128,
            ..tiny_config()
        }
    }

    /// The Hugging Face `config.json` text of `c` (the keys Qwen3-8B's carries, plus Qwen3-8B-FP8's
    /// `quantization_config` for an FP8 config).
    pub fn config_json(c: &Qwen3DenseConfig) -> String {
        let mut v = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": c.eos_token_ids,
            "head_dim": c.head_dim,
            "hidden_act": "silu",
            "hidden_size": c.hidden,
            "initializer_range": 0.02,
            "intermediate_size": c.intermediate,
            "max_position_embeddings": c.max_position,
            "max_window_layers": c.layers,
            "model_type": "qwen3",
            "num_attention_heads": c.q_heads,
            "num_hidden_layers": c.layers,
            "num_key_value_heads": c.kv_heads,
            "rms_norm_eps": c.rms_eps,
            "rope_scaling": null,
            "rope_theta": c.rope_theta,
            "sliding_window": null,
            "tie_word_embeddings": c.tie_word_embeddings,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0",
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": c.vocab
        });
        if c.format == DenseWeightFormat::Fp8Block128 {
            v["quantization_config"] = serde_json::json!({
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
            });
        }
        v.to_string()
    }

    /// What the tiny checkpoint's projections hold.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TinyWeights {
        /// bf16: small random values. FP8: random e4m3 codes with random block scales.
        Random,
        /// The bf16/FP8 twin: every projection value is an e4m3 value; the FP8 copy stores the
        /// codes with scales of exactly 1.0, the bf16 copy the same values as bf16 (exact, e4m3
        /// is a subset of bf16). Every other tensor is the same in both.
        Twin,
    }

    /// One tensor of the tiny checkpoint: name, safetensors dtype, shape, little-endian bytes.
    pub type TinyTensor = (String, &'static str, Vec<usize>, Vec<u8>);

    /// A finite e4m3 code from the pseudo-random stream: any sign and mantissa, exponent field
    /// 0..=4 (magnitudes up to 0.234), never a NaN code.
    fn e4m3_code(seed: u64, i: usize) -> u8 {
        let b = ((prand(seed, i) + 1.0) * 32768.0) as u32;
        let e = ((b >> 3) & 7) % 5;
        ((b & 0x80) | (e << 3) | (b & 7)) as u8
    }

    fn bf16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(|v| f32_to_bf16(v).to_le_bytes()).collect()
    }

    /// Every tensor of the tiny checkpoint of `c` with random weights, each tensor with its own
    /// seed so rows of different tensors differ, in checkpoint order.
    pub fn tensors(c: &Qwen3DenseConfig) -> Vec<TinyTensor> {
        tensors_with(c, TinyWeights::Random)
    }

    /// Every tensor of the tiny checkpoint of `c`, in checkpoint order (an FP8 projection is
    /// followed by its `weight_scale_inv`).
    pub fn tensors_with(c: &Qwen3DenseConfig, mode: TinyWeights) -> Vec<TinyTensor> {
        enum Kind {
            /// bf16 `bias + prand * scale`.
            Plain(f32, f32),
            /// A projection `[rows][cols]`.
            Proj,
        }
        let (h, i, hd) = (c.hidden, c.intermediate, c.head_dim);
        let mut spec: Vec<(String, Vec<usize>, Kind)> = vec![(
            "model.embed_tokens.weight".into(),
            vec![c.vocab, h],
            Kind::Plain(1.0, 0.0),
        )];
        for l in 0..c.layers {
            let p = format!("model.layers.{l}");
            let norm = Kind::Plain(0.1, 1.0);
            spec.push((format!("{p}.input_layernorm.weight"), vec![h], norm));
            spec.push((
                format!("{p}.self_attn.q_proj.weight"),
                vec![c.q_dim(), h],
                Kind::Proj,
            ));
            spec.push((
                format!("{p}.self_attn.k_proj.weight"),
                vec![c.kv_dim(), h],
                Kind::Proj,
            ));
            spec.push((
                format!("{p}.self_attn.v_proj.weight"),
                vec![c.kv_dim(), h],
                Kind::Proj,
            ));
            spec.push((
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, c.q_dim()],
                Kind::Proj,
            ));
            spec.push((
                format!("{p}.self_attn.q_norm.weight"),
                vec![hd],
                Kind::Plain(0.1, 1.0),
            ));
            spec.push((
                format!("{p}.self_attn.k_norm.weight"),
                vec![hd],
                Kind::Plain(0.1, 1.0),
            ));
            spec.push((
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Kind::Plain(0.1, 1.0),
            ));
            spec.push((format!("{p}.mlp.gate_proj.weight"), vec![i, h], Kind::Proj));
            spec.push((format!("{p}.mlp.up_proj.weight"), vec![i, h], Kind::Proj));
            spec.push((format!("{p}.mlp.down_proj.weight"), vec![h, i], Kind::Proj));
        }
        spec.push(("model.norm.weight".into(), vec![h], Kind::Plain(0.1, 1.0)));
        if !c.tie_word_embeddings {
            spec.push((
                "lm_head.weight".into(),
                vec![c.vocab, h],
                Kind::Plain(0.1, 0.0),
            ));
        }

        let fp8 = c.format == DenseWeightFormat::Fp8Block128;
        let mut out = Vec::new();
        for (index, (name, shape, kind)) in spec.into_iter().enumerate() {
            let seed = index as u64 + 1;
            let n: usize = shape.iter().product();
            let codes = || (0..n).map(move |j| e4m3_code(seed, j));
            match (kind, fp8, mode) {
                (Kind::Proj, false, TinyWeights::Twin) => {
                    let values = codes().map(fp8_e4m3fn_to_f32);
                    out.push((name, "BF16", shape, bf16_bytes(values)));
                }
                (Kind::Proj, true, mode) => {
                    let grid = vec![shape[0] / FP8_BLOCK, shape[1] / FP8_BLOCK];
                    let g = grid[0] * grid[1];
                    let scales = match mode {
                        TinyWeights::Twin => bf16_bytes((0..g).map(|_| 1.0)),
                        TinyWeights::Random => bf16_bytes(
                            (0..g).map(|j| 0.01 + 0.005 * (prand(seed ^ 0x5ca1e, j) + 1.0)),
                        ),
                    };
                    out.push((name.clone(), "F8_E4M3", shape, codes().collect()));
                    out.push((format!("{name}_scale_inv"), "BF16", grid, scales));
                }
                (kind, _, _) => {
                    let (scale, bias) = match kind {
                        Kind::Plain(s, b) => (s, b),
                        Kind::Proj => (0.05, 0.0),
                    };
                    let v = bf16_bytes((0..n).map(|j| bias + prand(seed, j) * scale));
                    out.push((name, "BF16", shape, v));
                }
            }
        }
        out
    }

    pub fn memory_source(c: &Qwen3DenseConfig) -> MemorySource {
        memory_source_of(tensors(c))
    }

    pub fn memory_source_of(tensors: Vec<TinyTensor>) -> MemorySource {
        let mut m = MemorySource::default();
        for (name, dtype, shape, bytes) in tensors {
            m.tensors.insert(name, (dtype.into(), shape, bytes));
        }
        m
    }

    /// Writes `config.json`, `generation_config.json`, two safetensors shards (layer 0 and the
    /// embedding in the first, the rest in the second) and `model.safetensors.index.json`.
    pub fn write_checkpoint(dir: &Path, c: &Qwen3DenseConfig) -> Result<(), String> {
        write_checkpoint_of(dir, c, tensors(c))
    }

    /// [`write_checkpoint`] with the given tensors.
    pub fn write_checkpoint_of(
        dir: &Path,
        c: &Qwen3DenseConfig,
        all: Vec<TinyTensor>,
    ) -> Result<(), String> {
        let io = |e: std::io::Error| format!("{}: {e}", dir.display());
        std::fs::create_dir_all(dir).map_err(io)?;
        std::fs::write(dir.join("config.json"), config_json(c)).map_err(io)?;
        std::fs::write(
            dir.join("generation_config.json"),
            serde_json::json!({ "eos_token_id": c.eos_token_ids }).to_string(),
        )
        .map_err(io)?;
        let split = all
            .iter()
            .position(|t| t.0.starts_with("model.layers.1."))
            .unwrap_or(all.len());
        let mut weight_map = serde_json::Map::new();
        for (index, part) in [&all[..split], &all[split..]].into_iter().enumerate() {
            let filename = format!("model-0000{}-of-00002.safetensors", index + 1);
            let mut header = serde_json::Map::new();
            let mut payload = Vec::new();
            for (name, dtype, shape, bytes) in part {
                header.insert(
                    name.clone(),
                    serde_json::json!({"dtype": dtype, "shape": shape,
                        "data_offsets": [payload.len(), payload.len() + bytes.len()]}),
                );
                payload.extend_from_slice(bytes);
                weight_map.insert(name.clone(), serde_json::json!(filename));
            }
            let header = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
            let mut file = (header.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(&header);
            file.extend_from_slice(&payload);
            std::fs::write(dir.join(&filename), file).map_err(io)?;
        }
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({ "metadata": {}, "weight_map": weight_map }).to_string(),
        )
        .map_err(io)
    }

    /// An `[n][k]` bf16 matrix on the host; GEMV in f32, sequential order.
    #[derive(Debug)]
    pub struct HostGemv {
        pub n: usize,
        pub k: usize,
        pub bits: Vec<u16>,
    }

    impl DenseGemv for HostGemv {
        fn n(&self) -> usize {
            self.n
        }
        fn k(&self) -> usize {
            self.k
        }
        fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String> {
            if x.len() != self.k || y.len() != self.n {
                return Err(format!(
                    "gemv [{}][{}] given x {} y {}",
                    self.n,
                    self.k,
                    x.len(),
                    y.len()
                ));
            }
            for (r, out) in y.iter_mut().enumerate() {
                let row = &self.bits[r * self.k..(r + 1) * self.k];
                *out = row.iter().zip(x).map(|(&w, &v)| bf16_to_f32(w) * v).sum();
            }
            Ok(0)
        }
    }

    /// What a record-only upload hands back: the shape, no weights.
    #[derive(Debug)]
    pub struct RecordedShape {
        pub n: usize,
        pub k: usize,
    }

    impl DenseGemv for RecordedShape {
        fn n(&self) -> usize {
            self.n
        }
        fn k(&self) -> usize {
            self.k
        }
        fn gemv_f32(&self, _: &[f32], _: &mut [f32]) -> Result<u64, String> {
            Err("record-only upload keeps no weights".into())
        }
    }

    /// A deliberately wrong uploader, for the negative controls.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum UploadMutant {
        /// Stores q|k|v with the k rows and the v rows exchanged (FP8: bytes and scale rows).
        SwapKv { q_rows: usize, kv_rows: usize },
        /// Stores gate|up with the gate rows and the up rows exchanged (FP8: bytes and scales).
        SwapGateUp { rows: usize },
        /// FP8 only: stores q|k|v with the k and v scale rows exchanged, the bytes right.
        SwapKvScales { q_rows: usize, kv_rows: usize },
        /// FP8 only: stores every scale grid transposed (`[k/128][n/128]` read as `[n/128][k/128]`).
        TransposeScales,
    }

    /// One upload as the recording fake saw it.
    #[derive(Debug)]
    pub struct UploadRecord {
        pub layer: Option<usize>,
        pub what: DenseTensor,
        pub n: usize,
        pub k: usize,
        /// Bytes handed over (FP8: weights plus scales).
        pub bytes: usize,
        /// The stored bf16 matrix (`keep` uploaders only).
        pub matrix: Option<Arc<HostGemv>>,
        /// The stored FP8 matrix (`keep` uploaders only).
        pub fp8: Option<Arc<HostFp8Gemv>>,
    }

    /// Records every upload; with `keep` it also stores the matrix for host GEMVs and the byte
    /// test. Like omega's `Bf16Tensor::upload` and `Fp8Tensor::upload` it refuses a wrong length,
    /// a NaN or infinite bf16 word or scale and an e4m3 NaN code, so every byte is looked at.
    #[derive(Debug, Default)]
    pub struct RecordingDenseUploader {
        pub keep: bool,
        pub mutant: Option<UploadMutant>,
        pub records: Mutex<Vec<UploadRecord>>,
    }

    impl RecordingDenseUploader {
        pub fn keeping() -> Self {
            Self {
                keep: true,
                ..Self::default()
            }
        }
        /// Shapes and byte counts only (the real-checkpoint run: no host copy of the model).
        pub fn record_only() -> Self {
            Self::default()
        }
        fn with(mutant: UploadMutant) -> Self {
            Self {
                keep: true,
                mutant: Some(mutant),
                ..Self::default()
            }
        }
        pub fn swapping_kv(c: &Qwen3DenseConfig) -> Self {
            Self::with(UploadMutant::SwapKv {
                q_rows: c.q_dim(),
                kv_rows: c.kv_dim(),
            })
        }
        pub fn swapping_gate_up(c: &Qwen3DenseConfig) -> Self {
            Self::with(UploadMutant::SwapGateUp {
                rows: c.intermediate,
            })
        }
        pub fn swapping_kv_scales(c: &Qwen3DenseConfig) -> Self {
            Self::with(UploadMutant::SwapKvScales {
                q_rows: c.q_dim(),
                kv_rows: c.kv_dim(),
            })
        }
        pub fn transposing_scales() -> Self {
            Self::with(UploadMutant::TransposeScales)
        }

        fn record(
            &self,
            layer: Option<usize>,
            what: DenseTensor,
            n: usize,
            k: usize,
            bytes: usize,
            matrix: Option<Arc<HostGemv>>,
            fp8: Option<Arc<HostFp8Gemv>>,
        ) {
            self.records.lock().unwrap().push(UploadRecord {
                layer,
                what,
                n,
                k,
                bytes,
                matrix,
                fp8,
            });
        }
    }

    /// Exchanges rows `a..a+rows` and `b..b+rows` of a row-major matrix of `width` columns
    /// (`a + rows <= b`).
    fn swap_rows<T>(m: &mut [T], width: usize, a: usize, b: usize, rows: usize) {
        let (lo, hi) = m.split_at_mut(b * width);
        lo[a * width..(a + rows) * width].swap_with_slice(&mut hi[..rows * width]);
    }

    impl DenseUploader for RecordingDenseUploader {
        fn upload_bf16(
            &self,
            layer: Option<usize>,
            what: DenseTensor,
            n: usize,
            k: usize,
            bits: &[u16],
        ) -> Result<Arc<dyn DenseGemv>, String> {
            if bits.len() != n * k {
                return Err(format!("{} words for [{n}][{k}]", bits.len()));
            }
            // bf16 exponent all ones = Inf or NaN.
            if let Some(at) = bits.iter().position(|&b| b & 0x7f80 == 0x7f80) {
                return Err(format!("word {at} is NaN or infinite (refused at upload)"));
            }
            let matrix = self.keep.then(|| {
                let mut stored = bits.to_vec();
                match (self.mutant, what) {
                    (Some(UploadMutant::SwapKv { q_rows, kv_rows }), DenseTensor::Qkv) => {
                        swap_rows(&mut stored, k, q_rows, q_rows + kv_rows, kv_rows);
                    }
                    (Some(UploadMutant::SwapGateUp { rows }), DenseTensor::GateUp) => {
                        swap_rows(&mut stored, k, 0, rows, rows);
                    }
                    _ => {}
                }
                Arc::new(HostGemv { n, k, bits: stored })
            });
            self.record(layer, what, n, k, bits.len() * 2, matrix.clone(), None);
            Ok(match matrix {
                Some(m) => m,
                None => Arc::new(RecordedShape { n, k }),
            })
        }

        fn upload_fp8(
            &self,
            layer: Option<usize>,
            what: DenseTensor,
            n: usize,
            k: usize,
            w: &[u8],
            scales: &[u16],
        ) -> Result<Arc<dyn DenseGemv>, String> {
            check_fp8_upload(n, k, w, scales)?;
            let fp8 = if self.keep {
                let (mut w, mut s) = (w.to_vec(), scales.to_vec());
                let (b, kb) = (FP8_BLOCK, k / FP8_BLOCK);
                match (self.mutant, what) {
                    (Some(UploadMutant::SwapKv { q_rows, kv_rows }), DenseTensor::Qkv) => {
                        swap_rows(&mut w, k, q_rows, q_rows + kv_rows, kv_rows);
                        swap_rows(&mut s, kb, q_rows / b, (q_rows + kv_rows) / b, kv_rows / b);
                    }
                    (Some(UploadMutant::SwapGateUp { rows }), DenseTensor::GateUp) => {
                        swap_rows(&mut w, k, 0, rows, rows);
                        swap_rows(&mut s, kb, 0, rows / b, rows / b);
                    }
                    (Some(UploadMutant::SwapKvScales { q_rows, kv_rows }), DenseTensor::Qkv) => {
                        swap_rows(&mut s, kb, q_rows / b, (q_rows + kv_rows) / b, kv_rows / b);
                    }
                    (Some(UploadMutant::TransposeScales), _) => {
                        let nb = n / b;
                        s = (0..nb * kb)
                            .map(|j| scales[(j % nb) * kb + j / nb])
                            .collect();
                    }
                    _ => {}
                }
                Some(Arc::new(HostFp8Gemv::new(n, k, w, s)?))
            } else {
                None
            };
            let bytes = w.len() + 2 * scales.len();
            self.record(layer, what, n, k, bytes, None, fp8.clone());
            Ok(match fp8 {
                Some(m) => m,
                None => Arc::new(RecordedShape { n, k }),
            })
        }
    }

    /// The byte test: the uploads must be, in order, each layer's q|k|v, o, gate|up, down, then
    /// the lm_head, with the right shapes and byte counts, and every stored row (FP8: every e4m3
    /// row and every scale row) must equal the checkpoint row it stands for (read again,
    /// independently, from `source`). The first wrong row is named.
    pub fn check_upload_bytes(
        c: &Qwen3DenseConfig,
        source: &mut dyn TensorSource,
        records: &[UploadRecord],
    ) -> Result<(), String> {
        let (h, i, qd, kvd) = (c.hidden, c.intermediate, c.q_dim(), c.kv_dim());
        // (tensor name, rows, cols) of each checkpoint part, in row order.
        type Part = (String, usize, usize);
        type Expected = (Option<usize>, DenseTensor, usize, usize, Vec<Part>);
        let mut expected: Vec<Expected> = Vec::new();
        for l in 0..c.layers {
            let p = format!("model.layers.{l}");
            let a = |n: &str| format!("{p}.self_attn.{n}.weight");
            let m = |n: &str| format!("{p}.mlp.{n}.weight");
            expected.push((
                Some(l),
                DenseTensor::Qkv,
                c.qkv_n(),
                h,
                vec![
                    (a("q_proj"), qd, h),
                    (a("k_proj"), kvd, h),
                    (a("v_proj"), kvd, h),
                ],
            ));
            expected.push((Some(l), DenseTensor::O, h, qd, vec![(a("o_proj"), h, qd)]));
            expected.push((
                Some(l),
                DenseTensor::GateUp,
                2 * i,
                h,
                vec![(m("gate_proj"), i, h), (m("up_proj"), i, h)],
            ));
            expected.push((
                Some(l),
                DenseTensor::Down,
                h,
                i,
                vec![(m("down_proj"), h, i)],
            ));
        }
        expected.push((
            None,
            DenseTensor::LmHead,
            c.vocab,
            h,
            vec![("lm_head.weight".into(), c.vocab, h)],
        ));
        if records.len() != expected.len() {
            return Err(format!(
                "{} uploads, expected {}",
                records.len(),
                expected.len()
            ));
        }
        for (rec, (layer, what, n, k, parts)) in records.iter().zip(expected) {
            let at = format!("upload {:?} {}", layer, what.name());
            let bytes = c.matrix_bytes(what, n, k);
            if (rec.layer, rec.what, rec.n, rec.k, rec.bytes) != (layer, what, n, k, bytes) {
                return Err(format!(
                    "{at}: got {:?} {} [{}][{}] {} bytes, expected [{n}][{k}] {bytes} bytes",
                    rec.layer,
                    rec.what.name(),
                    rec.n,
                    rec.k,
                    rec.bytes,
                ));
            }
            let mut row0 = 0usize;
            if c.is_fp8(what) {
                let m = rec
                    .fp8
                    .as_ref()
                    .ok_or_else(|| format!("{at}: no FP8 kept"))?;
                let kb = k / FP8_BLOCK;
                for (name, rows, cols) in parts {
                    let mut want = vec![0u8; rows * cols];
                    source.read_into(&name, "F8_E4M3", &[rows, cols], &mut want)?;
                    let got = &m.weights()[row0 * k..(row0 + rows) * k];
                    if let Some(r) = (0..rows)
                        .find(|&r| got[r * k..(r + 1) * k] != want[r * cols..(r + 1) * cols])
                    {
                        return Err(format!(
                            "{at}: row {} is not {name} row {r} (rows {row0}..{} must be {name})",
                            row0 + r,
                            row0 + rows
                        ));
                    }
                    let scale_name = format!("{name}_scale_inv");
                    let (srows, s0) = (rows / FP8_BLOCK, row0 / FP8_BLOCK);
                    let mut want = vec![0u16; srows * kb];
                    read_bf16(source, &scale_name, &[srows, kb], &mut want)?;
                    let got = &m.scales()[s0 * kb..(s0 + srows) * kb];
                    if let Some(r) = (0..srows)
                        .find(|&r| got[r * kb..(r + 1) * kb] != want[r * kb..(r + 1) * kb])
                    {
                        return Err(format!(
                            "{at}: scale row {} is not {scale_name} row {r} (scale rows {s0}..{} must be {scale_name})",
                            s0 + r,
                            s0 + srows
                        ));
                    }
                    row0 += rows;
                }
                continue;
            }
            let m = rec
                .matrix
                .as_ref()
                .ok_or_else(|| format!("{at}: not kept"))?;
            for (name, rows, cols) in parts {
                let mut want = vec![0u16; rows * cols];
                read_bf16(source, &name, &[rows, cols], &mut want)?;
                for r in 0..rows {
                    let got = &m.bits[(row0 + r) * k..(row0 + r + 1) * k];
                    if got != &want[r * cols..(r + 1) * cols] {
                        return Err(format!(
                            "{at}: row {} is not {name} row {r} (rows {row0}..{} must be {name})",
                            row0 + r,
                            row0 + rows
                        ));
                    }
                }
                row0 += rows;
            }
        }
        Ok(())
    }
}

/// Host-only test model (feature `host-test-engine`): the tiny dense checkpoint and the recording
/// uploader, so other crates can load a dense model without a chip. Never in a server build.
#[cfg(feature = "host-test-engine")]
pub mod host_test_model {
    use super::test_support::{memory_source, tiny_config, tiny_fp8_config, write_checkpoint};
    /// The recording uploader (`keeping()` for host GEMVs), to load the written checkpoint
    /// with [`load_qwen3_dense_omega`].
    pub use super::test_support::{HostGemv, RecordingDenseUploader, UploadRecord};
    use super::*;

    /// 2 layers, hidden 256, 4 query heads over 2 kv heads of 128 (q_dim 512 differs from hidden,
    /// so a transposed o_proj fails), intermediate 512, vocab 64.
    pub fn config() -> Qwen3DenseConfig {
        tiny_config()
    }

    /// The same shape stored as Qwen's FP8 (e4m3 projections, block-128 bf16 scales).
    pub fn config_fp8() -> Qwen3DenseConfig {
        tiny_fp8_config()
    }

    /// Writes the tiny checkpoint (config, generation config, two shards, index) into `dir`.
    pub fn write(dir: &Path) -> Result<(), String> {
        write_checkpoint(dir, &tiny_config())
    }

    /// Writes the tiny FP8 checkpoint (random e4m3 codes and scales) into `dir`.
    pub fn write_fp8(dir: &Path) -> Result<(), String> {
        write_checkpoint(dir, &tiny_fp8_config())
    }

    /// The tiny model loaded from memory, every matrix kept on the host for host GEMVs.
    pub fn load() -> Result<(Qwen3DenseOmegaWeights, DenseLoadReport), String> {
        let c = tiny_config();
        load_qwen3_dense_omega_from(
            &c,
            &mut memory_source(&c),
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::qwen3_moe::f32_to_bf16;
    use crate::qwen3_moe_omega::test_support::MemorySource;

    /// `~/models/qwen3-8b/config.json`, verbatim.
    const QWEN3_8B_CONFIG: &str = r#"{
  "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 12288,
  "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
  "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
  "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
  "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
}"#;
    const QWEN3_8B_GENERATION: &str = r#"{"eos_token_id": [151645, 151643]}"#;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("aien-own8b-{name}-{}", std::process::id()))
    }

    fn load_tiny(
        up: &RecordingDenseUploader,
    ) -> Result<(Qwen3DenseOmegaWeights, DenseLoadReport), String> {
        let c = tiny_config();
        load_qwen3_dense_omega_from(&c, &mut memory_source(&c), up, &mut |_| {})
    }

    /// The 8B config with one field replaced (`null` removes it).
    fn variant(key: &str, value: serde_json::Value) -> Result<Qwen3DenseConfig, String> {
        let mut v: serde_json::Value = serde_json::from_str(QWEN3_8B_CONFIG).unwrap();
        if value.is_null() {
            v.as_object_mut().unwrap().remove(key);
        } else {
            v[key] = value;
        }
        Qwen3DenseConfig::from_json(&v.to_string(), Some(QWEN3_8B_GENERATION))
    }

    fn refused(r: Result<Qwen3DenseConfig, String>, needle: &str) {
        let e = r.expect_err("config must be refused");
        assert!(e.contains(needle), "refusal {e:?} does not name {needle:?}");
    }

    // ADR 0004 step 1: the 8B's KV cache is built from this geometry (36 layers, 32/8, 128).
    #[test]
    fn kv_geometry_from_the_qwen3_8b_config() {
        let c = Qwen3DenseConfig::from_json(QWEN3_8B_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        assert_eq!(
            KvGeometry::from(&c),
            KvGeometry {
                layers: 36,
                q_heads: 32,
                kv_heads: 8,
                head_dim: 128,
                rope_theta: 1e6,
                block_size: KV_BLOCK_SIZE,
            }
        );
    }

    #[test]
    fn reads_the_qwen3_8b_config() {
        let c = Qwen3DenseConfig::from_json(QWEN3_8B_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        assert_eq!(
            (
                c.layers,
                c.hidden,
                c.q_heads,
                c.kv_heads,
                c.head_dim,
                c.intermediate,
                c.vocab
            ),
            (36, 4096, 32, 8, 128, 12288, 151936)
        );
        assert_eq!((c.rms_eps, c.rope_theta), (1e-6, 1e6));
        assert_eq!(c.eos_token_ids, vec![151645, 151643]);
        assert_eq!((c.qkv_n(), c.gate_up_n()), (6144, 24576));
        // Design section 0: 7,568,097,280 weights streamed per token (lm_head in, embedding out).
        assert_eq!(c.model_resident_bytes(), 2 * 7_568_097_280);
        assert_eq!(c.layer_resident_bytes(), 2 * 192_937_984);
        let real = std::path::Path::new(env!("HOME")).join("models/qwen3-8b");
        if real.join("config.json").exists() {
            assert_eq!(Qwen3DenseConfig::from_checkpoint(&real).unwrap(), c);
        }
    }

    #[test]
    fn refuses_qwen3_moe() {
        let moe = r#"{"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "hidden_size": 2048, "num_experts": 128, "tie_word_embeddings": false}"#;
        refused(Qwen3DenseConfig::from_json(moe, None), "qwen3_moe");
        refused(variant("model_type", "qwen3_moe".into()), "qwen3_moe");
        refused(variant("model_type", "llama".into()), "model_type");
    }

    #[test]
    fn tied_embeddings_config_parses() {
        let c = variant("tie_word_embeddings", true.into()).unwrap();
        assert!(c.tie_word_embeddings);
        assert!(
            !variant("tie_word_embeddings", false.into())
                .unwrap()
                .tie_word_embeddings
        );
        // The lm_head is not read through the loader buffer when tied: the largest projection
        // (gate|up, 2 * 12288 x 4096 words) sets it, not the 151936 x 4096 lm_head.
        assert_eq!(c.loader_buffer_words(), 2 * 12288 * 4096);
    }

    /// Qwen3-4B-Instruct-2507's shape (tied, k 2560 and 9728) is inside every kernel rule.
    #[test]
    fn qwen3_4b_config_parses_with_tied_embeddings() {
        let cfg = r#"{"architectures": ["Qwen3ForCausalLM"], "attention_bias": false,
  "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128,
  "hidden_act": "silu", "hidden_size": 2560, "initializer_range": 0.02,
  "intermediate_size": 9728, "max_position_embeddings": 262144, "max_window_layers": 36,
  "model_type": "qwen3", "num_attention_heads": 32, "num_hidden_layers": 36,
  "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_scaling": null,
  "rope_theta": 5000000, "sliding_window": null, "tie_word_embeddings": true,
  "torch_dtype": "bfloat16", "transformers_version": "4.51.0", "use_cache": true,
  "use_sliding_window": false, "vocab_size": 151936}"#;
        let c = Qwen3DenseConfig::from_json(cfg, None).unwrap();
        assert!(c.tie_word_embeddings);
        assert_eq!(c.format, DenseWeightFormat::Bf16);
        assert_eq!((c.hidden, c.intermediate, c.layers), (2560, 9728, 36));
        assert_eq!(c.qkv_n(), 6144);
        assert_eq!(c.rope_theta, 5e6);
    }

    /// Tied: no `lm_head.weight` in the checkpoint, the lm_head upload is the embedding bit for
    /// bit, and an untied config over the same tensors is refused (lm_head missing).
    #[test]
    fn tied_lm_head_is_the_embedding() {
        let c = Qwen3DenseConfig {
            tie_word_embeddings: true,
            ..tiny_config()
        };
        let tensors = tensors(&c);
        assert!(tensors.iter().all(|t| t.0 != "lm_head.weight"));
        let up = RecordingDenseUploader::keeping();
        let (w, report) = load_qwen3_dense_omega_from(
            &c,
            &mut memory_source_of(tensors.clone()),
            &up,
            &mut |_| {},
        )
        .unwrap();
        let records = up.records.lock().unwrap();
        let lm = records
            .iter()
            .find(|r| r.what == DenseTensor::LmHead)
            .unwrap();
        assert_eq!(lm.matrix.as_ref().unwrap().bits, w.embed_bf16);
        assert_eq!(report.uploads, 4 * c.layers + 1);
        let untied = tiny_config();
        let e = load_qwen3_dense_omega_from(
            &untied,
            &mut memory_source_of(tensors),
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(e.contains("lm_head"), "{e}");
    }

    #[test]
    fn refuses_k_not_a_multiple_of_256() {
        // hidden 4224 = 16.5 * 256: q|k|v, gate|up and lm_head have k = 4224.
        refused(variant("hidden_size", 4224.into()), "k a multiple of 256");
        // intermediate 12416: down has k = 12416.
        refused(
            variant("intermediate_size", 12416.into()),
            "down [4096][12416]",
        );
    }

    #[test]
    fn refuses_k_past_the_gemv_and_matmul_limits() {
        // down k = 16640 > 16384.
        refused(
            variant("intermediate_size", 16640.into()),
            "down [4096][16640]",
        );
        // lm_head n = 262400 > the resident matmul N limit 262144 (the GEMV itself would take it).
        refused(
            variant("vocab_size", 262400.into()),
            "resident matmul limit",
        );
    }

    #[test]
    fn refuses_attention_shapes_omega_cannot_run() {
        refused(variant("head_dim", 96.into()), "head_dim 96");
        refused(
            variant("num_key_value_heads", 5.into()),
            "QueryHeadsNotDivisible",
        );
        let mut c = tiny_config();
        c.kv_heads = 3;
        refused(c.validate().map(|_| c.clone()), "GQA");
        refused(variant("num_attention_heads", 128.into()), "GQA");
        refused(variant("head_dim", serde_json::Value::Null), "head_dim");
    }

    #[test]
    fn refuses_what_the_engine_does_not_implement() {
        refused(
            variant(
                "rope_scaling",
                serde_json::json!({"rope_type": "yarn", "factor": 4.0}),
            ),
            "rope_scaling",
        );
        refused(
            variant(
                "rope_scaling",
                serde_json::json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
            ),
            "rope_scaling is not implemented",
        );
        refused(variant("use_sliding_window", true.into()), "sliding_window");
        refused(variant("attention_bias", true.into()), "bias");
        refused(
            variant("architectures", serde_json::json!(["LlamaForCausalLM"])),
            "Qwen3ForCausalLM",
        );
        refused(variant("num_experts", 128.into()), "num_experts");
        refused(variant("torch_dtype", "float16".into()), "torch_dtype");
        refused(variant("dtype", "float32".into()), "dtype");
    }

    #[test]
    fn tiny_load_shapes_byte_counts_and_order() {
        let c = tiny_config();
        let up = RecordingDenseUploader::keeping();
        let (w, report) = load_tiny(&up).unwrap();
        let recs = up.records.lock().unwrap();
        let got: Vec<_> = recs
            .iter()
            .map(|r| (r.layer, r.what, r.n, r.k, r.bytes))
            .collect();
        let mut want = Vec::new();
        for l in 0..c.layers {
            want.push((Some(l), DenseTensor::Qkv, 1024, 256, 2 * 1024 * 256));
            want.push((Some(l), DenseTensor::O, 256, 512, 2 * 256 * 512));
            want.push((Some(l), DenseTensor::GateUp, 1024, 256, 2 * 1024 * 256));
            want.push((Some(l), DenseTensor::Down, 256, 512, 2 * 256 * 512));
        }
        want.push((None, DenseTensor::LmHead, 64, 256, 2 * 64 * 256));
        assert_eq!(got, want);
        assert_eq!(report.uploads, want.len());
        assert_eq!(report.resident_bytes, c.model_resident_bytes());
        assert_eq!(
            report.resident_bytes,
            want.iter().map(|w| w.4).sum::<usize>()
        );
        assert_eq!(w.embed_bf16.len(), c.vocab * c.hidden);
        assert_eq!(
            report.host_bytes,
            c.embed_host_bytes() + 4 * (c.hidden + c.layers * (2 * c.hidden + 2 * c.head_dim))
        );
        for l in &w.layers {
            assert_eq!((l.qkv.n(), l.qkv.k()), (c.qkv_n(), c.hidden));
            assert_eq!((l.gate_up.n(), l.gate_up.k()), (c.gate_up_n(), c.hidden));
            assert_eq!((l.q_norm.len(), l.k_norm.len()), (c.head_dim, c.head_dim));
        }
        assert!(w.embed_row(63).is_ok());
        assert!(
            w.embed_row(64).is_err(),
            "out-of-vocabulary token refused, not wrapped"
        );
    }

    #[test]
    fn byte_test_passes_and_concat_order_is_q_k_v_and_gate_up() {
        let c = tiny_config();
        let up = RecordingDenseUploader::keeping();
        load_tiny(&up).unwrap();
        let recs = up.records.lock().unwrap();
        check_upload_bytes(&c, &mut memory_source(&c), &recs).unwrap();

        // Spot check by hand: first row of each block of the layer-1 fused tensors.
        let src = memory_source(&c);
        let row0 = |name: &str, k: usize| -> Vec<u16> {
            src.tensors[name].2[..2 * k]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let (h, qd, kvd, i) = (c.hidden, c.q_dim(), c.kv_dim(), c.intermediate);
        let qkv = recs[4].matrix.as_ref().unwrap();
        let row = |m: &HostGemv, r: usize| m.bits[r * m.k..(r + 1) * m.k].to_vec();
        assert_eq!(
            row(qkv, 0),
            row0("model.layers.1.self_attn.q_proj.weight", h)
        );
        assert_eq!(
            row(qkv, qd),
            row0("model.layers.1.self_attn.k_proj.weight", h)
        );
        assert_eq!(
            row(qkv, qd + kvd),
            row0("model.layers.1.self_attn.v_proj.weight", h)
        );
        let gu = recs[6].matrix.as_ref().unwrap();
        assert_eq!(row(gu, 0), row0("model.layers.1.mlp.gate_proj.weight", h));
        assert_eq!(row(gu, i), row0("model.layers.1.mlp.up_proj.weight", h));
    }

    /// Negative control: an uploader that stores q|k|v with the k and v rows exchanged must fail
    /// the byte test, naming the first wrong row.
    #[test]
    fn negative_control_kv_swap_fails_the_byte_test() {
        let c = tiny_config();
        let up = RecordingDenseUploader::swapping_kv(&c);
        load_tiny(&up).unwrap();
        let err = check_upload_bytes(&c, &mut memory_source(&c), &up.records.lock().unwrap())
            .expect_err("a k/v row swap must fail the byte test");
        assert!(
            err.contains(
                "upload Some(0) q|k|v: row 512 is not model.layers.0.self_attn.k_proj.weight row 0"
            ),
            "{err}"
        );
    }

    /// Negative control for the MLP concat: gate and up rows exchanged must fail the byte test.
    #[test]
    fn negative_control_gate_up_swap_fails_the_byte_test() {
        let c = tiny_config();
        let up = RecordingDenseUploader::swapping_gate_up(&c);
        load_tiny(&up).unwrap();
        let err = check_upload_bytes(&c, &mut memory_source(&c), &up.records.lock().unwrap())
            .expect_err("a gate/up row swap must fail the byte test");
        assert!(
            err.contains(
                "upload Some(0) gate|up: row 0 is not model.layers.0.mlp.gate_proj.weight row 0"
            ),
            "{err}"
        );
    }

    #[test]
    fn nan_and_missing_tensors_are_refused() {
        let c = tiny_config();
        let mut src = memory_source(&c);
        let name = "model.layers.1.mlp.up_proj.weight";
        let bytes = &mut src.tensors.get_mut(name).unwrap().2;
        bytes[10..12].copy_from_slice(&f32_to_bf16(f32::NAN).to_le_bytes());
        let e = load_qwen3_dense_omega_from(
            &c,
            &mut src,
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(e.contains("layer 1 gate|up") && e.contains("NaN"), "{e}");

        let mut src = memory_source(&c);
        src.tensors.remove("model.layers.1.self_attn.v_proj.weight");
        let e = load_qwen3_dense_omega_from(
            &c,
            &mut src,
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(e.contains("model.layers.1.self_attn.v_proj.weight"), "{e}");

        let mut src = memory_source(&c);
        let k = src
            .tensors
            .get_mut("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        k.1 = vec![c.kv_dim() / 2, 2 * c.hidden];
        let e = load_qwen3_dense_omega_from(
            &c,
            &mut src,
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(e.contains("k_proj"), "{e}");
    }

    /// Design gate: peak host memory under 2 GB over the embedding. The loader's one buffer is
    /// the largest matrix; at the 8B shape that is the lm_head, 1.24 GB.
    #[test]
    fn transient_is_one_buffer_under_2gb_at_the_8b_shape() {
        let c8 = Qwen3DenseConfig::from_json(QWEN3_8B_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        let t = dense_loader_transient_bytes(&c8);
        assert_eq!(t, 2 * 151936 * 4096);
        assert!(t < 2_000_000_000, "{t}");
        let (_, report) = load_tiny(&RecordingDenseUploader::keeping()).unwrap();
        assert_eq!(
            report.transient_bytes,
            dense_loader_transient_bytes(&tiny_config())
        );
        assert!(report.peak_rss_bytes.is_some_and(|b| b > 0));
    }

    /// Fills every read with a finite pattern (0x3c bytes: bf16 1.17, e4m3 1.5), so a full-width
    /// model can be loaded without a checkpoint on disk. Reports the dtypes of `format`'s
    /// checkpoint.
    struct PatternSource(DenseWeightFormat);

    impl TensorSource for PatternSource {
        fn read_into(
            &mut self,
            name: &str,
            dtype: &str,
            _: &[usize],
            dst: &mut [u8],
        ) -> Result<(), String> {
            assert_eq!(Some(dtype.to_string()), self.dtype_of(name).unwrap());
            dst.fill(0x3c);
            Ok(())
        }

        fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
            let fp8 = self.0 == DenseWeightFormat::Fp8Block128;
            Ok(match (fp8, name) {
                (false, n) if n.ends_with("_scale_inv") => None,
                (true, n) if n.ends_with("_proj.weight") => Some("F8_E4M3".into()),
                _ => Some("BF16".into()),
            })
        }
    }

    /// Process peak RSS growth over one load, with the high-water mark reset first
    /// (`/proc/self/clear_refs` 5). `None` when `/proc` cannot do that. The mark is per process:
    /// only meaningful where nothing else allocates, so it runs in [`rss_probe_child`].
    fn measured_peak_growth(c: &Qwen3DenseConfig, up: &RecordingDenseUploader) -> Option<u64> {
        std::fs::write("/proc/self/clear_refs", "5").ok()?;
        let before = crate::qwen3_moe_omega::current_rss_bytes()?;
        let (w, report) =
            load_qwen3_dense_omega_from(c, &mut PatternSource(c.format), up, &mut |_| {}).unwrap();
        drop(w);
        Some(report.peak_rss_bytes?.saturating_sub(before))
    }

    /// One layer at the 8B widths with a 32k vocabulary (embedding 268 MB) in `format`.
    fn rss_config(format: DenseWeightFormat) -> Qwen3DenseConfig {
        let mut c =
            Qwen3DenseConfig::from_json(QWEN3_8B_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        c.format = format;
        c.layers = 1;
        c.vocab = 32768;
        c
    }

    const RSS_CHILD_ENV: &str = "AIEN_OWN8B_RSS_CHILD";
    const RSS_CHILD_TEST: &str = "qwen3_dense_omega::tests::rss_probe_child";

    /// The child side of [`measured_peak_growth_in_child`], ignored so it is not counted as a
    /// passing test; the parent runs it with `--ignored --exact`. It does nothing unless
    /// `AIEN_OWN8B_RSS_CHILD=<bf16|fp8>:<streaming|keeping>` is set; then it measures that one
    /// load and prints `OWN8B-RSS-GROWTH <bytes>` (or `skipped`). The harness prints it after the test name on
    /// the same line.
    #[test]
    #[ignore = "child process of the measured peak RSS gates; started by them"]
    fn rss_probe_child() {
        let Ok(case) = std::env::var(RSS_CHILD_ENV) else {
            return;
        };
        let (format, keep) = case.split_once(':').expect("case is <format>:<uploader>");
        let format = match format {
            "bf16" => DenseWeightFormat::Bf16,
            "fp8" => DenseWeightFormat::Fp8Block128,
            other => panic!("unknown format {other}"),
        };
        let up = match keep {
            "streaming" => RecordingDenseUploader::record_only(),
            "keeping" => RecordingDenseUploader::keeping(),
            other => panic!("unknown uploader {other}"),
        };
        match measured_peak_growth(&rss_config(format), &up) {
            Some(g) => println!("OWN8B-RSS-GROWTH {g}"),
            None => println!("OWN8B-RSS-GROWTH skipped"),
        }
    }

    /// [`measured_peak_growth`] in a child process (this test binary re-run on
    /// [`rss_probe_child`] alone), so the high-water mark counts this one load and no other test
    /// thread. `None` when the child could not reset the mark.
    fn measured_peak_growth_in_child(case: &str) -> Option<u64> {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                RSS_CHILD_TEST,
                "--ignored",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(RSS_CHILD_ENV, case)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "RSS child {case} failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let value = stdout
            .lines()
            .find_map(|l| l.split_once("OWN8B-RSS-GROWTH ").map(|(_, v)| v.trim()))
            .unwrap_or_else(|| panic!("RSS child {case} printed no measurement:\n{stdout}"));
        (value != "skipped").then(|| value.parse().unwrap())
    }

    /// The memory gate measured, not just computed: one layer at the 8B widths with a 32k
    /// vocabulary must raise the peak RSS by at most the embedding plus the loader's buffers
    /// plus 128 MB slack. Control: the same load through the keeping uploader (a full host copy
    /// of the model, as an f32 or bf16 image would be) must break the same bound, so the check
    /// cannot pass vacuously. Each load is measured in its own child process.
    fn assert_peak_rss_bound(format: DenseWeightFormat, name: &str) {
        let c = rss_config(format);
        let bound = (c.embed_host_bytes() + dense_loader_transient_bytes(&c) + (128 << 20)) as u64;
        let Some(streaming) = measured_peak_growth_in_child(&format!("{name}:streaming")) else {
            // A CI runner is a plain Linux VM: the gate must run there, never skip silently.
            assert!(
                std::env::var_os("CI").is_none(),
                "no /proc/self/clear_refs under CI: the measured peak RSS gate cannot run"
            );
            println!("no /proc/self/clear_refs: measured peak RSS check skipped");
            return;
        };
        assert!(
            streaming <= bound,
            "{name} streaming load grew peak RSS by {streaming}, bound {bound}"
        );
        let keeping = measured_peak_growth_in_child(&format!("{name}:keeping")).unwrap();
        assert!(
            keeping > bound,
            "control: a full {name} host copy grew peak RSS by only {keeping}, bound {bound}"
        );
    }

    #[test]
    fn measured_peak_rss_is_the_embedding_plus_one_buffer() {
        assert_peak_rss_bound(DenseWeightFormat::Bf16, "bf16");
    }

    /// The same gate at FP8: the loader's buffer is the bf16 lm_head (268 MB at this vocab, 1.24
    /// GB at the 8B), the e4m3 projections fit inside it.
    #[test]
    fn measured_peak_rss_at_fp8_widths() {
        let c = rss_config(DenseWeightFormat::Fp8Block128);
        assert_eq!(
            dense_loader_transient_bytes(&c),
            2 * 32768 * 4096 + 2 * (24576 / 128) * (4096 / 128)
        );
        assert_peak_rss_bound(DenseWeightFormat::Fp8Block128, "fp8");
    }

    /// The tiny model through the shard files: the loader's values equal
    /// `TransformerWeights::load_from_safetensors` (the f32 reference loader) bit for bit.
    #[test]
    fn loader_equals_transformer_weights_on_the_tiny_model() {
        let c = tiny_config();
        let dir = scratch("parity");
        let _ = std::fs::remove_dir_all(&dir);
        write_checkpoint(&dir, &c).unwrap();
        let up = RecordingDenseUploader::keeping();
        let (w, _) = load_qwen3_dense_omega(&dir, &up, &mut |_| {}).unwrap();
        assert_eq!(w.config, c);
        let mc = crate::model_dir::load_model_config(&dir).unwrap();
        let tw = crate::weights::TransformerWeights::load_from_safetensors(
            dir.join("model.safetensors.index.json"),
            &mc,
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        let wide = |m: &HostGemv| -> Vec<f32> { m.bits.iter().map(|&b| bf16_to_f32(b)).collect() };
        let cat = |parts: &[&[f32]]| -> Vec<f32> { parts.concat() };
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        let recs = up.records.lock().unwrap();
        let embed: Vec<f32> = w.embed_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        assert_eq!(bits(&embed), bits(&tw.embed_tokens));
        for (l, (dl, tl)) in w.layers.iter().zip(&tw.layers).enumerate() {
            let m = |j: usize| recs[4 * l + j].matrix.as_ref().unwrap();
            assert_eq!(
                bits(&wide(m(0))),
                bits(&cat(&[&tl.q_proj, &tl.k_proj, &tl.v_proj]))
            );
            assert_eq!(bits(&wide(m(1))), bits(&tl.o_proj));
            assert_eq!(bits(&wide(m(2))), bits(&cat(&[&tl.gate_proj, &tl.up_proj])));
            assert_eq!(bits(&wide(m(3))), bits(&tl.down_proj));
            assert_eq!(bits(&dl.input_norm), bits(&tl.input_layernorm));
            assert_eq!(bits(&dl.post_norm), bits(&tl.post_attention_layernorm));
            assert_eq!(bits(&dl.q_norm), bits(tl.q_norm.as_ref().unwrap()));
            assert_eq!(bits(&dl.k_norm), bits(tl.k_norm.as_ref().unwrap()));
        }
        assert_eq!(bits(&w.final_norm), bits(&tw.final_norm));
        let head = recs.last().unwrap().matrix.as_ref().unwrap();
        assert_eq!(bits(&wide(head)), bits(tw.lm_head.as_ref().unwrap()));

        // The fused GEMV gives the three separate projections' rows.
        let x: Vec<f32> = (0..c.hidden).map(prand_x).collect();
        let mut y = vec![0f32; c.qkv_n()];
        w.layers[0].qkv.gemv_f32(&x, &mut y).unwrap();
        let kv0 = c.q_dim();
        let k_row0: f32 = tw.layers[0].k_proj[..c.hidden]
            .iter()
            .zip(&x)
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(y[kv0].to_bits(), k_row0.to_bits());
    }

    fn prand_x(j: usize) -> f32 {
        crate::qwen3_moe_omega::test_support::prand(77, j)
    }

    /// The feature-gated door other crates use: the same tiny model from memory and from disk.
    #[cfg(feature = "host-test-engine")]
    #[test]
    fn host_test_model_loads_from_memory_and_from_disk() {
        let (mem, _) = super::host_test_model::load().unwrap();
        let dir = scratch("host-test-model");
        let _ = std::fs::remove_dir_all(&dir);
        super::host_test_model::write(&dir).unwrap();
        let (disk, _) =
            load_qwen3_dense_omega(&dir, &RecordingDenseUploader::keeping(), &mut |_| {}).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(mem.config, super::host_test_model::config());
        assert_eq!(disk.config, mem.config);
        assert_eq!(disk.embed_bf16, mem.embed_bf16);
        let x: Vec<f32> = (0..mem.config.hidden).map(prand_x).collect();
        let (mut a, mut b) = (vec![0f32; mem.config.vocab], vec![0f32; mem.config.vocab]);
        mem.lm_head.gemv_f32(&x, &mut a).unwrap();
        disk.lm_head.gemv_f32(&x, &mut b).unwrap();
        assert_eq!(a, b);
    }

    /// The real Qwen3-8B through the streaming loader and the record-only fake (host memory
    /// only, no chip): per-tensor bytes, total, peak RSS, wall time. Run with
    /// `cargo test -p aien-inference-abi --lib qwen3_dense_omega::tests::real_qwen3_8b -- --ignored --nocapture --test-threads=1`
    /// (`AIEN_QWEN3_8B_DIR` overrides `~/models/qwen3-8b`).
    #[test]
    #[ignore = "reads the 16 GB Qwen3-8B checkpoint"]
    fn real_qwen3_8b_streaming_load_host_only() {
        let dir = std::env::var_os("AIEN_QWEN3_8B_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("HOME")).join("models/qwen3-8b"));
        real_streaming_load(&dir, DenseWeightFormat::Bf16, 2 * 7_568_097_280);
    }

    /// One real-checkpoint load through the record-only uploader, its numbers printed as
    /// `OWN8B-LOAD` lines and held to the design's byte count, upload count and memory bound.
    fn real_streaming_load(dir: &std::path::Path, format: DenseWeightFormat, resident: usize) {
        let rss_before = crate::qwen3_moe_omega::current_rss_bytes().unwrap();
        let up = RecordingDenseUploader::record_only();
        let (w, report) = load_qwen3_dense_omega(dir, &up, &mut |_| {}).unwrap();
        let c = &w.config;
        assert_eq!(c.format, format);
        let recs = up.records.lock().unwrap();
        let mut per: Vec<(DenseTensor, usize, usize, usize, usize)> = Vec::new();
        for r in recs.iter() {
            match per.iter_mut().find(|p| p.0 == r.what) {
                Some(p) => {
                    p.3 += 1;
                    p.4 += r.bytes;
                }
                None => per.push((r.what, r.n, r.k, 1, r.bytes)),
            }
        }
        for (what, n, k, count, bytes) in &per {
            println!(
                "OWN8B-LOAD {format:?} tensor {:<8} [{n}][{k}] bytes_each={} count={count} bytes_total={bytes}",
                what.name(),
                c.matrix_bytes(*what, *n, *k)
            );
        }
        let peak = report.peak_rss_bytes.unwrap();
        println!(
            "OWN8B-LOAD {format:?} resident_bytes={} model_resident_bytes={} uploads={} host_bytes={} \
             embed_bytes={} transient_bytes={} rss_before={rss_before} peak_rss_before={} \
             peak_rss={peak} peak_over_start={} seconds={:.2}",
            report.resident_bytes,
            c.model_resident_bytes(),
            report.uploads,
            report.host_bytes,
            c.embed_host_bytes(),
            report.transient_bytes,
            report.peak_rss_before_bytes.unwrap(),
            peak - rss_before,
            report.seconds
        );
        assert_eq!(report.resident_bytes, resident);
        assert_eq!(report.resident_bytes, c.model_resident_bytes());
        assert_eq!(report.uploads, 4 * 36 + 1);
        assert!(
            (peak - rss_before) as usize <= c.embed_host_bytes() + 2_000_000_000,
            "peak RSS grew {} bytes, limit is the embedding plus 2 GB",
            peak - rss_before
        );
    }

    // -----------------------------------------------------------------------------------------
    // Stage 1b: the FP8 branch.
    // -----------------------------------------------------------------------------------------

    /// `~/models/qwen3-8b-fp8/config.json` (Qwen/Qwen3-8B-FP8 at 220b46e3), verbatim values.
    const QWEN3_8B_FP8_CONFIG: &str = r#"{
  "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 12288,
  "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
  "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
  "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
  "use_cache": true, "use_sliding_window": false, "vocab_size": 151936,
  "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "weight_block_size": [128, 128]}
}"#;

    /// The FP8 config with `quantization_config` replaced (`null` removes the key).
    fn quant_variant(q: serde_json::Value) -> Result<Qwen3DenseConfig, String> {
        let mut v: serde_json::Value = serde_json::from_str(QWEN3_8B_FP8_CONFIG).unwrap();
        if q.is_null() {
            v.as_object_mut().unwrap().remove("quantization_config");
        } else {
            v["quantization_config"] = q;
        }
        Qwen3DenseConfig::from_json(&v.to_string(), Some(QWEN3_8B_GENERATION))
    }

    #[test]
    fn reads_the_qwen3_8b_fp8_config() {
        let c =
            Qwen3DenseConfig::from_json(QWEN3_8B_FP8_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        assert_eq!(c.format, DenseWeightFormat::Fp8Block128);
        let bf16 = Qwen3DenseConfig::from_json(QWEN3_8B_CONFIG, Some(QWEN3_8B_GENERATION)).unwrap();
        assert_eq!(bf16.format, DenseWeightFormat::Bf16);
        assert_eq!(
            c,
            Qwen3DenseConfig {
                format: DenseWeightFormat::Fp8Block128,
                ..bf16
            }
        );
        // Design section 0.1: e4m3 weights 6,945,767,424 bytes, scales 847,872 (bf16), lm_head
        // bf16 1,244,659,712; per layer 192,937,984 + 23,552.
        assert_eq!(c.layer_resident_bytes(), 192_937_984 + 2 * 11_776);
        assert_eq!(
            c.model_resident_bytes(),
            6_945_767_424 + 847_872 + 1_244_659_712
        );
        // The loader's buffer is still the bf16 lm_head; the scale buffer is gate|up's grid.
        assert_eq!(
            dense_loader_transient_bytes(&c),
            2 * 151936 * 4096 + 2 * 192 * 32
        );
        // The bf16 engines' strict reader keeps refusing the FP8 checkpoint by name.
        let e = crate::model_dir::model_config_from_hf_json("x", QWEN3_8B_FP8_CONFIG, None)
            .unwrap_err();
        assert!(e.contains("quantization_config"), "{e}");
        let real = std::path::Path::new(env!("HOME")).join("models/qwen3-8b-fp8");
        if real.join("config.json").exists() {
            assert_eq!(Qwen3DenseConfig::from_checkpoint(&real).unwrap(), c);
        }
    }

    #[test]
    fn quantization_config_absent_null_or_exact_fp8() {
        assert_eq!(
            quant_variant(serde_json::Value::Null).unwrap().format,
            DenseWeightFormat::Bf16
        );
        let mut v: serde_json::Value = serde_json::from_str(QWEN3_8B_FP8_CONFIG).unwrap();
        v["quantization_config"] = serde_json::Value::Null;
        let c = Qwen3DenseConfig::from_json(&v.to_string(), None).unwrap();
        assert_eq!(c.format, DenseWeightFormat::Bf16);
        // Entries that cannot match a projection name are accepted (lm_head, the embedding and the
        // norms are bf16 anyway).
        let c = quant_variant(
            serde_json::json!({"activation_scheme": "dynamic", "fmt": "e4m3",
            "quant_method": "fp8", "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head", "model.layers.0.input_layernorm",
                "model.norm", "model.embed_tokens"]}),
        )
        .unwrap();
        assert_eq!(c.format, DenseWeightFormat::Fp8Block128);
    }

    #[test]
    fn refuses_foreign_quantization_configs_by_name() {
        let fp8 = |key: &str, value: serde_json::Value| {
            let mut q = serde_json::json!({"activation_scheme": "dynamic", "fmt": "e4m3",
                "quant_method": "fp8", "weight_block_size": [128, 128]});
            if value.is_null() {
                q.as_object_mut().unwrap().remove(key);
            } else {
                q[key] = value;
            }
            quant_variant(q)
        };
        refused(
            fp8("weight_block_size", serde_json::json!([64, 64])),
            "weight_block_size",
        );
        refused(
            fp8("weight_block_size", serde_json::Value::Null),
            "weight_block_size absent",
        );
        refused(fp8("fmt", "e5m2".into()), "fmt");
        refused(
            fp8("activation_scheme", "static".into()),
            "activation_scheme",
        );
        refused(
            fp8(
                "modules_to_not_convert",
                serde_json::json!(["lm_head", "model.layers.3.mlp.down_proj"]),
            ),
            "modules_to_not_convert names \"model.layers.3.mlp.down_proj\"",
        );
        refused(
            fp8("modules_to_not_convert", "lm_head".into()),
            "not a list",
        );
        // Hugging Face skips a module when an entry is a substring of its name, so a layer or
        // block prefix (the 30B's list really has "model.layers.0.mlp.gate") leaves projections
        // unconverted: refused at config, not only at load.
        for prefix in [
            "model.layers.0.mlp.gate",
            "model.layers.3.mlp",
            "model.layers.3",
            "self_attn",
            "",
        ] {
            refused(
                fp8("modules_to_not_convert", serde_json::json!([prefix])),
                &format!("modules_to_not_convert names \"{prefix}\""),
            );
        }
        refused(
            fp8("modules_to_not_convert", serde_json::json!([3])),
            "modules_to_not_convert names 3",
        );
        refused(
            fp8("ignored_layers", serde_json::json!([])),
            "ignored_layers",
        );
        // int4 recipes are refused for their method.
        refused(
            quant_variant(
                serde_json::json!({"quant_method": "gptq", "bits": 4, "group_size": 128}),
            ),
            "quant_method \"gptq\": only",
        );
        refused(
            quant_variant(serde_json::json!({"quant_method": "awq", "bits": 4})),
            "quant_method \"awq\"",
        );
        refused(quant_variant("fp8".into()), "is not an object");
    }

    #[test]
    fn fp8_shape_rules() {
        // head_dim 64 with 3 kv heads: kv_dim 192 is not a whole number of scale rows.
        let mut c = tiny_fp8_config();
        c.head_dim = 64;
        c.q_heads = 6;
        c.kv_heads = 3;
        refused(c.validate().map(|_| c.clone()), "kv_dim 192");
        // k past the FP8 GEMV's 32768 (aien_omega_fp8::MAX_K).
        let mut c = tiny_fp8_config();
        c.intermediate = aien_omega_fp8::MAX_K + 128;
        refused(c.validate().map(|_| c.clone()), "down [256][32896]");
        // The bf16 rule (k a multiple of 256) does not apply to an FP8 projection: k 384 is fine
        // for down, but the bf16 lm_head (k = hidden) keeps its rule.
        let mut c = tiny_fp8_config();
        c.intermediate = 384;
        c.validate().unwrap();
        c.hidden = 384;
        refused(c.validate().map(|_| c.clone()), "lm_head");
    }

    fn load_tiny_fp8(
        up: &RecordingDenseUploader,
    ) -> Result<(Qwen3DenseOmegaWeights, DenseLoadReport), String> {
        let c = tiny_fp8_config();
        load_qwen3_dense_omega_from(&c, &mut memory_source(&c), up, &mut |_| {})
    }

    #[test]
    fn tiny_fp8_load_shapes_byte_counts_and_order() {
        let c = tiny_fp8_config();
        let up = RecordingDenseUploader::keeping();
        let (w, report) = load_tiny_fp8(&up).unwrap();
        let recs = up.records.lock().unwrap();
        let got: Vec<_> = recs
            .iter()
            .map(|r| (r.layer, r.what, r.n, r.k, r.bytes, r.fp8.is_some()))
            .collect();
        let mut want = Vec::new();
        for l in 0..c.layers {
            let l = Some(l);
            want.push((l, DenseTensor::Qkv, 1024, 256, 1024 * 256 + 2 * 8 * 2, true));
            want.push((l, DenseTensor::O, 256, 512, 256 * 512 + 2 * 2 * 4, true));
            want.push((
                l,
                DenseTensor::GateUp,
                1024,
                256,
                1024 * 256 + 2 * 8 * 2,
                true,
            ));
            want.push((l, DenseTensor::Down, 256, 512, 256 * 512 + 2 * 2 * 4, true));
        }
        want.push((None, DenseTensor::LmHead, 64, 256, 2 * 64 * 256, false));
        assert_eq!(got, want);
        assert_eq!(report.uploads, want.len());
        assert_eq!(report.resident_bytes, c.model_resident_bytes());
        assert_eq!(
            report.resident_bytes,
            want.iter().map(|w| w.4).sum::<usize>()
        );
        assert_eq!(report.transient_bytes, dense_loader_transient_bytes(&c));
        assert_eq!(w.embed_bf16.len(), c.vocab * c.hidden);
    }

    #[test]
    fn fp8_byte_test_passes_and_stacks_bytes_and_scale_rows() {
        let c = tiny_fp8_config();
        let up = RecordingDenseUploader::keeping();
        load_tiny_fp8(&up).unwrap();
        let recs = up.records.lock().unwrap();
        check_upload_bytes(&c, &mut memory_source(&c), &recs).unwrap();

        // By hand, layer 1: the first row and first scale row of each stacked part.
        let src = memory_source(&c);
        let (h, qd, kvd, i, kb) = (c.hidden, c.q_dim(), c.kv_dim(), c.intermediate, 2);
        let scale_row0 = |name: &str| -> Vec<u16> {
            src.tensors[&format!("{name}_scale_inv")].2[..2 * kb]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let p = |n: &str| format!("model.layers.1.{n}.weight");
        let qkv = recs[4].fp8.as_ref().unwrap();
        for (name, row) in [
            (p("self_attn.q_proj"), 0),
            (p("self_attn.k_proj"), qd),
            (p("self_attn.v_proj"), qd + kvd),
        ] {
            assert_eq!(
                &qkv.weights()[row * h..(row + 1) * h],
                &src.tensors[&name].2[..h]
            );
            let srow = row / 128;
            assert_eq!(&qkv.scales()[srow * kb..(srow + 1) * kb], scale_row0(&name));
        }
        let gu = recs[6].fp8.as_ref().unwrap();
        assert_eq!(
            &gu.weights()[i * h..(i + 1) * h],
            &src.tensors[&p("mlp.up_proj")].2[..h]
        );
        assert_eq!(
            &gu.scales()[(i / 128) * kb..(i / 128 + 1) * kb],
            scale_row0(&p("mlp.up_proj"))
        );
    }

    fn byte_test_error(up: RecordingDenseUploader) -> String {
        let c = tiny_fp8_config();
        load_tiny_fp8(&up).unwrap();
        check_upload_bytes(&c, &mut memory_source(&c), &up.records.lock().unwrap())
            .expect_err("a wrong upload must fail the byte test")
    }

    /// Negative controls: k and v rows exchanged (bytes and scales), gate and up exchanged, only
    /// the k and v scale rows exchanged, and every scale grid transposed must each fail the FP8
    /// byte test, naming the first wrong row.
    #[test]
    fn negative_controls_fail_the_fp8_byte_test() {
        let c = tiny_fp8_config();
        let e = byte_test_error(RecordingDenseUploader::swapping_kv(&c));
        assert!(
            e.contains(
                "upload Some(0) q|k|v: row 512 is not model.layers.0.self_attn.k_proj.weight row 0"
            ),
            "{e}"
        );
        let e = byte_test_error(RecordingDenseUploader::swapping_gate_up(&c));
        assert!(
            e.contains(
                "upload Some(0) gate|up: row 0 is not model.layers.0.mlp.gate_proj.weight row 0"
            ),
            "{e}"
        );
        let e = byte_test_error(RecordingDenseUploader::swapping_kv_scales(&c));
        assert!(
            e.contains("upload Some(0) q|k|v: scale row 4 is not model.layers.0.self_attn.k_proj.weight_scale_inv row 0"),
            "{e}"
        );
        let e = byte_test_error(RecordingDenseUploader::transposing_scales());
        assert!(
            e.contains("upload Some(0) q|k|v: scale row 0 is not model.layers.0.self_attn.q_proj.weight_scale_inv row 0"),
            "{e}"
        );
    }

    fn load_fp8_err(edit: impl FnOnce(&mut MemorySource)) -> String {
        let c = tiny_fp8_config();
        let mut src = memory_source(&c);
        edit(&mut src);
        load_qwen3_dense_omega_from(
            &c,
            &mut src,
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err()
    }

    /// The header decides per tensor: an FP8 weight without `weight_scale_inv`, a bf16
    /// projection in an FP8 checkpoint, an FP8 lm_head, a transposed scale grid in the file, an
    /// e4m3 NaN code and a NaN scale are each refused by name.
    #[test]
    fn fp8_per_tensor_dtype_rule_and_refusals() {
        let v = "model.layers.1.self_attn.v_proj.weight";
        let e = load_fp8_err(|s| {
            s.tensors.remove(&format!("{v}_scale_inv"));
        });
        assert!(
            e.contains(&format!("{v}: F8_E4M3 without {v}_scale_inv: refused")),
            "{e}"
        );

        let e = load_fp8_err(|s| {
            let t = s.tensors.get_mut(v).unwrap();
            let words: Vec<u8> = t.2.iter().flat_map(|_| [0x80u8, 0x3c]).collect();
            *t = ("BF16".into(), t.1.clone(), words);
        });
        assert!(
            e.contains(&format!("{v}: BF16 projection in an FP8 checkpoint")),
            "{e}"
        );

        let e = load_fp8_err(|s| {
            let t = s.tensors.get_mut("lm_head.weight").unwrap();
            let bytes = vec![0x38u8; t.2.len() / 2];
            *t = ("F8_E4M3".into(), t.1.clone(), bytes);
        });
        assert!(e.contains("lm_head.weight: F8_E4M3"), "{e}");

        // The scale grid stored [k/128][n/128] in the file: refused by the shape check.
        let o = "model.layers.0.self_attn.o_proj.weight_scale_inv";
        let e = load_fp8_err(|s| {
            let t = s.tensors.get_mut(o).unwrap();
            t.1 = vec![t.1[1], t.1[0]];
        });
        assert!(e.contains(o) && e.contains("[4, 2]"), "{e}");

        let e = load_fp8_err(|s| {
            s.tensors
                .get_mut("model.layers.1.mlp.up_proj.weight")
                .unwrap()
                .2[17] = 0xff;
        });
        assert!(
            e.contains("layer 1 gate|up") && e.contains("e4m3 NaN code"),
            "{e}"
        );

        let e = load_fp8_err(|s| {
            let t = s
                .tensors
                .get_mut("model.layers.0.mlp.down_proj.weight_scale_inv")
                .unwrap();
            t.2[2..4].copy_from_slice(&0x7fc0u16.to_le_bytes());
        });
        assert!(
            e.contains("layer 0 down") && e.contains("scale 1 is NaN"),
            "{e}"
        );

        // A bf16 checkpoint (no quantization_config) holding FP8 tensors is refused too.
        let c = tiny_config();
        let mut src = memory_source_of(tensors(&tiny_fp8_config()));
        let e = load_qwen3_dense_omega_from(
            &c,
            &mut src,
            &RecordingDenseUploader::keeping(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(e.contains("q_proj.weight") && e.contains("F8_E4M3"), "{e}");
    }

    /// An uploader without an FP8 path refuses, named with the tensor.
    #[test]
    fn bf16_only_uploader_refuses_an_fp8_load() {
        struct Bf16Only;
        impl DenseUploader for Bf16Only {
            fn upload_bf16(
                &self,
                _: Option<usize>,
                _: DenseTensor,
                n: usize,
                k: usize,
                _: &[u16],
            ) -> Result<Arc<dyn DenseGemv>, String> {
                Ok(Arc::new(RecordedShape { n, k }))
            }
        }
        let c = tiny_fp8_config();
        let e = load_qwen3_dense_omega_from(&c, &mut memory_source(&c), &Bf16Only, &mut |_| {})
            .unwrap_err();
        assert!(
            e.contains("layer 0 q|k|v [1024][256] upload: this uploader has no FP8 path"),
            "{e}"
        );
    }

    /// The host reference's formula, element by element: with a one-hot `x` every output is a
    /// single product, so `y[r]` must be `e4m3(w[r][c]) * scale[r/128][c/128]` exactly (exact
    /// variant) and that product rounded to bf16 (dequantised variant). A scale indexed by the
    /// wrong block fails here.
    #[test]
    fn host_fp8_gemv_is_the_block_scaled_formula() {
        let up = RecordingDenseUploader::keeping();
        load_tiny_fp8(&up).unwrap();
        let recs = up.records.lock().unwrap();
        let lut = fp8_table();
        for rec in recs.iter().filter(|r| r.fp8.is_some()).take(4) {
            let m = rec.fp8.as_ref().unwrap();
            let (n, k) = (m.n(), m.k());
            let deq =
                HostFp8Gemv::dequantised_bf16(n, k, m.weights().to_vec(), m.scales().to_vec())
                    .unwrap();
            for col in [0, 127, 128, k - 1] {
                let mut x = vec![0f32; k];
                x[col] = 1.0;
                let (mut y, mut yd) = (vec![0f32; n], vec![0f32; n]);
                m.gemv_f32(&x, &mut y).unwrap();
                deq.gemv_f32(&x, &mut yd).unwrap();
                for r in 0..n {
                    let s = bf16_to_f32(m.scales()[(r / 128) * (k / 128) + col / 128]);
                    let want = lut[m.weights()[r * k + col] as usize] * s;
                    // Value equality: the zero terms of the sum make a -0.0 product +0.0.
                    assert_eq!(y[r], want, "{:?} r{r} c{col}", rec.what);
                    assert_eq!(
                        yd[r],
                        bf16_to_f32(f32_to_bf16(want)),
                        "{:?} r{r} c{col} dequantised",
                        rec.what
                    );
                }
            }
        }
        // The scales differ between blocks, so the formula check above has teeth.
        let qkv = recs[0].fp8.as_ref().unwrap();
        assert_ne!(qkv.scales()[0], qkv.scales()[1]);
        assert_ne!(qkv.scales()[0], qkv.scales()[2]);
    }

    /// Design 2.2 item 7: the same tiny model written twice, bf16 and FP8 (e4m3-exact values,
    /// scales exactly 1.0), loaded from disk by the two branches, gives the same host GEMV
    /// outputs bit for bit for every matrix (exact and dequantised FP8 variants alike), the same
    /// embedding and the same norms. Control: the FP8 twin through the k/v-swapping uploader
    /// must differ.
    #[test]
    fn bf16_and_fp8_twins_load_to_equal_host_gemvs() {
        let load = |c: &Qwen3DenseConfig, name: &str, up: RecordingDenseUploader| {
            let dir = scratch(name);
            let _ = std::fs::remove_dir_all(&dir);
            write_checkpoint_of(&dir, c, tensors_with(c, TinyWeights::Twin)).unwrap();
            let (w, _) = load_qwen3_dense_omega(&dir, &up, &mut |_| {}).unwrap();
            std::fs::remove_dir_all(&dir).unwrap();
            (w, up)
        };
        let (wb, upb) = load(
            &tiny_config(),
            "twin-bf16",
            RecordingDenseUploader::keeping(),
        );
        let (wf, upf) = load(
            &tiny_fp8_config(),
            "twin-fp8",
            RecordingDenseUploader::keeping(),
        );
        assert_eq!(wb.config.format, DenseWeightFormat::Bf16);
        assert_eq!(wf.config.format, DenseWeightFormat::Fp8Block128);
        assert_eq!(wb.embed_bf16, wf.embed_bf16);
        assert_eq!(wb.final_norm, wf.final_norm);
        for (a, b) in wb.layers.iter().zip(&wf.layers) {
            assert_eq!(
                (&a.input_norm, &a.post_norm, &a.q_norm, &a.k_norm),
                (&b.input_norm, &b.post_norm, &b.q_norm, &b.k_norm)
            );
        }
        let gemv = |m: &dyn DenseGemv, seed: u64| -> Vec<u32> {
            let x: Vec<f32> = (0..m.k())
                .map(|j| crate::qwen3_moe_omega::test_support::prand(seed, j))
                .collect();
            let mut y = vec![0f32; m.n()];
            m.gemv_f32(&x, &mut y).unwrap();
            y.iter().map(|v| v.to_bits()).collect()
        };
        let (rb, rf) = (upb.records.lock().unwrap(), upf.records.lock().unwrap());
        assert_eq!(rb.len(), rf.len());
        let mut compared = 0;
        for (b, f) in rb.iter().zip(rf.iter()) {
            assert_eq!((b.layer, b.what, b.n, b.k), (f.layer, f.what, f.n, f.k));
            let hb = b.matrix.as_ref().unwrap();
            let other: Arc<dyn DenseGemv> = match &f.fp8 {
                Some(m) => {
                    let d = HostFp8Gemv::dequantised_bf16(
                        m.n(),
                        m.k(),
                        m.weights().to_vec(),
                        m.scales().to_vec(),
                    )
                    .unwrap();
                    for seed in [1, 2] {
                        assert_eq!(gemv(&d, seed), gemv(hb.as_ref(), seed), "{:?}", f.what);
                    }
                    m.clone()
                }
                None => f.matrix.clone().unwrap(),
            };
            for seed in [1, 2, 3] {
                assert_eq!(
                    gemv(other.as_ref(), seed),
                    gemv(hb.as_ref(), seed),
                    "{:?} {:?} seed {seed}",
                    f.layer,
                    f.what
                );
            }
            compared += 1;
        }
        assert_eq!(compared, 4 * 2 + 1);

        let c = tiny_fp8_config();
        let (_, ups) = load(&c, "twin-fp8-swap", RecordingDenseUploader::swapping_kv(&c));
        let rs = ups.records.lock().unwrap();
        assert_ne!(
            gemv(rs[0].fp8.as_ref().unwrap().as_ref(), 1),
            gemv(rb[0].matrix.as_ref().unwrap().as_ref(), 1),
            "control: a k/v swap in the FP8 twin must change the q|k|v output"
        );
    }

    /// The real Qwen3-8B-FP8 through the FP8 branch and the record-only fake (host memory only,
    /// no chip; every e4m3 byte and scale is looked at for NaN): per-tensor bytes, total, peak
    /// RSS, wall time. Run with
    /// `cargo test -p aien-inference-abi --lib qwen3_dense_omega::tests::real_qwen3_8b_fp8 -- --ignored --nocapture --test-threads=1`
    /// (`AIEN_QWEN3_8B_FP8_DIR` overrides `~/models/qwen3-8b-fp8`).
    #[test]
    #[ignore = "reads the 9.4 GB Qwen3-8B-FP8 checkpoint"]
    fn real_qwen3_8b_fp8_streaming_load_host_only() {
        let dir = std::env::var_os("AIEN_QWEN3_8B_FP8_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("HOME")).join("models/qwen3-8b-fp8"));
        real_streaming_load(
            &dir,
            DenseWeightFormat::Fp8Block128,
            6_945_767_424 + 847_872 + 1_244_659_712,
        );
    }
}
