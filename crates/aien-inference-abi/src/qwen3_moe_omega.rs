//! Qwen3-MoE (Qwen3-30B-A3B family) on AIEN's own GPU engine (Omega), OWN-30B Stage A.
//!
//! Design: `~/handoffs/2026-10-13-own30b-design.md` (section 2, Stage A). No CUDA, no Mojo, no
//! Python: attention and every matmul run as Omega launches, routing / norms / rope / swiglu run
//! on the host for one row.
//!
//! Cut 1 (this file's first half): a general-model loader. The checkpoint path and the model
//! shape come from the caller and `config.json` (nothing is hard-coded to the Coder snapshot).
//! Each layer's FP8 block-128 weights are dequantised with the exact e4m3 conversion
//! (`fp8_e4m3fn_to_f32`) times the bf16 inverse scale, rounded to bf16 (round-to-nearest-even),
//! written straight into the `[k][n]` layout the engine wants, and uploaded as four resident
//! tensors per layer: fused q|k|v, o, every expert's gate|up side by side along N, and every
//! expert's down side by side along N. The host copy of a layer is reused for the next one, so
//! the peak host memory stays near one layer's transient plus the bf16 embedding.
//!
//! Second half: the Stage A forward (`Qwen3MoeOmegaEngine`): paged bf16 KV, host top-8 routing
//! with `norm_topk_prob`, expert windows through `omega_gpu_matmul_resident_windows_f32`,
//! sampling, prefill and decode, and a library entry (`load` / `generate` with a token callback)
//! for the HTTP server.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::moe_plan::MoeBatchPlan;
use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16, fp8_table, shard_path, Shard, FP8_BLOCK};
use crate::resident::transpose_bf16_for_upload;
mod forward_pass;
pub mod w8a8_control;
mod whole_prompt;

/// The one-list decode token (OWN-30B L2 cut 2), off unless `OWN30B_DECODE_GRAPH=1`.
#[path = "qwen3_moe_omega_dgraph.rs"]
pub mod dgraph;
/// Persistent KV export and import (sc#505): the `HostPagedPoolV1` payload of a token prefix.
#[path = "qwen3_moe_omega_kvio.rs"]
pub mod kvio;

/// Column tile of the engine's tile-major B layout: every expert window must start on one.
pub const WINDOW_TILE: usize = aien_omega_moe::TILE_N;
/// Engine limits for one resident tensor (`omega_blackwell_matmul.h` at omega.lock).
pub const MAX_RESIDENT_K: usize = 16384;
pub const MAX_RESIDENT_N: usize = 262144;

// ---------------------------------------------------------------------------
// Model shape, read from the checkpoint.
// ---------------------------------------------------------------------------

/// The Qwen3-MoE shape, read from `config.json` and validated against what Stage A can run.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    pub layers: usize,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub norm_topk_prob: bool,
    pub tie_word_embeddings: bool,
    /// Ids generation stops on (`config.json` eos plus `generation_config.json` eos list).
    pub eos_token_ids: Vec<u32>,
}

impl Qwen3MoeConfig {
    pub fn from_checkpoint(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut cfg = Self::from_json(&text)?;
        let gen_path = dir.join("generation_config.json");
        if let Ok(gen) = std::fs::read_to_string(&gen_path) {
            let v: serde_json::Value =
                serde_json::from_str(&gen).map_err(|e| format!("{}: {e}", gen_path.display()))?;
            for id in eos_ids(v.get("eos_token_id")) {
                if !cfg.eos_token_ids.contains(&id) {
                    cfg.eos_token_ids.push(id);
                }
            }
        }
        Ok(cfg)
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("config.json: {e}"))?;
        let model_type = v.get("model_type").and_then(|m| m.as_str()).unwrap_or("");
        if model_type != "qwen3_moe" {
            return Err(format!(
                "config.json model_type {model_type:?}, this loader is for qwen3_moe"
            ));
        }
        let num = |key: &str| -> Result<usize, String> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json: missing integer {key}"))
        };
        let flt = |key: &str| -> Result<f32, String> {
            v.get(key)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| format!("config.json: missing number {key}"))
        };
        let cfg = Self {
            layers: num("num_hidden_layers")?,
            hidden: num("hidden_size")?,
            q_heads: num("num_attention_heads")?,
            kv_heads: num("num_key_value_heads")?,
            head_dim: num("head_dim")?,
            experts: num("num_experts")?,
            top_k: num("num_experts_per_tok")?,
            moe_intermediate: num("moe_intermediate_size")?,
            vocab: num("vocab_size")?,
            rms_eps: flt("rms_norm_eps")?,
            rope_theta: flt("rope_theta")?,
            norm_topk_prob: v
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            eos_token_ids: eos_ids(v.get("eos_token_id")),
        };
        if !v
            .get("mlp_only_layers")
            .and_then(|x| x.as_array())
            .is_none_or(|a| a.is_empty())
        {
            return Err(
                "config.json: mlp_only_layers is not empty; every layer must be MoE".into(),
            );
        }
        if v.get("decoder_sparse_step").and_then(|x| x.as_u64()) != Some(1) {
            return Err("config.json: decoder_sparse_step must be 1 (every layer MoE)".into());
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Stage A limits: block-128 quantisation, head_dim 128 attention, one resident tensor per
    /// projection within the engine's K and N limits, top-k within one window call.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| Err(format!("config refused: {what}"));
        if self.layers == 0 || self.experts == 0 || self.top_k == 0 || self.vocab == 0 {
            return bad("zero layers, experts, top_k or vocab");
        }
        if !self.hidden.is_multiple_of(FP8_BLOCK)
            || !self.moe_intermediate.is_multiple_of(FP8_BLOCK)
        {
            return bad("hidden and moe_intermediate must be multiples of 128 (FP8 block)");
        }
        if self.head_dim != 128 || !self.q_heads.is_multiple_of(self.kv_heads) || self.q_heads > 64
        {
            return bad(
                "attention must be head_dim 128, q_heads a multiple of kv_heads, q_heads <= 64",
            );
        }
        if self.top_k > self.experts || self.top_k > aien_omega_moe::MAX_WINDOWS {
            return bad("top_k must be <= experts and <= one window call");
        }
        if !self.norm_topk_prob {
            return bad("norm_topk_prob=false is not implemented (Qwen3-MoE uses true)");
        }
        if self.tie_word_embeddings {
            return bad("tie_word_embeddings=true is not implemented");
        }
        for (what, k, n) in [
            ("q|k|v", self.hidden, self.qkv_n()),
            ("o", self.q_dim(), self.hidden),
            ("gate|up", self.hidden, self.gate_up_n()),
            ("down", self.moe_intermediate, self.down_n()),
            ("lm_head", self.hidden, self.vocab),
        ] {
            if k > MAX_RESIDENT_K || n > MAX_RESIDENT_N {
                return Err(format!(
                    "config refused: {what} tensor {k}x{n} exceeds the engine limit {MAX_RESIDENT_K}x{MAX_RESIDENT_N}"
                ));
            }
        }
        if !(2 * self.moe_intermediate).is_multiple_of(WINDOW_TILE)
            || !self.hidden.is_multiple_of(WINDOW_TILE)
        {
            return bad("expert windows must start on a column tile of 8");
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.q_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// Fused q|k|v output width.
    pub fn qkv_n(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    /// Width of one expert's gate|up window.
    pub fn gate_up_cols(&self) -> usize {
        2 * self.moe_intermediate
    }
    /// All experts' gate|up side by side.
    pub fn gate_up_n(&self) -> usize {
        self.experts * self.gate_up_cols()
    }
    /// Width of one expert's down window.
    pub fn down_cols(&self) -> usize {
        self.hidden
    }
    pub fn down_n(&self) -> usize {
        self.experts * self.down_cols()
    }
    /// First column of expert `e` in the gate|up tensor (gate first, then up).
    pub fn gate_up_col0(&self, expert: usize) -> usize {
        expert * self.gate_up_cols()
    }
    pub fn down_col0(&self, expert: usize) -> usize {
        expert * self.down_cols()
    }

    /// Bytes of one layer's four resident bf16 tensors.
    pub fn layer_resident_bytes(&self) -> usize {
        2 * (self.hidden * self.qkv_n()
            + self.q_dim() * self.hidden
            + self.hidden * self.gate_up_n()
            + self.moe_intermediate * self.down_n())
    }
    /// Bytes resident for the whole model (layers plus lm_head).
    pub fn model_resident_bytes(&self) -> usize {
        self.layers * self.layer_resident_bytes() + 2 * self.hidden * self.vocab
    }
    /// Bytes the engine reads per generated token (attention + top_k experts + lm_head), bf16.
    pub fn bytes_per_token_bf16(&self) -> usize {
        2 * self.layers * (self.hidden * self.qkv_n() + self.q_dim() * self.hidden)
            + 2 * self.layers
                * self.top_k
                * (self.hidden * self.gate_up_cols() + self.moe_intermediate * self.down_cols())
            + 2 * self.hidden * self.vocab
    }
}

fn eos_ids(v: Option<&serde_json::Value>) -> Vec<u32> {
    match v {
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Where the tensors come from: the checkpoint shards, or a fake in tests.
// ---------------------------------------------------------------------------

/// Reads one raw tensor (dtype and shape checked) into `dst`.
pub trait TensorSource {
    fn read_into(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        dst: &mut [u8],
    ) -> Result<(), String>;

    /// The dtype the checkpoint records for `name` (`"F8_E4M3"`, `"BF16"`, ...), `None` when the
    /// checkpoint has no such tensor. A source that cannot tell refuses.
    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        Err(format!("{name}: this tensor source cannot report dtypes"))
    }
}

/// The safetensors shards behind `model.safetensors.index.json` (the same validated reader as
/// the MoE capsule loader: dtype, shape and byte range are checked per tensor).
pub struct ShardSource {
    dir: PathBuf,
    weight_map: serde_json::Map<String, serde_json::Value>,
    shards: HashMap<String, Shard>,
}

impl ShardSource {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let index_path = dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&index_path).map_err(|e| format!("{}: {e}", index_path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .cloned()
            .ok_or_else(|| "index: missing weight_map object".to_string())?;
        Ok(Self {
            dir: dir.to_path_buf(),
            weight_map,
            shards: HashMap::new(),
        })
    }
}

impl ShardSource {
    /// The opened shard the index names for `name`, `None` when the index has no entry.
    fn shard_for(&mut self, name: &str) -> Result<Option<&Shard>, String> {
        let Some(filename) = self.weight_map.get(name) else {
            return Ok(None);
        };
        let filename = filename
            .as_str()
            .ok_or_else(|| format!("index: shard of {name} is not a string"))?;
        if !self.shards.contains_key(filename) {
            let path = shard_path(&self.dir, filename).map_err(|e| e.to_string())?;
            let shard = Shard::open(&path).map_err(|e| e.to_string())?;
            self.shards.insert(filename.to_string(), shard);
        }
        Ok(Some(&self.shards[filename]))
    }
}

impl TensorSource for ShardSource {
    fn read_into(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        dst: &mut [u8],
    ) -> Result<(), String> {
        self.shard_for(name)?
            .ok_or_else(|| format!("index: no shard for {name}"))?
            .read(name, dtype, shape, dst)
            .map_err(|e| e.to_string())
    }

    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        match self.shard_for(name)? {
            None => Ok(None),
            Some(shard) => shard
                .dtype(name)
                .map(|d| Some(d.to_string()))
                .ok_or_else(|| format!("{name}: in the index but not in its shard's header")),
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 block-128 -> bf16, written transposed into column windows.
// ---------------------------------------------------------------------------

/// One `[rows][k]` FP8 matrix with its `[ceil(rows/128)][k/128]` bf16 inverse scales, and the
/// column of the destination where its rows land (row `r` becomes column `col0 + r`).
pub struct Fp8Source<'a> {
    pub weights: &'a [u8],
    pub scales: &'a [u16],
    pub rows: usize,
    pub col0: usize,
}

/// Dequantises every source exactly (`fp8_e4m3fn_to_f32(w) * bf16(scale)`), rounds to bf16
/// (round-to-nearest-even, as the engine's own upload does) and writes `dst[k][col0 + r]`,
/// `dst` being `[k_dim][dst_n]` row-major: the transposed `[in][out]` layout a resident tensor
/// takes. The e4m3 NaN codes (0x7f, 0xff) are refused: a weight is never NaN, so one in the
/// file means a corrupt or foreign checkpoint.
pub fn dequant_fp8_sources_to_bf16_cols(
    sources: &[Fp8Source<'_>],
    k_dim: usize,
    dst: &mut [u16],
    dst_n: usize,
) -> Result<(), String> {
    if k_dim == 0 || !k_dim.is_multiple_of(FP8_BLOCK) {
        return Err(format!(
            "k_dim {k_dim} must be a positive multiple of {FP8_BLOCK}"
        ));
    }
    if dst.len() != k_dim * dst_n {
        return Err(format!(
            "dst holds {} values, {k_dim}x{dst_n} expected",
            dst.len()
        ));
    }
    let k_blocks = k_dim / FP8_BLOCK;
    for (i, s) in sources.iter().enumerate() {
        if s.weights.len() != s.rows * k_dim {
            return Err(format!(
                "source {i}: {} weight bytes, {}x{k_dim} expected",
                s.weights.len(),
                s.rows
            ));
        }
        let want = s.rows.div_ceil(FP8_BLOCK) * k_blocks;
        if s.scales.len() != want {
            return Err(format!(
                "source {i}: {} scales, {want} expected",
                s.scales.len()
            ));
        }
        if s.col0 + s.rows > dst_n {
            return Err(format!(
                "source {i}: columns {}..{} past dst_n {dst_n}",
                s.col0,
                s.col0 + s.rows
            ));
        }
    }
    let lut = fp8_table();
    let nan_seen = std::sync::atomic::AtomicBool::new(false);
    // Each task owns one 128-deep band of k (one scale block), all destination columns.
    dst.par_chunks_mut(dst_n * FP8_BLOCK)
        .enumerate()
        .for_each(|(kb, band)| {
            let k0 = kb * FP8_BLOCK;
            for s in sources {
                for r in 0..s.rows {
                    let scale = bf16_to_f32(s.scales[(r / FP8_BLOCK) * k_blocks + kb]);
                    let row = &s.weights[r * k_dim + k0..r * k_dim + k0 + FP8_BLOCK];
                    let col = s.col0 + r;
                    for (j, &w) in row.iter().enumerate() {
                        if w & 0x7f == 0x7f {
                            nan_seen.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        band[j * dst_n + col] = f32_to_bf16(lut[w as usize] * scale);
                    }
                }
            }
        });
    if nan_seen.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("FP8 weight holds an e4m3 NaN code (0x7f/0xff): refused".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The device side: resident matmuls, uploaded through a trait so tests need no chip.
// ---------------------------------------------------------------------------

/// A resident `k x n` weight used whole: `c[m x n] = a[m x k] * W`. Returns chip nanoseconds.
pub trait DenseMatmul: Send + Sync + std::fmt::Debug {
    fn k(&self) -> usize;
    fn n(&self) -> usize;
    fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String>;
}

/// A resident `k x n` tensor of experts side by side, used by column windows: for each window
/// `w`, `c[w][m x cols] = a[m x k] * W[:, col0s[w] .. col0s[w] + cols)`. Returns chip nanoseconds.
pub trait ExpertMatmul: Send + Sync + std::fmt::Debug {
    fn k(&self) -> usize;
    fn n(&self) -> usize;
    fn windows_f32(
        &self,
        m: usize,
        a: &[f32],
        col0s: &[u32],
        cols: usize,
        c: &mut [f32],
    ) -> Result<u64, String>;
}

/// `experts` resident FP8 projections of `[n][k]` with block-128 scales (the checkpoint bytes as
/// they are). `grouped_f32(ids, x, y)`: one launch for `ids.len()` experts; `x` is `k` values
/// shared by every group or `ids.len() * k` (one row per group); `y` is `ids.len() * n`,
/// group-major. A single projection is `ids == [0]`. Returns chip nanoseconds.
pub trait Fp8Gemv: Send + Sync + std::fmt::Debug {
    fn experts(&self) -> usize;
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    fn grouped_f32(&self, ids: &[u32], x: &[f32], y: &mut [f32]) -> Result<u64, String>;
    /// Device addresses `(weights, scales)` of one expert (for the router's directory).
    fn tensor_entry(&self, expert: usize) -> Result<(u64, u64), String> {
        let _ = expert;
        Err("this FP8 tensor has no device addresses".into())
    }
    /// `groups` experts in one launch whose group table and single activation row are already
    /// on the device (written by the [`DeviceRouter`] launch). Returns chip nanoseconds.
    fn grouped_dev(
        &self,
        groups: usize,
        tab_va: u64,
        x_va: u64,
        y: &mut [f32],
    ) -> Result<u64, String> {
        let _ = (groups, tab_va, x_va, y);
        Err("this FP8 tensor takes no device-resident group table".into())
    }
    /// Whether [`Self::gemm_f32`] (the prefill grouped GEMM) is available on this tensor.
    fn has_gemm(&self) -> bool {
        false
    }
    /// Prefill grouped GEMM, one launch: row `i` computes expert `expert_ids[i]` on activation
    /// row `token_rows[i]` of `x` (`x.len() / k` rows); `expert_ids` is sorted (non-decreasing);
    /// `y` is `rows * n` in row-table order and equals one GEMV per row. Returns chip nanoseconds.
    fn gemm_f32(
        &self,
        expert_ids: &[u32],
        token_rows: &[u32],
        x: &[f32],
        y: &mut [f32],
    ) -> Result<u64, String> {
        let _ = (expert_ids, token_rows, x, y);
        Err("this FP8 tensor has no prefill GEMM".into())
    }
    /// The omega C handle (`OmegaGpuFp8Tensor *`) for the one-list decode graph; null when this
    /// tensor is not an omega FP8 tensor.
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
}

/// Device addresses the router left its outputs at (valid until the next `route`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteAddrs {
    /// `x' = RMSNorm(x) * 2^64`, f32 `[hidden]`: the gate|up launch's activation row.
    pub x_va: u64,
    /// The gate|up group table (`top_k` entries of `{w_va, s_va}`).
    pub tab_gu_va: u64,
}

/// The MoE router as one device launch: RMSNorm of the residual row, router logits, softmax,
/// top-k and renormalised weights, with the gate|up group table and the normed row left on the
/// device for the grouped FP8 GEMV (no host round trip for either).
pub trait DeviceRouter: Send + Sync + std::fmt::Debug {
    /// The layer's `mlp.gate.weight` as bf16 bits `[experts][hidden]` and its
    /// `post_attention_layernorm` weight.
    fn set_layer(&self, layer: usize, gate_bf16: &[u16], norm_w: &[f32]) -> Result<(), String>;
    /// `{w_va, s_va}` per expert of the layer's gate|up and down FP8 tensors (flat, 2 words per expert).
    fn set_directory(&self, layer: usize, gu: &[u64], dn: &[u64]) -> Result<(), String>;
    /// Routes the residual row `x`: writes `top_k` ids (descending by logit) and weights, returns
    /// chip nanoseconds and where the device outputs are.
    fn route(
        &self,
        layer: usize,
        x: &[f32],
        eps: f32,
        ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<(u64, RouteAddrs), String>;
    /// Trace helper: the last call's `RMSNorm(x)` read back from the device (undoing the 2^64).
    fn read_post(&self, out: &mut [f32]) -> Result<(), String>;
    /// The omega C handle (`OmegaGpuRouter *`) for the one-list decode graph; null when none.
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
}

/// The lm_head as one streaming launch over the checkpoint's `[vocab][hidden]` bf16 rows.
pub trait LmHeadGemv: Send + Sync + std::fmt::Debug {
    fn logits_f32(&self, hidden: &[f32], out: &mut [f32]) -> Result<u64, String>;
    /// The omega C handle (`OmegaGpuBf16Tensor *`) for the one-list decode graph; null when none.
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
}

/// The lm_head in the form the loader chose.
#[derive(Debug, Clone)]
pub enum LmHead {
    /// `[hidden][vocab]` bf16 through the generic matmul (Stage A / cut 7 path).
    Dense(Arc<dyn DenseMatmul>),
    /// `[vocab][hidden]` bf16 through the streaming GEMV.
    Gemv(Arc<dyn LmHeadGemv>),
}

/// Which implementation to use for a step the device can take over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathChoice {
    /// The device one when the uploader offers it, else the host/dense one.
    #[default]
    Auto,
    /// Always the host/dense one.
    Host,
    /// The device one; the load fails when the uploader cannot offer it.
    Device,
}

/// Router and lm_head choices for a load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DevicePaths {
    pub router: PathChoice,
    pub lm_head: PathChoice,
}

/// How the loader hands weights to the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Everything dequantised to bf16 (Stage A): about 60.4 GB resident.
    Bf16,
    /// Experts as FP8 with the checkpoint scales (grouped GEMV, one launch per projection per
    /// layer); attention and lm_head bf16. About 31.5 GB resident.
    Fp8Experts,
    /// Experts and the q/k/v/o projections as FP8; lm_head bf16. About 30.6 GB resident.
    Fp8All,
}

impl WeightFormat {
    pub fn fp8_experts(self) -> bool {
        self != WeightFormat::Bf16
    }
    pub fn fp8_attention(self) -> bool {
        self == WeightFormat::Fp8All
    }
}

/// Makes `[k][n]` bf16 matrices resident. The real one is [`OmegaModelUploader`].
pub trait ModelUploader {
    fn upload_dense(
        &self,
        k: usize,
        n: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseMatmul>, String>;
    fn upload_experts(
        &self,
        k: usize,
        n: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn ExpertMatmul>, String>;
    /// FP8 projections as in the checkpoint: `w` is `experts * n * k` e4m3 bytes `[expert][n][k]`,
    /// `scales` is `experts * (n/128) * (k/128)` bf16 bits. Uploaders without FP8 refuse.
    fn upload_fp8(
        &self,
        experts: usize,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn Fp8Gemv>, String> {
        let _ = (experts, n, k, w, scales);
        Err("this uploader has no FP8 path".into())
    }
    /// The device router for `config` (its kernel is fixed at 128 experts, hidden 2048, top-8),
    /// or `None` when this uploader has none.
    fn new_router(&self, config: &Qwen3MoeConfig) -> Result<Option<Arc<dyn DeviceRouter>>, String> {
        let _ = config;
        Ok(None)
    }
    /// The lm_head as a streaming GEMV over `bits` = `[vocab = n][hidden = k]` bf16 (the
    /// checkpoint layout), or `None` when this uploader has none.
    fn upload_lm_head_gemv(
        &self,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Option<Arc<dyn LmHeadGemv>>, String> {
        let _ = (n, k, bits);
        Ok(None)
    }
}

/// Omega's whole-tensor resident matmul behind a lock.
#[derive(Debug)]
pub struct OmegaDense(Mutex<aien_omega_gpu::ResidentTensor>, usize, usize);

impl DenseMatmul for OmegaDense {
    fn k(&self) -> usize {
        self.1
    }
    fn n(&self) -> usize {
        self.2
    }
    fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "resident tensor lock poisoned".to_string())?;
        t.matmul_f32(m, a, c)
            .map(|i| i.raw.elapsed_ns)
            .map_err(|e| e.to_string())
    }
}

/// Omega's column-window matmul (cut 2) behind a lock.
#[derive(Debug)]
pub struct OmegaExperts(Mutex<aien_omega_moe::MoeTensor>, usize, usize);

impl ExpertMatmul for OmegaExperts {
    fn k(&self) -> usize {
        self.1
    }
    fn n(&self) -> usize {
        self.2
    }
    fn windows_f32(
        &self,
        m: usize,
        a: &[f32],
        col0s: &[u32],
        cols: usize,
        c: &mut [f32],
    ) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "expert tensor lock poisoned".to_string())?;
        t.windows_f32(m, a, col0s, cols, c)
            .map(|i| i.raw.elapsed_ns)
            .map_err(|e| e.to_string())
    }
}

/// Fallback when the window API is not linked: the whole-tensor product, windows copied out.
/// Correct, and far slower (every expert is computed): only for bring-up without cut 2.
#[derive(Debug)]
pub struct WholeTensorExperts {
    dense: OmegaDense,
    scratch: Mutex<Vec<f32>>,
}

impl ExpertMatmul for WholeTensorExperts {
    fn k(&self) -> usize {
        self.dense.k()
    }
    fn n(&self) -> usize {
        self.dense.n()
    }
    fn windows_f32(
        &self,
        m: usize,
        a: &[f32],
        col0s: &[u32],
        cols: usize,
        c: &mut [f32],
    ) -> Result<u64, String> {
        let n = self.n();
        let mut full = self
            .scratch
            .lock()
            .map_err(|_| "scratch lock poisoned".to_string())?;
        full.resize(m * n, 0.0);
        let ns = self.dense.matmul_f32(m, a, &mut full)?;
        copy_windows_out(&full, m, n, col0s, cols, c)?;
        Ok(ns)
    }
}

/// Copies `col0s.len()` column windows of a `[m][n]` product into `c` (window-major, `[m][cols]`
/// each). Shared by the fallback and the test fake.
pub fn copy_windows_out(
    full: &[f32],
    m: usize,
    n: usize,
    col0s: &[u32],
    cols: usize,
    c: &mut [f32],
) -> Result<(), String> {
    if c.len() != col0s.len() * m * cols {
        return Err(format!(
            "window output holds {} values, {} expected",
            c.len(),
            col0s.len() * m * cols
        ));
    }
    for (w, &col0) in col0s.iter().enumerate() {
        aien_omega_moe::check_window(n, col0 as usize, cols).map_err(|e| e.to_string())?;
        let col0 = col0 as usize;
        for r in 0..m {
            let src = &full[r * n + col0..r * n + col0 + cols];
            c[(w * m + r) * cols..(w * m + r + 1) * cols].copy_from_slice(src);
        }
    }
    Ok(())
}

/// The real uploader: Omega resident tensors; experts through the window API when it is linked,
/// the whole-tensor fallback otherwise (`uses_window_api` says which).
pub struct OmegaModelUploader;

/// Which attention path the engine runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnPath {
    /// Fused when omega's decode attention is linked and the model has its shape, else split.
    Auto,
    /// Host norms and rope, paged bf16 KV, omega's paged attention launch.
    Split,
    /// omega's one-launch decode attention (refused when not linked).
    Fused,
}

impl OmegaModelUploader {
    /// True when omega's one-launch decode attention (OWN-30B cut 8/9) is linked.
    pub fn uses_dattn_api() -> bool {
        aien_omega_dattn::is_native()
    }
    pub fn uses_window_api() -> bool {
        aien_omega_moe::is_native()
    }
    pub fn uses_fp8_api() -> bool {
        aien_omega_fp8::is_native()
    }
    /// True when the one-launch MoE router and the FP8 device-table entry points are linked.
    pub fn uses_router_api() -> bool {
        aien_omega_router::is_native() && aien_omega_fp8::has_device_tables()
    }
    /// True when the bf16 streaming GEMV (the lm_head kernel) is linked.
    pub fn uses_bf16_api() -> bool {
        aien_omega_bf16::is_native()
    }
}

impl ModelUploader for OmegaModelUploader {
    fn upload_dense(
        &self,
        k: usize,
        n: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseMatmul>, String> {
        let t =
            aien_omega_gpu::ResidentTensor::upload_bf16(k, n, bits).map_err(|e| e.to_string())?;
        Ok(Arc::new(OmegaDense(Mutex::new(t), k, n)))
    }
    fn upload_experts(
        &self,
        k: usize,
        n: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn ExpertMatmul>, String> {
        if Self::uses_window_api() {
            let t =
                aien_omega_moe::MoeTensor::upload_bf16(k, n, bits).map_err(|e| e.to_string())?;
            Ok(Arc::new(OmegaExperts(Mutex::new(t), k, n)))
        } else {
            let t = aien_omega_gpu::ResidentTensor::upload_bf16(k, n, bits)
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(WholeTensorExperts {
                dense: OmegaDense(Mutex::new(t), k, n),
                scratch: Mutex::new(Vec::new()),
            }))
        }
    }
    fn upload_fp8(
        &self,
        experts: usize,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn Fp8Gemv>, String> {
        if !Self::uses_fp8_api() {
            return Err("this binary carries no FP8 GEMV (omega before OWN-30B cut 5, or AIEN_OMEGA_GPU_LIB without it)".into());
        }
        let t = aien_omega_fp8::Fp8Tensor::upload(experts, n, k, w, scales)
            .map_err(|e| e.to_string())?;
        Ok(Arc::new(OmegaFp8(Mutex::new(t), experts, n, k)))
    }
    fn new_router(&self, config: &Qwen3MoeConfig) -> Result<Option<Arc<dyn DeviceRouter>>, String> {
        if !Self::uses_router_api() {
            return Ok(None);
        }
        if config.experts != aien_omega_router::EXPERTS
            || config.hidden != aien_omega_router::HIDDEN
            || config.top_k != aien_omega_router::TOP_K
        {
            return Ok(None);
        }
        let r = aien_omega_router::Router::alloc(config.layers).map_err(|e| e.to_string())?;
        Ok(Some(Arc::new(OmegaRouter(Mutex::new(r)))))
    }
    fn upload_lm_head_gemv(
        &self,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Option<Arc<dyn LmHeadGemv>>, String> {
        if !Self::uses_bf16_api() {
            return Ok(None);
        }
        let t = aien_omega_bf16::Bf16Tensor::upload(n, k, bits).map_err(|e| e.to_string())?;
        Ok(Some(Arc::new(OmegaLmHead(Mutex::new(t)))))
    }
}

/// Omega's FP8 grouped GEMV behind a lock.
#[derive(Debug)]
pub struct OmegaFp8(Mutex<aien_omega_fp8::Fp8Tensor>, usize, usize, usize);

impl Fp8Gemv for OmegaFp8 {
    fn experts(&self) -> usize {
        self.1
    }
    fn n(&self) -> usize {
        self.2
    }
    fn k(&self) -> usize {
        self.3
    }
    fn grouped_f32(&self, ids: &[u32], x: &[f32], y: &mut [f32]) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "fp8 tensor lock poisoned".to_string())?;
        let info = t.gemv_grouped_f32(ids, x, y).map_err(|e| e.to_string())?;
        Ok(info.elapsed_ns)
    }
    fn tensor_entry(&self, expert: usize) -> Result<(u64, u64), String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "fp8 tensor lock poisoned".to_string())?;
        t.entry(expert).map_err(|e| e.to_string())
    }
    fn grouped_dev(
        &self,
        groups: usize,
        tab_va: u64,
        x_va: u64,
        y: &mut [f32],
    ) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "fp8 tensor lock poisoned".to_string())?;
        let info = t
            .gemv_grouped_dev(groups, tab_va, x_va, y)
            .map_err(|e| e.to_string())?;
        Ok(info.elapsed_ns)
    }
    fn has_gemm(&self) -> bool {
        aien_omega_fp8::has_gemm()
    }
    fn gemm_f32(
        &self,
        expert_ids: &[u32],
        token_rows: &[u32],
        x: &[f32],
        y: &mut [f32],
    ) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "fp8 tensor lock poisoned".to_string())?;
        let info = t
            .gemm_f32(expert_ids, token_rows, x, y)
            .map_err(|e| e.to_string())?;
        Ok(info.elapsed_ns)
    }
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        self.0.lock().map_or(std::ptr::null_mut(), |t| t.as_raw())
    }
}

/// Omega's one-launch MoE router behind a lock.
#[derive(Debug)]
pub struct OmegaRouter(Mutex<aien_omega_router::Router>);

impl DeviceRouter for OmegaRouter {
    fn set_layer(&self, layer: usize, gate_bf16: &[u16], norm_w: &[f32]) -> Result<(), String> {
        let mut r = self
            .0
            .lock()
            .map_err(|_| "router lock poisoned".to_string())?;
        r.set_layer(layer, gate_bf16, norm_w)
            .map_err(|e| e.to_string())
    }
    fn set_directory(&self, layer: usize, gu: &[u64], dn: &[u64]) -> Result<(), String> {
        let mut r = self
            .0
            .lock()
            .map_err(|_| "router lock poisoned".to_string())?;
        r.set_directory(layer, gu, dn).map_err(|e| e.to_string())
    }
    fn route(
        &self,
        layer: usize,
        x: &[f32],
        eps: f32,
        ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<(u64, RouteAddrs), String> {
        let mut r = self
            .0
            .lock()
            .map_err(|_| "router lock poisoned".to_string())?;
        // norm_topk_prob is true (the config check refuses false); out_scale 2^64 gives the FP8
        // GEMV its activation row.
        let info = r
            .route(
                layer,
                x,
                eps,
                true,
                aien_omega_router::OUT_SCALE_FP8,
                ids,
                weights,
            )
            .map_err(|e| e.to_string())?;
        let o = r.device_outputs();
        Ok((
            info.elapsed_ns,
            RouteAddrs {
                x_va: o.xout_va,
                tab_gu_va: o.tab_gu_va,
            },
        ))
    }
    fn read_post(&self, out: &mut [f32]) -> Result<(), String> {
        let mut r = self
            .0
            .lock()
            .map_err(|_| "router lock poisoned".to_string())?;
        r.read_xout(out).map_err(|e| e.to_string())?;
        for v in out.iter_mut() {
            *v /= aien_omega_router::OUT_SCALE_FP8;
        }
        Ok(())
    }
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        self.0.lock().map_or(std::ptr::null_mut(), |r| r.as_raw())
    }
}

/// Omega's bf16 streaming GEMV (the lm_head) behind a lock.
#[derive(Debug)]
pub struct OmegaLmHead(Mutex<aien_omega_bf16::Bf16Tensor>);

impl LmHeadGemv for OmegaLmHead {
    fn logits_f32(&self, hidden: &[f32], out: &mut [f32]) -> Result<u64, String> {
        let t = self
            .0
            .lock()
            .map_err(|_| "lm_head lock poisoned".to_string())?;
        let info = t.gemv_f32(hidden, out).map_err(|e| e.to_string())?;
        Ok(info.elapsed_ns)
    }
    fn omega_raw(&self) -> *mut std::ffi::c_void {
        self.0.lock().map_or(std::ptr::null_mut(), |t| t.as_raw())
    }
}

// ---------------------------------------------------------------------------
// The loaded model.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct OmegaLayer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Router `[experts][hidden]` f32.
    pub router: Vec<f32>,
    pub attn: AttnWeights,
    pub experts: ExpertWeights,
}

/// The q|k|v and o projections of one layer, in the format the loader chose.
#[derive(Debug)]
pub enum AttnWeights {
    Bf16 {
        qkv: Arc<dyn DenseMatmul>,
        o: Arc<dyn DenseMatmul>,
    },
    Fp8 {
        qkv: Arc<dyn Fp8Gemv>,
        o: Arc<dyn Fp8Gemv>,
    },
}

/// The expert projections of one layer, in the format the loader chose.
#[derive(Debug)]
pub enum ExpertWeights {
    Bf16 {
        gate_up: Arc<dyn ExpertMatmul>,
        down: Arc<dyn ExpertMatmul>,
    },
    Fp8 {
        gate_up: Arc<dyn Fp8Gemv>,
        down: Arc<dyn Fp8Gemv>,
    },
}

/// The whole model: resident matmuls on the device, bf16 embedding and f32 norms/routers on the host.
pub struct Qwen3MoeOmegaWeights {
    pub config: Qwen3MoeConfig,
    /// `[vocab][hidden]` bf16 bits; a row is widened to f32 at lookup.
    pub embed_bf16: Vec<u16>,
    pub layers: Vec<OmegaLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: LmHead,
    /// The one-launch router, set up for every layer (None: host routing only).
    pub device_router: Option<Arc<dyn DeviceRouter>>,
}

impl Qwen3MoeOmegaWeights {
    pub fn embed_row(&self, token: u32) -> Vec<f32> {
        let h = self.config.hidden;
        let t = (token as usize) % self.config.vocab;
        self.embed_bf16[t * h..(t + 1) * h]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }

    /// Bytes the host keeps after the load (embedding, norms, routers).
    pub fn host_bytes(&self) -> usize {
        self.embed_bf16.len() * 2
            + self.final_norm.len() * 4
            + self
                .layers
                .iter()
                .map(|l| {
                    4 * (l.input_norm.len()
                        + l.post_norm.len()
                        + l.q_norm.len()
                        + l.k_norm.len()
                        + l.router.len())
                })
                .sum::<usize>()
    }
}

/// What a load measured.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadReport {
    pub resident_bytes: usize,
    pub format: WeightFormat,
    pub host_bytes: usize,
    /// Process peak resident set (VmHWM) after the load, if `/proc` is readable.
    pub peak_rss_bytes: Option<u64>,
    pub peak_rss_before_bytes: Option<u64>,
    pub seconds: f64,
    pub window_api: bool,
}

/// Peak resident set size of this process (`VmHWM` in `/proc/self/status`), bytes.
pub fn peak_rss_bytes() -> Option<u64> {
    proc_status_kb("VmHWM:").map(|kb| kb * 1024)
}

/// Current resident set size (`VmRSS`), bytes.
pub fn current_rss_bytes() -> Option<u64> {
    proc_status_kb("VmRSS:").map(|kb| kb * 1024)
}

fn proc_status_kb(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// Loads a checkpoint directory: `config.json` for the shape, the shards for the weights.
pub fn load_qwen3_moe_omega(
    dir: &Path,
    uploader: &dyn ModelUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    load_qwen3_moe_omega_with(dir, uploader, WeightFormat::Bf16, progress)
}

/// As [`load_qwen3_moe_omega`] with a chosen [`WeightFormat`].
pub fn load_qwen3_moe_omega_with(
    dir: &Path,
    uploader: &dyn ModelUploader,
    format: WeightFormat,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    load_qwen3_moe_omega_paths(dir, uploader, format, DevicePaths::default(), progress)
}

/// As [`load_qwen3_moe_omega_with`] with the router and lm_head paths chosen.
pub fn load_qwen3_moe_omega_paths(
    dir: &Path,
    uploader: &dyn ModelUploader,
    format: WeightFormat,
    paths: DevicePaths,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    let config = Qwen3MoeConfig::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    load_qwen3_moe_omega_from_paths(&config, &mut source, uploader, format, paths, progress)
}

/// The loader proper, over any tensor source (tests use an in-memory one).
pub fn load_qwen3_moe_omega_from(
    config: &Qwen3MoeConfig,
    source: &mut dyn TensorSource,
    uploader: &dyn ModelUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    load_qwen3_moe_omega_from_with(config, source, uploader, WeightFormat::Bf16, progress)
}

/// The loader proper with a chosen [`WeightFormat`]. FP8 projections go up as the checkpoint
/// bytes (`[n][k]` per expert, scales `[n/128][k/128]`), bf16 ones through the exact dequant.
pub fn load_qwen3_moe_omega_from_with(
    config: &Qwen3MoeConfig,
    source: &mut dyn TensorSource,
    uploader: &dyn ModelUploader,
    format: WeightFormat,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    load_qwen3_moe_omega_from_paths(
        config,
        source,
        uploader,
        format,
        DevicePaths::default(),
        progress,
    )
}

/// As [`load_qwen3_moe_omega_from_with`] with the router and lm_head implementations chosen:
/// [`PathChoice::Auto`] takes the device router / streaming lm_head GEMV when the uploader
/// offers them (the router also needs FP8 experts), [`PathChoice::Host`] keeps host routing
/// and the dense lm_head matmul, [`PathChoice::Device`] fails the load when unavailable.
pub fn load_qwen3_moe_omega_from_paths(
    config: &Qwen3MoeConfig,
    source: &mut dyn TensorSource,
    uploader: &dyn ModelUploader,
    format: WeightFormat,
    paths: DevicePaths,
    progress: &mut dyn FnMut(&str),
) -> Result<(Qwen3MoeOmegaWeights, LoadReport), String> {
    config.validate()?;
    let device_router: Option<Arc<dyn DeviceRouter>> = if paths.router == PathChoice::Host {
        None
    } else if !format.fp8_experts() {
        if paths.router == PathChoice::Device {
            return Err(
                "the device router needs FP8 experts (--weights fp8-experts or fp8)".into(),
            );
        }
        None
    } else {
        let r = uploader.new_router(config)?;
        if r.is_none() && paths.router == PathChoice::Device {
            return Err("the device router was asked for but is not available (omega before the router cut, or a model shape the kernel does not take)".into());
        }
        r
    };
    let mut resident = 0usize;
    let t0 = Instant::now();
    let peak_before = peak_rss_bytes();
    let (h, i, e) = (config.hidden, config.moe_intermediate, config.experts);
    let (qd, kvd, hd) = (config.q_dim(), config.kv_dim(), config.head_dim);
    let kb = h / FP8_BLOCK;
    let ib = i / FP8_BLOCK;
    let qkv_n = config.qkv_n();

    // Transient host buffers, sized once for the largest use and reused for every layer.
    let mut bits: Vec<u16> = vec![0; h * config.gate_up_n().max(qkv_n).max(qd)];
    let mut fp8: Vec<u8> = vec![0; (e * 2 * i * h).max(qd * h + 2 * kvd * h)];
    let mut scales: Vec<u16> =
        vec![0; (e * 2 * ib * kb).max((qd / FP8_BLOCK + 2 * (kvd / FP8_BLOCK)) * kb)];
    let mut small: Vec<u8> = Vec::new();

    let read_bf16_f32 = |source: &mut dyn TensorSource,
                         name: &str,
                         shape: &[usize],
                         small: &mut Vec<u8>|
     -> Result<Vec<f32>, String> {
        let n: usize = shape.iter().product();
        small.resize(n * 2, 0);
        source.read_into(name, "BF16", shape, small)?;
        Ok(small
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    };

    let mut layers = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        // --- fused q|k|v: sources q [qd][h], k [kvd][h], v [kvd][h] at columns 0, qd, qd+kvd ---
        let specs = [
            ("q_proj", qd, 0usize),
            ("k_proj", kvd, qd),
            ("v_proj", kvd, qd + kvd),
        ];
        let mut w_off = 0usize;
        let mut s_off = 0usize;
        let mut ranges = Vec::new();
        for (name, rows, col0) in specs {
            let wl = rows * h;
            let sl = (rows / FP8_BLOCK) * kb;
            source.read_into(
                &format!("{p}.self_attn.{name}.weight"),
                "F8_E4M3",
                &[rows, h],
                &mut fp8[w_off..w_off + wl],
            )?;
            read_scales(
                source,
                &format!("{p}.self_attn.{name}.weight_scale_inv"),
                &[rows / FP8_BLOCK, kb],
                &mut scales[s_off..s_off + sl],
            )?;
            ranges.push((w_off..w_off + wl, s_off..s_off + sl, rows, col0));
            w_off += wl;
            s_off += sl;
        }
        let sources: Vec<Fp8Source> = ranges
            .iter()
            .map(|(w, s, rows, col0)| Fp8Source {
                weights: &fp8[w.clone()],
                scales: &scales[s.clone()],
                rows: *rows,
                col0: *col0,
            })
            .collect();
        let qkv_fp8 = if format.fp8_attention() {
            // q, k, v rows stacked: [qkv_n][h] with scales [qkv_n/128][kb], exactly the chip layout.
            resident += w_off + s_off * 4;
            Some(uploader.upload_fp8(1, qkv_n, h, &fp8[..w_off], &scales[..s_off])?)
        } else {
            None
        };
        let qkv = if qkv_fp8.is_none() {
            dequant_fp8_sources_to_bf16_cols(&sources, h, &mut bits[..h * qkv_n], qkv_n)?;
            resident += 2 * h * qkv_n;
            Some(uploader.upload_dense(h, qkv_n, &bits[..h * qkv_n])?)
        } else {
            None
        };
        progress(&format!("{p}.self_attn.qkv"));

        // --- o: [h][qd] -> dst [qd][h] ---
        source.read_into(
            &format!("{p}.self_attn.o_proj.weight"),
            "F8_E4M3",
            &[h, qd],
            &mut fp8[..h * qd],
        )?;
        read_scales(
            source,
            &format!("{p}.self_attn.o_proj.weight_scale_inv"),
            &[kb, qd / FP8_BLOCK],
            &mut scales[..kb * (qd / FP8_BLOCK)],
        )?;
        let attn = if format.fp8_attention() {
            resident += h * qd + kb * (qd / FP8_BLOCK) * 4;
            let o =
                uploader.upload_fp8(1, h, qd, &fp8[..h * qd], &scales[..kb * (qd / FP8_BLOCK)])?;
            AttnWeights::Fp8 {
                qkv: qkv_fp8.expect("fp8 qkv"),
                o,
            }
        } else {
            dequant_fp8_sources_to_bf16_cols(
                &[Fp8Source {
                    weights: &fp8[..h * qd],
                    scales: &scales[..kb * (qd / FP8_BLOCK)],
                    rows: h,
                    col0: 0,
                }],
                qd,
                &mut bits[..qd * h],
                h,
            )?;
            resident += 2 * qd * h;
            let o = uploader.upload_dense(qd, h, &bits[..qd * h])?;
            AttnWeights::Bf16 {
                qkv: qkv.expect("bf16 qkv"),
                o,
            }
        };
        progress(&format!("{p}.self_attn.o_proj"));

        // --- experts gate|up: per expert gate [i][h] then up [i][h]; columns e*2i (gate), e*2i+i (up) ---
        let gu_n = config.gate_up_n();
        for ex in 0..e {
            for (j, proj) in ["gate_proj", "up_proj"].into_iter().enumerate() {
                let w0 = (ex * 2 + j) * i * h;
                let s0 = (ex * 2 + j) * ib * kb;
                let base = format!("{p}.mlp.experts.{ex}.{proj}");
                source.read_into(
                    &format!("{base}.weight"),
                    "F8_E4M3",
                    &[i, h],
                    &mut fp8[w0..w0 + i * h],
                )?;
                read_scales(
                    source,
                    &format!("{base}.weight_scale_inv"),
                    &[ib, kb],
                    &mut scales[s0..s0 + ib * kb],
                )?;
            }
        }
        let sources: Vec<Fp8Source> = (0..e)
            .flat_map(|ex| (0..2).map(move |j| (ex, j)))
            .map(|(ex, j)| {
                let w0 = (ex * 2 + j) * i * h;
                let s0 = (ex * 2 + j) * ib * kb;
                Fp8Source {
                    weights: &fp8[w0..w0 + i * h],
                    scales: &scales[s0..s0 + ib * kb],
                    rows: i,
                    col0: config.gate_up_col0(ex) + j * i,
                }
            })
            .collect();
        let gate_up_fp8 = if format.fp8_experts() {
            // [e][2i][h] bytes with scales [e][2ib][kb]: the capsule GATE_UP layout, as the chip wants it.
            resident += e * 2 * i * h + e * 2 * ib * kb * 4;
            Some(uploader.upload_fp8(
                e,
                2 * i,
                h,
                &fp8[..e * 2 * i * h],
                &scales[..e * 2 * ib * kb],
            )?)
        } else {
            None
        };
        let gate_up = if gate_up_fp8.is_none() {
            dequant_fp8_sources_to_bf16_cols(&sources, h, &mut bits[..h * gu_n], gu_n)?;
            resident += 2 * h * gu_n;
            Some(uploader.upload_experts(h, gu_n, &bits[..h * gu_n])?)
        } else {
            None
        };
        progress(&format!("{p}.mlp.experts.gate_up"));

        // --- experts down: per expert [h][i]; columns e*h ---
        let dn_n = config.down_n();
        for ex in 0..e {
            let base = format!("{p}.mlp.experts.{ex}.down_proj");
            source.read_into(
                &format!("{base}.weight"),
                "F8_E4M3",
                &[h, i],
                &mut fp8[ex * h * i..(ex + 1) * h * i],
            )?;
            read_scales(
                source,
                &format!("{base}.weight_scale_inv"),
                &[kb, ib],
                &mut scales[ex * kb * ib..(ex + 1) * kb * ib],
            )?;
        }
        let sources: Vec<Fp8Source> = (0..e)
            .map(|ex| Fp8Source {
                weights: &fp8[ex * h * i..(ex + 1) * h * i],
                scales: &scales[ex * kb * ib..(ex + 1) * kb * ib],
                rows: h,
                col0: config.down_col0(ex),
            })
            .collect();
        let experts = if format.fp8_experts() {
            resident += e * h * i + e * kb * ib * 4;
            let down = uploader.upload_fp8(e, h, i, &fp8[..e * h * i], &scales[..e * kb * ib])?;
            ExpertWeights::Fp8 {
                gate_up: gate_up_fp8.expect("fp8 gate_up"),
                down,
            }
        } else {
            dequant_fp8_sources_to_bf16_cols(&sources, i, &mut bits[..i * dn_n], dn_n)?;
            resident += 2 * i * dn_n;
            let down = uploader.upload_experts(i, dn_n, &bits[..i * dn_n])?;
            ExpertWeights::Bf16 {
                gate_up: gate_up.expect("bf16 gate_up"),
                down,
            }
        };
        progress(&format!("{p}.mlp.experts.down"));

        layers.push(OmegaLayer {
            input_norm: read_bf16_f32(
                source,
                &format!("{p}.input_layernorm.weight"),
                &[h],
                &mut small,
            )?,
            post_norm: read_bf16_f32(
                source,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
                &mut small,
            )?,
            q_norm: read_bf16_f32(
                source,
                &format!("{p}.self_attn.q_norm.weight"),
                &[hd],
                &mut small,
            )?,
            k_norm: read_bf16_f32(
                source,
                &format!("{p}.self_attn.k_norm.weight"),
                &[hd],
                &mut small,
            )?,
            router: read_bf16_f32(source, &format!("{p}.mlp.gate.weight"), &[e, h], &mut small)?,
            attn,
            experts,
        });
        if let (Some(router), Some(layer)) = (&device_router, layers.last()) {
            let li = layers.len() - 1;
            let ExpertWeights::Fp8 { gate_up, down } = &layer.experts else {
                return Err(format!("layer {li}: the device router needs FP8 experts"));
            };
            // The router carries the bf16 gate weights; the host copy is an exact widening of them.
            let gate_bf16: Vec<u16> = layer
                .router
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            router
                .set_layer(li, &gate_bf16, &layer.post_norm)
                .map_err(|e| format!("router layer {li}: {e}"))?;
            let (mut gu, mut dn) = (Vec::with_capacity(2 * e), Vec::with_capacity(2 * e));
            for ex in 0..e {
                let (w, s) = gate_up.tensor_entry(ex)?;
                gu.extend([w, s]);
                let (w, s) = down.tensor_entry(ex)?;
                dn.extend([w, s]);
            }
            router
                .set_directory(li, &gu, &dn)
                .map_err(|e| format!("router directory layer {li}: {e}"))?;
        }
    }
    drop(fp8);
    drop(scales);

    // lm_head: bf16 [vocab][h]. The streaming GEMV takes the rows as they are; the dense matmul
    // takes them transposed to [h][vocab].
    let vocab = config.vocab;
    let mut raw = vec![0u8; vocab * h * 2];
    source.read_into("lm_head.weight", "BF16", &[vocab, h], &mut raw)?;
    let gemv = if paths.lm_head == PathChoice::Host {
        None
    } else {
        let rows: Vec<u16> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let g = uploader.upload_lm_head_gemv(vocab, h, &rows)?;
        if g.is_none() && paths.lm_head == PathChoice::Device {
            return Err("the lm_head GEMV was asked for but is not available (omega before the bf16 GEMV cut)".into());
        }
        g
    };
    let lm_head = match gemv {
        Some(g) => LmHead::Gemv(g),
        None => {
            bits.clear();
            transpose_bf16_for_upload(&raw, vocab, h, &mut bits);
            LmHead::Dense(uploader.upload_dense(h, vocab, &bits)?)
        }
    };
    resident += 2 * h * vocab;
    progress("lm_head");
    drop(bits);

    // Embedding stays on the host as bf16 bits (half the f32 size).
    source.read_into("model.embed_tokens.weight", "BF16", &[vocab, h], &mut raw)?;
    let embed_bf16: Vec<u16> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    drop(raw);
    let final_norm = read_bf16_f32(source, "model.norm.weight", &[h], &mut small)?;

    let weights = Qwen3MoeOmegaWeights {
        config: config.clone(),
        embed_bf16,
        layers,
        final_norm,
        lm_head,
        device_router,
    };
    let report = LoadReport {
        resident_bytes: resident,
        format,
        host_bytes: weights.host_bytes(),
        peak_rss_bytes: peak_rss_bytes(),
        peak_rss_before_bytes: peak_before,
        seconds: t0.elapsed().as_secs_f64(),
        window_api: aien_omega_moe::is_native(),
    };
    Ok((weights, report))
}

fn read_scales(
    source: &mut dyn TensorSource,
    name: &str,
    shape: &[usize],
    dst: &mut [u16],
) -> Result<(), String> {
    // SAFETY: a u16 slice viewed as bytes of twice the length; the source writes little-endian
    // bf16 words, which is this host's byte order (aarch64 / x86 little endian).
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len() * 2) };
    source.read_into(name, "BF16", shape, bytes)
}

/// Largest transient the loader holds beyond the model's own host bytes, for the given shape:
/// the bf16 upload buffer, the FP8 layer buffer, its scales and the raw lm_head/embedding read.
pub fn loader_transient_bytes(config: &Qwen3MoeConfig) -> usize {
    let (h, i, e) = (config.hidden, config.moe_intermediate, config.experts);
    let bits = 2 * h * config.gate_up_n().max(config.qkv_n()).max(config.q_dim());
    let fp8 = (e * 2 * i * h).max(config.q_dim() * h + 2 * config.kv_dim() * h);
    let scales = 2
        * (e * 2 * (i / FP8_BLOCK) * (h / FP8_BLOCK))
            .max(((config.q_dim() + 2 * config.kv_dim()) / FP8_BLOCK) * (h / FP8_BLOCK));
    let raw = 2 * config.vocab * h;
    bits + fp8 + scales + raw
}

#[cfg(any(test, feature = "host-test-engine"))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod test_support {
    //! An in-memory checkpoint and a recording uploader with host matmuls, so the loader and the
    //! forward can be exercised end to end without a chip.
    use super::*;
    use crate::qwen3_moe::{f32_to_bf16, f32_to_fp8_e4m3fn};

    pub fn tiny_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            layers: 2,
            hidden: 256,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 128,
            experts: 4,
            top_k: 2,
            moe_intermediate: 128,
            vocab: 64,
            rms_eps: 1e-6,
            rope_theta: 1e7,
            norm_topk_prob: true,
            tie_word_embeddings: false,
            eos_token_ids: vec![3],
        }
    }

    /// Deterministic pseudo-random values in [-1, 1).
    pub fn prand(seed: u64, idx: usize) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(idx as u64 + 1);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        ((x & 0xFFFF) as f32 / 32768.0) - 1.0
    }

    #[derive(Default)]
    pub struct MemorySource {
        pub tensors: HashMap<String, (String, Vec<usize>, Vec<u8>)>,
        pub reads: Vec<String>,
    }

    impl MemorySource {
        fn put_bf16(&mut self, name: &str, shape: &[usize], seed: u64, scale: f32) {
            let n: usize = shape.iter().product();
            let bytes: Vec<u8> = (0..n)
                .flat_map(|i| f32_to_bf16(prand(seed, i) * scale).to_le_bytes())
                .collect();
            self.tensors
                .insert(name.to_string(), ("BF16".into(), shape.to_vec(), bytes));
        }
        fn put_fp8(&mut self, name: &str, rows: usize, cols: usize, seed: u64) {
            let w: Vec<u8> = (0..rows * cols)
                .map(|i| f32_to_fp8_e4m3fn(prand(seed, i) * 8.0))
                .collect();
            self.tensors
                .insert(name.to_string(), ("F8_E4M3".into(), vec![rows, cols], w));
            let sshape = [rows.div_ceil(FP8_BLOCK), cols / FP8_BLOCK];
            let s: Vec<u8> = (0..sshape[0] * sshape[1])
                .flat_map(|i| {
                    f32_to_bf16(0.02 + 0.01 * (prand(seed ^ 0x55, i) + 1.0)).to_le_bytes()
                })
                .collect();
            self.tensors.insert(
                format!("{name}_scale_inv"),
                ("BF16".into(), sshape.to_vec(), s),
            );
        }

        pub fn for_config(c: &Qwen3MoeConfig) -> Self {
            let mut m = Self::default();
            let (h, i, e, hd) = (c.hidden, c.moe_intermediate, c.experts, c.head_dim);
            m.put_bf16("model.embed_tokens.weight", &[c.vocab, h], 1, 1.0);
            m.put_bf16("lm_head.weight", &[c.vocab, h], 2, 0.1);
            m.put_bf16("model.norm.weight", &[h], 3, 1.0);
            for l in 0..c.layers {
                let p = format!("model.layers.{l}");
                let seed = 100 + l as u64 * 1000;
                m.put_fp8(
                    &format!("{p}.self_attn.q_proj.weight"),
                    c.q_dim(),
                    h,
                    seed + 1,
                );
                m.put_fp8(
                    &format!("{p}.self_attn.k_proj.weight"),
                    c.kv_dim(),
                    h,
                    seed + 2,
                );
                m.put_fp8(
                    &format!("{p}.self_attn.v_proj.weight"),
                    c.kv_dim(),
                    h,
                    seed + 3,
                );
                m.put_fp8(
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    c.q_dim(),
                    seed + 4,
                );
                m.put_bf16(
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[hd],
                    seed + 5,
                    1.0,
                );
                m.put_bf16(
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[hd],
                    seed + 6,
                    1.0,
                );
                m.put_bf16(&format!("{p}.input_layernorm.weight"), &[h], seed + 7, 1.0);
                m.put_bf16(
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[h],
                    seed + 8,
                    1.0,
                );
                m.put_bf16(&format!("{p}.mlp.gate.weight"), &[e, h], seed + 9, 1.0);
                for ex in 0..e {
                    let b = format!("{p}.mlp.experts.{ex}");
                    m.put_fp8(
                        &format!("{b}.gate_proj.weight"),
                        i,
                        h,
                        seed + 10 + ex as u64 * 3,
                    );
                    m.put_fp8(
                        &format!("{b}.up_proj.weight"),
                        i,
                        h,
                        seed + 11 + ex as u64 * 3,
                    );
                    m.put_fp8(
                        &format!("{b}.down_proj.weight"),
                        h,
                        i,
                        seed + 12 + ex as u64 * 3,
                    );
                }
            }
            m
        }

        pub fn fp8_f32(&self, name: &str) -> (Vec<f32>, usize, usize) {
            let (_, shape, w) = &self.tensors[name];
            let (_, _, s) = &self.tensors[&format!("{name}_scale_inv")];
            let (rows, cols) = (shape[0], shape[1]);
            let kb = cols / FP8_BLOCK;
            let scales: Vec<u16> = s
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            let lut = fp8_table();
            let out = (0..rows * cols)
                .map(|idx| {
                    let (r, k) = (idx / cols, idx % cols);
                    lut[w[idx] as usize] * bf16_to_f32(scales[(r / FP8_BLOCK) * kb + k / FP8_BLOCK])
                })
                .collect();
            (out, rows, cols)
        }
    }

    impl TensorSource for MemorySource {
        fn read_into(
            &mut self,
            name: &str,
            dtype: &str,
            shape: &[usize],
            dst: &mut [u8],
        ) -> Result<(), String> {
            let (dt, sh, bytes) = self
                .tensors
                .get(name)
                .ok_or_else(|| format!("no tensor {name}"))?;
            if dt != dtype || sh != shape {
                return Err(format!("{name}: {dt} {sh:?} asked as {dtype} {shape:?}"));
            }
            if bytes.len() != dst.len() {
                return Err(format!("{name}: {} bytes, dst {}", bytes.len(), dst.len()));
            }
            dst.copy_from_slice(bytes);
            self.reads.push(name.to_string());
            Ok(())
        }

        fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
            Ok(self.tensors.get(name).map(|t| t.0.clone()))
        }
    }

    /// A `[k][n]` bf16 matrix on the host; matmul in f32 with `a` rounded to bf16 as omega does.
    #[derive(Debug)]
    pub struct HostMatrix {
        pub k: usize,
        pub n: usize,
        pub bits: Vec<u16>,
    }

    impl HostMatrix {
        pub fn full(&self, m: usize, a: &[f32], c: &mut [f32]) {
            assert_eq!(a.len(), m * self.k);
            assert_eq!(c.len(), m * self.n);
            for r in 0..m {
                let ar: Vec<f32> = a[r * self.k..(r + 1) * self.k]
                    .iter()
                    .map(|&v| bf16_to_f32(f32_to_bf16(v)))
                    .collect();
                for j in 0..self.n {
                    let mut acc = 0.0f32;
                    for kk in 0..self.k {
                        acc += ar[kk] * bf16_to_f32(self.bits[kk * self.n + j]);
                    }
                    c[r * self.n + j] = acc;
                }
            }
        }
    }

    impl DenseMatmul for HostMatrix {
        fn k(&self) -> usize {
            self.k
        }
        fn n(&self) -> usize {
            self.n
        }
        fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String> {
            self.full(m, a, c);
            Ok(0)
        }
    }

    impl ExpertMatmul for HostMatrix {
        fn k(&self) -> usize {
            self.k
        }
        fn n(&self) -> usize {
            self.n
        }
        fn windows_f32(
            &self,
            m: usize,
            a: &[f32],
            col0s: &[u32],
            cols: usize,
            c: &mut [f32],
        ) -> Result<u64, String> {
            // Only the windows are computed (so a wrong col0 is a wrong answer, as on the chip).
            if c.len() != col0s.len() * m * cols {
                return Err("window output size".into());
            }
            for (w, &col0) in col0s.iter().enumerate() {
                aien_omega_moe::check_window(self.n, col0 as usize, cols)
                    .map_err(|e| e.to_string())?;
                for r in 0..m {
                    let ar: Vec<f32> = a[r * self.k..(r + 1) * self.k]
                        .iter()
                        .map(|&v| bf16_to_f32(f32_to_bf16(v)))
                        .collect();
                    for j in 0..cols {
                        let col = col0 as usize + j;
                        let mut acc = 0.0f32;
                        for kk in 0..self.k {
                            acc += ar[kk] * bf16_to_f32(self.bits[kk * self.n + col]);
                        }
                        c[(w * m + r) * cols + j] = acc;
                    }
                }
            }
            Ok(0)
        }
    }

    /// Records every upload (kind, k, n, bytes) and keeps the matrices for host matmuls.
    #[derive(Default)]
    pub struct RecordingUploader {
        pub uploads: Mutex<Vec<(&'static str, usize, usize, usize)>>,
        pub matrices: Mutex<Vec<Arc<HostMatrix>>>,
        /// Offer a host-computed device router and lm_head GEMV (the real uploader offers omega's).
        pub device_fakes: bool,
        /// Offer the prefill GEMM on the FP8 tensors (host-computed, checks the row table).
        pub gemm_fakes: bool,
        pub dev: Arc<Mutex<FakeDevice>>,
    }

    /// What the fake router "left on the device" for the next grouped launch.
    #[derive(Debug, Default)]
    pub struct FakeDevice {
        pub post: Vec<f32>,
        pub ids: Vec<u32>,
        pub routes: usize,
        pub grouped_dev_calls: usize,
        pub gemv_calls: usize,
        pub gemm_calls: usize,
        pub gemm_rows: usize,
    }

    pub const FAKE_X_VA: u64 = 0xA000;
    pub const FAKE_TAB_VA: u64 = 0xB000;

    /// Host router with the device router's contract: RMSNorm, router logits over the bf16 gate,
    /// top-k with renormalised weights; the normed row and the chosen ids stay in `FakeDevice`.
    #[derive(Debug)]
    pub struct FakeRouter {
        pub dev: Arc<Mutex<FakeDevice>>,
        pub experts: usize,
        pub hidden: usize,
        pub layers: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
        pub directories: Mutex<Vec<usize>>,
    }

    impl DeviceRouter for FakeRouter {
        fn set_layer(&self, layer: usize, gate_bf16: &[u16], norm_w: &[f32]) -> Result<(), String> {
            assert_eq!(gate_bf16.len(), self.experts * self.hidden);
            let gate: Vec<f32> = gate_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
            let mut l = self.layers.lock().unwrap();
            assert_eq!(l.len(), layer, "layers are set in order");
            l.push((gate, norm_w.to_vec()));
            Ok(())
        }
        fn set_directory(&self, layer: usize, gu: &[u64], dn: &[u64]) -> Result<(), String> {
            assert_eq!((gu.len(), dn.len()), (2 * self.experts, 2 * self.experts));
            // tensor_entry of expert x is (x + 1, 0): the directory must list them in order.
            assert!(gu
                .chunks(2)
                .enumerate()
                .all(|(x, p)| p == [x as u64 + 1, 0]));
            self.directories.lock().unwrap().push(layer);
            Ok(())
        }
        fn route(
            &self,
            layer: usize,
            x: &[f32],
            eps: f32,
            ids: &mut [u32],
            weights: &mut [f32],
        ) -> Result<(u64, RouteAddrs), String> {
            let l = self.layers.lock().unwrap();
            let (gate, norm) = &l[layer];
            let mut post = vec![0.0f32; self.hidden];
            rmsnorm(x, norm, eps, &mut post);
            let logits: Vec<f32> = (0..self.experts)
                .map(|e| {
                    gate[e * self.hidden..(e + 1) * self.hidden]
                        .iter()
                        .zip(&post)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect();
            let plan = MoeBatchPlan::from_logits(&logits, 1, self.experts, ids.len())
                .map_err(|e| e.to_string())?;
            ids.copy_from_slice(&plan.topk_experts);
            weights.copy_from_slice(&plan.topk_weights);
            let mut d = self.dev.lock().unwrap();
            d.post = post;
            d.ids = plan.topk_experts;
            d.routes += 1;
            Ok((
                0,
                RouteAddrs {
                    x_va: FAKE_X_VA,
                    tab_gu_va: FAKE_TAB_VA,
                },
            ))
        }
        fn read_post(&self, out: &mut [f32]) -> Result<(), String> {
            out.copy_from_slice(&self.dev.lock().unwrap().post);
            Ok(())
        }
    }

    /// `[vocab][hidden]` bf16 rows on the host; f32 dot products.
    #[derive(Debug)]
    pub struct FakeLmHead {
        pub dev: Arc<Mutex<FakeDevice>>,
        pub n: usize,
        pub k: usize,
        pub bits: Vec<u16>,
    }

    impl LmHeadGemv for FakeLmHead {
        fn logits_f32(&self, hidden: &[f32], out: &mut [f32]) -> Result<u64, String> {
            assert_eq!((hidden.len(), out.len()), (self.k, self.n));
            for (r, o) in out.iter_mut().enumerate() {
                *o = self.bits[r * self.k..(r + 1) * self.k]
                    .iter()
                    .zip(hidden)
                    .map(|(&w, &x)| bf16_to_f32(w) * x)
                    .sum();
            }
            self.dev.lock().unwrap().gemv_calls += 1;
            Ok(0)
        }
    }

    /// Exact host FP8 GEMV (`e4m3(w) * bf16(scale)` in f32) over the checkpoint bytes, the
    /// oracle the chip path is measured against in `own30b_check` and the fake for host tests.
    #[derive(Debug)]
    pub struct HostFp8 {
        pub experts: usize,
        pub n: usize,
        pub k: usize,
        pub w: Vec<u8>,
        pub scales: Vec<u16>,
        /// Set by the uploader when it offers the fake router: serves `tensor_entry` / `grouped_dev`.
        pub dev: Option<Arc<Mutex<FakeDevice>>>,
        /// Set by the uploader when it offers the prefill GEMM: counts the calls and rows.
        pub gemm: Option<Arc<Mutex<FakeDevice>>>,
    }

    impl Fp8Gemv for HostFp8 {
        fn has_gemm(&self) -> bool {
            self.gemm.is_some()
        }
        fn gemm_f32(
            &self,
            expert_ids: &[u32],
            token_rows: &[u32],
            x: &[f32],
            y: &mut [f32],
        ) -> Result<u64, String> {
            let dev = self.gemm.as_ref().ok_or_else(|| "no gemm".to_string())?;
            let (m, k, n) = (expert_ids.len(), self.k, self.n);
            assert_eq!(token_rows.len(), m);
            assert_eq!(y.len(), m * n);
            assert!(
                expert_ids.windows(2).all(|w| w[0] <= w[1]),
                "the row table must be sorted by expert"
            );
            let x_rows = x.len() / k;
            for (i, &tr) in token_rows.iter().enumerate() {
                assert!((tr as usize) < x_rows);
                self.grouped_f32(
                    &expert_ids[i..=i],
                    &x[tr as usize * k..(tr as usize + 1) * k],
                    &mut y[i * n..(i + 1) * n],
                )?;
            }
            let mut d = dev.lock().unwrap();
            d.gemm_calls += 1;
            d.gemm_rows += m;
            Ok(0)
        }
        fn tensor_entry(&self, expert: usize) -> Result<(u64, u64), String> {
            self.dev
                .as_ref()
                .map(|_| (expert as u64 + 1, 0))
                .ok_or_else(|| "no device".to_string())
        }
        fn grouped_dev(
            &self,
            groups: usize,
            tab_va: u64,
            x_va: u64,
            y: &mut [f32],
        ) -> Result<u64, String> {
            let dev = self.dev.as_ref().ok_or_else(|| "no device".to_string())?;
            assert_eq!(
                (tab_va, x_va),
                (FAKE_TAB_VA, FAKE_X_VA),
                "the engine must pass the router's addresses"
            );
            let (ids, post) = {
                let mut d = dev.lock().unwrap();
                d.grouped_dev_calls += 1;
                (d.ids.clone(), d.post.clone())
            };
            assert_eq!(ids.len(), groups);
            self.grouped_f32(&ids, &post, y)
        }

        fn experts(&self) -> usize {
            self.experts
        }
        fn n(&self) -> usize {
            self.n
        }
        fn k(&self) -> usize {
            self.k
        }
        fn grouped_f32(&self, ids: &[u32], x: &[f32], y: &mut [f32]) -> Result<u64, String> {
            let (n, k) = (self.n, self.k);
            let kb = k / FP8_BLOCK;
            let lut = fp8_table();
            assert!(x.len() == k || x.len() == ids.len() * k);
            assert_eq!(y.len(), ids.len() * n);
            for (g, &id) in ids.iter().enumerate() {
                let id = id as usize;
                assert!(id < self.experts);
                let xr = if x.len() == k {
                    x
                } else {
                    &x[g * k..(g + 1) * k]
                };
                let w = &self.w[id * n * k..(id + 1) * n * k];
                let s = &self.scales[id * (n / FP8_BLOCK) * kb..(id + 1) * (n / FP8_BLOCK) * kb];
                for r in 0..n {
                    let mut acc = 0.0f32;
                    for c in 0..k {
                        acc += lut[w[r * k + c] as usize]
                            * bf16_to_f32(s[(r / FP8_BLOCK) * kb + c / FP8_BLOCK])
                            * xr[c];
                    }
                    y[g * n + r] = acc;
                }
            }
            Ok(0)
        }
    }

    impl ModelUploader for RecordingUploader {
        fn upload_dense(
            &self,
            k: usize,
            n: usize,
            bits: &[u16],
        ) -> Result<Arc<dyn DenseMatmul>, String> {
            assert_eq!(bits.len(), k * n);
            self.uploads
                .lock()
                .unwrap()
                .push(("dense", k, n, bits.len() * 2));
            let m = Arc::new(HostMatrix {
                k,
                n,
                bits: bits.to_vec(),
            });
            self.matrices.lock().unwrap().push(m.clone());
            Ok(m)
        }
        fn upload_experts(
            &self,
            k: usize,
            n: usize,
            bits: &[u16],
        ) -> Result<Arc<dyn ExpertMatmul>, String> {
            assert_eq!(bits.len(), k * n);
            self.uploads
                .lock()
                .unwrap()
                .push(("experts", k, n, bits.len() * 2));
            let m = Arc::new(HostMatrix {
                k,
                n,
                bits: bits.to_vec(),
            });
            self.matrices.lock().unwrap().push(m.clone());
            Ok(m)
        }
        fn upload_fp8(
            &self,
            experts: usize,
            n: usize,
            k: usize,
            w: &[u8],
            scales: &[u16],
        ) -> Result<Arc<dyn Fp8Gemv>, String> {
            assert_eq!(w.len(), experts * n * k);
            assert_eq!(scales.len(), experts * (n / FP8_BLOCK) * (k / FP8_BLOCK));
            self.uploads
                .lock()
                .unwrap()
                .push(("fp8", experts * n, k, w.len() + scales.len() * 4));
            Ok(Arc::new(HostFp8 {
                experts,
                n,
                k,
                w: w.to_vec(),
                scales: scales.to_vec(),
                dev: self.device_fakes.then(|| self.dev.clone()),
                gemm: self.gemm_fakes.then(|| self.dev.clone()),
            }))
        }
        fn new_router(
            &self,
            config: &Qwen3MoeConfig,
        ) -> Result<Option<Arc<dyn DeviceRouter>>, String> {
            Ok(self.device_fakes.then(|| {
                Arc::new(FakeRouter {
                    dev: self.dev.clone(),
                    experts: config.experts,
                    hidden: config.hidden,
                    layers: Mutex::new(Vec::new()),
                    directories: Mutex::new(Vec::new()),
                }) as Arc<dyn DeviceRouter>
            }))
        }
        fn upload_lm_head_gemv(
            &self,
            n: usize,
            k: usize,
            bits: &[u16],
        ) -> Result<Option<Arc<dyn LmHeadGemv>>, String> {
            assert_eq!(bits.len(), n * k);
            Ok(self.device_fakes.then(|| {
                Arc::new(FakeLmHead {
                    dev: self.dev.clone(),
                    n,
                    k,
                    bits: bits.to_vec(),
                }) as Arc<dyn LmHeadGemv>
            }))
        }
    }
}

#[cfg(test)]
mod loader_tests {
    use super::test_support::*;
    use super::*;
    use crate::qwen3_moe::fp8_e4m3fn_to_f32;

    const REAL_CONFIG: &str = r#"{"architectures":["Qwen3MoeForCausalLM"],"bos_token_id":151643,"decoder_sparse_step":1,"eos_token_id":151645,"head_dim":128,"hidden_size":2048,"max_position_embeddings":262144,"mlp_only_layers":[],"model_type":"qwen3_moe","moe_intermediate_size":768,"norm_topk_prob":true,"num_attention_heads":32,"num_experts":128,"num_experts_per_tok":8,"num_hidden_layers":48,"num_key_value_heads":4,"rms_norm_eps":0.000001,"rope_theta":10000000,"tie_word_embeddings":false,"vocab_size":151936}"#;

    #[test]
    fn real_config_parses_and_sizes_match_the_design() {
        let c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        assert_eq!(
            (c.layers, c.hidden, c.q_heads, c.kv_heads, c.head_dim),
            (48, 2048, 32, 4, 128)
        );
        assert_eq!(
            (c.experts, c.top_k, c.moe_intermediate, c.vocab),
            (128, 8, 768, 151936)
        );
        assert_eq!(c.eos_token_ids, vec![151645]);
        assert_eq!(c.qkv_n(), 5120);
        assert_eq!(c.gate_up_n(), 196608);
        assert_eq!(c.down_n(), 262144);
        // Design section 0: experts 48 x 1.21 GB, attention 1.8 GB, lm_head 0.62 GB: about 60.4 GB.
        let layer = c.layer_resident_bytes();
        assert_eq!(
            layer,
            2 * (2048 * 5120 + 4096 * 2048 + 2048 * 196608 + 768 * 262144)
        );
        let total = c.model_resident_bytes();
        assert!((60.0e9..61.0e9).contains(&(total as f64)), "{total}");
        // bf16 bytes per generated token: about 6.1 GB (design section 0).
        let per_tok = c.bytes_per_token_bf16() as f64;
        assert!((6.0e9..6.2e9).contains(&per_tok), "{per_tok}");
    }

    #[test]
    fn config_refuses_shapes_stage_a_cannot_run() {
        let mut c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        c.head_dim = 64;
        assert!(c.validate().is_err());
        let mut c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        c.experts = 129;
        assert!(c.validate().unwrap_err().contains("down"));
        let mut c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        c.norm_topk_prob = false;
        assert!(c.validate().is_err());
        assert!(Qwen3MoeConfig::from_json(&REAL_CONFIG.replace("qwen3_moe", "qwen3")).is_err());
    }

    #[test]
    fn expert_windows_start_on_a_column_tile_and_stay_inside_the_tensor() {
        let c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        for e in 0..c.experts {
            let gu = c.gate_up_col0(e);
            let dn = c.down_col0(e);
            assert_eq!(gu % WINDOW_TILE, 0);
            assert_eq!(dn % WINDOW_TILE, 0);
            assert!(aien_omega_moe::check_window(c.gate_up_n(), gu, c.gate_up_cols()).is_ok());
            assert!(aien_omega_moe::check_window(c.down_n(), dn, c.down_cols()).is_ok());
        }
        assert_eq!(c.gate_up_col0(5), 7680);
        assert_eq!(c.down_col0(127), 260096);
        assert_eq!(c.down_col0(127) + c.down_cols(), MAX_RESIDENT_N);
        // Up half of expert e sits i columns after its gate half.
        assert_eq!(c.gate_up_col0(3) + c.moe_intermediate, 3 * 1536 + 768);
    }

    #[test]
    fn dequant_matches_the_exact_conversion_transposed_and_rounded() {
        // rows 200 (not a block multiple: 2 row blocks), k 256 (2 blocks).
        let (rows, k) = (200usize, 256usize);
        let w: Vec<u8> = (0..rows * k)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .map(|b| if b & 0x7f == 0x7f { 0 } else { b })
            .collect();
        let scales: Vec<u16> = (0..2 * 2)
            .map(|i| f32_to_bf16(0.5 + i as f32 * 0.25))
            .collect();
        let dst_n = 208;
        let mut dst = vec![0u16; k * dst_n];
        dequant_fp8_sources_to_bf16_cols(
            &[Fp8Source {
                weights: &w,
                scales: &scales,
                rows,
                col0: 8,
            }],
            k,
            &mut dst,
            dst_n,
        )
        .unwrap();
        for r in 0..rows {
            for kk in 0..k {
                let s = bf16_to_f32(scales[(r / 128) * 2 + kk / 128]);
                let want = f32_to_bf16(fp8_e4m3fn_to_f32(w[r * k + kk]) * s);
                assert_eq!(dst[kk * dst_n + 8 + r], want, "r={r} k={kk}");
            }
            // Spot values: the exact table says 0x38 is 1.0 and 0xBC is -1.5.
        }
        assert_eq!(bf16_to_f32(f32_to_bf16(fp8_e4m3fn_to_f32(0x38))), 1.0);
        assert_eq!(fp8_e4m3fn_to_f32(0xBC), -1.5);
        // Untouched columns stay zero.
        assert!(dst
            .iter()
            .enumerate()
            .filter(|(i, _)| i % dst_n < 8)
            .all(|(_, &v)| v == 0));
    }

    #[test]
    fn dequant_refuses_nan_codes_and_bad_shapes() {
        let w = vec![0x7fu8; 128];
        let s = vec![f32_to_bf16(1.0)];
        let mut dst = vec![0u16; 128 * 8];
        let err = dequant_fp8_sources_to_bf16_cols(
            &[Fp8Source {
                weights: &w,
                scales: &s,
                rows: 1,
                col0: 0,
            }],
            128,
            &mut dst,
            8,
        )
        .unwrap_err();
        assert!(err.contains("NaN"), "{err}");
        let w = vec![0u8; 128];
        let err = dequant_fp8_sources_to_bf16_cols(
            &[Fp8Source {
                weights: &w,
                scales: &s,
                rows: 1,
                col0: 8,
            }],
            128,
            &mut dst,
            8,
        )
        .unwrap_err();
        assert!(err.contains("past dst_n"), "{err}");
        let err = dequant_fp8_sources_to_bf16_cols(
            &[Fp8Source {
                weights: &w,
                scales: &[],
                rows: 1,
                col0: 0,
            }],
            128,
            &mut dst,
            8,
        )
        .unwrap_err();
        assert!(err.contains("scales"), "{err}");
    }

    #[test]
    fn loader_uploads_the_right_shapes_bytes_and_values() {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader::default();
        let mut seen = Vec::new();
        let (w, report) =
            load_qwen3_moe_omega_from(&c, &mut src, &up, &mut |s| seen.push(s.to_string()))
                .unwrap();
        let uploads = up.uploads.lock().unwrap().clone();
        // Per layer: qkv, o, gate_up, down; then lm_head.
        assert_eq!(uploads.len(), 4 * c.layers + 1);
        for l in 0..c.layers {
            assert_eq!(
                uploads[4 * l],
                ("dense", c.hidden, c.qkv_n(), 2 * c.hidden * c.qkv_n())
            );
            assert_eq!(
                uploads[4 * l + 1],
                ("dense", c.q_dim(), c.hidden, 2 * c.q_dim() * c.hidden)
            );
            assert_eq!(
                uploads[4 * l + 2],
                (
                    "experts",
                    c.hidden,
                    c.gate_up_n(),
                    2 * c.hidden * c.gate_up_n()
                )
            );
            assert_eq!(
                uploads[4 * l + 3],
                (
                    "experts",
                    c.moe_intermediate,
                    c.down_n(),
                    2 * c.moe_intermediate * c.down_n()
                )
            );
        }
        assert_eq!(
            uploads[4 * c.layers],
            ("dense", c.hidden, c.vocab, 2 * c.hidden * c.vocab)
        );
        let total: usize = uploads.iter().map(|u| u.3).sum();
        assert_eq!(total, c.model_resident_bytes());
        assert_eq!(report.resident_bytes, total);
        assert_eq!(seen.len(), uploads.len());
        assert_eq!(w.host_bytes(), report.host_bytes);
        assert_eq!(w.embed_bf16.len(), c.vocab * c.hidden);
        assert_eq!(w.layers[1].router.len(), c.experts * c.hidden);

        // Values: expert 2's up projection of layer 1, row 5, column k=130 must sit at
        // gate_up[k=130][col0(2) + i + 5], equal to the exact dequant rounded to bf16.
        let mats = up.matrices.lock().unwrap();
        let gate_up = &mats[4 + 2];
        let (upw, rows, cols) = src.fp8_f32("model.layers.1.mlp.experts.2.up_proj.weight");
        assert_eq!((rows, cols), (c.moe_intermediate, c.hidden));
        let col = c.gate_up_col0(2) + c.moe_intermediate + 5;
        assert_eq!(
            gate_up.bits[130 * c.gate_up_n() + col],
            f32_to_bf16(upw[5 * cols + 130])
        );
        // down of expert 3, layer 0: W[h=7][i=100] at down[k=100][col0(3) + 7].
        let down = &mats[3];
        let (dw, _, dcols) = src.fp8_f32("model.layers.0.mlp.experts.3.down_proj.weight");
        assert_eq!(
            down.bits[100 * c.down_n() + c.down_col0(3) + 7],
            f32_to_bf16(dw[7 * dcols + 100])
        );
        // v_proj row 3 of layer 0 lands at qkv column q_dim + kv_dim + 3.
        let qkv = &mats[0];
        let (vw, _, vcols) = src.fp8_f32("model.layers.0.self_attn.v_proj.weight");
        assert_eq!(
            qkv.bits[77 * c.qkv_n() + c.q_dim() + c.kv_dim() + 3],
            f32_to_bf16(vw[3 * vcols + 77])
        );
        // o_proj: W[h=9][qd=200] at o[k=200][9].
        let o = &mats[1];
        let (ow, _, ocols) = src.fp8_f32("model.layers.0.self_attn.o_proj.weight");
        assert_eq!(o.bits[200 * c.hidden + 9], f32_to_bf16(ow[9 * ocols + 200]));
        // lm_head [vocab][h] transposed: row 10, k 33 at lm_head[33][10].
        let lm = &mats[mats.len() - 1];
        let (_, _, raw) = &src.tensors["lm_head.weight"];
        let want = u16::from_le_bytes([
            raw[(10 * c.hidden + 33) * 2],
            raw[(10 * c.hidden + 33) * 2 + 1],
        ]);
        assert_eq!(lm.bits[33 * c.vocab + 10], want);
        // Embedding row is the bf16 row widened.
        let (_, _, eraw) = &src.tensors["model.embed_tokens.weight"];
        let e0 = u16::from_le_bytes([eraw[(5 * c.hidden) * 2], eraw[(5 * c.hidden) * 2 + 1]]);
        assert_eq!(w.embed_row(5)[0], bf16_to_f32(e0));
    }

    #[test]
    fn loader_transient_budget_is_one_layer_plus_embedding() {
        let c = Qwen3MoeConfig::from_json(REAL_CONFIG).unwrap();
        let t = loader_transient_bytes(&c) as f64;
        // bits 805 MB + fp8 403 MB + scales 1.5 MB + raw lm_head 622 MB: about 1.83 GB.
        assert!((1.7e9..2.0e9).contains(&t), "{t}");
        // With the fake uploader the process holds the matrices itself, so only the report's
        // bookkeeping is checked here; the real-load assertion is in own30b_check.
        assert!(peak_rss_bytes().is_some_and(|b| b > 0));
    }
}

// ===========================================================================
// Cut 3: the Stage A forward.
// ===========================================================================

use crate::qwen3_coder::rmsnorm_per_head;
use crate::tensor::{apply_rope, rmsnorm, scaled_dot_product_attention_single};
use crate::transformer_backend::{decode_sampling_seed, sample_with_params};
use aien_abi_core::SamplingParams;

/// Model-independent KV shape (ADR 0004 section 2.1). Produced by a model config, consumed by
/// every cache constructor and by persistence. bf16 elements, `HostPagedPoolV1` payload (ADR 0003).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvGeometry {
    pub layers: usize,
    /// Attend output width; the fused kernel's `fits` check.
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The fused cache applies RoPE on the chip.
    pub rope_theta: f32,
    /// [`KV_BLOCK_SIZE`].
    pub block_size: usize,
}

impl KvGeometry {
    /// The byte geometry of one `HostPagedPoolV1` block of this shape.
    pub fn pool_block(&self) -> kvio::PoolBlockGeometry {
        kvio::PoolBlockGeometry::for_geometry(self)
    }
}

impl From<&Qwen3MoeConfig> for KvGeometry {
    fn from(c: &Qwen3MoeConfig) -> Self {
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

/// The pool caches index blocks by [`KV_BLOCK_SIZE`]; a geometry with another size is refused.
fn check_block_size(who: &str, g: &KvGeometry) -> Result<(), String> {
    if g.block_size != KV_BLOCK_SIZE {
        return Err(format!(
            "{who}: block size {} is not this cache's layout ({KV_BLOCK_SIZE})",
            g.block_size
        ));
    }
    Ok(())
}

/// The model-independent half of a KV cache (ADR 0004 section 2.1): what the generation
/// controller may see. Split out of [`KvAttention`] as its supertrait.
pub trait KvStore: Send {
    /// The shape this cache was built for.
    fn geometry(&self) -> KvGeometry;
    /// Tokens the cache can hold.
    fn capacity(&self) -> usize;
    /// Positions held in every layer (the smallest per-layer filled count).
    fn len(&self) -> usize;
    /// Whether no layer holds any position.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// "split" (host norm/rope, `append` + `attend`), "host-pool" (the host bf16 pool) or "fused"
    /// (one launch per layer).
    fn path_name(&self) -> &'static str {
        "split"
    }
    /// Drops every position at or past `len` (the next append is at `len`). Prefix reuse;
    /// `truncate(0)` empties the cache (what `reset` did, for all four caches).
    fn truncate(&mut self, len: usize);
    /// Persistent KV park (sc#505): the K/V of positions `start..start + len` as the
    /// `HostPagedPoolV1` payload of [`kvio`] (`start` a multiple of [`KV_BLOCK_SIZE`]; every
    /// layer must hold those positions). `None`: this cache cannot export, nothing is parked.
    fn export_blocks(&mut self, _start: usize, _len: usize) -> Option<Result<Vec<u8>, String>> {
        None
    }
    /// Persistent KV restore (sc#505): writes a [`kvio`] payload for positions `start..start +
    /// len`; every layer must hold exactly `0..start` before the call and holds `0..start + len`
    /// after it. `None`: this cache cannot import, the caller prefills instead. On `Err` the
    /// cache may hold part of the payload: the caller must reset it.
    fn import_blocks(
        &mut self,
        _start: usize,
        _len: usize,
        _bytes: &[u8],
    ) -> Option<Result<(), String>> {
        None
    }
}

/// The KV cache of one sequence plus the attention over it. Omega's paged bf16 pool in
/// production; a host f32 cache in tests. Model-private: the controller sees only [`KvStore`].
pub trait KvAttention: KvStore {
    /// Stores the roped key and the value of position `pos` for `layer`.
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String>;
    /// `out[q_heads x head_dim]` = attention of the (unscaled) query over positions `0..context_len`.
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String>;
    /// Whether the chip reads the pool in place (no per-call staging).
    fn resident(&self) -> bool;
    /// The fused path (OWN-30B cut 9): from the raw fused q|k|v projection row of position
    /// `pos`, do the q/k per-head norms, RoPE, the K/V append and the attention over
    /// `0..=pos` on the chip in one launch, writing `out[q_heads x head_dim]`. `None` means
    /// this cache has no fused kernel and the engine runs the split path (`append` + `attend`).
    fn fused_attend(
        &mut self,
        _layer: usize,
        _pos: usize,
        _qkv: &[f32],
        _eps: f32,
        _out: &mut [f32],
    ) -> Option<Result<u64, String>> {
        None
    }
    /// omega's fused cache behind this attention, if any (the whole-prompt prefill drives it).
    fn fused_kv(&mut self) -> Option<&mut crate::qwen3_moe_omega_dattn::OmegaFusedKv> {
        None
    }
    /// The one-list decode graph (OWN-30B L2 cut 2): the omega C handle of the KV cache
    /// (`OmegaGpuDecodeAttnCache *`) after checking every layer holds exactly `0..pos`. `None`:
    /// this cache cannot take part in the graph.
    fn dgraph_cache(&mut self, _pos: usize) -> Option<Result<*mut std::ffi::c_void, String>> {
        None
    }
    /// The graph wrote position `pos` of every layer.
    fn dgraph_commit(&mut self, _pos: usize) {}
}

/// Positions `start..start + len` of a bf16 `aien_kv_cache` pool (blocks `0..n` in order, block
/// size [`KV_BLOCK_SIZE`]) as a [`kvio`] payload: the pool layout is the payload layout, so the
/// blocks are copied as they are and the rows past `len` in the last block are zeroed in the copy.
fn pool_export(
    pool: &aien_kv_cache::UnifiedKvTensorPool,
    filled: &[usize],
    start: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    let c = pool.config();
    let g = kvio::PoolBlockGeometry {
        layers: c.num_layers,
        kv_heads: c.num_kv_heads,
        head_dim: c.head_dim,
        block_size: c.block_size,
    };
    if c.dtype != aien_kv_cache::KvDType::Bf16 || g.block_stride() != pool.block_bytes() {
        return Err(format!(
            "kv export: pool {:?} with {} byte blocks is not the bf16 payload layout",
            c.dtype,
            pool.block_bytes()
        ));
    }
    if len == 0 || !start.is_multiple_of(g.block_size) {
        return Err(format!(
            "kv export {start}+{len}: start must be a multiple of {}, len non-zero",
            g.block_size
        ));
    }
    if let Some((l, f)) = filled.iter().enumerate().find(|(_, &f)| f < start + len) {
        return Err(format!(
            "kv export {start}+{len}: layer {l} holds only {f} positions"
        ));
    }
    let off = (start / g.block_size) * g.block_stride();
    let n = g.payload_len(len);
    if off + n > pool.total_bytes() {
        return Err(format!(
            "kv export {start}+{len}: past the pool ({} bytes)",
            pool.total_bytes()
        ));
    }
    // SAFETY: the pool owns total_bytes() readable bytes at base_ptr(); the range is checked.
    let mut out = unsafe { std::slice::from_raw_parts(pool.base_ptr().add(off), n) }.to_vec();
    kvio::zero_padding(&g, len, &mut out);
    Ok(out)
}

/// The inverse of [`pool_export`]: writes the payload's blocks into the pool and marks
/// `0..start + len` filled in every layer.
fn pool_import(
    pool: &mut aien_kv_cache::UnifiedKvTensorPool,
    filled: &mut [usize],
    start: usize,
    len: usize,
    bytes: &[u8],
) -> Result<(), String> {
    let c = pool.config();
    let g = kvio::PoolBlockGeometry {
        layers: c.num_layers,
        kv_heads: c.num_kv_heads,
        head_dim: c.head_dim,
        block_size: c.block_size,
    };
    if c.dtype != aien_kv_cache::KvDType::Bf16 || g.block_stride() != pool.block_bytes() {
        return Err(format!(
            "kv import: pool {:?} with {} byte blocks is not the bf16 payload layout",
            c.dtype,
            pool.block_bytes()
        ));
    }
    if len == 0 || !start.is_multiple_of(g.block_size) {
        return Err(format!(
            "kv import {start}+{len}: start must be a multiple of {}, len non-zero",
            g.block_size
        ));
    }
    if let Some((l, f)) = filled.iter().enumerate().find(|(_, &f)| f != start) {
        return Err(format!(
            "kv import at {start}: layer {l} holds {f} positions"
        ));
    }
    kvio::check_payload(&g, len, bytes)?;
    let off = (start / g.block_size) * g.block_stride();
    if off + bytes.len() > pool.total_bytes() {
        return Err(format!(
            "kv import {start}+{len}: past the pool ({} bytes)",
            pool.total_bytes()
        ));
    }
    // SAFETY: the pool owns total_bytes() writable bytes at base_mut_ptr(); the range is checked
    // and `bytes` is a separate allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), pool.base_mut_ptr().add(off), bytes.len())
    };
    filled.iter_mut().for_each(|f| *f = start + len);
    Ok(())
}

/// A host bf16 pool with the reference attention over the bf16 values: the same bytes as
/// [`OmegaPagedKv`] (append rounds to bf16 in the pool, export/import copy blocks), with no chip.
/// Host tests of the persistent KV path (sc#505) run the real engine over it.
pub struct HostPoolKv {
    pool: aien_kv_cache::UnifiedKvTensorPool,
    capacity: usize,
    geometry: KvGeometry,
    filled: Vec<usize>,
}

impl HostPoolKv {
    pub fn new(geometry: &KvGeometry, max_context: usize) -> Result<Self, String> {
        check_block_size("host pool kv", geometry)?;
        let num_blocks = max_context.div_ceil(KV_BLOCK_SIZE).max(1);
        let cfg = aien_kv_cache::KvPoolConfig::for_model(
            num_blocks,
            KV_BLOCK_SIZE,
            geometry.layers,
            geometry.kv_heads,
            geometry.head_dim,
            aien_kv_cache::KvDType::Bf16,
        );
        Ok(Self {
            pool: aien_kv_cache::UnifiedKvTensorPool::allocate(cfg)?,
            capacity: num_blocks * KV_BLOCK_SIZE,
            geometry: *geometry,
            filled: vec![0; geometry.layers],
        })
    }

    /// Positions held in `layer` alone (ADR 0004 test R1 checks every layer against the ledger;
    /// [`KvStore::len`] is only the smallest of them).
    pub fn layer_len(&self, layer: usize) -> usize {
        self.filled[layer]
    }

    fn row(&self, layer: usize, pos: usize, value: bool) -> Vec<f32> {
        let kv_dim = self.geometry.kv_heads * self.geometry.head_dim;
        let off = self
            .pool
            .element_offset(pos / KV_BLOCK_SIZE, layer, value, pos % KV_BLOCK_SIZE);
        // SAFETY: the offset is inside the pool (pos < filled <= capacity), kv_dim bf16 values.
        let b = unsafe { std::slice::from_raw_parts(self.pool.base_ptr().add(off), kv_dim * 2) };
        b.chunks_exact(2)
            .map(|w| bf16_to_f32(u16::from_le_bytes([w[0], w[1]])))
            .collect()
    }
}

impl KvStore for HostPoolKv {
    fn geometry(&self) -> KvGeometry {
        self.geometry
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn len(&self) -> usize {
        self.filled.iter().copied().min().unwrap_or(0)
    }
    fn path_name(&self) -> &'static str {
        "host-pool"
    }
    fn truncate(&mut self, len: usize) {
        for f in self.filled.iter_mut() {
            *f = (*f).min(len);
        }
    }
    fn export_blocks(&mut self, start: usize, len: usize) -> Option<Result<Vec<u8>, String>> {
        Some(pool_export(&self.pool, &self.filled, start, len))
    }
    fn import_blocks(
        &mut self,
        start: usize,
        len: usize,
        bytes: &[u8],
    ) -> Option<Result<(), String>> {
        Some(pool_import(
            &mut self.pool,
            &mut self.filled,
            start,
            len,
            bytes,
        ))
    }
}

impl KvAttention for HostPoolKv {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if pos != self.filled[layer] || pos >= self.capacity {
            return Err(format!(
                "host pool kv: append at {pos}, have {} (capacity {})",
                self.filled[layer], self.capacity
            ));
        }
        self.pool
            .write_token_kv(pos / KV_BLOCK_SIZE, layer, pos % KV_BLOCK_SIZE, k, v);
        self.filled[layer] = pos + 1;
        Ok(())
    }
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String> {
        if context_len > self.filled[layer] {
            return Err(format!(
                "host pool kv: attend over {context_len}, only {} stored",
                self.filled[layer]
            ));
        }
        let keys: Vec<Vec<f32>> = (0..context_len)
            .map(|p| self.row(layer, p, false))
            .collect();
        let values: Vec<Vec<f32>> = (0..context_len).map(|p| self.row(layer, p, true)).collect();
        scaled_dot_product_attention_single(
            q,
            &keys,
            &values,
            self.geometry.q_heads,
            self.geometry.kv_heads,
            self.geometry.head_dim,
            out,
        );
        Ok(0)
    }
    fn resident(&self) -> bool {
        false
    }
}

/// Host f32 cache with the reference attention (tests, and the oracle side of parity runs).
pub struct HostKv {
    geometry: KvGeometry,
    capacity: usize,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

impl HostKv {
    pub fn new(geometry: &KvGeometry, capacity: usize) -> Self {
        Self {
            geometry: *geometry,
            capacity,
            keys: vec![Vec::new(); geometry.layers],
            values: vec![Vec::new(); geometry.layers],
        }
    }
}

impl KvStore for HostKv {
    fn geometry(&self) -> KvGeometry {
        self.geometry
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn len(&self) -> usize {
        self.keys.iter().map(Vec::len).min().unwrap_or(0)
    }
    fn truncate(&mut self, len: usize) {
        for (k, v) in self.keys.iter_mut().zip(self.values.iter_mut()) {
            k.truncate(len);
            v.truncate(len);
        }
    }
}

impl KvAttention for HostKv {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if pos != self.keys[layer].len() || pos >= self.capacity {
            return Err(format!(
                "host kv: append at {pos}, have {} (capacity {})",
                self.keys[layer].len(),
                self.capacity
            ));
        }
        self.keys[layer].push(k.to_vec());
        self.values[layer].push(v.to_vec());
        Ok(())
    }
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String> {
        scaled_dot_product_attention_single(
            q,
            &self.keys[layer][..context_len],
            &self.values[layer][..context_len],
            self.geometry.q_heads,
            self.geometry.kv_heads,
            self.geometry.head_dim,
            out,
        );
        Ok(0)
    }
    fn resident(&self) -> bool {
        false
    }
}

/// Paged KV block size (tokens): the size the 8B daemon runs (`gb10_serving`), a power of two.
pub const KV_BLOCK_SIZE: usize = 16;

/// Most prompt tokens one batched prefill step takes: `PREFILL_CHUNK x top_k` expert rows must
/// fit one grouped FP8 launch (`aien_omega_fp8::MAX_GROUPS` = 64, top_k = 8).
pub const PREFILL_CHUNK: usize = 8;

/// Default prompt tokens per batched prefill step on the FP8 grouped GEMM (64 x top_k = 512 expert
/// rows, the size the kernel was measured at; the GEMM takes up to 128 tokens = 1024 rows).
pub const PREFILL_CHUNK_GEMM: usize = 64;

/// Omega's paged bf16 KV: one sequence, blocks `0..n` in order, the pool in the engine's resident
/// allocation when the driver grants it (attention reads in place), a host-mapped pool otherwise.
pub struct OmegaPagedKv {
    pool: aien_kv_cache::UnifiedKvTensorPool,
    layout: aien_omega_gpu::OmegaGpuKvLayout,
    block_ids: Vec<u32>,
    capacity: usize,
    geometry: KvGeometry,
    resident: bool,
    filled: Vec<usize>,
}

impl OmegaPagedKv {
    pub fn new(geometry: &KvGeometry, max_context: usize) -> Result<Self, String> {
        check_block_size("paged kv", geometry)?;
        let num_blocks = max_context.div_ceil(KV_BLOCK_SIZE).max(1);
        let cfg = aien_kv_cache::KvPoolConfig::for_model(
            num_blocks,
            KV_BLOCK_SIZE,
            geometry.layers,
            geometry.kv_heads,
            geometry.head_dim,
            aien_kv_cache::KvDType::Bf16,
        );
        let bytes = cfg.layout()?.total_bytes;
        let (pool, resident) = match aien_omega_gpu::resident_kv_pool_alloc(bytes) {
            Ok((ptr, len)) => {
                // SAFETY: omega owns `len` bytes at `ptr` until `resident_kv_pool_free`, which the
                // release callback runs exactly once; chip address == host address for this memory.
                let buf = unsafe {
                    aien_kv_cache::external_default_buffer(
                        ptr,
                        len,
                        ptr as u64,
                        Box::new(aien_omega_gpu::resident_kv_pool_free),
                    )
                };
                match buf {
                    Ok(b) => (
                        aien_kv_cache::UnifiedKvTensorPool::new(cfg.clone(), b)?,
                        true,
                    ),
                    Err(e) => {
                        aien_omega_gpu::resident_kv_pool_free();
                        return Err(e);
                    }
                }
            }
            Err(_) => (
                aien_kv_cache::UnifiedKvTensorPool::allocate(cfg.clone())?,
                false,
            ),
        };
        let d = pool.layout_desc();
        let layout = aien_omega_gpu::OmegaGpuKvLayout {
            block_stride_bytes: d.block_stride_bytes,
            layer_stride_bytes: d.layer_stride_bytes,
            kv_plane_stride_bytes: d.kv_plane_stride_bytes,
            token_stride_bytes: d.token_stride_bytes,
            head_stride_bytes: d.head_stride_bytes,
            pool_bytes: d.pool_bytes,
            num_blocks: d.num_blocks,
            num_layers: d.num_layers,
            block_size: d.block_size,
        };
        Ok(Self {
            pool,
            layout,
            block_ids: (0..num_blocks as u32).collect(),
            capacity: num_blocks * KV_BLOCK_SIZE,
            geometry: *geometry,
            resident,
            filled: vec![0; geometry.layers],
        })
    }

    pub fn pool_bytes(&self) -> usize {
        self.pool.total_bytes()
    }
}

impl KvStore for OmegaPagedKv {
    fn geometry(&self) -> KvGeometry {
        self.geometry
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn len(&self) -> usize {
        self.filled.iter().copied().min().unwrap_or(0)
    }
    fn truncate(&mut self, len: usize) {
        // The pool keeps its bytes; positions past `len` are rewritten by later appends.
        for f in self.filled.iter_mut() {
            *f = (*f).min(len);
        }
    }
    // The chip reads and writes this pool in place (GPU-uncached host-visible memory when
    // resident) and no launch is in flight between steps, so the host copies see the bytes the
    // attention reads.
    fn export_blocks(&mut self, start: usize, len: usize) -> Option<Result<Vec<u8>, String>> {
        Some(pool_export(&self.pool, &self.filled, start, len))
    }
    fn import_blocks(
        &mut self,
        start: usize,
        len: usize,
        bytes: &[u8],
    ) -> Option<Result<(), String>> {
        Some(pool_import(
            &mut self.pool,
            &mut self.filled,
            start,
            len,
            bytes,
        ))
    }
}

impl KvAttention for OmegaPagedKv {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if pos != self.filled[layer] || pos >= self.capacity {
            return Err(format!(
                "paged kv: append at {pos}, have {} (capacity {})",
                self.filled[layer], self.capacity
            ));
        }
        self.pool
            .write_token_kv(pos / KV_BLOCK_SIZE, layer, pos % KV_BLOCK_SIZE, k, v);
        self.filled[layer] = pos + 1;
        Ok(())
    }
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String> {
        if context_len > self.filled[layer] {
            return Err(format!(
                "paged kv: attend over {context_len}, only {} stored",
                self.filled[layer]
            ));
        }
        let nblocks = context_len.div_ceil(KV_BLOCK_SIZE);
        // SAFETY: the pool owns total_bytes() readable bytes at base_ptr() for its lifetime.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.pool.base_ptr(), self.pool.total_bytes()) };
        let info = aien_omega_gpu::paged_attention_bf16(
            q,
            bytes,
            &self.layout,
            &self.block_ids[..nblocks],
            context_len,
            layer,
            self.geometry.q_heads,
            self.geometry.kv_heads,
            self.geometry.head_dim,
            out,
        )
        .map_err(|e| {
            format!(
                "paged attention layer {layer} ctx {context_len}: {e}; {}",
                aien_omega_gpu::last_error()
            )
        })?;
        Ok(info.elapsed_ns)
    }
    fn resident(&self) -> bool {
        self.resident
    }
}

/// Generation request (field for field what the HTTP door's `GenerateParams` carries).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateParams {
    /// Hard cap on emitted tokens; stop tokens do not count.
    pub max_new_tokens: usize,
    /// `<= 0.001` means greedy.
    pub temperature: f32,
    /// `(0, 1]`; 1.0 disables nucleus filtering.
    pub top_p: f32,
    /// 0 disables.
    pub top_k: usize,
    /// Request seed; per-step seeds are `decode_sampling_seed(seed, step)`.
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
}

impl GenerateParams {
    pub fn greedy(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            stop_token_ids,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    /// A constraint said the output is complete (`StepRule::Finish`, ADR 0004 section 2.3).
    Constraint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutcome {
    pub emitted_tokens: usize,
    pub finish: FinishReason,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub prompt_tokens: usize,
    /// Prompt tokens whose KV was still valid from the previous generation (not re-prefilled).
    pub reused_prompt_tokens: usize,
    /// Sum of the per-step timings over the decode steps.
    pub decode_timing: StepTiming,
    /// Sum of the per-step timings over the prefill steps (batched steps count once).
    pub prefill_timing: StepTiming,
}

/// Where one step's time went. `*_ns` are host wall-clock sections; `chip_ns` is the sum of the
/// chip-reported kernel times; `launches` counts engine calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepTiming {
    pub total_ns: u64,
    pub attention_ns: u64,
    pub experts_ns: u64,
    pub lm_head_ns: u64,
    pub host_ns: u64,
    pub chip_ns: u64,
    pub launches: u64,
    /// Batched prefill only (both are inside `experts_ns`): routing (host logits and top-k, or the
    /// device router launches) and the host work around the expert GEMMs (sort, SwiGLU, combine).
    pub route_ns: u64,
    pub moe_host_ns: u64,
    /// One-list decode graph only (`OWN30B_DECODE_GRAPH=1`): tokens run as one command list,
    /// their submissions and the host time spent recording the lists.
    pub graph_tokens: u64,
    pub graph_launches: u64,
    pub graph_submits: u64,
    pub graph_record_ns: u64,
}

impl StepTiming {
    pub fn add(&mut self, o: &StepTiming) {
        self.graph_tokens += o.graph_tokens;
        self.graph_launches += o.graph_launches;
        self.graph_submits += o.graph_submits;
        self.graph_record_ns += o.graph_record_ns;
        self.route_ns += o.route_ns;
        self.moe_host_ns += o.moe_host_ns;
        self.total_ns += o.total_ns;
        self.attention_ns += o.attention_ns;
        self.experts_ns += o.experts_ns;
        self.lm_head_ns += o.lm_head_ns;
        self.host_ns += o.host_ns;
        self.chip_ns += o.chip_ns;
        self.launches += o.launches;
    }
}

/// What one traced layer did for one token (the parity harness replays it on the host).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerTrace {
    pub layer: usize,
    pub pos: usize,
    pub x_in: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub post: Vec<f32>,
    /// The device router's own x' (RMSNorm(x), read back), empty on the host-routing path. `post`
    /// is always the host's independent RMSNorm, so the MoE oracle checks the router too.
    pub post_device: Vec<f32>,
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub x_out: Vec<f32>,
}

/// Samples like the HTTP door's `sample_next`: top-k mask, then the workspace sampler with a
/// per-step seed, so greedy and seeded replies match across engines.
pub fn sample_next(logits: &[f32], params: &GenerateParams, step: usize) -> u32 {
    let masked;
    let logits = if params.top_k > 0 && params.top_k < logits.len() && params.temperature > 0.001 {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut out = vec![f32::NEG_INFINITY; logits.len()];
        for &i in &order[..params.top_k] {
            out[i] = logits[i];
        }
        masked = out;
        masked.as_slice()
    } else {
        logits
    };
    let sp = SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        max_tokens: 1,
        stop_token_ids: Vec::new(),
    };
    sample_with_params(logits, &sp, decode_sampling_seed(params.seed, step)).0
}

/// Every run setting of a [`Qwen3MoeOmegaEngine`] that can change a K/V byte (persistent KV
/// engine identity, sc#505). When unsure whether a setting moves a byte it is listed (fail
/// closed: a needless entry only costs a cache miss). The caller adds the settings that live
/// outside the engine (weight format, the omega environment switches, the build).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmegaKvSwitches {
    /// "fused" (one-launch attention, bf16 RNE on the chip), "split" (host norm and RoPE, the
    /// paged pool), or a test cache's name.
    pub attention_path: &'static str,
    /// Decode steps route with the device router.
    pub device_routing: bool,
    /// Prompt tokens per batched prefill step (the batch shape of the activation rounding).
    pub prefill_chunk: usize,
    /// Batched prefill on the FP8 grouped GEMM instead of the grouped GEMV.
    pub prefill_gemm: bool,
    /// GEMM prefill routes with the device router.
    pub prefill_device_routing: bool,
    /// Whole-prompt prefill (changes activation rounding, chip gate W-RC).
    pub whole_prompt: bool,
    /// Router logits and MoE combine of the whole-prompt prefill on the chip.
    pub whole_prompt_device_host: bool,
    /// The one-list decode graph (`OWN30B_DECODE_GRAPH=1`).
    pub decode_graph: bool,
    /// lm_head on the bf16 GEMV (logits only; listed anyway).
    pub lm_head_gemv: bool,
    /// Negative control: routing deliberately wrong. Persistence must refuse it.
    pub routing_mutant: bool,
    /// Parity tracing on (records activations; listed anyway).
    pub trace_layers: bool,
}

/// The Stage A engine: one sequence, token-at-a-time prefill and decode.
pub struct Qwen3MoeOmegaEngine {
    /// The one-list decode graph (built at the first graph decode step). Declared first so it is
    /// dropped before the weights and KV cache whose omega handles it holds.
    dgraph: Option<dgraph::GraphState>,
    /// `OWN30B_DECODE_GRAPH=1` at creation: decode steps run as one command list.
    decode_graph: bool,
    weights: Qwen3MoeOmegaWeights,
    kv: Box<dyn KvAttention>,
    /// Layers whose activations are recorded into `traces` (parity harness).
    pub trace_layers: Vec<usize>,
    pub traces: Vec<LayerTrace>,
    /// Negative control: every chosen expert id is replaced by its neighbour (weights stay), so
    /// every expert window is a wrong expert. The layer oracle must then disagree loudly.
    pub routing_mutant: bool,
    /// Route decode steps with the device router launch when the weights carry one. The host
    /// routing mutant forces the host path for its steps.
    pub device_routing: bool,
    pub last_timing: StepTiming,
    /// Prompt tokens per prefill step (1 = token-at-a-time). Batched prefill needs FP8 attention
    /// and experts; with the grouped GEMM a step takes up to 128 tokens, with the grouped GEMV 8
    /// (`step_batch` splits a larger chunk to fit).
    pub prefill_chunk: usize,
    /// Batched prefill uses the FP8 grouped GEMM (when every projection has it) instead of the
    /// grouped GEMV.
    pub prefill_gemm: bool,
    /// GEMM prefill routes with the device router (one launch per token and layer) instead of
    /// the host.
    pub prefill_device_routing: bool,
    pos: usize,
    /// Tokens whose KV is stored at positions `0..pos` (prefix reuse across generations).
    history: Vec<u32>,
    /// Whole-prompt prefill driver (OFF = None; `set_whole_prompt`). See `whole_prompt.rs`.
    whole_prompt: Option<Box<whole_prompt::EngineWholePrompt>>,
    /// The host head (ADR 0004 adapter): the last forward call's final hidden state, unread.
    /// `None` once read or forgotten. Empty on the decode graph, whose head is its pending token.
    head_hidden: Option<Vec<f32>>,
    /// The logits `ForwardPass::next` lends out (the last head read).
    head_logits: Vec<f32>,
}

impl Qwen3MoeOmegaEngine {
    pub fn new(weights: Qwen3MoeOmegaWeights, kv: Box<dyn KvAttention>) -> Self {
        let device_routing = weights.device_router.is_some();
        let prefill_gemm = gemm_everywhere(&weights);
        Self {
            dgraph: None,
            decode_graph: dgraph::requested(),
            weights,
            kv,
            trace_layers: Vec::new(),
            traces: Vec::new(),
            routing_mutant: false,
            device_routing,
            last_timing: StepTiming::default(),
            prefill_chunk: if prefill_gemm {
                PREFILL_CHUNK_GEMM
            } else {
                PREFILL_CHUNK
            },
            prefill_gemm,
            prefill_device_routing: false,
            pos: 0,
            history: Vec::new(),
            whole_prompt: None,
            head_hidden: None,
            head_logits: Vec::new(),
        }
    }

    /// Whether every layer runs attention and experts on the FP8 grouped GEMV (the only
    /// kernels that take one activation row per group, which batched prefill needs).
    pub fn batched_prefill_possible(&self) -> bool {
        self.weights.layers.iter().all(|l| {
            matches!(l.attn, AttnWeights::Fp8 { .. })
                && matches!(l.experts, ExpertWeights::Fp8 { .. })
        })
    }

    /// Runs `tokens` (consecutive positions) through every layer, layer-major: one grouped
    /// q|k|v GEMV with a row per token, one attention launch per token, one grouped o GEMV,
    /// then one gate|up and one down launch for all `tokens x top_k` expert rows. Returns the
    /// final-normed hidden state of the LAST token (the earlier ones are prefill only). The
    /// maths per token is the same as [`Self::step`] in the same order (the MoE sum is
    /// accumulated per token before the residual add), so greedy tokens must not change.
    /// Falls back to `step` per token when tracing, for one token, or without the FP8 kernels.
    pub fn step_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let m = tokens.len();
        if m == 0 {
            return Err("step_batch: empty batch".into());
        }
        let top_k = self.weights.config.top_k;
        if m == 1 || !self.trace_layers.is_empty() || !self.batched_prefill_possible() {
            let mut hidden = Vec::new();
            let mut timing = StepTiming::default();
            for &t in tokens {
                hidden = self.step(t)?;
                timing.add(&self.last_timing);
            }
            self.last_timing = timing;
            return Ok(hidden);
        }
        let gemm = self.prefill_gemm && gemm_everywhere(&self.weights);
        let limit = if gemm {
            aien_omega_fp8::GEMM_MAX_ROWS / top_k
        } else {
            aien_omega_fp8::MAX_GROUPS / top_k
        };
        if m > limit {
            // A chunk wider than one launch takes: run it as consecutive launches-sized pieces.
            let mut hidden = Vec::new();
            let mut timing = StepTiming::default();
            for piece in tokens.chunks(limit) {
                hidden = self.step_batch(piece)?;
                timing.add(&self.last_timing);
            }
            self.last_timing = timing;
            return Ok(hidden);
        }
        if gemm {
            return self.step_batch_gemm(tokens);
        }
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let (e, i, qkv_n) = (c.experts, c.moe_intermediate, c.qkv_n());
        let pos0 = self.pos;
        if pos0 + m > self.kv.capacity() {
            return Err(format!(
                "context full: {pos0} + {m} of {}",
                self.kv.capacity()
            ));
        }
        let g = m * top_k;
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| self.weights.embed_row(t))
            .collect();
        let mut x_norm = vec![0.0f32; m * h];
        let mut qkv_all = vec![0.0f32; m * qkv_n];
        let mut qn = vec![0.0f32; qd];
        let mut kn = vec![0.0f32; kvd];
        let mut attn_all = vec![0.0f32; m * qd];
        let mut proj_all = vec![0.0f32; m * h];
        let mut post = vec![0.0f32; m * h];
        let mut logits = vec![0.0f32; e];
        let mut ids = vec![0u32; g];
        let mut wts = vec![0.0f32; g];
        let mut post_rep = vec![0.0f32; g * h];
        let mut gu_all = vec![0.0f32; g * 2 * i];
        let mut act_all = vec![0.0f32; g * i];
        let mut down_all = vec![0.0f32; g * h];
        let mut moe_out = vec![0.0f32; h];
        let dense_ids = vec![0u32; m];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) =
                (&layer.attn, &layer.experts)
            else {
                return Err(format!(
                    "layer {layer_idx}: batched prefill needs FP8 attention and experts"
                ));
            };
            // --- attention: one grouped q|k|v launch (a row per token), one attention launch per token, one grouped o launch ---
            let t0 = Instant::now();
            for r in 0..m {
                rmsnorm(
                    &x[r * h..(r + 1) * h],
                    &layer.input_norm,
                    eps,
                    &mut x_norm[r * h..(r + 1) * h],
                );
            }
            timing.host_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            timing.chip_ns += qkv
                .grouped_f32(&dense_ids, &x_norm, &mut qkv_all)
                .map_err(|e| format!("layer {layer_idx} qkv x{m}: {e}"))?;
            timing.launches += 1;
            for r in 0..m {
                let pos = pos0 + r;
                let qkv_r = &qkv_all[r * qkv_n..(r + 1) * qkv_n];
                let out_r = &mut attn_all[r * qd..(r + 1) * qd];
                match self.kv.fused_attend(layer_idx, pos, qkv_r, eps, out_r) {
                    Some(res) => timing.chip_ns += res?,
                    None => {
                        let t2 = Instant::now();
                        rmsnorm_per_head(&qkv_r[..qd], &layer.q_norm, eps, c.q_heads, hd, &mut qn);
                        rmsnorm_per_head(
                            &qkv_r[qd..qd + kvd],
                            &layer.k_norm,
                            eps,
                            c.kv_heads,
                            hd,
                            &mut kn,
                        );
                        apply_rope(
                            &mut qn,
                            &mut kn,
                            pos,
                            c.q_heads,
                            c.kv_heads,
                            hd,
                            c.rope_theta,
                        );
                        self.kv.append(layer_idx, pos, &kn, &qkv_r[qd + kvd..])?;
                        timing.host_ns += t2.elapsed().as_nanos() as u64;
                        timing.chip_ns += self.kv.attend(layer_idx, &qn, pos + 1, out_r)?;
                    }
                }
                timing.launches += 1;
            }
            timing.chip_ns += o
                .grouped_f32(&dense_ids, &attn_all, &mut proj_all)
                .map_err(|e| format!("layer {layer_idx} o x{m}: {e}"))?;
            timing.launches += 1;
            timing.attention_ns += t1.elapsed().as_nanos() as u64;

            // --- residual, post norm, routing per token (host); the expert row table for the chunk ---
            let t3 = Instant::now();
            for r in 0..m {
                let xr = &mut x[r * h..(r + 1) * h];
                for (xi, p) in xr.iter_mut().zip(&proj_all[r * h..(r + 1) * h]) {
                    *xi += p;
                }
                let pr = &mut post[r * h..(r + 1) * h];
                rmsnorm(xr, &layer.post_norm, eps, pr);
                for (ex, l) in logits.iter_mut().enumerate() {
                    *l = layer.router[ex * h..(ex + 1) * h]
                        .iter()
                        .zip(pr.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }
                let plan = MoeBatchPlan::from_logits(&logits, 1, e, top_k)
                    .map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
                for (slot, (&ex, &w)) in
                    plan.topk_experts.iter().zip(&plan.topk_weights).enumerate()
                {
                    let gi = r * top_k + slot;
                    ids[gi] = if self.routing_mutant {
                        (ex + 1) % e as u32
                    } else {
                        ex
                    };
                    wts[gi] = w;
                    post_rep[gi * h..(gi + 1) * h].copy_from_slice(pr);
                }
            }
            timing.host_ns += t3.elapsed().as_nanos() as u64;

            // --- experts: one gate|up launch and one down launch for every (token, expert) row ---
            let t4 = Instant::now();
            timing.chip_ns += gate_up
                .grouped_f32(&ids, &post_rep, &mut gu_all)
                .map_err(|e| format!("layer {layer_idx} gate_up fp8 x{g}: {e}"))?;
            timing.launches += 1;
            let th = Instant::now();
            for gi in 0..g {
                swiglu(
                    &gu_all[gi * 2 * i..(gi + 1) * 2 * i],
                    i,
                    &mut act_all[gi * i..(gi + 1) * i],
                );
            }
            timing.host_ns += th.elapsed().as_nanos() as u64;
            timing.chip_ns += down
                .grouped_f32(&ids, &act_all, &mut down_all)
                .map_err(|e| format!("layer {layer_idx} down fp8 x{g}: {e}"))?;
            timing.launches += 1;
            let th = Instant::now();
            for r in 0..m {
                moe_out.iter_mut().for_each(|v| *v = 0.0);
                for slot in 0..top_k {
                    let gi = r * top_k + slot;
                    let w = wts[gi];
                    for (mo, d) in moe_out.iter_mut().zip(&down_all[gi * h..(gi + 1) * h]) {
                        *mo += w * d;
                    }
                }
                for (xi, mo) in x[r * h..(r + 1) * h].iter_mut().zip(&moe_out) {
                    *xi += mo;
                }
            }
            timing.host_ns += th.elapsed().as_nanos() as u64;
            timing.experts_ns += t4.elapsed().as_nanos() as u64;
        }
        let t5 = Instant::now();
        let mut hidden = vec![0.0f32; h];
        rmsnorm(
            &x[(m - 1) * h..],
            &self.weights.final_norm,
            eps,
            &mut hidden,
        );
        timing.host_ns += t5.elapsed().as_nanos() as u64;
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hidden)
    }

    /// Batched prefill on the FP8 grouped GEMM (OWN-30B cut 10): the same layer-major maths as the
    /// GEMV path above, but q|k|v, o, gate|up and down are each ONE launch for the whole chunk
    /// (up to 128 tokens = 1024 expert rows), and every distinct expert's weights are streamed
    /// once per pass of rows instead of once per row. Attention stays one fused launch per token.
    /// The expert rows are sorted by expert (a counting sort on the host); the GEMM output comes
    /// back in sorted order and the per-token MoE sum is accumulated in slot order, exactly as
    /// before, so the result is bit-identical to the GEMV path. Routing is the host's (logits on
    /// worker threads) or, with `prefill_device_routing`, the device router per token.
    fn step_batch_gemm(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let m = tokens.len();
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let top_k = c.top_k;
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let (e, i, qkv_n) = (c.experts, c.moe_intermediate, c.qkv_n());
        let pos0 = self.pos;
        if pos0 + m > self.kv.capacity() {
            return Err(format!(
                "context full: {pos0} + {m} of {}",
                self.kv.capacity()
            ));
        }
        let g = m * top_k;
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| self.weights.embed_row(t))
            .collect();
        let mut x_norm = vec![0.0f32; m * h];
        let mut qkv_all = vec![0.0f32; m * qkv_n];
        let mut qn = vec![0.0f32; qd];
        let mut kn = vec![0.0f32; kvd];
        let mut attn_all = vec![0.0f32; m * qd];
        let mut proj_all = vec![0.0f32; m * h];
        let mut post = vec![0.0f32; m * h];
        let mut logits = vec![0.0f32; m * e];
        let mut ids = vec![0u32; g];
        let mut wts = vec![0.0f32; g];
        let mut sorted_expert = vec![0u32; g];
        let mut sorted_token = vec![0u32; g];
        let mut inv = vec![0u32; g];
        let mut gu_all = vec![0.0f32; g * 2 * i];
        let mut act_all = vec![0.0f32; g * i];
        let mut down_all = vec![0.0f32; g * h];
        let dense_ids = vec![0u32; m];
        let rows_m: Vec<u32> = (0..m as u32).collect();
        let rows_g: Vec<u32> = (0..g as u32).collect();
        let device_router = match &self.weights.device_router {
            Some(r) if self.prefill_device_routing && !self.routing_mutant => Some(r.clone()),
            _ => None,
        };

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) =
                (&layer.attn, &layer.experts)
            else {
                return Err(format!(
                    "layer {layer_idx}: batched prefill needs FP8 attention and experts"
                ));
            };
            // --- attention: one q|k|v GEMM, one fused attention launch per token, one o GEMM ---
            let t0 = Instant::now();
            for r in 0..m {
                rmsnorm(
                    &x[r * h..(r + 1) * h],
                    &layer.input_norm,
                    eps,
                    &mut x_norm[r * h..(r + 1) * h],
                );
            }
            timing.host_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            timing.chip_ns += qkv
                .gemm_f32(&dense_ids, &rows_m, &x_norm, &mut qkv_all)
                .map_err(|e| format!("layer {layer_idx} qkv gemm x{m}: {e}"))?;
            timing.launches += 1;
            for r in 0..m {
                let pos = pos0 + r;
                let qkv_r = &qkv_all[r * qkv_n..(r + 1) * qkv_n];
                let out_r = &mut attn_all[r * qd..(r + 1) * qd];
                match self.kv.fused_attend(layer_idx, pos, qkv_r, eps, out_r) {
                    Some(res) => timing.chip_ns += res?,
                    None => {
                        let t2 = Instant::now();
                        rmsnorm_per_head(&qkv_r[..qd], &layer.q_norm, eps, c.q_heads, hd, &mut qn);
                        rmsnorm_per_head(
                            &qkv_r[qd..qd + kvd],
                            &layer.k_norm,
                            eps,
                            c.kv_heads,
                            hd,
                            &mut kn,
                        );
                        apply_rope(
                            &mut qn,
                            &mut kn,
                            pos,
                            c.q_heads,
                            c.kv_heads,
                            hd,
                            c.rope_theta,
                        );
                        self.kv.append(layer_idx, pos, &kn, &qkv_r[qd + kvd..])?;
                        timing.host_ns += t2.elapsed().as_nanos() as u64;
                        timing.chip_ns += self.kv.attend(layer_idx, &qn, pos + 1, out_r)?;
                    }
                }
                timing.launches += 1;
            }
            timing.chip_ns += o
                .gemm_f32(&dense_ids, &rows_m, &attn_all, &mut proj_all)
                .map_err(|e| format!("layer {layer_idx} o gemm x{m}: {e}"))?;
            timing.launches += 1;
            timing.attention_ns += t1.elapsed().as_nanos() as u64;

            // --- residual and post norm for every token, then routing ---
            let t3 = Instant::now();
            for r in 0..m {
                let xr = &mut x[r * h..(r + 1) * h];
                for (xi, p) in xr.iter_mut().zip(&proj_all[r * h..(r + 1) * h]) {
                    *xi += p;
                }
                rmsnorm(xr, &layer.post_norm, eps, &mut post[r * h..(r + 1) * h]);
            }
            timing.host_ns += t3.elapsed().as_nanos() as u64;
            let t4 = Instant::now();
            let tr = Instant::now();
            if let Some(router) = &device_router {
                for r in 0..m {
                    let (a, b) = (r * top_k, (r + 1) * top_k);
                    let (ns, _) = router
                        .route(
                            layer_idx,
                            &x[r * h..(r + 1) * h],
                            eps,
                            &mut ids[a..b],
                            &mut wts[a..b],
                        )
                        .map_err(|e| {
                            format!("layer {layer_idx} device router (token {r} of {m}): {e}")
                        })?;
                    timing.chip_ns += ns;
                    timing.launches += 1;
                }
            } else {
                router_logits(&layer.router, &post, h, e, &mut logits);
                let plan = MoeBatchPlan::from_logits(&logits, m, e, top_k)
                    .map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
                for (dst, &ex) in ids.iter_mut().zip(&plan.topk_experts) {
                    *dst = if self.routing_mutant {
                        (ex + 1) % e as u32
                    } else {
                        ex
                    };
                }
                wts.copy_from_slice(&plan.topk_weights);
            }
            timing.route_ns += tr.elapsed().as_nanos() as u64;

            // --- the (expert, row) table sorted by expert (stable counting sort) ---
            let ts = Instant::now();
            let mut start = vec![0u32; e + 1];
            for &ex in &ids {
                start[ex as usize + 1] += 1;
            }
            for ex in 0..e {
                start[ex + 1] += start[ex];
            }
            for (gi, &ex) in ids.iter().enumerate() {
                let slot = &mut start[ex as usize];
                sorted_expert[*slot as usize] = ex;
                sorted_token[*slot as usize] = (gi / top_k) as u32;
                inv[gi] = *slot;
                *slot += 1;
            }
            timing.moe_host_ns += ts.elapsed().as_nanos() as u64;

            // --- experts: one gate|up GEMM, SwiGLU, one down GEMM, over every (token, expert) row ---
            timing.chip_ns += gate_up
                .gemm_f32(&sorted_expert, &sorted_token, &post, &mut gu_all)
                .map_err(|e| format!("layer {layer_idx} gate_up gemm x{g}: {e}"))?;
            timing.launches += 1;
            let th = Instant::now();
            swiglu_rows(&gu_all, &mut act_all, i);
            timing.moe_host_ns += th.elapsed().as_nanos() as u64;
            timing.chip_ns += down
                .gemm_f32(&sorted_expert, &rows_g, &act_all, &mut down_all)
                .map_err(|e| format!("layer {layer_idx} down gemm x{g}: {e}"))?;
            timing.launches += 1;
            let th = Instant::now();
            combine_rows(&mut x, &down_all, &wts, &inv, h, top_k);
            timing.moe_host_ns += th.elapsed().as_nanos() as u64;
            timing.experts_ns += t4.elapsed().as_nanos() as u64;
        }
        let t5 = Instant::now();
        let mut hidden = vec![0.0f32; h];
        rmsnorm(
            &x[(m - 1) * h..],
            &self.weights.final_norm,
            eps,
            &mut hidden,
        );
        timing.host_ns += t5.elapsed().as_nanos() as u64;
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hidden)
    }

    /// The production constructor: loads the checkpoint onto Omega and allocates the paged KV.
    pub fn load_omega(
        dir: &Path,
        max_context: usize,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, LoadReport), String> {
        Self::load_omega_with(dir, max_context, WeightFormat::Bf16, progress)
    }

    /// As [`Self::load_omega`] with a chosen weight format (FP8 needs omega's FP8 GEMV linked).
    /// Attention takes the fused one-launch path when it is linked ([`AttnPath::Auto`]).
    pub fn load_omega_with(
        dir: &Path,
        max_context: usize,
        format: WeightFormat,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, LoadReport), String> {
        Self::load_omega_with_attn(dir, max_context, format, AttnPath::Auto, progress)
    }

    /// As [`Self::load_omega_with`] with the attention path chosen explicitly.
    pub fn load_omega_with_attn(
        dir: &Path,
        max_context: usize,
        format: WeightFormat,
        attn: AttnPath,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, LoadReport), String> {
        Self::load_omega_paths(
            dir,
            max_context,
            format,
            attn,
            DevicePaths::default(),
            progress,
        )
    }

    /// As [`Self::load_omega_with_attn`] with the router and lm_head paths chosen
    /// ([`DevicePaths::default`] takes the device ones when linked).
    pub fn load_omega_paths(
        dir: &Path,
        max_context: usize,
        format: WeightFormat,
        attn: AttnPath,
        paths: DevicePaths,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, LoadReport), String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine (build with AIEN_OMEGA_DIR or AIEN_OMEGA_GPU_LIB)".into());
        }
        let (weights, report) =
            load_qwen3_moe_omega_paths(dir, &OmegaModelUploader, format, paths, progress)?;
        let geometry = KvGeometry::from(&weights.config);
        let fused = match attn {
            AttnPath::Split => false,
            AttnPath::Fused => true,
            AttnPath::Auto => {
                OmegaModelUploader::uses_dattn_api()
                    && crate::qwen3_moe_omega_dattn::OmegaFusedKv::fits(&geometry)
            }
        };
        let kv: Box<dyn KvAttention> = if fused {
            let kv = crate::qwen3_moe_omega_dattn::OmegaFusedKv::new(
                &geometry,
                max_context,
                &weights.layers,
            )?;
            progress(&format!(
                "kv fused cache {} MB resident=true",
                kv.kv_bytes() >> 20
            ));
            Box::new(kv)
        } else {
            let kv = OmegaPagedKv::new(&geometry, max_context)?;
            progress(&format!(
                "kv pool {} MB resident={}",
                kv.pool_bytes() >> 20,
                kv.resident()
            ));
            Box::new(kv)
        };
        Ok((Self::new(weights, kv), report))
    }

    /// "split" or "fused": which attention path this engine runs.
    pub fn attention_path(&self) -> &'static str {
        self.kv.path_name()
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.weights.config
    }

    pub fn weights(&self) -> &Qwen3MoeOmegaWeights {
        &self.weights
    }

    pub fn max_context_tokens(&self) -> usize {
        self.kv.capacity()
    }

    pub fn kv_resident(&self) -> bool {
        self.kv.resident()
    }

    /// Current sequence length (tokens already in the KV).
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.dgraph_clear();
        self.kv.truncate(0);
        self.pos = 0;
        self.history.clear();
        self.traces.clear();
    }

    /// Tokens whose KV is stored (positions `0..position()`).
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Keeps the first `len` stored tokens and forgets the rest; the next step is at `len`.
    pub fn rewind(&mut self, len: usize) {
        let len = len.min(self.pos);
        self.dgraph_clear();
        self.kv.truncate(len);
        self.history.truncate(len);
        self.pos = len;
    }

    /// Tokens `generate` would keep from the stored history for `prompt`: the longest common
    /// prefix, capped at `prompt.len() - 1` (the last prompt token is always stepped). The same
    /// rule as `generate_inner`; persistent KV (sc#505) restores only when the store beats it.
    pub fn reusable_prefix(&self, prompt: &[u32]) -> usize {
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        common.min(prompt.len().saturating_sub(1))
    }

    /// Persistent KV park (sc#505): the K/V of stored positions `start..start + len` as the
    /// [`kvio`] payload (`start` a multiple of [`KV_BLOCK_SIZE`], `start + len <= position()`).
    pub fn export_kv(&mut self, start: usize, len: usize) -> Result<Vec<u8>, String> {
        if start + len > self.pos {
            return Err(format!(
                "kv export {start}+{len}: only {} positions stored",
                self.pos
            ));
        }
        self.kv.export_blocks(start, len).unwrap_or_else(|| {
            Err(format!(
                "the {} attention cache cannot export its KV",
                self.kv.path_name()
            ))
        })
    }

    /// Persistent KV restore (sc#505): replaces the stored sequence by `tokens` with the K/V of
    /// `payload` (a [`kvio`] payload of `tokens.len()` positions from position 0). On success the
    /// engine is exactly as if it had stepped `tokens` itself and kept them in its history, so the
    /// next `generate` reuses them by the usual prefix rule. On any error the engine is reset
    /// (empty history): the next `generate` prefills from position 0, never from a partial prefix.
    pub fn restore_kv(&mut self, tokens: &[u32], payload: &[u8]) -> Result<(), String> {
        self.reset();
        if tokens.is_empty() || tokens.len() >= self.kv.capacity() {
            return Err(format!(
                "kv restore of {} tokens, context is {}",
                tokens.len(),
                self.kv.capacity()
            ));
        }
        let r = self
            .kv
            .import_blocks(0, tokens.len(), payload)
            .unwrap_or_else(|| {
                Err(format!(
                    "the {} attention cache cannot import KV",
                    self.kv.path_name()
                ))
            });
        match r {
            Ok(()) => {
                self.history = tokens.to_vec();
                self.pos = tokens.len();
                Ok(())
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    /// Every setting of this engine that can change a K/V byte, for the persistent KV engine
    /// identity (sc#505). See [`OmegaKvSwitches`].
    pub fn kv_switches(&self) -> OmegaKvSwitches {
        OmegaKvSwitches {
            attention_path: self.kv.path_name(),
            device_routing: self.device_routing,
            prefill_chunk: self.prefill_chunk,
            prefill_gemm: self.prefill_gemm,
            prefill_device_routing: self.prefill_device_routing,
            whole_prompt: self.whole_prompt_on(),
            whole_prompt_device_host: self.whole_prompt_device_host_on(),
            decode_graph: self.decode_graph,
            lm_head_gemv: matches!(self.weights.lm_head, LmHead::Gemv(_)),
            routing_mutant: self.routing_mutant,
            trace_layers: !self.trace_layers.is_empty(),
        }
    }

    /// Runs `token` at the next position through every layer. Returns the final-normed hidden
    /// state (feed it to `logits`). Records `last_timing`.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let (e, top_k, i) = (c.experts, c.top_k, c.moe_intermediate);
        let pos = self.pos;
        if pos >= self.kv.capacity() {
            return Err(format!("context full: {pos} of {}", self.kv.capacity()));
        }
        let mut x = self.weights.embed_row(token);
        let mut x_norm = vec![0.0f32; h];
        let mut qkv_out = vec![0.0f32; c.qkv_n()];
        let mut qn = vec![0.0f32; qd];
        let mut kn = vec![0.0f32; kvd];
        let mut attn_out = vec![0.0f32; qd];
        let mut attn_proj = vec![0.0f32; h];
        let mut post = vec![0.0f32; h];
        let mut logits = vec![0.0f32; e];
        let mut gu_out = vec![0.0f32; top_k * 2 * i];
        let mut act = vec![0.0f32; i];
        let mut down_out = vec![0.0f32; h];
        let mut moe_out = vec![0.0f32; h];
        let mut col0s: Vec<u32> = vec![0; top_k];
        // FP8 grouped path: all top_k activations and down outputs in one buffer each.
        let mut act_all = vec![0.0f32; top_k * i];
        let mut down_all = vec![0.0f32; top_k * h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let traced = self.trace_layers.contains(&layer_idx);
            let x_in = if traced { x.clone() } else { Vec::new() };

            // --- attention ---
            let t0 = Instant::now();
            rmsnorm(&x, &layer.input_norm, eps, &mut x_norm);
            timing.host_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            timing.chip_ns += match &layer.attn {
                AttnWeights::Bf16 { qkv, .. } => qkv.matmul_f32(1, &x_norm, &mut qkv_out),
                AttnWeights::Fp8 { qkv, .. } => qkv.grouped_f32(&[0], &x_norm, &mut qkv_out),
            }
            .map_err(|e| format!("layer {layer_idx} qkv: {e}"))?;
            timing.launches += 1;
            match self
                .kv
                .fused_attend(layer_idx, pos, &qkv_out, eps, &mut attn_out)
            {
                // One launch: norms, rope, KV append and attention on the chip.
                Some(r) => timing.chip_ns += r?,
                // Split path: host norms and rope, KV write, then the paged attention launch.
                None => {
                    let t2 = Instant::now();
                    rmsnorm_per_head(&qkv_out[..qd], &layer.q_norm, eps, c.q_heads, hd, &mut qn);
                    rmsnorm_per_head(
                        &qkv_out[qd..qd + kvd],
                        &layer.k_norm,
                        eps,
                        c.kv_heads,
                        hd,
                        &mut kn,
                    );
                    apply_rope(
                        &mut qn,
                        &mut kn,
                        pos,
                        c.q_heads,
                        c.kv_heads,
                        hd,
                        c.rope_theta,
                    );
                    self.kv.append(layer_idx, pos, &kn, &qkv_out[qd + kvd..])?;
                    timing.host_ns += t2.elapsed().as_nanos() as u64;
                    timing.chip_ns += self.kv.attend(layer_idx, &qn, pos + 1, &mut attn_out)?;
                }
            }
            timing.launches += 1;
            timing.chip_ns += match &layer.attn {
                AttnWeights::Bf16 { o, .. } => o.matmul_f32(1, &attn_out, &mut attn_proj),
                AttnWeights::Fp8 { o, .. } => o.grouped_f32(&[0], &attn_out, &mut attn_proj),
            }
            .map_err(|e| format!("layer {layer_idx} o: {e}"))?;
            timing.launches += 1;
            timing.attention_ns += t1.elapsed().as_nanos() as u64;

            // --- residual, post norm, routing (host) ---
            let t3 = Instant::now();
            for (xi, p) in x.iter_mut().zip(&attn_proj) {
                *xi += p;
            }
            // The device router takes the residual row and leaves the normed row and the gate|up
            // group table on the chip: no host RMSNorm, router matmul, top-k or table build.
            // The host routing mutant (a host-side swap of the chosen ids) needs the host path.
            let device_router = match (&self.weights.device_router, &layer.experts) {
                (Some(r), ExpertWeights::Fp8 { .. })
                    if self.device_routing && !self.routing_mutant =>
                {
                    Some(r)
                }
                _ => None,
            };
            let mut addrs: Option<RouteAddrs> = None;
            let mut post_device: Vec<f32> = Vec::new();
            let t4;
            let (experts, topk_weights): (Vec<u32>, Vec<f32>) = if let Some(router) = device_router
            {
                timing.host_ns += t3.elapsed().as_nanos() as u64;
                t4 = Instant::now();
                let (mut ids, mut wts) = ([0u32; 16], [0.0f32; 16]);
                let (ns, a) = router
                    .route(layer_idx, &x, eps, &mut ids[..top_k], &mut wts[..top_k])
                    .map_err(|e| format!("layer {layer_idx} device router: {e}"))?;
                timing.chip_ns += ns;
                timing.launches += 1;
                addrs = Some(a);
                if traced {
                    // The oracle input is the host's own RMSNorm; the device x' is kept beside it.
                    rmsnorm(&x, &layer.post_norm, eps, &mut post);
                    post_device = vec![0.0f32; h];
                    router
                        .read_post(&mut post_device)
                        .map_err(|e| format!("layer {layer_idx} router readback: {e}"))?;
                }
                (ids[..top_k].to_vec(), wts[..top_k].to_vec())
            } else {
                rmsnorm(&x, &layer.post_norm, eps, &mut post);
                for (ex, l) in logits.iter_mut().enumerate() {
                    *l = layer.router[ex * h..(ex + 1) * h]
                        .iter()
                        .zip(&post)
                        .map(|(a, b)| a * b)
                        .sum();
                }
                let plan = MoeBatchPlan::from_logits(&logits, 1, e, top_k)
                    .map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
                let mut experts: Vec<u32> = plan.topk_experts.clone();
                if self.routing_mutant {
                    experts.iter_mut().for_each(|ex| *ex = (*ex + 1) % e as u32);
                }
                for (slot, &ex) in experts.iter().enumerate() {
                    col0s[slot] = c.gate_up_col0(ex as usize) as u32;
                }
                timing.host_ns += t3.elapsed().as_nanos() as u64;
                t4 = Instant::now();
                (experts, plan.topk_weights.clone())
            };

            moe_out.iter_mut().for_each(|v| *v = 0.0);
            match &layer.experts {
                // --- bf16 windows: 1 gate|up call over top_k windows, top_k down calls ---
                ExpertWeights::Bf16 { gate_up, down } => {
                    timing.chip_ns += gate_up
                        .windows_f32(1, &post, &col0s, 2 * i, &mut gu_out)
                        .map_err(|e| format!("layer {layer_idx} gate_up: {e}"))?;
                    timing.launches += top_k as u64;
                    for (slot, &ex) in experts.iter().enumerate() {
                        let th = Instant::now();
                        swiglu(&gu_out[slot * 2 * i..(slot + 1) * 2 * i], i, &mut act);
                        timing.host_ns += th.elapsed().as_nanos() as u64;
                        let col0 = c.down_col0(ex as usize) as u32;
                        timing.chip_ns += down
                            .windows_f32(1, &act, std::slice::from_ref(&col0), h, &mut down_out)
                            .map_err(|e| format!("layer {layer_idx} down expert {ex}: {e}"))?;
                        timing.launches += 1;
                        let w = topk_weights[slot];
                        for (m, d) in moe_out.iter_mut().zip(&down_out) {
                            *m += w * d;
                        }
                    }
                }
                // --- FP8 grouped: one gate|up launch (shared row), one down launch (a row per expert) ---
                ExpertWeights::Fp8 { gate_up, down } => {
                    timing.chip_ns += match addrs {
                        // The router left the normed row and the group table on the chip.
                        Some(a) => gate_up.grouped_dev(top_k, a.tab_gu_va, a.x_va, &mut gu_out),
                        None => gate_up.grouped_f32(&experts, &post, &mut gu_out),
                    }
                    .map_err(|e| format!("layer {layer_idx} gate_up fp8: {e}"))?;
                    timing.launches += 1;
                    let th = Instant::now();
                    for slot in 0..top_k {
                        swiglu(
                            &gu_out[slot * 2 * i..(slot + 1) * 2 * i],
                            i,
                            &mut act_all[slot * i..(slot + 1) * i],
                        );
                    }
                    timing.host_ns += th.elapsed().as_nanos() as u64;
                    timing.chip_ns += down
                        .grouped_f32(&experts, &act_all, &mut down_all)
                        .map_err(|e| format!("layer {layer_idx} down fp8: {e}"))?;
                    timing.launches += 1;
                    let th = Instant::now();
                    for (slot, &w) in topk_weights.iter().enumerate() {
                        for (m, d) in moe_out.iter_mut().zip(&down_all[slot * h..(slot + 1) * h]) {
                            *m += w * d;
                        }
                    }
                    timing.host_ns += th.elapsed().as_nanos() as u64;
                }
            }
            for (xi, m) in x.iter_mut().zip(&moe_out) {
                *xi += m;
            }
            timing.experts_ns += t4.elapsed().as_nanos() as u64;

            if traced {
                self.traces.push(LayerTrace {
                    layer: layer_idx,
                    pos,
                    x_in,
                    attn_proj: attn_proj.clone(),
                    post: post.clone(),
                    post_device,
                    experts: experts.clone(),
                    weights: topk_weights.clone(),
                    moe_out: moe_out.clone(),
                    x_out: x.clone(),
                });
            }
        }
        let t5 = Instant::now();
        let mut hidden = vec![0.0f32; h];
        rmsnorm(&x, &self.weights.final_norm, eps, &mut hidden);
        timing.host_ns += t5.elapsed().as_nanos() as u64;
        self.pos = pos + 1;
        self.history.push(token);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hidden)
    }

    /// Vocabulary logits of a final hidden state (one lm_head launch), added to `last_timing`.
    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        // A one-list decode step already ran the lm_head on the chip: read its logits back.
        if let Some(r) = self.dgraph_take_logits() {
            return r;
        }
        let t0 = Instant::now();
        let mut logits = vec![0.0f32; self.weights.config.vocab];
        let ns = match &self.weights.lm_head {
            LmHead::Dense(m) => m.matmul_f32(1, hidden, &mut logits),
            LmHead::Gemv(g) => g.logits_f32(hidden, &mut logits),
        }
        .map_err(|e| format!("lm_head: {e}"))?;
        let dt = t0.elapsed().as_nanos() as u64;
        self.last_timing.lm_head_ns += dt;
        self.last_timing.total_ns += dt;
        self.last_timing.chip_ns += ns;
        self.last_timing.launches += 1;
        Ok(logits)
    }

    /// Prefill `prompt` then decode with `sampler(logits, step)`; `on_token` gets every emitted
    /// token and may return `false` to cancel. Stop ids are consumed, never emitted. The engine
    /// is reset first (one sequence at a time).
    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        self.generate_inner(prompt, params, sampler, false, on_token)
    }

    /// `generate_with`; `argmax_sampler` says the sampler is the plain argmax (a greedy
    /// `sample_next`), so a one-list decode step's own chip argmax can stand in for it.
    fn generate_inner(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        argmax_sampler: bool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        self.dgraph_clear();
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() >= self.kv.capacity() {
            return Err(format!(
                "prompt of {} tokens, context is {}",
                prompt.len(),
                self.kv.capacity()
            ));
        }
        // Prefix reuse: the KV of the longest common prefix with the stored history stays; at
        // least the last prompt token is always stepped so the logits come from this prompt.
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let reused = common.min(prompt.len() - 1);
        if reused > 0 {
            self.rewind(reused);
        } else {
            self.reset();
        }
        let t0 = Instant::now();
        let mut hidden = Vec::new();
        let mut prefill_timing = StepTiming::default();
        let whole =
            self.whole_prompt_prefill(&prompt[reused..], &mut hidden, &mut prefill_timing)?;
        for chunk in prompt[reused + whole..].chunks(self.prefill_chunk.max(1)) {
            hidden = self.step_batch(chunk)?;
            prefill_timing.add(&self.last_timing);
        }
        let prefill_ms = t0.elapsed().as_millis() as u64;
        // The first decode step is the lm_head of the prefill's hidden state: start its timing
        // from zero, so the last prefill step is not counted twice (once here, once in decode).
        self.last_timing = StepTiming::default();
        let t1 = Instant::now();
        let mut decode_timing = StepTiming::default();
        let mut emitted = 0usize;
        let mut finish = FinishReason::Length;
        let mut step = 0usize;
        loop {
            if emitted >= params.max_new_tokens {
                break;
            }
            let next = match self.dgraph_take_greedy(argmax_sampler) {
                // The one-list step already took the argmax on the chip (same rule as sample_argmax).
                Some(t) => {
                    decode_timing.add(&self.last_timing);
                    t
                }
                None => {
                    let logits = self.logits(&hidden)?;
                    decode_timing.add(&self.last_timing);
                    sampler(&logits, step)
                }
            };
            step += 1;
            if params.stop_token_ids.contains(&next) {
                finish = FinishReason::Stop;
                break;
            }
            emitted += 1;
            if !on_token(next) {
                finish = FinishReason::Cancelled;
                break;
            }
            if emitted >= params.max_new_tokens || self.pos >= self.kv.capacity() {
                break;
            }
            hidden = self.decode_step(next)?;
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms: t1.elapsed().as_millis() as u64,
            prompt_tokens: prompt.len(),
            reused_prompt_tokens: reused,
            decode_timing,
            prefill_timing,
        })
    }

    /// `generate_with` using [`sample_next`] (the door's sampling rule).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        let p = params.clone();
        // sample_next is the plain argmax at temperature <= 0.001 (no top-k mask; sample_with_params greedy).
        let argmax = params.temperature <= 0.001;
        self.generate_inner(
            prompt,
            params,
            &mut |logits, step| sample_next(logits, &p, step),
            argmax,
            on_token,
        )
    }
}

/// The library entry for the HTTP door: `&self` generation behind one lock (one sequence at a
/// time, as the door's contract says). `Send + Sync`.
pub struct OmegaEngine {
    inner: Mutex<Qwen3MoeOmegaEngine>,
    max_context: usize,
    pub load_report: LoadReport,
}

impl OmegaEngine {
    /// Loads the checkpoint at `dir` onto Omega with a KV for `max_context` tokens.
    pub fn load(
        dir: &Path,
        max_context: usize,
        progress: &mut dyn FnMut(&str),
    ) -> Result<Self, String> {
        Self::load_with(dir, max_context, WeightFormat::Bf16, progress)
    }

    /// As [`Self::load`] with a chosen [`WeightFormat`].
    pub fn load_with(
        dir: &Path,
        max_context: usize,
        format: WeightFormat,
        progress: &mut dyn FnMut(&str),
    ) -> Result<Self, String> {
        let (engine, load_report) =
            Qwen3MoeOmegaEngine::load_omega_with(dir, max_context, format, progress)?;
        let max_context = engine.max_context_tokens();
        Ok(Self {
            inner: Mutex::new(engine),
            max_context,
            load_report,
        })
    }

    pub fn from_engine(engine: Qwen3MoeOmegaEngine, load_report: LoadReport) -> Self {
        let max_context = engine.max_context_tokens();
        Self {
            inner: Mutex::new(engine),
            max_context,
            load_report,
        }
    }

    pub fn kind(&self) -> &str {
        "omega"
    }

    pub fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| "engine lock poisoned".to_string())?;
        g.generate(prompt, params, on_token)
    }

    /// Same with the caller's sampler (the door passes its `sample_next`).
    pub fn generate_with(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| "engine lock poisoned".to_string())?;
        g.generate_with(prompt, params, sampler, on_token)
    }

    pub fn with_engine<R>(
        &self,
        f: impl FnOnce(&mut Qwen3MoeOmegaEngine) -> R,
    ) -> Result<R, String> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| "engine lock poisoned".to_string())?;
        Ok(f(&mut g))
    }
}

#[cfg(test)]
mod forward_tests {
    use super::test_support::*;
    use super::*;
    use crate::qwen3_moe::parity_stats;

    /// Plain f32 forward over the dequantised weights (rounded to bf16 as uploaded), no engine
    /// code shared: embedding, GQA attention with q/k norms and rope, top-k routing, swiglu
    /// experts, lm_head. Returns the logits at every position.
    fn naive_forward(src: &MemorySource, c: &Qwen3MoeConfig, tokens: &[u32]) -> Vec<Vec<f32>> {
        let r = |v: f32| bf16_to_f32(f32_to_bf16(v));
        let bf = |name: &str| -> Vec<f32> {
            let (_, _, b) = &src.tensors[name];
            b.chunks_exact(2)
                .map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))
                .collect()
        };
        let fp8 = |name: &str| -> Vec<f32> { src.fp8_f32(name).0.into_iter().map(r).collect() };
        let (h, qd, kvd, hd, i) = (
            c.hidden,
            c.q_dim(),
            c.kv_dim(),
            c.head_dim,
            c.moe_intermediate,
        );
        let matvec = |x: &[f32], w: &[f32], out_dim: usize| -> Vec<f32> {
            let x: Vec<f32> = x.iter().map(|&v| r(v)).collect();
            (0..out_dim)
                .map(|o| {
                    x.iter()
                        .zip(&w[o * x.len()..(o + 1) * x.len()])
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect()
        };
        let embed = bf("model.embed_tokens.weight");
        let lm = bf("lm_head.weight");
        let fnorm = bf("model.norm.weight");
        let mut keys = vec![Vec::<Vec<f32>>::new(); c.layers];
        let mut vals = vec![Vec::<Vec<f32>>::new(); c.layers];
        let mut out = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            let mut x = embed[t as usize * h..(t as usize + 1) * h].to_vec();
            for l in 0..c.layers {
                let p = format!("model.layers.{l}");
                let mut xn = vec![0.0; h];
                rmsnorm(
                    &x,
                    &bf(&format!("{p}.input_layernorm.weight")),
                    c.rms_eps,
                    &mut xn,
                );
                let q = matvec(&xn, &fp8(&format!("{p}.self_attn.q_proj.weight")), qd);
                let k = matvec(&xn, &fp8(&format!("{p}.self_attn.k_proj.weight")), kvd);
                let v = matvec(&xn, &fp8(&format!("{p}.self_attn.v_proj.weight")), kvd);
                let mut qn = vec![0.0; qd];
                let mut kn = vec![0.0; kvd];
                rmsnorm_per_head(
                    &q,
                    &bf(&format!("{p}.self_attn.q_norm.weight")),
                    c.rms_eps,
                    c.q_heads,
                    hd,
                    &mut qn,
                );
                rmsnorm_per_head(
                    &k,
                    &bf(&format!("{p}.self_attn.k_norm.weight")),
                    c.rms_eps,
                    c.kv_heads,
                    hd,
                    &mut kn,
                );
                apply_rope(
                    &mut qn,
                    &mut kn,
                    pos,
                    c.q_heads,
                    c.kv_heads,
                    hd,
                    c.rope_theta,
                );
                keys[l].push(kn.clone());
                vals[l].push(v);
                let mut att = vec![0.0; qd];
                scaled_dot_product_attention_single(
                    &qn, &keys[l], &vals[l], c.q_heads, c.kv_heads, hd, &mut att,
                );
                let o = matvec(&att, &fp8(&format!("{p}.self_attn.o_proj.weight")), h);
                for j in 0..h {
                    x[j] += o[j];
                }
                let mut post = vec![0.0; h];
                rmsnorm(
                    &x,
                    &bf(&format!("{p}.post_attention_layernorm.weight")),
                    c.rms_eps,
                    &mut post,
                );
                let router = bf(&format!("{p}.mlp.gate.weight"));
                let logits: Vec<f32> = (0..c.experts)
                    .map(|e| {
                        post.iter()
                            .zip(&router[e * h..(e + 1) * h])
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let plan = MoeBatchPlan::from_logits(&logits, 1, c.experts, c.top_k).unwrap();
                for slot in 0..c.top_k {
                    let e = plan.topk_experts[slot] as usize;
                    let b = format!("{p}.mlp.experts.{e}");
                    let g = matvec(&post, &fp8(&format!("{b}.gate_proj.weight")), i);
                    let u = matvec(&post, &fp8(&format!("{b}.up_proj.weight")), i);
                    let a: Vec<f32> = (0..i)
                        .map(|j| g[j] / (1.0 + (-g[j]).exp()) * u[j])
                        .collect();
                    let d = matvec(&a, &fp8(&format!("{b}.down_proj.weight")), h);
                    for j in 0..h {
                        x[j] += plan.topk_weights[slot] * d[j];
                    }
                }
            }
            let mut hid = vec![0.0; h];
            rmsnorm(&x, &fnorm, c.rms_eps, &mut hid);
            out.push(matvec(&hid, &lm, c.vocab));
        }
        out
    }

    fn engine() -> (Qwen3MoeOmegaEngine, MemorySource, Qwen3MoeConfig) {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader::default();
        let (w, _) = load_qwen3_moe_omega_from(&c, &mut src, &up, &mut |_| {}).unwrap();
        let kv = HostKv::new(&KvGeometry::from(&c), 32);
        (Qwen3MoeOmegaEngine::new(w, Box::new(kv)), src, c)
    }

    fn engine_with(
        format: WeightFormat,
    ) -> (
        Qwen3MoeOmegaEngine,
        MemorySource,
        Qwen3MoeConfig,
        LoadReport,
    ) {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader::default();
        let (w, report) =
            load_qwen3_moe_omega_from_with(&c, &mut src, &up, format, &mut |_| {}).unwrap();
        let kv = HostKv::new(&KvGeometry::from(&c), 32);
        (Qwen3MoeOmegaEngine::new(w, Box::new(kv)), src, c, report)
    }

    fn engine_fakes(
        format: WeightFormat,
        paths: DevicePaths,
        fakes: bool,
    ) -> Result<(Qwen3MoeOmegaEngine, RecordingUploader, Qwen3MoeConfig), String> {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader {
            device_fakes: fakes,
            ..Default::default()
        };
        let (w, _) =
            load_qwen3_moe_omega_from_paths(&c, &mut src, &up, format, paths, &mut |_| {})?;
        let kv = HostKv::new(&KvGeometry::from(&c), 32);
        Ok((Qwen3MoeOmegaEngine::new(w, Box::new(kv)), up, c))
    }

    #[test]
    fn decode_graph_switch_refuses_a_load_it_cannot_run_instead_of_falling_back() {
        // A host-KV, host-routed load: with the switch on, preparing and generating must both
        // fail with the plain reason; with it off, the same engine generates as before.
        let (mut e, _, c) = engine();
        assert!(!e.decode_graph_on() || dgraph::requested());
        e.decode_graph = true;
        let err = e.prepare_decode_graph().unwrap_err();
        assert!(err.contains("OWN30B_DECODE_GRAPH=1"), "{err}");
        let p = GenerateParams::greedy(3, Vec::new());
        let err = e.generate(&[1, 2, 3], &p, &mut |_| true).unwrap_err();
        assert!(err.contains("OWN30B_DECODE_GRAPH=1"), "{err}");
        e.decode_graph = false;
        let mut got = Vec::new();
        e.generate(&[1, 2, 3], &p, &mut |t| {
            got.push(t);
            true
        })
        .unwrap();
        assert_eq!(got.len(), 3);
        let _ = c;
    }

    #[test]
    fn device_router_and_lm_head_match_the_host_paths_and_add_one_launch_per_layer() {
        let tokens = [5u32, 17, 42, 3, 9, 60];
        for format in [WeightFormat::Fp8Experts, WeightFormat::Fp8All] {
            let (mut host, _, c) = engine_fakes(
                format,
                DevicePaths {
                    router: PathChoice::Host,
                    lm_head: PathChoice::Host,
                },
                true,
            )
            .unwrap();
            let (mut dev, up, _) = engine_fakes(format, DevicePaths::default(), true).unwrap();
            assert!(
                host.weights().device_router.is_none()
                    && matches!(host.weights().lm_head, LmHead::Dense(_))
            );
            assert!(
                dev.weights().device_router.is_some()
                    && matches!(dev.weights().lm_head, LmHead::Gemv(_))
            );
            for &t in &tokens {
                let hh = host.step(t).unwrap();
                let hd = dev.step(t).unwrap();
                // Same maths in the same order: the hidden state is bit-identical.
                assert_eq!(
                    hh, hd,
                    "{format:?}: hidden state differs between host and device routing"
                );
                let (lh, ld) = (host.logits(&hh).unwrap(), dev.logits(&hd).unwrap());
                let s = crate::qwen3_moe::parity_stats(&ld, &lh);
                assert!(s.cosine > 0.99999 && s.non_finite == 0, "{format:?}: {s:?}");
                assert_eq!(
                    crate::tensor::sample_argmax(&ld).0,
                    crate::tensor::sample_argmax(&lh).0
                );
            }
            let attn_launches = 3;
            // Device routing: attention 3 + router 1 + gate|up 1 + down 1 per layer, plus the lm_head.
            assert_eq!(
                dev.last_timing.launches as usize,
                c.layers * (attn_launches + 1 + 2) + 1
            );
            assert_eq!(
                host.last_timing.launches as usize,
                c.layers * (attn_launches + 2) + 1
            );
            let d = up.dev.lock().unwrap();
            assert_eq!(d.routes, tokens.len() * c.layers);
            assert_eq!(d.grouped_dev_calls, tokens.len() * c.layers);
            assert_eq!(d.gemv_calls, tokens.len());
        }
    }

    #[test]
    fn device_paths_are_refused_when_asked_for_and_missing_and_fall_back_on_auto() {
        let dev = DevicePaths {
            router: PathChoice::Device,
            lm_head: PathChoice::Auto,
        };
        let e = engine_fakes(WeightFormat::Fp8All, dev, false)
            .err()
            .expect("no router offered");
        assert!(e.contains("device router"), "{e}");
        let e = engine_fakes(WeightFormat::Bf16, dev, true)
            .err()
            .expect("router needs FP8 experts");
        assert!(e.contains("FP8 experts"), "{e}");
        let head = DevicePaths {
            router: PathChoice::Auto,
            lm_head: PathChoice::Device,
        };
        let e = engine_fakes(WeightFormat::Fp8All, head, false)
            .err()
            .expect("no gemv offered");
        assert!(e.contains("lm_head GEMV"), "{e}");
        // Auto without an offer is the host/dense path, with FP8 or bf16 experts alike.
        for format in [WeightFormat::Bf16, WeightFormat::Fp8All] {
            let (eng, _, _) = engine_fakes(format, DevicePaths::default(), false).unwrap();
            assert!(
                eng.weights().device_router.is_none()
                    && matches!(eng.weights().lm_head, LmHead::Dense(_))
            );
            assert!(!eng.device_routing);
        }
        // Auto with bf16 experts and an offering uploader: no router (it needs FP8 experts), but the GEMV.
        let (eng, _, _) = engine_fakes(WeightFormat::Bf16, DevicePaths::default(), true).unwrap();
        assert!(
            eng.weights().device_router.is_none()
                && matches!(eng.weights().lm_head, LmHead::Gemv(_))
        );
    }

    #[test]
    fn device_router_trace_keeps_the_host_post_as_oracle_input_and_the_mutant_forces_host_routing()
    {
        let (mut eng, up, c) =
            engine_fakes(WeightFormat::Fp8All, DevicePaths::default(), true).unwrap();
        eng.trace_layers = vec![0, 1];
        for &t in &[7u32, 1, 33] {
            eng.step(t).unwrap();
        }
        assert_eq!(eng.traces.len(), 6);
        for t in &eng.traces {
            assert_eq!(t.post_device.len(), c.hidden);
            assert_eq!(
                t.post, t.post_device,
                "fake router computes the same RMSNorm as the host"
            );
            assert_eq!(t.experts.len(), c.top_k);
        }
        let routes_before = up.dev.lock().unwrap().routes;
        // The host routing mutant swaps ids on the host: it must take the host path (no router call).
        eng.traces.clear();
        eng.reset();
        eng.routing_mutant = true;
        eng.step(7).unwrap();
        assert_eq!(
            up.dev.lock().unwrap().routes,
            routes_before,
            "mutant steps are host-routed"
        );
        assert_eq!(eng.last_timing.launches as usize, c.layers * (3 + 2));
        eng.routing_mutant = false;
        eng.device_routing = false;
        eng.reset();
        eng.step(7).unwrap();
        assert_eq!(
            up.dev.lock().unwrap().routes,
            routes_before,
            "device_routing=false is host routing"
        );
    }

    fn engine_gemm(
        gemm: bool,
        device: bool,
    ) -> (Qwen3MoeOmegaEngine, RecordingUploader, Qwen3MoeConfig) {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader {
            device_fakes: device,
            gemm_fakes: gemm,
            ..Default::default()
        };
        let paths = if device {
            DevicePaths::default()
        } else {
            DevicePaths {
                router: PathChoice::Host,
                lm_head: PathChoice::Host,
            }
        };
        let (w, _) = load_qwen3_moe_omega_from_paths(
            &c,
            &mut src,
            &up,
            WeightFormat::Fp8All,
            paths,
            &mut |_| {},
        )
        .unwrap();
        let kv = HostKv::new(&KvGeometry::from(&c), 32);
        (Qwen3MoeOmegaEngine::new(w, Box::new(kv)), up, c)
    }

    #[test]
    fn gemm_prefill_matches_the_gemv_and_per_token_paths_with_one_launch_per_projection() {
        let tokens = [5u32, 17, 42, 3, 9, 60, 7, 1, 33, 12, 25];
        let (mut per_token, _, c) = engine_gemm(false, false);
        let mut want = Vec::new();
        for &t in &tokens {
            want = per_token.step(t).unwrap();
        }
        // The GEMV batch path (no GEMM offered), chunks of 4.
        let (mut gemv, _, _) = engine_gemm(false, false);
        assert!(!gemv.prefill_gemm);
        let mut got = Vec::new();
        for chunk in tokens.chunks(4) {
            got = gemv.step_batch(chunk).unwrap();
        }
        assert_eq!(got, want, "GEMV batch differs from per-token");
        // The GEMM batch path, host routing and device routing, several chunk sizes.
        for device in [false, true] {
            for chunk_len in [11usize, 4, 3] {
                let (mut eng, up, _) = engine_gemm(true, device);
                assert!(eng.prefill_gemm, "the offered GEMM is used by default");
                eng.prefill_device_routing = device;
                let mut hidden = Vec::new();
                for chunk in tokens.chunks(chunk_len) {
                    hidden = eng.step_batch(chunk).unwrap();
                    // q|k|v, o, gate|up, down once per layer, attention per token, and one router launch per token when device-routed.
                    let m = chunk.len();
                    assert_eq!(
                        eng.last_timing.launches as usize,
                        c.layers * (4 + m + if device { m } else { 0 })
                    );
                }
                assert_eq!(hidden, want, "GEMM batch (device routing {device}, chunk {chunk_len}) differs from per-token");
                assert_eq!(eng.position(), tokens.len());
                let d = up.dev.lock().unwrap();
                assert_eq!(
                    d.gemm_calls,
                    tokens.chunks(chunk_len).count() * c.layers * 4
                );
                assert_eq!(d.gemm_rows, tokens.len() * c.layers * (2 + 2 * c.top_k));
                assert_eq!(d.routes, if device { tokens.len() * c.layers } else { 0 });
            }
        }
    }

    #[test]
    fn gemm_prefill_routing_mutant_changes_the_result_and_forces_host_routing() {
        let tokens = [5u32, 17, 42, 3, 9];
        let (mut clean, _, _) = engine_gemm(true, true);
        let want = clean.step_batch(&tokens).unwrap();
        let (mut bad, up, _) = engine_gemm(true, true);
        bad.prefill_device_routing = true;
        bad.routing_mutant = true;
        let got = bad.step_batch(&tokens).unwrap();
        assert_ne!(got, want, "swapped experts must change the hidden state");
        assert_eq!(
            up.dev.lock().unwrap().routes,
            0,
            "mutant steps are host-routed"
        );
    }

    #[test]
    fn fp8_paths_match_the_naive_reference_with_fewer_launches() {
        let tokens = [5u32, 17, 42, 3, 9, 60];
        for format in [WeightFormat::Fp8Experts, WeightFormat::Fp8All] {
            let (mut eng, src, c, report) = engine_with(format);
            assert_eq!(report.format, format);
            let want = naive_forward(&src, &c, &tokens);
            for (pos, &t) in tokens.iter().enumerate() {
                let hid = eng.step(t).unwrap();
                let got = eng.logits(&hid).unwrap();
                let s = parity_stats(&got, &want[pos]);
                assert!(
                    s.cosine > 0.9999 && s.non_finite == 0,
                    "{format:?} pos {pos}: {s:?}"
                );
                assert_eq!(
                    crate::tensor::sample_argmax(&got).0,
                    crate::tensor::sample_argmax(&want[pos]).0,
                    "{format:?} pos {pos}"
                );
            }
            // Experts: 2 launches per layer instead of 2 * top_k; attention still 3.
            assert_eq!(
                eng.last_timing.launches as usize,
                c.layers * (3 + 2) + 1,
                "{format:?}"
            );
            // Resident bytes: FP8 experts are 1 byte + f32 block scales, attention per format.
            let (h, i, e) = (c.hidden, c.moe_intermediate, c.experts);
            let experts = e * (2 * i * h + h * i)
                + e * 4 * (2 * (i / 128) * (h / 128) + (h / 128) * (i / 128));
            let attn = if format == WeightFormat::Fp8All {
                c.qkv_n() * h
                    + 4 * (c.qkv_n() / 128) * (h / 128)
                    + h * c.q_dim()
                    + 4 * (h / 128) * (c.q_dim() / 128)
            } else {
                2 * (h * c.qkv_n() + c.q_dim() * h)
            };
            assert_eq!(
                report.resident_bytes,
                c.layers * (experts + attn) + 2 * h * c.vocab,
                "{format:?}"
            );
        }
    }

    #[test]
    fn prefix_reuse_prefills_only_the_new_tokens_and_matches_a_cold_run() {
        let (mut cold, _, c, _) = engine_with(WeightFormat::Fp8All);
        let (mut warm, _, _, _) = engine_with(WeightFormat::Fp8All);
        let p = GenerateParams::greedy(4, vec![]);
        let turn1 = [5u32, 17, 42, 3, 9, 60];
        let mut reply1 = Vec::new();
        let o1 = warm
            .generate(&turn1, &p, &mut |t| {
                reply1.push(t);
                true
            })
            .unwrap();
        assert_eq!(o1.reused_prompt_tokens, 0);
        assert_eq!(
            warm.history().len(),
            turn1.len() + 3,
            "prompt plus every stepped reply token"
        );
        // Turn 2 resends turn 1, the reply, and new user tokens (a chat app resending the thread).
        let mut turn2 = turn1.to_vec();
        turn2.extend_from_slice(&reply1);
        turn2.extend_from_slice(&[11, 23, 8]);
        let mut warm_out = Vec::new();
        let o2 = warm
            .generate(&turn2, &p, &mut |t| {
                warm_out.push(t);
                true
            })
            .unwrap();
        assert_eq!(
            o2.reused_prompt_tokens,
            turn1.len() + 3,
            "prompt + stepped reply tokens reused"
        );
        let mut cold_out = Vec::new();
        let o3 = cold
            .generate(&turn2, &p, &mut |t| {
                cold_out.push(t);
                true
            })
            .unwrap();
        assert_eq!(o3.reused_prompt_tokens, 0);
        assert_eq!(
            warm_out, cold_out,
            "warm (prefix reused) and cold runs must agree token for token"
        );
        assert_eq!(warm.position(), cold.position());
        // A prompt equal to the whole history still steps its last token (fresh logits).
        let hist = warm.history().to_vec();
        let o4 = warm.generate(&hist, &p, &mut |_| true).unwrap();
        assert_eq!(o4.reused_prompt_tokens, hist.len() - 1);
        // An unrelated prompt starts cold.
        let o5 = warm.generate(&[1, 2, 3], &p, &mut |_| true).unwrap();
        assert_eq!(o5.reused_prompt_tokens, 0);
        assert!(c.layers > 0);
    }

    #[test]
    fn batched_prefill_matches_token_at_a_time_bit_for_bit_with_fewer_launches() {
        let argmax = |lg: &[f32]| {
            lg.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32
        };
        let (mut one, _, c, _) = engine_with(WeightFormat::Fp8All);
        let (mut batched, _, _, _) = engine_with(WeightFormat::Fp8All);
        assert!(batched.batched_prefill_possible());
        one.prefill_chunk = 1;
        batched.prefill_chunk = PREFILL_CHUNK;
        // 12 tokens: one full chunk of 8 and a tail of 4.
        let prompt = [5u32, 17, 42, 3, 9, 60, 11, 23, 8, 2, 7, 19];
        let p = GenerateParams::greedy(3, vec![]);
        let (mut l1, mut l2, mut t1, mut t2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let o1 = one
            .generate_with(
                &prompt,
                &p,
                &mut |lg, _| {
                    l1.push(lg.to_vec());
                    argmax(lg)
                },
                &mut |t| {
                    t1.push(t);
                    true
                },
            )
            .unwrap();
        let o2 = batched
            .generate_with(
                &prompt,
                &p,
                &mut |lg, _| {
                    l2.push(lg.to_vec());
                    argmax(lg)
                },
                &mut |t| {
                    t2.push(t);
                    true
                },
            )
            .unwrap();
        assert_eq!(t1, t2, "greedy tokens after batched prefill");
        assert_eq!(
            l1, l2,
            "logits after batched prefill must equal token-at-a-time bit for bit"
        );
        assert_eq!(one.history(), batched.history());
        // Launches: 5 per layer per token at chunk 1; 4 + tokens per layer per chunk when batched.
        assert_eq!(
            o1.prefill_timing.launches as usize,
            prompt.len() * c.layers * 5
        );
        assert_eq!(
            o2.prefill_timing.launches as usize,
            c.layers * ((4 + 8) + (4 + 4))
        );
        // A bf16 engine has no batched path: chunk 8 falls back to token-at-a-time, same launches as chunk 1.
        let (mut bf, _, _, _) = engine_with(WeightFormat::Bf16);
        assert!(!bf.batched_prefill_possible());
        bf.prefill_chunk = PREFILL_CHUNK;
        let o3 = bf.generate(&prompt, &p, &mut |_| true).unwrap();
        assert_eq!(
            o3.prefill_timing.launches as usize,
            prompt.len() * c.layers * (3 + 2 * c.top_k)
        );
    }

    #[test]
    fn fp8_experts_path_agrees_with_the_bf16_path_and_catches_the_routing_mutant() {
        let tokens = [7u32, 1, 33, 12];
        let (mut a, _, _, _) = engine_with(WeightFormat::Bf16);
        let (mut b, _, c, _) = engine_with(WeightFormat::Fp8Experts);
        a.trace_layers = vec![0];
        b.trace_layers = vec![0];
        for &t in &tokens {
            let ha = a.step(t).unwrap();
            let la = a.logits(&ha).unwrap();
            let hb = b.step(t).unwrap();
            let lb = b.logits(&hb).unwrap();
            let s = parity_stats(&lb, &la);
            assert!(s.cosine > 0.9999, "{s:?}");
        }
        for (ta, tb) in a.traces.iter().zip(&b.traces) {
            assert_eq!(ta.experts, tb.experts, "same routing at pos {}", ta.pos);
            let s = parity_stats(&tb.moe_out, &ta.moe_out);
            assert!(s.cosine > 0.9999, "moe_out pos {}: {s:?}", ta.pos);
        }
        let mut m = Qwen3MoeOmegaEngine::new(
            engine_with(WeightFormat::Fp8Experts).0.weights,
            Box::new(HostKv::new(&KvGeometry::from(&c), 32)),
        );
        m.trace_layers = vec![0];
        m.routing_mutant = true;
        for &t in &tokens {
            m.step(t).unwrap();
        }
        let worst = m
            .traces
            .iter()
            .zip(&b.traces)
            .map(|(x, y)| parity_stats(&x.moe_out, &y.moe_out).cosine)
            .fold(1.0, f64::min);
        assert!(
            worst < 0.99,
            "routing mutant must break the FP8 path too: {worst}"
        );
    }

    #[test]
    fn engine_forward_matches_a_naive_reference_at_every_position() {
        let (mut eng, src, c) = engine();
        let tokens = [5u32, 17, 42, 3, 9, 60];
        let want = naive_forward(&src, &c, &tokens);
        for (pos, &t) in tokens.iter().enumerate() {
            let hid = eng.step(t).unwrap();
            let got = eng.logits(&hid).unwrap();
            let s = parity_stats(&got, &want[pos]);
            assert!(s.cosine > 0.9999 && s.non_finite == 0, "pos {pos}: {s:?}");
            assert_eq!(
                crate::tensor::sample_argmax(&got).0,
                crate::tensor::sample_argmax(&want[pos]).0,
                "pos {pos}"
            );
        }
        assert_eq!(eng.position(), tokens.len());
        assert_eq!(
            eng.last_timing.launches as usize,
            c.layers * (3 + c.top_k + c.top_k) + 1
        );
    }

    #[test]
    fn routing_mutant_breaks_the_layer_and_the_trace_records_it() {
        let (mut eng, _, _) = engine();
        eng.trace_layers = vec![0];
        eng.step(5).unwrap();
        let good = eng.traces[0].clone();
        eng.reset();
        eng.routing_mutant = true;
        eng.step(5).unwrap();
        let bad = eng.traces[0].clone();
        assert_eq!(good.layer, 0);
        assert_eq!(
            good.post, bad.post,
            "attention is untouched by the routing mutant"
        );
        assert_ne!(good.experts, bad.experts);
        let s = parity_stats(&bad.moe_out, &good.moe_out);
        assert!(s.cosine < 0.99, "mutant must be caught: {s:?}");
    }

    #[test]
    fn generate_stops_on_stop_ids_caps_length_and_cancels() {
        let (mut eng, src, c) = engine();
        let prompt = [5u32, 17, 42];
        let want = naive_forward(&src, &c, &prompt);
        let first = crate::tensor::sample_argmax(&want[2]).0;
        // Length cap.
        let mut got = Vec::new();
        let o = eng
            .generate(&prompt, &GenerateParams::greedy(3, vec![]), &mut |t| {
                got.push(t);
                true
            })
            .unwrap();
        assert_eq!(o.finish, FinishReason::Length);
        assert_eq!(o.emitted_tokens, 3);
        assert_eq!(got[0], first);
        assert_eq!(o.prompt_tokens, 3);
        assert_eq!(
            eng.position(),
            3 + 2,
            "the last emitted token is not stepped"
        );
        // Stop id: the first greedy token is a stop id, nothing emitted.
        let o = eng
            .generate(
                &prompt,
                &GenerateParams::greedy(3, vec![first]),
                &mut |_| true,
            )
            .unwrap();
        assert_eq!((o.finish, o.emitted_tokens), (FinishReason::Stop, 0));
        // Cancel after the first token.
        let o = eng
            .generate(&prompt, &GenerateParams::greedy(3, vec![]), &mut |_| false)
            .unwrap();
        assert_eq!((o.finish, o.emitted_tokens), (FinishReason::Cancelled, 1));
        // The locked wrapper behaves the same and is Send + Sync.
        fn assert_sync<T: Send + Sync>(_: &T) {}
        let report = LoadReport {
            resident_bytes: 0,
            format: WeightFormat::Bf16,
            host_bytes: 0,
            peak_rss_bytes: None,
            peak_rss_before_bytes: None,
            seconds: 0.0,
            window_api: false,
        };
        let shared = OmegaEngine::from_engine(eng, report);
        assert_sync(&shared);
        let o = shared
            .generate(&prompt, &GenerateParams::greedy(2, vec![]), &mut |_| true)
            .unwrap();
        assert_eq!(o.emitted_tokens, 2);
        assert_eq!(shared.kind(), "omega");
    }

    #[test]
    fn seeded_sampling_replays_and_greedy_is_argmax() {
        let logits: Vec<f32> = (0..50).map(|i| ((i * 7919) % 50) as f32 / 10.0).collect();
        let p = GenerateParams {
            max_new_tokens: 1,
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            seed: 42,
            stop_token_ids: vec![],
        };
        assert_eq!(sample_next(&logits, &p, 3), sample_next(&logits, &p, 3));
        let g = GenerateParams::greedy(1, vec![]);
        assert_eq!(
            sample_next(&logits, &g, 0),
            crate::tensor::sample_argmax(&logits).0
        );
        let mut seen = std::collections::HashSet::new();
        for step in 0..64 {
            seen.insert(sample_next(&logits, &p, step));
        }
        assert!(seen.len() > 1 && seen.len() <= 20, "{}", seen.len());
    }
}

// ===========================================================================
// Parity harness support: host replays of a traced layer.
// ===========================================================================

use crate::qwen3_coder::{bf16_words, CoderShardCache, Qwen3CoderAttention};
use crate::qwen3_moe::{
    parity_stats, reference_moe_forward, reference_router_logits, ParityStats, Qwen3MoeLayer,
};

/// One layer's attention weights decoded exactly (FP8 x bf16 scale) to f32 row-major, for the
/// host replay. About 75 MB per layer.
pub fn load_attention_layer_f32(
    dir: &Path,
    config: &Qwen3MoeConfig,
    layer: usize,
) -> Result<Qwen3CoderAttention, String> {
    let mut cache = CoderShardCache::open(dir).map_err(|e| e.to_string())?;
    let (h, qd, kvd, hd) = (
        config.hidden,
        config.q_dim(),
        config.kv_dim(),
        config.head_dim,
    );
    let p = format!("model.layers.{layer}");
    let lut = fp8_table();
    let mut fp8_f32 = |name: &str, rows: usize, cols: usize| -> Result<Vec<f32>, String> {
        let w = cache
            .read_raw(name, "F8_E4M3", &[rows, cols], 1)
            .map_err(|e| e.to_string())?;
        let s = cache
            .read_raw(
                &format!("{name}_scale_inv"),
                "BF16",
                &[rows / FP8_BLOCK, cols / FP8_BLOCK],
                2,
            )
            .map_err(|e| e.to_string())?;
        let s = bf16_words(&s);
        let kb = cols / FP8_BLOCK;
        Ok((0..rows * cols)
            .map(|idx| {
                lut[w[idx] as usize]
                    * bf16_to_f32(s[(idx / cols / FP8_BLOCK) * kb + (idx % cols) / FP8_BLOCK])
            })
            .collect())
    };
    let q_proj = fp8_f32(&format!("{p}.self_attn.q_proj.weight"), qd, h)?;
    let k_proj = fp8_f32(&format!("{p}.self_attn.k_proj.weight"), kvd, h)?;
    let v_proj = fp8_f32(&format!("{p}.self_attn.v_proj.weight"), kvd, h)?;
    let o_proj = fp8_f32(&format!("{p}.self_attn.o_proj.weight"), h, qd)?;
    let mut bf16_f32 = |name: &str, n: usize| -> Result<Vec<f32>, String> {
        let b = cache
            .read_raw(name, "BF16", &[n], 2)
            .map_err(|e| e.to_string())?;
        Ok(bf16_words(&b).iter().map(|&x| bf16_to_f32(x)).collect())
    };
    Ok(Qwen3CoderAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: bf16_f32(&format!("{p}.self_attn.q_norm.weight"), hd)?,
        k_norm: bf16_f32(&format!("{p}.self_attn.k_norm.weight"), hd)?,
        input_norm: bf16_f32(&format!("{p}.input_layernorm.weight"), h)?,
        post_norm: bf16_f32(&format!("{p}.post_attention_layernorm.weight"), h)?,
    })
}

fn matvec_rows(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
    let k = x.len();
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            w[o * k..(o + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Host replay of one layer's attention over consecutive traces (positions 0..n of one
/// sequence) with exact f32 weights and an f32 KV: returns the `attn_proj` the host computes
/// for each trace, to compare with the chip's `trace.attn_proj`.
pub fn replay_attention(
    config: &Qwen3MoeConfig,
    attn: &Qwen3CoderAttention,
    traces: &[LayerTrace],
) -> Vec<Vec<f32>> {
    let (h, qd, kvd, hd, eps) = (
        config.hidden,
        config.q_dim(),
        config.kv_dim(),
        config.head_dim,
        config.rms_eps,
    );
    let mut keys: Vec<Vec<f32>> = Vec::new();
    let mut vals: Vec<Vec<f32>> = Vec::new();
    let mut out = Vec::new();
    for t in traces {
        let mut xn = vec![0.0; h];
        rmsnorm(&t.x_in, &attn.input_norm, eps, &mut xn);
        let q = matvec_rows(&xn, &attn.q_proj, qd);
        let k = matvec_rows(&xn, &attn.k_proj, kvd);
        let v = matvec_rows(&xn, &attn.v_proj, kvd);
        let mut qn = vec![0.0; qd];
        let mut kn = vec![0.0; kvd];
        rmsnorm_per_head(&q, &attn.q_norm, eps, config.q_heads, hd, &mut qn);
        rmsnorm_per_head(&k, &attn.k_norm, eps, config.kv_heads, hd, &mut kn);
        apply_rope(
            &mut qn,
            &mut kn,
            t.pos,
            config.q_heads,
            config.kv_heads,
            hd,
            config.rope_theta,
        );
        keys.push(kn);
        vals.push(v);
        let mut att = vec![0.0; qd];
        scaled_dot_product_attention_single(
            &qn,
            &keys,
            &vals,
            config.q_heads,
            config.kv_heads,
            hd,
            &mut att,
        );
        out.push(matvec_rows(&att, &attn.o_proj, h));
    }
    out
}

/// Host oracle of one traced MoE: routes `trace.post` with the capsule's router, runs the exact
/// FP8 oracle, and reports the parity of the chip's `moe_out` plus whether the chip's expert
/// choice equals the oracle's.
pub struct MoeParity {
    pub stats: ParityStats,
    pub oracle_experts: Vec<u32>,
    pub routing_agrees: bool,
    /// Largest |chip weight - oracle weight| over the chip's experts (an expert the oracle did
    /// not pick counts as its full weight). Cosine of `moe_out` is blind to a common scale, so the
    /// weights are compared directly.
    pub max_weight_err: f32,
}

pub fn replay_moe(layer: &Qwen3MoeLayer, trace: &LayerTrace, top_k: usize) -> MoeParity {
    let logits = reference_router_logits(layer, &trace.post, 1);
    let plan =
        MoeBatchPlan::from_logits(&logits, 1, layer.router_bf16().len() / layer.hidden, top_k)
            .expect("plan");
    let want = reference_moe_forward(layer, &trace.post, &plan);
    let mut chip_sorted = trace.experts.clone();
    chip_sorted.sort_unstable();
    let mut oracle_sorted = plan.topk_experts.clone();
    oracle_sorted.sort_unstable();
    let max_weight_err = trace
        .experts
        .iter()
        .zip(&trace.weights)
        .map(|(id, w)| {
            let want = plan
                .topk_experts
                .iter()
                .position(|e| e == id)
                .map_or(0.0, |p| plan.topk_weights[p]);
            (w - want).abs()
        })
        .fold(0.0f32, f32::max);
    MoeParity {
        stats: parity_stats(&trace.moe_out, &want),
        routing_agrees: chip_sorted == oracle_sorted,
        oracle_experts: plan.topk_experts,
        max_weight_err,
    }
}

/// SwiGLU of one expert's gate|up row: `act[j] = silu(gu[j]) * gu[i + j]`.
fn swiglu(gu: &[f32], i: usize, act: &mut [f32]) {
    for j in 0..i {
        let g = gu[j];
        act[j] = g / (1.0 + (-g).exp()) * gu[i + j];
    }
}

/// True when every layer's FP8 projections (q|k|v, o, gate|up, down) carry the prefill GEMM.
fn gemm_everywhere(weights: &Qwen3MoeOmegaWeights) -> bool {
    !weights.layers.is_empty()
        && weights.layers.iter().all(|l| match (&l.attn, &l.experts) {
            (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) => {
                qkv.has_gemm() && o.has_gemm() && gate_up.has_gemm() && down.has_gemm()
            }
            _ => false,
        })
}

/// Router logits for every row of `post` (`m x h`): `logits[r][ex] = dot(router[ex], post[r])`,
/// each dot summed in index order (so the value does not depend on the thread count).
fn router_logits(router: &[f32], post: &[f32], h: usize, e: usize, logits: &mut [f32]) {
    logits
        .par_chunks_mut(e)
        .zip(post.par_chunks(h))
        .for_each(|(lr, pr)| {
            for (ex, l) in lr.iter_mut().enumerate() {
                *l = router[ex * h..(ex + 1) * h]
                    .iter()
                    .zip(pr)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        });
}

/// SwiGLU of every gate|up row (`gu`: rows x 2i) into `act` (rows x i).
fn swiglu_rows(gu: &[f32], act: &mut [f32], i: usize) {
    act.par_chunks_mut(i)
        .zip(gu.par_chunks(2 * i))
        .for_each(|(a, g)| swiglu(g, i, a));
}

/// MoE combine and residual add per token: `x[r] += sum_slot wts[r*top_k+slot] * down[inv[..]]`,
/// accumulated in slot order from zero (the order the per-token path uses).
fn combine_rows(
    x: &mut [f32],
    down_sorted: &[f32],
    wts: &[f32],
    inv: &[u32],
    h: usize,
    top_k: usize,
) {
    x.par_chunks_mut(h).enumerate().for_each(|(r, xr)| {
        let mut acc = vec![0.0f32; h];
        for slot in 0..top_k {
            let gi = r * top_k + slot;
            let w = wts[gi];
            let row = inv[gi] as usize;
            for (a, d) in acc.iter_mut().zip(&down_sorted[row * h..(row + 1) * h]) {
                *a += w * d;
            }
        }
        for (xi, a) in xr.iter_mut().zip(&acc) {
            *xi += a;
        }
    });
}

/// Host-only test engine (sc#505, feature `host-test-engine`): a real [`OmegaEngine`] over the
/// tiny in-memory Qwen3-MoE checkpoint, host matmuls and a [`HostPoolKv`] (bf16 pool, the
/// `HostPagedPoolV1` bytes), so the persistent KV path of the HTTP door can be tested end to end
/// without a chip. Never in a server build.
#[cfg(feature = "host-test-engine")]
pub mod host_test_engine {
    use super::test_support::{tiny_config, MemorySource, RecordingUploader};
    use super::*;

    /// The tiny checkpoint's config (2 layers, 1 KV head of 128, 4 experts, top 2, vocab 64).
    pub fn config() -> Qwen3MoeConfig {
        tiny_config()
    }

    /// A fresh engine (same weights every call, so two calls stand for one server restarted) with
    /// room for `max_context` tokens.
    pub fn engine(format: WeightFormat, max_context: usize) -> Result<OmegaEngine, String> {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader::default();
        let (w, report) = load_qwen3_moe_omega_from_with(&c, &mut src, &up, format, &mut |_| {})?;
        let kv = HostPoolKv::new(&KvGeometry::from(&c), max_context)?;
        Ok(OmegaEngine::from_engine(
            Qwen3MoeOmegaEngine::new(w, Box::new(kv)),
            report,
        ))
    }
}

/// Persistent KV export and import on the engine (sc#505), host only.
#[cfg(test)]
mod kv_persist_tests {
    use super::test_support::*;
    use super::*;

    fn pool_engine(cap: usize) -> Qwen3MoeOmegaEngine {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader::default();
        let (w, _) =
            load_qwen3_moe_omega_from_with(&c, &mut src, &up, WeightFormat::Fp8All, &mut |_| {})
                .unwrap();
        Qwen3MoeOmegaEngine::new(
            w,
            Box::new(HostPoolKv::new(&KvGeometry::from(&c), cap).unwrap()),
        )
    }

    fn argmax(lg: &[f32]) -> u32 {
        lg.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32
    }

    /// Greedy turn: (tokens, every logit vector, reused prompt tokens).
    fn turn(
        e: &mut Qwen3MoeOmegaEngine,
        prompt: &[u32],
        n: usize,
    ) -> (Vec<u32>, Vec<Vec<f32>>, usize) {
        let (mut toks, mut logits) = (Vec::new(), Vec::new());
        let o = e
            .generate_with(
                prompt,
                &GenerateParams::greedy(n, vec![]),
                &mut |lg, _| {
                    logits.push(lg.to_vec());
                    argmax(lg)
                },
                &mut |t| {
                    toks.push(t);
                    true
                },
            )
            .unwrap();
        (toks, logits, o.reused_prompt_tokens)
    }

    fn rows(seed: u64, n: usize) -> Vec<f32> {
        (0..n).map(|i| prand(seed, i)).collect()
    }

    /// The 30B's KV shape (`~/models/qwen3-30b-a3b-instruct-2507-fp8` config.json: 48 layers, 32 q
    /// heads, 4 KV heads, head_dim 128, rope_theta 1e7); the other fields do not reach the KV cache.
    fn qwen3_30b_kv_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            rope_theta: 1e7,
            ..tiny_config()
        }
    }

    // ADR 0004 step 1: the geometry carries exactly the numbers the cache constructors and
    // `PoolBlockGeometry::for_config` read from `Qwen3MoeConfig` before the split, so every cache
    // and every exported byte is the same.
    #[test]
    fn kv_geometry_from_config_carries_the_numbers_for_config_read() {
        for c in [tiny_config(), qwen3_30b_kv_config()] {
            let g = KvGeometry::from(&c);
            assert_eq!(
                g,
                KvGeometry {
                    layers: c.layers,
                    q_heads: c.q_heads,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim,
                    rope_theta: c.rope_theta,
                    block_size: KV_BLOCK_SIZE,
                }
            );
            // What `for_config` built before the split, field by field.
            let before = kvio::PoolBlockGeometry {
                layers: c.layers,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                block_size: KV_BLOCK_SIZE,
            };
            assert_eq!(g.pool_block(), before);
            assert_eq!(kvio::PoolBlockGeometry::for_geometry(&g), before);
            assert_eq!(kvio::PoolBlockGeometry::for_config(&c), before);
        }
    }

    // The three host-buildable caches; OmegaFusedKv needs the chip and real weights (its geometry,
    // len and path are covered by the chip gates and `the_fused_cache_fits_only_the_kernel_shape`).
    #[test]
    fn every_host_buildable_cache_reports_its_geometry_len_and_path_through_the_kv_store_view() {
        let c = tiny_config();
        let g = KvGeometry::from(&c);
        let kvd = c.kv_heads * c.head_dim;
        let caches: Vec<(Box<dyn KvAttention>, &str)> = vec![
            (Box::new(HostKv::new(&g, 32)), "split"),
            (Box::new(HostPoolKv::new(&g, 32).unwrap()), "host-pool"),
            (Box::new(OmegaPagedKv::new(&g, 32).unwrap()), "split"),
        ];
        for (mut kv, path) in caches {
            let store: &mut dyn KvStore = &mut *kv;
            assert_eq!(store.geometry(), g, "{path}");
            assert_eq!(store.path_name(), path);
            assert_eq!(store.capacity(), 32, "{path}");
            assert_eq!(store.len(), 0, "{path}");
            for pos in 0..5 {
                kv.append(0, pos, &rows(pos as u64, kvd), &rows(pos as u64 + 1, kvd))
                    .unwrap();
            }
            // Layer 1 still holds nothing: the smallest per-layer count.
            assert_eq!(kv.len(), 0, "{path}");
            for pos in 0..3 {
                kv.append(1, pos, &rows(pos as u64, kvd), &rows(pos as u64 + 1, kvd))
                    .unwrap();
            }
            assert_eq!(kv.len(), 3, "{path}");
            kv.truncate(2);
            assert_eq!(kv.len(), 2, "{path}");
            // `truncate(0)` is what `reset` was: every layer empty, the next append is at 0.
            kv.truncate(0);
            assert_eq!(kv.len(), 0, "{path}");
            kv.append(0, 0, &rows(9, kvd), &rows(10, kvd)).unwrap();
            assert!(
                kv.append(1, 1, &rows(9, kvd), &rows(10, kvd)).is_err(),
                "{path}"
            );
        }
    }

    #[test]
    fn pool_caches_refuse_a_block_size_their_layout_does_not_use() {
        let g = KvGeometry {
            block_size: KV_BLOCK_SIZE * 2,
            ..KvGeometry::from(&tiny_config())
        };
        assert!(HostPoolKv::new(&g, 32).is_err());
        assert!(OmegaPagedKv::new(&g, 32).is_err());
    }

    #[test]
    fn the_fused_cache_fits_only_the_kernel_shape() {
        use crate::qwen3_moe_omega_dattn::OmegaFusedKv;
        assert!(OmegaFusedKv::fits(
            &KvGeometry::from(&qwen3_30b_kv_config())
        ));
        assert!(!OmegaFusedKv::fits(&KvGeometry::from(&tiny_config())));
        // The 8B's 8 KV heads wait for K2.
        let dense = KvGeometry {
            layers: 36,
            kv_heads: 8,
            ..KvGeometry::from(&qwen3_30b_kv_config())
        };
        assert!(!OmegaFusedKv::fits(&dense));
    }

    /// FNV-1a 64 of the `HostPagedPoolV1` export of positions `0..37` filled with `rows`.
    fn export_digest(c: &Qwen3MoeConfig, kv: &mut dyn KvAttention) -> (usize, u64) {
        let kvd = c.kv_heads * c.head_dim;
        for pos in 0..37 {
            for l in 0..c.layers {
                let s = pos as u64 * 100 + l as u64;
                kv.append(l, pos, &rows(s, kvd), &rows(s + 7, kvd)).unwrap();
            }
        }
        let b = kv.export_blocks(0, 37).unwrap().unwrap();
        let h = b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
            (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3)
        });
        (b.len(), h)
    }

    // ADR 0004 step 1: the payload bytes are those main exported before the split. The digests were
    // printed by this same fill and hash on origin/main 559f0f1e (constructors taking the config).
    #[test]
    fn exported_kv_bytes_match_the_digests_recorded_on_main() {
        for (c, want) in [
            (tiny_config(), (49_152usize, 0xbbf5_a8d4_5a90_9242u64)),
            (qwen3_30b_kv_config(), (4_718_592, 0x60da_5a45_da83_5b1a)),
        ] {
            let g = KvGeometry::from(&c);
            let host = export_digest(&c, &mut HostPoolKv::new(&g, 64).unwrap());
            let paged = export_digest(&c, &mut OmegaPagedKv::new(&g, 64).unwrap());
            assert_eq!(host, want, "host pool, {} layers", c.layers);
            assert_eq!(paged, want, "paged, {} layers", c.layers);
        }
    }

    #[test]
    fn paged_and_host_pool_caches_export_the_same_bytes_and_round_trip() {
        let c = tiny_config();
        let kvd = c.kv_heads * c.head_dim;
        // Stub build: a host pool. Linked native build (AIEN_REQUIRE_NATIVE_LIBS=1, abi-qualify
        // chip-inference-linked): the resident chip pool. The exported bytes must match either way.
        let mut paged = OmegaPagedKv::new(&KvGeometry::from(&c), 64).unwrap();
        let mut host = HostPoolKv::new(&KvGeometry::from(&c), 64).unwrap();
        for pos in 0..37 {
            for l in 0..c.layers {
                let (k, v) = (
                    rows(pos as u64 * 10 + l as u64, kvd),
                    rows(pos as u64 * 10 + l as u64 + 5, kvd),
                );
                paged.append(l, pos, &k, &v).unwrap();
                host.append(l, pos, &k, &v).unwrap();
            }
        }
        let g = kvio::PoolBlockGeometry::for_config(&c);
        for (start, len) in [(0usize, 37usize), (0, 16), (16, 21), (32, 1)] {
            let a = paged.export_blocks(start, len).unwrap().unwrap();
            let b = host.export_blocks(start, len).unwrap().unwrap();
            assert_eq!(a, b, "{start}+{len}");
            assert_eq!(a.len(), g.payload_len(len));
            kvio::check_payload(&g, len, &a).unwrap();
        }
        // Past what is stored: refused.
        assert!(paged.export_blocks(32, 6).unwrap().is_err());
        // Import into a fresh cache in two pieces, export again: identical bytes.
        let whole = host.export_blocks(0, 37).unwrap().unwrap();
        let mut fresh = OmegaPagedKv::new(&KvGeometry::from(&c), 64).unwrap();
        assert!(
            fresh
                .import_blocks(16, 21, &host.export_blocks(16, 21).unwrap().unwrap())
                .unwrap()
                .is_err(),
            "must start where the cache ends"
        );
        fresh
            .import_blocks(0, 16, &host.export_blocks(0, 16).unwrap().unwrap())
            .unwrap()
            .unwrap();
        fresh
            .import_blocks(16, 21, &host.export_blocks(16, 21).unwrap().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(fresh.export_blocks(0, 37).unwrap().unwrap(), whole);
        // The f32 test cache cannot export: None, never a guess.
        assert!(HostKv::new(&KvGeometry::from(&c), 8)
            .export_blocks(0, 1)
            .is_none());
    }

    #[test]
    fn restored_prefix_continues_exactly_like_the_uninterrupted_engine() {
        let turn1 = [
            5u32, 17, 42, 3, 9, 60, 11, 23, 8, 2, 7, 19, 33, 40, 1, 6, 14, 29,
        ];
        // Uninterrupted: turn 1, then turn 2 reusing the in-RAM prefix.
        let mut live = pool_engine(128);
        let (reply1, _, r1) = turn(&mut live, &turn1, 5);
        assert_eq!(r1, 0);
        let history = live.history().to_vec();
        let snapshot = live.export_kv(0, history.len()).unwrap();
        let mut turn2 = turn1.to_vec();
        turn2.extend_from_slice(&reply1);
        turn2.extend_from_slice(&[12, 31, 4]);
        let (want, want_logits, reused_live) = turn(&mut live, &turn2, 6);
        assert_eq!(
            reused_live,
            history.len(),
            "turn 2 reuses every stepped token of turn 1"
        );

        // Restarted: a fresh engine (same weights), the parked prefix restored, then turn 2.
        let mut restarted = pool_engine(128);
        restarted.restore_kv(&history, &snapshot).unwrap();
        assert_eq!(restarted.history(), &history[..]);
        assert_eq!(restarted.position(), history.len());
        assert_eq!(
            restarted.export_kv(0, history.len()).unwrap(),
            snapshot,
            "import then export is byte-identical"
        );
        assert_eq!(restarted.reusable_prefix(&turn2), reused_live);
        let (got, got_logits, reused_restored) = turn(&mut restarted, &turn2, 6);
        assert_eq!(
            reused_restored, reused_live,
            "the restore point is the uninterrupted reuse point"
        );
        assert_eq!(
            got, want,
            "restored continuation equals the uninterrupted one"
        );
        assert_eq!(got_logits, want_logits, "bit for bit");
    }

    #[test]
    fn a_damaged_or_impossible_restore_leaves_a_clean_engine_that_prefills_from_zero() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 11, 23, 8, 2];
        let mut a = pool_engine(64);
        let _ = turn(&mut a, &prompt, 3);
        let history = a.history().to_vec();
        let mut snap = a.export_kv(0, history.len()).unwrap();
        let mut cold = pool_engine(64);
        let (want, _, _) = turn(&mut cold, &prompt, 3);

        let mut b = pool_engine(64);
        let _ = turn(&mut b, &[1, 2, 3, 4], 2);
        // Wrong length, a padding byte set, an empty or oversized token list: all refused.
        assert!(b.restore_kv(&history, &snap[..snap.len() - 2]).is_err());
        assert_eq!(
            (b.position(), b.history().len()),
            (0, 0),
            "reset, not a partial prefix"
        );
        let last = snap.len() - 1;
        snap[last] ^= 1;
        assert!(b.restore_kv(&history, &snap).is_err());
        assert!(b.restore_kv(&[], &[]).is_err());
        assert!(b.restore_kv(&[1u32; 64], &snap).is_err());
        assert_eq!((b.position(), b.history().len()), (0, 0));
        let (got, _, reused) = turn(&mut b, &prompt, 3);
        assert_eq!(reused, 0);
        assert_eq!(got, want, "plain prefill after a refused restore");
        // An engine whose cache cannot export refuses the park.
        let (mut f32_engine, _, _, _) = {
            let c = tiny_config();
            let mut src = MemorySource::for_config(&c);
            let up = RecordingUploader::default();
            let (w, r) = load_qwen3_moe_omega_from_with(
                &c,
                &mut src,
                &up,
                WeightFormat::Fp8All,
                &mut |_| {},
            )
            .unwrap();
            (
                Qwen3MoeOmegaEngine::new(w, Box::new(HostKv::new(&KvGeometry::from(&c), 32))),
                (),
                (),
                r,
            )
        };
        let _ = turn(&mut f32_engine, &prompt, 2);
        assert!(f32_engine.export_kv(0, 4).is_err());
        assert!(f32_engine.export_kv(0, 99).is_err());
    }

    #[test]
    fn kv_switches_report_every_run_setting() {
        let mut e = pool_engine(32);
        let s = e.kv_switches();
        assert_eq!(s.attention_path, "host-pool");
        assert!(!s.whole_prompt && !s.decode_graph && !s.routing_mutant);
        e.prefill_chunk = 3;
        e.routing_mutant = true;
        let t = e.kv_switches();
        assert_eq!(t.prefill_chunk, 3);
        assert!(t.routing_mutant);
        assert_ne!(s, t);
    }
}
