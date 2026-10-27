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

//! Native end-to-end Qwen3-Coder-30B-A3B inference in Rust, no MAX graph, no Python.
//!
//! Loads the official FP8 checkpoint (`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`)
//! through the same validated shard reader as the MoE path, decodes weights to
//! FP32 once at load, and runs the full 48-layer forward on CPU: embedding,
//! per-layer GQA attention with QK norms and RoPE, the native MoE experts via
//! [`crate::qwen3_moe`], final norm, LM head, and sampling. Every GPU kernel in
//! Phase B plugs into this exact structure, so numerics stay comparable.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::moe_plan::MoeBatchPlan;
use crate::qwen3_moe::{
    bf16_to_f32, fp8_table, load_qwen3_fp8_moe_layer, reference_moe_forward,
    reference_router_logits, shard_path, Qwen3MoeError, Qwen3MoeLayer, Shard, FP8_BLOCK,
    QWEN3_A3B_HIDDEN, QWEN3_A3B_LAYERS,
};
use crate::tensor::{apply_rope, rmsnorm, sample_argmax, scaled_dot_product_attention_single};

pub const QWEN3_CODER_LAYERS: usize = QWEN3_A3B_LAYERS;
pub const QWEN3_CODER_HIDDEN: usize = QWEN3_A3B_HIDDEN;
pub const QWEN3_CODER_Q_HEADS: usize = 32;
pub const QWEN3_CODER_KV_HEADS: usize = 4;
pub const QWEN3_CODER_HEAD_DIM: usize = 128;
pub const QWEN3_CODER_Q_DIM: usize = QWEN3_CODER_Q_HEADS * QWEN3_CODER_HEAD_DIM;
pub const QWEN3_CODER_KV_DIM: usize = QWEN3_CODER_KV_HEADS * QWEN3_CODER_HEAD_DIM;
pub const QWEN3_CODER_VOCAB: usize = 151936;
pub const QWEN3_CODER_RMS_EPS: f32 = 1e-6;
pub const QWEN3_CODER_ROPE_THETA: f32 = 10_000_000.0;

#[derive(Debug)]
pub enum Qwen3CoderError {
    Io(String),
    Index(String),
    Contract(String),
}

impl std::fmt::Display for Qwen3CoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(d) => write!(f, "io: {d}"),
            Self::Index(d) => write!(f, "index: {d}"),
            Self::Contract(d) => write!(f, "contract: {d}"),
        }
    }
}

impl std::error::Error for Qwen3CoderError {}

impl From<Qwen3MoeError> for Qwen3CoderError {
    fn from(e: Qwen3MoeError) -> Self {
        Self::Contract(format!("moe layer: {e}"))
    }
}

/// One layer's attention weights decoded to FP32 row-major at load.
#[derive(Debug, Clone)]
pub struct Qwen3CoderAttention {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
}

/// Full Qwen3-Coder-30B-A3B weights. MoE experts stay in their validated
/// capsule form; everything else is FP32 decoded once at load.
pub struct Qwen3CoderWeights {
    pub embed: Vec<f32>,
    pub attns: Vec<Qwen3CoderAttention>,
    pub moes: Vec<Qwen3MoeLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Per-layer KV cache: one (k, v) pair per decoded position.
#[derive(Debug, Default)]
pub struct Qwen3CoderLayerKv {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

#[derive(Debug, Default)]
pub struct Qwen3CoderState {
    pub layers: Vec<Qwen3CoderLayerKv>,
}

impl Qwen3CoderState {
    pub fn new() -> Self {
        Self {
            layers: (0..QWEN3_CODER_LAYERS)
                .map(|_| Qwen3CoderLayerKv::default())
                .collect(),
        }
    }
}

/// Parallel FP32 vector-matrix multiply, f32 accumulation for speed on the
/// wide projections (Q, O, LM head). Narrow norms stay on `tensor::matmul_vec`.
fn matvec_f32_parallel(x: &[f32], w: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(out.len(), out_dim);
    out.par_chunks_exact_mut(1)
        .enumerate()
        .for_each(|(j, slot)| {
            let row = &w[j * in_dim..(j + 1) * in_dim];
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += x[i] * row[i];
            }
            slot[0] = sum;
        });
}

/// Decodes one FP8 block-128 matrix with BF16 inverse scales to FP32 row-major.
fn decode_fp8_block128(weights: &[u8], scales_inv: &[u16], rows: usize, cols: usize) -> Vec<f32> {
    let lut = fp8_table();
    let k_blocks = cols / FP8_BLOCK;
    assert_eq!(cols % FP8_BLOCK, 0, "cols must split into 128-wide blocks");
    assert_eq!(
        scales_inv.len(),
        rows.div_ceil(FP8_BLOCK) * k_blocks,
        "scale grid must match 128x128 blocks"
    );
    let mut out = vec![0.0f32; rows * cols];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, row_out)| {
            let row = &weights[r * cols..(r + 1) * cols];
            for b in 0..k_blocks {
                let scale = bf16_to_f32(scales_inv[(r / FP8_BLOCK) * k_blocks + b]);
                let dst = &mut row_out[b * FP8_BLOCK..(b + 1) * FP8_BLOCK];
                let src = &row[b * FP8_BLOCK..(b + 1) * FP8_BLOCK];
                for (d, &w) in dst.iter_mut().zip(src.iter()) {
                    *d = lut[w as usize] * scale;
                }
            }
        });
    out
}

/// Reinterprets a BF16 payload as u16 words. Panics on odd length or
/// misalignment rather than silently dropping bytes and shifting every word.
pub(crate) fn bf16_words(bytes: &[u8]) -> &[u16] {
    assert_eq!(bytes.len() % 2, 0, "bf16 payload must be even length");
    // SAFETY: u16 has no invalid bit patterns; prefix/suffix are checked empty.
    let (prefix, words, suffix) = unsafe { bytes.align_to::<u16>() };
    assert!(
        prefix.is_empty() && suffix.is_empty(),
        "bf16 payload misaligned"
    );
    words
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bf16_words(bytes).iter().map(|&b| bf16_to_f32(b)).collect()
}

pub(crate) struct CoderShardCache {
    checkpoint_dir: PathBuf,
    weight_map: serde_json::Map<String, serde_json::Value>,
    shards: HashMap<String, Shard>,
}

impl CoderShardCache {
    pub(crate) fn open(checkpoint_dir: &Path) -> Result<Self, Qwen3CoderError> {
        let index_path = checkpoint_dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&index_path)
                .map_err(|e| Qwen3CoderError::Io(format!("{}: {e}", index_path.display())))?,
        )
        .map_err(|e| Qwen3CoderError::Index(format!("{}: {e}", index_path.display())))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .cloned()
            .ok_or_else(|| Qwen3CoderError::Index("missing weight_map object".to_string()))?;
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            weight_map,
            shards: HashMap::new(),
        })
    }

    pub(crate) fn read_raw(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        elem_bytes: usize,
    ) -> Result<Vec<u8>, Qwen3CoderError> {
        let filename = self
            .weight_map
            .get(name)
            .and_then(|v| v.as_str())
            .ok_or_else(|| Qwen3CoderError::Index(format!("no shard for {name}")))?;
        if !self.shards.contains_key(filename) {
            let shard = Shard::open(&shard_path(&self.checkpoint_dir, filename)?)
                .map_err(|e| Qwen3CoderError::Contract(format!("shard open: {e}")))?;
            self.shards.insert(filename.to_string(), shard);
        }
        let expect_elems: usize = shape.iter().product();
        let mut dst = vec![0u8; expect_elems * elem_bytes];
        self.shards[filename]
            .read(name, dtype, shape, &mut dst)
            .map_err(|e| Qwen3CoderError::Contract(format!("shard read: {e}")))?;
        Ok(dst)
    }

    fn read_bf16_f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Qwen3CoderError> {
        Ok(decode_bf16(&self.read_raw(name, "BF16", shape, 2)?))
    }

    fn read_fp8_f32(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, Qwen3CoderError> {
        let scale_name = format!("{name}_scale_inv");
        let weights = self.read_raw(name, "F8_E4M3", &[rows, cols], 1)?;
        let scales = self.read_raw(
            &scale_name,
            "BF16",
            &[rows / FP8_BLOCK, cols / FP8_BLOCK],
            2,
        )?;
        Ok(decode_fp8_block128(
            &weights,
            bf16_words(&scales),
            rows,
            cols,
        ))
    }
}

/// Loads all 48 layers plus embeddings, final norm, and LM head.
pub fn load_qwen3_coder(checkpoint_dir: &Path) -> Result<Qwen3CoderWeights, Qwen3CoderError> {
    let h = QWEN3_CODER_HIDDEN;
    let qd = QWEN3_CODER_Q_DIM;
    let kvd = QWEN3_CODER_KV_DIM;
    let hd = QWEN3_CODER_HEAD_DIM;
    let mut cache = CoderShardCache::open(checkpoint_dir)?;

    let embed = cache.read_bf16_f32("model.embed_tokens.weight", &[QWEN3_CODER_VOCAB, h])?;
    let final_norm = cache.read_bf16_f32("model.norm.weight", &[h])?;
    let lm_head = cache.read_bf16_f32("lm_head.weight", &[QWEN3_CODER_VOCAB, h])?;

    let mut attns = Vec::with_capacity(QWEN3_CODER_LAYERS);
    let mut moes = Vec::with_capacity(QWEN3_CODER_LAYERS);
    for layer in 0..QWEN3_CODER_LAYERS {
        let p = format!("model.layers.{layer}");
        let q_proj = cache.read_fp8_f32(&format!("{p}.self_attn.q_proj.weight"), qd, h)?;
        let k_proj = cache.read_fp8_f32(&format!("{p}.self_attn.k_proj.weight"), kvd, h)?;
        let v_proj = cache.read_fp8_f32(&format!("{p}.self_attn.v_proj.weight"), kvd, h)?;
        let o_proj = cache.read_fp8_f32(&format!("{p}.self_attn.o_proj.weight"), h, qd)?;
        let q_norm = cache.read_bf16_f32(&format!("{p}.self_attn.q_norm.weight"), &[hd])?;
        let k_norm = cache.read_bf16_f32(&format!("{p}.self_attn.k_norm.weight"), &[hd])?;
        let input_norm = cache.read_bf16_f32(&format!("{p}.input_layernorm.weight"), &[h])?;
        let post_norm =
            cache.read_bf16_f32(&format!("{p}.post_attention_layernorm.weight"), &[h])?;
        attns.push(Qwen3CoderAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            input_norm,
            post_norm,
        });
        moes.push(load_qwen3_fp8_moe_layer(checkpoint_dir, layer)?);
    }

    Ok(Qwen3CoderWeights {
        embed,
        attns,
        moes,
        final_norm,
        lm_head,
    })
}

pub(crate) fn rmsnorm_per_head(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    num_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    for h in 0..num_heads {
        rmsnorm(
            &x[h * head_dim..(h + 1) * head_dim],
            weight,
            eps,
            &mut out[h * head_dim..(h + 1) * head_dim],
        );
    }
}

/// Runs one token through all 48 layers on the CPU (bit-exact reference: attention, norms,
/// RoPE and the routed experts). Returns the final hidden state.
pub fn forward_token(
    weights: &Qwen3CoderWeights,
    token_id: u32,
    pos: usize,
    state: &mut Qwen3CoderState,
) -> Vec<f32> {
    let h = QWEN3_CODER_HIDDEN;
    let qd = QWEN3_CODER_Q_DIM;
    let kvd = QWEN3_CODER_KV_DIM;
    let hd = QWEN3_CODER_HEAD_DIM;
    let eps = QWEN3_CODER_RMS_EPS;

    let token_idx = (token_id as usize) % QWEN3_CODER_VOCAB;
    let mut x = weights.embed[token_idx * h..(token_idx + 1) * h].to_vec();

    for (layer_idx, (attn, moe)) in weights.attns.iter().zip(weights.moes.iter()).enumerate() {
        let kv = &mut state.layers[layer_idx];

        let mut x_norm = vec![0.0f32; h];
        rmsnorm(&x, &attn.input_norm, eps, &mut x_norm);

        let mut q = vec![0.0f32; qd];
        let mut k = vec![0.0f32; kvd];
        let mut v = vec![0.0f32; kvd];
        matvec_f32_parallel(&x_norm, &attn.q_proj, &mut q, h, qd);
        matvec_f32_parallel(&x_norm, &attn.k_proj, &mut k, h, kvd);
        matvec_f32_parallel(&x_norm, &attn.v_proj, &mut v, h, kvd);

        let mut qn = vec![0.0f32; qd];
        let mut kn = vec![0.0f32; kvd];
        rmsnorm_per_head(&q, &attn.q_norm, eps, QWEN3_CODER_Q_HEADS, hd, &mut qn);
        rmsnorm_per_head(&k, &attn.k_norm, eps, QWEN3_CODER_KV_HEADS, hd, &mut kn);

        apply_rope(
            &mut qn,
            &mut kn,
            pos,
            QWEN3_CODER_Q_HEADS,
            QWEN3_CODER_KV_HEADS,
            hd,
            QWEN3_CODER_ROPE_THETA,
        );

        kv.keys.push(kn);
        kv.values.push(v);

        let mut attn_out = vec![0.0f32; qd];
        scaled_dot_product_attention_single(
            &qn,
            &kv.keys,
            &kv.values,
            QWEN3_CODER_Q_HEADS,
            QWEN3_CODER_KV_HEADS,
            hd,
            &mut attn_out,
        );

        let mut attn_proj = vec![0.0f32; h];
        matvec_f32_parallel(&attn_out, &attn.o_proj, &mut attn_proj, qd, h);
        for i in 0..h {
            x[i] += attn_proj[i];
        }

        let mut post = vec![0.0f32; h];
        rmsnorm(&x, &attn.post_norm, eps, &mut post);

        let router = reference_router_logits(moe, &post, 1);
        let plan = MoeBatchPlan::qwen3_coder_a3b(&router, 1).expect("routing plan must build");
        let moe_out = reference_moe_forward(moe, &post, &plan);
        for i in 0..h {
            x[i] += moe_out[i];
        }
    }

    let mut x_final = vec![0.0f32; h];
    rmsnorm(&x, &weights.final_norm, eps, &mut x_final);
    x_final
}

/// Projects a hidden state to vocabulary logits.
pub fn compute_logits(weights: &Qwen3CoderWeights, hidden: &[f32]) -> Vec<f32> {
    let h = QWEN3_CODER_HIDDEN;
    let mut logits = vec![0.0f32; QWEN3_CODER_VOCAB];
    matvec_f32_parallel(hidden, &weights.lm_head, &mut logits, h, QWEN3_CODER_VOCAB);
    logits
}

/// Prefills prompt tokens, then greedily decodes up to `max_new` tokens,
/// stopping at any id in `stop_ids`. Returns only the new tokens.
pub fn generate_greedy(
    weights: &Qwen3CoderWeights,
    prompt: &[u32],
    max_new: usize,
    stop_ids: &[u32],
) -> Vec<u32> {
    let mut state = Qwen3CoderState::new();
    let mut pos = 0usize;
    for &tok in prompt {
        forward_token(weights, tok, pos, &mut state);
        pos += 1;
    }
    let mut out = Vec::new();
    let mut prev = *prompt.last().unwrap_or(&0);
    for _ in 0..max_new {
        let hidden = forward_token(weights, prev, pos, &mut state);
        let (next, _) = sample_argmax(&compute_logits(weights, &hidden));
        pos += 1;
        if stop_ids.contains(&next) {
            break;
        }
        out.push(next);
        prev = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_moe::{dequant_matvec, QWEN3_A3B_INTERMEDIATE};

    /// 2-aligned storage so the misalignment tests control the offset exactly.
    fn aligned_bytes(words: &[u16]) -> Vec<u16> {
        words.to_vec()
    }

    #[test]
    fn bf16_words_reads_aligned_payload() {
        let backing = aligned_bytes(&[0x3f80, 0x4000]);
        // SAFETY: u16 storage viewed as bytes; lifetime tied to `backing`.
        let bytes = unsafe { std::slice::from_raw_parts(backing.as_ptr() as *const u8, 4) };
        assert_eq!(bf16_words(bytes), &[0x3f80, 0x4000]);
    }

    #[test]
    #[should_panic(expected = "misaligned")]
    fn bf16_words_rejects_misaligned_payload() {
        let backing = aligned_bytes(&[0, 0, 0]);
        // SAFETY: in-bounds view starting one byte into 2-aligned storage.
        let bytes =
            unsafe { std::slice::from_raw_parts((backing.as_ptr() as *const u8).add(1), 4) };
        let _ = bf16_words(bytes);
    }

    #[test]
    #[should_panic(expected = "even length")]
    fn bf16_words_rejects_odd_length() {
        let backing = aligned_bytes(&[0, 0]);
        // SAFETY: in-bounds 3-byte view of 4-byte storage.
        let bytes = unsafe { std::slice::from_raw_parts(backing.as_ptr() as *const u8, 3) };
        let _ = bf16_words(bytes);
    }

    #[test]
    fn coder_constants_match_official_config() {
        assert_eq!(QWEN3_CODER_LAYERS, 48);
        assert_eq!(QWEN3_CODER_HIDDEN, 2048);
        assert_eq!(QWEN3_CODER_Q_HEADS, 32);
        assert_eq!(QWEN3_CODER_KV_HEADS, 4);
        assert_eq!(QWEN3_CODER_HEAD_DIM, 128);
        assert_eq!(QWEN3_CODER_VOCAB, 151936);
        assert_eq!(QWEN3_A3B_INTERMEDIATE, 768);
    }

    #[test]
    fn fp8_block128_roundtrip_spot_check() {
        let lut = fp8_table();
        assert!((lut[0x38] - 1.0).abs() < 1e-6);
        assert!((lut[0xBC] + 1.5).abs() < 1e-6);
        let w = vec![0x38u8; 256];
        // 2x128 is one 128x128 scale block (row blocks round up).
        let s = vec![crate::qwen3_moe::f32_to_bf16(1.0); 1];
        let m = decode_fp8_block128(&w, &s, 2, 128);
        assert!(m.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        let v = vec![1.0f32; 128];
        let y = dequant_matvec(&w, &s, 2, 128, &v, &lut);
        assert!((y[0] - 128.0).abs() < 1e-3);
    }
}
