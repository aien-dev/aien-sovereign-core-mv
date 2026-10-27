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

//! Pure native Rust tensor operations for transformer forward execution.
//! Provides cache-friendly vector and matrix math, RMSNorm, RoPE, and SwiGLU activations.

use aien_abi_core::RopeParams;
use rayon::prelude::*;

/// Vector-matrix multiplication for PyTorch row-major weights: out = x * W^T
/// where x is [in_dim], W is row-major [out_dim, in_dim], and out is [out_dim].
/// Each out[j] is the inner dot product of vector x with the j-th row of W.
#[inline]
pub fn matmul_vec(x: &[f32], w: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(out.len(), out_dim);

    for j in 0..out_dim {
        let w_row = &w[j * in_dim..(j + 1) * in_dim];
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut sum4 = 0.0f64;
        let mut sum5 = 0.0f64;
        let mut sum6 = 0.0f64;
        let mut sum7 = 0.0f64;

        let mut i = 0;
        while i + 8 <= in_dim {
            sum0 += (x[i] as f64) * (w_row[i] as f64);
            sum1 += (x[i + 1] as f64) * (w_row[i + 1] as f64);
            sum2 += (x[i + 2] as f64) * (w_row[i + 2] as f64);
            sum3 += (x[i + 3] as f64) * (w_row[i + 3] as f64);
            sum4 += (x[i + 4] as f64) * (w_row[i + 4] as f64);
            sum5 += (x[i + 5] as f64) * (w_row[i + 5] as f64);
            sum6 += (x[i + 6] as f64) * (w_row[i + 6] as f64);
            sum7 += (x[i + 7] as f64) * (w_row[i + 7] as f64);
            i += 8;
        }
        let mut sum = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
        while i < in_dim {
            sum += (x[i] as f64) * (w_row[i] as f64);
            i += 1;
        }
        out[j] = sum as f32;
    }
}

/// Batched matrix multiplication for row-major weights: out = X * W^T
/// where X is [batch_size, in_dim], W is row-major [out_dim, in_dim], and out is [batch_size, out_dim].
/// Parallelizes batch rows across available CPU threads with Rayon.
pub fn matmul_batch(
    x: &[f32],
    w: &[f32],
    out: &mut [f32],
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(x.len(), batch_size * in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(out.len(), batch_size * out_dim);

    if batch_size == 1 {
        matmul_vec(x, w, out, in_dim, out_dim);
        return;
    }

    out.par_chunks_exact_mut(out_dim)
        .enumerate()
        .for_each(|(b, out_b)| {
            let x_b = &x[b * in_dim..(b + 1) * in_dim];
            matmul_vec(x_b, w, out_b, in_dim, out_dim);
        });
}

/// In-place Root Mean Square Normalization:
/// out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
#[inline]
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let dim = x.len();
    let mut sum_sq = 0.0f64;
    for &v in x.iter() {
        let vf = v as f64;
        sum_sq += vf * vf;
    }
    let mean_sq = sum_sq / (dim as f64);
    let scale = (1.0 / (mean_sq + (eps as f64)).sqrt()) as f32;

    for i in 0..dim {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Qwen3 per-head RMSNorm in place: `x` is `[heads][head_dim]`, every head is normalised over
/// its own `head_dim` values with the shared `weight` (`[head_dim]`), same maths as [`rmsnorm`].
pub fn rmsnorm_heads_in_place(x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    assert!(head_dim > 0 && x.len().is_multiple_of(head_dim) && weight.len() == head_dim);
    let mut scratch = vec![0.0f32; head_dim];
    for head in x.chunks_exact_mut(head_dim) {
        rmsnorm(head, weight, eps, &mut scratch);
        head.copy_from_slice(&scratch);
    }
}

/// Applies Rotary Positional Embeddings (RoPE) to query and key head slices.
/// Follows canonical Hugging Face LLaMA rotate_half pairing coordinate i with coordinate i + half_dim:
/// out[i] = q[i] * cos - q[i + half_dim] * sin
/// out[i + half_dim] = q[i] * sin + q[i + half_dim] * cos
#[inline]
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    apply_rope_params(
        q,
        k,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
        &RopeParams::plain(theta),
    );
}

/// The cos and sin of every rotary pair at `pos`: f64 frequency and angle, cast to f32.
/// Both the reference rope and the Omega GPU rope table use this.
pub fn rope_cos_sin(rope: &RopeParams, head_dim: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos = Vec::with_capacity(half_dim);
    let mut sin = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let rot = (pos as f64) * rope.inv_freq(i, head_dim);
        sin.push(rot.sin() as f32);
        cos.push(rot.cos() as f32);
    }
    (cos, sin)
}

/// [`apply_rope`] with explicit rope parameters (base theta and optional llama3 scaling).
#[inline]
pub fn apply_rope_params(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: &RopeParams,
) {
    let half_dim = head_dim / 2;
    let (cos, sin) = rope_cos_sin(rope, head_dim, pos);
    for (v, heads) in [(q, num_heads), (k, num_kv_heads)] {
        for h in 0..heads {
            let head_offset = h * head_dim;
            for i in 0..half_dim {
                let idx0 = head_offset + i;
                let idx1 = head_offset + i + half_dim;
                let v0 = v[idx0];
                let v1 = v[idx1];
                v[idx0] = v0 * cos[i] - v1 * sin[i];
                v[idx1] = v0 * sin[i] + v1 * cos[i];
            }
        }
    }
}

/// Scaled Dot-Product Grouped-Query Attention (GQA) for a single query token position:
/// Scores = Softmax((Q * K^T) / sqrt(d_k))
/// Out = Scores * V
pub fn scaled_dot_product_attention_single(
    q: &[f32],
    cached_k: &[Vec<f32>],
    cached_v: &[Vec<f32>],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), num_heads * head_dim);
    debug_assert_eq!(out.len(), num_heads * head_dim);
    debug_assert_eq!(cached_k.len(), cached_v.len());

    let seq_len = cached_k.len();
    if seq_len == 0 {
        out.fill(0.0);
        return;
    }

    let gqa_ratio = num_heads / num_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f64).sqrt();

    let mut scores = vec![0.0f64; seq_len];

    for h in 0..num_heads {
        let kv_head = h / gqa_ratio;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        // 1. Compute attention score for each cached token in f64
        for t in 0..seq_len {
            let k_t = &cached_k[t][kv_head * head_dim..(kv_head + 1) * head_dim];
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += (q_head[d] as f64) * (k_t[d] as f64);
            }
            scores[t] = dot * inv_sqrt_d;
        }

        // 2. Softmax over sequence length
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0f64;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        // 3. Weighted sum of values in f64
        let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
        for d in 0..head_dim {
            let mut sum = 0.0f64;
            for t in 0..seq_len {
                sum += scores[t] * (cached_v[t][kv_head * head_dim + d] as f64);
            }
            out_head[d] = sum as f32;
        }
    }
}

/// SwiGLU Feed-Forward Network:
/// out = (silu(x * W_gate) * (x * W_up)) * W_down
pub fn swiglu(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    out: &mut [f32],
) {
    let mut gate = vec![0.0f32; intermediate_dim];
    let mut up = vec![0.0f32; intermediate_dim];
    let mut activated = vec![0.0f32; intermediate_dim];

    matmul_vec(x, gate_w, &mut gate, hidden_dim, intermediate_dim);
    matmul_vec(x, up_w, &mut up, hidden_dim, intermediate_dim);

    // silu(x) = x / (1 + exp(-x)) with f64 precision
    for i in 0..intermediate_dim {
        let g = gate[i] as f64;
        let silu = g / (1.0 + (-g).exp());
        activated[i] = (silu * (up[i] as f64)) as f32;
    }

    matmul_vec(&activated, down_w, out, intermediate_dim, hidden_dim);
}

/// Softmax activation over logits vector in-place.
pub fn softmax_inplace(logits: &mut [f32]) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }
}

/// Greedy argmax token selection from logits vector.
/// Returns (token_id, logprob).
pub fn sample_argmax(logits: &[f32]) -> (u32, f32) {
    assert!(!logits.is_empty(), "Logits cannot be empty");

    let mut best_idx = 0;
    let mut best_val = logits[0];

    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }

    // Compute logprob = log(softmax(best_val))
    let max_val = best_val;
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    let logprob = best_val - max_val - sum_exp.ln();

    (best_idx as u32, logprob)
}

/// Temperature-scaled sampling with pseudo-random choice.
pub fn sample_temperature(logits: &[f32], temperature: f32, seed: u64) -> (u32, f32) {
    if temperature <= 0.001 {
        return sample_argmax(logits);
    }

    let mut scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    softmax_inplace(&mut scaled);

    // Deterministic pseudo-random number generator for reproducible sampling
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = ((state >> 33) as f32) / ((1u64 << 31) as f32);

    let mut cumulative = 0.0f32;
    for (idx, &prob) in scaled.iter().enumerate() {
        cumulative += prob;
        if r <= cumulative {
            return (idx as u32, prob.max(1e-12).ln());
        }
    }

    let last_idx = logits.len().saturating_sub(1);
    (last_idx as u32, scaled[last_idx].max(1e-12).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_invariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        rmsnorm(&x, &weight, 1e-5, &mut out);

        // RMS of output with all unit weights must equal sqrt(dim) * sqrt(1/dim) = 1.0
        let sum_sq: f32 = out.iter().map(|v| v * v).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rope_norm_preservation() {
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![0.5, 0.5, 0.5, 0.5];

        let orig_q_norm: f32 = q.iter().map(|v| v * v).sum();
        apply_rope(&mut q, &mut k, 5, 1, 1, 4, 10000.0);
        let new_q_norm: f32 = q.iter().map(|v| v * v).sum();

        assert!((orig_q_norm - new_q_norm).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_exact() {
        let x = vec![2.0, 3.0];
        // 3x2 row-major matrix [out_dim = 3, in_dim = 2]:
        // Row 0: [1.0, 4.0] -> 2*1 + 3*4 = 14
        // Row 1: [2.0, 5.0] -> 2*2 + 3*5 = 19
        // Row 2: [3.0, 6.0] -> 2*3 + 3*6 = 24
        let w = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut out = vec![0.0; 3];

        matmul_vec(&x, &w, &mut out, 2, 3);
        assert_eq!(out, vec![14.0, 19.0, 24.0]);
    }

    #[test]
    fn test_rope_rotate_half() {
        // Coordinate i pairs with i + half_dim
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![1.0, 2.0, 3.0, 4.0];

        // pos = 0 means angle = 0, so cos = 1, sin = 0 -> identity
        apply_rope(&mut q, &mut k, 0, 1, 1, 4, 10000.0);
        assert_eq!(q, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k, vec![1.0, 2.0, 3.0, 4.0]);

        // Verify norm preservation across non-zero position
        let mut q2 = vec![1.0, 2.0, 3.0, 4.0];
        let mut k2 = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = q2.iter().map(|v| v * v).sum();
        apply_rope(&mut q2, &mut k2, 7, 1, 1, 4, 10000.0);
        let norm_after: f32 = q2.iter().map(|v| v * v).sum();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn test_swiglu_activation() {
        let x = vec![1.0, 1.0];
        let gate_w = vec![1.0, 1.0]; // 2x1
        let up_w = vec![2.0, 2.0]; // 2x1
        let down_w = vec![1.0, 1.0]; // 1x2

        let mut out = vec![0.0; 2];
        swiglu(&x, &gate_w, &up_w, &down_w, 2, 1, &mut out);

        // gate = 2.0, silu(2.0) = 2.0 / (1 + exp(-2.0)) = 2.0 / 1.135335 = 1.761594
        // up = 4.0
        // activated = 1.761594 * 4.0 = 7.046376
        // out = [7.046376, 7.046376]
        assert!((out[0] - 7.046376).abs() < 1e-4);
        assert!((out[1] - 7.046376).abs() < 1e-4);
    }

    #[test]
    fn test_sample_argmax() {
        let logits = vec![-2.0, 0.5, 4.2, 1.1];
        let (tok, logprob) = sample_argmax(&logits);
        assert_eq!(tok, 2);
        assert!(logprob < 0.0);
    }
}
