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

//! Our CPU reference forward for BitNet b1.58 2B-4T (stage T2): the oracle the chip engine is
//! checked against. f32 arithmetic throughout, exact int32 ternary products.
//!
//! One layer, as spec 1.2 states it from `REF/model.py:237-242` (prefill path, `use_kernel=False`,
//! `REF/generate.py:56,63,70`), cross-checked with `HF/models/bitnet/modeling_bitnet.py`:
//!
//! ```text
//! a   = rms_norm(x, input_norm)                       model.py:237-241
//! q,k,v = BitLinear_qkv(a)  (one fused linear, rows [q | k | v], one weight scale per segment)
//! q,k = rope_half_split(q, k, position)               spec 1.3, HF :83-85, :299, :314
//! o   = causal GQA softmax(q k^T / sqrt(128)) v       HF :161; q head h reads kv head h / 4
//! h   = x + BitLinear_o(rms_norm(o, attn_sub_norm))   model.py:159-160
//! g,u = BitLinear_gate_up(rms_norm(h, post_norm))
//! out = h + BitLinear_down(rms_norm(relu(g)^2 * u, ffn_sub_norm))   model.py:192-197
//! logits = rms_norm(x_last_layer, final_norm) . embed^T            model.py:280-281 (tied)
//! ```
//!
//! BitLinear (spec 2.2, `REF/model.py:78-81`): per token `s_a = 127 / max(max|x|, 1e-5)`,
//! `q = clamp(round_half_even(x * s_a), -128, 127)`, then `y[r] = acc[r] / s_a * w_scale(r)` with
//! `acc` the exact int32 ternary product and `w_scale` the bf16 scale of the segment row `r`
//! belongs to (one scale each for q, k, v, gate, up; never the decode kernel's thirds,
//! `bitnet_kernels.h:80`).
//!
//! RMSNorm is `w * x / sqrt(mean(x^2) + eps)` with the square sum in f64 (HF :53-58 upcasts to
//! f32; f64 is the more exact oracle). RoPE angles are `pos * 500000^(-2j/128)` in f64, position
//! 0 for the first token (BOS when present).

use rayon::prelude::*;

use crate::bitnet_omega::{
    check_ternary_upload, code_trit, BitNetConfig, BitNetOmegaWeights, TernMatrix, TernPart,
    TernSegment, TernTensor, TernUploader, TRITS_PER_WORD,
};
use crate::qwen3_dense_omega::{DenseGemv, HostFp8Gemv};
use crate::qwen3_moe::bf16_to_f32;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Host matrices: the packed words unpacked once to one i8 trit per weight.
// ---------------------------------------------------------------------------

/// A ternary matrix held as `[n][k]` i8 trits (unpacked from [`crate::bitnet_omega::TERN_LAYOUT_ID`]
/// words after [`check_ternary_upload`]), with a token-batched exact int32 product.
#[derive(Debug)]
pub struct I8TernMatrix {
    n: usize,
    k: usize,
    trits: Vec<i8>,
    segments: Vec<TernSegment>,
}

impl I8TernMatrix {
    pub fn from_words(
        n: usize,
        k: usize,
        words: &[u32],
        segments: &[TernSegment],
    ) -> Result<Self, String> {
        check_ternary_upload(n, k, words, segments)?;
        let mut trits = vec![0i8; n * k];
        trits
            .par_chunks_mut(TRITS_PER_WORD)
            .zip(words.par_iter())
            .for_each(|(t, &w)| {
                for (i, v) in t.iter_mut().enumerate() {
                    // Validated above: no 0b11 code.
                    *v = code_trit(w >> (2 * i)).unwrap_or(0);
                }
            });
        Ok(Self {
            n,
            k,
            trits,
            segments: segments.to_vec(),
        })
    }

    /// `acc[t][r] = sum_c W[r][c] * x[t][c]` for `t` tokens, exact.
    pub fn gemm(&self, x: &[i8], t: usize, acc: &mut [i32]) -> Result<(), String> {
        let (n, k) = (self.n, self.k);
        if x.len() != t * k || acc.len() != t * n {
            return Err(format!(
                "ternary gemm [{n}][{k}] for {t} tokens given x {} acc {}",
                x.len(),
                acc.len()
            ));
        }
        const ROWS: usize = 32;
        const TOKENS: usize = 16;
        let blocks: Vec<Vec<i32>> = self
            .trits
            .par_chunks(ROWS * k)
            .map(|rows| {
                let nr = rows.len() / k;
                let mut out = vec![0i32; nr * t];
                for t0 in (0..t).step_by(TOKENS) {
                    let t1 = (t0 + TOKENS).min(t);
                    for (r, row) in rows.chunks_exact(k).enumerate() {
                        for tt in t0..t1 {
                            out[r * t + tt] = dot_i8(row, &x[tt * k..(tt + 1) * k]);
                        }
                    }
                }
                out
            })
            .collect();
        for (b, out) in blocks.iter().enumerate() {
            let nr = out.len() / t.max(1);
            for r in 0..nr {
                for tt in 0..t {
                    acc[tt * n + b * ROWS + r] = out[r * t + tt];
                }
            }
        }
        Ok(())
    }
}

impl TernMatrix for I8TernMatrix {
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
        self.gemm(x, 1, acc)
    }
    fn gemm_i8(&self, x: &[i8], t: usize, acc: &mut [i32]) -> Result<(), String> {
        self.gemm(x, t, acc)
    }
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    // Lanes of 16 so the compiler keeps independent vector sums.
    let mut lanes = [0i32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            lanes[i] += x[i] as i32 * y[i] as i32;
        }
    }
    let tail: i32 = ra.iter().zip(rb).map(|(&x, &y)| x as i32 * y as i32).sum();
    lanes.iter().sum::<i32>() + tail
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(&x, &y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

/// The host uploader for the reference: i8 trit matrices and the host e4m3 table.
#[derive(Debug, Default)]
pub struct HostRefUploader;

impl TernUploader for HostRefUploader {
    fn upload_ternary(
        &self,
        _layer: usize,
        _what: TernTensor,
        n: usize,
        k: usize,
        words: &[u32],
        segments: &[TernSegment],
    ) -> Result<Arc<dyn TernMatrix>, String> {
        Ok(Arc::new(I8TernMatrix::from_words(n, k, words, segments)?))
    }

    fn upload_table_fp8(
        &self,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        Ok(Arc::new(HostFp8Gemv::new(
            n,
            k,
            w.to_vec(),
            scales.to_vec(),
        )?))
    }
}

// ---------------------------------------------------------------------------
// The pieces of one layer.
// ---------------------------------------------------------------------------

/// Per-token int8 absmax quantisation (spec 2.2): returns `s_a`.
pub fn quant_act(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = 127.0 / amax.max(1e-5);
    for (o, &v) in q.iter_mut().zip(x) {
        *o = (v * s).round_ties_even().clamp(-128.0, 127.0) as i8;
    }
    s
}

/// `w * x / sqrt(mean(x^2) + eps)`, square sum in f64.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let inv = 1.0 / (ss / x.len() as f64 + eps as f64).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = (v as f64 * inv) as f32 * g;
    }
}

fn rms_norm_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(d)
        .zip(x.par_chunks(d))
        .for_each(|(o, xr)| rms_norm(xr, w, eps, o));
    out
}

/// cos and sin of `pos * theta^(-2j/d)`, j < d/2, in f64 then rounded to f32.
pub fn rope_table(pos: usize, d: usize, theta: f64) -> Vec<(f32, f32)> {
    (0..d / 2)
        .map(|j| {
            let a = pos as f64 * theta.powf(-((2 * j) as f64) / d as f64);
            let (s, c) = a.sin_cos();
            (c as f32, s as f32)
        })
        .collect()
}

/// Half-split rotation of one head: dim j pairs with j + d/2 (HF `rotate_half`).
pub fn rope_half_split(v: &mut [f32], table: &[(f32, f32)]) {
    let half = v.len() / 2;
    for (j, &(c, s)) in table.iter().enumerate() {
        let (x0, x1) = (v[j], v[j + half]);
        v[j] = x0 * c - x1 * s;
        v[j + half] = x1 * c + x0 * s;
    }
}

/// Deliberately wrong forwards, the negative controls of the T2 perplexity gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Spec 1.2.
    Reference,
    /// Wrong norm order: `attn_sub_norm` after `o_proj` instead of before it.
    SubNormAfterO,
    /// Both sub-norms skipped.
    SkipSubNorms,
    /// The q and v segment scales swapped. Swapping q and k alone would be a no-op (scores
    /// scale by `s_q * s_k` either way) and a v scale is undone by `attn_sub_norm`, so this
    /// scales every attention score by `s_v / s_q`.
    SwapQvScales,
    /// The q and k segment scales swapped: kept to show that control is vacuous.
    SwapQkScales,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::SubNormAfterO => "sub-norm-after-o",
            Self::SkipSubNorms => "skip-sub-norms",
            Self::SwapQvScales => "swap-q-v-scales",
            Self::SwapQkScales => "swap-q-k-scales",
        }
    }
}

/// Keys and values of every layer, `[len][kv_dim]` f32, roped keys.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub len: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(c: &BitNetConfig) -> Self {
        Self {
            len: 0,
            k: vec![Vec::new(); c.layers],
            v: vec![Vec::new(); c.layers],
        }
    }

    /// Drops every position from `len` on.
    pub fn truncate(&mut self, len: usize, kv_dim: usize) {
        for x in self.k.iter_mut().chain(self.v.iter_mut()) {
            x.truncate(len * kv_dim);
        }
        self.len = self.len.min(len);
    }
}

/// Causal GQA attention for `t` query rows at positions `pos0 ..` over the cached keys (roped)
/// and values `[pos0 + t][kv_dim]`: query head `h` reads kv head `h / (q_heads / kv_heads)`,
/// scores scaled by `head_dim^-0.5`, softmax over positions `0..=pos0 + i`. Returns `[t][q_dim]`.
pub fn attend(c: &BitNetConfig, q: &[f32], kc: &[f32], vc: &[f32], pos0: usize) -> Vec<f32> {
    let (qd, kvd, hd) = (c.q_dim(), c.kv_dim(), c.head_dim);
    let group = c.q_heads / c.kv_heads;
    let scale = (hd as f32).powf(-0.5);
    let t = q.len() / qd;
    let mut att = vec![0f32; t * qd];
    att.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
        let (i, head) = (idx / c.q_heads, idx % c.q_heads);
        let kvh = head / group;
        let p = pos0 + i;
        let qv = &q[i * qd + head * hd..i * qd + (head + 1) * hd];
        let scores: Vec<f32> = (0..=p)
            .map(|j| dot_f32(qv, &kc[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd]) * scale)
            .collect();
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
        let sum: f64 = e.iter().map(|&v| v as f64).sum();
        let inv = (1.0 / sum) as f32;
        for (j, &ej) in e.iter().enumerate() {
            let wgt = ej * inv;
            let vv = &vc[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
            for (od, &vd) in o.iter_mut().zip(vv) {
                *od += wgt * vd;
            }
        }
    });
    att
}

/// The reference model over loaded weights.
pub struct HostBitNetForward<'a> {
    pub w: &'a BitNetOmegaWeights,
    pub variant: Variant,
    /// The tied table widened to f32 once, `[vocab][hidden]`.
    embed: Vec<f32>,
}

impl<'a> HostBitNetForward<'a> {
    pub fn new(w: &'a BitNetOmegaWeights) -> Self {
        let embed = w.embed_bf16.par_iter().map(|&b| bf16_to_f32(b)).collect();
        Self {
            w,
            variant: Variant::Reference,
            embed,
        }
    }

    fn config(&self) -> &BitNetConfig {
        &self.w.config
    }

    /// One BitLinear over `t` token rows of `x`: `[t][n]` outputs.
    fn bit_linear(&self, m: &dyn TernMatrix, x: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let (n, k) = (m.n(), m.k());
        let mut q = vec![0i8; t * k];
        let s_a: Vec<f32> = q
            .par_chunks_mut(k)
            .zip(x.par_chunks(k))
            .map(|(qr, xr)| quant_act(xr, qr))
            .collect();
        let mut acc = vec![0i32; t * n];
        m.gemm_i8(&q, t, &mut acc)?;
        let mut seg: Vec<(TernPart, f32, usize)> = m
            .segments()
            .iter()
            .map(|s| (s.part, bf16_to_f32(s.scale_bf16), s.rows))
            .collect();
        let swap = match self.variant {
            Variant::SwapQvScales => Some((TernPart::Q, TernPart::V)),
            Variant::SwapQkScales => Some((TernPart::Q, TernPart::K)),
            _ => None,
        };
        if let Some((a, b)) = swap {
            let ia = seg.iter().position(|s| s.0 == a);
            let ib = seg.iter().position(|s| s.0 == b);
            if let (Some(ia), Some(ib)) = (ia, ib) {
                let sa = seg[ia].1;
                seg[ia].1 = seg[ib].1;
                seg[ib].1 = sa;
            }
        }
        let mut row_scale = Vec::with_capacity(n);
        for &(_, s, rows) in &seg {
            row_scale.extend(std::iter::repeat_n(s, rows));
        }
        let mut y = vec![0f32; t * n];
        y.par_chunks_mut(n)
            .zip(acc.par_chunks(n))
            .zip(s_a.par_iter())
            .for_each(|((yr, ar), &s)| {
                for ((o, &a), &ws) in yr.iter_mut().zip(ar).zip(&row_scale) {
                    *o = a as f32 / s * ws;
                }
            });
        Ok(y)
    }

    /// Runs `tokens` (positions `cache.len ..`) through every layer, appending to `cache`.
    /// Returns the final-norm hidden rows `[t][hidden]`; `taps` receives the residual stream
    /// after each listed layer, `[t][hidden]`. On an error the cache is rolled back to where it
    /// was, so it stays usable.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        tap_layers: &[usize],
        taps: &mut Vec<(usize, Vec<f32>)>,
    ) -> Result<Vec<f32>, String> {
        let len = cache.len;
        let r = self.forward_inner(tokens, cache, tap_layers, taps);
        if r.is_err() {
            cache.truncate(len, self.config().kv_dim());
        }
        r
    }

    fn forward_inner(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        tap_layers: &[usize],
        taps: &mut Vec<(usize, Vec<f32>)>,
    ) -> Result<Vec<f32>, String> {
        let c = self.config();
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let t = tokens.len();
        if t == 0 {
            return Err("forward of no tokens".into());
        }
        let pos0 = cache.len;
        if pos0 + t > c.max_position {
            return Err(format!(
                "positions {pos0}..{} beyond max_position {}",
                pos0 + t,
                c.max_position
            ));
        }
        let mut x = Vec::with_capacity(t * h);
        for &tok in tokens {
            x.extend(self.w.embed_row(tok)?);
        }
        let tables: Vec<Vec<(f32, f32)>> = (0..t)
            .map(|i| rope_table(pos0 + i, hd, f64::from(c.rope_theta)))
            .collect();

        for (li, layer) in self.w.layers.iter().enumerate() {
            // Attention.
            let a = rms_norm_rows(&x, &layer.input_norm, eps);
            let qkv = self.bit_linear(layer.qkv.as_ref(), &a, t)?;
            let n_qkv = qd + 2 * kvd;
            let mut q = vec![0f32; t * qd];
            for i in 0..t {
                let row = &qkv[i * n_qkv..(i + 1) * n_qkv];
                q[i * qd..(i + 1) * qd].copy_from_slice(&row[..qd]);
                for head in q[i * qd..(i + 1) * qd].chunks_exact_mut(hd) {
                    rope_half_split(head, &tables[i]);
                }
                let mut k = row[qd..qd + kvd].to_vec();
                for head in k.chunks_exact_mut(hd) {
                    rope_half_split(head, &tables[i]);
                }
                cache.k[li].extend_from_slice(&k);
                cache.v[li].extend_from_slice(&row[qd + kvd..]);
            }
            let att = attend(c, &q, &cache.k[li], &cache.v[li], pos0);
            let o = match self.variant {
                Variant::SubNormAfterO => {
                    let y = self.bit_linear(layer.o.as_ref(), &att, t)?;
                    rms_norm_rows(&y, &layer.attn_sub_norm, eps)
                }
                Variant::SkipSubNorms => self.bit_linear(layer.o.as_ref(), &att, t)?,
                _ => {
                    let n = rms_norm_rows(&att, &layer.attn_sub_norm, eps);
                    self.bit_linear(layer.o.as_ref(), &n, t)?
                }
            };
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }

            // Feed-forward.
            let f = rms_norm_rows(&x, &layer.post_norm, eps);
            let gu = self.bit_linear(layer.gate_up.as_ref(), &f, t)?;
            let im = c.intermediate;
            let mut inner = vec![0f32; t * im];
            for i in 0..t {
                let row = &gu[i * 2 * im..(i + 1) * 2 * im];
                let (g, u) = row.split_at(im);
                for ((o, &gv), &uv) in inner[i * im..(i + 1) * im].iter_mut().zip(g).zip(u) {
                    let r = gv.max(0.0);
                    *o = r * r * uv;
                }
            }
            let inner = match self.variant {
                Variant::SkipSubNorms => inner,
                _ => rms_norm_rows(&inner, &layer.ffn_sub_norm, eps),
            };
            let d = self.bit_linear(layer.down.as_ref(), &inner, t)?;
            for (xv, dv) in x.iter_mut().zip(&d) {
                *xv += dv;
            }
            if tap_layers.contains(&li) {
                taps.push((li, x.clone()));
            }
        }
        cache.len += t;
        Ok(rms_norm_rows(&x, &self.w.final_norm, eps))
    }

    /// Logits `[t][vocab]` of final-norm rows, with the tied 16-bit table (exact in f32).
    pub fn logits(&self, normed: &[f32]) -> Vec<f32> {
        let c = self.config();
        let (h, vocab) = (c.hidden, c.vocab);
        let t = normed.len() / h;
        if t == 0 {
            return Vec::new();
        }
        const ROWS: usize = 64;
        let blocks: Vec<Vec<f32>> = self
            .embed
            .par_chunks(ROWS * h)
            .map(|rows| {
                let nr = rows.len() / h;
                let mut out = vec![0f32; nr * t];
                for (r, row) in rows.chunks_exact(h).enumerate() {
                    for i in 0..t {
                        out[r * t + i] = dot_f32(row, &normed[i * h..(i + 1) * h]);
                    }
                }
                out
            })
            .collect();
        let mut logits = vec![0f32; t * vocab];
        for (b, out) in blocks.iter().enumerate() {
            for r in 0..out.len() / t {
                for i in 0..t {
                    logits[i * vocab + b * ROWS + r] = out[r * t + i];
                }
            }
        }
        logits
    }

    /// Teacher-forced negative log-likelihoods (natural log, f64): row `i` of `normed`
    /// predicts `targets[i]`. Done in chunks so the logits never exceed 64 rows.
    pub fn nll(&self, normed: &[f32], targets: &[u32]) -> Result<Vec<f64>, String> {
        let (h, vocab) = (self.config().hidden, self.config().vocab);
        if normed.len() != targets.len() * h {
            return Err(format!(
                "{} hidden rows for {} targets",
                normed.len() / h,
                targets.len()
            ));
        }
        let mut out = Vec::with_capacity(targets.len());
        for (rows, tg) in normed.chunks(64 * h).zip(targets.chunks(64)) {
            let lg = self.logits(rows);
            for (l, &tgt) in lg.chunks_exact(vocab).zip(tg) {
                if tgt as usize >= vocab {
                    return Err(format!("target {tgt} outside the vocabulary"));
                }
                out.push(log_sum_exp(l) - l[tgt as usize] as f64);
            }
        }
        Ok(out)
    }
}

/// `ln(sum exp(l))` in f64.
pub fn log_sum_exp(l: &[f32]) -> f64 {
    let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

/// The highest logit, lowest id on a tie.
pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests;
