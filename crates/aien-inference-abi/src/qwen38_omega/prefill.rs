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

//! Whole-prompt prefill for Qwen3.8-27B on omega's own prefill kernels (sc #642 step 9).
//!
//! The prompt goes through the model in chunks of at most `chunk` tokens; every layer of a chunk
//! costs a fixed number of chip calls, whatever the chunk length. The DeltaNet tails and states
//! and the KV cache end exactly where the decode graph ([`Qwen38Omega::step`]) continues.
//!
//! Per layer of a chunk of `m` tokens:
//!
//! | step | where |
//! |---|---|
//! | input RMSNorm (zero-centred, weights folded to `1 + w`) | host |
//! | quantise x (f32, k 5120) | chip, `omega_gpu_pf_quant` |
//! | mixer projection, one FP8 GEMM: qkv \| z (DeltaNet) or q \| gate \| k \| v (attention) | chip, `omega_gpu_pf_qgemm` |
//! | DeltaNet: a \| b, bf16 `[m][5120] x [5120][96]` | chip, `omega_gpu_matmul_resident_f32` |
//! | DeltaNet: conv, delta rule and gated norm for all `m` tokens | chip, `omega_gpu_gdn_prefill` |
//! | attention: rows gathered to the kernel's layout ([`aien_omega_q38::gather_table`]) | host |
//! | attention: q/k norm, RoPE, KV append, causal attention | chip, `omega_gpu_prefill_attn256` |
//! | attention: output gate, `* sigmoid(gate)` | host |
//! | quantise the mixer output (f32, k 6144), o_proj FP8 GEMM | chip |
//! | residual add, post-attention RMSNorm | host |
//! | quantise, gate \| up FP8 GEMM with the SwiGLU epilogue (bf16 act rows) | chip |
//! | quantise act (bf16, k 17408), down FP8 GEMM | chip |
//! | residual add | host |
//!
//! After the last chunk: the last token's final norm, the bf16 lm_head GEMV and a host argmax
//! give the first generated token.
//!
//! This first cut moves every mixer input and output through host memory: the GEMM outputs land
//! in GPU-uncached HOST buffers (nvos.h:1117 rule, as in the OWN-30B driver) and the omega host
//! entries copy through their own staging. The device-pointer entries are the speed follow-up.
use std::time::Instant;

use aien_omega_fp8::Fp8Tensor;
use aien_omega_pf::{
    build_tiles, Layout, OmegaPfDevice, PfBuffer, PfMem, PrefillDevice, QgemmCall, QuantIn, BLOCK,
    TILE_BYTES,
};
use aien_omega_q38 as q;
use rayon::prelude::*;

use super::{gpu, Graph, Qwen38Omega, EPS_L2};
use crate::qwen3_moe::bf16_to_f32;
use crate::tensor::rmsnorm;

/// What one [`Qwen38Omega::prefill`] call cost. Chip times are the omega info records' launch
/// times; call times are whole calls including omega's host copies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefillStats {
    pub tokens: usize,
    pub chunks: usize,
    /// The whole call: for a fresh sequence, the time to first token.
    pub wall_ns: u64,
    /// Quantisers and FP8 GEMMs: chip time, wall time of their calls, launches.
    pub gemm_chip_ns: u64,
    pub gemm_call_ns: u64,
    pub gemm_launches: u64,
    /// DeltaNet a|b resident bf16 matmul (wall time of the calls).
    pub ab_call_ns: u64,
    /// DeltaNet prefill calls.
    pub gdn_chip_ns: u64,
    pub gdn_call_ns: u64,
    /// Attention prefill calls.
    pub attn_chip_ns: u64,
    pub attn_call_ns: u64,
    /// Host work: embedding, norms, residual adds, copies out of the HOST buffers, row splits,
    /// gathers and the output gate.
    pub host_ns: u64,
    /// Final norm, lm_head and argmax.
    pub head_ns: u64,
}

/// The prefill's buffers for chunks of up to `m` tokens (sizes in the field comments).
pub(super) struct PfState {
    dev: OmegaPfDevice,
    m: usize,
    bm: usize,
    max_tiles: usize,
    /// HOST f32 `[m][6144]`: quantiser input (normed x, mixer output, post-normed x).
    xin: PfMem,
    /// DEV e4m3 `[m + bm][17408]` and f32 scales `[m + bm][17408 / 128]` (a GEMM tile reads
    /// activation rows up to `row0 + bm - 1`).
    xq: PfMem,
    xs: PfMem,
    /// HOST f32 `[m][16384]`: the mixer projection (row stride = that GEMM's `n`).
    mix: PfMem,
    /// DEV bf16 `[m][17408]`: SwiGLU act rows.
    act: PfMem,
    /// HOST f32 `[m][5120]`: o_proj and down outputs.
    out: PfMem,
    /// HOST `max_tiles` x 32 bytes: one GEMM's tile table (every call is synchronous).
    tiles: PfMem,
    /// [`q::gather_table`] for the model's rotary width.
    gather: Vec<u32>,
}

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

fn as_f32(b: &[u8]) -> Result<&[f32], String> {
    // SAFETY: every bit pattern is a valid f32.
    let (pre, mid, post) = unsafe { b.align_to::<f32>() };
    if !pre.is_empty() || !post.is_empty() {
        return Err("host buffer not f32-aligned".into());
    }
    Ok(mid)
}

fn as_f32_mut(b: &mut [u8]) -> Result<&mut [f32], String> {
    // SAFETY: every bit pattern is a valid f32.
    let (pre, mid, post) = unsafe { b.align_to_mut::<f32>() };
    if !pre.is_empty() || !post.is_empty() {
        return Err("host buffer not f32-aligned".into());
    }
    Ok(mid)
}

fn host<'a>(b: &'a PfMem, what: &str) -> Result<&'a [f32], String> {
    as_f32(
        b.host()
            .ok_or_else(|| format!("{what} is not a host buffer"))?,
    )
}

fn host_mut<'a>(b: &'a mut PfMem, what: &str) -> Result<&'a mut [f32], String> {
    as_f32_mut(
        b.host_mut()
            .ok_or_else(|| format!("{what} is not a host buffer"))?,
    )
}

/// `n` values of a HOST buffer copied into cached memory (one sequential pass, in parallel).
fn copy_out(b: &PfMem, what: &str, n: usize) -> Result<Vec<f32>, String> {
    let src = &host(b, what)?[..n];
    let mut v = vec![0f32; n];
    v.par_chunks_mut(1 << 16)
        .zip(src.par_chunks(1 << 16))
        .for_each(|(d, s)| d.copy_from_slice(s));
    Ok(v)
}

/// `src` written into the start of a HOST buffer (in parallel).
fn copy_in(b: &mut PfMem, what: &str, src: &[f32]) -> Result<(), String> {
    let dst = &mut host_mut(b, what)?[..src.len()];
    dst.par_chunks_mut(1 << 16)
        .zip(src.par_chunks(1 << 16))
        .for_each(|(d, s)| d.copy_from_slice(s));
    Ok(())
}

/// Index of the largest logit (the first one on a tie).
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// CPU model of the graph's two-stage ARGMAX (omega a647b6a9 `omega_blackwell_codegen_dec.h`
/// ARGMAX, `omega_gpu_q36_graph.c` `rec_argmax`): stage 1, CTA `c` of `ceil(n / span)` takes
/// `v[c * span ..]` (the tail padded with `NEG_PAD` -1e30) and writes its largest value and the
/// lowest index holding it; stage 2 does the same over the partials (padded with -1e30), carrying
/// their indices. omega's SASS simulator checks the kernel against this rule on ties
/// (`tests/gpu_cmdlist_test.c` "sim argmax tie", the last-equal-index mutant caught). The tests
/// take it as the chip's argmax (DRA-87).
#[cfg(test)]
pub(crate) fn chip_argmax_model(v: &[f32], span: usize) -> u32 {
    const NEG_PAD: f32 = -1e30;
    let lowest_max = |vals: &[(f32, u32)]| {
        let m = vals.iter().fold(f32::NEG_INFINITY, |m, &(x, _)| m.max(x));
        *vals
            .iter()
            .filter(|&&(x, _)| x == m)
            .min_by_key(|&&(_, i)| i)
            .expect("a CTA holds at least one value")
    };
    let ctas = v.len().div_ceil(span);
    let mut partials: Vec<(f32, u32)> = (0..ctas)
        .map(|c| {
            let vals: Vec<(f32, u32)> = (c * span..(c + 1) * span)
                .map(|i| (v.get(i).copied().unwrap_or(NEG_PAD), i as u32))
                .collect();
            lowest_max(&vals)
        })
        .collect();
    partials.resize(span.max(ctas), (NEG_PAD, u32::MAX));
    lowest_max(&partials).1
}

/// Chunk lengths for `n` tokens: at most `chunk` each, none shorter than the DeltaNet prefill's
/// minimum (the second to last chunk gives up tokens when the remainder would be too short).
pub(super) fn chunk_lengths(n: usize, chunk: usize) -> Vec<usize> {
    let min = q::GDN_PF_MIN_T;
    let mut out = Vec::with_capacity(n.div_ceil(chunk.max(1)));
    let mut rem = n;
    while rem > 0 {
        let m = if rem <= chunk {
            rem
        } else if rem - chunk < min {
            rem - min
        } else {
            chunk
        };
        out.push(m);
        rem -= m;
    }
    out
}

impl PfState {
    fn new(m: usize, rotary: usize) -> Result<Self, String> {
        let mut dev = OmegaPfDevice;
        let bm = dev.qgemm_bm();
        let kq = q::INTER; // widest quantiser input
        let max_tiles = m.div_ceil(bm) + 1;
        Ok(Self {
            xin: dev.alloc_host(m * q::MIXER_OUT * 4)?,
            xq: dev.alloc_dev((m + bm) * kq)?,
            xs: dev.alloc_dev((m + bm) * (kq / BLOCK) * 4)?,
            mix: dev.alloc_host(m * q::GDN_MIX * 4)?,
            act: dev.alloc_dev(m * q::INTER * 2)?,
            out: dev.alloc_host(m * q::HIDDEN * 4)?,
            tiles: dev.alloc_host(max_tiles * TILE_BYTES)?,
            gather: q::gather_table(rotary),
            dev,
            m,
            bm,
            max_tiles,
        })
    }

    /// Quantises `m` rows of `k` at `x_va` into `xq` / `xs`.
    fn quant(
        &mut self,
        k: usize,
        input: QuantIn,
        x_va: u64,
        m: usize,
        st: &mut PrefillStats,
    ) -> Result<(), String> {
        let t = Instant::now();
        st.gemm_chip_ns += self
            .dev
            .quant(k, input, m, x_va, self.xq.va(), self.xs.va())?;
        st.gemm_call_ns += ns(t);
        st.gemm_launches += 1;
        Ok(())
    }

    /// `y[m][n] = xq . W^T` for one FP8 tensor `[n][k]` (or the SwiGLU act rows of `n / 2`).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        w: &Fp8Tensor,
        n: usize,
        k: usize,
        layout: Layout,
        y_va: u64,
        m: usize,
        st: &mut PrefillStats,
    ) -> Result<(), String> {
        let t = Instant::now();
        let (w_va, s_va) = w.entry(0).map_err(|e| e.to_string())?;
        let tab = build_tiles(self.bm, &[w_va], &[s_va], &[m as u32]).map_err(|e| e.to_string())?;
        if tab.len() > self.max_tiles {
            return Err(format!(
                "tile table of {} entries > {}",
                tab.len(),
                self.max_tiles
            ));
        }
        let bytes = self
            .tiles
            .host_mut()
            .ok_or("tile table is not a host buffer")?;
        for (j, tl) in tab.iter().enumerate() {
            bytes[j * TILE_BYTES..(j + 1) * TILE_BYTES].copy_from_slice(&tl.to_bytes());
        }
        let call = QgemmCall {
            n,
            k,
            layout,
            tiles_va: self.tiles.va(),
            ntiles: tab.len(),
            rowidx_va: 0,
            xq_va: self.xq.va(),
            xs_va: self.xs.va(),
            y_va,
        };
        st.gemm_chip_ns += self.dev.qgemm(&call)?;
        st.gemm_call_ns += ns(t);
        st.gemm_launches += 1;
        Ok(())
    }
}

impl Qwen38Omega {
    /// Feeds `tokens` at the next positions in chunks of at most `chunk` tokens and returns the
    /// greedy next token (the first generated token), its logits and what the call cost. At
    /// least [`q::GDN_PF_MIN_T`] tokens; `chunk` in `6 ..= 32768`.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(u32, Vec<f32>, PrefillStats), String> {
        let (next, logits, st) = self.prefill_head(tokens, chunk, true)?;
        Ok((next, logits.ok_or("prefill: no logits")?, st))
    }

    /// [`Self::prefill`] without the logits (DRA-87, a greedy first token): the 35B returns the
    /// chip's argmax of the last token's head (the host rule: the largest logit, the first index
    /// on a tie) and reads no logits back; other models take the host argmax of the head's
    /// logits, as [`Self::prefill`] does.
    pub fn prefill_token(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(u32, PrefillStats), String> {
        self.prefill_head(tokens, chunk, false)
            .map(|(next, _, st)| (next, st))
    }

    fn prefill_head(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        want_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>, PrefillStats), String> {
        let t0 = Instant::now();
        if self.is_moe() {
            // The 35B: chunks of the prompt through the chip as rows (L369, the decode step's kernels,
            // the state bit for bit as token by token); AIEN_Q36_PREFILL_CHUNK=0 feeds the decode
            // graph token by token (the old path).
            let chunk = q36_prefill_chunk()?;
            let mut st = PrefillStats {
                tokens: tokens.len(),
                ..Default::default()
            };
            let mut next = 0u32;
            if chunk == 0 {
                for &t in tokens {
                    next = self.step(t)?.0;
                }
            } else {
                let (n, chunks) = self.prefill_moe(tokens, chunk)?;
                next = n;
                st.chunks = chunks;
            }
            let logits = if want_logits {
                Some(self.logits()?)
            } else {
                None
            };
            st.wall_ns = ns(t0);
            return Ok((next, logits, st));
        }
        let (last, mut st) = self.prefill_rows(tokens, chunk, false)?;
        let t = Instant::now();
        let logits = self.head_logits(&last)?;
        let next = argmax(&logits);
        st.head_ns = ns(t);
        st.wall_ns = ns(t0);
        Ok((next, want_logits.then_some(logits), st))
    }

    /// Feeds `tokens` at head 0's next positions for their state only: no head, no residual read
    /// back (a session's prompt rows before its snapshot point, DRA-65). The 35B: chunks of
    /// [`q36_prefill_chunk`] through the chip prefill (L369, any length, the state bit for bit as
    /// token by token; with 0 every token is stepped through the decode graph and the stats'
    /// `chunks` is 0); other models: [`Self::prefill_rows`] with `chunk` (at least
    /// [`q::GDN_PF_MIN_T`] tokens). Nothing to feed is a no-op.
    pub fn prefill_state(&mut self, tokens: &[u32], chunk: usize) -> Result<PrefillStats, String> {
        let t0 = Instant::now();
        if !self.is_moe() {
            return self.prefill_rows(tokens, chunk, false).map(|(_, st)| st);
        }
        let chunks = q36_state(self, tokens, q36_prefill_chunk()?)?;
        Ok(PrefillStats {
            tokens: tokens.len(),
            chunks,
            wall_ns: ns(t0),
            ..Default::default()
        })
    }

    /// The final norm and the bf16 lm_head on one residual row (`[hidden]`): `[vocab]` logits.
    pub fn head_logits(&mut self, residual: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.config.hidden;
        if residual.len() != h {
            return Err(format!(
                "head_logits: {} values (hidden {h})",
                residual.len()
            ));
        }
        let mut hidden = vec![0f32; h];
        rmsnorm(residual, &self.final_norm, self.config.eps, &mut hidden);
        let mut logits = vec![0f32; self.config.vocab];
        self.lm_head
            .gemv_f32(&hidden, &mut logits)
            .map_err(gpu("lm_head"))?;
        Ok(logits)
    }

    /// As [`Self::prefill`] without the head: the residual stream after the last layer of every
    /// token (`all`, `[n][hidden]`) or of the last token only (`[hidden]`). `wall_ns` covers the
    /// layers only.
    pub fn prefill_rows(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        all: bool,
    ) -> Result<(Vec<f32>, PrefillStats), String> {
        let t0 = Instant::now();
        let n = tokens.len();
        let (h, v) = (self.config.hidden, self.config.vocab);
        if self.is_moe() {
            // The 35B's chip prefill (L369) keeps its rows in its own buffers, not in the decode
            // graph's x, so the rows asked for come from decode steps: every token's when `all`;
            // else the others go in as chunks ([`Self::prefill_state`]) and the last is stepped.
            let (head, stepped) = if all || n == 0 {
                (&tokens[..0], tokens)
            } else {
                tokens.split_at(n - 1)
            };
            let chunks = self.prefill_state(head, chunk)?.chunks;
            let mut rows = Vec::with_capacity(if all { n * h } else { h });
            for &t in stepped {
                self.step(t)?;
                rows.extend(self.residual()?);
            }
            return Ok((
                rows,
                PrefillStats {
                    tokens: n,
                    chunks,
                    wall_ns: ns(t0),
                    ..Default::default()
                },
            ));
        }
        if !q::has_prefill() {
            return Err("qwen38 prefill: omega's prefill entries are not linked".into());
        }
        if n < q::GDN_PF_MIN_T {
            return Err(format!(
                "qwen38 prefill: {n} tokens (at least {})",
                q::GDN_PF_MIN_T
            ));
        }
        if !(2 * q::GDN_PF_MIN_T..=q::PATTN_MAX_M).contains(&chunk) {
            return Err(format!(
                "qwen38 prefill: chunk {chunk} (6..={})",
                q::PATTN_MAX_M
            ));
        }
        if self.pos + n > self.max_pos {
            return Err(format!(
                "qwen38 prefill: positions {} + {n} > KV cache of {}",
                self.pos, self.max_pos
            ));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= v) {
            return Err(format!("qwen38 prefill: token {t} outside the vocab {v}"));
        }
        let want = chunk.min(n);
        if self.pf.as_ref().is_none_or(|p| p.m < want) {
            self.pf = None; // free the old sizing first
            self.pf = Some(PfState::new(want, self.config.rotary_dim)?);
        }
        let mut st = PrefillStats {
            tokens: n,
            ..Default::default()
        };
        let mut rows = Vec::with_capacity(if all { n * h } else { h });
        let mut done = 0;
        for m in chunk_lengths(n, chunk) {
            let x = self.prefill_chunk(&tokens[done..done + m], &mut st)?;
            if all {
                rows.extend_from_slice(&x);
            } else if done + m == n {
                rows.extend_from_slice(&x[(m - 1) * h..]);
            }
            done += m;
            st.chunks += 1;
        }
        st.wall_ns = ns(t0);
        Ok((rows, st))
    }

    /// One chunk through every layer at positions `self.pos ..`: the residual stream after the
    /// last layer, `[m][hidden]`.
    fn prefill_chunk(&mut self, toks: &[u32], st: &mut PrefillStats) -> Result<Vec<f32>, String> {
        let m = toks.len();
        let pos0 = self.pos;
        let (h, eps, layers) = (self.config.hidden, self.config.eps, self.config.layers);
        let (mo, inter) = (q::MIXER_OUT, q::INTER);
        let pf = self.pf.as_mut().expect("prefill buffers sized");

        let t = Instant::now();
        let mut x: Vec<f32> = toks
            .iter()
            .flat_map(|&tk| {
                let r = tk as usize * h;
                self.embed[r..r + h].iter().map(|&b| bf16_to_f32(b))
            })
            .collect();
        let mut xn = vec![0f32; m * h];
        let mut att = vec![0f32; m * mo];
        st.host_ns += ns(t);

        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..layers {
            let lt = &self.layers[l];
            let (in_norm, post_norm) = &self.norms[l];
            let attention = self.config.full_attention[l];

            let t = Instant::now();
            xn.par_chunks_mut(h)
                .zip(x.par_chunks(h))
                .for_each(|(o, r)| rmsnorm(r, in_norm, eps, o));
            copy_in(&mut pf.xin, "xin", &xn)?;
            st.host_ns += ns(t);
            let xin_va = pf.xin.va();
            pf.quant(h, QuantIn::F32, xin_va, m, st)
                .map_err(|e| format!("layer {l} quant x: {e}"))?;
            let mix_n = if attention { q::ATT_MIX } else { q::GDN_MIX };
            let mix_va = pf.mix.va();
            pf.gemm(&lt.mix, mix_n, h, Layout::Plain, mix_va, m, st)
                .map_err(|e| format!("layer {l} mixer gemm x{m}: {e}"))?;

            let t = Instant::now();
            let mix = copy_out(&pf.mix, "mix", m * mix_n)?;
            st.host_ns += ns(t);
            if attention {
                let t = Instant::now();
                let mut qkv = vec![0f32; m * q::ATT_QKV];
                let mut gate = vec![0f32; m * mo];
                let (gq, gg) = pf.gather.split_at(q::ATT_QKV);
                qkv.par_chunks_mut(q::ATT_QKV)
                    .zip(gate.par_chunks_mut(mo))
                    .zip(mix.par_chunks(mix_n))
                    .for_each(|((a, g), r)| {
                        for (o, &s) in a.iter_mut().zip(gq) {
                            *o = r[s as usize];
                        }
                        for (o, &s) in g.iter_mut().zip(gg) {
                            *o = r[s as usize];
                        }
                    });
                st.host_ns += ns(t);
                let cost = self
                    .attn
                    .as_mut()
                    .expect("an attention layer has a cache")
                    .prefill(ai, pos0, m, &qkv, eps, &mut att)
                    .map_err(|e| format!("layer {l} attention x{m}: {e}"))?;
                st.attn_chip_ns += cost.chip_ns;
                st.attn_call_ns += cost.call_ns;
                ai += 1;
                let t = Instant::now();
                att.par_chunks_mut(mo)
                    .zip(gate.par_chunks(mo))
                    .for_each(|(a, g)| {
                        for (v, &gv) in a.iter_mut().zip(g) {
                            *v /= 1.0 + (-gv).exp();
                        }
                    });
                st.host_ns += ns(t);
            } else {
                let t = Instant::now();
                let mut qkv = vec![0f32; m * q::GDN_QKV];
                let mut z = vec![0f32; m * mo];
                qkv.par_chunks_mut(q::GDN_QKV)
                    .zip(z.par_chunks_mut(mo))
                    .zip(mix.par_chunks(mix_n))
                    .for_each(|((a, b), r)| {
                        a.copy_from_slice(&r[..q::GDN_QKV]);
                        b.copy_from_slice(&r[q::GDN_QKV..]);
                    });
                st.host_ns += ns(t);
                let t = Instant::now();
                let mut ab = vec![0f32; m * q::AB];
                self.ab_mm[gi]
                    .matmul_f32(m, &xn, &mut ab)
                    .map_err(|e| format!("layer {l} a|b matmul x{m}: {e}"))?;
                st.ab_call_ns += ns(t);
                let cost = self
                    .gdn
                    .as_mut()
                    .expect("a DeltaNet layer has state")
                    .prefill(gi, m, &qkv, &z, &ab, EPS_L2, eps, &mut att)
                    .map_err(|e| format!("layer {l} gdn prefill x{m}: {e}"))?;
                st.gdn_chip_ns += cost.chip_ns;
                st.gdn_call_ns += cost.call_ns;
                gi += 1;
            }

            // o_proj, residual, post norm
            let t = Instant::now();
            copy_in(&mut pf.xin, "xin", &att)?;
            st.host_ns += ns(t);
            pf.quant(mo, QuantIn::F32, xin_va, m, st)
                .map_err(|e| format!("layer {l} quant mixer out: {e}"))?;
            let out_va = pf.out.va();
            pf.gemm(&lt.o, h, mo, Layout::Plain, out_va, m, st)
                .map_err(|e| format!("layer {l} o gemm x{m}: {e}"))?;
            let t = Instant::now();
            let o = copy_out(&pf.out, "out", m * h)?;
            x.par_chunks_mut(1 << 14)
                .zip(o.par_chunks(1 << 14))
                .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(a, b)| *a += b));
            xn.par_chunks_mut(h)
                .zip(x.par_chunks(h))
                .for_each(|(o, r)| rmsnorm(r, post_norm, eps, o));
            copy_in(&mut pf.xin, "xin", &xn)?;
            st.host_ns += ns(t);

            // MLP
            pf.quant(h, QuantIn::F32, xin_va, m, st)
                .map_err(|e| format!("layer {l} quant post: {e}"))?;
            let act_va = pf.act.va();
            pf.gemm(&lt.gate_up, 2 * inter, h, Layout::SwigluNat, act_va, m, st)
                .map_err(|e| format!("layer {l} gate|up gemm x{m}: {e}"))?;
            pf.quant(inter, QuantIn::Bf16, act_va, m, st)
                .map_err(|e| format!("layer {l} quant act: {e}"))?;
            pf.gemm(&lt.down, h, inter, Layout::Plain, out_va, m, st)
                .map_err(|e| format!("layer {l} down gemm x{m}: {e}"))?;
            let t = Instant::now();
            let d = copy_out(&pf.out, "out", m * h)?;
            x.par_chunks_mut(1 << 14)
                .zip(d.par_chunks(1 << 14))
                .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(a, b)| *a += b));
            st.host_ns += ns(t);
        }
        self.pos += m;
        Ok(x)
    }
}

/// Prompt tokens per 35B prefill chunk (`AIEN_Q36_PREFILL_CHUNK`, default the graph's most,
/// [`q::q36::PREFILL_MAX_ROWS`]); 0 = token by token through the decode graph.
pub fn q36_prefill_chunk() -> Result<usize, String> {
    parse_q36_chunk(std::env::var("AIEN_Q36_PREFILL_CHUNK").ok().as_deref())
}

fn parse_q36_chunk(v: Option<&str>) -> Result<usize, String> {
    match v {
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n <= q::q36::PREFILL_MAX_ROWS => Ok(n),
            _ => Err(format!(
                "AIEN_Q36_PREFILL_CHUNK={v}: 0 (off) ..= {}",
                q::q36::PREFILL_MAX_ROWS
            )),
        },
        None => Ok(q::q36::PREFILL_MAX_ROWS),
    }
}

/// The lengths of the 35B prefill's chunks for `n` tokens: whole chunks of `chunk` and the rest
/// last (the q36 prefill takes any length from 1, unlike the 27B's [`chunk_lengths`]).
pub fn q36_chunk_lengths(n: usize, chunk: usize) -> Vec<usize> {
    if chunk == 0 {
        return Vec::new();
    }
    let mut out = vec![chunk; n / chunk];
    if n % chunk > 0 {
        out.push(n % chunk);
    }
    out
}

/// What the 35B's state feed drives: one decode step, or chip prefill chunks of `chunk` at head
/// 0's next positions without the head (the model; a recording fake in the session tests).
pub(crate) trait Q36Feed {
    fn q36_step(&mut self, t: u32) -> Result<(), String>;
    /// Returns the chunks run.
    fn q36_chunks(&mut self, tokens: &[u32], chunk: usize) -> Result<usize, String>;
}

impl Q36Feed for Qwen38Omega {
    fn q36_step(&mut self, t: u32) -> Result<(), String> {
        self.step(t).map(|_| ())
    }

    fn q36_chunks(&mut self, tokens: &[u32], chunk: usize) -> Result<usize, String> {
        Ok(self.prefill_moe_head(0, tokens, chunk, false)?.1)
    }
}

/// [`Qwen38Omega::prefill_state`] on the 35B: `c == 0` steps every token, else chunks of `c`.
/// Returns the chunks run (0 when stepped or nothing to feed).
pub(crate) fn q36_state<F: Q36Feed>(f: &mut F, tokens: &[u32], c: usize) -> Result<usize, String> {
    if c == 0 {
        for &t in tokens {
            f.q36_step(t)?;
        }
        return Ok(0);
    }
    if tokens.is_empty() {
        return Ok(0);
    }
    f.q36_chunks(tokens, c)
}

/// Walks `tokens` in the chunks of [`q36_chunk_lengths`] from position `pos0`: `each(chunk, pos,
/// last)`. Returns the chunks walked and the position after the last token.
pub(crate) fn q36_chunk_walk(
    tokens: &[u32],
    chunk: usize,
    pos0: usize,
    mut each: impl FnMut(&[u32], usize, bool) -> Result<(), String>,
) -> Result<(usize, usize), String> {
    let lengths = q36_chunk_lengths(tokens.len(), chunk);
    let (mut at, mut pos) = (0, pos0);
    for (i, &m) in lengths.iter().enumerate() {
        each(&tokens[at..at + m], pos, i + 1 == lengths.len())?;
        at += m;
        pos += m;
    }
    Ok((lengths.len(), pos))
}

impl Qwen38Omega {
    /// L369: fp8 GEMM weight bytes the 35B prefill has streamed so far on this graph (0 for other
    /// models); see `Graph::prefill_streamed`.
    pub fn q36_prefill_streamed(&self) -> u64 {
        match &self.graph {
            Graph::Moe(g) => g.prefill_streamed(),
            _ => 0,
        }
    }

    /// The 35B prefill (L369): `tokens` at the next positions of head 0 in chunks of at most
    /// `chunk` (1..=[`q::q36::PREFILL_MAX_ROWS`]); returns the last token's greedy token and the
    /// chunk count. The graph's logits / [`Self::logits`] are the last token's, as after `step`.
    pub fn prefill_moe(&mut self, tokens: &[u32], chunk: usize) -> Result<(u32, usize), String> {
        let (next, chunks) = self.prefill_moe_head(0, tokens, chunk, true)?;
        Ok((next.unwrap_or(0), chunks))
    }

    /// As [`Self::prefill_moe`] for graph head slot `head` at its next position (which moves on by
    /// `tokens.len()`); `want_token`: the last token's head and argmax.
    pub fn prefill_moe_head(
        &mut self,
        head: usize,
        tokens: &[u32],
        chunk: usize,
        want_token: bool,
    ) -> Result<(Option<u32>, usize), String> {
        let (h, v) = (self.config.hidden, self.config.vocab);
        let pos0 = self.head_pos(head)?;
        if tokens.is_empty() || chunk == 0 || chunk > q::q36::PREFILL_MAX_ROWS {
            return Err(format!(
                "q36 prefill: {} tokens in chunks of {chunk}",
                tokens.len()
            ));
        }
        if pos0 + tokens.len() > self.max_pos {
            return Err(format!(
                "q36 prefill: positions {pos0}..{} past the KV cache's {}",
                pos0 + tokens.len(),
                self.max_pos
            ));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= v) {
            return Err(format!("q36 prefill: token {t} outside the vocab {v}"));
        }
        let embed = &self.embed;
        let Graph::Moe(g) = &mut self.graph else {
            return Err("q36 prefill: not the 35B graph".into());
        };
        if g.prefill_cap() < chunk {
            g.prefill_alloc(chunk).map_err(gpu("q36 prefill_alloc"))?;
        }
        let mut next = None;
        let (n_chunks, pos) = q36_chunk_walk(tokens, chunk, pos0, |c, pos, last| {
            let mut x = vec![0f32; c.len() * h];
            for (row, &t) in x.chunks_mut(h).zip(c) {
                let t = t as usize;
                for (d, &b) in row.iter_mut().zip(&embed[t * h..(t + 1) * h]) {
                    *d = bf16_to_f32(b);
                }
            }
            let (tok, _) = g
                .prefill(head, &x, pos, last && want_token)
                .map_err(gpu("q36 prefill"))?;
            if last {
                next = tok;
            }
            Ok(())
        })?;
        *self.head_pos_mut(head)? = pos;
        Ok((next, n_chunks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_lengths_cover_and_respect_the_minimum() {
        for n in 3..200 {
            for chunk in [6, 7, 16, 64] {
                let c = chunk_lengths(n, chunk);
                assert_eq!(c.iter().sum::<usize>(), n, "n {n} chunk {chunk}");
                assert!(
                    c.iter().all(|&m| (q::GDN_PF_MIN_T..=chunk).contains(&m)),
                    "n {n} chunk {chunk}: {c:?}"
                );
            }
        }
        assert_eq!(chunk_lengths(130, 64), vec![64, 63, 3]);
        assert_eq!(chunk_lengths(129, 64), vec![64, 62, 3]);
        assert_eq!(chunk_lengths(131_072, 4096).len(), 32);
    }

    /// DRA-65: the 35B chunk boundaries at the prompt lengths the window measures, at the served
    /// chunk (128) and at a small one; every token in exactly one chunk, none longer than the cap.
    #[test]
    fn q36_chunks_at_the_boundaries() {
        let cap = q::q36::PREFILL_MAX_ROWS;
        assert_eq!(cap, 128);
        let cases: [(usize, &[usize]); 4] =
            [(1, &[1]), (127, &[127]), (128, &[128]), (129, &[128, 1])];
        for (n, want) in cases {
            assert_eq!(q36_chunk_lengths(n, cap), want, "n {n}");
        }
        assert_eq!(q36_chunk_lengths(2048, cap), vec![128; 16]);
        assert_eq!(q36_chunk_lengths(8192, cap), vec![128; 64]);
        assert_eq!(q36_chunk_lengths(8210, cap).len(), 65);
        assert_eq!(q36_chunk_lengths(8210, cap)[64], 18);
        assert!(q36_chunk_lengths(0, cap).is_empty());
        assert!(q36_chunk_lengths(5, 0).is_empty());
        for chunk in [1, 7, 64, cap] {
            for n in [1, chunk - 1, chunk, chunk + 1, 2048, 8192] {
                let c = q36_chunk_lengths(n, chunk);
                assert_eq!(c.iter().sum::<usize>(), n, "n {n} chunk {chunk}");
                assert_eq!(c.len(), n.div_ceil(chunk), "n {n} chunk {chunk}");
                assert!(
                    c.iter().all(|&m| (1..=chunk).contains(&m)),
                    "n {n} chunk {chunk}: {c:?}"
                );
            }
        }
    }

    #[test]
    fn q36_chunk_switch_parses_off_default_and_cap() {
        let cap = q::q36::PREFILL_MAX_ROWS;
        assert_eq!(parse_q36_chunk(None), Ok(cap));
        assert_eq!(parse_q36_chunk(Some("0")), Ok(0));
        assert_eq!(parse_q36_chunk(Some("16")), Ok(16));
        assert_eq!(parse_q36_chunk(Some("128")), Ok(128));
        assert!(parse_q36_chunk(Some("129")).is_err());
        assert!(parse_q36_chunk(Some("x")).is_err());
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 1.0, 2.0]), 1);
    }

    /// DRA-87 tie rule: the chip's two-stage ARGMAX (its CPU model, 2048 values per CTA over the
    /// 35B's 248320 logits: 122 CTAs, the last one padded) picks the same token as the host's
    /// greedy draw (`tensor::sample_argmax`, strict `>`) and this file's `argmax`: the largest
    /// logit, the lowest index on a tie, within a CTA, across CTAs and against -inf (the exact
    /// head's masked rows). NaN is outside the rule (the paths differ on it; finite logits only).
    /// A last-equal-index argmax (omega's ARGMAX mode-1 mutant) disagrees on the tie cases.
    #[test]
    fn chip_argmax_model_takes_the_host_rule() {
        const V: usize = 248_320;
        const SPAN: usize = 2048;
        let rand = |seed: u64| -> Vec<f32> {
            (0..V as u64)
                .map(|i| {
                    let mut x = seed ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    x ^= x >> 33;
                    ((x >> 40) % 64) as f32 * 0.25 - 8.0
                })
                .collect()
        };
        let mut cases: Vec<(&str, Vec<f32>)> =
            vec![("random, many ties", rand(1)), ("random", rand(2))];
        let mut v = vec![0.0f32; V];
        v[2047] = 3.0;
        v[2048] = 3.0;
        cases.push(("tie across a CTA edge", v));
        let mut v = vec![0.0f32; V];
        v[5] = 3.0;
        v[200_000] = 3.0;
        cases.push(("tie far apart", v));
        let mut v = vec![0.0f32; V];
        v[100] = 3.0;
        v[101] = 3.0;
        cases.push(("tie inside a CTA", v));
        cases.push(("all equal", vec![1.5f32; V]));
        let mut v = vec![f32::NEG_INFINITY; V];
        for i in [7usize, 4096, 131_072, V - 1] {
            v[i] = -2.0;
        }
        cases.push(("-inf masked, tied survivors", v));
        let mut v = vec![-1.0f32; V];
        v[V - 1] = 0.5;
        cases.push(("max in the padded last CTA", v));
        let last_equal = |v: &[f32]| -> u32 {
            let mut best = 0usize;
            for (i, &x) in v.iter().enumerate() {
                if x >= v[best] {
                    best = i;
                }
            }
            best as u32
        };
        for (what, v) in &cases {
            let host = crate::tensor::sample_argmax(v).0;
            assert_eq!(chip_argmax_model(v, SPAN), host, "{what}");
            assert_eq!(argmax(v), host, "{what}");
            if what.contains("tie") || what.contains("equal") {
                assert_ne!(last_equal(v), host, "{what}: the mutant must differ");
            }
        }
        // Small vocabularies: the same rule at a span of 8 (several CTAs, a padded tail).
        for n in [1usize, 7, 8, 9, 63, 64, 65] {
            for top in 0..n {
                let mut v: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
                v[top] = 9.0;
                v[n - 1 - (n - 1 - top) / 2] = 9.0;
                let host = crate::tensor::sample_argmax(&v).0;
                assert_eq!(chip_argmax_model(&v, 8), host, "n {n} top {top}");
            }
        }
    }
}
