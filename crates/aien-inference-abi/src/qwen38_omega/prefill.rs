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

use super::{gpu, Qwen38Omega, EPS_L2};
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
        let t0 = Instant::now();
        if self.w4_only() {
            // W4-only (L125): no FP8 copy for the batched path; the prompt goes through the decode graph token by token.
            let mut next = 0;
            for &t in tokens {
                next = self.step(t)?.0;
            }
            let st = PrefillStats {
                tokens: tokens.len(),
                wall_ns: ns(t0),
                ..PrefillStats::default()
            };
            return Ok((next, self.logits()?, st));
        }
        let (last, mut st) = self.prefill_rows(tokens, chunk, false)?;
        let t = Instant::now();
        let logits = self.head_logits(&last)?;
        let next = argmax(&logits);
        st.head_ns = ns(t);
        st.wall_ns = ns(t0);
        Ok((next, logits, st))
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
            .as_ref()
            .ok_or("head_logits: W4-only model (no bf16 lm_head); use step and logits")?
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
            let (Some(lt_mix), Some(lt_o), Some(lt_gate_up), Some(lt_down)) =
                (&lt.mix, &lt.o, &lt.gate_up, &lt.down)
            else {
                return Err(
                    "qwen38 prefill: W4-only model (no FP8 copy for the batched path)".into(),
                );
            };
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
            pf.gemm(lt_mix, mix_n, h, Layout::Plain, mix_va, m, st)
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
            pf.gemm(lt_o, h, mo, Layout::Plain, out_va, m, st)
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
            pf.gemm(lt_gate_up, 2 * inter, h, Layout::SwigluNat, act_va, m, st)
                .map_err(|e| format!("layer {l} gate|up gemm x{m}: {e}"))?;
            pf.quant(inter, QuantIn::Bf16, act_va, m, st)
                .map_err(|e| format!("layer {l} quant act: {e}"))?;
            pf.gemm(lt_down, h, inter, Layout::Plain, out_va, m, st)
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

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 1.0, 2.0]), 1);
    }
}
