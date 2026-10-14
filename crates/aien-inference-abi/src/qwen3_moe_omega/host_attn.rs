//! Host emulation of OWN-30B's causal attention for the drift split (W-L1I, 2026-10-14). New file.
//!
//! The decode attention (OFF path) and the whole-prompt prefill attention (ON path) share the K/V
//! rows (bf16(RoPE(RMSNorm(k) * k_norm)), bf16(v)) and the score scale (q' carries
//! (1/sqrt(128)) * log2(e), p = 2^(s - m)). They differ in (omega codegen_pattn.h / codegen_attn.h):
//!  * Q: the prefill rounds q' to bf16 (HMMA A operand); the decode keeps it f32.
//!  * P: the prefill packs p to bf16 for O += P V (F2FP) while l sums the f32 p; the decode keeps f32.
//!  * softmax order: the prefill runs an online softmax over blocks of 32 keys (kb = 0, 32, ..);
//!    the decode over chunks of 256 positions with a split combine.
//!
//! [`HostAttn`] computes one token's attention on the host with each of those switched on or off
//! independently ([`HostAttnMode`]), keeping its own K/V rows. Accumulation is f64 so that, with
//! every switch off, it is a near-exact reference: the difference to the chip's f32 decode kernel
//! is the emulator floor (measured per layer in [`HostAttn::chip_cmp`]).
use rayon::prelude::*;

use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};
use crate::tensor::{apply_rope, rmsnorm_heads_in_place};

/// Which of the prefill attention's roundings the emulation applies.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HostAttnMode {
    /// q' rounded to bf16 after the score scale (the prefill's HMMA A operand).
    pub q_bf16: bool,
    /// p rounded to bf16 in O += P V only; l keeps the f32 p (the prefill's F2FP pack).
    pub p_bf16: bool,
    /// Online softmax over 32-key blocks from key 0 (the prefill's order); off = one global max.
    pub online32: bool,
    /// TEST / NEGATIVE CONTROL ONLY: multiply the score scale by this (1.0 = none).
    pub qk_scale_mul: f32,
}

impl HostAttnMode {
    pub fn f32() -> Self {
        Self { q_bf16: false, p_bf16: false, online32: false, qk_scale_mul: 1.0 }
    }
}

fn bf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// Host K/V rows and the per-layer chip comparison of one run.
pub struct HostAttn {
    pub mode: HostAttnMode,
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    theta: f32,
    /// Per layer, per position: kv_heads x hd roped, bf16-rounded keys / bf16 values.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Per layer: (sum (host - chip)^2, sum chip^2) over every token's attention output, when the
    /// caller also ran the chip kernel on the same inputs.
    pub chip_cmp: Vec<(f64, f64)>,
}

impl HostAttn {
    pub fn new(mode: HostAttnMode, layers: usize, q_heads: usize, kv_heads: usize, hd: usize, theta: f32) -> Self {
        Self { mode, q_heads, kv_heads, hd, theta, k: vec![Vec::new(); layers], v: vec![Vec::new(); layers], chip_cmp: vec![(0.0, 0.0); layers] }
    }

    /// Positions stored for `layer`.
    pub fn len(&self, layer: usize) -> usize {
        self.k[layer].len() / (self.kv_heads * self.hd)
    }

    pub fn is_empty(&self) -> bool {
        self.k.iter().all(|k| k.is_empty())
    }

    /// relerr of the host attention output against the chip's, per layer (sqrt(d2 / r2)).
    pub fn chip_relerr(&self) -> Vec<f64> {
        self.chip_cmp.iter().map(|&(d, r)| if r > 0.0 { (d / r).sqrt() } else { 0.0 }).collect()
    }

    /// Adds one token's (host, chip) outputs to `layer`'s comparison.
    pub fn record_chip(&mut self, layer: usize, host: &[f32], chip: &[f32]) {
        let c = &mut self.chip_cmp[layer];
        for (a, b) in host.iter().zip(chip) {
            let d = *a as f64 - *b as f64;
            c.0 += d * d;
            c.1 += *b as f64 * *b as f64;
        }
    }

    /// Appends the K/V of position `pos` (must be the next one) from the raw q|k|v row and writes the
    /// attention output for that token (q heads x hd, unscaled) into `out`.
    pub fn attend(&mut self, layer: usize, pos: usize, qkv: &[f32], q_norm: &[f32], k_norm: &[f32], eps: f32, out: &mut [f32]) -> Result<(), String> {
        let (qh, kh, hd) = (self.q_heads, self.kv_heads, self.hd);
        let (qd, kvd) = (qh * hd, kh * hd);
        if qkv.len() != qd + 2 * kvd || out.len() != qd {
            return Err(format!("host attn: qkv {} out {}, want {} / {qd}", qkv.len(), out.len(), qd + 2 * kvd));
        }
        if self.len(layer) != pos {
            return Err(format!("host attn layer {layer}: step at {pos}, have {}", self.len(layer)));
        }
        let mut q = qkv[..qd].to_vec();
        let mut k = qkv[qd..qd + kvd].to_vec();
        rmsnorm_heads_in_place(&mut q, q_norm, hd, eps);
        rmsnorm_heads_in_place(&mut k, k_norm, hd, eps);
        apply_rope(&mut q, &mut k, pos, qh, kh, hd, self.theta);
        self.k[layer].extend(k.iter().map(|&x| bf(x)));
        self.v[layer].extend(qkv[qd + kvd..].iter().map(|&x| bf(x)));
        let scale = ((1.0f64 / (hd as f64).sqrt()) * std::f64::consts::LOG2_E) as f32 * self.mode.qk_scale_mul;
        for x in q.iter_mut() {
            *x *= scale;
            if self.mode.q_bf16 {
                *x = bf(*x);
            }
        }
        let (kc, vc, mode, n) = (&self.k[layer], &self.v[layer], self.mode, pos + 1);
        let group = qh / kh;
        out.par_chunks_mut(hd).enumerate().for_each(|(h, o)| {
            let g = h / group;
            let qv = &q[h * hd..(h + 1) * hd];
            let s: Vec<f32> = (0..n)
                .map(|j| {
                    let kr = &kc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
                    qv.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32
                })
                .collect();
            let vrow = |j: usize| &vc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
            let mut acc = vec![0.0f64; hd];
            let mut l = 0.0f64;
            let blk = if mode.online32 { 32 } else { n };
            let mut m = f32::NEG_INFINITY;
            let mut b0 = 0;
            while b0 < n {
                let b1 = (b0 + blk).min(n);
                let bm = s[b0..b1].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let m_new = m.max(bm);
                if m != f32::NEG_INFINITY && m_new != m {
                    let alpha = ((m - m_new) as f64).exp2();
                    l *= alpha;
                    acc.iter_mut().for_each(|a| *a *= alpha);
                }
                m = m_new;
                for (j, &sj) in s.iter().enumerate().take(b1).skip(b0) {
                    let p = ((sj - m) as f64).exp2() as f32;
                    l += p as f64;
                    let pv = if mode.p_bf16 { bf(p) } else { p } as f64;
                    for (a, vv) in acc.iter_mut().zip(vrow(j)) {
                        *a += pv * *vv as f64;
                    }
                }
                b0 = b1;
            }
            for (oo, a) in o.iter_mut().zip(&acc) {
                *oo = (*a / l) as f32;
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(seed: u64, n: usize, amp: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * amp
            })
            .collect()
    }

    fn run(mode: HostAttnMode, steps: usize) -> Vec<f32> {
        let (qh, kh, hd) = (8, 2, 32);
        let mut a = HostAttn::new(mode, 1, qh, kh, hd, 1e6);
        let (qn, kn) = (vec![1.0f32; hd], vec![1.0f32; hd]);
        let mut out = vec![0.0f32; qh * hd];
        let mut all = Vec::new();
        for p in 0..steps {
            let qkv = row(p as u64 + 1, (qh + 2 * kh) * hd, 8.0);
            a.attend(0, p, &qkv, &qn, &kn, 1e-6, &mut out).unwrap();
            all.extend_from_slice(&out);
        }
        all
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut r) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            d += (*x as f64 - *y as f64).powi(2);
            r += (*y as f64).powi(2);
        }
        (d / r).sqrt()
    }

    #[test]
    fn online_equals_global_in_f32() {
        let g = run(HostAttnMode::f32(), 100);
        let o = run(HostAttnMode { online32: true, ..HostAttnMode::f32() }, 100);
        assert!(rel(&o, &g) < 1e-6, "{}", rel(&o, &g));
    }

    #[test]
    fn first_token_is_its_value() {
        // One key: softmax weight 1 exactly, so the output is bf16(v) whatever Q/P rounding does.
        let o = run(HostAttnMode { q_bf16: true, p_bf16: true, online32: true, qk_scale_mul: 1.0 }, 1);
        let qkv = row(1, 12 * 32, 8.0);
        for h in 0..8 {
            for d in 0..32 {
                assert_eq!(o[h * 32 + d], bf(qkv[(8 + 2) * 32 + (h / 4) * 32 + d]));
            }
        }
    }

    #[test]
    fn bf16_switches_move_the_output_by_bf16_size() {
        let g = run(HostAttnMode::f32(), 100);
        for m in [HostAttnMode { q_bf16: true, ..HostAttnMode::f32() }, HostAttnMode { p_bf16: true, ..HostAttnMode::f32() }] {
            let e = rel(&run(m, 100), &g);
            assert!(e > 1e-5 && e < 2e-2, "{m:?}: {e}");
        }
        let mutant = rel(&run(HostAttnMode { qk_scale_mul: 1.05, ..HostAttnMode::f32() }, 100), &g);
        assert!(mutant > 1e-2, "qk scale mutant {mutant}");
    }
}
