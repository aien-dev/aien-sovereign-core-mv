//! Exact two-pass lm_head (HUM lever, ledger L314).
//!
//! The 35B head is bf16, `[248,320][2,048]`: 1.017 GB read per token. Pass 1 multiplies a
//! cheap copy `w~` (W4 or FP8, one bf16 scale per 128 columns) by the hidden vector `h`; pass 2
//! recomputes the exact bf16 logits only for the rows that can still be the argmax.
//!
//! Why the token cannot change. Let `L_i` be today's logit (bf16 row times `h`, f32 sums in any
//! order) and `s_i` the screen logit. With `D = w - w~` per 128-column group `g`:
//!   `|L_i - s_i| <= gamma |w_i||h| + sum_g |D_ig||h_g| + gamma |w~_i||h| <= B_i`
//! (Cauchy-Schwarz per group; `gamma = 2048 u / (1 - 2048 u)`, `u = 2^-24`, bounds any f32 dot
//! of length 2048 whatever its order, Higham 3.1). Let `i*` be the full argmax (first on ties)
//! and `tau = max_j (s_j - B_j)`. Every `L_j >= s_j - B_j`, so `L_i* >= tau`, so
//! `s_i* + B_i* >= L_i* >= tau`: `i*` is in `C = { i : s_i + B_i >= tau }`. The recheck uses the
//! same `L_i` bits and scans `C` in ascending order with the same strict `>`, so it returns `i*`.
//!
//! Every stored bound term (group norms, row norms) is rounded UP to bf16, so the stored bytes
//! are themselves an upper bound; the whole bound is then widened by `2^-20` for the rounding of
//! the few f64 operations that combine it.

use rayon::prelude::*;

use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3fn, fp8_table};
use crate::qwen38::{dot, Matrix};

/// Columns per scale and per bound group.
pub const GROUP: usize = 128;

/// The cheap copy pass 1 reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenKind {
    /// int4 in [-8, 7], scale = bf16(absmax / 7), round to nearest.
    W4,
    /// e4m3, scale = bf16(absmax / 448), round to nearest even.
    Fp8,
    /// e4m3 with one scale per 128 x 128 block, scale = bf16 rounded UP (absmax / 448): the layout
    /// omega's FP8 GEMV reads (`weight_scale_inv` [n/128][k/128]), so the chip screen reuses it.
    Fp8Block,
}

/// Which upper bound on `|L_i - s_i|` builds the candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundKind {
    /// `|D_i| |h|`: one stored value per row.
    Row,
    /// `sum_g |D_ig|_2 |h_g|_2`: 16 stored values per row.
    Group,
    /// `sum_g max|D_ig| |h_g|_1`: 16 stored values per row.
    GroupLinf,
    /// The smaller of `Group` and `GroupLinf` per row (both stored).
    Min,
}

/// Roundings per term the chip screen may add: k products and sums plus the 16 block partials.
pub const CHIP_GAMMA_LEN: usize = 2100;

/// What the chip's exact head uploads for a bf16 `head`: the e4m3 codes `[n][k]`, the block
/// scales as bf16 words `[n / 128][k / 128]` and the Row coefficients `[n]`
/// ([`Screen::row_coef`] at [`CHIP_GAMMA_LEN`]). Refuses when the codes do not reproduce the
/// screen the bound was computed for, bit for bit.
pub fn chip_screen(head: &Matrix) -> Result<(Vec<u8>, Vec<u16>, Vec<f32>), String> {
    let scales = fp8_block_scales(head);
    let codes = fp8_block_codes(head, &scales);
    let screen = Screen::build(head, ScreenKind::Fp8Block)?;
    let bad = screen.codes_mismatch(&codes, &scales);
    if bad != 0 {
        return Err(format!("exact head: {bad} rows of codes do not reproduce the screen"));
    }
    let coef = screen.row_coef(CHIP_GAMMA_LEN);
    Ok((codes, scales.iter().map(|&s| f32_to_bf16(s)).collect(), coef))
}

impl ScreenKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "w4" => Some(Self::W4),
            "fp8" => Some(Self::Fp8),
            "fp8b" => Some(Self::Fp8Block),
            _ => None,
        }
    }

    /// Bytes of codes + bf16 scales for an `[n][k]` head.
    pub fn bytes(self, n: usize, k: usize) -> u64 {
        let codes = match self {
            Self::W4 => n * k / 2,
            Self::Fp8 => n * k,
            Self::Fp8Block => return (n * k + (n / GROUP) * (k / GROUP) * 2) as u64,
        };
        (codes + n * (k / GROUP) * 2) as u64
    }
}

impl BoundKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "row" => Some(Self::Row),
            "group" => Some(Self::Group),
            "linf" => Some(Self::GroupLinf),
            "min" => Some(Self::Min),
            _ => None,
        }
    }

    /// Bytes of stored bound terms (bf16) for `n` rows of `k`, the fp margin's row norm included.
    pub fn bytes(self, n: usize, k: usize) -> u64 {
        let g = k / GROUP;
        let terms = match self {
            Self::Row => n,
            Self::Group | Self::GroupLinf => n * g,
            Self::Min => 2 * n * g,
        };
        ((terms + n) * 2) as u64
    }
}

/// `2048 u / (1 - 2048 u)` for length `k`: any f32 dot's error over `sum |a_i b_i|`.
pub fn gamma(k: usize) -> f64 {
    let nu = k as f64 * 2f64.powi(-24);
    nu / (1.0 - nu)
}

/// The smallest bf16 value `>= x` (`x >= 0`), as f32.
pub fn bf16_up(x: f64) -> f32 {
    let mut v = x as f32;
    if (v as f64) < x {
        v = f32::from_bits(v.to_bits() + 1);
    }
    let mut b = f32_to_bf16(v);
    if bf16_to_f32(b) < v {
        b += 1;
    }
    bf16_to_f32(b)
}

/// Pass 1's copy of a bf16 head and the stored bound terms, built once at load.
pub struct Screen {
    pub kind: ScreenKind,
    pub n: usize,
    pub k: usize,
    /// `w~` as f32 `[n][k]` (host; the chip reads codes + scales).
    deq: Vec<f32>,
    /// `|D_ig|_2` rounded up, `[n][k / GROUP]`.
    dnorm: Vec<f32>,
    /// `max |D_ig|` rounded up, `[n][k / GROUP]`.
    dmax: Vec<f32>,
    /// `|D_i|_2` rounded up, `[n]`.
    drow: Vec<f32>,
    /// `|w_i|_2 + |w~_i|_2` rounded up, `[n]` (the f32 rounding margin).
    rnorm: Vec<f32>,
}

/// [`ScreenKind::Fp8Block`] scales, `[n / 128][k / 128]`: bf16 rounded UP (absmax / 448), so
/// `|w / scale| <= 448` always (no e4m3 saturation). Every value is a bf16, so it uploads as is.
pub fn fp8_block_scales(head: &Matrix) -> Vec<f32> {
    let (n, g) = (head.n(), head.k() / GROUP);
    (0..(n / GROUP) * g)
        .into_par_iter()
        .map(|b| {
            let (br, gi) = (b / g, b % g);
            let amax = (br * GROUP..(br + 1) * GROUP)
                .flat_map(|r| head.bf16_row(r).unwrap()[gi * GROUP..(gi + 1) * GROUP].iter())
                .map(|&v| bf16_to_f32(v).abs())
                .fold(0f32, f32::max);
            bf16_up(amax as f64 / 448.0)
        })
        .collect()
}

/// e4m3 codes of a bf16 `head` under `scales` ([`fp8_block_scales`]), `[n][k]`: the bytes the
/// chip screen reads. `lut[code] * scale` is exactly the screen's `w~`.
pub fn fp8_block_codes(head: &Matrix, scales: &[f32]) -> Vec<u8> {
    let (n, k) = (head.n(), head.k());
    let g = k / GROUP;
    let mut out = vec![0u8; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(r, o)| {
        for (c, (d, &b)) in o.iter_mut().zip(head.bf16_row(r).unwrap()).enumerate() {
            let s = scales[(r / GROUP) * g + c / GROUP];
            *d = if s == 0.0 { 0 } else { f32_to_fp8_e4m3fn(bf16_to_f32(b) / s) };
        }
    });
    out
}

impl Screen {
    /// The chip's Row-bound coefficient per row: `c_i >= |D_i| + gamma(len) (|w_i| + |w~_i|)`
    /// rounded up to f32 and widened `2^-20`; the chip bound is `c_i |h|` (its own margins on top).
    /// `len` covers the chip screen's roundings per term (k products + the block partial sums).
    pub fn row_coef(&self, len: usize) -> Vec<f32> {
        let gm = gamma(len);
        self.drow
            .iter()
            .zip(&self.rnorm)
            .map(|(&d, &r)| {
                let x = (d as f64 + gm * r as f64) * (1.0 + 2f64.powi(-20));
                let mut v = x as f32;
                if (v as f64) < x {
                    v = f32::from_bits(v.to_bits() + 1);
                }
                v
            })
            .collect()
    }

    /// Rows whose exported e4m3 `codes` under block `scales` do not reproduce `w~` bit for bit.
    pub fn codes_mismatch(&self, codes: &[u8], scales: &[f32]) -> usize {
        let (k, g) = (self.k, self.k / GROUP);
        let lut = fp8_table();
        (0..self.n)
            .into_par_iter()
            .filter(|&r| {
                (0..k).any(|c| {
                    let s = scales[(r / GROUP) * g + c / GROUP];
                    (lut[codes[r * k + c] as usize] * s).to_bits() != self.deq[r * k + c].to_bits()
                })
            })
            .count()
    }
    /// Quantizes every row of the bf16 `head` (`k` a multiple of [`GROUP`]).
    pub fn build(head: &Matrix, kind: ScreenKind) -> Result<Self, String> {
        let (n, k) = (head.n(), head.k());
        if head.bf16_row(0).is_none() || k % GROUP != 0 {
            return Err(format!("screen: needs a bf16 head with k % {GROUP} == 0"));
        }
        let g = k / GROUP;
        let lut = fp8_table();
        let mut deq = vec![0f32; n * k];
        let mut dnorm = vec![0f32; n * g];
        let mut dmax = vec![0f32; n * g];
        let mut drow = vec![0f32; n];
        let mut rnorm = vec![0f32; n];
        if kind == ScreenKind::Fp8Block && n % GROUP != 0 {
            return Err(format!("screen: fp8b needs n % {GROUP} == 0"));
        }
        let block = if kind == ScreenKind::Fp8Block { fp8_block_scales(head) } else { Vec::new() };
        deq.par_chunks_mut(k)
            .zip(dnorm.par_chunks_mut(g))
            .zip(dmax.par_chunks_mut(g))
            .zip(drow.par_iter_mut().zip(rnorm.par_iter_mut()))
            .enumerate()
            .for_each(|(r, (((out, dn), dm), (dr, rn)))| {
                let row = head.bf16_row(r).unwrap();
                let (mut d2, mut w2, mut q2) = (0f64, 0f64, 0f64);
                for gi in 0..g {
                    let src = &row[gi * GROUP..(gi + 1) * GROUP];
                    let dst = &mut out[gi * GROUP..(gi + 1) * GROUP];
                    let amax = src.iter().map(|&b| bf16_to_f32(b).abs()).fold(0f32, f32::max);
                    let scale = match kind {
                        ScreenKind::W4 => bf16_to_f32(f32_to_bf16(amax / 7.0)),
                        ScreenKind::Fp8 => bf16_to_f32(f32_to_bf16(amax / 448.0)),
                        ScreenKind::Fp8Block => block[(r / GROUP) * g + gi],
                    };
                    let (mut gd2, mut gmax) = (0f64, 0f64);
                    for (o, &b) in dst.iter_mut().zip(src) {
                        let w = bf16_to_f32(b);
                        *o = if scale == 0.0 {
                            0.0
                        } else {
                            match kind {
                                ScreenKind::W4 => (w / scale).round().clamp(-8.0, 7.0) * scale,
                                ScreenKind::Fp8 | ScreenKind::Fp8Block => lut[f32_to_fp8_e4m3fn(w / scale) as usize] * scale,
                            }
                        };
                        let d = w as f64 - *o as f64;
                        gd2 += d * d;
                        gmax = gmax.max(d.abs());
                        w2 += w as f64 * w as f64;
                        q2 += *o as f64 * *o as f64;
                    }
                    d2 += gd2;
                    dn[gi] = bf16_up(gd2.sqrt() * (1.0 + 1e-12));
                    dm[gi] = bf16_up(gmax);
                }
                *dr = bf16_up(d2.sqrt() * (1.0 + 1e-12));
                *rn = bf16_up((w2.sqrt() + q2.sqrt()) * (1.0 + 1e-12));
            });
        Ok(Self { kind, n, k, deq, dnorm, dmax, drow, rnorm })
    }

    /// Pass 1: `s_i = w~_i . h` for every row.
    pub fn logits(&self, h: &[f32]) -> Vec<f32> {
        self.deq.par_chunks(self.k).map(|row| dot(row, h)).collect()
    }

    /// `B_i` for every row, times `scale` (1 = the proven bound; below 1 = a mutant).
    pub fn bounds(&self, h: &[f32], kind: BoundKind, scale: f64) -> Vec<f64> {
        let g = self.k / GROUP;
        let widen = 1.0 + 1e-12;
        let h2: Vec<f64> = h
            .chunks_exact(GROUP)
            .map(|c| c.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt() * widen)
            .collect();
        let h1: Vec<f64> = h
            .chunks_exact(GROUP)
            .map(|c| c.iter().map(|&v| (v as f64).abs()).sum::<f64>() * widen)
            .collect();
        let hn = h2.iter().map(|v| v * v).sum::<f64>().sqrt() * widen;
        let gm = gamma(self.k);
        (0..self.n)
            .into_par_iter()
            .map(|r| {
                let group = || (0..g).map(|i| self.dnorm[r * g + i] as f64 * h2[i]).sum::<f64>();
                let linf = || (0..g).map(|i| self.dmax[r * g + i] as f64 * h1[i]).sum::<f64>();
                let quant = match kind {
                    BoundKind::Row => self.drow[r] as f64 * hn,
                    BoundKind::Group => group(),
                    BoundKind::GroupLinf => linf(),
                    BoundKind::Min => group().min(linf()),
                };
                (quant + gm * self.rnorm[r] as f64 * hn) * scale * (1.0 + 2f64.powi(-20))
            })
            .collect()
    }
}

/// Candidate rows, ascending: `{ i : s_i + B_i >= max_j (s_j - B_j) }`.
pub fn candidates(s: &[f32], b: &[f64]) -> Vec<u32> {
    candidates_from(s, b, f64::NEG_INFINITY)
}

/// [`candidates`] given a lower bound on the winner's exact logit (the exact logit of any row,
/// rechecked first): `tau = max(lower, max_j (s_j - B_j))`. Still exact: `L_i* >= tau`.
pub fn candidates_from(s: &[f32], b: &[f64], lower: f64) -> Vec<u32> {
    let tau = s
        .iter()
        .zip(b)
        .map(|(&v, &e)| v as f64 - e)
        .fold(lower, f64::max);
    s.iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (&v, &e))| v as f64 + e >= tau)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Candidate rows (ascending) that hold the exact top `kk` (the door's top-k mask before it
/// samples): `tau = kk`-th largest `s_j - B_j`, at least `lower`. `kk` rows have `L >= tau`, so
/// `L_(kk) >= tau`; a row outside has `L <= s + B < tau`, strictly below every kept logit, so the
/// top-k set and its order (logit desc, row asc) are those of the full head.
pub fn candidates_topk(s: &[f32], b: &[f64], kk: usize, lower: f64) -> Vec<u32> {
    let mut low: Vec<f64> = s.iter().zip(b).map(|(&v, &e)| v as f64 - e).collect();
    let kk = kk.clamp(1, low.len());
    let (_, kth, _) = low.select_nth_unstable_by(kk - 1, |a, z| z.total_cmp(a));
    let tau = kth.max(lower);
    s.iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (&v, &e))| v as f64 + e >= tau)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Pass 2: today's exact logit for each candidate (same widen + [`dot`] as [`Matrix::matmul`]),
/// argmax in ascending row order, first on ties. Returns `(row, logit)`.
pub fn recheck(head: &Matrix, h: &[f32], cands: &[u32]) -> (usize, f32) {
    let exact: Vec<f32> = cands
        .par_iter()
        .map(|&r| {
            let row: Vec<f32> = head.bf16_row(r as usize).unwrap().iter().map(|&b| bf16_to_f32(b)).collect();
            dot(&row, h)
        })
        .collect();
    let mut best = 0;
    for (i, &v) in exact.iter().enumerate() {
        if v > exact[best] {
            best = i;
        }
    }
    (cands[best] as usize, exact[best])
}

/// The two-pass head end to end: `(argmax, |C|)`.
pub fn argmax(head: &Matrix, screen: &Screen, h: &[f32], kind: BoundKind, scale: f64) -> (usize, usize) {
    assert_eq!(h.len(), screen.k, "exact_head::argmax: h must be one hidden vector of k values");
    let s = screen.logits(h);
    let b = screen.bounds(h, kind, scale);
    let c = candidates(&s, &b);
    (recheck(head, h, &c).0, c.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38::argmax as full_argmax;

    fn prand(seed: u64, i: usize) -> f32 {
        let mut x = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn head(n: usize, k: usize, seed: u64) -> Matrix {
        let w = (0..n * k).map(|i| f32_to_bf16(prand(seed, i) * 0.05)).collect();
        Matrix::Bf16 { n, k, w }
    }

    #[test]
    fn bf16_up_never_rounds_down() {
        for &x in &[0.0, 1e-30, 0.1, 1.0 / 3.0, 7.0, 123.456, 1e30] {
            assert!(bf16_up(x) as f64 >= x, "{x}");
        }
        assert_eq!(bf16_up(1.0), 1.0);
    }

    #[test]
    fn two_pass_equals_full_argmax_on_random_heads() {
        let (n, k) = (4096, 256);
        for kind in [ScreenKind::W4, ScreenKind::Fp8, ScreenKind::Fp8Block] {
            let m = head(n, k, 7);
            let scr = Screen::build(&m, kind).unwrap();
            for t in 0..40 {
                // A hidden vector with a planted winner so the top gap is sometimes tiny.
                let mut h: Vec<f32> = (0..k).map(|i| prand(100 + t, i) * 3.0).collect();
                if t % 2 == 0 {
                    let row: Vec<f32> = m.bf16_row(t as usize * 17).unwrap().iter().map(|&b| bf16_to_f32(b)).collect();
                    h.iter_mut().zip(&row).for_each(|(a, b)| *a += b * 40.0);
                }
                let full = full_argmax(&m.matmul(&h, 1));
                for bk in [BoundKind::Row, BoundKind::Group, BoundKind::GroupLinf, BoundKind::Min] {
                    let (got, c) = argmax(&m, &scr, &h, bk, 1.0);
                    assert_eq!(got, full, "{kind:?} {bk:?} t {t}");
                    assert!(c >= 1 && c <= n);
                }
            }
        }
    }

    #[test]
    fn topk_candidates_hold_the_exact_top_k() {
        let (n, k, kk) = (4096, 256, 20);
        let m = head(n, k, 9);
        let scr = Screen::build(&m, ScreenKind::W4).unwrap();
        for t in 0..20 {
            let h: Vec<f32> = (0..k).map(|i| prand(900 + t, i) * 2.0).collect();
            let l = m.matmul(&h, 1);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
            let c = candidates_topk(&scr.logits(&h), &scr.bounds(&h, BoundKind::Min, 1.0), kk, f64::NEG_INFINITY);
            let mut sub: Vec<usize> = c.iter().map(|&r| r as usize).collect();
            sub.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
            assert_eq!(&sub[..kk], &order[..kk], "t {t}");
        }
    }

    #[test]
    fn ties_go_to_the_first_row_like_the_full_argmax() {
        let (n, k) = (300, 128);
        let mut w: Vec<u16> = (0..n * k).map(|i| f32_to_bf16(prand(3, i) * 0.01)).collect();
        // Rows 40 and 200 identical and largest along h = ones.
        for c in 0..k {
            w[40 * k + c] = f32_to_bf16(0.5);
            w[200 * k + c] = f32_to_bf16(0.5);
        }
        let m = Matrix::Bf16 { n, k, w };
        let h = vec![1f32; k];
        let scr = Screen::build(&m, ScreenKind::W4).unwrap();
        assert_eq!(full_argmax(&m.matmul(&h, 1)), 40);
        assert_eq!(argmax(&m, &scr, &h, BoundKind::Group, 1.0).0, 40);
    }

    #[test]
    fn a_zero_bound_mutant_can_lose_the_argmax() {
        // The screen alone (bound x 0) must miss the exact winner on some input, or this test
        // could not tell a working bound from none.
        let (n, k) = (4096, 256);
        let m = head(n, k, 11);
        let scr = Screen::build(&m, ScreenKind::W4).unwrap();
        let missed = (0..200).any(|t| {
            let h: Vec<f32> = (0..k).map(|i| prand(500 + t, i)).collect();
            argmax(&m, &scr, &h, BoundKind::Group, 0.0).0 != full_argmax(&m.matmul(&h, 1))
        });
        assert!(missed);
    }
}
