//! Dense f32 matrix products on the host: one register-blocked kernel, tasks over tiles of C.

use rayon::prelude::*;

const MR: usize = 4;
const NR: usize = 16;
/// Rows and columns of C one task owns, and the depth packed at a time.
const MB: usize = 64;
const NB: usize = 256;
const KC: usize = 512;

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: every task writes only its own rectangle of C (see `gemm`), so sharing the base
// pointer across threads never makes two writers touch one element.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// `t += Ap Bp` over `kc` steps: Ap `[kc][4]`, Bp `[kc][16]`, both packed.
#[cfg(target_arch = "aarch64")]
fn kernel(kc: usize, ap: &[f32], bp: &[f32], t: &mut [[f32; NR]; MR]) {
    use std::arch::aarch64::*;
    assert!(ap.len() >= kc * MR && bp.len() >= kc * NR);
    // SAFETY: NEON is part of the aarch64 baseline; every load reads inside `ap[..kc * 4]` or
    // `bp[..kc * 16]` (checked above) and every store writes inside `t`.
    unsafe {
        let mut c = [vdupq_n_f32(0.0); 16];
        let (pa, pb) = (ap.as_ptr(), bp.as_ptr());
        for kk in 0..kc {
            let a = vld1q_f32(pa.add(kk * MR));
            let b0 = vld1q_f32(pb.add(kk * NR));
            let b1 = vld1q_f32(pb.add(kk * NR + 4));
            let b2 = vld1q_f32(pb.add(kk * NR + 8));
            let b3 = vld1q_f32(pb.add(kk * NR + 12));
            macro_rules! row {
                ($r:literal) => {
                    c[$r * 4] = vfmaq_laneq_f32::<$r>(c[$r * 4], b0, a);
                    c[$r * 4 + 1] = vfmaq_laneq_f32::<$r>(c[$r * 4 + 1], b1, a);
                    c[$r * 4 + 2] = vfmaq_laneq_f32::<$r>(c[$r * 4 + 2], b2, a);
                    c[$r * 4 + 3] = vfmaq_laneq_f32::<$r>(c[$r * 4 + 3], b3, a);
                };
            }
            row!(0);
            row!(1);
            row!(2);
            row!(3);
        }
        for r in 0..MR {
            for q in 0..4 {
                vst1q_f32(t[r].as_mut_ptr().add(q * 4), c[r * 4 + q]);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn kernel(kc: usize, ap: &[f32], bp: &[f32], t: &mut [[f32; NR]; MR]) {
    *t = [[0f32; NR]; MR];
    for kk in 0..kc {
        for r in 0..MR {
            let av = ap[kk * MR + r];
            for j in 0..NR {
                t[r][j] = av.mul_add(bp[kk * NR + j], t[r][j]);
            }
        }
    }
}

/// A `[k][n]` right-hand side packed once: `[ceil(n / 16)][k][16]`, columns past `n` zero.
pub struct Packed {
    pub k: usize,
    pub n: usize,
    data: Vec<f32>,
}

impl Packed {
    pub fn new(b: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(b.len(), k * n);
        let panels = n.div_ceil(NR);
        let mut data = vec![0f32; panels * k * NR];
        data.par_chunks_mut(k * NR)
            .enumerate()
            .for_each(|(p, dst)| {
                let (c0, w) = (p * NR, NR.min(n - p * NR));
                for kk in 0..k {
                    dst[kk * NR..kk * NR + w].copy_from_slice(&b[kk * n + c0..kk * n + c0 + w]);
                }
            });
        Self { k, n, data }
    }

    /// Back to the plain `[k][n]`.
    pub fn unpack(&self) -> Vec<f32> {
        let (k, n) = (self.k, self.n);
        let mut out = vec![0f32; k * n];
        out.par_chunks_mut(n).enumerate().for_each(|(kk, row)| {
            for (j, o) in row.iter_mut().enumerate() {
                *o = self.data[((j / NR) * k + kk) * NR + j % NR];
            }
        });
        out
    }
}

/// `C = A B` (`acc` false) or `C += A B`; A `[m][k]`, B `[k][n]`, C `[m][n]`, all row-major.
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(b.len(), k * n, "gemm: B is not [{k}][{n}]");
    gemm_any(a, Rhs::Plain(b), c, m, k, n, acc);
}

/// [`gemm`] with a packed B.
pub fn gemm_p(a: &[f32], b: &Packed, c: &mut [f32], m: usize, acc: bool) {
    gemm_any(a, Rhs::Packed(b), c, m, b.k, b.n, acc);
}

/// `A B` as a new `[m][n]` with a packed B.
pub fn matmul_p(a: &[f32], b: &Packed, m: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * b.n];
    gemm_p(a, b, &mut c, m, false);
    c
}

#[derive(Clone, Copy)]
enum Rhs<'a> {
    Plain(&'a [f32]),
    Packed(&'a Packed),
}

fn gemm_any(a: &[f32], b: Rhs, c: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    assert_eq!(a.len(), m * k, "gemm: A is not [{m}][{k}]");
    assert_eq!(c.len(), m * n, "gemm: C is not [{m}][{n}]");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !acc {
            c.fill(0.0);
        }
        return;
    }
    let (mb, nb) = (m.div_ceil(MB), n.div_ceil(NB));
    let out = OutPtr(c.as_mut_ptr());
    (0..mb * nb).into_par_iter().for_each_init(
        || {
            let bl = if matches!(b, Rhs::Plain(_)) {
                KC * NB
            } else {
                0
            };
            (vec![0f32; MB * KC], vec![0f32; bl])
        },
        |(apack, bpack), task| {
            let out = out;
            let (bi, bj) = (task / nb, task % nb);
            let (i0, j0) = (bi * MB, bj * NB);
            let (rows, cols) = ((i0 + MB).min(m) - i0, (j0 + NB).min(n) - j0);
            let (ap_n, bp_n) = (rows.div_ceil(MR), cols.div_ceil(NR));
            let mut k0 = 0;
            while k0 < k {
                let kc = KC.min(k - k0);
                // A rows i0.. as [panel][kc][4], B columns j0.. as [panel][kc][16], zero padded.
                for p in 0..ap_n {
                    let dst = &mut apack[p * kc * MR..(p + 1) * kc * MR];
                    for r in 0..MR {
                        let i = p * MR + r;
                        if i < rows {
                            let src = &a[(i0 + i) * k + k0..(i0 + i) * k + k0 + kc];
                            for (kk, &v) in src.iter().enumerate() {
                                dst[kk * MR + r] = v;
                            }
                        } else {
                            for kk in 0..kc {
                                dst[kk * MR + r] = 0.0;
                            }
                        }
                    }
                }
                if let Rhs::Plain(b) = b {
                    for p in 0..bp_n {
                        let dst = &mut bpack[p * kc * NR..(p + 1) * kc * NR];
                        let (c0, w) = (p * NR, NR.min(cols - p * NR));
                        for kk in 0..kc {
                            let src = &b[(k0 + kk) * n + j0 + c0..(k0 + kk) * n + j0 + c0 + w];
                            let d = &mut dst[kk * NR..(kk + 1) * NR];
                            d[..w].copy_from_slice(src);
                            d[w..].iter_mut().for_each(|v| *v = 0.0);
                        }
                    }
                }
                let add = acc || k0 > 0;
                let mut t = [[0f32; NR]; MR];
                for pb in 0..bp_n {
                    let bp = match b {
                        Rhs::Plain(_) => &bpack[pb * kc * NR..(pb + 1) * kc * NR],
                        Rhs::Packed(pk) => {
                            let o = ((j0 / NR + pb) * k + k0) * NR;
                            &pk.data[o..o + kc * NR]
                        }
                    };
                    let w = NR.min(cols - pb * NR);
                    for pa in 0..ap_n {
                        kernel(kc, &apack[pa * kc * MR..(pa + 1) * kc * MR], bp, &mut t);
                        for r in 0..MR.min(rows - pa * MR) {
                            let (i, j) = (i0 + pa * MR + r, j0 + pb * NR);
                            // SAFETY: row i < i0 + rows <= m and columns j..j + w <= j0 + cols
                            // <= n lie inside C and inside this task's rectangle
                            // [i0, i0 + rows) x [j0, j0 + cols), which no other task writes.
                            let row =
                                unsafe { std::slice::from_raw_parts_mut(out.0.add(i * n + j), w) };
                            if add {
                                row.iter_mut().zip(&t[r]).for_each(|(o, v)| *o += v);
                            } else {
                                row.copy_from_slice(&t[r][..w]);
                            }
                        }
                    }
                }
                k0 += kc;
            }
        },
    );
}

/// `A B` as a new `[m][n]`.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    gemm(a, b, &mut c, m, k, n, false);
    c
}

/// `[r][c]` to `[c][r]`.
pub fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(x.len(), r * c);
    let mut out = vec![0f32; r * c];
    if r == 0 || c == 0 {
        return out;
    }
    out.par_chunks_mut(r).enumerate().for_each(|(j, row)| {
        for (i, o) in row.iter_mut().enumerate() {
            *o = x[i * c + j];
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_triple_loop_on_ragged_shapes() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (5, 7, 3),
            (67, 33, 270),
            (4, 16, 16),
            (130, 9, 17),
            (70, 1100, 300),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
                .collect();
            let mut c = vec![1f32; m * n];
            gemm(&a, &b, &mut c, m, k, n, true);
            for i in 0..m {
                for j in 0..n {
                    let want: f64 = 1.0
                        + (0..k)
                            .map(|kk| a[i * k + kk] as f64 * b[kk * n + j] as f64)
                            .sum::<f64>();
                    assert!(
                        (c[i * n + j] as f64 - want).abs() < 1e-4,
                        "({m},{k},{n}) at {i},{j}"
                    );
                }
            }
            let mut c2 = vec![1f32; m * n];
            gemm_p(&a, &Packed::new(&b, k, n), &mut c2, m, true);
            assert_eq!(c, c2, "packed B differs ({m},{k},{n})");
            assert_eq!(Packed::new(&b, k, n).unpack(), b);
            let t = transpose(&a, m, k);
            assert_eq!(transpose(&t, k, m), a);
        }
    }
}
