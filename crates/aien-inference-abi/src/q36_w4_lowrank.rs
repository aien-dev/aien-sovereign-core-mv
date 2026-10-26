//! Low-rank correction of the 4-bit expert down projections (W4Correct, docs/inference/q36/w4-gate.md "Low-rank arm").
//!
//! Arm m-dn (FP8 gate|up, nv16 down) missed the frozen rule only on confident flips of the held-out prompts. Here each
//! down matrix (routed experts and the shared one, every layer) gets a rank-r term C = U V (bf16, U `[hidden][r]`,
//! V `[r][inter]`) fitted to its rounding error E = W_fp8 - Q_nv16, so the host computes `Q a + U (V a)`.
//! Fit: plain = the best rank-r approximation of E (Frobenius); calibrated = the best in the output metric
//! tr((E - C) H (E - C)^T), H = sum g^2 a a^T the expert's down inputs on calibration text (q36_w4_gptq::Calib::add_dn), damped
//! like the GPTQ arm: with H + damp = L L^T, C = E L R R^T L^-1, R the top r eigenvectors of L^T E^T E L.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};

/// Sidecar magic (one record per (layer, slot): U then V, bf16 little endian).
pub const MAGIC: &[u8; 8] = b"AIENLR01";
/// Calibration H file magic: n per (layer, slot) as u64, then `[inter][inter]` f32 per (layer, slot).
pub const HDN_MAGIC: &[u8; 8] = b"AIENHDN1";
const HEADER: u64 = 64;
/// Damping of H's diagonal, as a fraction of its mean (the GPTQ arm's, frozen there).
pub const DAMP: f64 = crate::q36_w4_gptq::DAMP;

/// Symmetric eigen decomposition of `a` (`[n][n]`, row-major): eigenvalues descending and the matching eigenvectors as
/// rows (`[n][n]`). Householder tridiagonalisation then implicit QL (EISPACK tred2 / tql2, as in JAMA).
pub fn sym_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n);
    let mut v = a.to_vec(); // V[i][j] = v[i * n + j]
    let mut d = vec![0f64; n];
    let mut e = vec![0f64; n];
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
    }
    for i in (1..n).rev() {
        let mut scale = 0f64;
        let mut h = 0f64;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + v[j * n + j] * f;
                for k in j + 1..i {
                    g += v[k * n + j] * d[k];
                    e[k] += v[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..n - 1 {
        v[(n - 1) * n + i] = v[i * n + i];
        v[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k * n + i + 1] / h;
            }
            for j in 0..=i {
                let mut g = 0f64;
                for k in 0..=i {
                    g += v[k * n + i + 1] * v[k * n + j];
                }
                for k in 0..=i {
                    v[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
        v[(n - 1) * n + j] = 0.0;
    }
    v[(n - 1) * n + n - 1] = 1.0;
    e[0] = 0.0;
    // vectors as rows from here: vt[i][k] = V[k][i], so each QL rotation touches two contiguous rows
    let mut vt = vec![0f64; n * n];
    for k in 0..n {
        for i in 0..n {
            vt[i * n + k] = v[k * n + i];
        }
    }
    drop(v);
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0f64;
    let mut tst1 = 0f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m > l {
            loop {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in d.iter_mut().skip(l + 2) {
                    *x -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1f64;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0f64;
                let mut s2 = 0f64;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    let (lo, hi) = vt.split_at_mut((i + 1) * n);
                    let (ri, ri1) = (&mut lo[i * n..], &mut hi[..n]);
                    for (a, b) in ri.iter_mut().zip(ri1.iter_mut()) {
                        let hb = *b;
                        *b = s * *a + c * hb;
                        *a = c * *a - s * hb;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if !(e[l].abs() > eps * tst1) {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
    let vals = order.iter().map(|&i| d[i]).collect();
    let mut vecs = vec![0f64; n * n];
    for (o, &i) in order.iter().enumerate() {
        vecs[o * n..(o + 1) * n].copy_from_slice(&vt[i * n..(i + 1) * n]);
    }
    (vals, vecs)
}

/// `x^T x` for `x` `[t][k]` (f32), as f64 `[k][k]`.
pub fn gram(x: &[f32], t: usize, k: usize) -> Vec<f64> {
    let mut g = vec![0f64; k * k];
    let mut xt = vec![0f32; k * t];
    for (r, row) in x.chunks_exact(k).take(t).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            xt[c * t + r] = v;
        }
    }
    for i in 0..k {
        let ci = &xt[i * t..(i + 1) * t];
        for j in i..k {
            let cj = &xt[j * t..(j + 1) * t];
            let mut acc = [0f64; 4];
            let (a4, b4) = (ci.chunks_exact(4), cj.chunks_exact(4));
            let (ra, rb) = (a4.remainder(), b4.remainder());
            for (a, b) in a4.zip(b4) {
                for q in 0..4 {
                    acc[q] += a[q] as f64 * b[q] as f64;
                }
            }
            let tail: f64 = ra.iter().zip(rb).map(|(&a, &b)| a as f64 * b as f64).sum();
            let s = acc.iter().sum::<f64>() + tail;
            g[i * k + j] = s;
            g[j * k + i] = s;
        }
    }
    g
}

/// Lower Cholesky factor of `H + DAMP mean(diag) I` (`[k][k]`). None when H has no energy (expert never routed).
pub fn chol_damped(h: &[f64], k: usize) -> Result<Option<Vec<f64>>, String> {
    let mean = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    if !(mean > 0.0) {
        return Ok(None);
    }
    let mut a = h.to_vec();
    for i in 0..k {
        a[i * k + i] += DAMP * mean;
    }
    for j in 0..k {
        let mut s = a[j * k + j];
        for p in 0..j {
            s -= a[j * k + p] * a[j * k + p];
        }
        if !(s > 0.0) {
            return Err(format!("lowrank: H + damp is not positive definite at column {j}"));
        }
        let s = s.sqrt();
        a[j * k + j] = s;
        for i in j + 1..k {
            let mut x = a[i * k + j];
            for p in 0..j {
                x -= a[i * k + p] * a[j * k + p];
            }
            a[i * k + j] = x / s;
        }
    }
    for i in 0..k {
        for j in i + 1..k {
            a[i * k + j] = 0.0;
        }
    }
    Ok(Some(a))
}

/// A fitted correction: U `[h][r]`, V `[r][k]`, both as their bf16 values.
pub struct Fit {
    pub u: Vec<u16>,
    pub v: Vec<u16>,
}

/// Fits the rank-`r` term to `e` (`[h][k]`): plain when `l` (lower Cholesky of the damped H) is None, else calibrated.
pub fn fit(e: &[f32], h: usize, k: usize, r: usize, l: Option<&[f64]>) -> Fit {
    let g = gram(e, h, k);
    // the matrix whose top eigenvectors span the best subspace: G (plain) or L^T G L (calibrated)
    let m = match l {
        None => g,
        Some(l) => {
            // gl = G L, then m = L^T gl
            let mut gl = vec![0f64; k * k];
            for i in 0..k {
                for p in 0..k {
                    let gip = g[i * k + p];
                    if gip == 0.0 {
                        continue;
                    }
                    let lrow = &l[p * k..p * k + p + 1]; // L[p][j] = 0 for j > p
                    for (j, &lv) in lrow.iter().enumerate() {
                        gl[i * k + j] += gip * lv;
                    }
                }
            }
            let mut m = vec![0f64; k * k];
            for p in 0..k {
                let lrow = &l[p * k..p * k + p + 1]; // L^T[i][p] = L[p][i]
                for (i, &lv) in lrow.iter().enumerate() {
                    for j in 0..k {
                        m[i * k + j] += lv * gl[p * k + j];
                    }
                }
            }
            for i in 0..k {
                for j in 0..i {
                    let s = 0.5 * (m[i * k + j] + m[j * k + i]);
                    m[i * k + j] = s;
                    m[j * k + i] = s;
                }
            }
            m
        }
    };
    let (_, vecs) = sym_eig(&m, k);
    // P = L R (columns p_j), V rows x_j = L^-T r_j; plain: P = R, x_j = r_j
    let mut p = vec![0f64; r * k]; // p[j][i]
    let mut x = vec![0f64; r * k];
    for j in 0..r {
        let rj = &vecs[j * k..(j + 1) * k];
        match l {
            None => {
                p[j * k..(j + 1) * k].copy_from_slice(rj);
                x[j * k..(j + 1) * k].copy_from_slice(rj);
            }
            Some(l) => {
                for i in 0..k {
                    p[j * k + i] = (0..=i).map(|q| l[i * k + q] * rj[q]).sum();
                }
                // L^T x = r: back substitution (L^T upper, L^T[i][q] = L[q][i])
                for i in (0..k).rev() {
                    let mut s = rj[i];
                    for q in i + 1..k {
                        s -= l[q * k + i] * x[j * k + q];
                    }
                    x[j * k + i] = s / l[i * k + i];
                }
            }
        }
    }
    let mut u = vec![0u16; h * r];
    for row in 0..h {
        let er = &e[row * k..(row + 1) * k];
        for j in 0..r {
            let s: f64 = er.iter().zip(&p[j * k..(j + 1) * k]).map(|(&a, &b)| a as f64 * b).sum();
            u[row * r + j] = f32_to_bf16(s as f32);
        }
    }
    let v = x.iter().map(|&q| f32_to_bf16(q as f32)).collect();
    Fit { u, v }
}

/// `e - U V` (`[h][k]`).
pub fn residual(e: &[f32], h: usize, k: usize, r: usize, f: &Fit) -> Vec<f32> {
    let v: Vec<f32> = f.v.iter().map(|&b| bf16_to_f32(b)).collect();
    let mut d = e.to_vec();
    for row in 0..h {
        let ur: Vec<f32> = f.u[row * r..(row + 1) * r].iter().map(|&b| bf16_to_f32(b)).collect();
        let dr = &mut d[row * k..(row + 1) * k];
        for (j, &uj) in ur.iter().enumerate() {
            if uj == 0.0 {
                continue;
            }
            for (o, &vv) in dr.iter_mut().zip(&v[j * k..(j + 1) * k]) {
                *o -= uj * vv;
            }
        }
    }
    d
}

/// `<A, B>` (Frobenius) of two `[k][k]` matrices: tr(D H D^T) = <D^T D, H>.
pub fn frob(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Sidecar header fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub layers: usize,
    pub slots: usize,
    pub h: usize,
    pub k: usize,
    pub r: usize,
    /// 0 plain, 1 calibrated.
    pub method: u32,
}

impl Shape {
    fn record(&self) -> u64 {
        ((self.h * self.r + self.r * self.k) * 2) as u64
    }
    fn offset(&self, l: usize, e: usize) -> u64 {
        HEADER + (l * self.slots + e) as u64 * self.record()
    }
    fn header(&self) -> [u8; HEADER as usize] {
        let mut b = [0u8; HEADER as usize];
        b[..8].copy_from_slice(MAGIC);
        for (i, v) in [self.layers, self.slots, self.h, self.k, self.r, self.method as usize].iter().enumerate() {
            b[8 + 4 * i..12 + 4 * i].copy_from_slice(&(*v as u32).to_le_bytes());
        }
        b
    }
}

/// Writes records into a sidecar file (created with its full size).
pub struct Writer {
    file: File,
    pub shape: Shape,
}

impl Writer {
    pub fn create(path: &Path, shape: Shape) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        file.set_len(shape.offset(shape.layers, 0)).map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all_at(&shape.header(), 0).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { file, shape })
    }
    pub fn put(&self, l: usize, e: usize, f: &Fit) -> Result<(), String> {
        let s = self.shape;
        if f.u.len() != s.h * s.r || f.v.len() != s.r * s.k || l >= s.layers || e >= s.slots {
            return Err(format!("lowrank sidecar: record ({l}, {e}) does not fit {s:?}"));
        }
        let mut b = Vec::with_capacity(s.record() as usize);
        for &x in f.u.iter().chain(&f.v) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.file.write_all_at(&b, s.offset(l, e)).map_err(|e| format!("lowrank sidecar: {e}"))
    }
}

/// A sidecar opened for the host reference: records read per expert.
pub struct LowRank {
    file: File,
    pub shape: Shape,
}

impl LowRank {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut b = [0u8; HEADER as usize];
        file.read_exact_at(&mut b, 0).map_err(|e| format!("{}: {e}", path.display()))?;
        if &b[..8] != MAGIC {
            return Err(format!("{}: not a lowrank sidecar", path.display()));
        }
        let f = |i: usize| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
        let shape = Shape { layers: f(0), slots: f(1), h: f(2), k: f(3), r: f(4), method: f(5) as u32 };
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        if len != shape.offset(shape.layers, 0) {
            return Err(format!("{}: {len} bytes, header {shape:?} needs {}", path.display(), shape.offset(shape.layers, 0)));
        }
        Ok(Self { file, shape })
    }

    /// Record (layer `l`, slot `e`) as f32: (U `[h][r]`, V `[r][k]`).
    pub fn get(&self, l: usize, e: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let s = self.shape;
        if l >= s.layers || e >= s.slots {
            return Err(format!("lowrank sidecar: no record ({l}, {e}) in {s:?}"));
        }
        let mut b = vec![0u8; s.record() as usize];
        self.file.read_exact_at(&mut b, s.offset(l, e)).map_err(|x| format!("lowrank sidecar: {x}"))?;
        let w: Vec<f32> = b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let (u, v) = w.split_at(s.h * s.r);
        Ok((u.to_vec(), v.to_vec()))
    }
}

/// Calibration H of every (layer, slot) down input, written by `q36_w4_store hdump`.
pub struct Hdn {
    file: File,
    pub layers: usize,
    pub slots: usize,
    pub k: usize,
    pub n: Vec<u64>,
}

impl Hdn {
    fn data(layers: usize, slots: usize) -> u64 {
        HEADER + (layers * slots * 8) as u64
    }

    /// Writes `h_dn` / `n_dn` of a calibration (None = never routed: zeros).
    pub fn write(path: &Path, h_dn: &[Vec<Option<Vec<f64>>>], n_dn: &[Vec<u64>], k: usize) -> Result<(), String> {
        let (layers, slots) = (h_dn.len(), h_dn.first().map_or(0, |v| v.len()));
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut hd = [0u8; HEADER as usize];
        hd[..8].copy_from_slice(HDN_MAGIC);
        for (i, v) in [layers, slots, k].iter().enumerate() {
            hd[8 + 4 * i..12 + 4 * i].copy_from_slice(&(*v as u32).to_le_bytes());
        }
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        file.write_all_at(&hd, 0).map_err(err)?;
        let ns: Vec<u8> = n_dn.iter().flatten().flat_map(|n| n.to_le_bytes()).collect();
        file.write_all_at(&ns, HEADER).map_err(err)?;
        let base = Self::data(layers, slots);
        let mat = (k * k * 4) as u64;
        file.set_len(base + (layers * slots) as u64 * mat).map_err(err)?;
        for (l, row) in h_dn.iter().enumerate() {
            for (e, m) in row.iter().enumerate() {
                if let Some(m) = m {
                    let b: Vec<u8> = m.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
                    file.write_all_at(&b, base + (l * slots + e) as u64 * mat).map_err(err)?;
                }
            }
        }
        file.sync_all().map_err(err)
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut b = [0u8; HEADER as usize];
        file.read_exact_at(&mut b, 0).map_err(|e| format!("{}: {e}", path.display()))?;
        if &b[..8] != HDN_MAGIC {
            return Err(format!("{}: not an H file", path.display()));
        }
        let f = |i: usize| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
        let (layers, slots, k) = (f(0), f(1), f(2));
        let mut ns = vec![0u8; layers * slots * 8];
        file.read_exact_at(&mut ns, HEADER).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = ns.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let want = Self::data(layers, slots) + (layers * slots * k * k * 4) as u64;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        if len != want {
            return Err(format!("{}: {len} bytes, header needs {want}", path.display()));
        }
        Ok(Self { file, layers, slots, k, n })
    }

    /// H of (layer `l`, slot `e`) as f64, None when no calibration row reached it.
    pub fn get(&self, l: usize, e: usize) -> Result<Option<Vec<f64>>, String> {
        if self.n[l * self.slots + e] == 0 {
            return Ok(None);
        }
        let k = self.k;
        let mut b = vec![0u8; k * k * 4];
        let off = Self::data(self.layers, self.slots) + ((l * self.slots + e) * k * k * 4) as u64;
        self.file.read_exact_at(&mut b, off).map_err(|x| format!("H file: {x}"))?;
        Ok(Some(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn sym_eig_reconstructs() {
        let n = 37;
        let mut s = 7u64;
        let b: Vec<f64> = (0..n * n).map(|_| lcg(&mut s)).collect();
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|q| b[i * n + q] * b[j * n + q]).sum();
            }
        }
        let (vals, vecs) = sym_eig(&a, n);
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        for o in 0..n {
            let v = &vecs[o * n..(o + 1) * n];
            let norm: f64 = v.iter().map(|x| x * x).sum();
            assert!((norm - 1.0).abs() < 1e-10, "norm {norm}");
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a[i * n + j] * v[j]).sum();
                assert!((av - vals[o] * v[i]).abs() < 1e-9 * vals[0], "A v != lambda v");
            }
        }
    }

    /// A rank-r error is removed (up to bf16), both fits; a wrong-sign term makes it worse (red check).
    #[test]
    fn fit_removes_low_rank_error() {
        let (h, k, r) = (64, 32, 4);
        let mut s = 11u64;
        let a: Vec<f64> = (0..h * r).map(|_| lcg(&mut s)).collect();
        let b: Vec<f64> = (0..r * k).map(|_| lcg(&mut s)).collect();
        let e: Vec<f32> = (0..h * k).map(|i| (0..r).map(|q| a[(i / k) * r + q] * b[q * k + i % k]).sum::<f64>() as f32).collect();
        let x: Vec<f32> = (0..200 * k).map(|_| lcg(&mut s) as f32).collect();
        let hm = gram(&x, 200, k);
        let l = chol_damped(&hm, k).unwrap().unwrap();
        let before = frob(&gram(&e, h, k), &hm);
        for lo in [None, Some(&l[..])] {
            let f = fit(&e, h, k, r, lo);
            let d = residual(&e, h, k, r, &f);
            let after = frob(&gram(&d, h, k), &hm);
            assert!(after < 1e-4 * before, "after {after} before {before}");
            let neg = Fit { u: f.u.iter().map(|&x| x ^ 0x8000).collect(), v: f.v.clone() };
            let dn = residual(&e, h, k, r, &neg);
            assert!(frob(&gram(&dn, h, k), &hm) > 3.0 * before);
        }
    }

    /// The calibrated fit beats the plain one in the H metric when H is skewed.
    #[test]
    fn calibrated_fit_wins_in_its_metric() {
        let (h, k, r) = (96, 48, 4);
        let mut s = 5u64;
        let e: Vec<f32> = (0..h * k).map(|_| lcg(&mut s) as f32).collect();
        let x: Vec<f32> = (0..400 * k).map(|i| (lcg(&mut s) * if i % k < 6 { 20.0 } else { 1.0 }) as f32).collect();
        let hm = gram(&x, 400, k);
        let l = chol_damped(&hm, k).unwrap().unwrap();
        let m = |f: &Fit| frob(&gram(&residual(&e, h, k, r, f), h, k), &hm);
        assert!(m(&fit(&e, h, k, r, Some(&l))) < 0.9 * m(&fit(&e, h, k, r, None)));
    }

    #[test]
    fn sidecar_and_h_round_trip() {
        let dir = std::env::temp_dir().join(format!("w4lr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shape = Shape { layers: 2, slots: 3, h: 8, k: 4, r: 2, method: 1 };
        let w = Writer::create(&dir.join("s.lr"), shape).unwrap();
        let f = Fit { u: (0..16).map(|i| f32_to_bf16(i as f32)).collect(), v: (0..8).map(|i| f32_to_bf16(-(i as f32))).collect() };
        w.put(1, 2, &f).unwrap();
        drop(w);
        let lr = LowRank::open(&dir.join("s.lr")).unwrap();
        assert_eq!(lr.shape, shape);
        let (u, v) = lr.get(1, 2).unwrap();
        assert_eq!(u[15], 15.0);
        assert_eq!(v[7], -7.0);
        assert!(lr.get(0, 0).unwrap().0.iter().all(|&x| x == 0.0));
        let hd = vec![vec![None, Some(vec![1.0, 2.0, 2.0, 5.0])], vec![None, None]];
        Hdn::write(&dir.join("h.hdn"), &hd, &[vec![0, 3], vec![0, 0]], 2).unwrap();
        let hf = Hdn::open(&dir.join("h.hdn")).unwrap();
        assert_eq!(hf.get(0, 1).unwrap().unwrap(), vec![1.0, 2.0, 2.0, 5.0]);
        assert!(hf.get(1, 0).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
