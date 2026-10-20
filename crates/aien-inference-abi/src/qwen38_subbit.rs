//! Sub-4-bit weight codes for the Qwen3.8-27B drift study (ledger L134, SPEED-10). Host only.
//!
//! Question: if the body weights were stored at 3 bits instead of 4, how far would the model's
//! next-word choices move? This module holds the codes (the weights are rebuilt as f32 after a
//! round trip through the code, so the existing host reference runs unchanged):
//!
//! - `mxfp4`: the 4-bit control the GPU already runs (`e2m1` per weight, one power-of-two scale per
//!   32 weights, the OCP rule). Its drift on this host must land near the GPU's measured drift,
//!   or the harness is wrong.
//! - `rht-scalar:<bits>`: random sign flips plus a 128-point Hadamard rotation along `k`, a scale
//!   per group, then an optimal (Lloyd-Max) scalar quantiser. The rotation makes the weights close
//!   to Gaussian, which is what the codes below are built for.
//! - `trellis:<K>:<L>:<G>`: the same rotation and scale, then a bit-shift trellis code in the
//!   family of QTIP (Tseng et al. 2024) and EXL3: a state of `L` bits, `K` new bits per weight
//!   shift in, the weight value is a table lookup of the new state (`2^L` entries, trained on
//!   Gaussian data). The encoder is an exact Viterbi search. The cost is `K` bits per weight plus
//!   `(L - K) / T` for the first state of each sequence of `T` weights plus the scale.
//!
//! The rotation is orthonormal and the signs depend only on the column, so a kernel could rotate
//! the activation once per norm output and multiply with the stored rotated weights; here the
//! weights are rotated back instead, which gives the same products.

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Small numeric helpers.
// ---------------------------------------------------------------------------

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in the open interval (0, 1).
fn unit(z: u64) -> f64 {
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Inverse of the standard normal CDF (Acklam's rational approximation, relative error 1e-9).
pub fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -norm_inv(1.0 - p)
    }
}

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

// ---------------------------------------------------------------------------
// Rotation: signs by column, then a 128-point orthonormal Walsh-Hadamard transform.
// ---------------------------------------------------------------------------

pub const HAD: usize = 128;

fn sign_of(col: usize) -> f32 {
    if splitmix(0x51AB_u64 ^ col as u64) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// In-place orthonormal Walsh-Hadamard transform of 128 values. It is its own inverse.
pub fn fwht128(x: &mut [f32]) {
    debug_assert_eq!(x.len(), HAD);
    let mut h = 1;
    while h < HAD {
        for i in (0..HAD).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (HAD as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

/// Rotates a row (`k` a multiple of 128): signs, then the transform per 128-block.
fn rotate_row(row: &mut [f32], col0: usize) {
    for (b, blk) in row.chunks_exact_mut(HAD).enumerate() {
        for (i, v) in blk.iter_mut().enumerate() {
            *v *= sign_of(col0 + b * HAD + i);
        }
        fwht128(blk);
    }
}

/// Undoes [`rotate_row`].
fn unrotate_row(row: &mut [f32], col0: usize) {
    for (b, blk) in row.chunks_exact_mut(HAD).enumerate() {
        fwht128(blk);
        for (i, v) in blk.iter_mut().enumerate() {
            *v *= sign_of(col0 + b * HAD + i);
        }
    }
}

// ---------------------------------------------------------------------------
// MXFP4 (the control).
// ---------------------------------------------------------------------------

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Nearest e2m1 magnitude to `a >= 0`, ties to the even code, saturating at 6.
fn e2m1_round(a: f32) -> f32 {
    if a >= 6.0 {
        return 6.0;
    }
    let mut best = 0;
    for i in 1..8 {
        let (d, db) = ((a - E2M1[i]).abs(), (a - E2M1[best]).abs());
        if d < db || (d == db && i % 2 == 0) {
            best = i;
        }
    }
    E2M1[best]
}

/// One group of 32: scale `2^(floor(log2 amax) - 2)`, e2m1 per weight. The same rule as the GPU
/// converter (omega `w4` host oracle), written again here from its description.
fn mxfp4_group(g: &mut [f32]) {
    let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return;
    }
    let e = amax.log2().floor() as i32 - 2;
    let scale = 2f32.powi(e.clamp(-126, 127));
    for v in g.iter_mut() {
        let q = e2m1_round(v.abs() / scale);
        *v = q.copysign(*v) * scale;
    }
}

// ---------------------------------------------------------------------------
// Scalar Lloyd-Max for a unit Gaussian.
// ---------------------------------------------------------------------------

/// Optimal levels for 1 to 4 bits (Max 1960): positive half, ascending.
fn lloyd_max_levels(bits: usize) -> Vec<f32> {
    let pos: &[f32] = match bits {
        1 => &[0.7979],
        2 => &[0.4528, 1.5104],
        3 => &[0.2451, 0.7560, 1.3439, 2.1519],
        4 => &[0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326],
        _ => panic!("rht-scalar: {bits} bits not tabulated"),
    };
    let mut all: Vec<f32> = pos.iter().rev().map(|v| -v).collect();
    all.extend_from_slice(pos);
    all
}

fn nearest(levels: &[f32], x: f32) -> f32 {
    let mut best = levels[0];
    for &l in levels {
        if (x - l).abs() < (x - best).abs() {
            best = l;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Bit-shift trellis code.
// ---------------------------------------------------------------------------

/// A trellis code: `L`-bit state, `K` bits per weight, `code[s]` is the value of state `s`.
#[derive(Clone, Debug)]
pub struct Trellis {
    pub k_bits: usize,
    pub l_bits: usize,
    pub code: Vec<f32>,
}

/// Scratch for one encoder thread.
pub struct Scratch {
    cost: Vec<f32>,
    next: Vec<f32>,
    m: Vec<f32>,
    bp: Vec<u8>,
}

impl Trellis {
    /// Codebook of unit variance in a fixed pseudo-random order: the `2^L` Gaussian quantiles
    /// placed on the states by a seeded shuffle.
    pub fn gaussian_init(k_bits: usize, l_bits: usize) -> Self {
        assert!(k_bits >= 1 && l_bits > k_bits && l_bits <= 20);
        let n = 1usize << l_bits;
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (splitmix(0xC0DE ^ i as u64) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut code = vec![0f32; n];
        for (rank, &s) in order.iter().enumerate() {
            code[s] = norm_inv((rank as f64 + 0.5) / n as f64) as f32;
        }
        Self {
            k_bits,
            l_bits,
            code,
        }
    }

    pub fn scratch(&self, t: usize) -> Scratch {
        let n = 1usize << self.l_bits;
        let u = n >> self.k_bits;
        Scratch {
            cost: vec![0.0; n],
            next: vec![0.0; n],
            m: vec![0.0; u],
            bp: vec![0; t * u],
        }
    }

    /// Exact minimum-squared-error path for `w` (any start state). Writes the state per weight
    /// into `states` and returns the squared error.
    pub fn encode(&self, w: &[f32], sc: &mut Scratch, states: &mut [u32]) -> f32 {
        let (kb, lb) = (self.k_bits, self.l_bits);
        let n = 1usize << lb;
        let u = n >> kb;
        let width = 1usize << kb;
        let t = w.len();
        debug_assert!(sc.bp.len() >= t * u && states.len() == t);
        for (c, &cd) in sc.cost.iter_mut().zip(&self.code) {
            let d = cd - w[0];
            *c = d * d;
        }
        for step in 1..t {
            // m[u'] = min over the 2^K predecessors u' + j * U, and which j.
            let bp = &mut sc.bp[step * u..(step + 1) * u];
            sc.m.copy_from_slice(&sc.cost[..u]);
            bp.fill(0);
            for j in 1..width {
                let slice = &sc.cost[j * u..(j + 1) * u];
                for ((m, b), &c) in sc.m.iter_mut().zip(bp.iter_mut()).zip(slice) {
                    let better = c < *m;
                    *m = if better { c } else { *m };
                    *b = if better { j as u8 } else { *b };
                }
            }
            let wt = w[step];
            for ((nx, cd), &m) in sc
                .next
                .chunks_exact_mut(width)
                .zip(self.code.chunks_exact(width))
                .zip(&sc.m)
            {
                for (o, &c) in nx.iter_mut().zip(cd) {
                    let d = c - wt;
                    *o = m + d * d;
                }
            }
            std::mem::swap(&mut sc.cost, &mut sc.next);
        }
        let (mut best, mut best_c) = (0usize, f32::INFINITY);
        for (s, &c) in sc.cost.iter().enumerate() {
            if c < best_c {
                best = s;
                best_c = c;
            }
        }
        let mut s = best;
        for step in (0..t).rev() {
            states[step] = s as u32;
            if step > 0 {
                let j = sc.bp[step * u + (s >> kb)] as usize;
                s = (s >> kb) | (j << (lb - kb));
            }
        }
        best_c
    }

    /// The `L + K (T - 1)` bits of one sequence, from its states, as a bit string (one `bool`
    /// per bit, first state high bit first, then each step's new `K` bits).
    pub fn pack(&self, states: &[u32]) -> Vec<bool> {
        let (kb, lb) = (self.k_bits, self.l_bits);
        let mut bits = Vec::with_capacity(lb + kb * (states.len() - 1));
        for i in (0..lb).rev() {
            bits.push((states[0] >> i) & 1 == 1);
        }
        for &s in &states[1..] {
            for i in (0..kb).rev() {
                bits.push((s >> i) & 1 == 1);
            }
        }
        bits
    }

    /// Decodes a bit string from [`Self::pack`] back to the `T` weight values.
    pub fn unpack(&self, bits: &[bool], t: usize) -> Vec<f32> {
        let (kb, lb) = (self.k_bits, self.l_bits);
        assert_eq!(bits.len(), lb + kb * (t - 1));
        let mask = (1u32 << lb) - 1;
        let mut s = 0u32;
        for &b in &bits[..lb] {
            s = (s << 1) | b as u32;
        }
        let mut out = vec![self.code[s as usize]];
        let mut at = lb;
        for _ in 1..t {
            let mut nb = 0u32;
            for &b in &bits[at..at + kb] {
                nb = (nb << 1) | b as u32;
            }
            at += kb;
            s = ((s << kb) | nb) & mask;
            out.push(self.code[s as usize]);
        }
        out
    }

    /// Lloyd refinement of the codebook on synthetic unit Gaussian sequences: encode, then each
    /// state's value becomes the mean of the targets that chose it. Returns the relative error
    /// (error energy over signal energy) of each round, measured before that round's update.
    pub fn train_gaussian(&mut self, t: usize, seqs: usize, rounds: usize, seed: u64) -> Vec<f64> {
        let n = self.code.len();
        let mut history = Vec::new();
        for round in 0..rounds {
            let this = self.clone();
            let (sum, cnt, err) = (0..seqs)
                .into_par_iter()
                .fold(
                    || (vec![0f64; n], vec![0u32; n], 0f64),
                    |(mut sum, mut cnt, mut err), i| {
                        let mut sc = this.scratch(t);
                        let mut states = vec![0u32; t];
                        let w: Vec<f32> = (0..t)
                            .map(|j| {
                                let z = splitmix(seed ^ ((round * seqs + i) * t + j) as u64);
                                norm_inv(unit(z)) as f32
                            })
                            .collect();
                        err += this.encode(&w, &mut sc, &mut states) as f64;
                        for (&s, &x) in states.iter().zip(&w) {
                            sum[s as usize] += x as f64;
                            cnt[s as usize] += 1;
                        }
                        (sum, cnt, err)
                    },
                )
                .reduce(
                    || (vec![0f64; n], vec![0u32; n], 0f64),
                    |(mut a, mut ca, ea), (b, cb, eb)| {
                        a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                        ca.iter_mut().zip(&cb).for_each(|(x, y)| *x += y);
                        (a, ca, ea + eb)
                    },
                );
            history.push(err / (seqs * t) as f64);
            for s in 0..n {
                if cnt[s] > 0 {
                    self.code[s] = (sum[s] / cnt[s] as f64) as f32;
                }
            }
        }
        history
    }

    /// Relative error of the current codebook on fresh Gaussian data.
    pub fn gaussian_error(&self, t: usize, seqs: usize, seed: u64) -> f64 {
        let err: f64 = (0..seqs)
            .into_par_iter()
            .map(|i| {
                let mut sc = self.scratch(t);
                let mut states = vec![0u32; t];
                let w: Vec<f32> = (0..t)
                    .map(|j| norm_inv(unit(splitmix(seed ^ (i * t + j) as u64))) as f32)
                    .collect();
                self.encode(&w, &mut sc, &mut states) as f64
            })
            .sum();
        err / (seqs * t) as f64
    }

    /// Bits per weight of a sequence of `t` weights, without scales.
    pub fn bits_per_weight(&self, t: usize) -> f64 {
        (self.l_bits + self.k_bits * (t - 1)) as f64 / t as f64
    }
}

// ---------------------------------------------------------------------------
// Variants and the per-matrix transform.
// ---------------------------------------------------------------------------

/// Which code the body weights go through.
#[derive(Clone, Debug)]
pub enum Variant {
    Mxfp4,
    RhtScalar { bits: usize, group: usize },
    Trellis { tr: Trellis, group: usize, seq: usize },
}

impl Variant {
    /// `mxfp4`, `rht-scalar:<bits>:<group>`, `trellis:<K>:<L>:<group>[:<T>]`. `group` 0 means
    /// one scale per row. The trellis codebook is trained on Gaussian data (see
    /// [`Trellis::train_gaussian`]) unless `codebook` points at a saved one.
    pub fn parse(spec: &str, codebook_dir: Option<&std::path::Path>) -> Result<Self, String> {
        let p: Vec<&str> = spec.split(':').collect();
        let num = |i: usize| -> Result<usize, String> {
            p.get(i)
                .ok_or_else(|| format!("{spec}: missing field {i}"))?
                .parse()
                .map_err(|_| format!("{spec}: field {i} is not a number"))
        };
        match p[0] {
            "mxfp4" => Ok(Self::Mxfp4),
            "rht-scalar" => Ok(Self::RhtScalar {
                bits: num(1)?,
                group: num(2)?,
            }),
            "trellis" => {
                let (kb, lb, group) = (num(1)?, num(2)?, num(3)?);
                let seq = if p.len() > 4 { num(4)? } else { 256 };
                let mut tr = Trellis::gaussian_init(kb, lb);
                let file = codebook_dir.map(|d| d.join(format!("codebook-K{kb}-L{lb}.f32")));
                match file.as_ref().filter(|f| f.exists()) {
                    Some(f) => {
                        let raw = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
                        if raw.len() != 4 << lb {
                            return Err(format!("{}: wrong size", f.display()));
                        }
                        for (c, b) in tr.code.iter_mut().zip(raw.chunks_exact(4)) {
                            *c = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        }
                    }
                    None => {
                        let hist = tr.train_gaussian(256, 4096, 8, 0x7EA1);
                        eprintln!("codebook K{kb} L{lb}: Lloyd rounds, relative error {hist:?}");
                        if let Some(f) = file {
                            let raw: Vec<u8> = tr.code.iter().flat_map(|c| c.to_le_bytes()).collect();
                            std::fs::write(&f, raw).map_err(|e| format!("{}: {e}", f.display()))?;
                        }
                    }
                }
                Ok(Self::Trellis { tr, group, seq })
            }
            other => Err(format!("unknown variant {other}")),
        }
    }

    /// Stored bits per weight (code bits plus scales, scales counted as bf16 per group).
    pub fn bits_per_weight(&self, k: usize) -> f64 {
        match self {
            Self::Mxfp4 => 4.25,
            Self::RhtScalar { bits, group } => {
                *bits as f64 + 16.0 / if *group == 0 { k } else { *group } as f64
            }
            Self::Trellis { tr, group, seq } => {
                tr.bits_per_weight(*seq) + 16.0 / if *group == 0 { k } else { *group } as f64
            }
        }
    }

    /// Round-trips one `[n][k]` matrix through the code and returns the rebuilt weights.
    pub fn apply(&self, w: &[f32], n: usize, k: usize, colscale: Option<&[f32]>) -> Vec<f32> {
        let mut out = w.to_vec();
        let cs = colscale.filter(|_| !matches!(self, Self::Mxfp4));
        if let Some(s) = cs {
            assert_eq!(s.len(), k);
            out.par_chunks_mut(k).for_each(|row| row.iter_mut().zip(s).for_each(|(v, &f)| *v *= f));
        }
        match self {
            Self::Mxfp4 => {
                out.par_chunks_mut(32).for_each(mxfp4_group);
            }
            Self::RhtScalar { bits, group } => {
                let levels = lloyd_max_levels(*bits);
                let g = if *group == 0 { k } else { *group };
                out.par_chunks_mut(k).for_each(|row| {
                    rotate_row(row, 0);
                    for grp in row.chunks_exact_mut(g) {
                        let sc = group_scale(grp);
                        for v in grp.iter_mut() {
                            *v = if sc == 0.0 { 0.0 } else { nearest(&levels, *v / sc) * sc };
                        }
                    }
                    unrotate_row(row, 0);
                });
            }
            Self::Trellis { tr, group, seq } => {
                let g = if *group == 0 { k } else { *group };
                let t = *seq;
                assert!(k % t == 0 && k % g == 0 && t % HAD == 0, "k {k} vs seq {t} group {g}");
                out.par_chunks_mut(k).for_each_init(
                    || (tr.scratch(t), vec![0u32; t]),
                    |(sc, states), row| {
                        rotate_row(row, 0);
                        let scales: Vec<f32> =
                            row.chunks_exact(g).map(group_scale).collect();
                        let mut norm = vec![0f32; k];
                        for (gi, (grp, nrm)) in
                            row.chunks_exact(g).zip(norm.chunks_exact_mut(g)).enumerate()
                        {
                            let s = scales[gi];
                            for (o, &v) in nrm.iter_mut().zip(grp) {
                                *o = if s == 0.0 { 0.0 } else { v / s };
                            }
                        }
                        for (seg, dst) in norm.chunks_exact(t).zip(row.chunks_exact_mut(t)) {
                            tr.encode(seg, sc, states);
                            for (o, &s) in dst.iter_mut().zip(states.iter()) {
                                *o = tr.code[s as usize];
                            }
                        }
                        for (gi, grp) in row.chunks_exact_mut(g).enumerate() {
                            grp.iter_mut().for_each(|v| *v *= scales[gi]);
                        }
                        unrotate_row(row, 0);
                    },
                );
            }
        }
        if let Some(s) = cs {
            out.par_chunks_mut(k).for_each(|row| row.iter_mut().zip(s).for_each(|(v, &f)| *v /= f));
        }
        let _ = n;
        out
    }
}

/// Group scale: root mean square, stored as bf16.
fn group_scale(g: &[f32]) -> f32 {
    let ms = g.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / g.len() as f64;
    bf16_round(ms.sqrt() as f32)
}

/// Relative error energy of a rebuilt matrix: `|rebuilt - w|^2 / |w|^2`.
pub fn relative_error(w: &[f32], rebuilt: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in w.iter().zip(rebuilt) {
        let d = (a - b) as f64;
        num += d * d;
        den += (a as f64) * (a as f64);
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_orthonormal_and_self_inverse() {
        let orig: Vec<f32> = (0..HAD).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
        let mut x = orig.clone();
        fwht128(&mut x);
        let (e0, e1): (f32, f32) = (
            orig.iter().map(|v| v * v).sum(),
            x.iter().map(|v| v * v).sum(),
        );
        assert!((e0 - e1).abs() < 1e-3 * e0, "energy {e0} vs {e1}");
        fwht128(&mut x);
        for (a, b) in orig.iter().zip(&x) {
            assert!((a - b).abs() < 1e-4);
        }
        let mut row = vec![0f32; 256];
        row[3] = 1.0;
        let before = row.clone();
        rotate_row(&mut row, 0);
        assert!(row.iter().filter(|v| v.abs() > 1e-6).count() >= 128);
        unrotate_row(&mut row, 0);
        for (a, b) in before.iter().zip(&row) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn e2m1_grid_and_ties() {
        assert_eq!(e2m1_round(0.24), 0.0);
        assert_eq!(e2m1_round(0.26), 0.5);
        assert_eq!(e2m1_round(0.25), 0.0); // tie 0 / 0.5 goes to the even code 0
        assert_eq!(e2m1_round(2.5), 2.0); // tie 2 (code 4) / 3 (code 5) goes to even code 4
        assert_eq!(e2m1_round(5.0), 4.0); // tie 4 (code 6) / 6 (code 7) goes to even code 6
        assert_eq!(e2m1_round(100.0), 6.0);
        let mut g = [3.0f32; 32];
        g[0] = 6.0; // amax 6: e = 2 - 2 = 0, scale 1
        mxfp4_group(&mut g);
        assert_eq!(g[0], 6.0);
        assert_eq!(g[1], 3.0);
    }

    #[test]
    fn viterbi_matches_brute_force() {
        // L 4, K 2, T 5: every path is one of 2^(4 + 2*4) = 4096 bit strings.
        let tr = Trellis::gaussian_init(2, 4);
        let w = [0.3f32, -1.1, 0.8, 0.05, -0.6];
        let mut sc = tr.scratch(5);
        let mut states = [0u32; 5];
        let got = tr.encode(&w, &mut sc, &mut states);
        let mut best = f32::INFINITY;
        for code in 0u32..(1 << 12) {
            let bits: Vec<bool> = (0..12).rev().map(|i| (code >> i) & 1 == 1).collect();
            let vals = tr.unpack(&bits, 5);
            let e: f32 = vals.iter().zip(&w).map(|(a, b)| (a - b) * (a - b)).sum();
            best = best.min(e);
        }
        assert!((got - best).abs() < 1e-5, "viterbi {got} vs brute force {best}");
        let bits = tr.pack(&states);
        assert_eq!(bits.len(), 12);
        let back = tr.unpack(&bits, 5);
        let e: f32 = back.iter().zip(&w).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!((e - got).abs() < 1e-5, "pack/unpack round trip changed the error");
    }

    #[test]
    fn codebook_start_is_unit_gaussian_and_training_helps() {
        let mut tr = Trellis::gaussian_init(3, 8);
        let m: f64 = tr.code.iter().map(|&c| c as f64).sum::<f64>() / 256.0;
        let v: f64 = tr.code.iter().map(|&c| (c as f64) * (c as f64)).sum::<f64>() / 256.0;
        assert!(m.abs() < 1e-3 && (v - 1.0).abs() < 0.03, "mean {m} var {v}");
        let before = tr.gaussian_error(64, 256, 9);
        tr.train_gaussian(64, 512, 4, 1);
        let after = tr.gaussian_error(64, 256, 9);
        assert!(after < before, "training raised the error {before} -> {after}");
    }

    #[test]
    fn variants_round_trip_a_matrix() {
        let (n, k) = (4usize, 256usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| norm_inv(unit(splitmix(i as u64 + 5))) as f32 * 0.02)
            .collect();
        let tr = Variant::Trellis {
            tr: Trellis::gaussian_init(3, 8),
            group: 128,
            seq: 256,
        };
        for v in [
            Variant::Mxfp4,
            Variant::RhtScalar { bits: 3, group: 128 },
            tr,
        ] {
            let out = v.apply(&w, n, k, None);
            let e = relative_error(&w, &out);
            assert!(e > 0.0 && e < 0.2, "{v:?}: relative error {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Activation-weighted column scaling (the calibrated variant).
// ---------------------------------------------------------------------------

/// Mean square of every input channel of each matmul, per layer, from a calibration run. The
/// output error of a weight change `e` on column `c` is `d_c * e^2` (diagonal of the input second
/// moment), so columns the model uses hard are scaled up before the code and back down after, which
/// spends the code's error where the output cares less (the diagonal case of what GPTQ / LDLQ do).
#[derive(Clone, Debug)]
pub struct ActTable {
    pub exp: f32,
    pub xn: Vec<Vec<f32>>,
    pub mix: Vec<Vec<f32>>,
    pub mlp_in: Vec<Vec<f32>>,
    pub act: Vec<Vec<f32>>,
}

impl ActTable {
    pub fn from_means(
        rows: u64,
        sums: [&Vec<Vec<f64>>; 4],
    ) -> Self {
        let mean = |s: &Vec<Vec<f64>>| -> Vec<Vec<f32>> {
            s.iter()
                .map(|l| l.iter().map(|&v| (v / rows as f64) as f32).collect())
                .collect()
        };
        Self {
            exp: 0.25,
            xn: mean(sums[0]),
            mix: mean(sums[1]),
            mlp_in: mean(sums[2]),
            act: mean(sums[3]),
        }
    }

    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        let mut raw = b"ACT1".to_vec();
        raw.extend_from_slice(&(self.xn.len() as u32).to_le_bytes());
        for l in 0..self.xn.len() {
            for v in [&self.xn[l], &self.mix[l], &self.mlp_in[l], &self.act[l]] {
                raw.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v.iter() {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        std::fs::write(path, raw).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn read(path: &std::path::Path, exp: f32) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() < 8 || &raw[..4] != b"ACT1" {
            return Err(format!("{}: not an ACT1 file", path.display()));
        }
        let u32_at = |at: usize| u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]);
        let layers = u32_at(4) as usize;
        let mut at = 8;
        let mut t = Self {
            exp,
            xn: vec![],
            mix: vec![],
            mlp_in: vec![],
            act: vec![],
        };
        for _ in 0..layers {
            for which in 0..4 {
                let len = u32_at(at) as usize;
                at += 4;
                let v: Vec<f32> = raw[at..at + 4 * len]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                at += 4 * len;
                match which {
                    0 => t.xn.push(v),
                    1 => t.mix.push(v),
                    2 => t.mlp_in.push(v),
                    _ => t.act.push(v),
                }
            }
        }
        Ok(t)
    }

    /// Column scale for the tensor `name` (a text-model projection), or `None` if it is not one.
    pub fn colscale(&self, name: &str, k: usize) -> Option<Vec<f32>> {
        let rest = name.split("layers.").nth(1)?;
        let l: usize = rest.split('.').next()?.parse().ok()?;
        let d: &Vec<f32> = if ["q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z"]
            .iter()
            .any(|s| name.contains(s))
        {
            self.xn.get(l)?
        } else if name.contains("o_proj") || name.contains("out_proj") {
            self.mix.get(l)?
        } else if name.contains("gate_proj") || name.contains("up_proj") {
            self.mlp_in.get(l)?
        } else if name.contains("down_proj") {
            self.act.get(l)?
        } else {
            return None;
        };
        if d.len() != k || self.exp == 0.0 {
            return None;
        }
        let mean = d.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
        if mean <= 0.0 {
            return None;
        }
        Some(
            d.iter()
                .map(|&v| ((v as f64 / mean).clamp(1e-4, 1e4) as f32).powf(self.exp))
                .collect(),
        )
    }
}
