//! Error-feedback (GPTQ-style) MXFP4 rounding for the Qwen3.6-35B-A3B experts (L355).
//!
//! Plain rounding (each 32-weight group to its nearest e2m1 grid at the BEST power-of-two scale) failed the frozen 4-bit
//! accuracy rule (docs/inference/q36/w4-gate.md). Here each row is rounded left to right and every weight's rounding
//! error is pushed onto the weights still to come, weighted by the input statistics H = sum x x^T measured on calibration
//! text: the error that reaches the output (x . (w - q)) is what gets minimised, not the weight error. Same store format,
//! same kernel: only the nibbles and scales change.
//!
//! Inputs: gate|up of every expert (and the shared one) see the same MoE input, so one H per layer (2048 x 2048);
//! each expert's down sees its own activations, so one H per (layer, expert) (512 x 512), the shared expert as 256,
//! its rows scaled by the expert's output weight (router weight or shared gate: the output error is g (w - q) . a).
//! Rule per row: U = upper Cholesky factor of (H + damp)^-1; at each 32-column group start the group's scale is chosen
//! on the current, error-updated values (floor(log2 amax) - 2, and one up when the pass clips: the lower loss wins); column j:
//! q = nearest e2m1 x scale, e = (w - q) / U[j][j], w[j+1..] -= e * U[j][j+1..]. Rows are independent (threads).

use std::sync::Mutex;

/// One MXFP4 group: 32 weights share one E8M0 scale.
pub const GROUP: usize = 32;
/// Largest scale byte the kernel holds as a normal f32 (omega OMEGA_W4_E_MAX).
pub const E_MAX: i32 = 192;
/// Damping added to H's diagonal, as a fraction of its mean diagonal (fixed before the arm runs).
pub const DAMP: f64 = 0.01;

/// Calibration statistics gathered by `Qwen38Reference::moe` (FP8 weights).
pub struct Calib {
    pub hidden: usize,
    pub inter: usize,
    /// `[layer]` -> H of the MoE input, `[hidden][hidden]` row-major (full, symmetric).
    pub h_gu: Vec<Vec<f64>>,
    pub n_gu: Vec<u64>,
    /// `[layer][expert]` (expert == experts: the shared one) -> H of the down input, `[inter][inter]`, made on first use;
    /// rows come scaled by the expert's output weight (router weight, shared gate), so H = sum g^2 a a^T (20.6 GB at most).
    pub h_dn: Vec<Vec<Option<Vec<f64>>>>,
    pub n_dn: Vec<Vec<u64>>,
    pub threads: usize,
}

impl Calib {
    pub fn new(layers: usize, hidden: usize, inter: usize, experts: usize, threads: usize) -> Self {
        Self {
            hidden,
            inter,
            h_gu: (0..layers).map(|_| vec![0f64; hidden * hidden]).collect(),
            n_gu: vec![0; layers],
            h_dn: (0..layers).map(|_| (0..=experts).map(|_| None).collect()).collect(),
            n_dn: vec![vec![0; experts + 1]; layers],
            threads: threads.max(1),
        }
    }

    /// Adds the MoE inputs `x` (`[t][hidden]`) of layer `l`.
    pub fn add_gu(&mut self, l: usize, x: &[f32], t: usize) {
        if l >= self.h_gu.len() {
            return; // the MTP head's layer is not converted
        }
        syrk(&mut self.h_gu[l], x, t, self.hidden, self.threads);
        self.n_gu[l] += t as u64;
    }

    /// Adds the down inputs `a` (`[t][inter]`, each row already scaled by its output weight) of expert `e` in layer `l`.
    pub fn add_dn(&mut self, l: usize, e: usize, a: &[f32], t: usize) {
        if l >= self.h_dn.len() {
            return;
        }
        let k = self.inter;
        let h = self.h_dn[l][e].get_or_insert_with(|| vec![0f64; k * k]);
        syrk(h, a, t, k, 1);
        self.n_dn[l][e] += t as u64;
    }
}

/// h += x^T x for x `[t][k]` (h `[k][k]`, full), rows of h split over `threads`.
fn syrk<A: Copy + Send + From<f32> + std::ops::AddAssign>(h: &mut [A], x: &[f32], t: usize, k: usize, threads: usize) {
    if t == 0 {
        return;
    }
    // column-major copy so every h[i][j] is one contiguous dot over the t rows
    let mut xt = vec![0f32; k * t];
    for (r, row) in x.chunks_exact(k).take(t).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            xt[c * t + r] = v;
        }
    }
    let rows: Vec<(usize, &mut [A])> = h.chunks_exact_mut(k).enumerate().collect();
    let mut parts: Vec<Vec<(usize, &mut [A])>> = (0..threads).map(|_| Vec::new()).collect();
    for (n, r) in rows.into_iter().enumerate() {
        parts[n % threads].push(r);
    }
    let xt = &xt;
    std::thread::scope(|s| {
        for part in parts {
            s.spawn(move || {
                for (i, hrow) in part {
                    let ci = &xt[i * t..(i + 1) * t];
                    for j in i..k {
                        hrow[j] += A::from(dot32(ci, &xt[j * t..(j + 1) * t]));
                    }
                }
            });
        }
    });
    // mirror the upper triangle
    for i in 0..k {
        for j in 0..i {
            h[i * k + j] = h[j * k + i];
        }
    }
}

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for q in 0..8 {
            s[q] += x[q] * y[q];
        }
    }
    let mut t: f32 = s.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// U, the upper Cholesky factor of (H + damp * mean(diag) * I)^-1, row-major `[k][k]` (f32 for the row pass).
/// With J the order reversal: chol(J H J) = L gives H^-1 = (J L^-1 J)^T (J L^-1 J) and J L^-1 J is upper triangular.
/// Columns never seen (diagonal 0) get diagonal 1: their error is pushed nowhere (plain rounding).
pub fn hinv_upper(h: &[f64], k: usize) -> Result<Vec<f32>, String> {
    let mean = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    if !(mean > 0.0) {
        return Err("gptq: H has no energy (no calibration rows)".into());
    }
    let damp = DAMP * mean;
    // a = J (H + damp I) J
    let mut a = vec![0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let (ri, rj) = (k - 1 - i, k - 1 - j);
            let mut v = h[ri * k + rj];
            if i == j {
                v = if h[ri * k + ri] == 0.0 { 1.0 } else { v + damp };
            } else if h[ri * k + ri] == 0.0 || h[rj * k + rj] == 0.0 {
                v = 0.0;
            }
            a[i * k + j] = v;
        }
    }
    // in-place lower Cholesky a = L L^T
    for j in 0..k {
        let mut d = a[j * k + j];
        for p in 0..j {
            d -= a[j * k + p] * a[j * k + p];
        }
        if !(d > 0.0) {
            return Err(format!("gptq: H + damp is not positive definite at column {j}"));
        }
        let d = d.sqrt();
        a[j * k + j] = d;
        let (head, tail) = a.split_at_mut((j + 1) * k);
        let rj = &head[j * k..j * k + j];
        tail.chunks_exact_mut(k).for_each(|ri| {
            let mut v = ri[j];
            for p in 0..j {
                v -= ri[p] * rj[p];
            }
            ri[j] = v / d;
        });
    }
    // m = L^-1 (lower), column by column (L m = I), each column stored contiguous: mt[c][i] = m[i][c]
    let mut mt = vec![0f64; k * k];
    for c in 0..k {
        let col = &mut mt[c * k..(c + 1) * k];
        for i in c..k {
            let mut v = if i == c { 1.0 } else { 0.0 };
            for p in c..i {
                v -= a[i * k + p] * col[p];
            }
            col[i] = v / a[i * k + i];
        }
    }
    // U = J m J
    let mut u = vec![0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            u[i * k + j] = mt[(k - 1 - j) * k + (k - 1 - i)] as f32;
        }
    }
    Ok(u)
}

/// Nearest e2m1 index (0..7) of a magnitude in grid units, ties to even (omega omega_w4_e2m1_rne).
fn e2m1_rne(a: f64) -> u8 {
    const G: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if !(a < 6.0) {
        return 7;
    }
    let mut i = 0usize;
    while i < 7 && G[i + 1] <= a {
        i += 1;
    }
    let mid = (G[i] + G[i + 1]) / 2.0;
    if a < mid || (a == mid && i % 2 == 0) {
        i as u8
    } else {
        i as u8 + 1
    }
}

fn quant(v: f32, x: f64) -> (u8, f32) {
    let a = (v.abs() as f64) / x;
    let nib = e2m1_rne(a) | if v < 0.0 { 8 } else { 0 };
    (nib, aien_omega_q38::w4::e2m1(nib) * x as f32)
}

/// The group's first exponent (unbiased) on its current values: floor(log2 amax) - 2 (omega OCP). None for an all-zero
/// group. [`round_rows`] tries one up when the pass clips (with H = I that is omega's BEST rule).
fn group_exp(v: &[f32]) -> Option<i32> {
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs())) as f64;
    if amax == 0.0 {
        return None;
    }
    let e = ((amax.to_bits() >> 52) & 0x7ff) as i32 - 1023 - 2; // floor(log2 amax): the exponent field, exact (omega floor_log2)
    Some(e)
}

/// One group's pass at scale 2^e over `seg` = row[c0..block end] (rounded in place, the error pushed right inside the
/// block).
struct GroupPass {
    nib: [u8; GROUP],
    err: [f32; GROUP],
    /// Sum of (w - q)^2 / U[j][j]^2: the error-feedback (OBS) loss of the group's columns.
    loss: f64,
    /// A value above 6 x scale was met (clipped).
    clip: bool,
}

fn group_pass(seg: &mut [f32], c0: usize, k: usize, u: &[f32], e: i32) -> Result<GroupPass, String> {
    if e + 127 < 0 || e + 127 > E_MAX {
        return Err(format!("gptq: scale exponent {e} outside the kernel's range"));
    }
    let x = 2f64.powi(e);
    let mut gp = GroupPass { nib: [0; GROUP], err: [0.0; GROUP], loss: 0.0, clip: false };
    let end = c0 + seg.len();
    for jj in 0..GROUP {
        let j = c0 + jj;
        let v = seg[jj];
        gp.clip |= (v.abs() as f64) / x > 6.0;
        let (nib, q) = quant(v, x);
        let d = u[j * k + j];
        let er = (v - q) / d;
        gp.nib[jj] = nib;
        gp.err[jj] = er;
        gp.loss += ((v - q) as f64 / d as f64).powi(2);
        if er != 0.0 {
            for (o, &uu) in seg[jj + 1..].iter_mut().zip(&u[j * k + j + 1..j * k + end]) {
                *o -= er * uu;
            }
        }
        seg[jj] = q;
    }
    Ok(gp)
}

/// Columns per block: inside a block the error moves column by column; past it, once per block as one product
/// (the same arithmetic in another order, so the tables stay in cache).
const BLOCK: usize = 128;
/// Rows that share one pass over a block of `u`.
const TILE: usize = 16;

/// Rounds rows `w` (`[rows][k]`, f32, overwritten as the error moves right) with the upper factor `u` (`[k][k]`) into
/// nibbles (`[rows][k/2]`, low nibble = even column) and scale bytes (`[rows][k/32]`). Returns the summed squared weight
/// error and the summed squared input (for the stats line).
pub fn round_rows(w: &mut [f32], k: usize, u: &[f32], w4: &mut [u8], e8: &mut [u8]) -> Result<(f64, f64), String> {
    if !k.is_multiple_of(BLOCK) {
        return Err(format!("gptq: {k} columns is not whole {BLOCK} blocks"));
    }
    let rows = w.len() / k;
    let orig = w.to_vec();
    let mut err = vec![0f32; TILE * BLOCK];
    for r0 in (0..rows).step_by(TILE) {
        let tr = TILE.min(rows - r0);
        for b0 in (0..k).step_by(BLOCK) {
            let b1 = b0 + BLOCK;
            // inside the block, row by row; each group tries the BEST exponent and, when a value clips during the pass
            // (error feedback can push later columns past 6 x scale), one up, keeping the lower error-feedback loss
            for t in 0..tr {
                let r = r0 + t;
                let row = &mut w[r * k..(r + 1) * k];
                let (nb, sb) = (&mut w4[r * k / 2..(r + 1) * k / 2], &mut e8[r * k / GROUP..(r + 1) * k / GROUP]);
                for g in b0 / GROUP..b1 / GROUP {
                    let c0 = g * GROUP;
                    let Some(e0) = group_exp(&row[c0..c0 + GROUP]) else {
                        sb[g] = 127;
                        (c0..c0 + GROUP).for_each(|j| nb[j >> 1] &= !(15 << ((j & 1) * 4)));
                        err[t * BLOCK + c0 - b0..t * BLOCK + c0 - b0 + GROUP].iter_mut().for_each(|e| *e = 0.0);
                        continue;
                    };
                    let mut a = row[c0..b1].to_vec();
                    let (mut pick, mut ga) = (e0, group_pass(&mut a, c0, k, u, e0)?);
                    if ga.clip {
                        let mut b = row[c0..b1].to_vec();
                        let gb = group_pass(&mut b, c0, k, u, e0 + 1)?;
                        if gb.loss < ga.loss {
                            (pick, ga, a) = (e0 + 1, gb, b);
                        }
                    }
                    sb[g] = (pick + 127) as u8;
                    row[c0..b1].copy_from_slice(&a);
                    for (jj, &nib) in ga.nib.iter().enumerate() {
                        let j = c0 + jj;
                        nb[j >> 1] = (nb[j >> 1] & !(15 << ((j & 1) * 4))) | nib << ((j & 1) * 4);
                    }
                    err[t * BLOCK + c0 - b0..t * BLOCK + c0 - b0 + GROUP].copy_from_slice(&ga.err);
                }
            }
            // past the block: w[t][b1..] -= sum_p err[t][p] * u[b0 + p][b1..], in column chunks
            for c0 in (b1..k).step_by(BLOCK) {
                let c1 = (c0 + BLOCK).min(k);
                for p in 0..BLOCK {
                    let ur = &u[(b0 + p) * k + c0..(b0 + p) * k + c1];
                    for t in 0..tr {
                        let e = err[t * BLOCK + p];
                        if e != 0.0 {
                            let r = r0 + t;
                            for (o, &uu) in w[r * k + c0..r * k + c1].iter_mut().zip(ur) {
                                *o -= e * uu;
                            }
                        }
                    }
                }
            }
        }
    }
    let (mut se, mut sr) = (0f64, 0f64);
    for (q, o) in w.iter().zip(&orig) {
        se += ((q - o) as f64).powi(2);
        sr += (*o as f64).powi(2);
    }
    Ok((se, sr))
}

/// [`round_rows`] over `threads` tiles of rows. `w` `[n][k]` f32 in, nibbles and scale bytes out; (sq err, sq ref).
pub fn round_matrix(w: &mut [f32], n: usize, k: usize, u: &[f32], threads: usize) -> Result<(Vec<u8>, Vec<u8>, f64, f64), String> {
    let mut w4 = vec![0u8; n * k / 2];
    let mut e8 = vec![0u8; n * k / GROUP];
    let tile = n.div_ceil(threads.max(1)).max(1);
    let acc = Mutex::new((0f64, 0f64, None::<String>));
    std::thread::scope(|s| {
        for ((wt, nt), et) in w.chunks_mut(tile * k).zip(w4.chunks_mut(tile * k / 2)).zip(e8.chunks_mut(tile * k / GROUP)) {
            let acc = &acc;
            s.spawn(move || {
                let r = round_rows(wt, k, u, nt, et);
                let mut a = acc.lock().unwrap();
                match r {
                    Ok((se, sr)) => {
                        a.0 += se;
                        a.1 += sr;
                    }
                    Err(e) => a.2 = Some(e),
                }
            });
        }
    });
    let (se, sr, err) = acc.into_inner().unwrap();
    if let Some(e) = err {
        return Err(e);
    }
    Ok((w4, e8, se, sr))
}

/// Writes the 4-bit store with error-feedback rounding for the experts (projections 2 and 3 of
/// [`crate::q36_w4_store::layer_projs`]: gate|up with the layer's input H, each down with its expert's H) and the C
/// BEST rule for the trunk (projections 0 and 1). An expert the calibration never routed to gets H = I (plain BEST).
pub fn write_store(
    dir: &std::path::Path,
    out: &std::path::Path,
    calib: &Calib,
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<crate::q36_w4_store::Header, String> {
    let (h, inter) = (calib.hidden, calib.inter);
    let mut eye_dn = vec![0f64; inter * inter];
    (0..inter).for_each(|i| eye_dn[i * inter + i] = 1.0);
    let u_eye_dn = hinv_upper(&eye_dn, inter)?;
    let mut unseen = 0usize;
    let quant = &mut |l: usize, p: usize, name: &str, n: usize, k: usize, w8: &[u8], s: &[u16]| {
        let stats = |se: f64, sr: f64| aien_omega_q38::w4::W4Stats {
            weights: (n * k) as u64,
            groups: (n * k / GROUP) as u64,
            sq_err: se,
            sq_ref: sr,
            ..Default::default()
        };
        match p {
            2 => {
                if k != h || calib.n_gu[l] == 0 {
                    return Err(format!("{name}: [{n}][{k}] or no calibration rows for layer {l}"));
                }
                let u = hinv_upper(&calib.h_gu[l], h)?;
                let mut wf = widen_fp8(n, k, w8, s);
                let (w4, e8, se, sr) = round_matrix(&mut wf, n, k, &u, threads)?;
                Ok((w4, e8, None, stats(se, sr)))
            }
            3 => {
                if k != inter || !n.is_multiple_of(h) {
                    return Err(format!("{name}: [{n}][{k}] is not stacked [{h}][{inter}] experts"));
                }
                let experts = n / h;
                let wf = widen_fp8(n, k, w8, s);
                let mut w4 = vec![0u8; n * k / 2];
                let mut e8 = vec![0u8; n * k / GROUP];
                let acc = Mutex::new((0f64, 0f64, None::<String>, 0usize));
                let next = std::sync::atomic::AtomicUsize::new(0);
                let (w4p, e8p) = (w4.as_mut_ptr() as usize, e8.as_mut_ptr() as usize);
                std::thread::scope(|sc| {
                    for _ in 0..threads.max(1) {
                        sc.spawn(|| loop {
                            let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if e >= experts {
                                break;
                            }
                            let r = (|| -> Result<(f64, f64, bool), String> {
                                let (u, seen) = match calib.h_dn[l][e].as_ref().filter(|_| calib.n_dn[l][e] > 0) {
                                    Some(hd) => (hinv_upper(hd, inter)?, true),
                                    None => (u_eye_dn.clone(), false),
                                };
                                let mut blk = wf[e * h * k..(e + 1) * h * k].to_vec();
                                // SAFETY: expert e owns rows [e h, (e + 1) h) of w4 / e8 (disjoint per thread), alive for the scope.
                                let (nb, sb) = unsafe {
                                    (
                                        std::slice::from_raw_parts_mut((w4p as *mut u8).add(e * h * k / 2), h * k / 2),
                                        std::slice::from_raw_parts_mut((e8p as *mut u8).add(e * h * k / GROUP), h * k / GROUP),
                                    )
                                };
                                let (se, sr) = round_rows(&mut blk, k, &u, nb, sb)?;
                                Ok((se, sr, seen))
                            })();
                            let mut a = acc.lock().unwrap();
                            match r {
                                Ok((se, sr, seen)) => {
                                    a.0 += se;
                                    a.1 += sr;
                                    a.3 += usize::from(!seen);
                                }
                                Err(m) => a.2 = Some(format!("{name} expert {e}: {m}")),
                            }
                        });
                    }
                });
                let (se, sr, err, un) = acc.into_inner().unwrap();
                if let Some(m) = err {
                    return Err(m);
                }
                unseen += un;
                Ok((w4, e8, None, stats(se, sr)))
            }
            _ => aien_omega_q38::w4::from_fp8(n, k, w8, s, threads, aien_omega_q38::w4::W4_RULE_BEST)
                .map(|(a, b, st)| (a, b, None, st))
                .map_err(|e| format!("{name}: {e}")),
        }
    };
    let converter = format!(
        "L355 q36_w4_gptq: experts error-feedback MXFP4 (H from calibration, damp {DAMP} x mean diag, BEST group scale on \
         the updated values), trunk omega BEST; calibration rows per layer {:?}",
        calib.n_gu
    );
    let hdr = crate::q36_w4_store::write_store_with(dir, out, "gptq", &converter, canary()?, quant, log)?;
    log(&format!("gptq store: {unseen} (layer, expert) pairs never routed in calibration, rounded plainly"));
    Ok(hdr)
}

/// FP8 e4m3 rows with 128 x 128 bf16 block scales, widened to f32 (`[n][k]`).
pub fn widen_fp8(n: usize, k: usize, w8: &[u8], scales: &[u16]) -> Vec<f32> {
    let lut = crate::qwen3_moe::fp8_table();
    let kb = k / 128;
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            out[r * k + c] = lut[w8[r * k + c] as usize] * crate::qwen3_moe::bf16_to_f32(scales[(r / 128) * kb + c / 128]);
        }
    }
    out
}

/// The canary: [`crate::q36_w4_store::canary_fp8`] rounded with H = I (no error feedback), nibbles then scales, hex.
/// A store tagged `gptq` opens only when the linked rounding gives the same bytes.
pub fn canary() -> Result<String, String> {
    let (w, s) = crate::q36_w4_store::canary_fp8();
    let mut f = widen_fp8(128, 128, &w, &s);
    let mut eye = vec![0f64; 128 * 128];
    (0..128).for_each(|i| eye[i * 128 + i] = 1.0);
    let u = hinv_upper(&eye, 128)?;
    let (w4, e8, _, _) = round_matrix(&mut f, 128, 128, &u, 1)?;
    Ok([w4, e8].concat().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_h_is_plain_best_rounding() {
        // with H = I nothing moves: every group equals the C converter's BEST rounding of the same values
        let (w, s) = crate::q36_w4_store::canary_fp8();
        let mut f = widen_fp8(128, 128, &w, &s);
        let mut eye = vec![0f64; 128 * 128];
        (0..128).for_each(|i| eye[i * 128 + i] = 1.0);
        let u = hinv_upper(&eye, 128).unwrap();
        let (w4, e8, _, _) = round_matrix(&mut f, 128, 128, &u, 2).unwrap();
        if aien_omega_q38::w4::native() {
            let (c4, c8, _) = aien_omega_q38::w4::from_fp8(128, 128, &w, &s, 1, aien_omega_q38::w4::W4_RULE_BEST).unwrap();
            assert_eq!(e8, c8, "scales differ from the C BEST rule");
            assert_eq!(w4, c4, "nibbles differ from the C BEST rule");
        }
    }

    #[test]
    fn error_feedback_lowers_output_error() {
        // correlated inputs: the output error x.(w - q) over the calibration rows must drop against plain rounding
        let (k, t, n) = (128usize, 512usize, 8usize);
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let base: Vec<f32> = (0..t).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..t * k).map(|i| base[i / k] + 0.3 * rnd()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
        let mut h = vec![0f64; k * k];
        syrk(&mut h, &x, t, k, 2);
        let out_err = |q: &[f32]| -> f64 {
            let mut s = 0.0;
            for r in 0..t {
                for o in 0..n {
                    let d: f32 = (0..k).map(|c| x[r * k + c] * (w[o * k + c] - q[o * k + c])).sum();
                    s += (d as f64).powi(2);
                }
            }
            s
        };
        let deq = |w4: &[u8], e8: &[u8]| -> Vec<f32> {
            let mut q = vec![0f32; n * k];
            for r in 0..n {
                aien_omega_q38::w4::widen_row(w4, e8, k, r, &mut q[r * k..(r + 1) * k]);
            }
            q
        };
        let mut eye = vec![0f64; k * k];
        (0..k).for_each(|i| eye[i * k + i] = 1.0);
        let (p4, p8, _, _) = round_matrix(&mut w.clone(), n, k, &hinv_upper(&eye, k).unwrap(), 1).unwrap();
        let (g4, g8, _, _) = round_matrix(&mut w.clone(), n, k, &hinv_upper(&h, k).unwrap(), 1).unwrap();
        let (plain, gptq) = (out_err(&deq(&p4, &p8)), out_err(&deq(&g4, &g8)));
        assert!(gptq < 0.8 * plain, "gptq output error {gptq} not below plain {plain}");
    }

    #[test]
    fn hinv_upper_inverts() {
        // U^T U == (H + damp)^-1  <=>  (H + damp) U^T U == I
        let k = 16;
        let mut h = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                h[i * k + j] = 1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { 1.0 } else { 0.0 };
            }
        }
        let u = hinv_upper(&h, k).unwrap();
        let mean = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
        for i in 0..k {
            for j in 0..k {
                let mut v = 0f64;
                for p in 0..k {
                    let hp = h[i * k + p] + if i == p { DAMP * mean } else { 0.0 };
                    let mut utu = 0f64;
                    for q in 0..k {
                        utu += u[q * k + p] as f64 * u[q * k + j] as f64;
                    }
                    v += hp * utu;
                }
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-4, "({i},{j}) = {v}");
            }
        }
    }
}
