//! NVFP4-style 4-bit weights for the Qwen3.6-35B-A3B experts (L355, host arm; docs/inference/q36/w4-gate.md "NV16 arm").
//!
//! MXFP4 (one power-of-two scale per 32 weights) failed the frozen accuracy rule with every rounding tried
//! (docs/inference/q36/w4-results.md). This grid is finer: one e4m3 (FP8) scale per 16 weights, times one f32 global per
//! tensor. Value = e2m1(nibble) x e4m3(scale byte) x global. Nibbles as MXFP4 (`[n][k/2]`, low nibble = even column, sign
//! bit 3); scale bytes `[n][k/16]`. 4.5 bits per weight.
//!
//! Rule (fixed before the arm runs): global = amax(tensor) / (448 x 6); a group's scale byte = the e4m3 code nearest
//! (ties to the even code) to amax_g / 6 / global, at least the smallest non-zero code for a non-zero group; when a value
//! then clips (|w| > 6 x scale), the next code up is tried too and the lower squared error wins. One global per linear
//! (each expert's gate, up, down). Values: nearest e2m1,
//! ties to the even code (omega omega_w4_e2m1_rne; the PTX cvt.rn rule: 2.5 -> 2, 3.5 -> 4, 5 -> 4, 0.75 -> 1).

use std::sync::Mutex;

/// Weights per scale byte.
pub const GROUP: usize = 16;
/// The largest finite e4m3 value.
const E4M3_MAX: f32 = 448.0;

/// e4m3 code (0..=0x7e, positive) to its value (the checkpoint's FP8 table).
fn e4m3(code: u8) -> f32 {
    crate::qwen3_moe::fp8_table()[code as usize]
}

/// The positive e4m3 code nearest to `v` (ties to the even code), clamped to 0..=0x7e. Codes 0..=0x7e are increasing.
fn e4m3_rne(v: f32, lut: &[f32; 256]) -> u8 {
    if !(v > 0.0) {
        return 0;
    }
    if v >= E4M3_MAX {
        return 0x7e;
    }
    // largest code with value <= v
    let (mut lo, mut hi) = (0usize, 0x7e);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if lut[mid] <= v {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 0x7e || lut[lo] == v {
        return lo as u8;
    }
    let (a, b) = (lut[lo], lut[lo + 1]);
    let (da, db) = (v - a, b - v);
    if da < db || (da == db && lo % 2 == 0) {
        lo as u8
    } else {
        lo as u8 + 1
    }
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

/// One group at scale `x` (= e4m3 value x global): nibbles, squared error, clipped count.
fn group_at(v: &[f32], x: f64, nib: &mut [u8; GROUP]) -> (f64, u64) {
    let (mut se, mut clip) = (0f64, 0u64);
    for (j, &w) in v.iter().enumerate() {
        let a = (w.abs() as f64) / x;
        clip += u64::from(a > 6.0);
        let n = e2m1_rne(a) | if w < 0.0 { 8 } else { 0 };
        nib[j] = n;
        let back = aien_omega_q38::w4::e2m1(n) as f64 * x;
        se += (back - w as f64).powi(2);
    }
    (se, clip)
}

/// The tensor's global scale: amax / (448 x 6), 1 for an all-zero tensor.
pub fn global_of(w: &[f32]) -> f32 {
    let amax = w.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if amax > 0.0 {
        amax / (E4M3_MAX * 6.0)
    } else {
        1.0
    }
}

/// Rounds rows `w` (`[rows][k]`) with `global` into nibbles (`[rows][k/2]`) and scale bytes (`[rows][k/16]`).
/// Returns (sum sq err, sum sq ref, groups, clipped values).
fn round_rows(w: &[f32], k: usize, global: f32, w4: &mut [u8], s8: &mut [u8]) -> (f64, f64, u64, u64) {
    let lut = crate::qwen3_moe::fp8_table();
    let g = global as f64;
    let (mut se, mut sr, mut groups, mut sat) = (0f64, 0f64, 0u64, 0u64);
    let (mut nib, mut nib2) = ([0u8; GROUP], [0u8; GROUP]);
    for (gi, v) in w.chunks_exact(GROUP).enumerate() {
        groups += 1;
        sr += v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let base = gi * GROUP / 2;
        if amax == 0.0 {
            s8[gi] = 0;
            w4[base..base + GROUP / 2].iter_mut().for_each(|b| *b = 0);
            continue;
        }
        let mut code = e4m3_rne((amax as f64 / 6.0 / g) as f32, &lut).max(1);
        let (mut err, clip) = group_at(v, lut[code as usize] as f64 * g, &mut nib);
        let mut clipped = clip;
        if clip > 0 && code < 0x7e {
            let (err2, clip2) = group_at(v, lut[code as usize + 1] as f64 * g, &mut nib2);
            if err2 < err {
                code += 1;
                err = err2;
                clipped = clip2;
                nib = nib2;
            }
        }
        s8[gi] = code;
        for j in 0..GROUP / 2 {
            w4[base + j] = nib[2 * j] | nib[2 * j + 1] << 4;
        }
        se += err;
        sat += clipped;
    }
    let _ = k;
    (se, sr, groups, sat)
}

/// A whole tensor `w` (`[n][k]`, f32) to the nv16 grid on `threads` threads: (nibbles, scale bytes, global, stats).
pub fn quantize(w: &[f32], n: usize, k: usize, threads: usize) -> Result<(Vec<u8>, Vec<u8>, f32, aien_omega_q38::w4::W4Stats), String> {
    if !k.is_multiple_of(GROUP) || w.len() != n * k {
        return Err(format!("nv16: [{n}][{k}] with {} values", w.len()));
    }
    let global = global_of(w);
    let mut w4 = vec![0u8; n * k / 2];
    let mut s8 = vec![0u8; n * k / GROUP];
    let tile = n.div_ceil(threads.max(1)).max(1);
    let acc = Mutex::new(aien_omega_q38::w4::W4Stats::default());
    std::thread::scope(|s| {
        for ((wt, nt), st) in w.chunks(tile * k).zip(w4.chunks_mut(tile * k / 2)).zip(s8.chunks_mut(tile * k / GROUP)) {
            let acc = &acc;
            s.spawn(move || {
                let (se, sr, groups, sat) = round_rows(wt, k, global, nt, st);
                let mut a = acc.lock().unwrap();
                a.weights += wt.len() as u64;
                a.groups += groups;
                a.saturated += sat;
                a.sq_err += se;
                a.sq_ref += sr;
            });
        }
    });
    Ok((w4, s8, global, acc.into_inner().unwrap()))
}

/// Row `r` of an nv16 tensor as f32 into `out` (`k` values).
pub fn widen_row(w4: &[u8], s8: &[u8], global: f32, k: usize, r: usize, out: &mut [f32]) {
    let rb = &w4[r * k / 2..(r + 1) * k / 2];
    let sb = &s8[r * k / GROUP..(r + 1) * k / GROUP];
    for (gi, (o, &s)) in out[..k].chunks_exact_mut(GROUP).zip(sb).enumerate() {
        let x = e4m3(s) * global;
        for (j, o) in o.iter_mut().enumerate() {
            let c = gi * GROUP + j;
            *o = aien_omega_q38::w4::e2m1((rb[c >> 1] >> ((c & 1) * 4)) & 15) * x;
        }
    }
}

/// The canary: [`crate::q36_w4_store::canary_fp8`] on this grid, nibbles, scales, then the global's bits, hex.
pub fn canary() -> Result<String, String> {
    let (w, s) = crate::q36_w4_store::canary_fp8();
    let f = crate::q36_w4_gptq::widen_fp8(128, 128, &w, &s);
    let (w4, s8, g, _) = quantize(&f, 128, 128, 1)?;
    Ok([w4, s8, g.to_bits().to_le_bytes().to_vec()].concat().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_ties_go_to_the_even_code() {
        // codes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 at indices 0..7; a tie goes to the even index (the PTX cvt.rn rule)
        for (a, want) in [(0.25, 0u8), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6), (6.5, 7), (2.6, 5)] {
            assert_eq!(e2m1_rne(a), want, "{a}");
        }
    }

    #[test]
    fn e4m3_rne_is_the_nearest_code() {
        let lut = crate::qwen3_moe::fp8_table();
        let mut seed = 7u64;
        for _ in 0..20000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((seed >> 11) as f64 / (1u64 << 53) as f64 * 9.0 - 6.0).exp2() as f32 * 60.0;
            let c = e4m3_rne(v, &lut);
            let best = (0..=0x7eusize).map(|q| ((lut[q] - v).abs(), q)).fold((f32::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
            assert!((lut[c as usize] - v).abs() <= best.0, "{v}: code {c} vs {}", best.1);
        }
    }

    #[test]
    fn grid_values_round_trip_and_pick_never_loses_to_the_nearest_scale() {
        // a group made of exact grid values at a known scale comes back exactly; on random groups the picked scale is never
        // worse than the plain nearest scale (the clip retry only takes a lower error)
        let lut = crate::qwen3_moe::fp8_table();
        let g = 0.01f32;
        let s = lut[0x38]; // 1.0
        let grid = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let v: Vec<f32> = (0..GROUP).map(|j| grid[j % 8] * if j % 3 == 0 { -1.0 } else { 1.0 } * s * g).collect();
        let mut w4 = vec![0u8; GROUP / 2];
        let mut s8 = vec![0u8; 1];
        let (se, ..) = round_rows(&v, GROUP, g, &mut w4, &mut s8);
        assert_eq!(se, 0.0, "grid values must round-trip");
        let mut out = vec![0f32; GROUP];
        widen_row(&w4, &s8, g, GROUP, 0, &mut out);
        assert_eq!(out, v);
        // random groups: no non-clipping code beats the pick
        let mut seed = 99u64;
        for _ in 0..500 {
            let v: Vec<f32> = (0..GROUP)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.3
                })
                .collect();
            let (se, ..) = round_rows(&v, GROUP, g, &mut w4, &mut s8);
            let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let q = e4m3_rne(amax / 6.0 / g, &lut).max(1);
            let mut nb = [0u8; GROUP];
            let (e_near, _) = group_at(&v, lut[q as usize] as f64 * g as f64, &mut nb);
            assert!(se <= e_near, "pick {} err {se} worse than nearest {q} err {e_near}", s8[0]);
        }
    }

    #[test]
    fn mutant_changes_the_values() {
        let (w, s) = crate::q36_w4_store::canary_fp8();
        let f = crate::q36_w4_gptq::widen_fp8(128, 128, &w, &s);
        let (w4, s8, g, st) = quantize(&f, 128, 128, 2).unwrap();
        assert!(st.rel_rms() < 0.2, "rel rms {}", st.rel_rms());
        let m: Vec<u8> = w4.iter().map(|b| b & 0xEE).collect();
        let (mut a, mut b) = (vec![0f32; 128], vec![0f32; 128]);
        let mut differ = 0;
        for r in 0..128 {
            widen_row(&w4, &s8, g, 128, r, &mut a);
            widen_row(&m, &s8, g, 128, r, &mut b);
            differ += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        }
        assert!(differ > 1000, "mutant changed only {differ} values");
    }
}

/// Writes the store with every tensor (trunk and experts) on the nv16 grid; the host arm reads the parts it is told to.
/// One global per linear: each expert's gate, up (`expert_inter` rows each) and down (`hidden` rows); a trunk tensor has one.
pub fn write_store(
    dir: &std::path::Path,
    out: &std::path::Path,
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<crate::q36_w4_store::Header, String> {
    let c = crate::qwen38::Qwen38Config::from_checkpoint(dir)?;
    let s = c.moe.clone().ok_or("nv16 store: not a MoE checkpoint")?;
    let quant = &mut |_l: usize, p: usize, name: &str, n: usize, k: usize, w8: &[u8], sc: &[u16]| {
        let per = match p {
            2 => s.expert_inter,
            3 => c.hidden,
            _ => n,
        };
        if !n.is_multiple_of(per) {
            return Err(format!("{name}: {n} rows are not whole linears of {per}"));
        }
        let f = crate::q36_w4_gptq::widen_fp8(n, k, w8, sc);
        let (mut w4, mut s8, mut globals) = (Vec::with_capacity(n * k / 2), Vec::with_capacity(n * k / GROUP), Vec::new());
        let mut st = aien_omega_q38::w4::W4Stats::default();
        for blk in f.chunks(per * k) {
            let (a, b, g, t) = quantize(blk, per, k, threads).map_err(|e| format!("{name}: {e}"))?;
            w4.extend(a);
            s8.extend(b);
            globals.push(g);
            st.add(&t);
        }
        Ok((w4, s8, Some((globals, per)), st))
    };
    let converter = "L355 q36_w4_nv: e2m1 x e4m3 scale per 16 x f32 global per linear (amax / 2688), scale = nearest e4m3, one 
                     code up when a value clips and that lowers the squared error; e2m1 ties to the even code (cvt.rn)";
    crate::q36_w4_store::write_store_with(dir, out, "nv16", converter, canary()?, quant, log)
}
