//! MXFP4 weight simulation for the 4-bit drift study (sc#679 task B).
//!
//! OCP MX FP4: every 32 weights along k share one power-of-two scale
//! `X = 2^(floor(log2 amax) - 2)`, and each weight becomes e2m1 (0, 0.5, 1, 1.5, 2, 3, 4, 6),
//! round to nearest even, saturating at 6. `q * X` goes back into the checkpoint's own FP8 form:
//! an e4m3 code under a new power-of-two scale `2^s` per 128x128 block, with `s` the block's
//! largest group exponent minus 6 (so the largest weight, 6 * 2^e, lands on 384 <= 448). That
//! holds every `q * X` exactly unless a group sits more than 14 binades below the block's
//! largest group; those weights are rounded to e4m3 and counted in [`W4Stats::underflow`]. The
//! chip then runs the 4-bit model through the unchanged FP8 kernels.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};
use crate::qwen3_moe_omega_prefill::fake_quant::{e4m3_rne_sat, e4m3_to_f32};

const BLOCK: usize = 128;
/// Weights per MX scale.
pub const GROUP: usize = 32;
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

static ON: AtomicBool = AtomicBool::new(false);
static STATS: Mutex<W4Stats> = Mutex::new(W4Stats::new());

/// What the conversion did, summed over every converted tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct W4Stats {
    pub tensors: u64,
    pub weights: u64,
    pub groups: u64,
    /// Weights not exactly `q * X` after the e4m3 write-back (group far below its block's max).
    pub underflow: u64,
    /// Sum of squared weight change and of squared original weight (relative RMS error).
    pub sq_err: f64,
    pub sq_ref: f64,
}

impl W4Stats {
    const fn new() -> Self {
        Self {
            tensors: 0,
            weights: 0,
            groups: 0,
            underflow: 0,
            sq_err: 0.0,
            sq_ref: 0.0,
        }
    }
}

/// Every FP8 projection loaded from now on is converted to MXFP4 first.
pub fn set_mxfp4(on: bool) {
    ON.store(on, Ordering::SeqCst);
}

pub fn mxfp4_on() -> bool {
    ON.load(Ordering::SeqCst)
}

pub fn stats() -> W4Stats {
    *STATS.lock().expect("w4 stats")
}

/// e2m1 of `a >= 0`: nearest grid value, ties to the even code, saturating at 6.
fn e2m1(a: f64) -> f64 {
    if a >= 6.0 {
        return 6.0;
    }
    let i = E2M1.iter().rposition(|&g| g <= a).expect("a >= 0");
    let (lo, hi) = (E2M1[i], E2M1[i + 1]);
    let mid = (lo + hi) / 2.0;
    if a < mid || (a == mid && i % 2 == 0) {
        lo
    } else {
        hi
    }
}

/// `[n][k]` e4m3 codes with `[n/128][k/128]` bf16 scales, converted in place to MXFP4 values.
pub fn mxfp4_fp8(n: usize, k: usize, w: &mut [u8], scales: &mut [u16]) -> Result<(), String> {
    if !n.is_multiple_of(BLOCK) || !k.is_multiple_of(BLOCK) {
        return Err(format!("mxfp4: [{n}][{k}] is not whole 128 blocks"));
    }
    if w.len() != n * k || scales.len() != (n / BLOCK) * (k / BLOCK) {
        return Err("mxfp4: weight or scale length".into());
    }
    let mut st = W4Stats::new();
    st.tensors = 1;
    let kb = k / BLOCK;
    let gpr = BLOCK / GROUP;
    let mut q = vec![0f64; BLOCK * BLOCK];
    let mut ex: Vec<Option<i32>> = vec![None; BLOCK * gpr];
    for bi in 0..n / BLOCK {
        for bj in 0..kb {
            let s_old = bf16_to_f32(scales[bi * kb + bj]) as f64;
            for r in 0..BLOCK {
                let row = &w[(bi * BLOCK + r) * k + bj * BLOCK..][..BLOCK];
                for g in 0..gpr {
                    let v = |c: usize| e4m3_to_f32(row[g * GROUP + c]) as f64 * s_old;
                    let amax = (0..GROUP).map(|c| v(c).abs()).fold(0f64, f64::max);
                    let e = (amax > 0.0).then(|| amax.log2().floor() as i32 - 2);
                    ex[r * gpr + g] = e;
                    for c in 0..GROUP {
                        q[r * BLOCK + g * GROUP + c] = match e {
                            Some(e) => {
                                let x = v(c);
                                e2m1(x.abs() / 2f64.powi(e)).copysign(x)
                            }
                            None => 0.0,
                        };
                    }
                }
            }
            st.groups += (BLOCK * gpr) as u64;
            let Some(e_max) = ex.iter().flatten().copied().max() else {
                continue;
            };
            let s = e_max - 6;
            let s_new = 2f64.powi(s);
            let s_bf = f32_to_bf16(s_new as f32);
            if bf16_to_f32(s_bf) as f64 != s_new {
                return Err(format!("mxfp4: scale 2^{s} is not a bf16"));
            }
            scales[bi * kb + bj] = s_bf;
            for r in 0..BLOCK {
                let row = &mut w[(bi * BLOCK + r) * k + bj * BLOCK..][..BLOCK];
                for c in 0..BLOCK {
                    let old = e4m3_to_f32(row[c]) as f64 * s_old;
                    let target = match ex[r * gpr + c / GROUP] {
                        Some(e) => q[r * BLOCK + c] * 2f64.powi(e - s),
                        None => 0.0,
                    };
                    let code = e4m3_rne_sat(target as f32);
                    if e4m3_to_f32(code) as f64 != target {
                        st.underflow += 1;
                    }
                    row[c] = code;
                    let new = e4m3_to_f32(code) as f64 * s_new;
                    st.sq_err += (new - old) * (new - old);
                    st.sq_ref += old * old;
                }
            }
        }
    }
    st.weights = (n * k) as u64;
    let mut all = STATS.lock().expect("w4 stats");
    all.tensors += st.tensors;
    all.weights += st.weights;
    all.groups += st.groups;
    all.underflow += st.underflow;
    all.sq_err += st.sq_err;
    all.sq_ref += st.sq_ref;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_rounds_to_nearest_even() {
        assert_eq!(e2m1(0.25), 0.0); // tie 0 | 0.5 -> code 0
        assert_eq!(e2m1(0.75), 1.0); // tie 0.5 | 1 -> code 2
        assert_eq!(e2m1(2.5), 2.0); // tie 2 | 3 -> code 4
        assert_eq!(e2m1(3.5), 4.0); // tie 3 | 4 -> code 6
        assert_eq!(e2m1(5.0), 4.0); // tie 4 | 6 -> code 6
        assert_eq!(e2m1(5.1), 6.0);
        assert_eq!(e2m1(7.9), 6.0);
    }

    /// One block: values come back as e2m1 times the group's power of two, exactly.
    #[test]
    fn block_round_trip_is_mx_exact() {
        let (n, k) = (128, 128);
        let mut w: Vec<u8> = (0..n * k)
            .map(|i| ((i * 37 + 11) % 0x7e) as u8 | if i % 3 == 0 { 0x80 } else { 0 })
            .collect();
        let mut sc = vec![f32_to_bf16(0.01); 1];
        let before: Vec<f64> = w
            .iter()
            .map(|&c| e4m3_to_f32(c) as f64 * 0.01f32 as f64)
            .collect();
        mxfp4_fp8(n, k, &mut w, &mut sc).unwrap();
        let s = bf16_to_f32(sc[0]) as f64;
        assert_eq!(s.log2().fract(), 0.0, "power-of-two block scale");
        for r in 0..n {
            for g in 0..k / GROUP {
                let idx = |c: usize| r * k + g * GROUP + c;
                let amax = (0..GROUP)
                    .map(|c| before[idx(c)].abs())
                    .fold(0f64, f64::max);
                let x = 2f64.powi(amax.log2().floor() as i32 - 2);
                for c in 0..GROUP {
                    let got = e4m3_to_f32(w[idx(c)]) as f64 * s;
                    let want = e2m1(before[idx(c)].abs() / x).copysign(before[idx(c)]) * x;
                    assert_eq!(got, want, "row {r} group {g} col {c}");
                    assert!((got / x).abs() <= 6.0);
                }
            }
        }
    }
}
