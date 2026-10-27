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

//! Host mirror of omega's activation quantiser (W-L1G-C, 2026-10-14), for the W8A8 control.
//!
//! Per row and per 128-wide block: a power-of-two scale from the block's amax, then e4m3
//! round-to-nearest-even saturating to 448. Line for line the C reference `pq_ref`
//! (omega-own30b-tc tests/gpu_qgemm_test.c, 1e76e728) over `qm_e4m3_rne_sat` / `qm_e4m3`
//! (tests/bw_qmma_model.h). The chip quantiser is bit-identical to that reference (hardware
//! receipt OWN30B-L1G-GEMM-f3b18df9, "quantiser exact"), so `fake_quant_row` reproduces exactly
//! the rounding the ON path's GEMMs see; the golden table `pq_golden.txt` was printed by that C
//! code (investigations/2026-10-14-own30b-wp-w8a8-gate/src/pqgold) and is checked here.

use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};

/// Elements per scale block.
pub const BLOCK: usize = 128;

/// f32 -> e4m3 code, round to nearest even, saturate to +-448 (NaN not handled: never produced
/// by the quantiser's finite inputs; mirrors `qm_e4m3_rne_sat`).
pub fn e4m3_rne_sat(f: f32) -> u8 {
    let s: u8 = if f.is_sign_negative() { 0x80 } else { 0 };
    let a = (f as f64).abs();
    if f.is_infinite() || a > 448.0 {
        return s | 0x7e;
    }
    if a < 2f64.powi(-6) {
        return s | (a * 512.0).round_ties_even() as u8;
    }
    // a = m * 2^e, m in [1, 2)
    let mut e = a.log2().floor() as i32;
    let mut m = a / 2f64.powi(e);
    if m >= 2.0 {
        e += 1;
        m /= 2.0;
    } else if m < 1.0 {
        e -= 1;
        m *= 2.0;
    }
    let mut mm = ((m - 1.0) * 8.0).round_ties_even() as i32;
    if mm >= 8 {
        mm = 0;
        e += 1;
    }
    if e + 7 > 15 || (((e + 7) << 3) | mm) > 0x7e {
        return s | 0x7e;
    }
    s | ((((e + 7) << 3) | mm) as u8)
}

/// e4m3 code -> f32 (mirrors `qm_e4m3`).
pub fn e4m3_to_f32(c: u8) -> f32 {
    let (s, e, mm) = (c >> 7, ((c >> 3) & 15) as i32, (c & 7) as f32);
    let v = if e != 0 {
        (1.0 + mm / 8.0) * 2f32.powi(e - 7)
    } else {
        (mm / 8.0) * 2f32.powi(-6)
    };
    if s != 0 {
        -v
    } else {
        v
    }
}

/// The quantiser's block scale and its inverse for a block whose amax is `m` (`m >= 2^-100`).
pub fn block_scale(m: f32) -> (f32, f32) {
    let mb = m.to_bits();
    let (e, mant) = (mb >> 23, mb & 0x7f_ffff);
    let c = (mant + 0x1f_ffff) >> 23;
    (
        f32::from_bits((e + c - 8) << 23),
        f32::from_bits((262 - e - c) << 23),
    )
}

/// Quantises one row of `k` values (k a multiple of 128) exactly as `pq_ref`: writes the e4m3
/// codes to `q` and the block scales to `s`.
pub fn quant_row(x: &[f32], q: &mut [u8], s: &mut [f32]) {
    let k = x.len();
    assert!(
        k.is_multiple_of(BLOCK) && q.len() == k && s.len() == k / BLOCK,
        "quant_row: k {k} q {} s {}",
        q.len(),
        s.len()
    );
    for b in 0..k / BLOCK {
        let blk = &x[b * BLOCK..(b + 1) * BLOCK];
        let mut m = 2f32.powi(-100);
        for &v in blk {
            m = m.max(v.abs());
        }
        let (scale, inv) = block_scale(m);
        s[b] = scale;
        for (j, &v) in blk.iter().enumerate() {
            q[b * BLOCK + j] = e4m3_rne_sat(v * inv);
        }
    }
}

/// Quantise then dequantise one row in place: `x[j] = e4m3(x[j] / scale_b) * scale_b`.
pub fn fake_quant_row(x: &mut [f32]) {
    let k = x.len();
    let mut q = vec![0u8; k];
    let mut s = vec![0f32; k / BLOCK];
    quant_row(x, &mut q, &mut s);
    for j in 0..k {
        x[j] = e4m3_to_f32(q[j]) * s[j / BLOCK];
    }
}

/// Rounds every value to bf16 and back (the ON path's SwiGLU output is bf16 before its quant).
pub fn bf16_round_row(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = bf16_to_f32(f32_to_bf16(*v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = include_str!("pq_golden.txt");

    #[test]
    fn e4m3_edges() {
        assert_eq!(e4m3_rne_sat(448.0), 0x7e);
        assert_eq!(e4m3_rne_sat(1000.0), 0x7e);
        assert_eq!(e4m3_rne_sat(-1000.0), 0xfe);
        assert_eq!(e4m3_rne_sat(0.0), 0x00);
        assert_eq!(e4m3_rne_sat(1.0), 0x38);
        assert_eq!(e4m3_rne_sat(2f32.powi(-6)), 0x08);
        assert_eq!(e4m3_rne_sat(2f32.powi(-9)), 0x01);
        assert_eq!(
            e4m3_rne_sat(1.0625),
            0x38,
            "tie to even (1.0625 is halfway between 1.0 and 1.125)"
        );
        assert_eq!(
            e4m3_rne_sat(1.1875),
            0x3a,
            "tie to even (halfway between 1.125 and 1.25)"
        );
        for c in 0u8..=255 {
            if c & 0x7f == 0x7f {
                continue;
            }
            assert_eq!(
                e4m3_rne_sat(e4m3_to_f32(c)),
                if c == 0x80 { 0x80 } else { c },
                "round trip of code {c:#x}"
            );
        }
    }

    #[test]
    fn block_scale_rounds_amax_up_above_1_75() {
        assert_eq!(block_scale(1.0), (2f32.powi(-8), 256.0));
        assert_eq!(block_scale(1.75), (2f32.powi(-8), 256.0));
        assert_eq!(block_scale(1.8), (2f32.powi(-7), 128.0));
        assert_eq!(block_scale(3.0e4), (2f32.powi(7), 2f32.powi(-7)));
        let mut row = vec![0.3f32; 128];
        row[0] = 1.0;
        fake_quant_row(&mut row);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 0.3125);
    }

    #[test]
    fn matches_the_c_reference_golden() {
        // Golden: "S row blk scalebits" per block, then "Q xbits qbyte" per element (k = 256, 6 rows).
        let (k, rows) = (256usize, 6usize);
        let mut x = vec![0f32; rows * k];
        let mut want_q = vec![0u8; rows * k];
        let mut want_s = vec![0f32; rows * 2];
        let (mut nq, mut ns) = (0, 0);
        for line in GOLDEN.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            match f[0] {
                "S" => {
                    let (r, b): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
                    want_s[r * 2 + b] = f32::from_bits(u32::from_str_radix(f[3], 16).unwrap());
                    ns += 1;
                }
                "Q" => {
                    x[nq] = f32::from_bits(u32::from_str_radix(f[1], 16).unwrap());
                    want_q[nq] = u8::from_str_radix(f[2], 16).unwrap();
                    nq += 1;
                }
                _ => panic!("golden line {line}"),
            }
        }
        assert_eq!((nq, ns), (rows * k, rows * 2));
        let mut q = vec![0u8; rows * k];
        let mut s = vec![0f32; rows * 2];
        for r in 0..rows {
            quant_row(
                &x[r * k..(r + 1) * k],
                &mut q[r * k..(r + 1) * k],
                &mut s[r * 2..(r + 1) * 2],
            );
        }
        let bad_q = q.iter().zip(&want_q).filter(|(a, b)| a != b).count();
        let bad_s = s
            .iter()
            .zip(&want_s)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            (bad_q, bad_s),
            (0, 0),
            "codes / scales differing from the C reference"
        );
    }
}
