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

//! Host tests run everywhere (stub-aware). The chip parity test is #[ignore]d:
//! run it on the Spark through the heavy queue with a native build.
use aien_omega_gpu::*;

fn lcg(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
}

/// Round to nearest even bf16, return the bits.
fn bf16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    if f.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (r >> 16) as u16
}
fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn reference(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

#[test]
fn lock_is_full_sha() {
    assert_eq!(PINNED_OMEGA_SHA.len(), 40);
    // The build reads omega.lock; the constant must be exactly that file's sha.
    assert_eq!(PINNED_OMEGA_SHA, include_str!("../../../omega.lock").trim());
}

#[test]
fn rc_names() {
    assert_eq!(rc_name(0), "OK");
    assert_eq!(rc_name(-2), "TOO_LARGE");
    assert_eq!(rc_name(-5), "PARITY_FAIL");
    assert_eq!(rc_name(-99), "UNKNOWN");
}

#[test]
fn shape_mismatch_is_caught_before_ffi() {
    let mut c = vec![0f32; 4];
    let e = matmul_f32(2, 2, 2, &[0.0; 3], &[0.0; 4], &mut c).unwrap_err();
    assert!(matches!(e, OmegaGpuError::ShapeMismatch { what: "a", .. }));
}

#[test]
fn stub_reports_unavailable() {
    if is_native() {
        return;
    }
    let mut c = vec![0f32; 4];
    assert_eq!(
        matmul_f32(2, 2, 2, &[0.0; 4], &[0.0; 4], &mut c).unwrap_err(),
        OmegaGpuError::Unavailable
    );
}

#[test]
fn native_refuses_bad_args_without_chip() {
    if !is_native() {
        return;
    }
    let mut c = vec![0f32; 0];
    match matmul_f32(0, 16, 16, &[], &[0.0; 256], &mut c).unwrap_err() {
        OmegaGpuError::Rc { rc, name } => {
            assert_eq!(rc, -1);
            assert_eq!(name, "BAD_ARGS");
        }
        e => panic!("unexpected {e}"),
    }
}

#[test]
#[ignore = "needs the GB10 chip: run through lanes.sh queue --heavy with a native build"]
fn chip_parity_f32_and_bf16() {
    assert!(
        is_native(),
        "build with AIEN_OMEGA_DIR to link the chip library"
    );
    let shapes = [
        (16, 16, 16),
        (1, 64, 64),
        (33, 40, 24),
        (64, 256, 32),
        (1, 1024, 128),
    ];
    let mut seed = 12345u32;
    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
        // bf16 path: reference uses the bf16-rounded inputs
        let ab: Vec<u16> = a.iter().map(|&x| bf16_bits(x)).collect();
        let bb: Vec<u16> = b.iter().map(|&x| bf16_bits(x)).collect();
        let ar: Vec<f32> = ab.iter().map(|&x| bf16_val(x)).collect();
        let br: Vec<f32> = bb.iter().map(|&x| bf16_val(x)).collect();
        let want = reference(m, k, n, &ar, &br);
        let scale = (k as f32).sqrt() * 0.25 + 1e-3; // accumulation scale, inputs in [-0.5,0.5]
        for (label, got, info) in [
            {
                let mut c = vec![0f32; m * n];
                let i = matmul_bf16(m, k, n, &ab, &bb, &mut c).expect("bf16");
                ("bf16", c, i)
            },
            {
                let mut c = vec![0f32; m * n];
                let i = matmul_f32(m, k, n, &a, &b, &mut c).expect("f32");
                ("f32", c, i)
            },
        ] {
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f32, f32::max);
            println!(
                "CHIP_PARITY {label} {m}x{k}x{n} worst_abs={worst:e} scale={scale:e} chip_calls={} parity_verified={} chip={}",
                info.raw.chip_calls,
                info.raw.parity_verified,
                info.target_chip()
            );
            assert!(info.raw.parity_verified, "{label} {m}x{k}x{n}");
            assert!(
                worst < 1e-5 * scale * 10.0,
                "{label} {m}x{k}x{n} worst {worst}"
            );
        }
    }
}

/// Shape products that overflow `usize` are refused before the C call (TOO_LARGE), in debug
/// and release alike. Before abi-qual, `seq_len * num_kv_heads * head_dim` wrapped to 0 in a
/// release build, so empty cache slices passed the length check with each dimension still
/// inside u32 range; only omega's own argument checks stood between that call and an
/// out-of-bounds read.
#[test]
fn overflowing_shapes_are_refused_before_ffi() {
    let too_large = |e: OmegaGpuError| matches!(e, OmegaGpuError::Rc { rc: -2, .. });
    let q = vec![0f32; 1 << 16];
    let mut out = vec![0f32; 1 << 16];
    let e = gqa_attention_f32(&q, &[], &[], 1 << 31, 1, 1 << 17, 1 << 16, &mut out).unwrap_err();
    assert!(too_large(e.clone()), "{e:?}");
    let mut c = vec![0f32; 0];
    let e = matmul_f32(usize::MAX / 2 + 1, 2, 0, &[], &[], &mut c).unwrap_err();
    assert!(too_large(e.clone()), "{e:?}");
}
