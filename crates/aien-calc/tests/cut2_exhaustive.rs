//! Cut 2 exhaustive and cross-implementation tests: every code of every small format, the
//! sovereign-core FP8 reference, and Rust's own IEEE arithmetic as the oracle for f32.

use aien_calc::format::{decode, encode, Format};
use aien_calc::ratutil::RoundMode;
use aien_calc::Value;
use aien_inference_abi::qwen3_moe::{
    bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3fn, fp8_e4m3fn_to_f32,
};
use num_bigint::BigInt;
use num_rational::BigRational;

/// Exact rational of a finite f64 (independent of the crate under test).
fn rat_of_f64(v: f64) -> BigRational {
    assert!(v.is_finite());
    let bits = v.to_bits();
    let neg = bits >> 63 == 1;
    let e = ((bits >> 52) & 0x7ff) as i64;
    let m = bits & ((1u64 << 52) - 1);
    let (mant, exp) = if e == 0 {
        (m, -1074)
    } else {
        (m | (1u64 << 52), e - 1075)
    };
    let mut n = BigInt::from(mant);
    if neg {
        n = -n;
    }
    if exp >= 0 {
        BigRational::from_integer(n << exp as usize)
    } else {
        BigRational::new(n, BigInt::from(1u8) << (-exp) as usize)
    }
}

fn rat_of_f32(v: f32) -> BigRational {
    rat_of_f64(v as f64) // f32 -> f64 is exact
}

/// Does `got` (decoded by aien-calc) agree with the IEEE value `want`?
fn same_as_f32(got: &Value, want: f32) -> bool {
    match got {
        Value::NaN => want.is_nan(),
        Value::Inf { negative } => want.is_infinite() && (want < 0.0) == *negative,
        Value::NegZero => want == 0.0 && want.is_sign_negative(),
        Value::Num(r) => {
            !want.is_nan()
                && !want.is_infinite()
                && *r == rat_of_f32(want)
                && !(want == 0.0 && want.is_sign_negative())
        }
        _ => false,
    }
}

#[test]
fn e4m3_matches_the_sovereign_core_reference_for_all_256_codes() {
    let mut nan_codes = Vec::new();
    for code in 0u16..256 {
        let got = decode(Format::E4m3, code as u64).unwrap();
        let want = fp8_e4m3fn_to_f32(code as u8);
        assert!(
            same_as_f32(&got, want),
            "code {code:#04x}: calc {got:?} vs reference {want:?}"
        );
        if want.is_nan() {
            nan_codes.push(code);
        }
    }
    assert_eq!(nan_codes, vec![0x7f, 0xff]);
}

#[test]
fn e4m3_encode_matches_the_reference_on_a_dense_grid() {
    // Reference: saturating, round-to-nearest-even. Grid: every multiple of 2^-10 up to +-520,
    // plus every code midpoint and the f32 neighbours of each midpoint.
    let enc = |x: f32| {
        encode(
            Format::E4m3,
            &Value::Num(rat_of_f32(x)),
            Some(RoundMode::Rne),
            None,
        )
        .unwrap()
    };
    let mut checked = 0u64;
    let mut check = |x: f32| {
        let want = f32_to_fp8_e4m3fn(x) as u64;
        let got = enc(x);
        assert_eq!(
            got,
            want,
            "x = {x:?} ({:#x}): calc {got:#04x} vs reference {want:#04x}",
            x.to_bits()
        );
        checked += 1;
    };
    let mut k = -(520 * 1024i32);
    while k <= 520 * 1024 {
        check(k as f32 / 1024.0);
        k += 1;
    }
    let mags: Vec<f32> = (0u8..=0x7e).map(fp8_e4m3fn_to_f32).collect();
    for w in mags.windows(2) {
        let mid = (w[0] + w[1]) / 2.0; // exact in f32: both are short dyadic values
        for x in [
            mid,
            f32::from_bits(mid.to_bits() - 1),
            f32::from_bits(mid.to_bits() + 1),
        ] {
            check(x);
            check(-x);
        }
    }
    for x in [
        448.0f32,
        464.0,
        480.0,
        1e9,
        1e30,
        f32::MAX,
        f32::from_bits(1),
        1e-30,
    ] {
        check(x);
        check(-x);
    }
    assert!(checked > 1_000_000);
}

#[test]
fn bf16_decode_is_f32_from_bits_of_the_widened_code() {
    for code in 0u32..=0xffff {
        let got = decode(Format::Bf16, code as u64).unwrap();
        let want = f32::from_bits(code << 16);
        assert!(
            same_as_f32(&got, want),
            "bf16 {code:#06x}: {got:?} vs {want:?}"
        );
        assert_eq!(
            f32::from_bits(code << 16).to_bits(),
            bf16_to_f32(code as u16).to_bits()
        );
    }
}

/// Widen an f16 code to f32 by integer bit manipulation (oracle, independent of aien-calc).
fn f16_to_f32(code: u16) -> f32 {
    let sign = ((code as u32) & 0x8000) << 16;
    let e = ((code >> 10) & 0x1f) as u32;
    let m = (code & 0x3ff) as u32;
    if e == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (m << 13));
    }
    if e == 0 {
        // subnormal: m * 2^-24, exact in f32
        let mag = (m as f32) * f32::from_bits(0x3380_0000); // 2^-24
        return if sign != 0 { -mag } else { mag };
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

#[test]
fn f16_decode_matches_the_widened_f32_for_all_65536_codes() {
    for code in 0u32..=0xffff {
        let got = decode(Format::F16, code as u64).unwrap();
        let want = f16_to_f32(code as u16);
        assert!(
            same_as_f32(&got, want),
            "f16 {code:#06x}: {got:?} vs {want:?}"
        );
    }
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

#[test]
fn f32_decode_matches_from_bits_on_a_dense_sample() {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut codes: Vec<u32> = vec![
        0,
        1,
        2,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7fc0_0000,
        0x8000_0000,
        0x8000_0001,
        0xff80_0000,
        0xffff_ffff,
    ];
    for _ in 0..200_000 {
        codes.push((xorshift(&mut x) >> 16) as u32);
    }
    for code in codes {
        let got = decode(Format::F32, code as u64).unwrap();
        assert!(
            same_as_f32(&got, f32::from_bits(code)),
            "f32 {code:#010x}: {got:?}"
        );
    }
}

#[test]
fn f64_decode_matches_from_bits_on_a_dense_sample() {
    let mut x: u64 = 0xD1B5_4A32_D192_ED03;
    for _ in 0..100_000 {
        let bits = xorshift(&mut x);
        let v = f64::from_bits(bits);
        let got = decode(Format::F64, bits).unwrap();
        match got {
            Value::NaN => assert!(v.is_nan()),
            Value::Inf { negative } => assert!(v.is_infinite() && (v < 0.0) == negative),
            Value::NegZero => assert!(v == 0.0 && v.is_sign_negative()),
            Value::Num(r) => assert_eq!(r, rat_of_f64(v), "f64 {bits:#018x}"),
            other => panic!("unexpected {other:?}"),
        }
    }
}

#[test]
fn bits_of_val_round_trips_every_code_of_every_small_format() {
    for fmt in [
        Format::E4m3,
        Format::E5m2,
        Format::F16,
        Format::Bf16,
        Format::Int8,
        Format::Int4,
        Format::Uint8,
        Format::Uint4,
    ] {
        let n = 1u64 << fmt.width();
        let mut nan_codes = 0u64;
        for code in 0..n {
            let v = decode(fmt, code).unwrap();
            if v == Value::NaN {
                nan_codes += 1;
                assert!(
                    encode(fmt, &v, None, None).is_err(),
                    "{fmt:?} {code:#x}: NaN must be refused"
                );
                continue;
            }
            let back = encode(fmt, &v, Some(RoundMode::Rne), None).unwrap();
            assert_eq!(back, code, "{fmt:?} code {code:#x} -> {v:?} -> {back:#x}");
        }
        let expect_nan = match fmt {
            Format::E4m3 => 2,
            Format::E5m2 => 6,
            Format::F16 => 2 * 1023,
            Format::Bf16 => 2 * 127,
            _ => 0,
        };
        assert_eq!(nan_codes, expect_nan, "{fmt:?} NaN code count");
    }
}

#[test]
fn f32_and_f64_round_trip_on_a_sample() {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..100_000 {
        let r = xorshift(&mut x);
        for (fmt, code) in [(Format::F32, r & 0xffff_ffff), (Format::F64, r)] {
            let v = decode(fmt, code).unwrap();
            if v == Value::NaN {
                continue;
            }
            assert_eq!(
                encode(fmt, &v, None, None).unwrap(),
                code,
                "{fmt:?} {code:#x}"
            );
        }
    }
}

#[test]
fn f32_encode_agrees_with_rust_as_f32_across_the_range() {
    // `f64 as f32` is IEEE round-to-nearest-even, including subnormals and overflow to infinity.
    let mut x: u64 = 0x1234_5678_9ABC_DEF1;
    for _ in 0..200_000 {
        let r = xorshift(&mut x);
        // Force the exponent into [2^-160, 2^131] so we hit f32 subnormals and overflow.
        let e = 1023 - 160 + (r % 291);
        let bits = (r & 0x800f_ffff_ffff_ffff) | (e << 52);
        let v = f64::from_bits(bits);
        let want = (v as f32).to_bits() as u64;
        let got = encode(
            Format::F32,
            &Value::Num(rat_of_f64(v)),
            Some(RoundMode::Rne),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "f64 {bits:#018x} = {v:e}");
    }
}

#[test]
fn bf16_encode_agrees_with_the_reference_f32_to_bf16() {
    let mut x: u64 = 0xCAFE_F00D_1234_5678;
    for _ in 0..200_000 {
        let v = f32::from_bits((xorshift(&mut x) >> 8) as u32);
        if !v.is_finite() {
            continue;
        }
        let want = f32_to_bf16(v) as u64;
        let got = encode(
            Format::Bf16,
            &Value::Num(rat_of_f32(v)),
            Some(RoundMode::Rne),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "f32 {:#010x}", v.to_bits());
    }
}
