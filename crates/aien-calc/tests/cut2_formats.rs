//! Cut 2 tests: number-format lab. Expected values are derived by hand from the format
//! definitions (bit layouts and rounding rules), never from the code under test.

use aien_calc::format::{decode, encode, Format};
use aien_calc::{eval, Value};
use num_bigint::BigInt;
use num_rational::BigRational;

fn ev(expr: &str) -> String {
    match eval(expr, 20) {
        Ok(o) => o
            .canonical_value()
            .unwrap_or_else(|| panic!("`{expr}` was refused: {:?}", o.refusal())),
        Err(e) => panic!("`{expr}` failed to parse: {e}"),
    }
}

fn refusal(expr: &str) -> String {
    match eval(expr, 20) {
        Ok(o) => o
            .refusal()
            .unwrap_or_else(|| {
                panic!(
                    "`{expr}` should have been refused, got {:?}",
                    o.canonical_value()
                )
            })
            .to_string(),
        Err(e) => panic!("`{expr}` failed to parse: {e}"),
    }
}

fn check(cases: &[(&str, &str)]) {
    for (expr, want) in cases {
        assert_eq!(&ev(expr), want, "expression `{expr}`");
    }
}

#[test]
fn val_known_codes() {
    check(&[
        ("val(e4m3, 0x38)", "1/1"), // 0 0111 000
        ("val(e4m3, 0xB8)", "-1/1"),
        ("val(e4m3, 0x7e)", "448/1"), // 1.75 * 2^8
        ("val(e4m3, 0x01)", "1/512"), // 2^-9
        ("val(e4m3, 0x07)", "7/512"),
        ("val(e4m3, 0x08)", "1/64"), // smallest normal 2^-6
        ("val(e4m3, 0x80)", "-0"),
        ("val(e4m3, 0x00)", "0/1"),
        ("val(e4m3, 0x7f)", "nan"),
        ("val(e4m3, 0xff)", "nan"),
        ("val(e5m2, 0x7b)", "57344/1"), // 1.75 * 2^15
        ("val(e5m2, 0x7c)", "+inf"),
        ("val(e5m2, 0xfc)", "-inf"),
        ("val(e5m2, 0x7d)", "nan"),
        ("val(e5m2, 0x01)", "1/65536"), // 2^-16
        ("val(f16, 0x7bff)", "65504/1"),
        ("val(f16, 0x0001)", "1/16777216"), // 2^-24
        ("val(f16, 0x3c00)", "1/1"),
        ("val(bf16, 0x3f80)", "1/1"),
        ("val(bf16, 0xc000)", "-2/1"),
        ("val(f32, 0x3f800000)", "1/1"),
        (
            "val(f32, 0x00000001)",
            "1/713623846352979940529142984724747568191373312",
        ), // 2^-149
        ("val(f64, 0x3ff0000000000000)", "1/1"),
        ("val(f64, 0xc000000000000000)", "-2/1"),
        ("val(int8, 0xff)", "-1/1"),
        ("val(int8, 0x80)", "-128/1"),
        ("val(int4, 8)", "-8/1"),
        ("val(int4, 7)", "7/1"),
        ("val(uint4, 15)", "15/1"),
        ("val(uint8, 255)", "255/1"),
    ]);
    assert!(refusal("val(int8, 256)").contains("range"));
    assert!(refusal("val(e4m3, -1)").contains("range"));
    assert!(refusal("val(e4m3, 1/2)").contains("integer"));
}

#[test]
fn class_vectors() {
    check(&[
        ("class(f32, 2^-126)", "normal"),
        ("class(f32, 2^-127)", "subnormal"),
        ("class(f32, 2^-149)", "subnormal"),
        ("class(f32, 0)", "zero"),
        ("class(f32, 1)", "normal"),
        ("class(e4m3, val(e4m3, 0x01))", "subnormal"),
        ("class(e4m3, val(e4m3, 0x08))", "normal"),
        ("class(e4m3, val(e4m3, 0x80))", "zero"),
        ("class(e4m3, val(e4m3, 0x7f))", "nan"),
        ("class(e5m2, val(e5m2, 0x7c))", "inf"),
        ("class(bf16, 2^-126)", "normal"),
        ("class(bf16, 2^-127)", "subnormal"),
        ("class(int8, 5)", "normal"),
        ("class(int8, 0)", "zero"),
        ("class(e4m3, val(e4m3, 0x01)) == subnormal", "true"),
        ("class(e4m3, val(e4m3, 0x7f)) != nan", "false"),
    ]);
    assert!(refusal("class(f32, 2^-150)").contains("representable"));
    assert!(refusal("class(f32, 1/3)").contains("representable"));
}

#[test]
fn bits_known_encodings() {
    check(&[
        ("bits(f32, 0.1)", "1036831949/1"),          // 0x3DCCCCCD
        ("bits(f32, 1/3)", "1051372203/1"),          // 0x3EAAAAAB
        ("bits(f16, 0.1)", "11878/1"),               // 0x2E66
        ("bits(bf16, 0.1)", "15821/1"),              // 0x3DCD (0x3DCC + carry)
        ("bits(f64, 0.1)", "4591870180066957722/1"), // 0x3FB999999999999A
        ("bits(f32, 0.1, rtz)", "1036831948/1"),     // 0x3DCCCCCC
        ("bits(f32, 0.1, rup)", "1036831949/1"),
        ("bits(f32, 0.1, rdown)", "1036831948/1"),
        ("bits(f32, -0.1, rup)", "3184315596/1"), // -0.1 rup = toward zero = 0x80000000|0x3DCCCCCC
        ("bits(f32, -0.1, rdown)", "3184315597/1"),
        ("bits(e4m3, 1)", "56/1"),   // 0x38
        ("bits(e4m3, -1)", "184/1"), // 0xB8
        ("bits(e4m3, 0)", "0/1"),
        ("bits(e5m2, 1)", "60/1"),    // 0x3C
        ("bits(f16, 1)", "15360/1"),  // 0x3C00
        ("bits(bf16, 1)", "16256/1"), // 0x3F80
    ]);
}

#[test]
fn rne_tie_vectors() {
    check(&[
        // e4m3: 1.0 = 0x38, 1.125 = 0x39, 1.25 = 0x3A. Ties go to the even code.
        ("bits(e4m3, 17/16)", "56/1"),      // tie 1.0 / 1.125 -> 0x38
        ("bits(e4m3, 19/16)", "58/1"),      // tie 1.125 / 1.25 -> 0x3A
        ("bits(e4m3, 17/16, rna)", "57/1"), // away from zero -> 0x39
        ("bits(e4m3, 17/16, rtz)", "56/1"),
        ("bits(e4m3, 1.01, rup)", "57/1"),
        ("bits(e4m3, 1.1, rdown)", "56/1"), // 1.1 lies between 1.0 and 1.125
        ("bits(e4m3, -1.01, rup)", "184/1"), // toward +inf = toward zero for negatives: -1.0 = 0xB8
        ("bits(e4m3, -1.01, rdown)", "185/1"), // -1.125 = 0xB9
        // e4m3 subnormal ties: 2^-10 is half of the smallest subnormal 2^-9.
        ("bits(e4m3, 2^-10)", "0/1"), // tie between code 0 and code 1 -> even 0
        ("bits(e4m3, 3 * 2^-10)", "2/1"), // tie between code 1 and code 2 -> even 2
        ("bits(e4m3, 2^-11)", "0/1"),
        ("bits(e4m3, -2^-11)", "128/1"), // underflow keeps the sign: -0
        ("round(e4m3, -2^-11)", "-0"),
        // f16: 1 + 2^-11 is a tie between 0x3C00 and 0x3C01.
        ("bits(f16, 2049/2048)", "15360/1"),
        ("bits(f16, 2051/2048)", "15362/1"),
        // f16 overflow edge: max is 65504, the next step would be 65536, the tie point is 65520.
        ("bits(f16, 65519)", "31743/1"),
        ("bits(f16, 65520)", "31744/1"), // ties-to-even rounds to 65536, which overflows to +inf
        ("bits(f16, 65520, rtz)", "31743/1"),
        ("bits(f16, 65520, rdown)", "31743/1"),
        ("bits(f16, 65520, rup)", "31744/1"),
        ("bits(f16, -65520)", "64512/1"),
        ("bits(f16, -65520, rup)", "64511/1"),
        // bf16: 1 + 2^-8 is a tie between 0x3F80 and 0x3F81.
        ("bits(bf16, 257/256)", "16256/1"),
        ("bits(bf16, 259/256)", "16258/1"),
        ("round(e4m3, 17/16)", "1/1"),
        ("round(e4m3, 1.1, rup)", "9/8"),
    ]);
}

#[test]
fn e4m3_saturates_at_448() {
    check(&[
        ("bits(e4m3, 448)", "126/1"),
        ("bits(e4m3, 449)", "126/1"),
        ("bits(e4m3, 464)", "126/1"),
        ("bits(e4m3, 10^9)", "126/1"),
        ("bits(e4m3, -10^9)", "254/1"),
        ("round(e4m3, 500)", "448/1"),
        ("round(e4m3, -500)", "-448/1"),
        ("bits(e4m3, 10^9, saturate)", "126/1"),
    ]);
    assert!(refusal("bits(e4m3, 1000, refuse)").contains("overflow"));
    assert!(refusal("bits(e4m3, 1000, inf)").contains("no infinity"));
}

#[test]
fn ieee_overflow_goes_to_infinity_or_max_by_mode() {
    check(&[
        ("bits(e5m2, 10^9)", "124/1"),           // 0x7C = +inf
        ("bits(e5m2, 10^9, saturate)", "123/1"), // 0x7B = 57344
        ("bits(e5m2, 10^9, rtz)", "123/1"),
        ("bits(e5m2, 10^9, rtz, inf)", "124/1"),
        ("bits(e5m2, 57344)", "123/1"),
        ("bits(e5m2, 61439)", "123/1"),
        ("bits(e5m2, 61440)", "124/1"), // tie 57344 / 65536 -> even -> overflow
        ("bits(f32, 2^128)", "2139095040/1"), // 0x7F800000
        ("bits(f32, 2^128, saturate)", "2139095039/1"), // 0x7F7FFFFF
        ("bits(f32, 2^128 - 2^104)", "2139095039/1"), // exactly the max finite f32 stays finite
        ("bits(f32, 2^128 - 2^103)", "2139095040/1"), // half an ulp above max: ties-to-even overflows
    ]);
    assert!(refusal("bits(f32, 2^128, refuse)").contains("overflow"));
}

#[test]
fn integer_formats_round_and_saturate() {
    check(&[
        ("bits(int8, 3.5)", "4/1"),
        ("bits(int8, 2.5)", "2/1"),
        ("bits(int8, -2.5, rna)", "253/1"), // -3 = 0xFD
        ("bits(int8, -2.5, rtz)", "254/1"), // -2 = 0xFE
        ("bits(int8, 200)", "127/1"),
        ("bits(int8, -200)", "128/1"),
        ("bits(uint8, -5)", "0/1"),
        ("bits(uint8, 300)", "255/1"),
        ("bits(int4, 7.9)", "7/1"),
        ("round(int4, -9)", "-8/1"),
        ("round(int4, 3.5, rdown)", "3/1"),
        ("round(int4, 3.5, rup)", "4/1"),
    ]);
    assert!(refusal("bits(int8, 200, refuse)").contains("overflow"));
    assert!(refusal("bits(int8, 200, inf)").contains("no infinity"));
}

#[test]
fn nan_is_refused_not_converted() {
    assert!(refusal("bits(e4m3, val(e4m3, 0x7f))").contains("NaN"));
    assert!(refusal("bits(f32, val(f32, 0x7fc00000))").contains("NaN"));
    assert!(refusal("round(e5m2, val(e5m2, 0x7d))").contains("NaN"));
    assert!(refusal("val(e4m3, 0x7f) + 1").contains("NaN"));
    assert!(refusal("val(e4m3, 0x7f) == 1").contains("NaN"));
    // infinity round-trips, but is not arithmetic
    assert_eq!(ev("bits(e5m2, val(e5m2, 0x7c))"), "124/1");
    assert!(refusal("val(e5m2, 0x7c) + 1").contains("infinity"));
}

#[test]
fn ulp_vectors() {
    check(&[
        ("ulp(e4m3, 1)", "1/8"),
        ("ulp(e4m3, 0)", "1/512"),
        ("ulp(e4m3, 448)", "32/1"),
        ("ulp(e4m3, 1/16)", "1/128"),
        ("ulp(e4m3, 1/128)", "1/512"), // subnormal range: fixed spacing 2^-9
        ("ulp(f32, 1)", "1/8388608"),
        ("ulp(f16, 1)", "1/1024"),
        ("ulp(bf16, 1)", "1/128"),
        ("ulp(f64, 1)", "1/4503599627370496"),
        ("ulp(int8, 5)", "1/1"),
    ]);
    assert!(refusal("ulp(e4m3, 1000)").contains("range"));
}

#[test]
fn next_prev_vectors() {
    check(&[
        ("next(e4m3, 1)", "9/8"),
        ("prev(e4m3, 1)", "15/16"),
        ("next(e4m3, 0)", "1/512"),
        ("next(e4m3, val(e4m3, 0x80))", "1/512"),
        ("next(e4m3, -1/512)", "-0"),
        ("prev(e4m3, 1/512)", "0/1"),
        ("prev(e4m3, 0)", "-1/512"),
        ("next(e5m2, 57344)", "+inf"),
        ("prev(e5m2, val(e5m2, 0x7c))", "57344/1"),
        ("next(e5m2, val(e5m2, 0xfc))", "-57344/1"),
        ("next(f32, 1)", "8388609/8388608"),
        ("prev(f32, 1)", "16777215/16777216"),
        ("next(int8, -1)", "0/1"),
        ("next(uint4, 14)", "15/1"),
        ("prev(int8, 0)", "-1/1"),
    ]);
    assert!(refusal("next(e4m3, 448)").contains("no infinity"));
    assert!(refusal("prev(e4m3, -448)").contains("no infinity"));
    assert!(refusal("next(int8, 127)").contains("largest"));
    assert!(refusal("prev(int8, -128)").contains("smallest"));
    assert!(refusal("prev(uint4, 0)").contains("smallest"));
    assert!(refusal("next(e4m3, 447)").contains("representable"));
}

#[test]
fn ftz_and_daz_flush_subnormals_only() {
    check(&[
        ("ftz(f32, 2^-127)", "0/1"),
        ("ftz(f32, -2^-127)", "-0"),
        (
            "ftz(f32, 2^-126)",
            "1/85070591730234615865843651857942052864",
        ),
        ("daz(f32, 2^-149)", "0/1"),
        ("ftz(f32, 1)", "1/1"),
        ("ftz(f32, 1/3)", "1/3"),
        ("ftz(e4m3, 1/512)", "0/1"), // e4m3 smallest normal is 2^-6
        ("ftz(e4m3, 1/64)", "1/64"),
        ("ftz(f32, val(bf16, 0x0010))", "0/1"), // bf16 0x0010 = 2^-129, subnormal as f32
        (
            "daz(bf16, val(bf16, 0x0080))",
            "1/85070591730234615865843651857942052864",
        ), // 2^-126 is normal
    ]);
    assert!(refusal("ftz(int8, 3)").contains("float"));
}

#[test]
fn fields_vectors() {
    check(&[
        ("fields(e4m3, 0xB9)", "sign=1 exp=7 mant=1"),
        ("fields(f32, 0x3f800000)", "sign=0 exp=127 mant=0"),
        ("fields(e5m2, 0x7c)", "sign=0 exp=31 mant=0"),
        ("fields(f16, 0x0001)", "sign=0 exp=0 mant=1"),
    ]);
    assert!(refusal("fields(int8, 1)").contains("float"));
}

#[test]
fn arithmetic_on_decoded_values_stays_exact() {
    // negative zero is an ordinary zero in arithmetic and comparisons
    assert_eq!(ev("val(e4m3, 0x80) + 1"), "1/1");
    assert_eq!(ev("val(e4m3, 0x80) == 0"), "true");
    assert_eq!(ev("-val(e4m3, 0x80)"), "0/1");
    assert_eq!(ev("val(f16, 0x3c01) - 1"), "1/1024");
    assert_eq!(ev("val(e4m3, 0x38) + val(e4m3, 0x39)"), "17/8");
    assert_eq!(ev("val(f32, 0x3f800000) / 3 * 3"), "1/1");
}

#[test]
fn argument_errors_are_refusals() {
    assert!(refusal("val(e4m3)").contains("arguments"));
    assert!(refusal("val(1, 2)").contains("format"));
    assert!(refusal("bits(e4m3, 1, rne, rne)").contains("twice"));
    assert!(refusal("bits(e4m3, 1, 5)").contains("mode"));
    assert!(refusal("bits(e4m3, true)").contains("number"));
    assert!(refusal("class(e4m3)").contains("arguments"));
    assert!(refusal("val(nosuchfmt, 1)").contains("unknown name"));
}

#[test]
fn format_table_is_complete_and_named() {
    let names: Vec<&str> = Format::ALL.iter().map(|f| f.name()).collect();
    assert_eq!(
        names,
        ["e4m3", "e5m2", "f16", "bf16", "f32", "f64", "int8", "int4", "uint8", "uint4"]
    );
    let widths: Vec<u32> = Format::ALL.iter().map(|f| f.width()).collect();
    assert_eq!(widths, [8, 8, 16, 16, 32, 64, 8, 4, 8, 4]);
    for f in Format::ALL {
        assert_eq!(Format::from_name(f.name()), Some(f));
    }
    assert_eq!(Format::from_name("fp8"), None);
}

#[test]
fn library_api_matches_expressions() {
    let v = decode(Format::E4m3, 0x7e).unwrap();
    assert_eq!(v, Value::Num(BigRational::from_integer(BigInt::from(448))));
    assert_eq!(encode(Format::E4m3, &v, None, None).unwrap(), 0x7e);
    assert!(decode(Format::E4m3, 256).is_err());
}
