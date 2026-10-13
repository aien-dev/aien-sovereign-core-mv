//! Cut 4 tests: units, strict prefixes, `->`, shape helpers, and the own30b design section 0 numbers.
//! Expected values are derived by hand (and the section 0 chain re-derived independently with node BigInt).

use aien_calc::eval;

fn ev(expr: &str) -> String {
    match eval(expr, 20) {
        Ok(o) => o
            .canonical_value()
            .unwrap_or_else(|| panic!("`{expr}` refused: {:?}", o.refusal())),
        Err(e) => panic!("`{expr}` failed to parse: {e}"),
    }
}

fn refusal(expr: &str) -> String {
    match eval(expr, 20) {
        Ok(o) => o
            .refusal()
            .unwrap_or_else(|| panic!("`{expr}` should be refused, got {:?}", o.canonical_value()))
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
fn literals_and_conversions() {
    check(&[
        ("98 KB -> B", "98000/1 B"),
        ("98 kB -> B", "98000/1 B"),
        ("3.2 GiB -> B", "17179869184/5 B"), // 3.2 * 2^30 = 3435973836.8
        ("1 KiB -> B", "1024/1 B"),
        ("1 MiB -> KiB", "1024/1 KiB"),
        ("273 GB/s -> GB/s", "273/1 GB/s"),
        ("273 GB/s -> MB/s", "273000/1 MB/s"),
        ("0.5 s -> ms", "500/1 ms"),
        ("1 us -> ns", "1000/1 ns"),
        ("3 bit -> B", "3/8 B"),
        ("1 B -> bit", "8/1 bit"),
        ("2 TFLOP/s -> GFLOP/s", "2000/1 GFLOP/s"),
        ("1 Hz * 3 s", "3/1"), // dimensionless result is a plain number
        ("2 B/param * 4 param -> B", "8/1 B"),
    ]);
}

#[test]
fn gb_and_gib_differ_by_exactly_the_binary_factor() {
    assert_eq!(ev("1 GiB / 1 GB"), "2097152/1953125"); // 2^30 / 10^9 reduced
    assert_eq!(ev("(1 GiB - 1 GB) -> B"), "73741824/1 B");
    assert_eq!(ev("1 GB == 1000 MB"), "true");
    assert_eq!(ev("1 KiB == 1024 B"), "true");
    assert_eq!(ev("1 GB == 1 GiB"), "false");
    assert_eq!(ev("1 ms == 1000 us"), "true");
}

#[test]
fn quantities_do_arithmetic_with_dimension_tracking() {
    check(&[
        ("2 B + 3 B -> B", "5/1 B"),
        ("5 B - 7 B -> B", "-2/1 B"),
        ("2 * 3 B -> B", "6/1 B"),
        ("(6 B) / 2 -> B", "3/1 B"),
        ("-(3 B) -> B", "-3/1 B"),
        ("abs(-3 B) -> B", "3/1 B"),
        ("min(3 GB, 2 GiB) -> B", "2147483648/1 B"), // 2 GiB = 2147483648 B < 3 GB
        ("max(3 GB, 2 GiB) -> B", "3000000000/1 B"),
        ("(2 B)^2 -> B*B", "4/1 B*B"),
        ("(4 B)^2 / (2 B) -> B", "8/1 B"),
        ("1 / (2 s) -> Hz", "1/2 Hz"),
        ("3 tok / (2 s) -> tok/s", "3/2 tok/s"),
        ("2 GB < 3 GB", "true"),
        ("2 GB >= 3 GB", "false"),
    ]);
}

#[test]
fn mismatches_are_refusals_never_guesses() {
    for (expr, needle) in [
        ("1 B + 1 s", "unit mismatch"),
        ("1 B + 1", "unit mismatch"),
        ("1 - 1 B", "unit mismatch"),
        ("1 GB -> s", "unit mismatch"),
        ("5 -> B", "plain number"),
        ("2 B * 3 B -> B", "unit mismatch"),
        ("(1 B) < (1 s)", "unit mismatch"),
        ("1 B == 1", "unit mismatch"),
        ("(1 B) % 3", "plain number"),
        ("1 B << 2", "plain number"),
        ("2 ^ (1 B)", "plain number"),
        ("floor(1 B)", "number"),
        ("min(1 B, 1 s)", "unit mismatch"),
    ] {
        let r = refusal(expr);
        assert!(
            r.contains(needle),
            "`{expr}` refusal `{r}` should mention `{needle}`"
        );
    }
    // prefixes are strict: milli-bytes and Kibi-flops are not units
    for expr in ["1 mB", "1 Kb", "1 KiFLOP", "1 gB", "1 GIB"] {
        assert!(eval(expr, 20).is_err(), "`{expr}` should not parse");
    }
}

#[test]
fn receipts_name_the_prefixes_used() {
    let body =
        |e: &str| String::from_utf8(eval(e, 20).unwrap().receipt().canonical_bytes()).unwrap();
    assert!(
        body("3.2 GiB -> B").contains("\"prefixes\":[\"GiB:binary(2^30)\"]"),
        "{}",
        body("3.2 GiB -> B")
    );
    assert!(body("273 GB/s -> GB/s").contains("\"prefixes\":[\"GB:decimal(10^9)\"]"));
    assert!(body("1 ms == 1000 us")
        .contains("\"prefixes\":[\"ms:decimal(10^-3)\",\"us:decimal(10^-6)\"]"));
    assert!(body("1 KB -> B").contains("\"KB:decimal(10^3)\""));
    assert!(
        !body("1/3").contains("prefixes"),
        "plain receipts keep the old layout"
    );
    assert!(body("98 KB -> B").contains("\"kind\":\"quantity\""));
    assert!(body("98 KB -> B").contains("\"unit\":\"B\""));
}

#[test]
fn shape_helpers() {
    check(&[
        ("params(2048 x 4096)", "8388608/1 param"),
        ("params(5)", "5/1 param"),
        ("params(2 x 3 x 4)", "24/1 param"),
        ("matmul_flops(1, 2048, 4096)", "16777216/1 FLOP"), // 2 * 1 * 2048 * 4096
        ("bytes(params(2048 x 4096), bf16)", "16777216/1 B"),
        ("bytes(params(2048 x 4096), e4m3)", "8388608/1 B"),
        ("bytes(params(8), int4)", "4/1 B"),
        ("bytes(1, int4)", "1/2 B"),
        ("bytes(1, f32)", "4/1 B"),
        ("kv_bytes(48, 4, 128, bf16, 1)", "98304/1 B"), // 48 * 4 * 128 * 2 (K and V) * 2 B
        (
            "kv_bytes(48, 4, 128, bf16, 32768) -> GB",
            "6291456/1953125 GB",
        ), // 3221225472 / 10^9
    ]);
    assert!(refusal("params(0 x 5)").contains("positive"));
    assert!(refusal("params(1/2)").contains("integer"));
    assert!(refusal("bytes(5 B, bf16)").contains("count"));
    assert!(refusal("2048 x 4096").contains("params"));
    assert!(refusal("matmul_flops(1, 2)").contains("arguments"));
}

#[test]
fn own30b_design_section_0_reproduced_from_its_inputs() {
    // Shapes from the design: 48 layers, hidden 2048, 32 q heads x 128, 4 kv heads x 128, 128 experts,
    // top-8, moe_intermediate 768, vocab 151936.
    let attention = "48 * (params(2048 x 4096) + 2 * params(2048 x 512) + params(4096 x 2048))";
    assert_eq!(ev(attention), "905969664/1 param"); // design rounds to 0.907 B; exact is 0.906 B
    let experts = "48 * 8 * (params(2048 x 1536) + params(768 x 2048))";
    assert_eq!(ev(experts), "1811939328/1 param"); // "1.81 B"
    assert_eq!(ev("48 * params(128 x 2048)"), "12582912/1 param"); // "12.6 M"
    assert_eq!(ev("params(151936 x 2048)"), "311164928/1 param"); // "311 M"
    let total =
        format!("({attention}) + ({experts}) + 48 * params(128 x 2048) + params(151936 x 2048)");
    assert_eq!(ev(&total), "3041656832/1 param"); // "about 3.04 B active params"
                                                  // 3.04 B param * 2 B/param = 6083313664 B, the design's "6.1 GB/token"
    assert_eq!(ev(&format!("bytes({total}, bf16)")), "6083313664/1 B");
    assert_eq!(
        ev(&format!("bytes({total}, bf16) -> GB")),
        "11881472/1953125 GB"
    ); // 6.0833 GB
       // KV cache: 48 x 4 x 128 x 2 x 2 B = 98304 B, "98 KB/token", and 3.2 GB at 32k context
    assert_eq!(ev("kv_bytes(48, 4, 128, bf16, 1) -> KB"), "12288/125 KB"); // 98.304 KB
    assert_eq!(
        ev("kv_bytes(48, 4, 128, bf16, 32768) -> GB"),
        "6291456/1953125 GB"
    ); // 3.2212 GB
       // tokens per second at 273 GB/s: the design's 44.7 comes from the rounded 6.1 GB/token
    assert_eq!(ev("1 / (6.1 GB/tok / 273 GB/s) -> tok/s"), "2730/61 tok/s"); // 44.754...
    assert_eq!(ev("273 GB/s / (6.1 GB/tok) -> tok/s"), "2730/61 tok/s");
    // with the exact bytes per token the figure is 44.877 tok/s
    let exact = format!("273 GB/s / (bytes({total}, bf16) / 1 tok) -> tok/s");
    assert_eq!(ev(&exact), "533203125/11881472 tok/s");
    // time to read one token's weights at 273 GB/s
    assert_eq!(
        ev("(3.04e9 param * 2 B/param) / (273 GB/s) -> ms"),
        "6080/273 ms"
    ); // 22.27 ms
}

#[test]
fn decimal_text_carries_the_unit() {
    let o = eval("1 / (6.1 GB/tok / 273 GB/s) -> tok/s", 6).unwrap();
    assert!(o.text().contains("44.7541 tok/s"), "{}", o.text());
    let o = eval("98 KB -> B", 20).unwrap();
    assert!(o.text().contains("98000 B"), "{}", o.text());
}

#[test]
fn unit_expressions_round_trip_through_canonical_form() {
    for e in [
        "273 GB/s",
        "(3.04e9 param * 2 B/param) / (273 GB/s) -> ms",
        "params(2048 x 4096)",
        "98 KB -> B",
    ] {
        let a = aien_calc::parser::parse(e).unwrap();
        assert!(a.canonical().contains('('), "{}", a.canonical());
    }
}
