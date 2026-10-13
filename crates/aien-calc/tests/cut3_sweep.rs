//! Cut 3 tests: the sweep tool, with today's FP8 finding as the golden test.
//!
//! Hand derivation of the golden numbers (e4m3, 256 codes):
//! * NaN codes are 0x7F and 0xFF, so 254 codes are considered.
//! * The bf16 image of code b is ((b&0x80)<<8) | ((b&0x7F)<<4). Its exponent field is the e4m3
//!   exponent field (4 bits) shifted into bf16's 8-bit field, so bf16 value = x * 2^-120 exactly.
//! * e4m3 subnormals are codes 0x01..0x07 (and 0x81..0x87); their bf16 images have exponent field 0
//!   and value m * 2^-129 < 2^-126, i.e. they are f32-subnormal before the 2^120 rescale.
//!   That is 7 magnitudes, so 7 codes among the positive half and 14 over both signs.
//! * Without the rescale, only the two zeros (0x00, 0x80) still equal x, so 252 of 254 differ.

use aien_calc::sweep::sweep;
use std::process::Command;

const W: &str = "(($b & 0x80) << 8) | (($b & 0x7f) << 4)";

fn got_expr() -> String {
    format!("val(bf16, {W}) * 2^120")
}

fn flag_expr() -> String {
    format!("class(f32, val(bf16, {W})) == subnormal")
}

const NOT_NAN: &str = "class(e4m3, $x) != nan";

fn hex_codes(codes: &[u64]) -> Vec<String> {
    codes.iter().map(|c| format!("{c:#04x}")).collect()
}

#[test]
fn golden_fp8_bit_trick_is_exact_for_all_254_numbers() {
    let out = sweep("e4m3", "$x", &got_expr(), Some(&flag_expr()), Some(NOT_NAN)).unwrap();
    assert_eq!(out.total(), 256);
    assert_eq!(out.considered(), 254);
    assert_eq!(out.equal(), 254);
    assert_eq!(out.mismatch_count(), 0);
    assert!(out.refused().is_empty());
    // flag: both signs of the 7 subnormal magnitudes
    let want: Vec<u64> = (0x01..=0x07).chain(0x81..=0x87).collect();
    assert_eq!(out.flagged(), want.as_slice());
    let text = out.text();
    assert!(
        text.contains("equal 254/254") || text.contains("equal   254/254"),
        "{text}"
    );
    assert!(
        text.contains("flag true: 14 codes [0x01..0x07, 0x81..0x87]"),
        "{text}"
    );
    assert_eq!(
        text.lines().last().unwrap(),
        format!("receipt {}", out.receipt().id())
    );
}

#[test]
fn golden_positive_half_flags_exactly_codes_0x01_to_0x07() {
    // `$b < 0x7f` keeps the positive finite numbers (0x7F is NaN) and drops every negative code.
    let out = sweep(
        "e4m3",
        "$x",
        &got_expr(),
        Some(&flag_expr()),
        Some("$b < 0x7f"),
    )
    .unwrap();
    assert_eq!(out.considered(), 127);
    assert_eq!(out.equal(), 127);
    assert_eq!(
        hex_codes(out.flagged()),
        ["0x01", "0x02", "0x03", "0x04", "0x05", "0x06", "0x07"]
    );
    assert!(
        out.text().contains("flag true: 7 codes [0x01..0x07]"),
        "{}",
        out.text()
    );
}

#[test]
fn nan_codes_are_excluded_by_where_and_not_silently_dropped() {
    let out = sweep("e4m3", "$x", "$x", None, Some(NOT_NAN)).unwrap();
    assert_eq!(out.total() - out.considered(), 2);
    let none = sweep("e4m3", "$x", "$x", None, None).unwrap();
    assert_eq!(none.considered(), 256);
    assert_eq!(none.equal(), 256); // NaN equals NaN when compared as sweep results
}

#[test]
fn negative_control_dropping_the_rescale_breaks_every_nonzero_code() {
    let broken = format!("val(bf16, {W})");
    let out = sweep("e4m3", "$x", &broken, None, Some(NOT_NAN)).unwrap();
    assert_eq!(out.considered(), 254);
    assert_eq!(out.mismatch_count(), 252);
    assert_eq!(out.equal(), 2); // the two zeros
    assert!(out
        .mismatches()
        .iter()
        .all(|m| m.code != 0x00 && m.code != 0x80));
    // code 0x38 is 1.0; its bf16 image is 0x0380 = 2^-120
    let m = out.mismatches().iter().find(|m| m.code == 0x38).unwrap();
    assert_eq!(m.expect, "1/1");
    assert_eq!(m.got, "1/1329227995784915872903807060280344576");
}

#[test]
fn negative_control_off_by_one_ulp_breaks_all_254() {
    let out = sweep("e4m3", "$x", "$x + 1/512", None, Some(NOT_NAN)).unwrap();
    assert_eq!(out.mismatch_count(), 254);
    assert_eq!(out.equal(), 0);
    let m = out.mismatches().iter().find(|m| m.code == 0x38).unwrap();
    assert_eq!((m.expect.as_str(), m.got.as_str()), ("1/1", "513/512"));
}

#[test]
fn ftz_flag_reports_the_seven_values_a_flush_would_zero() {
    let w = format!("val(bf16, {W})");
    let flag = format!("ftz(f32, {w}) != {w}");
    let out = sweep("e4m3", "$x", &got_expr(), Some(&flag), Some("$b < 0x7f")).unwrap();
    assert_eq!(
        hex_codes(out.flagged()),
        ["0x01", "0x02", "0x03", "0x04", "0x05", "0x06", "0x07"]
    );
    // and what the flush would do to them: every one becomes zero
    let zeroed = sweep(
        "e4m3",
        "0",
        &format!("ftz(f32, {w})"),
        None,
        Some("$b >= 1 & $b <= 7"),
    );
    // `&` on booleans is refused, so that where clause must fail loudly rather than pass silently
    assert!(zeroed.is_err());
    let zeroed = sweep(
        "e4m3",
        "0",
        &format!("ftz(f32, {w})"),
        None,
        Some("($b - 1) < 7"),
    )
    .unwrap();
    assert_eq!(zeroed.considered(), 8); // $b - 1 < 7 holds for codes 0..=7
    assert_eq!(zeroed.equal(), 8);
}

#[test]
fn per_code_refusals_are_counted_not_hidden() {
    let out = sweep("e4m3", "$x", "1 / (1 / $x)", None, None).unwrap();
    // 0x00 and 0x80 divide by zero; 0x7F and 0xFF are NaN.
    let codes: Vec<u64> = out.refused().iter().map(|(c, _)| *c).collect();
    assert_eq!(codes, vec![0x00, 0x7f, 0x80, 0xff]);
    assert!(out.refused()[0].1.contains("division by zero"));
    assert!(out.refused()[1].1.contains("NaN"));
    assert_eq!(out.equal(), 252);
    assert_eq!(
        out.equal() + out.mismatch_count() + out.refused().len() as u64,
        out.considered()
    );
}

#[test]
fn integer_formats_sweep_with_x_as_the_integer() {
    let out = sweep("int4", "$x + $x", "$x * 2", None, None).unwrap();
    assert_eq!((out.total(), out.equal()), (16, 16));
    let out = sweep("uint4", "$x", "$b", None, None).unwrap();
    assert_eq!(out.equal(), 16);
    let out = sweep("int4", "$x", "$b", None, None).unwrap();
    assert_eq!(out.equal(), 8); // codes 0..=7 agree; 8..=15 are negative numbers
    assert_eq!(out.mismatch_count(), 8);
}

#[test]
fn full_16_bit_sweeps_run_and_round_trip() {
    let out = sweep(
        "bf16",
        "$x",
        "val(bf16, bits(bf16, $x))",
        None,
        Some("class(bf16, $x) != nan"),
    )
    .unwrap();
    assert_eq!(out.total(), 65536);
    assert_eq!(out.considered(), 65536 - 254);
    assert_eq!(out.equal(), 65536 - 254);
    let out = sweep(
        "f16",
        "$x",
        "val(f16, bits(f16, $x))",
        None,
        Some("class(f16, $x) != nan"),
    )
    .unwrap();
    assert_eq!(out.considered(), 65536 - 2046);
    assert_eq!(out.equal(), out.considered());
}

#[test]
fn big_formats_are_refused() {
    for f in ["f32", "f64"] {
        let e = sweep(f, "$x", "$x", None, None).err().unwrap();
        assert!(e.to_string().contains("refused"), "{e}");
    }
    assert!(sweep("nosuch", "$x", "$x", None, None).is_err());
}

#[test]
fn malformed_expressions_and_bad_where_are_errors() {
    assert!(sweep("e4m3", "$x +", "$x", None, None).is_err());
    assert!(sweep("e4m3", "$x", "$x", None, Some("$b + 1")).is_err()); // where must be a boolean
    assert!(
        sweep("e4m3", "$x", "$x", Some("$b"), None)
            .unwrap()
            .refused()
            .len()
            == 256
    ); // flag must be boolean
}

#[test]
fn table_digest_and_receipt_are_deterministic_and_input_sensitive() {
    let a = sweep("e4m3", "$x", &got_expr(), Some(&flag_expr()), Some(NOT_NAN)).unwrap();
    let b = sweep("e4m3", "$x", &got_expr(), Some(&flag_expr()), Some(NOT_NAN)).unwrap();
    assert_eq!(a.table_digest(), b.table_digest());
    assert_eq!(a.receipt().id(), b.receipt().id());
    assert_eq!(a.table_digest().len(), 64);
    // whitespace-only change: same receipt
    let c = sweep(
        "e4m3",
        "  $x ",
        &got_expr(),
        Some(&flag_expr()),
        Some(NOT_NAN),
    )
    .unwrap();
    assert_eq!(a.receipt().id(), c.receipt().id());
    // different where: different receipt
    let d = sweep(
        "e4m3",
        "$x",
        &got_expr(),
        Some(&flag_expr()),
        Some("$b < 0x7f"),
    )
    .unwrap();
    assert_ne!(a.receipt().id(), d.receipt().id());
    // a different table of results: different digest even if the summary counts match
    let e1 = sweep("e4m3", "$x", "$x", None, Some(NOT_NAN)).unwrap();
    let e2 = sweep("e4m3", "$x", "($x)", None, Some(NOT_NAN)).unwrap();
    assert_eq!(e1.table_digest(), e2.table_digest()); // same rows
    let e3 = sweep("e4m3", "$x", "$x * 1", Some("$b < 100"), Some(NOT_NAN)).unwrap();
    assert_ne!(e1.table_digest(), e3.table_digest()); // flag column differs
    let body = String::from_utf8(a.receipt().canonical_bytes()).unwrap();
    assert!(body.contains("\"tool\":\"sweep\""), "{body}");
    assert!(body.contains("\"exactness\":\"exact\""), "{body}");
    assert!(
        body.contains(&format!("\"table_digest\":\"{}\"", a.table_digest())),
        "{body}"
    );
    assert!(body.contains("\"flag_true_count\":14"), "{body}");
    assert!(body.contains("\"considered\":254"), "{body}");
}

#[test]
fn cli_runs_the_spec_command_line() {
    let bin = env!("CARGO_BIN_EXE_aien-calc");
    let out = Command::new(bin)
        .args([
            "sweep",
            "e4m3",
            "--expect",
            "$x",
            "--got",
            &got_expr(),
            "--flag",
            &flag_expr(),
            "--where",
            NOT_NAN,
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("equal   254/254"), "{text}");
    assert!(
        text.contains("flag true: 14 codes [0x01..0x07, 0x81..0x87]"),
        "{text}"
    );
    assert!(
        text.lines().last().unwrap().starts_with("receipt calc:"),
        "{text}"
    );

    // --json prints the receipt itself, and its blake3 is the id on the human run
    let json = Command::new(bin)
        .args([
            "sweep",
            "e4m3",
            "--expect",
            "$x",
            "--got",
            &got_expr(),
            "--flag",
            &flag_expr(),
            "--where",
            NOT_NAN,
            "--json",
        ])
        .output()
        .unwrap();
    let id = format!(
        "calc:{}",
        blake3::hash(json.stdout.strip_suffix(b"\n").unwrap()).to_hex()
    );
    assert!(text.contains(&id), "id {id} not in {text}");
}
