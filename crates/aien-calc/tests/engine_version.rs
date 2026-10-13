//! Receipt stability: the hashed `calc_version` is a semantic engine version, never a git commit.
//! Ids must not change when the code is rebuilt from a different commit, only when evaluation
//! semantics change (and then ENGINE_VERSION is bumped and the pinned ids below are regenerated).

use aien_calc::sweep::sweep;
use aien_calc::{eval, ENGINE_VERSION};

/// The canonical receipt body for `1/3` at 20 digits, written out by hand (not produced by the crate).
const HAND_BUILT_ONE_THIRD: &str = concat!(
    r#"{"calc_version":"aien-calc/1 grammar/1","exactness":"exact","inputs":{"expr":"(/ 1 3)","precision":20},"#,
    r#""method":{"decimal_rounding":"half-even","format_table":"v1","language":"calc-lang v1","#,
    r#""precision_significant_digits":20,"rounding_default":"rne"},"#,
    r#""result":{"decimal":"0.33333333333333333333","decimal_exact":false,"kind":"rational","value":"1/3"},"#,
    r#""schema":"aien-calc-receipt","tool":"eval","version":1,"warnings":[]}"#
);

#[test]
fn receipt_id_matches_a_hand_built_body_with_the_engine_version() {
    let r = eval("1/3", 20).unwrap();
    assert_eq!(
        String::from_utf8(r.receipt().canonical_bytes()).unwrap(),
        HAND_BUILT_ONE_THIRD
    );
    assert_eq!(
        r.receipt().id(),
        format!(
            "calc:{}",
            blake3::hash(HAND_BUILT_ONE_THIRD.as_bytes()).to_hex()
        )
    );
}

#[test]
fn engine_version_is_not_a_commit_hash() {
    assert_eq!(ENGINE_VERSION, "aien-calc/1 grammar/1");
    assert!(
        !ENGINE_VERSION.contains('+'),
        "no build metadata in the hashed version"
    );
}

#[test]
fn build_metadata_is_shown_but_not_hashed() {
    let r = eval("1/3", 20).unwrap();
    let text = r.text();
    assert!(text.contains("\nbuild   "), "{text}");
    assert!(text.contains("engine  aien-calc/1 grammar/1"), "{text}");
    let body = String::from_utf8(r.receipt().canonical_bytes()).unwrap();
    assert!(!body.contains("build"), "{body}");
    assert_eq!(
        text.lines().last().unwrap(),
        format!("receipt {}", r.receipt().id())
    );
}

/// Pinned ids. If one of these fails you changed what some expression means or how a receipt is
/// written. That requires bumping ENGINE_VERSION (and PINNED_ENGINE below), then regenerating this table.
const PINNED_ENGINE: &str = "aien-calc/1 grammar/1";
const EVAL_GOLDEN: &[(&str, u32, &str)] = &[
    (
        "1/3",
        20,
        "calc:f86dd1e83414e7f81f713e738cb42980f3106098f9f0837a241168d67951b42b",
    ),
    (
        "2^-126",
        20,
        "calc:16c806d6ba1d8fcd31148da68aaa3d7d70e46decaf6522413920de2f6f33421b",
    ),
    (
        "0.0019531 == 2^-9",
        20,
        "calc:9ad5976f01d3b5a9fe57c666cdbd87974f98b318e70c27abe6c2e0ebdd538a7f",
    ),
    (
        "bits(e4m3, 17/16)",
        20,
        "calc:8eb6cd8054d62867c6ed2f26ef412f12ce67a362fa9657d8da9c0f1f09a90d02",
    ),
    (
        "1/0",
        20,
        "calc:32f677befda21975faa081d753e06e42415a9f6314553aadfd14721b1b484bcd",
    ),
];
const SWEEP_GOLDEN_ID: &str =
    "calc:77e8166937fad42014b710bdd096b1b98271da10d98d500fd91cb26491f93f30";

#[test]
fn pinned_engine_matches_the_constant() {
    assert_eq!(
        ENGINE_VERSION, PINNED_ENGINE,
        "regenerate the pinned ids when you bump the engine version"
    );
}

#[test]
fn eval_ids_are_pinned() {
    for (expr, prec, want) in EVAL_GOLDEN {
        let got = eval(expr, *prec).unwrap().receipt().id();
        assert_eq!(
            &got, want,
            "receipt id of `{expr}` changed without an ENGINE_VERSION bump"
        );
    }
}

#[test]
fn sweep_id_is_pinned() {
    let w = "(($b & 0x80) << 8) | (($b & 0x7f) << 4)";
    let out = sweep(
        "e4m3",
        "$x",
        &format!("val(bf16, {w}) * 2^120"),
        Some(&format!("class(f32, val(bf16, {w})) == subnormal")),
        Some("class(e4m3, $x) != nan"),
    )
    .unwrap();
    assert_eq!(
        out.receipt().id(),
        SWEEP_GOLDEN_ID,
        "sweep receipt id changed without an ENGINE_VERSION bump"
    );
}
