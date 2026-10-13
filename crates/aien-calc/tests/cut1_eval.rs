//! Cut 1 tests: exact evaluation, refusals, and receipt identity.
//! Expected values are derived by hand (see comments), never by running the evaluator.

use aien_calc::{eval, Json};

/// Canonical exact result of `expr` (`p/q`, `true`, `false`, ...). Panics on syntax error or refusal.
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

#[test]
fn forty_hand_derived_cases() {
    let cases: &[(&str, &str)] = &[
        // literals
        ("1/3", "1/3"),
        ("2^-126", "1/85070591730234615865843651857942052864"), // 2^126 = 85070591730234615865843651857942052864
        ("0.0019531", "19531/10000000"),
        ("0.0019531 == 2^-9", "false"), // 2^-9 = 0.001953125, not 0.0019531
        ("0x7f & 7", "7/1"),            // 0b1111111 & 0b111
        ("3.04e9", "3040000000/1"),
        ("1.5e-3", "3/2000"), // 15/10000
        ("0b1000_0000", "128/1"),
        ("0xFF", "255/1"),
        ("1_000_000", "1000000/1"),
        // precedence and associativity
        ("1+2*3", "7/1"),
        ("(1+2)*3", "9/1"),
        ("2^3^2", "512/1"), // right associative: 2^9
        ("-2^2", "-4/1"),   // unary minus binds looser than ^
        ("(-2)^2", "4/1"),
        ("(-2)^3", "-8/1"),
        ("4 - 2 - 1", "1/1"), // left associative
        ("100 / 10 / 5", "2/1"),
        ("1 + 2 << 1", "6/1"),  // shifts bind looser than +
        ("1 << 2 & 7", "4/1"),  // & binds looser than <<
        ("1 | 2 == 3", "true"), // comparison binds loosest
        // exact powers
        ("2^-2", "1/4"),
        ("(2/3)^2", "4/9"),
        ("(2/3)^-2", "9/4"),
        ("10^-3", "1/1000"),
        ("2^120", "1329227995784915872903807060280344576/1"),
        ("2^64 - 1", "18446744073709551615/1"),
        ("2^10 * 2^-10", "1/1"),
        // exact decimal arithmetic where binary floats are wrong
        ("0.1 + 0.2", "3/10"),
        ("0.1 + 0.2 == 0.3", "true"),
        ("1/3 + 1/6", "1/2"),
        ("1/3 * 3", "1/1"),
        // remainder is truncated (sign of the dividend)
        ("7 % 3", "1/1"),
        ("-7 % 3", "-1/1"),
        ("7/2 % 1", "1/2"),
        // bitwise and shifts on integers
        ("1 << 10", "1024/1"),
        ("1024 >> 3", "128/1"),
        ("-8 >> 1", "-4/1"),
        ("-1 >> 1", "-1/1"), // arithmetic shift floors
        ("5 | 2", "7/1"),
        ("6 xor 3", "5/1"),
        ("6 & 3", "2/1"),
        ("-1 & 0xff", "255/1"), // two's complement semantics
        // functions
        ("abs(-3/4)", "3/4"),
        ("floor(-7/2)", "-4/1"),
        ("ceil(-7/2)", "-3/1"),
        ("trunc(-7/2)", "-3/1"),
        ("round_half_even(5/2)", "2/1"),
        ("round_half_even(7/2)", "4/1"),
        ("round_half_even(-5/2)", "-2/1"),
        ("gcd(12, 18)", "6/1"),
        ("lcm(4, 6)", "12/1"),
        ("min(1/3, 1/4)", "1/4"),
        ("max(1/3, 1/4)", "1/3"),
        ("sqrt(49/16)", "7/4"),
        ("log2(1/8)", "-3/1"),
        ("log2(1024)", "10/1"),
        ("relerr(99, 100)", "1/100"), // |99-100|/100
        // comparisons
        ("1/3 < 0.34", "true"),
        ("2 >= 3", "false"),
        ("3 != 3", "false"),
        ("2 <= 2", "true"),
        ("5 > 4", "true"),
    ];
    assert!(
        cases.len() >= 40,
        "need at least 40 cases, have {}",
        cases.len()
    );
    for (expr, want) in cases {
        assert_eq!(&ev(expr), want, "expression `{expr}`");
    }
}

#[test]
fn whitespace_does_not_matter() {
    assert_eq!(ev("  1 +\t2 *  3 "), "7/1");
    assert_eq!(ev("gcd( 12 ,18 )"), "6/1");
}

#[test]
fn refusals_are_first_class() {
    for (expr, needle) in [
        ("1/0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("0^-1", "division by zero"),
        ("1.5 & 1", "integer"),
        ("1 << 1/2", "integer"),
        ("log2(3)", "powers of two"),
        ("log2(0)", "positive"),
        ("sqrt(2)", "perfect square"),
        ("sqrt(-4)", "negative"),
        ("2^(1/2)", "integer"),
        ("2^(2^40)", "budget"),
        ("1 << 99999999999", "budget"),
        ("(2^100000)*(2^100000)", "budget"),
        ("$b", "unbound"),
        ("nosuch(1)", "unknown function"),
        ("gcd(1)", "arguments"),
        ("gcd(1/2, 3)", "integer"),
        ("true + 1", "number"),
        ("(1 < 2) < 3", "number"),
    ] {
        let r = refusal(expr);
        assert!(
            r.contains(needle),
            "`{expr}` refusal `{r}` should mention `{needle}`"
        );
    }
}

#[test]
fn syntax_errors_are_errors_not_receipts() {
    for expr in [
        "", "1 +", "(1", "1)", "1 2", "0x", "0b2", "1e", "1.", ".5", "@", "1 < 2 < 3", "gcd(1,)", "1__0",
        "$", "1e999999999", "((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((1",
    ] {
        assert!(eval(expr, 20).is_err(), "`{expr}` should be a syntax error");
    }
}

#[test]
fn decimal_rendering_is_exact_where_it_can_be() {
    let o = eval("1/3", 10).unwrap();
    assert!(o.text().contains("0.3333333333"), "{}", o.text());
    assert!(o.text().contains("rounded"), "{}", o.text());
    let o = eval("0.0019531", 20).unwrap();
    assert!(o.text().contains("0.0019531 (exact)"), "{}", o.text());
    // 2^-126 = 1.1754943508222875e-38; 4 significant digits, scientific form for tiny numbers.
    let o = eval("2^-126", 4).unwrap();
    assert!(o.text().contains("1.175e-38"), "{}", o.text());
    let o = eval("3.04e9", 20).unwrap();
    assert!(o.text().contains("3040000000 (exact)"), "{}", o.text());
}

#[test]
fn last_line_of_text_is_the_receipt_id() {
    let o = eval("1/3", 20).unwrap();
    let last = o.text().lines().last().unwrap().to_string();
    assert_eq!(last, format!("receipt {}", o.receipt().id()));
    assert!(o.receipt().id().starts_with("calc:"));
    assert_eq!(o.receipt().id().len(), "calc:".len() + 64);
}

// ---- receipt identity ----

#[test]
fn id_is_blake3_of_canonical_bytes() {
    let o = eval("1/3", 20).unwrap();
    let r = o.receipt();
    assert_eq!(
        r.id(),
        format!("calc:{}", blake3::hash(&r.canonical_bytes()).to_hex())
    );
}

#[test]
fn id_is_stable_across_whitespace_and_repeats() {
    let a = eval("1/3", 20).unwrap().receipt().id();
    let b = eval("  1 /   3 ", 20).unwrap().receipt().id();
    let c = eval("1/3", 20).unwrap().receipt().id();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn literal_spelling_is_canonicalised() {
    // 0x7f, 127 and 1_2_7 are the same number once parsed.
    let a = eval("0x7f", 20).unwrap().receipt().id();
    let b = eval("127", 20).unwrap().receipt().id();
    assert_eq!(a, b);
}

#[test]
fn id_changes_when_any_input_changes() {
    let base = eval("1/3", 20).unwrap().receipt().id();
    for (expr, prec) in [
        ("1/4", 20),
        ("1/3", 21),
        ("1/3+0", 20),
        ("3/1", 20),
        ("(1/3)*1", 20),
    ] {
        let other = eval(expr, prec).unwrap().receipt().id();
        assert_ne!(base, other, "`{expr}` precision {prec} must change the id");
    }
}

#[test]
fn refusals_get_a_receipt_marked_refused() {
    let o = eval("1/0", 20).unwrap();
    let body = String::from_utf8(o.receipt().canonical_bytes()).unwrap();
    assert!(body.contains("\"exactness\":\"refused\""), "{body}");
    assert!(body.contains("division by zero"), "{body}");
    let ok = eval("1/2", 20).unwrap();
    let body = String::from_utf8(ok.receipt().canonical_bytes()).unwrap();
    assert!(body.contains("\"exactness\":\"exact\""), "{body}");
    assert!(body.contains("\"schema\":\"aien-calc-receipt\""), "{body}");
    assert!(body.contains("\"tool\":\"eval\""), "{body}");
    assert!(body.contains("\"value\":\"1/2\""), "{body}");
    assert!(body.contains("calc-lang v1"), "{body}");
    assert!(
        !body.contains("\": ") && !body.contains(", "),
        "no whitespace between tokens: {body}"
    );
}

// ---- canonical JSON ----

fn shuffled_pair() -> (Json, Json) {
    let a = Json::obj(vec![
        ("zeta", Json::Int(1)),
        (
            "alpha",
            Json::obj(vec![("y", Json::Bool(true)), ("x", Json::Null)]),
        ),
        (
            "mid",
            Json::Arr(vec![Json::str("a\"b\\c\n\u{1}é"), Json::Int(-7)]),
        ),
    ]);
    let b = Json::obj(vec![
        (
            "mid",
            Json::Arr(vec![Json::str("a\"b\\c\n\u{1}é"), Json::Int(-7)]),
        ),
        (
            "alpha",
            Json::obj(vec![("x", Json::Null), ("y", Json::Bool(true))]),
        ),
        ("zeta", Json::Int(1)),
    ]);
    (a, b)
}

#[test]
fn canonical_json_is_exact_bytes() {
    let (a, _) = shuffled_pair();
    assert_eq!(
        a.to_canonical(),
        "{\"alpha\":{\"x\":null,\"y\":true},\"mid\":[\"a\\\"b\\\\c\\u000a\\u0001é\",-7],\"zeta\":1}"
    );
}

#[test]
fn canonical_json_big_integers_become_strings() {
    assert_eq!(
        Json::Int(9_007_199_254_740_992).to_canonical(),
        "9007199254740992"
    ); // 2^53 is still safe
    assert_eq!(
        Json::Int(9_007_199_254_740_993).to_canonical(),
        "\"9007199254740993\""
    );
    assert_eq!(
        Json::Int(-9_007_199_254_740_993).to_canonical(),
        "\"-9007199254740993\""
    );
}

/// What a stability check means: two objects with the same content in different key order must
/// serialise to the same bytes.
fn stable_across_key_order(canon: impl Fn(&Json) -> String) -> bool {
    let (a, b) = shuffled_pair();
    canon(&a) == canon(&b)
}

/// Mutant: a wrong canonicaliser that keeps insertion order instead of sorting keys.
fn mutant_unsorted(j: &Json) -> String {
    match j {
        Json::Obj(pairs) => {
            let inner: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("\"{k}\":{}", mutant_unsorted(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Json::Arr(items) => {
            let inner: Vec<String> = items.iter().map(mutant_unsorted).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_canonical(),
    }
}

#[test]
fn key_order_stability_holds_for_the_real_canonicaliser() {
    assert!(stable_across_key_order(|j| j.to_canonical()));
}

#[test]
fn mutant_canonicaliser_fails_the_stability_check() {
    assert!(
        !stable_across_key_order(mutant_unsorted),
        "the stability check must detect an order-dependent canonicaliser"
    );
}

#[test]
#[should_panic(expected = "duplicate")]
fn duplicate_keys_are_rejected() {
    let _ = Json::obj(vec![("a", Json::Null), ("a", Json::Null)]);
}
