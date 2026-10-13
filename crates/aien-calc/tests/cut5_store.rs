//! Cut 5 tests: receipt store, verify, export, b3sum compatibility, and the canonical readers
//! that make `verify` a real recomputation.

use aien_calc::json::Json;
use aien_calc::parser::{parse, Ast};
use aien_calc::store::Store;
use aien_calc::sweep::sweep;
use aien_calc::verify::verify_bytes;
use aien_calc::{eval, Receipt};
use std::path::PathBuf;
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("aien-calc-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

const EVAL_CORPUS: &[&str] = &[
    "1/3",
    "2^-126",
    "0.0019531 == 2^-9",
    "0x7f & 7",
    "1/0",
    "sqrt(2)",
    "log2(1/8)",
    "-2^2",
    "(2/3)^-2",
    "bits(e4m3, 17/16)",
    "val(e4m3, 0x80)",
    "class(f32, 2^-127)",
    "fields(e4m3, 0xB9)",
    "next(e5m2, 57344)",
    "ftz(f32, -2^-127)",
    "val(e4m3, 0x7f)",
    "98 KB -> B",
    "3.2 GiB -> B",
    "1 / (6.1 GB/tok / 273 GB/s) -> tok/s",
    "(3.04e9 param * 2 B/param) / (273 GB/s) -> ms",
    "params(2048 x 4096)",
    "kv_bytes(48, 4, 128, bf16, 32768) -> GB",
    "1 B + 1 s",
    "bytes(params(2 x 3 x 4), int4)",
];

fn all_receipts() -> Vec<Receipt> {
    let mut v: Vec<Receipt> = EVAL_CORPUS
        .iter()
        .map(|e| eval(e, 20).unwrap().receipt().clone())
        .collect();
    let w = "(($b & 0x80) << 8) | (($b & 0x7f) << 4)";
    v.push(
        sweep(
            "e4m3",
            "$x",
            &format!("val(bf16, {w}) * 2^120"),
            Some(&format!("class(f32, val(bf16, {w})) == subnormal")),
            Some("class(e4m3, $x) != nan"),
        )
        .unwrap()
        .receipt()
        .clone(),
    );
    v.push(
        sweep(
            "e4m3",
            "$x",
            "$x + 1/512",
            None,
            Some("class(e4m3, $x) != nan"),
        )
        .unwrap()
        .receipt()
        .clone(),
    );
    v.push(
        sweep("int4", "$x + $x", "$x * 2", None, None)
            .unwrap()
            .receipt()
            .clone(),
    );
    v.push(
        sweep("e4m3", "$x", "1 / (1 / $x)", None, None)
            .unwrap()
            .receipt()
            .clone(),
    );
    v
}

// ---- canonical readers ----

#[test]
fn json_parser_round_trips_canonical_bytes() {
    for r in all_receipts() {
        let text = String::from_utf8(r.canonical_bytes()).unwrap();
        let j = Json::parse(&text).unwrap();
        assert_eq!(j.to_canonical(), text);
    }
    let j = Json::parse(r#"{"b":[1,-2,"x\"y\\z\u000aé",null,true,false],"a":{"k":"v"}}"#).unwrap();
    assert_eq!(
        j.to_canonical(),
        "{\"a\":{\"k\":\"v\"},\"b\":[1,-2,\"x\\\"y\\\\z\\u000a\u{e9}\",null,true,false]}"
    );
    assert_eq!(j.get("a").and_then(|a| a.get("k")), Some(&Json::str("v")));
}

#[test]
fn json_parser_rejects_what_receipts_never_contain() {
    for bad in [
        "",
        "{",
        "[1,]",
        "1.5",
        "1e3",
        "{\"a\":1,}",
        "\"abc",
        "{\"a\" 1}",
        "nul",
        "[1] x",
        "{\"a\":1,\"a\":2}",
        "-",
        "01",
    ] {
        assert!(Json::parse(bad).is_err(), "`{bad}` must be rejected");
    }
}

#[test]
fn canonical_form_reads_back_to_the_same_ast() {
    let mut exprs: Vec<&str> = EVAL_CORPUS.to_vec();
    exprs.extend([
        "1 + 2 * 3",
        "-(3 B)",
        "0.1 + 0.2",
        "$b & 0x80",
        "e4m3",
        "1_000 << 3",
        "abs(-3 B)",
        "(1 < 2) == true",
        "x",
    ]);
    for e in exprs {
        let Ok(ast) = parse(e) else { continue };
        let canon = ast.canonical();
        let back = Ast::from_canonical(&canon).unwrap_or_else(|m| panic!("`{canon}`: {m}"));
        assert_eq!(back, ast, "`{e}` via `{canon}`");
        assert_eq!(back.canonical(), canon);
    }
    assert!(Ast::from_canonical("(+ 1").is_err());
    assert!(Ast::from_canonical("(+ 1 2) 3").is_err());
    assert!(Ast::from_canonical("").is_err());
}

// ---- store ----

#[test]
fn store_writes_the_canonical_bytes_verbatim_under_the_hash_name() {
    let dir = tmp("store-verbatim");
    let store = Store::new(&dir);
    let r = eval("1/3", 20).unwrap().receipt().clone();
    let path = store
        .save(
            &r,
            "/work",
            &["aien-calc".into(), "eval".into(), "1/3".into()],
        )
        .unwrap();
    assert_eq!(
        path,
        dir.join("receipts").join(format!("{}.json", r.id_hex()))
    );
    let on_disk = std::fs::read(&path).unwrap();
    assert_eq!(on_disk, r.canonical_bytes());
    assert!(
        !on_disk.ends_with(b"\n"),
        "no trailing newline: b3sum of the file must equal the id"
    );
    // b3sum compatibility: BLAKE3 of the file bytes is the id
    assert_eq!(format!("calc:{}", blake3::hash(&on_disk).to_hex()), r.id());
    // saving twice is idempotent
    store.save(&r, "/work", &[]).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), on_disk);
    assert_eq!(std::fs::read_dir(dir.join("receipts")).unwrap().count(), 1);
}

#[test]
fn ledger_gets_one_json_line_per_save() {
    let dir = tmp("store-ledger");
    let store = Store::new(&dir);
    let a = eval("1/3", 20).unwrap().receipt().clone();
    let b = eval("1/4", 20).unwrap().receipt().clone();
    store
        .save(
            &a,
            "/w1",
            &["aien-calc".into(), "eval".into(), "1/3".into()],
        )
        .unwrap();
    store
        .save(
            &b,
            "/w2",
            &["aien-calc".into(), "eval".into(), "1/4".into()],
        )
        .unwrap();
    store.save(&a, "/w3", &[]).unwrap();
    let text = std::fs::read_to_string(store.ledger_path()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    let first = Json::parse(lines[0]).unwrap();
    assert_eq!(first.get("id"), Some(&Json::Str(a.id())));
    assert_eq!(first.get("cwd"), Some(&Json::str("/w1")));
    assert_eq!(
        first.get("argv"),
        Some(&Json::Arr(vec![
            Json::str("aien-calc"),
            Json::str("eval"),
            Json::str("1/3")
        ]))
    );
    assert!(matches!(first.get("ts"), Some(Json::Int(t)) if *t > 1_700_000_000));
    assert_eq!(
        Json::parse(lines[2]).unwrap().get("id"),
        Some(&Json::Str(a.id()))
    );
}

#[test]
fn ids_resolve_in_long_short_and_bare_form() {
    let dir = tmp("store-resolve");
    let store = Store::new(&dir);
    let r = eval("2^-126", 20).unwrap().receipt().clone();
    store.save(&r, "/w", &[]).unwrap();
    let hex = r.id_hex();
    for form in [
        r.id(),
        hex.clone(),
        format!("calc:{}", &hex[..12]),
        hex[..12].to_string(),
    ] {
        assert_eq!(store.resolve(&form).unwrap(), hex, "form {form}");
        assert_eq!(store.load(&form).unwrap(), r.canonical_bytes());
    }
    assert!(store.resolve("calc:000000000000").is_err());
    assert!(store.resolve("calc:abc").is_err()); // too short to be unambiguous
    assert!(store.resolve("../../etc/passwd").is_err());
}

#[test]
fn export_copies_into_calc_receipts_next_to_the_evidence() {
    let dir = tmp("store-export");
    let store = Store::new(dir.join("home"));
    let r = eval("98 KB -> B", 20).unwrap().receipt().clone();
    store.save(&r, "/w", &[]).unwrap();
    let ev = dir.join("evidence");
    let out = store.export(&r.id(), &ev).unwrap();
    assert_eq!(
        out,
        ev.join("calc-receipts")
            .join(format!("{}.json", r.id_hex()))
    );
    assert_eq!(std::fs::read(&out).unwrap(), r.canonical_bytes());
    let v = verify_bytes(&std::fs::read(&out).unwrap(), Some(&r.id_hex()));
    assert!(v.matches, "{:?}", v.diffs);
    assert!(store.export("calc:ffffffffffff", &ev).is_err());
}

// ---- verify ----

#[test]
fn verify_matches_every_receipt_the_tests_produce() {
    let dir = tmp("verify-all");
    let store = Store::new(&dir);
    let receipts = all_receipts();
    assert!(receipts.len() >= 28);
    for r in &receipts {
        store.save(r, "/w", &[]).unwrap();
        let bytes = store.load(&r.id()).unwrap();
        let v = verify_bytes(&bytes, Some(&r.id_hex()));
        assert!(v.matches, "{}: {:?}", r.id(), v.diffs);
        assert_eq!(v.id, r.id());
    }
}

#[test]
fn a_flipped_bit_anywhere_in_a_stored_receipt_is_a_mismatch() {
    let r = eval("(3.04e9 param * 2 B/param) / (273 GB/s) -> ms", 20)
        .unwrap()
        .receipt()
        .clone();
    let good = r.canonical_bytes();
    assert!(verify_bytes(&good, Some(&r.id_hex())).matches);
    for i in 0..good.len() {
        let mut bad = good.clone();
        bad[i] ^= 0x01;
        // with the file name check, and without it (pure recomputation)
        assert!(
            !verify_bytes(&bad, Some(&r.id_hex())).matches,
            "byte {i} flipped, named check"
        );
        assert!(
            !verify_bytes(&bad, None).matches,
            "byte {i} flipped, recomputation only"
        );
    }
    let mut bad = good.clone();
    bad.push(b'\n');
    assert!(
        !verify_bytes(&bad, None).matches,
        "trailing newline is not canonical"
    );
}

#[test]
fn a_lying_but_well_formed_receipt_names_the_differing_field() {
    let r = eval("1/3", 20).unwrap().receipt().clone();
    let text = String::from_utf8(r.canonical_bytes()).unwrap();
    let forged = text.replace("\"value\":\"1/3\"", "\"value\":\"1/4\"");
    assert_ne!(forged, text);
    let v = verify_bytes(forged.as_bytes(), None);
    assert!(!v.matches);
    assert!(
        v.diffs
            .iter()
            .any(|d| d.contains("result.value") && d.contains("1/4") && d.contains("1/3")),
        "{:?}",
        v.diffs
    );
    // a forged receipt filed under its own (correct) hash still fails: the hash proves integrity, not truth
    let own = blake3::hash(forged.as_bytes()).to_hex().to_string();
    assert!(!verify_bytes(forged.as_bytes(), Some(&own)).matches);
}

#[test]
fn a_receipt_from_another_engine_version_is_reported_not_trusted() {
    let r = eval("1/3", 20).unwrap().receipt().clone();
    let text = String::from_utf8(r.canonical_bytes())
        .unwrap()
        .replace("aien-calc/1 grammar/1", "aien-calc/2 grammar/1");
    let v = verify_bytes(text.as_bytes(), None);
    assert!(!v.matches);
    assert!(
        v.diffs.iter().any(|d| d.contains("calc_version")),
        "{:?}",
        v.diffs
    );
}

#[test]
fn a_misnamed_file_is_a_mismatch() {
    let r = eval("1/3", 20).unwrap().receipt().clone();
    let wrong = eval("1/4", 20).unwrap().receipt().id_hex();
    let v = verify_bytes(&r.canonical_bytes(), Some(&wrong));
    assert!(!v.matches);
    assert!(
        v.diffs.iter().any(|d| d.contains("file name")),
        "{:?}",
        v.diffs
    );
}

#[test]
fn garbage_is_a_mismatch_not_a_panic() {
    for bytes in [
        &b""[..],
        b"not json",
        b"{}",
        b"{\"schema\":\"aien-calc-receipt\"}",
        b"\xff\xfe",
        b"[1,2]",
    ] {
        assert!(!verify_bytes(bytes, None).matches);
    }
}

// ---- command line ----

fn run(home: &PathBuf, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_aien-calc"))
        .args(args)
        .env("AIEN_CALC_HOME", home)
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn cli_stores_verifies_exports_and_hashes() {
    let home = tmp("cli");
    let (code, out, _) = run(&home, &["eval", "1 / (6.1 GB/tok / 273 GB/s) -> tok/s"]);
    assert_eq!(code, 0);
    let id = out
        .lines()
        .last()
        .unwrap()
        .strip_prefix("receipt ")
        .unwrap()
        .to_string();
    let hex = id.strip_prefix("calc:").unwrap();
    let file = home.join("receipts").join(format!("{hex}.json"));
    assert!(file.exists(), "{file:?}");
    assert!(std::fs::read_to_string(home.join("ledger.jsonl"))
        .unwrap()
        .contains(&id));

    let (code, out, _) = run(&home, &["verify", &id]);
    assert_eq!(
        (code, out.trim()),
        (0, format!("MATCH {id}").as_str()),
        "{out}"
    );
    let (code, out, _) = run(&home, &["verify", file.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");

    let ev = tmp("cli-evidence");
    let (code, out, _) = run(&home, &["export", &id, ev.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");
    let exported = ev.join("calc-receipts").join(format!("{hex}.json"));
    assert_eq!(
        std::fs::read(&exported).unwrap(),
        std::fs::read(&file).unwrap()
    );

    // b3 prints the same hash a b3sum run would
    let (code, out, _) = run(&home, &["b3", exported.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert_eq!(out.split_whitespace().next().unwrap(), hex);

    // tamper with the stored file: verify says MISMATCH and exits 1
    let mut bytes = std::fs::read(&file).unwrap();
    let i = bytes.iter().position(|b| *b == b'4').unwrap();
    bytes[i] = b'5';
    std::fs::write(&file, bytes).unwrap();
    let (code, out, _) = run(&home, &["verify", &id]);
    assert_eq!(code, 1);
    assert!(out.starts_with("MISMATCH"), "{out}");

    // --no-store leaves nothing behind
    let home2 = tmp("cli-nostore");
    let (code, _, _) = run(&home2, &["--no-store", "eval", "1/3"]);
    assert_eq!(code, 0);
    assert!(!home2.join("receipts").exists());
    // refusals are stored too, and verify
    let (code, out, _) = run(&home2, &["eval", "1/0"]);
    assert_eq!(code, 3);
    let id = out
        .lines()
        .last()
        .unwrap()
        .strip_prefix("receipt ")
        .unwrap()
        .to_string();
    let (code, out, _) = run(&home2, &["verify", &id]);
    assert_eq!(code, 0, "{out}");
}

#[test]
fn names_the_canonical_form_uses_cannot_be_called_as_functions() {
    // `(neg 3)` and `(x 2 3)` are canonical spellings of negation and shapes; a user function of
    // that name would make verify recompute a different expression.
    for e in ["neg(3)", "qty(3, B)", "x(2, 3)"] {
        assert!(eval(e, 20).is_err(), "`{e}` must be a syntax error");
    }
    assert!(eval("-(3)", 20).is_ok());
}
