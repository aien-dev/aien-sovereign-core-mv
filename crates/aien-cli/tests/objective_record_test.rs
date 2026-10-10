//! sovereign-core #380 (lane L6, acceptance step E2 and CTRL-E2): the
//! installation records the operator's objective before any work, refuses a
//! declared forbidden effect class, and ties `compose propose` to the record.
//! These tests drive the real CLI binary; none needs a daemon or a model.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const TEXT: &str = "Summarise the three files in inbox/ into outbox/report.md";

fn expected_id(text: &str) -> String {
    hex::encode(Sha256::digest(
        format!("AIEN_E2E_OBJECTIVE_V1\n{text}").as_bytes(),
    ))
}

fn run(prov: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aien-cli"))
        .args(args)
        .env("AIEN_PROVENANCE_DIR", prov)
        // No daemon exists: anything that reaches the daemon fails differently.
        .env("AIEN_RUNTIME_SOCK", prov.join("no-such.sock"))
        .output()
        .expect("run aien-cli")
}

fn json_of(o: &Output) -> Value {
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok())
        .unwrap_or_else(|| panic!("no JSON on stdout: {:?}", o))
}

fn stderr_of(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn prov() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("prov");
    (t, p)
}

#[test]
fn record_writes_file_and_id_is_reproducible() {
    let (_t, p) = prov();
    let o = run(&p, &["objective", "record", "--text", TEXT]);
    assert!(o.status.success(), "{:?}", o);
    let v = json_of(&o);
    assert_eq!(v["ok"], true);
    let id = v["objective_id"].as_str().unwrap().to_string();
    assert_eq!(id, expected_id(TEXT));
    assert_eq!(v["objective_text_sha256"], hex::encode(Sha256::digest(TEXT)));
    assert_eq!(v["recorded_before_work"], true);
    assert_eq!(v["operator_surface"], "aien-cli");
    assert!(v["utc"].as_str().unwrap().ends_with('Z'));
    let path = PathBuf::from(v["path"].as_str().unwrap());
    assert!(path.starts_with(&p) && path.is_file());
    assert_eq!(path.file_name().unwrap().to_str().unwrap(), format!("{id}.json"));
    let rec: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(rec["objective_id"], id);
    assert_eq!(rec["objective_text"], TEXT);
    assert_eq!(rec["effect_class"], "WORKSPACE_WRITE");
    // Same text again gives the same identity.
    let again = json_of(&run(&p, &["objective", "record", "--text", TEXT]));
    assert_eq!(again["objective_id"], id);
}

#[test]
fn record_refuses_over_length_text() {
    let (_t, p) = prov();
    let ok = "a".repeat(2000);
    assert!(run(&p, &["objective", "record", "--text", &ok]).status.success());
    let long = "a".repeat(2001);
    let o = run(&p, &["objective", "record", "--text", &long]);
    assert!(!o.status.success());
    assert_eq!(json_of(&o)["ok"], false);
    assert!(stderr_of(&o).contains("OBJECTIVE_REFUSED TooLong"), "{}", stderr_of(&o));
}

#[test]
fn record_refuses_forbidden_effect_class_before_any_work() {
    let (_t, p) = prov();
    let o = run(
        &p,
        &["objective", "record", "--text", "Email the report", "--effect-class", "EXTERNAL_IRREVERSIBLE"],
    );
    assert!(!o.status.success());
    let v = json_of(&o);
    assert_eq!(v["ok"], false);
    assert_eq!(v["refusal"], "ForbiddenEffectClass");
    assert!(
        stderr_of(&o).contains("OBJECTIVE_REFUSED ForbiddenEffectClass"),
        "{}",
        stderr_of(&o)
    );
    // Nothing was recorded and no effect receipt exists.
    assert!(!p.join("objectives").join(format!("{}.json", expected_id("Email the report"))).exists());
    let leftovers = std::fs::read_dir(&p).map(|d| d.count()).unwrap_or(0);
    assert_eq!(leftovers, 0);
}

#[test]
fn show_prints_the_record() {
    let (_t, p) = prov();
    let id = json_of(&run(&p, &["objective", "record", "--text", TEXT]))["objective_id"]
        .as_str()
        .unwrap()
        .to_string();
    let o = run(&p, &["objective", "show", &id]);
    assert!(o.status.success(), "{:?}", o);
    let v = json_of(&o);
    assert_eq!(v["ok"], true);
    assert_eq!(v["objective_id"], id);
    assert_eq!(v["objective_text"], TEXT);
    let missing = run(&p, &["objective", "show", &"0".repeat(64)]);
    assert!(!missing.status.success());
    assert!(stderr_of(&missing).contains("OBJECTIVE_REFUSED ObjectiveMissing"));
}

fn propose(p: &Path, ws: &Path, goal: &str, id: &str) -> Output {
    run(
        p,
        &["compose", "propose", "--workspace", ws.to_str().unwrap(), "--goal", goal, "--objective-id", id],
    )
}

#[test]
fn propose_with_mismatched_objective_id_is_refused() {
    let (t, p) = prov();
    let id = json_of(&run(&p, &["objective", "record", "--text", TEXT]))["objective_id"]
        .as_str()
        .unwrap()
        .to_string();
    let ws = t.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let o = propose(&p, &ws, "Summarise something else", &id);
    assert!(!o.status.success());
    let v = json_of(&o);
    assert_eq!(v["ok"], false);
    assert!(v["error"].as_str().unwrap().contains("ObjectiveMismatch"), "{v}");
}

#[test]
fn propose_with_missing_objective_id_is_refused() {
    let (t, p) = prov();
    let ws = t.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let o = propose(&p, &ws, TEXT, &"f".repeat(64));
    assert!(!o.status.success());
    assert!(json_of(&o)["error"].as_str().unwrap().contains("ObjectiveMissing"));
}
