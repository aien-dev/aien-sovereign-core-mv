//! Cut 8 tests: `scripts/calc-guard.sh` (Claude Code PreToolUse hook) and
//! `scripts/calc-receipts-gate.sh` (CI gate). Both are warn-only unless switched to enforce.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const GUARD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/calc-guard.sh");
const GATE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/calc-receipts-gate.sh"
);
const BIN: &str = env!("CARGO_BIN_EXE_aien-calc");

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/hooks/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

struct Out {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run_guard(input: &str, env: &[(&str, &str)]) -> Out {
    let mut c = Command::new("sh");
    c.arg(GUARD)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c.env_remove("CALC_GUARD_MODE");
    for (k, v) in env {
        c.env(k, v);
    }
    let mut child = c.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let o = child.wait_with_output().unwrap();
    Out {
        code: o.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&o.stdout).into(),
        stderr: String::from_utf8_lossy(&o.stderr).into(),
    }
}

fn decision(o: &Out) -> Option<String> {
    if o.stdout.trim().is_empty() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&o.stdout)
        .unwrap_or_else(|e| panic!("hook output is not JSON: {e}: {}", o.stdout));
    v["hookSpecificOutput"]["permissionDecision"]
        .as_str()
        .map(str::to_string)
}

fn context(o: &Out) -> String {
    let v: serde_json::Value = serde_json::from_str(&o.stdout).unwrap();
    v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn input(tool: &str, path: &str, field: &str, text: &str) -> String {
    serde_json::json!({"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": {"file_path": path, field: text}}).to_string()
}

// ---- the three spec fixtures ----

#[test]
fn numbers_without_an_id_are_denied_in_deny_mode() {
    let o = run_guard(
        &fixture("numbers_no_id.json"),
        &[("CALC_GUARD_MODE", "deny")],
    );
    assert_eq!(o.code, 0);
    assert_eq!(decision(&o).as_deref(), Some("deny"), "{}", o.stdout);
    let v: serde_json::Value = serde_json::from_str(&o.stdout).unwrap();
    let reason = v["hookSpecificOutput"]["permissionDecisionReason"]
        .as_str()
        .unwrap();
    assert!(
        reason.contains("This doc states derived numbers"),
        "{reason}"
    );
    assert!(reason.contains("aien-calc eval"), "{reason}");
    assert!(reason.contains("CALC-EXEMPT: <why>"), "{reason}");
}

#[test]
fn default_mode_is_warn_only_it_never_blocks() {
    let o = run_guard(&fixture("numbers_no_id.json"), &[]);
    assert_eq!(o.code, 0);
    assert_eq!(
        decision(&o),
        None,
        "warn mode must not set a permission decision: {}",
        o.stdout
    );
    let ctx = context(&o);
    assert!(
        ctx.contains("WARN") && ctx.contains("aien-calc eval"),
        "{ctx}"
    );
    assert!(o.stderr.contains("calc-guard"), "{}", o.stderr);
}

#[test]
fn numbers_with_a_receipt_id_are_allowed_silently() {
    for mode in ["deny", "warn"] {
        let o = run_guard(
            &fixture("numbers_with_id.json"),
            &[("CALC_GUARD_MODE", mode)],
        );
        assert_eq!(
            (o.code, o.stdout.as_str(), o.stderr.as_str()),
            (0, "", ""),
            "mode {mode}"
        );
    }
}

#[test]
fn numbers_with_an_exempt_line_are_allowed_silently() {
    for mode in ["deny", "warn"] {
        let o = run_guard(
            &fixture("numbers_exempt.json"),
            &[("CALC_GUARD_MODE", mode)],
        );
        assert_eq!(
            (o.code, o.stdout.as_str(), o.stderr.as_str()),
            (0, "", ""),
            "mode {mode}"
        );
    }
}

// ---- scope and thresholds ----

const NUMBERS: &str = "Weights 3.04 GB, bus 273 GB/s, 11 ms per token, 90 tok/s.";

#[test]
fn only_numbers_bearing_documents_are_checked() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    for path in [
        "/r/docs/adr/0001-x.md",
        "/r/docs/perf/evidence/run1/README.md",
        "/r/docs/evidence/notes.md",
        "/r/crates/x/FINDINGS.md",
        "/home/u/handoffs/2026-10-13-x.md",
        "/r/specs/model-providers/x.md",
    ] {
        let o = run_guard(&input("Write", path, "content", NUMBERS), &deny);
        assert_eq!(
            decision(&o).as_deref(),
            Some("deny"),
            "{path} should be checked"
        );
    }
    for path in [
        "/r/src/main.rs",
        "/r/docs/guide.md",
        "/r/README.md",
        "/r/notes/todo.md",
        "/home/u/handoffs/data.txt",
        "/r/docs/adr-draft/x.md",
        "/r/xdocs/adr/../../src/a.rs",
    ] {
        let o = run_guard(&input("Write", path, "content", NUMBERS), &deny);
        assert_eq!(o.stdout, "", "{path} must be ignored");
    }
}

#[test]
fn fewer_than_three_claims_pass() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    for text in [
        "Weights are 3.04 GB.",
        "Weights are 3.04 GB and the bus is 273 GB/s.",
        "ADR 0033, PR #6566, 2026-10-13, version 3.4.0, 12 files, 5 tests",
        "",
    ] {
        let o = run_guard(&input("Write", "/r/docs/adr/x.md", "content", text), &deny);
        assert_eq!(o.stdout, "", "`{text}`");
    }
    let three = "a 3 GB, b 5 ms, c 7 %";
    assert_eq!(
        decision(&run_guard(
            &input("Write", "/r/docs/adr/x.md", "content", three),
            &deny
        ))
        .as_deref(),
        Some("deny")
    );
}

#[test]
fn claim_patterns_follow_the_spec_units() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    let flagged = |text: &str| {
        decision(&run_guard(
            &input("Write", "/r/specs/x.md", "content", text),
            &deny,
        ))
        .is_some()
    };
    // each unit in the spec list counts
    for unit in [
        "GB", "MB", "KB", "GiB", "MiB", "B", "GB/s", "tok/s", "ms", "us", "ns", "%", "FLOP",
        "param", "launch",
    ] {
        let text = format!("1 {unit}, 2 {unit}, 3 {unit}");
        assert!(flagged(&text), "unit `{unit}` should count");
    }
    assert!(flagged("2^31, 2^-126 and 2^(8-1)"), "2^ counts");
    assert!(
        flagged("3.04GB, 273GB/s, 11ms"),
        "no space between number and unit"
    );
    // look-alikes do not count
    assert!(
        !flagged("3 Bytes, 4 msec, 5 users, 6 usage, 7 params*"),
        "words that merely start with a unit"
    );
    assert!(!flagged("v1B, abc12ms, x3GB"), "numbers glued to a name");
    assert!(!flagged("A 2026-10-13 build of PR 6566 touched 12 files"));
}

#[test]
fn edit_new_string_is_checked_but_an_existing_id_in_the_file_covers_it() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    let o = run_guard(
        &input("Edit", "/r/docs/adr/x.md", "new_string", NUMBERS),
        &deny,
    );
    assert_eq!(decision(&o).as_deref(), Some("deny"));
    // a file that already carries an id covers later edits
    let dir = std::env::temp_dir().join(format!("aien-guard-edit-{}", std::process::id()));
    let adr = dir.join("docs").join("adr");
    std::fs::create_dir_all(&adr).unwrap();
    let f = adr.join("0001.md");
    std::fs::write(
        &f,
        "see calc:a1e1ad51c34a215c15662adf2a3614a93b20653e08574462bab7f9d52168daaa\n",
    )
    .unwrap();
    let o = run_guard(
        &input("Edit", f.to_str().unwrap(), "new_string", NUMBERS),
        &deny,
    );
    assert_eq!(o.stdout, "");
}

#[test]
fn an_exempt_line_needs_a_reason_and_an_id_needs_enough_digits() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    for text in [
        format!("{NUMBERS}\nCALC-EXEMPT:"),
        format!("{NUMBERS}\nCALC-EXEMPT:   \n"),
        format!("{NUMBERS} calc:abc"),
        format!("{NUMBERS} calc:zzzzzzzzzzzz"),
    ] {
        let o = run_guard(&input("Write", "/r/docs/adr/x.md", "content", &text), &deny);
        assert_eq!(decision(&o).as_deref(), Some("deny"), "`{text}`");
    }
    let ok = format!("{NUMBERS} calc:0123456789ab");
    assert_eq!(
        run_guard(&input("Write", "/r/docs/adr/x.md", "content", &ok), &deny).stdout,
        ""
    );
}

#[test]
fn other_tools_garbage_input_and_off_mode_are_ignored() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    assert_eq!(
        run_guard(
            &input("Bash", "/r/docs/adr/x.md", "content", NUMBERS),
            &deny
        )
        .stdout,
        ""
    );
    assert_eq!(
        run_guard(
            &input("Read", "/r/docs/adr/x.md", "content", NUMBERS),
            &deny
        )
        .stdout,
        ""
    );
    for junk in ["", "not json", "{}", "[1]", "{\"tool_name\":\"Write\"}"] {
        let o = run_guard(junk, &deny);
        assert_eq!((o.code, o.stdout.as_str()), (0, ""), "`{junk}`");
    }
    let o = run_guard(
        &fixture("numbers_no_id.json"),
        &[("CALC_GUARD_MODE", "off")],
    );
    assert_eq!((o.code, o.stdout.as_str(), o.stderr.as_str()), (0, "", ""));
}

#[test]
fn multi_edit_new_strings_are_joined_and_checked() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    let inp = serde_json::json!({"tool_name": "MultiEdit", "tool_input": {"file_path": "/r/docs/adr/x.md",
        "edits": [{"old_string": "a", "new_string": "3 GB and 5 ms"}, {"old_string": "b", "new_string": "9 tok/s"}]}})
    .to_string();
    assert_eq!(decision(&run_guard(&inp, &deny)).as_deref(), Some("deny"));
}

#[test]
fn log_line_is_written_when_asked() {
    let dir = std::env::temp_dir().join(format!("aien-guard-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("guard.log");
    let _ = std::fs::remove_file(&log);
    run_guard(
        &fixture("numbers_no_id.json"),
        &[("CALC_GUARD_LOG", log.to_str().unwrap())],
    );
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(
        text.contains("docs/adr/0042-decode-budget.md")
            && text.contains("claims=5")
            && text.contains("warn"),
        "{text}"
    );
}

// ---- receipts gate ----

struct Repo {
    root: PathBuf,
    home: PathBuf,
}

impl Repo {
    fn new(name: &str) -> Repo {
        let root = std::env::temp_dir().join(format!("aien-gate-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("docs/adr/calc-receipts")).unwrap();
        let home = root.join("home");
        Repo { root, home }
    }

    /// Create a real receipt with the real binary and copy it next to the ADR.
    fn receipt(&self, expr: &str) -> String {
        let o = Command::new(BIN)
            .args(["eval", expr])
            .env("AIEN_CALC_HOME", &self.home)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&o.stdout).to_string();
        let id = text
            .lines()
            .last()
            .unwrap()
            .strip_prefix("receipt calc:")
            .unwrap()
            .to_string();
        let o = Command::new(BIN)
            .args(["export", &id, self.root.join("docs/adr").to_str().unwrap()])
            .env("AIEN_CALC_HOME", &self.home)
            .output()
            .unwrap();
        assert!(o.status.success());
        id
    }

    fn doc(&self, name: &str, text: &str) -> PathBuf {
        let p = self.root.join("docs/adr").join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    fn receipt_path(&self, id: &str) -> PathBuf {
        self.root
            .join("docs/adr/calc-receipts")
            .join(format!("{id}.json"))
    }

    fn gate(&self, files: &[&Path], mode: &str, enforce: bool) -> Out {
        let mut c = Command::new("sh");
        c.arg(GATE).args(["--mode", mode]).args(files);
        c.env("AIEN_CALC_BIN", BIN)
            .env("CALC_GATE_ROOT", &self.root)
            .current_dir(&self.root);
        c.env_remove("CALC_GATE_ENFORCE");
        if enforce {
            c.env("CALC_GATE_ENFORCE", "1");
        }
        let o = c.output().unwrap();
        Out {
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).into(),
            stderr: String::from_utf8_lossy(&o.stderr).into(),
        }
    }
}

fn combined(o: &Out) -> String {
    format!("{}{}", o.stdout, o.stderr)
}

#[test]
fn gate_passes_a_doc_whose_receipts_resolve_hash_and_verify() {
    let r = Repo::new("ok");
    let id = r.receipt("1/3");
    let id2 = r.receipt("98 KB -> B");
    let doc = r.doc("0001.md", &format!("one third (calc:{id}), and 98 KB in bytes calc:{id2}.\nPlain prose with no ids is fine.\n"));
    for mode in ["full", "hash"] {
        for enforce in [false, true] {
            let o = r.gate(&[&doc], mode, enforce);
            assert_eq!(o.code, 0, "{mode} enforce={enforce}: {}", combined(&o));
            assert!(combined(&o).contains("2 receipt"), "{}", combined(&o));
        }
    }
}

#[test]
fn gate_finds_receipts_under_docs_calc_receipts_too() {
    let r = Repo::new("docs-calc");
    let id = r.receipt("1/3");
    std::fs::create_dir_all(r.root.join("docs/calc-receipts")).unwrap();
    std::fs::rename(
        r.receipt_path(&id),
        r.root.join("docs/calc-receipts").join(format!("{id}.json")),
    )
    .unwrap();
    let doc = r.doc("0002.md", &format!("calc:{id}\n"));
    let o = r.gate(&[&doc], "full", true);
    assert_eq!(o.code, 0, "{}", combined(&o));
}

#[test]
fn gate_fails_when_the_receipt_file_is_missing() {
    let r = Repo::new("missing");
    let id = r.receipt("1/3");
    let doc = r.doc("0003.md", &format!("calc:{id}\n"));
    std::fs::remove_file(r.receipt_path(&id)).unwrap();
    let o = r.gate(&[&doc], "full", true);
    assert_eq!(o.code, 1, "{}", combined(&o));
    assert!(
        combined(&o).contains("receipt missing") && combined(&o).contains(&id),
        "{}",
        combined(&o)
    );
    let o = r.gate(&[&doc], "hash", true);
    assert_eq!(o.code, 1);
}

#[test]
fn gate_fails_when_the_hash_is_wrong() {
    let r = Repo::new("hash");
    let id = r.receipt("1/3");
    let doc = r.doc("0004.md", &format!("calc:{id}\n"));
    let p = r.receipt_path(&id);
    let mut bytes = std::fs::read(&p).unwrap();
    bytes.push(b'\n'); // a trailing newline is enough to break the id
    std::fs::write(&p, bytes).unwrap();
    for mode in ["full", "hash"] {
        let o = r.gate(&[&doc], mode, true);
        assert_eq!(o.code, 1, "{mode}: {}", combined(&o));
        assert!(
            combined(&o).contains("hash wrong"),
            "{mode}: {}",
            combined(&o)
        );
    }
}

#[test]
fn gate_full_mode_fails_a_forged_receipt_that_has_a_matching_hash() {
    let r = Repo::new("forged");
    let id = r.receipt("1/3");
    // A well-formed lie filed under its own correct hash: hash mode cannot see it, full mode must.
    let forged = std::fs::read_to_string(r.receipt_path(&id))
        .unwrap()
        .replace("\"value\":\"1/3\"", "\"value\":\"1/4\"");
    let o = Command::new(BIN)
        .args(["b3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut o = o;
    o.stdin
        .take()
        .unwrap()
        .write_all(forged.as_bytes())
        .unwrap();
    let out = o.wait_with_output().unwrap();
    let forged_id = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    std::fs::write(r.receipt_path(&forged_id), &forged).unwrap();
    let doc = r.doc("0005.md", &format!("one third is calc:{forged_id}\n"));
    let hash_only = r.gate(&[&doc], "hash", true);
    assert_eq!(
        hash_only.code,
        0,
        "hash mode checks presence and hash only: {}",
        combined(&hash_only)
    );
    let full = r.gate(&[&doc], "full", true);
    assert_eq!(full.code, 1, "{}", combined(&full));
    assert!(
        combined(&full).contains("verify mismatch") && combined(&full).contains("result.value"),
        "{}",
        combined(&full)
    );
}

#[test]
fn gate_is_warn_only_unless_enforced() {
    let r = Repo::new("warn");
    let id = r.receipt("1/3");
    let doc = r.doc("0006.md", &format!("calc:{id}\n"));
    std::fs::remove_file(r.receipt_path(&id)).unwrap();
    let o = r.gate(&[&doc], "full", false);
    assert_eq!(o.code, 0, "warn-only exits 0");
    assert!(
        combined(&o).contains("WARN") && combined(&o).contains("receipt missing"),
        "{}",
        combined(&o)
    );
    assert!(combined(&o).contains("warn-only"), "{}", combined(&o));
}

#[test]
fn gate_flags_short_ids_and_skips_files_outside_the_checked_directories() {
    let r = Repo::new("short");
    let doc = r.doc("0007.md", "see calc:0123456789ab for the figure\n");
    let o = r.gate(&[&doc], "hash", true);
    assert_eq!(o.code, 1);
    assert!(combined(&o).contains("short id"), "{}", combined(&o));
    let outside = r.root.join("notes.md");
    std::fs::write(&outside, "calc:0123456789ab\n").unwrap();
    let o = r.gate(&[&outside], "hash", true);
    assert_eq!(o.code, 0, "{}", combined(&o));
}

#[test]
fn gate_with_no_ids_in_a_checked_doc_passes() {
    let r = Repo::new("none");
    let doc = r.doc("0008.md", "No ids here.\n");
    assert_eq!(r.gate(&[&doc], "full", true).code, 0);
}

#[test]
fn model_names_with_a_size_suffix_are_not_byte_claims() {
    let deny = [("CALC_GUARD_MODE", "deny")];
    let flagged = |text: &str| {
        decision(&run_guard(
            &input("Write", "/r/specs/x.md", "content", text),
            &deny,
        ))
        .is_some()
    };
    // 30B / 1B / 4B here mean billions of parameters, not bytes
    assert!(!flagged(
        "Qwen3-30B, Llama-3.2-1B, OWN-30B, Qwen3-4B and a 7B model"
    ));
    assert!(!flagged("the 31B | 8B | 70B columns"));
    // real byte claims still count
    assert!(flagged("a 4096 B tile, 16 B per lane, 8 B header"));
}
