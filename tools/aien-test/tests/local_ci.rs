//! The local pipeline end to end, and the negative cases the mission lists
//! (22 to 27): an altered receipt, a bad signature, a bundle from another
//! commit, a missing required check, a hardware-scoped cached result under
//! `--release`, and an executor whose key the repository does not trust.
//! Each must leave MERGE ELIGIBILITY at BLOCKED; none may print PASS.
//!
//! The worker runs with `--sandbox none` (development mode,
//! `allow_unsandboxed`) here because the exec helper is the INFRA lane's
//! file; the sandbox tests at the end run only when
//! `/usr/local/sbin/aien-run-exec` exists.

use aien_test::bundle;
use aien_test::evidence::canonical;
use aien_test::plan::{self, Request};
use aien_test::runner::{self, Options};
use aien_test::sandbox::Sandbox;
use aien_test::verdict::Verdict;
use aien_test::verify::{self, Ask};
use aien_test::worker::{self, Config, StateDir};
use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);
/// The user-sandbox scenarios share one mirror name under
/// `/var/lib/aien-run/mirror` (`aien-test-selftest.git`): they take turns.
static USER_SANDBOX: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p =
            std::env::temp_dir().join(format!("aien-local-ci-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
    /// A directory the exec helper accepts as `--cwd`: the sandbox shares no
    /// `/tmp` with the caller, so sandboxed checks run in
    /// `/var/lib/aien-run/wt/aien-test-selftest/<40 hex>`.
    fn sandbox_wt() -> Tmp {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = format!(
            "{:0>40}",
            format!("{:x}{:x}{n:x}", std::process::id(), nanos)
        );
        let p = Path::new("/var/lib/aien-run/wt/aien-test-selftest").join(&id[id.len() - 40..]);
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn write(dir: &Path, rel: &str, content: &str, exec: bool) {
    let p = dir.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, content).unwrap();
    if exec {
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

fn gate(id: &str, input: &str, exec: &str, extra: &str) -> String {
    format!(
        "gate: {id}\nmanifest_version: 2\nowner: t\nrequires:\n  - host\ninputs:\n  - {input}\nrun:\n  exec: {exec}\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 10s\n{extra}"
    )
}

/// Everything one scenario needs: an origin with `main` (base) and `topic`
/// (head, changes `a/a.txt`), a trusted clone, a state dir, a signing key
/// whose public half is committed in `.aien/verify.pub`, and a fake `crumb`
/// that always says OK.
struct World {
    _root: Tmp,
    origin: PathBuf,
    clone: PathBuf,
    state: StateDir,
    key: PathBuf,
    other_key: PathBuf,
    crumb: PathBuf,
    base: String,
    head: String,
    /// A second head that changes `h/h.txt` (gate H, cache_scope hardware).
    head_h: String,
    wt_root: PathBuf,
}

const REPO: &str = "aien-test-selftest";

fn world() -> World {
    let root = Tmp::new("world");
    let r = root.path();
    let key = r.join("verify.key");
    let pubkey = bundle::keygen(&key).unwrap();
    let other_key = r.join("other.key");
    bundle::keygen(&other_key).unwrap();
    let crumb = r.join("crumb");
    write(
        r,
        "crumb",
        "#!/bin/sh\nif [ \"$1\" = close ]; then case \"$5\" in create|modify|delete|audit|test|build) ;; *) echo \"crumb: action must be create|modify|delete|audit|test|build\" >&2; exit 2;; esac; fi\necho \"crumb $1: OK\"\nexit 0\n",
        true,
    );

    let origin = r.join("origin");
    fs::create_dir_all(&origin).unwrap();
    git(&origin, &["init", "-q", "-b", "main"]);
    write(&origin, "tools/a.sh", "#!/bin/sh\necho PASS\n", true);
    write(&origin, "tools/h.sh", "#!/bin/sh\necho PASS\n", true);
    write(
        &origin,
        "tests/gates/A.gate",
        &gate("A", "a/", "tools/a.sh", ""),
        false,
    );
    write(
        &origin,
        "tests/gates/H.gate",
        &gate("H", "h/", "tools/h.sh", "cache_scope: hardware\n"),
        false,
    );
    write(&origin, "a/a.txt", "1\n", false);
    write(&origin, "h/h.txt", "1\n", false);
    write(
        &origin,
        ".aien/verify.pub",
        &format!("# trusted executors\n{pubkey}\n"),
        false,
    );
    write(&origin, ".crumb", "crumb: 1\n", false);
    git(&origin, &["add", "-A"]);
    git(&origin, &["commit", "-q", "-m", "base"]);
    let base = git(&origin, &["rev-parse", "HEAD"]);
    git(&origin, &["checkout", "-q", "-b", "topic"]);
    write(&origin, "a/a.txt", "2\n", false);
    git(&origin, &["add", "-A"]);
    git(&origin, &["commit", "-q", "-m", "change a"]);
    let head = git(&origin, &["rev-parse", "HEAD"]);
    git(&origin, &["checkout", "-q", "-b", "topic-h", "main"]);
    write(&origin, "h/h.txt", "2\n", false);
    git(&origin, &["add", "-A"]);
    git(&origin, &["commit", "-q", "-m", "change h"]);
    let head_h = git(&origin, &["rev-parse", "HEAD"]);
    git(&origin, &["checkout", "-q", "main"]);

    let clones = r.join("clones");
    fs::create_dir_all(&clones).unwrap();
    let clone = clones.join(REPO);
    git(&clones, &["clone", "-q", origin.to_str().unwrap(), REPO]);
    git(&clone, &["fetch", "-q", "--all"]);
    World {
        state: StateDir::new(&r.join("state")),
        wt_root: r.join("wt"),
        _root: root,
        origin,
        clone,
        key,
        other_key,
        crumb,
        base,
        head,
        head_h,
    }
}

impl World {
    fn config(&self) -> Config {
        let mut cfg = Config::new(
            self.state.clone(),
            &self.key,
            self.clone.parent().unwrap(),
            &self.wt_root,
        );
        cfg.crumb = Some(self.crumb.display().to_string());
        cfg.gh = self.crumb.display().to_string(); // never called unless github
                                                   // These scenarios run without the exec helper (development mode).
        cfg.allow_unsandboxed = true;
        cfg
    }

    fn ask(&self, commit: &str) -> Ask {
        Ask {
            state: self.state.clone(),
            commit: commit.to_string(),
            base: "origin/main".to_string(),
            repository: None,
            release: false,
            operator: false,
            hardware_identity: String::new(),
            crumb: Some(self.crumb.display().to_string()),
            keys_override: None,
        }
    }

    fn verify(&self, ask: &Ask) -> verify::Report {
        verify::run(&self.clone, ask)
    }

    fn bundle_path(&self, sha: &str) -> PathBuf {
        self.state.bundle_path(REPO, sha)
    }

    fn read_bundle(&self, sha: &str) -> Value {
        serde_json::from_slice(&fs::read(self.bundle_path(sha)).unwrap()).unwrap()
    }

    fn write_bundle(&self, sha: &str, v: &Value) {
        let p = self.bundle_path(sha);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, canonical(v).unwrap()).unwrap();
    }
}

fn line<'a>(r: &'a verify::Report, prefix: &str) -> &'a str {
    r.lines
        .iter()
        .find(|l| l.starts_with(prefix))
        .map(String::as_str)
        .unwrap_or_else(|| panic!("no line {prefix:?} in {:#?}", r.lines))
}

fn assert_blocked(r: &verify::Report, why: &str) {
    assert!(!r.ready, "{why}: {:#?}", r.lines);
    assert_eq!(
        line(r, "MERGE ELIGIBILITY:"),
        "MERGE ELIGIBILITY: BLOCKED",
        "{why}"
    );
    assert_ne!(line(r, "Final verdict:"), "Final verdict: PASS", "{why}");
}

/// The happy path once, shared by the negative tests through `verified_world`.
fn verified_world() -> World {
    let w = world();
    let s = worker::run_once(&w.config(), REPO, &w.head).expect("worker");
    assert_eq!(s.verdict, Verdict::Pass, "{} {:?}", s.reason, s.notes);
    assert_eq!(s.bundle_verdict, "PASS");
    // The fake crumb rejects actions the real tool rejects; no crumb note.
    assert!(
        !s.notes.iter().any(|n| n.starts_with("crumb ")),
        "{:?}",
        s.notes
    );
    assert!(w.bundle_path(&w.head).exists());
    assert_eq!(s.publication, "local");
    w
}

#[test]
fn worker_then_verify_is_ready() {
    let w = verified_world();
    let r = w.verify(&w.ask(&w.head));
    assert_eq!(r.lines[0], "AIEN LOCAL VERIFICATION");
    assert_eq!(line(&r, "Evidence status:"), "Evidence status: VERIFIED");
    assert_eq!(line(&r, "Crumb status:"), "Crumb status: CURRENT");
    assert_eq!(
        line(&r, "Hardware status:"),
        "Hardware status: NOT_REQUIRED"
    );
    assert_eq!(line(&r, "Final verdict:"), "Final verdict: PASS");
    assert_eq!(line(&r, "Executed gates:"), "Executed gates: 1 (A)");
    assert_eq!(line(&r, "Build time:"), "Build time: n/a");
    assert!(line(&r, "Test time:").ends_with(" ms"));
    assert_eq!(
        line(&r, "MERGE ELIGIBILITY:"),
        "MERGE ELIGIBILITY: READY",
        "{:#?}",
        r.lines
    );
    assert!(r.ready);
    // The journal has the head, the worktree is gone, the clone's checkout is untouched.
    let j = fs::read_to_string(w.state.journal()).unwrap();
    assert!(j.contains(&w.head) && j.contains("\"publication\":\"local\""));
    assert!(!w.wt_root.join(REPO).join(&w.head).exists());
    assert_eq!(git(&w.clone, &["rev-parse", "HEAD"]), w.base);
    assert_eq!(git(&w.clone, &["status", "--porcelain"]), "");
    // Fresh crumb check leaves no worktree behind.
    assert!(!git(&w.clone, &["worktree", "list"]).contains("aien-verify-"));
    let _ = &w.origin;
}

#[test]
fn altered_receipt_is_rejected() {
    let w = verified_world();
    let b = w.read_bundle(&w.head);
    let digest = b["checks"][0]["receipt"].as_str().unwrap().to_string();
    let p = w
        .state
        .evidence(REPO, &w.head)
        .join(format!("{digest}.json"));
    let text = fs::read_to_string(&p).unwrap();
    assert!(text.contains("\"derived_verdict\":\"PASS\""));
    fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(
        &p,
        text.replace(
            "\"derived_verdict\":\"PASS\"",
            "\"derived_verdict\":\"FAIL\"",
        ),
    )
    .unwrap();
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "altered receipt");
    assert!(
        r.lines.iter().any(|l| l.starts_with("Problem: A: receipt")),
        "{:#?}",
        r.lines
    );
}

#[test]
fn invalid_signature_is_rejected() {
    let w = verified_world();
    let mut b = w.read_bundle(&w.head);
    let sig = b["signature"]["sig"].as_str().unwrap().to_string();
    let flipped = format!(
        "{}{}",
        if sig.starts_with('0') { "1" } else { "0" },
        &sig[1..]
    );
    b["signature"]["sig"] = json!(flipped);
    w.write_bundle(&w.head, &b);
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "bad signature");
    assert_eq!(line(&r, "Evidence status:"), "Evidence status: INVALID");
    assert!(line(&r, "Evidence note:").contains("BAD_SIGNATURE"));
}

#[test]
fn bundle_from_another_commit_is_rejected() {
    let w = verified_world();
    let b = w.read_bundle(&w.head);
    // Present the head's bundle as the other branch head's.
    w.write_bundle(&w.head_h, &b);
    let r = w.verify(&w.ask(&w.head_h));
    assert_blocked(&r, "other commit");
    assert!(
        line(&r, "Evidence note:").contains("COMMIT_MISMATCH"),
        "{:#?}",
        r.lines
    );
}

#[test]
fn missing_required_check_never_passes() {
    let w = verified_world();
    let mut b = w.read_bundle(&w.head);
    b["checks"] = json!([]);
    b.as_object_mut().unwrap().remove("signature");
    // Re-signed by the trusted key: the algebra, not the signature, must refuse it.
    let key = bundle::load_signing_key(&w.key).unwrap();
    let signed = bundle::sign(&b, &key).unwrap();
    assert_eq!(
        signed["verdict"], "PASS",
        "an empty bundle declares PASS vacuously"
    );
    w.write_bundle(&w.head, &signed);
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "missing required check");
    assert!(
        line(&r, "Evidence note:").contains("MISSING_REQUIRED_CHECK: A"),
        "{:#?}",
        r.lines
    );
}

#[test]
fn hardware_cached_result_cannot_satisfy_release() {
    let w = world();
    // A hand-made bundle for the H head: H cached (PASS), signed by the trusted key.
    let wt = w.wt_root.join("h-plan");
    git(
        &w.clone,
        &[
            "worktree",
            "add",
            "-q",
            "--detach",
            wt.to_str().unwrap(),
            &w.head_h,
        ],
    );
    let p = plan::compute(&wt, &Request::new(&w.base, &w.head_h)).unwrap();
    git(
        &w.clone,
        &["worktree", "remove", "--force", wt.to_str().unwrap()],
    );
    let h = p
        .entries
        .iter()
        .find(|e| e.gate == "H")
        .expect("H selected");
    let key = bundle::load_signing_key(&w.key).unwrap();
    let vk = key.verifying_key();
    let tree = git(&w.clone, &["rev-parse", &format!("{}^{{tree}}", w.head_h)]);
    let unsigned = json!({
        "schema": bundle::SCHEMA,
        "repository": plan::repository_name(&w.clone, None),
        "commit": w.head_h,
        "tree_digest": aien_test::evidence::sha256_hex(tree.as_bytes()),
        "graph_digest": p.graph_digest,
        "receipt_id_scheme": "sha256-canonical-json",
        "checks": [{
            "gate": "H", "check_id": h.check_id, "exactness": "EXACT", "required": true,
            "status": "cached", "verdict": "PASS", "backend": "host",
            "resolution": { "resolved_from": "f".repeat(64) }
        }],
        "verdict": "PASS",
        "executor": { "id": bundle::fingerprint(&vk), "class": "spark", "machine_digest": "0".repeat(64) },
        "started_utc": "2026-10-11T00:00:00Z",
        "finished_utc": "2026-10-11T00:00:02Z",
    });
    let signed = bundle::sign(&unsigned, &key).unwrap();
    w.write_bundle(&w.head_h, &signed);
    let plain = w.verify(&w.ask(&w.head_h));
    assert_eq!(
        line(&plain, "Evidence status:"),
        "Evidence status: VERIFIED",
        "{:#?}",
        plain.lines
    );
    assert_eq!(line(&plain, "Cached gates:"), "Cached gates: 1 (H)");
    assert_eq!(line(&plain, "Final verdict:"), "Final verdict: PASS");
    assert_eq!(line(&plain, "Total time:"), "Total time: 2000 ms");
    let mut release = w.ask(&w.head_h);
    release.release = true;
    let r = w.verify(&release);
    assert_blocked(&r, "hardware cache under --release");
    assert_eq!(line(&r, "Cached gates:"), "Cached gates: none");
    assert!(
        r.lines
            .iter()
            .any(|l| l.contains("cannot satisfy --release")),
        "{:#?}",
        r.lines
    );
}

#[test]
fn executor_not_in_verify_pub_is_rejected() {
    let w = verified_world();
    let mut b = w.read_bundle(&w.head);
    b.as_object_mut().unwrap().remove("signature");
    let other = bundle::load_signing_key(&w.other_key).unwrap();
    b["executor"]["id"] = json!(bundle::fingerprint(&other.verifying_key()));
    let signed = bundle::sign(&b, &other).unwrap();
    w.write_bundle(&w.head, &signed);
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "untrusted executor");
    assert!(
        line(&r, "Evidence note:").contains("UNKNOWN_KEY"),
        "{:#?}",
        r.lines
    );
    // And with no trusted keys at all the answer is the same.
    let mut none = w.ask(&w.head);
    none.keys_override = Some(Vec::new());
    let r = w.verify(&none);
    assert_blocked(&r, "no keys");
}

#[test]
fn another_uid_cannot_read_the_key() {
    let w = world();
    let probe = Command::new("sudo")
        .args(["-n", "-u", "nobody", "true"])
        .output();
    match probe {
        Ok(o) if o.status.success() => {}
        _ => {
            println!("SKIP: sudo -n -u nobody is not available here");
            return;
        }
    }
    let mode = fs::metadata(&w.key).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let out = Command::new("sudo")
        .args(["-n", "-u", "nobody", "cat", w.key.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "nobody must not read the signing key"
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn github_down_queues_then_flushes() {
    let w = world();
    let fake = w.state.root.join("fake-gh");
    let log = w.state.root.join("gh.log");
    let down = w.state.root.join("gh.down");
    fs::create_dir_all(&w.state.root).unwrap();
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nif [ -e {} ]; then echo 'connect: network unreachable' >&2; exit 1; fi\necho \"$@\" >> {}\nexit 0\n",
            down.display(),
            log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(&down, "").unwrap();
    let mut cfg = w.config();
    cfg.github = true;
    cfg.gh = fake.display().to_string();
    let s = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert_eq!(s.verdict, Verdict::Pass, "{}", s.reason);
    assert_eq!(s.publication, "pending");
    assert!(
        s.notes.iter().any(|n| n.starts_with("GitHub unreachable")),
        "{:?}",
        s.notes
    );
    let queued: Vec<_> = fs::read_dir(w.state.outbox_dir())
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(queued.len(), 1);
    assert!(fs::read_to_string(w.state.journal())
        .unwrap()
        .contains("\"publication\":\"pending\""));
    // Still down: nothing posted.
    assert_eq!(worker::flush_outbox(&cfg), (0, 1));
    // Back: posted, outbox empty, the status names the bundle and is a success.
    fs::remove_file(&down).unwrap();
    assert_eq!(worker::flush_outbox(&cfg), (1, 0));
    assert_eq!(
        fs::read_dir(w.state.outbox_dir())
            .unwrap()
            .flatten()
            .count(),
        0
    );
    let l = fs::read_to_string(&log).unwrap();
    assert!(l.contains(&format!("statuses/{}", w.head)), "{l}");
    assert!(
        l.contains("state=success") && l.contains("context=AIEN Verification (Spark)"),
        "{l}"
    );
    // The description names the bundle; GitHub caps it at 140 characters, so
    // a long scratch path may cut the file name off (no assertion on ".json").
    assert!(l.contains(": bundle /"), "{l}");
    assert!(fs::read_to_string(w.state.journal())
        .unwrap()
        .contains("flushed from outbox"));
}

#[test]
fn loop_without_github_keeps_the_outbox() {
    // A status queued while GitHub was down must not be posted by a loop that
    // was started without --github (`aien-local-ci start`): posting is the
    // operator's choice, made with --github.
    let w = world();
    let fake = w.state.root.join("fake-gh");
    let log = w.state.root.join("gh.log");
    let down = w.state.root.join("gh.down");
    fs::create_dir_all(&w.state.root).unwrap();
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nif [ -e {} ]; then echo 'connect: network unreachable' >&2; exit 1; fi\necho \"$@\" >> {}\nexit 0\n",
            down.display(),
            log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(&down, "").unwrap();
    let mut cfg = w.config();
    cfg.github = true;
    cfg.gh = fake.display().to_string();
    let s = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert_eq!(s.publication, "pending");
    assert_eq!(worker::outbox_files(&w.state).len(), 1);
    // GitHub is back, but this loop runs without --github: nothing is posted.
    fs::remove_file(&down).unwrap();
    cfg.github = false;
    worker::run_loop(&cfg, Some(1));
    assert_eq!(worker::outbox_files(&w.state).len(), 1);
    assert!(!log.exists(), "{:?}", fs::read_to_string(&log));
    // With --github the same loop drains it.
    cfg.github = true;
    cfg.repos = Vec::new();
    worker::run_loop(&cfg, Some(1));
    assert!(worker::outbox_files(&w.state).is_empty());
    assert!(fs::read_to_string(&log)
        .unwrap()
        .contains(&format!("statuses/{}", w.head)));
}

#[test]
fn worker_no_cache_reruns_every_gate() {
    // The same head twice: the second run reuses the receipt; with
    // `--no-cache` (cfg.no_cache) it runs the gate again.
    let w = world();
    let mut cfg = w.config();
    let first = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert!(
        first.gates.iter().any(|(g, st, _)| g == "A" && st == "run"),
        "{:?}",
        first.gates
    );
    let again = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert!(
        again
            .gates
            .iter()
            .any(|(g, st, _)| g == "A" && st == "cached"),
        "{:?}",
        again.gates
    );
    cfg.no_cache = true;
    cfg.jobs = aien_test::resources::JobsSpec {
        host: Some(1),
        qemu: None,
    };
    let fresh = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert!(
        fresh.gates.iter().any(|(g, st, _)| g == "A" && st == "run"),
        "{:?}",
        fresh.gates
    );
    assert_eq!(fresh.verdict, Verdict::Pass, "{}", fresh.reason);
}

#[test]
fn trigger_file_is_served_by_the_loop() {
    let w = world();
    let cfg = w.config();
    fs::create_dir_all(w.state.trigger_dir()).unwrap();
    fs::write(w.state.trigger_dir().join(format!("{REPO}.{}", w.head)), "").unwrap();
    let served = worker::run_loop(&cfg, Some(1));
    assert_eq!(served, 1);
    assert!(w.bundle_path(&w.head).exists());
    assert_eq!(
        fs::read_dir(w.state.trigger_dir())
            .unwrap()
            .flatten()
            .count(),
        0
    );
}

#[test]
fn bare_exec_runs_and_escapes_are_refused() {
    // `exec: true` (bare) runs from the PATH; `exec: ../x` is a manifest reject.
    let t = Tmp::new("bare");
    let r = t.path();
    git(r, &["init", "-q", "-b", "main"]);
    write(r, "tests/gates/T.gate", &gate("T", "a/", "true", ""), false);
    write(r, "a/a.txt", "1\n", false);
    git(r, &["add", "-A"]);
    git(r, &["commit", "-q", "-m", "init"]);
    let ev = Tmp::new("bare-ev");
    let opts = Options::new(ev.path().join("out"));
    let o = runner::run_gate(r, &r.join("tests/gates/T.gate"), &opts).unwrap();
    assert_eq!(o.verdict, Verdict::Pass, "{}", o.reason);
    for bad in ["../../usr/bin/env", "/usr/bin/env"] {
        write(r, "tests/gates/T.gate", &gate("T", "a/", bad, ""), false);
        git(r, &["commit", "-q", "-am", "bad exec"]);
        match runner::run_gate(r, &r.join("tests/gates/T.gate"), &opts) {
            Err(runner::RunError::BadManifest(m)) => assert!(m.contains("run.exec"), "{m}"),
            other => panic!("{bad}: expected a manifest reject, got {other:?}"),
        }
    }
}

#[test]
fn user_sandbox_when_the_helper_exists() {
    if nested_in_sandbox("user_sandbox_when_the_helper_exists") {
        return;
    }
    if !Path::new(aien_test::sandbox::EXEC_HELPER).exists() {
        println!(
            "SKIP: {} is not installed (INFRA lane)",
            aien_test::sandbox::EXEC_HELPER
        );
        return;
    }
    let t = Tmp::sandbox_wt();
    let r = t.path();
    git(r, &["init", "-q", "-b", "main"]);
    write(r, "tools/a.sh", "#!/bin/sh\necho PASS\n", true);
    write(
        r,
        "tests/gates/A.gate",
        &gate("A", "a/", "tools/a.sh", ""),
        false,
    );
    write(r, "a/a.txt", "1\n", false);
    git(r, &["add", "-A"]);
    git(r, &["commit", "-q", "-m", "init"]);
    let ev = Tmp::new("sandbox-ev");
    let mut opts = Options::new(ev.path().join("out"));
    opts.sandbox = Sandbox::User;
    let o = runner::run_gate(r, &r.join("tests/gates/A.gate"), &opts).unwrap();
    println!("sandbox user: {} {}", o.verdict.as_str(), o.reason);
    assert_eq!(o.verdict, Verdict::Pass, "{}", o.reason);
}

// ---------------------------------------------------------------- ENGINE-2 review fixes

/// A fake `gh` that logs its arguments; `gh.down` makes every call fail and
/// `fail-<sha>` makes calls about that commit fail.
fn fake_gh(w: &World) -> (PathBuf, PathBuf, PathBuf) {
    let fake = w.state.root.join("fake-gh");
    let log = w.state.root.join("gh.log");
    let down = w.state.root.join("gh.down");
    fs::create_dir_all(&w.state.root).unwrap();
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nroot={}\nif [ -e \"$root/gh.down\" ]; then echo 'connect: network unreachable' >&2; exit 1; fi\nfor a in \"$@\"; do for f in \"$root\"/fail-*; do [ -e \"$f\" ] || continue; s=${{f##*/fail-}}; case \"$a\" in *\"$s\"*) echo 'refused' >&2; exit 1;; esac; done; done\necho \"$@\" >> {}\nexit 0\n",
            w.state.root.display(),
            log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    (fake, log, down)
}

#[test]
fn worker_refuses_unsandboxed_runs_and_bad_names() {
    let w = world();
    let mut cfg = w.config();
    cfg.allow_unsandboxed = false;
    let e = worker::run_once(&cfg, REPO, &w.head).unwrap_err();
    assert!(e.contains("--dev-unsandboxed"), "{e}");
    // Never with GitHub heads, even in development mode (the CLI applies it).
    cfg.allow_unsandboxed = true;
    let e = cfg.check_sandbox_policy_for(true).unwrap_err();
    assert!(e.contains("--github"), "{e}");
    let cfg = w.config();
    for (repo, sha) in [
        ("..", w.head.as_str()),
        ("a/b", w.head.as_str()),
        (".hidden", w.head.as_str()),
        (REPO, "HEAD"),
        (REPO, &w.head[..12]),
    ] {
        let e = worker::run_once(&cfg, repo, sha).unwrap_err();
        assert!(e.starts_with("refusing"), "{repo} {sha}: {e}");
    }
    // Nothing was checked out, signed or journaled.
    assert!(!w.wt_root.exists());
    assert!(!w.bundle_path(&w.head).exists());
    assert!(!w.state.journal().exists());
}

#[test]
fn worker_refuses_a_key_others_can_read() {
    let w = world();
    fs::set_permissions(&w.key, fs::Permissions::from_mode(0o644)).unwrap();
    let e = worker::run_once(&w.config(), REPO, &w.head).unwrap_err();
    assert!(e.contains("mode 644"), "{e}");
    assert!(!w.wt_root.exists());
    assert!(!w.bundle_path(&w.head).exists());
}

#[test]
fn github_success_needs_a_ready_verify() {
    let w = world();
    let (fake, log, _down) = fake_gh(&w);
    let mut cfg = w.config();
    cfg.github = true;
    cfg.gh = fake.display().to_string();
    // Signed by a key the repository does not list at BASE: the gates pass,
    // the bundle algebra says PASS, and still no success is posted.
    cfg.key = w.other_key.clone();
    let s = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert_eq!(s.verdict, Verdict::Pass, "{}", s.reason);
    assert_eq!(s.bundle_verdict, "PASS");
    assert!(!s.ready);
    assert_eq!(s.publication, "posted");
    let l = fs::read_to_string(&log).unwrap();
    assert!(l.contains("state=error"), "{l}");
    assert!(!l.contains("state=success"), "{l}");
    let j = fs::read_to_string(w.state.journal()).unwrap();
    assert!(j.contains("\"verify\":\"BLOCKED\""), "{j}");
    assert!(j.contains("UNKNOWN_KEY"), "{j}");
    // With the trusted key the same head is READY and success is posted.
    let mut cfg = cfg.clone();
    cfg.key = w.key.clone();
    let s = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    assert!(s.ready, "{:?}", s.notes);
    let l = fs::read_to_string(&log).unwrap();
    assert!(l.contains("state=success"), "{l}");
}

#[test]
fn outbox_replays_in_queue_order() {
    let w = world();
    let (fake, log, down) = fake_gh(&w);
    let mut cfg = w.config();
    cfg.github = true;
    cfg.gh = fake.display().to_string();
    fs::write(&down, "").unwrap();
    // Queue the heads so that queue order differs from name order.
    let (first, second) = if w.head > w.head_h {
        (w.head.clone(), w.head_h.clone())
    } else {
        (w.head_h.clone(), w.head.clone())
    };
    for sha in [&first, &second] {
        let s = worker::run_once(&cfg, REPO, sha).expect("worker");
        assert_eq!(s.publication, "pending");
    }
    let q = worker::outbox_files(&w.state);
    assert_eq!(q.len(), 2);
    assert!(q[0].to_string_lossy().contains(&first));
    // Still down: nothing posted, both pending.
    assert_eq!(worker::flush_outbox(&cfg), (0, 2));
    // Up, but GitHub refuses the oldest: the replay stops there, the newer
    // status does not overtake it.
    fs::remove_file(&down).unwrap();
    let block = w.state.root.join(format!("fail-{first}"));
    fs::write(&block, "").unwrap();
    assert_eq!(worker::flush_outbox(&cfg), (0, 2));
    assert!(!log.exists() || fs::read_to_string(&log).unwrap().is_empty());
    // Fully back: both posted, oldest first.
    fs::remove_file(&block).unwrap();
    assert_eq!(worker::flush_outbox(&cfg), (2, 0));
    assert!(worker::outbox_files(&w.state).is_empty());
    let l = fs::read_to_string(&log).unwrap();
    let a = l.find(&format!("statuses/{first}")).expect("first posted");
    let b = l
        .find(&format!("statuses/{second}"))
        .expect("second posted");
    assert!(a < b, "{l}");
}

#[test]
fn verify_rejects_another_check_id_or_manifest() {
    let w = verified_world();
    let key = bundle::load_signing_key(&w.key).unwrap();
    let original = w.read_bundle(&w.head);

    // A re-signed entry naming another check id.
    let mut b = original.clone();
    b.as_object_mut().unwrap().remove("signature");
    b["checks"][0]["check_id"] = json!("0".repeat(64));
    w.write_bundle(&w.head, &bundle::sign(&b, &key).unwrap());
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "check id");
    assert!(
        r.lines.iter().any(|l| l.contains("CHECK_ID_MISMATCH")),
        "{:#?}",
        r.lines
    );

    // A receipt made from another manifest, stored under its own digest and
    // named by a re-signed bundle.
    let digest = original["checks"][0]["receipt"].as_str().unwrap();
    let dir = w.state.evidence(REPO, &w.head);
    let mut receipt: Value =
        serde_json::from_slice(&fs::read(dir.join(format!("{digest}.json"))).unwrap()).unwrap();
    receipt["core"]["manifest_digest"] = json!("1".repeat(64));
    let (forged, _) = aien_test::evidence::Store::new(&dir)
        .put_receipt(&receipt)
        .unwrap();
    let mut b = original.clone();
    b.as_object_mut().unwrap().remove("signature");
    b["checks"][0]["receipt"] = json!(forged);
    w.write_bundle(&w.head, &bundle::sign(&b, &key).unwrap());
    let r = w.verify(&w.ask(&w.head));
    assert_blocked(&r, "manifest");
    assert!(
        r.lines.iter().any(|l| l.contains("MANIFEST_MISMATCH")),
        "{:#?}",
        r.lines
    );
    assert_eq!(line(&r, "Executed gates:"), "Executed gates: none");
}

#[test]
fn main_head_is_verified_against_its_first_parent() {
    let w = world();
    // Land the topic on main: the head is now the tip of origin/main.
    git(&w.origin, &["merge", "-q", "--ff-only", "topic"]);
    git(&w.clone, &["fetch", "-q", "--all"]);
    assert_eq!(git(&w.clone, &["rev-parse", "origin/main"]), w.head);
    let s = worker::run_once(&w.config(), REPO, &w.head).expect("worker");
    assert_eq!(s.verdict, Verdict::Pass, "{} {:?}", s.reason, s.notes);
    assert!(s.ready, "{:?}", s.notes);
    // `verify` with the default base reaches the same plan and is READY.
    let r = w.verify(&w.ask(&w.head));
    assert_eq!(
        line(&r, "MERGE ELIGIBILITY:"),
        "MERGE ELIGIBILITY: READY",
        "{:#?}",
        r.lines
    );
    assert_eq!(line(&r, "Executed gates:"), "Executed gates: 1 (A)");
}

#[test]
fn symlinked_exec_out_of_the_tree_never_runs() {
    let t = Tmp::new("symlink");
    let r = t.path();
    git(r, &["init", "-q", "-b", "main"]);
    fs::create_dir_all(r.join("tools")).unwrap();
    std::os::unix::fs::symlink("/bin/true", r.join("tools/run.sh")).unwrap();
    write(
        r,
        "tests/gates/T.gate",
        &gate("T", "a/", "tools/run.sh", ""),
        false,
    );
    write(r, "a/a.txt", "1\n", false);
    git(r, &["add", "-A"]);
    git(r, &["commit", "-q", "-m", "init"]);
    let ev = Tmp::new("symlink-ev");
    let opts = Options::new(ev.path().join("out"));
    let o = runner::run_gate(r, &r.join("tests/gates/T.gate"), &opts).unwrap();
    assert_eq!(o.verdict, Verdict::MissingImplementation, "{}", o.reason);
    assert_eq!(runner::exec_path(r, "tools/run.sh"), None);
}

#[test]
fn worker_in_the_user_sandbox_is_ready() {
    if nested_in_sandbox("worker_in_the_user_sandbox_is_ready") {
        return;
    }
    let _turn = USER_SANDBOX.lock().unwrap_or_else(|e| e.into_inner());
    if !Path::new(aien_test::sandbox::EXEC_HELPER).exists() {
        println!(
            "SKIP: {} is not installed (INFRA lane)",
            aien_test::sandbox::EXEC_HELPER
        );
        return;
    }
    let w = world();
    let mut cfg = w.config();
    cfg.allow_unsandboxed = false;
    cfg.sandbox = Sandbox::User;
    if !wt_tool().is_file() {
        println!(
            "SKIP: {} is not in this tree (INFRA lane)",
            wt_tool().display()
        );
        return;
    }
    cfg.wt_root = PathBuf::from("/var/lib/aien-run/wt");
    cfg.wt_tool = Some(wt_tool());
    let mirror = Path::new(worker::MIRROR_ROOT).join(format!("{REPO}.git"));
    let _mirror = MirrorCleanup(mirror.clone());
    let s = worker::run_once(&cfg, REPO, &w.head).expect("worker");
    println!(
        "worker --sandbox user: {} {} {:?}",
        s.verdict.as_str(),
        s.reason,
        s.notes
    );
    // The wt.sh worktree is gone again and no note says otherwise.
    assert!(
        !Path::new("/var/lib/aien-run/wt")
            .join(REPO)
            .join(&w.head)
            .exists(),
        "worktree left behind"
    );
    assert!(mirror.join("HEAD").exists(), "wt.sh keeps its mirror");
    assert_eq!(s.verdict, Verdict::Pass, "{} {:?}", s.reason, s.notes);
    assert!(s.ready, "{:?}", s.notes);
    let r = w.verify(&w.ask(&w.head));
    assert_eq!(
        line(&r, "MERGE ELIGIBILITY:"),
        "MERGE ELIGIBILITY: READY",
        "{:#?}",
        r.lines
    );
}

#[test]
fn worker_cli_refuses_unsandboxed_github_and_unflagged_none() {
    let bin = env!("CARGO_BIN_EXE_aien-test");
    let t = Tmp::new("cli");
    let state = t.path().join("state");
    let sha = "a".repeat(40);
    let run = |extra: &[&str]| {
        let mut args = vec![
            "worker",
            "--once",
            "repo",
            sha.as_str(),
            "--state-dir",
            state.to_str().unwrap(),
            "--clones",
            t.path().to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        Command::new(bin).args(&args).output().unwrap()
    };
    let o = run(&["--sandbox", "none"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--dev-unsandboxed"));
    let o = run(&["--sandbox", "none", "--dev-unsandboxed", "--github"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--github"));
    // Nothing ran: no state was written.
    assert!(!state.exists());
}

// ---------------------------------------------------------------- INFRA-2 alignment

/// INFRA's worktree tool, from this repository.
fn wt_tool() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../aien-local-ci/wt.sh")
}

/// Removes a test mirror under `/var/lib/aien-run/mirror` when dropped.
struct MirrorCleanup(PathBuf);

impl Drop for MirrorCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn worker_refuses_the_user_sandbox_without_the_wt_tool() {
    let w = world();
    let mut cfg = w.config();
    cfg.sandbox = Sandbox::User;
    cfg.wt_tool = None;
    let e = worker::run_once(&cfg, REPO, &w.head).unwrap_err();
    assert!(e.contains("--wt-tool"), "{e}");
    // Nothing was checked out.
    assert!(!w.wt_root.join(REPO).join(&w.head).exists());
}

#[test]
fn only_a_wt_sh_link_is_trusted() {
    let t = Tmp::new("wt-link");
    let wt = t.path().join("wt");
    let mirror = t.path().join("m.git");
    fs::create_dir_all(&wt).unwrap();
    // Owned by the test user, not root: refused even with the right target.
    fs::write(
        wt.join(".git"),
        format!("gitdir: {}/worktrees/x\n", mirror.display()),
    )
    .unwrap();
    // Mode 0644 as wt.sh leaves it, so only the owner is wrong.
    fs::set_permissions(wt.join(".git"), fs::Permissions::from_mode(0o644)).unwrap();
    let e = worker::check_worktree_link(&wt, &mirror).unwrap_err();
    assert!(e.contains("root-owned"), "{e}");
    // A real wt.sh worktree passes when one is present on this host.
    let real = Path::new("/var/lib/aien-run/wt/aien-sovereign-core");
    let found = fs::read_dir(real).ok().and_then(|mut rd| {
        rd.find_map(|e| {
            // The run directory layout (`<sha>/<repo>/`), or the older flat one.
            e.ok()
                .map(|e| e.path())
                .map(|p| {
                    let nested = p.join("aien-sovereign-core");
                    if nested.join(".git").is_file() {
                        nested
                    } else {
                        p
                    }
                })
                .filter(|p| p.join(".git").is_file())
        })
    });
    if let Some(p) = found {
        let m = Path::new(worker::MIRROR_ROOT).join("aien-sovereign-core.git");
        worker::check_worktree_link(&p, &m).expect("wt.sh worktree");
        let other = Path::new(worker::MIRROR_ROOT).join("elsewhere.git");
        assert!(worker::check_worktree_link(&p, &other).is_err());
    } else {
        println!("SKIP real wt.sh worktree: none present");
    }
}

#[test]
fn a_wrong_target_dir_is_refused_by_the_helper() {
    if nested_in_sandbox("a_wrong_target_dir_is_refused_by_the_helper") {
        return;
    }
    if !Path::new(aien_test::sandbox::EXEC_HELPER).exists() {
        println!(
            "SKIP: {} is not installed (INFRA lane)",
            aien_test::sandbox::EXEC_HELPER
        );
        return;
    }
    // The helper itself: exit 2, its prefix on stderr, nothing ran.
    let t = Tmp::sandbox_wt();
    let r = t.path();
    let bad = "/var/lib/aien-run/target/aien-test-selftest/other";
    let (program, args) = aien_test::sandbox::command(
        Sandbox::User,
        Path::new("/bin/true"),
        &[],
        r,
        10,
        "host",
        &[("CARGO_TARGET_DIR".to_string(), bad.to_string())],
    );
    let out = Command::new(program).args(&args).output().unwrap();
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stderr.starts_with(b"aien-run-exec:"), "{out:?}");
    assert!(runner::sandbox_refused(
        Sandbox::User,
        out.status.code(),
        &out.stderr
    ));
    assert!(!runner::sandbox_refused(
        Sandbox::None,
        out.status.code(),
        &out.stderr
    ));
    assert!(!runner::sandbox_refused(
        Sandbox::User,
        Some(2),
        b"own error\n"
    ));

    // Through the runner: a gate that expects exit 2 still never passes on
    // a refusal, and its receipt has no exit status.
    git(r, &["init", "-q", "-b", "main"]);
    write(
        r,
        "tests/gates/R.gate",
        &gate("R", "a/", "true", "").replace("exit: 0", "exit: 2"),
        false,
    );
    write(r, "a/a.txt", "1\n", false);
    git(r, &["add", "-A"]);
    git(r, &["commit", "-q", "-m", "init"]);
    let ev = Tmp::new("refused-ev");
    let mut opts = Options::new(ev.path().join("out"));
    opts.sandbox = Sandbox::User;
    opts.cargo_target_dir = Some(PathBuf::from(bad));
    let o = runner::run_gate(r, &r.join("tests/gates/R.gate"), &opts).unwrap();
    assert_eq!(o.verdict, Verdict::Fail, "{}", o.reason);
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(o.receipt_path.unwrap()).unwrap()).unwrap();
    assert_eq!(receipt["core"]["exit_status"], serde_json::Value::Null);
}

// ---------------------------------------------------------------- SANDBOX lane: sibling checkouts

/// The sibling source repository of these scenarios (its trusted clone sits
/// next to the world's clone, its mirror under `/var/lib/aien-run/mirror`).
const SIB: &str = "aien-test-selftest-sib";

/// A gate S that passes only when `../sib/lib/x.txt` says `hello`; as
/// `aien-run` it also demands that neither the sibling nor the run directory
/// is writable. Returns the head that adds S, the sibling's commit and a
/// sibling map naming it.
fn sibling_world(w: &World) -> (String, String, PathBuf) {
    let root = w.origin.parent().unwrap().to_path_buf();
    let so = root.join("sib-origin");
    fs::create_dir_all(&so).unwrap();
    git(&so, &["init", "-q", "-b", "main"]);
    write(&so, "lib/x.txt", "hello\n", false);
    git(&so, &["add", "-A"]);
    git(&so, &["commit", "-q", "-m", "sib"]);
    let sib_sha = git(&so, &["rev-parse", "HEAD"]);
    let clones = w.clone.parent().unwrap();
    git(clones, &["clone", "-q", so.to_str().unwrap(), SIB]);

    git(&w.origin, &["checkout", "-q", "-b", "topic-s", "main"]);
    write(
        &w.origin,
        "tools/s.sh",
        "#!/bin/sh\n\
         test \"$(cat ../sib/lib/x.txt)\" = hello || { echo 'no sibling' >&2; exit 1; }\n\
         echo \"AIEN-OBS sibling_commit \\\"$(git -C ../sib rev-parse HEAD)\\\"\"\n\
         if [ \"$(id -un)\" = aien-run ]; then\n\
           [ -w ../sib/lib/x.txt ] && { echo 'sibling writable' >&2; exit 1; }\n\
           [ -w .. ] && { echo 'run directory writable' >&2; exit 1; }\n\
         fi\n\
         echo PASS\n",
        true,
    );
    write(
        &w.origin,
        "tests/gates/S.gate",
        &gate("S", "tools/", "tools/s.sh", ""),
        false,
    );
    git(&w.origin, &["add", "-A"]);
    git(&w.origin, &["commit", "-q", "-m", "gate S reads ../sib"]);
    let head_s = git(&w.origin, &["rev-parse", "HEAD"]);
    git(&w.origin, &["checkout", "-q", "main"]);
    git(&w.clone, &["fetch", "-q", "--all"]);

    let map = root.join("siblings");
    fs::write(&map, format!("# test map\n{REPO} sib {SIB} main\n")).unwrap();
    (head_s, sib_sha, map)
}

fn gate_verdict(s: &worker::Summary, id: &str) -> Verdict {
    s.gates
        .iter()
        .find(|g| g.0 == id)
        .map(|g| g.2)
        .unwrap_or_else(|| panic!("no gate {id} in {:?}", s.gates))
}

#[test]
fn worker_provides_a_sibling_checkout_and_removes_it() {
    if nested_in_sandbox("worker_provides_a_sibling_checkout_and_removes_it") {
        return;
    }
    let w = world();
    let (head_s, sib_sha, map) = sibling_world(&w);
    let mut cfg = w.config();
    cfg.operator = true; // S is a new gate: a pipeline configuration change.

    // Negative control: no sibling map, so ../sib is missing and S fails.
    let s = worker::run_once(&cfg, REPO, &head_s).expect("worker");
    assert_eq!(gate_verdict(&s, "S"), Verdict::Fail, "{:?}", s.notes);

    cfg.siblings = Some(map);
    let s = worker::run_once(&cfg, REPO, &head_s).expect("worker");
    assert_eq!(
        gate_verdict(&s, "S"),
        Verdict::Pass,
        "{} {:?}",
        s.reason,
        s.notes
    );
    let want = format!("sibling ../sib: {SIB} origin/main {sib_sha}");
    assert!(s.notes.contains(&want), "{:?}", s.notes);
    // The run directory (checkout and sibling) is gone, and neither clone
    // still lists a worktree for it.
    assert!(!w.wt_root.join(REPO).join(&head_s).exists());
    let sib_clone = w.clone.parent().unwrap().join(SIB);
    for c in [&w.clone, &sib_clone] {
        let list = git(c, &["worktree", "list"]);
        assert_eq!(list.lines().count(), 1, "{list}");
    }
}

#[test]
fn a_broken_sibling_map_refuses_the_run_before_any_checkout() {
    let w = world();
    let mut cfg = w.config();
    let map = w.origin.parent().unwrap().join("bad-siblings");
    fs::write(&map, format!("{REPO} ../escape {SIB} main\n")).unwrap();
    cfg.siblings = Some(map.clone());
    let e = worker::run_once(&cfg, REPO, &w.head).unwrap_err();
    assert!(e.contains("sibling map line 1"), "{e}");
    assert!(!w.wt_root.join(REPO).join(&w.head).exists());
    // A map that cannot be read refuses too.
    cfg.siblings = Some(map.with_extension("missing"));
    let e = worker::run_once(&cfg, REPO, &w.head).unwrap_err();
    assert!(e.contains("cannot read the sibling map"), "{e}");
}

#[test]
fn worker_in_the_user_sandbox_reads_a_read_only_sibling() {
    if nested_in_sandbox("worker_in_the_user_sandbox_reads_a_read_only_sibling") {
        return;
    }
    let _turn = USER_SANDBOX.lock().unwrap_or_else(|e| e.into_inner());
    if !Path::new(aien_test::sandbox::EXEC_HELPER).exists() || !wt_tool().is_file() {
        println!("SKIP: exec helper or wt.sh not present (INFRA lane)");
        return;
    }
    let w = world();
    let (head_s, sib_sha, map) = sibling_world(&w);
    let mut cfg = w.config();
    cfg.allow_unsandboxed = false;
    cfg.sandbox = Sandbox::User;
    cfg.operator = true;
    cfg.wt_root = PathBuf::from("/var/lib/aien-run/wt");
    cfg.wt_tool = Some(wt_tool());
    cfg.siblings = Some(map);
    let _m1 = MirrorCleanup(Path::new(worker::MIRROR_ROOT).join(format!("{REPO}.git")));
    let _m2 = MirrorCleanup(Path::new(worker::MIRROR_ROOT).join(format!("{SIB}.git")));
    let s = worker::run_once(&cfg, REPO, &head_s).expect("worker");
    assert_eq!(
        gate_verdict(&s, "S"),
        Verdict::Pass,
        "{} {:?}",
        s.reason,
        s.notes
    );
    let want = format!("sibling ../sib: {SIB} origin/main {sib_sha}");
    assert!(s.notes.contains(&want), "{:?}", s.notes);
    assert!(
        !Path::new("/var/lib/aien-run/wt")
            .join(REPO)
            .join(&head_s)
            .exists(),
        "run directory left behind"
    );
}

/// True inside the user sandbox (`AIEN_SANDBOX_NESTED=1`, set by the worker's
/// sandbox wrapper): the five tests that start a sandbox or provision sibling
/// worktrees themselves cannot nest there (aien-run may not call the helper or
/// write `/var/lib/aien-run/wt`), so they skip and say so. They run everywhere
/// else: on GitHub runners and in operator runs. tests/gates/run.sh records
/// the skip as an observation.
fn nested_in_sandbox(test: &str) -> bool {
    let nested = std::env::var(aien_test::sandbox::NESTED_MARKER).as_deref() == Ok("1");
    if nested {
        println!(
            "SKIP-NESTED: {test}: inside the user sandbox ({}=1)",
            aien_test::sandbox::NESTED_MARKER
        );
    }
    nested
}

/// A new commit `name` on `branch` of the world's origin that writes `files`,
/// fetched into the trusted clone. Returns the commit.
fn commit_on(w: &World, branch: &str, files: &[(&str, &str)], name: &str) -> String {
    git(&w.origin, &["checkout", "-q", branch]);
    for (rel, content) in files {
        write(&w.origin, rel, content, false);
    }
    git(&w.origin, &["add", "-A"]);
    git(&w.origin, &["commit", "-q", "-m", name]);
    let sha = git(&w.origin, &["rev-parse", "HEAD"]);
    git(&w.origin, &["checkout", "-q", "main"]);
    git(&w.clone, &["fetch", "-q", "--all"]);
    sha
}

fn gate_status(s: &worker::Summary, id: &str) -> (String, Verdict) {
    s.gates
        .iter()
        .find(|g| g.0 == id)
        .map(|g| (g.1.clone(), g.2))
        .unwrap_or_else(|| panic!("no gate {id} in {:?}", s.gates))
}

/// With siblings, the worker reuses a result only for the same commit of
/// the repository and only while every sibling is at the same commit: the
/// sibling commits are in the cache key, and the evidence directory is the
/// head's own.
#[test]
fn worker_reuses_only_the_same_commit_while_its_sibling_commits_match() {
    if nested_in_sandbox("worker_reuses_only_the_same_commit_while_its_sibling_commits_match") {
        return;
    }
    let w = world();
    let (head_s, sib_sha, map) = sibling_world(&w);
    let mut cfg = w.config();
    cfg.operator = true; // S is a new gate: a pipeline configuration change.
    cfg.siblings = Some(map);
    let pass = |st: &str| (st.to_string(), Verdict::Pass);

    let r1 = worker::run_once(&cfg, REPO, &head_s).expect("worker r1");
    assert_eq!(gate_status(&r1, "S"), pass("run"), "{:?}", r1.notes);
    assert!(
        r1.notes.iter().any(|n| n.contains(&sib_sha)),
        "{:?}",
        r1.notes
    );

    // The same head again, sibling unchanged: reused.
    let r2 = worker::run_once(&cfg, REPO, &head_s).expect("worker r2");
    assert_eq!(gate_status(&r2, "S"), pass("cached"), "{:?}", r2.gates);

    // A new commit that leaves S's declared inputs (tools/) alone still runs
    // S: results never cross commits.
    let head_s2 = commit_on(&w, "topic-s", &[("h/h.txt", "s2\n")], "change h");
    let r3 = worker::run_once(&cfg, REPO, &head_s2).expect("worker r3");
    assert_eq!(gate_status(&r3, "S"), pass("run"), "{:?}", r3.gates);

    // The sibling moves to a new commit: the same head runs S again.
    let so = w.origin.parent().unwrap().join("sib-origin");
    write(&so, "lib/extra.txt", "world\n", false);
    git(&so, &["add", "-A"]);
    git(&so, &["commit", "-q", "-m", "sib v2"]);
    let sib_sha2 = git(&so, &["rev-parse", "HEAD"]);
    assert_ne!(sib_sha, sib_sha2);
    git(
        &w.clone.parent().unwrap().join(SIB),
        &["fetch", "-q", "--all"],
    );
    let r4 = worker::run_once(&cfg, REPO, &head_s).expect("worker r4");
    assert_eq!(gate_status(&r4, "S"), pass("run"), "{:?}", r4.gates);
    assert!(
        r4.notes.iter().any(|n| n.contains(&sib_sha2)),
        "{:?}",
        r4.notes
    );

    // Same head, sibling still at its new commit: reused again.
    let r5 = worker::run_once(&cfg, REPO, &head_s).expect("worker r5");
    assert_eq!(gate_status(&r5, "S"), pass("cached"), "{:?}", r5.gates);
}

/// Cache safety across commits: gate U declares only `u/` as its input but
/// its program also reads the root `README.md`, as SC-TEST's
/// `tier2_boundary` tests read the root `README.md` and `COVENANT.md`. A
/// later commit that changes only `README.md` makes U fail when it really
/// runs, so the worker must run U again, never hand it the earlier PASS.
#[test]
fn worker_never_reuses_a_pass_when_an_undeclared_file_changed_across_commits() {
    if nested_in_sandbox(
        "worker_never_reuses_a_pass_when_an_undeclared_file_changed_across_commits",
    ) {
        return;
    }
    let w = world();
    let mut cfg = w.config();
    cfg.operator = true; // U is a new gate: a pipeline configuration change.

    git(&w.origin, &["checkout", "-q", "-b", "topic-u", "main"]);
    write(
        &w.origin,
        "tools/u.sh",
        "#!/bin/sh\ngrep -q 'AGPL-3.0' README.md || { echo 'README.md lost its licence' >&2; exit 1; }\necho PASS\n",
        true,
    );
    git(&w.origin, &["checkout", "-q", "main"]);
    let head_u1 = commit_on(
        &w,
        "topic-u",
        &[
            ("u/u.txt", "1\n"),
            ("README.md", "Licence: AGPL-3.0\n"),
            ("tests/gates/U.gate", &gate("U", "u/", "tools/u.sh", "")),
        ],
        "gate U reads README.md",
    );
    let head_u2 = commit_on(
        &w,
        "topic-u",
        &[("README.md", "Licence: none\n")],
        "README.md loses its licence",
    );

    let r1 = worker::run_once(&cfg, REPO, &head_u1).expect("worker u1");
    assert_eq!(
        gate_status(&r1, "U"),
        ("run".to_string(), Verdict::Pass),
        "{:?}",
        r1.notes
    );
    let r2 = worker::run_once(&cfg, REPO, &head_u2).expect("worker u2");
    assert_eq!(
        gate_status(&r2, "U"),
        ("run".to_string(), Verdict::Fail),
        "a PASS from {head_u1} must not be handed to {head_u2}: {:?}",
        r2.gates
    );
}

/// An operator run of a pipeline configuration change (a head that adds a
/// gate under `tests/gates/`) is checked with the operator's permission too:
/// the independent check recomputes the plan with `operator` set, so a
/// passing operator run is READY and posts success. Without the operator the
/// same head stays BLOCKED_OPERATOR and never posts success.
#[test]
fn worker_operator_run_of_a_config_change_is_ready_and_posts_success() {
    if nested_in_sandbox("worker_operator_run_of_a_config_change_is_ready_and_posts_success") {
        return;
    }
    let w = world();
    let (fake, log, _down) = fake_gh(&w);
    let mut cfg = w.config();
    cfg.github = true;
    cfg.gh = fake.display().to_string();
    git(&w.origin, &["checkout", "-q", "-b", "topic-op", "main"]);
    write(&w.origin, "op/op.sh", "#!/bin/sh\necho PASS\n", true);
    git(&w.origin, &["checkout", "-q", "main"]);
    let head = commit_on(
        &w,
        "topic-op",
        &[
            ("op/op.txt", "1\n"),
            ("tests/gates/OP.gate", &gate("OP", "op/", "op/op.sh", "")),
        ],
        "new gate OP",
    );

    // Not the operator: blocked, never success.
    let s = worker::run_once(&cfg, REPO, &head).expect("worker (no operator)");
    assert!(!s.ready, "{:?}", s.notes);
    let l = fs::read_to_string(&log).unwrap();
    assert!(!l.contains("state=success"), "{l}");

    // The operator: the gate runs and passes, the check agrees, success.
    cfg.operator = true;
    let s = worker::run_once(&cfg, REPO, &head).expect("worker (operator)");
    assert_eq!(gate_verdict(&s, "OP"), Verdict::Pass, "{:?}", s.notes);
    assert_eq!(s.bundle_verdict, "PASS");
    assert!(s.ready, "{:?}", s.notes);
    let l = fs::read_to_string(&log).unwrap();
    assert!(l.contains("state=success"), "{l}");
    let j = fs::read_to_string(w.state.journal()).unwrap();
    assert!(
        j.lines().last().unwrap().contains("\"verify\":\"READY\""),
        "{j}"
    );

    // Permission must never be inherited from an earlier authorized run
    // of this exact head. A later non-operator run must publish a refusal.
    let prior_statuses = l;
    cfg.operator = false;
    let s = worker::run_once(&cfg, REPO, &head).expect("worker (operator revoked)");
    assert!(!s.ready, "operator authorization leaked into a later run");
    let statuses = fs::read_to_string(&log).unwrap();
    let new_statuses = statuses
        .strip_prefix(&prior_statuses)
        .expect("the worker must append to the existing status log");
    assert!(new_statuses.contains("state=error"), "{new_statuses}");
    assert!(!new_statuses.contains("state=success"), "{new_statuses}");
    let journal = fs::read_to_string(w.state.journal()).unwrap();
    assert!(
        journal
            .lines()
            .last()
            .unwrap()
            .contains("\"verify\":\"BLOCKED\""),
        "{journal}"
    );
}
