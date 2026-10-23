//! `aien-test verify COMMIT`: the independent check of a local verification
//! (ADR 0033 Decision 10, contracts C4 to C6, amendments A1 and A7).
//!
//! It reads the signed bundle the worker wrote, checks it against the public
//! keys the repository committed at BASE (`.aien/verify.pub`, one hex key per
//! line), recomputes the plan for BASE..COMMIT and compares the graph digest,
//! checks that every `run` entry has its receipt on disk, judges the verdict
//! with the planner's algebra, asks `crumb verify` about the tree at COMMIT,
//! and prints the fixed report block. Exit 0 only on MERGE ELIGIBILITY READY.

use crate::bundle::{self, Expect};
use crate::evidence::{self, utc_secs, Index, Store};
use crate::manifest::CacheScope;
use crate::plan::{self, Known, Plan, Request, Section};
use crate::process;
use crate::verdict::Verdict;
use crate::worker::StateDir;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const KEYS_FILE: &str = ".aien/verify.pub";

/// What the verifier was asked to do.
#[derive(Debug, Clone)]
pub struct Ask {
    pub state: StateDir,
    pub commit: String,
    pub base: String,
    pub repository: Option<String>,
    pub release: bool,
    pub operator: bool,
    pub hardware_identity: String,
    /// The crumb command; None skips the crumb check (status STALE).
    pub crumb: Option<String>,
    /// Keys to trust instead of `.aien/verify.pub` at BASE (tests).
    pub keys_override: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub lines: Vec<String>,
    pub ready: bool,
}

/// Public keys from `.aien/verify.pub` at `commit`, one hex key per line,
/// `#` comments and blank lines ignored.
pub fn keys_at(root: &Path, commit: &str) -> Vec<String> {
    let Some(bytes) = plan::show(root, commit, KEYS_FILE) else {
        return Vec::new();
    };
    parse_keys(&String::from_utf8_lossy(&bytes))
}

pub fn parse_keys(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| {
            l.split('#')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .filter(|l| l.len() == 64 && l.bytes().all(|b| b.is_ascii_hexdigit()))
        .collect()
}

/// A receipt the bundle points at, found in the commit's evidence directory.
struct Found {
    index: Index,
}

impl Found {
    fn load(dir: &Path) -> Found {
        Found {
            index: Index::load(&Store::new(dir)),
        }
    }
    fn get(&self, digest: &str) -> Option<&evidence::Stored> {
        self.index.get(digest)
    }
}

/// The sum of `volatile<pointer>` (a JSON pointer such as `/pool_wait_ms`)
/// over the receipts; None when no receipt has it.
fn sum_u64(index: &Index, digests: &[String], pointer: &str) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    for d in digests {
        if let Some(s) = index.get(d) {
            if let Some(v) = s
                .value
                .get("volatile")
                .and_then(|o| o.pointer(pointer))
                .and_then(Value::as_u64)
            {
                total += v;
                any = true;
            }
        }
    }
    any.then_some(total)
}

/// `crumb verify` over the tree at COMMIT, in a temporary detached worktree.
/// CURRENT only when crumb says OK; STALE otherwise, also when the command is
/// unavailable (the tree is then not proven current).
pub fn crumb_status(root: &Path, commit: &str, crumb: Option<&str>) -> (String, String) {
    let Some(crumb) = crumb else {
        return ("STALE".into(), "crumb check skipped".into());
    };
    let tmp = std::env::temp_dir().join(format!(
        "aien-verify-{}-{}",
        std::process::id(),
        &commit[..12]
    ));
    let _ = fs::remove_dir_all(&tmp);
    let mut add = Command::new("git");
    add.args(["worktree", "add", "--detach", "--quiet"])
        .arg(&tmp)
        .arg(commit)
        .current_dir(root)
        .env("GIT_OPTIONAL_LOCKS", "0");
    let out = match process::output_gated(&mut add) {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return (
                "STALE".into(),
                format!(
                    "cannot check out {commit}: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                ),
            );
        }
        Err(e) => return ("STALE".into(), format!("cannot run git: {e}")),
    };
    drop(out);
    // The crumb tool decides, exactly as the CRUMB-VERIFY gate does. A repository with no crumbs at all
    // (benchmarks, interplane) is "OK: 0 crumbs current" to the tool; calling it STALE here published every
    // PASS head of those repos as BLOCKED/error (2026-10-16..19).
    let result = {
        let mut cmd = Command::new(crumb);
        cmd.args(["verify", "."]).current_dir(&tmp);
        match process::output_gated(&mut cmd) {
            Ok(o) if o.status.success() => (
                "CURRENT".to_string(),
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            ),
            Ok(o) => (
                "STALE".to_string(),
                format!(
                    "{} {}",
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .last()
                        .unwrap_or("")
                        .trim(),
                    String::from_utf8_lossy(&o.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                        .trim()
                )
                .trim()
                .to_string(),
            ),
            Err(e) => ("STALE".to_string(), format!("cannot run {crumb}: {e}")),
        }
    };
    let mut rm = Command::new("git");
    rm.args(["worktree", "remove", "--force"])
        .arg(&tmp)
        .current_dir(root);
    let _ = process::output_gated(&mut rm);
    let _ = fs::remove_dir_all(&tmp);
    let mut prune = Command::new("git");
    prune.args(["worktree", "prune"]).current_dir(root);
    let _ = process::output_gated(&mut prune);
    result
}

fn verdict_line(v: &str) -> &str {
    match v {
        "PASS" | "FAIL" | "NOT_RUN" => v,
        "BLOCKED_HARDWARE" | "BLOCKED_OPERATOR" => "BLOCKED",
        _ => "NOT_RUN",
    }
}

/// Verify one commit's bundle against the repository at `root`.
pub fn run(root: &Path, ask: &Ask) -> Report {
    let repository = plan::repository_name(root, ask.repository.as_deref());
    let repo = StateDir::repo_key(&repository);
    let commit = ask.commit.to_ascii_lowercase();
    let mut lines: Vec<String> = vec!["AIEN LOCAL VERIFICATION".to_string()];
    let mut problems: Vec<String> = Vec::new();

    // The commit and base must exist here.
    let commit_ok = plan::resolve(root, &commit)
        .map(|c| c == commit)
        .unwrap_or(false);
    // The worker's BASE rule: a commit already on the base line is checked
    // against its first parent, any other against the base ref.
    let base = if commit_ok {
        plan::base_for(root, &ask.base, &commit).map(|(b, _)| b)
    } else {
        plan::resolve(root, &ask.base)
    };
    lines.push(format!("Repository: {repository}"));
    lines.push(format!("Commit: {commit}"));
    if !commit_ok {
        problems.push(format!("commit {commit} is not in this repository"));
    }
    let base = match base {
        Ok(b) => Some(b),
        Err(e) => {
            problems.push(e);
            None
        }
    };

    // The plan, fresh.
    let plan: Option<Plan> = match (&base, commit_ok) {
        (Some(b), true) => {
            let mut req = Request::new(b, &commit);
            req.operator = ask.operator;
            req.hardware_identity = ask.hardware_identity.clone();
            match plan::compute(root, &req) {
                Ok(p) => Some(p),
                Err(e) => {
                    problems.push(format!("cannot plan: {e}"));
                    None
                }
            }
        }
        _ => None,
    };

    // The bundle.
    let bundle_path = ask.state.bundle_path(&repo, &commit);
    let bundle: Option<Value> = match fs::read(&bundle_path) {
        Ok(b) => match serde_json::from_slice::<Value>(&b) {
            Ok(v) => Some(v),
            Err(e) => {
                problems.push(format!("bundle is not JSON: {e}"));
                None
            }
        },
        Err(_) => {
            problems.push(format!("no bundle at {}", bundle_path.display()));
            None
        }
    };

    // Keys at BASE (A1, A7): pick the one the bundle names; otherwise UNKNOWN_KEY.
    let keys = match (&ask.keys_override, &base) {
        (Some(k), _) => k.clone(),
        (None, Some(b)) => keys_at(root, b),
        (None, None) => Vec::new(),
    };
    let required: Vec<String> = plan
        .as_ref()
        .map(|p| {
            p.entries
                .iter()
                .filter(|e| e.required && e.section != Section::Blocked)
                .map(|e| e.gate.clone())
                .collect()
        })
        .unwrap_or_default();
    let mut evidence_status = "INVALID".to_string();
    let mut evidence_note = String::new();
    let mut verified: Option<bundle::Verified> = None;
    if let Some(b) = &bundle {
        if keys.is_empty() {
            evidence_note = format!("no trusted keys ({KEYS_FILE} at BASE is missing or empty)");
        } else {
            let named = b["signature"]["pubkey"]
                .as_str()
                .unwrap_or("")
                .to_ascii_lowercase();
            let key = keys
                .iter()
                .find(|k| **k == named)
                .cloned()
                .unwrap_or_else(|| keys[0].clone());
            let expect = Expect {
                repository: repository.clone(),
                commit: commit.clone(),
                pubkey_hex: key,
                required: required.clone(),
            };
            match bundle::verify(b, &expect) {
                Ok(v) => {
                    evidence_status = "VERIFIED".to_string();
                    verified = Some(v);
                }
                Err(r) => evidence_note = r.to_string(),
            }
        }
    }

    // Bundle against plan: graph digest, receipts, release rule.
    let mut known: BTreeMap<String, Known> = BTreeMap::new();
    let mut executed: Vec<String> = Vec::new();
    let mut cached: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let mut blocked: Vec<String> = Vec::new();
    let mut run_digests: Vec<String> = Vec::new();
    let evidence_dir = ask.state.evidence(&repo, &commit);
    let found = Found::load(&evidence_dir);
    if let (Some(b), Some(p)) = (&bundle, &plan) {
        if b["graph_digest"].as_str() != Some(p.graph_digest.as_str()) {
            evidence_status = "INVALID".to_string();
            let note = format!(
                "GRAPH_MISMATCH: bundle graph {} but the plan for {}..{} gives {}",
                b["graph_digest"].as_str().unwrap_or("?"),
                &p.base[..12.min(p.base.len())],
                &commit[..12],
                p.graph_digest
            );
            if evidence_note.is_empty() {
                evidence_note = note;
            } else {
                problems.push(note);
            }
        }
        let by_gate: BTreeMap<String, &plan::Entry> =
            p.entries.iter().map(|e| (e.gate.clone(), e)).collect();
        for c in b["checks"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            let gate = c["gate"].as_str().unwrap_or("").to_string();
            let status = c["status"].as_str().unwrap_or("");
            let v = Verdict::parse(c["verdict"].as_str().unwrap_or("")).unwrap_or(Verdict::NotRun);
            let entry = by_gate.get(&gate);
            let mut reason = String::new();
            let mut verdict = v;
            // The entry must name the check the plan computes for this gate
            // at COMMIT; an entry for another check id never counts.
            if let Some(e) = entry {
                if c["check_id"].as_str() != Some(e.check_id.as_str()) {
                    verdict = Verdict::NotRun;
                    reason = "CHECK_ID_MISMATCH".into();
                    problems.push(format!(
                        "{gate}: CHECK_ID_MISMATCH bundle {} but the plan gives {}",
                        c["check_id"].as_str().unwrap_or("?"),
                        e.check_id
                    ));
                }
            }
            // The manifest digest of this gate at COMMIT, for the receipt check.
            let manifest_digest = p
                .graph
                .index_of(&gate)
                .map(|i| p.graph.node(i).manifest.digest.clone());
            match status {
                "run" => {
                    let d = c["receipt"].as_str().unwrap_or("").to_string();
                    let receipt_manifest = |s: &evidence::Stored| {
                        s.value["core"]["manifest_digest"]
                            .as_str()
                            .map(str::to_string)
                    };
                    match found.get(&d) {
                        Some(s)
                            if s.gate() == gate
                                && s.commit() == commit
                                && s.verdict() == Some(v)
                                && manifest_digest.is_some()
                                && receipt_manifest(s) == manifest_digest =>
                        {
                            if reason.is_empty() {
                                executed.push(gate.clone());
                                run_digests.push(d.clone());
                            }
                        }
                        Some(s) if s.gate() == gate && receipt_manifest(s) != manifest_digest => {
                            verdict = Verdict::NotRun;
                            reason = "MANIFEST_MISMATCH".into();
                            problems.push(format!(
                                "{gate}: MANIFEST_MISMATCH receipt {d} was made from another manifest than the one at {}",
                                &commit[..12]
                            ));
                        }
                        Some(_) => {
                            verdict = Verdict::NotRun;
                            reason = "RECEIPT_MISMATCH".into();
                            problems.push(format!(
                                "{gate}: receipt {d} does not match the bundle entry"
                            ));
                        }
                        None => {
                            verdict = Verdict::NotRun;
                            reason = "RECEIPT_MISSING".into();
                            problems.push(format!(
                                "{gate}: receipt {d} is not in {}",
                                evidence_dir.display()
                            ));
                        }
                    }
                    if v == Verdict::Fail {
                        failed.push(gate.clone());
                    }
                }
                "cached" => {
                    let scope = entry.map(|e| e.cache_scope);
                    if ask.release && scope == Some(CacheScope::Hardware) {
                        verdict = Verdict::NotRun;
                        reason = "CACHE_SCOPE_HARDWARE_RELEASE".into();
                        problems.push(format!(
                            "{gate}: a cached hardware-scoped result cannot satisfy --release"
                        ));
                    } else {
                        cached.push(gate.clone());
                    }
                    if v == Verdict::Fail {
                        failed.push(gate.clone());
                    }
                }
                "blocked" => {
                    blocked.push(gate.clone());
                    if !matches!(v, Verdict::BlockedHardware | Verdict::BlockedOperator) {
                        verdict = Verdict::BlockedOperator;
                    }
                    reason = "BLOCKED".into();
                }
                _ => {
                    if v == Verdict::Fail {
                        failed.push(gate.clone());
                    }
                    reason = "NOT_RUN".into();
                }
            }
            if entry.is_none() {
                problems.push(format!("{gate}: in the bundle but not in the plan"));
            }
            known.insert(
                gate,
                Known {
                    verdict,
                    reason,
                    receipt: c["receipt"]
                        .as_str()
                        .or_else(|| c["resolution"]["resolved_from"].as_str())
                        .map(str::to_string),
                },
            );
        }
    }
    let judged = plan.as_ref().map(|p| plan::judge(p, &known));

    // The report body.
    let changed = plan.as_ref().map(|p| p.changed.len()).unwrap_or(0);
    let affected: Vec<String> = plan
        .as_ref()
        .map(|p| p.entries.iter().map(|e| e.gate.clone()).collect())
        .unwrap_or_default();
    let unmapped = plan
        .as_ref()
        .map(|p| p.unmapped.clone())
        .unwrap_or_default();
    lines.push(format!("Changed files: {changed}"));
    lines.push(format!("Affected gates: {}", list(&affected)));
    lines.push(format!("Executed gates: {}", list(&executed)));
    lines.push(format!("Cached gates: {}", list(&cached)));
    lines.push(format!("Failed gates: {}", list(&failed)));
    lines.push(format!("Blocked gates: {}", list(&blocked)));
    let (crumb, crumb_note) = if commit_ok {
        crumb_status(root, &commit, ask.crumb.as_deref())
    } else {
        ("STALE".into(), "commit missing".into())
    };
    lines.push(format!("Crumb status: {crumb}"));
    lines.push(format!("Evidence status: {evidence_status}"));
    let hardware = match &plan {
        Some(p)
            if p.entries
                .iter()
                .any(|e| e.section == Section::NeedsGb10 || e.section == Section::NeedsSpark) =>
        {
            let hw: Vec<&plan::Entry> = p
                .entries
                .iter()
                .filter(|e| e.section == Section::NeedsGb10 || e.section == Section::NeedsSpark)
                .collect();
            if hw.iter().all(|e| {
                known
                    .get(&e.gate)
                    .map(|k| k.verdict == Verdict::Pass)
                    .unwrap_or(false)
            }) {
                "QUALIFIED"
            } else {
                "BLOCKED"
            }
        }
        Some(_) => "NOT_REQUIRED",
        None => "BLOCKED",
    };
    lines.push(format!("Hardware status: {hardware}"));
    let final_verdict = match (&judged, evidence_status.as_str()) {
        (Some(j), "VERIFIED") => verdict_line(j.verdict.as_str()).to_string(),
        (Some(j), _) if j.verdict == Verdict::Fail => "FAIL".to_string(),
        (Some(j), _)
            if matches!(
                j.verdict,
                Verdict::BlockedHardware | Verdict::BlockedOperator
            ) =>
        {
            "BLOCKED".to_string()
        }
        _ => "NOT_RUN".to_string(),
    };
    lines.push(format!("Final verdict: {final_verdict}"));
    // A gate's batch-slot and target-dir waits fall inside its `duration_ms`;
    // they are queue time, not test time.
    let held = sum_u64(&found.index, &run_digests, "/batch_wait_ms").unwrap_or(0)
        + sum_u64(&found.index, &run_digests, "/target_wait_ms").unwrap_or(0);
    let queue = sum_u64(&found.index, &run_digests, "/pool_wait_ms").map(|q| q + held);
    let test = sum_u64(&found.index, &run_digests, "/duration_ms").map(|t| t.saturating_sub(held));
    let gpu_wait = sum_u64(&found.index, &run_digests, "/gpu_wait/waited_ms");
    let total = bundle.as_ref().and_then(|b| {
        let s = utc_secs(b["started_utc"].as_str()?)?;
        let f = utc_secs(b["finished_utc"].as_str()?)?;
        Some(f.saturating_sub(s) * 1000)
    });
    lines.push(format!("Queue time: {}", ms(queue)));
    lines.push(format!("GPU lock wait: {}", ms(gpu_wait)));
    lines.push("Build time: n/a".to_string());
    lines.push(format!("Test time: {}", ms(test)));
    lines.push(format!("Total time: {}", ms(total)));
    let ready = evidence_status == "VERIFIED"
        && final_verdict == "PASS"
        && crumb == "CURRENT"
        && unmapped.is_empty()
        && blocked.is_empty()
        && problems.is_empty()
        && judged.as_ref().is_some_and(|j| j.verdict == Verdict::Pass)
        && verified.as_ref().is_some_and(|v| v.verdict == "PASS");
    lines.push(format!(
        "MERGE ELIGIBILITY: {}",
        if ready { "READY" } else { "BLOCKED" }
    ));
    if !evidence_note.is_empty() {
        lines.push(format!("Evidence note: {evidence_note}"));
    }
    if !crumb_note.is_empty() {
        lines.push(format!("Crumb note: {crumb_note}"));
    }
    if let Some(j) = &judged {
        if !j.reason.is_empty() {
            lines.push(format!("Verdict reason: {}", j.reason));
        }
    }
    if !unmapped.is_empty() {
        lines.push(format!("Unmapped: {}", unmapped.join(" ")));
    }
    for p in &problems {
        lines.push(format!("Problem: {p}"));
    }
    lines.push(format!("Bundle: {}", bundle_path.display()));
    Report { lines, ready }
}

fn list(v: &[String]) -> String {
    if v.is_empty() {
        "none".to_string()
    } else {
        format!("{} ({})", v.len(), v.join(" "))
    }
}

fn ms(v: Option<u64>) -> String {
    match v {
        Some(ms) => format!("{ms} ms"),
        None => "n/a".to_string(),
    }
}

/// Where the trusted clone of `repository` is expected (`~/workspace/<repo>`).
pub fn default_clone(repository: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join("workspace")
            .join(StateDir::repo_key(repository)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse() {
        let a = "a".repeat(64);
        let text = format!(
            "# keys\n{a}\n\n  {}   # second\nnot-a-key\n",
            "B".repeat(64)
        );
        let k = parse_keys(&text);
        assert_eq!(k, vec![a, "b".repeat(64)]);
    }

    #[test]
    fn verdict_lines() {
        assert_eq!(verdict_line("PASS"), "PASS");
        assert_eq!(verdict_line("BLOCKED_HARDWARE"), "BLOCKED");
        assert_eq!(verdict_line("MISSING_IMPLEMENTATION"), "NOT_RUN");
        assert_eq!(list(&[]), "none");
        assert_eq!(list(&["a".into(), "b".into()]), "2 (a b)");
        assert_eq!(ms(None), "n/a");
        assert_eq!(ms(Some(12)), "12 ms");
    }

    #[test]
    fn missing_bundle_is_blocked_not_ready() {
        let repo = crate::testutil::init_repo("verify-empty", &[("README", "x\n", false)]);
        let head = plan::resolve(repo.path(), "HEAD").unwrap();
        let state = crate::testutil::TempDir::new("verify-state");
        let ask = Ask {
            state: StateDir::new(state.path()),
            commit: head.clone(),
            base: "HEAD".into(),
            repository: Some("github.com/t/r".into()),
            release: false,
            operator: false,
            hardware_identity: String::new(),
            crumb: None,
            keys_override: None,
        };
        let r = run(repo.path(), &ask);
        assert!(!r.ready);
        assert_eq!(r.lines[0], "AIEN LOCAL VERIFICATION");
        assert!(r.lines.iter().any(|l| l == "Evidence status: INVALID"));
        assert!(r.lines.iter().any(|l| l == "MERGE ELIGIBILITY: BLOCKED"));
        assert!(r
            .lines
            .iter()
            .any(|l| l.starts_with("Problem: no bundle at ")));
    }

    /// A fake crumb on disk that prints `out` and exits `rc`.
    fn fake_crumb(dir: &Path, out: &str, rc: i32) -> String {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("crumb");
        fs::write(&p, format!("#!/bin/sh\necho '{out}'\nexit {rc}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        p.display().to_string()
    }

    #[test]
    fn repo_without_crumbs_asks_the_tool() {
        // benchmarks and interplane have no .crumb at all; the gate's tool says OK, so readiness must too.
        let repo = crate::testutil::init_repo("verify-nocrumb", &[("README", "x\n", false)]);
        let head = plan::resolve(repo.path(), "HEAD").unwrap();
        let bin = crate::testutil::TempDir::new("verify-fake-crumb");
        let ok = fake_crumb(bin.path(), "crumb verify: OK: 0 crumbs current", 0);
        assert_eq!(
            crumb_status(repo.path(), &head, Some(&ok)),
            (
                "CURRENT".to_string(),
                "crumb verify: OK: 0 crumbs current".to_string()
            )
        );
        // The tool still has the last word: if it fails, readiness is STALE.
        let bad = fake_crumb(
            bin.path(),
            "crumb verify: FAIL: 1 of 1 crumbs stale; run: crumb compile",
            1,
        );
        assert_eq!(crumb_status(repo.path(), &head, Some(&bad)).0, "STALE");
        // No tool configured: still STALE (not proven current).
        assert_eq!(crumb_status(repo.path(), &head, None).0, "STALE");
    }
}
