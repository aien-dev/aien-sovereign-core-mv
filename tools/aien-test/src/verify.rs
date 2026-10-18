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

fn sum_u64(index: &Index, digests: &[String], key: &str) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    for d in digests {
        if let Some(s) = index.get(d) {
            if let Some(v) = s
                .value
                .get("volatile")
                .and_then(|o| o.get(key))
                .and_then(Value::as_u64)
            {
                total += v;
                any = true;
            }
        }
    }
    any.then_some(total)
}

/// The file at the repository root that pins the crumb tool: one
/// aien-architecture commit (sc#602).
pub const CRUMB_LOCK: &str = "crumb.lock";

/// The pinned commit in a `crumb.lock` text: exactly one 40-hex sha, surrounding
/// whitespace ignored. None for anything else (a malformed lock pins nothing and
/// the report says so).
pub fn parse_crumb_lock(text: &str) -> Option<String> {
    let pin = text.trim();
    (pin.len() == 40 && pin.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| pin.to_ascii_lowercase())
}

/// Which crumb tool the readiness check ran, for the report's `Crumb tool:` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrumbTool {
    /// Built from the aien-architecture commit `crumb.lock` names at COMMIT.
    Pinned { pin: String, path: PathBuf },
    /// The command the caller named (PATH `crumb`): COMMIT has no `crumb.lock`.
    Unpinned(String),
    /// COMMIT pins a tool but it could not be built; the reason is the string.
    Unavailable(String),
}

/// The crumb tool for COMMIT (sc#602). With a `crumb.lock` at COMMIT the tool is
/// built from the trusted sibling clone `<clones>/aien-architecture` (the parent
/// of `root`, where the worker keeps every trusted clone) into
/// `<cache>/<pin>/tools/crumb/crumb` and reused. The build is `git archive` of
/// `tools/crumb` at the pin plus `make crumb`, the same as the CRUMB-VERIFY gate
/// in tests/gates/run.sh, so the readiness line and the gate use one binary.
///
/// aien-architecture owns the tool and has no `crumb.lock`: its commits are
/// checked with the tool at COMMIT itself, as its hosted workflow does, so a
/// change to the crumb format is judged by the tool that writes it.
pub fn crumb_tool_for(root: &Path, commit: &str, fallback: &str, cache: &Path) -> CrumbTool {
    let pin = match plan::show(root, commit, CRUMB_LOCK) {
        Some(bytes) => match parse_crumb_lock(&String::from_utf8_lossy(&bytes)) {
            Some(pin) => pin,
            None => {
                return CrumbTool::Unavailable(format!(
                    "{CRUMB_LOCK} at {commit} is not one 40-hex commit"
                ));
            }
        },
        None if owns_crumb_tool(root, commit) => commit.to_ascii_lowercase(),
        None => return CrumbTool::Unpinned(fallback.to_string()),
    };
    let built = cache.join(&pin).join("tools").join("crumb").join("crumb");
    if built.is_file() {
        return CrumbTool::Pinned { pin, path: built };
    }
    let Some(clone) = root.parent().map(|p| p.join("aien-architecture")) else {
        return CrumbTool::Unavailable("repository root has no parent directory".into());
    };
    if !clone.join(".git").exists() {
        return CrumbTool::Unavailable(format!(
            "no trusted clone at {} to build crumb {pin} from",
            clone.display()
        ));
    }
    let dir = cache.join(&pin);
    let _ = fs::remove_dir_all(&dir);
    if let Err(e) = fs::create_dir_all(&dir) {
        return CrumbTool::Unavailable(format!("cannot create {}: {e}", dir.display()));
    }
    let tar = dir.join("crumb-tool.tar");
    let mut archive = Command::new("git");
    archive
        .args(["archive", "--format=tar", "-o"])
        .arg(&tar)
        .arg(&pin)
        .arg("tools/crumb")
        .current_dir(&clone)
        .env("GIT_OPTIONAL_LOCKS", "0");
    if let Err(e) = run_ok(&mut archive) {
        return CrumbTool::Unavailable(format!("git archive {pin} tools/crumb: {e}"));
    }
    let mut untar = Command::new("tar");
    untar.args(["-x", "-f"]).arg(&tar).arg("-C").arg(&dir);
    if let Err(e) = run_ok(&mut untar) {
        return CrumbTool::Unavailable(format!("tar: {e}"));
    }
    let _ = fs::remove_file(&tar);
    let mut make = Command::new("make");
    make.args(["-s", "-C"])
        .arg(dir.join("tools").join("crumb"))
        .arg("crumb");
    if let Err(e) = run_ok(&mut make) {
        return CrumbTool::Unavailable(format!("make crumb at {pin}: {e}"));
    }
    if built.is_file() {
        CrumbTool::Pinned { pin, path: built }
    } else {
        CrumbTool::Unavailable(format!("make produced no {}", built.display()))
    }
}

/// Run a build step; Err carries the last stderr line (or the spawn error).
/// True for the trusted aien-architecture clone (the worker's
/// `<clones>/aien-architecture`) at a commit that carries the tool source.
fn owns_crumb_tool(root: &Path, commit: &str) -> bool {
    root.file_name().is_some_and(|n| n == "aien-architecture")
        && plan::show(root, commit, "tools/crumb/crumb.c").is_some()
}

fn run_ok(cmd: &mut Command) -> Result<(), String> {
    match process::output_gated(cmd) {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr)
            .lines()
            .last()
            .unwrap_or("failed")
            .trim()
            .to_string()),
        Err(e) => Err(e.to_string()),
    }
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
    let result = if !tmp.join(".crumb").exists() {
        (
            "STALE".to_string(),
            "no .crumb at the repository root".to_string(),
        )
    } else {
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
    // sc#602: a crumb.lock at COMMIT pins the tool; the report names which one ran.
    let (crumb, crumb_note, crumb_tool) = if !commit_ok {
        ("STALE".to_string(), "commit missing".to_string(), None)
    } else {
        match ask.crumb.as_deref() {
            None => {
                let (s, n) = crumb_status(root, &commit, None);
                (s, n, None)
            }
            Some(fallback) => {
                let cache = ask.state.root.join("crumb-tool");
                match crumb_tool_for(root, &commit, fallback, &cache) {
                    CrumbTool::Pinned { pin, path } => {
                        let (s, n) = crumb_status(root, &commit, path.to_str());
                        (s, n, Some(format!("pinned {pin} (crumb.lock)")))
                    }
                    CrumbTool::Unpinned(cmd) => {
                        let (s, n) = crumb_status(root, &commit, Some(&cmd));
                        (
                            s,
                            n,
                            Some(format!("{cmd} on PATH (no crumb.lock at COMMIT)")),
                        )
                    }
                    CrumbTool::Unavailable(why) => (
                        "STALE".to_string(),
                        format!("pinned crumb tool unavailable: {why}"),
                        Some("pinned, not built".to_string()),
                    ),
                }
            }
        }
    };
    lines.push(format!("Crumb status: {crumb}"));
    if let Some(tool) = crumb_tool {
        lines.push(format!("Crumb tool: {tool}"));
    }
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
    let queue = sum_u64(&found.index, &run_digests, "pool_wait_ms");
    let test = sum_u64(&found.index, &run_digests, "duration_ms");
    let total = bundle.as_ref().and_then(|b| {
        let s = utc_secs(b["started_utc"].as_str()?)?;
        let f = utc_secs(b["finished_utc"].as_str()?)?;
        Some(f.saturating_sub(s) * 1000)
    });
    lines.push(format!("Queue time: {}", ms(queue)));
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
    fn crumb_lock_parses_one_sha_only() {
        let sha = "107a2796c7b9ad6dff6f94a01d6567eb1ec9c995";
        assert_eq!(
            parse_crumb_lock(&format!("  {}\n", sha.to_uppercase())),
            Some(sha.into())
        );
        assert_eq!(parse_crumb_lock("107a279\n"), None);
        assert_eq!(parse_crumb_lock(&format!("{sha}\n{sha}\n")), None);
        assert_eq!(
            parse_crumb_lock("not a sha at all, forty characters long!"),
            None
        );
        assert_eq!(parse_crumb_lock(""), None);
    }

    #[test]
    fn commit_without_crumb_lock_uses_the_named_command() {
        let repo = crate::testutil::init_repo("crumb-unpinned", &[("README", "x\n", false)]);
        let head = plan::resolve(repo.path(), "HEAD").unwrap();
        let cache = crate::testutil::TempDir::new("crumb-cache");
        assert_eq!(
            crumb_tool_for(repo.path(), &head, "crumb", cache.path()),
            CrumbTool::Unpinned("crumb".into())
        );
    }

    #[test]
    fn malformed_crumb_lock_pins_nothing_and_says_so() {
        let repo = crate::testutil::init_repo("crumb-badlock", &[(CRUMB_LOCK, "107a279\n", false)]);
        let head = plan::resolve(repo.path(), "HEAD").unwrap();
        let cache = crate::testutil::TempDir::new("crumb-cache");
        match crumb_tool_for(repo.path(), &head, "crumb", cache.path()) {
            CrumbTool::Unavailable(why) => assert!(why.contains("not one 40-hex commit"), "{why}"),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    /// A pinned tool is built from `<parent of root>/aien-architecture` at the
    /// pin and reused from the cache: a fake sibling whose Makefile writes a
    /// one-line `crumb` script stands in for the real tool.
    #[test]
    fn pinned_tool_is_built_from_the_sibling_clone_and_cached() {
        let base = crate::testutil::TempDir::new("crumb-pinned");
        let sib = base.path().join("aien-architecture");
        let tool_dir = sib.join("tools").join("crumb");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(
            tool_dir.join("Makefile"),
            "crumb:\n\tprintf '#!/bin/sh\\necho fake crumb\\n' > crumb && chmod +x crumb\n",
        )
        .unwrap();
        crate::testutil::git(&sib, &["init", "-q"]);
        crate::testutil::git(&sib, &["add", "."]);
        crate::testutil::git(&sib, &["commit", "-q", "-m", "tool"]);
        let pin = plan::resolve(&sib, "HEAD").unwrap();
        let repo = base.path().join("repo");
        fs::create_dir_all(&repo).unwrap();
        fs::write(repo.join(CRUMB_LOCK), format!("{pin}\n")).unwrap();
        crate::testutil::git(&repo, &["init", "-q"]);
        crate::testutil::git(&repo, &["add", "."]);
        crate::testutil::git(&repo, &["commit", "-q", "-m", "pin"]);
        let head = plan::resolve(&repo, "HEAD").unwrap();
        let cache = base.path().join("cache");
        let built = cache.join(&pin).join("tools").join("crumb").join("crumb");
        assert_eq!(
            crumb_tool_for(&repo, &head, "crumb", &cache),
            CrumbTool::Pinned {
                pin: pin.clone(),
                path: built.clone()
            }
        );
        assert!(built.is_file());
        // Second call: the sibling is gone, the cached build still answers.
        fs::remove_dir_all(&sib).unwrap();
        assert_eq!(
            crumb_tool_for(&repo, &head, "crumb", &cache),
            CrumbTool::Pinned { pin, path: built }
        );
    }

    /// aien-architecture (no `crumb.lock`, owns `tools/crumb`) is checked with
    /// the tool at COMMIT; any other repo carrying a `tools/crumb` is not.
    #[test]
    fn architecture_checks_itself_with_the_tool_at_commit() {
        let base = crate::testutil::TempDir::new("crumb-self");
        let makefile =
            "crumb:\n\tprintf '#!/bin/sh\\necho fake crumb\\n' > crumb && chmod +x crumb\n";
        for name in ["aien-architecture", "other"] {
            let tool_dir = base.path().join(name).join("tools").join("crumb");
            fs::create_dir_all(&tool_dir).unwrap();
            fs::write(tool_dir.join("Makefile"), makefile).unwrap();
            fs::write(tool_dir.join("crumb.c"), "int main(void){return 0;}\n").unwrap();
            let repo = base.path().join(name);
            crate::testutil::git(&repo, &["init", "-q"]);
            crate::testutil::git(&repo, &["add", "."]);
            crate::testutil::git(&repo, &["commit", "-q", "-m", "tool"]);
        }
        let arch = base.path().join("aien-architecture");
        let head = plan::resolve(&arch, "HEAD").unwrap();
        let cache = base.path().join("cache");
        let built = cache.join(&head).join("tools").join("crumb").join("crumb");
        assert_eq!(
            crumb_tool_for(&arch, &head, "crumb", &cache),
            CrumbTool::Pinned {
                pin: head.clone(),
                path: built.clone()
            }
        );
        assert!(built.is_file());
        let other = base.path().join("other");
        let head = plan::resolve(&other, "HEAD").unwrap();
        assert_eq!(
            crumb_tool_for(&other, &head, "crumb", &cache),
            CrumbTool::Unpinned("crumb".into())
        );
    }

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
}
