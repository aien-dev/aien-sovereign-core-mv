//! The gate memo (L164, Drake 2026-10-20: "speed this up without taking away
//! our rigor").
//!
//! The digest cache (`cache.rs`) already keys a result on the manifest, the
//! program, every declared input, the machine, the dependency receipts and
//! the sibling pins. The worker gives every head its own evidence directory
//! (`evidence/<repo>/<sha>`), so until now a result was reused only for the
//! very same commit: a docs-only branch reran every selected gate even when
//! nothing a gate reads had changed since a judged head.
//!
//! The memo lets the same cache key find a PASS in the evidence directory of
//! another judged head of the same repository. Rigor kept:
//! - the key is the cache key, unchanged (no new hit is possible that the
//!   cache itself would not grant within one commit);
//! - plus the toolchain id ([`toolchain_id`]): the compilers on the gate's
//!   PATH and the rustup toolchains, which the cache key does not see.
//!   Every fresh receipt now records it in `volatile.toolchain`; a receipt
//!   without it is never a memo source;
//! - only a fresh PASS is a source (never a FAIL, never a reuse), and it must
//!   pass every check of [`cache::usable`] in its own store;
//! - a gate whose manifest declares no `inputs` always runs (its digest would
//!   cover nothing); `cache: never` gates always run (the author's rule);
//! - the policy says which heads may be sources: `off` (the worker's
//!   default), `ancestor` (judged heads that are ancestors of the head being
//!   judged, so main's receipts serve every branch built on it) or `any`
//!   (every judged head of the repository).
//!
//! Why `off` is the default: the memo trusts a gate's `inputs:` list to be
//! complete. A gate that reads a file it does not declare would get an old
//! PASS on a commit that breaks it (tests/local_ci.rs
//! `worker_never_reuses_a_pass_when_an_undeclared_file_changed_across_commits`
//! is that case). The replay of 2026-10-20 measured the gain at 1.4%
//! (ancestor) to 4.2% (any) of gate-minutes (evidence/MERGE-TRAIN), which
//! does not pay for that risk. Turn it on for a repository only once its
//! gates' inputs are shown complete.
//! - the reuse receipt names the source receipt (`cache.reused_from`) and the
//!   source commit (`volatile.memo.from_commit`), and the source receipt and
//!   its logs are copied into this head's store so `why` and `verify` find
//!   them without the other directory.
//!
//! `aien-test memo-replay` replays the evidence store: which fresh runs the
//! memo would have answered, and the gate minutes that saves per head.

use crate::cache;
use crate::evidence::{canonical_digest, sha256_hex, Index, Store, Stored};
use crate::manifest::Manifest;
use crate::sandbox::{self, Sandbox};
use crate::verdict::Verdict;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Which judged heads may stand in for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Off,
    Ancestor,
    Any,
}

impl Policy {
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "off" => Ok(Policy::Off),
            "ancestor" => Ok(Policy::Ancestor),
            "any" => Ok(Policy::Any),
            _ => Err(format!("unknown memo policy {s:?} (off, ancestor, any)")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Off => "off",
            Policy::Ancestor => "ancestor",
            Policy::Any => "any",
        }
    }
}

/// Why a gate may not be answered by the memo, or None when it may.
pub fn ineligible(m: &Manifest) -> Option<&'static str> {
    if !m.cache_allowed {
        Some("the manifest says cache: never")
    } else if m.inputs.is_empty() {
        Some("the manifest declares no inputs, so the gate always runs")
    } else {
        None
    }
}

/// The tools a gate may compile with. Each is resolved on the gate's PATH.
pub const TOOLS: [&str; 10] = [
    "cc", "gcc", "c++", "g++", "clang", "ld", "as", "make", "rustc", "cargo",
];

/// The toolchain id: sha256 over every tool of [`TOOLS`] found on the PATH a
/// gate runs with (path and file bytes, symlinks followed), the rustup
/// settings, and the release manifest of every installed rustup toolchain
/// (it names the exact rustc build). A rustup proxy's bytes do not change
/// when the toolchain behind it does; the release manifests do.
pub fn toolchain_id(sandbox: Sandbox) -> String {
    let (path, rustup_home) = match sandbox {
        Sandbox::User => (
            sandbox::TRUSTED_PATH.to_string(),
            PathBuf::from("/var/lib/aien-run/rustup"),
        ),
        Sandbox::None => (
            std::env::var("PATH").unwrap_or_default(),
            std::env::var_os("RUSTUP_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".rustup")))
                .unwrap_or_default(),
        ),
    };
    toolchain_id_in(&path, &rustup_home)
}

fn file_digest(p: &Path) -> Option<String> {
    fs::read(p).ok().map(|b| sha256_hex(&b))
}

pub fn toolchain_id_in(path: &str, rustup_home: &Path) -> String {
    let mut tools = Vec::new();
    for t in TOOLS {
        let found = std::env::split_paths(path)
            .filter(|d| !d.as_os_str().is_empty())
            .map(|d| d.join(t))
            .find(|p| p.is_file());
        let (at, digest) = match found {
            Some(p) => (Some(p.display().to_string()), file_digest(&p)),
            None => (None, None),
        };
        tools.push(json!({ "tool": t, "path": at, "sha256": digest }));
    }
    let mut chains = Vec::new();
    if let Ok(rd) = fs::read_dir(rustup_home.join("toolchains")) {
        let mut names: Vec<String> = rd
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for n in names {
            let manifest = rustup_home
                .join("toolchains")
                .join(&n)
                .join("lib/rustlib/multirust-channel-manifest.toml");
            chains.push(json!({ "toolchain": n, "manifest_sha256": file_digest(&manifest) }));
        }
    }
    let settings = file_digest(&rustup_home.join("settings.toml"));
    canonical_digest(&json!({
        "tools": tools,
        "rustup_settings_sha256": settings,
        "rustup_toolchains": chains,
    }))
    .unwrap_or_default()
}

/// One judged head that may serve as a source.
#[derive(Debug, Clone)]
pub struct Source {
    pub commit: String,
    pub dir: PathBuf,
}

/// A fresh PASS in another head's store, found by cache key.
#[derive(Debug, Clone)]
pub struct Hit {
    pub commit: String,
    pub dir: PathBuf,
    pub digest: String,
    pub value: Value,
    pub finished_utc: String,
}

/// Every usable fresh PASS of the sources, by cache key; the newest wins.
#[derive(Debug, Clone, Default)]
pub struct Memo {
    pub policy: Option<Policy>,
    pub toolchain: String,
    pub sources: usize,
    by_key: BTreeMap<String, Hit>,
}

impl Memo {
    /// Read the sources. `env` is the environment the runner captures now
    /// (`runner::captured_env`); a receipt made under another one is skipped.
    pub fn build(policy: Policy, sources: &[Source], toolchain: &str, env: &Value) -> Memo {
        let mut by_key: BTreeMap<String, Hit> = BTreeMap::new();
        for s in sources {
            let store = Store::new(&s.dir);
            let index = Index::load(&store);
            for r in index.receipts() {
                if !source_ok(r, toolchain) {
                    continue;
                }
                let Some(key) = r.value["volatile"]["cache"]["key"].as_str() else {
                    continue;
                };
                if cache::usable(r, key, &index, &store, env).is_err() {
                    continue;
                }
                let hit = Hit {
                    commit: s.commit.clone(),
                    dir: s.dir.clone(),
                    digest: r.digest.clone(),
                    value: r.value.clone(),
                    finished_utc: r.finished_utc().to_string(),
                };
                match by_key.get(key) {
                    Some(h) if h.finished_utc >= hit.finished_utc => {}
                    _ => {
                        by_key.insert(key.to_string(), hit);
                    }
                }
            }
        }
        Memo {
            policy: Some(policy),
            toolchain: toolchain.to_string(),
            sources: sources.len(),
            by_key,
        }
    }

    pub fn find(&self, key: &str) -> Option<&Hit> {
        self.by_key.get(key)
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

/// A fresh PASS that records the same toolchain id.
fn source_ok(r: &Stored, toolchain: &str) -> bool {
    r.value["volatile"]["cache"]["reused"] == false
        && r.verdict() == Some(Verdict::Pass)
        && !toolchain.is_empty()
        && r.value["volatile"]["toolchain"].as_str() == Some(toolchain)
}

/// Copy the source receipt and its logs into `dst`, so this head's store
/// holds everything a reuse receipt points at. Existing identical files are
/// success (the store never overwrites).
pub fn import(hit: &Hit, dst: &Store) -> Result<(), String> {
    let src = Store::new(&hit.dir);
    for member in ["stdout_sha256", "stderr_sha256"] {
        let sha = hit.value["core"][member]
            .as_str()
            .ok_or_else(|| format!("source receipt has no {member}"))?;
        let bytes = fs::read(src.dir().join("blobs").join(format!("{sha}.log")))
            .map_err(|e| format!("cannot read source log {sha}: {e}"))?;
        let got = dst.put_blob(&bytes).map_err(|e| e.to_string())?;
        if got != sha {
            return Err(format!("source log {sha} hashes to {got}"));
        }
    }
    let (digest, _) = dst.put_receipt(&hit.value).map_err(|e| e.to_string())?;
    if digest != hit.digest {
        return Err(format!(
            "source receipt {} hashes to {digest} when copied",
            hit.digest
        ));
    }
    Ok(())
}

/// The judged heads of one repository's evidence directory that `policy`
/// allows as sources for `head`, newest first. `is_ancestor(a, b)` answers
/// "is commit a an ancestor of b" (git in the worker, a map in tests).
pub fn sources(
    repo_evidence: &Path,
    head: &str,
    policy: Policy,
    is_ancestor: &dyn Fn(&str, &str) -> bool,
) -> Vec<Source> {
    if policy == Policy::Off {
        return Vec::new();
    }
    let mut out: Vec<(std::time::SystemTime, Source)> = Vec::new();
    let Ok(rd) = fs::read_dir(repo_evidence) else {
        return Vec::new();
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let hex40 = name.len() == 40
            && name
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !hex40 || name == head || !e.path().is_dir() {
            continue;
        }
        if policy == Policy::Ancestor && !is_ancestor(&name, head) {
            continue;
        }
        let t = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        out.push((
            t,
            Source {
                commit: name,
                dir: e.path(),
            },
        ));
    }
    out.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.commit.cmp(&b.1.commit)));
    out.into_iter().map(|(_, s)| s).collect()
}

/// `git merge-base --is-ancestor a b` in `clone`.
pub fn git_is_ancestor(clone: &Path, a: &str, b: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(clone)
        .args(["merge-base", "--is-ancestor", a, b])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// `build_inputs_digest` of a gate whose manifest names no inputs and that
/// has no convention inputs: a digest of nothing. Such a gate is never
/// answered by the memo.
pub fn empty_inputs_digest() -> String {
    canonical_digest(&json!({ "inputs": [] })).unwrap_or_default()
}

/// The replay key of a receipt: the cache key members with every
/// dependency named by its own replay key instead of its receipt digest.
/// With the memo in place a reused dependency carries its source's digest,
/// so two receipts with equal replay keys have equal cache keys at the time
/// the dependent would be looked up.
fn replay_key(
    r: &Value,
    by_digest: &BTreeMap<String, Value>,
    memo: &mut BTreeMap<String, String>,
    depth: usize,
) -> Option<String> {
    let core = &r["core"];
    let mut deps = Vec::new();
    if let Some(list) = core["dependencies"].as_array() {
        for d in list {
            let gate = d["gate"].as_str()?;
            let mut digest = d["receipt"].as_str()?.to_string();
            // Follow a same-commit reuse to the fresh receipt it stands for.
            if let Some(v) = by_digest.get(&digest) {
                if v["volatile"]["cache"]["reused"] == true {
                    if let Some(f) = v["volatile"]["cache"]["reused_from"].as_str() {
                        digest = f.to_string();
                    }
                }
            }
            let k = match memo.get(&digest) {
                Some(k) => k.clone(),
                None if depth < 32 => {
                    let v = by_digest.get(&digest)?;
                    let k = replay_key(v, by_digest, memo, depth + 1)?;
                    memo.insert(digest.clone(), k.clone());
                    k
                }
                None => return None,
            };
            deps.push(json!({ "gate": gate, "key": k }));
        }
    }
    canonical_digest(&json!({
        "gate": core["gate"],
        "manifest_digest": core["manifest_digest"],
        "binary_sha256": core["binary_sha256"],
        "build_inputs_digest": core["build_inputs_digest"],
        "machine_digest": core["machine_digest"],
        "env": core["env"],
        "pins": core["pins"],
        "dependencies": deps,
    }))
    .ok()
}

/// One fresh run found in the store.
#[derive(Debug, Clone)]
pub struct Run {
    pub repo: String,
    pub commit: String,
    pub gate: String,
    pub started_utc: String,
    pub finished_utc: String,
    pub duration_ms: u64,
    pub pass: bool,
    pub cacheable: bool,
    pub key: Option<String>,
}

/// What the replay decided for one fresh run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fate {
    /// The memo would have answered it from this commit.
    Reused { from: String },
    /// It would still run, and why.
    Ran(&'static str),
}

/// Read every fresh receipt of one repository's evidence directory.
pub fn runs_of(repo: &str, repo_evidence: &Path) -> Vec<Run> {
    let mut by_digest: BTreeMap<String, Value> = BTreeMap::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(repo_evidence) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                dirs.push(e.path());
            }
        }
    }
    dirs.sort();
    for d in &dirs {
        for r in Index::load(&Store::new(d)).receipts() {
            by_digest.insert(r.digest.clone(), r.value.clone());
        }
    }
    let empty = empty_inputs_digest();
    let mut memo: BTreeMap<String, String> = BTreeMap::new();
    let mut runs = Vec::new();
    let digests: Vec<String> = by_digest.keys().cloned().collect();
    for dg in digests {
        let v = by_digest[&dg].clone();
        let vol = &v["volatile"];
        if vol["cache"]["reused"] != false {
            continue;
        }
        let verdict = v["core"]["derived_verdict"].as_str().unwrap_or("");
        if verdict != "PASS" && verdict != "FAIL" {
            continue;
        }
        let key = match memo.get(&dg) {
            Some(k) => Some(k.clone()),
            None => {
                let k = replay_key(&v, &by_digest, &mut memo, 0);
                if let Some(k) = &k {
                    memo.insert(dg.clone(), k.clone());
                }
                k
            }
        };
        let clean = v["core"]["tree_clean_before"] == true
            && v["core"]["tree_clean_after"] == true
            && v["core"]["commit_unchanged_after"] == true
            && vol["timeout_exceeded"] == false;
        let cacheable = vol["cache"]["mode"] != "never"
            && v["core"]["build_inputs_digest"].as_str() != Some(empty.as_str())
            && clean;
        runs.push(Run {
            repo: repo.to_string(),
            commit: v["core"]["commit"].as_str().unwrap_or("").to_string(),
            gate: v["core"]["gate"].as_str().unwrap_or("").to_string(),
            started_utc: vol["started_utc"].as_str().unwrap_or("").to_string(),
            finished_utc: vol["finished_utc"].as_str().unwrap_or("").to_string(),
            duration_ms: vol["duration_ms"].as_u64().unwrap_or(0),
            pass: verdict == "PASS",
            cacheable,
            key,
        });
    }
    runs.sort_by(|a, b| {
        a.started_utc
            .cmp(&b.started_utc)
            .then_with(|| a.gate.cmp(&b.gate))
    });
    runs
}

/// Decide each run in time order: reused when an earlier fresh PASS of
/// another commit, finished before this run started, has the same replay
/// key and the policy allows its commit.
pub fn replay(runs: &[Run], policy: Policy, is_ancestor: &dyn Fn(&str, &str) -> bool) -> Vec<Fate> {
    let mut fates = Vec::new();
    for (i, r) in runs.iter().enumerate() {
        let fate = if policy == Policy::Off {
            Fate::Ran("the memo is off")
        } else if !r.cacheable {
            Fate::Ran("not cacheable (cache: never, no inputs, or an unclean run)")
        } else if let Some(key) = &r.key {
            let src = runs[..i].iter().rev().find(|s| {
                s.pass
                    && s.cacheable
                    && s.key.as_ref() == Some(key)
                    && s.commit != r.commit
                    && s.finished_utc <= r.started_utc
                    && (policy == Policy::Any || is_ancestor(&s.commit, &r.commit))
            });
            match src {
                Some(s) => Fate::Reused {
                    from: s.commit.clone(),
                },
                None => Fate::Ran("no earlier PASS with the same inputs"),
            }
        } else {
            Fate::Ran("a dependency receipt is missing")
        };
        fates.push(fate);
    }
    fates
}

/// The commits of `runs` that were not run before `since` (UTC text).
pub fn commits_since(runs: &[Run], since: &str) -> BTreeSet<String> {
    runs.iter()
        .filter(|r| r.started_utc.as_str() >= since)
        .map(|r| r.commit.clone())
        .collect()
}

pub const MEMO_USAGE: &str = "usage: aien-test memo-replay --state-dir DIR [--clones DIR] [--since UTC] [--policy ancestor|any] [--repo NAME]... [--tsv FILE]";

pub fn is_memo_command(c: &str) -> bool {
    c == "memo-replay"
}

/// `aien-test memo-replay`: replay the evidence store against the memo.
pub fn cli_main(argv: &[String]) -> i32 {
    let mut state: Option<PathBuf> = None;
    let mut clones: Option<PathBuf> = None;
    let mut since = String::new();
    let mut policies: Vec<Policy> = Vec::new();
    let mut repos: Vec<String> = Vec::new();
    let mut tsv: Option<PathBuf> = None;
    let mut it = argv.iter().skip(1);
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{a} needs a value"))
        };
        let r: Result<(), String> = (|| {
            match a.as_str() {
                "--state-dir" => state = Some(PathBuf::from(val()?)),
                "--clones" => clones = Some(PathBuf::from(val()?)),
                "--since" => since = val()?,
                "--policy" => policies.push(Policy::parse(&val()?)?),
                "--repo" => repos.push(val()?),
                "--tsv" => tsv = Some(PathBuf::from(val()?)),
                _ => return Err(format!("unknown argument {a:?}")),
            }
            Ok(())
        })();
        if let Err(e) = r {
            eprintln!("aien-test memo-replay: {e}\n{MEMO_USAGE}");
            return 2;
        }
    }
    let Some(state) = state else {
        eprintln!("{MEMO_USAGE}");
        return 2;
    };
    if policies.is_empty() {
        policies = vec![Policy::Ancestor, Policy::Any];
    }
    let ev = state.join("evidence");
    if repos.is_empty() {
        if let Ok(rd) = fs::read_dir(&ev) {
            repos = rd
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
        }
        repos.sort();
    }
    let mut rows: Vec<String> = vec![
        "policy\trepo\tcommit\tgate\tstarted_utc\tduration_ms\tverdict\tfate\tfrom_or_why"
            .to_string(),
    ];
    for policy in &policies {
        let mut tot = (0u64, 0u64, 0usize, 0usize);
        println!(
            "== policy {} (runs started at or after {:?})",
            policy.as_str(),
            since
        );
        println!("repo\thead\tgate_runs\treused\tgate_min_before\tgate_min_after");
        for repo in &repos {
            let runs = runs_of(repo, &ev.join(repo));
            let clone = clones.as_ref().map(|c| c.join(repo));
            let anc = |a: &str, b: &str| match &clone {
                Some(c) => git_is_ancestor(c, a, b),
                None => false,
            };
            let fates = replay(&runs, *policy, &anc);
            let mut per: BTreeMap<String, (usize, usize, u64, u64)> = BTreeMap::new();
            for (r, f) in runs.iter().zip(&fates) {
                if r.started_utc.as_str() < since.as_str() {
                    continue;
                }
                let e = per.entry(r.commit.clone()).or_default();
                e.0 += 1;
                e.2 += r.duration_ms;
                let (fate, why) = match f {
                    Fate::Reused { from } => {
                        e.1 += 1;
                        ("REUSED", from.clone())
                    }
                    Fate::Ran(w) => {
                        e.3 += r.duration_ms;
                        ("RAN", w.to_string())
                    }
                };
                rows.push(format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    policy.as_str(),
                    repo,
                    r.commit,
                    r.gate,
                    r.started_utc,
                    r.duration_ms,
                    if r.pass { "PASS" } else { "FAIL" },
                    fate,
                    why
                ));
            }
            for (c, (n, reused, before, after)) in &per {
                println!(
                    "{repo}\t{}\t{n}\t{reused}\t{:.1}\t{:.1}",
                    &c[..12.min(c.len())],
                    *before as f64 / 60000.0,
                    *after as f64 / 60000.0
                );
                tot.0 += before;
                tot.1 += after;
                tot.2 += n;
                tot.3 += reused;
            }
        }
        println!(
            "TOTAL\t-\t{}\t{}\t{:.1}\t{:.1}",
            tot.2,
            tot.3,
            tot.0 as f64 / 60000.0,
            tot.1 as f64 / 60000.0
        );
    }
    if let Some(p) = tsv {
        if let Err(e) = fs::write(&p, rows.join("\n") + "\n") {
            eprintln!("aien-test memo-replay: cannot write {}: {e}", p.display());
            return 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;
    use crate::testutil::TempDir;

    fn gate(extra: &str) -> Manifest {
        let text = format!(
            "gate: G\nmanifest_version: 1\nowner: t\nrequires:\n  - host\n{extra}run:\n  exec: run.sh\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 1m\n"
        );
        manifest::parse(text.as_bytes()).expect("manifest")
    }

    #[test]
    fn a_gate_without_inputs_or_with_cache_never_always_runs() {
        assert!(ineligible(&gate("")).is_some());
        assert!(ineligible(&gate("inputs:\n  - src/\ncache: never\n")).is_some());
        assert!(ineligible(&gate("inputs:\n  - src/\n")).is_none());
    }

    fn run(commit: &str, gate: &str, key: &str, t0: &str, t1: &str, pass: bool) -> Run {
        Run {
            repo: "r".into(),
            commit: commit.into(),
            gate: gate.into(),
            started_utc: t0.into(),
            finished_utc: t1.into(),
            duration_ms: 60000,
            pass,
            cacheable: true,
            key: Some(key.into()),
        }
    }

    #[test]
    fn replay_reuses_only_an_earlier_pass_of_another_commit_with_the_same_key() {
        let runs = vec![
            run("a", "G", "k1", "01", "02", true),
            run("b", "G", "k1", "03", "04", true), // reused from a
            run("c", "G", "k2", "05", "06", true), // other key: runs
            run("d", "H", "k3", "07", "08", false), // FAIL is never a source
            run("e", "H", "k3", "09", "10", true), // so this runs
            run("f", "G", "k1", "01", "05", true), // started before a finished: runs
        ];
        let mut runs = runs;
        runs.sort_by(|x, y| x.started_utc.cmp(&y.started_utc));
        let fates = replay(&runs, Policy::Any, &|_, _| true);
        let by: BTreeMap<(String, String), Fate> = runs
            .iter()
            .zip(fates)
            .map(|(r, f)| ((r.commit.clone(), r.gate.clone()), f))
            .collect();
        assert_eq!(
            by[&("b".into(), "G".into())],
            Fate::Reused { from: "a".into() }
        );
        assert!(matches!(by[&("c".into(), "G".into())], Fate::Ran(_)));
        assert!(matches!(by[&("e".into(), "H".into())], Fate::Ran(_)));
        assert!(matches!(by[&("f".into(), "G".into())], Fate::Ran(_)));
    }

    #[test]
    fn ancestor_policy_needs_an_ancestor() {
        let runs = vec![
            run("a", "G", "k1", "01", "02", true),
            run("b", "G", "k1", "03", "04", true),
        ];
        let fates = replay(&runs, Policy::Ancestor, &|_, _| false);
        assert!(matches!(fates[1], Fate::Ran(_)));
        let fates = replay(&runs, Policy::Ancestor, &|a, b| a == "a" && b == "b");
        assert_eq!(fates[1], Fate::Reused { from: "a".into() });
        let fates = replay(&runs, Policy::Off, &|_, _| true);
        assert!(matches!(fates[1], Fate::Ran(_)));
    }

    #[test]
    fn uncacheable_runs_are_never_reused() {
        let mut b = run("b", "G", "k1", "03", "04", true);
        b.cacheable = false;
        let runs = vec![run("a", "G", "k1", "01", "02", true), b];
        let fates = replay(&runs, Policy::Any, &|_, _| true);
        assert!(matches!(fates[1], Fate::Ran(_)));
    }

    #[test]
    fn toolchain_id_follows_tool_bytes_and_rustup_manifests() {
        let t = TempDir::new("memo-tc");
        let bin = t.path().join("bin");
        let rh = t.path().join("rustup");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(rh.join("toolchains/1.0/lib/rustlib")).unwrap();
        fs::write(bin.join("cc"), b"cc v1").unwrap();
        let mf = rh.join("toolchains/1.0/lib/rustlib/multirust-channel-manifest.toml");
        fs::write(&mf, b"rustc 1.0").unwrap();
        let p = bin.display().to_string();
        let a = toolchain_id_in(&p, &rh);
        assert_eq!(a, toolchain_id_in(&p, &rh));
        fs::write(bin.join("cc"), b"cc v2").unwrap();
        let b = toolchain_id_in(&p, &rh);
        assert_ne!(a, b, "a changed compiler flips the toolchain id");
        fs::write(&mf, b"rustc 1.1").unwrap();
        assert_ne!(
            b,
            toolchain_id_in(&p, &rh),
            "a changed rustc release flips it"
        );
        fs::write(t.path().join("unrelated"), b"x").unwrap();
        let c = toolchain_id_in(&p, &rh);
        fs::write(t.path().join("unrelated"), b"y").unwrap();
        assert_eq!(c, toolchain_id_in(&p, &rh), "an unrelated file does not");
    }

    #[test]
    fn sources_skip_the_head_non_commits_and_non_ancestors() {
        let t = TempDir::new("memo-src");
        let a = "a".repeat(40);
        let b = "b".repeat(40);
        let h = "c".repeat(40);
        for d in [&a, &b, &h, &"notasha".to_string()] {
            fs::create_dir_all(t.path().join(d)).unwrap();
        }
        let any = sources(t.path(), &h, Policy::Any, &|_, _| false);
        let names: BTreeSet<String> = any.iter().map(|s| s.commit.clone()).collect();
        assert_eq!(names, [a.clone(), b.clone()].into_iter().collect());
        let anc = sources(t.path(), &h, Policy::Ancestor, &|x, _| x == a);
        assert_eq!(anc.len(), 1);
        assert_eq!(anc[0].commit, a);
        assert!(sources(t.path(), &h, Policy::Off, &|_, _| true).is_empty());
    }
}
