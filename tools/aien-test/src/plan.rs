//! The planner (ADR 0033 Decision 12; local pipeline amendments A1 and A4).
//!
//! `aien-test plan [--base REF]` reads two commits of the repository through
//! git objects, never through the working tree, so the same plan can be
//! recomputed later for a commit that is not checked out (`verify COMMIT`
//! does that to check the bundle's `graph_digest`). Only `plan` on a tree
//! with uncommitted changes adds those paths, and only with `--allow-dirty`.
//!
//! What selects a gate at HEAD:
//! - a changed path under one of its `inputs`, under a D3 implied input
//!   (`tests/<dir>/<name>.gate` implies `src/<dir>/<name>.*` and
//!   `tests/<dir>/<name>_test.*`), its own manifest file, or its golden file;
//! - a changed pin (`*.lock` whose content is one 40-hex line) in its inputs;
//! - crumb ancestry: the `.crumb` of every directory above a changed path is
//!   read at HEAD, and the paths named by its `related` or `links` members
//!   (if the file has them; today's schema has neither) count as changed;
//! - the always set (`.aien/always` at BASE, trust anchor A1): one gate id per
//!   line, selected whatever changed, never excluded by path filtering.
//!
//! The selection is closed over `depends_on` (a dependency is never dropped)
//! and over dependents (a gate whose dependency changed gets a new check id
//! and must run again).
//!
//! Rules that stop a commit from passing:
//! - a changed path that no gate's inputs cover and that `.aien/unmapped-ok`
//!   at BASE does not allow is UNMAPPED: the commit verdict is NOT_RUN with
//!   reason UNMAPPED_PATH. Manifest files, `.crumb` files and the pipeline's
//!   own configuration are always mapped (the next rule covers the latter);
//! - a change under `.aien/`, `tests/gates/`, `.github/workflows/`,
//!   `.forgejo/` or `tools/aien-test/` blocks the commit (BLOCKED_OPERATOR,
//!   PIPELINE_CONFIG_CHANGED) unless the operator passes `--operator`;
//! - a gate with `requires: operator` is Blocked (OPERATOR_STEP_PENDING) in
//!   an automatic run and takes its normal section under `--operator`: the
//!   operator starting the run by hand is what the manifest asks for;
//! - every gate that exists at BASE must exist at HEAD; a deleted gate gives
//!   NOT_RUN with reason GATE_REMOVED.
//!
//! The commit verdict algebra (`judge`) is Decision 7 applied to the plan: a
//! required gate whose dependency has no receipt is NOT_RUN
//! (UNVERIFIED_DEPENDENCY), a FAIL propagates (DEPENDENCY_FAILED), and
//! nothing promotes NOT_RUN to PASS. ADVISORY gates never count.
//!
//! Hardware identity: `cache_scope: hardware` gates need one and the ADR does
//! not define its content. When none is given the machine digest stands in,
//! and the plan says so.

use crate::evidence::{canonical_digest, sha256_hex};
use crate::graph::{self, Graph};
use crate::identity::{self, Binding};
use crate::manifest::{self, Exactness, Manifest};
use crate::process;
use crate::verdict::Verdict;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Paths whose change blocks a commit unless the operator says otherwise.
pub const PIPELINE_CONFIG: [&str; 5] = [
    ".aien/",
    "tests/gates/",
    ".github/workflows/",
    ".forgejo/",
    "tools/aien-test/",
];
pub const ALWAYS_FILE: &str = ".aien/always";
pub const UNMAPPED_OK_FILE: &str = ".aien/unmapped-ok";
const SKIP_DIRS: [&str; 5] = [".git", "target", "build", "vendor", "node_modules"];

/// What to plan.
#[derive(Debug, Clone)]
pub struct Request {
    /// Any revision git resolves (`origin/main`, a sha, `HEAD~1`).
    pub base: String,
    pub head: String,
    /// Uncommitted changes to add to the changed set (`plan --allow-dirty`
    /// only; `verify` never sets them).
    pub dirty: Vec<String>,
    /// `--operator`: a pipeline configuration change is allowed.
    pub operator: bool,
    pub compiler_identity: String,
    pub flags: Vec<String>,
    /// Empty: the machine digest stands in (recorded in the plan).
    pub hardware_identity: String,
    /// Gates known to have a reusable receipt (filled by `probe_cached`).
    pub cached: BTreeSet<String>,
}

impl Request {
    pub fn new(base: &str, head: &str) -> Request {
        Request {
            base: base.to_string(),
            head: head.to_string(),
            dirty: Vec::new(),
            operator: false,
            compiler_identity: String::new(),
            flags: Vec::new(),
            hardware_identity: String::new(),
            cached: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Section {
    Required,
    Cached,
    NeedsSpark,
    NeedsGb10,
    Advisory,
    Blocked,
}

impl Section {
    pub fn title(self) -> &'static str {
        match self {
            Section::Required => "Required",
            Section::Cached => "Cached",
            Section::NeedsSpark => "Needs Spark",
            Section::NeedsGb10 => "Needs GB10",
            Section::Advisory => "Advisory",
            Section::Blocked => "Blocked",
        }
    }

    pub const ALL: [Section; 6] = [
        Section::Required,
        Section::Cached,
        Section::NeedsSpark,
        Section::NeedsGb10,
        Section::Advisory,
        Section::Blocked,
    ];
}

/// One selected gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub gate: String,
    pub check_id: String,
    pub pool: String,
    pub exactness: Exactness,
    pub cache_scope: manifest::CacheScope,
    /// False only for ADVISORY gates.
    pub required: bool,
    pub section: Section,
    /// Why it is Blocked (empty otherwise).
    pub reason: String,
    pub depends_on: Vec<String>,
    pub always: bool,
    /// Why it was selected: the first matching path, `always`, `dependency`
    /// or `dependent`.
    pub because: String,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub base: String,
    pub head: String,
    pub changed: Vec<String>,
    /// Paths added by crumb ancestry (not themselves checked for mapping).
    pub crumb_paths: Vec<String>,
    pub pins: Vec<String>,
    pub unmapped: Vec<String>,
    /// Gates at BASE that are gone at HEAD.
    pub removed: Vec<String>,
    pub config_changed: Vec<String>,
    /// Set when the whole plan is blocked (PIPELINE_CONFIG_CHANGED).
    pub blocked: Option<String>,
    /// Selected gates, dependencies first.
    pub entries: Vec<Entry>,
    /// sha256 over the canonical `[{gate, check_id, depends_on}]` of the
    /// selected gates, sorted by gate.
    pub graph_digest: String,
    pub notes: Vec<String>,
    /// The graph at HEAD (every gate, not only the selected ones).
    pub graph: Graph,
}

// ---------------------------------------------------------------- git

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let mut cmd = process::git_command();
    cmd.env("GIT_OPTIONAL_LOCKS", "0")
        .arg("-C")
        .arg(root)
        .args(args);
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

fn git_text(root: &Path, args: &[&str]) -> Result<String, String> {
    Ok(String::from_utf8_lossy(&git(root, args)?).into_owned())
}

/// The 40-hex commit id of `rev`.
pub fn resolve(root: &Path, rev: &str) -> Result<String, String> {
    let s = git_text(
        root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{rev}^{{commit}}"),
        ],
    )
    .map_err(|_| format!("cannot resolve {rev:?} to a commit"))?;
    let s = s.trim().to_string();
    if is_hex(&s, 40) {
        Ok(s)
    } else {
        Err(format!("{rev:?} is not a commit"))
    }
}

/// The first of `candidates` that resolves.
pub fn resolve_first(root: &Path, candidates: &[&str]) -> Result<String, String> {
    for c in candidates {
        if let Ok(s) = resolve(root, c) {
            return Ok(s);
        }
    }
    Err(format!("none of {candidates:?} resolves to a commit"))
}

/// The BASE the worker and `verify` plan `head` against: `base_ref` (then
/// `main`); when `head` is already on that line (the tip of main or an older
/// main commit) its first parent, so a push to main verifies what it changed
/// and both sides compute the same plan. The flag says whether `head` is on
/// the base line (the `main` trust class).
pub fn base_for(root: &Path, base_ref: &str, head: &str) -> Result<(String, bool), String> {
    let first_parent = format!("{head}~1");
    let base = resolve_first(root, &[base_ref, "main", &first_parent])?;
    let on_base = base == head || git(root, &["merge-base", "--is-ancestor", head, &base]).is_ok();
    if on_base {
        Ok((resolve(root, &first_parent)?, true))
    } else {
        Ok((base, false))
    }
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Every tracked file of `commit`: path to blob id.
pub fn tree_rows(root: &Path, commit: &str) -> Result<BTreeMap<String, String>, String> {
    let out = git(root, &["ls-tree", "-r", "-z", commit])?;
    let text = String::from_utf8(out).map_err(|_| "git ls-tree output is not UTF-8".to_string())?;
    let mut rows = BTreeMap::new();
    for rec in text.split('\0').filter(|r| !r.is_empty()) {
        let (meta, path) = rec
            .split_once('\t')
            .ok_or_else(|| format!("unparsable ls-tree record {rec:?}"))?;
        let f: Vec<&str> = meta.split(' ').collect();
        if f.len() != 3 || f[2].len() != 40 {
            return Err(format!("unparsable ls-tree record {rec:?}"));
        }
        if f[1] == "blob" {
            rows.insert(path.to_string(), f[2].to_string());
        }
    }
    Ok(rows)
}

/// The bytes of `path` at `commit`, None when it does not exist there.
pub fn show(root: &Path, commit: &str, path: &str) -> Option<Vec<u8>> {
    git(root, &["show", &format!("{commit}:{path}")]).ok()
}

/// `git diff --name-only BASE...HEAD`: paths changed on HEAD's side since the
/// merge base.
pub fn changed_between(root: &Path, base: &str, head: &str) -> Result<Vec<String>, String> {
    let out = git(
        root,
        &["diff", "--name-only", "-z", &format!("{base}...{head}")],
    )?;
    let text = String::from_utf8_lossy(&out);
    let mut v: Vec<String> = text
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    v.sort();
    v.dedup();
    Ok(v)
}

/// Paths with uncommitted changes, untracked files included.
pub fn uncommitted(root: &Path) -> Result<Vec<String>, String> {
    let out = git(
        root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
    let text = String::from_utf8_lossy(&out);
    let mut v = Vec::new();
    let mut it = text.split('\0').filter(|s| !s.is_empty());
    while let Some(rec) = it.next() {
        if rec.len() < 4 {
            continue;
        }
        let xy = &rec[..2];
        v.push(rec[3..].to_string());
        if xy.starts_with('R') || xy.starts_with('C') {
            // The original name follows as its own record.
            if let Some(orig) = it.next() {
                v.push(orig.to_string());
            }
        }
    }
    v.sort();
    v.dedup();
    Ok(v)
}

fn lines_at(root: &Path, commit: &str, path: &str) -> Vec<String> {
    match show(root, commit, path) {
        Some(bytes) => String::from_utf8_lossy(&bytes)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect(),
        None => Vec::new(),
    }
}

/// `.aien/always` at `commit` (A1: read from BASE, never the working tree).
pub fn always_set(root: &Path, commit: &str) -> Vec<String> {
    lines_at(root, commit, ALWAYS_FILE)
}

/// `.aien/unmapped-ok` at `commit`.
pub fn unmapped_ok(root: &Path, commit: &str) -> Vec<String> {
    lines_at(root, commit, UNMAPPED_OK_FILE)
}

// ---------------------------------------------------------------- gates at a commit

fn skipped(path: &str, rows: &BTreeMap<String, String>) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    let mut prefix = String::new();
    for dir in &parts[..parts.len() - 1] {
        if SKIP_DIRS.contains(dir) {
            return true;
        }
        if !prefix.is_empty() {
            prefix.push('/');
        }
        prefix.push_str(dir);
        if rows.contains_key(&format!("{prefix}/.aien-test-ignore")) {
            return true;
        }
    }
    rows.contains_key(".aien-test-ignore")
}

/// The gate graph of `commit`, read from git objects, plus each gate's
/// manifest path relative to the root.
pub fn graph_at(
    root: &Path,
    commit: &str,
    rows: &BTreeMap<String, String>,
) -> Result<(Graph, BTreeMap<String, String>), String> {
    let mut entries: Vec<(PathBuf, Manifest)> = Vec::new();
    let mut rel_of: BTreeMap<String, String> = BTreeMap::new();
    for path in rows.keys() {
        if !path.ends_with(".gate") || skipped(path, rows) {
            continue;
        }
        let bytes =
            show(root, commit, path).ok_or_else(|| format!("cannot read {path} at {commit}"))?;
        let m = manifest::parse(&bytes).map_err(|e| format!("BAD_MANIFEST {path}: {}", e.0))?;
        rel_of.insert(m.gate.clone(), path.clone());
        entries.push((root.join(path), m));
    }
    let g = graph::build(entries).map_err(|e| e.to_string())?;
    Ok((g, rel_of))
}

/// Gate ids at `commit`, tolerating manifests that no longer parse (a broken
/// manifest at BASE is still a gate that must not vanish).
fn gate_ids_at(root: &Path, commit: &str, rows: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for path in rows.keys() {
        if !path.ends_with(".gate") || skipped(path, rows) {
            continue;
        }
        if let Some(bytes) = show(root, commit, path) {
            if let Ok(m) = manifest::parse(&bytes) {
                ids.insert(m.gate);
            } else if let Some(id) = String::from_utf8_lossy(&bytes)
                .lines()
                .find_map(|l| l.strip_prefix("gate:").map(|v| v.trim().to_string()))
            {
                ids.insert(id);
            }
        }
    }
    ids
}

fn stem_of(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rfind('.') {
        Some(i) if i > 0 => &name[..i],
        _ => name,
    }
}

/// D3 implied inputs of a manifest at `rel`, from the tree rows.
pub fn implied_inputs(rel: &str, rows: &BTreeMap<String, String>) -> Vec<String> {
    let mut out = Vec::new();
    let (dir, _) = rel.rsplit_once('/').unwrap_or(("", rel));
    let stem = stem_of(rel).to_string();
    let sub = match dir.strip_prefix("tests") {
        Some(s) => s.trim_start_matches('/').to_string(),
        None => return out,
    };
    let join = |base: &str| {
        if sub.is_empty() {
            base.to_string()
        } else {
            format!("{base}/{sub}")
        }
    };
    for (base, want) in [
        (join("src"), stem.clone()),
        (join("tests"), format!("{stem}_test")),
    ] {
        let prefix = format!("{base}/");
        for path in rows.keys() {
            if let Some(name) = path.strip_prefix(&prefix) {
                if !name.contains('/') && stem_of(name) == want {
                    out.push(path.clone());
                }
            }
        }
    }
    out
}

/// Every path a gate watches: explicit inputs, implied inputs, its manifest
/// and its golden file.
pub fn watched(m: &Manifest, rel: &str, rows: &BTreeMap<String, String>) -> Vec<String> {
    let mut v: Vec<String> = m
        .inputs
        .iter()
        .map(|i| i.trim_end_matches('/').to_string())
        .collect();
    v.extend(implied_inputs(rel, rows));
    v.push(rel.to_string());
    if let Some(g) = &m.oracle.golden {
        v.push(g.trim_end_matches('/').to_string());
    }
    v.sort();
    v.dedup();
    v
}

/// `input` is `path` or a directory above it.
pub fn covers(input: &str, path: &str) -> bool {
    let input = input.trim_end_matches('/');
    input.is_empty() || path == input || path.starts_with(&format!("{input}/"))
}

/// `*` matches any run of characters (slashes included); a pattern ending in
/// `/` matches everything under that directory; anything else is exact.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    if let Some(dir) = pattern.strip_suffix('/') {
        return covers(dir, path);
    }
    fn walk(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(b'*') => (0..=s.len()).any(|i| walk(&p[1..], &s[i..])),
            Some(c) => s.first() == Some(c) && walk(&p[1..], &s[1..]),
        }
    }
    walk(pattern.as_bytes(), path.as_bytes())
}

/// A pin: a `*.lock` file holding exactly one 40-hex commit id.
pub fn is_pin(path: &str, bytes: &[u8]) -> bool {
    path.ends_with(".lock") && is_hex(String::from_utf8_lossy(bytes).trim(), 40)
}

fn strings_of(v: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(arr) = v.as_array() {
        for item in arr {
            if let Some(s) = item.as_str() {
                out.push(s.to_string());
            } else if let Some(o) = item.as_object() {
                for k in ["path", "target", "dir", "id"] {
                    if let Some(s) = o.get(k).and_then(Value::as_str) {
                        out.push(s.to_string());
                        break;
                    }
                }
            }
        }
    }
    out
}

/// Crumb ancestry: for every directory above a changed path (the root
/// included), the paths its `.crumb` at `commit` names under `related` or
/// `links`. Today's crumb schema has neither member, so this is empty until
/// one appears; the hook is here so a change under a directory can pull in
/// what its crumb says belongs with it.
pub fn crumb_paths(
    root: &Path,
    commit: &str,
    changed: &[String],
    rows: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut dirs: BTreeSet<String> = BTreeSet::new();
    for p in changed {
        let mut cur = p.as_str();
        dirs.insert(String::new());
        while let Some((parent, _)) = cur.rsplit_once('/') {
            dirs.insert(parent.to_string());
            cur = parent;
        }
    }
    let mut out: BTreeSet<String> = BTreeSet::new();
    for dir in dirs {
        let file = if dir.is_empty() {
            ".crumb".to_string()
        } else {
            format!("{dir}/.crumb")
        };
        if !rows.contains_key(&file) {
            continue;
        }
        let Some(bytes) = show(root, commit, &file) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        for key in ["related", "links"] {
            for s in strings_of(&v[key]) {
                let s = s.trim_matches('/').to_string();
                if !s.is_empty() {
                    out.insert(s);
                }
            }
        }
    }
    out.into_iter().collect()
}

fn under_config(path: &str) -> bool {
    PIPELINE_CONFIG.iter().any(|c| covers(c, path))
}

fn is_crumb_file(path: &str) -> bool {
    path == ".crumb"
        || path.ends_with("/.crumb")
        || path == ".crumb.local"
        || path.ends_with("/.crumb.local")
}

// ---------------------------------------------------------------- the plan

/// Compute the plan for `req` in the repository at `root`.
pub fn compute(root: &Path, req: &Request) -> Result<Plan, String> {
    let base = resolve(root, &req.base)?;
    let head = resolve(root, &req.head)?;
    let mut notes: Vec<String> = Vec::new();

    let mut changed = changed_between(root, &base, &head)?;
    changed.extend(req.dirty.iter().cloned());
    changed.sort();
    changed.dedup();

    let rows_head = tree_rows(root, &head)?;
    let rows_base = tree_rows(root, &base)?;
    let (graph, rel_of) = graph_at(root, &head, &rows_head)?;
    let ids_base = gate_ids_at(root, &base, &rows_base);
    let removed: Vec<String> = ids_base
        .iter()
        .filter(|id| graph.index_of(id).is_none())
        .cloned()
        .collect();

    let config_changed: Vec<String> = changed
        .iter()
        .filter(|p| under_config(p))
        .cloned()
        .collect();
    let blocked = if !config_changed.is_empty() && !req.operator {
        Some("PIPELINE_CONFIG_CHANGED".to_string())
    } else {
        None
    };
    if !config_changed.is_empty() && req.operator {
        notes.push("pipeline configuration changed; allowed by --operator".to_string());
    }

    let crumb = crumb_paths(root, &head, &changed, &rows_head);
    let pins: Vec<String> = changed
        .iter()
        .filter(|p| {
            show(root, &head, p)
                .or_else(|| show(root, &base, p))
                .is_some_and(|b| is_pin(p, &b))
        })
        .cloned()
        .collect();
    let always = always_set(root, &base);
    let ok_patterns = unmapped_ok(root, &base);

    // What each gate watches, and which changed path (if any) touches it.
    let mut watched_of: Vec<Vec<String>> = Vec::with_capacity(graph.len());
    for i in 0..graph.len() {
        let node = graph.node(i);
        let rel = rel_of.get(&node.id).cloned().unwrap_or_default();
        watched_of.push(watched(&node.manifest, &rel, &rows_head));
    }
    let mut all_changed: Vec<String> = changed.clone();
    all_changed.extend(crumb.iter().cloned());
    let mut because: BTreeMap<usize, String> = BTreeMap::new();
    for (i, w) in watched_of.iter().enumerate() {
        if let Some(p) = all_changed
            .iter()
            .find(|p| w.iter().any(|input| covers(input, p)))
        {
            because.insert(i, p.clone());
        }
    }

    let unmapped: Vec<String> = changed
        .iter()
        .filter(|p| {
            let mapped = p.ends_with(".gate")
                || is_crumb_file(p)
                || under_config(p)
                || watched_of
                    .iter()
                    .any(|w| w.iter().any(|input| covers(input, p)))
                || ok_patterns.iter().any(|pat| glob_match(pat, p));
            !mapped
        })
        .cloned()
        .collect();

    let mut seeds: BTreeSet<usize> = because.keys().copied().collect();
    let mut always_idx: BTreeSet<usize> = BTreeSet::new();
    let mut removed = removed;
    for g in &always {
        match graph.index_of(g) {
            Some(i) => {
                always_idx.insert(i);
                seeds.insert(i);
            }
            None => {
                if !removed.contains(g) {
                    removed.push(g.clone());
                }
                notes.push(format!("always-set gate {g} does not exist at HEAD"));
            }
        }
    }
    let mut with_dependents: BTreeSet<usize> = seeds.clone();
    for &s in &seeds {
        with_dependents.extend(graph.transitive_dependents(s));
    }
    let targets: Vec<usize> = with_dependents.iter().copied().collect();
    let selected = graph.with_dependencies(&targets);

    // Check ids for every gate at HEAD, dependencies first.
    let mut binding = identity::local_binding(&req.hardware_identity)?;
    if binding.hardware_identity.is_empty() {
        binding.hardware_identity = binding.machine_digest.clone();
        if graph
            .nodes()
            .iter()
            .any(|n| n.manifest.cache_scope == manifest::CacheScope::Hardware)
        {
            notes.push("hardware identity not given; the machine digest stands in for cache_scope: hardware".to_string());
        }
    }
    let ids = check_ids(
        root,
        &head,
        &graph,
        &rows_head,
        &req.compiler_identity,
        &req.flags,
        &binding,
    )?;

    let mut entries: Vec<Entry> = Vec::new();
    for &i in graph.order() {
        if !selected.contains(&i) {
            continue;
        }
        let node = graph.node(i);
        let m = &node.manifest;
        let is_always = always_idx.contains(&i);
        let required = m.exactness != Exactness::Advisory;
        // A `requires: operator` gate is never scheduled automatically; the
        // operator starting the run by hand (`--operator`) is what lets it
        // run, exactly as for a pipeline configuration change (ADR 0033
        // Decision 2 and amendment A4; sc #614).
        let operator_gate = m.requires.iter().any(|r| r == "operator");
        if operator_gate && req.operator && blocked.is_none() {
            notes.push(format!("operator gate {}: started by --operator", node.id));
        }
        let (section, reason) = if let Some(b) = &blocked {
            (Section::Blocked, b.clone())
        } else if operator_gate && !req.operator {
            (Section::Blocked, "OPERATOR_STEP_PENDING".to_string())
        } else if !required {
            (Section::Advisory, String::new())
        } else if m.pool() == "gb10" {
            (Section::NeedsGb10, String::new())
        } else if m.pool() == "qemu"
            || m.requires
                .iter()
                .any(|r| r == "aarch64" || r == "bare_metal")
        {
            (Section::NeedsSpark, String::new())
        } else if req.cached.contains(&node.id) {
            (Section::Cached, String::new())
        } else {
            (Section::Required, String::new())
        };
        let why = if is_always {
            "always".to_string()
        } else if let Some(p) = because.get(&i) {
            p.clone()
        } else if seeds
            .iter()
            .any(|s| graph.transitive_dependents(*s).contains(&i))
        {
            "dependent".to_string()
        } else {
            "dependency".to_string()
        };
        entries.push(Entry {
            gate: node.id.clone(),
            check_id: ids.get(&node.id).cloned().unwrap_or_default(),
            pool: m.pool().to_string(),
            exactness: m.exactness,
            cache_scope: m.cache_scope,
            required,
            section,
            reason,
            depends_on: identity::declared_dependencies(m),
            always: is_always,
            because: why,
        });
    }
    let graph_digest = digest_of(&entries)?;
    Ok(Plan {
        base,
        head,
        changed,
        crumb_paths: crumb,
        pins,
        unmapped,
        removed,
        config_changed,
        blocked,
        entries,
        graph_digest,
        notes,
        graph,
    })
}

/// sha256 over the canonical `[{gate, check_id, depends_on}]`, sorted by gate.
pub fn digest_of(entries: &[Entry]) -> Result<String, String> {
    let mut rows: Vec<&Entry> = entries.iter().collect();
    rows.sort_by(|a, b| a.gate.cmp(&b.gate));
    let v: Vec<Value> = rows
        .iter()
        .map(|e| json!({ "gate": e.gate, "check_id": e.check_id, "depends_on": e.depends_on }))
        .collect();
    canonical_digest(&json!(v)).map_err(|e| e.to_string())
}

/// Check ids of every gate of `graph` at `commit`, computed from the tree
/// rows (no working tree), dependencies first.
pub fn check_ids(
    root: &Path,
    commit: &str,
    graph: &Graph,
    rows: &BTreeMap<String, String>,
    compiler_identity: &str,
    flags: &[String],
    binding: &Binding,
) -> Result<BTreeMap<String, String>, String> {
    let mut done: BTreeMap<String, String> = BTreeMap::new();
    for &i in graph.order() {
        let m = &graph.node(i).manifest;
        let mut dep_ids = Vec::new();
        for d in identity::declared_dependencies(m) {
            match done.get(&d) {
                Some(id) => dep_ids.push(id.clone()),
                None => return Err(format!("gate {} declares unknown dependency {d}", m.gate)),
            }
        }
        let golden = match &m.oracle.golden {
            Some(g) => show(root, commit, g),
            None => None,
        };
        let obj = identity::identity_object_from_parts(
            m,
            identity::input_blob_ids_from_rows(m, rows),
            golden.as_deref(),
            compiler_identity,
            flags,
            &dep_ids,
            binding,
        )?;
        done.insert(m.gate.clone(), identity::id_of(&obj)?);
    }
    Ok(done)
}

/// Gates with a reusable receipt right now, for the Cached section. Only
/// meaningful when `head` is the checked-out commit and the tree is clean;
/// the caller checks that. A gate counts as cached only when every gate it
/// depends on would be reused too.
pub fn probe_cached(root: &Path, graph: &Graph, opts: &crate::runner::Options) -> BTreeSet<String> {
    use crate::cache::{self, Lookup};
    use crate::evidence::{Index, Store};
    let store = Store::new(&opts.evidence_dir);
    let index = Index::load(&store);
    let env = crate::runner::captured_env();
    let mut stands_on: BTreeMap<usize, Option<String>> = BTreeMap::new();
    let mut out = BTreeSet::new();
    for &i in graph.order() {
        let node = graph.node(i);
        let m = &node.manifest;
        let mut reuse: Option<String> = None;
        let deps_ok = m.depends_on.iter().all(|d| {
            graph
                .index_of(d)
                .and_then(|di| stands_on.get(&di).cloned().flatten())
                .is_some()
        });
        if cache::policy(m.cache_allowed, opts.no_cache || opts.release) == cache::Mode::Allow
            && deps_ok
        {
            if let Ok(id) =
                crate::runner::identity(root, &node.path, m, &opts.hardware, opts.sandbox)
            {
                let deps: Vec<(String, String)> = m
                    .depends_on
                    .iter()
                    .filter_map(|d| {
                        let di = graph.index_of(d)?;
                        Some((d.clone(), stands_on.get(&di)?.clone()?))
                    })
                    .collect();
                let now = cache::Experiment {
                    manifest_digest: m.digest.clone(),
                    binary_sha256: id.binary_sha.clone(),
                    build_inputs_digest: id.inputs_digest.clone(),
                    machine_digest: id.machine_digest.clone(),
                    dependencies: deps,
                    pins: opts.pins.clone(),
                    args_override: None,
                };
                if let Ok(k) = cache::key(&now, &index) {
                    if let Lookup::Hit(src) = cache::lookup(&index, &store, &k, &env) {
                        if src.verdict() == Some(Verdict::Pass) {
                            reuse = Some(src.digest.clone());
                            out.insert(node.id.clone());
                        }
                    }
                }
            }
        }
        stands_on.insert(i, reuse);
    }
    out
}

// ---------------------------------------------------------------- the verdict algebra

/// What is known about one gate after a run (or from receipts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Known {
    pub verdict: Verdict,
    pub reason: String,
    /// The receipt digest, when one was written.
    pub receipt: Option<String>,
}

/// The commit verdict and the effective per-gate verdicts (Decision 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judged {
    pub verdict: Verdict,
    pub reason: String,
    pub gates: BTreeMap<String, (Verdict, String)>,
}

fn worse(a: Verdict, b: Verdict) -> Verdict {
    if b.exit_code() > a.exit_code() {
        b
    } else {
        a
    }
}

/// Apply the algebra to a plan and what is known about its gates. Unknown
/// gates are NOT_RUN (no receipt). A required gate is NOT_RUN with
/// UNVERIFIED_DEPENDENCY when a dependency has no PASS receipt, and FAIL with
/// DEPENDENCY_FAILED when a dependency failed. Advisory gates are judged but
/// never move the commit verdict. NOT_RUN is never promoted.
pub fn judge(plan: &Plan, known: &BTreeMap<String, Known>) -> Judged {
    let mut gates: BTreeMap<String, (Verdict, String)> = BTreeMap::new();
    for e in &plan.entries {
        let mut v = match known.get(&e.gate) {
            Some(k) => (k.verdict, k.reason.clone()),
            None => (Verdict::NotRun, "NO_RECEIPT".to_string()),
        };
        if e.section == Section::Blocked {
            v = (Verdict::BlockedOperator, e.reason.clone());
        }
        if known
            .get(&e.gate)
            .is_some_and(|k| k.verdict == Verdict::Pass && k.receipt.is_none())
        {
            v = (Verdict::NotRun, "NO_RECEIPT".to_string());
        }
        for d in &e.depends_on {
            let dep = gates.get(d).cloned().or_else(|| {
                known
                    .get(d)
                    .filter(|k| k.receipt.is_some() || k.verdict != Verdict::Pass)
                    .map(|k| (k.verdict, k.reason.clone()))
            });
            match dep {
                Some((Verdict::Pass, _)) => {}
                Some((Verdict::Fail, _)) => {
                    v = (Verdict::Fail, format!("DEPENDENCY_FAILED:{d}"));
                    break;
                }
                Some((other, _)) if other != Verdict::NotRun => {
                    v = (other, format!("DEP_BLOCKED:{d}"));
                    break;
                }
                _ => {
                    v = (Verdict::NotRun, format!("UNVERIFIED_DEPENDENCY:{d}"));
                    break;
                }
            }
        }
        gates.insert(e.gate.clone(), v);
    }
    let (verdict, reason) = if let Some(b) = &plan.blocked {
        (Verdict::BlockedOperator, b.clone())
    } else if !plan.unmapped.is_empty() {
        (Verdict::NotRun, "UNMAPPED_PATH".to_string())
    } else if !plan.removed.is_empty() {
        (Verdict::NotRun, "GATE_REMOVED".to_string())
    } else {
        let required: Vec<&Entry> = plan.entries.iter().filter(|e| e.required).collect();
        if required.is_empty() {
            (Verdict::NotRun, "NO_REQUIRED_GATES".to_string())
        } else {
            let mut worst = Verdict::Pass;
            let mut reason = String::new();
            let mut any_fail = false;
            for e in &required {
                let (v, r) = &gates[&e.gate];
                if *v == Verdict::Fail {
                    any_fail = true;
                }
                if v.exit_code() > worst.exit_code() {
                    worst = worse(worst, *v);
                    reason = format!("{}:{}", e.gate, r);
                }
            }
            if any_fail {
                let first = required
                    .iter()
                    .find(|e| gates[&e.gate].0 == Verdict::Fail)
                    .map(|e| format!("{}:{}", e.gate, gates[&e.gate].1))
                    .unwrap_or_default();
                (Verdict::Fail, first)
            } else {
                (worst, reason)
            }
        }
    };
    Judged {
        verdict,
        reason,
        gates,
    }
}

// ---------------------------------------------------------------- crate selection

/// Directories of workspace crate `name` and of every workspace crate that
/// depends on it (transitively), relative to the root, from `cargo metadata`.
pub fn crate_dirs(root: &Path, name: &str) -> Result<Vec<String>, String> {
    let mut cmd = Command::new("cargo");
    cmd.arg("metadata")
        .args(["--format-version", "1", "--no-deps"])
        .current_dir(root);
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run cargo: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let v: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("cargo metadata is not JSON: {e}"))?;
    crate_dirs_from(&v, root, name)
}

/// `crate_dirs` on metadata already read (for tests).
pub fn crate_dirs_from(meta: &Value, root: &Path, name: &str) -> Result<Vec<String>, String> {
    let packages = meta["packages"]
        .as_array()
        .ok_or("cargo metadata has no packages")?;
    let mut dir_of: BTreeMap<String, String> = BTreeMap::new();
    let mut deps_of: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let root_s = root.display().to_string();
    for p in packages {
        let n = p["name"].as_str().unwrap_or("").to_string();
        let mp = p["manifest_path"].as_str().unwrap_or("");
        let dir = Path::new(mp)
            .parent()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        let rel = dir
            .strip_prefix(&format!("{root_s}/"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if dir == root_s {
                    String::new()
                } else {
                    dir.clone()
                }
            });
        dir_of.insert(n.clone(), rel);
        let deps: Vec<String> = p["dependencies"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|d| d["path"].is_string())
                    .filter_map(|d| d["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        deps_of.insert(n, deps);
    }
    if !dir_of.contains_key(name) {
        return Err(format!("no workspace crate named {name}"));
    }
    let mut wanted: BTreeSet<String> = BTreeSet::new();
    let mut todo = vec![name.to_string()];
    while let Some(n) = todo.pop() {
        if !wanted.insert(n.clone()) {
            continue;
        }
        for (other, deps) in &deps_of {
            if deps.contains(&n) {
                todo.push(other.clone());
            }
        }
    }
    let mut dirs: Vec<String> = wanted
        .iter()
        .filter_map(|n| dir_of.get(n).cloned())
        .collect();
    dirs.sort();
    dirs.dedup();
    Ok(dirs)
}

/// Gates whose inputs cover one of `dirs`, or sit under one of them, plus
/// everything they depend on.
pub fn select_for_dirs(graph: &Graph, dirs: &[String]) -> BTreeSet<usize> {
    let seeds: Vec<usize> = (0..graph.len())
        .filter(|&i| {
            graph.node(i).manifest.inputs.iter().any(|input| {
                dirs.iter()
                    .any(|d| covers(input, d) || covers(d, input.trim_end_matches('/')))
            })
        })
        .collect();
    graph.with_dependencies(&seeds)
}

// ---------------------------------------------------------------- rendering

fn short(s: &str) -> String {
    s.chars().take(12).collect()
}

/// The plan as the operator sees it: exactly the sections Required, Cached,
/// Needs Spark, Needs GB10, Advisory, Blocked, then UNMAPPED.
pub fn render(repository: &str, plan: &Plan) -> Vec<String> {
    let mut out = vec![
        "AIEN PLAN".to_string(),
        format!("Repository: {repository}"),
        format!("Base: {}", plan.base),
        format!("Head: {}", plan.head),
        format!("Changed files: {}", plan.changed.len()),
    ];
    for p in &plan.changed {
        out.push(format!("  {p}"));
    }
    if !plan.crumb_paths.is_empty() {
        out.push(format!("Crumb paths: {}", plan.crumb_paths.join(" ")));
    }
    if !plan.pins.is_empty() {
        out.push(format!("Pins: {}", plan.pins.join(" ")));
    }
    out.push(format!("Graph digest: {}", plan.graph_digest));
    for s in Section::ALL {
        out.push(format!("{}:", s.title()));
        let mut any = false;
        for e in plan.entries.iter().filter(|e| e.section == s) {
            any = true;
            let mut line = format!(
                "  {} pool={} exactness={} check={} because={}",
                e.gate,
                e.pool,
                e.exactness.as_str(),
                short(&e.check_id),
                e.because
            );
            if !e.reason.is_empty() {
                line.push(' ');
                line.push_str(&e.reason);
            }
            out.push(line);
        }
        if !any {
            out.push("  (none)".to_string());
        }
    }
    out.push("UNMAPPED:".to_string());
    if plan.unmapped.is_empty() {
        out.push("  (none)".to_string());
    }
    for p in &plan.unmapped {
        out.push(format!("  {p}"));
    }
    if !plan.removed.is_empty() {
        out.push(format!("Removed gates: {}", plan.removed.join(" ")));
    }
    for n in &plan.notes {
        out.push(format!("Note: {n}"));
    }
    let j = judge(plan, &BTreeMap::new());
    let verdict = if let Some(b) = &plan.blocked {
        format!("BLOCKED_OPERATOR {b}")
    } else if !plan.unmapped.is_empty() {
        "NOT_RUN UNMAPPED_PATH".to_string()
    } else if !plan.removed.is_empty() {
        "NOT_RUN GATE_REMOVED".to_string()
    } else if plan.entries.is_empty() {
        "NOTHING_SELECTED".to_string()
    } else {
        let _ = j;
        "RUNNABLE".to_string()
    };
    out.push(format!("Plan verdict: {verdict}"));
    out
}

/// Exit code of `aien-test plan`: 0 runnable (or nothing selected), 2
/// NOT_RUN, 4 BLOCKED_OPERATOR.
pub fn exit_code(plan: &Plan) -> i32 {
    if plan.blocked.is_some() {
        4
    } else if !plan.unmapped.is_empty() || !plan.removed.is_empty() {
        2
    } else {
        0
    }
}

/// The repository name a bundle carries: `--repo`, else the `origin` remote
/// normalised to `github.com/OWNER/REPO`, else `github.com/aien-dev/<dir>`.
pub fn repository_name(root: &Path, explicit: Option<&str>) -> String {
    if let Some(r) = explicit {
        return r.to_string();
    }
    if let Ok(url) = git_text(root, &["remote", "get-url", "origin"]) {
        if let Some(n) = normalise_remote(url.trim()) {
            return n;
        }
    }
    let dir = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".to_string());
    format!("github.com/aien-dev/{dir}")
}

/// `git@github.com:o/r.git`, `https://github.com/o/r`, `ssh://git@github.com/o/r.git`
/// all give `github.com/o/r`. Other hosts keep their host name.
pub fn normalise_remote(url: &str) -> Option<String> {
    let s = url.trim_end_matches('/');
    let s = s.strip_suffix(".git").unwrap_or(s);
    let rest = match s.strip_prefix("git@") {
        Some(r) => r.replacen(':', "/", 1),
        None => ["ssh://git@", "https://", "http://"]
            .iter()
            .find_map(|p| s.strip_prefix(p))?
            .to_string(),
    };
    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 3 {
        return None;
    }
    Some(format!(
        "{}/{}/{}",
        parts[0],
        parts[parts.len() - 2],
        parts[parts.len() - 1]
    ))
}

/// Short helper for the worker and verify: sha256 of a string.
pub fn sha_of(s: &str) -> String {
    sha256_hex(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{git, init_repo, write_file, TempDir};

    fn gate(id: &str, inputs: &[&str], extra: &str) -> String {
        let mut s = format!("gate: {id}\nmanifest_version: 2\nowner: t\nrequires:\n  - host\n");
        if !inputs.is_empty() {
            s.push_str("inputs:\n");
            for i in inputs {
                s.push_str(&format!("  - {i}\n"));
            }
        }
        s.push_str("run:\n  exec: run.sh\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 5s\n");
        s.push_str(extra);
        s
    }

    /// A: inputs a/; B: inputs b/, depends on A; C: inputs c/ and shared/;
    /// A also watches shared/. S (security) watches sec/.
    fn repo() -> TempDir {
        let a = gate("A", &["a/", "shared/"], "");
        let b = gate("B", &["b/"], "depends_on:\n  - A\n");
        let c = gate("C", &["c/", "shared/"], "");
        let s = gate("S", &["sec/"], "");
        init_repo(
            "plan",
            &[
                ("gates/a.gate", a.as_str(), false),
                ("gates/b.gate", b.as_str(), false),
                ("gates/c.gate", c.as_str(), false),
                ("gates/s.gate", s.as_str(), false),
                ("run.sh", "#!/bin/sh\nexit 0\n", true),
                ("a/a.txt", "a\n", false),
                ("b/b.txt", "b\n", false),
                ("c/c.txt", "c\n", false),
                ("sec/s.txt", "s\n", false),
                ("shared/iface.txt", "v1\n", false),
                ("README.md", "hi\n", false),
                (".aien/always", "S\n", false),
                (".aien/unmapped-ok", "*.md\ndocs/\n", false),
            ],
        )
    }

    fn commit(t: &TempDir, files: &[(&str, &str)]) {
        for (rel, content) in files {
            write_file(t.path(), rel, content, false);
        }
        git(t.path(), &["add", "-A"]);
        git(t.path(), &["commit", "-q", "-m", "change"]);
    }

    fn plan_of(t: &TempDir) -> Plan {
        compute(t.path(), &Request::new("HEAD~1", "HEAD")).unwrap()
    }

    fn gates(p: &Plan, s: Section) -> Vec<&str> {
        p.entries
            .iter()
            .filter(|e| e.section == s)
            .map(|e| e.gate.as_str())
            .collect()
    }

    fn all_gates(p: &Plan) -> Vec<&str> {
        p.entries.iter().map(|e| e.gate.as_str()).collect()
    }

    #[test]
    fn m7_a_small_change_selects_only_the_affected_gates_plus_the_always_set() {
        let t = repo();
        commit(&t, &[("b/b.txt", "b2\n")]);
        let p = plan_of(&t);
        // B changed: B and its dependency A run; C is untouched; S is always.
        assert_eq!(all_gates(&p), ["A", "B", "S"]);
        assert!(p.unmapped.is_empty());
        assert!(p.blocked.is_none());
        assert_eq!(gates(&p, Section::Required), ["A", "B", "S"]);
        let b = p.entries.iter().find(|e| e.gate == "B").unwrap();
        assert_eq!(b.because, "b/b.txt");
        let a = p.entries.iter().find(|e| e.gate == "A").unwrap();
        assert_eq!(a.because, "dependency");
        assert!(p.entries.iter().find(|e| e.gate == "S").unwrap().always);
        assert_eq!(exit_code(&p), 0);
        assert_eq!(p.graph_digest.len(), 64);
    }

    #[test]
    fn m8_a_shared_interface_change_selects_every_gate_that_watches_it_and_their_dependents() {
        let t = repo();
        commit(&t, &[("shared/iface.txt", "v2\n")]);
        let p = plan_of(&t);
        // A and C watch shared/; B depends on A so its check id moves.
        assert_eq!(all_gates(&p), ["A", "B", "C", "S"]);
        assert_eq!(
            p.entries.iter().find(|e| e.gate == "B").unwrap().because,
            "dependent"
        );
    }

    #[test]
    fn m12_the_always_set_survives_path_filtering_and_cannot_be_excluded() {
        let t = repo();
        commit(&t, &[("README.md", "changed\n")]);
        let p = plan_of(&t);
        assert_eq!(all_gates(&p), ["S"]);
        assert!(p.unmapped.is_empty(), "*.md is allowed by unmapped-ok");
        // The always set is read from BASE: deleting it at HEAD changes nothing.
        commit(&t, &[(".aien/always", "")]);
        let p = compute(
            t.path(),
            &Request {
                operator: true,
                ..Request::new("HEAD~1", "HEAD")
            },
        )
        .unwrap();
        assert_eq!(all_gates(&p), ["S"]);
        assert!(p.entries[0].always);
    }

    #[test]
    fn an_unmapped_path_gives_not_run_unmapped_path() {
        let t = repo();
        commit(&t, &[("stray/file.txt", "x\n")]);
        let p = plan_of(&t);
        assert_eq!(p.unmapped, ["stray/file.txt"]);
        assert_eq!(exit_code(&p), 2);
        let j = judge(&p, &BTreeMap::new());
        assert_eq!(
            (j.verdict, j.reason.as_str()),
            (Verdict::NotRun, "UNMAPPED_PATH")
        );
        let text = render("github.com/x/y", &p).join("\n");
        assert!(text.contains("UNMAPPED:\n  stray/file.txt"));
        assert!(text.contains("Plan verdict: NOT_RUN UNMAPPED_PATH"));
        // docs/ is allowed, a gate file is always mapped.
        commit(
            &t,
            &[
                ("docs/x.md", "x\n"),
                ("gates/z.gate", &gate("Z", &["z/"], "")),
            ],
        );
        let p = plan_of(&t);
        assert!(p.unmapped.is_empty(), "{:?}", p.unmapped);
        assert_eq!(all_gates(&p), ["S", "Z"]);
    }

    #[test]
    fn pipeline_configuration_changes_block_unless_operator() {
        let t = repo();
        commit(&t, &[(".aien/always", "S\nA\n"), ("a/a.txt", "a2\n")]);
        let p = plan_of(&t);
        assert_eq!(p.blocked.as_deref(), Some("PIPELINE_CONFIG_CHANGED"));
        assert_eq!(p.config_changed, [".aien/always"]);
        assert_eq!(exit_code(&p), 4);
        assert!(p.entries.iter().all(|e| e.section == Section::Blocked));
        let j = judge(&p, &BTreeMap::new());
        assert_eq!(j.verdict, Verdict::BlockedOperator);
        let p = compute(
            t.path(),
            &Request {
                operator: true,
                ..Request::new("HEAD~1", "HEAD")
            },
        )
        .unwrap();
        assert!(p.blocked.is_none());
        assert_eq!(gates(&p, Section::Required), ["A", "B", "S"]);
        for cfg in [
            "tests/gates/x",
            ".github/workflows/ci.yml",
            ".forgejo/w.yml",
            "tools/aien-test/src/x.rs",
        ] {
            assert!(under_config(cfg), "{cfg}");
        }
        assert!(!under_config("tools/other/x"));
    }

    #[test]
    fn a_gate_deleted_at_head_is_gate_removed() {
        let t = repo();
        std::fs::remove_file(t.path().join("gates/c.gate")).unwrap();
        commit(&t, &[("a/a.txt", "a2\n")]);
        let p = plan_of(&t);
        assert_eq!(p.removed, ["C"]);
        assert!(p.unmapped.is_empty());
        assert_eq!(exit_code(&p), 2);
        let j = judge(&p, &BTreeMap::new());
        assert_eq!(
            (j.verdict, j.reason.as_str()),
            (Verdict::NotRun, "GATE_REMOVED")
        );
    }

    #[test]
    fn a_changed_pin_selects_every_gate_listing_it() {
        let a = gate("A", &["a/", "deps/x.lock"], "");
        let b = gate("B", &["b/"], "");
        let t = init_repo(
            "plan-pin",
            &[
                ("gates/a.gate", a.as_str(), false),
                ("gates/b.gate", b.as_str(), false),
                ("run.sh", "#!/bin/sh\n", true),
                ("deps/x.lock", &format!("{}\n", "a".repeat(40)), false),
            ],
        );
        commit(&t, &[("deps/x.lock", &format!("{}\n", "b".repeat(40)))]);
        let p = plan_of(&t);
        assert_eq!(p.pins, ["deps/x.lock"]);
        assert_eq!(all_gates(&p), ["A"]);
        assert!(is_pin(
            "x.lock",
            b"0123456789abcdef0123456789abcdef01234567\n"
        ));
        assert!(!is_pin("Cargo.lock", b"[[package]]\n"));
        assert!(!is_pin(
            "x.txt",
            b"0123456789abcdef0123456789abcdef01234567\n"
        ));
    }

    #[test]
    fn d3_implied_inputs_select_a_tests_dir_gate() {
        let g = gate("K", &[], "");
        let t = init_repo(
            "plan-d3",
            &[
                ("tests/kern/boot.gate", g.as_str(), false),
                ("run.sh", "#!/bin/sh\n", true),
                ("src/kern/boot.rs", "fn a(){}\n", false),
                ("tests/kern/boot_test.rs", "fn t(){}\n", false),
                ("src/kern/other.rs", "x\n", false),
            ],
        );
        let rows = tree_rows(t.path(), "HEAD").unwrap();
        let mut implied = implied_inputs("tests/kern/boot.gate", &rows);
        implied.sort();
        assert_eq!(implied, ["src/kern/boot.rs", "tests/kern/boot_test.rs"]);
        commit(&t, &[("src/kern/boot.rs", "fn b(){}\n")]);
        let p = plan_of(&t);
        assert_eq!(all_gates(&p), ["K"]);
        assert!(p.unmapped.is_empty());
        commit(&t, &[("src/kern/other.rs", "y\n")]);
        let p = plan_of(&t);
        assert!(p.entries.is_empty());
        assert_eq!(p.unmapped, ["src/kern/other.rs"]);
    }

    #[test]
    fn a_manifest_edit_selects_its_own_gate() {
        let t = repo();
        commit(
            &t,
            &[(
                "gates/c.gate",
                &gate("C", &["c/", "shared/"], "cache: never\n"),
            )],
        );
        let p = plan_of(&t);
        assert_eq!(all_gates(&p), ["C", "S"]);
    }

    #[test]
    fn crumb_related_paths_count_as_changed() {
        let t = repo();
        commit(
            &t,
            &[("b/.crumb", "{\"dir_name\":\"b\",\"related\":[\"c/\"]}\n")],
        );
        commit(&t, &[("b/b.txt", "b3\n")]);
        let p = plan_of(&t);
        assert_eq!(p.crumb_paths, ["c"]);
        assert_eq!(all_gates(&p), ["A", "B", "C", "S"]);
        // A crumb file itself is always mapped.
        assert!(is_crumb_file("b/.crumb") && is_crumb_file(".crumb.local"));
        // A crumb without related/links adds nothing (today's schema).
        commit(
            &t,
            &[
                ("b/.crumb", "{\"dir_name\":\"b\",\"purpose\":\"x\"}\n"),
                ("b/b.txt", "b4\n"),
            ],
        );
        let p = plan_of(&t);
        assert!(p.crumb_paths.is_empty());
        assert_eq!(all_gates(&p), ["A", "B", "S"]);
    }

    #[test]
    fn uncommitted_changes_are_listed_only_when_asked() {
        let t = repo();
        commit(&t, &[("a/a.txt", "a2\n")]);
        write_file(t.path(), "c/new.txt", "n\n", false);
        write_file(t.path(), "b/b.txt", "edit\n", false);
        let mut dirty = uncommitted(t.path()).unwrap();
        dirty.sort();
        assert_eq!(dirty, ["b/b.txt", "c/new.txt"]);
        let p = plan_of(&t);
        assert_eq!(all_gates(&p), ["A", "B", "S"]);
        let p = compute(
            t.path(),
            &Request {
                dirty,
                ..Request::new("HEAD~1", "HEAD")
            },
        )
        .unwrap();
        assert_eq!(all_gates(&p), ["A", "B", "C", "S"]);
    }

    #[test]
    fn the_plan_is_the_same_for_a_commit_that_is_not_checked_out() {
        let t = repo();
        commit(&t, &[("b/b.txt", "b2\n")]);
        let head = resolve(t.path(), "HEAD").unwrap();
        let base = resolve(t.path(), "HEAD~1").unwrap();
        let p1 = compute(t.path(), &Request::new(&base, &head)).unwrap();
        commit(
            &t,
            &[
                ("c/c.txt", "c2\n"),
                ("gates/a.gate", &gate("A", &["a/", "shared/", "x/"], "")),
            ],
        );
        let p2 = compute(t.path(), &Request::new(&base, &head)).unwrap();
        assert_eq!(p1.graph_digest, p2.graph_digest);
        assert_eq!(all_gates(&p1), all_gates(&p2));
        assert_eq!(p1.changed, p2.changed);
        // The newest commit's plan differs (A's manifest changed).
        let p3 = compute(t.path(), &Request::new(&base, "HEAD")).unwrap();
        assert_ne!(p3.graph_digest, p1.graph_digest);
    }

    #[test]
    fn sections_follow_exactness_pool_and_operator() {
        let adv = gate("ADV", &["adv/"], "exactness: ADVISORY\n");
        let gpu = gate("GPU", &["gpu/"], "").replace("  - host\n", "  - gb10\n");
        let q = gate("Q", &["q/"], "").replace("  - host\n", "  - qemu\n");
        let op = gate("OP", &["op/"], "").replace("  - host\n", "  - host\n  - operator\n");
        let t = init_repo(
            "plan-sections",
            &[
                ("gates/adv.gate", adv.as_str(), false),
                ("gates/gpu.gate", gpu.as_str(), false),
                ("gates/q.gate", q.as_str(), false),
                ("gates/op.gate", op.as_str(), false),
                ("run.sh", "#!/bin/sh\n", true),
                ("adv/x", "1\n", false),
                ("gpu/x", "1\n", false),
                ("q/x", "1\n", false),
                ("op/x", "1\n", false),
            ],
        );
        commit(
            &t,
            &[
                ("adv/x", "2\n"),
                ("gpu/x", "2\n"),
                ("q/x", "2\n"),
                ("op/x", "2\n"),
            ],
        );
        let mut req = Request::new("HEAD~1", "HEAD");
        req.cached.insert("Q".into());
        let p = compute(t.path(), &req).unwrap();
        assert_eq!(gates(&p, Section::Advisory), ["ADV"]);
        assert_eq!(gates(&p, Section::NeedsGb10), ["GPU"]);
        assert_eq!(
            gates(&p, Section::NeedsSpark),
            ["Q"],
            "qemu needs the Spark even when cached"
        );
        assert_eq!(gates(&p, Section::Blocked), ["OP"]);
        assert_eq!(
            p.entries.iter().find(|e| e.gate == "OP").unwrap().reason,
            "OPERATOR_STEP_PENDING"
        );
        assert!(!p.entries.iter().find(|e| e.gate == "ADV").unwrap().required);
        let text = render("r", &p).join("\n");
        for title in [
            "Required:",
            "Cached:",
            "Needs Spark:",
            "Needs GB10:",
            "Advisory:",
            "Blocked:",
            "UNMAPPED:",
        ] {
            assert!(text.contains(&format!("\n{title}\n")), "{title}");
        }
    }

    #[test]
    fn operator_gate_runs_under_operator_and_is_blocked_otherwise() {
        // sc #614: a `requires: operator` gate is OPERATOR_STEP_PENDING for an
        // automatic run and runs like any other gate when the operator
        // started the run by hand with `--operator` (ADR 0033 Decision 2).
        let op = gate("OP", &["op/"], "").replace("  - host\n", "  - host\n  - operator\n");
        let t = init_repo(
            "plan-operator-gate",
            &[
                ("gates/op.gate", op.as_str(), false),
                ("run.sh", "#!/bin/sh\n", true),
                ("op/x", "1\n", false),
            ],
        );
        commit(&t, &[("op/x", "2\n")]);
        let pass = |receipt: bool| {
            let mut known = BTreeMap::new();
            known.insert(
                "OP".to_string(),
                Known {
                    verdict: Verdict::Pass,
                    reason: "PASS".into(),
                    receipt: receipt.then(|| "d".repeat(64)),
                },
            );
            known
        };

        // Automatic run: Blocked, and a PASS receipt does not unblock it.
        let p = compute(t.path(), &Request::new("HEAD~1", "HEAD")).unwrap();
        assert_eq!(gates(&p, Section::Blocked), ["OP"]);
        assert_eq!(
            exit_code(&p),
            0,
            "a pending operator step is not a config block"
        );
        let j = judge(&p, &pass(true));
        assert_eq!(
            (j.verdict, j.reason.as_str()),
            (Verdict::BlockedOperator, "OP:OPERATOR_STEP_PENDING")
        );

        // Operator run: the gate is Required, runs, and its receipt counts.
        let p = compute(
            t.path(),
            &Request {
                operator: true,
                ..Request::new("HEAD~1", "HEAD")
            },
        )
        .unwrap();
        assert_eq!(gates(&p, Section::Required), ["OP"]);
        assert!(gates(&p, Section::Blocked).is_empty());
        let e = p.entries.iter().find(|e| e.gate == "OP").unwrap();
        assert!(e.reason.is_empty(), "reason {}", e.reason);
        assert!(
            p.notes
                .iter()
                .any(|n| n.contains("OP") && n.contains("--operator")),
            "{:?}",
            p.notes
        );
        let j = judge(&p, &pass(true));
        assert_eq!((j.verdict, j.reason.as_str()), (Verdict::Pass, ""));
        // Without a receipt nothing is promoted.
        let j = judge(&p, &pass(false));
        assert_eq!(j.verdict, Verdict::NotRun);
        let j = judge(&p, &BTreeMap::new());
        assert_eq!(j.verdict, Verdict::NotRun);
    }

    #[test]
    fn cached_section_needs_the_probe() {
        let t = repo();
        commit(&t, &[("a/a.txt", "a2\n")]);
        let mut req = Request::new("HEAD~1", "HEAD");
        req.cached.insert("A".into());
        let p = compute(t.path(), &req).unwrap();
        assert_eq!(gates(&p, Section::Cached), ["A"]);
        assert_eq!(gates(&p, Section::Required), ["B", "S"]);
    }

    fn known(v: Verdict, receipt: bool) -> Known {
        Known {
            verdict: v,
            reason: String::new(),
            receipt: receipt.then(|| "r".repeat(64)),
        }
    }

    #[test]
    fn t5_missing_dependency_receipt_is_unverified_dependency() {
        let t = repo();
        commit(&t, &[("b/b.txt", "b2\n")]);
        let p = plan_of(&t);
        let mut k = BTreeMap::new();
        k.insert("B".to_string(), known(Verdict::Pass, true));
        k.insert("S".to_string(), known(Verdict::Pass, true));
        let j = judge(&p, &k);
        assert_eq!(j.gates["A"].0, Verdict::NotRun);
        assert_eq!(
            j.gates["B"],
            (Verdict::NotRun, "UNVERIFIED_DEPENDENCY:A".to_string())
        );
        assert_eq!(j.verdict, Verdict::NotRun);
        // A PASS without a receipt is not evidence either.
        k.insert("A".to_string(), known(Verdict::Pass, false));
        let j = judge(&p, &k);
        assert_eq!(j.gates["A"], (Verdict::NotRun, "NO_RECEIPT".to_string()));
        assert_eq!(j.gates["B"].1, "UNVERIFIED_DEPENDENCY:A");
    }

    #[test]
    fn t6_a_failed_dependency_propagates_as_dependency_failed() {
        let t = repo();
        commit(&t, &[("b/b.txt", "b2\n")]);
        let p = plan_of(&t);
        let mut k = BTreeMap::new();
        k.insert("A".to_string(), known(Verdict::Fail, true));
        k.insert("B".to_string(), known(Verdict::Pass, true));
        k.insert("S".to_string(), known(Verdict::Pass, true));
        let j = judge(&p, &k);
        assert_eq!(
            j.gates["B"],
            (Verdict::Fail, "DEPENDENCY_FAILED:A".to_string())
        );
        assert_eq!(j.verdict, Verdict::Fail);
        assert!(j.reason.starts_with("A:"));
    }

    #[test]
    fn t7_not_run_is_never_promoted_and_all_pass_is_pass() {
        let t = repo();
        commit(&t, &[("b/b.txt", "b2\n")]);
        let p = plan_of(&t);
        let mut k = BTreeMap::new();
        k.insert("A".to_string(), known(Verdict::Pass, true));
        k.insert("B".to_string(), known(Verdict::Pass, true));
        k.insert("S".to_string(), known(Verdict::Pass, true));
        let j = judge(&p, &k);
        assert_eq!(j.verdict, Verdict::Pass);
        assert!(j.gates.values().all(|(v, _)| *v == Verdict::Pass));
        // One required gate NOT_RUN: the commit is NOT_RUN however the rest did.
        k.insert("S".to_string(), known(Verdict::NotRun, true));
        let j = judge(&p, &k);
        assert_eq!(j.verdict, Verdict::NotRun);
        assert_eq!(j.reason, "S:");
        // A security gate from the always set cannot be left out: no entry
        // for S means no receipt, which is NOT_RUN.
        k.remove("S");
        let j = judge(&p, &k);
        assert_eq!(j.gates["S"], (Verdict::NotRun, "NO_RECEIPT".to_string()));
        assert_eq!(j.verdict, Verdict::NotRun);
        // A blocked required gate blocks the commit.
        k.insert("S".to_string(), known(Verdict::BlockedHardware, true));
        assert_eq!(judge(&p, &k).verdict, Verdict::BlockedHardware);
    }

    #[test]
    fn advisory_gates_never_move_the_commit_verdict() {
        let adv = gate("ADV", &["a/"], "exactness: ADVISORY\n");
        let a = gate("A", &["a/"], "");
        let t = init_repo(
            "plan-adv",
            &[
                ("gates/adv.gate", adv.as_str(), false),
                ("gates/a.gate", a.as_str(), false),
                ("run.sh", "#!/bin/sh\n", true),
                ("a/x", "1\n", false),
            ],
        );
        commit(&t, &[("a/x", "2\n")]);
        let p = plan_of(&t);
        let mut k = BTreeMap::new();
        k.insert("A".to_string(), known(Verdict::Pass, true));
        k.insert("ADV".to_string(), known(Verdict::Fail, true));
        let j = judge(&p, &k);
        assert_eq!(j.verdict, Verdict::Pass);
        assert_eq!(j.gates["ADV"].0, Verdict::Fail);
    }

    #[test]
    fn glob_and_covers() {
        assert!(glob_match("*.md", "docs/a.md"));
        assert!(glob_match("docs/", "docs/x/y"));
        assert!(!glob_match("docs/", "docsx/y"));
        assert!(glob_match("exact.txt", "exact.txt"));
        assert!(!glob_match("exact.txt", "exact.txt2"));
        assert!(glob_match("a*b*c", "aXXbYc"));
        assert!(covers("a", "a/b"));
        assert!(covers("a/", "a"));
        assert!(!covers("a", "ab/c"));
        assert!(covers("", "anything"));
    }

    #[test]
    fn remote_names_normalise() {
        for u in [
            "git@github.com:aien-dev/x.git",
            "https://github.com/aien-dev/x",
            "https://github.com/aien-dev/x.git/",
            "ssh://git@github.com/aien-dev/x.git",
        ] {
            assert_eq!(
                normalise_remote(u).as_deref(),
                Some("github.com/aien-dev/x"),
                "{u}"
            );
        }
        assert_eq!(normalise_remote("/local/path"), None);
        let t = repo();
        assert_eq!(
            repository_name(t.path(), Some("github.com/a/b")),
            "github.com/a/b"
        );
        assert!(repository_name(t.path(), None).starts_with("github.com/aien-dev/"));
    }

    #[test]
    fn crate_dirs_follow_reverse_dependents() {
        let meta = json!({
            "packages": [
                {"name": "core", "manifest_path": "/r/crates/core/Cargo.toml", "dependencies": []},
                {"name": "mid", "manifest_path": "/r/crates/mid/Cargo.toml",
                 "dependencies": [{"name": "core", "path": "/r/crates/core"}]},
                {"name": "top", "manifest_path": "/r/tools/top/Cargo.toml",
                 "dependencies": [{"name": "mid", "path": "/r/crates/mid"}, {"name": "serde", "path": null}]},
                {"name": "other", "manifest_path": "/r/crates/other/Cargo.toml",
                 "dependencies": [{"name": "core", "path": null}]}
            ]
        });
        let root = Path::new("/r");
        assert_eq!(
            crate_dirs_from(&meta, root, "core").unwrap(),
            ["crates/core", "crates/mid", "tools/top"]
        );
        assert_eq!(crate_dirs_from(&meta, root, "top").unwrap(), ["tools/top"]);
        assert!(crate_dirs_from(&meta, root, "nope").is_err());
        let t = repo();
        let g = graph::load(t.path()).unwrap();
        let picked = select_for_dirs(&g, &["b".to_string()]);
        let names: Vec<&str> = picked.iter().map(|&i| g.node(i).id.as_str()).collect();
        assert_eq!(names, ["A", "B"]);
        // A crate directory above an input also selects the gate.
        let picked = select_for_dirs(&g, &[String::new()]);
        assert_eq!(picked.len(), 4);
    }

    #[test]
    fn resolve_and_rows() {
        let t = repo();
        let h = resolve(t.path(), "HEAD").unwrap();
        assert_eq!(h.len(), 40);
        assert!(resolve(t.path(), "no-such-ref").is_err());
        assert_eq!(resolve_first(t.path(), &["nope", "HEAD"]).unwrap(), h);
        let rows = tree_rows(t.path(), &h).unwrap();
        assert!(rows.contains_key("gates/a.gate"));
        assert_eq!(rows["a/a.txt"].len(), 40);
        assert_eq!(show(t.path(), &h, "a/a.txt").unwrap(), b"a\n");
        assert!(show(t.path(), &h, "missing").is_none());
        assert_eq!(always_set(t.path(), &h), ["S"]);
        assert_eq!(unmapped_ok(t.path(), &h), ["*.md", "docs/"]);
    }
}
