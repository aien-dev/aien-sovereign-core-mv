//! The trusted worker (ADR 0033 Decision 9; local pipeline contracts C1 to
//! C7) and the state directory every local-CI command shares.
//!
//! State lives under `~/.local/state/aien-local-ci/`:
//!
//! ```text
//! evidence/<repo>/<sha>/   receipts and blobs of that commit's run
//! bundles/<repo>/<sha>/bundle.json   the signed VerificationBundleV1
//! trigger/<repo>.<sha>     a file whose presence asks for a run
//! outbox/<repo>.<sha>.json a GitHub status that could not be posted yet
//! journal.jsonl            one line per finished head
//! worker.heartbeat         what the worker is doing right now
//! ```
//!
//! For one head: fetch into the trusted clone `<clones>/<repo>` (its checkout
//! is never touched), add a detached worktree `<wt-root>/<repo>/<sha>`,
//! `crumb claim` it, plan, run the selected gates with the sandbox and the
//! pools, write receipts, build and sign the bundle with `~/.aien/verify.key`,
//! write it, journal it, `crumb close`, remove the worktree. With `--github`
//! the commit status `AIEN Verification (Spark)` is posted: success only on
//! PASS, failure on FAIL, error otherwise. When GitHub cannot be reached the
//! status goes to the outbox, the journal says `publication: pending`, and the
//! next round of a loop started with `--github` flushes it (a loop without
//! `--github` leaves it queued). GitHub being down is never a local
//! failure.

use crate::bundle;
use crate::evidence::{canonical, canonical_digest, sha256_hex, utc_string};
use crate::graph;
use crate::plan::{self, Known, Section};
use crate::process;
use crate::resources::{Hardware, JobsSpec};
use crate::runner::{self, CacheNote, Options};
use crate::sandbox::Sandbox;
use crate::verdict::Verdict;
use crate::verify;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const CONTEXT: &str = "AIEN Verification (Spark)";
pub const EXECUTOR_CLASS: &str = "spark";
/// Where INFRA keeps the per-repository cargo target dirs (A2).
pub const TARGET_ROOT: &str = "/var/lib/aien-run/target";
/// Where INFRA's `wt.sh` keeps the bare mirrors its worktrees point into.
pub const MIRROR_ROOT: &str = "/var/lib/aien-run/mirror";

/// A repository directory key the worker accepts: letters, digits, `.`, `_`
/// and `-`, not starting with a dot (so never `.` or `..`).
pub fn valid_repo_key(repo: &str) -> bool {
    !repo.is_empty()
        && repo.len() <= 100
        && !repo.starts_with('.')
        && repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// A full lowercase 40-hex commit id.
pub fn valid_sha(sha: &str) -> bool {
    sha.len() == 40
        && sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The signing key must be a regular file only its owner can read or write
/// (mode 0600 or stricter): a key the `aien-run` user could read would let a
/// gate sign its own bundle.
pub fn check_key_private(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let meta = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat signing key {}: {e}", path.display()))?;
    if !meta.file_type().is_file() {
        return Err(format!(
            "signing key {} is not a regular file",
            path.display()
        ));
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(format!(
            "signing key {} has mode {mode:o}; it must be 0600 (no group or other access)",
            path.display()
        ));
    }
    Ok(())
}

/// The local-CI state directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDir {
    pub root: PathBuf,
}

impl StateDir {
    pub fn new(root: &Path) -> StateDir {
        StateDir {
            root: root.to_path_buf(),
        }
    }

    /// `~/.local/state/aien-local-ci`; None without HOME.
    pub fn default_dir() -> Option<StateDir> {
        let home = std::env::var_os("HOME")?;
        Some(StateDir::new(
            &PathBuf::from(home).join(".local/state/aien-local-ci"),
        ))
    }

    /// The directory key of a repository name: its last path segment.
    pub fn repo_key(repository: &str) -> String {
        repository
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(repository)
            .to_string()
    }

    pub fn evidence(&self, repo: &str, sha: &str) -> PathBuf {
        self.root.join("evidence").join(repo).join(sha)
    }

    pub fn bundle_path(&self, repo: &str, sha: &str) -> PathBuf {
        self.root
            .join("bundles")
            .join(repo)
            .join(sha)
            .join("bundle.json")
    }

    pub fn trigger_dir(&self) -> PathBuf {
        self.root.join("trigger")
    }

    pub fn outbox_dir(&self) -> PathBuf {
        self.root.join("outbox")
    }

    pub fn journal(&self) -> PathBuf {
        self.root.join("journal.jsonl")
    }

    pub fn heartbeat(&self) -> PathBuf {
        self.root.join("worker.heartbeat")
    }

    /// Append one JSON line to the journal.
    pub fn journal_append(&self, entry: &Value) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|e| format!("cannot create state dir: {e}"))?;
        let mut f = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.journal())
            .map_err(|e| format!("cannot open journal: {e}"))?;
        let mut line = serde_json::to_string(entry).map_err(|e| e.to_string())?;
        line.push('\n');
        f.write_all(line.as_bytes())
            .map_err(|e| format!("cannot write journal: {e}"))
    }

    /// Overwrite the heartbeat file (atomic rename).
    pub fn beat(&self, repo: &str, sha: &str, phase: &str) {
        let v = json!({
            "utc": utc_string(now_secs()),
            "pid": std::process::id(),
            "repo": repo,
            "sha": sha,
            "phase": phase,
        });
        let _ = fs::create_dir_all(&self.root);
        let tmp = self.root.join("worker.heartbeat.tmp");
        if fs::write(&tmp, serde_json::to_string(&v).unwrap_or_default()).is_ok() {
            let _ = fs::rename(&tmp, self.heartbeat());
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// How the worker is set up.
#[derive(Debug, Clone)]
pub struct Config {
    pub state: StateDir,
    pub key: PathBuf,
    /// Where the trusted clones live: `<clones>/<repo>`.
    pub clones: PathBuf,
    pub wt_root: PathBuf,
    pub sandbox: Sandbox,
    pub github: bool,
    pub poll: Duration,
    pub repos: Vec<String>,
    /// The ref the plan is based on (`origin/main`); a head that *is* that
    /// ref is planned against its first parent.
    pub base_ref: String,
    pub operator: bool,
    /// The crumb command (`crumb`); None skips crumb claim/close.
    pub crumb: Option<String>,
    /// The GitHub CLI (`gh`), looked up on PATH.
    pub gh: String,
    pub hardware_identity: String,
    /// Allow `Sandbox::None` (development only: gates then run as the
    /// worker user, who can read the signing key). Never with `github`.
    pub allow_unsandboxed: bool,
    /// Per-repository cargo target directories by trust class
    /// (`<target_root>/<repo>/main|pr`), used under `Sandbox::User`.
    pub target_root: PathBuf,
    /// INFRA's worktree tool (`tools/aien-local-ci/wt.sh`). Under
    /// `Sandbox::User` every checkout is made and removed with it; without it
    /// the worker refuses to run.
    pub wt_tool: Option<PathBuf>,
    /// The mirrors `wt_tool` keeps (`<mirror_root>/<repo>.git`).
    pub mirror_root: PathBuf,
    /// The operator's sibling map (`tools/aien-local-ci/siblings`): which
    /// read-only sibling checkouts a repository's gates reach as `../<dir>`.
    /// None: no siblings.
    pub siblings: Option<PathBuf>,
    /// `--jobs host=N,qemu=M` for the runs this worker starts (None: the
    /// machine defaults). An operator run of a repository whose gates share
    /// one checkout uses `--jobs host=1`.
    pub jobs: JobsSpec,
    /// `--no-cache`: run every selected gate, reuse no receipt.
    pub no_cache: bool,
}

impl Config {
    pub fn new(state: StateDir, key: &Path, clones: &Path, wt_root: &Path) -> Config {
        Config {
            state,
            key: key.to_path_buf(),
            clones: clones.to_path_buf(),
            wt_root: wt_root.to_path_buf(),
            sandbox: Sandbox::None,
            github: false,
            poll: Duration::from_secs(30),
            repos: Vec::new(),
            base_ref: "origin/main".to_string(),
            operator: false,
            crumb: Some("crumb".to_string()),
            gh: "gh".to_string(),
            hardware_identity: String::new(),
            allow_unsandboxed: false,
            target_root: PathBuf::from(TARGET_ROOT),
            wt_tool: None,
            mirror_root: PathBuf::from(MIRROR_ROOT),
            siblings: None,
            jobs: JobsSpec::default(),
            no_cache: false,
        }
    }

    /// The sandbox rule: `Sandbox::User` always; `Sandbox::None` only when
    /// `allow_unsandboxed` (development) and never when serving GitHub heads.
    pub fn check_sandbox_policy_for(&self, github: bool) -> Result<(), String> {
        if self.sandbox == Sandbox::User {
            return Ok(());
        }
        if github {
            return Err("--sandbox none is refused with --github: pull request code must run as the aien-run user".to_string());
        }
        if !self.allow_unsandboxed {
            return Err("--sandbox none runs gates as the worker user, who can read the signing key; it needs --dev-unsandboxed (development only)".to_string());
        }
        Ok(())
    }
}

/// What one head produced.
#[derive(Debug, Clone)]
pub struct Summary {
    pub repo: String,
    pub sha: String,
    pub repository: String,
    pub verdict: Verdict,
    pub reason: String,
    pub bundle_verdict: String,
    pub bundle_path: PathBuf,
    pub bundle_digest: String,
    pub gates: Vec<(String, String, Verdict)>,
    pub publication: String,
    /// What the independent check (`verify`) said about the bundle.
    pub ready: bool,
    pub notes: Vec<String>,
}

fn run_cmd(cwd: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);
    // git and crumb (which runs git) may open a trusted wt.sh worktree.
    process::apply_git_trust(&mut cmd);
    if program == "git" {
        cmd.env("GIT_OPTIONAL_LOCKS", "0");
    }
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run {program}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    run_cmd(cwd, "git", args)
}

/// One sibling checkout a repository's gates reach as `../<dir>`: a
/// read-only checkout of `<clones>/<source>` at `refs/remotes/origin/<git_ref>`
/// as that trusted clone has it when the run starts (never fetched for the
/// run, never the network).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sibling {
    pub dir: String,
    pub source: String,
    pub git_ref: String,
}

/// A branch name the sibling map accepts: letters, digits, `.`, `_`, `-` and
/// `/` between non-empty parts that do not start with a dot (so no `..`, no
/// leading `-`, nothing git would read as an option or a path escape).
pub fn valid_ref(r: &str) -> bool {
    !r.is_empty()
        && r.len() <= 200
        && !r.starts_with('-')
        && r.split('/').all(|p| !p.is_empty() && !p.starts_with('.'))
        && r.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
}

/// Parse the sibling map (`<repo> <dir> <source-repo> <ref>` per line, `#`
/// comments) and return the siblings of `repo`. Every line is checked, not
/// only `repo`'s: a malformed map is refused as a whole.
pub fn parse_siblings(text: &str, repo: &str) -> Result<Vec<Sibling>, String> {
    let mut out: Vec<Sibling> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let bad = |why: &str| format!("sibling map line {}: {why}: {raw:?}", n + 1);
        if f.len() != 4 {
            return Err(bad("needs <repo> <dir> <source-repo> <ref>"));
        }
        if !valid_repo_key(f[0]) || !valid_repo_key(f[1]) || !valid_repo_key(f[2]) {
            return Err(bad("repo, dir and source must be plain names"));
        }
        if !valid_ref(f[3]) {
            return Err(bad("bad ref"));
        }
        if f[1] == f[0] {
            return Err(bad("a sibling cannot take the checkout's own name"));
        }
        if !seen.insert((f[0], f[1])) {
            return Err(bad("the same dir twice for one repository"));
        }
        if f[0] != repo {
            continue;
        }
        out.push(Sibling {
            dir: f[1].to_string(),
            source: f[2].to_string(),
            git_ref: f[3].to_string(),
        });
    }
    Ok(out)
}

/// The siblings of `repo` from `cfg.siblings` (none without a map). A map
/// that cannot be read or parsed refuses the run: a gate must never run
/// against a half-provided neighbourhood without anyone noticing.
fn siblings_for(cfg: &Config, repo: &str) -> Result<Vec<Sibling>, String> {
    let Some(path) = &cfg.siblings else {
        return Ok(Vec::new());
    };
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read the sibling map {}: {e}", path.display()))?;
    parse_siblings(&text, repo)
}

/// One commit a repository's gates check out from a sibling (`pin` in its
/// `tests/gates/run.sh`): the sibling's source repository and the commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub source: String,
    pub sha: String,
}

/// Parse the pin list (`tools/aien-local-ci/pins` beside the sibling map;
/// `<repo> <source-repo> <sha40>` per line, `#` comments) and return the
/// pins of `repo`. Every line is checked, as in the sibling map.
pub fn parse_pins(text: &str, repo: &str) -> Result<Vec<Pin>, String> {
    let mut out = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let bad = |why: &str| format!("pin list line {}: {why}: {raw:?}", n + 1);
        if f.len() != 3 {
            return Err(bad("needs <repo> <source-repo> <sha40>"));
        }
        if !valid_repo_key(f[0]) || !valid_repo_key(f[1]) {
            return Err(bad("repo and source must be plain names"));
        }
        if !valid_sha(f[2]) {
            return Err(bad("the commit must be a full lowercase sha"));
        }
        if f[0] == repo {
            out.push(Pin {
                source: f[1].to_string(),
                sha: f[2].to_string(),
            });
        }
    }
    Ok(out)
}

/// The pins of `repo` from the `pins` file beside the sibling map (none
/// without a map or without the file). A file that exists but cannot be read
/// or parsed refuses the run, like a broken sibling map.
fn pins_for(cfg: &Config, repo: &str) -> Result<Vec<Pin>, String> {
    let Some(map) = &cfg.siblings else {
        return Ok(Vec::new());
    };
    let path = map.with_file_name("pins");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("cannot read the pin list {}: {e}", path.display()))?;
    parse_pins(&text, repo)
}

/// Make every pinned commit of a sibling present in that sibling's trusted
/// clone before the siblings are made, so the sandbox (which has no network)
/// finds it. Runs on the trusted side at provisioning time only: a missing
/// commit is fetched by its sha from the clone's own `origin`, then held by
/// `refs/remotes/aien-pins/<sha>` so `wt.sh sync` copies it into the mirror and
/// `git gc` keeps it. A pin that cannot be made present is a note; the gate
/// that needs it then fails on the missing commit, never PASS.
pub fn ensure_pins(clones: &Path, pins: &[Pin], sibs: &[Sibling], notes: &mut Vec<String>) {
    for p in pins {
        if !sibs.iter().any(|s| s.source == p.source) {
            notes.push(format!(
                "pin {}@{}: {} is not a sibling of this repository; not provided",
                p.source, p.sha, p.source
            ));
            continue;
        }
        let src = clones.join(&p.source);
        let want = format!("{}^{{commit}}", p.sha);
        let present = |src: &Path| git(src, &["cat-file", "-e", &want]).is_ok();
        if !present(&src) {
            match git(&src, &["fetch", "--quiet", "origin", &p.sha]) {
                Ok(_) if present(&src) => {
                    notes.push(format!("pin {}@{}: fetched from origin", p.source, p.sha))
                }
                Ok(_) => {
                    notes.push(format!(
                        "pin {}@{}: fetch gave no such commit; not provided",
                        p.source, p.sha
                    ));
                    continue;
                }
                Err(e) => {
                    notes.push(format!("pin {}@{}: not provided: {e}", p.source, p.sha));
                    continue;
                }
            }
        }
        let r = format!("refs/remotes/aien-pins/{}", p.sha);
        if let Err(e) = git(&src, &["update-ref", &r, &p.sha]) {
            notes.push(format!(
                "pin {}@{}: cannot hold it in {r}: {e}",
                p.source, p.sha
            ));
        }
    }
}

/// Where a head runs: `<wt_root>/<repo>/<sha>/` holds the checkout under test
/// at `<repo>/` and its siblings beside it, so `../<dir>` in the checkout is
/// the sibling. `nested` is false only for an older `wt.sh` that still makes
/// the checkout at the run directory itself (then no siblings are possible).
struct Checkout {
    clone: PathBuf,
    wt: PathBuf,
    run_dir: PathBuf,
    nested: bool,
}

/// Make the read-only sibling checkouts of one head, each from its trusted
/// clone at the clone's `origin/<ref>`. A sibling that cannot be made is a
/// note (its gates then fail on the missing directory, never PASS); every
/// sibling made is journaled with its commit. Returns what to remove later
/// under `Sandbox::None`, and the receipt `pins` (ADR 0028: `{name, commit}`
/// sorted by `name`): each sibling made, named by its `dir`, and each operator
/// pin, named `pin:<source>`. They enter `core.pins` and `cache.key`, so a
/// result is reused only while every sibling is at the same commit.
fn provision_siblings(
    cfg: &Config,
    repo: &str,
    sha: &str,
    co: &Checkout,
    sibs: &[Sibling],
    pins: &[Pin],
    notes: &mut Vec<String>,
) -> (Vec<(PathBuf, PathBuf)>, Value) {
    let mut made = Vec::new();
    let mut pin_entries: Vec<(String, String)> = Vec::new();
    if !sibs.is_empty() && !co.nested {
        notes.push(format!(
            "siblings not provided: the worktree tool makes the old flat layout (update {})",
            cfg.wt_tool
                .as_ref()
                .map(|t| t.display().to_string())
                .unwrap_or_default()
        ));
        return (made, json!([]));
    }
    for s in sibs {
        let src = cfg.clones.join(&s.source);
        let want = format!("refs/remotes/origin/{}^{{commit}}", s.git_ref);
        let sib_sha = match git(&src, &["rev-parse", "--verify", "--quiet", &want]) {
            Ok(o) if valid_sha(o.trim()) => o.trim().to_string(),
            _ => {
                notes.push(format!(
                    "sibling ../{}: {} has no origin/{}; not provided",
                    s.dir,
                    src.display(),
                    s.git_ref
                ));
                continue;
            }
        };
        let path = co.run_dir.join(&s.dir);
        let res = if cfg.sandbox == Sandbox::User {
            let tool = cfg
                .wt_tool
                .as_ref()
                .map(|t| t.display().to_string())
                .unwrap_or_default();
            let src_s = src.display().to_string();
            run_cmd(&src, &tool, &["sync", &s.source, &src_s]).and_then(|_| {
                run_cmd(
                    &src,
                    &tool,
                    &["sibling", repo, sha, &s.dir, &s.source, &sib_sha],
                )
            })
        } else {
            let _ = git(&src, &["worktree", "prune"]);
            git(
                &src,
                &[
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    &path.display().to_string(),
                    &sib_sha,
                ],
            )
        };
        match res {
            Ok(_) => {
                notes.push(format!(
                    "sibling ../{}: {} origin/{} {}",
                    s.dir, s.source, s.git_ref, sib_sha
                ));
                made.push((src, path));
                pin_entries.push((s.dir.clone(), sib_sha));
            }
            Err(e) => notes.push(format!("sibling ../{}: not provided: {e}", s.dir)),
        }
    }
    for p in pins {
        pin_entries.push((format!("pin:{}", p.source), p.sha.clone()));
    }
    // ADR 0028 receipt member `pins`: `{name, commit}` sorted by `name`.
    pin_entries.sort();
    let pin_json: Vec<Value> = pin_entries
        .into_iter()
        .map(|(name, commit)| json!({ "name": name, "commit": commit }))
        .collect();
    (made, Value::Array(pin_json))
}

/// Under `Sandbox::User`: make the worktree with INFRA's `wt.sh` (`sync` the
/// mirror from the trusted clone, `add` the head), so `aien-run` can read its
/// git data and write its files while the `.git` link stays root-owned. An
/// older tree for the same head is removed first and never reused (a `main`
/// qualification must not inherit what a pull request run left behind).
fn checkout_for_sandbox(
    cfg: &Config,
    repo: &str,
    sha: &str,
    clone: &Path,
) -> Result<(PathBuf, PathBuf, bool), String> {
    let tool = cfg.wt_tool.as_ref().ok_or_else(|| {
        "--sandbox user needs INFRA's worktree tool (--wt-tool, tools/aien-local-ci/wt.sh): \
         a checkout made from the private clone is unreadable to aien-run"
            .to_string()
    })?;
    let tool = tool.display().to_string();
    let clone_s = clone.display().to_string();
    run_cmd(clone, &tool, &["sync", repo, &clone_s])?;
    let mirror = cfg.mirror_root.join(format!("{repo}.git"));
    let commit = format!("{sha}^{{commit}}");
    if git(&mirror, &["cat-file", "-e", &commit]).is_err() {
        // A head no branch names (a pull request head, a trigger for a
        // detached commit): fetch that one commit into the mirror.
        git(&mirror, &["fetch", "--quiet", &clone_s, sha])?;
    }
    git(&mirror, &["cat-file", "-e", &commit])
        .map_err(|_| format!("commit {sha} is not in {}", mirror.display()))?;
    let run_dir = cfg.wt_root.join(repo).join(sha);
    if fs::symlink_metadata(&run_dir).is_ok() {
        run_cmd(clone, &tool, &["remove", repo, sha])?;
    }
    let made = run_cmd(clone, &tool, &["add", repo, sha])?;
    let made = PathBuf::from(made.trim());
    // The run directory holds the checkout at `<repo>/` (siblings beside it);
    // an older wt.sh still makes the checkout at the run directory itself.
    let nested = made == run_dir.join(repo);
    if !nested && made != run_dir {
        return Err(format!(
            "{tool} made {:?}, expected {}",
            made,
            run_dir.join(repo).display()
        ));
    }
    if let Err(e) = check_worktree_link(&made, &mirror) {
        let _ = run_cmd(clone, &tool, &["remove", repo, sha]);
        return Err(e);
    }
    Ok((made, run_dir, nested))
}

/// The checks that make a `wt.sh` worktree safe to open with the operator's
/// git: the directory is no symlink, its `.git` is a regular root-owned file
/// nobody else can write, and it points into `<mirror>/worktrees/`.
pub fn check_worktree_link(wt: &Path, mirror: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let canon = fs::canonicalize(wt).map_err(|e| format!("{}: {e}", wt.display()))?;
    if canon != wt {
        return Err(format!("{} is not a plain directory path", wt.display()));
    }
    let link = wt.join(".git");
    let md = fs::symlink_metadata(&link).map_err(|e| format!("{}: {e}", link.display()))?;
    if !md.file_type().is_file() || md.uid() != 0 || md.mode() & 0o022 != 0 {
        return Err(format!(
            "{} must be a root-owned file nobody else can write (uid {}, mode {:o})",
            link.display(),
            md.uid(),
            md.mode() & 0o7777
        ));
    }
    let text = fs::read_to_string(&link).map_err(|e| format!("{}: {e}", link.display()))?;
    let target = text
        .trim_end_matches('\n')
        .strip_prefix("gitdir: ")
        .ok_or_else(|| format!("{} has no gitdir line", link.display()))?;
    let target = Path::new(target);
    let want = mirror.join("worktrees");
    if target.parent() != Some(want.as_path())
        || target
            .components()
            .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(format!(
            "{} points to {}, not into {}",
            link.display(),
            target.display(),
            want.display()
        ));
    }
    Ok(())
}

/// Fetch the clone and add a detached worktree for `sha`. The clone's own
/// checkout is never touched.
fn checkout(
    cfg: &Config,
    repo: &str,
    sha: &str,
    notes: &mut Vec<String>,
) -> Result<Checkout, String> {
    let clone = cfg.clones.join(repo);
    if !clone.join(".git").exists() && !clone.join("HEAD").exists() {
        return Err(format!("no trusted clone at {}", clone.display()));
    }
    let present = |c: &Path| git(c, &["cat-file", "-e", &format!("{sha}^{{commit}}")]).is_ok();
    if !present(&clone) {
        if let Err(e) = git(&clone, &["fetch", "--quiet", "--all"]) {
            notes.push(format!("fetch failed: {e}"));
        }
    } else if let Err(e) = git(&clone, &["fetch", "--quiet", "--all"]) {
        notes.push(format!("fetch failed (commit already present): {e}"));
    }
    if !present(&clone) {
        return Err(format!("commit {sha} is not in {}", clone.display()));
    }
    if cfg.sandbox == Sandbox::User {
        let (wt, run_dir, nested) = checkout_for_sandbox(cfg, repo, sha, &clone)?;
        return Ok(Checkout {
            clone,
            wt,
            run_dir,
            nested,
        });
    }
    let run_dir = cfg.wt_root.join(repo).join(sha);
    let wt = run_dir.join(repo);
    if fs::symlink_metadata(&run_dir).is_ok() {
        let _ = git(
            &clone,
            &["worktree", "remove", "--force", &wt.display().to_string()],
        );
        let _ = fs::remove_dir_all(&run_dir);
        let _ = git(&clone, &["worktree", "prune"]);
    }
    fs::create_dir_all(&run_dir)
        .map_err(|e| format!("cannot create {}: {e}", run_dir.display()))?;
    git(
        &clone,
        &[
            "worktree",
            "add",
            "--detach",
            "--quiet",
            &wt.display().to_string(),
            sha,
        ],
    )?;
    Ok(Checkout {
        clone,
        wt,
        run_dir,
        nested: true,
    })
}

/// `Sandbox::None`: remove the checkout, every sibling made for it and the
/// run directory, and prune each clone's worktree list.
fn remove_checkout(co: &Checkout, made: &[(PathBuf, PathBuf)]) {
    let _ = git(
        &co.clone,
        &[
            "worktree",
            "remove",
            "--force",
            &co.wt.display().to_string(),
        ],
    );
    for (src, p) in made {
        let _ = git(
            src,
            &["worktree", "remove", "--force", &p.display().to_string()],
        );
    }
    let _ = fs::remove_dir_all(&co.run_dir);
    let _ = git(&co.clone, &["worktree", "prune"]);
    for (src, _) in made {
        let _ = git(src, &["worktree", "prune"]);
    }
}

fn crumb(cfg: &Config, wt: &Path, args: &[&str], notes: &mut Vec<String>) {
    let Some(c) = &cfg.crumb else {
        return;
    };
    if !wt.join(".crumb").exists() {
        notes.push(format!(
            "crumb {}: no .crumb at the worktree root, skipped",
            args[0]
        ));
        return;
    }
    match run_cmd(wt, c, args) {
        Ok(_) => {}
        Err(e) => notes.push(format!("crumb {}: {e}", args[0])),
    }
}

/// Verify one head end to end. Never panics on GitHub trouble.
pub fn run_once(cfg: &Config, repo: &str, sha: &str) -> Result<Summary, String> {
    // Refusals first: nothing is fetched, checked out or run before these.
    if !valid_repo_key(repo) {
        return Err(format!("refusing repository name {repo:?}"));
    }
    if !valid_sha(sha) {
        return Err(format!(
            "refusing commit {sha:?}: a lowercase 40-hex id is required"
        ));
    }
    // Development mode needs the explicit opt-in; the CLI also refuses it
    // with `--github` (tests post to a fake `gh` without the helper).
    cfg.check_sandbox_policy_for(false)?;
    check_key_private(&cfg.key)?;
    if cfg.sandbox == Sandbox::User && cfg.wt_tool.is_none() {
        return Err(
            "--sandbox user needs INFRA's worktree tool (--wt-tool, tools/aien-local-ci/wt.sh)"
                .to_string(),
        );
    }
    // The sibling map is read before anything is checked out: a broken map
    // refuses the run instead of leaving a half-made neighbourhood.
    let sibs = siblings_for(cfg, repo)?;
    let pins = pins_for(cfg, repo)?;
    let mut notes: Vec<String> = Vec::new();
    cfg.state.beat(repo, sha, "checkout");
    let co = checkout(cfg, repo, sha, &mut notes)?;
    ensure_pins(&cfg.clones, &pins, &sibs, &mut notes);
    let (made, pin_values) = provision_siblings(cfg, repo, sha, &co, &sibs, &pins, &mut notes);
    let (clone, wt) = (co.clone.clone(), co.wt.clone());
    // A wt.sh worktree belongs to aien-run; the operator's git opens it only
    // after `check_worktree_link` (see `process::trust_worktree`).
    let trust = (cfg.sandbox == Sandbox::User).then(|| process::trust_worktree(&wt));
    // Every sibling's commit is in `opts.pins` (`core.pins` and `cache.key`),
    // and the evidence directory is this head's own (`evidence/<repo>/<sha>`),
    // so a result is reused only for this same commit with every sibling at
    // the same commit. A declared sibling that could not be made turns the
    // cache off for the whole head.
    let fresh = made.len() < sibs.len();
    let result = run_in_worktree(cfg, repo, sha, &co, fresh, pin_values, &mut notes);
    crumb(
        cfg,
        &wt,
        &[
            "close",
            "aien-worker",
            ".",
            "worker",
            "test",
            "local verification finished",
        ],
        &mut notes,
    );
    drop(trust);
    if cfg.sandbox == Sandbox::User {
        if let Some(tool) = &cfg.wt_tool {
            let tool = tool.display().to_string();
            if let Err(e) = run_cmd(&clone, &tool, &["remove", repo, sha]) {
                notes.push(format!("worktree not removed: {e}"));
            }
        }
    } else {
        remove_checkout(&co, &made);
    }
    cfg.state.beat(repo, sha, "idle");
    let mut summary = result?;
    summary.notes.append(&mut notes);
    Ok(summary)
}

fn run_in_worktree(
    cfg: &Config,
    repo: &str,
    sha: &str,
    co: &Checkout,
    fresh: bool,
    pins: Value,
    notes: &mut Vec<String>,
) -> Result<Summary, String> {
    let (clone, wt) = (co.clone.as_path(), co.wt.as_path());
    let started = now_secs();
    crumb(
        cfg,
        wt,
        &[
            "claim",
            "aien-worker",
            ".",
            "worker",
            "local verification",
            "7200",
        ],
        notes,
    );
    let repository = plan::repository_name(clone, None);

    // Plan against the base ref; a head already on the base line is planned
    // against its first parent so a push to main still verifies what it
    // changed (the same rule `verify` uses).
    cfg.state.beat(repo, sha, "plan");
    let (base, on_main) = plan::base_for(wt, &cfg.base_ref, sha)?;
    let mut req = plan::Request::new(&base, sha);
    req.operator = cfg.operator;
    req.hardware_identity = cfg.hardware_identity.clone();
    let plan = plan::compute(wt, &req)?;

    // Run what the plan selects, through the sandbox and the pools.
    cfg.state.beat(repo, sha, "run");
    let evidence_dir = cfg.state.evidence(repo, sha);
    fs::create_dir_all(&evidence_dir).map_err(|e| format!("cannot create evidence dir: {e}"))?;
    let mut opts = Options::new(evidence_dir.clone());
    opts.sandbox = cfg.sandbox;
    opts.cargo_target_dir = target_dir_for(cfg, repo, on_main);
    opts.pool_sizes = opts.pool_sizes.with(cfg.jobs);
    opts.pins = pins;
    if fresh || cfg.no_cache {
        opts.no_cache = true;
    }
    let mut known: BTreeMap<String, Known> = BTreeMap::new();
    let mut notes_cache: BTreeMap<String, CacheNote> = BTreeMap::new();
    if plan.blocked.is_none() && !plan.entries.is_empty() {
        let g = graph::load(wt).map_err(|e| e.to_string())?;
        let mut sel = std::collections::BTreeSet::new();
        for e in plan
            .entries
            .iter()
            .filter(|e| e.section != Section::Blocked)
        {
            match g.index_of(&e.gate) {
                Some(i) => {
                    sel.insert(i);
                }
                None => notes.push(format!(
                    "gate {} is in the plan but not in the worktree graph",
                    e.gate
                )),
            }
        }
        let campaign = runner::run_graph(wt, &g, &sel, &opts).map_err(|e| e.to_string())?;
        for (id, o) in campaign.results {
            known.insert(
                id.clone(),
                Known {
                    verdict: o.verdict,
                    reason: o.reason.clone(),
                    receipt: o.receipt_digest.clone(),
                },
            );
            notes_cache.insert(id, o.cache);
        }
    }
    let judged = plan::judge(&plan, &known);

    // The bundle.
    cfg.state.beat(repo, sha, "bundle");
    let key = bundle::load_signing_key(&cfg.key)?;
    let vk = key.verifying_key();
    let machine = runner::machine(&Hardware::detect());
    let machine_digest = canonical_digest(&machine).map_err(|e| e.to_string())?;
    let tree = git(wt, &["rev-parse", &format!("{sha}^{{tree}}")])?;
    let mut checks: Vec<Value> = Vec::new();
    let mut entries: Vec<(bool, String, String)> = Vec::new();
    let mut gates: Vec<(String, String, Verdict)> = Vec::new();
    for e in &plan.entries {
        let (eff, _) = judged
            .gates
            .get(&e.gate)
            .cloned()
            .unwrap_or((Verdict::NotRun, String::new()));
        let k = known.get(&e.gate);
        let (status, verdict, extra): (&str, String, Value) =
            match (e.section, k, notes_cache.get(&e.gate)) {
                (Section::Blocked, _, _) => ("blocked", bundle_verdict(eff), json!({})),
                (_, Some(k), Some(CacheNote::Reused { from })) if k.receipt.is_some() => (
                    "cached",
                    bundle_verdict(eff),
                    json!({ "resolution": { "resolved_from": from, "receipt": k.receipt } }),
                ),
                (_, Some(k), _) if k.receipt.is_some() => {
                    ("run", bundle_verdict(eff), json!({ "receipt": k.receipt }))
                }
                _ => (
                    "not_run",
                    bundle_verdict(if eff == Verdict::Pass {
                        Verdict::NotRun
                    } else {
                        eff
                    }),
                    json!({}),
                ),
            };
        let mut entry = json!({
            "gate": e.gate,
            "check_id": e.check_id,
            "exactness": e.exactness.as_str(),
            "required": e.required,
            "status": status,
            "verdict": verdict,
            "backend": e.pool,
        });
        if let (Some(dst), Some(src)) = (entry.as_object_mut(), extra.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        checks.push(entry);
        entries.push((e.required, status.to_string(), verdict.clone()));
        gates.push((e.gate.clone(), status.to_string(), eff));
    }
    let bundle_verdict = bundle::compute_verdict(&entries).to_string();
    let unsigned = json!({
        "schema": bundle::SCHEMA,
        "repository": repository,
        "commit": sha,
        "tree_digest": sha256_hex(tree.trim().as_bytes()),
        "graph_digest": plan.graph_digest,
        "receipt_id_scheme": "sha256-canonical-json",
        "checks": checks,
        "verdict": bundle_verdict,
        "executor": { "id": bundle::fingerprint(&vk), "class": EXECUTOR_CLASS, "machine_digest": machine_digest },
        "started_utc": utc_string(started),
        "finished_utc": utc_string(now_secs()),
        // The planner's verdict beside the bundle algebra: a bundle with no
        // required checks is PASS by the algebra (nothing required failed)
        // while the plan may say NOT_RUN UNMAPPED_PATH or BLOCKED. Readers
        // that gate on the plan read this; `verify` recomputes it anyway.
        "plan": {
            "verdict": judged.verdict.as_str(),
            "reason": judged.reason,
            "unmapped": plan.unmapped,
            "removed": plan.removed,
            "blocked": plan.blocked,
        },
    });
    let signed = bundle::sign(&unsigned, &key).map_err(|r| format!("cannot sign bundle: {r}"))?;
    let bytes = canonical(&signed).map_err(|e| e.to_string())?;
    let bundle_digest = sha256_hex(&bytes);
    let bundle_path = cfg.state.bundle_path(repo, sha);
    fs::create_dir_all(bundle_path.parent().unwrap_or(&cfg.state.root))
        .map_err(|e| format!("cannot create bundle dir: {e}"))?;
    fs::write(&bundle_path, &bytes).map_err(|e| format!("cannot write bundle: {e}"))?;

    // The independent check of what was just written, exactly as `verify`
    // does it: keys at BASE, signature, graph digest, receipts, manifests,
    // crumb, with the same operator permission the plan had (a pipeline
    // configuration change the operator ran by hand is not BLOCKED here).
    // Nothing is published as success unless this says READY.
    cfg.state.beat(repo, sha, "verify");
    let check = verify::run(
        clone,
        &verify::Ask {
            state: cfg.state.clone(),
            commit: sha.to_string(),
            base: plan.base.clone(),
            repository: Some(repository.clone()),
            release: false,
            operator: cfg.operator,
            hardware_identity: cfg.hardware_identity.clone(),
            crumb: cfg.crumb.clone(),
            keys_override: None,
        },
    );
    let ready = check.ready;
    let check_problems: Vec<String> = check
        .lines
        .iter()
        .filter(|l| {
            l.starts_with("Problem: ")
                || l.starts_with("Evidence note: ")
                || l.starts_with("Crumb note: ")
        })
        .cloned()
        .collect();

    // Publication, then the journal.
    cfg.state.beat(repo, sha, "publish");
    let publication = if cfg.github {
        let status = Status {
            repository: repository.clone(),
            sha: sha.to_string(),
            state: publish_state(ready, &bundle_verdict, judged.verdict).to_string(),
            description: format!(
                "{} {} {}: bundle {}",
                if ready { "READY" } else { "BLOCKED" },
                bundle_verdict,
                judged.reason,
                bundle_path.display()
            )
            .chars()
            .take(140)
            .collect(),
        };
        match post_status(&cfg.gh, &status) {
            Ok(()) => "posted".to_string(),
            Err(e) => {
                notes.push(format!("GitHub unreachable, status queued: {e}"));
                queue_status(&cfg.state, repo, &status)?;
                "pending".to_string()
            }
        }
    } else {
        "local".to_string()
    };
    let unmapped = plan.unmapped.clone();
    cfg.state.journal_append(&json!({
        "utc": utc_string(now_secs()),
        "repo": repo,
        "repository": repository,
        "sha": sha,
        "base": plan.base,
        "verdict": judged.verdict.as_str(),
        "reason": judged.reason,
        "bundle_verdict": bundle_verdict,
        "bundle": bundle_path.display().to_string(),
        "bundle_digest": bundle_digest,
        "graph_digest": plan.graph_digest,
        "gates": gates.iter().map(|(g, s, v)| json!({ "gate": g, "status": s, "verdict": v.as_str() })).collect::<Vec<_>>(),
        "unmapped": unmapped,
        "verify": if ready { "READY" } else { "BLOCKED" },
        "verify_notes": check_problems,
        "publication": publication,
        "notes": notes,
    }))?;
    Ok(Summary {
        repo: repo.to_string(),
        sha: sha.to_string(),
        repository,
        verdict: judged.verdict,
        reason: judged.reason,
        bundle_verdict,
        bundle_path,
        bundle_digest,
        gates,
        publication,
        ready,
        notes: Vec::new(),
    })
}

/// `CARGO_TARGET_DIR` for the gates of one head. Under the user sandbox it is
/// the INFRA target dir of the head's trust class (A2):
/// `<target_root>/<repo>/main` for a head on the base line, `.../pr`
/// otherwise. The exec helper accepts only exactly those two paths under
/// `/var/lib/aien-run/target` and creates the class dir on demand. Never the
/// worker's own target dir, which `aien-run` cannot write. Without the
/// sandbox, the worker's own `CARGO_TARGET_DIR`.
fn target_dir_for(cfg: &Config, repo: &str, on_main: bool) -> Option<PathBuf> {
    match cfg.sandbox {
        Sandbox::None => std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from),
        Sandbox::User => {
            let class = if on_main { "main" } else { "pr" };
            Some(cfg.target_root.join(repo).join(class))
        }
    }
}

/// The commit status for a head: `success` only when the independent check
/// of the bundle just written says READY; `failure` when the bundle or the
/// plan says FAIL; `error` otherwise.
pub fn publish_state(ready: bool, bundle_verdict: &str, judged: Verdict) -> &'static str {
    match github_state(bundle_verdict, judged) {
        "success" if !ready => "error",
        s => s,
    }
}

fn bundle_verdict(v: Verdict) -> String {
    v.as_str().to_string()
}

/// success only on PASS, failure on FAIL, error on anything else.
pub fn github_state(bundle_verdict: &str, judged: Verdict) -> &'static str {
    match (bundle_verdict, judged) {
        ("PASS", Verdict::Pass) => "success",
        ("FAIL", _) | (_, Verdict::Fail) => "failure",
        _ => "error",
    }
}

/// A commit status to post.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub repository: String,
    pub sha: String,
    pub state: String,
    pub description: String,
}

fn owner_repo(repository: &str) -> Option<String> {
    let parts: Vec<&str> = repository.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() >= 3 {
        Some(format!(
            "{}/{}",
            parts[parts.len() - 2],
            parts[parts.len() - 1]
        ))
    } else if parts.len() == 2 {
        Some(format!("{}/{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// `gh api -X POST repos/O/R/statuses/SHA ...`.
pub fn post_status(gh: &str, s: &Status) -> Result<(), String> {
    let or = owner_repo(&s.repository)
        .ok_or_else(|| format!("cannot derive owner/repo from {}", s.repository))?;
    let mut cmd = Command::new(gh);
    cmd.args([
        "api",
        "-X",
        "POST",
        &format!("repos/{or}/statuses/{}", s.sha),
        "-f",
        &format!("state={}", s.state),
        "-f",
        &format!("context={CONTEXT}"),
        "-f",
        &format!("description={}", s.description),
    ]);
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run {gh}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Queue a status in the outbox. File names start with a strictly increasing
/// 20-digit sequence (nanoseconds since the epoch, bumped past the newest
/// queued file), so the outbox replays in the order statuses were queued. A
/// newer status for the same head replaces the older one and takes its place
/// at the end of the queue: the latest state is what GitHub must end with.
fn queue_status(state: &StateDir, repo: &str, s: &Status) -> Result<(), String> {
    let dir = state.outbox_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create outbox: {e}"))?;
    let suffix = format!(".{repo}.{}.json", s.sha);
    let mut newest: u128 = 0;
    for f in outbox_files(state) {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(seq) = name.split('.').next().and_then(|n| n.parse::<u128>().ok()) {
            newest = newest.max(seq);
        }
        if name.ends_with(&suffix) || name == suffix.trim_start_matches('.') {
            let _ = fs::remove_file(&f);
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = now.max(newest + 1);
    let v = json!({ "repository": s.repository, "sha": s.sha, "state": s.state, "description": s.description });
    let tmp = dir.join(format!(".tmp{suffix}"));
    fs::write(&tmp, serde_json::to_string(&v).unwrap_or_default())
        .map_err(|e| format!("cannot write outbox: {e}"))?;
    fs::rename(&tmp, dir.join(format!("{seq:020}{suffix}")))
        .map_err(|e| format!("cannot write outbox: {e}"))
}

/// The queued status files, in queue order (`*.json`, by name).
pub fn outbox_files(state: &StateDir) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(state.outbox_dir()) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            n.ends_with(".json") && !n.starts_with('.')
        })
        .collect();
    files.sort();
    files
}

/// Post the queued statuses in queue order; a success removes the file. The
/// first failure stops the replay so nothing overtakes an older status; the
/// rest stay queued for the next round. A file that is not a status is
/// renamed to `*.bad` (journaled) so it cannot block the queue. Returns
/// (posted, still pending).
pub fn flush_outbox(cfg: &Config) -> (usize, usize) {
    let files = outbox_files(&cfg.state);
    let mut posted = 0;
    for (i, f) in files.iter().enumerate() {
        let status = fs::read_to_string(f)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| {
                let s = Status {
                    repository: v["repository"].as_str()?.to_string(),
                    sha: v["sha"].as_str()?.to_string(),
                    state: v["state"].as_str()?.to_string(),
                    description: v["description"].as_str().unwrap_or("").to_string(),
                };
                (valid_sha(&s.sha)
                    && ["success", "failure", "error", "pending"].contains(&s.state.as_str()))
                .then_some(s)
            });
        let Some(s) = status else {
            let _ = fs::rename(f, f.with_extension("bad"));
            let _ = cfg.state.journal_append(&json!({
                "utc": utc_string(now_secs()),
                "publication": "dropped",
                "note": format!("outbox file {} is not a status; renamed to .bad", f.display()),
            }));
            continue;
        };
        match post_status(&cfg.gh, &s) {
            Ok(()) => {
                let _ = fs::remove_file(f);
                posted += 1;
                let _ = cfg.state.journal_append(&json!({
                    "utc": utc_string(now_secs()),
                    "sha": s.sha,
                    "repository": s.repository,
                    "state": s.state,
                    "publication": "posted",
                    "note": "flushed from outbox",
                }));
            }
            Err(_) => return (posted, files.len() - i),
        }
    }
    (posted, 0)
}

/// Pending trigger files as (repo, sha), oldest first (modification time,
/// then name). Names that are not `<repo>.<40-hex sha>` with a safe repo
/// key are ignored.
pub fn triggers(state: &StateDir) -> Vec<(String, String, PathBuf)> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(state.trigger_dir()) {
        let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        // Oldest first: by modification time, then by name.
        files.sort_by_key(|f| (fs::metadata(f).and_then(|m| m.modified()).ok(), f.clone()));
        for f in files {
            let name = f
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Some((repo, sha)) = name.rsplit_once('.') {
                let sha = sha.to_ascii_lowercase();
                if valid_sha(&sha) && valid_repo_key(repo) {
                    out.push((repo.to_string(), sha, f.clone()));
                }
            }
        }
    }
    out
}

/// Open PR heads and the main head of `repo` from GitHub (bounded: one call
/// per list). Errors are returned, never fatal to the loop.
pub fn github_heads(gh: &str, repository: &str) -> Result<Vec<String>, String> {
    let or = owner_repo(repository).ok_or("bad repository name")?;
    let mut shas = Vec::new();
    for args in [
        vec![
            "api",
            &format!("repos/{or}/pulls?state=open&per_page=50"),
            "--jq",
            ".[].head.sha",
        ],
        vec!["api", &format!("repos/{or}/commits/main"), "--jq", ".sha"],
    ] {
        let mut cmd = Command::new(gh);
        cmd.args(&args);
        let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run {gh}: {e}"))?;
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        for l in String::from_utf8_lossy(&out.stdout).lines() {
            let l = l.trim().to_ascii_lowercase();
            if valid_sha(&l) {
                shas.push(l);
            }
        }
    }
    shas.sort();
    shas.dedup();
    Ok(shas)
}

/// The loop: flush the outbox, serve trigger files, poll GitHub when asked,
/// sleep. `iterations` bounds the rounds (None: forever). Returns the number
/// of heads served.
pub fn run_loop(cfg: &Config, iterations: Option<u64>) -> usize {
    let mut served = 0;
    let mut round = 0u64;
    loop {
        round += 1;
        cfg.state.beat("-", "-", "poll");
        // Posting is the operator's choice (--github). A loop without it leaves
        // queued statuses where they are, so `aien-local-ci start` after an
        // offline --github run never posts them behind the operator's back.
        let (posted, pending) = if cfg.github {
            flush_outbox(cfg)
        } else {
            (0, outbox_files(&cfg.state).len())
        };
        if posted + pending > 0 {
            println!("OUTBOX posted={posted} pending={pending}");
        }
        let mut heads: Vec<(String, String, Option<PathBuf>)> = triggers(&cfg.state)
            .into_iter()
            .map(|(r, s, f)| (r, s, Some(f)))
            .collect();
        if cfg.github {
            for repo in &cfg.repos {
                let repository = plan::repository_name(&cfg.clones.join(repo), None);
                match github_heads(&cfg.gh, &repository) {
                    Ok(shas) => {
                        for s in shas {
                            if !cfg.state.bundle_path(repo, &s).exists() {
                                heads.push((repo.clone(), s, None));
                            }
                        }
                    }
                    Err(e) => println!("GITHUB unreachable for {repo}: {e} (will retry)"),
                }
            }
        }
        for (repo, sha, trigger) in heads {
            match run_once(cfg, &repo, &sha) {
                Ok(s) => {
                    println!(
                        "HEAD {repo} {sha} {} {} bundle {} publication {}",
                        s.verdict.as_str(),
                        s.reason,
                        s.bundle_path.display(),
                        s.publication
                    );
                    served += 1;
                }
                Err(e) => {
                    println!("HEAD {repo} {sha} ERROR {e}");
                    let _ = cfg.state.journal_append(&json!({
                        "utc": utc_string(now_secs()),
                        "repo": repo,
                        "sha": sha,
                        "verdict": "NOT_RUN",
                        "reason": format!("WORKER_ERROR:{e}"),
                        "publication": "none",
                    }));
                }
            }
            if let Some(f) = trigger {
                let _ = fs::remove_file(f);
            }
        }
        if iterations.is_some_and(|n| round >= n) {
            break;
        }
        std::thread::sleep(cfg.poll);
    }
    served
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_list_selects_the_repository_and_refuses_bad_lines() {
        let sha = "353b7d4be3f00169545caec1615cffa61f7d311a";
        let list =
            format!("# c\naien-sovereign-core aien-protocols {sha} # t\nomega physics {sha}\n");
        let p = parse_pins(&list, "aien-sovereign-core").unwrap();
        assert_eq!(
            p,
            vec![Pin {
                source: "aien-protocols".into(),
                sha: sha.into()
            }]
        );
        assert!(parse_pins(&list, "physics").unwrap().is_empty());
        for bad in [
            "a b".to_string(),
            format!("a b {sha} extra"),
            format!("a ../b {sha}"),
            "a b 353b7d4".to_string(),
            format!("a b {}", sha.to_uppercase()),
        ] {
            assert!(parse_pins(&bad, "a").is_err(), "{bad}");
        }
    }

    fn sh_git(dir: &Path, args: &[&str]) -> String {
        git(dir, args).unwrap_or_else(|e| panic!("git {args:?}: {e}"))
    }

    /// The trusted clone lacks a pinned commit that only its origin has (on a
    /// branch the clone never fetched): ensure_pins fetches it by sha and holds
    /// it under refs/remotes/aien-pins/. Negative controls: a pin of a
    /// repository that is not a sibling, and a commit origin does not have.
    #[test]
    fn ensure_pins_fetches_a_missing_pinned_commit_from_origin() {
        let t = crate::testutil::TempDir::new("pins");
        let origin = t.path().join("origin");
        fs::create_dir_all(&origin).unwrap();
        sh_git(&origin, &["init", "-q", "-b", "main"]);
        for (k, v) in [
            ("user.name", "t"),
            ("user.email", "t@t"),
            ("uploadpack.allowAnySHA1InWant", "true"),
        ] {
            sh_git(&origin, &["config", k, v]);
        }
        fs::write(origin.join("f"), "1").unwrap();
        sh_git(&origin, &["add", "-A"]);
        sh_git(&origin, &["commit", "-q", "-m", "one"]);
        sh_git(&origin, &["checkout", "-q", "-b", "side"]);
        fs::write(origin.join("f"), "2").unwrap();
        sh_git(&origin, &["commit", "-q", "-am", "two"]);
        let pinned = sh_git(&origin, &["rev-parse", "HEAD"]).trim().to_string();
        sh_git(&origin, &["checkout", "-q", "main"]);
        let clones = t.path().join("clones");
        fs::create_dir_all(&clones).unwrap();
        sh_git(
            &clones,
            &[
                "clone",
                "-q",
                "--single-branch",
                "--no-local",
                "-b",
                "main",
                origin.to_str().unwrap(),
                "src",
            ],
        );
        let src = clones.join("src");
        let want = format!("{pinned}^{{commit}}");
        assert!(
            git(&src, &["cat-file", "-e", &want]).is_err(),
            "the clone must lack the pin first"
        );

        let sibs = vec![Sibling {
            dir: "src".into(),
            source: "src".into(),
            git_ref: "main".into(),
        }];
        let absent = "0123456789abcdef0123456789abcdef01234567".to_string();
        let pins = vec![
            Pin {
                source: "src".into(),
                sha: pinned.clone(),
            },
            Pin {
                source: "other".into(),
                sha: pinned.clone(),
            },
            Pin {
                source: "src".into(),
                sha: absent.clone(),
            },
        ];
        let mut notes = Vec::new();
        ensure_pins(&clones, &pins, &sibs, &mut notes);
        assert!(git(&src, &["cat-file", "-e", &want]).is_ok(), "{notes:?}");
        let held = sh_git(
            &src,
            &["rev-parse", &format!("refs/remotes/aien-pins/{pinned}")],
        );
        assert_eq!(held.trim(), pinned);
        assert!(
            notes.iter().any(|n| n.contains("fetched from origin")),
            "{notes:?}"
        );
        assert!(
            notes.iter().any(|n| n.contains("other is not a sibling")),
            "{notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|n| n.contains(&absent) && n.contains("not provided")),
            "{notes:?}"
        );
        assert!(git(
            &src,
            &[
                "rev-parse",
                "--verify",
                "--quiet",
                &format!("refs/remotes/aien-pins/{absent}")
            ]
        )
        .is_err());
    }
    #[test]
    fn sibling_map_selects_the_repository_and_refuses_bad_lines() {
        let map = "# comment\n\
                   aegis-runtime  aien-protocols  aien-protocols  main\n\
                   benchmarks canonical-branch-native-runtime aien-sovereign-core feat/canonical-branch-native-runtime # trailing\n\
                   physics aien-architecture aien-architecture main\n\
                   physics omega omega main\n";
        let p = parse_siblings(map, "physics").unwrap();
        assert_eq!(
            p.iter().map(|s| s.dir.as_str()).collect::<Vec<_>>(),
            ["aien-architecture", "omega"]
        );
        let b = parse_siblings(map, "benchmarks").unwrap();
        assert_eq!(b[0].source, "aien-sovereign-core");
        assert_eq!(b[0].git_ref, "feat/canonical-branch-native-runtime");
        assert!(parse_siblings(map, "omega").unwrap().is_empty());
        assert!(parse_siblings("", "omega").unwrap().is_empty());
        // Every line is checked, also lines for other repositories.
        for bad in [
            "x y z",
            "x y z main extra",
            "x .. z main",
            "x y/z z main",
            "x y ../z main",
            ".x y z main",
            "x y z -main",
            "x y z a/../b",
            "x y z a//b",
            "x y z .hidden",
            "x y z a;b",
            "x x z main",
            "physics omega omega main\nphysics omega physics main",
        ] {
            let e = parse_siblings(&format!("ok a b main\n{bad}\n"), "ok");
            assert!(e.is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn sibling_refs_are_plain_branch_names() {
        for good in ["main", "feat/canonical-branch-native-runtime", "v1.2_x-y"] {
            assert!(valid_ref(good), "{good}");
        }
        for bad in [
            "", "-x", "/x", "x/", "a//b", "a/../b", ".x", "a/.b", "a b", "a~1", "a^", "a:b",
        ] {
            assert!(!valid_ref(bad), "{bad:?}");
        }
    }

    #[test]
    fn state_layout() {
        let s = StateDir::new(Path::new("/s"));
        assert_eq!(s.evidence("r", "abc"), PathBuf::from("/s/evidence/r/abc"));
        assert_eq!(
            s.bundle_path("r", "abc"),
            PathBuf::from("/s/bundles/r/abc/bundle.json")
        );
        assert_eq!(s.trigger_dir(), PathBuf::from("/s/trigger"));
        assert_eq!(s.journal(), PathBuf::from("/s/journal.jsonl"));
        assert_eq!(
            StateDir::repo_key("github.com/aien-dev/aien-sovereign-core"),
            "aien-sovereign-core"
        );
        assert_eq!(StateDir::repo_key("x"), "x");
    }

    #[test]
    fn github_states() {
        assert_eq!(github_state("PASS", Verdict::Pass), "success");
        assert_eq!(github_state("FAIL", Verdict::Fail), "failure");
        assert_eq!(github_state("NOT_RUN", Verdict::NotRun), "error");
        assert_eq!(github_state("NOT_RUN", Verdict::BlockedOperator), "error");
        // A PASS bundle whose plan-level verdict is not PASS is not success.
        assert_eq!(github_state("PASS", Verdict::NotRun), "error");
        assert_eq!(
            owner_repo("github.com/aien-dev/x").as_deref(),
            Some("aien-dev/x")
        );
        assert_eq!(owner_repo("aien-dev/x").as_deref(), Some("aien-dev/x"));
        assert_eq!(owner_repo("x"), None);
        // success needs the independent check too.
        assert_eq!(publish_state(true, "PASS", Verdict::Pass), "success");
        assert_eq!(publish_state(false, "PASS", Verdict::Pass), "error");
        assert_eq!(publish_state(false, "FAIL", Verdict::Fail), "failure");
        assert_eq!(publish_state(true, "NOT_RUN", Verdict::NotRun), "error");
    }

    #[test]
    fn names_and_sandbox_policy() {
        assert!(valid_repo_key("aien-sovereign-core"));
        assert!(valid_repo_key("a.b_c-1"));
        for bad in ["", ".", "..", ".x", "a/b", "a b", "a\0b"] {
            assert!(!valid_repo_key(bad), "{bad:?}");
        }
        assert!(valid_sha(&"a".repeat(40)));
        assert!(!valid_sha(&"A".repeat(40)));
        assert!(!valid_sha(&"a".repeat(39)));
        assert!(!valid_sha(&"g".repeat(40)));
        let mut cfg = Config::new(
            StateDir::new(Path::new("/s")),
            Path::new("/k"),
            Path::new("/c"),
            Path::new("/w"),
        );
        // The library default is the development sandbox, refused unless allowed.
        assert_eq!(cfg.sandbox, Sandbox::None);
        assert!(cfg.check_sandbox_policy_for(false).is_err());
        cfg.allow_unsandboxed = true;
        assert!(cfg.check_sandbox_policy_for(false).is_ok());
        assert!(cfg.check_sandbox_policy_for(true).is_err());
        cfg.sandbox = Sandbox::User;
        assert!(cfg.check_sandbox_policy_for(true).is_ok());
    }

    #[test]
    fn triggers_and_outbox_files() {
        let t = crate::testutil::TempDir::new("worker-state");
        let s = StateDir::new(t.path());
        fs::create_dir_all(s.trigger_dir()).unwrap();
        fs::write(s.trigger_dir().join(format!("repo.{}", "a".repeat(40))), "").unwrap();
        fs::write(s.trigger_dir().join("junk"), "").unwrap();
        fs::write(s.trigger_dir().join("repo.short"), "").unwrap();
        let t3 = triggers(&s);
        assert_eq!(t3.len(), 1);
        assert_eq!(
            (t3[0].0.as_str(), t3[0].1.as_str()),
            ("repo", "a".repeat(40).as_str())
        );
        let st = Status {
            repository: "github.com/o/r".into(),
            sha: "b".repeat(40),
            state: "success".into(),
            description: "d".into(),
        };
        queue_status(&s, "r", &st).unwrap();
        let q = outbox_files(&s);
        assert_eq!(q.len(), 1);
        let name = q[0].file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name.ends_with(&format!(".r.{}.json", "b".repeat(40))),
            "{name}"
        );
        assert_eq!(name.split('.').next().unwrap().len(), 20, "{name}");
        // Queue order: c after b; a newer status for b replaces the older one
        // and moves to the end.
        let c = Status {
            sha: "c".repeat(40),
            ..st.clone()
        };
        queue_status(&s, "r", &c).unwrap();
        let b2 = Status {
            state: "failure".into(),
            ..st.clone()
        };
        queue_status(&s, "r", &b2).unwrap();
        let q = outbox_files(&s);
        assert_eq!(q.len(), 2);
        assert!(q[0].to_string_lossy().contains(&"c".repeat(40)));
        assert!(q[1].to_string_lossy().contains(&"b".repeat(40)));
        assert!(fs::read_to_string(&q[1]).unwrap().contains("failure"));
        s.journal_append(&json!({"a": 1})).unwrap();
        s.journal_append(&json!({"b": 2})).unwrap();
        let j = fs::read_to_string(s.journal()).unwrap();
        assert_eq!(j.lines().count(), 2);
        s.beat("r", "sha", "plan");
        let hb: Value = serde_json::from_str(&fs::read_to_string(s.heartbeat()).unwrap()).unwrap();
        assert_eq!(hb["phase"], "plan");
    }
}
