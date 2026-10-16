//! The runner (ADR 0028, Slices A and B): one gate, or a whole graph of gates
//! run by worker threads that honour the resource pools and the dependencies.
//!
//! Order for one gate: parse manifest (BAD_MANIFEST means no receipt), pin the
//! commit, record whether the tree is clean, decide refusals, take the GPU
//! locks if it is a gb10 gate, run, measure what changed, derive the verdict,
//! write blobs and an immutable receipt.
//!
//! Choices where the ADR is silent (recorded in the PRs):
//! - A dirty tree is refused unless the operator passes `--allow-dirty` (the
//!   manifest grammar has no key for it and P4 forbids adding one here). A
//!   run allowed this way is recorded with `tree_clean_before = false`.
//! - "Tree changed after the run" means `git status --porcelain` differs from
//!   the pre-run output, so it also catches a run that dirties an already
//!   allowed-dirty tree further. In a graph run the check is per job and does
//!   not know which job wrote a file, so when one gate dirties the tree, gates
//!   that were running at the same moment fail with it. That is the safe side.
//! - `pins` is always empty.
//! - `gb10_present` is a heuristic: `/dev/nvidiactl` exists (UNVERIFIED).
//! - `run GATE` never starts a gate's dependencies. It looks their receipts
//!   up in the evidence store, for the current commit, and says NOT_RUN if one
//!   is not a PASS. `run_graph` starts them.

use crate::cache;
use crate::evidence::{build_receipt, canonical_digest, sha256_hex, utc_string, Index, Store};
use crate::graph::Graph;
use crate::manifest::{self, Manifest};
use crate::process::{self, RunSpec};
use crate::resources::{
    self, acquire_gpu, lock, GpuConfig, GpuGuard, Hardware, Permit, Pool, PoolSizes, Pools,
};
use crate::sandbox::{self, Sandbox};
use crate::verdict::{self, campaign_verdict, derive, parse_observations, DepFact, Facts, Verdict};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Component, Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Options {
    pub evidence_dir: PathBuf,
    pub allow_dirty: bool,
    /// `--no-cache`: never reuse an earlier result; every gate runs fresh.
    pub no_cache: bool,
    pub args_override: Option<Vec<String>>,
    pub kill_grace: Duration,
    /// Sizes of the host and qemu pools (the gb10 pool is always 1).
    pub pool_sizes: PoolSizes,
    /// Which special hardware this machine has.
    pub hardware: Hardware,
    /// Lock files and refusal checks for gb10 gates.
    pub gpu: GpuConfig,
    /// `--sandbox none|user` (contract C8). `None` keeps every earlier
    /// behaviour unchanged.
    pub sandbox: Sandbox,
    /// `CARGO_TARGET_DIR` as the runner itself was given it; passed through
    /// the sandbox as the only environment pair.
    pub cargo_target_dir: Option<PathBuf>,
    /// Where the per-target-dir lock files live (`resources::target_lock_path`).
    /// A cargo gate (`Manifest::uses_cargo`) holds its target dir's lock for
    /// its whole run when `cargo_target_dir` is set, so two gates never build
    /// in one target dir at once. The worker sets `<state>/locks/target`.
    pub target_lock_dir: PathBuf,
    /// `--release`: qualification mode. No cache reuse and the tree must be
    /// clean even with `--allow-dirty`.
    pub release: bool,
    /// `core.pins` and part of `cache.key` (ADR 0028: `{name, commit}` sorted
    /// by `name`). The worker sets the head's sibling and operator pins.
    pub pins: Value,
}

impl Options {
    /// Options for the real machine: detected pool sizes, hardware and GPU
    /// paths. Tests overwrite the last three with scratch values.
    pub fn new(evidence_dir: PathBuf) -> Options {
        Options {
            evidence_dir,
            allow_dirty: false,
            no_cache: false,
            args_override: None,
            kill_grace: Duration::from_secs(10),
            pool_sizes: PoolSizes::detect(),
            hardware: Hardware::detect(),
            gpu: GpuConfig::detect(),
            sandbox: Sandbox::None,
            cargo_target_dir: None,
            target_lock_dir: PathBuf::from(crate::resources::DEFAULT_TARGET_LOCK_DIR),
            release: false,
            pins: json!([]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// Manifest invalid: no receipt (P8).
    BadManifest(String),
    /// Could not bind or write evidence (CR-004 style fatal).
    Fatal(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::BadManifest(m) => write!(f, "BAD_MANIFEST {m}"),
            RunError::Fatal(m) => write!(f, "FATAL {m}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub verdict: Verdict,
    pub reason: String,
    pub receipt_path: Option<PathBuf>,
    pub receipt_digest: Option<String>,
    /// What the digest cache did for this gate.
    pub cache: CacheNote,
}

/// What the cache did, in words the report prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheNote {
    /// The result was reused, not rerun. Holds the digest of the fresh
    /// receipt it was taken from.
    Reused { from: String },
    /// The cache was consulted and had nothing usable; the gate ran.
    Miss(String),
    /// The cache was not consulted; the gate ran (or was refused). Holds why.
    Skipped(String),
}

/// The exec helper refused the call: exit 2 and stderr starting with its
/// `aien-run-exec:` prefix (INFRA C8). Only under `Sandbox::User`.
pub fn sandbox_refused(sandbox: Sandbox, exit: Option<i32>, stderr: &[u8]) -> bool {
    sandbox == Sandbox::User && exit == Some(2) && stderr.starts_with(b"aien-run-exec:")
}

fn git_out(root: &Path, args: &[&str]) -> Result<String, String> {
    // Several jobs may ask for the status at once; the status command must
    // not take the index lock (and fail the other caller) to refresh it.
    let mut cmd = process::git_command();
    cmd.env("GIT_OPTIONAL_LOCKS", "0")
        .arg("-C")
        .arg(root)
        .args(args);
    // The spawn goes through the spawn gate (see `process::SPAWN_GATE`) so a
    // git child never holds a GPU lock descriptor past the lock's release.
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn repo_root(dir: &Path) -> Result<PathBuf, String> {
    let s = git_out(dir, &["rev-parse", "--show-toplevel"])?;
    Ok(PathBuf::from(s.trim()))
}

/// The full commit id of HEAD (40 lower-case hex digits).
pub fn head(root: &Path) -> Result<String, String> {
    let s = git_out(root, &["rev-parse", "HEAD"])?.trim().to_string();
    if s.len() == 40
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
    {
        Ok(s)
    } else {
        Err(format!("HEAD is not a 40-hex commit: {s:?}"))
    }
}

fn porcelain(root: &Path) -> Result<String, String> {
    git_out(root, &["status", "--porcelain"])
}

/// Where receipts go when `--evidence-dir` is not given:
/// `$HOME/workspace/evidence-out/<repository directory name>`.
pub fn default_evidence_dir(root: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let name = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".to_string());
    Some(
        PathBuf::from(home)
            .join("workspace")
            .join("evidence-out")
            .join(name),
    )
}

/// Absolute, lexically normalized path whose deepest existing ancestor is
/// canonicalized, so a not-yet-created evidence dir can still be compared.
fn resolve(p: &Path) -> PathBuf {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut norm = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                norm.pop();
            }
            Component::CurDir => {}
            other => norm.push(other.as_os_str()),
        }
    }
    let mut existing = norm.clone();
    let mut tail = Vec::new();
    while !existing.exists() {
        match existing.file_name().map(|s| s.to_os_string()) {
            Some(n) => {
                tail.push(n);
                existing.pop();
            }
            None => break,
        }
    }
    let mut base = fs::canonicalize(&existing).unwrap_or(existing);
    for n in tail.iter().rev() {
        base.push(n);
    }
    base
}

const SKIP_DIRS: [&str; 5] = [".git", "target", "build", "vendor", "node_modules"];

/// D1: every `*.gate` file under `root`, skipping the usual build dirs and any
/// directory holding `.aien-test-ignore`. Sorted.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join(".aien-test-ignore").exists() {
            return;
        }
        let rd = match fs::read_dir(dir) {
            Ok(r) => r,
            Err(_) => return,
        };
        for e in rd.flatten() {
            let path = e.path();
            let ft = match e.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            let name = e.file_name().to_string_lossy().into_owned();
            if ft.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    walk(&path, out);
                }
            } else if ft.is_file() && name.ends_with(".gate") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

/// Resolve `GATE_FILE_OR_NAME`: an existing file path, else the unique
/// `.gate` file under the repository whose `gate:` equals the name (D2).
pub fn locate_gate(cwd: &Path, name: &str) -> Result<PathBuf, String> {
    let p = Path::new(name);
    let direct = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };
    if direct.is_file() {
        return fs::canonicalize(&direct).map_err(|e| format!("cannot resolve {name}: {e}"));
    }
    let root = repo_root(cwd)?;
    let mut found: Vec<PathBuf> = Vec::new();
    let mut unparsable = 0;
    for path in discover(&root) {
        match fs::read(&path) {
            Ok(bytes) => match manifest::parse(&bytes) {
                Ok(m) if m.gate == name => found.push(path),
                Ok(_) => {}
                Err(_) => unparsable += 1,
            },
            Err(_) => unparsable += 1,
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(format!(
            "no gate named {name} ({unparsable} unparsable gate files skipped)"
        )),
        _ => Err(format!(
            "duplicate gate id {name} in: {}",
            found
                .iter()
                .map(|f| f.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn collect_files(root: &Path, rel: &str, out: &mut BTreeSet<String>) {
    let path = root.join(rel.trim_end_matches('/'));
    let md = match fs::symlink_metadata(&path) {
        Ok(m) => m,
        Err(_) => {
            out.insert(rel.trim_end_matches('/').to_string());
            return;
        }
    };
    if md.is_dir() {
        if let Ok(rd) = fs::read_dir(&path) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name == ".git" {
                    continue;
                }
                let child = format!("{}/{}", rel.trim_end_matches('/'), name);
                collect_files(root, &child, out);
            }
        }
    } else if md.is_file() {
        out.insert(rel.trim_end_matches('/').to_string());
    }
}

/// Where `run.exec` is, for a run without the sandbox (the runner's PATH).
/// See [`exec_path_for`].
pub fn exec_path(root: &Path, exec: &str) -> Option<PathBuf> {
    exec_path_in(root, exec, std::env::var_os("PATH"))
}

/// Where `run.exec` is for a run under `sandbox`. A name with a slash is
/// repo-relative (the manifest already refuses absolute paths and `..`) and
/// must still be inside the repository after symlinks are followed. A bare
/// name is a program on the PATH, so a shared gate can say `exec: crumb`:
/// the runner's PATH without the sandbox, the exec helper's trusted PATH
/// ([`sandbox::TRUSTED_PATH`]) under `--sandbox user`, so the program hashed
/// into the receipt is the program that runs. None when it is not there.
pub fn exec_path_for(root: &Path, exec: &str, sandbox: Sandbox) -> Option<PathBuf> {
    match sandbox {
        Sandbox::None => exec_path(root, exec),
        Sandbox::User => exec_path_in(root, exec, Some(sandbox::TRUSTED_PATH.into())),
    }
}

fn exec_path_in(root: &Path, exec: &str, path: Option<std::ffi::OsString>) -> Option<PathBuf> {
    if exec.contains('/') {
        let p = root.join(exec);
        if !p.is_file() {
            return None;
        }
        // A symlink in the tree must not lead out of it.
        let inside = match (p.canonicalize(), root.canonicalize()) {
            (Ok(cp), Ok(cr)) => cp.starts_with(cr),
            _ => false,
        };
        return inside.then_some(p);
    }
    let path = path?;
    std::env::split_paths(&path)
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join(exec))
        .find(|p| p.is_file())
}

/// sha256 over the canonical list of (path, file sha256) for the explicit
/// `inputs` plus the D3 convention-implied inputs of `tests/<dir>/<name>.gate`.
/// A listed path that does not exist is recorded with a null digest. Compiler
/// identity is not included yet because nothing here builds.
pub fn build_inputs_digest(root: &Path, manifest_rel: Option<&Path>, m: &Manifest) -> String {
    let mut paths: BTreeSet<String> = BTreeSet::new();
    for i in &m.inputs {
        collect_files(root, i, &mut paths);
    }
    if let Some(rel) = manifest_rel {
        let dir = rel.parent().unwrap_or(Path::new(""));
        let stem = rel
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Ok(sub) = dir.strip_prefix("tests") {
            let sub = sub.to_string_lossy().into_owned();
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
                if let Ok(rd) = fs::read_dir(root.join(&base)) {
                    for e in rd.flatten() {
                        let p = e.path();
                        let fstem = p
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if p.is_file() && fstem == want {
                            paths.insert(format!("{base}/{}", e.file_name().to_string_lossy()));
                        }
                    }
                }
            }
        }
    }
    let entries: Vec<Value> = paths
        .iter()
        .map(|p| {
            let digest = fs::read(root.join(p)).ok().map(|b| sha256_hex(&b));
            json!({ "path": p, "sha256": digest })
        })
        .collect();
    canonical_digest(&json!({ "inputs": entries })).unwrap_or_default()
}

pub(crate) fn machine(hw: &Hardware) -> Value {
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|p| p.1.trim().to_string())
        .unwrap_or_default();
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(0);
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mem_kib = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    json!({
        "arch": std::env::consts::ARCH,
        "kernel_release": kernel,
        "cpu_model": cpu_model,
        "cpu_count": cpus,
        "mem_total_kib": mem_kib,
        "gb10_present": hw.gb10_present,
        "qemu_available": hw.qemu_available,
    })
}

pub(crate) fn captured_env() -> Value {
    let mut m = Map::new();
    for k in [
        "AIEN_PROOF_OFF",
        "CC",
        "CFLAGS",
        "CHIPRUN_SELFTEST",
        "LANG",
        "LC_ALL",
        "RUSTFLAGS",
        "TZ",
    ] {
        if let Ok(v) = std::env::var(k) {
            m.insert(k.to_string(), Value::String(v));
        }
    }
    Value::Object(m)
}

/// Whether the working tree has uncommitted changes (including untracked
/// files).
pub(crate) fn working_tree_dirty(root: &Path) -> Result<bool, String> {
    porcelain(root).map(|p| !p.is_empty())
}

/// The digests that identify one experiment besides its manifest and
/// dependencies: the program, the declared inputs, the machine.
pub(crate) struct Identity {
    pub binary_sha: Option<String>,
    pub inputs_digest: String,
    pub machine: Value,
    pub machine_digest: String,
}

pub(crate) fn identity(
    root: &Path,
    manifest_path: &Path,
    m: &Manifest,
    hw: &Hardware,
    sandbox: Sandbox,
) -> Result<Identity, crate::evidence::StoreError> {
    let binary_sha = exec_path_for(root, &m.exec, sandbox)
        .and_then(|p| fs::read(p).ok())
        .map(|b| sha256_hex(&b));
    let manifest_rel = manifest_path.strip_prefix(root).ok();
    let mut machine = machine(hw);
    // Under the user sandbox the exec helper is part of the machine a gate
    // ran on (network, mounts, limits): a result made under another helper
    // is not reused (SANDBOX lane: a FAIL made while the sandbox had no
    // loopback was reused after loopback came up).
    if sandbox == Sandbox::User {
        machine["sandbox_helper_sha256"] =
            json!(fs::read(sandbox::EXEC_HELPER).ok().map(|b| sha256_hex(&b)));
    }
    let machine_digest = canonical_digest(&machine)?;
    Ok(Identity {
        binary_sha,
        inputs_digest: build_inputs_digest(root, manifest_rel, m),
        machine,
        machine_digest,
    })
}
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn fatal<E: fmt::Display>(e: E) -> RunError {
    RunError::Fatal(e.to_string())
}

/// `core.dependencies`: gate to receipt digest, sorted by gate, only for the
/// dependencies that have a receipt for this commit.
fn dependency_rows(deps: &[DepFact]) -> BTreeMap<String, String> {
    deps.iter()
        .filter_map(|d| d.receipt.as_ref().map(|r| (d.gate.clone(), r.clone())))
        .collect()
}

/// Everything `execute_gate` needs besides the permit.
struct Job<'a> {
    root: &'a Path,
    manifest_path: &'a Path,
    m: &'a Manifest,
    opts: &'a Options,
    /// What is known about each `depends_on` gate right now.
    deps: Vec<DepFact>,
    /// How long the job waited for a pool slot.
    pool_wait_ms: u64,
}

/// Run one gate that has been parsed already. `permit` is the pool slot the
/// job holds (None for a lone `run`, and for gates that cannot run anyway); it
/// is released as soon as the job ends.
fn execute_gate(job: &Job<'_>, permit: Option<Permit<'_>>) -> Result<Outcome, RunError> {
    let (root, m, opts) = (job.root, job.m, job.opts);
    let started = now_secs();
    let t0 = Instant::now();

    let commit = head(root).map_err(RunError::Fatal)?;
    let before = porcelain(root).map_err(RunError::Fatal)?;
    let evidence_inside = resolve(&opts.evidence_dir).starts_with(resolve(root));
    let refusal: Option<String> = if evidence_inside {
        Some("evidence_inside".to_string())
    } else if !before.is_empty() && (!opts.allow_dirty || opts.release) {
        Some("dirty_tree".to_string())
    } else {
        None
    };

    // Resolved where it will run: a bare name under the user sandbox on the
    // helper's trusted PATH. The full path is what runs and what is hashed.
    let resolved = exec_path_for(root, &m.exec, opts.sandbox);
    let exec_exists = resolved.is_some();
    let exec_path = resolved.unwrap_or_else(|| root.join(&m.exec));
    let mut facts = Facts {
        refusal,
        exec_exists,
        deps: job.deps.clone(),
        gb10_present: opts.hardware.gb10_present,
        qemu_available: opts.hardware.qemu_available,
        ..Facts::default()
    };

    // Identify the experiment before anything runs, so the cache lookup and
    // the receipt agree on what was hashed (ADR Decision 7).
    let id = identity(root, job.manifest_path, m, &opts.hardware, opts.sandbox).map_err(fatal)?;
    let store = Store::new(&opts.evidence_dir);
    let tree_clean = before.is_empty();
    let mut mode = cache::policy(m.cache_allowed, opts.no_cache || opts.release);
    let off_reason: &str = if opts.release {
        "--release"
    } else if opts.no_cache {
        "--no-cache"
    } else if opts.args_override.is_some() {
        "replacement arguments were given after --"
    } else {
        "the tree has uncommitted changes"
    };
    if mode == cache::Mode::Allow && (opts.args_override.is_some() || !tree_clean) {
        mode = cache::Mode::Disabled;
    }
    let dependencies: Vec<(String, String)> = dependency_rows(&facts.deps).into_iter().collect();
    // Receipts are read when a lookup may happen, and to follow a reused
    // dependency back to the receipt it stands for.
    let index = if mode == cache::Mode::Allow || !dependencies.is_empty() {
        Index::load(&store)
    } else {
        Index::default()
    };
    let experiment = cache::Experiment {
        manifest_digest: m.digest.clone(),
        binary_sha256: id.binary_sha.clone(),
        build_inputs_digest: id.inputs_digest.clone(),
        machine_digest: id.machine_digest.clone(),
        dependencies,
        pins: opts.pins.clone(),
        args_override: opts.args_override.clone(),
    };
    let cache_key = cache::key(&experiment, &index).map_err(fatal)?;
    let mut note = match mode {
        cache::Mode::Never => CacheNote::Skipped("the manifest says cache: never".to_string()),
        cache::Mode::Disabled => CacheNote::Skipped(off_reason.to_string()),
        cache::Mode::Allow => {
            CacheNote::Skipped("the gate cannot run now, so there was nothing to reuse".to_string())
        }
    };
    // Reuse only when the gate would really have run: a refusal, a blocked
    // verdict or a dependency that is not PASS is reported as it is, never
    // hidden behind an old result.
    if mode == cache::Mode::Allow && !evidence_inside && verdict::pre_run(m, &facts).is_none() {
        match cache::lookup(&index, &store, &cache_key, &captured_env()) {
            cache::Lookup::Hit(src) => {
                let reused_verdict = src
                    .verdict()
                    .ok_or_else(|| fatal("a reusable receipt has no verdict"))?;
                let volatile = json!({
                    "started_utc": utc_string(started),
                    "finished_utc": utc_string(now_secs()),
                    "duration_ms": t0.elapsed().as_millis() as u64,
                    "pool": m.pool(),
                    "pool_wait_ms": job.pool_wait_ms,
                    "runner": { "name": "aien-test", "version": env!("CARGO_PKG_VERSION"), "impl": "rust" },
                    "cache": cache::cache_member(&cache_key, Some(&src.digest), mode),
                    "timeout_exceeded": false,
                });
                let receipt = build_receipt(cache::reused_core(src, &commit), volatile);
                let (digest, path) = store.put_receipt(&receipt).map_err(fatal)?;
                drop(permit);
                return Ok(Outcome {
                    verdict: reused_verdict,
                    reason: src.reason().to_string(),
                    receipt_path: Some(path),
                    receipt_digest: Some(digest),
                    cache: CacheNote::Reused {
                        from: src.digest.clone(),
                    },
                });
            }
            cache::Lookup::Miss(miss) => note = CacheNote::Miss(miss.text()),
        }
    }

    // A cargo gate holds its target dir for the whole run (build and run),
    // so no other gate builds in it or relinks a binary this one is
    // executing (sc fix/521). Taken after the pool slot and before the GPU
    // locks: the fixed order in `resources`. Waited for, never refused: the
    // holder is another gate or the worker's priming, both finite.
    let target_lock: Option<resources::TargetLock> = match &opts.cargo_target_dir {
        Some(dir) if m.uses_cargo() && verdict::pre_run(m, &facts).is_none() => {
            let path = resources::target_lock_path(&opts.target_lock_dir, dir);
            Some(resources::lock_target_dir(&path).map_err(|e| {
                RunError::Fatal(format!("cannot lock target dir {}: {e}", path.display()))
            })?)
        }
        _ => None,
    };

    // A gb10 gate takes the GPU locks only when nothing else already stops it.
    // A refusal (quiet flag, est_load, a held lock) becomes the gate's reason.
    let gpu: Option<GpuGuard> = if m.pool() == "gb10" && verdict::pre_run(m, &facts).is_none() {
        match acquire_gpu(&opts.gpu) {
            Ok(g) => Some(g),
            Err(r) => {
                facts.refusal = Some(r.reason().to_string());
                None
            }
        }
    } else {
        None
    };
    let will_run = verdict::pre_run(m, &facts).is_none();

    let mut stdout: Vec<u8> = Vec::new();
    let mut stderr: Vec<u8> = Vec::new();
    let mut after = before.clone();
    let mut timeout_exceeded = false;
    if will_run {
        let direct = RunSpec {
            program: exec_path.clone(),
            args: opts.args_override.clone().unwrap_or_else(|| m.args.clone()),
            cwd: root.to_path_buf(),
            timeout: Duration::from_millis(m.timeout_ms),
            kill_grace: opts.kill_grace,
            // gb10 jobs are never killed (ADR 0028 Decision 6): the timeout
            // only marks the receipt.
            enforce_timeout: m.pool() != "gb10",
        };
        // The sandbox seam (C8): `none` returns `direct` unchanged.
        let spec = sandbox::wrap(
            opts.sandbox,
            direct,
            m.pool(),
            &sandbox::env_pairs(opts.cargo_target_dir.as_deref()),
        );
        match process::run(&spec) {
            Ok(o) => {
                facts.exit_status = o.exit_code.map(i64::from);
                // Exit 2 with the helper's prefix is the helper refusing the
                // call (nothing ran): never a program result, so it can never
                // meet `expects.exit`, not even `exit: 2`.
                if sandbox_refused(opts.sandbox, o.exit_code, &o.stderr) {
                    facts.exit_status = None;
                }
                facts.timed_out = o.timed_out;
                timeout_exceeded = o.timeout_exceeded;
                stdout = o.stdout;
                stderr = o.stderr;
            }
            Err(e) => {
                stderr = format!("aien-test: cannot run {}: {e}\n", m.exec).into_bytes();
            }
        }
        // The job is over: close the lock descriptors at once.
        drop(gpu);
        drop(target_lock);
        after = porcelain(root).map_err(RunError::Fatal)?;
        let head_after = head(root).unwrap_or_default();
        facts.tree_changed_after = after != before;
        facts.head_moved = head_after != commit;
        facts.observations = parse_observations(&stdout);
    }
    drop(permit);
    let derived = derive(m, &facts);

    if evidence_inside {
        return Ok(Outcome {
            verdict: derived.verdict,
            reason: derived.reason,
            receipt_path: None,
            receipt_digest: None,
            cache: note,
        });
    }

    let stdout_sha = store.put_blob(&stdout).map_err(fatal)?;
    let stderr_sha = store.put_blob(&stderr).map_err(fatal)?;

    let observations: Vec<Value> = facts
        .observations
        .iter()
        .map(|o| json!({ "name": o.name, "value": o.value }))
        .collect();
    let rules: Vec<Value> = derived
        .rules
        .iter()
        .map(|(t, h)| json!({ "rule": t, "held": h }))
        .collect();
    let checks: Vec<Value> = m
        .checks
        .iter()
        .map(|c| {
            let passed = will_run && c == "clean_tree_after" && !facts.tree_changed_after;
            json!({ "name": c, "passed": passed })
        })
        .collect();
    let dependency_values: Vec<Value> = experiment
        .dependencies
        .iter()
        .map(|(gate, receipt)| json!({ "gate": gate, "receipt": receipt }))
        .collect();

    let mut core = Map::new();
    core.insert("gate".into(), json!(m.gate));
    core.insert("owner".into(), json!(m.owner));
    core.insert("manifest_digest".into(), json!(m.digest));
    core.insert("commit".into(), json!(commit));
    core.insert("pins".into(), opts.pins.clone());
    core.insert("tree_clean_before".into(), json!(before.is_empty()));
    core.insert("tree_clean_after".into(), json!(after.is_empty()));
    core.insert("commit_unchanged_after".into(), json!(!facts.head_moved));
    if let Some(b) = &id.binary_sha {
        core.insert("binary_sha256".into(), json!(b));
    }
    core.insert("build_inputs_digest".into(), json!(id.inputs_digest));
    core.insert("env".into(), captured_env());
    core.insert("machine".into(), id.machine.clone());
    core.insert("machine_digest".into(), json!(id.machine_digest));
    core.insert("exit_status".into(), json!(facts.exit_status));
    core.insert("stdout_sha256".into(), json!(stdout_sha));
    core.insert("stderr_sha256".into(), json!(stderr_sha));
    core.insert("observations".into(), Value::Array(observations));
    core.insert("rules".into(), Value::Array(rules));
    core.insert("derived_verdict".into(), json!(derived.verdict.as_str()));
    core.insert("reason".into(), json!(derived.reason));
    core.insert("mutants".into(), json!([]));
    core.insert("checks".into(), Value::Array(checks));
    core.insert("dependencies".into(), Value::Array(dependency_values));

    let volatile = json!({
        "started_utc": utc_string(started),
        "finished_utc": utc_string(now_secs()),
        "duration_ms": t0.elapsed().as_millis() as u64,
        "pool": m.pool(),
        "pool_wait_ms": job.pool_wait_ms,
        "runner": { "name": "aien-test", "version": env!("CARGO_PKG_VERSION"), "impl": "rust" },
        "cache": cache::cache_member(&cache_key, None, mode),
        "timeout_exceeded": timeout_exceeded,
    });

    let receipt = build_receipt(Value::Object(core), volatile);
    let (digest, path) = store.put_receipt(&receipt).map_err(fatal)?;
    Ok(Outcome {
        verdict: derived.verdict,
        reason: derived.reason,
        receipt_path: Some(path),
        receipt_digest: Some(digest),
        cache: note,
    })
}

/// Run one gate. Its dependencies are not started: their receipts for the
/// current commit are read from the evidence store, and a dependency without a
/// PASS receipt makes this gate NOT_RUN (ADR row 5).
pub fn run_gate(root: &Path, manifest_path: &Path, opts: &Options) -> Result<Outcome, RunError> {
    let bytes = fs::read(manifest_path)
        .map_err(|e| RunError::Fatal(format!("cannot read manifest: {e}")))?;
    let m = manifest::parse(&bytes).map_err(|e| RunError::BadManifest(e.0))?;
    let deps = if m.depends_on.is_empty() {
        Vec::new()
    } else {
        let commit = head(root).map_err(RunError::Fatal)?;
        Index::load(&Store::new(&opts.evidence_dir)).dep_facts(&m.depends_on, &commit)
    };
    let job = Job {
        root,
        manifest_path,
        m: &m,
        opts,
        deps,
        pool_wait_ms: 0,
    };
    execute_gate(&job, None)
}

/// The result of a graph run: one outcome per selected gate, in topological
/// order (a gate always comes after the gates it depends on).
#[derive(Debug, Clone)]
pub struct Campaign {
    pub results: Vec<(String, Outcome)>,
}

impl Campaign {
    /// ADR 0028 exit rule: any FAIL is FAIL, else the worst of the other
    /// non-PASS verdicts, else PASS.
    pub fn verdict(&self) -> Verdict {
        let all: Vec<Verdict> = self.results.iter().map(|(_, o)| o.verdict).collect();
        campaign_verdict(&all)
    }

    pub fn exit_code(&self) -> i32 {
        self.verdict().exit_code()
    }
}

/// What the scheduler shares between worker threads.
struct Sched {
    /// Selected gates whose dependencies have all finished and that have not
    /// started.
    ready: BTreeSet<usize>,
    /// Per gate: selected dependencies that have not finished yet.
    waiting_on: Vec<usize>,
    /// When each gate became ready (for `pool_wait_ms`).
    ready_at: Vec<Option<Instant>>,
    results: Vec<Option<Outcome>>,
    /// Selected gates that have not finished.
    remaining: usize,
    /// The first fatal error; it stops the campaign.
    error: Option<RunError>,
}

/// A gate chosen to start, with the facts about its dependencies and the pool
/// slot it holds (None when it cannot run, so it needs no slot).
struct Pick<'a> {
    node: usize,
    deps: Vec<DepFact>,
    permit: Option<Permit<'a>>,
}

struct GraphRun<'a> {
    root: &'a Path,
    graph: &'a Graph,
    selection: &'a BTreeSet<usize>,
    opts: &'a Options,
    pools: Pools,
    state: Mutex<Sched>,
    cv: Condvar,
    /// Facts about dependencies outside the selection, read from the store.
    outside: BTreeMap<String, DepFact>,
}

fn pool_of(m: &Manifest) -> Pool {
    Pool::from_name(m.pool()).unwrap_or(Pool::Host)
}

impl GraphRun<'_> {
    fn dep_facts(&self, results: &[Option<Outcome>], node: usize) -> Vec<DepFact> {
        self.graph
            .node(node)
            .manifest
            .depends_on
            .iter()
            .map(|id| {
                let finished = self.graph.index_of(id).and_then(|d| results[d].as_ref());
                match finished {
                    Some(o) => DepFact {
                        gate: id.clone(),
                        verdict: Some(o.verdict),
                        receipt: o.receipt_digest.clone(),
                    },
                    None => self.outside.get(id).cloned().unwrap_or_else(|| DepFact {
                        gate: id.clone(),
                        verdict: None,
                        receipt: None,
                    }),
                }
            })
            .collect()
    }

    /// Choose a ready gate that can start now. A gate whose dependencies are
    /// not all PASS cannot run, so it needs no pool slot and goes first; any
    /// other gate needs a free slot in its pool, and if there is none the
    /// next ready gate is tried (so a busy gb10 pool never holds up host
    /// work). Called with the scheduler lock held.
    fn pick(&self, st: &mut Sched) -> Option<Pick<'_>> {
        let mut chosen = None;
        for &i in &st.ready {
            let deps = self.dep_facts(&st.results, i);
            if !deps.iter().all(|d| d.verdict == Some(Verdict::Pass)) {
                chosen = Some(Pick {
                    node: i,
                    deps,
                    permit: None,
                });
                break;
            }
            let pool = pool_of(&self.graph.node(i).manifest);
            if let Some(permit) = self.pools.try_acquire(pool) {
                chosen = Some(Pick {
                    node: i,
                    deps,
                    permit: Some(permit),
                });
                break;
            }
        }
        if let Some(p) = &chosen {
            st.ready.remove(&p.node);
        }
        chosen
    }

    fn worker(&self) {
        loop {
            let (pick, waited_ms) = {
                let mut st = lock(&self.state);
                let pick = loop {
                    if st.remaining == 0 || st.error.is_some() {
                        return;
                    }
                    if let Some(p) = self.pick(&mut st) {
                        break p;
                    }
                    st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
                };
                let waited = st.ready_at[pick.node].map_or(0, |t| t.elapsed().as_millis() as u64);
                (pick, waited)
            };
            let i = pick.node;
            let node = self.graph.node(i);
            let job = Job {
                root: self.root,
                manifest_path: &node.path,
                m: &node.manifest,
                opts: self.opts,
                deps: pick.deps,
                pool_wait_ms: waited_ms,
            };
            let permit = pick.permit;
            let ran = catch_unwind(AssertUnwindSafe(|| execute_gate(&job, permit)));
            let outcome = match ran {
                Ok(r) => r,
                Err(_) => Err(RunError::Fatal(format!(
                    "internal error while running gate {}",
                    node.id
                ))),
            };
            let mut st = lock(&self.state);
            match outcome {
                Ok(o) => {
                    st.results[i] = Some(o);
                    st.remaining = st.remaining.saturating_sub(1);
                    for &d in self.graph.direct_dependents(i) {
                        if self.selection.contains(&d) {
                            st.waiting_on[d] = st.waiting_on[d].saturating_sub(1);
                            if st.waiting_on[d] == 0 {
                                st.ready.insert(d);
                                st.ready_at[d] = Some(Instant::now());
                            }
                        }
                    }
                }
                Err(e) => {
                    if st.error.is_none() {
                        st.error = Some(e);
                    }
                }
            }
            drop(st);
            // The permit went back inside `execute_gate`, so waiting workers
            // can use the slot as soon as they wake.
            self.cv.notify_all();
        }
    }
}

/// Run the gates in `selection` (node numbers of `graph`) with worker threads.
/// A gate starts when every selected gate it depends on has finished and its
/// pool has a free slot (host, qemu, or gb10 with the GPU locks). A gate whose
/// dependency did not PASS is not started: it gets a receipt saying NOT_RUN,
/// or the blocked verdict of the dependency (ADR rows 4 and 5). `selection`
/// should include the dependencies of its members; one that is left out is
/// looked up in the evidence store for the current commit.
pub fn run_graph(
    root: &Path,
    graph: &Graph,
    selection: &BTreeSet<usize>,
    opts: &Options,
) -> Result<Campaign, RunError> {
    if selection.is_empty() {
        return Ok(Campaign {
            results: Vec::new(),
        });
    }
    let commit = head(root).map_err(RunError::Fatal)?;
    let index = Index::load(&Store::new(&opts.evidence_dir));
    let start = Instant::now();

    let mut outside: BTreeMap<String, DepFact> = BTreeMap::new();
    let mut waiting_on = vec![0usize; graph.len()];
    let mut ready: BTreeSet<usize> = BTreeSet::new();
    let mut ready_at: Vec<Option<Instant>> = vec![None; graph.len()];
    for &i in selection {
        for &d in &graph.node(i).deps {
            if selection.contains(&d) {
                waiting_on[i] += 1;
            } else {
                let id = graph.node(d).id.clone();
                let facts = index.dep_facts(std::slice::from_ref(&id), &commit);
                if let Some(f) = facts.into_iter().next() {
                    outside.insert(id, f);
                }
            }
        }
        if waiting_on[i] == 0 {
            ready.insert(i);
            ready_at[i] = Some(start);
        }
    }

    let run = GraphRun {
        root,
        graph,
        selection,
        opts,
        pools: Pools::new(opts.pool_sizes),
        state: Mutex::new(Sched {
            ready,
            waiting_on,
            ready_at,
            results: vec![None; graph.len()],
            remaining: selection.len(),
            error: None,
        }),
        cv: Condvar::new(),
        outside,
    };

    // Enough workers for every slot of every pool to be busy at once.
    let workers = (opts.pool_sizes.host + opts.pool_sizes.qemu + 1)
        .min(selection.len())
        .max(1);
    thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| run.worker());
        }
    });

    let mut st = lock(&run.state);
    if let Some(e) = st.error.take() {
        return Err(e);
    }
    let mut results = Vec::with_capacity(selection.len());
    for &i in graph.order() {
        if !selection.contains(&i) {
            continue;
        }
        let id = &graph.node(i).id;
        match st.results[i].take() {
            Some(o) => results.push((id.clone(), o)),
            None => return Err(RunError::Fatal(format!("no result for gate {id}"))),
        }
    }
    Ok(Campaign { results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{git, init_repo, write_file, TempDir};
    use std::os::unix::fs::PermissionsExt;

    const OK_SCRIPT: &str = "#!/bin/sh\ntouch \"$1\"\necho \"AIEN-OBS n 3\"\necho PASS\n";

    fn gate_text(
        exec: &str,
        args: &[&str],
        exit: i32,
        observe: &[&str],
        timeout: &str,
        extra: &str,
    ) -> String {
        format!(
            "gate: T-1\nmanifest_version: 1\nowner: t\nrequires:\n  - host\nrun:\n  exec: {exec}\n  args: {}\nexpects:\n  exit: {exit}\n  observe: {}\n  verdict: PASS\ntimeout: {timeout}\n{extra}",
            serde_json::to_string(args).unwrap(),
            serde_json::to_string(observe).unwrap(),
        )
    }

    struct Fx {
        repo: TempDir,
        ev: TempDir,
        scratch: TempDir,
    }

    impl Fx {
        fn marker(&self) -> PathBuf {
            self.scratch.path().join("marker")
        }
        fn manifest(&self) -> PathBuf {
            self.repo.path().join("tests/t.gate")
        }
        fn opts(&self) -> Options {
            let mut o = Options::new(self.ev.path().join("out"));
            o.kill_grace = Duration::from_millis(300);
            o.hardware = Hardware {
                gb10_present: false,
                qemu_available: false,
            };
            o.gpu = GpuConfig::rooted_at(self.scratch.path());
            o
        }
    }

    /// A repo with `tools/run.sh` = script and a gate whose args are the
    /// marker path.
    fn fixture(script: &str, exit: i32, observe: &[&str], timeout: &str, extra: &str) -> Fx {
        let scratch = TempDir::new("scratch");
        let marker = scratch.path().join("marker");
        let gate = gate_text(
            "tools/run.sh",
            &[marker.to_str().unwrap()],
            exit,
            observe,
            timeout,
            extra,
        );
        let repo = init_repo(
            "repo",
            &[
                ("tools/run.sh", script, true),
                ("tests/t.gate", gate.as_str(), false),
                ("README", "hello\n", false),
            ],
        );
        Fx {
            repo,
            ev: TempDir::new("ev"),
            scratch,
        }
    }

    fn ok_fixture() -> Fx {
        fixture(OK_SCRIPT, 0, &["n == 3"], "20s", "")
    }

    fn read_json(p: &Path) -> Value {
        serde_json::from_slice(&fs::read(p).unwrap()).unwrap()
    }

    fn run(fx: &Fx, opts: &Options) -> Outcome {
        run_gate(fx.repo.path(), &fx.manifest(), opts).unwrap()
    }

    #[test]
    fn pass_writes_a_bound_receipt() {
        let fx = ok_fixture();
        let o = run(&fx, &fx.opts());
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert!(fx.marker().exists(), "script must have run");
        let path = o.receipt_path.unwrap();
        let r = read_json(&path);
        assert_eq!(r["schema"], "aien-test/EvidenceReceiptV1");
        let core = &r["core"];
        let head_now = git_out(fx.repo.path(), &["rev-parse", "HEAD"]).unwrap();
        assert_eq!(core["commit"], head_now.trim());
        assert_eq!(core["gate"], "T-1");
        assert_eq!(core["derived_verdict"], "PASS");
        assert_eq!(core["reason"], "");
        assert_eq!(core["exit_status"], 0);
        assert_eq!(core["tree_clean_before"], true);
        assert_eq!(core["tree_clean_after"], true);
        assert_eq!(core["commit_unchanged_after"], true);
        assert_eq!(core["observations"][0]["name"], "n");
        assert_eq!(core["observations"][0]["value"], 3);
        assert_eq!(core["rules"][0]["held"], true);
        let script_bytes = fs::read(fx.repo.path().join("tools/run.sh")).unwrap();
        assert_eq!(core["binary_sha256"], sha256_hex(&script_bytes));
        let gate_bytes = fs::read(fx.manifest()).unwrap();
        assert_eq!(core["manifest_digest"], sha256_hex(&gate_bytes));
        assert_eq!(core["dependencies"], json!([]));
        assert_eq!(r["volatile"]["pool"], "host");
        assert_eq!(r["volatile"]["timeout_exceeded"], false);
        // Receipt is read-only and named by its digest.
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o444
        );
        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            format!("{}.json", o.receipt_digest.unwrap())
        );
        // stdout blob holds the script output verbatim.
        let sha = core["stdout_sha256"].as_str().unwrap();
        let blob = fx.ev.path().join("out/blobs").join(format!("{sha}.log"));
        assert_eq!(fs::read(blob).unwrap(), b"AIEN-OBS n 3\nPASS\n");
    }

    #[test]
    fn dirty_tree_is_refused_and_nothing_runs() {
        let fx = ok_fixture();
        write_file(fx.repo.path(), "README", "changed\n", false);
        let o = run(&fx, &fx.opts());
        assert_eq!(o.verdict, Verdict::NotRun);
        assert_eq!(o.reason, "dirty_tree");
        assert!(!fx.marker().exists(), "script must not run on a dirty tree");
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["tree_clean_before"], false);
        assert_eq!(r["core"]["exit_status"], Value::Null);
    }

    #[test]
    fn untracked_file_counts_as_dirty() {
        let fx = ok_fixture();
        write_file(fx.repo.path(), "stray.txt", "x", false);
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "dirty_tree")
        );
        assert!(!fx.marker().exists());
    }

    #[test]
    fn allow_dirty_runs_and_records_it() {
        let fx = ok_fixture();
        write_file(fx.repo.path(), "README", "changed\n", false);
        let mut opts = fx.opts();
        opts.allow_dirty = true;
        let o = run(&fx, &opts);
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert!(fx.marker().exists());
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["tree_clean_before"], false);
    }

    #[test]
    fn timeout_is_enforced_with_fail() {
        let fx = fixture("#!/bin/sh\nsleep 20\n", 0, &[], "300ms", "");
        let t = Instant::now();
        let o = run(&fx, &fx.opts());
        assert!(t.elapsed() < Duration::from_secs(10));
        assert_eq!((o.verdict, o.reason.as_str()), (Verdict::Fail, "timeout"));
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["volatile"]["timeout_exceeded"], true);
    }

    #[test]
    fn rule_mismatch_fails_even_if_the_script_prints_pass() {
        let fx = fixture(
            "#!/bin/sh\necho \"AIEN-OBS n 4\"\necho PASS\necho 'AIEN-OBS verdict \"PASS\"'\n",
            0,
            &["n == 3"],
            "20s",
            "",
        );
        let o = run(&fx, &fx.opts());
        assert_eq!((o.verdict, o.reason.as_str()), (Verdict::Fail, "rule:0"));
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["rules"][0]["held"], false);
    }

    #[test]
    fn exit_status_must_match() {
        let fx = fixture(
            "#!/bin/sh\necho \"AIEN-OBS n 3\"\nexit 2\n",
            0,
            &["n == 3"],
            "20s",
            "",
        );
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::Fail, "rc_nonzero")
        );
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["exit_status"], 2);
    }

    #[test]
    fn nonzero_expected_exit_passes() {
        let fx = fixture("#!/bin/sh\nexit 7\n", 7, &[], "20s", "");
        assert_eq!(run(&fx, &fx.opts()).verdict, Verdict::Pass);
    }

    #[test]
    fn missing_exec_is_missing_implementation() {
        let fx = ok_fixture();
        fs::remove_file(fx.repo.path().join("tools/run.sh")).unwrap();
        git(fx.repo.path(), &["add", "-A"]);
        git(fx.repo.path(), &["commit", "-q", "-m", "rm"]);
        let o = run(&fx, &fx.opts());
        assert_eq!(o.verdict, Verdict::MissingImplementation);
        assert_eq!(o.reason, "NO_RUN_EXEC");
        let r = read_json(&o.receipt_path.unwrap());
        assert!(r["core"].get("binary_sha256").is_none());
    }

    #[test]
    fn evidence_dir_inside_repo_is_refused() {
        let fx = ok_fixture();
        let mut opts = fx.opts();
        opts.evidence_dir = fx.repo.path().join("evidence");
        let o = run(&fx, &opts);
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "evidence_inside")
        );
        assert!(o.receipt_path.is_none());
        assert!(!fx.repo.path().join("evidence").exists());
        assert!(!fx.marker().exists());
    }

    #[test]
    fn tree_dirtied_by_the_run_fails() {
        let fx = fixture("#!/bin/sh\necho more >> README\n", 0, &[], "20s", "");
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::Fail, "dirty_after")
        );
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["tree_clean_after"], false);
    }

    #[test]
    fn head_moving_during_the_run_fails() {
        let fx = fixture(
            "#!/bin/sh\ngit -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false commit -q --allow-empty -m moved\n",
            0,
            &[],
            "20s",
            "",
        );
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::Fail, "head_moved")
        );
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["commit_unchanged_after"], false);
    }

    #[test]
    fn args_after_double_dash_replace_manifest_args() {
        let fx = ok_fixture();
        let other = fx.scratch.path().join("other");
        let mut opts = fx.opts();
        opts.args_override = Some(vec![other.to_str().unwrap().to_string()]);
        let o = run(&fx, &opts);
        assert_eq!(o.verdict, Verdict::Pass);
        assert!(other.exists());
        assert!(!fx.marker().exists());
    }

    #[test]
    fn bad_manifest_writes_no_receipt() {
        let fx = ok_fixture();
        write_file(fx.repo.path(), "tests/t.gate", "gate: X\nbogus: 1\n", false);
        match run_gate(fx.repo.path(), &fx.manifest(), &fx.opts()) {
            Err(RunError::BadManifest(_)) => {}
            other => panic!("expected BadManifest, got {other:?}"),
        }
        assert!(!fx.ev.path().join("out").exists());
    }

    #[test]
    fn operator_and_missing_hardware_gates_do_not_run() {
        let fx = ok_fixture();
        let gate = fs::read_to_string(fx.manifest()).unwrap();
        write_file(
            fx.repo.path(),
            "tests/t.gate",
            &gate.replace("  - host\n", "  - host\n  - operator\n"),
            false,
        );
        git(fx.repo.path(), &["add", "-A"]);
        git(fx.repo.path(), &["commit", "-q", "-m", "op"]);
        let o = run(&fx, &fx.opts());
        assert_eq!(o.verdict, Verdict::BlockedOperator);
        assert!(!fx.marker().exists());
        write_file(
            fx.repo.path(),
            "tests/t.gate",
            &gate.replace("  - host\n", "  - gb10\n"),
            false,
        );
        git(fx.repo.path(), &["add", "-A"]);
        git(fx.repo.path(), &["commit", "-q", "-m", "gb10"]);
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::BlockedHardware, "NO_GB10")
        );
        assert!(!fx.marker().exists());
    }

    #[test]
    fn unfinished_receipts_are_never_modified_by_a_second_run() {
        let fx = ok_fixture();
        let first = run(&fx, &fx.opts()).receipt_path.unwrap();
        let before = fs::read(&first).unwrap();
        let second = run(&fx, &fx.opts());
        assert_eq!(second.verdict, Verdict::Pass);
        assert_eq!(fs::read(&first).unwrap(), before);
    }

    #[test]
    fn the_exec_helper_is_part_of_the_machine_only_under_the_user_sandbox() {
        let fx = ok_fixture();
        let m = manifest::parse(&fs::read(fx.manifest()).unwrap()).unwrap();
        let hw = fx.opts().hardware;
        let none = identity(fx.repo.path(), &fx.manifest(), &m, &hw, Sandbox::None).unwrap();
        let user = identity(fx.repo.path(), &fx.manifest(), &m, &hw, Sandbox::User).unwrap();
        assert!(none.machine.get("sandbox_helper_sha256").is_none());
        let want = fs::read(sandbox::EXEC_HELPER).ok().map(|b| sha256_hex(&b));
        assert_eq!(user.machine["sandbox_helper_sha256"], json!(want));
        // Another helper is another machine: no cache entry carries over.
        assert_ne!(none.machine_digest, user.machine_digest);
    }

    #[test]
    fn inputs_digest_tracks_input_bytes() {
        let fx = ok_fixture();
        let m = manifest::parse(&fs::read(fx.manifest()).unwrap()).unwrap();
        let mut m2 = m.clone();
        m2.inputs = vec!["README".to_string()];
        let a = build_inputs_digest(fx.repo.path(), None, &m2);
        assert_eq!(a, build_inputs_digest(fx.repo.path(), None, &m2));
        write_file(fx.repo.path(), "README", "different\n", false);
        let b = build_inputs_digest(fx.repo.path(), None, &m2);
        assert_ne!(a, b);
    }

    #[test]
    fn convention_inputs_are_found() {
        let repo = init_repo(
            "conv",
            &[
                ("src/rt/world.c", "int x;", false),
                ("tests/rt/world_test.c", "int t;", false),
                ("tests/rt/other.c", "int o;", false),
            ],
        );
        let m = manifest::parse(gate_text("tools/x.sh", &[], 0, &[], "5s", "").as_bytes()).unwrap();
        let rel = Path::new("tests/rt/world.gate");
        let a = build_inputs_digest(repo.path(), Some(rel), &m);
        write_file(repo.path(), "src/rt/world.c", "int y;", false);
        let b = build_inputs_digest(repo.path(), Some(rel), &m);
        assert_ne!(a, b, "implied src input must feed the digest");
        write_file(repo.path(), "tests/rt/other.c", "int p;", false);
        let c = build_inputs_digest(repo.path(), Some(rel), &m);
        assert_eq!(b, c, "unrelated file must not feed the digest");
    }

    #[test]
    fn locate_by_path_name_and_duplicates() {
        let g1 = gate_text("tools/x.sh", &[], 0, &[], "5s", "").replace("T-1", "ONE");
        let g2 = gate_text("tools/x.sh", &[], 0, &[], "5s", "").replace("T-1", "TWO");
        let repo = init_repo(
            "loc",
            &[
                ("tests/a/one.gate", g1.as_str(), false),
                ("tests/b/two.gate", g2.as_str(), false),
                ("target/skipped.gate", g1.as_str(), false),
                ("tests/bad.gate", "nonsense\n", false),
            ],
        );
        let p = locate_gate(repo.path(), "ONE").unwrap();
        assert!(p.ends_with("tests/a/one.gate"), "{p:?}");
        let p = locate_gate(repo.path(), "tests/b/two.gate").unwrap();
        assert!(p.ends_with("tests/b/two.gate"));
        assert!(locate_gate(repo.path(), "NOPE").is_err());
        write_file(repo.path(), "tests/c/dup.gate", g1.as_str(), false);
        let e = locate_gate(repo.path(), "ONE").unwrap_err();
        assert!(e.contains("duplicate"), "{e}");
    }

    #[test]
    fn ignore_marker_hides_a_directory() {
        let g = gate_text("tools/x.sh", &[], 0, &[], "5s", "");
        let repo = init_repo(
            "ign",
            &[
                ("tests/a.gate", g.as_str(), false),
                ("tests/skip/b.gate", g.as_str(), false),
                ("tests/skip/.aien-test-ignore", "", false),
            ],
        );
        let found = discover(repo.path());
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("tests/a.gate"));
    }

    #[test]
    fn resolve_handles_missing_dirs_and_dotdot() {
        let t = TempDir::new("res");
        let p = resolve(&t.path().join("a/b/../c"));
        assert_eq!(p, t.path().join("a/c"));
    }

    // ---- Slice B: pools, dependencies, graph runs ----

    fn gb10_fixture(script: &str, timeout: &str) -> Fx {
        let fx = fixture(script, 0, &[], timeout, "");
        let gate = fs::read_to_string(fx.manifest()).unwrap();
        write_file(
            fx.repo.path(),
            "tests/t.gate",
            &gate.replace("  - host\n", "  - gb10\n"),
            false,
        );
        git(fx.repo.path(), &["add", "-A"]);
        git(fx.repo.path(), &["commit", "-q", "-m", "gb10"]);
        fx
    }

    fn gb10_opts(fx: &Fx) -> Options {
        let mut o = fx.opts();
        o.hardware.gb10_present = true;
        o
    }

    const TOUCH_MARKER: &str = "#!/bin/sh\ntouch \"$1\"\n";

    #[test]
    fn gb10_gate_runs_under_the_gpu_lock_and_records_its_pool() {
        let fx = gb10_fixture(TOUCH_MARKER, "20s");
        let opts = gb10_opts(&fx);
        let o = run(&fx, &opts);
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert!(fx.marker().exists());
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["volatile"]["pool"], "gb10");
        assert_eq!(r["volatile"]["timeout_exceeded"], false);
        assert_eq!(r["core"]["machine"]["gb10_present"], true);
        assert!(opts.gpu.lock_path.exists(), "the lock file is created");
        assert!(
            acquire_gpu(&opts.gpu).is_ok(),
            "the lock is released after the job"
        );
    }

    #[test]
    fn gb10_gate_without_the_hardware_is_blocked() {
        let fx = gb10_fixture(TOUCH_MARKER, "20s");
        let o = run(&fx, &fx.opts());
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::BlockedHardware, "NO_GB10")
        );
        assert!(!fx.marker().exists());
    }

    #[test]
    fn gb10_gate_is_refused_while_the_quiet_flag_exists() {
        let fx = gb10_fixture(TOUCH_MARKER, "20s");
        let opts = gb10_opts(&fx);
        let flag = opts.gpu.quiet_flag.clone().unwrap();
        fs::write(&flag, "hold: testing\n").unwrap();
        let o = run(&fx, &opts);
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "quiet_flag")
        );
        assert!(!fx.marker().exists());
        assert_eq!(fs::read_to_string(&flag).unwrap(), "hold: testing\n");
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["reason"], "quiet_flag");
    }

    #[test]
    fn gb10_gate_is_refused_while_another_holder_has_the_lock() {
        let fx = gb10_fixture(TOUCH_MARKER, "20s");
        let opts = gb10_opts(&fx);
        let held = acquire_gpu(&opts.gpu).unwrap();
        let o = run(&fx, &opts);
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "gpu_lock")
        );
        assert!(!fx.marker().exists(), "a refused job must not run");
        drop(held);
        let o = run(&fx, &opts);
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert!(fx.marker().exists());
    }

    #[test]
    fn gb10_timeout_is_advisory_and_never_kills_the_job() {
        let fx = gb10_fixture("#!/bin/sh\nsleep 1\ntouch \"$1\"\n", "300ms");
        let o = run(&fx, &gb10_opts(&fx));
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert!(fx.marker().exists(), "the job must be left to finish");
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["volatile"]["timeout_exceeded"], true);
    }

    /// `(gate id, depends_on ids separated by spaces, requires, script)`.
    type GateSpec = (&'static str, &'static str, &'static str, String);

    fn graph_gate_text(id: &str, deps: &str, requires: &str) -> String {
        let dep_block = if deps.trim().is_empty() {
            String::new()
        } else {
            format!(
                "depends_on:\n{}",
                deps.split_whitespace()
                    .map(|d| format!("  - {d}\n"))
                    .collect::<String>()
            )
        };
        format!(
            "gate: {id}\nmanifest_version: 1\nowner: t\nrequires:\n  - {requires}\nrun:\n  exec: tools/{id}.sh\n  args: []\nexpects:\n  exit: 0\n  observe: []\n  verdict: PASS\ntimeout: 20s\n{dep_block}"
        )
    }

    /// Script that appends `ran-<id>` to the shared log and exits with `exit`.
    fn logging(id: &str, exit: i32) -> String {
        format!("#!/bin/sh\necho ran-{id} >> \"@S@/log\"\nexit {exit}\n")
    }

    /// Script that logs `start-<id>`, takes a moment, then logs `end-<id>`.
    fn slow(id: &str) -> String {
        format!(
            "#!/bin/sh\necho start-{id} >> \"@S@/log\"\nsleep 0.2\necho end-{id} >> \"@S@/log\"\n"
        )
    }

    /// Script that proves another job is running at the same time: it waits
    /// (at most about ten seconds) for the other script's start marker.
    fn rendezvous(me: &str, other: &str) -> String {
        format!(
            "#!/bin/sh\ntouch \"@S@/started-{me}\"\ni=0\nwhile [ ! -f \"@S@/started-{other}\" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done\nif [ -f \"@S@/started-{other}\" ]; then touch \"@S@/saw-{me}\"; fi\n"
        )
    }

    struct Gfx {
        repo: TempDir,
        ev: TempDir,
        scratch: TempDir,
    }

    /// A repository with one gate (tests/<id>.gate) and script
    /// (tools/<id>.sh) per entry; `@S@` in a script is the scratch directory.
    fn gfx(gates: &[GateSpec]) -> Gfx {
        gfx_with(gates, "", &[])
    }

    /// `inputs` block that marks a gate as a cargo gate (it builds in the
    /// target dir): the rule `Manifest::uses_cargo` looks for.
    const CARGO_INPUTS: &str = "inputs:\n  - Cargo.lock\n";

    /// `gfx` with `extra` appended to every manifest and `files` added to
    /// the repository (a `Cargo.lock` for the cargo gates).
    fn gfx_with(gates: &[GateSpec], extra: &str, more: &[(&str, &str)]) -> Gfx {
        let scratch = TempDir::new("gscratch");
        let s = scratch.path().to_str().unwrap().to_string();
        let mut files: Vec<(String, String, bool)> = Vec::new();
        for (id, deps, requires, script) in gates {
            files.push((format!("tools/{id}.sh"), script.replace("@S@", &s), true));
            files.push((
                format!("tests/{id}.gate"),
                format!("{}{extra}", graph_gate_text(id, deps, requires)),
                false,
            ));
        }
        for (p, c) in more {
            files.push((p.to_string(), c.to_string(), false));
        }
        let refs: Vec<(&str, &str, bool)> = files
            .iter()
            .map(|(p, c, x)| (p.as_str(), c.as_str(), *x))
            .collect();
        let repo = init_repo("grepo", &refs);
        Gfx {
            repo,
            ev: TempDir::new("gev"),
            scratch,
        }
    }

    impl Gfx {
        fn gate_path(&self, id: &str) -> PathBuf {
            self.repo.path().join(format!("tests/{id}.gate"))
        }

        fn opts(&self) -> Options {
            let mut o = Options::new(self.ev.path().join("out"));
            o.kill_grace = Duration::from_millis(300);
            o.hardware = Hardware {
                gb10_present: false,
                qemu_available: false,
            };
            o.gpu = GpuConfig::rooted_at(self.scratch.path());
            o.pool_sizes = PoolSizes { host: 2, qemu: 1 };
            o
        }

        fn graph(&self) -> Graph {
            crate::graph::load(self.repo.path()).unwrap()
        }

        fn campaign(&self, opts: &Options) -> Campaign {
            let g = self.graph();
            let all: BTreeSet<usize> = (0..g.len()).collect();
            run_graph(self.repo.path(), &g, &all, opts).unwrap()
        }

        fn log(&self) -> Vec<String> {
            fs::read_to_string(self.scratch.path().join("log"))
                .unwrap_or_default()
                .lines()
                .map(|l| l.to_string())
                .collect()
        }
    }

    fn outcome<'a>(c: &'a Campaign, id: &str) -> &'a Outcome {
        &c.results
            .iter()
            .find(|(g, _)| g == id)
            .expect("gate is in the campaign")
            .1
    }

    fn verdict_and_reason<'a>(c: &'a Campaign, id: &str) -> (Verdict, &'a str) {
        let o = outcome(c, id);
        (o.verdict, o.reason.as_str())
    }

    #[test]
    fn a_chain_runs_in_order_and_each_receipt_names_its_dependencies() {
        let fx = gfx(&[
            ("G-A", "", "host", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
            ("G-C", "G-B", "host", logging("G-C", 0)),
        ]);
        let c = fx.campaign(&fx.opts());
        assert_eq!(c.exit_code(), 0);
        let ids: Vec<&str> = c.results.iter().map(|(g, _)| g.as_str()).collect();
        assert_eq!(ids, vec!["G-A", "G-B", "G-C"]);
        assert!(c.results.iter().all(|(_, o)| o.verdict == Verdict::Pass));
        assert_eq!(fx.log(), vec!["ran-G-A", "ran-G-B", "ran-G-C"]);
        let a = outcome(&c, "G-A").receipt_digest.clone().unwrap();
        let ra = read_json(outcome(&c, "G-A").receipt_path.as_ref().unwrap());
        let rb = read_json(outcome(&c, "G-B").receipt_path.as_ref().unwrap());
        assert_eq!(ra["core"]["dependencies"], json!([]));
        assert_eq!(
            rb["core"]["dependencies"],
            json!([{ "gate": "G-A", "receipt": a }])
        );
        assert!(rb["volatile"]["pool_wait_ms"].is_u64());
    }

    #[test]
    fn a_failed_dependency_makes_dependents_not_run_with_the_reason_recorded() {
        let fx = gfx(&[
            ("G-A", "", "host", logging("G-A", 1)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
            ("G-C", "G-B", "host", logging("G-C", 0)),
            ("G-D", "", "host", logging("G-D", 0)),
        ]);
        let c = fx.campaign(&fx.opts());
        assert_eq!(verdict_and_reason(&c, "G-A"), (Verdict::Fail, "rc_nonzero"));
        assert_eq!(
            verdict_and_reason(&c, "G-B"),
            (Verdict::NotRun, "DEP_NOT_PASS:G-A")
        );
        assert_eq!(
            verdict_and_reason(&c, "G-C"),
            (Verdict::NotRun, "DEP_NOT_PASS:G-B")
        );
        assert_eq!(verdict_and_reason(&c, "G-D"), (Verdict::Pass, ""));
        assert_eq!(c.exit_code(), 1);
        let mut log = fx.log();
        log.sort();
        assert_eq!(log, vec!["ran-G-A", "ran-G-D"], "B and C must not run");
        let a = outcome(&c, "G-A").receipt_digest.clone().unwrap();
        let rb = read_json(outcome(&c, "G-B").receipt_path.as_ref().unwrap());
        assert_eq!(rb["core"]["derived_verdict"], "NOT_RUN");
        assert_eq!(rb["core"]["reason"], "DEP_NOT_PASS:G-A");
        assert_eq!(rb["core"]["exit_status"], Value::Null);
        assert_eq!(
            rb["core"]["dependencies"],
            json!([{ "gate": "G-A", "receipt": a }])
        );
    }

    #[test]
    fn a_blocked_dependency_blocks_its_dependents_the_same_way() {
        let fx = gfx(&[
            ("G-A", "", "gb10", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
            ("G-C", "G-B", "host", logging("G-C", 0)),
        ]);
        let c = fx.campaign(&fx.opts());
        assert_eq!(
            verdict_and_reason(&c, "G-A"),
            (Verdict::BlockedHardware, "NO_GB10")
        );
        assert_eq!(
            verdict_and_reason(&c, "G-B"),
            (Verdict::BlockedHardware, "DEP_BLOCKED:G-A")
        );
        assert_eq!(
            verdict_and_reason(&c, "G-C"),
            (Verdict::BlockedHardware, "DEP_BLOCKED:G-B")
        );
        assert!(fx.log().is_empty(), "nothing may run");
        assert_eq!(c.exit_code(), 3);
    }

    #[test]
    fn independent_gates_run_at_the_same_time_when_the_pool_allows() {
        let fx = gfx(&[
            ("G-A", "", "host", rendezvous("a", "b")),
            ("G-B", "", "host", rendezvous("b", "a")),
        ]);
        let c = fx.campaign(&fx.opts());
        assert_eq!(c.exit_code(), 0);
        assert!(fx.scratch.path().join("saw-a").exists(), "A never saw B");
        assert!(fx.scratch.path().join("saw-b").exists(), "B never saw A");
    }

    /// Two cargo gates of one head share `CARGO_TARGET_DIR`. Run at the same
    /// time, one relinks a test binary the other is executing (sc fix/521:
    /// SC-AGENT-PREFLIGHT relinked `aien_inference_abi-*` under SC-TEST, which
    /// then failed with `current_exe: NotFound`). The runner must hold the
    /// target dir for one cargo gate at a time, whatever the pool allows.
    #[test]
    fn cargo_gates_sharing_a_target_dir_never_overlap() {
        let fx = gfx_with(
            &[
                ("C-A", "", "host", slow("C-A")),
                ("C-B", "", "host", slow("C-B")),
            ],
            CARGO_INPUTS,
            &[("Cargo.lock", "# lock\n")],
        );
        let mut opts = fx.opts();
        opts.cargo_target_dir = Some(fx.scratch.path().join("target"));
        opts.target_lock_dir = fx.scratch.path().join("locks");
        let c = fx.campaign(&opts);
        assert_eq!(c.exit_code(), 0);
        let log = fx.log();
        assert_eq!(log.len(), 4, "{log:?}");
        let first = log[0].trim_start_matches("start-").to_string();
        let second = if first == "C-A" { "C-B" } else { "C-A" };
        assert_eq!(
            log,
            vec![
                format!("start-{first}"),
                format!("end-{first}"),
                format!("start-{second}"),
                format!("end-{second}"),
            ],
            "the two cargo gates overlapped in one target dir"
        );
        let lock = crate::resources::target_lock_path(
            &opts.target_lock_dir,
            &fx.scratch.path().join("target"),
        );
        assert!(
            lock.exists(),
            "the lock file {} was never taken",
            lock.display()
        );
    }

    /// The lock is per target dir and only for gates that build in it: gates
    /// without cargo inputs stay parallel even when a target dir is set.
    #[test]
    fn non_cargo_gates_stay_parallel_when_a_target_dir_is_set() {
        let fx = gfx(&[
            ("G-A", "", "host", rendezvous("a", "b")),
            ("G-B", "", "host", rendezvous("b", "a")),
        ]);
        let mut opts = fx.opts();
        opts.cargo_target_dir = Some(fx.scratch.path().join("target"));
        opts.target_lock_dir = fx.scratch.path().join("locks");
        let c = fx.campaign(&opts);
        assert_eq!(c.exit_code(), 0);
        assert!(fx.scratch.path().join("saw-a").exists(), "A never saw B");
        assert!(fx.scratch.path().join("saw-b").exists(), "B never saw A");
    }

    #[test]
    fn a_pool_of_one_runs_gates_one_at_a_time() {
        let fx = gfx(&[
            ("G-A", "", "host", slow("G-A")),
            ("G-B", "", "host", slow("G-B")),
            ("G-C", "", "host", slow("G-C")),
        ]);
        let mut opts = fx.opts();
        opts.pool_sizes = PoolSizes { host: 1, qemu: 1 };
        let c = fx.campaign(&opts);
        assert_eq!(c.exit_code(), 0);
        let log = fx.log();
        assert_eq!(log.len(), 6, "{log:?}");
        for pair in log.chunks(2) {
            assert!(pair[0].starts_with("start-"), "{log:?}");
            assert_eq!(pair[1], pair[0].replacen("start-", "end-", 1), "{log:?}");
        }
    }

    #[test]
    fn gb10_and_host_gates_run_together_and_record_their_pools() {
        let fx = gfx(&[
            ("G-A", "", "gb10", logging("G-A", 0)),
            ("G-B", "", "host", logging("G-B", 0)),
        ]);
        let mut opts = fx.opts();
        opts.hardware.gb10_present = true;
        let c = fx.campaign(&opts);
        assert_eq!(c.exit_code(), 0);
        let r = read_json(outcome(&c, "G-A").receipt_path.as_ref().unwrap());
        assert_eq!(r["volatile"]["pool"], "gb10");
        let r = read_json(outcome(&c, "G-B").receipt_path.as_ref().unwrap());
        assert_eq!(r["volatile"]["pool"], "host");
    }

    #[test]
    fn a_held_gpu_lock_refuses_the_gb10_gate_and_its_dependents_do_not_run() {
        let fx = gfx(&[
            ("G-A", "", "gb10", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
        ]);
        let mut opts = fx.opts();
        opts.hardware.gb10_present = true;
        let held = acquire_gpu(&opts.gpu).unwrap();
        let c = fx.campaign(&opts);
        drop(held);
        assert_eq!(verdict_and_reason(&c, "G-A"), (Verdict::NotRun, "gpu_lock"));
        assert_eq!(
            verdict_and_reason(&c, "G-B"),
            (Verdict::NotRun, "DEP_NOT_PASS:G-A")
        );
        assert!(fx.log().is_empty());
        assert_eq!(c.exit_code(), 2);
    }

    #[test]
    fn run_gate_reads_dependency_receipts_from_the_store() {
        let fx = gfx(&[
            ("G-A", "", "host", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
        ]);
        let opts = fx.opts();
        let run_one = |id: &str| run_gate(fx.repo.path(), &fx.gate_path(id), &opts).unwrap();
        let o = run_one("G-B");
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "DEP_NOT_PASS:G-A")
        );
        assert!(fx.log().is_empty(), "no receipt for A, so B must not run");
        assert_eq!(run_one("G-A").verdict, Verdict::Pass);
        let o = run_one("G-B");
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert_eq!(fx.log(), vec!["ran-G-A", "ran-G-B"]);
        // A receipt for an older commit does not count.
        git(
            fx.repo.path(),
            &["commit", "-q", "--allow-empty", "-m", "next"],
        );
        let o = run_one("G-B");
        assert_eq!(
            (o.verdict, o.reason.as_str()),
            (Verdict::NotRun, "DEP_NOT_PASS:G-A")
        );
        assert_eq!(fx.log().len(), 2);
    }

    #[test]
    fn a_dependency_outside_the_selection_comes_from_the_store() {
        let fx = gfx(&[
            ("G-A", "", "host", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
        ]);
        let opts = fx.opts();
        let g = fx.graph();
        let only_b: BTreeSet<usize> = [g.index_of("G-B").unwrap()].into_iter().collect();
        let c = run_graph(fx.repo.path(), &g, &only_b, &opts).unwrap();
        assert_eq!(c.results.len(), 1);
        assert_eq!(
            verdict_and_reason(&c, "G-B"),
            (Verdict::NotRun, "DEP_NOT_PASS:G-A")
        );
        let a = run_gate(fx.repo.path(), &fx.gate_path("G-A"), &opts).unwrap();
        assert_eq!(a.verdict, Verdict::Pass);
        let c = run_graph(fx.repo.path(), &g, &only_b, &opts).unwrap();
        assert_eq!(verdict_and_reason(&c, "G-B"), (Verdict::Pass, ""));
    }

    #[test]
    fn an_empty_selection_is_an_empty_campaign() {
        let fx = gfx(&[("G-A", "", "host", logging("G-A", 0))]);
        let g = fx.graph();
        let c = run_graph(fx.repo.path(), &g, &BTreeSet::new(), &fx.opts()).unwrap();
        assert!(c.results.is_empty());
        assert_eq!(c.exit_code(), 0);
    }

    #[test]
    fn a_fatal_error_stops_the_campaign_instead_of_hanging() {
        let fx = gfx(&[
            ("G-A", "", "host", logging("G-A", 0)),
            ("G-B", "G-A", "host", logging("G-B", 0)),
            ("G-C", "", "host", logging("G-C", 0)),
        ]);
        let blocker = fx.scratch.path().join("not-a-directory");
        fs::write(&blocker, "x").unwrap();
        let mut opts = fx.opts();
        opts.evidence_dir = blocker.join("out");
        let g = fx.graph();
        let all: BTreeSet<usize> = (0..g.len()).collect();
        match run_graph(fx.repo.path(), &g, &all, &opts) {
            Err(RunError::Fatal(_)) => {}
            other => panic!("expected a fatal error, got {other:?}"),
        }
    }

    #[test]
    fn default_evidence_dir_is_named_after_the_repository() {
        let p = default_evidence_dir(Path::new("/some/where/my-repo"));
        if let Some(p) = p {
            assert!(p.ends_with("workspace/evidence-out/my-repo"), "{p:?}");
        }
    }

    /// A repo whose gate names a bare program (no slash), resolved on PATH.
    fn bare_fixture(exec: &str) -> Fx {
        let scratch = TempDir::new("scratch");
        let gate = gate_text(exec, &[], 0, &[], "20s", "");
        let repo = init_repo(
            "repo",
            &[
                ("tests/t.gate", gate.as_str(), false),
                ("README", "hello\n", false),
            ],
        );
        Fx {
            repo,
            ev: TempDir::new("ev"),
            scratch,
        }
    }

    #[test]
    fn exec_path_rules() {
        let fx = ok_fixture();
        let root = fx.repo.path();
        // A slash: repo-relative, present or not.
        assert_eq!(
            exec_path(root, "tools/run.sh"),
            Some(root.join("tools/run.sh"))
        );
        assert_eq!(exec_path(root, "tools/missing.sh"), None);
        // No slash: the PATH, never the repository root.
        write_file(root, "true", "#!/bin/sh\nexit 1\n", true);
        let p = exec_path(root, "true").expect("true is on PATH");
        assert_ne!(p, root.join("true"));
        assert!(p.is_absolute());
        assert_eq!(exec_path(root, "aien-test-no-such-program-x"), None);
    }

    #[test]
    fn bare_exec_runs_from_path() {
        let fx = bare_fixture("true");
        let o = run(&fx, &fx.opts());
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        let r = read_json(&o.receipt_path.unwrap());
        assert_eq!(r["core"]["exit_status"], 0);
        assert!(
            r["core"]["binary_sha256"].is_string(),
            "the PATH program is hashed"
        );
    }

    #[test]
    fn bare_exec_not_on_path_is_missing_implementation() {
        let fx = bare_fixture("aien-test-no-such-program-x");
        let o = run(&fx, &fx.opts());
        assert_eq!(
            o.verdict,
            Verdict::MissingImplementation,
            "reason {}",
            o.reason
        );
    }

    #[test]
    fn escaping_exec_is_rejected_never_not_run() {
        for bad in ["../../usr/bin/env", "/usr/bin/env", "tools/../../bin/true"] {
            let fx = bare_fixture(bad);
            match run_gate(fx.repo.path(), &fx.manifest(), &fx.opts()) {
                Err(RunError::BadManifest(m)) => {
                    assert!(m.contains("run.exec"), "{bad}: {m}")
                }
                other => panic!("{bad} must be a manifest reject, got {other:?}"),
            }
        }
    }
}
