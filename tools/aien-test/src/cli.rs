//! Argument parsing for the `aien-test` commands (ADR 0028 Decision 1 and 8,
//! ADR 0033 Decisions 10 and 12). Hand-written; the surface is small.
//!
//! ```text
//! aien-test run GATE_FILE_OR_NAME [flags] [-- ARGS...]
//! aien-test test ./... | gate:NAME | changed | crate:NAME [flags]
//! aien-test why GATE_FILE_OR_NAME [flags]
//! aien-test list [flags]
//! aien-test identity GATE_FILE_OR_NAME [--compiler-identity S] [--flag F]... [--hardware-identity S]
//! aien-test plan [--base REF] [--repo NAME] [--operator] [--allow-dirty]
//! aien-test status [--state-dir DIR]
//! aien-test verify COMMIT [--repo NAME] [--base REF] [--state-dir DIR] [--release]
//! aien-test worker [--once REPO SHA] [--poll SECS] [--repos LIST] [--github] [--iterations N]
//!                  [--state-dir DIR] [--key FILE] [--clones DIR] [--wt-root DIR] [--wt-tool FILE] [--siblings FILE] [--sandbox user|none --dev-unsandboxed]
//!                  [--backoff-cap SECS] [--quiet-after SECS] [--prime FILE] [--cold] [--prime-only] (the last two with --once)
//!                  [--target-quota-gb G] [--disk-floor-gb G] [--probe]
//! ```
//!
//! `--host/--qemu/--gb10` and `--mutants` are later slices and are refused.

use crate::resources::{parse_jobs, JobsSpec};
use crate::sandbox::Sandbox;
use std::path::PathBuf;

pub const USAGE: &str = "usage:
  aien-test run GATE_FILE_OR_NAME [flags] [-- ARGS...]
  aien-test test ./... | gate:NAME | changed | crate:NAME [flags]
  aien-test why GATE_FILE_OR_NAME [flags]
  aien-test list [flags]
  aien-test identity GATE_FILE_OR_NAME [--compiler-identity S] [--flag F]... [--hardware-identity S]
  aien-test plan [--base REF] [--repo NAME] [--operator] [--allow-dirty]
  aien-test status [--state-dir DIR]
  aien-test verify COMMIT [--repo NAME] [--base REF] [--state-dir DIR] [--release]
  aien-test worker [--once REPO SHA] [--poll SECS] [--repos A,B] [--github] [--iterations N]
                   [--state-dir DIR] [--key FILE] [--clones DIR] [--wt-root DIR] [--wt-tool FILE] [--siblings FILE] [--sandbox user|none --dev-unsandboxed]
flags: --evidence-dir DIR  --allow-dirty  --no-cache  --release  --jobs host=N,qemu=M  --wait-gpu
       --sandbox none|user  --base REF  --operator";

/// Which gates `test` runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// `./...`: every gate in the repository.
    All,
    /// `gate:NAME`: that gate and everything it depends on.
    Gate(String),
    /// `changed`: what the planner selects for BASE..HEAD.
    Changed,
    /// `crate:NAME`: the gates covering a workspace crate and its dependents.
    Crate(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Run { gate: String },
    Test(Selection),
    Why { gate: String },
    List,
    Identity { gate: String },
    Plan,
    Status,
    Verify { commit: String },
    Worker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub command: Command,
    pub evidence_dir: Option<PathBuf>,
    pub allow_dirty: bool,
    /// `--no-cache`: never reuse an earlier result; run every gate fresh.
    pub no_cache: bool,
    /// `--release`: qualification mode (no cache, clean tree, no
    /// hardware-scoped cache entries accepted by `verify`).
    pub release: bool,
    /// Arguments after `--`; they replace `run.args` (CR-061). `run` only.
    pub args: Option<Vec<String>>,
    /// `--jobs host=N,qemu=M`: the pool sizes that were named.
    pub jobs: JobsSpec,
    /// `--wait-gpu`: wait for a held GPU lock instead of refusing.
    pub wait_gpu: bool,
    /// `identity` only: the compiler identity string (default empty: nothing
    /// here builds yet).
    pub compiler_identity: String,
    /// `identity` only: `--flag F`, repeatable, order kept.
    pub flags: Vec<String>,
    /// `identity` only: needed for `cache_scope: hardware`.
    pub hardware_identity: Option<String>,
    /// `--sandbox none|user` (C8), default none.
    pub sandbox: Sandbox,
    /// Whether `--sandbox` was given (the worker defaults to `user`).
    pub sandbox_given: bool,
    /// worker: `--dev-unsandboxed`: allow `--sandbox none` (development only).
    pub dev_unsandboxed: bool,
    /// `--base REF` for plan, test changed, verify.
    pub base: Option<String>,
    /// `--repo NAME`: the repository name for bundles.
    pub repo: Option<String>,
    /// `--operator`: allow a pipeline configuration change.
    pub operator: bool,
    /// `--state-dir DIR`: the local CI state (default `~/.local/state/aien-local-ci`).
    pub state_dir: Option<PathBuf>,
    /// worker: `--key FILE` (default `~/.aien/verify.key`).
    pub key: Option<PathBuf>,
    /// worker: `--clones DIR` where the trusted clones live (default `~/workspace`).
    pub clones: Option<PathBuf>,
    /// worker: `--wt-root DIR` where detached worktrees go (default `/var/lib/aien-run/wt`).
    pub wt_root: Option<PathBuf>,
    /// worker: `--wt-tool FILE` INFRA's worktree tool (default
    /// `<clones>/aien-sovereign-core/tools/aien-local-ci/wt.sh`).
    pub wt_tool: Option<PathBuf>,
    /// `--siblings FILE`: the operator's sibling map (default
    /// `<clones>/aien-sovereign-core/tools/aien-local-ci/siblings`).
    pub siblings: Option<PathBuf>,
    /// worker: `--once REPO SHA`.
    pub once: Option<(String, String)>,
    /// worker: `--poll SECS` (default 30).
    pub poll: Option<u64>,
    /// worker: `--backoff-cap SECS` ceiling of the backed-off retries (default 900).
    pub backoff_cap: Option<u64>,
    /// worker: `--quiet-after SECS` idle time before backing off (default 600).
    pub quiet_after: Option<u64>,
    /// worker: `--cold`: empty the pr class target dir first (with --once).
    pub cold: bool,
    /// worker: `--prime-only` (with --once REPO MAIN): prime the pr class dir
    /// from that main head and run nothing (with --cold: empty it first).
    pub prime_only: bool,
    /// worker: `--prime FILE` the warm-standby prime map (default
    /// `<clones>/aien-sovereign-core/tools/aien-local-ci/prime`).
    pub prime: Option<PathBuf>,
    /// worker: `--target-quota-gb G` per-repository target dir quota (default 120).
    pub target_quota_gb: Option<u64>,
    /// worker: `--disk-floor-gb G` free space priming keeps (default 50).
    pub disk_floor_gb: Option<u64>,
    /// worker: `--probe`: print the cheap GitHub probe once and exit.
    pub probe: bool,
    /// worker: `--repos A,B`.
    pub repos: Vec<String>,
    /// worker: `--github`: poll GitHub and post commit statuses.
    pub github: bool,
    /// worker: `--iterations N`: stop the loop after N rounds (0: forever).
    pub iterations: Option<u64>,
}

fn parse_selection(s: &str) -> Result<Selection, String> {
    if s == "./..." {
        return Ok(Selection::All);
    }
    if s == "changed" {
        return Ok(Selection::Changed);
    }
    if let Some(name) = s.strip_prefix("crate:") {
        if name.is_empty() {
            return Err("crate: needs a crate name".to_string());
        }
        return Ok(Selection::Crate(name.to_string()));
    }
    match s.strip_prefix("gate:") {
        Some(name) if !name.is_empty() => Ok(Selection::Gate(name.to_string())),
        _ => Err(format!(
            "unsupported test selection {s:?}: use ./..., gate:NAME, changed or crate:NAME"
        )),
    }
}

const COMMANDS: [&str; 9] = [
    "run", "test", "why", "list", "identity", "plan", "status", "verify", "worker",
];
const WORKER_FLAGS: [&str; 21] = [
    "--target-quota-gb",
    "--disk-floor-gb",
    "--probe",
    "--prime-only",
    "--backoff-cap",
    "--quiet-after",
    "--cold",
    "--prime",
    "--dev-unsandboxed",
    "--key",
    "--clones",
    "--wt-root",
    "--wt-tool",
    "--siblings",
    "--once",
    "--poll",
    "--repos",
    "--github",
    "--iterations",
    "--state-dir",
    "--sandbox",
];

pub fn parse_args(argv: &[String]) -> Result<Cli, String> {
    let name = argv
        .first()
        .map(|s| s.as_str())
        .ok_or_else(|| "no command".to_string())?;
    if !COMMANDS.contains(&name) {
        return Err(format!("unknown command {name:?}"));
    }
    let mut positional: Option<String> = None;
    let mut evidence_dir: Option<PathBuf> = None;
    let mut allow_dirty = false;
    let mut no_cache = false;
    let mut release = false;
    let mut args: Option<Vec<String>> = None;
    let mut jobs = JobsSpec::default();
    let mut wait_gpu = false;
    let mut compiler_identity = String::new();
    let mut flags: Vec<String> = Vec::new();
    let mut hardware_identity: Option<String> = None;
    let mut identity_flags = false;
    let mut sandbox = Sandbox::None;
    let mut sandbox_given = false;
    let mut dev_unsandboxed = false;
    let mut base: Option<String> = None;
    let mut repo: Option<String> = None;
    let mut operator = false;
    let mut state_dir: Option<PathBuf> = None;
    let mut key: Option<PathBuf> = None;
    let mut clones: Option<PathBuf> = None;
    let mut wt_root: Option<PathBuf> = None;
    let mut wt_tool: Option<PathBuf> = None;
    let mut siblings: Option<PathBuf> = None;
    let mut once: Option<(String, String)> = None;
    let mut poll: Option<u64> = None;
    let mut backoff_cap: Option<u64> = None;
    let mut quiet_after: Option<u64> = None;
    let mut cold = false;
    let mut prime_only = false;
    let mut prime: Option<PathBuf> = None;
    let mut target_quota_gb: Option<u64> = None;
    let mut disk_floor_gb: Option<u64> = None;
    let mut probe = false;
    let mut repos: Vec<String> = Vec::new();
    let mut github = false;
    let mut iterations: Option<u64> = None;
    let mut worker_flags: Vec<&str> = Vec::new();
    let mut i = 1;
    let value = |i: &mut usize, a: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{a} needs a value"))
    };
    while i < argv.len() {
        let a = argv[i].as_str();
        if a == "--" {
            args = Some(argv[i + 1..].to_vec());
            break;
        } else if a == "--evidence-dir" {
            evidence_dir = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--jobs" {
            jobs = parse_jobs(&value(&mut i, a)?)?;
        } else if a == "--compiler-identity" || a == "--flag" || a == "--hardware-identity" {
            identity_flags = true;
            let v = value(&mut i, a)?;
            match a {
                "--compiler-identity" => compiler_identity = v,
                "--flag" => flags.push(v),
                _ => hardware_identity = Some(v),
            }
        } else if a == "--allow-dirty" {
            allow_dirty = true;
        } else if a == "--no-cache" {
            no_cache = true;
        } else if a == "--release" {
            release = true;
        } else if a == "--wait-gpu" {
            wait_gpu = true;
        } else if a == "--operator" {
            operator = true;
        } else if a == "--base" {
            base = Some(value(&mut i, a)?);
        } else if a == "--repo" {
            repo = Some(value(&mut i, a)?);
        } else if a == "--sandbox" {
            sandbox = Sandbox::parse(&value(&mut i, a)?)?;
            sandbox_given = true;
        } else if a == "--state-dir" {
            state_dir = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--key" {
            worker_flags.push("--key");
            key = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--clones" {
            worker_flags.push("--clones");
            clones = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--wt-root" {
            worker_flags.push("--wt-root");
            wt_root = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--wt-tool" {
            worker_flags.push("--wt-tool");
            wt_tool = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--siblings" {
            worker_flags.push("--siblings");
            siblings = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--once" {
            worker_flags.push("--once");
            let r = value(&mut i, a)?;
            let s = value(&mut i, a)?;
            if s.len() != 40 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("--once needs REPO and a 40-hex SHA".to_string());
            }
            once = Some((r, s.to_ascii_lowercase()));
        } else if a == "--poll" {
            worker_flags.push("--poll");
            let v = value(&mut i, a)?;
            poll = Some(
                v.parse::<u64>()
                    .map_err(|_| format!("--poll needs seconds, not {v:?}"))?,
            );
        } else if a == "--backoff-cap" || a == "--quiet-after" {
            worker_flags.push(if a == "--backoff-cap" {
                "--backoff-cap"
            } else {
                "--quiet-after"
            });
            let v = value(&mut i, a)?;
            let n = v
                .parse::<u64>()
                .map_err(|_| format!("{a} needs seconds, not {v:?}"))?;
            if a == "--backoff-cap" {
                backoff_cap = Some(n);
            } else {
                quiet_after = Some(n);
            }
        } else if a == "--target-quota-gb" || a == "--disk-floor-gb" {
            let quota = a == "--target-quota-gb";
            worker_flags.push(if quota {
                "--target-quota-gb"
            } else {
                "--disk-floor-gb"
            });
            let v = value(&mut i, a)?;
            let n = v
                .parse::<u64>()
                .ok()
                .filter(|n| *n > 0 && *n < 1 << 20)
                .ok_or_else(|| format!("{a} needs whole GiB above 0, not {v:?}"))?;
            if quota {
                target_quota_gb = Some(n);
            } else {
                disk_floor_gb = Some(n);
            }
        } else if a == "--probe" {
            worker_flags.push("--probe");
            probe = true;
        } else if a == "--prime-only" {
            worker_flags.push("--prime-only");
            prime_only = true;
        } else if a == "--cold" {
            worker_flags.push("--cold");
            cold = true;
        } else if a == "--prime" {
            worker_flags.push("--prime");
            prime = Some(PathBuf::from(value(&mut i, a)?));
        } else if a == "--iterations" {
            worker_flags.push("--iterations");
            let v = value(&mut i, a)?;
            iterations = Some(
                v.parse::<u64>()
                    .map_err(|_| format!("--iterations needs a number, not {v:?}"))?,
            );
        } else if a == "--repos" {
            worker_flags.push("--repos");
            repos = value(&mut i, a)?
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
        } else if a == "--dev-unsandboxed" {
            worker_flags.push("--dev-unsandboxed");
            dev_unsandboxed = true;
        } else if a == "--github" {
            worker_flags.push("--github");
            github = true;
        } else if a.starts_with('-') {
            return Err(format!("unknown argument {a:?}"));
        } else if positional.is_none() {
            positional = Some(a.to_string());
        } else {
            return Err(format!("unexpected argument {a:?}"));
        }
        i += 1;
    }
    if args.is_some() && name != "run" {
        return Err("-- ARGS only apply to the run command".to_string());
    }
    if identity_flags && !["identity", "plan", "verify", "worker"].contains(&name) {
        return Err(
            "--compiler-identity, --flag and --hardware-identity only apply to identity, plan, verify and worker"
                .to_string(),
        );
    }
    if prime_only && once.is_none() {
        return Err("--prime-only needs --once REPO MAIN-SHA".to_string());
    }
    if cold && once.is_none() {
        return Err(
            "--cold only applies to worker --once (it empties the pr class target dir)".to_string(),
        );
    }
    if name != "worker" {
        if let Some(f) = worker_flags
            .iter()
            .find(|f| WORKER_FLAGS.contains(f) && !["--state-dir", "--sandbox"].contains(f))
        {
            return Err(format!("{f} only applies to worker"));
        }
    }
    if sandbox != Sandbox::None && !["run", "test", "worker"].contains(&name) {
        return Err("--sandbox only applies to run, test and worker".to_string());
    }
    if base.is_some() && !["plan", "test", "verify", "worker"].contains(&name) {
        return Err("--base only applies to plan, test, verify and worker".to_string());
    }
    if operator && !["plan", "test", "worker"].contains(&name) {
        return Err("--operator only applies to plan, test and worker".to_string());
    }
    if repo.is_some() && !["plan", "verify", "worker"].contains(&name) {
        return Err("--repo only applies to plan, verify and worker".to_string());
    }
    if state_dir.is_some() && !["status", "verify", "worker"].contains(&name) {
        return Err("--state-dir only applies to status, verify and worker".to_string());
    }
    let command = match name {
        "identity" => Command::Identity {
            gate: positional.ok_or("missing GATE_FILE_OR_NAME")?,
        },
        "run" => Command::Run {
            gate: positional.ok_or("missing GATE_FILE_OR_NAME")?,
        },
        "why" => Command::Why {
            gate: positional.ok_or("missing GATE_FILE_OR_NAME")?,
        },
        "test" => {
            let selection = positional.ok_or("missing ./..., gate:NAME, changed or crate:NAME")?;
            Command::Test(parse_selection(&selection)?)
        }
        "verify" => {
            let commit = positional.ok_or("missing COMMIT")?;
            Command::Verify { commit }
        }
        _ => {
            if let Some(p) = positional {
                return Err(format!("unexpected argument {p:?}"));
            }
            match name {
                "plan" => Command::Plan,
                "status" => Command::Status,
                "worker" => Command::Worker,
                _ => Command::List,
            }
        }
    };
    Ok(Cli {
        command,
        evidence_dir,
        allow_dirty,
        no_cache,
        release,
        args,
        jobs,
        wait_gpu,
        compiler_identity,
        flags,
        hardware_identity,
        sandbox,
        sandbox_given,
        dev_unsandboxed,
        base,
        repo,
        operator,
        state_dir,
        key,
        clones,
        wt_root,
        wt_tool,
        siblings,
        once,
        poll,
        backoff_cap,
        quiet_after,
        cold,
        prime_only,
        prime,
        target_quota_gb,
        disk_floor_gb,
        probe,
        repos,
        github,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    fn run_gate(c: &Cli) -> &str {
        match &c.command {
            Command::Run { gate } => gate,
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn minimal() {
        let c = parse_args(&v(&["run", "G"])).unwrap();
        assert_eq!(run_gate(&c), "G");
        assert!(!c.allow_dirty);
        assert!(!c.no_cache);
        assert!(!c.release);
        assert!(!c.wait_gpu);
        assert_eq!(c.args, None);
        assert_eq!(c.evidence_dir, None);
        assert_eq!(c.jobs, JobsSpec::default());
        assert_eq!(c.sandbox, Sandbox::None);
        assert_eq!(c.base, None);
        assert!(!c.operator);
    }

    #[test]
    fn full() {
        let c = parse_args(&v(&[
            "run",
            "--allow-dirty",
            "G",
            "--evidence-dir",
            "/e",
            "--",
            "-x",
            "y",
        ]))
        .unwrap();
        assert_eq!(run_gate(&c), "G");
        assert!(c.allow_dirty);
        assert_eq!(c.evidence_dir, Some(PathBuf::from("/e")));
        assert_eq!(c.args, Some(v(&["-x", "y"])));
    }

    #[test]
    fn no_cache_and_release_flags() {
        let c = parse_args(&v(&["run", "G", "--no-cache"])).unwrap();
        assert!(c.no_cache);
        let c = parse_args(&v(&["test", "--no-cache", "./..."])).unwrap();
        assert!(c.no_cache);
        assert_eq!(c.command, Command::Test(Selection::All));
        let c = parse_args(&v(&["run", "G", "--release"])).unwrap();
        assert!(c.release && !c.no_cache);
        let c = parse_args(&v(&["verify", "a".repeat(40).as_str(), "--release"])).unwrap();
        assert!(c.release);
    }

    #[test]
    fn empty_override_args_are_kept() {
        let c = parse_args(&v(&["run", "G", "--"])).unwrap();
        assert_eq!(c.args, Some(Vec::new()));
    }

    #[test]
    fn test_selections() {
        let c = parse_args(&v(&["test", "./..."])).unwrap();
        assert_eq!(c.command, Command::Test(Selection::All));
        let c = parse_args(&v(&["test", "gate:G-1"])).unwrap();
        assert_eq!(c.command, Command::Test(Selection::Gate("G-1".to_string())));
        let c = parse_args(&v(&["test", "changed", "--base", "origin/main"])).unwrap();
        assert_eq!(c.command, Command::Test(Selection::Changed));
        assert_eq!(c.base.as_deref(), Some("origin/main"));
        let c = parse_args(&v(&["test", "crate:aien-runtime"])).unwrap();
        assert_eq!(
            c.command,
            Command::Test(Selection::Crate("aien-runtime".into()))
        );
    }

    #[test]
    fn unsupported_selections_are_refused() {
        for bad in ["crate:", "gate:", "G", "./", ".", "changed:x"] {
            assert!(parse_args(&v(&["test", bad])).is_err(), "{bad}");
        }
        assert!(parse_args(&v(&["test"])).is_err());
    }

    #[test]
    fn jobs_and_wait_gpu() {
        let c = parse_args(&v(&[
            "test",
            "./...",
            "--jobs",
            "host=3,qemu=2",
            "--wait-gpu",
        ]))
        .unwrap();
        assert_eq!(c.jobs.host, Some(3));
        assert_eq!(c.jobs.qemu, Some(2));
        assert!(c.wait_gpu);
        let c = parse_args(&v(&["test", "./...", "--jobs", "qemu=1"])).unwrap();
        assert_eq!(c.jobs.host, None);
        assert_eq!(c.jobs.qemu, Some(1));
    }

    #[test]
    fn bad_jobs_are_refused() {
        for bad in ["host=0", "gb10=2", "host", "host=x", "disk=1", ""] {
            assert!(
                parse_args(&v(&["test", "./...", "--jobs", bad])).is_err(),
                "{bad}"
            );
        }
        assert!(parse_args(&v(&["test", "./...", "--jobs"])).is_err());
    }

    #[test]
    fn why_and_list() {
        let c = parse_args(&v(&["why", "G-1"])).unwrap();
        assert_eq!(
            c.command,
            Command::Why {
                gate: "G-1".to_string()
            }
        );
        let c = parse_args(&v(&["list"])).unwrap();
        assert_eq!(c.command, Command::List);
        assert!(parse_args(&v(&["why"])).is_err());
        assert!(parse_args(&v(&["list", "extra"])).is_err());
    }

    #[test]
    fn identity_command_and_its_flags() {
        let c = parse_args(&v(&[
            "identity",
            "G",
            "--compiler-identity",
            "rustc 1",
            "--flag",
            "-O",
            "--flag",
            "-C",
            "--hardware-identity",
            "gb10-1",
        ]))
        .unwrap();
        assert_eq!(c.command, Command::Identity { gate: "G".into() });
        assert_eq!(c.compiler_identity, "rustc 1");
        assert_eq!(c.flags, v(&["-O", "-C"]));
        assert_eq!(c.hardware_identity.as_deref(), Some("gb10-1"));
        assert!(parse_args(&v(&["identity"])).is_err());
        assert!(parse_args(&v(&["identity", "G", "--flag"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--flag", "x"])).is_err());
        assert!(parse_args(&v(&["list", "--compiler-identity", "x"])).is_err());
    }

    #[test]
    fn override_args_only_for_run() {
        assert!(parse_args(&v(&["test", "./...", "--", "x"])).is_err());
        assert!(parse_args(&v(&["why", "G", "--", "x"])).is_err());
    }

    #[test]
    fn sandbox_flag() {
        let c = parse_args(&v(&["run", "G", "--sandbox", "user"])).unwrap();
        assert_eq!(c.sandbox, Sandbox::User);
        let c = parse_args(&v(&["test", "changed", "--sandbox", "none"])).unwrap();
        assert_eq!(c.sandbox, Sandbox::None);
        assert!(parse_args(&v(&["run", "G", "--sandbox", "root"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--sandbox"])).is_err());
        assert!(parse_args(&v(&["list", "--sandbox", "user"])).is_err());
    }

    #[test]
    fn plan_status_verify() {
        let c = parse_args(&v(&[
            "plan",
            "--base",
            "main",
            "--repo",
            "github.com/a/b",
            "--operator",
            "--allow-dirty",
        ]))
        .unwrap();
        assert_eq!(c.command, Command::Plan);
        assert_eq!(c.base.as_deref(), Some("main"));
        assert_eq!(c.repo.as_deref(), Some("github.com/a/b"));
        assert!(c.operator && c.allow_dirty);
        assert!(parse_args(&v(&["plan", "extra"])).is_err());
        let c = parse_args(&v(&["status", "--state-dir", "/s"])).unwrap();
        assert_eq!(c.command, Command::Status);
        assert_eq!(c.state_dir, Some(PathBuf::from("/s")));
        let sha = "b".repeat(40);
        let c = parse_args(&v(&["verify", &sha, "--repo", "x", "--base", "main"])).unwrap();
        assert_eq!(
            c.command,
            Command::Verify {
                commit: sha.clone()
            }
        );
        assert!(parse_args(&v(&["verify"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--base", "x"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--operator"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--state-dir", "/s"])).is_err());
        assert!(parse_args(&v(&["list", "--repo", "x"])).is_err());
    }

    #[test]
    fn power_and_warm_flags() {
        let sha = "c".repeat(40);
        let c = parse_args(&v(&[
            "worker",
            "--backoff-cap",
            "300",
            "--quiet-after",
            "120",
            "--prime",
            "/p",
        ]))
        .unwrap();
        assert_eq!((c.backoff_cap, c.quiet_after), (Some(300), Some(120)));
        assert_eq!(c.prime, Some(PathBuf::from("/p")));
        assert!(!c.cold);
        let c = parse_args(&v(&["worker", "--once", "r", &sha, "--cold"])).unwrap();
        assert!(c.cold);
        assert!(parse_args(&v(&["worker", "--cold"]))
            .unwrap_err()
            .contains("--cold only applies"));
        assert!(parse_args(&v(&["worker", "--backoff-cap", "x"])).is_err());
        assert!(parse_args(&v(&["worker", "--quiet-after"])).is_err());
        assert!(parse_args(&v(&["status", "--backoff-cap", "5"]))
            .unwrap_err()
            .contains("only applies to worker"));
        assert!(parse_args(&v(&["status", "--prime", "/p"])).is_err());
    }

    #[test]
    fn worker_flags() {
        let sha = "c".repeat(40);
        let c = parse_args(&v(&[
            "worker",
            "--once",
            "repo",
            &sha,
            "--poll",
            "5",
            "--repos",
            "a,b",
            "--github",
            "--iterations",
            "3",
            "--state-dir",
            "/s",
            "--key",
            "/k",
            "--clones",
            "/c",
            "--wt-root",
            "/w",
            "--wt-tool",
            "/t/wt.sh",
            "--siblings",
            "/t/siblings",
            "--sandbox",
            "user",
        ]))
        .unwrap();
        assert_eq!(c.command, Command::Worker);
        assert_eq!(c.once, Some(("repo".to_string(), sha.clone())));
        assert_eq!(c.poll, Some(5));
        assert_eq!(c.repos, v(&["a", "b"]));
        assert!(c.github);
        assert_eq!(c.iterations, Some(3));
        assert_eq!(c.key, Some(PathBuf::from("/k")));
        assert_eq!(c.clones, Some(PathBuf::from("/c")));
        assert_eq!(c.wt_root, Some(PathBuf::from("/w")));
        assert_eq!(c.wt_tool, Some(PathBuf::from("/t/wt.sh")));
        assert_eq!(c.siblings, Some(PathBuf::from("/t/siblings")));
        assert_eq!(c.sandbox, Sandbox::User);
        assert!(parse_args(&v(&["worker", "--once", "repo", "short"])).is_err());
        assert!(parse_args(&v(&["worker", "--poll", "x"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--poll", "5"])).is_err());
        assert!(parse_args(&v(&["plan", "--github"])).is_err());
        assert!(parse_args(&v(&["worker", "extra"])).is_err());
    }

    #[test]
    fn errors() {
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&v(&["test", "G"])).is_err());
        assert!(parse_args(&v(&["bogus", "G"])).is_err());
        assert!(parse_args(&v(&["run"])).is_err());
        assert!(parse_args(&v(&["run", "G", "H"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--bogus"])).is_err());
        assert!(parse_args(&v(&["run", "G", "--evidence-dir"])).is_err());
    }
}
