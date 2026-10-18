//! The sandbox seam (contract C8 of the local verification pipeline).
//!
//! `--sandbox none` (the default) runs a gate's program directly, exactly as
//! before. `--sandbox user` runs it through the privileged exec helper
//! `/usr/local/sbin/aien-run-exec`, which drops to the unprivileged
//! `aien-run` user and applies the pool's profile:
//!
//! ```text
//! sudo -n /usr/local/sbin/aien-run-exec --cwd CWD --timeout SECS \
//!     --profile host|qemu|gb10 [--env K=V]... -- PROGRAM ARGS...
//! ```
//!
//! Only the environment pairs the gate needs are passed: the manifest grammar
//! has no `env` key today, so that set is empty, plus `CARGO_TARGET_DIR` when
//! the runner itself was given one. The helper is INFRA's; this module only
//! builds the command line and never checks that the helper exists (a
//! missing helper fails the spawn, which the receipt records as rc_nonzero
//! with the error on stderr, never as PASS).

use crate::process::RunSpec;
use std::path::{Path, PathBuf};

pub const EXEC_HELPER: &str = "/usr/local/sbin/aien-run-exec";
pub const SUDO: &str = "sudo";
/// The PATH the exec helper sets for the `aien-run` user (`RUN_PATH` in
/// `aien-run-exec`). Bare `run.exec` names resolve here under `--sandbox user`.
pub const TRUSTED_PATH: &str = "/var/lib/aien-run/cargo/bin:/usr/local/bin:/usr/bin:/bin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sandbox {
    /// Run the program directly (every earlier behaviour unchanged).
    #[default]
    None,
    /// Run it as the `aien-run` user through the exec helper.
    User,
}

impl Sandbox {
    pub fn parse(s: &str) -> Result<Sandbox, String> {
        match s {
            "none" => Ok(Sandbox::None),
            "user" => Ok(Sandbox::User),
            other => Err(format!("--sandbox must be none or user, not {other:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Sandbox::None => "none",
            Sandbox::User => "user",
        }
    }
}

/// Set to `1` in every program the user sandbox runs (`--env` to the helper).
/// Tests that start a sandbox themselves skip when they see it: aien-run
/// cannot call the helper or write `/var/lib/aien-run/wt`.
pub const NESTED_MARKER: &str = "AIEN_SANDBOX_NESTED";

/// The program and arguments to spawn for `program ARGS` under `sandbox`.
/// `profile` is the gate's pool name; `env` the `K=V` pairs to pass through.
pub fn command(
    sandbox: Sandbox,
    program: &Path,
    args: &[String],
    cwd: &Path,
    timeout_secs: u64,
    profile: &str,
    env: &[(String, String)],
) -> (PathBuf, Vec<String>) {
    match sandbox {
        Sandbox::None => (program.to_path_buf(), args.to_vec()),
        Sandbox::User => {
            let mut v: Vec<String> = vec![
                "-n".to_string(),
                EXEC_HELPER.to_string(),
                "--cwd".to_string(),
                cwd.display().to_string(),
                "--timeout".to_string(),
                timeout_secs.to_string(),
                "--profile".to_string(),
                profile.to_string(),
            ];
            // Always set: the program knows it runs inside the sandbox, so self-tests
            // that would need to start a sandbox themselves can skip (they cannot nest).
            v.push("--env".to_string());
            v.push(format!("{NESTED_MARKER}=1"));
            for (k, val) in env {
                v.push("--env".to_string());
                v.push(format!("{k}={val}"));
            }
            v.push("--".to_string());
            v.push(program.display().to_string());
            v.extend(args.iter().cloned());
            (PathBuf::from(SUDO), v)
        }
    }
}

/// Rewrite a direct `RunSpec` for the sandbox. `None` returns it unchanged,
/// byte for byte. The timeout policy (`enforce_timeout`) is not touched: a
/// gb10 job stays advisory and is never killed.
pub fn wrap(sandbox: Sandbox, spec: RunSpec, profile: &str, env: &[(String, String)]) -> RunSpec {
    if sandbox == Sandbox::None {
        return spec;
    }
    let secs = spec.timeout.as_secs().max(1);
    // The runner's own pairs on the spec go through the helper too: sudo
    // would drop them from the environment.
    let mut pairs: Vec<(String, String)> = env.to_vec();
    pairs.extend(spec.env.iter().cloned());
    let (program, args) = command(
        sandbox,
        &spec.program,
        &spec.args,
        &spec.cwd,
        secs,
        profile,
        &pairs,
    );
    RunSpec {
        program,
        args,
        env: Vec::new(),
        ..spec
    }
}

/// The pairs the runner passes through: only `CARGO_TARGET_DIR` when set.
/// The manifest grammar has no `env` key, so nothing comes from the gate.
pub fn env_pairs(cargo_target_dir: Option<&Path>) -> Vec<(String, String)> {
    let mut v = Vec::new();
    if let Some(d) = cargo_target_dir {
        v.push(("CARGO_TARGET_DIR".to_string(), d.display().to_string()));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn spec() -> RunSpec {
        RunSpec {
            program: PathBuf::from("/repo/tools/run.sh"),
            args: vec!["--fast".into(), "x y".into()],
            cwd: PathBuf::from("/repo"),
            timeout: Duration::from_millis(90_500),
            kill_grace: Duration::from_secs(10),
            enforce_timeout: false,
            env: Vec::new(),
            env_remove: Vec::new(),
        }
    }

    #[test]
    fn none_is_the_identity() {
        let s = spec();
        let w = wrap(
            Sandbox::None,
            spec(),
            "gb10",
            &[("CARGO_TARGET_DIR".into(), "/t".into())],
        );
        assert_eq!(w.program, s.program);
        assert_eq!(w.args, s.args);
        assert_eq!(w.cwd, s.cwd);
        assert_eq!(w.timeout, s.timeout);
        assert!(!w.enforce_timeout);
        let (p, a) = command(Sandbox::None, &s.program, &s.args, &s.cwd, 5, "host", &[]);
        assert_eq!(p, s.program);
        assert_eq!(a, s.args);
    }

    #[test]
    fn user_forwards_the_spec_env_to_the_helper() {
        // The runner's own pairs (the GPU lock marker) ride on the spec and reach
        // the sandboxed program through `--env`, after the nested marker.
        let mut s = spec();
        s.env = vec![("AIEN_GB10_LOCK_HELD".to_string(), "1".to_string())];
        let w = wrap(Sandbox::User, s, "gb10", &[]);
        let i = w
            .args
            .iter()
            .position(|a| a == "AIEN_GB10_LOCK_HELD=1")
            .expect("marker is passed as --env");
        assert_eq!(w.args[i - 1], "--env");
        assert!(
            w.env.is_empty(),
            "sudo would drop it; the helper carries it"
        );
    }

    #[test]
    fn user_builds_the_c8_command_line() {
        let env = vec![("CARGO_TARGET_DIR".to_string(), "/tmp/t".to_string())];
        let w = wrap(Sandbox::User, spec(), "qemu", &env);
        assert_eq!(w.program, PathBuf::from("sudo"));
        assert_eq!(
            w.args,
            [
                "-n",
                EXEC_HELPER,
                "--cwd",
                "/repo",
                "--timeout",
                "90",
                "--profile",
                "qemu",
                "--env",
                "AIEN_SANDBOX_NESTED=1",
                "--env",
                "CARGO_TARGET_DIR=/tmp/t",
                "--",
                "/repo/tools/run.sh",
                "--fast",
                "x y",
            ]
        );
        // The timeout policy is untouched: gb10 stays advisory.
        assert!(!w.enforce_timeout);
        assert_eq!(w.timeout, Duration::from_millis(90_500));
        assert_eq!(w.cwd, PathBuf::from("/repo"));
    }

    #[test]
    fn no_env_means_only_the_nested_marker_and_a_tiny_timeout_rounds_up_to_one() {
        let mut s = spec();
        s.timeout = Duration::from_millis(10);
        let w = wrap(Sandbox::User, s, "host", &[]);
        let envs: Vec<&String> = w
            .args
            .windows(2)
            .filter(|p| p[0] == "--env")
            .map(|p| &p[1])
            .collect();
        assert_eq!(envs, vec!["AIEN_SANDBOX_NESTED=1"]);
        let pos = w.args.iter().position(|a| a == "--timeout").unwrap();
        assert_eq!(w.args[pos + 1], "1");
        assert_eq!(env_pairs(None), Vec::<(String, String)>::new());
        assert_eq!(
            env_pairs(Some(Path::new("/x"))),
            vec![("CARGO_TARGET_DIR".to_string(), "/x".to_string())]
        );
    }

    #[test]
    fn parse_and_names() {
        assert_eq!(Sandbox::parse("none").unwrap(), Sandbox::None);
        assert_eq!(Sandbox::parse("user").unwrap(), Sandbox::User);
        assert!(Sandbox::parse("root").is_err());
        assert_eq!(Sandbox::default(), Sandbox::None);
        assert_eq!(Sandbox::User.as_str(), "user");
    }
}
