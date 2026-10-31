//! `switchboard`: list, open and check isolated Claude Code profiles
//! (ADR 0030 phase 1, issue #592). The only place in the crate that starts a
//! process.

use aien_switchboard::cli;
use aien_switchboard::launch::{LaunchError, LaunchPlan, Spawner};
use aien_switchboard::{Env, VersionProbe};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};

/// Replaces this process with `claude`, so `claude` owns the terminal and its
/// signals directly. Only returns if the operating system refused.
struct ExecSpawner;

impl Spawner for ExecSpawner {
    fn spawn(&mut self, plan: &LaunchPlan) -> Result<i32, LaunchError> {
        let e = Command::new(&plan.program)
            .env_clear()
            .envs(&plan.env)
            .exec();
        Err(LaunchError::Spawn {
            program: plan.program.clone(),
            message: e.kind().to_string(),
        })
    }
}

/// Runs `<program> --version` with exactly the given environment and no
/// input.
struct CommandProbe;

impl VersionProbe for CommandProbe {
    fn version(&mut self, program: &Path, env: &Env) -> Result<String, String> {
        let out = Command::new(program)
            .arg("--version")
            .env_clear()
            .envs(env)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| e.kind().to_string())?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(format!("exit status {}", out.status))
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env: Env = std::env::vars_os().collect();
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("switchboard: cannot read the working folder: {}", e.kind());
            std::process::exit(1);
        }
    };
    let code = cli::run(
        &args,
        &env,
        &cwd,
        &mut ExecSpawner,
        &mut CommandProbe,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}
