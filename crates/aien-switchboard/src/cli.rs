//! The `switchboard` command line: `accounts`, `launch <alias>`, `doctor`.
//!
//! Everything the command touches comes in as arguments (environment, working
//! folder, spawner, version probe, output streams) so tests drive it with
//! fakes and a temp HOME.

use crate::account::Credential;
use crate::doctor::{doctor, VersionProbe};
use crate::launch::{claude_profiles, launch, Env, Spawner};
use crate::rules::RuleTable;
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: switchboard [--rules FILE] <command>

commands:
  accounts         list the accounts in the rule file
  launch <alias>   open claude in that profile's own settings folder
  doctor           check the setup (never reads a secret)

The rule file defaults to $HOME/.switchboard/rules.toml.";

/// Exit codes: 0 done, 1 the command failed, 2 usage error. `launch` returns
/// the exit code of `claude`.
pub fn run(
    args: &[String],
    env: &Env,
    cwd: &Path,
    spawner: &mut dyn Spawner,
    probe: &mut dyn VersionProbe,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let mut rules_path: Option<PathBuf> = None;
    let mut rest: Vec<&str> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rules" => match it.next() {
                Some(p) => rules_path = Some(PathBuf::from(p)),
                None => return usage(err, "--rules needs a file"),
            },
            "-h" | "--help" | "help" => {
                let _ = writeln!(out, "{USAGE}");
                return 0;
            }
            other => rest.push(other),
        }
    }
    let (cmd, alias) = match rest.as_slice() {
        ["accounts"] => ("accounts", None),
        ["doctor"] => ("doctor", None),
        ["launch", alias] => ("launch", Some(*alias)),
        [] => return usage(err, "no command given"),
        ["launch"] => return usage(err, "launch needs a profile alias"),
        _ => return usage(err, &format!("unexpected arguments: {}", rest.join(" "))),
    };

    let path = match rules_path.or_else(|| default_rules_path(env)) {
        Some(p) => p,
        None => return usage(err, "HOME is not set; pass --rules FILE"),
    };
    let rules = match RuleTable::from_path(&path) {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(err, "switchboard: {e}");
            return 1;
        }
    };

    match (cmd, alias) {
        ("accounts", _) => {
            accounts(&rules, out);
            0
        }
        ("doctor", _) => {
            let report = doctor(&rules, env, cwd, probe);
            let _ = writeln!(out, "rule file: {}\n{report}", path.display());
            i32::from(report.failed())
        }
        (_, Some(alias)) => {
            let _ = writeln!(err, "switchboard: opening profile {alias}");
            match launch(&rules, alias, env, spawner) {
                Ok(code) => code,
                Err(e) => {
                    let _ = writeln!(err, "switchboard: {e}");
                    1
                }
            }
        }
        _ => usage(err, "unknown command"),
    }
}

fn usage(err: &mut dyn Write, why: &str) -> i32 {
    let _ = writeln!(err, "switchboard: {why}\n\n{USAGE}");
    2
}

fn default_rules_path(env: &Env) -> Option<PathBuf> {
    let home = env.get(OsStr::new("HOME"))?;
    Some(Path::new(home).join(".switchboard").join("rules.toml"))
}

fn accounts(rules: &RuleTable, out: &mut dyn Write) {
    let launchable: Vec<&str> = claude_profiles(rules)
        .iter()
        .map(|(a, _)| a.id.as_str())
        .collect();
    let _ = writeln!(
        out,
        "{:<16} {:<11} {:<8} {:<7} WHERE",
        "ID", "PROVIDER", "PROGRAM", "LAUNCH"
    );
    for a in rules.accounts() {
        let where_ = match &a.credential {
            Credential::SettingsFolder(p) => format!("folder {}", p.display()),
            Credential::KeyRef(k) => format!("vault key name {k}"),
            Credential::None => "no login needed".to_string(),
        };
        let can = if launchable.contains(&a.id.as_str()) {
            "yes"
        } else {
            "-"
        };
        let _ = writeln!(
            out,
            "{:<16} {:<11} {:<8} {:<7} {where_}",
            a.id.as_str(),
            a.provider.as_str(),
            a.program,
            can
        );
    }
}
