//! `switchboard launch <alias>`: open the official, unmodified `claude` in
//! one profile's own settings folder (ADR 0030 phase 1, issue #592).
//!
//! The profile folder comes from the rule file (`settings_dir` of an account
//! whose program is `claude`). Switchboard sets `CLAUDE_CONFIG_DIR` itself and
//! never passes an inherited one. The environment `claude` sees is built from
//! an explicit allowlist, so every credential-selecting variable in Claude
//! Code's authentication precedence (API key, bearer token, OAuth token,
//! Anthropic profile, federation, Bedrock / Vertex / Foundry, base URL) and
//! every cloud credential variable stays behind. Proxy variables pass only
//! when the rule file lists them under `[launch] pass_env`.
//!
//! This module plans and checks; it never starts a process and never opens a
//! credentials file. Starting the process is the `Spawner`'s job: the
//! `switchboard` binary holds the real one, tests use `FakeSpawner`.

use crate::account::{Account, Credential, Provider};
use crate::rules::RuleTable;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Every credential-selecting variable named in ADR 0030 amendment A and
/// issue #592. None of these is ever passed to `claude`. The allowlist below
/// already drops them; this list exists so doctor can name them and tests can
/// prove the allowlist covers them.
pub const STRIPPED_ENV: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_BASE_URL",
];

/// Variable families that can carry or select a credential (cloud provider
/// keys, Anthropic Vertex / Bedrock settings, Claude Code switches). Never
/// passed, whatever the allowlist or the rule file says.
pub const STRIPPED_PREFIXES: &[&str] = &["ANTHROPIC_", "CLAUDE_", "AWS_", "GOOGLE_", "AZURE_"];

/// Proxy variables. They pass only when the rule file lists them, because a
/// proxy URL can carry a user name and password.
pub const PROXY_ENV: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
];

/// Variables passed from the shell when present: what a terminal program
/// needs to find itself, draw, and open the browser for the operator's own
/// sign-in. Nothing here selects or carries an Anthropic credential.
const ALLOWED_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "COLORTERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "LANG",
    "LANGUAGE",
    "TZ",
    "TMPDIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "BROWSER",
    "EDITOR",
    "VISUAL",
    "SSH_AUTH_SOCK",
];

/// Locale variables (`LC_ALL`, `LC_CTYPE`, ...) pass as a family.
const ALLOWED_PREFIXES: &[&str] = &["LC_"];

/// The file Claude Code keeps its login in, inside the profile folder (Claude
/// Code authentication docs, "Credential management"). Switchboard only ever
/// checks that it exists and reads its permission bits.
pub const CREDENTIALS_FILE: &str = ".credentials.json";

/// The variable Claude Code reads to find a profile folder.
pub const CONFIG_DIR_VAR: &str = "CLAUDE_CONFIG_DIR";

/// The program name that marks an account as a Claude Code profile.
pub const CLAUDE_PROGRAM: &str = "claude";

/// An environment: variable name to value.
pub type Env = BTreeMap<OsString, OsString>;

/// True for a variable that selects or carries a credential and must never
/// reach `claude`: a listed name or one of the stripped families.
pub fn is_credential_selecting(name: &str) -> bool {
    STRIPPED_ENV.contains(&name) || STRIPPED_PREFIXES.iter().any(|p| name.starts_with(p))
}

fn allowed(name: &str) -> bool {
    ALLOWED_ENV.contains(&name) || ALLOWED_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Build the environment for `claude` from the inherited one: allowlisted
/// names, plus listed proxy names, minus every credential-selecting name,
/// plus `CLAUDE_CONFIG_DIR` set to `config_dir` (any inherited value is
/// dropped). Non-UTF-8 names are dropped.
pub fn launch_env(inherited: &Env, config_dir: &Path, pass_env: &[String]) -> Env {
    let mut out = Env::new();
    for (k, v) in inherited {
        let Some(name) = k.to_str() else { continue };
        let listed_proxy = PROXY_ENV.contains(&name) && pass_env.iter().any(|p| p.as_str() == name);
        if (allowed(name) || listed_proxy) && !is_credential_selecting(name) {
            out.insert(k.clone(), v.clone());
        }
    }
    out.insert(
        OsString::from(CONFIG_DIR_VAR),
        config_dir.as_os_str().to_owned(),
    );
    out
}

fn is_executable_file(p: &Path) -> bool {
    fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// Find `program` the way a shell would: a name containing `/` is used as
/// given, anything else is searched for in `path` (a `PATH` value). Only an
/// executable regular file counts.
pub fn find_program(program: &str, path: Option<&OsStr>) -> Option<PathBuf> {
    if program.contains('/') {
        let p = PathBuf::from(program);
        return is_executable_file(&p).then_some(p);
    }
    std::env::split_paths(path?)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(program))
        .find(|p| is_executable_file(p))
}

fn is_claude_program(program: &str) -> bool {
    Path::new(program).file_name() == Some(OsStr::new(CLAUDE_PROGRAM))
}

/// Every Claude Code profile in the rule file: Anthropic accounts served by
/// `claude` with their own settings folder, in file order.
pub fn claude_profiles(rules: &RuleTable) -> Vec<(&Account, &Path)> {
    rules
        .accounts()
        .iter()
        .filter_map(|a| match &a.credential {
            Credential::SettingsFolder(dir)
                if a.provider == Provider::Anthropic && is_claude_program(&a.program) =>
            {
                Some((a, dir.as_path()))
            }
            _ => None,
        })
        .collect()
}

/// Look up one Claude Code profile by alias (its account id).
pub fn profile<'a>(
    rules: &'a RuleTable,
    alias: &str,
) -> Result<(&'a Account, &'a Path), LaunchError> {
    let profiles = claude_profiles(rules);
    if let Some(found) = profiles.iter().find(|(a, _)| a.id.as_str() == alias) {
        return Ok(*found);
    }
    if rules.accounts().iter().any(|a| a.id.as_str() == alias) {
        return Err(LaunchError::NotAClaudeProfile {
            alias: alias.to_string(),
        });
    }
    Err(LaunchError::UnknownAlias {
        alias: alias.to_string(),
        known: profiles.iter().map(|(a, _)| a.id.to_string()).collect(),
    })
}

/// Everything needed to start `claude` for one profile.
#[derive(Clone, PartialEq, Eq)]
pub struct LaunchPlan {
    pub alias: String,
    /// Resolved path of the executable, so nothing searches `PATH` again.
    pub program: PathBuf,
    pub config_dir: PathBuf,
    /// The complete environment `claude` gets; the spawner clears everything
    /// else.
    pub env: Env,
}

/// Shows variable names only, never values.
impl fmt::Debug for LaunchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = self.env.keys().map(|k| k.to_string_lossy()).collect();
        f.debug_struct("LaunchPlan")
            .field("alias", &self.alias)
            .field("program", &self.program)
            .field("config_dir", &self.config_dir)
            .field("env_names", &names)
            .finish()
    }
}

/// State of a profile folder that is fit to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirState {
    /// Not created yet; the first launch creates it with mode 0700.
    Missing,
    /// A real folder (not a link) that only the owner can open.
    Private,
}

/// Why a profile folder is not fit to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirProblem {
    NotADirectory,
    /// STUB (red step).
    NotOwner {
        uid: u32,
    },
    /// STUB (red step).
    SharedWith(String),
    /// A symbolic link could point two profiles at one folder.
    Symlink,
    /// Group or others have some access; `mode` is the permission bits.
    TooOpen {
        mode: u32,
    },
    Io(String),
}

impl fmt::Display for DirProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirProblem::NotADirectory => f.write_str("exists but is not a folder"),
            DirProblem::Symlink => {
                f.write_str("is a symbolic link; each profile needs its own real folder")
            }
            DirProblem::TooOpen { mode } => write!(
                f,
                "has mode {mode:03o}; only you may open it (run: chmod 700 on the folder)"
            ),
            DirProblem::Io(m) => write!(f, "cannot be checked: {m}"),
            DirProblem::NotOwner { uid } => write!(f, "STUB {uid}"),
            DirProblem::SharedWith(o) => write!(f, "STUB {o}"),
        }
    }
}

/// Check a profile folder without changing it.
pub fn check_profile_dir(path: &Path) -> Result<DirState, DirProblem> {
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(DirState::Missing),
        Err(e) => return Err(DirProblem::Io(e.kind().to_string())),
    };
    if meta.file_type().is_symlink() {
        return Err(DirProblem::Symlink);
    }
    if !meta.is_dir() {
        return Err(DirProblem::NotADirectory);
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(DirProblem::TooOpen { mode });
    }
    Ok(DirState::Private)
}

/// Make sure the profile folder exists and is private: create it with mode
/// 0700 if missing, refuse it if it is a link, not a folder, or open to
/// others. Never loosens or silently fixes an existing folder.
pub fn prepare_profile_dir(path: &Path) -> Result<(), LaunchError> {
    let fail = |problem| LaunchError::ProfileDir {
        path: path.to_path_buf(),
        problem,
    };
    match check_profile_dir(path).map_err(fail)? {
        DirState::Private => Ok(()),
        DirState::Missing => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| fail(DirProblem::Io(e.kind().to_string())))?;
            }
            fs::DirBuilder::new()
                .mode(0o700)
                .create(path)
                .map_err(|e| fail(DirProblem::Io(e.kind().to_string())))?;
            // The process umask can only remove bits; set them exactly.
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .map_err(|e| fail(DirProblem::Io(e.kind().to_string())))?;
            match check_profile_dir(path).map_err(fail)? {
                DirState::Private => Ok(()),
                DirState::Missing => Err(fail(DirProblem::Io("vanished after create".into()))),
            }
        }
    }
}

/// Plan a launch: find the profile, build its environment, and find the
/// executable on the cleaned `PATH`. Changes nothing on disk.
pub fn plan_launch(
    rules: &RuleTable,
    alias: &str,
    inherited: &Env,
) -> Result<LaunchPlan, LaunchError> {
    let (account, dir) = profile(rules, alias)?;
    let env = launch_env(inherited, dir, rules.pass_env());
    let program = find_program(
        &account.program,
        env.get(OsStr::new("PATH")).map(|p| p.as_os_str()),
    )
    .ok_or_else(|| LaunchError::ExecutableNotFound {
        program: account.program.clone(),
    })?;
    Ok(LaunchPlan {
        alias: alias.to_string(),
        program,
        config_dir: dir.to_path_buf(),
        env,
    })
}

/// Starts the planned program. The real one (in the `switchboard` binary)
/// clears the environment, sets exactly `plan.env`, and hands the terminal to
/// `claude`.
pub trait Spawner {
    /// Returns the program's exit code.
    fn spawn(&mut self, plan: &LaunchPlan) -> Result<i32, LaunchError>;
}

/// Records plans instead of starting anything; answers `exit_code`.
#[derive(Debug, Default)]
pub struct FakeSpawner {
    pub plans: Vec<LaunchPlan>,
    pub exit_code: i32,
}

impl Spawner for FakeSpawner {
    fn spawn(&mut self, plan: &LaunchPlan) -> Result<i32, LaunchError> {
        self.plans.push(plan.clone());
        Ok(self.exit_code)
    }
}

/// Plan, prepare the private profile folder, then start `claude` once.
/// Manual launch may open any Claude Code profile in the rule file; route
/// rules do not apply here.
pub fn launch(
    rules: &RuleTable,
    alias: &str,
    inherited: &Env,
    spawner: &mut dyn Spawner,
) -> Result<i32, LaunchError> {
    let plan = plan_launch(rules, alias, inherited)?;
    prepare_profile_dir(&plan.config_dir)?;
    spawner.spawn(&plan)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// No account has this id. `known` lists the Claude Code profiles.
    UnknownAlias {
        alias: String,
        known: Vec<String>,
    },
    /// The account exists but is not a Claude Code profile.
    NotAClaudeProfile {
        alias: String,
    },
    /// The program is not an executable file on the cleaned `PATH`.
    ExecutableNotFound {
        program: String,
    },
    ProfileDir {
        path: PathBuf,
        problem: DirProblem,
    },
    /// The operating system refused to start the program.
    Spawn {
        program: PathBuf,
        message: String,
    },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchError::UnknownAlias { alias, known } => {
                if known.is_empty() {
                    write!(f, "no profile named {alias:?}; the rule file has no Claude Code profiles")
                } else {
                    write!(f, "no profile named {alias:?}; profiles: {}", known.join(", "))
                }
            }
            LaunchError::NotAClaudeProfile { alias } => write!(
                f,
                "{alias:?} is not a Claude Code profile (it needs provider = \"anthropic\", program = \"claude\" and a settings_dir)"
            ),
            LaunchError::ExecutableNotFound { program } => {
                write!(f, "program {program:?} not found as an executable file on PATH")
            }
            LaunchError::ProfileDir { path, problem } => {
                write!(f, "profile folder {} {problem}", path.display())
            }
            LaunchError::Spawn { program, message } => {
                write!(f, "could not start {}: {message}", program.display())
            }
        }
    }
}

impl std::error::Error for LaunchError {}

/// STUB (red step).
pub fn check_profile_dir_owned_by(path: &Path, _uid: u32) -> Result<DirState, DirProblem> {
    check_profile_dir(path)
}
