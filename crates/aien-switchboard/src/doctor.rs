//! `switchboard doctor`: check the setup without reading any secret.
//!
//! What it reports: the `claude` executable and its version, each profile
//! folder (exists, real folder, private), whether each profile's credentials
//! file is present and its permission bits (never opened), credential
//! variables set in the shell (names only), and settings that could bring an
//! API key back in: `apiKeyHelper` or an `env` block in the profile's
//! `settings.json` and in the working repo's `.claude/settings.json` and
//! `.claude/settings.local.json` (setting names only, never values). It also
//! flags Anthropic profile files in `~/.config/anthropic` by presence,
//! because Claude Code's precedence lets some of them outrank the claude.ai
//! login.

use crate::launch::{
    check_profile_dir, claude_profiles, find_program, folder_shared_with, is_credential_selecting,
    launch_env, DirProblem, DirState, Env, CONFIG_DIR_VAR, CREDENTIALS_FILE,
};
use crate::rules::RuleTable;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

impl Level {
    fn tag(self) -> &'static str {
        match self {
            Level::Ok => "[ OK ]",
            Level::Warn => "[WARN]",
            Level::Fail => "[FAIL]",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub level: Level,
    /// What was checked: a profile alias, a program name, a file path.
    pub subject: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub findings: Vec<Finding>,
}

impl Report {
    fn push(&mut self, level: Level, subject: impl Into<String>, message: impl Into<String>) {
        self.findings.push(Finding {
            level,
            subject: subject.into(),
            message: message.into(),
        });
    }

    /// True when any check failed; `switchboard doctor` then exits 1.
    pub fn failed(&self) -> bool {
        self.findings.iter().any(|f| f.level == Level::Fail)
    }

    fn count(&self, level: Level) -> usize {
        self.findings.iter().filter(|f| f.level == level).count()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in &self.findings {
            writeln!(f, "{} {}: {}", x.level.tag(), x.subject, x.message)?;
        }
        write!(
            f,
            "doctor: {} ok, {} warnings, {} failures",
            self.count(Level::Ok),
            self.count(Level::Warn),
            self.count(Level::Fail)
        )
    }
}

/// Asks a program for its version. The real probe (in the `switchboard`
/// binary) runs `<program> --version` with exactly `env`; tests use
/// `FakeProbe`.
pub trait VersionProbe {
    fn version(&mut self, program: &Path, env: &Env) -> Result<String, String>;
}

/// Answers a fixed result and records which programs were asked.
#[derive(Debug)]
pub struct FakeProbe {
    pub answer: Result<String, String>,
    pub calls: Vec<PathBuf>,
}

impl VersionProbe for FakeProbe {
    fn version(&mut self, program: &Path, _env: &Env) -> Result<String, String> {
        self.calls.push(program.to_path_buf());
        self.answer.clone()
    }
}

/// First line, printable characters only, at most 80 of them.
fn tidy(s: &str) -> String {
    s.lines()
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| !c.is_control())
        .take(80)
        .collect::<String>()
        .trim()
        .to_string()
}

/// Settings files larger than this are not parsed.
const MAX_SETTINGS_BYTES: u64 = 1 << 20;

/// Run every check. Reads only folder and file metadata, plus the JSON of
/// non-credential settings files, of which only key names are reported.
pub fn doctor(rules: &RuleTable, env: &Env, cwd: &Path, probe: &mut dyn VersionProbe) -> Report {
    let mut r = Report::default();
    let profiles = claude_profiles(rules);
    if profiles.is_empty() {
        r.push(
            Level::Warn,
            "rule file",
            "no Claude Code profiles (accounts with provider = \"anthropic\", program = \"claude\" and a settings_dir)",
        );
    } else {
        let names: Vec<&str> = profiles.iter().map(|(a, _)| a.id.as_str()).collect();
        r.push(
            Level::Ok,
            "rule file",
            format!("{} Claude Code profiles: {}", names.len(), names.join(", ")),
        );
    }

    check_shell_env(&mut r, env);

    // The environment the version probe gets: the launch allowlist, without a
    // profile folder so no profile is touched.
    let mut probe_env = launch_env(env, Path::new("/"), rules.pass_env());
    probe_env.remove(OsStr::new(CONFIG_DIR_VAR));
    let path_var = probe_env.get(OsStr::new("PATH")).cloned();
    let mut programs: Vec<&str> = profiles.iter().map(|(a, _)| a.program.as_str()).collect();
    programs.sort_unstable();
    programs.dedup();
    for program in programs {
        match find_program(program, path_var.as_deref()) {
            None => r.push(
                Level::Fail,
                program,
                "not found as an executable file on PATH; install Claude Code or fix PATH",
            ),
            Some(exe) => match probe.version(&exe, &probe_env) {
                Ok(v) => r.push(
                    Level::Ok,
                    program,
                    format!("found {}, version {}", exe.display(), tidy(&v)),
                ),
                Err(e) => r.push(
                    Level::Warn,
                    program,
                    format!("found {} but --version failed: {}", exe.display(), tidy(&e)),
                ),
            },
        }
    }

    for (account, dir) in &profiles {
        check_profile(&mut r, rules, account.id.as_str(), dir);
    }

    for path in repo_settings_files(cwd) {
        scan_settings(&mut r, &path);
    }

    if let Some(home) = env.get(OsStr::new("HOME")) {
        check_anthropic_profiles(&mut r, &Path::new(home).join(".config").join("anthropic"));
    }
    r
}

fn check_shell_env(r: &mut Report, env: &Env) {
    let names: Vec<&str> = env
        .keys()
        .filter_map(|k| k.to_str())
        .filter(|k| is_credential_selecting(k) && *k != CONFIG_DIR_VAR)
        .collect();
    if names.is_empty() {
        r.push(Level::Ok, "shell", "no credential-selecting variables set");
    } else {
        r.push(
            Level::Warn,
            "shell",
            format!(
                "credential-selecting variables set: {}; launch never passes them to claude",
                names.join(", ")
            ),
        );
    }
}

fn check_profile(r: &mut Report, rules: &RuleTable, alias: &str, dir: &Path) {
    if let Some(other) = folder_shared_with(rules, alias, dir) {
        let p = DirProblem::SharedWith(other);
        r.push(Level::Fail, alias, format!("folder {} {p}", dir.display()));
        return;
    }
    match check_profile_dir(dir) {
        Ok(DirState::Missing) => {
            r.push(
                Level::Warn,
                alias,
                format!(
                    "folder not created yet: {}; `switchboard launch {alias}` creates it (mode 700), then sign in",
                    dir.display()
                ),
            );
            return;
        }
        Ok(DirState::Private) => {
            r.push(
                Level::Ok,
                alias,
                format!("folder {} is private (700)", dir.display()),
            );
        }
        Err(p @ DirProblem::TooOpen { .. }) => {
            r.push(Level::Fail, alias, format!("folder {} {p}", dir.display()));
        }
        Err(p) => {
            r.push(Level::Fail, alias, format!("folder {} {p}", dir.display()));
            return;
        }
    }
    check_credentials(r, alias, &dir.join(CREDENTIALS_FILE));
    scan_settings(r, &dir.join("settings.json"));
}

/// Presence and permission bits only. The file is never opened.
fn check_credentials(r: &mut Report, alias: &str, path: &Path) {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => r.push(
            Level::Warn,
            alias,
            format!("not signed in yet (no {CREDENTIALS_FILE}); run `switchboard launch {alias}` and sign in"),
        ),
        Err(e) => r.push(
            Level::Warn,
            alias,
            format!("cannot check {CREDENTIALS_FILE}: {}", e.kind()),
        ),
        Ok(m) if m.file_type().is_symlink() => r.push(
            Level::Warn,
            alias,
            "credentials file present but is a symbolic link; it may be shared with another profile",
        ),
        Ok(m) if !m.is_file() => r.push(
            Level::Warn,
            alias,
            format!("{CREDENTIALS_FILE} exists but is not a regular file"),
        ),
        Ok(m) => {
            let mode = m.permissions().mode() & 0o777;
            if mode == 0o600 {
                r.push(Level::Ok, alias, "credentials file present (mode 600, not opened)");
            } else {
                r.push(
                    Level::Warn,
                    alias,
                    format!("credentials file present with mode {mode:03o}, expected 600 (not opened)"),
                );
            }
        }
    }
}

/// `.claude/settings.json` and `.claude/settings.local.json` in the working
/// folder and, when different, in the repository root above it (the nearest
/// folder holding `.git`).
fn repo_settings_files(cwd: &Path) -> Vec<PathBuf> {
    let mut roots = vec![cwd.to_path_buf()];
    if let Some(root) = cwd.ancestors().find(|a| a.join(".git").exists()) {
        if root != cwd {
            roots.push(root.to_path_buf());
        }
    }
    roots
        .iter()
        .flat_map(|root| {
            ["settings.json", "settings.local.json"]
                .into_iter()
                .map(move |f| root.join(".claude").join(f))
        })
        .collect()
}

/// Warn on `apiKeyHelper` or an `env` block. Reports setting names only.
fn scan_settings(r: &mut Report, path: &Path) {
    let subject = path.display().to_string();
    // Never follow a link into a credentials file.
    if let Ok(real) = fs::canonicalize(path) {
        if real.file_name() == Some(OsStr::new(CREDENTIALS_FILE)) {
            r.push(
                Level::Warn,
                subject,
                "points at a credentials file; not opened",
            );
            return;
        }
    }
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            r.push(Level::Warn, subject, format!("cannot read: {}", e.kind()));
            return;
        }
    };
    let mut text = String::new();
    if file
        .take(MAX_SETTINGS_BYTES + 1)
        .read_to_string(&mut text)
        .is_err()
    {
        r.push(
            Level::Warn,
            subject,
            "cannot read as text; apiKeyHelper and env not checked",
        );
        return;
    }
    if text.len() as u64 > MAX_SETTINGS_BYTES {
        r.push(
            Level::Warn,
            subject,
            "larger than 1 MiB; apiKeyHelper and env not checked",
        );
        return;
    }
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            // Line and column only: the parser's own message can quote input.
            r.push(
                Level::Warn,
                subject,
                format!(
                    "cannot parse as JSON (line {}, column {}); apiKeyHelper and env not checked",
                    e.line(),
                    e.column()
                ),
            );
            return;
        }
    };
    let Some(obj) = value.as_object() else {
        r.push(
            Level::Warn,
            subject,
            "is not a JSON object; apiKeyHelper and env not checked",
        );
        return;
    };
    let mut clean = true;
    if obj.contains_key("apiKeyHelper") {
        clean = false;
        r.push(
            Level::Warn,
            subject.clone(),
            "sets apiKeyHelper: a script that can supply an API key and outrank the claude.ai login",
        );
    }
    if let Some(block) = obj.get("env") {
        clean = false;
        let (count, risky) = match block.as_object() {
            Some(m) => {
                let risky: Vec<&str> = m
                    .keys()
                    .map(|k| k.as_str())
                    .filter(|k| is_credential_selecting(k))
                    .collect();
                (m.len(), risky)
            }
            None => (0, Vec::new()),
        };
        let detail = if risky.is_empty() {
            "none of them select a credential".to_string()
        } else {
            format!("credential-selecting: {}", risky.join(", "))
        };
        r.push(
            Level::Warn,
            subject.clone(),
            format!("has an env block ({count} variables, {detail}); claude applies it after launch, so it can bring an API key back in"),
        );
    }
    if clean {
        r.push(Level::Ok, subject, "no apiKeyHelper, no env block");
    }
}

/// Presence only. An `oidc_federation` active profile outranks the claude.ai
/// login (Claude Code authentication docs, "Anthropic profiles and federation
/// credentials").
fn check_anthropic_profiles(r: &mut Report, dir: &Path) {
    let present: Vec<&str> = ["active_config", "configs"]
        .into_iter()
        .filter(|n| fs::symlink_metadata(dir.join(n)).is_ok())
        .collect();
    if !present.is_empty() {
        r.push(
            Level::Warn,
            dir.display().to_string(),
            format!(
                "Anthropic profile files present ({}); a federation profile there can outrank the claude.ai login; check /status in each profile",
                present.join(", ")
            ),
        );
    }
}
