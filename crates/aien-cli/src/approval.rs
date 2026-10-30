// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Approval floors ordered above every mode (sc#667).
//!
//! One ordering rule the chat loop cannot skip. For every tool request:
//!
//! 1. **Floors**, unconditional in every mode including `yolo` and `off`:
//!    a hardline command list, deletion or rewrite of the runtime's own files
//!    (binary, approval policy, checkpoint store, session journals), `sudo`
//!    fed a password through stdin, and the operator's own deny patterns.
//! 2. **Policy deny** from the existing nine-tier `SafetyEngine` (confinement,
//!    vault rules). A mode loosens questions, never a denial.
//! 3. **Mode**: `yolo` and `off` allow what is left.
//! 4. **Permanent allowlist** from the operator's policy file.
//! 5. **Presence**: with no person present, anything that still needs a
//!    decision is denied (unattended defaults to deny).
//! 6. **Dangerous patterns** and the engine's own questions decide whether a
//!    decision is needed at all.
//! 7. **Decision**: in `smart` mode a guardian model answers approve, deny or
//!    escalate behind a circuit breaker; otherwise, or on escalate, the person.
//!
//! The functions in the pure section take values and return values: no
//! clock, disk, environment or terminal. The edges (policy file, presence,
//! runtime paths, the prompt and the guardian model) sit at the bottom and in
//! `tools.rs`, so the pure part can move to Omega as a semantic identity
//! (ADR 0024).
//!
//! Prior art: Hermes Agent, Nous Research, MIT licence, commit 5480918
//! (read 2026-10-18), `tools/approval_floors.py`, `tools/approval.py`,
//! `tools/approval_smart.py`. Mechanism re-implemented from reading; no code
//! copied.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

use crate::safety::{is_high_risk_command, is_high_risk_path, SafetyDecision};

// ---------------------------------------------------------------------------
// Pure part
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Ask the person whenever a decision is needed (the default).
    Manual,
    /// Ask the guardian model first; it may approve, deny or escalate.
    Smart,
    /// Approve everything the floors and the policy denials leave.
    Yolo,
    /// Approvals switched off; same effect as `Yolo`. Floors still hold.
    Off,
}

#[cfg(test)]
pub const ALL_MODES: [ApprovalMode; 4] = [
    ApprovalMode::Manual,
    ApprovalMode::Smart,
    ApprovalMode::Yolo,
    ApprovalMode::Off,
];

impl ApprovalMode {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "manual" | "ask" => Some(Self::Manual),
            "smart" | "guardian" => Some(Self::Smart),
            "yolo" => Some(Self::Yolo),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Person,
    Unattended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Floor {
    Hardline,
    RuntimeSelfDelete,
    SudoPipedPassword,
    OperatorDeny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Floor(Floor),
    PolicyDeny,
    Mode,
    Allowlist,
    Presence,
    Clear,
    Guardian,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Allow {
        by: Stage,
    },
    Deny {
        by: Stage,
        reason: String,
    },
    /// Smart mode with a closed breaker: the edge asks the guardian, then
    /// calls [`after_guardian`].
    AskGuardian {
        reason: String,
    },
    AskPerson {
        reason: String,
    },
}

/// The operator's policy file as written on disk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OperatorPolicy {
    #[serde(default)]
    pub mode: Option<ApprovalMode>,
    /// Regular expressions; a match on the request's subject denies it in
    /// every mode.
    #[serde(default)]
    pub deny: Vec<String>,
    /// Regular expressions; a match skips the question (never a floor).
    #[serde(default)]
    pub allow: Vec<String>,
}

/// Compiled policy. Fail closed: a deny pattern that does not compile, or a
/// policy file that does not parse, denies every request with the reason.
#[derive(Debug, Clone, Default)]
pub struct CompiledPolicy {
    pub mode: Option<ApprovalMode>,
    deny: Vec<Regex>,
    allow: Vec<Regex>,
    broken: Option<String>,
    pub warnings: Vec<String>,
}

impl CompiledPolicy {
    pub fn compile(policy: &OperatorPolicy) -> Self {
        let mut out = CompiledPolicy {
            mode: policy.mode,
            ..Default::default()
        };
        for (i, pattern) in policy.deny.iter().enumerate() {
            match Regex::new(pattern) {
                Ok(re) => out.deny.push(re),
                Err(err) => {
                    out.broken = Some(format!(
                        "operator deny pattern {} ({:?}) does not compile: {}",
                        i + 1,
                        pattern,
                        err
                    ));
                }
            }
        }
        for (i, pattern) in policy.allow.iter().enumerate() {
            match Regex::new(pattern) {
                Ok(re) => out.allow.push(re),
                // An allow pattern that does not compile grants nothing.
                Err(err) => out.warnings.push(format!(
                    "operator allow pattern {} ({:?}) ignored: {}",
                    i + 1,
                    pattern,
                    err
                )),
            }
        }
        out
    }

    /// Why every request is being denied, when the policy is broken.
    pub fn broken(&self) -> Option<&str> {
        self.broken.as_deref()
    }

    /// The policy file exists but cannot be read or parsed.
    pub fn unreadable(reason: String) -> Self {
        CompiledPolicy {
            broken: Some(format!("approval policy unreadable: {}", reason)),
            ..Default::default()
        }
    }
}

/// Values the edge measures once per session.
#[derive(Debug, Clone, Default)]
pub struct RuntimeFacts {
    /// Files and directories the agent must never delete or rewrite.
    pub protected: Vec<PathBuf>,
    /// Expansion for `~` and `$HOME` in commands.
    pub home: PathBuf,
    /// Base for relative paths when a request names no `cwd`.
    pub cwd: PathBuf,
}

/// One tool request at the gate.
pub struct Gate<'a> {
    pub tool: &'a str,
    pub args: &'a Value,
    /// What the existing nine-tier `SafetyEngine` said.
    pub base: &'a SafetyDecision,
    pub mode: ApprovalMode,
    pub presence: Presence,
    pub policy: &'a CompiledPolicy,
    pub facts: &'a RuntimeFacts,
    /// A guardian is configured and its breaker is closed.
    pub guardian_ready: bool,
}

/// The ordering rule. Floors first, every mode.
pub fn decide(gate: &Gate) -> Verdict {
    if let Some((floor, reason)) = check_floors(gate.tool, gate.args, gate.policy, gate.facts) {
        return Verdict::Deny {
            by: Stage::Floor(floor),
            reason,
        };
    }
    if let SafetyDecision::Deny(reason) = gate.base {
        return Verdict::Deny {
            by: Stage::PolicyDeny,
            reason: reason.clone(),
        };
    }
    if matches!(gate.mode, ApprovalMode::Yolo | ApprovalMode::Off) {
        return Verdict::Allow { by: Stage::Mode };
    }
    let subject = subject(gate.tool, gate.args);
    if gate
        .policy
        .allow
        .iter()
        .any(|re| matches_subject(re, &subject))
    {
        return Verdict::Allow {
            by: Stage::Allowlist,
        };
    }
    let reason = match needs_decision(gate.tool, gate.args, gate.base) {
        None => return Verdict::Allow { by: Stage::Clear },
        Some(reason) => reason,
    };
    if gate.presence == Presence::Unattended {
        return Verdict::Deny {
            by: Stage::Presence,
            reason: format!("no person present to approve, unattended default is deny: {reason}"),
        };
    }
    if gate.mode == ApprovalMode::Smart && gate.guardian_ready {
        return Verdict::AskGuardian { reason };
    }
    Verdict::AskPerson { reason }
}

/// What the guardian model answered.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardianAnswer {
    Approve,
    Deny(String),
    Escalate,
    /// Timeout, transport error or an answer that did not parse.
    Failed(String),
}

/// Second half of [`decide`] after an `AskGuardian` verdict.
pub fn after_guardian(answer: &GuardianAnswer, reason: String) -> Verdict {
    match answer {
        GuardianAnswer::Approve => Verdict::Allow {
            by: Stage::Guardian,
        },
        GuardianAnswer::Deny(why) => Verdict::Deny {
            by: Stage::Guardian,
            reason: format!("guardian denied: {why}"),
        },
        GuardianAnswer::Escalate | GuardianAnswer::Failed(_) => Verdict::AskPerson { reason },
    }
}

/// Read the guardian's reply. Only the first word counts; anything else is a
/// failure, which escalates to the person and counts against the breaker.
pub fn parse_guardian_reply(text: &str) -> GuardianAnswer {
    let trimmed = text.trim();
    let first = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_ascii_uppercase();
    match first.as_str() {
        "APPROVE" => GuardianAnswer::Approve,
        "DENY" => GuardianAnswer::Deny(trimmed.chars().take(200).collect()),
        "ESCALATE" => GuardianAnswer::Escalate,
        _ => GuardianAnswer::Failed(format!(
            "unparseable guardian reply: {:?}",
            trimmed.chars().take(80).collect::<String>()
        )),
    }
}

/// Circuit breaker in front of the guardian model. After `threshold`
/// consecutive failures it opens and stays open for the session, so a broken
/// or slow guardian costs at most `threshold` calls; every request then goes
/// to the person.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardianBreaker {
    threshold: u32,
    consecutive_failures: u32,
    open: bool,
}

impl GuardianBreaker {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold: threshold.max(1),
            consecutive_failures: 0,
            open: false,
        }
    }

    pub fn is_closed(&self) -> bool {
        !self.open
    }

    pub fn record(&mut self, answer: &GuardianAnswer) {
        if matches!(answer, GuardianAnswer::Failed(_)) {
            self.consecutive_failures += 1;
            if self.consecutive_failures >= self.threshold {
                self.open = true;
            }
        } else {
            self.consecutive_failures = 0;
        }
    }
}

impl Default for GuardianBreaker {
    fn default() -> Self {
        Self::new(3)
    }
}

/// The text a request is matched against: the command for `run_command`,
/// the path for file tools, otherwise the tool name and its arguments.
pub fn subject(tool: &str, args: &Value) -> String {
    match tool {
        "run_command" => command_of(args).to_string(),
        "view_file"
        | "write_to_file"
        | "create_file"
        | "replace_file_content"
        | "edit_file"
        | "list_dir"
        | "list_directory"
        | "create_dir" => path_of(args).to_string(),
        _ => format!("{} {}", tool, args),
    }
}

fn command_of(args: &Value) -> &str {
    args.get("command")
        .or_else(|| args.get("CommandLine"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn path_of(args: &Value) -> &str {
    args.get("path")
        .or_else(|| args.get("target_file"))
        .or_else(|| args.get("TargetFile"))
        .or_else(|| args.get("AbsolutePath"))
        .or_else(|| args.get("directory_path"))
        .or_else(|| args.get("DirectoryPath"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn cwd_of<'a>(args: &'a Value, facts: &'a RuntimeFacts) -> PathBuf {
    match args
        .get("cwd")
        .or_else(|| args.get("Cwd"))
        .and_then(Value::as_str)
    {
        Some(c) if !c.is_empty() => lexical_abs(c, &facts.home, &facts.cwd),
        _ => facts.cwd.clone(),
    }
}

fn matches_subject(re: &Regex, subject: &str) -> bool {
    re.is_match(subject) || re.is_match(&normalize_command(subject))
}

/// Quotes and backslashes removed, whitespace collapsed, lower case.
/// Closes the cheap spellings that slip past a plain substring match
/// (`r''m -rf /`, `rm  -rf   /`, `RM -RF /`).
pub fn normalize_command(cmd: &str) -> String {
    strip_quotes(cmd).to_lowercase()
}

/// Split a shell line into simple commands on `;`, `&&`, `||`, `|`, `&`,
/// newlines, `$(`, `(`, `)` and backticks. Returns each segment and whether
/// it receives a pipe from the previous one.
fn segments(normalized: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut piped_next = false;
    let mut piped_current = false;
    let chars: Vec<char> = normalized.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let boundary = match (c, next) {
            ('&', Some('&')) | ('|', Some('|')) => {
                i += 1;
                true
            }
            ('|', _) => {
                piped_next = true;
                true
            }
            ('$', Some('(')) => {
                i += 1;
                true
            }
            (';' | '&' | '\n' | '`' | '(' | ')' | '{' | '}', _) => true,
            _ => false,
        };
        if boundary {
            if !current.trim().is_empty() {
                out.push((current.trim().to_string(), piped_current));
            }
            current.clear();
            piped_current = piped_next;
            piped_next = false;
        } else {
            current.push(c);
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        out.push((current.trim().to_string(), piped_current));
    }
    out
}

/// Wrappers that run the next word as the real command.
const WRAPPERS: [&str; 9] = [
    "sudo", "env", "nice", "nohup", "time", "command", "exec", "busybox", "doas",
];

/// Strip wrappers and `NAME=value` assignments; return the real command's
/// base name and its arguments.
fn real_command(tokens: &[&str]) -> Option<(String, Vec<String>)> {
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        let base = t.rsplit('/').next().unwrap_or(t);
        if WRAPPERS.contains(&base) {
            i += 1;
            // Skip the wrapper's own flags (and a numeric niceness).
            while i < tokens.len()
                && (tokens[i].starts_with('-') || tokens[i].parse::<i64>().is_ok())
            {
                i += 1;
            }
            continue;
        }
        if t.contains('=') && !t.starts_with('-') && !t.starts_with('/') {
            i += 1;
            continue;
        }
        let args = tokens[i + 1..].iter().map(|s| s.to_string()).collect();
        return Some((base.to_string(), args));
    }
    None
}

const CRITICAL_ROOTS: [&str; 16] = [
    "/", "/*", "~", "~/", "~/*", "$home", "${home}", "/home", "/root", "/etc", "/usr", "/boot",
    "/var", "/bin", "/lib", "/sbin",
];

fn is_critical_target(arg: &str) -> bool {
    let trimmed = if arg.len() > 1 {
        arg.trim_end_matches('/')
    } else {
        arg
    };
    let trimmed = if trimmed.is_empty() { "/" } else { trimmed };
    CRITICAL_ROOTS.contains(&trimmed)
        || CRITICAL_ROOTS.contains(&arg)
        || matches!(trimmed, "/home/*" | "/root/*" | "$home/*" | "${home}/*")
}

fn has_short_flag(args: &[String], flag: char) -> bool {
    args.iter()
        .any(|a| a.starts_with('-') && !a.starts_with("--") && a[1..].contains(flag))
}

const RAW_DISKS: [&str; 8] = [
    "/dev/sd",
    "/dev/nvme",
    "/dev/mmcblk",
    "/dev/vd",
    "/dev/hd",
    "/dev/xvd",
    "/dev/disk",
    "/dev/mapper/",
];

fn names_raw_disk(arg: &str) -> bool {
    RAW_DISKS.iter().any(|d| arg.starts_with(d))
}

/// Floor 1: commands nothing may approve.
pub fn hardline(cmd: &str) -> Option<String> {
    let norm = normalize_command(cmd);
    let squeezed: String = norm.chars().filter(|c| !c.is_whitespace()).collect();
    if squeezed.contains(":(){:|:&};:") {
        return Some("fork bomb".into());
    }
    if norm.contains("--no-preserve-root") {
        return Some("rm --no-preserve-root".into());
    }
    for disk in RAW_DISKS {
        if squeezed.contains(&format!(">{disk}")) {
            return Some(format!("redirect onto raw disk {disk}*"));
        }
    }
    for (segment, _) in segments(&norm) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();
        let Some((name, args)) = real_command(&tokens) else {
            continue;
        };
        let targets: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
        match name.as_str() {
            "rm" => {
                let recursive =
                    has_short_flag(&args, 'r') || args.iter().any(|a| a == "--recursive");
                if recursive {
                    if let Some(t) = targets.iter().find(|t| is_critical_target(t)) {
                        return Some(format!("recursive delete of {t}"));
                    }
                }
            }
            "chmod" | "chown" | "chgrp" => {
                // Arguments are lower case here, so `-R` reads as `-r`.
                let recursive =
                    has_short_flag(&args, 'r') || args.iter().any(|a| a == "--recursive");
                if recursive && targets.iter().any(|t| is_critical_target(t)) {
                    return Some(format!("recursive {name} of a system root"));
                }
            }
            n if n.starts_with("mkfs") || n == "wipefs" || n == "mkswap" => {
                if args.iter().any(|a| names_raw_disk(a)) {
                    return Some(format!("{n} on a raw disk"));
                }
            }
            "dd" => {
                if args
                    .iter()
                    .any(|a| a.strip_prefix("of=").is_some_and(names_raw_disk))
                {
                    return Some("dd onto a raw disk".into());
                }
            }
            "shred" => {
                if args.iter().any(|a| names_raw_disk(a)) {
                    return Some("shred of a raw disk".into());
                }
            }
            "kill" if args.iter().any(|a| a == "1" || a == "-1") => {
                return Some("kill of init or of every process".into());
            }
            _ => {}
        }
    }
    None
}

/// Floor 3: `sudo` reading a password from stdin (`-S`, `--stdin`, or a pipe
/// into sudo). A password in a command line ends up in history and logs.
pub fn sudo_piped_password(cmd: &str) -> Option<String> {
    // Case matters here: `sudo -S` reads stdin, `sudo -s` opens a shell.
    let stripped = strip_quotes(cmd);
    for (segment, piped_in) in segments(&stripped) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();
        let Some(pos) = tokens
            .iter()
            .position(|t| t.rsplit('/').next().map(str::to_lowercase).as_deref() == Some("sudo"))
        else {
            continue;
        };
        let flags: Vec<&str> = tokens[pos + 1..]
            .iter()
            .take_while(|t| t.starts_with('-'))
            .copied()
            .collect();
        if flags.contains(&"--stdin")
            || flags
                .iter()
                .any(|f| !f.starts_with("--") && f[1..].contains('S'))
        {
            return Some("sudo -S reads a password from stdin".into());
        }
        if segment.contains("<<") {
            return Some("sudo fed through a here-document or here-string".into());
        }
        if piped_in && pos == 0 {
            return Some("password piped into sudo".into());
        }
    }
    None
}

/// Quotes and backslashes removed, whitespace collapsed, case kept.
fn strip_quotes(cmd: &str) -> String {
    let stripped: String = cmd
        .replace("\\\n", " ")
        .chars()
        .filter(|c| !matches!(c, '\'' | '"' | '\\'))
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lexical absolute path: `~` and `$HOME` expanded, relative joined to
/// `base`, `.` and `..` folded. No disk access.
pub fn lexical_abs(path: &str, home: &Path, base: &Path) -> PathBuf {
    let expanded = if path == "~" || path == "$home" || path == "$HOME" {
        home.to_path_buf()
    } else if let Some(rest) = path
        .strip_prefix("~/")
        .or_else(|| path.strip_prefix("$HOME/"))
        .or_else(|| path.strip_prefix("$home/"))
        .or_else(|| path.strip_prefix("${HOME}/"))
    {
        home.join(rest)
    } else {
        PathBuf::from(path)
    };
    let joined = if expanded.is_relative() {
        base.join(expanded)
    } else {
        expanded
    };
    let mut out = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn is_protected(path: &Path, facts: &RuntimeFacts) -> Option<PathBuf> {
    facts
        .protected
        .iter()
        // The path is inside a protected entry, or a protected entry is
        // inside the path (deleting a parent deletes the runtime too).
        .find(|p| path.starts_with(p) || (p.starts_with(path) && path != Path::new("")))
        .cloned()
}

const MUTATING: [&str; 16] = [
    "rm", "rmdir", "mv", "shred", "truncate", "unlink", "cp", "ln", "chmod", "chown", "tee", "dd",
    "install", "rsync", "find", "sed",
];

/// Floor 2: the agent deleting or rewriting the runtime itself.
pub fn runtime_self_delete(tool: &str, args: &Value, facts: &RuntimeFacts) -> Option<String> {
    if facts.protected.is_empty() {
        return None;
    }
    match tool {
        "write_to_file" | "create_file" | "replace_file_content" | "edit_file" | "create_dir" => {
            let raw = path_of(args);
            if raw.is_empty() {
                return None;
            }
            let p = lexical_abs(raw, &facts.home, &facts.cwd);
            // Creating something *above* a protected path is harmless for a
            // file write; only writes inside a protected entry count here.
            let hit = facts.protected.iter().find(|r| p.starts_with(r))?;
            Some(format!(
                "{} would rewrite the runtime's own file {}",
                tool,
                hit.display()
            ))
        }
        "run_command" => {
            let cwd = cwd_of(args, facts);
            let original = command_of(args);
            // Paths are case-sensitive: segment the quote-stripped original,
            // compare verbs in lower case.
            let stripped: String = original
                .chars()
                .filter(|c| !matches!(c, '\'' | '"' | '\\'))
                .collect();
            for (segment, _) in segments(&stripped) {
                let tokens: Vec<&str> = segment.split_whitespace().collect();
                let redirect_targets: Vec<&str> = tokens
                    .iter()
                    .enumerate()
                    .filter_map(|(i, t)| {
                        if let Some(rest) = t.find('>').map(|i| t[i..].trim_start_matches('>')) {
                            if rest.is_empty() {
                                tokens.get(i + 1).copied()
                            } else {
                                Some(rest)
                            }
                        } else {
                            None
                        }
                    })
                    .collect();
                for t in &redirect_targets {
                    let p = lexical_abs(t, &facts.home, &cwd);
                    if let Some(hit) = facts.protected.iter().find(|r| p.starts_with(r)) {
                        return Some(format!(
                            "shell redirect would overwrite the runtime's own file {}",
                            hit.display()
                        ));
                    }
                }
                let lowered: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
                let lowered_refs: Vec<&str> = lowered.iter().map(String::as_str).collect();
                let Some((name, _)) = real_command(&lowered_refs) else {
                    continue;
                };
                if !MUTATING.contains(&name.as_str()) {
                    continue;
                }
                // `sed` only mutates with -i; `find` only with -delete/-exec.
                if name == "sed" && !lowered.iter().any(|t| t.starts_with("-i")) {
                    continue;
                }
                if name == "find" && !lowered.iter().any(|t| t == "-delete" || t == "-exec") {
                    continue;
                }
                // Deleting a parent deletes the runtime too, so deleting
                // verbs (and the source side of `mv`) also match a parent of
                // a protected entry. Other verbs must point inside one.
                let operands: Vec<&str> = tokens
                    .iter()
                    .skip_while(|t| t.to_lowercase().rsplit('/').next() != Some(name.as_str()))
                    .skip(1)
                    .filter(|t| !t.starts_with('-'))
                    .copied()
                    .collect();
                for (i, t) in operands.iter().enumerate() {
                    let deleting =
                        matches!(name.as_str(), "rm" | "rmdir" | "shred" | "unlink" | "find")
                            || (name == "mv" && i + 1 < operands.len());
                    let p = lexical_abs(t, &facts.home, &cwd);
                    let hit = if deleting {
                        is_protected(&p, facts)
                    } else {
                        facts.protected.iter().find(|r| p.starts_with(r)).cloned()
                    };
                    if let Some(hit) = hit {
                        return Some(format!(
                            "{} would delete or rewrite the runtime's own file {}",
                            name,
                            hit.display()
                        ));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// All four floors. Order only changes the reason given; every floor denies.
pub fn check_floors(
    tool: &str,
    args: &Value,
    policy: &CompiledPolicy,
    facts: &RuntimeFacts,
) -> Option<(Floor, String)> {
    if let Some(broken) = &policy.broken {
        return Some((Floor::OperatorDeny, broken.clone()));
    }
    if tool == "run_command" {
        let cmd = command_of(args);
        if let Some(why) = hardline(cmd) {
            return Some((Floor::Hardline, format!("hardline floor: {why}")));
        }
        if let Some(why) = sudo_piped_password(cmd) {
            return Some((Floor::SudoPipedPassword, format!("sudo floor: {why}")));
        }
    }
    if let Some(why) = runtime_self_delete(tool, args, facts) {
        return Some((Floor::RuntimeSelfDelete, format!("runtime floor: {why}")));
    }
    let subject = subject(tool, args);
    if let Some(re) = policy.deny.iter().find(|re| matches_subject(re, &subject)) {
        return Some((
            Floor::OperatorDeny,
            format!("operator deny pattern {:?} matched", re.as_str()),
        ));
    }
    None
}

/// Does this request need a decision (person or guardian)? `None` = clear.
fn needs_decision(tool: &str, args: &Value, base: &SafetyDecision) -> Option<String> {
    if let SafetyDecision::AskUser(reason) = base {
        return Some(reason.clone());
    }
    match tool {
        "run_command" if is_high_risk_command(command_of(args)) => {
            Some("dangerous command pattern".into())
        }
        "write_to_file" | "create_file" | "replace_file_content" | "edit_file"
            if is_high_risk_path(path_of(args)) =>
        {
            Some("write to an operating system path".into())
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Edges: policy file, presence, mode, runtime facts
// ---------------------------------------------------------------------------

/// `$AIEN_APPROVAL_POLICY`, else `~/basecamp/approval-policy.json`.
pub fn policy_path(home: &Path) -> PathBuf {
    std::env::var_os("AIEN_APPROVAL_POLICY")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join("basecamp/approval-policy.json"))
}

/// Missing file = empty policy. Unreadable or unparseable = deny everything.
pub fn load_policy(path: &Path) -> CompiledPolicy {
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<OperatorPolicy>(&text) {
            Ok(policy) => CompiledPolicy::compile(&policy),
            Err(err) => CompiledPolicy::unreadable(format!("{}: {}", path.display(), err)),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => CompiledPolicy::default(),
        Err(err) => CompiledPolicy::unreadable(format!("{}: {}", path.display(), err)),
    }
}

/// `$AIEN_APPROVAL_MODE`, else the policy file's mode, else manual. An
/// unknown word in the variable falls back to manual (the strictest).
pub fn resolve_mode(policy: &CompiledPolicy) -> ApprovalMode {
    match std::env::var("AIEN_APPROVAL_MODE") {
        Ok(v) => ApprovalMode::parse(&v).unwrap_or(ApprovalMode::Manual),
        Err(_) => policy.mode.unwrap_or(ApprovalMode::Manual),
    }
}

/// A person is present when stdin is a terminal and `AIEN_UNATTENDED` is not
/// set to a true value.
pub fn resolve_presence() -> Presence {
    use std::io::IsTerminal;
    let flagged = std::env::var("AIEN_UNATTENDED")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if !flagged && std::io::stdin().is_terminal() {
        Presence::Person
    } else {
        Presence::Unattended
    }
}

/// The runtime's own files: this binary, the policy file, the checkpoint
/// store and the session journals.
pub fn runtime_facts(home: &Path, policy_file: &Path) -> RuntimeFacts {
    let cwd = std::env::current_dir().unwrap_or_else(|_| home.to_path_buf());
    let mut protected = vec![
        lexical_abs(&policy_file.to_string_lossy(), home, &cwd),
        crate::checkpoint::store_root(home),
        home.join("basecamp/sessions"),
    ];
    if let Ok(exe) = std::env::current_exe() {
        protected.push(exe);
    }
    RuntimeFacts {
        protected,
        home: home.to_path_buf(),
        cwd,
    }
}

/// Everything the gate needs from the environment, read once per request so
/// an operator's edit to the policy file applies at once.
#[derive(Debug, Clone)]
pub struct GateContext {
    pub mode: ApprovalMode,
    pub presence: Presence,
    pub policy: CompiledPolicy,
    pub facts: RuntimeFacts,
}

impl GateContext {
    pub fn from_env() -> Self {
        let home = crate::platform::PlatformContext::detect().home_dir;
        let path = policy_path(&home);
        let policy = load_policy(&path);
        Self {
            mode: resolve_mode(&policy),
            presence: resolve_presence(),
            facts: runtime_facts(&home, &path),
            policy,
        }
    }
}

fn breaker() -> &'static std::sync::Mutex<GuardianBreaker> {
    static BREAKER: std::sync::OnceLock<std::sync::Mutex<GuardianBreaker>> =
        std::sync::OnceLock::new();
    BREAKER.get_or_init(|| std::sync::Mutex::new(GuardianBreaker::default()))
}

pub fn guardian_ready() -> bool {
    breaker().lock().map(|b| b.is_closed()).unwrap_or(false)
}

pub fn record_guardian(answer: &GuardianAnswer) {
    if let Ok(mut b) = breaker().lock() {
        b.record(answer);
    }
}

pub const GUARDIAN_PROMPT: &str = "You are the approval guardian for an agent's tool request on the operator's machine. \
Answer with exactly one word first: APPROVE if the request is routine and reversible, DENY if it would destroy data, \
leak secrets or harm the machine, ESCALATE if a person should decide. After the word you may give one short reason.";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn facts() -> RuntimeFacts {
        RuntimeFacts {
            protected: vec![
                PathBuf::from("/home/op/basecamp/approval-policy.json"),
                PathBuf::from("/home/op/basecamp/checkpoints"),
                PathBuf::from("/home/op/basecamp/sessions"),
                PathBuf::from("/opt/aien/bin/aien-cli"),
            ],
            home: PathBuf::from("/home/op"),
            cwd: PathBuf::from("/home/op/work"),
        }
    }

    fn policy(deny: &[&str], allow: &[&str]) -> CompiledPolicy {
        CompiledPolicy::compile(&OperatorPolicy {
            mode: None,
            deny: deny.iter().map(|s| s.to_string()).collect(),
            allow: allow.iter().map(|s| s.to_string()).collect(),
        })
    }

    fn run(
        tool: &str,
        args: &Value,
        base: &SafetyDecision,
        mode: ApprovalMode,
        presence: Presence,
        policy: &CompiledPolicy,
    ) -> Verdict {
        decide(&Gate {
            tool,
            args,
            base,
            mode,
            presence,
            policy,
            facts: &facts(),
            guardian_ready: true,
        })
    }

    fn cmd(c: &str) -> Value {
        json!({"command": c, "cwd": "/home/op/work"})
    }

    // Acceptance: an operator deny pattern blocks a command in every mode,
    // one test per mode, with a person present and unattended.
    fn operator_deny_blocks(mode: ApprovalMode) {
        let pol = policy(&[r"git\s+push"], &[r".*"]);
        for presence in [Presence::Person, Presence::Unattended] {
            let v = run(
                "run_command",
                &cmd("git push origin main"),
                &SafetyDecision::Allow,
                mode,
                presence,
                &pol,
            );
            assert!(
                matches!(
                    v,
                    Verdict::Deny {
                        by: Stage::Floor(Floor::OperatorDeny),
                        ..
                    }
                ),
                "mode {mode:?} presence {presence:?}: {v:?}"
            );
        }
        // Negative control: the same mode and policy let an unmatched command through.
        let v = run(
            "run_command",
            &cmd("git status"),
            &SafetyDecision::Allow,
            mode,
            Presence::Person,
            &pol,
        );
        assert!(matches!(v, Verdict::Allow { .. }), "mode {mode:?}: {v:?}");
    }

    #[test]
    fn operator_deny_blocks_in_manual_mode() {
        operator_deny_blocks(ApprovalMode::Manual);
    }

    #[test]
    fn operator_deny_blocks_in_smart_mode() {
        operator_deny_blocks(ApprovalMode::Smart);
    }

    #[test]
    fn operator_deny_blocks_in_yolo_mode() {
        operator_deny_blocks(ApprovalMode::Yolo);
    }

    #[test]
    fn operator_deny_blocks_in_off_mode() {
        operator_deny_blocks(ApprovalMode::Off);
    }

    #[test]
    fn operator_deny_outranks_the_allowlist_and_spelling_tricks() {
        let pol = policy(&[r"git push"], &[r"^git push"]);
        for spelling in ["git push", "git  push", "g''it push", "GIT PUSH --force"] {
            let v = run(
                "run_command",
                &cmd(spelling),
                &SafetyDecision::Allow,
                ApprovalMode::Yolo,
                Presence::Person,
                &pol,
            );
            assert!(
                matches!(
                    v,
                    Verdict::Deny {
                        by: Stage::Floor(Floor::OperatorDeny),
                        ..
                    }
                ),
                "{spelling}: {v:?}"
            );
        }
    }

    #[test]
    fn operator_deny_covers_file_tools_by_path() {
        let pol = policy(&[r"/secrets/"], &[]);
        let v = run(
            "write_to_file",
            &json!({"path": "/home/op/work/secrets/a.txt", "content": "x"}),
            &SafetyDecision::Allow,
            ApprovalMode::Off,
            Presence::Person,
            &pol,
        );
        assert!(
            matches!(
                v,
                Verdict::Deny {
                    by: Stage::Floor(Floor::OperatorDeny),
                    ..
                }
            ),
            "{v:?}"
        );
    }

    #[test]
    fn broken_policy_fails_closed_in_every_mode() {
        let pol = policy(&["(unclosed"], &[]);
        let unreadable = CompiledPolicy::unreadable("bad json".into());
        for mode in ALL_MODES {
            for p in [&pol, &unreadable] {
                let v = run(
                    "view_file",
                    &json!({"path": "/home/op/work/a"}),
                    &SafetyDecision::Allow,
                    mode,
                    Presence::Person,
                    p,
                );
                assert!(
                    matches!(
                        v,
                        Verdict::Deny {
                            by: Stage::Floor(Floor::OperatorDeny),
                            ..
                        }
                    ),
                    "{v:?}"
                );
            }
        }
        // A broken allow pattern grants nothing and is reported.
        let pol = policy(&[], &["(unclosed"]);
        assert_eq!(pol.warnings.len(), 1);
    }

    #[test]
    fn hardline_floor_holds_in_every_mode() {
        let pol = policy(&[], &[r".*"]);
        let bad = [
            "rm -rf /",
            "rm -rf /*",
            "sudo rm -fr ~",
            "rm -r -f $HOME",
            "cd /tmp && rm -Rf /home/",
            "r''m -rf /",
            "rm --recursive --force /etc",
            "rm -rf --no-preserve-root /x",
            "mkfs.ext4 /dev/nvme0n1p2",
            "dd if=/dev/zero of=/dev/sda bs=1M",
            "echo x > /dev/sda",
            ":(){ :|:& };:",
            "kill -9 -1",
            "chmod -R 777 /",
            "nice -n 10 rm -rf /usr",
            "X=1 rm -rf /boot",
        ];
        for mode in ALL_MODES {
            for c in bad {
                let v = run(
                    "run_command",
                    &cmd(c),
                    &SafetyDecision::Allow,
                    mode,
                    Presence::Person,
                    &pol,
                );
                assert!(
                    matches!(
                        v,
                        Verdict::Deny {
                            by: Stage::Floor(Floor::Hardline),
                            ..
                        }
                    ),
                    "{mode:?} {c}: {v:?}"
                );
            }
        }
    }

    #[test]
    fn hardline_negative_controls_pass() {
        for c in [
            "rm -rf /tmp/build",
            "rm -rf ./target",
            "rm -rf ~/work/old",
            "rm /home/op/a.txt",
            "dd if=/dev/zero of=/tmp/disk.img bs=1M count=10",
            "mkfs.ext4 /tmp/disk.img",
            "kill -9 4242",
            "chmod -R 755 ./bin",
            "echo hi > /dev/null",
            "cargo test",
        ] {
            assert_eq!(hardline(c), None, "{c}");
        }
    }

    #[test]
    fn sudo_password_floor_holds_in_every_mode() {
        let pol = policy(&[], &[r".*"]);
        for mode in ALL_MODES {
            for c in [
                "echo hunter2 | sudo -S apt install x",
                "sudo -S true <<< hunter2",
                "sudo --stdin ls",
                "printf pw | sudo ls",
                "sudo -kS ls",
            ] {
                let v = run(
                    "run_command",
                    &cmd(c),
                    &SafetyDecision::Allow,
                    mode,
                    Presence::Person,
                    &pol,
                );
                assert!(
                    matches!(
                        v,
                        Verdict::Deny {
                            by: Stage::Floor(Floor::SudoPipedPassword),
                            ..
                        }
                    ),
                    "{mode:?} {c}: {v:?}"
                );
            }
        }
        // Negative controls: plain sudo, and sudo -s (a shell, lower case).
        for c in ["sudo ls /root", "sudo -s", "sudo -u op ls | grep x"] {
            assert_eq!(sudo_piped_password(c), None, "{c}");
        }
    }

    #[test]
    fn runtime_floor_holds_in_every_mode() {
        let pol = policy(&[], &[r".*"]);
        let cases = [
            ("run_command", cmd("rm -rf ~/basecamp/checkpoints")),
            ("run_command", cmd("rm -rf /home/op/basecamp")),
            (
                "run_command",
                cmd("cd /tmp; mv /opt/aien/bin/aien-cli /tmp/x"),
            ),
            (
                "run_command",
                cmd("echo '{}' > ~/basecamp/approval-policy.json"),
            ),
            (
                "run_command",
                json!({"command": "rm -r sessions", "cwd": "/home/op/basecamp"}),
            ),
            (
                "run_command",
                cmd("sed -i s/deny/x/ /home/op/basecamp/approval-policy.json"),
            ),
            (
                "run_command",
                cmd("find /home/op/basecamp/sessions -delete"),
            ),
            (
                "write_to_file",
                json!({"path": "/home/op/basecamp/approval-policy.json", "content": "{}"}),
            ),
            (
                "replace_file_content",
                json!({"path": "../basecamp/checkpoints/ledger/s.jsonl", "target": "a", "replacement": "b"}),
            ),
        ];
        for mode in ALL_MODES {
            for (tool, args) in &cases {
                let v = run(
                    tool,
                    args,
                    &SafetyDecision::Allow,
                    mode,
                    Presence::Person,
                    &pol,
                );
                assert!(
                    matches!(
                        v,
                        Verdict::Deny {
                            by: Stage::Floor(Floor::RuntimeSelfDelete),
                            ..
                        }
                    ),
                    "{mode:?} {tool} {args}: {v:?}"
                );
            }
        }
        // Negative controls: reading the runtime's files and writing beside them.
        let f = facts();
        for (tool, args) in [
            ("run_command", cmd("cat ~/basecamp/approval-policy.json")),
            ("run_command", cmd("ls /home/op/basecamp/sessions")),
            ("run_command", cmd("rm -rf /home/op/basecamp/other")),
            (
                "write_to_file",
                json!({"path": "/home/op/basecamp/notes.md", "content": "x"}),
            ),
        ] {
            assert_eq!(runtime_self_delete(tool, &args, &f), None, "{tool} {args}");
        }
    }

    #[test]
    fn floors_come_before_mode_order_is_load_bearing() {
        // The policy-engine denial also survives yolo and off.
        let pol = policy(&[], &[]);
        for mode in ALL_MODES {
            let v = run(
                "write_to_file",
                &json!({"path": "/home/op/work/.env"}),
                &SafetyDecision::Deny("vault".into()),
                mode,
                Presence::Person,
                &pol,
            );
            assert_eq!(
                v,
                Verdict::Deny {
                    by: Stage::PolicyDeny,
                    reason: "vault".into()
                }
            );
        }
    }

    #[test]
    fn modes_after_the_floors() {
        let pol = policy(&[], &[]);
        let ask = SafetyDecision::AskUser("q".into());
        let a = cmd("make deploy");
        assert_eq!(
            run(
                "run_command",
                &a,
                &ask,
                ApprovalMode::Yolo,
                Presence::Unattended,
                &pol
            ),
            Verdict::Allow { by: Stage::Mode }
        );
        assert_eq!(
            run(
                "run_command",
                &a,
                &ask,
                ApprovalMode::Off,
                Presence::Person,
                &pol
            ),
            Verdict::Allow { by: Stage::Mode }
        );
        assert_eq!(
            run(
                "run_command",
                &a,
                &ask,
                ApprovalMode::Manual,
                Presence::Person,
                &pol
            ),
            Verdict::AskPerson { reason: "q".into() }
        );
        assert_eq!(
            run(
                "run_command",
                &a,
                &ask,
                ApprovalMode::Smart,
                Presence::Person,
                &pol
            ),
            Verdict::AskGuardian { reason: "q".into() }
        );
    }

    #[test]
    fn unattended_defaults_to_deny_but_clear_requests_pass() {
        let pol = policy(&[], &[]);
        for mode in [ApprovalMode::Manual, ApprovalMode::Smart] {
            let v = run(
                "run_command",
                &cmd("shutdown -h now"),
                &SafetyDecision::Allow,
                mode,
                Presence::Unattended,
                &pol,
            );
            assert!(
                matches!(
                    v,
                    Verdict::Deny {
                        by: Stage::Presence,
                        ..
                    }
                ),
                "{v:?}"
            );
            let v = run(
                "run_command",
                &cmd("cargo test"),
                &SafetyDecision::Allow,
                mode,
                Presence::Unattended,
                &pol,
            );
            assert_eq!(v, Verdict::Allow { by: Stage::Clear });
        }
    }

    #[test]
    fn allowlist_skips_the_question_not_a_floor() {
        let pol = policy(&[], &[r"^shutdown -h now$"]);
        let v = run(
            "run_command",
            &cmd("shutdown -h now"),
            &SafetyDecision::Allow,
            ApprovalMode::Manual,
            Presence::Unattended,
            &pol,
        );
        assert_eq!(
            v,
            Verdict::Allow {
                by: Stage::Allowlist
            }
        );
        let v = run(
            "run_command",
            &cmd("rm -rf /"),
            &SafetyDecision::Allow,
            ApprovalMode::Manual,
            Presence::Person,
            &policy(&[], &[r".*"]),
        );
        assert!(matches!(
            v,
            Verdict::Deny {
                by: Stage::Floor(Floor::Hardline),
                ..
            }
        ));
    }

    #[test]
    fn guardian_answers_and_breaker() {
        assert_eq!(parse_guardian_reply("APPROVE"), GuardianAnswer::Approve);
        assert_eq!(
            parse_guardian_reply(" escalate: unsure"),
            GuardianAnswer::Escalate
        );
        assert!(matches!(
            parse_guardian_reply("DENY wipes data"),
            GuardianAnswer::Deny(_)
        ));
        assert!(matches!(
            parse_guardian_reply("Sure, go ahead"),
            GuardianAnswer::Failed(_)
        ));

        assert_eq!(
            after_guardian(&GuardianAnswer::Approve, "r".into()),
            Verdict::Allow {
                by: Stage::Guardian
            }
        );
        assert!(matches!(
            after_guardian(&GuardianAnswer::Deny("x".into()), "r".into()),
            Verdict::Deny {
                by: Stage::Guardian,
                ..
            }
        ));
        for a in [GuardianAnswer::Escalate, GuardianAnswer::Failed("t".into())] {
            assert_eq!(
                after_guardian(&a, "r".into()),
                Verdict::AskPerson { reason: "r".into() }
            );
        }

        let mut b = GuardianBreaker::new(3);
        let fail = GuardianAnswer::Failed("timeout".into());
        b.record(&fail);
        b.record(&fail);
        b.record(&GuardianAnswer::Escalate); // a real answer resets the count
        b.record(&fail);
        b.record(&fail);
        assert!(b.is_closed());
        b.record(&fail);
        assert!(!b.is_closed());
        b.record(&GuardianAnswer::Approve); // stays open for the session
        assert!(!b.is_closed());

        // An open breaker sends smart mode to the person.
        let pol = policy(&[], &[]);
        let v = decide(&Gate {
            tool: "run_command",
            args: &cmd("make deploy"),
            base: &SafetyDecision::AskUser("q".into()),
            mode: ApprovalMode::Smart,
            presence: Presence::Person,
            policy: &pol,
            facts: &facts(),
            guardian_ready: b.is_closed(),
        });
        assert_eq!(v, Verdict::AskPerson { reason: "q".into() });
    }

    #[test]
    fn decide_is_deterministic() {
        let pol = policy(&["x"], &["y"]);
        let a = cmd("echo y");
        let first = run(
            "run_command",
            &a,
            &SafetyDecision::Allow,
            ApprovalMode::Manual,
            Presence::Person,
            &pol,
        );
        for _ in 0..10 {
            assert_eq!(
                run(
                    "run_command",
                    &a,
                    &SafetyDecision::Allow,
                    ApprovalMode::Manual,
                    Presence::Person,
                    &pol
                ),
                first
            );
        }
    }
}
