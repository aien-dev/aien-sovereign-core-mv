//! `aien agents sync` and `aien agents check` (aien-architecture ADR 0038,
//! "agent guidance travels with the tree").
//!
//! Reads the agent-agnostic manifest `agents.toml` at the repo root, validates
//! it, and translates it into the files each coding agent reads:
//!
//! - claude-code: hook entries in `.claude/settings.json` (merged, never
//!   clobbered: our matcher groups carry the `aien-agents` marker key and only
//!   those are replaced) and skill copies under `.claude/skills/<name>/`.
//! - codex: one block in `AGENTS.md` between `<!-- aien-agents:begin -->` and
//!   `<!-- aien-agents:end -->`.
//! - hermes: skill copies under `$HERMES_HOME/skills/aien/<component>-<skill>/`
//!   (global scope only).
//!
//! Every write is computed first as a [`Plan`]; `sync` applies it, `--dry-run`
//! prints it, `check` fails when any step would change a file. Copied skill
//! directories carry a [`SKILL_MARKER`] file so a later sync can remove the
//! ones the manifest no longer lists without touching skills it did not write.
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use toml::Spanned;

pub const MANIFEST: &str = "agents.toml";
/// Marker key on every `.claude/settings.json` matcher group we write.
pub const MARKER_KEY: &str = "aien-agents";
/// Marker file inside every skill directory we copy.
pub const SKILL_MARKER: &str = ".aien-agents-sync";
pub const BLOCK_BEGIN: &str = "<!-- aien-agents:begin -->";
pub const BLOCK_END: &str = "<!-- aien-agents:end -->";
const EDIT_MATCHER: &str = "Edit|Write|MultiEdit|NotebookEdit";
const HOOK_TIMEOUT_SECS: u64 = 10;

// ------------------------------------------------------------------
// Manifest
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Event {
    PreCommand,
    PreEdit,
    PostEdit,
    SessionStart,
}

impl Event {
    pub const ALL: [Event; 4] = [
        Event::PreCommand,
        Event::PreEdit,
        Event::PostEdit,
        Event::SessionStart,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Event::PreCommand => "pre-command",
            Event::PreEdit => "pre-edit",
            Event::PostEdit => "post-edit",
            Event::SessionStart => "session-start",
        }
    }

    fn parse(s: &str) -> Option<Event> {
        Event::ALL.into_iter().find(|e| e.as_str() == s)
    }

    /// Claude Code event name and tool matcher for this manifest event.
    fn claude(self) -> (&'static str, Option<&'static str>) {
        match self {
            Event::PreCommand => ("PreToolUse", Some("Bash")),
            Event::PreEdit => ("PreToolUse", Some(EDIT_MATCHER)),
            Event::PostEdit => ("PostToolUse", Some(EDIT_MATCHER)),
            Event::SessionStart => ("SessionStart", None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// Repo-relative directory holding SKILL.md, as written in the manifest.
    pub path: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct HookEntry {
    pub event: Event,
    /// Repo-relative script path, as written in the manifest.
    pub command: String,
    /// One-line rule taken from the script's header comment.
    pub rule: String,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub root: PathBuf,
    pub component: String,
    pub tree: String,
    pub skills: Vec<SkillEntry>,
    pub hooks: Vec<HookEntry>,
    pub allow: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", MANIFEST, self.line, self.message)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    schema: Spanned<i64>,
    component: Spanned<RawComponent>,
    #[serde(default)]
    skills: Vec<RawSkill>,
    #[serde(default)]
    hooks: Vec<RawHook>,
    attachments: Option<Spanned<RawAttachments>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawComponent {
    name: Spanned<String>,
    tree: Spanned<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSkill {
    path: Spanned<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHook {
    event: Spanned<String>,
    command: Spanned<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAttachments {
    #[serde(default)]
    allow: Vec<Spanned<String>>,
}

fn line_of(text: &str, offset: usize) -> usize {
    let end = offset.min(text.len());
    text.as_bytes()[..end]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1
}

fn is_safe_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// Repo-relative path made of plain segments only (no absolute paths, no
/// `..`, no characters that would need shell quoting in a hook command).
fn is_safe_rel_path(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('/')
        && s.split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'/')
}

/// Value of a `field:` line in the leading `---` frontmatter block.
pub(crate) fn frontmatter_field(content: &str, field: &str) -> Option<String> {
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return None;
    }
    let prefix = format!("{field}:");
    for line in lines {
        let t = line.trim();
        if t == "---" {
            break;
        }
        if let Some(rest) = t.strip_prefix(&prefix) {
            let v = rest.trim().trim_matches('"').trim_matches('\'').to_string();
            return (!v.is_empty()).then_some(v);
        }
    }
    None
}

/// One-line rule from a hook script header: the first comment paragraph
/// after the shebang, without the `<file>: <event> hook.` lead-in, cut to its
/// first sentence.
pub fn hook_rule(script: &str) -> Option<String> {
    let mut para = Vec::new();
    for line in script.lines().skip_while(|l| l.starts_with("#!")) {
        let t = line.trim_end();
        match t.strip_prefix('#') {
            Some(rest) if !rest.trim().is_empty() => para.push(rest.trim().to_string()),
            Some(_) if para.is_empty() => continue,
            _ => break,
        }
    }
    let mut text = para.join(" ");
    if let Some((head, rest)) = text.split_once(": ") {
        if head.ends_with(".sh") && !head.contains(' ') {
            text = rest.to_string();
        }
    }
    if let Some((head, rest)) = text.split_once(". ") {
        if head.ends_with(" hook") && head.split_whitespace().count() <= 3 {
            text = rest.to_string();
        }
    }
    let first = first_sentence(&text).trim().to_string();
    (!first.is_empty()).then_some(first)
}

/// Text up to the first `. ` that sits outside backticks and is followed by
/// a capital letter (so `...` inside a quoted command does not end it).
fn first_sentence(text: &str) -> &str {
    let b = text.as_bytes();
    let mut in_code = false;
    for i in 0..b.len() {
        match b[i] {
            b'`' => in_code = !in_code,
            b'.' if !in_code
                && b.get(i + 1) == Some(&b' ')
                && b.get(i + 2).is_some_and(|c| c.is_ascii_uppercase()) =>
            {
                return &text[..=i];
            }
            _ => {}
        }
    }
    text
}

/// Parse and validate `agents.toml` text against the repo at `root`.
pub fn parse_manifest(root: &Path, text: &str) -> Result<Manifest, Vec<ManifestError>> {
    let raw: RawManifest = toml::from_str(text).map_err(|e| {
        let line = e.span().map(|s| line_of(text, s.start)).unwrap_or(1);
        vec![ManifestError {
            line,
            message: e.message().trim().to_string(),
        }]
    })?;
    let mut errors = Vec::new();
    let mut err = |offset: usize, message: String| {
        errors.push(ManifestError {
            line: line_of(text, offset),
            message,
        })
    };

    if *raw.schema.get_ref() != 1 {
        err(
            raw.schema.span().start,
            format!("unsupported schema {} (expected 1)", raw.schema.get_ref()),
        );
    }
    let component = raw.component.into_inner();
    let name = component.name.get_ref().clone();
    if !is_safe_name(&name) {
        err(
            component.name.span().start,
            format!("component name {name:?} must be lowercase letters, digits, - or _"),
        );
    }
    let tree = component.tree.get_ref().clone();
    if tree != "core" && tree != "attachment" {
        err(
            component.tree.span().start,
            format!("unknown tree {tree:?} (expected core or attachment)"),
        );
    }

    let mut skills = Vec::new();
    let mut seen_skills = BTreeSet::new();
    for s in &raw.skills {
        let at = s.path.span().start;
        let p = s.path.get_ref();
        if !is_safe_rel_path(p) {
            err(
                at,
                format!("skill path {p:?} must be a plain repo-relative path"),
            );
            continue;
        }
        let dir = root.join(p);
        if !dir.is_dir() {
            err(at, format!("skill directory {p} does not exist"));
            continue;
        }
        let skill_md = dir.join("SKILL.md");
        let content = match fs::read_to_string(&skill_md) {
            Ok(c) => c,
            Err(_) => {
                err(at, format!("skill directory {p} has no SKILL.md"));
                continue;
            }
        };
        let base = p.rsplit('/').next().unwrap_or(p).to_string();
        let sname = frontmatter_field(&content, "name").unwrap_or(base);
        if !is_safe_name(&sname) {
            err(at, format!("skill name {sname:?} in {p}/SKILL.md must be lowercase letters, digits, - or _"));
            continue;
        }
        if !seen_skills.insert(sname.clone()) {
            err(at, format!("duplicate skill name {sname:?}"));
            continue;
        }
        let description = match frontmatter_field(&content, "description") {
            Some(d) => d,
            None => {
                err(at, format!("{p}/SKILL.md frontmatter has no description"));
                continue;
            }
        };
        skills.push(SkillEntry {
            path: p.clone(),
            name: sname,
            description,
        });
    }

    let mut hooks = Vec::new();
    for h in &raw.hooks {
        let ev = h.event.get_ref();
        let event = match Event::parse(ev) {
            Some(e) => Some(e),
            None => {
                err(
                    h.event.span().start,
                    format!(
                        "unknown hook event {ev:?} (expected pre-command, pre-edit, post-edit or session-start)"
                    ),
                );
                None
            }
        };
        let at = h.command.span().start;
        let c = h.command.get_ref();
        if !is_safe_rel_path(c) {
            err(
                at,
                format!("hook command {c:?} must be a plain repo-relative path"),
            );
            continue;
        }
        let file = root.join(c);
        let script = match fs::read_to_string(&file) {
            Ok(s) if file.is_file() => s,
            _ => {
                err(at, format!("hook file {c} does not exist"));
                continue;
            }
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&file)
                .map(|m| m.permissions().mode())
                .unwrap_or(0);
            if mode & 0o111 == 0 {
                err(at, format!("hook file {c} is not executable"));
                continue;
            }
        }
        if let Some(event) = event {
            hooks.push(HookEntry {
                event,
                command: c.clone(),
                rule: hook_rule(&script).unwrap_or_else(|| "(no rule in header comment)".into()),
            });
        }
    }

    let mut allow = Vec::new();
    if let Some(att) = raw.attachments {
        if tree != "core" {
            err(
                att.span().start,
                "[attachments] is only allowed when tree = \"core\"".to_string(),
            );
        }
        for a in att.into_inner().allow {
            if !is_safe_name(a.get_ref()) {
                err(
                    a.span().start,
                    format!(
                        "attachment name {:?} must be lowercase letters, digits, - or _",
                        a.get_ref()
                    ),
                );
            } else {
                allow.push(a.into_inner());
            }
        }
    }

    if !errors.is_empty() {
        errors.sort_by_key(|e| e.line);
        return Err(errors);
    }
    Ok(Manifest {
        root: root.to_path_buf(),
        component: name,
        tree,
        skills,
        hooks,
        allow,
    })
}

pub fn load_manifest(root: &Path) -> Result<Manifest, Vec<ManifestError>> {
    let text = fs::read_to_string(root.join(MANIFEST)).map_err(|e| {
        vec![ManifestError {
            line: 0,
            message: format!("cannot read {}: {e}", root.join(MANIFEST).display()),
        }]
    })?;
    parse_manifest(root, &text)
}

/// Walk up from `start` to the first directory holding `agents.toml`,
/// stopping at the repository root (the first directory holding `.git`).
pub fn find_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join(MANIFEST).is_file() {
            return Some(d.to_path_buf());
        }
        if d.join(".git").exists() {
            return None;
        }
        dir = d.parent();
    }
    None
}

// ------------------------------------------------------------------
// Plan
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agent {
    ClaudeCode,
    Codex,
    Hermes,
}

impl Agent {
    pub fn as_str(self) -> &'static str {
        match self {
            Agent::ClaudeCode => "claude-code",
            Agent::Codex => "codex",
            Agent::Hermes => "hermes",
        }
    }
    fn supports(self, scope: Scope) -> bool {
        !matches!(
            (self, scope),
            (Agent::Codex, Scope::Global) | (Agent::Hermes, Scope::Project)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Project,
    Global,
}

/// Where global-scope files go. Tests point these at tempdirs.
#[derive(Debug, Clone)]
pub struct Homes {
    /// Claude Code config dir (`$CLAUDE_CONFIG_DIR`, default `~/.claude`).
    pub claude: PathBuf,
    /// Hermes skills dir (`$HERMES_HOME/skills`, default `~/.hermes/skills`).
    pub hermes_skills: PathBuf,
}

impl Homes {
    pub fn from_env() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let claude = std::env::var_os("CLAUDE_CONFIG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".claude"));
        let hermes = std::env::var_os("HERMES_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".hermes"));
        Homes {
            claude,
            hermes_skills: hermes.join("skills"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Write {
        path: PathBuf,
        bytes: Vec<u8>,
        executable: bool,
    },
    Remove {
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Create,
    Update,
    Unchanged,
    Remove,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Create => "create",
            Status::Update => "update",
            Status::Unchanged => "unchanged",
            Status::Remove => "remove",
        }
    }
}

impl Op {
    pub fn path(&self) -> &Path {
        match self {
            Op::Write { path, .. } | Op::Remove { path } => path,
        }
    }

    pub fn status(&self) -> Status {
        match self {
            Op::Write { path, bytes, .. } => match fs::read(path) {
                Ok(cur) if &cur == bytes => Status::Unchanged,
                Ok(_) => Status::Update,
                Err(_) => Status::Create,
            },
            Op::Remove { .. } => Status::Remove,
        }
    }

    fn apply(&self) -> Result<(), String> {
        match self {
            Op::Write {
                path,
                bytes,
                executable,
            } => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
                }
                fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = if *executable { 0o755 } else { 0o644 };
                    fs::set_permissions(path, fs::Permissions::from_mode(mode))
                        .map_err(|e| format!("chmod {}: {e}", path.display()))?;
                }
                #[cfg(not(unix))]
                let _ = executable;
                Ok(())
            }
            Op::Remove { path } => {
                let r = if path.is_dir() {
                    fs::remove_dir_all(path)
                } else {
                    fs::remove_file(path)
                };
                r.map_err(|e| format!("remove {}: {e}", path.display()))
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Plan {
    pub steps: Vec<(Agent, Op)>,
}

impl Plan {
    /// Steps that would change something on disk.
    pub fn changes(&self) -> Vec<(Agent, &Op, Status)> {
        self.steps
            .iter()
            .map(|(a, op)| (*a, op, op.status()))
            .filter(|(_, _, s)| *s != Status::Unchanged)
            .collect()
    }

    pub fn apply(&self) -> Result<usize, String> {
        let mut n = 0;
        for (_, op) in &self.steps {
            if op.status() != Status::Unchanged {
                op.apply()?;
                n += 1;
            }
        }
        Ok(n)
    }
}

pub fn agents_for(choice: &str, scope: Scope) -> Result<Vec<Agent>, String> {
    let agent = match choice {
        "all" => {
            return Ok([Agent::ClaudeCode, Agent::Codex, Agent::Hermes]
                .into_iter()
                .filter(|a| a.supports(scope))
                .collect())
        }
        "claude-code" => Agent::ClaudeCode,
        "codex" => Agent::Codex,
        "hermes" => Agent::Hermes,
        other => {
            return Err(format!(
                "unknown agent {other:?} (expected claude-code, codex, hermes or all)"
            ))
        }
    };
    if !agent.supports(scope) {
        return Err(match agent {
            Agent::Codex => {
                "codex: only --scope project is supported (the block lives in the repo AGENTS.md)"
                    .into()
            }
            _ => {
                "hermes: only --scope global is supported (skills go to $HERMES_HOME/skills)".into()
            }
        });
    }
    Ok(vec![agent])
}

pub fn build_plan(
    m: &Manifest,
    agents: &[Agent],
    scope: Scope,
    homes: &Homes,
) -> Result<Plan, String> {
    let mut plan = Plan::default();
    for &agent in agents {
        let ops = match agent {
            Agent::ClaudeCode => claude_code_ops(m, scope, homes)?,
            Agent::Codex => codex_ops(m)?,
            Agent::Hermes => {
                let dest = homes.hermes_skills.join("aien");
                skill_copy_ops(m, &dest, &format!("{}-", m.component))?
            }
        };
        plan.steps.extend(ops.into_iter().map(|op| (agent, op)));
    }
    Ok(plan)
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn claude_code_ops(m: &Manifest, scope: Scope, homes: &Homes) -> Result<Vec<Op>, String> {
    let (base, abs_root) = match scope {
        Scope::Project => (m.root.join(".claude"), None),
        Scope::Global => (
            homes.claude.clone(),
            Some(m.root.canonicalize().unwrap_or_else(|_| m.root.clone())),
        ),
    };
    let mut ops = skill_copy_ops(m, &base.join("skills"), "")?;

    let settings_path = base.join("settings.json");
    let existing = match fs::read_to_string(&settings_path) {
        Ok(t) => Some(
            serde_json::from_str::<Value>(&t)
                .map_err(|e| format!("{} is not valid JSON: {e}", settings_path.display()))?,
        ),
        Err(_) => None,
    };
    let mut settings = match existing.clone() {
        Some(Value::Object(o)) => o,
        Some(_) => return Err(format!("{} is not a JSON object", settings_path.display())),
        None => Map::new(),
    };
    let mut hooks = match settings.remove("hooks") {
        Some(Value::Object(o)) => o,
        Some(_) => {
            return Err(format!(
                "{}: \"hooks\" is not an object",
                settings_path.display()
            ))
        }
        None => Map::new(),
    };

    // Drop the groups a previous sync of this component wrote.
    let ours = |g: &Value| g.get(MARKER_KEY).and_then(Value::as_str) == Some(&m.component);
    for list in hooks.values_mut() {
        if let Value::Array(a) = list {
            a.retain(|g| !ours(g));
        }
    }

    for event in Event::ALL {
        let entries: Vec<Value> = m
            .hooks
            .iter()
            .filter(|h| h.event == event)
            .map(|h| {
                let command = match &abs_root {
                    None => format!("\"$CLAUDE_PROJECT_DIR\"/{}", h.command),
                    Some(r) => shell_quote(&r.join(&h.command).display().to_string()),
                };
                json!({"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECS})
            })
            .collect();
        if entries.is_empty() {
            continue;
        }
        let (cc_event, matcher) = event.claude();
        let mut group = Map::new();
        if let Some(mt) = matcher {
            group.insert("matcher".into(), json!(mt));
        }
        group.insert("hooks".into(), Value::Array(entries));
        group.insert(MARKER_KEY.into(), json!(m.component));
        let list = hooks
            .entry(cc_event.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        match list {
            Value::Array(a) => a.push(Value::Object(group)),
            _ => {
                return Err(format!(
                    "{}: hooks.{cc_event} is not an array",
                    settings_path.display()
                ))
            }
        }
    }
    hooks.retain(|_, v| !matches!(v, Value::Array(a) if a.is_empty()));
    if !hooks.is_empty() {
        settings.insert("hooks".into(), Value::Object(hooks));
    }

    // Leave the file alone when there is nothing to say and nothing was there.
    if existing.is_some() || !settings.is_empty() {
        let mut text =
            serde_json::to_string_pretty(&Value::Object(settings)).map_err(|e| e.to_string())?;
        text.push('\n');
        ops.push(Op::Write {
            path: settings_path,
            bytes: text.into_bytes(),
            executable: false,
        });
    }
    Ok(ops)
}

fn marker_text(m: &Manifest, skill: &SkillEntry) -> String {
    format!(
        "# Written by `aien agents sync` from agents.toml. Removed or replaced on the next sync.\ncomponent = {}\nsource = {}\n",
        m.component, skill.path
    )
}

fn marker_component(dir: &Path) -> Option<String> {
    let text = fs::read_to_string(dir.join(SKILL_MARKER)).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("component = "))
        .map(|s| s.trim().to_string())
}

fn walk_files(dir: &Path, rel: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let ft = e.file_type().map_err(|err| err.to_string())?;
        let r = rel.join(e.file_name());
        if ft.is_dir() {
            walk_files(&e.path(), &r, out)?;
        } else if ft.is_file() {
            out.push(r);
        }
    }
    Ok(())
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .map(|md| md.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// Files that stay in the source tree: crumbs govern their own directory
/// (copying one would plant a false crumb), and the marker is ours.
fn is_never_copied(rel: &Path) -> bool {
    matches!(
        rel.file_name().and_then(|n| n.to_str()),
        Some(".crumb" | ".crumb.local" | "crumb.lock" | SKILL_MARKER)
    )
}

/// Copy every manifest skill to `dest/<prefix><name>/`, and remove copies of
/// this component's skills that the manifest no longer lists.
fn skill_copy_ops(m: &Manifest, dest: &Path, prefix: &str) -> Result<Vec<Op>, String> {
    let mut ops = Vec::new();
    let mut wanted = BTreeSet::new();
    for s in &m.skills {
        let dir_name = format!("{prefix}{}", s.name);
        wanted.insert(dir_name.clone());
        let target = dest.join(&dir_name);
        if target.exists() && marker_component(&target).as_deref() != Some(m.component.as_str()) {
            return Err(format!(
                "{} exists and was not written by `aien agents sync` for {}; refusing to overwrite it",
                target.display(),
                m.component
            ));
        }
        let src = m.root.join(&s.path);
        let mut files = Vec::new();
        walk_files(&src, Path::new(""), &mut files)?;
        let mut keep = BTreeSet::new();
        for rel in files {
            if is_never_copied(&rel) {
                continue;
            }
            let from = src.join(&rel);
            let bytes = fs::read(&from).map_err(|e| format!("read {}: {e}", from.display()))?;
            ops.push(Op::Write {
                path: target.join(&rel),
                bytes,
                executable: is_executable(&from),
            });
            keep.insert(rel);
        }
        ops.push(Op::Write {
            path: target.join(SKILL_MARKER),
            bytes: marker_text(m, s).into_bytes(),
            executable: false,
        });
        if target.is_dir() {
            let mut present = Vec::new();
            walk_files(&target, Path::new(""), &mut present)?;
            for rel in present {
                if rel.as_os_str() != SKILL_MARKER && !keep.contains(&rel) {
                    ops.push(Op::Remove {
                        path: target.join(rel),
                    });
                }
            }
        }
    }
    if let Ok(entries) = fs::read_dir(dest) {
        let mut stale: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .filter(|p| {
                let n = p.file_name().unwrap_or_default().to_string_lossy();
                n.starts_with(prefix) && !wanted.contains(n.as_ref())
            })
            .filter(|p| marker_component(p).as_deref() == Some(m.component.as_str()))
            .collect();
        stale.sort();
        ops.extend(stale.into_iter().map(|path| Op::Remove { path }));
    }
    Ok(ops)
}

pub fn codex_block(m: &Manifest) -> String {
    let mut b = String::new();
    b.push_str(BLOCK_BEGIN);
    b.push('\n');
    b.push_str("## Agent guidance (generated from agents.toml)\n\n");
    b.push_str(&format!(
        "Written by `aien agents sync`; edit `agents.toml`, the skills or the hooks, then re-run it. Component `{}` (tree: {}).\n",
        m.component, m.tree
    ));
    if !m.skills.is_empty() {
        b.push_str("\nSkills (read the one that matches your task before you start):\n\n");
        for s in &m.skills {
            b.push_str(&format!(
                "- [{}]({}/SKILL.md): {}\n",
                s.name, s.path, s.description
            ));
        }
    }
    if !m.hooks.is_empty() {
        b.push_str(
            "\nRules the hooks in this repo enforce (follow them even where hooks do not run):\n\n",
        );
        for h in &m.hooks {
            b.push_str(&format!(
                "- {} [`{}`]({}): {}\n",
                h.event.as_str(),
                h.command.rsplit('/').next().unwrap_or(&h.command),
                h.command,
                h.rule
            ));
        }
    }
    b.push_str(BLOCK_END);
    b.push('\n');
    b
}

/// Replace the marked block in `doc`, or append it when absent.
pub fn splice_block(doc: &str, block: &str) -> Result<String, String> {
    let begins = doc.matches(BLOCK_BEGIN).count();
    let ends = doc.matches(BLOCK_END).count();
    match (begins, ends) {
        (0, 0) => {
            let mut out = doc.to_string();
            if !out.is_empty() {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push('\n');
            }
            out.push_str(block);
            Ok(out)
        }
        (1, 1) => {
            let start = doc.find(BLOCK_BEGIN).unwrap_or(0);
            let end_marker = doc.find(BLOCK_END).unwrap_or(0);
            if end_marker < start {
                return Err(
                    "AGENTS.md: aien-agents end marker comes before the begin marker".into(),
                );
            }
            let mut end = end_marker + BLOCK_END.len();
            if doc[end..].starts_with('\n') {
                end += 1;
            }
            Ok(format!("{}{}{}", &doc[..start], block, &doc[end..]))
        }
        _ => Err(format!(
            "AGENTS.md has {begins} begin and {ends} end aien-agents markers; expected one of each"
        )),
    }
}

fn codex_ops(m: &Manifest) -> Result<Vec<Op>, String> {
    let path = m.root.join("AGENTS.md");
    let doc = fs::read_to_string(&path).unwrap_or_default();
    let out = splice_block(&doc, &codex_block(m))?;
    Ok(vec![Op::Write {
        path,
        bytes: out.into_bytes(),
        executable: false,
    }])
}

// ------------------------------------------------------------------
// CLI
// ------------------------------------------------------------------

const USAGE: &str = "usage: aien agents sync [--agent claude-code|codex|hermes|all] [--scope project|global] [--dry-run]
       aien agents check [--agent claude-code|codex|hermes|all] [--scope project|global]

Reads agents.toml at the repo root (ADR 0038) and writes the files each agent reads.
  claude-code  .claude/settings.json hooks + .claude/skills/<name>/ (project or global)
  codex        aien-agents block in AGENTS.md (project only)
  hermes       $HERMES_HOME/skills/aien/<component>-<skill>/ (global only)
check exits 1 when the manifest is invalid or a generated file is stale.";

struct Opts {
    agent: String,
    scope: Scope,
    dry_run: bool,
}

fn parse_opts(args: &[String], allow_dry_run: bool) -> Result<Opts, String> {
    let mut o = Opts {
        agent: "all".into(),
        scope: Scope::Project,
        dry_run: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (key, inline) = match a.split_once('=') {
            Some((k, v)) => (k, Some(v.to_string())),
            None => (a.as_str(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or_else(|| format!("{key} needs a value"))
        };
        match key {
            "--agent" => o.agent = value()?,
            "--scope" => {
                o.scope = match value()?.as_str() {
                    "project" => Scope::Project,
                    "global" => Scope::Global,
                    other => {
                        return Err(format!(
                            "unknown scope {other:?} (expected project or global)"
                        ))
                    }
                }
            }
            "--dry-run" if allow_dry_run && inline.is_none() => o.dry_run = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(o)
}

fn display_path(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .map(|r| r.display().to_string())
        .unwrap_or_else(|_| p.display().to_string())
}

/// Entry point for `aien agents ...`. Returns the process exit code.
pub fn run(args: &[String], cwd: &Path, homes: &Homes) -> i32 {
    let (sub, rest) = match args.split_first() {
        Some((s, r)) if s == "sync" || s == "check" => (s.as_str(), r),
        _ => {
            eprintln!("{USAGE}");
            return 2;
        }
    };
    let opts = match parse_opts(rest, sub == "sync") {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aien agents {sub}: {e}\n{USAGE}");
            return 2;
        }
    };
    let agents = match agents_for(&opts.agent, opts.scope) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("aien agents {sub}: {e}");
            return 2;
        }
    };
    let Some(root) = find_root(cwd) else {
        eprintln!(
            "aien agents {sub}: no {MANIFEST} between {} and the repository root",
            cwd.display()
        );
        return 1;
    };
    let manifest = match load_manifest(&root) {
        Ok(m) => m,
        Err(errors) => {
            for e in &errors {
                eprintln!("{}", e);
            }
            eprintln!(
                "aien agents {sub}: {} error(s) in {}",
                errors.len(),
                root.join(MANIFEST).display()
            );
            return 1;
        }
    };
    let plan = match build_plan(&manifest, &agents, opts.scope, homes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("aien agents {sub}: {e}");
            return 1;
        }
    };
    let names: Vec<&str> = agents.iter().map(|a| a.as_str()).collect();
    println!(
        "{MANIFEST}: component {} (tree {}), {} skill(s), {} hook(s), {} allowed attachment(s); agents: {}",
        manifest.component,
        manifest.tree,
        manifest.skills.len(),
        manifest.hooks.len(),
        manifest.allow.len(),
        names.join(", ")
    );
    let changes = plan.changes();
    let print_changes = |label: &str| {
        for (agent, op, status) in &changes {
            println!(
                "{label}{:<9} {:<11} {}",
                status.as_str(),
                agent.as_str(),
                display_path(&root, op.path())
            );
        }
    };
    match sub {
        "check" => {
            if changes.is_empty() {
                println!("check: up to date ({} file(s) compared)", plan.steps.len());
                0
            } else {
                print_changes("stale: ");
                println!(
                    "check: STALE, {} of {} step(s) differ; run `aien agents sync`",
                    changes.len(),
                    plan.steps.len()
                );
                1
            }
        }
        _ if opts.dry_run => {
            print_changes("would ");
            println!(
                "dry run: {} change(s), {} unchanged; nothing written",
                changes.len(),
                plan.steps.len() - changes.len()
            );
            0
        }
        _ => {
            print_changes("");
            match plan.apply() {
                Ok(n) => {
                    println!(
                        "sync: {n} change(s) written, {} unchanged",
                        plan.steps.len() - n
                    );
                    0
                }
                Err(e) => {
                    eprintln!("aien agents sync: {e}");
                    1
                }
            }
        }
    }
}

pub fn handle_agents_command(args: &[String]) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let code = run(args, &cwd, &Homes::from_env());
    if code != 0 {
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const HOOK: &str = "#!/usr/bin/env bash\n# no-admin-merge.sh: pre-command hook. Blocks `gh pr merge ... --admin`.\n#\n# Canonical copy: aien-architecture.\nexit 0\n";

    fn write(root: &Path, rel: &str, text: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, text).unwrap();
    }

    fn write_exec(root: &Path, rel: &str, text: &str) {
        write(root, rel, text);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root.join(rel), fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    const MANIFEST_OK: &str = r#"schema = 1
[component]
name = "demo"
tree = "core"

[[skills]]
path = "skills/demo-basics"

[[skills]]
path = "skills/unslop"

[[hooks]]
event = "pre-command"
command = "tools/agent-hooks/no-admin-merge.sh"

[[hooks]]
event = "pre-edit"
command = "tools/agent-hooks/protected-paths.sh"

[attachments]
allow = []
"#;

    fn repo() -> (TempDir, TempDir) {
        let repo = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let r = repo.path();
        fs::create_dir(r.join(".git")).unwrap();
        write(
            r,
            "skills/demo-basics/SKILL.md",
            "---\nname: demo-basics\ndescription: How to build demo.\n---\n\nBody.\n",
        );
        write_exec(
            r,
            "skills/demo-basics/scripts/run.sh",
            "#!/bin/sh\necho hi\n",
        );
        write(r, "skills/demo-basics/.crumb", "crumb\n");
        write(
            r,
            "skills/unslop/SKILL.md",
            "---\nname: unslop\ndescription: \"Voice rules.\"\n---\nBody.\n",
        );
        write_exec(r, "tools/agent-hooks/no-admin-merge.sh", HOOK);
        write_exec(
            r,
            "tools/agent-hooks/protected-paths.sh",
            "#!/usr/bin/env bash\n# protected-paths.sh: pre-edit hook. Blocks edits to .crumb and tests/gates/ unless\n# AIEN_PIPELINE_EDIT=1 is set. More text here.\nexit 0\n",
        );
        write(r, MANIFEST, MANIFEST_OK);
        write(r, "AGENTS.md", "# Agents\n\nHand-written rules.\n");
        (repo, home)
    }

    fn homes(h: &Path) -> Homes {
        Homes {
            claude: h.join(".claude"),
            hermes_skills: h.join(".hermes/skills"),
        }
    }

    fn sync(root: &Path, agents: &[Agent], scope: Scope, h: &Homes) -> usize {
        let m = load_manifest(root).unwrap();
        build_plan(&m, agents, scope, h).unwrap().apply().unwrap()
    }

    fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut files = Vec::new();
        walk_files(dir, Path::new(""), &mut files).unwrap();
        files
            .into_iter()
            .map(|f| {
                let b = fs::read(dir.join(&f)).unwrap();
                (f, b)
            })
            .collect()
    }

    #[test]
    fn parses_valid_manifest() {
        let (repo, _h) = repo();
        let m = load_manifest(repo.path()).unwrap();
        assert_eq!(m.component, "demo");
        assert_eq!(m.tree, "core");
        let names: Vec<_> = m.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["demo-basics", "unslop"]);
        assert_eq!(m.skills[1].description, "Voice rules.");
        assert_eq!(m.hooks.len(), 2);
        assert_eq!(m.hooks[0].event, Event::PreCommand);
        assert_eq!(m.hooks[0].rule, "Blocks `gh pr merge ... --admin`.");
        assert_eq!(
            m.hooks[1].rule,
            "Blocks edits to .crumb and tests/gates/ unless AIEN_PIPELINE_EDIT=1 is set."
        );
        assert!(m.allow.is_empty());
    }

    #[test]
    fn validation_errors_carry_line_numbers() {
        let (repo, _h) = repo();
        let text = r#"schema = 1
[component]
name = "demo"
tree = "core"
[[skills]]
path = "skills/missing"
[[hooks]]
event = "pre-commit"
command = "tools/agent-hooks/no-admin-merge.sh"
[[hooks]]
event = "pre-edit"
command = "tools/agent-hooks/nope.sh"
"#;
        let errs = parse_manifest(repo.path(), text).unwrap_err();
        let got: Vec<(usize, bool)> = errs
            .iter()
            .map(|e| {
                (
                    e.line,
                    e.message.contains("does not exist")
                        || e.message.contains("unknown hook event"),
                )
            })
            .collect();
        assert_eq!(got, [(6, true), (8, true), (12, true)], "{errs:?}");
        assert!(errs[0].message.contains("skills/missing"));
        assert!(errs[1].message.contains("pre-commit"));
        assert!(errs[2].message.contains("nope.sh"));
        assert_eq!(
            errs[0].to_string(),
            format!("{MANIFEST}:6: skill directory skills/missing does not exist")
        );
    }

    #[test]
    fn bad_toml_and_unknown_keys_are_errors_with_lines() {
        let (repo, _h) = repo();
        let errs =
            parse_manifest(repo.path(), "schema = 1\n[component]\nname = \"demo\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].line, 3, "{errs:?}");

        let errs = parse_manifest(
            repo.path(),
            "schema = 1\n[component]\nname = \"demo\"\ntree = \"core\"\ncolour = \"red\"\n",
        )
        .unwrap_err();
        assert_eq!(errs[0].line, 5, "{errs:?}");
        assert!(errs[0].message.contains("colour"), "{errs:?}");

        let errs = parse_manifest(
            repo.path(),
            "schema = 2\n[component]\nname = \"demo\"\ntree = \"leaf\"\n",
        )
        .unwrap_err();
        let lines: Vec<_> = errs.iter().map(|e| e.line).collect();
        assert_eq!(lines, [1, 4], "{errs:?}");
    }

    #[test]
    fn attachments_only_on_core_and_paths_must_be_relative() {
        let (repo, _h) = repo();
        let errs = parse_manifest(
            repo.path(),
            "schema = 1\n[component]\nname = \"demo\"\ntree = \"attachment\"\n[[skills]]\npath = \"../x\"\n[attachments]\nallow = [\"omega\"]\n",
        )
        .unwrap_err();
        let lines: Vec<_> = errs.iter().map(|e| e.line).collect();
        assert_eq!(lines, [6, 7], "{errs:?}");
    }

    #[test]
    fn non_executable_hook_is_an_error() {
        let (repo, _h) = repo();
        write(repo.path(), "tools/agent-hooks/no-admin-merge.sh", HOOK);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                repo.path().join("tools/agent-hooks/no-admin-merge.sh"),
                fs::Permissions::from_mode(0o644),
            )
            .unwrap();
            let errs = load_manifest(repo.path()).unwrap_err();
            assert!(errs[0].message.contains("not executable"), "{errs:?}");
        }
    }

    #[test]
    fn sync_is_idempotent_and_check_passes_after() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        let agents = agents_for("all", Scope::Project).unwrap();
        assert_eq!(agents, [Agent::ClaudeCode, Agent::Codex]);
        let m = load_manifest(r).unwrap();
        assert!(!build_plan(&m, &agents, Scope::Project, &h)
            .unwrap()
            .changes()
            .is_empty());

        assert!(sync(r, &agents, Scope::Project, &h) > 0);
        let first = snapshot(r);
        assert_eq!(sync(r, &agents, Scope::Project, &h), 0);
        assert_eq!(first, snapshot(r));
        assert!(build_plan(&m, &agents, Scope::Project, &h)
            .unwrap()
            .changes()
            .is_empty());
        assert_eq!(run(&["check".into()], r, &h), 0);

        assert!(r.join(".claude/skills/demo-basics/SKILL.md").is_file());
        assert!(r.join(".claude/skills/unslop/.aien-agents-sync").is_file());
        assert!(
            !r.join(".claude/skills/demo-basics/.crumb").exists(),
            "crumbs are never copied"
        );
        assert!(is_executable(
            &r.join(".claude/skills/demo-basics/scripts/run.sh")
        ));
        let s: Value =
            serde_json::from_slice(&fs::read(r.join(".claude/settings.json")).unwrap()).unwrap();
        let pre = s["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 2);
        assert_eq!(pre[0]["matcher"], "Bash");
        assert_eq!(pre[0][MARKER_KEY], "demo");
        assert_eq!(
            pre[0]["hooks"][0]["command"],
            "\"$CLAUDE_PROJECT_DIR\"/tools/agent-hooks/no-admin-merge.sh"
        );
        assert_eq!(pre[1]["matcher"], EDIT_MATCHER);

        // A manifest change makes check fail until the next sync.
        write(
            r,
            "skills/unslop/SKILL.md",
            "---\nname: unslop\ndescription: Changed.\n---\n",
        );
        assert_eq!(run(&["check".into()], r, &h), 1);
        assert_eq!(run(&["sync".into(), "--dry-run".into()], r, &h), 0);
        assert_eq!(run(&["check".into()], r, &h), 1, "dry run must not write");
        assert_eq!(run(&["sync".into()], r, &h), 0);
        assert_eq!(run(&["check".into()], r, &h), 0);
    }

    #[test]
    fn merge_keeps_foreign_hook_entries_and_settings() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        let foreign = json!({
            "model": "opus",
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "/usr/local/bin/guard.sh"}]},
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "old"}], MARKER_KEY: "demo"},
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "theirs"}], MARKER_KEY: "omega"}
                ],
                "Stop": [{"hooks": [{"type": "command", "command": "notify.sh"}]}]
            }
        });
        write(
            r,
            ".claude/settings.json",
            &serde_json::to_string_pretty(&foreign).unwrap(),
        );
        sync(r, &[Agent::ClaudeCode], Scope::Project, &h);
        sync(r, &[Agent::ClaudeCode], Scope::Project, &h);
        let s: Value =
            serde_json::from_slice(&fs::read(r.join(".claude/settings.json")).unwrap()).unwrap();
        assert_eq!(s["model"], "opus");
        assert_eq!(s["hooks"]["Stop"][0]["hooks"][0]["command"], "notify.sh");
        let cmds: Vec<String> = s["hooks"]["PreToolUse"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["hooks"][0]["command"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(cmds.len(), 4, "{cmds:?}");
        assert_eq!(cmds[0], "/usr/local/bin/guard.sh");
        assert_eq!(cmds[1], "theirs");
        assert!(!cmds.contains(&"old".to_string()));
        assert!(cmds[2].ends_with("no-admin-merge.sh"));
        assert!(cmds[3].ends_with("protected-paths.sh"));
    }

    #[test]
    fn removed_hooks_and_skills_are_cleaned_up_but_foreign_skills_stay() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        write(
            r,
            ".claude/skills/handmade/SKILL.md",
            "---\nname: handmade\n---\n",
        );
        sync(r, &[Agent::ClaudeCode], Scope::Project, &h);
        write(r, MANIFEST, "schema = 1\n[component]\nname = \"demo\"\ntree = \"core\"\n[[skills]]\npath = \"skills/unslop\"\n");
        sync(r, &[Agent::ClaudeCode], Scope::Project, &h);
        assert!(!r.join(".claude/skills/demo-basics").exists());
        assert!(r.join(".claude/skills/unslop/SKILL.md").is_file());
        assert!(r.join(".claude/skills/handmade/SKILL.md").is_file());
        let s: Value =
            serde_json::from_slice(&fs::read(r.join(".claude/settings.json")).unwrap()).unwrap();
        assert!(s.get("hooks").is_none(), "{s}");
    }

    #[test]
    fn refuses_to_overwrite_a_skill_it_did_not_write() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        write(r, ".claude/skills/unslop/SKILL.md", "mine\n");
        let m = load_manifest(r).unwrap();
        let e = build_plan(&m, &[Agent::ClaudeCode], Scope::Project, &h).unwrap_err();
        assert!(e.contains("refusing to overwrite"), "{e}");
        assert_eq!(
            fs::read_to_string(r.join(".claude/skills/unslop/SKILL.md")).unwrap(),
            "mine\n"
        );
    }

    #[test]
    fn agents_md_block_is_replaced_not_duplicated() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        write(r, "AGENTS.md", "# Agents\n\nHand-written rules.\n");
        sync(r, &[Agent::Codex], Scope::Project, &h);
        let once = fs::read_to_string(r.join("AGENTS.md")).unwrap();
        assert!(once.starts_with("# Agents\n\nHand-written rules.\n\n<!-- aien-agents:begin -->\n"));
        assert!(once.contains("- [demo-basics](skills/demo-basics/SKILL.md): How to build demo.\n"));
        assert!(once.contains("- pre-command [`no-admin-merge.sh`](tools/agent-hooks/no-admin-merge.sh): Blocks `gh pr merge ... --admin`.\n"));

        // Text after the block survives, and a changed manifest replaces it in place.
        fs::write(r.join("AGENTS.md"), format!("{once}\n## Footer\n")).unwrap();
        write(
            r,
            "skills/unslop/SKILL.md",
            "---\nname: unslop\ndescription: New words.\n---\n",
        );
        sync(r, &[Agent::Codex], Scope::Project, &h);
        let twice = fs::read_to_string(r.join("AGENTS.md")).unwrap();
        assert_eq!(twice.matches(BLOCK_BEGIN).count(), 1);
        assert_eq!(twice.matches(BLOCK_END).count(), 1);
        assert!(twice.contains("New words."));
        assert!(!twice.contains("Voice rules."));
        assert!(twice.ends_with("\n## Footer\n"));
        assert_eq!(sync(r, &[Agent::Codex], Scope::Project, &h), 0);
    }

    #[test]
    fn malformed_agents_md_markers_are_refused() {
        let doc = format!("{BLOCK_BEGIN}\nx\n{BLOCK_BEGIN}\n{BLOCK_END}\n");
        assert!(splice_block(&doc, "b\n").is_err());
        let doc = format!("{BLOCK_END}\n{BLOCK_BEGIN}\n");
        assert!(splice_block(&doc, "b\n").is_err());
        assert_eq!(splice_block("", "b\n").unwrap(), "b\n");
    }

    #[test]
    fn hermes_and_global_claude_scope() {
        let (repo, home) = repo();
        let h = homes(home.path());
        let r = repo.path();
        assert!(agents_for("hermes", Scope::Project).is_err());
        assert!(agents_for("codex", Scope::Global).is_err());
        let agents = agents_for("all", Scope::Global).unwrap();
        assert_eq!(agents, [Agent::ClaudeCode, Agent::Hermes]);
        sync(r, &agents, Scope::Global, &h);
        assert!(home
            .path()
            .join(".hermes/skills/aien/demo-unslop/SKILL.md")
            .is_file());
        assert!(home
            .path()
            .join(".hermes/skills/aien/demo-demo-basics/SKILL.md")
            .is_file());
        assert!(home.path().join(".claude/skills/unslop/SKILL.md").is_file());
        let s: Value =
            serde_json::from_slice(&fs::read(home.path().join(".claude/settings.json")).unwrap())
                .unwrap();
        let cmd = s["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(
            cmd.starts_with('\'') && cmd.ends_with("tools/agent-hooks/no-admin-merge.sh'"),
            "{cmd}"
        );
        assert!(
            !r.join(".claude").exists(),
            "global scope must not write into the repo"
        );
        assert_eq!(sync(r, &agents, Scope::Global, &h), 0);
    }

    #[test]
    fn find_root_stops_at_git_root() {
        let (repo, _h) = repo();
        let sub = repo.path().join("crates/x");
        fs::create_dir_all(&sub).unwrap();
        assert_eq!(find_root(&sub).unwrap(), repo.path());
        fs::remove_file(repo.path().join(MANIFEST)).unwrap();
        assert!(find_root(&sub).is_none());
    }
}
