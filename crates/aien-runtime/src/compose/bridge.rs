//! The compose bridge implementation: proposal parsing, destination and input
//! classification, prompts, bounded retries, the composition home and
//! `ComposeBridge`. Callers use the interface in `crate::compose`.

use crate::control::{ControlCommand, ControlResponse};
use std::sync::Arc;

use crate::control::{
    ComposeNoteReport, ComposeRecallReport, ComposeRecordView, ComposeRecoverReport,
    ComposeTaskReport, InputOmitted, InputRef, ProposalAttempt,
};
use crate::cortex_mark::{self, Mark, MarkRefusal, Plan};
use aien_omega_compose::{hex, note_bytes, Compose, ComposeError, NoteKind, RootKind};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One model reply: the text and how many tokens were generated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub tokens: usize,
    /// How generation stopped: "eos" | "max_tokens" | "aborted" | "preempted"
    /// (ACCEPTANCE-v5 Q3); None when unknown.
    pub finish_reason: Option<String>,
    /// NEXT-PHASE-1 v6 R1: the generated token ids in order (None when the
    /// proposer does not expose them). Recorded only; nothing reads them.
    pub token_ids: Option<Vec<u32>>,
    /// NEXT-PHASE-1 v6 R1: how many prompt token ids were submitted.
    pub prompt_tokens: Option<usize>,
    /// NEXT-PHASE-1 v6 R1: `token_ids_sha256` of the submitted prompt ids.
    pub prompt_ids_sha256: Option<String>,
    /// How the backend chose the generated tokens (sc#294), as the scheduler
    /// reported it when the sequence finished. None when the proposer or the
    /// backend does not observe it: an absent claim, not greedy.
    pub decoding: Option<aien_abi_core::DecodeObservation>,
    /// The tensor backend and its op counters when the call finished (sc#337).
    /// None when the proposer or the backend does not report them: no claim.
    pub ops: Option<aien_abi_core::OpEvidence>,
}

/// sha256 (hex) of token ids, each as 4 little-endian bytes (NEXT-PHASE-1 v6
/// R1): the daemon and the CPU reference driver hash prompts the same way.
pub fn token_ids_sha256(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    hex(&h.finalize())
}

/// The "model" Skill's work: prompt text and a wall limit in, one reply
/// out. The server installs the real inference path (the one `StreamTurn`
/// uses); tests install a deterministic proposer.
pub type ComposeProposer =
    Arc<dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync>;

/// The one Skill name; its procedure digest is sha256 of this name.
const COMPOSE_MODEL_SKILL: &str = "aien.model.propose-file-change";

/// `ComposeTaskReport::proposer` of a run driven by the proposer hook
/// (crate::approved): the reply came from an approved proposal, not a model.
const APPROVED_PROPOSER_LABEL: &str = "hook:approved-proposal";

/// Composition home: `$AIEN_COMPOSE_DIR`, else `$AIEN_RUNTIME_STATE_DIR/compose`,
/// else `$XDG_STATE_HOME/aien-runtime/compose`, else
/// `$HOME/.local/state/aien-runtime/compose`. Never /tmp.
pub fn compose_dir_from_env() -> Result<PathBuf, String> {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    if let Some(d) = get("AIEN_COMPOSE_DIR") {
        return Ok(PathBuf::from(d.trim()));
    }
    if let Some(d) = get("AIEN_RUNTIME_STATE_DIR") {
        return Ok(PathBuf::from(d.trim()).join("compose"));
    }
    if let Some(d) = get("XDG_STATE_HOME") {
        return Ok(PathBuf::from(d.trim()).join("aien-runtime/compose"));
    }
    if let Some(h) = get("HOME") {
        return Ok(PathBuf::from(h.trim()).join(".local/state/aien-runtime/compose"));
    }
    Err(
        "no AIEN_COMPOSE_DIR, AIEN_RUNTIME_STATE_DIR, XDG_STATE_HOME or HOME for the compose home"
            .into(),
    )
}

/// The provisioned machine root: 32 random bytes kept beside the home
/// (`<home>.machine-root`, mode 0600), created once. STOPGAP until AIENOS
/// machine provisioning owns the AienMachineId root.
fn machine_root(dir: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    let mut name = dir.as_os_str().to_owned();
    name.push(".machine-root");
    let path = PathBuf::from(name);
    if let Ok(b) = std::fs::read(&path) {
        if b.len() == 32 {
            return Ok(b);
        }
        return Err(format!(
            "{} holds {} bytes, expected 32",
            path.display(),
            b.len()
        ));
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create {}: {e}", p.display()))?;
    }
    let mut root = vec![0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut root))
        .map_err(|e| format!("read /dev/urandom: {e}"))?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    std::io::Write::write_all(&mut f, &root)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    f.sync_all()
        .map_err(|e| format!("sync {}: {e}", path.display()))?;
    Ok(root)
}

fn proposal_handle(text: &str) -> u64 {
    let d = Sha256::digest(text.as_bytes());
    let h = u64::from_le_bytes(d[..8].try_into().expect("8 bytes"));
    if h == 0 {
        1
    } else {
        h
    }
}

/// ALLEN identity gate (aien-allen, ADR 0035), run once per compose-home open.
/// Not engaged (AIEN_ALLEN_SUBJECT unset): one log line, nothing else changes.
/// Engaged: any refusal is FATAL (message, nonzero exit), never a fallback.
/// Returns the resolved identity when engaged, so the compose home can keep it
/// (persona profile, arch#159); `None` when not engaged.
fn allen_gate(
    dir: &Path,
    compose: &mut Compose,
    machine_id: &[u8; 32],
) -> Option<aien_allen::Resolved> {
    use aien_allen::{Context, Gate};
    static NOT_ENGAGED_ONCE: std::sync::Once = std::sync::Once::new();
    let lineage = compose.record(1).ok().map(|r| r.digest);
    let ctx = Context {
        machine_id: *machine_id,
        lineage,
    };
    // The operator adoption variable is honoured once per process: after this
    // process wrote the pin, later opens (recover, re-open) see it as spent.
    static ADOPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let get = |k: &str| {
        if k == aien_allen::ENV_ADOPT && ADOPTED.load(std::sync::atomic::Ordering::SeqCst) {
            return None;
        }
        std::env::var(k).ok()
    };
    match aien_allen::gate(dir, &ctx, &get) {
        Gate::NotEngaged => {
            NOT_ENGAGED_ONCE.call_once(|| println!("{}", aien_allen::NOT_ENGAGED_LINE));
            None
        }
        Gate::Engaged(r) => {
            if r.adopted {
                ADOPTED.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let adopted = if r.adopted {
                " (ADOPTED: pin written once, operator-approved)"
            } else {
                ""
            };
            println!(
                "ALLEN: engaged agent={} head_sequence={} chain_verified={}{adopted}",
                aien_allen::hex(&r.agent),
                r.head_seq,
                r.chain_verified
            );
            Some(r)
        }
        Gate::Refused(why) => {
            let msg = format!("FATAL ALLEN refused: {why}");
            tracing::error!("{msg}");
            eprintln!("{msg}");
            std::process::exit(aien_allen::EXIT_REFUSED);
        }
    }
}

/// A named refusal for an rxc_host open error: the operator learns what is
/// wrong and that `RecoverComposeHome` is the remedy (never a silent repair).
fn refusal(dir: &Path, e: &ComposeError) -> String {
    let why = match e {
        ComposeError::Code { code: -3, .. } => {
            "Cortex journal tail torn (E_TORN, CX_ERR_TORN); nothing was cut. Run RecoverComposeHome (aien compose recover)"
                .to_string()
        }
        ComposeError::Code { code: -11, detail } => format!(
            "Cortex journal behind its J-Space anchor (E_REPLAY, RX_ERR_REPLAY {detail}). Run RecoverComposeHome (aien compose recover)"
        ),
        other => other.to_string(),
    };
    format!("compose home {} refused: {why}", dir.display())
}

/// A refusal by the Cortex record mark (ACCEPTANCE-v3 2.2), in the same form.
fn mark_refusal(dir: &Path, r: &MarkRefusal) -> String {
    format!("compose home {} refused: {r}", dir.display())
}

/// The record chain (`cortex_mark::chain_step`) from `start` over the digests
/// of records #`lo`..=#`hi`, as the open home reads them.
fn record_chain(
    compose: &mut Compose,
    start: [u8; 32],
    lo: u64,
    hi: u64,
) -> Result<[u8; 32], String> {
    let mut c = start;
    for id in lo..=hi {
        let d = compose
            .record(id)
            .map_err(|e| format!("record {id}: {e}"))?
            .digest;
        cortex_mark::chain_step(&mut c, &d);
    }
    Ok(c)
}

/// One proposed file change: a path relative to the workspace and the
/// complete new content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileProposal {
    pub path: String,
    pub content: String,
}

fn bare(s: &str) -> &str {
    s.trim()
        .trim_matches(|c| matches!(c, '`' | '"' | '\'' | '*'))
        .trim()
}

/// A path the workspace may hold: relative, at most 255 bytes, letters,
/// digits and `._-/`, no empty, `.` or `..` component, not absolute, not `~`.
fn check_relative_path(p: &str) -> Result<(), String> {
    if p.is_empty() {
        return Err("empty path on the filename line".into());
    }
    if p.starts_with('/') || p.starts_with('~') || p.split('/').any(|c| c == "..") {
        return Err(format!("path {p:?} is outside the workspace"));
    }
    let ok = p.len() <= 255
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
        && p.split('/').all(|c| !c.is_empty() && c != ".");
    if ok {
        Ok(())
    } else {
        Err(format!("path {p:?} is not a plain relative path"))
    }
}

/// A code-fence line: its backtick run length and whether an info string
/// (such as `bash`) follows the run. Not a fence if fewer than three
/// backticks start the line or the info string itself holds a backtick.
fn fence_line(l: &str) -> Option<(usize, bool)> {
    let t = l.trim_start();
    let n = t.bytes().take_while(|&b| b == b'`').count();
    if n < 3 {
        return None;
    }
    let info = t[n..].trim();
    if info.contains('`') {
        return None;
    }
    Some((n, !info.is_empty()))
}

/// Index (into `body`, which starts after the opening fence of backtick
/// length `open_len`) of the line that closes the outer fence, or None.
/// CommonMark: a closing fence is bare (no info string) and at least as long
/// as the opener, so shorter inner fences are plain content. Inner fences of
/// the same length that carry an info string (` ```bash `) open a nested block
/// that its own bare fence must close first. If that scan finds no balanced
/// close (unbalanced reply), the LAST bare fence of sufficient length is
/// taken, so content is never cut at an inner fence. A bare fence at depth 0
/// that still has fences after it is a bare inner opener, so the outer fence
/// is closed by the last fence of the reply. Limit: prose that itself holds a
/// fence AFTER the real close is read as part of the file. With no bare fence at
/// all there is no close: the caller keeps the whole body.
fn outer_fence_close(body: &[&str], open_len: usize) -> Option<usize> {
    let fences: Vec<(usize, bool)> = body
        .iter()
        .enumerate()
        .filter_map(|(i, l)| fence_line(l).filter(|f| f.0 >= open_len).map(|f| (i, f.1)))
        .collect();
    let mut depth = 0usize;
    let mut last_bare = None;
    for (k, &(i, info)) in fences.iter().enumerate() {
        if info {
            depth += 1;
        } else if depth > 0 {
            depth -= 1;
            last_bare = Some(i);
        } else if k + 1 == fences.len() {
            return Some(i);
        } else {
            // A bare fence with more fences after it opens a bare inner block;
            // only the final fence can then be the outer close.
            depth += 1;
        }
    }
    last_bare
}

/// The model's answer in the fixed proposal template: the first nonempty
/// line is `filename: <relative path>` (case-insensitive key; quotes,
/// backticks and asterisks around it or the path are ignored, and so are leading
/// `./` segments of the path, sc#402), everything
/// after it is the complete content (blank lines around it and one
/// surrounding code fence are dropped; it ends with one newline).
/// Err names why the reply is refused. Deterministic: the Skill, the AEGIS
/// contract, the authorization and the write all use this one reading.
pub fn check_file_proposal(text: &str) -> Result<FileProposal, String> {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|l| !l.trim().is_empty())
        .ok_or("empty reply")?;
    let first = bare(lines[at]);
    const KEY: &str = "filename:";
    if first.len() < KEY.len() || !first[..KEY.len()].eq_ignore_ascii_case(KEY) {
        return Err("no filename line: the first line must be 'filename: <relative path>'".into());
    }
    // sc#402: `./report.md` is `report.md`; the strict check then runs as before.
    let path = crate::destination::strip_leading_curdir(bare(&first[KEY.len()..])).to_string();
    check_relative_path(&path)?;
    let mut body: Vec<&str> = lines[at + 1..].to_vec();
    while body.first().is_some_and(|l| l.trim().is_empty()) {
        body.remove(0);
    }
    if body
        .first()
        .is_some_and(|l| l.trim_start().starts_with("```"))
    {
        let open_len = body[0]
            .trim_start()
            .bytes()
            .take_while(|&b| b == b'`')
            .count();
        body.remove(0);
        if let Some(end) = outer_fence_close(&body, open_len) {
            body.truncate(end);
        }
    }
    while body.last().is_some_and(|l| l.trim().is_empty()) {
        body.pop();
    }
    if body.iter().all(|l| l.trim().is_empty()) {
        return Err("empty content after the filename line".into());
    }
    let mut content = body.join("\n");
    content.push('\n');
    Ok(FileProposal { path, content })
}

/// `check_file_proposal` without the reason.
pub fn parse_file_proposal(text: &str) -> Option<FileProposal> {
    check_file_proposal(text).ok()
}

/// At most this many proposals per task (ACCEPTANCE-v2 Section 3b).
const COMPOSE_MAX_ATTEMPTS: u32 = 3;
/// Wall budget for all attempts of one task: rx_compose_run's 30 s
/// quiescence wait (not raised) less a 1 s margin (ACCEPTANCE-v3 Section 3b).
const COMPOSE_SKILL_BUDGET: std::time::Duration = std::time::Duration::from_secs(29);
/// Wall budget of a full-document task (one that creates a new file): 120 s.
const COMPOSE_DOC_BUDGET: std::time::Duration = std::time::Duration::from_secs(120);
/// Env: wall budget of a small edit, in ms. Unset = `COMPOSE_SKILL_BUDGET` (29 s).
const COMPOSE_EDIT_BUDGET_ENV: &str = "AIEN_COMPOSE_EDIT_BUDGET_MS";
/// Env: wall budget of a full-document task, in ms. Unset = `COMPOSE_DOC_BUDGET` (120 s).
const COMPOSE_DOC_BUDGET_ENV: &str = "AIEN_COMPOSE_DOC_BUDGET_MS";
/// Env: output token cap of a full-document task. Unset = `COMPOSE_DOC_MAX_TOKENS_DEFAULT`.
/// (Small edits keep `AIEN_COMPOSE_MAX_TOKENS`, read by the daemon, default 48.)
const COMPOSE_DOC_MAX_TOKENS_ENV: &str = "AIEN_COMPOSE_DOC_MAX_TOKENS";
/// The one-budget setting this replaced. Setting it is refused, never ignored.
const COMPOSE_BUDGET_ENV_RETIRED: &str = "AIEN_COMPOSE_BUDGET_MS";
/// Default output token cap of a full-document task.
const COMPOSE_DOC_MAX_TOKENS_DEFAULT: usize = 1024;
/// Accepted range of `AIEN_COMPOSE_DOC_MAX_TOKENS`.
const COMPOSE_DOC_MAX_TOKENS_RANGE: std::ops::RangeInclusive<usize> = 16..=4096;
/// Accepted range of both budget settings: omega waits budget + 1 000 ms and accepts at most 600 000 ms.
const COMPOSE_BUDGET_MS_RANGE: std::ops::RangeInclusive<u64> = 1_000..=599_000;
/// Margin between the skill budget and omega's settle wait.
const COMPOSE_WAIT_MARGIN: std::time::Duration = std::time::Duration::from_millis(1_000);

/// What a compose task is, for limits: a small edit of an existing file, or
/// the creation of a whole new document. Decided by `classify_target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    Edit,
    Document,
}

/// What the goal's named target is, decided by whether the path EXISTS (not by
/// whether an edit block can be built from it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetClass {
    /// No existing file is named: the task creates a new document.
    New,
    /// An existing small UTF-8 file inside the workspace: path and content.
    Edit(String, String),
    /// An existing path that no edit block can be built for (too large,
    /// non-UTF-8, unreadable, or resolving outside the workspace). Never a
    /// new document, never overwritten.
    Refused(String),
}

impl TargetClass {
    pub fn kind(&self) -> ProposalKind {
        match self {
            Self::Edit(..) => ProposalKind::Edit,
            _ => ProposalKind::Document,
        }
    }
}

/// The two wall budgets in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComposeBudgets {
    pub edit: std::time::Duration,
    pub doc: std::time::Duration,
}

impl Default for ComposeBudgets {
    fn default() -> Self {
        Self {
            edit: COMPOSE_SKILL_BUDGET,
            doc: COMPOSE_DOC_BUDGET,
        }
    }
}

impl ComposeBudgets {
    pub(crate) fn for_kind(&self, kind: ProposalKind) -> std::time::Duration {
        match kind {
            ProposalKind::Edit => self.edit,
            ProposalKind::Document => self.doc,
        }
    }
}

/// Parse one budget setting named `name`. `None` (unset) gives `default`; an
/// empty, non-numeric or out-of-range value is refused, never replaced by the default.
fn parse_compose_budget(
    name: &str,
    default: std::time::Duration,
    v: Option<&str>,
) -> Result<std::time::Duration, String> {
    let Some(raw) = v else {
        return Ok(default);
    };
    let ms: u64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("{name}={raw:?} is not a whole number of milliseconds"))?;
    if !COMPOSE_BUDGET_MS_RANGE.contains(&ms) {
        return Err(format!(
            "{name}={ms} is outside {}..={} ms",
            COMPOSE_BUDGET_MS_RANGE.start(),
            COMPOSE_BUDGET_MS_RANGE.end()
        ));
    }
    Ok(std::time::Duration::from_millis(ms))
}

/// Parse `AIEN_COMPOSE_DOC_MAX_TOKENS` (same refusal rule as the budgets).
fn parse_compose_doc_max_tokens(v: Option<&str>) -> Result<usize, String> {
    let Some(raw) = v else {
        return Ok(COMPOSE_DOC_MAX_TOKENS_DEFAULT);
    };
    let n: usize = raw.trim().parse().map_err(|_| {
        format!("{COMPOSE_DOC_MAX_TOKENS_ENV}={raw:?} is not a whole number of tokens")
    })?;
    if !COMPOSE_DOC_MAX_TOKENS_RANGE.contains(&n) {
        return Err(format!(
            "{COMPOSE_DOC_MAX_TOKENS_ENV}={n} is outside {}..={} tokens",
            COMPOSE_DOC_MAX_TOKENS_RANGE.start(),
            COMPOSE_DOC_MAX_TOKENS_RANGE.end()
        ));
    }
    Ok(n)
}

fn env_opt(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name} unreadable: {e}")),
    }
}

/// Env switch for the desk MAC on `ComposeAuthorize` (#297, #328). Required
/// by default: unset or `1` = required. `0` is the dev-only opt-out, accepted
/// only in a dev run (`aien_inference_abi::strict::dev_fallback_active`); a
/// strict run, which is what release qualification runs, refuses it. Anything
/// else stops the daemon (a typo must not leave the requirement silently off).
pub const AUTHORIZE_DESK_ENV: &str = "AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK";

/// The rule behind [`authorize_requires_desk_from_env`], for a switch value
/// and whether this is a dev run.
pub fn authorize_desk_setting(value: Option<&str>, dev: bool) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") if dev => Ok(false),
        Some("0") => Err(format!(
            "{AUTHORIZE_DESK_ENV}=0 is a dev-only opt-out and is prohibited here: this is a strict run \
             (no {}=1), as release qualification is. Create the desk key with \
             `aien compose desk-key --create 1` and unset {AUTHORIZE_DESK_ENV}",
            aien_inference_abi::strict::DEV_FALLBACK_ENV
        )),
        Some(o) => Err(format!(
            "{AUTHORIZE_DESK_ENV} must be 1 (required, the default) or 0 (dev-only opt-out), got {o:?}"
        )),
    }
}

pub fn authorize_requires_desk_from_env() -> Result<bool, String> {
    authorize_desk_setting(
        env_opt(AUTHORIZE_DESK_ENV)?.as_deref(),
        aien_inference_abi::strict::dev_fallback_active(),
    )
}

/// Startup check (#328): a bridge that requires the desk must find a loadable
/// desk key in its compose home, or the daemon refuses to start. Never creates
/// a key: minting one is the operator's `aien compose desk-key --create`.
pub(crate) fn check_desk_key_at_start(compose_dir: &Path) -> Result<(), String> {
    let path = crate::approved_auth::desk_key_path(compose_dir);
    crate::approved_auth::DeskKey::load(&path).map(|_| ()).map_err(|e| {
        format!(
            "the approval desk is required ({AUTHORIZE_DESK_ENV} unset or 1) but its key is not usable: {e}. \
             Create it with `aien compose desk-key --create 1` (AIEN_COMPOSE_DIR={}); the daemon refuses to \
             start without it",
            compose_dir.display()
        )
    })
}

/// The daemon's startup rule from the environment (#328): the setting,
/// after the desk key check when the desk is required.
pub fn authorize_desk_at_start_from_env() -> Result<bool, String> {
    let required = authorize_requires_desk_from_env()?;
    if required {
        if let Ok(dir) = compose_dir_from_env() {
            check_desk_key_at_start(&dir)?;
        }
    }
    Ok(required)
}

/// The startup line that states the authorize rule this daemon enforces.
pub(crate) fn authorize_mac_line(required: bool) -> &'static str {
    if required {
        "Authorize MAC: on (aien compose authorize requires the approval desk key MAC)"
    } else {
        "Authorize MAC: OFF, DEV OPT-OUT (AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=0 in a dev run): \
         authorize is authenticated by the OS user only; not valid for release qualification"
    }
}

/// Both budgets from the environment. Refuses the retired single-budget
/// setting (a stale `AIEN_COMPOSE_BUDGET_MS` must not be silently ignored).
pub fn compose_budgets_from_env() -> Result<ComposeBudgets, String> {
    if std::env::var_os(COMPOSE_BUDGET_ENV_RETIRED).is_some() {
        return Err(format!(
            "{COMPOSE_BUDGET_ENV_RETIRED} is retired; set {COMPOSE_EDIT_BUDGET_ENV} and {COMPOSE_DOC_BUDGET_ENV}"
        ));
    }
    Ok(ComposeBudgets {
        edit: parse_compose_budget(
            COMPOSE_EDIT_BUDGET_ENV,
            COMPOSE_SKILL_BUDGET,
            env_opt(COMPOSE_EDIT_BUDGET_ENV)?.as_deref(),
        )?,
        doc: parse_compose_budget(
            COMPOSE_DOC_BUDGET_ENV,
            COMPOSE_DOC_BUDGET,
            env_opt(COMPOSE_DOC_BUDGET_ENV)?.as_deref(),
        )?,
    })
}

/// The document token cap from the environment.
pub fn compose_doc_max_tokens_from_env() -> Result<usize, String> {
    parse_compose_doc_max_tokens(env_opt(COMPOSE_DOC_MAX_TOKENS_ENV)?.as_deref())
}

/// Stable prefix of `wall_clock_reason`: `propose_task_checked` classes an
/// attempt as "timeout" by it, and the token-limit cut (`length_cut_refusal`) never has it.
const WALL_CLOCK_REASON_PREFIX: &str = "model proposal exceeded ";

/// Reason text of a wall-clock timeout.
pub(crate) fn wall_clock_reason(limit_ms: u128) -> String {
    format!("{WALL_CLOCK_REASON_PREFIX}{limit_ms} ms")
}

/// Time one full attempt needs, from the v3 measurement: 173-token retry
/// prompt prefill (2 839 + 896 ms) + 47 decode steps x 166.6 ms + 60 ms
/// = 11 625 ms, rounded up (ACCEPTANCE-v3 Section 3b). Attempt k > 1 starts
/// only if at least this much budget is left.
const COMPOSE_ATTEMPT_BUDGET: std::time::Duration = std::time::Duration::from_millis(12_000);
/// Same rule for a full document. DIAGNOSTIC-LONG-1 (sovereign-core branch
/// 031756/oq3-long-diag) measured replies of 313..649 tokens finishing by
/// themselves in 28.6..61.9 s (~88 ms/token plus prefill), so a retry that
/// has to write a whole document needs about the shortest measured one:
/// 28.6 s, rounded up to 30 s. With the 120 s budget, a first attempt up to the
/// longest measured (61.9 s) leaves 58 s and still allows a retry; one that ran
/// to the token cap (1024 tokens, ~90 s) leaves ~30 s and may not. Capped at
/// half the budget so a small custom doc budget keeps the retry rule meaningful.
const COMPOSE_DOC_ATTEMPT_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// The per-attempt budget for `kind` under wall budget `budget`.
fn compose_attempt_budget(kind: ProposalKind, budget: std::time::Duration) -> std::time::Duration {
    match kind {
        ProposalKind::Edit => COMPOSE_ATTEMPT_BUDGET,
        ProposalKind::Document => COMPOSE_DOC_ATTEMPT_BUDGET.min(budget / 2),
    }
}

/// Assistant-response prefix of the production RunComposeTask template: the
/// assistant turn starts with it and the model generates the path and the
/// content after it (ACCEPTANCE-v4 Section 2(1)). The reply the parser reads
/// is this prefix followed by the generated text.
///
/// NEXT-PHASE-1 v5 engine cut: no trailing space. A separate space token after
/// the colon was measured harmful; the model's own tokenization puts the space
/// in front of the path.
pub const COMPOSE_ASSISTANT_PREFIX: &str = "filename:";

/// The compose assistant-response prefix for a model's chat template. Zephyr
/// and Llama 3 keep `COMPOSE_ASSISTANT_PREFIX` byte for byte. ChatML
/// (SmolLM2-Instruct) uses the same text: the proposal template asks for a
/// `filename: <relative path>` first line, and in SmolLM2's byte-level BPE
/// `filename:` ends on the colon token, so the path keeps its own leading
/// space token as in Llama 3. `proposal_prompt` is the same for every template.
pub fn compose_assistant_prefix(template: &aien_inference_abi::ChatTemplate) -> &'static str {
    use aien_inference_abi::ChatTemplate;
    match template {
        ChatTemplate::Zephyr | ChatTemplate::Llama3 => COMPOSE_ASSISTANT_PREFIX,
        // A plain model never reaches generation (chat render is refused first).
        ChatTemplate::None => COMPOSE_ASSISTANT_PREFIX,
        ChatTemplate::ChatMl { .. }
        | ChatTemplate::ChatMlQwen3
        | ChatTemplate::ChatMlQwen3NoThink => "filename:",
    }
}

/// The logical name of the authorized workspace as the model sees it (sc#429).
///
/// A constant: the model-visible prompt never carries the machine-specific
/// absolute path (host name, user name, job folder), so equal logical inputs
/// give equal prompt bytes and token ids on every machine. The real path stays
/// in the authorization, confinement, execution and evidence layers
/// (`RunComposeTask` canonicalizes the workspace, the task id hashes it, the
/// commit record stores it).
const WORKSPACE_LABEL: &str = "workspace";

/// The fixed proposal template of the production RunComposeTask path. The
/// workspace is named by `WORKSPACE_LABEL`, never by its path (sc#429).
pub fn proposal_prompt(goal: &str) -> String {
    let workspace = WORKSPACE_LABEL;
    format!(
        "Goal: {goal}\nAuthorized workspace: {workspace}\n\
         Propose exactly one file change inside the workspace.\n\
         Answer in exactly this format and nothing else:\n\
         filename: <relative path>\n\
         <the complete new file content>"
    )
}

/// Largest existing file whose content the edit-mode block carries
/// (NEXT-PHASE-1 v6 T5); a larger file gets no block.
const COMPOSE_EDIT_MAX_BYTES: u64 = 8192;

/// Decide the task kind from the goal's DESTINATION only (issue #288): the
/// path the goal asks to create or change, found by
/// `crate::destination::named_destination`; paths it merely reads ("about
/// README.md") are never classified. The destination is then judged by the disk:
/// a small UTF-8 file inside the workspace is `Edit`; a missing path inside the
/// workspace is `New`; everything else is `Refused` with the reason: an
/// absolute, `~`, `..` or non-plain path, a symlink or parent resolving outside
/// the workspace, a directory, a file over `COMPOSE_EDIT_MAX_BYTES`, non-UTF-8
/// or unreadable content, and an ambiguous destination (several candidates).
/// A goal naming no destination is `New` (the model names the file).
pub(crate) fn classify_target(goal: &str, ws: &Path) -> TargetClass {
    classify_destination(goal, ws).0
}

/// `classify_target` plus the destination path it judged (None when the goal
/// names none). This is the single decision of a task: the prompt, the budget,
/// the merge target and the check of the reply's `filename:` all use it.
pub fn classify_destination(goal: &str, ws: &Path) -> (TargetClass, Option<String>) {
    let (class, dest, _) = classify_with_inputs(goal, ws);
    (class, dest)
}

/// Directories of the workspace the goal points at: every "Y folder" phrase and
/// every directory name used as a modifier ("the three inbox files").
fn goal_dirs(goal: &str, ws: &Path) -> Vec<String> {
    const NOUNS: &[&str] = &["files", "documents", "docs", "notes", "texts"];
    let mut out = crate::destination::goal_folders(goal);
    let ws_words: Vec<&str> = goal.split_whitespace().collect();
    for pair in ws_words.windows(2) {
        let trim = |c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-';
        let (w, next) = (pair[0].trim_matches(trim), pair[1].trim_matches(trim));
        if !w.is_empty()
            && NOUNS.contains(&next.to_ascii_lowercase().as_str())
            && !out.iter().any(|o| o == w)
            && ws.join(w).is_dir()
        {
            out.push(w.to_string());
        }
    }
    out
}

/// Where a named input word lives in the workspace: the word itself, else the
/// same name inside a directory the goal points at. None when it does not exist.
fn locate_input(word: &str, ws: &Path, dirs: &[String]) -> Option<String> {
    let exists = |p: &str| std::fs::symlink_metadata(ws.join(p)).is_ok();
    if exists(word) {
        return Some(word.to_string());
    }
    if word.contains('/') {
        return None;
    }
    dirs.iter()
        .map(|d| format!("{d}/{word}"))
        .find(|p| exists(p))
}

/// `classify_destination` plus the other paths the goal names (inputs, as the
/// goal wrote them). A name that exists in the workspace and is not the named
/// destination is an input, never a write target (sc#383).
fn classify_with_inputs(goal: &str, ws: &Path) -> (TargetClass, Option<String>, Vec<String>) {
    let Ok(ws) = std::fs::canonicalize(ws) else {
        return (TargetClass::New, None, Vec::new());
    };
    let is_file = |w: &str| {
        check_relative_path(w).is_ok() && std::fs::metadata(ws.join(w)).is_ok_and(|m| m.is_file())
    };
    let dirs = goal_dirs(goal, &ws);
    let is_input = |w: &str| {
        check_relative_path(w).is_ok() && locate_input(w, &ws, &dirs).is_some_and(|p| is_file(&p))
    };
    let n = match crate::destination::named_paths(goal, &is_file, &is_input) {
        Ok(n) => n,
        Err(e) => return (TargetClass::Refused(format!("{e}")), None, Vec::new()),
    };
    let Some(w) = n.dest else {
        return (TargetClass::New, None, n.inputs);
    };
    let class = classify_named(&w, &ws);
    (class, Some(w), n.inputs)
}

/// Input files for the prompt (sc#382): the paths the goal reads, plus the
/// regular files directly inside a folder the goal reads from. All resolved
/// strictly inside the canonical workspace, UTF-8 only, sorted by name, bounded
/// by `COMPOSE_INPUTS_MAX_FILES` and `COMPOSE_INPUTS_MAX_BYTES`. Returns the
/// prompt block ("" when nothing is shown), what was shown and what was left
/// out. A named input that leaves the workspace is refused. Never writes.
fn gather_inputs(
    goal: &str,
    ws: &Path,
    named: &[String],
    dest: Option<&str>,
) -> Result<(String, Vec<InputRef>, Vec<InputOmitted>), String> {
    let none = (String::new(), Vec::new(), Vec::new());
    let Ok(ws) = std::fs::canonicalize(ws) else {
        return Ok(none);
    };
    let dirs = goal_dirs(goal, &ws);
    let mut omitted: Vec<InputOmitted> = Vec::new();
    let mut cands: Vec<String> = Vec::new();
    let refuse = |w: &str, why: &str| format!("RunComposeTask: input {w} {why}; refusing");
    for w in named {
        if check_relative_path(w).is_err() {
            return Err(refuse(
                w,
                "is not a plain relative path inside the workspace",
            ));
        }
        let Some(rel) = locate_input(w, &ws, &dirs) else {
            continue; // does not exist: nothing to read, as before
        };
        match std::fs::canonicalize(ws.join(&rel)) {
            Ok(full) if full.starts_with(&ws) => {
                if full.is_file() && Some(rel.as_str()) != dest && !cands.contains(&rel) {
                    cands.push(rel);
                }
            }
            _ => return Err(refuse(w, "resolves outside the workspace")),
        }
    }
    for d in crate::destination::input_folders(goal) {
        let Ok(full) = std::fs::canonicalize(ws.join(&d)) else {
            continue;
        };
        if check_relative_path(&d).is_err() || !full.starts_with(&ws) || !full.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&full) else {
            continue;
        };
        for e in rd.flatten() {
            let rel = format!("{d}/{}", e.file_name().to_string_lossy());
            let ft = e.file_type();
            if ft.as_ref().is_ok_and(|t| t.is_file()) {
                if Some(rel.as_str()) != dest && !cands.contains(&rel) {
                    cands.push(rel);
                }
            } else if ft.is_ok_and(|t| t.is_symlink()) {
                omitted.push(InputOmitted {
                    name: rel,
                    reason: "symlink, not followed".into(),
                });
            }
        }
    }
    cands.sort();
    let mut included: Vec<InputRef> = Vec::new();
    let mut block = String::new();
    let mut used = 0usize;
    for rel in cands {
        let skip = |reason: String| InputOmitted {
            name: rel.clone(),
            reason,
        };
        if included.len() >= COMPOSE_INPUTS_MAX_FILES {
            omitted.push(skip(format!(
                "over the {COMPOSE_INPUTS_MAX_FILES}-file input limit"
            )));
            continue;
        }
        let Ok(meta) = std::fs::metadata(ws.join(&rel)) else {
            omitted.push(skip("unreadable".into()));
            continue;
        };
        if meta.len() as usize > COMPOSE_INPUTS_MAX_BYTES - used {
            omitted.push(skip(format!(
                "{} bytes does not fit the {COMPOSE_INPUTS_MAX_BYTES}-byte input limit",
                meta.len()
            )));
            continue;
        }
        let Ok(bytes) = std::fs::read(ws.join(&rel)) else {
            omitted.push(skip("unreadable".into()));
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            omitted.push(skip("not UTF-8 text".into()));
            continue;
        };
        if text.len() > COMPOSE_INPUTS_MAX_BYTES - used {
            omitted.push(skip("does not fit the input byte limit".into()));
            continue;
        }
        used += text.len();
        let nl = if text.ends_with('\n') { "" } else { "\n" };
        block.push_str(&format!("--- {rel} ---\n{text}{nl}--- end {rel} ---\n"));
        included.push(InputRef {
            sha256: hex(&Sha256::digest(text.as_bytes())),
            bytes: text.len(),
            name: rel,
        });
    }
    omitted.sort_by(|a, b| a.name.cmp(&b.name));
    if !block.is_empty() {
        block = format!(
            "\nInput files from the workspace (read only, material for the answer, not files to write):\n{block}"
        );
    }
    Ok((block, included, omitted))
}

fn classify_named(w: &str, ws: &Path) -> TargetClass {
    let refuse = |why: &str| {
        TargetClass::Refused(format!(
            "destination {w} cannot be written safely ({why}); refusing"
        ))
    };
    if check_relative_path(w).is_err() {
        return refuse("not a plain relative path inside the workspace");
    }
    let joined = ws.join(w);
    // symlink_metadata: a dangling or outward symlink still EXISTS.
    let Ok(lmeta) = std::fs::symlink_metadata(&joined) else {
        // Missing: a new document, if its nearest existing parent is a
        // directory inside the workspace (no symlinked parent leaving it).
        let mut anc = joined.parent();
        while let Some(a) = anc {
            if std::fs::symlink_metadata(a).is_ok() {
                break;
            }
            anc = a.parent();
        }
        return match anc.map(std::fs::canonicalize) {
            Some(Ok(p)) if p.starts_with(ws) && p.is_dir() => TargetClass::New,
            Some(Ok(p)) if p.starts_with(ws) => refuse("a parent is not a directory"),
            Some(Ok(_)) => refuse("a parent resolves outside the workspace"),
            _ => refuse("unresolvable parent"),
        };
    };
    let refuse = |why: &str| {
        TargetClass::Refused(format!(
            "destination {w} already exists but cannot be edited safely ({why}); refusing instead of treating it as a new document"
        ))
    };
    let full = match std::fs::canonicalize(&joined) {
        Ok(f) => f,
        Err(_) if lmeta.file_type().is_symlink() => return refuse("unresolvable symlink"),
        Err(_) => return refuse("unresolvable path"),
    };
    if !full.starts_with(ws) {
        return refuse("resolves outside the workspace");
    }
    let Ok(meta) = std::fs::metadata(&full) else {
        return refuse("unreadable");
    };
    if meta.is_dir() {
        return refuse("it is a directory");
    }
    if !meta.is_file() {
        return refuse("not a regular file");
    }
    if meta.len() > COMPOSE_EDIT_MAX_BYTES {
        return refuse(&format!(
            "{} bytes is over the {COMPOSE_EDIT_MAX_BYTES}-byte edit limit",
            meta.len()
        ));
    }
    let Ok(bytes) = std::fs::read(&full) else {
        return refuse("unreadable");
    };
    match String::from_utf8(bytes) {
        Ok(content) => TargetClass::Edit(w.to_string(), content),
        Err(_) => refuse("not UTF-8 text"),
    }
}

/// The additive new-document block, appended when the goal names a destination
/// that does not exist yet: the model is told which file to write and must
/// answer with that `filename:` line (issue #288).
pub fn new_document_block(path: &str) -> String {
    format!("\nWrite the new file {path}. The filename line of your answer must be exactly: filename: {path}")
}

/// The additive edit-mode block (NEXT-PHASE-1 v6 T5), appended to the fixed
/// proposal template only when the goal names an existing file.
fn edit_block(path: &str, content: &str) -> String {
    let nl = if content.ends_with('\n') { "" } else { "\n" };
    format!(
        "\nThe file {path} already exists. Its current content is:\n{content}{nl}\
         Write the complete new content of {path}: keep every existing line and make the requested change."
    )
}

/// The prompt of the production RunComposeTask path: the fixed template,
/// plus the edit-mode block when the goal names an existing file of the
/// canonical workspace `ws` (NEXT-PHASE-1 v6).
pub fn task_prompt(goal: &str, ws: &Path) -> String {
    task_prompt_and_target(goal, ws).0
}

/// `task_prompt` plus the edit target it showed the model: the path and the
/// exact content of the block, read once (NEXT-PHASE-1 v7 T5). The Skill
/// merges an edit reply into this same content, so the bytes the model saw
/// are the bytes the merge keeps.
fn task_prompt_and_target(goal: &str, ws: &Path) -> (String, Option<(String, String)>) {
    let (prompt, target, _) = task_plan(goal, ws)
        .unwrap_or_else(|_| (proposal_prompt(goal), None, ProposalKind::Document));
    (prompt, target)
}

/// Prompt, edit target and kind of a task, or the refusal when the goal names
/// an existing file that no edit block can be built for.
fn task_plan(goal: &str, ws: &Path) -> Result<TaskPrompt, String> {
    task_decision(goal, ws).map(|(plan, _)| plan)
}

/// `task_plan` plus the destination path the goal named (None when it names
/// none). The one decision of a task: a reply whose `filename:` differs from
/// the destination is refused by the Skill.
pub fn task_decision(goal: &str, ws: &Path) -> Result<(TaskPrompt, Option<String>), String> {
    task_decision_with_inputs(goal, ws).map(|(plan, dest, _, _)| (plan, dest))
}

/// A task decision with the inputs its prompt carries and the candidates left out.
type TaskDecisionInputs = (TaskPrompt, Option<String>, Vec<InputRef>, Vec<InputOmitted>);

/// `task_decision` plus the inputs its prompt carries and the candidates left
/// out (sc#382). The prompt is the fixed template, the input block when the
/// goal points at workspace files to read, then the edit or new-document block.
fn task_decision_with_inputs(goal: &str, ws: &Path) -> Result<TaskDecisionInputs, String> {
    let mut prompt = proposal_prompt(goal);
    let (class, dest, named) = classify_with_inputs(goal, ws);
    let kind = class.kind();
    if let TargetClass::Refused(why) = class {
        return Err(format!("RunComposeTask: {why}"));
    }
    let (block, included, omitted) = gather_inputs(goal, ws, &named, dest.as_deref())?;
    prompt.push_str(&block);
    match class {
        TargetClass::Refused(_) => unreachable!("refused above"),
        TargetClass::Edit(path, content) => {
            prompt.push_str(&edit_block(&path, &content));
            Ok((
                (prompt, Some((path, content)), kind),
                dest,
                included,
                omitted,
            ))
        }
        TargetClass::New => {
            if let Some(d) = &dest {
                prompt.push_str(&new_document_block(d));
            }
            Ok(((prompt, None, kind), dest, included, omitted))
        }
    }
}

/// Total bytes of input text a proposal prompt carries (sc#382).
const COMPOSE_INPUTS_MAX_BYTES: usize = 16384;
/// Most input files a proposal prompt carries (sc#382).
const COMPOSE_INPUTS_MAX_FILES: usize = 16;

/// Largest prior-lines x reply-lines table `merge_edit_reply` builds.
const COMPOSE_EDIT_MERGE_MAX_CELLS: usize = 4 << 20;

/// NEXT-PHASE-1 v7 T5: an edit-mode reply never removes an existing line.
///
/// v6 T5 showed the model answering an edit goal with only the changed part
/// (the heading it edits under plus the new line), which the v6 Skill wrote
/// as the whole file. Here the reply's content is merged into `prior`, the
/// exact content the model was shown: the reply lines that equal prior lines
/// (longest common subsequence, trailing whitespace ignored) are anchors,
/// every other reply line is inserted next to its anchor (after the anchor
/// above it; before the first anchor when no anchor is above it), and every
/// prior line is kept with its own bytes, in order. A whole-file reply that
/// keeps every line therefore merges to itself.
///
/// Limits: edit mode is additive. A line the reply leaves out or rewrites
/// stays in the file (a rewrite appears as an added line next to the old
/// one). Err when the reply shares no non-blank line with `prior` (no place
/// for the change), when it changes nothing, or when the table would exceed
/// `COMPOSE_EDIT_MERGE_MAX_CELLS`. The result ends with one newline.
#[cfg(test)]
fn merge_edit_reply(prior: &str, reply_content: &str) -> Result<String, String> {
    merge_edit_reply_placed(prior, reply_content, EditPlacement::NextToAnchor)
}

/// Where the reply lines that are not anchors go (sc#394).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditPlacement {
    /// Next to the anchor above (before the first anchor when none is above).
    NextToAnchor,
    /// After the last non-blank prior line, in reply order.
    AtEnd,
}

impl EditPlacement {
    /// `AtEnd` when the goal contains the word `append` as a whole word,
    /// case-insensitive (split on every non-alphanumeric character, so
    /// `appendix` and `appended` do not match); otherwise `NextToAnchor`.
    fn for_goal(goal: &str) -> Self {
        let append = goal
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w.eq_ignore_ascii_case("append"));
        if append {
            EditPlacement::AtEnd
        } else {
            EditPlacement::NextToAnchor
        }
    }
}

/// `merge_edit_reply` with the placement of the new lines chosen by the
/// caller. Kept prior lines, the shared-line requirement, the cell limit and
/// the single final newline are the same for both placements.
fn merge_edit_reply_placed(
    prior: &str,
    reply_content: &str,
    placement: EditPlacement,
) -> Result<String, String> {
    let mut p: Vec<&str> = prior.split('\n').collect();
    if prior.ends_with('\n') || prior.is_empty() {
        p.pop();
    }
    let r: Vec<&str> = reply_content.lines().collect();
    let (n, m) = (p.len(), r.len());
    if (n + 1).saturating_mul(m + 1) > COMPOSE_EDIT_MERGE_MAX_CELLS {
        return Err(format!(
            "edit reply too large to merge ({n} file lines x {m} reply lines)"
        ));
    }
    let eq = |i: usize, j: usize| p[i].trim_end() == r[j].trim_end();
    // l[i][j] = LCS length of p[i..] and r[j..].
    let w = m + 1;
    let mut l = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            l[i * w + j] = if eq(i, j) {
                l[(i + 1) * w + j + 1] + 1
            } else {
                l[(i + 1) * w + j].max(l[i * w + j + 1])
            };
        }
    }
    let mut anchors: Vec<(usize, usize)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if eq(i, j) && l[i * w + j] == l[(i + 1) * w + j + 1] + 1 {
            anchors.push((i, j));
            i += 1;
            j += 1;
        } else if l[(i + 1) * w + j] >= l[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if !anchors.iter().any(|&(i, _)| !p[i].trim().is_empty()) {
        return Err(
            "edit reply shares no line with the file: copy the existing line the change goes under, then the new lines"
                .into(),
        );
    }
    let mut out: Vec<&str> = Vec::with_capacity(n + m);
    if placement == EditPlacement::AtEnd {
        // Every prior line stays in place; the non-anchor reply lines follow
        // the last non-blank prior line, in reply order.
        let mut is_anchor = vec![false; m];
        for &(_, aj) in &anchors {
            is_anchor[aj] = true;
        }
        let last = p
            .iter()
            .rposition(|l| !l.trim().is_empty())
            .map_or(0, |i| i + 1);
        out.extend(&p[..last]);
        out.extend(
            r.iter()
                .zip(&is_anchor)
                .filter(|(_, a)| !**a)
                .map(|(l, _)| *l),
        );
        out.extend(&p[last..]);
    } else {
        let (mut pi, mut rj) = (0, 0);
        for (k, &(ai, aj)) in anchors.iter().chain(std::iter::once(&(n, m))).enumerate() {
            if k == 0 {
                // No anchor above: new lines hug the first anchor from above.
                out.extend(&p[pi..ai]);
                out.extend(&r[rj..aj]);
            } else {
                // New lines hug the anchor above them.
                out.extend(&r[rj..aj]);
                out.extend(&p[pi..ai]);
            }
            if ai < n {
                out.push(p[ai]);
            }
            (pi, rj) = (ai + 1, aj + 1);
        }
    }
    let mut merged = out.join("\n");
    merged.push('\n');
    let mut kept = p.join("\n");
    kept.push('\n');
    if merged == kept {
        return Err("edit reply changes nothing in the file".into());
    }
    Ok(merged)
}

/// The proposal text the Skill hands to AEGIS for one parsed reply. In edit
/// mode (`edit` = the target path and the content shown to the model) a reply
/// naming that path is merged into the content by `merge_edit_reply` and
/// returned in the canonical `filename: <path>` form; any other reply is
/// returned unchanged (the v5 whole-file proposal).
pub fn edit_proposal(reply: &str, edit: Option<(&str, &str)>) -> Result<String, String> {
    edit_proposal_placed(reply, edit, EditPlacement::NextToAnchor)
}

/// `edit_proposal` with the placement of the goal (`EditPlacement::for_goal`).
fn edit_proposal_placed(
    reply: &str,
    edit: Option<(&str, &str)>,
    placement: EditPlacement,
) -> Result<String, String> {
    let p = check_file_proposal(reply)?;
    let Some((path, prior)) = edit.filter(|(path, _)| *path == p.path) else {
        return Ok(reply.to_string());
    };
    let merged = merge_edit_reply_placed(prior, &p.content, placement)?;
    let text = format!("filename: {path}\n{merged}");
    match check_file_proposal(&text) {
        Ok(q) if q.path == path && q.content == merged => Ok(text),
        _ => Err(format!(
            "the merged content of {path} does not survive the proposal format"
        )),
    }
}

/// Why a reply that stopped at the token limit is never a proposal
/// (NEXT-PHASE-1 v6 N2): its content may be cut anywhere.
fn length_cut_refusal(g: &Generation) -> Option<String> {
    (g.finish_reason.as_deref() == Some("max_tokens")).then(|| {
        format!(
            "reply cut at the token limit after {} tokens (finish_reason max_tokens); a cut reply is never a proposal",
            g.tokens
        )
    })
}

/// The one correction line added to attempt k > 1.
pub fn retry_prompt(base: &str, reason: &str) -> String {
    format!(
        "{base}\nYour previous answer was refused ({reason}). \
         Start your answer with the line \"filename: <relative path>\"."
    )
}

/// Largest refused reply quoted back by [`corrective_retry_prompt`]; a longer
/// one is cut at a character boundary and marked. A reply is at most the
/// document token cap (1024 tokens in the v5 tasks), far under this.
const COMPOSE_FEEDBACK_MAX_BYTES: usize = 8192;

/// The markers around the quoted refused reply.
const REFUSED_REPLY_BEGIN: &str = "<<<REFUSED ANSWER";
const REFUSED_REPLY_END: &str = "REFUSED ANSWER>>>";

/// sc#384 v8 (Drake 2026-10-12, Option 3): the retry prompt that shows the
/// model its own refused reply next to the exact validation failure. The
/// goal, the requirements, the budgets and the attempt limit are untouched;
/// the quoted reply is prompt text only and never reaches the write path
/// (only a reply that passes every check becomes a proposal). Without a reply
/// (the previous attempt timed out or failed) it is [`retry_prompt`] byte for
/// byte, the v7 prompt.
pub fn corrective_retry_prompt(base: &str, reason: &str, refused: Option<&str>) -> String {
    let Some(text) = refused else {
        return retry_prompt(base, reason);
    };
    let mut shown = text.replace(REFUSED_REPLY_END, "[marker removed]");
    let mut cut = "";
    if shown.len() > COMPOSE_FEEDBACK_MAX_BYTES {
        let mut end = COMPOSE_FEEDBACK_MAX_BYTES;
        while !shown.is_char_boundary(end) {
            end -= 1;
        }
        shown.truncate(end);
        cut = "\n[the rest of the refused answer is not shown]";
    }
    let nl = if shown.ends_with('\n') { "" } else { "\n" };
    format!(
        "{base}\nYour previous answer was refused before anything was written ({reason}). \
         Here it is, between the markers, so you can see what to fix:\n\
         {REFUSED_REPLY_BEGIN}\n{shown}{nl}{REFUSED_REPLY_END}{cut}\n\
         Write a new complete answer that meets every requirement of the goal. \
         Start your answer with the line \"filename: <relative path>\"."
    )
}

fn new_attempt(k: u32) -> ProposalAttempt {
    ProposalAttempt {
        attempt: k,
        ms: 0,
        tokens: 0,
        outcome: String::new(),
        reason: None,
        text_sha256: None,
        text: None,
        aegis: None,
        finish_reason: None,
        token_ids: None,
        prompt_tokens: None,
        prompt_ids_sha256: None,
        decoding: None,
        ops: None,
        unmet_requirements: Vec::new(),
        shown_refused_text_sha256: None,
    }
}

/// The approved-proposal path of the Skill: exactly one attempt that returns
/// `text` unchanged, through the same requirement check as a model reply (the
/// bytes a human approved are still refused, before any write, when the goal
/// states a requirement they do not meet).
pub(crate) fn propose_approved(
    text: &str,
    prompt: &str,
    reqs: &[crate::requirements::Requirement],
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
) -> SkillOutput {
    let text = text.to_string();
    let one = move |_: &str, _: std::time::Duration| {
        Ok(Generation {
            text: text.clone(),
            finish_reason: Some("approved".into()),
            ..Default::default()
        })
    };
    propose_task_checked(&one, prompt, None, reqs, budget, attempt_budget, 1)
}

#[cfg(test)]
pub(crate) const TEST_DROP_RECORD_MARKER: &str = "drop-the-record-marker";

/// Why a task is refused when its requirement record is gone (issue #289).
pub(crate) const MISSING_RECORD_REASON: &str =
    "requirement record missing for this task: refused, an absent record is never treated as no requirements";

/// A task refused before the model ran: one attempt records the reason.
pub(crate) fn refused_without_model(why: String) -> SkillOutput {
    let mut a = new_attempt(1);
    a.outcome = "refused".into();
    a.reason = Some(why.clone());
    (Err(why), vec![a])
}

/// The AEGIS verify decision for one task: the result names exactly the
/// proposal the Skill recorded, that proposal parses as one file change, and
/// the COMPLETE content (the bytes that would be saved) meets every requirement
/// of the goal. A missing requirement record, or any uncertain span, refuses.
pub(crate) fn verify_task_result(
    ex: Option<&crate::requirements::Extraction>,
    proposal: Option<&SkillOutput>,
    result: u64,
) -> bool {
    let Some(ex) = ex else { return false };
    if ex.refusal().is_some() {
        return false;
    }
    let Some((Ok(t), _)) = proposal else {
        return false;
    };
    proposal_handle(t) == result
        && parse_file_proposal(t).is_some_and(|p| {
            crate::requirements::refusal_reason(&ex.requirements, &p.content).is_none()
        })
}

/// Runs up to `max_attempts` proposals within `budget`: attempt 1 always
/// starts; attempt k > 1 starts only if the remaining budget is at least
/// `attempt_budget` (the measured cost of one full attempt), and gets the
/// remaining budget as its limit. A reply that passes `check_file_proposal`
/// becomes `edit_proposal(reply, edit)`, so in edit mode the proposal is the
/// merged file; each attempt still records the model's own reply text.
///
/// Requirement validation: a parsed reply
/// whose COMPLETE content (the merged file in edit mode, i.e. the bytes that
/// would be saved) fails any of `reqs` is refused like a parse failure, the
/// unmet requirement(s) are recorded on the attempt, and the next attempt
/// gets them as its retry reason. `budget` is one deadline for the whole
/// task, started when this function starts and never reset per attempt.
/// Exhaustion returns Err: no proposal exists, so nothing is approved,
/// committed or written.
fn propose_task_checked(
    proposer: &(dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync),
    base: &str,
    edit: Option<(&str, &str)>,
    reqs: &[crate::requirements::Requirement],
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
    max_attempts: u32,
) -> (Result<String, String>, Vec<ProposalAttempt>) {
    propose_task_placed(
        proposer,
        base,
        edit,
        EditPlacement::NextToAnchor,
        reqs,
        budget,
        attempt_budget,
        max_attempts,
    )
}

/// `propose_task_checked` with the placement of the goal's new lines.
#[allow(clippy::too_many_arguments)]
fn propose_task_placed(
    proposer: &(dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync),
    base: &str,
    edit: Option<(&str, &str)>,
    placement: EditPlacement,
    reqs: &[crate::requirements::Requirement],
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
    max_attempts: u32,
) -> (Result<String, String>, Vec<ProposalAttempt>) {
    let start = std::time::Instant::now();
    let mut attempts: Vec<ProposalAttempt> = Vec::new();
    // An edit only adds lines (`merge_edit_reply` keeps every prior line), so a
    // prior file that already breaks a MaxLines can never satisfy it: refuse
    // now with a recorded reason instead of spending attempts.
    if let Some((path, prior)) = edit {
        let over: Vec<String> = reqs
            .iter()
            .filter(|r| matches!(r, crate::requirements::Requirement::MaxLines(_)))
            .filter_map(|r| r.check(prior))
            .collect();
        if !over.is_empty() {
            let why = format!(
                "unmet requirement: the existing {path} already breaks it ({}) and an edit only adds lines",
                over.join("; ")
            );
            let mut a = new_attempt(1);
            a.outcome = "refused".into();
            a.reason = Some(why.clone());
            a.unmet_requirements = over;
            return (Err(why), vec![a]);
        }
    }
    let mut last_reason = "no attempt made".to_string();
    // The previous attempt's refused reply (sc#384 v8), quoted in the next
    // prompt; None after a timeout or error, which leave no reply.
    let mut last_refused: Option<String> = None;
    let need_ms = (attempt_budget.as_millis() as u64).max(1);
    for k in 1..=max_attempts {
        let remaining = budget.saturating_sub(start.elapsed());
        if k > 1 && (remaining.as_millis() as u64) < need_ms {
            last_reason = format!(
                "{last_reason}; no attempt {k}: {} ms left < {need_ms} ms per-attempt budget",
                remaining.as_millis()
            );
            break;
        }
        let prompt = if k == 1 {
            base.to_string()
        } else {
            corrective_retry_prompt(base, &last_reason, last_refused.as_deref())
        };
        let shown = last_refused
            .take()
            .filter(|_| k > 1)
            .map(|t| hex(&Sha256::digest(t.as_bytes())));
        let t0 = std::time::Instant::now();
        let out = proposer(&prompt, remaining);
        let last_ms = t0.elapsed().as_millis() as u64;
        let mut a = new_attempt(k);
        a.shown_refused_text_sha256 = shown;
        a.ms = last_ms;
        match out {
            Ok(g) => {
                a.tokens = g.tokens;
                a.finish_reason = g.finish_reason.clone();
                a.token_ids = g.token_ids.clone();
                a.prompt_tokens = g.prompt_tokens;
                a.prompt_ids_sha256 = g.prompt_ids_sha256.clone();
                a.decoding = g.decoding.clone();
                a.ops = g.ops.clone();
                a.text_sha256 = Some(hex(&Sha256::digest(g.text.as_bytes())));
                a.text = Some(g.text.clone());
                // NEXT-PHASE-1 v6 N2: a length-cut reply is refused before parsing.
                let checked = match length_cut_refusal(&g) {
                    Some(why) => Err(why),
                    None => edit_proposal_placed(&g.text, edit, placement).and_then(|p| {
                        // The complete content that would be saved.
                        let content = check_file_proposal(&p)?.content;
                        match crate::requirements::refusal_reason(reqs, &content) {
                            Some(why) => {
                                a.unmet_requirements = crate::requirements::unmet(reqs, &content);
                                Err(why)
                            }
                            None => Ok(p),
                        }
                    }),
                };
                match checked {
                    Ok(proposal) => {
                        a.outcome = "parsed".into();
                        attempts.push(a);
                        return (Ok(proposal), attempts);
                    }
                    Err(e) => {
                        a.outcome = "refused".into();
                        a.reason = Some(e.clone());
                        last_reason = e;
                        last_refused = Some(g.text.clone());
                    }
                }
            }
            Err(e) => {
                a.outcome = if e.starts_with(WALL_CLOCK_REASON_PREFIX) {
                    "timeout"
                } else {
                    "error"
                }
                .into();
                a.reason = Some(e.clone());
                last_reason = e;
            }
        }
        attempts.push(a);
    }
    (Err(last_reason), attempts)
}

pub(crate) fn record_view(
    compose: &mut Compose,
    r: &aien_omega_compose::Record,
) -> ComposeRecordView {
    let host = r.subject == aien_omega_compose::SUBJECT_HOST;
    let note = if !host {
        None
    } else if r.tag == aien_omega_compose::ffi::RXC_HOST_TAG_REPAIR_TAIL {
        Some("repair_tail".to_string())
    } else {
        NoteKind::from_tag(r.tag).map(|k| k.name().to_string())
    };
    let text = if host && NoteKind::from_tag(r.tag).is_some() {
        compose
            .payload(r.id)
            .ok()
            .and_then(|p| note_bytes(&p))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    } else {
        None
    };
    ComposeRecordView {
        id: r.id,
        cls: r.cls,
        kind: r.kind,
        subject: r.subject,
        tag: r.tag,
        links: r.links.to_vec(),
        digest: hex(&r.digest),
        verified: r.verified == 1,
        note,
        text,
    }
}

const CLOSED_REFUSAL: &str = "compose home closed: the daemon is shutting down";

pub(crate) struct ComposeHome {
    pub(crate) compose: Compose,
    /// The ALLEN identity this home resolved (`None` = not engaged).
    pub(crate) allen: Option<aien_allen::Resolved>,
    /// ALLEN scoped memory (arch#159), opened with the identity above.
    pub(crate) memory: crate::allen_memory::MemoryState,
    pub(crate) machine_id: String,
    prompts: Arc<parking_lot::Mutex<HashMap<u64, TaskEntry>>>,
    /// Proposer hook: task -> approved proposal text (crate::approved).
    approved: Arc<parking_lot::Mutex<HashMap<u64, String>>>,
    /// What the model Skill returned per task: the text, or why it gave
    /// none, and every attempt it made.
    proposals: Arc<parking_lot::Mutex<HashMap<u64, SkillOutput>>>,
    /// The Cortex record mark (ACCEPTANCE-v3 2.1): where it lives, the raw
    /// machine id it binds, and the last mark written (or verified).
    mark_path: PathBuf,
    machine_raw: [u8; 32],
    mark: Option<Mark>,
    /// seq of the last mark this home knows of (0 = none yet).
    mark_seq: u64,
    /// Wall budgets in force; every run sets omega's wait from its own.
    budgets: ComposeBudgets,
}

impl ComposeHome {
    /// Write a new record mark when the journal holds more records than the
    /// last mark (ACCEPTANCE-v3 2.1). Called only after appends returned,
    /// so the mark never runs ahead of the journal. Returns the mark written.
    pub(crate) fn advance_mark(&mut self) -> Result<Option<Mark>, String> {
        let n = self
            .compose
            .info()
            .map_err(|e| format!("record mark: info: {e}"))?
            .records;
        // A version 1 mark (no chain) is rewritten even at the same count.
        if self
            .mark
            .as_ref()
            .is_some_and(|m| m.records > n || (m.records == n && m.chain.is_some()))
        {
            return Ok(None);
        }
        // Extend the verified chain of the last mark; without one, hash the
        // whole journal as this home opened it.
        let (from, start) = match &self.mark {
            Some(Mark {
                records,
                chain: Some(c),
                ..
            }) => (*records, *c),
            _ => (0, [0u8; 32]),
        };
        let chain = record_chain(&mut self.compose, start, from + 1, n)
            .map_err(|e| format!("record mark: {e}"))?;
        let digest = if n == 0 {
            [0u8; 32]
        } else {
            self.compose
                .record(n)
                .map_err(|e| format!("record mark: record {n}: {e}"))?
                .digest
        };
        let m = Mark {
            machine_id: self.machine_raw,
            seq: self.mark_seq + 1,
            records: n,
            digest,
            chain: Some(chain),
        };
        cortex_mark::write(&self.mark_path, &m)?;
        self.mark_seq = m.seq;
        self.mark = Some(m.clone());
        Ok(Some(m))
    }
}

type SkillOutput = (Result<String, String>, Vec<ProposalAttempt>);
/// The prompt of one task, its edit target and its kind (`task_plan`).
pub type TaskPrompt = (String, Option<(String, String)>, ProposalKind);
/// A task's plan plus the requirements its goal states (`requirements::extract`).
/// Prompt plan, requirements, and the destination the goal named (one decision).
type TaskEntry = (
    TaskPrompt,
    crate::requirements::Extraction,
    (Option<String>, PathBuf),
    EditPlacement,
);

/// Owns the composition home of this process.
pub struct ComposeBridge {
    dir: PathBuf,
    proposer: ComposeProposer,
    /// Proposer for full-document tasks (own token cap); `None` = `proposer`.
    doc_proposer: Option<ComposeProposer>,
    proposer_label: String,
    home: std::sync::Mutex<Option<ComposeHome>>,
    /// Set when the start-up reconcile failed or was refused (ACCEPTANCE-v3
    /// 2.5): effect commands refuse until an operator reconcile succeeds.
    reconcile_failed: std::sync::Mutex<Option<String>>,
    /// Set by `close`: the daemon is shutting down, the home is closed and
    /// must not be reopened lazily by a connection task that outlives `run`
    /// (sovereign-core #306).
    closed: std::sync::atomic::AtomicBool,
    /// sovereign-core #297, #328: when true, `ComposeAuthorize` needs the
    /// approval desk's MAC. Default true (Drake decision b, 2026-10-08);
    /// false is the legacy OS-user-only authorize, only by explicit choice.
    authorize_requires_desk: bool,
    /// The loaded model files and the daemon start, set by the daemon after
    /// load. Without them a compose proposal gets no generation record.
    identity: std::sync::Mutex<
        Option<(
            crate::generation::ModelIdentity,
            crate::generation::DaemonStart,
        )>,
    >,
    /// Correlation trace (sovereign-core #398). Observational only: events
    /// carry ids and digests, never prompts, proposals or file contents, and
    /// nothing reads them to decide. `NullSink` by default (tracing off).
    trace: Arc<dyn aien_trace::TraceSink>,
    /// One trace id per bridge (per daemon start and compose home).
    trace_id: aien_trace::TraceId,
    /// Event id of the first event this bridge emitted; the parent of all its
    /// later events. 0 until the first event.
    trace_root: std::sync::atomic::AtomicU64,
}

impl ComposeBridge {
    pub fn new(dir: PathBuf, proposer: ComposeProposer, proposer_label: &str) -> Self {
        Self {
            dir,
            proposer,
            doc_proposer: None,
            proposer_label: proposer_label.to_string(),
            home: std::sync::Mutex::new(None),
            reconcile_failed: std::sync::Mutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
            authorize_requires_desk: true,
            identity: std::sync::Mutex::new(None),
            trace: Arc::new(aien_trace::NullSink),
            trace_id: aien_trace::TraceId::derive(
                aien_trace::now_unix_ms(),
                u64::from(std::process::id()),
                0,
            ),
            trace_root: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Installs the correlation trace sink (#398). Default `NullSink`.
    pub(crate) fn with_trace_sink(mut self, sink: Arc<dyn aien_trace::TraceSink>) -> Self {
        self.trace = sink;
        self
    }

    /// This bridge's trace id (correlation key, not a secret).
    pub fn trace_id(&self) -> aien_trace::TraceId {
        self.trace_id
    }

    /// True when a trace sink is installed. Call sites check this before
    /// building ids so that a disabled sink costs no allocation.
    pub(crate) fn trace_enabled(&self) -> bool {
        self.trace.enabled()
    }

    /// Emits one correlation event for this bridge. The first event becomes
    /// the root; every later one is its child. With the sink disabled this
    /// allocates nothing and reads no clock (callers build the ids only after
    /// `trace_enabled`). Returns the event id (0 if off).
    pub(crate) fn trace_event(
        &self,
        kind: aien_trace::EventKind,
        status: aien_trace::EventStatus,
        effect_certainty: Option<aien_trace::EffectCertainty>,
        ids: aien_trace::CorrelationIds,
        refs: aien_trace::EvidenceRefs,
        note: &str,
    ) -> u64 {
        use std::sync::atomic::Ordering;
        if !self.trace.enabled() {
            return 0;
        }
        let id = self.trace.next_event_id();
        let parent = self
            .trace_root
            .compare_exchange(0, id, Ordering::AcqRel, Ordering::Acquire)
            .err();
        self.trace.emit(aien_trace::TraceEvent {
            trace_id: self.trace_id,
            event_id: id,
            parent_event_id: parent,
            at_unix_ms: aien_trace::now_unix_ms(),
            kind,
            status,
            effect_certainty,
            ids,
            refs,
            note: (!note.is_empty()).then(|| aien_trace::BoundedNote::new(note)),
        });
        id
    }

    /// Use `doc_proposer` (its own token cap) for full-document tasks.
    pub fn with_doc_proposer(mut self, doc_proposer: ComposeProposer) -> Self {
        self.doc_proposer = Some(doc_proposer);
        self
    }

    /// The model files this daemon loaded (digests hashed at load) and when it
    /// started. Set once by the daemon; the compose task uses them to write the
    /// generation record of the proposal it commits. Evidence only.
    pub fn set_model_identity(
        &self,
        identity: crate::generation::ModelIdentity,
        started: crate::generation::DaemonStart,
    ) {
        *self.identity.lock().unwrap_or_else(|e| e.into_inner()) = Some((identity, started));
    }

    /// Turn the desk-MAC requirement on `ComposeAuthorize` on or off (#297).
    /// On by default (#328); off is for dev and tests of the legacy path, and a
    /// server refuses to start with it off in a strict run.
    pub fn with_authorize_requires_desk(mut self, on: bool) -> Self {
        self.authorize_requires_desk = on;
        self
    }

    /// True when `ComposeAuthorize` needs the approval desk's MAC (#297).
    pub fn authorize_requires_desk(&self) -> bool {
        self.authorize_requires_desk
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Close the compose home and refuse to reopen it. Daemon shutdown calls
    /// this before `run` returns: connection tasks hold their own `Arc` of the
    /// bridge and can outlive `run`, so without it `rxc_host_close` (the drop of
    /// the home) ran at an unspecified later time and a successor opening the
    /// same home in that window saw a journal behind its J-Space anchor
    /// (E_REPLAY, sovereign-core #306). Waits for a command in flight (it holds
    /// the home lock). Blocking: call it from a blocking context.
    pub fn close(&self) {
        let mut guard = match self.home.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        *guard = None;
    }

    /// True after `close`.
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn open_home(&self) -> Result<ComposeHome, String> {
        self.open_home_marked(None)
    }

    /// Open the home and check its Cortex record mark (ACCEPTANCE-v3 2.2).
    /// `rebuilt_from` = the seq of a mark RecoverComposeHome just set aside:
    /// the home then opens without a mark and gets a fresh one.
    fn open_home_marked(&self, rebuilt_from: Option<u64>) -> Result<ComposeHome, String> {
        // The one place every lazy open goes through (with_home, run_task_inner,
        // record_digest, recover): callers hold the home lock and close() sets
        // `closed` under the same lock, so no site can reopen after close.
        if self.is_closed() {
            return Err(CLOSED_REFUSAL.to_string());
        }
        // Refuse a bad budget before anything is opened (never fall back).
        let budgets = compose_budgets_from_env()
            .map_err(|e| format!("compose home {} refused: {e}", self.dir.display()))?;
        let root = machine_root(&self.dir)?;
        let mark_path = cortex_mark::mark_path(&self.dir);
        let existing = match rebuilt_from {
            Some(_) => None,
            None => cortex_mark::read(&mark_path)
                .map_err(|why| mark_refusal(&self.dir, &MarkRefusal::Damaged(why)))?,
        };
        let journal_present = std::fs::metadata(self.dir.join("cortex.cx"))
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        // rxc_host_open appends nothing: `info.records` is the read-only probe
        // count, so the count check runs before the open appends anything.
        let (mut compose, info) =
            Compose::open(&self.dir, RootKind::Provisioned, &root, 0xA1E4_0001)
                .map_err(|e: ComposeError| refusal(&self.dir, &e))?;
        // omega settles after a fixed wait. Every run sets it again to its own
        // budget + 1 s (run_task_inner); here the edit value is checked once, so an
        // old omega without rxc_host_set_wait_ms refuses a changed edit budget at open.
        compose
            .set_wait_ms((budgets.edit + COMPOSE_WAIT_MARGIN).as_millis() as u32)
            .map_err(|e| refusal(&self.dir, &e))?;
        let plan = cortex_mark::plan(
            existing.as_ref(),
            &info.machine_id,
            info.records,
            journal_present,
        )
        .map_err(|r| mark_refusal(&self.dir, &r))?;
        let prompts: Arc<parking_lot::Mutex<HashMap<u64, TaskEntry>>> = Arc::default();
        let proposals: Arc<parking_lot::Mutex<HashMap<u64, SkillOutput>>> = Arc::default();
        let approved: Arc<parking_lot::Mutex<HashMap<u64, String>>> = Arc::default();
        let (pr, pp, proposer) = (prompts.clone(), proposals.clone(), self.proposer.clone());
        let doc_proposer = self
            .doc_proposer
            .clone()
            .unwrap_or_else(|| self.proposer.clone());
        let pa = approved.clone();
        compose
            .register_skill(COMPOSE_MODEL_SKILL, None, 10, move |task| {
                // The requirement record of this task. A missing record is a refusal,
                // never an empty requirement list (issue #289).
                let entry = pr.lock().get(&task).cloned();
                let Some(((prompt, target, kind), ex, dest, placement)) = entry else {
                    pp.lock()
                        .insert(task, refused_without_model(MISSING_RECORD_REASON.into()));
                    return None;
                };
                // Goal text that looks like a measurable requirement but could not
                // be read reliably: refuse before any model call.
                if let Some(why) = ex.refusal() {
                    pp.lock().insert(task, refused_without_model(why));
                    return None;
                }
                let reqs = ex.requirements;
                // Proposer hook (crate::approved): an approved proposal for this
                // task replaces the model's reply. One attempt, the same
                // template check, the same AEGIS contract and commit below.
                // edit = None: an approved whole-file proposal never goes
                // through merge_edit_reply, so the committed bytes are the
                // approved bytes even when the path already exists.
                let budget = budgets.for_kind(kind);
                let attempt_budget = compose_attempt_budget(kind, budget);
                let fixed = pa.lock().get(&task).cloned();
                let (out, attempts) = match fixed {
                    Some(text) => propose_approved(&text, &prompt, &reqs, budget, attempt_budget),
                    // Fixed template + bounded automatic retry (ACCEPTANCE-v2 3a, 3b),
                    // measured per-attempt budget (ACCEPTANCE-v3 3b); an edit reply
                    // is merged into the content the model was shown (v7 T5).
                    None => {
                        let edit = target.as_ref().map(|(p, c)| (p.as_str(), c.as_str()));
                        let p = match kind {
                            ProposalKind::Edit => &proposer,
                            ProposalKind::Document => &doc_proposer,
                        };
                        propose_task_placed(
                            p.as_ref(),
                            &prompt,
                            edit,
                            placement,
                            &reqs,
                            budget,
                            attempt_budget,
                            COMPOSE_MAX_ATTEMPTS,
                        )
                    }
                };
                let (out, attempts) = enforce_destination(
                    (out, attempts),
                    dest.0.as_deref(),
                    &dest.1,
                    kind == ProposalKind::Edit,
                );
                let h = out.as_ref().ok().map(|t| proposal_handle(t));
                pp.lock().insert(task, (out, attempts));
                // Test seam (compiled only into unit tests): lose the requirement
                // record between the Skill and AEGIS verify, as a bug would (#289).
                #[cfg(test)]
                if prompt.contains(TEST_DROP_RECORD_MARKER) {
                    pr.lock().remove(&task);
                }
                h
            })
            .map_err(|e| format!("compose register skill: {e}"))?;
        let (pv, pq) = (proposals.clone(), prompts.clone());
        compose
            .set_verify(move |task, result| {
                // AEGIS contract: the result names exactly the proposal text
                // the Skill recorded for this task, and that text parses as one
                // file change (a relative path and nonempty content).
                // The requirements of the goal are checked again on these same
                // bytes, the ones that would be saved. No requirement record
                // for the task means NO approval (issue #289).
                let ex = pq.lock().get(&task).map(|p| p.1.clone());
                let proposals = pv.lock();
                verify_task_result(ex.as_ref(), proposals.get(&task), result)
            })
            .map_err(|e| format!("compose set verify: {e}"))?;
        // Open the composition now, so a home behind its anchor is refused here
        // with its name (E_REPLAY), not inside the first run.
        compose.info().map_err(|e| refusal(&self.dir, &e))?;
        // The digest check needs the open home (the open appended the five
        // start-up records; ACCEPTANCE-v3 G1, G3).
        if let Plan::Verify(m) = &plan {
            if m.records > 0 {
                let at = compose.record(m.records).ok().map(|r| r.digest);
                cortex_mark::check_digest(m, at).map_err(|r| mark_refusal(&self.dir, &r))?;
                // Receipt continuity (#408 row 8): every record before the
                // newest, through the chain the mark carries.
                if m.chain.is_some() {
                    let at = record_chain(&mut compose, [0u8; 32], 1, m.records).ok();
                    cortex_mark::check_chain(m, at).map_err(|r| mark_refusal(&self.dir, &r))?;
                }
            }
        }
        // ALLEN identity gate: after the open (record 1 exists, skills are
        // registered), before the home is handed out. Fatal when engaged.
        let allen = allen_gate(&self.dir, &mut compose, &info.machine_id);
        let memory = crate::allen_memory::MemoryState::open(&self.dir, allen.as_ref());
        let mut home = ComposeHome {
            compose,
            allen,
            memory,
            budgets,
            machine_id: hex(&info.machine_id),
            prompts,
            approved,
            proposals,
            mark_path,
            machine_raw: info.machine_id,
            mark: existing.clone(),
            mark_seq: existing
                .as_ref()
                .map_or(rebuilt_from.unwrap_or(0), |m| m.seq),
        };
        let written = home.advance_mark().map_err(|e| {
            format!(
                "compose home {} refused: record mark update failed (E_MARK_WRITE): {e}",
                self.dir.display()
            )
        })?;
        let event = match rebuilt_from {
            Some(_) => written.as_ref().map(|w| {
                format!(
                    "Cortex mark: mark rebuilt by RecoverComposeHome (seq {}, records {})",
                    w.seq, w.records
                )
            }),
            None => cortex_mark::open_event(&plan, info.records, written.as_ref()),
        };
        if let Some(line) = event {
            println!("{line}");
        }
        Ok(home)
    }

    /// Start-up reconcile failed or was refused: effect commands refuse
    /// until an operator reconcile succeeds (ACCEPTANCE-v3 2.5).
    pub fn set_reconcile_failed(&self, why: String) {
        if let Ok(mut g) = self.reconcile_failed.lock() {
            *g = Some(why);
        }
    }

    /// Why effect commands are refused, if they are.
    pub fn reconcile_failed(&self) -> Option<String> {
        match self.reconcile_failed.lock() {
            Ok(g) => g.clone(),
            Err(_) => Some("reconcile state lock poisoned".into()),
        }
    }

    pub(crate) fn clear_reconcile_failed(&self) {
        if let Ok(mut g) = self.reconcile_failed.lock() {
            *g = None;
        }
    }

    /// Blocking: runs inference inside the Skill. Call from a blocking thread.
    pub fn run_task(&self, goal: &str, workspace: &str) -> ControlResponse {
        self.run_task_in(goal, workspace, None)
    }

    /// `run_task` with an operator-named ALLEN memory context (`None` = no memory).
    pub fn run_task_in(
        &self,
        goal: &str,
        workspace: &str,
        context: Option<&str>,
    ) -> ControlResponse {
        match self.run_task_inner(goal, workspace, None, context) {
            Ok(r) => ControlResponse::ComposeTaskResult(Box::new(r)),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// Proposer hook entry (crate::approved): one compose run whose model
    /// Skill returns `approved_text` instead of calling the proposer. The run
    /// is otherwise the RunComposeTask run: J-Space branch, AEGIS verify
    /// callback, World commit, Cortex records. Blocking.
    pub(crate) fn run_approved_task(
        &self,
        goal: &str,
        workspace: &str,
        approved_text: &str,
    ) -> Result<ComposeTaskReport, String> {
        self.run_task_inner(goal, workspace, Some(approved_text), None)
    }

    fn run_task_inner(
        &self,
        goal: &str,
        workspace: &str,
        approved_text: Option<&str>,
        context: Option<&str>,
    ) -> Result<ComposeTaskReport, String> {
        if goal.trim().is_empty() {
            return Err("RunComposeTask: empty goal".into());
        }
        let ws = std::fs::canonicalize(workspace)
            .map_err(|e| format!("RunComposeTask: workspace {workspace}: {e}"))?;
        if !ws.is_dir() {
            return Err(format!(
                "RunComposeTask: workspace {} is not a directory",
                ws.display()
            ));
        }
        let (mut plan, dest, inputs, inputs_omitted) = task_decision_with_inputs(goal, &ws)?;
        // The requirements the goal states are checked on every path: the model
        // path and an approved proposal (refused before any write when unmet).
        let machine_goal =
            approved_text.is_some() && goal.starts_with(crate::approved::APPROVED_GOAL_PREFIX);
        let mut reqs = if machine_goal {
            crate::requirements::Extraction::default()
        } else {
            crate::requirements::analyze(goal)
        };
        // Added-line counts are measured against the file the proposal replaces
        // (the edit target's content, or nothing for a new file).
        if reqs.needs_prior() {
            let prior = plan.1.as_ref().map_or("", |(_, c)| c.as_str());
            reqs = reqs.resolved(prior);
        }
        let recognized: Vec<String> = reqs.requirements.iter().map(|r| r.label()).collect();
        let uncertain = reqs.uncertain.clone();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("clock: {e}"))?;
        let mut h = Sha256::new();
        h.update(goal.as_bytes());
        h.update([0u8]);
        h.update(ws.as_os_str().as_encoded_bytes());
        h.update(now.as_nanos().to_le_bytes());
        let d = h.finalize();
        let task = u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) | 1;

        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        // ALLEN persona (arch#159): only a model run reads it; an approved
        // proposal runs no model, so it carries no persona. Not engaged: the
        // prompt is exactly what it was.
        // ALLEN memory (arch#159): a model run with an operator-named context only.
        let (memory_block, memory) = match approved_text {
            None => {
                let (b, r) = crate::allen_memory::for_task(&home.memory, context)?;
                (b, Some(r))
            }
            Some(_) => (None, None),
        };
        plan.0 = crate::allen_memory::prefix_prompt(&plan.0, memory_block.as_deref());
        let persona_ctx = match approved_text {
            None => crate::persona::context_for(&self.dir, home.allen.as_ref()),
            Some(_) => None,
        };
        plan.0 = crate::persona::prefix_prompt(&plan.0, persona_ctx.as_ref());
        let persona = match approved_text {
            None => Some(crate::persona::report_for(persona_ctx.as_ref())),
            Some(_) => None,
        };
        // Deadlines agree on both sides: omega settles budget + 1 s after the run
        // starts, the Skill gives up at budget. Set only here, under the lock that
        // serializes runs, never while rx_compose_run is in flight. An omega without
        // rxc_host_set_wait_ms refuses a budget that needs a longer wait (no silent 30 s wait).
        let kind = plan.2;
        let wait = home.budgets.for_kind(kind) + COMPOSE_WAIT_MARGIN;
        home.compose
            .set_wait_ms(wait.as_millis() as u32)
            .map_err(|e| {
                format!(
                    "compose run refused: {kind:?} budget needs omega settle wait {} ms: {e}",
                    wait.as_millis()
                )
            })?;
        if let Some(t) = approved_text {
            home.approved.lock().insert(task, t.to_string());
        }
        home.prompts.lock().insert(
            task,
            (
                plan,
                reqs,
                (dest, ws.clone()),
                EditPlacement::for_goal(goal),
            ),
        );
        let run = home.compose.run(task, now.as_micros() as u64);
        home.prompts.lock().remove(&task);
        home.approved.lock().remove(&task);
        let output = home.proposals.lock().remove(&task);
        // A run appends whether or not it commits: move the record mark first.
        home.advance_mark()
            .map_err(|e| format!("compose run: record mark update failed (E_MARK_WRITE): {e}"))?;
        let r = run.map_err(|e| format!("compose run: {e}"))?;
        let committed = r.committed == 1;
        let (out, mut proposal_attempts) =
            output.unwrap_or((Err("the Skill did not run".into()), Vec::new()));
        // The attempt handed to AEGIS is the parsed one; record its verdict.
        if let Some(a) = proposal_attempts.iter_mut().find(|a| a.outcome == "parsed") {
            a.aegis = Some(if committed { "pass" } else { "fail" }.into());
        }
        let (proposal, uncommitted_proposal, proposer_error) = match out {
            Ok(t) if committed => (Some(t), None, None),
            Ok(t) => (None, Some(t), None),
            Err(e) => (None, None, Some(e)),
        };
        let parsed = proposal.as_deref().and_then(parse_file_proposal);
        // sovereign-core #261: the daemon's own record that THIS run committed
        // THIS proposal. `ComposeAuthorize` mints only from it. An approved
        // proposal has its own replay claim and grant, so it gets none.
        let mut compose_commit = None;
        let mut generation_record = None;
        if let (true, None, Some(p), Some(text)) = (
            committed,
            approved_text,
            parsed.as_ref(),
            proposal.as_deref(),
        ) {
            let wsc = std::fs::canonicalize(&ws)
                .map_err(|e| format!("workspace {}: {e}", ws.display()))?
                .display()
                .to_string();
            // Provenance (arch#162): the generation record of the attempt the
            // proposal came from, written now under the lock this run holds
            // (the model call itself cannot take it). Evidence only.
            generation_record = self.write_task_generation(home, task, &proposal_attempts);
            let prov = crate::generation::Provenance::new(generation_record, home.allen.as_ref());
            let id = crate::effects::write_compose_commit(
                home,
                &wsc,
                task,
                r.cx_promotion,
                r.cx_evidence,
                &hex(&Sha256::digest(text.as_bytes())),
                &p.path,
                &hex(&Sha256::digest(p.content.as_bytes())),
                Some(&hex(&Sha256::digest(goal.as_bytes()))),
                &prov,
            )
            .map_err(|e| {
                format!(
                    "compose run committed (promotion #{}) but its commit record was not written: {e}",
                    r.cx_promotion
                )
            })?;
            home.advance_mark().map_err(|e| {
                format!("compose run: record mark update failed (E_MARK_WRITE): {e}")
            })?;
            compose_commit = Some(id);
        }
        // Correlation only (#398): the turn happened, what record names it,
        // and whether it committed. No proposal text, no goal text.
        if self.trace_enabled() {
            self.trace_event(
                aien_trace::EventKind::ModelTurn,
                if proposer_error.is_some() {
                    aien_trace::EventStatus::Failed
                } else {
                    aien_trace::EventStatus::Ok
                },
                None,
                aien_trace::CorrelationIds {
                    generation_record,
                    decision_id: compose_commit.map(|c| aien_trace::BoundedId::new(&c.to_string())),
                    ..Default::default()
                },
                aien_trace::EvidenceRefs::default(),
                if proposer_error.is_some() {
                    "proposer_error"
                } else if committed {
                    "compose_commit"
                } else {
                    "uncommitted"
                },
            );
        }
        Ok(ComposeTaskReport {
            compose_commit,
            generation_record,

            compose_dir: self.dir.display().to_string(),
            machine_id: home.machine_id.clone(),
            task,
            outcome: r.outcome,
            committed,
            branch_count: r.n_branches,
            branches_reclaimed: r.branches_reclaimed,
            winner: (r.winner != aien_omega_compose::NONE).then_some(r.winner),
            aegis_pass_mask: r.aegis_pass_mask,
            cx_goal: r.cx_goal,
            cx_candidates: r.cx_candidate.iter().copied().filter(|&x| x != 0).collect(),
            cx_evidence: r.cx_evidence,
            cx_promotion: r.cx_promotion,
            cx_admissions: r.cx_admission.iter().copied().filter(|&x| x != 0).collect(),
            winner_digest: hex(&r.winner_digest),
            record_digest: hex(&r.record_digest),
            proposal_path: parsed.as_ref().map(|p| p.path.clone()),
            proposal_content_sha256: parsed
                .as_ref()
                .map(|p| hex(&Sha256::digest(p.content.as_bytes()))),
            proposal_sha256: proposal
                .as_ref()
                .map(|t| hex(&Sha256::digest(t.as_bytes()))),
            proposal,
            uncommitted_proposal,
            proposer_error,
            proposal_attempts,
            requirements_recognized: recognized,
            requirements_uncertain: uncertain,
            proposer: match approved_text {
                Some(_) => APPROVED_PROPOSER_LABEL.to_string(),
                None => self.proposer_label.clone(),
            },
            persona,
            memory,
            inputs,
            inputs_omitted,
        })
    }

    /// Write the generation record of the proposal attempt a committed task
    /// came from. `None` (with the reason logged when a write failed) when the
    /// daemon has no model identity, the attempt did not expose its token ids,
    /// or the append failed: no record means no claim, and the commit stands.
    fn write_task_generation(
        &self,
        home: &mut ComposeHome,
        task: u64,
        attempts: &[ProposalAttempt],
    ) -> Option<u64> {
        let (identity, started) = self
            .identity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        let a = attempts.iter().find(|a| a.outcome == "parsed")?;
        let (ids, text, prompt_sha, prompt_tokens) = (
            a.token_ids.as_deref()?,
            a.text.as_deref()?,
            a.prompt_ids_sha256.as_deref()?,
            a.prompt_tokens?,
        );
        let record = crate::generation::build_compose_record(
            &identity,
            &crate::generation::ComposeEvidence {
                prompt_ids_sha256: prompt_sha,
                prompt_tokens,
                output_ids: ids,
                text,
                finish_reason: a.finish_reason.as_deref().unwrap_or("unknown"),
                task,
                attempt: a.attempt,
                decoding: a.decoding.as_ref(),
                ops: a.ops.as_ref(),
            },
            started,
        );
        match crate::generation::write(home, &record) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("generation record not written: {e}");
                eprintln!("generation record not written: {e}");
                None
            }
        }
    }

    /// Append the daemon's generation record (evidence only; see
    /// `crate::generation`). Returns the record id only once it is written.
    pub(crate) fn record_generation(&self, record: &serde_json::Value) -> Result<u64, String> {
        self.with_home(|home| crate::generation::write(home, record))
    }

    pub(crate) fn with_home<T>(
        &self,
        f: impl FnOnce(&mut ComposeHome) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        let r = f(home);
        // Ok or not, the closure may have appended: the mark follows.
        if let Err(e) = home.advance_mark() {
            return Err(format!(
                "record mark update failed after the operation (E_MARK_WRITE): {e}"
            ));
        }
        r
    }

    /// S1 / S5: one operator record through the composition's writer. Gated
    /// records cannot be forged here, on any path (the socket handler calls
    /// this too): authorizations, effect phases, replay and compose-commit
    /// records are written only by the daemon (sovereign-core #261).
    pub fn note(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        if let Err(e) = crate::effects::check_reserved_note(kind, text) {
            return ControlResponse::Error(e);
        }
        self.write_note(kind, text, links)
    }

    /// `note` WITHOUT the reserved-record check. Cargo feature `test-support`
    /// only: it stands for an attacker who can append to the journal directly,
    /// so the ledger's own defences can be exercised. Not in a normal build.
    #[doc(hidden)]
    #[cfg(feature = "test-support")]
    pub fn note_unchecked(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        self.write_note(kind, text, links)
    }

    fn write_note(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        let k = match kind {
            "constraint" => NoteKind::Constraint,
            "authorization" => NoteKind::Authorization,
            "effect" => NoteKind::Effect,
            other => {
                return ControlResponse::Error(format!(
                    "ComposeNote: kind {other:?} (constraint, authorization, effect)"
                ))
            }
        };
        if links.len() > 4 {
            return ControlResponse::Error("ComposeNote: at most 4 links".into());
        }
        let mut l = [0u64; 4];
        l[..links.len()].copy_from_slice(links);
        let r = self.with_home(|home| {
            let id = home
                .compose
                .note(k, l, text.as_bytes())
                .map_err(|e| format!("ComposeNote: {e}"))?;
            let rec = home
                .compose
                .record(id)
                .map_err(|e| format!("ComposeNote: re-read {id}: {e}"))?;
            Ok(ComposeNoteReport {
                machine_id: home.machine_id.clone(),
                id,
                kind: k.name().to_string(),
                digest: hex(&rec.digest),
                text_sha256: hex(&Sha256::digest(text.as_bytes())),
                links: links.to_vec(),
            })
        });
        match r {
            Ok(r) => ControlResponse::ComposeNoted(r),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// ALLEN persona profile commands (arch#159). Opens the home if needed (the
    /// identity is resolved there). Blocking.
    pub fn allen_command(&self, cmd: &ControlCommand) -> ControlResponse {
        let r = self.with_home(|home| {
            if crate::allen_memory::is_memory_command(cmd) {
                return Ok(crate::allen_memory::handle_command(&home.memory, cmd));
            }
            Ok(crate::persona::handle_allen_command(
                &self.dir,
                home.allen.as_ref(),
                &self.proposer_label,
                cmd,
            ))
        });
        r.unwrap_or_else(ControlResponse::Error)
    }

    /// S6 / S8: host records plus the cited ids, digests re-checked.
    pub fn recall(&self, ids: &[u64], prefix: Option<u64>) -> ControlResponse {
        let r = self.with_home(|home| {
            let info = home
                .compose
                .info()
                .map_err(|e| format!("ComposeRecall: {e}"))?;
            let (host_recs, _) = home
                .compose
                .recall(aien_omega_compose::SUBJECT_HOST, 4096)
                .map_err(|e| format!("ComposeRecall: host records: {e}"))?;
            let host = host_recs
                .iter()
                .map(|r| record_view(&mut home.compose, r))
                .collect();
            let (mut cited, mut missing) = (Vec::new(), Vec::new());
            for &id in ids {
                match home.compose.record(id) {
                    Ok(r) => cited.push(record_view(&mut home.compose, &r)),
                    Err(_) => missing.push(id),
                }
            }
            let prefix_digest = match prefix {
                Some(n) if n <= info.records => {
                    let mut h = Sha256::new();
                    for id in 1..=n {
                        let r = home
                            .compose
                            .record(id)
                            .map_err(|e| format!("ComposeRecall: record {id}: {e}"))?;
                        h.update(r.digest);
                    }
                    Some(hex(&h.finalize()))
                }
                _ => None,
            };
            Ok(ComposeRecallReport {
                compose_dir: self.dir.display().to_string(),
                machine_id: home.machine_id.clone(),
                records_total: info.records,
                host,
                cited,
                missing,
                prefix,
                prefix_digest,
                compose_native: aien_omega_compose::LINKED,
                omega_sha: if aien_omega_compose::LINKED {
                    aien_omega_compose::EXPECTED_OMEGA_SHA.to_string()
                } else {
                    String::new()
                },
            })
        });
        match r {
            Ok(r) => ControlResponse::ComposeRecalled(Box::new(r)),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// Operator repair: close this process's handle, run rxc_host_recover,
    /// then check the Cortex record mark (ACCEPTANCE-v3 2.3): a mark that is
    /// damaged, ahead of the records kept, or names a different record is
    /// kept as `<mark>.lost-<seq>` (never deleted), the home is reopened with
    /// a fresh mark and one host `constraint` record names the repair.
    pub fn recover(&self) -> ControlResponse {
        if self.is_closed() {
            return ControlResponse::Error(CLOSED_REFUSAL.to_string());
        }
        let mut guard = match self.home.lock() {
            Ok(g) => g,
            Err(_) => return ControlResponse::Error("compose home lock poisoned".into()),
        };
        *guard = None;
        let root = match machine_root(&self.dir) {
            Ok(r) => r,
            Err(e) => return ControlResponse::Error(format!("RecoverComposeHome: {e}")),
        };
        let mark_path = cortex_mark::mark_path(&self.dir);
        let old_mark = cortex_mark::read(&mark_path);
        let r = match Compose::recover(&self.dir, RootKind::Provisioned, &root) {
            Ok(r) => r,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome {}: {e}",
                    self.dir.display()
                ))
            }
        };
        let mut report = ComposeRecoverReport {
            compose_dir: self.dir.display().to_string(),
            repaired: r.repaired == 1,
            tail_torn: r.tail_torn == 1,
            cause: r.cause,
            cut_lo: r.cut_lo,
            cut_hi: r.cut_hi,
            records_kept: r.records_kept,
            dropped_records: r.dropped_records,
            anchor_records: r.anchor_records,
            repair_record: r.event_id,
            cut_bytes_kept: r.cut_bytes_kept,
            cut_sha256: hex(&r.cut_sha256),
            opens: r.opens == 1,
            open_rc: r.open_rc,
            rolled_back: r.rolled_back,
            recovered_completed: r.recovered_completed,
            mark_lost: 0,
            mark_kept_as: None,
            mark_repair_record: 0,
        };
        // (old mark if readable, records lost, why) when the mark must go.
        let set_aside: Option<(Option<Mark>, u64, &str)> = match old_mark {
            Err(_) => Some((None, 0, "damaged")),
            Ok(None) => None,
            Ok(Some(m)) if m.records > r.records_kept => {
                let lost = m.records - r.records_kept;
                Some((Some(m), lost, "ahead of the journal"))
            }
            Ok(Some(m)) => match self.open_home() {
                Ok(home) => {
                    *guard = Some(home);
                    None
                }
                Err(e) if e.contains("E_MARK_DIGEST") => {
                    Some((Some(m), 0, "names a different record"))
                }
                Err(e) if e.contains("E_MARK_CHAIN") => {
                    Some((Some(m), 0, "records before the newest differ"))
                }
                // Refused for another reason: reported as before, mark kept.
                Err(_) => None,
            },
        };
        let Some((old, lost, why)) = set_aside else {
            return ControlResponse::ComposeRecovered(Box::new(report));
        };
        let old_seq = old.as_ref().map(|m| m.seq);
        let kept_as = cortex_mark::lost_path(&mark_path, old_seq);
        if let Err(e) = std::fs::rename(&mark_path, &kept_as) {
            return ControlResponse::Error(format!(
                "RecoverComposeHome: keep the record mark as {}: {e}",
                kept_as.display()
            ));
        }
        let mut home = match self.open_home_marked(Some(old_seq.unwrap_or(0))) {
            Ok(h) => h,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome: record mark kept as {}; the home still refuses: {e}",
                    kept_as.display()
                ))
            }
        };
        let text = serde_json::json!({
            "repair": "cortex-mark",
            "why": why,
            "mark_records": old.as_ref().map(|m| m.records),
            "journal_records": r.records_kept,
            "lost": lost,
            "old_seq": old_seq,
            "old_digest": old.as_ref().map(|m| hex(&m.digest)),
            "old_chain": old.as_ref().and_then(|m| m.chain.map(|c| hex(&c))),
            "kept_as": kept_as.display().to_string(),
        })
        .to_string();
        let id = match home
            .compose
            .note(NoteKind::Constraint, [0; 4], text.as_bytes())
            .and_then(|id| home.compose.record(id).map(|_| id))
        {
            Ok(id) => id,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome: record mark kept as {}; repair record failed: {e}",
                    kept_as.display()
                ))
            }
        };
        if let Err(e) = home.advance_mark() {
            return ControlResponse::Error(format!(
                "RecoverComposeHome: repair record #{id} written; record mark update failed (E_MARK_WRITE): {e}"
            ));
        }
        report.opens = true;
        report.mark_lost = lost;
        report.mark_kept_as = Some(kept_as.display().to_string());
        report.mark_repair_record = id;
        *guard = Some(home);
        ControlResponse::ComposeRecovered(Box::new(report))
    }

    /// Recall: the Cortex record by id, digest re-checked (cut 2 uses this
    /// to prove cited ids survive a restart).
    pub fn record_digest(&self, id: u64) -> Result<String, String> {
        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        home.compose
            .record(id)
            .map(|r| hex(&r.digest))
            .map_err(|e| format!("compose record {id}: {e}"))
    }
}

/// The reply must write the destination the goal named (issue #288): the
/// approved proposal and a model reply are both held to the same path, so the
/// save and the commit land on the file the prompt and budget were chosen for.
/// A goal that named no destination leaves the model free to name the file.
fn enforce_destination(
    out: SkillOutput,
    dest: Option<&str>,
    ws: &Path,
    is_edit: bool,
) -> SkillOutput {
    let (res, attempts) = out;
    let res = match res {
        Ok(t) => match parse_file_proposal(&t) {
            Some(p) if dest.is_some_and(|d| p.path != d) => Err(format!(
                "the proposal writes {} but the goal named the destination {}",
                p.path,
                dest.unwrap_or_default()
            )),
            // Defence in depth: a new-document task never replaces a file
            // that exists (symlinks and dangling links count as existing).
            Some(p) if !is_edit && std::fs::symlink_metadata(ws.join(&p.path)).is_ok() => {
                Err(format!(
                    "the proposal writes {} which already exists; refusing to overwrite it as a new document",
                    p.path
                ))
            }
            _ => Ok(t),
        },
        r => r,
    };
    (res, attempts)
}

#[cfg(test)]
mod requirement_tests {
    use super::*;
    use crate::requirements::extract;

    const SHORT: &str = "filename: DOC.md\na\nb\nc\n";

    #[test]
    fn approved_text_is_checked_against_the_goal_requirements() {
        let reqs = extract("write DOC.md in at least 10 lines");
        let (out, a) = propose_approved(
            SHORT,
            "base",
            &reqs,
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        let why = out.unwrap_err();
        assert!(
            why.contains("at least 10 non-empty lines, found 3"),
            "{why}"
        );
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].outcome, "refused");
        assert_eq!(
            a[0].unmet_requirements,
            ["at least 10 non-empty lines, found 3"]
        );
        // Met, or none stated: the approved bytes pass through unchanged.
        let (out, _) = propose_approved(
            SHORT,
            "base",
            &extract("at least 3 lines"),
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        assert_eq!(out.unwrap(), SHORT);
        let (out, a) = propose_approved(
            SHORT,
            "base",
            &[],
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        assert_eq!(out.unwrap(), SHORT);
        assert_eq!(a[0].finish_reason.as_deref(), Some("approved"));
    }

    #[test]
    fn approved_run_reports_recognized_and_refuses_unmet_before_any_write() {
        let _home = crate::home_guard::home_slot();
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let proposer: ComposeProposer = Arc::new(|_: &str, _: std::time::Duration| {
            panic!("the model proposer ran on the approved path")
        });
        let bridge = ComposeBridge::new(tmp.path().join("compose"), proposer, "test:approved");
        let r = bridge.run_approved_task(
            "write DOC.md in at least 10 lines",
            ws.to_str().unwrap(),
            SHORT,
        );
        if !aien_omega_compose::LINKED {
            assert!(r.is_err());
            return;
        }
        let r = r.unwrap();
        assert!(!r.committed, "{r:?}");
        assert!(r.proposal.is_none());
        assert_eq!(r.requirements_recognized, ["at least 10 non-empty lines"]);
        assert!(r.proposer_error.unwrap().contains("found 3"));
        assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
        // The same bytes pass when the goal states a requirement they meet.
        let r = bridge
            .run_approved_task(
                "write DOC.md in at least 3 lines",
                ws.to_str().unwrap(),
                SHORT,
            )
            .unwrap();
        assert!(r.committed, "{r:?}");
        assert_eq!(r.requirements_recognized, ["at least 3 non-empty lines"]);
        assert_eq!(r.proposal.as_deref(), Some(SHORT));
    }
}

#[cfg(test)]
mod verify_boundary_tests {
    use super::*;
    use crate::requirements::analyze;

    const DOC: &str = "filename: DOC.md\na\nb\nc\n";

    fn recorded(text: &str) -> SkillOutput {
        (Ok(text.to_string()), Vec::new())
    }

    #[test]
    fn verify_accepts_exactly_the_recorded_bytes_that_meet_the_goal() {
        let ex = analyze("write DOC.md in at least 3 lines");
        let out = recorded(DOC);
        let h = proposal_handle(DOC);
        assert!(verify_task_result(Some(&ex), Some(&out), h));
        // a result naming other bytes is refused
        assert!(!verify_task_result(Some(&ex), Some(&out), h ^ 1));
        // bytes that no longer meet the requirement are refused
        let short = analyze("write DOC.md in at least 4 lines");
        assert!(!verify_task_result(Some(&short), Some(&out), h));
        // a skill that returned no proposal is refused
        let none: SkillOutput = (Err("no".into()), Vec::new());
        assert!(!verify_task_result(Some(&ex), Some(&none), h));
        assert!(!verify_task_result(Some(&ex), None, h));
    }

    /// Issue #289: an absent requirement record used to become an empty list,
    /// so the verify step approved any well-formed file.
    #[test]
    fn missing_requirement_record_is_refused_never_empty() {
        let out = recorded(DOC);
        let h = proposal_handle(DOC);
        // the same bytes that pass with a record are refused without one
        let ex = analyze("write DOC.md in at least 3 lines");
        assert!(verify_task_result(Some(&ex), Some(&out), h));
        assert!(!verify_task_result(None, Some(&out), h));
        // and even a goal with no requirements needs its (empty) record
        let empty = analyze("write DOC.md");
        assert!(verify_task_result(Some(&empty), Some(&out), h));
        assert!(!verify_task_result(None, Some(&out), h));
    }

    #[test]
    fn uncertain_goal_is_refused_at_verify_even_when_bytes_look_fine() {
        let ex = analyze("write DOC.md with at least 3 lines of context");
        assert!(!ex.uncertain.is_empty());
        assert!(!verify_task_result(
            Some(&ex),
            Some(&recorded(DOC)),
            proposal_handle(DOC)
        ));
    }

    #[test]
    fn refused_without_model_records_one_refused_attempt() {
        let (out, a) = refused_without_model(MISSING_RECORD_REASON.into());
        assert_eq!(out.unwrap_err(), MISSING_RECORD_REASON);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].outcome, "refused");
        assert!(a[0].text.is_none());
    }
}

#[cfg(test)]
mod verify_callback_integration_tests {
    use super::*;

    fn bridge(dir: &std::path::Path) -> ComposeBridge {
        let proposer: ComposeProposer = Arc::new(|_: &str, _: std::time::Duration| {
            Ok(Generation {
                text: "filename: DOC.md\na\nb\nc\n".into(),
                finish_reason: Some("eos".into()),
                ..Default::default()
            })
        });
        ComposeBridge::new(dir.join("compose"), proposer, "test:verify-callback")
    }

    /// Issue #289 through the real compose run and the real verify callback:
    /// the same document commits with its requirement record and is refused
    /// by AEGIS (nothing committed, no promotion) when the record is gone.
    #[test]
    fn lost_requirement_record_is_refused_by_the_real_verify_callback() {
        let _home = crate::home_guard::home_slot();
        if !aien_omega_compose::LINKED {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let b = bridge(tmp.path());
        let ok = b.run_task("write DOC.md in at least 3 lines", ws.to_str().unwrap());
        let ControlResponse::ComposeTaskResult(ok) = ok else {
            panic!("{ok:?}")
        };
        assert!(ok.committed, "{ok:?}");
        let goal = format!("write DOC2.md in at least 3 lines {TEST_DROP_RECORD_MARKER}");
        let lost = b.run_task(&goal, ws.to_str().unwrap());
        let ControlResponse::ComposeTaskResult(lost) = lost else {
            panic!("{lost:?}")
        };
        assert!(!lost.committed, "{lost:?}");
        assert_eq!(lost.aegis_pass_mask & 1, 0, "{lost:?}");
        assert_eq!(lost.cx_promotion, 0, "{lost:?}");
    }
}

#[cfg(test)]
mod destination_enforcement_tests {
    use super::*;

    #[test]
    fn reply_must_write_the_named_destination() {
        let ok = "filename: docs/SUMMARY.md\n# s\n".to_string();
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().to_path_buf();
        let run = |d: Option<&str>, t: &str| {
            enforce_destination((Ok(t.to_string()), vec![]), d, &ws, false).0
        };
        assert_eq!(run(Some("docs/SUMMARY.md"), &ok), Ok(ok.clone()));
        let e = run(Some("README.md"), &ok).unwrap_err();
        assert!(
            e.contains("docs/SUMMARY.md") && e.contains("README.md"),
            "{e}"
        );
        assert_eq!(run(None, &ok), Ok(ok));
    }
}

#[cfg(test)]
mod new_document_overwrite_tests {
    use super::*;

    #[test]
    fn new_kind_proposal_never_replaces_an_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        std::fs::write(ws.join("README.md"), "# r\n").unwrap();
        std::os::unix::fs::symlink("nowhere", ws.join("dangling.md")).unwrap();
        let run = |dest: Option<&str>, path: &str, edit: bool| {
            let t = format!("filename: {path}\nbody\n");
            enforce_destination((Ok(t), vec![]), dest, ws, edit).0
        };
        for dest in [None, Some("README.md")] {
            let e = run(dest, "README.md", false).unwrap_err();
            assert!(e.contains("already exists"), "{e}");
        }
        assert!(run(None, "dangling.md", false).is_err());
        assert!(run(None, "fresh.md", false).is_ok());
        // An edit-kind task may write its own (existing) target.
        assert!(run(Some("README.md"), "README.md", true).is_ok());
    }
}

#[cfg(test)]
mod folder_destination_enforcement_tests {
    use super::*;

    #[test]
    fn proposal_in_the_folder_is_accepted_for_ctrl_goal() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("outbox")).unwrap();
        let ws = std::fs::canonicalize(tmp.path()).unwrap();
        let goal = "Write a file named ctrl.md in the outbox folder containing one line: control.";
        let dest = task_decision(goal, &ws).unwrap().1;
        let t = "filename: outbox/ctrl.md\ncontrol\n".to_string();
        let r = enforce_destination((Ok(t.clone()), vec![]), dest.as_deref(), &ws, false).0;
        assert_eq!(r, Ok(t));
    }
}

#[cfg(test)]
mod append_placement_tests {
    use super::*;

    const PRIOR: &str = "# Report\n\nOne.\nTwo.\nThree.\n";
    const APPEND: &str = "Append one line to report.md naming the topic.";

    #[test]
    fn append_goal_places_new_lines_at_end() {
        let placement = EditPlacement::for_goal(APPEND);
        let out = merge_edit_reply_placed(PRIOR, "# Report\nThe topic is harbours.\n", placement)
            .unwrap();
        assert_eq!(out, format!("{PRIOR}The topic is harbours.\n"));
    }

    #[test]
    fn append_goal_with_several_new_lines_keeps_reply_order() {
        let placement = EditPlacement::for_goal(APPEND);
        let out = merge_edit_reply_placed(PRIOR, "# Report\nNew A.\nNew B.\nNew C.\n", placement)
            .unwrap();
        assert_eq!(out, format!("{PRIOR}New A.\nNew B.\nNew C.\n"));
    }

    #[test]
    fn non_append_goal_keeps_anchor_placement() {
        let placement = EditPlacement::for_goal("Add a line under the heading of report.md");
        assert_eq!(placement, EditPlacement::NextToAnchor);
        let reply = "# Report\nNew line.\n";
        let out = merge_edit_reply_placed(PRIOR, reply, placement).unwrap();
        assert_eq!(out, "# Report\nNew line.\n\nOne.\nTwo.\nThree.\n");
        assert_eq!(out, merge_edit_reply(PRIOR, reply).unwrap());
    }

    #[test]
    fn append_goal_still_needs_a_shared_line() {
        let placement = EditPlacement::for_goal(APPEND);
        assert!(merge_edit_reply_placed(PRIOR, "Nothing shared.\n", placement).is_err());
        assert!(merge_edit_reply_placed(PRIOR, "\n", placement).is_err());
        assert!(merge_edit_reply_placed(PRIOR, PRIOR, placement).is_err());
    }

    #[test]
    fn append_word_detection() {
        let at_end = |g: &str| EditPlacement::for_goal(g) == EditPlacement::AtEnd;
        assert!(at_end("Append one line"));
        assert!(at_end("append"));
        assert!(at_end("APPEND"));
        assert!(at_end("Please, append: a line to report.md."));
        assert!(!at_end("Add a line under the heading of report.md"));
        assert!(!at_end("Write the appendix"));
        assert!(!at_end("The appended line"));
    }
}

/// sc#402 (RUN-3-GB10 E4): a leading `./` names the same workspace file as
/// the bare relative path; every other unsafe or non-plain form stays refused.
#[cfg(test)]
mod dotslash_tests {
    use super::*;

    fn path_of(name: &str) -> Result<String, String> {
        check_file_proposal(&format!("filename: {name}\n# Report\nbody\n")).map(|p| p.path)
    }

    #[test]
    fn leading_dot_slash_names_the_bare_path() {
        assert_eq!(path_of("./report.md"), Ok("report.md".into()));
        assert_eq!(path_of("././a/b.md"), Ok("a/b.md".into()));
        assert_eq!(path_of("`./report.md`"), Ok("report.md".into()));
        assert_eq!(path_of("report.md"), Ok("report.md".into()));
    }

    #[test]
    fn every_other_form_is_refused_as_before() {
        for bad in [
            "../x", "./../x", "/x", "a/../b", "a/./b", ".", "./", "~/x", ".//x", "a//b", "a\\b",
        ] {
            assert!(path_of(bad).is_err(), "{bad:?} must stay refused");
        }
        assert!(check_file_proposal("filename: \n# R\nbody\n").is_err());
        assert!(
            check_relative_path("./report.md").is_err(),
            "the strict check stays strict"
        );
    }

    #[test]
    fn an_edit_reply_naming_dot_slash_merges_into_the_named_file() {
        let prior = "# Report\n\nOne.\n";
        let out = edit_proposal_placed(
            "filename: ./report.md\n# Report\n\nOne.\nThe topic is harbours.\n",
            Some(("report.md", prior)),
            EditPlacement::for_goal("Append one line to report.md naming the topic."),
        )
        .unwrap();
        assert_eq!(
            out,
            "filename: report.md\n# Report\n\nOne.\nThe topic is harbours.\n"
        );
    }
}

/// The daemon proposer's timeout text (server.rs `wall_clock_reason`) and the
/// classifier that records an attempt as "timeout" share one prefix; a cut at
/// the token limit never carries it. Internal seam: the interface tests feed
/// the literal text, so this pins the daemon's own function to it.
#[cfg(test)]
mod wall_clock_reason_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn token_cut_and_wall_clock_reasons_are_distinct() {
        assert_eq!(wall_clock_reason(5), "model proposal exceeded 5 ms");
        let cut = |_: &str, _: Duration| {
            Ok(Generation {
                text: "filename: a.md\nline\n".into(),
                tokens: 1024,
                finish_reason: Some("max_tokens".into()),
                ..Default::default()
            })
        };
        let (out, a) = propose_task_checked(
            &cut,
            "p",
            None,
            &[],
            Duration::from_secs(120),
            Duration::from_secs(30),
            1,
        );
        let why = out.expect_err("a cut reply is never a proposal");
        assert_eq!(a[0].outcome, "refused");
        assert!(
            why.contains(
                "reply cut at the token limit after 1024 tokens (finish_reason max_tokens)"
            ),
            "{why}"
        );
        assert!(!why.contains("exceeded"), "{why}");
        let slow = |_: &str, limit: Duration| -> Result<Generation, String> {
            Err(wall_clock_reason(limit.as_millis()))
        };
        let (out, a) = propose_task_checked(
            &slow,
            "p",
            None,
            &[],
            Duration::from_secs(120),
            Duration::from_secs(30),
            1,
        );
        let why = out.expect_err("timed out");
        assert_eq!(a[0].outcome, "timeout");
        assert!(
            why.starts_with("model proposal exceeded ") && why.ends_with(" ms"),
            "{why}"
        );
        assert!(
            !why.contains("token limit") && !why.contains("max_tokens"),
            "{why}"
        );
    }
}
