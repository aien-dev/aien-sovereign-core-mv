//! NEXT-PHASE-2: durable effect intents on the compose home's Cortex journal
//! (ACCEPTANCE-v2 section 2).
//!
//! An external effect (today: the S5 `write_file`) is bracketed by host
//! records of kind `effect`, text JSON with a `phase` field:
//!
//!   intent     appended by the daemon, durable (Cortex `CX_OPEN_SYNC`)
//!              BEFORE the executor touches the world; one per authorization
//!   ack        appended after the write; its state is read from the world
//!   reconcile  appended later (daemon start, operator request, operator
//!              declaration) for an intent that never got a terminal state
//!
//! States: OPEN (intent, nothing after it), DONE, NOT_DONE, UNRESOLVED.
//! The world decides: target digest = content digest -> DONE, = the digest
//! the operator authorized against (`prior_sha256`) -> NOT_DONE, else
//! UNRESOLVED. An effect is never re-run by this module; a retry after
//! NOT_DONE needs a new authorization.
//!
//! Operator control records (kind `authorization`, text with `control`):
//! `stop`, `resume`, `revoke`. A stop is durable; a grant older than the
//! newest stop is stale forever.
//!
//! The ledger is rebuilt from the journal on every call, under the
//! compose-home lock, so the check and the intent append are one step.
use crate::control::{
    ComposeControlReport, ComposeNoteReport, ComposeReconcileReport, ComposeRecordView,
    ControlResponse, ReconcileDeclare, ReconcileOutcome,
};
use crate::spine::{record_view, ComposeBridge, ComposeHome};
use aien_omega_compose::{hex, NoteKind};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

pub const PHASE_INTENT: &str = "intent";
pub const PHASE_ACK: &str = "ack";
pub const PHASE_RECONCILE: &str = "reconcile";
/// Host records the ledger reads in one recall; more is refused, not truncated.
const HOST_RECALL_MAX: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectState {
    Open,
    Done,
    NotDone,
    Unresolved,
}

impl EffectState {
    pub fn name(self) -> &'static str {
        match self {
            Self::Open => "OPEN",
            Self::Done => "DONE",
            Self::NotDone => "NOT_DONE",
            Self::Unresolved => "UNRESOLVED",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "DONE" => Some(Self::Done),
            "NOT_DONE" => Some(Self::NotDone),
            "UNRESOLVED" => Some(Self::Unresolved),
            _ => None,
        }
    }
    pub fn terminal(self) -> bool {
        matches!(self, Self::Done | Self::NotDone)
    }
}

/// A named refusal at the effect boundary (`EFFECT_REFUSED <name>: detail`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub name: &'static str,
    pub detail: String,
}

impl Refusal {
    fn new(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EFFECT_REFUSED {}: {}", self.name, self.detail)
    }
}

/// One authorization record, as the ledger reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grant {
    pub id: u64,
    pub proposal_sha256: String,
    pub path: String,
    pub content_sha256: String,
    /// Absolute target (absent in grants written before NEXT-PHASE-2).
    pub target: Option<String>,
    /// `Some(None)`: authorized against an absent target. `None`: the grant
    /// predates NEXT-PHASE-2 and names no prior state (always stale).
    pub prior_sha256: Option<Option<String>>,
    /// The workspace the target must stay inside (sovereign-core #249). A
    /// grant without one opens no intent.
    pub workspace: Option<String>,
    /// Set on the grant the daemon writes after an approved compose (#249).
    pub approved: Option<ApprovedGrant>,
    /// The record's links.
    pub links: Vec<u64>,
}

/// Marker field of the grant the daemon itself writes after a COMMITTED
/// approved compose (sovereign-core #249). Reserved: `ComposeNote` refuses it.
pub const APPROVED_GRANT: &str = "approved_grant";

/// What an approved grant must be backed by: a COMMITTED replay claim whose
/// commit evidence names the grant's proposal, promotion and evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedLink {
    pub approval_key: String,
    pub replay_claim: u64,
    pub cx_promotion: u64,
    pub cx_evidence: u64,
}

/// What the ledger reads back from an approved grant: its link and the bound
/// approval identity its text names (so the approval key can be recomputed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedGrant {
    pub link: ApprovedLink,
    pub identity: crate::approved_auth::ApprovalIdentity,
}

/// What an executor asks for before it touches the world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentRequest {
    pub authorization: u64,
    pub proposal_sha256: String,
    pub path: String,
    pub target: String,
    pub content_sha256: String,
    pub executor_pid: u32,
    pub executor_start: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentRow {
    pub id: u64,
    pub authorization: u64,
    pub path: String,
    pub target: String,
    pub content_sha256: String,
    pub prior_sha256: Option<String>,
    pub executor_pid: u32,
    pub executor_start: u64,
    pub state: EffectState,
    /// Record that set `state` (0 while OPEN).
    pub state_record: u64,
    /// Target digest of the newest UNRESOLVED record (so one is written per
    /// distinct digest, not one per daemon start).
    pub unresolved_digest: Option<Option<String>>,
}

/// The effect ledger of one compose home, rebuilt from its host records.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    pub grants: BTreeMap<u64, Grant>,
    /// Approved grants by replay claim: more than one for a claim opens none.
    pub approved_by_claim: BTreeMap<u64, Vec<u64>>,
    pub intents: BTreeMap<u64, IntentRow>,
    /// authorization id -> intent id (one intent per authorization).
    pub spent: BTreeMap<u64, u64>,
    /// authorization id -> legacy (pre-NEXT-PHASE-2) write_file effect record.
    pub legacy_spent: BTreeMap<u64, u64>,
    /// authorization id -> revoke record.
    pub revoked: BTreeMap<u64, u64>,
    pub stops: Vec<u64>,
    pub resumes: Vec<u64>,
}

fn corrupt(id: u64, why: impl fmt::Display) -> Refusal {
    Refusal::new("CorruptLedger", format!("host record #{id}: {why}"))
}

fn s(v: &Value, k: &str, id: u64) -> Result<String, Refusal> {
    v.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| corrupt(id, format!("field {k:?} missing or not a string")))
}

fn u(v: &Value, k: &str, id: u64) -> Result<u64, Refusal> {
    v.get(k)
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt(id, format!("field {k:?} missing or not a number")))
}

/// `null` -> Some(None), "hex" -> Some(Some), missing -> None.
fn opt_digest(v: &Value, k: &str, id: u64) -> Result<Option<Option<String>>, Refusal> {
    match v.get(k) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(x)) => Ok(Some(Some(x.clone()))),
        Some(_) => Err(corrupt(id, format!("field {k:?} is not a digest or null"))),
    }
}

impl Ledger {
    /// Rebuild from host records, oldest first. Any unverified record, or a
    /// gated record that does not parse, refuses the whole ledger: effects
    /// are never decided on damaged state.
    pub fn from_records(host: &[ComposeRecordView]) -> Result<Self, Refusal> {
        let mut l = Ledger::default();
        for r in host {
            if !r.verified {
                return Err(corrupt(r.id, "digest does not verify"));
            }
            let Some(text) = r.text.as_deref() else {
                continue;
            };
            match r.note.as_deref() {
                Some("authorization") => l.read_authorization(r.id, text, &r.links)?,
                Some("effect") => l.read_effect(r.id, text)?,
                _ => {}
            }
        }
        Ok(l)
    }

    fn read_authorization(&mut self, id: u64, text: &str, links: &[u64]) -> Result<(), Refusal> {
        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            // An authorization that is not JSON authorizes nothing.
            Err(_) => return Ok(()),
        };
        if let Some(c) = v.get("control") {
            match c.as_str() {
                Some("stop") => self.stops.push(id),
                Some("resume") => self.resumes.push(id),
                Some("revoke") => {
                    let a = u(&v, "authorization", id)?;
                    self.revoked.entry(a).or_insert(id);
                }
                _ => return Err(corrupt(id, "unknown control record")),
            }
            return Ok(());
        }
        let (Some(p), Some(path), Some(c)) = (
            v.get("proposal_sha256").and_then(Value::as_str),
            v.get("path").and_then(Value::as_str),
            v.get("content_sha256").and_then(Value::as_str),
        ) else {
            return Ok(());
        };
        self.grants.insert(
            id,
            Grant {
                id,
                proposal_sha256: p.to_string(),
                path: path.to_string(),
                content_sha256: c.to_string(),
                target: v.get("target").and_then(Value::as_str).map(str::to_string),
                prior_sha256: opt_digest(&v, "prior_sha256", id)?,
                workspace: v
                    .get("workspace")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                approved: match v.get(APPROVED_GRANT) {
                    None => None,
                    Some(_) => {
                        let link = ApprovedLink {
                            approval_key: s(&v, "approval_key", id)?,
                            replay_claim: u(&v, "replay_claim", id)?,
                            cx_promotion: u(&v, "cx_promotion", id)?,
                            cx_evidence: u(&v, "cx_evidence", id)?,
                        };
                        self.approved_by_claim
                            .entry(link.replay_claim)
                            .or_default()
                            .push(id);
                        Some(ApprovedGrant {
                            identity: crate::approved_auth::ApprovalIdentity {
                                trace_id: s(&v, "trace_id", id)?,
                                request_id: s(&v, "request_id", id)?,
                                approval_id: s(&v, "approval_id", id)?,
                                approver: s(&v, "approver", id)?,
                                path: path.to_string(),
                                content_sha256: c.to_string(),
                                approved_proposal_sha256: s(&v, "approved_proposal_sha256", id)?,
                                desk_key_id: s(&v, "desk_key_id", id)?,
                                workspace: s(&v, "workspace", id)?,
                            },
                            link,
                        })
                    }
                },
                links: links.to_vec(),
            },
        );
        Ok(())
    }

    fn read_effect(&mut self, id: u64, text: &str) -> Result<(), Refusal> {
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return Ok(());
        };
        let Some(phase) = v.get("phase") else {
            // NEXT-PHASE-1 effect notes: a write_file spends its authorization.
            if v.get("tool").and_then(Value::as_str) == Some("write_file") {
                if let Some(a) = v.get("authorization").and_then(Value::as_u64) {
                    self.legacy_spent.entry(a).or_insert(id);
                }
            }
            return Ok(());
        };
        match phase.as_str() {
            Some(PHASE_INTENT) => {
                let a = u(&v, "authorization", id)?;
                if let Some(prev) = self.spent.get(&a) {
                    return Err(corrupt(
                        id,
                        format!("second intent for authorization #{a} (first #{prev})"),
                    ));
                }
                let ex = v.get("executor").cloned().unwrap_or(Value::Null);
                let row = IntentRow {
                    id,
                    authorization: a,
                    path: s(&v, "path", id)?,
                    target: s(&v, "target", id)?,
                    content_sha256: s(&v, "content_sha256", id)?,
                    prior_sha256: opt_digest(&v, "prior_sha256", id)?
                        .ok_or_else(|| corrupt(id, "intent without prior_sha256"))?,
                    executor_pid: u(&ex, "pid", id)? as u32,
                    executor_start: u(&ex, "start", id)?,
                    state: EffectState::Open,
                    state_record: 0,
                    unresolved_digest: None,
                };
                self.spent.insert(a, id);
                self.intents.insert(id, row);
            }
            Some(PHASE_ACK) | Some(PHASE_RECONCILE) => {
                let i = u(&v, "intent", id)?;
                let st = EffectState::parse(&s(&v, "state", id)?)
                    .ok_or_else(|| corrupt(id, "unknown effect state"))?;
                let disk = opt_digest(&v, "disk_sha256", id)?.unwrap_or(None);
                let row = self
                    .intents
                    .get_mut(&i)
                    .ok_or_else(|| corrupt(id, format!("names intent #{i}, which is not one")))?;
                if row.state.terminal() {
                    return Err(corrupt(
                        id,
                        format!(
                            "intent #{i} already {} by #{}",
                            row.state.name(),
                            row.state_record
                        ),
                    ));
                }
                row.state = st;
                row.state_record = id;
                if st == EffectState::Unresolved {
                    row.unresolved_digest = Some(disk);
                }
            }
            _ => return Err(corrupt(id, "unknown effect phase")),
        }
        Ok(())
    }

    /// True while the newest stop/resume record is a stop.
    pub fn stopped(&self) -> Option<u64> {
        let stop = self.stops.last().copied()?;
        match self.resumes.last() {
            Some(&r) if r > stop => None,
            _ => Some(stop),
        }
    }

    /// The effect-boundary check (ACCEPTANCE-v2 2.3). `current` is the
    /// target's digest now (None = absent).
    pub fn check_intent(
        &self,
        req: &IntentRequest,
        current: &Option<String>,
    ) -> Result<&Grant, Refusal> {
        let a = req.authorization;
        if let Some(stop) = self.stopped() {
            return Err(Refusal::new(
                "Stopped",
                format!("operator stop #{stop} is in force; resume needs an authorized operator"),
            ));
        }
        let g = self.grants.get(&a).ok_or_else(|| {
            Refusal::new(
                "NotAuthorized",
                format!("cortex.cx#{a} is not a verified authorization"),
            )
        })?;
        if g.proposal_sha256 != req.proposal_sha256
            || g.path != req.path
            || g.content_sha256 != req.content_sha256
        {
            return Err(Refusal::new(
                "NotAuthorized",
                format!("authorization #{a} names a different proposal"),
            ));
        }
        if let Some(&i) = self.spent.get(&a) {
            let row = &self.intents[&i];
            return Err(if row.state.terminal() {
                Refusal::new(
                    "AlreadySpent",
                    format!(
                        "authorization #{a} was spent by intent #{i} ({}); a new effect needs a new authorization",
                        row.state.name()
                    ),
                )
            } else {
                Refusal::new(
                    "ReconciliationRequired",
                    format!(
                        "intent #{i} for authorization #{a} is {}; run aien compose reconcile",
                        row.state.name()
                    ),
                )
            });
        }
        if let Some(&e) = self.legacy_spent.get(&a) {
            return Err(Refusal::new(
                "AlreadySpent",
                format!("authorization #{a} was spent by effect #{e} (NEXT-PHASE-1 record)"),
            ));
        }
        if let Some(&r) = self.revoked.get(&a) {
            return Err(Refusal::new(
                "Revoked",
                format!("authorization #{a} revoked by #{r}"),
            ));
        }
        if let Some(&stop) = self.stops.iter().rev().find(|&&x| x > a) {
            return Err(Refusal::new(
                "Stale",
                format!("authorization #{a} predates operator stop #{stop}"),
            ));
        }
        let (Some(target), Some(prior)) = (&g.target, &g.prior_sha256) else {
            return Err(Refusal::new(
                "Stale",
                format!("authorization #{a} names no target state (written before effect intents)"),
            ));
        };
        if target != &req.target {
            return Err(Refusal::new(
                "NotAuthorized",
                format!(
                    "authorization #{a} names target {target}, not {}",
                    req.target
                ),
            ));
        }
        if prior != current {
            return Err(Refusal::new(
                "Stale",
                format!(
                    "target changed since authorization #{a} (authorized against {}, now {})",
                    prior.as_deref().unwrap_or("absent"),
                    current.as_deref().unwrap_or("absent")
                ),
            ));
        }
        Ok(g)
    }

    /// Intents without a terminal state, oldest first.
    pub fn unsettled(&self) -> impl Iterator<Item = &IntentRow> {
        self.intents.values().filter(|r| !r.state.terminal())
    }

    /// One JSON row per intent, for `aien compose effects`.
    pub fn view(&self) -> Value {
        json!({
            "stopped": self.stopped(),
            "stops": self.stops, "resumes": self.resumes,
            "revoked": self.revoked.iter().map(|(a, r)| json!({"authorization": a, "record": r})).collect::<Vec<_>>(),
            "legacy_spent": self.legacy_spent.iter().map(|(a, e)| json!({"authorization": a, "effect": e})).collect::<Vec<_>>(),
            "intents": self.intents.values().map(|r| json!({
                "intent": r.id, "authorization": r.authorization, "path": r.path,
                "target": r.target, "content_sha256": r.content_sha256,
                "prior_sha256": r.prior_sha256, "state": r.state.name(),
                "state_record": r.state_record,
                "executor": {"pid": r.executor_pid, "start": r.executor_start},
            })).collect::<Vec<_>>(),
        })
    }
}

/// sha256 of a file's bytes, None when it does not exist. Anything else
/// that is not a readable regular file is an error.
pub fn file_sha256(path: &Path) -> Result<Option<String>, String> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("stat {}: {e}", path.display())),
        Ok(m) if !m.file_type().is_file() => {
            return Err(format!("{} is not a regular file", path.display()))
        }
        Ok(_) => {}
    }
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(Some(hex(&Sha256::digest(&b))))
}

/// The world check of ACCEPTANCE-v2 2.2. Returns the state and the digest seen.
pub fn world_state(row: &IntentRow) -> (EffectState, Result<Option<String>, String>) {
    let now = file_sha256(Path::new(&row.target));
    let st = match &now {
        Ok(Some(d)) if *d == row.content_sha256 => EffectState::Done,
        Ok(d) if *d == row.prior_sha256 => EffectState::NotDone,
        _ => EffectState::Unresolved,
    };
    (st, now)
}

/// Start time of a process (clock ticks since boot, /proc/<pid>/stat field 22).
pub fn process_start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 2 (comm) may hold spaces and parentheses: fields after the last ')'.
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

/// This process as an executor: (pid, start ticks).
pub fn self_executor() -> (u32, u64) {
    let pid = std::process::id();
    (pid, process_start_ticks(pid).unwrap_or(0))
}

/// Is the executor that opened an intent still running?
pub fn executor_alive(pid: u32, start: u64) -> bool {
    pid != 0 && start != 0 && process_start_ticks(pid) == Some(start)
}

/// Effect commands refuse while the start-up reconcile has not succeeded
/// (ACCEPTANCE-v3 2.5): `EFFECT_REFUSED ReconcileFailed`.
fn reconcile_gate(b: &ComposeBridge) -> Result<(), String> {
    match b.reconcile_failed() {
        None => Ok(()),
        Some(why) => Err(Refusal::new(
            "ReconcileFailed",
            format!("the start-up reconcile did not complete ({why}); run aien compose reconcile"),
        )
        .to_string()),
    }
}

/// ComposeNote may not forge gated records (ACCEPTANCE-v2 2.7).
pub fn check_reserved_note(kind: &str, text: &str) -> Result<(), String> {
    let Ok(v) = serde_json::from_str::<Value>(text) else {
        return Ok(());
    };
    match kind {
        "effect" if v.get("phase").is_some() => Err(
            "ComposeNote: effect records with a \"phase\" are written only by the effect commands"
                .into(),
        ),
        // sovereign-core #249: approved-submission replay records come only from
        // crate::approved_replay.
        "effect" if v.get(crate::approved_replay::FIELD).is_some() => Err(
            "ComposeNote: approved_submission records are written only by the replay ledger".into(),
        ),
        "authorization" if v.get("control").is_some() => {
            Err("ComposeNote: control records are written only by ComposeControl".into())
        }
        // sovereign-core #249: the approved grant comes only from the daemon,
        // after a COMMITTED approved compose.
        "authorization" if v.get(APPROVED_GRANT).is_some() => Err(
            "ComposeNote: approved_grant records are written only by ComposeApprovedProposal"
                .into(),
        ),
        // sovereign-core #249: a grant names its workspace and a target inside it.
        "authorization" if v.get("proposal_sha256").is_some() => {
            let (Some(ws), Some(path), Some(target)) = (
                v.get("workspace").and_then(Value::as_str),
                v.get("path").and_then(Value::as_str),
                v.get("target").and_then(Value::as_str),
            ) else {
                return Err(
                    "ComposeNote: an authorization grant must name its workspace, path and target"
                        .into(),
                );
            };
            confine_target(ws, path, target)
                .map_err(|r| format!("ComposeNote: authorization grant refused: {r}"))
        }
        // ACCEPTANCE-v3 2.4: repair records come only from RecoverComposeHome.
        "constraint" if v.get("repair").is_some() => Err(
            "ComposeNote: constraint records with a \"repair\" field are written only by RecoverComposeHome"
                .into(),
        ),
        _ => Ok(()),
    }
}

/// Workspace confinement of an effect target (sovereign-core #249): the
/// workspace is an absolute, canonical directory that is not `/`; `path` is
/// relative with plain components only; `target` is exactly
/// `workspace/path`; its parent resolves (symlinks followed) to
/// `workspace/<parent of path>`; and the target itself, when present, is a
/// regular file, never a symlink.
pub fn confine_target(workspace: &str, path: &str, target: &str) -> Result<(), Refusal> {
    use std::path::Component;
    let out = |w: String| Refusal::new("OutsideWorkspace", w);
    let ws = Path::new(workspace);
    let canon =
        std::fs::canonicalize(ws).map_err(|e| out(format!("workspace {workspace}: {e}")))?;
    if !ws.is_absolute() || canon != ws || !canon.is_dir() || canon == Path::new("/") {
        return Err(out(format!(
            "workspace {workspace} is not an absolute canonical directory other than /"
        )));
    }
    let rel = Path::new(path);
    if path.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(out(format!("path {path:?} is not a plain relative path")));
    }
    let want = ws.join(rel);
    if Path::new(target) != want {
        return Err(out(format!(
            "target {target} is not {} (workspace {workspace} + path {path})",
            want.display()
        )));
    }
    let parent = want.parent().unwrap_or(ws);
    let real = std::fs::canonicalize(parent)
        .map_err(|e| out(format!("target directory {}: {e}", parent.display())))?;
    if real != parent || !real.starts_with(&canon) {
        return Err(out(format!(
            "target directory {} resolves to {} (outside or through a symlink)",
            parent.display(),
            real.display()
        )));
    }
    match std::fs::symlink_metadata(&want) {
        Ok(m) if !m.file_type().is_file() => Err(out(format!(
            "target {target} exists and is not a regular file (symlink or other)"
        ))),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(out(format!("target {target}: {e}"))),
    }
}

/// An approved grant must be backed by its COMMITTED replay claim: same
/// approval key, commit evidence naming the grant's proposal, promotion and
/// evidence, and the record linking all three.
fn check_approved_backing(
    l: &Ledger,
    g: &Grant,
    host: &[ComposeRecordView],
) -> Result<(), Refusal> {
    let Some(ag) = &g.approved else {
        return Ok(());
    };
    let a = &ag.link;
    let no = |w: &str| {
        Refusal::new(
            "NotAuthorized",
            format!(
                "approved grant #{} is not backed by a committed approved compose: {w}",
                g.id
            ),
        )
    };
    // One approval, one grant: a copy of the grant (same claim) opens nothing.
    if l.approved_by_claim.get(&a.replay_claim).map(Vec::len) != Some(1) {
        return Err(no("more than one approved grant names this replay claim"));
    }
    // The grant's fields are the bound approval: its identity hashes to the
    // claim's approval key (path, content, workspace and ids all covered), and
    // the target is the bound workspace plus path.
    let id = &ag.identity;
    let bound_target = Path::new(&id.workspace).join(&id.path);
    if crate::approved_auth::approval_key(id) != a.approval_key
        || g.target.as_deref().map(Path::new) != Some(bound_target.as_path())
        || g.workspace.as_deref() != Some(id.workspace.as_str())
    {
        return Err(no("grant fields are not the bound approval"));
    }
    let rl =
        crate::approved_replay::ReplayLedger::from_records(host).map_err(|r| no(&r.to_string()))?;
    let row = rl
        .claims
        .get(&a.replay_claim)
        .ok_or_else(|| no("no such replay claim"))?;
    let ev = row.evidence.as_ref();
    if row.state != crate::approved_replay::ClaimState::Committed
        || row.keys.approval_key != a.approval_key
        || (&row.keys.request_id, &row.keys.approval_id, &row.keys.trace_id)
            != (&id.request_id, &id.approval_id, &id.trace_id)
        || ev.map(|e| {
            (
                e.compose_proposal_sha256.as_str(),
                e.cx_promotion,
                e.cx_evidence,
            )
        }) != Some((g.proposal_sha256.as_str(), a.cx_promotion, a.cx_evidence))
        || ![a.cx_promotion, a.cx_evidence, a.replay_claim]
            .iter()
            .all(|x| g.links.contains(x))
    {
        return Err(no("claim state, key, evidence or links differ"));
    }
    Ok(())
}

/// The grant the daemon writes after a COMMITTED approved compose (#249):
/// reserved (`approved_grant`), confined to `workspace`, linked to the
/// promotion, evidence and replay claim. Returns (record id, target).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_approved_grant(
    b: &ComposeBridge,
    workspace: &str,
    path: &str,
    content_sha256: &str,
    approver: &str,
    ids: &Value,
    link: &ApprovedLink,
    proposal_sha256: &str,
) -> Result<(u64, String), String> {
    b.with_home(|home| {
        let target = Path::new(workspace).join(path).display().to_string();
        confine_target(workspace, path, &target).map_err(|r| r.to_string())?;
        let prior = file_sha256(Path::new(&target))?;
        let mut text = json!({
            APPROVED_GRANT: 1, "proposal_sha256": proposal_sha256, "path": path,
            "content_sha256": content_sha256, "approver": approver, "target": target,
            "workspace": workspace, "prior_sha256": prior,
            "approval_key": link.approval_key, "replay_claim": link.replay_claim,
            "cx_promotion": link.cx_promotion, "cx_evidence": link.cx_evidence,
        });
        if let (Some(t), Some(i)) = (text.as_object_mut(), ids.as_object()) {
            for (k, v) in i {
                t.entry(k.clone()).or_insert(v.clone());
            }
        }
        let n = append(
            home,
            NoteKind::Authorization,
            &[link.cx_promotion, link.cx_evidence, link.replay_claim],
            &text,
        )?;
        Ok((n.id, target))
    })
}

fn host_views(home: &mut ComposeHome) -> Result<Vec<ComposeRecordView>, String> {
    let (recs, total) = home
        .compose
        .recall(aien_omega_compose::SUBJECT_HOST, HOST_RECALL_MAX)
        .map_err(|e| format!("effect ledger: host records: {e}"))?;
    if total > recs.len() as u64 {
        return Err(format!(
            "{}",
            Refusal::new(
                "CorruptLedger",
                format!("{total} host records exceed the ledger's {HOST_RECALL_MAX}")
            )
        ));
    }
    Ok(recs
        .iter()
        .map(|r| record_view(&mut home.compose, r))
        .collect())
}

fn ledger(home: &mut ComposeHome) -> Result<Ledger, String> {
    Ledger::from_records(&host_views(home)?).map_err(|r| r.to_string())
}

fn append(
    home: &mut ComposeHome,
    kind: NoteKind,
    links: &[u64],
    text: &Value,
) -> Result<ComposeNoteReport, String> {
    let text = text.to_string();
    let mut l = [0u64; 4];
    for (slot, &x) in l.iter_mut().zip(links.iter().filter(|&&x| x != 0)) {
        *slot = x;
    }
    let id = home
        .compose
        .note(kind, l, text.as_bytes())
        .map_err(|e| format!("effect record: {e}"))?;
    let rec = home
        .compose
        .record(id)
        .map_err(|e| format!("effect record: re-read {id}: {e}"))?;
    Ok(ComposeNoteReport {
        machine_id: home.machine_id.clone(),
        id,
        kind: kind.name().to_string(),
        digest: hex(&rec.digest),
        text_sha256: hex(&Sha256::digest(text.as_bytes())),
        links: l.iter().copied().filter(|&x| x != 0).collect(),
    })
}

fn noted(r: Result<ComposeNoteReport, String>) -> ControlResponse {
    match r {
        Ok(n) => ControlResponse::ComposeNoted(n),
        Err(e) => ControlResponse::Error(e),
    }
}

/// ComposeEffectIntent: every check of 2.3 and the durable intent, in one
/// step under the home lock. The answer's id is the intent.
pub fn open_intent(b: &ComposeBridge, req: &IntentRequest) -> ControlResponse {
    if let Err(e) = reconcile_gate(b) {
        return ControlResponse::Error(e);
    }
    noted(b.with_home(|home| {
        let views = host_views(home)?;
        let l = Ledger::from_records(&views).map_err(|r| r.to_string())?;
        let current = file_sha256(Path::new(&req.target))
            .map_err(|e| Refusal::new("Stale", format!("target unreadable: {e}")).to_string())?;
        let g = l.check_intent(req, &current).map_err(|r| r.to_string())?;
        // sovereign-core #249: confinement, and an approved grant's backing.
        let ws = g.workspace.as_deref().ok_or_else(|| {
            Refusal::new(
                "NotAuthorized",
                format!("authorization #{} names no workspace", g.id),
            )
            .to_string()
        })?;
        confine_target(ws, &req.path, &req.target).map_err(|r| r.to_string())?;
        check_approved_backing(&l, g, &views).map_err(|r| r.to_string())?;
        let text = json!({
            "phase": PHASE_INTENT, "tool": "write_file", "authorization": g.id,
            "proposal_sha256": req.proposal_sha256, "path": req.path, "target": req.target,
            "content_sha256": req.content_sha256, "prior_sha256": current,
            "executor": {"pid": req.executor_pid, "start": req.executor_start},
        });
        append(home, NoteKind::Effect, &[g.id], &text)
    }))
}

/// ComposeEffectAck: the executor's report after the write. The state is
/// read from the world; the executor's own claim is kept beside it.
pub fn ack(b: &ComposeBridge, intent: u64, reported: &Value) -> ControlResponse {
    if let Err(e) = reconcile_gate(b) {
        return ControlResponse::Error(e);
    }
    noted(b.with_home(|home| {
        let l = ledger(home)?;
        let row = l
            .intents
            .get(&intent)
            .ok_or_else(|| format!("ComposeEffectAck: #{intent} is not an effect intent"))?;
        if row.state.terminal() {
            return Err(format!(
                "ComposeEffectAck: intent #{intent} is already {} (record #{})",
                row.state.name(),
                row.state_record
            ));
        }
        let (st, disk) = world_state(row);
        let text = json!({
            "phase": PHASE_ACK, "intent": intent, "authorization": row.authorization,
            "tool": "write_file", "path": row.path, "content_sha256": row.content_sha256,
            "state": st.name(), "disk_sha256": disk.as_ref().ok().cloned().flatten(),
            "disk_error": disk.as_ref().err(), "executor_reported": reported,
        });
        append(home, NoteKind::Effect, &[intent, row.authorization], &text)
    }))
}

/// Reconcile every unsettled intent whose executor is gone, or record one
/// operator declaration. `by` names who decided ("reconcile@start", ...).
pub fn reconcile(
    b: &ComposeBridge,
    declare: Option<&ReconcileDeclare>,
    by: &str,
) -> ControlResponse {
    let r = b.with_home(|home| {
        let l = ledger(home)?;
        let mut out = Vec::new();
        let mut checked = 0u64;
        let rows: Vec<IntentRow> = match declare {
            Some(d) => {
                let row = l
                    .intents
                    .get(&d.intent)
                    .ok_or_else(|| format!("reconcile: #{} is not an effect intent", d.intent))?;
                if row.state.terminal() {
                    return Err(format!(
                        "reconcile: intent #{} is already {}",
                        d.intent,
                        row.state.name()
                    ));
                }
                vec![row.clone()]
            }
            None => l.unsettled().cloned().collect(),
        };
        for row in rows {
            checked += 1;
            if executor_alive(row.executor_pid, row.executor_start) {
                out.push(ReconcileOutcome {
                    intent: row.id,
                    authorization: row.authorization,
                    state: row.state.name().into(),
                    disk_sha256: None,
                    record: None,
                    note: format!(
                        "executor pid {} still running; not decided",
                        row.executor_pid
                    ),
                });
                continue;
            }
            let (world, disk) = world_state(&row);
            let disk_hex = disk.as_ref().ok().cloned().flatten();
            let (st, who) = match declare {
                Some(d) => {
                    let st = match d.state.as_str() {
                        "done" | "DONE" => EffectState::Done,
                        "not_done" | "NOT_DONE" => EffectState::NotDone,
                        other => {
                            return Err(format!("reconcile: --declare {other:?} (done, not_done)"))
                        }
                    };
                    if d.approver.trim().is_empty() {
                        return Err("reconcile: a declaration needs --approver".into());
                    }
                    (st, format!("operator:{}", d.approver.trim()))
                }
                None => (world, by.to_string()),
            };
            if st == EffectState::Unresolved && row.unresolved_digest.as_ref() == Some(&disk_hex) {
                out.push(ReconcileOutcome {
                    intent: row.id,
                    authorization: row.authorization,
                    state: st.name().into(),
                    disk_sha256: disk_hex,
                    record: None,
                    note: "still UNRESOLVED, already recorded for this target digest".into(),
                });
                continue;
            }
            let text = json!({
                "phase": PHASE_RECONCILE, "intent": row.id, "authorization": row.authorization,
                "state": st.name(), "world": world.name(), "disk_sha256": disk_hex,
                "disk_error": disk.as_ref().err(), "by": who,
            });
            let n = append(home, NoteKind::Effect, &[row.id, row.authorization], &text)?;
            out.push(ReconcileOutcome {
                intent: row.id,
                authorization: row.authorization,
                state: st.name().into(),
                disk_sha256: disk_hex,
                record: Some(n.id),
                note: who,
            });
        }
        Ok(ComposeReconcileReport {
            compose_dir: b.dir().display().to_string(),
            machine_id: home.machine_id.clone(),
            checked,
            outcomes: out,
        })
    });
    match r {
        Ok(r) => {
            // A full reconcile that completed lifts the refusal of 2.5; a
            // single declaration does not.
            if declare.is_none() {
                b.clear_reconcile_failed();
            }
            ControlResponse::ComposeReconciled(Box::new(r))
        }
        Err(e) => ControlResponse::Error(e),
    }
}

/// ComposeControl: stop, resume, revoke (ACCEPTANCE-v2 2.5, 2.6).
pub fn control(
    b: &ComposeBridge,
    action: &str,
    approver: &str,
    authorization: Option<u64>,
) -> ControlResponse {
    let r = b.with_home(|home| {
        if approver.trim().is_empty() {
            return Err(format!("ComposeControl {action}: --approver is required"));
        }
        let l = ledger(home)?;
        let (text, links, revoked) = match action {
            "stop" | "resume" => (
                json!({"control": action, "approver": approver.trim()}),
                vec![],
                None,
            ),
            "revoke" => {
                let a =
                    authorization.ok_or("ComposeControl revoke: --authorization is required")?;
                if !l.grants.contains_key(&a) {
                    return Err(format!(
                        "ComposeControl revoke: #{a} is not an authorization"
                    ));
                }
                if l.spent.contains_key(&a)
                    || l.legacy_spent.contains_key(&a)
                    || l.revoked.contains_key(&a)
                {
                    // Spent (or already revoked): nothing to revoke, nothing recorded.
                    return Ok(ComposeControlReport {
                        action: action.into(),
                        recorded: None,
                        revoked: Some(false),
                    });
                }
                (
                    json!({"control": "revoke", "authorization": a, "approver": approver.trim()}),
                    vec![a],
                    Some(true),
                )
            }
            other => {
                return Err(format!(
                    "ComposeControl: action {other:?} (stop, resume, revoke)"
                ))
            }
        };
        let n = append(home, NoteKind::Authorization, &links, &text)?;
        Ok(ComposeControlReport {
            action: action.into(),
            recorded: Some(n),
            revoked,
        })
    });
    match r {
        Ok(r) => ControlResponse::ComposeControlled(Box::new(r)),
        Err(e) => ControlResponse::Error(e),
    }
}

/// The tail of every start line after which effect commands refuse.
pub const RECONCILE_GATE_NOTE: &str =
    "effect commands refuse until a successful reconcile (aien compose reconcile)";

/// Daemon start: reconcile an existing home once; one line for the log.
pub fn reconcile_at_start(b: &ComposeBridge) -> String {
    // Test builds only (ACCEPTANCE-v3 2.6): a panic inside the spawn_blocking
    // task. Release builds abort on it (ACCEPTANCE-v4 row C7c).
    #[cfg(feature = "fault-hold")]
    if std::env::var("AIEN_FAULT_HOLD").as_deref() == Ok("reconcile_panic") {
        panic!("fault hold reconcile_panic: forced start-up reconcile failure (test build)");
    }
    if !b.dir().join("cortex.cx").exists() {
        return "Reconcile: no compose home yet".into();
    }
    // Test builds only (ACCEPTANCE-v4 2.1): force the error arm below, the
    // path a release build takes (it aborts on panic, Cargo.toml).
    #[cfg(feature = "fault-hold")]
    let result = if std::env::var("AIEN_FAULT_HOLD").as_deref() == Ok("reconcile_error") {
        ControlResponse::Error(
            "fault hold reconcile_error: forced start-up reconcile error (test build)".into(),
        )
    } else {
        reconcile(b, None, "reconcile@start")
    };
    #[cfg(not(feature = "fault-hold"))]
    let result = reconcile(b, None, "reconcile@start");
    match result {
        ControlResponse::ComposeReconciled(r) => {
            let n = |st: &str| {
                r.outcomes
                    .iter()
                    .filter(|o| o.state == st && o.record.is_some())
                    .count()
            };
            let pending = r.outcomes.iter().filter(|o| o.record.is_none()).count();
            format!(
                "Reconcile: checked {} unsettled effect(s): DONE {}, NOT_DONE {}, UNRESOLVED {} recorded, {} left as they were; {}",
                r.checked,
                n("DONE"),
                n("NOT_DONE"),
                n("UNRESOLVED"),
                pending,
                serde_json::to_string(&r.outcomes).unwrap_or_default()
            )
        }
        ControlResponse::Error(e) => {
            b.set_reconcile_failed(format!("refused: {e}"));
            format!("Reconcile: refused: {e}; {RECONCILE_GATE_NOTE}")
        }
        other => {
            b.set_reconcile_failed(format!("unexpected {other:?}"));
            format!("Reconcile: unexpected {other:?}; {RECONCILE_GATE_NOTE}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64, note: &str, text: Value) -> ComposeRecordView {
        ComposeRecordView {
            id,
            cls: 0,
            kind: 0,
            subject: 0,
            tag: 0,
            links: vec![],
            digest: String::new(),
            verified: true,
            note: Some(note.into()),
            text: Some(text.to_string()),
        }
    }

    fn grant(id: u64, prior: Value) -> ComposeRecordView {
        rec(
            id,
            "authorization",
            json!({"proposal_sha256": "p", "path": "N.md", "content_sha256": "c",
                   "approver": "drake", "target": "/w/N.md", "prior_sha256": prior}),
        )
    }

    fn req(a: u64) -> IntentRequest {
        IntentRequest {
            authorization: a,
            proposal_sha256: "p".into(),
            path: "N.md".into(),
            target: "/w/N.md".into(),
            content_sha256: "c".into(),
            executor_pid: 1,
            executor_start: 1,
        }
    }

    fn intent(id: u64, a: u64) -> ComposeRecordView {
        rec(
            id,
            "effect",
            json!({"phase": "intent", "tool": "write_file", "authorization": a, "path": "N.md",
                   "target": "/w/N.md", "content_sha256": "c", "prior_sha256": null,
                   "proposal_sha256": "p", "executor": {"pid": 1, "start": 1}}),
        )
    }

    fn settle(id: u64, phase: &str, i: u64, st: &str) -> ComposeRecordView {
        rec(
            id,
            "effect",
            json!({"phase": phase, "intent": i, "state": st, "disk_sha256": null}),
        )
    }

    fn refusal(l: &Ledger, a: u64, cur: Option<&str>) -> &'static str {
        l.check_intent(&req(a), &cur.map(str::to_string))
            .unwrap_err()
            .name
    }

    #[test]
    fn fresh_grant_opens_once() {
        let l = Ledger::from_records(&[grant(2, Value::Null)]).unwrap();
        assert_eq!(l.check_intent(&req(2), &None).unwrap().id, 2);
        let l = Ledger::from_records(&[grant(2, Value::Null), intent(3, 2)]).unwrap();
        assert_eq!(refusal(&l, 2, None), "ReconciliationRequired");
    }

    #[test]
    fn terminal_states_spend_the_grant() {
        for st in ["DONE", "NOT_DONE"] {
            let l = Ledger::from_records(&[
                grant(2, Value::Null),
                intent(3, 2),
                settle(4, "ack", 3, st),
            ])
            .unwrap();
            assert_eq!(refusal(&l, 2, None), "AlreadySpent", "{st}");
        }
        let l = Ledger::from_records(&[
            grant(2, Value::Null),
            intent(3, 2),
            settle(4, "reconcile", 3, "UNRESOLVED"),
        ])
        .unwrap();
        assert_eq!(refusal(&l, 2, None), "ReconciliationRequired");
        assert_eq!(l.intents[&3].unresolved_digest, Some(None));
    }

    #[test]
    fn stop_resume_stale_and_revoke() {
        let stop = rec(
            3,
            "authorization",
            json!({"control": "stop", "approver": "drake"}),
        );
        let resume = rec(
            4,
            "authorization",
            json!({"control": "resume", "approver": "drake"}),
        );
        let l = Ledger::from_records(&[grant(2, Value::Null), stop.clone()]).unwrap();
        assert_eq!(refusal(&l, 2, None), "Stopped");
        let l =
            Ledger::from_records(&[grant(2, Value::Null), stop.clone(), resume.clone()]).unwrap();
        assert_eq!(refusal(&l, 2, None), "Stale");
        let l = Ledger::from_records(&[grant(2, Value::Null), stop, resume, grant(5, Value::Null)])
            .unwrap();
        assert!(l.check_intent(&req(5), &None).is_ok());
        let revoke = rec(
            3,
            "authorization",
            json!({"control": "revoke", "authorization": 2, "approver": "d"}),
        );
        let l = Ledger::from_records(&[grant(2, Value::Null), revoke]).unwrap();
        assert_eq!(refusal(&l, 2, None), "Revoked");
    }

    #[test]
    fn world_change_and_legacy_grants_are_stale() {
        let l = Ledger::from_records(&[grant(2, Value::Null)]).unwrap();
        assert_eq!(refusal(&l, 2, Some("x")), "Stale");
        let legacy = rec(
            2,
            "authorization",
            json!({"proposal_sha256": "p", "path": "N.md", "content_sha256": "c", "approver": "d"}),
        );
        let l = Ledger::from_records(&[legacy]).unwrap();
        assert_eq!(refusal(&l, 2, None), "Stale");
        let old = rec(
            3,
            "effect",
            json!({"tool": "write_file", "authorization": 2, "success": true}),
        );
        let l = Ledger::from_records(&[grant(2, Value::Null), old]).unwrap();
        assert_eq!(refusal(&l, 2, None), "AlreadySpent");
    }

    #[test]
    fn other_proposal_or_target_is_not_authorized() {
        let l = Ledger::from_records(&[grant(2, Value::Null)]).unwrap();
        let mut r = req(2);
        r.content_sha256 = "other".into();
        assert_eq!(l.check_intent(&r, &None).unwrap_err().name, "NotAuthorized");
        let mut r = req(2);
        r.target = "/elsewhere".into();
        assert_eq!(l.check_intent(&r, &None).unwrap_err().name, "NotAuthorized");
        assert_eq!(refusal(&l, 9, None), "NotAuthorized");
    }

    #[test]
    fn damaged_or_inconsistent_ledger_is_refused() {
        let mut bad = grant(2, Value::Null);
        bad.verified = false;
        assert_eq!(
            Ledger::from_records(&[bad]).unwrap_err().name,
            "CorruptLedger"
        );
        let two = [grant(2, Value::Null), intent(3, 2), intent(4, 2)];
        assert_eq!(
            Ledger::from_records(&two).unwrap_err().name,
            "CorruptLedger"
        );
        let after = [
            grant(2, Value::Null),
            intent(3, 2),
            settle(4, "ack", 3, "DONE"),
            settle(5, "reconcile", 3, "NOT_DONE"),
        ];
        assert_eq!(
            Ledger::from_records(&after).unwrap_err().name,
            "CorruptLedger"
        );
        let orphan = [settle(4, "ack", 3, "DONE")];
        assert_eq!(
            Ledger::from_records(&orphan).unwrap_err().name,
            "CorruptLedger"
        );
    }

    #[test]
    fn world_state_reads_the_target() {
        let d = tempfile::tempdir().unwrap();
        let t = d.path().join("N.md");
        let c = hex(&Sha256::digest(b"new"));
        let mut row = IntentRow {
            id: 3,
            authorization: 2,
            path: "N.md".into(),
            target: t.display().to_string(),
            content_sha256: c.clone(),
            prior_sha256: None,
            executor_pid: 0,
            executor_start: 0,
            state: EffectState::Open,
            state_record: 0,
            unresolved_digest: None,
        };
        assert_eq!(world_state(&row).0, EffectState::NotDone);
        std::fs::write(&t, b"new").unwrap();
        assert_eq!(world_state(&row).0, EffectState::Done);
        std::fs::write(&t, b"other").unwrap();
        assert_eq!(world_state(&row).0, EffectState::Unresolved);
        row.prior_sha256 = Some(hex(&Sha256::digest(b"other")));
        assert_eq!(world_state(&row).0, EffectState::NotDone);
        std::fs::remove_file(&t).unwrap();
        std::fs::create_dir(&t).unwrap();
        assert_eq!(world_state(&row).0, EffectState::Unresolved);
    }

    #[test]
    fn executor_liveness_uses_pid_and_start_time() {
        let (pid, start) = self_executor();
        assert!(start > 0);
        assert!(executor_alive(pid, start));
        assert!(!executor_alive(pid, start + 1));
        assert!(!executor_alive(0, 0));
    }

    #[test]
    fn reserved_records_cannot_be_forged_through_compose_note() {
        assert!(check_reserved_note("effect", r#"{"phase":"ack","intent":3}"#).is_err());
        assert!(check_reserved_note("authorization", r#"{"control":"resume"}"#).is_err());
        assert!(check_reserved_note("effect", r#"{"tool":"inspect"}"#).is_ok());
        assert!(check_reserved_note("constraint", "plain text").is_ok());
    }

    #[test]
    fn grants_are_confined_and_the_approved_kind_is_reserved() {
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let ws = root.join("ws");
        std::fs::create_dir_all(ws.join("d")).unwrap();
        let w = ws.to_str().unwrap();
        let tg = |p: &str| ws.join(p).display().to_string();
        assert!(confine_target(w, "N.md", &tg("N.md")).is_ok());
        assert!(confine_target(w, "d/N.md", &tg("d/N.md")).is_ok());
        let out = |ws: &str, p: &str, t: &str| confine_target(ws, p, t).unwrap_err().name;
        assert_eq!(out(w, "N.md", "/etc/N.md"), "OutsideWorkspace");
        assert_eq!(out(w, "../N.md", &tg("../N.md")), "OutsideWorkspace");
        assert_eq!(out(w, "/etc/passwd", "/etc/passwd"), "OutsideWorkspace");
        assert_eq!(out(w, "./N.md", &tg("./N.md")), "OutsideWorkspace");
        assert_eq!(out(w, "", w), "OutsideWorkspace");
        assert_eq!(out("/", "etc/passwd", "/etc/passwd"), "OutsideWorkspace");
        assert_eq!(out("ws", "N.md", "ws/N.md"), "OutsideWorkspace");
        assert_eq!(
            out(&format!("{w}/d/.."), "N.md", &format!("{w}/d/../N.md")),
            "OutsideWorkspace"
        );
        std::os::unix::fs::symlink(&root, ws.join("up")).unwrap();
        assert_eq!(out(w, "up/x", &tg("up/x")), "OutsideWorkspace");
        std::os::unix::fs::symlink("/etc/hostname", ws.join("l")).unwrap();
        assert_eq!(out(w, "l", &tg("l")), "OutsideWorkspace");
        std::os::unix::fs::symlink(&ws, root.join("wl")).unwrap();
        let wl = root.join("wl").display().to_string();
        assert_eq!(out(&wl, "N.md", &format!("{wl}/N.md")), "OutsideWorkspace");

        let g = |extra: Value| {
            let mut v = json!({"proposal_sha256": "p", "path": "N.md", "content_sha256": "c",
                "approver": "x", "target": tg("N.md"), "prior_sha256": null, "workspace": w});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            v.to_string()
        };
        assert!(check_reserved_note("authorization", &g(json!({}))).is_ok());
        assert!(check_reserved_note("authorization", &g(json!({"approved_grant": 1}))).is_err());
        assert!(check_reserved_note("authorization", &g(json!({"target": "/etc/N.md"}))).is_err());
        let mut no_ws: Value = serde_json::from_str(&g(json!({}))).unwrap();
        no_ws.as_object_mut().unwrap().remove("workspace");
        assert!(check_reserved_note("authorization", &no_ws.to_string()).is_err());

        // Approved grants (476ca4 c25): fields bound to the approval key, one per
        // claim, and a COMMITTED claim behind it.
        let rec = |id: u64, links: Vec<u64>, text: String| ComposeRecordView {
            links,
            ..super::tests::rec(id, "authorization", serde_json::from_str(&text).unwrap())
        };
        let ident = crate::approved_auth::ApprovalIdentity {
            trace_id: "t".into(),
            request_id: "r".into(),
            approval_id: "a".into(),
            approver: "x".into(),
            path: "N.md".into(),
            content_sha256: "c".into(),
            approved_proposal_sha256: "s".into(),
            desk_key_id: "k".into(),
            workspace: w.into(),
        };
        let key = crate::approved_auth::approval_key(&ident);
        let approved = |extra: Value| {
            let mut e = json!({"approved_grant": 1, "approval_key": key, "replay_claim": 3,
                "cx_promotion": 1, "cx_evidence": 2, "trace_id": "t", "request_id": "r",
                "approval_id": "a", "approved_proposal_sha256": "s", "desk_key_id": "k"});
            for (k, x) in extra.as_object().unwrap() {
                e[k] = x.clone();
            }
            g(e)
        };
        let why = |recs: &[ComposeRecordView], id: u64| {
            let l = Ledger::from_records(recs).unwrap();
            let r = check_approved_backing(&l, &l.grants[&id], recs).unwrap_err();
            assert_eq!(r.name, "NotAuthorized");
            r.detail
        };
        // Bound fields, no claim behind it.
        let one = rec(5, vec![1, 2, 3], approved(json!({})));
        assert!(why(std::slice::from_ref(&one), 5).contains("no such replay claim"));
        // Path, content, target or ids not the ones the approval key binds.
        for bad in [
            json!({"path": "O.md", "target": tg("O.md")}),
            json!({"content_sha256": "other"}),
            json!({"request_id": "r2"}),
            json!({"approval_key": "0".repeat(64)}),
        ] {
            let r = rec(5, vec![1, 2, 3], approved(bad.clone()));
            assert!(
                why(std::slice::from_ref(&r), 5).contains("not the bound approval"),
                "{bad}"
            );
        }
        // A byte copy of the grant (same claim, same links): neither opens.
        let copy = rec(6, vec![1, 2, 3], approved(json!({})));
        let both = [one.clone(), copy];
        for id in [5, 6] {
            assert!(why(&both, id).contains("more than one approved grant"));
        }
        // A grant without the marker needs no backing; the approved one without
        // its fields is a corrupt ledger, not a generic grant.
        assert!(Ledger::from_records(&[rec(6, vec![], g(json!({"approved_grant": 1})))]).is_err());
    }
}
