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
//! Grants (sovereign-core #261). The ledger honours only grants the daemon wrote
//! itself: `approved_grant` (ComposeApprovedProposal, backed by a COMMITTED
//! replay claim) and `minted_grant` (ComposeAuthorize, backed by a
//! `compose_commit` record the daemon wrote when its own compose run
//! committed the proposal). A caller-written `authorization` note (any record
//! with neither marker, including ones already in a journal from before this
//! rule) is read but never honoured: it opens no intent, and an intent it
//! opened earlier is settled UNRESOLVED by the world check (never DONE, never
//! NOT_DONE by the daemon itself) and an operator may not declare it DONE; an
//! operator `--declare not_done` is still accepted.
//!
//! The ledger is rebuilt from the journal on every call, under the
//! compose-home lock, so the check and the intent append are one step.
use crate::compose::{record_view, ComposeBridge, ComposeHome};
use crate::control::{
    ComposeControlReport, ComposeNoteReport, ComposeReconcileReport, ComposeRecordView,
    ControlResponse, ObjectiveBinding, ReconcileDeclare, ReconcileOutcome,
};
use aien_omega_compose::{hex, NoteKind};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::os::fd::AsRawFd;
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
    /// Set on the grant `ComposeAuthorize` minted (#261).
    pub minted: Option<MintedGrant>,
    /// sovereign-core #408: the objective record the grant was approved for.
    pub objective: Option<GrantObjective>,
    /// The record's links.
    pub links: Vec<u64>,
}

impl Grant {
    /// True for a grant the daemon wrote itself (approved or minted). A
    /// caller-written note, or an old one already in a journal, is not.
    pub fn honoured(&self) -> bool {
        self.approved.is_some() || self.minted.is_some()
    }
}

/// Marker field of the grant the daemon itself writes after a COMMITTED
/// approved compose (sovereign-core #249). Reserved: `ComposeNote` refuses it.
pub const APPROVED_GRANT: &str = "approved_grant";

/// Marker field of the grant `ComposeAuthorize` mints (sovereign-core #261).
/// Reserved: `ComposeNote` refuses every `authorization` note.
/// Prefix of the `disk_error` of an intent whose grant the daemon did not mint.
pub const NOT_DAEMON_MINTED: &str = "NotDaemonMinted";

pub const MINTED_GRANT: &str = "minted_grant";

/// Marker field of the record the daemon writes when its own compose run
/// COMMITTED a one-file proposal (sovereign-core #261). Reserved.
pub const COMPOSE_COMMIT: &str = "compose_commit";

/// sovereign-core #408: the objective a grant is bound to, as minted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantObjective {
    pub id: String,
    /// sha256 of the objective text the record held when the grant was minted.
    pub text_sha256: String,
    /// sha256 of the record file's bytes then.
    pub record_sha256: String,
}

/// What a minted grant is backed by: the daemon's own compose-commit record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintedGrant {
    pub commit: u64,
}

/// A compose-commit record as the ledger reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitRow {
    pub id: u64,
    pub cx_promotion: u64,
    pub cx_evidence: u64,
    pub proposal_sha256: String,
    pub path: String,
    pub content_sha256: String,
    pub workspace: String,
    /// sovereign-core #408: sha256 of the goal the compose ran for (absent in
    /// commit records written before #408).
    pub goal_sha256: Option<String>,
}

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
    /// sovereign-core #408: the objective of the grant, carried on the intent.
    pub objective_id: Option<String>,
}

/// The effect ledger of one compose home, rebuilt from its host records.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    pub grants: BTreeMap<u64, Grant>,
    /// Approved grants by replay claim: more than one for a claim opens none.
    pub approved_by_claim: BTreeMap<u64, Vec<u64>>,
    /// Compose-commit records the daemon wrote (#261), by record id.
    pub commits: BTreeMap<u64, CommitRow>,
    /// Minted grants by compose-commit record, oldest first.
    pub minted_by_commit: BTreeMap<u64, Vec<u64>>,
    /// Desk-MAC nonces already used by a minted grant (#297): one grant per nonce.
    pub desk_nonces: std::collections::BTreeSet<String>,
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

/// Missing -> None, a string -> Some, anything else is damaged state.
fn opt_str(v: &Value, k: &str, id: u64) -> Result<Option<String>, Refusal> {
    match v.get(k) {
        None => Ok(None),
        Some(Value::String(x)) => Ok(Some(x.clone())),
        Some(_) => Err(corrupt(id, format!("field {k:?} is not a string"))),
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
        let objective = match opt_str(&v, "objective_id", id)? {
            None => None,
            Some(oid) => Some(GrantObjective {
                id: oid,
                text_sha256: s(&v, "objective_text_sha256", id)?,
                record_sha256: s(&v, "objective_record_sha256", id)?,
            }),
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
                minted: match v.get(MINTED_GRANT) {
                    None => None,
                    Some(_) => {
                        let commit = u(&v, "compose_commit", id)?;
                        self.minted_by_commit.entry(commit).or_default().push(id);
                        if let Some(n) = v.get("desk_nonce").and_then(Value::as_str) {
                            self.desk_nonces.insert(n.to_string());
                        }
                        Some(MintedGrant { commit })
                    }
                },
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
                objective,
                links: links.to_vec(),
            },
        );
        Ok(())
    }

    fn read_effect(&mut self, id: u64, text: &str) -> Result<(), Refusal> {
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return Ok(());
        };
        if v.get(COMPOSE_COMMIT).is_some() {
            self.commits.insert(
                id,
                CommitRow {
                    id,
                    cx_promotion: u(&v, "cx_promotion", id)?,
                    cx_evidence: u(&v, "cx_evidence", id)?,
                    proposal_sha256: s(&v, "proposal_sha256", id)?,
                    path: s(&v, "path", id)?,
                    content_sha256: s(&v, "content_sha256", id)?,
                    workspace: s(&v, "workspace", id)?,
                    goal_sha256: opt_str(&v, "goal_sha256", id)?,
                },
            );
            return Ok(());
        }
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
                    objective_id: opt_str(&v, "objective_id", id)?,
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
        if !g.honoured() {
            return Err(Refusal::new(
                "NotAuthorized",
                format!(
                    "authorization #{a} is a caller-written note; the daemon honours only grants it minted itself (sovereign-core #261): run aien compose authorize"
                ),
            ));
        }
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
            "intents": self.intents.values().map(|r| {
                let mut row = json!({
                    "intent": r.id, "authorization": r.authorization, "path": r.path,
                    "target": r.target, "content_sha256": r.content_sha256,
                    "prior_sha256": r.prior_sha256, "state": r.state.name(),
                    "state_record": r.state_record,
                    "executor": {"pid": r.executor_pid, "start": r.executor_start},
                });
                // #408: only when the intent names one, so older rows read as before.
                if let Some(o) = &r.objective_id {
                    row["objective_id"] = json!(o);
                }
                row
            }).collect::<Vec<_>>(),
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
    state_from(row, file_sha256(Path::new(&row.target)))
}

fn state_from(
    row: &IntentRow,
    now: Result<Option<String>, String>,
) -> (EffectState, Result<Option<String>, String>) {
    let st = match &now {
        Ok(Some(d)) if *d == row.content_sha256 => EffectState::Done,
        Ok(d) if *d == row.prior_sha256 => EffectState::NotDone,
        _ => EffectState::Unresolved,
    };
    (st, now)
}

/// The text of one reconcile record, exactly as it is journalled (`append` writes
/// `to_string()` of it). `st` is the decided state, `world` what the world check saw,
/// `disk` the digest it read or its error, `who` who decided. The seeded .om world
/// check (om/effects/world.om) is compared byte for byte against this function.
pub fn reconcile_text(
    row: &IntentRow,
    st: EffectState,
    world: EffectState,
    disk: &Result<Option<String>, String>,
    who: &str,
    objective: Option<&ObjectiveBinding>,
) -> Value {
    let mut text = json!({
        "phase": PHASE_RECONCILE, "intent": row.id, "authorization": row.authorization,
        "state": st.name(), "world": world.name(),
        "disk_sha256": disk.as_ref().ok().cloned().flatten(),
        "disk_error": disk.as_ref().err(), "by": who,
    });
    if let Some(o) = objective {
        text["objective_id"] = json!(o.objective_id);
        text["objective_record"] = json!(o.record);
    }
    text
}

/// The world check, confined (476ca4 c28): the target is read only while it
/// still resolves inside its grant's workspace (`confine_target` again, at
/// ack and reconcile time, so a directory swapped for a symlink after the
/// intent opened is seen). When confinement refuses, or the grant names no
/// workspace, or its grant is not daemon-minted, the state is UNRESOLVED, never
/// DONE or NOT_DONE.
pub fn confined_world_state(
    l: &Ledger,
    row: &IntentRow,
) -> (EffectState, Result<Option<String>, String>) {
    // sovereign-core #261: an intent whose grant the daemon did not write
    // (a caller-written note, opened before this rule) is never settled
    // DONE or NOT_DONE by the world check; it stays UNRESOLVED and the operator
    // is told why. (An operator declaration of NOT_DONE is still accepted;
    // DONE is refused.)
    if !l
        .grants
        .get(&row.authorization)
        .is_some_and(Grant::honoured)
    {
        return (
            EffectState::Unresolved,
            Err(format!(
                "{NOT_DAEMON_MINTED}: authorization #{} was not minted by the daemon (sovereign-core #261)",
                row.authorization
            )),
        );
    }
    let ws = l
        .grants
        .get(&row.authorization)
        .and_then(|g| g.workspace.clone());
    let Some(ws) = ws else {
        return (
            EffectState::Unresolved,
            Err(format!(
                "OutsideWorkspace: authorization #{} names no workspace",
                row.authorization
            )),
        );
    };
    // sovereign-core #267: confine and read through the same held directory
    // descriptors, so a swap after the check cannot redirect the read.
    match open_confined(&ws, &row.path, &row.target) {
        // The file exists but cannot be read: the plain read error, as before.
        Err(r) if r.name == "Unreadable" => (EffectState::Unresolved, Err(r.detail)),
        Err(r) => {
            let r = outside_of(r);
            (
                EffectState::Unresolved,
                Err(format!("{}: {}", r.name, r.detail)),
            )
        }
        Ok(t) => state_from(row, t.sha256()),
    }
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
    // sovereign-core #261: no caller writes an authorization, whatever it
    // says. Grants come from ComposeAuthorize and ComposeApprovedProposal,
    // control records from ComposeControl; all are written by the daemon.
    if kind == crate::generation::GENERATION {
        return Err(
            "ComposeNote: generation records are written only by the daemon when a turn finishes"
                .into(),
        );
    }
    if kind == "authorization" {
        return Err(
            "ComposeNote: authorization records are written only by the daemon (ComposeAuthorize, ComposeApprovedProposal, ComposeControl); a caller-written grant is never honoured (sovereign-core #261)"
                .into(),
        );
    }
    let Ok(v) = serde_json::from_str::<Value>(text) else {
        return Ok(());
    };
    // Provenance evidence is copied by the daemon along its own chain; no caller sets it.
    if v.get(crate::generation::PROVENANCE).is_some() {
        return Err(
            "ComposeNote: the provenance field is written only by the daemon (generation link and ALLEN agent id)"
                .into(),
        );
    }
    match kind {
        // The daemon's generation record (evidence only, never an input to any
        // decision): effect-class note with the marker field.
        "effect" if v.get(crate::generation::GENERATION).is_some() => Err(
            "ComposeNote: generation records are written only by the daemon when a turn finishes"
                .into(),
        ),
        "effect" if v.get("phase").is_some() => Err(
            "ComposeNote: effect records with a \"phase\" are written only by the effect commands"
                .into(),
        ),
        // sovereign-core #323: a phase-less `write_file` note spends its
        // authorization (NEXT-PHASE-1 journals, still read). No caller writes
        // a new one: it would block a live grant and void its revoke.
        "effect" if v.get("tool").and_then(Value::as_str) == Some("write_file") => Err(
            "ComposeNote: write_file effect records are written only by the effect commands (a NEXT-PHASE-1 spend note is never accepted from a caller)"
                .into(),
        ),
        // sovereign-core #249: approved-submission replay records come only from
        // crate::approved_replay.
        "effect" if v.get(crate::approved_replay::FIELD).is_some() => Err(
            "ComposeNote: approved_submission records are written only by the replay ledger".into(),
        ),
        // sovereign-core #261: the daemon's own compose-commit record.
        "effect" if v.get(COMPOSE_COMMIT).is_some() => Err(
            "ComposeNote: compose_commit records are written only by the compose run".into(),
        ),
        // ACCEPTANCE-v3 2.4: repair records come only from RecoverComposeHome.
        "constraint" if v.get("repair").is_some() => Err(
            "ComposeNote: constraint records with a \"repair\" field are written only by RecoverComposeHome"
                .into(),
        ),
        _ => Ok(()),
    }
}

/// Shape checks of `confine_target` (no filesystem walk below the workspace):
/// returns the plain components of `path`.
fn confine_shape<'a>(
    workspace: &str,
    path: &'a str,
    target: &str,
) -> Result<Vec<&'a std::ffi::OsStr>, Refusal> {
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
    Ok(rel
        .components()
        .filter_map(|c| match c {
            Component::Normal(n) => Some(n),
            _ => None,
        })
        .collect())
}

/// An open handle on a confined target (sovereign-core #267): the file was
/// opened by one `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` call below a
/// held descriptor of the workspace root, and is read through that descriptor
/// only, never re-opened by path, so the check and the read name the same
/// object even if a path component is swapped afterwards. `file` is the
/// regular file, or None when it does not exist.
pub struct ConfinedTarget {
    #[allow(dead_code)] // held so the chain stays open for the handle's life
    dir: std::os::fd::OwnedFd,
    pub file: Option<std::fs::File>,
    target: String,
}

impl ConfinedTarget {
    /// sha256 of the bytes behind the held descriptor, None when absent.
    pub fn sha256(&self) -> Result<Option<String>, String> {
        use std::io::Read;
        let Some(f) = &self.file else {
            return Ok(None);
        };
        let mut r = f;
        let mut h = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = r
                .read(&mut buf)
                .map_err(|e| format!("read {}: {e}", self.target))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(Some(hex(&h.finalize())))
    }
}

/// Test seams (unit tests and, through the `test-support` feature, the
/// integration tests): a swap hook that fires inside the real call between
/// confinement and the read, one at the start of `open_confined`, and a switch
/// that forces the ENOSYS fallback walk. Never compiled into the daemon.
#[cfg(any(test, feature = "test-support"))]
pub mod test_hooks {
    use std::cell::{Cell, RefCell};
    type Hook = RefCell<Option<Box<dyn FnOnce()>>>;
    thread_local! {
        static PAUSE: Hook = const { RefCell::new(None) };
        static BEFORE: Hook = const { RefCell::new(None) };
        static FALLBACK: Cell<bool> = const { Cell::new(false) };
    }
    /// Run `f` once, right after confinement succeeded and before any read.
    pub fn set_pause(f: impl FnOnce() + 'static) {
        PAUSE.with(|p| *p.borrow_mut() = Some(Box::new(f)));
    }
    /// Run `f` once, at the start of `open_confined`, before any descriptor opens.
    pub fn set_before(f: impl FnOnce() + 'static) {
        BEFORE.with(|p| *p.borrow_mut() = Some(Box::new(f)));
    }
    /// Force the per-component fallback walk (as on a kernel without openat2).
    pub fn set_force_fallback(on: bool) {
        FALLBACK.with(|f| f.set(on));
    }
    pub(super) fn run_pause() {
        if let Some(f) = PAUSE.with(|p| p.borrow_mut().take()) {
            f();
        }
    }
    pub(super) fn run_before() {
        if let Some(f) = BEFORE.with(|p| p.borrow_mut().take()) {
            f();
        }
    }
    pub(super) fn force_fallback() -> bool {
        FALLBACK.with(|f| f.get())
    }
}

/// `openat2(2)` syscall number: 437 on every architecture that has it.
const SYS_OPENAT2: libc::c_long = 437;

/// `struct open_how` of openat2(2) (libc marks its own copy non-exhaustive).
#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

/// Open `rel` below the directory `dirfd`, resolved by the kernel in ONE
/// step with RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS:
/// no symlink anywhere in `rel` is followed and nothing can resolve above
/// `dirfd`. Falls back to a per-component `openat(O_NOFOLLOW)` walk only when
/// the kernel lacks openat2 (ENOSYS); that walk is weaker (a directory
/// renamed out of the workspace mid-walk is not caught).
fn open_beneath(
    dirfd: libc::c_int,
    rel: &Path,
    flags: libc::c_int,
) -> std::io::Result<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(rel.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))?;
    let how = OpenHow {
        flags: (flags | libc::O_CLOEXEC) as u64,
        mode: 0,
        resolve: libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS,
    };
    #[cfg(any(test, feature = "test-support"))]
    let force_fallback = test_hooks::force_fallback();
    #[cfg(not(any(test, feature = "test-support")))]
    let force_fallback = false;
    let fd = if force_fallback {
        -1
    } else {
        // SAFETY: valid C string and a correctly sized open_how; a descriptor
        // returned (>= 0) is owned by nobody else and wrapped immediately.
        unsafe {
            libc::syscall(
                SYS_OPENAT2,
                dirfd,
                c.as_ptr(),
                &how as *const OpenHow,
                std::mem::size_of::<OpenHow>(),
            )
        }
    };
    if fd >= 0 {
        return Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd as libc::c_int) });
    }
    let err = if force_fallback {
        std::io::Error::from_raw_os_error(libc::ENOSYS)
    } else {
        std::io::Error::last_os_error()
    };
    if err.raw_os_error() != Some(libc::ENOSYS) {
        return Err(err);
    }
    // Fallback: per-component openat with O_NOFOLLOW.
    let comps: Vec<_> = rel.components().collect();
    let mut cur: Option<std::os::fd::OwnedFd> = None;
    for (i, comp) in comps.iter().enumerate() {
        let name = std::ffi::CString::new(comp.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))?;
        let fl = if i + 1 == comps.len() {
            flags
        } else {
            libc::O_RDONLY | libc::O_DIRECTORY
        };
        let base = cur.as_ref().map_or(dirfd, |f| f.as_raw_fd());
        // SAFETY: as above.
        let fd =
            unsafe { libc::openat(base, name.as_ptr(), fl | libc::O_CLOEXEC | libc::O_NOFOLLOW) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        cur = Some(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) });
    }
    cur.ok_or_else(|| std::io::Error::from_raw_os_error(libc::EINVAL))
}

/// Open a confined target (see `ConfinedTarget`). Ancestors of the workspace
/// root itself are trusted (checked canonical, opened `O_NOFOLLOW`);
/// everything below it is resolved beneath that held descriptor.
///
/// Refusal names. `OutsideWorkspace`: the shape checks, an escape, an
/// intermediate symlink or non-directory. Three INTERNAL names let each caller
/// keep the refusal it always gave (they never leave this module):
/// `NotRegular` (the final component exists and is a symlink, directory or
/// other non-file), `Unreadable` (the file exists but cannot be opened) and
/// `MissingParent` (a directory above the target is absent).
pub fn open_confined(workspace: &str, path: &str, target: &str) -> Result<ConfinedTarget, Refusal> {
    use std::os::fd::FromRawFd;
    let out = |w: String| Refusal::new("OutsideWorkspace", w);
    confine_shape(workspace, path, target)?;
    let root = std::ffi::CString::new(workspace)
        .map_err(|_| out(format!("workspace {workspace} holds a NUL")))?;
    // SAFETY: valid C string; result checked and wrapped immediately.
    let rfd = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if rfd < 0 {
        return Err(out(format!(
            "workspace {workspace}: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: `rfd` is a fresh descriptor owned by nobody else.
    let dir = unsafe { std::os::fd::OwnedFd::from_raw_fd(rfd) };
    #[cfg(any(test, feature = "test-support"))]
    test_hooks::run_before();
    let rel = Path::new(path);
    let not_regular = || {
        Refusal::new(
            "NotRegular",
            format!("target {target} exists and is not a regular file (symlink or other)"),
        )
    };
    let dir_flags = libc::O_RDONLY | libc::O_DIRECTORY;
    let parent = rel.parent().filter(|p| !p.as_os_str().is_empty());
    // O_NONBLOCK so opening a FIFO cannot hang; the type is checked on the fd.
    let file = match open_beneath(dir.as_raw_fd(), rel, libc::O_RDONLY | libc::O_NONBLOCK) {
        Ok(fd) => {
            let f = std::fs::File::from(fd);
            let m = f
                .metadata()
                .map_err(|e| Refusal::new("Unreadable", format!("read {target}: {e}")))?;
            if !m.file_type().is_file() {
                return Err(not_regular());
            }
            Some(f)
        }
        // ELOOP: a symlink. In the last component it is a non-regular target;
        // in a directory component it is an escape route.
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
            let leaf = parent.is_none_or(|p| open_beneath(dir.as_raw_fd(), p, dir_flags).is_ok());
            return Err(if leaf {
                not_regular()
            } else {
                out(format!(
                    "target directory of {target} passes through a symlink"
                ))
            });
        }
        // EXDEV: RESOLVE_BENEATH refused an escape. ENOTDIR: a component is not a directory.
        Err(e) if matches!(e.raw_os_error(), Some(libc::EXDEV | libc::ENOTDIR)) => {
            return Err(out(format!(
                "target directory of {target}: {e} (outside or not a directory)"
            )))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Absent file is fine; an absent parent directory is not.
            if let Some(p) = parent {
                if let Err(pe) = open_beneath(dir.as_raw_fd(), p, dir_flags) {
                    return Err(match pe.raw_os_error() {
                        Some(libc::ENOENT) => Refusal::new(
                            "MissingParent",
                            format!("target directory of {target}: {pe} (missing)"),
                        ),
                        _ => out(format!(
                            "target directory of {target}: {pe} (outside or a symlink)"
                        )),
                    });
                }
            }
            None
        }
        Err(e) => return Err(Refusal::new("Unreadable", format!("read {target}: {e}"))),
    };
    #[cfg(any(test, feature = "test-support"))]
    test_hooks::run_pause();
    Ok(ConfinedTarget {
        dir,
        file,
        target: target.to_string(),
    })
}

/// The refusal an outside caller sees: the internal names collapse to
/// `OutsideWorkspace`, exactly what `confine_target` always returned for them.
fn outside_of(r: Refusal) -> Refusal {
    match r.name {
        "NotRegular" | "MissingParent" | "Unreadable" => Refusal::new("OutsideWorkspace", r.detail),
        _ => r,
    }
}

/// Workspace confinement of an effect target (sovereign-core #249): the
/// workspace is an absolute, canonical directory that is not `/`; `path` is
/// relative with plain components only; `target` is exactly
/// `workspace/path`; every directory below the workspace is a real directory,
/// not a symlink; and the target itself, when present, is a regular file,
/// never a symlink. A file that exists but cannot be opened is still
/// confined (its read fails later, as before). Where the bytes are then read,
/// use `open_confined` and read through it (sovereign-core #267), so the
/// check and the read cannot be split by a swap.
pub fn confine_target(workspace: &str, path: &str, target: &str) -> Result<(), Refusal> {
    match open_confined(workspace, path, target) {
        Ok(_) => Ok(()),
        Err(r) => Err(outside_of(r)),
    }
}

/// `confine_target` and the sha256 of the target in one held walk.
pub fn confined_sha256(
    workspace: &str,
    path: &str,
    target: &str,
) -> Result<Option<String>, String> {
    open_confined(workspace, path, target)
        .map_err(|r| outside_of(r).to_string())?
        .sha256()
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
        || (
            &row.keys.request_id,
            &row.keys.approval_id,
            &row.keys.trace_id,
        ) != (&id.request_id, &id.approval_id, &id.trace_id)
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
    base: Option<&str>,
) -> Result<(u64, String), String> {
    b.with_home(|home| {
        let target = Path::new(workspace).join(path).display().to_string();
        // #267 follow-up: confine and hash through one held handle.
        let prior = confined_sha256(workspace, path, &target)?;
        // The file must still be the one the bound requirements were checked against.
        if let Some(base) = base {
            let now = prior.as_deref().unwrap_or("absent");
            if now != base {
                return Err(format!(
                    "BaseChanged: {path} changed after the requirements were checked (bound {base}, now {now}); no grant was written"
                ));
            }
        }
        let mut text = json!({
            APPROVED_GRANT: 1, "proposal_sha256": proposal_sha256, "path": path,
            "content_sha256": content_sha256, "approver": approver, "target": target,
            "workspace": workspace, "prior_sha256": prior,
            "approval_key": link.approval_key, "replay_claim": link.replay_claim,
            "cx_promotion": link.cx_promotion, "cx_evidence": link.cx_evidence,
            // An approved proposal runs no model: no generation record, the ALLEN only.
            crate::generation::PROVENANCE:
                crate::generation::Provenance::new(None, home.allen.as_ref()).to_json(),
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

/// A minted grant must be backed by the compose-commit record the daemon
/// wrote when its own compose run committed this proposal (#261): same
/// proposal, path, content and workspace, target = workspace + path, and the
/// grant links the record.
fn check_minted_backing(l: &Ledger, g: &Grant) -> Result<(), Refusal> {
    let Some(m) = &g.minted else {
        return Ok(());
    };
    let no = |w: &str| {
        Refusal::new(
            "NotAuthorized",
            format!(
                "minted grant #{} is not backed by a committed compose: {w}",
                g.id
            ),
        )
    };
    let c = l
        .commits
        .get(&m.commit)
        .ok_or_else(|| no("no such compose-commit record"))?;
    let bound_target = Path::new(&c.workspace).join(&c.path);
    if c.proposal_sha256 != g.proposal_sha256
        || c.path != g.path
        || c.content_sha256 != g.content_sha256
        || g.workspace.as_deref() != Some(c.workspace.as_str())
        || g.target.as_deref().map(Path::new) != Some(bound_target.as_path())
        || !g.links.contains(&m.commit)
    {
        return Err(no("grant fields differ from the committed proposal"));
    }
    Ok(())
}

/// Copy the provenance evidence of record `source` onto `text`, a record the
/// daemon is about to append. EVIDENCE ONLY: this is the single place the
/// ledger-side code reads provenance, it never fails and is called after every
/// decision of its caller (`provenance_changes_no_decision` scans for that).
fn stamp(text: &mut Value, views: &[ComposeRecordView], source: u64) {
    text[crate::generation::PROVENANCE] = crate::generation::inherit(views, source).to_json();
}

/// The record the daemon writes when its own compose run COMMITTED a
/// one-file proposal (#261): the only thing `ComposeAuthorize` mints from.
/// Linked to the promotion and evidence. Returns the record id.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compose_commit(
    home: &mut ComposeHome,
    workspace: &str,
    task: u64,
    cx_promotion: u64,
    cx_evidence: u64,
    proposal_sha256: &str,
    path: &str,
    content_sha256: &str,
    goal_sha256: Option<&str>,
    prov: &crate::generation::Provenance,
) -> Result<u64, String> {
    // Refused at write time: a record never names a generation id that is not
    // a verified generation record of this ledger.
    if let Some(g) = prov.generation_record {
        if !crate::generation::generation_exists(&host_views(home)?, g) {
            return Err(format!(
                "compose-commit not written: provenance names generation record #{g}, which is not a verified generation record"
            ));
        }
    }
    let mut text = json!({
        COMPOSE_COMMIT: 1, "task": task, "cx_promotion": cx_promotion,
        "cx_evidence": cx_evidence, "proposal_sha256": proposal_sha256,
        "path": path, "content_sha256": content_sha256, "workspace": workspace,
        crate::generation::PROVENANCE: prov.to_json(),
    });
    // #408: the goal digest, so a grant can be bound to the objective record
    // whose text this compose ran for.
    if let Some(g) = goal_sha256 {
        text["goal_sha256"] = json!(g);
    }
    append(home, NoteKind::Effect, &[cx_promotion, cx_evidence], &text).map(|n| n.id)
}

/// What `ComposeAuthorize` asks for: the operator's approval of one committed
/// proposal. Only ids and digests; path, content and target come from the
/// daemon's own commit record, never from the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintRequest {
    pub cx_promotion: u64,
    pub proposal_sha256: String,
    pub workspace: String,
    pub approver: String,
    /// Constraint records the grant links (at most 3).
    pub constraints: Vec<u64>,
}

/// ComposeAuthorize (#261): mint the one grant for a proposal this daemon
/// committed. Refused when no commit record names (promotion, proposal), the
/// workspace differs from the one the compose ran for, the target escapes it
/// (`confine_target`), or an earlier grant for the same commit is still live
/// or has an unsettled intent. A revoked, stale or NOT_DONE grant does not block a new one (each grant
/// is spent exactly once); a grant that settled DONE does: one committed
/// proposal gives at most one DONE effect.
pub fn mint_grant(b: &ComposeBridge, req: &MintRequest) -> ControlResponse {
    authorize(b, req, None)
}

/// `ComposeAuthorize` with the optional approval-desk proof (#297). When the
/// bridge runs with the desk switch ON, a missing or wrong proof, a missing
/// desk key, a MAC over other fields, or a reused nonce is refused and
/// NOTHING is written. With the switch OFF the proof is ignored (legacy
/// behaviour: OS-user authentication only).
/// The refusal name out of a `Refusal` display string, or "error" for any
/// other error text. Used only for trace notes (#398); never for decisions.
fn refusal_name(e: &str) -> &str {
    e.strip_prefix("EFFECT_REFUSED ")
        .and_then(|rest| rest.split(':').next())
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .unwrap_or("error")
}

/// Emits one correlation event for a control response (#398): `Ok` with the
/// note's record id in `id_slot`, `Rejected` with the refusal name for a
/// refusal, `Failed` for any other error. Observational only.
fn trace_response(
    b: &ComposeBridge,
    kind: aien_trace::EventKind,
    r: &ControlResponse,
    mut ids: aien_trace::CorrelationIds,
    id_slot: fn(&mut aien_trace::CorrelationIds, u64),
    certainty_ok: Option<aien_trace::EffectCertainty>,
) {
    use aien_trace::EventStatus;
    let (status, certainty, note) = match r {
        ControlResponse::ComposeNoted(n) => {
            id_slot(&mut ids, n.id);
            (EventStatus::Ok, certainty_ok, "")
        }
        ControlResponse::Error(e) if e.starts_with("EFFECT_REFUSED ") => (
            EventStatus::Rejected,
            Some(aien_trace::EffectCertainty::NoEffect),
            refusal_name(e),
        ),
        ControlResponse::Error(_) => (EventStatus::Failed, None, "error"),
        _ => (EventStatus::Malformed, None, "unexpected_response"),
    };
    b.trace_event(
        kind,
        status,
        certainty,
        ids,
        aien_trace::EvidenceRefs::default(),
        note,
    );
}

pub fn authorize(
    b: &ComposeBridge,
    req: &MintRequest,
    proof: Option<&crate::control::DeskProof>,
) -> ControlResponse {
    authorize_with_objective(b, req, proof, None)
}

/// sovereign-core #408: [`authorize`] for an approval that names the objective
/// record it is for. The daemon re-reads the record (crate::objective); its
/// text must be the goal this daemon composed for (`goal_sha256` of the
/// commit record). The grant then carries the objective id and both digests.
/// A missing or different record is refused by name and nothing is written.
/// `None` mints exactly as [`authorize`] always did.
pub fn authorize_with_objective(
    b: &ComposeBridge,
    req: &MintRequest,
    proof: Option<&crate::control::DeskProof>,
    objective_id: Option<&str>,
) -> ControlResponse {
    let r = authorize_inner(b, req, proof, objective_id);
    if b.trace_enabled() {
        trace_response(
            b,
            aien_trace::EventKind::AuthorityDecided,
            &r,
            aien_trace::CorrelationIds {
                decision_id: Some(aien_trace::BoundedId::new(&req.cx_promotion.to_string())),
                ..Default::default()
            },
            |ids, id| ids.grant_id = Some(aien_trace::BoundedId::new(&id.to_string())),
            Some(aien_trace::EffectCertainty::NoEffect),
        );
    }
    r
}

fn authorize_inner(
    b: &ComposeBridge,
    req: &MintRequest,
    proof: Option<&crate::control::DeskProof>,
    objective_id: Option<&str>,
) -> ControlResponse {
    if let Err(e) = reconcile_gate(b) {
        return ControlResponse::Error(e);
    }
    noted(b.with_home(|home| {
        let no = |name: &'static str, why: String| Refusal::new(name, why).to_string();
        if req.approver.trim().is_empty() {
            return Err(no("NotAuthorized", "ComposeAuthorize needs an approver".into()));
        }
        if req.constraints.len() > 3 {
            return Err(no("NotAuthorized", "at most 3 constraint links".into()));
        }
        let desk = if b.authorize_requires_desk() {
            let Some(p) = proof else {
                return Err(no(
                    "DeskMacRequired",
                    "this daemon requires the approval desk MAC on ComposeAuthorize (AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=1); none was supplied".into(),
                ));
            };
            if p.nonce.is_empty()
                || p.nonce.len() > 128
                || !p.nonce.bytes().all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
            {
                return Err(no("DeskMacInvalid", "nonce must be 1 to 128 of [A-Za-z0-9._-]".into()));
            }
            let key = crate::approved_auth::DeskKey::load(&crate::approved_auth::desk_key_path(b.dir()))
                .map_err(|e| no("NoDesk", e))?;
            Some((key, p))
        } else {
            None
        };
        let views = host_views(home)?;
        let l = Ledger::from_records(&views).map_err(|r| r.to_string())?;
        let mut hits = l
            .commits
            .values()
            .filter(|c| c.cx_promotion == req.cx_promotion && c.proposal_sha256 == req.proposal_sha256);
        let (Some(c), None) = (hits.next(), hits.next()) else {
            return Err(no(
                "NotAuthorized",
                format!(
                    "no compose-commit record names promotion #{} and this proposal: only a proposal this daemon committed (aien compose propose) can be authorized",
                    req.cx_promotion
                ),
            ));
        };
        let ws = std::fs::canonicalize(&req.workspace)
            .map_err(|e| no("OutsideWorkspace", format!("workspace {}: {e}", req.workspace)))?
            .display()
            .to_string();
        if ws != c.workspace {
            return Err(no(
                "OutsideWorkspace",
                format!("the proposal was committed for workspace {}, not {ws}", c.workspace),
            ));
        }
        let target = Path::new(&c.workspace).join(&c.path).display().to_string();
        // #267 follow-up: keep the confined handle; the prior hash below is
        // read from it, not re-resolved by path after the MAC checks.
        let held = open_confined(&c.workspace, &c.path, &target)
            .map_err(|r| outside_of(r).to_string())?;
        if let Some((key, p)) = &desk {
            // Path, content and workspace are the daemon's own commit record;
            // the MAC must cover exactly those.
            let binding = crate::approved_auth::AuthorizeBinding {
                cx_promotion: c.cx_promotion,
                proposal_sha256: c.proposal_sha256.clone(),
                path: c.path.clone(),
                content_sha256: c.content_sha256.clone(),
                workspace: c.workspace.clone(),
                approver: req.approver.trim().to_string(),
                constraints: req.constraints.clone(),
                nonce: p.nonce.clone(),
                desk_key_id: key.id().to_string(),
            };
            key.verify_authorize(&binding, &p.mac)
                .map_err(|e| no("DeskMacInvalid", e))?;
            if l.desk_nonces.contains(&p.nonce) {
                return Err(no(
                    "Replayed",
                    format!("authorize nonce {} already minted a grant; a replayed authorize mints nothing", p.nonce),
                ));
            }
        }
        let prior = held.sha256()?;
        for &gid in l.minted_by_commit.get(&c.id).into_iter().flatten() {
            let g = &l.grants[&gid];
            // The spent intent is judged FIRST: a revoke or a stop after the grant
            // never undoes a DONE effect, nor settles an intent that may still
            // land. Only an unspent revoked or stopped grant gives way.
            let dead = l.revoked.contains_key(&gid) || l.stops.iter().any(|&s| s > gid);
            match l.spent.get(&gid).map(|i| &l.intents[i]) {
                None if dead => continue,
                // Settled: the grant is spent for good (a new effect needs a
                // new grant, as before).
                Some(row) if row.state == EffectState::Done => {
                    return Err(no(
                        "AlreadySpent",
                        format!(
                            "grant #{gid} for this proposal settled DONE (intent #{}); one committed proposal gives at most one effect, new content needs a new compose",
                            row.id
                        ),
                    ))
                }
                Some(row) if row.state.terminal() => continue,
                Some(row) => {
                    return Err(no(
                        "ReconciliationRequired",
                        format!(
                            "intent #{} for grant #{gid} is {}; run aien compose reconcile",
                            row.id,
                            row.state.name()
                        ),
                    ))
                }
                None if g.prior_sha256.as_ref() != Some(&prior) => continue, // stale
                None => {
                    return Err(no(
                        "AlreadyAuthorized",
                        format!("grant #{gid} for this proposal is still live; revoke it first"),
                    ))
                }
            }
        }
        // #408: judged after every older refusal, so those keep their order.
        let objective = match objective_id {
            None => None,
            Some(oid) => {
                let Some(goal) = c.goal_sha256.as_deref() else {
                    return Err(no(
                        crate::objective::MISMATCH,
                        format!(
                            "compose-commit #{} names no goal digest (written before sovereign-core #408); objective {oid} cannot be bound to it",
                            c.id
                        ),
                    ));
                };
                Some(crate::objective::check(oid, goal).map_err(|(n, d)| no(n, d))?)
            }
        };
        let mut links = vec![c.id];
        links.extend(&req.constraints);
        let mut text = json!({
            MINTED_GRANT: 1, "compose_commit": c.id, "proposal_sha256": c.proposal_sha256,
            "path": c.path, "content_sha256": c.content_sha256,
            "approver": req.approver.trim(), "target": target, "workspace": c.workspace,
            "prior_sha256": prior, "cx_promotion": c.cx_promotion, "cx_evidence": c.cx_evidence,
        });
        if let Some((key, p)) = &desk {
            text["desk_nonce"] = json!(p.nonce);
            text["desk_key_id"] = json!(key.id());
        }
        if let Some(o) = &objective {
            text["objective_id"] = json!(o.objective_id);
            text["objective_text_sha256"] = json!(o.text_sha256);
            text["objective_record_sha256"] = json!(o.record_sha256);
        }
        stamp(&mut text, &views, c.id);
        append(home, NoteKind::Authorization, &links, &text)
    }))
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

pub(crate) fn append(
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

/// What `open_intent` learned about the target before the grant checks.
enum IntentRead {
    /// The hash of the target (None when absent), read through the handle
    /// confinement opened.
    Prior(Option<String>),
    /// The target cannot be read as a regular file (a directory, a leaf
    /// symlink, no permission). Refused at once as "Stale", exactly where the
    /// old by-path read refused it, before the grant is looked at.
    Stale(String),
    /// A directory above the target is absent: the old read saw "absent"; so
    /// does the grant check, then the confinement refusal follows.
    MissingParent(String),
    /// The path escapes the workspace (symlinked or non-directory component,
    /// bad shape). Nothing outside is read; the grant checks run with the
    /// grant's own prior standing in, then this refusal follows.
    Outside(String),
}

/// The prior hash an intent records (#267 follow-up): when the grant names a
/// workspace, the target is confined and hashed through one held handle
/// BEFORE any other check, so what is compared with the grant is what
/// confinement checked. A grant with no workspace keeps the by-path read (it
/// is refused right after: it names no workspace).
fn intent_prior(l: &Ledger, req: &IntentRequest) -> IntentRead {
    let stale = |e: String| {
        IntentRead::Stale(Refusal::new("Stale", format!("target unreadable: {e}")).to_string())
    };
    let ws = l
        .grants
        .get(&req.authorization)
        .and_then(|g| g.workspace.as_deref());
    let Some(ws) = ws else {
        return match file_sha256(Path::new(&req.target)) {
            Ok(p) => IntentRead::Prior(p),
            Err(e) => stale(e),
        };
    };
    match open_confined(ws, &req.path, &req.target) {
        Ok(t) => match t.sha256() {
            Ok(p) => IntentRead::Prior(p),
            Err(e) => stale(e),
        },
        Err(r) => match r.name {
            "NotRegular" | "Unreadable" => stale(r.detail),
            "MissingParent" => IntentRead::MissingParent(outside_of(r).to_string()),
            _ => IntentRead::Outside(r.to_string()),
        },
    }
}

/// ComposeEffectIntent: every check of 2.3 and the durable intent, in one
/// step under the home lock. The answer's id is the intent.
pub fn open_intent(b: &ComposeBridge, req: &IntentRequest) -> ControlResponse {
    let r = open_intent_inner(b, req);
    if b.trace_enabled() {
        trace_response(
            b,
            aien_trace::EventKind::ToolRequested,
            &r,
            aien_trace::CorrelationIds {
                grant_id: Some(aien_trace::BoundedId::new(&req.authorization.to_string())),
                ..Default::default()
            },
            |ids, id| ids.tool_request_id = Some(aien_trace::BoundedId::new(&id.to_string())),
            Some(aien_trace::EffectCertainty::NoEffect),
        );
    }
    r
}

fn open_intent_inner(b: &ComposeBridge, req: &IntentRequest) -> ControlResponse {
    if let Err(e) = reconcile_gate(b) {
        return ControlResponse::Error(e);
    }
    noted(b.with_home(|home| {
        let views = host_views(home)?;
        let l = Ledger::from_records(&views).map_err(|r| r.to_string())?;
        // Refusal order is the one callers always saw: an unreadable target is
        // "Stale" first; then the grant checks; then confinement. A confinement
        // refusal found while reading is held back until the grant checks ran.
        let (current, deferred) = match intent_prior(&l, req) {
            IntentRead::Prior(p) => (p, None),
            IntentRead::Stale(e) => return Err(e),
            IntentRead::MissingParent(e) => (None, Some(e)),
            IntentRead::Outside(e) => (
                l.grants
                    .get(&req.authorization)
                    .and_then(|g| g.prior_sha256.clone().flatten()),
                Some(e),
            ),
        };
        let g = l.check_intent(req, &current).map_err(|r| r.to_string())?;
        // sovereign-core #249: a grant names its workspace; confinement (already
        // done above, through the handle that was read); an approved grant's backing.
        if g.workspace.is_none() {
            return Err(Refusal::new(
                "NotAuthorized",
                format!("authorization #{} names no workspace", g.id),
            )
            .to_string());
        }
        if let Some(e) = deferred {
            return Err(e);
        }
        check_approved_backing(&l, g, &views).map_err(|r| r.to_string())?;
        check_minted_backing(&l, g).map_err(|r| r.to_string())?;
        // #408: an objective-bound grant opens an intent only while its record
        // still holds the approved text (re-read now, after any restart).
        if let Some(o) = &g.objective {
            crate::objective::check(&o.id, &o.text_sha256).map_err(|(n, d)| {
                Refusal::new(
                    n,
                    format!(
                        "authorization #{} is bound to objective {}: {d}",
                        g.id, o.id
                    ),
                )
                .to_string()
            })?;
        }
        let mut text = json!({
            "phase": PHASE_INTENT, "tool": "write_file", "authorization": g.id,
            "proposal_sha256": req.proposal_sha256, "path": req.path, "target": req.target,
            "content_sha256": req.content_sha256, "prior_sha256": current,
            "executor": {"pid": req.executor_pid, "start": req.executor_start},
        });
        if let Some(o) = &g.objective {
            text["objective_id"] = json!(o.id);
            text["objective_text_sha256"] = json!(o.text_sha256);
        }
        stamp(&mut text, &views, g.id);
        append(home, NoteKind::Effect, &[g.id], &text)
    }))
}

/// ComposeEffectAck: the executor's report after the write. The state is
/// read from the world; the executor's own claim is kept beside it.
pub fn ack(b: &ComposeBridge, intent: u64, reported: &Value) -> ControlResponse {
    if let Err(e) = reconcile_gate(b) {
        return ControlResponse::Error(e);
    }
    // The settled state, for the trace event only (#398).
    let mut settled: Option<String> = None;
    let mut grant: Option<u64> = None;
    let r = noted(b.with_home(|home| {
        let views = host_views(home)?;
        let l = Ledger::from_records(&views).map_err(|r| r.to_string())?;
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
        let (st, disk) = confined_world_state(&l, row);
        settled = Some(st.name().to_string());
        grant = Some(row.authorization);
        let mut text = json!({
            "phase": PHASE_ACK, "intent": intent, "authorization": row.authorization,
            "tool": "write_file", "path": row.path, "content_sha256": row.content_sha256,
            "state": st.name(), "disk_sha256": disk.as_ref().ok().cloned().flatten(),
            "disk_error": disk.as_ref().err(), "executor_reported": reported,
        });
        if let Some(o) = &row.objective_id {
            text["objective_id"] = json!(o);
        }
        stamp(&mut text, &views, intent);
        append(home, NoteKind::Effect, &[intent, row.authorization], &text)
    }));
    // Correlation only (#398): the settled state of this intent as the ledger
    // judged it. DONE is the only `effect_occurred`; NOT_DONE is `no_effect`;
    // UNRESOLVED stays `uncertain` and nothing reading a trace may retry it.
    if b.trace_enabled() {
        use aien_trace::{EffectCertainty, EventStatus};
        let (status, certainty, note) = match (&r, settled.as_deref()) {
            (ControlResponse::ComposeNoted(_), Some("DONE")) => (
                EventStatus::Ok,
                Some(EffectCertainty::EffectOccurred),
                "DONE",
            ),
            // NOT_DONE: the world check says the effect did not happen, so it is a
            // definite no-effect outcome (C2 `rejected`), not `failed` (ran and
            // errored).
            (ControlResponse::ComposeNoted(_), Some("NOT_DONE")) => (
                EventStatus::Rejected,
                Some(EffectCertainty::NoEffect),
                "NOT_DONE",
            ),
            (ControlResponse::ComposeNoted(_), _) => (
                EventStatus::Uncertain,
                Some(EffectCertainty::Uncertain),
                "UNRESOLVED",
            ),
            (ControlResponse::Error(e), _) if e.starts_with("EFFECT_REFUSED ") => {
                (EventStatus::Rejected, None, refusal_name(e))
            }
            (ControlResponse::Error(_), _) => (EventStatus::Failed, None, "error"),
            _ => (EventStatus::Malformed, None, "unexpected_response"),
        };
        b.trace_event(
            aien_trace::EventKind::EffectExecuted,
            status,
            certainty,
            aien_trace::CorrelationIds {
                tool_request_id: Some(aien_trace::BoundedId::new(&intent.to_string())),
                grant_id: grant.map(|g| aien_trace::BoundedId::new(&g.to_string())),
                ..Default::default()
            },
            aien_trace::EvidenceRefs::default(),
            note,
        );
    }
    r
}

/// sovereign-core #408: re-read the objective record of an objective-bound
/// intent for a reconcile report. Never decides the effect's state.
fn objective_binding(l: &Ledger, row: &IntentRow) -> Option<ObjectiveBinding> {
    let id = row.objective_id.as_ref()?;
    let refused = |name: &str, detail: String| ObjectiveBinding {
        objective_id: id.clone(),
        record: name.to_string(),
        detail: Some(detail),
        record_unchanged: None,
    };
    let approved = l
        .grants
        .get(&row.authorization)
        .and_then(|g| g.objective.as_ref())
        .filter(|a| &a.id == id);
    let Some(a) = approved else {
        return Some(refused(
            crate::objective::MISMATCH,
            format!(
                "intent #{} names objective {id}; its grant #{} is not bound to it",
                row.id, row.authorization
            ),
        ));
    };
    Some(match crate::objective::check(id, &a.text_sha256) {
        Ok(s) => ObjectiveBinding {
            objective_id: id.clone(),
            record: "bound".into(),
            detail: None,
            record_unchanged: Some(s.record_sha256 == a.record_sha256),
        },
        Err((n, d)) => refused(n, d),
    })
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
            let objective = objective_binding(&l, &row);
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
                    objective: objective.clone(),
                });
                continue;
            }
            let (world, disk) = confined_world_state(&l, &row);
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
            // Never DONE for a target outside its workspace, not even by declaration.
            if st == EffectState::Done
                && disk.as_ref().is_err_and(|e| {
                    e.starts_with("OutsideWorkspace") || e.starts_with(NOT_DAEMON_MINTED)
                })
            {
                return Err(format!(
                    "reconcile: intent #{} cannot be declared DONE: {}",
                    row.id,
                    disk.as_ref().err().map(String::as_str).unwrap_or_default()
                ));
            }
            if st == EffectState::Unresolved && row.unresolved_digest.as_ref() == Some(&disk_hex) {
                out.push(ReconcileOutcome {
                    intent: row.id,
                    authorization: row.authorization,
                    state: st.name().into(),
                    disk_sha256: disk_hex,
                    record: None,
                    note: "still UNRESOLVED, already recorded for this target digest".into(),
                    objective: objective.clone(),
                });
                continue;
            }
            let text = reconcile_text(&row, st, world, &disk, &who, objective.as_ref());
            let n = append(home, NoteKind::Effect, &[row.id, row.authorization], &text)?;
            out.push(ReconcileOutcome {
                intent: row.id,
                authorization: row.authorization,
                state: st.name().into(),
                disk_sha256: disk_hex,
                record: Some(n.id),
                note: who,
                objective,
            });
        }
        Ok(ComposeReconcileReport {
            compose_dir: b.dir().display().to_string(),
            machine_id: home.machine_id.clone(),
            checked,
            outcomes: out,
        })
    });
    let out = match r {
        Ok(r) => {
            // A full reconcile that completed lifts the refusal of 2.5; a
            // single declaration does not.
            if declare.is_none() {
                b.clear_reconcile_failed();
            }
            ControlResponse::ComposeReconciled(Box::new(r))
        }
        Err(e) => ControlResponse::Error(e),
    };
    // Correlation only (#398): one event per reconcile call. `by` is a short
    // caller label ("reconcile@start", an operator name), never free text
    // from a model; the note is bounded anyway.
    if b.trace_enabled() {
        let (status, note) = match &out {
            ControlResponse::ComposeReconciled(_) => (aien_trace::EventStatus::Ok, by),
            ControlResponse::Error(_) => (aien_trace::EventStatus::Failed, "reconcile_failed"),
            _ => (aien_trace::EventStatus::Malformed, "unexpected_response"),
        };
        b.trace_event(
            aien_trace::EventKind::ResultReceived,
            status,
            None,
            aien_trace::CorrelationIds {
                tool_request_id: declare.map(|d| aien_trace::BoundedId::new(&d.intent.to_string())),
                ..Default::default()
            },
            aien_trace::EvidenceRefs::default(),
            note,
        );
    }
    out
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

    #[test]
    fn refusal_name_reads_only_the_name() {
        assert_eq!(
            refusal_name("EFFECT_REFUSED NotAuthorized: needs an approver"),
            "NotAuthorized"
        );
        assert_eq!(refusal_name("EFFECT_REFUSED Revoked: grant #3"), "Revoked");
        assert_eq!(refusal_name("EFFECT_REFUSED : odd"), "error");
        assert_eq!(refusal_name("compose home refused: journal"), "error");
        assert_eq!(refusal_name(""), "error");
    }

    /// #398: the compose spine emits correlation events for authority, intent,
    /// ack and reconcile outcomes; refusals carry only the refusal name, and no
    /// event carries goal, proposal or path text.
    #[test]
    fn compose_spine_emits_correlation_events_without_content() {
        use aien_trace::{EffectCertainty, EventKind, EventStatus, MemorySink};
        let _home = crate::home_guard::home_slot();
        if !aien_omega_compose::LINKED {
            eprintln!("NOT_RUN: stub compose build");
            return;
        }
        let t = tempfile::tempdir().unwrap();
        let proposer: crate::compose::ComposeProposer =
            std::sync::Arc::new(|_: &str, _: std::time::Duration| {
                Ok(crate::compose::Generation::default())
            });
        let sink = std::sync::Arc::new(MemorySink::new());
        let b = ComposeBridge::new(t.path().join("compose"), proposer, "test:trace")
            .with_trace_sink(sink.clone());
        // Start-up reconcile on an empty home: one result_received event.
        let _ = reconcile_at_start(&b);
        // Refused authorization: no approver.
        let r = authorize(
            &b,
            &MintRequest {
                cx_promotion: 7,
                proposal_sha256: "00".repeat(32),
                workspace: t.path().display().to_string(),
                approver: "   ".into(),
                constraints: vec![],
            },
            None,
        );
        assert!(
            matches!(r, ControlResponse::Error(ref e) if e.contains("NotAuthorized")),
            "{r:?}"
        );
        // Ack of an intent that does not exist: plain error, never a certainty.
        let r = ack(&b, 999, &json!({"wrote": true}));
        assert!(matches!(r, ControlResponse::Error(_)), "{r:?}");
        let events = sink.drain();
        let kinds: Vec<EventKind> = events.iter().map(|e| e.kind).collect();
        assert!(
            kinds.contains(&EventKind::AuthorityDecided)
                && kinds.contains(&EventKind::EffectExecuted),
            "{kinds:?}"
        );
        let auth = events
            .iter()
            .find(|e| e.kind == EventKind::AuthorityDecided)
            .unwrap();
        assert_eq!(auth.status, EventStatus::Rejected);
        assert_eq!(auth.effect_certainty, Some(EffectCertainty::NoEffect));
        assert_eq!(
            auth.note.as_ref().map(|n| n.as_str()),
            Some("NotAuthorized")
        );
        assert_eq!(auth.ids.decision_id.as_ref().map(|d| d.as_str()), Some("7"));
        let eff = events
            .iter()
            .find(|e| e.kind == EventKind::EffectExecuted)
            .unwrap();
        assert_eq!(eff.status, EventStatus::Failed);
        assert_eq!(eff.effect_certainty, None);
        assert_eq!(
            eff.ids.tool_request_id.as_ref().map(|d| d.as_str()),
            Some("999")
        );
        // One trace id, one root, every other event a child of it.
        let root = events
            .iter()
            .filter(|e| e.parent_event_id.is_none())
            .count();
        assert_eq!(root, 1, "{events:?}");
        let root_id = events
            .iter()
            .find(|e| e.parent_event_id.is_none())
            .unwrap()
            .event_id;
        assert!(events.iter().all(|e| e.trace_id == b.trace_id()));
        assert!(events
            .iter()
            .filter(|e| e.event_id != root_id)
            .all(|e| e.parent_event_id == Some(root_id)));
        // No content: the serialized events never carry the workspace path or
        // the proposal digest, only ids, kinds, statuses and refusal names.
        let text = serde_json::to_string(&events).unwrap();
        assert!(!text.contains(&t.path().display().to_string()), "{text}");
        assert!(!text.contains(&"00".repeat(32)), "{text}");
        let tree = aien_trace::reconstruct(&events).unwrap();
        assert_eq!(tree.len(), events.len());
    }

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
            json!({"minted_grant": 1, "compose_commit": 1, "proposal_sha256": "p",
                   "path": "N.md", "content_sha256": "c",
                   "approver": "drake", "target": "/w/N.md", "prior_sha256": prior}),
        )
    }

    /// A grant record nobody minted: what a caller-written note looks like.
    fn caller_grant(id: u64) -> ComposeRecordView {
        rec(
            id,
            "authorization",
            json!({"proposal_sha256": "p", "path": "N.md", "content_sha256": "c",
                   "approver": "attacker", "target": "/w/N.md", "prior_sha256": null,
                   "workspace": "/w"}),
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

    /// A real daemon-built generation record, as a ledger row.
    fn generation_row(id: u64) -> ComposeRecordView {
        let identity = crate::generation::ModelIdentity {
            model_sha256: "m".repeat(64),
            model_digest_kind: "file".into(),
            model_path: "/models/m.safetensors".into(),
            tokenizer_sha256: "t".repeat(64),
            tokenizer_path: "/models/tokenizer.json".into(),
        };
        let record = crate::generation::build_record(
            &identity,
            &crate::generation::TurnEvidence {
                prompt_ids: &[1, 2, 3],
                output_ids: &[100, 101],
                text: "x100 x101",
                total_tokens: 2,
                finish_reason: "eos",
                request_id: 7,
                operation_id: 8,
                decoding: None,
                ops: None,
            },
            crate::generation::DaemonStart(1),
        );
        rec(id, "effect", record)
    }

    /// Evidence only: generation records, anywhere in the ledger, change no
    /// ledger state and no authorization or intent decision.
    #[test]
    fn generation_records_change_no_decision() {
        let base = vec![
            grant(2, Value::Null),
            intent(3, 2),
            settle(4, "ack", 3, "DONE"),
            grant(5, Value::Null),
        ];
        let mut with = base.clone();
        with.push(generation_row(6));
        with.push(generation_row(7));
        let (a, b) = (
            Ledger::from_records(&base).unwrap(),
            Ledger::from_records(&with).unwrap(),
        );
        assert_eq!(a.view(), b.view());
        assert_eq!(refusal(&a, 2, None), refusal(&b, 2, None));
        assert_eq!(refusal(&a, 2, None), "AlreadySpent");
        assert_eq!(
            a.check_intent(&req(5), &None).map(|g| g.id),
            b.check_intent(&req(5), &None).map(|g| g.id)
        );
        // A generation record alone opens nothing and spends nothing.
        let only = Ledger::from_records(&[generation_row(2)]).unwrap();
        assert_eq!(only.view(), Ledger::default().view());
    }

    #[test]
    fn generation_records_cannot_be_forged_through_compose_note() {
        let g = r#"{"generation":1,"model_sha256":"x"}"#;
        assert!(check_reserved_note("effect", g).is_err());
        assert!(check_reserved_note("generation", "anything").is_err());
        assert!(check_reserved_note("generation", g).is_err());
    }

    #[test]
    fn provenance_cannot_be_set_through_compose_note() {
        let p = r#"{"provenance":{"generation_record":3,"allen_agent":"none"}}"#;
        for kind in ["effect", "constraint", "authorization", "provenance"] {
            assert!(check_reserved_note(kind, p).is_err(), "{kind}");
        }
        // A forged copy of the daemon's own records, with provenance, is refused too.
        let c = r#"{"compose_commit":1,"provenance":{"generation_record":3}}"#;
        assert!(check_reserved_note("effect", c).is_err());
        // Without the field an ordinary note is still fine.
        assert!(check_reserved_note("constraint", r#"{"a":1}"#).is_ok());
    }

    fn commit_row(id: u64, prov: Option<Value>) -> ComposeRecordView {
        let mut v = json!({"compose_commit": 1, "task": 1, "cx_promotion": 1, "cx_evidence": 1,
            "proposal_sha256": "p", "path": "N.md", "content_sha256": "c", "workspace": "/w"});
        if let Some(p) = prov {
            v["provenance"] = p;
        }
        rec(id, "effect", v)
    }

    fn with_prov(mut r: ComposeRecordView, prov: Value) -> ComposeRecordView {
        let mut v: Value = serde_json::from_str(r.text.as_deref().unwrap()).unwrap();
        v["provenance"] = prov;
        r.text = Some(v.to_string());
        r
    }

    /// Evidence only: whatever the provenance fields say (valid, absent, naming
    /// nothing, the wrong type, hostile), the ledger state and every decision
    /// are the ones the same records give without them.
    #[test]
    fn provenance_changes_no_decision() {
        let chain = |prov: Option<Value>| {
            let mut v = vec![
                commit_row(1, prov.clone()),
                grant(2, Value::Null),
                intent(3, 2),
                settle(4, "ack", 3, "DONE"),
                grant(5, Value::Null),
                intent(6, 5),
            ];
            if let Some(p) = prov {
                for r in v.iter_mut().skip(1) {
                    *r = with_prov(r.clone(), p.clone());
                }
            }
            v
        };
        let base = Ledger::from_records(&chain(None)).unwrap();
        for prov in [
            json!({"generation_record": 7, "allen_agent": "ab".repeat(32)}),
            json!({"generation_record": 99999, "allen_agent": "none"}),
            json!({"generation_record": null, "allen_agent": "none"}),
            json!({"generation_record": null, "allen_agent": "none", "generation_record_stale": 9, "provenance_note": "generation_record_stale"}),
            json!({"generation_record": "seven", "allen_agent": 5}),
            json!("garbage"),
            json!(null),
            json!({"generation_record": u64::MAX, "allen_agent": "\u{0}x".repeat(500)}),
        ] {
            let l = Ledger::from_records(&chain(Some(prov.clone()))).unwrap();
            assert_eq!(base.view(), l.view(), "{prov}");
            for a in [2u64, 5, 9] {
                assert_eq!(refusal(&base, a, None), refusal(&l, a, None), "{prov} #{a}");
            }
            assert_eq!(
                base.check_intent(&req(5), &None).map(|g| g.id),
                l.check_intent(&req(5), &None).map(|g| g.id)
            );
            assert_eq!(base.commits, l.commits, "{prov}");
            assert_eq!(base.grants, l.grants, "{prov}");
            assert_eq!(base.intents, l.intents, "{prov}");
        }
    }

    /// The same, for the code: the only reader of provenance is
    /// `Provenance::from_text`, reached only through `stamp`/`inherit`, and the
    /// ledger-side code that mentions provenance is a short, named list of
    /// writers that call it after their decisions are made. Nothing in
    /// `Ledger`, `check_*`, the world checks or reconcile can see it.
    #[test]
    fn provenance_is_read_by_no_decision_code() {
        let src = include_str!("effects.rs");
        let prod = src.split("#[cfg(test)]\nmod tests").next().unwrap();
        let tokens = [
            "PROVENANCE",
            "Provenance",
            "provenance",
            "stamp(",
            "inherit(",
            "generation_record",
            "allen_agent",
            "stale_generation",
            "provenance_note",
        ];
        // The enclosing `fn` of every mention.
        let mut cur = String::from("<top>");
        let mut hits: Vec<(String, String)> = Vec::new();
        for line in prod.lines() {
            let t = line.trim_start();
            if let Some(i) = t.find("fn ") {
                let pre = &t[..i];
                if pre.is_empty() || pre == "pub " || pre == "pub(crate) " {
                    cur = t[i + 3..]
                        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .next()
                        .unwrap()
                        .to_string();
                }
            }
            if t.starts_with("//") {
                continue;
            }
            if tokens.iter().any(|k| line.contains(k)) {
                hits.push((cur.clone(), t.to_string()));
            }
        }
        let writers = [
            "stamp",
            "write_compose_commit",
            "write_approved_grant",
            "authorize",
            "authorize_inner",
            "open_intent",
            "open_intent_inner",
            "ack",
            "check_reserved_note",
        ];
        for (f, line) in &hits {
            assert!(
                writers.contains(&f.as_str()),
                "{f} mentions provenance: {line}"
            );
        }
        // Inside the writers that also decide, the only mention is the stamp
        // call (or the reserved-field refusal), never a read of the fields.
        for (f, line) in &hits {
            if [
                "authorize",
                "authorize_inner",
                "open_intent",
                "open_intent_inner",
                "ack",
            ]
            .contains(&f.as_str())
            {
                assert!(
                    line.starts_with("stamp(&mut text, &views, "),
                    "{f} touches provenance other than by stamp: {line}"
                );
            }
        }
        // Everything else in the crate: the field's name and its reader appear
        // only in generation.rs (definition), effects.rs (above), compose/bridge.rs
        // (the writer of the commit) and server.rs/control.rs (plumbing). Every
        // source file, subdirectories included.
        let mut files = Vec::new();
        let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    dirs.push(p);
                } else {
                    files.push(p);
                }
            }
        }
        for p in files {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".rs") {
                continue;
            }
            let s = std::fs::read_to_string(&p).unwrap();
            let s = s.split("#[cfg(test)]").next().unwrap();
            let reads = s.contains("Provenance::from_text") || s.contains("generation::inherit");
            let allowed = ["generation.rs", "effects.rs"].contains(&name.as_str());
            assert!(!reads || allowed, "{name} reads provenance");
        }
        // The Ledger types carry no provenance field at all.
        for ty in [
            "pub struct Grant",
            "pub struct CommitRow",
            "pub struct IntentRow",
            "pub struct Ledger",
        ] {
            let body = prod
                .split(ty)
                .nth(1)
                .unwrap()
                .split("\n}\n")
                .next()
                .unwrap();
            assert!(!body.to_lowercase().contains("provenance"), "{ty}");
        }
    }

    #[test]
    fn old_records_without_provenance_read_as_explicit_none() {
        use crate::generation::{inherit, Provenance, NO_AGENT};
        let old = vec![commit_row(1, None), grant(2, Value::Null), intent(3, 2)];
        // The ledger opens and verifies as before.
        assert!(Ledger::from_records(&old).is_ok());
        let p = inherit(&old, 1);
        assert_eq!(p, Provenance::default());
        assert_eq!(p.allen_agent, NO_AGENT);
        assert_eq!(p.generation_record, None);
        // Missing record, or text that is not a record at all: the plain default
        // (no provenance field was ever there), never an error.
        assert_eq!(inherit(&old, 77), Provenance::default());
        assert_eq!(Provenance::from_text("not json"), Provenance::default());
        // A provenance field that is PRESENT but damaged is not silently "none":
        // the values default and a visible note says why.
        let bad_agent = Provenance::from_text(
            r#"{"provenance":{"allen_agent":"nope","generation_record":-1}}"#,
        );
        assert_eq!(bad_agent.allen_agent, NO_AGENT);
        assert_eq!(bad_agent.note.as_deref(), Some("allen_agent_malformed"));
        assert_ne!(bad_agent, Provenance::default());
        assert_eq!(
            bad_agent.to_json()["provenance_note"],
            "allen_agent_malformed"
        );
        let not_obj = Provenance::from_text(r#"{"provenance":"garbage"}"#);
        assert_eq!(not_obj.note.as_deref(), Some("provenance_malformed"));
        // A generation id that names no generation record is not copied as a
        // plain null: the old id stays visible and a note says it is stale.
        let named = vec![commit_row(
            1,
            Some(json!({"generation_record": 9, "allen_agent": "ab".repeat(32)})),
        )];
        let p = inherit(&named, 1);
        assert_eq!(p.generation_record, None);
        assert_eq!(p.stale_generation, Some(9));
        assert_eq!(p.note.as_deref(), Some("generation_record_stale"));
        assert_eq!(p.allen_agent, "ab".repeat(32));
        let j = p.to_json();
        assert_eq!(j["generation_record"], Value::Null);
        assert_eq!(j["generation_record_stale"], 9);
        assert_eq!(j["provenance_note"], "generation_record_stale");
        // The marker survives the next copy down the chain (grant -> intent).
        let next = vec![rec(
            2,
            "effect",
            json!({"phase": "intent", "provenance": j}),
        )];
        let q = inherit(&next, 2);
        assert_eq!(q.stale_generation, Some(9));
        assert_eq!(q.note.as_deref(), Some("generation_record_stale"));
        // ... and kept when it does (the record must be a verified generation record).
        let mut with_gen = named.clone();
        with_gen.push(generation_row(9));
        assert_eq!(inherit(&with_gen, 1).generation_record, Some(9));
        // A non-generation record (or an unverified one) does not count.
        let mut not_gen = named;
        not_gen.push(rec(9, "effect", json!({"x": 1})));
        assert_eq!(inherit(&not_gen, 1).generation_record, None);
        let mut bad = with_gen;
        bad[1].verified = false;
        assert_eq!(inherit(&bad, 1).generation_record, None);
    }

    /// A record that names a nonexistent generation id is refused at write time.
    #[test]
    fn a_commit_naming_no_generation_record_is_not_written() {
        let _home = crate::home_guard::home_slot();
        use crate::compose::{ComposeBridge, Generation};
        use crate::generation::Provenance;
        if !aien_omega_compose::LINKED {
            eprintln!("NOT_RUN: stub compose build");
            return;
        }
        let t = tempfile::tempdir().unwrap();
        let proposer: crate::compose::ComposeProposer =
            std::sync::Arc::new(|_: &str, _: std::time::Duration| Ok(Generation::default()));
        let b = ComposeBridge::new(t.path().join("compose"), proposer, "test:fixed");
        let note = match b.note("constraint", "an ordinary note", &[]) {
            ControlResponse::ComposeNoted(n) => n.id,
            other => panic!("{other:?}"),
        };
        let write = |g: Option<u64>| {
            b.with_home(|home| {
                write_compose_commit(
                    home,
                    "/w",
                    1,
                    1,
                    1,
                    "p",
                    "N.md",
                    "c",
                    None,
                    &Provenance {
                        generation_record: g,
                        allen_agent: "none".into(),
                        ..Default::default()
                    },
                )
            })
        };
        let before = host_count(&b);
        for bad in [999_999u64, note, 1] {
            let e = write(Some(bad)).unwrap_err();
            assert!(e.contains("not a verified generation record"), "{bad}: {e}");
        }
        assert_eq!(host_count(&b), before, "nothing was written");
        // A real generation record is accepted, and none is accepted.
        let g = b
            .with_home(|home| crate::generation::write(home, &generation_json()))
            .unwrap();
        let ok = write(Some(g));
        assert!(ok.is_ok(), "{ok:?}");
        let ok = write(None);
        assert!(ok.is_ok(), "{ok:?}");
    }

    fn generation_json() -> Value {
        json!({"generation": 1, "v": 1, "model_sha256": "x"})
    }

    fn host_count(b: &crate::compose::ComposeBridge) -> usize {
        match b.recall(&[], None) {
            ControlResponse::ComposeRecalled(r) => r.host.len(),
            other => panic!("{other:?}"),
        }
    }

    /// sovereign-core #408: records written before the objective fields load
    /// exactly as before (no objective), and the new fields are read back.
    #[test]
    fn objective_fields_are_additive_on_grants_and_intents() {
        let old = Ledger::from_records(&[grant(2, Value::Null), intent(3, 2)]).unwrap();
        assert_eq!(old.grants[&2].objective, None);
        assert_eq!(old.intents[&3].objective_id, None);
        assert!(old.view()["intents"][0].get("objective_id").is_none());
        let oid = "a".repeat(64);
        let mut g = json!({"minted_grant": 1, "compose_commit": 1, "proposal_sha256": "p",
            "path": "N.md", "content_sha256": "c", "approver": "drake", "target": "/w/N.md",
            "prior_sha256": null, "objective_id": oid, "objective_text_sha256": "b".repeat(64),
            "objective_record_sha256": "c".repeat(64)});
        let mut i = json!({"phase": "intent", "tool": "write_file", "authorization": 2,
            "path": "N.md", "target": "/w/N.md", "content_sha256": "c", "prior_sha256": null,
            "proposal_sha256": "p", "executor": {"pid": 1, "start": 1}, "objective_id": oid});
        let l = Ledger::from_records(&[
            rec(2, "authorization", g.clone()),
            rec(3, "effect", i.clone()),
        ])
        .unwrap();
        let o = l.grants[&2].objective.as_ref().unwrap();
        assert_eq!(
            (o.id.as_str(), o.text_sha256.as_str()),
            (oid.as_str(), "b".repeat(64).as_str())
        );
        assert_eq!(l.intents[&3].objective_id.as_deref(), Some(oid.as_str()));
        assert_eq!(l.view()["intents"][0]["objective_id"], json!(oid));
        // A commit without the goal digest (written before #408) still loads.
        let c = rec(
            1,
            "effect",
            json!({"compose_commit": 1, "task": 1, "cx_promotion": 1,
            "cx_evidence": 1, "proposal_sha256": "p", "path": "N.md", "content_sha256": "c",
            "workspace": "/w"}),
        );
        assert_eq!(
            Ledger::from_records(&[c]).unwrap().commits[&1].goal_sha256,
            None
        );
        // A daemon-written grant naming an objective without its digest is damaged state.
        g.as_object_mut().unwrap().remove("objective_text_sha256");
        assert!(Ledger::from_records(&[rec(2, "authorization", g)]).is_err());
        i["objective_id"] = json!(7);
        assert!(Ledger::from_records(&[grant(2, Value::Null), rec(3, "effect", i)]).is_err());
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
    fn a_caller_written_grant_is_never_honoured() {
        // Fields that would pass every other check; no marker, so no grant.
        let l = Ledger::from_records(&[caller_grant(2)]).unwrap();
        let e = l.check_intent(&req(2), &None).unwrap_err();
        assert_eq!(e.name, "NotAuthorized");
        assert!(e.detail.contains("caller-written"), "{}", e.detail);
        // An intent it opened before the rule is never settled by the world.
        let l = Ledger::from_records(&[caller_grant(2), intent(3, 2)]).unwrap();
        let (st, disk) = confined_world_state(&l, &l.intents[&3]);
        assert_eq!(st, EffectState::Unresolved);
        assert!(disk.unwrap_err().starts_with(NOT_DAEMON_MINTED));
        // A minted grant with no compose-commit record behind it opens nothing.
        let l = Ledger::from_records(&[grant(2, Value::Null)]).unwrap();
        let e = check_minted_backing(&l, &l.grants[&2]).unwrap_err();
        assert!(
            e.detail.contains("no such compose-commit record"),
            "{}",
            e.detail
        );
    }

    #[test]
    fn world_change_and_legacy_grants_are_stale() {
        let l = Ledger::from_records(&[grant(2, Value::Null)]).unwrap();
        assert_eq!(refusal(&l, 2, Some("x")), "Stale");
        let legacy = rec(
            2,
            "authorization",
            json!({"minted_grant": 1, "compose_commit": 1, "proposal_sha256": "p",
                   "path": "N.md", "content_sha256": "c", "approver": "d"}),
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

    /// #267 red/green: a parent swapped for a symlink to a directory holding
    /// the expected bytes, exactly between confinement and the read, must not
    /// make the world check DONE. (Old code: confine_target, then read by
    /// path, reached the outside file and returned DONE.)
    #[test]
    fn swap_between_confinement_and_read_is_not_done() {
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let (ws, outside) = (root.join("ws"), root.join("outside"));
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("d/f"), b"inside").unwrap();
        std::fs::write(outside.join("f"), b"expected").unwrap();
        let l = race_ledger(&ws, b"expected");
        let (w2, o2) = (ws.clone(), outside.clone());
        test_hooks::set_pause(move || {
            std::fs::rename(w2.join("d"), w2.join("d.held")).unwrap();
            std::os::unix::fs::symlink(&o2, w2.join("d")).unwrap();
        });
        let (st, disk) = confined_world_state(&l, &l.intents[&3]);
        assert_ne!(st, EffectState::Done, "read outside bytes: {disk:?}");
        assert_eq!(st, EffectState::Unresolved);
        assert_eq!(disk.unwrap(), Some(hex(&Sha256::digest(b"inside"))));
    }

    /// #267 follow-up, grant creation: the REAL `write_approved_grant`, with the
    /// parent swapped for a symlink to a directory holding other bytes exactly
    /// between its confinement and its read. The grant must record the hash of
    /// the file confinement checked, and a `base` equal to that hash must
    /// still pass. (With the by-path read the recorded prior is the OUTSIDE
    /// hash and `base` is refused as BaseChanged.)
    #[test]
    fn write_approved_grant_records_the_prior_of_the_confined_file() {
        let _home = crate::home_guard::home_slot();
        use crate::compose::{ComposeBridge, Generation};
        if !aien_omega_compose::LINKED {
            eprintln!("NOT_RUN: stub compose build");
            return;
        }
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let (ws, outside) = (root.join("ws"), root.join("outside"));
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("d/NOTES.md"), b"inside").unwrap();
        std::fs::write(outside.join("NOTES.md"), b"expected").unwrap();
        let proposer: crate::compose::ComposeProposer =
            std::sync::Arc::new(|_: &str, _: std::time::Duration| {
                Ok(Generation {
                    text: "filename: d/NOTES.md\nkeep it\n".to_string(),
                    tokens: 4,
                    finish_reason: Some("eos".into()),
                    ..Default::default()
                })
            });
        let b = ComposeBridge::new(root.join("compose"), proposer, "test:fixed");
        let r = b.run_task("g", ws.to_str().unwrap());
        let ControlResponse::ComposeTaskResult(rep) = r else {
            panic!("run_task did not produce a result: {r:?}")
        };
        let cx = rep.cx_promotion;
        let link = ApprovedLink {
            approval_key: "k".into(),
            replay_claim: cx,
            cx_promotion: cx,
            cx_evidence: cx,
        };
        let w = ws.to_str().unwrap();
        let prior_of = |id: u64| -> Value {
            let ControlResponse::ComposeRecalled(r) = b.recall(&[id], None) else {
                panic!("recall failed")
            };
            let v: Value = serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap();
            v["prior_sha256"].clone()
        };
        for base in [None, Some(hex(&Sha256::digest(b"inside")))] {
            let (w2, o2) = (ws.clone(), outside.clone());
            test_hooks::set_pause(move || swap_in_symlink(&w2, &o2));
            let r = write_approved_grant(
                &b,
                w,
                "d/NOTES.md",
                "c",
                "drake",
                &json!({}),
                &link,
                "p",
                base.as_deref(),
            );
            let (id, _) = r.unwrap_or_else(|e| panic!("grant refused: {e}"));
            assert_eq!(
                prior_of(id),
                json!(hex(&Sha256::digest(b"inside"))),
                "the grant recorded bytes that confinement did not check"
            );
            // put the directory back for the next round
            std::fs::remove_file(ws.join("d")).unwrap();
            std::fs::rename(ws.join("d.held"), ws.join("d")).unwrap();
        }
    }

    fn swap_in_symlink(ws: &Path, outside: &Path) {
        std::fs::rename(ws.join("d"), ws.join("d.held")).unwrap();
        std::os::unix::fs::symlink(outside, ws.join("d")).unwrap();
    }

    fn h(b: &[u8]) -> Option<String> {
        Some(hex(&Sha256::digest(b)))
    }

    /// Both the openat2 path and the forced ENOSYS fallback walk give the same
    /// exact refusal for every shape: a symlink in a middle component, a symlink
    /// to a file in a middle component (ELOOP) as against a plain file there
    /// (ENOTDIR), a symlink as the last component, a missing parent, and a
    /// `..` path.
    #[test]
    fn fallback_walk_refuses_symlinks_and_outside_paths() {
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let (ws, outside) = (root.join("ws"), root.join("outside"));
        std::fs::create_dir_all(ws.join("a/b")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("f"), b"expected").unwrap();
        std::fs::write(ws.join("a/b/f"), b"inside").unwrap();
        std::fs::write(ws.join("a/file"), b"plain file").unwrap();
        std::os::unix::fs::symlink(&outside, ws.join("s")).unwrap();
        std::os::unix::fs::symlink(&outside, ws.join("a/l")).unwrap();
        std::os::unix::fs::symlink(outside.join("f"), ws.join("a/b/lf")).unwrap();
        std::os::unix::fs::symlink(ws.join("a/file"), ws.join("a/lfile")).unwrap();
        let w = ws.to_str().unwrap();
        let open = |p: &str| open_confined(w, p, &ws.join(p).display().to_string());
        for forced in [false, true] {
            test_hooks::set_force_fallback(forced);
            assert_eq!(open("a/b/f").unwrap().sha256().unwrap(), h(b"inside"));
            assert!(open("a/b/new").unwrap().file.is_none());
            for (p, want) in [
                ("s/f", "OutsideWorkspace"),       // symlink to a dir, middle
                ("a/l/f", "OutsideWorkspace"),     // same, deeper
                ("a/lfile/f", "OutsideWorkspace"), // symlink to a FILE, middle: ELOOP
                ("a/file/f", "OutsideWorkspace"),  // plain file, middle: ENOTDIR
                ("a/b/lf", "NotRegular"),          // symlink, last component
                ("s", "NotRegular"),               // symlink, only component
                ("a/missing/f", "MissingParent"),
            ] {
                let e = open(p)
                    .err()
                    .unwrap_or_else(|| panic!("{p} was opened (forced={forced})"));
                assert_eq!(e.name, want, "{p} (forced={forced}): {}", e.detail);
            }
            let e = open_confined(w, "../outside/f", &format!("{w}/../outside/f"));
            assert_eq!(e.err().unwrap().name, "OutsideWorkspace");
        }
        test_hooks::set_force_fallback(false);
    }

    /// Ledger with one daemon-minted grant for `ws` and one open intent
    /// (#3) writing `d/f` with content `want`.
    fn race_ledger(ws: &Path, want: &[u8]) -> Ledger {
        let target = ws.join("d/f").display().to_string();
        let g = rec(
            2,
            "authorization",
            json!({"minted_grant": 1, "compose_commit": 1, "proposal_sha256": "p",
                   "path": "d/f", "content_sha256": hex(&Sha256::digest(want)),
                   "approver": "drake", "target": target, "prior_sha256": null,
                   "workspace": ws.display().to_string()}),
        );
        let i = rec(
            3,
            "effect",
            json!({"phase": "intent", "tool": "write_file", "authorization": 2, "path": "d/f",
                   "target": target, "content_sha256": hex(&Sha256::digest(want)),
                   "prior_sha256": null, "proposal_sha256": "p",
                   "executor": {"pid": 1, "start": 1}}),
        );
        Ledger::from_records(&[g, i]).unwrap()
    }

    /// #267, deterministic: after confinement hands back a held handle, a
    /// swap of the parent directory for a symlink to outside bytes cannot
    /// redirect the read; the handle still reads the inside object.
    #[test]
    fn held_handle_ignores_a_parent_swap() {
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let (ws, outside) = (root.join("ws"), root.join("outside"));
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("d/f"), b"inside").unwrap();
        std::fs::write(outside.join("f"), b"outside").unwrap();
        let target = ws.join("d/f").display().to_string();
        let h = open_confined(ws.to_str().unwrap(), "d/f", &target).unwrap();
        std::fs::rename(ws.join("d"), ws.join("d.held")).unwrap();
        std::os::unix::fs::symlink(&outside, ws.join("d")).unwrap();
        // By path the old code would now read the outside file ...
        assert_eq!(
            file_sha256(Path::new(&target)).unwrap(),
            Some(hex(&Sha256::digest(b"outside")))
        );
        // ... the held handle reads what confinement checked.
        assert_eq!(h.sha256().unwrap(), Some(hex(&Sha256::digest(b"inside"))));
        // A fresh walk refuses the swapped directory.
        let e = open_confined(ws.to_str().unwrap(), "d/f", &target)
            .err()
            .unwrap();
        assert_eq!(e.name, "OutsideWorkspace");
    }

    /// #267 acceptance: a parent directory swapped for a symlink to a
    /// directory holding the expected bytes, raced against ack/reconcile's
    /// `confined_world_state`, never yields DONE (the inside file differs).
    #[test]
    fn parent_swap_race_never_reads_outside_bytes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let t = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(t.path()).unwrap();
        let (ws, outside) = (root.join("ws"), root.join("outside"));
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("d/f"), b"inside").unwrap();
        std::fs::write(outside.join("f"), b"expected").unwrap();
        let l = race_ledger(&ws, b"expected");
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let swapper = {
            let (stop, ws, outside) = (stop.clone(), ws.clone(), outside.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // dir -> symlink to outside, then back.
                    std::fs::rename(ws.join("d"), ws.join("d.held")).unwrap();
                    std::os::unix::fs::symlink(&outside, ws.join("d")).unwrap();
                    std::fs::remove_file(ws.join("d")).unwrap();
                    std::fs::rename(ws.join("d.held"), ws.join("d")).unwrap();
                }
            })
        };
        let (mut done, mut seen) = (0u32, std::collections::BTreeMap::<String, u32>::new());
        let end = std::time::Instant::now() + std::time::Duration::from_secs(8);
        while std::time::Instant::now() < end {
            let (st, disk) = confined_world_state(&l, &l.intents[&3]);
            if st == EffectState::Done {
                done += 1;
            }
            *seen
                .entry(format!("{} {:?}", st.name(), disk.is_ok()))
                .or_default() += 1;
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert_eq!(done, 0, "DONE for bytes outside the workspace: {seen:?}");
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
            objective_id: None,
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
        // #323: the NEXT-PHASE-1 spend note, with or without an authorization.
        let legacy = r#"{"tool":"write_file","authorization":2,"success":true}"#;
        assert!(check_reserved_note("effect", legacy).is_err());
        assert!(check_reserved_note("effect", r#"{"tool":"write_file"}"#).is_err());
        // The ledger still reads one from an old journal.
        let l = Ledger::from_records(&[
            grant(2, Value::Null),
            rec(3, "effect", serde_json::from_str(legacy).unwrap()),
        ])
        .unwrap();
        assert_eq!(l.legacy_spent.get(&2), Some(&3));
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
        // sovereign-core #261: every authorization note is refused, whatever it says.
        for text in [
            g(json!({})),
            g(json!({"approved_grant": 1})),
            g(json!({"minted_grant": 1, "compose_commit": 1})),
            g(json!({"target": "/etc/N.md"})),
            json!({"control": "resume", "approver": "x"}).to_string(),
            "not json".to_string(),
        ] {
            let e = check_reserved_note("authorization", &text).unwrap_err();
            assert!(e.contains("sovereign-core #261"), "{e}");
        }
        assert!(check_reserved_note(
            "effect",
            &json!({"compose_commit": 1, "proposal_sha256": "p"}).to_string()
        )
        .is_err());

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
