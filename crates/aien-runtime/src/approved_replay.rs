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

//! Restart-safe replay protection for approved proposals (sovereign-core #249, part B).
//!
//! Every approved submission is claimed in the compose home's Cortex journal
//! BEFORE anything runs, and its lifecycle is written there as host records of
//! kind `effect` with an `approved_submission` field (never a `phase`, so the
//! effect ledger of crate::effects ignores them; ComposeNote cannot forge them,
//! see effects::check_reserved_note):
//!
//!   accepted      the claim: approval key, request id, approval id, trace id,
//!                 the claiming process (pid + start ticks). Durable before return.
//!   in_flight     written immediately before the compose run starts
//!   committed     the run committed exactly the approved proposal (evidence kept)
//!   failed        the run returned and reported that the World did not commit
//!   not_executed  the claim ended before any run started (refusal after the
//!                 claim, or a crash between accepted and in_flight)
//!   uncertain     the run may have changed the World but completion is not
//!                 proven (run error, committed-but-mismatch, crash in flight)
//!   declared      an operator settled an uncertain claim (what the operator saw);
//!                 it changes the reported state only
//!
//! Invariant: a claim key (approval key, request id, approval id; each on its
//! own) is accepted at most once, ever. Every later submission naming any of
//! them is refused with the claim's state, whatever that state is: an approval
//! is single-use even when its run never started, so no interrupted state is
//! ever guessed into a second run. A retry after a lost response gets the
//! committed evidence back in the refusal (`AlreadyCommitted`).
//!
//! Atomicity: the ledger is rebuilt from the journal under the compose-home
//! lock and the check and the `accepted` append happen in that one locked step
//! (the same pattern as effects::open_intent), so two concurrent submissions
//! cannot both be accepted. The journal is opened with CX_OPEN_SYNC: an
//! appended record is durable when `note` returns.
//!
//! The approval identity and its canonical key belong to crate::approved_auth
//! (`ApprovalIdentity`, `approval_key`); this module never recomputes them. It
//! takes the key and the two ids from the caller ([`ClaimKeys`]).
//!
//! Limits: the ledger reads at most HOST_RECALL_MAX host records in one recall
//! and refuses (fails closed) beyond that; liveness of a claiming process is
//! pid + start ticks (effects::executor_alive).
use crate::compose::{record_view, ComposeBridge, ComposeHome};
use crate::control::ComposeRecordView;
use crate::effects::{executor_alive, self_executor};
use aien_omega_compose::NoteKind;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;

/// The JSON field that marks a replay record (reserved against ComposeNote).
pub const FIELD: &str = "approved_submission";
const HOST_RECALL_MAX: u32 = 4096;

/// The keys one submission claims, taken from the authenticated identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimKeys {
    /// `approved_auth::approval_key(&identity)`: sha256 hex of the binding.
    pub approval_key: String,
    pub request_id: String,
    pub approval_id: String,
    /// Correlation only; not a claim key.
    pub trace_id: String,
}

impl ClaimKeys {
    fn keys(&self) -> [String; 3] {
        [
            format!("key:{}", self.approval_key),
            format!("request:{}", self.request_id),
            format!("approval:{}", self.approval_id),
        ]
    }
    fn check(&self) -> Result<(), Refusal> {
        let k = &self.approval_key;
        if k.len() != 64
            || !k
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Refusal::new(
                "BadKey",
                "approval_key is not 64 lowercase hex",
            ));
        }
        if self.request_id.is_empty() || self.approval_id.is_empty() {
            return Err(Refusal::new("BadKey", "empty request_id or approval_id"));
        }
        Ok(())
    }
}

/// What a committed run left behind (returned again on a retry).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommitEvidence {
    pub compose_proposal_sha256: String,
    pub cx_promotion: u64,
    pub cx_evidence: u64,
    pub task: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimState {
    Accepted,
    InFlight,
    Committed,
    Failed,
    NotExecuted,
    Uncertain,
}

impl ClaimState {
    pub fn name(self) -> &'static str {
        match self {
            Self::Accepted => "ACCEPTED",
            Self::InFlight => "IN_FLIGHT",
            Self::Committed => "COMMITTED",
            Self::Failed => "FAILED",
            Self::NotExecuted => "NOT_EXECUTED",
            Self::Uncertain => "UNCERTAIN",
        }
    }
    pub fn terminal(self) -> bool {
        !matches!(self, Self::Accepted | Self::InFlight)
    }
}

/// A named refusal: `REPLAY_REFUSED <Name>: ...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub name: &'static str,
    pub detail: String,
    /// The existing claim the refusal is about, when there is one.
    pub claim: Option<u64>,
    /// Set for AlreadyCommitted: the original result, for a caller whose
    /// response was lost.
    pub evidence: Option<CommitEvidence>,
}

impl Refusal {
    fn new(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            detail: detail.into(),
            claim: None,
            evidence: None,
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "REPLAY_REFUSED {}: {}", self.name, self.detail)
    }
}

/// A claim this process holds (the `accepted` record id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Claim {
    pub id: u64,
}

/// One claim as the ledger reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimRow {
    pub id: u64,
    pub keys: ClaimKeys,
    pub executor_pid: u32,
    pub executor_start: u64,
    pub state: ClaimState,
    /// The record that set `state` (the claim itself while ACCEPTED).
    pub state_record: u64,
    pub evidence: Option<CommitEvidence>,
    pub reason: Option<String>,
    /// Operator declaration on an UNCERTAIN claim: (record, "committed" | "not_committed").
    pub declared: Option<(u64, String)>,
}

/// The replay ledger, rebuilt from the journal's host records.
#[derive(Debug, Default)]
pub struct ReplayLedger {
    pub claims: BTreeMap<u64, ClaimRow>,
    /// claim key -> claim id
    index: BTreeMap<String, u64>,
}

fn corrupt(id: u64, why: impl Into<String>) -> Refusal {
    Refusal::new("CorruptLedger", format!("record #{id}: {}", why.into()))
}
fn s(v: &Value, k: &str, id: u64) -> Result<String, Refusal> {
    v.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| corrupt(id, format!("missing {k}")))
}
fn u(v: &Value, k: &str, id: u64) -> Result<u64, Refusal> {
    v.get(k)
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt(id, format!("missing {k}")))
}

impl ReplayLedger {
    pub fn from_records(host: &[ComposeRecordView]) -> Result<Self, Refusal> {
        let mut l = Self::default();
        for r in host {
            if !r.verified {
                return Err(corrupt(r.id, "digest does not verify"));
            }
            if r.note.as_deref() != Some("effect") {
                continue;
            }
            let Some(text) = r.text.as_deref() else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<Value>(text) else {
                continue;
            };
            let Some(phase) = v.get(FIELD) else { continue };
            let phase = phase
                .as_str()
                .ok_or_else(|| corrupt(r.id, "phase is not a string"))?;
            l.read(r.id, phase, &v)?;
        }
        Ok(l)
    }

    fn read(&mut self, id: u64, phase: &str, v: &Value) -> Result<(), Refusal> {
        if phase == "accepted" {
            let keys = ClaimKeys {
                approval_key: s(v, "approval_key", id)?,
                request_id: s(v, "request_id", id)?,
                approval_id: s(v, "approval_id", id)?,
                trace_id: s(v, "trace_id", id)?,
            };
            for k in keys.keys() {
                if let Some(prev) = self.index.insert(k.clone(), id) {
                    return Err(corrupt(id, format!("{k} already claimed by #{prev}")));
                }
            }
            let ex = v.get("executor").cloned().unwrap_or(Value::Null);
            self.claims.insert(
                id,
                ClaimRow {
                    id,
                    keys,
                    executor_pid: u(&ex, "pid", id)? as u32,
                    executor_start: u(&ex, "start", id)?,
                    state: ClaimState::Accepted,
                    state_record: id,
                    evidence: None,
                    reason: None,
                    declared: None,
                },
            );
            return Ok(());
        }
        let c = u(v, "claim", id)?;
        let row = self
            .claims
            .get_mut(&c)
            .ok_or_else(|| corrupt(id, format!("names claim #{c}, which is not one")))?;
        let reason = v.get("reason").and_then(Value::as_str).map(str::to_string);
        // Allowed transitions; anything else is a corrupt ledger (fail closed).
        let next = match (phase, row.state) {
            ("in_flight", ClaimState::Accepted) => ClaimState::InFlight,
            ("committed", ClaimState::InFlight) => ClaimState::Committed,
            ("failed", ClaimState::InFlight) => ClaimState::Failed,
            ("uncertain", ClaimState::InFlight) => ClaimState::Uncertain,
            ("not_executed", ClaimState::Accepted) => ClaimState::NotExecuted,
            ("declared", ClaimState::Uncertain) => {
                if row.declared.is_some() {
                    return Err(corrupt(id, format!("claim #{c} declared twice")));
                }
                let d = s(v, "declared", id)?;
                if d != "committed" && d != "not_committed" {
                    return Err(corrupt(
                        id,
                        "declared is neither committed nor not_committed",
                    ));
                }
                row.declared = Some((id, d));
                return Ok(());
            }
            (p, st) => {
                return Err(corrupt(
                    id,
                    format!("{p} after {} on claim #{c}", st.name()),
                ));
            }
        };
        if next == ClaimState::Committed {
            let e = v.get("evidence").cloned().unwrap_or(Value::Null);
            row.evidence =
                Some(serde_json::from_value(e).map_err(|e| corrupt(id, format!("evidence: {e}")))?);
        }
        row.state = next;
        row.state_record = id;
        row.reason = reason;
        Ok(())
    }

    /// The existing claim on any of these keys, if one exists.
    pub fn find(&self, keys: &ClaimKeys) -> Option<&ClaimRow> {
        keys.keys()
            .iter()
            .find_map(|k| self.index.get(k))
            .and_then(|id| self.claims.get(id))
    }

    /// Why a new claim on `keys` is refused (None: it may be claimed).
    pub fn check(&self, keys: &ClaimKeys) -> Option<Refusal> {
        let row = self.find(keys)?;
        let alive = executor_alive(row.executor_pid, row.executor_start);
        let (name, what): (&'static str, String) = match row.state {
            ClaimState::Committed => ("AlreadyCommitted", "committed; the original result is attached".into()),
            ClaimState::Failed => ("AlreadyFailed", "ran and did not commit; a new approval is needed".into()),
            ClaimState::NotExecuted => ("AlreadyConsumed", "claimed, never ran; a new approval is needed".into()),
            ClaimState::Uncertain => ("Uncertain", "may have committed; an operator must reconcile it, it is never re-run".into()),
            ClaimState::Accepted | ClaimState::InFlight if alive => ("InFlight", format!("{} in a running process", row.state.name())),
            // The claiming process is gone and nothing settled the claim yet:
            // before in_flight nothing ran; after it, completion is unknown.
            ClaimState::Accepted => ("AlreadyConsumed", "claimed by a process that is gone, never ran; a new approval is needed".into()),
            ClaimState::InFlight => ("Uncertain", "in flight in a process that is gone; an operator must reconcile it, it is never re-run".into()),
        };
        let mut r = Refusal::new(
            name,
            format!(
                "claim #{} (request {}, approval {}, key {}) is {}: {what}",
                row.id,
                row.keys.request_id,
                row.keys.approval_id,
                row.keys.approval_key,
                row.state.name()
            ),
        );
        r.claim = Some(row.id);
        r.evidence = row.evidence.clone();
        Some(r)
    }
}

fn ledger(home: &mut ComposeHome) -> Result<ReplayLedger, Refusal> {
    let (recs, total) = home
        .compose
        .recall(aien_omega_compose::SUBJECT_HOST, HOST_RECALL_MAX)
        .map_err(|e| Refusal::new("Journal", format!("host records: {e}")))?;
    if total > recs.len() as u64 {
        return Err(Refusal::new(
            "CorruptLedger",
            format!("{total} host records exceed the replay ledger's {HOST_RECALL_MAX}"),
        ));
    }
    let views: Vec<ComposeRecordView> = recs
        .iter()
        .map(|r| record_view(&mut home.compose, r))
        .collect();
    ReplayLedger::from_records(&views)
}

fn append(home: &mut ComposeHome, link: u64, text: &Value) -> Result<u64, Refusal> {
    let mut l = [0u64; 4];
    l[0] = link;
    home.compose
        .note(NoteKind::Effect, l, text.to_string().as_bytes())
        .map_err(|e| Refusal::new("Journal", format!("replay record: {e}")))
}

/// Run `f` under the compose-home lock; a lock/open/mark failure is a refusal.
fn locked<T>(
    b: &ComposeBridge,
    f: impl FnOnce(&mut ComposeHome) -> Result<T, Refusal>,
) -> Result<T, Refusal> {
    let mut inner: Option<Refusal> = None;
    let r = b.with_home(|home| {
        f(home).map_err(|e| {
            let s = e.to_string();
            inner = Some(e);
            s
        })
    });
    match r {
        Ok(t) => Ok(t),
        Err(e) => Err(inner.unwrap_or_else(|| Refusal::new("Journal", e))),
    }
}

/// Claim `keys` for one execution. Durable before it returns. Refused if any
/// key was ever claimed, with that claim's state.
pub fn claim(b: &ComposeBridge, keys: &ClaimKeys) -> Result<Claim, Refusal> {
    keys.check()?;
    locked(b, |home| {
        let l = ledger(home)?;
        if let Some(r) = l.check(keys) {
            return Err(r);
        }
        let (pid, start) = self_executor();
        if start == 0 {
            return Err(Refusal::new(
                "Executor",
                "cannot read this process's start ticks",
            ));
        }
        let text = json!({
            FIELD: "accepted",
            "approval_key": keys.approval_key, "request_id": keys.request_id,
            "approval_id": keys.approval_id, "trace_id": keys.trace_id,
            "executor": {"pid": pid, "start": start},
        });
        Ok(Claim {
            id: append(home, 0, &text)?,
        })
    })
}

/// Append one lifecycle record after checking the claim is ours and in `from`.
fn step(b: &ComposeBridge, c: &Claim, from: ClaimState, text: Value) -> Result<u64, Refusal> {
    locked(b, |home| {
        let l = ledger(home)?;
        let row = l
            .claims
            .get(&c.id)
            .ok_or_else(|| Refusal::new("NoClaim", format!("#{} is not a claim", c.id)))?;
        let (pid, start) = self_executor();
        if (row.executor_pid, row.executor_start) != (pid, start) {
            return Err(Refusal::new(
                "NotOwner",
                format!("claim #{} belongs to another process", c.id),
            ));
        }
        if row.state != from {
            return Err(Refusal::new(
                "WrongState",
                format!(
                    "claim #{} is {}, expected {}",
                    c.id,
                    row.state.name(),
                    from.name()
                ),
            ));
        }
        append(home, c.id, &text)
    })
}

/// Immediately before the compose run.
pub fn mark_in_flight(b: &ComposeBridge, c: &Claim) -> Result<(), Refusal> {
    step(
        b,
        c,
        ClaimState::Accepted,
        json!({FIELD: "in_flight", "claim": c.id}),
    )
    .map(|_| ())
}

/// The run committed exactly the approved proposal.
pub fn commit(b: &ComposeBridge, c: &Claim, e: &CommitEvidence) -> Result<(), Refusal> {
    let text = json!({FIELD: "committed", "claim": c.id, "evidence": e});
    step(b, c, ClaimState::InFlight, text).map(|_| ())
}

/// The run returned and reported that the World did not commit
/// (`ComposeTaskReport.committed == false`).
pub fn fail(b: &ComposeBridge, c: &Claim, reason: &str) -> Result<(), Refusal> {
    let text = json!({FIELD: "failed", "claim": c.id, "reason": reason});
    step(b, c, ClaimState::InFlight, text).map(|_| ())
}

/// The claim ends before the run started (any refusal after `claim`).
pub fn not_executed(b: &ComposeBridge, c: &Claim, reason: &str) -> Result<(), Refusal> {
    let text = json!({FIELD: "not_executed", "claim": c.id, "reason": reason});
    step(b, c, ClaimState::Accepted, text).map(|_| ())
}

/// The run may have changed the World and success is not proven: the run
/// returned an error, or committed something other than the approved proposal.
pub fn uncertain(b: &ComposeBridge, c: &Claim, reason: &str) -> Result<(), Refusal> {
    let text = json!({FIELD: "uncertain", "claim": c.id, "reason": reason});
    step(b, c, ClaimState::InFlight, text).map(|_| ())
}

/// Operator settles an UNCERTAIN claim after looking at the World. The claim
/// stays consumed either way.
pub fn declare(
    b: &ComposeBridge,
    claim: u64,
    committed: bool,
    by: &str,
    note: &str,
) -> Result<u64, Refusal> {
    locked(b, |home| {
        let l = ledger(home)?;
        let row = l
            .claims
            .get(&claim)
            .ok_or_else(|| Refusal::new("NoClaim", format!("#{claim} is not a claim")))?;
        if row.state != ClaimState::Uncertain || row.declared.is_some() {
            return Err(Refusal::new(
                "WrongState",
                format!(
                    "claim #{claim} is {} (declared: {})",
                    row.state.name(),
                    row.declared.is_some()
                ),
            ));
        }
        let d = if committed {
            "committed"
        } else {
            "not_committed"
        };
        append(
            home,
            claim,
            &json!({FIELD: "declared", "claim": claim, "declared": d, "by": by, "note": note}),
        )
    })
}

/// Daemon start: settle every claim whose process is gone. ACCEPTED (never
/// ran) -> not_executed; IN_FLIGHT -> uncertain. Nothing is re-run.
pub fn reconcile_at_start(b: &ComposeBridge) -> Result<String, Refusal> {
    if !b.dir().join("cortex.cx").exists() {
        return Ok("Replay reconcile: no compose home yet".into());
    }
    locked(b, |home| {
        let l = ledger(home)?;
        let (mut ne, mut un) = (0, 0);
        for row in l.claims.values() {
            if row.state.terminal() || executor_alive(row.executor_pid, row.executor_start) {
                continue;
            }
            let (phase, why) = match row.state {
                ClaimState::Accepted => {
                    ne += 1;
                    (
                        "not_executed",
                        "claiming process gone before the run started (reconcile@start)",
                    )
                }
                _ => {
                    un += 1;
                    (
                        "uncertain",
                        "claiming process gone while in flight (reconcile@start)",
                    )
                }
            };
            append(
                home,
                row.id,
                &json!({FIELD: phase, "claim": row.id, "reason": why, "by": "reconcile@start"}),
            )?;
        }
        Ok(format!(
            "Replay reconcile: {} claims, {ne} -> NOT_EXECUTED, {un} -> UNCERTAIN",
            l.claims.len()
        ))
    })
}
