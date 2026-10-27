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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use aien_capability::{
    catalog_digest, speculation_safe, Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor,
    ToolEffects, WorldId,
};
use aien_trace::{
    now_unix_ms, BoundedId, BoundedNote, CorrelationIds, EffectCertainty, EventKind, EventStatus,
    EvidenceRefs, NullSink, TraceEvent, TraceId, TraceSink,
};
use serde_json::Value;

use crate::approval::{ApprovalError, ApprovalGrant, ApprovalRecord, GrantState, Reservation};
use crate::authority::{
    intent_digest, AuthorityContext, AuthorityDecision, AuthorityOutcome, EffectAuthority,
    EffectScope, Exposure,
};
use crate::effect::{mint, mint_reserved};
use crate::{
    AuthorizedEffect, CallOutcome, CapabilitySnapshot, EffectIntent, EffectReceipt, Error, McpWire,
    ToolResult,
};

const SNAPSHOT_TTL: Duration = Duration::from_secs(60);

/// Live MCP connection. Private to this crate: callers receive snapshots and receipts.
struct McpSession {
    epoch: u64,
    wire: Arc<dyn McpWire>,
    tools: Vec<ToolDescriptor>,
    catalog_digest: Digest32,
}

enum LedgerEntry {
    InFlight,
    Completed(EffectReceipt),
    Uncertain,
}

/// Which effect owns a ledger key: the intent digest (provider, tool, arguments, catalog), the
/// world and the winning J-node. The same four things an approval binds, so a key cannot be
/// replayed as a receipt for a different effect.
#[derive(Clone, Copy, PartialEq, Eq)]
struct EffectIdentity {
    intent: Digest32,
    world_id: WorldId,
    winning_jnode: JNodeId,
}

struct LedgerRow {
    identity: EffectIdentity,
    entry: LedgerEntry,
}

pub(crate) struct Inner {
    sessions: HashMap<String, McpSession>,
    ledger: HashMap<EffectId, LedgerRow>,
    pub(crate) approvals: HashMap<Digest32, ApprovalRecord>,
    pub(crate) next_approval: u64,
    pub(crate) next_reservation: u64,
}

/// Owns admitted MCP sessions and the idempotency ledger.
#[derive(Clone)]
pub struct McpBroker {
    inner: Arc<Mutex<Inner>>,
}

impl McpBroker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                sessions: HashMap::new(),
                ledger: HashMap::new(),
                approvals: HashMap::new(),
                next_approval: 0,
                next_reservation: 0,
            })),
        }
    }

    /// Enroll a provider. Speculative branches cannot do this.
    pub async fn admit(&self, provider: ProviderId, wire: Arc<dyn McpWire>) -> Result<(), Error> {
        let tools = wire.list_tools().await?;
        let session = McpSession {
            epoch: 1,
            catalog_digest: catalog_digest(&tools),
            tools,
            wire,
        };
        let mut inner = self.lock();
        inner
            .sessions
            .insert(provider.as_str().to_string(), session);
        Ok(())
    }

    /// Replace the cached catalog after the provider's schema changes.
    pub async fn replace_catalog(
        &self,
        provider: &ProviderId,
        tools: Vec<ToolDescriptor>,
    ) -> Result<(), Error> {
        let mut inner = self.lock();
        let session = inner
            .sessions
            .get_mut(provider.as_str())
            .ok_or_else(|| Error::NotAdmitted(provider.clone()))?;
        let digest = catalog_digest(&tools);
        if digest != session.catalog_digest {
            session.epoch = session.epoch.saturating_add(1);
            session.catalog_digest = digest;
            session.tools = tools;
        }
        Ok(())
    }

    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for McpBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// Lane J-Space is allowed to hold.
#[derive(Clone)]
pub struct SpeculativeLane {
    broker: McpBroker,
}

impl SpeculativeLane {
    pub fn new(broker: McpBroker) -> Self {
        Self { broker }
    }

    /// Refresh an already admitted provider and return the snapshot.
    ///
    /// This does not dial. A provider missing from the session map is
    /// `NotAdmitted`. Opening a transport is enrollment on [`crate::SessionManager`].
    pub async fn discovery_snapshot(
        &self,
        provider: &ProviderId,
    ) -> Result<CapabilitySnapshot, Error> {
        let wire = {
            let inner = self.broker.lock();
            let session = inner
                .sessions
                .get(provider.as_str())
                .ok_or_else(|| Error::NotAdmitted(provider.clone()))?;
            session.wire.clone()
        };
        let tools = wire.list_tools().await?;
        let mut inner = self.broker.lock();
        let session = inner
            .sessions
            .get_mut(provider.as_str())
            .ok_or_else(|| Error::NotAdmitted(provider.clone()))?;
        let digest = catalog_digest(&tools);
        if digest != session.catalog_digest {
            session.epoch = session.epoch.saturating_add(1);
            session.catalog_digest = digest;
        }
        session.tools = tools;
        let generated_at = SystemTime::now();
        Ok(CapabilitySnapshot {
            provider: provider.clone(),
            session_epoch: session.epoch,
            catalog_digest: session.catalog_digest,
            tools: session.tools.clone().into(),
            generated_at,
            expires_at: generated_at + SNAPSHOT_TTL,
        })
    }

    pub async fn invoke_speculative(&self, call: SpeculativeToolCall) -> Result<ToolResult, Error> {
        let wire = {
            let inner = self.broker.lock();
            let session = inner
                .sessions
                .get(call.provider.as_str())
                .ok_or_else(|| Error::NotAdmitted(call.provider.clone()))?;
            let tool = find_tool(session, &call.tool_name).ok_or_else(|| Error::UnknownTool {
                provider: call.provider.clone(),
                tool: call.tool_name.clone(),
            })?;
            if !speculation_safe(tool.effects()) {
                return Err(Error::EffectRequiresCommit);
            }
            if tool
                .effects()
                .intersects(ToolEffects::SPAWN_PROCESS | ToolEffects::LOCAL_EPHEMERAL)
                && !call.sandboxed
            {
                return Err(Error::SandboxRequired);
            }
            session.wire.clone()
        };
        match wire.call_tool(&call.tool_name, &call.arguments).await? {
            CallOutcome::Finished(output) => Ok(ToolResult { output }),
            CallOutcome::Rejected(reason) => Err(Error::Rejected(reason)),
            CallOutcome::Uncertain => Err(Error::ReconciliationRequired),
        }
    }

    pub async fn stage_effect_intent(
        &self,
        provider: &ProviderId,
        tool_name: &str,
        arguments: Value,
    ) -> Result<EffectIntent, Error> {
        let inner = self.broker.lock();
        let session = inner
            .sessions
            .get(provider.as_str())
            .ok_or_else(|| Error::NotAdmitted(provider.clone()))?;
        let tool = find_tool(session, tool_name).ok_or_else(|| Error::UnknownTool {
            provider: provider.clone(),
            tool: tool_name.to_string(),
        })?;
        if speculation_safe(tool.effects()) {
            return Err(Error::SpeculationSafe);
        }
        Ok(EffectIntent {
            provider: provider.clone(),
            tool_name: tool_name.to_string(),
            arguments,
            capability_digest: session.catalog_digest,
        })
    }
}

/// Request J-Space may make on the speculative lane.
pub struct SpeculativeToolCall {
    pub provider: ProviderId,
    pub tool_name: String,
    pub arguments: Value,
    pub sandboxed: bool,
}

/// Lane the Effect Broker holds. `execute_effect` accepts only an authorized effect.
#[derive(Clone)]
pub struct EffectLane {
    broker: McpBroker,
    exposure: Option<Exposure>,
    clock: Option<Clock>,
    trace: LaneTrace,
}

/// Correlation tap of one lane. Observational: it never changes a verdict.
#[derive(Clone)]
struct LaneTrace {
    sink: Arc<dyn TraceSink>,
    trace_id: TraceId,
    parent: Option<u64>,
}

impl LaneTrace {
    fn off() -> Self {
        Self {
            sink: Arc::new(NullSink),
            trace_id: TraceId([0; 16]),
            parent: None,
        }
    }

    fn enabled(&self) -> bool {
        self.sink.enabled()
    }

    fn emit(
        &self,
        kind: EventKind,
        status: EventStatus,
        certainty: Option<EffectCertainty>,
        ids: CorrelationIds,
        refs: EvidenceRefs,
        note: Option<&str>,
    ) {
        self.sink.emit(TraceEvent {
            trace_id: self.trace_id,
            event_id: self.sink.next_event_id(),
            parent_event_id: self.parent,
            at_unix_ms: now_unix_ms(),
            kind,
            status,
            effect_certainty: certainty,
            ids,
            refs,
            note: note.map(BoundedNote::new),
        });
    }
}

const STALE_INTENT: &str = "intent was staged against a stale catalog";
const NOT_ADMITTED: &str = "provider is not admitted";
/// The lane has no way to enforce a restriction list, so a non-empty one is refused rather than
/// minted as if it were a plain `Allow`.
const UNENFORCED_RESTRICTIONS: &str =
    "restricted allow refused: the lane cannot enforce restrictions";

fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

// Canonical, length-prefixed encoding of a JSON value: object keys sorted, so the
// result does not depend on serde_json's map order feature.
fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.push(0),
        Value::Bool(b) => out.extend_from_slice(&[1, *b as u8]),
        Value::Number(n) => {
            out.push(2);
            put_str(out, &n.to_string());
        }
        Value::String(s) => {
            out.push(3);
            put_str(out, s);
        }
        Value::Array(a) => {
            out.push(4);
            out.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for x in a {
                put_value(out, x);
            }
        }
        Value::Object(m) => {
            out.push(5);
            out.extend_from_slice(&(m.len() as u64).to_le_bytes());
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for k in keys {
                put_str(out, k);
                put_value(out, &m[k]);
            }
        }
    }
}

/// SHA-256 over a canonical encoding of every receipt field: a domain tag, then fixed-width
/// ids and digests, then length-prefixed strings, then the output value with sorted object
/// keys. A trace carries this digest, never the output itself. The independent receipt
/// verifier, not this digest, governs whether a receipt is believed.
pub fn receipt_digest(r: &EffectReceipt) -> [u8; 32] {
    let mut b = Vec::new();
    b.extend_from_slice(b"aien-mcp.effect-receipt.v1\0");
    b.extend_from_slice(&r.effect_id.0);
    b.extend_from_slice(&r.world_id.0.to_le_bytes());
    b.extend_from_slice(&r.winning_jnode.0.to_le_bytes());
    b.extend_from_slice(&r.policy_digest.0);
    put_str(&mut b, r.provider.as_str());
    put_str(&mut b, &r.tool_name);
    b.extend_from_slice(&r.capability_digest.0);
    put_value(&mut b, &r.output);
    Digest32::of(&b).0
}

/// Host-supplied time source, in the same unit as `now` and `expires_at`.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

impl EffectLane {
    pub fn new(broker: McpBroker) -> Self {
        Self {
            broker,
            exposure: None,
            clock: None,
            trace: LaneTrace::off(),
        }
    }

    /// A lane whose `execute_effect` re-checks approval expiry at commit using `clock`. The crate
    /// reads no wall clock itself. A lane without a clock refuses to run any effect that holds an
    /// approval reservation (`Error::ApprovalClockMissing`) and releases the grant. Effects
    /// without a reservation do not need a clock.
    pub fn with_clock(&self, clock: Clock) -> Self {
        Self {
            clock: Some(clock),
            ..self.clone()
        }
    }

    /// A lane that tells authorities what the model was exposed to. Host/runtime call only:
    /// an intent cannot carry exposure. A lane built with [`Self::new`] has no exposure, which
    /// [`crate::EffectClassAuthority`] treats as `external_untrusted` for effect classes (fail closed).
    pub fn with_exposure(&self, exposure: Exposure) -> Self {
        Self {
            broker: self.broker.clone(),
            exposure: Some(exposure),
            clock: self.clock.clone(),
            trace: self.trace.clone(),
        }
    }

    /// Attach a correlation trace (observational only). Every `authority_decided` and
    /// `effect_executed` event of this lane becomes a child of `parent_event` in `trace_id`. Events
    /// carry digests and ids, never arguments or outputs. The default is a `NullSink`.
    pub fn with_trace(
        self,
        sink: Arc<dyn TraceSink>,
        trace_id: TraceId,
        parent_event: Option<u64>,
    ) -> Self {
        Self {
            trace: LaneTrace {
                sink,
                trace_id,
                parent: parent_event,
            },
            ..self
        }
    }

    /// Ask `authority` whether `intent` may run, and mint an `AuthorizedEffect` only on
    /// `Allow` or an `AllowRestricted` with no restrictions. This is the only production path to
    /// an `AuthorizedEffect`. A non-empty restriction list is denied: nothing here can enforce it.
    ///
    /// The lane, not the caller, reads the live descriptor for the authority. A stale intent
    /// (digest differs from the live catalog) is denied before the authority is asked.
    /// `RequireApproval` returns `Pending` with the intent digest; no approval is consumed here
    /// (see [`Self::authorize_approved`]). `Deny` and `Contain` return errors.
    pub fn authorize(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
    ) -> Result<AuthorizedEffect<EffectIntent>, AuthorityOutcome> {
        self.authorize_inner(intent, scope, authority, None)
    }

    /// Like [`Self::authorize`], but a `RequireApproval` verdict is satisfied by spending `grant`.
    ///
    /// The grant must have been issued by an [`crate::ApprovalDesk`] for exactly this intent and
    /// scope, must be unspent, unrevoked and unexpired (`now < expires_at`). A refused grant is not spent
    /// unless it was already spent. `Deny` and `Contain` stay refused whatever the grant says,
    /// and an `Allow` verdict never touches the grant.
    pub fn authorize_approved(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
        grant: &ApprovalGrant,
        now: u64,
    ) -> Result<AuthorizedEffect<EffectIntent>, AuthorityOutcome> {
        self.authorize_inner(intent, scope, authority, Some((grant, now)))
    }

    fn authorize_inner(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
        approval: Option<(&ApprovalGrant, u64)>,
    ) -> Result<AuthorizedEffect<EffectIntent>, AuthorityOutcome> {
        if !self.trace.enabled() {
            return self.authorize_core(intent, scope, authority, approval);
        }
        let digest = intent_digest(&intent);
        let mut ids = self.effect_ids(&intent.provider, &intent.tool_name, scope.world_id);
        let presented = approval.map(|(g, _)| hex(&g.id().0));
        let result = self.authorize_core(intent, scope, authority, approval);
        let (status, note): (EventStatus, Option<&str>) = match &result {
            Ok(_) => (EventStatus::Ok, None),
            Err(AuthorityOutcome::Pending { .. }) => (EventStatus::ApprovalPending, None),
            Err(AuthorityOutcome::Denied(r)) if r == STALE_INTENT => {
                (EventStatus::Rejected, Some("stale_intent"))
            }
            Err(AuthorityOutcome::Denied(r)) if r == NOT_ADMITTED => {
                (EventStatus::Rejected, Some("provider_not_admitted"))
            }
            Err(AuthorityOutcome::Denied(_)) => (EventStatus::Rejected, None),
            Err(AuthorityOutcome::Contained(_)) => (EventStatus::Rejected, Some("contained")),
            Err(AuthorityOutcome::Approval(e)) => (EventStatus::Rejected, Some(approval_name(e))),
            Err(AuthorityOutcome::Execution(_)) => (EventStatus::Failed, None),
        };
        let grant = match &result {
            Ok(e) => e.approval_id().map(|d| hex(&d.0)),
            Err(_) => presented,
        };
        if let Some(grant) = grant {
            ids.grant_id = Some(BoundedId::new(&grant));
        }
        let refs = EvidenceRefs {
            intent_digest: aien_trace::Digest::new(&hex(&digest.0)).ok(),
            ..Default::default()
        };
        self.trace
            .emit(EventKind::AuthorityDecided, status, None, ids, refs, note);
        result
    }

    fn effect_ids(&self, provider: &ProviderId, tool: &str, world: WorldId) -> CorrelationIds {
        let mut name = String::with_capacity(provider.as_str().len() + 1 + tool.len());
        name.push_str(provider.as_str());
        name.push(':');
        name.push_str(tool);
        CorrelationIds {
            world_id: Some(world.0),
            tool_request_id: Some(BoundedId::new(&name)),
            ..Default::default()
        }
    }

    fn authorize_core(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
        approval: Option<(&ApprovalGrant, u64)>,
    ) -> Result<AuthorizedEffect<EffectIntent>, AuthorityOutcome> {
        let (descriptor, live) = {
            let inner = self.broker.lock();
            let session = inner
                .sessions
                .get(intent.provider.as_str())
                .ok_or_else(|| AuthorityOutcome::Denied(NOT_ADMITTED.into()))?;
            (
                find_tool(session, &intent.tool_name).cloned(),
                session.catalog_digest,
            )
        };
        if intent.capability_digest != live {
            return Err(AuthorityOutcome::Denied(STALE_INTENT.into()));
        }
        let ctx = AuthorityContext::new(scope, descriptor, live, self.exposure.clone());
        let decision = authority.authorize(&intent, &ctx);
        let policy_digest = {
            let mut bytes = format!("{decision:?}|{}|", intent.tool_name).into_bytes();
            bytes.extend_from_slice(&live.0);
            Digest32::of(&bytes)
        };
        match decision {
            AuthorityDecision::AllowRestricted(ref restrictions) if !restrictions.is_empty() => {
                Err(AuthorityOutcome::Denied(UNENFORCED_RESTRICTIONS.into()))
            }
            AuthorityDecision::Allow | AuthorityDecision::AllowRestricted(_) => Ok(mint(
                intent,
                scope.world_id,
                scope.winning_jnode,
                policy_digest,
                live,
                scope.idempotency_key,
            )),
            AuthorityDecision::RequireApproval(reason) => {
                let Some((grant, now)) = approval else {
                    return Err(AuthorityOutcome::Pending {
                        intent_digest: intent_digest(&intent),
                        reason,
                    });
                };
                let reservation = self
                    .reserve(grant, &intent, &scope, now)
                    .map_err(AuthorityOutcome::Approval)?;
                Ok(mint_reserved(
                    intent,
                    scope.world_id,
                    scope.winning_jnode,
                    policy_digest,
                    live,
                    scope.idempotency_key,
                    reservation,
                ))
            }
            AuthorityDecision::Deny(reason) => Err(AuthorityOutcome::Denied(reason)),
            AuthorityDecision::Contain(reason) => Err(AuthorityOutcome::Contained(reason)),
        }
    }

    /// `authorize`, then `execute_effect`. Nothing reaches the provider unless authorized.
    pub async fn authorize_and_execute(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
    ) -> Result<EffectReceipt, AuthorityOutcome> {
        let effect = self.authorize(intent, scope, authority)?;
        self.execute_effect(effect)
            .await
            .map_err(AuthorityOutcome::Execution)
    }

    /// `authorize_approved`, then `execute_effect`.
    ///
    /// If `grant` is already spent for this same effect (same intent, key, world, J-node), the
    /// request is a replay: the ledger's existing outcome is returned and nothing is minted.
    /// Any other use of a spent grant is refused with `ApprovalError::Consumed`.
    pub async fn authorize_and_execute_approved(
        &self,
        intent: EffectIntent,
        scope: EffectScope,
        authority: &dyn EffectAuthority,
        grant: &ApprovalGrant,
        now: u64,
    ) -> Result<EffectReceipt, AuthorityOutcome> {
        if let Some(existing) = self.replay_of(grant, &intent, &scope) {
            if self.trace.enabled() {
                let mut ids = self.effect_ids(&intent.provider, &intent.tool_name, scope.world_id);
                ids.grant_id = Some(BoundedId::new(&hex(&grant.id().0)));
                self.trace_execution(&existing, true, ids, Some(intent_digest(&intent)));
            }
            return existing.map_err(AuthorityOutcome::Execution);
        }
        let effect = self.authorize_approved(intent, scope, authority, grant, now)?;
        self.execute_effect(effect)
            .await
            .map_err(AuthorityOutcome::Execution)
    }

    /// Ledger outcome for a spent grant presented again for the effect it was spent on.
    fn replay_of(
        &self,
        grant: &ApprovalGrant,
        intent: &EffectIntent,
        scope: &EffectScope,
    ) -> Option<Result<EffectReceipt, Error>> {
        let inner = self.broker.lock();
        let record = inner.approvals.get(&grant.id())?;
        if record.state != GrantState::Spent || !record.binds(intent, scope) {
            return None;
        }
        let row = inner.ledger.get(&scope.idempotency_key)?;
        if row.identity != identity(intent, scope.world_id, scope.winning_jnode) {
            return Some(Err(Error::IdempotencyConflict));
        }
        match &row.entry {
            LedgerEntry::Completed(receipt) => Some(Ok(receipt.clone())),
            LedgerEntry::Uncertain => Some(Err(Error::ReconciliationRequired)),
            LedgerEntry::InFlight => Some(Err(Error::EffectInFlight)),
        }
    }

    /// Check and reserve `grant` atomically under the broker lock. The returned reservation is
    /// committed by `execute_effect` and released if the effect is dropped or cancelled.
    fn reserve(
        &self,
        grant: &ApprovalGrant,
        intent: &EffectIntent,
        scope: &EffectScope,
        now: u64,
    ) -> Result<Reservation, ApprovalError> {
        let mut inner = self.broker.lock();
        let record = inner
            .approvals
            .get_mut(&grant.id())
            .ok_or(ApprovalError::Unknown)?;
        if !record.binds(intent, scope) {
            return Err(ApprovalError::Mismatch);
        }
        match record.state {
            GrantState::Spent => return Err(ApprovalError::Consumed),
            GrantState::Reserved(_) => return Err(ApprovalError::Reserved),
            GrantState::Available => {}
        }
        if record.revoked {
            return Err(ApprovalError::Revoked);
        }
        if now >= record.expires_at {
            return Err(ApprovalError::Expired);
        }
        let token = {
            inner.next_reservation = inner.next_reservation.saturating_add(1);
            inner.next_reservation
        };
        let record = inner
            .approvals
            .get_mut(&grant.id())
            .ok_or(ApprovalError::Unknown)?;
        record.state = GrantState::Reserved(token);
        record.release_reason = None;
        Ok(Reservation::new(self.broker.clone(), grant.id(), token))
    }

    pub async fn execute_effect(
        &self,
        effect: AuthorizedEffect<EffectIntent>,
    ) -> Result<EffectReceipt, Error> {
        if !self.trace.enabled() {
            return self.execute_core(effect, &mut false).await;
        }
        let mut ids = self.effect_ids(
            &effect.intent().provider,
            &effect.intent().tool_name,
            effect.world_id(),
        );
        ids.grant_id = effect.approval_id().map(|d| BoundedId::new(&hex(&d.0)));
        let digest = intent_digest(effect.intent());
        let mut replay = false;
        let result = self.execute_core(effect, &mut replay).await;
        self.trace_execution(&result, replay, ids, Some(digest));
        result
    }

    fn trace_execution(
        &self,
        result: &Result<EffectReceipt, Error>,
        replay: bool,
        ids: CorrelationIds,
        intent: Option<Digest32>,
    ) {
        let mut refs = EvidenceRefs {
            intent_digest: intent.and_then(|d| aien_trace::Digest::new(&hex(&d.0)).ok()),
            ..Default::default()
        };
        let (status, certainty, note) = match result {
            Ok(receipt) => {
                refs.effect_receipt_digest =
                    aien_trace::Digest::new(&hex(&receipt_digest(receipt))).ok();
                let note = replay.then_some("ledger_replay");
                (EventStatus::Ok, Some(EffectCertainty::EffectOccurred), note)
            }
            Err(Error::Rejected(_)) => {
                (EventStatus::Rejected, Some(EffectCertainty::NoEffect), None)
            }
            Err(Error::ReconciliationRequired) => (
                EventStatus::Uncertain,
                Some(EffectCertainty::Uncertain),
                None,
            ),
            Err(Error::EffectInFlight) => (EventStatus::Rejected, None, Some("in_flight")),
            Err(_) => (
                EventStatus::Rejected,
                Some(EffectCertainty::NoEffect),
                Some("refused"),
            ),
        };
        self.trace.emit(
            EventKind::EffectExecuted,
            status,
            certainty,
            ids,
            refs,
            note,
        );
    }

    async fn execute_core(
        &self,
        effect: AuthorizedEffect<EffectIntent>,
        replay: &mut bool,
    ) -> Result<EffectReceipt, Error> {
        let key = effect.idempotency_key();
        let provider = effect.intent().provider.clone();
        let tool_name = effect.intent().tool_name.clone();
        let arguments = effect.intent().arguments.clone();
        let staged_catalog = effect.intent().capability_digest;
        let who = identity(effect.intent(), effect.world_id(), effect.winning_jnode());
        let wire = {
            let mut inner = self.broker.lock();
            if let Some(existing) = inner.ledger.get(&key) {
                // A key names one effect. A request for another effect under the same key did not
                // run and must not be answered with the first effect's receipt.
                if existing.identity != who {
                    return Err(Error::IdempotencyConflict);
                }
                return match &existing.entry {
                    LedgerEntry::Completed(receipt) => {
                        *replay = true;
                        Ok(receipt.clone())
                    }
                    LedgerEntry::Uncertain => Err(Error::ReconciliationRequired),
                    LedgerEntry::InFlight => Err(Error::EffectInFlight),
                };
            }
            let wire = {
                let session = inner
                    .sessions
                    .get(provider.as_str())
                    .ok_or_else(|| Error::NotAdmitted(provider.clone()))?;
                if session.catalog_digest != effect.capability_digest()
                    || staged_catalog != effect.capability_digest()
                {
                    return Err(Error::StaleCapability);
                }
                if find_tool(session, &tool_name).is_none() {
                    return Err(Error::UnknownTool {
                        provider: provider.clone(),
                        tool: tool_name.clone(),
                    });
                }
                session.wire.clone()
            };
            // Point of no return: the provider is about to be called. Until here a refusal drops
            // the effect and releases its grant; from here the grant is spent.
            if let Some(reservation) = effect.reservation() {
                let now = self.clock.as_ref().map(|c| c());
                reservation.commit_locked(&mut inner, now)?;
            }
            inner.ledger.insert(
                key,
                LedgerRow {
                    identity: who,
                    entry: LedgerEntry::InFlight,
                },
            );
            wire
        };

        let outcome = wire.call_tool(&tool_name, &arguments).await;
        let mut inner = self.broker.lock();
        match outcome {
            Ok(CallOutcome::Finished(output)) => {
                let receipt = EffectReceipt {
                    effect_id: key,
                    world_id: effect.world_id(),
                    winning_jnode: effect.winning_jnode(),
                    policy_digest: effect.policy_digest(),
                    provider,
                    tool_name,
                    capability_digest: effect.capability_digest(),
                    output,
                };
                inner.ledger.insert(
                    key,
                    LedgerRow {
                        identity: who,
                        entry: LedgerEntry::Completed(receipt.clone()),
                    },
                );
                Ok(receipt)
            }
            Ok(CallOutcome::Uncertain) | Err(_) => {
                inner.ledger.insert(
                    key,
                    LedgerRow {
                        identity: who,
                        entry: LedgerEntry::Uncertain,
                    },
                );
                Err(Error::ReconciliationRequired)
            }
            Ok(CallOutcome::Rejected(reason)) => {
                inner.ledger.remove(&key);
                Err(Error::Rejected(reason))
            }
        }
    }
}

fn identity(intent: &EffectIntent, world_id: WorldId, winning_jnode: JNodeId) -> EffectIdentity {
    EffectIdentity {
        intent: intent_digest(intent),
        world_id,
        winning_jnode,
    }
}

fn find_tool<'a>(session: &'a McpSession, name: &str) -> Option<&'a ToolDescriptor> {
    session.tools.iter().find(|tool| tool.name() == name)
}

fn approval_name(e: &ApprovalError) -> &'static str {
    match e {
        ApprovalError::Unknown => "approval_unknown",
        ApprovalError::Mismatch => "approval_mismatch",
        ApprovalError::Consumed => "approval_consumed",
        ApprovalError::Reserved => "approval_reserved",
        ApprovalError::Expired => "approval_expired",
        ApprovalError::Revoked => "approval_revoked",
    }
}
