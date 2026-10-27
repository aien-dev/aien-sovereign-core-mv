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

//! L3-AUTH negative controls: what an approval binds, what the ledger binds, and what a faulty
//! provider can and cannot cause. Public API only. Every assertion that matters counts provider
//! executions (`calls`), not return values.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use aien_mcp::memory::MemoryWire;
use aien_mcp::{
    ApprovalDesk, ApprovalError, AuthorityOutcome, CallOutcome, EffectClassAuthority, EffectIntent,
    EffectLane, EffectScope, Error, McpBroker, McpWire, SpeculativeLane,
};
use serde_json::{json, Value};

fn scope(label: &str) -> EffectScope {
    EffectScope {
        world_id: WorldId(1),
        winning_jnode: JNodeId(1),
        idempotency_key: EffectId::from_label(label),
    }
}

struct H {
    broker: McpBroker,
    lane: EffectLane,
    desk: ApprovalDesk,
    calls: Arc<AtomicUsize>,
    provider: ProviderId,
    digest: Digest32,
}

impl H {
    fn intent(&self, tool: &str, arguments: Value) -> EffectIntent {
        EffectIntent {
            provider: self.provider.clone(),
            tool_name: tool.into(),
            arguments,
            capability_digest: self.digest,
        }
    }
}

/// Provider with a write-class tool `w`, a second write-class tool `x`, and a local-ephemeral
/// tool `e` (which `EffectClassAuthority` allows without approval). `wire` decides every call.
async fn harness_with(wire: Arc<dyn McpWire>, calls: Arc<AtomicUsize>) -> H {
    let broker = McpBroker::new();
    let provider = ProviderId::new("p");
    broker.admit(provider.clone(), wire).await.unwrap();
    let digest = SpeculativeLane::new(broker.clone())
        .discovery_snapshot(&provider)
        .await
        .unwrap()
        .catalog_digest;
    H {
        lane: EffectLane::new(broker.clone()).with_clock(std::sync::Arc::new(|| 1)),
        desk: ApprovalDesk::new(broker.clone()),
        broker,
        calls,
        provider,
        digest,
    }
}

fn tools() -> Vec<ToolDescriptor> {
    vec![
        ToolDescriptor::new("w", ToolEffects::WORLD_MUTATION, Digest32::of(b"w")),
        ToolDescriptor::new("x", ToolEffects::WORLD_MUTATION, Digest32::of(b"x")),
        ToolDescriptor::new("e", ToolEffects::LOCAL_EPHEMERAL, Digest32::of(b"e")),
    ]
}

async fn harness(outcome: CallOutcome) -> H {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let wire = MemoryWire::new(tools(), move |_, _| {
        c.fetch_add(1, Ordering::SeqCst);
        outcome.clone()
    });
    harness_with(Arc::new(wire), calls).await
}

fn approval(e: AuthorityOutcome) -> ApprovalError {
    match e {
        AuthorityOutcome::Approval(a) => a,
        other => panic!("expected Approval, got {other:?}"),
    }
}

// ---- what a grant binds ----

#[tokio::test]
async fn grant_is_bound_to_the_winning_jnode() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("j"), 100);
    let mut other = scope("j");
    other.winning_jnode = JNodeId(2);
    let e = h
        .lane
        .authorize_approved(intent.clone(), other, &EffectClassAuthority, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Mismatch);
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    // Not burned: the named effect still runs, once.
    let r = h
        .lane
        .authorize_and_execute_approved(intent, scope("j"), &EffectClassAuthority, &g, 1)
        .await;
    assert!(r.is_ok());
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn grant_is_bound_to_the_tool() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let g = h
        .desk
        .issue(&h.intent("w", json!({"a": 1})), scope("t"), 100);
    // Same arguments, same key, same catalog, other tool of the same effect class.
    let e = h
        .lane
        .authorize_and_execute_approved(
            h.intent("x", json!({"a": 1})),
            scope("t"),
            &EffectClassAuthority,
            &g,
            1,
        )
        .await
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Mismatch);
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

// ---- single use under concurrency ----

#[tokio::test]
async fn one_grant_spent_from_many_threads_mints_exactly_once() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("race"), 100);
    let minted = AtomicUsize::new(0);
    let refused = AtomicUsize::new(0);
    // Winners are kept alive: a dropped effect would release the grant for the next thread.
    let held = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                match h.lane.authorize_approved(
                    intent.clone(),
                    scope("race"),
                    &EffectClassAuthority,
                    &g,
                    1,
                ) {
                    Ok(e) => {
                        held.lock().unwrap().push(e);
                        minted.fetch_add(1, Ordering::SeqCst)
                    }
                    Err(AuthorityOutcome::Approval(ApprovalError::Reserved)) => {
                        refused.fetch_add(1, Ordering::SeqCst)
                    }
                    Err(other) => panic!("unexpected {other:?}"),
                };
            });
        }
    });
    assert_eq!(minted.load(Ordering::SeqCst), 1);
    assert_eq!(refused.load(Ordering::SeqCst), 15);
}

#[test]
fn one_grant_executed_from_many_threads_reaches_the_provider_once() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let h = rt.block_on(harness(CallOutcome::Finished(json!({"ok": true}))));
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("race2"), 100);
    let receipts = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                let r = rt.block_on(h.lane.authorize_and_execute_approved(
                    intent.clone(),
                    scope("race2"),
                    &EffectClassAuthority,
                    &g,
                    1,
                ));
                if let Ok(rc) = r {
                    receipts.lock().unwrap().push(rc);
                }
            });
        }
    });
    assert_eq!(h.calls.load(Ordering::SeqCst), 1, "provider reached once");
    // Every success is the one effect's receipt.
    let receipts = receipts.into_inner().unwrap();
    assert!(!receipts.is_empty());
    assert!(receipts.windows(2).all(|w| w[0] == w[1]));
}

// ---- provider faults are never success ----

struct Faulty {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl McpWire for Faulty {
    async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, Error> {
        Ok(tools())
    }
    async fn call_tool(&self, _: &str, _: &Value) -> Result<CallOutcome, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::Transport("connection reset".into()))
    }
}

#[tokio::test]
async fn transport_error_is_uncertain_never_success_and_never_retried() {
    let calls = Arc::new(AtomicUsize::new(0));
    let h = harness_with(
        Arc::new(Faulty {
            calls: calls.clone(),
        }),
        calls.clone(),
    )
    .await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("f"), 100);
    let first = h
        .lane
        .authorize_and_execute_approved(intent.clone(), scope("f"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(
        first,
        AuthorityOutcome::Execution(Error::ReconciliationRequired)
    );
    // The same effect again (same grant: replay path) and a fresh grant (same key): both are told
    // to reconcile, and the provider is not touched a second time.
    let again = h
        .lane
        .authorize_and_execute_approved(intent.clone(), scope("f"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(
        again,
        AuthorityOutcome::Execution(Error::ReconciliationRequired)
    );
    let g2 = h.desk.issue(&intent, scope("f"), 100);
    let fresh = h
        .lane
        .authorize_and_execute_approved(intent, scope("f"), &EffectClassAuthority, &g2, 1)
        .await
        .unwrap_err();
    assert_eq!(
        fresh,
        AuthorityOutcome::Execution(Error::ReconciliationRequired)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn provider_rejection_is_an_error_and_the_spent_grant_stays_spent() {
    let h = harness(CallOutcome::Rejected("no such directory".into())).await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("rj"), 100);
    let e = h
        .lane
        .authorize_and_execute_approved(intent.clone(), scope("rj"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(
        e,
        AuthorityOutcome::Execution(Error::Rejected("no such directory".into()))
    );
    // One approval bought one attempt. The same grant cannot buy a second.
    let e = h
        .lane
        .authorize_and_execute_approved(intent, scope("rj"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Consumed);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

// ---- the ledger binds the effect, not just the key ----

#[tokio::test]
async fn a_reused_idempotency_key_never_reports_success_for_a_different_effect() {
    let h = harness(CallOutcome::Finished(json!({"done": true}))).await;
    let a = EffectClassAuthority;
    // `e` is allowed without approval, so this is the plain `authorize_and_execute` path.
    let first = h
        .lane
        .authorize_and_execute(h.intent("e", json!({"n": 1})), scope("k"), &a)
        .await
        .unwrap();
    assert_eq!(first.tool_name, "e");
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    // A different request under the same key did not run, so it must not come back as a receipt.
    let other = h
        .lane
        .authorize_and_execute(h.intent("e", json!({"n": 2})), scope("k"), &a)
        .await;
    assert!(
        matches!(
            other,
            Err(AuthorityOutcome::Execution(Error::IdempotencyConflict))
        ),
        "{other:?}"
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 1, "second effect never ran");
    // The identical request is still an ordinary replay.
    let same = h
        .lane
        .authorize_and_execute(h.intent("e", json!({"n": 1})), scope("k"), &a)
        .await
        .unwrap();
    assert_eq!(same, first);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    let _ = &h.broker;
}

#[tokio::test]
async fn a_key_names_one_effect_in_one_world() {
    let h = harness(CallOutcome::Finished(json!({"done": true}))).await;
    let a = EffectClassAuthority;
    let intent = h.intent("e", json!({"n": 1}));
    h.lane
        .authorize_and_execute(intent.clone(), scope("w1"), &a)
        .await
        .unwrap();
    // Same effect and key, other world: not a replay, and no receipt that names world 1.
    let mut other = scope("w1");
    other.world_id = WorldId(2);
    let r = h.lane.authorize_and_execute(intent, other, &a).await;
    assert!(
        matches!(
            r,
            Err(AuthorityOutcome::Execution(Error::IdempotencyConflict))
        ),
        "{r:?}"
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_approved_effect_cannot_borrow_the_receipt_of_an_earlier_effect_on_its_key() {
    let h = harness(CallOutcome::Finished(json!({"done": true}))).await;
    let a = EffectClassAuthority;
    // Key `shared` is first used by an allowed effect.
    h.lane
        .authorize_and_execute(h.intent("e", json!({"n": 1})), scope("shared"), &a)
        .await
        .unwrap();
    // A different, approval-class effect is then approved under the same key.
    let write = h.intent("w", json!({"path": "x"}));
    let g = h.desk.issue(&write, scope("shared"), 100);
    let r = h
        .lane
        .authorize_and_execute_approved(write.clone(), scope("shared"), &a, &g, 1)
        .await;
    assert!(
        matches!(
            r,
            Err(AuthorityOutcome::Execution(Error::IdempotencyConflict))
        ),
        "{r:?}"
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 1, "the write never ran");
    // The grant was spent at mint and the conflict does not hand out a receipt on replay either.
    let again = h
        .lane
        .authorize_and_execute_approved(write, scope("shared"), &a, &g, 1)
        .await;
    assert!(
        matches!(
            again,
            Err(AuthorityOutcome::Execution(Error::IdempotencyConflict))
        ),
        "{again:?}"
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

// ---- cancellation: a withdrawn grant is dead ----

#[tokio::test]
async fn a_revoked_grant_cannot_be_spent_and_runs_nothing() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("rv"), 100);
    assert!(h.desk.revoke(&g));
    assert!(h.desk.revoke(&g), "revoking twice is still revoked");
    let e = h
        .lane
        .authorize_approved(intent.clone(), scope("rv"), &EffectClassAuthority, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Revoked);
    let e = h
        .lane
        .authorize_and_execute_approved(intent, scope("rv"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Revoked);
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_spent_or_foreign_grant_cannot_be_revoked() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let intent = h.intent("w", json!({"a": 1}));
    let g = h.desk.issue(&intent, scope("sp"), 100);
    let r = h
        .lane
        .authorize_and_execute_approved(intent.clone(), scope("sp"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap();
    // Too late: the effect ran. Revoke says so and the replay still returns the one receipt.
    assert!(!h.desk.revoke(&g));
    let again = h
        .lane
        .authorize_and_execute_approved(intent, scope("sp"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap();
    assert_eq!(r, again);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    // A grant from another desk is not this desk's to revoke.
    let other = harness(CallOutcome::Finished(json!({}))).await;
    let foreign = other
        .desk
        .issue(&other.intent("w", json!({})), scope("f"), 100);
    assert!(!h.desk.revoke(&foreign));
    assert!(other.desk.revoke(&foreign));
}

#[tokio::test]
async fn revoking_one_grant_leaves_another_for_the_same_effect_usable() {
    let h = harness(CallOutcome::Finished(json!({"ok": true}))).await;
    let intent = h.intent("w", json!({"a": 1}));
    let dead = h.desk.issue(&intent, scope("two"), 100);
    let live = h.desk.issue(&intent, scope("two"), 100);
    assert!(h.desk.revoke(&dead));
    let r = h
        .lane
        .authorize_and_execute_approved(intent, scope("two"), &EffectClassAuthority, &live, 1)
        .await;
    assert!(r.is_ok());
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}
