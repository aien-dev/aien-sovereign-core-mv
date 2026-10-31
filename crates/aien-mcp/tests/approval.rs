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

//! Single-use approval consumption, through the public API only.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use aien_mcp::memory::MemoryWire;
use aien_mcp::{
    ApprovalDesk, ApprovalError, AuthorityContext, AuthorityDecision, AuthorityOutcome,
    CallOutcome, EffectAuthority, EffectClassAuthority, EffectIntent, EffectLane, EffectScope,
    Exposure, McpBroker, SpeculativeLane, TrustLevel,
};
use serde_json::json;

struct Fixed(AuthorityDecision);
impl EffectAuthority for Fixed {
    fn authorize(&self, _: &EffectIntent, _: &AuthorityContext) -> AuthorityDecision {
        self.0.clone()
    }
}

struct H {
    broker: McpBroker,
    lane: EffectLane,
    desk: ApprovalDesk,
    calls: Arc<AtomicUsize>,
    intent: EffectIntent,
}

fn scope(label: &str) -> EffectScope {
    EffectScope {
        world_id: WorldId(1),
        winning_jnode: JNodeId(1),
        idempotency_key: EffectId::from_label(label),
    }
}

async fn harness() -> H {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let tool = ToolDescriptor::new("t", ToolEffects::WORLD_MUTATION, Digest32::of(b"t"));
    let wire = MemoryWire::new(vec![tool], move |_, _| {
        c.fetch_add(1, Ordering::SeqCst);
        CallOutcome::Finished(json!({"done": true}))
    });
    let broker = McpBroker::new();
    let provider = ProviderId::new("p");
    broker
        .admit(provider.clone(), Arc::new(wire))
        .await
        .unwrap();
    let live = SpeculativeLane::new(broker.clone())
        .discovery_snapshot(&provider)
        .await
        .unwrap()
        .catalog_digest;
    let intent = EffectIntent {
        provider,
        tool_name: "t".into(),
        arguments: json!({"a": 1}),
        capability_digest: live,
    };
    H {
        lane: EffectLane::new(broker.clone()).with_clock(std::sync::Arc::new(|| 1)),
        desk: ApprovalDesk::new(broker.clone()),
        broker,
        calls,
        intent,
    }
}

fn approval(e: AuthorityOutcome) -> ApprovalError {
    match e {
        AuthorityOutcome::Approval(a) => a,
        other => panic!("expected Approval, got {other:?}"),
    }
}

#[tokio::test]
async fn first_use_mints_and_runs_once() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("a"), 100);
    let r = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("a"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap();
    assert_eq!(r.output, json!({"done": true}));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn second_mint_with_the_same_grant_is_refused() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("b"), 100);
    let a = EffectClassAuthority;
    // Hold the minted effect: a dropped one would release the grant (see tests/two_phase.rs).
    let held = h
        .lane
        .authorize_approved(h.intent.clone(), scope("b"), &a, &g, 1)
        .unwrap();
    let err = h
        .lane
        .authorize_approved(h.intent.clone(), scope("b"), &a, &g, 1)
        .unwrap_err();
    assert_eq!(approval(err), ApprovalError::Reserved);
    // A cloned grant is the same grant.
    let err = h
        .lane
        .authorize_approved(h.intent.clone(), scope("b"), &a, &g.clone(), 1)
        .unwrap_err();
    assert_eq!(approval(err), ApprovalError::Reserved);
    // Once the held effect runs, the grant is spent for good.
    h.lane.execute_effect(held).await.unwrap();
    let err = h
        .lane
        .authorize_approved(h.intent.clone(), scope("b"), &a, &g, 1)
        .unwrap_err();
    assert_eq!(approval(err), ApprovalError::Consumed);
}

#[tokio::test]
async fn replay_after_completion_returns_the_existing_receipt() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("r"), 100);
    let a = EffectClassAuthority;
    let first = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("r"), &a, &g, 1)
        .await
        .unwrap();
    // Even after the grant has expired, the same logical request gets the same outcome.
    let again = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("r"), &a, &g, 10_000)
        .await
        .unwrap();
    assert_eq!(first, again);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn spent_grant_cannot_run_a_different_request_under_the_same_key() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("k"), 100);
    let a = EffectClassAuthority;
    h.lane
        .authorize_and_execute_approved(h.intent.clone(), scope("k"), &a, &g, 1)
        .await
        .unwrap();
    let mut other = h.intent.clone();
    other.arguments = json!({"a": 2});
    let err = h
        .lane
        .authorize_and_execute_approved(other, scope("k"), &a, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(approval(err), ApprovalError::Mismatch);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn grant_for_a_different_effect_is_refused_and_not_spent() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("x"), 100);
    let a = EffectClassAuthority;
    let mut other = h.intent.clone();
    other.arguments = json!({"a": 2});
    let e = h
        .lane
        .authorize_approved(other, scope("x"), &a, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Mismatch);
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), scope("y"), &a, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Mismatch);
    let mut w = scope("x");
    w.world_id = WorldId(2);
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), w, &a, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Mismatch);
    // The mismatches burned nothing: the right request still works.
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), scope("x"), &a, &g, 1)
        .is_ok());
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn expired_grant_is_refused_and_not_spent() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("e"), 50);
    let a = EffectClassAuthority;
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), scope("e"), &a, &g, 50)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Expired);
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), scope("e"), &a, &g, 49)
        .is_ok());
}

#[tokio::test]
async fn catalog_change_makes_the_grant_stale() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("c"), 100);
    let provider = h.intent.provider.clone();
    let changed = vec![
        ToolDescriptor::new("t", ToolEffects::WORLD_MUTATION, Digest32::of(b"t")),
        ToolDescriptor::new("u", ToolEffects::PURE, Digest32::of(b"u")),
    ];
    h.broker.replace_catalog(&provider, changed).await.unwrap();
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), scope("c"), &EffectClassAuthority, &g, 1)
        .unwrap_err();
    assert!(matches!(e, AuthorityOutcome::Denied(_)));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn deny_and_contain_stay_refused_and_keep_the_grant() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("d"), 100);
    let deny = Fixed(AuthorityDecision::Deny("no".into()));
    let e = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("d"), &deny, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(e, AuthorityOutcome::Denied("no".into()));
    let contain = Fixed(AuthorityDecision::Contain("box".into()));
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), scope("d"), &contain, &g, 1)
        .unwrap_err();
    assert_eq!(e, AuthorityOutcome::Contained("box".into()));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    // Grant untouched by the refusals.
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), scope("d"), &EffectClassAuthority, &g, 1)
        .is_ok());
}

#[tokio::test]
async fn allow_does_not_spend_the_grant() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("l"), 100);
    let allow = Fixed(AuthorityDecision::Allow);
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), scope("l"), &allow, &g, 1)
        .is_ok());
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), scope("l"), &EffectClassAuthority, &g, 1)
        .is_ok());
}

#[tokio::test]
async fn grant_from_another_broker_is_unknown() {
    let h = harness().await;
    let other = harness().await;
    let g = other.desk.issue(&other.intent, scope("z"), 100);
    let e = h
        .lane
        .authorize_approved(h.intent.clone(), scope("z"), &EffectClassAuthority, &g, 1)
        .unwrap_err();
    assert_eq!(approval(e), ApprovalError::Unknown);
}

#[tokio::test]
async fn unapproved_request_is_still_pending() {
    let h = harness().await;
    let e = h
        .lane
        .authorize(h.intent.clone(), scope("n"), &EffectClassAuthority)
        .unwrap_err();
    assert!(matches!(e, AuthorityOutcome::Pending { .. }));
}

#[tokio::test]
async fn approved_grant_authorizes_effect_pending_on_untrusted_exposure() {
    // E3: an effect held back by untrusted exposure is still released by an approval grant.
    let h = harness().await;
    let lane = h.lane.with_exposure(Exposure::new(
        vec!["web-1".into()],
        TrustLevel::ExternalUntrusted,
    ));
    let a = EffectClassAuthority;
    assert!(matches!(
        lane.authorize(h.intent.clone(), scope("ex"), &a),
        Err(AuthorityOutcome::Pending { .. })
    ));
    let g = h.desk.issue(&h.intent, scope("ex"), 100);
    let r = lane
        .authorize_and_execute_approved(h.intent.clone(), scope("ex"), &a, &g, 1)
        .await
        .unwrap();
    assert_eq!(r.output, json!({"done": true}));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}
