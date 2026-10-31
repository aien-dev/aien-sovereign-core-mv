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

//! Two-phase approval spend (issue #206), through the public API only: reserve at mint, commit in
//! `execute_effect`, release when the minted effect is dropped or cancelled un-executed.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use aien_mcp::memory::MemoryWire;
use aien_mcp::{
    ApprovalDesk, ApprovalError, ApprovalGrant, AuthorityOutcome, CallOutcome,
    EffectClassAuthority, EffectIntent, EffectLane, EffectScope, Error, GrantStatus, McpBroker,
    SpeculativeLane,
};
use serde_json::json;

struct H {
    lane: EffectLane,
    desk: ApprovalDesk,
    calls: Arc<AtomicUsize>,
    intent: EffectIntent,
    broker: McpBroker,
    provider: ProviderId,
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
        let n = c.fetch_add(1, Ordering::SeqCst);
        CallOutcome::Finished(json!({"done": true, "n": n}))
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
        provider: provider.clone(),
        tool_name: "t".into(),
        arguments: json!({"a": 1}),
        capability_digest: live,
    };
    H {
        lane: EffectLane::new(broker.clone()).with_clock(Arc::new(|| 1)),
        desk: ApprovalDesk::new(broker.clone()),
        broker,
        calls,
        intent,
        provider,
    }
}

fn mint(
    h: &H,
    label: &str,
    g: &ApprovalGrant,
    now: u64,
) -> Result<aien_mcp::AuthorizedEffect<EffectIntent>, AuthorityOutcome> {
    h.lane.authorize_approved(
        h.intent.clone(),
        scope(label),
        &EffectClassAuthority,
        g,
        now,
    )
}

fn approval(e: AuthorityOutcome) -> ApprovalError {
    match e {
        AuthorityOutcome::Approval(a) => a,
        other => panic!("expected Approval, got {other:?}"),
    }
}

#[tokio::test]
async fn refused_after_mint_keeps_grant_usable() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("a"), 100);
    // The host mints, then its own checks refuse and it drops the effect.
    let effect = mint(&h, "a", &g, 1).unwrap();
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Reserved));
    drop(effect);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    // The same grant still buys exactly one run.
    let r = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("a"), &EffectClassAuthority, &g, 2)
        .await
        .unwrap();
    assert_eq!(r.output["done"], json!(true));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropped_effect_releases_and_records_why() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("d"), 100);
    assert_eq!(h.desk.release_reason(&g), None);
    drop(mint(&h, "d", &g, 1).unwrap());
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    assert_eq!(
        h.desk.release_reason(&g).as_deref(),
        Some("dropped before execute_effect")
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    assert!(mint(&h, "d", &g, 1).is_ok());
}

#[tokio::test]
async fn explicit_cancel_releases_with_a_reason_and_kills_clones() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("c"), 100);
    let effect = mint(&h, "c", &g, 1).unwrap();
    assert_eq!(effect.approval_id(), Some(g.approval_id()));
    let clone = effect.clone();
    effect.cancel("digest check refused");
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    assert_eq!(
        h.desk.release_reason(&g).as_deref(),
        Some("digest check refused")
    );
    // A surviving clone must not run on a grant that was given back.
    assert_eq!(
        h.lane.execute_effect(clone).await.unwrap_err(),
        Error::ApprovalNotReserved
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    // Dropping the dead clone did not disturb a later reservation.
    let again = mint(&h, "c", &g, 1).unwrap();
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Reserved));
    drop(again);
}

#[tokio::test]
async fn stale_cancel_cannot_release_a_later_reservation() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("s"), 100);
    let first = mint(&h, "s", &g, 1).unwrap();
    let stale = first.clone();
    first.cancel("first");
    let second = mint(&h, "s", &g, 1).unwrap();
    stale.cancel("late");
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Reserved));
    h.lane.execute_effect(second).await.unwrap();
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn double_mint_while_reserved_is_refused() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("m"), 100);
    let held = mint(&h, "m", &g, 1).unwrap();
    assert_eq!(
        approval(mint(&h, "m", &g, 1).unwrap_err()),
        ApprovalError::Reserved
    );
    // The desk cannot withdraw a grant an effect is holding.
    assert!(!h.desk.revoke(&g));
    drop(held);
    assert!(h.desk.revoke(&g));
    assert_eq!(
        approval(mint(&h, "m", &g, 1).unwrap_err()),
        ApprovalError::Revoked
    );
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Revoked));
}

#[tokio::test]
async fn execute_commits_once() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("e"), 100);
    let effect = mint(&h, "e", &g, 1).unwrap();
    h.lane.execute_effect(effect).await.unwrap();
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Spent));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    // Spent is final: no release by drop, and no second mint.
    assert_eq!(
        approval(mint(&h, "e", &g, 1).unwrap_err()),
        ApprovalError::Consumed
    );
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Spent));
    assert_eq!(h.desk.release_reason(&g), None);
}

#[tokio::test]
async fn replay_after_execute_returns_the_original_receipt() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("r"), 100);
    let first = h
        .lane
        .authorize_and_execute_approved(h.intent.clone(), scope("r"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap();
    let again = h
        .lane
        .authorize_and_execute_approved(
            h.intent.clone(),
            scope("r"),
            &EffectClassAuthority,
            &g,
            10_000,
        )
        .await
        .unwrap();
    assert_eq!(first, again);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Spent));
}

#[tokio::test]
async fn expired_grant_released_then_remint_is_refused() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("x"), 10);
    drop(mint(&h, "x", &g, 1).unwrap());
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    assert_eq!(
        approval(mint(&h, "x", &g, 10).unwrap_err()),
        ApprovalError::Expired
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn refusal_inside_execute_effect_before_dispatch_releases() {
    let h = harness().await;
    let g = h.desk.issue(&h.intent, scope("z"), 100);
    let effect = mint(&h, "z", &g, 1).unwrap();
    // The catalog moves between mint and execute: execute_effect refuses as stale.
    let tool = ToolDescriptor::new("t", ToolEffects::WORLD_MUTATION, Digest32::of(b"t2"));
    h.broker
        .replace_catalog(&h.provider, vec![tool])
        .await
        .unwrap();
    assert_eq!(
        h.lane.execute_effect(effect).await.unwrap_err(),
        Error::StaleCapability
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
}

#[test]
fn concurrent_mint_race_has_one_winner() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let h = rt.block_on(harness());
    let g = h.desk.issue(&h.intent, scope("race"), 100);
    let barrier = Barrier::new(2);
    let wins = AtomicUsize::new(0);
    let reserved = AtomicUsize::new(0);
    let held = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                barrier.wait();
                match mint(&h, "race", &g, 1) {
                    Ok(e) => {
                        held.lock().unwrap().push(e);
                        wins.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(AuthorityOutcome::Approval(ApprovalError::Reserved)) => {
                        reserved.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(other) => panic!("unexpected {other:?}"),
                }
            });
        }
    });
    assert_eq!(wins.load(Ordering::SeqCst), 1);
    assert_eq!(reserved.load(Ordering::SeqCst), 1);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Reserved));
}

#[tokio::test]
async fn grant_from_another_broker_has_no_status() {
    let h = harness().await;
    let other = harness().await;
    let g = other.desk.issue(&other.intent, scope("o"), 100);
    assert_eq!(h.desk.status(&g), None);
}

#[tokio::test]
async fn held_effect_past_expiry_is_refused_at_commit_and_released() {
    let h = harness().await;
    let clock = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let c = clock.clone();
    let lane = h
        .lane
        .with_clock(Arc::new(move || c.load(Ordering::SeqCst)));
    let g = h.desk.issue(&h.intent, scope("exp"), 10);
    let effect = lane
        .authorize_approved(h.intent.clone(), scope("exp"), &EffectClassAuthority, &g, 1)
        .unwrap();
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Reserved));
    clock.store(10, Ordering::SeqCst);
    assert_eq!(
        lane.execute_effect(effect).await.unwrap_err(),
        Error::ApprovalExpired
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    assert_eq!(
        h.desk.release_reason(&g).as_deref(),
        Some("expired before execute")
    );
    assert_eq!(
        approval(mint(&h, "exp", &g, 10).unwrap_err()),
        ApprovalError::Expired
    );
}

#[tokio::test]
async fn clock_before_expiry_still_commits() {
    let h = harness().await;
    let lane = h.lane.with_clock(Arc::new(|| 5));
    let g = h.desk.issue(&h.intent, scope("ok"), 10);
    let effect = lane
        .authorize_approved(h.intent.clone(), scope("ok"), &EffectClassAuthority, &g, 1)
        .unwrap();
    lane.execute_effect(effect).await.unwrap();
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Spent));
}

#[tokio::test]
async fn provider_panic_after_commit_leaves_the_grant_spent_and_never_reruns() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let tool = ToolDescriptor::new("t", ToolEffects::WORLD_MUTATION, Digest32::of(b"t"));
    let wire = MemoryWire::new(vec![tool], move |_, _| {
        c.fetch_add(1, Ordering::SeqCst);
        panic!("provider blew up");
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
    let lane = EffectLane::new(broker.clone()).with_clock(Arc::new(|| 1));
    let desk = ApprovalDesk::new(broker);
    let g = desk.issue(&intent, scope("pan"), 100);
    let (l2, i2, g2) = (lane.clone(), intent.clone(), g.clone());
    let joined = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let _ = rt.block_on(l2.authorize_and_execute_approved(
            i2,
            scope("pan"),
            &EffectClassAuthority,
            &g2,
            1,
        ));
    })
    .join();
    assert!(joined.is_err(), "provider panic must propagate");
    assert_eq!(desk.status(&g), Some(GrantStatus::Spent));
    let again = lane
        .authorize_and_execute_approved(intent, scope("pan"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap_err();
    assert_eq!(again, AuthorityOutcome::Execution(Error::EffectInFlight));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn reserved_effect_on_a_lane_without_a_clock_fails_closed_and_releases() {
    let h = harness().await;
    let no_clock = EffectLane::new(h.broker.clone());
    let g = h.desk.issue(&h.intent, scope("nc"), 100);
    let effect = no_clock
        .authorize_approved(h.intent.clone(), scope("nc"), &EffectClassAuthority, &g, 1)
        .unwrap();
    assert_eq!(
        no_clock.execute_effect(effect).await.unwrap_err(),
        Error::ApprovalClockMissing
    );
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
    assert_eq!(h.desk.status(&g), Some(GrantStatus::Available));
    assert_eq!(
        h.desk.release_reason(&g).as_deref(),
        Some("no clock: expiry cannot be checked")
    );
    // The same grant runs once on a lane that has a clock.
    h.lane
        .authorize_and_execute_approved(h.intent.clone(), scope("nc"), &EffectClassAuthority, &g, 1)
        .await
        .unwrap();
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn effect_without_a_reservation_runs_on_a_lane_without_a_clock() {
    let h = harness().await;
    let no_clock = EffectLane::new(h.broker.clone());
    // Allow verdict: no grant involved, so no clock is needed.
    struct AllowAll;
    impl aien_mcp::EffectAuthority for AllowAll {
        fn authorize(
            &self,
            _: &EffectIntent,
            _: &aien_mcp::AuthorityContext,
        ) -> aien_mcp::AuthorityDecision {
            aien_mcp::AuthorityDecision::Allow
        }
    }
    let r = no_clock
        .authorize_and_execute(h.intent.clone(), scope("na"), &AllowAll)
        .await
        .unwrap();
    assert_eq!(r.output["done"], json!(true));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}
