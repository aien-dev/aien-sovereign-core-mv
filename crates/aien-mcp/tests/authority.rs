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

//! Negative controls for the host-owned authority seam. This file is outside the crate, so it
//! can touch only the public API: no `AuthorizedEffect` can be built here except through
//! `EffectLane::authorize`.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use aien_mcp::memory::MemoryWire;
use aien_mcp::{
    AuthorityContext, AuthorityDecision, AuthorityOutcome, CallOutcome, EffectAuthority,
    EffectClassAuthority, EffectIntent, EffectLane, EffectScope, Exposure, McpBroker, TrustLevel,
};
use serde_json::json;

struct Fixed(AuthorityDecision);
impl EffectAuthority for Fixed {
    fn authorize(&self, _: &EffectIntent, _: &AuthorityContext) -> AuthorityDecision {
        self.0.clone()
    }
}

struct H {
    lane: EffectLane,
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

async fn harness(effects: ToolEffects) -> H {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let tool = ToolDescriptor::new("t", effects, Digest32::of(b"t"));
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
    // An intent is plain data (public fields). Possessing one grants nothing.
    let live = aien_mcp::SpeculativeLane::new(broker.clone())
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
        lane: EffectLane::new(broker),
        calls,
        intent,
    }
}

#[tokio::test]
async fn deny_mints_nothing_and_never_executes() {
    let h = harness(ToolEffects::PURE).await;
    let a = Fixed(AuthorityDecision::Deny("no".into()));
    assert!(h.lane.authorize(h.intent.clone(), scope("d"), &a).is_err());
    let r = h.lane.authorize_and_execute(h.intent, scope("d"), &a).await;
    assert_eq!(r.unwrap_err(), AuthorityOutcome::Denied("no".into()));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn contain_mints_nothing_and_never_executes() {
    let h = harness(ToolEffects::PURE).await;
    let a = Fixed(AuthorityDecision::Contain("box".into()));
    let r = h.lane.authorize_and_execute(h.intent, scope("c"), &a).await;
    assert_eq!(r.unwrap_err(), AuthorityOutcome::Contained("box".into()));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn require_approval_is_pending_and_never_executes() {
    let h = harness(ToolEffects::WORLD_MUTATION).await;
    let want = aien_mcp::intent_digest(&h.intent);
    let r = h
        .lane
        .authorize_and_execute(h.intent, scope("p"), &EffectClassAuthority)
        .await;
    match r.unwrap_err() {
        AuthorityOutcome::Pending { intent_digest, .. } => assert_eq!(intent_digest, want),
        other => panic!("expected Pending, got {other:?}"),
    }
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn allow_executes_and_returns_receipt() {
    let h = harness(ToolEffects::LOCAL_EPHEMERAL).await;
    let r = h
        .lane
        .authorize_and_execute(h.intent, scope("a"), &EffectClassAuthority)
        .await
        .unwrap();
    assert_eq!(r.output, json!({"done": true}));
    assert_eq!(r.tool_name, "t");
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn allow_restricted_with_restrictions_is_denied_and_nothing_runs() {
    // The lane cannot enforce a restriction list, so it must not mint as if it were `Allow`.
    let h = harness(ToolEffects::PURE).await;
    let a = Fixed(AuthorityDecision::AllowRestricted(vec!["x".into()]));
    let r = h.lane.authorize_and_execute(h.intent, scope("r"), &a).await;
    assert!(matches!(r.unwrap_err(), AuthorityOutcome::Denied(_)));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn allow_restricted_without_restrictions_mints() {
    let h = harness(ToolEffects::PURE).await;
    let a = Fixed(AuthorityDecision::AllowRestricted(Vec::new()));
    assert!(h
        .lane
        .authorize_and_execute(h.intent, scope("r"), &a)
        .await
        .is_ok());
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unknown_descriptor_is_denied() {
    let h = harness(ToolEffects::PURE).await;
    let mut intent = h.intent.clone();
    intent.tool_name = "ghost".into();
    let r = h
        .lane
        .authorize_and_execute(intent, scope("u"), &EffectClassAuthority)
        .await;
    assert!(matches!(r.unwrap_err(), AuthorityOutcome::Denied(_)));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn stale_intent_is_denied_before_the_authority_is_asked() {
    let h = harness(ToolEffects::PURE).await;
    let mut intent = h.intent.clone();
    intent.capability_digest = Digest32::of(b"old");
    let a = Fixed(AuthorityDecision::Allow);
    let r = h.lane.authorize_and_execute(intent, scope("s"), &a).await;
    assert!(matches!(r.unwrap_err(), AuthorityOutcome::Denied(_)));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn effect_class_table() {
    use AuthorityDecision as D;
    let cases = [
        (ToolEffects::PURE, "allow"),
        (ToolEffects::READ_FILESYSTEM, "allow"),
        (ToolEffects::READ_NETWORK, "allow"),
        (ToolEffects::LOCAL_EPHEMERAL, "allow"),
        (ToolEffects::EXTERNAL_WRITE, "approval"),
        (ToolEffects::WORLD_MUTATION, "approval"),
        (ToolEffects::SECRET_BEARING, "approval"),
        (ToolEffects::SPAWN_PROCESS, "approval"),
        (ToolEffects::EXTERNAL_IRREVERSIBLE, "deny"),
        (
            ToolEffects::EXTERNAL_IRREVERSIBLE | ToolEffects::READ_FILESYSTEM,
            "deny",
        ),
    ];
    for (effects, want) in cases {
        let h = harness(effects).await;
        let r = h
            .lane
            .authorize(h.intent.clone(), scope("t"), &EffectClassAuthority);
        let got = match &r {
            Ok(_) => "allow",
            Err(AuthorityOutcome::Pending { .. }) => "approval",
            Err(AuthorityOutcome::Denied(_)) => "deny",
            Err(_) => "other",
        };
        assert_eq!(got, want, "{effects:?}");
    }
    let _ = D::Allow;
}

// ---- E3: exposure input (INTERPLANE 0.3 cut E3) ----

fn exposure(floor: TrustLevel) -> Exposure {
    Exposure::new(vec!["in-1".into()], floor)
}

const EFFECT_BITS: [ToolEffects; 4] = [
    ToolEffects::EXTERNAL_WRITE,
    ToolEffects::WORLD_MUTATION,
    ToolEffects::SECRET_BEARING,
    ToolEffects::SPAWN_PROCESS,
];

#[test]
fn trust_order_and_unknown_wire_value() {
    assert!(TrustLevel::TrustedRuntime > TrustLevel::UserSupplied);
    assert!(TrustLevel::UserSupplied > TrustLevel::WorkspaceUntrusted);
    assert!(TrustLevel::WorkspaceUntrusted > TrustLevel::ExternalUntrusted);
    assert_eq!(
        TrustLevel::from_wire("user_supplied"),
        TrustLevel::UserSupplied
    );
    assert_eq!(
        TrustLevel::from_wire("unknown"),
        TrustLevel::ExternalUntrusted
    );
    assert_eq!(
        TrustLevel::from_wire("garbage"),
        TrustLevel::ExternalUntrusted
    );
}

#[tokio::test]
async fn untrusted_exposure_makes_effects_pending_never_allow() {
    for floor in [
        TrustLevel::WorkspaceUntrusted,
        TrustLevel::ExternalUntrusted,
    ] {
        for fx in EFFECT_BITS {
            let h = harness(fx).await;
            let r = h
                .lane
                .with_exposure(exposure(floor))
                .authorize_and_execute(h.intent, scope("x"), &EffectClassAuthority)
                .await;
            match r {
                Err(AuthorityOutcome::Pending { reason, .. }) => {
                    assert!(reason.contains("exposure"), "{reason}")
                }
                other => panic!("expected Pending, got {other:?}"),
            }
            assert_eq!(h.calls.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn read_class_is_unaffected_by_exposure() {
    for fx in [
        ToolEffects::PURE,
        ToolEffects::READ_FILESYSTEM,
        ToolEffects::READ_NETWORK,
    ] {
        let h = harness(fx).await;
        let r = h
            .lane
            .with_exposure(exposure(TrustLevel::ExternalUntrusted))
            .authorize_and_execute(h.intent, scope("r"), &EffectClassAuthority)
            .await;
        assert!(r.is_ok());
        assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn absent_exposure_fails_closed_for_effects_only() {
    let h = harness(ToolEffects::EXTERNAL_WRITE).await;
    let r = h
        .lane
        .authorize(h.intent.clone(), scope("n"), &EffectClassAuthority);
    match r {
        Err(AuthorityOutcome::Pending { reason, .. }) => assert!(reason.contains("exposure")),
        other => panic!("expected Pending, got {other:?}"),
    }
    let h = harness(ToolEffects::PURE).await;
    assert!(h
        .lane
        .authorize(h.intent.clone(), scope("n2"), &EffectClassAuthority)
        .is_ok());
}

#[tokio::test]
async fn context_exposes_only_what_the_host_set() {
    struct Probe(std::sync::Mutex<Option<Option<Exposure>>>);
    impl EffectAuthority for Probe {
        fn authorize(&self, _: &EffectIntent, c: &AuthorityContext) -> AuthorityDecision {
            *self.0.lock().unwrap() = Some(c.exposure().cloned());
            AuthorityDecision::Allow
        }
    }
    let h = harness(ToolEffects::PURE).await;
    let p = Probe(Default::default());
    h.lane.authorize(h.intent.clone(), scope("p1"), &p).unwrap();
    assert_eq!(p.0.lock().unwrap().take(), Some(None));
    let e = exposure(TrustLevel::UserSupplied);
    h.lane
        .with_exposure(e.clone())
        .authorize(h.intent.clone(), scope("p2"), &p)
        .unwrap();
    assert_eq!(p.0.lock().unwrap().take(), Some(Some(e)));
}

#[tokio::test]
async fn user_supplied_or_better_keeps_the_existing_effect_table() {
    for floor in [TrustLevel::UserSupplied, TrustLevel::TrustedRuntime] {
        let h = harness(ToolEffects::EXTERNAL_WRITE).await;
        let r = h.lane.with_exposure(exposure(floor)).authorize(
            h.intent.clone(),
            scope("t"),
            &EffectClassAuthority,
        );
        match r {
            Err(AuthorityOutcome::Pending { reason, .. }) => {
                assert!(!reason.contains("exposure"), "{reason}")
            }
            other => panic!("expected Pending, got {other:?}"),
        }
    }
}
