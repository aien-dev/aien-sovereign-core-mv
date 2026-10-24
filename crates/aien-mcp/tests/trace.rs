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

//! Correlation trace events from the effect lane (#398), through the public API.
//! Events are observational: these tests also check that nothing sensitive is in them.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use aien_mcp::memory::MemoryWire;
use aien_mcp::{
    receipt_digest, ApprovalDesk, AuthorityContext, AuthorityDecision, CallOutcome,
    EffectAuthority, EffectIntent, EffectLane, EffectReceipt, EffectScope, McpBroker,
    SpeculativeLane,
};
use aien_trace::{
    EffectCertainty, EventKind, EventStatus, MemorySink, TraceEvent, TraceId, TraceSink,
};
use serde_json::{json, Value};

const SECRET_ARG: &str = "SECRET-ARGUMENT-VALUE-1";
const SECRET_OUT: &str = "SECRET-OUTPUT-VALUE-2";

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
    sink: Arc<MemorySink>,
    calls: Arc<AtomicUsize>,
    intent: EffectIntent,
}

fn scope(label: &str) -> EffectScope {
    EffectScope {
        world_id: WorldId(7),
        winning_jnode: JNodeId(1),
        idempotency_key: EffectId::from_label(label),
    }
}

async fn harness(outcome: CallOutcome) -> H {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let tool = ToolDescriptor::new("send", ToolEffects::WORLD_MUTATION, Digest32::of(b"send"));
    let wire = MemoryWire::new(vec![tool], move |_, _| {
        c.fetch_add(1, Ordering::SeqCst);
        outcome.clone()
    });
    let broker = McpBroker::new();
    let provider = ProviderId::new("mail");
    broker
        .admit(provider.clone(), Arc::new(wire))
        .await
        .unwrap();
    let live = SpeculativeLane::new(broker.clone())
        .discovery_snapshot(&provider)
        .await
        .unwrap()
        .catalog_digest;
    let sink = Arc::new(MemorySink::new());
    let dynsink: Arc<dyn TraceSink> = sink.clone();
    let lane = EffectLane::new(broker.clone())
        .with_clock(Arc::new(|| 1))
        .with_trace(dynsink, TraceId::derive(1, 2, 3), Some(41));
    H {
        lane,
        desk: ApprovalDesk::new(broker.clone()),
        broker,
        sink,
        calls,
        intent: EffectIntent {
            provider,
            tool_name: "send".into(),
            arguments: json!({"body": SECRET_ARG}),
            capability_digest: live,
        },
    }
}

fn finished() -> CallOutcome {
    CallOutcome::Finished(json!({"result": SECRET_OUT}))
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[tokio::test]
async fn allow_then_finished_gives_ok_events_with_receipt_digest() {
    let h = harness(finished()).await;
    let r = h
        .lane
        .authorize_and_execute(
            h.intent.clone(),
            scope("a"),
            &Fixed(AuthorityDecision::Allow),
        )
        .await
        .unwrap();
    let ev = h.sink.drain();
    assert_eq!(ev.len(), 2);
    assert_eq!(ev[0].kind, EventKind::AuthorityDecided);
    assert_eq!(ev[0].status, EventStatus::Ok);
    assert_eq!(ev[0].parent_event_id, Some(41));
    assert_eq!(ev[0].ids.world_id, Some(7));
    assert_eq!(
        ev[0].ids.tool_request_id.as_ref().unwrap().as_str(),
        "mail:send"
    );
    assert!(is_hex64(
        ev[0].refs.intent_digest.as_ref().unwrap().as_str()
    ));
    assert_eq!(ev[1].kind, EventKind::EffectExecuted);
    assert_eq!(ev[1].status, EventStatus::Ok);
    assert_eq!(
        ev[1].effect_certainty,
        Some(EffectCertainty::EffectOccurred)
    );
    let d = ev[1].refs.effect_receipt_digest.as_ref().unwrap().as_str();
    assert!(is_hex64(d));
    let want: String = receipt_digest(&r)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(d, want);
}

#[tokio::test]
async fn deny_contain_and_stale_are_rejected_and_nothing_runs() {
    let h = harness(finished()).await;
    let _ = h.lane.authorize(
        h.intent.clone(),
        scope("d"),
        &Fixed(AuthorityDecision::Deny("no".into())),
    );
    let _ = h.lane.authorize(
        h.intent.clone(),
        scope("c"),
        &Fixed(AuthorityDecision::Contain("x".into())),
    );
    let mut stale = h.intent.clone();
    stale.capability_digest = Digest32::of(b"old");
    let _ = h
        .lane
        .authorize(stale, scope("s"), &Fixed(AuthorityDecision::Allow));
    let ev = h.sink.drain();
    assert_eq!(ev.len(), 3);
    assert!(ev.iter().all(|e| e.status == EventStatus::Rejected));
    assert_eq!(ev[0].note, None);
    assert_eq!(ev[1].note.as_ref().unwrap().as_str(), "contained");
    assert_eq!(ev[2].note.as_ref().unwrap().as_str(), "stale_intent");
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn require_approval_is_approval_pending_then_grant_id_appears() {
    let h = harness(finished()).await;
    let auth = Fixed(AuthorityDecision::RequireApproval("human".into()));
    assert!(h
        .lane
        .authorize(h.intent.clone(), scope("p"), &auth)
        .is_err());
    let ev = h.sink.drain();
    assert_eq!(ev[0].status, EventStatus::ApprovalPending);
    assert_eq!(ev[0].ids.grant_id, None);

    let grant = h.desk.issue(&h.intent, scope("p"), 100);
    h.lane
        .authorize_and_execute_approved(h.intent.clone(), scope("p"), &auth, &grant, 1)
        .await
        .unwrap();
    let ev = h.sink.drain();
    assert_eq!(ev[0].status, EventStatus::Ok);
    assert!(is_hex64(ev[0].ids.grant_id.as_ref().unwrap().as_str()));
    assert_eq!(ev[1].ids.grant_id, ev[0].ids.grant_id);

    // Spending the grant for another effect is refused and named.
    let other = scope("other");
    assert!(h
        .lane
        .authorize_approved(h.intent.clone(), other, &auth, &grant, 1)
        .is_err());
    let ev = h.sink.drain();
    assert_eq!(ev[0].status, EventStatus::Rejected);
    assert!(ev[0].note.is_some());
}

#[tokio::test]
async fn uncertain_provider_gives_uncertain_status_and_certainty() {
    let h = harness(CallOutcome::Uncertain).await;
    let r = h
        .lane
        .authorize_and_execute(
            h.intent.clone(),
            scope("u"),
            &Fixed(AuthorityDecision::Allow),
        )
        .await;
    assert!(r.is_err());
    let ev = h.sink.drain();
    assert_eq!(ev[1].kind, EventKind::EffectExecuted);
    assert_eq!(ev[1].status, EventStatus::Uncertain);
    assert_eq!(ev[1].effect_certainty, Some(EffectCertainty::Uncertain));
    assert_eq!(ev[1].refs.effect_receipt_digest, None);
}

#[tokio::test]
async fn rejected_provider_call_is_no_effect() {
    let h = harness(CallOutcome::Rejected("bad".into())).await;
    let _ = h
        .lane
        .authorize_and_execute(
            h.intent.clone(),
            scope("r"),
            &Fixed(AuthorityDecision::Allow),
        )
        .await;
    let ev = h.sink.drain();
    assert_eq!(ev[1].status, EventStatus::Rejected);
    assert_eq!(ev[1].effect_certainty, Some(EffectCertainty::NoEffect));
}

#[tokio::test]
async fn second_execute_is_a_ledger_replay_without_a_second_call() {
    let h = harness(finished()).await;
    let allow = Fixed(AuthorityDecision::Allow);
    let first = h
        .lane
        .authorize_and_execute(h.intent.clone(), scope("k"), &allow)
        .await
        .unwrap();
    let second = h
        .lane
        .authorize_and_execute(h.intent.clone(), scope("k"), &allow)
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    let ev = h.sink.drain();
    let execs: Vec<&TraceEvent> = ev
        .iter()
        .filter(|e| e.kind == EventKind::EffectExecuted)
        .collect();
    assert_eq!(execs.len(), 2);
    assert_eq!(execs[0].note, None);
    assert_eq!(execs[1].note.as_ref().unwrap().as_str(), "ledger_replay");
    assert_eq!(execs[1].status, EventStatus::Ok);
    assert_eq!(
        execs[0].refs.effect_receipt_digest,
        execs[1].refs.effect_receipt_digest
    );
}

#[tokio::test]
async fn no_event_carries_arguments_outputs_or_forbidden_keys() {
    let h = harness(finished()).await;
    let allow = Fixed(AuthorityDecision::Allow);
    for (i, a) in [
        AuthorityDecision::Allow,
        AuthorityDecision::Deny("no".into()),
        AuthorityDecision::RequireApproval("h".into()),
    ]
    .into_iter()
    .enumerate()
    {
        let _ = h
            .lane
            .authorize_and_execute(h.intent.clone(), scope(&format!("j{i}")), &Fixed(a))
            .await;
    }
    let _ = allow;
    let ev = h.sink.drain();
    assert!(ev.len() >= 4);
    let text = serde_json::to_string(&ev).unwrap();
    assert!(!text.contains(SECRET_ARG));
    assert!(!text.contains(SECRET_OUT));
    fn keys(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    out.push(k.clone());
                    keys(v, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }
    let mut ks = Vec::new();
    keys(&serde_json::to_value(&ev).unwrap(), &mut ks);
    for k in ks {
        for bad in ["arguments", "output", "prompt", "content"] {
            assert!(!k.contains(bad), "key {k}");
        }
    }
}

#[tokio::test]
async fn default_lane_emits_nothing_and_still_works() {
    let h = harness(finished()).await;
    let plain = EffectLane::new(h.broker.clone());
    let r = plain
        .authorize_and_execute(
            h.intent.clone(),
            scope("n"),
            &Fixed(AuthorityDecision::Allow),
        )
        .await;
    assert!(r.is_ok());
    assert!(h.sink.drain().is_empty());
}

fn receipt() -> EffectReceipt {
    EffectReceipt {
        effect_id: EffectId::from_label("e"),
        world_id: WorldId(1),
        winning_jnode: JNodeId(2),
        policy_digest: Digest32::of(b"p"),
        provider: ProviderId::new("mail"),
        tool_name: "send".into(),
        capability_digest: Digest32::of(b"c"),
        output: json!({"b": 1, "a": [1, 2]}),
    }
}

#[test]
fn receipt_digest_depends_on_output_and_tool_name_but_not_key_order() {
    let base = receipt();
    let d = receipt_digest(&base);
    assert_eq!(d, receipt_digest(&base.clone()));
    let mut o = base.clone();
    o.output = json!({"b": 2, "a": [1, 2]});
    assert_ne!(d, receipt_digest(&o));
    let mut t = base.clone();
    t.tool_name = "send2".into();
    assert_ne!(d, receipt_digest(&t));
    let mut w = base.clone();
    w.world_id = WorldId(9);
    assert_ne!(d, receipt_digest(&w));
    let mut k = base.clone();
    k.output = serde_json::from_str(r#"{"a":[1,2],"b":1}"#).unwrap();
    assert_eq!(d, receipt_digest(&k));
}
