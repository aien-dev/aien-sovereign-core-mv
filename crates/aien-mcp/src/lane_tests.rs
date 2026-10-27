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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{
    Digest32, EffectId, JNodeId, ProviderId, ToolDescriptor, ToolEffects, WorldId,
};
use serde_json::json;

use crate::memory::MemoryWire;
use crate::{
    authorize_for_test, CallOutcome, EffectLane, Error, McpBroker, SpeculativeLane,
    SpeculativeToolCall,
};

fn tool(name: &str, effects: ToolEffects) -> ToolDescriptor {
    ToolDescriptor::new(name, effects, Digest32::of(name.as_bytes()))
}

struct Harness {
    broker: McpBroker,
    calls: Arc<AtomicUsize>,
    provider: ProviderId,
}

async fn harness(tools: Vec<ToolDescriptor>, outcome: CallOutcome) -> Harness {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let wire = MemoryWire::new(tools, move |_name, _args| {
        counted.fetch_add(1, Ordering::SeqCst);
        outcome.clone()
    });
    let broker = McpBroker::new();
    let provider = ProviderId::new("mail");
    broker
        .admit(provider.clone(), Arc::new(wire))
        .await
        .unwrap();
    Harness {
        broker,
        calls,
        provider,
    }
}

#[tokio::test]
async fn unadmitted_provider_cannot_be_discovered() {
    let broker = McpBroker::new();
    let lane = SpeculativeLane::new(broker);
    let err = lane
        .discovery_snapshot(&ProviderId::new("unknown"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotAdmitted(_)));
}

#[tokio::test]
async fn discovery_returns_a_snapshot_and_does_not_call_a_tool() {
    let h = harness(
        vec![tool("mail.search", ToolEffects::READ_NETWORK)],
        CallOutcome::Finished(json!({})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker.clone());
    let snapshot = lane.discovery_snapshot(&h.provider).await.unwrap();
    assert_eq!(snapshot.provider, h.provider);
    assert_eq!(snapshot.tools.len(), 1);
    assert_eq!(snapshot.tools[0].name(), "mail.search");
    assert_ne!(snapshot.catalog_digest, Digest32::of(&[]));
    assert!(snapshot.expires_at > snapshot.generated_at);
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn read_only_tool_call_runs_on_the_speculative_lane() {
    let h = harness(
        vec![tool(
            "mail.search",
            ToolEffects::READ_NETWORK | ToolEffects::SECRET_BEARING,
        )],
        CallOutcome::Finished(json!({"hits": 1})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker);
    let result = lane
        .invoke_speculative(SpeculativeToolCall {
            provider: h.provider,
            tool_name: "mail.search".into(),
            arguments: json!({}),
            sandboxed: false,
        })
        .await
        .unwrap();
    assert_eq!(result.output, json!({"hits": 1}));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn irreversible_tool_call_is_rejected_before_the_session_is_used() {
    let h = harness(
        vec![tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)],
        CallOutcome::Finished(json!({"sent": true})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker);
    let err = lane
        .invoke_speculative(SpeculativeToolCall {
            provider: h.provider,
            tool_name: "mail.send".into(),
            arguments: json!({"to": "a@b.c"}),
            sandboxed: true,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::EffectRequiresCommit));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn local_ephemeral_requires_a_sandbox() {
    let h = harness(
        vec![tool(
            "build.compile",
            ToolEffects::SPAWN_PROCESS | ToolEffects::LOCAL_EPHEMERAL,
        )],
        CallOutcome::Finished(json!({"ok": true})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker.clone());
    let err = lane
        .invoke_speculative(SpeculativeToolCall {
            provider: h.provider.clone(),
            tool_name: "build.compile".into(),
            arguments: json!({}),
            sandboxed: false,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::SandboxRequired));
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);

    lane.invoke_speculative(SpeculativeToolCall {
        provider: h.provider,
        tool_name: "build.compile".into(),
        arguments: json!({}),
        sandboxed: true,
    })
    .await
    .unwrap();
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn irreversible_work_is_staged_and_the_session_is_not_called() {
    let h = harness(
        vec![tool(
            "mail.send",
            ToolEffects::EXTERNAL_IRREVERSIBLE | ToolEffects::SECRET_BEARING,
        )],
        CallOutcome::Finished(json!({})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker);
    let intent = lane
        .stage_effect_intent(&h.provider, "mail.send", json!({"to": "a@b.c"}))
        .await
        .unwrap();
    assert_eq!(intent.tool_name, "mail.send");
    assert_eq!(h.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn speculation_safe_tool_is_not_staged_as_an_external_effect() {
    let h = harness(
        vec![tool("mail.search", ToolEffects::READ_NETWORK)],
        CallOutcome::Finished(json!({})),
    )
    .await;
    let lane = SpeculativeLane::new(h.broker);
    let err = lane
        .stage_effect_intent(&h.provider, "mail.search", json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::SpeculationSafe));
}

#[tokio::test]
async fn effect_lane_calls_the_warm_session_when_the_catalog_digest_matches() {
    let h = harness(
        vec![tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)],
        CallOutcome::Finished(json!({"id": "m1"})),
    )
    .await;
    let speculative = SpeculativeLane::new(h.broker.clone());
    let snapshot = speculative.discovery_snapshot(&h.provider).await.unwrap();
    let epoch = snapshot.session_epoch;
    let intent = speculative
        .stage_effect_intent(&h.provider, "mail.send", json!({"to": "a@b.c"}))
        .await
        .unwrap();
    let authorized = authorize_for_test(
        intent,
        WorldId(7),
        JNodeId(3),
        Digest32::of(b"policy"),
        snapshot.catalog_digest,
        EffectId::from_label("send-1"),
    );
    let receipt = EffectLane::new(h.broker.clone())
        .execute_effect(authorized)
        .await
        .unwrap();
    assert_eq!(receipt.output, json!({"id": "m1"}));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);

    let after = speculative.discovery_snapshot(&h.provider).await.unwrap();
    assert_eq!(after.session_epoch, epoch);
}

#[tokio::test]
async fn stale_catalog_digest_does_not_call_the_tool() {
    let tools = vec![tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)];
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let wire = MemoryWire::new(tools.clone(), move |_name, _args| {
        counted.fetch_add(1, Ordering::SeqCst);
        CallOutcome::Finished(json!({"id": "m1"}))
    });
    let broker = McpBroker::new();
    let provider = ProviderId::new("mail");
    broker
        .admit(provider.clone(), Arc::new(wire))
        .await
        .unwrap();
    let speculative = SpeculativeLane::new(broker.clone());
    let snapshot = speculative.discovery_snapshot(&provider).await.unwrap();
    let intent = speculative
        .stage_effect_intent(&provider, "mail.send", json!({}))
        .await
        .unwrap();
    let authorized = authorize_for_test(
        intent,
        WorldId(1),
        JNodeId(1),
        Digest32::of(b"policy"),
        snapshot.catalog_digest,
        EffectId::from_label("send-stale"),
    );

    let changed = vec![
        tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE),
        tool("mail.search", ToolEffects::READ_NETWORK),
    ];
    broker.replace_catalog(&provider, changed).await.unwrap();

    let err = EffectLane::new(broker)
        .execute_effect(authorized)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::StaleCapability));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_completed_effect_is_not_sent_twice() {
    let h = harness(
        vec![tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)],
        CallOutcome::Finished(json!({"id": "m1"})),
    )
    .await;
    let speculative = SpeculativeLane::new(h.broker.clone());
    let snapshot = speculative.discovery_snapshot(&h.provider).await.unwrap();
    let intent = speculative
        .stage_effect_intent(&h.provider, "mail.send", json!({}))
        .await
        .unwrap();
    let key = EffectId::from_label("send-once");
    let first = authorize_for_test(
        intent.clone(),
        WorldId(1),
        JNodeId(1),
        Digest32::of(b"policy"),
        snapshot.catalog_digest,
        key,
    );
    let lane = EffectLane::new(h.broker);
    let receipt = lane.execute_effect(first).await.unwrap();
    let second = authorize_for_test(
        intent,
        WorldId(1),
        JNodeId(1),
        Digest32::of(b"policy"),
        snapshot.catalog_digest,
        key,
    );
    let replay = lane.execute_effect(second).await.unwrap();
    assert_eq!(receipt, replay);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_uncertain_call_is_reconciled_instead_of_retried() {
    let h = harness(
        vec![tool("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)],
        CallOutcome::Uncertain,
    )
    .await;
    let speculative = SpeculativeLane::new(h.broker.clone());
    let snapshot = speculative.discovery_snapshot(&h.provider).await.unwrap();
    let intent = speculative
        .stage_effect_intent(&h.provider, "mail.send", json!({}))
        .await
        .unwrap();
    let key = EffectId::from_label("send-uncertain");
    let lane = EffectLane::new(h.broker);
    let first = lane
        .execute_effect(authorize_for_test(
            intent.clone(),
            WorldId(1),
            JNodeId(1),
            Digest32::of(b"policy"),
            snapshot.catalog_digest,
            key,
        ))
        .await
        .unwrap_err();
    assert!(matches!(first, Error::ReconciliationRequired));
    let second = lane
        .execute_effect(authorize_for_test(
            intent,
            WorldId(1),
            JNodeId(1),
            Digest32::of(b"policy"),
            snapshot.catalog_digest,
            key,
        ))
        .await
        .unwrap_err();
    assert!(matches!(second, Error::ReconciliationRequired));
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);
}
