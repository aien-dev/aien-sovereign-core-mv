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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aien_capability::{Digest32, EffectId, JNodeId, ProviderId, ToolEffects, WorldId};
use rmcp::model::JsonObject;
use serde_json::{json, Value};

use crate::transport::LocalTool;
use crate::{authorize_for_test, Error, LocalToolServer, SessionManager, SpeculativeToolCall};

fn object_schema(extra: bool) -> JsonObject {
    let mut schema = json!({
        "type": "object",
        "properties": { "q": { "type": "string" } }
    });
    if extra {
        schema["properties"]["limit"] = json!({ "type": "integer" });
    }
    match schema {
        Value::Object(map) => map,
        _ => JsonObject::new(),
    }
}

fn effects(names: &[(&str, ToolEffects)]) -> HashMap<String, ToolEffects> {
    names
        .iter()
        .map(|(name, effects)| ((*name).to_string(), *effects))
        .collect()
}

struct Linked {
    sessions: SessionManager,
    server: LocalToolServer,
    provider: ProviderId,
    calls: Arc<AtomicUsize>,
}

async fn link(enrolled: HashMap<String, ToolEffects>) -> Linked {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let server = LocalToolServer::new(vec![LocalTool::new(
        "mail.search",
        "search mail",
        object_schema(false),
        move |arguments| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"echo": arguments}))
        },
    )]);
    let (server_io, client_io) = tokio::io::duplex(8 * 1024);
    let background = server.clone();
    tokio::spawn(async move {
        if let Ok(running) = background.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    let sessions = SessionManager::new();
    let provider = ProviderId::new("mail");
    sessions
        .enroll_transport(provider.clone(), client_io, enrolled)
        .await
        .unwrap();
    Linked {
        sessions,
        server,
        provider,
        calls,
    }
}

#[tokio::test]
async fn an_unenrolled_server_is_not_visible_to_the_speculative_lane() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let _server = LocalToolServer::new(vec![LocalTool::new(
        "mail.search",
        "search mail",
        object_schema(false),
        move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(json!({}))
        },
    )]);
    let sessions = SessionManager::new();
    let err = sessions
        .speculative_lane()
        .discovery_snapshot(&ProviderId::new("mail"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotAdmitted(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn discovery_refreshes_an_admitted_provider_over_the_duplex() {
    let linked = link(effects(&[("mail.search", ToolEffects::READ_NETWORK)])).await;
    let lane = linked.sessions.speculative_lane();
    let first = lane.discovery_snapshot(&linked.provider).await.unwrap();
    assert_eq!(first.tools.len(), 1);
    assert_eq!(first.tools[0].name(), "mail.search");
    assert_eq!(first.tools[0].effects(), ToolEffects::READ_NETWORK);
    assert_eq!(linked.calls.load(Ordering::SeqCst), 0);

    let result = lane
        .invoke_speculative(SpeculativeToolCall {
            provider: linked.provider.clone(),
            tool_name: "mail.search".into(),
            arguments: json!({"q": "inbox"}),
            sandboxed: false,
        })
        .await
        .unwrap();
    assert_eq!(result.output["echo"]["q"], json!("inbox"));
    assert_eq!(linked.calls.load(Ordering::SeqCst), 1);

    linked.server.replace_tools(vec![LocalTool::new(
        "mail.search",
        "search mail",
        object_schema(true),
        |_| Ok(json!({})),
    )]);
    let refreshed = lane.discovery_snapshot(&linked.provider).await.unwrap();
    assert_ne!(refreshed.catalog_digest, first.catalog_digest);
    assert_eq!(refreshed.session_epoch, first.session_epoch + 1);
}

#[tokio::test]
async fn a_tool_without_an_enrolled_class_does_not_enter_the_snapshot() {
    let linked = link(effects(&[("mail.search", ToolEffects::READ_NETWORK)])).await;
    let lane = linked.sessions.speculative_lane();
    let before = lane.discovery_snapshot(&linked.provider).await.unwrap();
    linked.server.replace_tools(vec![
        LocalTool::new("mail.search", "search mail", object_schema(false), |_| {
            Ok(json!({}))
        }),
        LocalTool::new("mail.send", "send mail", object_schema(false), |_| {
            Ok(json!({"sent": true}))
        }),
    ]);
    let err = lane.discovery_snapshot(&linked.provider).await.unwrap_err();
    assert!(matches!(err, Error::UnclassifiedTool(_)));

    linked.server.replace_tools(vec![LocalTool::new(
        "mail.search",
        "search mail",
        object_schema(false),
        |_| Ok(json!({})),
    )]);
    let after = lane.discovery_snapshot(&linked.provider).await.unwrap();
    assert_eq!(after.catalog_digest, before.catalog_digest);
    assert_eq!(after.session_epoch, before.session_epoch);
}

#[tokio::test]
async fn irreversible_calls_stay_on_the_effect_lane() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let server = LocalToolServer::new(vec![LocalTool::new(
        "mail.send",
        "send mail",
        object_schema(false),
        move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"id": "m1"}))
        },
    )]);
    let (server_io, client_io) = tokio::io::duplex(8 * 1024);
    let background = server.clone();
    tokio::spawn(async move {
        if let Ok(running) = background.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    let sessions = SessionManager::new();
    let provider = ProviderId::new("mail");
    sessions
        .enroll_transport(
            provider.clone(),
            client_io,
            effects(&[("mail.send", ToolEffects::EXTERNAL_IRREVERSIBLE)]),
        )
        .await
        .unwrap();
    let lane = sessions.speculative_lane();
    let err = lane
        .invoke_speculative(SpeculativeToolCall {
            provider: provider.clone(),
            tool_name: "mail.send".into(),
            arguments: json!({}),
            sandboxed: true,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::EffectRequiresCommit));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let snapshot = lane.discovery_snapshot(&provider).await.unwrap();
    let intent = lane
        .stage_effect_intent(&provider, "mail.send", json!({}))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let receipt = sessions
        .effect_lane()
        .execute_effect(authorize_for_test(
            intent,
            WorldId(1),
            JNodeId(1),
            Digest32::of(b"policy"),
            snapshot.catalog_digest,
            EffectId::from_label("send-rmcp"),
        ))
        .await
        .unwrap();
    assert_eq!(receipt.output, json!({"id": "m1"}));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
