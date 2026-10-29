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

use aien_localhttp::Client;
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::db::Database;
use crate::embeddings::fetch_embedding;
use crate::models::*;

pub struct AppState {
    pub db: Arc<Database>,
    pub http_client: Client,
    pub encoder_url: String,
    pub worker_limit: Arc<Semaphore>,
}

pub async fn write_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<WritePayload>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    match payload {
        WritePayload::Entity { value } => {
            let mut emb = None;
            let text_to_embed = if !value.content.trim().is_empty() {
                format!("{}: {}", value.canonical_name, value.content)
            } else {
                value.canonical_name.clone()
            };

            match fetch_embedding(&state.http_client, &text_to_embed, &state.encoder_url).await {
                Ok(vec) => emb = Some(vec),
                Err(e) => {
                    tracing::warn!(
                        "Embedding encoder unavailable, proceeding with lexical only: {}",
                        e
                    );
                }
            }

            match state.db.upsert_entity(&value, emb.as_deref()) {
                Ok(receipt) => Ok((
                    StatusCode::CREATED,
                    Json(CortexReceipt {
                        recorded: true,
                        receipt,
                    }),
                )
                    .into_response()),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )),
            }
        }
        WritePayload::Claim { value } => match state.db.upsert_claim(&value) {
            Ok(receipt) => Ok((
                StatusCode::CREATED,
                Json(CortexReceipt {
                    recorded: true,
                    receipt,
                }),
            )
                .into_response()),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )),
        },
        WritePayload::Retract { value } => {
            match state
                .db
                .retract_target(&value.target_type, &value.target_id)
            {
                Ok(receipt) => Ok((
                    StatusCode::CREATED,
                    Json(CortexReceipt {
                        recorded: true,
                        receipt,
                    }),
                )
                    .into_response()),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                )),
            }
        }
    }
}

pub async fn search_get_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start = Instant::now();
    let q = params.q.or(params.query).unwrap_or_default();
    let limit = params.limit.unwrap_or(12).min(50);
    let space = params.space.as_deref();

    match state.db.search_entities(&q, space, limit) {
        Ok(results) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(Json(SearchResponse {
                results,
                degraded: Vec::new(),
                elapsed_ms,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn search_post_handler(
    State(state): State<Arc<AppState>>,
    Json(params): Json<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start = Instant::now();
    let q = params.q.or(params.query).unwrap_or_default();
    let limit = params.limit.unwrap_or(12).min(50);
    let space = params.space.as_deref();

    match state.db.search_entities(&q, space, limit) {
        Ok(results) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(Json(SearchResponse {
                results,
                degraded: Vec::new(),
                elapsed_ms,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn recall_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RecallPayload>,
) -> Result<Json<RecallResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start = Instant::now();
    let space = payload.space.as_deref();
    let limit = payload.limit.min(20);

    let mut emb = None;
    match fetch_embedding(&state.http_client, &payload.query, &state.encoder_url).await {
        Ok(vec) => emb = Some(vec),
        Err(e) => {
            tracing::warn!("Embedding encoder unavailable for recall: {}", e);
        }
    }

    match state
        .db
        .recall_entities(&payload.query, emb.as_deref(), space, limit)
    {
        Ok(results) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(Json(RecallResponse {
                results,
                degraded: Vec::new(),
                elapsed_ms,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

#[derive(serde::Deserialize)]
pub struct GetParams {
    pub id: Option<String>,
    pub name: Option<String>,
    pub space: Option<String>,
}

pub async fn get_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GetParams>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let target = params.id.or(params.name).unwrap_or_default();
    if target.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Missing id or name parameter"})),
        ));
    }

    match state.db.get_entity(&target, params.space.as_deref()) {
        Ok(Some(entity)) => Ok(Json(entity).into_response()),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Entity not found"})),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn traverse_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TraversePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state
        .db
        .traverse_claims(&payload.subject_id, payload.space.as_deref())
    {
        Ok(claims) => Ok(Json(json!({"claims": claims}))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "cortex-rs",
        "version": "0.1.0",
        "space": "atlas-memory",
        "runtime": "native-arm64-rust"
    }))
}

// =========================================================================
// Episodic Plane Handlers (v2 API)
// =========================================================================

#[derive(serde::Deserialize)]
pub struct SessionEventsQuery {
    pub branch_id: Option<String>,
    pub after_seq: Option<i64>,
    pub limit: Option<usize>,
    pub recent: Option<bool>,
}

#[derive(serde::Deserialize)]
pub struct WatermarkQuery {
    pub kind: String,
    pub version: String,
}

pub async fn create_session_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<CreateSessionInput>,
) -> Result<(StatusCode, Json<CortexSession>), (StatusCode, Json<serde_json::Value>)> {
    match state.db.create_session(&input) {
        Ok(session) => Ok((StatusCode::CREATED, Json(session))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn get_session_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<CortexSession>, (StatusCode, Json<serde_json::Value>)> {
    match state.db.get_session(&session_id) {
        Ok(Some(session)) => Ok(Json(session)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Session {} not found", session_id)})),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn close_session_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state.db.close_session(&session_id) {
        Ok(()) => Ok(Json(json!({"status": "closed", "sessionId": session_id}))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn batch_append_events_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
    Json(payload): Json<BatchAppendEventsInput>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    match state
        .db
        .batch_append_events(&session_id, &payload.branch_id, &payload.events)
    {
        Ok(events) => Ok((StatusCode::CREATED, Json(json!({ "events": events })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn get_session_events_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
    Query(params): Query<SessionEventsQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let branch = params.branch_id.as_deref().unwrap_or("main");
    let limit = params.limit.unwrap_or(50).min(500);
    let fetched = if params.recent.unwrap_or(false) && params.after_seq.is_none() {
        state
            .db
            .get_recent_session_events(&session_id, branch, limit)
    } else {
        state
            .db
            .get_session_events(&session_id, branch, params.after_seq, limit)
    };
    match fetched {
        Ok(events) => Ok(Json(json!({ "events": events }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn get_watermark_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
    Query(params): Query<WatermarkQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state
        .db
        .get_watermark(&session_id, &params.kind, &params.version)
    {
        Ok(Some(seq)) => Ok(Json(json!({
            "sessionId": session_id,
            "processorKind": params.kind,
            "processorVersion": params.version,
            "watermarkSeq": seq
        }))),
        Ok(None) => Ok(Json(json!({
            "sessionId": session_id,
            "processorKind": params.kind,
            "processorVersion": params.version,
            "watermarkSeq": null
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn set_watermark_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
    Json(input): Json<SetWatermarkInput>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state.db.set_watermark(
        &session_id,
        &input.processor_kind,
        &input.processor_version,
        input.watermark_seq,
    ) {
        Ok(()) => Ok(Json(json!({
            "status": "ok",
            "sessionId": session_id,
            "processorKind": input.processor_kind,
            "processorVersion": input.processor_version,
            "watermarkSeq": input.watermark_seq
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

#[derive(serde::Deserialize)]
pub struct CandidatesQuery {
    pub space: Option<String>,
    pub state: Option<String>,
    pub limit: Option<usize>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteCandidatePayload {
    pub policy_id: Option<String>,
    pub verifier_receipt: Option<String>,
}

pub async fn get_candidates_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CandidatesQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let limit = params.limit.unwrap_or(50);
    match state
        .db
        .get_memory_candidates(params.space.as_deref(), params.state.as_deref(), limit)
    {
        Ok(candidates) => Ok(Json(json!({ "candidates": candidates }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

pub async fn get_candidate_by_id_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(candidate_id): AxumPath<String>,
) -> Result<Json<MemoryCandidate>, (StatusCode, Json<serde_json::Value>)> {
    match state.db.get_candidate_by_id(&candidate_id) {
        Ok(Some(candidate)) => Ok(Json(candidate)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Candidate {} not found", candidate_id) })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

pub async fn promote_candidate_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(candidate_id): AxumPath<String>,
    Json(payload): Json<PromoteCandidatePayload>,
) -> Result<(StatusCode, Json<PromotionReceipt>), (StatusCode, Json<serde_json::Value>)> {
    let input = PromoteCandidateInput {
        candidate_id,
        policy_id: payload.policy_id,
        verifier_receipt: payload.verifier_receipt,
    };
    match state.db.promote_candidate(&input) {
        Ok(receipt) => Ok((StatusCode::CREATED, Json(receipt))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

pub async fn compile_context_handler(
    State(state): State<Arc<AppState>>,
    Json(query): Json<crate::compiler::MemoryQuery>,
) -> Result<Json<crate::compiler::MemoryContext>, (StatusCode, Json<serde_json::Value>)> {
    let compiler = crate::compiler::MemoryCompiler::new(Arc::clone(&state.db));
    match compiler.compile_context(&query) {
        Ok(context) => Ok(Json(context)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

pub async fn process_session_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(session_id): AxumPath<String>,
    Query(params): Query<SessionEventsQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let branch = params.branch_id.as_deref().unwrap_or("main").to_string();
    let branch_for_worker = branch.clone();
    let permit = state
        .worker_limit
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({ "error": "worker capacity exceeded" })),
            )
        })?;
    let db = Arc::clone(&state.db);
    let session_for_worker = session_id.clone();
    let joined = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::task::spawn_blocking(move || {
            let _permit: OwnedSemaphorePermit = permit;
            crate::workers::ProcessingScheduler::new(db)
                .process_session(&session_for_worker, &branch_for_worker)
        }),
    )
    .await
    .map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "worker deadline exceeded" })),
        )
    })?
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
    })?;
    match joined {
        Ok((summary, candidates)) => Ok(Json(json!({
            "status": "processed",
            "sessionId": session_id,
            "branchId": branch,
            "summary": summary,
            "candidates": candidates
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> Arc<AppState> {
        let db = Arc::new(Database::open_in_memory().expect("in memory db"));
        Arc::new(AppState {
            db,
            http_client: Client::new(),
            encoder_url: "http://127.0.0.1:18081".to_string(),
            worker_limit: Arc::new(Semaphore::new(2)),
        })
    }

    #[tokio::test]
    async fn test_get_handler_missing_parameter() {
        let state = create_test_state();
        let query = GetParams {
            id: None,
            name: None,
            space: None,
        };
        let err = get_handler(State(state), Query(query)).await.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(err.1 .0["error"], "Missing id or name parameter");
    }

    #[tokio::test]
    async fn test_get_handler_not_found() {
        let state = create_test_state();
        let query = GetParams {
            id: Some("non_existent_uuid".to_string()),
            name: None,
            space: Some("atlas-memory".to_string()),
        };
        let err = get_handler(State(state), Query(query)).await.unwrap_err();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
        assert_eq!(err.1 .0["error"], "Entity not found");
    }

    #[tokio::test]
    async fn test_get_handler_success_and_unknown_space() {
        let state = create_test_state();
        let input = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "discovery".to_string(),
            canonical_name: "test_entity_get".to_string(),
            content: "Payload content".to_string(),
            aliases: vec![],
            metadata: json!({"key": "val"}),
            confidence: 1.0,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };
        state.db.upsert_entity(&input, None).unwrap();

        // 1. Success fetch
        let query_ok = GetParams {
            id: None,
            name: Some("test_entity_get".to_string()),
            space: Some("atlas-memory".to_string()),
        };
        let resp = get_handler(State(state.clone()), Query(query_ok)).await;
        assert!(resp.is_ok());

        // 2. Unknown space fetch returns 404
        let query_unknown_space = GetParams {
            id: None,
            name: Some("test_entity_get".to_string()),
            space: Some("non_existent_space".to_string()),
        };
        let err_unknown = get_handler(State(state), Query(query_unknown_space))
            .await
            .unwrap_err();
        assert_eq!(err_unknown.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_search_unknown_space_returns_empty() {
        let state = create_test_state();
        let params = SearchParams {
            q: Some("anything".to_string()),
            query: None,
            space: Some("completely_unknown_space_slug".to_string()),
            limit: Some(10),
            include_retracted: Some(false),
        };
        let Json(res) = search_get_handler(State(state), Query(params))
            .await
            .unwrap();
        assert_eq!(res.results.len(), 0);
    }

    #[tokio::test]
    async fn test_write_entity_and_retract_lifecycle() {
        let state = create_test_state();

        // 1. Write entity
        let write_payload = WritePayload::Entity {
            value: EntityWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                entity_type: "lesson".to_string(),
                canonical_name: "lifecycle_test".to_string(),
                content: "Lifecycle verification".to_string(),
                aliases: vec![],
                metadata: json!({}),
                confidence: 1.0,
                valid_from: None,
                valid_to: None,
                external_id: None,
            },
        };
        let write_res = write_handler(State(state.clone()), Json(write_payload)).await;
        assert!(write_res.is_ok());

        let entity = state
            .db
            .get_entity("lifecycle_test", Some("atlas-memory"))
            .unwrap()
            .unwrap();
        assert_eq!(entity.canonical_name, "lifecycle_test");

        // 2. Retract entity
        let retract_payload = WritePayload::Retract {
            value: RetractInput {
                target_type: "entity".to_string(),
                target_id: entity.id.clone(),
                reason: Some("Obsolescence".to_string()),
            },
        };
        let retract_res = write_handler(State(state.clone()), Json(retract_payload)).await;
        assert!(retract_res.is_ok());

        let after = state
            .db
            .get_entity("lifecycle_test", Some("atlas-memory"))
            .unwrap();
        assert!(after.is_none());
    }

    #[tokio::test]
    async fn test_health_handler_structure() {
        let Json(health) = health_handler().await;
        assert_eq!(health["status"], "ok");
        assert_eq!(health["service"], "cortex-rs");
        assert_eq!(health["space"], "atlas-memory");
    }

    #[tokio::test]
    async fn test_v2_session_and_events_api_lifecycle() {
        let state = create_test_state();

        // 1. Create session
        let session_input = CreateSessionInput {
            id: None,
            space: "atlas-memory".to_string(),
            agent_id: Some("agent-atlas".to_string()),
            world_id: Some("world-01".to_string()),
            parent_session_id: None,
            fork_event_id: None,
            retention_class: "standard".to_string(),
            metadata: json!({"task": "concurrency_test"}),
        };

        let (status, Json(session)) =
            create_session_handler(State(state.clone()), Json(session_input))
                .await
                .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(session.status, "active");
        assert_eq!(session.agent_id, Some("agent-atlas".to_string()));

        let session_id = session.id;

        // 2. Get session
        let Json(fetched_session) =
            get_session_handler(State(state.clone()), AxumPath(session_id.clone()))
                .await
                .unwrap();
        assert_eq!(fetched_session.id, session_id);
        assert_eq!(fetched_session.status, "active");

        // 3. Append events
        let batch_input = BatchAppendEventsInput {
            branch_id: "main".to_string(),
            events: vec![
                SessionEventInput {
                    id: Some("ev-001".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: None,
                    event_type: "user_message".to_string(),
                    role: Some("user".to_string()),
                    content: Some("Deploy the memory engine.".to_string()),
                    payload: json!({}),
                    sensitivity: None,
                },
                SessionEventInput {
                    id: Some("ev-002".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: Some("ev-001".to_string()),
                    event_type: "assistant_message".to_string(),
                    role: Some("assistant".to_string()),
                    content: Some("Initializing Cortex multi-plane architecture.".to_string()),
                    payload: json!({}),
                    sensitivity: None,
                },
            ],
        };

        let (status, Json(batch_res)) = batch_append_events_handler(
            State(state.clone()),
            AxumPath(session_id.clone()),
            Json(batch_input),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        let events = batch_res["events"].as_array().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["sequence"], 1);
        assert_eq!(events[1]["sequence"], 2);

        // 4. Test idempotency: re-append ev-001
        let replay_input = BatchAppendEventsInput {
            branch_id: "main".to_string(),
            events: vec![SessionEventInput {
                id: Some("ev-001".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: None,
                event_type: "user_message".to_string(),
                role: Some("user".to_string()),
                content: Some("Deploy the memory engine.".to_string()),
                payload: json!({}),
                sensitivity: None,
            }],
        };
        let (_, Json(replay_res)) = batch_append_events_handler(
            State(state.clone()),
            AxumPath(session_id.clone()),
            Json(replay_input),
        )
        .await
        .unwrap();
        let replayed = replay_res["events"].as_array().unwrap();
        assert_eq!(replayed[0]["id"], "ev-001");
        assert_eq!(replayed[0]["sequence"], 1);

        // 5. Query events
        let query = SessionEventsQuery {
            branch_id: Some("main".to_string()),
            after_seq: Some(0),
            limit: Some(10),
            recent: None,
        };
        let Json(queried) = get_session_events_handler(
            State(state.clone()),
            AxumPath(session_id.clone()),
            Query(query),
        )
        .await
        .unwrap();
        assert_eq!(queried["events"].as_array().unwrap().len(), 2);

        // 6. Watermarks
        let set_wm = SetWatermarkInput {
            processor_kind: "summary".to_string(),
            processor_version: "v1".to_string(),
            watermark_seq: 2,
        };
        let Json(wm_res) = set_watermark_handler(
            State(state.clone()),
            AxumPath(session_id.clone()),
            Json(set_wm),
        )
        .await
        .unwrap();
        assert_eq!(wm_res["status"], "ok");

        let wm_query = WatermarkQuery {
            kind: "summary".to_string(),
            version: "v1".to_string(),
        };
        let Json(get_wm) = get_watermark_handler(
            State(state.clone()),
            AxumPath(session_id.clone()),
            Query(wm_query),
        )
        .await
        .unwrap();
        assert_eq!(get_wm["watermarkSeq"], 2);

        // 7. Close session
        let Json(close_res) =
            close_session_handler(State(state.clone()), AxumPath(session_id.clone()))
                .await
                .unwrap();
        assert_eq!(close_res["status"], "closed");

        let Json(closed_session) =
            get_session_handler(State(state.clone()), AxumPath(session_id.clone()))
                .await
                .unwrap();
        assert_eq!(closed_session.status, "closed");
        assert!(closed_session.closed_at.is_some());
    }

    #[tokio::test]
    async fn test_v2_compiler_and_process_api_e2e() {
        let state = create_test_state();

        // 1. Create session
        let (status, Json(session)) = create_session_handler(
            State(state.clone()),
            Json(CreateSessionInput {
                id: Some("sess-e2e-compiler".to_string()),
                space: "atlas-memory".to_string(),
                agent_id: Some("atlas-prime".to_string()),
                world_id: None,
                parent_session_id: None,
                fork_event_id: None,
                retention_class: "standard".to_string(),
                metadata: json!({}),
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);

        // 2. Append events (user preference + tool discovery)
        let batch = BatchAppendEventsInput {
            branch_id: "main".to_string(),
            events: vec![
                SessionEventInput {
                    id: Some("ev-e2e-1".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: None,
                    event_type: "user_message".to_string(),
                    role: Some("user".to_string()),
                    content: Some("I prefer Rust for the sovereign stack.".to_string()),
                    payload: json!({}),
                    sensitivity: None,
                },
                SessionEventInput {
                    id: Some("ev-e2e-2".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: Some("ev-e2e-1".to_string()),
                    event_type: "tool_result".to_string(),
                    role: Some("tool".to_string()),
                    content: Some("Verified SQLite WAL mode active on Boston.".to_string()),
                    payload: json!({"verified": true}),
                    sensitivity: None,
                },
            ],
        };
        let _ = batch_append_events_handler(
            State(state.clone()),
            AxumPath(session.id.clone()),
            Json(batch),
        )
        .await
        .unwrap();

        // 3. Trigger asynchronous workers on session
        let query_proc = SessionEventsQuery {
            branch_id: Some("main".to_string()),
            after_seq: None,
            limit: Some(50),
            recent: None,
        };
        let Json(proc_res) = process_session_handler(
            State(state.clone()),
            AxumPath(session.id.clone()),
            Query(query_proc),
        )
        .await
        .unwrap();
        assert_eq!(proc_res["status"], "processed");
        assert!(proc_res["summary"].is_object());
        let candidates = proc_res["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 2);

        // 4. Query candidates list
        let Json(cands_list) = get_candidates_handler(
            State(state.clone()),
            Query(CandidatesQuery {
                space: Some("atlas-memory".to_string()),
                state: Some("pending".to_string()),
                limit: Some(10),
            }),
        )
        .await
        .unwrap();
        assert!(!cands_list["candidates"].as_array().unwrap().is_empty());

        let cand_id = candidates[0]["id"].as_str().unwrap();

        // 5. Promote candidate
        let (p_status, Json(p_receipt)) = promote_candidate_handler(
            State(state.clone()),
            AxumPath(cand_id.to_string()),
            Json(PromoteCandidatePayload {
                policy_id: Some("e2e-policy".to_string()),
                verifier_receipt: Some("e2e-verified".to_string()),
            }),
        )
        .await
        .unwrap();
        assert_eq!(p_status, StatusCode::CREATED);
        assert_eq!(p_receipt.candidate_id, cand_id);

        // 6. Compile Context
        let mem_query = crate::compiler::MemoryQuery {
            query: "user".to_string(),
            session_id: Some(session.id.clone()),
            branch_id: Some("main".to_string()),
            space: "atlas-memory".to_string(),
            intent: crate::compiler::MemoryIntent::Conversation,
            token_budget: 1024,
            include_candidates: true,
            minimum_verification: None,
        };

        let Json(context) = compile_context_handler(State(state.clone()), Json(mem_query))
            .await
            .unwrap();

        assert!(!context.episodic.is_empty());
        assert!(!context.canonical.is_empty());
        assert!(context.rendered_prompt.contains("<CORTEX_MEMORY>"));
        assert!(context.rendered_prompt.contains("<EPISODIC_CONTEXT>"));
        assert!(context.rendered_prompt.contains("<VERIFIED_KNOWLEDGE>"));
    }
}
