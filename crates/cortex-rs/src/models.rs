#![allow(dead_code)]
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

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CortexEntity {
    pub id: String,
    pub space_id: String,
    pub space_slug: String,
    pub entity_type: String,
    pub canonical_name: String,
    pub content: String,
    pub aliases: Vec<String>,
    pub metadata: Value,
    pub confidence: f64,
    pub revision: i64,
    pub retracted: bool,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CortexClaim {
    pub id: String,
    pub space_id: String,
    pub space_slug: String,
    pub subject_entity_id: String,
    pub predicate: String,
    pub object_entity_id: Option<String>,
    pub literal_value: Option<Value>,
    pub confidence: f64,
    pub metadata: Value,
    pub retracted: bool,
    pub created_at: String,
    pub verification_tier: VerificationTier,
    pub status: ClaimStatus,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptDetails {
    pub id: String,
    pub operation: String,
    pub target_type: String,
    pub target_id: String,
    pub committed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexReceipt {
    pub recorded: bool,
    pub receipt: ReceiptDetails,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityWriteInput {
    pub id: Option<String>,
    #[serde(default = "default_space")]
    pub space: String,
    #[serde(default = "default_entity_type")]
    pub entity_type: String,
    pub canonical_name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_json_object")]
    pub metadata: Value,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub external_id: Option<String>,
}

fn default_space() -> String {
    "atlas-memory".to_string()
}
fn default_entity_type() -> String {
    "discovery".to_string()
}
fn default_json_object() -> Value {
    serde_json::json!({})
}
fn default_confidence() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimWriteInput {
    pub id: Option<String>,
    #[serde(default = "default_space")]
    pub space: String,
    pub subject_entity_id: String,
    pub predicate: String,
    pub object_entity_id: Option<String>,
    pub literal_value: Option<Value>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default = "default_json_object")]
    pub metadata: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetractInput {
    pub target_type: String,
    pub target_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub enum WritePayload {
    #[serde(rename = "entity")]
    Entity { value: EntityWriteInput },
    #[serde(rename = "claim")]
    Claim { value: ClaimWriteInput },
    #[serde(rename = "retract")]
    Retract { value: RetractInput },
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchParams {
    #[serde(default)]
    pub q: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    pub space: Option<String>,
    #[serde(default = "default_search_limit")]
    pub limit: Option<usize>,
    #[serde(default)]
    pub include_retracted: Option<bool>,
}

fn default_search_limit() -> Option<usize> {
    Some(12)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    #[serde(flatten)]
    pub entity: CortexEntity,
    pub lexical_score: f64,
    pub graph_score: f64,
    pub semantic_score: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub degraded: Vec<String>,
    #[serde(rename = "elapsedMs")]
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecallPayload {
    pub query: String,
    pub space: Option<String>,
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
}

fn default_recall_limit() -> usize {
    5
}
fn default_token_budget() -> usize {
    1200
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecallComponentScores {
    pub lexical: f64,
    pub semantic: f64,
    pub confidence_recency: f64,
    pub graph: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecallItem {
    pub entity: CortexEntity,
    pub final_score: f64,
    pub component_scores: RecallComponentScores,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallResponse {
    pub results: Vec<RecallItem>,
    pub degraded: Vec<String>,
    #[serde(rename = "elapsedMs")]
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraversePayload {
    pub subject_id: String,
    pub space: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CortexSession {
    pub id: String,
    pub space_slug: String,
    pub agent_id: Option<String>,
    pub world_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub fork_event_id: Option<String>,
    pub created_at: String,
    pub closed_at: Option<String>,
    pub status: String,
    pub retention_class: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionInput {
    pub id: Option<String>,
    #[serde(default = "default_space")]
    pub space: String,
    pub agent_id: Option<String>,
    pub world_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub fork_event_id: Option<String>,
    #[serde(default = "default_retention_class")]
    pub retention_class: String,
    #[serde(default = "default_json_object")]
    pub metadata: Value,
}

fn default_retention_class() -> String {
    "standard".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CortexSessionEvent {
    pub id: String,
    pub session_id: String,
    pub sequence: i64,
    pub branch_id: String,
    pub parent_event_id: Option<String>,
    pub event_type: String,
    pub role: Option<String>,
    pub content: Option<String>,
    pub payload: Value,
    pub created_at: String,
    pub content_hash: String,
    pub sensitivity: Option<String>,
    pub redacted: bool,
    pub segment_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEventInput {
    pub id: Option<String>,
    pub branch_id: Option<String>,
    pub parent_event_id: Option<String>,
    pub event_type: String,
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default = "default_json_object")]
    pub payload: Value,
    pub sensitivity: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchAppendEventsInput {
    #[serde(default = "default_branch_id")]
    pub branch_id: String,
    pub events: Vec<SessionEventInput>,
}

fn default_branch_id() -> String {
    "main".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessingWatermark {
    pub session_id: String,
    pub processor_kind: String,
    pub processor_version: String,
    pub watermark_seq: i64,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetWatermarkInput {
    pub processor_kind: String,
    pub processor_version: String,
    pub watermark_seq: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventSegment {
    pub id: String,
    pub session_id: String,
    pub branch_id: String,
    pub start_seq: i64,
    pub end_seq: i64,
    pub merkle_root: String,
    pub prev_segment_root: Option<String>,
    pub sealed_at: String,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: String,
    pub session_id: String,
    pub branch_id: String,
    pub level: i64,
    pub start_seq: i64,
    pub end_seq: i64,
    pub summary_text: String,
    pub source_hash: String,
    pub processor_version: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerificationTier {
    T0Direct,
    T1Corroborated,
    T2Verified,
    T3Controlled,
}

impl VerificationTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::T0Direct => "T0Direct",
            Self::T1Corroborated => "T1Corroborated",
            Self::T2Verified => "T2Verified",
            Self::T3Controlled => "T3Controlled",
        }
    }

    pub fn from_str_opt(s: &str) -> Self {
        match s {
            "T1Corroborated" => Self::T1Corroborated,
            "T2Verified" => Self::T2Verified,
            "T3Controlled" => Self::T3Controlled,
            _ => Self::T0Direct,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Active,
    Superseded,
    Disputed,
    Retracted,
    Invalidated,
}

impl ClaimStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Disputed => "disputed",
            Self::Retracted => "retracted",
            Self::Invalidated => "invalidated",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryCandidate {
    pub id: String,
    pub space_slug: String,
    pub session_id: String,
    pub branch_id: String,
    pub memory_type: String,
    pub subject: String,
    pub predicate: String,
    pub object_value: Value,
    pub scope: String,
    pub confidence: f64,
    pub verification_tier: VerificationTier,
    pub state: String,
    pub extractor_version: String,
    pub created_at: String,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateWriteInput {
    pub id: Option<String>,
    #[serde(default = "default_space")]
    pub space: String,
    pub session_id: String,
    #[serde(default = "default_branch_id")]
    pub branch_id: String,
    pub memory_type: String,
    pub subject: String,
    pub predicate: String,
    pub object_value: Value,
    #[serde(default = "default_scope")]
    pub scope: String,
    pub confidence: f64,
    pub verification_tier: Option<VerificationTier>,
    pub extractor_version: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

fn default_scope() -> String {
    "global".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionReceipt {
    pub id: String,
    pub candidate_id: String,
    pub claim_id: String,
    pub policy_id: String,
    pub policy_version: String,
    pub achieved_tier: String,
    pub verifier_receipt: Option<String>,
    pub promoted_at: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteCandidateInput {
    pub candidate_id: String,
    pub policy_id: Option<String>,
    pub verifier_receipt: Option<String>,
}
