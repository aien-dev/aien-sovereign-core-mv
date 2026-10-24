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
use std::sync::Arc;
use uuid::Uuid;

use crate::db::Database;
use crate::models::{ClaimStatus, VerificationTier};
use crate::security::SecurityMembrane;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryIntent {
    #[default]
    Conversation,
    Coding,
    Planning,
    Research,
    ToolExecution,
    Reflection,
    MemoryAdjudication,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryQuery {
    pub query: String,
    pub session_id: Option<String>,
    pub branch_id: Option<String>,
    #[serde(default = "default_space")]
    pub space: String,
    #[serde(default)]
    pub intent: MemoryIntent,
    #[serde(default = "default_compiler_token_budget")]
    pub token_budget: usize,
    #[serde(default)]
    pub include_candidates: bool,
    pub minimum_verification: Option<VerificationTier>,
}

fn default_space() -> String {
    "atlas-memory".to_string()
}

fn default_compiler_token_budget() -> usize {
    2048
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalContextItem {
    pub text: String,
    pub claim_id: String,
    pub verification_tier: VerificationTier,
    pub status: ClaimStatus,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodicContextItem {
    pub kind: String,
    pub level: Option<i64>,
    pub sequence_range: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HypothesisContextItem {
    pub candidate_id: String,
    pub assertion: String,
    pub confidence: f64,
    pub verification: String,
    pub advisory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryContext {
    pub episodic: Vec<EpisodicContextItem>,
    pub canonical: Vec<CanonicalContextItem>,
    pub hypotheses: Vec<HypothesisContextItem>,
    pub rendered_prompt: String,
    pub omitted_count: usize,
    pub token_count: usize,
    pub context_revision: String,
}

pub struct MemoryCompiler {
    db: Arc<Database>,
}

impl MemoryCompiler {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Approximate token count using word/whitespace heuristic (~4 chars per token).
    fn estimate_tokens(text: &str) -> usize {
        (text.len() / 4).max(1)
    }

    /// Compile a structured, bounded MemoryContext from episodic, canonical, and candidate tiers.
    pub fn compile_context(&self, q: &MemoryQuery) -> Result<MemoryContext, rusqlite::Error> {
        let budget = q.token_budget;

        // Partition budget based on intent
        let (episodic_share, canonical_share, hypothesis_share): (f64, f64, f64) = match q.intent {
            MemoryIntent::Conversation => (0.50, 0.40, 0.0),
            MemoryIntent::Coding | MemoryIntent::ToolExecution => (0.25, 0.65, 0.0),
            MemoryIntent::Planning | MemoryIntent::Research => (0.35, 0.55, 0.0),
            MemoryIntent::Reflection => (0.45, 0.45, 0.0),
            MemoryIntent::MemoryAdjudication => (0.30, 0.40, 0.20),
        };

        let ep_budget = (budget as f64 * episodic_share) as usize;
        let can_budget = (budget as f64 * canonical_share) as usize;
        let hyp_budget = if q.include_candidates {
            (budget as f64 * hypothesis_share.max(0.15)) as usize
        } else {
            0
        };

        let mut episodic_items = Vec::new();
        let mut canonical_items = Vec::new();
        let mut hypothesis_items = Vec::new();
        let mut used_tokens = 0;
        let mut omitted_count = 0;

        // 1. Compile Episodic Context (if session_id provided)
        if let Some(session_id) = &q.session_id {
            let branch = q.branch_id.as_deref().unwrap_or("main");

            // A. Include highest level summary if available
            let summaries = self.db.get_session_summaries(session_id, branch, None)?;
            for s in summaries.iter().rev() {
                let text = format!(
                    "[Summary L{}: seq {}-{}] {}",
                    s.level, s.start_seq, s.end_seq, s.summary_text
                );
                let cost = Self::estimate_tokens(&text);
                if used_tokens + cost <= ep_budget {
                    used_tokens += cost;
                    episodic_items.push(EpisodicContextItem {
                        kind: "summary".to_string(),
                        level: Some(s.level),
                        sequence_range: Some(format!("{}-{}", s.start_seq, s.end_seq)),
                        text,
                    });
                    break; // Include top summary
                }
            }

            // B. Include recent raw safe events
            let recent_events = self.db.get_recent_session_events(session_id, branch, 15)?;
            for ev in &recent_events {
                let view = SecurityMembrane::to_safe_view(ev, self.db.hmac_key());
                if let Some(content) = view.safe_content {
                    let role = view.role.as_deref().unwrap_or("event");
                    let text = format!("[{}: seq {}] {}", role, view.sequence, content);
                    let cost = Self::estimate_tokens(&text);
                    if used_tokens + cost <= ep_budget {
                        used_tokens += cost;
                        episodic_items.push(EpisodicContextItem {
                            kind: "event".to_string(),
                            level: None,
                            sequence_range: Some(view.sequence.to_string()),
                            text,
                        });
                    } else {
                        omitted_count += 1;
                    }
                }
            }
        }

        // 2. Compile Canonical Knowledge (entities and claims)
        let recall_results = self.db.recall_entities(&q.query, None, Some(&q.space), 8)?;
        for item in recall_results {
            let ent = &item.entity;

            // Fetch active claims for this entity
            let claims = self
                .db
                .traverse_claims(&ent.canonical_name, Some(&q.space))?;
            for cl in claims {
                let tier = cl.verification_tier;
                if let Some(min_t) = q.minimum_verification {
                    if tier < min_t {
                        continue;
                    }
                }

                let val_str = cl
                    .literal_value
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "true".to_string());
                let text = format!(
                    "[{}] {} -> {} = {}",
                    tier.as_str(),
                    ent.canonical_name,
                    cl.predicate,
                    val_str
                );
                let cost = Self::estimate_tokens(&text);

                if used_tokens + cost <= ep_budget + can_budget {
                    used_tokens += cost;
                    canonical_items.push(CanonicalContextItem {
                        text,
                        claim_id: cl.id,
                        verification_tier: tier,
                        status: cl.status,
                        evidence_count: cl.evidence_count,
                    });
                } else {
                    omitted_count += 1;
                }
            }
        }

        // 3. Compile Unverified Hypotheses (STRICTLY QUARANTINED)
        if q.include_candidates && hyp_budget > 0 {
            let candidates = self
                .db
                .get_memory_candidates(Some(&q.space), Some("pending"), 5)?;
            for cand in candidates {
                let val_str = cand.object_value.to_string();
                let text = format!(
                    "Candidate {}: {} -> {} = {}",
                    cand.id, cand.subject, cand.predicate, val_str
                );
                let cost = Self::estimate_tokens(&text);

                if used_tokens + cost <= budget {
                    used_tokens += cost;
                    hypothesis_items.push(HypothesisContextItem {
                        candidate_id: cand.id,
                        assertion: text,
                        confidence: cand.confidence,
                        verification: cand.verification_tier.as_str().to_string(),
                        advisory: "UNVERIFIED HYPOTHESIS: Do not treat as established fact."
                            .to_string(),
                    });
                } else {
                    omitted_count += 1;
                }
            }
        }

        // 4. Render prompt section
        let mut prompt_lines = Vec::new();
        prompt_lines.push("<CORTEX_MEMORY>".to_string());

        prompt_lines.push("<EPISODIC_CONTEXT>".to_string());
        if episodic_items.is_empty() {
            prompt_lines.push("None.".to_string());
        } else {
            for item in &episodic_items {
                prompt_lines.push(item.text.clone());
            }
        }
        prompt_lines.push("</EPISODIC_CONTEXT>".to_string());

        prompt_lines.push("<VERIFIED_KNOWLEDGE>".to_string());
        if canonical_items.is_empty() {
            prompt_lines.push("None.".to_string());
        } else {
            for item in &canonical_items {
                prompt_lines.push(item.text.clone());
            }
        }
        prompt_lines.push("</VERIFIED_KNOWLEDGE>".to_string());

        prompt_lines.push("<UNVERIFIED_HYPOTHESES>".to_string());
        if hypothesis_items.is_empty() {
            prompt_lines.push("None.".to_string());
        } else {
            for item in &hypothesis_items {
                prompt_lines.push(format!(
                    "[{}] {} (Confidence: {:.2}) - {}",
                    item.candidate_id, item.assertion, item.confidence, item.advisory
                ));
            }
        }
        prompt_lines.push("</UNVERIFIED_HYPOTHESES>".to_string());

        prompt_lines.push("</CORTEX_MEMORY>".to_string());

        let rendered_prompt = prompt_lines.join("\n");
        let context_revision = format!("ctx-{}", Uuid::new_v4().simple());

        Ok(MemoryContext {
            episodic: episodic_items,
            canonical: canonical_items,
            hypotheses: hypothesis_items,
            rendered_prompt,
            omitted_count,
            token_count: used_tokens,
            context_revision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        CandidateWriteInput, ClaimWriteInput, CreateSessionInput, EntityWriteInput,
        SessionEventInput,
    };
    use serde_json::json;

    #[test]
    fn test_memory_compiler_knapsack_and_quarantine() {
        let db = Arc::new(Database::open_in_memory().unwrap());

        // 1. Seed canonical memory
        let ent = db
            .upsert_entity(
                &EntityWriteInput {
                    id: None,
                    space: "atlas-memory".to_string(),
                    entity_type: "architecture".to_string(),
                    canonical_name: "Cortex_Store".to_string(),
                    content: "Cortex engine uses SQLite WAL mode".to_string(),
                    aliases: vec![],
                    metadata: json!({}),
                    confidence: 1.0,
                    valid_from: None,
                    valid_to: None,
                    external_id: None,
                },
                None,
            )
            .unwrap();

        db.upsert_claim(&ClaimWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            subject_entity_id: ent.target_id.clone(),
            predicate: "concurrency_mode".to_string(),
            object_entity_id: None,
            literal_value: Some(json!("single_writer_multi_reader")),
            confidence: 1.0,
            metadata: json!({}),
        })
        .unwrap();

        // 2. Seed session and episodic turns
        let sess = db
            .create_session(&CreateSessionInput {
                id: Some("sess-compiler".to_string()),
                space: "atlas-memory".to_string(),
                agent_id: Some("atlas".to_string()),
                world_id: None,
                parent_session_id: None,
                fork_event_id: None,
                retention_class: "standard".to_string(),
                metadata: json!({}),
            })
            .unwrap();

        db.batch_append_events(
            &sess.id,
            "main",
            &[SessionEventInput {
                id: Some("ev-c1".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: None,
                event_type: "user_message".to_string(),
                role: Some("user".to_string()),
                content: Some("How does the memory compiler work?".to_string()),
                payload: json!({}),
                sensitivity: None,
            }],
        )
        .unwrap();

        // 3. Seed pending candidate
        db.insert_memory_candidate(&CandidateWriteInput {
            id: Some("cand-unverified".to_string()),
            space: "atlas-memory".to_string(),
            session_id: sess.id.clone(),
            branch_id: "main".to_string(),
            memory_type: "speculation".to_string(),
            subject: "Compiler".to_string(),
            predicate: "performance_improvement".to_string(),
            object_value: json!("50x"),
            scope: "global".to_string(),
            confidence: 0.65,
            verification_tier: Some(VerificationTier::T0Direct),
            extractor_version: "0.2.0".to_string(),
            evidence_ids: vec![],
        })
        .unwrap();

        let compiler = MemoryCompiler::new(Arc::clone(&db));

        // Test A: Normal query (include_candidates = false)
        let q_normal = MemoryQuery {
            query: "Cortex_Store".to_string(),
            session_id: Some(sess.id.clone()),
            branch_id: Some("main".to_string()),
            space: "atlas-memory".to_string(),
            intent: MemoryIntent::Coding,
            token_budget: 1024,
            include_candidates: false,
            minimum_verification: None,
        };

        let ctx_normal = compiler.compile_context(&q_normal).unwrap();
        assert!(!ctx_normal.canonical.is_empty());
        assert!(!ctx_normal.episodic.is_empty());
        assert!(ctx_normal.hypotheses.is_empty());
        assert!(ctx_normal.rendered_prompt.contains("<VERIFIED_KNOWLEDGE>"));
        assert!(ctx_normal
            .rendered_prompt
            .contains("<UNVERIFIED_HYPOTHESES>\nNone."));

        // Test B: Adjudication query with candidates enabled
        let q_adjudicate = MemoryQuery {
            query: "Cortex_Store".to_string(),
            session_id: Some(sess.id.clone()),
            branch_id: Some("main".to_string()),
            space: "atlas-memory".to_string(),
            intent: MemoryIntent::MemoryAdjudication,
            token_budget: 1024,
            include_candidates: true,
            minimum_verification: None,
        };

        let ctx_adj = compiler.compile_context(&q_adjudicate).unwrap();
        assert!(!ctx_adj.hypotheses.is_empty());
        assert_eq!(ctx_adj.hypotheses[0].candidate_id, "cand-unverified");
        assert!(ctx_adj
            .rendered_prompt
            .contains("UNVERIFIED HYPOTHESIS: Do not treat as established fact."));
    }
}
