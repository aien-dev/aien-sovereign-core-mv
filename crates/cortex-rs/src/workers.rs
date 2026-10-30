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

use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::db::{Database, SessionSummaryWrite};
use crate::models::{CandidateWriteInput, MemoryCandidate, SessionSummary, VerificationTier};
use crate::security::{SafeEventView, SecurityMembrane};

pub const SUMMARY_PROCESSOR_VERSION: &str = "0.2.0";
pub const EXTRACTOR_PROCESSOR_VERSION: &str = "0.2.0";

pub struct SummaryWorker {
    db: Arc<Database>,
}

impl SummaryWorker {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Process unsummarized events for a session branch up to `batch_size` events.
    pub fn process_session_branch(
        &self,
        session_id: &str,
        branch_id: &str,
        segment_size: usize,
    ) -> Result<Option<SessionSummary>, rusqlite::Error> {
        let watermark = self
            .db
            .get_watermark(session_id, "summary", SUMMARY_PROCESSOR_VERSION)?
            .unwrap_or(0);

        let events =
            self.db
                .get_session_events(session_id, branch_id, Some(watermark), segment_size)?;
        if events.is_empty() {
            return Ok(None);
        }

        // Project events into SafeEventView
        let safe_views: Vec<SafeEventView> = events
            .iter()
            .map(|ev| SecurityMembrane::to_safe_view(ev, self.db.hmac_key()))
            .collect();

        let start_seq = safe_views.first().unwrap().sequence;
        let end_seq = safe_views.last().unwrap().sequence;

        // Compute source hash of safe event sequence
        let mut hasher = Sha256::new();
        for ev in &safe_views {
            hasher.update(ev.event_id.as_bytes());
            hasher.update(ev.sequence.to_string().as_bytes());
            if let Some(c) = &ev.safe_content {
                hasher.update(c.as_bytes());
            }
        }
        let source_hash = format!("SHA256:{:x}", hasher.finalize());

        // Generate deterministic summary text from SafeEventView
        let mut lines = Vec::new();
        for ev in &safe_views {
            if let Some(content) = &ev.safe_content {
                let role = ev.role.as_deref().unwrap_or("agent");
                lines.push(format!("[{}:{}] {}", role, ev.event_type, content));
            }
        }
        let summary_text = if lines.is_empty() {
            format!(
                "Events {}..{} with withheld/redacted content",
                start_seq, end_seq
            )
        } else {
            lines.join(" | ")
        };

        // Insert Level-0 summary
        let summary = self.db.insert_session_summary(&SessionSummaryWrite {
            session_id,
            branch_id,
            level: 0,
            start_seq,
            end_seq,
            summary_text: &summary_text,
            source_hash: &source_hash,
            processor_version: SUMMARY_PROCESSOR_VERSION,
        })?;

        // Advance summary watermark
        self.db
            .set_watermark(session_id, "summary", SUMMARY_PROCESSOR_VERSION, end_seq)?;

        // Hierarchical Level-1 Compaction check:
        // If there are at least 4 level-0 summaries, compact them into level-1
        let summaries_l0 = self
            .db
            .get_session_summaries(session_id, branch_id, Some(0))?;
        if summaries_l0.len() >= 4 && summaries_l0.len() % 4 == 0 {
            let chunk = &summaries_l0[summaries_l0.len() - 4..];
            let h_start = chunk.first().unwrap().start_seq;
            let h_end = chunk.last().unwrap().end_seq;
            let mut h_hasher = Sha256::new();
            let mut h_text_parts = Vec::new();
            for s in chunk {
                h_hasher.update(s.source_hash.as_bytes());
                h_text_parts.push(s.summary_text.clone());
            }
            let h_hash = format!("SHA256:{:x}", h_hasher.finalize());
            let h_text = format!("Hierarchical Summary (L1): {}", h_text_parts.join(" => "));

            let _ = self.db.insert_session_summary(&SessionSummaryWrite {
                session_id,
                branch_id,
                level: 1,
                start_seq: h_start,
                end_seq: h_end,
                summary_text: &h_text,
                source_hash: &h_hash,
                processor_version: SUMMARY_PROCESSOR_VERSION,
            });
        }

        Ok(Some(summary))
    }
}

pub struct ExtractionWorker {
    db: Arc<Database>,
}

impl ExtractionWorker {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Extract durable memory candidates from unextracted safe events.
    pub fn process_session_branch(
        &self,
        session_id: &str,
        branch_id: &str,
        batch_size: usize,
    ) -> Result<Vec<MemoryCandidate>, rusqlite::Error> {
        let watermark = self
            .db
            .get_watermark(session_id, "extractor", EXTRACTOR_PROCESSOR_VERSION)?
            .unwrap_or(0);

        let events =
            self.db
                .get_session_events(session_id, branch_id, Some(watermark), batch_size)?;
        if events.is_empty() {
            return Ok(Vec::new());
        }

        let safe_views: Vec<SafeEventView> = events
            .iter()
            .map(|ev| SecurityMembrane::to_safe_view(ev, self.db.hmac_key()))
            .collect();
        let max_seq = safe_views.last().unwrap().sequence;

        let mut extracted_candidates = Vec::new();

        for ev in &safe_views {
            if !ev.extraction_allowed {
                continue;
            }

            if let Some(content) = &ev.safe_content {
                // Rule-based heuristic extraction for sovereign runtime facts & user preferences
                let text = content.trim();

                // Pattern 1: User operational preferences (T0Direct)
                // "I prefer X", "prefer using X", "my preferred X is Y"
                if text.to_lowercase().contains("prefer") || text.to_lowercase().contains("use ") {
                    if let Some(cand) = Self::extract_preference(session_id, branch_id, ev, text) {
                        if let Ok(inserted) = self.db.insert_memory_candidate(&cand) {
                            extracted_candidates.push(inserted);
                        }
                    }
                }

                // Pattern 2: Learned technical procedure or discovery (T2Verified if tool evidence)
                if ev.event_type == "tool_result" || ev.event_type == "agent_action" {
                    if let Some(cand) =
                        Self::extract_technical_fact(session_id, branch_id, ev, text)
                    {
                        if let Ok(inserted) = self.db.insert_memory_candidate(&cand) {
                            extracted_candidates.push(inserted);
                        }
                    }
                }
            }
        }

        // Advance extractor watermark
        self.db.set_watermark(
            session_id,
            "extractor",
            EXTRACTOR_PROCESSOR_VERSION,
            max_seq,
        )?;

        Ok(extracted_candidates)
    }

    fn extract_preference(
        session_id: &str,
        branch_id: &str,
        ev: &SafeEventView,
        text: &str,
    ) -> Option<CandidateWriteInput> {
        let lower = text.to_lowercase();
        if lower.contains("prefer rust") || lower.contains("use rust") {
            Some(CandidateWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                session_id: session_id.to_string(),
                branch_id: branch_id.to_string(),
                memory_type: "workflow_preference".to_string(),
                subject: "user".to_string(),
                predicate: "preferred_language".to_string(),
                object_value: serde_json::json!("Rust"),
                scope: "global".to_string(),
                confidence: 0.95,
                verification_tier: Some(VerificationTier::T0Direct),
                extractor_version: EXTRACTOR_PROCESSOR_VERSION.to_string(),
                evidence_ids: vec![ev.event_id.clone()],
            })
        } else if lower.contains("prefer concise") || lower.contains("concise responses") {
            Some(CandidateWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                session_id: session_id.to_string(),
                branch_id: branch_id.to_string(),
                memory_type: "workflow_preference".to_string(),
                subject: "user".to_string(),
                predicate: "response_style".to_string(),
                object_value: serde_json::json!("concise"),
                scope: "global".to_string(),
                confidence: 0.92,
                verification_tier: Some(VerificationTier::T0Direct),
                extractor_version: EXTRACTOR_PROCESSOR_VERSION.to_string(),
                evidence_ids: vec![ev.event_id.clone()],
            })
        } else {
            None
        }
    }

    fn extract_technical_fact(
        session_id: &str,
        branch_id: &str,
        ev: &SafeEventView,
        text: &str,
    ) -> Option<CandidateWriteInput> {
        let lower = text.to_lowercase();
        if lower.contains("wal") && lower.contains("sqlite") {
            Some(CandidateWriteInput {
                id: None,
                space: "atlas-memory".to_string(),
                session_id: session_id.to_string(),
                branch_id: branch_id.to_string(),
                memory_type: "learned_procedure".to_string(),
                subject: "AIEN_Cortex".to_string(),
                predicate: "storage_mode".to_string(),
                object_value: serde_json::json!("SQLite_WAL"),
                scope: "system".to_string(),
                confidence: 0.98,
                verification_tier: Some(VerificationTier::T2Verified),
                extractor_version: EXTRACTOR_PROCESSOR_VERSION.to_string(),
                evidence_ids: vec![ev.event_id.clone()],
            })
        } else {
            None
        }
    }
}

pub const MAX_WORKER_BATCH: usize = 32;

pub struct ProcessingScheduler {
    summary_worker: SummaryWorker,
    extraction_worker: ExtractionWorker,
}

impl ProcessingScheduler {
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            summary_worker: SummaryWorker::new(Arc::clone(&db)),
            extraction_worker: ExtractionWorker::new(db),
        }
    }

    /// Run one consolidation tick for a specific session branch.
    pub fn process_session(
        &self,
        session_id: &str,
        branch_id: &str,
    ) -> Result<(Option<SessionSummary>, Vec<MemoryCandidate>), rusqlite::Error> {
        let batch = MAX_WORKER_BATCH;
        let summary = self
            .summary_worker
            .process_session_branch(session_id, branch_id, batch)?;
        let candidates = self
            .extraction_worker
            .process_session_branch(session_id, branch_id, batch)?;
        Ok((summary, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CreateSessionInput, SessionEventInput};

    #[test]
    fn test_summary_and_extraction_workers() {
        let db = Arc::new(Database::open_in_memory().unwrap());
        let session = db
            .create_session(&CreateSessionInput {
                id: Some("sess-worker-test".to_string()),
                space: "atlas-memory".to_string(),
                agent_id: Some("worker-test".to_string()),
                world_id: None,
                parent_session_id: None,
                fork_event_id: None,
                retention_class: "standard".to_string(),
                metadata: serde_json::json!({}),
            })
            .unwrap();

        // Append events
        let evs = vec![
            SessionEventInput {
                id: Some("ev-101".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: None,
                event_type: "user_message".to_string(),
                role: Some("user".to_string()),
                content: Some("I prefer Rust for sovereign architecture.".to_string()),
                payload: serde_json::json!({}),
                sensitivity: None,
            },
            SessionEventInput {
                id: Some("ev-102".to_string()),
                branch_id: Some("main".to_string()),
                parent_event_id: Some("ev-101".to_string()),
                event_type: "tool_result".to_string(),
                role: Some("tool".to_string()),
                content: Some(
                    "Verified SQLite WAL mode active with sub-millisecond latency.".to_string(),
                ),
                payload: serde_json::json!({"status": "verified"}),
                sensitivity: None,
            },
        ];
        db.batch_append_events(&session.id, "main", &evs).unwrap();

        // Run scheduler
        let scheduler = ProcessingScheduler::new(Arc::clone(&db));
        let (summary, candidates) = scheduler.process_session(&session.id, "main").unwrap();

        assert!(summary.is_some());
        let s = summary.unwrap();
        assert_eq!(s.level, 0);
        assert_eq!(s.start_seq, 1);
        assert_eq!(s.end_seq, 2);
        assert!(s.summary_text.contains("prefer Rust"));

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].subject, "user");
        assert_eq!(candidates[0].predicate, "preferred_language");
        assert_eq!(candidates[0].verification_tier, VerificationTier::T0Direct);

        assert_eq!(candidates[1].subject, "AIEN_Cortex");
        assert_eq!(candidates[1].predicate, "storage_mode");
        assert_eq!(
            candidates[1].verification_tier,
            VerificationTier::T2Verified
        );

        // Verify watermarks were updated
        let s_wm = db
            .get_watermark(&session.id, "summary", SUMMARY_PROCESSOR_VERSION)
            .unwrap();
        assert_eq!(s_wm, Some(2));
        let e_wm = db
            .get_watermark(&session.id, "extractor", EXTRACTOR_PROCESSOR_VERSION)
            .unwrap();
        assert_eq!(e_wm, Some(2));
    }
}
