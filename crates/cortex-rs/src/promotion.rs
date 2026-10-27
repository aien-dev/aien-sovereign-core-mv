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

use crate::db::Database;
use crate::models::{MemoryCandidate, PromoteCandidateInput, PromotionReceipt, VerificationTier};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionDecision {
    Promote,
    Hold { reason: String },
    Reject { reason: String },
}

pub fn enforce_promotion_policy(candidate: &MemoryCandidate) -> Result<(), String> {
    match PromotionEngine::evaluate_candidate(candidate) {
        PromotionDecision::Promote => Ok(()),
        PromotionDecision::Hold { reason } | PromotionDecision::Reject { reason } => Err(reason),
    }
}

pub struct PromotionEngine {
    db: Arc<Database>,
}

impl PromotionEngine {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    pub fn evaluate_candidate(candidate: &MemoryCandidate) -> PromotionDecision {
        if candidate.state == "promoted" {
            return PromotionDecision::Reject {
                reason: "Candidate already promoted".to_string(),
            };
        }
        if candidate.state == "rejected" || candidate.state == "retracted" {
            return PromotionDecision::Reject {
                reason: format!("Candidate is in terminal state: {}", candidate.state),
            };
        }

        match candidate.verification_tier {
            VerificationTier::T0Direct => {
                // T0: Direct assertion (e.g. user preference) requires at least 1 valid evidence reference
                if candidate.evidence_ids.is_empty() {
                    PromotionDecision::Hold {
                        reason: "T0 Direct requires at least one source evidence reference"
                            .to_string(),
                    }
                } else if candidate.confidence < 0.80 {
                    PromotionDecision::Hold {
                        reason: format!(
                            "Confidence {} below T0 threshold 0.80",
                            candidate.confidence
                        ),
                    }
                } else {
                    PromotionDecision::Promote
                }
            }
            VerificationTier::T1Corroborated => {
                // T1: Corroborated knowledge requires >= 2 independent evidence references
                if candidate.evidence_ids.len() < 2 {
                    PromotionDecision::Hold {
                        reason: format!(
                            "T1 Corroborated requires at least 2 independent observations, found {}",
                            candidate.evidence_ids.len()
                        ),
                    }
                } else if candidate.confidence < 0.85 {
                    PromotionDecision::Hold {
                        reason: format!(
                            "Confidence {} below T1 threshold 0.85",
                            candidate.confidence
                        ),
                    }
                } else {
                    PromotionDecision::Promote
                }
            }
            VerificationTier::T2Verified => {
                // T2: Technical/empirical claim requires tool evidence or verifier receipt
                if candidate.evidence_ids.is_empty() {
                    PromotionDecision::Hold {
                        reason: "T2 Verified requires tool evidence or benchmark proof".to_string(),
                    }
                } else if candidate.confidence < 0.90 {
                    PromotionDecision::Hold {
                        reason: format!(
                            "Confidence {} below T2 threshold 0.90",
                            candidate.confidence
                        ),
                    }
                } else {
                    PromotionDecision::Promote
                }
            }
            VerificationTier::T3Controlled => {
                // T3: Controlled/authority claims always require explicit administrative sign-off
                PromotionDecision::Hold {
                    reason: "T3 Controlled knowledge requires explicit manual authorization"
                        .to_string(),
                }
            }
        }
    }

    /// Automatically evaluate and promote eligible candidates for a space.
    pub fn auto_promote_eligible(
        &self,
        space: Option<&str>,
    ) -> Result<Vec<PromotionReceipt>, rusqlite::Error> {
        let candidates = self.db.get_memory_candidates(space, Some("pending"), 50)?;
        let mut receipts = Vec::new();

        for cand in candidates {
            match Self::evaluate_candidate(&cand) {
                PromotionDecision::Promote => {
                    let input = PromoteCandidateInput {
                        candidate_id: cand.id.clone(),
                        policy_id: Some("auto-promotion-policy-v1".to_string()),
                        verifier_receipt: Some(format!(
                            "Auto-verified tier {:?}",
                            cand.verification_tier
                        )),
                    };
                    if let Ok(receipt) = self.db.promote_candidate(&input) {
                        receipts.push(receipt);
                    }
                }
                PromotionDecision::Hold { reason } => {
                    // Update candidate state to 'held' with reason
                    let _ = self.db.update_candidate_state(&cand.id, "held");
                    tracing::debug!("Holding candidate {}: {}", cand.id, reason);
                }
                PromotionDecision::Reject { reason } => {
                    let _ = self.db.update_candidate_state(&cand.id, "rejected");
                    tracing::debug!("Rejecting candidate {}: {}", cand.id, reason);
                }
            }
        }

        Ok(receipts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CandidateWriteInput, CreateSessionInput};
    use serde_json::json;

    #[test]
    fn test_promotion_policy_evaluation_tiers() {
        // T0 without evidence -> Hold
        let t0_no_ev = MemoryCandidate {
            id: "c1".to_string(),
            space_slug: "atlas-memory".to_string(),
            session_id: "s1".to_string(),
            branch_id: "main".to_string(),
            memory_type: "workflow_preference".to_string(),
            subject: "user".to_string(),
            predicate: "preferred_language".to_string(),
            object_value: json!("Rust"),
            scope: "global".to_string(),
            confidence: 0.95,
            verification_tier: VerificationTier::T0Direct,
            state: "pending".to_string(),
            extractor_version: "0.2.0".to_string(),
            created_at: "2026-09-21".to_string(),
            evidence_ids: vec![],
        };
        assert!(matches!(
            PromotionEngine::evaluate_candidate(&t0_no_ev),
            PromotionDecision::Hold { .. }
        ));

        // T0 with evidence -> Promote
        let mut t0_with_ev = t0_no_ev.clone();
        t0_with_ev.evidence_ids.push("ev-1".to_string());
        assert_eq!(
            PromotionEngine::evaluate_candidate(&t0_with_ev),
            PromotionDecision::Promote
        );

        // T1 with 1 evidence -> Hold
        let mut t1 = t0_with_ev.clone();
        t1.verification_tier = VerificationTier::T1Corroborated;
        assert!(matches!(
            PromotionEngine::evaluate_candidate(&t1),
            PromotionDecision::Hold { .. }
        ));

        // T1 with 2 evidence -> Promote
        t1.evidence_ids.push("ev-2".to_string());
        assert_eq!(
            PromotionEngine::evaluate_candidate(&t1),
            PromotionDecision::Promote
        );

        // T3 -> Always Hold (manual authorization)
        let mut t3 = t1.clone();
        t3.verification_tier = VerificationTier::T3Controlled;
        assert!(matches!(
            PromotionEngine::evaluate_candidate(&t3),
            PromotionDecision::Hold { .. }
        ));
    }

    #[test]
    fn test_atomic_promotion_and_temporal_supersession() {
        let db = Arc::new(Database::open_in_memory().unwrap());
        db.create_session(&CreateSessionInput {
            id: Some("sess-promo".to_string()),
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
            "sess-promo",
            "main",
            &[
                crate::models::SessionEventInput {
                    id: Some("ev-001".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: None,
                    event_type: "user_message".to_string(),
                    role: Some("user".to_string()),
                    content: Some("I prefer Python for scripts.".to_string()),
                    payload: json!({}),
                    sensitivity: None,
                },
                crate::models::SessionEventInput {
                    id: Some("ev-002".to_string()),
                    branch_id: Some("main".to_string()),
                    parent_event_id: None,
                    event_type: "user_message".to_string(),
                    role: Some("user".to_string()),
                    content: Some("I prefer Rust for the runtime.".to_string()),
                    payload: json!({}),
                    sensitivity: None,
                },
            ],
        )
        .unwrap();

        let bare = db
            .insert_memory_candidate(&CandidateWriteInput {
                id: Some("cand-bare".to_string()),
                space: "atlas-memory".to_string(),
                session_id: "sess-promo".to_string(),
                branch_id: "main".to_string(),
                memory_type: "workflow_preference".to_string(),
                subject: "developer".to_string(),
                predicate: "editor".to_string(),
                object_value: json!("vim"),
                scope: "global".to_string(),
                confidence: 0.99,
                verification_tier: Some(VerificationTier::T0Direct),
                extractor_version: "0.2.0".to_string(),
                evidence_ids: vec![],
            })
            .unwrap();
        assert!(db
            .promote_candidate(&PromoteCandidateInput {
                candidate_id: bare.id,
                policy_id: Some("p1".to_string()),
                verifier_receipt: None,
            })
            .is_err());

        // 1. Insert candidate C1: User prefers Python
        let cand1 = db
            .insert_memory_candidate(&CandidateWriteInput {
                id: Some("cand-01".to_string()),
                space: "atlas-memory".to_string(),
                session_id: "sess-promo".to_string(),
                branch_id: "main".to_string(),
                memory_type: "workflow_preference".to_string(),
                subject: "developer".to_string(),
                predicate: "preferred_language".to_string(),
                object_value: json!("Python"),
                scope: "global".to_string(),
                confidence: 0.95,
                verification_tier: Some(VerificationTier::T0Direct),
                extractor_version: "0.2.0".to_string(),
                evidence_ids: vec!["ev-001".to_string()],
            })
            .unwrap();

        // Promote C1
        let receipt1 = db
            .promote_candidate(&PromoteCandidateInput {
                candidate_id: cand1.id.clone(),
                policy_id: Some("p1".to_string()),
                verifier_receipt: None,
            })
            .unwrap();
        assert_eq!(receipt1.candidate_id, "cand-01");

        // Verify claim is active
        let claims_after_1 = db
            .traverse_claims("developer", Some("atlas-memory"))
            .unwrap();
        assert_eq!(claims_after_1.len(), 1);
        assert_eq!(claims_after_1[0].literal_value, Some(json!("Python")));
        assert_eq!(
            claims_after_1[0].verification_tier,
            VerificationTier::T0Direct
        );
        assert_eq!(claims_after_1[0].evidence_count, 1);
        assert_eq!(claims_after_1[0].status, crate::models::ClaimStatus::Active);

        // 2. Insert candidate C2: User now prefers Rust (contradiction & temporal update)
        let cand2 = db
            .insert_memory_candidate(&CandidateWriteInput {
                id: Some("cand-02".to_string()),
                space: "atlas-memory".to_string(),
                session_id: "sess-promo".to_string(),
                branch_id: "main".to_string(),
                memory_type: "workflow_preference".to_string(),
                subject: "developer".to_string(),
                predicate: "preferred_language".to_string(),
                object_value: json!("Rust"),
                scope: "global".to_string(),
                confidence: 0.99,
                verification_tier: Some(VerificationTier::T0Direct),
                extractor_version: "0.2.0".to_string(),
                evidence_ids: vec!["ev-002".to_string()],
            })
            .unwrap();

        // Promote C2
        let receipt2 = db
            .promote_candidate(&PromoteCandidateInput {
                candidate_id: cand2.id.clone(),
                policy_id: Some("p1".to_string()),
                verifier_receipt: None,
            })
            .unwrap();
        assert_eq!(receipt2.candidate_id, "cand-02");

        // Verify temporal supersession: only 1 active claim remains and it's "Rust"!
        let claims_after_2 = db
            .traverse_claims("developer", Some("atlas-memory"))
            .unwrap();
        assert_eq!(claims_after_2.len(), 1);
        assert_eq!(claims_after_2[0].literal_value, Some(json!("Rust")));
        assert_eq!(claims_after_2[0].evidence_count, 1);
    }
}
