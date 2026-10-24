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

use crate::config::ExperimentConfig;
use chrono::{DateTime, Utc};
use cortex_rs::{ClaimStatus, VerificationTier};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct EdgeView {
    pub claim_id: String,
    pub subject: String,
    pub object: String,
    pub confidence: f64,
    pub tier: VerificationTier,
    pub evidence_count: usize,
    pub created_at: String,
    pub status: ClaimStatus,
    pub retracted: bool,
    pub subject_space: String,
    pub object_space: String,
    pub claim_space: String,
}

pub fn is_admitted(edge: &EdgeView, query_space: &str) -> bool {
    edge.status == ClaimStatus::Active
        && !edge.retracted
        && !edge.subject.is_empty()
        && !edge.object.is_empty()
        && edge.subject != edge.object
        && edge.subject_space == query_space
        && edge.object_space == query_space
        && edge.claim_space == query_space
}

pub fn tier_weight(config: &ExperimentConfig, tier: VerificationTier) -> f64 {
    match tier {
        VerificationTier::T0Direct => config.tier_t0,
        VerificationTier::T1Corroborated => config.tier_t1,
        VerificationTier::T2Verified => config.tier_t2,
        VerificationTier::T3Controlled => config.tier_t3,
    }
}

pub fn evidence_factor(evidence_count: usize) -> f64 {
    let n = evidence_count as f64;
    (1.0 + n) / (2.0 + n)
}

pub fn freshness(config: &ExperimentConfig, created_at: &str, snapshot_time: DateTime<Utc>) -> f64 {
    let Ok(created) = DateTime::parse_from_rfc3339(created_at) else {
        return config.freshness_floor;
    };
    let age_days = (snapshot_time - created.with_timezone(&Utc)).num_seconds() as f64 / 86_400.0;
    if age_days <= 0.0 {
        return 1.0;
    }
    let raw = 0.5_f64.powf(age_days / config.freshness_half_life_days);
    raw.clamp(config.freshness_floor, 1.0)
}

pub fn edge_weight(
    config: &ExperimentConfig,
    edge: &EdgeView,
    snapshot_time: DateTime<Utc>,
) -> f64 {
    edge.confidence
        * tier_weight(config, edge.tier)
        * evidence_factor(edge.evidence_count)
        * freshness(config, &edge.created_at, snapshot_time)
}

#[derive(Debug, Clone)]
pub struct WalkEdge {
    pub claim_id: String,
    pub neighbor: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CandidateSuggestion {
    pub query_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct WalkOutput {
    pub graph_score: BTreeMap<String, f64>,
    pub paths_scored: usize,
    pub suggestions: Vec<(String, String, f64)>,
}

pub fn walk_paths(
    config: &ExperimentConfig,
    anchors: &[(String, f64)],
    adj: &BTreeMap<String, Vec<WalkEdge>>,
    direct: &BTreeSet<(String, String)>,
) -> WalkOutput {
    let anchor_ids: BTreeSet<&str> = anchors.iter().map(|(id, _)| id.as_str()).collect();
    let mut graph_score: BTreeMap<String, f64> = BTreeMap::new();
    let mut paths_scored = 0usize;
    let mut indirect: BTreeMap<(String, String), f64> = BTreeMap::new();

    for (anchor_id, anchor_base) in anchors {
        let Some(first_hops) = adj.get(anchor_id) else {
            continue;
        };
        for edge in first_hops.iter().take(config.max_neighbors) {
            if paths_scored >= config.max_paths {
                break;
            }
            if edge.neighbor == *anchor_id {
                continue;
            }
            let contribution = anchor_base * edge.weight;
            award(config, &mut graph_score, &edge.neighbor, contribution);
            paths_scored += 1;
            if config.max_hops < 2 {
                continue;
            }
            let Some(second_hops) = adj.get(&edge.neighbor) else {
                continue;
            };
            for edge2 in second_hops.iter().take(config.max_neighbors) {
                if paths_scored >= config.max_paths {
                    break;
                }
                if edge2.neighbor == *anchor_id || edge2.neighbor == edge.neighbor {
                    continue;
                }
                let path_score = edge.weight * edge2.weight * config.hop_decay;
                let contribution = anchor_base * path_score;
                award(config, &mut graph_score, &edge2.neighbor, contribution);
                if anchor_ids.contains(edge2.neighbor.as_str()) {
                    let pair = ordered_pair(anchor_id, &edge2.neighbor);
                    let slot = indirect.entry(pair).or_insert(0.0);
                    if contribution > *slot {
                        *slot = contribution;
                    }
                }
                paths_scored += 1;
            }
        }
    }

    let suggestions = indirect
        .into_iter()
        .filter(|(pair, _)| !direct.contains(pair))
        .map(|(pair, score)| (pair.0, pair.1, score))
        .collect();

    WalkOutput {
        graph_score,
        paths_scored,
        suggestions,
    }
}

fn award(
    config: &ExperimentConfig,
    graph_score: &mut BTreeMap<String, f64>,
    entity: &str,
    contribution: f64,
) {
    if let Some(existing) = graph_score.get_mut(entity) {
        if contribution > *existing {
            *existing = contribution;
        }
        return;
    }
    if graph_score.len() < config.max_graph_entities {
        graph_score.insert(entity.to_string(), contribution);
    }
}

pub fn ordered_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedEntity {
    pub id: String,
    pub baseline: f64,
    pub graph_score: f64,
    pub final_score: f64,
}

pub fn fuse_and_rank(
    config: &ExperimentConfig,
    baseline: &[(String, f64)],
    graph_score: &BTreeMap<String, f64>,
) -> Vec<RankedEntity> {
    let mut ranked: Vec<RankedEntity> = baseline
        .iter()
        .map(|(id, baseline_score)| {
            let graph = graph_score.get(id).copied().unwrap_or(0.0);
            RankedEntity {
                id: id.clone(),
                baseline: *baseline_score,
                graph_score: graph,
                final_score: baseline_score + config.graph_mix * graph,
            }
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .final_score
            .partial_cmp(&left.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    ranked.truncate(config.top_k);
    ranked
}

pub fn select_anchors(config: &ExperimentConfig, scored: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut ordered = scored.to_vec();
    ordered.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ordered.truncate(config.anchor_count);
    ordered
}
