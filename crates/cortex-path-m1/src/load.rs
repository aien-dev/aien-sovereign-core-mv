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
use crate::score::{edge_weight, is_admitted, ordered_pair, EdgeView, WalkEdge};
use chrono::{DateTime, Utc};
use cortex_rs::Database;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub struct LoadedGraph {
    pub edges: usize,
    pub adjacency: BTreeMap<String, Vec<WalkEdge>>,
    pub direct: BTreeSet<(String, String)>,
}

pub fn load_admitted_graph(
    db: &Database,
    space: &str,
    entity_ids: &[String],
    config: &ExperimentConfig,
    snapshot_time: DateTime<Utc>,
) -> Result<LoadedGraph, String> {
    let mut seen_claims = BTreeSet::new();
    let mut admitted_claims = 0usize;
    let mut raw: Vec<(String, String, WalkEdge)> = Vec::new();
    let mut direct = BTreeSet::new();
    let mut space_of: HashMap<String, String> = HashMap::new();

    for entity_id in entity_ids {
        let claims = db
            .traverse_claims(entity_id, Some(space))
            .map_err(|err| err.to_string())?;
        for claim in claims {
            if !seen_claims.insert(claim.id.clone()) {
                continue;
            }
            let Some(object_id) = claim.object_entity_id.clone() else {
                continue;
            };
            let subject_space = entity_space(db, &claim.subject_entity_id, space, &mut space_of)?;
            let object_space = entity_space(db, &object_id, space, &mut space_of)?;
            let view = EdgeView {
                claim_id: claim.id.clone(),
                subject: claim.subject_entity_id.clone(),
                object: object_id.clone(),
                confidence: claim.confidence,
                tier: claim.verification_tier,
                evidence_count: claim.evidence_count,
                created_at: claim.created_at.clone(),
                status: claim.status,
                retracted: claim.retracted,
                subject_space,
                object_space,
                claim_space: claim.space_slug.clone(),
            };
            if !is_admitted(&view, space) {
                continue;
            }
            let weight = edge_weight(config, &view, snapshot_time);
            admitted_claims += 1;
            direct.insert(ordered_pair(&view.subject, &view.object));
            raw.push((
                view.subject.clone(),
                view.object.clone(),
                WalkEdge {
                    claim_id: view.claim_id.clone(),
                    neighbor: view.object.clone(),
                    weight,
                },
            ));
            raw.push((
                view.object.clone(),
                view.subject.clone(),
                WalkEdge {
                    claim_id: view.claim_id,
                    neighbor: view.subject,
                    weight,
                },
            ));
        }
    }

    let mut adjacency: BTreeMap<String, Vec<WalkEdge>> = BTreeMap::new();
    for (from, _to, edge) in raw {
        adjacency.entry(from).or_default().push(edge);
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_by(|left, right| {
            right
                .weight
                .partial_cmp(&left.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.claim_id.cmp(&right.claim_id))
                .then_with(|| left.neighbor.cmp(&right.neighbor))
        });
    }

    let edges = admitted_claims;
    Ok(LoadedGraph {
        edges,
        adjacency,
        direct,
    })
}

fn entity_space(
    db: &Database,
    entity_id: &str,
    query_space: &str,
    cache: &mut HashMap<String, String>,
) -> Result<String, String> {
    if let Some(found) = cache.get(entity_id) {
        return Ok(found.clone());
    }
    let found = db
        .get_entity(entity_id, Some(query_space))
        .map_err(|err| err.to_string())?
        .map(|entity| entity.space_slug)
        .unwrap_or_default();
    cache.insert(entity_id.to_string(), found.clone());
    Ok(found)
}
