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

use serde::Serialize;

/// Frozen Milestone 1 constants. The JSON encoding of this value is `config_sha256`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct ExperimentConfig {
    pub schema: u32,
    pub anchor_count: usize,
    pub max_hops: u8,
    pub max_neighbors: usize,
    pub max_paths: usize,
    pub max_graph_entities: usize,
    pub top_k: usize,
    pub full_scan_limit: usize,
    pub graph_mix: f64,
    pub hop_decay: f64,
    pub freshness_half_life_days: f64,
    pub freshness_floor: f64,
    pub tier_t0: f64,
    pub tier_t1: f64,
    pub tier_t2: f64,
    pub tier_t3: f64,
    pub entity_bytes: u64,
    pub edge_bytes: u64,
    pub path_bytes: u64,
    pub result_bytes: u64,
    pub bootstrap_samples: u32,
    pub bootstrap_seed: u64,
    pub suggestion_predicate: String,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            anchor_count: 8,
            max_hops: 2,
            max_neighbors: 16,
            max_paths: 64,
            max_graph_entities: 32,
            top_k: 10,
            full_scan_limit: 1_000_000,
            graph_mix: 0.25,
            hop_decay: 0.5,
            freshness_half_life_days: 180.0,
            freshness_floor: 0.25,
            tier_t0: 0.4,
            tier_t1: 0.6,
            tier_t2: 0.8,
            tier_t3: 1.0,
            entity_bytes: 256,
            edge_bytes: 96,
            path_bytes: 48,
            result_bytes: 128,
            bootstrap_samples: 10_000,
            bootstrap_seed: 0x0050_4541_524C,
            suggestion_predicate: "path_context.related".to_string(),
        }
    }
}

impl ExperimentConfig {
    pub fn canonical_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|err| err.to_string())
    }
}
