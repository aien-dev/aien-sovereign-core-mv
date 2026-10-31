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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalFixture {
    #[serde(default)]
    pub kind: String,
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalCase {
    pub id: String,
    pub space_slug: String,
    pub query: String,
    pub relevant_entity_ids: Vec<String>,
    pub query_embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureRejection {
    pub reasons: Vec<String>,
}

pub fn load_fixture(bytes: &[u8]) -> Result<EvalFixture, String> {
    serde_json::from_slice(bytes).map_err(|err| format!("fixture JSON: {err}"))
}

pub fn structural_rejection(fixture: &EvalFixture) -> FixtureRejection {
    let mut reasons = Vec::new();
    if fixture.cases.len() < 50 {
        reasons.push(format!(
            "fixture has {} cases; a scientific run needs at least 50",
            fixture.cases.len()
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for case in &fixture.cases {
        if !seen.insert(case.id.clone()) {
            reasons.push(format!("duplicate case id {}", case.id));
        }
        if case.relevant_entity_ids.is_empty() {
            reasons.push(format!("case {} has no relevant entities", case.id));
        }
        if case.query_embedding.is_empty()
            || case.query_embedding.iter().any(|value| !value.is_finite())
        {
            reasons.push(format!("case {} needs a finite query embedding", case.id));
        }
        if case.query.trim().is_empty() || case.space_slug.trim().is_empty() {
            reasons.push(format!("case {} needs a query and a space", case.id));
        }
    }
    FixtureRejection { reasons }
}

pub fn relational_case(query: &str, canonical_names: &[String]) -> bool {
    let needle = query.to_lowercase();
    canonical_names
        .iter()
        .any(|name| !name.to_lowercase().contains(&needle))
}
