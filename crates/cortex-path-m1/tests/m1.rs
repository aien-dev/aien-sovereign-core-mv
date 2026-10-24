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

use chrono::{TimeZone, Utc};
use cortex_path_m1::config::ExperimentConfig;
use cortex_path_m1::fixture::{self, EvalCase, EvalFixture};
use cortex_path_m1::metrics::{self, ci_excludes_zero};
use cortex_path_m1::run::{self, RunOutcome};
use cortex_path_m1::score::{self, EdgeView, WalkEdge};
use cortex_rs::{ClaimStatus, ClaimWriteInput, Database, EntityWriteInput, VerificationTier};
use rusqlite::Connection;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn snapshot_time() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 22, 0, 0, 0).unwrap()
}

fn entity(id: &str, name: &str) -> EntityWriteInput {
    EntityWriteInput {
        id: Some(id.to_string()),
        space: "atlas-memory".to_string(),
        entity_type: "discovery".to_string(),
        canonical_name: name.to_string(),
        content: String::new(),
        aliases: Vec::new(),
        metadata: json!({}),
        confidence: 1.0,
        valid_from: None,
        valid_to: None,
        external_id: None,
    }
}

fn claim(id: &str, subject: &str, object: &str) -> ClaimWriteInput {
    ClaimWriteInput {
        id: Some(id.to_string()),
        space: "atlas-memory".to_string(),
        subject_entity_id: subject.to_string(),
        predicate: "links".to_string(),
        object_entity_id: Some(object.to_string()),
        literal_value: None,
        confidence: 1.0,
        metadata: json!({}),
    }
}

fn edge(status: ClaimStatus, retracted: bool, object: Option<&str>) -> EdgeView {
    EdgeView {
        claim_id: "c".to_string(),
        subject: "a".to_string(),
        object: object.unwrap_or("b").to_string(),
        confidence: 1.0,
        tier: VerificationTier::T0Direct,
        evidence_count: 0,
        created_at: snapshot_time().to_rfc3339(),
        status,
        retracted,
        subject_space: "atlas-memory".to_string(),
        object_space: "atlas-memory".to_string(),
        claim_space: "atlas-memory".to_string(),
    }
}

#[test]
fn admission_keeps_active_same_space_entity_edges() {
    let active = edge(ClaimStatus::Active, false, Some("b"));
    assert!(score::is_admitted(&active, "atlas-memory"));
    assert!(!score::is_admitted(
        &edge(ClaimStatus::Superseded, false, Some("b")),
        "atlas-memory"
    ));
    assert!(!score::is_admitted(
        &edge(ClaimStatus::Disputed, false, Some("b")),
        "atlas-memory"
    ));
    assert!(!score::is_admitted(
        &edge(ClaimStatus::Invalidated, false, Some("b")),
        "atlas-memory"
    ));
    assert!(!score::is_admitted(
        &edge(ClaimStatus::Active, true, Some("b")),
        "atlas-memory"
    ));
    let mut literal = edge(ClaimStatus::Active, false, Some("b"));
    literal.object.clear();
    assert!(!score::is_admitted(&literal, "atlas-memory"));
    let mut other_space = edge(ClaimStatus::Active, false, Some("b"));
    other_space.object_space = "other".to_string();
    assert!(!score::is_admitted(&other_space, "atlas-memory"));
}

#[test]
fn edge_weight_uses_tier_evidence_and_age() {
    let config = ExperimentConfig::default();
    let mut view = edge(ClaimStatus::Active, false, Some("b"));
    view.tier = VerificationTier::T2Verified;
    view.evidence_count = 0;
    view.confidence = 1.0;
    let fresh = score::edge_weight(&config, &view, snapshot_time());
    assert!((fresh - 0.4).abs() < 1e-9);
    view.created_at = "2025-09-22T00:00:00+00:00".to_string();
    let aged = score::edge_weight(&config, &view, snapshot_time());
    assert!(aged < fresh);
    assert!(aged >= config.freshness_floor * 0.4);
}

#[test]
fn two_hops_score_and_the_third_does_not() {
    let config = ExperimentConfig::default();
    let mut adj = BTreeMap::new();
    let mut link = |from: &str, to: &str, claim: &str| {
        adj.entry(from.to_string())
            .or_insert_with(Vec::new)
            .push(WalkEdge {
                claim_id: claim.to_string(),
                neighbor: to.to_string(),
                weight: 1.0,
            });
    };
    link("a", "b", "ab");
    link("b", "c", "bc");
    link("c", "d", "cd");
    let walked = score::walk_paths(&config, &[("a".to_string(), 1.0)], &adj, &BTreeSet::new());
    assert!(walked.graph_score.contains_key("b"));
    assert!(walked.graph_score.contains_key("c"));
    assert!(!walked.graph_score.contains_key("d"));
    assert!((walked.graph_score["c"] - 0.5).abs() < 1e-9);
}

#[test]
fn zero_graph_fusion_keeps_baseline_order_and_breaks_ties_by_id() {
    let config = ExperimentConfig::default();
    let ranked = score::fuse_and_rank(
        &config,
        &[
            ("b".to_string(), 0.2),
            ("a".to_string(), 0.2),
            ("c".to_string(), 0.9),
        ],
        &BTreeMap::new(),
    );
    let ids: Vec<_> = ranked.iter().map(|item| item.id.as_str()).collect();
    assert_eq!(ids, vec!["c", "a", "b"]);
    assert!((ranked[1].final_score - 0.2).abs() < 1e-12);
}

#[test]
fn indirect_anchor_pair_is_a_suggestion_until_a_direct_edge_exists() {
    let config = ExperimentConfig::default();
    let mut adj = BTreeMap::new();
    for (from, to, claim) in [
        ("a", "c", "ac"),
        ("c", "b", "cb"),
        ("b", "c", "cb"),
        ("c", "a", "ac"),
    ] {
        adj.entry(from.to_string())
            .or_insert_with(Vec::new)
            .push(WalkEdge {
                claim_id: claim.to_string(),
                neighbor: to.to_string(),
                weight: 1.0,
            });
    }
    let anchors = vec![("a".to_string(), 0.9), ("b".to_string(), 0.9)];
    let open = score::walk_paths(&config, &anchors, &adj, &BTreeSet::new());
    assert_eq!(open.suggestions.len(), 1);
    let mut direct = BTreeSet::new();
    direct.insert(score::ordered_pair("a", "b"));
    let closed = score::walk_paths(&config, &anchors, &adj, &direct);
    assert!(closed.suggestions.is_empty());
}

#[test]
fn ndcg_recall_and_bootstrap_match_the_contract() {
    let relevant = vec!["a".to_string()];
    let perfect = metrics::ndcg_at_k(&["a".to_string(), "b".to_string()], &relevant, 10);
    assert!((perfect - 1.0).abs() < 1e-12);
    let second = metrics::ndcg_at_k(&["b".to_string(), "a".to_string()], &relevant, 10);
    assert!((second - (1.0 / 3.0_f64.log2())).abs() < 1e-12);
    assert!(
        (metrics::recall_at_k(&["b".to_string(), "a".to_string()], &relevant, 10) - 1.0).abs()
            < 1e-12
    );
    let positive = metrics::bootstrap_mean_ci(&[0.2, 0.2, 0.2, 0.2], 200, 0x504541524C).unwrap();
    assert!(ci_excludes_zero(positive));
    let flat = metrics::bootstrap_mean_ci(&[0.0, 0.0, 0.0, 0.0], 200, 0x504541524C).unwrap();
    assert!(!ci_excludes_zero(flat));
    let once = metrics::bootstrap_mean_ci(&[0.1, 0.4], 100, 0x504541524C).unwrap();
    let twice = metrics::bootstrap_mean_ci(&[0.1, 0.4], 100, 0x504541524C).unwrap();
    assert_eq!(once, twice);
}

#[test]
fn example_fixture_is_not_a_scientific_verdict() {
    let bytes =
        include_bytes!("../../../experiments/fixtures/pearl_cortex_m1_entities.example.json");
    let fixture = fixture::load_fixture(bytes).unwrap();
    let rejection = fixture::structural_rejection(&fixture);
    assert!(!rejection.reasons.is_empty());
}

#[test]
fn scoring_does_not_insert_a_candidate_and_leaves_the_source_hash() {
    let dir = std::env::temp_dir().join(format!("cortex-path-m1-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.sqlite");
    {
        let db = Database::open(&source).unwrap();
        db.upsert_entity(&entity("a", "hit-a"), None).unwrap();
        db.upsert_entity(&entity("b", "hit-b"), None).unwrap();
        db.upsert_entity(&entity("c", "bridge"), None).unwrap();
        db.upsert_claim(&claim("ac", "a", "c")).unwrap();
        db.upsert_claim(&claim("cb", "c", "b")).unwrap();
    }
    checkpoint(&source);
    let before = run::sha256_file(&source).unwrap();
    let mut cases = Vec::new();
    for index in 0..50 {
        let relevant = if index < 15 {
            vec!["c".to_string()]
        } else {
            vec!["a".to_string()]
        };
        cases.push(EvalCase {
            id: format!("q{index}"),
            space_slug: "atlas-memory".to_string(),
            query: "hit".to_string(),
            relevant_entity_ids: relevant,
            query_embedding: vec![1.0, 0.0],
        });
    }
    let fixture = EvalFixture {
        kind: "unit".to_string(),
        cases,
    };
    let bytes = serde_json::to_vec(&fixture).unwrap();
    let outcome = run::run_experiment(&source, &bytes, snapshot_time(), b"test-binary").unwrap();
    match outcome {
        RunOutcome::Report(report) => {
            assert_eq!(report.writes, 0);
            assert!(!report.queries[0].suggestions.is_empty());
        }
        RunOutcome::Rejected(rejection) => panic!("expected a scored run: {:?}", rejection.reasons),
    }
    assert_eq!(run::sha256_file(&source).unwrap(), before);
    let _ = std::fs::remove_dir_all(&dir);
}

fn checkpoint(path: &Path) {
    let conn = Connection::open(path).unwrap();
    let _: (i64, i64, i64) = conn
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .unwrap();
    drop(conn);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}
