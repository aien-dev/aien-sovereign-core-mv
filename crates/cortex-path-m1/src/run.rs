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
use crate::fixture::{self, EvalFixture, FixtureRejection};
use crate::load::load_admitted_graph;
use crate::metrics::{self, ci_excludes_zero};
use crate::score::{self, CandidateSuggestion};
use chrono::{DateTime, Utc};
use cortex_rs::Database;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct QueryRow {
    pub query_id: String,
    pub baseline_ids: Vec<String>,
    pub treatment_ids: Vec<String>,
    pub baseline_ndcg: f64,
    pub treatment_ndcg: f64,
    pub absolute_ndcg_delta: f64,
    pub relative_ndcg: Option<f64>,
    pub baseline_recall: f64,
    pub treatment_recall: f64,
    pub baseline_latency_ms: f64,
    pub treatment_latency_ms: f64,
    pub baseline_accounted_bytes: u64,
    pub treatment_accounted_bytes: u64,
    pub paths_scored: usize,
    pub suggestions: Vec<CandidateSuggestion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ZeroBaselineRow {
    pub query_id: String,
    pub baseline_ndcg: f64,
    pub treatment_ndcg: f64,
    pub absolute_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentReport {
    pub dataset_snapshot_sha256: String,
    pub eval_fixture_sha256: String,
    pub experiment_binary_sha256: String,
    pub config_sha256: String,
    pub snapshot_time: String,
    pub queries: Vec<QueryRow>,
    pub relative_mean_ndcg: Option<f64>,
    pub absolute_mean_ndcg_delta: f64,
    pub zero_baseline: Vec<ZeroBaselineRow>,
    pub recall_at_10_delta: f64,
    pub baseline_p95_ms: f64,
    pub treatment_p95_ms: f64,
    pub baseline_peak_accounted_bytes: u64,
    pub treatment_peak_accounted_bytes: u64,
    pub bootstrap_low: Option<f64>,
    pub bootstrap_high: Option<f64>,
    pub writes: u64,
    pub pass: bool,
    pub strong_pass: bool,
    pub fail: bool,
}

#[derive(Debug)]
pub enum RunOutcome {
    Rejected(FixtureRejection),
    Report(Box<ExperimentReport>),
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

pub fn run_experiment(
    snapshot: &Path,
    fixture_bytes: &[u8],
    snapshot_time: DateTime<Utc>,
    binary_bytes: &[u8],
) -> Result<RunOutcome, String> {
    let fixture = fixture::load_fixture(fixture_bytes)?;
    let mut rejection = fixture::structural_rejection(&fixture);
    if !rejection.reasons.is_empty() {
        return Ok(RunOutcome::Rejected(rejection));
    }

    refuse_sidecar(snapshot)?;
    let source_hash = sha256_file(snapshot)?;
    let temp_dir = std::env::temp_dir().join(format!(
        "cortex-path-m1-{}",
        sha256_bytes(source_hash.as_bytes())
    ));
    fs::create_dir_all(&temp_dir).map_err(|err| err.to_string())?;
    let copy_path = temp_dir.join("snapshot.sqlite");
    fs::copy(snapshot, &copy_path).map_err(|err| format!("copy snapshot: {err}"))?;

    let outcome = run_on_copy(
        &copy_path,
        &fixture,
        fixture_bytes,
        snapshot_time,
        binary_bytes,
        &source_hash,
        &mut rejection,
    );
    let after = sha256_file(snapshot)?;
    if after != source_hash {
        return Err("source snapshot hash changed during the run".to_string());
    }
    outcome
}

fn run_on_copy(
    copy_path: &Path,
    fixture: &EvalFixture,
    fixture_bytes: &[u8],
    snapshot_time: DateTime<Utc>,
    binary_bytes: &[u8],
    source_hash: &str,
    rejection: &mut FixtureRejection,
) -> Result<RunOutcome, String> {
    let db = Database::open(copy_path).map_err(|err| err.to_string())?;
    let before = table_counts(copy_path)?;
    let mut relational = 0usize;
    for case in &fixture.cases {
        let mut names = Vec::new();
        for entity_id in &case.relevant_entity_ids {
            match db
                .get_entity(entity_id, Some(&case.space_slug))
                .map_err(|err| err.to_string())?
            {
                Some(entity) => names.push(entity.canonical_name),
                None => rejection
                    .reasons
                    .push(format!("case {} missing entity {entity_id}", case.id)),
            }
        }
        if fixture::relational_case(&case.query, &names) {
            relational += 1;
        }
    }
    if relational < 15 {
        rejection.reasons.push(format!(
            "fixture has {relational} relational cases; a scientific run needs at least 15"
        ));
    }
    if !rejection.reasons.is_empty() {
        return Ok(RunOutcome::Rejected(rejection.clone()));
    }

    let config = ExperimentConfig::default();
    let mut rows = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        rows.push(score_case(&db, case, &config, snapshot_time)?);
    }
    let after = table_counts(copy_path)?;
    let writes = count_delta(&before, &after);

    Ok(RunOutcome::Report(Box::new(build_report(
        &config,
        rows,
        source_hash,
        &sha256_bytes(fixture_bytes),
        &sha256_bytes(binary_bytes),
        snapshot_time,
        writes,
    )?)))
}

fn score_case(
    db: &Database,
    case: &fixture::EvalCase,
    config: &ExperimentConfig,
    snapshot_time: DateTime<Utc>,
) -> Result<QueryRow, String> {
    let space = Some(case.space_slug.as_str());
    let baseline_started = Instant::now();
    let baseline_hits = db
        .recall_entities(
            &case.query,
            Some(&case.query_embedding),
            space,
            config.top_k,
        )
        .map_err(|err| err.to_string())?;
    let baseline_latency_ms = elapsed_ms(baseline_started);
    let baseline_ids: Vec<String> = baseline_hits
        .iter()
        .map(|item| item.entity.id.clone())
        .collect();

    let treatment_started = Instant::now();
    let scored = db
        .recall_entities(
            &case.query,
            Some(&case.query_embedding),
            space,
            config.full_scan_limit,
        )
        .map_err(|err| err.to_string())?;
    let baseline_pairs: Vec<(String, f64)> = scored
        .iter()
        .map(|item| (item.entity.id.clone(), item.final_score))
        .collect();
    let entity_ids: Vec<String> = baseline_pairs.iter().map(|(id, _)| id.clone()).collect();
    let anchors = score::select_anchors(config, &baseline_pairs);
    let graph = load_admitted_graph(db, &case.space_slug, &entity_ids, config, snapshot_time)?;
    let walked = score::walk_paths(config, &anchors, &graph.adjacency, &graph.direct);
    let ranked = score::fuse_and_rank(config, &baseline_pairs, &walked.graph_score);
    let treatment_latency_ms = elapsed_ms(treatment_started);
    let treatment_ids: Vec<String> = ranked.iter().map(|item| item.id.clone()).collect();

    let baseline_ndcg = metrics::ndcg_at_k(&baseline_ids, &case.relevant_entity_ids, config.top_k);
    let treatment_ndcg =
        metrics::ndcg_at_k(&treatment_ids, &case.relevant_entity_ids, config.top_k);
    let relative_ndcg = if baseline_ndcg > 0.0 {
        Some((treatment_ndcg - baseline_ndcg) / baseline_ndcg)
    } else {
        None
    };
    let baseline_recall =
        metrics::recall_at_k(&baseline_ids, &case.relevant_entity_ids, config.top_k);
    let treatment_recall =
        metrics::recall_at_k(&treatment_ids, &case.relevant_entity_ids, config.top_k);
    let entity_bytes = entity_ids.len() as u64 * config.entity_bytes;
    let top_bytes = config.top_k as u64 * config.result_bytes;
    let suggestions = walked
        .suggestions
        .iter()
        .map(|(subject, object, suggestion_score)| CandidateSuggestion {
            query_id: case.id.clone(),
            subject: subject.clone(),
            predicate: config.suggestion_predicate.clone(),
            object: object.clone(),
            score: *suggestion_score,
        })
        .collect();

    Ok(QueryRow {
        query_id: case.id.clone(),
        baseline_ids,
        treatment_ids,
        baseline_ndcg,
        treatment_ndcg,
        absolute_ndcg_delta: treatment_ndcg - baseline_ndcg,
        relative_ndcg,
        baseline_recall,
        treatment_recall,
        baseline_latency_ms,
        treatment_latency_ms,
        baseline_accounted_bytes: entity_bytes + top_bytes,
        treatment_accounted_bytes: entity_bytes
            + graph.edges as u64 * config.edge_bytes
            + walked.paths_scored as u64 * config.path_bytes
            + top_bytes,
        paths_scored: walked.paths_scored,
        suggestions,
    })
}

fn build_report(
    config: &ExperimentConfig,
    queries: Vec<QueryRow>,
    source_hash: &str,
    fixture_hash: &str,
    binary_hash: &str,
    snapshot_time: DateTime<Utc>,
    writes: u64,
) -> Result<ExperimentReport, String> {
    let relatives: Vec<f64> = queries.iter().filter_map(|row| row.relative_ndcg).collect();
    let relative_mean_ndcg = if relatives.is_empty() {
        None
    } else {
        Some(metrics::mean(&relatives))
    };
    let absolute_mean_ndcg_delta = metrics::mean(
        &queries
            .iter()
            .map(|row| row.absolute_ndcg_delta)
            .collect::<Vec<_>>(),
    );
    let zero_baseline = queries
        .iter()
        .filter(|row| row.baseline_ndcg == 0.0)
        .map(|row| ZeroBaselineRow {
            query_id: row.query_id.clone(),
            baseline_ndcg: row.baseline_ndcg,
            treatment_ndcg: row.treatment_ndcg,
            absolute_delta: row.absolute_ndcg_delta,
        })
        .collect();
    let recall_at_10_delta = metrics::mean(
        &queries
            .iter()
            .map(|row| row.treatment_recall - row.baseline_recall)
            .collect::<Vec<_>>(),
    );
    let baseline_p95_ms =
        metrics::percentile_95(queries.iter().map(|row| row.baseline_latency_ms).collect());
    let treatment_p95_ms =
        metrics::percentile_95(queries.iter().map(|row| row.treatment_latency_ms).collect());
    let interval =
        metrics::bootstrap_mean_ci(&relatives, config.bootstrap_samples, config.bootstrap_seed);
    let ci_ok = interval.map(ci_excludes_zero).unwrap_or(false);
    let latency_ok = treatment_p95_ms <= 1.5 * baseline_p95_ms.max(f64::EPSILON);
    let baseline_peak = queries
        .iter()
        .map(|row| row.baseline_accounted_bytes)
        .max()
        .unwrap_or(0);
    let treatment_peak = queries
        .iter()
        .map(|row| row.treatment_accounted_bytes)
        .max()
        .unwrap_or(0);
    let memory_ok = treatment_peak as f64 <= 2.0 * baseline_peak.max(1) as f64;
    let recall_ok = recall_at_10_delta >= -0.01;
    let gain_ok = relative_mean_ndcg.unwrap_or(0.0) >= 0.05;
    let pass = writes == 0 && ci_ok && latency_ok && memory_ok && recall_ok && gain_ok;
    let strong_pass = pass
        && relative_mean_ndcg.unwrap_or(0.0) >= 0.10
        && recall_at_10_delta > 0.0
        && treatment_p95_ms <= 1.25 * baseline_p95_ms.max(f64::EPSILON);
    let config_sha256 = sha256_bytes(&config.canonical_json()?);

    Ok(ExperimentReport {
        dataset_snapshot_sha256: source_hash.to_string(),
        eval_fixture_sha256: fixture_hash.to_string(),
        experiment_binary_sha256: binary_hash.to_string(),
        config_sha256,
        snapshot_time: snapshot_time.to_rfc3339(),
        queries,
        relative_mean_ndcg,
        absolute_mean_ndcg_delta,
        zero_baseline,
        recall_at_10_delta,
        baseline_p95_ms,
        treatment_p95_ms,
        baseline_peak_accounted_bytes: baseline_peak,
        treatment_peak_accounted_bytes: treatment_peak,
        bootstrap_low: interval.map(|pair| pair.0),
        bootstrap_high: interval.map(|pair| pair.1),
        writes,
        pass,
        strong_pass,
        fail: !pass,
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn refuse_sidecar(snapshot: &Path) -> Result<(), String> {
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(format!("{}{suffix}", snapshot.display()));
        if sidecar.exists() {
            return Err(format!(
                "snapshot has {suffix}; checkpoint it to one file before hashing"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TableCounts {
    memory_candidates: i64,
    promotion_receipts: i64,
    claims: i64,
}

fn table_counts(path: &Path) -> Result<TableCounts, String> {
    let conn =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|err| err.to_string())?;
    let one = |sql: &str| -> Result<i64, String> {
        conn.query_row(sql, [], |row| row.get(0))
            .map_err(|err| err.to_string())
    };
    Ok(TableCounts {
        memory_candidates: one("SELECT COUNT(*) FROM memory_candidates")?,
        promotion_receipts: one("SELECT COUNT(*) FROM promotion_receipts")?,
        claims: one("SELECT COUNT(*) FROM claims")?,
    })
}

fn count_delta(before: &TableCounts, after: &TableCounts) -> u64 {
    let candidates = (after.memory_candidates - before.memory_candidates).max(0) as u64;
    let promotions = (after.promotion_receipts - before.promotion_receipts).max(0) as u64;
    let claims = (after.claims - before.claims).max(0) as u64;
    candidates + promotions + claims
}
