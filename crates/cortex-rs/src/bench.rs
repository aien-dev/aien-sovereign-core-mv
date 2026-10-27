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

use crate::db::Database;
use crate::models::{ClaimWriteInput, EntityWriteInput};
use std::fs;
use std::time::Instant;

pub fn run_benchmark(records: usize) -> Result<(), Box<dyn std::error::Error>> {
    if records == 0 {
        eprintln!("Error: --bench-records must be at least 1.");
        return Ok(());
    }

    let timestamp_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_path = std::env::temp_dir().join(format!(
        "cortex_bench_{}_{}.db",
        std::process::id(),
        timestamp_nanos
    ));
    let _ = fs::remove_file(&tmp_path);

    println!("================================================================================");
    println!("CORTEX-RS FLAGSHIP REPRODUCTION BENCHMARK");
    println!("Target: SQLite WAL + FTS5 Inverted Index + Graph Claim Traversal");
    println!("Records to ingest: {}", records);
    println!("================================================================================");

    let db = Database::open(&tmp_path)?;

    // 1. Ingestion Benchmark
    println!("\n[1/3] Benchmarking Entity Ingestion (FTS5 + SQLite WAL)...");
    let mut write_latencies_us: Vec<f64> = Vec::with_capacity(records);
    let mut entity_ids: Vec<String> = Vec::with_capacity(records);

    let sample_domains = [
        ("tpm_vault", "Hardware TPM 2.0 key attestation and zero disk plaintext secret security."),
        ("inference_scheduler", "Dynamic batching and continuous sequence allocation on Blackwell GB10 unified memory."),
        ("kv_cache", "Paged KV cache allocation with zero copy tensor memory reuse under high concurrency."),
        ("cortex_memory", "Lexical BM25 indexing with ONNX vector embeddings and sub-millisecond retrieval."),
        ("kernel_optimization", "GPU kernel tile optimization for FP8 and BF16 GEMM matrix multiplication."),
        ("resilience_protocol", "Crash-only idempotency, WAL transaction rollbacks, and self-healing supervisory loops."),
        ("consensus_engine", "Decentralized state replication with cryptographic hash chained audit records."),
        ("compiler_pipeline", "AOT binary compilation with strict link-time optimization and zero runtime interpreter overhead."),
    ];

    let start_ingest = Instant::now();
    for i in 0..records {
        let (domain, desc) = sample_domains[i % sample_domains.len()];
        let canonical_name = format!("{}:entity_{:05}", domain, i);
        let content = format!(
            "{} Record ID {} generated for production deployment metrics verification.",
            desc, i
        );
        let input = EntityWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            entity_type: "lesson".to_string(),
            canonical_name,
            content,
            aliases: vec![format!("alias_{:05}", i)],
            metadata: serde_json::json!({"index": i, "domain": domain}),
            confidence: 1.0,
            valid_from: None,
            valid_to: None,
            external_id: None,
        };

        let t0 = Instant::now();
        let receipt = db.upsert_entity(&input, None)?;
        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
        write_latencies_us.push(elapsed);
        entity_ids.push(receipt.target_id);
    }
    let total_ingest_time = start_ingest.elapsed();
    let ingest_qps = records as f64 / total_ingest_time.as_secs_f64();

    // 2. Search Benchmark
    println!(
        "[2/3] Benchmarking FTS5 Lexical Search across {} records...",
        records
    );
    let search_queries = [
        "tpm vault",
        "unified memory",
        "KV cache allocation",
        "lexical BM25",
        "kernel tile optimization",
        "crash-only idempotency",
        "cryptographic audit",
        "zero runtime interpreter",
        "metrics verification",
        "nonexistent_token_xyz_404",
    ];

    let query_iterations = 50;
    let total_queries = search_queries.len() * query_iterations;
    let mut search_latencies_us: Vec<f64> = Vec::with_capacity(total_queries);

    let start_search = Instant::now();
    for _ in 0..query_iterations {
        for query in &search_queries {
            let t0 = Instant::now();
            let results = db.search_entities(query, Some("atlas-memory"), 10)?;
            let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
            search_latencies_us.push(elapsed);
            let _ = results.len();
        }
    }
    let total_search_time = start_search.elapsed();
    let search_qps = total_queries as f64 / total_search_time.as_secs_f64();

    // 3. Graph Traversal Benchmark
    println!("[3/3] Benchmarking Bidirectional Graph Claim Traversal...");
    let num_claims = records.min(200);
    for i in 0..num_claims {
        let sub = &entity_ids[i];
        let obj = &entity_ids[(i + 1) % records];
        let claim = ClaimWriteInput {
            id: None,
            space: "atlas-memory".to_string(),
            subject_entity_id: sub.clone(),
            predicate: "depends_on".to_string(),
            object_entity_id: Some(obj.clone()),
            literal_value: None,
            confidence: 1.0,
            metadata: serde_json::json!({}),
        };
        db.upsert_claim(&claim)?;
    }

    let mut traverse_latencies_us: Vec<f64> = Vec::with_capacity(num_claims);
    let start_traverse = Instant::now();
    for id in &entity_ids[..num_claims] {
        let t0 = Instant::now();
        let _claims = db.traverse_claims(id, Some("atlas-memory"))?;
        let elapsed = t0.elapsed().as_secs_f64() * 1_000_000.0;
        traverse_latencies_us.push(elapsed);
    }
    let total_traverse_time = start_traverse.elapsed();
    let traverse_qps = if total_traverse_time.as_secs_f64() > 0.0 {
        num_claims as f64 / total_traverse_time.as_secs_f64()
    } else {
        0.0
    };

    // Compute percentiles
    write_latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    search_latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    traverse_latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |vec: &[f64], p: f64| -> f64 {
        if vec.is_empty() {
            return 0.0;
        }
        let idx = ((vec.len() as f64 * p) / 100.0).round() as usize;
        vec[idx.min(vec.len().saturating_sub(1))]
    };

    println!("\n================================================================================");
    println!("BENCHMARK RESULTS SUMMARY (Reproducible Single-Command Output)");
    println!("================================================================================");
    println!("Corpus Size:               {:>10} entities", records);
    println!("Total Search Queries Run:  {:>10}", total_queries);
    println!("--------------------------------------------------------------------------------");
    println!("METRIC                          p50 (µs)    p95 (µs)    p99 (µs)      Rate / QPS");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Entity Ingestion (WAL+FTS5)     {:>8.1}    {:>8.1}    {:>8.1}    {:>9.1} writes/s",
        percentile(&write_latencies_us, 50.0),
        percentile(&write_latencies_us, 95.0),
        percentile(&write_latencies_us, 99.0),
        ingest_qps
    );
    println!(
        "FTS5 Lexical Search             {:>8.1}    {:>8.1}    {:>8.1}    {:>9.1} qps",
        percentile(&search_latencies_us, 50.0),
        percentile(&search_latencies_us, 95.0),
        percentile(&search_latencies_us, 99.0),
        search_qps
    );
    println!(
        "Graph Claim Traversal           {:>8.1}    {:>8.1}    {:>8.1}    {:>9.1} qps",
        percentile(&traverse_latencies_us, 50.0),
        percentile(&traverse_latencies_us, 95.0),
        percentile(&traverse_latencies_us, 99.0),
        traverse_qps
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        "Search p50 in milliseconds:      {:.3} ms",
        percentile(&search_latencies_us, 50.0) / 1000.0
    );
    println!(
        "Search p99 in milliseconds:      {:.3} ms",
        percentile(&search_latencies_us, 99.0) / 1000.0
    );
    println!("================================================================================");

    // Explicitly drop db to flush and close SQLite WAL before file removal
    drop(db);
    let _ = fs::remove_file(&tmp_path);
    let _ = fs::remove_file(format!("{}-wal", tmp_path.display()));
    let _ = fs::remove_file(format!("{}-shm", tmp_path.display()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_zero_records_safe_exit() {
        let res = run_benchmark(0);
        assert!(res.is_ok());
    }

    #[test]
    fn test_benchmark_minimal_records() {
        let res = run_benchmark(2);
        assert!(res.is_ok());
    }
}
