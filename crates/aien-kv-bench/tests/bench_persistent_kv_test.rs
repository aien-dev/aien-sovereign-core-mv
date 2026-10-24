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

//! C9 (sc#485, ADR 0003 sections 10, 15.1, 16.5; CUTS.md C9): the persistent
//! KV benchmark's output contract. Host only: no GPU, no store I/O here (the
//! binary is only run with arguments it must refuse before doing any work).

use aien_kv_bench::report::{
    build_report, required_not_run_rows, summarize, BenchError, Machine, ReportMeta, Row,
    RowStatus, Workload,
};
use std::process::Command;

fn ok_row(mode: &str, cache_state: &str, latency: &[u64]) -> Row {
    let mut row = Row::new_ok(mode, "hosted", cache_state, 2, 512, "second_turn");
    row.queue_depth = Some(4);
    row.completed_requests = Some(latency.len() as u64);
    row.failed_requests = Some(0);
    row.latency_samples_us = latency.to_vec();
    row.ttft_samples_us = latency.iter().map(|l| l / 2).collect();
    row.decode_tokens_per_s = Some(100.0);
    row.peak_allocated_bytes = Some(1 << 20);
    row.peak_holder_bytes = Some(4096);
    row.peak_staging_bytes = Some(3072);
    row.peak_rss_bytes = Some(8 << 20);
    row.restored_tokens = Some(0);
    row.fill_percentiles();
    row
}

fn meta() -> ReportMeta {
    ReportMeta {
        measurement_id: "BENCH-PKV-HOST-TEST".into(),
        git_sha: "0".repeat(40),
        git_dirty: false,
        command: "bench_persistent_kv --modes cold-prefill".into(),
        started_utc: "2026-10-13T00:00:00Z".into(),
        platform: "hosted".into(),
        machine: Machine::default(),
        backend: "ReferenceCpuBackend".into(),
        model: "test".into(),
        workload: Workload::default(),
    }
}

/// B6: a row that reports numbers must carry the completed-request count,
/// request latency p50/p95, TTFT p50/p95 and every peak; a row missing any of
/// them is refused by name, and an accepted report serializes all of them.
#[test]
fn bench_output_requires_completed_requests_latency_percentiles_and_peaks() {
    let good = ok_row("cold-prefill", "none", &[100, 200, 300, 400]);
    good.check().expect("a complete row is accepted");

    type Strip = fn(&mut Row);
    let cases: [(&str, Strip); 7] = [
        ("completed_requests", |r| r.completed_requests = None),
        ("p50_us", |r| r.p50_us = None),
        ("p95_us", |r| r.p95_us = None),
        ("ttft_p95_us", |r| r.ttft_p95_us = None),
        ("peak_allocated_bytes", |r| r.peak_allocated_bytes = None),
        ("peak_staging_bytes", |r| r.peak_staging_bytes = None),
        ("peak_rss_bytes", |r| r.peak_rss_bytes = None),
    ];
    for (field, strip) in cases {
        let mut row = good.clone();
        strip(&mut row);
        match row.check() {
            Err(BenchError::MissingField { field: f, .. }) => assert_eq!(f, field),
            other => panic!("row without {field} must be refused, got {other:?}"),
        }
    }

    // Zero completed requests is not a measurement.
    let mut empty = good.clone();
    empty.completed_requests = Some(0);
    assert!(empty.check().is_err(), "zero completed requests refused");

    // Percentiles must come from the samples (nearest rank) and be ordered.
    assert_eq!(good.p50_us, Some(200));
    assert_eq!(good.p95_us, Some(400));
    let mut swapped = good.clone();
    swapped.p50_us = Some(500);
    assert!(swapped.check().is_err(), "p50 above p95 refused");

    let report = build_report(meta(), vec![good]).expect("report");
    let json = serde_json::to_value(&report).expect("json");
    let row = json["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .find(|r| r["mode"] == "cold-prefill")
        .expect("measured row present");
    for key in [
        "completed_requests",
        "failed_requests",
        "p50_us",
        "p95_us",
        "ttft_p50_us",
        "ttft_p95_us",
        "decode_tokens_per_s",
        "peak_allocated_bytes",
        "peak_staging_bytes",
        "peak_rss_bytes",
        "cache_state",
        "platform",
        "queue_depth",
    ] {
        assert!(!row[key].is_null(), "serialized row carries {key}: {row:#}");
    }
    assert!(
        json["samples_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64),
        "raw samples are digested"
    );
}

/// B7: no row is reported without its mode and cache_state labels, and the
/// binary refuses to start without an explicit mode list or with an unknown
/// mode label (it never picks a default mode for the operator).
#[test]
fn bench_binary_refuses_to_report_without_mode_and_cache_state_labels() {
    let mut no_cache = ok_row("cold-prefill", "none", &[10, 20]);
    no_cache.cache_state = None;
    assert!(
        matches!(no_cache.check(), Err(BenchError::MissingLabel { .. })),
        "row without cache_state refused"
    );
    let mut no_mode = ok_row("cold-prefill", "none", &[10, 20]);
    no_mode.mode = String::new();
    assert!(
        matches!(no_mode.check(), Err(BenchError::MissingLabel { .. })),
        "row without mode refused"
    );
    let mut bad_cache = ok_row("cold-prefill", "none", &[10, 20]);
    bad_cache.cache_state = Some("warm".into());
    assert!(bad_cache.check().is_err(), "unknown cache_state refused");
    assert!(
        build_report(meta(), vec![no_cache]).is_err(),
        "report with an unlabelled row refused"
    );

    let bin = env!("CARGO_BIN_EXE_bench_persistent_kv");
    let dir = std::env::temp_dir().join(format!("pkv-c9-b7-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("out.json");

    let no_modes = Command::new(bin)
        .args(["--out", out.to_str().unwrap()])
        .output()
        .expect("run binary");
    assert!(!no_modes.status.success(), "no --modes: must refuse");
    let err = String::from_utf8_lossy(&no_modes.stderr);
    assert!(err.contains("--modes"), "refusal names --modes: {err}");

    let bad_mode = Command::new(bin)
        .args(["--modes", "warm-ish", "--out", out.to_str().unwrap()])
        .output()
        .expect("run binary");
    assert!(!bad_mode.status.success(), "unknown mode: must refuse");
    let err = String::from_utf8_lossy(&bad_mode.stderr);
    assert!(err.contains("warm-ish"), "refusal names the label: {err}");

    assert!(!out.exists(), "a refused run writes no report");
    let _ = std::fs::remove_dir_all(&dir);
}

/// ADR 15.1: page-cache hits and device reads are never averaged together.
#[test]
fn bench_refuses_mixed_cache_state_summary() {
    let page = ok_row("restore-from-page-cache", "page_cache", &[100, 110, 120]);
    let page2 = ok_row("restore-from-page-cache", "page_cache", &[130, 140]);
    let device = ok_row("restore-from-device", "device", &[900, 950]);

    let s = summarize(&[&page, &page2]).expect("same cache_state summarizes");
    assert_eq!(s.cache_state, "page_cache");
    assert_eq!(s.samples, 5);
    assert_eq!(s.p50_us, 120);
    assert_eq!(s.p95_us, 140);

    match summarize(&[&page, &device]) {
        Err(BenchError::MixedCacheState { .. }) => {}
        other => panic!("mixed page_cache/device summary must be refused, got {other:?}"),
    }
    let mut unlabelled = page.clone();
    unlabelled.cache_state = None;
    assert!(
        summarize(&[&page, &unlabelled]).is_err(),
        "an unlabelled row cannot join a summary"
    );
    let not_run = Row::not_run("restore-from-device", "hosted", "device", "no root");
    assert!(
        summarize(&[&not_run]).is_err(),
        "a NOT_RUN row has nothing to summarize"
    );
}

/// B8: every report carries the native physical rows as NOT_RUN with a
/// reason, never as numbers, and a report that drops them is refused.
#[test]
fn bench_records_not_run_rows_for_native_physical() {
    let required = required_not_run_rows();
    let platforms: Vec<&str> = required.iter().map(|r| r.platform.as_str()).collect();
    assert!(platforms.contains(&"native_physical"), "Spark native row");
    assert!(
        platforms.contains(&"native physical (Pi 5)"),
        "Pi 5 physical row"
    );
    for row in &required {
        assert_eq!(row.status, RowStatus::NotRun);
        assert!(row.reason.as_deref().is_some_and(|r| !r.is_empty()));
        row.check().expect("a NOT_RUN row with a reason is valid");
    }

    let report =
        build_report(meta(), vec![ok_row("cold-prefill", "none", &[5, 6])]).expect("report");
    for want in &required {
        let found = report
            .rows
            .iter()
            .find(|r| r.platform == want.platform && r.mode == want.mode)
            .unwrap_or_else(|| panic!("report records {} NOT_RUN", want.platform));
        assert_eq!(found.status, RowStatus::NotRun);
    }
    let json = serde_json::to_string(&report).expect("json");
    assert!(json.contains("\"NOT_RUN\""), "status serialized as NOT_RUN");

    // A NOT_RUN row may not carry numbers, and needs a reason.
    let mut faked = Row::not_run("native-physical", "native_physical", "device", "aienos#336");
    faked.p50_us = Some(1);
    assert!(faked.check().is_err(), "NOT_RUN with numbers refused");
    let mut silent = Row::not_run("native-physical", "native_physical", "device", "x");
    silent.reason = None;
    assert!(silent.check().is_err(), "NOT_RUN without a reason refused");

    // A physical platform can never report measured numbers from this host run.
    let mut claimed = ok_row("native-physical", "device", &[1, 2]);
    claimed.platform = "native_physical".into();
    assert!(
        build_report(meta(), vec![claimed]).is_err(),
        "a measured native physical row from a host run is refused"
    );
}

/// The measuring path end to end on a small row (not a CUTS red test): the
/// restore row really restores through the host store and the daemon wiring,
/// the cold row restores nothing, and both rows pass the contract.
#[tokio::test]
async fn small_rows_measure_through_the_runtime_and_restore_from_the_host_store() {
    use aien_kv_bench::run::{run_row, Mode, RunSpec};
    let root = std::env::temp_dir().join(format!("pkv-c9-rows-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let spec = RunSpec {
        modes: vec![Mode::ColdPrefill, Mode::RestorePageCache],
        concurrency: vec![2],
        prompt_tokens: vec![80],
        rounds: 2,
        turn1_max_tokens: 4,
        turn2_suffix_tokens: 8,
        turn2_max_tokens: 4,
        block_size: 16,
        chunk_blocks: 2,
        store_root: root.clone(),
    };
    let restore = run_row(&spec, Mode::RestorePageCache, 2, 80)
        .await
        .expect("restore row");
    restore.check().expect("restore row valid");
    assert_eq!(restore.completed_requests, Some(4));
    assert_eq!(restore.cache_state.as_deref(), Some("page_cache"));
    // Turn 1 computes 80 + 4 - 1 = 83 tokens: two full 32-token chunks per
    // conversation, restored by each of the 4 measured turn-2 requests.
    assert_eq!(restore.restored_tokens, Some(4 * 64));
    assert!(restore.peak_holder_bytes.unwrap() > 0, "park holders seen");
    assert!(restore.peak_staging_bytes.unwrap() > 0);

    let cold = run_row(&spec, Mode::ColdPrefill, 2, 80)
        .await
        .expect("cold row");
    cold.check().expect("cold row valid");
    assert_eq!(cold.restored_tokens, Some(0));
    assert_eq!(cold.queue_depth, Some(0));
    assert!(
        run_row(&spec, Mode::RestoreDevice, 1, 80).await.is_err(),
        "the device mode is never measured"
    );
    let _ = std::fs::remove_dir_all(&root);
}
