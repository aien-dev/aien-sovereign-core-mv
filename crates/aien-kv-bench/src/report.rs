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

//! The benchmark's output contract (CUTS.md C9, schema B6; ADR 0003 sections
//! 10, 15.1 and 16.5).
//!
//! Every row carries a mode, a platform and a `cache_state` label. A row is
//! either `OK` (measured, every metric present) or `NOT_RUN` (a reason and no
//! numbers). Latency summaries never mix cache states. Every report carries
//! the native physical rows as `NOT_RUN` (B8): a host run cannot speak for
//! native hardware.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

/// Version tag of the JSON this module writes.
pub const SCHEMA: &str = "aien.pkv.bench.v1";

/// The `cache_state` labels (ADR 15.1). `none`: the mode reads nothing from
/// the store; `page_cache`: objects were written moments before the read and
/// no cache drop happened; `device`: residency verified as zero before the
/// read; `unknown`: not verified either way.
pub const CACHE_STATES: [&str; 4] = ["none", "page_cache", "device", "unknown"];

/// Platform labels of ADR 16.5. Only `hosted` may carry numbers in a host run.
pub const HOSTED: &str = "hosted";
pub const PLATFORMS: [&str; 4] = [
    HOSTED,
    "native_emulator",
    "native physical (Pi 5)",
    "native_physical",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowStatus {
    #[serde(rename = "OK")]
    Ok,
    #[serde(rename = "NOT_RUN")]
    NotRun,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BenchError {
    MissingLabel { label: &'static str, mode: String },
    BadLabel { label: &'static str, value: String },
    MissingField { field: &'static str, mode: String },
    Inconsistent { mode: String, what: String },
    MixedCacheState { first: String, other: String },
    NothingToSummarize,
    MissingNotRun { platform: String },
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::MissingLabel { label, mode } => {
                write!(f, "row {mode:?} has no {label} label")
            }
            BenchError::BadLabel { label, value } => write!(f, "unknown {label} label {value:?}"),
            BenchError::MissingField { field, mode } => {
                write!(f, "row {mode:?} reports no {field}")
            }
            BenchError::Inconsistent { mode, what } => write!(f, "row {mode:?}: {what}"),
            BenchError::MixedCacheState { first, other } => write!(
                f,
                "refusing to summarize cache_state {first:?} together with {other:?}"
            ),
            BenchError::NothingToSummarize => write!(f, "no measured row to summarize"),
            BenchError::MissingNotRun { platform } => {
                write!(f, "report has no NOT_RUN row for {platform}")
            }
        }
    }
}

impl std::error::Error for BenchError {}

/// One mode at one concurrency and prompt length, or one NOT_RUN entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub mode: String,
    pub status: RowStatus,
    pub reason: Option<String>,
    pub platform: String,
    pub cache_state: Option<String>,
    pub concurrency: Option<u32>,
    pub prompt_tokens: Option<u32>,
    /// `second_turn` (the measured request continues a conversation) or
    /// `single_turn` (resident-only: one request, nothing leaves memory).
    pub workload: Option<String>,
    /// Store I/O operations the storage worker may have dispatched at once
    /// (`WorkerConfig::dispatch_depth`); 0 when the mode uses no store.
    pub queue_depth: Option<u32>,
    pub completed_requests: Option<u64>,
    pub failed_requests: Option<u64>,
    /// Request latency, submit to the finish event (microseconds).
    pub p50_us: Option<u64>,
    pub p95_us: Option<u64>,
    /// Submit to the first generated token.
    pub ttft_p50_us: Option<u64>,
    pub ttft_p95_us: Option<u64>,
    /// Tokens after each request's first, over the decode window.
    pub decode_tokens_per_s: Option<f64>,
    /// `kv_pool_allocated_bytes` (constant after start, ADR 10).
    pub peak_allocated_bytes: Option<u64>,
    /// Peak `kv_pool_holder_blocks` times the block stride. Holders are pool
    /// blocks, so this is inside `peak_allocated_bytes`, never added to it.
    pub peak_holder_bytes: Option<u64>,
    /// `kv_host_tail_scratch_bytes` for store modes (ADR 10), else 0.
    pub peak_staging_bytes: Option<u64>,
    /// `VmHWM` after a reset at the start of the row.
    pub peak_rss_bytes: Option<u64>,
    /// Prefix tokens restored from the store over the measured requests.
    pub restored_tokens: Option<u64>,
    pub latency_samples_us: Vec<u64>,
    pub ttft_samples_us: Vec<u64>,
}

impl Row {
    /// An empty measured row; the harness fills the metrics.
    pub fn new_ok(
        mode: &str,
        platform: &str,
        cache_state: &str,
        concurrency: u32,
        prompt_tokens: u32,
        workload: &str,
    ) -> Self {
        Self {
            mode: mode.to_string(),
            status: RowStatus::Ok,
            reason: None,
            platform: platform.to_string(),
            cache_state: Some(cache_state.to_string()),
            concurrency: Some(concurrency),
            prompt_tokens: Some(prompt_tokens),
            workload: Some(workload.to_string()),
            queue_depth: None,
            completed_requests: None,
            failed_requests: None,
            p50_us: None,
            p95_us: None,
            ttft_p50_us: None,
            ttft_p95_us: None,
            decode_tokens_per_s: None,
            peak_allocated_bytes: None,
            peak_holder_bytes: None,
            peak_staging_bytes: None,
            peak_rss_bytes: None,
            restored_tokens: None,
            latency_samples_us: Vec::new(),
            ttft_samples_us: Vec::new(),
        }
    }

    /// A row that was not run, with the reason and no numbers.
    pub fn not_run(mode: &str, platform: &str, cache_state: &str, reason: &str) -> Self {
        let mut row = Self::new_ok(mode, platform, cache_state, 0, 0, "none");
        row.status = RowStatus::NotRun;
        row.reason = Some(reason.to_string());
        row.concurrency = None;
        row.prompt_tokens = None;
        row.workload = None;
        row
    }

    /// Sets the four percentiles from the raw samples (nearest rank).
    pub fn fill_percentiles(&mut self) {
        self.p50_us = percentile(&self.latency_samples_us, 50);
        self.p95_us = percentile(&self.latency_samples_us, 95);
        self.ttft_p50_us = percentile(&self.ttft_samples_us, 50);
        self.ttft_p95_us = percentile(&self.ttft_samples_us, 95);
    }

    fn metric_fields(&self) -> [(&'static str, bool); 13] {
        [
            ("completed_requests", self.completed_requests.is_some()),
            ("failed_requests", self.failed_requests.is_some()),
            ("p50_us", self.p50_us.is_some()),
            ("p95_us", self.p95_us.is_some()),
            ("ttft_p50_us", self.ttft_p50_us.is_some()),
            ("ttft_p95_us", self.ttft_p95_us.is_some()),
            ("decode_tokens_per_s", self.decode_tokens_per_s.is_some()),
            ("peak_allocated_bytes", self.peak_allocated_bytes.is_some()),
            ("peak_holder_bytes", self.peak_holder_bytes.is_some()),
            ("peak_staging_bytes", self.peak_staging_bytes.is_some()),
            ("peak_rss_bytes", self.peak_rss_bytes.is_some()),
            ("queue_depth", self.queue_depth.is_some()),
            ("restored_tokens", self.restored_tokens.is_some()),
        ]
    }

    /// Refuses a row without its labels, a measured row missing any metric,
    /// and a NOT_RUN row that carries numbers or no reason.
    pub fn check(&self) -> Result<(), BenchError> {
        let mode = self.mode.clone();
        if self.mode.trim().is_empty() {
            return Err(BenchError::MissingLabel {
                label: "mode",
                mode,
            });
        }
        if self.platform.trim().is_empty() {
            return Err(BenchError::MissingLabel {
                label: "platform",
                mode,
            });
        }
        if !PLATFORMS.contains(&self.platform.as_str()) {
            return Err(BenchError::BadLabel {
                label: "platform",
                value: self.platform.clone(),
            });
        }
        let Some(cache) = self.cache_state.as_deref().filter(|c| !c.is_empty()) else {
            return Err(BenchError::MissingLabel {
                label: "cache_state",
                mode,
            });
        };
        if !CACHE_STATES.contains(&cache) {
            return Err(BenchError::BadLabel {
                label: "cache_state",
                value: cache.to_string(),
            });
        }
        match self.status {
            RowStatus::NotRun => {
                if self.reason.as_deref().is_none_or(|r| r.trim().is_empty()) {
                    return Err(BenchError::MissingField {
                        field: "reason",
                        mode,
                    });
                }
                if let Some((field, _)) = self.metric_fields().iter().find(|(_, set)| *set) {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: format!("NOT_RUN row carries {field}"),
                    });
                }
                if !self.latency_samples_us.is_empty() || !self.ttft_samples_us.is_empty() {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: "NOT_RUN row carries samples".into(),
                    });
                }
                Ok(())
            }
            RowStatus::Ok => {
                for (field, label) in [
                    ("concurrency", self.concurrency.is_some()),
                    ("prompt_tokens", self.prompt_tokens.is_some()),
                    ("workload", self.workload.is_some()),
                ] {
                    if !label {
                        return Err(BenchError::MissingField { field, mode });
                    }
                }
                if let Some((field, _)) = self.metric_fields().iter().find(|(_, set)| !*set) {
                    return Err(BenchError::MissingField { field, mode });
                }
                let completed = self.completed_requests.unwrap_or(0);
                if completed == 0 {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: "zero completed requests".into(),
                    });
                }
                if self.latency_samples_us.len() as u64 != completed
                    || self.ttft_samples_us.len() as u64 != completed
                {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: format!(
                            "{} latency and {} ttft samples for {completed} completed requests",
                            self.latency_samples_us.len(),
                            self.ttft_samples_us.len()
                        ),
                    });
                }
                let want = (
                    percentile(&self.latency_samples_us, 50),
                    percentile(&self.latency_samples_us, 95),
                    percentile(&self.ttft_samples_us, 50),
                    percentile(&self.ttft_samples_us, 95),
                );
                let have = (self.p50_us, self.p95_us, self.ttft_p50_us, self.ttft_p95_us);
                if want != have {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: format!("percentiles {have:?} do not match the samples {want:?}"),
                    });
                }
                if self.peak_allocated_bytes == Some(0) || self.peak_rss_bytes == Some(0) {
                    return Err(BenchError::Inconsistent {
                        mode,
                        what: "a zero allocation or RSS peak is not a measurement".into(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Nearest-rank percentile: the smallest sample with at least `p` percent of
/// the samples at or below it. `None` for no samples.
pub fn percentile(samples: &[u64], p: u32) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len() as u64;
    let rank = (u64::from(p) * n).div_ceil(100).max(1);
    Some(sorted[(rank - 1) as usize])
}

/// A latency summary over rows that share one cache state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub cache_state: String,
    pub samples: usize,
    pub p50_us: u64,
    pub p95_us: u64,
}

/// Pools the request-latency samples of `rows`. Refuses NOT_RUN or invalid
/// rows and any mix of cache states (ADR 15.1: page-cache hits and device
/// reads are never averaged together).
pub fn summarize(rows: &[&Row]) -> Result<Summary, BenchError> {
    let mut cache: Option<&str> = None;
    let mut samples = Vec::new();
    for row in rows {
        row.check()?;
        if row.status != RowStatus::Ok {
            return Err(BenchError::NothingToSummarize);
        }
        let c = row.cache_state.as_deref().unwrap_or_default();
        match cache {
            None => cache = Some(c),
            Some(first) if first != c => {
                return Err(BenchError::MixedCacheState {
                    first: first.to_string(),
                    other: c.to_string(),
                })
            }
            Some(_) => {}
        }
        samples.extend_from_slice(&row.latency_samples_us);
    }
    let (Some(cache), Some(p50), Some(p95)) =
        (cache, percentile(&samples, 50), percentile(&samples, 95))
    else {
        return Err(BenchError::NothingToSummarize);
    };
    Ok(Summary {
        cache_state: cache.to_string(),
        samples: samples.len(),
        p50_us: p50,
        p95_us: p95,
    })
}

/// The rows every report carries as NOT_RUN (B8, ADR 11 and 16.5).
pub fn required_not_run_rows() -> Vec<Row> {
    vec![
        Row::not_run(
            "native-physical",
            "native_physical",
            "device",
            "Spark-native physical run is blocked on aienos#336 (not authorized); a host run cannot stand in for it",
        ),
        Row::not_run(
            "native-physical",
            "native physical (Pi 5)",
            "device",
            "Pi 5 physical lane is cut P3 (ADR 16), needs Drake's authorization and a free Pi; not run",
        ),
    ]
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub hostname: String,
    pub kernel: String,
    pub arch: String,
    pub cpu_model: String,
    pub cpus: usize,
    /// The process nice value during the run.
    pub nice: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub concurrency: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub rounds: u32,
    pub turn1_max_tokens: u32,
    pub turn2_suffix_tokens: u32,
    pub turn2_max_tokens: u32,
    pub block_size: u32,
    pub chunk_blocks: u32,
    pub dispatch_depth: u32,
}

/// Provenance of one run (docs/plans/public-surface-cleanup.md bar: commit,
/// machine, exact command, raw samples, digest, definitions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportMeta {
    pub measurement_id: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub command: String,
    pub started_utc: String,
    pub platform: String,
    pub machine: Machine,
    pub backend: String,
    pub model: String,
    pub workload: Workload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub schema: String,
    #[serde(flatten)]
    pub meta: ReportMeta,
    pub definitions: BTreeMap<String, String>,
    pub rows: Vec<Row>,
    /// sha256 of the JSON array of every row's (mode, concurrency,
    /// prompt_tokens, latency samples, ttft samples), in row order.
    pub samples_sha256: String,
}

/// The measurement definitions written into every report.
pub fn definitions() -> BTreeMap<String, String> {
    [
        ("p50_us/p95_us", "request latency: submit_work call to the step that emitted the Finished event; nearest-rank percentiles over every measured request of the row"),
        ("ttft_p50_us/ttft_p95_us", "submit_work call to the end of the step that emitted the request's first token"),
        ("decode_tokens_per_s", "sum over requests of (tokens - 1) divided by the summed per-round window from the earliest first token to the last finish"),
        ("peak_allocated_bytes", "kv_pool_allocated_bytes: the KV pool allocation, constant after start (ADR 0003 section 10)"),
        ("peak_holder_bytes", "peak kv_pool_holder_blocks times the block stride, sampled after every step; inside the pool, not added to it"),
        ("peak_staging_bytes", "kv_host_tail_scratch_bytes (one block stride) for modes that use the store, 0 otherwise; buffered reads use the page cache, reported as cache_state"),
        ("peak_rss_bytes", "VmHWM of the bench process after writing 5 to /proc/self/clear_refs at the start of the row"),
        ("queue_depth", "store I/O operations the storage worker may dispatch at once (WorkerConfig.dispatch_depth); 0 when the mode uses no store"),
        ("resident-only", "single_turn: one request per conversation (prompt, turn2_max_tokens), store off; nothing ever leaves memory"),
        ("cold-prefill", "second_turn: turn 1 then turn 2 = turn-1 prompt + turn-1 output + suffix, store off; turn 2 prefills everything"),
        ("restore-from-page-cache", "second_turn with the host store on: turn 1 parks, turn 2 restores the stored prefix (objects written moments before, no cache drop) and prefills the rest; every measured request must commit a restore"),
        ("store bounds", "restore-from-page-cache sets AIEN_KV_STORE_MAX_RESTORE_OPS and AIEN_KV_STORE_MAX_PARK_OPS to concurrency x (chunks of the longest turn + 1), never below the defaults 48 and 64 (run::store_op_bounds); the old park default of 16 refused 8 of 24 parks at 8 conversations (ADR 0003 section 18.4)"),
        ("timing", "token and finish times are taken when the bench drains the completion channel after each spine step (step granularity)"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn samples_digest(rows: &[Row]) -> String {
    let samples: Vec<_> = rows
        .iter()
        .map(|r| {
            (
                &r.mode,
                r.concurrency,
                r.prompt_tokens,
                &r.latency_samples_us,
                &r.ttft_samples_us,
            )
        })
        .collect();
    let bytes = serde_json::to_vec(&samples).unwrap_or_default();
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Builds and checks a report: appends the required NOT_RUN rows, refuses any
/// invalid row, and refuses a measured row on a platform other than `hosted`.
pub fn build_report(meta: ReportMeta, mut rows: Vec<Row>) -> Result<Report, BenchError> {
    for row in &rows {
        row.check()?;
        if row.status == RowStatus::Ok && row.platform != HOSTED {
            return Err(BenchError::Inconsistent {
                mode: row.mode.clone(),
                what: format!(
                    "a host run cannot report measured numbers for platform {:?}",
                    row.platform
                ),
            });
        }
    }
    for want in required_not_run_rows() {
        if !rows
            .iter()
            .any(|r| r.platform == want.platform && r.mode == want.mode)
        {
            rows.push(want);
        }
    }
    let report = Report {
        schema: SCHEMA.to_string(),
        meta,
        definitions: definitions(),
        samples_sha256: samples_digest(&rows),
        rows,
    };
    check_report(&report)?;
    Ok(report)
}

/// Re-checks a report read back from disk: schema, rows, the NOT_RUN physical
/// rows and the samples digest.
pub fn check_report(report: &Report) -> Result<(), BenchError> {
    if report.schema != SCHEMA {
        return Err(BenchError::BadLabel {
            label: "schema",
            value: report.schema.clone(),
        });
    }
    for row in &report.rows {
        row.check()?;
    }
    for want in required_not_run_rows() {
        let ok = report.rows.iter().any(|r| {
            r.platform == want.platform && r.mode == want.mode && r.status == RowStatus::NotRun
        });
        if !ok {
            return Err(BenchError::MissingNotRun {
                platform: want.platform,
            });
        }
    }
    if report.samples_sha256 != samples_digest(&report.rows) {
        return Err(BenchError::Inconsistent {
            mode: "report".into(),
            what: "samples_sha256 does not match the rows".into(),
        });
    }
    Ok(())
}

fn opt<T: fmt::Display>(v: Option<T>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}

/// The Markdown table for BENCH.md, generated from a checked report (ADR 0001
/// decision 3: no hand-edited numbers).
pub fn render_markdown(report: &Report) -> Result<String, BenchError> {
    check_report(report)?;
    let m = &report.meta;
    let mut out = String::new();
    out.push_str(&format!(
        "Measurement `{}`, commit `{}`{}, {} UTC, platform `{}`, backend `{}`, host `{}` ({}, {} CPUs, kernel {}, nice {}).\n\n",
        m.measurement_id,
        m.git_sha,
        if m.git_dirty { " (dirty tree)" } else { "" },
        m.started_utc,
        m.platform,
        m.backend,
        m.machine.hostname,
        m.machine.arch,
        m.machine.cpus,
        m.machine.kernel,
        m.machine.nice,
    ));
    out.push_str(&format!("Command: `{}`\n\n", m.command));
    out.push_str("| mode | status | cache_state | conc | prompt | done/failed | p50 us | p95 us | ttft p50 us | ttft p95 us | decode tok/s | restored tok | pool bytes | holder peak | staging | peak RSS | queue depth |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in &report.rows {
        if r.status == RowStatus::NotRun {
            out.push_str(&format!(
                "| {} ({}) | NOT_RUN | {} | - | - | - | - | - | - | - | - | - | - | - | - | - | - |\n",
                r.mode,
                r.platform,
                opt(r.cache_state.as_deref()),
            ));
            continue;
        }
        out.push_str(&format!(
            "| {} | OK | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.mode,
            opt(r.cache_state.as_deref()),
            opt(r.concurrency),
            opt(r.prompt_tokens),
            opt(r.completed_requests),
            opt(r.failed_requests),
            opt(r.p50_us),
            opt(r.p95_us),
            opt(r.ttft_p50_us),
            opt(r.ttft_p95_us),
            r.decode_tokens_per_s
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".into()),
            opt(r.restored_tokens),
            opt(r.peak_allocated_bytes),
            opt(r.peak_holder_bytes),
            opt(r.peak_staging_bytes),
            opt(r.peak_rss_bytes),
            opt(r.queue_depth),
        ));
    }
    out.push_str("\nNOT_RUN reasons:\n\n");
    for r in report.rows.iter().filter(|r| r.status == RowStatus::NotRun) {
        out.push_str(&format!(
            "- {} ({}): {}\n",
            r.mode,
            r.platform,
            r.reason.as_deref().unwrap_or_default()
        ));
    }
    Ok(out)
}
