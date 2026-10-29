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

//! Independent verifier for campaign C1 (`docs/campaigns/c1/SPEC.md`).
//!
//! Consumes a frozen evidence bundle and the frozen spec directory, and emits PASS/FAIL per
//! check. It links no crate under test: every judgement is rebuilt from raw bundle files.

pub mod bundle;
pub mod ids;
pub mod kv;
pub mod placement;
pub mod report;

use bundle::Manifest;
use kv::{KvCensus, KvEvent, ReportedKvMetrics};
use placement::{OpRecord, PlacementSpec, WitnessEvent};
use report::{CheckResult, Report};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn load_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))
        })
        .collect()
}

pub fn load_toml<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

#[derive(Debug, Deserialize)]
struct KvContract {
    profiles: BTreeMap<String, KvProfile>,
}

#[derive(Debug, Deserialize)]
struct KvProfile {
    num_blocks: u32,
}

/// Pool size comes from the frozen contract, never from the code under test.
fn contract_pool_total(manifest: &Manifest, spec_dir: &Path) -> Result<u32, String> {
    let contract: KvContract = load_toml(&spec_dir.join("kv_contract.toml"))?;
    contract
        .profiles
        .get(&manifest.kv_profile)
        .map(|p| p.num_blocks)
        .ok_or_else(|| format!("kv_profile {} not in kv_contract.toml", manifest.kv_profile))
}

fn check_kv(bundle: &Path, pool_total: u32, report: &mut Report) -> Result<(), String> {
    let events: Vec<KvEvent> = load_jsonl(&bundle.join("kv_allocations.jsonl"))?;
    let census: KvCensus = load_json(&bundle.join("kv_census.json"))?;
    let metrics: ReportedKvMetrics = load_json(&bundle.join("kv_metrics.json"))?;

    let (state, replay_violations) = kv::replay(pool_total, &events);
    report.push(CheckResult::from_violations(
        "KV journal replay (identity, isolation, idempotent release)",
        replay_violations,
    ));
    report.push(CheckResult::from_violations(
        "KV census conservation",
        kv::check_census_conservation(&census),
    ));
    report.push(CheckResult::from_violations(
        "KV replay == raw census",
        kv::check_replay_matches_census(&state, &census),
    ));
    report.push(CheckResult::from_violations(
        "KV replay == reported metrics",
        kv::check_replay_matches_metrics(&state, &metrics),
    ));
    Ok(())
}

fn check_placement(bundle: &Path, spec_dir: &Path) -> Result<Vec<String>, String> {
    let spec: PlacementSpec = load_toml(&spec_dir.join("placement.toml"))?;
    let records: Vec<OpRecord> = load_jsonl(&bundle.join("op_records.jsonl"))?;
    let witness: Vec<WitnessEvent> = load_jsonl(&bundle.join("witness.jsonl"))?;
    Ok(placement::check_placement(&spec, &records, &witness))
}

/// Runs every implemented check. A missing or unreadable input is itself a FAIL.
pub fn verify_bundle(bundle: &Path, spec_dir: &Path) -> Report {
    let mut report = Report::default();

    report.push(CheckResult::from_violations(
        "bundle integrity (sha256sums)",
        bundle::check_sha256sums(bundle),
    ));

    let manifest = load_json::<Manifest>(&bundle.join("manifest.json"));
    report.push(CheckResult::from_violations(
        "manifest binds frozen spec",
        match &manifest {
            Ok(m) => bundle::check_manifest_against_spec(m, spec_dir),
            Err(e) => vec![e.clone()],
        },
    ));

    let pool_total = manifest.and_then(|m| contract_pool_total(&m, spec_dir));
    match pool_total {
        Ok(total) => {
            if let Err(e) = check_kv(bundle, total, &mut report) {
                report.push(CheckResult::from_violations("KV evidence present", vec![e]));
            }
        }
        Err(e) => report.push(CheckResult::from_violations("KV contract profile", vec![e])),
    }

    report.push(CheckResult::from_violations(
        "device placement (dispatch record == witness == placement.toml)",
        check_placement(bundle, spec_dir).unwrap_or_else(|e| vec![e]),
    ));

    report
}
