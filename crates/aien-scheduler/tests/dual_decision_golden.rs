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

//! Before/after proof for the `serve.admit_preempt` tap (DUAL-3a).
//!
//! `build_scheduled_batch` was split into a thin wrapper and a traced body so
//! an observer can be shown the decision. This test uses only APIs that exist
//! on `main` before that change: it runs the pre-registered workload set with
//! no observer and compares the sha256 of the canonical trace of every
//! decision, per scenario, against
//! `tests/fixtures/dual_serve_admit_preempt.golden.json`.
//!
//! The fixture was produced on the base commit (recorded in the evidence
//! receipt) by running this same file there with
//! `DUAL_WRITE_GOLDEN=1 cargo test -p aien-scheduler --test dual_decision_golden`.
//! On the candidate commit the test compares; it never writes unless asked.

mod dual_common;

use dual_common::{build, canonical_json, run, scenarios, ScenarioTrace};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dual_serve_admit_preempt.golden.json")
}

fn all_traces() -> Vec<ScenarioTrace> {
    scenarios()
        .iter()
        .map(|s| {
            let (mut scheduler, kv) = build(s);
            run(s, &mut scheduler, &kv)
        })
        .collect()
}

fn digests(traces: &[ScenarioTrace]) -> BTreeMap<String, String> {
    traces
        .iter()
        .map(|t| {
            let hex = Sha256::digest(canonical_json(t))
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            (t.name.clone(), format!("sha256:{hex}"))
        })
        .collect()
}

#[test]
fn decisions_match_golden_fixture_from_base_commit() {
    let now = digests(&all_traces());
    let path = fixture_path();

    if std::env::var_os("DUAL_WRITE_GOLDEN").is_some() {
        let bytes = serde_json::to_vec_pretty(&now).expect("serialize");
        std::fs::write(&path, &bytes).expect("write golden fixture");
        eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
        return;
    }

    let golden = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden fixture {}: {e}. Produce it on the base commit with DUAL_WRITE_GOLDEN=1",
            path.display()
        )
    });
    let golden: BTreeMap<String, String> = serde_json::from_slice(&golden).expect("parse golden");
    assert_eq!(
        now.keys().collect::<Vec<_>>(),
        golden.keys().collect::<Vec<_>>(),
        "scenario set differs from the golden fixture"
    );
    for (name, digest) in &now {
        assert_eq!(
            digest, &golden[name],
            "scenario {name} decides differently from the base commit"
        );
    }
}

#[test]
fn workload_is_deterministic_run_to_run() {
    let a = all_traces();
    let b = all_traces();
    assert!(canonical_json(&a) == canonical_json(&b));
}

#[test]
fn drained_scenarios_leak_nothing() {
    for s in scenarios() {
        let (mut scheduler, kv) = build(&s);
        let trace = run(&s, &mut scheduler, &kv);
        if s.expect_drained {
            assert_eq!(
                trace.final_kv.allocated_blocks, 0,
                "{}: KV blocks held",
                s.name
            );
            assert_eq!(trace.final_kv.active_tables, 0, "{}: tables left", s.name);
            assert!(trace.final_kv.active_blocks.is_empty(), "{}", s.name);
            assert_eq!(trace.final_scheduler.running, 0, "{}", s.name);
            assert_eq!(trace.final_scheduler.waiting, 0, "{}", s.name);
            assert_eq!(trace.final_scheduler.preempted, 0, "{}", s.name);
            assert_eq!(trace.final_scheduler.arena_active, 0, "{}", s.name);
        } else {
            // The refused request stays queued and never allocated.
            assert_eq!(trace.final_kv.allocated_blocks, 0, "{}", s.name);
            assert!(
                trace.steps.iter().any(|st| st.error.is_some()),
                "{}: expected a refusal",
                s.name
            );
        }
    }
}
