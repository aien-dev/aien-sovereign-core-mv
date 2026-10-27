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

//! Guards that no event sequence reaches (defense in depth), tested on a
//! hand-built state.

use super::core::{Binding, BindingLevel, TaskEntry};
use super::{ProvisionalConfig, ResourceRequest, SchedError, Scheduler};
use crate::execution::TaskState;
use crate::readiness::{OperationClass, ProofClaim};
use std::path::Path;

fn req() -> ResourceRequest {
    ResourceRequest {
        cpu_slots: 1,
        memory_bytes: 1,
    }
}

/// Producer `p` at generation 2; consumer `c` in `state`, still bound to
/// generation 1. Revocation always clears such a binding, so only a
/// corrupted state can carry one.
fn stale_binding(state: TaskState) -> Scheduler {
    let cfg = ProvisionalConfig {
        enabled: true,
        ..ProvisionalConfig::default()
    };
    let store = Path::new("/nonexistent/spark-harness-m2-unit-store");
    let mut s = Scheduler::new(cfg, store, Box::new(|| 100)).unwrap();
    s.core.graph.add_node("p").unwrap();
    s.core.graph.add_node("c").unwrap();
    s.core.graph.add_edge("p", "c").unwrap();
    let mut p = TaskEntry::new(req(), 0);
    p.generation = 2;
    let mut c = TaskEntry::new(req(), 0);
    c.state = state;
    c.lease_expires_at = Some(1_000);
    c.bindings.insert(
        "p".to_string(),
        Binding {
            generation: 1,
            level: BindingLevel::Interface,
            claim: ProofClaim {
                receipt_id: "r".to_string(),
                commit: "c".to_string(),
                digest: "d".to_string(),
            },
        },
    );
    s.core.tasks.insert("p".to_string(), p);
    s.core.tasks.insert("c".to_string(), c);
    s
}

#[test]
fn binding_to_an_old_producer_generation_is_refused() {
    let s = stale_binding(TaskState::Provisional);
    let r = s.may_start("c", OperationClass::ReadOnlyAnalysis);
    assert!(matches!(r, Err(SchedError::StaleBinding { .. })), "{r:?}");
    let s = stale_binding(TaskState::AwaitingFinalProof);
    let r = s.may_finalize("c");
    assert!(matches!(r, Err(SchedError::StaleBinding { .. })), "{r:?}");
}

/// Measures the whole-state copy `deliver` makes per event, next to the
/// full in-memory deliver (copy + apply + dispatch scan) and one journal
/// append with fsync. Run with
/// `cargo test --release -p spark-harness --lib -- --ignored --nocapture`
/// (task count from `M2_COST_TASKS`, default 2000).
#[test]
#[ignore]
fn deliver_cost_is_small_next_to_the_journal_fsync() {
    use super::{Event, EventKind};
    use std::time::Instant;
    let tasks: usize = std::env::var("M2_COST_TASKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000);
    const N: u32 = 200;
    let cfg = ProvisionalConfig {
        enabled: true,
        ..ProvisionalConfig::default()
    };
    let store = Path::new("/nonexistent/spark-harness-m2-unit-store");
    let mut s = Scheduler::new(cfg, store, Box::new(|| 1)).unwrap();
    for i in 0..tasks {
        let producers = if i == 0 {
            vec![]
        } else {
            vec!["t0".to_string()]
        };
        let kind = EventKind::RegisterTask {
            task: format!("t{i}"),
            resources: req(),
            producers,
        };
        s.deliver(Event {
            id: format!("reg-{i}"),
            kind,
        })
        .unwrap();
    }
    let start = Instant::now();
    for _ in 0..N {
        std::hint::black_box(s.core.clone());
    }
    let copy = start.elapsed() / N;
    let start = Instant::now();
    for i in 0..N {
        let ev = Event {
            id: format!("tick-{i}"),
            kind: EventKind::Tick,
        };
        s.deliver(ev).unwrap();
    }
    let deliver = start.elapsed() / N;
    let dir = std::env::temp_dir().join(format!("spark-harness-m2-cost-{}", uuid::Uuid::new_v4()));
    let header = serde_json::json!({ "format": "cost" });
    let mut j = crate::journal::Journal::open::<serde_json::Value>(&dir.join("j.jsonl"), &header)
        .unwrap()
        .journal;
    let line = serde_json::json!({ "seq": 1, "id": "tick", "at": 1, "kind": { "type": "tick" } });
    let start = Instant::now();
    for _ in 0..N {
        j.append(&line).unwrap();
    }
    let fsync = start.elapsed() / N;
    drop(j);
    let _ = std::fs::remove_dir_all(&dir);
    println!(
        "M2_DELIVER_COST tasks={tasks} state_copy={copy:?} in_memory_deliver={deliver:?} \
         journal_append_fsync={fsync:?}"
    );
}
