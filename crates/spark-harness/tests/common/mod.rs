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

//! Shared fixtures for the Headstart M2 scheduler tests. Deterministic:
//! a fixed receipt clock and a manual monotonic scheduler clock.
#![allow(dead_code)]

use serde_json::{json, Value};
use spark_harness::completion::CompletionSubmission;
use spark_harness::readiness::{InterfaceSubmission, ProofClaim};
use spark_harness::scheduler::{
    Clock, Effects, Event, EventKind, Outcome, ProvisionalConfig, ResourceLimits, ResourceRequest,
    SchedError, Scheduler,
};
use spark_harness::{HarnessEngine, WorkflowState};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const COMMIT: &str = "1111111111111111111111111111111111111111";

fn fixed_receipt_clock() -> u64 {
    1_760_000_000
}

/// Temp copy of the shipped definitions, removed on drop.
pub struct Fixture {
    pub dir: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("spark-harness-m2-{}", uuid::Uuid::new_v4()));
        let src = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::create_dir_all(dir.join("schemas")).unwrap();
        for f in ["workflows.json", "evals.json"] {
            fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        for entry in fs::read_dir(src.join("schemas")).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), dir.join("schemas").join(entry.file_name())).unwrap();
        }
        Fixture { dir }
    }

    pub fn engine(&self) -> HarnessEngine {
        let mut e = HarnessEngine::load(&self.dir).expect("shipped definitions load");
        e.completion_clock = fixed_receipt_clock;
        e
    }

    pub fn journal(&self) -> PathBuf {
        self.dir.join("sched").join("journal.jsonl")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

pub fn clock(ticks: &Arc<AtomicU64>) -> Clock {
    let t = Arc::clone(ticks);
    Box::new(move || t.load(Ordering::SeqCst))
}

pub fn enabled() -> ProvisionalConfig {
    ProvisionalConfig {
        enabled: true,
        queue_capacity: 8,
        limits: ResourceLimits {
            cpu_slots: 8,
            memory_bytes: 1 << 30,
            max_concurrency: 8,
        },
        lease_ttl: 1_000,
        dedup_window: 64,
    }
}

pub fn req(cpu: u32) -> ResourceRequest {
    ResourceRequest {
        cpu_slots: cpu,
        memory_bytes: 1 << 20,
    }
}

pub fn engineering_fix_artifact(task: &str) -> Value {
    json!({
        "bug": format!("Parser rejects valid token sequence in {task}"),
        "reproduction_steps": ["Run parser test"],
        "failure_evidence": "thread panicked at line 42",
        "regression_test": "tests::test_parser",
        "diff_summary": "Fixed token parser logic",
        "retest_evidence": "All tests passing cleanly"
    })
}

/// One scheduler plus the harness engine whose store it verifies against.
pub struct H {
    pub fx: Fixture,
    pub engine: HarnessEngine,
    pub ticks: Arc<AtomicU64>,
    pub sched: Scheduler,
    n: u64,
}

impl H {
    pub fn new(cfg: ProvisionalConfig) -> Self {
        let fx = Fixture::new();
        let engine = fx.engine();
        let ticks = Arc::new(AtomicU64::new(0));
        let sched = Scheduler::new(cfg, &engine.completion_store, clock(&ticks)).unwrap();
        H {
            fx,
            engine,
            ticks,
            sched,
            n: 0,
        }
    }

    pub fn journaled(cfg: ProvisionalConfig) -> Self {
        let fx = Fixture::new();
        let engine = fx.engine();
        let ticks = Arc::new(AtomicU64::new(0));
        let (unaudited, report) =
            Scheduler::open(cfg, &engine.completion_store, &fx.journal(), clock(&ticks)).unwrap();
        assert_eq!(report.entries, 0);
        let (sched, revoked) = unaudited.audit().unwrap();
        assert!(revoked.is_empty());
        H {
            fx,
            engine,
            ticks,
            sched,
            n: 0,
        }
    }

    /// Drop the scheduler (closing the journal), replay it from disk and
    /// audit it. Returns the torn bytes dropped and the tasks the audit
    /// revoked.
    pub fn reopen(&mut self, cfg: ProvisionalConfig) -> Result<(usize, Vec<String>), SchedError> {
        let placeholder = Scheduler::new(
            cfg.clone(),
            &self.engine.completion_store,
            clock(&self.ticks),
        )?;
        drop(std::mem::replace(&mut self.sched, placeholder));
        let (unaudited, report) = Scheduler::open(
            cfg,
            &self.engine.completion_store,
            &self.fx.journal(),
            clock(&self.ticks),
        )?;
        let (sched, revoked) = unaudited.audit()?;
        self.sched = sched;
        Ok((report.torn_bytes, revoked))
    }

    /// Simulate a process restart: the new process's clock starts at 0.
    pub fn restart(&mut self, cfg: ProvisionalConfig) -> Result<(usize, Vec<String>), SchedError> {
        self.ticks.store(0, Ordering::SeqCst);
        self.reopen(cfg)
    }

    pub fn advance(&self, d: u64) {
        self.ticks.fetch_add(d, Ordering::SeqCst);
    }

    pub fn next_id(&mut self) -> String {
        self.n += 1;
        format!("ev-{}", self.n)
    }

    /// Deliver with a fresh id; check invariants either way.
    pub fn send(&mut self, kind: EventKind) -> Result<Outcome, SchedError> {
        let id = self.next_id();
        let r = self.sched.deliver(Event { id, kind });
        self.sched.check_invariants().unwrap();
        r
    }

    pub fn ok(&mut self, kind: EventKind) -> Effects {
        self.advance(10);
        match self.send(kind.clone()) {
            Ok(Outcome::Applied(fx)) => fx,
            other => panic!("event {kind:?} not applied: {other:?}"),
        }
    }

    pub fn register(&mut self, task: &str, cpu: u32, producers: &[&str]) -> Effects {
        self.ok(EventKind::RegisterTask {
            task: task.into(),
            resources: req(cpu),
            producers: producers.iter().map(|p| p.to_string()).collect(),
        })
    }

    pub fn edge(&mut self, producer: &str, consumer: &str) {
        self.ok(EventKind::AddDependency {
            producer: producer.into(),
            consumer: consumer.into(),
        });
    }

    pub fn generation(&self, task: &str) -> u64 {
        self.sched.generation(task).unwrap()
    }

    /// Mint an interface receipt in the harness store (not delivered).
    pub fn mint_interface(&self, task: &str, version: u32) -> ProofClaim {
        self.engine
            .publish_interface(&InterfaceSubmission {
                task_id: task.into(),
                generation: self.generation(task),
                source_commit: COMMIT.into(),
                interface: json!({ "task": task, "api_version": version }),
            })
            .unwrap()
            .claim()
    }

    pub fn publish(&mut self, task: &str, version: u32) -> (ProofClaim, Effects) {
        let claim = self.mint_interface(task, version);
        let generation = self.generation(task);
        let fx = self.ok(EventKind::InterfacePublished {
            task: task.into(),
            generation,
            claim: claim.clone(),
        });
        (claim, fx)
    }

    /// Run the M1 verified completion for the task's current generation.
    pub fn mint_completion(&self, task: &str) -> ProofClaim {
        let mut wf = WorkflowState {
            id: format!("wf-{task}"),
            workflow_type: "engineering_fix".into(),
            current_state: "review".into(),
            history: Vec::new(),
            created_at: "2026-10-15T00:00:00Z".into(),
        };
        let rec = self
            .engine
            .complete_verified_workflow(
                &mut wf,
                &CompletionSubmission {
                    task_id: task.into(),
                    generation: self.generation(task),
                    source_commit: COMMIT.into(),
                    artifact: engineering_fix_artifact(task),
                },
            )
            .unwrap();
        ProofClaim {
            receipt_id: rec.receipt_id,
            commit: rec.source_commit,
            digest: rec.artifact_digest,
        }
    }

    pub fn complete(&mut self, task: &str) -> ProofClaim {
        let claim = self.mint_completion(task);
        let generation = self.generation(task);
        self.ok(EventKind::CompletionVerified {
            task: task.into(),
            generation,
            claim: claim.clone(),
        });
        claim
    }

    pub fn work_done(&mut self, task: &str) -> Effects {
        let generation = self.generation(task);
        self.ok(EventKind::WorkDone {
            task: task.into(),
            generation,
            result_digest: format!("result-of-{task}"),
        })
    }

    pub fn finalize(&mut self, task: &str) -> Effects {
        let generation = self.generation(task);
        self.ok(EventKind::Finalize {
            task: task.into(),
            generation,
        })
    }

    /// Work done, finalize, then the M1 completion: the task becomes final
    /// for its consumers.
    pub fn finish(&mut self, task: &str) -> Effects {
        self.work_done(task);
        self.finalize(task);
        let generation = self.generation(task);
        let claim = self.mint_completion(task);
        self.ok(EventKind::CompletionVerified {
            task: task.into(),
            generation,
            claim,
        })
    }
}
