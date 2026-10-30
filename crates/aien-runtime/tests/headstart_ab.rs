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

//! Headstart M6 (#538): the opt-in A/B qualification target for provisional
//! preparation (crate::provisional, M3) on the M2 scheduler (spark-harness
//! `scheduler`). Compiled only with `--features provisional-prep`; every test
//! here is `#[ignore]`, so the default suites never run it.
//!
//! One driver, two arms. Both arms run the same driver, the same worker
//! threads, the same `ResourceLimits::max_concurrency`, a journaled
//! `PrepAdapter` over a fresh harness store and a fresh workspace. The only
//! difference is `ProvisionalConfig::enabled`: `off` is the real sequential
//! dispatch of M2 (`Core::eligible` needs every producer final), `on` lets a
//! consumer start once its producers published a verified interface. No arm
//! sets a prep hook, a sleep or any configured cost. The work units are the
//! real code paths:
//! - producer work: the M1 interface receipt (`publish_interface`) and the M1
//!   verified workflow completion (`complete_verified_workflow`);
//! - preparation: `PrepAdapter::prepare`, the whitelisted compose plan
//!   preview over a seeded workspace;
//! - commit: the fenced `PrepAdapter::submit_approved` (the real approved
//!   proposal path and World commit) in a LINKED build. The stub build
//!   refuses every commit (provisional_prep_test.rs `committed_or_stub`), so
//!   a STUB run passes the real fence around a no-op call instead and counts
//!   no World commit: STUB lines are overhead and correctness checks only,
//!   never a result.
//!
//! Shapes: `chain` (a narrow critical chain, the representative mechanism
//! workload), `parallel` (independent tasks, an overhead and regression
//! check: no gain is expected) and `failure` (a producer fails, another
//! changes generation, while consumers may be preparing; descendants of the
//! failed producer must never commit, and no commit may rest on the old
//! generation).
//!
//! Output: one JSON line per sample, prefixed `HS-M6-AB `, also appended to
//! `$HS_M6_OUT` when set. `scripts/bench/headstart-m6-ab.sh` runs one sample
//! per process under `quietlock hold`. Ready queue depth is sampled with the
//! existing read-only `Scheduler::queue` after every event this driver
//! delivers, and blocked time comes from the existing
//! `Scheduler::timing_trace` (journaled event times): no scheduler code was
//! changed for this target.
#![cfg(feature = "provisional-prep")]
// The per-sample `json!` record is larger than the default macro recursion limit.
#![recursion_limit = "256"]

#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::approved::{ApprovedProposal, ProposerHook};
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::provisional::{
    Delivered, FenceError, FenceProof, PrepAdapter, PrepError, ReadinessBinding,
};
use aien_runtime::{ComposeBridge, ComposeProposer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use spark_harness::completion::CompletionSubmission;
use spark_harness::execution::TaskState;
use spark_harness::readiness::InterfaceSubmission;
use spark_harness::scheduler::{
    monotonic_clock, CommitResult, Effects, Event, EventKind, Outcome, ProvisionalConfig,
    ResourceLimits, ResourceRequest, SchedError,
};
use spark_harness::{HarnessEngine, WorkflowState};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const SCHEMA: &str = "headstart-m6-ab/v1";
const COMMIT: &str = "1111111111111111111111111111111111111111";
/// A sample that has not settled after this long is reported as a deadlock.
const SAMPLE_DEADLINE: Duration = Duration::from_secs(240);
/// How long a failing or regenerating root waits for its consumers to
/// prepare before it fails anyway (only the `on` arm dispatches them early).
const FAIL_POINT_WAIT: Duration = Duration::from_secs(10);
const DEFAULT_THREADS: usize = 4;
const INBOX_FILES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Off,
    On,
}

impl Arm {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Arm::Off),
            "on" => Some(Arm::On),
            _ => None,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Arm::Off => "off",
            Arm::On => "on",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Chain,
    Parallel,
    Failure,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::Chain, Shape::Parallel, Shape::Failure];
    fn parse(s: &str) -> Option<Self> {
        Shape::ALL.into_iter().find(|x| x.as_str() == s)
    }
    fn as_str(self) -> &'static str {
        match self {
            Shape::Chain => "chain",
            Shape::Parallel => "parallel",
            Shape::Failure => "failure",
        }
    }
    fn class(self) -> &'static str {
        match self {
            Shape::Chain => "representative-mechanism",
            Shape::Parallel => "overhead-regression",
            Shape::Failure => "failure-cancellation",
        }
    }
}

/// The DAG of one shape: (task, producers) in registration order, and the
/// roots the failure shape fails or moves to a new generation.
struct Plan {
    tasks: Vec<(String, Vec<String>)>,
    fail_root: Option<String>,
    regen_root: Option<String>,
}

fn plan(shape: Shape) -> Plan {
    let t = |s: &str, p: &[&str]| (s.to_string(), p.iter().map(|x| x.to_string()).collect());
    match shape {
        Shape::Chain => {
            let names: Vec<String> = (0..6).map(|i| format!("t{i:02}")).collect();
            let mut tasks = Vec::new();
            for (i, n) in names.iter().enumerate() {
                let p = if i == 0 {
                    vec![]
                } else {
                    vec![names[i - 1].clone()]
                };
                tasks.push((n.clone(), p));
            }
            Plan {
                tasks,
                fail_root: None,
                regen_root: None,
            }
        }
        Shape::Parallel => Plan {
            tasks: (0..8).map(|i| (format!("p{i:02}"), vec![])).collect(),
            fail_root: None,
            regen_root: None,
        },
        Shape::Failure => Plan {
            tasks: vec![
                t("f", &[]),
                t("g", &[]),
                t("fc0", &["f"]),
                t("fc1", &["f"]),
                t("fcc0", &["fc0"]),
                t("gc0", &["g"]),
                t("gc1", &["g"]),
            ],
            fail_root: Some("f".into()),
            regen_root: Some("g".into()),
        },
    }
}

fn cfg(arm: Arm, threads: usize) -> ProvisionalConfig {
    ProvisionalConfig {
        enabled: arm == Arm::On,
        queue_capacity: 64,
        limits: ResourceLimits {
            cpu_slots: 64,
            memory_bytes: 1 << 34,
            max_concurrency: threads as u32,
        },
        // Nanosecond ticks (monotonic_clock): ten minutes.
        lease_ttl: 600_000_000_000,
        dedup_window: 4096,
    }
}

fn copy_definitions(dir: &Path) {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../spark-harness");
    std::fs::create_dir_all(dir.join("schemas")).unwrap();
    for f in ["workflows.json", "evals.json"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    for e in std::fs::read_dir(src.join("schemas")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dir.join("schemas").join(e.file_name())).unwrap();
    }
}

/// Deterministic input notes the plan preview reads (`inbox/`), inside the
/// compose input bounds (bridge.rs COMPOSE_INPUTS_MAX_FILES 16,
/// COMPOSE_INPUTS_MAX_BYTES 16384).
fn seed_workspace(ws: &Path) {
    let inbox = ws.join("inbox");
    std::fs::create_dir_all(&inbox).unwrap();
    for i in 0..INBOX_FILES {
        let mut s = String::new();
        for line in 0..24 {
            s.push_str(&format!(
                "note {i} line {line}: the parser accepts token sequence {}\n",
                (i * 31 + line * 7) % 97
            ));
        }
        std::fs::write(inbox.join(format!("note-{i}.txt")), s).unwrap();
    }
}

fn goal(task: &str) -> String {
    format!("Read the text files in the inbox folder and write NOTES-{task}.md with a summary")
}

fn artifact(task: &str, generation: u64) -> Value {
    json!({
        "bug": format!("Parser rejects valid token sequence in {task} generation {generation}"),
        "reproduction_steps": ["Run parser test"],
        "failure_evidence": "thread panicked at line 42",
        "regression_test": "tests::test_parser",
        "diff_summary": "Fixed token parser logic",
        "retest_evidence": "All tests passing cleanly"
    })
}

fn model_never_called() -> ComposeProposer {
    Arc::new(|_: &str, _: Duration| panic!("the model proposer ran on the approved path"))
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn interplane_sha(path: &str, content: &str) -> String {
    let s = format!(
        "{{\"content\":{},\"path\":{}}}",
        serde_json::to_string(content).unwrap(),
        serde_json::to_string(path).unwrap()
    );
    sha(s.as_bytes())
}

/// Process CPU time and peak RSS (getrusage RUSAGE_SELF: every thread of
/// this process; ru_maxrss is the process high-water mark in KiB).
fn rusage() -> (f64, f64, i64) {
    // SAFETY: getrusage only writes the struct we pass.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    assert_eq!(rc, 0, "getrusage failed");
    let ms = |t: libc::timeval| t.tv_sec as f64 * 1000.0 + t.tv_usec as f64 / 1000.0;
    (ms(ru.ru_utime), ms(ru.ru_stime), ru.ru_maxrss)
}

fn loadavg_1m() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The real World commit (LINKED) or nothing (STUB).
struct Compose {
    bridge: Arc<ComposeBridge>,
    hook: ProposerHook,
    desk: DeskKey,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<(String, u64)>,
    queued: BTreeSet<(String, u64)>,
    active: usize,
    changes: u64,
    stop: bool,
}

#[derive(Default)]
struct Metrics {
    prep_ns: u128,
    producer_ns: u128,
    commit_ns: u128,
    preparations: Vec<(String, u64, bool)>,
    prep_refused: u64,
    dispatches_abandoned: u64,
    revoked: u64,
    aborted: u64,
    deferred: u64,
    depth_samples: Vec<usize>,
    interface_receipts: Vec<Value>,
    completion_receipts: Vec<Value>,
    commits: Vec<Value>,
    committed: Vec<(String, u64)>,
    world_commits: u64,
    commit_refused: Vec<Value>,
    commit_waits: u64,
    fail_points: Vec<Value>,
    driver_errors: Vec<String>,
}

struct Run<'a> {
    adapter: Arc<PrepAdapter>,
    engine: &'a HarnessEngine,
    ws: PathBuf,
    compose: Option<Compose>,
    plan: &'a Plan,
    q: Mutex<Queue>,
    cv: Condvar,
    ids: AtomicU64,
    regen_done: AtomicBool,
    m: Mutex<Metrics>,
    deadline: Instant,
}

enum Stop {
    /// The sample deadline passed (reported as a deadlock).
    Deadline,
}

impl Run<'_> {
    fn err(&self, e: String) {
        self.m.lock().unwrap().driver_errors.push(e);
    }

    /// A step failed: a driver error while the task is still at
    /// `generation`, an abandoned dispatch once M2 moved it on.
    fn bail(&self, task: &str, generation: u64, e: String) {
        if self.current(task, generation) {
            self.err(e);
        } else {
            self.m.lock().unwrap().dispatches_abandoned += 1;
        }
    }

    fn notify(&self) {
        self.q.lock().unwrap().changes += 1;
        self.cv.notify_all();
    }

    fn enqueue(&self, task: &str) {
        let Some(generation) = self.adapter.generation(task) else {
            return;
        };
        let mut q = self.q.lock().unwrap();
        if q.queued.insert((task.to_string(), generation)) {
            q.jobs.push_back((task.to_string(), generation));
        }
    }

    /// Queue a job for every `Provisional` task without one (a dispatch
    /// made inside the fence, whose effects the adapter does not return).
    fn sweep(&self) {
        for (t, _) in &self.plan.tasks {
            if self.adapter.state(t) == Some(TaskState::Provisional) {
                self.enqueue(t);
            }
        }
    }

    fn after(&self, fx: &Effects) {
        for t in &fx.dispatched {
            self.enqueue(t);
        }
        let depth = self.adapter.inspect(|s| s.queue().len());
        {
            let mut m = self.m.lock().unwrap();
            m.revoked += fx.revoked.len() as u64;
            m.aborted += fx.aborted.len() as u64;
            m.depth_samples.push(depth);
        }
        self.notify();
    }

    fn send(&self, kind: EventKind) -> Result<Effects, String> {
        let id = format!("m6-{}", self.ids.fetch_add(1, Ordering::SeqCst));
        let what = format!("{kind:?}");
        match self.adapter.deliver(Event { id, kind }) {
            Ok(Delivered::Applied(Outcome::Applied(fx))) => {
                self.after(&fx);
                Ok(fx)
            }
            Ok(Delivered::Applied(Outcome::Duplicate)) => Err(format!("duplicate: {what}")),
            Ok(Delivered::Deferred { committing, .. }) => {
                self.m.lock().unwrap().deferred += 1;
                Err(format!("deferred behind {committing}: {what}"))
            }
            Err(e) => Err(format!("{e}: {what}")),
        }
    }

    /// Wait until `cond` returns Some. Re-checked after every change any
    /// thread reports, so no wakeup is lost; bounded by the sample deadline.
    fn wait_until<T>(
        &self,
        until: Instant,
        mut cond: impl FnMut() -> Option<T>,
    ) -> Result<T, Stop> {
        loop {
            let seen = self.q.lock().unwrap().changes;
            if let Some(v) = cond() {
                return Ok(v);
            }
            let mut q = self.q.lock().unwrap();
            while q.changes == seen && !q.stop {
                let now = Instant::now();
                if now >= self.deadline {
                    q.stop = true;
                    self.cv.notify_all();
                    return Err(Stop::Deadline);
                }
                if now >= until {
                    return Err(Stop::Deadline);
                }
                let wait = until.min(self.deadline) - now;
                q = self
                    .cv
                    .wait_timeout(q, wait.min(Duration::from_millis(500)))
                    .unwrap()
                    .0;
            }
            if q.stop {
                return Err(Stop::Deadline);
            }
        }
    }

    fn current(&self, task: &str, generation: u64) -> bool {
        self.adapter.generation(task) == Some(generation)
    }

    fn producers(&self, task: &str) -> &[String] {
        &self
            .plan
            .tasks
            .iter()
            .find(|(t, _)| t == task)
            .expect("planned task")
            .1
    }

    fn consumers(&self, task: &str) -> Vec<String> {
        self.plan
            .tasks
            .iter()
            .filter(|(_, p)| p.iter().any(|x| x == task))
            .map(|(t, _)| t.clone())
            .collect()
    }

    fn worker(&self) {
        loop {
            let job = {
                let mut q = self.q.lock().unwrap();
                loop {
                    if q.stop {
                        break None;
                    }
                    if let Some(j) = q.jobs.pop_front() {
                        q.active += 1;
                        break Some(j);
                    }
                    if q.active == 0 {
                        break None;
                    }
                    if Instant::now() >= self.deadline {
                        q.stop = true;
                        break None;
                    }
                    q = self
                        .cv
                        .wait_timeout(q, Duration::from_millis(500))
                        .unwrap()
                        .0;
                }
            };
            let Some((task, generation)) = job else {
                self.cv.notify_all();
                return;
            };
            self.job(&task, generation);
            self.sweep();
            self.q.lock().unwrap().active -= 1;
            self.notify();
        }
    }

    /// One dispatch of `task` at `generation`, start to commit. Abandoned
    /// (and counted) as soon as the task leaves that generation.
    fn job(&self, task: &str, generation: u64) {
        if !self.current(task, generation)
            || self.adapter.state(task) != Some(TaskState::Provisional)
        {
            self.m.lock().unwrap().dispatches_abandoned += 1;
            return;
        }
        // Preparation: the whitelisted plan preview.
        let t = Instant::now();
        let r = self
            .adapter
            .prepare(task, generation, &goal(task), &self.ws);
        let d = t.elapsed().as_nanos();
        {
            let mut m = self.m.lock().unwrap();
            m.prep_ns += d;
            match &r {
                Ok(p) => m
                    .preparations
                    .push((task.to_string(), generation, p.is_provisional())),
                Err(_) => m.prep_refused += 1,
            }
        }
        match r {
            Ok(_) => {}
            Err(PrepError::StaleGeneration { .. } | PrepError::Revoked { .. }) => {
                self.m.lock().unwrap().dispatches_abandoned += 1;
                return;
            }
            Err(PrepError::Scheduler(_)) if !self.current(task, generation) => {
                self.m.lock().unwrap().dispatches_abandoned += 1;
                return;
            }
            Err(e) => return self.err(format!("prepare {task} g{generation}: {e}")),
        }
        // Producer work, part 1: the M1 interface receipt.
        let t = Instant::now();
        let rec = self.engine.publish_interface(&InterfaceSubmission {
            task_id: task.into(),
            generation,
            source_commit: COMMIT.into(),
            interface: json!({ "task": task, "api_version": generation }),
        });
        self.m.lock().unwrap().producer_ns += t.elapsed().as_nanos();
        let claim = match rec {
            Ok(r) => r.claim(),
            Err(e) => return self.bail(task, generation, format!("publish_interface {task}: {e}")),
        };
        self.m.lock().unwrap().interface_receipts.push(json!({
            "task": task, "generation": generation, "receipt_id": claim.receipt_id,
        }));
        if let Err(e) = self.send(EventKind::InterfacePublished {
            task: task.into(),
            generation,
            claim,
        }) {
            return self.bail(task, generation, e);
        }
        if self.plan.fail_root.as_deref() == Some(task) {
            self.fail_point(task, generation, true);
            return;
        }
        if self.plan.regen_root.as_deref() == Some(task)
            && !self.regen_done.swap(true, Ordering::SeqCst)
        {
            self.fail_point(task, generation, false);
            return;
        }
        if let Err(e) = self.send(EventKind::WorkDone {
            task: task.into(),
            generation,
            result_digest: format!("result-of-{task}-g{generation}"),
        }) {
            return self.bail(task, generation, e);
        }
        // Finalize needs every producer final (M1 completion accepted).
        let producers = self.producers(task).to_vec();
        let ready = self.wait_until(self.deadline, || {
            if !self.current(task, generation) {
                return Some(false);
            }
            let all = self.adapter.inspect(|s| {
                producers.iter().all(|p| {
                    s.task(p)
                        .is_some_and(|t| t.state.is_final() && t.completion.is_some())
                })
            });
            all.then_some(true)
        });
        match ready {
            Ok(true) => {}
            Ok(false) => {
                self.m.lock().unwrap().dispatches_abandoned += 1;
                return;
            }
            Err(Stop::Deadline) => return,
        }
        if let Err(e) = self.send(EventKind::Finalize {
            task: task.into(),
            generation,
        }) {
            if self.current(task, generation) {
                self.err(e);
            } else {
                self.m.lock().unwrap().dispatches_abandoned += 1;
            }
            return;
        }
        // Producer work, part 2: the M1 verified workflow completion.
        let mut wf = WorkflowState {
            id: format!("wf-{task}-g{generation}"),
            workflow_type: "engineering_fix".into(),
            current_state: "review".into(),
            history: Vec::new(),
            created_at: "2026-10-16T00:00:00Z".into(),
        };
        let t = Instant::now();
        let rec = self.engine.complete_verified_workflow(
            &mut wf,
            &CompletionSubmission {
                task_id: task.into(),
                generation,
                source_commit: COMMIT.into(),
                artifact: artifact(task, generation),
            },
        );
        self.m.lock().unwrap().producer_ns += t.elapsed().as_nanos();
        let rec = match rec {
            Ok(r) => r,
            Err(e) => {
                return self.bail(
                    task,
                    generation,
                    format!("complete_verified_workflow {task}: {e}"),
                )
            }
        };
        self.m.lock().unwrap().completion_receipts.push(json!({
            "task": task, "generation": generation, "receipt_id": rec.receipt_id,
            "artifact_digest": rec.artifact_digest,
        }));
        if let Err(e) = self.send(EventKind::CompletionVerified {
            task: task.into(),
            generation,
            claim: spark_harness::readiness::ProofClaim {
                receipt_id: rec.receipt_id,
                commit: rec.source_commit,
                digest: rec.artifact_digest,
            },
        }) {
            return self.bail(task, generation, e);
        }
        self.commit(task, generation);
    }

    /// The failure shape: wait (bounded) until every consumer prepared or
    /// is `Blocked` and not queued for dispatch, then fail the producer or move it to a new
    /// generation. Only the `on` arm can have consumers preparing here.
    fn fail_point(&self, task: &str, generation: u64, fail: bool) {
        let consumers = self.consumers(task);
        let until = Instant::now() + FAIL_POINT_WAIT;
        let reached = self
            .wait_until(until, || {
                consumers
                    .iter()
                    .all(|c| {
                        self.adapter.prepared(c).is_some()
                            || self.adapter.inspect(|s| {
                                s.task(c)
                                    .is_some_and(|t| t.state == TaskState::Blocked && !t.queued)
                            })
                    })
                    .then_some(())
            })
            .is_ok();
        let prepared: Vec<&String> = consumers
            .iter()
            .filter(|c| self.adapter.prepared(c).is_some())
            .collect();
        self.m.lock().unwrap().fail_points.push(json!({
            "task": task, "generation": generation,
            "action": if fail { "producer_failed" } else { "new_generation" },
            "consumers_prepared": prepared, "wait_completed": reached,
        }));
        let r = if fail {
            self.send(EventKind::ProducerFailed {
                task: task.into(),
                generation,
                reason: "M6 failure shape".into(),
            })
        } else {
            self.send(EventKind::NewGeneration {
                task: task.into(),
                generation: generation + 1,
            })
        };
        if let Err(e) = r {
            self.err(e);
        }
    }

    fn approval(&self, c: &Compose, task: &str, generation: u64) -> ApprovedProposal {
        let path = format!("NOTES-{task}.md");
        let content =
            format!("{task} generation {generation}: approved after the producers were final\n");
        let request = format!("req-{task}-g{generation}");
        let mut p = ApprovedProposal {
            request_id: request.clone(),
            trace_id: format!("trace-{request}"),
            approval_id: format!("appr-{request}"),
            approver: "interplane-host".into(),
            path: path.clone(),
            content: content.clone(),
            approved_proposal_sha256: interplane_sha(&path, &content),
            content_sha256: sha(content.as_bytes()),
            approval_mac: String::new(),
            requirements: Some(String::new()),
            requirements_mac: String::new(),
            requirements_base: None,
        };
        c.desk.seal(&mut p, &self.ws);
        p
    }

    /// The fenced commit. LINKED: the real approved-proposal path. STUB: the
    /// same fence around a no-op (no World commit). A commit refused because
    /// another one runs in the same chain waits for a change and retries.
    fn commit(&self, task: &str, generation: u64) {
        let ws = self.ws.display().to_string();
        loop {
            // Read before submitting: a commit that ends between the refusal and
            // this read would otherwise leave the retry waiting on an unrelated change.
            let seen = self.q.lock().unwrap().changes;
            let t = Instant::now();
            let r: Result<(FenceProof, CommitResult, Option<Value>), FenceError> =
                match &self.compose {
                    Some(c) => {
                        let p = self.approval(c, task, generation);
                        self.adapter
                            .submit_approved(task, generation, &c.hook, &p, &ws)
                            .map(|f| {
                                let rec = match &f.outcome {
                                    Ok(rep) => json!({
                                        "state": rep.state, "request_id": rep.request_id,
                                        "trace_id": rep.trace_id, "replay_claim": rep.replay_claim,
                                        "approved_grant": rep.approved_grant,
                                        "compose_proposal_sha256": rep.compose_proposal_sha256,
                                        "cx_promotion": rep.task.as_ref().map(|t| t.cx_promotion),
                                        "cx_evidence": rep.task.as_ref().map(|t| t.cx_evidence),
                                        "world_committed": rep.task.as_ref().map(|t| t.committed),
                                    }),
                                    Err(e) => json!({
                                        "refused_by": e.refused_by, "name": e.name,
                                        "trace_id": e.trace_id, "replay_claim": e.replay_claim,
                                    }),
                                };
                                (f.proof, f.result, Some(rec))
                            })
                    }
                    None => self
                        .adapter
                        .fenced_for_tests(
                            task,
                            generation,
                            |_| (),
                            |_| CommitResult::Committed {
                                detail: "STUB no-op: no World commit".into(),
                            },
                        )
                        .map(|f| (f.proof, f.result, None)),
                };
            let d = t.elapsed().as_nanos();
            self.m.lock().unwrap().commit_ns += d;
            match r {
                Ok((proof, result, rec)) => {
                    self.record_commit(task, generation, &proof, &result, rec);
                    self.notify();
                    return;
                }
                Err(FenceError::Scheduler(SchedError::CommitInProgress { .. })) => {
                    self.m.lock().unwrap().commit_waits += 1;
                    let waited = self.wait_until(self.deadline, || {
                        (self.q.lock().unwrap().changes != seen).then_some(())
                    });
                    if waited.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    if self.current(task, generation) {
                        self.err(format!("fence {task} g{generation}: {e}"));
                    } else {
                        self.m.lock().unwrap().dispatches_abandoned += 1;
                    }
                    return;
                }
            }
        }
    }

    fn record_commit(
        &self,
        task: &str,
        generation: u64,
        proof: &FenceProof,
        result: &CommitResult,
        rec: Option<Value>,
    ) {
        let binding = |b: &ReadinessBinding| {
            json!({
                "producer": b.producer_task, "generation": b.producer_generation,
                "level": format!("{:?}", b.level), "receipt_id": b.receipt_id,
            })
        };
        let mut m = self.m.lock().unwrap();
        let world = rec
            .as_ref()
            .is_some_and(|r| r["state"] == "COMMITTED" && r["world_committed"] == true);
        if world {
            m.world_commits += 1;
        }
        let entry = json!({
            "task": task, "generation": generation,
            "result": result,
            "own_completion": binding(&proof.completion),
            "producers": proof.producers.iter().map(binding).collect::<Vec<_>>(),
            "commit_receipt": rec,
        });
        if matches!(result, CommitResult::Committed { .. }) {
            m.committed.push((task.to_string(), generation));
            m.commits.push(entry);
        } else {
            m.commit_refused.push(entry);
        }
    }
}

/// One sample: a fresh store, workspace, journal and (LINKED) compose home;
/// the shape's DAG driven to quiescence on `threads` workers.
fn run_sample(arm: Arm, shape: Shape, threads: usize) -> Value {
    let linked = aien_omega_compose::LINKED;
    let tmp = tempfile::tempdir().unwrap();
    let hd = tmp.path().join("harness");
    copy_definitions(&hd);
    let ws = tmp.path().join("ws");
    seed_workspace(&ws);
    let engine = HarnessEngine::load(&hd).expect("shipped definitions load");
    let (adapter, revoked) = PrepAdapter::open(
        cfg(arm, threads),
        &engine.completion_store,
        &tmp.path().join("journal.jsonl"),
        monotonic_clock(),
    )
    .unwrap();
    assert!(revoked.is_empty());
    let compose = linked.then(|| {
        let dir = tmp.path().join("compose");
        let desk = DeskKey::create(&desk_key_path(&dir)).unwrap();
        let bridge = Arc::new(ComposeBridge::new(
            dir,
            model_never_called(),
            "test:model-never-called",
        ));
        let hook = ProposerHook::new(bridge.clone());
        Compose { bridge, hook, desk }
    });
    let plan = plan(shape);
    let run = Run {
        adapter: Arc::new(adapter),
        engine: &engine,
        ws: ws.clone(),
        compose,
        plan: &plan,
        q: Mutex::new(Queue::default()),
        cv: Condvar::new(),
        ids: AtomicU64::new(0),
        regen_done: AtomicBool::new(false),
        m: Mutex::new(Metrics::default()),
        deadline: Instant::now() + SAMPLE_DEADLINE,
    };
    let load = loadavg_1m();
    let (u0, s0, _) = rusage();
    let t0 = Instant::now();
    for (task, producers) in &plan.tasks {
        if let Err(e) = run.send(EventKind::RegisterTask {
            task: task.clone(),
            resources: ResourceRequest {
                cpu_slots: 1,
                memory_bytes: 1 << 20,
            },
            producers: producers.clone(),
        }) {
            run.err(e);
        }
    }
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| run.worker());
        }
    });
    let wall = t0.elapsed();
    let (u1, s1, rss) = rusage();
    let deadlock = run.q.lock().unwrap().stop;
    if let Some(c) = &run.compose {
        c.bridge.close();
    }
    let invariants = run.adapter.inspect(|s| s.check_invariants());
    let trace = run.adapter.inspect(|s| s.timing_trace());
    let final_states: BTreeMap<String, &'static str> = plan
        .tasks
        .iter()
        .map(|(t, _)| {
            (
                t.clone(),
                run.adapter.state(t).map_or("missing", |s| s.as_str()),
            )
        })
        .collect();
    let final_generations: BTreeMap<String, u64> = plan
        .tasks
        .iter()
        .map(|(t, _)| (t.clone(), run.adapter.generation(t).unwrap_or(0)))
        .collect();
    let sum = |k: &str| -> f64 {
        trace["tasks"]
            .as_array()
            .map(|a| a.iter().filter_map(|t| t[k].as_u64()).sum::<u64>())
            .unwrap_or(0) as f64
            / 1e6
    };
    let m = run.m.into_inner().unwrap();
    let wall_ms = ms(wall);
    let prep_ms = m.prep_ns as f64 / 1e6;
    let producer_ms = m.producer_ns as f64 / 1e6;
    let commit_ms = m.commit_ns as f64 / 1e6;
    // Preparation results made at a generation the task did not commit at.
    let discarded = m
        .preparations
        .iter()
        .filter(|(t, g, _)| !m.committed.iter().any(|(ct, cg)| ct == t && cg == g))
        .count();
    let depth_peak = m.depth_samples.iter().copied().max().unwrap_or(0);
    let depth_mean = if m.depth_samples.is_empty() {
        0.0
    } else {
        m.depth_samples.iter().sum::<usize>() as f64 / m.depth_samples.len() as f64
    };
    let line = json!({
        "schema": SCHEMA,
        "arm": arm.as_str(),
        "shape": shape.as_str(),
        "shape_class": shape.class(),
        "build": if linked { "LINKED" } else { "STUB" },
        "result_class": if linked { "linked-sample" } else { "STUB: overhead and correctness only, not a result" },
        "threads": threads,
        "max_concurrency": threads,
        "tasks": plan.tasks.len(),
        "wall_ms": wall_ms,
        "prep_ms": prep_ms,
        "producer_ms": producer_ms,
        "commit_ms": commit_ms,
        "ceiling_estimate": prep_ms.min(producer_ms) / wall_ms,
        "cpu_user_ms": u1 - u0,
        "cpu_sys_ms": s1 - s0,
        "max_rss_kb_process": rss,
        "queue_depth_peak": depth_peak,
        "queue_depth_mean": depth_mean,
        "queue_depth_samples": m.depth_samples.len(),
        "blocked_ms_total": sum("blocked"),
        "provisional_ms_total": sum("provisional"),
        "awaiting_final_proof_ms_total": sum("awaiting_final_proof"),
        "preparations": m.preparations.len(),
        "preparations_provisional": m.preparations.iter().filter(|p| p.2).count(),
        "prep_refused": m.prep_refused,
        "discards": {
            "prep_results_discarded": discarded,
            "dispatches_abandoned": m.dispatches_abandoned,
            "revoked": m.revoked,
            "aborted": m.aborted,
        },
        "deferred_events": m.deferred,
        "commit_waits_same_chain": m.commit_waits,
        "commits": m.committed.len(),
        "world_commits": m.world_commits,
        "commit_records": m.commits,
        "commit_refused": m.commit_refused,
        "interface_receipts": m.interface_receipts,
        "completion_receipts": m.completion_receipts,
        "fail_points": m.fail_points,
        "final_states": final_states,
        "final_generations": final_generations,
        "invariants": invariants.err(),
        "deadlock": deadlock,
        "driver_errors": m.driver_errors,
        "loadavg_1m_start": load,
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    });
    check_sample(&line, &plan);
    line
}

/// The correctness gates every sample must pass, in both arms and builds.
fn check_sample(v: &Value, plan: &Plan) {
    assert_eq!(v["deadlock"], false, "deadlock: {v}");
    assert!(v["invariants"].is_null(), "M2 invariants: {v}");
    assert_eq!(
        v["driver_errors"].as_array().map(|a| a.len()),
        Some(0),
        "{v}"
    );
    assert_eq!(v["deferred_events"], 0, "{v}");
    let commits = v["commit_records"].as_array().unwrap();
    // No double commit: one commit per task at most.
    let mut seen = BTreeSet::new();
    for c in commits {
        let t = c["task"].as_str().unwrap();
        assert!(seen.insert(t.to_string()), "double commit of {t}: {v}");
        // Every commit rests on final proofs only.
        assert_eq!(c["own_completion"]["level"], "FinalVerified", "{v}");
        for p in c["producers"].as_array().unwrap() {
            assert_eq!(p["level"], "FinalVerified", "{v}");
            let pt = p["producer"].as_str().unwrap();
            assert_eq!(
                p["generation"], v["final_generations"][pt],
                "commit of {t} rests on a stale generation of {pt}: {v}"
            );
        }
    }
    if v["build"] == "LINKED" {
        assert_eq!(
            v["world_commits"].as_u64(),
            Some(commits.len() as u64),
            "every fenced commit is a World commit: {v}"
        );
    } else {
        assert_eq!(v["world_commits"], 0, "{v}");
    }
    if v["arm"] == "off" {
        assert_eq!(
            v["preparations_provisional"], 0,
            "flag off never prepares on an interface: {v}"
        );
    }
    match v["shape"].as_str().unwrap() {
        "chain" | "parallel" => {
            assert_eq!(commits.len(), plan.tasks.len(), "every task commits: {v}");
            for (t, _) in &plan.tasks {
                assert_eq!(v["final_states"][t], "committed", "{v}");
            }
        }
        "failure" => {
            for t in ["fc0", "fc1", "fcc0", "f"] {
                assert!(
                    !seen.contains(t),
                    "{t} committed after its producer failed: {v}"
                );
            }
            assert_eq!(v["final_states"]["f"], "aborted", "{v}");
            for t in ["g", "gc0", "gc1"] {
                assert_eq!(v["final_states"][t], "committed", "{v}");
            }
            assert_eq!(v["final_generations"]["g"], 2, "{v}");
        }
        other => panic!("unknown shape {other}"),
    }
}

fn emit(mut line: Value, label: &str, tag: &str, sample: u64) {
    line["label"] = json!(label);
    line["tag"] = json!(tag);
    line["sample"] = json!(sample);
    let text = serde_json::to_string(&line).unwrap();
    println!("HS-M6-AB {text}");
    if let Ok(out) = std::env::var("HS_M6_OUT") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .unwrap_or_else(|e| panic!("HS_M6_OUT {out}: {e}"));
        writeln!(f, "{text}").unwrap();
    }
}

fn threads() -> usize {
    std::env::var("HS_M6_THREADS")
        .ok()
        .map(|s| s.parse().expect("HS_M6_THREADS is a number"))
        .unwrap_or(DEFAULT_THREADS)
}

/// One sample, chosen by the environment (one per process, so the RSS
/// high-water mark is per sample): HS_M6_ARM (off|on), HS_M6_SHAPE
/// (chain|parallel|failure), HS_M6_SAMPLE (index), HS_M6_TAG (A, B, AA1,
/// AA2), HS_M6_THREADS (default 4), HS_M6_OUT (JSON lines file).
#[test]
#[ignore = "A/B qualification sample: run by scripts/bench/headstart-m6-ab.sh"]
fn headstart_ab_sample() {
    let _home = home_guard::home_slot();
    let arm = std::env::var("HS_M6_ARM").expect("HS_M6_ARM (off|on)");
    let shape = std::env::var("HS_M6_SHAPE").expect("HS_M6_SHAPE (chain|parallel|failure)");
    let arm = Arm::parse(&arm).expect("HS_M6_ARM is off or on");
    let shape = Shape::parse(&shape).expect("HS_M6_SHAPE is chain, parallel or failure");
    let sample = std::env::var("HS_M6_SAMPLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let tag = std::env::var("HS_M6_TAG").unwrap_or_default();
    emit(run_sample(arm, shape, threads()), "SAMPLE", &tag, sample);
}

/// One run per arm and shape in one process, labelled SMOKE: proves the
/// target works and that both arms end in the same task states (no state
/// divergence). Not a measurement.
#[test]
#[ignore = "A/B harness smoke run (not a measurement)"]
fn headstart_ab_smoke() {
    let _home = home_guard::home_slot();
    for shape in Shape::ALL {
        let off = run_sample(Arm::Off, shape, threads());
        let on = run_sample(Arm::On, shape, threads());
        assert_eq!(
            off["final_states"],
            on["final_states"],
            "flag on and off diverged on {}",
            shape.as_str()
        );
        assert_eq!(off["commits"], on["commits"], "{}", shape.as_str());
        emit(off, "SMOKE", "", 0);
        emit(on, "SMOKE", "", 0);
    }
}
