//! Headstart M3 (#537): provisional engineering preparation and the final
//! commit fence (crate::provisional). Compiled only with
//! `--features provisional-prep`; without it this binary holds no tests.
//!
//! The M2 scheduler and the M1/M2 harness store are real (spark-harness,
//! shipped workflow and eval definitions). The World commit is the real
//! approved-proposal path; tests that need it to commit run fully only in a
//! linked build (cfg compose_linked) and check the stub refusal otherwise.
//! Timing lines start with `M3-TIMING` (run with `--nocapture`).
#![cfg(feature = "provisional-prep")]

#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::approved::{
    ApprovedComposeReport, ApprovedProposal, ApprovedRefusal, ProposerHook,
};
use aien_runtime::approved_auth::{desk_key_path, AuthorizeBinding, DeskKey};
use aien_runtime::control::{ControlCommand, ControlResponse, DeskProof};
use aien_runtime::effects::MintRequest;
use aien_runtime::provisional::{
    classify_approved, classify_authorize, deferred_journal_path, DeliverError, Delivered,
    FenceError, FenceKey, PrepAdapter, PrepError, PrepOperation, DEFERRED_CAPACITY,
    DEFERRED_MAX_BYTES,
};
use aien_runtime::{ComposeBridge, ComposeProposer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use spark_harness::completion::CompletionSubmission;
use spark_harness::execution::TaskState;
use spark_harness::readiness::{InterfaceSubmission, OperationClass, ProofClaim, ReadinessLevel};
use spark_harness::scheduler::{
    Clock, CommitResult, Effects, Event, EventKind, Outcome, ProvisionalConfig, ResourceLimits,
    ResourceRequest, SchedError,
};
use spark_harness::{HarnessEngine, WorkflowState};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const COMMIT: &str = "1111111111111111111111111111111111111111";
const PATH: &str = "NOTES.md";
const CONTENT: &str = "approved after the producers were final\n";
const GOAL: &str = "Write NOTES.md: one line about the parser interface";

// ---------- harness store and scheduler ----------

fn fixed_receipt_clock() -> u64 {
    1_760_000_000
}

fn cfg(enabled: bool, max_concurrency: u32) -> ProvisionalConfig {
    ProvisionalConfig {
        enabled,
        queue_capacity: 8,
        limits: ResourceLimits {
            cpu_slots: 8,
            memory_bytes: 1 << 30,
            max_concurrency,
        },
        lease_ttl: 1_000_000,
        dedup_window: 64,
    }
}

fn clock(t: &Arc<AtomicU64>) -> Clock {
    let t = Arc::clone(t);
    Box::new(move || t.load(Ordering::SeqCst))
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

fn artifact(task: &str) -> Value {
    json!({
        "bug": format!("Parser rejects valid token sequence in {task}"),
        "reproduction_steps": ["Run parser test"],
        "failure_evidence": "thread panicked at line 42",
        "regression_test": "tests::test_parser",
        "diff_summary": "Fixed token parser logic",
        "retest_evidence": "All tests passing cleanly"
    })
}

/// One adapter, its harness store, a compose home and a workspace, all in
/// one temp dir: `<tmp>/harness`, `<tmp>/compose`, `<tmp>/ws`.
struct H {
    tmp: tempfile::TempDir,
    engine: HarnessEngine,
    ticks: Arc<AtomicU64>,
    adapter: Arc<PrepAdapter>,
    n: u64,
}

impl H {
    fn new(c: ProvisionalConfig) -> Self {
        Self::build(c, false)
    }

    fn journaled(c: ProvisionalConfig) -> Self {
        Self::build(c, true)
    }

    fn build(c: ProvisionalConfig, journaled: bool) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let hd = tmp.path().join("harness");
        copy_definitions(&hd);
        std::fs::create_dir_all(tmp.path().join("ws")).unwrap();
        let mut engine = HarnessEngine::load(&hd).expect("shipped definitions load");
        engine.completion_clock = fixed_receipt_clock;
        let ticks = Arc::new(AtomicU64::new(0));
        let adapter = if journaled {
            let (a, revoked) = PrepAdapter::open(
                c,
                &engine.completion_store,
                &tmp.path().join("journal.jsonl"),
                clock(&ticks),
            )
            .unwrap();
            assert!(revoked.is_empty());
            a
        } else {
            PrepAdapter::new(c, &engine.completion_store, clock(&ticks)).unwrap()
        };
        H {
            tmp,
            engine,
            ticks,
            adapter: Arc::new(adapter),
            n: 0,
        }
    }

    /// A daemon restart: drop the adapter, replay its journal, audit.
    fn restart(&mut self, c: ProvisionalConfig) -> Vec<String> {
        let placeholder =
            PrepAdapter::new(c.clone(), &self.engine.completion_store, clock(&self.ticks)).unwrap();
        drop(std::mem::replace(&mut self.adapter, Arc::new(placeholder)));
        self.ticks.store(0, Ordering::SeqCst);
        let (a, revoked) = PrepAdapter::open(
            c,
            &self.engine.completion_store,
            &self.tmp.path().join("journal.jsonl"),
            clock(&self.ticks),
        )
        .unwrap();
        self.adapter = Arc::new(a);
        revoked
    }

    fn ws(&self) -> PathBuf {
        self.tmp.path().join("ws")
    }

    fn deliver(&mut self, kind: EventKind) -> Result<Delivered, DeliverError> {
        self.n += 1;
        self.ticks.fetch_add(10, Ordering::SeqCst);
        let r = self.adapter.deliver(Event {
            id: format!("ev-{}", self.n),
            kind,
        });
        self.adapter.inspect(|s| s.check_invariants()).unwrap();
        r
    }

    /// An event no commit holds back: applied, or refused by M2.
    fn send(&mut self, kind: EventKind) -> Result<Outcome, SchedError> {
        match self.deliver(kind) {
            Ok(Delivered::Applied(o)) => Ok(o),
            Err(DeliverError::Scheduler(e)) => Err(e),
            other => panic!("not applied or refused by M2: {other:?}"),
        }
    }

    fn ok(&mut self, kind: EventKind) -> Effects {
        match self.send(kind.clone()) {
            Ok(Outcome::Applied(fx)) => fx,
            other => panic!("{kind:?} not applied: {other:?}"),
        }
    }

    fn gen(&self, task: &str) -> u64 {
        self.adapter.generation(task).unwrap()
    }

    fn state(&self, task: &str) -> TaskState {
        self.adapter.state(task).unwrap()
    }

    fn register(&mut self, task: &str, producers: &[&str]) -> Effects {
        self.ok(EventKind::RegisterTask {
            task: task.into(),
            resources: ResourceRequest {
                cpu_slots: 1,
                memory_bytes: 1 << 20,
            },
            producers: producers.iter().map(|p| p.to_string()).collect(),
        })
    }

    fn mint_interface(&self, task: &str, version: u32) -> ProofClaim {
        self.engine
            .publish_interface(&InterfaceSubmission {
                task_id: task.into(),
                generation: self.gen(task),
                source_commit: COMMIT.into(),
                interface: json!({ "task": task, "api_version": version }),
            })
            .unwrap()
            .claim()
    }

    fn publish(&mut self, task: &str, version: u32) -> (ProofClaim, Effects) {
        let claim = self.mint_interface(task, version);
        let generation = self.gen(task);
        let fx = self.ok(EventKind::InterfacePublished {
            task: task.into(),
            generation,
            claim: claim.clone(),
        });
        (claim, fx)
    }

    fn mint_completion(&self, task: &str) -> ProofClaim {
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
                    generation: self.gen(task),
                    source_commit: COMMIT.into(),
                    artifact: artifact(task),
                },
            )
            .unwrap();
        ProofClaim {
            receipt_id: rec.receipt_id,
            commit: rec.source_commit,
            digest: rec.artifact_digest,
        }
    }

    fn work_done(&mut self, task: &str) {
        let generation = self.gen(task);
        self.ok(EventKind::WorkDone {
            task: task.into(),
            generation,
            result_digest: format!("result-of-{task}"),
        });
    }

    fn finalize(&mut self, task: &str) {
        let generation = self.gen(task);
        self.ok(EventKind::Finalize {
            task: task.into(),
            generation,
        });
    }

    fn complete(&mut self, task: &str) -> ProofClaim {
        let claim = self.mint_completion(task);
        let generation = self.gen(task);
        self.ok(EventKind::CompletionVerified {
            task: task.into(),
            generation,
            claim: claim.clone(),
        });
        claim
    }

    /// Work done, finalize, M1 completion: the task is final and committable.
    fn finish(&mut self, task: &str) {
        self.work_done(task);
        self.finalize(task);
        self.complete(task);
        assert_eq!(self.state(task), TaskState::Committable);
    }

    fn prepare(
        &self,
        task: &str,
    ) -> Result<aien_runtime::provisional::PreparationResult, PrepError> {
        self.adapter.prepare(task, self.gen(task), GOAL, &self.ws())
    }
}

// ---------- compose home and approvals ----------

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

struct Compose {
    bridge: Arc<ComposeBridge>,
    hook: ProposerHook,
    ws: String,
}

impl Compose {
    fn new(h: &H) -> Self {
        DeskKey::create(&desk_key_path(&h.tmp.path().join("compose"))).unwrap();
        Self::reopen(h)
    }

    /// A fresh bridge on the same compose home and desk key (a daemon restart).
    fn reopen(h: &H) -> Self {
        let bridge = Arc::new(ComposeBridge::new(
            h.tmp.path().join("compose"),
            model_never_called(),
            "test:model-never-called",
        ));
        let hook = ProposerHook::new(bridge.clone());
        Compose {
            bridge,
            hook,
            ws: h.ws().display().to_string(),
        }
    }

    /// An approval sealed with this home's desk key.
    fn approval(&self, request: &str) -> ApprovedProposal {
        let mut p = ApprovedProposal {
            request_id: request.into(),
            trace_id: format!("trace-{request}"),
            approval_id: format!("appr-{request}"),
            approver: "interplane-host".into(),
            path: PATH.into(),
            content: CONTENT.into(),
            approved_proposal_sha256: interplane_sha(PATH, CONTENT),
            content_sha256: sha(CONTENT.as_bytes()),
            approval_mac: String::new(),
            requirements: Some(String::new()),
            requirements_mac: String::new(),
            requirements_base: None,
        };
        DeskKey::load(&desk_key_path(self.bridge.dir()))
            .unwrap()
            .seal(&mut p, Path::new(&self.ws));
        p
    }
}

type Submit = Result<ApprovedComposeReport, Box<ApprovedRefusal>>;

/// The canonical path's answer for a valid approval: COMMITTED when linked,
/// the stub refusal otherwise. Returns the report when linked.
fn committed_or_stub(r: Submit) -> Option<ApprovedComposeReport> {
    if !aien_omega_compose::LINKED {
        let e = r.expect_err("stub build must refuse");
        assert!(
            e.refused_by == "REPLAY_REFUSED" || e.name == "ComposeError",
            "{e}"
        );
        return None;
    }
    let r = r.unwrap_or_else(|e| panic!("want COMMITTED, got {e}"));
    assert_eq!(r.state, "COMMITTED");
    let t = r.task.as_ref().expect("a compose run");
    assert!(t.committed && t.cx_promotion != 0, "{t:?}");
    Some(r)
}

/// Every file under `dir` with its sha256, for "nothing was written" checks.
fn tree(dir: &Path) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.clone(), sha(&std::fs::read(&p).unwrap_or_default()));
            }
        }
    }
    out
}

/// Classifier for `fenced_for_tests`: the call committed.
fn committed<T>(_: &T) -> CommitResult {
    CommitResult::Committed {
        detail: "test".into(),
    }
}

fn fence_err<T: std::fmt::Debug>(r: Result<T, FenceError>) -> FenceError {
    match r {
        Err(e) => e,
        Ok(v) => panic!("the fence let a call through: {v:?}"),
    }
}

// ---------- tests ----------

#[test]
fn the_one_whitelisted_operation_is_read_only_analysis() {
    let op = PrepOperation::ComposePlanPreview;
    assert_eq!(op.class(), OperationClass::ReadOnlyAnalysis);
    for c in OperationClass::ALL {
        if c != OperationClass::ReadOnlyAnalysis {
            assert_ne!(op.class(), c);
        }
    }
}

struct Timing {
    critical_ms: f64,
    overlap_ms: f64,
    prep_ms: f64,
    prep_started_before_producer_final: bool,
}

const PRODUCER_WORK: Duration = Duration::from_millis(60);
const PREP_LOAD: Duration = Duration::from_millis(40);

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Producer A (60 ms of simulated work on another thread) and consumer B,
/// whose preparation is the real whitelisted operation plus 40 ms of
/// simulated preparation load (the test hook, outside the adapter mutex).
/// Ends with B's fenced World commit through the approved path.
fn two_stage(enabled: bool) -> Timing {
    let mut h = H::new(cfg(enabled, 4));
    let c = Compose::new(&h);
    h.adapter
        .set_prep_hook(Some(Arc::new(|_: &str| std::thread::sleep(PREP_LOAD))));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let t0 = Instant::now();
    let producer = std::thread::spawn(move || {
        std::thread::sleep(PRODUCER_WORK);
        Instant::now()
    });
    let mut prep = None;
    if enabled {
        assert_eq!(h.state("b"), TaskState::Provisional);
        let s = Instant::now();
        let r = h.prepare("b").expect("early preparation");
        assert!(r.is_provisional(), "bound to an interface, not a final");
        prep = Some((s, Instant::now()));
    } else {
        assert_eq!(h.state("b"), TaskState::Blocked);
        assert!(matches!(
            h.prepare("b"),
            Err(PrepError::Scheduler(SchedError::WrongState { .. }))
        ));
    }
    let producer_end = producer.join().unwrap();
    h.finish("a");
    if !enabled {
        assert_eq!(h.state("b"), TaskState::Provisional);
        let s = Instant::now();
        let r = h.prepare("b").expect("sequential preparation");
        assert!(!r.is_provisional());
        prep = Some((s, Instant::now()));
    }
    h.finish("b");
    let _home = home_guard::home_slot();
    let g = h.gen("b");
    let out = h
        .adapter
        .submit_approved("b", g, &c.hook, &c.approval("req-two"), &c.ws)
        .expect("fence passes");
    committed_or_stub(out.outcome);
    let end = Instant::now();
    let (ps, pe) = prep.unwrap();
    let overlap = pe.min(producer_end).saturating_duration_since(ps.max(t0));
    Timing {
        critical_ms: ms(end - t0),
        overlap_ms: ms(overlap),
        prep_ms: ms(pe - ps),
        prep_started_before_producer_final: ps < producer_end,
    }
}

#[test]
fn two_stage_overlap_measured_against_the_sequential_fallback() {
    let on = two_stage(true);
    let off = two_stage(false);
    for (name, t) in [("enabled", &on), ("disabled", &off)] {
        eprintln!(
            "M3-TIMING two_stage {name}: critical_path_ms={:.2} prep_ms={:.2} overlap_ms={:.2} producer_work_ms={} prep_load_ms={} linked={}",
            t.critical_ms,
            t.prep_ms,
            t.overlap_ms,
            PRODUCER_WORK.as_millis(),
            PREP_LOAD.as_millis(),
            aien_omega_compose::LINKED
        );
    }
    assert!(on.prep_started_before_producer_final && on.overlap_ms > 0.0);
    assert!(!off.prep_started_before_producer_final);
    assert_eq!(off.overlap_ms, 0.0, "disabled means no early start");
}

#[test]
fn three_stage_chain_prepares_two_consumers_while_the_root_runs() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let c = Compose::new(&h);
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("c", &["b"]);
    let (ia, _) = h.publish("a", 1);
    assert_eq!(h.state("b"), TaskState::Provisional);
    let (ib, _) = h.publish("b", 1);
    assert_eq!(h.state("c"), TaskState::Provisional);
    let t0 = Instant::now();
    let rb = h.prepare("b").unwrap();
    let rc = h.prepare("c").unwrap();
    eprintln!(
        "M3-TIMING three_stage: two preparations while the root is not final took {:.3} ms",
        ms(t0.elapsed())
    );
    assert_eq!(h.state("a"), TaskState::Provisional, "root still running");
    // Each result is bound to its producer's interface receipt.
    let b0 = &rb.bindings()[0];
    assert_eq!(
        (b0.producer_task.as_str(), b0.receipt_id.as_str()),
        ("a", ia.receipt_id.as_str())
    );
    assert_eq!(
        (b0.source_commit.as_str(), b0.artifact_digest.as_str()),
        (COMMIT, ia.digest.as_str())
    );
    assert_eq!(b0.producer_generation, 1);
    assert_eq!(b0.level, ReadinessLevel::InterfaceVerified);
    assert_eq!(rc.bindings()[0].receipt_id, ib.receipt_id);
    assert_eq!(rc.operation(), PrepOperation::ComposePlanPreview);
    assert_eq!(rc.class(), OperationClass::ReadOnlyAnalysis);
    assert_eq!(rc.plan_sha256().len(), 64);
    // The fence refuses C until the whole chain is final.
    let gc = h.gen("c");
    let p = c.approval("req-three");
    assert!(matches!(
        fence_err(h.adapter.submit_approved("c", gc, &c.hook, &p, &c.ws)),
        FenceError::NotCommittable { .. }
    ));
    h.finish("a");
    h.finish("b");
    h.work_done("c");
    h.finalize("c");
    // Committable but without its own M1 completion: still refused.
    assert!(matches!(
        fence_err(h.adapter.submit_approved("c", gc, &c.hook, &p, &c.ws)),
        FenceError::NoCompletion(_)
    ));
    h.complete("c");
    let out = h
        .adapter
        .submit_approved("c", gc, &c.hook, &p, &c.ws)
        .expect("fence passes once A, B and C are final");
    assert_eq!(out.proof.producers.len(), 1);
    assert_eq!(out.proof.producers[0].level, ReadinessLevel::FinalVerified);
    assert_eq!(out.proof.completion.level, ReadinessLevel::FinalVerified);
    committed_or_stub(out.outcome);
}

#[test]
fn failed_producer_discards_the_dependent_result() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.prepare("b").unwrap();
    assert!(h.adapter.prepared("b").is_some());
    h.ok(EventKind::ProducerFailed {
        task: "a".into(),
        generation: 1,
        reason: "tests failed".into(),
    });
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Blocked, 2));
    assert!(h.adapter.prepared("b").is_none(), "result discarded");
    assert!(matches!(
        h.adapter.prepare("b", 1, GOAL, &h.ws()),
        Err(PrepError::StaleGeneration { .. })
    ));
    assert!(matches!(
        fence_err(h.adapter.fenced_for_tests("b", 1, |_| (), committed)),
        FenceError::StaleGeneration { .. }
    ));
}

#[test]
fn completed_producer_upgrades_the_binding_to_final() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    assert!(h.prepare("b").unwrap().is_provisional());
    h.finish("a");
    let r = h.prepare("b").unwrap();
    assert!(!r.is_provisional());
    assert_eq!(r.bindings()[0].level, ReadinessLevel::FinalVerified);
}

#[test]
fn changed_interface_revokes_and_rebinds_to_the_new_receipt() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    let (v1, _) = h.publish("a", 1);
    assert_eq!(
        h.prepare("b").unwrap().bindings()[0].receipt_id,
        v1.receipt_id
    );
    let (v2, fx) = h.publish("a", 2);
    assert_ne!(v1.receipt_id, v2.receipt_id);
    assert!(fx.revoked.contains(&"b".to_string()));
    assert!(h.adapter.prepared("b").is_none());
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Provisional, 2));
    let r = h.prepare("b").unwrap();
    assert_eq!(
        (r.generation(), r.bindings()[0].receipt_id.as_str()),
        (2, v2.receipt_id.as_str())
    );
}

#[test]
fn new_producer_generation_makes_old_generations_stale() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.prepare("b").unwrap();
    h.ok(EventKind::NewGeneration {
        task: "a".into(),
        generation: 2,
    });
    assert!(h.adapter.prepared("b").is_none());
    assert_eq!(h.gen("b"), 2);
    for g in [1, 3] {
        assert!(matches!(
            h.adapter.prepare("b", g, GOAL, &h.ws()),
            Err(PrepError::StaleGeneration { .. })
        ));
    }
}

#[test]
fn cancellation_discards_results_and_refuses_preparation() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.prepare("b").unwrap();
    h.ok(EventKind::Cancel {
        task: "a".into(),
        generation: 1,
    });
    assert_eq!(h.state("b"), TaskState::Aborted);
    assert!(h.adapter.prepared("b").is_none());
    assert!(matches!(
        h.prepare("b"),
        Err(PrepError::Scheduler(SchedError::WrongState { .. }))
    ));
}

#[test]
fn duplicate_event_is_a_noop_and_keeps_the_result() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    let claim = h.mint_interface("a", 1);
    let ev = Event {
        id: "publish-a".into(),
        kind: EventKind::InterfacePublished {
            task: "a".into(),
            generation: 1,
            claim,
        },
    };
    assert!(matches!(
        h.adapter.deliver(ev.clone()),
        Ok(Delivered::Applied(Outcome::Applied(_)))
    ));
    let r = h.prepare("b").unwrap();
    assert_eq!(
        h.adapter.deliver(ev).unwrap(),
        Delivered::Applied(Outcome::Duplicate)
    );
    assert_eq!(h.adapter.prepared("b"), Some(r));
    assert_eq!(h.gen("b"), 1);
}

#[test]
fn invalid_receipts_are_refused_by_the_event_gate_and_the_fence() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    // A claim naming no stored receipt never reaches the scheduler.
    let forged = ProofClaim {
        receipt_id: "ab".repeat(32),
        commit: COMMIT.into(),
        digest: "cd".repeat(32),
    };
    assert!(matches!(
        h.send(EventKind::InterfacePublished {
            task: "a".into(),
            generation: 1,
            claim: forged,
        }),
        Err(SchedError::Proof { .. })
    ));
    assert_eq!(h.state("b"), TaskState::Blocked);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    // Tamper with B's own M1 receipt in the store after it was accepted.
    let rid = h
        .adapter
        .inspect(|s| s.task("b").unwrap().completion.clone().unwrap().receipt_id);
    let file = h
        .engine
        .completion_store
        .join(aien_proof::evidence::RECEIPT_DIR)
        .join(format!("{rid}.json"));
    let original = std::fs::read(&file).unwrap();
    let text = String::from_utf8(original.clone()).unwrap();
    std::fs::write(&file, text.replacen("spark-harness", "spark-harnesz", 1)).unwrap();
    let mut ran = false;
    assert!(matches!(
        fence_err(
            h.adapter
                .fenced_for_tests("b", 1, |_| ran = true, committed)
        ),
        FenceError::CompletionInvalid { .. }
    ));
    assert!(!ran);
    std::fs::write(&file, &original).unwrap();
    // Tamper with the producer's receipt instead: M2 may_finalize refuses.
    let ra = h
        .adapter
        .inspect(|s| s.task("a").unwrap().completion.clone().unwrap().receipt_id);
    let fa = file.with_file_name(format!("{ra}.json"));
    let oa = std::fs::read(&fa).unwrap();
    std::fs::write(
        &fa,
        String::from_utf8(oa.clone())
            .unwrap()
            .replacen("spark-harness", "spark-harnesz", 1),
    )
    .unwrap();
    assert!(matches!(
        fence_err(
            h.adapter
                .fenced_for_tests("b", 1, |_| ran = true, committed)
        ),
        FenceError::Scheduler(SchedError::Proof { .. })
    ));
    assert!(!ran);
    std::fs::write(&fa, &oa).unwrap();
    h.adapter
        .fenced_for_tests("b", 1, |_| ran = true, committed)
        .unwrap();
    assert!(ran);
}

#[test]
fn early_fence_alone_commits_nothing_and_writes_nothing() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let c = Compose::new(&h);
    let mut ws_file = std::fs::File::create(h.ws().join("parser.md")).unwrap();
    std::io::Write::write_all(&mut ws_file, b"# Parser\n").unwrap();
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let p = c.approval("req-early");
    let compose_dir = c.bridge.dir().to_path_buf();
    let before = (tree(&h.ws()), tree(&compose_dir));
    // The early gate passes and preparation runs ...
    let r = h.prepare("b").unwrap();
    assert!(r.is_provisional());
    // ... but no commit or grant path is reachable from it.
    let g = h.gen("b");
    {
        assert!(matches!(
            fence_err(h.adapter.submit_approved("b", g, &c.hook, &p, &c.ws)),
            FenceError::NotCommittable {
                state: TaskState::Provisional,
                ..
            }
        ));
        let req = MintRequest {
            cx_promotion: 1,
            proposal_sha256: sha(b"x"),
            workspace: c.ws.clone(),
            approver: "drake".into(),
            constraints: vec![],
        };
        assert!(matches!(
            fence_err(h.adapter.authorize("b", g, &c.bridge, &req, None, None)),
            FenceError::NotCommittable {
                state: TaskState::Provisional,
                ..
            }
        ));
    }
    h.work_done("b");
    assert!(matches!(
        fence_err(
            h.adapter
                .submit_approved("b", h.gen("b"), &c.hook, &p, &c.ws)
        ),
        FenceError::NotCommittable {
            state: TaskState::AwaitingFinalProof,
            ..
        }
    ));
    assert_eq!(
        (tree(&h.ws()), tree(&compose_dir)),
        before,
        "preparation and refused fences wrote nothing"
    );
    assert!(!h.ws().join(PATH).exists());
    // The same approval commits once the chain is final: the fence refused it.
    h.finish("a");
    h.finalize("b");
    h.complete("b");
    let out = h
        .adapter
        .submit_approved("b", h.gen("b"), &c.hook, &p, &c.ws)
        .unwrap();
    committed_or_stub(out.outcome);
    assert!(
        !h.ws().join(PATH).exists(),
        "a World commit is not an effect"
    );
}

#[test]
fn unauthorized_effects_are_still_refused_by_the_canonical_path() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let c = Compose::new(&h);
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    // The fence passes; the desk MAC does not.
    let mut p = c.approval("req-forged");
    p.approver = "drake".into();
    let out = h
        .adapter
        .submit_approved("b", 1, &c.hook, &p, &c.ws)
        .expect("fence passes");
    let e = out.outcome.expect_err("unauthenticated approval");
    assert_eq!(
        (e.refused_by.as_str(), e.name.as_str()),
        ("PROPOSAL_REFUSED", "Unauthenticated")
    );
    // Refused before any replay claim: nothing durable, the task aborts.
    assert!(matches!(out.result, CommitResult::Failed { .. }));
    assert_eq!(h.state("b"), TaskState::Aborted);
    assert!(!h.ws().join(PATH).exists());
    // ComposeAuthorize through the fence, no desk proof and no compose
    // commit: refused by the existing authorize rules.
    h.register("d", &["a"]);
    h.finish("d");
    let req = MintRequest {
        cx_promotion: 1,
        proposal_sha256: sha(b"never committed"),
        workspace: c.ws.clone(),
        approver: "drake".into(),
        constraints: vec![],
    };
    let out = h
        .adapter
        .authorize("d", 1, &c.bridge, &req, None, None)
        .expect("fence passes");
    match out.outcome {
        ControlResponse::Error(e) => {
            if aien_omega_compose::LINKED {
                assert!(e.contains("DeskMacRequired"), "{e}");
            }
        }
        other => panic!("authorize must refuse: {other:?}"),
    }
}

#[test]
fn duplicate_commit_is_refused_and_a_restart_does_not_run_it_again() {
    let _home = home_guard::home_slot();
    let c0 = cfg(true, 4);
    let mut h = H::journaled(c0.clone());
    let c = Compose::new(&h);
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.prepare("b").unwrap();
    h.finish("a");
    h.finish("b");
    let p = c.approval("req-dup");
    let first = h
        .adapter
        .submit_approved("b", 1, &c.hook, &p, &c.ws)
        .unwrap();
    // The stub refuses the replay claim itself (no claim was made, nothing
    // ran): recorded as failed, the task aborts.
    let want = if aien_omega_compose::LINKED {
        assert!(matches!(first.result, CommitResult::Committed { .. }));
        TaskState::Committed
    } else {
        match &first.result {
            CommitResult::Failed { reason } => assert!(
                reason.starts_with("REPLAY_REFUSED Journal: ")
                    && reason.ends_with("is not linked (stub build)"),
                "{reason}"
            ),
            other => panic!("want Failed, got {other:?}"),
        }
        TaskState::Aborted
    };
    assert!(first.record_error.is_none());
    let report = committed_or_stub(first.outcome);
    assert_eq!(h.state("b"), want);
    // Same process, same task generation: refused before the hook runs.
    assert!(matches!(
        fence_err(h.adapter.submit_approved("b", 1, &c.hook, &p, &c.ws)),
        FenceError::AlreadySubmitted { .. }
    ));
    // Restart: the journal replays the outcome, the audit passes, no result
    // survives.
    let revoked = h.restart(c0);
    c.bridge.close();
    drop(c);
    let c = Compose::reopen(&h);
    assert!(revoked.is_empty(), "{revoked:?}");
    assert_eq!(h.state("b"), want);
    assert!(h.adapter.prepared("b").is_none());
    // The new process has no in-memory record of the call, but the journaled
    // outcome refuses a second run before the hook.
    match fence_err(h.adapter.submit_approved("b", 1, &c.hook, &p, &c.ws)) {
        FenceError::NotCommittable { state, .. } => assert_eq!(state, want),
        other => panic!("want NotCommittable, got {other}"),
    }
    // Under the fence, the durable replay ledger still answers an exact retry
    // without a second compose run.
    let again = c.hook.submit(&p, &c.ws);
    if let Some(first) = report {
        let r = again.expect("an exact retry gets the original result");
        assert_eq!(r.state, "ALREADY_COMMITTED");
        assert!(r.task.is_none(), "nothing ran again");
        assert_eq!(r.replay_claim, first.replay_claim);
        assert_eq!(r.grant_links, first.grant_links);
    } else {
        let e = again.expect_err("stub build must refuse the retry");
        assert_eq!(
            (e.refused_by.as_str(), e.name.as_str(), e.replay_claim),
            ("REPLAY_REFUSED", "Journal", None),
            "{e}"
        );
    }
}

#[test]
fn restart_with_a_provisional_task_drops_the_result_and_prepares_again() {
    let c0 = cfg(true, 4);
    let mut h = H::journaled(c0.clone());
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let r = h.prepare("b").unwrap();
    let revoked = h.restart(c0);
    assert!(revoked.is_empty());
    assert_eq!(h.state("b"), TaskState::Provisional);
    assert!(
        h.adapter.prepared("b").is_none(),
        "results are never journaled"
    );
    let again = h.prepare("b").unwrap();
    assert_eq!(again.plan_sha256(), r.plan_sha256());
    // A producer receipt removed from the store while the daemon was down:
    // the audit revokes the consumer at replay.
    let c1 = cfg(true, 4);
    let rid = h
        .adapter
        .inspect(|s| s.task("a").unwrap().interface.clone().unwrap().receipt_id);
    std::fs::remove_file(
        h.engine
            .completion_store
            .join(aien_proof::evidence::RECEIPT_DIR)
            .join(format!("{rid}.json")),
    )
    .unwrap();
    let revoked = h.restart(c1);
    assert!(revoked.contains(&"a".to_string()), "{revoked:?}");
    assert_eq!(h.state("b"), TaskState::Blocked);
}

/// M6 (#538 task 1): a task's own M1 receipt that is malformed (not a
/// receipt at all) or deleted after M2 accepted it is refused at the final
/// fence, and the effect never runs. The tampered case is covered by
/// `invalid_receipts_are_refused_by_the_event_gate_and_the_fence`.
#[test]
fn a_malformed_or_missing_own_receipt_is_refused_at_the_fence() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let rid = h
        .adapter
        .inspect(|s| s.task("b").unwrap().completion.clone().unwrap().receipt_id);
    let file = h
        .engine
        .completion_store
        .join(aien_proof::evidence::RECEIPT_DIR)
        .join(format!("{rid}.json"));
    let original = std::fs::read(&file).unwrap();
    let mut ran = false;
    for (case, bytes) in [
        ("truncated", Some(original[..original.len() / 2].to_vec())),
        ("not json", Some(b"\x00\xffnot a receipt".to_vec())),
        ("empty", Some(Vec::new())),
        ("deleted", None),
    ] {
        match bytes {
            Some(b) => std::fs::write(&file, b).unwrap(),
            None => std::fs::remove_file(&file).unwrap(),
        }
        let e = fence_err(
            h.adapter
                .fenced_for_tests("b", 1, |_| ran = true, committed),
        );
        assert!(
            matches!(e, FenceError::CompletionInvalid { .. }),
            "{case}: {e:?}"
        );
        assert!(!ran, "{case}: the effect ran");
        assert_eq!(h.state("b"), TaskState::Committable, "{case}");
    }
    std::fs::write(&file, &original).unwrap();
    h.adapter
        .fenced_for_tests("b", 1, |_| ran = true, committed)
        .unwrap();
    assert!(ran);
    assert_eq!(h.state("b"), TaskState::Committed);
}

/// M6 (#538 task 1): a restart while a task is unfinished never treats it as
/// committed. Provisional or work-done tasks replay short of `Committable`
/// and the fence refuses them (`NotCommittable`). A finalized task whose M1
/// completion was minted but never delivered to M2 replays without a
/// completion (M2 reaches `Committable` at `Finalize`, see
/// spark_harness::scheduler), so the fence refuses it (`NoCompletion`).
/// Nothing runs until the real completion is delivered after the restart.
#[test]
fn a_restart_with_an_unfinished_task_is_never_treated_as_committed() {
    let c = cfg(true, 4);
    let mut h = H::journaled(c.clone());
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("d", &["a"]);
    h.publish("a", 1);
    // b: provisional work done while its producer is not final.
    h.work_done("b");
    // d: provisional, nothing done.
    let mut ran = false;
    let not_final = |s: TaskState| {
        !matches!(
            s,
            TaskState::Committable | TaskState::Committing | TaskState::Committed
        )
    };
    let refused = |h: &H, t: &str, ran: &mut bool| {
        let e = fence_err(
            h.adapter
                .fenced_for_tests(t, h.gen(t), |_| *ran = true, committed),
        );
        assert!(matches!(e, FenceError::NotCommittable { .. }), "{t}: {e:?}");
    };
    h.restart(c.clone());
    for t in ["b", "d"] {
        assert!(not_final(h.state(t)), "{t}: {:?}", h.state(t));
        refused(&h, t, &mut ran);
    }
    // a becomes final; b finalizes and its completion is minted in M1, but the
    // daemon goes down before CompletionVerified reaches M2.
    h.finish("a");
    h.finalize("b");
    let minted = h.mint_completion("b");
    h.restart(c.clone());
    assert_eq!(h.state("b"), TaskState::Committable);
    assert!(
        h.adapter
            .inspect(|s| s.task("b").unwrap().completion.clone())
            .is_none(),
        "a receipt M2 never accepted is not a completion"
    );
    let e = fence_err(
        h.adapter
            .fenced_for_tests("b", h.gen("b"), |_| ran = true, committed),
    );
    assert!(
        matches!(e, FenceError::NoCompletion(ref t) if t == "b"),
        "{e:?}"
    );
    assert!(!ran, "an unfinished task ran its effect");
    assert_eq!(h.state("b"), TaskState::Committable);
    // Only the real completion, delivered after the restart, makes it final.
    let generation = h.gen("b");
    h.ok(EventKind::CompletionVerified {
        task: "b".into(),
        generation,
        claim: minted,
    });
    assert_eq!(h.state("b"), TaskState::Committable);
    h.adapter
        .fenced_for_tests("b", generation, |_| ran = true, committed)
        .unwrap();
    assert!(ran);
    assert_eq!(h.state("b"), TaskState::Committed);
    assert!(not_final(h.state("d")));
}

#[test]
fn revocation_during_preparation_discards_the_result() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let adapter = Arc::clone(&h.adapter);
    h.adapter.set_prep_hook(Some(Arc::new(move |_: &str| {
        adapter
            .deliver(Event {
                id: "fail-a-mid-prep".into(),
                kind: EventKind::ProducerFailed {
                    task: "a".into(),
                    generation: 1,
                    reason: "failed while b prepared".into(),
                },
            })
            .unwrap();
    })));
    assert!(matches!(h.prepare("b"), Err(PrepError::Revoked { .. })));
    assert!(h.adapter.prepared("b").is_none());
    assert_eq!(h.state("b"), TaskState::Blocked);
}

#[test]
fn an_interface_change_during_preparation_refuses_the_old_generation() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let store_dir = h.tmp.path().join("harness");
    let adapter = Arc::clone(&h.adapter);
    // While b's operation runs, a publishes a new interface (store write,
    // then the M2 event).
    h.adapter.set_prep_hook(Some(Arc::new(move |_: &str| {
        let mut e = HarnessEngine::load(&store_dir).unwrap();
        e.completion_clock = fixed_receipt_clock;
        let v2 = e
            .publish_interface(&InterfaceSubmission {
                task_id: "a".into(),
                generation: 1,
                source_commit: COMMIT.into(),
                interface: json!({ "task": "a", "api_version": 2 }),
            })
            .unwrap()
            .claim();
        adapter
            .deliver(Event {
                id: "republish-a-mid-prep".into(),
                kind: EventKind::InterfacePublished {
                    task: "a".into(),
                    generation: 1,
                    claim: v2,
                },
            })
            .unwrap();
    })));
    // b is revoked and dispatched again at generation 2 before the operation
    // ends, so M2 would admit it again: only the generation recheck refuses
    // the result made against the old interface.
    assert!(matches!(h.prepare("b"), Err(PrepError::Revoked { .. })));
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Provisional, 2));
    assert!(h.adapter.prepared("b").is_none());
    h.adapter.set_prep_hook(None);
    assert_eq!(h.prepare("b").unwrap().generation(), 2);
}

#[test]
fn a_producer_receipt_that_stops_verifying_during_preparation_refuses_the_result() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    let (v1, _) = h.publish("a", 1);
    let gen_b = h.gen("b");
    let file = h
        .engine
        .completion_store
        .join(aien_proof::evidence::RECEIPT_DIR)
        .join(format!("{}.json", v1.receipt_id));
    let original = std::fs::read(&file).unwrap();
    let (f2, o2) = (file.clone(), original.clone());
    // No event: the producer's receipt is altered in the store while b's
    // operation runs, so only the post-operation M2 check can see it.
    h.adapter.set_prep_hook(Some(Arc::new(move |_: &str| {
        let text = String::from_utf8(o2.clone()).unwrap();
        std::fs::write(&f2, text.replacen("spark-harness", "spark-harnesz", 1)).unwrap();
    })));
    assert!(matches!(h.prepare("b"), Err(PrepError::Revoked { .. })));
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Provisional, gen_b));
    assert!(h.adapter.prepared("b").is_none());
    h.adapter.set_prep_hook(None);
    std::fs::write(&file, &original).unwrap();
    h.prepare("b").unwrap();
}

#[test]
fn a_producer_completing_during_preparation_is_recorded_as_final() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.work_done("a");
    h.finalize("a");
    let claim = h.mint_completion("a");
    let gen_b = h.gen("b");
    let adapter = Arc::clone(&h.adapter);
    h.adapter.set_prep_hook(Some(Arc::new(move |_: &str| {
        adapter
            .deliver(Event {
                id: "complete-a-mid-prep".into(),
                kind: EventKind::CompletionVerified {
                    task: "a".into(),
                    generation: 1,
                    claim: claim.clone(),
                },
            })
            .unwrap();
    })));
    // Admitted on the interface, the producer went final while the
    // operation ran. Nothing was revoked, so the consumer keeps its
    // generation, and the stored result carries the readiness re-checked
    // after the operation (final), never the admission-time one.
    let r = h.prepare("b").unwrap();
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Provisional, gen_b));
    assert!(!r.is_provisional());
    assert_eq!(r.bindings()[0].level, ReadinessLevel::FinalVerified);
    let stored = h.adapter.prepared("b").unwrap();
    assert_eq!(stored.bindings()[0].level, ReadinessLevel::FinalVerified);
}

#[test]
fn one_preparation_per_task_at_a_time() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = std::sync::Mutex::new(release_rx);
    let calls = AtomicU64::new(0);
    // Only the first call parks; a second admitted call would run straight
    // through and fail the Busy assertion below instead of deadlocking.
    h.adapter.set_prep_hook(Some(Arc::new(move |_: &str| {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            entered_tx.send(()).unwrap();
            let _ = release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10));
        }
    })));
    let (a2, ws) = (Arc::clone(&h.adapter), h.ws());
    let first = std::thread::spawn(move || a2.prepare("b", 1, GOAL, &ws).map(|_| ()));
    entered_rx.recv().unwrap();
    assert!(matches!(h.prepare("b"), Err(PrepError::Busy(_))));
    release_tx.send(()).unwrap();
    first.join().unwrap().unwrap();
    h.adapter.set_prep_hook(None);
    h.prepare("b").unwrap();
}

#[test]
fn concurrency_is_bounded_by_the_m2_permits() {
    let mut h = H::new(cfg(true, 1));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("c", &["a"]);
    h.publish("a", 1);
    // A holds the only permit: neither consumer may prepare yet.
    assert_eq!(h.adapter.permits_in_use().running, 1);
    for t in ["b", "c"] {
        assert_eq!(h.state(t), TaskState::Blocked);
        assert!(h.prepare(t).is_err());
    }
    h.work_done("a");
    // One consumer at a time, in task-id order.
    assert_eq!(
        (h.state("b"), h.state("c")),
        (TaskState::Provisional, TaskState::Blocked)
    );
    h.prepare("b").unwrap();
    assert!(h.prepare("c").is_err());
    h.work_done("b");
    assert_eq!(h.state("c"), TaskState::Provisional);
    h.prepare("c").unwrap();
    assert!(h.adapter.permits_in_use().running <= 1);
}

#[test]
fn the_commit_decision_freezes_the_inputs_without_holding_any_lock() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    // An unrelated task with a live lease, heartbeating during the call.
    h.register("x", &[]);
    assert_eq!(h.state("x"), TaskState::Provisional);
    let adapter = Arc::clone(&h.adapter);
    let store_dir = h.tmp.path().join("harness");
    let wait = Duration::from_secs(10);
    let fenced = h.adapter.fenced_for_tests(
        "b",
        1,
        |_| {
            let during = adapter.state("b");
            // Each probe runs on its own thread and must finish while the
            // call is still running: neither the adapter mutex nor a store
            // lock is held.
            let (tx, rx) = mpsc::channel();
            let a2 = Arc::clone(&adapter);
            let t = tx.clone();
            std::thread::spawn(move || {
                let r = a2.deliver(Event {
                    id: "revoke-a-during-commit".into(),
                    kind: EventKind::Revoke {
                        task: "a".into(),
                        generation: 1,
                        reason: "interface withdrawn".into(),
                    },
                });
                t.send(format!("revoke {r:?}")).unwrap();
            });
            let revoke = rx.recv_timeout(wait).expect("revoke blocked by the call");
            let a3 = Arc::clone(&adapter);
            let t = tx.clone();
            std::thread::spawn(move || {
                let r = a3.deliver(Event {
                    id: "beat-x-during-commit".into(),
                    kind: EventKind::Heartbeat {
                        task: "x".into(),
                        generation: 1,
                        beat: 1,
                    },
                });
                t.send(format!("beat {r:?}")).unwrap();
            });
            let beat = rx
                .recv_timeout(wait)
                .expect("heartbeat blocked by the call");
            std::thread::spawn(move || {
                let mut e = HarnessEngine::load(&store_dir).unwrap();
                e.completion_clock = fixed_receipt_clock;
                let r = e.publish_interface(&InterfaceSubmission {
                    task_id: "a".into(),
                    generation: 1,
                    source_commit: COMMIT.into(),
                    interface: json!({ "task": "a", "api_version": 9 }),
                });
                tx.send(format!("store {:?}", r.map(|_| ()))).unwrap();
            });
            let store = rx
                .recv_timeout(wait)
                .expect("store writer blocked by the call");
            // A second fence for the same task generation while the first
            // call runs: refused as already submitted, and it never runs.
            let a4 = Arc::clone(&adapter);
            let (tx2, rx2) = mpsc::channel();
            std::thread::spawn(move || {
                let r = a4.fenced_for_tests("b", 1, |_| (), committed);
                tx2.send(format!("{:?}", r.map(|_| "ran"))).unwrap();
            });
            let second = rx2
                .recv_timeout(wait)
                .expect("second fence blocked by the call");
            (during, revoke, beat, store, second)
        },
        committed,
    );
    let out = fenced.expect("fence passes");
    let (during, revoke, beat, store, second) = out.outcome;
    assert!(second.starts_with("Err(AlreadySubmitted"), "{second}");
    assert_eq!(during, Some(TaskState::Committing));
    assert!(
        revoke.starts_with("revoke Ok(Deferred") && revoke.contains("committing: \"b\""),
        "an upstream revoke must be held back while committing: {revoke}"
    );
    assert!(beat.contains("Ok(Applied(Applied"), "{beat}");
    assert_eq!(store, "store Ok(())");
    assert_eq!(out.result, committed(&()));
    assert!(out.record_error.is_none());
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Committed, 1));
    // After the outcome the held-back revoke applied, and stopped at the
    // committed task.
    assert_eq!(out.redelivered.len(), 1);
    assert_eq!(out.redelivered[0].id, "revoke-a-during-commit");
    assert!(
        matches!(out.redelivered[0].result, Ok(Outcome::Applied(_))),
        "{:?}",
        out.redelivered[0].result
    );
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Committed, 1));
    assert!(matches!(
        fence_err(h.adapter.fenced_for_tests("b", 1, |_| (), committed)),
        FenceError::AlreadySubmitted { .. }
    ));
}

#[test]
fn a_panicking_commit_is_uncertain_and_never_runs_again() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("c", &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.finish("a");
    h.finish("b");
    assert_eq!(h.state("c"), TaskState::Provisional);
    let adapter = Arc::clone(&h.adapter);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.fenced_for_tests("b", 1, |_| panic!("commit blew up"), committed)
    }));
    assert!(r.is_err(), "the panic is resumed");
    assert_eq!(h.state("b"), TaskState::CommitUncertain);
    assert_eq!(h.state("c"), TaskState::Blocked, "consumers revoked");
    let mut ran = false;
    assert!(matches!(
        fence_err(
            h.adapter
                .fenced_for_tests("b", 1, |_| ran = true, committed)
        ),
        FenceError::AlreadySubmitted { .. }
    ));
    assert!(!ran);
}

#[test]
fn a_failed_commit_aborts_the_task_and_a_new_generation_may_commit() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let failed = |_: &()| CommitResult::Failed {
        reason: "refused before any write".into(),
    };
    let out = h.adapter.fenced_for_tests("b", 1, |_| (), failed).unwrap();
    assert!(matches!(out.result, CommitResult::Failed { .. }));
    assert_eq!(h.state("b"), TaskState::Aborted);
    h.ok(EventKind::Retry {
        task: "b".into(),
        generation: 1,
    });
    assert_eq!(h.gen("b"), 2);
    h.finish("b");
    let out = h
        .adapter
        .fenced_for_tests("b", 2, |_| (), committed)
        .unwrap();
    assert!(out.record_error.is_none());
    assert_eq!(h.state("b"), TaskState::Committed);
}

#[test]
fn outcomes_are_classified_conservatively() {
    let refusal = |name: &str, claim: Option<u64>| {
        let e = ApprovedRefusal {
            refused_by: "REPLAY_REFUSED".into(),
            name: name.into(),
            detail: "d".into(),
            request_id: "r".into(),
            trace_id: "t".into(),
            approval_id: "a".into(),
            replay_claim: claim,
        };
        classify_approved(&Err(Box::new(e)))
    };
    assert!(matches!(
        refusal("Unauthenticated", None),
        CommitResult::Failed { .. }
    ));
    assert!(matches!(
        refusal("NotCommitted", Some(3)),
        CommitResult::Failed { .. }
    ));
    assert!(matches!(
        refusal("ComposeError", Some(3)),
        CommitResult::Uncertain { .. }
    ));
    assert!(matches!(
        refusal("Mismatch", Some(3)),
        CommitResult::Uncertain { .. }
    ));
    assert!(matches!(
        refusal("GrantNotWritten", Some(3)),
        CommitResult::Uncertain { .. }
    ));
    let e = |s: &str| classify_authorize(&ControlResponse::Error(s.into()));
    assert!(matches!(
        e("EFFECT_REFUSED DeskMacRequired: none"),
        CommitResult::Failed { .. }
    ));
    assert!(matches!(
        e("effect record: re-read 7: io"),
        CommitResult::Uncertain { .. }
    ));
    assert!(matches!(
        classify_authorize(&ControlResponse::ComposeNoted(
            aien_runtime::control::ComposeNoteReport {
                machine_id: "m".into(),
                id: 7,
                kind: "authorization".into(),
                digest: "d".into(),
                text_sha256: "s".into(),
                links: vec![],
            }
        )),
        CommitResult::Committed { .. }
    ));
}

#[test]
fn disabled_flag_is_the_default_and_sequential() {
    assert!(!ProvisionalConfig::default().enabled);
    let mut h = H::new(cfg(false, 4));
    assert!(!h.adapter.enabled());
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    assert_eq!(h.state("b"), TaskState::Blocked);
    assert!(h.prepare("b").is_err());
    h.finish("a");
    let r = h.prepare("b").unwrap();
    assert!(!r.is_provisional());
}

// ---------- through the daemon socket ----------

async fn daemon(
    socket: &Path,
    bridge: Arc<ComposeBridge>,
    adapter: Arc<PrepAdapter>,
) -> (
    aien_runtime::client::AienRuntimeClient,
    tokio::task::JoinHandle<Result<(), String>>,
) {
    let cfg = aien_scheduler::SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = aien_runtime::spine::AienRuntimeSpine::new(
        64,
        cfg,
        aien_kv_cache::create_shared_kv_manager(256, 16),
    );
    let server = aien_runtime::server::AienRuntimeServer::new(spine, socket)
        .with_compose_bridge(bridge)
        .with_prep_adapter(adapter);
    let handle = tokio::spawn(async move {
        server
            .run(aien_inference_abi::MockInferenceBackend::new(1))
            .await
    });
    let client = aien_runtime::client::AienRuntimeClient::new(socket);
    for _ in 0..250 {
        if client.is_alive().await {
            return (client, handle);
        }
        if handle.is_finished() {
            panic!("daemon stopped: {:?}", handle.await);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

fn fence_refused(r: &ControlResponse, name: &str) -> bool {
    matches!(r, ControlResponse::Error(e) if e.starts_with(&format!("FENCE_REFUSED {name}")))
}

/// The original `ComposeApprovedProposal` answer: a result or a refusal from
/// the canonical path, never a fence error.
fn canonical_approved(r: &ControlResponse) -> bool {
    matches!(
        r,
        ControlResponse::ComposeApprovedResult(_) | ControlResponse::ComposeApprovedRefused(_)
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_daemon_fences_bound_requests_and_runs_unbound_ones_unchanged() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let c = Compose::new(&h);
    let socket = h.tmp.path().join("d.sock");
    let (client, server) = daemon(&socket, c.bridge.clone(), h.adapter.clone()).await;
    let send = |cmd: ControlCommand| {
        let client = &client;
        async move { client.send_command(cmd).await.unwrap() }
    };
    let approved = |p: &ApprovedProposal| ControlCommand::ComposeApprovedProposal {
        proposal: p.clone(),
        workspace: c.ws.clone(),
    };
    let authorize = |digest: &str| ControlCommand::ComposeAuthorize {
        cx_promotion: 1,
        proposal_sha256: digest.into(),
        workspace: c.ws.clone(),
        approver: "drake".into(),
        constraints: vec![],
        desk_proof: None,
        objective_id: None,
    };

    // Early overlap: B prepares while its producer A is only interface
    // verified.
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    assert!(h.prepare("b").unwrap().is_provisional());
    let bound = c.approval("req-bound");
    let free = c.approval("req-free");
    let commit_key = FenceKey::Approval(bound.approval_id.clone());
    let grant_key = FenceKey::Proposal(sha(b"bound grant"));
    h.adapter.bind_approval(commit_key.clone(), "b", 1).unwrap();
    h.adapter.bind_approval(grant_key.clone(), "b", 1).unwrap();
    assert!(h
        .adapter
        .binding(&FenceKey::Approval(free.approval_id.clone()))
        .is_none());

    // Bound, while B is provisional: the fence refuses both kinds and
    // nothing runs.
    let compose_dir = c.bridge.dir().to_path_buf();
    let before = (tree(&h.ws()), tree(&compose_dir));
    let r = send(approved(&bound)).await;
    assert!(fence_refused(&r, "NotCommittable"), "{r:?}");
    let r = send(authorize(&sha(b"bound grant"))).await;
    assert!(fence_refused(&r, "NotCommittable"), "{r:?}");
    assert_eq!(
        (tree(&h.ws()), tree(&compose_dir)),
        before,
        "a fenced request wrote nothing"
    );
    assert_eq!(h.state("b"), TaskState::Provisional);

    // Unbound: the original calls run, whatever the M2 state.
    let r = send(authorize(&sha(b"never committed"))).await;
    match &r {
        ControlResponse::Error(e) if aien_omega_compose::LINKED => {
            assert!(e.starts_with("EFFECT_REFUSED DeskMacRequired"), "{e}")
        }
        ControlResponse::Error(e) => assert!(
            e.starts_with("compose home ") && e.ends_with("is not linked (stub build)"),
            "{e}"
        ),
        other => panic!("the canonical authorize must refuse: {other:?}"),
    }
    let r = send(approved(&free)).await;
    assert!(canonical_approved(&r), "{r:?}");
    assert_eq!(
        h.state("b"),
        TaskState::Provisional,
        "unbound runs touch no task"
    );

    // Once A and B are final the bound approval passes the fence and the
    // canonical path decides: the World commit and its own grant, in one
    // fenced call.
    h.finish("a");
    h.finish("b");
    let r = send(approved(&bound)).await;
    assert!(canonical_approved(&r), "{r:?}");
    let report = match &r {
        ControlResponse::ComposeApprovedResult(rep) if aien_omega_compose::LINKED => {
            assert_eq!(rep.state, "COMMITTED");
            Some(rep.clone())
        }
        // The stub refuses the replay claim itself: nothing durable.
        ControlResponse::ComposeApprovedRefused(_) if !aien_omega_compose::LINKED => None,
        other => panic!("unexpected answer: {other:?}"),
    };
    let Some(report) = report else {
        // The refusal is classified `Failed`: B is aborted and keeps both
        // bindings, so a resend is fenced and never reaches the canonical
        // path unfenced.
        assert_eq!(h.state("b"), TaskState::Aborted);
        assert_eq!(h.adapter.binding(&commit_key), Some(("b".into(), 1)));
        assert_eq!(h.adapter.binding(&grant_key), Some(("b".into(), 1)));
        let r = send(approved(&bound)).await;
        assert!(fence_refused(&r, "AlreadySubmitted"), "{r:?}");
        let r = send(authorize(&sha(b"bound grant"))).await;
        assert!(fence_refused(&r, "AlreadySubmitted"), "{r:?}");
        assert_eq!(h.state("b"), TaskState::Aborted);
        let _ = client.send_command(ControlCommand::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
        return;
    };
    // B committed, so M2 dropped both of its bindings.
    assert!(h.adapter.binding(&commit_key).is_none());
    assert!(h.adapter.binding(&grant_key).is_none());
    assert_eq!(h.adapter.inspect(|s| s.approvals_len()), 0);
    assert_eq!(h.state("b"), TaskState::Committed);
    assert!(report.approved_grant.is_some(), "{report:?}");
    // An exact retry goes to the canonical path, whose replay ledger
    // answers it without a second run.
    let r = send(approved(&bound)).await;
    match &r {
        ControlResponse::ComposeApprovedResult(rep) => {
            assert_eq!(rep.state, "ALREADY_COMMITTED");
            assert!(rep.task.is_none(), "nothing ran again");
        }
        other => panic!("want ALREADY_COMMITTED, got {other:?}"),
    }
    // No later grant stage exists for this flow. M2 refuses to bind one to
    // the ended task, and the canonical mint refuses a desk-signed grant
    // against the approved commit: the approved path writes no
    // `compose_commit` record (compose/bridge.rs `run_task_inner`), which
    // `ComposeAuthorize` requires (effects.rs).
    let digest = report.compose_proposal_sha256.clone();
    let cx = report.grant_links[0];
    let late = h
        .adapter
        .bind_approval(FenceKey::Proposal(digest.clone()), "b", 1);
    assert!(
        matches!(late, Err(SchedError::WrongState { .. })),
        "{late:?}"
    );
    let key = DeskKey::load(&desk_key_path(c.bridge.dir())).unwrap();
    let nonce = format!("m3-grant-{cx}");
    let mac = key.sign_authorize(&AuthorizeBinding {
        cx_promotion: cx,
        proposal_sha256: digest.clone(),
        path: PATH.into(),
        content_sha256: sha(CONTENT.as_bytes()),
        workspace: c.ws.clone(),
        approver: "drake".into(),
        constraints: vec![],
        nonce: nonce.clone(),
        desk_key_id: key.id().into(),
    });
    let r = send(ControlCommand::ComposeAuthorize {
        cx_promotion: cx,
        proposal_sha256: digest,
        workspace: c.ws.clone(),
        approver: "drake".into(),
        constraints: vec![],
        desk_proof: Some(DeskProof { nonce, mac }),
        objective_id: None,
    })
    .await;
    match &r {
        ControlResponse::Error(e) => {
            assert!(e.starts_with("EFFECT_REFUSED NotAuthorized"), "{e}")
        }
        other => panic!("want the canonical NotAuthorized refusal, got {other:?}"),
    }
    assert_eq!(h.state("b"), TaskState::Committed);
    // In process the generation is spent.
    assert!(matches!(
        fence_err(h.adapter.submit_approved("b", 1, &c.hook, &bound, &c.ws)),
        FenceError::AlreadySubmitted { .. }
    ));

    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_aborted_task_keeps_its_bound_requests_fenced_until_retried_and_bound_again() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let c = Compose::new(&h);
    let socket = h.tmp.path().join("d.sock");
    let (client, server) = daemon(&socket, c.bridge.clone(), h.adapter.clone()).await;
    let send = |cmd: ControlCommand| {
        let client = &client;
        async move { client.send_command(cmd).await.unwrap() }
    };
    let authorize = |digest: &str| ControlCommand::ComposeAuthorize {
        cx_promotion: 1,
        proposal_sha256: digest.into(),
        workspace: c.ws.clone(),
        approver: "drake".into(),
        constraints: vec![],
        desk_proof: None,
        objective_id: None,
    };
    // The canonical call's own refusal (no desk MAC, or the stub), never a
    // fence error: what an unfenced run answers.
    let canonical = |r: &ControlResponse| matches!(r, ControlResponse::Error(e) if !e.starts_with("FENCE_REFUSED"));
    let r = send(authorize(&sha(b"never bound"))).await;
    assert!(canonical(&r), "control, unbound: {r:?}");

    // (a) Cancel, producer failure and lease expiry: the binding stays and
    // the request is refused.
    type Abort = fn(&mut H, &str);
    let aborts: [(&str, Abort); 3] = [
        ("cancel", |h, t| {
            let generation = h.gen(t);
            h.ok(EventKind::Cancel {
                task: t.into(),
                generation,
            });
        }),
        ("producer failed", |h, t| {
            let generation = h.gen(t);
            h.ok(EventKind::ProducerFailed {
                task: t.into(),
                generation,
                reason: "test".into(),
            });
        }),
        ("lease expired", |h, t| {
            // The producer finishes first, so only b's provisional lease
            // runs out (a running producer's expiry would revoke b instead).
            h.finish(&t.replacen('b', "a", 1));
            h.ticks.fetch_add(2_000_000, Ordering::SeqCst);
            h.ok(EventKind::Tick);
        }),
    ];
    for (i, (how, abort)) in aborts.into_iter().enumerate() {
        let (a, b) = (format!("a{i}"), format!("b{i}"));
        h.register(&a, &[]);
        h.register(&b, &[a.as_str()]);
        h.publish(&a, 1);
        assert_eq!(h.state(&b), TaskState::Provisional, "{how}");
        let digest = sha(how.as_bytes());
        h.adapter
            .bind_approval(FenceKey::Proposal(digest.clone()), &b, 1)
            .unwrap();
        abort(&mut h, &b);
        assert_eq!(h.state(&b), TaskState::Aborted, "{how}");
        let r = send(authorize(&digest)).await;
        assert!(fence_refused(&r, "NotCommittable"), "{how}: {r:?}");
    }

    // (b) A committable task is cancelled, retried and committable again:
    // the old binding refuses until the request is bound at the new
    // generation, and then the request runs through the fence.
    h.register("p", &[]);
    h.register("q", &["p"]);
    h.publish("p", 1);
    h.finish("p");
    h.finish("q");
    let digest = sha(b"retried grant");
    let key = FenceKey::Proposal(digest.clone());
    h.adapter.bind_approval(key.clone(), "q", 1).unwrap();
    h.ok(EventKind::Cancel {
        task: "q".into(),
        generation: 1,
    });
    let r = send(authorize(&digest)).await;
    assert!(fence_refused(&r, "NotCommittable"), "{r:?}");
    h.ok(EventKind::Retry {
        task: "q".into(),
        generation: 1,
    });
    h.finish("q");
    assert_eq!(h.gen("q"), 2);
    let r = send(authorize(&digest)).await;
    assert!(fence_refused(&r, "StaleGeneration"), "{r:?}");
    assert_eq!(h.state("q"), TaskState::Committable);
    h.adapter.bind_approval(key.clone(), "q", 2).unwrap();
    let r = send(authorize(&digest)).await;
    assert!(canonical(&r), "bound again, it runs fenced: {r:?}");
    if aien_omega_compose::LINKED {
        // The canonical refusal (no desk MAC) is a failed commit: q is
        // aborted, keeps the binding, and a resend stays fenced.
        assert_eq!(h.state("q"), TaskState::Aborted);
        assert_eq!(h.adapter.binding(&key), Some(("q".into(), 2)));
        let r = send(authorize(&digest)).await;
        assert!(fence_refused(&r, "AlreadySubmitted"), "{r:?}");
    } else {
        // The stub answer is not an EFFECT_REFUSED: uncertain, terminal.
        // The binding is kept, so a resend is refused by the fence instead
        // of running unfenced.
        assert_eq!(h.state("q"), TaskState::CommitUncertain);
        assert_eq!(h.adapter.binding(&key), Some(("q".into(), 2)));
        let r = send(authorize(&digest)).await;
        assert!(fence_refused(&r, "AlreadySubmitted"), "{r:?}");
    }

    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
}

// The model proposal of the `RunComposeTask` flow, as in
// provenance_link_test.rs.
const RUN_GOAL: &str = "write the note";
const RUN_PROPOSAL: &str = "filename: NOTES.md\nkeep every change inside the workspace\n";
const RUN_CONTENT: &str = "keep every change inside the workspace\n";

fn model_proposer() -> ComposeProposer {
    Arc::new(|_: &str, _: Duration| {
        Ok(aien_runtime::compose::Generation {
            text: RUN_PROPOSAL.to_string(),
            tokens: 3,
            finish_reason: Some("eos".into()),
            token_ids: Some(vec![11, 12, 13]),
            prompt_tokens: Some(5),
            prompt_ids_sha256: Some(aien_runtime::compose::token_ids_sha256(&[1, 2, 3, 4, 5])),
            decoding: Some(aien_inference_abi::DecodeObservation {
                greedy_tokens: 3,
                ..Default::default()
            }),
            ops: Some(aien_inference_abi::OpEvidence {
                backend: "reference-cpu".into(),
                native_fallbacks: 0,
                reference_runs: 0,
                report: "OP_REPORT native=[] reference=[] native_fallbacks=[] reference_runs=[]"
                    .into(),
                gpu_executions: None,
            }),
        })
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bound_grant_after_an_unfenced_compose_run_mints_through_the_fence() {
    let _home = home_guard::home_slot();
    let mut h = H::new(cfg(true, 4));
    let home = h.tmp.path().join("compose");
    DeskKey::create(&desk_key_path(&home)).unwrap();
    let bridge = Arc::new(ComposeBridge::new(
        home.clone(),
        model_proposer(),
        "test:m3-model",
    ));
    let ws = std::fs::canonicalize(h.ws()).unwrap().display().to_string();
    let socket = h.tmp.path().join("g.sock");
    let (client, server) = daemon(&socket, bridge.clone(), h.adapter.clone()).await;
    let send = |cmd: ControlCommand| {
        let client = &client;
        async move { client.send_command(cmd).await.unwrap() }
    };
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let digest = sha(RUN_PROPOSAL.as_bytes());
    let key = FenceKey::Proposal(digest.clone());
    h.adapter.bind_approval(key.clone(), "b", 1).unwrap();

    // The model run has no binding kind: it runs unfenced, whatever B's
    // state, and records the compose commit the grant needs.
    let r = send(ControlCommand::RunComposeTask {
        context: None,
        goal: RUN_GOAL.into(),
        workspace: ws.clone(),
    })
    .await;
    let cx = match &r {
        ControlResponse::ComposeTaskResult(rep) if aien_omega_compose::LINKED => {
            let v = serde_json::to_value(&**rep).unwrap();
            assert_eq!(v["committed"], true, "{v}");
            assert!(v["compose_commit"].as_u64().is_some(), "{v}");
            v["cx_promotion"].as_u64().unwrap()
        }
        ControlResponse::Error(e) if !aien_omega_compose::LINKED => {
            assert!(e.ends_with("is not linked (stub build)"), "{e}");
            1
        }
        other => panic!("unexpected run answer: {other:?}"),
    };
    assert_eq!(h.state("b"), TaskState::Provisional);
    let desk = DeskKey::load(&desk_key_path(&home)).unwrap();
    let grant = |nonce: &str| {
        let mac = desk.sign_authorize(&AuthorizeBinding {
            cx_promotion: cx,
            proposal_sha256: digest.clone(),
            path: PATH.into(),
            content_sha256: sha(RUN_CONTENT.as_bytes()),
            workspace: ws.clone(),
            approver: "drake".into(),
            constraints: vec![],
            nonce: nonce.into(),
            desk_key_id: desk.id().into(),
        });
        ControlCommand::ComposeAuthorize {
            cx_promotion: cx,
            proposal_sha256: digest.clone(),
            workspace: ws.clone(),
            approver: "drake".into(),
            constraints: vec![],
            desk_proof: Some(DeskProof {
                nonce: nonce.into(),
                mac,
            }),
            objective_id: None,
        }
    };

    // Bound, while B is provisional: the grant is the fenced call and is
    // refused; nothing is written.
    let before = (tree(Path::new(&ws)), tree(&home));
    let r = send(grant("m3-run-early")).await;
    assert!(fence_refused(&r, "NotCommittable"), "{r:?}");
    assert_eq!(
        (tree(Path::new(&ws)), tree(&home)),
        before,
        "a fenced grant wrote nothing"
    );

    // Final: the grant passes the fence and the canonical mint decides.
    h.finish("a");
    h.finish("b");
    let r = send(grant("m3-run-final")).await;
    if aien_omega_compose::LINKED {
        match &r {
            ControlResponse::ComposeNoted(_) => {}
            other => panic!("want the grant minted, got {other:?}"),
        }
        assert_eq!(h.state("b"), TaskState::Committed);
    } else {
        match &r {
            ControlResponse::Error(e) => assert!(
                e.starts_with("compose home ") && e.ends_with("is not linked (stub build)"),
                "{e}"
            ),
            other => panic!("the stub authorize must refuse: {other:?}"),
        }
        // Not an `EFFECT_REFUSED`: classified uncertain (fail closed).
        assert_eq!(h.state("b"), TaskState::CommitUncertain);
    }
    let r = send(grant("m3-run-final")).await;
    if aien_omega_compose::LINKED {
        // B committed: the binding is dropped and the retry is the original
        // call, which the replay ledger refuses.
        assert!(h.adapter.binding(&key).is_none());
        match &r {
            ControlResponse::Error(e) => assert!(e.starts_with("EFFECT_REFUSED "), "{e}"),
            other => panic!("a second grant must be refused: {other:?}"),
        }
    } else {
        // B is uncertain: the binding is kept for good, so the retry stays
        // fenced and is refused (the task ended) instead of running unfenced.
        assert_eq!(h.adapter.binding(&key), Some(("b".into(), 1)));
        assert!(fence_refused(&r, "AlreadySubmitted"), "{r:?}");
    }

    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_binding_survives_a_daemon_restart_and_the_request_stays_fenced() {
    let _home = home_guard::home_slot();
    let c0 = cfg(true, 4);
    let mut h = H::journaled(c0.clone());
    let c = Compose::new(&h);
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    let bound = c.approval("req-bound-restart");
    let free = c.approval("req-free-restart");
    let key = FenceKey::Approval(bound.approval_id.clone());
    h.adapter.bind_approval(key.clone(), "b", 1).unwrap();
    // The daemon restarts: a new adapter replays the journal.
    let revoked = h.restart(c0);
    assert!(revoked.is_empty(), "{revoked:?}");
    assert_eq!(h.adapter.binding(&key), Some(("b".to_string(), 1)));
    let socket = h.tmp.path().join("r.sock");
    let (client, server) = daemon(&socket, c.bridge.clone(), h.adapter.clone()).await;
    let compose_dir = c.bridge.dir().to_path_buf();
    let before = (tree(&h.ws()), tree(&compose_dir));
    let r = client
        .send_command(ControlCommand::ComposeApprovedProposal {
            proposal: bound.clone(),
            workspace: c.ws.clone(),
        })
        .await
        .unwrap();
    assert!(fence_refused(&r, "NotCommittable"), "{r:?}");
    assert_eq!(
        (tree(&h.ws()), tree(&compose_dir)),
        before,
        "the restarted daemon still fenced the bound request"
    );
    // Control: an unbound approval runs the original call after the restart.
    let r = client
        .send_command(ControlCommand::ComposeApprovedProposal {
            proposal: free,
            workspace: c.ws.clone(),
        })
        .await
        .unwrap();
    assert!(canonical_approved(&r), "{r:?}");
    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
}

#[test]
fn a_restart_during_a_commit_is_uncertain_and_never_runs_it_again() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("c", &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.finish("a");
    h.finish("b");
    assert_eq!(h.state("c"), TaskState::Provisional);
    let journal = h.tmp.path().join("journal.jsonl");
    let copy = h.tmp.path().join("journal-mid-commit.jsonl");
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    // The daemon dies while the commit runs: its journal at that moment is
    // opened by a new process.
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                std::fs::copy(&journal, &copy).unwrap();
                // The audit's own revocations: settling the interrupted
                // commit is not one of them (its consumers are revoked by
                // the settled outcome itself, checked below).
                let (a, _audit) =
                    PrepAdapter::open(cfg(true, 4), &store, &copy, clock(&ticks)).unwrap();
                let mut ran = false;
                let again = a.fenced_for_tests("b", 1, |_| ran = true, committed).err();
                (a.state("b"), a.generation("c"), again, ran)
            },
            committed,
        )
        .unwrap();
    let (state, gen_c, again, ran) = out.outcome;
    assert_eq!(state, Some(TaskState::CommitUncertain));
    assert_eq!(gen_c, Some(2), "the consumer was revoked");
    assert!(
        matches!(
            again,
            Some(FenceError::NotCommittable {
                state: TaskState::CommitUncertain,
                ..
            })
        ),
        "{again:?}"
    );
    assert!(!ran, "the restarted process ran nothing");
    // The process that did not die records its outcome as usual.
    assert_eq!(h.state("b"), TaskState::Committed);
}

#[test]
fn a_revoke_during_a_commit_is_applied_after_the_outcome() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("d", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    assert_eq!(h.state("d"), TaskState::Provisional);
    h.prepare("d").unwrap();
    let adapter = Arc::clone(&h.adapter);
    let revoke = |id: &str, reason: &str| Event {
        id: id.into(),
        kind: EventKind::Revoke {
            task: "a".into(),
            generation: 1,
            reason: reason.into(),
        },
    };
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                let first = adapter.deliver(revoke("revoke-a", "withdrawn"));
                let again = adapter.deliver(revoke("revoke-a", "withdrawn"));
                let conflict = adapter.deliver(revoke("revoke-a", "other reason"));
                let d = (
                    adapter.state("d"),
                    adapter.generation("d"),
                    adapter.prepared("d").is_some(),
                );
                (first, again, conflict, d, adapter.deferred_len())
            },
            committed,
        )
        .unwrap();
    let (first, again, conflict, d, queued) = out.outcome;
    let deferred = Delivered::Deferred {
        committing: "b".into(),
        queued: 1,
    };
    assert_eq!(first.unwrap(), deferred);
    assert_eq!(again.unwrap(), deferred, "a redelivery is not queued twice");
    assert!(
        matches!(
            conflict,
            Err(DeliverError::Scheduler(SchedError::EventIdConflict(_)))
        ),
        "{conflict:?}"
    );
    assert_eq!(
        d,
        (Some(TaskState::Provisional), Some(1), true),
        "nothing applied while b commits"
    );
    assert_eq!(queued, 1);
    // After the outcome: the revoke applied and invalidated the other
    // consumer; the committed task is not reached.
    assert_eq!(out.redelivered.len(), 1);
    assert_eq!(out.redelivered[0].id, "revoke-a");
    match &out.redelivered[0].result {
        Ok(Outcome::Applied(fx)) => assert!(fx.revoked.contains(&"d".to_string()), "{fx:?}"),
        other => panic!("want the revoke applied, got {other:?}"),
    }
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!(h.gen("d"), 2);
    assert!(
        h.adapter.prepared("d").is_none(),
        "the result was discarded"
    );
    assert_eq!((h.state("b"), h.gen("b")), (TaskState::Committed, 1));
}

#[test]
fn an_event_held_by_another_running_commit_stays_queued() {
    let mut h = H::new(cfg(true, 8));
    // Two independent chains: a -> b and x -> y.
    for (p, c) in [("a", "b"), ("x", "y")] {
        h.register(p, &[]);
        h.register(c, &[p]);
        h.publish(p, 1);
        h.finish(p);
        h.finish(c);
    }
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "y",
            1,
            |_| {
                // Held back by y's commit.
                let held = adapter.deliver(Event {
                    id: "revoke-x".into(),
                    kind: EventKind::Revoke {
                        task: "x".into(),
                        generation: 1,
                        reason: "withdrawn".into(),
                    },
                });
                // b's whole commit runs and ends while y still commits: its
                // drain must keep the event y still refuses.
                let b = adapter
                    .fenced_for_tests("b", 1, |_| (), committed)
                    .map(|f| f.redelivered.len());
                (held, b, adapter.deferred_len(), adapter.generation("x"))
            },
            committed,
        )
        .unwrap();
    let (held, b, queued, gen_x) = out.outcome;
    assert!(
        matches!(held, Ok(Delivered::Deferred { ref committing, .. }) if committing == "y"),
        "{held:?}"
    );
    assert_eq!(b.unwrap(), 0, "nothing was reported by b's drain");
    assert_eq!((queued, gen_x), (1, Some(1)), "still queued, not applied");
    // y's outcome delivers it.
    assert_eq!(out.redelivered.len(), 1);
    assert_eq!(out.redelivered[0].id, "revoke-x");
    assert!(matches!(out.redelivered[0].result, Ok(Outcome::Applied(_))));
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!(
        (h.state("b"), h.state("y")),
        (TaskState::Committed, TaskState::Committed)
    );
}

#[test]
fn the_deferred_queue_is_bounded_and_fails_closed() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                let mut answers = Vec::new();
                for i in 0..=DEFERRED_CAPACITY {
                    answers.push(adapter.deliver(Event {
                        id: format!("revoke-{i}"),
                        kind: EventKind::Revoke {
                            task: "a".into(),
                            generation: 1,
                            reason: format!("withdrawn {i}"),
                        },
                    }));
                }
                answers
            },
            committed,
        )
        .unwrap();
    for (i, a) in out.outcome[..DEFERRED_CAPACITY].iter().enumerate() {
        match a {
            Ok(Delivered::Deferred { queued, .. }) => assert_eq!(*queued, i + 1),
            other => panic!("event {i}: want Deferred, got {other:?}"),
        }
    }
    match &out.outcome[DEFERRED_CAPACITY] {
        Err(DeliverError::DeferredFull {
            committing,
            capacity,
        }) => assert_eq!((committing.as_str(), *capacity), ("b", DEFERRED_CAPACITY)),
        other => panic!("want DeferredFull, got {other:?}"),
    }
    // Every queued event is delivered again and reported, oldest first; the
    // refused one was never queued.
    let ids: Vec<String> = out.redelivered.iter().map(|r| r.id.clone()).collect();
    let want: Vec<String> = (0..DEFERRED_CAPACITY)
        .map(|i| format!("revoke-{i}"))
        .collect();
    assert_eq!(ids, want);
    assert!(matches!(out.redelivered[0].result, Ok(Outcome::Applied(_))));
    assert!(!out.redelivered[0].dropped);
    // The first revoke moved a to generation 2, so the rest are refused as
    // stale. A stale event can never apply (generations only grow), so each
    // is reported as dropped and leaves the queue: the queue is empty again
    // instead of holding 63 dead slots.
    assert!(out.redelivered[1..]
        .iter()
        .all(|r| { matches!(r.result, Err(SchedError::StaleGeneration { .. })) && r.dropped }));
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!(h.state("b"), TaskState::Committed);
}

#[test]
fn a_stale_deferred_event_is_dropped_and_leaves_the_sidecar() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let side = deferred_journal_path(&h.tmp.path().join("journal.jsonl"));
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                (
                    adapter.deliver(revoke_a("revoke-first")),
                    adapter.deliver(revoke_a("revoke-second")),
                )
            },
            committed,
        )
        .unwrap();
    assert!(matches!(
        out.outcome.0,
        Ok(Delivered::Deferred { queued: 1, .. })
    ));
    assert!(matches!(
        out.outcome.1,
        Ok(Delivered::Deferred { queued: 2, .. })
    ));
    // The first revoke applies (a moves to generation 2); the second names
    // generation 1 and is stale for good: reported, dropped, and gone from
    // the sidecar, so the queue has its slot back.
    assert_eq!(out.redelivered.len(), 2);
    assert_eq!(out.redelivered[0].id, "revoke-first");
    assert!(matches!(out.redelivered[0].result, Ok(Outcome::Applied(_))));
    assert!(!out.redelivered[0].dropped);
    assert_eq!(out.redelivered[1].id, "revoke-second");
    assert!(
        matches!(
            out.redelivered[1].result,
            Err(SchedError::StaleGeneration { ref task, submitted: 1, current: 2 }) if task == "a"
        ),
        "{:?}",
        out.redelivered[1].result
    );
    assert!(out.redelivered[1].dropped);
    assert!(out.deferred_error.is_none());
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
    assert_eq!(h.gen("a"), 2);
    // A restart replays nothing: the drop was journaled.
    drop(adapter);
    h.restart(cfg(true, 4));
    assert!(h.adapter.take_replayed_deferred().is_empty());
    assert_eq!(h.adapter.deferred_len(), 0);
}

#[test]
fn an_uncertain_commit_keeps_the_binding_so_the_request_stays_fenced() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let key = FenceKey::Approval("approval-uncertain".into());
    h.adapter.bind_approval(key.clone(), "b", 1).unwrap();
    let adapter = Arc::clone(&h.adapter);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.fenced_for_tests(
            "b",
            1,
            |_| panic!("commit blew up before any write"),
            committed,
        )
    }));
    assert!(r.is_err(), "the panic is resumed");
    assert_eq!(h.state("b"), TaskState::CommitUncertain);
    // The binding is kept: the daemon still routes this request through the
    // fence, and the fence refuses it (the task ended) instead of letting
    // the same request run unfenced. Only a new key, never bound, runs as
    // the original call.
    assert_eq!(h.adapter.binding(&key), Some(("b".to_string(), 1)));
    let mut ran = false;
    assert!(matches!(
        fence_err(
            h.adapter
                .fenced_for_tests("b", 1, |_| ran = true, committed)
        ),
        FenceError::AlreadySubmitted { .. }
    ));
    assert!(!ran);
    assert!(matches!(
        h.adapter
            .bind_approval(FenceKey::Approval("approval-late".into()), "b", 1),
        Err(SchedError::WrongState { .. })
    ));
    // The binding survives a restart along with the terminal state.
    drop(adapter);
    h.restart(cfg(true, 4));
    assert_eq!(h.state("b"), TaskState::CommitUncertain);
    assert_eq!(h.adapter.binding(&key), Some(("b".to_string(), 1)));
}

#[test]
fn the_fence_events_are_refused_by_deliver() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let claim = h
        .adapter
        .inspect(|s| s.task("b").unwrap().completion.clone().unwrap());
    let kinds = [
        (
            "begin_commit",
            EventKind::BeginCommit {
                task: "b".into(),
                generation: 1,
                completion: claim,
                producers: BTreeMap::new(),
            },
        ),
        (
            "commit_outcome",
            EventKind::CommitOutcome {
                task: "b".into(),
                generation: 1,
                outcome: committed(&()),
            },
        ),
        (
            "bind_approval",
            EventKind::BindApproval {
                key: "approval:x".into(),
                task: "b".into(),
                generation: 1,
            },
        ),
    ];
    for (name, kind) in kinds {
        match h.deliver(kind) {
            Err(DeliverError::FenceOnly(k)) => assert_eq!(k, name),
            other => panic!("{name}: want FenceOnly, got {other:?}"),
        }
    }
    assert_eq!(h.state("b"), TaskState::Committable);
    assert!(h.adapter.binding(&FenceKey::Approval("x".into())).is_none());
    // Control: an ordinary event is delivered.
    assert!(matches!(
        h.deliver(EventKind::Tick),
        Ok(Delivered::Applied(_))
    ));
}

#[test]
fn begin_commit_holds_the_store_writers_out() {
    let mut h = H::new(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let store_dir = h.tmp.path().join("harness");
    // A writer for a task M2 does not know: it takes the same store lock
    // but changes nothing the fence verifies.
    let writer = move |version: u32| {
        let (tx, rx) = mpsc::channel();
        let dir = store_dir.clone();
        std::thread::spawn(move || {
            let mut e = HarnessEngine::load(&dir).unwrap();
            e.completion_clock = fixed_receipt_clock;
            let r = e.publish_interface(&InterfaceSubmission {
                task_id: "outside".into(),
                generation: 1,
                source_commit: COMMIT.into(),
                interface: json!({ "task": "outside", "api_version": version }),
            });
            tx.send(r.map(|_| ()).map_err(|e| e.to_string())).unwrap();
        });
        rx
    };
    // Control: with no fence running, the writer finishes at once.
    writer(7)
        .recv_timeout(Duration::from_secs(10))
        .expect("writer runs")
        .unwrap();
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = std::sync::Mutex::new(release_rx);
    let entered_tx = std::sync::Mutex::new(entered_tx);
    h.adapter.set_begin_hook(Some(Arc::new(move |_: &str| {
        entered_tx.lock().unwrap().send(()).unwrap();
        let _ = release_rx
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10));
    })));
    let a2 = Arc::clone(&h.adapter);
    let fence = std::thread::spawn(move || {
        a2.fenced_for_tests("b", 1, |_| (), committed)
            .map(|f| f.result)
            .map_err(|e| e.to_string())
    });
    entered_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the fence reached its hook");
    // Inside begin_commit, between the checks and the journaled transition:
    // a store writer must wait.
    let rx = writer(8);
    assert!(
        matches!(
            rx.recv_timeout(Duration::from_millis(300)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "a store writer ran inside begin_commit"
    );
    release_tx.send(()).unwrap();
    rx.recv_timeout(Duration::from_secs(10))
        .expect("the writer runs once the fence releases its locks")
        .unwrap();
    assert_eq!(fence.join().unwrap().unwrap(), committed(&()));
    h.adapter.set_begin_hook(None);
}

// ---------- the journaled deferred queue ----------

fn revoke_a(id: &str) -> Event {
    Event {
        id: id.into(),
        kind: EventKind::Revoke {
            task: "a".into(),
            generation: 1,
            reason: "withdrawn".into(),
        },
    }
}

#[test]
fn a_revoke_deferred_by_a_commit_that_crashed_is_applied_on_open() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("d", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    assert_eq!(h.state("d"), TaskState::Provisional);
    let journal = h.tmp.path().join("journal.jsonl");
    let sidecar = deferred_journal_path(&journal);
    let copy = h.tmp.path().join("journal-crash.jsonl");
    let bare = h.tmp.path().join("journal-crash-no-sidecar.jsonl");
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                let deferred = adapter.deliver(revoke_a("revoke-a"));
                // The daemon dies here, before the drain: the journal and
                // its sidecar as they are now.
                std::fs::copy(&journal, &copy).unwrap();
                std::fs::copy(&sidecar, deferred_journal_path(&copy)).unwrap();
                std::fs::copy(&journal, &bare).unwrap();
                let (a, _audit) =
                    PrepAdapter::open(cfg(true, 4), &store, &copy, clock(&ticks)).unwrap();
                let replayed = a.take_replayed_deferred();
                let again = a.take_replayed_deferred().len();
                let reopened = (
                    a.state("b"),
                    a.generation("d"),
                    a.deferred_len(),
                    std::fs::read_to_string(deferred_journal_path(&copy)).unwrap(),
                );
                // Negative control: the same crash without the sidecar
                // loses the revoke.
                let (n, _audit) =
                    PrepAdapter::open(cfg(true, 4), &store, &bare, clock(&ticks)).unwrap();
                let control = (n.generation("d"), n.take_replayed_deferred().len());
                (deferred, replayed, again, reopened, control)
            },
            committed,
        )
        .unwrap();
    let (deferred, replayed, again, reopened, control) = out.outcome;
    assert!(
        matches!(deferred, Ok(Delivered::Deferred { ref committing, .. }) if committing == "b"),
        "{deferred:?}"
    );
    // Reopened: the interrupted commit settled first, then the revoke.
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, "revoke-a");
    match &replayed[0].result {
        Ok(Outcome::Applied(fx)) => assert!(fx.revoked.contains(&"d".to_string()), "{fx:?}"),
        other => panic!("want the revoke applied, got {other:?}"),
    }
    assert_eq!(again, 0, "taken once");
    assert_eq!(
        reopened,
        (Some(TaskState::CommitUncertain), Some(2), 0, String::new())
    );
    assert_eq!(control, (Some(1), 0));
    // The process that did not die drained it as usual and emptied the
    // sidecar.
    assert_eq!(out.redelivered.len(), 1);
    assert!(out.deferred_error.is_none());
    assert_eq!(h.gen("d"), 2);
    assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), "");
    // A second open of the drained journal replays nothing.
    drop(adapter);
    let n = h.restart(cfg(true, 4));
    assert!(n.is_empty(), "{n:?}");
    assert!(h.adapter.take_replayed_deferred().is_empty());
    assert_eq!(h.gen("d"), 2);
}

#[test]
fn a_bad_deferred_sidecar_fails_the_open() {
    let h = H::journaled(cfg(true, 4));
    let copy = h.tmp.path().join("journal-copy.jsonl");
    std::fs::copy(h.tmp.path().join("journal.jsonl"), &copy).unwrap();
    let side = deferred_journal_path(&copy);
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    let open = || PrepAdapter::open(cfg(true, 4), &store, &copy, clock(&ticks));
    let lines = |events: Vec<Event>| -> String {
        events
            .iter()
            .map(|e| serde_json::to_string(e).unwrap() + "\n")
            .collect()
    };
    let revokes =
        |n: usize| -> Vec<Event> { (0..n).map(|i| revoke_a(&format!("revoke-{i}"))).collect() };
    let bind = Event {
        id: "bind-x".into(),
        kind: EventKind::BindApproval {
            key: "approval:x".into(),
            task: "a".into(),
            generation: 1,
        },
    };
    let reason = "r".repeat(DEFERRED_MAX_BYTES as usize / DEFERRED_CAPACITY + 1);
    let large: Vec<Event> = (0..DEFERRED_CAPACITY)
        .map(|i| Event {
            id: format!("large-{i}"),
            kind: EventKind::Revoke {
                task: "a".into(),
                generation: 1,
                reason: reason.clone(),
            },
        })
        .collect();
    for (what, text) in [
        ("not json", "not json\n".to_string()),
        ("over the cap", lines(revokes(DEFERRED_CAPACITY + 1))),
        ("fence-only", lines(vec![bind])),
        // Well-formed and under the event cap, but over the byte cap.
        ("too large", lines(large)),
    ] {
        std::fs::write(&side, text).unwrap();
        match open() {
            Err(SchedError::Journal(e)) => assert!(e.contains("deferred"), "{what}: {e}"),
            Err(e) => panic!("{what}: want a journal error, got {e}"),
            Ok(_) => panic!("{what}: the open must fail"),
        }
    }
    // Control: exactly the cap opens and every event is delivered. Each is
    // refused here (no task is registered), reported, and kept queued and
    // journaled until it is discarded.
    let full = lines(revokes(DEFERRED_CAPACITY));
    std::fs::write(&side, &full).unwrap();
    let (a, _) = open().unwrap();
    let replayed = a.take_replayed_deferred();
    assert_eq!(replayed.len(), DEFERRED_CAPACITY);
    assert!(replayed
        .iter()
        .all(|r| matches!(r.result, Err(SchedError::UnknownTask(_)))));
    assert_eq!(a.deferred_len(), DEFERRED_CAPACITY);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), full);
    for i in 0..DEFERRED_CAPACITY {
        let id = format!("revoke-{i}");
        let ev = a.discard_deferred(&id).unwrap().expect("queued");
        assert_eq!(ev.id, id);
    }
    assert_eq!(a.deferred_len(), 0);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
}

#[test]
fn a_deferred_event_that_cannot_be_journaled_is_refused() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let side = deferred_journal_path(&h.tmp.path().join("journal.jsonl"));
    // A non-empty directory where the sidecar goes: the rename fails.
    std::fs::create_dir_all(side.join("block")).unwrap();
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                (
                    adapter.deliver(revoke_a("revoke-a")),
                    adapter.deferred_len(),
                )
            },
            committed,
        )
        .unwrap();
    let (r, queued) = out.outcome;
    assert!(matches!(r, Err(DeliverError::DeferredJournal(_))), "{r:?}");
    assert_eq!(queued, 0, "nothing was queued");
    assert!(out.redelivered.is_empty());
    assert_eq!(h.gen("a"), 1, "and nothing applied");
    // Control: with the sidecar writable the same event applies.
    std::fs::remove_dir_all(&side).unwrap();
    assert!(matches!(
        h.adapter.deliver(revoke_a("revoke-a")),
        Ok(Delivered::Applied(_))
    ));
    assert_eq!(h.gen("a"), 2);
}

#[test]
fn a_failed_sidecar_rewrite_after_the_drain_is_reported() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.register("d", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let side = deferred_journal_path(&h.tmp.path().join("journal.jsonl"));
    let adapter = Arc::clone(&h.adapter);
    let out = h
        .adapter
        .fenced_for_tests(
            "b",
            1,
            |_| {
                let r = adapter.deliver(revoke_a("revoke-a"));
                // The sidecar becomes unwritable before the drain.
                std::fs::remove_file(&side).unwrap();
                std::fs::create_dir_all(side.join("block")).unwrap();
                r
            },
            committed,
        )
        .unwrap();
    assert!(matches!(out.outcome, Ok(Delivered::Deferred { .. })));
    // The revoke still applied; the failed rewrite is reported.
    assert_eq!(out.redelivered.len(), 1);
    assert!(matches!(out.redelivered[0].result, Ok(Outcome::Applied(_))));
    assert_eq!(h.gen("d"), 2);
    let e = out.deferred_error.expect("the rewrite failure is reported");
    assert!(e.contains("deferred"), "{e}");
}

#[test]
fn an_open_whose_sidecar_cannot_be_rewritten_fails() {
    let mut h = H::journaled(cfg(true, 4));
    // a is registered, so the deferred revoke applies on open and the
    // sidecar must be rewritten without it.
    h.register("a", &[]);
    let dir = h.tmp.path().join("ro");
    std::fs::create_dir_all(&dir).unwrap();
    let copy = dir.join("j.jsonl");
    std::fs::copy(h.tmp.path().join("journal.jsonl"), &copy).unwrap();
    let side = deferred_journal_path(&copy);
    std::fs::write(
        &side,
        serde_json::to_string(&revoke_a("revoke-a")).unwrap() + "\n",
    )
    .unwrap();
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    let open = || PrepAdapter::open(cfg(true, 4), &store, &copy, clock(&ticks));
    // A directory where the rewrite's temp file goes: the journal opens and
    // locks as usual, only the sidecar rewrite fails.
    let blocker = dir.join(format!(".j.jsonl.deferred.tmp-{}", std::process::id()));
    std::fs::create_dir_all(blocker.join("x")).unwrap();
    match open() {
        Err(SchedError::Journal(e)) => assert!(e.contains("j.jsonl.deferred"), "{e}"),
        Err(e) => panic!("want the sidecar rewrite error, got {e}"),
        Ok(_) => panic!("the open must fail when the sidecar cannot be rewritten"),
    }
    // Control: without the blocker, the same open succeeds. The failed
    // open had journaled the revoke before its rewrite failed, so a is at
    // generation 2 once, and the redelivered copy is refused as stale,
    // reported, and dropped (it can never apply); the sidecar is empty.
    std::fs::remove_dir_all(&blocker).unwrap();
    let (a, _) = open().unwrap();
    assert_eq!(a.generation("a"), Some(2));
    let replayed = a.take_replayed_deferred();
    assert_eq!(replayed.len(), 1);
    assert!(
        matches!(replayed[0].result, Err(SchedError::StaleGeneration { .. })),
        "{:?}",
        replayed[0].result
    );
    assert!(replayed[0].dropped);
    assert_eq!(a.deferred_len(), 0);
    assert!(a.discard_deferred("revoke-a").unwrap().is_none());
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
}

#[test]
fn a_refused_redelivery_stays_queued_until_discarded() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    h.register("b", &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let side = deferred_journal_path(&h.tmp.path().join("journal.jsonl"));
    // A cancel of b while b commits is refused with CommitInProgress and
    // deferred.
    let cancel_b = Event {
        id: "cancel-b".into(),
        kind: EventKind::Cancel {
            task: "b".into(),
            generation: 1,
        },
    };
    let line = serde_json::to_string(&cancel_b).unwrap() + "\n";
    let adapter = Arc::clone(&h.adapter);
    let ev = cancel_b.clone();
    let out = h
        .adapter
        .fenced_for_tests("b", 1, |_| adapter.deliver(ev), committed)
        .unwrap();
    assert!(
        matches!(out.outcome, Ok(Delivered::Deferred { ref committing, .. }) if committing == "b"),
        "{:?}",
        out.outcome
    );
    // Redelivered after the commit: b is committed, so the cancel is
    // refused (a committed task cannot abort). It is reported and kept, in memory and in the sidecar.
    assert_eq!(h.state("b"), TaskState::Committed);
    assert_eq!(out.redelivered.len(), 1);
    assert!(
        matches!(out.redelivered[0].result, Err(SchedError::Transition(_))),
        "{:?}",
        out.redelivered[0].result
    );
    assert!(out.deferred_error.is_none());
    let queued: Vec<String> = h
        .adapter
        .deferred_events()
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(queued, vec!["cancel-b".to_string()]);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), line);
    // A restart replays it again: refused again, still kept.
    drop(adapter);
    h.restart(cfg(true, 4));
    let replayed = h.adapter.take_replayed_deferred();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, "cancel-b");
    assert!(
        matches!(replayed[0].result, Err(SchedError::Transition(_))),
        "{:?}",
        replayed[0].result
    );
    assert_eq!(h.adapter.deferred_len(), 1);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), line);
    // Only an explicit discard removes it, and the removal is journaled.
    assert!(h.adapter.discard_deferred("unknown").unwrap().is_none());
    assert_eq!(h.adapter.deferred_len(), 1);
    let gone = h.adapter.discard_deferred("cancel-b").unwrap();
    assert_eq!(gone.map(|e| e.id), Some("cancel-b".to_string()));
    assert_eq!(h.adapter.deferred_len(), 0);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
    assert!(h.adapter.discard_deferred("cancel-b").unwrap().is_none());
    h.restart(cfg(true, 4));
    assert!(h.adapter.take_replayed_deferred().is_empty());
}

#[test]
fn a_discard_that_cannot_be_journaled_keeps_the_event() {
    let h = H::journaled(cfg(true, 4));
    let copy = h.tmp.path().join("journal-copy.jsonl");
    std::fs::copy(h.tmp.path().join("journal.jsonl"), &copy).unwrap();
    let side = deferred_journal_path(&copy);
    let line = serde_json::to_string(&revoke_a("revoke-a")).unwrap() + "\n";
    std::fs::write(&side, &line).unwrap();
    let store = h.engine.completion_store.clone();
    let (a, _) = PrepAdapter::open(cfg(true, 4), &store, &copy, clock(&h.ticks)).unwrap();
    // No task a: refused and kept.
    assert_eq!(a.deferred_len(), 1);
    std::fs::remove_file(&side).unwrap();
    std::fs::create_dir_all(side.join("block")).unwrap();
    assert!(matches!(
        a.discard_deferred("revoke-a"),
        Err(DeliverError::DeferredJournal(_))
    ));
    assert_eq!(a.deferred_len(), 1, "the failed discard keeps the event");
    // Control: writable again, the discard succeeds.
    std::fs::remove_dir_all(&side).unwrap();
    assert!(a.discard_deferred("revoke-a").unwrap().is_some());
    assert_eq!(a.deferred_len(), 0);
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
}

#[test]
fn a_second_adapter_on_the_same_journal_is_refused() {
    let mut h = H::journaled(cfg(true, 4));
    h.register("a", &[]);
    let journal = h.tmp.path().join("journal.jsonl");
    let side = deferred_journal_path(&journal);
    let line = serde_json::to_string(&revoke_a("revoke-x")).unwrap() + "\n";
    std::fs::write(&side, &line).unwrap();
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    let open = |p: &Path| PrepAdapter::open(cfg(true, 4), &store, p, clock(&ticks));
    let held = |what: &str, r: Result<(PrepAdapter, Vec<String>), SchedError>| match r {
        Err(SchedError::Journal(e)) => {
            assert!(e.contains("held by another process"), "{what}: {e}")
        }
        Err(e) => panic!("{what}: want the journal lock error, got {e}"),
        Ok(_) => panic!("{what}: a second adapter on one journal must be refused"),
    };
    // The same path, a symlink to the file, a path through a symlinked
    // directory and a path with a `..` step all name the one journal.
    let link = h.tmp.path().join("link.jsonl");
    std::os::unix::fs::symlink(&journal, &link).unwrap();
    let dir_link = h.tmp.path().join("dirlink");
    std::os::unix::fs::symlink(h.tmp.path(), &dir_link).unwrap();
    std::fs::create_dir_all(h.tmp.path().join("sub")).unwrap();
    held("same path", open(&journal));
    held("file symlink", open(&link));
    held("dir symlink", open(&dir_link.join("journal.jsonl")));
    held("dot-dot", open(&h.tmp.path().join("sub/../journal.jsonl")));
    // None of the refused opens read or rewrote the first adapter's
    // sidecar, and the first adapter still works.
    assert_eq!(std::fs::read_to_string(&side).unwrap(), line);
    assert_eq!(h.gen("a"), 1);
    // Negative control: a different journal in the same directory opens,
    // with its own sidecar.
    let other = h.tmp.path().join("journal.log");
    assert_ne!(deferred_journal_path(&other), side);
    let (o, _) = open(&other).unwrap();
    assert_eq!(o.deferred_len(), 0);
    drop(o);
    // Once the first adapter is gone the symlinked path opens the journal,
    // and that adapter alone delivers the sidecar it now owns.
    drop(h.adapter);
    let (b, _) = open(&link).unwrap();
    let replayed = b.take_replayed_deferred();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, "revoke-x");
    assert!(matches!(replayed[0].result, Ok(Outcome::Applied(_))));
    assert_eq!(b.generation("a"), Some(2));
    assert_eq!(std::fs::read_to_string(&side).unwrap(), "");
}

#[test]
fn a_journal_named_like_a_sidecar_is_refused() {
    let h = H::journaled(cfg(true, 4));
    let tmp = &h.tmp;
    let store = h.engine.completion_store.clone();
    let ticks = Arc::clone(&h.ticks);
    let bad = tmp.path().join("x.deferred");
    match PrepAdapter::open(cfg(true, 4), &store, &bad, clock(&ticks)) {
        Err(SchedError::Journal(e)) => assert!(e.contains(".deferred"), "{e}"),
        Err(e) => panic!("want a journal error, got {e}"),
        Ok(_) => panic!("a journal ending in .deferred must be refused"),
    }
    assert!(!bad.exists(), "nothing was created");
    // Negative control: the same directory, a plain name.
    let good = tmp.path().join("x.jsonl");
    assert!(PrepAdapter::open(cfg(true, 4), &store, &good, clock(&ticks)).is_ok());
    assert_eq!(
        deferred_journal_path(&good),
        tmp.path().join("x.jsonl.deferred")
    );
}
