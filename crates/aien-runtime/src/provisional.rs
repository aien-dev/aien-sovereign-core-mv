//! Provisional engineering preparation in front of the canonical compose
//! path (Headstart M3, issue #537). Compiled only with cargo feature
//! `provisional-prep` (default off); the default build has no trace of it and
//! the control ABI (`crate::control`) is the same in both builds: no command,
//! field or response is added. With the feature, a daemon given an adapter
//! (`AienRuntimeServer::with_prep_adapter`) sends a `ComposeApprovedProposal`
//! or `ComposeAuthorize` request through the fence only when
//! [`PrepAdapter::bind_approval`] bound it to a task generation; every other
//! request runs the original call unchanged. A binding is a journaled M2
//! `BindApproval` event, so it survives a daemon restart (the replayed
//! scheduler still fences the request). M2 drops the bindings of a task only
//! when its commit ended (`Committed` or `CommitUncertain`) and refuses a
//! binding to such a task; the request is then unbound again, the daemon
//! runs it as the original call and the canonical durable guards decide (see
//! Duplicates below). An `Aborted` task (cancelled, failed, lease expired, a
//! failed commit, or an abort cascaded from an ancestor) keeps its bindings,
//! so its bound requests are refused (`NotCommittable`); after a retry they
//! name a superseded generation and are refused (`StaleGeneration`) until
//! they are bound again. A bound request never falls back to the unfenced
//! call while its task may still commit. Each task holds at most
//! `MAX_APPROVAL_BINDINGS_PER_TASK` keys; a new key past that is refused.
//!
//! Two halves:
//!
//! 1. Early preparation. [`PrepAdapter::prepare`] runs exactly one whitelisted
//!    operation, [`PrepOperation::ComposePlanPreview`], for a task the M2
//!    scheduler (`spark_harness::scheduler`) has dispatched as `Provisional`.
//!    It maps to [`OperationClass::ReadOnlyAnalysis`], so the M2 gate
//!    `may_start` allows it on an interface-verified producer. The operation
//!    is `compose::rederive::task_decision`: it reads the workspace to build
//!    the compose prompt and destination for a goal and never writes
//!    (bridge.rs `gather_inputs` doc: "Never writes"). No compose home is
//!    opened, no model runs, no tool is called, nothing goes to the network.
//!    The only output is an in-memory [`PreparationResult`] bound to each
//!    producer's source commit, generation, artifact digest and the receipt
//!    id the store verified.
//!
//! 2. Final commit fence. [`PrepAdapter::submit_approved`] (World commit
//!    through `approved::ProposerHook::submit`) and
//!    [`PrepAdapter::authorize`] (`effects::authorize_with_objective`, the
//!    `ComposeAuthorize` path) run the existing call only when, for the
//!    task's current generation: the M2 task is `Committable`, M2
//!    `may_finalize` passes (every producer final-verified, recursively), and
//!    the task's own M1 completion receipt verifies (`verify_final`, which is
//!    M1 `verify_completion`). The fence is a prerequisite only. Desk MAC,
//!    AEGIS verify, the replay ledger, the effect ledger and the World commit
//!    stay inside the unchanged calls and still decide.
//!
//!    One task generation has one fenced call. Each bound flow has exactly
//!    one durable step this daemon takes for it, so that step is the fence:
//!    - `ComposeApprovedProposal` (`submit_approved`): the proposer hook
//!      runs the World commit and writes its own grant in the same call
//!      (approved.rs `submit`, `write_approved_grant`); it writes no
//!      `compose_commit` record (compose/bridge.rs `run_task_inner` writes
//!      one only when `approved_text` is None), so no later
//!      `ComposeAuthorize` can mint against it.
//!    - `RunComposeTask` then `ComposeAuthorize`: the run is a model call
//!      this adapter does not fence (see Duplicates); its compose-commit
//!      record is what `ComposeAuthorize` mints against (effects.rs). The
//!      grant (`authorize`) is the fenced call, from `Committable`.
//!    Both start from `Committable` with the same checks.
//!
//! No check-to-commit window, and no lock held across the call. The fence
//! checks run in one critical section that holds (a) the adapter mutex, the
//! only way to deliver a scheduler event, and (b) the harness store's
//! interface and completion locks (`<store>/interfaces/.lock`,
//! `<store>/completions/.lock`, the same aien-proof flock files
//! `publish_interface` and `complete_verified_workflow` take). The same
//! critical section ends with the journaled M2 `BeginCommit`, which moves
//! the task to `Committing` and freezes its generation,
//! its own completion claim and every producer's completion claim. That
//! event is the decision point. Both locks are then released and the
//! existing call runs. While the task is `Committing`, M2 refuses every
//! event that would change what was checked (`SchedError::CommitInProgress`
//! for a revoke, cancel, failure, proof change or new generation of the
//! task or any ancestor; a second commit in the same dependency chain), and
//! the M2 audit and lease expiry defer theirs. Store writes after the
//! decision cannot change the call: it runs on the proof taken under the
//! locks and never re-reads the store. The call's result is journaled as
//! `CommitOutcome` (`Committed`, `Failed` which aborts the task, or
//! `Uncertain` which is terminal); consumers are revoked on the last two. A
//! panic is recorded as `Uncertain`. A replay that finds a `Committing` task
//! with no outcome marks it `CommitUncertain` and never runs it again (the
//! approved replay ledger's rule for an in-flight claim). Heartbeats, other
//! fences and harness writers in every process proceed while a call runs.
//!
//! Refused events are not lost: an event M2 refuses with `CommitInProgress`
//! through [`PrepAdapter::deliver`] is queued ([`Delivered::Deferred`], at
//! most [`DEFERRED_CAPACITY`]; a full queue refuses the event with
//! [`DeliverError::DeferredFull`] and queues nothing) and delivered again
//! right after the commit outcome is recorded, under the same mutex, before
//! any other event or fence. What each redelivery did is returned in
//! [`Fenced::redelivered`]. Only an event M2 applied (or answered as a
//! duplicate) leaves the queue: one still refused by another running commit,
//! and one M2 refuses for any other reason (a stale generation, a wrong
//! state, a journal write failure), stays queued and journaled, so a revoke
//! or cancel is never dropped without a decision. It is delivered again
//! after every later outcome and on every open, and it holds a queue slot
//! until the caller removes it ([`PrepAdapter::discard_deferred`]); a full
//! queue refuses new deferrals (fail closed). With a journaled scheduler
//! ([`PrepAdapter::open`]) the queue is journaled in a sidecar next to the M2
//! journal ([`deferred_journal_path`]): rewritten atomically (temp file,
//! fsync, rename, directory fsync) before `Deferred` is returned and after
//! every drain or discard that changed it. A queue that cannot be written
//! refuses the event ([`DeliverError::DeferredJournal`]); a failed rewrite
//! after a drain is reported ([`Fenced::deferred_error`]) and the next
//! rewrite replaces it. On open, after the M2 audit settled every
//! interrupted commit, the journaled events are delivered again in order, so
//! a revoke or cancel deferred by a commit that never recorded its outcome
//! still lands after it ([`PrepAdapter::take_replayed_deferred`] returns
//! what each did). An event that was already applied before the crash is a
//! duplicate, or a stale generation once its task moved on (spark-harness
//! `scheduler/dedup.rs`); the second stays queued until discarded. A sidecar
//! that does not parse, holds more than [`DEFERRED_CAPACITY`] events or
//! holds a fence-only event fails the open. One adapter owns a journal and
//! its sidecar: open resolves the journal path to its canonical form, takes
//! the M2 journal lock (`<journal>.lock`, held for the adapter's life; a
//! second open of the same journal is refused) and only then reads the
//! sidecar, whose name extends the journal's full file name, so distinct
//! journals never share one.
//!
//! Duplicates: a second fenced call for the same (task, generation) is
//! refused before the existing call runs ([`FenceError::AlreadySubmitted`]
//! in this process; M2 also refuses a second `BeginCommit` once the task
//! left `Committable`, across restarts too because it is journaled).
//! Across restarts the existing durable guards
//! decide: the approved replay ledger (`approved_replay::claim`) refuses a
//! claimed approval for ever and answers an exact retry with
//! `ALREADY_COMMITTED` without a second run; `ComposeAuthorize` refuses a
//! used desk nonce and a live or spent grant (effects.rs `AlreadyAuthorized`,
//! `AlreadySpent`). A model `RunComposeTask` run has no such durable guard,
//! so this adapter does not offer it.
//!
//! Invalidation: after every delivered event, preparation results whose task
//! is no longer active at the generation they were made for are dropped. M2
//! returns consumers of a failed, cancelled, restarted or changed producer to
//! `Blocked` at a new generation (or `Aborted`), so their results go. A
//! result is also refused, not stored, when the task moved while the
//! operation ran. Results are never journaled: a restart discards them all.
//!
//! Resources: preparation runs on the caller's thread without the adapter
//! mutex, the compose home lock or the runtime spine lock. Only a
//! `Provisional` task with a live lease may prepare, and at most one
//! preparation per task is in flight, so concurrent preparations are bounded
//! by the M2 permits (`ResourceLimits::max_concurrency`). With
//! `ProvisionalConfig::enabled` false (the default) M2 dispatches a consumer
//! only after every producer is final, which is the sequential order.
//!
//! Not in scope: aien-trace is not used (observation is optional); no new
//! receipt type; AEGIS, World, KV and inference scheduling are unchanged.
//! spark-harness gains only the M2 `Committing` state and the
//! `BeginCommit`, `CommitOutcome` and `BindApproval` events.

use crate::approved::{ApprovedComposeReport, ApprovedProposal, ApprovedRefusal, ProposerHook};
use crate::compose::ComposeBridge;
use crate::control::{ControlResponse, DeskProof};
use crate::effects::MintRequest;
use aien_proof::lock::FileLock;
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use spark_harness::completion::CONSUMED_DIR;
use spark_harness::execution::TaskState;
use spark_harness::readiness::{
    verify_final, ArtifactReadiness, OperationClass, ReadinessLevel, INTERFACE_DIR,
};
use spark_harness::scheduler::{
    Clock, CommitResult, Event, EventKind, Outcome, PermitPool, ProvisionalConfig, SchedError,
    Scheduler,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};

/// The whitelist of early preparation operations. One entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrepOperation {
    /// `compose::rederive::task_decision(goal, workspace)`: the compose
    /// prompt and destination a goal would get. Reads the workspace only.
    ComposePlanPreview,
}

impl PrepOperation {
    /// The M2 operation class the scheduler gates this operation as.
    pub fn class(self) -> OperationClass {
        match self {
            PrepOperation::ComposePlanPreview => OperationClass::ReadOnlyAnalysis,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PrepOperation::ComposePlanPreview => "compose_plan_preview",
        }
    }
}

/// One producer artifact a result or a fence decision rests on, copied from
/// a store-verified [`ArtifactReadiness`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadinessBinding {
    pub producer_task: String,
    pub source_commit: String,
    pub producer_generation: u64,
    pub artifact_digest: String,
    pub receipt_id: String,
    pub level: ReadinessLevel,
}

impl From<&ArtifactReadiness> for ReadinessBinding {
    fn from(r: &ArtifactReadiness) -> Self {
        ReadinessBinding {
            producer_task: r.producer_task().to_string(),
            source_commit: r.source_commit().to_string(),
            producer_generation: r.producer_generation(),
            artifact_digest: r.artifact_digest().to_string(),
            receipt_id: r.receipt_id().to_string(),
            level: r.level(),
        }
    }
}

/// A provisional preparation result. Built only by [`PrepAdapter::prepare`];
/// no public constructor and no `Deserialize`. It authorizes nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PreparationResult {
    task: String,
    generation: u64,
    operation: PrepOperation,
    class: OperationClass,
    plan_sha256: String,
    destination: Option<String>,
    bindings: Vec<ReadinessBinding>,
}

impl PreparationResult {
    pub fn task(&self) -> &str {
        &self.task
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn operation(&self) -> PrepOperation {
        self.operation
    }
    pub fn class(&self) -> OperationClass {
        self.class
    }
    /// sha256 hex of the compose prompt the goal would get.
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn destination(&self) -> Option<&str> {
        self.destination.as_deref()
    }
    pub fn bindings(&self) -> &[ReadinessBinding] {
        &self.bindings
    }
    /// True when at least one producer was only interface-verified.
    pub fn is_provisional(&self) -> bool {
        self.bindings
            .iter()
            .any(|b| b.level != ReadinessLevel::FinalVerified)
    }
}

/// What the fence verified, inside the same critical section as the call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FenceProof {
    pub task: String,
    pub generation: u64,
    /// The task's own M1 completion.
    pub completion: ReadinessBinding,
    /// Every producer, final-verified (M2 `may_finalize`).
    pub producers: Vec<ReadinessBinding>,
}

/// The existing call's own answer, plus the fence proof it ran under. The
/// answer may still be a refusal: the fence does not decide authorization.
#[derive(Debug)]
pub struct Fenced<T> {
    pub proof: FenceProof,
    pub outcome: T,
    /// How the M2 task ended (`CommitOutcome`), classified from `outcome`.
    pub result: CommitResult,
    /// Set when journaling the outcome failed: the task then stays
    /// `Committing` (never re-run) until a restart marks it
    /// `CommitUncertain`.
    pub record_error: Option<SchedError>,
    /// Deferred events delivered again right after the outcome.
    pub redelivered: Vec<Redelivered>,
    /// Set when rewriting the deferred sidecar after the drain failed. The
    /// sidecar then still lists events M2 already applied; replaying them
    /// on open is a duplicate, or a stale generation that stays queued
    /// until discarded. The next rewrite replaces it.
    pub deferred_error: Option<String>,
}

/// Most events [`PrepAdapter::deliver`] holds back while commits run.
pub const DEFERRED_CAPACITY: usize = 64;

/// Largest deferred sidecar [`PrepAdapter::open`] reads, in bytes.
pub const DEFERRED_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Suffix of the deferred-event sidecar, appended to the journal's full
/// file name. A journal whose name ends with it is refused by
/// [`PrepAdapter::open`], so no journal is another journal's sidecar.
pub const DEFERRED_SUFFIX: &str = ".deferred";

/// The deferred-event sidecar of the M2 journal `journal`:
/// `<journal>.deferred`. Distinct journal paths give distinct sidecars.
pub fn deferred_journal_path(journal: &Path) -> PathBuf {
    let mut name = journal.as_os_str().to_os_string();
    name.push(DEFERRED_SUFFIX);
    PathBuf::from(name)
}

/// What [`PrepAdapter::deliver`] did with an event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivered {
    /// M2 applied it, or recognised a duplicate.
    Applied(Outcome),
    /// M2 refused it because `committing` is committing. It is queued (at
    /// position `queued`, from 1) and delivered again right after that
    /// commit's outcome is recorded.
    Deferred { committing: String, queued: usize },
}

#[derive(Debug)]
pub enum DeliverError {
    Scheduler(SchedError),
    /// `BeginCommit`, `CommitOutcome` and `BindApproval` go only through the
    /// fence and [`PrepAdapter::bind_approval`].
    FenceOnly(&'static str),
    /// The deferred queue is full: the event was not queued and not
    /// applied; deliver it again later.
    DeferredFull {
        committing: String,
        capacity: usize,
    },
    /// The deferred sidecar could not be written: the event was not queued
    /// and not applied; deliver it again later.
    DeferredJournal(String),
}

impl fmt::Display for DeliverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeliverError::Scheduler(e) => write!(f, "DELIVER_REFUSED Scheduler: {e}"),
            DeliverError::FenceOnly(k) => write!(
                f,
                "DELIVER_REFUSED FenceOnly: {k} events are delivered only by the fence"
            ),
            DeliverError::DeferredFull {
                committing,
                capacity,
            } => write!(
                f,
                "DELIVER_REFUSED DeferredFull: task '{committing}' is committing and {capacity} events already wait; nothing was queued"
            ),
            DeliverError::DeferredJournal(e) => write!(
                f,
                "DELIVER_REFUSED DeferredJournal: the deferred queue could not be journaled ({e}); nothing was queued"
            ),
        }
    }
}

impl std::error::Error for DeliverError {}

/// One deferred event delivered again after a commit outcome.
#[derive(Debug)]
pub struct Redelivered {
    pub id: String,
    pub result: Result<Outcome, SchedError>,
}

#[derive(Debug)]
pub enum PrepError {
    Scheduler(SchedError),
    /// The caller's generation is not the task's current one.
    StaleGeneration {
        task: String,
        expected: u64,
        current: Option<u64>,
    },
    /// A preparation for this task is already running.
    Busy(String),
    /// The task moved (revoked, cancelled, re-dispatched) while the
    /// operation ran; the result was discarded.
    Revoked {
        task: String,
        detail: String,
    },
    /// The operation itself refused (for example a workspace escape).
    Operation(String),
}

impl fmt::Display for PrepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrepError::Scheduler(e) => write!(f, "PREP_REFUSED Scheduler: {e}"),
            PrepError::StaleGeneration {
                task,
                expected,
                current,
            } => write!(
                f,
                "PREP_REFUSED StaleGeneration: task '{task}' asked at generation {expected}, scheduler has {current:?}"
            ),
            PrepError::Busy(t) => write!(f, "PREP_REFUSED Busy: task '{t}' is already preparing"),
            PrepError::Revoked { task, detail } => {
                write!(f, "PREP_REFUSED Revoked: task '{task}': {detail}; result discarded")
            }
            PrepError::Operation(e) => write!(f, "PREP_REFUSED Operation: {e}"),
        }
    }
}

impl std::error::Error for PrepError {}

#[derive(Debug)]
pub enum FenceError {
    /// M2 refused (`may_finalize`, unknown task).
    Scheduler(SchedError),
    StaleGeneration {
        task: String,
        expected: u64,
        current: Option<u64>,
    },
    NotCommittable {
        task: String,
        state: TaskState,
    },
    /// The task has no accepted M1 completion for its current generation.
    NoCompletion(String),
    /// The task's own M1 completion receipt no longer verifies.
    CompletionInvalid {
        task: String,
        error: String,
    },
    /// This process already ran the fenced call for this task generation.
    AlreadySubmitted {
        task: String,
        generation: u64,
    },
    StoreLock(String),
}

impl fmt::Display for FenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenceError::Scheduler(e) => write!(f, "FENCE_REFUSED Scheduler: {e}"),
            FenceError::StaleGeneration {
                task,
                expected,
                current,
            } => write!(
                f,
                "FENCE_REFUSED StaleGeneration: task '{task}' asked at generation {expected}, scheduler has {current:?}"
            ),
            FenceError::NotCommittable { task, state } => write!(
                f,
                "FENCE_REFUSED NotCommittable: task '{task}' is {}",
                state.as_str()
            ),
            FenceError::NoCompletion(t) => {
                write!(f, "FENCE_REFUSED NoCompletion: task '{t}' has no accepted M1 completion")
            }
            FenceError::CompletionInvalid { task, error } => {
                write!(f, "FENCE_REFUSED CompletionInvalid: task '{task}': {error}")
            }
            FenceError::AlreadySubmitted { task, generation } => write!(
                f,
                "FENCE_REFUSED AlreadySubmitted: task '{task}' generation {generation} already ran its fenced call"
            ),
            FenceError::StoreLock(e) => write!(f, "FENCE_REFUSED StoreLock: {e}"),
        }
    }
}

impl std::error::Error for FenceError {}

#[cfg(feature = "test-support")]
type PrepHook = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

struct Inner {
    sched: Scheduler,
    prepared: BTreeMap<String, PreparationResult>,
    in_flight: BTreeSet<String>,
    submitted: BTreeSet<(String, u64)>,
    /// Events M2 refused with `CommitInProgress`, oldest first.
    deferred: VecDeque<Event>,
    /// Where `deferred` is journaled (journaled schedulers only).
    deferred_path: Option<PathBuf>,
    /// What replaying the journaled deferred events on open did.
    replayed: Vec<Redelivered>,
}

impl Inner {
    fn current_generation(&self, task: &str, expected: u64) -> Result<(), Option<u64>> {
        match self.sched.generation(task) {
            Some(g) if g == expected => Ok(()),
            other => Err(other),
        }
    }

    /// Drop every result whose task is no longer active at its generation.
    fn prune(&mut self) {
        let sched = &self.sched;
        self.prepared.retain(|task, r| {
            sched
                .task(task)
                .is_some_and(|t| t.state.is_active() && t.generation == r.generation)
        });
    }

    /// Deliver every deferred event again, oldest first. Only an event M2
    /// applied or answered as a duplicate leaves the queue; one still
    /// refused by a running commit stays queued silently, and one refused
    /// for any other reason stays queued and is returned with its refusal.
    /// The sidecar is rewritten when anything left the queue; the error, if
    /// that failed, is returned too.
    fn redeliver(&mut self) -> (Vec<Redelivered>, Option<String>) {
        let mut out = Vec::new();
        let mut keep = VecDeque::new();
        let before = self.deferred.len();
        while let Some(ev) = self.deferred.pop_front() {
            match self.sched.deliver(ev.clone()) {
                Err(SchedError::CommitInProgress { .. }) => keep.push_back(ev),
                Ok(o) => out.push(Redelivered {
                    id: ev.id,
                    result: Ok(o),
                }),
                Err(e) => {
                    out.push(Redelivered {
                        id: ev.id.clone(),
                        result: Err(e),
                    });
                    keep.push_back(ev);
                }
            }
        }
        self.deferred = keep;
        self.prune();
        let error = if self.deferred.len() == before {
            None
        } else {
            self.persist_deferred().err()
        };
        (out, error)
    }

    /// Rewrite the sidecar from `deferred` (no-op without a journal).
    fn persist_deferred(&self) -> Result<(), String> {
        let Some(path) = &self.deferred_path else {
            return Ok(());
        };
        let mut buf = String::new();
        for ev in &self.deferred {
            buf.push_str(&serde_json::to_string(ev).map_err(|e| e.to_string())?);
            buf.push('\n');
        }
        write_atomic(path, buf.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Temp file in the same directory, fsync, rename over `path`, fsync the
/// directory. The temp file is removed on failure.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sidecar has no parent dir",
        )
    })?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        std::fs::File::open(dir)?.sync_all()
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Read the deferred sidecar: absent is empty; anything that does not
/// parse, is too large, holds too many events or holds a fence-only event
/// is an error.
fn read_deferred(path: &Path) -> Result<VecDeque<Event>, SchedError> {
    let bad = |what: String| SchedError::Journal(format!("{}: {what}", path.display()));
    let len = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(VecDeque::new()),
        Err(e) => return Err(bad(e.to_string())),
    };
    if len > DEFERRED_MAX_BYTES {
        return Err(bad(format!(
            "{len} bytes, more than the {DEFERRED_MAX_BYTES} a deferred queue may hold"
        )));
    }
    let text = std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?;
    let mut out = VecDeque::new();
    for (n, line) in text.lines().enumerate() {
        if out.len() == DEFERRED_CAPACITY {
            return Err(bad(format!(
                "more than {DEFERRED_CAPACITY} deferred events"
            )));
        }
        let ev: Event =
            serde_json::from_str(line).map_err(|e| bad(format!("line {}: {e}", n + 1)))?;
        if fence_only(&ev.kind).is_some() {
            return Err(bad(format!("line {}: a fence-only event", n + 1)));
        }
        out.push_back(ev);
    }
    Ok(out)
}

/// `journal` with its directory (created if missing) and, when it exists,
/// the file itself resolved to canonical form, so every path to one journal
/// names the same lock file and sidecar. A name ending with
/// [`DEFERRED_SUFFIX`] is refused.
fn canonical_journal(journal: &Path) -> Result<PathBuf, SchedError> {
    let bad = |what: String| SchedError::Journal(format!("{}: {what}", journal.display()));
    let name = journal
        .file_name()
        .ok_or_else(|| bad("a journal path needs a file name".to_string()))?;
    let dir = journal
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| bad(e.to_string()))?;
    let dir = std::fs::canonicalize(dir).map_err(|e| bad(e.to_string()))?;
    let path = match std::fs::canonicalize(dir.join(name)) {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => dir.join(name),
        Err(e) => return Err(bad(e.to_string())),
    };
    if path
        .as_os_str()
        .to_string_lossy()
        .ends_with(DEFERRED_SUFFIX)
    {
        return Err(bad(format!(
            "a journal name may not end with {DEFERRED_SUFFIX}, the deferred sidecar suffix"
        )));
    }
    Ok(path)
}

/// The fence's own event kinds, which only the fence delivers.
fn fence_only(kind: &EventKind) -> Option<&'static str> {
    match kind {
        EventKind::BeginCommit { .. } => Some("begin_commit"),
        EventKind::CommitOutcome { .. } => Some("commit_outcome"),
        EventKind::BindApproval { .. } => Some("bind_approval"),
        _ => None,
    }
}

/// The adapter: owns the M2 scheduler for engineering tasks and fences the
/// existing compose commit calls. Share it behind an `Arc`; every method
/// takes `&self`.
pub struct PrepAdapter {
    inner: Mutex<Inner>,
    store: PathBuf,
    #[cfg(feature = "test-support")]
    prep_hook: Mutex<Option<PrepHook>>,
    #[cfg(feature = "test-support")]
    begin_hook: Mutex<Option<PrepHook>>,
}

impl PrepAdapter {
    fn with_scheduler(sched: Scheduler, store: &Path, deferred_path: Option<PathBuf>) -> Self {
        PrepAdapter {
            inner: Mutex::new(Inner {
                sched,
                prepared: BTreeMap::new(),
                in_flight: BTreeSet::new(),
                submitted: BTreeSet::new(),
                deferred: VecDeque::new(),
                deferred_path,
                replayed: Vec::new(),
            }),
            store: store.to_path_buf(),
            #[cfg(feature = "test-support")]
            prep_hook: Mutex::new(None),
            #[cfg(feature = "test-support")]
            begin_hook: Mutex::new(None),
        }
    }

    /// In-memory scheduler over the harness store `store`
    /// (`HarnessEngine::completion_store`).
    pub fn new(cfg: ProvisionalConfig, store: &Path, clock: Clock) -> Result<Self, SchedError> {
        Ok(Self::with_scheduler(
            Scheduler::new(cfg, store, clock)?,
            store,
            None,
        ))
    }

    /// Journaled scheduler: create or replay `journal`, then run the M2
    /// audit before anything can use it (it settles every interrupted
    /// commit as `CommitUncertain`), then deliver the journaled deferred
    /// events again ([`PrepAdapter::take_replayed_deferred`]). Returns the
    /// tasks the audit revoked. No preparation result survives: they are
    /// never journaled.
    pub fn open(
        cfg: ProvisionalConfig,
        store: &Path,
        journal: &Path,
        clock: Clock,
    ) -> Result<(Self, Vec<String>), SchedError> {
        let journal = canonical_journal(journal)?;
        let (unaudited, _report) = Scheduler::open(cfg, store, &journal, clock)?;
        // The M2 journal lock is held from here on, for the adapter's life:
        // no other adapter reads or writes this sidecar.
        let path = deferred_journal_path(&journal);
        let deferred = read_deferred(&path)?;
        let (sched, revoked) = unaudited.audit()?;
        let adapter = Self::with_scheduler(sched, store, Some(path));
        {
            let mut g = adapter.inner.lock();
            g.deferred = deferred;
            // No commit is running after the audit, so every event is
            // delivered; the refused ones stay queued, and the sidecar is
            // rewritten if anything left. If that rewrite fails the open
            // fails: the next open replays the same events. M2 answers the
            // ones already applied as duplicates, or, once their generation
            // moved on, refuses them as stale; refused ones stay queued.
            let (replayed, error) = g.redeliver();
            if let Some(e) = error {
                return Err(SchedError::Journal(e));
            }
            g.replayed = replayed;
        }
        Ok((adapter, revoked))
    }

    /// What delivering the journaled deferred events on open did, oldest
    /// first. Taken once; later calls return nothing.
    pub fn take_replayed_deferred(&self) -> Vec<Redelivered> {
        std::mem::take(&mut self.inner.lock().replayed)
    }

    /// The runtime enable flag: M2 `ProvisionalConfig::enabled` (default
    /// false, which is sequential dispatch).
    pub fn enabled(&self) -> bool {
        self.inner.lock().sched.config().enabled
    }

    /// Deliver one M2 event, then drop the preparation results it
    /// invalidated. An event M2 refuses because a commit is running is
    /// queued and delivered again after that commit's outcome (see the
    /// module docs). The fence's own event kinds are refused here.
    pub fn deliver(&self, event: Event) -> Result<Delivered, DeliverError> {
        if let Some(kind) = fence_only(&event.kind) {
            return Err(DeliverError::FenceOnly(kind));
        }
        let mut g = self.inner.lock();
        let r = g.sched.deliver(event.clone());
        g.prune();
        match r {
            Ok(o) => Ok(Delivered::Applied(o)),
            Err(SchedError::CommitInProgress { committing, .. }) => {
                if let Some(i) = g.deferred.iter().position(|e| e.id == event.id) {
                    return if g.deferred[i] == event {
                        Ok(Delivered::Deferred {
                            committing,
                            queued: i + 1,
                        })
                    } else {
                        Err(DeliverError::Scheduler(SchedError::EventIdConflict(
                            event.id,
                        )))
                    };
                }
                if g.deferred.len() >= DEFERRED_CAPACITY {
                    return Err(DeliverError::DeferredFull {
                        committing,
                        capacity: DEFERRED_CAPACITY,
                    });
                }
                g.deferred.push_back(event);
                // Journaled before `Deferred` is returned, so a crash
                // before the drain still delivers it on the next open.
                if let Err(e) = g.persist_deferred() {
                    g.deferred.pop_back();
                    return Err(DeliverError::DeferredJournal(e));
                }
                Ok(Delivered::Deferred {
                    committing,
                    queued: g.deferred.len(),
                })
            }
            Err(e) => Err(DeliverError::Scheduler(e)),
        }
    }

    /// How many deferred events are queued: waiting for a commit outcome,
    /// or refused on redelivery and kept until discarded.
    pub fn deferred_len(&self) -> usize {
        self.inner.lock().deferred.len()
    }

    /// The queued deferred events, oldest first.
    pub fn deferred_events(&self) -> Vec<Event> {
        self.inner.lock().deferred.iter().cloned().collect()
    }

    /// Remove the queued deferred event `id` (the caller's decision about
    /// an event M2 keeps refusing) and rewrite the sidecar. `Ok(None)` when
    /// no such event is queued. If the rewrite fails the event stays queued
    /// and the error is returned.
    pub fn discard_deferred(&self, id: &str) -> Result<Option<Event>, DeliverError> {
        let mut g = self.inner.lock();
        let Some(i) = g.deferred.iter().position(|e| e.id == id) else {
            return Ok(None);
        };
        let ev = g.deferred.remove(i).expect("position is in range");
        if let Err(e) = g.persist_deferred() {
            g.deferred.insert(i, ev);
            return Err(DeliverError::DeferredJournal(e));
        }
        Ok(Some(ev))
    }

    /// Read-only access to the scheduler (state, generations, invariants,
    /// timing trace). Holds the adapter mutex while `f` runs.
    pub fn inspect<T>(&self, f: impl FnOnce(&Scheduler) -> T) -> T {
        f(&self.inner.lock().sched)
    }

    pub fn state(&self, task: &str) -> Option<TaskState> {
        self.inner.lock().sched.state(task)
    }

    pub fn generation(&self, task: &str) -> Option<u64> {
        self.inner.lock().sched.generation(task)
    }

    pub fn permits_in_use(&self) -> PermitPool {
        self.inner.lock().sched.permits_in_use()
    }

    /// The task's current preparation result, if it is still valid.
    pub fn prepared(&self, task: &str) -> Option<PreparationResult> {
        self.inner.lock().prepared.get(task).cloned()
    }

    /// Test seam: called between the admission check and the operation,
    /// outside the adapter mutex.
    #[cfg(feature = "test-support")]
    pub fn set_prep_hook(&self, hook: Option<PrepHook>) {
        *self.prep_hook.lock() = hook;
    }

    /// Run the whitelisted operation for `task` at `generation`. Admission:
    /// current generation, no preparation in flight for the task, and M2
    /// `may_start(task, ReadOnlyAnalysis)` (a `Provisional` task with a live
    /// lease, every producer bound at its current generation and
    /// store-verified at a level that permits the class). The operation runs
    /// outside the mutex; the result is stored only if the task is still at
    /// the same generation and `may_start` passes again, both checked under
    /// the mutex that stores it.
    pub fn prepare(
        &self,
        task: &str,
        generation: u64,
        goal: &str,
        workspace: &Path,
    ) -> Result<PreparationResult, PrepError> {
        let op = PrepOperation::ComposePlanPreview;
        {
            let mut g = self.inner.lock();
            g.current_generation(task, generation).map_err(|current| {
                PrepError::StaleGeneration {
                    task: task.to_string(),
                    expected: generation,
                    current,
                }
            })?;
            if g.in_flight.contains(task) {
                return Err(PrepError::Busy(task.to_string()));
            }
            g.sched
                .may_start(task, op.class())
                .map_err(PrepError::Scheduler)?;
            g.in_flight.insert(task.to_string());
        }
        #[cfg(feature = "test-support")]
        {
            let hook = self.prep_hook.lock().clone();
            if let Some(h) = hook {
                h(task);
            }
        }
        let out = plan_preview(goal, workspace);
        let mut g = self.inner.lock();
        g.in_flight.remove(task);
        let (plan_sha256, destination) = out.map_err(PrepError::Operation)?;
        let revoked = |detail: String| PrepError::Revoked {
            task: task.to_string(),
            detail,
        };
        g.current_generation(task, generation).map_err(|current| {
            revoked(format!(
                "generation moved from {generation} to {current:?} during preparation"
            ))
        })?;
        // No separate binding comparison: M2 sets a task's bindings only when
        // it dispatches it, and leaving `Provisional` always moves the task to
        // a new generation first, so an unchanged generation means unchanged
        // bindings. The readiness recorded below is the fresh one.
        let readiness = g
            .sched
            .may_start(task, op.class())
            .map_err(|e| revoked(e.to_string()))?;
        let result = PreparationResult {
            task: task.to_string(),
            generation,
            operation: op,
            class: op.class(),
            plan_sha256,
            destination,
            bindings: readiness.iter().map(ReadinessBinding::from).collect(),
        };
        g.prepared.insert(task.to_string(), result.clone());
        Ok(result)
    }

    /// The harness store locks every interface and completion writer takes.
    fn store_locks(&self) -> Result<[FileLock; 2], FenceError> {
        let lock = |dir: &str| {
            let d = self.store.join(dir);
            std::fs::create_dir_all(&d)
                .and_then(|_| FileLock::acquire(&d.join(".lock")))
                .map_err(|e| FenceError::StoreLock(format!("{}: {e}", d.display())))
        };
        Ok([lock(INTERFACE_DIR)?, lock(CONSUMED_DIR)?])
    }

    /// The fence (see the module docs). Under the adapter mutex and the store
    /// locks: every check, then the journaled M2 `BeginCommit`, which moves
    /// the task to `Committing` and freezes its inputs. Both locks are
    /// released before `commit` runs. Its result, classified by `classify`,
    /// is journaled as the M2 `CommitOutcome`, and the deferred events are
    /// delivered again. A panic in `commit` is recorded as `Uncertain` and
    /// then resumed.
    fn fenced<T>(
        &self,
        task: &str,
        generation: u64,
        commit: impl FnOnce(&FenceProof) -> T,
        classify: impl FnOnce(&T) -> CommitResult,
    ) -> Result<Fenced<T>, FenceError> {
        let proof = self.begin_commit(task, generation)?;
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| commit(&proof)));
        let result = match &ran {
            Ok(v) => classify(v),
            Err(_) => CommitResult::Uncertain {
                reason: "the fenced call panicked".to_string(),
            },
        };
        let (recorded, redelivered, deferred_error) =
            self.record_outcome(task, generation, result.clone());
        match ran {
            Ok(outcome) => Ok(Fenced {
                proof,
                outcome,
                result,
                record_error: recorded.err(),
                redelivered,
                deferred_error,
            }),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Check and transition, atomically: the fence checks and the journaled
    /// `BeginCommit` under the adapter mutex and both store locks.
    fn begin_commit(&self, task: &str, generation: u64) -> Result<FenceProof, FenceError> {
        let mut g = self.inner.lock();
        let _store = self.store_locks()?;
        let key = (task.to_string(), generation);
        let submitted = || FenceError::AlreadySubmitted {
            task: task.to_string(),
            generation,
        };
        if g.submitted.contains(&key) {
            return Err(submitted());
        }
        g.current_generation(task, generation)
            .map_err(|current| FenceError::StaleGeneration {
                task: task.to_string(),
                expected: generation,
                current,
            })?;
        let entry = g
            .sched
            .task(task)
            .ok_or_else(|| FenceError::Scheduler(SchedError::UnknownTask(task.to_string())))?;
        if entry.state != TaskState::Committable {
            return Err(FenceError::NotCommittable {
                task: task.to_string(),
                state: entry.state,
            });
        }
        let claim = entry
            .completion
            .clone()
            .ok_or_else(|| FenceError::NoCompletion(task.to_string()))?;
        let mut inputs = BTreeMap::new();
        for p in entry.bindings.keys() {
            let c = g.sched.task(p).and_then(|t| t.completion.clone());
            inputs.insert(
                p.clone(),
                c.ok_or_else(|| FenceError::NoCompletion(p.clone()))?,
            );
        }
        let producers = g.sched.may_finalize(task).map_err(FenceError::Scheduler)?;
        let own = verify_final(&self.store, task, generation, &claim).map_err(|e| {
            FenceError::CompletionInvalid {
                task: task.to_string(),
                error: e.to_string(),
            }
        })?;
        #[cfg(feature = "test-support")]
        if let Some(hook) = self.begin_hook.lock().clone() {
            hook(task);
        }
        let begin = Event {
            id: event_id("begin", task, generation),
            kind: EventKind::BeginCommit {
                task: task.to_string(),
                generation,
                completion: claim,
                producers: inputs,
            },
        };
        // M2 re-runs `may_finalize` and `verify_final` here
        // (still under both locks), checks the inputs equal its state,
        // refuses a second commit in the same chain, and journals the
        // transition.
        match g.sched.deliver(begin).map_err(FenceError::Scheduler)? {
            Outcome::Applied(_) => {}
            Outcome::Duplicate => return Err(submitted()),
        }
        // Spent before the call: a call that fails or panics is not retried
        // at this generation (fail closed).
        g.submitted.insert(key);
        g.prune();
        Ok(FenceProof {
            task: task.to_string(),
            generation,
            completion: ReadinessBinding::from(&own),
            producers: producers.iter().map(ReadinessBinding::from).collect(),
        })
    }

    /// Journal the commit outcome, then deliver the deferred events again,
    /// under one lock. If journaling fails the task stays `Committing`
    /// (frozen, never re-run) until a restart settles it as
    /// `CommitUncertain`, and the deferred events refused by it stay queued
    /// (in the sidecar too, so that open delivers them after the settle).
    fn record_outcome(
        &self,
        task: &str,
        generation: u64,
        outcome: CommitResult,
    ) -> (Result<(), SchedError>, Vec<Redelivered>, Option<String>) {
        let mut g = self.inner.lock();
        let r = g.sched.deliver(Event {
            id: event_id("outcome", task, generation),
            kind: EventKind::CommitOutcome {
                task: task.to_string(),
                generation,
                outcome,
            },
        });
        g.prune();
        let (redelivered, deferred_error) = g.redeliver();
        (r.map(|_| ()), redelivered, deferred_error)
    }

    /// Fenced `ComposeApprovedProposal`: the existing proposer hook (verify,
    /// desk MAC, replay claim, compose run with AEGIS verify and World
    /// commit, grant) runs unchanged once the fence passes.
    pub fn submit_approved(
        &self,
        task: &str,
        generation: u64,
        hook: &ProposerHook,
        proposal: &ApprovedProposal,
        workspace: &str,
    ) -> Result<Fenced<Result<ApprovedComposeReport, Box<ApprovedRefusal>>>, FenceError> {
        self.fenced(
            task,
            generation,
            |_| hook.submit(proposal, workspace),
            classify_approved,
        )
    }

    /// Fenced `ComposeAuthorize`, the grant against a compose commit an
    /// unfenced `RunComposeTask` recorded: the existing grant minting (desk
    /// MAC, compose-commit record match, confinement, nonce and grant
    /// replay rules) runs unchanged once the fence passes.
    pub fn authorize(
        &self,
        task: &str,
        generation: u64,
        bridge: &ComposeBridge,
        req: &MintRequest,
        desk_proof: Option<&DeskProof>,
        objective_id: Option<&str>,
    ) -> Result<Fenced<ControlResponse>, FenceError> {
        self.fenced(
            task,
            generation,
            |_| crate::effects::authorize_with_objective(bridge, req, desk_proof, objective_id),
            classify_authorize,
        )
    }

    /// Bind a control request to the M2 task generation whose fence it must
    /// pass, as a journaled M2 `BindApproval`, so the binding survives a
    /// restart. Only bound requests are fenced; the daemon runs every other
    /// request exactly as before. A later binding for the same key replaces
    /// the earlier one. M2 refuses a stale generation, a task whose commit
    /// ended and a new key past the task's `MAX_APPROVAL_BINDINGS_PER_TASK`;
    /// it drops the binding only when the task's commit ends (an aborted or
    /// retried task keeps it, see the module docs).
    pub fn bind_approval(
        &self,
        key: FenceKey,
        task: &str,
        generation: u64,
    ) -> Result<Outcome, SchedError> {
        let encoded = key.encoded();
        // Task ids hold no control characters, so the NUL separator keeps
        // (key, task) pairs distinct.
        let id = event_id("bind", &format!("{encoded}\0{task}"), generation);
        let mut g = self.inner.lock();
        let r = g.sched.deliver(Event {
            id,
            kind: EventKind::BindApproval {
                key: encoded,
                task: task.to_string(),
                generation,
            },
        });
        g.prune();
        r
    }

    /// The task generation `key` is bound to, if any (M2 state, so it
    /// includes bindings replayed from the journal).
    pub fn binding(&self, key: &FenceKey) -> Option<(String, u64)> {
        let g = self.inner.lock();
        let b = g.sched.approval(&key.encoded())?;
        Some((b.task.clone(), b.generation))
    }

    /// Daemon entry for `ComposeApprovedProposal`. None when the approval is
    /// not bound (the caller runs the original call). A fence refusal is a
    /// `FENCE_REFUSED` error and nothing runs.
    pub fn route_approved(
        &self,
        bridge: &std::sync::Arc<ComposeBridge>,
        proposal: &ApprovedProposal,
        workspace: &str,
    ) -> Option<ControlResponse> {
        let (task, generation) = self.binding(&FenceKey::Approval(proposal.approval_id.clone()))?;
        let hook = ProposerHook::new(bridge.clone());
        Some(
            match self.submit_approved(&task, generation, &hook, proposal, workspace) {
                Ok(f) => match f.outcome {
                    Ok(r) => ControlResponse::ComposeApprovedResult(Box::new(r)),
                    Err(r) => ControlResponse::ComposeApprovedRefused(r),
                },
                Err(e) => ControlResponse::Error(e.to_string()),
            },
        )
    }

    /// Daemon entry for `ComposeAuthorize`, keyed by the proposal digest.
    /// None when it is not bound (the caller runs the original call).
    pub fn route_authorize(
        &self,
        bridge: &ComposeBridge,
        req: &MintRequest,
        desk_proof: Option<&DeskProof>,
        objective_id: Option<&str>,
    ) -> Option<ControlResponse> {
        let (task, generation) = self.binding(&FenceKey::Proposal(req.proposal_sha256.clone()))?;
        Some(
            match self.authorize(&task, generation, bridge, req, desk_proof, objective_id) {
                Ok(f) => f.outcome,
                Err(e) => ControlResponse::Error(e.to_string()),
            },
        )
    }

    /// Test seam: the fence around an arbitrary call, so a test can act
    /// while the call runs. `classify` maps its result to the outcome M2
    /// records.
    #[cfg(feature = "test-support")]
    pub fn fenced_for_tests<T>(
        &self,
        task: &str,
        generation: u64,
        commit: impl FnOnce(&FenceProof) -> T,
        classify: impl FnOnce(&T) -> CommitResult,
    ) -> Result<Fenced<T>, FenceError> {
        self.fenced(task, generation, commit, classify)
    }

    /// Test seam: called inside `begin_commit` after every check and before
    /// the journaled `BeginCommit`, with the adapter mutex and both store
    /// locks held.
    #[cfg(feature = "test-support")]
    pub fn set_begin_hook(&self, hook: Option<PrepHook>) {
        *self.begin_hook.lock() = hook;
    }
}

/// Which control request a binding fences.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FenceKey {
    /// `ComposeApprovedProposal`, by `ApprovedProposal::approval_id`.
    Approval(String),
    /// `ComposeAuthorize`, by `MintRequest::proposal_sha256`.
    Proposal(String),
}

impl FenceKey {
    /// The M2 binding key: the request kind and its id.
    pub fn encoded(&self) -> String {
        match self {
            FenceKey::Approval(id) => format!("approval:{id}"),
            FenceKey::Proposal(sha) => format!("proposal:{sha}"),
        }
    }
}

/// A scheduler event id for this task generation: a fixed prefix and a
/// digest, so any task id fits the M2 identifier limit.
fn event_id(kind: &str, task: &str, generation: u64) -> String {
    let mut h = Sha256::new();
    h.update(task.as_bytes());
    h.update([0]);
    h.update(generation.to_be_bytes());
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    format!("m3-{kind}-{hex}")
}

/// What an approved submission did. Ok: committed (`COMMITTED`, or
/// `ALREADY_COMMITTED` for an exact retry). A refusal without a replay
/// claim came before `approved_replay::claim` (approved.rs `submit`:
/// verify, desk MAC, confinement, bound requirements): nothing durable.
/// `NotCommitted`: the ledger recorded `fail` (the World did not commit).
/// Anything else after a claim may have reached the World: uncertain.
pub fn classify_approved(r: &Result<ApprovedComposeReport, Box<ApprovedRefusal>>) -> CommitResult {
    match r {
        Ok(report) => CommitResult::Committed {
            detail: report.state.clone(),
        },
        Err(e) if e.replay_claim.is_none() || e.name == "NotCommitted" => CommitResult::Failed {
            reason: e.to_string(),
        },
        Err(e) => CommitResult::Uncertain {
            reason: e.to_string(),
        },
    }
}

/// What `ComposeAuthorize` did. `ComposeNoted`: the grant was written.
/// `EFFECT_REFUSED`: refused by name before the only write (effects.rs
/// `authorize_inner`, every refusal precedes its `append`): nothing durable.
/// Any other error (for example a failed `append` or re-read) may have
/// written: uncertain.
pub fn classify_authorize(r: &ControlResponse) -> CommitResult {
    match r {
        ControlResponse::ComposeNoted(n) => CommitResult::Committed {
            detail: format!("grant {}", n.id),
        },
        ControlResponse::Error(e) if e.starts_with("EFFECT_REFUSED ") => {
            CommitResult::Failed { reason: e.clone() }
        }
        other => CommitResult::Uncertain {
            reason: format!("{other:?}").chars().take(512).collect(),
        },
    }
}

/// The whitelisted operation: compose prompt digest and destination.
fn plan_preview(goal: &str, workspace: &Path) -> Result<(String, Option<String>), String> {
    let ((prompt, _edit, _kind), destination) =
        crate::compose::rederive::task_decision(goal, workspace)?;
    let digest = Sha256::digest(prompt.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Ok((hex, destination))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_whitelist_is_one_reversible_preparation_class() {
        let op = PrepOperation::ComposePlanPreview;
        assert_eq!(op.class(), OperationClass::ReadOnlyAnalysis);
        assert!(op.class().is_reversible_preparation());
        assert!(ReadinessLevel::InterfaceVerified.permits(op.class()));
        assert_eq!(op.as_str(), "compose_plan_preview");
    }
}
