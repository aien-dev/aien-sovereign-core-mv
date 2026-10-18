//! Pure scheduler state and event application.
//!
//! [`Core::apply`] is deterministic: it reads only the current state, the
//! config and the journal entry (including the entry's recorded time). It
//! never touches the harness store or the clock. That is what lets a journal
//! replay rebuild the exact same state. Store checks happen before an event
//! is journaled (`gates.rs`) and again at every `may_start` / `may_finalize`.

use super::dedup::Seen;
use super::error::SchedError;
use super::permits::{PermitPool, ResourceRequest};
use super::timing::TaskTiming;
use super::ProvisionalConfig;
use crate::completion::{artifact_digest, check_identifier};
use crate::dependency::DependencyGraph;
use crate::execution::{check_transition, TaskState};
use crate::readiness::ProofClaim;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Everything the scheduler can be told. Producer proof events carry only a
/// [`ProofClaim`] pointer; the receipt itself is re-read from the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventKind {
    /// Register a task with its producers. A task with no producers is
    /// eligible at once, so dependencies are declared here, atomically.
    RegisterTask {
        task: String,
        resources: ResourceRequest,
        producers: Vec<String>,
    },
    /// Add a producer to a task that is still `Blocked`.
    AddDependency {
        producer: String,
        consumer: String,
    },
    InterfacePublished {
        task: String,
        generation: u64,
        claim: ProofClaim,
    },
    CompletionVerified {
        task: String,
        generation: u64,
        claim: ProofClaim,
    },
    /// The producer's body or interface changed and work restarts at a new,
    /// higher generation.
    NewGeneration {
        task: String,
        generation: u64,
    },
    ProducerFailed {
        task: String,
        generation: u64,
        reason: String,
    },
    /// The task's provisional work finished. `result_digest` names the
    /// provisional result, which is discarded on revocation.
    WorkDone {
        task: String,
        generation: u64,
        result_digest: String,
    },
    // The task-directed events below carry the task's current generation,
    // like the proof events above: a redelivery after the task moved on is
    // refused as stale, so duplicate detection never depends on a window.
    Finalize {
        task: String,
        generation: u64,
    },
    /// `beat` must rise with every heartbeat of one lease (start at 1), so
    /// a redelivered heartbeat is refused without remembering its id.
    Heartbeat {
        task: String,
        generation: u64,
        beat: u64,
    },
    Cancel {
        task: String,
        generation: u64,
    },
    /// `generation` is the aborted generation; the retry runs at the next.
    Retry {
        task: String,
        generation: u64,
    },
    Revoke {
        task: String,
        generation: u64,
        reason: String,
    },
    /// Emitted only by `Scheduler::audit` (callers cannot deliver it):
    /// revoke every listed task that is still active, then dispatch once,
    /// so no consumer is dispatched between two audit revocations.
    AuditRevoke {
        revocations: Vec<Revocation>,
    },
    /// Lets time pass: expires leases.
    Tick,
    /// `Committable -> Committing` (Headstart M3, issue #537). Freezes the
    /// task at `generation` with its own accepted completion and every
    /// producer's accepted completion, all of which must equal the current
    /// state and verify in the store. Refused when another commit is
    /// running in the same dependency chain.
    BeginCommit {
        task: String,
        generation: u64,
        completion: ProofClaim,
        producers: BTreeMap<String, ProofClaim>,
    },
    /// Ends a `Committing` task with the wrapped call's outcome.
    CommitOutcome {
        task: String,
        generation: u64,
        outcome: CommitResult,
    },
    /// Bind an external request `key` (opaque to M2, at most
    /// [`MAX_APPROVAL_KEY_LEN`] bytes) to the commit fence of `task` at
    /// `generation`. A later bind of the same key replaces it. Refused for
    /// a task whose commit ended (`Committed`, `CommitUncertain`). Every
    /// binding of a task is dropped when the task is `Committed`, and only
    /// then: an `Aborted` task keeps its bindings, and so does a retried or
    /// revoked one (the binding then names a superseded generation), so the
    /// request stays fenced until it is bound again. A `CommitUncertain`
    /// task keeps its bindings too, for good: whether its commit landed is
    /// unknown, so a bound request delivered again is refused by the fence
    /// (the task is terminal) instead of running unfenced; only a request
    /// under a new key, which nobody bound, runs as the original call.
    /// Each task holds at most [`MAX_APPROVAL_BINDINGS_PER_TASK`] keys, so
    /// one task cannot take another task's slots and the table is bounded
    /// by the number of tasks. A new key for a full task reclaims one of
    /// the task's bindings at a superseded generation (the lowest
    /// generation, then the lowest key, so a replay reclaims the same one);
    /// the request that key named is then unbound, and its fence was
    /// already a refusal (`StaleGeneration`). A full task whose bindings
    /// are all at its current generation refuses the new key
    /// (`BindingsFull`).
    BindApproval {
        key: String,
        task: String,
        generation: u64,
    },
}

/// Most request keys [`EventKind::BindApproval`] holds for one task.
pub const MAX_APPROVAL_BINDINGS_PER_TASK: usize = 64;
/// Longest request key [`EventKind::BindApproval`] accepts, in bytes.
pub const MAX_APPROVAL_KEY_LEN: usize = 512;

/// One bound request: the task generation whose commit fence it must pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApprovalBinding {
    pub task: String,
    pub generation: u64,
}

/// A task in one of these states has run its commit, for good: no request
/// may be bound to it any more. `Aborted` is not one of them: a cancelled,
/// failed or expired task may be retried, and its bound requests must stay
/// fenced.
fn binding_ended(state: TaskState) -> bool {
    matches!(state, TaskState::Committed | TaskState::CommitUncertain)
}

/// A task in this state holds no request binding: its commit landed, so a
/// bound request delivered again would be a duplicate the durable ledgers
/// answer. `CommitUncertain` is not one of them: whether that commit landed
/// is unknown, so its bound requests stay fenced (and refused, the task
/// being terminal) rather than running unfenced.
fn bindings_released(state: TaskState) -> bool {
    matches!(state, TaskState::Committed)
}

/// How a commit ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case", deny_unknown_fields)]
pub enum CommitResult {
    /// The commit succeeded: `Committed`.
    Committed { detail: String },
    /// The commit was refused before any durable effect: `Aborted`
    /// (retryable at the next generation); consumers are revoked.
    Failed { reason: String },
    /// The outcome is unknown: `CommitUncertain`, never run again;
    /// consumers are revoked.
    Uncertain { reason: String },
}

/// What a `BeginCommit` froze.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommitInputs {
    pub completion: ProofClaim,
    pub producers: BTreeMap<String, ProofClaim>,
    pub began_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Revocation {
    pub task: String,
    pub reason: String,
}

/// One delivered event. Duplicate delivery is detected by `id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub id: String,
    pub kind: EventKind,
}

/// One journal line: the accepted event plus its sequence number and the
/// monotonic time it was applied at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalEntry {
    pub seq: u64,
    pub id: String,
    pub at: u64,
    pub kind: EventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BindingLevel {
    Interface,
    Final,
}

/// Which producer artifact a consumer was dispatched against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Binding {
    pub generation: u64,
    pub level: BindingLevel,
    pub claim: ProofClaim,
}

/// Scheduler view of one task. Returned by shared reference only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TaskEntry {
    pub state: TaskState,
    /// This task's own producer generation. Starts at 1, only increases.
    pub generation: u64,
    pub resources: ResourceRequest,
    /// Accepted interface claim for the current generation.
    pub interface: Option<ProofClaim>,
    /// Accepted M1 completion claim for the current generation.
    pub completion: Option<ProofClaim>,
    pub bindings: BTreeMap<String, Binding>,
    pub lease_expires_at: Option<u64>,
    /// Highest heartbeat beat accepted for the current lease.
    pub last_beat: u64,
    pub provisional_result: Option<String>,
    pub queued: bool,
    pub last_reason: Option<String>,
    pub timing: TaskTiming,
    /// Set by `BeginCommit`; kept after the outcome as the record of what
    /// the commit was checked against.
    pub commit: Option<CommitInputs>,
}

impl TaskEntry {
    pub(crate) fn new(resources: ResourceRequest, at: u64) -> Self {
        TaskEntry {
            state: TaskState::Blocked,
            generation: 1,
            resources,
            interface: None,
            completion: None,
            bindings: BTreeMap::new(),
            lease_expires_at: None,
            last_beat: 0,
            provisional_result: None,
            queued: false,
            last_reason: None,
            timing: TaskTiming::new(at),
            commit: None,
        }
    }

    /// Final for consumers: `Committable`, `Committing` or `Committed`
    /// with an accepted completion.
    pub(crate) fn final_ready(&self) -> bool {
        self.state.is_final() && self.completion.is_some()
    }

    pub(crate) fn interface_ready(&self) -> bool {
        self.state.is_active() && self.interface.is_some()
    }

    fn lease_valid(&self, at: u64) -> bool {
        self.lease_expires_at.is_some_and(|exp| at < exp)
    }
}

/// What applying an event did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Effects {
    pub dispatched: Vec<String>,
    pub revoked: Vec<String>,
    pub aborted: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Core {
    pub(crate) graph: DependencyGraph,
    pub(crate) tasks: BTreeMap<String, TaskEntry>,
    pub(crate) queue: VecDeque<String>,
    pub(crate) pool: PermitPool,
    /// Bounded memory of applied event ids (`dedup.rs`).
    pub(crate) seen: Seen,
    /// Bound requests by key ([`EventKind::BindApproval`]).
    pub(crate) approvals: BTreeMap<String, ApprovalBinding>,
    pub(crate) seq: u64,
    pub(crate) last_at: u64,
}

pub(crate) fn payload_digest(kind: &EventKind) -> String {
    artifact_digest(&serde_json::to_value(kind).unwrap_or_default())
}

fn bad(m: impl Into<String>) -> SchedError {
    SchedError::InvalidEvent(m.into())
}

impl Core {
    pub(crate) fn task(&self, id: &str) -> Result<&TaskEntry, SchedError> {
        self.tasks
            .get(id)
            .ok_or_else(|| SchedError::UnknownTask(id.to_string()))
    }

    fn task_mut(&mut self, id: &str) -> Result<&mut TaskEntry, SchedError> {
        self.tasks
            .get_mut(id)
            .ok_or_else(|| SchedError::UnknownTask(id.to_string()))
    }

    pub(crate) fn require_generation(&self, id: &str, generation: u64) -> Result<(), SchedError> {
        let current = self.task(id)?.generation;
        let (task, submitted) = (id.to_string(), generation);
        match generation.cmp(&current) {
            std::cmp::Ordering::Less => Err(SchedError::StaleGeneration {
                task,
                submitted,
                current,
            }),
            std::cmp::Ordering::Greater => Err(SchedError::FutureGeneration {
                task,
                submitted,
                current,
            }),
            std::cmp::Ordering::Equal => Ok(()),
        }
    }

    fn require_state(
        &self,
        id: &str,
        ok: impl Fn(TaskState) -> bool,
        action: &'static str,
    ) -> Result<(), SchedError> {
        let state = self.task(id)?.state;
        if ok(state) {
            Ok(())
        } else {
            Err(SchedError::WrongState {
                task: id.to_string(),
                state,
                action,
            })
        }
    }

    /// The only place a task changes state. Leaving `Provisional` releases
    /// the task's permits and lease.
    fn set_state(&mut self, id: &str, to: TaskState, at: u64) -> Result<(), SchedError> {
        let Core { tasks, pool, .. } = self;
        let t = tasks
            .get_mut(id)
            .ok_or_else(|| SchedError::UnknownTask(id.to_string()))?;
        check_transition(t.state, to)?;
        t.timing.close(t.state, at);
        if t.state.holds_permits() && !to.holds_permits() {
            pool.release(t.resources);
            t.lease_expires_at = None;
        }
        t.state = to;
        Ok(())
    }

    fn dequeue(&mut self, id: &str) {
        if let Some(t) = self.tasks.get_mut(id) {
            if t.queued {
                t.queued = false;
                self.queue.retain(|q| q != id);
            }
        }
    }

    fn clear_work(t: &mut TaskEntry, reason: &str) {
        t.interface = None;
        t.completion = None;
        t.bindings.clear();
        t.provisional_result = None;
        t.last_reason = Some(reason.to_string());
    }

    fn reset_to_blocked(
        &mut self,
        id: &str,
        generation: u64,
        at: u64,
        reason: &str,
        fx: &mut Effects,
    ) -> Result<(), SchedError> {
        if self.task(id)?.state != TaskState::Blocked {
            self.set_state(id, TaskState::Blocked, at)?;
        }
        let t = self.task_mut(id)?;
        t.generation = generation;
        t.timing.revocations += 1;
        Self::clear_work(t, reason);
        fx.revoked.push(id.to_string());
        Ok(())
    }

    fn abort(
        &mut self,
        id: &str,
        at: u64,
        reason: &str,
        fx: &mut Effects,
    ) -> Result<(), SchedError> {
        self.dequeue(id);
        self.set_state(id, TaskState::Aborted, at)?;
        Self::clear_work(self.task_mut(id)?, reason);
        fx.aborted.push(id.to_string());
        Ok(())
    }

    /// Return every active descendant of `producer` to `Blocked` at a new
    /// generation and discard its provisional result. Queued descendants
    /// leave the queue and are re-checked by [`Core::pump`].
    fn revoke_descendants(
        &mut self,
        producer: &str,
        at: u64,
        reason: &str,
        fx: &mut Effects,
    ) -> Result<(), SchedError> {
        self.guard_descendants(producer)?;
        for d in self.affected(producer) {
            let t = self.task(&d)?;
            if t.state.is_active() {
                let next = t.generation + 1;
                let why = format!("{reason} (upstream '{producer}')");
                self.reset_to_blocked(&d, next, at, &why, fx)?;
            } else {
                self.dequeue(&d);
            }
        }
        Ok(())
    }

    /// Descendants of `id` that an upstream change reaches, breadth-first,
    /// excluding `id`. A `Committed` or `CommitUncertain` task is done: it
    /// is neither returned nor walked through (its consumers are reached
    /// only along other paths).
    pub(crate) fn affected(&self, id: &str) -> Vec<String> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        let mut q: VecDeque<&String> = self.graph.consumers_of(id).collect();
        while let Some(n) = q.pop_front() {
            if !seen.insert(n.clone()) {
                continue;
            }
            if self
                .tasks
                .get(n)
                .is_some_and(|t| t.state.is_commit_terminal())
            {
                continue;
            }
            out.push(n.clone());
            q.extend(self.graph.consumers_of(n));
        }
        out
    }

    /// The `Committing` task an upstream change of `id` would reach.
    fn committing_below(&self, id: &str) -> Option<String> {
        self.affected(id)
            .into_iter()
            .find(|d| self.tasks.get(d).map(|t| t.state) == Some(TaskState::Committing))
    }

    fn guard_descendants(&self, id: &str) -> Result<(), SchedError> {
        match self.committing_below(id) {
            Some(committing) => Err(SchedError::CommitInProgress {
                task: id.to_string(),
                committing,
            }),
            None => Ok(()),
        }
    }

    /// Refuse an event that would change `id` itself while it is
    /// `Committing`, or reach a `Committing` task below it.
    pub(crate) fn guard_commit(&self, id: &str) -> Result<(), SchedError> {
        if self.task(id)?.state == TaskState::Committing {
            return Err(SchedError::CommitInProgress {
                task: id.to_string(),
                committing: id.to_string(),
            });
        }
        self.guard_descendants(id)
    }

    /// A `Committing` task in the same chain as `id` (upstream or
    /// downstream along [`Core::affected`] paths), other than `id`.
    fn commit_in_chain(&self, id: &str) -> Option<String> {
        if let Some(c) = self.committing_below(id) {
            return Some(c);
        }
        self.tasks
            .iter()
            .filter(|(other, t)| t.state == TaskState::Committing && other.as_str() != id)
            .map(|(other, _)| other)
            .find(|other| self.affected(other).iter().any(|d| d == id))
            .cloned()
    }

    fn eligible(&self, cfg: &ProvisionalConfig, id: &str) -> bool {
        let Some(t) = self.tasks.get(id) else {
            return false;
        };
        t.state == TaskState::Blocked
            && self.graph.producers_of(id).all(|p| {
                self.tasks
                    .get(p)
                    .is_some_and(|pt| pt.final_ready() || (cfg.enabled && pt.interface_ready()))
            })
    }

    fn bindings_for(&self, id: &str) -> BTreeMap<String, Binding> {
        let mut out = BTreeMap::new();
        for p in self.graph.producers_of(id) {
            let Some(pt) = self.tasks.get(p) else {
                continue;
            };
            let (level, claim) = if pt.final_ready() {
                (BindingLevel::Final, pt.completion.clone())
            } else {
                (BindingLevel::Interface, pt.interface.clone())
            };
            if let Some(claim) = claim {
                out.insert(
                    p.clone(),
                    Binding {
                        generation: pt.generation,
                        level,
                        claim,
                    },
                );
            }
        }
        out
    }

    /// Fill the bounded ready queue in task-id order, then dispatch from its
    /// head while permits allow (head-of-line, no reordering).
    fn pump(
        &mut self,
        cfg: &ProvisionalConfig,
        at: u64,
        fx: &mut Effects,
    ) -> Result<(), SchedError> {
        loop {
            let stale: Vec<String> = self
                .queue
                .iter()
                .filter(|id| !self.eligible(cfg, id))
                .cloned()
                .collect();
            for id in stale {
                self.dequeue(&id);
            }
            let ready: Vec<String> = self
                .tasks
                .iter()
                .filter(|(id, t)| !t.queued && self.eligible(cfg, id))
                .map(|(id, _)| id.clone())
                .collect();
            for id in ready {
                if self.queue.len() >= cfg.queue_capacity {
                    break;
                }
                self.queue.push_back(id.clone());
                self.task_mut(&id)?.queued = true;
            }
            let mut progressed = false;
            while let Some(head) = self.queue.front().cloned() {
                let req = self.task(&head)?.resources;
                if !self.pool.try_acquire(&cfg.limits, req) {
                    break;
                }
                self.queue.pop_front();
                let bindings = self.bindings_for(&head);
                self.set_state(&head, TaskState::Provisional, at)?;
                let t = self.task_mut(&head)?;
                t.queued = false;
                t.bindings = bindings;
                t.lease_expires_at = Some(at.saturating_add(cfg.lease_ttl));
                // One lease per generation: leaving Provisional always ends
                // in a new generation before the next dispatch.
                t.last_beat = 0;
                t.timing.dispatches += 1;
                t.timing.first_dispatch_at.get_or_insert(at);
                fx.dispatched.push(head);
                progressed = true;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Apply one journal entry. On error the caller discards this copy of
    /// the state, so a refused event changes nothing.
    pub(crate) fn apply(
        &mut self,
        cfg: &ProvisionalConfig,
        e: &JournalEntry,
    ) -> Result<Effects, SchedError> {
        if e.seq != self.seq + 1 {
            return Err(bad(format!(
                "sequence {} out of order, expected {}",
                e.seq,
                self.seq + 1
            )));
        }
        if self.seen.contains(&e.id) {
            return Err(SchedError::EventIdConflict(e.id.clone()));
        }
        if e.at < self.last_at {
            return Err(bad("event time is earlier than the previous event"));
        }
        self.last_at = e.at;
        let at = e.at;
        let mut fx = Effects::default();
        match &e.kind {
            EventKind::RegisterTask {
                task,
                resources,
                producers,
            } => {
                check_identifier("task", task).map_err(|err| bad(err.to_string()))?;
                if !cfg.limits.admits(*resources) {
                    return Err(SchedError::ExceedsCapacity(task.clone()));
                }
                self.graph.add_node(task)?;
                for p in producers {
                    self.graph.add_edge(p, task)?;
                }
                self.tasks
                    .insert(task.clone(), TaskEntry::new(*resources, at));
            }
            EventKind::AddDependency { producer, consumer } => {
                self.graph.check_edge(producer, consumer)?;
                self.require_state(consumer, |s| s == TaskState::Blocked, "gain a dependency")?;
                self.graph.add_edge(producer, consumer)?;
                self.dequeue(consumer);
            }
            EventKind::InterfacePublished {
                task,
                generation,
                claim,
            }
            | EventKind::CompletionVerified {
                task,
                generation,
                claim,
            } => {
                let is_interface = matches!(e.kind, EventKind::InterfacePublished { .. });
                self.require_generation(task, *generation)?;
                self.require_state(task, TaskState::is_active, "publish proof")?;
                let t = self.task_mut(task)?;
                let slot = if is_interface {
                    &mut t.interface
                } else {
                    &mut t.completion
                };
                let changed = slot.is_some() && slot.as_ref() != Some(claim);
                *slot = Some(claim.clone());
                if changed {
                    self.revoke_descendants(task, at, "producer proof changed", &mut fx)?;
                }
            }
            EventKind::NewGeneration { task, generation } => {
                let current = self.task(task)?.generation;
                if *generation <= current {
                    return Err(SchedError::StaleGeneration {
                        task: task.clone(),
                        submitted: *generation,
                        current,
                    });
                }
                self.require_state(
                    task,
                    |s| s != TaskState::Aborted,
                    "start a new generation (use retry)",
                )?;
                self.guard_commit(task)?;
                self.reset_to_blocked(task, *generation, at, "new generation", &mut fx)?;
                self.revoke_descendants(task, at, "producer new generation", &mut fx)?;
            }
            EventKind::ProducerFailed {
                task,
                generation,
                reason,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s != TaskState::Aborted, "fail")?;
                self.guard_commit(task)?;
                self.abort(task, at, &format!("failed: {reason}"), &mut fx)?;
                self.revoke_descendants(task, at, "producer failed", &mut fx)?;
            }
            EventKind::WorkDone {
                task,
                generation,
                result_digest,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s == TaskState::Provisional, "report work done")?;
                if !self.task(task)?.lease_valid(at) {
                    return Err(SchedError::LeaseExpired(task.clone()));
                }
                if result_digest.is_empty() {
                    return Err(bad("result_digest is empty"));
                }
                self.set_state(task, TaskState::AwaitingFinalProof, at)?;
                self.task_mut(task)?.provisional_result = Some(result_digest.clone());
            }
            EventKind::Finalize { task, generation } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s == TaskState::AwaitingFinalProof, "finalize")?;
                for p in self.graph.producers_of(task) {
                    let pt = self.task(p)?;
                    let bound = self.task(task)?.bindings.get(p).map(|b| b.generation);
                    if bound != Some(pt.generation) || !pt.final_ready() {
                        return Err(SchedError::ProducerNotFinal {
                            producer: p.clone(),
                            detail: "not committable with a completion at the bound generation"
                                .to_string(),
                        });
                    }
                }
                self.set_state(task, TaskState::Committable, at)?;
                self.task_mut(task)?.timing.committable_at = Some(at);
            }
            EventKind::Heartbeat {
                task,
                generation,
                beat,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s == TaskState::Provisional, "heartbeat")?;
                let t = self.task_mut(task)?;
                if !t.lease_valid(at) {
                    return Err(SchedError::LeaseExpired(task.clone()));
                }
                if *beat <= t.last_beat {
                    return Err(SchedError::StaleHeartbeat {
                        task: task.clone(),
                        beat: *beat,
                        last: t.last_beat,
                    });
                }
                t.last_beat = *beat;
                t.lease_expires_at = Some(at.saturating_add(cfg.lease_ttl));
            }
            EventKind::Cancel { task, generation } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s != TaskState::Aborted, "cancel")?;
                self.guard_commit(task)?;
                self.abort(task, at, "cancelled", &mut fx)?;
                for d in self.affected(task) {
                    if self.task(&d)?.state != TaskState::Aborted {
                        self.abort(&d, at, &format!("ancestor '{task}' cancelled"), &mut fx)?;
                    }
                }
            }
            EventKind::Retry { task, generation } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s == TaskState::Aborted, "retry")?;
                self.set_state(task, TaskState::Blocked, at)?;
                let t = self.task_mut(task)?;
                t.generation += 1;
                Self::clear_work(t, "retry");
            }
            EventKind::Revoke {
                task,
                generation,
                reason,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, TaskState::is_active, "be revoked")?;
                let next = self.task(task)?.generation + 1;
                self.reset_to_blocked(task, next, at, reason, &mut fx)?;
                self.revoke_descendants(task, at, reason, &mut fx)?;
            }
            EventKind::AuditRevoke { revocations } => {
                let first = revocations
                    .first()
                    .ok_or_else(|| bad("empty audit revoke"))?;
                self.require_state(&first.task, TaskState::is_active, "be revoked")?;
                for r in revocations {
                    // Earlier entries may already have revoked this one.
                    let t = self.task(&r.task)?;
                    if t.state.is_active() {
                        let next = t.generation + 1;
                        self.reset_to_blocked(&r.task, next, at, &r.reason, &mut fx)?;
                        self.revoke_descendants(&r.task, at, &r.reason, &mut fx)?;
                    }
                }
            }
            EventKind::Tick => {
                let expired: Vec<String> = self
                    .tasks
                    .iter()
                    .filter(|(_, t)| t.state == TaskState::Provisional && !t.lease_valid(at))
                    .map(|(id, _)| id.clone())
                    // An expiry that would reach a running commit waits for
                    // its outcome (a later Tick expires it).
                    .filter(|id| self.guard_commit(id).is_ok())
                    .collect();
                for id in expired {
                    if self.task(&id)?.state == TaskState::Provisional {
                        self.abort(&id, at, "lease expired", &mut fx)?;
                        self.revoke_descendants(&id, at, "producer lease expired", &mut fx)?;
                    }
                }
            }
            EventKind::BeginCommit {
                task,
                generation,
                completion,
                producers,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(task, |s| s == TaskState::Committable, "begin a commit")?;
                if let Some(committing) = self.commit_in_chain(task) {
                    return Err(SchedError::CommitInProgress {
                        task: task.clone(),
                        committing,
                    });
                }
                if self.task(task)?.completion.as_ref() != Some(completion) {
                    return Err(bad(format!(
                        "begin_commit completion for '{task}' is not its accepted completion"
                    )));
                }
                let expected: Vec<&String> = self.graph.producers_of(task).collect();
                if expected.len() != producers.len()
                    || expected.iter().any(|p| !producers.contains_key(*p))
                {
                    return Err(bad(format!(
                        "begin_commit for '{task}' must name exactly its producers"
                    )));
                }
                for (p, claim) in producers {
                    let pt = self.task(p)?;
                    // A producer that is itself committing is refused above.
                    if !matches!(pt.state, TaskState::Committable | TaskState::Committed)
                        || pt.completion.as_ref() != Some(claim)
                    {
                        return Err(SchedError::ProducerNotFinal {
                            producer: p.clone(),
                            detail: format!(
                                "state {}, completion differs from the begin_commit input: {}",
                                pt.state.as_str(),
                                pt.completion.as_ref() != Some(claim)
                            ),
                        });
                    }
                }
                self.set_state(task, TaskState::Committing, at)?;
                self.task_mut(task)?.commit = Some(CommitInputs {
                    completion: completion.clone(),
                    producers: producers.clone(),
                    began_at: at,
                });
            }
            EventKind::CommitOutcome {
                task,
                generation,
                outcome,
            } => {
                self.require_generation(task, *generation)?;
                self.require_state(
                    task,
                    |s| s == TaskState::Committing,
                    "record a commit outcome",
                )?;
                // BeginCommit refused any second commit in this chain, so
                // no Committing task is reachable from here.
                match outcome {
                    CommitResult::Committed { detail } => {
                        self.set_state(task, TaskState::Committed, at)?;
                        self.task_mut(task)?.last_reason = Some(format!("committed: {detail}"));
                    }
                    CommitResult::Failed { reason } => {
                        self.abort(task, at, &format!("commit failed: {reason}"), &mut fx)?;
                        self.revoke_descendants(task, at, "producer commit failed", &mut fx)?;
                    }
                    CommitResult::Uncertain { reason } => {
                        self.revoke_descendants(task, at, "producer commit uncertain", &mut fx)?;
                        self.set_state(task, TaskState::CommitUncertain, at)?;
                        self.task_mut(task)?.last_reason =
                            Some(format!("commit uncertain: {reason}"));
                    }
                }
            }
            EventKind::BindApproval {
                key,
                task,
                generation,
            } => {
                if key.is_empty() || key.len() > MAX_APPROVAL_KEY_LEN {
                    return Err(bad(format!(
                        "bind_approval key must be 1 to {MAX_APPROVAL_KEY_LEN} bytes"
                    )));
                }
                self.require_generation(task, *generation)?;
                self.require_state(
                    task,
                    |s| !binding_ended(s),
                    "bind a request to a task that already ended",
                )?;
                let rebind = self.approvals.get(key).is_some_and(|b| b.task == *task);
                let held = self.approvals.values().filter(|b| b.task == *task).count();
                if !rebind && held >= MAX_APPROVAL_BINDINGS_PER_TASK {
                    // Reclaim one binding at a superseded generation:
                    // lowest generation, then lowest key (BTreeMap order),
                    // so a replay reclaims the same one.
                    let current = self.task(task)?.generation;
                    let superseded = self
                        .approvals
                        .iter()
                        .filter(|(_, b)| b.task == *task && b.generation < current)
                        .min_by_key(|(k, b)| (b.generation, (*k).clone()))
                        .map(|(k, _)| k.clone());
                    match superseded {
                        Some(k) => {
                            self.approvals.remove(&k);
                        }
                        None => {
                            return Err(SchedError::BindingsFull {
                                task: task.clone(),
                                capacity: MAX_APPROVAL_BINDINGS_PER_TASK,
                            })
                        }
                    }
                }
                self.approvals.insert(
                    key.clone(),
                    ApprovalBinding {
                        task: task.clone(),
                        generation: *generation,
                    },
                );
            }
        }
        self.pump(cfg, at, &mut fx)?;
        // Every binding of a task whose commit landed (by this event or by
        // a cascade from it) is dropped. Aborted and uncertain tasks keep
        // theirs (see `bindings_released`). Deterministic, so a replay
        // rebuilds the same table.
        let tasks = &self.tasks;
        self.approvals.retain(|_, b| {
            tasks
                .get(&b.task)
                .is_some_and(|t| !bindings_released(t.state))
        });
        self.seen
            .record(&e.id, payload_digest(&e.kind), &e.kind, cfg.dedup_window);
        let tasks = &self.tasks;
        self.seen.prune(|t| tasks.get(t).map(|x| x.generation));
        self.seq = e.seq;
        Ok(fx)
    }
}
