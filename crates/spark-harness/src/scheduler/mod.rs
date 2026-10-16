//! Provisional dependency scheduler (Headstart M2, issue #535).
//!
//! Lets a consumer task start reversible, sandboxed preparation as soon as
//! its producers publish a harness-verified interface, while final
//! completion waits for every producer's M1 verified completion. This is an
//! engineering-workflow scheduler only; it does not touch inference
//! scheduling, KV/prefill fences, aien-trace or World commit.
//!
//! Opt-in: [`ProvisionalConfig::enabled`] defaults to `false`. Disabled means
//! no early start: a consumer stays `Blocked` until every producer is
//! final-verified and `Committable`, which is plain sequential behavior. No
//! existing spark-harness API or MCP tool uses this module.
//!
//! Model:
//! - Tasks and producer -> consumer edges form a cycle-checked DAG
//!   (`dependency.rs`). Task states and the transition table live in
//!   `execution.rs`.
//! - Everything is an [`Event`] with a caller-chosen id. Redelivering the
//!   same id with the same payload is a no-op ([`Outcome::Duplicate`]); the
//!   same id with a different payload is refused.
//! - [`Scheduler::deliver`] runs the store checks for the event, applies it
//!   to a copy of the state, journals it, then swaps the copy in. A refused
//!   event changes nothing and is not journaled.
//! - Producers have monotonic generations. Events for an older generation are
//!   refused as stale. A producer failure, a changed interface or completion
//!   at the same generation, a new generation, a lost lease or a revocation
//!   returns every active descendant to `Blocked` at a new generation and
//!   discards its provisional result.
//! - A bounded ready queue feeds dispatch. Dispatch takes CPU, memory and
//!   concurrency permits all at once and grants a lease; only `Provisional`
//!   tasks hold permits.
//!
//! Persistence choice: an append-only JSON-lines journal (`journal.rs`) of
//! accepted events, opened with [`Scheduler::open`]. Replay re-applies the
//! events through the same pure state machine, using the times recorded in
//! the journal, so it rebuilds the same state (and the same timing trace).
//! Replay does not re-read the store, so `open` returns an [`Unaudited`]
//! handle whose only method is `audit`: no event, dispatch or gate query
//! can run on replayed state until every active task was re-verified
//! against the store. After that every `may_start` and `may_finalize`
//! still re-verifies at call time. A task becomes `Committable` only
//! through a journaled `Finalize` event, which is journaled only after
//! `may_finalize` passed, so replay cannot turn an unfinished provisional
//! task into a committable one. The journal is trusted like the store
//! itself (filesystem ownership); it is not signed.
//!
//! Time after a restart: a new process gets a new clock, so `open` rebases
//! it onto the last journaled time (`now = last_at + clock() - clock_at_open`).
//! Leases of workers that died with the old process therefore expire at
//! most `lease_ttl` ticks after the restart. The downtime itself is not
//! counted.
//!
//! Cost per event: `deliver` applies the event to a copy of the state so a
//! refused event changes nothing. The copy is O(tasks + edges +
//! `dedup_window`), the same order as the dispatch scan every event already
//! does; `deliver_cost_is_small_next_to_the_journal_fsync` (ignored test)
//! measures it. Journal compaction (a snapshot header so replay does not
//! start from the first event) is a follow-up; until then the journal and
//! replay time grow with the event count.

pub mod core;
mod dedup;
pub mod error;
mod gates;
pub mod permits;
pub mod timing;
#[cfg(test)]
mod unit_tests;

pub use self::core::{
    ApprovalBinding, Binding, BindingLevel, CommitInputs, CommitResult, Effects, Event, EventKind,
    JournalEntry, Revocation, TaskEntry, MAX_APPROVAL_BINDINGS_PER_TASK, MAX_APPROVAL_KEY_LEN,
};
pub use self::error::SchedError;
pub use self::permits::{PermitPool, ResourceLimits, ResourceRequest};

use self::core::{payload_digest, Core};
use crate::completion::check_identifier;
use crate::execution::TaskState;
use crate::journal::Journal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Monotonic clock in ticks. Injected so tests are deterministic.
pub type Clock = Box<dyn Fn() -> u64 + Send + Sync>;

/// Nanoseconds since this clock was created, from `std::time::Instant`.
/// A new process starts a new base; [`Scheduler::open`] rebases it onto
/// the last journaled time (see the module docs).
pub fn monotonic_clock() -> Clock {
    let start = std::time::Instant::now();
    Box::new(move || u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

pub const JOURNAL_FORMAT: &str = "spark-harness-provisional-journal";
/// Version 2: task-directed events carry a generation and heartbeats a
/// beat number. A version 1 journal
/// fails the header check on open and is never replayed.
pub const JOURNAL_VERSION: u64 = 2;
pub const TRACE_SCHEMA: &str = "spark-harness-provisional-trace";
/// Event ids starting with this are reserved for events the scheduler
/// emits itself (the audit). [`Scheduler::deliver`] refuses them.
pub const RESERVED_ID_PREFIX: &str = "spark-harness:";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvisionalConfig {
    /// Allow early start on interface-verified producers. Default `false`.
    pub enabled: bool,
    /// Maximum ready-queue length (at least 1).
    pub queue_capacity: usize,
    pub limits: ResourceLimits,
    /// Lease length in clock ticks (at least 1).
    pub lease_ttl: u64,
    /// How many generation-free event ids are remembered for duplicate
    /// detection (at least 1). See `dedup.rs`.
    pub dedup_window: usize,
}

impl Default for ProvisionalConfig {
    fn default() -> Self {
        ProvisionalConfig {
            enabled: false,
            queue_capacity: 64,
            limits: ResourceLimits::default(),
            lease_ttl: 300_000_000_000,
            dedup_window: 4096,
        }
    }
}

impl ProvisionalConfig {
    fn validate(&self) -> Result<(), SchedError> {
        if self.queue_capacity == 0
            || self.lease_ttl == 0
            || self.limits.max_concurrency == 0
            || self.dedup_window == 0
        {
            return Err(SchedError::InvalidEvent(
                "config needs queue_capacity, lease_ttl, max_concurrency and dedup_window of at least 1"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Outcome {
    Applied(Effects),
    /// The event id was already applied with the same payload.
    Duplicate,
}

/// What [`Scheduler::open`] found in the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayReport {
    pub entries: usize,
    pub torn_bytes: usize,
}

pub struct Scheduler {
    cfg: ProvisionalConfig,
    store: PathBuf,
    clock: Clock,
    /// `now = clock_base + clock() - clock_origin` (see the module docs).
    clock_base: u64,
    clock_origin: u64,
    core: Core,
    journal: Option<Journal>,
}

/// A replayed scheduler that has not been checked against the store yet.
/// Its only method is [`Unaudited::audit`].
#[must_use = "a replayed scheduler is usable only after audit()"]
pub struct Unaudited {
    sched: Scheduler,
}

impl Unaudited {
    /// Mark every replayed `Committing` task `CommitUncertain` (its outcome
    /// was lost; it is never run again), re-verify every replayed active
    /// task against the store, revoke the ones that fail, and hand out the
    /// scheduler.
    pub fn audit(mut self) -> Result<(Scheduler, Vec<String>), SchedError> {
        self.sched.settle_interrupted_commits()?;
        let revoked = self.sched.audit()?;
        Ok((self.sched, revoked))
    }
}

impl Scheduler {
    /// In-memory scheduler (no journal). `store` is the harness store that
    /// holds interface and completion receipts (`HarnessEngine::completion_store`).
    pub fn new(cfg: ProvisionalConfig, store: &Path, clock: Clock) -> Result<Self, SchedError> {
        cfg.validate()?;
        Ok(Scheduler {
            cfg,
            store: store.to_path_buf(),
            clock,
            clock_base: 0,
            clock_origin: 0,
            core: Core::default(),
            journal: None,
        })
    }

    /// Journaled scheduler: create `journal` or replay it. The journal's
    /// header must carry the same config, because dispatch depends on it.
    /// The replayed state must pass [`Unaudited::audit`] before use.
    pub fn open(
        cfg: ProvisionalConfig,
        store: &Path,
        journal: &Path,
        clock: Clock,
    ) -> Result<(Unaudited, ReplayReport), SchedError> {
        let mut s = Scheduler::new(cfg, store, clock)?;
        let header =
            json!({ "format": JOURNAL_FORMAT, "version": JOURNAL_VERSION, "config": s.cfg });
        let opened =
            Journal::open::<JournalEntry>(journal, &header).map_err(SchedError::Journal)?;
        for entry in &opened.entries {
            s.core.apply(&s.cfg, entry).map_err(|e| {
                SchedError::Journal(format!("replay of entry {} failed: {e}", entry.seq))
            })?;
        }
        if !opened.entries.is_empty() {
            s.clock_base = s.core.last_at;
            s.clock_origin = (s.clock)();
        }
        let report = ReplayReport {
            entries: opened.entries.len(),
            torn_bytes: opened.torn_bytes,
        };
        s.journal = Some(opened.journal);
        Ok((Unaudited { sched: s }, report))
    }

    pub fn config(&self) -> &ProvisionalConfig {
        &self.cfg
    }

    fn now(&self) -> u64 {
        let elapsed = (self.clock)().saturating_sub(self.clock_origin);
        self.clock_base
            .saturating_add(elapsed)
            .max(self.core.last_at)
    }

    /// Validate, journal and apply one event. See the module docs. Ids with
    /// [`RESERVED_ID_PREFIX`] and `AuditRevoke` events are refused: only
    /// the scheduler emits those.
    pub fn deliver(&mut self, event: Event) -> Result<Outcome, SchedError> {
        if event.id.starts_with(RESERVED_ID_PREFIX) {
            return Err(SchedError::InvalidEvent(format!(
                "event ids starting with '{RESERVED_ID_PREFIX}' are reserved"
            )));
        }
        if matches!(event.kind, EventKind::AuditRevoke { .. }) {
            return Err(SchedError::InvalidEvent(
                "audit_revoke is emitted only by audit()".to_string(),
            ));
        }
        self.deliver_unchecked(event)
    }

    pub(crate) fn deliver_unchecked(&mut self, event: Event) -> Result<Outcome, SchedError> {
        check_identifier("event id", &event.id)
            .map_err(|e| SchedError::InvalidEvent(e.to_string()))?;
        if let Some(prev) = self.core.seen.get(&event.id) {
            return if *prev == payload_digest(&event.kind) {
                Ok(Outcome::Duplicate)
            } else {
                Err(SchedError::EventIdConflict(event.id))
            };
        }
        self.verify_proofs(&event.kind)?;
        let entry = JournalEntry {
            seq: self.core.seq + 1,
            id: event.id,
            at: self.now(),
            kind: event.kind,
        };
        let mut next = self.core.clone();
        let fx = next.apply(&self.cfg, &entry)?;
        if let Some(j) = self.journal.as_mut() {
            j.append(&entry).map_err(SchedError::Journal)?;
        }
        self.core = next;
        Ok(Outcome::Applied(fx))
    }

    pub fn task(&self, id: &str) -> Option<&TaskEntry> {
        self.core.tasks.get(id)
    }

    pub fn state(&self, id: &str) -> Option<TaskState> {
        self.task(id).map(|t| t.state)
    }

    pub fn generation(&self, id: &str) -> Option<u64> {
        self.task(id).map(|t| t.generation)
    }

    /// The fence a bound request key must pass ([`EventKind::BindApproval`]).
    pub fn approval(&self, key: &str) -> Option<&ApprovalBinding> {
        self.core.approvals.get(key)
    }

    /// How many request keys are bound.
    pub fn approvals_len(&self) -> usize {
        self.core.approvals.len()
    }

    pub fn queue(&self) -> Vec<String> {
        self.core.queue.iter().cloned().collect()
    }

    pub fn permits_in_use(&self) -> PermitPool {
        self.core.pool
    }

    /// How many event ids are remembered for duplicate detection.
    pub fn remembered_event_ids(&self) -> usize {
        self.core.seen.len()
    }

    /// Full scheduler state as JSON, for replay comparisons and debugging.
    pub fn snapshot(&self) -> Value {
        serde_json::to_value(&self.core).unwrap_or(Value::Null)
    }

    /// Structural invariants. Tests call this after every step.
    pub fn check_invariants(&self) -> Result<(), String> {
        let mut pool = PermitPool::default();
        for (id, t) in &self.core.tasks {
            if t.state.holds_permits() {
                pool.running += 1;
                pool.cpu_in_use += t.resources.cpu_slots;
                pool.memory_in_use += t.resources.memory_bytes;
                if self.core.graph.producers_of(id).count() != t.bindings.len() {
                    return Err(format!("running task '{id}' lacks a producer binding"));
                }
            } else if t.lease_expires_at.is_some() {
                return Err(format!(
                    "task '{id}' is {} but holds a lease",
                    t.state.as_str()
                ));
            }
            let in_queue = self.core.queue.iter().any(|q| q == id);
            if t.queued != in_queue || (in_queue && t.state != TaskState::Blocked) {
                return Err(format!("task '{id}' queue flag disagrees with the queue"));
            }
        }
        if pool != self.core.pool {
            return Err(format!(
                "permits in use {:?} differ from running tasks {:?}",
                self.core.pool, pool
            ));
        }
        if self.core.queue.len() > self.cfg.queue_capacity {
            return Err("ready queue exceeds its bound".to_string());
        }
        Ok(())
    }

    /// Per-task blocked and provisional time as of the last event, for the
    /// M6 A/B analysis. Derived only from journaled event times.
    pub fn timing_trace(&self) -> Value {
        let now = self.core.last_at;
        let tasks: Vec<Value> = self
            .core
            .tasks
            .iter()
            .map(|(id, t)| {
                let timing = t.timing.as_of(t.state, now);
                json!({
                    "task": id,
                    "state": t.state.as_str(),
                    "generation": t.generation,
                    "blocked": timing.blocked,
                    "provisional": timing.provisional,
                    "awaiting_final_proof": timing.awaiting_final_proof,
                    "registered_at": timing.registered_at,
                    "first_dispatch_at": timing.first_dispatch_at,
                    "committable_at": timing.committable_at,
                    "dispatches": timing.dispatches,
                    "revocations": timing.revocations,
                })
            })
            .collect();
        json!({
            "schema": TRACE_SCHEMA,
            "version": 1,
            "enabled": self.cfg.enabled,
            "clock": "injected monotonic clock ticks",
            "as_of": now,
            "tasks": tasks,
        })
    }
}
