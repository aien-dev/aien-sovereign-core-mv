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

//! DUAL decision-site observer for `serve.admit_preempt` (ARCH-0031 §7.3,
//! DUAL-3 instrumentation). READ-ONLY by construction.
//!
//! `AienScheduler::build_scheduled_batch` is the one place in this crate that
//! decides admission, the preemption victim and the prefill chunk allocation.
//! This module lets a caller install a [`DecisionObserver`] that receives,
//! after each decision, one immutable [`AdmitPreemptRecord`] describing the
//! alternatives the scheduler actually looked at, the choice it made, and the
//! resource readings it made the choice with.
//!
//! What the tap cannot do:
//!
//! - It holds no reference to the scheduler, the KV manager or the arena. The
//!   hook receives `&AdmitPreemptRecord` (a copy of already-made decisions) and
//!   returns `()`. There is no return value the scheduler reads.
//! - With no observer installed (`None`, the default) every tracing branch is
//!   skipped and the decision code runs exactly as before. The parity tests in
//!   `tests/dual_decision_parity.rs` and `tests/dual_observer_parity.rs` hold
//!   the serialized decisions byte-for-byte equal in both configurations.
//! - No field of a record is an authority input anywhere. A consumer that
//!   reads one as a gate input fails its gate (ADR 0031 §6).
//!
//! The record names the production state machine's own identities
//! (`SequenceId`, step counter, KV block counts). It introduces no second
//! ledger: every number is copied from a local the scheduler computed for
//! its own decision, at the instant it computed it.
//!
//! ```compile_fail
//! // Negative control, held by the type system: an observer cannot change
//! // the record it is shown. `observe_admit_preempt` takes `&AdmitPreemptRecord`.
//! use aien_scheduler::dual_observer::{AdmitPreemptRecord, DecisionObserver};
//! struct Tamper;
//! impl DecisionObserver for Tamper {
//!     fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) {
//!         record.choice.preempted.clear();
//!     }
//! }
//! ```
//!
//! ```compile_fail
//! // Negative control, held by the type system: the hook returns `()`, so an
//! // observer cannot hand a different choice back to the scheduler.
//! use aien_scheduler::dual_observer::{AdmitPreemptChoice, AdmitPreemptRecord, DecisionObserver};
//! struct Override;
//! impl DecisionObserver for Override {
//!     fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) -> AdmitPreemptChoice {
//!         record.choice.clone()
//!     }
//! }
//! ```

use crate::SequenceId;
use serde::Serialize;
use std::sync::Mutex;

/// Registered decision-site id (DUAL_CURRENT_STATE §4).
pub const SERVE_ADMIT_PREEMPT_SITE: &str = "serve.admit_preempt";

/// Scheduler limits in force for the decision, copied from `SchedulerConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchLimits {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_prefill_tokens: usize,
    pub prefill_chunk_size: usize,
    pub chunk_prefill: bool,
    pub watermark_blocks: usize,
}

/// KV pool reading. `free_blocks` is `AienKvManager::available_blocks`,
/// `total_blocks` is `total_block_count`, `active_tables` is
/// `active_sequence_count`. `allocated_blocks` is `total - free`, the same
/// arithmetic as `allocated_block_count`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct KvReading {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub active_tables: usize,
}

/// One running sequence the watermark check ranked as a preemption victim,
/// in the order the scheduler sorted them (lowest priority first; equal
/// priorities keep their `running_sequences` order, `sort_by_key` is stable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PreemptionVictim {
    pub seq: SequenceId,
    pub priority: u8,
}

/// What step 2 of `build_scheduled_batch` did with one running sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum RunningVerdict {
    /// Arena has no record for the id (skipped).
    NoRecord,
    /// Prefilled, table ready: scheduled for one decode token.
    Decode,
    /// Prefilled flag set but the table was never computed: sent back to the
    /// waiting queue head for a fresh prefill (PREFILL-GATE).
    GateRefusedReprefill,
    /// In-flight chunked prefill: this many prompt tokens scheduled now.
    ContinuePrefill { chunk_len: usize },
    /// In-flight chunked prefill but no chunk fit (remaining or budget was 0).
    ContinuePrefillNoChunk,
    /// Prefilled sequence whose table is missing from the KV manager (skipped).
    DecodeNoTable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunningItem {
    pub seq: SequenceId,
    pub verdict: RunningVerdict,
}

/// Why the running-sequence loop stopped before examining every running id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LoopStop {
    BatchSizeReached,
    BatchTokensReached,
}

/// What step 3 of `build_scheduled_batch` did with one queue head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum AdmissionVerdict {
    /// Stale id at the queue head: dropped, loop continues.
    StaleDropped,
    /// Arena record missing: dropped, loop continues.
    MissingRecord,
    /// `available < min_needed` while something running could free blocks:
    /// the sequence stays queued, loop stops.
    WaitForBlocks,
    /// `available < min_needed` and nothing can free blocks (or the need
    /// exceeds the whole pool): `build_scheduled_batch` returned `Err`.
    RefusedPoolExhausted,
    /// Existing block table with computed K/V (a fork): entered decode
    /// directly, reserving `needed_blocks` for its first copy-on-write.
    AdmittedDecode,
    /// Allocated `new_blocks` physical blocks and scheduled the first
    /// `chunk_len` prompt tokens for prefill.
    AdmittedPrefill { chunk_len: usize, new_blocks: usize },
    /// Prefill budget left no room for a first chunk: pushed back to its
    /// queue head, loop stops.
    ChunkBudgetExhausted,
}

/// One queue head examined by the admission loop, with the exact numbers the
/// scheduler compared. Resource fields are zero for verdicts reached before
/// the KV manager was consulted (`StaleDropped`, `MissingRecord`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionCandidate {
    pub seq: SequenceId,
    pub from_preempted_queue: bool,
    pub prompt_len: usize,
    /// `AienKvManager::incremental_blocks_needed(seq, prompt_len)`.
    pub needed_blocks: usize,
    /// `AienKvManager::available_blocks()` at that instant.
    pub free_blocks: usize,
    /// Blocks promised to sequences admitted earlier in this same step.
    pub reserved_blocks: usize,
    /// `watermark_blocks` for a preempted head, 0 for a waiting head.
    pub watermark_headroom: usize,
    /// `needed_blocks + watermark_headroom`.
    pub min_needed: usize,
    /// `AienKvManager::total_block_count()`.
    pub pool_blocks: usize,
    pub verdict: AdmissionVerdict,
}

/// Everything the scheduler looked at while deciding.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AdmitPreemptAlternatives {
    /// `running_sequences` after `purge_stale_work`, before any preemption.
    pub running_at_entry: Vec<SequenceId>,
    /// `waiting_queue` after `purge_stale_work`, front first.
    pub waiting_at_entry: Vec<SequenceId>,
    /// `preempted_queue` after `purge_stale_work`, front first.
    pub preempted_at_entry: Vec<SequenceId>,
    /// Ranked victims when the pool was below the watermark; empty otherwise.
    pub preemption_victims_considered: Vec<PreemptionVictim>,
    pub running_examined: Vec<RunningItem>,
    pub running_loop_stop: Option<LoopStop>,
    pub admission_examined: Vec<AdmissionCandidate>,
}

/// How `build_scheduled_batch` returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum BatchOutcome {
    Batch {
        prefill_requests: usize,
        decode_requests: usize,
        /// Sum of prefill chunk lengths plus one per decode request, the same
        /// quantity the scheduler compares against `max_batch_tokens`.
        batch_tokens: usize,
    },
    Empty,
    Error(String),
}

/// The production choice, copied from the batch the scheduler built.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmitPreemptChoice {
    /// Victims actually preempted this step (at most one today).
    pub preempted: Vec<SequenceId>,
    /// `decode_requests` in batch order (running decodes, then fork admits).
    pub decode_scheduled: Vec<SequenceId>,
    /// `prefill_requests` in batch order with each chunk's token count.
    pub prefill_scheduled: Vec<(SequenceId, usize)>,
    /// Sequences the prefill gate sent back to the waiting queue head.
    pub reprefill_requeued: Vec<SequenceId>,
    pub outcome: BatchOutcome,
}

/// One decision at `serve.admit_preempt`. Immutable once emitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmitPreemptRecord {
    pub site: &'static str,
    /// `AienScheduler::step_id` after this step's increment; equals the
    /// `step_id` of the batch returned, when one was.
    pub step_id: u64,
    pub limits: BatchLimits,
    /// `arena.active_count()` after purge, before the decision.
    pub arena_active_at_entry: usize,
    pub kv_at_entry: KvReading,
    /// `available < watermark_blocks && !running.is_empty()`, the exact
    /// predicate the watermark check evaluated.
    pub below_watermark_at_entry: bool,
    pub kv_at_exit: KvReading,
    pub running_count_at_exit: usize,
    pub waiting_count_at_exit: usize,
    pub preempted_count_at_exit: usize,
    pub alternatives: AdmitPreemptAlternatives,
    pub choice: AdmitPreemptChoice,
}

/// Read-only tap. The scheduler calls `observe_admit_preempt` once per
/// `build_scheduled_batch`, after the decision is final, with a shared
/// reference; the return type is `()`. Nothing an implementation does can
/// reach the scheduler's state.
pub trait DecisionObserver: Send + Sync {
    fn observe_admit_preempt(&self, record: &AdmitPreemptRecord);
}

/// Observer that keeps every record it is shown. Used by the parity tests
/// and the overhead benchmark; also a reasonable sink for recorded replays.
#[derive(Default)]
pub struct RecordingObserver {
    records: Mutex<Vec<AdmitPreemptRecord>>,
}

impl RecordingObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records
            .lock()
            .expect("recording observer poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes and returns every record seen so far.
    pub fn take(&self) -> Vec<AdmitPreemptRecord> {
        std::mem::take(&mut *self.records.lock().expect("recording observer poisoned"))
    }
}

impl DecisionObserver for RecordingObserver {
    fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) {
        self.records
            .lock()
            .expect("recording observer poisoned")
            .push(record.clone());
    }
}

/// Observer that counts records and the serialized bytes they would occupy,
/// without storing them. Used by the overhead benchmark to separate the cost
/// of producing records from the cost of keeping them.
#[derive(Default)]
pub struct CountingObserver {
    records: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
}

impl CountingObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> u64 {
        self.records.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl DecisionObserver for CountingObserver {
    fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) {
        use std::sync::atomic::Ordering::Relaxed;
        self.records.fetch_add(1, Relaxed);
        let bytes = serde_json::to_vec(record).map(|v| v.len()).unwrap_or(0) as u64;
        self.bytes.fetch_add(bytes, Relaxed);
    }
}

/// Per-step accumulator. Exists only while an observer is installed; every
/// `if let Some(trace)` in `build_scheduled_batch` is skipped otherwise.
#[derive(Debug, Default)]
pub(crate) struct AdmitPreemptTrace {
    pub(crate) arena_active_at_entry: usize,
    pub(crate) kv_at_entry: KvReading,
    pub(crate) below_watermark_at_entry: bool,
    pub(crate) alternatives: AdmitPreemptAlternatives,
    pub(crate) preempted: Vec<SequenceId>,
    pub(crate) decode_scheduled: Vec<SequenceId>,
    pub(crate) prefill_scheduled: Vec<(SequenceId, usize)>,
    pub(crate) reprefill_requeued: Vec<SequenceId>,
}

impl AdmissionCandidate {
    /// A queue head dropped before the KV manager was consulted; every
    /// resource field is zero because the scheduler never computed one.
    pub(crate) fn before_kv(
        seq: SequenceId,
        from_preempted_queue: bool,
        verdict: AdmissionVerdict,
    ) -> Self {
        Self {
            seq,
            from_preempted_queue,
            prompt_len: 0,
            needed_blocks: 0,
            free_blocks: 0,
            reserved_blocks: 0,
            watermark_headroom: 0,
            min_needed: 0,
            pool_blocks: 0,
            verdict,
        }
    }
}
