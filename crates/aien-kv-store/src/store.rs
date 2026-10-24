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

//! The `KvStore` contract (ADR 0003 section 3.2, revised by section 13).
//!
//! Binding rules, in one place:
//!
//! * `submit_*` is bounded and non-blocking. `submit_store` only **reserves**
//!   (`payload + 4096` bytes of quota); it never evicts and never touches the
//!   filesystem. Reclamation and unlinks run on the worker (section 7.4).
//! * Every accepted job yields **exactly one terminal completion** (done,
//!   failed or cancelled), including cancelled and failed jobs. A caller that
//!   cancels or abandons a job still polls that terminal completion before it
//!   frees the job's memory (section 5.5).
//! * Completion capacity is reserved at submit (three slots per job), so
//!   `poll` never overflows and the worker never blocks on a full queue.
//! * While the store is `Recovering`, `lookup` misses and `submit_*` refuses
//!   with `QueueFull` (section 4.1).
//! * Rename done but directory sync failed: the object is served, the job ends
//!   with `PublishOutcome::Indeterminate`, the store goes `Degraded`.
//! * `shutdown` returns at its deadline with a `DrainReport` naming every job
//!   still running; nothing is freed to meet the deadline (section 7.5).

use crate::identity::{ChunkKey, NamespaceId};
use crate::pin::{HolderGuard, HolderHandle, PinnedDest, PinnedSource};
use crate::record::ChunkHeader;
use aien_platform::PlatformError;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Never reused within a store lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub u64);

/// The caller's token, echoed in every completion so stale completions can be
/// dropped (ADR 5.3). The full tuple `(seq_id, epoch)`: the scheduler's
/// sequence id already uses all 64 bits (`slot << 32 | generation`,
/// aien-scheduler/src/sequence.rs `to_u64`), so the residency epoch cannot be
/// packed beside it. Two tokens name the same request only if both fields
/// match; a reused sequence id in a later epoch is a different incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Incarnation {
    pub seq_id: u64,
    pub epoch: u32,
}

impl Incarnation {
    pub const fn new(seq_id: u64, epoch: u32) -> Self {
        Self { seq_id, epoch }
    }

    /// Epoch 0: for callers that have no residency epoch (flush barriers,
    /// fixtures).
    pub const fn seq(seq_id: u64) -> Self {
        Self { seq_id, epoch: 0 }
    }
}

/// Returned by `pin`; hand it back to `unpin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PinToken(pub u64);

/// How far a job has got. Ordered: later stages compare greater.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobStage {
    /// Accepted; source pinned; counted in the in-flight bounds.
    Submitted,
    /// Store: bytes written to the temporary object and synced. Load: every
    /// segment landed in the destination and verified.
    Completed,
    /// Store: visible to `lookup` and to recovery. Not a load stage (a load is
    /// published by the caller when it commits the table, ADR 3.3).
    Published,
    /// Store: persistence is claimed. Flush: the barrier finished.
    Durable,
}

/// How a store job ended with respect to durability (ADR 2.4, P4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishOutcome {
    /// Loads, flushes, failed and cancelled jobs, and stores whose durability
    /// is claimed later by a flush (`OnFlush` adapters).
    NotApplicable,
    /// Published and durable at the same instant (host adapter).
    Durable,
    /// Published (complete, indexed, served) but its survival across power loss
    /// is unknown: the directory sync failed after a successful rename.
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreState {
    /// A full pass over the store has not finished; admission is closed.
    Recovering,
    Open,
    /// An indeterminate publication or a failed file sync: loads are served,
    /// no new stores until reopen.
    Degraded,
    /// Fail-stop: nothing is served until reopen.
    Failed,
}

/// Exactly one terminal per accepted job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminal {
    Done(JobStage),
    Failed(PlatformError),
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub job: JobId,
    pub incarnation: Incarnation,
    /// The chunk key; all zero for a flush.
    pub key: ChunkKey,
    /// The stage this completion reports (for a failed job, the last stage reached).
    pub stage: JobStage,
    /// `None`: progress report (a store reached `Completed`). `Some`: the
    /// job's one terminal completion; nothing follows it.
    pub terminal: Option<Terminal>,
    pub publish: PublishOutcome,
    pub bytes: u64,
    pub elapsed_us: u64,
    /// Load terminals from the [`crate::worker::StorageWorker`]: the landing
    /// holder, handed back so it outlives the terminal the caller sees (Grok
    /// C5 P1-2). Every waiter of a coalesced load gets a copy; the holder is
    /// released when the last copy is dropped (after S3 adopted its blocks,
    /// or on any failure). Raw stores release at the terminal: `None`.
    pub holder: Option<LandingHolder>,
}

/// A shared landing holder (see [`Completion::holder`]).
#[derive(Clone)]
pub struct LandingHolder(Arc<HolderGuard>);

impl LandingHolder {
    pub fn new(guard: HolderGuard) -> Self {
        Self(Arc::new(guard))
    }

    pub fn handle(&self) -> HolderHandle {
        self.0.handle()
    }
}

impl core::fmt::Debug for LandingHolder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("LandingHolder")
    }
}

impl PartialEq for LandingHolder {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for LandingHolder {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    pub state: StoreState,
    pub quota_bytes: u64,
    /// Physical bytes of every object found or published, any namespace, plus
    /// `unindexed_bytes` (ADR 4.1).
    pub used_bytes: u64,
    /// Submit-time reservations of stores that have not published yet.
    pub reserved_bytes: u64,
    /// Bytes of objects marked for removal whose unlink has not returned.
    pub evicting_bytes: u64,
    pub unindexed_bytes: u64,
    pub unindexed_objects: u64,
    pub objects: u64,
    pub max_objects: u64,
    pub restore_ops: u32,
    pub max_restore_ops: u32,
    pub park_ops: u32,
    pub max_park_ops: u32,
    pub inflight_bytes: u64,
    pub max_inflight_bytes: u64,
    pub max_waiters_per_job: u32,
}

/// Where a resumable recovery pass stands (ADR 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecoveryProgress {
    /// The walk finished and admission opened (or the store is `Degraded`).
    pub done: bool,
    /// The store is `Failed`: recovery stopped and is never `done` (Grok C4 P2-6).
    pub failed: bool,
    pub indexed: u64,
    pub bytes: u64,
    pub unindexed_objects: u64,
    pub unindexed_bytes: u64,
    pub foreign_namespace: u64,
    pub corrupt_deleted: u64,
    pub tmp_deleted: u64,
    pub elapsed_us: u64,
}

/// `Cancelled`: the job never touched the destination and published nothing.
/// `TooLate`: it will complete (or already did); ignore by incarnation. Either
/// way the job yields its one terminal completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    Cancelled,
    TooLate,
}

/// A job `shutdown` could not finish by its deadline (ADR 7.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnfinishedJob {
    pub job: JobId,
    pub key: ChunkKey,
    pub bytes: u64,
    pub stage: JobStage,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DrainReport {
    /// Stores that were already past the file sync and finished their publication.
    pub published: u64,
    /// Jobs cancelled during the drain.
    pub cancelled: u64,
    /// Jobs still running at the deadline. Their buffers stay owned by the
    /// store until each reaches its terminal completion.
    pub unfinished: Vec<UnfinishedJob>,
}

pub trait KvStore: Send + Sync {
    fn namespace(&self) -> NamespaceId;
    fn capacity(&self) -> Capacity;
    /// Index only, no I/O, O(keys). `Some(len)` means a published object of
    /// `len` payload bytes exists. While `state` is `Recovering` (or `Failed`),
    /// and for keys marked for removal, every key is a miss.
    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>>;
    /// Bounded and non-blocking; never evicts and never touches the
    /// filesystem. Reserves `payload + 4096` bytes of quota. Refuses with
    /// `QueueFull` when the class's ops, its bytes or the completion slots
    /// would exceed their limits (and always while `Recovering`),
    /// `OutOfMemory` when `used + reserved + request` exceeds quota plus the
    /// bytes that are evictable (unpinned, not in flight, not already
    /// evicting), `DeviceUnavailable` when `Degraded`, `Failed` or shut down,
    /// `InvalidLayout` when the header and source disagree. A store of a key
    /// that is already in flight returns the existing job's id.
    fn submit_store(
        &self,
        header: ChunkHeader,
        src: PinnedSource,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError>;
    /// Same refusals as `submit_store` (a `Degraded` store still serves loads),
    /// plus `ObjectNotFound` for a key that is not published and
    /// `InvalidLayout` when `expect` or `dst` disagrees with the stored object.
    /// Nothing is written to `dst` before validation passes.
    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError>;
    /// Durability barrier for everything `Published` before the call.
    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError>;
    /// Coalescing: register one more waiter, named by its own `inc`, on an
    /// in-flight load; `QueueFull` past `max_waiters_per_job` waiters on the job
    /// or the total bound. A raw store (fake, host adapter) runs one job and
    /// yields one terminal carrying the opener's token; the storage worker
    /// fans that out to one terminal per waiter, each with its own token
    /// (ADR 5.3), so no completion ever stands for several incarnations.
    fn attach_waiter(&self, job: JobId, inc: Incarnation) -> Result<(), PlatformError>;
    /// Non-blocking: drains at most `max` completions, oldest first.
    fn poll(&self, max: usize) -> Vec<Completion>;
    /// Returns immediately. The destination or source of the job is safe to
    /// release only after the job's terminal completion has been polled
    /// (ADR 5.5).
    fn cancel(&self, job: JobId) -> CancelOutcome;
    /// Pins against eviction and deletion while a restore chain is being
    /// assembled. All-or-nothing; `QueueFull` past `max_restore_ops` live tokens.
    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError>;
    fn unpin(&self, token: PinToken);
    /// Marks keys for removal (lookups miss at once); skips pinned keys and
    /// keys with a job in flight; the unlink runs on the worker and the bytes
    /// are subtracted when it returns. Returns the number of keys marked.
    fn delete(&self, keys: &[ChunkKey]) -> u64;
    /// Resumable recovery: call until `done`, each call bounded by `slice_ms`.
    /// Admission stays closed until `done`.
    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress;
    /// Stops intake, cancels what can be cancelled, lets jobs past the file
    /// sync finish, and returns at `intake_deadline_ms` with the jobs still
    /// running listed in `unfinished`.
    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport;
}
