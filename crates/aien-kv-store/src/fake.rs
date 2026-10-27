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

//! Deterministic in-memory `KvStore` (ADR 0003 section 13): a virtual clock,
//! scripted latencies and faults, no threads, no real time, no I/O. The same
//! script gives the same completions in the same order.
//!
//! The virtual clock is the **only** place I/O and eviction happen: `advance`
//! (and `run_until_idle`, `shutdown`) run the worker's steps. `submit_*`
//! never does, which a test can assert by counting I/O calls (`FakeStats::io_ops`).

use crate::identity::{ChunkKey, NamespaceId};
use crate::pin::{HolderGuard, PinnedDest, PinnedSource};
use crate::record::{ChunkHeader, MAX_SEGMENTS};
use crate::store::{
    CancelOutcome, Capacity, Completion, DrainReport, Incarnation, JobId, JobStage, KvStore,
    PinToken, PublishOutcome, RecoveryProgress, StoreState, Terminal, UnfinishedJob,
};
use aien_platform::{ObjectId, PlatformError};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::string::String;
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::vec::Vec;

/// What `Durable` means (ADR 3.3 table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Native-like: a store ends `Published` (not durable); a later flush job
    /// makes it so.
    OnFlush,
    /// Host-like: publication includes the directory sync; the store's
    /// terminal completion carries `Durable` (or `Indeterminate`).
    WithPublish,
}

/// One injectable misbehaviour, attached to the next job through `JobScript`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// `submit_*` refuses with `QueueFull` regardless of the real occupancy.
    QueueFull,
    /// The write fails with `OutOfMemory`: the filesystem refuses despite the quota (P1).
    DiskFull,
    /// A flush job fails (`HardwareFault`); the store fail-stops (`Failed`) on
    /// `OnFlush`, goes `Degraded` on `WithPublish`.
    FlushFail,
    /// The write is cut short: the job fails and a partial temporary object
    /// remains until `recover_step` (P1, crash style).
    TornWrite,
    /// `fsync(file)` fails (P2): the job fails, nothing is published, the temp
    /// is unlinked, and the store enters `Degraded` (`Failed` on `OnFlush`).
    FsyncFileFail,
    /// Host-only (ignored on `OnFlush`): rename succeeds, `fsync(dir)` fails
    /// (P4): the object is published and served, the job ends `Indeterminate`,
    /// the store goes `Degraded`.
    RenameOkDirsyncFail,
    /// `fsync(file)` takes this many extra virtual microseconds and cannot be
    /// cancelled while it runs.
    StallFsync(u64),
    /// Reclamation of this store job takes this many extra microseconds.
    StallEviction(u64),
    /// The first victim unlink of this store job fails: the victim stays, the
    /// job fails with `HardwareFault`.
    UnlinkFail,
    /// Do not start until `FakeKvStore::release(job)`.
    Hold,
    /// Deliberate adapter misbehaviour for testing callers: every completion
    /// of this job is reported twice (the terminal count becomes 2).
    DuplicateCompletion,
    /// Deliberate adapter misbehaviour for testing callers: after `cancel`
    /// answered `Cancelled` and the terminal `Cancelled` was reported, a stale
    /// `Completed` progress completion is still reported.
    CompletionAfterCancel,
}

/// Latency and faults for the next job submitted. Scripts are consumed in FIFO
/// order, one per `submit_*` call, whatever the call's outcome.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobScript {
    /// Virtual microseconds from start to `Completed` (write plus file sync).
    pub completed_after_us: u64,
    /// Virtual microseconds from `Completed` to publication (rename plus dir sync).
    pub published_after_us: u64,
    pub faults: Vec<Fault>,
}

impl JobScript {
    pub fn latency(completed_after_us: u64, published_after_us: u64) -> Self {
        Self {
            completed_after_us,
            published_after_us,
            faults: Vec::new(),
        }
    }

    pub fn fault(fault: Fault) -> Self {
        Self::default().with(fault)
    }

    pub fn with(mut self, fault: Fault) -> Self {
        self.faults.push(fault);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeConfig {
    pub namespace: NamespaceId,
    pub quota_bytes: u64,
    pub max_objects: u64,
    pub max_restore_ops: u32,
    pub max_park_ops: u32,
    pub max_inflight_bytes: u64,
    pub max_waiters_per_job: u32,
    pub max_waiters_total: u32,
    pub durability: DurabilityMode,
    /// Virtual microseconds one unlink takes.
    pub unlink_us: u64,
    /// Virtual microseconds a flush barrier takes.
    pub flush_us: u64,
    /// Default latencies for jobs with no script.
    pub default_completed_after_us: u64,
    pub default_published_after_us: u64,
    /// Start in `Recovering`; the unindexed disk entries are walked by `recover_step`.
    pub start_recovering: bool,
    /// Disk entries one millisecond of `recover_step` slice accounts for.
    pub recovery_entries_per_ms: u64,
}

impl FakeConfig {
    /// Defaults: 64 MiB quota, 1024 objects, 48 restore and 16 park ops
    /// (ADR 7.1), 256 MiB in flight, 16 waiters per job and 1024 in total,
    /// `WithPublish`, instant unlink and flush, 100 us to complete and 100 us
    /// to publish, open at once.
    pub fn new(namespace: NamespaceId) -> Self {
        Self {
            namespace,
            quota_bytes: 64 << 20,
            max_objects: 1024,
            max_restore_ops: 48,
            max_park_ops: 16,
            max_inflight_bytes: 256 << 20,
            max_waiters_per_job: 16,
            max_waiters_total: 1024,
            durability: DurabilityMode::WithPublish,
            unlink_us: 0,
            flush_us: 0,
            default_completed_after_us: 100,
            default_published_after_us: 100,
            start_recovering: false,
            recovery_entries_per_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FakeStats {
    /// Every I/O-like step: payload write, file sync, rename, dir sync, payload
    /// read, unlink, flush. `submit_*` never changes it.
    pub io_ops: u64,
    pub reads: u64,
    pub writes: u64,
    pub unlinks: u64,
    pub queue_full_total: u64,
    pub disk_full_total: u64,
    pub cancelled_total: u64,
    pub evicted_bytes_total: u64,
    pub dedup_total: u64,
    pub indeterminate_total: u64,
    /// Terminal completions generated, by kind.
    pub terminal_done: u64,
    pub terminal_failed: u64,
    pub terminal_cancelled: u64,
    /// High-water mark of unpolled completions.
    pub peak_completion_queue: u64,
    /// Completions queued beyond the reserved capacity: only the two
    /// deliberate-misbehaviour faults can raise it.
    pub completion_overflow_total: u64,
    /// Reads or writes of pool memory whose holder had already been released.
    pub poison_violations: u64,
    /// Store jobs that reached publication (any durability outcome).
    pub published_total: u64,
    pub shutdown_unfinished_jobs: u64,
    pub shutdown_unfinished_bytes: u64,
}

/// A file found on the fake disk by `recover_step`.
#[derive(Debug, Clone)]
pub enum DiskEntry {
    /// A complete object: indexed when the namespace matches, counted always.
    Object {
        header: ChunkHeader,
        payload: Vec<u8>,
        segment_bytes: usize,
    },
    /// A leftover `tmp/*.part`: unlinked, not counted.
    Tmp,
    /// An object whose header region fails validation: unlinked.
    CorruptHeader { physical_bytes: u64 },
}

type Kb = [u8; 32];

fn kb(k: &ChunkKey) -> Kb {
    k.0 .0
}

const HEADER_REGION: u64 = 4096;

fn digests_of(payload: &[u8], segment_bytes: usize) -> Vec<[u8; 32]> {
    payload
        .chunks(segment_bytes.max(1))
        .map(|s| {
            let mut h = Sha256::new();
            h.update(s);
            h.finalize().into()
        })
        .collect()
}

fn hw(msg: &str) -> PlatformError {
    PlatformError::HardwareFault(String::from(msg))
}

/// A published object on the fake disk.
struct Object {
    header: ChunkHeader,
    payload: Vec<u8>,
    segment_bytes: usize,
    digests: Vec<[u8; 32]>,
    /// Insertion order: the oldest object is evicted first.
    seq: u64,
    publish_seq: u64,
    durable: bool,
    indeterminate: bool,
    evicting: bool,
}

impl Object {
    fn phys(&self) -> u64 {
        self.payload.len() as u64 + HEADER_REGION
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Store,
    Load,
    Flush,
    /// A store of a key that is already published: nothing to write.
    Noop,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Held,
    Start { at: u64 },
    Writing { at: u64 },
    Syncing { at: u64 },
    Publishing { at: u64 },
    Reading { at: u64 },
    Flushing { at: u64 },
    NoopDone { at: u64 },
}

impl Phase {
    fn at(&self) -> Option<u64> {
        match *self {
            Phase::Held => None,
            Phase::Start { at }
            | Phase::Writing { at }
            | Phase::Syncing { at }
            | Phase::Publishing { at }
            | Phase::Reading { at }
            | Phase::Flushing { at }
            | Phase::NoopDone { at } => Some(at),
        }
    }
}

struct Staged {
    payload: Vec<u8>,
    segment_bytes: usize,
}

struct Job {
    id: u64,
    kind: Kind,
    inc: Incarnation,
    key: ChunkKey,
    park: bool,
    /// Credit bytes (`payload_len + 4096`), 0 for a flush or a no-op store.
    bytes: u64,
    payload_len: u64,
    /// Quota reservation still held (store jobs).
    reserved: u64,
    submitted_at: u64,
    stage: JobStage,
    phase: Phase,
    cancelling: bool,
    script: JobScript,
    header: Option<ChunkHeader>,
    src: Option<PinnedSource>,
    dst: Option<PinnedDest>,
    staged: Option<Staged>,
    cover_seq: u64,
    attached: u32,
    noop_stage: JobStage,
    /// Objects (key, generation) this store marked to make room; unlinked
    /// only after it publishes, returned to service otherwise.
    victims: Vec<(Kb, u64)>,
}

impl Job {
    fn has(&self, f: Fault) -> bool {
        self.script.faults.contains(&f)
    }

    fn stall_fsync(&self) -> u64 {
        self.script
            .faults
            .iter()
            .map(|f| if let Fault::StallFsync(x) = f { *x } else { 0 })
            .sum()
    }

    fn stall_eviction(&self) -> u64 {
        self.script
            .faults
            .iter()
            .map(|f| {
                if let Fault::StallEviction(x) = f {
                    *x
                } else {
                    0
                }
            })
            .sum()
    }
}

struct Unlink {
    key: Kb,
    /// The object generation being removed.
    seq: u64,
    at: u64,
    fail: bool,
    evict: bool,
}

struct Inner {
    cfg: FakeConfig,
    now: u64,
    state: StoreState,
    closed: bool,
    next_job: u64,
    next_seq: u64,
    publish_seq: u64,
    next_pin: u64,
    scripts: VecDeque<JobScript>,
    objects: BTreeMap<Kb, Object>,
    foreign_bytes: u64,
    unindexed_bytes: u64,
    unindexed_objects: u64,
    reserved: u64,
    jobs: BTreeMap<u64, Job>,
    inflight_store: BTreeMap<Kb, u64>,
    unlinks: Vec<Unlink>,
    pins: BTreeMap<u64, Vec<Kb>>,
    pin_counts: BTreeMap<Kb, u32>,
    restore_ops: u32,
    park_ops: u32,
    restore_bytes: u64,
    park_bytes: u64,
    attached_total: u32,
    queue: VecDeque<Completion>,
    term_counts: BTreeMap<u64, u32>,
    stats: FakeStats,
    torn: u64,
    disk: VecDeque<DiskEntry>,
    recovery: RecoveryProgress,
    poison_unacked: u64,
    /// Guards of finished jobs; dropped by the caller after the lock is released.
    graveyard: Vec<HolderGuard>,
}

impl Inner {
    fn new(cfg: FakeConfig) -> Self {
        Self {
            now: 0,
            state: if cfg.start_recovering {
                StoreState::Recovering
            } else {
                StoreState::Open
            },
            closed: false,
            next_job: 1,
            next_seq: 1,
            publish_seq: 0,
            next_pin: 1,
            scripts: VecDeque::new(),
            objects: BTreeMap::new(),
            foreign_bytes: 0,
            unindexed_bytes: 0,
            unindexed_objects: 0,
            reserved: 0,
            jobs: BTreeMap::new(),
            inflight_store: BTreeMap::new(),
            unlinks: Vec::new(),
            pins: BTreeMap::new(),
            pin_counts: BTreeMap::new(),
            restore_ops: 0,
            park_ops: 0,
            restore_bytes: 0,
            park_bytes: 0,
            attached_total: 0,
            queue: VecDeque::new(),
            term_counts: BTreeMap::new(),
            stats: FakeStats::default(),
            torn: 0,
            disk: VecDeque::new(),
            recovery: RecoveryProgress::default(),
            poison_unacked: 0,
            graveyard: Vec::new(),
            cfg,
        }
    }

    // ---- accounting ------------------------------------------------------

    fn indexed_bytes(&self) -> u64 {
        self.objects.values().map(Object::phys).sum()
    }

    fn used(&self) -> u64 {
        self.indexed_bytes() + self.foreign_bytes + self.unindexed_bytes
    }

    fn evicting_bytes(&self) -> u64 {
        self.objects
            .values()
            .filter(|o| o.evicting)
            .map(Object::phys)
            .sum()
    }

    fn store_reservations(&self) -> u64 {
        self.jobs
            .values()
            .filter(|j| j.kind == Kind::Store && j.reserved > 0)
            .count() as u64
    }

    fn bytes_after(&self) -> u64 {
        self.used() - self.evicting_bytes() + self.reserved
    }

    fn count_after(&self) -> u64 {
        let live = self.objects.values().filter(|o| !o.evicting).count() as u64;
        live + self.unindexed_objects + self.store_reservations()
    }

    fn is_busy(&self, key: &Kb) -> bool {
        self.pin_counts.get(key).copied().unwrap_or(0) > 0
            || self
                .jobs
                .values()
                .any(|j| matches!(j.kind, Kind::Load | Kind::Noop) && kb(&j.key) == *key)
    }

    /// (bytes, count) of objects eviction could reclaim now.
    fn evictable(&self) -> (u64, u64) {
        let mut b = 0;
        let mut n = 0;
        for (k, o) in &self.objects {
            if !o.evicting && !self.is_busy(k) {
                b += o.phys();
                n += 1;
            }
        }
        (b, n)
    }

    fn pick_victim(&self) -> Option<Kb> {
        self.objects
            .iter()
            .filter(|(k, o)| !o.evicting && !self.is_busy(k))
            .min_by_key(|(_, o)| o.seq)
            .map(|(k, _)| *k)
    }

    fn completion_capacity(&self) -> usize {
        3 * (self.cfg.max_restore_ops as usize + self.cfg.max_park_ops as usize)
    }

    fn park_byte_limit(&self) -> u64 {
        self.cfg.max_inflight_bytes / 4
    }

    fn restore_byte_limit(&self) -> u64 {
        self.cfg.max_inflight_bytes - self.park_byte_limit()
    }

    fn capacity(&self) -> Capacity {
        Capacity {
            state: self.state,
            quota_bytes: self.cfg.quota_bytes,
            used_bytes: self.used(),
            reserved_bytes: self.reserved,
            evicting_bytes: self.evicting_bytes(),
            unindexed_bytes: self.unindexed_bytes,
            unindexed_objects: self.unindexed_objects,
            objects: self.objects.len() as u64,
            max_objects: self.cfg.max_objects,
            restore_ops: self.restore_ops,
            max_restore_ops: self.cfg.max_restore_ops,
            park_ops: self.park_ops,
            max_park_ops: self.cfg.max_park_ops,
            inflight_bytes: self.restore_bytes + self.park_bytes,
            max_inflight_bytes: self.cfg.max_inflight_bytes,
            max_waiters_per_job: self.cfg.max_waiters_per_job,
        }
    }

    fn set_degraded(&mut self) {
        if self.state == StoreState::Open {
            self.state = StoreState::Degraded;
        }
    }

    // ---- completions -----------------------------------------------------

    fn push(&mut self, c: Completion) {
        self.queue.push_back(c);
        let len = self.queue.len();
        if len > self.completion_capacity() {
            self.stats.completion_overflow_total += 1;
        }
        self.stats.peak_completion_queue = self.stats.peak_completion_queue.max(len as u64);
    }

    fn progress(&mut self, job: &Job, stage: JobStage) {
        let c = Completion {
            job: JobId(job.id),
            incarnation: job.inc,
            key: job.key,
            stage,
            terminal: None,
            publish: PublishOutcome::NotApplicable,
            bytes: job.payload_len,
            elapsed_us: self.now - job.submitted_at,
            holder: None,
        };
        self.push(c.clone());
        if job.has(Fault::DuplicateCompletion) {
            self.push(c);
        }
    }

    /// The one place a job ends: releases its credits and reservation, queues
    /// its terminal completion and hands its buffers to the graveyard.
    fn finish(
        &mut self,
        mut job: Job,
        terminal: Terminal,
        stage: JobStage,
        publish: PublishOutcome,
    ) {
        if job.park {
            self.park_ops -= 1;
            self.park_bytes -= job.bytes;
        } else {
            self.restore_ops -= 1;
            self.restore_bytes -= job.bytes;
        }
        self.reserved -= job.reserved;
        job.reserved = 0;
        if job.kind == Kind::Store {
            let published = matches!(terminal, Terminal::Done(_));
            self.settle_victims(&mut job, published);
            if self.inflight_store.get(&kb(&job.key)) == Some(&job.id) {
                self.inflight_store.remove(&kb(&job.key));
            }
        }
        self.attached_total -= job.attached;
        match &terminal {
            Terminal::Done(_) => self.stats.terminal_done += 1,
            Terminal::Failed(_) => self.stats.terminal_failed += 1,
            Terminal::Cancelled => {
                self.stats.terminal_cancelled += 1;
                self.stats.cancelled_total += 1;
            }
        }
        let copies = if job.has(Fault::DuplicateCompletion) {
            2
        } else {
            1
        };
        *self.term_counts.entry(job.id).or_insert(0) += copies;
        let c = Completion {
            job: JobId(job.id),
            incarnation: job.inc,
            key: job.key,
            stage,
            terminal: Some(terminal.clone()),
            publish,
            bytes: job.payload_len,
            elapsed_us: self.now - job.submitted_at,
            holder: None,
        };
        for _ in 0..copies {
            self.push(c.clone());
        }
        if terminal == Terminal::Cancelled && job.has(Fault::CompletionAfterCancel) {
            let mut stale = c;
            stale.stage = JobStage::Completed;
            stale.terminal = None;
            stale.publish = PublishOutcome::NotApplicable;
            self.push(stale);
        }
        if let Some(s) = job.src.take() {
            self.graveyard.push(s.into_guard());
        }
        if let Some(d) = job.dst.take() {
            self.graveyard.push(d.into_guard());
        }
    }

    /// A store job ended. Its victims were only marked; queue their unlinks
    /// when it published (`StallEviction` delays them, `UnlinkFail` fails the
    /// first), otherwise return them to service (ADR 3.4, Grok C4 P1-1).
    fn settle_victims(&mut self, job: &mut Job, published: bool) {
        let victims = std::mem::take(&mut job.victims);
        let mut first = true;
        for (key, seq) in victims {
            if published {
                let fail = first && job.has(Fault::UnlinkFail);
                first = false;
                self.unlinks.push(Unlink {
                    key,
                    seq,
                    at: self.now + self.cfg.unlink_us + job.stall_eviction(),
                    fail,
                    evict: true,
                });
            } else if let Some(o) = self.objects.get_mut(&key).filter(|o| o.seq == seq) {
                o.evicting = false;
            }
        }
    }

    fn fail(&mut self, job: Job, e: PlatformError) {
        let stage = job.stage;
        self.finish(
            job,
            Terminal::Failed(e),
            stage,
            PublishOutcome::NotApplicable,
        );
    }

    fn violation(&mut self) {
        self.stats.poison_violations += 1;
        self.poison_unacked += 1;
    }

    // ---- submit ----------------------------------------------------------

    fn take_script(&mut self) -> JobScript {
        self.scripts.pop_front().unwrap_or_else(|| {
            JobScript::latency(
                self.cfg.default_completed_after_us,
                self.cfg.default_published_after_us,
            )
        })
    }

    fn queue_full(&mut self) -> PlatformError {
        self.stats.queue_full_total += 1;
        PlatformError::QueueFull
    }

    /// Refusals that depend only on the store's state (and the scripted fault).
    fn gate(&mut self, script: &JobScript, needs_open: bool) -> Result<(), PlatformError> {
        if script.faults.contains(&Fault::QueueFull) || self.state == StoreState::Recovering {
            return Err(self.queue_full());
        }
        if self.closed
            || self.state == StoreState::Failed
            || (needs_open && self.state == StoreState::Degraded)
        {
            return Err(PlatformError::DeviceUnavailable);
        }
        Ok(())
    }

    fn credit_check(&mut self, park: bool, bytes: u64) -> Result<(), PlatformError> {
        let (ops, max_ops, used_bytes, limit) = if park {
            (
                self.park_ops,
                self.cfg.max_park_ops,
                self.park_bytes,
                self.park_byte_limit(),
            )
        } else {
            (
                self.restore_ops,
                self.cfg.max_restore_ops,
                self.restore_bytes,
                self.restore_byte_limit(),
            )
        };
        let slots = self.queue.len() + 3 * self.jobs.len() + 3;
        if ops >= max_ops || used_bytes + bytes > limit || slots > self.completion_capacity() {
            return Err(self.queue_full());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn new_job(
        &mut self,
        kind: Kind,
        inc: Incarnation,
        key: ChunkKey,
        bytes: u64,
        payload_len: u64,
        reserved: u64,
        script: JobScript,
        phase: Phase,
    ) -> Job {
        let id = self.next_job;
        self.next_job += 1;
        let park = kind != Kind::Load;
        if park {
            self.park_ops += 1;
            self.park_bytes += bytes;
        } else {
            self.restore_ops += 1;
            self.restore_bytes += bytes;
        }
        self.reserved += reserved;
        Job {
            id,
            kind,
            inc,
            key,
            park,
            bytes,
            payload_len,
            reserved,
            submitted_at: self.now,
            stage: JobStage::Submitted,
            phase,
            cancelling: false,
            script,
            header: None,
            src: None,
            dst: None,
            staged: None,
            cover_seq: 0,
            attached: 0,
            noop_stage: JobStage::Submitted,
            victims: Vec::new(),
        }
    }

    fn start_phase(&self, script: &JobScript) -> Phase {
        if script.faults.contains(&Fault::Hold) {
            Phase::Held
        } else {
            Phase::Start { at: self.now }
        }
    }

    fn submit_store(
        &mut self,
        header: ChunkHeader,
        slot: &mut Option<PinnedSource>,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let script = self.take_script();
        self.gate(&script, true)?;
        let src = slot.as_ref().ok_or(PlatformError::InvalidLayout)?;
        // The header and the source must agree (ADR 3.2); nothing is reserved on refusal.
        if header.namespace_id != self.cfg.namespace
            || header.payload_len == 0
            || src.total_bytes() != Some(header.payload_len)
            || header.block_size == 0
            || src.block_count() == 0
            || src.block_count() > MAX_SEGMENTS
        {
            return Err(PlatformError::InvalidLayout);
        }
        let full = (src.block_count() as u64 - 1) * header.block_size as u64;
        let tail = (header.token_count as u64).checked_sub(full);
        match tail {
            Some(t)
                if t >= 1
                    && t <= header.block_size as u64
                    && t == src.tail_valid_tokens() as u64 => {}
            _ => return Err(PlatformError::InvalidLayout),
        }
        let key = header.chunk_key;
        let k = kb(&key);
        // Never dedup into a job that is being cancelled: it will never publish.
        if let Some(&jid) = self
            .inflight_store
            .get(&k)
            .filter(|jid| self.jobs.get(jid).is_some_and(|j| !j.cancelling))
        {
            self.stats.dedup_total += 1;
            return Ok(JobId(jid));
        }
        if let Some(o) = self.objects.get(&k).filter(|o| !o.evicting) {
            let stage = if o.durable {
                JobStage::Durable
            } else {
                JobStage::Published
            };
            self.credit_check(true, 0)?;
            let at = self.now;
            let mut job = self.new_job(
                Kind::Noop,
                inc,
                key,
                0,
                0,
                0,
                script,
                Phase::NoopDone { at },
            );
            job.noop_stage = stage;
            self.stats.dedup_total += 1;
            let id = job.id;
            self.jobs.insert(id, job);
            return Ok(JobId(id));
        }
        let bytes = header.payload_len + HEADER_REGION;
        self.credit_check(true, bytes)?;
        let (eb, en) = self.evictable();
        if self.bytes_after() + bytes > self.cfg.quota_bytes + eb
            || self.count_after() + 1 > self.cfg.max_objects + en
        {
            self.stats.disk_full_total += 1;
            return Err(PlatformError::OutOfMemory);
        }
        let phase = self.start_phase(&script);
        let mut job = self.new_job(
            Kind::Store,
            inc,
            key,
            bytes,
            header.payload_len,
            bytes,
            script,
            phase,
        );
        job.header = Some(header);
        job.src = slot.take();
        let id = job.id;
        self.inflight_store.insert(k, id);
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    fn submit_load(
        &mut self,
        key: ChunkKey,
        expect: &ChunkHeader,
        slot: &mut Option<PinnedDest>,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let script = self.take_script();
        self.gate(&script, false)?;
        let dst = slot.as_ref().ok_or(PlatformError::InvalidLayout)?;
        let o = self
            .objects
            .get(&kb(&key))
            .filter(|o| !o.evicting)
            .ok_or(PlatformError::ObjectNotFound(key.0))?;
        let mut a = o.header;
        let mut b = *expect;
        a.written_unix_ns = 0;
        b.written_unix_ns = 0;
        if a != b
            || dst.total_bytes() != Some(o.payload.len() as u64)
            || dst.block_bytes() != o.segment_bytes
        {
            return Err(PlatformError::InvalidLayout);
        }
        let (bytes, len) = (o.phys(), o.payload.len() as u64);
        self.credit_check(false, bytes)?;
        let phase = self.start_phase(&script);
        let mut job = self.new_job(Kind::Load, inc, key, bytes, len, 0, script, phase);
        job.dst = slot.take();
        let id = job.id;
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    fn submit_flush(&mut self, inc: Incarnation) -> Result<JobId, PlatformError> {
        let script = self.take_script();
        // A degraded store may still make what it published durable.
        self.gate(&script, false)?;
        self.credit_check(true, 0)?;
        let at = self.now + self.cfg.flush_us;
        let mut job = self.new_job(
            Kind::Flush,
            inc,
            ChunkKey(ObjectId([0; 32])),
            0,
            0,
            0,
            script,
            Phase::Flushing { at },
        );
        job.cover_seq = self.publish_seq;
        let id = job.id;
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    // ---- the worker ------------------------------------------------------

    fn reap_cancels(&mut self) {
        let ids: Vec<u64> = self
            .jobs
            .iter()
            .filter(|(_, j)| j.cancelling)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let job = self.jobs.remove(&id).expect("cancelling job");
            if job.staged.is_some() {
                // The temporary object was written and synced: unlink it.
                self.stats.unlinks += 1;
                self.stats.io_ops += 1;
            }
            let stage = job.stage;
            self.finish(
                job,
                Terminal::Cancelled,
                stage,
                PublishOutcome::NotApplicable,
            );
        }
    }

    /// Run every due step in time order. `target: None` runs until nothing is
    /// scheduled. With `jump`, the clock ends at `target`.
    fn run(&mut self, target: Option<u64>, jump: bool) {
        self.reap_cancels();
        loop {
            let mut best: Option<(u64, u8, u64)> = None;
            for (i, u) in self.unlinks.iter().enumerate() {
                let c = (u.at, 0u8, i as u64);
                if best.is_none_or(|b| c < b) {
                    best = Some(c);
                }
            }
            for (id, j) in &self.jobs {
                if j.cancelling {
                    continue;
                }
                if let Some(at) = j.phase.at() {
                    let c = (at, 1u8, *id);
                    if best.is_none_or(|b| c < b) {
                        best = Some(c);
                    }
                }
            }
            let Some((at, kind, id)) = best else { break };
            if target.is_some_and(|t| at > t) {
                break;
            }
            self.now = self.now.max(at);
            if kind == 0 {
                self.do_unlink(id as usize);
            } else {
                self.step_job(id);
            }
        }
        if jump {
            if let Some(t) = target {
                self.now = self.now.max(t);
            }
        }
    }

    fn do_unlink(&mut self, idx: usize) {
        let u = self.unlinks.remove(idx);
        self.stats.io_ops += 1;
        let present = self.objects.get(&u.key).is_some_and(|o| o.seq == u.seq);
        if u.fail {
            // Still on disk: back in service, still counted. No job waits.
            if let Some(o) = self.objects.get_mut(&u.key).filter(|_| present) {
                o.evicting = false;
            }
            return;
        }
        if present {
            let o = self.objects.remove(&u.key).expect("object present");
            self.stats.unlinks += 1;
            if u.evict {
                self.stats.evicted_bytes_total += o.phys();
            }
        }
    }

    fn step_job(&mut self, id: u64) {
        let mut job = self.jobs.remove(&id).expect("scheduled job");
        match (job.kind, job.phase) {
            (Kind::Store, Phase::Start { .. }) => self.start_store(job),
            (Kind::Store, Phase::Writing { .. }) => self.end_write(job),
            (Kind::Store, Phase::Syncing { .. }) => self.end_sync(job),
            (Kind::Store, Phase::Publishing { .. }) => self.publish(job),
            (Kind::Load, Phase::Start { .. }) => {
                job.phase = Phase::Reading {
                    at: self.now + job.script.completed_after_us,
                };
                self.jobs.insert(id, job);
            }
            (Kind::Load, Phase::Reading { .. }) => self.end_read(job),
            (Kind::Flush, Phase::Flushing { .. }) => self.end_flush(job),
            (Kind::Noop, Phase::NoopDone { .. }) => {
                let stage = job.noop_stage;
                self.finish(
                    job,
                    Terminal::Done(stage),
                    stage,
                    PublishOutcome::NotApplicable,
                );
            }
            (kind, _) => {
                let _ = kind;
                unreachable!("job scheduled in a phase it cannot be in")
            }
        }
    }

    /// Worker-side reclamation (ADR 7.4): mark victims, then write at once.
    /// The victims' unlinks wait for the publish (`settle_victims`).
    fn start_store(&mut self, mut job: Job) {
        while self.bytes_after() > self.cfg.quota_bytes
            || self.count_after() + 1 > self.cfg.max_objects
        {
            // `job` is not in the map, so `count_after` misses its own reservation
            // (bytes use the scalar `self.reserved`, which already holds it): add 1.
            let Some(victim) = self.pick_victim() else {
                // Nothing evictable (everything pinned or in flight): the need cannot be met.
                self.fail(job, PlatformError::OutOfMemory);
                return;
            };
            let o = self.objects.get_mut(&victim).expect("victim");
            o.evicting = true;
            job.victims.push((victim, o.seq));
        }
        job.phase = Phase::Writing {
            at: self.now + job.script.completed_after_us,
        };
        self.jobs.insert(job.id, job);
    }

    fn end_write(&mut self, mut job: Job) {
        if job.has(Fault::DiskFull) {
            self.stats.disk_full_total += 1;
            self.fail(job, PlatformError::OutOfMemory);
            return;
        }
        if job.has(Fault::TornWrite) {
            self.torn += 1;
            self.fail(job, hw("torn write"));
            return;
        }
        let src = job.src.as_ref().expect("store source");
        let mut data = Vec::with_capacity(job.payload_len as usize);
        for i in 0..src.block_count() {
            match src.block(i) {
                Ok(b) => data.extend_from_slice(b),
                Err(e) => {
                    self.violation();
                    self.fail(job, e);
                    return;
                }
            }
        }
        let segment_bytes = src.block_bytes();
        job.staged = Some(Staged {
            payload: data,
            segment_bytes,
        });
        let stall = job.stall_fsync();
        if stall > 0 || job.has(Fault::FsyncFileFail) {
            job.phase = Phase::Syncing {
                at: self.now + stall,
            };
            self.jobs.insert(job.id, job);
        } else {
            self.complete_write(job);
        }
    }

    fn end_sync(&mut self, mut job: Job) {
        if job.has(Fault::FsyncFileFail) {
            // P2: nothing is published, the temporary object is unlinked.
            job.staged = None;
            self.stats.unlinks += 1;
            self.stats.io_ops += 1;
            if self.cfg.durability == DurabilityMode::OnFlush {
                self.state = StoreState::Failed;
            } else {
                self.set_degraded();
            }
            self.fail(job, hw("fsync(file) failed"));
        } else {
            self.complete_write(job);
        }
    }

    fn complete_write(&mut self, mut job: Job) {
        self.stats.writes += 1;
        self.stats.io_ops += 2; // payload write and file sync
        job.stage = JobStage::Completed;
        self.progress(&job, JobStage::Completed);
        job.phase = Phase::Publishing {
            at: self.now + job.script.published_after_us,
        };
        self.jobs.insert(job.id, job);
    }

    fn publish(&mut self, mut job: Job) {
        let staged = job.staged.take().expect("staged payload");
        let header = job.header.expect("store header");
        let with_publish = self.cfg.durability == DurabilityMode::WithPublish;
        let dirsync_fail = with_publish && job.has(Fault::RenameOkDirsyncFail);
        self.stats.io_ops += if with_publish { 2 } else { 1 };
        self.stats.published_total += 1;
        self.publish_seq += 1;
        let seq = self.next_seq;
        self.next_seq += 1;
        let digests = digests_of(&staged.payload, staged.segment_bytes);
        self.objects.insert(
            kb(&job.key),
            Object {
                header,
                payload: staged.payload,
                segment_bytes: staged.segment_bytes,
                digests,
                seq,
                publish_seq: self.publish_seq,
                durable: with_publish && !dirsync_fail,
                indeterminate: dirsync_fail,
                evicting: false,
            },
        );
        // The reservation became used bytes.
        self.reserved -= job.reserved;
        job.reserved = 0;
        job.stage = JobStage::Published;
        if dirsync_fail {
            self.stats.indeterminate_total += 1;
            self.set_degraded();
            self.finish(
                job,
                Terminal::Failed(hw("fsync(dir) failed after rename")),
                JobStage::Published,
                PublishOutcome::Indeterminate,
            );
        } else if with_publish {
            job.stage = JobStage::Durable;
            self.finish(
                job,
                Terminal::Done(JobStage::Durable),
                JobStage::Durable,
                PublishOutcome::Durable,
            );
        } else {
            self.finish(
                job,
                Terminal::Done(JobStage::Published),
                JobStage::Published,
                PublishOutcome::NotApplicable,
            );
        }
    }

    fn end_read(&mut self, mut job: Job) {
        let k = kb(&job.key);
        let Some(o) = self.objects.get(&k) else {
            let key = job.key.0;
            self.fail(job, PlatformError::ObjectNotFound(key));
            return;
        };
        let good = digests_of(&o.payload, o.segment_bytes) == o.digests;
        if !good {
            // A failed hash removes the object (cache semantics, ADR 1.4).
            self.objects.remove(&k);
            self.stats.unlinks += 1;
            self.stats.io_ops += 2;
            self.fail(job, PlatformError::InvalidLayout);
            return;
        }
        let o = self.objects.get(&k).expect("object");
        let dst = job.dst.as_mut().expect("load destination");
        for (i, seg) in o.payload.chunks(o.segment_bytes).enumerate() {
            if let Err(e) = dst.write_block(i, seg) {
                self.violation();
                self.fail(job, e);
                return;
            }
        }
        self.stats.reads += 1;
        self.stats.io_ops += 1;
        job.stage = JobStage::Completed;
        self.finish(
            job,
            Terminal::Done(JobStage::Completed),
            JobStage::Completed,
            PublishOutcome::NotApplicable,
        );
    }

    fn end_flush(&mut self, mut job: Job) {
        self.stats.io_ops += 1;
        if job.has(Fault::FlushFail) {
            if self.cfg.durability == DurabilityMode::OnFlush {
                self.state = StoreState::Failed;
            } else {
                self.set_degraded();
            }
            self.fail(job, hw("flush failed"));
            return;
        }
        let cover = job.cover_seq;
        for o in self.objects.values_mut() {
            if o.publish_seq <= cover && !o.indeterminate {
                o.durable = true;
            }
        }
        job.stage = JobStage::Durable;
        self.finish(
            job,
            Terminal::Done(JobStage::Durable),
            JobStage::Durable,
            PublishOutcome::Durable,
        );
    }

    // ---- recovery --------------------------------------------------------

    fn recover_step(&mut self, slice_ms: u64) -> RecoveryProgress {
        for _ in 0..std::mem::take(&mut self.torn) {
            self.disk.push_back(DiskEntry::Tmp);
        }
        let per_ms = self.cfg.recovery_entries_per_ms.max(1);
        let budget = slice_ms.saturating_mul(per_ms);
        let mut n = 0;
        while n < budget {
            let Some(e) = self.disk.pop_front() else {
                break;
            };
            n += 1;
            self.stats.io_ops += 1;
            match e {
                DiskEntry::Tmp => {
                    self.recovery.tmp_deleted += 1;
                    self.stats.unlinks += 1;
                }
                DiskEntry::CorruptHeader { .. } => {
                    self.recovery.corrupt_deleted += 1;
                    self.stats.unlinks += 1;
                }
                DiskEntry::Object {
                    header,
                    payload,
                    segment_bytes,
                } => {
                    let phys = payload.len() as u64 + HEADER_REGION;
                    self.recovery.bytes += phys;
                    if header.namespace_id != self.cfg.namespace {
                        self.foreign_bytes += phys;
                        self.recovery.foreign_namespace += 1;
                    } else if (self.objects.len() as u64) < self.cfg.max_objects {
                        let seq = self.next_seq;
                        self.next_seq += 1;
                        let digests = digests_of(&payload, segment_bytes);
                        self.objects.insert(
                            kb(&header.chunk_key),
                            Object {
                                header,
                                payload,
                                segment_bytes,
                                digests,
                                seq,
                                publish_seq: 0,
                                durable: true,
                                indeterminate: false,
                                evicting: false,
                            },
                        );
                        self.recovery.indexed += 1;
                    } else {
                        self.unindexed_objects += 1;
                        self.unindexed_bytes += phys;
                        self.recovery.unindexed_objects += 1;
                        self.recovery.unindexed_bytes += phys;
                    }
                }
            }
        }
        self.recovery.elapsed_us += n.saturating_mul(1000).div_ceil(per_ms);
        self.recovery.done = self.disk.is_empty();
        if self.recovery.done && self.state == StoreState::Recovering {
            self.state = StoreState::Open;
        }
        self.recovery
    }
}

/// Deterministic in-memory `KvStore`; see the module docs.
pub struct FakeKvStore {
    inner: Mutex<Inner>,
}

impl FakeKvStore {
    pub fn new(config: FakeConfig) -> Self {
        Self {
            inner: Mutex::new(Inner::new(config)),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `f` under the lock, then drop released holder guards outside it (ADR 8).
    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let (r, garbage) = {
            let mut g = self.lock();
            let r = f(&mut g);
            (r, std::mem::take(&mut g.graveyard))
        };
        drop(garbage);
        r
    }

    /// Queue a script for the next `submit_*` call.
    pub fn script(&self, script: JobScript) {
        self.with(|i| i.scripts.push_back(script));
    }

    pub fn now_us(&self) -> u64 {
        self.with(|i| i.now)
    }

    /// Move the virtual clock forward, running every due step in time order.
    pub fn advance(&self, us: u64) {
        self.with(|i| {
            let t = i.now + us;
            i.run(Some(t), true);
        });
    }

    /// Run every scheduled step, jumping the clock to each; held jobs stay held.
    pub fn run_until_idle(&self) {
        self.with(|i| i.run(None, false));
    }

    /// End a `Hold`: the job starts at the current time.
    pub fn release(&self, job: JobId) -> bool {
        self.with(|i| {
            let now = i.now;
            match i.jobs.get_mut(&job.0) {
                Some(j) if j.phase == Phase::Held => {
                    j.phase = Phase::Start { at: now };
                    true
                }
                _ => false,
            }
        })
    }

    pub fn stats(&self) -> FakeStats {
        self.with(|i| i.stats)
    }

    pub fn payload_of(&self, key: &ChunkKey) -> Option<Vec<u8>> {
        self.with(|i| i.objects.get(&kb(key)).map(|o| o.payload.clone()))
    }

    /// `Some(true)` after publication on `WithPublish` or after a flush that
    /// started later than the publication on `OnFlush`.
    pub fn is_durable(&self, key: &ChunkKey) -> Option<bool> {
        self.with(|i| i.objects.get(&kb(key)).map(|o| o.durable))
    }

    pub fn is_indeterminate(&self, key: &ChunkKey) -> Option<bool> {
        self.with(|i| i.objects.get(&kb(key)).map(|o| o.indeterminate))
    }

    /// Flip one payload byte of a published object (a hash failure on next load).
    pub fn corrupt_payload(&self, key: &ChunkKey) -> bool {
        self.with(|i| match i.objects.get_mut(&kb(key)) {
            Some(o) if !o.payload.is_empty() => {
                o.payload[0] ^= 0xFF;
                true
            }
            _ => false,
        })
    }

    /// Put a published object in place without a job (test setup). It is durable.
    pub fn inject_object(&self, header: ChunkHeader, payload: Vec<u8>, segment_bytes: usize) {
        self.with(|i| {
            let seq = i.next_seq;
            i.next_seq += 1;
            let digests = digests_of(&payload, segment_bytes);
            i.objects.insert(
                kb(&header.chunk_key),
                Object {
                    header,
                    payload,
                    segment_bytes,
                    digests,
                    seq,
                    publish_seq: 0,
                    durable: true,
                    indeterminate: false,
                    evicting: false,
                },
            );
        });
    }

    /// Put a file on the fake disk that only `recover_step` will find.
    pub fn inject_disk_entry(&self, entry: DiskEntry) {
        self.with(|i| i.disk.push_back(entry));
    }

    /// Partial temporary objects left by torn writes, until `recover_step`.
    pub fn torn_residue(&self) -> u64 {
        self.with(|i| i.torn)
    }

    /// Terminal completions generated for `job` so far (1 once it ended, in
    /// every case but the duplicate-completion fault).
    pub fn terminal_count(&self, job: JobId) -> u32 {
        self.with(|i| i.term_counts.get(&job.0).copied().unwrap_or(0))
    }

    /// Jobs that were accepted and have not reached their terminal completion.
    pub fn live_jobs(&self) -> Vec<JobId> {
        self.with(|i| i.jobs.keys().map(|k| JobId(*k)).collect())
    }

    /// Stage a live job has reached.
    pub fn job_stage(&self, job: JobId) -> Option<JobStage> {
        self.with(|i| i.jobs.get(&job.0).map(|j| j.stage))
    }

    /// Completions generated and not yet polled.
    pub fn pending_completions(&self) -> usize {
        self.with(|i| i.queue.len())
    }

    /// Completion queue capacity: three slots per possible in-flight op.
    pub fn completion_capacity(&self) -> usize {
        self.with(|i| i.completion_capacity())
    }

    /// Acknowledge recorded poison violations (they otherwise fail the test
    /// when the store is dropped); returns how many there were.
    pub fn take_poison_violations(&self) -> u64 {
        self.with(|i| std::mem::take(&mut i.poison_unacked))
    }

    /// True when the internal lock is free right now. A holder-release
    /// callback calls this to prove it runs outside the store's lock (ADR 8).
    pub fn lock_is_free(&self) -> bool {
        match self.inner.try_lock() {
            Ok(_) => true,
            Err(TryLockError::Poisoned(_)) => true,
            Err(TryLockError::WouldBlock) => false,
        }
    }
}

impl Drop for FakeKvStore {
    fn drop(&mut self) {
        let unacked = self.lock().poison_unacked;
        if unacked > 0 && !std::thread::panicking() {
            panic!(
                "fake adapter: {unacked} access(es) to pool memory whose holder was already released"
            );
        }
    }
}

impl KvStore for FakeKvStore {
    fn namespace(&self) -> NamespaceId {
        self.with(|i| i.cfg.namespace)
    }

    fn capacity(&self) -> Capacity {
        self.with(|i| i.capacity())
    }

    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>> {
        self.with(|i| {
            let serving = !matches!(i.state, StoreState::Recovering | StoreState::Failed);
            keys.iter()
                .map(|k| {
                    i.objects
                        .get(&kb(k))
                        .filter(|o| serving && !o.evicting)
                        .map(|o| o.payload.len() as u64)
                })
                .collect()
        })
    }

    fn submit_store(
        &self,
        header: ChunkHeader,
        src: PinnedSource,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        // Whatever stays in `slot` is dropped here, after the lock is released.
        let mut slot = Some(src);
        self.with(|i| i.submit_store(header, &mut slot, inc))
    }

    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let mut slot = Some(dst);
        self.with(|i| i.submit_load(key, expect, &mut slot, inc))
    }

    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError> {
        self.with(|i| i.submit_flush(inc))
    }

    fn attach_waiter(&self, job: JobId, _inc: Incarnation) -> Result<(), PlatformError> {
        self.with(|i| {
            let per_job = i.cfg.max_waiters_per_job;
            let total = i.cfg.max_waiters_total;
            let attached_total = i.attached_total;
            let j = match i.jobs.get_mut(&job.0) {
                Some(j) if j.kind == Kind::Load && !j.cancelling => j,
                _ => {
                    return Err(PlatformError::TaskRejected(String::from(
                        "not an in-flight load",
                    )))
                }
            };
            // The requester that submitted the load is the first waiter.
            if 1 + j.attached >= per_job || attached_total >= total {
                i.stats.queue_full_total += 1;
                return Err(PlatformError::QueueFull);
            }
            j.attached += 1;
            i.attached_total += 1;
            Ok(())
        })
    }

    fn poll(&self, max: usize) -> Vec<Completion> {
        self.with(|i| {
            let n = max.min(i.queue.len());
            i.queue.drain(..n).collect()
        })
    }

    fn cancel(&self, job: JobId) -> CancelOutcome {
        self.with(|i| match i.jobs.get_mut(&job.0) {
            None => CancelOutcome::TooLate,
            Some(j) if j.cancelling => CancelOutcome::Cancelled,
            Some(j) => {
                let cancellable = !matches!(
                    (j.kind, j.phase),
                    (Kind::Flush | Kind::Noop, _) | (_, Phase::Syncing { .. })
                );
                if cancellable {
                    j.cancelling = true;
                    CancelOutcome::Cancelled
                } else {
                    CancelOutcome::TooLate
                }
            }
        })
    }

    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError> {
        self.with(|i| {
            if i.state == StoreState::Recovering {
                return Err(i.queue_full());
            }
            if i.state == StoreState::Failed || i.closed {
                return Err(PlatformError::DeviceUnavailable);
            }
            if i.pins.len() as u32 >= i.cfg.max_restore_ops {
                return Err(i.queue_full());
            }
            for k in keys {
                if !i.objects.get(&kb(k)).is_some_and(|o| !o.evicting) {
                    return Err(PlatformError::ObjectNotFound(k.0));
                }
            }
            let id = i.next_pin;
            i.next_pin += 1;
            let list: Vec<Kb> = keys.iter().map(kb).collect();
            for k in &list {
                *i.pin_counts.entry(*k).or_insert(0) += 1;
            }
            i.pins.insert(id, list);
            Ok(PinToken(id))
        })
    }

    fn unpin(&self, token: PinToken) {
        self.with(|i| {
            if let Some(list) = i.pins.remove(&token.0) {
                for k in list {
                    if let Some(c) = i.pin_counts.get_mut(&k) {
                        *c -= 1;
                        if *c == 0 {
                            i.pin_counts.remove(&k);
                        }
                    }
                }
            }
        })
    }

    fn delete(&self, keys: &[ChunkKey]) -> u64 {
        self.with(|i| {
            if matches!(i.state, StoreState::Recovering | StoreState::Failed) {
                return 0;
            }
            let mut marked = 0;
            for k in keys {
                let k = kb(k);
                if i.is_busy(&k) || i.inflight_store.contains_key(&k) {
                    continue;
                }
                let Some(o) = i.objects.get_mut(&k).filter(|o| !o.evicting) else {
                    continue;
                };
                o.evicting = true;
                let seq = o.seq;
                i.unlinks.push(Unlink {
                    key: k,
                    seq,
                    at: i.now + i.cfg.unlink_us,
                    fail: false,
                    evict: false,
                });
                marked += 1;
            }
            marked
        })
    }

    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress {
        self.with(|i| i.recover_step(slice_ms))
    }

    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport {
        self.with(|i| {
            i.closed = true;
            let published_before = i.stats.published_total;
            // Cancel what can be cancelled; stores past the file sync finish.
            let mut cancelled = 0;
            for j in i.jobs.values_mut() {
                let cancellable = !matches!(
                    (j.kind, j.phase),
                    (Kind::Flush | Kind::Noop, _)
                        | (Kind::Store, Phase::Publishing { .. })
                        | (_, Phase::Syncing { .. })
                );
                if cancellable && !j.cancelling {
                    j.cancelling = true;
                }
                if cancellable {
                    cancelled += 1;
                }
            }
            let deadline = i.now + intake_deadline_ms.saturating_mul(1000);
            i.run(Some(deadline), false);
            let unfinished: Vec<UnfinishedJob> = i
                .jobs
                .values()
                .map(|j| UnfinishedJob {
                    job: JobId(j.id),
                    key: j.key,
                    bytes: j.bytes,
                    stage: j.stage,
                })
                .collect();
            if !unfinished.is_empty() {
                i.now = i.now.max(deadline);
            }
            i.stats.shutdown_unfinished_jobs = unfinished.len() as u64;
            i.stats.shutdown_unfinished_bytes = unfinished.iter().map(|u| u.bytes).sum();
            DrainReport {
                published: i.stats.published_total - published_before,
                cancelled,
                unfinished,
            }
        })
    }
}
