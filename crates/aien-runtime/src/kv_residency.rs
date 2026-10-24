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

//! Persistent KV in the runtime (cut C6, ADR 0003 sections 5, 6, 7.3 and 8):
//! park at the scheduler's finish transaction, restore at submit, and the
//! per-sequence residency states that keep both honest.
//!
//! Who runs what (ADR 5.4 and 8):
//!
//! * **S1 finish-fork** runs in [`ParkHook::on_finish`], inside the scheduler's
//!   finish transaction under the KV write guard it already holds, before the
//!   sequence's own `free_sequence`. It only forks one holder per chunk
//!   (`fork_range`) and pushes a [`ParkIntent`] to an outbox. No store call
//!   except the O(keys) index `lookup`, no I/O.
//! * The spine drains the outbox right after the scheduler step returns and
//!   calls `submit_store` for each intent with no lock held.
//! * **S2 restore-alloc** runs on the spine thread in `submit_*`: one holder
//!   per chunk, refused when the pool would fall below the scheduler's
//!   watermark.
//! * **S3 restore-commit** runs on the spine thread at the start of `step`,
//!   after `poll`, never on the storage worker: verify the incarnation and that
//!   every job of the transaction is `Completed`, `assemble_table`, the
//!   backend's `restore_sequence`, then the scheduler's
//!   `admit_restored_prefix`.
//! * **S4 holder-release** is the release callback of each holder guard. It
//!   takes the KV write lock, so a guard is never dropped while this thread
//!   holds that lock (every drop below happens after the guard was released).
//!
//! States (ADR 5.1). `Resident` is the absence of an entry. A park does not
//! need a per-sequence state: the sequence is freed after the fork, and the
//! park jobs are tracked by job id until their terminals.
//!
//! ```text
//! submit: chain hit --S2--> Restoring{txn}
//! Restoring --every job Completed + S3--> Prefilling --suffix prefilled--> Resident
//! Restoring --any job fails / cancelled--> Draining{txn} --every job terminal--> Failed
//! Failed --plain prefill started--> Resident
//! ```
//!
//! A completion is committed only if its incarnation `(seq_id, epoch)` equals
//! the transaction's and the scheduler still knows the sequence; anything else
//! is counted in `store_stale_completions_total` and only drops its holder copy.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aien_abi_core::AienInferenceBackend;
use aien_kv_cache::{holder_id, AienKvManager, KvLayout, SharedKvManager, SuffixPolicy};
use aien_kv_store::holder::HolderGenerations;
use aien_kv_store::worker::{LoadTicket, StorageWorker};
use aien_kv_store::{
    chunk_chain, ChunkHeader, ChunkKey, Completion, HolderGuard, Incarnation, JobId, KvError,
    KvStore, LandingHolder, LayoutStrides, NamespaceFields, NamespaceId, PinToken, PinnedDest,
    PinnedSource, PublishOutcome, Terminal,
};
use aien_platform::PlatformError;
use aien_scheduler::{AienScheduler, FinishContext, FinishHook, SequenceId, SequencePhase};
use parking_lot::Mutex;

/// Completions drained per `poll` call; the loop repeats until the queue is empty.
const POLL_BATCH: usize = 256;

/// Chain heads remembered by the retry window at once. Past it a new failure
/// is not recorded (that head keeps being looked up), so memory stays bounded
/// whatever the request mix (DOCS SILENT on a bound; C7 choice).
const RETRY_HEADS_MAX: usize = 4096;

/// A monotonic millisecond clock. The daemon uses [`monotonic_ms_clock`];
/// tests inject their own so no assertion depends on wall-clock sleeps.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Milliseconds since the clock was made, from `std::time::Instant`.
pub fn monotonic_ms_clock() -> Clock {
    let t0 = std::time::Instant::now();
    Arc::new(move || t0.elapsed().as_millis() as u64)
}

/// What the daemon needs at shutdown: the storage worker and the configured
/// intake deadline. [`KvDrainHandle::drain`] blocks for at most about the
/// deadline (ADR 7.5); run it off the async runtime's worker threads.
pub struct KvDrainHandle {
    store: Arc<StorageWorker>,
    deadline_ms: u64,
}

impl KvDrainHandle {
    /// Stops intake, cancels what has not started, lets jobs past the file
    /// sync finish, and returns at the deadline with every unfinished job
    /// named. Nothing is freed to meet the deadline: the worker keeps each
    /// job's buffers until its terminal completion.
    pub fn drain(&self) -> aien_kv_store::DrainReport {
        self.store.shutdown(self.deadline_ms)
    }

    pub fn deadline_ms(&self) -> u64 {
        self.deadline_ms
    }
}

/// The byte strides of the pool layout as the namespace records them (ADR 1.2
/// field 10).
pub fn layout_strides(layout: &KvLayout) -> LayoutStrides {
    LayoutStrides {
        head_stride: layout.head_stride_bytes as u64,
        token_stride: layout.token_stride_bytes as u64,
        kv_plane_stride: layout.kv_plane_stride_bytes as u64,
        layer_stride: layout.layer_stride_bytes as u64,
        block_stride: layout.block_stride_bytes as u64,
        element_bytes: layout.element_bytes as u64,
    }
}

/// Settings of the residency layer (METRICS_AND_SETTINGS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvResidencyConfig {
    /// `AIEN_KV_CHUNK_BLOCKS`: chunk size in pool blocks.
    pub chunk_blocks: u32,
    /// `AIEN_KV_CONTEXT_TOKENS`: the restorable prefix never exceeds it.
    pub context_cap: Option<usize>,
    /// `AIEN_KV_RESTORE_MAX_BLOCKS`: holder blocks of restores in flight at once.
    pub restore_max_blocks: usize,
    /// `AIEN_KV_PARK`.
    pub park: bool,
    /// `AIEN_KV_RESTORE`.
    pub restore: bool,
    /// `AIEN_KV_RESTORE_RETRY_WINDOW_MS`: after two failed restores of one
    /// chain head within this many milliseconds, that head is prefilled
    /// without a lookup until the older failure leaves the window.
    pub restore_retry_window_ms: u64,
    /// `AIEN_KV_STORE_DRAIN_MS` (parsed by `WorkerConfig`): the shutdown intake
    /// deadline handed to the store (ADR 7.5).
    pub drain_ms: u64,
}

/// The namespace inputs with owned strings, so the residency layer can live as
/// long as the runtime. [`KvIdentity::fields`] lends them back as the
/// borrowed [`NamespaceFields`] the codec takes.
#[derive(Debug, Clone)]
pub struct KvIdentity {
    template: NamespaceFields<'static>,
    model_digest_kind: String,
    model_id: String,
    sharing_namespace: String,
    namespace: NamespaceId,
}

impl KvIdentity {
    /// Copies `fields` and computes the namespace id (refused as the codec
    /// refuses it).
    pub fn new(fields: &NamespaceFields<'_>) -> Result<Self, KvError> {
        let namespace = fields.namespace_id()?;
        Ok(Self {
            template: NamespaceFields {
                model_digest_kind: "",
                model_id: "",
                sharing_namespace: "",
                ..*fields
            },
            model_digest_kind: fields.model_digest_kind.to_string(),
            model_id: fields.model_id.to_string(),
            sharing_namespace: fields.sharing_namespace.to_string(),
            namespace,
        })
    }

    pub fn fields(&self) -> NamespaceFields<'_> {
        NamespaceFields {
            model_digest_kind: &self.model_digest_kind,
            model_id: &self.model_id,
            sharing_namespace: &self.sharing_namespace,
            ..self.template
        }
    }

    pub fn namespace(&self) -> NamespaceId {
        self.namespace
    }
}

/// Counters (METRICS_AND_SETTINGS names without the `kv_` prefix).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvResidencyStats {
    /// Chunks forked into holders at a finish transaction (S1).
    pub park_intents_total: u64,
    /// Finish transactions with nothing to park (no computed prefix, table not
    /// computed, no pool, or parking off).
    pub park_skipped_total: u64,
    /// Chunks not parked because the store already has them, or a store of the
    /// same key was already in flight.
    pub park_dedup_total: u64,
    /// Park jobs accepted by the store.
    pub park_jobs_total: u64,
    pub park_bytes_total: u64,
    /// Park jobs whose terminal reported publication.
    pub park_published_total: u64,
    /// Refused at submit, or ended failed or cancelled.
    pub park_failed_total: u64,
    /// Finish hooks that failed (the sequence was freed anyway).
    pub park_hook_failed_total: u64,
    /// Restore loads this runtime started or attached to.
    pub restore_jobs_total: u64,
    pub restore_bytes_total: u64,
    /// Loads that attached to an in-flight load of the same key (ADR 7.3).
    pub restore_coalesced_total: u64,
    /// Transactions that ended without a restore after a job failed.
    pub restore_failed_total: u64,
    /// Requests that got a plain prefill instead of a restore (miss, refusal or failure).
    pub restore_fallback_prefill_total: u64,
    /// S2 refusals because the pool would fall below the watermark.
    pub restore_refused_watermark_total: u64,
    /// Transactions committed (S3 and the scheduler admission succeeded).
    pub restore_committed_total: u64,
    pub restore_tokens_total: u64,
    /// Restore transactions abandoned because their request left the scheduler.
    pub restore_abandoned_total: u64,
    /// Park publications reported `Indeterminate` (served, durability unknown;
    /// ADR 2.4), whatever the terminal says (the adapters report it on a
    /// failed terminal after the rename).
    pub park_indeterminate_total: u64,
    /// Requests prefilled without a lookup because their chain head failed
    /// twice within the retry window. Also counted as fallbacks.
    pub restore_retry_suppressed_total: u64,
    pub store_stale_completions_total: u64,
    pub store_cancelled_total: u64,
    pub store_queue_full_total: u64,
}

/// One sequence's residency (ADR 5.1). `Resident` means "no entry".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyState {
    Resident,
    Restoring,
    Draining,
    Failed,
    Prefilling,
}

/// A point-in-time view for status lines and tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvResidencyStatus {
    pub stats: KvResidencyStats,
    pub restoring: usize,
    pub draining: usize,
    pub failed: usize,
    pub prefilling: usize,
    /// Park jobs between submit and their terminal.
    pub parking_jobs: usize,
    /// `kv_pool_holder_blocks`.
    pub holder_blocks: usize,
    /// `kv_restore_inflight_blocks`: holder blocks of restores not yet adopted
    /// or drained.
    pub restore_inflight_blocks: usize,
    /// Waiters of this runtime still expecting a terminal from an in-flight
    /// restore load: the opener plus every attached waiter, summed over the
    /// landings. Zero once every restore is terminal; anything else after that
    /// is a leak of the attached-waiter count.
    pub landing_waiters: usize,
}

/// What S1 hands the spine: one chunk's holder and header.
#[derive(Debug)]
pub struct ParkIntent {
    holder: u64,
    generation: u32,
    blocks: Vec<usize>,
    tail_valid: u32,
    header: ChunkHeader,
    inc: Incarnation,
}

/// The part shared by the finish hook (scheduler side) and the spine.
struct Shared {
    kv: SharedKvManager,
    store: Arc<StorageWorker>,
    identity: KvIdentity,
    chunk_tokens: usize,
    block_size: usize,
    park: bool,
    /// Leaf lock; taken under the KV lock (S1, S2) or with no lock held (S4).
    gens: Arc<Mutex<HolderGenerations>>,
    /// Leaf lock held for one push or one take (ADR 8 rule 4).
    outbox: Mutex<Vec<ParkIntent>>,
    hook_counts: Mutex<HookCounts>,
}

#[derive(Default)]
struct HookCounts {
    intents: u64,
    skipped: u64,
    dedup: u64,
    failed: u64,
}

fn now_unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// The guard of one holder: dropping it frees the holder's table (S4) and then
/// its generation. Never dropped while the dropping thread holds the KV lock.
fn holder_guard(
    kv: &SharedKvManager,
    gens: &Arc<Mutex<HolderGenerations>>,
    holder: u64,
    generation: u32,
) -> HolderGuard {
    let kv = kv.clone();
    let gens = gens.clone();
    HolderGuard::new(move || {
        let _ = kv.write().free_sequence(holder);
        gens.lock().release(generation);
    })
}

/// Allocates a holder generation and id. The generation is released on refusal.
fn alloc_holder(gens: &Mutex<HolderGenerations>) -> Result<(u64, u32), String> {
    let generation = gens
        .lock()
        .alloc()
        .map_err(|e| format!("holder generations: {e:?}"))?;
    match holder_id(u64::from(generation)) {
        Ok(id) => Ok((id, generation)),
        Err(e) => {
            gens.lock().release(generation);
            Err(e)
        }
    }
}

/// S1 (ADR 6.1): installed as the scheduler's finish hook.
pub struct ParkHook {
    shared: Arc<Shared>,
}

impl ParkHook {
    fn skip(&self) -> Result<(), String> {
        self.shared.hook_counts.lock().skipped += 1;
        Ok(())
    }

    /// Frees holders forked so far in this transaction (we hold the KV guard,
    /// so this is a direct free, not a guard drop).
    fn undo(&self, kv: &mut AienKvManager, made: Vec<ParkIntent>) {
        for it in made {
            let _ = kv.free_sequence(it.holder);
            self.shared.gens.lock().release(it.generation);
        }
    }
}

impl FinishHook for ParkHook {
    fn on_finish(&self, kv: &mut AienKvManager, ctx: &FinishContext<'_>) -> Result<(), String> {
        let sh = &self.shared;
        if !sh.park {
            return self.skip();
        }
        let Some(prefix) = ctx.computed_prefix.as_deref() else {
            return self.skip();
        };
        if prefix.is_empty() || kv.pool().is_none() {
            return self.skip();
        }
        match kv.get_block_table(ctx.request_id) {
            Some(t)
                if t.prefill_state.is_ready()
                    && t.total_tokens >= prefix.len()
                    && t.block_ids.len() * sh.block_size >= prefix.len() => {}
            _ => return self.skip(),
        }
        let fields = sh.identity.fields();
        let ns = sh.identity.namespace();
        let keys = chunk_chain(&ns, sh.chunk_tokens as u32, prefix).map_err(|e| {
            sh.hook_counts.lock().failed += 1;
            format!("park: key chain: {e}")
        })?;
        // Index-only lookup (ADR 8 rule 3; lock order KV -> store index).
        let found = sh.store.lookup(&keys);
        let inc = Incarnation::seq(ctx.request_id);
        let written = now_unix_ns();
        let mut made: Vec<ParkIntent> = Vec::new();
        let mut dedup = 0u64;
        for (i, key) in keys.iter().enumerate() {
            if found.get(i).copied().flatten().is_some() {
                dedup += 1;
                continue;
            }
            let first = i * sh.chunk_tokens;
            let tokens = (prefix.len() - first).min(sh.chunk_tokens);
            let first_block = first / sh.block_size;
            let n_blocks = tokens.div_ceil(sh.block_size);
            let tail_valid = (tokens - (n_blocks - 1) * sh.block_size) as u32;
            let header = match fields.chunk_header(
                &ns,
                *key,
                i.checked_sub(1).map(|p| &keys[p]),
                first as u32,
                tokens as u32,
                written,
            ) {
                Ok(h) => h,
                Err(e) => {
                    self.undo(kv, made);
                    sh.hook_counts.lock().failed += 1;
                    return Err(format!("park: header: {e}"));
                }
            };
            let (holder, generation) = match alloc_holder(&sh.gens) {
                Ok(v) => v,
                Err(e) => {
                    self.undo(kv, made);
                    sh.hook_counts.lock().failed += 1;
                    return Err(format!("park: {e}"));
                }
            };
            match kv.fork_range(ctx.request_id, first_block, n_blocks, holder) {
                Ok(blocks) => made.push(ParkIntent {
                    holder,
                    generation,
                    blocks,
                    tail_valid,
                    header,
                    inc,
                }),
                Err(e) => {
                    sh.gens.lock().release(generation);
                    self.undo(kv, made);
                    sh.hook_counts.lock().failed += 1;
                    return Err(format!("park: fork_range: {e}"));
                }
            }
        }
        {
            let mut c = sh.hook_counts.lock();
            c.intents += made.len() as u64;
            c.dedup += dedup;
        }
        sh.outbox.lock().extend(made);
        Ok(())
    }
}

#[derive(Debug)]
enum SlotState {
    Pending,
    /// Completed and validated; the landing holder copy is kept until S3
    /// adopted its blocks (or the transaction drained).
    Completed(Option<LandingHolder>),
    /// Failed or cancelled terminal received.
    Ended,
}

#[derive(Debug)]
struct ChunkSlot {
    job: JobId,
    holder: u64,
    state: SlotState,
}

#[derive(Debug)]
struct RestoreTxn {
    inc: Incarnation,
    /// The first chunk key of the prompt chain (retry-window bookkeeping).
    head: ChunkKey,
    restored: usize,
    blocks: usize,
    chunks: Vec<ChunkSlot>,
    pin: Option<PinToken>,
}

impl RestoreTxn {
    fn pending(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| matches!(c.state, SlotState::Pending))
            .count()
    }

    fn completed(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| matches!(c.state, SlotState::Completed(_)))
            .count()
    }

    /// Takes every landing holder copy out of the transaction (the caller drops
    /// them with no lock held).
    fn take_landings(&mut self) -> Vec<LandingHolder> {
        self.chunks
            .iter_mut()
            .filter_map(|c| match &mut c.state {
                SlotState::Completed(h) => h.take(),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug)]
struct Entry {
    seq: SequenceId,
    state: ResidencyState,
    epoch: u32,
    txn: Option<RestoreTxn>,
}

impl Entry {
    fn transition(&mut self, state: ResidencyState) {
        self.state = state;
        self.epoch = self.epoch.wrapping_add(1);
    }
}

/// S2 output: (holder id, generation, blocks) per chunk, the pool base and
/// the block size.
type Allocated = (Vec<(u64, u32, Vec<usize>)>, *mut u8, usize);

/// A restore load another transaction of this runtime may attach to: which
/// holder the bytes land in, and how many of our waiters still expect a
/// terminal.
#[derive(Debug)]
struct Landing {
    holder: u64,
    waiters: u32,
}

/// The spine-owned residency map and the park/restore machinery.
pub struct KvResidency {
    shared: Arc<Shared>,
    config: KvResidencyConfig,
    watermark_blocks: usize,
    entries: HashMap<u64, Entry>,
    /// Park jobs in flight: job -> payload bytes.
    parks: HashMap<JobId, u64>,
    landings: HashMap<JobId, Landing>,
    restore_inflight_blocks: usize,
    stats: KvResidencyStats,
    /// Milliseconds on a monotonic clock (injected by tests).
    clock: Clock,
    /// Chain head -> times (ms) of its failed restores still inside the window.
    retry_failures: HashMap<ChunkKey, Vec<u64>>,
}

impl KvResidency {
    /// `store` must serve `identity`'s namespace; `kv` must be the pooled
    /// manager the scheduler and the backend share. Chunks are whole blocks
    /// (`chunk_blocks * block_size` tokens), which is what the identity's
    /// `chunk_tokens` must say.
    pub fn new(
        config: KvResidencyConfig,
        identity: KvIdentity,
        store: Arc<StorageWorker>,
        kv: SharedKvManager,
    ) -> Result<Self, String> {
        let fields = identity.fields();
        let block_size = kv.read().block_size();
        if config.chunk_blocks == 0 || block_size == 0 {
            return Err("kv residency: chunk_blocks and block_size must be nonzero".into());
        }
        let chunk_tokens = config.chunk_blocks as usize * block_size;
        if fields.block_size as usize != block_size || fields.chunk_tokens as usize != chunk_tokens
        {
            return Err(format!(
                "kv residency: identity says block {} chunk {}, pool says block {block_size} chunk {chunk_tokens}",
                fields.block_size, fields.chunk_tokens
            ));
        }
        if store.namespace() != identity.namespace() {
            return Err("kv residency: store namespace differs from the identity".into());
        }
        {
            let guard = kv.read();
            let pool = guard
                .pool()
                .ok_or("kv residency: the KV manager has no tensor pool")?;
            if pool.block_bytes() as u64 != fields.layout.block_stride {
                return Err("kv residency: pool block bytes differ from the layout".into());
            }
        }
        let park = config.park;
        Ok(Self {
            shared: Arc::new(Shared {
                kv,
                store,
                identity,
                chunk_tokens,
                block_size,
                park,
                gens: Arc::new(Mutex::new(HolderGenerations::new())),
                outbox: Mutex::new(Vec::new()),
                hook_counts: Mutex::new(HookCounts::default()),
            }),
            config,
            watermark_blocks: 0,
            entries: HashMap::new(),
            parks: HashMap::new(),
            landings: HashMap::new(),
            restore_inflight_blocks: 0,
            stats: KvResidencyStats::default(),
            clock: monotonic_ms_clock(),
            retry_failures: HashMap::new(),
        })
    }

    /// The scheduler hook (S1). The spine installs it.
    pub fn finish_hook(&self) -> Arc<dyn FinishHook> {
        Arc::new(ParkHook {
            shared: self.shared.clone(),
        })
    }

    /// Replaces the retry-window clock (tests).
    pub fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// The store's admission state and accounting (`kv_store_*` status fields).
    pub fn store_capacity(&self) -> aien_kv_store::Capacity {
        self.shared.store.capacity()
    }

    /// The `status` reply fields (METRICS_AND_SETTINGS names).
    pub fn persist_report(&self) -> crate::control::KvPersistReport {
        use aien_kv_store::StoreState;
        let s = self.status();
        let c = self.store_capacity();
        let st = &s.stats;
        crate::control::KvPersistReport {
            kv_store_state: match c.state {
                StoreState::Recovering => "recovering",
                StoreState::Open => "open",
                StoreState::Degraded => "degraded",
                StoreState::Failed => "failed",
            }
            .to_string(),
            kv_store_persisted_bytes: c.used_bytes,
            kv_store_persisted_objects: c.objects + c.unindexed_objects,
            kv_store_quota_bytes: c.quota_bytes,
            kv_store_max_objects: c.max_objects,
            kv_store_over_cap: c.unindexed_objects > 0 || c.used_bytes > c.quota_bytes,
            kv_store_reserved_bytes: c.reserved_bytes,
            kv_store_unindexed_bytes: c.unindexed_bytes,
            kv_store_unindexed_objects: c.unindexed_objects,
            kv_store_inflight_ops: u64::from(c.restore_ops) + u64::from(c.park_ops),
            kv_store_inflight_bytes: c.inflight_bytes,
            kv_store_restore_ops: u64::from(c.restore_ops),
            kv_store_park_ops: u64::from(c.park_ops),
            kv_restore_inflight_blocks: s.restore_inflight_blocks as u64,
            kv_park_jobs_total: st.park_jobs_total,
            kv_park_bytes_total: st.park_bytes_total,
            kv_park_failed_total: st.park_failed_total,
            kv_park_dedup_total: st.park_dedup_total,
            kv_park_indeterminate_total: st.park_indeterminate_total,
            kv_restore_jobs_total: st.restore_jobs_total,
            kv_restore_bytes_total: st.restore_bytes_total,
            kv_restore_failed_total: st.restore_failed_total,
            kv_restore_coalesced_total: st.restore_coalesced_total,
            kv_restore_fallback_prefill_total: st.restore_fallback_prefill_total,
            kv_restore_retry_suppressed_total: st.restore_retry_suppressed_total,
            kv_restore_tokens_total: st.restore_tokens_total,
            kv_store_stale_completions_total: st.store_stale_completions_total,
            kv_store_cancelled_total: st.store_cancelled_total,
            kv_store_queue_full_total: st.store_queue_full_total,
            kv_store_refused_reason: String::new(),
            kv_store_key_refused_total: aien_kv_store::seal::key_refusals_total(),
            kv_store_seal_rejected_total: aien_kv_store::seal::open_rejected_total(),
        }
    }

    /// The shutdown drain (ADR 7.5) with the configured deadline.
    pub fn drain_handle(&self) -> KvDrainHandle {
        KvDrainHandle {
            store: self.shared.store.clone(),
            deadline_ms: self.config.drain_ms,
        }
    }

    /// Failures of `head` still inside the window at `now`.
    fn recent_failures(&self, head: &ChunkKey, now: u64) -> usize {
        let window = self.config.restore_retry_window_ms;
        self.retry_failures.get(head).map_or(0, |ts| {
            ts.iter()
                .filter(|&&t| now.saturating_sub(t) < window)
                .count()
        })
    }

    /// Records a failed restore of the chain starting at `head`. Only the two
    /// newest failures matter; entries whose failures all left the window are
    /// dropped first.
    fn record_restore_failure(&mut self, head: ChunkKey) {
        let now = (self.clock)();
        let window = self.config.restore_retry_window_ms;
        self.retry_failures
            .retain(|_, ts| ts.iter().any(|&t| now.saturating_sub(t) < window));
        if !self.retry_failures.contains_key(&head) && self.retry_failures.len() >= RETRY_HEADS_MAX
        {
            return;
        }
        let ts = self.retry_failures.entry(head).or_default();
        ts.push(now);
        if ts.len() > 2 {
            ts.remove(0);
        }
    }

    /// The scheduler's watermark, used by S2 (ADR 7.1).
    pub fn set_watermark_blocks(&mut self, blocks: usize) {
        self.watermark_blocks = blocks;
    }

    pub fn namespace(&self) -> NamespaceId {
        self.shared.identity.namespace()
    }

    pub fn chunk_tokens(&self) -> usize {
        self.shared.chunk_tokens
    }

    pub fn residency_of(&self, seq: u64) -> ResidencyState {
        self.entries
            .get(&seq)
            .map_or(ResidencyState::Resident, |e| e.state)
    }

    /// Store job ids of `seq`'s restore transaction, in chunk order.
    pub fn restore_jobs_of(&self, seq: u64) -> Vec<JobId> {
        self.entries
            .get(&seq)
            .and_then(|e| e.txn.as_ref())
            .map(|t| t.chunks.iter().map(|c| c.job).collect())
            .unwrap_or_default()
    }

    /// `(completed chunks, total chunks)` of `seq`'s restore transaction.
    pub fn restore_progress(&self, seq: u64) -> Option<(usize, usize)> {
        let t = self.entries.get(&seq)?.txn.as_ref()?;
        Some((t.completed(), t.chunks.len()))
    }

    pub fn status(&self) -> KvResidencyStatus {
        let mut stats = self.stats.clone();
        {
            let c = self.shared.hook_counts.lock();
            stats.park_intents_total = c.intents;
            stats.park_skipped_total = c.skipped;
            stats.park_dedup_total += c.dedup;
            stats.park_hook_failed_total = c.failed;
        }
        let count = |s: ResidencyState| self.entries.values().filter(|e| e.state == s).count();
        KvResidencyStatus {
            stats,
            restoring: count(ResidencyState::Restoring),
            draining: count(ResidencyState::Draining),
            failed: count(ResidencyState::Failed),
            prefilling: count(ResidencyState::Prefilling),
            parking_jobs: self.parks.len() + self.shared.outbox.lock().len(),
            holder_blocks: self.shared.kv.read().holder_block_count(),
            restore_inflight_blocks: self.restore_inflight_blocks,
            landing_waiters: self.landings.values().map(|l| l.waiters as usize).sum(),
        }
    }

    // -----------------------------------------------------------------------
    // Park: submit what S1 forked
    // -----------------------------------------------------------------------

    /// Submits every queued park intent (called by the spine right after the
    /// scheduler step, with no lock held). A refused submit drops the source,
    /// which frees its holder (S4).
    pub fn drain_outbox(&mut self) {
        let intents = std::mem::take(&mut *self.shared.outbox.lock());
        if intents.is_empty() {
            return;
        }
        let base = match self.shared.kv.read().pool() {
            Some(p) => p.base_ptr(),
            None => return,
        };
        let block_bytes = self.shared.kv.read().pool().map_or(0, |p| p.block_bytes());
        for it in intents {
            let guard = holder_guard(&self.shared.kv, &self.shared.gens, it.holder, it.generation);
            // SAFETY: the blocks belong to the pool behind `base` and are
            // referenced by the holder table `it.holder` (fork_range in S1), so
            // they are neither freed nor reused until the guard runs its
            // release. Nobody writes them meanwhile: the finished sequence that
            // wrote them is freed, and a later sharer copies on write.
            let src =
                unsafe { PinnedSource::new(base, block_bytes, it.blocks, it.tail_valid, guard) };
            let bytes = it.header.payload_len;
            match self.shared.store.submit_store(it.header, src, it.inc) {
                Ok(job) if self.parks.contains_key(&job) => self.stats.park_dedup_total += 1,
                Ok(job) => {
                    self.parks.insert(job, bytes);
                    self.stats.park_jobs_total += 1;
                    self.stats.park_bytes_total += bytes;
                }
                Err(e) => {
                    if matches!(e, PlatformError::QueueFull) {
                        self.stats.store_queue_full_total += 1;
                    }
                    self.stats.park_failed_total += 1;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Restore at submit (S2)
    // -----------------------------------------------------------------------

    fn fallback(&mut self) {
        self.stats.restore_fallback_prefill_total += 1;
    }

    /// Looks the prompt of `seq` up in the store and, on a hit, starts its
    /// restore: S2 allocates the destination holders, the loads are submitted
    /// under one transaction and the request is held (`AwaitingRestore`).
    /// Every miss or refusal leaves the request a plain waiting request.
    pub fn try_restore(&mut self, sched: &mut AienScheduler, seq: SequenceId) {
        if !self.config.restore {
            return;
        }
        let Some(prompt) = sched.arena().get(seq).map(|r| r.prompt.clone()) else {
            return;
        };
        let ct = self.shared.chunk_tokens;
        let bs = self.shared.block_size;
        let mut cap = prompt.len().saturating_sub(1);
        if let Some(c) = self.config.context_cap {
            cap = cap.min(c);
        }
        if cap == 0 {
            return self.fallback();
        }
        let ns = self.shared.identity.namespace();
        let Ok(keys) = chunk_chain(&ns, ct as u32, &prompt[..cap]) else {
            return self.fallback();
        };
        // Retry window: a chain head whose restore failed twice within the
        // window is prefilled without a lookup.
        if let Some(head) = keys.first() {
            if self.recent_failures(head, (self.clock)()) >= 2 {
                self.stats.restore_retry_suppressed_total += 1;
                return self.fallback();
            }
        }
        // Walk to the first miss. The chain's last key is a partial tail only
        // when `cap` is not a chunk multiple, and it was keyed with exactly the
        // tokens up to the cap, so a hit there is a tail of exactly that length.
        let hits = self.shared.store.lookup(&keys);
        let n = hits.iter().take_while(|h| h.is_some()).count();
        if n == 0 {
            return self.fallback();
        }
        let spans: Vec<(usize, usize)> = (0..n)
            .map(|i| {
                let first = i * ct;
                (first, (cap - first).min(ct))
            })
            .collect();
        let needed: usize = spans.iter().map(|&(_, t)| t.div_ceil(bs)).sum();
        if self.restore_inflight_blocks + needed > self.config.restore_max_blocks {
            return self.fallback();
        }
        let pin = match self.shared.store.pin(&keys[..n]) {
            Ok(p) => p,
            Err(e) => {
                if matches!(e, PlatformError::QueueFull) {
                    self.stats.store_queue_full_total += 1;
                }
                return self.fallback();
            }
        };

        // S2 under one KV write guard; guards are built after it is released.
        let allocated: Result<Allocated, bool> = {
            let mut kv = self.shared.kv.write();
            if kv.free_block_count() < self.watermark_blocks + needed {
                Err(true)
            } else {
                let mut made: Vec<(u64, u32, Vec<usize>)> = Vec::with_capacity(n);
                let mut ok = true;
                for &(first, tokens) in &spans {
                    let Ok((holder, generation)) = alloc_holder(&self.shared.gens) else {
                        ok = false;
                        break;
                    };
                    match kv.allocate_sequence(holder, &prompt[first..first + tokens]) {
                        Ok(blocks) => made.push((holder, generation, blocks)),
                        Err(_) => {
                            self.shared.gens.lock().release(generation);
                            ok = false;
                            break;
                        }
                    }
                }
                let pool = kv.pool_mut().map(|p| (p.base_mut_ptr(), p.block_bytes()));
                match (ok, pool) {
                    (true, Some((base, block_bytes))) => Ok((made, base, block_bytes)),
                    _ => {
                        for (holder, generation, _) in made {
                            let _ = kv.free_sequence(holder);
                            self.shared.gens.lock().release(generation);
                        }
                        Err(false)
                    }
                }
            }
        };
        let (made, base, block_bytes) = match allocated {
            Ok(v) => v,
            Err(watermark) => {
                if watermark {
                    self.stats.restore_refused_watermark_total += 1;
                }
                self.shared.store.unpin(pin);
                return self.fallback();
            }
        };

        let inc = Incarnation::new(seq.to_u64(), 1);
        let fields = self.shared.identity.fields();
        let mut chunks: Vec<ChunkSlot> = Vec::with_capacity(n);
        let mut restored = 0usize;
        let mut blocks_used = 0usize;
        for (i, (holder, generation, blocks)) in made.into_iter().enumerate() {
            let (first, tokens) = spans[i];
            let n_blocks = blocks.len();
            let guard = holder_guard(&self.shared.kv, &self.shared.gens, holder, generation);
            // Once a chunk could not be submitted, later chunks are not either
            // (the restored prefix must be contiguous); their guards free them.
            if chunks.len() != i {
                drop(guard);
                continue;
            }
            let Ok(header) = fields.chunk_header(
                &ns,
                keys[i],
                i.checked_sub(1).map(|p| &keys[p]),
                first as u32,
                tokens as u32,
                0,
            ) else {
                drop(guard);
                continue;
            };
            // SAFETY: the blocks were just allocated to the holder table
            // `holder` (S2) and are invisible to the scheduler until S3
            // assembles them; the guard keeps them until the last landing copy
            // is dropped.
            let dst = unsafe { PinnedDest::new(base, block_bytes, blocks, guard) };
            match self
                .shared
                .store
                .submit_load_coalesced(keys[i], &header, move || dst, inc)
            {
                Ok(LoadTicket::Started(job)) => {
                    self.landings.insert(job, Landing { holder, waiters: 1 });
                    chunks.push(ChunkSlot {
                        job,
                        holder,
                        state: SlotState::Pending,
                    });
                }
                Ok(LoadTicket::Attached(job)) => match self.landings.get_mut(&job) {
                    Some(l) => {
                        l.waiters += 1;
                        self.stats.restore_coalesced_total += 1;
                        chunks.push(ChunkSlot {
                            job,
                            holder: l.holder,
                            state: SlotState::Pending,
                        });
                    }
                    None => {
                        // A load this runtime did not start: its landing holder
                        // is unknown here, so it cannot be assembled. Withdraw;
                        // its terminal arrives later and is counted stale.
                        let _ = self.shared.store.cancel_waiter(job, inc);
                        continue;
                    }
                },
                Err(e) => {
                    if matches!(e, PlatformError::QueueFull) {
                        self.stats.store_queue_full_total += 1;
                    }
                    continue;
                }
            }
            self.stats.restore_jobs_total += 1;
            self.stats.restore_bytes_total += header.payload_len;
            restored += tokens;
            blocks_used += n_blocks;
        }
        if chunks.is_empty() {
            self.shared.store.unpin(pin);
            return self.fallback();
        }
        let mut entry = Entry {
            seq,
            state: ResidencyState::Restoring,
            epoch: inc.epoch,
            txn: Some(RestoreTxn {
                inc,
                head: keys[0],
                restored,
                blocks: blocks_used,
                chunks,
                pin: Some(pin),
            }),
        };
        self.restore_inflight_blocks += blocks_used;
        if sched.hold_for_restore(seq).is_err() {
            // Not a plain waiting request after all: drain what was submitted.
            self.begin_drain(&mut entry);
        }
        self.entries.insert(seq.to_u64(), entry);
    }

    // -----------------------------------------------------------------------
    // Completions, S3, draining
    // -----------------------------------------------------------------------

    /// Moves a restoring entry to `Draining` and withdraws its pending loads.
    fn begin_drain(&mut self, entry: &mut Entry) {
        if entry.state == ResidencyState::Draining {
            return;
        }
        entry.transition(ResidencyState::Draining);
        if let Some(txn) = &entry.txn {
            for c in &txn.chunks {
                if matches!(c.state, SlotState::Pending) {
                    let _ = self.shared.store.cancel_waiter(c.job, txn.inc);
                }
            }
        }
    }

    fn landing_terminal(&mut self, job: JobId) {
        if let Some(l) = self.landings.get_mut(&job) {
            l.waiters = l.waiters.saturating_sub(1);
            if l.waiters == 0 {
                self.landings.remove(&job);
            }
        }
    }

    fn on_park_completion(&mut self, c: Completion) {
        let Some(t) = &c.terminal else {
            return;
        };
        self.parks.remove(&c.job);
        // ADR 2.4: published and served, durability unknown. Adapters report it
        // on a failed terminal (the directory sync failed after the rename);
        // counted whatever the terminal says.
        if c.publish == PublishOutcome::Indeterminate {
            self.stats.park_indeterminate_total += 1;
        }
        match t {
            Terminal::Done(_) => self.stats.park_published_total += 1,
            Terminal::Failed(_) => self.stats.park_failed_total += 1,
            Terminal::Cancelled => {
                self.stats.store_cancelled_total += 1;
                self.stats.park_failed_total += 1;
            }
        }
    }

    /// Routes one restore completion. Returns the holder copy that must be
    /// dropped by the caller (stale or failed), if any.
    fn on_restore_completion(&mut self, mut c: Completion) -> Option<LandingHolder> {
        let terminal = c.terminal.clone()?;
        self.landing_terminal(c.job);
        let key = c.incarnation.seq_id;
        let Some(mut entry) = self.entries.remove(&key) else {
            self.stats.store_stale_completions_total += 1;
            return c.holder.take();
        };
        let live = entry.txn.as_ref().is_some_and(|t| t.inc == c.incarnation)
            && matches!(
                entry.state,
                ResidencyState::Restoring | ResidencyState::Draining
            );
        let slot = live
            .then(|| {
                entry.txn.as_mut().and_then(|t| {
                    t.chunks
                        .iter_mut()
                        .find(|s| s.job == c.job && matches!(s.state, SlotState::Pending))
                })
            })
            .flatten();
        let Some(slot) = slot else {
            self.stats.store_stale_completions_total += 1;
            self.entries.insert(key, entry);
            return c.holder.take();
        };
        let mut drop_me = None;
        let failed = match terminal {
            Terminal::Done(_) => {
                slot.state = SlotState::Completed(c.holder.take());
                false
            }
            Terminal::Failed(_) => {
                slot.state = SlotState::Ended;
                drop_me = c.holder.take();
                true
            }
            Terminal::Cancelled => {
                slot.state = SlotState::Ended;
                self.stats.store_cancelled_total += 1;
                drop_me = c.holder.take();
                true
            }
        };
        if failed && entry.state == ResidencyState::Restoring {
            self.begin_drain(&mut entry);
        }
        self.entries.insert(key, entry);
        drop_me
    }

    /// Ends a transaction that will not be committed: releases its accounting
    /// and pin and returns its landing copies for the caller to drop.
    fn close_txn(&mut self, txn: &mut RestoreTxn) -> Vec<LandingHolder> {
        self.restore_inflight_blocks = self.restore_inflight_blocks.saturating_sub(txn.blocks);
        txn.blocks = 0;
        if let Some(pin) = txn.pin.take() {
            self.shared.store.unpin(pin);
        }
        txn.take_landings()
    }

    /// S3 for one fully completed transaction. On any refusal the request
    /// falls back to a plain prefill and nothing the scheduler can see changed.
    fn commit<B: AienInferenceBackend + ?Sized>(
        &mut self,
        sched: &mut AienScheduler,
        backend: &mut B,
        entry: &mut Entry,
        garbage: &mut Vec<LandingHolder>,
    ) {
        let Some(mut txn) = entry.txn.take() else {
            return;
        };
        let seq = entry.seq;
        let seq_u64 = seq.to_u64();
        let holders: Vec<u64> = txn.chunks.iter().map(|c| c.holder).collect();
        // The incarnation matched; the sequence must also still exist (ADR 5.3).
        let prompt = sched.arena().get(seq).map(|r| r.prompt.clone());
        let assembled = if prompt.is_none() {
            Err("sequence left the scheduler".to_string())
        } else {
            let mut kv = self.shared.kv.write();
            let mut res = Ok(());
            for &h in &holders {
                // The load verified every segment into the holder's blocks:
                // that is the holder's completion fence (DOCS SILENT, see C6
                // report). Idempotent for a holder another waiter completed.
                if let Err(e) = kv
                    .mark_prefill_pending(h)
                    .and_then(|_| kv.complete_prefill(h))
                {
                    res = Err(format!("{e}"));
                    break;
                }
            }
            res.and_then(|_| {
                kv.assemble_table(seq_u64, &holders, SuffixPolicy::AllowPartialTail)
                    .map(|_| ())
            })
        };
        garbage.extend(self.close_txn(&mut txn));
        let restored = txn.restored;
        let ok = assembled.is_ok()
            && prompt.as_ref().is_some_and(|p| restored < p.len())
            && match prompt
                .as_ref()
                .map(|p| backend.restore_sequence(seq_u64, &p[..restored]))
            {
                Some(Ok(())) => match sched.admit_restored_prefix(seq, restored) {
                    Ok(()) => true,
                    Err(_) => {
                        let _ = backend.release_sequence(seq_u64);
                        false
                    }
                },
                _ => false,
            };
        if ok {
            entry.transition(ResidencyState::Prefilling);
            self.stats.restore_committed_total += 1;
            self.stats.restore_tokens_total += restored as u64;
        } else {
            if assembled.is_ok() {
                let _ = self.shared.kv.write().free_sequence(seq_u64);
            }
            let _ = sched.release_restore(seq);
            entry.transition(ResidencyState::Failed);
            self.stats.restore_failed_total += 1;
            self.record_restore_failure(txn.head);
            self.fallback();
        }
    }

    /// The spine calls this at the start of every `step`: abandon restores
    /// whose request left the scheduler, route completions, commit or drain
    /// finished transactions, and retire settled entries. Never blocks.
    pub fn poll<B: AienInferenceBackend + ?Sized>(
        &mut self,
        sched: &mut AienScheduler,
        backend: &mut B,
    ) {
        let mut garbage: Vec<LandingHolder> = Vec::new();

        // 1. Requests that left the scheduler (cancelled, client gone): their
        //    waiters are withdrawn and the entry dropped, so every later
        //    completion for them is stale. Their holders stay with the worker
        //    until each job's terminal (ADR 5.5).
        let gone: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, e)| sched.arena().is_stale(e.seq))
            .map(|(&k, _)| k)
            .collect();
        for k in gone {
            if let Some(mut e) = self.entries.remove(&k) {
                if let Some(mut txn) = e.txn.take() {
                    for c in &txn.chunks {
                        if matches!(c.state, SlotState::Pending) {
                            let _ = self.shared.store.cancel_waiter(c.job, txn.inc);
                        }
                    }
                    garbage.extend(self.close_txn(&mut txn));
                    self.stats.restore_abandoned_total += 1;
                }
            }
        }

        // 2. Completions.
        loop {
            let batch = self.shared.store.poll(POLL_BATCH);
            if batch.is_empty() {
                break;
            }
            for c in batch {
                if self.parks.contains_key(&c.job) {
                    self.on_park_completion(c);
                } else if let Some(h) = self.on_restore_completion(c) {
                    garbage.push(h);
                }
            }
        }

        // 3. Commit or finish draining.
        let keys: Vec<u64> = self.entries.keys().copied().collect();
        for k in keys {
            let Some(mut e) = self.entries.remove(&k) else {
                continue;
            };
            match e.state {
                ResidencyState::Restoring if e.txn.as_ref().is_some_and(|t| t.pending() == 0) => {
                    self.commit(sched, backend, &mut e, &mut garbage);
                }
                ResidencyState::Draining if e.txn.as_ref().is_none_or(|t| t.pending() == 0) => {
                    if let Some(mut txn) = e.txn.take() {
                        garbage.extend(self.close_txn(&mut txn));
                        self.record_restore_failure(txn.head);
                    }
                    let _ = sched.release_restore(e.seq);
                    e.transition(ResidencyState::Failed);
                    self.stats.restore_failed_total += 1;
                    self.fallback();
                }
                _ => {}
            }
            // 4. Retire settled entries: a restored request once its suffix is
            //    prefilled, a failed one once its plain prefill started.
            let settled = match sched.arena().get(e.seq) {
                // A restoring or draining entry of a request that left is
                // closed by step 1 (next poll), never dropped here.
                None => matches!(e.state, ResidencyState::Prefilling | ResidencyState::Failed),
                Some(rec) => match e.state {
                    ResidencyState::Prefilling => rec.is_prefilled,
                    ResidencyState::Failed => !matches!(
                        rec.phase,
                        SequencePhase::Waiting | SequencePhase::AwaitingRestore
                    ),
                    _ => false,
                },
            };
            if !settled {
                self.entries.insert(k, e);
            }
        }

        // Landing copies are dropped here, with no lock held (S4 takes the KV lock).
        drop(garbage);
    }
}
