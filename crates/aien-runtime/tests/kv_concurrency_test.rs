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

//! C8 (sc#485 B1 to B5; ADR 0003 sections 5.2, 5.5, 7 and 8): concurrency
//! proof of persistent KV through the real runtime spine, the real scheduler
//! and the host CPU backend (`NativeTransformerBackend` over
//! `ReferenceCpuBackend` with the reference test weights), the real C5 storage
//! worker and the deterministic C2 fake store.
//!
//! Host only: no GPU, no CUDA, no files. Every K/V byte is computed by the CPU
//! model, so restored bytes and greedy continuations are compared with cold
//! runs of the same prompt.
//!
//! Stalls are gates, never wall-clock bounds: [`Gate`] sits between the
//! storage worker and the fake and can hold every load or store it is handed
//! (the fake's `Hold` fault) until the test releases or cancels it, and can
//! make stores fail as if the disk were full. Progress is counted in steps,
//! and a step counts only once the storage worker has run every job it can
//! ([`settle`]), so a busy machine cannot spend a step bound.
//! The only wall clock in this file is a hang guard on loops that wait for the
//! worker thread; it never decides a pass.
//!
//! Geometry: block size 4 tokens, chunk of 2 blocks (8 tokens). A 15-token
//! prompt with `max_tokens = 2` finishes with a computed prefix of 16 tokens
//! (the prompt and the first generated token; the newest token has no K/V),
//! which is exactly two full chunks.

use aien_abi_core::{ModelConfig, SamplingParams};
use aien_inference_abi::backend::TensorBackend;
use aien_inference_abi::{NativeTransformerBackend, ReferenceCpuBackend, TransformerWeights};
use aien_kv_cache::SharedKvManager;
use aien_kv_store::fake::{FakeConfig, FakeKvStore, Fault, JobScript};
use aien_kv_store::worker::{StorageWorker, WorkerConfig};
use aien_kv_store::{
    chunk_chain, BackendFormat, CancelOutcome, Capacity, ChunkHeader, ChunkKey, Completion,
    DrainReport, Incarnation, JobId, KvDtype, KvStore, NamespaceId, PinToken, PinnedDest,
    PinnedSource, RecoveryProgress,
};
use aien_platform::PlatformError;
use aien_runtime::compute_config::{HostEngineIdentity, ModelBinding, StoreInputs};
use aien_runtime::kv_residency::{
    layout_strides, KvIdentity, KvResidency, KvResidencyConfig, KvResidencyStatus, ResidencyState,
};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{
    ChannelCompletionSink, CompletionEvent, CompletionSinkId, SchedulerConfig, SequenceId,
    SequencePhase,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedReceiver;

const BLOCK: usize = 4;
const CHUNK_BLOCKS: u32 = 2;
const CHUNK: usize = BLOCK * CHUNK_BLOCKS as usize;

/// Hang guard for loops that wait on the storage worker thread. Never a pass
/// criterion: every assertion counts steps or reads state.
const HANG_GUARD: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// The gate: a KvStore between the storage worker and the fake
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Load,
    Store,
}

/// Delegates to the fake. While `hold_loads` / `hold_stores` is set, each job
/// of that kind is submitted with the fake's `Hold` fault and recorded, so the
/// test can release or cancel it. While `disk_full` is set, every store fails
/// at its write with `OutOfMemory` (the fake's `DiskFull`). Only the storage
/// worker thread submits to the inner store, so a script placed right before
/// the inner `submit_*` is the one that call consumes.
struct Gate {
    inner: Arc<FakeKvStore>,
    hold_loads: AtomicBool,
    hold_stores: AtomicBool,
    disk_full: AtomicBool,
    held: Mutex<Vec<(Kind, ChunkKey, JobId)>>,
    /// Every load and store that reached the store, in order.
    seen: Mutex<Vec<(Kind, ChunkKey)>>,
    /// Passes of the storage worker loop. The worker calls `poll` on the store
    /// exactly once per pass, after it has dispatched and driven, so every
    /// `poll` closes one pass.
    passes: AtomicU64,
    /// The latest pass that handed the store nothing (no submit, cancel or
    /// waiter) and reaped nothing from it: the worker had no work it could
    /// move without the spine or the test. 0 before the first such pass.
    quiet: AtomicU64,
    /// The store was touched since the last `poll`.
    touched: AtomicBool,
}

impl Gate {
    fn new(inner: Arc<FakeKvStore>) -> Self {
        Self {
            inner,
            hold_loads: AtomicBool::new(false),
            hold_stores: AtomicBool::new(false),
            disk_full: AtomicBool::new(false),
            held: Mutex::new(Vec::new()),
            seen: Mutex::new(Vec::new()),
            passes: AtomicU64::new(0),
            quiet: AtomicU64::new(0),
            touched: AtomicBool::new(false),
        }
    }

    fn script(&self, kind: Kind) -> bool {
        let hold = match kind {
            Kind::Load => self.hold_loads.load(Ordering::SeqCst),
            Kind::Store => self.hold_stores.load(Ordering::SeqCst),
        };
        let mut s = JobScript::default();
        if hold {
            s = s.with(Fault::Hold);
        }
        if kind == Kind::Store && self.disk_full.load(Ordering::SeqCst) {
            s = s.with(Fault::DiskFull);
        }
        if s != JobScript::default() {
            self.inner.script(s);
        }
        hold
    }

    fn after(&self, kind: Kind, key: ChunkKey, hold: bool, r: &Result<JobId, PlatformError>) {
        self.touched.store(true, Ordering::SeqCst);
        self.seen.lock().push((kind, key));
        if let (true, Ok(job)) = (hold, r) {
            self.held.lock().push((kind, key, *job));
        }
    }

    fn held_count(&self, kind: Kind) -> usize {
        self.held.lock().iter().filter(|h| h.0 == kind).count()
    }

    fn seen_count(&self, kind: Kind) -> usize {
        self.seen.lock().iter().filter(|h| h.0 == kind).count()
    }

    /// Lets every held job of `kind` start; returns how many.
    fn release(&self, kind: Kind) -> usize {
        let mut held = self.held.lock();
        let mut n = 0;
        held.retain(|&(k, _, job)| {
            if k == kind {
                assert!(self.inner.release(job), "held job {job:?} not held");
                n += 1;
                false
            } else {
                true
            }
        });
        n
    }

    /// Cancels every held job of `kind` at the store (a mid-transfer cancel,
    /// ADR 5.5); returns how many were cancelled.
    fn cancel(&self, kind: Kind) -> usize {
        let mut held = self.held.lock();
        let mut n = 0;
        held.retain(|&(k, _, job)| {
            if k == kind {
                assert_eq!(self.inner.cancel(job), CancelOutcome::Cancelled);
                n += 1;
                false
            } else {
                true
            }
        });
        n
    }
}

impl KvStore for Gate {
    fn namespace(&self) -> NamespaceId {
        self.inner.namespace()
    }
    fn capacity(&self) -> Capacity {
        self.inner.capacity()
    }
    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>> {
        self.inner.lookup(keys)
    }
    fn submit_store(
        &self,
        header: ChunkHeader,
        src: PinnedSource,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let key = header.chunk_key;
        let hold = self.script(Kind::Store);
        let r = self.inner.submit_store(header, src, inc);
        self.after(Kind::Store, key, hold, &r);
        r
    }
    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let hold = self.script(Kind::Load);
        let r = self.inner.submit_load(key, expect, dst, inc);
        self.after(Kind::Load, key, hold, &r);
        r
    }
    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError> {
        self.touched.store(true, Ordering::SeqCst);
        self.inner.submit_flush(inc)
    }
    fn attach_waiter(&self, job: JobId, inc: Incarnation) -> Result<(), PlatformError> {
        self.touched.store(true, Ordering::SeqCst);
        self.inner.attach_waiter(job, inc)
    }
    fn poll(&self, max: usize) -> Vec<Completion> {
        let c = self.inner.poll(max);
        let pass = self.passes.fetch_add(1, Ordering::SeqCst) + 1;
        if !self.touched.swap(false, Ordering::SeqCst) && c.is_empty() {
            self.quiet.store(pass, Ordering::SeqCst);
        }
        c
    }
    fn cancel(&self, job: JobId) -> CancelOutcome {
        self.touched.store(true, Ordering::SeqCst);
        self.inner.cancel(job)
    }
    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError> {
        self.inner.pin(keys)
    }
    fn unpin(&self, token: PinToken) {
        self.inner.unpin(token)
    }
    fn delete(&self, keys: &[ChunkKey]) -> u64 {
        self.inner.delete(keys)
    }
    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress {
        self.inner.recover_step(slice_ms)
    }
    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport {
        self.inner.shutdown(intake_deadline_ms)
    }
}

// ---------------------------------------------------------------------------
// Rig: the real runtime over the CPU backend
// ---------------------------------------------------------------------------

/// The CPU reference model of the C7 greedy test (shared_kv_runtime_test.rs).
fn model() -> ModelConfig {
    ModelConfig {
        model_id: "c8-concurrency-reference".to_string(),
        max_sequence_length: 256,
        block_size: BLOCK,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        num_kv_heads: 2,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 97,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        // Outside the 97-token vocabulary: no sampled token ends a turn early
        // (an empty list means the legacy stop set 0, 1, 2).
        eos_token_ids: vec![1000],
        qk_norm: false,
    }
}

#[derive(Clone)]
struct Opts {
    total_blocks: usize,
    watermark_blocks: usize,
    /// `None`: persistence off (the cold baselines).
    persist: bool,
    restore_max_blocks: usize,
    max_restore_ops: u32,
    max_park_ops: u32,
    max_batch_size: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            total_blocks: 64,
            watermark_blocks: 0,
            persist: true,
            restore_max_blocks: 64,
            max_restore_ops: 48,
            max_park_ops: 16,
            max_batch_size: 16,
        }
    }
}

struct Store {
    fake: Arc<FakeKvStore>,
    gate: Arc<Gate>,
    worker: Arc<StorageWorker>,
    ns: NamespaceId,
}

struct Rig {
    spine: AienRuntimeSpine,
    backend: NativeTransformerBackend,
    kv: SharedKvManager,
    store: Option<Store>,
    sink: CompletionSinkId,
    rx: UnboundedReceiver<CompletionEvent>,
    /// Tokens streamed to the completion sink, per sequence id.
    tokens: HashMap<u64, Vec<u32>>,
    steps: u64,
}

fn rig_with(opts: Opts) -> Rig {
    let config = model();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        SchedulerConfig {
            max_batch_size: opts.max_batch_size,
            max_batch_tokens: 4096,
            max_prefill_tokens: 4096,
            prefill_chunk_size: 4096,
            chunk_prefill: true,
            watermark_blocks: opts.watermark_blocks,
        },
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: opts.total_blocks,
        },
    )
    .expect("runtime");
    let kv = spine.kv_manager.clone();
    let store = opts.persist.then(|| install_store(&mut spine, &kv, &opts));
    let (sink, rx) = ChannelCompletionSink::channel();
    let sink = spine.register_completion_sink(Arc::new(sink));
    Rig {
        spine,
        backend,
        kv,
        store,
        sink,
        rx,
        tokens: HashMap::new(),
        steps: 0,
    }
}

/// Binds the namespace exactly as the daemon wiring does
/// (`shared_kv::enable_persistent_kv`), but over the gated fake store.
fn install_store(spine: &mut AienRuntimeSpine, kv: &SharedKvManager, opts: &Opts) -> Store {
    let config = model();
    let layout = layout_strides(kv.read().pool().expect("pool").layout());
    let arith = ReferenceCpuBackend::new().name();
    let binding = ModelBinding::from_model(&config, arith).expect("binding");
    let engine = HostEngineIdentity::reference(arith).expect("engine identity");
    let fields = binding
        .namespace_fields(
            &StoreInputs {
                model_sha256: [8u8; 32],
                model_digest_kind: "sha256",
                dtype: KvDtype::Bf16,
                layout,
                backend_format: BackendFormat::HostPagedPoolV1,
                chunk_tokens: CHUNK as u32,
                sharing_namespace: "default",
            },
            &engine.identity(),
        )
        .expect("fields");
    let identity = KvIdentity::new(&fields).expect("identity");
    let ns = identity.namespace();
    let mut fc = FakeConfig::new(ns);
    fc.max_restore_ops = opts.max_restore_ops;
    fc.max_park_ops = opts.max_park_ops;
    let fake = Arc::new(FakeKvStore::new(fc));
    let gate = Arc::new(Gate::new(fake.clone()));
    let drive = {
        let fake = fake.clone();
        move || fake.advance(1000)
    };
    let worker = Arc::new(
        StorageWorker::spawn(
            gate.clone(),
            WorkerConfig {
                max_restore_ops: opts.max_restore_ops,
                max_park_ops: opts.max_park_ops,
                idle_poll: Duration::from_micros(200),
                ..WorkerConfig::default()
            },
            drive,
        )
        .expect("spawn worker"),
    );
    let residency = KvResidency::new(
        KvResidencyConfig {
            chunk_blocks: CHUNK_BLOCKS,
            context_cap: None,
            restore_max_blocks: opts.restore_max_blocks,
            park: true,
            restore: true,
            restore_retry_window_ms: 60_000,
            drain_ms: 2000,
        },
        identity,
        worker.clone(),
        kv.clone(),
    )
    .expect("residency");
    spine.enable_kv_residency(residency);
    Store {
        fake,
        gate,
        worker,
        ns,
    }
}

fn rig() -> Rig {
    rig_with(Opts::default())
}

fn st(r: &Rig) -> &Store {
    r.store.as_ref().expect("persistence on")
}

fn status(r: &Rig) -> KvResidencyStatus {
    r.spine.kv_residency_status().expect("residency installed")
}

fn state_of(r: &Rig, seq: u64) -> ResidencyState {
    r.spine.kv_residency().expect("residency").residency_of(seq)
}

fn phase_of(r: &Rig, seq: u64) -> Option<SequencePhase> {
    let id = SequenceId::from_u64(seq).ok()?;
    r.spine.scheduler.arena().get(id).map(|rec| rec.phase)
}

fn finished(r: &Rig, seq: u64) -> bool {
    phase_of(r, seq).is_none()
}

fn emitted(r: &Rig, seq: u64) -> usize {
    r.tokens.get(&seq).map_or(0, Vec::len)
}

fn toks(r: &Rig, seq: u64) -> Vec<u32> {
    r.tokens.get(&seq).cloned().unwrap_or_default()
}

fn free_blocks(r: &Rig) -> usize {
    r.kv.read().free_block_count()
}

fn holder_blocks(r: &Rig) -> usize {
    r.kv.read().holder_block_count()
}

fn table(r: &Rig, seq: u64) -> Vec<usize> {
    r.kv.read()
        .get_block_table(seq)
        .map(|t| t.block_ids.clone())
        .unwrap_or_default()
}

/// Reference count of every block with at least one reference.
fn refcounts(r: &Rig) -> BTreeMap<usize, usize> {
    r.kv.read()
        .debug_active_blocks()
        .into_iter()
        .map(|(b, n, _)| (b, n))
        .collect()
}

/// With no storage holder alive, every block's reference count equals the
/// number of `live` tables that list it, and the free list holds the rest.
fn assert_refcounts_match_tables(r: &Rig, live: &[u64], what: &str) {
    assert_eq!(holder_blocks(r), 0, "{what}: holders alive");
    let mut want: BTreeMap<usize, usize> = BTreeMap::new();
    for &s in live {
        for b in table(r, s) {
            *want.entry(b).or_default() += 1;
        }
    }
    assert_eq!(refcounts(r), want, "{what}: refcounts vs live tables");
    let total = r.kv.read().total_block_count();
    assert_eq!(free_blocks(r), total - want.len(), "{what}: free list");
}

fn read_block(r: &Rig, b: usize) -> Vec<u8> {
    let kv = r.kv.read();
    let pool = kv.pool().expect("pool");
    let len = pool.block_bytes();
    // SAFETY: test-only read of one in-range pool block while the KV read lock
    // is held; the spine does not run on another thread.
    unsafe { std::slice::from_raw_parts(pool.base_ptr().add(b * len), len) }.to_vec()
}

fn read_blocks(r: &Rig, blocks: &[usize]) -> Vec<u8> {
    blocks.iter().flat_map(|&b| read_block(r, b)).collect()
}

/// Byte equality that reports the length and the first differing offset
/// instead of dumping kilobytes.
fn same_bytes(got: &[u8], want: &[u8], what: &str) {
    let first = got.iter().zip(want).position(|(a, b)| a != b);
    assert!(
        got.len() == want.len() && first.is_none(),
        "{what}: {} vs {} bytes, first difference at {first:?}",
        got.len(),
        want.len()
    );
}

fn keys(r: &Rig, tokens: &[u32]) -> Vec<ChunkKey> {
    chunk_chain(&st(r).ns, CHUNK as u32, tokens).expect("chain")
}

fn payload(r: &Rig, key: &ChunkKey) -> Option<Vec<u8>> {
    st(r).fake.payload_of(key)
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens,
        stop_token_ids: vec![],
    }
}

fn submit(r: &mut Rig, prompt: &[u32], max_tokens: usize) -> u64 {
    r.spine
        .submit_work(
            Arc::from(prompt.to_vec()),
            greedy(max_tokens),
            1,
            Some(r.sink),
        )
        .expect("submit")
}

fn collect(r: &mut Rig) {
    while let Ok(ev) = r.rx.try_recv() {
        if let CompletionEvent::Token { seq_id, token } = ev {
            r.tokens.entry(seq_id.to_u64()).or_default().push(token);
        }
    }
}

/// One spine step, then [`settle`]: a step is counted only once the storage
/// worker has had its turn, so step bounds measure the spine's progress and
/// not how much CPU the OS gave the worker thread.
async fn step(r: &mut Rig) {
    r.spine.step(&mut r.backend).await.expect("step");
    r.steps += 1;
    settle(r);
    collect(r);
}

/// Waits until a whole pass of the storage worker loop that began after this
/// call handed the store nothing and reaped nothing. The pass sees every job
/// the spine has submitted so far; the fake is driven 1 ms per pass and runs
/// a job in 200 us, so a quiet pass means every job the worker was handed is
/// finished and its completion is waiting for the spine, or is held by the
/// gate, or waits behind a dispatch depth full of held jobs. No-op with
/// persistence off.
fn settle(r: &Rig) {
    let Some(s) = r.store.as_ref() else {
        return;
    };
    // Pass `seen + 1` may have begun before this call; pass `seen + 2` began
    // after its poll, so it is the first that surely saw every submission.
    let want = s.gate.passes.load(Ordering::SeqCst) + 2;
    let t0 = Instant::now();
    while s.gate.quiet.load(Ordering::SeqCst) < want {
        assert!(
            t0.elapsed() < HANG_GUARD,
            "hang guard: storage worker never settled"
        );
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// Steps until `cond` holds, at most `max_steps` steps (the progress bound).
async fn step_until(r: &mut Rig, what: &str, max_steps: u64, cond: impl Fn(&Rig) -> bool) {
    let t0 = Instant::now();
    let start = r.steps;
    loop {
        if cond(r) {
            return;
        }
        assert!(
            r.steps - start < max_steps,
            "no progress within {max_steps} steps: {what}"
        );
        assert!(t0.elapsed() < HANG_GUARD, "hang guard: {what}");
        step(r).await;
    }
}

/// Waits (without stepping) until the worker handed the store `n` held jobs
/// of `kind`.
fn wait_held(r: &Rig, kind: Kind, n: usize) {
    let t0 = Instant::now();
    while st(r).gate.held_count(kind) < n {
        assert!(t0.elapsed() < HANG_GUARD, "hang guard: {n} held {kind:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Runs `prompt` to its end and waits until every park it caused is terminal.
async fn run_and_park(r: &mut Rig, prompt: &[u32], max_tokens: usize) -> u64 {
    let seq = submit(r, prompt, max_tokens);
    step_until(r, "turn finished and parks terminal", 500, |r| {
        finished(r, seq) && status(r).parking_jobs == 0
    })
    .await;
    seq
}

/// The greedy tokens of `prompt` on a fresh runtime with persistence off.
async fn cold(prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut r = rig_with(Opts {
        persist: false,
        ..Opts::default()
    });
    let seq = submit(&mut r, prompt, max_tokens);
    step_until(&mut r, "cold run", 500, |r| finished(r, seq)).await;
    let t = toks(&r, seq);
    assert_eq!(t.len(), max_tokens, "cold run emitted every token");
    t
}

fn prompt_a() -> Vec<u32> {
    vec![5, 17, 42, 3, 88, 61, 9, 23, 70, 11, 36, 2, 44, 19, 80]
}

fn prompt_b() -> Vec<u32> {
    vec![31, 7, 64, 12, 90, 27, 55, 4, 73, 18, 39, 66, 21, 85, 50]
}

/// Parks `prompt` (15 tokens, `max_tokens = 2`) and returns its 16-token
/// computed prefix: the prompt and its first generated token.
async fn park_prefix(r: &mut Rig, prompt: &[u32]) -> Vec<u32> {
    assert_eq!(prompt.len(), 15);
    let seq = run_and_park(r, prompt, 2).await;
    let mut c = prompt.to_vec();
    c.push(toks(r, seq)[0]);
    for k in keys(r, &c) {
        assert!(payload(r, &k).is_some(), "prefix chunk published");
    }
    c
}

fn with_suffix(prefix: &[u32], suffix: &[u32]) -> Vec<u32> {
    let mut v = prefix.to_vec();
    v.extend_from_slice(suffix);
    v
}

fn unrelated() -> Vec<u32> {
    vec![60, 1, 47, 33, 8, 92, 14, 26, 58, 77]
}

// ---------------------------------------------------------------------------
// B1: slow-I/O progress
// ---------------------------------------------------------------------------

/// #485 B1 (ADR 6.3 step 2, 8 rule 1): while a restore is stalled at the store
/// (its loads held by the gate), a resident sequence emits one token on every
/// step, and the restoring request is never scheduled. After the gate opens,
/// both outputs equal their cold greedy baselines.
#[tokio::test]
async fn resident_sequence_emits_tokens_before_unrelated_restore_completes() {
    let mut r = rig();
    let prefix = park_prefix(&mut r, &prompt_a()).await;
    let q_prompt = with_suffix(&prefix, &[13, 29, 51]);

    let resident = submit(&mut r, &unrelated(), 16);
    step_until(&mut r, "resident decoding", 50, |r| {
        emitted(r, resident) >= 1
    })
    .await;

    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let q = submit(&mut r, &q_prompt, 4);
    assert_eq!(state_of(&r, q), ResidencyState::Restoring);
    wait_held(&r, Kind::Load, 2);

    for i in 0..8 {
        let before = emitted(&r, resident);
        step(&mut r).await;
        assert_eq!(
            emitted(&r, resident),
            before + 1,
            "step {i}: resident emitted while the restore is stalled"
        );
        assert_eq!(phase_of(&r, q), Some(SequencePhase::AwaitingRestore));
        assert_eq!(emitted(&r, q), 0, "restoring request scheduled");
        assert_eq!(st(&r).gate.held_count(Kind::Load), 2, "loads still held");
    }
    assert!(
        emitted(&r, resident) >= 9,
        "resident made progress before the restore completed"
    );

    assert_eq!(st(&r).gate.release(Kind::Load), 2);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    step_until(&mut r, "both finished, parks terminal", 500, |r| {
        finished(r, q) && finished(r, resident) && status(r).parking_jobs == 0
    })
    .await;

    assert_eq!(toks(&r, q), cold(&q_prompt, 4).await, "restored greedy");
    assert_eq!(
        toks(&r, resident),
        cold(&unrelated(), 16).await,
        "resident greedy"
    );
    let s = status(&r).stats;
    assert_eq!(s.restore_committed_total, 1);
    assert_eq!(s.restore_tokens_total, 16);
    assert_refcounts_match_tables(&r, &[], "end");
}

// ---------------------------------------------------------------------------
// B2: concurrent park and restore of a shared prefix; fork and cancel
// ---------------------------------------------------------------------------

/// #485 B2 (ADR 5.2, 7.3): a park of a sequence that extends a stored prefix
/// and two restores of that prefix are all in flight at once. The stored
/// prefix bytes are never rewritten (same-key publication dedup), each
/// restored table holds exactly the parked bytes, the coalesced restores share
/// the prefix blocks by reference count, and every count returns to the live
/// tables once the holders are gone.
#[tokio::test]
async fn concurrent_park_and_restore_of_shared_prefix_preserve_bytes_and_refcounts() {
    let mut r = rig();
    let prefix = park_prefix(&mut r, &prompt_a()).await;
    let pk = keys(&r, &prefix);
    let p0 = payload(&r, &pk[0]).expect("chunk 0");
    let p1 = payload(&r, &pk[1]).expect("chunk 1");

    // S extends the prefix: its own restore runs ungated, then its finish
    // parks the new chunks; those stores are held.
    let s_prompt = with_suffix(&prefix, &[3, 14, 15, 92, 65, 35, 89, 79]);
    st(&r).gate.hold_stores.store(true, Ordering::SeqCst);
    let s = submit(&mut r, &s_prompt, 2);
    step_until(&mut r, "S finished", 200, |r| finished(r, s)).await;
    let s_stats = status(&r).stats;
    assert_eq!(s_stats.restore_tokens_total, 16, "S restored the prefix");
    assert_eq!(s_stats.park_dedup_total, 2, "prefix chunks not re-stored");
    // S's computed prefix: 24 prompt tokens + 1 = 25: chunk 2 full, 1-token tail.
    wait_held(&r, Kind::Store, 2);
    assert_eq!(status(&r).parking_jobs, 2);

    // Two restores of the prefix while S's park is in flight; the second
    // attaches to the first's loads.
    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let q1_prompt = with_suffix(&prefix, &[10, 20, 30]);
    let q2_prompt = with_suffix(&prefix, &[40, 50, 60, 70, 80]);
    let q1 = submit(&mut r, &q1_prompt, 6);
    let q2 = submit(&mut r, &q2_prompt, 6);
    wait_held(&r, Kind::Load, 2);
    assert_eq!(status(&r).stats.restore_coalesced_total, 2);
    for _ in 0..3 {
        step(&mut r).await;
        assert_eq!(phase_of(&r, q1), Some(SequencePhase::AwaitingRestore));
        assert_eq!(phase_of(&r, q2), Some(SequencePhase::AwaitingRestore));
    }
    // Park holders (chunk 2: 2 blocks, tail: 1) and one set of landing
    // holders (4 blocks): nothing else is referenced.
    assert_eq!(holder_blocks(&r), 7);
    assert!(refcounts(&r).values().all(|&n| n == 1));
    assert_eq!(free_blocks(&r), r.kv.read().total_block_count() - 7);

    // Publish S's chunks first, then let the loads land.
    assert_eq!(st(&r).gate.release(Kind::Store), 2);
    st(&r).gate.hold_stores.store(false, Ordering::SeqCst);
    step_until(&mut r, "S parks terminal", 200, |r| {
        status(r).parking_jobs == 0
    })
    .await;
    assert_eq!(status(&r).stats.park_published_total, 2 + 2);
    assert_eq!(st(&r).gate.release(Kind::Load), 2);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    step_until(&mut r, "both restores committed", 200, |r| {
        status(r).stats.restore_committed_total == 3
    })
    .await;

    let t1 = table(&r, q1);
    let t2 = table(&r, q2);
    assert_eq!(t1[..4], t2[..4], "coalesced prefix blocks are shared");
    let mut want = p0.clone();
    want.extend_from_slice(&p1);
    same_bytes(
        &read_blocks(&r, &t1[..4]),
        &want,
        "q1 holds the parked bytes",
    );
    let rc = refcounts(&r);
    for b in &t1[..4] {
        assert_eq!(rc[b], 2, "prefix block {b} shared by exactly q1 and q2");
    }
    for b in t1[4..].iter().chain(&t2[4..]) {
        assert_eq!(rc[b], 1, "suffix block {b} owned by one request");
    }
    assert_refcounts_match_tables(&r, &[q1, q2], "after commit");

    step_until(&mut r, "q1 q2 finished, parks terminal", 500, |r| {
        finished(r, q1) && finished(r, q2) && status(r).parking_jobs == 0
    })
    .await;
    assert_eq!(toks(&r, q1), cold(&q1_prompt, 6).await, "q1 greedy");
    assert_eq!(toks(&r, q2), cold(&q2_prompt, 6).await, "q2 greedy");
    same_bytes(
        &payload(&r, &pk[0]).expect("published"),
        &p0,
        "chunk 0 never rewritten",
    );
    same_bytes(
        &payload(&r, &pk[1]).expect("published"),
        &p1,
        "chunk 1 never rewritten",
    );

    // S's own chunk 2 (published while the loads were in flight) restores to
    // the same greedy continuation as a cold run.
    let t_prompt = with_suffix(&s_prompt, &[11, 22, 33]);
    let t = submit(&mut r, &t_prompt, 3);
    step_until(&mut r, "t finished", 200, |r| finished(r, t)).await;
    assert_eq!(status(&r).stats.restore_tokens_total, 16 * 3 + 24);
    assert_eq!(toks(&r, t), cold(&t_prompt, 3).await, "t greedy");
    step_until(&mut r, "parks terminal", 200, |r| {
        status(r).parking_jobs == 0
    })
    .await;
    assert_refcounts_match_tables(&r, &[], "end");
}

/// #485 B2 (ADR 5.2, 5.5): a child is forked from a running parent; the
/// parent finishes and its park forks storage holders over blocks the child
/// shares; the park is cancelled at the store while the child decodes. The
/// child keeps its blocks and their bytes, decodes every step, and its output
/// equals the same fork run with persistence off; nothing is published.
#[tokio::test]
async fn fork_then_cancel_parent_during_park_keeps_child_valid() {
    async fn fork_run(r: &mut Rig) -> (u64, u64) {
        let parent = submit(r, &prompt_b(), 3);
        step_until(r, "parent prefilled", 50, |r| emitted(r, parent) >= 1).await;
        r.spine
            .fork_subagent(parent, 0, Some(r.sink))
            .expect("fork");
        let child = r
            .spine
            .scheduler
            .arena()
            .active_ids()
            .into_iter()
            .find(|&id| {
                r.spine
                    .scheduler
                    .arena()
                    .get(id)
                    .is_some_and(|rec| rec.branch_parent.map(|p| p.to_u64()) == Some(parent))
            })
            .expect("child")
            .to_u64();
        r.spine
            .scheduler
            .arena_mut()
            .get_mut(SequenceId::from_u64(child).unwrap())
            .expect("child record")
            .sampling_params
            .max_tokens = 10;
        (parent, child)
    }

    // Baseline: the same fork with persistence off.
    let mut base = rig_with(Opts {
        persist: false,
        ..Opts::default()
    });
    let (bp, bc) = fork_run(&mut base).await;
    step_until(&mut base, "baseline finished", 200, |r| {
        finished(r, bp) && finished(r, bc)
    })
    .await;
    let want_child = toks(&base, bc);
    assert!(want_child.len() >= 8, "baseline child decoded");

    let mut r = rig();
    st(&r).gate.hold_stores.store(true, Ordering::SeqCst);
    let (parent, child) = fork_run(&mut r).await;
    step_until(&mut r, "parent finished", 50, |r| finished(r, parent)).await;
    assert!(!finished(&r, child));
    // Parent computed prefix: 15 + 2 = 17: two full chunks and a 1-token tail.
    wait_held(&r, Kind::Store, 3);
    assert_eq!(status(&r).parking_jobs, 3);

    // The 15-token prompt fills blocks 0 to 2 and 3 tokens of block 3. The
    // first decode after the fork wrote into the shared partial block 3, which
    // copied on write (ADR 5.2), so the child shares the full blocks 0 to 2
    // with the parent's park holders and owns its block 3 alone.
    let child_table = table(&r, child);
    let shared = &child_table[..3];
    let rc = refcounts(&r);
    for b in shared {
        assert_eq!(rc[b], 2, "block {b}: child and one park holder");
    }
    assert_eq!(rc[&child_table[3]], 1, "child's own copy of block 3");
    let before = read_blocks(&r, shared);

    // The child decodes while the park is held.
    for _ in 0..2 {
        let n = emitted(&r, child);
        step(&mut r).await;
        assert_eq!(emitted(&r, child), n + 1, "child progress during park");
    }

    // Cancel the parent's park mid-transfer.
    assert_eq!(st(&r).gate.cancel(Kind::Store), 3);
    st(&r).gate.hold_stores.store(false, Ordering::SeqCst);
    step_until(&mut r, "park cancelled", 50, |r| {
        status(r).parking_jobs == 0
    })
    .await;
    let s = status(&r).stats;
    assert_eq!(s.park_failed_total, 3, "cancelled parks count as failed");
    assert_eq!(s.park_published_total, 0);
    let ck = keys(&r, &prompt_b());
    assert!(payload(&r, &ck[0]).is_none(), "nothing published");
    assert!(!finished(&r, child), "child still running");
    assert_eq!(table(&r, child)[..3], *shared, "child kept its blocks");
    same_bytes(&read_blocks(&r, shared), &before, "child bytes unchanged");
    assert_refcounts_match_tables(&r, &[child], "after cancel");

    step_until(&mut r, "child finished, parks terminal", 200, |r| {
        finished(r, child) && status(r).parking_jobs == 0
    })
    .await;
    assert_eq!(toks(&r, child), want_child, "child greedy unaffected");
    assert_refcounts_match_tables(&r, &[], "end");
}

// ---------------------------------------------------------------------------
// B3: duplicate loads; finding 6: coalesced waiters
// ---------------------------------------------------------------------------

/// #485 B3 (ADR 7.3): two identical requests restoring the same stored prefix
/// at the same time cause one load per chunk, not two; both get the restored
/// prefix and the cold greedy continuation.
#[tokio::test]
async fn duplicate_restore_requests_share_one_load() {
    let mut r = rig();
    let prefix = park_prefix(&mut r, &prompt_a()).await;
    let q_prompt = with_suffix(&prefix, &[13, 29, 51]);
    let reads_before = st(&r).fake.stats().reads;
    let loads_before = st(&r).gate.seen_count(Kind::Load);

    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let a = submit(&mut r, &q_prompt, 4);
    let b = submit(&mut r, &q_prompt, 4);
    assert_eq!(state_of(&r, a), ResidencyState::Restoring);
    assert_eq!(state_of(&r, b), ResidencyState::Restoring);
    wait_held(&r, Kind::Load, 2);
    let s = status(&r).stats;
    assert_eq!(s.restore_coalesced_total, 2, "b attached to a's loads");
    assert_eq!(st(&r).worker.stats().coalesced_total, 2);

    st(&r).gate.release(Kind::Load);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    step_until(&mut r, "both finished, parks terminal", 300, |r| {
        finished(r, a) && finished(r, b) && status(r).parking_jobs == 0
    })
    .await;

    assert_eq!(
        st(&r).gate.seen_count(Kind::Load) - loads_before,
        2,
        "one load per chunk"
    );
    assert_eq!(
        st(&r).fake.stats().reads - reads_before,
        2,
        "one read per chunk"
    );
    let want = cold(&q_prompt, 4).await;
    assert_eq!(toks(&r, a), want);
    assert_eq!(toks(&r, b), want);
    let s = status(&r).stats;
    assert_eq!(s.restore_committed_total, 2);
    assert_eq!(s.restore_tokens_total, 32);
    assert_refcounts_match_tables(&r, &[], "end");
}

/// Late attach (C8 surviving mutant): the second request arrives only after
/// the first request's loads were dispatched to the store and are held there.
/// It attaches to the in-flight loads (never a third read), both requests get
/// the cold greedy output, and the attached-waiter count on each landing is
/// released with the terminals: no landing is left behind.
#[tokio::test]
async fn late_attach_after_dispatch_shares_the_load_and_releases_its_landing() {
    let mut r = rig();
    let prefix = park_prefix(&mut r, &prompt_a()).await;
    let q_prompt = with_suffix(&prefix, &[13, 29, 51]);
    let reads_before = st(&r).fake.stats().reads;
    let loads_before = st(&r).gate.seen_count(Kind::Load);
    // The park run itself was a plain prefill (counted as a fallback).
    let fallback_before = status(&r).stats.restore_fallback_prefill_total;

    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let a = submit(&mut r, &q_prompt, 4);
    assert_eq!(state_of(&r, a), ResidencyState::Restoring);
    // Both loads are at the store (held) before b exists.
    wait_held(&r, Kind::Load, 2);
    assert_eq!(status(&r).stats.restore_coalesced_total, 0);
    assert_eq!(
        status(&r).landing_waiters,
        2,
        "one waiter (the opener) per in-flight load"
    );

    let b = submit(&mut r, &q_prompt, 4);
    assert_eq!(state_of(&r, b), ResidencyState::Restoring);
    let s = status(&r).stats;
    assert_eq!(s.restore_coalesced_total, 2, "b attached late to a's loads");
    assert_eq!(st(&r).worker.stats().coalesced_total, 2);
    assert_eq!(
        st(&r).gate.seen_count(Kind::Load) - loads_before,
        2,
        "no new load for b"
    );
    assert_eq!(
        status(&r).landing_waiters,
        4,
        "the late attach adds one waiter to each of the two loads"
    );

    st(&r).gate.release(Kind::Load);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    step_until(&mut r, "both finished, parks terminal", 300, |r| {
        finished(r, a) && finished(r, b) && status(r).parking_jobs == 0
    })
    .await;

    assert_eq!(
        st(&r).fake.stats().reads - reads_before,
        2,
        "one read per chunk"
    );
    let want = cold(&q_prompt, 4).await;
    assert_eq!(toks(&r, a), want);
    assert_eq!(toks(&r, b), want);
    let s = status(&r).stats;
    assert_eq!(s.restore_committed_total, 2);
    assert_eq!(
        s.restore_fallback_prefill_total, fallback_before,
        "neither fell back"
    );
    assert_eq!(
        status(&r).landing_waiters,
        0,
        "every landing released once its opener and its late waiter saw a terminal"
    );
    assert_eq!(st(&r).worker.stats().stale_completions_total, 0);
    assert_refcounts_match_tables(&r, &[], "end");
}

/// Review finding 6 (ADR 7.3): coalescing happens at the landing buffer, never
/// at the request table. Two requests with the same stored prefix and
/// different suffixes share the prefix blocks by reference count, each owns
/// its suffix blocks, and finishing one leaves the other's table, bytes and
/// greedy output intact.
#[tokio::test]
async fn coalesced_waiters_with_different_suffixes_get_independent_tables() {
    let mut r = rig();
    let prefix = park_prefix(&mut r, &prompt_a()).await;
    let pk = keys(&r, &prefix);
    let mut parked = payload(&r, &pk[0]).expect("chunk 0");
    parked.extend(payload(&r, &pk[1]).expect("chunk 1"));
    let short_prompt = with_suffix(&prefix, &[10, 20, 30]);
    let long_prompt = with_suffix(&prefix, &[40, 50, 60, 70, 80, 90, 7]);

    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let short = submit(&mut r, &short_prompt, 2);
    let long = submit(&mut r, &long_prompt, 8);
    wait_held(&r, Kind::Load, 2);
    assert_eq!(status(&r).stats.restore_coalesced_total, 2);
    st(&r).gate.release(Kind::Load);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    step_until(&mut r, "both committed", 200, |r| {
        status(r).stats.restore_committed_total == 2
    })
    .await;

    let ts = table(&r, short);
    let tl = table(&r, long);
    assert_eq!(ts[..4], tl[..4], "prefix blocks shared");
    same_bytes(&read_blocks(&r, &ts[..4]), &parked, "parked bytes restored");
    for b in &ts[4..] {
        assert!(!tl.contains(b), "suffix block {b} shared between requests");
    }
    let rc = refcounts(&r);
    for b in &ts[..4] {
        assert_eq!(rc[b], 2, "prefix block {b}: one reference per request");
    }
    assert_refcounts_match_tables(&r, &[short, long], "after commit");

    step_until(&mut r, "short finished", 100, |r| finished(r, short)).await;
    assert!(!finished(&r, long), "long still decoding");
    let tl_now = table(&r, long);
    assert_eq!(tl_now[..4], tl[..4], "long kept the shared prefix blocks");
    same_bytes(
        &read_blocks(&r, &tl_now[..4]),
        &parked,
        "long's prefix bytes",
    );
    step_until(&mut r, "parks of short terminal", 200, |r| {
        status(r).parking_jobs == 0
    })
    .await;
    assert_refcounts_match_tables(&r, &[long], "after short finished");

    step_until(&mut r, "long finished, parks terminal", 300, |r| {
        finished(r, long) && status(r).parking_jobs == 0
    })
    .await;
    assert_eq!(
        toks(&r, short),
        cold(&short_prompt, 2).await,
        "short greedy"
    );
    assert_eq!(toks(&r, long), cold(&long_prompt, 8).await, "long greedy");
    assert_ne!(
        toks(&r, short)[..2],
        toks(&r, long)[..2],
        "suffixes diverge"
    );
    assert_refcounts_match_tables(&r, &[], "end");
}

// ---------------------------------------------------------------------------
// B4: memory pressure; B3 + B4: queue-full and disk-full under load
// ---------------------------------------------------------------------------

/// #485 B4 (ADR 7.1): a 20-block pool against about 39 blocks of offered work,
/// a restore block budget of two restores, four restore and two park store
/// operations, and restore loads held at first. Resident requests and
/// restoring requests all finish within a bounded number of steps, each with
/// its cold greedy output; the restore budget is never exceeded; every block
/// and holder comes back.
///
/// Two running sequences at a time (`max_batch_size = 2`) keep the scheduler
/// from preempting, so every greedy comparison here is about restores, not
/// resumes. The test asserts that no preemption happened. Resume after
/// preemption has its own test below
/// (`preempted_sequence_resumes_without_re_emitting_tokens`).
#[tokio::test]
async fn memory_pressure_restore_and_admission_make_progress_without_deadlock() {
    let opts = Opts {
        total_blocks: 20,
        watermark_blocks: 2,
        restore_max_blocks: 8,
        max_restore_ops: 4,
        max_park_ops: 2,
        max_batch_size: 2,
        ..Opts::default()
    };
    let mut r = rig_with(opts.clone());
    let pa = park_prefix(&mut r, &prompt_a()).await;
    let pb = park_prefix(&mut r, &prompt_b()).await;

    let mut work: Vec<(Vec<u32>, usize)> = Vec::new();
    for i in 0..3u32 {
        let p: Vec<u32> = (0..12).map(|t| (t * 7 + i * 13 + 1) % 97).collect();
        work.push((p, 8));
    }
    for (i, base) in [&pa, &pb, &pa, &pb].iter().enumerate() {
        let i = i as u32;
        work.push((with_suffix(base, &[i + 1, i + 40, i + 70]), 4));
    }

    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let mut seqs = Vec::new();
    for (p, n) in &work {
        seqs.push(submit(&mut r, p, *n));
    }
    let restoring = status(&r).restoring;
    assert!(restoring >= 1, "some restore started");
    let mut peak_restore_blocks = status(&r).restore_inflight_blocks;
    let mut min_free = free_blocks(&r);
    // While the loads are held, the plain requests still make progress.
    let emitted_plain = |r: &Rig| seqs[..3].iter().map(|&s| emitted(r, s)).sum::<usize>();
    let before = emitted_plain(&r);
    for _ in 0..5 {
        step(&mut r).await;
        peak_restore_blocks = peak_restore_blocks.max(status(&r).restore_inflight_blocks);
        min_free = min_free.min(free_blocks(&r));
    }
    assert!(
        emitted_plain(&r) >= before + 5,
        "plain requests progressed while restores were stalled"
    );
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    st(&r).gate.release(Kind::Load);

    let t0 = Instant::now();
    let start = r.steps;
    while !(seqs.iter().all(|&s| finished(&r, s)) && status(&r).parking_jobs == 0) {
        assert!(
            r.steps - start < 2000,
            "no progress within 2000 steps (deadlock): {:?}",
            status(&r)
        );
        assert!(t0.elapsed() < HANG_GUARD, "hang guard");
        step(&mut r).await;
        peak_restore_blocks = peak_restore_blocks.max(status(&r).restore_inflight_blocks);
        min_free = min_free.min(free_blocks(&r));
    }

    assert_eq!(
        r.spine.scheduler.metrics().preempted_requests,
        0,
        "no preemption (see the doc comment)"
    );
    assert!(
        min_free <= opts.watermark_blocks + 4,
        "the pool was under pressure: lowest free count {min_free}"
    );
    assert!(
        peak_restore_blocks <= opts.restore_max_blocks,
        "restore budget exceeded: {peak_restore_blocks}"
    );
    for (i, (p, n)) in work.iter().enumerate() {
        assert_eq!(toks(&r, seqs[i]), cold(p, *n).await, "request {i} greedy");
    }
    let s = status(&r).stats;
    assert!(
        s.restore_committed_total >= 1,
        "some restore committed: {s:?}"
    );
    assert_eq!(
        s.restore_committed_total + s.restore_fallback_prefill_total,
        4 + 3 + 2,
        "every request either restored or prefilled: {s:?}"
    );
    assert_eq!(st(&r).worker.stats().completion_overflow_total, 0);
    assert_refcounts_match_tables(&r, &[], "end");
}

fn prompt_c() -> Vec<u32> {
    (0..15).map(|t| (t * 11 + 3) % 97).collect()
}

fn prompt_d() -> Vec<u32> {
    (0..15).map(|t| (t * 13 + 6) % 97).collect()
}

fn prompt_e() -> Vec<u32> {
    (0..15).map(|t| (t * 17 + 9) % 97).collect()
}

/// #485 B3 + B4 (ADR 7.1, 7.4; #484 A4): two park operations allowed and both
/// held, so the next turn's parks are refused with `QueueFull`; two restore
/// operations allowed and both held, so a second restore is a plain prefill;
/// then the disk is full and every park write fails. Throughout, a resident
/// sequence emits one token per step, and objects published before stay
/// served. Every output equals its cold greedy baseline and no holder or block
/// leaks.
#[tokio::test]
async fn queue_full_and_disk_full_under_load_never_lose_resident_progress() {
    let mut r = rig_with(Opts {
        max_restore_ops: 2,
        max_park_ops: 2,
        ..Opts::default()
    });
    let pa = park_prefix(&mut r, &prompt_a()).await;
    let pb = park_prefix(&mut r, &prompt_b()).await;
    let pk = keys(&r, &pa);
    let p0 = payload(&r, &pk[0]).expect("chunk 0");

    let resident = submit(&mut r, &unrelated(), 40);
    step_until(&mut r, "resident decoding", 50, |r| {
        emitted(r, resident) >= 1
    })
    .await;
    let mut n;

    // Park queue full: p1's two parks are held and take both park slots;
    // p2's two parks are refused at submit and their holders released.
    st(&r).gate.hold_stores.store(true, Ordering::SeqCst);
    let p1 = submit(&mut r, &prompt_c(), 2);
    step_until(&mut r, "p1 finished", 50, |r| finished(r, p1)).await;
    wait_held(&r, Kind::Store, 2);
    let before = status(&r).stats;
    let p2 = submit(&mut r, &prompt_d(), 2);
    n = emitted(&r, resident);
    step_until(&mut r, "p2 finished", 50, |r| finished(r, p2)).await;
    assert!(emitted(&r, resident) > n, "resident progressed");
    let s = status(&r).stats;
    assert_eq!(s.store_queue_full_total, before.store_queue_full_total + 2);
    assert_eq!(s.park_failed_total, before.park_failed_total + 2);
    assert_eq!(status(&r).parking_jobs, 2, "only p1's held parks in flight");
    assert_eq!(holder_blocks(&r), 4, "p2's refused holders released");
    for _ in 0..3 {
        n = emitted(&r, resident);
        step(&mut r).await;
        assert_eq!(
            emitted(&r, resident),
            n + 1,
            "resident while park queue full"
        );
    }

    // Restore queue full: qa's two loads are held and take both restore
    // slots; qb (another stored prefix) is refused and prefills plainly.
    st(&r).gate.hold_loads.store(true, Ordering::SeqCst);
    let qa_prompt = with_suffix(&pa, &[13, 29, 51]);
    let qa = submit(&mut r, &qa_prompt, 3);
    assert_eq!(state_of(&r, qa), ResidencyState::Restoring);
    wait_held(&r, Kind::Load, 2);
    let before = status(&r).stats;
    let qb_prompt = with_suffix(&pb, &[4, 8, 15]);
    let qb = submit(&mut r, &qb_prompt, 3);
    assert_eq!(
        state_of(&r, qb),
        ResidencyState::Resident,
        "qb not restoring"
    );
    let s = status(&r).stats;
    assert_eq!(
        s.restore_fallback_prefill_total,
        before.restore_fallback_prefill_total + 1
    );
    assert!(s.store_queue_full_total > before.store_queue_full_total);
    n = emitted(&r, resident);
    step_until(&mut r, "qb finished", 100, |r| finished(r, qb)).await;
    assert!(emitted(&r, resident) > n, "resident progressed");
    assert_eq!(phase_of(&r, qa), Some(SequencePhase::AwaitingRestore));

    // Disk full: every park submitted from now on fails at its write. p1's
    // held parks were accepted before and publish when released.
    st(&r).gate.disk_full.store(true, Ordering::SeqCst);
    st(&r).gate.hold_stores.store(false, Ordering::SeqCst);
    assert_eq!(st(&r).gate.release(Kind::Store), 2);
    st(&r).gate.hold_loads.store(false, Ordering::SeqCst);
    assert_eq!(st(&r).gate.release(Kind::Load), 2);
    step_until(&mut r, "p1's parks published", 100, |r| {
        status(r).parking_jobs == 0
    })
    .await;
    let failed = status(&r).stats.park_failed_total;
    let disk_full = st(&r).fake.stats().disk_full_total;
    let p3 = submit(&mut r, &prompt_e(), 2);
    step_until(&mut r, "p3 and qa finished, parks terminal", 300, |r| {
        finished(r, p3) && finished(r, qa) && finished(r, qb) && status(r).parking_jobs == 0
    })
    .await;
    let s = status(&r).stats;
    assert!(
        s.park_failed_total >= failed + 2,
        "disk-full parks failed: {s:?}"
    );
    assert!(st(&r).fake.stats().disk_full_total >= disk_full + 2);
    let mut p3_computed = prompt_e();
    p3_computed.push(toks(&r, p3)[0]);
    for k in keys(&r, &p3_computed) {
        assert!(
            payload(&r, &k).is_none(),
            "nothing published on a full disk"
        );
    }
    same_bytes(
        &payload(&r, &pk[0]).expect("published"),
        &p0,
        "earlier object intact",
    );
    assert_eq!(s.restore_committed_total, 1, "qa restored: {s:?}");

    // The last usable KV is still served after the full disk.
    st(&r).gate.disk_full.store(false, Ordering::SeqCst);
    let qc_prompt = with_suffix(&pa, &[77, 66, 55]);
    let qc = submit(&mut r, &qc_prompt, 3);
    assert_eq!(state_of(&r, qc), ResidencyState::Restoring);
    step_until(&mut r, "resident, qc finished; parks terminal", 500, |r| {
        finished(r, qc) && finished(r, resident) && status(r).parking_jobs == 0
    })
    .await;
    assert_eq!(status(&r).stats.restore_committed_total, 2, "qc restored");

    assert_eq!(toks(&r, resident), cold(&unrelated(), 40).await, "resident");
    assert_eq!(toks(&r, p1), cold(&prompt_c(), 2).await, "p1");
    assert_eq!(toks(&r, p2), cold(&prompt_d(), 2).await, "p2");
    assert_eq!(toks(&r, qa), cold(&qa_prompt, 3).await, "qa");
    assert_eq!(toks(&r, qb), cold(&qb_prompt, 3).await, "qb");
    assert_eq!(toks(&r, p3), cold(&prompt_e(), 2).await, "p3");
    assert_eq!(toks(&r, qc), cold(&qc_prompt, 3).await, "qc");
    assert_refcounts_match_tables(&r, &[], "end");
}

// ---------------------------------------------------------------------------
// Preemption under KV pressure with persistence off: resume must not re-emit
// ---------------------------------------------------------------------------

/// A sequence preempted by the watermark check and resumed later must deliver
/// every token to its sink exactly once, and its output must equal a run that
/// was never preempted. Persistence is off (no store, no worker): this is pure
/// scheduler behaviour.
///
/// Geometry: a 10-block pool with a watermark of 3. Request A (15-token
/// prompt, 4 blocks) decodes alone for a few steps. Request B (another 4
/// blocks) is admitted while 3 or more blocks are free, which leaves fewer
/// than the watermark, so the next step preempts A (the lowest-priority
/// running sequence, first in running order). A needs 4 blocks plus the
/// watermark headroom to come back, so it resumes only after B finishes.
#[tokio::test]
async fn preempted_sequence_resumes_without_re_emitting_tokens() {
    let max_a = 12;
    let max_b = 4;
    let want_a = cold(&prompt_a(), max_a).await;
    let want_b = cold(&prompt_b(), max_b).await;

    let mut r = rig_with(Opts {
        persist: false,
        total_blocks: 10,
        watermark_blocks: 3,
        max_batch_size: 4,
        ..Opts::default()
    });
    let a = submit(&mut r, &prompt_a(), max_a);
    step_until(&mut r, "A emitted three tokens", 50, |r| emitted(r, a) >= 3).await;
    let emitted_before = emitted(&r, a);
    assert!(!finished(&r, a), "A still running");
    let b = submit(&mut r, &prompt_b(), max_b);
    step_until(&mut r, "A preempted", 50, |r| {
        r.spine.scheduler.metrics().preempted_requests >= 1
    })
    .await;
    assert_eq!(
        phase_of(&r, a),
        Some(SequencePhase::Preempted),
        "A is the preempted sequence"
    );
    step_until(&mut r, "A and B finished", 500, |r| {
        finished(r, a) && finished(r, b)
    })
    .await;

    assert_eq!(toks(&r, b), want_b, "B greedy");
    let got_a = toks(&r, a);
    assert!(
        emitted_before < got_a.len(),
        "A emitted more after resuming"
    );
    assert_eq!(
        got_a.len(),
        max_a,
        "every token of A reached the sink exactly once: {got_a:?}"
    );
    assert_eq!(
        got_a, want_a,
        "A after preemption equals the uninterrupted run"
    );
    assert_refcounts_match_tables(&r, &[], "end");
}
