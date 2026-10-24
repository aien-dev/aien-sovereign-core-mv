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

//! C6 (sc#484 + #485 seam, ADR 0003 sections 5, 6, 7.3, 8 and 18): the runtime
//! parks a finished turn at the scheduler's finish transaction and restores a
//! stored prefix when a request is submitted.
//!
//! Every test runs the real spine, the real scheduler, a real pooled KV manager
//! and the real C5 storage worker over the deterministic C2 fake store (host
//! only: no GPU, no files). The backend is a scripted stand-in that records
//! what the scheduler asked of it; the K/V bytes are painted into the pool by
//! the test so restored bytes can be compared with parked ones.
//!
//! Geometry: block size 4 tokens, chunk of 2 blocks (8 tokens). A prompt of 14
//! tokens with `max_tokens = 3` finishes with a computed prefix of 16 tokens
//! (prompt plus the first two generated tokens; the newest token has no K/V),
//! which is exactly two full chunks.

use aien_abi_core::{
    AienInferenceBackend, DecodeOutput, ModelConfig, SamplingParams, ScheduledBatch,
    SequenceRequest, StepMetrics,
};
use aien_kv_cache::{create_shared_kv_manager_with_pool, KvDType, KvPoolConfig, SharedKvManager};
use aien_kv_store::fake::{FakeConfig, FakeKvStore, Fault, JobScript};
use aien_kv_store::worker::{StorageWorker, WorkerConfig};
use aien_kv_store::{
    chunk_chain, BackendFormat, CancelOutcome, Capacity, ChunkHeader, ChunkKey, Completion,
    DrainReport, Incarnation, JobId, KvDtype, KvStore, NamespaceFields, NamespaceId, PinToken,
    PinnedDest, PinnedSource, RecoveryProgress,
};
use aien_platform::PlatformError;
use aien_runtime::kv_residency::{
    layout_strides, KvIdentity, KvResidency, KvResidencyConfig, KvResidencyStatus, ResidencyState,
};
use aien_runtime::shared_kv::KvPersistSettings;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{SchedulerConfig, SequenceId, SequencePhase};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BLOCK: usize = 4;
const CHUNK_BLOCKS: u32 = 2;
const CHUNK: usize = BLOCK * CHUNK_BLOCKS as usize;
const TOTAL_BLOCKS: usize = 64;

// ---------------------------------------------------------------------------
// Backend stand-in
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Rec {
    /// (request id, prompt tokens in this prefill chunk), in call order.
    prefills: Vec<(u64, usize)>,
    decodes: Vec<u64>,
    /// (request id, restored prefix length) passed to `restore_sequence`.
    restores: Vec<(u64, usize)>,
    /// When set, `restore_sequence` refuses (an S3 failure after every load
    /// completed: the objects stay in the store).
    fail_restore: bool,
}

struct TestBackend {
    rec: Arc<Mutex<Rec>>,
}

#[async_trait]
impl AienInferenceBackend for TestBackend {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut rec = self.rec.lock();
        let mut outputs = Vec::new();
        let mut prefill = 0;
        for r in &batch.prefill_requests {
            prefill += r.prompt_tokens.len();
            rec.prefills.push((r.request_id, r.prompt_tokens.len()));
            outputs.push(DecodeOutput::Token {
                request_id: r.request_id,
                token_id: 100,
                logprob: None,
            });
        }
        for &id in &batch.decode_requests {
            rec.decodes.push(id);
            outputs.push(DecodeOutput::Token {
                request_id: id,
                token_id: 101,
                logprob: None,
            });
        }
        Ok((
            outputs,
            StepMetrics {
                prefill_tokens_processed: prefill,
                decode_tokens_emitted: batch.decode_requests.len(),
                step_latency_us: 1,
                active_kv_blocks: 0,
            },
        ))
    }

    fn restore_sequence(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(), String> {
        let mut rec = self.rec.lock();
        if rec.fail_restore {
            return Err("restore refused (test switch)".into());
        }
        rec.restores.push((seq_id, tokens.len()));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

struct Rig {
    spine: AienRuntimeSpine,
    backend: TestBackend,
    rec: Arc<Mutex<Rec>>,
    kv: SharedKvManager,
    fake: Arc<FakeKvStore>,
    store: Arc<LateCancel>,
    worker: Arc<StorageWorker>,
    ns: NamespaceId,
    /// The residency clock (milliseconds), advanced by the test.
    clock: Arc<AtomicU64>,
}

fn pool_config() -> KvPoolConfig {
    KvPoolConfig {
        num_blocks: TOTAL_BLOCKS,
        block_size: BLOCK,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 8,
        dtype: KvDType::Bf16,
    }
}

fn fields(kv: &SharedKvManager) -> NamespaceFields<'static> {
    let layout = *kv.read().pool().expect("pool").layout();
    NamespaceFields {
        model_sha256: [7u8; 32],
        model_digest_kind: "sha256",
        model_id: "c6-test",
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 8,
        block_size: BLOCK as u32,
        rope_theta: 10000.0,
        rope_scaling: None,
        qk_norm: false,
        dtype: KvDtype::Bf16,
        layout: layout_strides(&layout),
        backend_format: BackendFormat::HostPagedPoolV1,
        chunk_tokens: CHUNK as u32,
        sharing_namespace: "default",
        compute_config_digest: [1u8; 32],
    }
}

/// The fake store, plus one switch: while `refuse_cancel` is set, `cancel`
/// answers `TooLate` (the transfer is past the point where it can be stopped,
/// ADR 5.5), so the job runs to its own terminal. Everything else delegates.
struct LateCancel {
    inner: Arc<FakeKvStore>,
    refuse_cancel: AtomicBool,
}

impl KvStore for LateCancel {
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
        self.inner.submit_store(header, src, inc)
    }
    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        self.inner.submit_load(key, expect, dst, inc)
    }
    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError> {
        self.inner.submit_flush(inc)
    }
    fn attach_waiter(&self, job: JobId, inc: Incarnation) -> Result<(), PlatformError> {
        self.inner.attach_waiter(job, inc)
    }
    fn poll(&self, max: usize) -> Vec<Completion> {
        self.inner.poll(max)
    }
    fn cancel(&self, job: JobId) -> CancelOutcome {
        if self.refuse_cancel.load(Ordering::SeqCst) {
            CancelOutcome::TooLate
        } else {
            self.inner.cancel(job)
        }
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

fn rig_with(watermark_blocks: usize, worker_paused: bool) -> Rig {
    let kv = create_shared_kv_manager_with_pool(TOTAL_BLOCKS, BLOCK, pool_config()).expect("pool");
    let f = fields(&kv);
    let identity = KvIdentity::new(&f).expect("identity");
    let ns = identity.namespace();
    let fake = Arc::new(FakeKvStore::new(FakeConfig::new(ns)));
    let store = Arc::new(LateCancel {
        inner: fake.clone(),
        refuse_cancel: AtomicBool::new(false),
    });
    let drive = {
        let fake = fake.clone();
        move || fake.advance(1000)
    };
    let worker = Arc::new(
        StorageWorker::spawn(
            store.clone(),
            WorkerConfig {
                idle_poll: Duration::from_micros(200),
                start_paused: worker_paused,
                ..WorkerConfig::default()
            },
            drive,
        )
        .expect("spawn worker"),
    );
    let sched = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 4096,
        max_prefill_tokens: 4096,
        prefill_chunk_size: 4096,
        chunk_prefill: true,
        watermark_blocks,
    };
    let mut spine = AienRuntimeSpine::new(64, sched, kv.clone());
    let residency = KvResidency::new(
        KvResidencyConfig {
            chunk_blocks: CHUNK_BLOCKS,
            context_cap: None,
            restore_max_blocks: TOTAL_BLOCKS,
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
    let clock = Arc::new(AtomicU64::new(0));
    let mut residency = residency;
    {
        let clock = clock.clone();
        residency.set_clock(Arc::new(move || clock.load(Ordering::SeqCst)));
    }
    spine.enable_kv_residency(residency);
    let rec = Arc::new(Mutex::new(Rec::default()));
    Rig {
        spine,
        backend: TestBackend { rec: rec.clone() },
        rec,
        kv,
        fake,
        store,
        worker,
        ns,
        clock,
    }
}

fn rig() -> Rig {
    rig_with(0, false)
}

fn request(prompt: &[u32], max_tokens: usize) -> SequenceRequest {
    SequenceRequest {
        request_id: 0,
        prompt_tokens: prompt.to_vec(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority: 1,
    }
}

fn status(r: &Rig) -> KvResidencyStatus {
    r.spine
        .kv_residency()
        .expect("residency installed")
        .status()
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

fn free_blocks(r: &Rig) -> usize {
    r.kv.read().free_block_count()
}

fn holder_blocks(r: &Rig) -> usize {
    r.kv.read().holder_block_count()
}

async fn step(r: &mut Rig) {
    r.spine.step(&mut r.backend).await.expect("step");
}

/// Steps (sleeping a little between steps so the storage worker thread runs)
/// until `cond` holds.
async fn step_until(r: &mut Rig, what: &str, cond: impl Fn(&Rig) -> bool) {
    let t0 = Instant::now();
    loop {
        if cond(r) {
            return;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "timed out waiting for: {what}; status {:?}",
            status(r)
        );
        step(r).await;
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Distinct bytes for block `j` of the parked turn.
fn pattern(j: usize, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (j as u8).wrapping_mul(29).wrapping_add(7) ^ (i as u8))
        .collect()
}

fn block_len(r: &Rig) -> usize {
    r.kv.read().pool().expect("pool").block_bytes()
}

fn read_block(r: &Rig, b: usize) -> Vec<u8> {
    let kv = r.kv.read();
    let pool = kv.pool().expect("pool");
    let len = pool.block_bytes();
    // SAFETY: test-only read of one in-range pool block while the KV read lock
    // is held; nothing writes the pool on this thread meanwhile.
    unsafe { std::slice::from_raw_parts(pool.base_ptr().add(b * len), len) }.to_vec()
}

fn paint(r: &Rig, seq: u64) {
    let mut kv = r.kv.write();
    let blocks = kv
        .get_block_table(seq)
        .expect("table to paint")
        .block_ids
        .clone();
    let pool = kv.pool_mut().expect("pool");
    let len = pool.block_bytes();
    let base = pool.base_mut_ptr();
    for (j, &b) in blocks.iter().enumerate() {
        let bytes = pattern(j, len);
        // SAFETY: in-range block of the pool, written under the KV write lock.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(b * len), len) };
    }
}

fn prompt14() -> Vec<u32> {
    (1..=14u32).map(|t| t * 3 + 1).collect()
}

/// The computed prefix the scheduler reports for `prompt14()` after a turn
/// with `max_tokens = 3`: the prompt, then the prefill token (100) and the
/// first decode token (101); the last decode token has no K/V.
fn computed16() -> Vec<u32> {
    let mut v = prompt14();
    v.extend_from_slice(&[100, 101]);
    v
}

fn keys(r: &Rig, tokens: &[u32]) -> Vec<ChunkKey> {
    chunk_chain(&r.ns, CHUNK as u32, tokens).expect("chain")
}

/// Runs one turn to completion, painting its blocks after the prefill, and
/// waits until every park job it caused reached its terminal completion.
async fn park_turn(r: &mut Rig, prompt: &[u32], max_tokens: usize) -> u64 {
    let seq = r.spine.submit_request(request(prompt, max_tokens));
    step(r).await;
    paint(r, seq);
    step_until(r, "turn finished and parks terminal", |r| {
        finished(r, seq) && status(r).parking_jobs == 0
    })
    .await;
    seq
}

fn first_prefill_len(r: &Rig, seq: u64) -> Option<usize> {
    r.rec
        .lock()
        .prefills
        .iter()
        .find(|(id, _)| *id == seq)
        .map(|&(_, n)| n)
}

fn restores_of(r: &Rig, seq: u64) -> Vec<usize> {
    r.rec
        .lock()
        .restores
        .iter()
        .filter(|(id, _)| *id == seq)
        .map(|&(_, n)| n)
        .collect()
}

fn restore_jobs(r: &Rig, seq: u64) -> Vec<JobId> {
    r.spine
        .kv_residency()
        .expect("residency")
        .restore_jobs_of(seq)
}

/// Waits until the worker handed `job` to the fake store, and returns the
/// fake's id for it.
async fn inner_of(r: &mut Rig, job: JobId) -> JobId {
    let t0 = Instant::now();
    loop {
        if let Some(inner) = r.worker.inner_job(job) {
            return inner;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "job never dispatched"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------
// Park
// ---------------------------------------------------------------------------

/// ADR 6.2: a finished turn publishes its full chunks and its partial tail,
/// keyed over the computed prefix only (the newest token, which has no K/V, is
/// never stored), and every storage holder is released afterwards.
#[tokio::test]
async fn finished_turn_publishes_full_chunks_and_partial_tail() {
    let mut r = rig();
    let prompt: Vec<u32> = (1..=20u32).collect();
    park_turn(&mut r, &prompt, 3).await;

    // Computed: 20 prompt tokens + 100 + 101 = 22 = 8 + 8 + 6.
    let mut computed = prompt.clone();
    computed.extend_from_slice(&[100, 101]);
    let ks = keys(&r, &computed);
    assert_eq!(ks.len(), 3);
    let bb = block_len(&r);
    for (i, k) in ks.iter().enumerate() {
        let payload = r.fake.payload_of(k).expect("chunk published");
        assert_eq!(payload.len(), 2 * bb, "chunk {i} is two pool blocks");
        if i < 2 {
            // Full chunks carry the painted bytes of their blocks unchanged.
            let mut want = pattern(2 * i, bb);
            want.extend(pattern(2 * i + 1, bb));
            assert_eq!(payload, want, "chunk {i} bytes");
        }
    }

    // The 23-token chain (with the uncomputed last token) is never stored.
    let mut all = computed.clone();
    all.push(101);
    let over = keys(&r, &all);
    assert!(
        r.fake.payload_of(&over[2]).is_none(),
        "uncomputed tail stored"
    );

    let s = status(&r);
    assert_eq!(s.stats.park_jobs_total, 3);
    assert_eq!(s.stats.park_published_total, 3);
    assert_eq!(s.stats.park_failed_total, 0);
    assert_eq!(holder_blocks(&r), 0, "holders released after publication");
    assert_eq!(
        free_blocks(&r),
        TOTAL_BLOCKS,
        "every block back in the pool"
    );
}

/// ADR 5.4 / 8 rule 1: a park never makes `step` wait for storage. With the
/// worker not dispatching at all, the finishing step returns, other requests
/// keep decoding, and the park holders keep their blocks until the store
/// finally reports the terminal.
#[tokio::test]
async fn park_never_blocks_step() {
    let mut r = rig_with(0, true);
    let parked = r.spine.submit_request(request(&prompt14(), 3));
    let other = r.spine.submit_request(request(&[9, 8, 7, 6, 5], 40));
    step(&mut r).await;
    paint(&r, parked);
    let t0 = Instant::now();
    while !finished(&r, parked) {
        step(&mut r).await;
        assert!(t0.elapsed() < Duration::from_secs(5), "turn never finished");
    }
    let s = status(&r);
    assert_eq!(s.parking_jobs, 2, "two park jobs in flight");
    assert_eq!(s.stats.park_published_total, 0);
    assert!(holder_blocks(&r) >= 4, "holders keep the parked blocks");

    let decodes_before = r.rec.lock().decodes.iter().filter(|&&d| d == other).count();
    for _ in 0..5 {
        let t = Instant::now();
        step(&mut r).await;
        assert!(
            t.elapsed() < Duration::from_secs(1),
            "step waited on storage"
        );
    }
    let decodes_after = r.rec.lock().decodes.iter().filter(|&&d| d == other).count();
    assert_eq!(
        decodes_after,
        decodes_before + 5,
        "other request kept decoding"
    );
    assert_eq!(status(&r).parking_jobs, 2, "parks still in flight");

    r.worker.resume_dispatch();
    step_until(&mut r, "parks published", |r| status(r).parking_jobs == 0).await;
    assert_eq!(status(&r).stats.park_published_total, 2);
    assert_eq!(holder_blocks(&r), 0);
}

// ---------------------------------------------------------------------------
// Restore
// ---------------------------------------------------------------------------

/// ADR 6.3 step 1: the restorable prefix is capped at `prompt.len() - 1`, so the
/// last prompt token is always computed. A prompt that equals the parked
/// 16-token prefix restores only the first chunk (8 tokens): the second chunk
/// would reach the cap, and a 7-token tail was never stored.
#[tokio::test]
async fn restore_caps_prefix_at_prompt_len_minus_one() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;

    let seq = r.spine.submit_request(request(&computed16(), 2));
    assert_eq!(state_of(&r, seq), ResidencyState::Restoring);
    step_until(&mut r, "restored request finished", |r| finished(r, seq)).await;

    assert_eq!(restores_of(&r, seq), vec![8]);
    assert_eq!(first_prefill_len(&r, seq), Some(8), "suffix prefill only");
    let s = status(&r);
    assert_eq!(s.stats.restore_tokens_total, 8);
    assert_eq!(s.stats.restore_committed_total, 1);
}

/// ADR 6.3 step 4: a miss is a plain prefill. The submit succeeds, nothing is
/// refused, nothing is held, the request completes.
#[tokio::test]
async fn restore_miss_falls_back_to_plain_prefill_with_no_refusal() {
    let mut r = rig();
    let prompt: Vec<u32> = (50..70u32).collect();
    let seq = r.spine.submit_request(request(&prompt, 3));
    assert_eq!(state_of(&r, seq), ResidencyState::Resident);
    assert_eq!(phase_of(&r, seq), Some(SequencePhase::Waiting));
    step_until(&mut r, "request finished", |r| finished(r, seq)).await;

    assert_eq!(first_prefill_len(&r, seq), Some(20), "plain prefill");
    assert!(restores_of(&r, seq).is_empty());
    let s = status(&r);
    assert_eq!(s.stats.restore_jobs_total, 0);
    assert_eq!(s.stats.restore_fallback_prefill_total, 1);
    assert_eq!(s.stats.restore_failed_total, 0);
}

/// ADR 5.5 / 6.3 step 4: a failed load frees the restore holders and the
/// request is prefilled from position zero.
#[tokio::test]
async fn restore_failure_frees_holder_and_prefills() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;
    let ks = keys(&r, &computed16());
    assert!(r.fake.corrupt_payload(&ks[0]), "corrupt chunk 0");
    let fallbacks = status(&r).stats.restore_fallback_prefill_total;

    let mut prompt = computed16();
    prompt.extend_from_slice(&[500, 501, 502, 503]);
    let seq = r.spine.submit_request(request(&prompt, 2));
    assert_eq!(state_of(&r, seq), ResidencyState::Restoring);
    step_until(&mut r, "request finished and parks terminal", |r| {
        finished(r, seq) && status(r).parking_jobs == 0
    })
    .await;

    assert!(restores_of(&r, seq).is_empty(), "nothing restored");
    assert_eq!(
        first_prefill_len(&r, seq),
        Some(prompt.len()),
        "plain prefill"
    );
    let s = status(&r);
    assert_eq!(s.stats.restore_failed_total, 1);
    assert_eq!(s.stats.restore_fallback_prefill_total, fallbacks + 1);
    assert_eq!(s.restore_inflight_blocks, 0);
    assert_eq!(holder_blocks(&r), 0, "restore holders freed");
    assert_eq!(free_blocks(&r), TOTAL_BLOCKS, "no block leaked");
}

/// ADR 5.3 (B3 through the runtime): a restore that outlives its request
/// cannot touch the next request that reuses the same arena slot. The second
/// request has a different sequence id (new generation); the first request's
/// completions are counted stale and only release their holder copies.
#[tokio::test]
async fn stale_completion_cannot_touch_reused_sequence_id() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;

    let mut prompt = computed16();
    prompt.extend_from_slice(&[600, 601, 602]);
    r.fake.script(JobScript::fault(Fault::Hold));
    r.fake.script(JobScript::fault(Fault::Hold));
    let a = r.spine.submit_request(request(&prompt, 2));
    let jobs = restore_jobs(&r, a);
    assert_eq!(jobs.len(), 2);
    let mut inner = Vec::new();
    for &j in &jobs {
        inner.push(inner_of(&mut r, j).await);
    }

    // The client goes away: the request leaves the scheduler arena.
    assert!(r
        .spine
        .scheduler
        .arena_mut()
        .free_sequence(SequenceId::from_u64(a).unwrap()));
    let b = r.spine.submit_request(request(&prompt, 2));
    assert_eq!(a >> 32, b >> 32, "same arena slot");
    assert_ne!(a, b, "new incarnation");

    step(&mut r).await;
    for &i in &inner {
        assert!(r.fake.release(i));
    }
    let t0 = Instant::now();
    while !finished(&r, b) {
        assert!(
            r.kv.read().get_block_table(a).is_none(),
            "stale id got a table"
        );
        assert!(t0.elapsed() < Duration::from_secs(10), "b never finished");
        step(&mut r).await;
        std::thread::sleep(Duration::from_millis(1));
    }
    step_until(&mut r, "parks terminal", |r| status(r).parking_jobs == 0).await;

    assert!(
        restores_of(&r, a).is_empty(),
        "the old request restored nothing"
    );
    assert_eq!(restores_of(&r, b), vec![16]);
    assert_eq!(first_prefill_len(&r, b), Some(3));
    let s = status(&r);
    assert_eq!(
        s.stats.store_stale_completions_total, 2,
        "a's two terminals"
    );
    assert_eq!(
        s.restore_inflight_blocks, 0,
        "a's restore accounting closed"
    );
    assert_eq!(s.stats.restore_committed_total, 1);
    assert_eq!(holder_blocks(&r), 0);
    assert_eq!(free_blocks(&r), TOTAL_BLOCKS);
}

/// ADR 6.3 step 2: a request whose restore is in flight is skipped by
/// admission until every chunk of its restore completed; then it prefills only
/// the remaining suffix.
#[tokio::test]
async fn awaiting_restore_is_not_admitted_until_all_chunks_completed() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;

    let mut prompt = computed16();
    prompt.extend_from_slice(&[700, 701, 702]);
    r.fake.script(JobScript::default());
    r.fake.script(JobScript::fault(Fault::Hold));
    let seq = r.spine.submit_request(request(&prompt, 2));
    let jobs = restore_jobs(&r, seq);
    assert_eq!(jobs.len(), 2);
    let held = inner_of(&mut r, jobs[1]).await;

    step_until(&mut r, "first chunk completed", |r| {
        r.spine.kv_residency().unwrap().restore_progress(seq) == Some((1, 2))
    })
    .await;
    for _ in 0..5 {
        step(&mut r).await;
        assert_eq!(phase_of(&r, seq), Some(SequencePhase::AwaitingRestore));
        assert_eq!(first_prefill_len(&r, seq), None, "admitted too early");
        assert!(r.kv.read().get_block_table(seq).is_none());
    }

    assert!(r.fake.release(held));
    step_until(&mut r, "request finished", |r| finished(r, seq)).await;
    assert_eq!(restores_of(&r, seq), vec![16]);
    assert_eq!(first_prefill_len(&r, seq), Some(3));
}

/// ADR 5.4 S2 / 7.1: a restore that would push free blocks below the
/// scheduler's watermark is refused before anything is allocated or submitted,
/// and the request becomes a plain prefill.
#[tokio::test]
async fn restore_refused_below_watermark_falls_back() {
    let mut r = rig_with(4, false);
    park_turn(&mut r, &prompt14(), 3).await;

    // Leave 6 free blocks: a 16-token restore needs 4, 6 < 4 + 4.
    let foreign = 0xDEAD_0000_0001u64;
    r.kv.write()
        .allocate_sequence(foreign, &vec![0u32; (TOTAL_BLOCKS - 6) * BLOCK])
        .expect("occupy");
    assert_eq!(free_blocks(&r), 6);

    let mut prompt = computed16();
    prompt.extend_from_slice(&[800, 801, 802]);
    let fallbacks = status(&r).stats.restore_fallback_prefill_total;
    let seq = r.spine.submit_request(request(&prompt, 2));
    assert_eq!(state_of(&r, seq), ResidencyState::Resident);
    assert_eq!(phase_of(&r, seq), Some(SequencePhase::Waiting));
    let s = status(&r);
    assert_eq!(s.stats.restore_jobs_total, 0, "nothing submitted");
    assert_eq!(s.stats.restore_refused_watermark_total, 1);
    assert_eq!(s.stats.restore_fallback_prefill_total, fallbacks + 1);
    assert_eq!(holder_blocks(&r), 0, "nothing allocated");
    assert_eq!(free_blocks(&r), 6);

    r.kv.write().free_sequence(foreign).expect("free");
    step_until(&mut r, "request finished", |r| finished(r, seq)).await;
    assert_eq!(first_prefill_len(&r, seq), Some(prompt.len()));
    assert!(restores_of(&r, seq).is_empty());
}

/// ADR 5.5 (B3, review finding 5): chunk 1 fails while chunk 0 is still in
/// transfer. The transaction drains: the request stays held, chunk 0's
/// destination stays allocated until chunk 0's own terminal arrives, and only
/// then the holders are freed and the request prefills from zero. Chunk 0's
/// cancel is answered `TooLate`, so its terminal is a successful load whose
/// bytes are discarded.
#[tokio::test]
async fn multi_chunk_restore_failure_does_not_free_destination_in_use() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;
    let ks = keys(&r, &computed16());
    assert!(r.fake.corrupt_payload(&ks[1]), "corrupt chunk 1");
    // Chunk 0 is mid-transfer: its cancel comes too late, so it must run to
    // its own terminal before its destination may be released.
    r.store.refuse_cancel.store(true, Ordering::SeqCst);

    let mut prompt = computed16();
    prompt.extend_from_slice(&[900, 901, 902]);
    r.fake.script(JobScript::fault(Fault::Hold));
    r.fake.script(JobScript::default());
    let seq = r.spine.submit_request(request(&prompt, 2));
    let jobs = restore_jobs(&r, seq);
    assert_eq!(jobs.len(), 2);
    let held = inner_of(&mut r, jobs[0]).await;

    step_until(&mut r, "transaction draining", |r| {
        state_of(r, seq) == ResidencyState::Draining
    })
    .await;
    for _ in 0..5 {
        step(&mut r).await;
        std::thread::sleep(Duration::from_millis(1));
        assert_eq!(state_of(&r, seq), ResidencyState::Draining);
        assert_eq!(phase_of(&r, seq), Some(SequencePhase::AwaitingRestore));
        assert_eq!(first_prefill_len(&r, seq), None, "prefilled while in use");
        assert!(
            holder_blocks(&r) >= 2,
            "chunk 0 destination freed while its job runs"
        );
        assert!(r.kv.read().get_block_table(seq).is_none());
    }

    assert!(r.fake.release(held));
    step_until(&mut r, "request finished and parks terminal", |r| {
        finished(r, seq) && status(r).parking_jobs == 0
    })
    .await;
    assert!(restores_of(&r, seq).is_empty());
    assert_eq!(first_prefill_len(&r, seq), Some(prompt.len()));
    let s = status(&r);
    assert_eq!(s.stats.restore_failed_total, 1);
    assert_eq!(holder_blocks(&r), 0);
    assert_eq!(free_blocks(&r), TOTAL_BLOCKS);
}

/// ADR 5.4 S3: the request's table is assembled from the chunk holders only
/// when every job of the transaction is terminal, and then it holds the parked
/// bytes; the landing holders are released once the table adopted the blocks.
#[tokio::test]
async fn restore_txn_adopts_assembled_table_only_after_all_jobs_terminal() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;

    let mut prompt = computed16();
    prompt.extend_from_slice(&[1000, 1001, 1002]);
    r.fake.script(JobScript::default());
    r.fake.script(JobScript::fault(Fault::Hold));
    let seq = r.spine.submit_request(request(&prompt, 40));
    let jobs = restore_jobs(&r, seq);
    let held = inner_of(&mut r, jobs[1]).await;

    step_until(&mut r, "first chunk completed", |r| {
        r.spine.kv_residency().unwrap().restore_progress(seq) == Some((1, 2))
    })
    .await;
    for _ in 0..3 {
        step(&mut r).await;
        assert!(
            r.kv.read().get_block_table(seq).is_none(),
            "table assembled before every job was terminal"
        );
    }

    assert!(r.fake.release(held));
    step_until(&mut r, "restore committed", |r| {
        !restores_of(r, seq).is_empty()
    })
    .await;
    let table =
        r.kv.read()
            .get_block_table(seq)
            .expect("assembled table")
            .block_ids
            .clone();
    assert!(table.len() >= 4);
    let bb = block_len(&r);
    for (j, &b) in table.iter().take(4).enumerate() {
        assert_eq!(read_block(&r, b), pattern(j, bb), "restored block {j}");
    }
    assert_eq!(
        holder_blocks(&r),
        0,
        "landing holders released after adoption"
    );
    assert_eq!(state_of(&r, seq), ResidencyState::Prefilling);
    assert_eq!(restores_of(&r, seq), vec![16]);
    assert_eq!(first_prefill_len(&r, seq), Some(3));
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// METRICS_AND_SETTINGS: persistence is off unless `AIEN_KV_STORE_DIR` is set;
/// the chunk size is in blocks; bad values are refused, not defaulted.
#[test]
fn kv_persist_settings_are_off_by_default_and_refuse_bad_values() {
    let off = KvPersistSettings::from_lookup(|_| None).expect("defaults");
    assert!(off.store_dir.is_none());
    assert!(!off.enabled());
    assert_eq!(off.chunk_blocks, 16);
    assert_eq!(off.namespace, "default");

    let on = KvPersistSettings::from_lookup(|k| match k {
        "AIEN_KV_STORE_DIR" => Some("/tmp/kv".to_string()),
        "AIEN_KV_CHUNK_BLOCKS" => Some("4".to_string()),
        "AIEN_KV_RESTORE" => Some("0".to_string()),
        _ => None,
    })
    .expect("parse");
    assert!(on.enabled());
    assert_eq!(on.chunk_blocks, 4);
    assert!(on.park);
    assert!(!on.restore);

    for (k, v) in [
        ("AIEN_KV_CHUNK_BLOCKS", "0"),
        ("AIEN_KV_CHUNK_BLOCKS", "x"),
        ("AIEN_KV_PARK", "yes"),
        ("AIEN_KV_STORE_NAMESPACE", &"n".repeat(65)),
    ] {
        let got = KvPersistSettings::from_lookup(|q| (q == k).then(|| v.to_string()));
        assert!(got.is_err(), "{k}={v} accepted");
    }
}

// ---------------------------------------------------------------------------
// C7: retry window and indeterminate publications
// ---------------------------------------------------------------------------

/// Submits `prompt` and runs it to completion; returns the sequence id.
async fn run_to_finish(r: &mut Rig, prompt: &[u32]) -> u64 {
    let seq = r.spine.submit_request(request(prompt, 2));
    step_until(r, "request finished", |r| finished(r, seq)).await;
    seq
}

/// `AIEN_KV_RESTORE_RETRY_WINDOW_MS` (METRICS_AND_SETTINGS): after two failed
/// restores of one chain head within the window, the next request with that
/// head is prefilled without a lookup. Once the older failure leaves the
/// window, one lookup is allowed again. The clock is injected; no wall-clock
/// waits decide the outcome.
#[tokio::test]
async fn restore_retry_window_prefills_without_lookup_after_two_failures() {
    let mut r = rig();
    park_turn(&mut r, &prompt14(), 3).await;
    let mut prompt = computed16();
    prompt.extend_from_slice(&[500, 501, 502, 503]);
    r.rec.lock().fail_restore = true;

    // Failure 1 at t = 0 and failure 2 at t = 1 s: each looked up and loaded.
    for t in [0u64, 1_000] {
        r.clock.store(t, Ordering::SeqCst);
        let jobs = status(&r).stats.restore_jobs_total;
        let seq = run_to_finish(&mut r, &prompt).await;
        assert_eq!(
            first_prefill_len(&r, seq),
            Some(prompt.len()),
            "plain prefill"
        );
        let s = status(&r);
        assert_eq!(
            s.stats.restore_jobs_total,
            jobs + 2,
            "t={t}: lookup and load ran"
        );
    }
    assert_eq!(status(&r).stats.restore_failed_total, 2);

    // Third request at t = 2 s: inside the window, no lookup, no load.
    r.clock.store(2_000, Ordering::SeqCst);
    let before = status(&r).stats;
    let seq = r.spine.submit_request(request(&prompt, 2));
    assert_eq!(state_of(&r, seq), ResidencyState::Resident, "not restoring");
    assert_eq!(phase_of(&r, seq), Some(SequencePhase::Waiting));
    step_until(&mut r, "request finished", |r| finished(r, seq)).await;
    let s = status(&r).stats;
    assert_eq!(s.restore_jobs_total, before.restore_jobs_total, "no load");
    assert_eq!(s.restore_retry_suppressed_total, 1);
    assert_eq!(
        s.restore_fallback_prefill_total,
        before.restore_fallback_prefill_total + 1
    );
    assert_eq!(first_prefill_len(&r, seq), Some(prompt.len()));

    // A different chain head is not affected.
    let other: Vec<u32> = (900..920u32).collect();
    run_to_finish(&mut r, &other).await;
    assert_eq!(status(&r).stats.restore_retry_suppressed_total, 1);

    // t = 60.5 s: the t = 0 failure left the 60 s window; one failure remains,
    // so the lookup runs again, and with the backend healthy it restores.
    r.clock.store(60_500, Ordering::SeqCst);
    r.rec.lock().fail_restore = false;
    let seq = r.spine.submit_request(request(&prompt, 2));
    assert_eq!(state_of(&r, seq), ResidencyState::Restoring);
    step_until(&mut r, "request finished", |r| finished(r, seq)).await;
    assert_eq!(restores_of(&r, seq), vec![16], "restored after the window");
    assert_eq!(status(&r).stats.restore_retry_suppressed_total, 1);
    assert_eq!(holder_blocks(&r), 0);
}

/// ADR 2.4 / METRICS `kv_park_indeterminate_total`: a publication whose
/// directory sync failed after the rename is served but counted indeterminate.
///
/// Dispatch stays paused until both park jobs are submitted. The faulted first
/// job puts the fake store into `Degraded`, and a store in that state refuses
/// new submissions (`DeviceUnavailable`); with a running worker the second
/// submission raced that terminal and lost about one time in six (one job
/// counted, two failures), so the race is removed rather than tolerated.
#[tokio::test]
async fn indeterminate_park_publication_is_counted() {
    let mut r = rig_with(0, true);
    r.fake.script(JobScript::fault(Fault::RenameOkDirsyncFail));
    let seq = r.spine.submit_request(request(&prompt14(), 3));
    step(&mut r).await;
    paint(&r, seq);
    step_until(&mut r, "turn finished with both parks submitted", |r| {
        finished(r, seq) && status(r).parking_jobs == 2
    })
    .await;
    r.worker.resume_dispatch();
    step_until(&mut r, "parks terminal", |r| status(r).parking_jobs == 0).await;
    let s = status(&r).stats;
    assert_eq!(s.park_jobs_total, 2, "{s:?}");
    assert_eq!(s.park_indeterminate_total, 1, "{s:?}");
    // The object is published and served, its durability unknown (the fake
    // reports it on a failed terminal and the store goes Degraded).
    let ks = keys(&r, &computed16());
    let flags: Vec<Option<bool>> = ks.iter().map(|k| r.fake.is_indeterminate(k)).collect();
    assert!(
        flags.contains(&Some(true)),
        "an indeterminate object exists: {flags:?}"
    );
}
