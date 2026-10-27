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

//! C7 (sc#484; ADR 0003 sections 4.1, 6.5, 7.5 and 11): persistent KV across a
//! real process restart, through the daemon wiring
//! (`shared_kv::enable_persistent_kv`), on a temp-dir host filesystem store.
//!
//! Every restart test runs real child processes: the test binary re-runs itself
//! (`std::env::current_exe`, the pattern of `approved_crash_test.rs:80-89`)
//! with `--exact restart_child --ignored`, and the role in `KV_RS_ROLE`. Child 1
//! parks and exits; child 2 is a fresh process that recovers the store from
//! disk and restores. Host only: no GPU, no CUDA. The greedy test uses the CPU
//! reference backend (`ReferenceCpuBackend`) with the reference test weights.
//!
//! Geometry of the painted tests (as in kv_park_restore_test.rs): block size 4,
//! chunk of 2 blocks (8 tokens). A 14-token prompt with `max_tokens = 3`
//! finishes with a computed prefix of 16 tokens: two full chunks.

use aien_abi_core::{
    AienInferenceBackend, DecodeOutput, ModelConfig, SamplingParams, ScheduledBatch,
    SequenceRequest, StepMetrics,
};
use aien_inference_abi::backend::TensorBackend;
use aien_inference_abi::{ReferenceCpuBackend, TransformerWeights};
use aien_kv_cache::{create_shared_kv_manager_with_pool, KvDType, KvPoolConfig, SharedKvManager};
use aien_kv_store::{EngineConfig, EngineIdentity, StoreState, WeightFormat};
use aien_runtime::compute_config::HostEngineIdentity;
use aien_runtime::kv_residency::KvResidencyStatus;
use aien_runtime::shared_kv::{
    build_shared_kv_runtime, enable_persistent_kv, KvPersistSettings, ModelKvBinding,
    SharedKvSizing,
};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{ChannelCompletionSink, CompletionEvent, SchedulerConfig, SequenceId};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BLOCK: usize = 4;
const CHUNK_BLOCKS: u32 = 2;
const TOTAL_BLOCKS: usize = 64;

// ---------------------------------------------------------------------------
// Backend stand-in for the painted tests (no K/V writes: the test paints them)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Rec {
    prefills: Vec<(u64, usize)>,
    restores: Vec<(u64, usize)>,
}

struct PaintBackend {
    rec: Arc<Mutex<Rec>>,
}

#[async_trait]
impl AienInferenceBackend for PaintBackend {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut rec = self.rec.lock();
        let mut outputs = Vec::new();
        for r in &batch.prefill_requests {
            rec.prefills.push((r.request_id, r.prompt_tokens.len()));
            outputs.push(DecodeOutput::Token {
                request_id: r.request_id,
                token_id: 100,
                logprob: None,
            });
        }
        for &id in &batch.decode_requests {
            outputs.push(DecodeOutput::Token {
                request_id: id,
                token_id: 101,
                logprob: None,
            });
        }
        Ok((outputs, StepMetrics::default()))
    }

    fn restore_sequence(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(), String> {
        self.rec.lock().restores.push((seq_id, tokens.len()));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Identity inputs
// ---------------------------------------------------------------------------

/// The model the painted pool stands for (1 layer, 1 KV head, head dim 8).
fn paint_model(rms_norm_eps: f32) -> ModelConfig {
    ModelConfig {
        model_id: "c7-paint".to_string(),
        max_sequence_length: 256,
        block_size: BLOCK,
        num_layers: 1,
        num_heads: 1,
        head_dim: 8,
        num_kv_heads: 1,
        hidden_dim: 8,
        intermediate_dim: 16,
        vocab_size: 97,
        rms_norm_eps,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

/// The CPU reference model of the greedy test (shared_kv_runtime_test.rs).
fn greedy_model() -> ModelConfig {
    ModelConfig {
        model_id: "c7-greedy-reference".to_string(),
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
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 4096,
        max_prefill_tokens: 4096,
        prefill_chunk_size: 4096,
        chunk_prefill: true,
        watermark_blocks: 0,
    }
}

/// A 0700 temp dir: the host store refuses a root others can reach and
/// `tempfile` follows the umask (ADR 0003 section 19).
fn private_tempdir() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    tempfile::Builder::new()
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir()
        .expect("tempdir")
}

/// A generated test store key (ADR 0003 section 20), never a vault secret. The
/// same key across "restarts" so restored objects open.
const TEST_STORE_KEY: &str = "6b3f0c2d9a1e4f5b7c8d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d";

fn settings(dir: &Path) -> KvPersistSettings {
    let dir = dir.to_string_lossy().to_string();
    KvPersistSettings::from_lookup(|k| match k {
        "AIEN_KV_STORE_DIR" => Some(dir.clone()),
        "AIEN_KV_STORE_KEY" => Some(TEST_STORE_KEY.to_string()),
        "AIEN_KV_CHUNK_BLOCKS" => Some(CHUNK_BLOCKS.to_string()),
        _ => None,
    })
    .expect("settings")
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

// ---------------------------------------------------------------------------
// Painted rig (A1, A3)
// ---------------------------------------------------------------------------

struct PaintRig {
    spine: AienRuntimeSpine,
    backend: PaintBackend,
    rec: Arc<Mutex<Rec>>,
    kv: SharedKvManager,
}

fn paint_pool() -> SharedKvManager {
    create_shared_kv_manager_with_pool(
        TOTAL_BLOCKS,
        BLOCK,
        KvPoolConfig {
            num_blocks: TOTAL_BLOCKS,
            block_size: BLOCK,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: KvDType::Bf16,
        },
    )
    .expect("pool")
}

/// The daemon path: spine over one pool, then `enable_persistent_kv`.
/// The engine identity of the painted runs (ADR 1.2.2, sc#504). Each input can
/// be overridden by the parent through `KV_RS_WF`, `KV_RS_BUILD`, `KV_RS_ROUND`
/// and `KV_RS_WHOLE_PROMPT`, so one restart test changes exactly one of them.
struct PaintEngine {
    build: String,
    round: String,
    weight_format: WeightFormat,
    config_digest: [u8; 32],
}

impl PaintEngine {
    fn from_env() -> Self {
        let weight_format = match env_or("KV_RS_WF", "as-file").as_str() {
            "as-file" => WeightFormat::AsFile,
            "bf16" => WeightFormat::Bf16,
            "fp8-experts" => WeightFormat::Fp8Experts,
            "fp8-all" => WeightFormat::Fp8All,
            other => panic!("KV_RS_WF {other:?}"),
        };
        let config = EngineConfig {
            num_experts: 0,
            experts_per_token: 0,
            expert_intermediate_dim: 0,
            whole_prompt_prefill: env_or("KV_RS_WHOLE_PROMPT", "0") == "1",
            device_router: false,
            matmul_path: "c7-paint",
        };
        Self {
            build: env_or("KV_RS_BUILD", "c7-paint-build"),
            round: env_or("KV_RS_ROUND", "c7-paint@epoch-1"),
            weight_format,
            config_digest: config.digest().expect("engine config digest"),
        }
    }

    fn identity(&self) -> EngineIdentity<'_> {
        EngineIdentity {
            weight_format: self.weight_format,
            engine_build_id: &self.build,
            rounding_class: &self.round,
            engine_config_digest: self.config_digest,
        }
    }
}

fn paint_rig(dir: &Path, sha: u8, eps: f32) -> PaintRig {
    let kv = paint_pool();
    let mut spine = AienRuntimeSpine::new(64, scheduler(), kv.clone());
    let model = paint_model(eps);
    let engine = PaintEngine::from_env();
    let line = enable_persistent_kv(
        &mut spine,
        &settings(dir),
        &ModelKvBinding {
            model: &model,
            arith_id: "c7-paint-backend",
            model_sha256: [sha; 32],
            model_digest_kind: "sha256",
            engine: engine.identity(),
        },
        None,
    )
    .expect("enable persistent KV")
    .expect("persistence on with a store dir");
    assert!(line.contains("persistent KV"), "startup line: {line}");
    let rec = Arc::new(Mutex::new(Rec::default()));
    PaintRig {
        spine,
        backend: PaintBackend { rec: rec.clone() },
        rec,
        kv,
    }
}

fn kv_status(spine: &AienRuntimeSpine) -> KvResidencyStatus {
    spine.kv_residency_status().expect("residency installed")
}

fn store_state(spine: &AienRuntimeSpine) -> StoreState {
    spine
        .kv_residency()
        .expect("residency")
        .store_capacity()
        .state
}

/// Recovery runs on the storage worker in slices; the step loop keeps running
/// meanwhile. Wait (bounded) until the walk is done and admission is open.
fn wait_store_open(spine: &AienRuntimeSpine) {
    let t0 = Instant::now();
    while store_state(spine) == StoreState::Recovering {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "recovery never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(store_state(spine), StoreState::Open);
}

fn finished(spine: &AienRuntimeSpine, seq: u64) -> bool {
    SequenceId::from_u64(seq)
        .ok()
        .and_then(|id| spine.scheduler.arena().get(id).map(|_| ()))
        .is_none()
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

async fn paint_step_until(r: &mut PaintRig, what: &str, cond: impl Fn(&PaintRig) -> bool) {
    let t0 = Instant::now();
    loop {
        if cond(r) {
            return;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "timed out waiting for: {what}; status {:?}",
            kv_status(&r.spine)
        );
        r.spine.step(&mut r.backend).await.expect("step");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Distinct bytes for block `j` of the parked turn; `salt` makes a different
/// pattern (the negative control expects it and must fail).
fn pattern(j: usize, len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|i| {
            (j as u8)
                .wrapping_mul(29)
                .wrapping_add(7)
                .wrapping_add(salt)
                ^ (i as u8)
        })
        .collect()
}

fn table_blocks(kv: &SharedKvManager, seq: u64) -> Vec<usize> {
    kv.read()
        .get_block_table(seq)
        .expect("block table")
        .block_ids
        .clone()
}

fn read_block(kv: &SharedKvManager, b: usize) -> Vec<u8> {
    let kv = kv.read();
    let pool = kv.pool().expect("pool");
    let len = pool.block_bytes();
    // SAFETY: test-only read of one in-range pool block under the KV read lock.
    unsafe { std::slice::from_raw_parts(pool.base_ptr().add(b * len), len) }.to_vec()
}

fn paint(kv: &SharedKvManager, seq: u64) {
    let mut kv = kv.write();
    let blocks = kv.get_block_table(seq).expect("table").block_ids.clone();
    let pool = kv.pool_mut().expect("pool");
    let len = pool.block_bytes();
    let base = pool.base_mut_ptr();
    for (j, &b) in blocks.iter().enumerate() {
        let bytes = pattern(j, len, 0);
        // SAFETY: in-range pool block, written under the KV write lock.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(b * len), len) };
    }
}

fn prompt14() -> Vec<u32> {
    (1..=14u32).map(|t| t * 3 + 1).collect()
}

/// `prompt14()` plus the two computed generated tokens (100, 101) plus four
/// new tokens: restorable prefix 16 tokens, two full chunks.
fn restore_prompt() -> Vec<u32> {
    let mut v = prompt14();
    v.extend_from_slice(&[100, 101, 500, 501, 502, 503]);
    v
}

/// Shuts the store down with the configured drain deadline (ADR 7.5) and
/// checks that every park reached its terminal before the process exits.
fn drain(spine: &AienRuntimeSpine) {
    let handle = spine.kv_drain_handle().expect("drain handle");
    let report = handle.drain();
    assert!(
        report.unfinished.is_empty(),
        "unfinished at exit: {:?}",
        report.unfinished
    );
}

async fn child_paint_park(dir: &Path) {
    let sha: u8 = env_or("KV_RS_SHA", "7").parse().expect("sha");
    let eps: f32 = env_or("KV_RS_EPS", "0.00001").parse().expect("eps");
    let mut r = paint_rig(dir, sha, eps);
    wait_store_open(&r.spine);
    let seq = r.spine.submit_request(request(&prompt14(), 3));
    r.spine.step(&mut r.backend).await.expect("prefill step");
    paint(&r.kv, seq);
    paint_step_until(&mut r, "turn finished and parks published", |r| {
        let s = kv_status(&r.spine);
        finished(&r.spine, seq) && s.parking_jobs == 0 && s.stats.park_published_total == 2
    })
    .await;
    drain(&r.spine);
}

async fn child_paint_restore(dir: &Path) {
    let sha: u8 = env_or("KV_RS_SHA", "7").parse().expect("sha");
    let eps: f32 = env_or("KV_RS_EPS", "0.00001").parse().expect("eps");
    let salt: u8 = env_or("KV_RS_SALT", "0").parse().expect("salt");
    let expect = env_or("KV_RS_EXPECT", "restored");
    let mut r = paint_rig(dir, sha, eps);
    wait_store_open(&r.spine);

    // A6 (restart accounting): every object found at recovery is counted,
    // whatever its namespace. The miss role runs right after the park, so it
    // finds exactly the two parked chunks (of another namespace); later
    // children also find what earlier restore children parked.
    let cap = r.spine.kv_residency().expect("residency").store_capacity();
    if expect == "miss" {
        assert_eq!(cap.objects, 2, "both parked chunks rediscovered: {cap:?}");
    } else {
        assert!(cap.objects >= 2, "parked chunks rediscovered: {cap:?}");
    }
    assert!(cap.used_bytes > 0, "{cap:?}");

    let prompt = restore_prompt();
    let seq = r.spine.submit_request(request(&prompt, 3));
    if expect == "restored" {
        paint_step_until(&mut r, "restore committed", |r| {
            r.rec.lock().restores.iter().any(|&(id, _)| id == seq)
        })
        .await;
        let s = kv_status(&r.spine);
        assert_eq!(s.stats.restore_tokens_total, 16, "{s:?}");
        let len = r.kv.read().pool().expect("pool").block_bytes();
        let blocks = table_blocks(&r.kv, seq);
        for (j, &b) in blocks.iter().take(4).enumerate() {
            assert_eq!(
                read_block(&r.kv, b),
                pattern(j, len, salt),
                "restored block {j} differs from the bytes parked before the restart"
            );
        }
    } else {
        paint_step_until(&mut r, "request finished", |r| finished(&r.spine, seq)).await;
        let s = kv_status(&r.spine);
        assert_eq!(
            s.stats.restore_jobs_total, 0,
            "foreign namespace served: {s:?}"
        );
        assert_eq!(s.stats.restore_fallback_prefill_total, 1, "{s:?}");
        assert!(r.rec.lock().restores.is_empty());
        let first = r
            .rec
            .lock()
            .prefills
            .iter()
            .find(|(id, _)| *id == seq)
            .map(|&(_, n)| n);
        assert_eq!(
            first,
            Some(prompt.len()),
            "plain prefill from position zero"
        );
    }
    // Let this turn's own parks reach their terminals, so the drain has
    // nothing to cancel.
    paint_step_until(&mut r, "turn finished and parks terminal", |r| {
        finished(&r.spine, seq) && kv_status(&r.spine).parking_jobs == 0
    })
    .await;
    drain(&r.spine);
}

// ---------------------------------------------------------------------------
// Greedy rig (A2): the CPU reference backend over the shared pool
// ---------------------------------------------------------------------------

fn prompt17() -> Vec<u32> {
    vec![
        5, 17, 42, 3, 88, 61, 9, 23, 70, 11, 36, 2, 44, 19, 80, 7, 31,
    ]
}

const GREEDY_TOKENS: usize = 4;

/// Runs `prompt17()` greedily on a fresh runtime, with persistence through
/// the daemon wiring when `dir` is given. Returns the generated tokens and the
/// spine (for status and the drain).
async fn greedy_run(dir: Option<&Path>) -> (Vec<u32>, AienRuntimeSpine) {
    let config = greedy_model();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, mut backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: TOTAL_BLOCKS,
        },
    )
    .expect("runtime");
    let host_engine =
        HostEngineIdentity::reference(ReferenceCpuBackend::new().name()).expect("engine identity");
    if let Some(dir) = dir {
        enable_persistent_kv(
            &mut spine,
            &settings(dir),
            &ModelKvBinding {
                model: &config,
                arith_id: ReferenceCpuBackend::new().name(),
                model_sha256: [3u8; 32],
                model_digest_kind: "sha256",
                engine: host_engine.identity(),
            },
            None,
        )
        .expect("enable")
        .expect("on");
        wait_store_open(&spine);
    }
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));
    let seq = spine
        .submit_work(
            Arc::from(prompt17()),
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: GREEDY_TOKENS,
                stop_token_ids: vec![],
            },
            1,
            Some(sink_id),
        )
        .expect("submit");
    let t0 = Instant::now();
    loop {
        let parks_done = spine
            .kv_residency_status()
            .is_none_or(|s| s.parking_jobs == 0);
        if finished(&spine, seq) && parks_done {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "greedy run stuck");
        spine.step(&mut backend).await.expect("step");
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut tokens = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let CompletionEvent::Token { token, .. } = ev {
            tokens.push(token);
        }
    }
    assert_eq!(tokens.len(), GREEDY_TOKENS, "every greedy token reported");
    (tokens, spine)
}

fn write_tokens(path: &Path, tokens: &[u32]) {
    let text: Vec<String> = tokens.iter().map(u32::to_string).collect();
    std::fs::write(path, text.join(",")).expect("write tokens");
}

fn read_tokens(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .expect("tokens file")
        .split(',')
        .map(|t| t.parse().expect("token"))
        .collect()
}

async fn child_greedy(dir: &Path, out: &Path, expect_restore: bool) {
    let (tokens, spine) = greedy_run(Some(dir)).await;
    let s = kv_status(&spine);
    if expect_restore {
        assert_eq!(s.stats.restore_tokens_total, 16, "restored prefix: {s:?}");
        assert_eq!(s.stats.restore_committed_total, 1, "{s:?}");
    } else {
        assert_eq!(s.stats.restore_jobs_total, 0, "cold run restored: {s:?}");
        assert!(s.stats.park_published_total >= 2, "{s:?}");
    }
    write_tokens(out, &tokens);
    drain(&spine);
}

// ---------------------------------------------------------------------------
// The child entry point and the parent's spawner
// ---------------------------------------------------------------------------

/// Hidden helper: the child. Does nothing unless the parent set `KV_RS_ROLE`.
#[tokio::test]
#[ignore]
async fn restart_child() {
    let (Ok(role), Ok(dir)) = (std::env::var("KV_RS_ROLE"), std::env::var("KV_RS_DIR")) else {
        return;
    };
    let dir = PathBuf::from(dir);
    match role.as_str() {
        "paint-park" => child_paint_park(&dir).await,
        "paint-restore" => child_paint_restore(&dir).await,
        "greedy-park" => child_greedy(&dir, &dir.join("cold.txt"), false).await,
        "greedy-restore" => child_greedy(&dir, &dir.join("restored.txt"), true).await,
        other => panic!("unknown role {other}"),
    }
}

/// Runs one child process; returns whether it succeeded.
fn run_child(role: &str, dir: &Path, env: &[(&str, &str)]) -> bool {
    let mut cmd = std::process::Command::new(std::env::current_exe().expect("current_exe"));
    cmd.args([
        "--exact",
        "restart_child",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]);
    // The child reads only what the parent hands it.
    for (k, _) in std::env::vars() {
        if k.starts_with("AIEN_KV_") || k.starts_with("KV_RS_") {
            cmd.env_remove(k);
        }
    }
    cmd.env("KV_RS_ROLE", role).env("KV_RS_DIR", dir);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn child");
    if !out.status.success() {
        eprintln!(
            "child {role} {env:?} failed: {:?}\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .chain(String::from_utf8_lossy(&out.stderr).lines())
                .filter(|l| l.contains("panicked")
                    || l.contains("assert")
                    || l.contains("left")
                    || l.contains("right")
                    || l.contains("differs"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    out.status.success()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A1 (process restart row of ADR 11): child 1 parks a painted turn on a host
/// filesystem store and exits through the drain; child 2, a new process,
/// recovers the store from disk and restores the prefix into its own pool with
/// the bytes child 1 parked. Negative control: a child 2 that expects a
/// different byte pattern must fail, so the comparison is not vacuous.
#[test]
fn process_restart_rediscovers_and_restores_identical_bytes() {
    let dir = private_tempdir();
    assert!(
        run_child("paint-park", dir.path(), &[]),
        "child 1 (park) failed"
    );
    assert!(
        run_child("paint-restore", dir.path(), &[]),
        "child 2 (restore after restart) failed"
    );
    assert!(
        !run_child("paint-restore", dir.path(), &[("KV_RS_SALT", "1")]),
        "negative control: a child expecting other bytes passed"
    );
}

/// A2 (ADR 6.5): a prompt run cold records its greedy tokens and parks; after
/// a process restart the same prompt restores its 16-token prefix, prefills the
/// remainder on the same CPU backend and must emit the same greedy tokens.
/// The in-process run without persistence is the uninterrupted baseline.
#[tokio::test]
async fn greedy_continuation_after_restart_matches_uninterrupted_baseline() {
    let dir = private_tempdir();
    assert!(
        run_child("greedy-park", dir.path(), &[]),
        "cold child failed"
    );
    assert!(
        run_child("greedy-restore", dir.path(), &[]),
        "restored child failed"
    );
    let cold = read_tokens(&dir.path().join("cold.txt"));
    let restored = read_tokens(&dir.path().join("restored.txt"));
    let (baseline, _) = greedy_run(None).await;
    assert_eq!(
        cold, baseline,
        "cold child differs from the in-process baseline"
    );
    assert_eq!(
        restored, baseline,
        "greedy continuation after restart differs from the uninterrupted baseline"
    );
}

/// A3 end to end: objects parked under one weights digest are rediscovered
/// and counted after a restart, but a runtime with another digest never
/// serves them (plain prefill). Positive control: the right digest restores.
#[test]
fn wrong_model_digest_namespace_is_never_served_after_restart() {
    let dir = private_tempdir();
    assert!(run_child("paint-park", dir.path(), &[("KV_RS_SHA", "7")]));
    assert!(
        run_child(
            "paint-restore",
            dir.path(),
            &[("KV_RS_SHA", "8"), ("KV_RS_EXPECT", "miss")]
        ),
        "a foreign weights digest was served or not counted"
    );
    assert!(
        run_child("paint-restore", dir.path(), &[("KV_RS_SHA", "7")]),
        "positive control: the parking digest must restore"
    );
}

/// Review finding 1 end to end: same weights digest, different `rms_norm_eps`
/// in the resolved `ModelConfig`: a different compute-config digest, so a
/// different namespace, never served after a restart.
#[test]
fn different_rms_norm_eps_namespace_is_never_served_after_restart() {
    let dir = private_tempdir();
    assert!(run_child(
        "paint-park",
        dir.path(),
        &[("KV_RS_EPS", "0.00001")]
    ));
    assert!(
        run_child(
            "paint-restore",
            dir.path(),
            &[("KV_RS_EPS", "0.000001"), ("KV_RS_EXPECT", "miss")]
        ),
        "a different rms_norm_eps was served"
    );
    assert!(
        run_child("paint-restore", dir.path(), &[("KV_RS_EPS", "0.00001")]),
        "positive control: the parking eps must restore"
    );
}

/// sc#504 (ADR 1.2.2): park under one engine identity, restart with exactly
/// one engine input changed: the parked objects are rediscovered and counted,
/// never served (plain prefill from position zero). Positive control: the
/// parking identity restores.
fn engine_change_is_never_served_after_restart(var: &str, parked: &str, changed: &str) {
    let dir = private_tempdir();
    assert!(run_child("paint-park", dir.path(), &[(var, parked)]));
    assert!(
        run_child(
            "paint-restore",
            dir.path(),
            &[(var, changed), ("KV_RS_EXPECT", "miss")]
        ),
        "{var}={changed} served objects parked with {var}={parked}"
    );
    assert!(
        run_child("paint-restore", dir.path(), &[(var, parked)]),
        "positive control: {var}={parked} must restore"
    );
}

#[test]
fn weight_format_change_is_never_served_after_restart() {
    engine_change_is_never_served_after_restart("KV_RS_WF", "bf16", "fp8-experts");
}

#[test]
fn engine_build_change_is_never_served_after_restart() {
    engine_change_is_never_served_after_restart("KV_RS_BUILD", "omega@aaaa", "omega@bbbb");
}

#[test]
fn rounding_class_change_is_never_served_after_restart() {
    engine_change_is_never_served_after_restart("KV_RS_ROUND", "class-a", "class-b");
}

#[test]
fn whole_prompt_prefill_change_is_never_served_after_restart() {
    engine_change_is_never_served_after_restart("KV_RS_WHOLE_PROMPT", "0", "1");
}

/// A8 (METRICS_AND_SETTINGS): the status report carries the pool allocation
/// in bytes, which never changes when blocks are freed, separately from the
/// block counts, and the persistent-KV fields when persistence is on.
#[tokio::test]
async fn metrics_report_allocated_bytes_separately_from_free_blocks() {
    let dir = private_tempdir();
    let mut r = paint_rig(dir.path(), 7, 1e-5);
    wait_store_open(&r.spine);
    let pool_bytes = r.kv.read().pool().expect("pool").layout().total_bytes as u64;

    let before = r.spine.status_report();
    assert_eq!(before.kv_pool_allocated_bytes, pool_bytes);
    assert_eq!(before.kv_pool_total_blocks, TOTAL_BLOCKS);
    assert_eq!(before.kv_pool_free_blocks, TOTAL_BLOCKS);
    assert_eq!(before.kv_pool_used_blocks, 0);
    let p = before.kv_persist.clone().expect("persistent KV fields");
    assert_eq!(p.kv_store_state, "open");
    assert!(p.kv_store_quota_bytes > 0);

    let seq = r.spine.submit_request(request(&prompt14(), 3));
    r.spine.step(&mut r.backend).await.expect("step");
    let during = r.spine.status_report();
    assert!(during.kv_pool_free_blocks < TOTAL_BLOCKS, "blocks in use");
    assert_eq!(
        during.kv_pool_used_blocks + during.kv_pool_free_blocks,
        TOTAL_BLOCKS
    );
    assert_eq!(
        during.kv_pool_allocated_bytes, pool_bytes,
        "allocation is constant"
    );

    paint_step_until(&mut r, "turn finished and parked", |r| {
        finished(&r.spine, seq) && kv_status(&r.spine).parking_jobs == 0
    })
    .await;
    let after = r.spine.status_report();
    assert_eq!(after.kv_pool_free_blocks, TOTAL_BLOCKS, "blocks freed");
    assert_eq!(
        after.kv_pool_allocated_bytes, pool_bytes,
        "freeing blocks does not release RAM"
    );
    let p = after.kv_persist.expect("persistent KV fields");
    assert_eq!(p.kv_park_jobs_total, 2);
    assert!(p.kv_store_persisted_objects >= 2);
    assert!(p.kv_store_persisted_bytes > 0);
    drain(&r.spine);
}

/// `AIEN_KV_STORE_DIR` unset: no store, no hook, no status fields, nothing on
/// disk. The runtime behaves exactly as before.
#[tokio::test]
async fn persistence_off_when_store_dir_unset_changes_nothing() {
    let kv = paint_pool();
    let mut spine = AienRuntimeSpine::new(64, scheduler(), kv.clone());
    let off = KvPersistSettings::from_lookup(|_| None).expect("defaults");
    let model = paint_model(1e-5);
    let got = enable_persistent_kv(
        &mut spine,
        &off,
        &ModelKvBinding {
            model: &model,
            arith_id: "x",
            model_sha256: [7; 32],
            model_digest_kind: "sha256",
            engine: PaintEngine::from_env().identity(),
        },
        None,
    )
    .expect("off is not an error");
    assert!(got.is_none());
    assert!(spine.kv_residency().is_none());
    assert!(spine.kv_drain_handle().is_none());
    assert!(spine.status_report().kv_persist.is_none());

    let rec = Arc::new(Mutex::new(Rec::default()));
    let mut backend = PaintBackend { rec: rec.clone() };
    let seq = spine.submit_request(request(&prompt14(), 2));
    for _ in 0..5 {
        spine.step(&mut backend).await.expect("step");
    }
    assert!(finished(&spine, seq));
    assert_eq!(kv.read().free_block_count(), TOTAL_BLOCKS);
    assert_eq!(kv.read().holder_block_count(), 0, "nothing parked");
}

/// `AIEN_KV_STORE_DIR` set but `AIEN_KV_STORE_KEY` missing or malformed: fail
/// closed (ADR 0003 section 20). No store is opened, nothing is written, no
/// park and no restore; requests run as plain prefill, and the status says why.
#[tokio::test]
async fn persistence_refused_without_a_valid_store_key_runs_plain_and_says_why() {
    let short = &TEST_STORE_KEY[..62];
    let long = format!("{TEST_STORE_KEY}00");
    for (label, key, reason) in [
        ("missing", None, "key_missing"),
        ("malformed", Some("not-a-key"), "key_malformed"),
        ("short", Some(short), "key_malformed"),
        ("long", Some(long.as_str()), "key_malformed"),
    ] {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "aien-kv-c7-nokey-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_string_lossy().to_string();
        let s = KvPersistSettings::from_lookup(|k| match k {
            "AIEN_KV_STORE_DIR" => Some(d.clone()),
            "AIEN_KV_CHUNK_BLOCKS" => Some(CHUNK_BLOCKS.to_string()),
            "AIEN_KV_STORE_KEY" => key.map(str::to_string),
            _ => None,
        })
        .expect("a bad key is a refusal, not a start failure");
        let kv = paint_pool();
        let mut spine = AienRuntimeSpine::new(64, scheduler(), kv.clone());
        let model = paint_model(1e-5);
        let line = enable_persistent_kv(
            &mut spine,
            &s,
            &ModelKvBinding {
                model: &model,
                arith_id: "c7-paint-backend",
                model_sha256: [7; 32],
                model_digest_kind: "sha256",
                engine: PaintEngine::from_env().identity(),
            },
            None,
        )
        .expect("refusal is not an error")
        .expect("the refusal is announced");
        assert!(
            line.contains("REFUSED") && line.contains(reason),
            "{label}: {line}"
        );
        assert!(spine.kv_residency().is_none(), "{label}: no store");
        assert!(spine.kv_drain_handle().is_none());
        let p = spine
            .status_report()
            .kv_persist
            .expect("refusal is visible in the status");
        assert_eq!(p.kv_store_state, "refused", "{label}");
        assert_eq!(p.kv_store_refused_reason, reason, "{label}");
        assert!(p.kv_store_key_refused_total >= 1, "{label}");
        assert!(
            std::fs::read_dir(&dir).unwrap().next().is_none(),
            "{label}: nothing written under the store dir"
        );
        let _ = std::fs::remove_dir_all(&dir);

        let rec = Arc::new(Mutex::new(Rec::default()));
        let mut backend = PaintBackend { rec: rec.clone() };
        let seq = spine.submit_request(request(&prompt14(), 2));
        for _ in 0..5 {
            spine.step(&mut backend).await.expect("step");
        }
        assert!(finished(&spine, seq), "{label}: plain prefill still serves");
        assert_eq!(kv.read().holder_block_count(), 0, "{label}: nothing parked");
    }
}

/// The settings Debug text never carries the store key.
#[test]
fn persistent_kv_settings_debug_never_shows_the_store_key() {
    let s = settings(Path::new("/tmp/kv-c7-debug"));
    let shown = format!("{s:?}");
    assert!(!shown.contains(&TEST_STORE_KEY[..16]), "{shown}");
}

/// A bad persistent-KV setting stops the start, and the message names the
/// setting.
#[test]
fn bad_persistent_kv_settings_fail_startup_with_the_setting_name() {
    for name in [
        "AIEN_KV_CHUNK_BLOCKS",
        "AIEN_KV_RESTORE_MAX_BLOCKS",
        "AIEN_KV_RESTORE_RETRY_WINDOW_MS",
        "AIEN_KV_STORE_QUOTA_BYTES",
        "AIEN_KV_STORE_MAX_OBJECTS",
        "AIEN_KV_STORE_RECOVERY_SLICE_MS",
        "AIEN_KV_STORE_MAX_RESTORE_OPS",
        "AIEN_KV_STORE_MAX_PARK_OPS",
        "AIEN_KV_STORE_MAX_INFLIGHT_BYTES",
        "AIEN_KV_STORE_MAX_WAITERS_PER_JOB",
        "AIEN_KV_STORE_DRAIN_MS",
        "AIEN_KV_PARK",
        "AIEN_KV_RESTORE",
    ] {
        let err = KvPersistSettings::from_lookup(|k| match k {
            "AIEN_KV_STORE_DIR" => Some("/tmp/kv-c7".to_string()),
            k if k == name => Some("bogus".to_string()),
            _ => None,
        })
        .expect_err(name);
        assert!(err.contains(name), "{name}: error does not name it: {err}");
    }
    let ok = KvPersistSettings::from_lookup(|k| {
        (k == "AIEN_KV_RESTORE_RETRY_WINDOW_MS").then(|| "1500".to_string())
    })
    .expect("valid window");
    assert_eq!(ok.restore_retry_window_ms, 1500);
    let default = KvPersistSettings::from_lookup(|_| None).expect("defaults");
    assert_eq!(default.restore_retry_window_ms, 60_000);
}

/// The `AIEN_STEP_LOG=1` line gains the persistent-KV counters under their
/// METRICS_AND_SETTINGS names.
#[test]
fn step_log_line_carries_kv_counters() {
    let mut s = KvResidencyStatus::default();
    s.stats.park_jobs_total = 3;
    s.stats.park_failed_total = 1;
    s.stats.park_indeterminate_total = 2;
    s.stats.restore_jobs_total = 4;
    s.stats.restore_fallback_prefill_total = 5;
    s.stats.store_stale_completions_total = 6;
    s.holder_blocks = 7;
    s.restore_inflight_blocks = 8;
    let line = aien_runtime::server::kv_step_log_fields(&s);
    for want in [
        "kv_park_jobs_total=3",
        "kv_park_failed_total=1",
        "kv_park_indeterminate_total=2",
        "kv_restore_jobs_total=4",
        "kv_restore_fallback_prefill_total=5",
        "kv_store_stale_completions_total=6",
        "kv_pool_holder_blocks=7",
        "kv_restore_inflight_blocks=8",
    ] {
        assert!(line.contains(want), "missing {want}: {line}");
    }
}

/// ADR 11: every row of the guarantee matrix carries one of PLANNED,
/// TESTED(test name), NOT CLAIMED or NOT_RUN in both adapter columns, and each
/// TESTED row names a test that exists in this workspace.
#[test]
fn guarantee_matrix_rows_are_labelled() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let adr = std::fs::read_to_string(
        root.join("docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md"),
    )
    .expect("ADR 0003");
    let section = adr
        .split("## 11. Guarantee matrix")
        .nth(1)
        .expect("section 11")
        .split("\n## ")
        .next()
        .expect("section body");
    let rows: Vec<Vec<String>> = section
        .lines()
        .filter(|l| l.starts_with('|'))
        .skip(2) // header and separator
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect();
    assert_eq!(rows.len(), 6, "matrix rows");

    let mut sources = String::new();
    collect_rs(&root.join("crates"), &mut sources);
    let mut tested = Vec::new();
    for row in &rows {
        for cell in &row[1..3] {
            let ok = ["PLANNED", "TESTED(", "NOT CLAIMED", "NOT_RUN"]
                .iter()
                .any(|l| cell.starts_with(l));
            assert!(ok, "row {:?}: unlabelled cell {cell:?}", row[0]);
            if let Some(rest) = cell.strip_prefix("TESTED(") {
                let name = rest.split(')').next().expect("receipt");
                for test in name.split(',').map(str::trim) {
                    assert!(
                        sources.contains(&format!("fn {test}(")),
                        "row {:?} names test {test} which does not exist",
                        row[0]
                    );
                    tested.push(test.to_string());
                }
            }
        }
    }
    // The C7 rows this cut proves.
    for must in [
        "process_restart_rediscovers_and_restores_identical_bytes",
        "greedy_continuation_after_restart_matches_uninterrupted_baseline",
        "store_then_recreate_adapter_serves_same_bytes",
    ] {
        assert!(
            tested.iter().any(|t| t == must),
            "{must} not labelled TESTED"
        );
    }
}

fn collect_rs(dir: &Path, out: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            if let Ok(s) = std::fs::read_to_string(&p) {
                out.push_str(&s);
            }
        }
    }
}
