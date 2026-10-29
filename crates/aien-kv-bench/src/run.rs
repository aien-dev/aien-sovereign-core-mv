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

//! The measuring half of `bench_persistent_kv`: drives the real runtime spine,
//! the real scheduler and the native transformer backend on the host CPU
//! reference arithmetic (`ReferenceCpuBackend`, reference test weights), with
//! the C4 host filesystem store through the daemon wiring
//! (`shared_kv::enable_persistent_kv`) for the restore mode. Host only: no
//! GPU, no CUDA, no root.

use crate::report::{Row, HOSTED};
use aien_abi_core::{ModelConfig, SamplingParams};
use aien_inference_abi::backend::TensorBackend;
use aien_inference_abi::{NativeTransformerBackend, ReferenceCpuBackend, TransformerWeights};
use aien_runtime::shared_kv::{
    build_shared_kv_runtime, enable_persistent_kv, KvPersistSettings, ModelKvBinding,
    SharedKvSizing, KV_CHUNK_BLOCKS_ENV, KV_RESTORE_MAX_BLOCKS_ENV, KV_STORE_DIR_ENV,
};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{ChannelCompletionSink, CompletionEvent, SchedulerConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The vocabulary of the reference test model; every token id is below it.
pub const VOCAB: u32 = 97;

/// A stop id the model can never produce.
const NO_EOS: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    ResidentOnly,
    ColdPrefill,
    RestorePageCache,
    RestoreDevice,
}

impl Mode {
    pub const ALL: [Mode; 4] = [
        Mode::ResidentOnly,
        Mode::ColdPrefill,
        Mode::RestorePageCache,
        Mode::RestoreDevice,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Mode::ResidentOnly => "resident-only",
            Mode::ColdPrefill => "cold-prefill",
            Mode::RestorePageCache => "restore-from-page-cache",
            Mode::RestoreDevice => "restore-from-device",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.label() == s)
    }

    pub fn cache_state(self) -> &'static str {
        match self {
            Mode::ResidentOnly | Mode::ColdPrefill => "none",
            Mode::RestorePageCache => "page_cache",
            Mode::RestoreDevice => "device",
        }
    }

    fn uses_store(self) -> bool {
        matches!(self, Mode::RestorePageCache | Mode::RestoreDevice)
    }
}

/// Why `restore-from-device` is never measured by this binary.
pub const DEVICE_NOT_RUN_REASON: &str = "a device read needs the page cache dropped (root: /proc/sys/vm/drop_caches) and verified empty with mincore (ADR 0003 section 15.1); this harness never uses sudo, and O_DIRECT reads are cut X1/X2";

#[derive(Debug, Clone)]
pub struct RunSpec {
    pub modes: Vec<Mode>,
    pub concurrency: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub rounds: u32,
    pub turn1_max_tokens: usize,
    pub turn2_suffix_tokens: usize,
    pub turn2_max_tokens: usize,
    pub block_size: usize,
    pub chunk_blocks: u32,
    pub store_root: PathBuf,
}

/// The reference model: the C7 greedy test model (kv_restart_test.rs) with a
/// 16-token block and room for a 2048-token conversation.
pub fn model_config(block_size: usize, max_sequence_length: usize) -> ModelConfig {
    ModelConfig {
        model_id: "pkv-bench-reference".to_string(),
        max_sequence_length,
        block_size,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        num_kv_heads: 2,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: VOCAB as usize,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        // Never emitted (above the vocabulary): every request runs to its
        // max_tokens, so rows compare equal work (an empty list would use
        // the legacy stop ids, which the reference weights do emit).
        eos_token_ids: vec![NO_EOS],
        qk_norm: false,
    }
}

pub fn model_label() -> String {
    let c = model_config(16, 0);
    format!(
        "reference test weights (TransformerWeights::reference_test_weights): {} layers, hidden {}, {} heads, {} kv heads, head_dim {}, vocab {}; synthetic, not a real model",
        c.num_layers, c.hidden_dim, c.num_heads, c.num_kv_heads, c.head_dim, c.vocab_size
    )
}

/// Deterministic prompt tokens, distinct per (round, conversation, salt).
fn tokens(seed: u64, len: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % u64::from(VOCAB)) as u32
        })
        .collect()
}

/// Peak RSS reset (Linux: writing 5 to clear_refs resets VmHWM for this
/// process; no privilege needed for our own process).
pub fn reset_peak_rss() -> Result<(), String> {
    std::fs::write("/proc/self/clear_refs", "5").map_err(|e| format!("clear_refs: {e}"))
}

pub fn peak_rss_bytes() -> Result<u64, String> {
    let status =
        std::fs::read_to_string("/proc/self/status").map_err(|e| format!("status: {e}"))?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| "no VmHWM in /proc/self/status".to_string())
}

struct Peaks {
    holder_blocks: usize,
}

struct Done {
    latency_us: Vec<u64>,
    ttft_us: Vec<u64>,
    /// Generated tokens per prompt, in submission order (empty when failed).
    tokens: Vec<Vec<u32>>,
    failed: u64,
    decode_tokens: u64,
    decode_window: Duration,
}

struct Bench {
    spine: AienRuntimeSpine,
    backend: NativeTransformerBackend,
    rx: tokio::sync::mpsc::UnboundedReceiver<CompletionEvent>,
    sink: aien_scheduler::CompletionSinkId,
    block_stride: u64,
    pool_bytes: u64,
    /// `WorkerConfig.dispatch_depth` of the store, 0 without a store.
    queue_depth: u32,
    peaks: Peaks,
    deadline: Instant,
}

impl Bench {
    fn sample(&mut self) {
        if let Some(s) = self.spine.kv_residency_status() {
            self.peaks.holder_blocks = self.peaks.holder_blocks.max(s.holder_blocks);
        }
    }

    async fn step(&mut self) -> Result<(), String> {
        if Instant::now() > self.deadline {
            return Err("row exceeded its time budget".into());
        }
        self.spine.step(&mut self.backend).await?;
        self.sample();
        Ok(())
    }

    /// Steps until no park job is in flight (outside any measured window).
    async fn wait_parks(&mut self) -> Result<(), String> {
        while self
            .spine
            .kv_residency_status()
            .is_some_and(|s| s.parking_jobs > 0)
        {
            self.step().await?;
            std::thread::sleep(Duration::from_micros(200));
        }
        Ok(())
    }

    async fn wait_store_open(&mut self) -> Result<(), String> {
        use aien_kv_store::StoreState;
        loop {
            let state = self.spine.kv_residency().map(|r| r.store_capacity().state);
            match state {
                None | Some(StoreState::Open) => return Ok(()),
                Some(StoreState::Recovering) => {
                    self.step().await?;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Some(other) => return Err(format!("store did not open: {other:?}")),
            }
        }
    }

    /// Submits every prompt, steps until each has finished, and returns the
    /// timings. Token times are taken when the channel is drained after a step.
    async fn run_batch(&mut self, prompts: &[Vec<u32>], max_tokens: usize) -> Result<Done, String> {
        let mut submitted: HashMap<u64, Instant> = HashMap::new();
        let mut order = Vec::new();
        for p in prompts {
            let at = Instant::now();
            let seq = self.spine.submit_work(
                Arc::from(p.clone()),
                SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens,
                    stop_token_ids: vec![],
                },
                1,
                Some(self.sink),
            )?;
            submitted.insert(seq, at);
            order.push(seq);
        }
        let mut first: HashMap<u64, Instant> = HashMap::new();
        let mut finished: HashMap<u64, Instant> = HashMap::new();
        let mut tokens: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut failed = 0u64;
        while finished.len() + (failed as usize) < order.len() {
            self.step().await?;
            let now = Instant::now();
            while let Ok(ev) = self.rx.try_recv() {
                match ev {
                    CompletionEvent::Token { seq_id, token } => {
                        let id = seq_id.to_u64();
                        first.entry(id).or_insert(now);
                        tokens.entry(id).or_default().push(token);
                    }
                    CompletionEvent::Finished { seq_id, .. } => {
                        finished.insert(seq_id.to_u64(), now);
                    }
                    CompletionEvent::Error { seq_id, message } => {
                        eprintln!("request {} failed: {message}", seq_id.to_u64());
                        failed += 1;
                    }
                }
            }
        }
        let mut done = Done {
            latency_us: Vec::new(),
            ttft_us: Vec::new(),
            tokens: Vec::new(),
            failed,
            decode_tokens: 0,
            decode_window: Duration::ZERO,
        };
        let mut window_start: Option<Instant> = None;
        let mut window_end: Option<Instant> = None;
        for id in order {
            let (Some(end), Some(t1)) = (finished.get(&id), first.get(&id)) else {
                done.tokens.push(Vec::new());
                continue;
            };
            let at = submitted[&id];
            let toks = tokens.remove(&id).unwrap_or_default();
            if toks.len() != max_tokens {
                return Err(format!(
                    "request {id} produced {} tokens, expected {max_tokens}",
                    toks.len()
                ));
            }
            done.latency_us
                .push(end.duration_since(at).as_micros() as u64);
            done.ttft_us.push(t1.duration_since(at).as_micros() as u64);
            done.decode_tokens += toks.len().saturating_sub(1) as u64;
            window_start = Some(window_start.map_or(*t1, |w| w.min(*t1)));
            window_end = Some(window_end.map_or(*end, |w| w.max(*end)));
            done.tokens.push(toks);
        }
        if let (Some(s), Some(e)) = (window_start, window_end) {
            done.decode_window = e.duration_since(s);
        }
        Ok(done)
    }
}

/// Per-class store bounds (ADR 7.1) large enough that every chunk of every
/// conversation finishing in the same step is admitted: `c` times the chunks of
/// the longest turn plus one tail, never below the defaults (48 restore, 16
/// park). With the defaults, 8 conversations of 512 tokens finishing together
/// submit 24 parks and 8 are refused `QueueFull` (by design: not parked this
/// turn), so their next turn prefills and the row could not be called a restore.
pub fn store_op_bounds(spec: &RunSpec, c: u32, turn_tokens: usize) -> (u32, u32) {
    let defaults = aien_kv_store::worker::WorkerConfig::default();
    let chunk_tokens = spec.chunk_blocks as usize * spec.block_size;
    let per_conv = turn_tokens.div_ceil(chunk_tokens.max(1)) as u32 + 1;
    let need = c.saturating_mul(per_conv);
    (
        need.max(defaults.max_restore_ops),
        need.max(defaults.max_park_ops),
    )
}

fn build(spec: &RunSpec, mode: Mode, c: u32, prompt: u32) -> Result<Bench, String> {
    let turn2_len =
        prompt as usize + spec.turn1_max_tokens + spec.turn2_suffix_tokens + spec.turn2_max_tokens;
    let blocks_per_conv = turn2_len.div_ceil(spec.block_size) + 1;
    let total_blocks = c as usize * blocks_per_conv * 3 + 64;
    let config = model_config(spec.block_size, turn2_len + spec.block_size);
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        SchedulerConfig {
            max_batch_size: 16,
            max_batch_tokens: 4096,
            max_prefill_tokens: 4096,
            prefill_chunk_size: 4096,
            chunk_prefill: true,
            watermark_blocks: 0,
        },
        SharedKvSizing {
            arena_capacity: 4 * c as usize + 16,
            total_blocks,
        },
    )?;
    let (block_stride, pool_bytes) = {
        let guard = spine.kv_manager.read();
        let layout = guard
            .pool()
            .ok_or("runtime KV manager has no pool")?
            .layout();
        (layout.block_stride_bytes as u64, layout.total_bytes as u64)
    };
    let mut queue_depth = 0;
    if mode.uses_store() {
        let dir = spec
            .store_root
            .join(format!("{}-c{c}-p{prompt}", mode.label()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        // The host store refuses a root group or others can reach (ADR 0003
        // section 19), so the run directory is created 0700.
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let dir_s = dir.to_string_lossy().to_string();
        let chunk = spec.chunk_blocks.to_string();
        let restore_max = total_blocks.to_string();
        let (restore_ops, park_ops) = store_op_bounds(spec, c, turn2_len);
        let (restore_ops, park_ops) = (restore_ops.to_string(), park_ops.to_string());
        let mut settings = KvPersistSettings::from_lookup(|k| match k {
            KV_STORE_DIR_ENV => Some(dir_s.clone()),
            KV_CHUNK_BLOCKS_ENV => Some(chunk.clone()),
            KV_RESTORE_MAX_BLOCKS_ENV => Some(restore_max.clone()),
            "AIEN_KV_STORE_MAX_RESTORE_OPS" => Some(restore_ops.clone()),
            "AIEN_KV_STORE_MAX_PARK_OPS" => Some(park_ops.clone()),
            _ => None,
        })?;
        // A fresh random store key per run: objects are sealed as in the daemon
        // (ADR 0003 section 20), so restore timings include decryption.
        settings.store_key = Some(Ok(std::sync::Arc::new(
            aien_kv_store::seal::SealKey::generate()
                .map_err(|e| format!("bench store key: {e:?}"))?,
        )));
        queue_depth = settings.worker.dispatch_depth;
        let host_engine = aien_runtime::compute_config::HostEngineIdentity::reference(
            ReferenceCpuBackend::new().name(),
        )
        .map_err(|e| format!("engine identity: {e}"))?;
        enable_persistent_kv(
            &mut spine,
            &settings,
            &ModelKvBinding {
                model: &config,
                arith_id: ReferenceCpuBackend::new().name(),
                // Synthetic weights have no file; a fixed digest names them.
                model_sha256: [0xB9; 32],
                model_digest_kind: "sha256",
                engine: host_engine.identity(),
            },
            None,
        )?
        .ok_or("persistence did not switch on")?;
    }
    let (sink, rx) = ChannelCompletionSink::channel();
    let sink = spine.register_completion_sink(Arc::new(sink));
    Ok(Bench {
        spine,
        backend,
        rx,
        sink,
        block_stride,
        pool_bytes,
        queue_depth,
        peaks: Peaks { holder_blocks: 0 },
        deadline: Instant::now() + Duration::from_secs(300),
    })
}

/// Measures one (mode, concurrency, prompt length) row.
pub async fn run_row(spec: &RunSpec, mode: Mode, c: u32, prompt: u32) -> Result<Row, String> {
    if mode == Mode::RestoreDevice {
        return Err("restore-from-device is NOT_RUN by design".into());
    }
    let workload = if mode == Mode::ResidentOnly {
        "single_turn"
    } else {
        "second_turn"
    };
    let mut b = build(spec, mode, c, prompt)?;
    b.wait_store_open().await?;
    reset_peak_rss()?;
    let mut row = Row::new_ok(
        mode.label(),
        HOSTED,
        mode.cache_state(),
        c,
        prompt,
        workload,
    );
    let mut failed = 0u64;
    let mut decode_tokens = 0u64;
    let mut decode_window = Duration::ZERO;
    let mut restored = 0u64;
    for round in 0..spec.rounds {
        let prompts: Vec<Vec<u32>> = (0..c)
            .map(|i| {
                tokens(
                    u64::from(round) << 32 | u64::from(i) << 8 | u64::from(prompt % 251),
                    prompt as usize,
                )
            })
            .collect();
        let measured = if mode == Mode::ResidentOnly {
            b.run_batch(&prompts, spec.turn2_max_tokens).await?
        } else {
            let turn1 = b.run_batch(&prompts, spec.turn1_max_tokens).await?;
            if turn1.failed > 0 {
                return Err(format!("{} turn-1 requests failed", turn1.failed));
            }
            b.wait_parks().await?;
            let turn2: Vec<Vec<u32>> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut t = p.clone();
                    t.extend(&turn1.tokens[i]);
                    t.extend(tokens(
                        0xC9 << 40 | u64::from(round) << 16 | i as u64,
                        spec.turn2_suffix_tokens,
                    ));
                    t
                })
                .collect();
            let before = b.spine.kv_residency_status().map(|s| s.stats);
            let done = b.run_batch(&turn2, spec.turn2_max_tokens).await?;
            let after = b.spine.kv_residency_status().map(|s| s.stats);
            match (mode, before, after) {
                (Mode::RestorePageCache, Some(before), Some(after)) => {
                    let committed = after.restore_committed_total - before.restore_committed_total;
                    if committed != u64::from(c) {
                        return Err(format!(
                            "round {round}: {committed} of {c} turn-2 requests committed a restore (fallbacks {}); refusing to label the row restore; before {:?} after {:?}",
                            after.restore_fallback_prefill_total
                                - before.restore_fallback_prefill_total,
                            before,
                            after
                        ));
                    }
                    restored += after.restore_tokens_total - before.restore_tokens_total;
                }
                (Mode::ColdPrefill, None, None) => {}
                other => {
                    return Err(format!(
                        "residency state does not match the mode: {other:?}"
                    ))
                }
            }
            done
        };
        failed += measured.failed;
        decode_tokens += measured.decode_tokens;
        decode_window += measured.decode_window;
        row.latency_samples_us.extend(&measured.latency_us);
        row.ttft_samples_us.extend(&measured.ttft_us);
        b.wait_parks().await?;
    }
    row.completed_requests = Some(row.latency_samples_us.len() as u64);
    row.failed_requests = Some(failed);
    row.decode_tokens_per_s = Some(if decode_window.is_zero() {
        0.0
    } else {
        decode_tokens as f64 / decode_window.as_secs_f64()
    });
    row.peak_allocated_bytes = Some(b.pool_bytes);
    row.peak_holder_bytes = Some(b.peaks.holder_blocks as u64 * b.block_stride);
    row.peak_staging_bytes = Some(if mode.uses_store() { b.block_stride } else { 0 });
    row.peak_rss_bytes = Some(peak_rss_bytes()?);
    row.queue_depth = Some(b.queue_depth);
    row.restored_tokens = Some(restored);
    row.fill_percentiles();
    if let Some(drain) = b.spine.kv_drain_handle() {
        let report = drain.drain();
        if !report.unfinished.is_empty() {
            return Err(format!("store drain left {} jobs", report.unfinished.len()));
        }
    }
    row.check().map_err(|e| e.to_string())?;
    Ok(row)
}
