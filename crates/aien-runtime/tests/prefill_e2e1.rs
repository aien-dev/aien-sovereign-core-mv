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

//! PREFILL-E2E-1: the E2E-0 end-to-end gate at N = 1, 2, 8, 32, 128 and 500
//! branches from ONE shared root prompt (TinyLlama, CPU, strict checkpoint).
//!
//! `#[ignore]`d for normal CI. Run with `--ignored`; it FAILS (never skips)
//! when `AIEN_E2E_CHECKPOINT` or `AIEN_E2E1_RECEIPT` is unset.
//!
//! - `AIEN_E2E_CHECKPOINT`: TinyLlama-1.1B-Chat-v1.0 directory (holding
//!   `model.safetensors` and `tokenizer.json`) or the `.safetensors` file.
//! - `AIEN_E2E1_RECEIPT`: path of the JSON receipt (an array, one object per N).
//!
//! The helpers (strict load, tolerance reader, recording backend, parity,
//! receipt helpers) are copied from `prefill_e2e0.rs` (not edited) with two
//! changes: the recording backend keeps every step's logits only for the
//! branches that get a control run (all branches keep step 1), and the gate
//! body is a loop over N with per-N pool sizing and per-N checks.
//!
//! Per N the checks are: (a) allocated is not ready, prefill executed in >= 2
//! chunks, the fence made the root ready before any fork, a fork before the
//! fence is refused; (b) every branch's step 1 equals the root continuation,
//! and a documented sample of branches (at least 8, first and last included,
//! all of them for N <= 8) equals an independent control at every step, at the
//! frozen tolerance 0.0 (docs/campaigns/c1/acceptance.toml
//! [numerics.noise_floor.tinyllama]); (c) the prefix is held once, and the
//! physical KV peak stays within prefix blocks + N x per-branch suffix blocks
//! and below the logical bytes for N > 1; (d) branches diverge (distinct
//! request ids give distinct sampling seeds, crates/aien-inference-abi/src/
//! transformer_backend.rs `decode_sampling_seed`) and a decode batch mixed
//! different branch inputs; (e) backend sequences, KV blocks, arena, worlds and
//! scheduler all empty after each N.
//!
//! Scheduler admission (GB10-2): `build_scheduled_batch` step 3 asks
//! `AienKvManager::incremental_blocks_needed` for the NEW blocks a sequence
//! needs, so a fork child that shares the root blocks needs only its
//! copy-on-write block. The pool is sized to the true requirement (prefix + N x
//! private suffix + watermark, see `pool_plan`) and records the pool size and
//! the reason. A step-guard hit, an idle-while-waiting step (starved branch) or
//! a scheduler/spine error is FAIL.
//!
//! The run stops at the first N that fails (later counts are listed in the
//! receipt as `not_run_after_failure`), so a failing gate stays failing and
//! mutants die fast. All checks of an N are evaluated and the receipt for the
//! whole run is written before any assert fires.

#![recursion_limit = "256"]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, FinishReason, ModelConfig, NativeTransformerBackend,
    PrefillState, ReferenceCpuBackend, SamplingParams, ScheduledBatch, StepMetrics,
    TinyLlamaTokenizer, TransformerWeights,
};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::swarm::SwarmConfig;
use aien_scheduler::SchedulerConfig;
use async_trait::async_trait;
use sha2::{Digest, Sha256};

/// Branch counts of the gate, run in this order.
const BRANCH_COUNTS: [usize; 6] = [1, 2, 8, 32, 128, 500];
/// Branch count from which (d) requires distinct outputs and a mixed decode
/// batch; below it the counts are recorded only (sampling coincidence).
const DIVERGENCE_MIN_N: usize = 8;
/// Decode budget per branch. 4 tokens keep N = 500 finishable on CPU: the run
/// does N x 4 batched decode rows plus one full control (217-token prefill + 4
/// decodes) per sampled branch, and 4 tokens still cover a partial-block
/// copy-on-write, a mixed decode batch and per-step parity.
const MAX_TOKENS_PER_BRANCH: usize = 4;
/// Branches that get a full per-step control run (N <= 8: all of them).
const SAMPLED_CONTROLS: usize = 8;
/// The daemon's prefill chunk size (crates/aien-cli/src/commands.rs,
/// `run_daemon_server` scheduler config). The root prompt must exceed it.
const PREFILL_CHUNK: usize = 128;
/// Control sequence id inside each fresh control backend.
const CONTROL_ID: u64 = 1;

const SYSTEM_PROMPT: &str = "You are a careful engineering assistant. You answer in plain \
English, you show the steps of your reasoning, and you never invent numbers you were not given.";

const USER_PROMPT: &str = "A small workshop builds wooden chairs. Each chair needs four legs, \
one seat, two side rails, one back rest and twelve screws. The workshop has ninety legs, \
twenty seats, forty side rails, eighteen back rests and two hundred screws in stock. Two of \
the seats are cracked and cannot be used, and one box of twenty screws turned out to be the \
wrong size. The carpenter wants to know how many complete chairs can be built today, which \
part runs out first, and how many of every other part will be left over afterwards. After \
that, suggest the smallest order of extra parts that would let the workshop build exactly \
twenty five chairs tomorrow, and explain briefly why that order is the smallest one.";

// ---------------------------------------------------------------------------
// Environment, strict checkpoint, acceptance tolerance
// ---------------------------------------------------------------------------

struct CheckpointPaths {
    model: PathBuf,
    tokenizer: PathBuf,
}

fn required_env(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => panic!(
            "PREFILL_E2E1_ENV: {name} is unset; this gate never skips (set it and rerun with \
             --ignored)"
        ),
    }
}

fn checkpoint_paths(raw: &str) -> CheckpointPaths {
    let p = PathBuf::from(raw);
    let (model, dir) = if p.is_dir() {
        (p.join("model.safetensors"), p.clone())
    } else {
        let dir = p.parent().map(Path::to_path_buf).unwrap_or_default();
        (p.clone(), dir)
    };
    let tokenizer = dir.join("tokenizer.json");
    assert!(
        model.is_file(),
        "STRICT_CHECKPOINT_VIOLATION: checkpoint {} does not exist; refusing to fall back \
         (AIEN_REQUIRE_CHECKPOINT=1 semantics)",
        model.display()
    );
    assert!(
        tokenizer.is_file(),
        "STRICT_CHECKPOINT_VIOLATION: tokenizer {} does not exist; refusing to fall back",
        tokenizer.display()
    );
    CheckpointPaths { model, tokenizer }
}

fn sha256_file_hex(path: &Path) -> String {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("STRICT_CHECKPOINT_VIOLATION: open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: renormalize.
            let mut e: i32 = 0;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (exp32 << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

/// Reads `count` values starting at element `first` of tensor `name` straight
/// from the safetensors file (8-byte header length, JSON header, data), with
/// no aien code involved.
fn read_tensor_values(path: &Path, name: &str, first: usize, count: usize) -> Vec<f32> {
    let mut file = std::fs::File::open(path).expect("open checkpoint");
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .expect("safetensors header length");
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).expect("safetensors header");
    let header: serde_json::Value = serde_json::from_slice(&header).expect("header JSON");
    let info = header
        .get(name)
        .unwrap_or_else(|| panic!("STRICT_CHECKPOINT_VIOLATION: tensor {name} not in checkpoint"));
    let dtype = info["dtype"].as_str().expect("dtype").to_string();
    let start = info["data_offsets"][0].as_u64().expect("data_offsets") as usize;
    let width = match dtype.as_str() {
        "F32" => 4,
        "BF16" | "F16" => 2,
        other => panic!("unsupported checkpoint dtype {other}"),
    };
    let mut raw = vec![0u8; count * width];
    file.seek(SeekFrom::Start(
        (8 + header_len + start + first * width) as u64,
    ))
    .expect("seek tensor");
    file.read_exact(&mut raw).expect("read tensor bytes");
    raw.chunks_exact(width)
        .map(|c| match dtype.as_str() {
            "F32" => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            "BF16" => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
            _ => f16_to_f32(u16::from_le_bytes([c[0], c[1]])),
        })
        .collect()
}

/// Loads TinyLlama with no fallback and proves the weights came from the file.
fn strict_load(paths: &CheckpointPaths) -> TransformerWeights {
    let config = ModelConfig::tinyllama_1_1b();
    let weights =
        TransformerWeights::load_from_safetensors(&paths.model, &config).unwrap_or_else(|e| {
            panic!(
                "STRICT_CHECKPOINT_VIOLATION: {} failed to load against the {} config: {e:?}; \
                 refusing to fall back (AIEN_REQUIRE_CHECKPOINT=1 semantics)",
                paths.model.display(),
                config.model_id
            )
        });
    let hidden = config.hidden_dim;
    let vocab = config.vocab_size;
    assert_eq!(
        weights.embed_tokens.len(),
        vocab * hidden,
        "STRICT_CHECKPOINT_VIOLATION: embedding shape does not match the TinyLlama config"
    );
    for row in [0usize, vocab - 1] {
        let from_file = read_tensor_values(
            &paths.model,
            "model.embed_tokens.weight",
            row * hidden,
            hidden,
        );
        let loaded = &weights.embed_tokens[row * hidden..(row + 1) * hidden];
        assert!(
            loaded
                .iter()
                .zip(&from_file)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "STRICT_CHECKPOINT_VIOLATION: loaded embedding row {row} differs from the bytes in \
             {} (weights did not come from the checkpoint; reference fallback?)",
            paths.model.display()
        );
    }
    weights
}

struct Tolerance {
    abs: f64,
    rel: f64,
    rel_eps: f64,
    floor_status: String,
    floor_model_sha256: String,
}

/// Minimal reader for `key = value` lines of one TOML table (no new crate).
fn toml_table(text: &str, table: &str) -> HashMap<String, String> {
    let header = format!("[{table}]");
    let mut out = HashMap::new();
    let mut inside = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            inside = line == header;
            continue;
        }
        if !inside || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let v = v.split('#').next().unwrap_or("").trim().trim_matches('"');
            out.insert(k.trim().to_string(), v.to_string());
        }
    }
    out
}

fn acceptance_tolerance() -> Tolerance {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/campaigns/c1/acceptance.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("PARITY_TOLERANCE: cannot read {}: {e}", path.display()));
    let floor = toml_table(&text, "numerics.noise_floor.tinyllama");
    let numerics = toml_table(&text, "numerics");
    let num = |t: &HashMap<String, String>, k: &str| -> f64 {
        t.get(k)
            .unwrap_or_else(|| panic!("PARITY_TOLERANCE: {k} missing in {}", path.display()))
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("PARITY_TOLERANCE: {k} not a number: {e}"))
    };
    Tolerance {
        abs: num(&floor, "tolerance_abs"),
        rel: num(&floor, "tolerance_rel"),
        rel_eps: num(&numerics, "relative_error_epsilon"),
        floor_status: floor.get("status").cloned().unwrap_or_default(),
        floor_model_sha256: floor.get("model_sha256").cloned().unwrap_or_default(),
    }
}

// ---------------------------------------------------------------------------
// Recording backend
// ---------------------------------------------------------------------------

struct PrefillChunk {
    id: u64,
    len: usize,
    state_before: Option<PrefillState>,
}

struct ForkRecord {
    parent: u64,
    child: u64,
    parent_ready: bool,
    ok: bool,
    child_tokens_ok: bool,
    temperature: f32,
}

/// State at the first fork hook call (right after the fence).
struct ForkSnapshot {
    root_state: Option<PrefillState>,
    root_backend_tokens: usize,
    root_pending_token: Option<u32>,
    root_kv_values: usize,
    root_kv_nonzero: usize,
}

/// KV state at the first decode batch (all branches forked, nothing decoded).
struct PrefixSnapshot {
    root_blocks: Vec<usize>,
    branches_share_root_blocks: bool,
    physical_pages: usize,
    logical_pages: usize,
    shared_pages: usize,
    allocated_blocks: usize,
    block_bytes: usize,
}

/// Delegates to the real backend and records what the gate needs.
struct E2eTap {
    inner: NativeTransformerBackend,
    root: u64,
    branches: Vec<u64>,
    prompt: Vec<u32>,
    launch_at: Instant,
    prefill_chunks: Vec<PrefillChunk>,
    prefill_wall: Duration,
    forks: Vec<ForkRecord>,
    fork_snapshot: Option<ForkSnapshot>,
    prefix_snapshot: Option<PrefixSnapshot>,
    decode_logits: HashMap<u64, Vec<Vec<f32>>>,
    decode_tokens: HashMap<u64, Vec<u32>>,
    decode_batches: Vec<Vec<(u64, u32)>>,
    first_branch_token: Option<Duration>,
    peak_physical_pages: usize,
    /// Branches whose every step keeps its logits (the others keep step 1 only).
    keep_all: std::collections::HashSet<u64>,
}

impl E2eTap {
    fn kv_state(&self, id: u64) -> Option<PrefillState> {
        self.inner
            .kv_manager
            .as_ref()
            .and_then(|kv| kv.read().prefill_state(id))
    }

    fn take_fork_snapshot(&mut self) {
        let root = self.root;
        let (values, nonzero) = match self.inner.kv_manager.as_ref() {
            Some(kv) => {
                let kv = kv.read();
                let mut values = 0usize;
                let mut nonzero = 0usize;
                for layer in 0..self.inner.weights.config.num_layers {
                    if let Ok((k, v)) = kv.gather_layer_kv(root, layer) {
                        values += k.len() + v.len();
                        nonzero += k.iter().chain(v.iter()).filter(|x| **x != 0.0).count();
                    }
                }
                (values, nonzero)
            }
            None => (0, 0),
        };
        self.fork_snapshot = Some(ForkSnapshot {
            root_state: self.kv_state(root),
            root_backend_tokens: self
                .inner
                .sequences
                .get(&root)
                .map_or(0, |s| s.tokens.len()),
            root_pending_token: self.inner.pending_prefill_token.get(&root).copied(),
            root_kv_values: values,
            root_kv_nonzero: nonzero,
        });
    }

    fn take_prefix_snapshot(&mut self) {
        let Some(kv) = self.inner.kv_manager.as_ref() else {
            return;
        };
        let kv = kv.read();
        let root_blocks = kv
            .get_block_table(self.root)
            .map(|t| t.block_ids.clone())
            .unwrap_or_default();
        let branches_share_root_blocks = self.branches.iter().all(|b| {
            kv.get_block_table(*b)
                .is_some_and(|t| !root_blocks.is_empty() && t.block_ids == root_blocks)
        });
        let m = kv.metrics();
        self.prefix_snapshot = Some(PrefixSnapshot {
            root_blocks,
            branches_share_root_blocks,
            physical_pages: m.physical_pages,
            logical_pages: m.logical_pages,
            shared_pages: m.shared_pages,
            allocated_blocks: kv.allocated_block_count(),
            block_bytes: kv.tensor_pool().map_or(0, |p| p.block_bytes()),
        });
    }

    fn record_fork(&mut self, parent: u64, child: u64, temperature: f32, ok: bool) {
        let child_tokens_ok = ok
            && match (
                self.inner.sequences.get(&child),
                self.fork_snapshot
                    .as_ref()
                    .and_then(|s| s.root_pending_token),
            ) {
                (Some(seq), Some(pending)) => {
                    seq.tokens.len() == self.prompt.len() + 1
                        && seq.tokens[..self.prompt.len()] == self.prompt[..]
                        && seq.tokens[self.prompt.len()] == pending
                }
                _ => false,
            };
        let parent_ready = self.kv_state(parent).is_some_and(|s| s.is_ready());
        self.forks.push(ForkRecord {
            parent,
            child,
            parent_ready,
            ok,
            child_tokens_ok,
            temperature,
        });
    }
}

#[async_trait]
impl AienInferenceBackend for E2eTap {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut outputs = Vec::new();
        let mut metrics = StepMetrics::default();
        if !batch.prefill_requests.is_empty() {
            for req in &batch.prefill_requests {
                let state_before = self.kv_state(req.request_id);
                self.prefill_chunks.push(PrefillChunk {
                    id: req.request_id,
                    len: req.prompt_tokens.len(),
                    state_before,
                });
            }
            let prefill_only = ScheduledBatch {
                decode_requests: Vec::new(),
                ..batch.clone()
            };
            let t0 = Instant::now();
            let (o, m) = self.inner.execute_step(&prefill_only).await?;
            self.prefill_wall += t0.elapsed();
            outputs.extend(o);
            metrics = m;
        }
        if !batch.decode_requests.is_empty() {
            // Normally taken at the first fork hook call. If the spine never
            // called the hook, take it here so b4/b5 still judge the prefill
            // and b7 reports the missing fork.
            if self.fork_snapshot.is_none() {
                self.take_fork_snapshot();
            }
            if self.prefix_snapshot.is_none() {
                self.take_prefix_snapshot();
            }
            let fed: Vec<(u64, u32)> = batch
                .decode_requests
                .iter()
                .map(|id| {
                    let last = self
                        .inner
                        .pending_prefill_token
                        .get(id)
                        .copied()
                        .or_else(|| {
                            self.inner
                                .sequences
                                .get(id)
                                .and_then(|s| s.tokens.last().copied())
                        });
                    (*id, last.unwrap_or(u32::MAX))
                })
                .collect();
            self.decode_batches.push(fed);
            let (o, logits) = self
                .inner
                .forward_decode_batch_with_logits(&batch.decode_requests)?;
            // Rows ended as Preempted (no KV slot) get no logits.
            let preempted: Vec<u64> = o
                .iter()
                .filter_map(|out| match out {
                    DecodeOutput::Finished {
                        request_id,
                        reason: FinishReason::Preempted,
                        ..
                    } => Some(*request_id),
                    _ => None,
                })
                .collect();
            let computed = batch
                .decode_requests
                .iter()
                .filter(|id| !preempted.contains(*id));
            for (id, row) in computed.zip(logits) {
                let rows = self.decode_logits.entry(*id).or_default();
                if rows.is_empty() || self.keep_all.contains(id) {
                    rows.push(row);
                }
                if let Some(tok) = self.inner.sequences.get(id).and_then(|s| s.tokens.last()) {
                    self.decode_tokens.entry(*id).or_default().push(*tok);
                }
            }
            if self.first_branch_token.is_none()
                && o.iter().any(|out| {
                    matches!(out, DecodeOutput::Token { request_id, .. }
                        if self.branches.contains(request_id))
                })
            {
                self.first_branch_token = Some(self.launch_at.elapsed());
            }
            if let Some(kv) = self.inner.kv_manager.as_ref() {
                let pages = kv.read().metrics().physical_pages;
                self.peak_physical_pages = self.peak_physical_pages.max(pages);
            }
            outputs.extend(o);
            metrics.decode_tokens_emitted += batch.decode_requests.len();
        }
        Ok((outputs, metrics))
    }

    fn manages_kv_cache(&self) -> bool {
        self.inner.manages_kv_cache()
    }

    fn fork_sequence(&mut self, parent_id: u64, child_id: u64) -> Result<(), String> {
        if self.fork_snapshot.is_none() {
            self.take_fork_snapshot();
        }
        let result = self.inner.fork_sequence(parent_id, child_id);
        self.record_fork(parent_id, child_id, 0.0, result.is_ok());
        result
    }

    fn fork_sequence_with_sampling(
        &mut self,
        parent_id: u64,
        child_id: u64,
        sampling: &SamplingParams,
    ) -> Result<(), String> {
        if self.fork_snapshot.is_none() {
            self.take_fork_snapshot();
        }
        let result = self
            .inner
            .fork_sequence_with_sampling(parent_id, child_id, sampling);
        self.record_fork(parent_id, child_id, sampling.temperature, result.is_ok());
        result
    }

    fn release_sequence(&mut self, seq_id: u64) -> Result<(), String> {
        // The trait hook (runtime reclaim), not the inherent method of the same
        // name, which also frees KV.
        AienInferenceBackend::release_sequence(&mut self.inner, seq_id)
    }
}

// ---------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Parity {
    max_abs: f64,
    max_rel: f64,
    compared_values: usize,
    violations: usize,
    first_violation: Option<String>,
}

impl Parity {
    /// Element passes iff |a-c| <= tolerance_abs or |a-c| / max(|c|, eps) <=
    /// tolerance_rel (acceptance.toml [numerics] relative error definition).
    /// NaN never passes.
    fn compare(&mut self, what: &str, row: &[f32], control: &[f32], tol: &Tolerance) {
        if row.len() != control.len() {
            self.violations += 1;
            if self.first_violation.is_none() {
                self.first_violation = Some(format!(
                    "{what}: length {} vs control {}",
                    row.len(),
                    control.len()
                ));
            }
            return;
        }
        for (i, (a, c)) in row.iter().zip(control).enumerate() {
            let abs = (f64::from(*a) - f64::from(*c)).abs();
            let rel = abs / f64::from(c.abs()).max(tol.rel_eps);
            self.compared_values += 1;
            if abs.is_nan() || abs > self.max_abs {
                self.max_abs = if abs.is_nan() { f64::INFINITY } else { abs };
            }
            if rel.is_nan() || rel > self.max_rel {
                self.max_rel = if rel.is_nan() { f64::INFINITY } else { rel };
            }
            let ok = abs <= tol.abs || rel <= tol.rel;
            if !ok {
                self.violations += 1;
                if self.first_violation.is_none() {
                    self.first_violation = Some(format!(
                        "{what}: logit {i} = {a} vs control {c} (abs {abs:e}, rel {rel:e})"
                    ));
                }
            }
        }
    }

    fn pass(&self) -> bool {
        self.violations == 0 && self.compared_values > 0
    }
}

/// Fresh unpaged backend (own dense K/V, no KV manager), one-shot prefill of
/// the root prompt, then the root's sampled token. Returns the backend.
fn control_after_prompt(
    weights: &TransformerWeights,
    prompt: &[u32],
    root_token: u32,
) -> NativeTransformerBackend {
    let mut control = NativeTransformerBackend::new(weights.clone());
    assert!(control.kv_manager.is_none(), "control must be unpaged");
    control
        .prefill_sequence(CONTROL_ID, prompt)
        .expect("control one-shot prefill");
    control
        .sequences
        .get_mut(&CONTROL_ID)
        .expect("control sequence")
        .tokens
        .push(root_token);
    control
}

// ---------------------------------------------------------------------------
// Receipt helpers
// ---------------------------------------------------------------------------

fn hardware() -> serde_json::Value {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let processors = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    let mut fields: HashMap<String, Vec<String>> = HashMap::new();
    for line in cpuinfo.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            if matches!(
                k,
                "model name" | "Hardware" | "CPU implementer" | "CPU part" | "CPU architecture"
            ) {
                let entry = fields.entry(k.to_string()).or_default();
                let v = v.trim().to_string();
                if !entry.contains(&v) {
                    entry.push(v);
                }
            }
        }
    }
    let uname = std::process::Command::new("uname")
        .arg("-a")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    serde_json::json!({
        "source": "/proc/cpuinfo + uname -a, read at runtime",
        "processors": processors,
        "cpuinfo": fields,
        "uname": uname,
    })
}

/// `git rev-parse HEAD` in this crate's directory, at runtime.
fn commit_sha() -> Option<String> {
    std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

struct Check {
    name: &'static str,
    pass: bool,
    message: String,
}

fn check(name: &'static str, pass: bool, message: String) -> Check {
    Check {
        name,
        pass,
        message,
    }
}

// ---------------------------------------------------------------------------
// Pool sizing and branch sampling
// ---------------------------------------------------------------------------

/// Extra free blocks on top of the exact need (prefix + N x private suffix +
/// watermark). Zero: the scheduler admits by the KV manager's incremental
/// need (GB10-2), so the pool is sized to the true requirement.
const POOL_SLACK_BLOCKS: usize = 0;

struct PoolPlan {
    prefix_blocks: usize,
    per_branch_suffix_blocks: usize,
    watermark_blocks: usize,
    total_blocks: usize,
}

/// Blocks the pool needs for `n` branches over a `prompt_len`-token root.
///
/// - prefix: the root's blocks, held once.
/// - per branch: a branch appends at most MAX_TOKENS_PER_BRANCH + 1 tokens
///   (the sampled root token plus its decodes). A shared partial last block is
///   copied on the first append (copy-on-write, crates/aien-kv-cache/src/
///   lib.rs `append_token_with_slot`), so the private blocks a branch owns are
///   ceil((prompt_len % block_size + appends) / block_size).
/// - watermark: the scheduler preempts when free blocks fall below it, so the
///   pool keeps that many free on top. Admission itself needs only the
///   incremental blocks (GB10-2: `AienKvManager::incremental_blocks_needed`),
///   not a whole prompt per branch.
fn pool_plan(prompt_len: usize, block_size: usize, n: usize, watermark: usize) -> PoolPlan {
    let prefix_blocks = prompt_len.div_ceil(block_size);
    let appends = MAX_TOKENS_PER_BRANCH + 1;
    let per_branch_suffix_blocks = (prompt_len % block_size + appends).div_ceil(block_size);
    PoolPlan {
        prefix_blocks,
        per_branch_suffix_blocks,
        watermark_blocks: watermark,
        total_blocks: prefix_blocks + n * per_branch_suffix_blocks + watermark + POOL_SLACK_BLOCKS,
    }
}

/// Indices of the branches that get a full per-step control: all of them for
/// n <= SAMPLED_CONTROLS, otherwise SAMPLED_CONTROLS evenly spaced indices
/// from 0 to n - 1 (first and last always included).
fn sampled_indices(n: usize) -> Vec<usize> {
    if n <= SAMPLED_CONTROLS {
        return (0..n).collect();
    }
    (0..SAMPLED_CONTROLS)
        .map(|i| i * (n - 1) / (SAMPLED_CONTROLS - 1))
        .collect()
}

fn mem_total_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

// ---------------------------------------------------------------------------
// One run at one N
// ---------------------------------------------------------------------------

struct Shared<'a> {
    weights: &'a TransformerWeights,
    prompt: &'a [u32],
    tol: &'a Tolerance,
    common: &'a serde_json::Value,
}

struct RunResult {
    receipt: serde_json::Value,
    checks: Vec<Check>,
    pass: bool,
}

async fn run_one(n: usize, sh: &Shared<'_>) -> RunResult {
    let prompt = sh.prompt;
    let prompt_len = prompt.len();
    let tol = sh.tol;

    // Fresh controller state per N (RuntimeController persists operation ids).
    let state_dir = tempfile::tempdir().expect("state dir");
    std::env::set_var("AIEN_RUNTIME_STATE_DIR", state_dir.path());

    let block_size = sh.weights.config.block_size;
    let watermark = 2usize;
    let plan = pool_plan(prompt_len, block_size, n, watermark);
    let max_batch = 32usize.max(n + 2);
    let scheduler = SchedulerConfig {
        max_batch_size: max_batch,
        max_batch_tokens: 4096,
        max_prefill_tokens: 8192,
        prefill_chunk_size: PREFILL_CHUNK,
        chunk_prefill: true,
        watermark_blocks: watermark,
    };
    let (mut spine, backend) = build_shared_kv_runtime(
        sh.weights.clone(),
        Arc::new(ReferenceCpuBackend::new()),
        scheduler,
        SharedKvSizing {
            arena_capacity: n + 16,
            total_blocks: plan.total_blocks,
        },
    )
    .expect("build shared KV runtime");
    let kv = spine.kv_manager.clone();
    let backend_name = format!(
        "NativeTransformerBackend/{} (paged, pooled shared KV)",
        backend.tensor_backend.name()
    );

    let mut checks: Vec<Check> = Vec::new();

    // (a) one root submitted via the spine swarm API.
    let launch_at = Instant::now();
    let swarm_id = spine
        .launch_swarm(
            SwarmConfig {
                model_handle: 1,
                branch_count: n,
                max_active_sequences: n,
                max_tokens_per_branch: MAX_TOKENS_PER_BRANCH,
                root_world_id: 0,
                priority: 1,
            },
            prompt,
        )
        .expect("launch swarm");
    let swarm = spine
        .swarm_manager
        .get_swarm(swarm_id)
        .expect("swarm record")
        .clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();
    let pending = spine.swarm_manager.pending_root_prefills();
    checks.push(check(
        "a_one_root",
        pending.len() == 1
            && pending[0].1 == root
            && pending[0].2 == prompt
            && spine.swarm_manager.active_swarm_count() == 1
            && branches.len() == n,
        format!(
            "(a) ONE_ROOT_VIOLATION: N={n}: expected exactly one pending root ({root}) with the \
             submitted prompt and {n} branches; pending roots {:?}, swarms {}, branches {}",
            pending.iter().map(|p| p.1).collect::<Vec<_>>(),
            spine.swarm_manager.active_swarm_count(),
            branches.len()
        ),
    ));

    // (a) allocating the root's KV does not make it ready.
    let root_state = kv.read().prefill_state(root);
    checks.push(check(
        "a_allocated_not_ready",
        root_state == Some(PrefillState::Allocated),
        format!(
            "(a) ALLOCATED_IS_NOT_READY_VIOLATION: N={n}: root KV state right after launch is \
             {root_state:?}, expected Allocated (not ready)"
        ),
    ));

    let sampled_idx = sampled_indices(n);
    let sampled_ids: Vec<u64> = sampled_idx.iter().map(|i| branches[*i]).collect();
    let mut tap = E2eTap {
        inner: backend,
        root,
        branches: branches.clone(),
        prompt: prompt.to_vec(),
        launch_at,
        prefill_chunks: Vec::new(),
        prefill_wall: Duration::ZERO,
        forks: Vec::new(),
        fork_snapshot: None,
        prefix_snapshot: None,
        decode_logits: HashMap::new(),
        decode_tokens: HashMap::new(),
        decode_batches: Vec::new(),
        first_branch_token: None,
        peak_physical_pages: 0,
        keep_all: sampled_ids.iter().copied().collect(),
    };

    // (a) no fork before the fence: neither the KV fork nor the backend fork.
    let kv_fork = {
        let mut guard = kv.write();
        spine
            .swarm_manager
            .fork_branches_from_ready_root(swarm_id, &mut spine.arena, &mut guard)
    };
    let backend_fork = tap.inner.fork_sequence(root, branches[0]);
    let any_branch_table = branches
        .iter()
        .any(|b| kv.read().get_block_table(*b).is_some());
    checks.push(check(
        "a_no_fork_before_ready",
        kv_fork.is_err()
            && backend_fork.is_err()
            && !any_branch_table
            && !tap.inner.sequences.contains_key(&branches[0]),
        format!(
            "(a) FORK_BEFORE_FENCE_VIOLATION: N={n}: before the fence the KV fork returned \
             {kv_fork:?}, the backend fork returned {backend_fork:?}, branch KV tables exist: \
             {any_branch_table}"
        ),
    ));

    // Run spine steps until every branch finished. A step-guard hit, an idle
    // step while work is waiting (starved branch) and a step error are FAILs.
    let step_guard = 100 + 2 * n;
    let mut run_error: Option<String> = None;
    let mut steps = 0usize;
    let mut idle = 0usize;
    let mut guard_hit = false;
    let mut starved = false;
    while spine.scheduler.running_count() > 0
        || spine.scheduler.waiting_count() > 0
        || spine.swarm_manager.is_root_prefill_pending(swarm_id)
    {
        if steps >= step_guard {
            guard_hit = true;
            break;
        }
        steps += 1;
        match spine.step(&mut tap).await {
            Err(e) => {
                run_error = Some(e);
                break;
            }
            Ok(Some(_)) => idle = 0,
            Ok(None) => {
                idle += 1;
                if idle >= 3 {
                    starved = true;
                    break;
                }
            }
        }
    }
    let run_note = format!(
        "; spine steps {steps}/{step_guard}, step guard hit {guard_hit}, starved {starved}, \
         step error {run_error:?}"
    );
    let run_ok = run_error.is_none() && !guard_hit && !starved;
    checks.push(check(
        "run_completed",
        run_ok,
        format!(
            "(a) RUN_INCOMPLETE_VIOLATION: N={n}: the run did not finish cleanly (a step-guard \
             hit or a starved branch is a FAIL, never silent; pool {} blocks){run_note}",
            plan.total_blocks
        ),
    ));

    // (a) model prefill executed.
    let root_chunks: Vec<&PrefillChunk> =
        tap.prefill_chunks.iter().filter(|c| c.id == root).collect();
    let root_prefilled: usize = root_chunks.iter().map(|c| c.len).sum();
    let other_prefills = tap.prefill_chunks.len() - root_chunks.len();
    let snap = tap.fork_snapshot.as_ref();
    let prefill_ok = root_prefilled == prompt_len
        && root_chunks.len() >= 2
        && other_prefills == 0
        && snap.is_some_and(|s| {
            s.root_backend_tokens == prompt_len && s.root_kv_values > 0 && s.root_kv_nonzero > 0
        });
    checks.push(check(
        "a_prefill_executed",
        prefill_ok,
        format!(
            "(a) PREFILL_EXECUTED_VIOLATION: N={n}: backend prefilled {root_prefilled}/{prompt_len} \
             root tokens in {} chunks, {other_prefills} non-root prefills; at the fence the backend \
             held {:?} root tokens and the pool held {:?} non-zero of {:?} root K/V values{run_note}",
            root_chunks.len(),
            snap.map(|s| s.root_backend_tokens),
            snap.map(|s| s.root_kv_nonzero),
            snap.map(|s| s.root_kv_values),
        ),
    ));

    // (a) the completion fence made the root ready, and only after the prefill.
    let chunks_not_ready = root_chunks
        .iter()
        .all(|c| !matches!(c.state_before, Some(s) if s.is_ready()));
    checks.push(check(
        "a_fence_made_ready",
        chunks_not_ready && snap.is_some_and(|s| s.root_state.is_some_and(|st| st.is_ready())),
        format!(
            "(a) FENCE_VIOLATION: N={n}: root states before each prefill chunk {:?}; at the first \
             fork {:?}{run_note}",
            root_chunks
                .iter()
                .map(|c| c.state_before)
                .collect::<Vec<_>>(),
            snap.map(|s| s.root_state),
        ),
    ));

    // N branches forked from the ready root, backend state included.
    let good_forks: Vec<&ForkRecord> = tap
        .forks
        .iter()
        .filter(|f| f.ok && f.parent == root && f.parent_ready && f.child_tokens_ok)
        .collect();
    let forked_children: Vec<u64> = good_forks.iter().map(|f| f.child).collect();
    let all_forked = branches.iter().all(|b| forked_children.contains(b))
        && good_forks.len() == n
        && tap.forks.len() == n
        && good_forks.iter().all(|f| f.temperature > 0.001);
    let root_token = snap.and_then(|s| s.root_pending_token);
    // Root continuation: fresh unpaged control, prompt + root token, one decode.
    // EVERY branch's step 1 is compared against it.
    let mut step1 = Parity::default();
    let mut step1_detail = String::new();
    if let (true, Some(root_token)) = (all_forked, root_token) {
        let mut control = control_after_prompt(&tap.inner.weights, prompt, root_token);
        let (_o, ctl) = control
            .forward_decode_batch_with_logits(&[CONTROL_ID])
            .expect("root continuation control decode");
        let root_continuation = ctl.into_iter().next().expect("control logits");
        for b in &branches {
            match tap.decode_logits.get(b).and_then(|v| v.first()) {
                Some(first) => step1.compare(
                    &format!("branch {b} step 1"),
                    first,
                    &root_continuation,
                    tol,
                ),
                None => {
                    step1.violations += 1;
                    step1_detail.push_str(&format!(" branch {b} never decoded;"));
                }
            }
        }
    }
    checks.push(check(
        "b_forks_and_step1",
        all_forked && root_token.is_some() && step1.pass(),
        format!(
            "(a/b) BRANCH_FORK_VIOLATION: N={n}: {} fork hook calls, {} valid (parent ready, child \
             = prompt + root token, sampling params); root token {root_token:?}; step-1 of all {n} \
             branches vs root continuation: max_abs {:e}, violations {}{}{step1_detail}{run_note}",
            tap.forks.len(),
            good_forks.len(),
            step1.max_abs,
            step1.violations,
            step1
                .first_violation
                .as_deref()
                .map_or(String::new(), |v| format!(", first: {v}")),
        ),
    ));

    // (b) sampled branches, every step, vs independent controls.
    let mut parity = Parity::default();
    let mut b9_detail = String::new();
    let generated: Vec<Vec<u32>> = branches
        .iter()
        .map(|b| tap.decode_tokens.get(b).cloned().unwrap_or_default())
        .collect();
    match root_token {
        Some(root_token) if run_ok => {
            for (idx, b) in sampled_idx.iter().zip(&sampled_ids) {
                let gen = &generated[*idx];
                let rows = tap.decode_logits.get(b).cloned().unwrap_or_default();
                if rows.is_empty() || rows.len() != gen.len() {
                    parity.violations += 1;
                    b9_detail.push_str(&format!(
                        " branch {b}: {} logit rows for {} tokens;",
                        rows.len(),
                        gen.len()
                    ));
                    continue;
                }
                let mut control = control_after_prompt(&tap.inner.weights, prompt, root_token);
                for (k, row) in rows.iter().enumerate() {
                    let (_o, ctl) = control
                        .forward_decode_batch_with_logits(&[CONTROL_ID])
                        .expect("control decode");
                    let ctl_row = ctl.into_iter().next().expect("control logits");
                    parity.compare(&format!("branch {b} step {}", k + 1), row, &ctl_row, tol);
                    // Teacher forcing with this branch's own token only.
                    let seq = control.sequences.get_mut(&CONTROL_ID).expect("control seq");
                    seq.tokens.pop();
                    seq.tokens.push(gen[k]);
                }
            }
        }
        _ => {
            parity.violations += 1;
            b9_detail.push_str(" no root token / run failed;");
        }
    }
    let parity_pass = parity.pass() && step1.pass() && sampled_ids.len() >= n.min(SAMPLED_CONTROLS);
    checks.push(check(
        "b_parity_vs_controls",
        parity_pass,
        format!(
            "(b) BRANCH_PARITY_VIOLATION: N={n}: {} sampled controls; max_abs {:e}, max_rel {:e} \
             over {} values vs tolerance abs {:e} rel {:e} (acceptance.toml); {} violations{}{b9_detail}{run_note}",
            sampled_ids.len(),
            parity.max_abs,
            parity.max_rel,
            parity.compared_values,
            tol.abs,
            tol.rel,
            parity.violations,
            parity
                .first_violation
                .as_deref()
                .map_or(String::new(), |v| format!(", first: {v}")),
        ),
    ));

    // (c) common prefix held once, physical KV bounded.
    let prefix = tap.prefix_snapshot.as_ref();
    let block_bytes = prefix.map_or(0, |p| p.block_bytes);
    let r = plan.prefix_blocks;
    let prefix_ok = prefix.is_some_and(|p| {
        p.root_blocks.len() == r
            && p.branches_share_root_blocks
            && p.physical_pages == r
            && p.allocated_blocks == r
            && p.shared_pages == r
            && p.logical_pages == (n + 1) * r
    });
    checks.push(check(
        "c_prefix_held_once",
        prefix_ok,
        format!(
            "(c) PREFIX_MULTIPLIED_VIOLATION: N={n}: at the first decode: root blocks {:?} (expected \
             {r}), branches share them {:?}, KvMetrics physical {:?} logical {:?} (expected {}) \
             shared {:?}, allocated {:?}{run_note}",
            prefix.map(|p| p.root_blocks.len()),
            prefix.map(|p| p.branches_share_root_blocks),
            prefix.map(|p| p.physical_pages),
            prefix.map(|p| p.logical_pages),
            (n + 1) * r,
            prefix.map(|p| p.shared_pages),
            prefix.map(|p| p.allocated_blocks),
        ),
    ));
    let prefix_bytes = r * block_bytes;
    let per_branch_suffix_bytes = plan.per_branch_suffix_blocks * block_bytes;
    let bound_bytes = prefix_bytes + n * per_branch_suffix_bytes;
    let peak_bytes = tap.peak_physical_pages * block_bytes;
    let logical_bytes = prefix.map_or(0, |p| p.logical_pages * block_bytes);
    let physical_bytes = prefix.map_or(0, |p| p.physical_pages * block_bytes);
    let bounded = block_bytes > 0
        && tap.peak_physical_pages >= r
        && peak_bytes <= bound_bytes
        && (n == 1 || peak_bytes < logical_bytes);
    checks.push(check(
        "c_physical_kv_bounded",
        bounded,
        format!(
            "(c) PHYSICAL_KV_BOUND_VIOLATION: N={n}: physical peak {peak_bytes} bytes ({} pages) vs \
             bound prefix {prefix_bytes} + {n} x {per_branch_suffix_bytes} = {bound_bytes} bytes, \
             logical {logical_bytes} bytes (for N > 1 the peak must be below logical){run_note}",
            tap.peak_physical_pages
        ),
    ));

    // (d) divergence and isolation (isolation proper is the controls above).
    let distinct: std::collections::HashSet<&Vec<u32>> = generated.iter().collect();
    let all_generated = generated.iter().all(|g| !g.is_empty());
    // Branches diverge only by their sampling seeds, so with few branches and
    // 4 tokens each, identical outputs are an ordinary coincidence (observed at
    // N=2, 2026-10-03). Divergence is required from DIVERGENCE_MIN_N branches
    // on, where every branch sampling the same 4 tokens would be a defect;
    // below that the distinct count is recorded, not judged.
    let divergence_required = n >= DIVERGENCE_MIN_N;
    let diverged = all_generated && (!divergence_required || distinct.len() >= 2);
    let mixed_batch = !divergence_required
        || tap.decode_batches.iter().any(|rows| {
            let fed: Vec<u32> = rows
                .iter()
                .filter(|(id, _)| branches.contains(id))
                .map(|(_, t)| *t)
                .collect();
            fed.len() >= 2 && fed.iter().any(|t| *t != fed[0])
        });
    checks.push(check(
        "d_isolation",
        diverged && mixed_batch && parity_pass,
        format!(
            "(d) BRANCH_ISOLATION_VIOLATION: N={n}: every branch produced tokens {all_generated}, \
             {} distinct outputs among {n} (need >= 2 when N >= {DIVERGENCE_MIN_N}), a decode batch mixed different \
             branch inputs {mixed_batch}, control parity {parity_pass}{run_note}",
            distinct.len()
        ),
    ));

    // (e) everything reclaimed.
    let backend_sequences = tap.inner.sequences.len();
    let backend_pending = tap.inner.pending_prefill_token.len();
    let backend_sampling = std::iter::once(root)
        .chain(branches.iter().copied())
        .filter(|id| tap.inner.sampling_params(*id).is_some())
        .count();
    let kv_allocated = kv.read().allocated_block_count();
    let arena_active = spine.arena.active_count();
    let branch_worlds = swarm
        .branch_worlds
        .iter()
        .filter(|w| spine.world_store.get_world(**w).is_some())
        .count();
    let scheduler_live = spine.scheduler.running_count() + spine.scheduler.waiting_count();
    let finished = spine.scheduler.metrics().finished_requests as usize;
    checks.push(check(
        "e_reclaimed",
        run_ok
            && finished == n
            && backend_sequences == 0
            && backend_pending == 0
            && backend_sampling == 0
            && kv_allocated == 0
            && arena_active == 0
            && branch_worlds == 0
            && scheduler_live == 0
            && spine.swarm_manager.active_swarm_count() == 0,
        format!(
            "(e) RECLAIM_VIOLATION: N={n}: finished {finished}/{n}; backend sequences \
             {backend_sequences}, pending prefill tokens {backend_pending}, sampling entries \
             {backend_sampling}; KV allocated blocks {kv_allocated}; arena active {arena_active}; \
             branch worlds {branch_worlds}; scheduler live {scheduler_live}; active swarms {}{run_note}",
            spine.swarm_manager.active_swarm_count()
        ),
    ));

    let pass = checks.iter().all(|c| c.pass);
    let prefill_secs = tap.prefill_wall.as_secs_f64();
    let ttft_ms: Option<f64> = tap.first_branch_token.map(|d| d.as_secs_f64() * 1e3);
    let prefill_tps: Option<f64> = if prefill_secs > 0.0 {
        Some(root_prefilled as f64 / prefill_secs)
    } else {
        None
    };
    let pool_bytes = plan.total_blocks * block_bytes;
    let parity_result = if parity_pass { "PASS" } else { "FAIL" };
    let verdict = if pass { "PASS" } else { "FAIL" };
    let checks_json: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| serde_json::json!({ "name": c.name, "pass": c.pass }))
        .collect();
    let mut receipt = sh.common.clone();
    let obj = receipt.as_object_mut().expect("common receipt object");
    let fields = serde_json::json!({
        "branch_count": n,
        "max_tokens_per_branch": MAX_TOKENS_PER_BRANCH,
        "backend": backend_name,
        "prompt_chunks": root_chunks.len(),
        "ttft_ms": ttft_ms,
        "ttft_definition": "launch_swarm call to the end of the first execute_step that emitted a branch token",
        "prefill_tokens_per_s": prefill_tps,
        "physical_kv_bytes": physical_bytes,
        "physical_kv_bytes_peak": peak_bytes,
        "logical_kv_bytes": logical_bytes,
        "prefix_bytes": prefix_bytes,
        "per_branch_suffix_bytes": per_branch_suffix_bytes,
        "physical_kv_bytes_bound": bound_bytes,
        "bound_formula": "prefix_bytes + branch_count x per_branch_suffix_bytes",
        "kv_block_bytes": block_bytes,
        "kv_source": "KvMetrics physical_pages/logical_pages x pool block_bytes at the first decode batch; peak = max physical_pages after each decode step",
        "pool": {
            "total_blocks": plan.total_blocks,
            "pool_bytes": pool_bytes,
            "prefix_blocks": plan.prefix_blocks,
            "per_branch_suffix_blocks": plan.per_branch_suffix_blocks,
            "watermark_blocks": plan.watermark_blocks,
            "slack_blocks": POOL_SLACK_BLOCKS,
            "max_batch_size": max_batch,
            "host_mem_total_bytes": mem_total_bytes(),
            "headroom_reason": "scheduler admission (crates/aien-scheduler/src/lib.rs build_scheduled_batch step 3) asks AienKvManager::incremental_blocks_needed (GB10-2), so a fork child needs only its copy-on-write block at admission; the pool holds prefix + N x private suffix blocks + watermark, no whole-prompt headroom and no slack.",
        },
        "parity": {
            "result": parity_result,
            "max_abs": parity.max_abs,
            "max_rel": parity.max_rel,
            "tolerance_abs": tol.abs,
            "tolerance_rel": tol.rel,
            "relative_error_epsilon": tol.rel_eps,
            "compared_values": parity.compared_values,
            "tolerance_source": "docs/campaigns/c1/acceptance.toml [numerics.noise_floor.tinyllama]",
            "sampled_controls": sampled_ids.len(),
            "sampled_branch_indices": sampled_idx,
            "sampling_rule": "all branches when N <= 8, else 8 evenly spaced indices i*(N-1)/7 incl. first and last; every branch's step 1 is compared against the root continuation",
            "step1_all_branches": { "compared_values": step1.compared_values, "violations": step1.violations, "max_abs": step1.max_abs },
        },
        "distinct_outputs": distinct.len(),
        "generated_tokens": generated,
        "spine_steps": steps,
        "step_guard": step_guard,
        "step_guard_hit": guard_hit,
        "starved": starved,
        "run_error": run_error,
        "checks": checks_json,
        "verdict": verdict,
    });
    for (k, v) in fields.as_object().expect("fields").iter() {
        obj.insert(k.clone(), v.clone());
    }
    RunResult {
        receipt,
        checks,
        pass,
    }
}

fn write_receipt(path: &Path, receipts: &[serde_json::Value]) {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("receipt directory");
        }
    }
    let line = serde_json::to_string(receipts).expect("receipt JSON");
    std::fs::write(path, format!("{line}\n")).expect("write receipt");
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "PREFILL-E2E-1 real-checkpoint gate at N = 1/8/32/128/500: run with --ignored and \
            AIEN_E2E_CHECKPOINT + AIEN_E2E1_RECEIPT set (forge on the Spark)"]
async fn prefill_e2e1_real_checkpoint_n_branch_gate() {
    let checkpoint = required_env("AIEN_E2E_CHECKPOINT");
    let receipt_path = PathBuf::from(required_env("AIEN_E2E1_RECEIPT"));

    let paths = checkpoint_paths(&checkpoint);
    let model_sha256 = sha256_file_hex(&paths.model);
    let tokenizer_sha256 = sha256_file_hex(&paths.tokenizer);
    let weights = strict_load(&paths);
    let model_id = weights.config.model_id.clone();
    let tokenizer = TinyLlamaTokenizer::from_file(&paths.tokenizer).unwrap_or_else(|e| {
        panic!(
            "STRICT_CHECKPOINT_VIOLATION: tokenizer {} failed to parse: {e:?}",
            paths.tokenizer.display()
        )
    });
    let tol = acceptance_tolerance();
    assert_eq!(
        tol.floor_status, "measured",
        "PARITY_TOLERANCE: [numerics.noise_floor.tinyllama] status must be measured"
    );
    assert_eq!(
        tol.floor_model_sha256, model_sha256,
        "PARITY_TOLERANCE: the frozen TinyLlama floor was measured on a different checkpoint"
    );
    assert!(
        tol.abs == 0.0 && tol.rel == 0.0,
        "PARITY_TOLERANCE: the frozen tolerance is exact equality (0.0); got abs {:e} rel {:e}",
        tol.abs,
        tol.rel
    );

    let prompt_text = TinyLlamaTokenizer::format_prompt(Some(SYSTEM_PROMPT), USER_PROMPT);
    let prompt = tokenizer.encode(&prompt_text).expect("encode root prompt");
    let prompt_len = prompt.len();
    assert!(
        prompt_len > PREFILL_CHUNK,
        "PROMPT_LENGTH: root prompt is {prompt_len} tokens; it must exceed the {PREFILL_CHUNK}-token \
         prefill chunk so the chunk continuation runs"
    );
    let prompt_digest = {
        let mut h = Sha256::new();
        for t in &prompt {
            h.update(t.to_le_bytes());
        }
        hex(&h.finalize())
    };
    println!("PREFILL_E2E1 root prompt: {prompt_len} tokens");

    let common = serde_json::json!({
        "gate": "PREFILL-E2E-1",
        "model": { "path": paths.model.display().to_string(), "id": model_id },
        "model_sha256": model_sha256,
        "tokenizer": { "path": paths.tokenizer.display().to_string() },
        "tokenizer_sha256": tokenizer_sha256,
        "prompt_tokens": prompt_len,
        "prompt_digest_sha256": prompt_digest,
        "prompt_digest_encoding": "sha256 over the token ids as little-endian u32",
        "hardware": hardware(),
        "kernel_exec_count": serde_json::Value::Null,
        "fallback_count": serde_json::Value::Null,
        "counters_reason": "ReferenceCpuBackend keeps no kernel/fallback counters; they exist only \
                            on OmegaGb10Backend (CPU path per the 2026-10-01 queen ruling)",
        "commit_sha": commit_sha(),
        "commit_sha_source": "git rev-parse HEAD in CARGO_MANIFEST_DIR at runtime",
        "gb10": "NOT_RUN",
        "max_comparison": "NOT_RUN",
    });
    let shared = Shared {
        weights: &weights,
        prompt: &prompt,
        tol: &tol,
        common: &common,
    };

    let mut receipts: Vec<serde_json::Value> = Vec::new();
    let mut results: Vec<(usize, Vec<Check>)> = Vec::new();
    let mut failed_at: Option<usize> = None;
    for &n in &BRANCH_COUNTS {
        if let Some(bad) = failed_at {
            receipts.push(serde_json::json!({
                "gate": "PREFILL-E2E-1",
                "branch_count": n,
                "verdict": "NOT_RUN",
                "not_run_after_failure": bad,
            }));
            continue;
        }
        println!("PREFILL_E2E1 running N = {n}");
        let run = run_one(n, &shared).await;
        println!(
            "PREFILL_E2E1_RECEIPT N={n} {}",
            serde_json::to_string(&run.receipt).expect("receipt JSON")
        );
        receipts.push(run.receipt);
        if !run.pass {
            failed_at = Some(n);
        }
        results.push((n, run.checks));
        // Partial receipt after every N so a later crash keeps the evidence.
        write_receipt(&receipt_path, &receipts);
    }
    write_receipt(&receipt_path, &receipts);

    // Assert every check, in order, only after the receipt is on disk.
    for (_n, checks) in &results {
        for c in checks {
            assert!(c.pass, "{}", c.message);
        }
    }
    assert!(
        failed_at.is_none(),
        "PREFILL_E2E1 failed at N = {failed_at:?}"
    );
}
