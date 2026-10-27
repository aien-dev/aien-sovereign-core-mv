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

//! PREFILL-E2E-2: append 24 tokens to an already prefilled context that is
//! shared with a witness, recompute only the new part, fork two branches off
//! the longer context and decode them (TinyLlama, CPU, strict checkpoint).
//!
//! `#[ignore]`d for normal CI. Run with `--ignored`; it FAILS (never skips)
//! when `AIEN_E2E_CHECKPOINT` or `AIEN_E2E2_RECEIPT` is unset.
//!
//! `AIEN_E2E_CHECKPOINT` is the TinyLlama-1.1B-Chat-v1.0 directory (holding
//! `model.safetensors` and `tokenizer.json`) or the `.safetensors` file.
//! `AIEN_E2E2_RECEIPT` is the path of the JSON receipt (one object).
//!
//! Level of the test: the backend, driven with hand-built `ScheduledBatch`
//! values through `execute_step` for every prefill step (two chunks of 128 and
//! 89 tokens, then the append). There is no scheduler and no spine step. The
//! KV manager calls the scheduler makes around a prefill (`allocate_sequence`
//! for the whole prompt, `mark_prefill_pending`, `complete_prefill`, then
//! `fork_prefilled` plus the backend fork hook) are made here by hand in the
//! same order (crates/aien-scheduler/src/lib.rs `build_scheduled_batch` and
//! `step`). Decode steps call `forward_decode_batch_with_logits`, the inherent
//! method that `execute_step` itself reaches through `forward_decode_batch`
//! (both end in `forward_decode_batch_impl`), because `execute_step` does not
//! return logits and the gate compares full logit vectors at tolerance 0.0.
//!
//! Flow: the 217-token prompt is prefilled in two chunks and fenced. A witness
//! sequence W is forked from it BEFORE the append, so the root's partial tail
//! block (block 13, 9 of 16 slots used) is shared. The root then receives ONE
//! more prefill request for the same sequence id holding only the 24 new
//! tokens. That append starts mid-block (block 13, slot 9) and crosses two
//! block boundaries, ending at 241 tokens in 16 blocks. Its first write must
//! copy-on-write the shared tail (crates/aien-kv-cache/src/lib.rs
//! `append_token_with_slot`). Two branches are then forked off the 241-token
//! root and decode 4 tokens each, in one batch together with W. Branch 2 is
//! teacher forced to its second best token after every step so the two
//! branches really diverge, which is what lets the isolation checks bite.
//!
//! Controls: a fresh UNPAGED `NativeTransformerBackend` prefills all 241 tokens
//! in one shot (the E1 `control_after_prompt` pattern) and decodes the same 4
//! steps per branch with the branch's own fed tokens; a second unpaged
//! sequence prefills only the 217-token prompt and is the control for W.
//! Tolerance is 0.0 (docs/campaigns/c1/acceptance.toml
//! `[numerics.noise_floor.tinyllama]`), asserted explicitly.
//!
//! Counting backend: a `TensorBackend` wrapper that forwards every method to
//! `ReferenceCpuBackend` (so the arithmetic is bit identical) and records the
//! position of every `apply_rope` call. It is injected through the
//! `Arc<dyn TensorBackend>` parameter of `build_shared_kv_runtime`, so no
//! runtime code changes. During the append the recorded positions must be
//! exactly 217..=240, each once per layer.
//!
//! Checks (each has its own marker in the failure message, all failing
//! messages are printed and the receipt is written before any assert fires):
//! a_kv_equals_control KV_PARITY_VIOLATION, b_logits_and_tokens
//! LOGITS_TOKENS_VIOLATION, c_blocks OLD_BLOCKS_REWRITTEN_VIOLATION,
//! d_only_new_computed RECOMPUTE_ALL_VIOLATION, e_isolation
//! BRANCH_ISOLATION_VIOLATION, e_cow_witness COW_VIOLATION,
//! f_no_pool_exhaustion POOL_EXHAUSTION_VIOLATION (or APPEND_ERROR_VIOLATION
//! when the append returns Err), g_reclaimed RECLAIM_VIOLATION, h_fence
//! FENCE_VIOLATION (carried over from E1).
//!
//! Evidence label: host CPU only (ReferenceCpuBackend, FP32). Not a GB10 GPU
//! result and not a MAX comparison; both are recorded as NOT_RUN.

#![recursion_limit = "512"]

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use aien_inference_abi::tensor::sample_argmax;
use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, FinishReason, ModelConfig, NativeTransformerBackend,
    PrefillState, ReferenceCpuBackend, SamplingParams, ScheduledBatch, SequenceRequest,
    TensorBackend, TinyLlamaTokenizer, TransformerWeights,
};
use aien_kv_cache::SharedKvManager;
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing, SHARED_KV_DTYPE};
use aien_scheduler::SchedulerConfig;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

/// Prompt length the block arithmetic below is built for (E1 prompt).
const PROMPT_TOKENS: usize = 217;
/// Tokens appended to the prefilled context in ONE prefill request.
const APPENDED_TOKENS: usize = 24;
const TOTAL_TOKENS: usize = PROMPT_TOKENS + APPENDED_TOKENS;
/// Decode budget per branch (and for the witness).
const DECODE_STEPS: usize = 4;
/// The daemon's prefill chunk size (crates/aien-cli/src/commands.rs).
const PREFILL_CHUNK: usize = 128;
/// Extra free blocks on top of the exact need.
const POOL_SLACK_BLOCKS: usize = 3;

/// Sequence ids in the paged runtime.
const ROOT: u64 = 100;
const WITNESS: u64 = 101;
const BRANCH_1: u64 = 102;
const BRANCH_2: u64 = 103;
/// Never gets a table; target of the fork-before-fence refusal probe.
const SCRATCH: u64 = 999;
const BRANCHES: [u64; 2] = [BRANCH_1, BRANCH_2];
/// Sequence ids inside the single unpaged control backend.
const CTL_ROOT: u64 = 1;
const CTL_B1: u64 = 2;
const CTL_B2: u64 = 3;
const CTL_W: u64 = 4;

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

/// Fixed text for the appended span. It is tokenized without special tokens and
/// the first APPENDED_TOKENS ids are used (the text is longer than that).
const APPEND_TEXT: &str = "Please also list the tools the carpenter needs on the bench for \
this job: a saw, a drill, a sander and a screwdriver, and say which one is used first and why.";

// ---------------------------------------------------------------------------
// Environment, strict checkpoint, acceptance tolerance (copied from E1)
// ---------------------------------------------------------------------------

struct CheckpointPaths {
    model: PathBuf,
    tokenizer: PathBuf,
}

fn required_env(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => panic!(
            "PREFILL_E2E2_ENV: {name} is unset; this gate never skips (set it and rerun with \
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
// Counting tensor backend
// ---------------------------------------------------------------------------

/// Forwards every `TensorBackend` method to `ReferenceCpuBackend` and records
/// the position of every `apply_rope` call. Nothing else is changed, so the
/// arithmetic is bit identical to the plain reference backend.
struct CountingBackend {
    inner: ReferenceCpuBackend,
    rope_positions: Mutex<Vec<usize>>,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            inner: ReferenceCpuBackend::new(),
            rope_positions: Mutex::new(Vec::new()),
        }
    }

    /// Returns the positions recorded since the last call and clears them.
    fn take_rope_positions(&self) -> Vec<usize> {
        std::mem::take(&mut *self.rope_positions.lock())
    }
}

#[allow(clippy::too_many_arguments)]
impl TensorBackend for CountingBackend {
    fn name(&self) -> &'static str {
        "CountingBackend(ReferenceCpuBackend)"
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.inner.rmsnorm(out, x, weight, eps);
    }

    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        self.inner.rmsnorm_heads(x, weight, head_dim, eps);
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope: &aien_inference_abi::RopeParams,
    ) {
        self.rope_positions.lock().push(pos);
        self.inner
            .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, rope);
    }

    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        self.inner.matmul_vec(out, x, weight, out_dim, in_dim);
    }

    fn matmul_batch(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        self.inner
            .matmul_batch(out, x, weight, batch_size, in_dim, out_dim);
    }

    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        self.inner.swiglu(out, gate, up);
    }

    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.inner.gqa_attention(
            out,
            q,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn paged_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_ids: &[usize],
        context_len: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.inner.paged_attention(
            out,
            q,
            pool,
            block_ids,
            context_len,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn paged_attention_batch(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_tables: &[i32],
        context_lens: &[i32],
        max_blocks_per_seq: usize,
        num_seqs: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.inner.paged_attention_batch(
            out,
            q,
            pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_seqs,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        self.inner
            .compute_logits(logits, hidden, embed_weight, vocab_size, hidden_dim);
    }
}

// ---------------------------------------------------------------------------
// Pool sizing
// ---------------------------------------------------------------------------

struct PoolPlan {
    block_size: usize,
    prompt_blocks: usize,
    final_blocks: usize,
    append_new_blocks: usize,
    branch_cow_blocks: usize,
    needed_blocks: usize,
    slack_blocks: usize,
    total_blocks: usize,
}

/// Blocks the pool needs for the whole run, derived (not guessed):
///
/// The prompt holds `ceil(217 / 16) = 14` blocks. The append reaches 241
/// tokens = 16 blocks. Its first write copies the shared partial tail block
/// (one new block, crates/aien-kv-cache/src/lib.rs `append_token_with_slot`
/// copy-on-write path), and the two block boundaries it crosses need
/// `16 - 14 = 2` fresh blocks, so the append allocates 3 blocks. Each of the
/// two branches copies the shared 1-token tail block once on its first
/// decode (2 blocks). The witness decodes in place in its own original tail
/// block (the root already copied away from it), so it needs none. Total
/// `14 + 3 + 2 = 19`, plus a small slack.
fn pool_plan(block_size: usize) -> PoolPlan {
    let prompt_blocks = PROMPT_TOKENS.div_ceil(block_size);
    let final_blocks = TOTAL_TOKENS.div_ceil(block_size);
    let append_new_blocks = final_blocks - prompt_blocks + 1;
    let branch_cow_blocks = BRANCHES.len();
    let needed_blocks = prompt_blocks + append_new_blocks + branch_cow_blocks;
    PoolPlan {
        block_size,
        prompt_blocks,
        final_blocks,
        append_new_blocks,
        branch_cow_blocks,
        needed_blocks,
        slack_blocks: POOL_SLACK_BLOCKS,
        total_blocks: needed_blocks + POOL_SLACK_BLOCKS,
    }
}

// ---------------------------------------------------------------------------
// KV helpers (all guard use stays inside these plain functions, never across
// an await point)
// ---------------------------------------------------------------------------

type KvGather = Vec<(Vec<f32>, Vec<f32>)>;

fn table_of(kv: &SharedKvManager, seq: u64) -> (Vec<usize>, usize) {
    kv.read()
        .get_block_table(seq)
        .map_or((Vec::new(), 0), |t| (t.block_ids.clone(), t.total_tokens))
}

/// SHA-256 over every K and V value (as raw bits) of one physical block, all
/// layers, all slots. Used to prove blocks are byte-identical before/after.
fn block_digest(
    kv: &SharedKvManager,
    block: usize,
    layers: usize,
    block_size: usize,
    kv_dim: usize,
) -> String {
    let guard = kv.read();
    let Some(pool) = guard.tensor_pool() else {
        return "no-pool".to_string();
    };
    let mut hasher = Sha256::new();
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    for layer in 0..layers {
        for slot in 0..block_size {
            pool.read_token_kv(block, layer, slot, &mut k, &mut v);
            for x in k.iter().chain(v.iter()) {
                hasher.update(x.to_bits().to_le_bytes());
            }
        }
    }
    hex(&hasher.finalize())
}

fn block_digests(
    kv: &SharedKvManager,
    ids: &[usize],
    layers: usize,
    block_size: usize,
    kv_dim: usize,
) -> Vec<String> {
    ids.iter()
        .map(|b| block_digest(kv, *b, layers, block_size, kv_dim))
        .collect()
}

/// Block bookkeeping the KV manager keeps for one physical block
/// (`aien_kv_cache::KvBlock`: `ref_count`, `num_tokens`, `is_shared`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockMeta {
    ref_count: usize,
    num_tokens: usize,
    is_shared: bool,
}

impl BlockMeta {
    fn json(self) -> serde_json::Value {
        serde_json::json!({
            "ref_count": self.ref_count,
            "num_tokens": self.num_tokens,
            "is_shared": self.is_shared,
        })
    }
}

fn block_meta(kv: &SharedKvManager, block: usize) -> Option<BlockMeta> {
    kv.read().get_block(block).map(|b| BlockMeta {
        ref_count: b.ref_count,
        num_tokens: b.num_tokens,
        is_shared: b.is_shared,
    })
}

/// Indexes whose digests differ between two snapshots of the same block list.
fn changed_indexes(a: &[String], b: &[String]) -> Vec<usize> {
    a.iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect()
}

/// PASS or FAIL label for one sub-result inside a check message.
fn tag(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn gather_all(kv: &SharedKvManager, seq: u64, layers: usize) -> Result<KvGather, String> {
    let mut out = Vec::with_capacity(layers);
    for layer in 0..layers {
        out.push(kv.read().gather_layer_kv(seq, layer)?);
    }
    Ok(out)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_bits(a: &[(Vec<f32>, Vec<f32>)], b: &[(Vec<f32>, Vec<f32>)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((ak, av), (bk, bv))| bits_eq(ak, bk) && bits_eq(av, bv))
}

/// Compares the K and V of paged sequence `seq` (gathered from the shared
/// pool) with the dense K/V of control sequence `ctl_id`, layer by layer.
fn compare_kv(
    parity: &mut Parity,
    what: &str,
    kv: &SharedKvManager,
    seq: u64,
    control: &NativeTransformerBackend,
    ctl_id: u64,
    tol: &Tolerance,
) -> Result<(), String> {
    let layers = control.weights.config.num_layers;
    let ctl = control
        .sequences
        .get(&ctl_id)
        .ok_or_else(|| format!("control sequence {ctl_id} missing"))?;
    for layer in 0..layers {
        let (k, v) = kv.read().gather_layer_kv(seq, layer)?;
        let cache = ctl
            .layers
            .get(layer)
            .ok_or_else(|| format!("control layer {layer} missing"))?;
        parity.compare(&format!("{what} K layer {layer}"), &k, &cache.flat_k, tol);
        parity.compare(&format!("{what} V layer {layer}"), &v, &cache.flat_v, tol);
    }
    Ok(())
}

/// Highest logit that is not the argmax and not a special token (ids 0..=2).
fn second_best(logits: &[f32]) -> u32 {
    let (best, _) = sample_argmax(logits);
    let mut idx = 0usize;
    let mut val = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if i as u32 == best || i <= 2 {
            continue;
        }
        if *v > val {
            val = *v;
            idx = i;
        }
    }
    idx as u32
}

fn token_digest(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    hex(&h.finalize())
}

/// One prefill request through `execute_step`, no scheduler. Returns the token
/// and logprob the backend sampled after the last token of the request.
async fn prefill_step(
    backend: &mut NativeTransformerBackend,
    id: u64,
    tokens: &[u32],
    sampling: &SamplingParams,
    step_id: u64,
) -> Result<(u32, f32), String> {
    let batch = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: id,
            prompt_tokens: tokens.to_vec(),
            sampling_params: sampling.clone(),
            arrival_time_ns: 0,
            priority: 1,
        }],
        decode_requests: Vec::new(),
        block_tables: HashMap::new(),
        step_id,
    };
    let (outputs, _metrics) = backend.execute_step(&batch).await?;
    for out in outputs {
        if let DecodeOutput::Token {
            request_id,
            token_id,
            logprob,
        } = out
        {
            if request_id == id {
                return Ok((token_id, logprob.unwrap_or(f32::NAN)));
            }
        }
    }
    Err(format!(
        "execute_step returned no sampled token for request {id}"
    ))
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// What the two forked branches did to the root's blocks during the FIRST
/// decode step, read at three moments: right after the branch forks (before any
/// decode write), right after the first decode step, and at the end of the
/// run. The end-of-run facts alone cannot see a branch that wrote into the
/// shared 1-token tail block on step 1 (the second step copies the block once
/// it holds 2 tokens, so the table ids look private again by the end), so the
/// step-1 snapshot is what makes copy-on-write failures visible. Defaults mean
/// "not captured" and every check that needs them then fails.
#[derive(Default)]
struct Isolation {
    /// Table entry `final_blocks - 1` of the root, branch 1 and branch 2 right
    /// after the first decode step.
    root_tail: Option<usize>,
    b1_tail: Option<usize>,
    b2_tail: Option<usize>,
    /// Bookkeeping of the root's tail block after the forks, after step 1 and
    /// at the end.
    tail_meta_before: Option<BlockMeta>,
    tail_meta_step1: Option<BlockMeta>,
    tail_meta_end: Option<BlockMeta>,
    /// SHA-256 of the raw K/V (every layer, every slot, K and V) of each block
    /// of the root's table, in table order, at the same three moments.
    root_digests_before: Vec<String>,
    root_digests_step1: Vec<String>,
    root_digests_end: Vec<String>,
}

struct Reclaim {
    allocated: usize,
    free: usize,
    kv_sequences: usize,
    physical_pages: usize,
    backend_sequences: usize,
    backend_pending: usize,
    backend_sampling: usize,
    free_errors: Vec<String>,
}

/// Everything the run observed. Filled stage by stage; a stage that fails
/// leaves the later fields at their defaults and every check that needs them
/// then fails (never panics).
#[derive(Default)]
struct Evidence {
    // Fence (carried over from E1).
    state_after_alloc: Option<PrefillState>,
    state_after_pending: Option<PrefillState>,
    state_after_chunk1: Option<PrefillState>,
    state_after_chunk2: Option<PrefillState>,
    state_after_fence: Option<PrefillState>,
    state_root_after_witness_fork: Option<PrefillState>,
    state_root_after_append: Option<PrefillState>,
    chunk_lens: Vec<usize>,
    early_kv_fork_refused: bool,
    early_backend_fork_refused: bool,
    witness_forked: bool,
    pending_217: Option<u32>,
    pending_241: Option<u32>,
    sampled_217: Option<(u32, f32)>,
    sampled_241: Option<(u32, f32)>,
    // Before the append (root and witness share these 14 blocks).
    root_ids_before: Vec<usize>,
    digests_before: Vec<String>,
    block_bytes: usize,
    physical_before: usize,
    cow_before: usize,
    // After the append.
    append_error: Option<String>,
    append_positions: Vec<usize>,
    prefill_rope_calls: usize,
    root_ids_after: Vec<usize>,
    root_total_after: usize,
    witness_ids_after: Vec<usize>,
    witness_total_after: usize,
    digests_after: Vec<String>,
    physical_after: usize,
    cow_after: usize,
    free_after_append: usize,
    root_kv_after: KvGather,
    // Controls.
    ctl_217: Option<(u32, f32)>,
    ctl_241: Option<(u32, f32)>,
    root_kv: Parity,
    witness_kv: Parity,
    // Decode.
    fed: HashMap<u64, Vec<u32>>,
    sampled: HashMap<u64, Vec<u32>>,
    rows: HashMap<u64, Vec<Vec<f32>>>,
    preempted: Vec<(u64, usize)>,
    logits_b: Parity,
    logits_w: Parity,
    token_mismatch: Vec<u64>,
    branch_kv: Parity,
    witness_final_kv: Parity,
    // First decode step snapshots (see `Isolation`).
    iso: Isolation,
    // After decode.
    root_stable: bool,
    b1_ids: Vec<usize>,
    b2_ids: Vec<usize>,
    b1_total: usize,
    b2_total: usize,
    physical_final: usize,
    free_final: usize,
    // Reclaim.
    reclaim: Option<Reclaim>,
    // Timings.
    prefill_ms: f64,
    append_ms: f64,
    control_prefill_ms: f64,
}

/// What the stages need from the outside.
struct Env<'a> {
    weights: &'a TransformerWeights,
    prompt: &'a [u32],
    appended: &'a [u32],
    tol: &'a Tolerance,
    counting: Arc<CountingBackend>,
    plan: &'a PoolPlan,
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// Runs every stage in order and fills `ev`. Returns Err at the first stage
/// that cannot continue; the caller still evaluates all checks on what was
/// recorded and writes the receipt.
async fn run_stages(env: &Env<'_>, ev: &mut Evidence) -> Result<(), String> {
    // Fresh controller state (RuntimeController persists operation ids).
    let state_dir = tempfile::tempdir().map_err(|e| format!("state dir: {e}"))?;
    std::env::set_var("AIEN_RUNTIME_STATE_DIR", state_dir.path());

    let cfg = &env.weights.config;
    let block_size = cfg.block_size;
    let layers = cfg.num_layers;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    // The counting wrapper goes in through the Arc<dyn TensorBackend>
    // parameter: no runtime change.
    let tensor: Arc<dyn TensorBackend> = env.counting.clone();
    let scheduler = SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 4096,
        max_prefill_tokens: 8192,
        prefill_chunk_size: PREFILL_CHUNK,
        chunk_prefill: true,
        watermark_blocks: 2,
    };
    let (spine, mut backend) = build_shared_kv_runtime(
        env.weights.clone(),
        tensor,
        scheduler,
        SharedKvSizing {
            arena_capacity: 16,
            total_blocks: env.plan.total_blocks,
        },
    )?;
    let kv = spine.kv_manager.clone();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 64,
        stop_token_ids: Vec::new(),
    };

    // Stage 1: allocate the whole prompt, prefill in two chunks, fence. The
    // scheduler does exactly this order (allocate_sequence, mark_prefill_pending,
    // one execute_step per chunk, complete_prefill after the last chunk).
    let t_prefill = Instant::now();
    kv.write().allocate_sequence(ROOT, env.prompt)?;
    ev.state_after_alloc = kv.read().prefill_state(ROOT);
    kv.write()
        .mark_prefill_pending(ROOT)
        .map_err(|e| format!("mark_prefill_pending: {e}"))?;
    ev.state_after_pending = kv.read().prefill_state(ROOT);

    let (first, second) = env.prompt.split_at(PREFILL_CHUNK);
    ev.chunk_lens = vec![first.len(), second.len()];
    prefill_step(&mut backend, ROOT, first, &sampling, 1).await?;
    ev.state_after_chunk1 = kv.read().prefill_state(ROOT);
    let sampled_217 = prefill_step(&mut backend, ROOT, second, &sampling, 2).await?;
    ev.state_after_chunk2 = kv.read().prefill_state(ROOT);
    ev.sampled_217 = Some(sampled_217);
    ev.pending_217 = backend.pending_prefill_token.get(&ROOT).copied();

    // A fork before the fence must be refused by the KV gate and by the
    // backend hook, and must leave nothing behind.
    let early_kv = kv.write().fork_prefilled(ROOT, SCRATCH);
    let scratch_table = kv.read().get_block_table(SCRATCH).is_some();
    ev.early_kv_fork_refused = early_kv.is_err() && !scratch_table;
    let early_backend = backend.fork_sequence(ROOT, SCRATCH);
    ev.early_backend_fork_refused =
        early_backend.is_err() && !backend.sequences.contains_key(&SCRATCH);

    kv.write()
        .complete_prefill(ROOT)
        .map_err(|e| format!("complete_prefill: {e}"))?;
    ev.state_after_fence = kv.read().prefill_state(ROOT);
    ev.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    ev.prefill_rope_calls = env.counting.take_rope_positions().len();

    // Stage 2: fork the witness BEFORE the append (copy-on-write option A), so
    // the root's partial tail block is shared when the append starts.
    let (ids_before, _total_before) = table_of(&kv, ROOT);
    ev.root_ids_before = ids_before.clone();
    ev.digests_before = block_digests(&kv, &ids_before, layers, block_size, kv_dim);
    ev.block_bytes = kv.read().tensor_pool().map_or(0, |p| p.block_bytes());
    kv.write()
        .fork_prefilled(ROOT, WITNESS)
        .map_err(|e| format!("witness fork_prefilled: {e}"))?;
    backend.fork_sequence(ROOT, WITNESS)?;
    ev.witness_forked = true;
    ev.state_root_after_witness_fork = kv.read().prefill_state(ROOT);
    ev.physical_before = kv.read().metrics().physical_pages;
    ev.cow_before = kv.read().cow_faults();

    // Stage 3: ONE more prefill request for the same sequence id holding only
    // the 24 new tokens. The backend continues at offset = tokens already in
    // the sequence (transformer_backend.rs `prefill_prompt_layer_by_layer_paged`).
    let t_append = Instant::now();
    let appended = prefill_step(&mut backend, ROOT, env.appended, &sampling, 3).await;
    ev.append_ms = t_append.elapsed().as_secs_f64() * 1e3;
    ev.append_positions = env.counting.take_rope_positions();
    match appended {
        Ok(sample) => ev.sampled_241 = Some(sample),
        Err(e) => {
            ev.append_error = Some(e.clone());
            return Err(format!("append execute_step failed: {e}"));
        }
    }
    ev.pending_241 = backend.pending_prefill_token.get(&ROOT).copied();
    let (ids_after, total_after) = table_of(&kv, ROOT);
    ev.root_ids_after = ids_after;
    ev.root_total_after = total_after;
    let (witness_ids, witness_total) = table_of(&kv, WITNESS);
    ev.witness_ids_after = witness_ids;
    ev.witness_total_after = witness_total;
    ev.digests_after = block_digests(&kv, &ids_before, layers, block_size, kv_dim);
    ev.physical_after = kv.read().metrics().physical_pages;
    ev.cow_after = kv.read().cow_faults();
    ev.free_after_append = kv.read().free_block_count();
    ev.state_root_after_append = kv.read().prefill_state(ROOT);
    ev.root_kv_after = gather_all(&kv, ROOT, layers)?;

    // Stage 4: unpaged one-shot controls. The 241-token control is the E1
    // control_after_prompt pattern with the append folded into the prompt; the
    // 217-token control is the witness control.
    let t_control = Instant::now();
    let mut control = NativeTransformerBackend::new(env.weights.clone());
    // The control is dense (no pool) but is the reference for a pool of
    // SHARED_KV_DTYPE, so it rounds K/V at birth exactly as the paged backend does.
    control.kv_dtype = SHARED_KV_DTYPE;
    if control.kv_manager.is_some() {
        return Err("control must be unpaged".to_string());
    }
    let mut full = env.prompt.to_vec();
    full.extend_from_slice(env.appended);
    let logits_241 = control.prefill_sequence(CTL_ROOT, &full)?;
    let logits_217 = control.prefill_sequence(CTL_W, env.prompt)?;
    ev.control_prefill_ms = t_control.elapsed().as_secs_f64() * 1e3;
    ev.ctl_241 = Some(sample_argmax(&logits_241));
    ev.ctl_217 = Some(sample_argmax(&logits_217));
    compare_kv(
        &mut ev.root_kv,
        "root",
        &kv,
        ROOT,
        &control,
        CTL_ROOT,
        env.tol,
    )?;
    compare_kv(
        &mut ev.witness_kv,
        "witness",
        &kv,
        WITNESS,
        &control,
        CTL_W,
        env.tol,
    )?;

    // Branch and witness controls start from the same fed tokens as the paged
    // sequences: the root's pending token (241 context) and the witness's
    // pending token (217 context), exactly what the fork hook gives them.
    let p217 = ev
        .pending_217
        .ok_or("root has no pending token after the fence")?;
    let p241 = ev
        .pending_241
        .ok_or("root has no pending token after the append")?;
    let base = control
        .sequences
        .get(&CTL_ROOT)
        .ok_or("control root sequence missing")?
        .clone();
    let mut ctl_b1 = base.clone();
    ctl_b1.tokens.push(p241);
    control.sequences.insert(CTL_B1, ctl_b1);
    let mut ctl_b2 = base;
    ctl_b2.tokens.push(p241);
    control.sequences.insert(CTL_B2, ctl_b2);
    control
        .sequences
        .get_mut(&CTL_W)
        .ok_or("control witness sequence missing")?
        .tokens
        .push(p217);

    // Stage 5: fork two branches off the 241-token root (KV gate first, then
    // the backend hook, as the swarm manager does) and decode 4 steps in one
    // batch together with the witness.
    for b in BRANCHES {
        kv.write()
            .fork_prefilled(ROOT, b)
            .map_err(|e| format!("branch {b} fork_prefilled: {e}"))?;
        backend.fork_sequence(ROOT, b)?;
    }
    // Isolation snapshot 1: the root's blocks exactly as the forks left them,
    // before any decode write. The tail block (table entry `final_blocks - 1`)
    // holds 1 token and is shared by the root and both branches here.
    let tail_pos = env.plan.final_blocks.saturating_sub(1);
    let root_blocks = ev.root_ids_after.clone();
    ev.iso.root_digests_before = block_digests(&kv, &root_blocks, layers, block_size, kv_dim);
    ev.iso.tail_meta_before = root_blocks.get(tail_pos).and_then(|b| block_meta(&kv, *b));
    let order = [BRANCH_1, BRANCH_2, WITNESS];
    let mut dead: Vec<u64> = Vec::new();
    for step in 1..=DECODE_STEPS {
        let live: Vec<u64> = order
            .iter()
            .copied()
            .filter(|id| !dead.contains(id))
            .collect();
        if live.is_empty() {
            break;
        }
        for id in &live {
            let last = backend
                .sequences
                .get(id)
                .and_then(|s| s.tokens.last().copied())
                .ok_or_else(|| format!("sequence {id} has no tokens before decode step {step}"))?;
            ev.fed.entry(*id).or_default().push(last);
        }
        let (outs, rows) = backend.forward_decode_batch_with_logits(&live)?;
        if step == 1 {
            // Isolation snapshot 2: straight after the first decode step, before
            // the second step can copy a block the first step wrote into.
            ev.iso.root_tail = table_of(&kv, ROOT).0.get(tail_pos).copied();
            ev.iso.b1_tail = table_of(&kv, BRANCH_1).0.get(tail_pos).copied();
            ev.iso.b2_tail = table_of(&kv, BRANCH_2).0.get(tail_pos).copied();
            ev.iso.tail_meta_step1 = root_blocks.get(tail_pos).and_then(|b| block_meta(&kv, *b));
            ev.iso.root_digests_step1 =
                block_digests(&kv, &root_blocks, layers, block_size, kv_dim);
        }
        let preempted: Vec<u64> = outs
            .iter()
            .filter_map(|o| match o {
                DecodeOutput::Finished {
                    request_id,
                    reason: FinishReason::Preempted,
                    ..
                } => Some(*request_id),
                _ => None,
            })
            .collect();
        let computed: Vec<u64> = live
            .iter()
            .copied()
            .filter(|id| !preempted.contains(id))
            .collect();
        if rows.len() != computed.len() {
            return Err(format!(
                "decode step {step}: {} logit rows for {} computed rows",
                rows.len(),
                computed.len()
            ));
        }
        for (id, row) in computed.iter().zip(rows) {
            let tok = backend
                .sequences
                .get(id)
                .and_then(|s| s.tokens.last().copied())
                .ok_or_else(|| format!("sequence {id} has no tokens after decode step {step}"))?;
            ev.sampled.entry(*id).or_default().push(tok);
            ev.rows.entry(*id).or_default().push(row);
        }
        for id in preempted {
            ev.preempted.push((id, step));
            dead.push(id);
        }
        // Teacher forcing: branch 2 continues with its second best token so
        // the two branches diverge from step 2 on.
        if step < DECODE_STEPS && computed.contains(&BRANCH_2) {
            let forced = ev
                .rows
                .get(&BRANCH_2)
                .and_then(|r| r.last())
                .map(|row| second_best(row));
            if let (Some(forced), Some(seq)) = (forced, backend.sequences.get_mut(&BRANCH_2)) {
                seq.tokens.pop();
                seq.tokens.push(forced);
            }
        }
    }

    // Stage 6: the controls decode the same steps with the same fed tokens and
    // every step's full logit vector is compared at tolerance 0.0.
    for (id, ctl_id, name) in [
        (BRANCH_1, CTL_B1, "branch1"),
        (BRANCH_2, CTL_B2, "branch2"),
        (WITNESS, CTL_W, "witness"),
    ] {
        let rows = ev.rows.get(&id).cloned().unwrap_or_default();
        let fed = ev.fed.get(&id).cloned().unwrap_or_default();
        let sampled = ev.sampled.get(&id).cloned().unwrap_or_default();
        for (k, row) in rows.iter().enumerate() {
            let (_outs, ctl_rows) = control.forward_decode_batch_with_logits(&[ctl_id])?;
            let ctl_row = ctl_rows
                .into_iter()
                .next()
                .ok_or_else(|| format!("control {name} step {} gave no logits", k + 1))?;
            let target = if id == WITNESS {
                &mut ev.logits_w
            } else {
                &mut ev.logits_b
            };
            target.compare(&format!("{name} step {}", k + 1), row, &ctl_row, env.tol);
            let ctl_tok = control
                .sequences
                .get(&ctl_id)
                .and_then(|s| s.tokens.last().copied());
            if sampled.get(k).copied() != ctl_tok {
                ev.token_mismatch.push(id);
            }
            if let Some(next) = fed.get(k + 1).copied() {
                if let Some(seq) = control.sequences.get_mut(&ctl_id) {
                    seq.tokens.pop();
                    seq.tokens.push(next);
                }
            }
        }
    }
    compare_kv(
        &mut ev.branch_kv,
        "branch1",
        &kv,
        BRANCH_1,
        &control,
        CTL_B1,
        env.tol,
    )?;
    compare_kv(
        &mut ev.branch_kv,
        "branch2",
        &kv,
        BRANCH_2,
        &control,
        CTL_B2,
        env.tol,
    )?;
    compare_kv(
        &mut ev.witness_final_kv,
        "witness-final",
        &kv,
        WITNESS,
        &control,
        CTL_W,
        env.tol,
    )?;

    // Stage 7: the root is untouched by everything the branches and the
    // witness did, and the tables look as copy-on-write says they must.
    let (root_ids_end, root_total_end) = table_of(&kv, ROOT);
    let root_end = gather_all(&kv, ROOT, layers)?;
    ev.root_stable = root_ids_end == ev.root_ids_after
        && root_total_end == ev.root_total_after
        && same_bits(&root_end, &ev.root_kv_after);
    // Isolation snapshot 3: the same block facts at the end of the run.
    ev.iso.tail_meta_end = root_blocks.get(tail_pos).and_then(|b| block_meta(&kv, *b));
    ev.iso.root_digests_end = block_digests(&kv, &root_blocks, layers, block_size, kv_dim);
    let (b1_ids, b1_total) = table_of(&kv, BRANCH_1);
    let (b2_ids, b2_total) = table_of(&kv, BRANCH_2);
    ev.b1_ids = b1_ids;
    ev.b1_total = b1_total;
    ev.b2_ids = b2_ids;
    ev.b2_total = b2_total;
    ev.physical_final = kv.read().metrics().physical_pages;
    ev.free_final = kv.read().free_block_count();

    // Stage 8: reclaim. The KV tables are freed the way the scheduler and the
    // swarm manager free them; the backend hook (trait method, which never
    // touches the KV manager) removes the per-request backend state.
    let mut free_errors: Vec<String> = Vec::new();
    for id in [ROOT, WITNESS, BRANCH_1, BRANCH_2] {
        if let Err(e) = kv.write().free_sequence(id) {
            free_errors.push(format!("{id}: {e}"));
        }
        AienInferenceBackend::release_sequence(&mut backend, id)?;
    }
    let backend_sampling = [ROOT, WITNESS, BRANCH_1, BRANCH_2]
        .iter()
        .filter(|id| backend.sampling_params(**id).is_some())
        .count();
    ev.reclaim = Some(Reclaim {
        allocated: kv.read().allocated_block_count(),
        free: kv.read().free_block_count(),
        kv_sequences: kv.read().active_sequence_count(),
        physical_pages: kv.read().metrics().physical_pages,
        backend_sequences: backend.sequences.len(),
        backend_pending: backend.pending_prefill_token.len(),
        backend_sampling,
        free_errors,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

fn parity_summary(p: &Parity) -> String {
    format!(
        "{} violations in {} values (max_abs {:e}, max_rel {:e}){}",
        p.violations,
        p.compared_values,
        p.max_abs,
        p.max_rel,
        p.first_violation
            .as_ref()
            .map(|f| format!(", first: {f}"))
            .unwrap_or_default()
    )
}

fn same_sample(a: Option<(u32, f32)>, b: Option<(u32, f32)>) -> bool {
    match (a, b) {
        (Some((ta, la)), Some((tb, lb))) => ta == tb && la.to_bits() == lb.to_bits(),
        _ => false,
    }
}

/// What the counting backend saw during the append.
struct PositionStats {
    min: Option<usize>,
    max: Option<usize>,
    count: usize,
    distinct: usize,
    per_layer_count: usize,
    /// Exactly the positions PROMPT_TOKENS..TOTAL_TOKENS, each once per layer.
    exact: bool,
}

fn position_stats(positions: &[usize], layers: usize) -> PositionStats {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for p in positions {
        *counts.entry(*p).or_default() += 1;
    }
    let expected: Vec<usize> = (PROMPT_TOKENS..TOTAL_TOKENS).collect();
    let keys: Vec<usize> = counts.keys().copied().collect();
    let exact = keys == expected && counts.values().all(|c| *c == layers);
    PositionStats {
        min: counts.keys().next().copied(),
        max: counts.keys().next_back().copied(),
        count: positions.len(),
        distinct: counts.len(),
        per_layer_count: positions.len().checked_div(layers).unwrap_or(0),
        exact,
    }
}

/// Verdicts on the first-decode-step snapshots (see `Isolation`). Both the
/// isolation check and the receipt read these, so they report the same facts.
struct FirstStep {
    /// Table entry the shared tail block sits at (`final_blocks - 1`).
    tail: usize,
    /// Tokens the root's tail block holds: 241 - 15 * 16 = 1.
    tail_tokens: usize,
    /// Right after step 1 branch 1, branch 2 and the root hold three distinct
    /// tail blocks, and the root's tail block id did not move.
    tails_private: bool,
    /// After the forks the tail block is shared by the root and both branches
    /// (ref_count 3, shared flag set) and holds `tail_tokens` tokens.
    meta_before_ok: bool,
    /// After step 1 both branches have copied away: the root is the sole owner
    /// (ref_count 1, shared flag clear) and nobody wrote into the block, so it
    /// still holds `tail_tokens` tokens. Same expectation at the end.
    meta_step1_ok: bool,
    meta_end_ok: bool,
    /// Every one of the three digest snapshots covers every root block.
    snapshots_complete: bool,
    /// Root block indexes whose raw K/V changed between the forks and step 1,
    /// and between the forks and the end.
    changed_step1: Vec<usize>,
    changed_end: Vec<usize>,
    blocks_same_step1: bool,
    blocks_same_end: bool,
    /// The root's tail block (all slots, all layers, K and V) is bit identical
    /// after step 1 and at the end.
    tail_snapshot_equal: bool,
}

fn first_step(plan: &PoolPlan, ev: &Evidence) -> FirstStep {
    let iso = &ev.iso;
    let tail = plan.final_blocks.saturating_sub(1);
    let tail_tokens = TOTAL_TOKENS.saturating_sub(tail * plan.block_size);
    let tails_private = matches!(
        (iso.b1_tail, iso.b2_tail, iso.root_tail),
        (Some(x), Some(y), Some(r)) if x != y && x != r && y != r
    ) && iso.root_tail == ev.root_ids_after.get(tail).copied();
    let shared = BlockMeta {
        ref_count: 1 + BRANCHES.len(),
        num_tokens: tail_tokens,
        is_shared: true,
    };
    let sole_owner = BlockMeta {
        ref_count: 1,
        num_tokens: tail_tokens,
        is_shared: false,
    };
    let snapshots_complete = iso.root_digests_before.len() == plan.final_blocks
        && iso.root_digests_step1.len() == plan.final_blocks
        && iso.root_digests_end.len() == plan.final_blocks;
    let changed_step1 = changed_indexes(&iso.root_digests_before, &iso.root_digests_step1);
    let changed_end = changed_indexes(&iso.root_digests_before, &iso.root_digests_end);
    let blocks_same_step1 = snapshots_complete && changed_step1.is_empty();
    let blocks_same_end = snapshots_complete && changed_end.is_empty();
    let tail_snapshot_equal =
        snapshots_complete && !changed_step1.contains(&tail) && !changed_end.contains(&tail);
    FirstStep {
        tail,
        tail_tokens,
        tails_private,
        meta_before_ok: iso.tail_meta_before == Some(shared),
        meta_step1_ok: iso.tail_meta_step1 == Some(sole_owner),
        meta_end_ok: iso.tail_meta_end == Some(sole_owner),
        snapshots_complete,
        changed_step1,
        changed_end,
        blocks_same_step1,
        blocks_same_end,
        tail_snapshot_equal,
    }
}

/// Builds the nine checks from the evidence. Every message carries its marker
/// and the numbers behind it. Nothing here can panic on missing evidence.
fn evaluate(env: &Env<'_>, ev: &Evidence) -> Vec<Check> {
    let cfg = &env.weights.config;
    let layers = cfg.num_layers;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let vocab = cfg.vocab_size;
    let plan = env.plan;
    let tail_idx = plan.prompt_blocks - 1;
    let mut checks: Vec<Check> = Vec::new();

    // (a) K/V of the root equals a one-shot 241-token control, the witness
    // equals a 217-token control, and both still equal their controls after
    // the branches and the witness decoded.
    let root_expected = layers * 2 * TOTAL_TOKENS * kv_dim;
    let witness_expected = layers * 2 * PROMPT_TOKENS * kv_dim;
    let branch_expected = BRANCHES.len() * layers * 2 * (TOTAL_TOKENS + DECODE_STEPS) * kv_dim;
    let witness_final_expected = layers * 2 * (PROMPT_TOKENS + DECODE_STEPS) * kv_dim;
    let a_root = ev.root_kv.pass() && ev.root_kv.compared_values == root_expected;
    let a_witness = ev.witness_kv.pass() && ev.witness_kv.compared_values == witness_expected;
    let a_branches = ev.branch_kv.pass() && ev.branch_kv.compared_values == branch_expected;
    let a_witness_final =
        ev.witness_final_kv.pass() && ev.witness_final_kv.compared_values == witness_final_expected;
    let pass_a = a_root && a_witness && a_branches && a_witness_final;
    checks.push(check(
        "a_kv_equals_control",
        pass_a,
        format!(
            "(a) KV_PARITY_VIOLATION: each sub-comparison below carries its own verdict; root \
             after append vs one-shot 241 control: {}, {} (expected {} values); witness vs 217 \
             control: {}, {} (expected {}); two branches after decode vs controls: {}, {} \
             (expected {}); witness after decode vs control: {}, {} (expected {})",
            tag(a_root),
            parity_summary(&ev.root_kv),
            root_expected,
            tag(a_witness),
            parity_summary(&ev.witness_kv),
            witness_expected,
            tag(a_branches),
            parity_summary(&ev.branch_kv),
            branch_expected,
            tag(a_witness_final),
            parity_summary(&ev.witness_final_kv),
            witness_final_expected
        ),
    ));

    // (b) the sampled tokens and every logit vector equal the controls.
    let b_expected = BRANCHES.len() * DECODE_STEPS * vocab;
    let w_expected = DECODE_STEPS * vocab;
    let b_sample_217 = same_sample(ev.sampled_217, ev.ctl_217);
    let b_sample_241 = same_sample(ev.sampled_241, ev.ctl_241);
    let b_branch_logits = ev.logits_b.pass() && ev.logits_b.compared_values == b_expected;
    let b_witness_logits = ev.logits_w.pass() && ev.logits_w.compared_values == w_expected;
    let b_tokens = ev.token_mismatch.is_empty();
    let pass_b = b_sample_217 && b_sample_241 && b_branch_logits && b_witness_logits && b_tokens;
    checks.push(check(
        "b_logits_and_tokens",
        pass_b,
        format!(
            "(b) LOGITS_TOKENS_VIOLATION: each sub-comparison below carries its own verdict; \
             sample after 217 {:?} vs control {:?}: {}; sample after 241 {:?} vs control {:?}: \
             {}; branch logits: {}, {} (expected {} values); witness logits: {}, {} (expected \
             {}); decode token mismatches (sequence ids) {:?}: {}",
            ev.sampled_217,
            ev.ctl_217,
            tag(b_sample_217),
            ev.sampled_241,
            ev.ctl_241,
            tag(b_sample_241),
            tag(b_branch_logits),
            parity_summary(&ev.logits_b),
            b_expected,
            tag(b_witness_logits),
            parity_summary(&ev.logits_w),
            w_expected,
            ev.token_mismatch,
            tag(b_tokens)
        ),
    ));

    // (c) the 14 old blocks are byte identical (raw K/V of every layer and
    // slot, hashed) after the append, and the 13 full ones are still the same
    // block ids in the root's table.
    let digests_same =
        ev.digests_before.len() == plan.prompt_blocks && ev.digests_before == ev.digests_after;
    let changed: Vec<usize> = ev
        .digests_before
        .iter()
        .zip(&ev.digests_after)
        .enumerate()
        .filter(|(_, (before, after))| before != after)
        .map(|(i, _)| i)
        .collect();
    let prefix_before = ev.root_ids_before.get(..tail_idx);
    let prefix_same = prefix_before.is_some() && prefix_before == ev.root_ids_after.get(..tail_idx);
    checks.push(check(
        "c_blocks",
        ev.root_ids_before.len() == plan.prompt_blocks && digests_same && prefix_same,
        format!(
            "(c) OLD_BLOCKS_REWRITTEN_VIOLATION: old block ids {:?}, root table after append {:?}; \
             {} digests before, {} after; old block indexes whose bytes changed {:?}; the first {} \
             block ids unchanged {}",
            ev.root_ids_before,
            ev.root_ids_after,
            ev.digests_before.len(),
            ev.digests_after.len(),
            changed,
            tail_idx,
            prefix_same
        ),
    ));

    // (d) only the 24 new positions were computed.
    let stats = position_stats(&ev.append_positions, layers);
    checks.push(check(
        "d_only_new_computed",
        stats.exact && stats.count == APPENDED_TOKENS * layers,
        format!(
            "(d) RECOMPUTE_ALL_VIOLATION: the append computed positions {:?}..={:?} ({} distinct, \
             {} rope calls, {} per layer); expected exactly {}..={} ({} positions) once per layer \
             ({} layers, {} calls)",
            stats.min,
            stats.max,
            stats.distinct,
            stats.count,
            stats.per_layer_count,
            PROMPT_TOKENS,
            TOTAL_TOKENS - 1,
            APPENDED_TOKENS,
            layers,
            APPENDED_TOKENS * layers
        ),
    ));

    // (e) branch isolation: the root is untouched by what the branches did,
    // the branches share the full prefix blocks, keep private tails, and really
    // diverged (so cross contamination would show).
    let fed_at = |id: u64, i: usize| ev.fed.get(&id).and_then(|f| f.get(i)).copied();
    let diverged = matches!(
        (fed_at(BRANCH_1, 1), fed_at(BRANCH_2, 1)),
        (Some(a), Some(b)) if a != b
    );
    let tail = plan.final_blocks - 1;
    let root_prefix = ev.root_ids_after.get(..tail);
    let prefix_shared = root_prefix.is_some()
        && ev.b1_ids.get(..tail) == root_prefix
        && ev.b2_ids.get(..tail) == root_prefix;
    let tails_private = matches!(
        (ev.b1_ids.get(tail), ev.b2_ids.get(tail), ev.root_ids_after.get(tail)),
        (Some(x), Some(y), Some(r)) if x != y && x != r && y != r
    );
    let branch_total = TOTAL_TOKENS + DECODE_STEPS;
    // The end-of-run facts above cannot see a branch that wrote into the shared
    // 1-token tail block on its first decode step (the second step copies the
    // block once it holds 2 tokens, so the ids look private again by the end,
    // and the root's own 241 positions never include the written slot). The
    // snapshots taken right after the forks and right after step 1 can.
    let fs = first_step(plan, ev);
    let pass_e = ev.root_stable
        && diverged
        && prefix_shared
        && tails_private
        && ev.b1_total == branch_total
        && ev.b2_total == branch_total
        && ev.physical_final == plan.needed_blocks
        && fs.tails_private
        && fs.meta_before_ok
        && fs.meta_step1_ok
        && fs.meta_end_ok
        && fs.blocks_same_step1
        && fs.blocks_same_end;
    checks.push(check(
        "e_isolation",
        pass_e,
        format!(
            "(e) BRANCH_ISOLATION_VIOLATION: root unchanged by branch decode {}; branches fed \
             different tokens at step 2 {} ({:?} vs {:?}); the first {} block ids shared with the \
             root {}; private tail blocks (branch 1 {:?}, branch 2 {:?}, root {:?}) {}; branch \
             totals {} and {} (expected {}); physical pages at the end {} (expected {}); first \
             decode step: tail block ids right after step 1 (branch 1 {:?}, branch 2 {:?}, root \
             {:?}) distinct and the root's id unmoved: {}; tail block (ref_count, tokens, shared \
             flag) after the forks {:?} (expected ref_count {}, {} tokens, shared): {}, after step \
             1 {:?} (expected the root as sole owner: ref_count 1, {} tokens, not shared): {}, at \
             the end {:?}: {}; raw K/V of all {} root blocks (every slot, layer, K and V) vs the \
             snapshot taken after the forks: after step 1 changed block indexes {:?} ({}), at the \
             end {:?} ({}); root tail block byte identical: {}; snapshots complete: {}",
            ev.root_stable,
            diverged,
            fed_at(BRANCH_1, 1),
            fed_at(BRANCH_2, 1),
            tail,
            prefix_shared,
            ev.b1_ids.get(tail),
            ev.b2_ids.get(tail),
            ev.root_ids_after.get(tail),
            tails_private,
            ev.b1_total,
            ev.b2_total,
            branch_total,
            ev.physical_final,
            plan.needed_blocks,
            ev.iso.b1_tail,
            ev.iso.b2_tail,
            ev.iso.root_tail,
            tag(fs.tails_private),
            ev.iso.tail_meta_before,
            1 + BRANCHES.len(),
            fs.tail_tokens,
            tag(fs.meta_before_ok),
            ev.iso.tail_meta_step1,
            fs.tail_tokens,
            tag(fs.meta_step1_ok),
            ev.iso.tail_meta_end,
            tag(fs.meta_end_ok),
            plan.final_blocks,
            fs.changed_step1,
            tag(fs.blocks_same_step1),
            fs.changed_end,
            tag(fs.blocks_same_end),
            fs.tail_snapshot_equal,
            fs.snapshots_complete
        ),
    ));

    // (e2) copy on write: the witness shares the root's tail when the append
    // starts, so the first appended token must copy it. The witness keeps the
    // original tail block, byte identical (also covered by (c)).
    let cow_delta = ev.cow_after.saturating_sub(ev.cow_before);
    let witness_untouched =
        ev.witness_ids_after == ev.root_ids_before && ev.witness_total_after == PROMPT_TOKENS;
    let root_tail_moved = matches!(
        (ev.root_ids_after.get(tail_idx), ev.root_ids_before.get(tail_idx)),
        (Some(after), Some(before)) if after != before
    );
    let pages_before_ok = ev.physical_before == plan.prompt_blocks;
    let pages_after_ok = ev.physical_after == plan.prompt_blocks + plan.append_new_blocks;
    checks.push(check(
        "e_cow_witness",
        cow_delta == 1
            && witness_untouched
            && root_tail_moved
            && pages_before_ok
            && pages_after_ok,
        format!(
            "(e) COW_VIOLATION: copy-on-write faults during the append {} (expected 1); witness \
             table {:?} with {} tokens (expected the old table {:?} with {}); root tail block moved \
             {} (before {:?}, after {:?}); physical pages before {} (expected {}) and after {} \
             (expected {})",
            cow_delta,
            ev.witness_ids_after,
            ev.witness_total_after,
            ev.root_ids_before,
            PROMPT_TOKENS,
            root_tail_moved,
            ev.root_ids_before.get(tail_idx),
            ev.root_ids_after.get(tail_idx),
            ev.physical_before,
            plan.prompt_blocks,
            ev.physical_after,
            plan.prompt_blocks + plan.append_new_blocks
        ),
    ));

    // (f) the pool was big enough: no allocation failed, nothing was preempted
    // and the free block counts match the plan exactly.
    let free_after_expected = plan
        .total_blocks
        .saturating_sub(plan.prompt_blocks + plan.append_new_blocks);
    let free_final_expected = plan.total_blocks.saturating_sub(plan.needed_blocks);
    let pass_f = ev.append_error.is_none()
        && ev.root_total_after == TOTAL_TOKENS
        && ev.root_ids_after.len() == plan.final_blocks
        && ev.preempted.is_empty()
        && ev.free_after_append == free_after_expected
        && ev.free_final == free_final_expected;
    let append_note = ev
        .append_error
        .as_ref()
        .map(|e| format!("; APPEND_ERROR_VIOLATION: the append returned Err: {e}"))
        .unwrap_or_default();
    checks.push(check(
        "f_no_pool_exhaustion",
        pass_f,
        format!(
            "(f) POOL_EXHAUSTION_VIOLATION: pool of {} blocks (plan needs {}); root holds {} tokens \
             in {} blocks after the append (expected {} in {}); free blocks after append {} \
             (expected {}) and at the end {} (expected {}); preempted decode rows {:?}{}",
            plan.total_blocks,
            plan.needed_blocks,
            ev.root_total_after,
            ev.root_ids_after.len(),
            TOTAL_TOKENS,
            plan.final_blocks,
            ev.free_after_append,
            free_after_expected,
            ev.free_final,
            free_final_expected,
            ev.preempted,
            append_note
        ),
    ));

    // (g) everything reclaimed.
    let (pass_g, reclaim_note) = match &ev.reclaim {
        Some(r) => (
            r.allocated == 0
                && r.free == plan.total_blocks
                && r.kv_sequences == 0
                && r.physical_pages == 0
                && r.backend_sequences == 0
                && r.backend_pending == 0
                && r.backend_sampling == 0
                && r.free_errors.is_empty(),
            format!(
                "allocated blocks {} (expected 0), free {} (expected {}), KV sequences {}, physical \
                 pages {}, backend sequences {}, pending prefill tokens {}, sampling entries {}, \
                 free errors {:?}",
                r.allocated,
                r.free,
                plan.total_blocks,
                r.kv_sequences,
                r.physical_pages,
                r.backend_sequences,
                r.backend_pending,
                r.backend_sampling,
                r.free_errors
            ),
        ),
        None => (false, "the run never reached the reclaim stage".to_string()),
    };
    checks.push(check(
        "g_reclaimed",
        pass_g,
        format!("(g) RECLAIM_VIOLATION: {reclaim_note}"),
    ));

    // (h) the completion fence, carried over from E1, plus the witness fork.
    let states_ok = ev.state_after_alloc == Some(PrefillState::Allocated)
        && ev.state_after_pending == Some(PrefillState::PrefillPending)
        && ev.state_after_chunk1 == Some(PrefillState::PrefillPending)
        && ev.state_after_chunk2 == Some(PrefillState::PrefillPending)
        && ev.state_after_fence == Some(PrefillState::PrefillReady)
        && ev.state_root_after_witness_fork == Some(PrefillState::SharedFrozen)
        && ev.state_root_after_append == Some(PrefillState::SharedFrozen);
    let chunks_ok = ev.chunk_lens == [PREFILL_CHUNK, PROMPT_TOKENS - PREFILL_CHUNK];
    checks.push(check(
        "h_fence",
        states_ok
            && chunks_ok
            && ev.early_kv_fork_refused
            && ev.early_backend_fork_refused
            && ev.witness_forked,
        format!(
            "(h) FENCE_VIOLATION: root KV states after allocate {:?}, mark pending {:?}, chunk 1 \
             {:?}, chunk 2 {:?}, fence {:?}, witness fork {:?}, append {:?}; chunk lengths {:?}; \
             KV fork before the fence refused {}; backend fork before the fence refused {}; \
             witness forked {}",
            ev.state_after_alloc,
            ev.state_after_pending,
            ev.state_after_chunk1,
            ev.state_after_chunk2,
            ev.state_after_fence,
            ev.state_root_after_witness_fork,
            ev.state_root_after_append,
            ev.chunk_lens,
            ev.early_kv_fork_refused,
            ev.early_backend_fork_refused,
            ev.witness_forked
        ),
    ));

    checks
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

/// Run-wide facts the main test gathers before the stages run.
struct Meta {
    model_path: String,
    model_id: String,
    model_sha256: String,
    tokenizer_path: String,
    tokenizer_sha256: String,
    prompt_digest: String,
    appended_digest: String,
}

fn parity_json(p: &Parity) -> serde_json::Value {
    serde_json::json!({
        "pass": p.pass(),
        "compared_values": p.compared_values,
        "violations": p.violations,
        "max_abs": p.max_abs,
        "max_rel": p.max_rel,
        "first_violation": p.first_violation,
    })
}

fn build_receipt(
    meta: &Meta,
    env: &Env<'_>,
    ev: &Evidence,
    checks: &[Check],
    run_error: Option<&str>,
) -> serde_json::Value {
    let plan = env.plan;
    let layers = env.weights.config.num_layers;
    let stats = position_stats(&ev.append_positions, layers);
    let fs = first_step(plan, ev);
    let verdict = if run_error.is_none() && !checks.is_empty() && checks.iter().all(|c| c.pass) {
        "PASS"
    } else {
        "FAIL"
    };
    let append_s = ev.append_ms / 1e3;
    let append_tokens_per_s = if append_s > 0.0 {
        APPENDED_TOKENS as f64 / append_s
    } else {
        0.0
    };
    let control_s = ev.control_prefill_ms / 1e3;
    let control_tokens = TOTAL_TOKENS + PROMPT_TOKENS;
    let control_tokens_per_s = if control_s > 0.0 {
        control_tokens as f64 / control_s
    } else {
        0.0
    };
    let bytes = |pages: usize| pages * ev.block_bytes;
    serde_json::json!({
        "gate": "PREFILL-E2E-2",
        "backend": "NativeTransformerBackend over CountingBackend(ReferenceCpuBackend), shared pooled \
                    KV manager, FP32 pool, host CPU",
        "commit_sha": commit_sha(),
        "commit_sha_source": "git rev-parse HEAD in CARGO_MANIFEST_DIR at runtime",
        "model": { "path": meta.model_path, "id": meta.model_id, "sha256": meta.model_sha256 },
        "tokenizer": { "path": meta.tokenizer_path, "sha256": meta.tokenizer_sha256 },
        "hardware": hardware(),
        "gb10": "NOT_RUN",
        "max_comparison": "NOT_RUN",
        "prompt_tokens": PROMPT_TOKENS,
        "appended_tokens": APPENDED_TOKENS,
        "total_tokens": TOTAL_TOKENS,
        "decode_steps": DECODE_STEPS,
        "prefill_chunks": ev.chunk_lens,
        "prompt_digest_sha256": meta.prompt_digest,
        "appended_digest_sha256": meta.appended_digest,
        "digest_encoding": "sha256 over the token ids as little-endian u32",
        "tolerance": { "abs": env.tol.abs, "rel": env.tol.rel },
        "checks": checks
            .iter()
            .map(|c| serde_json::json!({
                "name": c.name,
                "pass": c.pass,
                "message": if c.pass { serde_json::Value::Null } else { serde_json::json!(c.message) },
            }))
            .collect::<Vec<_>>(),
        "parity": {
            "root_kv_vs_one_shot_241": parity_json(&ev.root_kv),
            "witness_kv_vs_217": parity_json(&ev.witness_kv),
            "branch_logits": parity_json(&ev.logits_b),
            "witness_logits": parity_json(&ev.logits_w),
            "branch_kv_after_decode": parity_json(&ev.branch_kv),
            "witness_kv_after_decode": parity_json(&ev.witness_final_kv),
            "token_mismatch_sequence_ids": ev.token_mismatch,
        },
        "samples": {
            "after_217": ev.sampled_217.map(|(t, l)| serde_json::json!({ "token": t, "logprob_bits": l.to_bits() })),
            "after_241": ev.sampled_241.map(|(t, l)| serde_json::json!({ "token": t, "logprob_bits": l.to_bits() })),
            "control_217": ev.ctl_217.map(|(t, l)| serde_json::json!({ "token": t, "logprob_bits": l.to_bits() })),
            "control_241": ev.ctl_241.map(|(t, l)| serde_json::json!({ "token": t, "logprob_bits": l.to_bits() })),
        },
        "computed_positions_during_append": {
            "min": stats.min,
            "max": stats.max,
            "count": stats.count,
            "distinct": stats.distinct,
            "per_layer_count": stats.per_layer_count,
            "exact": stats.exact,
            "prompt_prefill_rope_calls": ev.prefill_rope_calls,
        },
        "old_block_ids": {
            "before": ev.root_ids_before,
            "after_append_root_table": ev.root_ids_after,
            "after_append_witness_table": ev.witness_ids_after,
            "digests_unchanged": ev.digests_before == ev.digests_after,
        },
        "branch_isolation_first_step": {
            "tail_block_index": fs.tail,
            "tail_tokens_expected": fs.tail_tokens,
            "tails_after_step1": {
                "branch1": ev.iso.b1_tail,
                "branch2": ev.iso.b2_tail,
                "root": ev.iso.root_tail,
            },
            "tails_distinct_and_root_unmoved": fs.tails_private,
            "shared_tail_block_before_decode": ev.iso.tail_meta_before.map(BlockMeta::json),
            "shared_tail_block_after_step1": ev.iso.tail_meta_step1.map(BlockMeta::json),
            "shared_tail_block_at_end": ev.iso.tail_meta_end.map(BlockMeta::json),
            "shared_tail_num_tokens_after_step1": ev.iso.tail_meta_step1.map(|m| m.num_tokens),
            "shared_tail_meta_ok": {
                "before_decode": fs.meta_before_ok,
                "after_step1": fs.meta_step1_ok,
                "at_end": fs.meta_end_ok,
            },
            "root_blocks_snapshots_complete": fs.snapshots_complete,
            "root_blocks_changed_after_step1": fs.changed_step1,
            "root_blocks_changed_at_end": fs.changed_end,
            "root_tail_block_snapshot_equal": fs.tail_snapshot_equal,
        },
        "kv_block_bytes": ev.block_bytes,
        "physical_kv": {
            "pages_before_append": ev.physical_before,
            "pages_after_append": ev.physical_after,
            "pages_after_decode": ev.physical_final,
            "bytes_before_append": bytes(ev.physical_before),
            "bytes_after_append": bytes(ev.physical_after),
            "bytes_after_decode": bytes(ev.physical_final),
            "cow_faults_during_append": ev.cow_after.saturating_sub(ev.cow_before),
        },
        "timings": {
            "prefill_ms": ev.prefill_ms,
            "append_ms": ev.append_ms,
            "control_prefill_ms": ev.control_prefill_ms,
            "control_prefill_tokens": control_tokens,
            "append_tokens_per_s": append_tokens_per_s,
            "control_tokens_per_s": control_tokens_per_s,
        },
        "preempted_decode_rows": ev.preempted,
        "pool": {
            "block_size": plan.block_size,
            "kv_dtype": "Fp32",
            "prompt_blocks": plan.prompt_blocks,
            "final_blocks": plan.final_blocks,
            "append_new_blocks": plan.append_new_blocks,
            "branch_cow_blocks": plan.branch_cow_blocks,
            "needed_blocks": plan.needed_blocks,
            "slack_blocks": plan.slack_blocks,
            "total_blocks": plan.total_blocks,
            "formula": "needed = ceil(217/bs) + (ceil(241/bs) - ceil(217/bs) + 1) + 2 branch tail \
                        copies; total = needed + slack; the +1 is the copy-on-write copy of the \
                        shared partial tail block",
        },
        "verdict": verdict,
        "run_error": run_error,
    })
}

fn write_receipt(path: &Path, receipt: &serde_json::Value) {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("receipt directory");
        }
    }
    let line = serde_json::to_string(receipt).expect("receipt JSON");
    std::fs::write(path, format!("{line}\n")).expect("write receipt");
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// The prompt must be longer than one prefill chunk so the chunk continuation
/// really runs before the append.
const _: () = assert!(PROMPT_TOKENS > PREFILL_CHUNK);

#[tokio::test]
#[ignore = "PREFILL-E2E-2 real-checkpoint gate (append 24 tokens to a shared prefilled context): \
            run with --ignored and AIEN_E2E_CHECKPOINT + AIEN_E2E2_RECEIPT set (forge on the Spark)"]
async fn prefill_e2e2_real_checkpoint_append_gate() {
    let checkpoint = required_env("AIEN_E2E_CHECKPOINT");
    let receipt_path = PathBuf::from(required_env("AIEN_E2E2_RECEIPT"));

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

    let block_size = weights.config.block_size;
    let prompt_text = TinyLlamaTokenizer::format_prompt(Some(SYSTEM_PROMPT), USER_PROMPT);
    let prompt = tokenizer.encode(&prompt_text).expect("encode root prompt");
    assert_eq!(
        prompt.len(),
        PROMPT_TOKENS,
        "PROMPT_LENGTH: the root prompt is {} tokens; the block arithmetic of this gate is built \
         for {PROMPT_TOKENS}",
        prompt.len()
    );
    assert_eq!(
        block_size, 16,
        "PROMPT_LENGTH: the block arithmetic of this gate is built for 16-slot blocks"
    );
    assert_eq!(
        PROMPT_TOKENS % block_size,
        9,
        "PROMPT_LENGTH: the prompt must end 9 tokens into its last block so the append starts \
         mid-block (got {})",
        PROMPT_TOKENS % block_size
    );
    let appended_all = tokenizer
        .encode_with_special(APPEND_TEXT, false)
        .expect("encode appended text");
    assert!(
        appended_all.len() >= APPENDED_TOKENS,
        "APPEND_LENGTH: the appended text is {} tokens; need at least {APPENDED_TOKENS}",
        appended_all.len()
    );
    let appended = &appended_all[..APPENDED_TOKENS];

    let plan = pool_plan(block_size);
    let meta = Meta {
        model_path: paths.model.display().to_string(),
        model_id,
        model_sha256,
        tokenizer_path: paths.tokenizer.display().to_string(),
        tokenizer_sha256,
        prompt_digest: token_digest(&prompt),
        appended_digest: token_digest(appended),
    };
    let env = Env {
        weights: &weights,
        prompt: &prompt,
        appended,
        tol: &tol,
        counting: Arc::new(CountingBackend::new()),
        plan: &plan,
    };
    println!(
        "PREFILL_E2E2 prompt {} tokens, appended {} tokens, pool {} blocks (needs {})",
        prompt.len(),
        appended.len(),
        plan.total_blocks,
        plan.needed_blocks
    );

    // A FAIL receipt exists from the start so a crash still leaves evidence.
    write_receipt(
        &receipt_path,
        &build_receipt(
            &meta,
            &env,
            &Evidence::default(),
            &[],
            Some("run did not finish"),
        ),
    );

    let mut ev = Evidence::default();
    let run_error = run_stages(&env, &mut ev).await.err();
    let checks = evaluate(&env, &ev);
    write_receipt(
        &receipt_path,
        &build_receipt(&meta, &env, &ev, &checks, run_error.as_deref()),
    );

    // Print every failing message before any assert fires, so every marker
    // that applies is in the log.
    println!(
        "PREFILL_E2E2 append took {:.1} ms for {} tokens; one-shot controls took {:.1} ms",
        ev.append_ms, APPENDED_TOKENS, ev.control_prefill_ms
    );
    if let Some(e) = &run_error {
        println!("PREFILL_E2E2 RUN_ERROR: {e}");
    }
    for c in &checks {
        if c.pass {
            println!("PREFILL_E2E2 check {} PASS", c.name);
        } else {
            println!("PREFILL_E2E2 check {} FAIL: {}", c.name, c.message);
        }
    }

    if let Some(e) = &run_error {
        panic!("RUN_ERROR: {e}");
    }
    for c in &checks {
        assert!(c.pass, "{}", c.message);
    }
}
