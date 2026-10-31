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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelHandle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextHandle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchHandle(pub u64);

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AienUsageReceipt {
    pub prefix_tokens: usize,
    pub private_tokens: usize,
    pub logical_pages: usize,
    pub physical_pages: usize,
    pub shared_pages: usize,
    pub private_pages: usize,
    pub cow_faults: usize,
    pub physical_kv_bytes: usize,
    pub bytes_saved_vs_full_copy: usize,
}

fn default_num_kv_heads() -> usize {
    8
}
fn default_vocab_size() -> usize {
    151936
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

/// Llama 3 rotary frequency smoothing (`rope_scaling` with `rope_type: "llama3"` in a
/// Hugging Face `config.json`). Long wavelengths are divided by `factor`, short ones kept,
/// the band between `high_freq_factor` and `low_freq_factor` is interpolated
/// (transformers `_compute_llama3_parameters`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

impl Llama3RopeScaling {
    /// Applies the llama3 smoothing to one unscaled inverse frequency.
    pub fn scale_inv_freq(&self, inv_freq: f64) -> f64 {
        let old_context = self.original_max_position_embeddings as f64;
        let low_freq_wavelen = old_context / self.low_freq_factor;
        let high_freq_wavelen = old_context / self.high_freq_factor;
        let wavelen = 2.0 * std::f64::consts::PI / inv_freq;
        if wavelen < high_freq_wavelen {
            inv_freq
        } else if wavelen > low_freq_wavelen {
            inv_freq / self.factor
        } else {
            let smooth = (old_context / wavelen - self.low_freq_factor)
                / (self.high_freq_factor - self.low_freq_factor);
            (1.0 - smooth) * inv_freq / self.factor + smooth * inv_freq
        }
    }
}

/// Rotary position embedding parameters of one model: the base `theta` and, for the
/// Llama 3 family, the llama3 frequency smoothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    pub theta: f32,
    pub llama3: Option<Llama3RopeScaling>,
}

impl RopeParams {
    /// Unscaled rope: inverse frequency `theta^(-2i/d)`.
    pub const fn plain(theta: f32) -> Self {
        Self {
            theta,
            llama3: None,
        }
    }

    /// Inverse frequency of rotary pair `i` (`0 <= i < head_dim / 2`), in f64.
    /// The unscaled expression is exactly the one the reference rope has always used,
    /// so models without scaling get bit-identical tables.
    pub fn inv_freq(&self, i: usize, head_dim: usize) -> f64 {
        let exponent = (2 * i) as f64 / (head_dim as f64);
        let base = 1.0 / (self.theta as f64).powf(exponent);
        match &self.llama3 {
            None => base,
            Some(scaling) => scaling.scale_inv_freq(base),
        }
    }

    /// All `head_dim / 2` inverse frequencies.
    pub fn inv_freqs(&self, head_dim: usize) -> Vec<f64> {
        (0..head_dim / 2)
            .map(|i| self.inv_freq(i, head_dim))
            .collect()
    }

    /// Bit pattern identifying these parameters (for table caches).
    pub fn cache_key(&self) -> [u64; 5] {
        match &self.llama3 {
            None => [self.theta.to_bits() as u64, 0, 0, 0, 0],
            Some(s) => [
                self.theta.to_bits() as u64 | (1 << 32),
                s.factor.to_bits(),
                s.low_freq_factor.to_bits(),
                s.high_freq_factor.to_bits(),
                s.original_max_position_embeddings as u64,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub max_sequence_length: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_kv_heads: usize,
    #[serde(default)]
    pub hidden_dim: usize,
    #[serde(default)]
    pub intermediate_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Llama 3 rope smoothing; `None` is plain rope.
    #[serde(default)]
    pub rope_scaling: Option<Llama3RopeScaling>,
    /// True when the output projection reuses `model.embed_tokens.weight`
    /// (the checkpoint has no `lm_head.weight`).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Token ids that end a sequence (`eos_token_id` of `generation_config.json`).
    /// Empty means the legacy TinyLlama set, see [`ModelConfig::stop_token_ids`].
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    /// True for Qwen3 (`Qwen3ForCausalLM`): every attention layer carries per-head RMSNorm
    /// weights `self_attn.q_norm.weight` and `self_attn.k_norm.weight` (length `head_dim`),
    /// applied to Q and K per head before RoPE. `head_dim` is then read from `config.json`
    /// and `num_heads * head_dim` may differ from `hidden_dim`. Not serialized when false, so
    /// every other model's serialized config is unchanged.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qk_norm: bool,
}

/// Why an [`AttentionGeometry`] was refused. Every variant names the offending numbers so a
/// refusal in a log says what to fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionGeometryError {
    ZeroQueryHeads,
    ZeroKvHeads,
    ZeroHeadDim,
    /// `num_q_heads` is not a whole multiple of `num_kv_heads` (every kv head must serve the
    /// same number of query heads).
    QueryHeadsNotDivisible {
        num_q_heads: usize,
        num_kv_heads: usize,
    },
    /// `num_q_heads * head_dim` or `num_kv_heads * head_dim` does not fit in `usize`.
    Overflow {
        heads: usize,
        head_dim: usize,
    },
    /// A KV pool (or any KV buffer) is laid out for a different kv-head count or head width
    /// than this geometry.
    KvPoolMismatch {
        geometry_kv_heads: usize,
        geometry_head_dim: usize,
        pool_kv_heads: usize,
        pool_head_dim: usize,
    },
}

impl std::fmt::Display for AttentionGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroQueryHeads => write!(f, "attention geometry: num_q_heads is 0"),
            Self::ZeroKvHeads => write!(f, "attention geometry: num_kv_heads is 0"),
            Self::ZeroHeadDim => write!(f, "attention geometry: head_dim is 0"),
            Self::QueryHeadsNotDivisible {
                num_q_heads,
                num_kv_heads,
            } => write!(
                f,
                "attention geometry: num_q_heads {num_q_heads} is not a multiple of num_kv_heads {num_kv_heads}"
            ),
            Self::Overflow { heads, head_dim } => write!(
                f,
                "attention geometry: {heads} heads x head_dim {head_dim} overflows usize"
            ),
            Self::KvPoolMismatch {
                geometry_kv_heads,
                geometry_head_dim,
                pool_kv_heads,
                pool_head_dim,
            } => write!(
                f,
                "attention geometry: model has {geometry_kv_heads} kv heads x head_dim {geometry_head_dim} but the KV pool is laid out for {pool_kv_heads} kv heads x head_dim {pool_head_dim}"
            ),
        }
    }
}

impl std::error::Error for AttentionGeometryError {}

/// The checked head geometry of one attention layer: `num_q_heads` query heads sharing
/// `num_kv_heads` key/value heads, every head `head_dim` wide.
///
/// Construction is the only way to get a value, so a geometry in hand always satisfies:
/// no zero count, `num_q_heads` divisible by `num_kv_heads`, and both `q_dim` and `kv_dim`
/// fit in `usize`. The contract deliberately does NOT require a power-of-two GQA ratio,
/// `q_dim == hidden_dim` (Qwen3-Coder: q_dim 4096, hidden 2048) or `num_q_heads ==
/// num_kv_heads`; a kernel with a narrower envelope refuses on its own and says so.
///
/// Flattened layouts the geometry describes:
/// - q and attention output: `[num_q_heads][head_dim]` (`q_dim` values), head h at
///   `h * head_dim`;
/// - one token of K or V: `[num_kv_heads][head_dim]` (`kv_dim` values);
/// - query head h reads kv head `h / gqa_ratio` ([`Self::kv_head_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "AttentionGeometryRaw", into = "AttentionGeometryRaw")]
pub struct AttentionGeometry {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Serde shape of [`AttentionGeometry`]; deserialising re-runs the checks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AttentionGeometryRaw {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl TryFrom<AttentionGeometryRaw> for AttentionGeometry {
    type Error = AttentionGeometryError;
    fn try_from(r: AttentionGeometryRaw) -> Result<Self, Self::Error> {
        Self::new(r.num_q_heads, r.num_kv_heads, r.head_dim)
    }
}

impl From<AttentionGeometry> for AttentionGeometryRaw {
    fn from(g: AttentionGeometry) -> Self {
        Self {
            num_q_heads: g.num_q_heads,
            num_kv_heads: g.num_kv_heads,
            head_dim: g.head_dim,
        }
    }
}

impl AttentionGeometry {
    /// Checks and builds the geometry. See the type docs for the exact rules.
    pub fn new(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionGeometryError> {
        if num_q_heads == 0 {
            return Err(AttentionGeometryError::ZeroQueryHeads);
        }
        if num_kv_heads == 0 {
            return Err(AttentionGeometryError::ZeroKvHeads);
        }
        if head_dim == 0 {
            return Err(AttentionGeometryError::ZeroHeadDim);
        }
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(AttentionGeometryError::QueryHeadsNotDivisible {
                num_q_heads,
                num_kv_heads,
            });
        }
        if num_q_heads.checked_mul(head_dim).is_none() {
            return Err(AttentionGeometryError::Overflow {
                heads: num_q_heads,
                head_dim,
            });
        }
        // num_kv_heads <= num_q_heads here, so kv_dim fits whenever q_dim does.
        Ok(Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn num_q_heads(&self) -> usize {
        self.num_q_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Width of one flattened query (and of one attention output): `num_q_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_q_heads * self.head_dim
    }

    /// Width of one token's K (or V) row: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Query heads per kv head (1 for multi-head attention).
    pub fn gqa_ratio(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// The kv head query head `q_head` reads. `None` when `q_head >= num_q_heads`.
    pub fn kv_head_of(&self, q_head: usize) -> Option<usize> {
        (q_head < self.num_q_heads).then(|| q_head / self.gqa_ratio())
    }

    /// Refuses a KV pool or buffer laid out for a different kv-head count or head width.
    /// Call it before any native (chip) code reads the pool with this geometry.
    pub fn check_kv_pool(
        &self,
        pool_kv_heads: usize,
        pool_head_dim: usize,
    ) -> Result<(), AttentionGeometryError> {
        if pool_kv_heads != self.num_kv_heads || pool_head_dim != self.head_dim {
            return Err(AttentionGeometryError::KvPoolMismatch {
                geometry_kv_heads: self.num_kv_heads,
                geometry_head_dim: self.head_dim,
                pool_kv_heads,
                pool_head_dim,
            });
        }
        Ok(())
    }
}

impl ModelConfig {
    pub fn tinyllama_1_1b() -> Self {
        Self {
            model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            max_sequence_length: 2048,
            block_size: 16,
            num_layers: 22,
            num_heads: 32,
            head_dim: 64,
            num_kv_heads: 4,
            hidden_dim: 2048,
            intermediate_dim: 5632,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_ids: Vec::new(),
            qk_norm: false,
        }
    }

    /// Stop set the backend ends a sequence on before the scheduler sees the token:
    /// UNK 0, BOS 1 and EOS 2 of the TinyLlama vocabulary.
    pub const LEGACY_STOP_TOKEN_IDS: [u32; 3] = [0, 1, 2];

    /// The backend's end-of-sequence set: `eos_token_ids` when the model description
    /// names them, else [`ModelConfig::LEGACY_STOP_TOKEN_IDS`].
    pub fn stop_token_ids(&self) -> &[u32] {
        if self.eos_token_ids.is_empty() {
            &Self::LEGACY_STOP_TOKEN_IDS
        } else {
            &self.eos_token_ids
        }
    }

    /// Rotary embedding parameters of this model.
    pub fn rope(&self) -> RopeParams {
        RopeParams {
            theta: self.rope_theta,
            llama3: self.rope_scaling,
        }
    }

    /// The checked head geometry of this model's attention layers
    /// (`num_heads` query heads, `num_kv_heads` kv heads, `head_dim`).
    pub fn attention_geometry(&self) -> Result<AttentionGeometry, AttentionGeometryError> {
        AttentionGeometry::new(self.num_heads, self.num_kv_heads, self.head_dim)
    }

    pub fn hidden_dim(&self) -> usize {
        if self.hidden_dim > 0 {
            self.hidden_dim
        } else {
            self.num_heads * self.head_dim
        }
    }

    pub fn intermediate_dim(&self) -> usize {
        if self.intermediate_dim > 0 {
            self.intermediate_dim
        } else {
            self.hidden_dim() * 4
        }
    }

    pub fn vocab_size(&self) -> usize {
        if self.vocab_size > 0 {
            self.vocab_size
        } else {
            151936
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16".to_string(),
            max_sequence_length: 32768,
            block_size: 16,
            num_layers: 48,
            num_heads: 32,
            head_dim: 128,
            num_kv_heads: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_ids: Vec::new(),
            qk_norm: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 512,
            stop_token_ids: vec![0, 1, 2],
        }
    }
}

/// How a backend actually chose the tokens of one request, counted at the
/// point of choice, never read back from the request's `SamplingParams`
/// (Qwen3 v4 review, sc#294: greedy decoding must be observed, not only
/// documented). Evidence only: no scheduling or decoding decision reads it.
///
/// Counts every token the backend chose for the request: the token after the
/// final prefill chunk (a mid-prompt chunk's discarded pick is not counted)
/// and every decode token, the stop token included.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecodeObservation {
    /// Tokens taken as the argmax of the logits (greedy).
    pub greedy_tokens: u64,
    /// Tokens drawn at random from the temperature-scaled distribution.
    pub sampled_tokens: u64,
    /// Temperature of the sampled draws (the last draw's). None when no draw sampled.
    pub temperature: Option<f32>,
    /// top_p of the sampled decode draws when nucleus filtering was applied
    /// (`0 < top_p < 1`). None when no draw applied it.
    pub top_p: Option<f32>,
    /// The request id the sampled draws were seeded from (the stream is a
    /// pure function of it and the position). None when no draw sampled.
    pub seed_request_id: Option<u64>,
}

// Temperatures recorded here passed through the sampler, which never sees a
// NaN from the runtime; equality is used only to compare records.
impl Eq for DecodeObservation {}

/// Which tensor backend ran the model and how many operations it ran on the
/// reference CPU path, as process totals since the backend was built (sc#337:
/// "no silent fallback" must be shown, not inferred from the absence of a
/// crash). Evidence only: nothing decides on it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpEvidence {
    /// The tensor backend's name (`TensorBackend::name`).
    pub backend: String,
    /// Runs of ops the backend claims native that fell back to the reference
    /// CPU path. A production build panics on the first one, so a finished
    /// strict call always records 0.
    pub native_fallbacks: u64,
    /// Runs of ops on the reference CPU path by design (outside the native mask).
    pub reference_runs: u64,
    /// The full `OP_REPORT native=[..] reference=[..] native_fallbacks=[..]
    /// reference_runs=[..]` line.
    pub report: String,
    /// Operations the backend ran on its chip, as a process total (sc#402: the
    /// Omega GB10 engine's `chip_calls`). ABSENT, not zero, for a backend that
    /// has no chip or does not count its chip calls; serialized only when present,
    /// so records of such backends are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_executions: Option<u64>,
}

impl DecodeObservation {
    /// "greedy" (every chosen token was the argmax), "sampled" (every one was
    /// drawn), "mixed", or "none" (no token was chosen).
    pub fn mode(&self) -> &'static str {
        match (self.greedy_tokens > 0, self.sampled_tokens > 0) {
            (true, false) => "greedy",
            (false, true) => "sampled",
            (true, true) => "mixed",
            (false, false) => "none",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRequest {
    pub request_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub arrival_time_ns: u64,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledBatch {
    pub prefill_requests: Vec<SequenceRequest>,
    pub decode_requests: Vec<u64>,
    pub block_tables: HashMap<u64, Vec<usize>>,
    pub step_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    StopToken,
    LengthLimit,
    Aborted,
    Preempted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecodeOutput {
    Token {
        request_id: u64,
        token_id: u32,
        logprob: Option<f32>,
    },
    Finished {
        request_id: u64,
        reason: FinishReason,
        total_tokens: usize,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMetrics {
    pub prefill_tokens_processed: usize,
    pub decode_tokens_emitted: usize,
    pub step_latency_us: u64,
    pub active_kv_blocks: usize,
}

#[async_trait]
pub trait AienInferenceBackend: Send + Sync {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String>;
    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String>;

    /// Indicates whether the backend directly manages and appends tokens into the KV cache.
    /// When true, the scheduler avoids redundant secondary append operations during decode steps.
    fn manages_kv_cache(&self) -> bool {
        false
    }

    /// PREFILL-E2E-0 bullets 6-8: forks the backend's per-sequence state
    /// (tokens, position, last token) from a prefilled parent into a child.
    /// Called by the runtime spine once per branch AFTER the root passed the
    /// prefill completion fence and the KV manager shared the root blocks
    /// (`fork_branches_from_ready_root`), and BEFORE any branch is decoded.
    /// Must never copy or re-fork K/V: the physical KV fork is already done.
    /// Default: no-op, for backends that keep no per-sequence state.
    fn fork_sequence(&mut self, _parent_id: u64, _child_id: u64) -> Result<(), String> {
        Ok(())
    }

    /// PREFILL-E2E C6 (bullet 11): `fork_sequence` that also hands the
    /// backend the child's own sampling params, so a branch decodes with its
    /// request's temperature/top_p instead of argmax. Decode batches carry
    /// only request ids (`ScheduledBatch::decode_requests`), so the fork hook
    /// is where a branch's params reach the backend.
    /// Default: plain `fork_sequence` (params ignored).
    fn fork_sequence_with_sampling(
        &mut self,
        parent_id: u64,
        child_id: u64,
        _sampling: &SamplingParams,
    ) -> Result<(), String> {
        self.fork_sequence(parent_id, child_id)
    }

    /// Drops the backend's per-sequence state for `seq_id` (PREFILL-E2E
    /// bullet 12). The runtime spine calls it for every finished branch, for
    /// the swarm root once its last branch finished, and for every sequence of
    /// a cancelled swarm, always after the runtime freed the KV blocks and
    /// never while a step is executing. Must not free KV: the runtime owns the
    /// block tables. Unknown ids are a no-op (a cancelled branch may already
    /// have finished). Default: no-op.
    fn release_sequence(&mut self, _seq_id: u64) -> Result<(), String> {
        Ok(())
    }

    /// Persistent KV (ADR 0003 section 6.2, review finding 3): the tokens of
    /// `seq_id` whose K/V is actually written in the KV pool, in order. This is
    /// never the whole token list: after a decode step the last token has been
    /// sampled but not yet run through the model, so it has no K/V. `None` for
    /// an unknown sequence or a backend that does not keep its K/V in the shared
    /// pool. Read-only; the scheduler calls it at the finish transaction while
    /// the backend still holds the sequence, before the KV lock is taken.
    /// Default: `None` (no claim, so nothing is ever parked from this backend).
    fn computed_prefix(&self, _seq_id: u64) -> Option<Vec<u32>> {
        None
    }

    /// Persistent KV (ADR 0003 section 6.3): gives `seq_id` backend state for a
    /// prefix whose K/V a restore already placed in its KV block table, so the
    /// next prefill chunk starts at position `tokens.len()`. `tokens` is the
    /// restored prefix. Never copies or computes K/V. Called by the runtime
    /// after the restored table was assembled and before the scheduler admits
    /// the remaining prompt. Default: `Err`, because a backend that does not
    /// override this cannot continue from restored K/V and the caller must fall
    /// back to a plain prefill.
    fn restore_sequence(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(), String> {
        Err(format!(
            "restore_sequence {} ({} tokens) refused: this backend does not support restoring a persisted KV prefix",
            seq_id,
            tokens.len()
        ))
    }

    /// Takes (and forgets) how the backend chose `request_id`'s tokens so
    /// far. The scheduler calls it once, when the request finishes, and
    /// attaches it to `CompletionEvent::Finished`. Default: None, for a
    /// backend that does not observe its decoding (an absent claim, not greedy).
    fn take_decode_observation(&mut self, _request_id: u64) -> Option<DecodeObservation> {
        None
    }

    /// The tensor backend and its op counters so far (sc#337). The scheduler
    /// reads it when a request finishes and attaches it to
    /// `CompletionEvent::Finished`. Default: None, for a backend that does not
    /// account its ops (an absent claim, not zero fallbacks).
    fn op_evidence(&self) -> Option<OpEvidence> {
        None
    }
}
