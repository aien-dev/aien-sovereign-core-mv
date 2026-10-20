//! Pure native Rust transformer inference backend executing real forward passes.
//! Dispatches tensor math through the decoupled TensorBackend trait for CPU and Omega GB10 execution.

use crate::backend::{ReferenceCpuBackend, TensorBackend};
use crate::tensor::{sample_argmax, sample_temperature};
use crate::weights::{LayerKvCache, SequenceState, TransformerWeights};
use crate::{
    AienInferenceBackend, AienUsageReceipt, BranchHandle, ContextHandle, DecodeObservation,
    DecodeOutput, FinishReason, ModelConfig, SamplingParams, ScheduledBatch, StepMetrics,
};
use aien_kv_cache::{
    create_shared_kv_manager_with_pool, round_to_dtype_in_place, AienKvManager, KvDType,
    KvPoolConfig, SharedKvManager,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Stable prefix of every error returned when the paged KV pool cannot supply
/// the blocks a prefill append or a single-token decode step needs
/// (PREFILL-I25). The rest of the message says how many new blocks were needed
/// and how many were free. Callers and tests may match on this prefix.
pub const KV_POOL_EXHAUSTED_PREFIX: &str = "KV pool exhausted:";

/// splitmix64 mixing step (the generator behind java.util.SplittableRandom):
/// a bijective 64-bit mixer, in-house, no outside crate.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic seed for sampling the token at `position` (the sequence's
/// token count before the new token) of request `request_id` (PREFILL-E2E
/// C6). Swarm branches get distinct request ids from the runtime arena
/// (`AienRuntimeSpine::launch_swarm` submits each branch with
/// `request_id: child_seq.as_u64()`), so sibling branches forked from one
/// root draw different streams; the same request replays the same stream.
pub fn decode_sampling_seed(request_id: u64, position: usize) -> u64 {
    splitmix64(request_id ^ splitmix64(position as u64))
}

/// Samples one token from `logits` under `params` with the given seed.
/// - `temperature <= 0.001`: argmax (unchanged greedy path).
/// - otherwise softmax(logits / temperature) in f64; if `0 < top_p < 1`
///   only the smallest highest-probability set whose mass reaches `top_p`
///   is kept (ties broken by lower token id); one uniform draw from the
///   seed picks a token from the kept mass. (`SamplingParams` has no top-k
///   field, so there is no top-k.)
///
/// Returns the token and its log-probability under the kept distribution.
pub fn sample_with_params(logits: &[f32], params: &SamplingParams, seed: u64) -> (u32, f32) {
    sample_with_params_observed(logits, params, seed, &mut DecodeObservation::default())
}

/// `sample_with_params` that also counts, inside the branch it takes, how the
/// token was chosen (sc#294): greedy, or a draw at `temperature` (with
/// `top_p` when nucleus filtering ran). The seed's request id is the caller's
/// to record. The chosen token is identical to `sample_with_params`.
pub fn sample_with_params_observed(
    logits: &[f32],
    params: &SamplingParams,
    seed: u64,
    obs: &mut DecodeObservation,
) -> (u32, f32) {
    let (token, logprob, _) = sample_with_params_traced(logits, params, seed, obs);
    (token, logprob)
}

/// What one draw had to choose from (OWN-8B stage S): the kept set after the
/// temperature softmax and the nucleus cut, counted inside the sampler, so a
/// confident model and a sampler that ignores its seed can be told apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawTrace {
    /// Tokens with non-zero probability in the set the draw chose from. 1 for a
    /// greedy pick; 1 for a sampled draw means the seed had nothing to choose.
    pub kept: usize,
    /// Probability of the most likely token after temperature, before the
    /// nucleus cut. 1.0 for a greedy pick.
    pub top_prob: f64,
}

/// `sample_with_params_observed` that also returns the [`DrawTrace`]. This is
/// the one implementation of the sampling math; the other two call it.
pub fn sample_with_params_traced(
    logits: &[f32],
    params: &SamplingParams,
    seed: u64,
    obs: &mut DecodeObservation,
) -> (u32, f32, DrawTrace) {
    if params.temperature <= 0.001 {
        obs.greedy_tokens += 1;
        let (token, logprob) = sample_argmax(logits);
        let trace = DrawTrace {
            kept: 1,
            top_prob: 1.0,
        };
        return (token, logprob, trace);
    }
    obs.sampled_tokens += 1;
    obs.temperature = Some(params.temperature);
    assert!(!logits.is_empty(), "Logits cannot be empty");
    let inv_t = 1.0f64 / params.temperature as f64;
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&v| ((v as f64 - max) * inv_t).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut kept: Vec<usize> = (0..probs.len()).collect();
    let top_p = params.top_p as f64;
    if top_p > 0.0 && top_p < 1.0 {
        obs.top_p = Some(params.top_p);
        kept.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut cum = 0.0f64;
        let mut keep = 0usize;
        for &i in &kept {
            cum += probs[i];
            keep += 1;
            if cum >= top_p {
                break;
            }
        }
        kept.truncate(keep);
    }

    let mass: f64 = kept.iter().map(|&i| probs[i]).sum();
    let trace = DrawTrace {
        kept: kept.iter().filter(|&&i| probs[i] > 0.0).count(),
        // The largest logit contributed exp(0) = 1 to `sum`.
        top_prob: 1.0 / sum,
    };
    // 53 random bits -> uniform in [0, 1), scaled to the kept mass.
    let r = ((seed >> 11) as f64 / (1u64 << 53) as f64) * mass;
    let mut cum = 0.0f64;
    for &i in &kept {
        cum += probs[i];
        if r < cum {
            return (i as u32, (probs[i] / mass).max(1e-300).ln() as f32, trace);
        }
    }
    let last = *kept.last().expect("kept set is never empty");
    let logprob = (probs[last] / mass).max(1e-300).ln() as f32;
    (last as u32, logprob, trace)
}

/// The draw `sample_with_params_traced` makes over `logits` with everything
/// outside the `k` best set to -inf, computed from those `k` entries alone: one
/// O(V) scan for the winners, then softmax, nucleus cut and draw over `k`.
///
/// The result is bit-identical to masking then calling
/// `sample_with_params_traced` (token, log-probability and trace): winners rank
/// by higher logit then lower token id; the softmax sum adds the winners' terms
/// in ascending token id, which is the order the full pass adds its non-zero
/// terms (the masked terms are exact zeros); the nucleus order is probability
/// descending, lower id first. Returns `None`, leaving `obs` untouched, whenever
/// that equality is not guaranteed (a NaN or infinite maximum, a nucleus that
/// never reaches `top_p`, a draw that falls off the end, `temperature <= 0.001`,
/// `k` of 0 or at least the vocabulary): the caller then takes the full path.
pub fn sample_top_k_traced(
    logits: &[f32],
    k: usize,
    params: &SamplingParams,
    seed: u64,
    obs: &mut DecodeObservation,
) -> Option<(u32, f32, DrawTrace)> {
    if params.temperature <= 0.001 || k == 0 || k >= logits.len() {
        return None;
    }
    // Winners, best first: (logit, token id). Scanning ascending ids means a
    // later equal logit never displaces an earlier one, so the tie rule holds.
    let mut win: Vec<(f32, u32)> = Vec::with_capacity(k);
    for (i, &v) in logits[..k].iter().enumerate() {
        if v.is_nan() {
            return None;
        }
        let at = win.partition_point(|w| w.0 >= v);
        win.insert(at, (v, i as u32));
    }
    let mut floor = win[k - 1].0;
    for (i, &v) in logits.iter().enumerate().skip(k) {
        // `!(v <= floor)` is also true for NaN, which bails below.
        if !(v <= floor) {
            if v.is_nan() {
                return None;
            }
            let at = win.partition_point(|w| w.0 >= v);
            win.insert(at, (v, i as u32));
            win.pop();
            floor = win[k - 1].0;
        }
    }
    let max = win[0].0 as f64;
    if !max.is_finite() {
        return None;
    }
    // Ascending token id: the summation order of the full pass.
    win.sort_unstable_by_key(|w| w.1);
    let inv_t = 1.0f64 / params.temperature as f64;
    let mut probs: Vec<(u32, f64)> = win
        .iter()
        .map(|&(v, i)| (i, ((v as f64 - max) * inv_t).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|p| p.1).sum();
    for p in probs.iter_mut() {
        p.1 /= sum;
    }

    let top_p = params.top_p as f64;
    let cut = top_p > 0.0 && top_p < 1.0;
    if cut {
        probs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let mut cum = 0.0f64;
        let mut keep = 0usize;
        let mut reached = false;
        for p in &probs {
            cum += p.1;
            keep += 1;
            if cum >= top_p {
                reached = true;
                break;
            }
        }
        if !reached {
            return None;
        }
        probs.truncate(keep);
    }

    let mass: f64 = probs.iter().map(|p| p.1).sum();
    let trace = DrawTrace {
        kept: probs.iter().filter(|p| p.1 > 0.0).count(),
        top_prob: 1.0 / sum,
    };
    let r = ((seed >> 11) as f64 / (1u64 << 53) as f64) * mass;
    let mut cum = 0.0f64;
    for &(i, p) in &probs {
        cum += p;
        if r < cum {
            obs.sampled_tokens += 1;
            obs.temperature = Some(params.temperature);
            if cut {
                obs.top_p = Some(params.top_p);
            }
            return Some((i, (p / mass).max(1e-300).ln() as f32, trace));
        }
    }
    None
}

/// Fully native Rust transformer backend executing real tensor forward computation.
/// Decouples execution orchestration from compute hardware via the TensorBackend trait.
pub struct NativeTransformerBackend {
    pub weights: TransformerWeights,
    pub sequences: HashMap<u64, SequenceState>,
    pub tensor_backend: Arc<dyn TensorBackend>,
    pub kv_manager: Option<SharedKvManager>,
    /// Element type K/V are rounded to at birth when no physical pool is attached
    /// (a pool's own dtype wins; see `storage_dtype`). `Fp32` = no rounding. A dense
    /// control backend sets this to the pool dtype it is the reference for.
    pub kv_dtype: KvDType,
    /// Per-request sampling params used by decode (PREFILL-E2E C6). Recorded
    /// from the prefill request (`execute_step`) and from the fork hook
    /// (`fork_sequence_with_sampling`; a plain `fork_sequence` child inherits
    /// its parent's entry). A request with no entry decodes with argmax.
    sampling: HashMap<u64, SamplingParams>,
    /// Token sampled after the most recent `execute_step` prefill chunk of a request,
    /// not yet appended to `sequences[id].tokens`. A prefill request does not say
    /// whether it is the final chunk of its prompt, so the sample is held here:
    /// a following prefill chunk discards it (it was mid-prefill), and the first
    /// decode (or fork) commits it. While a request is prefilling, `tokens.len()`
    /// therefore always equals the number of positions whose K/V is cached.
    pub pending_prefill_token: HashMap<u64, u32>,
    /// How each request's tokens were actually chosen (sc#294), filled inside the
    /// branch that chose them and taken once by the scheduler when the request
    /// finishes (`take_decode_observation`).
    decode_observed: HashMap<u64, DecodeTally>,
}

/// Per-request tally behind `DecodeObservation` (sc#294). The pick after a
/// prefill chunk is held apart and replaced by the next chunk's, like
/// `pending_prefill_token`, so a mid-prompt pick that is discarded is not counted.
#[derive(Debug, Default)]
struct DecodeTally {
    prefill_pick: DecodeObservation,
    decode: DecodeObservation,
}

impl DecodeTally {
    fn merged(self) -> DecodeObservation {
        let (p, d) = (self.prefill_pick, self.decode);
        DecodeObservation {
            greedy_tokens: p.greedy_tokens + d.greedy_tokens,
            sampled_tokens: p.sampled_tokens + d.sampled_tokens,
            temperature: d.temperature.or(p.temperature),
            top_p: d.top_p.or(p.top_p),
            seed_request_id: d.seed_request_id.or(p.seed_request_id),
        }
    }
}

impl NativeTransformerBackend {
    /// Creates a new backend with default ReferenceCpuBackend oracle.
    pub fn new(weights: TransformerWeights) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            sampling: HashMap::new(),
            decode_observed: HashMap::new(),
            tensor_backend: Arc::new(ReferenceCpuBackend::new()),
            kv_manager: None,
            pending_prefill_token: HashMap::new(),
            kv_dtype: KvDType::Fp32,
        }
    }

    /// Creates a backend with an explicit TensorBackend implementation.
    pub fn with_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
    ) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            sampling: HashMap::new(),
            decode_observed: HashMap::new(),
            tensor_backend,
            kv_manager: None,
            pending_prefill_token: HashMap::new(),
            kv_dtype: KvDType::Fp32,
        }
    }

    /// Explicit constructor for ReferenceCpuBackend golden oracle.
    pub fn new_reference(weights: TransformerWeights) -> Self {
        Self::new(weights)
    }

    /// Convenience constructor with deterministic reference weights and ReferenceCpuBackend.
    pub fn with_reference_weights(config: &ModelConfig) -> Self {
        let weights = TransformerWeights::reference_test_weights(config);
        Self::new(weights)
    }

    /// Explicit constructor for the GB10 GPU backend (Omega native kernels, no CUDA).
    pub fn new_blackwell(weights: TransformerWeights) -> Self {
        Self::with_backend(weights, Arc::new(crate::OmegaGb10Backend::new()))
    }

    /// Convenience constructor with deterministic reference weights and OmegaGb10Backend.
    pub fn with_blackwell_backend(config: &ModelConfig) -> Self {
        let weights = TransformerWeights::reference_test_weights(config);
        Self::new_blackwell(weights)
    }

    /// Constructor attaching a physical reference-counted KV page pool for Copy-on-Write branching.
    pub fn with_paged_kv(
        weights: TransformerWeights,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        let pool_cfg = KvPoolConfig::for_model(
            total_blocks,
            block_size,
            weights.config.num_layers,
            weights.config.num_kv_heads,
            weights.config.head_dim,
            KvDType::Fp32,
        );
        let kv_mgr = create_shared_kv_manager_with_pool(total_blocks, block_size, pool_cfg)?;
        Ok(Self {
            weights,
            sequences: HashMap::new(),
            sampling: HashMap::new(),
            decode_observed: HashMap::new(),
            tensor_backend: Arc::new(ReferenceCpuBackend::new()),
            kv_manager: Some(kv_mgr),
            pending_prefill_token: HashMap::new(),
            kv_dtype: KvDType::Fp32,
        })
    }

    /// Constructor attaching physical paged KV cache with explicit TensorBackend compute hardware.
    pub fn with_paged_kv_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        let pool_cfg = KvPoolConfig::for_model(
            total_blocks,
            block_size,
            weights.config.num_layers,
            weights.config.num_kv_heads,
            weights.config.head_dim,
            KvDType::Fp32,
        );
        let kv_mgr = create_shared_kv_manager_with_pool(total_blocks, block_size, pool_cfg)?;
        Ok(Self {
            weights,
            sequences: HashMap::new(),
            sampling: HashMap::new(),
            decode_observed: HashMap::new(),
            tensor_backend,
            kv_manager: Some(kv_mgr),
            pending_prefill_token: HashMap::new(),
            kv_dtype: KvDType::Fp32,
        })
    }

    /// Constructor attaching an existing shared KV manager and explicit TensorBackend compute hardware.
    pub fn with_shared_kv_and_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
        kv_manager: SharedKvManager,
    ) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            sampling: HashMap::new(),
            decode_observed: HashMap::new(),
            tensor_backend,
            kv_manager: Some(kv_manager),
            pending_prefill_token: HashMap::new(),
            kv_dtype: KvDType::Fp32,
        }
    }

    /// Sampling params decode will use for `request_id`, if any were recorded
    /// (PREFILL-E2E C6). `None` means decode takes the argmax.
    pub fn sampling_params(&self, request_id: u64) -> Option<&SamplingParams> {
        self.sampling.get(&request_id)
    }

    /// Creates a new immutable root context from prompt tokens.
    pub fn prefill_sequence(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        if prompt_tokens.is_empty() {
            return Err("Cannot prefill empty prompt".to_string());
        }

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().allocate_sequence(seq_id, prompt_tokens)?;
        }

        let num_layers = self.weights.config.num_layers;
        let kv_dtype = self.storage_dtype();
        let seq = self
            .sequences
            .entry(seq_id)
            .or_insert_with(|| SequenceState {
                tokens: Vec::new(),
                layers: vec![LayerKvCache::default(); num_layers],
                kv_dtype,
            });

        let last_hidden = Self::prefill_prompt_layer_by_layer_paged(
            &self.weights,
            &*self.tensor_backend,
            prompt_tokens,
            seq,
            seq_id,
            self.kv_manager.as_ref(),
        )?;

        let logits = Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &last_hidden);
        Ok(logits)
    }

    pub fn create_context(&mut self, prompt_tokens: &[u32]) -> Result<ContextHandle, String> {
        if prompt_tokens.is_empty() {
            return Err("Cannot create context with empty prompt".to_string());
        }
        let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
        let handle = ContextHandle(id);

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().allocate_sequence(handle.0, prompt_tokens)?;
        }

        let num_layers = self.weights.config.num_layers;
        let kv_dtype = self.storage_dtype();
        let seq = self
            .sequences
            .entry(handle.0)
            .or_insert_with(|| SequenceState {
                tokens: Vec::new(),
                layers: vec![LayerKvCache::default(); num_layers],
                kv_dtype,
            });

        Self::prefill_prompt_layer_by_layer_paged(
            &self.weights,
            &*self.tensor_backend,
            prompt_tokens,
            seq,
            handle.0,
            self.kv_manager.as_ref(),
        )?;

        Ok(handle)
    }

    /// Appends the token sampled after a completed `execute_step` prefill (if any)
    /// to the sequence, so the next decode feeds it at position `prompt_len`.
    fn commit_pending_prefill_token(&mut self, seq_id: u64) {
        if let Some(tok) = self.pending_prefill_token.remove(&seq_id) {
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.tokens.push(tok);
            }
        }
    }

    /// Forks an existing context into an independently decoding branch with zero KV copying.
    ///
    /// Handle-API helper for library tests (`paged_transformer_parity.rs`,
    /// `branch_client_harness.rs`, the unit tests below). It is NOT on the
    /// runtime path: the spine forks backend state only through the trait hook
    /// `AienInferenceBackend::fork_sequence`, which is gated on PrefillReady and
    /// never copies K/V. This method is ungated and, without a KV manager,
    /// copies the parent's dense K/V per branch; no code in aien-runtime or
    /// aien-cli calls it (grep `fork_context`, PREFILL-E2E C4).
    pub fn fork_context(&mut self, parent: ContextHandle) -> Result<BranchHandle, String> {
        self.commit_pending_prefill_token(parent.0);
        let parent_seq = self
            .sequences
            .get(&parent.0)
            .ok_or_else(|| format!("Parent context {} not found", parent.0))?;

        let child_id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
        let child = BranchHandle(child_id);

        let child_seq = if self.kv_manager.is_some() {
            SequenceState {
                tokens: parent_seq.tokens.clone(),
                layers: Vec::new(),
                kv_dtype: parent_seq.kv_dtype,
            }
        } else {
            parent_seq.clone()
        };

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().fork_context(parent.0, child.0)?;
        }

        self.sequences.insert(child.0, child_seq);
        Ok(child)
    }

    /// Releases a branch and immediately reclaims its private COW pages.
    pub fn release_branch(&mut self, branch: BranchHandle) -> Result<(), String> {
        self.sequences.remove(&branch.0);
        self.pending_prefill_token.remove(&branch.0);
        if let Some(kv_mgr) = &self.kv_manager {
            let _ = kv_mgr.write().release_branch(branch.0);
        }
        Ok(())
    }

    /// Executes one autoregressive decode step for a branch, triggering COW if tail page is shared.
    pub fn decode_branch_step(&mut self, branch: BranchHandle) -> Result<(u32, Vec<f32>), String> {
        self.commit_pending_prefill_token(branch.0);
        let seq = self
            .sequences
            .get_mut(&branch.0)
            .ok_or_else(|| format!("Branch {} not found", branch.0))?;

        let pos = seq.tokens.len().saturating_sub(1);
        let last_token = *seq.tokens.last().unwrap_or(&1);

        let hidden = Self::try_forward_token_impl_paged(
            &self.weights,
            &*self.tensor_backend,
            last_token,
            pos,
            seq,
            branch.0,
            self.kv_manager.as_ref(),
        )?;

        let logits = Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &hidden);
        let (sampled_tok, _) = sample_argmax(&logits);
        seq.tokens.push(sampled_tok);

        Ok((sampled_tok, logits))
    }

    /// Ingests branch-specific prompt/delta tokens into the physical paged KV cache, triggering COW on shared blocks.
    pub fn append_branch_token(&mut self, branch: BranchHandle, token: u32) -> Result<(), String> {
        // An explicitly supplied token replaces any sample held from a prefill chunk.
        self.pending_prefill_token.remove(&branch.0);
        let seq = self
            .sequences
            .get_mut(&branch.0)
            .ok_or_else(|| format!("Branch {} not found", branch.0))?;

        let pos = seq.tokens.len();
        let _hidden = Self::try_forward_token_impl_paged(
            &self.weights,
            &*self.tensor_backend,
            token,
            pos,
            seq,
            branch.0,
            self.kv_manager.as_ref(),
        )?;
        seq.tokens.push(token);
        Ok(())
    }

    /// Ingests multiple branch-specific delta tokens sequentially.
    pub fn append_branch_tokens(
        &mut self,
        branch: BranchHandle,
        tokens: &[u32],
    ) -> Result<(), String> {
        for &tok in tokens {
            self.append_branch_token(branch, tok)?;
        }
        Ok(())
    }

    /// Generates tokens autoregressively, invoking a callback for each generated token.
    pub fn generate_tokens_streaming<F>(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        mut on_token: F,
    ) -> Result<(), String>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err("Cannot generate from empty prompt".to_string());
        }

        let logits = self.prefill_sequence(seq_id, prompt_tokens)?;

        let (first_tok, _) = if temperature <= 0.001 {
            sample_argmax(&logits)
        } else {
            sample_temperature(&logits, temperature, seq_id)
        };

        if stop_tokens.contains(&first_tok) {
            self.release_sequence(seq_id);
            return Ok(());
        }

        if !on_token(first_tok) {
            self.release_sequence(seq_id);
            return Ok(());
        }

        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            seq.tokens.push(first_tok);
        }

        for step in 1..max_tokens {
            let seq = match self.sequences.get_mut(&seq_id) {
                Some(s) => s,
                None => break,
            };

            let pos = seq.tokens.len().saturating_sub(1);
            let last_token = *seq.tokens.last().unwrap_or(&1);

            let hidden = match Self::try_forward_token_impl_paged(
                &self.weights,
                &*self.tensor_backend,
                last_token,
                pos,
                seq,
                seq_id,
                self.kv_manager.as_ref(),
            ) {
                Ok(hidden) => hidden,
                Err(e) => {
                    // This standalone generator owns its KV: do not leak the
                    // sequence when a decode step cannot get a block.
                    self.release_sequence(seq_id);
                    return Err(e);
                }
            };

            let next_logits =
                Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &hidden);
            let (next_tok, _) = if temperature <= 0.001 {
                sample_argmax(&next_logits)
            } else {
                sample_temperature(&next_logits, temperature, seq_id.wrapping_add(step as u64))
            };

            if stop_tokens.contains(&next_tok) {
                break;
            }

            if !on_token(next_tok) {
                break;
            }

            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.tokens.push(next_tok);
            }
        }

        self.release_sequence(seq_id);
        Ok(())
    }

    /// Generates tokens autoregressively from a sequence of prompt tokens.
    pub fn generate_tokens(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        let mut generated = Vec::with_capacity(max_tokens);
        self.generate_tokens_streaming(
            seq_id,
            prompt_tokens,
            max_tokens,
            temperature,
            stop_tokens,
            |tok| {
                generated.push(tok);
                true
            },
        )?;
        Ok(generated)
    }

    /// Releases a sequence from memory and frees associated KV cache resources.
    pub fn release_sequence(&mut self, seq_id: u64) {
        self.sequences.remove(&seq_id);
        self.pending_prefill_token.remove(&seq_id);
        self.decode_observed.remove(&seq_id);
        if let Some(kv_mgr) = &self.kv_manager {
            let _ = kv_mgr.write().release_branch(seq_id);
        }
    }

    /// Loads TinyLlama weights from safetensors with optional paged KV cache and compute backend.
    pub fn load_tinyllama_safetensors<P: AsRef<std::path::Path>>(
        checkpoint_path: P,
        tensor_backend: Option<Arc<dyn TensorBackend>>,
        total_blocks: Option<usize>,
    ) -> Result<Self, String> {
        let config = ModelConfig::tinyllama_1_1b();
        let weights = TransformerWeights::load_from_safetensors(checkpoint_path.as_ref(), &config)
            .map_err(|e| format!("Failed to load safetensors: {}", e))?;

        let backend = tensor_backend.unwrap_or_else(|| {
            let surface = crate::ExecutionSurface::detect();
            if surface.is_accelerated_gpu() {
                Arc::new(crate::OmegaGb10Backend::new())
            } else {
                Arc::new(ReferenceCpuBackend::new())
            }
        });

        if let Some(blocks) = total_blocks {
            Self::with_paged_kv_backend(weights, backend, blocks, config.block_size)
        } else {
            Ok(Self::with_backend(weights, backend))
        }
    }

    /// Returns a telemetry receipt for a branch, calculating shared pages and physical bytes saved.
    pub fn get_usage_receipt(&self, branch: BranchHandle) -> Result<AienUsageReceipt, String> {
        let kv_mgr = self
            .kv_manager
            .as_ref()
            .ok_or_else(|| "KV manager not initialized on backend".to_string())?
            .read();

        let table = kv_mgr
            .get_block_table(branch.0)
            .ok_or_else(|| format!("Branch {} not found in KV cache", branch.0))?;
        let metrics = kv_mgr.metrics();
        let total_tokens = table.total_tokens;

        let mut shared_pages = 0;
        let mut private_pages = 0;
        for &b in &table.block_ids {
            if let Some(block) = kv_mgr.get_block(b) {
                if block.is_shared {
                    shared_pages += 1;
                } else {
                    private_pages += 1;
                }
            }
        }

        let bytes_per_block = kv_mgr.tensor_pool().map(|p| p.block_bytes()).unwrap_or(0);
        let physical_kv_bytes = (shared_pages + private_pages) * bytes_per_block;

        Ok(AienUsageReceipt {
            prefix_tokens: shared_pages * kv_mgr.block_size(),
            private_tokens: total_tokens.saturating_sub(shared_pages * kv_mgr.block_size()),
            logical_pages: table.block_ids.len(),
            physical_pages: metrics.physical_pages,
            shared_pages,
            private_pages,
            cow_faults: metrics.cow_faults,
            physical_kv_bytes,
            bytes_saved_vs_full_copy: shared_pages * bytes_per_block,
        })
    }

    /// Internal forward pass executing one token forward pass through the transformer.
    /// sc draft-verify (2026-10-12): the speculative-verification primitive.
    /// Forwards `block` (block[0] = the sequence's pending token, the one whose
    /// K/V is not computed yet; block[1..] = draft tokens) at positions
    /// offset..offset+n, offset = the K/V length of `seq_id` in the paged pool,
    /// in ONE row-batched pass per layer: the matmuls take all n rows (as the
    /// prompt read does), attention is the existing paged batch call with the
    /// sequence's block table n times and context lengths offset+1..offset+n
    /// (causal, pool read in place, no staging), the lm_head runs on every row.
    /// Returns the greedy argmax of each row: row t predicts position
    /// offset+t+1, so draft token block[t+1] is accepted while it equals row t.
    /// The K/V of every row is left in the pool: the caller truncates to
    /// offset + accepted + 1 (`truncate_kv`) and keeps its own token list
    /// (`sequences[seq].tokens` is NOT touched here). On any error after the
    /// slots were appended the pool is truncated back to `offset`. Needs the
    /// paged pool (the GB10 serving construction); refused otherwise.
    pub fn verify_block(&mut self, seq_id: u64, block: &[u32]) -> Result<Vec<u32>, String> {
        let n = block.len();
        let vocab = self.weights.config.vocab_size();
        let logits = self.verify_block_logits(seq_id, block)?;
        Ok((0..n)
            .map(|t| sample_argmax(&logits[t * vocab..(t + 1) * vocab]).0)
            .collect())
    }

    /// `verify_block` before the argmax: the full logits of every row, `n * vocab` f32.
    pub fn verify_block_logits(&mut self, seq_id: u64, block: &[u32]) -> Result<Vec<f32>, String> {
        let n = block.len();
        if n == 0 || n > 64 {
            return Err(format!("verify_block: {n} rows (1 to 64)"));
        }
        let mgr = self
            .kv_manager
            .clone()
            .ok_or_else(|| "verify_block needs the paged KV pool".to_string())?;
        let weights = &self.weights;
        let backend: &dyn TensorBackend = &*self.tensor_backend;
        let geom = weights
            .config
            .attention_geometry()
            .map_err(|e| e.to_string())?;
        backend.check_model(&weights.config)?;
        let (num_heads, num_kv_heads, head_dim) =
            (geom.num_q_heads(), geom.num_kv_heads(), geom.head_dim());
        let (q_dim, kv_dim) = (geom.q_dim(), geom.kv_dim());
        let hidden_dim = weights.config.hidden_dim();
        let intermediate_dim = weights.config.intermediate_dim();
        let vocab = weights.config.vocab_size();
        let eps = weights.config.rms_norm_eps;
        let rope = weights.config.rope();
        // Slots for the n positions, appended up front (as the prompt read does).
        let (offset, slots, block_ids) = {
            let mut w = mgr.write();
            if w.tensor_pool().is_none() {
                return Err("verify_block needs the paged KV pool (tensor pool)".into());
            }
            let tbl = w
                .get_block_table(seq_id)
                .cloned()
                .ok_or_else(|| format!("verify_block: sequence {seq_id} has no block table"))?;
            let offset = tbl.total_tokens;
            if let Some((needed, available)) = Self::kv_pool_shortfall(&w, seq_id, n) {
                return Err(Self::kv_pool_exhausted_error(
                    &format!("verify block of {n} tokens at position {offset}"),
                    seq_id,
                    needed,
                    available,
                ));
            }
            let mut slots = Vec::with_capacity(n);
            for t in 0..n {
                slots.push(w.append_token_with_slot(seq_id).map_err(|e| {
                    format!(
                        "{KV_POOL_EXHAUSTED_PREFIX} verify append of position {} failed: {e}",
                        offset + t
                    )
                })?);
            }
            let ids = w
                .get_block_table(seq_id)
                .map(|t| t.block_ids.clone())
                .unwrap_or_default();
            (offset, slots, ids)
        };
        let run = || -> Result<Vec<f32>, String> {
            let mut states: Vec<Vec<f32>> = block
                .iter()
                .map(|&tok| {
                    let i = (tok as usize) % vocab;
                    weights.embed_tokens[i * hidden_dim..(i + 1) * hidden_dim].to_vec()
                })
                .collect();
            let mut x_flat = vec![0.0f32; n * hidden_dim];
            let flatten = |x_flat: &mut [f32], states: &[Vec<f32>]| {
                for (t, s) in states.iter().enumerate() {
                    x_flat[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(s);
                }
            };
            let mut x_norm = vec![0.0f32; n * hidden_dim];
            let mut q_b = vec![0.0f32; n * q_dim];
            let mut k_b = vec![0.0f32; n * kv_dim];
            let mut v_b = vec![0.0f32; n * kv_dim];
            let qkv_dim = q_dim + 2 * kv_dim;
            let mut qkv_b = vec![0.0f32; n * qkv_dim];
            let mut attn_b = vec![0.0f32; n * q_dim];
            let mut proj_b = vec![0.0f32; n * hidden_dim];
            let mut post_b = vec![0.0f32; n * hidden_dim];
            let mut gate_b = vec![0.0f32; n * intermediate_dim];
            let mut up_b = vec![0.0f32; n * intermediate_dim];
            let mut act_b = vec![0.0f32; n * intermediate_dim];
            let mut gu_b = vec![0.0f32; n * 2 * intermediate_dim];
            let mut mlp_b = vec![0.0f32; n * hidden_dim];
            let max_blocks = block_ids.len().max(1);
            let mut tables = vec![-1i32; n * max_blocks];
            let mut ctx = vec![0i32; n];
            for t in 0..n {
                for (j, &b) in block_ids.iter().enumerate() {
                    tables[t * max_blocks + j] =
                        i32::try_from(b).map_err(|_| "block id past i32")?;
                }
                ctx[t] = i32::try_from(offset + t + 1).map_err(|_| "context past i32")?;
            }
            for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
                flatten(&mut x_flat, &states);
                backend.rmsnorm_rows(&mut x_norm, &x_flat, &layer_w.input_layernorm, n, eps);
                if let Some(w) = layer_w.qkv_proj.as_ref() {
                    backend.matmul_batch_w(&mut qkv_b, &x_norm, w.as_ref(), n, hidden_dim, qkv_dim);
                    for t in 0..n {
                        let row = &qkv_b[t * qkv_dim..(t + 1) * qkv_dim];
                        q_b[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[..q_dim]);
                        k_b[t * kv_dim..(t + 1) * kv_dim]
                            .copy_from_slice(&row[q_dim..q_dim + kv_dim]);
                        v_b[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&row[q_dim + kv_dim..]);
                    }
                } else {
                    backend.matmul_batch_w(
                        &mut q_b,
                        &x_norm,
                        layer_w.q_proj.as_ref(),
                        n,
                        hidden_dim,
                        q_dim,
                    );
                    backend.matmul_batch_w(
                        &mut k_b,
                        &x_norm,
                        layer_w.k_proj.as_ref(),
                        n,
                        hidden_dim,
                        kv_dim,
                    );
                    backend.matmul_batch_w(
                        &mut v_b,
                        &x_norm,
                        layer_w.v_proj.as_ref(),
                        n,
                        hidden_dim,
                        kv_dim,
                    );
                }
                layer_w.apply_qk_norm_on(backend, &mut q_b, &mut k_b, head_dim, eps);
                for t in 0..n {
                    let q_t = &mut q_b[t * q_dim..(t + 1) * q_dim];
                    let k_t = &mut k_b[t * kv_dim..(t + 1) * kv_dim];
                    let v_t = &v_b[t * kv_dim..(t + 1) * kv_dim];
                    backend.apply_rope(
                        q_t,
                        k_t,
                        offset + t,
                        head_dim,
                        num_heads,
                        num_kv_heads,
                        &rope,
                    );
                    let (block_id, slot) = slots[t];
                    mgr.write()
                        .write_explicit_token_kv(block_id, layer_idx, slot, k_t, v_t)
                        .map_err(|e| {
                            format!(
                                "verify KV write layer {layer_idx} position {}: {e}",
                                offset + t
                            )
                        })?;
                }
                {
                    let r = mgr.read();
                    let pool = r.tensor_pool().ok_or("pool vanished")?;
                    backend.paged_attention_batch(
                        &mut attn_b,
                        &q_b,
                        pool,
                        &tables,
                        &ctx,
                        max_blocks,
                        n,
                        layer_idx,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                }
                backend.matmul_batch_w(
                    &mut proj_b,
                    &attn_b,
                    layer_w.o_proj.as_ref(),
                    n,
                    q_dim,
                    hidden_dim,
                );
                for t in 0..n {
                    let p = &proj_b[t * hidden_dim..(t + 1) * hidden_dim];
                    for i in 0..hidden_dim {
                        states[t][i] += p[i];
                    }
                }
                flatten(&mut x_flat, &states);
                backend.rmsnorm_rows(
                    &mut post_b,
                    &x_flat,
                    &layer_w.post_attention_layernorm,
                    n,
                    eps,
                );
                if let Some(w) = layer_w.gate_up_proj.as_ref() {
                    let gu = 2 * intermediate_dim;
                    backend.matmul_batch_w(&mut gu_b, &post_b, w.as_ref(), n, hidden_dim, gu);
                    for t in 0..n {
                        let row = &gu_b[t * gu..(t + 1) * gu];
                        gate_b[t * intermediate_dim..(t + 1) * intermediate_dim]
                            .copy_from_slice(&row[..intermediate_dim]);
                        up_b[t * intermediate_dim..(t + 1) * intermediate_dim]
                            .copy_from_slice(&row[intermediate_dim..]);
                    }
                } else {
                    backend.matmul_batch_w(
                        &mut gate_b,
                        &post_b,
                        layer_w.gate_proj.as_ref(),
                        n,
                        hidden_dim,
                        intermediate_dim,
                    );
                    backend.matmul_batch_w(
                        &mut up_b,
                        &post_b,
                        layer_w.up_proj.as_ref(),
                        n,
                        hidden_dim,
                        intermediate_dim,
                    );
                }
                backend.swiglu(&mut act_b, &gate_b, &up_b);
                backend.matmul_batch_w(
                    &mut mlp_b,
                    &act_b,
                    layer_w.down_proj.as_ref(),
                    n,
                    intermediate_dim,
                    hidden_dim,
                );
                for t in 0..n {
                    let m = &mlp_b[t * hidden_dim..(t + 1) * hidden_dim];
                    for i in 0..hidden_dim {
                        states[t][i] += m[i];
                    }
                }
            }
            flatten(&mut x_flat, &states);
            let mut final_b = vec![0.0f32; n * hidden_dim];
            backend.rmsnorm_rows(&mut final_b, &x_flat, &weights.final_norm, n, eps);
            let mut logits = vec![0.0f32; n * vocab];
            backend.matmul_batch_w(
                &mut logits,
                &final_b,
                weights.output_matrix(),
                n,
                hidden_dim,
                vocab,
            );
            Ok(logits)
        };
        let out = run();
        if out.is_err() {
            let _ = mgr.write().truncate_sequence(seq_id, offset);
        }
        out
    }

    /// Drops the K/V at or past `new_total` (pool and dense copies). The token
    /// list is the caller's.
    pub fn truncate_kv(&mut self, seq_id: u64, new_total: usize) -> Result<(), String> {
        if let Some(m) = &self.kv_manager {
            m.write().truncate_sequence(seq_id, new_total)?;
        }
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            for l in &mut seq.layers {
                let kvd = l.cached_k.first().map_or(0, |r| r.len());
                l.cached_k.truncate(new_total);
                l.cached_v.truncate(new_total);
                l.flat_k.truncate(new_total * kvd);
                l.flat_v.truncate(new_total * kvd);
            }
        }
        Ok(())
    }

    /// Forwards the sequence's last token (the pending one) at its position and
    /// returns the logits of the next position; the token list is untouched.
    pub fn decode_pending(&mut self, seq_id: u64) -> Result<Vec<f32>, String> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| format!("decode_pending: sequence {seq_id} not found"))?;
        let pos = seq
            .tokens
            .len()
            .checked_sub(1)
            .ok_or_else(|| "decode_pending: empty sequence".to_string())?;
        let tok = seq.tokens[pos];
        let hidden = Self::try_forward_token_impl_paged(
            &self.weights,
            &*self.tensor_backend,
            tok,
            pos,
            seq,
            seq_id,
            self.kv_manager.as_ref(),
        )?;
        Ok(Self::compute_logits_impl(
            &self.weights,
            &*self.tensor_backend,
            &hidden,
        ))
    }

    pub fn forward_token_impl(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::forward_token_impl_paged(weights, backend, token_id, pos, seq_state, 0, None)
    }

    /// Fresh physical blocks `appends` consecutive appends on `seq_id` would
    /// allocate. The rule lives in one place, `AienKvManager::
    /// blocks_needed_for_appends`; this only forwards to it.
    fn kv_blocks_needed_for_appends(kv: &AienKvManager, seq_id: u64, appends: usize) -> usize {
        kv.blocks_needed_for_appends(seq_id, appends)
    }

    /// `Some((needed, available))` when `appends` appends to `seq_id` need more
    /// new blocks than the pool has free, `None` when they fit. Read-only.
    fn kv_pool_shortfall(
        kv: &AienKvManager,
        seq_id: u64,
        appends: usize,
    ) -> Option<(usize, usize)> {
        let needed = Self::kv_blocks_needed_for_appends(kv, seq_id, appends);
        let available = kv.available_blocks();
        (needed > available).then_some((needed, available))
    }

    /// Message for a capacity failure; always starts with `KV_POOL_EXHAUSTED_PREFIX`.
    fn kv_pool_exhausted_error(what: &str, seq_id: u64, needed: usize, available: usize) -> String {
        format!(
            "{KV_POOL_EXHAUSTED_PREFIX} {what} for sequence {seq_id} needs {needed} new KV blocks \
             but only {available} are free; the block table, the pool and the sequence state are unchanged"
        )
    }

    /// Element type every new sequence's K/V are rounded to the moment they are
    /// produced: the attached pool's dtype (what `write_token_kv` stores), else
    /// `kv_dtype`. One rounding point keeps dense cache, pool and attention on the
    /// same bits (PREFILL-E2E-2).
    pub fn storage_dtype(&self) -> KvDType {
        self.kv_manager
            .as_ref()
            .and_then(|m| m.read().tensor_pool().map(|p| p.config().dtype))
            .unwrap_or(self.kv_dtype)
    }

    /// Forward pass executing one token forward pass through the transformer with optional paged COW KV.
    ///
    /// Panics when the paged KV pool cannot supply the block this token needs
    /// (see `try_forward_token_impl_paged`, which returns the error instead).
    pub fn forward_token_impl_paged(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
        seq_id: u64,
        kv_manager: Option<&SharedKvManager>,
    ) -> Vec<f32> {
        Self::try_forward_token_impl_paged(
            weights, backend, token_id, pos, seq_state, seq_id, kv_manager,
        )
        .unwrap_or_else(|e| panic!("forward_token_impl_paged: {e}"))
    }

    /// Same as `forward_token_impl_paged`, but returns a loud `Err` (message
    /// starting with `KV_POOL_EXHAUSTED_PREFIX`) when the paged KV pool has no
    /// block for this token, instead of continuing with no stored K/V
    /// (PREFILL-I25). The capacity check runs before anything is mutated, so a
    /// failed call leaves the block table, the pool and `seq_state` unchanged.
    /// A sequence with no block table in the manager still takes the dense
    /// per-layer cache path, as before.
    pub fn try_forward_token_impl_paged(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
        seq_id: u64,
        kv_manager: Option<&SharedKvManager>,
    ) -> Result<Vec<f32>, String> {
        let hidden_dim = weights.config.hidden_dim();
        // Checked once per forward: zero heads, non-divisible heads or an overflowing width
        // are refused here instead of becoming a wrong slice later.
        let geom = weights
            .config
            .attention_geometry()
            .map_err(|e| e.to_string())?;
        backend.check_model(&weights.config)?;
        let num_heads = geom.num_q_heads();
        let num_kv_heads = geom.num_kv_heads();
        let head_dim = geom.head_dim();
        let q_dim = geom.q_dim();
        let kv_dim = geom.kv_dim();
        let intermediate_dim = weights.config.intermediate_dim();
        let eps = weights.config.rms_norm_eps;
        let rope = weights.config.rope();

        let token_idx = (token_id as usize) % weights.config.vocab_size();
        let mut x =
            weights.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].to_vec();

        let mut x_norm = vec![0.0f32; hidden_dim];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let qkv_dim = q_dim + 2 * kv_dim;
        let mut qkv = vec![0.0f32; qkv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut attn_proj = vec![0.0f32; hidden_dim];
        let mut post_norm = vec![0.0f32; hidden_dim];
        let mut gate = vec![0.0f32; intermediate_dim];
        let mut up = vec![0.0f32; intermediate_dim];
        let mut activated = vec![0.0f32; intermediate_dim];
        let mut gate_up = vec![0.0f32; 2 * intermediate_dim];
        let mut mlp_out = vec![0.0f32; hidden_dim];

        // Reserve this token's slot. A full pool is a loud error (checked before
        // anything is mutated), never a silent fall back to "no stored K/V".
        // A sequence with no block table in the manager keeps the dense
        // per-layer cache path, as before.
        let block_slot = match kv_manager {
            Some(mgr) => {
                let mut w = mgr.write();
                if w.get_block_table(seq_id).is_some() {
                    if let Some((needed, available)) = Self::kv_pool_shortfall(&w, seq_id, 1) {
                        return Err(Self::kv_pool_exhausted_error(
                            &format!("decode step at position {pos}"),
                            seq_id,
                            needed,
                            available,
                        ));
                    }
                    let slot = w.append_token_with_slot(seq_id).map_err(|e| {
                        format!(
                            "{KV_POOL_EXHAUSTED_PREFIX} append of position {pos} for sequence \
                             {seq_id} failed after the capacity check: {e}"
                        )
                    })?;
                    Some(slot)
                } else {
                    None
                }
            }
            None => None,
        };

        for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
            backend.rmsnorm(&mut x_norm, &x, &layer_w.input_layernorm, eps);
            if let Some(w) = layer_w.qkv_proj.as_ref() {
                backend.matmul_vec_w(&mut qkv, &x_norm, w.as_ref(), qkv_dim, hidden_dim);
                q.copy_from_slice(&qkv[..q_dim]);
                k.copy_from_slice(&qkv[q_dim..q_dim + kv_dim]);
                v.copy_from_slice(&qkv[q_dim + kv_dim..]);
            } else {
                backend.matmul_vec_w(&mut q, &x_norm, layer_w.q_proj.as_ref(), q_dim, hidden_dim);
                backend.matmul_vec_w(&mut k, &x_norm, layer_w.k_proj.as_ref(), kv_dim, hidden_dim);
                backend.matmul_vec_w(&mut v, &x_norm, layer_w.v_proj.as_ref(), kv_dim, hidden_dim);
            }

            layer_w.apply_qk_norm_on(backend, &mut q, &mut k, head_dim, eps);
            backend.apply_rope(
                &mut q,
                &mut k,
                pos,
                head_dim,
                num_heads,
                num_kv_heads,
                &rope,
            );
            // Round once, at birth: pool write, dense cache and attention all see
            // the same bits (PREFILL-E2E-2).
            round_to_dtype_in_place(seq_state.kv_dtype, &mut k);
            round_to_dtype_in_place(seq_state.kv_dtype, &mut v);

            if let (Some((block_id, slot)), Some(mgr)) = (block_slot, kv_manager) {
                {
                    let mut w = mgr.write();
                    // A manager with no physical pool has nowhere to store K/V;
                    // that case is skipped, as before. With a pool the write
                    // cannot fail silently any more.
                    if w.tensor_pool().is_some() {
                        w.write_explicit_token_kv(block_id, layer_idx, slot, &k, &v)
                            .map_err(|e| {
                                format!(
                                    "KV write failed for sequence {seq_id} layer {layer_idx} \
                                     position {pos}: {e}"
                                )
                            })?;
                    }
                }

                let mgr_read = mgr.read();
                if let (Some(pool), Some(table)) =
                    (mgr_read.tensor_pool(), mgr_read.get_block_table(seq_id))
                {
                    let total_tokens = table.total_tokens;
                    backend.paged_attention(
                        &mut attn_out,
                        &q,
                        pool,
                        &table.block_ids,
                        total_tokens,
                        layer_idx,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                }
            } else {
                let kv_cache = &mut seq_state.layers[layer_idx];
                kv_cache.cached_k.push(k.clone());
                kv_cache.cached_v.push(v.clone());
                kv_cache.flat_k.extend_from_slice(&k);
                kv_cache.flat_v.extend_from_slice(&v);

                let seq_len = kv_cache.cached_k.len();
                backend.gqa_attention(
                    &mut attn_out,
                    &q,
                    &kv_cache.flat_k,
                    &kv_cache.flat_v,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }

            backend.matmul_vec_w(
                &mut attn_proj,
                &attn_out,
                layer_w.o_proj.as_ref(),
                hidden_dim,
                q_dim,
            );
            for i in 0..hidden_dim {
                x[i] += attn_proj[i];
            }

            backend.rmsnorm(&mut post_norm, &x, &layer_w.post_attention_layernorm, eps);
            if let Some(w) = layer_w.gate_up_proj.as_ref() {
                backend.matmul_vec_w(
                    &mut gate_up,
                    &post_norm,
                    w.as_ref(),
                    2 * intermediate_dim,
                    hidden_dim,
                );
                backend.swiglu(
                    &mut activated,
                    &gate_up[..intermediate_dim],
                    &gate_up[intermediate_dim..],
                );
            } else {
                backend.matmul_vec_w(
                    &mut gate,
                    &post_norm,
                    layer_w.gate_proj.as_ref(),
                    intermediate_dim,
                    hidden_dim,
                );
                backend.matmul_vec_w(
                    &mut up,
                    &post_norm,
                    layer_w.up_proj.as_ref(),
                    intermediate_dim,
                    hidden_dim,
                );
                backend.swiglu(&mut activated, &gate, &up);
            }
            backend.matmul_vec_w(
                &mut mlp_out,
                &activated,
                layer_w.down_proj.as_ref(),
                hidden_dim,
                intermediate_dim,
            );

            for i in 0..hidden_dim {
                x[i] += mlp_out[i];
            }
        }

        let mut x_final = vec![0.0f32; hidden_dim];
        backend.rmsnorm(&mut x_final, &x, &weights.final_norm, eps);
        Ok(x_final)
    }

    pub fn compute_logits_impl(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        hidden_state: &[f32],
    ) -> Vec<f32> {
        let vocab_size = weights.config.vocab_size();
        let hidden_dim = weights.config.hidden_dim();
        let mut logits = vec![0.0f32; vocab_size];
        backend.compute_logits_w(
            &mut logits,
            hidden_state,
            weights.output_matrix(),
            vocab_size,
            hidden_dim,
        );
        logits
    }

    /// Prefill all prompt tokens layer-by-layer rather than token-by-token.
    pub fn prefill_prompt_layer_by_layer(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        prompt_tokens: &[u32],
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::prefill_prompt_layer_by_layer_paged(
            weights,
            backend,
            prompt_tokens,
            seq_state,
            0,
            None,
        )
        .unwrap_or_else(|e| panic!("prefill without a KV manager cannot hit the KV pool: {e}"))
    }

    /// Prefill all prompt tokens layer-by-layer with optional paged COW KV registration.
    ///
    /// In paged mode, positions beyond the block table's allocated length are
    /// appended to the table. If the pool cannot supply the blocks those
    /// appends need, the call fails with an `Err` starting with
    /// `KV_POOL_EXHAUSTED_PREFIX` BEFORE anything is mutated: the block table,
    /// the free-block count and `seq_state` stay exactly as they were, so the
    /// same call can be retried once blocks are freed (PREFILL-I25).
    pub fn prefill_prompt_layer_by_layer_paged(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        prompt_tokens: &[u32],
        seq_state: &mut SequenceState,
        seq_id: u64,
        kv_manager: Option<&SharedKvManager>,
    ) -> Result<Vec<f32>, String> {
        let n = prompt_tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Tokens already in this sequence (earlier prefill chunks). This chunk's
        // token t sits at absolute position `offset + t`: that position drives RoPE,
        // the paged block/slot, and the causal attention length.
        let offset = seq_state.tokens.len();
        let kv_dtype = seq_state.kv_dtype;
        if n == 1 && offset == 0 {
            // The token is recorded only after the forward pass succeeded, so a
            // loud KV failure leaves `seq_state` untouched.
            let hidden = Self::try_forward_token_impl_paged(
                weights,
                backend,
                prompt_tokens[0],
                0,
                seq_state,
                seq_id,
                kv_manager,
            )?;
            seq_state.tokens.push(prompt_tokens[0]);
            return Ok(hidden);
        }

        let hidden_dim = weights.config.hidden_dim();
        // Checked once per forward: zero heads, non-divisible heads or an overflowing width
        // are refused here instead of becoming a wrong slice later.
        let geom = weights
            .config
            .attention_geometry()
            .map_err(|e| e.to_string())?;
        backend.check_model(&weights.config)?;
        let num_heads = geom.num_q_heads();
        let num_kv_heads = geom.num_kv_heads();
        let head_dim = geom.head_dim();
        let q_dim = geom.q_dim();
        let kv_dim = geom.kv_dim();
        let intermediate_dim = weights.config.intermediate_dim();
        let eps = weights.config.rms_norm_eps;
        let rope = weights.config.rope();

        // Paged mode: one physical (block, slot) per absolute position, resolved once
        // (not per layer). Positions inside the table's allocated length (the
        // scheduler allocates the whole prompt up front) use the existing blocks;
        // positions beyond it are appended to the table. Resolved first, before
        // `seq_state` is touched, so a pool that cannot hold the appends fails the
        // whole call cleanly (PREFILL-I25).
        let num_layers = weights.config.num_layers;
        let mut paged_slots: Vec<Option<(usize, usize)>> = vec![None; n];
        if let Some(mgr) = kv_manager {
            let mut w = mgr.write();
            if w.tensor_pool().is_some() {
                let block_size = w.block_size();
                if let Some(tbl) = w.get_block_table(seq_id).cloned() {
                    let covered =
                        |p: usize| p < tbl.total_tokens && p / block_size < tbl.block_ids.len();
                    let appends = (0..n).filter(|&t| !covered(offset + t)).count();
                    if let Some((needed, available)) = Self::kv_pool_shortfall(&w, seq_id, appends)
                    {
                        return Err(Self::kv_pool_exhausted_error(
                            &format!("prefill of {n} tokens at position {offset}"),
                            seq_id,
                            needed,
                            available,
                        ));
                    }
                    for (t, slot_out) in paged_slots.iter_mut().enumerate() {
                        let p = offset + t;
                        *slot_out = if covered(p) {
                            Some((tbl.block_ids[p / block_size], p % block_size))
                        } else {
                            let slot = w.append_token_with_slot(seq_id).map_err(|e| {
                                format!(
                                    "{KV_POOL_EXHAUSTED_PREFIX} append of position {p} for \
                                     sequence {seq_id} failed after the capacity check: {e}"
                                )
                            })?;
                            Some(slot)
                        };
                    }
                }
            }
        }

        seq_state.tokens.extend_from_slice(prompt_tokens);

        let mut states: Vec<Vec<f32>> = prompt_tokens
            .iter()
            .map(|&tok| {
                let token_idx = (tok as usize) % weights.config.vocab_size();
                let slice =
                    &weights.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
                slice.to_vec()
            })
            .collect();

        let mut x_norm_batch = vec![0.0f32; n * hidden_dim];
        let mut q_batch = vec![0.0f32; n * q_dim];
        let mut k_batch = vec![0.0f32; n * kv_dim];
        let mut v_batch = vec![0.0f32; n * kv_dim];
        let qkv_dim = q_dim + 2 * kv_dim;
        let mut qkv_batch = vec![0.0f32; n * qkv_dim];
        let mut attn_out_batch = vec![0.0f32; n * q_dim];
        let mut attn_proj_batch = vec![0.0f32; n * hidden_dim];
        let mut post_norm_batch = vec![0.0f32; n * hidden_dim];
        let mut gate_batch = vec![0.0f32; n * intermediate_dim];
        let mut up_batch = vec![0.0f32; n * intermediate_dim];
        let mut act_batch = vec![0.0f32; n * intermediate_dim];
        let mut gate_up_batch = vec![0.0f32; n * 2 * intermediate_dim];
        let mut mlp_out_batch = vec![0.0f32; n * hidden_dim];

        // The dense per-layer cache must hold K/V for positions 0..offset before this
        // chunk's attention runs. Paged mode clears it after each prefill, so rebuild
        // it from the physical pool.
        if seq_state.layers.len() != num_layers {
            seq_state.layers = vec![LayerKvCache::default(); num_layers];
            if offset > 0 {
                if let Some(mgr) = kv_manager {
                    let r = mgr.read();
                    if let (Some(pool), Some(tbl)) = (r.tensor_pool(), r.get_block_table(seq_id)) {
                        for (layer_idx, cache) in seq_state.layers.iter_mut().enumerate() {
                            let (k, v) = pool.gather_layer_kv(&tbl.block_ids, offset, layer_idx);
                            for (k_t, v_t) in k.chunks(kv_dim).zip(v.chunks(kv_dim)) {
                                cache.cached_k.push(k_t.to_vec());
                                cache.cached_v.push(v_t.to_vec());
                            }
                            cache.flat_k = k;
                            cache.flat_v = v;
                        }
                    }
                }
            }
        }

        // Contiguous copy of the residual rows for the row-batched norms (one launch per
        // chunk and layer instead of one per token).
        let mut x_flat = vec![0.0f32; n * hidden_dim];
        let flatten = |x_flat: &mut [f32], states: &[Vec<f32>]| {
            for (t, s) in states.iter().enumerate() {
                x_flat[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(s);
            }
        };

        for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
            let kv_cache = &mut seq_state.layers[layer_idx];

            flatten(&mut x_flat, &states);
            backend.rmsnorm_rows(&mut x_norm_batch, &x_flat, &layer_w.input_layernorm, n, eps);

            if let Some(w) = layer_w.qkv_proj.as_ref() {
                backend.matmul_batch_w(
                    &mut qkv_batch,
                    &x_norm_batch,
                    w.as_ref(),
                    n,
                    hidden_dim,
                    qkv_dim,
                );
                for t in 0..n {
                    let row = &qkv_batch[t * qkv_dim..(t + 1) * qkv_dim];
                    q_batch[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[..q_dim]);
                    k_batch[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&row[q_dim..q_dim + kv_dim]);
                    v_batch[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&row[q_dim + kv_dim..]);
                }
            } else {
                backend.matmul_batch_w(
                    &mut q_batch,
                    &x_norm_batch,
                    layer_w.q_proj.as_ref(),
                    n,
                    hidden_dim,
                    q_dim,
                );
                backend.matmul_batch_w(
                    &mut k_batch,
                    &x_norm_batch,
                    layer_w.k_proj.as_ref(),
                    n,
                    hidden_dim,
                    kv_dim,
                );
                backend.matmul_batch_w(
                    &mut v_batch,
                    &x_norm_batch,
                    layer_w.v_proj.as_ref(),
                    n,
                    hidden_dim,
                    kv_dim,
                );
            }

            // q/k norm is per head with one weight, so the whole chunk is one `heads` run
            // per tensor (n * num_heads rows for q, n * num_kv_heads for k).
            layer_w.apply_qk_norm_on(backend, &mut q_batch, &mut k_batch, head_dim, eps);

            for t in 0..n {
                let q_t = &mut q_batch[t * q_dim..(t + 1) * q_dim];
                let k_t = &mut k_batch[t * kv_dim..(t + 1) * kv_dim];
                let v_t = &mut v_batch[t * kv_dim..(t + 1) * kv_dim];

                let pos = offset + t;
                backend.apply_rope(q_t, k_t, pos, head_dim, num_heads, num_kv_heads, &rope);
                // Round once, at birth, so this chunk's in-flight K/V and the earlier
                // chunks gathered back from the pool carry the same bits (PREFILL-E2E-2).
                round_to_dtype_in_place(kv_dtype, k_t);
                round_to_dtype_in_place(kv_dtype, v_t);

                if let (Some((block_id, slot)), Some(mgr)) = (paged_slots[t], kv_manager) {
                    // Slots exist only when a physical pool is attached, so this
                    // write has nowhere to fail silently; any error is loud.
                    mgr.write()
                        .write_explicit_token_kv(block_id, layer_idx, slot, k_t, v_t)
                        .map_err(|e| {
                            format!(
                                "KV write failed for sequence {seq_id} layer {layer_idx} \
                                 position {pos}: {e}"
                            )
                        })?;
                }

                kv_cache.cached_k.push(k_t.to_vec());
                kv_cache.cached_v.push(v_t.to_vec());
                kv_cache.flat_k.extend_from_slice(k_t);
                kv_cache.flat_v.extend_from_slice(v_t);
            }
            // The chunk's attention in one call: the cache now holds offset + n tokens and row
            // t attends over the first offset + t + 1 of them (omega stages the KV once and
            // sends the rows in a few launches, omega#406, instead of one launch per token).
            backend.gqa_attention_rows(
                &mut attn_out_batch,
                &q_batch,
                &kv_cache.flat_k,
                &kv_cache.flat_v,
                offset,
                n,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            backend.matmul_batch_w(
                &mut attn_proj_batch,
                &attn_out_batch,
                layer_w.o_proj.as_ref(),
                n,
                q_dim,
                hidden_dim,
            );
            for t in 0..n {
                let proj_t = &attn_proj_batch[t * hidden_dim..(t + 1) * hidden_dim];
                for i in 0..hidden_dim {
                    states[t][i] += proj_t[i];
                }
            }

            flatten(&mut x_flat, &states);
            backend.rmsnorm_rows(
                &mut post_norm_batch,
                &x_flat,
                &layer_w.post_attention_layernorm,
                n,
                eps,
            );

            if let Some(w) = layer_w.gate_up_proj.as_ref() {
                let gu = 2 * intermediate_dim;
                backend.matmul_batch_w(
                    &mut gate_up_batch,
                    &post_norm_batch,
                    w.as_ref(),
                    n,
                    hidden_dim,
                    gu,
                );
                // Un-interleave the fused rows into the gate and up planes so the whole
                // chunk is one elementwise launch.
                for t in 0..n {
                    let row = &gate_up_batch[t * gu..(t + 1) * gu];
                    gate_batch[t * intermediate_dim..(t + 1) * intermediate_dim]
                        .copy_from_slice(&row[..intermediate_dim]);
                    up_batch[t * intermediate_dim..(t + 1) * intermediate_dim]
                        .copy_from_slice(&row[intermediate_dim..]);
                }
                backend.swiglu(&mut act_batch, &gate_batch, &up_batch);
            } else {
                backend.matmul_batch_w(
                    &mut gate_batch,
                    &post_norm_batch,
                    layer_w.gate_proj.as_ref(),
                    n,
                    hidden_dim,
                    intermediate_dim,
                );
                backend.matmul_batch_w(
                    &mut up_batch,
                    &post_norm_batch,
                    layer_w.up_proj.as_ref(),
                    n,
                    hidden_dim,
                    intermediate_dim,
                );
                backend.swiglu(&mut act_batch, &gate_batch, &up_batch);
            }

            backend.matmul_batch_w(
                &mut mlp_out_batch,
                &act_batch,
                layer_w.down_proj.as_ref(),
                n,
                intermediate_dim,
                hidden_dim,
            );
            for t in 0..n {
                let mlp_t = &mlp_out_batch[t * hidden_dim..(t + 1) * hidden_dim];
                for i in 0..hidden_dim {
                    states[t][i] += mlp_t[i];
                }
            }
        }

        // Drop the dense copy only when the physical pool holds the K/V (it is rebuilt
        // from the pool if a later chunk continues this sequence).
        if kv_manager.is_some_and(|m| m.read().tensor_pool().is_some()) {
            seq_state.layers.clear();
        }

        let last_x = &states[n - 1];
        let mut x_final = vec![0.0f32; hidden_dim];
        backend.rmsnorm(&mut x_final, last_x, &weights.final_norm, eps);
        Ok(x_final)
    }

    /// Single-token forward pass executing through the configured TensorBackend trait.
    pub fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::forward_token_impl(
            &self.weights,
            &*self.tensor_backend,
            token_id,
            pos,
            seq_state,
        )
    }

    /// Computes logits projection executing through the configured TensorBackend trait.
    pub fn compute_logits(&self, hidden_state: &[f32]) -> Vec<f32> {
        Self::compute_logits_impl(&self.weights, &*self.tensor_backend, hidden_state)
    }

    /// Forward pass executing one token decode step across a batch of D sequences concurrently.
    /// Employs batched GEMM (M=D) for all linear projections and batched paged attention.
    ///
    /// Every id must already have backend state (from a prefill or from
    /// `fork_sequence`). An unknown id, or one with no tokens, is an error and
    /// nothing is computed: decoding it would start from token 1 at position 0
    /// over an empty cache (PREFILL-E2E C4).
    pub fn forward_decode_batch(
        &mut self,
        decode_req_ids: &[u64],
    ) -> Result<Vec<DecodeOutput>, String> {
        self.forward_decode_batch_impl(decode_req_ids)
            .map(|(outputs, _logits, _vocab)| outputs)
    }

    /// Same as `forward_decode_batch`, also returning each sequence's logits
    /// (in `decode_req_ids` order, minus any sequence ended as Preempted for
    /// lack of a KV slot, which gets no logits) for parity checks.
    pub fn forward_decode_batch_with_logits(
        &mut self,
        decode_req_ids: &[u64],
    ) -> Result<(Vec<DecodeOutput>, Vec<Vec<f32>>), String> {
        let (outputs, logits_batch, vocab_size) = self.forward_decode_batch_impl(decode_req_ids)?;
        let logits = if vocab_size == 0 {
            Vec::new()
        } else {
            logits_batch
                .chunks(vocab_size)
                .map(|row| row.to_vec())
                .collect()
        };
        Ok((outputs, logits))
    }

    fn forward_decode_batch_impl(
        &mut self,
        decode_req_ids: &[u64],
    ) -> Result<(Vec<DecodeOutput>, Vec<f32>, usize), String> {
        if decode_req_ids.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }

        let hidden_dim = self.weights.config.hidden_dim();
        // Checked once per forward: zero heads, non-divisible heads or an overflowing width
        // are refused here instead of becoming a wrong slice later.
        let geom = self
            .weights
            .config
            .attention_geometry()
            .map_err(|e| e.to_string())?;
        self.tensor_backend.check_model(&self.weights.config)?;
        let num_heads = geom.num_q_heads();
        let num_kv_heads = geom.num_kv_heads();
        let head_dim = geom.head_dim();
        let q_dim = geom.q_dim();
        let kv_dim = geom.kv_dim();
        let intermediate_dim = self.weights.config.intermediate_dim();
        let eps = self.weights.config.rms_norm_eps;
        let rope = self.weights.config.rope();
        let vocab_size = self.weights.config.vocab_size();

        // 1. Gather active sequence requests and their current positions / last tokens
        // Validated before anything is written, so a refused batch leaves no state behind.
        let mut valid_reqs = Vec::with_capacity(decode_req_ids.len());
        for &req_id in decode_req_ids {
            self.commit_pending_prefill_token(req_id);
            let seq = self.sequences.get(&req_id).ok_or_else(|| {
                format!(
                    "decode refused: request {} has no backend sequence state \
                     (never prefilled and never forked via fork_sequence)",
                    req_id
                )
            })?;
            let last_token = *seq.tokens.last().ok_or_else(|| {
                format!(
                    "decode refused: request {} has no tokens (no prefilled prompt)",
                    req_id
                )
            })?;
            let pos = seq.tokens.len() - 1;
            valid_reqs.push((req_id, last_token, pos));
        }

        // 1b. Reserve append slots in KV cache for this decode step. A sequence
        // whose slot cannot be reserved (pool exhausted) has nowhere to put this
        // step's K/V, so it is not computed and gets no token: it is ended as
        // Preempted and the scheduler frees its KV. Before the backend owned
        // decode K/V, the scheduler's append check did the same
        // (aien-scheduler/src/lib.rs, `!backend.manages_kv_cache()` branch).
        let mut slot_failures: Vec<DecodeOutput> = Vec::new();
        let block_slots: Vec<Option<(usize, usize)>> = if let Some(mgr) = &self.kv_manager {
            let mut write_mgr = mgr.write();
            let mut kept_reqs = Vec::with_capacity(valid_reqs.len());
            let mut kept_slots = Vec::with_capacity(valid_reqs.len());
            for req in valid_reqs.drain(..) {
                match write_mgr.append_token_with_slot(req.0) {
                    Ok(slot) => {
                        kept_reqs.push(req);
                        kept_slots.push(Some(slot));
                    }
                    Err(_) => {
                        let total_tokens = self.sequences.get(&req.0).map_or(0, |s| s.tokens.len());
                        slot_failures.push(DecodeOutput::Finished {
                            request_id: req.0,
                            reason: FinishReason::Preempted,
                            total_tokens,
                        });
                    }
                }
            }
            valid_reqs = kept_reqs;
            kept_slots
        } else {
            vec![None; valid_reqs.len()]
        };

        let d = valid_reqs.len();
        if d == 0 {
            return Ok((slot_failures, Vec::new(), 0));
        }

        // 2. Allocate batch activations
        let mut x = vec![0.0f32; d * hidden_dim];
        let mut x_norm = vec![0.0f32; d * hidden_dim];
        let mut q_batch = vec![0.0f32; d * q_dim];
        let mut k_batch = vec![0.0f32; d * kv_dim];
        let mut v_batch = vec![0.0f32; d * kv_dim];
        let qkv_dim = q_dim + 2 * kv_dim;
        let mut qkv_batch = vec![0.0f32; d * qkv_dim];
        let mut attn_out_batch = vec![0.0f32; d * q_dim];
        let mut attn_proj_batch = vec![0.0f32; d * hidden_dim];
        let mut post_norm_batch = vec![0.0f32; d * hidden_dim];
        let mut gate_batch = vec![0.0f32; d * intermediate_dim];
        let mut up_batch = vec![0.0f32; d * intermediate_dim];
        let mut act_batch = vec![0.0f32; d * intermediate_dim];
        let mut gate_up_batch = vec![0.0f32; d * 2 * intermediate_dim];
        let mut mlp_out_batch = vec![0.0f32; d * hidden_dim];
        let mut logits_batch = vec![0.0f32; d * vocab_size];

        // 3. Populate initial embeddings
        for (i, &(_seq_id, last_token, _pos)) in valid_reqs.iter().enumerate() {
            self.weights
                .embed_token(last_token, &mut x[i * hidden_dim..(i + 1) * hidden_dim]);
        }

        // 4. (Decode slots were reserved in step 1b.)

        // 5. Transformer layer loop
        for (layer_idx, layer_w) in self.weights.layers.iter().enumerate() {
            // a. Pre-attention RMSNorm
            for i in 0..d {
                self.tensor_backend.rmsnorm(
                    &mut x_norm[i * hidden_dim..(i + 1) * hidden_dim],
                    &x[i * hidden_dim..(i + 1) * hidden_dim],
                    &layer_w.input_layernorm,
                    eps,
                );
            }

            // b. Batched Q, K, V projections (M = D)
            if let Some(w) = layer_w.qkv_proj.as_ref() {
                self.tensor_backend.matmul_batch_w(
                    &mut qkv_batch,
                    &x_norm,
                    w.as_ref(),
                    d,
                    hidden_dim,
                    qkv_dim,
                );
                for t in 0..d {
                    let row = &qkv_batch[t * qkv_dim..(t + 1) * qkv_dim];
                    q_batch[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[..q_dim]);
                    k_batch[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&row[q_dim..q_dim + kv_dim]);
                    v_batch[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&row[q_dim + kv_dim..]);
                }
            } else {
                self.tensor_backend.matmul_batch_w(
                    &mut q_batch,
                    &x_norm,
                    layer_w.q_proj.as_ref(),
                    d,
                    hidden_dim,
                    q_dim,
                );
                self.tensor_backend.matmul_batch_w(
                    &mut k_batch,
                    &x_norm,
                    layer_w.k_proj.as_ref(),
                    d,
                    hidden_dim,
                    kv_dim,
                );
                self.tensor_backend.matmul_batch_w(
                    &mut v_batch,
                    &x_norm,
                    layer_w.v_proj.as_ref(),
                    d,
                    hidden_dim,
                    kv_dim,
                );
            }

            // c. RoPE across all D sequences at their respective positions, then
            //    round K/V once at birth (PREFILL-E2E-2).
            for (i, &(seq_id, _tok, pos)) in valid_reqs.iter().enumerate() {
                let kv_dtype = self
                    .sequences
                    .get(&seq_id)
                    .map(|s| s.kv_dtype)
                    .unwrap_or_else(|| self.storage_dtype());
                layer_w.apply_qk_norm_on(
                    self.tensor_backend.as_ref(),
                    &mut q_batch[i * q_dim..(i + 1) * q_dim],
                    &mut k_batch[i * kv_dim..(i + 1) * kv_dim],
                    head_dim,
                    eps,
                );
                self.tensor_backend.apply_rope(
                    &mut q_batch[i * q_dim..(i + 1) * q_dim],
                    &mut k_batch[i * kv_dim..(i + 1) * kv_dim],
                    pos,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    &rope,
                );
                round_to_dtype_in_place(kv_dtype, &mut k_batch[i * kv_dim..(i + 1) * kv_dim]);
                round_to_dtype_in_place(kv_dtype, &mut v_batch[i * kv_dim..(i + 1) * kv_dim]);
            }

            // d. Write K & V into KV cache
            if let Some(mgr) = &self.kv_manager {
                let mut write_mgr = mgr.write();
                for (i, slot_opt) in block_slots.iter().enumerate() {
                    if let Some((block_id, slot)) = *slot_opt {
                        let k_slice = &k_batch[i * kv_dim..(i + 1) * kv_dim];
                        let v_slice = &v_batch[i * kv_dim..(i + 1) * kv_dim];
                        let _ = write_mgr
                            .write_explicit_token_kv(block_id, layer_idx, slot, k_slice, v_slice);
                    }
                }
            } else {
                for (i, &(seq_id, _, _)) in valid_reqs.iter().enumerate() {
                    if let Some(seq) = self.sequences.get_mut(&seq_id) {
                        let kv_cache = &mut seq.layers[layer_idx];
                        let k_slice = &k_batch[i * kv_dim..(i + 1) * kv_dim];
                        let v_slice = &v_batch[i * kv_dim..(i + 1) * kv_dim];
                        kv_cache.cached_k.push(k_slice.to_vec());
                        kv_cache.cached_v.push(v_slice.to_vec());
                        kv_cache.flat_k.extend_from_slice(k_slice);
                        kv_cache.flat_v.extend_from_slice(v_slice);
                    }
                }
            }

            // e. Attention execution
            let paged_computed = if let Some(mgr) = &self.kv_manager {
                let read_mgr = mgr.read();
                if let Some(pool) = read_mgr.tensor_pool() {
                    let mut block_tables = Vec::with_capacity(d);
                    let mut context_lens = Vec::with_capacity(d);
                    let mut max_blocks = 1;
                    for &(seq_id, _, _pos) in &valid_reqs {
                        if let Some(table) = read_mgr.get_block_table(seq_id) {
                            if table.block_ids.len() > max_blocks {
                                max_blocks = table.block_ids.len();
                            }
                            block_tables.push(table.block_ids.clone());
                            context_lens.push(table.total_tokens as i32);
                        } else {
                            block_tables.push(Vec::new());
                            context_lens.push(0);
                        }
                    }

                    let mut flat_block_tables = vec![-1i32; d * max_blocks];
                    for (i, bt) in block_tables.iter().enumerate() {
                        for (j, &blk) in bt.iter().enumerate() {
                            flat_block_tables[i * max_blocks + j] = blk as i32;
                        }
                    }

                    self.tensor_backend.paged_attention_batch(
                        &mut attn_out_batch,
                        &q_batch,
                        pool,
                        &flat_block_tables,
                        &context_lens,
                        max_blocks,
                        d,
                        layer_idx,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if !paged_computed {
                for (i, &(seq_id, _, _pos)) in valid_reqs.iter().enumerate() {
                    let q_slice = &q_batch[i * q_dim..(i + 1) * q_dim];
                    let out_slice = &mut attn_out_batch[i * q_dim..(i + 1) * q_dim];
                    if let Some(seq) = self.sequences.get(&seq_id) {
                        let kv_cache = &seq.layers[layer_idx];
                        let seq_len = kv_cache.cached_k.len();
                        self.tensor_backend.gqa_attention(
                            out_slice,
                            q_slice,
                            &kv_cache.flat_k,
                            &kv_cache.flat_v,
                            seq_len,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        );
                    }
                }
            }

            // f. Attention output projection (M = D)
            self.tensor_backend.matmul_batch_w(
                &mut attn_proj_batch,
                &attn_out_batch,
                layer_w.o_proj.as_ref(),
                d,
                q_dim,
                hidden_dim,
            );

            // g. Residual connection
            for idx in 0..x.len() {
                x[idx] += attn_proj_batch[idx];
            }

            // h. Post-attention RMSNorm
            for i in 0..d {
                self.tensor_backend.rmsnorm(
                    &mut post_norm_batch[i * hidden_dim..(i + 1) * hidden_dim],
                    &x[i * hidden_dim..(i + 1) * hidden_dim],
                    &layer_w.post_attention_layernorm,
                    eps,
                );
            }

            // i. MLP Gate & Up projections (M = D)
            if let Some(w) = layer_w.gate_up_proj.as_ref() {
                let gu = 2 * intermediate_dim;
                self.tensor_backend.matmul_batch_w(
                    &mut gate_up_batch,
                    &post_norm_batch,
                    w.as_ref(),
                    d,
                    hidden_dim,
                    gu,
                );
                // j. SwiGLU activation on the fused rows
                for i in 0..d {
                    let row = &gate_up_batch[i * gu..(i + 1) * gu];
                    self.tensor_backend.swiglu(
                        &mut act_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                        &row[..intermediate_dim],
                        &row[intermediate_dim..],
                    );
                }
            } else {
                self.tensor_backend.matmul_batch_w(
                    &mut gate_batch,
                    &post_norm_batch,
                    layer_w.gate_proj.as_ref(),
                    d,
                    hidden_dim,
                    intermediate_dim,
                );
                self.tensor_backend.matmul_batch_w(
                    &mut up_batch,
                    &post_norm_batch,
                    layer_w.up_proj.as_ref(),
                    d,
                    hidden_dim,
                    intermediate_dim,
                );
                // j. SwiGLU activation
                for i in 0..d {
                    self.tensor_backend.swiglu(
                        &mut act_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                        &gate_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                        &up_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                    );
                }
            }

            // k. MLP Down projection (M = D)
            self.tensor_backend.matmul_batch_w(
                &mut mlp_out_batch,
                &act_batch,
                layer_w.down_proj.as_ref(),
                d,
                intermediate_dim,
                hidden_dim,
            );

            // l. Residual connection
            for idx in 0..x.len() {
                x[idx] += mlp_out_batch[idx];
            }
        }

        // 6. Final RMSNorm
        for i in 0..d {
            self.tensor_backend.rmsnorm(
                &mut x_norm[i * hidden_dim..(i + 1) * hidden_dim],
                &x[i * hidden_dim..(i + 1) * hidden_dim],
                &self.weights.final_norm,
                eps,
            );
        }

        // 7. Batched Vocabulary Logits Projection (M = D)
        self.tensor_backend.matmul_batch_w(
            &mut logits_batch,
            &x_norm,
            self.weights.output_matrix(),
            d,
            hidden_dim,
            vocab_size,
        );

        // 8. Sampling and output emission (slot failures from step 1b first)
        let mut outputs = slot_failures;
        outputs.reserve(d);
        // The model's end-of-sequence set (legacy TinyLlama UNK/BOS/EOS when the config
        // names none): a Llama 3 vocabulary has ordinary text at ids 0..=2.
        let stop_tokens = self.weights.config.stop_token_ids().to_vec();

        for (i, &(seq_id, _, pos)) in valid_reqs.iter().enumerate() {
            let logits = &logits_batch[i * vocab_size..(i + 1) * vocab_size];
            // PREFILL-E2E C6: the request's own sampling params (recorded at
            // prefill or by the fork hook), seeded per request and position.
            // No params, or temperature 0: argmax as before.
            // sc#294: counted inside the branch taken.
            let obs = &mut self.decode_observed.entry(seq_id).or_default().decode;
            let (sampled_tok, logprob) = match self.sampling.get(&seq_id) {
                Some(params) => {
                    let seed = decode_sampling_seed(seq_id, pos + 1);
                    let picked = sample_with_params_observed(logits, params, seed, obs);
                    if obs.sampled_tokens > 0 {
                        obs.seed_request_id = Some(seq_id);
                    }
                    picked
                }
                None => {
                    obs.greedy_tokens += 1;
                    sample_argmax(logits)
                }
            };

            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.tokens.push(sampled_tok);

                if stop_tokens.contains(&sampled_tok) {
                    outputs.push(DecodeOutput::Finished {
                        request_id: seq_id,
                        reason: FinishReason::StopToken,
                        total_tokens: seq.tokens.len(),
                    });
                } else {
                    outputs.push(DecodeOutput::Token {
                        request_id: seq_id,
                        token_id: sampled_tok,
                        logprob: Some(logprob),
                    });
                }
            }
        }

        Ok((outputs, logits_batch, vocab_size))
    }
}

#[async_trait]
impl AienInferenceBackend for NativeTransformerBackend {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.weights = TransformerWeights::reference_test_weights(config);
        self.sequences.clear();
        self.sampling.clear();
        self.pending_prefill_token.clear();
        self.decode_observed.clear();
        Ok(())
    }

    fn manages_kv_cache(&self) -> bool {
        self.kv_manager.is_some()
    }

    /// Forks per-sequence backend state (tokens, hence position and last
    /// token) from a prefilled parent into a child whose physical KV the
    /// shared KV manager has already forked (`fork_prefilled`). Never copies
    /// or re-forks K/V. Refuses when:
    /// - there is no KV manager, or it has no physical tensor pool: the K/V
    ///   would live in the backend's dense per-sequence cache and a fork
    ///   would have to copy it per branch (bullet 10);
    /// - the parent is not PrefillReady/SharedFrozen in the KV manager
    ///   (bullet 6: no fork before the completion fence);
    /// - the child has no KV block table yet (the KV fork must come first);
    /// - the parent is unknown to the backend, or the child already exists.
    fn fork_sequence(&mut self, parent_id: u64, child_id: u64) -> Result<(), String> {
        let kv_mgr = self.kv_manager.as_ref().ok_or_else(|| {
            format!(
                "fork_sequence {} -> {} refused: backend has no shared KV manager \
                 (unpaged fork would copy dense K/V)",
                parent_id, child_id
            )
        })?;
        {
            let kv = kv_mgr.read();
            if kv.tensor_pool().is_none() {
                return Err(format!(
                    "fork_sequence {} -> {} refused: KV manager has no physical tensor pool",
                    parent_id, child_id
                ));
            }
            match kv.prefill_state(parent_id) {
                Some(state) if state.is_ready() => {}
                Some(state) => {
                    return Err(format!(
                        "fork_sequence {} -> {} refused: parent prefill state {:?} is not ready \
                         (completion fence not passed)",
                        parent_id, child_id, state
                    ))
                }
                None => {
                    return Err(format!(
                        "fork_sequence {} -> {} refused: parent has no KV block table",
                        parent_id, child_id
                    ))
                }
            }
            if kv.get_block_table(child_id).is_none() {
                return Err(format!(
                    "fork_sequence {} -> {} refused: child has no KV block table \
                     (physical KV fork must precede the backend fork)",
                    parent_id, child_id
                ));
            }
        }
        if self.sequences.contains_key(&child_id) {
            return Err(format!(
                "fork_sequence {} -> {} refused: child already has backend state",
                parent_id, child_id
            ));
        }
        // The token prefill sampled but did not append yet (`pending_prefill_token`)
        // belongs to the parent's stream. The child gets it appended to its own copy
        // of the tokens; the parent is NOT mutated (its tokens stay prompt-only and
        // its pending token stays pending, so the parent's own next decode commits it).
        let mut parent_tokens = self
            .sequences
            .get(&parent_id)
            .ok_or_else(|| {
                format!(
                    "fork_sequence {} -> {} refused: parent unknown to the backend",
                    parent_id, child_id
                )
            })?
            .tokens
            .clone();
        if let Some(&pending) = self.pending_prefill_token.get(&parent_id) {
            parent_tokens.push(pending);
        }
        let kv_dtype = self.storage_dtype();
        self.sequences.insert(
            child_id,
            SequenceState {
                tokens: parent_tokens,
                // Paged mode: K/V lives in the shared pool, never in per-sequence layers.
                layers: Vec::new(),
                kv_dtype,
            },
        );
        // PREFILL-E2E C6: a plain fork inherits the parent's sampling params
        // (seeded by the child's own id at decode, so streams still differ).
        if let Some(params) = self.sampling.get(&parent_id).cloned() {
            self.sampling.insert(child_id, params);
        }
        Ok(())
    }

    /// PREFILL-E2E C6: same checks and state fork as `fork_sequence`, then
    /// records the branch's own sampling params for decode.
    fn fork_sequence_with_sampling(
        &mut self,
        parent_id: u64,
        child_id: u64,
        sampling: &SamplingParams,
    ) -> Result<(), String> {
        self.fork_sequence(parent_id, child_id)?;
        self.sampling.insert(child_id, sampling.clone());
        Ok(())
    }

    /// ADR 0003 section 6.2 (review finding 3): the tokens whose K/V is written
    /// in the shared pool. VERIFIED rule, from the decode and prefill code above:
    /// prefill leaves `tokens` holding exactly the cached positions and the
    /// sampled token apart in `pending_prefill_token`, so `computed = tokens`;
    /// every decode step first commits the pending token (or feeds the last
    /// token), computes K/V for the LAST entry of `tokens`, then appends the
    /// newly sampled token, which has no K/V, so `computed = tokens[..len - 1]`.
    /// The answer is also capped at the block table's `total_tokens`, so it can
    /// never claim a position the table does not cover. `None` for an unknown
    /// sequence, without a shared manager and pool (K/V is then in the dense
    /// per-sequence cache, which no holder can pin), or without a block table.
    fn computed_prefix(&self, seq_id: u64) -> Option<Vec<u32>> {
        let mgr = self.kv_manager.as_ref()?;
        let table_tokens = {
            let kv = mgr.read();
            kv.tensor_pool()?;
            kv.total_tokens(seq_id)?
        };
        let seq = self.sequences.get(&seq_id)?;
        let computed = if self.pending_prefill_token.contains_key(&seq_id) {
            seq.tokens.len()
        } else {
            seq.tokens.len().checked_sub(1)?
        };
        Some(seq.tokens[..computed.min(table_tokens)].to_vec())
    }

    /// ADR 0003 section 6.3: backend state for a prefix whose K/V a restore put
    /// in `seq_id`'s block table. Same refusals as `fork_sequence` where they
    /// apply, nothing changed on any refusal:
    /// - no shared KV manager, or one with no physical tensor pool;
    /// - no block table for `seq_id` (the restored table is assembled first);
    /// - the table is not `PrefillPending` (DOCS SILENT: `fork_sequence` demands
    ///   a ready parent because it shares finished K/V; here the table is the
    ///   request's own and the suffix prefill must still run, so the narrowest
    ///   rule is the exact state `assemble_table` leaves, and a ready table is
    ///   refused because decode could start on it);
    /// - empty `tokens`, or a count different from the table's `total_tokens`;
    /// - `seq_id` already has backend state.
    ///
    /// On success `tokens` is the restored prefix, so the next prefill chunk
    /// sits at position `tokens.len()`. No pending sampled token exists.
    fn restore_sequence(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(), String> {
        let kv_mgr = self.kv_manager.as_ref().ok_or_else(|| {
            format!(
                "restore_sequence {seq_id} refused: backend has no shared KV manager \
                 (unpaged restore would copy dense K/V)"
            )
        })?;
        {
            let kv = kv_mgr.read();
            if kv.tensor_pool().is_none() {
                return Err(format!(
                    "restore_sequence {seq_id} refused: KV manager has no physical tensor pool"
                ));
            }
            let table = kv.get_block_table(seq_id).ok_or_else(|| {
                format!(
                    "restore_sequence {seq_id} refused: sequence has no KV block table \
                     (the restored table must be assembled first)"
                )
            })?;
            if table.prefill_state != aien_kv_cache::PrefillState::PrefillPending {
                return Err(format!(
                    "restore_sequence {seq_id} refused: prefill state {:?} is not PrefillPending",
                    table.prefill_state
                ));
            }
            if tokens.is_empty() {
                return Err(format!(
                    "restore_sequence {seq_id} refused: empty token list"
                ));
            }
            if tokens.len() != table.total_tokens {
                return Err(format!(
                    "restore_sequence {seq_id} refused: block table covers {} tokens, \
                     {} were given",
                    table.total_tokens,
                    tokens.len()
                ));
            }
        }
        if self.sequences.contains_key(&seq_id) {
            return Err(format!(
                "restore_sequence {seq_id} refused: sequence already has backend state"
            ));
        }
        let kv_dtype = self.storage_dtype();
        self.sequences.insert(
            seq_id,
            SequenceState {
                tokens: tokens.to_vec(),
                // Paged mode: K/V lives in the shared pool, never in per-sequence layers.
                layers: Vec::new(),
                kv_dtype,
            },
        );
        Ok(())
    }

    /// PREFILL-E2E C5 (bullet 12): removes this request's entry from the
    /// per-sequence map (tokens and, unpaged, dense K/V layers) and any pending
    /// prefill token (C3). The map is the
    /// backend's only per-request state. Unlike the inherent
    /// `NativeTransformerBackend::release_sequence` (standalone generation,
    /// which owns its KV), this never touches the KV manager: in the runtime
    /// the scheduler and swarm manager free the block tables, and freeing here
    /// too would race their accounting. Unknown ids are a no-op.
    fn release_sequence(&mut self, seq_id: u64) -> Result<(), String> {
        self.sequences.remove(&seq_id);
        self.pending_prefill_token.remove(&seq_id);
        // PREFILL-E2E C6: the per-request sampling params entry goes too.
        self.sampling.remove(&seq_id);
        self.decode_observed.remove(&seq_id);
        Ok(())
    }

    /// sc#294: how this request's tokens were chosen, counted inside the
    /// sampling branch taken (`sample_with_params_observed` and the prefill
    /// pick); removed once taken.
    fn take_decode_observation(&mut self, request_id: u64) -> Option<DecodeObservation> {
        self.decode_observed
            .remove(&request_id)
            .map(DecodeTally::merged)
    }

    /// sc#337: the tensor backend's name and op counters so far.
    fn op_evidence(&self) -> Option<crate::OpEvidence> {
        let mut e = self
            .tensor_backend
            .op_report()
            .evidence(self.tensor_backend.name());
        e.gpu_executions = self.tensor_backend.gpu_executions(); // sc#402
        Some(e)
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let t0 = std::time::Instant::now();
        let mut outputs = Vec::new();
        let mut prefill_tokens = 0;
        let num_layers = self.weights.config.num_layers;

        // 1. Prefill Requests
        for req in &batch.prefill_requests {
            prefill_tokens += req.prompt_tokens.len();
            // PREFILL-E2E C6: decode of this request uses its sampling params.
            self.sampling
                .insert(req.request_id, req.sampling_params.clone());

            if let Some(kv_mgr) = &self.kv_manager {
                let mut mgr = kv_mgr.write();
                if mgr.get_block_table(req.request_id).is_none() {
                    let _ = mgr.allocate_sequence(req.request_id, &req.prompt_tokens);
                }
            }

            let kv_dtype = self.storage_dtype();
            let seq = self
                .sequences
                .entry(req.request_id)
                .or_insert_with(|| SequenceState {
                    tokens: Vec::new(),
                    layers: vec![LayerKvCache::default(); num_layers],
                    kv_dtype,
                });

            let last_hidden = Self::prefill_prompt_layer_by_layer_paged(
                &self.weights,
                &*self.tensor_backend,
                &req.prompt_tokens,
                seq,
                req.request_id,
                self.kv_manager.as_ref(),
            )?;
            // A further prefill chunk: the sample held from the previous chunk was
            // mid-prompt and is discarded. Done only after this chunk succeeded, so a
            // chunk that fails loudly (KV pool exhausted) keeps the held sample and
            // can be retried against unchanged state (PREFILL-I25).
            self.pending_prefill_token.remove(&req.request_id);

            let logits =
                Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &last_hidden);
            // sc#294: this chunk's pick replaces the previous chunk's (only the
            // final chunk's pick is kept, like `pending_prefill_token`).
            let mut pick = DecodeObservation::default();
            let (sampled_tok, logprob) = if req.sampling_params.temperature <= 0.001 {
                pick.greedy_tokens = 1;
                sample_argmax(&logits)
            } else {
                pick.sampled_tokens = 1;
                pick.temperature = Some(req.sampling_params.temperature);
                pick.seed_request_id = Some(req.request_id);
                sample_temperature(&logits, req.sampling_params.temperature, req.request_id)
            };
            self.decode_observed
                .entry(req.request_id)
                .or_default()
                .prefill_pick = pick;

            // Not pushed to `seq.tokens`: if another chunk of this prompt follows, the
            // next prefill call replaces this sample; otherwise the first decode
            // commits it. Pushing here would put a sampled token between prompt chunks.
            self.pending_prefill_token
                .insert(req.request_id, sampled_tok);

            outputs.push(DecodeOutput::Token {
                request_id: req.request_id,
                token_id: sampled_tok,
                logprob: Some(logprob),
            });
        }

        // 2. Decode Requests (Batched across D active sequences)
        if !batch.decode_requests.is_empty() {
            let decode_outputs = self.forward_decode_batch(&batch.decode_requests)?;
            outputs.extend(decode_outputs);
        }

        let elapsed_us = t0.elapsed().as_micros() as u64;
        let metrics = StepMetrics {
            prefill_tokens_processed: prefill_tokens,
            decode_tokens_emitted: batch.decode_requests.len() + batch.prefill_requests.len(),
            step_latency_us: elapsed_us,
            active_kv_blocks: batch.block_tables.values().map(|v| v.len()).sum(),
        };

        Ok((outputs, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_constructors() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            ..Default::default()
        };

        let cpu_backend = NativeTransformerBackend::with_reference_weights(&config);
        assert_eq!(cpu_backend.tensor_backend.name(), "ReferenceCpuBackend");

        let blackwell_backend = NativeTransformerBackend::with_blackwell_backend(&config);
        assert!(blackwell_backend
            .tensor_backend
            .name()
            .starts_with("OmegaGb10Backend"));
    }

    #[test]
    fn test_forward_token_and_logits_execution() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            ..Default::default()
        };

        let backend = NativeTransformerBackend::with_reference_weights(&config);
        let mut seq_state = SequenceState {
            tokens: vec![1],
            layers: vec![LayerKvCache::default(); config.num_layers],
            kv_dtype: KvDType::Fp32,
        };

        let hidden = backend.forward_token(1, 0, &mut seq_state);
        assert_eq!(hidden.len(), config.hidden_dim());

        let logits = backend.compute_logits(&hidden);
        assert_eq!(logits.len(), config.vocab_size());
    }

    #[test]
    fn test_paged_cow_context_and_branch_fork() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };

        let weights = TransformerWeights::reference_test_weights(&config);
        let mut backend = NativeTransformerBackend::with_paged_kv(weights, 200, 16).unwrap();

        // 1. Create parent context with 30 prompt tokens (2 blocks)
        let prompt: Vec<u32> = (0..30).collect();
        let parent = backend.create_context(&prompt).unwrap();

        // 2. Fork 4 branches
        let b1 = backend.fork_context(parent).unwrap();
        let b2 = backend.fork_context(parent).unwrap();
        let b3 = backend.fork_context(parent).unwrap();
        let b4 = backend.fork_context(parent).unwrap();

        // Initial receipt before decode: 2 shared pages
        let r_init = backend.get_usage_receipt(b1).unwrap();
        assert_eq!(r_init.shared_pages, 2);
        assert_eq!(r_init.private_pages, 0);

        // 3. Decode step for each branch -> triggers COW divergence on partial block 1
        let (t1, _) = backend.decode_branch_step(b1).unwrap();
        let (t2, _) = backend.decode_branch_step(b2).unwrap();
        let (t3, _) = backend.decode_branch_step(b3).unwrap();
        let (t4, _) = backend.decode_branch_step(b4).unwrap();

        // Tokens generated
        assert_eq!(t1, t2); // Same deterministic greedy choice initially
        assert_eq!(t3, t4);

        let r_after = backend.get_usage_receipt(b1).unwrap();
        assert_eq!(r_after.cow_faults, 4); // All 4 branches COWed tail page
        assert_eq!(r_after.shared_pages, 1); // Prefix block 0 remains shared
        assert_eq!(r_after.private_pages, 1); // Divergent block 1 is private

        // 4. Release branches
        backend.release_branch(b1).unwrap();
        backend.release_branch(b2).unwrap();
        backend.release_branch(b3).unwrap();
        backend.release_branch(b4).unwrap();
    }

    #[test]
    fn test_native_transformer_generate_tokens() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };
        let mut backend = NativeTransformerBackend::with_reference_weights(&config);
        let prompt = vec![1, 5, 9];
        let stop_tokens = vec![0];
        let generated = backend
            .generate_tokens(555, &prompt, 4, 0.0, &stop_tokens)
            .unwrap();
        assert_eq!(generated.len(), 4);
        assert!(!generated.contains(&0));
        assert_eq!(backend.sequences.len(), 0);
    }

    #[test]
    fn test_native_transformer_generate_tokens_streaming() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };
        let mut backend = NativeTransformerBackend::with_reference_weights(&config);
        let prompt = vec![1, 5, 9];
        let stop_tokens = vec![0];
        let mut streamed = Vec::new();
        backend
            .generate_tokens_streaming(777, &prompt, 4, 0.0, &stop_tokens, |tok| {
                streamed.push(tok);
                true
            })
            .unwrap();
        assert_eq!(streamed.len(), 4);
        assert!(!streamed.contains(&0));
        assert_eq!(backend.sequences.len(), 0);
    }

    // PREFILL-E2E-0 bullet 4, cut C3: a prompt longer than one prefill chunk must
    // give the same result as a one-shot prefill. Chunk 128 matches the daemon's
    // scheduler chunk size; 300 tokens = 128 + 128 + 44.
    const C3_PROMPT_LEN: usize = 300;
    const C3_CHUNK: usize = 128;
    const C3_ABS_TOL: f32 = 1e-4;

    fn c3_config() -> ModelConfig {
        ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        }
    }

    /// Deterministic prompt avoiding token ids 0..=2 (UNK/BOS/EOS).
    fn c3_prompt() -> Vec<u32> {
        (0..C3_PROMPT_LEN)
            .map(|i| ((i * 37 + 11) % 253 + 3) as u32)
            .collect()
    }

    fn c3_backend(paged: bool) -> NativeTransformerBackend {
        let weights = TransformerWeights::reference_test_weights(&c3_config());
        if paged {
            NativeTransformerBackend::with_paged_kv(weights, 64, 16).unwrap()
        } else {
            NativeTransformerBackend::new(weights)
        }
    }

    fn c3_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Last-position logits of a 300-token prompt prefilled in chunks of 128 equal
    /// the one-shot prefill within abs 1e-4, and the sequence holds exactly the
    /// prompt after every chunk. In paged mode the first chunk allocates only its
    /// own blocks, so later chunks exercise the append path and the pool rebuild.
    fn c3_logits_parity(paged: bool) {
        let prompt = c3_prompt();
        let id = 7u64;

        let mut one_shot = c3_backend(paged);
        let reference = one_shot.prefill_sequence(id, &prompt).unwrap();

        let mut chunked = c3_backend(paged);
        let mut fed = 0usize;
        let mut last = Vec::new();
        for chunk in prompt.chunks(C3_CHUNK) {
            last = chunked.prefill_sequence(id, chunk).unwrap();
            fed += chunk.len();
            assert_eq!(
                chunked.sequences[&id].tokens,
                prompt[..fed],
                "sequence must hold exactly the prompt tokens fed so far"
            );
        }
        assert_eq!(fed, C3_PROMPT_LEN);

        let diff = c3_max_abs_diff(&reference, &last);
        assert!(
            diff <= C3_ABS_TOL,
            "chunked vs one-shot last-position logits differ by {} (> {}), paged={}",
            diff,
            C3_ABS_TOL,
            paged
        );

        if paged {
            let kv = chunked.kv_manager.as_ref().unwrap().read();
            assert_eq!(kv.total_tokens(id), Some(C3_PROMPT_LEN));
        } else {
            for layer in &chunked.sequences[&id].layers {
                assert_eq!(layer.cached_k.len(), C3_PROMPT_LEN);
                assert_eq!(
                    layer.flat_k.len(),
                    one_shot.sequences[&id].layers[0].flat_k.len()
                );
            }
        }
    }

    #[test]
    fn c3_chunked_prefill_logits_match_one_shot_unpaged() {
        c3_logits_parity(false);
    }

    #[test]
    fn c3_chunked_prefill_logits_match_one_shot_paged() {
        c3_logits_parity(true);
    }

    /// Through `execute_step` (the scheduler path): no sampled token is pushed
    /// between prompt chunks, the sequence holds exactly prompt_len tokens before
    /// the first decode, and prefill + one decode match the one-shot control.
    /// Paged mode pre-allocates the whole prompt, as `prefill_detached` does.
    async fn c3_execute_step_chunked(paged: bool) {
        use crate::{SamplingParams, SequenceRequest};
        let prompt = c3_prompt();
        let id = 9u64;
        let req = |toks: &[u32]| SequenceRequest {
            request_id: id,
            prompt_tokens: toks.to_vec(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            arrival_time_ns: 0,
            priority: 0,
        };
        let batch = |prefill: Vec<SequenceRequest>, decode: Vec<u64>| ScheduledBatch {
            prefill_requests: prefill,
            decode_requests: decode,
            block_tables: HashMap::new(),
            step_id: 0,
        };

        let mut one_shot = c3_backend(paged);
        if let Some(kv) = &one_shot.kv_manager {
            kv.write().allocate_sequence(id, &prompt).unwrap();
        }
        let (out_one, _) = one_shot
            .execute_step(&batch(vec![req(&prompt)], vec![]))
            .await
            .unwrap();
        one_shot
            .execute_step(&batch(vec![], vec![id]))
            .await
            .unwrap();

        let mut chunked = c3_backend(paged);
        if let Some(kv) = &chunked.kv_manager {
            kv.write().allocate_sequence(id, &prompt).unwrap();
        }
        let mut fed = 0usize;
        let mut last_out = Vec::new();
        for chunk in prompt.chunks(C3_CHUNK) {
            let (out, metrics) = chunked
                .execute_step(&batch(vec![req(chunk)], vec![]))
                .await
                .unwrap();
            fed += chunk.len();
            assert_eq!(metrics.prefill_tokens_processed, chunk.len());
            assert_eq!(
                chunked.sequences[&id].tokens,
                prompt[..fed],
                "no sampled token may be pushed between prompt chunks"
            );
            last_out = out;
        }
        assert_eq!(chunked.sequences[&id].tokens.len(), C3_PROMPT_LEN);

        let first_token = |out: &[DecodeOutput]| match out.first() {
            Some(DecodeOutput::Token { token_id, .. }) => *token_id,
            other => panic!("expected a sampled token after prefill, got {:?}", other),
        };
        let first_chunked = first_token(&last_out[..]);
        assert_eq!(first_chunked, first_token(&out_one[..]));

        chunked
            .execute_step(&batch(vec![], vec![id]))
            .await
            .unwrap();
        let toks = &chunked.sequences[&id].tokens;
        assert_eq!(toks.len(), C3_PROMPT_LEN + 2);
        assert_eq!(toks[..C3_PROMPT_LEN], prompt[..]);
        assert_eq!(toks[C3_PROMPT_LEN], first_chunked);
        assert_eq!(*toks, one_shot.sequences[&id].tokens);
    }

    #[tokio::test]
    async fn c3_chunked_prefill_execute_step_unpaged() {
        c3_execute_step_chunked(false).await;
    }

    #[tokio::test]
    async fn c3_chunked_prefill_execute_step_paged() {
        c3_execute_step_chunked(true).await;
    }

    /// sc#294: prefill in chunks, then `decodes` decode steps, through
    /// `execute_step` (the scheduler path). Returns the observation the
    /// scheduler would take at finish and the chosen tokens after the prompt.
    async fn sc294_run(params: SamplingParams, decodes: usize) -> (DecodeObservation, Vec<u32>) {
        use crate::SequenceRequest;
        let prompt = c3_prompt();
        let id = 11u64;
        let batch = |prefill: Vec<SequenceRequest>, decode: Vec<u64>| ScheduledBatch {
            prefill_requests: prefill,
            decode_requests: decode,
            block_tables: HashMap::new(),
            step_id: 0,
        };
        let mut b = c3_backend(false);
        for chunk in prompt.chunks(C3_CHUNK) {
            let req = SequenceRequest {
                request_id: id,
                prompt_tokens: chunk.to_vec(),
                sampling_params: params.clone(),
                arrival_time_ns: 0,
                priority: 0,
            };
            b.execute_step(&batch(vec![req], vec![])).await.unwrap();
        }
        for _ in 0..decodes {
            b.execute_step(&batch(vec![], vec![id])).await.unwrap();
        }
        let chosen = b.sequences[&id].tokens[C3_PROMPT_LEN..].to_vec();
        let obs = b.take_decode_observation(id).expect("observed");
        assert_eq!(b.take_decode_observation(id), None, "taken once");
        (obs, chosen)
    }

    /// Greedy request: every chosen token is counted as greedy, the
    /// discarded mid-prompt picks are not, and nothing claims sampling.
    #[tokio::test]
    async fn sc294_greedy_request_is_observed_greedy() {
        const {
            assert!(
                C3_PROMPT_LEN > C3_CHUNK,
                "prompt must take more than one chunk"
            )
        };
        let greedy = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let (obs, chosen) = sc294_run(greedy, 3).await;
        // 1 pick after the final prefill chunk (committed by the first decode)
        // plus 3 decode picks; the last decode's pick is pushed too.
        assert_eq!(chosen.len(), 4);
        assert_eq!(obs.greedy_tokens, 4);
        assert_eq!(obs.sampled_tokens, 0);
        assert_eq!(obs.mode(), "greedy");
        assert_eq!(
            (obs.temperature, obs.top_p, obs.seed_request_id),
            (None, None, None)
        );
    }

    /// Sampling request: every token is counted as drawn, with the
    /// temperature, the top_p the decode draws applied, and the seed's request id.
    #[tokio::test]
    async fn sc294_sampled_request_is_observed_sampled() {
        let sampled = SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            ..Default::default()
        };
        let (obs, _) = sc294_run(sampled, 3).await;
        assert_eq!(obs.sampled_tokens, 4);
        assert_eq!(obs.greedy_tokens, 0);
        assert_eq!(obs.mode(), "sampled");
        assert_eq!(obs.temperature, Some(0.7));
        assert_eq!(obs.top_p, Some(0.95));
        assert_eq!(obs.seed_request_id, Some(11));
    }

    /// Observing never changes the chosen token: the same token and logprob
    /// as `sample_with_params`, greedy and sampled, over many seeds.
    #[test]
    fn sc294_observing_does_not_change_the_choice() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32) * 0.1).collect();
        for temperature in [0.0f32, 0.0005, 0.3, 0.7, 1.5] {
            for top_p in [0.0f32, 0.5, 0.95, 1.0] {
                let p = SamplingParams {
                    temperature,
                    top_p,
                    ..Default::default()
                };
                for seed in 0..50u64 {
                    let seed = decode_sampling_seed(seed, 3);
                    let mut o = DecodeObservation::default();
                    assert_eq!(
                        sample_with_params(&logits, &p, seed),
                        sample_with_params_observed(&logits, &p, seed, &mut o)
                    );
                    let (t, lp, _) = sample_with_params_traced(&logits, &p, seed, &mut o);
                    assert_eq!(sample_with_params(&logits, &p, seed), (t, lp));
                }
            }
        }
    }

    /// OWN-8B stage S: the kept set is counted where the draw happens.
    #[test]
    fn stage_s_trace_counts_the_kept_set() {
        let mut o = DecodeObservation::default();
        let p = |temperature: f32, top_p: f32| SamplingParams {
            temperature,
            top_p,
            ..Default::default()
        };
        // Flat 4-way: everything is kept, top probability 1/4.
        let flat = [0.0f32; 4];
        let (_, _, tr) = sample_with_params_traced(&flat, &p(1.0, 1.0), 9, &mut o);
        assert_eq!(tr.kept, 4);
        assert!((tr.top_prob - 0.25).abs() < 1e-12, "{tr:?}");
        // One token holds more than top_p of the mass: the nucleus keeps it alone.
        let peaked = [10.0f32, 0.0, 0.0, 0.0];
        let (t, _, tr) = sample_with_params_traced(&peaked, &p(0.7, 0.8), 9, &mut o);
        assert_eq!((t, tr.kept), (0, 1));
        assert!(tr.top_prob > 0.8, "{tr:?}");
        // Masked (-inf) logits are not counted as kept.
        let masked = [1.0f32, 1.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let (_, _, tr) = sample_with_params_traced(&masked, &p(1.0, 1.0), 9, &mut o);
        assert_eq!(tr.kept, 2);
        // Greedy: one candidate.
        let (t, _, tr) = sample_with_params_traced(&flat, &p(0.0, 1.0), 9, &mut o);
        assert_eq!((t, tr.kept, tr.top_prob), (0, 1, 1.0));
    }

    /// A decoding request with no recorded sampling params takes the argmax
    /// arm of decode; those picks are counted as greedy too.
    #[tokio::test]
    async fn sc294_decode_without_params_is_observed_greedy() {
        use crate::SequenceRequest;
        let id = 7u64;
        let mut b = c3_backend(false);
        let req = SequenceRequest {
            request_id: id,
            prompt_tokens: c3_prompt()[..4].to_vec(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            arrival_time_ns: 0,
            priority: 0,
        };
        let batch = |prefill: Vec<SequenceRequest>, decode: Vec<u64>| ScheduledBatch {
            prefill_requests: prefill,
            decode_requests: decode,
            block_tables: HashMap::new(),
            step_id: 0,
        };
        b.execute_step(&batch(vec![req], vec![])).await.unwrap();
        b.sampling.remove(&id);
        for _ in 0..2 {
            b.execute_step(&batch(vec![], vec![id])).await.unwrap();
        }
        let obs = b.take_decode_observation(id).expect("observed");
        assert_eq!((obs.greedy_tokens, obs.sampled_tokens), (3, 0));
        assert_eq!(obs.mode(), "greedy");
    }

    /// A released or never-seen request has no observation.
    #[tokio::test]
    async fn sc294_release_drops_the_observation() {
        use crate::SequenceRequest;
        let mut b = c3_backend(false);
        let req = SequenceRequest {
            request_id: 5,
            prompt_tokens: c3_prompt()[..4].to_vec(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            arrival_time_ns: 0,
            priority: 0,
        };
        let batch = ScheduledBatch {
            prefill_requests: vec![req],
            decode_requests: vec![],
            block_tables: HashMap::new(),
            step_id: 0,
        };
        b.execute_step(&batch).await.unwrap();
        AienInferenceBackend::release_sequence(&mut b, 5).unwrap();
        assert_eq!(b.take_decode_observation(5), None);
        assert_eq!(b.take_decode_observation(6), None);
    }
}
