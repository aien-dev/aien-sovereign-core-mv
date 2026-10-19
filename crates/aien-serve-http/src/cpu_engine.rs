//! The real-but-slow engine: the workspace's CPU reference forward for
//! Qwen3-30B-A3B (`aien_inference_abi::qwen3_coder`), exact FP8 expert
//! dequant, f32 attention. It makes the server end-to-end usable before the
//! Omega (own GPU kernels) engine lands. Expect seconds per token.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen3_coder::{
    compute_logits, forward_token, load_qwen3_coder, Qwen3CoderState, Qwen3CoderWeights,
};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::sampling::SamplingObserver;

pub struct CpuReferenceEngine {
    weights: Qwen3CoderWeights,
    max_context: usize,
}

impl CpuReferenceEngine {
    /// Loads the whole checkpoint (31 GB of FP8 plus f32 attention copies; several
    /// minutes and roughly 40 GB of host memory).
    pub fn load(checkpoint_dir: &Path, max_context: usize) -> Result<Self, String> {
        let weights = load_qwen3_coder(checkpoint_dir)
            .map_err(|e| format!("load {}: {e}", checkpoint_dir.display()))?;
        Ok(Self {
            weights,
            max_context,
        })
    }
}

impl Engine for CpuReferenceEngine {
    fn kind(&self) -> &str {
        "cpu-reference"
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn generate(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        if prompt.is_empty() {
            return Err(EngineError::Failed("empty prompt".into()));
        }
        if prompt.len() >= self.max_context {
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max: self.max_context,
            });
        }
        let mut state = Qwen3CoderState::new();
        let t0 = Instant::now();
        let mut hidden = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            hidden = forward_token(&self.weights, tok, pos, &mut state);
        }
        let prefill_ms = t0.elapsed().as_millis() as u64;

        let t1 = Instant::now();
        let mut sampler = SamplingObserver::new(params);
        let mut pos = prompt.len();
        let mut emitted = 0usize;
        let finish = loop {
            if emitted >= params.max_new_tokens || pos >= self.max_context {
                break FinishReason::Length;
            }
            let logits = compute_logits(&self.weights, &hidden);
            let next = sampler.sample(&logits, emitted);
            if params.stop_token_ids.contains(&next) {
                break FinishReason::Stop;
            }
            emitted += 1;
            if !on_token(next) {
                break FinishReason::Cancelled;
            }
            if emitted >= params.max_new_tokens || pos >= self.max_context {
                break FinishReason::Length;
            }
            hidden = forward_token(&self.weights, next, pos, &mut state);
            pos += 1;
        };
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms: t1.elapsed().as_millis() as u64,
            sampling: Some(sampler.record()),
            mask: None,
        })
    }
}
