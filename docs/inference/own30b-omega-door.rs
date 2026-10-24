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

//! Omega engine behind H's `Engine` trait (OWN-30B cut 3 x cut 4 seam).
//! Drop-in for `crates/aien-serve-http/src/omega_engine.rs`; in that crate replace
//! `aien_omega_abi` with `aien_inference_abi` and add `--engine omega` in main.rs:
//!     "omega" => Box::new(OmegaDoor::load(&args.model_dir, args.max_context)?)
use std::path::Path;

use aien_omega_abi::qwen3_moe_omega::{self as omega, OmegaEngine};
use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};

pub struct OmegaDoor(OmegaEngine);

impl OmegaDoor {
    /// Loads the checkpoint onto Omega (about 61 GB resident plus the KV pool). Blocks for the
    /// load; `max_context` is the KV capacity in tokens (rounded up to 16-token blocks).
    pub fn load(model_dir: &Path, max_context: usize) -> Result<Self, String> {
        let engine = OmegaEngine::load(model_dir, max_context, &mut |what| eprintln!("omega load: {what}"))?;
        let r = &engine.load_report;
        eprintln!(
            "omega engine ready: {:.1}s, resident {:.1} GB, host {:.1} GB, window_api={}",
            r.seconds,
            r.resident_bytes as f64 / 1e9,
            r.host_bytes as f64 / 1e9,
            r.window_api
        );
        Ok(Self(engine))
    }
}

impl Engine for OmegaDoor {
    fn kind(&self) -> &str {
        self.0.kind()
    }

    fn max_context_tokens(&self) -> usize {
        self.0.max_context_tokens()
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        let max = self.0.max_context_tokens();
        if prompt.is_empty() || prompt.len() >= max {
            return Err(EngineError::PromptTooLong { len: prompt.len(), max });
        }
        let q = omega::GenerateParams {
            max_new_tokens: p.max_new_tokens,
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
            stop_token_ids: p.stop_token_ids.clone(),
        };
        let o = self.0.generate(prompt, &q, on_token).map_err(EngineError::Failed)?;
        Ok(GenerateOutcome {
            emitted_tokens: o.emitted_tokens,
            finish: match o.finish {
                omega::FinishReason::Stop => FinishReason::Stop,
                omega::FinishReason::Length => FinishReason::Length,
                omega::FinishReason::Cancelled => FinishReason::Cancelled,
            },
            prefill_ms: o.prefill_ms,
            decode_ms: o.decode_ms,
        })
    }
}
