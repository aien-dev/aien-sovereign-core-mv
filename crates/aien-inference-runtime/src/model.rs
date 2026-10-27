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

use aien_inference_abi::backend::{ReferenceCpuBackend, TensorBackend};
use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::weights::TransformerWeights;
use aien_inference_abi::ModelConfig;
use aien_kv_cache::SharedKvManager;
use aien_scheduler::AienScheduler;
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_sequence_id() -> u64 {
    SEQUENCE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub struct EmbeddedModel {
    pub tokenizer: TinyLlamaTokenizer,
    pub transformer: NativeTransformerBackend,
    pub scheduler: Option<Arc<RwLock<AienScheduler>>>,
    pub kv_cache: Option<SharedKvManager>,
    pub config: ModelConfig,
}

impl EmbeddedModel {
    pub fn new(
        tokenizer: TinyLlamaTokenizer,
        transformer: NativeTransformerBackend,
        scheduler: Option<Arc<RwLock<AienScheduler>>>,
        kv_cache: Option<SharedKvManager>,
        config: ModelConfig,
    ) -> Self {
        Self {
            tokenizer,
            transformer,
            scheduler,
            kv_cache,
            config,
        }
    }

    pub fn with_reference_weights(config: &ModelConfig) -> Result<Self, String> {
        let transformer = NativeTransformerBackend::with_reference_weights(config);
        let tokenizer_path = find_tokenizer_path()
            .ok_or_else(|| "No valid tokenizer.json found on filesystem".to_string())?;

        let tokenizer = TinyLlamaTokenizer::from_file(&tokenizer_path).map_err(|e| {
            format!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?;

        Ok(Self {
            tokenizer,
            transformer,
            scheduler: None,
            kv_cache: None,
            config: config.clone(),
        })
    }

    pub fn load_checkpoint<P: AsRef<Path>>(
        checkpoint_path: P,
        tokenizer_path: P,
        use_gpu: bool,
        paged_kv: bool,
    ) -> Result<Self, String> {
        let config = ModelConfig::tinyllama_1_1b();
        let weights = TransformerWeights::load_from_safetensors(checkpoint_path.as_ref(), &config)
            .map_err(|e| format!("Failed to load safetensors checkpoint: {}", e))?;

        // FB-1 cut 6: the native Omega engine is the only GPU backend (no CUDA).
        let tensor_backend: Arc<dyn TensorBackend> = if use_gpu {
            let omega = aien_inference_abi::OmegaGb10Backend::new();
            if omega.is_available() {
                info!("Binding EmbeddedModel to the native Omega GPU engine");
                Arc::new(omega)
            } else if aien_inference_abi::strict::production_strict() {
                return Err(aien_inference_abi::strict::violation(
                    "GPU backend requested but the Omega GPU engine is not linked (stub build)",
                ));
            } else {
                info!("Omega GPU engine not linked, falling back to CPU reference execution");
                Arc::new(ReferenceCpuBackend::new())
            }
        } else {
            Arc::new(ReferenceCpuBackend::new())
        };

        let transformer = if paged_kv {
            NativeTransformerBackend::with_paged_kv_backend(
                weights,
                tensor_backend,
                2048,
                config.block_size,
            )?
        } else {
            NativeTransformerBackend::with_backend(weights, tensor_backend)
        };

        let kv_cache = transformer.kv_manager.clone();
        let tokenizer = TinyLlamaTokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
            format!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.as_ref().display(),
                e
            )
        })?;

        Ok(Self {
            tokenizer,
            transformer,
            scheduler: None,
            kv_cache,
            config,
        })
    }

    pub fn load_default_or_fallback() -> Result<Self, String> {
        let model_path = find_checkpoint_path();
        let tokenizer_path = find_tokenizer_path();

        if let (Some(mp), Some(tp)) = (model_path, tokenizer_path) {
            info!("Loading real TinyLlama checkpoint from {}", mp.display());
            Self::load_checkpoint(&mp, &tp, true, true)
        } else {
            if aien_inference_abi::strict::production_strict() {
                return Err(aien_inference_abi::strict::violation(
                    "no real checkpoint and tokenizer found on disk",
                ));
            }
            warn!("Real model checkpoint not found on disk, using reference test weights");
            Self::with_reference_weights(&ModelConfig::tinyllama_1_1b())
        }
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        mut on_token: F,
    ) -> Result<(), String>
    where
        F: FnMut(&str) -> bool,
    {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| format!("Encoding failed: {}", e))?;

        let seq_id = generate_sequence_id();
        let stop_tokens = [
            TinyLlamaTokenizer::EOS_TOKEN_ID,
            TinyLlamaTokenizer::UNK_TOKEN_ID,
        ];

        self.transformer.generate_tokens_streaming(
            seq_id,
            &prompt_tokens,
            max_tokens,
            temperature,
            &stop_tokens,
            |tok| {
                if let Ok(piece) = self.tokenizer.decode(&[tok]) {
                    on_token(&piece)
                } else {
                    true
                }
            },
        )
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, String> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| format!("Encoding failed: {}", e))?;

        let seq_id = generate_sequence_id();
        let stop_tokens = [
            TinyLlamaTokenizer::EOS_TOKEN_ID,
            TinyLlamaTokenizer::UNK_TOKEN_ID,
        ];

        let generated_ids = self.transformer.generate_tokens(
            seq_id,
            &prompt_tokens,
            max_tokens,
            temperature,
            &stop_tokens,
        )?;

        self.tokenizer
            .decode(&generated_ids)
            .map_err(|e| format!("Decoding failed: {}", e))
    }
}

/// Pinned TinyLlama snapshot in the Hugging Face cache, located from
/// HF_HOME or HOME. Returns None when neither is set.
fn tinyllama_snapshot_dir() -> Option<PathBuf> {
    let hub = match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub"),
    };
    Some(hub.join("models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"))
}

fn find_checkpoint_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TINYLLAMA_MODEL_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    if let Some(base) = tinyllama_snapshot_dir() {
        let default_snap = base.join("model.safetensors");
        if default_snap.exists() {
            return Some(default_snap);
        }
    }

    None
}

fn find_tokenizer_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TINYLLAMA_TOKENIZER_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../aien-inference-abi/fixtures/tokenizer.json"),
        manifest_dir.join("fixtures/tokenizer.json"),
        PathBuf::from("crates/aien-inference-abi/fixtures/tokenizer.json"),
    ];
    let hf_tok = tinyllama_snapshot_dir().map(|d| d.join("tokenizer.json"));

    for c in candidates.iter().chain(hf_tok.iter()) {
        if c.exists() {
            return Some(c.clone());
        }
    }

    None
}
