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

//! Native weight data structures and zero-dependency loaders for transformer models.

use crate::resident::{MatrixRef, MatrixWeight};
use crate::tensor::{
    apply_rope_params, matmul_vec, rmsnorm, rmsnorm_heads_in_place,
    scaled_dot_product_attention_single, swiglu,
};
use crate::ModelConfig;
use aien_kv_cache::KvDType;
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    pub cached_k: Vec<Vec<f32>>,
    pub cached_v: Vec<Vec<f32>>,
    pub flat_k: Vec<f32>,
    pub flat_v: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SequenceState {
    pub tokens: Vec<u32>,
    pub layers: Vec<LayerKvCache>,
    /// Element type K/V are rounded to the moment they are produced, before any
    /// reader (dense cache, pool write, attention) sees them. The pool's dtype when
    /// a physical pool is attached, else the backend's `kv_dtype` (default `Fp32`).
    pub kv_dtype: KvDType,
}

#[derive(Debug, Clone)]
pub struct TransformerLayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_proj: MatrixWeight,
    pub k_proj: MatrixWeight,
    pub v_proj: MatrixWeight,
    pub o_proj: MatrixWeight,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: MatrixWeight,
    pub up_proj: MatrixWeight,
    pub down_proj: MatrixWeight,
    /// Resident loader only (sc#384 decode speed): q, k and v rows in one device matrix
    /// `[q_dim + 2*kv_dim, hidden]`. The forward paths use it when present (one launch
    /// instead of three), else the three matrices above. Same kernel, same arithmetic per row.
    pub qkv_proj: Option<MatrixWeight>,
    /// Resident loader only: gate rows then up rows, `[2*inter, hidden]`, one launch.
    pub gate_up_proj: Option<MatrixWeight>,
    /// Qwen3 per-head RMSNorm weights `[head_dim]` for Q and K (`config.qk_norm`); `None` for
    /// Llama-architecture models.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
}

impl TransformerLayerWeights {
    /// Qwen3: RMSNorm each query head and each key head over `head_dim`, in place, right
    /// after the projections and before RoPE (Hugging Face `modeling_qwen3.py`
    /// `Qwen3Attention.forward`: `q_norm(q_proj(x).view(.., head_dim))`, then rotary).
    /// A no-op for models without q_norm/k_norm.
    pub fn apply_qk_norm(&self, q: &mut [f32], k: &mut [f32], head_dim: usize, eps: f32) {
        if let Some(w) = &self.q_norm {
            rmsnorm_heads_in_place(q, w, head_dim, eps);
        }
        if let Some(w) = &self.k_norm {
            rmsnorm_heads_in_place(k, w, head_dim, eps);
        }
    }

    /// [`Self::apply_qk_norm`] through `backend` (`TensorBackend::rmsnorm_heads`), so a device
    /// backend runs it and accounts for it like any other op. A no-op without q_norm/k_norm.
    pub fn apply_qk_norm_on(
        &self,
        backend: &dyn crate::backend::TensorBackend,
        q: &mut [f32],
        k: &mut [f32],
        head_dim: usize,
        eps: f32,
    ) {
        if let Some(w) = &self.q_norm {
            backend.rmsnorm_heads(q, w, head_dim, eps);
        }
        if let Some(w) = &self.k_norm {
            backend.rmsnorm_heads(k, w, head_dim, eps);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub config: ModelConfig,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<TransformerLayerWeights>,
    pub final_norm: Vec<f32>,
    /// Output projection `[vocab, hidden]`. `None` when `config.tie_word_embeddings` on the host
    /// path: the projection is `embed_tokens` itself (no second copy). On the GB10 resident path
    /// it is `Some(Resident)` even for tied models (the embedding's own device copy). Read it
    /// through [`TransformerWeights::output_matrix`].
    pub lm_head: Option<MatrixWeight>,
}

/// Diagnostic capture of intermediate activations and shapes across all transformer layers.
#[derive(Debug, Clone, Default)]
pub struct ForwardDiagnostics {
    pub activations: std::collections::HashMap<String, Vec<f32>>,
    pub shapes: std::collections::HashMap<String, Vec<usize>>,
}

impl TransformerWeights {
    /// The `[vocab, hidden]` matrix that projects the final hidden state to logits:
    /// `lm_head`, or `embed_tokens` for a model with tied word embeddings.
    #[inline]
    pub fn output_projection(&self) -> &[f32] {
        self.output_matrix().host_or_refuse("output_projection")
    }

    /// [`Self::output_projection`] as host values or a device handle; what the backends take.
    #[inline]
    pub fn output_matrix(&self) -> MatrixRef<'_> {
        match &self.lm_head {
            Some(lm_head) => lm_head.as_ref(),
            None => MatrixRef::Host(&self.embed_tokens),
        }
    }

    /// Copies the embedding vector for a given token ID into a destination slice.
    #[inline]
    pub fn embed_token(&self, token_id: u32, out: &mut [f32]) {
        let hidden_dim = self.config.hidden_dim();
        let token_idx = (token_id as usize) % self.config.vocab_size();
        let start = token_idx * hidden_dim;
        let end = start + hidden_dim;
        out.copy_from_slice(&self.embed_tokens[start..end]);
    }

    /// Returns a direct slice reference to the token embedding.
    #[inline]
    pub fn token_embedding(&self, token_id: u32) -> &[f32] {
        let hidden_dim = self.config.hidden_dim();
        let token_idx = (token_id as usize) % self.config.vocab_size();
        let start = token_idx * hidden_dim;
        let end = start + hidden_dim;
        &self.embed_tokens[start..end]
    }

    /// Creates deterministically initialized weights for tests and headless verification.
    pub fn reference_test_weights(config: &ModelConfig) -> Self {
        let hidden_dim = config.hidden_dim();
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let intermediate_dim = config.intermediate_dim();
        let vocab_size = config.vocab_size();

        let mut lcg_state: u64 = 42;
        let mut next_float = || -> f32 {
            lcg_state = lcg_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((lcg_state >> 33) as f32) / ((1u64 << 31) as f32);
            (val - 0.5) * 0.05
        };

        let mut embed_tokens = vec![0.0f32; vocab_size * hidden_dim];
        for v in embed_tokens.iter_mut() {
            *v = next_float();
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let mut input_layernorm = vec![1.0f32; hidden_dim];
            for v in input_layernorm.iter_mut() {
                *v = 1.0 + next_float() * 0.1;
            }

            let mut q_proj = vec![0.0f32; hidden_dim * q_dim];
            for v in q_proj.iter_mut() {
                *v = next_float();
            }

            let mut k_proj = vec![0.0f32; hidden_dim * kv_dim];
            for v in k_proj.iter_mut() {
                *v = next_float();
            }

            let mut v_proj = vec![0.0f32; hidden_dim * kv_dim];
            for v in v_proj.iter_mut() {
                *v = next_float();
            }

            let mut o_proj = vec![0.0f32; q_dim * hidden_dim];
            for v in o_proj.iter_mut() {
                *v = next_float();
            }

            let mut post_attention_layernorm = vec![1.0f32; hidden_dim];
            for v in post_attention_layernorm.iter_mut() {
                *v = 1.0 + next_float() * 0.1;
            }

            let mut gate_proj = vec![0.0f32; hidden_dim * intermediate_dim];
            for v in gate_proj.iter_mut() {
                *v = next_float();
            }

            let mut up_proj = vec![0.0f32; hidden_dim * intermediate_dim];
            for v in up_proj.iter_mut() {
                *v = next_float();
            }

            let mut down_proj = vec![0.0f32; intermediate_dim * hidden_dim];
            for v in down_proj.iter_mut() {
                *v = next_float();
            }

            layers.push(TransformerLayerWeights {
                input_layernorm,
                q_proj: q_proj.into(),
                k_proj: k_proj.into(),
                v_proj: v_proj.into(),
                o_proj: o_proj.into(),
                post_attention_layernorm,
                gate_proj: gate_proj.into(),
                up_proj: up_proj.into(),
                down_proj: down_proj.into(),
                qkv_proj: None,
                gate_up_proj: None,
                q_norm: config.qk_norm.then(|| vec![1.0f32; config.head_dim]),
                k_norm: config.qk_norm.then(|| vec![1.0f32; config.head_dim]),
            });
        }

        let mut final_norm = vec![1.0f32; hidden_dim];
        for v in final_norm.iter_mut() {
            *v = 1.0 + next_float() * 0.1;
        }

        let lm_head = (!config.tie_word_embeddings).then(|| {
            let mut lm_head = vec![0.0f32; hidden_dim * vocab_size];
            for v in lm_head.iter_mut() {
                *v = next_float();
            }
            lm_head
        });

        Self {
            config: config.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head: lm_head.map(Into::into),
        }
    }

    /// Single-token forward pass through all transformer layers.
    /// Returns the output vector after final RMSNorm of shape [hidden_dim].
    pub fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim();
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = self.config.intermediate_dim();
        let eps = self.config.rms_norm_eps;
        let rope = self.config.rope();

        // 1. Embedding lookup: x = embed_tokens[token_id]
        let token_idx = (token_id as usize) % self.config.vocab_size();
        let embed_slice = &self.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
        let mut x = embed_slice.to_vec();

        // 2. Transformer layers
        for (layer_idx, layer_w) in self.layers.iter().enumerate() {
            let kv_cache = &mut seq_state.layers[layer_idx];

            // 2a. Input RMSNorm
            let mut x_norm = vec![0.0f32; hidden_dim];
            rmsnorm(&x, &layer_w.input_layernorm, eps, &mut x_norm);

            // 2b. Q, K, V Projections
            let mut q = vec![0.0f32; q_dim];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            matmul_vec(&x_norm, &layer_w.q_proj, &mut q, hidden_dim, q_dim);
            matmul_vec(&x_norm, &layer_w.k_proj, &mut k, hidden_dim, kv_dim);
            matmul_vec(&x_norm, &layer_w.v_proj, &mut v, hidden_dim, kv_dim);

            layer_w.apply_qk_norm(&mut q, &mut k, head_dim, eps);

            // 2c. Rotary Positional Embeddings (RoPE)
            apply_rope_params(
                &mut q,
                &mut k,
                pos,
                num_heads,
                num_kv_heads,
                head_dim,
                &rope,
            );

            // 2d. Append to KV Cache
            kv_cache.cached_k.push(k);
            kv_cache.cached_v.push(v);

            // 2e. Scaled Dot-Product Attention
            let mut attn_out = vec![0.0f32; q_dim];
            scaled_dot_product_attention_single(
                &q,
                &kv_cache.cached_k,
                &kv_cache.cached_v,
                num_heads,
                num_kv_heads,
                head_dim,
                &mut attn_out,
            );

            // 2f. Output projection and residual connection
            let mut attn_proj = vec![0.0f32; hidden_dim];
            matmul_vec(
                &attn_out,
                &layer_w.o_proj,
                &mut attn_proj,
                q_dim,
                hidden_dim,
            );
            for i in 0..hidden_dim {
                x[i] += attn_proj[i];
            }

            // 2g. Post-Attention RMSNorm
            let mut post_norm = vec![0.0f32; hidden_dim];
            rmsnorm(&x, &layer_w.post_attention_layernorm, eps, &mut post_norm);

            // 2h. SwiGLU MLP and residual connection
            let mut mlp_out = vec![0.0f32; hidden_dim];
            swiglu(
                &post_norm,
                &layer_w.gate_proj,
                &layer_w.up_proj,
                &layer_w.down_proj,
                hidden_dim,
                intermediate_dim,
                &mut mlp_out,
            );
            for i in 0..hidden_dim {
                x[i] += mlp_out[i];
            }
        }

        // 3. Final RMSNorm
        let mut x_final = vec![0.0f32; hidden_dim];
        rmsnorm(&x, &self.final_norm, eps, &mut x_final);
        x_final
    }

    /// Projects hidden states through LM Head and computes real vocabulary logits.
    pub fn compute_logits(&self, hidden_state: &[f32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim();
        let vocab_size = self.config.vocab_size();
        let mut logits = vec![0.0f32; vocab_size];
        matmul_vec(
            hidden_state,
            self.output_projection(),
            &mut logits,
            hidden_dim,
            vocab_size,
        );
        logits
    }

    /// Executes sequence forward pass capturing all intermediate activations and logits.
    pub fn forward_sequence_with_diagnostics(&self, tokens: &[u32]) -> ForwardDiagnostics {
        let n = tokens.len();
        let hidden_dim = self.config.hidden_dim();
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = self.config.intermediate_dim();
        let vocab_size = self.config.vocab_size();
        let eps = self.config.rms_norm_eps;
        let rope = self.config.rope();

        let mut activations = std::collections::HashMap::new();
        let mut shapes = std::collections::HashMap::new();

        // Register tensor shapes
        shapes.insert("embed_tokens".to_string(), vec![1, n, hidden_dim]);
        shapes.insert("final_norm".to_string(), vec![1, n, hidden_dim]);
        shapes.insert("logits".to_string(), vec![1, n, vocab_size]);
        shapes.insert("last_token_logits".to_string(), vec![vocab_size]);

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("layers.{}", layer_idx);
            shapes.insert(format!("{}.post_rmsnorm", prefix), vec![1, n, hidden_dim]);
            shapes.insert(
                format!("{}.post_rope_q", prefix),
                vec![1, num_heads, n, head_dim],
            );
            shapes.insert(
                format!("{}.post_rope_k", prefix),
                vec![1, num_kv_heads, n, head_dim],
            );
            shapes.insert(format!("{}.post_attention", prefix), vec![1, n, hidden_dim]);
            shapes.insert(
                format!("{}.post_attn_residual", prefix),
                vec![1, n, hidden_dim],
            );
            shapes.insert(format!("{}.post_attn_norm", prefix), vec![1, n, hidden_dim]);
            shapes.insert(format!("{}.post_swiglu", prefix), vec![1, n, hidden_dim]);
            shapes.insert(format!("{}.post_residual", prefix), vec![1, n, hidden_dim]);
        }

        // Allocate output buffers
        let mut embed_tokens_buf = vec![0.0f32; n * hidden_dim];
        let mut final_norm_buf = vec![0.0f32; n * hidden_dim];
        let mut logits_buf = vec![0.0f32; n * vocab_size];

        let mut post_rmsnorm_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();
        let mut post_rope_q_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; num_heads * n * head_dim])
            .collect();
        let mut post_rope_k_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; num_kv_heads * n * head_dim])
            .collect();
        let mut post_attention_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();
        let mut post_attn_residual_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();
        let mut post_attn_norm_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();
        let mut post_swiglu_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();
        let mut post_residual_bufs: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| vec![0.0f32; n * hidden_dim])
            .collect();

        let mut seq_state = SequenceState {
            tokens: Vec::with_capacity(n),
            layers: vec![LayerKvCache::default(); self.config.num_layers],
            kv_dtype: KvDType::Fp32,
        };

        for (pos, &token_id) in tokens.iter().enumerate() {
            seq_state.tokens.push(token_id);

            // 1. Embedding lookup
            let token_idx = (token_id as usize) % vocab_size;
            let embed_slice =
                &self.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
            embed_tokens_buf[pos * hidden_dim..(pos + 1) * hidden_dim].copy_from_slice(embed_slice);
            let mut x = embed_slice.to_vec();

            // 2. Transformer layers
            for (layer_idx, layer_w) in self.layers.iter().enumerate() {
                let kv_cache = &mut seq_state.layers[layer_idx];

                // 2a. Input RMSNorm
                let mut x_norm = vec![0.0f32; hidden_dim];
                rmsnorm(&x, &layer_w.input_layernorm, eps, &mut x_norm);
                post_rmsnorm_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&x_norm);

                // 2b. Q, K, V Projections
                let mut q = vec![0.0f32; q_dim];
                let mut k = vec![0.0f32; kv_dim];
                let mut v = vec![0.0f32; kv_dim];
                matmul_vec(&x_norm, &layer_w.q_proj, &mut q, hidden_dim, q_dim);
                matmul_vec(&x_norm, &layer_w.k_proj, &mut k, hidden_dim, kv_dim);
                matmul_vec(&x_norm, &layer_w.v_proj, &mut v, hidden_dim, kv_dim);

                layer_w.apply_qk_norm(&mut q, &mut k, head_dim, eps);

                // 2c. Rotary Positional Embeddings (RoPE)
                apply_rope_params(
                    &mut q,
                    &mut k,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    &rope,
                );

                // Store RoPE Q in [1, num_heads, n, head_dim] layout
                for h in 0..num_heads {
                    let src_slice = &q[h * head_dim..(h + 1) * head_dim];
                    let dst_idx = h * (n * head_dim) + pos * head_dim;
                    post_rope_q_bufs[layer_idx][dst_idx..dst_idx + head_dim]
                        .copy_from_slice(src_slice);
                }

                // Store RoPE K in [1, num_kv_heads, n, head_dim] layout
                for kv_h in 0..num_kv_heads {
                    let src_slice = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let dst_idx = kv_h * (n * head_dim) + pos * head_dim;
                    post_rope_k_bufs[layer_idx][dst_idx..dst_idx + head_dim]
                        .copy_from_slice(src_slice);
                }

                // 2d. Append to KV Cache
                kv_cache.cached_k.push(k);
                kv_cache.cached_v.push(v);

                // 2e. Scaled Dot-Product Attention
                let mut attn_out = vec![0.0f32; q_dim];
                scaled_dot_product_attention_single(
                    &q,
                    &kv_cache.cached_k,
                    &kv_cache.cached_v,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    &mut attn_out,
                );

                // 2f. Output projection and residual connection
                let mut attn_proj = vec![0.0f32; hidden_dim];
                matmul_vec(
                    &attn_out,
                    &layer_w.o_proj,
                    &mut attn_proj,
                    q_dim,
                    hidden_dim,
                );
                post_attention_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&attn_proj);

                for i in 0..hidden_dim {
                    x[i] += attn_proj[i];
                }
                post_attn_residual_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&x);

                // 2g. Post-Attention RMSNorm
                let mut post_norm = vec![0.0f32; hidden_dim];
                rmsnorm(&x, &layer_w.post_attention_layernorm, eps, &mut post_norm);
                post_attn_norm_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&post_norm);

                // 2h. SwiGLU MLP and residual connection
                let mut mlp_out = vec![0.0f32; hidden_dim];
                swiglu(
                    &post_norm,
                    &layer_w.gate_proj,
                    &layer_w.up_proj,
                    &layer_w.down_proj,
                    hidden_dim,
                    intermediate_dim,
                    &mut mlp_out,
                );
                post_swiglu_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&mlp_out);

                for i in 0..hidden_dim {
                    x[i] += mlp_out[i];
                }
                post_residual_bufs[layer_idx][pos * hidden_dim..(pos + 1) * hidden_dim]
                    .copy_from_slice(&x);
            }

            // 3. Final RMSNorm
            let mut x_final = vec![0.0f32; hidden_dim];
            rmsnorm(&x, &self.final_norm, eps, &mut x_final);
            final_norm_buf[pos * hidden_dim..(pos + 1) * hidden_dim].copy_from_slice(&x_final);

            // 4. LM Head Logits
            let tok_logits = self.compute_logits(&x_final);
            logits_buf[pos * vocab_size..(pos + 1) * vocab_size].copy_from_slice(&tok_logits);
        }

        let last_token_logits = logits_buf[(n - 1) * vocab_size..n * vocab_size].to_vec();

        activations.insert("embed_tokens".to_string(), embed_tokens_buf);
        activations.insert("final_norm".to_string(), final_norm_buf);
        activations.insert("logits".to_string(), logits_buf);
        activations.insert("last_token_logits".to_string(), last_token_logits);

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("layers.{}", layer_idx);
            activations.insert(
                format!("{}.post_rmsnorm", prefix),
                post_rmsnorm_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_rope_q", prefix),
                post_rope_q_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_rope_k", prefix),
                post_rope_k_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_attention", prefix),
                post_attention_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_attn_residual", prefix),
                post_attn_residual_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_attn_norm", prefix),
                post_attn_norm_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_swiglu", prefix),
                post_swiglu_bufs[layer_idx].clone(),
            );
            activations.insert(
                format!("{}.post_residual", prefix),
                post_residual_bufs[layer_idx].clone(),
            );
        }

        ForwardDiagnostics {
            activations,
            shapes,
        }
    }

    /// Autoregressive greedy decode for a given sequence of prompt tokens.
    pub fn generate_greedy(&self, prompt_tokens: &[u32], num_steps: usize) -> Vec<u32> {
        let mut seq_state = SequenceState {
            tokens: Vec::with_capacity(prompt_tokens.len() + num_steps),
            layers: vec![LayerKvCache::default(); self.config.num_layers],
            kv_dtype: KvDType::Fp32,
        };

        let mut last_hidden = Vec::new();
        for (idx, &token_id) in prompt_tokens.iter().enumerate() {
            seq_state.tokens.push(token_id);
            last_hidden = self.forward_token(token_id, idx, &mut seq_state);
        }

        let mut generated = Vec::with_capacity(num_steps);
        let mut logits = self.compute_logits(&last_hidden);
        let (mut next_tok, _) = crate::tensor::sample_argmax(&logits);
        generated.push(next_tok);

        for _ in 1..num_steps {
            let pos = seq_state.tokens.len();
            seq_state.tokens.push(next_tok);
            let hidden = self.forward_token(next_tok, pos, &mut seq_state);
            logits = self.compute_logits(&hidden);
            next_tok = crate::tensor::sample_argmax(&logits).0;
            generated.push(next_tok);
        }

        generated
    }

    /// Loads model weights directly from a standard safetensors binary buffer.
    pub fn from_safetensors_bytes(bytes: &[u8], config: &ModelConfig) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err(
                "Invalid safetensors binary: buffer smaller than 8 bytes header".to_string(),
            );
        }

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err("Invalid safetensors binary: header length exceeds buffer".to_string());
        }

        let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
            .map_err(|e| format!("Safetensors header is not valid UTF-8: {}", e))?;

        let header: Value = serde_json::from_str(header_str)
            .map_err(|e| format!("Failed to parse safetensors JSON header: {}", e))?;

        let data_offset = 8 + header_len;
        let data_bytes = &bytes[data_offset..];

        let extract_tensor = |name: &str| -> Option<Vec<f32>> {
            let info = header.get(name)?;
            let offsets = info.get("data_offsets")?.as_array()?;
            let start = offsets.first()?.as_u64()? as usize;
            let end = offsets.get(1)?.as_u64()? as usize;
            let dtype = info.get("dtype")?.as_str()?;

            if end > data_bytes.len() || start > end {
                return None;
            }

            let raw_slice = &data_bytes[start..end];
            match dtype {
                "F32" => {
                    let count = (end - start) / 4;
                    let mut floats = Vec::with_capacity(count);
                    for chunk in raw_slice.chunks_exact(4) {
                        floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    Some(floats)
                }
                "BF16" => {
                    let count = (end - start) / 2;
                    let mut floats = Vec::with_capacity(count);
                    for chunk in raw_slice.chunks_exact(2) {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        floats.push(f32::from_bits((bits as u32) << 16));
                    }
                    Some(floats)
                }
                "F16" => {
                    let count = (end - start) / 2;
                    let mut floats = Vec::with_capacity(count);
                    for chunk in raw_slice.chunks_exact(2) {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        floats.push(half_to_float(bits));
                    }
                    Some(floats)
                }
                _ => None,
            }
        };

        let mut weights = Self::reference_test_weights(config);

        if let Some(t) = extract_tensor("model.embed_tokens.weight") {
            weights.embed_tokens = t;
        }

        for (idx, layer) in weights.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", idx);
            if let Some(t) = extract_tensor(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm = t;
            }
            if let Some(t) = extract_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.q_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.self_attn.k_proj.weight", prefix)) {
                layer.k_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.self_attn.v_proj.weight", prefix)) {
                layer.v_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.self_attn.o_proj.weight", prefix)) {
                layer.o_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.post_attention_layernorm.weight", prefix))
            {
                layer.post_attention_layernorm = t;
            }
            if let Some(t) = extract_tensor(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.gate_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.mlp.up_proj.weight", prefix)) {
                layer.up_proj = t.into();
            }
            if let Some(t) = extract_tensor(&format!("{}.mlp.down_proj.weight", prefix)) {
                layer.down_proj = t.into();
            }
            if config.qk_norm {
                layer.q_norm = extract_tensor(&format!("{}.self_attn.q_norm.weight", prefix));
                layer.k_norm = extract_tensor(&format!("{}.self_attn.k_norm.weight", prefix));
            }
        }

        if let Some(t) = extract_tensor("model.norm.weight") {
            weights.final_norm = t;
        }

        if !config.tie_word_embeddings {
            if let Some(t) = extract_tensor("lm_head.weight") {
                weights.lm_head = Some(t.into());
            }
        }

        Ok(weights)
    }

    /// Loads weights strictly from a pre-validated LoadedCheckpoint without synthetic fallback.
    /// FP32 tensors are decoded from the checkpoint capsule on demand. They are not stored twice.
    pub fn from_loaded_checkpoint(
        checkpoint: &crate::checkpoint::LoadedCheckpoint,
        config: &ModelConfig,
    ) -> Result<Self, crate::checkpoint::CheckpointError> {
        let decode = |name: &str| -> Result<Vec<f32>, crate::checkpoint::CheckpointError> {
            checkpoint.decode_fp32(name).map_err(|err| match err {
                crate::capsule::CapsuleError::MissingTensor(name) => {
                    crate::checkpoint::CheckpointError::MissingTensor(name)
                }
                other => crate::checkpoint::CheckpointError::Capsule(other.to_string()),
            })
        };

        let embed_tokens = decode("model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            let prefix = format!("model.layers.{}", idx);

            let input_layernorm = decode(&format!("{}.input_layernorm.weight", prefix))?;
            let q_proj = decode(&format!("{}.self_attn.q_proj.weight", prefix))?;
            let k_proj = decode(&format!("{}.self_attn.k_proj.weight", prefix))?;
            let v_proj = decode(&format!("{}.self_attn.v_proj.weight", prefix))?;
            let o_proj = decode(&format!("{}.self_attn.o_proj.weight", prefix))?;
            let post_attention_layernorm =
                decode(&format!("{}.post_attention_layernorm.weight", prefix))?;
            let gate_proj = decode(&format!("{}.mlp.gate_proj.weight", prefix))?;
            let up_proj = decode(&format!("{}.mlp.up_proj.weight", prefix))?;
            let down_proj = decode(&format!("{}.mlp.down_proj.weight", prefix))?;
            let (q_norm, k_norm) = if config.qk_norm {
                (
                    Some(decode(&format!("{}.self_attn.q_norm.weight", prefix))?),
                    Some(decode(&format!("{}.self_attn.k_norm.weight", prefix))?),
                )
            } else {
                (None, None)
            };

            layers.push(TransformerLayerWeights {
                input_layernorm,
                q_proj: q_proj.into(),
                k_proj: k_proj.into(),
                v_proj: v_proj.into(),
                o_proj: o_proj.into(),
                post_attention_layernorm,
                gate_proj: gate_proj.into(),
                up_proj: up_proj.into(),
                down_proj: down_proj.into(),
                qkv_proj: None,
                gate_up_proj: None,
                q_norm,
                k_norm,
            });
        }

        let final_norm = decode("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(decode("lm_head.weight")?)
        };

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head: lm_head.map(Into::into),
        })
    }

    /// Loads model weights strictly from a safetensors checkpoint on disk, validated against
    /// the tensor catalog `config` describes (`checkpoint::llama_catalog`). `path` is one
    /// `.safetensors` file or a sharded `model.safetensors.index.json`.
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        config: &ModelConfig,
    ) -> Result<Self, crate::checkpoint::CheckpointError> {
        let catalog = crate::checkpoint::llama_catalog(config);
        let checkpoint = crate::checkpoint::load_checkpoint_with_catalog(path, &catalog)?;
        Self::from_loaded_checkpoint(&checkpoint, config)
    }
}

/// Converts IEEE 754 half-precision float (f16) bits to f32.
fn half_to_float(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut f = frac;
            let mut shift = 0;
            while (f & 0x400) == 0 {
                f <<= 1;
                shift += 1;
            }
            let exp32 = (127 - 15 - shift + 1) as u32;
            let frac32 = (f & 0x3ff) << 13;
            f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
    } else {
        let exp32 = exp + 127 - 15;
        let frac32 = frac << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_weights_initialization() {
        let config = ModelConfig {
            model_id: "test-llama".to_string(),
            max_sequence_length: 2048,
            block_size: 16,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            num_kv_heads: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_ids: Vec::new(),
            qk_norm: false,
        };

        let weights = TransformerWeights::reference_test_weights(&config);
        assert_eq!(weights.embed_tokens.len(), 256 * 64);
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.layers[0].q_proj.len(), 64 * 64);
        assert_eq!(weights.layers[0].k_proj.len(), 64 * 64);
        assert_eq!(weights.layers[0].v_proj.len(), 64 * 64);
        assert_eq!(weights.layers[0].gate_proj.len(), 64 * 128);
        assert_eq!(weights.layers[0].down_proj.len(), 128 * 64);
        assert_eq!(weights.final_norm.len(), 64);
        assert_eq!(weights.output_projection().len(), 64 * 256);
    }

    #[test]
    fn test_from_loaded_checkpoint_success_and_missing_key() {
        let config = ModelConfig {
            model_id: "test-llama".to_string(),
            max_sequence_length: 2048,
            block_size: 16,
            num_layers: 1,
            num_heads: 4,
            head_dim: 16,
            num_kv_heads: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_ids: Vec::new(),
            qk_norm: false,
        };

        // 1. Missing tensor fails loudly with CheckpointError::MissingTensor
        let empty_checkpoint = crate::checkpoint::LoadedCheckpoint::new();
        let err =
            TransformerWeights::from_loaded_checkpoint(&empty_checkpoint, &config).unwrap_err();
        assert_eq!(
            err,
            crate::checkpoint::CheckpointError::MissingTensor(
                "model.embed_tokens.weight".to_string()
            )
        );

        // 2. Populated checkpoint loads correctly.
        // Values are exact in bf16 so decode_fp32 round-trips without rounding.
        let bf16_filled = |name: &str, shape: Vec<usize>, value: f32| {
            let n: usize = shape.iter().product();
            let values = vec![value; n];
            (
                name.to_string(),
                shape,
                crate::checkpoint::encode_fp32_to_bf16(&values),
            )
        };
        let checkpoint = crate::checkpoint::LoadedCheckpoint::from_bf16_tensors(vec![
            bf16_filled("model.embed_tokens.weight", vec![256, 64], 0.5),
            bf16_filled("model.layers.0.input_layernorm.weight", vec![64], 1.0),
            bf16_filled("model.layers.0.self_attn.q_proj.weight", vec![64, 64], 2.0),
            bf16_filled("model.layers.0.self_attn.k_proj.weight", vec![64, 64], 0.5),
            bf16_filled("model.layers.0.self_attn.v_proj.weight", vec![64, 64], 1.0),
            bf16_filled("model.layers.0.self_attn.o_proj.weight", vec![64, 64], 0.5),
            bf16_filled(
                "model.layers.0.post_attention_layernorm.weight",
                vec![64],
                1.0,
            ),
            bf16_filled("model.layers.0.mlp.gate_proj.weight", vec![128, 64], 2.0),
            bf16_filled("model.layers.0.mlp.up_proj.weight", vec![128, 64], 0.5),
            bf16_filled("model.layers.0.mlp.down_proj.weight", vec![64, 128], 1.0),
            bf16_filled("model.norm.weight", vec![64], 1.0),
            bf16_filled("lm_head.weight", vec![256, 64], 0.5),
        ]);

        let weights = TransformerWeights::from_loaded_checkpoint(&checkpoint, &config).unwrap();
        assert_eq!(weights.embed_tokens.len(), 256 * 64);
        assert_eq!(weights.embed_tokens[0], 0.5);
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.layers[0].q_proj[0], 2.0);
        assert_eq!(weights.final_norm.len(), 64);
        assert_eq!(weights.final_norm[0], 1.0);
        assert_eq!(weights.output_projection()[0], 0.5);
    }
}
