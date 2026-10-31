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

//! AIEN Native Transformer engine benchmark driver with Blackwell GB10 GPU acceleration.
//! Executes real forward computation on DGX Spark GB10 in unified memory.

use crate::metrics::{calculate_joules_per_token, calculate_percentile};
use crate::oracle::{verify_token_sequence_match, ORACLE_BENCHMARK_128_TOKENS};
use crate::telemetry::HardwareMonitor;
use crate::types::{BenchmarkConfig, ConcurrencyRunResult};
use aien_inference_abi::tensor::sample_argmax;
use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::weights::{LayerKvCache, SequenceState, TransformerWeights};
use aien_inference_abi::{KvDType, ModelConfig};
use std::path::PathBuf;
use std::time::Instant;

pub struct AienEngineHandle {
    pub backend: NativeTransformerBackend,
    pub tokenizer: TinyLlamaTokenizer,
}

impl AienEngineHandle {
    pub fn init(config: &BenchmarkConfig) -> Result<Self, String> {
        eprintln!("Initializing AIEN Native Transformer with Blackwell GB10 GPU Backend...");
        let model_config = ModelConfig {
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
        };

        let safetensors_path = format!("{}/model.safetensors", config.model_path);
        eprintln!("Loading safetensors weights from {}...", safetensors_path);
        let weights = TransformerWeights::load_from_safetensors(&safetensors_path, &model_config)
            .map_err(|e| format!("Failed to construct TransformerWeights: {}", e))?;

        let backend = NativeTransformerBackend::new_blackwell(weights);
        eprintln!(
            "AIEN Backend initialized: {}",
            backend.tensor_backend.name()
        );

        let tokenizer_path = resolve_tokenizer_path(&config.tokenizer_path, &config.model_path)?;
        eprintln!("Loading tokenizer from {}...", tokenizer_path.display());
        let tokenizer = TinyLlamaTokenizer::from_file(&tokenizer_path).map_err(|e| {
            format!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?;

        Ok(Self { backend, tokenizer })
    }
}

fn resolve_tokenizer_path(configured: &str, model_path: &str) -> Result<PathBuf, String> {
    let candidates = [
        PathBuf::from(configured),
        PathBuf::from(model_path).join("tokenizer.json"),
        PathBuf::from("crates/aien-inference-abi/fixtures/tokenizer.json"),
    ];

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(format!(
        "Tokenizer file not found. Checked: {:?}",
        candidates
    ))
}

struct SingleRequestResult {
    request_id: usize,
    ttft_ms: f64,
    itls_ms: Vec<f64>,
    output_tokens: Vec<u32>,
}

fn execute_single_request(
    request_id: usize,
    weights: &TransformerWeights,
    tensor_backend: &dyn aien_inference_abi::backend::TensorBackend,
    prompt_tokens: &[u32],
    max_output_tokens: usize,
) -> SingleRequestResult {
    let num_layers = weights.config.num_layers;
    // new_blackwell, the constructor this bench calls, sets kv_dtype to Fp32 and
    // attaches no pool. storage_dtype() is that value with no pool, and
    // prefill_sequence / execute_step write it onto each new SequenceState.
    let mut seq = SequenceState {
        tokens: Vec::with_capacity(prompt_tokens.len() + max_output_tokens),
        layers: vec![LayerKvCache::default(); num_layers],
        kv_dtype: KvDType::Fp32,
    };

    let req_start = Instant::now();

    // Prefill: forward all prompt tokens layer-by-layer
    let last_hidden = NativeTransformerBackend::prefill_prompt_layer_by_layer(
        weights,
        tensor_backend,
        prompt_tokens,
        &mut seq,
    );

    let prefill_logits =
        NativeTransformerBackend::compute_logits_impl(weights, tensor_backend, &last_hidden);
    let (first_tok, _) = sample_argmax(&prefill_logits);
    let ttft_ms = req_start.elapsed().as_secs_f64() * 1000.0;
    seq.tokens.push(first_tok);

    let mut output_tokens = Vec::with_capacity(max_output_tokens);
    output_tokens.push(first_tok);
    let mut itls_ms = Vec::with_capacity(max_output_tokens);

    let stop_tokens = [
        TinyLlamaTokenizer::UNK_TOKEN_ID,
        TinyLlamaTokenizer::BOS_TOKEN_ID,
        TinyLlamaTokenizer::EOS_TOKEN_ID,
    ];

    if stop_tokens.contains(&first_tok) {
        return SingleRequestResult {
            request_id,
            ttft_ms,
            itls_ms,
            output_tokens,
        };
    }

    // Decode loop
    let remaining_tokens = max_output_tokens.saturating_sub(1);
    for _ in 0..remaining_tokens {
        let step_start = Instant::now();
        let last_tok = *seq.tokens.last().unwrap_or(&1);
        let pos = seq.tokens.len().saturating_sub(1);

        let hidden = NativeTransformerBackend::forward_token_impl(
            weights,
            tensor_backend,
            last_tok,
            pos,
            &mut seq,
        );
        let logits =
            NativeTransformerBackend::compute_logits_impl(weights, tensor_backend, &hidden);
        let (next_tok, _) = sample_argmax(&logits);
        let itl_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        seq.tokens.push(next_tok);
        output_tokens.push(next_tok);
        itls_ms.push(itl_ms);

        if stop_tokens.contains(&next_tok) {
            break;
        }
    }

    SingleRequestResult {
        request_id,
        ttft_ms,
        itls_ms,
        output_tokens,
    }
}

/// Runs a concurrency sweep for AIEN Native Transformer at concurrency level C.
pub async fn run_aien_concurrency_sweep(
    handle: &mut AienEngineHandle,
    config: &BenchmarkConfig,
    concurrency: usize,
) -> Result<ConcurrencyRunResult, String> {
    eprintln!(
        "Executing AIEN FP32 Native ({}) benchmark at concurrency C={}...",
        handle.backend.tensor_backend.name(),
        concurrency
    );

    let prompt_tokens = handle
        .tokenizer
        .encode(&config.prompt)
        .map_err(|e| format!("Tokenizer encode error: {}", e))?;

    let monitor = HardwareMonitor::start(None);
    let wall_clock_start = Instant::now();

    let weights = &handle.backend.weights;
    let tensor_backend = handle.backend.tensor_backend.as_ref();
    let max_output_tokens = config.output_token_count;

    // Concurrently execute requests using scoped threads up to max_parallel_workers
    let max_parallel_workers = 16.min(concurrency);
    let mut all_results: Vec<SingleRequestResult> = Vec::with_capacity(concurrency);

    // Process requests in chunks of max_parallel_workers
    for chunk_start in (0..concurrency).step_by(max_parallel_workers) {
        let chunk_end = (chunk_start + max_parallel_workers).min(concurrency);
        let chunk_size = chunk_end - chunk_start;

        let chunk_results = std::thread::scope(|s| {
            let mut thread_handles = Vec::with_capacity(chunk_size);
            for req_idx in chunk_start..chunk_end {
                let p_tokens = &prompt_tokens;
                thread_handles.push(s.spawn(move || {
                    execute_single_request(
                        req_idx,
                        weights,
                        tensor_backend,
                        p_tokens,
                        max_output_tokens,
                    )
                }));
            }

            let mut results = Vec::with_capacity(chunk_size);
            for th in thread_handles {
                if let Ok(res) = th.join() {
                    results.push(res);
                }
            }
            results
        });

        all_results.extend(chunk_results);
    }

    let wall_clock_elapsed = wall_clock_start.elapsed().as_secs_f64();
    let (avg_power_watts, peak_power_watts, peak_rss_gb) = monitor.stop().await;

    if all_results.is_empty() {
        return Err("All requests failed during AIEN concurrency sweep.".to_string());
    }

    all_results.sort_by_key(|r| r.request_id);

    let mut all_ttft = Vec::with_capacity(concurrency);
    let mut all_itl = Vec::new();
    let mut total_output_tokens = 0usize;

    for r in &all_results {
        all_ttft.push(r.ttft_ms);
        all_itl.extend_from_slice(&r.itls_ms);
        total_output_tokens += r.output_tokens.len();
    }

    let throughput = (total_output_tokens as f64) / wall_clock_elapsed;
    let ttft_p50 = calculate_percentile(&all_ttft, 50.0);
    let ttft_p95 = calculate_percentile(&all_ttft, 95.0);
    let ttft_p99 = calculate_percentile(&all_ttft, 99.0);
    let itl_p50 = calculate_percentile(&all_itl, 50.0);
    let itl_p95 = calculate_percentile(&all_itl, 95.0);
    let itl_p99 = calculate_percentile(&all_itl, 99.0);
    let energy_j_per_tok =
        calculate_joules_per_token(avg_power_watts, wall_clock_elapsed, total_output_tokens);

    let first_req_tokens = &all_results[0].output_tokens;
    let parity_match_rate_pct =
        verify_token_sequence_match(first_req_tokens, &ORACLE_BENCHMARK_128_TOKENS);
    let sample_text = handle
        .tokenizer
        .decode(first_req_tokens)
        .unwrap_or_default();

    Ok(ConcurrencyRunResult {
        engine: format!("AIEN FP32 ({})", handle.backend.tensor_backend.name()),
        concurrency,
        total_requests: all_results.len(),
        total_output_tokens,
        elapsed_wall_clock_secs: wall_clock_elapsed,
        throughput_tokens_sec: throughput,
        ttft_p50_ms: ttft_p50,
        ttft_p95_ms: ttft_p95,
        ttft_p99_ms: ttft_p99,
        itl_p50_ms: itl_p50,
        itl_p95_ms: itl_p95,
        itl_p99_ms: itl_p99,
        peak_rss_gb,
        avg_gpu_power_watts: avg_power_watts,
        peak_gpu_power_watts: peak_power_watts,
        energy_joules_per_token: energy_j_per_tok,
        parity_match_rate_pct,
        sample_text,
    })
}
