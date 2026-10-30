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

//! Common data types and schemas for the neutral apples-to-apples benchmark suite.

use serde::{Deserialize, Serialize};

/// High-level benchmark execution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub prompt: String,
    pub prompt_token_count: usize,
    pub output_token_count: usize,
    pub concurrency_levels: Vec<usize>,
    pub cool_down_target_power_watts: f64,
    pub cool_down_max_temp_c: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            model_path: format!("{}/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6", std::env::var("HOME").unwrap_or_default()),
            tokenizer_path: format!("{}/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json", std::env::var("HOME").unwrap_or_default()),
            prompt: "<|system|>\nYou are a helpful and truthful AI assistant. Explain technical concepts with clarity and precision.</s>\n<|user|>\nProvide an overview of modern computer architecture, memory hierarchies, cache coherency protocols, and unified memory subsystems across CPU and GPU accelerators. Specifically discuss: virtual memory translation lookaside buffers page tables interconnect bandwidth latency coherence invalidation snooping directory protocols numa domains shared address spaces hardware acceleration tensor pipelines register files instruction sched</s>\n<|assistant|>\n".to_string(),
            prompt_token_count: 128,
            output_token_count: 128,
            concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64],
            cool_down_target_power_watts: 12.0,
            cool_down_max_temp_c: 42.0,
        }
    }
}

/// Aggregated statistical and hardware telemetry for an engine run at a given concurrency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyRunResult {
    pub engine: String,
    pub concurrency: usize,
    pub total_requests: usize,
    pub total_output_tokens: usize,
    pub elapsed_wall_clock_secs: f64,
    pub throughput_tokens_sec: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
    pub peak_rss_gb: f64,
    pub avg_gpu_power_watts: f64,
    pub peak_gpu_power_watts: f64,
    pub energy_joules_per_token: f64,
    pub parity_match_rate_pct: f64,
    pub sample_text: String,
}

/// Summary report of the full benchmark run across all engines and concurrency sweeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark_timestamp: String,
    pub hardware: String,
    pub model_id: String,
    pub precision: String,
    pub prompt_tokens: usize,
    pub target_output_tokens: usize,
    pub results: Vec<ConcurrencyRunResult>,
    pub parity_oracle_sequence: Vec<u32>,
}
