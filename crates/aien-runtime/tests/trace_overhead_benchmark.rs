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

//! Overhead of the swarm trace events (#398). Run by hand:
//! `cargo test -p aien-runtime --release --test trace_overhead_benchmark -- --ignored --nocapture`
//! Prints one line per run; asserts nothing about timing.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use aien_inference_abi::{
    ModelConfig, NativeTransformerBackend, ReferenceCpuBackend, TransformerWeights,
};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::swarm::SwarmConfig;
use aien_scheduler::SchedulerConfig;

const PROMPT: [u32; 6] = [5, 17, 42, 3, 88, 61];

fn model_config() -> ModelConfig {
    ModelConfig {
        model_id: "trace-reference".to_string(),
        max_sequence_length: 256,
        block_size: 4,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        num_kv_heads: 2,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 97,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 2,
    }
}

fn swarm_config(branches: usize, max_tokens: usize) -> SwarmConfig {
    SwarmConfig {
        model_handle: 1,
        branch_count: branches,
        max_active_sequences: branches,
        max_tokens_per_branch: max_tokens,
        root_world_id: 0,
        priority: 1,
    }
}

/// One fresh controller state directory per test process.
fn fresh_state_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("aien-trace-test-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).expect("create fresh state dir");
        std::env::set_var("AIEN_RUNTIME_STATE_DIR", &dir);
        dir
    })
}

fn build(branches: usize) -> (AienRuntimeSpine, NativeTransformerBackend) {
    fresh_state_dir();
    let weights = TransformerWeights::reference_test_weights(&model_config());
    build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        scheduler_config(),
        SharedKvSizing {
            arena_capacity: branches + 16,
            total_blocks: (branches + 16) * 4,
        },
    )
    .expect("build shared KV runtime")
}

async fn run_to_completion(spine: &mut AienRuntimeSpine, backend: &mut NativeTransformerBackend) {
    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0)
        && steps < 500
    {
        steps += 1;
        spine.step(backend).await.expect("spine step");
    }
}

use aien_trace::{MemorySink, NullSink, TraceSink};

const BRANCHES: usize = 64;

async fn run(kind: &str, sink: Arc<dyn TraceSink>) {
    let (mut spine, mut backend) = build(BRANCHES);
    spine.swarm_manager.set_trace_sink(sink);
    let t = std::time::Instant::now();
    spine
        .launch_swarm(swarm_config(BRANCHES, 3), &PROMPT)
        .expect("launch");
    run_to_completion(&mut spine, &mut backend).await;
    println!(
        "TRACE_OVERHEAD kind={kind} branches={BRANCHES} elapsed_ms={:.3}",
        t.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        BRANCHES
    );
}

#[tokio::test]
#[ignore]
async fn trace_overhead_null_sink() {
    run("null", Arc::new(NullSink)).await;
}

#[tokio::test]
#[ignore]
async fn trace_overhead_memory_sink() {
    run("memory", Arc::new(MemorySink::new())).await;
}
