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

//! A sequence preempted under KV pressure must resume without re-emitting
//! tokens.
//!
//! Runs the real spine, scheduler and native backend over the CPU reference
//! tensor backend, built through the daemon's factory
//! (`aien_runtime::shared_kv::build_shared_kv_runtime`). No persistence.
//!
//! Before the fix, the scheduler kept a preempted sequence's generated tokens
//! but re-prefilled only its prompt, and the backend's per-sequence state was
//! never released on preemption. The resume then either re-sampled and
//! re-emitted tokens the sink already had, or the backend refused a decode at
//! a stale position.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_abi_core::{ModelConfig, SamplingParams};
use aien_inference_abi::{NativeTransformerBackend, ReferenceCpuBackend, TransformerWeights};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{
    ChannelCompletionSink, CompletionEvent, CompletionSinkId, SchedulerConfig, SequenceId,
    SequencePhase,
};
use tokio::sync::mpsc::UnboundedReceiver;

const HANG_GUARD: Duration = Duration::from_secs(30);

fn model() -> ModelConfig {
    ModelConfig {
        model_id: "preempt-resume-reference".to_string(),
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
        // Outside the 97-token vocabulary: no sampled token ends a turn early
        // (an empty list means the legacy stop set 0, 1, 2).
        eos_token_ids: vec![1000],
        qk_norm: false,
    }
}

struct Rig {
    spine: AienRuntimeSpine,
    backend: NativeTransformerBackend,
    sink: CompletionSinkId,
    rx: UnboundedReceiver<CompletionEvent>,
    /// Tokens streamed to the completion sink, per sequence id.
    tokens: HashMap<u64, Vec<u32>>,
    steps: u64,
}

fn rig(total_blocks: usize, watermark_blocks: usize, max_batch_size: usize) -> Rig {
    let config = model();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        SchedulerConfig {
            max_batch_size,
            max_batch_tokens: 4096,
            max_prefill_tokens: 4096,
            prefill_chunk_size: 4096,
            chunk_prefill: true,
            watermark_blocks,
        },
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks,
        },
    )
    .expect("runtime");
    let (sink, rx) = ChannelCompletionSink::channel();
    let sink = spine.register_completion_sink(Arc::new(sink));
    Rig {
        spine,
        backend,
        sink,
        rx,
        tokens: HashMap::new(),
        steps: 0,
    }
}

fn phase_of(r: &Rig, seq: u64) -> Option<SequencePhase> {
    let id = SequenceId::from_u64(seq).ok()?;
    r.spine.scheduler.arena().get(id).map(|rec| rec.phase)
}

fn finished(r: &Rig, seq: u64) -> bool {
    phase_of(r, seq).is_none()
}

fn emitted(r: &Rig, seq: u64) -> usize {
    r.tokens.get(&seq).map_or(0, Vec::len)
}

fn toks(r: &Rig, seq: u64) -> Vec<u32> {
    r.tokens.get(&seq).cloned().unwrap_or_default()
}

fn submit(r: &mut Rig, prompt: &[u32], max_tokens: usize) -> u64 {
    r.spine
        .submit_work(
            Arc::from(prompt.to_vec()),
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens,
                stop_token_ids: vec![],
            },
            1,
            Some(r.sink),
        )
        .expect("submit")
}

async fn step(r: &mut Rig) {
    r.spine.step(&mut r.backend).await.expect("step");
    r.steps += 1;
    while let Ok(ev) = r.rx.try_recv() {
        if let CompletionEvent::Token { seq_id, token } = ev {
            r.tokens.entry(seq_id.to_u64()).or_default().push(token);
        }
    }
}

/// Steps until `cond` holds, at most `max_steps` steps.
async fn step_until(r: &mut Rig, what: &str, max_steps: u64, cond: impl Fn(&Rig) -> bool) {
    let t0 = Instant::now();
    let start = r.steps;
    loop {
        if cond(r) {
            return;
        }
        assert!(
            r.steps - start < max_steps,
            "no progress within {max_steps} steps: {what}"
        );
        assert!(t0.elapsed() < HANG_GUARD, "hang guard: {what}");
        step(r).await;
    }
}

/// The greedy tokens of `prompt` on a roomy runtime that never preempts.
async fn uninterrupted(prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut r = rig(64, 0, 16);
    let seq = submit(&mut r, prompt, max_tokens);
    step_until(&mut r, "uninterrupted run", 500, |r| finished(r, seq)).await;
    assert_eq!(r.spine.scheduler.metrics().preempted_requests, 0);
    let t = toks(&r, seq);
    assert_eq!(t.len(), max_tokens, "uninterrupted run emitted every token");
    t
}

fn prompt_a() -> Vec<u32> {
    vec![5, 17, 42, 3, 88, 61, 9, 23, 70, 11, 36, 2, 44, 19, 80]
}

fn prompt_b() -> Vec<u32> {
    vec![31, 7, 64, 12, 90, 27, 55, 4, 73, 18, 39, 66, 21, 85, 50]
}

/// A sequence preempted by the watermark check and resumed later delivers
/// every token to its sink exactly once, and its output equals a run that was
/// never preempted.
///
/// Geometry: a 10-block pool with a watermark of 3. Request A (15-token
/// prompt, 4 blocks) decodes alone for a few steps. Request B (another 4
/// blocks) is admitted while 3 or more blocks are free, which leaves fewer
/// than the watermark, so the next step preempts A (the lowest-priority
/// running sequence, first in running order). A needs 4 blocks plus the
/// watermark headroom to come back, so it resumes only after B finishes.
#[tokio::test]
async fn preempted_sequence_resumes_without_re_emitting_tokens() {
    let max_a = 12;
    let max_b = 4;
    let want_a = uninterrupted(&prompt_a(), max_a).await;
    let want_b = uninterrupted(&prompt_b(), max_b).await;

    let mut r = rig(10, 3, 4);
    let a = submit(&mut r, &prompt_a(), max_a);
    step_until(&mut r, "A emitted three tokens", 50, |r| emitted(r, a) >= 3).await;
    let emitted_before = emitted(&r, a);
    assert!(!finished(&r, a), "A still running");
    let b = submit(&mut r, &prompt_b(), max_b);
    step_until(&mut r, "A preempted", 50, |r| {
        r.spine.scheduler.metrics().preempted_requests >= 1
    })
    .await;
    assert_eq!(
        phase_of(&r, a),
        Some(SequencePhase::Preempted),
        "A is the preempted sequence"
    );
    step_until(&mut r, "A and B finished", 500, |r| {
        finished(r, a) && finished(r, b)
    })
    .await;

    assert_eq!(toks(&r, b), want_b, "B greedy");
    let got_a = toks(&r, a);
    assert!(
        emitted_before < got_a.len(),
        "A emitted more after resuming"
    );
    assert_eq!(
        got_a.len(),
        max_a,
        "every token of A reached the sink exactly once: {got_a:?}"
    );
    assert_eq!(
        got_a, want_a,
        "A after preemption equals the uninterrupted run"
    );
    let kv = r.spine.kv_manager.read();
    assert_eq!(
        kv.free_block_count(),
        kv.total_block_count(),
        "every block returned at the end"
    );
}
