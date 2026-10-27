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

//! Swarm and scheduler correlation trace events (#398), end to end on the real spine.
//! Events are observational: they never change scheduling, KV or world handling.

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

use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse};
use aien_runtime::trace_observer::TraceDecisionObserver;
use aien_trace::{reconstruct, EventKind, EventStatus, MemorySink, TraceEvent, TraceSink};

fn count(ev: &[TraceEvent], k: EventKind) -> usize {
    ev.iter().filter(|e| e.kind == k).count()
}

#[tokio::test]
async fn three_branch_swarm_reconstructs_with_valid_parents() {
    let (mut spine, mut backend) = build(3);
    let sink = Arc::new(MemorySink::new());
    spine.swarm_manager.set_trace_sink(sink.clone());
    let swarm_id = spine
        .launch_swarm(swarm_config(3, 3), &PROMPT)
        .expect("launch");
    run_to_completion(&mut spine, &mut backend).await;

    let ev = sink.drain();
    assert_eq!(count(&ev, EventKind::SwarmLaunched), 1);
    assert_eq!(count(&ev, EventKind::BranchForked), 3);
    assert_eq!(count(&ev, EventKind::SequenceFinished), 3);
    assert_eq!(count(&ev, EventKind::SwarmCancelled), 0);

    let tree = reconstruct(&ev).expect("every parent valid, one trace");
    assert_eq!(tree.roots().len(), 1);
    let root = tree.roots()[0];
    assert_eq!(tree.get(root).unwrap().kind, EventKind::SwarmLaunched);
    assert_eq!(tree.get(root).unwrap().ids.swarm_id, Some(swarm_id));
    assert_eq!(tree.children(root).len(), 6);
    let record = spine.swarm_manager.get_swarm(swarm_id).unwrap();
    assert_eq!(record.trace_root_event, Some(root));
    assert_eq!(spine.swarm_manager.trace_id(swarm_id), Some(ev[0].trace_id));
    // Each branch id appears in a fork and in a finish event.
    for b in &record.branch_sequences {
        let id = b.as_u64();
        for kind in [EventKind::BranchForked, EventKind::SequenceFinished] {
            assert!(ev
                .iter()
                .any(|e| e.kind == kind && e.ids.sequence_id == Some(id)));
        }
    }
    let a: Vec<u64> = tree.chronological().iter().map(|e| e.event_id).collect();
    let b: Vec<u64> = tree.chronological().iter().map(|e| e.event_id).collect();
    assert_eq!(a, b, "chronological order is stable");
    assert_eq!(a.len(), ev.len());
    let keys: Vec<(u64, u64)> = tree
        .chronological()
        .iter()
        .map(|e| (e.at_unix_ms, e.event_id))
        .collect();
    assert!(keys.windows(2).all(|w| w[0] <= w[1]));
}

#[tokio::test]
async fn cancelled_swarm_emits_swarm_cancelled() {
    let (mut spine, mut backend) = build(3);
    let sink = Arc::new(MemorySink::new());
    spine.swarm_manager.set_trace_sink(sink.clone());
    let swarm_id = spine
        .launch_swarm(swarm_config(3, 16), &PROMPT)
        .expect("launch");
    spine.step(&mut backend).await.expect("prefill step");
    spine.step(&mut backend).await.expect("first decode step");

    let resp = spine.handle_control_command(ControlEnvelope {
        protocol_version: 1,
        request_id: 1,
        operation_id: 0x398,
        operator_session: 1,
        command: ControlCommand::CancelSwarm(swarm_id),
    });
    assert!(
        matches!(resp, ControlResponse::SwarmCancelled { .. }),
        "{resp:?}"
    );

    let ev = sink.drain();
    let cancelled: Vec<&TraceEvent> = ev
        .iter()
        .filter(|e| e.kind == EventKind::SwarmCancelled)
        .collect();
    assert_eq!(cancelled.len(), 1);
    assert_eq!(cancelled[0].status, EventStatus::Cancelled);
    assert_eq!(cancelled[0].ids.swarm_id, Some(swarm_id));
    assert!(reconstruct(&ev).is_ok());
}

#[tokio::test]
async fn scheduler_tap_emits_admitted_for_registered_sequences_only() {
    let (mut spine, mut backend) = build(3);
    let sink = Arc::new(MemorySink::new());
    spine.swarm_manager.set_trace_sink(sink.clone());
    let dynsink: Arc<dyn TraceSink> = sink.clone();
    let observer = Arc::new(TraceDecisionObserver::new(dynsink));
    spine
        .scheduler
        .set_decision_observer(Some(observer.clone()));

    let swarm_id = spine
        .launch_swarm(swarm_config(3, 3), &PROMPT)
        .expect("launch");
    let trace = spine.swarm_manager.trace_id(swarm_id).unwrap();
    let record = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let root_event = record.trace_root_event.unwrap();
    // Register only the first branch.
    let watched = record.branch_sequences[0].as_u64();
    observer.register(watched, trace, root_event);
    run_to_completion(&mut spine, &mut backend).await;

    let ev = sink.drain();
    let admitted: Vec<&TraceEvent> = ev
        .iter()
        .filter(|e| e.kind == EventKind::SequenceAdmitted)
        .collect();
    assert!(
        !admitted.is_empty(),
        "the registered branch is admitted at least once"
    );
    assert!(admitted.iter().all(|e| e.ids.sequence_id == Some(watched)));
    assert!(admitted
        .iter()
        .all(|e| e.parent_event_id == Some(root_event)));
    assert!(admitted.iter().all(|e| e.ids.step_id.is_some()));
    assert!(reconstruct(&ev).is_ok());
}
