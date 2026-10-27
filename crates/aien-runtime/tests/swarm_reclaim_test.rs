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

//! PREFILL-E2E-0 cut C5: all sequences, worlds and KV reclaimed (bullet 12).
//!
//! Drives the real NativeTransformerBackend (reference weights, CPU) through
//! spine + scheduler + one pooled KV shared by both
//! (`aien_runtime::shared_kv::build_shared_kv_runtime`):
//! - natural completion: one root + 4 branches run to completion; afterwards
//!   the backend holds no per-sequence state, the shared KV has no allocated
//!   block, the spine arena has no active sequence and every branch world is
//!   gone (only the root world remains);
//! - cancel midway: the same swarm is cancelled after its first decode step;
//!   the same end state must hold once the spine released the cancelled
//!   sequences' backend state.
//!
//! Mutant (outside the repo, ~/workspace/hive/PE2E-C5/mutant.patch): the
//! spine no longer calls `release_sequence` (finish path and cancel drain).
//! Then the backend keeps the root and branch entries and both tests fail
//! on the backend release check with the message `RECLAIM_BACKEND_LEAK`
//! (the forge greps the mutant output for it, once per test).
//!
//! State isolation: `AienRuntimeSpine::new` builds a `RuntimeController`
//! (spine.rs:55) that loads and persists processed operation ids at
//! `$AIEN_RUNTIME_STATE_DIR/processed_operations.json`, falling back to the
//! shared `/tmp/aien-runtime-processed-ops.json` (control.rs:102-108,
//! 119-127, 134-144). The cancel test sends a fixed operation id, so every
//! run gets its own fresh state directory (as swarm_completion_test.rs:12-14
//! does), else a rerun is refused with "Operation N already processed".
//!
//! Known gap, not covered here: `CancelSwarm` does not remove the cancelled
//! branches from the scheduler's queues (no scheduler cancel API exists), so
//! the cancel test does not step the spine again after the cancel.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use aien_inference_abi::{
    ModelConfig, NativeTransformerBackend, ReferenceCpuBackend, TransformerWeights,
};
use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::swarm::{SwarmConfig, SwarmRecord};
use aien_scheduler::SchedulerConfig;

const BRANCHES: usize = 4;
const PROMPT: [u32; 6] = [5, 17, 42, 3, 88, 61];

fn c5_config() -> ModelConfig {
    ModelConfig {
        model_id: "pe2e-c5-reclaim-reference".to_string(),
        max_sequence_length: 256,
        // 6-token prompt over 4-token blocks: the shared tail block is
        // partial, so each branch's first decode copies it (COW) and owns a
        // private block that must be reclaimed too.
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

fn c5_scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 2,
    }
}

fn c5_swarm(max_tokens: usize) -> SwarmConfig {
    SwarmConfig {
        model_handle: 1,
        branch_count: BRANCHES,
        max_active_sequences: BRANCHES,
        max_tokens_per_branch: max_tokens,
        root_world_id: 0,
        priority: 1,
    }
}

/// One fresh, empty controller state directory per test process (pid +
/// nanosecond clock, so neither a rerun nor a reused pid sees old ids). Set
/// once: both tests run as threads of one process and the env var is
/// process-wide, so per-test values would race.
fn fresh_state_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "aien-swarm-reclaim-test-{}-{}",
            std::process::id(),
            nanos
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create fresh state dir");
        std::env::set_var("AIEN_RUNTIME_STATE_DIR", &dir);
        dir
    })
}

fn build() -> (AienRuntimeSpine, NativeTransformerBackend) {
    // Before the spine (and its RuntimeController) is built.
    fresh_state_dir();
    let weights = TransformerWeights::reference_test_weights(&c5_config());
    build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        c5_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime")
}

fn launch(spine: &mut AienRuntimeSpine, max_tokens: usize) -> SwarmRecord {
    let swarm_id = spine
        .launch_swarm(c5_swarm(max_tokens), &PROMPT)
        .expect("launch swarm");
    spine.swarm_manager.get_swarm(swarm_id).unwrap().clone()
}

/// Bullet 12 end state, shared by both tests.
fn assert_fully_reclaimed(
    spine: &AienRuntimeSpine,
    backend: &NativeTransformerBackend,
    swarm: &SwarmRecord,
    worlds_before_launch: usize,
) {
    let leaked: Vec<u64> = backend.sequences.keys().copied().collect();
    assert!(
        leaked.is_empty(),
        "RECLAIM_VIOLATION: RECLAIM_BACKEND_LEAK backend still holds per-sequence state for {:?} (root {}, branches {:?})",
        leaked,
        swarm.root_sequence_id.as_u64(),
        swarm
            .branch_sequences
            .iter()
            .map(|b| b.as_u64())
            .collect::<Vec<_>>()
    );
    // C6 x C5: the sampling map entry is dropped with the sequence.
    assert!(
        backend.sampling_params(swarm.root_sequence_id.as_u64()).is_none()
            && swarm
                .branch_sequences
                .iter()
                .all(|b| backend.sampling_params(b.as_u64()).is_none()),
        "RECLAIM_VIOLATION: RECLAIM_SAMPLING_LEAK backend still holds sampling params for a released sequence"
    );
    let kv = spine.kv_manager.read();
    assert_eq!(
        kv.allocated_block_count(),
        0,
        "RECLAIM_VIOLATION: shared KV still has allocated blocks"
    );
    assert!(kv
        .get_block_table(swarm.root_sequence_id.as_u64())
        .is_none());
    for b in &swarm.branch_sequences {
        assert!(kv.get_block_table(b.as_u64()).is_none());
    }
    drop(kv);
    assert_eq!(
        spine.arena.active_count(),
        0,
        "RECLAIM_VIOLATION: spine arena still has active sequences"
    );
    for &w in &swarm.branch_worlds {
        assert!(
            spine.world_store.get_world(w).is_none(),
            "RECLAIM_VIOLATION: branch world {} still alive",
            w
        );
    }
    assert_eq!(
        spine.world_store.active_world_count(),
        worlds_before_launch + 1,
        "only the root world may remain"
    );
    assert_eq!(spine.swarm_manager.active_swarm_count(), 0);
}

#[tokio::test]
async fn swarm_completion_reclaims_backend_kv_arena_and_worlds() {
    let (mut spine, mut backend) = build();
    let worlds_before = spine.world_store.active_world_count();
    let swarm = launch(&mut spine, 3);
    assert_eq!(swarm.branch_sequences.len(), BRANCHES);

    let mut steps = 0;
    let mut peak_backend_sequences = 0usize;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0) && steps < 50
    {
        steps += 1;
        spine.step(&mut backend).await.expect("spine step");
        peak_backend_sequences = peak_backend_sequences.max(backend.sequences.len());
    }
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        BRANCHES,
        "every branch must finish"
    );
    // Counterexample guard: the run really created backend state for the
    // root and every branch, so an empty map at the end means reclaim.
    assert_eq!(
        peak_backend_sequences,
        BRANCHES + 1,
        "root + every branch must have held backend state during the run"
    );

    assert_fully_reclaimed(&spine, &backend, &swarm, worlds_before);
}

#[tokio::test]
async fn swarm_cancel_midway_reclaims_backend_kv_arena_and_worlds() {
    let (mut spine, mut backend) = build();
    let worlds_before = spine.world_store.active_world_count();
    // Long enough that no branch finishes before the cancel.
    let swarm = launch(&mut spine, 16);

    // Step 1: root prefill + fence + backend forks. Step 2: first decode.
    spine.step(&mut backend).await.expect("prefill step");
    spine.step(&mut backend).await.expect("first decode step");
    assert_eq!(spine.scheduler.metrics().finished_requests, 0);
    assert_eq!(
        backend.sequences.len(),
        BRANCHES + 1,
        "root + every branch must hold backend state before the cancel"
    );
    assert!(spine.kv_manager.read().allocated_block_count() > 0);

    let resp = spine.handle_control_command(ControlEnvelope {
        protocol_version: 1,
        request_id: 1,
        operation_id: 0xC5,
        operator_session: 1,
        command: ControlCommand::CancelSwarm(swarm.id),
    });
    assert!(
        matches!(resp, ControlResponse::SwarmCancelled { swarm_id } if swarm_id == swarm.id),
        "cancel must succeed: {:?}",
        resp
    );

    let released = spine
        .release_pending_backend_sequences(&mut backend)
        .expect("release cancelled backend state");
    assert_eq!(released, BRANCHES + 1, "root + every branch released");

    assert_fully_reclaimed(&spine, &backend, &swarm, worlds_before);
}
