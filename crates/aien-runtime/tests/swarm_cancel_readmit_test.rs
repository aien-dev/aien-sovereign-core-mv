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

//! PREFILL-E2E CX1: a cancelled swarm's branches are never re-admitted after
//! the spine released their backend state.
//!
//! `CancelSwarm` freed the branches in the spine arena, KV and worlds, but the
//! scheduler keeps its own arena (same ids), so queued branches stayed live and
//! were admitted and decoded after the backend released them. The fix frees the
//! ids in the scheduler arena too, which turns queued copies stale. This test
//! pins that behaviour: launch a swarm with a small batch so some branches queue, cancel it, step
//! the spine several more times, and require
//! - no backend `execute_step` request for a cancelled id after its
//!   `release_sequence`,
//! - empty scheduler waiting and preempted queues,
//! - zero leaks (arena, KV, backend state).
//!   Every failure message carries the marker READMIT_VIOLATION.
//!
//! Mutant (outside the repo, ~/workspace/hive/PE2E-CX1/mutant.patch): the
//! spine no longer frees the cancelled ids in the scheduler arena, so queued
//! branches stay admissible.
//!
//! State isolation: same per-process AIEN_RUNTIME_STATE_DIR as
//! swarm_reclaim_test.rs, because the cancel sends a fixed operation id.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, ModelConfig, ScheduledBatch, StepMetrics,
};
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse};
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::swarm::SwarmConfig;
use aien_scheduler::SchedulerConfig;
use async_trait::async_trait;

fn fresh_state_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "aien-swarm-cancel-readmit-test-{}-{}",
            std::process::id(),
            nanos
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create fresh state dir");
        std::env::set_var("AIEN_RUNTIME_STATE_DIR", &dir);
        dir
    })
}

/// Records releases and flags any later request for a released id.
#[derive(Default)]
struct RecordingBackend {
    released: HashSet<u64>,
    violations: Vec<String>,
    decodes: usize,
}

#[async_trait]
impl AienInferenceBackend for RecordingBackend {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut outputs = Vec::new();
        let ids = batch
            .prefill_requests
            .iter()
            .map(|r| r.request_id)
            .chain(batch.decode_requests.iter().copied());
        for id in ids {
            if self.released.contains(&id) {
                self.violations.push(format!(
                    "execute_step asked to run id {} after its release_sequence",
                    id
                ));
            }
            self.decodes += 1;
            outputs.push(DecodeOutput::Token {
                request_id: id,
                token_id: 101,
                logprob: None,
            });
        }
        let metrics = StepMetrics {
            prefill_tokens_processed: batch
                .prefill_requests
                .iter()
                .map(|r| r.prompt_tokens.len())
                .sum(),
            decode_tokens_emitted: outputs.len(),
            step_latency_us: 0,
            active_kv_blocks: batch.block_tables.values().map(|v| v.len()).sum(),
        };
        Ok((outputs, metrics))
    }

    fn release_sequence(&mut self, seq_id: u64) -> Result<(), String> {
        self.released.insert(seq_id);
        Ok(())
    }
}

#[tokio::test]
async fn cancelled_swarm_branches_are_never_readmitted() {
    fresh_state_dir();
    let kv_manager = create_shared_kv_manager(256, 16);
    // Small batch: only 2 of the 4 branches fit, the others stay queued.
    let sched_cfg = SchedulerConfig {
        max_batch_size: 2,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 16,
        chunk_prefill: true,
        watermark_blocks: 4,
    };
    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let mut backend = RecordingBackend::default();

    let prompt: Vec<u32> = (500..540).collect();
    let config = SwarmConfig {
        model_handle: 1,
        branch_count: 4,
        max_active_sequences: 4,
        max_tokens_per_branch: 64,
        root_world_id: 0,
        priority: 1,
    };
    let swarm_id = spine.launch_swarm(config, &prompt).expect("launch");

    // Run until some branches decode and some are still queued.
    let mut steps = 0;
    while (backend.decodes == 0 || spine.scheduler.waiting_count() == 0) && steps < 20 {
        steps += 1;
        spine
            .step(&mut backend)
            .await
            .unwrap_or_else(|e| panic!("READMIT_VIOLATION: pre-cancel step failed: {}", e));
    }
    assert!(
        backend.decodes > 0 && spine.scheduler.waiting_count() > 0,
        "READMIT_VIOLATION: test setup, no queued branch before the cancel \
         (decodes {}, waiting {}, preempted {})",
        backend.decodes,
        spine.scheduler.waiting_count(),
        spine.scheduler.preempted_count()
    );

    let resp = spine.handle_control_command(ControlEnvelope {
        protocol_version: 1,
        request_id: 1,
        operation_id: 0xC1,
        operator_session: 1,
        command: ControlCommand::CancelSwarm(swarm_id),
    });
    assert!(
        matches!(resp, ControlResponse::SwarmCancelled { swarm_id: s } if s == swarm_id),
        "READMIT_VIOLATION: test setup, cancel refused: {:?}",
        resp
    );

    // Step well past the point where the first post-cancel step released
    // the backend state and the scheduler could re-admit a queued branch.
    for i in 0..8 {
        spine.step(&mut backend).await.unwrap_or_else(|e| {
            panic!(
                "READMIT_VIOLATION: step {} after cancel failed: {}",
                i + 1,
                e
            )
        });
        assert!(
            backend.violations.is_empty(),
            "READMIT_VIOLATION: {:?}",
            backend.violations
        );
    }

    assert!(
        !backend.released.is_empty(),
        "READMIT_VIOLATION: the spine never released the cancelled ids"
    );
    assert_eq!(
        spine.scheduler.waiting_count() + spine.scheduler.preempted_count(),
        0,
        "READMIT_VIOLATION: cancelled ids still in scheduler queues (waiting {}, preempted {})",
        spine.scheduler.waiting_count(),
        spine.scheduler.preempted_count()
    );
    assert_eq!(
        spine.scheduler.running_count(),
        0,
        "READMIT_VIOLATION: scheduler still runs a cancelled sequence"
    );
    assert_eq!(
        spine.arena.active_count(),
        0,
        "READMIT_VIOLATION: spine arena still has active sequences"
    );
    assert_eq!(
        spine.kv_manager.read().allocated_block_count(),
        0,
        "READMIT_VIOLATION: shared KV still has allocated blocks"
    );
    assert_eq!(spine.swarm_manager.active_swarm_count(), 0);
}
