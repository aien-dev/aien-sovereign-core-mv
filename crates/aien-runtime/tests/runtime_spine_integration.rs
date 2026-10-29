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

//! Comprehensive integration tests for the canonical AienRuntimeSpine.

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::{create_shared_kv_manager, KvDType, KvPoolConfig};
use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse, LaunchSwarmReq};
use aien_runtime::sequence::SequenceArena;
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::world::{EffectIntent, WorldDraft, WorldStore};
use aien_scheduler::SchedulerConfig;

#[test]
fn test_sequence_arena_generational_invalidation() {
    let mut arena = SequenceArena::new(16);
    let id1 = arena.allocate(1, 1, 10, 100).unwrap();
    assert_eq!(id1.slot, 0);
    assert_eq!(id1.generation, 1);
    assert!(arena.validate(id1));

    // Free the sequence -> increments slot generation
    assert!(arena.free(id1));
    assert!(!arena.validate(id1), "Old SequenceId must be invalid");

    // Re-allocate the slot -> receives generation 2
    let id2 = arena.allocate(1, 1, 10, 200).unwrap();
    assert_eq!(id2.slot, id1.slot);
    assert_eq!(id2.generation, 2);

    // Old SequenceId still rejected
    assert!(!arena.validate(id1));
    assert!(arena.validate(id2));
}

#[test]
fn test_context_composition_and_missing_prefill() {
    let mut composer = aien_runtime::context::ContextComposer::new();
    let sys_tokens = vec![1, 2, 3, 4];
    let user_tokens = vec![10, 20, 30, 40, 50];
    let root_rev = composer.create_root_revision(sys_tokens, user_tokens);

    assert_eq!(composer.get_revision(root_rev).unwrap().total_tokens, 9);

    // Sequence with prefill_cursor = 4 needs 5 tokens
    let span = composer.compute_missing_prefill(root_rev, 4).unwrap();
    assert_eq!(span.start, 4);
    assert_eq!(span.len, 5);
    assert_eq!(span.tokens, vec![10, 20, 30, 40, 50]);

    // Append memory
    let mem_tokens = vec![100, 101, 102];
    let rev2 = composer.append_memory(root_rev, 42, mem_tokens).unwrap();
    assert_eq!(composer.get_revision(rev2).unwrap().total_tokens, 12);

    let span2 = composer.compute_missing_prefill(rev2, 9).unwrap();
    assert_eq!(span2.start, 9);
    assert_eq!(span2.len, 3);
    assert_eq!(span2.tokens, vec![100, 101, 102]);
}

#[test]
fn test_world_forking_and_effect_staging() {
    let mut store = WorldStore::new();
    let root_world = store.create_root_world(1, 1, 10, 100);

    // Fork child world: constant-time root sharing
    let child_world = store.fork_world(root_world, 200).unwrap();
    let child_man = store.get_world(child_world).unwrap();
    assert_eq!(child_man.parent, Some(root_world));
    assert_eq!(child_man.effect_log.len(), 0);

    // Stage mutations in draft
    let mut draft = WorldDraft::new(child_world);
    draft.stage_object("report.md".to_string(), b"# Security Audit".to_vec());
    draft.record_effect(EffectIntent::WriteFile {
        path: "report.md".to_string(),
        content_hash: [1u8; 32],
    });

    let committed_world = store.commit_draft(draft, 300).unwrap();
    let committed_man = store.get_world(committed_world).unwrap();
    assert_eq!(committed_man.parent, Some(child_world));
    assert_eq!(committed_man.effect_log.len(), 1);

    // Parent world remains unmodified
    let root_man = store.get_world(root_world).unwrap();
    assert_eq!(root_man.effect_log.len(), 0);
}

#[tokio::test]
async fn test_swarm_launch_and_step_execution() {
    let pool_cfg = KvPoolConfig::for_tinyllama(256, 16, KvDType::Bf16);
    let kv_manager = create_shared_kv_manager(256, 16);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 64,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(128, sched_cfg, kv_manager.clone());
    let mut backend = MockInferenceBackend::new(1);

    let prompt: Vec<u32> = (0..32).collect();

    let root_world = spine.world_store.create_root_world(1, 1, 10, 100);

    // Launch swarm of 16 branches via ControlEnvelope
    let launch_req = LaunchSwarmReq {
        model_handle: 1,
        branch_count: 16,
        max_active_sequences: 32,
        max_tokens_per_branch: 4,
        root_world_id: root_world,
        priority: 5,
        prompt_tokens: prompt.clone(),
    };

    // Unique per run so persistent idempotency from earlier runs cannot
    // pollute this test. The replay check below reuses the same ID in run.
    let operation_id: u128 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let env = ControlEnvelope {
        protocol_version: 1,
        request_id: 1,
        operation_id,
        operator_session: 1,
        command: ControlCommand::LaunchSwarm(launch_req),
    };

    let resp = spine.handle_control_command(env.clone());
    match resp {
        ControlResponse::SwarmAccepted {
            swarm_id,
            operation_id: accepted_id,
        } => {
            assert_eq!(swarm_id, 1);
            assert_eq!(accepted_id, operation_id);
        }
        _ => panic!("Expected SwarmAccepted response"),
    }

    // Replay of same operation_id must be rejected (idempotency)
    let replay_resp = spine.handle_control_command(env);
    match replay_resp {
        ControlResponse::Error(msg) => {
            assert!(msg.contains("already processed"));
        }
        _ => panic!("Expected Error response on operation replay"),
    }

    // Verify 16 child branches + 1 root sequence allocated in arena
    assert_eq!(spine.arena.active_count(), 17);

    // Advance engine steps until all scheduled branch sequences finish
    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0) && steps < 50
    {
        steps += 1;
        let _ = spine.step(&mut backend).await.unwrap();
    }

    // All 16 branches completed; natural completion reclaims the root anchor too
    assert_eq!(
        spine.arena.active_count(),
        0,
        "Swarm completion reclaims the root anchor; zero sequences remain"
    );
    let status = spine.status_report();
    assert_eq!(status.active_sequences, 0);
}
