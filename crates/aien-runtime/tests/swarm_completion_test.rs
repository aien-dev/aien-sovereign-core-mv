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

//! Integration test: swarm lifecycle through the spine with a mock backend.
//! Natural completion must reclaim the root sequence and all branch worlds.

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::control::LaunchSwarmReq;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;

#[tokio::test]
async fn swarm_natural_completion_reclaims_everything() {
    let state_dir = std::env::temp_dir().join(format!("aien-swarm-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&state_dir);
    std::env::set_var("AIEN_RUNTIME_STATE_DIR", &state_dir);

    let kv_manager = create_shared_kv_manager(1024, 16);
    let scheduler_config = SchedulerConfig {
        max_batch_size: 64,
        max_batch_tokens: 4096,
        prefill_chunk_size: 64,
        watermark_blocks: 16,
        chunk_prefill: true,
        max_prefill_tokens: 4096,
    };
    let mut spine = AienRuntimeSpine::new(1024, scheduler_config, kv_manager);

    let launch = LaunchSwarmReq {
        model_handle: 1,
        branch_count: 4,
        max_active_sequences: 8,
        max_tokens_per_branch: 4,
        root_world_id: 0,
        priority: 1,
        prompt_tokens: vec![1, 2, 3],
    };
    let swarm_id = spine.handle_control_command(aien_runtime::control::ControlEnvelope {
        protocol_version: 1,
        request_id: 1,
        operation_id: 100,
        operator_session: 1,
        command: aien_runtime::control::ControlCommand::LaunchSwarm(launch),
    });
    println!("launch response: {:?}", swarm_id);

    let mut backend = MockInferenceBackend::new(1);
    let metrics = spine
        .run_until_complete(&mut backend, 200)
        .await
        .expect("run_until_complete");
    println!("steps executed: {}", metrics.len());

    let status = spine.status_report();
    println!(
        "final: active_sequences={} active_swarms={} active_worlds={} cow_faults={}",
        status.active_sequences, status.active_swarms, status.active_worlds, status.cow_faults
    );

    assert_eq!(
        status.active_sequences, 0,
        "root sequence must be reclaimed after branches finish"
    );
    assert_eq!(status.active_swarms, 0, "swarm must reach Completed");
    assert_eq!(
        status.active_worlds, 1,
        "branch worlds must drop; only the swarm root world remains"
    );
}
