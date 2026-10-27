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

//! Integration test: InspectSwarm reports the requested swarm, not runtime-wide totals.

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::control::{
    ControlCommand, ControlEnvelope, ControlResponse, LaunchSwarmReq, RuntimeStatusReport,
};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;

fn envelope(operation_id: u64, command: ControlCommand) -> ControlEnvelope {
    ControlEnvelope {
        protocol_version: 1,
        request_id: operation_id,
        operation_id: operation_id.into(),
        operator_session: 1,
        command,
    }
}

fn launch(spine: &mut AienRuntimeSpine, operation_id: u64, branch_count: usize) -> u64 {
    let req = LaunchSwarmReq {
        model_handle: 1,
        branch_count,
        max_active_sequences: 8,
        max_tokens_per_branch: 4,
        root_world_id: 0,
        priority: 1,
        prompt_tokens: vec![1, 2, 3],
    };
    match spine.handle_control_command(envelope(operation_id, ControlCommand::LaunchSwarm(req))) {
        ControlResponse::SwarmAccepted { swarm_id, .. } => swarm_id,
        other => panic!("launch failed: {:?}", other),
    }
}

fn inspect(spine: &mut AienRuntimeSpine, operation_id: u64, swarm_id: u64) -> ControlResponse {
    spine.handle_control_command(envelope(
        operation_id,
        ControlCommand::InspectSwarm(swarm_id),
    ))
}

fn status(resp: ControlResponse) -> RuntimeStatusReport {
    match resp {
        ControlResponse::Status(report) => report,
        other => panic!("expected Status, got {:?}", other),
    }
}

#[tokio::test]
async fn inspect_swarm_is_scoped_to_the_requested_swarm() {
    let state_dir = std::env::temp_dir().join(format!("aien-swarm-inspect-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&state_dir);
    std::env::set_var("AIEN_RUNTIME_STATE_DIR", &state_dir);

    let scheduler_config = SchedulerConfig {
        max_batch_size: 64,
        max_batch_tokens: 4096,
        prefill_chunk_size: 64,
        watermark_blocks: 16,
        chunk_prefill: true,
        max_prefill_tokens: 4096,
    };
    let mut spine =
        AienRuntimeSpine::new(1024, scheduler_config, create_shared_kv_manager(1024, 16));

    let small = launch(&mut spine, 1, 2);
    let large = launch(&mut spine, 2, 5);

    let small_status = status(inspect(&mut spine, 3, small));
    let large_status = status(inspect(&mut spine, 4, large));
    assert_eq!(
        small_status.active_sequences, 2,
        "only this swarm's branches"
    );
    assert_eq!(
        large_status.active_sequences, 5,
        "only this swarm's branches"
    );
    assert_eq!(small_status.active_swarms, 1);
    assert_eq!(small_status.active_worlds, 2, "one live world per branch");
    assert!(
        spine.status_report().active_sequences > small_status.active_sequences,
        "runtime-wide count must exceed a single swarm's count"
    );

    let mut backend = MockInferenceBackend::new(1);
    spine
        .run_until_complete(&mut backend, 200)
        .await
        .expect("run_until_complete");

    let done = status(inspect(&mut spine, 5, small));
    assert_eq!(
        done.active_sequences, 0,
        "completed swarm has no active branches"
    );
    assert_eq!(done.active_swarms, 0);
    assert_eq!(
        done.active_worlds, 0,
        "branch worlds are dropped on completion"
    );

    assert!(matches!(
        inspect(&mut spine, 6, 999),
        ControlResponse::Error(_)
    ));
}
