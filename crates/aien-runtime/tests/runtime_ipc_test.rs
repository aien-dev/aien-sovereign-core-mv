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

//! Integration test for AienRuntimeServer and AienRuntimeClient over local UNIX domain socket.

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::LaunchSwarmReq;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use std::time::Duration;
use tempfile::TempDir;

#[tokio::test]
async fn test_runtime_server_client_ipc_lifecycle() {
    let temp_dir = TempDir::new().expect("Failed to create tempdir");
    let socket_path = temp_dir.path().join("runtime.sock");
    // sc#328: the daemon refuses to start without the (default) desk key.
    std::env::set_var("AIEN_COMPOSE_DIR", temp_dir.path().join("compose"));
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &temp_dir.path().join("compose"),
    ))
    .unwrap();

    let kv_manager = create_shared_kv_manager(1024, 16);
    let scheduler_config = SchedulerConfig {
        max_batch_size: 64,
        max_batch_tokens: 4096,
        prefill_chunk_size: 64,
        watermark_blocks: 16,
        chunk_prefill: true,
        max_prefill_tokens: 4096,
    };

    let spine = AienRuntimeSpine::new(1024, scheduler_config, kv_manager);
    let server = AienRuntimeServer::new(spine, &socket_path);

    // Run server in background task
    let server_handle = tokio::spawn(async move {
        let backend = MockInferenceBackend::new(1);
        server.run(backend).await
    });

    // Wait for server to bind socket
    let client = AienRuntimeClient::new(&socket_path);
    let mut alive = false;
    for _ in 0..50 {
        if client.is_alive().await {
            alive = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(alive, "Runtime server failed to bind and become alive");

    // 1. Initial status query
    let initial_status = client.get_status().await.expect("Failed to get status");
    assert_eq!(initial_status.active_sequences, 0);
    assert_eq!(initial_status.active_swarms, 0);

    // 2. Launch swarm over socket IPC
    let prompt_tokens: Vec<u32> = (1..=32).collect();
    let launch_req = LaunchSwarmReq {
        model_handle: 1,
        branch_count: 8,
        max_active_sequences: 8,
        max_tokens_per_branch: 8,
        root_world_id: 0,
        priority: 1,
        prompt_tokens,
    };

    let swarm_id = client
        .launch_swarm(launch_req)
        .await
        .expect("Failed to launch swarm over IPC");
    assert_eq!(swarm_id, 1);

    // 3. Allow background engine worker to execute branch steps
    let mut completed = false;
    for _ in 0..100 {
        let status = client.get_status().await.expect("Failed to query status");
        // Root sequence remains as context anchor, child branches finish
        if status.active_sequences <= 1 {
            completed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(completed, "Swarm branches did not complete within budget");

    // 4. Shutdown server cleanly over IPC
    client
        .shutdown()
        .await
        .expect("Failed to send shutdown command");

    server_handle
        .await
        .expect("Server task panicked")
        .expect("Server exited with error");
    assert_eq!((), ());

    // Verify socket file was cleaned up on clean exit
    assert!(
        !socket_path.exists(),
        "Socket file should be unlinked after shutdown"
    );
}
