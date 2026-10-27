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

//! Integration tests for AIEN CLI and runtime IPC protocol.

use aien_inference_abi::MockInferenceBackend;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::LaunchSwarmReq;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::{create_shared_kv_manager, SchedulerConfig};
use std::time::Duration;
use tempfile::TempDir;

#[tokio::test]
async fn test_cli_runtime_ipc_lifecycle() {
    let temp_dir = TempDir::new().expect("Failed to create tempdir");
    let socket_path = temp_dir.path().join("aien-test.sock");
    std::env::set_var("AIEN_RUNTIME_SOCK", socket_path.to_str().unwrap());
    // Keep the idempotency log out of the shared /tmp default used by a live runtime.
    std::env::set_var("AIEN_RUNTIME_STATE_DIR", temp_dir.path());
    // sc#328: the daemon refuses to start without the (default) desk key.
    std::env::set_var("AIEN_COMPOSE_DIR", temp_dir.path().join("compose"));
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &temp_dir.path().join("compose"),
    ))
    .unwrap();

    let kv_manager = create_shared_kv_manager(512, 16);
    let sched_cfg = SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 8,
    };

    let spine = AienRuntimeSpine::new(256, sched_cfg, kv_manager);
    let server = AienRuntimeServer::new(spine, &socket_path);

    let server_handle = tokio::spawn(async move {
        let backend = MockInferenceBackend::new(1);
        server.run(backend).await
    });

    let client = AienRuntimeClient::default_client();
    let mut ready = false;
    for _ in 0..50 {
        if client.is_alive().await {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        ready,
        "Runtime server must become alive on designated socket"
    );

    // 1. Check status
    let status = client
        .get_status()
        .await
        .expect("Status query must succeed");
    assert_eq!(status.active_sequences, 0);

    // 2. Launch swarm
    let prompt: Vec<u32> = (1..=16).collect();
    let req = LaunchSwarmReq {
        model_handle: 1,
        branch_count: 4,
        max_active_sequences: 8,
        max_tokens_per_branch: 8,
        root_world_id: 0,
        priority: 1,
        prompt_tokens: prompt,
    };

    let swarm_id = client.launch_swarm(req).await.expect("Launch must succeed");
    assert_eq!(swarm_id, 1);

    // 3. Inspect swarm
    let swarm_status = client
        .inspect_swarm(swarm_id)
        .await
        .expect("Inspect must succeed");
    // The mock backend may finish all branches before this request lands, so
    // assert the swarm-scoped invariant rather than a still-running count.
    assert!(
        swarm_status.active_sequences <= 4,
        "at most the 4 launched branches"
    );
    assert_eq!(
        swarm_status.active_swarms == 1,
        swarm_status.active_sequences > 0,
        "swarm is active exactly while it has running branches"
    );

    // 4. Clean shutdown
    client.shutdown().await.expect("Shutdown must succeed");
    server_handle
        .await
        .expect("Server task panicked")
        .expect("Server exited with error");
    assert_eq!((), ());
    assert!(!socket_path.exists(), "Socket must be removed on shutdown");
}
