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

//! VAC M3a: the recall reply says whether the native Omega composition library
//! is linked. Runs in both builds (stub: recall is an explicit refusal naming the stub; linked: true and
//! the pinned 40-hex sha). One test per binary (it sets AIEN_COMPOSE_DIR).
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use std::time::Duration;

#[tokio::test]
async fn recall_reports_whether_compose_is_native() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("AIEN_COMPOSE_DIR", tmp.path().join("compose"));
    // sc#328: the daemon refuses to start without the (default) desk key.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &tmp.path().join("compose"),
    ))
    .unwrap();
    let socket = tmp.path().join("runtime.sock");
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(256, 16));
    let server = AienRuntimeServer::new(spine, &socket);
    let handle = tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
    let c = AienRuntimeClient::new(&socket);
    let mut up = false;
    for _ in 0..200 {
        if c.is_alive().await {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(up, "daemon did not come up");
    match c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(r) => {
            assert_eq!(r.compose_native, aien_omega_compose::LINKED);
            if aien_omega_compose::LINKED {
                assert_eq!(r.omega_sha.len(), 40);
                assert!(r.omega_sha.chars().all(|ch| ch.is_ascii_hexdigit()));
                assert_eq!(r.omega_sha, aien_omega_compose::EXPECTED_OMEGA_SHA);
            } else {
                assert!(r.omega_sha.is_empty());
            }
        }
        // The stub cannot open a compose home at all: recall is an explicit refusal
        // (a verifier reads that as "not native").
        ControlResponse::Error(e) if !aien_omega_compose::LINKED => {
            assert!(e.contains("not linked (stub build)"), "{e}");
        }
        other => panic!("recall: {other:?}"),
    }
    let _ = c.send_command(ControlCommand::Shutdown).await;
    let _ = handle.await;
}
