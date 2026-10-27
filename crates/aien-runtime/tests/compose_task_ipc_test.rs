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

//! NEXT-PHASE-1 cut 1b: RunComposeTask over the runtime socket. No tokenizer
//! is loaded, so the real "model" Skill (the StreamTurn inference path)
//! fails and nothing commits: this proves the routing, the blocking hand-off
//! and the result record, not the model.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_compose_task_over_socket() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    // Only this test binary sets it; one test per binary avoids env races.
    std::env::set_var("AIEN_COMPOSE_DIR", tmp.path().join("compose"));
    // sc#328: the daemon refuses to start without the (default) desk key.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &tmp.path().join("compose"),
    ))
    .unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
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
    let client = AienRuntimeClient::new(&socket);
    let mut alive = false;
    for _ in 0..100 {
        if client.is_alive().await {
            alive = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(alive);

    let resp = client
        .send_command(ControlCommand::RunComposeTask {
            context: None,
            goal: "propose one change".into(),
            workspace: ws.to_str().unwrap().into(),
        })
        .await
        .unwrap();
    match resp {
        ControlResponse::ComposeTaskResult(r) => {
            if !aien_omega_compose::LINKED {
                panic!("stub build returned a compose result");
            }
            assert!(
                !r.committed,
                "no tokenizer: the model Skill must fail, {r:?}"
            );
            assert_eq!(r.proposal, None);
            assert_eq!(r.proposer, "model:StreamTurn-path");
            assert_eq!(r.branch_count, 1);
        }
        ControlResponse::Error(e) => {
            if aien_omega_compose::LINKED {
                panic!("linked build returned {e}");
            }
            assert!(e.contains("not linked"), "{e}");
        }
        other => panic!("unexpected {other:?}"),
    }
    // cut 2: a host note and its recall over the same socket.
    let noted = client
        .send_command(ControlCommand::ComposeNote {
            kind: "constraint".into(),
            text: "only the workspace".into(),
            links: vec![],
        })
        .await
        .unwrap();
    match (noted, aien_omega_compose::LINKED) {
        (ControlResponse::ComposeNoted(n), true) => {
            let back = client
                .send_command(ControlCommand::ComposeRecall {
                    ids: vec![n.id],
                    prefix: Some(n.id),
                })
                .await
                .unwrap();
            match back {
                ControlResponse::ComposeRecalled(r) => {
                    assert_eq!(r.machine_id, n.machine_id);
                    assert_eq!(r.cited.len(), 1);
                    assert_eq!(r.cited[0].digest, n.digest);
                    assert_eq!(r.cited[0].text.as_deref(), Some("only the workspace"));
                    assert!(r.prefix_digest.is_some());
                }
                other => panic!("recall: {other:?}"),
            }
        }
        (ControlResponse::Error(e), false) => assert!(e.contains("not linked"), "{e}"),
        (other, linked) => panic!("note (linked {linked}): {other:?}"),
    }
    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
}
