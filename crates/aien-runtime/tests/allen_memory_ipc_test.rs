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

//! ALLEN memory commands over the runtime socket (arch#159): routing only. ALLEN is not engaged here, so status answers "not_engaged" and a
//! change is refused. In a stub omega build the compose home cannot open and
//! the daemon says so (the answer is an error, never a made-up identity).
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_allen::ENV_SUBJECT;
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn allen_memory_commands_route_through_the_socket() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    // Only this test binary sets these; one test per binary avoids env races.
    std::env::set_var("AIEN_COMPOSE_DIR", tmp.path().join("compose"));
    // sc#328: the daemon refuses to start without the (default) desk key.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &tmp.path().join("compose"),
    ))
    .unwrap();
    std::env::remove_var(ENV_SUBJECT);
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
    let linked = aien_omega_compose::LINKED;

    let st = client
        .send_command(ControlCommand::AllenStatus)
        .await
        .unwrap();
    match (st, linked) {
        (ControlResponse::AllenStatusReport(s), true) => {
            assert_eq!(s.identity, "not_engaged");
            assert_eq!(s.model, "model:StreamTurn-path");
            assert_eq!(s.unsupported, vec!["avatar", "voice"]);
        }
        (ControlResponse::Error(e), false) => assert!(e.contains("not linked"), "{e}"),
        (o, l) => panic!("unexpected {o:?} (linked={l})"),
    }
    let set = client
        .send_command(ControlCommand::AllenMemoryPut {
            context: "work".into(),
            kind: "fact".into(),
            text: "a note".into(),
        })
        .await
        .unwrap();
    match (set, linked) {
        (ControlResponse::AllenRefused(r), true) => assert_eq!(r.code, "not_engaged"),
        (ControlResponse::Error(e), false) => assert!(e.contains("not linked"), "{e}"),
        (o, l) => panic!("unexpected {o:?} (linked={l})"),
    }
    assert!(!tmp.path().join("compose.allen-memory").exists());
    let _ = client.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
}
