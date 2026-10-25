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

//! sc#328 (Drake decision b, 2026-10-08): the server's own startup check. A
//! compose bridge that requires the approval desk (the default) refuses to
//! start without a loadable desk key, before the socket exists; with the key
//! it serves. Covers both bridge sources: a preset bridge and the one the
//! daemon builds from AIEN_COMPOSE_DIR. A preset bridge built with the desk
//! off is the dev-only opt-out: a strict run (release qualification) refuses
//! it, a dev run serves. No composition library is needed (the
//! check runs before any compose work), so this runs in every build.
//!
//! One test: AIEN_COMPOSE_DIR is process-wide.
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, AUTHORIZE_DESK_ENV};
use aien_runtime::control::ControlCommand;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn server(socket: &Path, preset: Option<&Path>, desk: bool) -> AienRuntimeServer {
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(256, 16));
    let s = AienRuntimeServer::new(spine, socket);
    match preset {
        Some(dir) => {
            let never: ComposeProposer =
                Arc::new(|_: &str, _: Duration| Err("no proposer in this test".to_string()));
            s.with_compose_bridge(Arc::new(
                ComposeBridge::new(dir.to_path_buf(), never, "test:none")
                    .with_authorize_requires_desk(desk),
            ))
        }
        None => s,
    }
}

/// `run` must return the refusal on its own, without binding the socket.
async fn refused(socket: &Path, preset: Option<&Path>, compose: &Path) {
    let s = server(socket, preset, true);
    let e = tokio::time::timeout(Duration::from_secs(30), s.run(MockInferenceBackend::new(1)))
        .await
        .expect("the daemon did not refuse within 30 s")
        .expect_err("the daemon started without the desk key");
    assert!(e.contains("desk-key --create"), "{e}");
    assert!(
        e.contains(&desk_key_path(compose).display().to_string()),
        "{e}"
    );
    assert!(!socket.exists(), "socket bound before the refusal");
    assert!(
        !desk_key_path(compose).exists(),
        "startup must never mint a key"
    );
}

/// With the key the daemon serves; then it is shut down cleanly.
async fn serves(socket: &Path, preset: Option<&Path>, desk: bool) {
    let s = server(socket, preset, desk);
    let h = tokio::spawn(async move { s.run(MockInferenceBackend::new(1)).await });
    let c = AienRuntimeClient::new(socket);
    let mut alive = false;
    for _ in 0..250 {
        if c.is_alive().await {
            alive = true;
            break;
        }
        if h.is_finished() {
            panic!("daemon stopped: {:?}", h.await);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(alive, "daemon did not come up");
    let _ = c.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), h).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bridge_that_requires_the_desk_refuses_to_start_without_its_key() {
    let tmp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    std::env::remove_var(AUTHORIZE_DESK_ENV);

    // A preset bridge, desk required by default.
    let pre = root.join("preset");
    refused(&root.join("a.sock"), Some(&pre), &pre).await;
    DeskKey::create(&desk_key_path(&pre)).unwrap();
    serves(&root.join("b.sock"), Some(&pre), true).await;

    // The daemon's own bridge from AIEN_COMPOSE_DIR.
    let env = root.join("env");
    std::env::set_var("AIEN_COMPOSE_DIR", &env);
    refused(&root.join("c.sock"), None, &env).await;
    DeskKey::create(&desk_key_path(&env)).unwrap();
    serves(&root.join("d.sock"), None, true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bridge_with_the_desk_off_is_refused_in_a_strict_run() {
    let tmp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    let off = root.join("off");
    let socket = root.join("e.sock");
    if aien_inference_abi::strict::dev_fallback_active() {
        // A dev run: the opt-out serves (and says so at startup).
        serves(&socket, Some(&off), false).await;
        return;
    }
    let s = server(&socket, Some(&off), false);
    let e = tokio::time::timeout(Duration::from_secs(30), s.run(MockInferenceBackend::new(1)))
        .await
        .expect("the daemon did not refuse within 30 s")
        .expect_err("a strict run served with the desk off");
    assert!(e.contains("release qualification"), "{e}");
    assert!(!socket.exists(), "socket bound before the refusal");
}
