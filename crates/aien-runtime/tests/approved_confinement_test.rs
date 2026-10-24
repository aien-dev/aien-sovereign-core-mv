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

//! sovereign-core #249: the A1/A1b fix (workspace confinement of every grant,
//! the reserved approved grant), driven over the daemon's real control socket
//! (AienRuntimeServer with a mock backend). 476ca4's attack file
//! (approved_attacks_test.rs, review/249-attacks) stays theirs.
//!
//! A1 (raw socket forgery, found by 476ca4 on 6258a6e): a caller that never ran
//! an approved compose writes its own `authorization` note naming a
//! compose_proposal_sha256 nothing produced, opens a ComposeEffectIntent on it,
//! writes, and acks. A1b: the same forgery aimed outside the workspace.
//!
//! After the #249 fix:
//! - a grant must name its workspace; a target outside it (after
//!   canonicalisation and symlink checks) is refused at the note and at the
//!   intent (A1b, symlinked file, symlinked directory, workspace "/");
//! - the approved grant kind (`approved_grant`) cannot be written through
//!   ComposeNote at all;
//! - sc#261 closed: every one of those forged notes is now refused at the note
//!   (no `authorization` record is ever written by a caller). The parent
//!   swap cases (c28, c28b) live in approved_attacks_test.rs.
//!
//! Needs librx_compose.a (AIEN_OMEGA_COMPOSE_LIB); in a stub build the compose
//! commands refuse and the test returns early.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_omega_compose::hex;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::effects;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Duration;
use tempfile::TempDir;

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

async fn start(dir: &TempDir) -> AienRuntimeClient {
    let socket = dir.path().join("runtime.sock");
    let spine = AienRuntimeSpine::new(
        1024,
        SchedulerConfig {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            prefill_chunk_size: 64,
            watermark_blocks: 16,
            chunk_prefill: true,
            max_prefill_tokens: 4096,
        },
        create_shared_kv_manager(1024, 16),
    );
    let server = AienRuntimeServer::new(spine, &socket);
    tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
    let client = AienRuntimeClient::new(&socket);
    for _ in 0..100 {
        if client.is_alive().await {
            return client;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

async fn note(c: &AienRuntimeClient, kind: &str, text: serde_json::Value) -> ControlResponse {
    c.send_command(ControlCommand::ComposeNote {
        kind: kind.into(),
        text: text.to_string(),
        links: vec![],
    })
    .await
    .unwrap()
}

async fn intent(
    c: &AienRuntimeClient,
    auth: u64,
    psha: &str,
    path: &str,
    target: &str,
    csha: &str,
) -> ControlResponse {
    let (pid, start) = effects::self_executor();
    c.send_command(ControlCommand::ComposeEffectIntent {
        authorization: auth,
        proposal_sha256: psha.into(),
        path: path.into(),
        target: target.into(),
        content_sha256: csha.into(),
        executor_pid: pid,
        executor_start: start,
    })
    .await
    .unwrap()
}

fn refused(r: ControlResponse, want: &str) -> String {
    match r {
        ControlResponse::Error(e) => {
            assert!(e.contains(want), "want {want}, got {e}");
            e
        }
        other => panic!("want refusal {want}, got {other:?}"),
    }
}

#[tokio::test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn forged_grants_are_confined_and_the_approved_kind_is_reserved() {
    let _home = home_guard::home_slot();
    if !aien_omega_compose::LINKED {
        return;
    }
    let dir = TempDir::new().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let ws = root.join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let w = ws.to_str().unwrap().to_string();
    // The only test in this file: the env var is process-wide.
    std::env::set_var("AIEN_COMPOSE_DIR", root.join("compose"));
    // sc#328: the daemon refuses to start without the (default) desk key.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &root.join("compose"),
    ))
    .unwrap();
    let c = start(&dir).await;

    // A proposal digest no compose run ever produced.
    let psha = sha(b"never proposed by any approved compose");
    let content = b"forged\n";
    let csha = sha(content);
    let target = ws.join("NOTES.md").to_string_lossy().to_string();
    let grant = |path: &str, target: &str, ws: Option<&str>| {
        let mut g = json!({"proposal_sha256": psha, "path": path, "content_sha256": csha,
            "approver": "attacker", "target": target, "prior_sha256": null});
        if let Some(w) = ws {
            g["workspace"] = json!(w);
        }
        g
    };

    // sovereign-core #261: every shape of caller-written authorization is
    // refused at the note, before any confinement question arises: no
    // workspace, a target outside, `..`, an absolute path, a symlinked file,
    // a symlinked directory, a well-formed in-workspace grant, and the
    // reserved approved_grant kind.
    let refused_note = |g: serde_json::Value| note(&c, "authorization", g);
    let outside = root.join("outside.txt").to_string_lossy().to_string();
    std::fs::write(&outside, "outside\n").unwrap();
    std::os::unix::fs::symlink(&outside, ws.join("link.txt")).unwrap();
    let lt = ws.join("link.txt").to_string_lossy().to_string();
    let elsewhere = root.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, ws.join("sub")).unwrap();
    let st = ws.join("sub/x.txt").to_string_lossy().to_string();
    let mut fake = grant("NOTES.md", &target, Some(&w));
    fake["approved_grant"] = json!(1);
    fake["approval_key"] = json!("0".repeat(64));
    fake["replay_claim"] = json!(1);
    fake["cx_promotion"] = json!(1);
    fake["cx_evidence"] = json!(1);
    let mut minted = grant("NOTES.md", &target, Some(&w));
    minted["minted_grant"] = json!(1);
    minted["compose_commit"] = json!(1);
    for g in [
        grant("NOTES.md", &target, None),
        grant("NOTES.md", &outside, Some(&w)),
        grant("../outside.txt", &outside, Some(&w)),
        grant("etc/passwd", "/etc/passwd", Some("/")),
        grant("link.txt", &lt, Some(&w)),
        grant("sub/x.txt", &st, Some(&w)),
        grant("NOTES.md", &target, Some(&w)),
        fake,
        minted,
    ] {
        refused(refused_note(g).await, "sovereign-core #261");
    }

    // Nothing was written, and an intent on any id is refused.
    refused(
        intent(&c, 1, &psha, "NOTES.md", &target, &csha).await,
        "NotAuthorized",
    );
    assert!(!std::path::Path::new(&target).exists());
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside\n");
    // The parent-directory-swap cases (476ca4 c28, c28b) are covered on the
    // daemon-written approved grant in approved_attacks_test.rs.
}
