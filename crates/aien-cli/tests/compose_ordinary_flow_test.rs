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

//! sovereign-core #261: the ordinary `aien compose` flow (propose, authorize,
//! execute) with the daemon minting the grant, driven through the real CLI
//! binary against an in-process daemon whose compose bridge has a fixed
//! proposer (no model). The bridge keeps its default (sc#328): authorize
//! requires the approval desk MAC, so the flow creates a desk key and signs
//! with `--desk`, as an operator does. Needs the linked composition archive;
//! in a stub build each test is IGNORED (cfg compose_linked from build.rs),
//! never passed.
#[path = "../../aien-runtime/tests/support/home_guard.rs"]
mod home_guard;

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const PROPOSAL: &str = "filename: NOTES.md\nkeep every change inside the workspace\n";
const CONTENT: &str = "keep every change inside the workspace\n";

fn proposer() -> ComposeProposer {
    Arc::new(|_prompt: &str, _limit: Duration| {
        Ok(Generation {
            text: PROPOSAL.to_string(),
            tokens: 8,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

struct Rig {
    _tmp: tempfile::TempDir,
    socket: PathBuf,
    ws: PathBuf,
    dir: PathBuf,
    client: AienRuntimeClient,
    handle: tokio::task::JoinHandle<Result<(), String>>,
}

async fn rig() -> Option<Rig> {
    if !aien_omega_compose::LINKED {
        panic!("built without compose_linked");
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    let ws = root.join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let socket = root.join("aien.sock");
    DeskKey::create(&desk_key_path(&root.join("compose"))).unwrap();
    let bridge = Arc::new(ComposeBridge::new(
        root.join("compose"),
        proposer(),
        "test:fixed-proposer",
    ));
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(256, 16));
    let server = AienRuntimeServer::new(spine, &socket).with_compose_bridge(bridge);
    let handle = tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
    let client = AienRuntimeClient::new(&socket);
    for _ in 0..250 {
        if client.is_alive().await {
            return Some(Rig {
                dir: root.clone(),
                _tmp: tmp,
                socket,
                ws,
                client,
                handle,
            });
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

impl Rig {
    /// Run `aien-cli compose <args>`; returns (exit code, JSON on stdout).
    async fn cli(&self, args: Vec<String>) -> (i32, Value) {
        let (sock, prov, compose) = (
            self.socket.clone(),
            self.dir.join("prov"),
            self.dir.join("compose"),
        );
        tokio::task::spawn_blocking(move || {
            let o = std::process::Command::new(env!("CARGO_BIN_EXE_aien-cli"))
                .arg("compose")
                .args(&args)
                .env("AIEN_RUNTIME_SOCK", &sock)
                .env("AIEN_PROVENANCE_DIR", &prov)
                .env("AIEN_COMPOSE_DIR", &compose)
                .output()
                .expect("run aien-cli");
            let out = String::from_utf8_lossy(&o.stdout).to_string();
            let v = out
                .lines()
                .rev()
                .find_map(|l| serde_json::from_str::<Value>(l).ok())
                .unwrap_or_else(|| json!({"unparsed": out}));
            (o.status.code().unwrap_or(-1), v)
        })
        .await
        .unwrap()
    }

    async fn propose(&self) -> PathBuf {
        let ws = self.ws.display().to_string();
        let (code, v) = self
            .cli(vec![
                "propose".into(),
                "--goal".into(),
                "write the note".into(),
                "--workspace".into(),
                ws,
            ])
            .await;
        assert_eq!(code, 0, "{v}");
        let report = self.dir.join("S3.json");
        std::fs::write(&report, v.to_string()).unwrap();
        report
    }

    fn args(&self, sub: &str, report: &Path, extra: &[&str]) -> Vec<String> {
        let mut a = vec![
            sub.to_string(),
            "--report".into(),
            report.display().to_string(),
            "--workspace".into(),
            self.ws.display().to_string(),
        ];
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    }

    async fn down(self) {
        let _ = self.client.send_command(ControlCommand::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), self.handle).await;
    }
}

static ONE_HOME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
async fn ordinary_flow_propose_authorize_execute_reaches_done() {
    let _home = home_guard::home_slot();
    let _t = ONE_HOME.lock().await;
    let Some(r) = rig().await else { return };
    let report = r.propose().await;
    let (code, auth) = r
        .cli(r.args(
            "authorize",
            &report,
            &["--approver", "drake", "--desk", "1"],
        ))
        .await;
    assert_eq!(code, 0, "{auth}");
    let id = auth["authorization"]["id"].as_u64().expect("grant id");
    // The grant is the daemon's own record, not a client note.
    let text = match r
        .client
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![id],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(x) => x.cited[0].text.clone().unwrap(),
        other => panic!("{other:?}"),
    };
    assert!(text.contains("\"minted_grant\":1"), "{text}");
    let ids = id.to_string();
    let (code, done) = r
        .cli(r.args("execute", &report, &["--authorization", &ids]))
        .await;
    assert_eq!(code, 0, "{done}");
    assert_eq!(done["state"], "DONE", "{done}");
    assert_eq!(
        std::fs::read_to_string(r.ws.join("NOTES.md")).unwrap(),
        CONTENT
    );
    // Replay: the same grant opens nothing a second time.
    let (code, again) = r
        .cli(r.args("execute", &report, &["--authorization", &ids]))
        .await;
    assert_ne!(code, 0, "{again}");
    assert!(again.to_string().contains("AlreadySpent"), "{again}");
    r.down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
async fn a_client_written_authorization_note_is_refused_and_opens_nothing() {
    let _home = home_guard::home_slot();
    let _t = ONE_HOME.lock().await;
    let Some(r) = rig().await else { return };
    let report = r.propose().await;
    let rep: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    let psha = rep["report"]["proposal_sha256"].as_str().unwrap();
    let csha = rep["report"]["proposal_content_sha256"].as_str().unwrap();
    // What the old CLI wrote: a complete, confined, well-formed grant note.
    let note = json!({"proposal_sha256": psha, "path": "NOTES.md", "content_sha256": csha,
        "approver": "attacker", "target": r.ws.join("NOTES.md").display().to_string(),
        "prior_sha256": null, "workspace": r.ws.display().to_string()});
    match r
        .client
        .send_command(ControlCommand::ComposeNote {
            kind: "authorization".into(),
            text: note.to_string(),
            links: vec![],
        })
        .await
        .unwrap()
    {
        ControlResponse::Error(e) => assert!(e.contains("sovereign-core #261"), "{e}"),
        other => panic!("ATTACK SUCCEEDED: {other:?}"),
    }
    // Nothing was authorized, so nothing is written, whatever id is named.
    let (code, v) = r
        .cli(r.args("execute", &report, &["--authorization", "1"]))
        .await;
    assert_ne!(code, 0, "{v}");
    assert!(!r.ws.join("NOTES.md").exists());
    r.down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
async fn authorize_without_the_desk_mac_is_refused_and_mints_nothing() {
    let _home = home_guard::home_slot();
    // sc#328: the default bridge requires the desk; an OS-user-only authorize
    // (what every client sent before #297) is refused and opens nothing.
    let _t = ONE_HOME.lock().await;
    let Some(r) = rig().await else { return };
    let report = r.propose().await;
    let (code, v) = r
        .cli(r.args("authorize", &report, &["--approver", "drake"]))
        .await;
    assert_ne!(code, 0, "{v}");
    assert!(v.to_string().contains("DeskMacRequired"), "{v}");
    let (code, v) = r
        .cli(r.args("execute", &report, &["--authorization", "1"]))
        .await;
    assert_ne!(code, 0, "{v}");
    assert!(!r.ws.join("NOTES.md").exists());
    // The same proposal, signed at the desk, is authorized: the refusal was
    // the missing MAC, not a broken rig.
    let (code, auth) = r
        .cli(r.args(
            "authorize",
            &report,
            &["--approver", "drake", "--desk", "1"],
        ))
        .await;
    assert_eq!(code, 0, "{auth}");
    assert!(auth["authorization"]["id"].as_u64().is_some(), "{auth}");
    r.down().await;
}
