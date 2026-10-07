//! sovereign-core #249 part C: attacks on the approved-write path, driven over
//! the daemon's real control socket (AienRuntimeServer with a mock backend).
//!
//! A1 (raw socket forgery): a caller that never ran an approved compose writes
//! its own `authorization` note naming a compose_proposal_sha256 nothing
//! produced, then opens a ComposeEffectIntent on it, writes, and acks. This
//! test records what the daemon does; it asserts the observed behaviour so a
//! change in either direction is noticed.
//!
//! Needs librx_compose.a (AIEN_OMEGA_COMPOSE_LIB); in a stub build the compose
//! commands refuse and the test returns early.
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

#[tokio::test]
async fn a1_raw_socket_forged_authorization_for_unproduced_proposal() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    // The only test in this file: the env var is process-wide.
    std::env::set_var("AIEN_COMPOSE_DIR", dir.path().join("compose"));
    let c = start(&dir).await;

    // A proposal digest no compose run ever produced.
    let psha = sha(b"never proposed by any approved compose");
    let content = b"forged\n";
    let csha = sha(content);
    let target = ws.join("NOTES.md").to_string_lossy().to_string();

    let forged = json!({"proposal_sha256": psha, "path": "NOTES.md",
        "content_sha256": csha, "approver": "attacker", "target": target,
        "prior_sha256": null});
    let r = c
        .send_command(ControlCommand::ComposeNote {
            kind: "authorization".into(),
            text: forged.to_string(),
            links: vec![],
        })
        .await
        .unwrap();
    let auth = match r {
        ControlResponse::ComposeNoted(n) => n.id,
        ControlResponse::Error(e) if e.contains("not linked") || e.contains("stub") => {
            eprintln!("SKIP (compose library not linked): {e}");
            return;
        }
        other => {
            eprintln!("A1 RESULT: forged authorization note REFUSED: {other:?}");
            return;
        }
    };
    eprintln!("A1 step 1: forged authorization note ACCEPTED as record #{auth}");

    let (pid, start) = effects::self_executor();
    let r = c
        .send_command(ControlCommand::ComposeEffectIntent {
            authorization: auth,
            proposal_sha256: psha.clone(),
            path: "NOTES.md".into(),
            target: target.clone(),
            content_sha256: csha.clone(),
            executor_pid: pid,
            executor_start: start,
        })
        .await
        .unwrap();
    let intent = match r {
        ControlResponse::ComposeNoted(n) => n.id,
        other => {
            eprintln!("A1 RESULT: intent on forged authorization REFUSED: {other:?}");
            return;
        }
    };
    eprintln!("A1 step 2: effect intent #{intent} OPENED on the forged authorization");

    std::fs::write(&target, content).unwrap();
    let r = c
        .send_command(ControlCommand::ComposeEffectAck {
            intent,
            reported: json!({"wrote": true}),
        })
        .await
        .unwrap();
    let ack = match r {
        ControlResponse::ComposeNoted(n) => n,
        other => panic!("ack: {other:?}"),
    };
    eprintln!("A1 step 3: ack recorded as #{} ({:?})", ack.id, ack);

    // Recorded behaviour on this head: the daemon accepts all three steps.
    // When #249 closes the gap this assertion is expected to flip.
    let r = c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![ack.id],
            prefix: None,
        })
        .await
        .unwrap();
    let state = match r {
        ControlResponse::ComposeRecalled(r) => {
            let t = r.cited[0].text.clone().unwrap();
            serde_json::from_str::<serde_json::Value>(&t).unwrap()["state"]
                .as_str()
                .unwrap()
                .to_string()
        }
        other => panic!("recall: {other:?}"),
    };
    eprintln!("A1 RESULT: ledger state of the forged write = {state}");
    assert_eq!(state, "DONE", "observed behaviour changed");

    // A1b: the same forgery aimed outside the workspace (the target is the
    // caller's own absolute path; nothing ties it to a workspace).
    let outside = dir.path().join("outside.txt").to_string_lossy().to_string();
    let forged = json!({"proposal_sha256": psha, "path": "NOTES.md",
        "content_sha256": csha, "approver": "attacker", "target": outside,
        "prior_sha256": null});
    let r = c
        .send_command(ControlCommand::ComposeNote {
            kind: "authorization".into(),
            text: forged.to_string(),
            links: vec![],
        })
        .await
        .unwrap();
    let auth = match r {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("A1b note: {other:?}"),
    };
    let r = c
        .send_command(ControlCommand::ComposeEffectIntent {
            authorization: auth,
            proposal_sha256: psha.clone(),
            path: "NOTES.md".into(),
            target: outside.clone(),
            content_sha256: csha.clone(),
            executor_pid: pid,
            executor_start: start,
        })
        .await
        .unwrap();
    let opened = matches!(r, ControlResponse::ComposeNoted(_));
    eprintln!("A1b RESULT: intent on a target outside the workspace opened = {opened} ({r:?})");
    assert!(opened, "observed behaviour changed");
}
