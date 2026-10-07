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
//! - STILL OPEN (sc#261, out of #249 scope): a same-user caller can write a
//!   generic grant inside a workspace it names and reach DONE. The last block
//!   asserts that observed behaviour so the follow-up flips it visibly.
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

    // A1 as found: a grant that names no workspace is refused.
    refused(
        note(&c, "authorization", grant("NOTES.md", &target, None)).await,
        "must name its workspace",
    );

    // A1b: target outside the workspace, in every shape.
    let outside = root.join("outside.txt").to_string_lossy().to_string();
    refused(
        note(&c, "authorization", grant("NOTES.md", &outside, Some(&w))).await,
        "OutsideWorkspace",
    );
    refused(
        note(
            &c,
            "authorization",
            grant("../outside.txt", &outside, Some(&w)),
        )
        .await,
        "OutsideWorkspace",
    );
    refused(
        note(
            &c,
            "authorization",
            grant("etc/passwd", "/etc/passwd", Some("/")),
        )
        .await,
        "OutsideWorkspace",
    );
    // A symlinked file and a symlinked directory inside the workspace.
    std::fs::write(&outside, "outside\n").unwrap();
    std::os::unix::fs::symlink(&outside, ws.join("link.txt")).unwrap();
    let lt = ws.join("link.txt").to_string_lossy().to_string();
    refused(
        note(&c, "authorization", grant("link.txt", &lt, Some(&w))).await,
        "OutsideWorkspace",
    );
    let elsewhere = root.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, ws.join("sub")).unwrap();
    let st = ws.join("sub/x.txt").to_string_lossy().to_string();
    refused(
        note(&c, "authorization", grant("sub/x.txt", &st, Some(&w))).await,
        "OutsideWorkspace",
    );

    // The approved grant kind cannot be forged through ComposeNote.
    let mut fake = grant("NOTES.md", &target, Some(&w));
    fake["approved_grant"] = json!(1);
    fake["approval_key"] = json!("0".repeat(64));
    fake["replay_claim"] = json!(1);
    fake["cx_promotion"] = json!(1);
    fake["cx_evidence"] = json!(1);
    refused(note(&c, "authorization", fake).await, "approved_grant");

    // A grant written while its target was inside, retargeted at the intent:
    // the intent names a target outside -> refused (no intent record).
    let ok = match note(&c, "authorization", grant("NOTES.md", &target, Some(&w))).await {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("in-workspace generic grant: {other:?}"),
    };
    refused(
        intent(&c, ok, &psha, "NOTES.md", &outside, &csha).await,
        "NotAuthorized",
    );
    // The target swapped for a symlink after the grant: refused at the intent
    // (the target read refuses a non-regular file before confinement does).
    std::os::unix::fs::symlink(&outside, &target).unwrap();
    refused(
        intent(&c, ok, &psha, "NOTES.md", &target, &csha).await,
        "EFFECT_REFUSED",
    );
    std::fs::remove_file(&target).unwrap();
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside\n");

    // STILL OPEN (sc#261): the generic grant inside the named workspace opens.
    let i = match intent(&c, ok, &psha, "NOTES.md", &target, &csha).await {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("generic grant intent: {other:?}"),
    };
    std::fs::write(&target, content).unwrap();
    let ack = match c
        .send_command(ControlCommand::ComposeEffectAck {
            intent: i,
            reported: json!({"wrote": true}),
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("ack: {other:?}"),
    };
    let state = match c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![ack],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(r) => {
            serde_json::from_str::<serde_json::Value>(r.cited[0].text.as_deref().unwrap()).unwrap()
                ["state"]
                .as_str()
                .unwrap()
                .to_string()
        }
        other => panic!("recall: {other:?}"),
    };
    eprintln!("A1 generic path (sc#261, still open to same-user callers): {state}");
    assert_eq!(state, "DONE", "observed behaviour changed: update sc#261");
}
