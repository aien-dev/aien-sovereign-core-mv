//! #249 C: `ComposeApprovedProposal` over the daemon socket (the production
//! entry), against the real server: an authenticated approval commits; a
//! forged approver is refused with nothing appended; replay records cannot be
//! forged through `ComposeNote` (so a forgery can neither consume nor
//! validate an approval); after a daemon restart over the same compose home
//! every replay is refused and the same approval's retry gets the original
//! result back without a second run. One test per binary (it sets
//! AIEN_COMPOSE_DIR).
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_omega_compose::hex;
use aien_runtime::approved::{approved_proposal_sha256, ApprovedProposal};
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Duration;

const PATH: &str = "NOTES.md";
const CONTENT: &str = "approved through the daemon\n";

fn signed(dir: &Path, request: &str, approval: &str, approver: &str) -> ApprovedProposal {
    let mut p = ApprovedProposal {
        request_id: request.into(),
        trace_id: format!("trace-{request}"),
        approval_id: approval.into(),
        approver: approver.into(),
        path: PATH.into(),
        content: CONTENT.into(),
        approved_proposal_sha256: approved_proposal_sha256(PATH, CONTENT),
        content_sha256: hex(&Sha256::digest(CONTENT.as_bytes())),
        approval_mac: String::new(),
    };
    p.approval_mac = DeskKey::load(&desk_key_path(dir)).unwrap().sign(&p);
    p
}

async fn start(
    socket: &Path,
) -> (
    AienRuntimeClient,
    tokio::task::JoinHandle<Result<(), String>>,
) {
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(256, 16));
    let server = AienRuntimeServer::new(spine, socket);
    let handle = tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
    let client = AienRuntimeClient::new(socket);
    for _ in 0..200 {
        if client.is_alive().await {
            return (client, handle);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

async fn submit(c: &AienRuntimeClient, p: &ApprovedProposal, ws: &Path) -> ControlResponse {
    c.send_command(ControlCommand::ComposeApprovedProposal {
        proposal: p.clone(),
        workspace: ws.display().to_string(),
    })
    .await
    .unwrap()
}

async fn records(c: &AienRuntimeClient) -> u64 {
    match c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(r) => r.records_total,
        other => panic!("recall: {other:?}"),
    }
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

fn refusal(r: ControlResponse) -> (String, String) {
    match r {
        ControlResponse::ComposeApprovedRefused(e) => (e.refused_by.clone(), e.name.clone()),
        other => panic!("want a refusal, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn approved_proposal_over_the_daemon_socket() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("compose");
    std::env::set_var("AIEN_COMPOSE_DIR", &dir);
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    DeskKey::create(&desk_key_path(&dir)).unwrap();
    let socket = tmp.path().join("runtime.sock");
    let (c, handle) = start(&socket).await;

    // Before anything: forged approver (correct bytes, hashes and the MAC of
    // the honest approval) is refused, nothing appended.
    let good = signed(&dir, "req-1", "appr-1", "interplane-host");
    let mut forged = good.clone();
    forged.approver = "drake".into();
    let out = submit(&c, &forged, &ws).await;
    assert_eq!(
        refusal(out),
        ("PROPOSAL_REFUSED".into(), "Unauthenticated".into())
    );
    if !aien_omega_compose::LINKED {
        // Stub build: the authenticated proposal is refused at the claim.
        let (by, _) = refusal(submit(&c, &good, &ws).await);
        assert_eq!(by, "REPLAY_REFUSED");
        let _ = c.send_command(ControlCommand::Shutdown).await;
        return;
    }
    // The forgery opened the home only to read nothing: no record written.
    let n0 = records(&c).await;

    // Replay records cannot be forged through ComposeNote: neither a claim
    // (which would consume req-2/appr-2) nor a commit.
    for text in [
        json!({"approved_submission": "accepted", "approval_key": "0".repeat(64),
               "request_id": "req-2", "approval_id": "appr-2", "trace_id": "trace-req-2",
               "executor": {"pid": 1, "start": 1}}),
        json!({"approved_submission": "committed", "claim": 1,
               "evidence": {"compose_proposal_sha256": "x", "cx_promotion": 1, "cx_evidence": 1, "task": 1}}),
    ] {
        match note(&c, "effect", text).await {
            ControlResponse::Error(e) => assert!(e.contains("approved_submission"), "{e}"),
            other => panic!("forged replay record accepted: {other:?}"),
        }
    }
    // Under another kind the field is plain text: written, but never read as a claim.
    match note(
        &c,
        "constraint",
        json!({"approved_submission": "accepted", "request_id": "req-2", "approval_id": "appr-2"}),
    )
    .await
    {
        ControlResponse::ComposeNoted(_) => {}
        other => panic!("constraint note: {other:?}"),
    }
    assert_eq!(records(&c).await, n0 + 1);

    // The authenticated approval commits through the production path.
    let first = match submit(&c, &good, &ws).await {
        ControlResponse::ComposeApprovedResult(r) => *r,
        other => panic!("want COMMITTED, got {other:?}"),
    };
    assert_eq!(first.state, "COMMITTED");
    assert_eq!(
        (
            first.request_id.as_str(),
            first.trace_id.as_str(),
            first.approval_id.as_str()
        ),
        ("req-1", "trace-req-1", "appr-1")
    );
    let t = first.task.as_ref().unwrap();
    assert!(t.committed && t.aegis_pass_mask & 1 == 1 && t.cx_promotion != 0);
    assert_eq!(first.grant_links, vec![t.cx_promotion, t.cx_evidence]);
    assert!(!ws.join(PATH).exists(), "the command writes nothing");
    // The forged notes did not consume req-2/appr-2.
    let second = signed(&dir, "req-2", "appr-2", "interplane-host");
    match submit(&c, &second, &ws).await {
        ControlResponse::ComposeApprovedResult(r) => assert_eq!(r.state, "COMMITTED"),
        other => panic!("req-2 must commit: {other:?}"),
    }
    // Replay in the same daemon.
    let n1 = records(&c).await;
    match submit(&c, &good, &ws).await {
        ControlResponse::ComposeApprovedResult(r) => {
            assert_eq!(r.state, "ALREADY_COMMITTED");
            assert!(r.task.is_none());
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        refusal(submit(&c, &signed(&dir, "req-3", "appr-1", "interplane-host"), &ws).await),
        ("REPLAY_REFUSED".into(), "AlreadyCommitted".into())
    );
    assert_eq!(records(&c).await, n1);

    // Daemon restart over the same compose home.
    let _ = c.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    let socket2 = tmp.path().join("runtime2.sock");
    let (c2, handle2) = start(&socket2).await;
    let n2 = records(&c2).await;
    match submit(&c2, &good, &ws).await {
        ControlResponse::ComposeApprovedResult(r) => {
            assert_eq!(r.state, "ALREADY_COMMITTED");
            assert_eq!(r.compose_proposal_sha256, first.compose_proposal_sha256);
            assert_eq!(r.grant_links, first.grant_links);
        }
        other => panic!("{other:?}"),
    }
    for p in [
        signed(&dir, "req-1", "appr-9", "interplane-host"),
        signed(&dir, "req-9", "appr-1", "interplane-host"),
        signed(&dir, "req-2", "appr-2", "another-approver"),
    ] {
        let (by, name) = refusal(submit(&c2, &p, &ws).await);
        assert_eq!(
            (by.as_str(), name.as_str()),
            ("REPLAY_REFUSED", "AlreadyCommitted")
        );
    }
    assert_eq!(
        records(&c2).await,
        n2,
        "a replay after restart appended records"
    );
    let _ = c2.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), handle2).await;
}
