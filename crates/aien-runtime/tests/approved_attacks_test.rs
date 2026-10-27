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

//! sovereign-core #249 part C: adversarial tests of the approved-write path,
//! each driven over the daemon's real control socket (AienRuntimeServer with a
//! mock backend; the approved proposal text is the proposer, no model).
//!
//! Every test asserts the SECURE outcome: it passes when the attack is refused
//! and fails ("ATTACK SUCCEEDED") when it is not. Tests share one process-wide
//! AIEN_COMPOSE_DIR, so they take turns.
//!
//! Needs librx_compose.a (AIEN_OMEGA_COMPOSE_LIB); in a stub build every test
//! returns early. The two crash tests need `--features fault-hold` (the
//! production build compiles the crash points to nothing).
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_omega_compose::hex;
use aien_runtime::approved::{approved_proposal_sha256, ApprovedProposal};
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse};
use aien_runtime::effects;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

const PATH: &str = "NOTES.md";
const CONTENT: &str = "approved through the daemon\n";
const APPROVER: &str = "interplane-host";

/// One daemon at a time: AIEN_COMPOSE_DIR is process-wide.
static TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn home(tmp: &Path) -> PathBuf {
    tmp.join("compose")
}

fn ws(tmp: &Path) -> PathBuf {
    let w = tmp.join("ws");
    std::fs::create_dir_all(&w).unwrap();
    w
}

/// A compose home with a desk key, and a workspace beside it.
fn fresh() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    DeskKey::create(&desk_key_path(&home(tmp.path()))).unwrap();
    ws(tmp.path());
    tmp
}

fn unsigned(request: &str, approval: &str, path: &str, content: &str) -> ApprovedProposal {
    ApprovedProposal {
        request_id: request.into(),
        trace_id: format!("trace-{request}"),
        approval_id: approval.into(),
        approver: APPROVER.into(),
        path: path.into(),
        content: content.into(),
        approved_proposal_sha256: approved_proposal_sha256(path, content),
        content_sha256: sha(content.as_bytes()),
        approval_mac: String::new(),
        requirements: Some(String::new()),
        requirements_mac: String::new(),
        requirements_base: None,
    }
}

/// Signed by the real desk key of `tmp`'s compose home.
fn signed(tmp: &Path, request: &str, approval: &str) -> ApprovedProposal {
    sign(tmp, unsigned(request, approval, PATH, CONTENT))
}

fn sign(tmp: &Path, mut p: ApprovedProposal) -> ApprovedProposal {
    DeskKey::load(&desk_key_path(&home(tmp)))
        .unwrap()
        .seal(&mut p, &ws(tmp));
    p
}

struct Daemon {
    c: AienRuntimeClient,
    socket: PathBuf,
    h: tokio::task::JoinHandle<Result<(), String>>,
}

/// Start a daemon over `tmp`'s compose home (one at a time per compose home).
async fn up(tmp: &Path, name: &str) -> Daemon {
    std::env::set_var("AIEN_COMPOSE_DIR", home(tmp));
    let socket = tmp.join(name);
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
    let h = tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
    let c = AienRuntimeClient::new(&socket);
    for _ in 0..250 {
        if c.is_alive().await {
            return Daemon { c, socket, h };
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

async fn down(d: Daemon) {
    let _ = d.c.send_command(ControlCommand::Shutdown).await;
    let _ = tokio::time::timeout(Duration::from_secs(5), d.h).await;
}

async fn submit(c: &AienRuntimeClient, p: &ApprovedProposal, ws: &Path) -> ControlResponse {
    c.send_command(ControlCommand::ComposeApprovedProposal {
        proposal: p.clone(),
        workspace: ws.display().to_string(),
    })
    .await
    .unwrap()
}

/// (records_total, every host record's text).
async fn journal(c: &AienRuntimeClient) -> (u64, Vec<String>) {
    match c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(r) => (
            r.records_total,
            r.host.iter().filter_map(|h| h.text.clone()).collect(),
        ),
        other => panic!("recall: {other:?}"),
    }
}

fn phase_count(texts: &[String], phase: &str) -> usize {
    let needle = format!("\"approved_submission\":\"{phase}\"");
    texts.iter().filter(|t| t.contains(&needle)).count()
}

/// The refusal (refused_by, name), or panic naming the attack as successful.
fn refused(r: ControlResponse, attack: &str) -> (String, String) {
    match r {
        ControlResponse::ComposeApprovedRefused(e) => (e.refused_by.clone(), e.name.clone()),
        other => panic!("ATTACK SUCCEEDED ({attack}): {other:?}"),
    }
}

fn committed(r: ControlResponse) -> aien_runtime::approved::ApprovedComposeReport {
    match r {
        ControlResponse::ComposeApprovedResult(r) => *r,
        other => panic!("honest approval did not commit: {other:?}"),
    }
}

/// One field of an honestly signed approval changed after signing (hashes
/// recomputed where the field feeds them, so only the MAC can catch it).
async fn tampered(attack: &str, edit: impl FnOnce(&Path, &mut ApprovedProposal)) {
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let good = signed(tmp.path(), "req-t", "appr-t");
    let mut p = good.clone();
    edit(tmp.path(), &mut p);
    let (n0, _) = journal(&d.c).await;
    let (by, name) = refused(submit(&d.c, &p, &ws(tmp.path())).await, attack);
    assert_eq!(
        (by.as_str(), name.as_str()),
        ("PROPOSAL_REFUSED", "Unauthenticated")
    );
    assert_eq!(
        journal(&d.c).await.0,
        n0,
        "{attack}: refusal appended records"
    );
    // The forgery did not consume the honest approval.
    assert_eq!(
        committed(submit(&d.c, &good, &ws(tmp.path())).await).state,
        "COMMITTED"
    );
    down(d).await;
}

fn rehash(p: &mut ApprovedProposal) {
    p.approved_proposal_sha256 = approved_proposal_sha256(&p.path, &p.content);
    p.content_sha256 = sha(p.content.as_bytes());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c01_forged_approver() {
    let _home = home_guard::home_slot();
    tampered("forged approver", |_, p| p.approver = "drake".into()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c02a_invalid_mac_random() {
    let _home = home_guard::home_slot();
    tampered("random MAC", |_, p| p.approval_mac = "ab".repeat(32)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c02b_invalid_mac_empty_and_short() {
    let _home = home_guard::home_slot();
    tampered("empty MAC", |_, p| p.approval_mac.clear()).await;
    tampered("short MAC", |_, p| p.approval_mac.truncate(63)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c02c_invalid_mac_uppercase_of_the_real_one() {
    let _home = home_guard::home_slot();
    tampered("uppercase MAC", |_, p| {
        p.approval_mac = p.approval_mac.to_uppercase()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c02d_mac_from_another_desk_key() {
    let _home = home_guard::home_slot();
    tampered("MAC under another desk key", |tmp, p| {
        let other = DeskKey::create(&tmp.join("other-desk").join("k")).unwrap();
        p.approval_mac = other.sign(p, &ws(tmp));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c03_changed_trace_id() {
    let _home = home_guard::home_slot();
    tampered("changed trace_id", |_, p| p.trace_id = "trace-other".into()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c04_changed_request_id() {
    let _home = home_guard::home_slot();
    tampered("changed request_id", |_, p| {
        p.request_id = "req-other".into()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c05_changed_approval_id() {
    let _home = home_guard::home_slot();
    tampered("changed approval_id", |_, p| {
        p.approval_id = "appr-other".into()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c06_changed_path() {
    let _home = home_guard::home_slot();
    tampered("changed path", |_, p| {
        p.path = "OTHER.md".into();
        rehash(p);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c07_changed_content() {
    let _home = home_guard::home_slot();
    tampered("changed content", |_, p| {
        p.content = "attacker content\n".into();
        rehash(p);
    })
    .await;
}

/// Invalid authentication evidence on the daemon side: a desk key file other
/// users can read, or no desk key, refuses everything (NoDesk), nothing runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c08_desk_key_open_to_others_or_missing() {
    let _home = home_guard::home_slot();
    use std::os::unix::fs::PermissionsExt;
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let p = signed(tmp.path(), "req-k", "appr-k");
    let key = desk_key_path(&home(tmp.path()));
    let (n0, _) = journal(&d.c).await;
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
    let (_, name) = refused(submit(&d.c, &p, &ws(tmp.path())).await, "desk key 0644");
    assert_eq!(name, "NoDesk");
    let saved = std::fs::read(&key).unwrap();
    std::fs::remove_file(&key).unwrap();
    let (_, name) = refused(submit(&d.c, &p, &ws(tmp.path())).await, "no desk key");
    assert_eq!(name, "NoDesk");
    assert_eq!(
        journal(&d.c).await.0,
        n0,
        "a NoDesk refusal appended records"
    );
    // Restored: the same approval was not consumed by the refusals.
    std::fs::write(&key, saved).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(
        committed(submit(&d.c, &p, &ws(tmp.path())).await).state,
        "COMMITTED"
    );
    down(d).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c09_same_process_replay() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let w = ws(tmp.path());
    let p = signed(tmp.path(), "req-r", "appr-r");
    let first = committed(submit(&d.c, &p, &w).await);
    assert_eq!(first.state, "COMMITTED");
    let (n1, _) = journal(&d.c).await;
    // The same approval again: the original result, no second run.
    let again = committed(submit(&d.c, &p, &w).await);
    assert_eq!(
        again.state, "ALREADY_COMMITTED",
        "ATTACK SUCCEEDED: ran twice"
    );
    assert!(again.task.is_none());
    assert_eq!(again.compose_proposal_sha256, first.compose_proposal_sha256);
    // Either id reused under a fresh, validly signed approval: refused.
    for q in [
        signed(tmp.path(), "req-r", "appr-new"),
        signed(tmp.path(), "req-new", "appr-r"),
    ] {
        let (by, _) = refused(submit(&d.c, &q, &w).await, "id reuse");
        assert_eq!(by, "REPLAY_REFUSED");
    }
    let (n2, texts) = journal(&d.c).await;
    assert_eq!(n2, n1, "a replay appended records");
    assert_eq!(phase_count(&texts, "committed"), 1);
    down(d).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c10_replay_after_daemon_restart() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let w = ws(tmp.path());
    let p = signed(tmp.path(), "req-a", "appr-a");
    let d = up(tmp.path(), "s1.sock").await;
    let first = committed(submit(&d.c, &p, &w).await);
    down(d).await;
    let d = up(tmp.path(), "s2.sock").await;
    let (n0, _) = journal(&d.c).await;
    let again = committed(submit(&d.c, &p, &w).await);
    assert_eq!(
        again.state, "ALREADY_COMMITTED",
        "ATTACK SUCCEEDED: ran after restart"
    );
    assert_eq!(again.grant_links, first.grant_links);
    let (by, _) = refused(
        submit(&d.c, &signed(tmp.path(), "req-b", "appr-a"), &w).await,
        "approval_id reuse after restart",
    );
    assert_eq!(by, "REPLAY_REFUSED");
    let (n1, texts) = journal(&d.c).await;
    assert_eq!(n1, n0, "replay after restart appended records");
    assert_eq!(phase_count(&texts, "committed"), 1);
    down(d).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c11_concurrent_duplicate_submissions() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let w = ws(tmp.path());
    let p = signed(tmp.path(), "req-c", "appr-c");
    let mut tasks = Vec::new();
    for _ in 0..12 {
        let (s, p, w) = (d.socket.clone(), p.clone(), w.clone());
        tasks.push(tokio::spawn(async move {
            submit(&AienRuntimeClient::new(&s), &p, &w).await
        }));
    }
    let mut ran = 0;
    for t in tasks {
        match t.await.unwrap() {
            ControlResponse::ComposeApprovedResult(r) if r.task.is_some() => ran += 1,
            ControlResponse::ComposeApprovedResult(r) => assert_eq!(r.state, "ALREADY_COMMITTED"),
            ControlResponse::ComposeApprovedRefused(r) => {
                assert_eq!(r.refused_by, "REPLAY_REFUSED")
            }
            other => panic!("{other:?}"),
        }
    }
    let (_, texts) = journal(&d.c).await;
    assert_eq!(
        ran, 1,
        "ATTACK SUCCEEDED: {ran} compose runs for one approval"
    );
    assert_eq!(phase_count(&texts, "accepted"), 1);
    assert_eq!(phase_count(&texts, "committed"), 1);
    down(d).await;
}

/// The response is lost: the caller hangs up after the daemon committed the
/// approval, without reading the answer. The retry gets the original result
/// back (ALREADY_COMMITTED); the approval ran exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c14a_retry_after_response_loss() {
    let _home = home_guard::home_slot();
    lost_response(true).await;
}

/// The caller hangs up at once, before any answer. Whatever the daemon did
/// with the abandoned request, the retry runs the approval at most once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c14b_retry_after_immediate_hangup() {
    let _home = home_guard::home_slot();
    lost_response(false).await;
}

async fn lost_response(wait: bool) {
    use tokio::io::AsyncWriteExt;
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let w = ws(tmp.path());
    let p = signed(tmp.path(), "req-l", "appr-l");
    let env = ControlEnvelope {
        protocol_version: 1,
        request_id: 7,
        operation_id: 7,
        operator_session: 1,
        command: ControlCommand::ComposeApprovedProposal {
            proposal: p.clone(),
            workspace: w.display().to_string(),
        },
    };
    let mut s = tokio::net::UnixStream::connect(&d.socket).await.unwrap();
    s.write_all(format!("{}\n", serde_json::to_string(&env).unwrap()).as_bytes())
        .await
        .unwrap();
    // Read nothing; hang up only once the daemon has committed the approval,
    // so the response (not the request) is what is lost.
    let mut ran = !wait;
    for _ in 0..(if wait { 400 } else { 0 }) {
        if phase_count(&journal(&d.c).await.1, "committed") == 1 {
            ran = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    drop(s); // response lost
    assert!(ran, "the first submission never committed");
    let mut last = None;
    for _ in 0..200 {
        match submit(&d.c, &p, &w).await {
            ControlResponse::ComposeApprovedResult(r) => {
                last = Some(r.state.clone());
                break;
            }
            ControlResponse::ComposeApprovedRefused(r) if r.name == "InFlight" => {
                tokio::time::sleep(Duration::from_millis(25)).await
            }
            other => panic!("retry after a lost response: {other:?}"),
        }
    }
    let (_, texts) = journal(&d.c).await;
    if wait {
        assert_eq!(
            last.as_deref(),
            Some("ALREADY_COMMITTED"),
            "retry must get the original result"
        );
    }
    assert!(last.is_some(), "retry never got a result");
    assert_eq!(phase_count(&texts, "accepted"), 1);
    assert_eq!(
        phase_count(&texts, "committed"),
        1,
        "ATTACK SUCCEEDED: ran twice"
    );
    eprintln!("c14 (wait {wait}): retry state = {last:?}");
    down(d).await;
}

/// Replay records cannot be written by a caller (a forged claim would consume,
/// a forged commit would validate an approval).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c15_forged_replay_records_via_compose_note() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    for phase in [
        "accepted",
        "in_flight",
        "committed",
        "failed",
        "not_executed",
        "uncertain",
        "declared",
    ] {
        let text = json!({"approved_submission": phase, "claim": 1, "request_id": "req-f",
            "approval_id": "appr-f", "approval_key": "0".repeat(64)});
        match d
            .c
            .send_command(ControlCommand::ComposeNote {
                kind: "effect".into(),
                text: text.to_string(),
                links: vec![],
            })
            .await
            .unwrap()
        {
            ControlResponse::Error(_) => {}
            other => panic!("ATTACK SUCCEEDED: forged {phase} record written: {other:?}"),
        }
    }
    let w = ws(tmp.path());
    assert_eq!(
        committed(submit(&d.c, &signed(tmp.path(), "req-f", "appr-f"), &w).await).state,
        "COMMITTED"
    );
    down(d).await;
}

/// Even a desk-signed approval may not name a path outside the workspace.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c16_desk_signed_path_escaping_the_workspace() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let w = ws(tmp.path());
    let (n0, _) = journal(&d.c).await;
    for (i, path) in ["../escape.md", "/tmp/escape.md", "sub/../../escape.md"]
        .iter()
        .enumerate()
    {
        let p = sign(
            tmp.path(),
            unsigned(&format!("req-p{i}"), &format!("appr-p{i}"), path, CONTENT),
        );
        let r = submit(&d.c, &p, &w).await;
        let (by, name) = refused(r, &format!("path {path}"));
        eprintln!("c16: {path} -> {by} {name}");
    }
    assert_eq!(journal(&d.c).await.0, n0);
    down(d).await;
}

/// c17 / c17b: an approval signed for workspace A presented with workspace B
/// is Unauthenticated, appends nothing, and does not consume the approval:
/// presented with A afterwards it commits, and its grant targets A.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c17_approval_redirected_to_another_workspace() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let other = tmp.path().join("other-ws");
    std::fs::create_dir_all(&other).unwrap();
    let p = signed(tmp.path(), "req-w", "appr-w");
    let (n0, _) = journal(&d.c).await;
    let (by, name) = refused(
        submit(&d.c, &p, &other).await,
        "approval for one workspace used in another",
    );
    assert_eq!(
        (by.as_str(), name.as_str()),
        ("PROPOSAL_REFUSED", "Unauthenticated")
    );
    assert_eq!(
        journal(&d.c).await.0,
        n0,
        "c17b: the refusal appended records"
    );
    let r = committed(submit(&d.c, &p, &ws(tmp.path())).await);
    assert_eq!(
        r.state, "COMMITTED",
        "c17b: the refusal consumed the approval"
    );
    let want = canon(&ws(tmp.path())).join(PATH).display().to_string();
    assert_eq!(r.target.as_deref(), Some(want.as_str()));
    down(d).await;
}

/// c17c: the bound workspace presented through a symlink to it canonicalises
/// to the same directory: accepted, and the grant targets the real directory
/// (never the link). A symlink to a DIFFERENT directory is Unauthenticated.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c17c_workspace_through_a_symlink() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let other = tmp.path().join("other-ws");
    std::fs::create_dir_all(&other).unwrap();
    let to_other = tmp.path().join("link-other");
    std::os::unix::fs::symlink(&other, &to_other).unwrap();
    let p = signed(tmp.path(), "req-y", "appr-y");
    let (_, name) = refused(
        submit(&d.c, &p, &to_other).await,
        "symlink to another workspace",
    );
    assert_eq!(name, "Unauthenticated");
    let to_ws = tmp.path().join("link-ws");
    std::os::unix::fs::symlink(ws(tmp.path()), &to_ws).unwrap();
    let r = committed(submit(&d.c, &p, &to_ws).await);
    let want = canon(&ws(tmp.path())).join(PATH).display().to_string();
    assert_eq!(
        r.target.as_deref(),
        Some(want.as_str()),
        "grant target through the link"
    );
    down(d).await;
}

fn canon(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap()
}

/// A compose home inside the workspace (the desk key within the model's reach).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c20_workspace_containing_the_compose_home() {
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    // Signed for the overlapping workspace itself: only confinement can refuse.
    let mut p = unsigned("req-o", "appr-o", PATH, CONTENT);
    DeskKey::load(&desk_key_path(&home(tmp.path())))
        .unwrap()
        .seal(&mut p, tmp.path());
    let (_, name) = refused(submit(&d.c, &p, tmp.path()).await, "overlapping workspace");
    assert_eq!(name, "Confinement");
    down(d).await;
}

/// A1: a caller writes its own `authorization` note (workspace, path and a
/// confined target, as #262 requires) for a proposal digest no approved
/// compose produced, then opens an effect intent on it.
///
/// sovereign-core #261: the daemon refuses a raw `authorization` note on every
/// path, so the forged grant is never written and no intent can open.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c18_a1_raw_forged_authorization() {
    let opened = forged_authorization(|tmp| {
        let w = canon(&ws(tmp));
        (w.clone(), PATH.into(), w.join(PATH))
    })
    .await;
    assert_eq!(
        opened, "note refused",
        "ATTACK SUCCEEDED (sc#261 A1): a caller-written authorization opened an effect intent"
    );
}

/// A1b: the same forgery naming a target that is not workspace/path: refused
/// at the note (#262 confinement).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c19_a1b_forged_authorization_outside_workspace() {
    let opened = forged_authorization(|tmp| {
        let w = canon(&ws(tmp));
        (w, PATH.into(), canon(tmp).join("outside.txt"))
    })
    .await;
    assert_eq!(
        opened, "note refused",
        "ATTACK SUCCEEDED: target outside its workspace"
    );
}

/// A1c: the note names its OWN workspace: the parent of the real workspace,
/// which also holds the compose home. Refused like c18 (sc#261).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c19c_a1c_forged_authorization_names_its_own_workspace() {
    let opened = forged_authorization(|tmp| {
        let w = canon(tmp);
        (w.clone(), "outside.txt".into(), w.join("outside.txt"))
    })
    .await;
    assert_eq!(
        opened, "note refused",
        "ATTACK SUCCEEDED (sc#261 A1c): a note naming its own workspace opened an effect intent"
    );
}

/// "opened" = an effect intent opened on the caller-written authorization,
/// "note refused" = ComposeNote refused it, otherwise the intent's refusal.
async fn forged_authorization(at: impl FnOnce(&Path) -> (PathBuf, String, PathBuf)) -> String {
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (w, path, target) = at(tmp.path());
    let target = target.display().to_string();
    let psha = sha(b"never proposed by any approved compose");
    let csha = sha(b"forged\n");
    let note = json!({"proposal_sha256": psha, "path": path, "content_sha256": csha,
        "approver": "attacker", "workspace": w.display().to_string(), "target": target,
        "prior_sha256": null});
    let auth = match d
        .c
        .send_command(ControlCommand::ComposeNote {
            kind: "authorization".into(),
            text: note.to_string(),
            links: vec![],
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeNoted(n) => n.id,
        ControlResponse::Error(e) => {
            eprintln!("forged authorization: note refused: {e}");
            down(d).await;
            return "note refused".into();
        }
        other => panic!("{other:?}"),
    };
    let r = intent(&d.c, auth, &psha, &path, &target, &csha).await;
    eprintln!("forged authorization: intent: {r:?}");
    down(d).await;
    match r {
        ControlResponse::ComposeNoted(_) => "opened".into(),
        ControlResponse::Error(e) => e,
        other => panic!("{other:?}"),
    }
}

async fn intent(
    c: &AienRuntimeClient,
    authorization: u64,
    proposal_sha256: &str,
    path: &str,
    target: &str,
    content_sha256: &str,
) -> ControlResponse {
    let (pid, start) = effects::self_executor();
    c.send_command(ControlCommand::ComposeEffectIntent {
        authorization,
        proposal_sha256: proposal_sha256.into(),
        path: path.into(),
        target: target.into(),
        content_sha256: content_sha256.into(),
        executor_pid: pid,
        executor_start: start,
    })
    .await
    .unwrap()
}

/// The effect intent for an approved compose's grant.
async fn grant_intent(
    c: &AienRuntimeClient,
    r: &aien_runtime::approved::ApprovedComposeReport,
    grant: u64,
) -> ControlResponse {
    let target = r.target.clone().expect("grant target");
    intent(
        c,
        grant,
        &r.compose_proposal_sha256,
        &r.path,
        &target,
        &r.content_sha256,
    )
    .await
}

fn opened(r: ControlResponse, what: &str) -> u64 {
    match r {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("{what}: intent did not open: {other:?}"),
    }
}

fn not_opened(r: ControlResponse, attack: &str) -> String {
    match r {
        ControlResponse::Error(e) => e,
        other => panic!("ATTACK SUCCEEDED ({attack}): {other:?}"),
    }
}

/// The `state` the ack record for `intent` holds.
async fn ack_state(c: &AienRuntimeClient, intent: u64) -> String {
    match c
        .send_command(ControlCommand::ComposeEffectAck {
            intent,
            reported: json!({"executor": "attack-test"}),
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeNoted(_) => {}
        other => panic!("ack: {other:?}"),
    }
    let needle = format!("\"intent\":{intent},");
    let (_, texts) = journal(c).await;
    let t = texts
        .iter()
        .rev()
        .find(|t| t.contains("\"phase\":\"ack\"") && t.contains(&needle))
        .expect("ack record");
    let v: serde_json::Value = serde_json::from_str(t).unwrap();
    v["state"].as_str().unwrap().to_string()
}

/// An approved compose, committed, with its grant id.
async fn approved_grant(
    d: &Daemon,
    tmp: &Path,
) -> (aien_runtime::approved::ApprovedComposeReport, u64) {
    let r = committed(submit(&d.c, &signed(tmp, "req-g", "appr-g"), &ws(tmp)).await);
    let g = r.approved_grant.expect("COMMITTED report names its grant");
    (r, g)
}

/// The host text and links of record `id`.
async fn text_of(c: &AienRuntimeClient, id: u64) -> (String, Vec<u64>) {
    match c
        .send_command(ControlCommand::ComposeRecall {
            ids: vec![id],
            prefix: None,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeRecalled(r) => r
            .host
            .iter()
            .find(|h| h.id == id)
            .map(|h| (h.text.clone().expect("record text"), h.links.clone()))
            .expect("record"),
        other => panic!("recall: {other:?}"),
    }
}

/// c26: the honest path. Approved compose -> grant -> intent -> write -> ack
/// DONE; a second intent on the same grant is AlreadySpent; the approval
/// presented again is ALREADY_COMMITTED and carries no new grant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c26_happy_path_one_approval_one_effect() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (r, g) = approved_grant(&d, tmp.path()).await;
    let i = opened(grant_intent(&d.c, &r, g).await, "honest grant");
    std::fs::write(r.target.as_deref().unwrap(), CONTENT).unwrap();
    assert_eq!(ack_state(&d.c, i).await, "DONE");
    let e = not_opened(
        grant_intent(&d.c, &r, g).await,
        "second effect on one grant",
    );
    assert!(e.contains("AlreadySpent"), "{e}");
    let again = committed(
        submit(
            &d.c,
            &signed(tmp.path(), "req-g", "appr-g"),
            &ws(tmp.path()),
        )
        .await,
    );
    assert_eq!(again.state, "ALREADY_COMMITTED");
    assert_eq!(again.approved_grant, None);
    down(d).await;
}

/// c21: after the intent opens, the target is swapped for a symlink to a file
/// outside the workspace holding the approved bytes. The ack must not record
/// DONE. A symlink placed BEFORE the intent: the intent is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c21_symlink_swap_of_the_target() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (r, g) = approved_grant(&d, tmp.path()).await;
    let target = PathBuf::from(r.target.clone().unwrap());
    let outside = tmp.path().join("outside.txt");
    std::fs::write(&outside, CONTENT).unwrap();
    // Before the intent: refused.
    std::os::unix::fs::symlink(&outside, &target).unwrap();
    let e = not_opened(
        grant_intent(&d.c, &r, g).await,
        "intent on a symlinked target",
    );
    eprintln!("c21 before: {e}");
    std::fs::remove_file(&target).unwrap();
    // After the intent: the ack reads the world and must not say DONE.
    let i = opened(grant_intent(&d.c, &r, g).await, "honest grant");
    std::os::unix::fs::symlink(&outside, &target).unwrap();
    let st = ack_state(&d.c, i).await;
    assert_ne!(
        st, "DONE",
        "ATTACK SUCCEEDED: DONE recorded through a symlink"
    );
    eprintln!("c21 after: ack state {st}");
    down(d).await;
}

/// c28: the approved path is in a subdirectory. After the intent opens, the
/// subdirectory is swapped for a symlink to a directory outside the
/// workspace, and the approved bytes are written there. The ack must not
/// record DONE (the bytes are outside the workspace).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c28_parent_directory_swap_after_the_intent() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let w = ws(tmp.path());
    std::fs::create_dir_all(w.join("sub")).unwrap();
    let d = up(tmp.path(), "s.sock").await;
    let p = sign(
        tmp.path(),
        unsigned("req-p", "appr-p", "sub/NOTES.md", CONTENT),
    );
    let r = committed(submit(&d.c, &p, &w).await);
    let g = r.approved_grant.expect("grant");
    let i = opened(grant_intent(&d.c, &r, g).await, "honest grant");
    let outdir = tmp.path().join("outdir");
    std::fs::create_dir_all(&outdir).unwrap();
    std::fs::rename(w.join("sub"), w.join("sub-moved")).unwrap();
    std::os::unix::fs::symlink(&outdir, w.join("sub")).unwrap();
    std::fs::write(outdir.join("NOTES.md"), CONTENT).unwrap();
    let st = ack_state(&d.c, i).await;
    eprintln!("c28: ack state {st}");
    down(d).await;
    assert_ne!(
        st, "DONE",
        "ATTACK SUCCEEDED: DONE recorded for bytes outside the workspace (parent symlink)"
    );
}

/// c28b: the same parent swap, settled by reconcile instead of the ack. The
/// intent names an executor that has already exited, so ComposeReconcile
/// decides it from the world: it must not record DONE.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c28b_parent_directory_swap_settled_by_reconcile() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let w = ws(tmp.path());
    std::fs::create_dir_all(w.join("sub")).unwrap();
    let d = up(tmp.path(), "s.sock").await;
    let p = sign(
        tmp.path(),
        unsigned("req-q", "appr-q", "sub/NOTES.md", CONTENT),
    );
    let r = committed(submit(&d.c, &p, &w).await);
    let g = r.approved_grant.expect("grant");
    // An executor that is gone: a child that has exited. A dead pid reads as
    // not alive whatever start ticks the intent names.
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    let (_, start) = effects::self_executor();
    let i = match d
        .c
        .send_command(ControlCommand::ComposeEffectIntent {
            authorization: g,
            proposal_sha256: r.compose_proposal_sha256.clone(),
            path: r.path.clone(),
            target: r.target.clone().unwrap(),
            content_sha256: r.content_sha256.clone(),
            executor_pid: pid,
            executor_start: start,
        })
        .await
        .unwrap()
    {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("intent did not open: {other:?}"),
    };
    let outdir = tmp.path().join("outdir");
    std::fs::create_dir_all(&outdir).unwrap();
    std::fs::rename(w.join("sub"), w.join("sub-moved")).unwrap();
    std::os::unix::fs::symlink(&outdir, w.join("sub")).unwrap();
    std::fs::write(outdir.join("NOTES.md"), CONTENT).unwrap();
    let out =
        d.c.send_command(ControlCommand::ComposeReconcile { declare: None })
            .await
            .unwrap();
    down(d).await;
    let ControlResponse::ComposeReconciled(rep) = out else {
        panic!("reconcile: {out:?}");
    };
    let o = rep
        .outcomes
        .iter()
        .find(|o| o.intent == i)
        .expect("the intent was looked at");
    eprintln!("c28b: reconcile {} ({})", o.state, o.note);
    assert_ne!(
        o.state, "DONE",
        "ATTACK SUCCEEDED: reconcile settled DONE for bytes outside the workspace"
    );
}

/// c22: an `approved_grant` record sent through the socket's ComposeNote
/// (a copy of the daemon's own grant, same links) is refused and appends
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c22_forged_approved_grant_via_socket() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (_, g) = approved_grant(&d, tmp.path()).await;
    let (text, links) = text_of(&d.c, g).await;
    let (n0, _) = journal(&d.c).await;
    let out =
        d.c.send_command(ControlCommand::ComposeNote {
            kind: "authorization".into(),
            text,
            links,
        })
        .await
        .unwrap();
    assert!(
        matches!(out, ControlResponse::Error(_)),
        "ATTACK SUCCEEDED: second approved_grant noted: {out:?}"
    );
    assert_eq!(journal(&d.c).await.0, n0);
    down(d).await;
}

/// Journal level: a ComposeBridge on the compose home with the daemon down
/// writes records directly (no ComposeNote reserved-record check). This is
/// an attacker who can already write the compose home; the tests check the
/// ledger's own consistency rules, not the socket boundary.
fn bridge(tmp: &Path) -> aien_runtime::compose::ComposeBridge {
    let never: aien_runtime::compose::ComposeProposer =
        std::sync::Arc::new(|_: &str, _: Duration| Err("no model".to_string()));
    aien_runtime::compose::ComposeBridge::new(home(tmp), never, "attack")
}

fn raw_note(
    b: &aien_runtime::compose::ComposeBridge,
    kind: &str,
    text: &str,
    links: &[u64],
) -> u64 {
    match b.note_unchecked(kind, text, links) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("raw note: {other:?}"),
    }
}

/// A forged approved grant backed by a claim left in `end` (failed,
/// uncertain, uncertain-declared-committed): open_intent must refuse it.
async fn grant_on_unfinished_claim(end: &str) {
    use aien_runtime::approved_replay as rp;
    let _t = TURN.lock().await;
    let tmp = fresh();
    std::env::set_var("AIEN_COMPOSE_DIR", home(tmp.path()));
    let (g, csha, psha, target) = {
        let b = bridge(tmp.path());
        let w = canon(&ws(tmp.path()));
        let csha = sha(CONTENT.as_bytes());
        // A complete bound identity: the grant parses and its identity hashes
        // to the claim's key, so only the claim's state can refuse it.
        let ident = aien_runtime::approved_auth::ApprovalIdentity {
            trace_id: "t".into(),
            request_id: format!("req-{end}"),
            approval_id: format!("appr-{end}"),
            approver: "attacker".into(),
            path: PATH.into(),
            content_sha256: csha.clone(),
            approved_proposal_sha256: approved_proposal_sha256(PATH, CONTENT),
            desk_key_id: "forged".into(),
            workspace: w.display().to_string(),
        };
        let keys = rp::ClaimKeys {
            approval_key: aien_runtime::approved_auth::approval_key(&ident),
            request_id: ident.request_id.clone(),
            approval_id: ident.approval_id.clone(),
            trace_id: ident.trace_id.clone(),
        };
        let c = rp::claim(&b, &keys).unwrap();
        rp::mark_in_flight(&b, &c).unwrap();
        match end {
            "failed" => rp::fail(&b, &c, "attack").unwrap(),
            _ => rp::uncertain(&b, &c, "attack").unwrap(),
        }
        if end == "declared" {
            rp::declare(&b, c.id, true, "attacker", "attack").unwrap();
        }
        let target = w.join(PATH).display().to_string();
        let psha = sha(b"forged compose proposal");
        // Promotion and evidence: any two existing records, linked.
        let pr = raw_note(&b, "constraint", "{\"fake\":\"promotion\"}", &[]);
        let ev = raw_note(&b, "constraint", "{\"fake\":\"evidence\"}", &[]);
        let text = json!({
            "approved_grant": 1, "proposal_sha256": psha, "path": PATH,
            "content_sha256": csha, "approver": "attacker", "target": target,
            "workspace": w.display().to_string(), "prior_sha256": null,
            "approval_key": keys.approval_key, "replay_claim": c.id,
            "cx_promotion": pr, "cx_evidence": ev,
            "trace_id": ident.trace_id, "request_id": ident.request_id,
            "approval_id": ident.approval_id,
            "approved_proposal_sha256": ident.approved_proposal_sha256,
            "desk_key_id": ident.desk_key_id,
        });
        let g = raw_note(&b, "authorization", &text.to_string(), &[pr, ev, c.id]);
        (g, csha, psha, target)
    };
    let d = up(tmp.path(), "s.sock").await;
    let e = not_opened(
        intent(&d.c, g, &psha, PATH, &target, &csha).await,
        &format!("approved grant on a {end} claim"),
    );
    assert!(e.contains("NotAuthorized"), "{end}: {e}");
    down(d).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c23_forged_grant_on_failed_claim() {
    let _home = home_guard::home_slot();
    grant_on_unfinished_claim("failed").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c24a_forged_grant_on_uncertain_claim() {
    let _home = home_guard::home_slot();
    grant_on_unfinished_claim("uncertain").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c24b_forged_grant_on_uncertain_claim_declared_committed() {
    let _home = home_guard::home_slot();
    grant_on_unfinished_claim("declared").await;
}

/// c25 (journal level): a byte-for-byte copy of the daemon's real grant,
/// same links, appended directly. One approval must never open two effect
/// intents.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c25_duplicate_approved_grant_for_one_committed_claim() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (r, g) = approved_grant(&d, tmp.path()).await;
    let (text, links) = text_of(&d.c, g).await;
    down(d).await;
    let g2 = raw_note(&bridge(tmp.path()), "authorization", &text, &links);
    let d = up(tmp.path(), "s2.sock").await;
    let first = grant_intent(&d.c, &r, g).await;
    let second = grant_intent(&d.c, &r, g2).await;
    let both = matches!(
        (&first, &second),
        (
            ControlResponse::ComposeNoted(_),
            ControlResponse::ComposeNoted(_)
        )
    );
    eprintln!("c25: real grant #{g}: {first:?}\nc25: copy #{g2}: {second:?}");
    down(d).await;
    assert!(
        !both,
        "ATTACK SUCCEEDED (journal level): two effect intents from one approval"
    );
}

/// #306: the daemon must close its compose home before `run` returns, even
/// while a client connection is still open. Connection tasks hold their own
/// handle on the bridge and outlive `run`; if the close is left to them, a
/// successor that opens the same home right after shutdown races the close and
/// is refused with E_REPLAY (journal behind its J-Space anchor).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
async fn c25b_shutdown_closes_the_compose_home_while_a_connection_is_open() {
    let _home = home_guard::home_slot();
    let _t = TURN.lock().await;
    let tmp = fresh();
    let d = up(tmp.path(), "s.sock").await;
    let (_r, _g) = approved_grant(&d, tmp.path()).await;
    // An idle client connection that outlives the daemon's run().
    let idle = tokio::net::UnixStream::connect(&d.socket).await.unwrap();
    down(d).await;
    // Another opener on the same home must get in at once, every time.
    for i in 0..5 {
        let b = bridge(tmp.path());
        let n = raw_note(&b, "constraint", &format!("after shutdown {i}"), &[]);
        assert!(n > 0);
        drop(b);
    }
    drop(idle);
}

/// #306: once `close` has run, NO path may open the home again: a compose task,
/// a record_digest, a note and a recover all refuse, and the journal gets no
/// new bytes. (Every lazy open goes through one check.)
#[test]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
fn c25c_a_closed_bridge_never_reopens_the_home() {
    let _home = home_guard::home_slot();
    let tmp = fresh();
    let b = bridge(tmp.path());
    let first = raw_note(&b, "constraint", "before close", &[]);
    assert!(first > 0);
    let journal = home(tmp.path()).join("cortex.cx");
    let before = std::fs::read(&journal).expect("journal exists after the first note");
    b.close();
    assert!(b.is_closed());
    let refused = |what: &str, r: String| {
        assert!(r.contains("compose home closed"), "{what} not refused: {r}");
    };
    refused("record_digest", b.record_digest(first).unwrap_err());
    let ws = ws(tmp.path());
    match b.run_task("write NOTES.md", &ws.display().to_string()) {
        ControlResponse::Error(e) => refused("run_task", e),
        other => panic!("run_task after close: {other:?}"),
    }
    match b.note_unchecked("constraint", "after close", &[]) {
        ControlResponse::Error(e) => refused("note", e),
        other => panic!("note after close: {other:?}"),
    }
    match b.recover() {
        ControlResponse::Error(e) => refused("recover", e),
        other => panic!("recover after close: {other:?}"),
    }
    assert_eq!(
        std::fs::read(&journal).unwrap(),
        before,
        "the journal changed after close"
    );
}

/// Crash tests: a child process runs a whole daemon, submits one approval
/// over its socket and is aborted at a crash point inside the daemon. The
/// parent restarts a daemon over the same compose home and retries.
#[cfg(feature = "fault-hold")]
mod crash {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn crash_child_daemon() {
        let Ok(tmp) = std::env::var("ATTACK_CRASH_TMP") else {
            return;
        };
        let tmp = PathBuf::from(tmp);
        let d = up(&tmp, "child.sock").await;
        let p = signed(&tmp, "req-x", "appr-x");
        let out = submit(&d.c, &p, &ws(&tmp)).await;
        panic!("child was not stopped at its crash point: {out:?}");
    }

    fn run_child(tmp: &Path, hold: &str) {
        use std::os::unix::process::ExitStatusExt;
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "crash::crash_child_daemon",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("AIEN_FAULT_HOLD", hold)
            .env("ATTACK_CRASH_TMP", tmp)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert_eq!(st.signal(), Some(6), "child must abort at {hold}: {st:?}");
    }

    async fn after_crash(hold: &str, want_phase: &str, want_refusal: &str) {
        let _t = TURN.lock().await;
        let tmp = fresh();
        run_child(tmp.path(), hold);
        let d = up(tmp.path(), "after.sock").await;
        let (n0, texts) = journal(&d.c).await;
        assert_eq!(phase_count(&texts, "accepted"), 1, "the child claimed once");
        assert_eq!(
            phase_count(&texts, want_phase),
            1,
            "restart reconcile: {texts:?}"
        );
        let w = ws(tmp.path());
        let p = signed(tmp.path(), "req-x", "appr-x");
        let r = submit(&d.c, &p, &w).await;
        let (by, name) = refused(r, &format!("retry after crash at {hold}"));
        assert_eq!(
            (by.as_str(), name.as_str()),
            ("REPLAY_REFUSED", want_refusal)
        );
        let (n1, texts) = journal(&d.c).await;
        assert_eq!(n1, n0, "a refused retry appended records");
        assert_eq!(phase_count(&texts, "committed"), 0);
        assert!(!w.join(PATH).exists());
        down(d).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[cfg_attr(
        not(compose_linked),
        ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
    )]
    async fn c12_crash_before_execution() {
        after_crash("approved_after_claim", "not_executed", "AlreadyConsumed").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[cfg_attr(
        not(compose_linked),
        ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
    )]
    async fn c13_crash_after_execution_started() {
        after_crash("approved_after_compose", "uncertain", "Uncertain").await;
    }

    /// c27: crash after the claim is COMMITTED, before the grant is written.
    /// The approval stays spent and no grant ever appears: no effect possible.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[cfg_attr(
        not(compose_linked),
        ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
    )]
    async fn c27_crash_after_commit_before_grant() {
        let _t = TURN.lock().await;
        let tmp = fresh();
        run_child(tmp.path(), "approved_after_commit");
        let d = up(tmp.path(), "after.sock").await;
        let (n0, texts) = journal(&d.c).await;
        assert_eq!(phase_count(&texts, "committed"), 1, "{texts:?}");
        let grants = |t: &[String]| {
            t.iter()
                .filter(|x| x.contains("\"approved_grant\""))
                .count()
        };
        assert_eq!(grants(&texts), 0, "a grant was written before the crash");
        let w = ws(tmp.path());
        let r = submit(&d.c, &signed(tmp.path(), "req-x", "appr-x"), &w).await;
        match &r {
            ControlResponse::ComposeApprovedResult(rep) => {
                assert_eq!(rep.state, "ALREADY_COMMITTED");
                assert_eq!(
                    rep.approved_grant, None,
                    "ATTACK SUCCEEDED: retry minted a grant"
                );
            }
            ControlResponse::ComposeApprovedRefused(_) => {}
            other => panic!("retry: {other:?}"),
        }
        let (n1, texts) = journal(&d.c).await;
        assert_eq!(
            grants(&texts),
            0,
            "ATTACK SUCCEEDED: a grant appeared after restart"
        );
        assert_eq!(n1, n0, "the retry appended records");
        assert!(!w.join(PATH).exists());
        eprintln!("c27: retry -> {r:?}");
        down(d).await;
    }
}
