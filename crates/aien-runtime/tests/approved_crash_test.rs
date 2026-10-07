//! #249 B crash boundaries of the approved-proposal path, with a real process
//! crash: a child process (this test binary, re-run on the hidden helper
//! test) submits one approved proposal and aborts at a fault-hold point; the
//! parent then opens the same compose home as a restarted daemon would
//! (start-up replay reconcile) and retries.
//!
//!   approved_after_claim    boundary 1: accepted, dies before compose
//!   approved_after_compose  boundary 2: the World committed, dies before the
//!                           replay commit record
//!
//! Needs the `fault-hold` feature (test builds only):
//! `cargo test -p aien-runtime --features fault-hold --test approved_crash_test`.
#![cfg(feature = "fault-hold")]
use aien_omega_compose::hex;
use aien_runtime::approved::{ApprovedProposal, ProposerHook};
use aien_runtime::approved_auth::{desk_key_path, DeskKey};
use aien_runtime::approved_replay;
use aien_runtime::control::{ComposeRecallReport, ControlResponse};
use aien_runtime::effects::Ledger;
use aien_runtime::spine::{ComposeBridge, ComposeProposer};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

const PATH: &str = "NOTES.md";
const CONTENT: &str = "approved by the INTERPLANE desk\n";

fn never() -> ComposeProposer {
    Arc::new(|_: &str, _: std::time::Duration| panic!("model proposer ran on the hook path"))
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn proposal(b: &ComposeBridge) -> ApprovedProposal {
    let mut p = ApprovedProposal {
        request_id: "req-crash".into(),
        trace_id: "trace-crash".into(),
        approval_id: "appr-crash".into(),
        approver: "interplane-host".into(),
        path: PATH.into(),
        content: CONTENT.into(),
        approved_proposal_sha256: aien_runtime::approved::approved_proposal_sha256(PATH, CONTENT),
        content_sha256: sha(CONTENT.as_bytes()),
        approval_mac: String::new(),
    };
    p.approval_mac = DeskKey::load(&desk_key_path(b.dir())).unwrap().sign(&p);
    p
}

fn recalled(b: &ComposeBridge) -> ComposeRecallReport {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => *r,
        other => panic!("recall: {other:?}"),
    }
}

/// Hidden helper: the child. Does nothing unless the parent set the env.
#[test]
#[ignore]
fn crash_child() {
    let (Ok(dir), Ok(ws)) = (std::env::var("CRASH_DIR"), std::env::var("CRASH_WS")) else {
        return;
    };
    let b = Arc::new(ComposeBridge::new(dir.into(), never(), "t"));
    let hook = ProposerHook::new(b.clone());
    let out = hook.submit(&proposal(&b), &ws);
    // Reaching here means the hold point did not fire.
    panic!("child was not stopped at its hold point: {out:?}");
}

fn run_child(hold: &str, dir: &Path, ws: &Path) {
    let st = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "crash_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("AIEN_FAULT_HOLD", hold)
        .env("CRASH_DIR", dir)
        .env("CRASH_WS", ws)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(
        st.signal(),
        Some(libc_sigabrt()),
        "child must abort at {hold}: {st:?}"
    );
}

fn libc_sigabrt() -> i32 {
    6
}

/// Replay claims (`accepted` records) in the journal.
fn claims(r: &ComposeRecallReport) -> usize {
    r.host
        .iter()
        .filter(|h| {
            h.text
                .as_deref()
                .is_some_and(|t| t.contains("\"approved_submission\":\"accepted\""))
        })
        .count()
}

static HOMES: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn boundary(hold: &str, want_reconciled: &str, want_refusal: &str) {
    let _turn = HOMES.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let dir = tmp.path().join("compose");
    DeskKey::create(&desk_key_path(&dir)).unwrap();
    run_child(hold, &dir, &ws);

    // The restarted daemon: a fresh bridge, start-up reconcile first.
    let b = Arc::new(ComposeBridge::new(dir.clone(), never(), "t"));
    let line = approved_replay::reconcile_at_start(&b).unwrap();
    assert!(line.contains(want_reconciled), "{line}");
    let before = recalled(&b);
    assert_eq!(claims(&before), 1, "the child claimed once");
    // Any compose run appends goal, candidate, evidence and promotion
    // records: an unchanged total proves nothing ran again.
    let total = before.records_total;

    // Retry of the same approval and replays under either id: refused, never run.
    let hook = ProposerHook::new(b.clone());
    let p = proposal(&b);
    let e = hook.submit(&p, ws.to_str().unwrap()).unwrap_err();
    assert_eq!(e.name, want_refusal, "{e}");
    assert_eq!(e.refused_by, "REPLAY_REFUSED");
    let mut q = p.clone();
    q.request_id = "req-crash-2".into();
    q.approval_mac = DeskKey::load(&desk_key_path(&dir)).unwrap().sign(&q);
    assert_eq!(
        hook.submit(&q, ws.to_str().unwrap()).unwrap_err().name,
        want_refusal
    );
    let after = recalled(&b);
    assert_eq!(
        after.records_total, total,
        "a refused retry appended records"
    );
    // No effect: no grant, no intent, nothing on disk.
    let l = Ledger::from_records(&after.host).unwrap();
    assert!(l.grants.is_empty() && l.intents.is_empty());
    assert!(!ws.join(PATH).exists());
}

/// Boundary 1: accepted, the process dies before compose. After restart the
/// claim is NOT_EXECUTED and the approval stays consumed (a new approval is
/// needed); nothing ran.
#[test]
fn crash_after_claim_is_consumed_never_run() {
    if !aien_omega_compose::LINKED {
        return;
    }
    boundary(
        "approved_after_claim",
        "1 -> NOT_EXECUTED",
        "AlreadyConsumed",
    );
}

/// Boundary 2: the World committed, the process dies before the replay commit
/// record. After restart the claim is UNCERTAIN (recorded), every retry is
/// refused: no second World commit, no effect.
#[test]
fn crash_after_compose_is_uncertain_never_rerun() {
    if !aien_omega_compose::LINKED {
        return;
    }
    boundary("approved_after_compose", "1 -> UNCERTAIN", "Uncertain");
}
