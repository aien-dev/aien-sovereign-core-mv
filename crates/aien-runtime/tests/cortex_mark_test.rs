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

//! NEXT-PHASE-2 v3: the external Cortex record mark, RecoverComposeHome
//! keeping the old mark, the reserved repair key, and the reconcile-failed
//! refusal, through the real compose home (ACCEPTANCE-v3 section 2).
//!
//! With librx_compose.a linked this drives the bridge as the daemon does (no
//! socket); in a stub build it only checks the refusal.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ComposeNoteReport, ControlResponse};
use aien_runtime::cortex_mark::{self, Mark};
use aien_runtime::effects::{self, IntentRequest};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// rxc_host allows 4 open handles per process (RXC_HOST_MAX_HANDLES): the
/// tests that open homes run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn proposer() -> ComposeProposer {
    Arc::new(|_prompt: &str, _limit: std::time::Duration| {
        Ok(Generation {
            text: "filename: NOTES.md\nnote\n".into(),
            tokens: 4,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

fn noted(r: ControlResponse) -> ComposeNoteReport {
    match r {
        ControlResponse::ComposeNoted(n) => n,
        other => panic!("expected ComposeNoted, got {other:?}"),
    }
}

fn total(b: &ComposeBridge) -> u64 {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => r.records_total,
        other => panic!("recall: {other:?}"),
    }
}

fn recall_error(b: &ComposeBridge) -> String {
    match b.recall(&[], None) {
        ControlResponse::Error(e) => e,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn mark_of(home: &Path) -> Mark {
    cortex_mark::read(&cortex_mark::mark_path(home))
        .unwrap()
        .expect("mark present")
}

fn grant(b: &ComposeBridge, n: u32) -> u64 {
    // Any host record will do: only the journal length matters here, and a
    // caller can no longer write an authorization (sovereign-core #261).
    noted(b.note(
        "constraint",
        &json!({"approver": "drake", "n": n}).to_string(),
        &[],
    ))
    .id
}

/// A home with history; returns (tempdir, home dir, records after setup).
fn setup() -> Option<(tempfile::TempDir, PathBuf)> {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let r = b.note("constraint", "keep changes inside the workspace", &[]);
    if !aien_omega_compose::LINKED {
        assert!(matches!(r, ControlResponse::Error(_)));
        return None;
    }
    noted(r);
    // New home: a mark exists and matches the journal.
    let n = total(&b);
    let m = mark_of(&home);
    assert_eq!(m.records, n);
    assert!(m.seq >= 1);
    Some((tmp, home))
}

#[test]
fn boundary_cut_is_refused_and_recover_keeps_the_old_mark() {
    let _home = home_guard::home_slot();
    let _serial = serial();
    let Some((_tmp, home)) = setup() else { return };
    let cx = home.join("cortex.cx");
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    grant(&b, 1);
    let size_before_a2 = std::fs::metadata(&cx).unwrap().len();
    grant(&b, 2);
    let m = mark_of(&home);
    assert_eq!(m.records, total(&b));
    drop(b);

    // Cut exactly at the record boundary before grant 2.
    let f = std::fs::OpenOptions::new().write(true).open(&cx).unwrap();
    f.set_len(size_before_a2).unwrap();
    drop(f);
    let damaged = std::fs::read(&cx).unwrap();
    let mark_bytes = std::fs::read(cortex_mark::mark_path(&home)).unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let e = recall_error(&b);
    assert!(e.contains("E_MARK_TRUNCATED"), "{e}");
    assert!(e.contains("1 record(s) lost"), "{e}");
    // A refusal touches nothing.
    assert_eq!(std::fs::read(&cx).unwrap(), damaged);
    assert_eq!(
        std::fs::read(cortex_mark::mark_path(&home)).unwrap(),
        mark_bytes
    );

    // Recover: the old mark is kept, a repair record names the loss.
    let rep = match b.recover() {
        ControlResponse::ComposeRecovered(r) => *r,
        other => panic!("recover: {other:?}"),
    };
    assert_eq!(rep.mark_lost, 1, "{rep:?}");
    let kept = PathBuf::from(rep.mark_kept_as.clone().unwrap());
    assert!(kept
        .to_string_lossy()
        .ends_with(&format!(".cortex-mark.lost-{}", m.seq)));
    assert_eq!(std::fs::read(&kept).unwrap(), mark_bytes);
    assert!(rep.mark_repair_record > 0);
    let r = match b.recall(&[rep.mark_repair_record], None) {
        ControlResponse::ComposeRecalled(r) => *r,
        other => panic!("recall after recover: {other:?}"),
    };
    let t: Value = serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap();
    assert_eq!(t["repair"], "cortex-mark");
    assert_eq!(t["lost"], 1);
    assert_eq!(r.cited[0].note.as_deref(), Some("constraint"));
    let fresh = mark_of(&home);
    assert_eq!(fresh.records, r.records_total);
    assert!(fresh.seq > m.seq);
    drop(b);
    // Opens again cleanly.
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    total(&b);
}

#[test]
fn crash_between_append_and_mark_is_not_corruption() {
    let _home = home_guard::home_slot();
    let _serial = serial();
    let Some((_tmp, home)) = setup() else { return };
    let mp = cortex_mark::mark_path(&home);
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    total(&b);
    let before = std::fs::read(&mp).unwrap();
    let m0 = mark_of(&home);
    grant(&b, 1);
    drop(b);
    // The on-disk state of a process killed after the append, before its
    // mark update: the journal is one record longer than the mark.
    std::fs::write(&mp, &before).unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let n = total(&b);
    let m1 = mark_of(&home);
    assert_eq!(m1.records, n);
    assert!(m1.seq > m0.seq);
}

#[test]
fn digest_mismatch_and_damaged_mark_are_refused() {
    let _home = home_guard::home_slot();
    let _serial = serial();
    let Some((_tmp, home)) = setup() else { return };
    let mp = cortex_mark::mark_path(&home);
    let mut m = mark_of(&home);
    m.digest = [0x5a; 32];
    cortex_mark::write(&mp, &m).unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let e = recall_error(&b);
    assert!(e.contains("E_MARK_DIGEST"), "{e}");
    drop(b);

    std::fs::write(&mp, b"not a mark").unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let e = recall_error(&b);
    assert!(e.contains("(E_MARK)"), "{e}");
    let rep = match b.recover() {
        ControlResponse::ComposeRecovered(r) => *r,
        other => panic!("recover: {other:?}"),
    };
    let kept = rep.mark_kept_as.unwrap();
    assert!(kept.ends_with(".cortex-mark.lost-damaged"), "{kept}");
    assert_eq!(std::fs::read(&kept).unwrap(), b"not a mark");
    total(&b);
}

#[test]
fn missing_mark_is_adopted_not_refused() {
    let _home = home_guard::home_slot();
    let _serial = serial();
    let Some((_tmp, home)) = setup() else { return };
    let mp = cortex_mark::mark_path(&home);
    std::fs::remove_file(&mp).unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let n = total(&b);
    let m = mark_of(&home);
    assert_eq!((m.records, m.seq), (n, 1));
}

#[test]
fn repair_key_is_reserved() {
    assert!(effects::check_reserved_note("constraint", r#"{"repair":"cortex-mark"}"#).is_err());
    assert!(effects::check_reserved_note("constraint", r#"{"note":"x"}"#).is_ok());
    assert!(effects::check_reserved_note("constraint", "plain text").is_ok());
}

#[test]
fn effect_commands_refuse_until_reconcile_succeeds() {
    let _home = home_guard::home_slot();
    let _serial = serial();
    let Some((tmp, home)) = setup() else { return };
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    b.set_reconcile_failed("failed: test".into());
    let req = IntentRequest {
        authorization: 1,
        proposal_sha256: "00".into(),
        path: "NOTES.md".into(),
        target: tmp.path().join("NOTES.md").display().to_string(),
        content_sha256: "00".into(),
        executor_pid: 0,
        executor_start: 0,
    };
    let refused = |r: ControlResponse| match r {
        ControlResponse::Error(e) => e,
        other => panic!("expected an error, got {other:?}"),
    };
    let e = refused(effects::open_intent(&b, &req));
    assert!(e.starts_with("EFFECT_REFUSED ReconcileFailed:"), "{e}");
    let e = refused(effects::ack(&b, 1, &json!({})));
    assert!(e.starts_with("EFFECT_REFUSED ReconcileFailed:"), "{e}");
    // Authorize and recall stay available.
    grant(&b, 1);
    total(&b);
    assert!(matches!(
        effects::reconcile(&b, None, "test"),
        ControlResponse::ComposeReconciled(_)
    ));
    assert_eq!(b.reconcile_failed(), None);
    let e = refused(effects::open_intent(&b, &req));
    assert!(!e.contains("ReconcileFailed"), "{e}");
}
