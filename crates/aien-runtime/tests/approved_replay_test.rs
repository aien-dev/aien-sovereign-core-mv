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

//! Replay ledger for approved proposals (crate::approved_replay, sovereign-core #249 part B).
//! Pure ledger rules run in every build; the journal tests need the linked
//! composition library and are IGNORED in a stub build (cfg compose_linked, set by
//! build.rs from aien-omega-compose's links metadata).
//! A "crashed" claim is made by a child process (this test binary re-run with
//! REPLAY_CHILD set) that claims and exits without settling.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::approved_replay::{
    self as replay, ClaimKeys, ClaimState, CommitEvidence, ReplayLedger, FIELD,
};
use aien_runtime::compose::{ComposeBridge, ComposeProposer};
use aien_runtime::control::ComposeRecordView;
use aien_runtime::effects;
use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Barrier};

fn proposer() -> ComposeProposer {
    Arc::new(|_: &str, _: std::time::Duration| panic!("no model on the replay path"))
}

fn bridge(dir: &Path) -> ComposeBridge {
    ComposeBridge::new(dir.join("compose"), proposer(), "test:replay")
}

fn keys(n: &str) -> ClaimKeys {
    let k = format!(
        "{:0>64}",
        n.bytes().map(|b| format!("{:02x}", b)).collect::<String>()
    );
    ClaimKeys {
        approval_key: k[k.len() - 64..].to_string(),
        request_id: format!("req-{n}"),
        approval_id: format!("appr-{n}"),
        trace_id: format!("trace-{n}"),
    }
}

fn evidence() -> CommitEvidence {
    CommitEvidence {
        compose_proposal_sha256: "ab".repeat(32),
        cx_promotion: 17,
        cx_evidence: 15,
        task: 99,
    }
}

fn refused_as<T: std::fmt::Debug>(r: Result<T, replay::Refusal>, name: &str) -> replay::Refusal {
    match r {
        Err(e) => {
            assert_eq!(e.name, name, "want {name}, got {e}");
            assert!(e
                .to_string()
                .starts_with(&format!("REPLAY_REFUSED {name}:")));
            e
        }
        Ok(v) => panic!("want REPLAY_REFUSED {name}, got {v:?}"),
    }
}

// ---------- pure ledger rules (every build) ----------

fn view(id: u64, text: serde_json::Value) -> ComposeRecordView {
    ComposeRecordView {
        id,
        cls: 0,
        kind: 0,
        subject: 0,
        tag: 4,
        links: vec![],
        digest: String::new(),
        verified: true,
        note: Some("effect".into()),
        text: Some(text.to_string()),
    }
}

fn accepted(id: u64, k: &ClaimKeys) -> ComposeRecordView {
    // pid 0: an executor that is never alive (effects::executor_alive).
    view(
        id,
        json!({FIELD: "accepted", "approval_key": k.approval_key, "request_id": k.request_id,
               "approval_id": k.approval_id, "trace_id": k.trace_id,
               "executor": {"pid": 0, "start": 0}}),
    )
}

fn step(id: u64, claim: u64, phase: &str) -> ComposeRecordView {
    view(
        id,
        json!({FIELD: phase, "claim": claim, "reason": "test", "evidence": evidence()}),
    )
}

#[test]
fn ledger_lifecycle_and_refusal_names() {
    let k = keys("a");
    let cases: Vec<(Vec<ComposeRecordView>, ClaimState, &str)> = vec![
        (
            vec![accepted(1, &k)],
            ClaimState::Accepted,
            "AlreadyConsumed",
        ),
        (
            vec![accepted(1, &k), step(2, 1, "in_flight")],
            ClaimState::InFlight,
            "Uncertain",
        ),
        (
            vec![
                accepted(1, &k),
                step(2, 1, "in_flight"),
                step(3, 1, "committed"),
            ],
            ClaimState::Committed,
            "AlreadyCommitted",
        ),
        (
            vec![
                accepted(1, &k),
                step(2, 1, "in_flight"),
                step(3, 1, "failed"),
            ],
            ClaimState::Failed,
            "AlreadyFailed",
        ),
        (
            vec![
                accepted(1, &k),
                step(2, 1, "in_flight"),
                step(3, 1, "uncertain"),
            ],
            ClaimState::Uncertain,
            "Uncertain",
        ),
        (
            vec![accepted(1, &k), step(2, 1, "not_executed")],
            ClaimState::NotExecuted,
            "AlreadyConsumed",
        ),
    ];
    for (recs, state, name) in cases {
        let l = ReplayLedger::from_records(&recs).unwrap();
        assert_eq!(l.claims[&1].state, state);
        let r = l.check(&k).expect("claimed keys are refused");
        assert_eq!(r.name, name, "{state:?}");
        assert_eq!(r.claim, Some(1));
        assert_eq!(r.evidence.is_some(), state == ClaimState::Committed);
    }
}

#[test]
fn ledger_each_key_is_claimed_on_its_own() {
    let k = keys("a");
    let l = ReplayLedger::from_records(&[accepted(1, &k)]).unwrap();
    let mut only_request = keys("b");
    only_request.request_id = k.request_id.clone();
    let mut only_approval = keys("c");
    only_approval.approval_id = k.approval_id.clone();
    let mut only_key = keys("d");
    only_key.approval_key = k.approval_key.clone();
    for (what, other) in [
        ("request", only_request),
        ("approval", only_approval),
        ("key", only_key),
    ] {
        assert!(l.check(&other).is_some(), "same {what} id must be refused");
    }
    assert!(l.check(&keys("e")).is_none(), "fresh keys are free");
}

#[test]
fn ledger_refuses_corrupt_histories() {
    let k = keys("a");
    let bad: Vec<Vec<ComposeRecordView>> = vec![
        vec![accepted(1, &k), accepted(2, &k)], // a key claimed twice
        vec![accepted(1, &k), step(2, 1, "committed")], // committed without in_flight
        vec![
            accepted(1, &k),
            step(2, 1, "in_flight"),
            step(3, 1, "not_executed"),
        ],
        vec![
            accepted(1, &k),
            step(2, 1, "in_flight"),
            step(3, 1, "committed"),
            step(4, 1, "failed"),
        ],
        vec![step(2, 1, "in_flight")], // names no claim
        vec![accepted(1, &k), step(2, 1, "bogus")],
    ];
    for recs in bad {
        let e = ReplayLedger::from_records(&recs).unwrap_err();
        assert_eq!(e.name, "CorruptLedger", "{e}");
    }
    let mut unverified = accepted(1, &k);
    unverified.verified = false;
    assert_eq!(
        ReplayLedger::from_records(&[unverified]).unwrap_err().name,
        "CorruptLedger"
    );
}

#[test]
fn compose_note_cannot_forge_replay_records() {
    for phase in ["accepted", "in_flight", "committed", "declared"] {
        let t = json!({FIELD: phase, "claim": 1}).to_string();
        assert!(
            effects::check_reserved_note("effect", &t).is_err(),
            "{phase}"
        );
    }
}

// ---------- journal (linked composition library) ----------

#[test]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
fn claim_is_durable_and_single_use_across_restart() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let k = keys("restart");
    {
        let b = bridge(tmp.path());
        let c = replay::claim(&b, &k).unwrap();
        // Same process, same keys: the running claim refuses.
        refused_as(replay::claim(&b, &k), "InFlight");
        replay::mark_in_flight(&b, &c).unwrap();
        replay::commit(&b, &c, &evidence()).unwrap();
        // Lifecycle cannot go backwards or repeat.
        refused_as(replay::commit(&b, &c, &evidence()), "WrongState");
        refused_as(replay::fail(&b, &c, "x"), "WrongState");
    }
    // A new bridge on the same compose home = a daemon restart.
    let b = bridge(tmp.path());
    assert!(replay::reconcile_at_start(&b)
        .unwrap()
        .contains("0 -> NOT_EXECUTED"));
    let e = refused_as(replay::claim(&b, &k), "AlreadyCommitted");
    assert_eq!(
        e.evidence,
        Some(evidence()),
        "a lost response can be recovered"
    );
    for (what, changed) in [
        (
            "request",
            ClaimKeys {
                approval_key: keys("x").approval_key,
                approval_id: "appr-x".into(),
                ..k.clone()
            },
        ),
        (
            "approval",
            ClaimKeys {
                approval_key: keys("y").approval_key,
                request_id: "req-y".into(),
                ..k.clone()
            },
        ),
        (
            "key",
            ClaimKeys {
                request_id: "req-z".into(),
                approval_id: "appr-z".into(),
                ..k.clone()
            },
        ),
    ] {
        refused_as(replay::claim(&b, &changed), "AlreadyCommitted");
        let _ = what;
    }
    // A fresh approval is accepted.
    replay::claim(&b, &keys("fresh")).unwrap();
}

#[test]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
fn failed_and_not_executed_claims_stay_consumed() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let b = bridge(tmp.path());
    let c = replay::claim(&b, &keys("ne")).unwrap();
    replay::not_executed(&b, &c, "authentication refused after claim").unwrap();
    refused_as(replay::claim(&b, &keys("ne")), "AlreadyConsumed");
    let c = replay::claim(&b, &keys("f")).unwrap();
    replay::mark_in_flight(&b, &c).unwrap();
    replay::fail(&b, &c, "World did not commit").unwrap();
    refused_as(replay::claim(&b, &keys("f")), "AlreadyFailed");
    let c = replay::claim(&b, &keys("u")).unwrap();
    replay::mark_in_flight(&b, &c).unwrap();
    replay::uncertain(&b, &c, "run returned an error").unwrap();
    refused_as(replay::claim(&b, &keys("u")), "Uncertain");
    replay::declare(&b, c.id, false, "operator", "World checked: no commit").unwrap();
    refused_as(replay::claim(&b, &keys("u")), "Uncertain");
    refused_as(
        replay::declare(&b, c.id, true, "operator", "again"),
        "WrongState",
    );
}

#[test]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
fn concurrent_duplicates_at_most_one_claim() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let b = Arc::new(bridge(tmp.path()));
    let n = 16;
    let gate = Arc::new(Barrier::new(n));
    let hs: Vec<_> = (0..n)
        .map(|i| {
            let (b, gate) = (b.clone(), gate.clone());
            std::thread::spawn(move || {
                // Half the threads reuse only the request id, half the whole keys.
                let mut k = keys("race");
                if i % 2 == 1 {
                    k.approval_key = keys(&format!("r{i}")).approval_key;
                    k.approval_id = format!("appr-r{i}");
                }
                gate.wait();
                replay::claim(&b, &k)
            })
        })
        .collect();
    let ok = hs
        .into_iter()
        .map(|h| h.join().unwrap())
        .filter(|r| r.is_ok())
        .count();
    assert_eq!(ok, 1, "exactly one concurrent claim wins");
}

/// Child mode: claim (and optionally go in flight), then exit without settling.
/// Run only by crash_child.
#[test]
#[ignore = "child mode, run by crash_before_and_during_execution_fail_closed"]
fn child_claim_then_crash() {
    let _home = home_guard::home_slot();
    let Ok(mode) = std::env::var("REPLAY_CHILD") else {
        return;
    };
    let dir = std::env::var("REPLAY_DIR").unwrap();
    let b = bridge(Path::new(&dir));
    let c = replay::claim(&b, &keys(&mode)).unwrap();
    if mode == "inflight" {
        replay::mark_in_flight(&b, &c).unwrap();
    }
    std::process::exit(0);
}

fn crash_child(dir: &Path, mode: &str) {
    let st = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_claim_then_crash",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("REPLAY_CHILD", mode)
        .env("REPLAY_DIR", dir)
        .status()
        .unwrap();
    assert!(st.success(), "child {mode}: {st}");
}

#[test]
#[cfg_attr(
    not(compose_linked),
    ignore = "needs the linked librx_compose.a (AIEN_OMEGA_COMPOSE_LIB)"
)]
fn crash_before_and_during_execution_fail_closed() {
    let tmp = tempfile::tempdir().unwrap();
    crash_child(tmp.path(), "before");
    crash_child(tmp.path(), "inflight");
    let b = bridge(tmp.path());
    // Before the start-up reconcile: a gone process is never "in flight", and
    // neither claim can be taken again.
    refused_as(replay::claim(&b, &keys("before")), "AlreadyConsumed");
    refused_as(replay::claim(&b, &keys("inflight")), "Uncertain");
    let s = replay::reconcile_at_start(&b).unwrap();
    assert!(
        s.contains("1 -> NOT_EXECUTED") && s.contains("1 -> UNCERTAIN"),
        "{s}"
    );
    refused_as(replay::claim(&b, &keys("before")), "AlreadyConsumed");
    refused_as(replay::claim(&b, &keys("inflight")), "Uncertain");
    // The reconcile is idempotent: nothing left to settle.
    let s = replay::reconcile_at_start(&b).unwrap();
    assert!(
        s.contains("0 -> NOT_EXECUTED") && s.contains("0 -> UNCERTAIN"),
        "{s}"
    );
}
