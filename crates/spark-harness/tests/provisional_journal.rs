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

//! Headstart M2 (#535): the scheduler journal survives restarts, ignores a
//! torn final line, refuses corruption, and replay never creates a
//! committable task without re-verification.

mod common;

use common::{enabled, H};
use spark_harness::execution::TaskState;
use spark_harness::scheduler::{EventKind, ProvisionalConfig, SchedError, Scheduler};
use std::fs::{self, OpenOptions};
use std::io::Write;

fn scenario(h: &mut H) {
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.register("c", 1, &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.advance(40);
    h.work_done("c");
    h.publish("a", 2); // interface change: b and c revoked, re-dispatched
    h.publish("b", 1);
    h.finish("a");
}

#[test]
fn replay_reproduces_state_and_trace_exactly() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let snap = h.sched.snapshot();
    let trace = h.sched.timing_trace();
    // Move the clock: replay must use journaled times, not the clock.
    h.advance(500);
    assert_eq!(h.reopen(enabled()).unwrap(), (0, vec![]));
    assert_eq!(h.sched.snapshot(), snap);
    assert_eq!(h.sched.timing_trace(), trace);
    h.sched.check_invariants().unwrap();
    // The replayed scheduler keeps working and keeps journaling.
    h.work_done("b");
    h.finalize("b");
    let snap2 = h.sched.snapshot();
    h.reopen(enabled()).unwrap();
    assert_eq!(h.sched.snapshot(), snap2);
    assert_eq!(h.sched.state("b"), Some(TaskState::Committable));
}

#[test]
fn torn_last_line_is_ignored_and_truncated() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let snap = h.sched.snapshot();
    let path = h.fx.journal();
    let before = fs::read(&path).unwrap();
    // Simulate a crash mid-append of the next event.
    let torn = br#"{"seq":999,"id":"ev-torn","at":1,"kind":{"type":"fin"#;
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(torn)
        .unwrap();
    assert_eq!(h.reopen(enabled()).unwrap().0, torn.len());
    assert_eq!(h.sched.snapshot(), snap);
    assert_eq!(fs::read(&path).unwrap(), before, "torn tail truncated");
    // Next append starts on a clean line and replays.
    h.work_done("b");
    let snap2 = h.sched.snapshot();
    assert_eq!(h.reopen(enabled()).unwrap().0, 0);
    assert_eq!(h.sched.snapshot(), snap2);
}

#[test]
fn version_1_journal_is_refused() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let path = h.fx.journal();
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.lines().next().unwrap().contains("\"version\":2"));
    let old = text.replacen("\"version\":2", "\"version\":1", 1);
    fs::write(&path, &old).unwrap();
    let r = h.reopen(enabled());
    assert!(
        matches!(r, Err(SchedError::Journal(ref m)) if m.contains("header mismatch")),
        "{r:?}"
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), old);
}

#[test]
fn refused_journal_is_not_truncated() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let path = h.fx.journal();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(br#"{"seq":999,"id":"ev-torn""#)
        .unwrap();
    let before = fs::read(&path).unwrap();
    let other = ProvisionalConfig {
        lease_ttl: 7,
        ..enabled()
    };
    assert!(matches!(h.reopen(other), Err(SchedError::Journal(_))));
    assert_eq!(
        fs::read(&path).unwrap(),
        before,
        "refused journal untouched"
    );
}

#[test]
fn restart_rebases_the_clock_so_dead_leases_expire() {
    let mut h = H::journaled(enabled());
    h.advance(1_000_000); // the old process ran for a long time
    h.register("a", 2, &[]);
    assert_eq!(h.sched.state("a"), Some(TaskState::Provisional));
    let lease_ends = h.sched.task("a").unwrap().lease_expires_at.unwrap();
    // The worker dies with the process. The new process's clock starts at 0.
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.permits_in_use().cpu_in_use, 2);
    let last = lease_ends - enabled().lease_ttl;
    h.advance(enabled().lease_ttl - 11);
    let fx = h.ok(EventKind::Tick); // at lease end - 1
    assert!(fx.aborted.is_empty(), "{fx:?}");
    let fx = h.ok(EventKind::Tick);
    assert_eq!(fx.aborted, vec!["a"]);
    assert_eq!(h.sched.permits_in_use().running, 0);
    assert!(h.sched.timing_trace()["as_of"].as_u64().unwrap() > last);
}

#[test]
fn forged_finalize_in_the_journal_fails_replay() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.work_done("b");
    // a is not final, so no Finalize for b could have been journaled.
    let path = h.fx.journal();
    let text = fs::read_to_string(&path).unwrap();
    let last: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    let forged = serde_json::json!({
        "seq": last["seq"].as_u64().unwrap() + 1,
        "id": "forged",
        "at": last["at"],
        "kind": { "type": "finalize", "task": "b", "generation": 1 },
    });
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(f, "{forged}").unwrap();
    drop(f);
    let r = h.reopen(enabled());
    assert!(matches!(r, Err(SchedError::Journal(_))), "{r:?}");
}

#[test]
fn corrupt_complete_line_fails_closed() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let path = h.fx.journal();
    let text = fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    lines[3] = "{\"not\":\"an event\"}";
    fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
    let r = h.reopen(enabled());
    assert!(matches!(r, Err(SchedError::Journal(_))), "{r:?}");
}

#[test]
fn reordered_or_duplicated_lines_fail_replay() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let path = h.fx.journal();
    let text = fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    let dup = lines[2];
    lines.insert(3, dup);
    fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
    let r = h.reopen(enabled());
    assert!(matches!(r, Err(SchedError::Journal(_))), "{r:?}");
}

#[test]
fn config_mismatch_is_refused() {
    let mut h = H::journaled(enabled());
    scenario(&mut h);
    let other = ProvisionalConfig {
        enabled: false,
        ..enabled()
    };
    let r = h.reopen(other);
    assert!(matches!(r, Err(SchedError::Journal(_))), "{r:?}");
}

#[test]
fn journal_has_a_single_writer() {
    let h = H::journaled(enabled());
    let second = Scheduler::open(
        enabled(),
        &h.engine.completion_store,
        &h.fx.journal(),
        common::clock(&h.ticks),
    );
    assert!(matches!(second, Err(SchedError::Journal(_))));
}

#[test]
fn replay_never_turns_provisional_work_committable() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.work_done("b");
    // Crash before a is final: b replays as AwaitingFinalProof, not more.
    h.reopen(enabled()).unwrap();
    assert_eq!(h.sched.state("b"), Some(TaskState::AwaitingFinalProof));
    assert!(h.sched.may_finalize("b").is_err());
    assert!(h
        .send(EventKind::Finalize {
            task: "b".into(),
            generation: h.generation("b")
        })
        .is_err());
    assert!(h.sched.audit().unwrap().is_empty());
}

#[test]
fn replayed_committable_task_is_reverified_against_the_store() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.work_done("b");
    h.finish("a");
    h.finalize("b");
    assert!(h.sched.may_finalize("b").is_ok());
    let a_completion = h.sched.task("a").unwrap().completion.clone().unwrap();
    // The store loses a's completion receipt while the scheduler is down.
    fs::remove_file(
        h.engine
            .completion_store
            .join("receipts")
            .join(format!("{}.json", a_completion.receipt_id)),
    )
    .unwrap();
    // Replay is pure, so a and b come back Committable; `open` hands out
    // only an unaudited handle, and its audit revokes both before any event
    // or dispatch can run on them.
    let (_, revoked) = h.reopen(enabled()).unwrap();
    assert_eq!(revoked, vec!["a", "b"]);
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
    // A consumer registered after the restart is not dispatched on a.
    h.register("c", 1, &["a"]);
    assert_eq!(h.sched.state("c"), Some(TaskState::Blocked));
    assert!(h.sched.may_finalize("b").is_err());
    // The revocation itself is journaled.
    assert_eq!(h.reopen(enabled()).unwrap().1, Vec::<String>::new());
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
}
