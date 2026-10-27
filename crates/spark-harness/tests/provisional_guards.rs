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

//! Headstart M2 (#535): one test per guard that a deletion mutation must
//! break, plus bounded duplicate memory and reserved audit ids.

mod common;

use common::{enabled, H};
use serde_json::json;
use spark_harness::completion::CompletionError;
use spark_harness::execution::TaskState;
use spark_harness::readiness::{
    verify_interface, InterfaceSubmission, OperationClass, ReadinessLevel,
};
use spark_harness::scheduler::{
    Event, EventKind, Outcome, ProvisionalConfig, Revocation, SchedError, RESERVED_ID_PREFIX,
};

fn remove_receipt(h: &H, receipt_id: &str) {
    std::fs::remove_file(
        h.engine
            .completion_store
            .join("receipts")
            .join(format!("{receipt_id}.json")),
    )
    .unwrap();
}

#[test]
fn heartbeat_after_lease_expiry_is_refused() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.advance(1_000);
    // No Tick has run yet, so a is still Provisional, but its lease is gone.
    let r = h.send(EventKind::Heartbeat {
        task: "a".into(),
        generation: h.generation("a"),
        beat: 1,
    });
    assert!(matches!(r, Err(SchedError::LeaseExpired(_))), "{r:?}");
    assert_eq!(h.sched.state("a"), Some(TaskState::Provisional));
    let fx = h.ok(EventKind::Tick);
    assert_eq!(fx.aborted, vec!["a"]);
}

#[test]
fn finalize_rechecks_the_whole_ancestor_chain() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.register("c", 1, &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.work_done("c");
    h.finish("a");
    h.finish("b");
    assert!(h.sched.may_finalize("c").is_ok());
    // Only the grandparent's completion disappears; b's own proof is fine.
    let a = h.sched.task("a").unwrap().completion.clone().unwrap();
    remove_receipt(&h, &a.receipt_id);
    assert!(h.sched.may_finalize("b").is_err());
    let r = h.sched.may_finalize("c");
    assert!(
        matches!(r, Err(SchedError::Proof { ref task, .. }) if task == "a"),
        "{r:?}"
    );
    assert!(h
        .send(EventKind::Finalize {
            task: "c".into(),
            generation: h.generation("c")
        })
        .is_err());
    assert_eq!(h.sched.state("c"), Some(TaskState::AwaitingFinalProof));
}

#[test]
fn proofs_are_accepted_only_for_active_tasks() {
    let mut h = H::new(enabled());
    h.register("p", 1, &[]);
    h.register("b", 1, &["p"]);
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
    // A valid receipt for b exists, but b has not started.
    let claim = h.mint_interface("b", 1);
    let r = h.send(EventKind::InterfacePublished {
        task: "b".into(),
        generation: 1,
        claim,
    });
    assert!(matches!(r, Err(SchedError::WrongState { .. })), "{r:?}");
    assert!(h.sched.task("b").unwrap().interface.is_none());
}

#[test]
fn failed_final_upgrade_falls_back_to_the_interface_level() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.finish("a");
    // b was bound at the interface level; a is final now, so it upgrades.
    assert!(h.sched.may_start("b", OperationClass::Commit).is_ok());
    let a = h.sched.task("a").unwrap().completion.clone().unwrap();
    remove_receipt(&h, &a.receipt_id);
    // The interface still verifies: reversible work may go on...
    let r = h
        .sched
        .may_start("b", OperationClass::ReadOnlyAnalysis)
        .unwrap();
    assert_eq!(r[0].level(), ReadinessLevel::InterfaceVerified);
    // ...but nothing irreversible.
    assert!(matches!(
        h.sched.may_start("b", OperationClass::Commit),
        Err(SchedError::NotPermitted { .. })
    ));
}

#[test]
fn reserved_ids_and_audit_events_cannot_be_delivered() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    let r = h.sched.deliver(Event {
        id: format!("{RESERVED_ID_PREFIX}audit-3"),
        kind: EventKind::Tick,
    });
    assert!(matches!(r, Err(SchedError::InvalidEvent(_))), "{r:?}");
    let r = h.send(EventKind::AuditRevoke {
        revocations: vec![Revocation {
            task: "a".into(),
            reason: "forged".into(),
        }],
    });
    assert!(matches!(r, Err(SchedError::InvalidEvent(_))), "{r:?}");
    assert_eq!(h.sched.state("a"), Some(TaskState::Provisional));
    // A caller id that looks like an old audit id does not block the audit.
    let (claim, _) = h.publish("a", 1);
    h.sched
        .deliver(Event {
            id: "audit-4-a".into(),
            kind: EventKind::Tick,
        })
        .unwrap();
    remove_receipt(&h, &claim.receipt_id);
    assert_eq!(h.sched.audit().unwrap(), vec!["a"]);
}

#[test]
fn duplicate_memory_is_bounded() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    let (claim, _) = h.publish("a", 1);
    let publish = Event {
        id: "pub-a-1".into(),
        kind: EventKind::InterfacePublished {
            task: "a".into(),
            generation: 1,
            claim,
        },
    };
    // Same claim again under a new id: no change, so it applies cleanly.
    assert!(matches!(
        h.sched.deliver(publish.clone()),
        Ok(Outcome::Applied(_))
    ));
    assert_eq!(h.sched.deliver(publish.clone()), Ok(Outcome::Duplicate));
    let bump = Event {
        id: "gen-a-2".into(),
        kind: EventKind::NewGeneration {
            task: "a".into(),
            generation: 2,
        },
    };
    h.sched.deliver(bump.clone()).unwrap();
    assert_eq!(h.sched.deliver(bump), Ok(Outcome::Duplicate));
    // The generation-1 bucket is gone: the old event is refused as stale,
    // and either way it changes nothing.
    let before = h.sched.snapshot();
    assert!(matches!(
        h.sched.deliver(publish),
        Err(SchedError::StaleGeneration { .. })
    ));
    assert_eq!(h.sched.snapshot(), before);
    // Heartbeats live in a window of `dedup_window` ids, however many come.
    let first = Event {
        id: "beat-1".into(),
        kind: EventKind::Heartbeat {
            task: "a".into(),
            generation: 2,
            beat: 1,
        },
    };
    h.sched.deliver(first.clone()).unwrap();
    for beat in 2..=500 {
        h.ok(EventKind::Heartbeat {
            task: "a".into(),
            generation: 2,
            beat,
        });
    }
    let window = enabled().dedup_window;
    assert!(
        h.sched.remembered_event_ids() <= window + 2,
        "{} ids remembered",
        h.sched.remembered_event_ids()
    );
    // The first beat left the window long ago; redelivered, it is refused
    // by its beat number and the lease is not renewed.
    let before = h.sched.snapshot();
    let r = h.sched.deliver(first);
    assert!(matches!(r, Err(SchedError::StaleHeartbeat { .. })), "{r:?}");
    assert_eq!(h.sched.snapshot(), before);
}

/// Inspector reproduction: with a tiny window, a Cancel redelivered after
/// the task was retried and the window rolled over must not abort again.
#[test]
fn cancel_redelivered_after_retry_and_window_rollover_is_refused() {
    let mut h = H::new(ProvisionalConfig {
        dedup_window: 4,
        ..enabled()
    });
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    let cancel = Event {
        id: "cancel-a".into(),
        kind: EventKind::Cancel {
            task: "a".into(),
            generation: 1,
        },
    };
    let fx = match h.sched.deliver(cancel.clone()) {
        Ok(Outcome::Applied(fx)) => fx,
        other => panic!("{other:?}"),
    };
    assert_eq!(fx.aborted, vec!["a", "b"]);
    for t in ["a", "b"] {
        h.ok(EventKind::Retry {
            task: t.into(),
            generation: 1,
        });
    }
    for _ in 0..6 {
        h.ok(EventKind::Tick);
    }
    assert_eq!(h.sched.state("a"), Some(TaskState::Provisional));
    let before = h.sched.snapshot();
    let r = h.sched.deliver(cancel);
    assert!(
        matches!(r, Err(SchedError::StaleGeneration { .. })),
        "{r:?}"
    );
    assert_eq!(h.sched.snapshot(), before);
    assert_eq!(h.sched.state("a"), Some(TaskState::Provisional));
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
}

/// A late Retry must not undo a later Cancel of the retried generation.
#[test]
fn retry_redelivered_after_a_second_cancel_is_refused() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.ok(EventKind::Cancel {
        task: "a".into(),
        generation: 1,
    });
    let retry = Event {
        id: "retry-a".into(),
        kind: EventKind::Retry {
            task: "a".into(),
            generation: 1,
        },
    };
    h.sched.deliver(retry.clone()).unwrap();
    h.ok(EventKind::Cancel {
        task: "a".into(),
        generation: 2,
    });
    assert_eq!(h.sched.state("a"), Some(TaskState::Aborted));
    let r = h.sched.deliver(retry);
    assert!(
        matches!(r, Err(SchedError::StaleGeneration { .. })),
        "{r:?}"
    );
    assert_eq!(h.sched.state("a"), Some(TaskState::Aborted));
    assert_eq!(h.generation("a"), 2);
}

/// Late Revoke and Finalize redeliveries must not act on a newer generation.
#[test]
fn revoke_and_finalize_redelivered_after_rerun_are_refused() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.work_done("b");
    let finalize = Event {
        id: "fin-b".into(),
        kind: EventKind::Finalize {
            task: "b".into(),
            generation: 1,
        },
    };
    h.sched.deliver(finalize.clone()).unwrap();
    let revoke = Event {
        id: "rv-b".into(),
        kind: EventKind::Revoke {
            task: "b".into(),
            generation: 1,
            reason: "operator".into(),
        },
    };
    h.sched.deliver(revoke.clone()).unwrap();
    // b re-runs at generation 2 against the final a.
    assert_eq!(h.sched.state("b"), Some(TaskState::Provisional));
    let r = h.sched.deliver(revoke);
    assert!(
        matches!(r, Err(SchedError::StaleGeneration { .. })),
        "{r:?}"
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::Provisional));
    h.work_done("b");
    let r = h.sched.deliver(finalize);
    assert!(
        matches!(r, Err(SchedError::StaleGeneration { .. })),
        "{r:?}"
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::AwaitingFinalProof));
    assert_eq!(h.generation("b"), 2);
}

#[test]
fn interface_republished_at_a_new_commit_retires_the_old_receipt() {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    let old = h.mint_interface("a", 1);
    // Same generation and interface, new commit: a new receipt replaces it.
    let new = h
        .engine
        .publish_interface(&InterfaceSubmission {
            task_id: "a".into(),
            generation: 1,
            source_commit: "2222222222222222222222222222222222222222".into(),
            interface: json!({ "task": "a", "api_version": 1 }),
        })
        .unwrap()
        .claim();
    assert_eq!(old.digest, new.digest);
    assert_ne!(old.receipt_id, new.receipt_id);
    let store = &h.engine.completion_store;
    assert!(verify_interface(store, "a", 1, &new).is_ok());
    assert!(matches!(
        verify_interface(store, "a", 1, &old),
        Err(CompletionError::NotCurrentCompletion(_))
    ));
}
