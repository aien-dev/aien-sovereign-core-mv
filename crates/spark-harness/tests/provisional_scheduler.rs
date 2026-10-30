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

//! Headstart M2 (#535): provisional scheduling on verified interfaces,
//! final completion only on verified producer completions.

mod common;

use common::{enabled, req, COMMIT, H};
use serde_json::json;
use spark_harness::completion::CompletionError;
use spark_harness::dependency::GraphError;
use spark_harness::execution::TaskState;
use spark_harness::readiness::{
    verify_interface, InterfaceSubmission, OperationClass, ProofClaim, ReadinessLevel,
};
use spark_harness::scheduler::{
    BindingLevel, Event, EventKind, Outcome, ProvisionalConfig, ResourceLimits, SchedError,
};

const IRREVERSIBLE: [OperationClass; 4] = [
    OperationClass::Commit,
    OperationClass::Publish,
    OperationClass::PersistentMutation,
    OperationClass::ExternalEffect,
];

#[test]
fn finalize_rechecks_the_interface_the_consumer_prepared_against() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    h.publish("a", 1);
    h.work_done("b");
    h.finish("a");
    assert!(h.sched.may_finalize("b").is_ok());
    // a's interface changes in the store; the event has not arrived yet.
    h.mint_interface("a", 2);
    assert!(matches!(
        h.sched.may_finalize("b"),
        Err(SchedError::Proof { .. })
    ));
    assert!(h
        .send(EventKind::Finalize {
            task: "b".into(),
            generation: h.generation("b")
        })
        .is_err());
    assert_eq!(state(&h, "b"), TaskState::AwaitingFinalProof);
}

fn state(h: &H, t: &str) -> TaskState {
    h.sched.state(t).unwrap()
}

fn two_task(h: &mut H) {
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
}

fn chain(h: &mut H) {
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.register("c", 1, &["b"]);
}

#[test]
fn disabled_is_the_default() {
    assert!(!ProvisionalConfig::default().enabled);
}

#[test]
fn two_task_dag_overlaps_preparation_then_finalizes_on_final_proof() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    assert_eq!(state(&h, "a"), TaskState::Provisional);
    assert_eq!(state(&h, "b"), TaskState::Blocked);
    assert!(matches!(
        h.sched.may_start("b", OperationClass::ReadOnlyAnalysis),
        Err(SchedError::WrongState { .. })
    ));

    let (_, fx) = h.publish("a", 1);
    assert_eq!(fx.dispatched, vec!["b".to_string()]);
    assert_eq!(state(&h, "b"), TaskState::Provisional);
    assert_eq!(
        h.sched.task("b").unwrap().bindings["a"].level,
        BindingLevel::Interface
    );
    let grant = h
        .sched
        .may_start("b", OperationClass::SandboxedCheck)
        .unwrap();
    assert_eq!(grant[0].level(), ReadinessLevel::InterfaceVerified);
    assert_eq!(grant[0].producer_task(), "a");
    assert_eq!(grant[0].producer_generation(), 1);

    h.advance(100);
    h.work_done("b");
    assert_eq!(state(&h, "b"), TaskState::AwaitingFinalProof);
    assert!(matches!(
        h.sched.may_finalize("b"),
        Err(SchedError::ProducerNotFinal { .. })
    ));
    assert!(matches!(
        h.send(EventKind::Finalize {
            task: "b".into(),
            generation: h.generation("b")
        }),
        Err(SchedError::ProducerNotFinal { .. })
    ));

    h.advance(50);
    h.finish("a");
    let fin = h.sched.may_finalize("b").unwrap();
    assert_eq!(fin[0].level(), ReadinessLevel::FinalVerified);
    h.finalize("b");
    assert_eq!(state(&h, "b"), TaskState::Committable);

    let trace = h.sched.timing_trace();
    println!(
        "M2_TIMING_TRACE {}",
        serde_json::to_string_pretty(&trace).unwrap()
    );
    let b = &trace["tasks"][1];
    assert_eq!(b["task"], "b");
    assert!(b["provisional"].as_u64().unwrap() > 0);
    assert!(b["blocked"].as_u64().unwrap() > 0);
    // b started while a was still provisional: overlap.
    let a_done = h.sched.task("a").unwrap().timing.committable_at.unwrap();
    assert!(h.sched.task("b").unwrap().timing.first_dispatch_at.unwrap() < a_done);
}

#[test]
fn three_task_chain_prepares_concurrently_and_finalizes_in_order() {
    let mut h = H::new(enabled());
    chain(&mut h);
    h.publish("a", 1);
    h.publish("b", 1);
    for t in ["a", "b", "c"] {
        assert_eq!(state(&h, t), TaskState::Provisional, "{t}");
    }
    assert_eq!(h.sched.permits_in_use().running, 3);
    for op in IRREVERSIBLE {
        assert!(h.sched.may_start("c", op).is_err());
    }
    // b finished and completed but a is not final: b is not final for c.
    h.work_done("b");
    assert!(h
        .send(EventKind::Finalize {
            task: "b".into(),
            generation: h.generation("b")
        })
        .is_err());
    h.finish("a");
    h.finalize("b");
    h.complete("b");
    // c's interface binding is upgraded to final once both are final.
    let grant = h.sched.may_start("c", OperationClass::Publish).unwrap();
    assert_eq!(grant[0].level(), ReadinessLevel::FinalVerified);
    h.work_done("c");
    h.finalize("c");
    assert_eq!(state(&h, "c"), TaskState::Committable);
}

#[test]
fn no_irreversible_op_from_interface_verified_alone() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    h.publish("a", 1);
    for op in IRREVERSIBLE {
        match h.sched.may_start("b", op) {
            Err(SchedError::NotPermitted { level, .. }) => {
                assert_eq!(level, ReadinessLevel::InterfaceVerified)
            }
            other => panic!("{op:?} must be refused, got {other:?}"),
        }
        assert!(h.sched.may_start_named("b", op.as_str()).is_err());
    }
    // Finalized without an M1 completion is still not final.
    h.work_done("a");
    h.finalize("a");
    for op in IRREVERSIBLE {
        assert!(h.sched.may_start("b", op).is_err());
    }
    // Only the M1 completion makes it final.
    h.complete("a");
    for op in IRREVERSIBLE {
        assert!(h.sched.may_start("b", op).is_ok());
    }
}

#[test]
fn unknown_operation_names_fail_closed() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    h.publish("a", 1);
    assert!(h.sched.may_start_named("b", "read_only_analysis").is_ok());
    for bad in [
        "deploy",
        "Commit",
        "READ_ONLY_ANALYSIS",
        "",
        "sandboxed-check",
    ] {
        assert!(
            matches!(
                h.sched.may_start_named("b", bad),
                Err(SchedError::UnknownOperation(_))
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn disabled_flag_keeps_consumers_blocked_until_final() {
    let mut h = H::new(ProvisionalConfig {
        enabled: false,
        ..enabled()
    });
    two_task(&mut h);
    h.publish("a", 1);
    let b = h.sched.task("b").unwrap();
    assert_eq!(b.state, TaskState::Blocked);
    assert!(!b.queued);
    assert_eq!(h.sched.permits_in_use().running, 1);
    h.work_done("a");
    h.finalize("a");
    assert_eq!(state(&h, "b"), TaskState::Blocked);
    let generation = h.generation("a");
    let claim = h.mint_completion("a");
    let fx = h.ok(EventKind::CompletionVerified {
        task: "a".into(),
        generation,
        claim,
    });
    assert_eq!(fx.dispatched, vec!["b".to_string()]);
    assert_eq!(
        h.sched.task("b").unwrap().bindings["a"].level,
        BindingLevel::Final
    );
    assert!(h.sched.may_start("b", OperationClass::Commit).is_ok());
}

#[test]
fn cycles_and_missing_nodes_are_refused_without_change() {
    let mut h = H::new(enabled());
    chain(&mut h);
    let before = h.sched.snapshot();
    let r = h.send(EventKind::AddDependency {
        producer: "c".into(),
        consumer: "a".into(),
    });
    assert!(
        matches!(&r, Err(SchedError::Graph(GraphError::Cycle(p))) if p.len() == 4),
        "{r:?}"
    );
    let r = h.send(EventKind::AddDependency {
        producer: "zz".into(),
        consumer: "b".into(),
    });
    assert!(
        matches!(r, Err(SchedError::Graph(GraphError::MissingNode(_)))),
        "{r:?}"
    );
    let r = h.send(EventKind::RegisterTask {
        task: "d".into(),
        resources: req(1),
        producers: vec!["a".into(), "nope".into()],
    });
    assert!(
        matches!(r, Err(SchedError::Graph(GraphError::MissingNode(_)))),
        "{r:?}"
    );
    assert_eq!(h.sched.snapshot(), before);
    // A running consumer cannot gain a dependency; a blocked one can.
    h.register("e", 1, &[]);
    assert!(matches!(
        h.send(EventKind::AddDependency {
            producer: "b".into(),
            consumer: "e".into(),
        }),
        Err(SchedError::WrongState { .. })
    ));
    h.edge("e", "c");
    h.publish("a", 1);
    h.publish("b", 1);
    assert_eq!(state(&h, "c"), TaskState::Blocked, "c still waits for e");
    h.publish("e", 1);
    assert_eq!(state(&h, "c"), TaskState::Provisional);
}

#[test]
fn cancellation_propagates_to_descendants_and_frees_permits() {
    let mut h = H::new(enabled());
    chain(&mut h);
    h.register("d", 1, &[]);
    h.publish("a", 1);
    h.publish("b", 1);
    let fx = h.ok(EventKind::Cancel {
        task: "a".into(),
        generation: h.generation("a"),
    });
    assert_eq!(fx.aborted, vec!["a", "b", "c"]);
    for t in ["a", "b", "c"] {
        assert_eq!(state(&h, t), TaskState::Aborted);
    }
    assert_eq!(state(&h, "d"), TaskState::Provisional);
    assert_eq!(h.sched.permits_in_use().running, 1);
}

#[test]
fn stale_generations_are_rejected_and_cannot_finalize() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    let (old_claim, _) = h.publish("a", 1);
    h.work_done("b");
    let old_completion = h.mint_completion("a");
    let fx = h.ok(EventKind::NewGeneration {
        task: "a".into(),
        generation: 2,
    });
    assert_eq!(fx.revoked, vec!["a", "b"]);
    let b = h.sched.task("b").unwrap();
    assert_eq!(b.state, TaskState::Blocked);
    assert_eq!(b.generation, 2);
    assert!(b.provisional_result.is_none());
    for kind in [
        EventKind::InterfacePublished {
            task: "a".into(),
            generation: 1,
            claim: old_claim.clone(),
        },
        EventKind::CompletionVerified {
            task: "a".into(),
            generation: 1,
            claim: old_completion.clone(),
        },
        EventKind::NewGeneration {
            task: "a".into(),
            generation: 2,
        },
    ] {
        assert!(
            matches!(h.send(kind), Err(SchedError::StaleGeneration { .. })),
            "stale event accepted"
        );
    }
    // A generation-1 receipt claimed as generation 2 fails the binding.
    let r = h.send(EventKind::InterfacePublished {
        task: "a".into(),
        generation: 2,
        claim: old_claim,
    });
    assert!(matches!(r, Err(SchedError::Proof { .. })), "{r:?}");
    let r = h.send(EventKind::WorkDone {
        task: "b".into(),
        generation: 1,
        result_digest: "late".into(),
    });
    assert!(
        matches!(r, Err(SchedError::StaleGeneration { .. })),
        "{r:?}"
    );
    // The engine itself refuses a stale interface generation.
    let mut stale = InterfaceSubmission {
        task_id: "a".into(),
        generation: 2,
        source_commit: COMMIT.into(),
        interface: json!({"v": 2}),
    };
    h.engine.publish_interface(&stale).unwrap();
    stale.generation = 1;
    assert!(matches!(
        h.engine.publish_interface(&stale),
        Err(CompletionError::StaleGeneration { .. })
    ));
}

#[test]
fn producer_failure_revokes_descendants_and_discards_results() {
    let mut h = H::new(enabled());
    chain(&mut h);
    h.publish("a", 1);
    h.publish("b", 1);
    h.work_done("b");
    assert!(h.sched.task("b").unwrap().provisional_result.is_some());
    let fx = h.ok(EventKind::ProducerFailed {
        task: "a".into(),
        generation: 1,
        reason: "build broke".into(),
    });
    assert_eq!(fx.aborted, vec!["a"]);
    assert_eq!(fx.revoked, vec!["b", "c"]);
    for t in ["b", "c"] {
        let e = h.sched.task(t).unwrap();
        assert_eq!(e.state, TaskState::Blocked);
        assert!(e.provisional_result.is_none() && e.bindings.is_empty() && e.interface.is_none());
        assert_eq!(e.generation, 2);
    }
    assert_eq!(h.sched.permits_in_use().running, 0);
}

#[test]
fn retry_after_failure_runs_to_committable() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    h.publish("a", 1);
    h.ok(EventKind::ProducerFailed {
        task: "a".into(),
        generation: 1,
        reason: "flaky".into(),
    });
    assert!(h
        .send(EventKind::Retry {
            task: "b".into(),
            generation: h.generation("b")
        })
        .is_err());
    let fx = h.ok(EventKind::Retry {
        task: "a".into(),
        generation: h.generation("a"),
    });
    assert_eq!(fx.dispatched, vec!["a"]);
    assert_eq!(h.generation("a"), 2);
    h.publish("a", 1);
    assert_eq!(state(&h, "b"), TaskState::Provisional);
    assert_eq!(h.sched.task("b").unwrap().bindings["a"].generation, 2);
    h.work_done("b");
    h.finish("a");
    h.finalize("b");
    assert_eq!(state(&h, "b"), TaskState::Committable);
}

#[test]
fn interface_change_at_same_generation_revokes_and_rebinds() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    let (v1, _) = h.publish("a", 1);
    h.work_done("b");
    let (v2, fx) = h.publish("a", 2);
    assert_ne!(v1, v2);
    assert_eq!(fx.revoked, vec!["b"]);
    assert_eq!(fx.dispatched, vec!["b"]);
    let b = h.sched.task("b").unwrap();
    assert_eq!(b.bindings["a"].claim, v2);
    assert!(b.provisional_result.is_none());
    assert_eq!((b.timing.revocations, b.timing.dispatches), (1, 2));
    let store = &h.engine.completion_store;
    assert!(matches!(
        verify_interface(store, "a", 1, &v1),
        Err(CompletionError::NotCurrentCompletion(_))
    ));
    // Republishing the same interface is idempotent: same receipt.
    assert_eq!(h.mint_interface("a", 2), v2);
}

#[test]
fn duplicate_event_delivery_is_idempotent() {
    let mut h = H::new(enabled());
    let ev = Event {
        id: "reg-a".into(),
        kind: EventKind::RegisterTask {
            task: "a".into(),
            resources: req(1),
            producers: vec![],
        },
    };
    assert!(matches!(
        h.sched.deliver(ev.clone()),
        Ok(Outcome::Applied(_))
    ));
    let before = h.sched.snapshot();
    assert_eq!(h.sched.deliver(ev), Ok(Outcome::Duplicate));
    assert_eq!(h.sched.snapshot(), before);
    let conflict = Event {
        id: "reg-a".into(),
        kind: EventKind::RegisterTask {
            task: "other".into(),
            resources: req(1),
            producers: vec![],
        },
    };
    assert!(matches!(
        h.sched.deliver(conflict),
        Err(SchedError::EventIdConflict(_))
    ));
    assert!(h.sched.task("other").is_none());
}

#[test]
fn forged_digest_missing_and_wrong_kind_proofs_are_refused() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    let good = h.mint_interface("a", 1);
    let completion = h.mint_completion("a");
    let forged_digest = ProofClaim {
        digest: "0".repeat(64),
        ..good.clone()
    };
    let missing = ProofClaim {
        receipt_id: "a".repeat(64),
        ..good.clone()
    };
    let wrong_commit = ProofClaim {
        commit: "2".repeat(40),
        ..good.clone()
    };
    let before = h.sched.snapshot();
    for (claim, interface) in [
        (forged_digest, true),
        (missing, true),
        (wrong_commit, true),
        (completion.clone(), true),
        (good.clone(), false),
    ] {
        let kind = if interface {
            EventKind::InterfacePublished {
                task: "a".into(),
                generation: 1,
                claim,
            }
        } else {
            EventKind::CompletionVerified {
                task: "a".into(),
                generation: 1,
                claim,
            }
        };
        let r = h.send(kind);
        assert!(matches!(r, Err(SchedError::Proof { .. })), "{r:?}");
    }
    assert_eq!(h.sched.snapshot(), before);
    assert_eq!(state(&h, "b"), TaskState::Blocked);

    // Receipts cannot be smuggled in through the submission.
    let smuggled = json!({
        "task_id": "a", "generation": 1, "source_commit": COMMIT,
        "interface": {"x": 1}, "receipt_id": good.receipt_id
    });
    assert!(serde_json::from_value::<InterfaceSubmission>(smuggled).is_err());
    let not_object = InterfaceSubmission {
        task_id: "a".into(),
        generation: 1,
        source_commit: COMMIT.into(),
        interface: json!("just a string"),
    };
    assert!(matches!(
        h.engine.publish_interface(&not_object),
        Err(CompletionError::InvalidSubmission(_))
    ));
}

#[test]
fn tampered_store_fails_gates_and_audit_revokes() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    let (claim, _) = h.publish("a", 1);
    assert!(h
        .sched
        .may_start("b", OperationClass::ReadOnlyAnalysis)
        .is_ok());
    let path = h
        .engine
        .completion_store
        .join("receipts")
        .join(format!("{}.json", claim.receipt_id));
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("interface:shape", "interface:shapf")).unwrap();
    assert!(matches!(
        h.sched.may_start("b", OperationClass::ReadOnlyAnalysis),
        Err(SchedError::Proof { .. })
    ));
    // The producer whose own claim fails is revoked, which cascades to b;
    // a re-runs at a new generation, b waits for a new interface.
    let revoked = h.sched.audit().unwrap();
    assert_eq!(revoked, vec!["a", "b"]);
    assert_eq!(state(&h, "a"), TaskState::Provisional);
    assert_eq!(h.generation("a"), 2);
    assert_eq!(state(&h, "b"), TaskState::Blocked);
    assert_eq!(h.sched.audit().unwrap(), Vec::<String>::new());
    h.sched.check_invariants().unwrap();
}

#[test]
fn resource_budget_bounds_queue_and_permits() {
    let mut h = H::new(ProvisionalConfig {
        enabled: true,
        queue_capacity: 1,
        limits: ResourceLimits {
            cpu_slots: 3,
            memory_bytes: 1 << 30,
            max_concurrency: 2,
        },
        lease_ttl: 1_000,
        dedup_window: 64,
    });
    let r = h.send(EventKind::RegisterTask {
        task: "huge".into(),
        resources: req(4),
        producers: vec![],
    });
    assert!(matches!(r, Err(SchedError::ExceedsCapacity(_))), "{r:?}");
    for t in ["x", "y", "z"] {
        h.register(t, 2, &[]);
    }
    assert_eq!(state(&h, "x"), TaskState::Provisional);
    assert_eq!(h.sched.queue(), vec!["y"]);
    let z = h.sched.task("z").unwrap();
    assert_eq!((z.state, z.queued), (TaskState::Blocked, false));
    assert_eq!(h.sched.permits_in_use().cpu_in_use, 2);
    let fx = h.work_done("x");
    assert_eq!(fx.dispatched, vec!["y"]);
    assert_eq!(h.sched.queue(), vec!["z"]);
    h.work_done("y");
    assert_eq!(state(&h, "z"), TaskState::Provisional);
    h.register("w", 1, &[]);
    assert_eq!(h.sched.permits_in_use().cpu_in_use, 3);
    // Concurrency bound: after z finishes, two 1-slot tasks would fit the
    // CPU budget next to w, but only max_concurrency = 2 tasks run.
    h.work_done("z");
    h.register("u", 1, &[]);
    h.register("v", 1, &[]);
    let pool = h.sched.permits_in_use();
    assert_eq!((pool.running, pool.cpu_in_use), (2, 2));
    assert_eq!(h.sched.queue(), vec!["v"]);
    assert_eq!(state(&h, "v"), TaskState::Blocked);
}

#[test]
fn blocked_tasks_hold_no_permits() {
    let mut h = H::new(enabled());
    h.register("a", 3, &[]);
    h.register("b", 3, &["a"]);
    h.register("c", 3, &["a"]);
    let pool = h.sched.permits_in_use();
    assert_eq!((pool.running, pool.cpu_in_use), (1, 3));
    h.publish("a", 1);
    // b fits (6 of 8 slots); c would need 9 and waits in the queue.
    assert_eq!(state(&h, "b"), TaskState::Provisional);
    let c = h.sched.task("c").unwrap();
    assert_eq!((c.state, c.queued), (TaskState::Blocked, true));
    assert!(c.lease_expires_at.is_none());
    assert_eq!(h.sched.permits_in_use().cpu_in_use, 6);
    // Revocation returns b to Blocked and frees its permits; c is then
    // blocked too (a has no interface at its new generation).
    h.ok(EventKind::NewGeneration {
        task: "a".into(),
        generation: 2,
    });
    for t in ["b", "c"] {
        assert_eq!(state(&h, t), TaskState::Blocked);
    }
    let pool = h.sched.permits_in_use();
    assert_eq!((pool.running, pool.cpu_in_use), (1, 3));
}

#[test]
fn leases_expire_into_abort_and_heartbeats_renew() {
    let mut h = H::new(enabled());
    two_task(&mut h);
    h.publish("a", 1);
    h.advance(900);
    h.ok(EventKind::Heartbeat {
        task: "a".into(),
        generation: h.generation("a"),
        beat: 1,
    });
    h.advance(900);
    assert!(h.sched.may_start("a", OperationClass::Commit).is_ok());
    h.advance(200);
    assert!(matches!(
        h.sched.may_start("a", OperationClass::Commit),
        Err(SchedError::LeaseExpired(_))
    ));
    assert!(matches!(
        h.send(EventKind::WorkDone {
            task: "a".into(),
            generation: 1,
            result_digest: "late".into()
        }),
        Err(SchedError::LeaseExpired(_))
    ));
    let fx = h.ok(EventKind::Tick);
    assert_eq!(fx.aborted, vec!["a"]);
    assert_eq!(fx.revoked, vec!["b"]);
    assert_eq!(h.sched.permits_in_use().running, 0);
}
