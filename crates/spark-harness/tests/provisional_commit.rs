//! Headstart M3 (#537): the `Committing` state. A journaled `BeginCommit`
//! freezes a `Committable` task and its inputs; nothing upstream may change
//! them until a `CommitOutcome` is recorded; a replay that finds no outcome
//! marks the task `CommitUncertain` and never runs it again. Bound request
//! keys are journaled, kept while a task is aborted or retried, and dropped
//! when its commit ended.

mod common;

use common::{enabled, H};
use spark_harness::execution::TaskState;
use spark_harness::readiness::ProofClaim;
use spark_harness::scheduler::{
    CommitResult, Effects, EventKind, Outcome, SchedError, MAX_APPROVAL_BINDINGS_PER_TASK,
    MAX_APPROVAL_KEY_LEN, RESERVED_ID_PREFIX,
};
use std::collections::BTreeMap;

fn remove_receipt(h: &H, receipt_id: &str) {
    std::fs::remove_file(
        h.engine
            .completion_store
            .join("receipts")
            .join(format!("{receipt_id}.json")),
    )
    .unwrap();
}

fn completion(h: &H, task: &str) -> ProofClaim {
    h.sched.task(task).unwrap().completion.clone().unwrap()
}

/// `BeginCommit` built from the task's current accepted state.
fn begin_kind(h: &H, task: &str) -> EventKind {
    let producers: BTreeMap<String, ProofClaim> = h
        .sched
        .task(task)
        .unwrap()
        .bindings
        .keys()
        .map(|p| (p.clone(), completion(h, p)))
        .collect();
    EventKind::BeginCommit {
        task: task.into(),
        generation: h.generation(task),
        completion: completion(h, task),
        producers,
    }
}

fn begin(h: &mut H, task: &str) -> Result<Outcome, SchedError> {
    h.advance(10);
    let kind = begin_kind(h, task);
    h.send(kind)
}

fn outcome(h: &mut H, task: &str, outcome: CommitResult) -> Effects {
    let generation = h.generation(task);
    h.ok(EventKind::CommitOutcome {
        task: task.into(),
        generation,
        outcome,
    })
}

fn committed() -> CommitResult {
    CommitResult::Committed {
        detail: "test".into(),
    }
}

fn in_progress(r: &Result<Outcome, SchedError>, committing: &str) -> bool {
    matches!(r, Err(SchedError::CommitInProgress { committing: c, .. }) if c == committing)
}

/// a -> b -> c with a and b final. c is still provisional.
fn chain() -> H {
    let mut h = H::new(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.register("c", 1, &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.finish("a");
    h.finish("b");
    assert_eq!(h.sched.state("b"), Some(TaskState::Committable));
    assert_eq!(h.sched.state("c"), Some(TaskState::Provisional));
    h
}

#[test]
fn begin_commit_freezes_the_inputs_and_the_outcome_ends_it() {
    let mut h = chain();
    let gen_b = h.generation("b");
    assert!(matches!(begin(&mut h, "b"), Ok(Outcome::Applied(_))));
    assert_eq!(h.sched.state("b"), Some(TaskState::Committing));
    let inputs = h.sched.task("b").unwrap().commit.clone().unwrap();
    assert_eq!(inputs.completion, completion(&h, "b"));
    assert_eq!(inputs.producers.get("a"), Some(&completion(&h, "a")));
    // The consumer keeps working against the committing producer.
    assert_eq!(h.sched.state("c"), Some(TaskState::Provisional));
    outcome(&mut h, "b", committed());
    assert_eq!(h.sched.state("b"), Some(TaskState::Committed));
    assert_eq!(h.generation("b"), gen_b);
    // Done: a second begin is refused, as is a second outcome.
    assert!(matches!(
        begin(&mut h, "b"),
        Err(SchedError::WrongState { .. })
    ));
    let r = h.send(EventKind::CommitOutcome {
        task: "b".into(),
        generation: gen_b,
        outcome: committed(),
    });
    assert!(matches!(r, Err(SchedError::WrongState { .. })), "{r:?}");
}

#[test]
fn while_committing_upstream_and_own_changes_are_refused() {
    let mut h = chain();
    let gen_a = h.generation("a");
    let gen_b = h.generation("b");
    begin(&mut h, "b").unwrap();
    let refused_on_a = [
        EventKind::Revoke {
            task: "a".into(),
            generation: gen_a,
            reason: "test".into(),
        },
        EventKind::Cancel {
            task: "a".into(),
            generation: gen_a,
        },
        EventKind::NewGeneration {
            task: "a".into(),
            generation: gen_a + 1,
        },
        EventKind::ProducerFailed {
            task: "a".into(),
            generation: gen_a,
            reason: "test".into(),
        },
        EventKind::InterfacePublished {
            task: "a".into(),
            generation: gen_a,
            claim: h.mint_interface("a", 2),
        },
    ];
    for kind in refused_on_a {
        let r = h.send(kind.clone());
        assert!(in_progress(&r, "b"), "{kind:?}: {r:?}");
    }
    let refused_on_b = [
        EventKind::Cancel {
            task: "b".into(),
            generation: gen_b,
        },
        EventKind::NewGeneration {
            task: "b".into(),
            generation: gen_b + 1,
        },
        EventKind::ProducerFailed {
            task: "b".into(),
            generation: gen_b,
            reason: "test".into(),
        },
    ];
    for kind in refused_on_b {
        let r = h.send(kind.clone());
        assert!(in_progress(&r, "b"), "{kind:?}: {r:?}");
    }
    // Revoke needs an active task; Committing is not one.
    let r = h.send(EventKind::Revoke {
        task: "b".into(),
        generation: gen_b,
        reason: "test".into(),
    });
    assert!(matches!(r, Err(SchedError::WrongState { .. })), "{r:?}");
    assert_eq!(h.sched.state("a"), Some(TaskState::Committable));
    assert_eq!(h.sched.state("b"), Some(TaskState::Committing));
    assert_eq!(h.generation("a"), gen_a);
    assert_eq!(h.generation("b"), gen_b);

    // Once the outcome is in, the same revoke is accepted and does not
    // reach the committed task.
    outcome(&mut h, "b", committed());
    let fx = h.ok(EventKind::Revoke {
        task: "a".into(),
        generation: gen_a,
        reason: "test".into(),
    });
    assert!(!fx.revoked.contains(&"b".to_string()), "{fx:?}");
    assert_eq!(h.sched.state("b"), Some(TaskState::Committed));
    assert_eq!(h.sched.state("c"), Some(TaskState::Provisional));
}

#[test]
fn without_a_commit_the_same_revoke_reaches_the_consumer() {
    // Negative control for the test above.
    let mut h = chain();
    let gen_a = h.generation("a");
    let gen_b = h.generation("b");
    h.ok(EventKind::Revoke {
        task: "a".into(),
        generation: gen_a,
        reason: "test".into(),
    });
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
    assert_eq!(h.generation("b"), gen_b + 1);
}

#[test]
fn begin_commit_refuses_inputs_that_differ_from_the_state() {
    let mut h = chain();
    let good = begin_kind(&h, "b");
    let EventKind::BeginCommit {
        generation,
        completion: own,
        producers,
        ..
    } = good.clone()
    else {
        unreachable!()
    };
    let mut other = own.clone();
    other.digest = format!("{}0", other.digest);
    let with =
        |completion: ProofClaim, producers: BTreeMap<String, ProofClaim>| EventKind::BeginCommit {
            task: "b".into(),
            generation,
            completion,
            producers,
        };
    let mut missing = producers.clone();
    missing.clear();
    let mut extra = producers.clone();
    extra.insert("c".into(), own.clone());
    let mut wrong_producer = producers.clone();
    wrong_producer.insert("a".into(), other.clone());
    let cases = [
        ("own completion", with(other.clone(), producers.clone())),
        ("no producers", with(own.clone(), missing)),
        ("extra producer", with(own.clone(), extra)),
        ("producer claim", with(own.clone(), wrong_producer)),
        (
            "stale generation",
            EventKind::BeginCommit {
                task: "b".into(),
                generation: generation + 1,
                completion: own.clone(),
                producers: producers.clone(),
            },
        ),
    ];
    for (what, kind) in cases {
        let r = h.send(kind);
        assert!(r.is_err(), "{what}: {r:?}");
        assert_eq!(h.sched.state("b"), Some(TaskState::Committable), "{what}");
    }
    // Not committable yet.
    let r = h.send(EventKind::BeginCommit {
        task: "c".into(),
        generation: h.generation("c"),
        completion: own.clone(),
        producers: BTreeMap::from([("b".to_string(), completion(&h, "b"))]),
    });
    assert!(matches!(r, Err(SchedError::WrongState { .. })), "{r:?}");
    // A producer receipt that no longer verifies.
    remove_receipt(&h, &completion(&h, "a").receipt_id);
    let r = h.send(good.clone());
    assert!(
        matches!(r, Err(SchedError::Proof { ref task, .. }) if task == "a"),
        "{r:?}"
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::Committable));
}

#[test]
fn begin_commit_accepts_the_matching_inputs() {
    // Positive control for the refusals above.
    let mut h = chain();
    let kind = begin_kind(&h, "b");
    assert!(matches!(h.send(kind), Ok(Outcome::Applied(_))));
}

#[test]
fn only_one_commit_runs_per_dependency_chain() {
    let mut h = chain();
    h.register("x", 1, &[]);
    h.publish("x", 1);
    h.finish("x");
    // c becomes committable too.
    h.publish("c", 1);
    h.finish("c");
    begin(&mut h, "b").unwrap();
    // Below and above b: refused.
    assert!(in_progress(&begin(&mut h, "c"), "b"));
    assert!(in_progress(&begin(&mut h, "a"), "b"));
    // An unrelated task commits at the same time.
    assert!(begin(&mut h, "x").is_ok());
    outcome(&mut h, "b", committed());
    // a is upstream of a committed task only: it may commit now.
    assert!(begin(&mut h, "c").is_ok());
    assert!(begin(&mut h, "a").is_ok());
}

#[test]
fn failed_commit_aborts_retryably_and_revokes_consumers() {
    let mut h = chain();
    let gen_b = h.generation("b");
    let gen_c = h.generation("c");
    begin(&mut h, "b").unwrap();
    outcome(
        &mut h,
        "b",
        CommitResult::Failed {
            reason: "refused".into(),
        },
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::Aborted));
    assert_eq!(h.sched.state("c"), Some(TaskState::Blocked));
    assert_eq!(h.generation("c"), gen_c + 1);
    h.ok(EventKind::Retry {
        task: "b".into(),
        generation: gen_b,
    });
    assert_eq!(h.generation("b"), gen_b + 1);
}

#[test]
fn uncertain_commit_is_terminal_and_revokes_consumers() {
    let mut h = chain();
    let gen_b = h.generation("b");
    let gen_c = h.generation("c");
    begin(&mut h, "b").unwrap();
    outcome(
        &mut h,
        "b",
        CommitResult::Uncertain {
            reason: "transport".into(),
        },
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    assert_eq!(h.sched.state("c"), Some(TaskState::Blocked));
    assert_eq!(h.generation("c"), gen_c + 1);
    for kind in [
        EventKind::Retry {
            task: "b".into(),
            generation: gen_b,
        },
        EventKind::Cancel {
            task: "b".into(),
            generation: gen_b,
        },
        EventKind::NewGeneration {
            task: "b".into(),
            generation: gen_b + 1,
        },
    ] {
        let r = h.send(kind.clone());
        assert!(r.is_err(), "{kind:?}: {r:?}");
        assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    }
    assert!(begin(&mut h, "b").is_err());
}

#[test]
fn replayed_commit_without_an_outcome_becomes_uncertain_and_never_reruns() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.register("c", 1, &["b"]);
    h.publish("a", 1);
    h.publish("b", 1);
    h.finish("a");
    h.finish("b");
    let gen_c = h.generation("c");
    begin(&mut h, "b").unwrap();
    // Crash before the outcome.
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    assert_eq!(h.sched.state("c"), Some(TaskState::Blocked));
    assert_eq!(h.generation("c"), gen_c + 1);
    let journal = std::fs::read_to_string(h.fx.journal()).unwrap();
    assert!(journal.contains(&format!("{RESERVED_ID_PREFIX}commit-uncertain-")));
    // Neither a new begin nor a late outcome is accepted.
    assert!(begin(&mut h, "b").is_err());
    let r = h.send(EventKind::CommitOutcome {
        task: "b".into(),
        generation: h.generation("b"),
        outcome: committed(),
    });
    assert!(matches!(r, Err(SchedError::WrongState { .. })), "{r:?}");
    // A second restart settles nothing new.
    let lines = std::fs::read_to_string(h.fx.journal())
        .unwrap()
        .lines()
        .count();
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    assert_eq!(
        std::fs::read_to_string(h.fx.journal())
            .unwrap()
            .lines()
            .count(),
        lines
    );
}

#[test]
fn replayed_commit_with_an_outcome_keeps_it() {
    // Negative control: a recorded outcome survives the restart unchanged.
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.publish("a", 1);
    h.finish("a");
    begin(&mut h, "a").unwrap();
    outcome(&mut h, "a", committed());
    let snap = h.sched.snapshot();
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.state("a"), Some(TaskState::Committed));
    assert_eq!(h.sched.snapshot()["tasks"], snap["tasks"]);
}

#[test]
fn audit_defers_a_revocation_that_would_reach_a_running_commit() {
    let mut h = chain();
    begin(&mut h, "b").unwrap();
    remove_receipt(&h, &completion(&h, "a").receipt_id);
    let revoked = h.sched.audit().unwrap();
    assert!(revoked.is_empty(), "{revoked:?}");
    assert_eq!(h.sched.state("a"), Some(TaskState::Committable));
    assert_eq!(h.sched.state("b"), Some(TaskState::Committing));
    // After the outcome the deferred revocation goes through and stops at
    // the committed task.
    outcome(&mut h, "b", committed());
    let revoked = h.sched.audit().unwrap();
    assert!(revoked.contains(&"a".to_string()), "{revoked:?}");
    assert_eq!(h.sched.state("b"), Some(TaskState::Committed));
}

#[test]
fn audit_without_a_commit_revokes_the_whole_chain() {
    // Negative control for the deferral above.
    let mut h = chain();
    remove_receipt(&h, &completion(&h, "a").receipt_id);
    let revoked = h.sched.audit().unwrap();
    assert!(revoked.contains(&"a".to_string()), "{revoked:?}");
    assert_eq!(h.sched.state("b"), Some(TaskState::Blocked));
}

#[test]
fn begin_commit_redelivery_is_a_duplicate() {
    let mut h = chain();
    h.advance(10);
    let kind = begin_kind(&h, "b");
    let event = spark_harness::scheduler::Event {
        id: "begin-b".into(),
        kind,
    };
    assert!(matches!(
        h.sched.deliver(event.clone()),
        Ok(Outcome::Applied(_))
    ));
    let r = h.sched.deliver(event);
    assert!(matches!(r, Ok(Outcome::Duplicate)), "{r:?}");
    assert_eq!(h.sched.state("b"), Some(TaskState::Committing));
}

// ---------- bound request keys ----------

fn bind(h: &mut H, key: &str, task: &str) -> Result<Outcome, SchedError> {
    h.advance(1);
    let generation = h.generation(task);
    h.send(EventKind::BindApproval {
        key: key.into(),
        task: task.into(),
        generation,
    })
}

fn bound(h: &H, key: &str) -> Option<(String, u64)> {
    h.sched
        .approval(key)
        .map(|b| (b.task.clone(), b.generation))
}

fn wrong_state(r: &Result<Outcome, SchedError>) -> bool {
    matches!(r, Err(SchedError::WrongState { .. }))
}

fn cancel(h: &mut H, task: &str) {
    let generation = h.generation(task);
    h.ok(EventKind::Cancel {
        task: task.into(),
        generation,
    });
}

fn retry(h: &mut H, task: &str) {
    let generation = h.generation(task);
    h.ok(EventKind::Retry {
        task: task.into(),
        generation,
    });
}

#[test]
fn bindings_survive_a_replay_and_are_dropped_when_the_task_commits() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    let gen_b = h.generation("b");
    bind(&mut h, "approval:x", "b").unwrap();
    bind(&mut h, "proposal:y", "b").unwrap();
    h.restart(enabled()).unwrap();
    assert_eq!(bound(&h, "approval:x"), Some(("b".into(), gen_b)));
    assert_eq!(bound(&h, "proposal:y"), Some(("b".into(), gen_b)));
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    begin(&mut h, "b").unwrap();
    // Running: still bound.
    assert_eq!(h.sched.approvals_len(), 2);
    outcome(&mut h, "b", committed());
    assert_eq!(h.sched.approvals_len(), 0);
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.approvals_len(), 0);
}

#[test]
fn bindings_are_dropped_only_when_the_commit_landed() {
    // Aborted keeps the binding (the task may be retried); uncertain keeps
    // it too (whether the commit landed is unknown, so the bound request
    // stays fenced and refused instead of running unfenced); only a landed
    // commit releases it.
    for (result, ended, kept) in [
        (
            CommitResult::Failed {
                reason: "refused".into(),
            },
            TaskState::Aborted,
            true,
        ),
        (
            CommitResult::Uncertain {
                reason: "transport".into(),
            },
            TaskState::CommitUncertain,
            true,
        ),
        (committed(), TaskState::Committed, false),
    ] {
        let mut h = chain();
        let gen_b = h.generation("b");
        bind(&mut h, "approval:x", "b").unwrap();
        // A consumer's binding is kept: the consumer is revoked to
        // `Blocked`, which is not an end.
        bind(&mut h, "approval:c", "c").unwrap();
        begin(&mut h, "b").unwrap();
        outcome(&mut h, "b", result.clone());
        assert_eq!(h.sched.state("b"), Some(ended), "{result:?}");
        let want = kept.then(|| ("b".to_string(), gen_b));
        assert_eq!(bound(&h, "approval:x"), want, "{result:?}");
        assert!(bound(&h, "approval:c").is_some(), "{result:?}");
    }
}

#[test]
fn an_uncertain_commit_keeps_its_bindings_for_good_and_takes_no_new_one() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    h.publish("a", 1);
    h.finish("a");
    h.finish("b");
    let gen_b = h.generation("b");
    bind(&mut h, "approval:x", "b").unwrap();
    begin(&mut h, "b").unwrap();
    outcome(
        &mut h,
        "b",
        CommitResult::Uncertain {
            reason: "the fenced call panicked".into(),
        },
    );
    assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    // The bound request stays bound to the terminal generation, so a fence
    // on it is a refusal (`AlreadySubmitted` / wrong state), never an
    // unfenced run. A new key cannot be bound to the task.
    assert_eq!(bound(&h, "approval:x"), Some(("b".into(), gen_b)));
    assert!(wrong_state(&bind(&mut h, "approval:new", "b")));
    assert_eq!(bound(&h, "approval:new"), None);
    assert!(begin(&mut h, "b").is_err());
    // Rebinding the held key is refused too: the task ended.
    assert!(wrong_state(&bind(&mut h, "approval:x", "b")));
    assert_eq!(bound(&h, "approval:x"), Some(("b".into(), gen_b)));
    // A replay rebuilds the same table.
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.state("b"), Some(TaskState::CommitUncertain));
    assert_eq!(bound(&h, "approval:x"), Some(("b".into(), gen_b)));
    assert_eq!(h.sched.approvals_len(), 1);
}

#[test]
fn an_aborted_task_keeps_its_bindings_however_it_aborted() {
    // Cancel, producer failure and lease expiry abort c (provisional); a
    // failed commit aborts b. Each keeps the binding at the same
    // generation, so the bound request stays fenced (`NotCommittable`).
    type Abort = fn(&mut H);
    let aborts: [(&str, &str, Abort); 4] = [
        ("cancel", "c", |h| cancel(h, "c")),
        ("producer failed", "c", |h| {
            let generation = h.generation("c");
            h.ok(EventKind::ProducerFailed {
                task: "c".into(),
                generation,
                reason: "test".into(),
            });
        }),
        ("lease expired", "c", |h| {
            h.advance(5_000);
            h.ok(EventKind::Tick);
        }),
        ("commit failed", "b", |h| {
            begin(h, "b").unwrap();
            outcome(
                h,
                "b",
                CommitResult::Failed {
                    reason: "EFFECT_REFUSED".into(),
                },
            );
        }),
    ];
    for (how, task, abort) in aborts {
        let mut h = chain();
        let generation = h.generation(task);
        bind(&mut h, "approval:x", task).unwrap();
        abort(&mut h);
        assert_eq!(h.sched.state(task), Some(TaskState::Aborted), "{how}");
        assert_eq!(h.generation(task), generation, "{how}");
        assert_eq!(
            bound(&h, "approval:x"),
            Some((task.to_string(), generation)),
            "{how}"
        );
    }
}

#[test]
fn a_cancel_keeps_the_bindings_of_every_task_it_aborts() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("b", 1, &["a"]);
    bind(&mut h, "approval:a", "a").unwrap();
    bind(&mut h, "approval:b", "b").unwrap();
    cancel(&mut h, "a");
    assert_eq!(h.sched.state("b"), Some(TaskState::Aborted));
    assert_eq!(h.sched.approvals_len(), 2);
    h.restart(enabled()).unwrap();
    assert_eq!(h.sched.approvals_len(), 2);
    assert_eq!(bound(&h, "approval:b"), Some(("b".into(), 1)));
}

#[test]
fn a_retry_keeps_the_old_binding_until_the_request_is_bound_again() {
    let mut h = chain();
    let generation = h.generation("c");
    bind(&mut h, "approval:x", "c").unwrap();
    cancel(&mut h, "c");
    retry(&mut h, "c");
    assert_eq!(h.generation("c"), generation + 1);
    // Still bound, at the superseded generation: the fence refuses it
    // (`StaleGeneration`) instead of the request running unfenced.
    assert_eq!(bound(&h, "approval:x"), Some(("c".into(), generation)));
    bind(&mut h, "approval:x", "c").unwrap();
    assert_eq!(bound(&h, "approval:x"), Some(("c".into(), generation + 1)));
    assert_eq!(h.sched.approvals_len(), 1);
}

#[test]
fn bind_refuses_bad_keys_stale_generations_and_ended_commits() {
    let mut h = chain();
    let gen_b = h.generation("b");
    for key in [String::new(), "k".repeat(MAX_APPROVAL_KEY_LEN + 1)] {
        let r = bind(&mut h, &key, "b");
        assert!(matches!(r, Err(SchedError::InvalidEvent(_))), "{r:?}");
    }
    assert!(bind(&mut h, &"k".repeat(MAX_APPROVAL_KEY_LEN), "b").is_ok());
    let r = h.send(EventKind::BindApproval {
        key: "approval:old".into(),
        task: "b".into(),
        generation: gen_b + 1,
    });
    assert!(
        matches!(r, Err(SchedError::FutureGeneration { .. })),
        "{r:?}"
    );
    // Committing is not an end: a bind is accepted.
    begin(&mut h, "b").unwrap();
    assert!(bind(&mut h, "approval:during", "b").is_ok());
    outcome(&mut h, "b", committed());
    assert!(wrong_state(&bind(&mut h, "approval:late", "b")));
    assert_eq!(bound(&h, "approval:late"), None);

    let mut h = chain();
    begin(&mut h, "b").unwrap();
    outcome(
        &mut h,
        "b",
        CommitResult::Uncertain {
            reason: "transport".into(),
        },
    );
    assert!(wrong_state(&bind(&mut h, "approval:late", "b")));
    assert_eq!(h.sched.approvals_len(), 0);
    // Aborted is not an end: a request may be bound to it (and is refused
    // by the fence until a retry and a new bind).
    cancel(&mut h, "c");
    assert!(bind(&mut h, "approval:aborted", "c").is_ok());
    assert_eq!(h.sched.approvals_len(), 1);
}

#[test]
fn the_binding_table_is_bounded_per_task_and_fails_closed() {
    let mut h = chain();
    for i in 0..MAX_APPROVAL_BINDINGS_PER_TASK {
        bind(&mut h, &format!("approval:{i}"), "c").unwrap();
    }
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    let r = bind(&mut h, "approval:one-more", "c");
    assert!(
        matches!(r, Err(SchedError::BindingsFull { ref task, .. }) if task == "c"),
        "{r:?}"
    );
    assert_eq!(bound(&h, "approval:one-more"), None);
    // Rebinding a key c holds replaces it.
    assert!(bind(&mut h, "approval:0", "c").is_ok());
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
}

#[test]
fn aborted_tasks_cannot_exhaust_the_table_for_other_tasks() {
    let mut h = chain();
    h.register("x", 1, &[]);
    for i in 0..MAX_APPROVAL_BINDINGS_PER_TASK {
        bind(&mut h, &format!("approval:c-{i}"), "c").unwrap();
    }
    cancel(&mut h, "c");
    // c keeps its full share while aborted and after a retry. A new key
    // for the retried task reclaims one superseded binding (see
    // `a_full_task_reclaims_its_superseded_bindings_oldest_first`), so the
    // share stays at the cap.
    retry(&mut h, "c");
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    assert!(bind(&mut h, "approval:c-new", "c").is_ok());
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    // Its own keys may be bound again at the new generation.
    assert!(bind(&mut h, "approval:c-1", "c").is_ok());
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    // An unrelated task still gets its whole share.
    for i in 0..MAX_APPROVAL_BINDINGS_PER_TASK {
        bind(&mut h, &format!("approval:x-{i}"), "x").unwrap();
    }
    assert_eq!(h.sched.approvals_len(), 2 * MAX_APPROVAL_BINDINGS_PER_TASK);
    let r = bind(&mut h, "approval:x-new", "x");
    assert!(
        matches!(r, Err(SchedError::BindingsFull { ref task, .. }) if task == "x"),
        "{r:?}"
    );
}

#[test]
fn a_full_task_reclaims_its_superseded_bindings_oldest_first() {
    let mut h = H::journaled(enabled());
    h.register("a", 1, &[]);
    h.register("c", 1, &["a"]);
    let g1 = h.generation("c");
    for i in 0..MAX_APPROVAL_BINDINGS_PER_TASK {
        bind(&mut h, &format!("approval:c-{i}"), "c").unwrap();
    }
    // Full at the current generation: a new key is refused, nothing moves.
    let r = bind(&mut h, "approval:c-new", "c");
    assert!(matches!(r, Err(SchedError::BindingsFull { .. })), "{r:?}");
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    // Retried: every binding now names the superseded generation g1.
    cancel(&mut h, "c");
    retry(&mut h, "c");
    let g2 = h.generation("c");
    assert_eq!(g2, g1 + 1);
    // A new key reclaims the lowest superseded key (BTreeMap order:
    // "approval:c-0", then "approval:c-1", then "approval:c-10").
    bind(&mut h, "approval:c-new", "c").unwrap();
    assert_eq!(bound(&h, "approval:c-0"), None);
    assert_eq!(bound(&h, "approval:c-new"), Some(("c".into(), g2)));
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    bind(&mut h, "approval:c-newer", "c").unwrap();
    assert_eq!(bound(&h, "approval:c-1"), None);
    assert_eq!(bound(&h, "approval:c-10"), Some(("c".into(), g1)));
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    // Rebinding a held key moves it to g2 without reclaiming anything.
    bind(&mut h, "approval:c-10", "c").unwrap();
    assert_eq!(bound(&h, "approval:c-10"), Some(("c".into(), g2)));
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK);
    // Reclaim never touches another task's bindings.
    h.register("x", 1, &[]);
    bind(&mut h, "approval:x-0", "x").unwrap();
    bind(&mut h, "approval:c-newest", "c").unwrap();
    assert_eq!(bound(&h, "approval:x-0"), Some(("x".into(), 1)));
    assert_eq!(bound(&h, "approval:c-11"), None);
    // A replay reclaims the same bindings.
    let table: Vec<(String, Option<(String, u64)>)> = ["c-0", "c-1", "c-10", "c-11", "c-12"]
        .iter()
        .map(|k| (format!("approval:{k}"), bound(&h, &format!("approval:{k}"))))
        .collect();
    h.restart(enabled()).unwrap();
    for (key, want) in table {
        assert_eq!(bound(&h, &key), want, "{key}");
    }
    assert_eq!(h.sched.approvals_len(), MAX_APPROVAL_BINDINGS_PER_TASK + 1);
    // Once every binding is at the current generation again, the cap
    // refuses new keys as before.
    for i in 0..MAX_APPROVAL_BINDINGS_PER_TASK {
        let key = format!("approval:c-{i}");
        if bound(&h, &key).is_some_and(|(_, g)| g == g1) {
            bind(&mut h, &key, "c").unwrap();
        }
    }
    let r = bind(&mut h, "approval:c-last", "c");
    assert!(matches!(r, Err(SchedError::BindingsFull { .. })), "{r:?}");
}
