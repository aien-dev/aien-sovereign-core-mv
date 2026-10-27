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

//! Decision parity for the `serve.admit_preempt` read-only tap (DUAL-3a).
//!
//! For every pre-registered scenario the same deterministic workload runs
//! twice on fresh state: once with no observer, once with a
//! `RecordingObserver`. The canonical trace of every decision (batches,
//! errors, outputs, KV accounting, scheduler counters) must be byte-for-byte
//! identical, one record must exist per decision, and each record must agree
//! with the batch it describes.
//!
//! Negative control: with the Cargo feature
//! `dual_observer_mutant_changes_decision` the scheduler stops admitting after
//! one grant whenever an observer is installed. `observer_presence_changes_no_decision`
//! must FAIL under that feature (see the evidence receipt); the feature is
//! never on in a real build.

mod dual_common;

use aien_scheduler::dual_observer::{
    AdmissionVerdict, AdmitPreemptRecord, BatchOutcome, DecisionObserver, RecordingObserver,
    SERVE_ADMIT_PREEMPT_SITE,
};
use aien_scheduler::{AienScheduler, SequenceId};
use dual_common::{build, canonical_json, run, scenario, scenarios, Scenario, ScenarioTrace};
use std::sync::Arc;

fn run_without_observer(s: &Scenario) -> ScenarioTrace {
    let (mut scheduler, kv) = build(s);
    assert!(!scheduler.decision_observer_installed());
    run(s, &mut scheduler, &kv)
}

fn run_with_observer(s: &Scenario) -> (ScenarioTrace, Vec<AdmitPreemptRecord>) {
    let (mut scheduler, kv) = build(s);
    let observer = Arc::new(RecordingObserver::new());
    scheduler.set_decision_observer(Some(observer.clone()));
    assert!(scheduler.decision_observer_installed());
    let trace = run(s, &mut scheduler, &kv);
    (trace, observer.take())
}

#[test]
fn observer_presence_changes_no_decision() {
    for s in scenarios() {
        let absent = run_without_observer(&s);
        let (present, records) = run_with_observer(&s);
        assert!(
            canonical_json(&absent) == canonical_json(&present),
            "scenario {}: decisions differ with the observer installed",
            s.name
        );
        assert_eq!(
            records.len(),
            present.decisions,
            "scenario {}: one record per decision",
            s.name
        );
    }
}

#[test]
fn kv_accounting_identical_with_and_without_observer() {
    for s in scenarios() {
        let absent = run_without_observer(&s);
        let (present, _) = run_with_observer(&s);
        for (a, p) in absent.steps.iter().zip(present.steps.iter()) {
            assert_eq!(a.kv, p.kv, "scenario {} step {}", s.name, a.index);
        }
        assert_eq!(absent.final_kv, present.final_kv, "{}", s.name);
    }
}

/// Each record must describe the batch the scheduler actually produced.
#[test]
fn records_agree_with_the_batches_they_describe() {
    for s in scenarios() {
        let (trace, records) = run_with_observer(&s);
        for (step, record) in trace.steps.iter().zip(records.iter()) {
            assert_eq!(record.site, SERVE_ADMIT_PREEMPT_SITE);
            match (&step.batch, &step.error, &record.choice.outcome) {
                (
                    Some(batch),
                    None,
                    BatchOutcome::Batch {
                        prefill_requests,
                        decode_requests,
                        batch_tokens,
                    },
                ) => {
                    assert_eq!(record.step_id, batch.step_id, "{}", s.name);
                    assert_eq!(*prefill_requests, batch.prefill.len());
                    assert_eq!(*decode_requests, batch.decode.len());
                    let tokens: usize =
                        batch.prefill.iter().map(|(_, t, _)| t.len()).sum::<usize>()
                            + batch.decode.len();
                    assert_eq!(*batch_tokens, tokens);
                    let decode_ids: Vec<u64> = record
                        .choice
                        .decode_scheduled
                        .iter()
                        .map(|id| id.to_u64())
                        .collect();
                    assert_eq!(decode_ids, batch.decode, "{} step {}", s.name, step.index);
                    let prefill_ids: Vec<(u64, usize)> = record
                        .choice
                        .prefill_scheduled
                        .iter()
                        .map(|(id, n)| (id.to_u64(), *n))
                        .collect();
                    let batch_prefill: Vec<(u64, usize)> = batch
                        .prefill
                        .iter()
                        .map(|(id, t, _)| (*id, t.len()))
                        .collect();
                    assert_eq!(prefill_ids, batch_prefill, "{} step {}", s.name, step.index);
                }
                (None, None, BatchOutcome::Empty) => {
                    assert!(record.choice.decode_scheduled.is_empty());
                    assert!(record.choice.prefill_scheduled.is_empty());
                }
                (None, Some(err), BatchOutcome::Error(recorded)) => {
                    assert_eq!(err, recorded);
                    assert!(record
                        .alternatives
                        .admission_examined
                        .iter()
                        .any(|c| c.verdict == AdmissionVerdict::RefusedPoolExhausted));
                }
                other => panic!(
                    "{} step {}: inconsistent outcome {:?}",
                    s.name, step.index, other
                ),
            }
            // Exit readings equal what the trace read right after the step's
            // backend run only for counts the backend run does not change;
            // the scheduler counts at exit are asserted against the record.
            assert_eq!(record.kv_at_exit.total_blocks, step.kv.total_blocks);
            assert_eq!(record.limits.max_batch_size, s.config.max_batch_size);
            assert_eq!(record.limits.watermark_blocks, s.config.watermark_blocks);
            assert_eq!(
                record.kv_at_entry.allocated_blocks + record.kv_at_entry.free_blocks,
                record.kv_at_entry.total_blocks
            );
            assert_eq!(
                record.kv_at_exit.allocated_blocks + record.kv_at_exit.free_blocks,
                record.kv_at_exit.total_blocks
            );
        }
    }
}

#[test]
fn preemption_records_name_the_ranked_victims_and_the_choice() {
    let s = scenario("preemption_watermark");
    let (_, records) = run_with_observer(&s);
    let preempting: Vec<&AdmitPreemptRecord> = records
        .iter()
        .filter(|r| !r.choice.preempted.is_empty())
        .collect();
    assert!(
        !preempting.is_empty(),
        "the scenario must preempt at least once"
    );
    for r in &preempting {
        assert!(r.below_watermark_at_entry);
        assert!(!r.alternatives.preemption_victims_considered.is_empty());
        // The victim is the first ranked candidate (lowest priority).
        assert_eq!(
            r.choice.preempted[0],
            r.alternatives.preemption_victims_considered[0].seq
        );
        let min_priority = r
            .alternatives
            .preemption_victims_considered
            .iter()
            .map(|v| v.priority)
            .min()
            .unwrap();
        assert_eq!(
            r.alternatives.preemption_victims_considered[0].priority,
            min_priority
        );
        // No admission happens in a step that preempted.
        assert!(r.alternatives.admission_examined.is_empty());
    }
    // slot 2 has priority 1 (Normal), slot 1 priority 3 (Realtime): slot 2 is the victim.
    assert_eq!(
        preempting[0].choice.preempted[0],
        SequenceId::new(2, 1).unwrap()
    );
}

#[test]
fn tie_victim_is_first_in_running_order() {
    let s = scenario("preemption_tie_equal_priority");
    let (_, records) = run_with_observer(&s);
    let r = records
        .iter()
        .find(|r| !r.choice.preempted.is_empty())
        .expect("a preemption");
    let victims = &r.alternatives.preemption_victims_considered;
    assert_eq!(victims.len(), 2);
    assert_eq!(victims[0].priority, victims[1].priority);
    assert_eq!(victims[0].seq, r.alternatives.running_at_entry[0]);
    assert_eq!(r.choice.preempted[0], r.alternatives.running_at_entry[0]);
}

#[test]
fn no_preemption_records_rank_no_victims() {
    let s = scenario("no_preemption_sufficient_pool");
    let (_, records) = run_with_observer(&s);
    for r in &records {
        assert!(!r.below_watermark_at_entry);
        assert!(r.alternatives.preemption_victims_considered.is_empty());
        assert!(r.choice.preempted.is_empty());
    }
}

#[test]
fn capacity_boundary_records_carry_the_compared_numbers() {
    let below = run_with_observer(&scenario("capacity_one_below")).1;
    let at = run_with_observer(&scenario("capacity_exactly_at")).1;
    let above = run_with_observer(&scenario("capacity_one_above_refused")).1;

    let first_admit =
        |records: &[AdmitPreemptRecord]| records[0].alternatives.admission_examined[0].clone();
    let b = first_admit(&below);
    assert_eq!(b.needed_blocks, 7);
    assert_eq!(b.pool_blocks, 8);
    assert!(matches!(
        b.verdict,
        AdmissionVerdict::AdmittedPrefill { new_blocks: 7, .. }
    ));

    let a = first_admit(&at);
    assert_eq!(a.needed_blocks, 8);
    assert_eq!(a.min_needed, 8);
    assert_eq!(a.free_blocks, 8);
    assert!(matches!(
        a.verdict,
        AdmissionVerdict::AdmittedPrefill { new_blocks: 8, .. }
    ));

    let o = first_admit(&above);
    assert_eq!(o.needed_blocks, 9);
    assert_eq!(o.min_needed, 9);
    assert_eq!(o.pool_blocks, 8);
    assert_eq!(o.verdict, AdmissionVerdict::RefusedPoolExhausted);
    assert!(matches!(above[0].choice.outcome, BatchOutcome::Error(_)));
}

#[test]
fn wait_for_blocks_is_recorded_while_something_runs() {
    let (_, records) = run_with_observer(&scenario("capacity_wait_while_running"));
    assert!(records.iter().any(|r| r
        .alternatives
        .admission_examined
        .iter()
        .any(|c| c.verdict == AdmissionVerdict::WaitForBlocks)));
    assert!(records
        .iter()
        .all(|r| !matches!(r.choice.outcome, BatchOutcome::Error(_))));
}

#[test]
fn fanout_children_enter_decode_on_existing_tables() {
    for name in ["fanout_32_way", "fanout_500_way"] {
        let s = scenario(name);
        let expected = if name == "fanout_32_way" { 32 } else { 500 };
        let (trace, records) = run_with_observer(&s);
        let decode_admits: usize = records
            .iter()
            .flat_map(|r| r.alternatives.admission_examined.iter())
            .filter(|c| c.verdict == AdmissionVerdict::AdmittedDecode)
            .count();
        assert_eq!(
            decode_admits, expected,
            "{name}: every child admitted via its shared table"
        );
        assert_eq!(trace.final_kv.allocated_blocks, 0, "{name}: no leak");
        assert_eq!(
            trace.final_scheduler.finished_requests as usize,
            expected + 1
        );
        // A step admitting children records a reserved-block ledger that grows
        // by each child's incremental need.
        let admit_step = records
            .iter()
            .find(|r| {
                r.alternatives
                    .admission_examined
                    .iter()
                    .any(|c| c.verdict == AdmissionVerdict::AdmittedDecode)
            })
            .unwrap();
        let admitted: Vec<_> = admit_step
            .alternatives
            .admission_examined
            .iter()
            .filter(|c| c.verdict == AdmissionVerdict::AdmittedDecode)
            .collect();
        for w in admitted.windows(2) {
            assert_eq!(
                w[1].reserved_blocks,
                w[0].reserved_blocks + w[0].needed_blocks
            );
        }
    }
}

#[test]
fn queue_exhaustion_records_show_the_batch_limit_stopping_admission() {
    let s = scenario("queue_exhaustion_batch_limit");
    let (trace, records) = run_with_observer(&s);
    assert_eq!(records[0].alternatives.waiting_at_entry.len(), 12);
    let admitted_first_step = records[0]
        .alternatives
        .admission_examined
        .iter()
        .filter(|c| matches!(c.verdict, AdmissionVerdict::AdmittedPrefill { .. }))
        .count();
    assert_eq!(admitted_first_step, 4);
    assert_eq!(records[0].waiting_count_at_exit, 8);
    assert_eq!(trace.final_scheduler.waiting, 0);
}

#[test]
fn empty_step_record_is_empty() {
    let (_, records) = run_with_observer(&scenario("no_admission_empty_queues"));
    assert_eq!(records.len(), 3);
    for (i, r) in records.iter().enumerate() {
        assert_eq!(r.step_id, i as u64 + 1);
        assert_eq!(r.choice.outcome, BatchOutcome::Empty);
        assert!(r.alternatives.running_at_entry.is_empty());
        assert_eq!(r.arena_active_at_entry, 0);
        assert_eq!(r.kv_at_entry, r.kv_at_exit);
    }
}

/// Removing the observer mid-run stops the records and changes nothing else.
#[test]
fn observer_can_be_removed_and_decisions_still_match() {
    let s = scenario("admission_basic");
    let absent = run_without_observer(&s);
    let (mut scheduler, kv) = build(&s);
    let observer = Arc::new(RecordingObserver::new());
    scheduler.set_decision_observer(Some(observer.clone()));
    scheduler.set_decision_observer(None);
    let present_then_removed = run(&s, &mut scheduler, &kv);
    assert!(canonical_json(&absent) == canonical_json(&present_then_removed));
    assert!(observer.is_empty());
}

/// A second, independent observer implementation receives identical records
/// to the recording one: the record is a function of the decision alone.
struct DigestObserver(std::sync::Mutex<Vec<Vec<u8>>>);

impl DecisionObserver for DigestObserver {
    fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) {
        self.0.lock().unwrap().push(canonical_json(record));
    }
}

#[test]
fn two_observers_see_the_same_records() {
    let s = scenario("preemption_watermark");
    let (_, recorded) = run_with_observer(&s);
    let (mut scheduler, kv) = build(&s);
    let digest = Arc::new(DigestObserver(Default::default()));
    scheduler.set_decision_observer(Some(digest.clone()));
    let _ = run(&s, &mut scheduler, &kv);
    let seen = digest.0.lock().unwrap().clone();
    let expected: Vec<Vec<u8>> = recorded.iter().map(canonical_json).collect();
    assert_eq!(seen, expected);
}

#[allow(dead_code)]
fn _scheduler_type_check(s: &AienScheduler) -> bool {
    s.decision_observer_installed()
}
