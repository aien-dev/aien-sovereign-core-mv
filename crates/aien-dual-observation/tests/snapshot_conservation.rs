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

//! Conservation and zero-denominator tests for the DUAL serve observation
//! snapshot, over the real scheduler and KV manager with the mock backend.
//!
//! Every assertion compares two existing ledgers of the production state
//! machine against each other; nothing here is a model of what they should
//! be. The workload submits requests only through `submit_request` and never
//! cancels, so the sequence ledgers close exactly (see `lib.rs` field docs
//! for the two cases, fork children and cancellation, where
//! `admitted_requests` is not the right denominator).

use aien_abi_core::{SamplingParams, ScheduledBatch, SequenceRequest, StepMetrics};
use aien_dual_observation::{DeltaError, ObservationDelta, ServeObservationSnapshot};
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_scheduler::{AienScheduler, SchedulerConfig};

fn request(id: u64, prompt_len: usize, priority: u8, max_tokens: usize) -> SequenceRequest {
    SequenceRequest {
        request_id: id,
        prompt_tokens: (0..prompt_len as u32).collect(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority,
    }
}

fn config(max_batch_size: usize, watermark_blocks: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size,
        max_batch_tokens: 1024,
        max_prefill_tokens: 100,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks,
    }
}

/// Checks that hold for every snapshot of a scheduler fed only through
/// `submit_request` without cancellation.
fn assert_conserved(s: &ServeObservationSnapshot, where_: &str) {
    // KV: the free list and the refcount ledger partition the pool.
    assert_eq!(
        s.kv_physical_pages + s.kv_free_blocks,
        s.kv_total_blocks,
        "{where_}: physical (ref_count>0) + free == total"
    );
    assert_eq!(
        s.kv_allocated_blocks + s.kv_free_blocks,
        s.kv_total_blocks,
        "{where_}: allocated + free == total"
    );
    assert_eq!(s.kv_used_blocks, s.kv_physical_pages, "{where_}");
    assert_eq!(
        s.kv_shared_pages + s.kv_private_pages,
        s.kv_physical_pages,
        "{where_}: shared + private == physical"
    );
    assert!(s.kv_logical_pages >= s.kv_physical_pages, "{where_}");
    // Sequences: the arena holds exactly the queued and running records.
    assert_eq!(
        s.arena_active,
        s.running + s.waiting + s.preempted,
        "{where_}: arena == running + waiting + preempted"
    );
    // Lifetime ledger: every admitted request is finished or still held.
    assert_eq!(
        s.admitted_requests,
        s.finished_requests + (s.running + s.waiting + s.preempted) as u64,
        "{where_}: admitted == finished + held"
    );
    // Densities are the exact counters.
    assert_eq!(s.kv_block_density.numerator, s.kv_allocated_blocks);
    assert_eq!(s.kv_block_density.denominator, s.kv_total_blocks);
    assert_eq!(s.running_slot_density.numerator, s.running);
    assert_eq!(s.running_slot_density.denominator, s.max_batch_size);
    assert_eq!(
        s.kv_sharing_density.numerator,
        s.kv_logical_pages - s.kv_physical_pages
    );
    assert_eq!(s.kv_sharing_density.denominator, s.kv_logical_pages);
    assert_eq!(
        s.kv_below_watermark,
        s.kv_free_blocks < s.watermark_blocks,
        "{where_}"
    );
    if let Some(step) = &s.last_step {
        // StepMetrics::active_kv_blocks is allocated_block_count after the
        // step, read before the caller captured; a capture right after the
        // step sees the same number.
        assert_eq!(step.active_kv_blocks, s.kv_allocated_blocks, "{where_}");
    }
    if let Some(b) = &s.last_batch {
        assert_eq!(
            s.batch_slot_density.numerator,
            b.prefill_requests + b.decode_requests
        );
        assert_eq!(s.batch_token_density.numerator, b.batch_tokens);
        assert!(b.prefill_requests + b.decode_requests <= s.max_batch_size);
    }
}

/// Runs a scheduler through a mixed workload (admission, chunked prefill,
/// watermark preemption, completion) and snapshots after every step.
async fn run_and_snapshot(
    kv_blocks: usize,
    cfg: SchedulerConfig,
    requests: Vec<SequenceRequest>,
    max_steps: usize,
) -> Vec<ServeObservationSnapshot> {
    let kv = create_shared_kv_manager(kv_blocks, 16);
    let mut scheduler = AienScheduler::new(cfg, kv.clone());
    let mut backend = MockInferenceBackend::new(0);
    let mut snaps = Vec::new();

    snaps.push(ServeObservationSnapshot::capture(
        &scheduler,
        &kv.read(),
        None,
        None,
    ));
    for r in requests {
        scheduler.submit_request(r);
        snaps.push(ServeObservationSnapshot::capture(
            &scheduler,
            &kv.read(),
            None,
            None,
        ));
    }
    let mut steps = 0;
    while steps < max_steps
        && (scheduler.waiting_count() + scheduler.running_count() + scheduler.preempted_count() > 0)
    {
        steps += 1;
        let result = scheduler.step(&mut backend).await;
        let metrics: Option<StepMetrics> = match result {
            Ok(Some((_, m))) => Some(m),
            Ok(None) => None,
            Err(e) => panic!("step error in conservation workload: {e}"),
        };
        snaps.push(ServeObservationSnapshot::capture(
            &scheduler,
            &kv.read(),
            metrics.as_ref(),
            None,
        ));
    }
    assert_eq!(scheduler.running_count(), 0, "workload must drain");
    snaps
}

fn mixed_requests() -> Vec<SequenceRequest> {
    vec![
        request(1, 20, 3, 3),
        request(2, 150, 1, 2),
        request(3, 7, 0, 4),
        request(4, 64, 2, 1),
        request(5, 33, 1, 2),
    ]
}

#[tokio::test]
async fn ledgers_close_at_every_snapshot_under_plenty() {
    let snaps = run_and_snapshot(256, config(8, 2), mixed_requests(), 64).await;
    assert!(snaps.len() > 6);
    for (i, s) in snaps.iter().enumerate() {
        assert_conserved(s, &format!("plenty snapshot {i}"));
    }
    let last = snaps.last().unwrap();
    assert_eq!(last.kv_allocated_blocks, 0);
    assert_eq!(last.arena_active, 0);
    assert_eq!(last.finished_requests, 5);
    assert_eq!(last.kv_block_density.value(), Some(0.0));
}

#[tokio::test]
async fn ledgers_close_at_every_snapshot_under_watermark_pressure() {
    // 6-block pool, watermark 3: two 32-token prompts fill 4 blocks, the pool
    // drops below the watermark and the lower priority sequence is preempted.
    let cfg = SchedulerConfig {
        max_batch_size: 4,
        max_batch_tokens: 1024,
        max_prefill_tokens: 512,
        prefill_chunk_size: 128,
        chunk_prefill: false,
        watermark_blocks: 3,
    };
    let reqs = vec![request(1, 32, 3, 4), request(2, 32, 1, 4)];
    let snaps = run_and_snapshot(6, cfg, reqs, 64).await;
    for (i, s) in snaps.iter().enumerate() {
        assert_conserved(s, &format!("pressure snapshot {i}"));
    }
    assert!(
        snaps.iter().any(|s| s.kv_below_watermark && s.running > 0),
        "the workload must cross the watermark while something runs"
    );
    let last = snaps.last().unwrap();
    assert!(
        last.preempted_requests >= 1,
        "a preemption must have happened"
    );
    assert_eq!(last.finished_requests, 2);
    assert_eq!(last.kv_allocated_blocks, 0);
}

#[tokio::test]
async fn deltas_are_exact_counter_differences() {
    let snaps = run_and_snapshot(256, config(8, 2), mixed_requests(), 64).await;
    let first = &snaps[0];
    let after_submits = &snaps[5];
    let last = snaps.last().unwrap();

    let d = ObservationDelta::between(first, after_submits).unwrap();
    assert_eq!(d.steps, 0);
    assert_eq!(d.arrivals, 5);
    assert_eq!(d.completions, 0);
    assert_eq!(d.waiting_change, 5);

    let d = ObservationDelta::between(after_submits, last).unwrap();
    assert_eq!(d.arrivals, 0);
    assert_eq!(d.completions, 5);
    assert_eq!(d.steps, last.generation - after_submits.generation);
    assert_eq!(d.waiting_change, -5);
    assert_eq!(d.kv_allocated_change, 0);
    assert_eq!(d.prefill_tokens, last.total_prefill_tokens);
    assert_eq!(d.decode_tokens, last.total_decode_tokens);
    // Prefill tokens over the run equal the prompt tokens submitted.
    assert_eq!(d.prefill_tokens, 20 + 150 + 7 + 64 + 33);

    // Consecutive deltas sum to the whole.
    let mut arrivals = 0;
    let mut completions = 0;
    let mut steps = 0;
    for w in snaps.windows(2) {
        let d = ObservationDelta::between(&w[0], &w[1]).unwrap();
        arrivals += d.arrivals;
        completions += d.completions;
        steps += d.steps;
    }
    assert_eq!(arrivals, 5);
    assert_eq!(completions, 5);
    assert_eq!(steps, last.generation);
}

#[tokio::test]
async fn reversed_or_foreign_snapshots_are_refused() {
    let snaps = run_and_snapshot(256, config(8, 2), mixed_requests(), 64).await;
    let first = &snaps[0];
    let last = snaps.last().unwrap();
    assert!(matches!(
        ObservationDelta::between(last, first),
        Err(DeltaError::GenerationWentBackwards { .. })
    ));

    let other = run_and_snapshot(32, config(8, 2), vec![request(9, 5, 1, 1)], 8).await;
    assert!(matches!(
        ObservationDelta::between(first, other.last().unwrap()),
        Err(DeltaError::PoolSizeChanged { .. })
    ));

    // Same pool size, more steps (60 decode tokens) but fewer admissions: the
    // generation moves forward while the lifetime counter went backwards.
    let fewer = run_and_snapshot(256, config(8, 2), vec![request(9, 5, 1, 60)], 80).await;
    assert!(fewer.last().unwrap().generation > last.generation);
    assert!(matches!(
        ObservationDelta::between(last, fewer.last().unwrap()),
        Err(DeltaError::CounterWentBackwards {
            counter: "admitted_requests",
            ..
        })
    ));
}

#[test]
fn zero_capacity_densities_are_none_not_zero() {
    let kv = create_shared_kv_manager(0, 16);
    let cfg = SchedulerConfig {
        max_batch_size: 0,
        max_batch_tokens: 0,
        max_prefill_tokens: 0,
        prefill_chunk_size: 0,
        chunk_prefill: true,
        watermark_blocks: 0,
    };
    let scheduler = AienScheduler::new(cfg, kv.clone());
    let s = ServeObservationSnapshot::capture(&scheduler, &kv.read(), None, None);
    assert_eq!(s.kv_total_blocks, 0);
    assert_eq!(s.kv_block_density.value(), None);
    assert_eq!(s.kv_sharing_density.value(), None);
    assert_eq!(s.running_slot_density.value(), None);
    assert_eq!(s.batch_slot_density.value(), None);
    assert_eq!(s.batch_token_density.value(), None);
    assert!(!s.kv_below_watermark);
    assert_eq!(s.last_step, None);
    assert_eq!(s.last_batch, None);
    assert_conserved(&s, "zero capacity");
}

#[test]
fn batch_reading_copies_sizes_and_the_token_sum() {
    let kv = create_shared_kv_manager(16, 16);
    let scheduler = AienScheduler::new(config(4, 1), kv.clone());
    let batch = ScheduledBatch {
        prefill_requests: vec![request(1, 10, 1, 1), request(2, 5, 1, 1)],
        decode_requests: vec![3, 4, 5],
        block_tables: Default::default(),
        step_id: 42,
    };
    let s = ServeObservationSnapshot::capture(&scheduler, &kv.read(), None, Some(&batch));
    let b = s.last_batch.unwrap();
    assert_eq!(b.step_id, 42);
    assert_eq!(b.prefill_requests, 2);
    assert_eq!(b.decode_requests, 3);
    assert_eq!(b.batch_tokens, 15 + 3);
    assert_eq!(s.batch_slot_density.value(), Some(5.0 / 4.0));
    assert_eq!(s.batch_token_density.value(), Some(18.0 / 1024.0));
}

#[test]
fn snapshot_serializes_and_round_trips() {
    let kv = create_shared_kv_manager(8, 16);
    let scheduler = AienScheduler::new(config(4, 1), kv.clone());
    let s = ServeObservationSnapshot::capture(&scheduler, &kv.read(), None, None);
    let json = serde_json::to_string(&s).unwrap();
    let back: ServeObservationSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}
