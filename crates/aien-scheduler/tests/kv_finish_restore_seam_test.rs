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

//! C6a (sc#484, ADR 0003 sections 5.2, 6.1, 6.3; review findings 2, 3, 4): the
//! scheduler side of the persistent-KV seam.
//!
//! - `FinishHook` runs inside the finish transaction, in the same KV write
//!   guard and before `free_sequence`, at both finish sites (finding 2).
//! - `computed_prefix` leaves out the last token, which has no K/V (finding 3).
//! - `admit_restored_prefix` puts a restored sequence into the prefill phase,
//!   counted in `running_sequences`, and never lets it decode before the
//!   remaining prompt is prefilled (finding 4).
//! - `AwaitingRestore` keeps a sequence out of admission without blocking the
//!   sequences behind it.
//!
//! Persistence off (no hook, nothing held) is the behaviour the older scheduler
//! tests already pin; `no_hook_no_hold_leaves_the_finish_path_unchanged` repeats
//! the one observable here.

use aien_abi_core::{
    AienInferenceBackend, DecodeOutput, FinishReason, ModelConfig, SamplingParams, ScheduledBatch,
    SequenceRequest, StepMetrics,
};
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::{
    create_shared_kv_manager, holder_id, AienKvManager, PrefillState, SharedKvManager, SuffixPolicy,
};
use aien_scheduler::{
    AienScheduler, FinishContext, FinishHook, SchedulerConfig, SequenceId, SequencePhase,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 4096,
        max_prefill_tokens: 4096,
        prefill_chunk_size: 4096,
        chunk_prefill: true,
        watermark_blocks: 0,
    }
}

fn request(id: u64, prompt: &[u32], max_tokens: usize) -> SequenceRequest {
    SequenceRequest {
        request_id: id,
        prompt_tokens: prompt.to_vec(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority: 1,
    }
}

/// A backend with just enough knobs for the seam: what `computed_prefix` says,
/// whether it claims to manage the KV cache, and whether a decode step ends the
/// request from the backend side (`DecodeOutput::Finished`, the second finish site).
struct Script {
    manages_kv: bool,
    prefix: Option<Vec<u32>>,
    finish_on_decode: bool,
}

impl Script {
    fn plain() -> Self {
        Self {
            manages_kv: false,
            prefix: None,
            finish_on_decode: false,
        }
    }
}

#[async_trait]
impl AienInferenceBackend for Script {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut outputs = Vec::new();
        let mut prefill = 0;
        for r in &batch.prefill_requests {
            prefill += r.prompt_tokens.len();
            outputs.push(DecodeOutput::Token {
                request_id: r.request_id,
                token_id: 100,
                logprob: None,
            });
        }
        for &id in &batch.decode_requests {
            outputs.push(if self.finish_on_decode {
                DecodeOutput::Finished {
                    request_id: id,
                    reason: FinishReason::StopToken,
                    total_tokens: 1,
                }
            } else {
                DecodeOutput::Token {
                    request_id: id,
                    token_id: 101,
                    logprob: None,
                }
            });
        }
        Ok((
            outputs,
            StepMetrics {
                prefill_tokens_processed: prefill,
                decode_tokens_emitted: batch.decode_requests.len(),
                step_latency_us: 1,
                active_kv_blocks: 0,
            },
        ))
    }

    fn manages_kv_cache(&self) -> bool {
        self.manages_kv
    }

    fn computed_prefix(&self, _seq_id: u64) -> Option<Vec<u32>> {
        self.prefix.clone()
    }
}

/// What the hook saw, captured while the finish transaction was running.
#[derive(Debug, Default, Clone)]
struct Seen {
    calls: usize,
    table_present_during_hook: bool,
    free_blocks_during_hook: usize,
    lock_free_during_hook: bool,
    reason: Option<FinishReason>,
    request_id: u64,
    prefix: Option<Vec<u32>>,
    prompt: Vec<u32>,
}

/// Forks every block of the finishing sequence into one holder (job 7), like
/// the runtime's park will, unless told to fail.
struct ForkAllHook {
    kv: SharedKvManager,
    seen: Mutex<Seen>,
    fail_after_recording: bool,
}

impl ForkAllHook {
    fn new(kv: &SharedKvManager) -> Arc<Self> {
        Arc::new(Self {
            kv: kv.clone(),
            seen: Mutex::new(Seen::default()),
            fail_after_recording: false,
        })
    }

    fn failing(kv: &SharedKvManager) -> Arc<Self> {
        Arc::new(Self {
            kv: kv.clone(),
            seen: Mutex::new(Seen::default()),
            fail_after_recording: true,
        })
    }

    fn seen(&self) -> Seen {
        self.seen.lock().clone()
    }
}

impl FinishHook for ForkAllHook {
    fn on_finish(&self, kv: &mut AienKvManager, ctx: &FinishContext<'_>) -> Result<(), String> {
        let blocks = kv
            .get_block_table(ctx.request_id)
            .map(|t| t.block_ids.len())
            .unwrap_or(0);
        {
            let mut s = self.seen.lock();
            s.calls += 1;
            s.table_present_during_hook = blocks > 0;
            s.free_blocks_during_hook = kv.available_blocks();
            // The scheduler holds the write guard for the whole transaction:
            // nobody else can read the manager between the fork and the free.
            s.lock_free_during_hook = self.kv.try_read().is_some();
            s.reason = Some(ctx.reason);
            s.request_id = ctx.request_id;
            s.prefix = ctx.computed_prefix.clone();
            s.prompt = ctx.prompt.to_vec();
        }
        if self.fail_after_recording {
            return Err("park refused".to_string());
        }
        if blocks > 0 {
            kv.fork_range(ctx.request_id, 0, blocks, holder_id(7)?)?;
        }
        Ok(())
    }
}

fn scheduler(blocks: usize, block_size: usize) -> (SharedKvManager, AienScheduler) {
    let kv = create_shared_kv_manager(blocks, block_size);
    let s = AienScheduler::new(config(), kv.clone());
    (kv, s)
}

async fn run_to_finish(s: &mut AienScheduler, b: &mut dyn AienInferenceBackend) -> usize {
    let mut steps = 0;
    while steps < 50 {
        steps += 1;
        let Some((outs, _)) = s.step(b).await.unwrap() else {
            break;
        };
        if outs
            .iter()
            .any(|o| matches!(o, DecodeOutput::Finished { .. }))
        {
            break;
        }
    }
    steps
}

// ---------------------------------------------------------------------------
// Finding 2: the finish transaction.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finish_transaction_forks_holder_before_free() {
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = MockInferenceBackend::new(1);
    let seq = s.submit_request(request(0, &[1, 2, 3, 4], 3));
    let id = seq.to_u64();

    run_to_finish(&mut s, &mut backend).await;

    let seen = hook.seen();
    assert_eq!(seen.calls, 1, "exactly one hook call per finished request");
    assert!(
        seen.table_present_during_hook,
        "the sequence's own table is still there when the hook runs (before free)"
    );
    assert!(
        !seen.lock_free_during_hook,
        "the hook runs inside the KV write guard the scheduler holds for the finish"
    );
    assert_eq!(seen.request_id, id);
    assert_eq!(seen.reason, Some(FinishReason::LengthLimit));

    let r = kv.read();
    assert!(r.get_block_table(id).is_none(), "the sequence is freed");
    let holder = r.get_block_table(holder_id(7).unwrap()).expect("holder");
    assert_eq!(
        holder.block_ids.len(),
        2,
        "prompt 4 + 2 appended = 2 blocks"
    );
    for &b in &holder.block_ids {
        assert_eq!(
            r.get_block(b).unwrap().ref_count,
            1,
            "the holder's reference is the only one left, and it is not zero"
        );
    }
    assert_eq!(
        r.available_blocks(),
        16 - 2,
        "the holder's blocks never reached the free list"
    );
    assert_eq!(
        seen.free_blocks_during_hook,
        16 - 2,
        "the sequence's blocks were still allocated when the hook ran"
    );
    drop(r);

    // The finished request is gone from the scheduler as before.
    assert_eq!(s.running_count(), 0);
    assert_eq!(s.metrics().finished_requests, 1);
}

#[tokio::test]
async fn finish_hook_runs_at_the_backend_originated_finish_site_too() {
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = Script {
        finish_on_decode: true,
        ..Script::plain()
    };
    let seq = s.submit_request(request(0, &[1, 2, 3, 4], 10));

    run_to_finish(&mut s, &mut backend).await;

    let seen = hook.seen();
    assert_eq!(seen.calls, 1);
    assert!(
        seen.table_present_during_hook,
        "hook ran before free_sequence"
    );
    assert_eq!(seen.reason, Some(FinishReason::StopToken));
    let r = kv.read();
    assert!(r.get_block_table(seq.to_u64()).is_none());
    assert!(r.get_block_table(holder_id(7).unwrap()).is_some());
    assert_eq!(s.metrics().finished_requests, 1);
}

#[tokio::test]
async fn finish_hook_failure_still_frees_sequence() {
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::failing(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = MockInferenceBackend::new(1);
    let seq = s.submit_request(request(0, &[1, 2, 3, 4], 3));

    let mut finished = false;
    for _ in 0..10 {
        let Some((outs, _)) = s.step(&mut backend).await.unwrap() else {
            break;
        };
        finished |= outs
            .iter()
            .any(|o| matches!(o, DecodeOutput::Finished { .. }));
        if finished {
            break;
        }
    }

    assert!(finished, "a failing hook never stops the request finishing");
    assert_eq!(hook.seen().calls, 1);
    let r = kv.read();
    assert!(
        r.get_block_table(seq.to_u64()).is_none(),
        "the sequence is freed even though the hook failed"
    );
    assert_eq!(r.available_blocks(), 16, "no block leaked");
    drop(r);
    assert_eq!(s.metrics().finished_requests, 1);
    assert_eq!(
        s.metrics().finish_hook_failures,
        1,
        "the failure is counted, not swallowed"
    );
}

#[tokio::test]
async fn finish_hook_is_not_run_for_a_preempted_finish() {
    // One block of 4 tokens, a 4-token prompt: the first decode append has no
    // block, the scheduler ends the request as Preempted (K/V incomplete).
    let (kv, mut s) = scheduler(1, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = MockInferenceBackend::new(1);
    s.submit_request(request(0, &[1, 2, 3, 4], 8));

    s.step(&mut backend).await.unwrap().unwrap();

    assert_eq!(
        s.metrics().preempted_requests,
        1,
        "the request was preempted"
    );
    assert_eq!(hook.seen().calls, 0, "a preempted finish is never parked");
    assert_eq!(kv.read().available_blocks(), 1);
}

#[tokio::test]
async fn no_hook_no_hold_leaves_the_finish_path_unchanged() {
    let (kv, mut s) = scheduler(16, 4);
    let mut backend = MockInferenceBackend::new(1);
    let seq = s.submit_request(request(0, &[1, 2, 3, 4], 3));
    run_to_finish(&mut s, &mut backend).await;
    assert!(kv.read().get_block_table(seq.to_u64()).is_none());
    assert_eq!(kv.read().available_blocks(), 16);
    assert_eq!(s.metrics().finished_requests, 1);
    assert_eq!(s.metrics().finish_hook_failures, 0);
    assert_eq!(s.awaiting_restore_count(), 0);
}

// ---------------------------------------------------------------------------
// Finding 3: the computed prefix.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn computed_prefix_excludes_uncomputed_last_token() {
    let (_kv, mut s) = scheduler(16, 4);
    let mut backend = MockInferenceBackend::new(1);
    let prompt = [1u32, 2, 3, 4];
    let seq = s.submit_request(request(0, &prompt, 8));

    assert_eq!(
        s.computed_prefix(seq),
        None,
        "nothing generated yet, nothing computed"
    );

    // After prefill the sampled token (100) is pending: it has no K/V.
    s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(s.computed_prefix(seq), Some(prompt.to_vec()));

    // After one decode step token 100 was fed and has K/V; 101 is the new, uncomputed last.
    s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(s.computed_prefix(seq), Some(vec![1, 2, 3, 4, 100]));

    s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(s.computed_prefix(seq), Some(vec![1, 2, 3, 4, 100, 101]));

    let stale = SequenceId::new(seq.slot, seq.generation + 5).unwrap();
    assert_eq!(
        s.computed_prefix(stale),
        None,
        "an id from another incarnation"
    );
}

#[tokio::test]
async fn finish_hook_receives_the_computed_prefix_without_the_last_token() {
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = MockInferenceBackend::new(1);
    s.submit_request(request(0, &[1, 2, 3, 4], 3));

    run_to_finish(&mut s, &mut backend).await;

    let seen = hook.seen();
    // 3 tokens generated (100, 101, 101); the last one has no K/V.
    assert_eq!(seen.prefix, Some(vec![1, 2, 3, 4, 100, 101]));
    assert_eq!(seen.prompt, vec![1, 2, 3, 4]);
}

#[tokio::test]
async fn backend_computed_prefix_wins_and_a_silent_managed_backend_gives_none() {
    // The backend knows what is in the pool: its answer is passed through.
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = Script {
        prefix: Some(vec![1, 2, 3, 4]),
        finish_on_decode: true,
        ..Script::plain()
    };
    s.submit_request(request(0, &[1, 2, 3, 4], 10));
    run_to_finish(&mut s, &mut backend).await;
    assert_eq!(hook.seen().prefix, Some(vec![1, 2, 3, 4]));

    // A backend that manages the KV cache itself and says nothing: the
    // scheduler cannot know which positions are written, so it claims none.
    let (kv, mut s) = scheduler(16, 4);
    let hook = ForkAllHook::new(&kv);
    s.set_finish_hook(Some(hook.clone()));
    let mut backend = Script {
        manages_kv: true,
        finish_on_decode: true,
        ..Script::plain()
    };
    s.submit_request(request(0, &[1, 2, 3, 4], 10));
    run_to_finish(&mut s, &mut backend).await;
    assert_eq!(hook.seen().calls, 1);
    assert_eq!(hook.seen().prefix, None);
}

// ---------------------------------------------------------------------------
// Finding 4: admitting a restored prefix.
// ---------------------------------------------------------------------------

fn prompt40() -> Vec<u32> {
    (10..50).collect()
}

/// Builds what the runtime's restore commit (S3) leaves: a holder with the
/// first `restored` tokens of K/V, assembled into the request's own
/// `PrefillPending` table.
fn assemble_restored(kv: &SharedKvManager, seq: SequenceId, prompt: &[u32], restored: usize) {
    let holder = holder_id(1).unwrap();
    let mut w = kv.write();
    w.allocate_sequence(holder, &prompt[..restored]).unwrap();
    w.mark_prefill_pending(holder).unwrap();
    w.complete_prefill(holder).unwrap();
    w.assemble_table(seq.to_u64(), &[holder], SuffixPolicy::AllowPartialTail)
        .unwrap();
}

#[tokio::test]
async fn restored_prefix_does_not_enter_decode_before_remaining_prompt_prefilled() {
    let (kv, mut s) = {
        let kv = create_shared_kv_manager(16, 16);
        let cfg = SchedulerConfig {
            prefill_chunk_size: 4,
            ..config()
        };
        (kv.clone(), AienScheduler::new(cfg, kv))
    };
    let mut backend = MockInferenceBackend::new(1);
    let prompt = prompt40();
    let seq = s.submit_request(request(0, &prompt, 4));
    s.hold_for_restore(seq).unwrap();
    assemble_restored(&kv, seq, &prompt, 32);
    s.admit_restored_prefix(seq, 32).unwrap();

    let rec = s.arena().get(seq).unwrap();
    assert_eq!(rec.phase, SequencePhase::Prefill);
    assert!(!rec.is_prefilled, "restored is not prefilled");
    assert_eq!(rec.prompt_tokens_prefilled, 32);
    assert_eq!(
        kv.read().prefill_state(seq.to_u64()),
        Some(PrefillState::PrefillPending),
        "the table is never completed at restore"
    );

    // Remaining prompt: 8 tokens in chunks of 4. Two prefill steps, no decode.
    let batch = s.build_scheduled_batch().unwrap().unwrap();
    assert!(
        batch.decode_requests.is_empty(),
        "no decode on a restored-only table"
    );
    assert_eq!(batch.prefill_requests.len(), 1);
    assert_eq!(
        batch.prefill_requests[0].prompt_tokens,
        prompt[32..36].to_vec()
    );
    // Building a batch changes no sequence state; the real step below runs it.
    let (outs, m) = s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(m.prefill_tokens_processed, 4);
    assert!(outs
        .iter()
        .all(|o| !matches!(o, DecodeOutput::Token { .. })));
    assert!(!kv.read().prefill_state(seq.to_u64()).unwrap().is_ready());

    let (_, m) = s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(
        m.prefill_tokens_processed, 4,
        "the final chunk of the prompt"
    );
    assert!(
        kv.read().prefill_state(seq.to_u64()).unwrap().is_ready(),
        "the completion fence ran after the last chunk, and only then"
    );
    let rec = s.arena().get(seq).unwrap();
    assert!(rec.is_prefilled);
    assert_eq!(rec.prompt_tokens_prefilled, 40);
    assert_eq!(rec.phase, SequencePhase::Decode);

    let (_, m) = s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(m.prefill_tokens_processed, 0);
    assert_eq!(m.decode_tokens_emitted, 1, "decode starts after the prompt");
}

#[test]
fn restored_prefix_counts_in_running_sequences() {
    let (kv, mut s) = scheduler(16, 16);
    let prompt = prompt40();
    let seq = s.submit_request(request(0, &prompt, 4));
    s.hold_for_restore(seq).unwrap();
    assert_eq!(s.running_count(), 0);
    assert_eq!(s.waiting_count(), 1);
    assemble_restored(&kv, seq, &prompt, 32);

    let before: Vec<usize> = {
        let r = kv.read();
        let t = r.get_block_table(seq.to_u64()).unwrap();
        t.block_ids
            .iter()
            .map(|&b| r.get_block(b).unwrap().ref_count)
            .collect()
    };

    s.admit_restored_prefix(seq, 32).unwrap();

    assert_eq!(s.running_count(), 1, "counted in running_sequences");
    assert_eq!(s.waiting_count(), 0, "and out of the waiting queue");
    assert_eq!(s.awaiting_restore_count(), 0);
    let r = kv.read();
    let t = r.get_block_table(seq.to_u64()).unwrap();
    let after: Vec<usize> = t
        .block_ids
        .iter()
        .map(|&b| r.get_block(b).unwrap().ref_count)
        .collect();
    assert_eq!(before, after, "admission changes no reference count");
    assert_eq!(t.total_tokens, 32);
}

#[test]
fn admit_restored_prefix_refusals_change_nothing() {
    let (kv, mut s) = scheduler(16, 16);
    let prompt = prompt40();
    let seq = s.submit_request(request(0, &prompt, 4));

    // Not held for restore: a plain waiting request cannot be restored into.
    assert!(s.admit_restored_prefix(seq, 32).is_err());

    s.hold_for_restore(seq).unwrap();
    // No table yet.
    assert!(s.admit_restored_prefix(seq, 32).is_err());
    assemble_restored(&kv, seq, &prompt, 32);

    // 0, a count that leaves no token to compute (prompt_len), and one past it.
    for bad in [0usize, 40, 41] {
        assert!(
            s.admit_restored_prefix(seq, bad).is_err(),
            "restored = {bad}"
        );
    }
    // Count different from what the table holds.
    assert!(s.admit_restored_prefix(seq, 16).is_err());
    // A stale id.
    let stale = SequenceId::new(seq.slot, seq.generation + 1).unwrap();
    assert!(s.admit_restored_prefix(stale, 32).is_err());

    assert_eq!(s.running_count(), 0);
    assert_eq!(s.waiting_count(), 1);
    assert_eq!(s.awaiting_restore_count(), 1);
    let rec = s.arena().get(seq).unwrap();
    assert_eq!(rec.phase, SequencePhase::AwaitingRestore);
    assert_eq!(rec.prompt_tokens_prefilled, 0);
    assert_eq!(
        kv.read().prefill_state(seq.to_u64()),
        Some(PrefillState::PrefillPending)
    );

    // A table that is already ready is refused: decode could start on it.
    kv.write().complete_prefill(seq.to_u64()).unwrap();
    assert!(s.admit_restored_prefix(seq, 32).is_err());
    assert_eq!(s.running_count(), 0);
}

#[test]
fn hold_and_release_refusals() {
    let (_kv, mut s) = scheduler(16, 16);
    let seq = s.submit_request(request(0, &[1, 2, 3], 2));
    assert!(s.release_restore(seq).is_err(), "not held");
    s.hold_for_restore(seq).unwrap();
    assert!(s.hold_for_restore(seq).is_err(), "already held");
    let stale = SequenceId::new(seq.slot, seq.generation + 1).unwrap();
    assert!(s.hold_for_restore(stale).is_err());
    assert!(s.release_restore(stale).is_err());
    s.release_restore(seq).unwrap();
    assert_eq!(s.arena().get(seq).unwrap().phase, SequencePhase::Waiting);
}

#[tokio::test]
async fn awaiting_restore_is_not_admitted_until_released() {
    let (kv, mut s) = scheduler(16, 4);
    let mut backend = MockInferenceBackend::new(1);

    // Alone and held: nothing to schedule, and no "pool exhausted" refusal.
    let a = s.submit_request(request(0, &[1, 2, 3, 4], 2));
    s.hold_for_restore(a).unwrap();
    assert!(
        s.step(&mut backend).await.unwrap().is_none(),
        "held request is not admitted"
    );
    assert!(kv.read().get_block_table(a.to_u64()).is_none());

    // A request queued behind the held one is not blocked by it.
    let b = s.submit_request(request(0, &[9, 8, 7, 6], 2));
    let (outs, m) = s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(m.prefill_tokens_processed, 4);
    assert!(outs.iter().any(|o| matches!(
        o,
        DecodeOutput::Token { request_id, .. } if *request_id == b.to_u64()
    )));
    assert_eq!(
        s.arena().get(a).unwrap().phase,
        SequencePhase::AwaitingRestore
    );
    assert!(kv.read().get_block_table(a.to_u64()).is_none());
    assert_eq!(s.awaiting_restore_count(), 1);
    assert_eq!(
        s.waiting_count(),
        1,
        "still queued, still first in line once released"
    );

    // Released: plain prefill from zero (the restore failed or missed).
    s.release_restore(a).unwrap();
    assert_eq!(s.arena().get(a).unwrap().phase, SequencePhase::Waiting);
    assert_eq!(s.awaiting_restore_count(), 0);
    let (_, m) = s.step(&mut backend).await.unwrap().unwrap();
    assert_eq!(
        m.prefill_tokens_processed, 4,
        "released request is admitted with its whole prompt"
    );
    assert!(kv.read().get_block_table(a.to_u64()).is_some());
}

#[tokio::test]
async fn a_held_sequence_freed_by_cancel_is_purged_and_never_admitted() {
    let (kv, mut s) = scheduler(16, 4);
    let mut backend = MockInferenceBackend::new(1);
    let a = s.submit_request(request(0, &[1, 2, 3, 4], 2));
    s.hold_for_restore(a).unwrap();

    // Cancel: the arena slot is freed, so the id is stale (incarnation check).
    assert!(s.arena_mut().free_sequence(a));
    assert!(s.step(&mut backend).await.unwrap().is_none());
    assert_eq!(s.waiting_count(), 0, "the stale id left the queue");
    assert_eq!(s.awaiting_restore_count(), 0);

    // The slot is reused with a new generation: the old id can neither be
    // held, released nor restored into.
    let c = s.submit_request(request(0, &[5, 6, 7, 8], 2));
    assert_eq!(c.slot, a.slot);
    assert_ne!(c.generation, a.generation);
    assert!(s.release_restore(a).is_err());
    assert!(s.admit_restored_prefix(a, 2).is_err());
    assert_eq!(s.arena().get(c).unwrap().phase, SequencePhase::Waiting);
    assert_eq!(kv.read().available_blocks(), 16);
}
