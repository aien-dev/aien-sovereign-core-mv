#![allow(clippy::too_many_arguments)]
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

pub mod dual_observer;
pub mod sequence;

pub use sequence::*;

use crate::dual_observer::{
    AdmissionCandidate, AdmissionVerdict, AdmitPreemptAlternatives, AdmitPreemptChoice,
    AdmitPreemptRecord, AdmitPreemptTrace, BatchLimits, BatchOutcome, DecisionObserver, KvReading,
    LoopStop, PreemptionVictim, RunningItem, RunningVerdict, SERVE_ADMIT_PREEMPT_SITE,
};
use aien_abi_core::{
    AienInferenceBackend, DecodeOutput, FinishReason, ScheduledBatch, SequenceRequest, StepMetrics,
};
use aien_inference_abi::KV_POOL_EXHAUSTED_PREFIX;
use aien_kv_cache::{AienKvManager, PrefillGateError, PrefillState};
use aien_platform::{InferenceWork, Priority};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// Prefill readiness gate (PREFILL-GATE). A sequence may enter decode only
/// when its block table K/V was computed by a completed prefill
/// (`BlockTable::is_prefill_ready`). Having a block table is not enough:
/// `allocate_sequence` only reserves and zeroes blocks.
#[inline]
fn prefill_gate_allows_decode(table_ready: bool) -> bool {
    if cfg!(feature = "prefill_mutant_no_check") {
        // MUTANT: readiness check removed. Tests must catch this.
        return true;
    }
    table_ready
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_prefill_tokens: usize,
    pub prefill_chunk_size: usize,
    pub chunk_prefill: bool,
    pub watermark_blocks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            max_prefill_tokens: 2048,
            prefill_chunk_size: 512,
            chunk_prefill: true,
            watermark_blocks: 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    pub total_steps: u64,
    pub admitted_requests: u64,
    pub finished_requests: u64,
    pub preempted_requests: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub avg_step_latency_us: f64,
    pub chunked_prefill_steps: u64,
    /// Finish hooks that returned an error (persistent KV, ADR 0003 section 6.1).
    /// The sequence is freed regardless.
    pub finish_hook_failures: u64,
}

/// What a [`FinishHook`] is told about the request that is finishing.
#[derive(Debug)]
pub struct FinishContext<'a> {
    pub seq_id: SequenceId,
    /// `seq_id` as the u64 the KV manager and the backend know it by. It carries
    /// the arena generation, so it is the incarnation of this request.
    pub request_id: u64,
    pub reason: FinishReason,
    /// The tokens whose K/V is written, in order, without the last token (which
    /// has none). Taken from the backend when it answers (it knows what is in the
    /// pool); otherwise, for a backend that does not manage the KV cache itself,
    /// derived from the scheduler's own record (`AienScheduler::computed_prefix`).
    /// `None` when nobody can say, in which case nothing may be parked.
    pub computed_prefix: Option<Vec<u32>>,
    pub prompt: &'a [u32],
}

/// Persistent KV (ADR 0003 section 6.1, review finding 2). Runs inside the
/// scheduler's finish transaction, under the KV write guard it already takes for
/// `free_sequence`, and BEFORE that free, so the sequence's blocks have a
/// reference at every instant. Typical use: `fork_range` into storage holders
/// and push an intent to a lock-free outbox. The hook must not block, do I/O or
/// take any other lock held across an await. It is not run for a `Preempted`
/// finish (its K/V is incomplete) and not for an id the arena no longer knows.
/// An `Err` is counted (`SchedulerMetrics::finish_hook_failures`) and the
/// sequence is freed anyway; undoing a partial fork is the hook's own business
/// (DOCS SILENT; the narrowest rule is that the scheduler never retries or
/// rolls back for it). A panic is not caught.
pub trait FinishHook: Send + Sync {
    fn on_finish(&self, kv: &mut AienKvManager, ctx: &FinishContext<'_>) -> Result<(), String>;
}

/// Native continuous batching LLM scheduler managing SequenceArena,
/// decoupled CompletionRouter, and zero-token-clone queues.
pub struct AienScheduler {
    config: SchedulerConfig,
    kv_manager: Arc<RwLock<AienKvManager>>,
    arena: SequenceArena,
    completion_router: CompletionRouter,
    waiting_queue: VecDeque<SequenceId>,
    preempted_queue: VecDeque<SequenceId>,
    /// Sequences whose K/V this scheduler freed while they stay live (watermark
    /// preemption, a failed append, a prefill-gate refusal). The backend still
    /// holds their per-sequence state (positions, dense caches, a held sample),
    /// which would make the re-prefill start at a stale position. `step` drops
    /// that state through `release_sequence` before the next `execute_step`.
    pending_backend_releases: Vec<SequenceId>,
    running_sequences: Vec<SequenceId>,
    step_id: u64,
    metrics: SchedulerMetrics,
    /// DUAL `serve.admit_preempt` read-only tap. `None` (the default) skips
    /// every tracing branch in `build_scheduled_batch`.
    observer: Option<Arc<dyn DecisionObserver>>,
    /// Persistent KV finish hook. `None` (the default) is today's behaviour exactly.
    finish_hook: Option<Arc<dyn FinishHook>>,
}

impl AienScheduler {
    pub fn new(config: SchedulerConfig, kv_manager: Arc<RwLock<AienKvManager>>) -> Self {
        Self {
            config,
            kv_manager,
            arena: SequenceArena::new(),
            completion_router: CompletionRouter::new(),
            waiting_queue: VecDeque::new(),
            preempted_queue: VecDeque::new(),
            pending_backend_releases: Vec::new(),
            running_sequences: Vec::new(),
            step_id: 0,
            metrics: SchedulerMetrics::default(),
            observer: None,
            finish_hook: None,
        }
    }

    /// Installs (or with `None` removes) the finish hook (see [`FinishHook`]).
    pub fn set_finish_hook(&mut self, hook: Option<Arc<dyn FinishHook>>) {
        self.finish_hook = hook;
    }

    pub fn finish_hook_installed(&self) -> bool {
        self.finish_hook.is_some()
    }

    /// Installs (or with `None` removes) the DUAL `serve.admit_preempt`
    /// observer. The observer is shown each decision after it is final and
    /// cannot change it (see `dual_observer`).
    pub fn set_decision_observer(&mut self, observer: Option<Arc<dyn DecisionObserver>>) {
        self.observer = observer;
    }

    pub fn decision_observer_installed(&self) -> bool {
        self.observer.is_some()
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    pub fn arena(&self) -> &SequenceArena {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut SequenceArena {
        &mut self.arena
    }

    pub fn completion_router(&self) -> &CompletionRouter {
        &self.completion_router
    }

    pub fn completion_router_mut(&mut self) -> &mut CompletionRouter {
        &mut self.completion_router
    }

    pub fn register_completion_sink(&mut self, sink: Arc<dyn CompletionSink>) -> CompletionSinkId {
        self.completion_router.register(sink)
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    pub fn preempted_count(&self) -> usize {
        self.preempted_queue.len()
    }

    /// Takes the ids whose backend per-sequence state must be dropped through
    /// `AienInferenceBackend::release_sequence` before the next `execute_step`
    /// (see `pending_backend_releases`). `step` does this itself; a driver that
    /// executes `build_scheduled_batch` or `build_batch_plan` on its own must
    /// call it and release each id, or a resumed sequence computes at stale
    /// positions.
    pub fn take_pending_backend_releases(&mut self) -> Vec<SequenceId> {
        std::mem::take(&mut self.pending_backend_releases)
    }

    pub fn running_count(&self) -> usize {
        self.running_sequences.len()
    }

    /// Submits a SequenceRequest, creating an immutable PromptHandle and allocating a SequenceId.
    pub fn submit_request(&mut self, request: SequenceRequest) -> SequenceId {
        let prompt_slice: Box<[u32]> = request.prompt_tokens.into_boxed_slice();
        let prompt_handle: PromptHandle = Arc::from(prompt_slice);
        let priority = match request.priority {
            0 => Priority::Background,
            1 => Priority::Normal,
            2 => Priority::Interactive,
            _ => Priority::Realtime,
        };

        let seq_id = if request.request_id != 0 && (request.request_id >> 32) > 0 {
            if let Ok(explicit_id) = SequenceId::from_u64(request.request_id) {
                self.arena
                    .insert_with_id(
                        explicit_id,
                        prompt_handle.clone(),
                        aien_platform::ModelHandle(0),
                        aien_platform::KvHandle(0),
                        priority,
                        None,
                        None,
                        request.sampling_params.max_tokens as u32,
                        request.sampling_params.clone(),
                        None,
                    )
                    .unwrap_or_else(|_| {
                        self.arena
                            .insert_with(
                                prompt_handle,
                                aien_platform::ModelHandle(0),
                                aien_platform::KvHandle(0),
                                priority,
                                None,
                                None,
                                request.sampling_params.max_tokens as u32,
                                request.sampling_params,
                                None,
                            )
                            .expect("Arena allocation failed")
                    })
            } else {
                self.arena
                    .insert_with(
                        prompt_handle,
                        aien_platform::ModelHandle(0),
                        aien_platform::KvHandle(0),
                        priority,
                        None,
                        None,
                        request.sampling_params.max_tokens as u32,
                        request.sampling_params,
                        None,
                    )
                    .expect("Arena allocation failed")
            }
        } else {
            self.arena
                .insert_with(
                    prompt_handle,
                    aien_platform::ModelHandle(0),
                    aien_platform::KvHandle(0),
                    priority,
                    None,
                    None,
                    request.sampling_params.max_tokens as u32,
                    request.sampling_params,
                    None,
                )
                .expect("Arena allocation failed")
        };

        self.waiting_queue.push_back(seq_id);
        self.metrics.admitted_requests += 1;
        seq_id
    }

    /// Submits structured platform InferenceWork without cloning prompt tokens.
    pub fn submit_work(
        &mut self,
        work: InferenceWork,
        prompt: PromptHandle,
        sampling_params: Option<aien_abi_core::SamplingParams>,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        let sampling = sampling_params.unwrap_or_default();
        let seq_id = self.arena.insert_with(
            prompt,
            work.model,
            work.kv,
            work.priority,
            work.deadline,
            work.branch_parent
                .and_then(|p| SequenceId::from_u64(p).ok()),
            work.next_token_budget,
            sampling,
            sink_id,
        )?;

        self.waiting_queue.push_back(seq_id);
        self.metrics.admitted_requests += 1;
        Ok(seq_id)
    }

    /// Zero-copy sequence fork preserving immutable PromptHandle.
    pub fn fork_sequence(
        &mut self,
        parent_id: SequenceId,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        if self.arena.is_stale(parent_id) {
            return Err(format!(
                "Parent sequence {} is stale or inactive",
                parent_id
            ));
        }

        let child_id = self.arena.fork(parent_id, sink_id)?;

        // Fork block table in KV manager
        {
            let mut kv = self.kv_manager.write();
            kv.fork_sequence(parent_id.to_u64(), child_id.to_u64())?;
        }

        self.running_sequences.push(child_id);
        Ok(child_id)
    }

    /// Legacy backward compatibility for fork_subagent(parent_id, child_id).
    pub fn fork_subagent(&mut self, parent_raw: u64, child_raw: u64) -> Result<(), String> {
        self.fork_subagent_with_sink(parent_raw, child_raw, None)
    }

    /// Subagent fork with completion sink routing for streamed output.
    pub fn fork_subagent_with_sink(
        &mut self,
        parent_raw: u64,
        child_raw: u64,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(), String> {
        self.fork_subagent_with_sink_ids(parent_raw, child_raw, sink_id)
            .map(|_| ())
    }

    /// Same as `fork_subagent_with_sink`, returning the (parent, child) KV
    /// sequence ids the fork was recorded under, so the runtime can fork the
    /// backend's per-sequence state under the same ids (PREFILL-E2E C4).
    pub fn fork_subagent_with_sink_ids(
        &mut self,
        parent_raw: u64,
        child_raw: u64,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(u64, u64), String> {
        let parent_id = self
            .running_sequences
            .iter()
            .copied()
            .find(|id| id.to_u64() == parent_raw || id.slot as u64 == parent_raw)
            .or_else(|| SequenceId::from_u64(parent_raw).ok())
            .ok_or_else(|| format!("Parent sequence {} not found in running pool", parent_raw))?;

        if self.arena.is_stale(parent_id) {
            return Err(format!("Parent sequence {} is stale", parent_id));
        }

        let child_id = self.arena.fork(parent_id, sink_id)?;

        let kv_ids = {
            let mut kv = self.kv_manager.write();
            match kv.fork_sequence(parent_id.to_u64(), child_id.to_u64()) {
                Ok(_) => (parent_id.to_u64(), child_id.to_u64()),
                Err(_) => {
                    kv.fork_sequence(parent_raw, child_raw)?;
                    (parent_raw, child_raw)
                }
            }
        };

        self.running_sequences.push(child_id);
        Ok(kv_ids)
    }

    /// Tokens of `seq_id` whose K/V is written, derived from the scheduler's own
    /// record: the prompt plus every generated token except the last (ADR 0003
    /// section 6.2, review finding 3). A decode step feeds the previous token
    /// through the model and samples a new one, so the newest token has no K/V;
    /// after prefill the first sampled token is likewise pending. `None` for an
    /// id this arena does not know (another incarnation) or a sequence that has
    /// not finished its prefill and produced a token. This is the scheduler's
    /// bookkeeping, not proof the bytes are in the pool: a backend that keeps its
    /// own K/V answers through `AienInferenceBackend::computed_prefix` instead.
    pub fn computed_prefix(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        let rec = self.arena.get(seq_id)?;
        let (_newest, computed) = rec.generated_tokens.split_last()?;
        if !rec.is_prefilled {
            return None;
        }
        let mut prefix = Vec::with_capacity(rec.prompt.len() + computed.len());
        prefix.extend_from_slice(&rec.prompt);
        prefix.extend_from_slice(computed);
        Some(prefix)
    }

    /// The finish transaction (ADR 0003 section 6.1): in one KV write guard,
    /// run the finish hook (if any), then `free_sequence`. The backend is asked
    /// for the computed prefix first, before the guard, and is not touched under
    /// it. With no hook this is exactly `free_sequence` under the write guard.
    fn finish_kv_transaction(
        &mut self,
        backend: &(dyn AienInferenceBackend + '_),
        seq_id: SequenceId,
        reason: FinishReason,
    ) {
        let request_id = seq_id.to_u64();
        let hook = match &self.finish_hook {
            Some(hook) if reason != FinishReason::Preempted && !self.arena.is_stale(seq_id) => {
                Some(hook.clone())
            }
            _ => None,
        };
        let Some(hook) = hook else {
            let _ = self.kv_manager.write().free_sequence(request_id);
            return;
        };
        let computed_prefix = match backend.computed_prefix(request_id) {
            Some(prefix) => Some(prefix),
            None if !backend.manages_kv_cache() => self.computed_prefix(seq_id),
            None => None,
        };
        let prompt = self.arena.get(seq_id).map(|rec| rec.prompt.clone());
        let ctx = FinishContext {
            seq_id,
            request_id,
            reason,
            computed_prefix,
            prompt: prompt.as_deref().map_or(&[][..], |p| p),
        };
        let mut kv = self.kv_manager.write();
        if hook.on_finish(&mut kv, &ctx).is_err() {
            self.metrics.finish_hook_failures += 1;
        }
        let _ = kv.free_sequence(request_id);
    }

    fn is_awaiting_restore(&self, id: SequenceId) -> bool {
        self.arena
            .get(id)
            .is_some_and(|rec| rec.phase == SequencePhase::AwaitingRestore)
    }

    /// Persistent KV (ADR 0003 section 6.3): holds a request that is waiting in
    /// the queue while a restore of its prefix is in flight. It stays in the
    /// waiting queue in its place, phase `AwaitingRestore`, and admission skips
    /// it without blocking the requests behind it. Refused (nothing changed) for
    /// an id this arena no longer knows, or one that is not a plain waiting
    /// request in the waiting queue.
    pub fn hold_for_restore(&mut self, seq_id: SequenceId) -> Result<(), String> {
        let rec = self
            .arena
            .get(seq_id)
            .ok_or_else(|| format!("hold_for_restore: sequence {seq_id} is stale or unknown"))?;
        if rec.phase != SequencePhase::Waiting || !self.waiting_queue.contains(&seq_id) {
            return Err(format!(
                "hold_for_restore: sequence {seq_id} is {:?}, not a waiting request",
                rec.phase
            ));
        }
        if let Some(rec) = self.arena.get_mut(seq_id) {
            rec.phase = SequencePhase::AwaitingRestore;
        }
        Ok(())
    }

    /// Ends a hold without restoring (the restore missed, failed or was
    /// stale): the request is `Waiting` again, in its original place, and will
    /// be admitted with a plain prefill from zero. Refused for an id that is
    /// stale or not held.
    pub fn release_restore(&mut self, seq_id: SequenceId) -> Result<(), String> {
        if !self.is_awaiting_restore(seq_id) {
            return Err(format!(
                "release_restore: sequence {seq_id} is stale or not awaiting a restore"
            ));
        }
        if let Some(rec) = self.arena.get_mut(seq_id) {
            rec.phase = SequencePhase::Waiting;
        }
        Ok(())
    }

    /// Requests currently held by `hold_for_restore`.
    pub fn awaiting_restore_count(&self) -> usize {
        self.waiting_queue
            .iter()
            .filter(|&&id| self.is_awaiting_restore(id))
            .count()
    }

    /// Persistent KV (ADR 0003 section 6.3, review finding 4): admits a held
    /// request whose first `restored` prompt tokens already have K/V in its
    /// block table (assembled by the caller from restored chunks). It bypasses
    /// the waiting-queue admission, which would take a ready table straight to
    /// decode and so skip the unprocessed suffix. Instead the sequence enters
    /// `running_sequences` in phase `Prefill` with `prompt_tokens_prefilled =
    /// restored` and `is_prefilled = false`, so the ordinary "continuing
    /// chunked prefill" branch schedules `prompt[restored..]` and the ordinary
    /// completion fence marks the table ready after the last chunk. The table is
    /// left `PrefillPending`, never completed here.
    ///
    /// Everything is checked first and any refusal changes nothing: the id must
    /// be live and awaiting a restore, `0 < restored < prompt.len()` (at least
    /// one token is always computed, so the fence always runs and the logits
    /// come from this prompt), and the KV table must exist, be `PrefillPending`
    /// and hold exactly `restored` tokens. No reference count is touched.
    pub fn admit_restored_prefix(
        &mut self,
        seq_id: SequenceId,
        restored: usize,
    ) -> Result<(), String> {
        let rec = self
            .arena
            .get(seq_id)
            .filter(|rec| rec.phase == SequencePhase::AwaitingRestore)
            .ok_or_else(|| {
                format!(
                    "admit_restored_prefix: sequence {seq_id} is stale or not awaiting a restore"
                )
            })?;
        let prompt_len = rec.prompt.len();
        if restored == 0 || restored >= prompt_len {
            return Err(format!(
                "admit_restored_prefix: restored {restored} must be in 1..{prompt_len} \
                 so at least one prompt token is computed"
            ));
        }
        {
            let kv = self.kv_manager.read();
            let table = kv.get_block_table(seq_id.to_u64()).ok_or_else(|| {
                format!("admit_restored_prefix: sequence {seq_id} has no KV block table")
            })?;
            if table.prefill_state != PrefillState::PrefillPending {
                return Err(format!(
                    "admit_restored_prefix: table is {:?}, expected PrefillPending",
                    table.prefill_state
                ));
            }
            if table.total_tokens != restored {
                return Err(format!(
                    "admit_restored_prefix: table holds {} tokens, restored is {restored}",
                    table.total_tokens
                ));
            }
        }
        let pos = self
            .waiting_queue
            .iter()
            .position(|&id| id == seq_id)
            .ok_or_else(|| {
                format!("admit_restored_prefix: sequence {seq_id} is not in the waiting queue")
            })?;
        self.waiting_queue.remove(pos);
        if let Some(rec) = self.arena.get_mut(seq_id) {
            rec.phase = SequencePhase::Prefill;
            rec.prompt_tokens_prefilled = restored;
            rec.is_prefilled = false;
        }
        self.running_sequences.push(seq_id);
        Ok(())
    }

    /// Index in the waiting queue of the first request admission may consider:
    /// requests held for a restore are skipped, not dropped, and do not block
    /// the ones behind them. With nothing held this is always 0.
    fn first_admittable_waiting(&self) -> Option<usize> {
        self.waiting_queue
            .iter()
            .position(|&id| !self.is_awaiting_restore(id))
    }

    /// Discards stale items from queue heads and internal lists without panics.
    fn purge_stale_work(&mut self) {
        while let Some(&id) = self.waiting_queue.front() {
            if self.arena.is_stale(id) {
                self.waiting_queue.pop_front();
            } else {
                break;
            }
        }

        while let Some(&id) = self.preempted_queue.front() {
            if self.arena.is_stale(id) {
                self.preempted_queue.pop_front();
            } else {
                break;
            }
        }

        let arena = &self.arena;
        self.running_sequences.retain(|&id| !arena.is_stale(id));
    }

    fn kv_reading(&self) -> KvReading {
        let kv = self.kv_manager.read();
        let total_blocks = kv.total_block_count();
        let free_blocks = kv.available_blocks();
        KvReading {
            total_blocks,
            free_blocks,
            allocated_blocks: total_blocks - free_blocks,
            active_tables: kv.active_sequence_count(),
        }
    }

    /// Builds the next scheduled batch enforcing chunked prefill budgets and watermark preemption.
    ///
    /// DUAL `serve.admit_preempt`: when an observer is installed, the decision
    /// is traced while it is made and one `AdmitPreemptRecord` is handed to the
    /// observer after it is final. Without an observer no trace exists and the
    /// decision code path is the same as before the tap.
    pub fn build_scheduled_batch(&mut self) -> Result<Option<ScheduledBatch>, String> {
        let mut trace = self.observer.is_some().then(AdmitPreemptTrace::default);
        let result = self.build_scheduled_batch_traced(&mut trace);
        if let (Some(observer), Some(trace)) = (self.observer.clone(), trace) {
            let outcome = match &result {
                Ok(Some(batch)) => BatchOutcome::Batch {
                    prefill_requests: batch.prefill_requests.len(),
                    decode_requests: batch.decode_requests.len(),
                    batch_tokens: batch
                        .prefill_requests
                        .iter()
                        .map(|r| r.prompt_tokens.len())
                        .sum::<usize>()
                        + batch.decode_requests.len(),
                },
                Ok(None) => BatchOutcome::Empty,
                Err(e) => BatchOutcome::Error(e.clone()),
            };
            let record = AdmitPreemptRecord {
                site: SERVE_ADMIT_PREEMPT_SITE,
                step_id: self.step_id,
                limits: BatchLimits {
                    max_batch_size: self.config.max_batch_size,
                    max_batch_tokens: self.config.max_batch_tokens,
                    max_prefill_tokens: self.config.max_prefill_tokens,
                    prefill_chunk_size: self.config.prefill_chunk_size,
                    chunk_prefill: self.config.chunk_prefill,
                    watermark_blocks: self.config.watermark_blocks,
                },
                arena_active_at_entry: trace.arena_active_at_entry,
                kv_at_entry: trace.kv_at_entry,
                below_watermark_at_entry: trace.below_watermark_at_entry,
                kv_at_exit: self.kv_reading(),
                running_count_at_exit: self.running_sequences.len(),
                waiting_count_at_exit: self.waiting_queue.len(),
                preempted_count_at_exit: self.preempted_queue.len(),
                alternatives: trace.alternatives,
                choice: AdmitPreemptChoice {
                    preempted: trace.preempted,
                    decode_scheduled: trace.decode_scheduled,
                    prefill_scheduled: trace.prefill_scheduled,
                    reprefill_requeued: trace.reprefill_requeued,
                    outcome,
                },
            };
            observer.observe_admit_preempt(&record);
        }
        result
    }

    fn build_scheduled_batch_traced(
        &mut self,
        trace: &mut Option<AdmitPreemptTrace>,
    ) -> Result<Option<ScheduledBatch>, String> {
        self.step_id += 1;
        self.purge_stale_work();

        if let Some(t) = trace.as_mut() {
            t.arena_active_at_entry = self.arena.active_count();
            t.kv_at_entry = self.kv_reading();
            t.alternatives = AdmitPreemptAlternatives {
                running_at_entry: self.running_sequences.clone(),
                waiting_at_entry: self.waiting_queue.iter().copied().collect(),
                preempted_at_entry: self.preempted_queue.iter().copied().collect(),
                ..AdmitPreemptAlternatives::default()
            };
        }

        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();
        let mut block_tables = HashMap::new();

        let mut current_tokens = 0;
        let mut prefill_budget = self.config.max_prefill_tokens;
        let mut preempted_this_step = false;
        // New blocks promised to sequences admitted into decode this step
        // (copy-on-write on their first token) but not allocated until the backend runs.
        let mut reserved_blocks = 0usize;

        // 1. Watermark Memory Pressure Check: Preempt lowest priority sequence if below watermark
        {
            let kv = self.kv_manager.read();
            let available = kv.available_blocks();
            if let Some(t) = trace.as_mut() {
                t.below_watermark_at_entry =
                    available < self.config.watermark_blocks && !self.running_sequences.is_empty();
            }
            if available < self.config.watermark_blocks && !self.running_sequences.is_empty() {
                let mut candidates: Vec<(SequenceId, Priority)> = self
                    .running_sequences
                    .iter()
                    .filter_map(|&id| self.arena.get(id).map(|rec| (id, rec.priority)))
                    .collect();
                candidates.sort_by_key(|c| c.1 as u8);
                if let Some(t) = trace.as_mut() {
                    t.alternatives.preemption_victims_considered = candidates
                        .iter()
                        .map(|&(seq, p)| PreemptionVictim {
                            seq,
                            priority: p as u8,
                        })
                        .collect();
                }

                if let Some((preempt_id, _)) = candidates.first() {
                    let preempt_id = *preempt_id;
                    drop(kv);
                    if let Some(pos) = self.running_sequences.iter().position(|&x| x == preempt_id)
                    {
                        if let Some(t) = trace.as_mut() {
                            t.preempted.push(preempt_id);
                        }
                        self.running_sequences.remove(pos);
                        let _ = self.kv_manager.write().free_sequence(preempt_id.to_u64());
                        self.pending_backend_releases.push(preempt_id);
                        if let Some(rec) = self.arena.get_mut(preempt_id) {
                            rec.phase = SequencePhase::Preempted;
                            rec.prompt_tokens_prefilled = 0;
                            rec.is_prefilled = false;
                        }
                        self.preempted_queue.push_back(preempt_id);
                        self.metrics.preempted_requests += 1;
                        preempted_this_step = true;
                    }
                }
            }
        }

        // 2. Schedule active decode sequences and in-flight prefill chunks
        let mut running_ids = self.running_sequences.clone();
        running_ids.sort();
        let mut needs_prefill: Vec<SequenceId> = Vec::new();

        for seq_id in running_ids {
            if decode_requests.len() + prefill_requests.len() >= self.config.max_batch_size {
                if let Some(t) = trace.as_mut() {
                    t.alternatives.running_loop_stop = Some(LoopStop::BatchSizeReached);
                }
                break;
            }
            if current_tokens >= self.config.max_batch_tokens {
                if let Some(t) = trace.as_mut() {
                    t.alternatives.running_loop_stop = Some(LoopStop::BatchTokensReached);
                }
                break;
            }

            let seq = match self.arena.get_mut(seq_id) {
                Some(s) => s,
                None => {
                    if let Some(t) = trace.as_mut() {
                        t.alternatives.running_examined.push(RunningItem {
                            seq: seq_id,
                            verdict: RunningVerdict::NoRecord,
                        });
                    }
                    continue;
                }
            };

            if seq.is_prefilled {
                // Active decode step, gated on computed K/V (PREFILL-GATE).
                let table_state = self
                    .kv_manager
                    .read()
                    .get_block_table(seq_id.to_u64())
                    .map(|t| (t.block_ids.clone(), t.is_prefill_ready()));
                let mut verdict = RunningVerdict::DecodeNoTable;
                if let Some((blocks, ready)) = table_state {
                    if prefill_gate_allows_decode(ready) {
                        block_tables.insert(seq_id.to_u64(), blocks);
                        decode_requests.push(seq_id.to_u64());
                        current_tokens += 1;
                        verdict = RunningVerdict::Decode;
                    } else {
                        // The table exists but its K/V was never computed:
                        // send the sequence back for a real prefill, never decode.
                        seq.is_prefilled = false;
                        seq.prompt_tokens_prefilled = 0;
                        seq.phase = SequencePhase::Waiting;
                        needs_prefill.push(seq_id);
                        verdict = RunningVerdict::GateRefusedReprefill;
                    }
                }
                if let Some(t) = trace.as_mut() {
                    if verdict == RunningVerdict::Decode {
                        t.decode_scheduled.push(seq_id);
                    }
                    t.alternatives.running_examined.push(RunningItem {
                        seq: seq_id,
                        verdict,
                    });
                }
            } else {
                // Continuing chunked prefill
                let total_prompt_len = seq.prefill_input_len();
                let remaining = total_prompt_len.saturating_sub(seq.prompt_tokens_prefilled);
                let chunk_size = std::cmp::min(remaining, self.config.prefill_chunk_size);
                let chunk_size = std::cmp::min(chunk_size, prefill_budget);

                if chunk_size > 0 {
                    let start = seq.prompt_tokens_prefilled;
                    let end = start + chunk_size;
                    let chunk_tokens = seq.prefill_input()[start..end].to_vec();

                    let chunk_req = SequenceRequest {
                        request_id: seq_id.to_u64(),
                        prompt_tokens: chunk_tokens,
                        sampling_params: seq.sampling_params.clone(),
                        arrival_time_ns: seq.arrival_time_ns,
                        priority: seq.priority as u8,
                    };

                    if let Some(table) = self.kv_manager.read().get_block_table(seq_id.to_u64()) {
                        block_tables.insert(seq_id.to_u64(), table.block_ids.clone());
                    }

                    prefill_requests.push(chunk_req);
                    current_tokens += chunk_size;
                    prefill_budget = prefill_budget.saturating_sub(chunk_size);
                }
                if let Some(t) = trace.as_mut() {
                    let verdict = if chunk_size > 0 {
                        t.prefill_scheduled.push((seq_id, chunk_size));
                        RunningVerdict::ContinuePrefill {
                            chunk_len: chunk_size,
                        }
                    } else {
                        RunningVerdict::ContinuePrefillNoChunk
                    };
                    t.alternatives.running_examined.push(RunningItem {
                        seq: seq_id,
                        verdict,
                    });
                }
            }
        }

        // Sequences refused decode by the prefill gate: drop the uncomputed
        // (possibly shared) table and requeue at the front for a fresh prefill.
        for seq_id in needs_prefill {
            if let Some(t) = trace.as_mut() {
                t.reprefill_requeued.push(seq_id);
            }
            self.running_sequences.retain(|&x| x != seq_id);
            let _ = self.kv_manager.write().free_sequence(seq_id.to_u64());
            self.pending_backend_releases.push(seq_id);
            self.waiting_queue.push_front(seq_id);
        }

        // 3. Admit waiting or preempted requests if capacity permits and not under preemption pressure
        if !preempted_this_step {
            while decode_requests.len() + prefill_requests.len() < self.config.max_batch_size
                && current_tokens < self.config.max_batch_tokens
                && prefill_budget > 0
            {
                // Requests held for a restore are skipped, not dropped, and do not
                // block the ones behind them. With nothing held the index is 0.
                let waiting_idx = self.first_admittable_waiting();
                let (next_seq_id, is_preempted) = if let Some(&id) = self.preempted_queue.front() {
                    (id, true)
                } else if let Some(idx) = waiting_idx {
                    (self.waiting_queue[idx], false)
                } else {
                    break;
                };
                let waiting_idx = waiting_idx.unwrap_or(0);

                if self.arena.is_stale(next_seq_id) {
                    if let Some(t) = trace.as_mut() {
                        t.alternatives
                            .admission_examined
                            .push(AdmissionCandidate::before_kv(
                                next_seq_id,
                                is_preempted,
                                AdmissionVerdict::StaleDropped,
                            ));
                    }
                    if is_preempted {
                        self.preempted_queue.pop_front();
                    } else {
                        self.waiting_queue.remove(waiting_idx);
                    }
                    continue;
                }

                let prompt_len = match self.arena.get(next_seq_id) {
                    Some(s) => s.prefill_input_len(),
                    None => {
                        if let Some(t) = trace.as_mut() {
                            t.alternatives
                                .admission_examined
                                .push(AdmissionCandidate::before_kv(
                                    next_seq_id,
                                    is_preempted,
                                    AdmissionVerdict::MissingRecord,
                                ));
                        }
                        if is_preempted {
                            self.preempted_queue.pop_front();
                        } else {
                            self.waiting_queue.remove(waiting_idx);
                        }
                        continue;
                    }
                };

                // The KV manager is the only authority on how many NEW physical
                // blocks this sequence needs (its own pool block size, blocks it
                // already owns or shares, a pending copy-on-write). The need is
                // read BEFORE the sequence leaves its queue.
                let (needed_blocks, free_blocks, pool_blocks) = {
                    let kv = self.kv_manager.read();
                    (
                        kv.incremental_blocks_needed(next_seq_id.to_u64(), prompt_len),
                        kv.available_blocks(),
                        kv.total_block_count(),
                    )
                };
                // Blocks promised to sequences admitted earlier in this step
                // that have not allocated them yet (forks entering decode).
                let available = free_blocks.saturating_sub(reserved_blocks);

                // Preempted sequences require headroom above watermark to prevent thrashing
                let watermark_headroom = if is_preempted {
                    self.config.watermark_blocks
                } else {
                    0
                };
                let min_needed = needed_blocks + watermark_headroom;

                // DUAL tap: one AdmissionCandidate per queue head examined, with
                // the exact numbers compared above. Expands to nothing when no
                // observer is installed (`trace` is None).
                macro_rules! trace_admission {
                    ($verdict:expr) => {
                        if let Some(t) = trace.as_mut() {
                            t.alternatives.admission_examined.push(AdmissionCandidate {
                                seq: next_seq_id,
                                from_preempted_queue: is_preempted,
                                prompt_len,
                                needed_blocks,
                                free_blocks,
                                reserved_blocks,
                                watermark_headroom,
                                min_needed,
                                pool_blocks,
                                verdict: $verdict,
                            });
                        }
                    };
                }

                if available < min_needed {
                    // Waiting is only honest while something can still free blocks.
                    // A request larger than the whole pool, or one that waits with
                    // nothing running and nothing scheduled, would wait forever.
                    let nothing_can_free = self.running_sequences.is_empty()
                        && decode_requests.is_empty()
                        && prefill_requests.is_empty();
                    if min_needed > pool_blocks || nothing_can_free {
                        trace_admission!(AdmissionVerdict::RefusedPoolExhausted);
                        return Err(format!(
                            "{KV_POOL_EXHAUSTED_PREFIX} admission of sequence {} needs {} new KV \
                             blocks (plus {} watermark) but only {} are free ({} in the pool) and \
                             nothing is running that could free more; the request stays queued",
                            next_seq_id.to_u64(),
                            needed_blocks,
                            watermark_headroom,
                            available,
                            pool_blocks,
                        ));
                    }
                    trace_admission!(AdmissionVerdict::WaitForBlocks);
                    break;
                }

                if is_preempted {
                    self.preempted_queue.pop_front();
                } else {
                    self.waiting_queue.remove(waiting_idx);
                }

                let seq = match self.arena.get_mut(next_seq_id) {
                    Some(s) => s,
                    None => continue,
                };

                let chunk_size = if self.config.chunk_prefill {
                    std::cmp::min(prompt_len, self.config.prefill_chunk_size)
                } else {
                    prompt_len
                };
                let chunk_size = std::cmp::min(chunk_size, prefill_budget);

                // Scheduler Zero-Copy Fork Invariant: Sequence admission must inspect existing block tables in KvCache.
                // Pre-forked branches skip redundant prefill and enter decode directly ONLY when the
                // shared prefix K/V was computed by a completed prefill (PREFILL-GATE). A block table
                // alone proves allocation, not computation: allocate_sequence only reserves and zeroes.
                let existing_table = self
                    .kv_manager
                    .read()
                    .get_block_table(next_seq_id.to_u64())
                    .map(|t| (t.block_ids.clone(), t.is_prefill_ready()));

                match existing_table {
                    Some((blocks, ready)) if prefill_gate_allows_decode(ready) => {
                        block_tables.insert(next_seq_id.to_u64(), blocks);
                        seq.is_prefilled = true;
                        seq.phase = SequencePhase::Decode;
                        seq.prompt_tokens_prefilled = prompt_len;
                        decode_requests.push(next_seq_id.to_u64());
                        current_tokens += 1;
                        trace_admission!(AdmissionVerdict::AdmittedDecode);
                        if let Some(t) = trace.as_mut() {
                            t.decode_scheduled.push(next_seq_id);
                        }
                        reserved_blocks += needed_blocks;
                        self.running_sequences.push(next_seq_id);
                        // MUTANT ONLY (feature `dual_observer_mutant_changes_decision`):
                        // an observer's presence stops admission after one grant.
                        // The parity tests must FAIL with this feature on.
                        if cfg!(feature = "dual_observer_mutant_changes_decision")
                            && trace.is_some()
                        {
                            break;
                        }
                        continue;
                    }
                    Some(_) => {
                        // Allocated but not computed: drop the table (and its share of
                        // any parent blocks) and fall through to allocate + prefill.
                        let _ = self.kv_manager.write().free_sequence(next_seq_id.to_u64());
                    }
                    None => {}
                }

                if chunk_size == 0 {
                    trace_admission!(AdmissionVerdict::ChunkBudgetExhausted);
                    if is_preempted {
                        self.preempted_queue.push_front(next_seq_id);
                    } else {
                        self.waiting_queue.insert(waiting_idx, next_seq_id);
                    }
                    break;
                }

                // Allocate blocks for admitted sequence
                let assigned_blocks = self
                    .kv_manager
                    .write()
                    .allocate_sequence(next_seq_id.to_u64(), &seq.prefill_input())?;
                self.kv_manager
                    .write()
                    .mark_prefill_pending(next_seq_id.to_u64())
                    .map_err(|e| e.to_string())?;

                let new_blocks = assigned_blocks.len();
                block_tables.insert(next_seq_id.to_u64(), assigned_blocks);

                let chunk_tokens = seq.prefill_input()[0..chunk_size].to_vec();
                let chunk_req = SequenceRequest {
                    request_id: next_seq_id.to_u64(),
                    prompt_tokens: chunk_tokens,
                    sampling_params: seq.sampling_params.clone(),
                    arrival_time_ns: seq.arrival_time_ns,
                    priority: seq.priority as u8,
                };

                prefill_requests.push(chunk_req);
                current_tokens += chunk_size;
                prefill_budget = prefill_budget.saturating_sub(chunk_size);

                seq.phase = SequencePhase::Prefill;
                self.running_sequences.push(next_seq_id);
                trace_admission!(AdmissionVerdict::AdmittedPrefill {
                    chunk_len: chunk_size,
                    new_blocks,
                });
                if let Some(t) = trace.as_mut() {
                    t.prefill_scheduled.push((next_seq_id, chunk_size));
                }
                // MUTANT ONLY (feature `dual_observer_mutant_changes_decision`),
                // see the decode admission above.
                if cfg!(feature = "dual_observer_mutant_changes_decision") && trace.is_some() {
                    break;
                }
            }
        }

        if prefill_requests.is_empty() && decode_requests.is_empty() {
            return Ok(None);
        }

        Ok(Some(ScheduledBatch {
            prefill_requests,
            decode_requests,
            block_tables,
            step_id: self.step_id,
        }))
    }

    /// Builds a structured zero-copy BatchPlan for native batch engines.
    pub fn build_batch_plan(&mut self) -> Result<Option<BatchPlan>, String> {
        let batch = match self.build_scheduled_batch()? {
            Some(b) => b,
            None => return Ok(None),
        };

        let mut prefill_spans = Vec::new();
        let mut decode_items = Vec::new();
        let mut total_tokens = 0;

        for req in &batch.prefill_requests {
            if let Ok(seq_id) = SequenceId::from_u64(req.request_id) {
                if let Some(record) = self.arena.get(seq_id) {
                    let length = req.prompt_tokens.len();
                    prefill_spans.push(PrefillSpan {
                        seq_id,
                        start_pos: record.prompt_tokens_prefilled,
                        length,
                    });
                    total_tokens += length;
                }
            }
        }

        for &req_id in &batch.decode_requests {
            if let Ok(seq_id) = SequenceId::from_u64(req_id) {
                if let Some(record) = self.arena.get(seq_id) {
                    decode_items.push(DecodeItem {
                        seq_id,
                        token_pos: record.total_tokens(),
                    });
                    total_tokens += 1;
                }
            }
        }

        Ok(Some(BatchPlan {
            step_id: batch.step_id,
            prefill_spans,
            decode_items,
            total_tokens,
        }))
    }

    /// Runs a model prefill over `prompt` for a KV sequence this scheduler does
    /// not own (a swarm root, which is never decoded), then fires the
    /// completion fence. The prompt is sent in chunks of `prefill_chunk_size`
    /// (capped by `max_prefill_tokens`). PrefillReady is set only after every
    /// chunk's `execute_step` returned Ok and the backend reported processing
    /// the whole chunk; any error leaves the table not ready.
    ///
    /// Returns Ok(None) when the table is already ready (nothing to compute),
    /// Ok(Some(metrics)) after a completed prefill.
    pub async fn prefill_detached(
        &mut self,
        backend: &mut (dyn AienInferenceBackend + '_),
        kv_seq_id: u64,
        prompt: &[u32],
    ) -> Result<Option<StepMetrics>, String> {
        let blocks = {
            let mut kv = self.kv_manager.write();
            if kv.get_block_table(kv_seq_id).is_none() {
                kv.allocate_sequence(kv_seq_id, prompt)?;
            }
            let table = kv.get_block_table(kv_seq_id).ok_or_else(|| {
                PrefillGateError::UnknownSequence { seq_id: kv_seq_id }.to_string()
            })?;
            if table.is_prefill_ready() {
                return Ok(None);
            }
            let blocks = table.block_ids.clone();
            kv.mark_prefill_pending(kv_seq_id)
                .map_err(|e| e.to_string())?;
            blocks
        };

        let chunk_len = self
            .config
            .prefill_chunk_size
            .min(self.config.max_prefill_tokens)
            .max(1);
        let t0 = Instant::now();
        let mut prefill_tokens = 0usize;

        for span in prompt.chunks(chunk_len) {
            self.step_id += 1;
            let mut block_tables = HashMap::new();
            block_tables.insert(kv_seq_id, blocks.clone());
            let batch = ScheduledBatch {
                prefill_requests: vec![SequenceRequest {
                    request_id: kv_seq_id,
                    prompt_tokens: span.to_vec(),
                    sampling_params: aien_abi_core::SamplingParams::default(),
                    arrival_time_ns: 0,
                    priority: Priority::Normal as u8,
                }],
                decode_requests: Vec::new(),
                block_tables,
                step_id: self.step_id,
            };
            let (_outputs, metrics) = backend.execute_step(&batch).await?;
            if metrics.prefill_tokens_processed < span.len() {
                return Err(format!(
                    "prefill gate: backend processed {} of {} prompt tokens for sequence {}; not marking ready",
                    metrics.prefill_tokens_processed,
                    span.len(),
                    kv_seq_id
                ));
            }
            prefill_tokens += span.len();
        }

        // PREFILL-GATE completion fence: the forward over the whole prompt span completed.
        self.kv_manager
            .write()
            .complete_prefill(kv_seq_id)
            .map_err(|e| e.to_string())?;

        self.metrics.total_prefill_tokens += prefill_tokens as u64;
        let active_kv_blocks = self.kv_manager.read().allocated_block_count();
        Ok(Some(StepMetrics {
            prefill_tokens_processed: prefill_tokens,
            decode_tokens_emitted: 0,
            step_latency_us: t0.elapsed().as_micros() as u64,
            active_kv_blocks,
        }))
    }

    /// Advances the engine by one step using the configured backend.
    pub async fn step(
        &mut self,
        backend: &mut (dyn AienInferenceBackend + '_),
    ) -> Result<Option<(Vec<DecodeOutput>, StepMetrics)>, String> {
        let batch = self.build_scheduled_batch()?;
        // Sequences preempted (or sent back for a fresh prefill) since the last
        // step: their KV is freed and no step is executing, so the backend may
        // drop its per-sequence state now. Done before `execute_step`, so a
        // resumed sequence is prefilled from position 0, not a stale one.
        for seq_id in self.take_pending_backend_releases() {
            backend.release_sequence(seq_id.to_u64())?;
        }
        let Some(batch) = batch else {
            return Ok(None);
        };

        let t0 = Instant::now();
        let (outputs, metrics) = backend.execute_step(&batch).await?;
        let step_latency = t0.elapsed().as_micros() as u64;

        // Update prefill progress for chunked sequences
        for prefill_req in &batch.prefill_requests {
            if let Ok(seq_id) = SequenceId::from_u64(prefill_req.request_id) {
                if let Some(seq) = self.arena.get_mut(seq_id) {
                    if !seq.is_prefilled {
                        seq.prompt_tokens_prefilled += prefill_req.prompt_tokens.len();
                        if seq.prompt_tokens_prefilled >= seq.prefill_input_len() {
                            seq.is_prefilled = true;
                            seq.phase = SequencePhase::Decode;
                            // PREFILL-GATE completion fence: execute_step returned Ok
                            // for the final prompt chunk, so the K/V is computed.
                            let _ = self
                                .kv_manager
                                .write()
                                .complete_prefill(prefill_req.request_id);
                        }
                    }
                }
            }
        }

        let mut final_outputs = Vec::new();

        for output in outputs {
            match output {
                DecodeOutput::Token {
                    request_id,
                    token_id,
                    logprob,
                } => {
                    let mut is_finished = false;
                    let mut finish_reason = FinishReason::StopToken;
                    let mut total_tokens = 0;
                    let mut should_emit_token = false;

                    if let Ok(seq_id) = SequenceId::from_u64(request_id) {
                        if let Some(seq) = self.arena.get_mut(seq_id) {
                            if seq.is_prefilled {
                                should_emit_token = true;
                                seq.tokens_generated += 1;
                                seq.generated_tokens.push(token_id);
                                total_tokens = seq.tokens_generated;

                                if seq.sampling_params.stop_token_ids.contains(&token_id) {
                                    is_finished = true;
                                    finish_reason = FinishReason::StopToken;
                                } else if seq.tokens_generated >= seq.sampling_params.max_tokens {
                                    is_finished = true;
                                    finish_reason = FinishReason::LengthLimit;
                                } else if !backend.manages_kv_cache() {
                                    // Append token in KV manager if backend does not manage it directly
                                    let append_result =
                                        self.kv_manager.write().append_token(request_id);
                                    if append_result.is_err() {
                                        is_finished = true;
                                        finish_reason = FinishReason::Preempted;
                                    }
                                }

                                if let Some(sink_id) = seq.sink_id {
                                    self.completion_router.emit(
                                        sink_id,
                                        CompletionEvent::Token {
                                            seq_id,
                                            token: token_id,
                                        },
                                    );
                                }
                            }
                        }

                        if is_finished {
                            if let Some(pos) =
                                self.running_sequences.iter().position(|&x| x == seq_id)
                            {
                                self.running_sequences.remove(pos);
                            }
                            let finished_sink = self.arena.get(seq_id).and_then(|seq| seq.sink_id);
                            if finish_reason == FinishReason::Preempted {
                                let _ = self.kv_manager.write().free_sequence(request_id);
                                self.pending_backend_releases.push(seq_id);
                                if let Some(seq) = self.arena.get_mut(seq_id) {
                                    seq.phase = SequencePhase::Preempted;
                                    seq.prompt_tokens_prefilled = 0;
                                    seq.is_prefilled = false;
                                }
                                self.preempted_queue.push_back(seq_id);
                                self.metrics.preempted_requests += 1;
                            } else {
                                // Finish transaction: hook, then free, one KV guard (ADR 0003 6.1).
                                self.finish_kv_transaction(&*backend, seq_id, finish_reason);
                                self.arena.free_sequence(seq_id);
                                self.metrics.finished_requests += 1;
                            }

                            final_outputs.push(DecodeOutput::Finished {
                                request_id,
                                reason: finish_reason,
                                total_tokens,
                            });
                            // sc#294: taken on every finish, listened to or not.
                            let decoding = backend.take_decode_observation(request_id);
                            // sc#337: the op counters at this finish, listened to or not.
                            let ops = backend.op_evidence();
                            if let Some(sink_id) = finished_sink {
                                self.completion_router.emit(
                                    sink_id,
                                    CompletionEvent::Finished {
                                        seq_id,
                                        finish_reason,
                                        total_tokens,
                                        decoding,
                                        ops,
                                    },
                                );
                            }
                        } else if should_emit_token {
                            final_outputs.push(DecodeOutput::Token {
                                request_id,
                                token_id,
                                logprob,
                            });
                        }
                    }
                }
                DecodeOutput::Finished {
                    request_id,
                    reason,
                    total_tokens,
                } => {
                    if let Ok(seq_id) = SequenceId::from_u64(request_id) {
                        if let Some(pos) = self.running_sequences.iter().position(|&x| x == seq_id)
                        {
                            self.running_sequences.remove(pos);
                        }
                        let finished_sink = self.arena.get(seq_id).and_then(|seq| seq.sink_id);
                        // Backend-originated finish: same transaction (ADR 0003 6.1).
                        self.finish_kv_transaction(&*backend, seq_id, reason);
                        self.arena.free_sequence(seq_id);
                        self.metrics.finished_requests += 1;
                        // sc#294: taken on every finish, listened to or not.
                        let decoding = backend.take_decode_observation(request_id);
                        // sc#337: the op counters at this finish, listened to or not.
                        let ops = backend.op_evidence();
                        if let Some(sink_id) = finished_sink {
                            self.completion_router.emit(
                                sink_id,
                                CompletionEvent::Finished {
                                    seq_id,
                                    finish_reason: reason,
                                    total_tokens,
                                    decoding,
                                    ops,
                                },
                            );
                        }
                    }
                    final_outputs.push(DecodeOutput::Finished {
                        request_id,
                        reason,
                        total_tokens,
                    });
                }
            }
        }

        self.metrics.total_steps += 1;
        self.metrics.total_prefill_tokens += metrics.prefill_tokens_processed as u64;
        self.metrics.total_decode_tokens += metrics.decode_tokens_emitted as u64;

        if metrics.prefill_tokens_processed > 0 && self.config.chunk_prefill {
            self.metrics.chunked_prefill_steps += 1;
        }

        let total_steps = self.metrics.total_steps as f64;
        self.metrics.avg_step_latency_us = (self.metrics.avg_step_latency_us * (total_steps - 1.0)
            + step_latency as f64)
            / total_steps;

        let active_kv_blocks = self.kv_manager.read().allocated_block_count();

        Ok(Some((
            final_outputs,
            StepMetrics {
                prefill_tokens_processed: metrics.prefill_tokens_processed,
                decode_tokens_emitted: metrics.decode_tokens_emitted,
                step_latency_us: step_latency,
                active_kv_blocks,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_abi_core::SamplingParams;
    use aien_inference_abi::MockInferenceBackend;
    use aien_kv_cache::create_shared_kv_manager;

    #[tokio::test]
    async fn test_scheduler_lifecycle_and_stale_work_rejection() {
        let kv_manager = create_shared_kv_manager(64, 16);
        let config = SchedulerConfig::default();
        let mut scheduler = AienScheduler::new(config, kv_manager);
        let mut backend = MockInferenceBackend::new(10);

        let req = SequenceRequest {
            request_id: 1,
            prompt_tokens: vec![1, 2, 3, 4],
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 3,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        };

        let seq_id = scheduler.submit_request(req);
        assert_eq!(scheduler.waiting_count(), 1);

        // Step 1: Admits and prefills (emits token 1)
        let res1 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res1.0.len(), 1);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(res1.1.prefill_tokens_processed, 4);

        // Manually simulate a stale work ticket in waiting queue:
        let fake_stale_id = SequenceId::new(seq_id.slot, seq_id.generation + 10).unwrap();
        scheduler.waiting_queue.push_back(fake_stale_id);

        // Step 2: Executes decode token 2 and silently discards fake_stale_id without error or panic
        let res2 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res2.0.len(), 1);
        assert_eq!(scheduler.waiting_count(), 0); // Stale work was discarded

        // Step 3: Hits max_tokens (3) and finishes
        let res3 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res3.0.len(), 1);
        match &res3.0[0] {
            DecodeOutput::Finished {
                request_id,
                reason,
                total_tokens,
            } => {
                assert_eq!(*request_id, seq_id.to_u64());
                assert_eq!(*reason, FinishReason::LengthLimit);
                assert_eq!(*total_tokens, 3);
            }
            _ => panic!("Expected Finished output"),
        }
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.metrics().finished_requests, 1);
    }

    #[tokio::test]
    async fn test_chunked_prefill_segmentation() {
        let kv_manager = create_shared_kv_manager(100, 16);
        let config = SchedulerConfig {
            max_batch_size: 16,
            max_batch_tokens: 1024,
            max_prefill_tokens: 512,
            prefill_chunk_size: 64,
            chunk_prefill: true,
            watermark_blocks: 2,
        };
        let mut scheduler = AienScheduler::new(config, kv_manager);
        let mut backend = MockInferenceBackend::new(10);

        let req = SequenceRequest {
            request_id: 100,
            prompt_tokens: (0..128).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 2,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        };

        scheduler.submit_request(req);

        // Step 1: Processes chunk 1 (64 tokens)
        let res1 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res1.1.prefill_tokens_processed, 64);
        assert_eq!(scheduler.running_count(), 1);

        // Step 2: Processes chunk 2 (64 tokens, completes prompt)
        let res2 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res2.1.prefill_tokens_processed, 64);

        // Step 3: Decode begins
        let res3 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res3.1.decode_tokens_emitted, 1);
    }

    #[tokio::test]
    async fn test_watermark_pressure_preemption() {
        let kv_manager = create_shared_kv_manager(6, 16);
        let config = SchedulerConfig {
            max_batch_size: 4,
            max_batch_tokens: 1024,
            max_prefill_tokens: 512,
            prefill_chunk_size: 128,
            chunk_prefill: false,
            watermark_blocks: 3,
        };
        let mut scheduler = AienScheduler::new(config, kv_manager.clone());
        let mut backend = MockInferenceBackend::new(10);

        scheduler.submit_request(SequenceRequest {
            request_id: 1,
            prompt_tokens: (0..32).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 10,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 5,
        });

        scheduler.submit_request(SequenceRequest {
            request_id: 2,
            prompt_tokens: (0..32).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 10,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        });

        // Step 1: Admits both
        scheduler.step(&mut backend).await.unwrap();

        // Step 2: Memory pressure triggers preemption of lower priority
        scheduler.step(&mut backend).await.unwrap();
        assert_eq!(scheduler.metrics().preempted_requests, 1);
        assert_eq!(scheduler.preempted_count(), 1);
        assert_eq!(scheduler.running_count(), 1);
    }

    // PREFILL-GATE unit tests (new readiness API).

    fn gate_request(request_id: u64, prompt: Vec<u32>, max_tokens: usize) -> SequenceRequest {
        SequenceRequest {
            request_id,
            prompt_tokens: prompt,
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

    #[tokio::test]
    async fn test_prefill_gate_scheduler_fence_marks_ready_only_after_last_chunk() {
        let kv_manager = create_shared_kv_manager(100, 16);
        let config = SchedulerConfig {
            max_batch_size: 16,
            max_batch_tokens: 1024,
            max_prefill_tokens: 512,
            prefill_chunk_size: 64,
            chunk_prefill: true,
            watermark_blocks: 2,
        };
        let mut scheduler = AienScheduler::new(config, kv_manager.clone());
        let mut backend = MockInferenceBackend::new(0);

        let seq = scheduler.submit_request(gate_request(0, (0..128).collect(), 2));
        let id = seq.to_u64();

        // Chunk 1 of 2: prefill issued, not complete.
        scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(
            kv_manager.read().prefill_state(id),
            Some(aien_kv_cache::PrefillState::PrefillPending),
            "a half-prefilled table must not be ready"
        );

        // Chunk 2 of 2: completion fence fires.
        scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(
            kv_manager.read().prefill_state(id),
            Some(aien_kv_cache::PrefillState::PrefillReady)
        );

        let res3 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res3.1.decode_tokens_emitted, 1);
    }

    #[test]
    fn test_prefill_gate_ready_fork_enters_decode_directly() {
        // Positive control: a branch forked from a computed root decodes
        // without redundant prefill. The gate is not "never decode".
        let kv_manager = create_shared_kv_manager(64, 16);
        let mut scheduler = AienScheduler::new(SchedulerConfig::default(), kv_manager.clone());

        let root = SequenceId::new(1, 1).unwrap().to_u64();
        let child = SequenceId::new(2, 1).unwrap().to_u64();
        let prompt: Vec<u32> = (0..20).collect();
        {
            let mut kv = kv_manager.write();
            kv.allocate_sequence(root, &prompt).unwrap();
            kv.mark_prefill_pending(root).unwrap();
            kv.complete_prefill(root).unwrap();
            kv.fork_prefilled(root, child).unwrap();
        }

        scheduler.submit_request(gate_request(child, prompt, 4));
        let batch = scheduler.build_scheduled_batch().unwrap().unwrap();
        assert!(batch.decode_requests.contains(&child));
        assert!(batch.prefill_requests.is_empty());
    }

    #[tokio::test]
    async fn test_prefill_gate_running_sequence_with_unready_table_is_reprefilled() {
        let kv_manager = create_shared_kv_manager(64, 16);
        let mut scheduler = AienScheduler::new(SchedulerConfig::default(), kv_manager.clone());
        let mut backend = MockInferenceBackend::new(0);

        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        let seq = scheduler.submit_request(gate_request(0, prompt.clone(), 8));
        let id = seq.to_u64();
        scheduler.step(&mut backend).await.unwrap().unwrap();
        assert!(kv_manager
            .read()
            .get_block_table(id)
            .unwrap()
            .is_prefill_ready());
        // The step that completed the prefill also sampled (and emitted) the
        // first token, so the sequence now carries one generated token.
        let generated = scheduler.arena.get(seq).unwrap().generated_tokens.clone();
        assert_eq!(generated.len(), 1, "prefill completion samples one token");

        // Counterexample: the running sequence's table is replaced by an
        // allocated-but-uncomputed one. It must be prefilled again, not decoded.
        {
            let mut kv = kv_manager.write();
            kv.free_sequence(id).unwrap();
            kv.allocate_sequence(id, &prompt).unwrap();
        }
        let batch = scheduler.build_scheduled_batch().unwrap().unwrap();
        assert!(
            !batch.decode_requests.contains(&id),
            "PREFILL_GATE_VIOLATION: running sequence decoded over an uncomputed table"
        );
        // The re-prefill covers the prompt and the already-emitted token, so the
        // next sample is a new token. Prompt-only would re-sample and re-emit it.
        let expected: Vec<u32> = prompt.iter().copied().chain(generated).collect();
        assert!(batch
            .prefill_requests
            .iter()
            .any(|r| r.request_id == id && r.prompt_tokens == expected));
    }

    #[test]
    fn test_prefill_gate_reprefill_without_generated_tokens_is_exactly_the_prompt() {
        // A sequence refused by the gate before it generated anything must be
        // re-prefilled with exactly its prompt, nothing more.
        let kv_manager = create_shared_kv_manager(64, 16);
        let mut scheduler = AienScheduler::new(SchedulerConfig::default(), kv_manager.clone());

        let root = SequenceId::new(1, 1).unwrap().to_u64();
        let child = SequenceId::new(2, 1).unwrap().to_u64();
        let prompt: Vec<u32> = (0..20).collect();
        {
            let mut kv = kv_manager.write();
            kv.allocate_sequence(root, &prompt).unwrap();
            kv.mark_prefill_pending(root).unwrap();
            kv.complete_prefill(root).unwrap();
            kv.fork_prefilled(root, child).unwrap();
        }
        let seq = scheduler.submit_request(gate_request(child, prompt.clone(), 4));
        let batch = scheduler.build_scheduled_batch().unwrap().unwrap();
        assert!(batch.decode_requests.contains(&child));
        assert!(scheduler
            .arena
            .get(seq)
            .unwrap()
            .generated_tokens
            .is_empty());

        {
            let mut kv = kv_manager.write();
            kv.free_sequence(child).unwrap();
            kv.allocate_sequence(child, &prompt).unwrap();
        }
        let batch = scheduler.build_scheduled_batch().unwrap().unwrap();
        assert!(!batch.decode_requests.contains(&child));
        assert!(batch
            .prefill_requests
            .iter()
            .any(|r| r.request_id == child && r.prompt_tokens == prompt));
    }
}
