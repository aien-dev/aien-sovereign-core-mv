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

//! Canonical AienRuntimeSpine
//! Unifies SequenceArena, AienScheduler, AienKvManager, WorldStore, ContextComposer,
//! and SwarmManager in a single process address space.

use crate::context::ContextComposer;
use crate::control::{
    ControlCommand, ControlEnvelope, ControlResponse, RuntimeController, RuntimeStatusReport,
};
use crate::kv_residency::{KvDrainHandle, KvResidency, KvResidencyStatus};
use crate::sequence::{SequenceArena, SequenceId, SequenceState};
use crate::swarm::{SwarmConfig, SwarmManager, SwarmState};
use crate::world::WorldStore;

use aien_abi_core::{AienInferenceBackend, SamplingParams, StepMetrics};
use aien_kv_cache::AienKvManager;
use aien_scheduler::{
    AienScheduler, CompletionSink, CompletionSinkId, PromptHandle, SchedulerConfig,
};
use parking_lot::RwLock;
use std::sync::Arc;

pub struct AienRuntimeSpine {
    pub arena: SequenceArena,
    pub kv_manager: Arc<RwLock<AienKvManager>>,
    pub scheduler: AienScheduler,
    pub world_store: WorldStore,
    pub context_composer: ContextComposer,
    pub swarm_manager: SwarmManager,
    pub controller: RuntimeController,
    pub step_counter: u64,
    /// Subagent forks whose KV is already forked but whose backend state is
    /// not yet: (parent, child) KV ids, applied via the backend's
    /// `fork_sequence` at the start of the next `step`, before any decode.
    pending_backend_forks: Vec<(u64, u64)>,
    /// PREFILL-E2E C5: sequence ids whose KV and arena slots were reclaimed
    /// without a backend in reach (swarm cancel). Their backend per-sequence
    /// state is released by `release_pending_backend_sequences`, which `step`
    /// calls first.
    pending_backend_releases: Vec<u64>,
    /// Persistent KV (ADR 0003 section 6): park at the finish transaction,
    /// restore at submit. `None` (the default) is today's behaviour exactly.
    kv_residency: Option<KvResidency>,
    /// Why persistence was refused at start (ADR 0003 section 20: store key
    /// missing or malformed). `Some` only with the store dir set; then there
    /// is no residency and every prompt is prefilled.
    kv_persist_refused: Option<&'static str>,
}

impl AienRuntimeSpine {
    pub fn new(
        arena_capacity: usize,
        scheduler_config: SchedulerConfig,
        kv_manager: Arc<RwLock<AienKvManager>>,
    ) -> Self {
        let scheduler = AienScheduler::new(scheduler_config, kv_manager.clone());
        Self {
            arena: SequenceArena::new(arena_capacity),
            kv_manager,
            scheduler,
            world_store: WorldStore::new(),
            context_composer: ContextComposer::new(),
            swarm_manager: SwarmManager::new(),
            controller: RuntimeController::new(),
            step_counter: 0,
            pending_backend_forks: Vec::new(),
            pending_backend_releases: Vec::new(),
            kv_residency: None,
            kv_persist_refused: None,
        }
    }

    /// Submits structured InferenceWork with PromptHandle and optional completion sink.
    pub fn submit_inference_work(
        &mut self,
        work: aien_platform::queue::InferenceWork,
        prompt: PromptHandle,
        sampling_params: Option<SamplingParams>,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(), String> {
        self.scheduler
            .submit_work(work, prompt, sampling_params, sink_id)
            .map(|_| ())
    }

    /// Submits a prompt ticket to the scheduler for execution, constructing default InferenceWork.
    pub fn submit_work(
        &mut self,
        prompt: PromptHandle,
        sampling_params: SamplingParams,
        priority: u8,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<u64, String> {
        self.step_counter += 1;
        let seq_id = self.step_counter;
        let plat_priority = match priority {
            0 => aien_platform::Priority::Background,
            1 => aien_platform::Priority::Normal,
            2 => aien_platform::Priority::Interactive,
            _ => aien_platform::Priority::Realtime,
        };
        let work = aien_platform::queue::InferenceWork {
            sequence: seq_id,
            model: aien_platform::ModelHandle(1),
            kv: aien_platform::queue::KvHandle(seq_id),
            priority: plat_priority,
            deadline: None,
            branch_parent: None,
            next_token_budget: sampling_params.max_tokens as u32,
        };
        let assigned = self
            .scheduler
            .submit_work(work, prompt, Some(sampling_params), sink_id)?;
        self.try_restore(assigned);
        Ok(assigned.to_u64())
    }

    /// Submits a plain `SequenceRequest` (the scheduler assigns the id) and,
    /// with persistent KV enabled, starts the restore of its stored prefix
    /// (ADR 0003 section 6.3). A miss or any refusal leaves a plain prefill;
    /// a request is never refused because of the cache.
    pub fn submit_request(&mut self, request: aien_abi_core::SequenceRequest) -> u64 {
        let seq = self.scheduler.submit_request(request);
        self.try_restore(seq);
        seq.to_u64()
    }

    fn try_restore(&mut self, seq: aien_scheduler::SequenceId) {
        if let Some(r) = self.kv_residency.as_mut() {
            r.try_restore(&mut self.scheduler, seq);
        }
    }

    /// Turns persistent KV on: installs the park hook in the scheduler's finish
    /// transaction (S1) and hands S2 the scheduler's watermark.
    pub fn enable_kv_residency(&mut self, mut residency: KvResidency) {
        residency.set_watermark_blocks(self.scheduler.config().watermark_blocks);
        self.scheduler
            .set_finish_hook(Some(residency.finish_hook()));
        self.kv_residency = Some(residency);
    }

    /// Records that persistence was refused (fail closed): no residency is
    /// installed, and the status reports state `refused` with `reason`.
    pub fn refuse_kv_persistence(&mut self, reason: &'static str) {
        self.kv_persist_refused = Some(reason);
    }

    pub fn kv_residency(&self) -> Option<&KvResidency> {
        self.kv_residency.as_ref()
    }

    /// Persistent-KV status fields (METRICS_AND_SETTINGS), `None` when off.
    pub fn kv_residency_status(&self) -> Option<KvResidencyStatus> {
        self.kv_residency.as_ref().map(KvResidency::status)
    }

    /// The shutdown drain of the persistent-KV store (ADR 7.5), `None` when off.
    pub fn kv_drain_handle(&self) -> Option<KvDrainHandle> {
        self.kv_residency.as_ref().map(KvResidency::drain_handle)
    }

    /// Registers a completion sink for streaming output events.
    pub fn register_completion_sink(&mut self, sink: Arc<dyn CompletionSink>) -> CompletionSinkId {
        self.scheduler.register_completion_sink(sink)
    }

    /// Zero-copy subagent sequence branching with completion sink routing.
    pub fn fork_subagent(
        &mut self,
        parent_id: u64,
        child_id: u64,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(), String> {
        let ids = self
            .scheduler
            .fork_subagent_with_sink_ids(parent_id, child_id, sink_id)?;
        // The backend is not reachable here; its state is forked at the start
        // of the next step, before the child can be decoded.
        self.pending_backend_forks.push(ids);
        Ok(())
    }

    /// Executes runtime steps in a closed loop until all active requests complete or max_steps is reached.
    pub async fn run_until_complete<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
        max_steps: usize,
    ) -> Result<Vec<StepMetrics>, String> {
        let mut all_metrics = Vec::new();
        for _ in 0..max_steps {
            if self.scheduler.waiting_count() == 0
                && self.scheduler.running_count() == 0
                && self.scheduler.preempted_count() == 0
            {
                break;
            }
            if let Some(metrics) = self.step(backend).await? {
                all_metrics.push(metrics);
            }
        }
        Ok(all_metrics)
    }

    /// Advances the engine by one transactional step.
    pub async fn step<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
    ) -> Result<Option<StepMetrics>, String> {
        self.step_counter += 1;

        // PREFILL-E2E C5: backend state of sequences cancelled since the last
        // step; their KV is already freed. CancelSwarm also frees them in the
        // scheduler arena, so queued copies are stale and never re-admitted.
        self.release_pending_backend_sequences(backend)?;

        // PREFILL-E2E C4: backend state for subagent forks made since the
        // last step, before the scheduler can decode the children.
        for (parent, child) in std::mem::take(&mut self.pending_backend_forks) {
            backend.fork_sequence(parent, child)?;
        }

        // PREFILL-GATE: run the model prefill over each pending swarm root
        // prompt BEFORE any branch can be scheduled, then share the computed
        // root blocks with the branches. Branches never see root blocks that
        // have not passed the completion fence.
        let root_prefill_tokens = self.prefill_pending_swarm_roots(backend).await?;

        // Persistent KV (ADR 0003 sections 5.4 and 8): completions are polled on
        // this thread, never waited for; S3 commits finished restores here,
        // before the scheduler builds its batch, so a restored request's suffix
        // can be scheduled in this very step.
        if let Some(r) = self.kv_residency.as_mut() {
            r.poll(&mut self.scheduler, backend);
        }

        // Execute scheduler step via backend
        let step_result = self.scheduler.step(backend).await;

        // Park what this step's finish transactions forked (S1), with no lock
        // held; submit never blocks on I/O.
        if let Some(r) = self.kv_residency.as_mut() {
            r.drain_outbox();
        }
        let step_result = step_result?;

        if step_result.is_none() && root_prefill_tokens > 0 {
            return Ok(Some(StepMetrics {
                prefill_tokens_processed: root_prefill_tokens,
                active_kv_blocks: self.kv_manager.read().allocated_block_count(),
                ..StepMetrics::default()
            }));
        }

        if let Some((outputs, mut metrics)) = step_result {
            metrics.prefill_tokens_processed += root_prefill_tokens;
            // Generational SequenceId validation on step completion
            for output in &outputs {
                match output {
                    aien_abi_core::DecodeOutput::Token {
                        request_id,
                        token_id: _,
                        logprob: _,
                    } => {
                        let seq_id = SequenceId::from_u64(*request_id);
                        if let Some(record) = self.arena.get_mut(seq_id) {
                            record.generated_tokens += 1;
                        }
                    }
                    aien_abi_core::DecodeOutput::Finished {
                        request_id,
                        reason: _,
                        total_tokens: _,
                    } => {
                        let seq_id = SequenceId::from_u64(*request_id);
                        if let Some(record) = self.arena.get_mut(seq_id) {
                            record.state = SequenceState::Completed;
                        }
                        self.arena.free(seq_id);
                        let reclaimed_root = {
                            let mut kv = self.kv_manager.write();
                            self.swarm_manager.note_sequence_finished(
                                seq_id,
                                &mut self.arena,
                                &mut kv,
                                &mut self.world_store,
                            )
                        };
                        // PREFILL-E2E C5 (bullet 12): drop the backend's
                        // per-sequence state. Safe order: this runs after
                        // execute_step returned, so the backend no longer
                        // reads this sequence's KV (the scheduler already
                        // freed it), and the KV lock is released first. The
                        // root goes last, once its final branch finished:
                        // branches hold their own copy of the root's tokens
                        // (fork_sequence) and never read the root's state.
                        backend.release_sequence(*request_id)?;
                        if let Some(root) = reclaimed_root {
                            backend.release_sequence(root.as_u64())?;
                        }
                    }
                }
            }
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    /// PREFILL-E2E C5: releases the backend per-sequence state of every
    /// sequence reclaimed without a backend in reach (swarm cancel). Runs at
    /// the start of each `step`; callers that stop stepping after a cancel
    /// call it directly. Returns the number of ids released.
    pub fn release_pending_backend_sequences<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
    ) -> Result<usize, String> {
        let ids = std::mem::take(&mut self.pending_backend_releases);
        // A cancelled sequence must not be forked from or into afterwards.
        self.pending_backend_forks
            .retain(|(parent, child)| !ids.contains(parent) && !ids.contains(child));
        for &id in &ids {
            backend.release_sequence(id)?;
        }
        Ok(ids.len())
    }

    /// Prefills every pending swarm root through the scheduler's completion
    /// fence and, once the root is PrefillReady, shares its blocks with the
    /// branches. Returns the number of root prompt tokens prefilled.
    async fn prefill_pending_swarm_roots<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
    ) -> Result<usize, String> {
        let mut prefilled = 0usize;
        for (swarm_id, root_kv_id, prompt) in self.swarm_manager.pending_root_prefills() {
            if let Some(m) = self
                .scheduler
                .prefill_detached(backend, root_kv_id, &prompt)
                .await?
            {
                prefilled += m.prefill_tokens_processed;
            }
            {
                let mut kv = self.kv_manager.write();
                self.swarm_manager.fork_branches_from_ready_root(
                    swarm_id,
                    &mut self.arena,
                    &mut kv,
                )?;
            }
            // PREFILL-E2E C4 (bullets 6-8): fork the backend's per-sequence
            // state (tokens, position, last token) for each branch right after
            // the physical KV fork and before any branch can be scheduled for
            // decode. The KV lock is released first: the backend reads the
            // same shared KV manager to check the parent is PrefillReady.
            // PREFILL-E2E C6 (bullet 11): the fork hook also hands each
            // branch its own sampling params (the same ones its scheduler
            // request carries), so branches decode with temperature and a
            // per-branch seed instead of all taking the argmax.
            let (branches, sampling): (Vec<u64>, SamplingParams) = self
                .swarm_manager
                .get_swarm(swarm_id)
                .map(|s| {
                    (
                        s.branch_sequences.iter().map(|b| b.as_u64()).collect(),
                        branch_sampling_params(&s.config),
                    )
                })
                .unwrap_or_else(|| (Vec::new(), SamplingParams::default()));
            for child in branches {
                backend.fork_sequence_with_sampling(root_kv_id, child, &sampling)?;
            }
        }
        Ok(prefilled)
    }

    /// Launches a swarm of agents sharing an immutable root World and physical KV blocks.
    /// Branch KV is shared from the root only after the root prompt prefill
    /// completes (next `step`); until then branches cannot be decoded.
    pub fn launch_swarm(
        &mut self,
        config: SwarmConfig,
        prompt_tokens: &[u32],
    ) -> Result<u64, String> {
        let mut kv = self.kv_manager.write();
        let swarm_id = self.swarm_manager.launch_swarm(
            config,
            &mut self.arena,
            &mut kv,
            &mut self.world_store,
            prompt_tokens,
            self.step_counter,
        )?;

        // Enqueue branch requests into scheduler
        if let Some(swarm) = self.swarm_manager.get_swarm(swarm_id) {
            for &child_seq in &swarm.branch_sequences {
                let req = aien_abi_core::SequenceRequest {
                    request_id: child_seq.as_u64(),
                    prompt_tokens: prompt_tokens.to_vec(),
                    sampling_params: branch_sampling_params(&swarm.config),
                    arrival_time_ns: self.step_counter,
                    priority: swarm.config.priority,
                };
                self.scheduler.submit_request(req);
            }
        }

        Ok(swarm_id)
    }

    /// Handles a typed operator command with idempotency verification.
    pub fn handle_control_command(&mut self, envelope: ControlEnvelope) -> ControlResponse {
        if self
            .controller
            .is_operation_processed(envelope.operation_id)
        {
            return ControlResponse::Error(format!(
                "Operation {} already processed",
                envelope.operation_id
            ));
        }

        let resp = match envelope.command {
            ControlCommand::LaunchSwarm(req) => {
                let config = SwarmConfig {
                    model_handle: req.model_handle,
                    branch_count: req.branch_count,
                    max_active_sequences: req.max_active_sequences,
                    max_tokens_per_branch: req.max_tokens_per_branch,
                    root_world_id: req.root_world_id,
                    priority: req.priority,
                };
                match self.launch_swarm(config, &req.prompt_tokens) {
                    Ok(swarm_id) => {
                        record_processed(&mut self.controller, envelope.operation_id);
                        ControlResponse::SwarmAccepted {
                            swarm_id,
                            operation_id: envelope.operation_id,
                        }
                    }
                    Err(e) => ControlResponse::Error(e),
                }
            }
            ControlCommand::CancelSwarm(swarm_id) => {
                let mut kv = self.kv_manager.write();
                match self.swarm_manager.cancel_swarm(
                    swarm_id,
                    &mut self.arena,
                    &mut kv,
                    &mut self.world_store,
                ) {
                    Ok(released) => {
                        // KV, arena slots and branch worlds are reclaimed;
                        // backend state follows on the next step (C5).
                        // The scheduler keeps its own arena, with the same ids
                        // (submit_request inserts them explicitly). Free them
                        // there too, so queued or running copies turn stale
                        // and are never admitted or decoded again.
                        for &id in &released {
                            if let Ok(sid) = aien_scheduler::sequence::SequenceId::from_u64(id) {
                                self.scheduler.arena_mut().free_sequence(sid);
                            }
                        }
                        self.pending_backend_releases.extend(released);
                        record_processed(&mut self.controller, envelope.operation_id);
                        ControlResponse::SwarmCancelled { swarm_id }
                    }
                    Err(e) => ControlResponse::Error(e),
                }
            }
            ControlCommand::GetRuntimeStatus => ControlResponse::Status(self.status_report()),
            ControlCommand::InspectSwarm(swarm_id) => match self.swarm_status_report(swarm_id) {
                Some(report) => ControlResponse::Status(report),
                None => ControlResponse::Error(format!("Swarm {} not found", swarm_id)),
            },
            ControlCommand::StreamTurn { .. } => ControlResponse::Error(
                "StreamTurn is handled on the socket connection, not as a one-shot command".into(),
            ),
            ControlCommand::RunComposeTask { .. }
            | ControlCommand::ComposeNote { .. }
            | ControlCommand::ComposeRecall { .. }
            | ControlCommand::RecoverComposeHome
            | ControlCommand::ComposeEffectIntent { .. }
            | ControlCommand::ComposeEffectAck { .. }
            | ControlCommand::ComposeReconcile { .. }
            | ControlCommand::ComposeControl { .. }
            | ControlCommand::ComposeAuthorize { .. }
            | ControlCommand::ComposeApprovedProposal { .. }
            | ControlCommand::AllenStatus
            | ControlCommand::AllenProfileShow
            | ControlCommand::AllenProfileSet { .. }
            | ControlCommand::AllenProfileHistory
            | ControlCommand::AllenProfileRevert { .. }
            | ControlCommand::AllenMemoryPut { .. }
            | ControlCommand::AllenMemoryRecall { .. }
            | ControlCommand::AllenMemoryInspect { .. }
            | ControlCommand::AllenMemoryCorrect { .. }
            | ControlCommand::AllenMemoryForget { .. }
            | ControlCommand::AllenMemoryExport { .. }
            | ControlCommand::AllenGoalsList { .. }
            | ControlCommand::AllenGoalAdd { .. }
            | ControlCommand::AllenGoalClose { .. }
            | ControlCommand::AllenProfileReset { .. } => ControlResponse::Error(
                "compose commands are handled on the socket connection, not as one-shot commands"
                    .into(),
            ),
            ControlCommand::Shutdown => {
                record_processed(&mut self.controller, envelope.operation_id);
                ControlResponse::ShutdownAck
            }
        };

        resp
    }

    /// Status scoped to one swarm. Sequence, swarm, and world counts cover only
    /// this swarm; KV block and page fields describe the shared runtime pool.
    pub fn swarm_status_report(&self, swarm_id: u64) -> Option<RuntimeStatusReport> {
        let swarm = self.swarm_manager.get_swarm(swarm_id)?;
        let running = swarm.state == SwarmState::Running;
        let active_sequences = if running {
            swarm
                .branch_sequences
                .len()
                .saturating_sub(swarm.finished_branches.len())
        } else {
            0
        };
        let active_worlds = swarm
            .branch_worlds
            .iter()
            .filter(|&&w| self.world_store.get_world(w).is_some())
            .count();
        Some(RuntimeStatusReport {
            active_sequences,
            active_swarms: usize::from(running),
            active_worlds,
            ..self.status_report()
        })
    }

    pub fn status_report(&self) -> RuntimeStatusReport {
        let mut report = self.pool_status_report();
        // After the pool guard is released: the residency status takes the KV
        // read lock itself (a nested read could wait behind a queued writer).
        report.kv_persist = match (&self.kv_residency, self.kv_persist_refused) {
            (Some(r), _) => Some(Box::new(r.persist_report())),
            (None, Some(reason)) => Some(Box::new(crate::control::KvPersistReport {
                kv_store_state: "refused".to_string(),
                kv_store_refused_reason: reason.to_string(),
                kv_store_key_refused_total: aien_kv_store::seal::key_refusals_total(),
                ..Default::default()
            })),
            (None, None) => None,
        };
        report
    }

    fn pool_status_report(&self) -> RuntimeStatusReport {
        let kv = self.kv_manager.read();
        let metrics = kv.metrics();
        let free = kv.free_block_count();
        let allocated = kv.allocated_block_count();
        let total = free + allocated;
        // Honest proxy until the GB10 telemetry hook lands: KV memory pressure
        // as a percentage. This is memory pressure, not SM activity.
        let pressure_pct = if total > 0 {
            (allocated as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        RuntimeStatusReport {
            active_sequences: self.arena.active_count(),
            active_swarms: self.swarm_manager.active_swarm_count(),
            active_worlds: self.world_store.active_world_count(),
            free_kv_blocks: free,
            total_kv_blocks: total,
            shared_kv_pages: metrics.shared_pages,
            cow_faults: metrics.cow_faults,
            gpu_utilization_pct: pressure_pct,
            kv_pool_allocated_bytes: kv.pool().map_or(0, |p| p.layout().total_bytes as u64),
            kv_pool_total_blocks: total,
            kv_pool_free_blocks: free,
            kv_pool_used_blocks: allocated,
            kv_pool_holder_blocks: kv.holder_block_count(),
            kv_persist: None,
        }
    }
}

/// Sampling params of every swarm branch: submitted with the branch's
/// scheduler request (`launch_swarm`) and handed to the backend by the fork
/// hook (`prefill_pending_swarm_roots`), so both sides agree.
fn branch_sampling_params(config: &SwarmConfig) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        max_tokens: config.max_tokens_per_branch,
        stop_token_ids: vec![2],
    }
}

/// Records a processed operation id. The operation has already run, so a
/// failed save does not change the reply; it is reported on stderr so the
/// loss of durability is never silent (#299).
fn record_processed(controller: &mut RuntimeController, operation_id: u128) {
    if let Err(e) = controller.mark_operation_processed(operation_id) {
        eprintln!("aien-runtime: {e}");
    }
}
