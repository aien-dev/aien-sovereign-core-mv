//! Canonical AienRuntimeSpine
//! Unifies SequenceArena, AienScheduler, AienKvManager, WorldStore, ContextComposer,
//! and SwarmManager in a single process address space.

use crate::context::ContextComposer;
use crate::control::{
    ControlCommand, ControlEnvelope, ControlResponse, RuntimeController, RuntimeStatusReport,
};
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
        Ok(assigned.to_u64())
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

        // Execute scheduler step via backend
        let step_result = self.scheduler.step(backend).await?;

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

// ---------------------------------------------------------------------------
// NEXT-PHASE-1 cut 1b: the compose bridge (omega COMPOSITION-2 via
// aien-omega-compose). One composition home per process, opened on first use
// and kept open: the Cortex journal takes an exclusive writer lock.
// ---------------------------------------------------------------------------

use crate::control::{
    ComposeNoteReport, ComposeRecallReport, ComposeRecordView, ComposeRecoverReport,
    ComposeTaskReport, ProposalAttempt,
};
use crate::cortex_mark::{self, Mark, MarkRefusal, Plan};
use aien_omega_compose::{hex, note_bytes, Compose, ComposeError, NoteKind, RootKind};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One model reply: the text and how many tokens were generated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub tokens: usize,
    /// How generation stopped: "eos" | "max_tokens" | "aborted" | "preempted"
    /// (ACCEPTANCE-v5 Q3); None when unknown.
    pub finish_reason: Option<String>,
    /// NEXT-PHASE-1 v6 R1: the generated token ids in order (None when the
    /// proposer does not expose them). Recorded only; nothing reads them.
    pub token_ids: Option<Vec<u32>>,
    /// NEXT-PHASE-1 v6 R1: how many prompt token ids were submitted.
    pub prompt_tokens: Option<usize>,
    /// NEXT-PHASE-1 v6 R1: `token_ids_sha256` of the submitted prompt ids.
    pub prompt_ids_sha256: Option<String>,
    /// How the backend chose the generated tokens (sc#294), as the scheduler
    /// reported it when the sequence finished. None when the proposer or the
    /// backend does not observe it: an absent claim, not greedy.
    pub decoding: Option<aien_abi_core::DecodeObservation>,
    /// The tensor backend and its op counters when the call finished (sc#337).
    /// None when the proposer or the backend does not report them: no claim.
    pub ops: Option<aien_abi_core::OpEvidence>,
}

/// sha256 (hex) of token ids, each as 4 little-endian bytes (NEXT-PHASE-1 v6
/// R1): the daemon and the CPU reference driver hash prompts the same way.
pub fn token_ids_sha256(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    hex(&h.finalize())
}

/// The "model" Skill's work: prompt text and a wall limit in, one reply
/// out. The server installs the real inference path (the one `StreamTurn`
/// uses); tests install a deterministic proposer.
pub type ComposeProposer =
    Arc<dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync>;

/// The one Skill name; its procedure digest is sha256 of this name.
pub const COMPOSE_MODEL_SKILL: &str = "aien.model.propose-file-change";

/// `ComposeTaskReport::proposer` of a run driven by the proposer hook
/// (crate::approved): the reply came from an approved proposal, not a model.
pub const APPROVED_PROPOSER_LABEL: &str = "hook:approved-proposal";

/// Composition home: `$AIEN_COMPOSE_DIR`, else `$AIEN_RUNTIME_STATE_DIR/compose`,
/// else `$XDG_STATE_HOME/aien-runtime/compose`, else
/// `$HOME/.local/state/aien-runtime/compose`. Never /tmp.
pub fn compose_dir_from_env() -> Result<PathBuf, String> {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    if let Some(d) = get("AIEN_COMPOSE_DIR") {
        return Ok(PathBuf::from(d.trim()));
    }
    if let Some(d) = get("AIEN_RUNTIME_STATE_DIR") {
        return Ok(PathBuf::from(d.trim()).join("compose"));
    }
    if let Some(d) = get("XDG_STATE_HOME") {
        return Ok(PathBuf::from(d.trim()).join("aien-runtime/compose"));
    }
    if let Some(h) = get("HOME") {
        return Ok(PathBuf::from(h.trim()).join(".local/state/aien-runtime/compose"));
    }
    Err(
        "no AIEN_COMPOSE_DIR, AIEN_RUNTIME_STATE_DIR, XDG_STATE_HOME or HOME for the compose home"
            .into(),
    )
}

/// The provisioned machine root: 32 random bytes kept beside the home
/// (`<home>.machine-root`, mode 0600), created once. STOPGAP until AIENOS
/// machine provisioning owns the AienMachineId root.
fn machine_root(dir: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    let mut name = dir.as_os_str().to_owned();
    name.push(".machine-root");
    let path = PathBuf::from(name);
    if let Ok(b) = std::fs::read(&path) {
        if b.len() == 32 {
            return Ok(b);
        }
        return Err(format!(
            "{} holds {} bytes, expected 32",
            path.display(),
            b.len()
        ));
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("create {}: {e}", p.display()))?;
    }
    let mut root = vec![0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut root))
        .map_err(|e| format!("read /dev/urandom: {e}"))?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    std::io::Write::write_all(&mut f, &root)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    f.sync_all()
        .map_err(|e| format!("sync {}: {e}", path.display()))?;
    Ok(root)
}

fn proposal_handle(text: &str) -> u64 {
    let d = Sha256::digest(text.as_bytes());
    let h = u64::from_le_bytes(d[..8].try_into().expect("8 bytes"));
    if h == 0 {
        1
    } else {
        h
    }
}

/// ALLEN identity gate (aien-allen, ADR 0035), run once per compose-home open.
/// Not engaged (AIEN_ALLEN_SUBJECT unset): one log line, nothing else changes.
/// Engaged: any refusal is FATAL (message, nonzero exit), never a fallback.
/// Returns the resolved identity when engaged, so the compose home can keep it
/// (persona profile, arch#159); `None` when not engaged.
fn allen_gate(
    dir: &Path,
    compose: &mut Compose,
    machine_id: &[u8; 32],
) -> Option<aien_allen::Resolved> {
    use aien_allen::{Context, Gate};
    static NOT_ENGAGED_ONCE: std::sync::Once = std::sync::Once::new();
    let lineage = compose.record(1).ok().map(|r| r.digest);
    let ctx = Context {
        machine_id: *machine_id,
        lineage,
    };
    // The operator adoption variable is honoured once per process: after this
    // process wrote the pin, later opens (recover, re-open) see it as spent.
    static ADOPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let get = |k: &str| {
        if k == aien_allen::ENV_ADOPT && ADOPTED.load(std::sync::atomic::Ordering::SeqCst) {
            return None;
        }
        std::env::var(k).ok()
    };
    match aien_allen::gate(dir, &ctx, &get) {
        Gate::NotEngaged => {
            NOT_ENGAGED_ONCE.call_once(|| println!("{}", aien_allen::NOT_ENGAGED_LINE));
            None
        }
        Gate::Engaged(r) => {
            if r.adopted {
                ADOPTED.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let adopted = if r.adopted {
                " (ADOPTED: pin written once, operator-approved)"
            } else {
                ""
            };
            println!(
                "ALLEN: engaged agent={} head_sequence={} chain_verified={}{adopted}",
                aien_allen::hex(&r.agent),
                r.head_seq,
                r.chain_verified
            );
            Some(r)
        }
        Gate::Refused(why) => {
            let msg = format!("FATAL ALLEN refused: {why}");
            tracing::error!("{msg}");
            eprintln!("{msg}");
            std::process::exit(aien_allen::EXIT_REFUSED);
        }
    }
}

/// A named refusal for an rxc_host open error: the operator learns what is
/// wrong and that `RecoverComposeHome` is the remedy (never a silent repair).
fn refusal(dir: &Path, e: &ComposeError) -> String {
    let why = match e {
        ComposeError::Code { code: -3, .. } => {
            "Cortex journal tail torn (E_TORN, CX_ERR_TORN); nothing was cut. Run RecoverComposeHome (aien compose recover)"
                .to_string()
        }
        ComposeError::Code { code: -11, detail } => format!(
            "Cortex journal behind its J-Space anchor (E_REPLAY, RX_ERR_REPLAY {detail}). Run RecoverComposeHome (aien compose recover)"
        ),
        other => other.to_string(),
    };
    format!("compose home {} refused: {why}", dir.display())
}

/// A refusal by the Cortex record mark (ACCEPTANCE-v3 2.2), in the same form.
fn mark_refusal(dir: &Path, r: &MarkRefusal) -> String {
    format!("compose home {} refused: {r}", dir.display())
}

/// One proposed file change: a path relative to the workspace and the
/// complete new content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileProposal {
    pub path: String,
    pub content: String,
}

fn bare(s: &str) -> &str {
    s.trim()
        .trim_matches(|c| matches!(c, '`' | '"' | '\'' | '*'))
        .trim()
}

/// A path the workspace may hold: relative, at most 255 bytes, letters,
/// digits and `._-/`, no empty, `.` or `..` component, not absolute, not `~`.
fn check_relative_path(p: &str) -> Result<(), String> {
    if p.is_empty() {
        return Err("empty path on the filename line".into());
    }
    if p.starts_with('/') || p.starts_with('~') || p.split('/').any(|c| c == "..") {
        return Err(format!("path {p:?} is outside the workspace"));
    }
    let ok = p.len() <= 255
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
        && p.split('/').all(|c| !c.is_empty() && c != ".");
    if ok {
        Ok(())
    } else {
        Err(format!("path {p:?} is not a plain relative path"))
    }
}

/// A code-fence line: its backtick run length and whether an info string
/// (such as `bash`) follows the run. Not a fence if fewer than three
/// backticks start the line or the info string itself holds a backtick.
fn fence_line(l: &str) -> Option<(usize, bool)> {
    let t = l.trim_start();
    let n = t.bytes().take_while(|&b| b == b'`').count();
    if n < 3 {
        return None;
    }
    let info = t[n..].trim();
    if info.contains('`') {
        return None;
    }
    Some((n, !info.is_empty()))
}

/// Index (into `body`, which starts after the opening fence of backtick
/// length `open_len`) of the line that closes the outer fence, or None.
/// CommonMark: a closing fence is bare (no info string) and at least as long
/// as the opener, so shorter inner fences are plain content. Inner fences of
/// the same length that carry an info string (` ```bash `) open a nested block
/// that its own bare fence must close first. If that scan finds no balanced
/// close (unbalanced reply), the LAST bare fence of sufficient length is
/// taken, so content is never cut at an inner fence. A bare fence at depth 0
/// that still has fences after it is a bare inner opener, so the outer fence
/// is closed by the last fence of the reply. Limit: prose that itself holds a
/// fence AFTER the real close is read as part of the file. With no bare fence at
/// all there is no close: the caller keeps the whole body.
fn outer_fence_close(body: &[&str], open_len: usize) -> Option<usize> {
    let fences: Vec<(usize, bool)> = body
        .iter()
        .enumerate()
        .filter_map(|(i, l)| fence_line(l).filter(|f| f.0 >= open_len).map(|f| (i, f.1)))
        .collect();
    let mut depth = 0usize;
    let mut last_bare = None;
    for (k, &(i, info)) in fences.iter().enumerate() {
        if info {
            depth += 1;
        } else if depth > 0 {
            depth -= 1;
            last_bare = Some(i);
        } else if k + 1 == fences.len() {
            return Some(i);
        } else {
            // A bare fence with more fences after it opens a bare inner block;
            // only the final fence can then be the outer close.
            depth += 1;
        }
    }
    last_bare
}

/// The model's answer in the fixed proposal template: the first nonempty
/// line is `filename: <relative path>` (case-insensitive key; quotes,
/// backticks and asterisks around it or the path are ignored), everything
/// after it is the complete content (blank lines around it and one
/// surrounding code fence are dropped; it ends with one newline).
/// Err names why the reply is refused. Deterministic: the Skill, the AEGIS
/// contract, the authorization and the write all use this one reading.
pub fn check_file_proposal(text: &str) -> Result<FileProposal, String> {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|l| !l.trim().is_empty())
        .ok_or("empty reply")?;
    let first = bare(lines[at]);
    const KEY: &str = "filename:";
    if first.len() < KEY.len() || !first[..KEY.len()].eq_ignore_ascii_case(KEY) {
        return Err("no filename line: the first line must be 'filename: <relative path>'".into());
    }
    let path = bare(&first[KEY.len()..]).to_string();
    check_relative_path(&path)?;
    let mut body: Vec<&str> = lines[at + 1..].to_vec();
    while body.first().is_some_and(|l| l.trim().is_empty()) {
        body.remove(0);
    }
    if body
        .first()
        .is_some_and(|l| l.trim_start().starts_with("```"))
    {
        let open_len = body[0]
            .trim_start()
            .bytes()
            .take_while(|&b| b == b'`')
            .count();
        body.remove(0);
        if let Some(end) = outer_fence_close(&body, open_len) {
            body.truncate(end);
        }
    }
    while body.last().is_some_and(|l| l.trim().is_empty()) {
        body.pop();
    }
    if body.iter().all(|l| l.trim().is_empty()) {
        return Err("empty content after the filename line".into());
    }
    let mut content = body.join("\n");
    content.push('\n');
    Ok(FileProposal { path, content })
}

/// `check_file_proposal` without the reason.
pub fn parse_file_proposal(text: &str) -> Option<FileProposal> {
    check_file_proposal(text).ok()
}

/// At most this many proposals per task (ACCEPTANCE-v2 Section 3b).
pub const COMPOSE_MAX_ATTEMPTS: u32 = 3;
/// Wall budget for all attempts of one task: rx_compose_run's 30 s
/// quiescence wait (not raised) less a 1 s margin (ACCEPTANCE-v3 Section 3b).
pub const COMPOSE_SKILL_BUDGET: std::time::Duration = std::time::Duration::from_secs(29);
/// Wall budget of a full-document task (one that creates a new file): 120 s.
pub const COMPOSE_DOC_BUDGET: std::time::Duration = std::time::Duration::from_secs(120);
/// Env: wall budget of a small edit, in ms. Unset = `COMPOSE_SKILL_BUDGET` (29 s).
pub const COMPOSE_EDIT_BUDGET_ENV: &str = "AIEN_COMPOSE_EDIT_BUDGET_MS";
/// Env: wall budget of a full-document task, in ms. Unset = `COMPOSE_DOC_BUDGET` (120 s).
pub const COMPOSE_DOC_BUDGET_ENV: &str = "AIEN_COMPOSE_DOC_BUDGET_MS";
/// Env: output token cap of a full-document task. Unset = `COMPOSE_DOC_MAX_TOKENS_DEFAULT`.
/// (Small edits keep `AIEN_COMPOSE_MAX_TOKENS`, read by the daemon, default 48.)
pub const COMPOSE_DOC_MAX_TOKENS_ENV: &str = "AIEN_COMPOSE_DOC_MAX_TOKENS";
/// The one-budget setting this replaced. Setting it is refused, never ignored.
pub const COMPOSE_BUDGET_ENV_RETIRED: &str = "AIEN_COMPOSE_BUDGET_MS";
/// Default output token cap of a full-document task.
pub const COMPOSE_DOC_MAX_TOKENS_DEFAULT: usize = 1024;
/// Accepted range of `AIEN_COMPOSE_DOC_MAX_TOKENS`.
pub const COMPOSE_DOC_MAX_TOKENS_RANGE: std::ops::RangeInclusive<usize> = 16..=4096;
/// Accepted range of both budget settings: omega waits budget + 1 000 ms and accepts at most 600 000 ms.
pub const COMPOSE_BUDGET_MS_RANGE: std::ops::RangeInclusive<u64> = 1_000..=599_000;
/// Margin between the skill budget and omega's settle wait.
pub const COMPOSE_WAIT_MARGIN: std::time::Duration = std::time::Duration::from_millis(1_000);

/// What a compose task is, for limits: a small edit of an existing file, or
/// the creation of a whole new document. Decided by `classify_target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    Edit,
    Document,
}

/// What the goal's named target is, decided by whether the path EXISTS (not by
/// whether an edit block can be built from it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetClass {
    /// No existing file is named: the task creates a new document.
    New,
    /// An existing small UTF-8 file inside the workspace: path and content.
    Edit(String, String),
    /// An existing path that no edit block can be built for (too large,
    /// non-UTF-8, unreadable, or resolving outside the workspace). Never a
    /// new document, never overwritten.
    Refused(String),
}

impl TargetClass {
    pub fn kind(&self) -> ProposalKind {
        match self {
            Self::Edit(..) => ProposalKind::Edit,
            _ => ProposalKind::Document,
        }
    }
}

/// The two wall budgets in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComposeBudgets {
    pub edit: std::time::Duration,
    pub doc: std::time::Duration,
}

impl Default for ComposeBudgets {
    fn default() -> Self {
        Self {
            edit: COMPOSE_SKILL_BUDGET,
            doc: COMPOSE_DOC_BUDGET,
        }
    }
}

impl ComposeBudgets {
    pub fn for_kind(&self, kind: ProposalKind) -> std::time::Duration {
        match kind {
            ProposalKind::Edit => self.edit,
            ProposalKind::Document => self.doc,
        }
    }
}

/// Parse one budget setting named `name`. `None` (unset) gives `default`; an
/// empty, non-numeric or out-of-range value is refused, never replaced by the default.
pub fn parse_compose_budget(
    name: &str,
    default: std::time::Duration,
    v: Option<&str>,
) -> Result<std::time::Duration, String> {
    let Some(raw) = v else {
        return Ok(default);
    };
    let ms: u64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("{name}={raw:?} is not a whole number of milliseconds"))?;
    if !COMPOSE_BUDGET_MS_RANGE.contains(&ms) {
        return Err(format!(
            "{name}={ms} is outside {}..={} ms",
            COMPOSE_BUDGET_MS_RANGE.start(),
            COMPOSE_BUDGET_MS_RANGE.end()
        ));
    }
    Ok(std::time::Duration::from_millis(ms))
}

/// Parse `AIEN_COMPOSE_DOC_MAX_TOKENS` (same refusal rule as the budgets).
pub fn parse_compose_doc_max_tokens(v: Option<&str>) -> Result<usize, String> {
    let Some(raw) = v else {
        return Ok(COMPOSE_DOC_MAX_TOKENS_DEFAULT);
    };
    let n: usize = raw.trim().parse().map_err(|_| {
        format!("{COMPOSE_DOC_MAX_TOKENS_ENV}={raw:?} is not a whole number of tokens")
    })?;
    if !COMPOSE_DOC_MAX_TOKENS_RANGE.contains(&n) {
        return Err(format!(
            "{COMPOSE_DOC_MAX_TOKENS_ENV}={n} is outside {}..={} tokens",
            COMPOSE_DOC_MAX_TOKENS_RANGE.start(),
            COMPOSE_DOC_MAX_TOKENS_RANGE.end()
        ));
    }
    Ok(n)
}

fn env_opt(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name} unreadable: {e}")),
    }
}

/// Env switch for the desk MAC on `ComposeAuthorize` (#297, #328). Required
/// by default: unset or `1` = required. `0` is the dev-only opt-out, accepted
/// only in a dev run (`aien_inference_abi::strict::dev_fallback_active`); a
/// strict run, which is what release qualification runs, refuses it. Anything
/// else stops the daemon (a typo must not leave the requirement silently off).
pub const AUTHORIZE_DESK_ENV: &str = "AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK";

/// The rule behind [`authorize_requires_desk_from_env`], for a switch value
/// and whether this is a dev run.
pub fn authorize_desk_setting(value: Option<&str>, dev: bool) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") if dev => Ok(false),
        Some("0") => Err(format!(
            "{AUTHORIZE_DESK_ENV}=0 is a dev-only opt-out and is prohibited here: this is a strict run \
             (no {}=1), as release qualification is. Create the desk key with \
             `aien compose desk-key --create 1` and unset {AUTHORIZE_DESK_ENV}",
            aien_inference_abi::strict::DEV_FALLBACK_ENV
        )),
        Some(o) => Err(format!(
            "{AUTHORIZE_DESK_ENV} must be 1 (required, the default) or 0 (dev-only opt-out), got {o:?}"
        )),
    }
}

pub fn authorize_requires_desk_from_env() -> Result<bool, String> {
    authorize_desk_setting(
        env_opt(AUTHORIZE_DESK_ENV)?.as_deref(),
        aien_inference_abi::strict::dev_fallback_active(),
    )
}

/// Startup check (#328): a bridge that requires the desk must find a loadable
/// desk key in its compose home, or the daemon refuses to start. Never creates
/// a key: minting one is the operator's `aien compose desk-key --create`.
pub fn check_desk_key_at_start(compose_dir: &Path) -> Result<(), String> {
    let path = crate::approved_auth::desk_key_path(compose_dir);
    crate::approved_auth::DeskKey::load(&path).map(|_| ()).map_err(|e| {
        format!(
            "the approval desk is required ({AUTHORIZE_DESK_ENV} unset or 1) but its key is not usable: {e}. \
             Create it with `aien compose desk-key --create 1` (AIEN_COMPOSE_DIR={}); the daemon refuses to \
             start without it",
            compose_dir.display()
        )
    })
}

/// The daemon's startup rule from the environment (#328): the setting,
/// after the desk key check when the desk is required.
pub fn authorize_desk_at_start_from_env() -> Result<bool, String> {
    let required = authorize_requires_desk_from_env()?;
    if required {
        if let Ok(dir) = compose_dir_from_env() {
            check_desk_key_at_start(&dir)?;
        }
    }
    Ok(required)
}

/// The startup line that states the authorize rule this daemon enforces.
pub fn authorize_mac_line(required: bool) -> &'static str {
    if required {
        "Authorize MAC: on (aien compose authorize requires the approval desk key MAC)"
    } else {
        "Authorize MAC: OFF, DEV OPT-OUT (AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=0 in a dev run): \
         authorize is authenticated by the OS user only; not valid for release qualification"
    }
}

/// Both budgets from the environment. Refuses the retired single-budget
/// setting (a stale `AIEN_COMPOSE_BUDGET_MS` must not be silently ignored).
pub fn compose_budgets_from_env() -> Result<ComposeBudgets, String> {
    if std::env::var_os(COMPOSE_BUDGET_ENV_RETIRED).is_some() {
        return Err(format!(
            "{COMPOSE_BUDGET_ENV_RETIRED} is retired; set {COMPOSE_EDIT_BUDGET_ENV} and {COMPOSE_DOC_BUDGET_ENV}"
        ));
    }
    Ok(ComposeBudgets {
        edit: parse_compose_budget(
            COMPOSE_EDIT_BUDGET_ENV,
            COMPOSE_SKILL_BUDGET,
            env_opt(COMPOSE_EDIT_BUDGET_ENV)?.as_deref(),
        )?,
        doc: parse_compose_budget(
            COMPOSE_DOC_BUDGET_ENV,
            COMPOSE_DOC_BUDGET,
            env_opt(COMPOSE_DOC_BUDGET_ENV)?.as_deref(),
        )?,
    })
}

/// The document token cap from the environment.
pub fn compose_doc_max_tokens_from_env() -> Result<usize, String> {
    parse_compose_doc_max_tokens(env_opt(COMPOSE_DOC_MAX_TOKENS_ENV)?.as_deref())
}

/// Stable prefix of `wall_clock_reason`: `propose_task_with_retries` classes an
/// attempt as "timeout" by it, and the token-limit cut (`length_cut_refusal`) never has it.
pub const WALL_CLOCK_REASON_PREFIX: &str = "model proposal exceeded ";

/// Reason text of a wall-clock timeout.
pub fn wall_clock_reason(limit_ms: u128) -> String {
    format!("{WALL_CLOCK_REASON_PREFIX}{limit_ms} ms")
}

/// Time one full attempt needs, from the v3 measurement: 173-token retry
/// prompt prefill (2 839 + 896 ms) + 47 decode steps x 166.6 ms + 60 ms
/// = 11 625 ms, rounded up (ACCEPTANCE-v3 Section 3b). Attempt k > 1 starts
/// only if at least this much budget is left.
pub const COMPOSE_ATTEMPT_BUDGET: std::time::Duration = std::time::Duration::from_millis(12_000);
/// Same rule for a full document. DIAGNOSTIC-LONG-1 (sovereign-core branch
/// 031756/oq3-long-diag) measured replies of 313..649 tokens finishing by
/// themselves in 28.6..61.9 s (~88 ms/token plus prefill), so a retry that
/// has to write a whole document needs about the shortest measured one:
/// 28.6 s, rounded up to 30 s. With the 120 s budget, a first attempt up to the
/// longest measured (61.9 s) leaves 58 s and still allows a retry; one that ran
/// to the token cap (1024 tokens, ~90 s) leaves ~30 s and may not. Capped at
/// half the budget so a small custom doc budget keeps the retry rule meaningful.
pub const COMPOSE_DOC_ATTEMPT_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// The per-attempt budget for `kind` under wall budget `budget`.
pub fn compose_attempt_budget(
    kind: ProposalKind,
    budget: std::time::Duration,
) -> std::time::Duration {
    match kind {
        ProposalKind::Edit => COMPOSE_ATTEMPT_BUDGET,
        ProposalKind::Document => COMPOSE_DOC_ATTEMPT_BUDGET.min(budget / 2),
    }
}

/// Assistant-response prefix of the production RunComposeTask template: the
/// assistant turn starts with it and the model generates the path and the
/// content after it (ACCEPTANCE-v4 Section 2(1)). The reply the parser reads
/// is this prefix followed by the generated text.
///
/// NEXT-PHASE-1 v5 engine cut: no trailing space. A separate space token after
/// the colon was measured harmful; the model's own tokenization puts the space
/// in front of the path.
pub const COMPOSE_ASSISTANT_PREFIX: &str = "filename:";

/// The compose assistant-response prefix for a model's chat template. Zephyr
/// and Llama 3 keep `COMPOSE_ASSISTANT_PREFIX` byte for byte. ChatML
/// (SmolLM2-Instruct) uses the same text: the proposal template asks for a
/// `filename: <relative path>` first line, and in SmolLM2's byte-level BPE
/// `filename:` ends on the colon token, so the path keeps its own leading
/// space token as in Llama 3. `proposal_prompt` is the same for every template.
pub fn compose_assistant_prefix(template: &aien_inference_abi::ChatTemplate) -> &'static str {
    use aien_inference_abi::ChatTemplate;
    match template {
        ChatTemplate::Zephyr | ChatTemplate::Llama3 => COMPOSE_ASSISTANT_PREFIX,
        // A plain model never reaches generation (chat render is refused first).
        ChatTemplate::None => COMPOSE_ASSISTANT_PREFIX,
        ChatTemplate::ChatMl { .. } | ChatTemplate::ChatMlQwen3 => "filename:",
    }
}

/// The fixed proposal template of the production RunComposeTask path.
pub fn proposal_prompt(goal: &str, workspace: &str) -> String {
    format!(
        "Goal: {goal}\nAuthorized workspace: {workspace}\n\
         Propose exactly one file change inside the workspace.\n\
         Answer in exactly this format and nothing else:\n\
         filename: <relative path>\n\
         <the complete new file content>"
    )
}

/// Largest existing file whose content the edit-mode block carries
/// (NEXT-PHASE-1 v6 T5); a larger file gets no block.
pub const COMPOSE_EDIT_MAX_BYTES: u64 = 8192;

/// NEXT-PHASE-1 v6 edit mode: the goal's DESTINATION (see `destination.rs`,
/// issue #288; paths the goal only reads are ignored) when it is a plain
/// relative path naming an existing regular UTF-8 file of at most
/// `COMPOSE_EDIT_MAX_BYTES` inside the canonical workspace `ws`. Returns that
/// path and the file's content. None when the destination is missing, unsafe
/// or ambiguous, so a new-file goal keeps the v5 prompt byte for byte.
pub fn existing_target(goal: &str, ws: &Path) -> Option<(String, String)> {
    match classify_target(goal, ws) {
        TargetClass::Edit(p, c) => Some((p, c)),
        _ => None,
    }
}

/// Decide the task kind from the goal's DESTINATION only (issue #288): the
/// path the goal asks to create or change, found by
/// `crate::destination::named_destination`; paths it merely reads ("about
/// README.md") are never classified. The destination is then judged by the disk:
/// a small UTF-8 file inside the workspace is `Edit`; a missing path inside the
/// workspace is `New`; everything else is `Refused` with the reason: an
/// absolute, `~`, `..` or non-plain path, a symlink or parent resolving outside
/// the workspace, a directory, a file over `COMPOSE_EDIT_MAX_BYTES`, non-UTF-8
/// or unreadable content, and an ambiguous destination (several candidates).
/// A goal naming no destination is `New` (the model names the file).
pub fn classify_target(goal: &str, ws: &Path) -> TargetClass {
    classify_destination(goal, ws).0
}

/// `classify_target` plus the destination path it judged (None when the goal
/// names none). This is the single decision of a task: the prompt, the budget,
/// the merge target and the check of the reply's `filename:` all use it.
pub fn classify_destination(goal: &str, ws: &Path) -> (TargetClass, Option<String>) {
    let Ok(ws) = std::fs::canonicalize(ws) else {
        return (TargetClass::New, None);
    };
    let is_file = |w: &str| {
        check_relative_path(w).is_ok() && std::fs::metadata(ws.join(w)).is_ok_and(|m| m.is_file())
    };
    let w = match crate::destination::named_destination(goal, &is_file) {
        Ok(Some(w)) => w,
        Ok(None) => return (TargetClass::New, None),
        Err(e) => return (TargetClass::Refused(format!("{e}")), None),
    };
    let class = classify_named(&w, &ws);
    (class, Some(w))
}

fn classify_named(w: &str, ws: &Path) -> TargetClass {
    let refuse = |why: &str| {
        TargetClass::Refused(format!(
            "destination {w} cannot be written safely ({why}); refusing"
        ))
    };
    if check_relative_path(w).is_err() {
        return refuse("not a plain relative path inside the workspace");
    }
    let joined = ws.join(w);
    // symlink_metadata: a dangling or outward symlink still EXISTS.
    let Ok(lmeta) = std::fs::symlink_metadata(&joined) else {
        // Missing: a new document, if its nearest existing parent is a
        // directory inside the workspace (no symlinked parent leaving it).
        let mut anc = joined.parent();
        while let Some(a) = anc {
            if std::fs::symlink_metadata(a).is_ok() {
                break;
            }
            anc = a.parent();
        }
        return match anc.map(std::fs::canonicalize) {
            Some(Ok(p)) if p.starts_with(ws) && p.is_dir() => TargetClass::New,
            Some(Ok(p)) if p.starts_with(ws) => refuse("a parent is not a directory"),
            Some(Ok(_)) => refuse("a parent resolves outside the workspace"),
            _ => refuse("unresolvable parent"),
        };
    };
    let refuse = |why: &str| {
        TargetClass::Refused(format!(
            "destination {w} already exists but cannot be edited safely ({why}); refusing instead of treating it as a new document"
        ))
    };
    let full = match std::fs::canonicalize(&joined) {
        Ok(f) => f,
        Err(_) if lmeta.file_type().is_symlink() => return refuse("unresolvable symlink"),
        Err(_) => return refuse("unresolvable path"),
    };
    if !full.starts_with(ws) {
        return refuse("resolves outside the workspace");
    }
    let Ok(meta) = std::fs::metadata(&full) else {
        return refuse("unreadable");
    };
    if meta.is_dir() {
        return refuse("it is a directory");
    }
    if !meta.is_file() {
        return refuse("not a regular file");
    }
    if meta.len() > COMPOSE_EDIT_MAX_BYTES {
        return refuse(&format!(
            "{} bytes is over the {COMPOSE_EDIT_MAX_BYTES}-byte edit limit",
            meta.len()
        ));
    }
    let Ok(bytes) = std::fs::read(&full) else {
        return refuse("unreadable");
    };
    match String::from_utf8(bytes) {
        Ok(content) => TargetClass::Edit(w.to_string(), content),
        Err(_) => refuse("not UTF-8 text"),
    }
}

/// The additive new-document block, appended when the goal names a destination
/// that does not exist yet: the model is told which file to write and must
/// answer with that `filename:` line (issue #288).
pub fn new_document_block(path: &str) -> String {
    format!("\nWrite the new file {path}. The filename line of your answer must be exactly: filename: {path}")
}

/// The additive edit-mode block (NEXT-PHASE-1 v6 T5), appended to the fixed
/// proposal template only when the goal names an existing file.
pub fn edit_block(path: &str, content: &str) -> String {
    let nl = if content.ends_with('\n') { "" } else { "\n" };
    format!(
        "\nThe file {path} already exists. Its current content is:\n{content}{nl}\
         Write the complete new content of {path}: keep every existing line and make the requested change."
    )
}

/// The prompt of the production RunComposeTask path: the fixed template,
/// plus the edit-mode block when the goal names an existing file of the
/// canonical workspace `ws` (NEXT-PHASE-1 v6).
pub fn task_prompt(goal: &str, ws: &Path) -> String {
    task_prompt_and_target(goal, ws).0
}

/// `task_prompt` plus the edit target it showed the model: the path and the
/// exact content of the block, read once (NEXT-PHASE-1 v7 T5). The Skill
/// merges an edit reply into this same content, so the bytes the model saw
/// are the bytes the merge keeps.
pub fn task_prompt_and_target(goal: &str, ws: &Path) -> (String, Option<(String, String)>) {
    let (prompt, target, _) = task_plan(goal, ws).unwrap_or_else(|_| {
        (
            proposal_prompt(goal, &ws.display().to_string()),
            None,
            ProposalKind::Document,
        )
    });
    (prompt, target)
}

/// Prompt, edit target and kind of a task, or the refusal when the goal names
/// an existing file that no edit block can be built for.
pub fn task_plan(goal: &str, ws: &Path) -> Result<TaskPrompt, String> {
    task_decision(goal, ws).map(|(plan, _)| plan)
}

/// `task_plan` plus the destination path the goal named (None when it names
/// none). The one decision of a task: a reply whose `filename:` differs from
/// the destination is refused by the Skill.
pub fn task_decision(goal: &str, ws: &Path) -> Result<(TaskPrompt, Option<String>), String> {
    let mut prompt = proposal_prompt(goal, &ws.display().to_string());
    let (class, dest) = classify_destination(goal, ws);
    let kind = class.kind();
    match class {
        TargetClass::Refused(why) => Err(format!("RunComposeTask: {why}")),
        TargetClass::Edit(path, content) => {
            prompt.push_str(&edit_block(&path, &content));
            Ok(((prompt, Some((path, content)), kind), dest))
        }
        TargetClass::New => {
            if let Some(d) = &dest {
                prompt.push_str(&new_document_block(d));
            }
            Ok(((prompt, None, kind), dest))
        }
    }
}

/// Largest prior-lines x reply-lines table `merge_edit_reply` builds.
pub const COMPOSE_EDIT_MERGE_MAX_CELLS: usize = 4 << 20;

/// NEXT-PHASE-1 v7 T5: an edit-mode reply never removes an existing line.
///
/// v6 T5 showed the model answering an edit goal with only the changed part
/// (the heading it edits under plus the new line), which the v6 Skill wrote
/// as the whole file. Here the reply's content is merged into `prior`, the
/// exact content the model was shown: the reply lines that equal prior lines
/// (longest common subsequence, trailing whitespace ignored) are anchors,
/// every other reply line is inserted next to its anchor (after the anchor
/// above it; before the first anchor when no anchor is above it), and every
/// prior line is kept with its own bytes, in order. A whole-file reply that
/// keeps every line therefore merges to itself.
///
/// Limits: edit mode is additive. A line the reply leaves out or rewrites
/// stays in the file (a rewrite appears as an added line next to the old
/// one). Err when the reply shares no non-blank line with `prior` (no place
/// for the change), when it changes nothing, or when the table would exceed
/// `COMPOSE_EDIT_MERGE_MAX_CELLS`. The result ends with one newline.
pub fn merge_edit_reply(prior: &str, reply_content: &str) -> Result<String, String> {
    let mut p: Vec<&str> = prior.split('\n').collect();
    if prior.ends_with('\n') || prior.is_empty() {
        p.pop();
    }
    let r: Vec<&str> = reply_content.lines().collect();
    let (n, m) = (p.len(), r.len());
    if (n + 1).saturating_mul(m + 1) > COMPOSE_EDIT_MERGE_MAX_CELLS {
        return Err(format!(
            "edit reply too large to merge ({n} file lines x {m} reply lines)"
        ));
    }
    let eq = |i: usize, j: usize| p[i].trim_end() == r[j].trim_end();
    // l[i][j] = LCS length of p[i..] and r[j..].
    let w = m + 1;
    let mut l = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            l[i * w + j] = if eq(i, j) {
                l[(i + 1) * w + j + 1] + 1
            } else {
                l[(i + 1) * w + j].max(l[i * w + j + 1])
            };
        }
    }
    let mut anchors: Vec<(usize, usize)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if eq(i, j) && l[i * w + j] == l[(i + 1) * w + j + 1] + 1 {
            anchors.push((i, j));
            i += 1;
            j += 1;
        } else if l[(i + 1) * w + j] >= l[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if !anchors.iter().any(|&(i, _)| !p[i].trim().is_empty()) {
        return Err(
            "edit reply shares no line with the file: copy the existing line the change goes under, then the new lines"
                .into(),
        );
    }
    let mut out: Vec<&str> = Vec::with_capacity(n + m);
    let (mut pi, mut rj) = (0, 0);
    for (k, &(ai, aj)) in anchors.iter().chain(std::iter::once(&(n, m))).enumerate() {
        if k == 0 {
            // No anchor above: new lines hug the first anchor from above.
            out.extend(&p[pi..ai]);
            out.extend(&r[rj..aj]);
        } else {
            // New lines hug the anchor above them.
            out.extend(&r[rj..aj]);
            out.extend(&p[pi..ai]);
        }
        if ai < n {
            out.push(p[ai]);
        }
        (pi, rj) = (ai + 1, aj + 1);
    }
    let mut merged = out.join("\n");
    merged.push('\n');
    let mut kept = p.join("\n");
    kept.push('\n');
    if merged == kept {
        return Err("edit reply changes nothing in the file".into());
    }
    Ok(merged)
}

/// The proposal text the Skill hands to AEGIS for one parsed reply. In edit
/// mode (`edit` = the target path and the content shown to the model) a reply
/// naming that path is merged into the content by `merge_edit_reply` and
/// returned in the canonical `filename: <path>` form; any other reply is
/// returned unchanged (the v5 whole-file proposal).
pub fn edit_proposal(reply: &str, edit: Option<(&str, &str)>) -> Result<String, String> {
    let p = check_file_proposal(reply)?;
    let Some((path, prior)) = edit.filter(|(path, _)| *path == p.path) else {
        return Ok(reply.to_string());
    };
    let merged = merge_edit_reply(prior, &p.content)?;
    let text = format!("filename: {path}\n{merged}");
    match check_file_proposal(&text) {
        Ok(q) if q.path == path && q.content == merged => Ok(text),
        _ => Err(format!(
            "the merged content of {path} does not survive the proposal format"
        )),
    }
}

/// Why a reply that stopped at the token limit is never a proposal
/// (NEXT-PHASE-1 v6 N2): its content may be cut anywhere.
pub fn length_cut_refusal(g: &Generation) -> Option<String> {
    (g.finish_reason.as_deref() == Some("max_tokens")).then(|| {
        format!(
            "reply cut at the token limit after {} tokens (finish_reason max_tokens); a cut reply is never a proposal",
            g.tokens
        )
    })
}

/// The one correction line added to attempt k > 1.
pub fn retry_prompt(base: &str, reason: &str) -> String {
    format!(
        "{base}\nYour previous answer was refused ({reason}). \
         Start your answer with the line \"filename: <relative path>\"."
    )
}

/// Runs up to `max_attempts` proposals within `budget`: attempt 1 always
/// starts; attempt k > 1 starts only if the remaining budget is at least
/// `attempt_budget` (the measured cost of one full attempt), and gets the
/// remaining budget as its limit. Returns the first reply that passes
/// `check_file_proposal` (or the last refusal reason) and every attempt made.
pub fn propose_with_retries(
    proposer: &(dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync),
    base: &str,
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
    max_attempts: u32,
) -> (Result<String, String>, Vec<ProposalAttempt>) {
    propose_task_with_retries(proposer, base, None, budget, attempt_budget, max_attempts)
}

/// `propose_with_retries` with the edit target of `task_prompt_and_target`:
/// a parsed reply becomes `edit_proposal(reply, edit)`, so in edit mode the
/// returned proposal is the merged file (NEXT-PHASE-1 v7 T5). Each attempt
/// still records the model's own reply text.
pub fn propose_task_with_retries(
    proposer: &(dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync),
    base: &str,
    edit: Option<(&str, &str)>,
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
    max_attempts: u32,
) -> (Result<String, String>, Vec<ProposalAttempt>) {
    propose_task_checked(
        proposer,
        base,
        edit,
        &[],
        budget,
        attempt_budget,
        max_attempts,
    )
}

fn new_attempt(k: u32) -> ProposalAttempt {
    ProposalAttempt {
        attempt: k,
        ms: 0,
        tokens: 0,
        outcome: String::new(),
        reason: None,
        text_sha256: None,
        text: None,
        aegis: None,
        finish_reason: None,
        token_ids: None,
        prompt_tokens: None,
        prompt_ids_sha256: None,
        decoding: None,
        ops: None,
        unmet_requirements: Vec::new(),
    }
}

/// The approved-proposal path of the Skill: exactly one attempt that returns
/// `text` unchanged, through the same requirement check as a model reply (the
/// bytes a human approved are still refused, before any write, when the goal
/// states a requirement they do not meet).
pub(crate) fn propose_approved(
    text: &str,
    prompt: &str,
    reqs: &[crate::requirements::Requirement],
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
) -> SkillOutput {
    let text = text.to_string();
    let one = move |_: &str, _: std::time::Duration| {
        Ok(Generation {
            text: text.clone(),
            finish_reason: Some("approved".into()),
            ..Default::default()
        })
    };
    propose_task_checked(&one, prompt, None, reqs, budget, attempt_budget, 1)
}

#[cfg(test)]
pub(crate) const TEST_DROP_RECORD_MARKER: &str = "drop-the-record-marker";

/// Why a task is refused when its requirement record is gone (issue #289).
pub(crate) const MISSING_RECORD_REASON: &str =
    "requirement record missing for this task: refused, an absent record is never treated as no requirements";

/// A task refused before the model ran: one attempt records the reason.
pub(crate) fn refused_without_model(why: String) -> SkillOutput {
    let mut a = new_attempt(1);
    a.outcome = "refused".into();
    a.reason = Some(why.clone());
    (Err(why), vec![a])
}

/// The AEGIS verify decision for one task: the result names exactly the
/// proposal the Skill recorded, that proposal parses as one file change, and
/// the COMPLETE content (the bytes that would be saved) meets every requirement
/// of the goal. A missing requirement record, or any uncertain span, refuses.
pub(crate) fn verify_task_result(
    ex: Option<&crate::requirements::Extraction>,
    proposal: Option<&SkillOutput>,
    result: u64,
) -> bool {
    let Some(ex) = ex else { return false };
    if ex.refusal().is_some() {
        return false;
    }
    let Some((Ok(t), _)) = proposal else {
        return false;
    };
    proposal_handle(t) == result
        && parse_file_proposal(t).is_some_and(|p| {
            crate::requirements::refusal_reason(&ex.requirements, &p.content).is_none()
        })
}

/// `propose_task_with_retries` plus requirement validation: a parsed reply
/// whose COMPLETE content (the merged file in edit mode, i.e. the bytes that
/// would be saved) fails any of `reqs` is refused like a parse failure, the
/// unmet requirement(s) are recorded on the attempt, and the next attempt
/// gets them as its retry reason. `budget` is one deadline for the whole
/// task, started when this function starts and never reset per attempt.
/// Exhaustion returns Err: no proposal exists, so nothing is approved,
/// committed or written.
pub fn propose_task_checked(
    proposer: &(dyn Fn(&str, std::time::Duration) -> Result<Generation, String> + Send + Sync),
    base: &str,
    edit: Option<(&str, &str)>,
    reqs: &[crate::requirements::Requirement],
    budget: std::time::Duration,
    attempt_budget: std::time::Duration,
    max_attempts: u32,
) -> (Result<String, String>, Vec<ProposalAttempt>) {
    let start = std::time::Instant::now();
    let mut attempts: Vec<ProposalAttempt> = Vec::new();
    // An edit only adds lines (`merge_edit_reply` keeps every prior line), so a
    // prior file that already breaks a MaxLines can never satisfy it: refuse
    // now with a recorded reason instead of spending attempts.
    if let Some((path, prior)) = edit {
        let over: Vec<String> = reqs
            .iter()
            .filter(|r| matches!(r, crate::requirements::Requirement::MaxLines(_)))
            .filter_map(|r| r.check(prior))
            .collect();
        if !over.is_empty() {
            let why = format!(
                "unmet requirement: the existing {path} already breaks it ({}) and an edit only adds lines",
                over.join("; ")
            );
            let mut a = new_attempt(1);
            a.outcome = "refused".into();
            a.reason = Some(why.clone());
            a.unmet_requirements = over;
            return (Err(why), vec![a]);
        }
    }
    let mut last_reason = "no attempt made".to_string();
    let need_ms = (attempt_budget.as_millis() as u64).max(1);
    for k in 1..=max_attempts {
        let remaining = budget.saturating_sub(start.elapsed());
        if k > 1 && (remaining.as_millis() as u64) < need_ms {
            last_reason = format!(
                "{last_reason}; no attempt {k}: {} ms left < {need_ms} ms per-attempt budget",
                remaining.as_millis()
            );
            break;
        }
        let prompt = if k == 1 {
            base.to_string()
        } else {
            retry_prompt(base, &last_reason)
        };
        let t0 = std::time::Instant::now();
        let out = proposer(&prompt, remaining);
        let last_ms = t0.elapsed().as_millis() as u64;
        let mut a = new_attempt(k);
        a.ms = last_ms;
        match out {
            Ok(g) => {
                a.tokens = g.tokens;
                a.finish_reason = g.finish_reason.clone();
                a.token_ids = g.token_ids.clone();
                a.prompt_tokens = g.prompt_tokens;
                a.prompt_ids_sha256 = g.prompt_ids_sha256.clone();
                a.decoding = g.decoding.clone();
                a.ops = g.ops.clone();
                a.text_sha256 = Some(hex(&Sha256::digest(g.text.as_bytes())));
                a.text = Some(g.text.clone());
                // NEXT-PHASE-1 v6 N2: a length-cut reply is refused before parsing.
                let checked = match length_cut_refusal(&g) {
                    Some(why) => Err(why),
                    None => edit_proposal(&g.text, edit).and_then(|p| {
                        // The complete content that would be saved.
                        let content = check_file_proposal(&p)?.content;
                        match crate::requirements::refusal_reason(reqs, &content) {
                            Some(why) => {
                                a.unmet_requirements = crate::requirements::unmet(reqs, &content);
                                Err(why)
                            }
                            None => Ok(p),
                        }
                    }),
                };
                match checked {
                    Ok(proposal) => {
                        a.outcome = "parsed".into();
                        attempts.push(a);
                        return (Ok(proposal), attempts);
                    }
                    Err(e) => {
                        a.outcome = "refused".into();
                        a.reason = Some(e.clone());
                        last_reason = e;
                    }
                }
            }
            Err(e) => {
                a.outcome = if e.starts_with(WALL_CLOCK_REASON_PREFIX) {
                    "timeout"
                } else {
                    "error"
                }
                .into();
                a.reason = Some(e.clone());
                last_reason = e;
            }
        }
        attempts.push(a);
    }
    (Err(last_reason), attempts)
}

pub(crate) fn record_view(
    compose: &mut Compose,
    r: &aien_omega_compose::Record,
) -> ComposeRecordView {
    let host = r.subject == aien_omega_compose::SUBJECT_HOST;
    let note = if !host {
        None
    } else if r.tag == aien_omega_compose::ffi::RXC_HOST_TAG_REPAIR_TAIL {
        Some("repair_tail".to_string())
    } else {
        NoteKind::from_tag(r.tag).map(|k| k.name().to_string())
    };
    let text = if host && NoteKind::from_tag(r.tag).is_some() {
        compose
            .payload(r.id)
            .ok()
            .and_then(|p| note_bytes(&p))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    } else {
        None
    };
    ComposeRecordView {
        id: r.id,
        cls: r.cls,
        kind: r.kind,
        subject: r.subject,
        tag: r.tag,
        links: r.links.to_vec(),
        digest: hex(&r.digest),
        verified: r.verified == 1,
        note,
        text,
    }
}

const CLOSED_REFUSAL: &str = "compose home closed: the daemon is shutting down";

pub(crate) struct ComposeHome {
    pub(crate) compose: Compose,
    /// The ALLEN identity this home resolved (`None` = not engaged).
    pub(crate) allen: Option<aien_allen::Resolved>,
    /// ALLEN scoped memory (arch#159), opened with the identity above.
    pub(crate) memory: crate::allen_memory::MemoryState,
    pub(crate) machine_id: String,
    prompts: Arc<parking_lot::Mutex<HashMap<u64, TaskEntry>>>,
    /// Proposer hook: task -> approved proposal text (crate::approved).
    approved: Arc<parking_lot::Mutex<HashMap<u64, String>>>,
    /// What the model Skill returned per task: the text, or why it gave
    /// none, and every attempt it made.
    proposals: Arc<parking_lot::Mutex<HashMap<u64, SkillOutput>>>,
    /// The Cortex record mark (ACCEPTANCE-v3 2.1): where it lives, the raw
    /// machine id it binds, and the last mark written (or verified).
    mark_path: PathBuf,
    machine_raw: [u8; 32],
    mark: Option<Mark>,
    /// seq of the last mark this home knows of (0 = none yet).
    mark_seq: u64,
    /// Wall budgets in force; every run sets omega's wait from its own.
    budgets: ComposeBudgets,
}

impl ComposeHome {
    /// Write a new record mark when the journal holds more records than the
    /// last mark (ACCEPTANCE-v3 2.1). Called only after appends returned,
    /// so the mark never runs ahead of the journal. Returns the mark written.
    pub(crate) fn advance_mark(&mut self) -> Result<Option<Mark>, String> {
        let n = self
            .compose
            .info()
            .map_err(|e| format!("record mark: info: {e}"))?
            .records;
        if self.mark.as_ref().is_some_and(|m| m.records >= n) {
            return Ok(None);
        }
        let digest = if n == 0 {
            [0u8; 32]
        } else {
            self.compose
                .record(n)
                .map_err(|e| format!("record mark: record {n}: {e}"))?
                .digest
        };
        let m = Mark {
            machine_id: self.machine_raw,
            seq: self.mark_seq + 1,
            records: n,
            digest,
        };
        cortex_mark::write(&self.mark_path, &m)?;
        self.mark_seq = m.seq;
        self.mark = Some(m.clone());
        Ok(Some(m))
    }
}

type SkillOutput = (Result<String, String>, Vec<ProposalAttempt>);
/// The prompt of one task, its edit target and its kind (`task_plan`).
pub type TaskPrompt = (String, Option<(String, String)>, ProposalKind);
/// A task's plan plus the requirements its goal states (`requirements::extract`).
/// Prompt plan, requirements, and the destination the goal named (one decision).
type TaskEntry = (
    TaskPrompt,
    crate::requirements::Extraction,
    (Option<String>, PathBuf),
);

/// Owns the composition home of this process.
pub struct ComposeBridge {
    dir: PathBuf,
    proposer: ComposeProposer,
    /// Proposer for full-document tasks (own token cap); `None` = `proposer`.
    doc_proposer: Option<ComposeProposer>,
    proposer_label: String,
    home: std::sync::Mutex<Option<ComposeHome>>,
    /// Set when the start-up reconcile failed or was refused (ACCEPTANCE-v3
    /// 2.5): effect commands refuse until an operator reconcile succeeds.
    reconcile_failed: std::sync::Mutex<Option<String>>,
    /// Set by `close`: the daemon is shutting down, the home is closed and
    /// must not be reopened lazily by a connection task that outlives `run`
    /// (sovereign-core #306).
    closed: std::sync::atomic::AtomicBool,
    /// sovereign-core #297, #328: when true, `ComposeAuthorize` needs the
    /// approval desk's MAC. Default true (Drake decision b, 2026-10-08);
    /// false is the legacy OS-user-only authorize, only by explicit choice.
    authorize_requires_desk: bool,
    /// The loaded model files and the daemon start, set by the daemon after
    /// load. Without them a compose proposal gets no generation record.
    identity: std::sync::Mutex<
        Option<(
            crate::generation::ModelIdentity,
            crate::generation::DaemonStart,
        )>,
    >,
}

impl ComposeBridge {
    pub fn new(dir: PathBuf, proposer: ComposeProposer, proposer_label: &str) -> Self {
        Self {
            dir,
            proposer,
            doc_proposer: None,
            proposer_label: proposer_label.to_string(),
            home: std::sync::Mutex::new(None),
            reconcile_failed: std::sync::Mutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
            authorize_requires_desk: true,
            identity: std::sync::Mutex::new(None),
        }
    }

    /// Use `doc_proposer` (its own token cap) for full-document tasks.
    pub fn with_doc_proposer(mut self, doc_proposer: ComposeProposer) -> Self {
        self.doc_proposer = Some(doc_proposer);
        self
    }

    /// The model files this daemon loaded (digests hashed at load) and when it
    /// started. Set once by the daemon; the compose task uses them to write the
    /// generation record of the proposal it commits. Evidence only.
    pub fn set_model_identity(
        &self,
        identity: crate::generation::ModelIdentity,
        started: crate::generation::DaemonStart,
    ) {
        *self.identity.lock().unwrap_or_else(|e| e.into_inner()) = Some((identity, started));
    }

    /// Turn the desk-MAC requirement on `ComposeAuthorize` on or off (#297).
    /// On by default (#328); off is for dev and tests of the legacy path, and a
    /// server refuses to start with it off in a strict run.
    pub fn with_authorize_requires_desk(mut self, on: bool) -> Self {
        self.authorize_requires_desk = on;
        self
    }

    /// True when `ComposeAuthorize` needs the approval desk's MAC (#297).
    pub fn authorize_requires_desk(&self) -> bool {
        self.authorize_requires_desk
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Close the compose home and refuse to reopen it. Daemon shutdown calls
    /// this before `run` returns: connection tasks hold their own `Arc` of the
    /// bridge and can outlive `run`, so without it `rxc_host_close` (the drop of
    /// the home) ran at an unspecified later time and a successor opening the
    /// same home in that window saw a journal behind its J-Space anchor
    /// (E_REPLAY, sovereign-core #306). Waits for a command in flight (it holds
    /// the home lock). Blocking: call it from a blocking context.
    pub fn close(&self) {
        let mut guard = match self.home.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        *guard = None;
    }

    /// True after `close`.
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn open_home(&self) -> Result<ComposeHome, String> {
        self.open_home_marked(None)
    }

    /// Open the home and check its Cortex record mark (ACCEPTANCE-v3 2.2).
    /// `rebuilt_from` = the seq of a mark RecoverComposeHome just set aside:
    /// the home then opens without a mark and gets a fresh one.
    fn open_home_marked(&self, rebuilt_from: Option<u64>) -> Result<ComposeHome, String> {
        // The one place every lazy open goes through (with_home, run_task_inner,
        // record_digest, recover): callers hold the home lock and close() sets
        // `closed` under the same lock, so no site can reopen after close.
        if self.is_closed() {
            return Err(CLOSED_REFUSAL.to_string());
        }
        // Refuse a bad budget before anything is opened (never fall back).
        let budgets = compose_budgets_from_env()
            .map_err(|e| format!("compose home {} refused: {e}", self.dir.display()))?;
        let root = machine_root(&self.dir)?;
        let mark_path = cortex_mark::mark_path(&self.dir);
        let existing = match rebuilt_from {
            Some(_) => None,
            None => cortex_mark::read(&mark_path)
                .map_err(|why| mark_refusal(&self.dir, &MarkRefusal::Damaged(why)))?,
        };
        let journal_present = std::fs::metadata(self.dir.join("cortex.cx"))
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        // rxc_host_open appends nothing: `info.records` is the read-only probe
        // count, so the count check runs before the open appends anything.
        let (mut compose, info) =
            Compose::open(&self.dir, RootKind::Provisioned, &root, 0xA1E4_0001)
                .map_err(|e: ComposeError| refusal(&self.dir, &e))?;
        // omega settles after a fixed wait. Every run sets it again to its own
        // budget + 1 s (run_task_inner); here the edit value is checked once, so an
        // old omega without rxc_host_set_wait_ms refuses a changed edit budget at open.
        compose
            .set_wait_ms((budgets.edit + COMPOSE_WAIT_MARGIN).as_millis() as u32)
            .map_err(|e| refusal(&self.dir, &e))?;
        let plan = cortex_mark::plan(
            existing.as_ref(),
            &info.machine_id,
            info.records,
            journal_present,
        )
        .map_err(|r| mark_refusal(&self.dir, &r))?;
        let prompts: Arc<parking_lot::Mutex<HashMap<u64, TaskEntry>>> = Arc::default();
        let proposals: Arc<parking_lot::Mutex<HashMap<u64, SkillOutput>>> = Arc::default();
        let approved: Arc<parking_lot::Mutex<HashMap<u64, String>>> = Arc::default();
        let (pr, pp, proposer) = (prompts.clone(), proposals.clone(), self.proposer.clone());
        let doc_proposer = self
            .doc_proposer
            .clone()
            .unwrap_or_else(|| self.proposer.clone());
        let pa = approved.clone();
        compose
            .register_skill(COMPOSE_MODEL_SKILL, None, 10, move |task| {
                // The requirement record of this task. A missing record is a refusal,
                // never an empty requirement list (issue #289).
                let entry = pr.lock().get(&task).cloned();
                let Some(((prompt, target, kind), ex, dest)) = entry else {
                    pp.lock()
                        .insert(task, refused_without_model(MISSING_RECORD_REASON.into()));
                    return None;
                };
                // Goal text that looks like a measurable requirement but could not
                // be read reliably: refuse before any model call.
                if let Some(why) = ex.refusal() {
                    pp.lock().insert(task, refused_without_model(why));
                    return None;
                }
                let reqs = ex.requirements;
                // Proposer hook (crate::approved): an approved proposal for this
                // task replaces the model's reply. One attempt, the same
                // template check, the same AEGIS contract and commit below.
                // edit = None: an approved whole-file proposal never goes
                // through merge_edit_reply, so the committed bytes are the
                // approved bytes even when the path already exists.
                let budget = budgets.for_kind(kind);
                let attempt_budget = compose_attempt_budget(kind, budget);
                let fixed = pa.lock().get(&task).cloned();
                let (out, attempts) = match fixed {
                    Some(text) => propose_approved(&text, &prompt, &reqs, budget, attempt_budget),
                    // Fixed template + bounded automatic retry (ACCEPTANCE-v2 3a, 3b),
                    // measured per-attempt budget (ACCEPTANCE-v3 3b); an edit reply
                    // is merged into the content the model was shown (v7 T5).
                    None => {
                        let edit = target.as_ref().map(|(p, c)| (p.as_str(), c.as_str()));
                        let p = match kind {
                            ProposalKind::Edit => &proposer,
                            ProposalKind::Document => &doc_proposer,
                        };
                        propose_task_checked(
                            p.as_ref(),
                            &prompt,
                            edit,
                            &reqs,
                            budget,
                            attempt_budget,
                            COMPOSE_MAX_ATTEMPTS,
                        )
                    }
                };
                let (out, attempts) = enforce_destination(
                    (out, attempts),
                    dest.0.as_deref(),
                    &dest.1,
                    kind == ProposalKind::Edit,
                );
                let h = out.as_ref().ok().map(|t| proposal_handle(t));
                pp.lock().insert(task, (out, attempts));
                // Test seam (compiled only into unit tests): lose the requirement
                // record between the Skill and AEGIS verify, as a bug would (#289).
                #[cfg(test)]
                if prompt.contains(TEST_DROP_RECORD_MARKER) {
                    pr.lock().remove(&task);
                }
                h
            })
            .map_err(|e| format!("compose register skill: {e}"))?;
        let (pv, pq) = (proposals.clone(), prompts.clone());
        compose
            .set_verify(move |task, result| {
                // AEGIS contract: the result names exactly the proposal text
                // the Skill recorded for this task, and that text parses as one
                // file change (a relative path and nonempty content).
                // The requirements of the goal are checked again on these same
                // bytes, the ones that would be saved. No requirement record
                // for the task means NO approval (issue #289).
                let ex = pq.lock().get(&task).map(|p| p.1.clone());
                let proposals = pv.lock();
                verify_task_result(ex.as_ref(), proposals.get(&task), result)
            })
            .map_err(|e| format!("compose set verify: {e}"))?;
        // Open the composition now, so a home behind its anchor is refused here
        // with its name (E_REPLAY), not inside the first run.
        compose.info().map_err(|e| refusal(&self.dir, &e))?;
        // The digest check needs the open home (the open appended the five
        // start-up records; ACCEPTANCE-v3 G1, G3).
        if let Plan::Verify(m) = &plan {
            if m.records > 0 {
                let at = compose.record(m.records).ok().map(|r| r.digest);
                cortex_mark::check_digest(m, at).map_err(|r| mark_refusal(&self.dir, &r))?;
            }
        }
        // ALLEN identity gate: after the open (record 1 exists, skills are
        // registered), before the home is handed out. Fatal when engaged.
        let allen = allen_gate(&self.dir, &mut compose, &info.machine_id);
        let memory = crate::allen_memory::MemoryState::open(&self.dir, allen.as_ref());
        let mut home = ComposeHome {
            compose,
            allen,
            memory,
            budgets,
            machine_id: hex(&info.machine_id),
            prompts,
            approved,
            proposals,
            mark_path,
            machine_raw: info.machine_id,
            mark: existing.clone(),
            mark_seq: existing
                .as_ref()
                .map_or(rebuilt_from.unwrap_or(0), |m| m.seq),
        };
        let written = home.advance_mark().map_err(|e| {
            format!(
                "compose home {} refused: record mark update failed (E_MARK_WRITE): {e}",
                self.dir.display()
            )
        })?;
        let event = match rebuilt_from {
            Some(_) => written.as_ref().map(|w| {
                format!(
                    "Cortex mark: mark rebuilt by RecoverComposeHome (seq {}, records {})",
                    w.seq, w.records
                )
            }),
            None => cortex_mark::open_event(&plan, info.records, written.as_ref()),
        };
        if let Some(line) = event {
            println!("{line}");
        }
        Ok(home)
    }

    /// Start-up reconcile failed or was refused: effect commands refuse
    /// until an operator reconcile succeeds (ACCEPTANCE-v3 2.5).
    pub fn set_reconcile_failed(&self, why: String) {
        if let Ok(mut g) = self.reconcile_failed.lock() {
            *g = Some(why);
        }
    }

    /// Why effect commands are refused, if they are.
    pub fn reconcile_failed(&self) -> Option<String> {
        match self.reconcile_failed.lock() {
            Ok(g) => g.clone(),
            Err(_) => Some("reconcile state lock poisoned".into()),
        }
    }

    pub(crate) fn clear_reconcile_failed(&self) {
        if let Ok(mut g) = self.reconcile_failed.lock() {
            *g = None;
        }
    }

    /// Blocking: runs inference inside the Skill. Call from a blocking thread.
    pub fn run_task(&self, goal: &str, workspace: &str) -> ControlResponse {
        self.run_task_in(goal, workspace, None)
    }

    /// `run_task` with an operator-named ALLEN memory context (`None` = no memory).
    pub fn run_task_in(
        &self,
        goal: &str,
        workspace: &str,
        context: Option<&str>,
    ) -> ControlResponse {
        match self.run_task_inner(goal, workspace, None, context) {
            Ok(r) => ControlResponse::ComposeTaskResult(Box::new(r)),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// Proposer hook entry (crate::approved): one compose run whose model
    /// Skill returns `approved_text` instead of calling the proposer. The run
    /// is otherwise the RunComposeTask run: J-Space branch, AEGIS verify
    /// callback, World commit, Cortex records. Blocking.
    pub(crate) fn run_approved_task(
        &self,
        goal: &str,
        workspace: &str,
        approved_text: &str,
    ) -> Result<ComposeTaskReport, String> {
        self.run_task_inner(goal, workspace, Some(approved_text), None)
    }

    fn run_task_inner(
        &self,
        goal: &str,
        workspace: &str,
        approved_text: Option<&str>,
        context: Option<&str>,
    ) -> Result<ComposeTaskReport, String> {
        if goal.trim().is_empty() {
            return Err("RunComposeTask: empty goal".into());
        }
        let ws = std::fs::canonicalize(workspace)
            .map_err(|e| format!("RunComposeTask: workspace {workspace}: {e}"))?;
        if !ws.is_dir() {
            return Err(format!(
                "RunComposeTask: workspace {} is not a directory",
                ws.display()
            ));
        }
        let (mut plan, dest) = task_decision(goal, &ws)?;
        // The requirements the goal states are checked on every path: the model
        // path and an approved proposal (refused before any write when unmet).
        let machine_goal =
            approved_text.is_some() && goal.starts_with(crate::approved::APPROVED_GOAL_PREFIX);
        let mut reqs = if machine_goal {
            crate::requirements::Extraction::default()
        } else {
            crate::requirements::analyze(goal)
        };
        // Added-line counts are measured against the file the proposal replaces
        // (the edit target's content, or nothing for a new file).
        if reqs.needs_prior() {
            let prior = plan.1.as_ref().map_or("", |(_, c)| c.as_str());
            reqs = reqs.resolved(prior);
        }
        let recognized: Vec<String> = reqs.requirements.iter().map(|r| r.label()).collect();
        let uncertain = reqs.uncertain.clone();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("clock: {e}"))?;
        let mut h = Sha256::new();
        h.update(goal.as_bytes());
        h.update([0u8]);
        h.update(ws.as_os_str().as_encoded_bytes());
        h.update(now.as_nanos().to_le_bytes());
        let d = h.finalize();
        let task = u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) | 1;

        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        // ALLEN persona (arch#159): only a model run reads it; an approved
        // proposal runs no model, so it carries no persona. Not engaged: the
        // prompt is exactly what it was.
        // ALLEN memory (arch#159): a model run with an operator-named context only.
        let (memory_block, memory) = match approved_text {
            None => {
                let (b, r) = crate::allen_memory::for_task(&home.memory, context)?;
                (b, Some(r))
            }
            Some(_) => (None, None),
        };
        plan.0 = crate::allen_memory::prefix_prompt(&plan.0, memory_block.as_deref());
        let persona_ctx = match approved_text {
            None => crate::persona::context_for(&self.dir, home.allen.as_ref()),
            Some(_) => None,
        };
        plan.0 = crate::persona::prefix_prompt(&plan.0, persona_ctx.as_ref());
        let persona = match approved_text {
            None => Some(crate::persona::report_for(persona_ctx.as_ref())),
            Some(_) => None,
        };
        // Deadlines agree on both sides: omega settles budget + 1 s after the run
        // starts, the Skill gives up at budget. Set only here, under the lock that
        // serializes runs, never while rx_compose_run is in flight. An omega without
        // rxc_host_set_wait_ms refuses a budget that needs a longer wait (no silent 30 s wait).
        let kind = plan.2;
        let wait = home.budgets.for_kind(kind) + COMPOSE_WAIT_MARGIN;
        home.compose
            .set_wait_ms(wait.as_millis() as u32)
            .map_err(|e| {
                format!(
                    "compose run refused: {kind:?} budget needs omega settle wait {} ms: {e}",
                    wait.as_millis()
                )
            })?;
        if let Some(t) = approved_text {
            home.approved.lock().insert(task, t.to_string());
        }
        home.prompts
            .lock()
            .insert(task, (plan, reqs, (dest, ws.clone())));
        let run = home.compose.run(task, now.as_micros() as u64);
        home.prompts.lock().remove(&task);
        home.approved.lock().remove(&task);
        let output = home.proposals.lock().remove(&task);
        // A run appends whether or not it commits: move the record mark first.
        home.advance_mark()
            .map_err(|e| format!("compose run: record mark update failed (E_MARK_WRITE): {e}"))?;
        let r = run.map_err(|e| format!("compose run: {e}"))?;
        let committed = r.committed == 1;
        let (out, mut proposal_attempts) =
            output.unwrap_or((Err("the Skill did not run".into()), Vec::new()));
        // The attempt handed to AEGIS is the parsed one; record its verdict.
        if let Some(a) = proposal_attempts.iter_mut().find(|a| a.outcome == "parsed") {
            a.aegis = Some(if committed { "pass" } else { "fail" }.into());
        }
        let (proposal, uncommitted_proposal, proposer_error) = match out {
            Ok(t) if committed => (Some(t), None, None),
            Ok(t) => (None, Some(t), None),
            Err(e) => (None, None, Some(e)),
        };
        let parsed = proposal.as_deref().and_then(parse_file_proposal);
        // sovereign-core #261: the daemon's own record that THIS run committed
        // THIS proposal. `ComposeAuthorize` mints only from it. An approved
        // proposal has its own replay claim and grant, so it gets none.
        let mut compose_commit = None;
        let mut generation_record = None;
        if let (true, None, Some(p), Some(text)) = (
            committed,
            approved_text,
            parsed.as_ref(),
            proposal.as_deref(),
        ) {
            let wsc = std::fs::canonicalize(&ws)
                .map_err(|e| format!("workspace {}: {e}", ws.display()))?
                .display()
                .to_string();
            // Provenance (arch#162): the generation record of the attempt the
            // proposal came from, written now under the lock this run holds
            // (the model call itself cannot take it). Evidence only.
            generation_record = self.write_task_generation(home, task, &proposal_attempts);
            let prov = crate::generation::Provenance::new(generation_record, home.allen.as_ref());
            let id = crate::effects::write_compose_commit(
                home,
                &wsc,
                task,
                r.cx_promotion,
                r.cx_evidence,
                &hex(&Sha256::digest(text.as_bytes())),
                &p.path,
                &hex(&Sha256::digest(p.content.as_bytes())),
                &prov,
            )
            .map_err(|e| {
                format!(
                    "compose run committed (promotion #{}) but its commit record was not written: {e}",
                    r.cx_promotion
                )
            })?;
            home.advance_mark().map_err(|e| {
                format!("compose run: record mark update failed (E_MARK_WRITE): {e}")
            })?;
            compose_commit = Some(id);
        }
        Ok(ComposeTaskReport {
            compose_commit,
            generation_record,

            compose_dir: self.dir.display().to_string(),
            machine_id: home.machine_id.clone(),
            task,
            outcome: r.outcome,
            committed,
            branch_count: r.n_branches,
            branches_reclaimed: r.branches_reclaimed,
            winner: (r.winner != aien_omega_compose::NONE).then_some(r.winner),
            aegis_pass_mask: r.aegis_pass_mask,
            cx_goal: r.cx_goal,
            cx_candidates: r.cx_candidate.iter().copied().filter(|&x| x != 0).collect(),
            cx_evidence: r.cx_evidence,
            cx_promotion: r.cx_promotion,
            cx_admissions: r.cx_admission.iter().copied().filter(|&x| x != 0).collect(),
            winner_digest: hex(&r.winner_digest),
            record_digest: hex(&r.record_digest),
            proposal_path: parsed.as_ref().map(|p| p.path.clone()),
            proposal_content_sha256: parsed
                .as_ref()
                .map(|p| hex(&Sha256::digest(p.content.as_bytes()))),
            proposal_sha256: proposal
                .as_ref()
                .map(|t| hex(&Sha256::digest(t.as_bytes()))),
            proposal,
            uncommitted_proposal,
            proposer_error,
            proposal_attempts,
            requirements_recognized: recognized,
            requirements_uncertain: uncertain,
            proposer: match approved_text {
                Some(_) => APPROVED_PROPOSER_LABEL.to_string(),
                None => self.proposer_label.clone(),
            },
            persona,
            memory,
        })
    }

    /// Write the generation record of the proposal attempt a committed task
    /// came from. `None` (with the reason logged when a write failed) when the
    /// daemon has no model identity, the attempt did not expose its token ids,
    /// or the append failed: no record means no claim, and the commit stands.
    fn write_task_generation(
        &self,
        home: &mut ComposeHome,
        task: u64,
        attempts: &[ProposalAttempt],
    ) -> Option<u64> {
        let (identity, started) = self
            .identity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        let a = attempts.iter().find(|a| a.outcome == "parsed")?;
        let (ids, text, prompt_sha, prompt_tokens) = (
            a.token_ids.as_deref()?,
            a.text.as_deref()?,
            a.prompt_ids_sha256.as_deref()?,
            a.prompt_tokens?,
        );
        let record = crate::generation::build_compose_record(
            &identity,
            &crate::generation::ComposeEvidence {
                prompt_ids_sha256: prompt_sha,
                prompt_tokens,
                output_ids: ids,
                text,
                finish_reason: a.finish_reason.as_deref().unwrap_or("unknown"),
                task,
                attempt: a.attempt,
                decoding: a.decoding.as_ref(),
                ops: a.ops.as_ref(),
            },
            started,
        );
        match crate::generation::write(home, &record) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("generation record not written: {e}");
                eprintln!("generation record not written: {e}");
                None
            }
        }
    }

    /// Append the daemon's generation record (evidence only; see
    /// `crate::generation`). Returns the record id only once it is written.
    pub fn record_generation(&self, record: &serde_json::Value) -> Result<u64, String> {
        self.with_home(|home| crate::generation::write(home, record))
    }

    pub(crate) fn with_home<T>(
        &self,
        f: impl FnOnce(&mut ComposeHome) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        let r = f(home);
        // Ok or not, the closure may have appended: the mark follows.
        if let Err(e) = home.advance_mark() {
            return Err(format!(
                "record mark update failed after the operation (E_MARK_WRITE): {e}"
            ));
        }
        r
    }

    /// S1 / S5: one operator record through the composition's writer. Gated
    /// records cannot be forged here, on any path (the socket handler calls
    /// this too): authorizations, effect phases, replay and compose-commit
    /// records are written only by the daemon (sovereign-core #261).
    pub fn note(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        if let Err(e) = crate::effects::check_reserved_note(kind, text) {
            return ControlResponse::Error(e);
        }
        self.write_note(kind, text, links)
    }

    /// `note` WITHOUT the reserved-record check. Cargo feature `test-support`
    /// only: it stands for an attacker who can append to the journal directly,
    /// so the ledger's own defences can be exercised. Not in a normal build.
    #[doc(hidden)]
    #[cfg(feature = "test-support")]
    pub fn note_unchecked(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        self.write_note(kind, text, links)
    }

    fn write_note(&self, kind: &str, text: &str, links: &[u64]) -> ControlResponse {
        let k = match kind {
            "constraint" => NoteKind::Constraint,
            "authorization" => NoteKind::Authorization,
            "effect" => NoteKind::Effect,
            other => {
                return ControlResponse::Error(format!(
                    "ComposeNote: kind {other:?} (constraint, authorization, effect)"
                ))
            }
        };
        if links.len() > 4 {
            return ControlResponse::Error("ComposeNote: at most 4 links".into());
        }
        let mut l = [0u64; 4];
        l[..links.len()].copy_from_slice(links);
        let r = self.with_home(|home| {
            let id = home
                .compose
                .note(k, l, text.as_bytes())
                .map_err(|e| format!("ComposeNote: {e}"))?;
            let rec = home
                .compose
                .record(id)
                .map_err(|e| format!("ComposeNote: re-read {id}: {e}"))?;
            Ok(ComposeNoteReport {
                machine_id: home.machine_id.clone(),
                id,
                kind: k.name().to_string(),
                digest: hex(&rec.digest),
                text_sha256: hex(&Sha256::digest(text.as_bytes())),
                links: links.to_vec(),
            })
        });
        match r {
            Ok(r) => ControlResponse::ComposeNoted(r),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// ALLEN persona profile commands (arch#159). Opens the home if needed (the
    /// identity is resolved there). Blocking.
    pub fn allen_command(&self, cmd: &ControlCommand) -> ControlResponse {
        let r = self.with_home(|home| {
            if crate::allen_memory::is_memory_command(cmd) {
                return Ok(crate::allen_memory::handle_command(&home.memory, cmd));
            }
            Ok(crate::persona::handle_allen_command(
                &self.dir,
                home.allen.as_ref(),
                &self.proposer_label,
                cmd,
            ))
        });
        r.unwrap_or_else(ControlResponse::Error)
    }

    /// S6 / S8: host records plus the cited ids, digests re-checked.
    pub fn recall(&self, ids: &[u64], prefix: Option<u64>) -> ControlResponse {
        let r = self.with_home(|home| {
            let info = home
                .compose
                .info()
                .map_err(|e| format!("ComposeRecall: {e}"))?;
            let (host_recs, _) = home
                .compose
                .recall(aien_omega_compose::SUBJECT_HOST, 4096)
                .map_err(|e| format!("ComposeRecall: host records: {e}"))?;
            let host = host_recs
                .iter()
                .map(|r| record_view(&mut home.compose, r))
                .collect();
            let (mut cited, mut missing) = (Vec::new(), Vec::new());
            for &id in ids {
                match home.compose.record(id) {
                    Ok(r) => cited.push(record_view(&mut home.compose, &r)),
                    Err(_) => missing.push(id),
                }
            }
            let prefix_digest = match prefix {
                Some(n) if n <= info.records => {
                    let mut h = Sha256::new();
                    for id in 1..=n {
                        let r = home
                            .compose
                            .record(id)
                            .map_err(|e| format!("ComposeRecall: record {id}: {e}"))?;
                        h.update(r.digest);
                    }
                    Some(hex(&h.finalize()))
                }
                _ => None,
            };
            Ok(ComposeRecallReport {
                compose_dir: self.dir.display().to_string(),
                machine_id: home.machine_id.clone(),
                records_total: info.records,
                host,
                cited,
                missing,
                prefix,
                prefix_digest,
                compose_native: aien_omega_compose::LINKED,
                omega_sha: if aien_omega_compose::LINKED {
                    aien_omega_compose::EXPECTED_OMEGA_SHA.to_string()
                } else {
                    String::new()
                },
            })
        });
        match r {
            Ok(r) => ControlResponse::ComposeRecalled(Box::new(r)),
            Err(e) => ControlResponse::Error(e),
        }
    }

    /// Operator repair: close this process's handle, run rxc_host_recover,
    /// then check the Cortex record mark (ACCEPTANCE-v3 2.3): a mark that is
    /// damaged, ahead of the records kept, or names a different record is
    /// kept as `<mark>.lost-<seq>` (never deleted), the home is reopened with
    /// a fresh mark and one host `constraint` record names the repair.
    pub fn recover(&self) -> ControlResponse {
        if self.is_closed() {
            return ControlResponse::Error(CLOSED_REFUSAL.to_string());
        }
        let mut guard = match self.home.lock() {
            Ok(g) => g,
            Err(_) => return ControlResponse::Error("compose home lock poisoned".into()),
        };
        *guard = None;
        let root = match machine_root(&self.dir) {
            Ok(r) => r,
            Err(e) => return ControlResponse::Error(format!("RecoverComposeHome: {e}")),
        };
        let mark_path = cortex_mark::mark_path(&self.dir);
        let old_mark = cortex_mark::read(&mark_path);
        let r = match Compose::recover(&self.dir, RootKind::Provisioned, &root) {
            Ok(r) => r,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome {}: {e}",
                    self.dir.display()
                ))
            }
        };
        let mut report = ComposeRecoverReport {
            compose_dir: self.dir.display().to_string(),
            repaired: r.repaired == 1,
            tail_torn: r.tail_torn == 1,
            cause: r.cause,
            cut_lo: r.cut_lo,
            cut_hi: r.cut_hi,
            records_kept: r.records_kept,
            dropped_records: r.dropped_records,
            anchor_records: r.anchor_records,
            repair_record: r.event_id,
            cut_bytes_kept: r.cut_bytes_kept,
            cut_sha256: hex(&r.cut_sha256),
            opens: r.opens == 1,
            open_rc: r.open_rc,
            rolled_back: r.rolled_back,
            recovered_completed: r.recovered_completed,
            mark_lost: 0,
            mark_kept_as: None,
            mark_repair_record: 0,
        };
        // (old mark if readable, records lost, why) when the mark must go.
        let set_aside: Option<(Option<Mark>, u64, &str)> = match old_mark {
            Err(_) => Some((None, 0, "damaged")),
            Ok(None) => None,
            Ok(Some(m)) if m.records > r.records_kept => {
                let lost = m.records - r.records_kept;
                Some((Some(m), lost, "ahead of the journal"))
            }
            Ok(Some(m)) => match self.open_home() {
                Ok(home) => {
                    *guard = Some(home);
                    None
                }
                Err(e) if e.contains("E_MARK_DIGEST") => {
                    Some((Some(m), 0, "names a different record"))
                }
                // Refused for another reason: reported as before, mark kept.
                Err(_) => None,
            },
        };
        let Some((old, lost, why)) = set_aside else {
            return ControlResponse::ComposeRecovered(Box::new(report));
        };
        let old_seq = old.as_ref().map(|m| m.seq);
        let kept_as = cortex_mark::lost_path(&mark_path, old_seq);
        if let Err(e) = std::fs::rename(&mark_path, &kept_as) {
            return ControlResponse::Error(format!(
                "RecoverComposeHome: keep the record mark as {}: {e}",
                kept_as.display()
            ));
        }
        let mut home = match self.open_home_marked(Some(old_seq.unwrap_or(0))) {
            Ok(h) => h,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome: record mark kept as {}; the home still refuses: {e}",
                    kept_as.display()
                ))
            }
        };
        let text = serde_json::json!({
            "repair": "cortex-mark",
            "why": why,
            "mark_records": old.as_ref().map(|m| m.records),
            "journal_records": r.records_kept,
            "lost": lost,
            "old_seq": old_seq,
            "old_digest": old.as_ref().map(|m| hex(&m.digest)),
            "kept_as": kept_as.display().to_string(),
        })
        .to_string();
        let id = match home
            .compose
            .note(NoteKind::Constraint, [0; 4], text.as_bytes())
            .and_then(|id| home.compose.record(id).map(|_| id))
        {
            Ok(id) => id,
            Err(e) => {
                return ControlResponse::Error(format!(
                    "RecoverComposeHome: record mark kept as {}; repair record failed: {e}",
                    kept_as.display()
                ))
            }
        };
        if let Err(e) = home.advance_mark() {
            return ControlResponse::Error(format!(
                "RecoverComposeHome: repair record #{id} written; record mark update failed (E_MARK_WRITE): {e}"
            ));
        }
        report.opens = true;
        report.mark_lost = lost;
        report.mark_kept_as = Some(kept_as.display().to_string());
        report.mark_repair_record = id;
        *guard = Some(home);
        ControlResponse::ComposeRecovered(Box::new(report))
    }

    /// Recall: the Cortex record by id, digest re-checked (cut 2 uses this
    /// to prove cited ids survive a restart).
    pub fn record_digest(&self, id: u64) -> Result<String, String> {
        let mut guard = self
            .home
            .lock()
            .map_err(|_| "compose home lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(self.open_home()?);
        }
        let home = guard.as_mut().expect("opened above");
        home.compose
            .record(id)
            .map(|r| hex(&r.digest))
            .map_err(|e| format!("compose record {id}: {e}"))
    }
}

/// The reply must write the destination the goal named (issue #288): the
/// approved proposal and a model reply are both held to the same path, so the
/// save and the commit land on the file the prompt and budget were chosen for.
/// A goal that named no destination leaves the model free to name the file.
fn enforce_destination(
    out: SkillOutput,
    dest: Option<&str>,
    ws: &Path,
    is_edit: bool,
) -> SkillOutput {
    let (res, attempts) = out;
    let res = match res {
        Ok(t) => match parse_file_proposal(&t) {
            Some(p) if dest.is_some_and(|d| p.path != d) => Err(format!(
                "the proposal writes {} but the goal named the destination {}",
                p.path,
                dest.unwrap_or_default()
            )),
            // Defence in depth: a new-document task never replaces a file
            // that exists (symlinks and dangling links count as existing).
            Some(p) if !is_edit && std::fs::symlink_metadata(ws.join(&p.path)).is_ok() => {
                Err(format!(
                    "the proposal writes {} which already exists; refusing to overwrite it as a new document",
                    p.path
                ))
            }
            _ => Ok(t),
        },
        r => r,
    };
    (res, attempts)
}

#[cfg(test)]
mod requirement_tests {
    use super::*;
    use crate::requirements::extract;

    const SHORT: &str = "filename: DOC.md\na\nb\nc\n";

    #[test]
    fn approved_text_is_checked_against_the_goal_requirements() {
        let reqs = extract("write DOC.md in at least 10 lines");
        let (out, a) = propose_approved(
            SHORT,
            "base",
            &reqs,
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        let why = out.unwrap_err();
        assert!(
            why.contains("at least 10 non-empty lines, found 3"),
            "{why}"
        );
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].outcome, "refused");
        assert_eq!(
            a[0].unmet_requirements,
            ["at least 10 non-empty lines, found 3"]
        );
        // Met, or none stated: the approved bytes pass through unchanged.
        let (out, _) = propose_approved(
            SHORT,
            "base",
            &extract("at least 3 lines"),
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        assert_eq!(out.unwrap(), SHORT);
        let (out, a) = propose_approved(
            SHORT,
            "base",
            &[],
            COMPOSE_SKILL_BUDGET,
            COMPOSE_ATTEMPT_BUDGET,
        );
        assert_eq!(out.unwrap(), SHORT);
        assert_eq!(a[0].finish_reason.as_deref(), Some("approved"));
    }

    #[test]
    fn approved_run_reports_recognized_and_refuses_unmet_before_any_write() {
        let _home = crate::home_guard::home_slot();
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let proposer: ComposeProposer = Arc::new(|_: &str, _: std::time::Duration| {
            panic!("the model proposer ran on the approved path")
        });
        let bridge = ComposeBridge::new(tmp.path().join("compose"), proposer, "test:approved");
        let r = bridge.run_approved_task(
            "write DOC.md in at least 10 lines",
            ws.to_str().unwrap(),
            SHORT,
        );
        if !aien_omega_compose::LINKED {
            assert!(r.is_err());
            return;
        }
        let r = r.unwrap();
        assert!(!r.committed, "{r:?}");
        assert!(r.proposal.is_none());
        assert_eq!(r.requirements_recognized, ["at least 10 non-empty lines"]);
        assert!(r.proposer_error.unwrap().contains("found 3"));
        assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
        // The same bytes pass when the goal states a requirement they meet.
        let r = bridge
            .run_approved_task(
                "write DOC.md in at least 3 lines",
                ws.to_str().unwrap(),
                SHORT,
            )
            .unwrap();
        assert!(r.committed, "{r:?}");
        assert_eq!(r.requirements_recognized, ["at least 3 non-empty lines"]);
        assert_eq!(r.proposal.as_deref(), Some(SHORT));
    }
}

#[cfg(test)]
mod verify_boundary_tests {
    use super::*;
    use crate::requirements::analyze;

    const DOC: &str = "filename: DOC.md\na\nb\nc\n";

    fn recorded(text: &str) -> SkillOutput {
        (Ok(text.to_string()), Vec::new())
    }

    #[test]
    fn verify_accepts_exactly_the_recorded_bytes_that_meet_the_goal() {
        let ex = analyze("write DOC.md in at least 3 lines");
        let out = recorded(DOC);
        let h = proposal_handle(DOC);
        assert!(verify_task_result(Some(&ex), Some(&out), h));
        // a result naming other bytes is refused
        assert!(!verify_task_result(Some(&ex), Some(&out), h ^ 1));
        // bytes that no longer meet the requirement are refused
        let short = analyze("write DOC.md in at least 4 lines");
        assert!(!verify_task_result(Some(&short), Some(&out), h));
        // a skill that returned no proposal is refused
        let none: SkillOutput = (Err("no".into()), Vec::new());
        assert!(!verify_task_result(Some(&ex), Some(&none), h));
        assert!(!verify_task_result(Some(&ex), None, h));
    }

    /// Issue #289: an absent requirement record used to become an empty list,
    /// so the verify step approved any well-formed file.
    #[test]
    fn missing_requirement_record_is_refused_never_empty() {
        let out = recorded(DOC);
        let h = proposal_handle(DOC);
        // the same bytes that pass with a record are refused without one
        let ex = analyze("write DOC.md in at least 3 lines");
        assert!(verify_task_result(Some(&ex), Some(&out), h));
        assert!(!verify_task_result(None, Some(&out), h));
        // and even a goal with no requirements needs its (empty) record
        let empty = analyze("write DOC.md");
        assert!(verify_task_result(Some(&empty), Some(&out), h));
        assert!(!verify_task_result(None, Some(&out), h));
    }

    #[test]
    fn uncertain_goal_is_refused_at_verify_even_when_bytes_look_fine() {
        let ex = analyze("write DOC.md with at least 3 lines of context");
        assert!(!ex.uncertain.is_empty());
        assert!(!verify_task_result(
            Some(&ex),
            Some(&recorded(DOC)),
            proposal_handle(DOC)
        ));
    }

    #[test]
    fn refused_without_model_records_one_refused_attempt() {
        let (out, a) = refused_without_model(MISSING_RECORD_REASON.into());
        assert_eq!(out.unwrap_err(), MISSING_RECORD_REASON);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].outcome, "refused");
        assert!(a[0].text.is_none());
    }
}

#[cfg(test)]
mod verify_callback_integration_tests {
    use super::*;

    fn bridge(dir: &std::path::Path) -> ComposeBridge {
        let proposer: ComposeProposer = Arc::new(|_: &str, _: std::time::Duration| {
            Ok(Generation {
                text: "filename: DOC.md\na\nb\nc\n".into(),
                finish_reason: Some("eos".into()),
                ..Default::default()
            })
        });
        ComposeBridge::new(dir.join("compose"), proposer, "test:verify-callback")
    }

    /// Issue #289 through the real compose run and the real verify callback:
    /// the same document commits with its requirement record and is refused
    /// by AEGIS (nothing committed, no promotion) when the record is gone.
    #[test]
    fn lost_requirement_record_is_refused_by_the_real_verify_callback() {
        let _home = crate::home_guard::home_slot();
        if !aien_omega_compose::LINKED {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let b = bridge(tmp.path());
        let ok = b.run_task("write DOC.md in at least 3 lines", ws.to_str().unwrap());
        let ControlResponse::ComposeTaskResult(ok) = ok else {
            panic!("{ok:?}")
        };
        assert!(ok.committed, "{ok:?}");
        let goal = format!("write DOC2.md in at least 3 lines {TEST_DROP_RECORD_MARKER}");
        let lost = b.run_task(&goal, ws.to_str().unwrap());
        let ControlResponse::ComposeTaskResult(lost) = lost else {
            panic!("{lost:?}")
        };
        assert!(!lost.committed, "{lost:?}");
        assert_eq!(lost.aegis_pass_mask & 1, 0, "{lost:?}");
        assert_eq!(lost.cx_promotion, 0, "{lost:?}");
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

#[cfg(test)]
mod destination_enforcement_tests {
    use super::*;

    #[test]
    fn reply_must_write_the_named_destination() {
        let ok = "filename: docs/SUMMARY.md\n# s\n".to_string();
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().to_path_buf();
        let run = |d: Option<&str>, t: &str| {
            enforce_destination((Ok(t.to_string()), vec![]), d, &ws, false).0
        };
        assert_eq!(run(Some("docs/SUMMARY.md"), &ok), Ok(ok.clone()));
        let e = run(Some("README.md"), &ok).unwrap_err();
        assert!(
            e.contains("docs/SUMMARY.md") && e.contains("README.md"),
            "{e}"
        );
        assert_eq!(run(None, &ok), Ok(ok));
    }
}

#[cfg(test)]
mod new_document_overwrite_tests {
    use super::*;

    #[test]
    fn new_kind_proposal_never_replaces_an_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        std::fs::write(ws.join("README.md"), "# r\n").unwrap();
        std::os::unix::fs::symlink("nowhere", ws.join("dangling.md")).unwrap();
        let run = |dest: Option<&str>, path: &str, edit: bool| {
            let t = format!("filename: {path}\nbody\n");
            enforce_destination((Ok(t), vec![]), dest, ws, edit).0
        };
        for dest in [None, Some("README.md")] {
            let e = run(dest, "README.md", false).unwrap_err();
            assert!(e.contains("already exists"), "{e}");
        }
        assert!(run(None, "dangling.md", false).is_err());
        assert!(run(None, "fresh.md", false).is_ok());
        // An edit-kind task may write its own (existing) target.
        assert!(run(Some("README.md"), "README.md", true).is_ok());
    }
}
