//! PREFILL-GATE end to end: a launched swarm must run the root prompt through
//! the model before any branch decodes.
//!
//! Uses only APIs that already exist on main (cfd9982), so this file also runs
//! against main, where it must FAIL (see scripts/prefill-gate/fails-on-main.sh):
//! on main SwarmManager::launch_swarm forks branches from allocated-but-zeroed
//! root blocks (crates/aien-runtime/src/swarm.rs:95-111 at cfd9982) and the
//! scheduler decodes them with no prefill at all.
//!
//! The stock MockInferenceBackend ignores prompt content, so this test uses a
//! recording backend that logs every prefill span and every decode, in order.
//!
//! PREFILL-E2E C4 (bullets 6, 7, 8) extends it: the recording backend also
//! logs the backend fork hook (`fork_sequence`) together with the parent's
//! KV prefill state at that moment, and refuses a fork whose parent is not
//! ready. The swarm test asserts the order prefill fence -> fork_sequence per
//! branch -> first decode. Further tests drive the real
//! NativeTransformerBackend (reference weights, CPU) through spine +
//! scheduler + one pooled KV: a fork before the fence is refused, a decode
//! of an unknown id is an error, and each branch's first decode logits equal
//! the root's own continuation logits (abs 1e-5).
//!
//! Mutant (outside the repo, ~/workspace/hive/PE2E-C4/mutant.patch): the
//! spine no longer calls `fork_sequence` and `forward_decode_batch` again
//! creates an empty state (`or_insert_with`) for an unknown id. Then no Fork
//! events are recorded, the unknown-id decode is accepted, and branches decode
//! from token 1 at position 0, so the tests below fail.
//!
//! PREFILL-E2E C6 (bullet 11, "no branch can observe another branch's
//! suffix"): branches now decode with their own sampling params (handed over
//! by `fork_sequence_with_sampling`) and a per-request seed, so they diverge.
//! `sampled_branches_diverge_and_each_matches_its_own_teacher_forced_control`
//! checks that two or more branches produce different tokens, and that every
//! decode step of each branch has the same logits (abs 1e-5) as a fresh
//! unpaged backend fed the root prompt + root token + that branch's own
//! generated tokens. Mutants (outside the repo): ~/workspace/hive/PE2E-C6/
//! mutant.patch makes each decode row write its K/V into its sibling's slot
//! (parity fails: BRANCH_ISOLATION_VIOLATION); mutant_a.patch seeds every
//! branch identically (divergence fails: BRANCH_DIVERGENCE_VIOLATION).

use std::collections::HashMap;

use std::sync::Arc;

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, ModelConfig, NativeTransformerBackend, ReferenceCpuBackend,
    SamplingParams, ScheduledBatch, SequenceRequest, StepMetrics, TransformerWeights,
};
use aien_kv_cache::{create_shared_kv_manager, SharedKvManager};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::swarm::SwarmConfig;
use aien_scheduler::SchedulerConfig;
use async_trait::async_trait;

#[derive(Debug, Clone)]
enum Event {
    Prefill {
        id: u64,
        tokens: Vec<u32>,
    },
    Decode {
        id: u64,
    },
    Fork {
        parent: u64,
        child: u64,
        parent_ready: bool,
    },
}

#[derive(Default)]
struct RecordingBackend {
    events: Vec<Event>,
    /// The spine's KV manager, read in `fork_sequence` to record (and gate on)
    /// whether the parent passed the prefill completion fence.
    kv: Option<SharedKvManager>,
}

#[async_trait]
impl AienInferenceBackend for RecordingBackend {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut outputs = Vec::new();
        let mut prefill_tokens = 0;
        for req in &batch.prefill_requests {
            prefill_tokens += req.prompt_tokens.len();
            self.events.push(Event::Prefill {
                id: req.request_id,
                tokens: req.prompt_tokens.clone(),
            });
            outputs.push(DecodeOutput::Token {
                request_id: req.request_id,
                token_id: 100,
                logprob: None,
            });
        }
        for &id in &batch.decode_requests {
            self.events.push(Event::Decode { id });
            outputs.push(DecodeOutput::Token {
                request_id: id,
                token_id: 101,
                logprob: None,
            });
        }
        let metrics = StepMetrics {
            prefill_tokens_processed: prefill_tokens,
            decode_tokens_emitted: batch.decode_requests.len() + batch.prefill_requests.len(),
            step_latency_us: 0,
            active_kv_blocks: batch.block_tables.values().map(|v| v.len()).sum(),
        };
        Ok((outputs, metrics))
    }

    fn fork_sequence(&mut self, parent_id: u64, child_id: u64) -> Result<(), String> {
        let parent_ready = self
            .kv
            .as_ref()
            .and_then(|kv| kv.read().prefill_state(parent_id))
            .map(|s| s.is_ready())
            .unwrap_or(false);
        self.events.push(Event::Fork {
            parent: parent_id,
            child: child_id,
            parent_ready,
        });
        if parent_ready {
            Ok(())
        } else {
            Err(format!(
                "recording backend: fork {} -> {} refused, parent not PrefillReady",
                parent_id, child_id
            ))
        }
    }
}

#[tokio::test]
async fn prefill_gate_swarm_branches_never_decode_before_root_prompt_prefill() {
    let kv_manager = create_shared_kv_manager(256, 16);
    let sched_cfg = SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 16,
        chunk_prefill: true,
        watermark_blocks: 4,
    };
    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let mut backend = RecordingBackend {
        kv: Some(kv_manager.clone()),
        ..RecordingBackend::default()
    };

    let prompt: Vec<u32> = (500..540).collect();
    let branch_count = 4usize;
    let config = SwarmConfig {
        model_handle: 1,
        branch_count,
        max_active_sequences: branch_count,
        max_tokens_per_branch: 3,
        root_world_id: 0,
        priority: 1,
    };
    let swarm_id = spine.launch_swarm(config, &prompt).expect("launch");
    let swarm = spine
        .swarm_manager
        .get_swarm(swarm_id)
        .expect("swarm record")
        .clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();

    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0)
        && steps < 100
    {
        steps += 1;
        spine.step(&mut backend).await.expect("step");
    }

    // Replay the backend log in order. Every branch decode must come after
    // the full prompt was run through the model, for the shared root or for
    // that branch itself.
    let mut prefilled: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut branch_decodes = 0usize;
    let covers_prompt = |seen: Option<&Vec<u32>>| {
        seen.map(|s| s.len() >= prompt.len() && s[..prompt.len()] == prompt[..])
            .unwrap_or(false)
    };
    for event in &backend.events {
        match event {
            Event::Prefill { id, tokens } => {
                prefilled.entry(*id).or_default().extend_from_slice(tokens);
            }
            Event::Decode { id } => {
                if branches.contains(id) {
                    branch_decodes += 1;
                    assert!(
                        covers_prompt(prefilled.get(&root)) || covers_prompt(prefilled.get(id)),
                        "PREFILL_GATE_VIOLATION: branch {} decoded before its prompt was prefilled \
                         (root prefilled {:?} tokens, branch prefilled {:?} tokens, prompt {} tokens)",
                        id,
                        prefilled.get(&root).map(|v| v.len()),
                        prefilled.get(id).map(|v| v.len()),
                        prompt.len()
                    );
                }
            }
            Event::Fork { .. } => {}
        }
    }

    // PREFILL-E2E C4 order: prefill fence -> fork_sequence per branch -> first decode.
    let last_root_prefill = backend
        .events
        .iter()
        .rposition(|e| matches!(e, Event::Prefill { id, .. } if *id == root))
        .expect("root prompt must be prefilled");
    for &b in &branches {
        let forks: Vec<usize> = backend
            .events
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, Event::Fork { child, .. } if *child == b))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            forks.len(),
            1,
            "BACKEND_FORK_VIOLATION: branch {} must be forked in the backend exactly once, got {}",
            b,
            forks.len()
        );
        let fork_at = forks[0];
        match &backend.events[fork_at] {
            Event::Fork {
                parent,
                parent_ready,
                ..
            } => {
                assert_eq!(
                    *parent, root,
                    "branch {} forked from {} not root",
                    b, parent
                );
                assert!(
                    *parent_ready,
                    "BACKEND_FORK_VIOLATION: branch {} forked before the root prefill fence",
                    b
                );
            }
            _ => unreachable!(),
        }
        assert!(
            fork_at > last_root_prefill,
            "BACKEND_FORK_VIOLATION: branch {} forked (event {}) before the last root prefill chunk (event {})",
            b,
            fork_at,
            last_root_prefill
        );
        let first_decode = backend
            .events
            .iter()
            .position(|e| matches!(e, Event::Decode { id } if *id == b))
            .expect("every branch decodes");
        assert!(
            fork_at < first_decode,
            "BACKEND_FORK_VIOLATION: branch {} decoded (event {}) before its backend fork (event {})",
            b,
            first_decode,
            fork_at
        );
    }

    assert!(
        branch_decodes > 0,
        "branches must decode once the root prompt is prefilled"
    );
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        branch_count,
        "every branch must finish"
    );
}

// ---------------------------------------------------------------------------
// PREFILL-E2E C4: real NativeTransformerBackend (reference weights, CPU)
// through spine + scheduler + one pooled KV.
// ---------------------------------------------------------------------------

const PARITY_TOL: f32 = 1e-5;

fn c4_config() -> ModelConfig {
    ModelConfig {
        model_id: "pe2e-c4-fork-hooks-reference".to_string(),
        max_sequence_length: 256,
        // 6-token prompt over 4-token blocks: the root's tail block is partial
        // and shared, so each branch's first decode copies it (COW).
        block_size: 4,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        num_kv_heads: 2,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 97,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

fn c4_scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 2,
    }
}

const C4_PROMPT: [u32; 6] = [5, 17, 42, 3, 88, 61];

/// Delegates to the native backend and records each sequence's decode logits.
struct LogitsTap {
    inner: NativeTransformerBackend,
    decode_logits: HashMap<u64, Vec<Vec<f32>>>,
    /// Token each decode step sampled, per sequence (C6).
    decode_tokens: HashMap<u64, Vec<u32>>,
    /// Per decode batch: (id, last token fed) for every row (C6).
    decode_batches: Vec<Vec<(u64, u32)>>,
    /// Each sequence's backend token stream as it was just before its first decode.
    first_decode_tokens: HashMap<u64, Vec<u32>>,
}

#[async_trait]
impl AienInferenceBackend for LogitsTap {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let mut outputs = Vec::new();
        let mut metrics = StepMetrics::default();
        if !batch.prefill_requests.is_empty() {
            let prefill_only = ScheduledBatch {
                decode_requests: Vec::new(),
                ..batch.clone()
            };
            let (o, m) = self.inner.execute_step(&prefill_only).await?;
            outputs.extend(o);
            metrics = m;
        }
        if !batch.decode_requests.is_empty() {
            let fed: Vec<(u64, u32)> = batch
                .decode_requests
                .iter()
                .map(|id| {
                    let last = self
                        .inner
                        .sequences
                        .get(id)
                        .and_then(|s| s.tokens.last().copied())
                        .unwrap_or(u32::MAX);
                    (*id, last)
                })
                .collect();
            self.decode_batches.push(fed);
            for id in &batch.decode_requests {
                if let Some(seq) = self.inner.sequences.get(id) {
                    self.first_decode_tokens
                        .entry(*id)
                        .or_insert_with(|| seq.tokens.clone());
                }
            }
            let (o, logits) = self
                .inner
                .forward_decode_batch_with_logits(&batch.decode_requests)?;
            for (id, row) in batch.decode_requests.iter().zip(logits) {
                self.decode_logits.entry(*id).or_default().push(row);
                if let Some(tok) = self.inner.sequences.get(id).and_then(|s| s.tokens.last()) {
                    self.decode_tokens.entry(*id).or_default().push(*tok);
                }
            }
            outputs.extend(o);
            metrics.decode_tokens_emitted += batch.decode_requests.len();
        }
        Ok((outputs, metrics))
    }

    fn manages_kv_cache(&self) -> bool {
        self.inner.manages_kv_cache()
    }

    fn fork_sequence(&mut self, parent_id: u64, child_id: u64) -> Result<(), String> {
        self.inner.fork_sequence(parent_id, child_id)
    }

    fn fork_sequence_with_sampling(
        &mut self,
        parent_id: u64,
        child_id: u64,
        sampling: &SamplingParams,
    ) -> Result<(), String> {
        self.inner
            .fork_sequence_with_sampling(parent_id, child_id, sampling)
    }
}

fn c4_swarm_config(branch_count: usize, max_tokens: usize) -> SwarmConfig {
    SwarmConfig {
        model_handle: 1,
        branch_count,
        max_active_sequences: branch_count,
        max_tokens_per_branch: max_tokens,
        root_world_id: 0,
        priority: 1,
    }
}

#[tokio::test]
async fn backend_fork_before_prefill_fence_is_refused() {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, mut backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");

    let swarm_id = spine
        .launch_swarm(c4_swarm_config(2, 2), &C4_PROMPT)
        .expect("launch swarm");
    let swarm = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();

    // Run the model over the root prompt WITHOUT the completion fence: the
    // backend now knows the root, but the KV manager has not marked it ready.
    let mut block_tables = HashMap::new();
    block_tables.insert(
        root,
        spine
            .kv_manager
            .read()
            .get_block_table(root)
            .unwrap()
            .block_ids
            .clone(),
    );
    let unfenced = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: root,
            prompt_tokens: C4_PROMPT.to_vec(),
            sampling_params: SamplingParams::default(),
            arrival_time_ns: 0,
            priority: 1,
        }],
        decode_requests: Vec::new(),
        block_tables,
        step_id: 1,
    };
    backend
        .execute_step(&unfenced)
        .await
        .expect("unfenced prefill");
    assert!(backend.sequences.contains_key(&root));
    let state = spine.kv_manager.read().prefill_state(root).unwrap();
    assert!(
        !state.is_ready(),
        "root must not be ready without the fence"
    );

    let err = backend
        .fork_sequence(root, branches[0])
        .expect_err("BACKEND_FORK_VIOLATION: fork before the prefill fence must be refused");
    assert!(
        err.contains("not ready"),
        "unexpected refusal reason: {err}"
    );
    assert!(
        !backend.sequences.contains_key(&branches[0]),
        "a refused fork must not create backend state"
    );

    // Parent unknown to the KV manager is refused too.
    assert!(backend.fork_sequence(9_999_999, 9_999_998).is_err());
}

#[tokio::test]
async fn backend_decode_of_unknown_request_id_is_an_error() {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (_spine, mut backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");

    let unknown = 4_242_424u64;
    let direct = backend.forward_decode_batch(&[unknown]);
    assert!(
        direct.is_err(),
        "DECODE_STATE_VIOLATION: decoding an unknown request id must be an error, got {:?}",
        direct.ok()
    );
    assert!(
        !backend.sequences.contains_key(&unknown),
        "DECODE_STATE_VIOLATION: a refused decode must not create an empty sequence state"
    );

    let batch = ScheduledBatch {
        prefill_requests: Vec::new(),
        decode_requests: vec![unknown],
        block_tables: HashMap::new(),
        step_id: 1,
    };
    assert!(
        backend.execute_step(&batch).await.is_err(),
        "DECODE_STATE_VIOLATION: execute_step must refuse a decode of an unknown request id"
    );
    assert!(!backend.sequences.contains_key(&unknown));
}

#[tokio::test]
async fn branch_first_decode_logits_equal_root_continuation_logits() {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights.clone(),
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");
    let mut tap = LogitsTap {
        inner: backend,
        decode_logits: HashMap::new(),
        decode_tokens: HashMap::new(),
        decode_batches: Vec::new(),
        first_decode_tokens: HashMap::new(),
    };

    let branch_count = 3usize;
    let swarm_id = spine
        .launch_swarm(c4_swarm_config(branch_count, 2), &C4_PROMPT)
        .expect("launch swarm");
    let swarm = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();

    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0) && steps < 50
    {
        steps += 1;
        spine.step(&mut tap).await.expect("spine step");
    }
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        branch_count,
        "every branch must finish"
    );

    // Root's own continuation, computed independently: a fresh backend with
    // its own pooled KV prefills the same prompt under the same request id
    // (prefill sampling is seeded by request id) and decodes one step.
    let mut control = NativeTransformerBackend::with_paged_kv(weights, 64, config.block_size)
        .expect("control backend");
    let control_batch = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: root,
            prompt_tokens: C4_PROMPT.to_vec(),
            sampling_params: SamplingParams::default(),
            arrival_time_ns: 0,
            priority: 1,
        }],
        decode_requests: Vec::new(),
        block_tables: HashMap::new(),
        step_id: 1,
    };
    control
        .execute_step(&control_batch)
        .await
        .expect("control prefill");
    assert_eq!(
        control.sequences.get(&root).map(|s| s.tokens.clone()),
        tap.inner.sequences.get(&root).map(|s| s.tokens.clone()),
        "control root must hold the same tokens as the spine root after prefill"
    );
    let (_out, control_logits) = control
        .forward_decode_batch_with_logits(&[root])
        .expect("control root continuation");
    let root_logits = &control_logits[0];
    assert!(root_logits.iter().any(|x| *x != 0.0));

    for &b in &branches {
        let first = tap
            .decode_logits
            .get(&b)
            .and_then(|v| v.first())
            .unwrap_or_else(|| panic!("branch {} never decoded", b));
        assert_eq!(first.len(), root_logits.len());
        let max_diff = first
            .iter()
            .zip(root_logits.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= PARITY_TOL,
            "BACKEND_FORK_VIOLATION: branch {} step-1 logits differ from the root's own \
             continuation by {} (> {})",
            b,
            max_diff,
            PARITY_TOL
        );
    }
}

/// C4c: the token prefill sampled but did not append (`pending_prefill_token`)
/// belongs to the root's stream. `fork_sequence` must give each child that token
/// appended to its own copy of the parent's tokens WITHOUT mutating the parent, so every
/// branch's first decode feeds that token at
/// position `prompt_len`, exactly like the root's own continuation.
#[tokio::test]
async fn backend_fork_carries_parent_pending_prefill_token() {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");
    let mut tap = LogitsTap {
        inner: backend,
        decode_logits: HashMap::new(),
        decode_tokens: HashMap::new(),
        decode_batches: Vec::new(),
        first_decode_tokens: HashMap::new(),
    };

    let branch_count = 3usize;
    let swarm_id = spine
        .launch_swarm(c4_swarm_config(branch_count, 2), &C4_PROMPT)
        .expect("launch swarm");
    let swarm = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();

    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0) && steps < 50
    {
        steps += 1;
        spine.step(&mut tap).await.expect("spine step");
    }

    let root_tokens = tap
        .inner
        .sequences
        .get(&root)
        .expect("root state")
        .tokens
        .clone();
    assert_eq!(
        root_tokens,
        C4_PROMPT.to_vec(),
        "fork must not mutate the parent: root tokens stay the prompt only"
    );
    let pending = *tap
        .inner
        .pending_prefill_token
        .get(&root)
        .expect("fork must leave the parent's pending prefill token untouched");
    let mut expected_child = root_tokens.clone();
    expected_child.push(pending);
    for &b in &branches {
        let first = tap
            .first_decode_tokens
            .get(&b)
            .unwrap_or_else(|| panic!("branch {} never decoded", b));
        assert_eq!(
            first, &expected_child,
            "PENDING_TOKEN_VIOLATION: branch {} must start its first decode from the root's \
             prompt plus the sampled token",
            b
        );
    }
}

/// PREFILL-E2E C6, bullet 11: sampled branches diverge, and no branch sees
/// another branch's suffix. Reference weights, CPU, spine + scheduler + one
/// pooled KV (`build_shared_kv_runtime`).
#[tokio::test]
async fn sampled_branches_diverge_and_each_matches_its_own_teacher_forced_control() {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights.clone(),
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");
    let mut tap = LogitsTap {
        inner: backend,
        decode_logits: HashMap::new(),
        decode_tokens: HashMap::new(),
        decode_batches: Vec::new(),
        first_decode_tokens: HashMap::new(),
    };

    let branch_count = 3usize;
    let max_tokens = 8usize;
    let swarm_id = spine
        .launch_swarm(c4_swarm_config(branch_count, max_tokens), &C4_PROMPT)
        .expect("launch swarm");
    let swarm = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let root = swarm.root_sequence_id.as_u64();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();

    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0)
        && steps < 100
    {
        steps += 1;
        spine.step(&mut tap).await.expect("spine step");
    }
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        branch_count,
        "every branch must finish"
    );

    // The fork hook carried each branch's own sampling params.
    for &b in &branches {
        let params = tap.inner.sampling_params(b).unwrap_or_else(|| {
            panic!(
                "BRANCH_SAMPLING_VIOLATION: branch {} has no sampling params (fork hook did not carry them)",
                b
            )
        });
        assert!(
            params.temperature > 0.001,
            "branch {} must sample (temperature {})",
            b,
            params.temperature
        );
    }

    let generated: Vec<Vec<u32>> = branches
        .iter()
        .map(|b| tap.decode_tokens.get(b).cloned().unwrap_or_default())
        .collect();
    for (b, g) in branches.iter().zip(&generated) {
        assert!(!g.is_empty(), "branch {} never decoded", b);
    }

    // 1. Divergence: with temperature > 0 and per-branch seeds the branches
    //    must not all generate the same tokens.
    assert!(
        generated.iter().any(|g| *g != generated[0]),
        "BRANCH_DIVERGENCE_VIOLATION: all {} branches generated identical tokens {:?}",
        branch_count,
        generated
    );

    // Precondition for 2: some decode batch held two or more branches fed
    // different tokens, so a cross-branch K/V mixup would change logits.
    assert!(
        tap.decode_batches.iter().any(|rows| {
            let fed: Vec<u32> = rows
                .iter()
                .filter(|(id, _)| branches.contains(id))
                .map(|(_, t)| *t)
                .collect();
            fed.len() >= 2 && fed.iter().any(|t| *t != fed[0])
        }),
        "C6 precondition: no decode batch held two branches with different inputs ({:?})",
        tap.decode_batches
    );

    // 2. Isolation: each branch's per-step logits equal a teacher-forced,
    //    independent control (fresh unpaged backend, dense private K/V) fed
    //    the root prompt, the root's sampled token and only that branch's own
    //    generated tokens.
    let root_tokens = tap
        .inner
        .sequences
        .get(&root)
        .expect("root backend state")
        .tokens
        .clone();
    // C4 semantics: the root keeps prompt-only tokens; the sampled prefill token
    // stays pending and each fork child carries it.
    assert_eq!(root_tokens, C4_PROMPT.to_vec());
    let root_token = *tap
        .inner
        .pending_prefill_token
        .get(&root)
        .expect("root pending prefill token");

    for (b, gen) in branches.iter().zip(&generated) {
        let branch_logits = tap.decode_logits.get(b).expect("branch logits");
        assert_eq!(branch_logits.len(), gen.len());

        let ctl_id = 1u64;
        let mut control = NativeTransformerBackend::new(weights.clone());
        assert!(control.kv_manager.is_none(), "control must be unpaged");
        control
            .prefill_sequence(ctl_id, &C4_PROMPT)
            .expect("control prefill");
        control
            .sequences
            .get_mut(&ctl_id)
            .unwrap()
            .tokens
            .push(root_token);

        for (k, row) in branch_logits.iter().enumerate() {
            let (_out, ctl_logits) = control
                .forward_decode_batch_with_logits(&[ctl_id])
                .expect("control decode");
            let ctl_row = &ctl_logits[0];
            assert_eq!(row.len(), ctl_row.len());
            assert!(ctl_row.iter().any(|x| *x != 0.0));
            let max_diff = row
                .iter()
                .zip(ctl_row.iter())
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= PARITY_TOL,
                "BRANCH_ISOLATION_VIOLATION: branch {} step {} logits differ from its own \
                 teacher-forced control by {} (> {})",
                b,
                k + 1,
                max_diff,
                PARITY_TOL
            );
            // Teacher forcing: replace the control's own pick with the token
            // this branch actually generated at this step.
            let seq = control.sequences.get_mut(&ctl_id).unwrap();
            seq.tokens.pop();
            seq.tokens.push(gen[k]);
        }
    }
}

/// One branch of a sampled swarm run: id, sampling params, backend tokens just
/// before its first decode, per-step decode logits, per-step sampled tokens.
type BranchRun = (u64, SamplingParams, Vec<u32>, Vec<Vec<f32>>, Vec<u32>);

/// Runs the C6 sampled swarm (reference weights, CPU, spine + scheduler + one
/// pooled KV) in a fresh runtime to completion.
async fn sampled_swarm_run(branch_count: usize, max_tokens: usize) -> Vec<BranchRun> {
    let config = c4_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let (mut spine, backend) = build_shared_kv_runtime(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        c4_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");
    let mut tap = LogitsTap {
        inner: backend,
        decode_logits: HashMap::new(),
        decode_tokens: HashMap::new(),
        decode_batches: Vec::new(),
        first_decode_tokens: HashMap::new(),
    };
    let swarm_id = spine
        .launch_swarm(c4_swarm_config(branch_count, max_tokens), &C4_PROMPT)
        .expect("launch swarm");
    let swarm = spine.swarm_manager.get_swarm(swarm_id).unwrap().clone();
    let branches: Vec<u64> = swarm.branch_sequences.iter().map(|s| s.as_u64()).collect();
    let mut steps = 0;
    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0)
        && steps < 100
    {
        steps += 1;
        spine.step(&mut tap).await.expect("spine step");
    }
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        branch_count,
        "every branch must finish"
    );
    branches
        .iter()
        .map(|&b| {
            (
                b,
                tap.inner
                    .sampling_params(b)
                    .cloned()
                    .expect("branch sampling params"),
                tap.first_decode_tokens
                    .get(&b)
                    .cloned()
                    .expect("branch decoded"),
                tap.decode_logits.get(&b).cloned().unwrap_or_default(),
                tap.decode_tokens.get(&b).cloned().unwrap_or_default(),
            )
        })
        .collect()
}

/// Branch-specific sampling follows the reproducibility contract in
/// crates/aien-inference-abi/src/transformer_backend.rs `decode_sampling_seed`
/// (PREFILL-E2E C6): each decode token is `sample_with_params(logits, the
/// branch's own params, decode_sampling_seed(request id, token count before the
/// new token))`; sibling branches draw from different seed streams; and "the
/// same request replays the same stream": the same swarm in a fresh runtime
/// gives the same ids, logits and tokens bit for bit. Marker SEED_CONTRACT_VIOLATION.
#[tokio::test]
async fn sampled_branches_follow_the_seed_contract_and_replay_bit_identically() {
    let first = sampled_swarm_run(3, 8).await;
    for (b, params, before, logits, gen) in &first {
        assert!(params.temperature > 0.001, "branch {} must sample", b);
        assert!(!gen.is_empty(), "branch {} never decoded", b);
        assert_eq!(logits.len(), gen.len());
        for (k, row) in logits.iter().enumerate() {
            let seed = aien_inference_abi::decode_sampling_seed(*b, before.len() + k);
            let (tok, _) = aien_inference_abi::sample_with_params(row, params, seed);
            assert_eq!(
                tok, gen[k],
                "SEED_CONTRACT_VIOLATION: branch {} step {} sampled {} but the contract seed gives {}",
                b, k + 1, gen[k], tok
            );
        }
    }
    // sibling branches use different seed streams at every position they share
    for i in 0..first.len() {
        for j in (i + 1)..first.len() {
            let (bi, _, pi, _, gi) = &first[i];
            let (bj, _, _, _, _) = &first[j];
            assert_ne!(bi, bj);
            for pos in pi.len()..pi.len() + gi.len() {
                assert_ne!(
                    aien_inference_abi::decode_sampling_seed(*bi, pos),
                    aien_inference_abi::decode_sampling_seed(*bj, pos),
                    "SEED_CONTRACT_VIOLATION: branches {} and {} share the seed for position {}",
                    bi,
                    bj,
                    pos
                );
            }
        }
    }
    // replay in a fresh runtime: identical ids, logits (bitwise) and tokens
    let second = sampled_swarm_run(3, 8).await;
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            a.0, b.0,
            "SEED_CONTRACT_VIOLATION: replay assigned different branch ids"
        );
        assert_eq!(
            a.2, b.2,
            "SEED_CONTRACT_VIOLATION: branch {} context differs on replay",
            a.0
        );
        assert_eq!(
            a.4, b.4,
            "SEED_CONTRACT_VIOLATION: branch {} tokens differ on replay",
            a.0
        );
        let bits = |rows: &Vec<Vec<f32>>| -> Vec<u32> {
            rows.iter().flatten().map(|x| x.to_bits()).collect()
        };
        assert_eq!(
            bits(&a.3),
            bits(&b.3),
            "SEED_CONTRACT_VIOLATION: branch {} logits differ on replay",
            a.0
        );
    }
}
