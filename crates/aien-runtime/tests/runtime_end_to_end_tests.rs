//! End-to-end integration tests: Scheduler ticket submission -> Omega GB10 transformer -> CompletionSink channel.
//! Verifies zero socket, zero HTTP, and zero RPC boundaries in the entire execution path.

use aien_inference_abi::native_ops::{NativeOpMask, OpReport, TensorOp};
use aien_inference_abi::{
    ModelConfig, NativeTransformerBackend, OmegaGb10Backend, ReferenceCpuBackend, SamplingParams,
    TensorBackend, TransformerWeights,
};
use aien_kv_cache::{create_shared_kv_manager, KvDType, KvPoolConfig};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{ChannelCompletionSink, CompletionEvent, PromptHandle, SchedulerConfig};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn checkpoint_identity() -> (String, Option<String>) {
    let explicit = std::env::var("AIEN_CHECKPOINT_ID")
        .ok()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty());
    let discovered = discover_safetensors_sha256();
    (
        explicit.unwrap_or_else(|| "unspecified".to_string()),
        discovered,
    )
}

fn discover_safetensors_sha256() -> Option<String> {
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("AIEN_MODEL_DIR") {
        dirs.push(std::path::PathBuf::from(dir));
    }
    dirs.push(std::path::PathBuf::from("models"));
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::PathBuf::from(home).join("models"));
    }
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file = if path.is_file()
                && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            {
                Some(path)
            } else if path.is_dir() && path.join("model.safetensors").is_file() {
                Some(path.join("model.safetensors"))
            } else {
                None
            };
            if let Some(file) = file {
                return sha256_file(&file).ok();
            }
        }
    }
    None
}

fn sha256_file(path: &std::path::Path) -> Result<String, std::io::Error> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn test_micro_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-micro-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        num_kv_heads: 2,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Micro model in a shape the native Omega engine accepts: attention needs head_dim 64 and a
/// power-of-two q/kv head ratio, rmsnorm needs dim % 128 == 0 (omega
/// omega_gpu_attention_api.c:603-604, omega_gpu_elementwise_api.c:527). The CPU micro shape
/// above (head_dim 16, hidden 64) is refused by the engine, which in a production build is a
/// strict violation, so the GB10 tests must not use it.
fn gb10_micro_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-micro-gb10-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 64,
        num_kv_heads: 2,
        hidden_dim: 256,
        intermediate_dim: 512,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Counts calls per op and delegates every op to the wrapped backend, so a test can show which
/// ops the runtime really drove (fallback counts alone cannot show that an op ran at all).
struct CountingBackend {
    inner: Arc<dyn TensorBackend>,
    calls: [AtomicU64; TensorOp::COUNT],
    /// Largest `num_seqs` seen in one `paged_attention_batch` call.
    max_batch_seqs: AtomicU64,
}

impl CountingBackend {
    fn new(inner: Arc<dyn TensorBackend>) -> Self {
        Self {
            inner,
            calls: Default::default(),
            max_batch_seqs: AtomicU64::new(0),
        }
    }

    fn hit(&self, op: TensorOp) {
        self.calls[TensorOp::ALL.iter().position(|o| *o == op).unwrap()]
            .fetch_add(1, Ordering::Relaxed);
    }

    fn calls(&self, op: TensorOp) -> u64 {
        self.calls[TensorOp::ALL.iter().position(|o| *o == op).unwrap()].load(Ordering::Relaxed)
    }

    fn line(&self) -> String {
        let parts: Vec<String> = TensorOp::ALL
            .into_iter()
            .map(|op| format!("{}:{}", op.name(), self.calls(op)))
            .collect();
        format!(
            "OP_CALLS {} max_batch_seqs={}",
            parts.join(","),
            self.max_batch_seqs.load(Ordering::Relaxed)
        )
    }
}

impl TensorBackend for CountingBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn fallback_count(&self) -> u64 {
        self.inner.fallback_count()
    }
    fn native_ops(&self) -> NativeOpMask {
        self.inner.native_ops()
    }
    fn op_report(&self) -> OpReport {
        self.inner.op_report()
    }
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.hit(TensorOp::Rmsnorm);
        self.inner.rmsnorm(out, x, weight, eps)
    }
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        theta: f32,
    ) {
        self.hit(TensorOp::ApplyRope);
        self.inner
            .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, theta)
    }
    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        self.hit(TensorOp::MatmulVec);
        self.inner.matmul_vec(out, x, weight, out_dim, in_dim)
    }
    fn matmul_batch(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        self.hit(TensorOp::MatmulBatch);
        self.inner
            .matmul_batch(out, x, weight, batch_size, in_dim, out_dim)
    }
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        self.hit(TensorOp::Swiglu);
        self.inner.swiglu(out, gate, up)
    }
    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::GqaAttention);
        self.inner.gqa_attention(
            out,
            q,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn paged_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_ids: &[usize],
        context_len: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::PagedAttention);
        self.inner.paged_attention(
            out,
            q,
            pool,
            block_ids,
            context_len,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn paged_attention_batch(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_tables: &[i32],
        context_lens: &[i32],
        max_blocks_per_seq: usize,
        num_seqs: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::PagedAttentionBatch);
        self.max_batch_seqs
            .fetch_max(num_seqs as u64, Ordering::Relaxed);
        self.inner.paged_attention_batch(
            out,
            q,
            pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_seqs,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        self.hit(TensorOp::ComputeLogits);
        self.inner
            .compute_logits(logits, hidden, embed_weight, vocab_size, hidden_dim)
    }
}

/// Several sequences decoding together through the runtime spine on a shared bf16 KV tensor
/// pool. Each decode step goes through `forward_decode_batch`, which calls
/// `paged_attention_batch` once per layer for all running sequences. Returns the tokens
/// emitted per sequence and checks every KV block is reclaimed.
async fn run_decode_batch(
    backend: Arc<dyn TensorBackend>,
    config: &ModelConfig,
    prompts: &[&[u32]],
    max_tokens: usize,
) -> Vec<usize> {
    let (total_blocks, block_size) = (64usize, 16usize);
    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };
    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();
    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };
    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(config);
    let mut runner =
        NativeTransformerBackend::with_shared_kv_and_backend(weights, backend, kv_manager.clone());
    let mut sinks = Vec::new();
    for p in prompts {
        let (sink, rx) = ChannelCompletionSink::channel();
        let sink_id = spine.register_completion_sink(Arc::new(sink));
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop_token_ids: vec![],
        };
        spine
            .submit_work(Arc::from(*p), sampling, 1, Some(sink_id))
            .unwrap();
        sinks.push(rx);
    }
    let _ = spine
        .run_until_complete(&mut runner, 4 * (max_tokens + 4))
        .await
        .unwrap();
    let mut emitted = Vec::new();
    for mut rx in sinks {
        let mut n = 0usize;
        while let Ok(event) = rx.try_recv() {
            if let CompletionEvent::Token { .. } = event {
                n += 1;
            }
        }
        emitted.push(n);
    }
    assert_eq!(
        kv_manager.read().allocated_block_count(),
        0,
        "KV blocks must be reclaimed after the batch completes"
    );
    emitted
}

#[tokio::test]
async fn test_end_to_end_ticket_submission_and_channel_streaming_cpu() {
    let config = test_micro_model_config();
    let total_blocks = 64;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());

    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        kv_manager,
    );

    // 1. Create in-memory channel sink with zero socket/RPC boundary
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));

    // 2. Submit immutable PromptHandle ticket
    let prompt: PromptHandle = Arc::from([1u32, 10, 25, 42].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 5,
        stop_token_ids: vec![0],
    };

    let seq_id = spine
        .submit_work(prompt, sampling, 1, Some(sink_id))
        .unwrap();

    // 3. Drive execution steps until completion
    let metrics = spine.run_until_complete(&mut backend, 20).await.unwrap();
    assert!(!metrics.is_empty(), "Must have executed at least 1 step");

    // 4. Collect streamed events from channel
    let mut received_tokens = Vec::new();
    let mut saw_finished = false;

    while let Ok(event) = rx.try_recv() {
        match event {
            CompletionEvent::Token { seq_id: sid, token } => {
                assert_eq!(sid.to_u64(), seq_id);
                received_tokens.push(token);
            }
            CompletionEvent::Finished {
                seq_id: sid,
                total_tokens,
                finish_reason,
            } => {
                assert_eq!(sid.to_u64(), seq_id);
                saw_finished = true;
                eprintln!(
                    "Finished event: total_tokens={}, received_tokens={}, reason={:?}",
                    total_tokens,
                    received_tokens.len(),
                    finish_reason
                );
            }
            CompletionEvent::Error {
                seq_id: sid,
                message,
            } => {
                panic!("Unexpected error for seq_id {}: {}", sid, message);
            }
        }
    }

    assert!(
        saw_finished,
        "Expected CompletionEvent::Finished on channel"
    );
    eprintln!("Received tokens: {:?}", received_tokens);
    assert!(
        received_tokens.len() >= 4,
        "Expected at least 4 generated tokens before stop token, got {}",
        received_tokens.len()
    );
}

#[tokio::test]
async fn test_end_to_end_subagent_branching_with_independent_sinks() {
    let config = test_micro_model_config();
    let total_blocks = 128;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        kv_manager.clone(),
    );

    // Parent ticket
    let (sink_parent, mut rx_parent) = ChannelCompletionSink::channel();
    let sink_p_id = spine.register_completion_sink(Arc::new(sink_parent));

    let prompt: PromptHandle = Arc::from([1u32, 15, 30, 45].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 6,
        stop_token_ids: vec![0],
    };

    let parent_id = spine
        .submit_work(prompt, sampling, 2, Some(sink_p_id))
        .unwrap();

    // Step twice to prefill and decode initial token
    let _ = spine.step(&mut backend).await.unwrap();
    let _ = spine.step(&mut backend).await.unwrap();

    // Fork subagent 1 with independent sink
    let (sink_child1, mut rx_child1) = ChannelCompletionSink::channel();
    let sink_c1_id = spine.register_completion_sink(Arc::new(sink_child1));
    let child1_id = parent_id + 100;
    spine
        .fork_subagent(parent_id, child1_id, Some(sink_c1_id))
        .unwrap();

    // Fork subagent 2 with independent sink
    let (sink_child2, mut rx_child2) = ChannelCompletionSink::channel();
    let sink_c2_id = spine.register_completion_sink(Arc::new(sink_child2));
    let child2_id = parent_id + 200;
    spine
        .fork_subagent(parent_id, child2_id, Some(sink_c2_id))
        .unwrap();

    // Verify prefix sharing in physical KV pool right after fork
    {
        let kv = kv_manager.read();
        let m = kv.metrics();
        assert!(
            m.shared_pages >= 1,
            "Parent and subagents must share physical prefix blocks without copy"
        );
    }

    // Run until parent and both subagents complete
    let _ = spine.run_until_complete(&mut backend, 30).await.unwrap();

    // Verify parent received its events
    let mut parent_tokens = Vec::new();
    while let Ok(event) = rx_parent.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            parent_tokens.push(token);
        }
    }
    assert!(
        !parent_tokens.is_empty(),
        "Parent must have streamed tokens"
    );

    // Verify child 1 received its events independently
    let mut child1_tokens = Vec::new();
    while let Ok(event) = rx_child1.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            child1_tokens.push(token);
        }
    }
    assert!(
        !child1_tokens.is_empty(),
        "Child 1 must have streamed tokens"
    );

    // Verify child 2 received its events independently
    let mut child2_tokens = Vec::new();
    while let Ok(event) = rx_child2.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            child2_tokens.push(token);
        }
    }
    assert!(
        !child2_tokens.is_empty(),
        "Child 2 must have streamed tokens"
    );

    eprintln!(
        "Running: {}, Waiting: {}, Preempted: {}, Allocated blocks: {}",
        spine.scheduler.running_count(),
        spine.scheduler.waiting_count(),
        spine.scheduler.preempted_count(),
        kv_manager.read().allocated_block_count()
    );
    {
        let kv = kv_manager.read();
        eprintln!(
            "Active sequences in KV: {}, metrics: {:?}, active blocks: {:?}",
            kv.active_sequence_count(),
            kv.metrics(),
            kv.debug_active_blocks()
        );
    }
    // Verify zero leaked physical pages after all sequences completed
    {
        let kv = kv_manager.read();
        assert_eq!(
            kv.allocated_block_count(),
            0,
            "All physical blocks must be reclaimed upon sequence completion"
        );
    }
}

#[tokio::test]
async fn test_end_to_end_gb10_hardware_execution_if_available() {
    let gpu_backend = Arc::new(OmegaGb10Backend::new());
    if !gpu_backend.is_available() {
        eprintln!("Skipping GB10 test: Omega GPU engine not linked");
        return;
    }

    let config = gb10_micro_model_config();
    let total_blocks = 64;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        gpu_backend.clone(),
        kv_manager,
    );

    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));

    let prompt: PromptHandle = Arc::from([5u32, 12, 33, 77].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 4,
        stop_token_ids: vec![0],
    };

    let seq_id = spine
        .submit_work(prompt, sampling, 1, Some(sink_id))
        .unwrap();

    let _ = spine.run_until_complete(&mut backend, 15).await.unwrap();

    let mut tokens = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let CompletionEvent::Token { seq_id: sid, token } = event {
            assert_eq!(sid.to_u64(), seq_id);
            tokens.push(token);
        }
    }

    assert_eq!(
        tokens.len(),
        4,
        "Expected 4 tokens generated on the GB10 GPU"
    );
    assert!(
        gpu_backend.chip_calls() > 0,
        "Omega GPU kernels must have executed"
    );
    assert_eq!(
        gpu_backend.fallback_count(),
        0,
        "Zero silent CPU fallback allowed on the GB10"
    );
    eprintln!(
        "Omega GB10 End-to-End Execution Certified: {} tokens emitted via GPU kernels (count = {})",
        tokens.len(),
        gpu_backend.chip_calls()
    );
}

/// Release gate for issue #94. Ordinary CI may skip the body. `AIEN_REQUIRE_RELEASE=1`
/// fails the run unless this process executed on GB10, emitted tokens, stayed at
/// zero fallback, and reclaimed KV blocks. The JSON line is the machine-readable artifact.
#[tokio::test]
async fn release_golden_path_records_whether_gb10_ran() {
    let require = std::env::var("AIEN_REQUIRE_RELEASE")
        .map(|value| value.trim() == "1" || value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let gpu_backend = Arc::new(OmegaGb10Backend::new());
    let commit = std::env::var("GITHUB_SHA").unwrap_or_else(|_| "local".into());
    let (checkpoint, discovered_weights) = checkpoint_identity();
    if !gpu_backend.is_available() {
        let artifact = serde_json::json!({
            "test": "release_golden_path",
            "skipped": true,
            "reason": "Omega GPU engine not linked (stub build)",
            "commit": commit,
            "checkpoint_id": checkpoint,
        "discovered_safetensors_sha256": discovered_weights,
            "require_release": require,
            "gpu_executions": gpu_backend.chip_calls(),
            "fallback_count": gpu_backend.fallback_count(),
        });
        eprintln!("{artifact}");
        assert!(
            !require,
            "AIEN_REQUIRE_RELEASE is set but the GB10 body did not run: {artifact}"
        );
        return;
    }

    let config = gb10_micro_model_config();
    let total_blocks = 64;
    let block_size = 16;
    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };
    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();
    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };
    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        gpu_backend.clone(),
        kv_manager.clone(),
    );
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));
    let prompt: PromptHandle = Arc::from([5u32, 12, 33, 77].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 4,
        stop_token_ids: vec![0],
    };
    let seq_id = spine
        .submit_work(prompt.clone(), sampling.clone(), 1, Some(sink_id))
        .unwrap();
    let _ = spine.step(&mut backend).await.unwrap();
    let _ = spine.step(&mut backend).await.unwrap();
    let (child_sink, mut child_rx) = ChannelCompletionSink::channel();
    let child_sink_id = spine.register_completion_sink(Arc::new(child_sink));
    let branch = spine.fork_subagent(seq_id, seq_id.wrapping_add(1), Some(child_sink_id));
    let branch_error = branch.as_ref().err().cloned();
    let shared_pages = kv_manager.read().metrics().shared_pages;
    let _ = spine.run_until_complete(&mut backend, 20).await.unwrap();
    let mut tokens = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            tokens.push(token);
        }
    }
    let mut child_tokens = Vec::new();
    while let Ok(event) = child_rx.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            child_tokens.push(token);
        }
    }
    let leaked = kv_manager.read().allocated_block_count();
    let artifact = serde_json::json!({
        "test": "release_golden_path",
        "skipped": false,
        "commit": commit,
        "checkpoint_id": checkpoint,
        "discovered_safetensors_sha256": discovered_weights,
        "model_id": config.model_id,
        "device": gpu_backend.name(),
        "gpu_executions": gpu_backend.chip_calls(),
        "fallback_count": gpu_backend.fallback_count(),
        "tokens": tokens.len(),
        "child_tokens": child_tokens.len(),
        "shared_kv_pages": shared_pages,
        "branch_accepted": branch.is_ok(),
        "branch_error": branch_error,
        "leaked_kv_blocks": leaked,
    });
    eprintln!("{artifact}");
    assert!(branch.is_ok(), "{artifact}");
    assert!(shared_pages >= 1, "{artifact}");
    assert!(!child_tokens.is_empty(), "{artifact}");
    assert!(gpu_backend.chip_calls() > 0, "{artifact}");
    assert_eq!(gpu_backend.fallback_count(), 0, "{artifact}");
    assert_eq!(tokens.len(), 4, "{artifact}");
    assert_eq!(leaked, 0, "{artifact}");
    if require {
        assert_ne!(
            checkpoint, "unspecified",
            "release run needs AIEN_CHECKPOINT_ID: {artifact}"
        );
    }
}

const BATCH_PROMPTS: [&[u32]; 3] = [&[5, 12, 33, 77], &[9, 2, 44], &[101, 7, 13, 21, 3]];
const BATCH_TOKENS: usize = 6;

/// Host wiring check for the GB10 case below: on the engine-supported shape, several
/// sequences decoding together reach `paged_attention_batch` with more than one sequence
/// per call. Runs on the CPU reference, so it holds on every build and in CI.
#[tokio::test]
async fn decode_batch_reaches_paged_attention_batch_on_reference() {
    let counting = Arc::new(CountingBackend::new(Arc::new(ReferenceCpuBackend::new())));
    let emitted = run_decode_batch(
        counting.clone(),
        &gb10_micro_model_config(),
        &BATCH_PROMPTS,
        BATCH_TOKENS,
    )
    .await;
    eprintln!("{} emitted={emitted:?}", counting.line());
    assert_eq!(emitted, vec![BATCH_TOKENS; BATCH_PROMPTS.len()]);
    assert!(
        counting.calls(TensorOp::PagedAttentionBatch) > 0,
        "decode never reached paged_attention_batch: {}",
        counting.line()
    );
    assert!(
        counting.max_batch_seqs.load(Ordering::Relaxed) >= 2,
        "decode batch never held more than one sequence: {}",
        counting.line()
    );
}

/// GB10 chip case: the same multi-sequence decode on the native Omega engine. Every
/// `paged_attention_batch` call (and every other op) must run native: fallback count 0,
/// no chip errors, an empty op report. In a production build a fallback is already fatal
/// (strict.rs); the asserts also hold a dev build to zero.
#[tokio::test]
async fn gb10_decode_batch_runs_paged_attention_batch_natively() {
    let gpu_backend = Arc::new(OmegaGb10Backend::new());
    if !gpu_backend.is_available() {
        eprintln!("Skipping GB10 test: Omega GPU engine not linked");
        return;
    }
    let counting = Arc::new(CountingBackend::new(gpu_backend.clone()));
    let emitted = run_decode_batch(
        counting.clone(),
        &gb10_micro_model_config(),
        &BATCH_PROMPTS,
        BATCH_TOKENS,
    )
    .await;
    let report = gpu_backend.op_report();
    eprintln!(
        "GB10_PAGED_BATCH {} emitted={emitted:?} chip_calls={} chip_errors={} fallback_count={} report={} last_error={:?}",
        counting.line(),
        gpu_backend.chip_calls(),
        gpu_backend.chip_errors(),
        gpu_backend.fallback_count(),
        report.line(),
        gpu_backend.last_error()
    );
    assert_eq!(emitted, vec![BATCH_TOKENS; BATCH_PROMPTS.len()]);
    assert!(
        counting.calls(TensorOp::PagedAttentionBatch) > 0,
        "{}",
        counting.line()
    );
    assert!(
        counting.max_batch_seqs.load(Ordering::Relaxed) >= 2,
        "{}",
        counting.line()
    );
    assert_eq!(gpu_backend.fallback_count(), 0, "{}", report.line());
    assert_eq!(gpu_backend.chip_errors(), 0, "{}", gpu_backend.last_error());
    assert!(
        report.native_fallbacks.is_empty() && report.reference_runs.is_empty(),
        "{}",
        report.line()
    );
    assert!(gpu_backend.chip_calls() > 0);
    eprintln!("GB10_PAGED_BATCH verdict: PASS");
}
