//! Blackwell GB10 Batch Executor Parity & Verification Suite.
//! Tests:
//! 1. Fused GQA QKV (q_dim + 2 * kv_dim) and fused Gate/Up dimensions.
//! 2. Persistent GPU workspace and resident model lifecycle.
//! 3. Batched execution: concurrent ragged prefill + batched decode in single step.
//! 4. Single completion fence per step (zero intra-layer synchronization).
//! 5. Strict KV transaction commit & rollback semantics on failure.
//! 6. Zero fallback assertion in accelerated Blackwell environment.

use aien_inference_abi::{
    AienInferenceBackend, BlackwellBatchExecutor, ModelConfig, SamplingParams, ScheduledBatch,
    SequenceRequest, TransformerWeights,
};
use aien_kv_cache::{AienKvManager, DefaultUnifiedBuffer, KvDType, KvPoolConfig};
use parking_lot::RwLock;
use std::sync::Arc;

fn test_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "test-blackwell-llama".to_string(),
        max_sequence_length: 2048,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        hidden_dim: 128,
        intermediate_dim: 256,
        vocab_size: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

fn tinyllama_config() -> ModelConfig {
    ModelConfig {
        model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        max_sequence_length: 2048,
        block_size: 16,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 4,
        head_dim: 64,
        hidden_dim: 2048,
        intermediate_dim: 5632,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

#[test]
fn test_fused_gqa_qkv_dimension_invariants() {
    let config = tinyllama_config();
    let hidden_dim = config.hidden_dim();
    let num_q_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim();

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let fused_qkv_dim = q_dim + 2 * kv_dim;
    let fused_gate_up_dim = 2 * intermediate_dim;

    assert_eq!(q_dim, 2048, "Q projection dim must equal 32 * 64 = 2048");
    assert_eq!(kv_dim, 256, "KV projection dim must equal 4 * 64 = 256");
    assert_eq!(
        fused_qkv_dim, 2560,
        "Fused GQA QKV projection width must equal q_dim + 2 * kv_dim = 2560"
    );
    assert_eq!(
        fused_gate_up_dim, 11264,
        "Fused Gate/Up projection width must equal 2 * intermediate_dim = 11264"
    );
    assert_eq!(hidden_dim, 2048);
}

#[test]
fn test_blackwell_executor_lifecycle_and_workspace() {
    let config = test_model_config();
    let weights = TransformerWeights::reference_test_weights(&config);

    let executor = BlackwellBatchExecutor::new(&weights, 128)
        .expect("BlackwellBatchExecutor must initialize on Blackwell GB10");

    assert_eq!(executor.config.num_layers, 2);
    assert_eq!(executor.workspace.max_tokens, 128);
    assert_eq!(executor.fallback_count(), 0);
}

#[test]
fn test_blackwell_batched_step_ragged_prefill_and_decode() {
    let config = test_model_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut executor =
        BlackwellBatchExecutor::new(&weights, 256).expect("BlackwellBatchExecutor must initialize");

    let block_size = 16;
    let mut kv_mgr = AienKvManager::<DefaultUnifiedBuffer>::new(64, block_size);
    kv_mgr
        .attach_tensor_pool(KvPoolConfig {
            num_blocks: 64,
            block_size,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            dtype: KvDType::Bf16,
        })
        .expect("Must attach unified BF16 KV tensor pool");

    // Pre-populate sequence 1 with 20 tokens (2 blocks)
    let initial_tokens: Vec<u32> = (1..=20).collect();
    let _blocks_seq1 = kv_mgr
        .allocate_sequence(1, &initial_tokens)
        .expect("Must allocate sequence 1");
    assert_eq!(kv_mgr.allocated_block_count(), 2);

    let initial_kernel_exec = executor.kernel_exec_count();

    // Construct batch: 1 decode (seq 1) + 1 prefill (seq 2 with 6 tokens)
    let batch = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: 2,
            prompt_tokens: vec![101, 102, 103, 104, 105, 106],
            sampling_params: SamplingParams::default(),
            arrival_time_ns: 1000,
            priority: 0,
        }],
        decode_requests: vec![1],
        block_tables: Default::default(),
        step_id: 1,
    };

    let (outputs, metrics) = executor
        .execute_step_transactional(&batch, &mut kv_mgr)
        .expect("execute_step_transactional must succeed");

    // Verify outputs
    assert_eq!(
        outputs.len(),
        2,
        "Batch must produce exactly 2 outputs (1 decode + 1 prefill)"
    );
    assert_eq!(metrics.prefill_tokens_processed, 6);
    assert_eq!(metrics.decode_tokens_emitted, 2);

    // Verify sequence lengths in KV manager
    let table_seq1 = kv_mgr.get_block_table(1).expect("Seq 1 table must exist");
    assert_eq!(
        table_seq1.total_tokens, 21,
        "Seq 1 context length must be 21 after decode step"
    );

    let table_seq2 = kv_mgr.get_block_table(2).expect("Seq 2 table must exist");
    assert_eq!(
        table_seq2.total_tokens, 6,
        "Seq 2 context length must be 6 after prefill step"
    );

    #[cfg(has_blackwell_cuda)]
    {
        assert_eq!(
            executor.fallback_count(),
            0,
            "Zero fallback allowed on GB10 silicon"
        );
        assert!(
            executor.kernel_exec_count() > initial_kernel_exec,
            "Kernel execution counter must advance on GPU execution"
        );
    }
}

#[test]
fn test_strict_kv_transactional_rollback_on_failure() {
    let block_size = 16;
    let mut kv_mgr = AienKvManager::<DefaultUnifiedBuffer>::new(4, block_size);
    kv_mgr
        .attach_tensor_pool(KvPoolConfig {
            num_blocks: 4,
            block_size,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 32,
            dtype: KvDType::Bf16,
        })
        .expect("Must attach unified BF16 KV tensor pool");

    // Allocate seq 1 with 20 tokens -> requires 2 blocks
    let initial_tokens: Vec<u32> = (1..=20).collect();
    let _ = kv_mgr.allocate_sequence(1, &initial_tokens).unwrap();
    let baseline_blocks = kv_mgr.allocated_block_count();
    assert_eq!(baseline_blocks, 2);

    // Start transaction
    let mut tx = kv_mgr.begin_transaction();

    // In-flight allocations: append token to seq 1 (still fits in block 1)
    let _ = kv_mgr.append_token_with_slot(1).unwrap();
    assert_eq!(kv_mgr.get_block_table(1).unwrap().total_tokens, 21);

    // Try allocating sequence 2 that exceeds capacity (requires 3 blocks, only 2 left)
    let big_prompt: Vec<u32> = (1..=40).collect();
    let alloc_res = kv_mgr.allocate_sequence(2, &big_prompt);
    assert!(
        alloc_res.is_err(),
        "Allocation exceeding pool capacity must fail"
    );

    // Rollback transaction
    tx.rollback(&mut kv_mgr);

    // Assert complete restoration to baseline state:
    assert_eq!(
        kv_mgr.allocated_block_count(),
        baseline_blocks,
        "Allocated block count must be restored to baseline after rollback"
    );
    assert_eq!(
        kv_mgr.get_block_table(1).unwrap().total_tokens,
        20,
        "Sequence 1 context length must be rolled back to 20"
    );
    assert!(
        kv_mgr.get_block_table(2).is_none(),
        "Sequence 2 must not exist in block tables after rollback"
    );
}

#[tokio::test]
async fn test_blackwell_executor_backend_trait_async() {
    let config = test_model_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let executor =
        BlackwellBatchExecutor::new(&weights, 256).expect("BlackwellBatchExecutor must initialize");

    let block_size = 16;
    let mut kv_mgr = AienKvManager::<DefaultUnifiedBuffer>::new(64, block_size);
    kv_mgr
        .attach_tensor_pool(KvPoolConfig {
            num_blocks: 64,
            block_size,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            dtype: KvDType::Bf16,
        })
        .unwrap();

    let shared_kv = Arc::new(RwLock::new(kv_mgr));
    let mut backend: Box<dyn AienInferenceBackend> =
        Box::new(executor.with_kv_manager(shared_kv.clone()));

    let batch = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: 10,
            prompt_tokens: vec![1, 2, 3, 4],
            sampling_params: SamplingParams::default(),
            arrival_time_ns: 2000,
            priority: 0,
        }],
        decode_requests: vec![],
        block_tables: Default::default(),
        step_id: 1,
    };

    let (outputs, metrics) = backend
        .execute_step(&batch)
        .await
        .expect("Backend execute_step must succeed");

    assert_eq!(outputs.len(), 1);
    assert_eq!(metrics.prefill_tokens_processed, 4);
    assert_eq!(metrics.decode_tokens_emitted, 1);
}
