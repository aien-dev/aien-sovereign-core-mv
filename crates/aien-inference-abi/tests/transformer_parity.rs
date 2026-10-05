//! Numerical correctness and parity test suite for NativeTransformerBackend.
//! Verifies transformer execution, RMSNorm, RoPE, GQA, and multi-step autoregressive generation.

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, ModelConfig, NativeTransformerBackend, SamplingParams,
    ScheduledBatch, SequenceRequest, TransformerWeights,
};
use std::collections::HashMap;

fn test_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-transformer-micro-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        num_kv_heads: 2, // GQA: 4 query heads, 2 KV heads
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

#[tokio::test]
async fn test_native_transformer_prefill_and_decode_step() {
    let config = test_model_config();
    let mut backend = NativeTransformerBackend::with_reference_weights(&config);

    // 1. Prefill Step
    let prompt = vec![12u32, 45u32, 89u32, 120u32];
    let req = SequenceRequest {
        request_id: 1001,
        prompt_tokens: prompt.clone(),
        sampling_params: SamplingParams {
            temperature: 0.0, // Greedy argmax
            top_p: 1.0,
            max_tokens: 32,
            stop_token_ids: vec![0, 1, 2],
        },
        arrival_time_ns: 0,
        priority: 1,
    };

    let prefill_batch = ScheduledBatch {
        prefill_requests: vec![req],
        decode_requests: Vec::new(),
        block_tables: {
            let mut m = HashMap::new();
            m.insert(1001, vec![0, 1]);
            m
        },
        step_id: 1,
    };

    let (outputs, metrics) = backend.execute_step(&prefill_batch).await.unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(metrics.prefill_tokens_processed, 4);
    assert_eq!(metrics.decode_tokens_emitted, 1);

    let _first_token = match &outputs[0] {
        DecodeOutput::Token {
            request_id,
            token_id,
            logprob,
        } => {
            assert_eq!(*request_id, 1001);
            assert!(*token_id < 256, "Token ID must be within vocab_size");
            assert!(logprob.is_some());
            let lp = logprob.unwrap();
            assert!(lp <= 0.0, "Logprob must be non-positive");
            *token_id
        }
        _ => panic!("Expected DecodeOutput::Token"),
    };

    // 2. Decode Step
    let decode_batch = ScheduledBatch {
        prefill_requests: Vec::new(),
        decode_requests: vec![1001],
        block_tables: {
            let mut m = HashMap::new();
            m.insert(1001, vec![0, 1]);
            m
        },
        step_id: 2,
    };

    let (decode_outputs, decode_metrics) = backend.execute_step(&decode_batch).await.unwrap();

    assert_eq!(decode_outputs.len(), 1);
    assert_eq!(decode_metrics.prefill_tokens_processed, 0);
    assert_eq!(decode_metrics.decode_tokens_emitted, 1);

    match &decode_outputs[0] {
        DecodeOutput::Token {
            request_id,
            token_id,
            logprob,
        } => {
            assert_eq!(*request_id, 1001);
            assert!(*token_id < 256);
            assert!(logprob.is_some());
            let lp = logprob.unwrap();
            assert!(lp <= 0.0);
        }
        DecodeOutput::Finished { request_id, .. } => {
            assert_eq!(*request_id, 1001);
        }
    }
}

#[tokio::test]
async fn test_multi_step_autoregressive_generation() {
    let config = test_model_config();
    let mut backend = NativeTransformerBackend::with_reference_weights(&config);

    // Initial prompt
    let req = SequenceRequest {
        request_id: 2002,
        prompt_tokens: vec![5, 10, 15],
        sampling_params: SamplingParams {
            temperature: 0.0, // deterministic greedy
            top_p: 1.0,
            max_tokens: 16,
            stop_token_ids: vec![0],
        },
        arrival_time_ns: 0,
        priority: 1,
    };

    let mut batch = ScheduledBatch {
        prefill_requests: vec![req],
        decode_requests: Vec::new(),
        block_tables: {
            let mut m = HashMap::new();
            m.insert(2002, vec![0, 1]);
            m
        },
        step_id: 1,
    };

    let mut generated_tokens = Vec::new();

    // Step 1: Prefill
    let (outputs, _) = backend.execute_step(&batch).await.unwrap();
    if let DecodeOutput::Token { token_id, .. } = &outputs[0] {
        generated_tokens.push(*token_id);
    }

    // Steps 2..6: 5 autoregressive decode steps
    for step in 2..=6 {
        batch.prefill_requests.clear();
        batch.decode_requests = vec![2002];
        batch.step_id = step;

        let (step_outputs, metrics) = backend.execute_step(&batch).await.unwrap();
        assert_eq!(step_outputs.len(), 1);
        assert_eq!(metrics.decode_tokens_emitted, 1);

        match &step_outputs[0] {
            DecodeOutput::Token { token_id, .. } => {
                generated_tokens.push(*token_id);
            }
            DecodeOutput::Finished { .. } => break,
        }
    }

    assert!(
        !generated_tokens.is_empty(),
        "Must have generated real model tokens"
    );
    // Verify sequence history
    let seq_state = backend.sequences.get(&2002).unwrap();
    assert_eq!(seq_state.tokens.len(), 3 + generated_tokens.len());
}

#[test]
fn test_safetensors_bytes_parser() {
    let config = test_model_config();

    // Construct a valid minimal safetensors binary buffer
    let header_json =
        r#"{"model.norm.weight":{"dtype":"F32","shape":[64],"data_offsets":[0,256]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut buffer = Vec::new();
    buffer.extend_from_slice(&header_len.to_le_bytes());
    buffer.extend_from_slice(header_bytes);

    // 64 floats = 256 bytes
    let raw_floats: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    for f in raw_floats {
        buffer.extend_from_slice(&f.to_le_bytes());
    }

    let weights = TransformerWeights::from_safetensors_bytes(&buffer, &config).unwrap();
    assert_eq!(weights.final_norm.len(), 64);
    assert!((weights.final_norm[1] - 0.1).abs() < 1e-6);
    assert!((weights.final_norm[10] - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_blackwell_vs_reference_cpu_autoregressive_parity() {
    let config = test_model_config();
    let mut cpu_backend = NativeTransformerBackend::with_reference_weights(&config);
    let mut gpu_backend = NativeTransformerBackend::with_blackwell_backend(&config);

    let prompt = vec![7u32, 19u32, 42u32, 108u32];
    let req_cpu = SequenceRequest {
        request_id: 3001,
        prompt_tokens: prompt.clone(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 16,
            stop_token_ids: vec![0],
        },
        arrival_time_ns: 0,
        priority: 1,
    };
    let req_gpu = SequenceRequest {
        request_id: 3001,
        prompt_tokens: prompt.clone(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 16,
            stop_token_ids: vec![0],
        },
        arrival_time_ns: 0,
        priority: 1,
    };

    let mut batch_cpu = ScheduledBatch {
        prefill_requests: vec![req_cpu],
        decode_requests: Vec::new(),
        block_tables: {
            let mut m = HashMap::new();
            m.insert(3001, vec![0, 1]);
            m
        },
        step_id: 1,
    };
    let mut batch_gpu = ScheduledBatch {
        prefill_requests: vec![req_gpu],
        decode_requests: Vec::new(),
        block_tables: {
            let mut m = HashMap::new();
            m.insert(3001, vec![0, 1]);
            m
        },
        step_id: 1,
    };

    // The micro config uses head_dim 16, which omega attention does not support (see
    // omega_backend.rs), and a stub build has no engine at all, so the "GPU" leg is never
    // fully native here. A strict process must refuse it. A dev process runs it with
    // counted fallbacks; in a stub build both legs are then the same reference math, so
    // this is a wiring check, not chip parity. Chip parity lives in
    // tests/omega_backend_parity.rs and the strict real-model gate (heavy queue).
    let native = aien_omega_gpu::is_native();
    if aien_inference_abi::strict::production_strict() {
        let probe = batch_gpu.clone();
        let refused = tokio::spawn(async move { gpu_backend.execute_step(&probe).await })
            .await
            .expect_err("a strict process must refuse a GPU leg that cannot run natively");
        assert!(refused.is_panic(), "{refused}");
        let payload = refused.into_panic();
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains(aien_inference_abi::strict::STRICT_VIOLATION_PREFIX),
            "{msg}"
        );
        return;
    }
    let mut cpu_tokens = Vec::new();
    let mut gpu_tokens = Vec::new();

    let (out_cpu, _) = cpu_backend.execute_step(&batch_cpu).await.unwrap();
    let (out_gpu, _) = gpu_backend.execute_step(&batch_gpu).await.unwrap();

    if let (
        DecodeOutput::Token {
            token_id: t_cpu, ..
        },
        DecodeOutput::Token {
            token_id: t_gpu, ..
        },
    ) = (&out_cpu[0], &out_gpu[0])
    {
        if !native {
            assert_eq!(
                t_cpu, t_gpu,
                "Prefill first token mismatch between CPU and the GPU leg"
            );
        }
        cpu_tokens.push(*t_cpu);
        gpu_tokens.push(*t_gpu);
    }

    for step in 2..=8 {
        batch_cpu.prefill_requests.clear();
        batch_cpu.decode_requests = vec![3001];
        batch_cpu.step_id = step;

        batch_gpu.prefill_requests.clear();
        batch_gpu.decode_requests = vec![3001];
        batch_gpu.step_id = step;

        let (step_out_cpu, _) = cpu_backend.execute_step(&batch_cpu).await.unwrap();
        let (step_out_gpu, _) = gpu_backend.execute_step(&batch_gpu).await.unwrap();

        if let (
            DecodeOutput::Token {
                token_id: t_cpu, ..
            },
            DecodeOutput::Token {
                token_id: t_gpu, ..
            },
        ) = (&step_out_cpu[0], &step_out_gpu[0])
        {
            if !native {
                assert_eq!(
                    t_cpu, t_gpu,
                    "Decode step {} token mismatch: CPU={}, GPU={}",
                    step, t_cpu, t_gpu
                );
            }
            cpu_tokens.push(*t_cpu);
            gpu_tokens.push(*t_gpu);
        }
    }

    // Dev run: the GPU leg cannot be fully native (see above), so its fallbacks must show.
    let fallbacks = gpu_backend.tensor_backend.fallback_count();
    assert!(
        fallbacks > 0,
        "the GPU leg ran {fallbacks} counted fallbacks; expected some"
    );
    if !native {
        assert_eq!(
            cpu_tokens, gpu_tokens,
            "stub dev build: both legs are the reference math and must match bitwise"
        );
    }
    println!(
        "Parity verified over {} tokens: {:?}",
        cpu_tokens.len(),
        cpu_tokens
    );
}

#[tokio::test]
async fn test_multi_sequence_batched_decode_parity_with_sequential() {
    let config = test_model_config();
    let weights = TransformerWeights::reference_test_weights(&config);

    let mut backend_seq = NativeTransformerBackend::new(weights.clone());
    let mut backend_batch = NativeTransformerBackend::new(weights);

    let num_seqs = 8;
    let seq_ids: Vec<u64> = (1001..1001 + num_seqs as u64).collect();

    // 1. Prefill all 8 sequences on both backends
    for (i, &seq_id) in seq_ids.iter().enumerate() {
        let prompt = vec![
            (10 + i * 3) as u32,
            (25 + i * 7) as u32,
            (40 + i * 11) as u32,
        ];
        let req = SequenceRequest {
            request_id: seq_id,
            prompt_tokens: prompt,
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 16,
                stop_token_ids: vec![0],
            },
            arrival_time_ns: 0,
            priority: 1,
        };

        let mut block_tables = HashMap::new();
        block_tables.insert(seq_id, vec![i * 2, i * 2 + 1]);

        let batch = ScheduledBatch {
            prefill_requests: vec![req],
            decode_requests: Vec::new(),
            block_tables,
            step_id: 1,
        };

        let (out_seq, _) = backend_seq.execute_step(&batch).await.unwrap();
        let (out_batch, _) = backend_batch.execute_step(&batch).await.unwrap();

        match (&out_seq[0], &out_batch[0]) {
            (
                DecodeOutput::Token {
                    token_id: t1,
                    logprob: lp1,
                    ..
                },
                DecodeOutput::Token {
                    token_id: t2,
                    logprob: lp2,
                    ..
                },
            ) => {
                assert_eq!(t1, t2, "Prefill token mismatch for seq_id {}", seq_id);
                assert!(
                    (lp1.unwrap() - lp2.unwrap()).abs() < 1e-4,
                    "Logprob mismatch for seq_id {}",
                    seq_id
                );
            }
            _ => panic!("Expected DecodeOutput::Token"),
        }
    }

    // 2. Perform 5 decode steps: sequential (1 sequence per batch) vs batched (all 8 sequences per batch)
    for step in 2..=6 {
        // Sequential execution: 8 individual batches of size 1
        let mut seq_step_tokens = HashMap::new();
        for &seq_id in &seq_ids {
            let mut block_tables = HashMap::new();
            block_tables.insert(seq_id, vec![0, 1]);

            let batch = ScheduledBatch {
                prefill_requests: Vec::new(),
                decode_requests: vec![seq_id],
                block_tables,
                step_id: step,
            };

            let (outputs, _) = backend_seq.execute_step(&batch).await.unwrap();
            assert_eq!(outputs.len(), 1);
            if let DecodeOutput::Token { token_id, .. } = outputs[0] {
                seq_step_tokens.insert(seq_id, token_id);
            }
        }

        // Batched execution: 1 single batch with D = 8 sequences
        let mut batched_block_tables = HashMap::new();
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            batched_block_tables.insert(seq_id, vec![i * 2, i * 2 + 1]);
        }

        let batch = ScheduledBatch {
            prefill_requests: Vec::new(),
            decode_requests: seq_ids.clone(),
            block_tables: batched_block_tables,
            step_id: step,
        };

        let (batched_outputs, metrics) = backend_batch.execute_step(&batch).await.unwrap();
        assert_eq!(batched_outputs.len(), num_seqs);
        assert_eq!(metrics.decode_tokens_emitted, num_seqs);

        // Verify exact token parity across all 8 sequences
        for out in &batched_outputs {
            if let DecodeOutput::Token {
                request_id,
                token_id,
                ..
            } = out
            {
                let expected_tok = seq_step_tokens.get(request_id).copied().unwrap();
                assert_eq!(
                    *token_id, expected_tok,
                    "Step {} mismatch for sequence {}: batched={}, sequential={}",
                    step, request_id, token_id, expected_tok
                );
            }
        }
    }

    // 3. Verify entire token trajectory is identical across all 8 sequences
    for &seq_id in &seq_ids {
        let tokens_seq = &backend_seq.sequences.get(&seq_id).unwrap().tokens;
        let tokens_batch = &backend_batch.sequences.get(&seq_id).unwrap().tokens;
        assert_eq!(
            tokens_seq, tokens_batch,
            "Full token history mismatch for seq_id {}",
            seq_id
        );
    }
}
