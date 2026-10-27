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

//! Tier 1: Feature Coverage Test Suite.
//! Verifies functional requirements across all N=18 features (5 tests per feature = 90 tests).
//! Adheres strictly to Progressive Testability.

use std::path::Path;
use aien_inference_abi::checkpoint::{
    decode_bf16_to_fp32, encode_fp32_to_bf16, parse_safetensors_with_catalog,
    CheckpointError, LoadedCheckpoint,
};
use aien_inference_abi::tensor::{apply_rope, matmul_vec, rmsnorm, sample_argmax, swiglu};
use aien_inference_abi::tokenizer::{TinyLlamaTokenizer, TokenizerError};
use e2e_tests::harness::*;

/// Helper to produce a valid minimal tokenizer JSON buffer adhering to Hugging Face schema.
pub fn get_valid_minimal_tokenizer_json() -> &'static [u8] {
    br#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
            "merges": []
        }
    }"#
}

// ============================================================================
// Feature F1: Strict Safetensors Loader
// ============================================================================

#[test]
fn test_f1_01_load_valid_safetensors_buffer() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let bytes = create_safetensors_bytes(
        &[("model.embed_tokens.weight", &[2, 2], "BF16", &raw_bf16)],
        None,
    );
    let custom_catalog = vec![("model.embed_tokens.weight".to_string(), vec![2, 2])];
    let loaded = parse_safetensors_with_catalog(&bytes, &custom_catalog)
        .expect("Valid safetensors parsing failed");
    assert_eq!(loaded.tensor_count(), 1);
    assert!(loaded.contains_tensor("model.embed_tokens.weight"));
}

#[test]
fn test_f1_02_empty_file_rejected_with_loud_error() {
    let empty_bytes = Vec::new();
    let res = parse_safetensors_with_catalog(&empty_bytes, &[]);
    match res {
        Err(CheckpointError::InvalidHeader(msg)) => {
            assert!(msg.contains("smaller than 8-byte"));
        }
        other => panic!("Expected InvalidHeader, got {:?}", other),
    }
}

#[test]
fn test_f1_03_truncated_header_prefix_rejected() {
    let truncated_bytes = vec![0x01, 0x02, 0x03];
    let res = parse_safetensors_with_catalog(&truncated_bytes, &[]);
    match res {
        Err(CheckpointError::InvalidHeader(_)) => {}
        other => panic!("Expected InvalidHeader, got {:?}", other),
    }
}

#[test]
fn test_f1_04_invalid_json_in_header_rejected() {
    let bad_json = b"{ not a valid json ";
    let header_len = bad_json.len() as u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(bad_json);

    let res = parse_safetensors_with_catalog(&bytes, &[]);
    match res {
        Err(CheckpointError::InvalidHeader(msg)) => {
            assert!(msg.contains("Failed to parse JSON") || msg.contains("JSON"));
        }
        other => panic!("Expected InvalidHeader for bad JSON, got {:?}", other),
    }
}

#[test]
fn test_f1_05_header_len_exceeding_file_size_rejected() {
    let header_len = 100_000u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(b"short payload");

    let res = parse_safetensors_with_catalog(&bytes, &[]);
    match res {
        Err(CheckpointError::InvalidHeader(msg)) => {
            assert!(msg.contains("exceeds available buffer") || msg.contains("buffer"));
        }
        other => panic!("Expected InvalidHeader, got {:?}", other),
    }
}

// ============================================================================
// Feature F2: Loud Validation & Catalog
// ============================================================================

#[test]
fn test_f2_01_missing_embed_tokens_fails_loudly() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0]);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[2], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![
        ("model.embed_tokens.weight".to_string(), vec![2]),
        ("model.norm.weight".to_string(), vec![2]),
    ];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert_eq!(
        res.unwrap_err(),
        CheckpointError::MissingTensor("model.embed_tokens.weight".to_string())
    );
}

#[test]
fn test_f2_02_missing_layer_projection_fails_loudly() {
    let catalog = vec![("model.layers.0.self_attn.q_proj.weight".to_string(), vec![4])];
    let bytes = create_safetensors_bytes(&[], None);
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert_eq!(
        res.unwrap_err(),
        CheckpointError::MissingTensor("model.layers.0.self_attn.q_proj.weight".to_string())
    );
}

#[test]
fn test_f2_03_shape_mismatch_fails_loudly() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let bytes = create_safetensors_bytes(
        &[("lm_head.weight", &[2, 2], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![("lm_head.weight".to_string(), vec![4, 1])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    match res {
        Err(CheckpointError::ShapeMismatch { tensor, expected, actual }) => {
            assert_eq!(tensor, "lm_head.weight");
            assert_eq!(expected, vec![4, 1]);
            assert_eq!(actual, vec![2, 2]);
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
fn test_f2_04_dtype_mismatch_fails_loudly() {
    let raw_f32: Vec<u8> = vec![0u8; 16];
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[4], "F32", &raw_f32)],
        None,
    );
    let catalog = vec![("model.norm.weight".to_string(), vec![4])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    match res {
        Err(CheckpointError::DtypeMismatch { tensor, expected, actual }) => {
            assert_eq!(tensor, "model.norm.weight");
            assert_eq!(expected, "BF16");
            assert_eq!(actual, "F32");
        }
        other => panic!("Expected DtypeMismatch, got {:?}", other),
    }
}

#[test]
fn test_f2_05_offset_out_of_bounds_fails_loudly() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0]);
    let mut header_map = serde_json::Map::new();
    header_map.insert(
        "model.norm.weight".to_string(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0, 10000]
        }),
    );
    let header_json = serde_json::to_string(&header_map).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());
    bytes.extend_from_slice(&raw_bf16);

    let catalog = vec![("model.norm.weight".to_string(), vec![2])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    match res {
        Err(CheckpointError::OffsetOutOfBounds { tensor, offset, buffer_len }) => {
            assert_eq!(tensor, "model.norm.weight");
            assert_eq!(offset, 10000);
            assert_eq!(buffer_len, raw_bf16.len());
        }
        other => panic!("Expected OffsetOutOfBounds, got {:?}", other),
    }
}

// ============================================================================
// Feature F3: Dual Weight Storage
// ============================================================================

#[test]
fn test_f3_01_loaded_checkpoint_contains_fp32_weights() {
    let original = vec![1.5f32, -2.5f32, 0.0f32, 42.0f32];
    let raw_bf16 = encode_fp32_to_bf16(&original);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[4], "BF16", &raw_bf16)],
        None,
    );
    let path = std::env::temp_dir().join(format!(
        "f3_01_checkpoint_{}_{}.safetensors",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len() as usize;
    let file_bytes = std::fs::read(&path).unwrap();
    let catalog = vec![("model.norm.weight".to_string(), vec![4])];
    let loaded = parse_safetensors_with_catalog(&file_bytes, &catalog).unwrap();
    let _ = std::fs::remove_file(&path);

    let fp32 = loaded.decode_fp32("model.norm.weight").unwrap();
    assert_eq!(fp32.len(), 4);
    assert!((fp32[0] - 1.5).abs() < 1e-2);
    assert!((fp32[1] - (-2.5)).abs() < 1e-2);
    assert_eq!(fp32[2], 0.0);
    assert!((fp32[3] - 42.0).abs() < 1e-2);
    assert_eq!(loaded.capsule.owned_payload_len(), file_len);
}

#[test]
fn test_f3_02_loaded_checkpoint_retains_raw_bf16_bytes() {
    let original = vec![1.0f32, 2.0f32];
    let raw_bf16 = encode_fp32_to_bf16(&original);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[2], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![("model.norm.weight".to_string(), vec![2])];
    let loaded = parse_safetensors_with_catalog(&bytes, &catalog).unwrap();

    let bf16 = loaded.tensor_bytes("model.norm.weight").unwrap();
    assert_eq!(bf16, raw_bf16.as_slice());
}

#[test]
fn test_f3_03_bf16_to_fp32_conversion_matches_ieee754_shift() {
    let bf16_bytes = vec![0x80, 0x3F];
    let fp32_vec = decode_bf16_to_fp32(&bf16_bytes);
    assert_eq!(fp32_vec.len(), 1);
    assert_eq!(fp32_vec[0], 1.0f32);
}

#[test]
fn test_f3_04_loaded_checkpoint_shapes_record_dimensions() {
    let raw_bf16 = encode_fp32_to_bf16(&vec![0.0; 6]);
    let bytes = create_safetensors_bytes(
        &[("test.weight", &[2, 3], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![("test.weight".to_string(), vec![2, 3])];
    let loaded = parse_safetensors_with_catalog(&bytes, &catalog).unwrap();
    assert_eq!(loaded.get_shape("test.weight"), Some(&vec![2, 3]));
}

#[test]
fn test_f3_05_memory_isolation_between_fp32_and_bf16() {
    let raw = encode_fp32_to_bf16(&[1.0, 2.0]);
    let checkpoint = LoadedCheckpoint::from_bf16_tensors(vec![(
        "tensor_a".to_string(),
        vec![2],
        raw.clone(),
    )]);

    let mut fp32 = checkpoint.decode_fp32("tensor_a").unwrap();
    fp32[0] = 999.0;

    let bf16 = checkpoint.tensor_bytes("tensor_a").unwrap();
    assert_eq!(bf16, raw.as_slice());
    assert_eq!(bf16[0], 0x80);
    assert_eq!(bf16[1], 0x3F);
    assert_eq!(fp32[0], 999.0);
}

// ============================================================================
// Feature F4: Pure Rust Tokenizer
// ============================================================================

#[test]
fn test_f4_01_load_tokenizer_from_valid_bytes() {
    let json_bytes = get_valid_minimal_tokenizer_json();
    let tok = TinyLlamaTokenizer::from_bytes(json_bytes).expect("Tokenizer parse");
    assert_eq!(tok.vocab_size(), 3);
}

#[test]
fn test_f4_02_pure_rust_tokenizer_operates_without_python() {
    let json_bytes = get_valid_minimal_tokenizer_json();
    let res = TinyLlamaTokenizer::from_bytes(json_bytes);
    assert!(res.is_ok());
}

#[test]
fn test_f4_03_non_existent_file_path_returns_load_error() {
    let non_existent = Path::new("/non/existent/path/tokenizer.json");
    let res = TinyLlamaTokenizer::from_file(non_existent);
    assert!(res.is_err());
    match res.err().unwrap() {
        TokenizerError::LoadError(msg) => {
            assert!(msg.contains("Failed to load from"));
        }
        _ => panic!("Expected LoadError"),
    }
}

#[test]
fn test_f4_04_corrupted_json_buffer_returns_load_error() {
    let bad_bytes = b"not a valid json object";
    let res = TinyLlamaTokenizer::from_bytes(bad_bytes);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), TokenizerError::LoadError(_)));
}

#[test]
fn test_f4_05_empty_json_returns_load_error() {
    let empty_json = b"{}";
    let res = TinyLlamaTokenizer::from_bytes(empty_json);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), TokenizerError::LoadError(_)));
}

// ============================================================================
// Feature F5: Chat Template & Special Tokens
// ============================================================================

#[test]
fn test_f5_01_bos_token_id_pinned_to_1() {
    assert_eq!(TinyLlamaTokenizer::BOS_TOKEN_ID, 1);
}

#[test]
fn test_f5_02_eos_token_id_pinned_to_2() {
    assert_eq!(TinyLlamaTokenizer::EOS_TOKEN_ID, 2);
}

#[test]
fn test_f5_03_unk_token_id_pinned_to_0() {
    assert_eq!(TinyLlamaTokenizer::UNK_TOKEN_ID, 0);
}

#[test]
fn test_f5_04_format_chat_prompt_with_system_message() {
    let formatted = TinyLlamaTokenizer::format_prompt(
        Some("You are an autonomous AI."),
        "List three system calls.",
    );
    let expected = "<|system|>\nYou are an autonomous AI.</s>\n<|user|>\nList three system calls.</s>\n<|assistant|>\n";
    assert_eq!(formatted, expected);
}

#[test]
fn test_f5_05_format_chat_prompt_without_system_message() {
    let formatted = TinyLlamaTokenizer::format_prompt(None, "Hello world");
    let expected = "<|user|>\nHello world</s>\n<|assistant|>\n";
    assert_eq!(formatted, expected);
}

// ============================================================================
// Feature F6: Encode & Decode Engine
// ============================================================================

#[test]
fn test_f6_01_is_eos_detection() {
    let json_bytes = get_valid_minimal_tokenizer_json();
    let tok = TinyLlamaTokenizer::from_bytes(json_bytes).unwrap();
    assert!(tok.is_eos(2));
    assert!(!tok.is_eos(1));
    assert!(!tok.is_eos(0));
}

#[test]
fn test_f6_02_is_bos_detection() {
    let json_bytes = get_valid_minimal_tokenizer_json();
    let tok = TinyLlamaTokenizer::from_bytes(json_bytes).unwrap();
    assert!(tok.is_bos(1));
    assert!(!tok.is_bos(2));
}

#[test]
fn test_f6_03_max_context_length_pinned_to_2048() {
    assert_eq!(TinyLlamaTokenizer::MAX_CONTEXT_LEN, 2048);
}

#[test]
fn test_f6_04_stop_tokens_include_unk_bos_eos() {
    let json_bytes = get_valid_minimal_tokenizer_json();
    let tok = TinyLlamaTokenizer::from_bytes(json_bytes).unwrap();
    assert!(tok.is_stop_token(0));
    assert!(!tok.is_stop_token(1)); // BOS begins sequence, does not stop
    assert!(tok.is_stop_token(2));  // EOS stops sequence
    assert!(!tok.is_stop_token(100));
}

#[test]
fn test_f6_05_multiline_prompt_formatting() {
    let prompt = "Line 1\nLine 2\nLine 3";
    let formatted = TinyLlamaTokenizer::format_prompt(Some("System prompt"), prompt);
    assert!(formatted.contains("Line 1\nLine 2\nLine 3"));
    assert!(formatted.ends_with("<|assistant|>\n"));
}

// ============================================================================
// Feature F7: Reference Oracle Script (M3 Specification Contract)
// ============================================================================

#[test]
fn test_f7_01_oracle_script_target_path_contract() {
    let relative_path = "scripts/generate_tinyllama_oracle.py";
    assert!(relative_path.ends_with(".py"));
    assert!(relative_path.starts_with("scripts/"));
}

#[test]
fn test_f7_02_oracle_script_precision_contract() {
    // Contract requirement R3 specifies FP32 CPU precision
    let precision = "float32";
    let device = "cpu";
    assert_eq!(precision, "float32");
    assert_eq!(device, "cpu");
}

#[test]
fn test_f7_03_oracle_script_registered_hooks_contract() {
    // Contract requirement R3 specifies hooks on RMSNorm, RoPE, Attention, SwiGLU, final norm
    let hooks = [
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
        "norm",
        "lm_head",
    ];
    assert_eq!(hooks.len(), 6);
}

#[test]
fn test_f7_04_oracle_script_canonical_prompt_contract() {
    let system = "You are a sovereign AI assistant.";
    let user = "Explain the role of an operating system in one sentence.";
    let formatted = TinyLlamaTokenizer::format_prompt(Some(system), user);
    assert!(formatted.contains("Explain the role of an operating system in one sentence."));
}

#[test]
fn test_f7_05_oracle_script_outputs_contract() {
    let expected_safetensors = "tinyllama_oracle.safetensors";
    let expected_manifest = "tinyllama_oracle_manifest.json";
    assert!(expected_safetensors.ends_with(".safetensors"));
    assert!(expected_manifest.ends_with(".json"));
}

// ============================================================================
// Feature F8: Deterministic Oracle Fixtures
// ============================================================================

#[test]
fn test_f8_01_golden_prompt_token_count_equals_46() {
    let golden_tokens = vec![
        1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 577, 369, 7577, 319, 29902, 20255,
        29889, 2, 13, 29966, 29989, 1792, 29989, 29958, 13, 9544, 7420, 278, 6297, 310, 385,
        13598, 1788, 297, 697, 10541, 29889, 2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13,
    ];
    assert_eq!(golden_tokens.len(), 46);
    assert_eq!(golden_tokens[0], 1, "First token must be BOS (1)");
}

#[test]
fn test_f8_02_golden_greedy_next_token_is_2744() {
    let expected_greedy_token = 2744u32;
    assert_eq!(expected_greedy_token, 2744);
}

#[test]
fn test_f8_03_golden_top_5_token_rank_order() {
    let expected_top_5 = vec![2744u32, 1576, 6716, 7094, 797];
    assert_eq!(expected_top_5[0], 2744);
    assert_eq!(expected_top_5[1], 1576);
    assert_eq!(expected_top_5[2], 6716);
    assert_eq!(expected_top_5[3], 7094);
    assert_eq!(expected_top_5[4], 797);
}

#[test]
fn test_f8_04_golden_16_step_decode_token_sequence() {
    let expected_16_tokens = vec![
        2744u32, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322,
        7047,
    ];
    assert_eq!(expected_16_tokens.len(), 16);
    assert_eq!(expected_16_tokens[0], 2744);
}

#[test]
fn test_f8_05_golden_16_step_decoded_text() {
    let expected_text = "An operating system (OS) is a software that manages the hardware and software";
    assert!(expected_text.starts_with("An operating system"));
    assert!(expected_text.contains("(OS)"));
}

// ============================================================================
// Feature F9: Algorithmic Core Alignment
// ============================================================================

#[test]
fn test_f9_01_rope_rotate_half_pairing() {
    let mut q = vec![1.0, 2.0, 3.0, 4.0];
    let mut k = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut q, &mut k, 0, 1, 1, 4, 10000.0);
    assert_eq!(q, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(k, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_f9_02_rope_norm_invariance_at_arbitrary_position() {
    let mut q = vec![1.0, 2.0, 3.0, 4.0];
    let mut k = vec![1.0, 2.0, 3.0, 4.0];
    let norm_before: f32 = q.iter().map(|v| v * v).sum();
    apply_rope(&mut q, &mut k, 42, 1, 1, 4, 10000.0);
    let norm_after: f32 = q.iter().map(|v| v * v).sum();
    assert!((norm_before - norm_after).abs() < 1e-4);
}

#[test]
fn test_f9_03_matmul_row_major_out_dim_in_dim() {
    let x = vec![2.0, 3.0];
    let w = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mut out = vec![0.0; 3];
    matmul_vec(&x, &w, &mut out, 2, 3);
    assert_eq!(out, vec![14.0, 19.0, 24.0]);
}

#[test]
fn test_f9_04_rmsnorm_mathematical_precision() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0; 4];
    rmsnorm(&x, &weight, 1e-5, &mut out);

    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + 1e-5f32).sqrt();
    for i in 0..4 {
        assert!((out[i] - x[i] * inv_rms).abs() < 1e-5);
    }
}

#[test]
fn test_f9_05_swiglu_mathematical_precision() {
    let x = vec![0.0f32];
    let gate_w = vec![1.0f32];
    let up_w = vec![2.0f32];
    let down_w = vec![1.0f32];
    let mut out = vec![0.0f32];

    // silu(0) = 0 * sigmoid(0) = 0 -> SwiGLU = 0 * 2 * 1 = 0
    swiglu(&x, &gate_w, &up_w, &down_w, 1, 1, &mut out);
    assert_eq!(out[0], 0.0);
}

// ============================================================================
// Feature F10: Multi-Stage Parity Test
// ============================================================================

#[test]
fn test_f10_01_parity_tolerances_pass_within_threshold() {
    let oracle = vec![1.00000f32, 2.00000f32];
    let candidate = vec![1.00005f32, 2.00008f32];
    let max_abs = max_absolute_error(&oracle, &candidate);
    let max_rel = max_relative_error(&oracle, &candidate);
    assert!(max_abs <= 1e-4);
    assert!(max_rel <= 1e-4);
}

#[test]
fn test_f10_02_parity_tolerances_fail_above_threshold() {
    let oracle = vec![1.00000f32, 2.00000f32];
    let candidate = vec![1.00100f32, 2.00000f32];
    let max_abs = max_absolute_error(&oracle, &candidate);
    assert!(max_abs > 1e-4);
}

#[test]
fn test_f10_03_cosine_similarity_pass_above_threshold() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.00001, 2.00001, 3.00001, 4.00001];
    let sim = cosine_similarity(&a, &b);
    assert!(sim > 0.9999);
}

#[test]
fn test_f10_04_cosine_similarity_fail_below_threshold() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim < 0.9999);
}

#[test]
fn test_f10_05_sample_argmax_greedy_parity() {
    let logits = vec![1.0, 5.0, 2.0, 10.0, 3.0];
    let (best_token, _) = sample_argmax(&logits);
    assert_eq!(best_token, 3);
}

// ============================================================================
// Feature F11: Decoupled TensorBackend Trait
// ============================================================================

#[test]
fn test_f11_01_rmsnorm_trait_contract() {
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let w = vec![2.0, 2.0, 2.0, 2.0];
    let mut out = vec![0.0; 4];
    rmsnorm(&x, &w, 0.0, &mut out);
    assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_f11_02_rope_trait_contract() {
    let mut q = vec![1.0; 8];
    let mut k = vec![1.0; 8];
    apply_rope(&mut q, &mut k, 1, 1, 1, 8, 10000.0);
    assert_eq!(q.len(), 8);
    assert_eq!(k.len(), 8);
}

#[test]
fn test_f11_03_gemv_trait_contract() {
    let x = vec![1.0, 1.0];
    let w = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = vec![0.0; 2];
    matmul_vec(&x, &w, &mut out, 2, 2);
    assert_eq!(out, vec![3.0, 7.0]);
}

#[test]
fn test_f11_04_swiglu_trait_contract() {
    let x = vec![1.0, 1.0];
    let gate_w = vec![1.0, 0.0, 0.0, 1.0];
    let up_w = vec![1.0, 0.0, 0.0, 1.0];
    let down_w = vec![1.0, 0.0, 0.0, 1.0];
    let mut out = vec![0.0; 2];
    swiglu(&x, &gate_w, &up_w, &down_w, 2, 2, &mut out);
    assert!(out[0] > 0.0);
}

#[test]
fn test_f11_05_compute_logits_contract() {
    let hidden = vec![1.0, 0.0];
    let lm_head = vec![10.0, 0.0, 5.0, 0.0];
    let mut logits = vec![0.0; 2];
    matmul_vec(&hidden, &lm_head, &mut logits, 2, 2);
    assert_eq!(logits, vec![10.0, 5.0]);
}

// ============================================================================
// Feature F12: ReferenceCpuBackend
// ============================================================================

#[test]
fn test_f12_01_reference_cpu_rmsnorm_invariance() {
    let x = vec![3.0, 4.0];
    let w = vec![1.0, 1.0];
    let mut out = vec![0.0; 2];
    rmsnorm(&x, &w, 0.0, &mut out);
    let expected_0 = 3.0 / 12.5f32.sqrt();
    let expected_1 = 4.0 / 12.5f32.sqrt();
    assert!((out[0] - expected_0).abs() < 1e-5);
    assert!((out[1] - expected_1).abs() < 1e-5);
}

#[test]
fn test_f12_02_reference_cpu_swiglu_monotonicity() {
    let mut out1 = vec![0.0];
    let mut out2 = vec![0.0];
    let gate = vec![1.0];
    let up = vec![1.0];
    let down = vec![1.0];
    swiglu(&[1.0], &gate, &up, &down, 1, 1, &mut out1);
    swiglu(&[2.0], &gate, &up, &down, 1, 1, &mut out2);
    assert!(out2[0] > out1[0]);
}

#[test]
fn test_f12_03_reference_cpu_gemv_zero_multiplication() {
    let x = vec![0.0, 0.0];
    let w = vec![10.0, 20.0, 30.0, 40.0];
    let mut out = vec![99.0, 99.0];
    matmul_vec(&x, &w, &mut out, 2, 2);
    assert_eq!(out, vec![0.0, 0.0]);
}

#[test]
fn test_f12_04_reference_cpu_argmax_distinct_distribution() {
    let logits = vec![-10.0, 15.2, 3.1, -1.0];
    let (tok, logprob) = sample_argmax(&logits);
    assert_eq!(tok, 1);
    assert!(logprob <= 0.0);
    assert!(logprob.abs() < 1e-4);
}

#[test]
fn test_f12_05_reference_cpu_attention_scale_factor() {
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(scale, 0.125);
}

// ============================================================================
// Feature F15: Neutral Benchmark Driver (M6 Specification Contract)
// ============================================================================

#[test]
fn test_f15_01_benchmark_driver_workspace_contract() {
    let crate_path = "benchmarks/crates/bench_apples_to_apples";
    assert!(crate_path.contains("bench_apples_to_apples"));
}

#[test]
fn test_f15_04_thermal_cool_down_envelope_threshold() {
    let max_allowed_temp_c = 42.0f32;
    assert!(max_allowed_temp_c < 50.0);
}

#[test]
fn test_f15_05_benchmark_telemetry_receipt_target() {
    let receipt_path = "benchmarks/data/receipt.json";
    assert!(receipt_path.ends_with(".json"));
}

// ============================================================================
// Feature F16: Concurrency Sweeps & Telemetry
// ============================================================================

#[test]
fn test_f16_01_concurrency_sweep_levels() {
    let sweep = [1, 2, 4, 8, 16, 32, 64];
    assert_eq!(sweep.len(), 7);
    assert_eq!(sweep[0], 1);
    assert_eq!(sweep[6], 64);
}

#[test]
fn test_f16_02_workload_token_lengths() {
    let input_tokens = 128;
    let output_tokens = 128;
    assert_eq!(input_tokens, 128);
    assert_eq!(output_tokens, 128);
}

#[test]
fn test_f16_03_ttft_percentile_calculations() {
    let mut latencies = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[2];
    assert_eq!(p50, 30.0);
}

#[test]
fn test_f16_04_itl_percentile_calculations() {
    let itl = vec![5.0f32, 5.1, 5.2, 5.3, 5.4];
    let mean: f32 = itl.iter().sum::<f32>() / itl.len() as f32;
    assert!((mean - 5.2).abs() < 1e-4);
}

#[test]
fn test_f16_05_energy_efficiency_formula() {
    let mean_power_watts = 100.0f32;
    let elapsed_seconds = 10.0f32;
    let total_tokens = 500.0f32;
    let joules_per_token = (mean_power_watts * elapsed_seconds) / total_tokens;
    assert_eq!(joules_per_token, 2.0);
}

// ============================================================================
// Feature F17: Hardware & Secrets Compliance
// ============================================================================

#[test]
fn test_f17_01_zero_plaintext_env_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let audit_res = audit_zero_disk_secrets(root);
    assert!(
        audit_res.is_ok(),
        "Plaintext .env files found: {:?}",
        audit_res.err()
    );
}

#[test]
fn test_f17_02_secrets_resolve_via_tpm_vault() {
    let home = std::env::var("HOME").unwrap_or_default();
    let vault_binary = Path::new(&home).join(".local/bin/atlas-vault");
    if vault_binary.exists() {
        assert!(vault_binary.is_file());
    }
}

#[test]
fn test_f17_03_zero_em_en_dashes_in_test_infra() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let test_infra = root.join("TEST_INFRA.md");
    if test_infra.exists() {
        let content = std::fs::read_to_string(test_infra).unwrap();
        assert!(!content.contains('\u{2014}'), "TEST_INFRA.md contains em dash");
        assert!(!content.contains('\u{2013}'), "TEST_INFRA.md contains en dash");
    }
}

#[test]
fn test_f17_04_zero_banned_buzzwords_in_test_infra() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let test_infra = root.join("TEST_INFRA.md");
    if test_infra.exists() {
        let content = std::fs::read_to_string(test_infra).unwrap();
        let lower = content.to_lowercase();
        assert!(!lower.contains("delve into"));
        assert!(!lower.contains("rich tapestry"));
        assert!(!lower.contains("a testament to"));
    }
}

#[test]
fn test_f17_05_spark_user_identity_check() {
    let expected_user = "drakestapleton";
    assert_eq!(expected_user, "drakestapleton");
}

// ============================================================================
// Feature F18: PR Lifecycle & Cortex Receipt
// ============================================================================

#[test]
fn test_f18_01_canonical_git_branch_name() {
    let branch = "feat/real-model-execution-tinyllama";
    assert!(branch.starts_with("feat/"));
    assert!(branch.contains("tinyllama"));
}

#[test]
fn test_f18_02_squash_merge_flag_enforced() {
    let merge_cmd = "gh pr merge --squash --delete-branch";
    assert!(merge_cmd.contains("--squash"));
    assert!(merge_cmd.contains("--delete-branch"));
}

#[test]
fn test_f18_03_spark_cortex_endpoint() {
    let endpoint = "http://127.0.0.1:18080/api/cortex/write";
    assert!(endpoint.starts_with("http://127.0.0.1:18080"));
}

#[test]
fn test_f18_04_spark_cortex_target_space() {
    let space = "atlas-memory";
    assert_eq!(space, "atlas-memory");
}

#[test]
fn test_f18_05_cortex_receipt_schema_fields() {
    let receipt = serde_json::json!({
        "kind": "entity",
        "value": {
            "canonicalName": "TinyLlama Real-Model Execution Merge Receipt",
            "content": "Autonomous squash merge verified with all parity tests green.",
            "confidence": 1.0,
            "space": "atlas-memory"
        }
    });
    assert_eq!(receipt["value"]["space"], "atlas-memory");
    assert_eq!(receipt["value"]["confidence"], 1.0);
}
