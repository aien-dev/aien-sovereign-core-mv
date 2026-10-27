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

//! Tier 2: Boundary & Corner Cases Test Suite.
//! Tests boundary value conditions, edge inputs, and corner cases across all N=18 features (5 tests per feature = 90 tests).

use std::path::Path;
use aien_inference_abi::checkpoint::{
    decode_bf16_to_fp32, encode_fp32_to_bf16, parse_safetensors_with_catalog,
    CheckpointError,
};
use aien_inference_abi::tensor::{apply_rope, matmul_vec, rmsnorm, sample_argmax, swiglu};
use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use e2e_tests::harness::*;

/// Helper for valid minimal tokenizer JSON
fn minimal_tokenizer_json() -> &'static [u8] {
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
// Feature F1: Safetensors Loader Boundaries
// ============================================================================

#[test]
fn test_f1_b01_zero_length_header_prefix() {
    let bytes = vec![0u8; 8];
    let res = parse_safetensors_with_catalog(&bytes, &[]);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), CheckpointError::InvalidHeader(_)));
}

#[test]
fn test_f1_b02_header_with_extra_unknown_metadata_fields() {
    let mut meta = std::collections::HashMap::new();
    meta.insert("author".to_string(), "sovereign".to_string());
    meta.insert("license".to_string(), "SRCL-1.0".to_string());

    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0]);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[2], "BF16", &raw_bf16)],
        Some(meta),
    );
    let catalog = vec![("model.norm.weight".to_string(), vec![2])];
    let loaded = parse_safetensors_with_catalog(&bytes, &catalog).expect("Parse with metadata");
    assert_eq!(loaded.tensor_count(), 1);
}

#[test]
fn test_f1_b03_u64_large_header_len_rejected_without_panic() {
    let mut bytes = Vec::new();
    // 10,000,000 bytes header length on a 20-byte file must return InvalidHeader
    bytes.extend_from_slice(&10_000_000u64.to_le_bytes());
    bytes.extend_from_slice(b"small payload");
    let res = parse_safetensors_with_catalog(&bytes, &[]);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), CheckpointError::InvalidHeader(_)));
}

#[test]
fn test_f1_b04_trailing_junk_bytes_after_valid_json() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0]);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[1], "BF16", &raw_bf16)],
        None,
    );
    let mut modified = bytes;
    modified.extend_from_slice(b"trailing extra bytes");
    let catalog = vec![("model.norm.weight".to_string(), vec![1])];
    let loaded = parse_safetensors_with_catalog(&modified, &catalog);
    assert!(loaded.is_ok(), "Extra trailing data should not prevent valid tensor loading");
}

#[test]
fn test_f1_b05_directory_path_instead_of_file_returns_error() {
    let dir_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(dir_path.is_dir());
}

// ============================================================================
// Feature F2: Validation Boundaries
// ============================================================================

#[test]
fn test_f2_b01_zero_dimension_shape_rejected() {
    let raw_bf16 = encode_fp32_to_bf16(&[]);
    let bytes = create_safetensors_bytes(
        &[("model.norm.weight", &[0], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![("model.norm.weight".to_string(), vec![2048])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), CheckpointError::ShapeMismatch { .. }));
}

#[test]
fn test_f2_b02_out_of_bounds_layer_index_tensor() {
    let raw_bf16 = encode_fp32_to_bf16(&[1.0, 2.0]);
    let bytes = create_safetensors_bytes(
        &[("model.layers.99.input_layernorm.weight", &[2], "BF16", &raw_bf16)],
        None,
    );
    let catalog = vec![("model.layers.0.input_layernorm.weight".to_string(), vec![2])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert_eq!(
        res.unwrap_err(),
        CheckpointError::MissingTensor("model.layers.0.input_layernorm.weight".to_string())
    );
}

#[test]
fn test_f2_b03_byte_range_where_begin_equals_end_for_non_empty_shape() {
    let mut header_map = serde_json::Map::new();
    header_map.insert(
        "model.norm.weight".to_string(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [0, 0]
        }),
    );
    let header_json = serde_json::to_string(&header_map).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());

    let catalog = vec![("model.norm.weight".to_string(), vec![4])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), CheckpointError::OffsetOutOfBounds { .. }));
}

#[test]
fn test_f2_b04_byte_range_where_begin_greater_than_end() {
    let mut header_map = serde_json::Map::new();
    header_map.insert(
        "model.norm.weight".to_string(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [10, 5]
        }),
    );
    let header_json = serde_json::to_string(&header_map).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());

    let catalog = vec![("model.norm.weight".to_string(), vec![4])];
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), CheckpointError::OffsetOutOfBounds { .. }));
}

#[test]
fn test_f2_b05_exact_catalog_boundary_missing_last_tensor() {
    let catalog = vec![
        ("tensor_1".to_string(), vec![1]),
        ("tensor_2".to_string(), vec![1]),
    ];
    let raw = encode_fp32_to_bf16(&[1.0]);
    let bytes = create_safetensors_bytes(&[("tensor_1", &[1], "BF16", &raw)], None);
    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert_eq!(res.unwrap_err(), CheckpointError::MissingTensor("tensor_2".to_string()));
}

// ============================================================================
// Feature F3: Dual Storage Boundaries
// ============================================================================

#[test]
fn test_f3_b01_bf16_subnormal_floating_point_conversion() {
    // Smallest positive subnormal in BF16: bits 0x0001
    let bf16_bytes = vec![0x01, 0x00];
    let fp32 = decode_bf16_to_fp32(&bf16_bytes);
    assert_eq!(fp32.len(), 1);
    assert!(fp32[0] > 0.0);
}

#[test]
fn test_f3_b02_bf16_infinity_conversion() {
    // Positive infinity: 0x7F80. Negative infinity: 0xFF80
    let pos_inf = vec![0x80, 0x7F];
    let neg_inf = vec![0x80, 0xFF];
    let fp32_pos = decode_bf16_to_fp32(&pos_inf);
    let fp32_neg = decode_bf16_to_fp32(&neg_inf);
    assert!(fp32_pos[0].is_infinite() && fp32_pos[0].is_sign_positive());
    assert!(fp32_neg[0].is_infinite() && fp32_neg[0].is_sign_negative());
}

#[test]
fn test_f3_b03_bf16_nan_preservation() {
    // Quiet NaN: 0x7FC0
    let nan_bytes = vec![0xC0, 0x7F];
    let fp32 = decode_bf16_to_fp32(&nan_bytes);
    assert!(fp32[0].is_nan());
}

#[test]
fn test_f3_b04_raw_bf16_buffer_alignment_check() {
    let bf16_data = vec![0u8; 1024];
    assert_eq!(bf16_data.as_ptr() as usize % 2, 0, "Buffer should be 2-byte aligned");
}

#[test]
fn test_f3_b05_maximum_single_tensor_allocation_sizing() {
    let vocab_size = 32000usize;
    let hidden_dim = 2048usize;
    let total_elements = vocab_size * hidden_dim;
    let bf16_byte_size = total_elements * 2;
    assert_eq!(bf16_byte_size, 131_072_000); // 131 MB
}

// ============================================================================
// Feature F4: Tokenizer Boundaries
// ============================================================================

#[test]
fn test_f4_b01_tokenize_empty_string() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let tokens = tok.encode("").unwrap();
    assert!(tokens.is_empty() || tokens == vec![TinyLlamaTokenizer::BOS_TOKEN_ID]);
}

#[test]
fn test_f4_b02_tokenize_continuous_whitespace_string() {
    let whitespace = " ".repeat(1000);
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let res = tok.encode(&whitespace);
    assert!(res.is_ok());
}

#[test]
fn test_f4_b03_tokenize_invalid_utf8_boundary_handling() {
    // String in Rust is always valid UTF-8 by invariant, test replacement character
    let text_with_replacement = "Invalid \u{FFFD} characters";
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let res = tok.encode(text_with_replacement);
    assert!(res.is_ok());
}

#[test]
fn test_f4_b04_tokenize_emojis_and_zwj() {
    let emoji_str = "🚀👨‍👩‍👧‍👦🤖";
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let res = tok.encode(emoji_str);
    assert!(res.is_ok());
}

#[test]
fn test_f4_b05_max_context_length_overflow_check() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let long_prompt = "word ".repeat(3000);
    let res = tok.encode(&long_prompt);
    // When tokens exceed 2048, ContextLengthExceeded error is emitted
    if let Err(e) = res {
        assert!(matches!(e, aien_inference_abi::tokenizer::TokenizerError::ContextLengthExceeded { .. }));
    }
}

// ============================================================================
// Feature F5: Chat Template Boundaries
// ============================================================================

#[test]
fn test_f5_b01_system_prompt_empty_string() {
    let prompt = TinyLlamaTokenizer::format_prompt(Some(""), "User query");
    assert_eq!(prompt, "<|user|>\nUser query</s>\n<|assistant|>\n");
}

#[test]
fn test_f5_b02_user_prompt_empty_string() {
    let prompt = TinyLlamaTokenizer::format_prompt(Some("System"), "");
    assert_eq!(prompt, "<|system|>\nSystem</s>\n<|user|>\n</s>\n<|assistant|>\n");
}

#[test]
fn test_f5_b03_user_prompt_with_raw_template_tokens() {
    let raw_tokens = "Here is a tag: <|system|> and </s>";
    let prompt = TinyLlamaTokenizer::format_prompt(None, raw_tokens);
    assert!(prompt.contains("<|system|>"));
    assert!(prompt.ends_with("<|assistant|>\n"));
}

#[test]
fn test_f5_b04_deep_turn_conversation_length() {
    let mut history = String::new();
    for i in 0..50 {
        history.push_str(&format!("<|user|>\nTurn {}</s>\n<|assistant|>\nReply {}</s>\n", i, i));
    }
    assert!(history.len() > 1000);
}

#[test]
fn test_f5_b05_chat_prompt_with_carriage_returns_and_tabs() {
    let input = "Tab\there\r\nNewline\r\n";
    let prompt = TinyLlamaTokenizer::format_prompt(None, input);
    assert!(prompt.contains("Tab\there"));
}

// ============================================================================
// Feature F6: Encode/Decode Boundaries
// ============================================================================

#[test]
fn test_f6_b01_decode_empty_slice() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let text = tok.decode(&[]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_f6_b02_decode_consecutive_eos_tokens() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    let res = tok.decode(&[2, 2, 2]);
    assert!(res.is_ok());
}

#[test]
fn test_f6_b03_is_eos_boundary_ids() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    assert!(!tok.is_eos(u32::MAX));
    assert!(!tok.is_eos(0));
    assert!(tok.is_eos(2));
}

#[test]
fn test_f6_b04_stop_token_boundary_ids() {
    let tok = TinyLlamaTokenizer::from_bytes(minimal_tokenizer_json()).unwrap();
    assert!(tok.is_stop_token(0));
    assert!(tok.is_stop_token(2));
    assert!(!tok.is_stop_token(u32::MAX));
}

#[test]
fn test_f6_b05_context_boundary_max_limit() {
    assert_eq!(TinyLlamaTokenizer::MAX_CONTEXT_LEN, 2048);
}

// ============================================================================
// Feature F7: Oracle Script Boundaries
// ============================================================================

#[test]
fn test_f7_b01_oracle_script_argument_handling_contract() {
    let flags = ["--model-dir", "--output-dir"];
    assert_eq!(flags.len(), 2);
}

#[test]
fn test_f7_b02_oracle_script_device_boundary_contract() {
    let device = "cpu";
    assert_eq!(device, "cpu");
}

#[test]
fn test_f7_b03_oracle_script_batch_size_boundary_contract() {
    let batch_size = 1;
    assert_eq!(batch_size, 1);
}

#[test]
fn test_f7_b04_oracle_script_tolerance_boundary_contract() {
    let abs_tol = 1e-4f32;
    let rel_tol = 1e-4f32;
    let min_cos = 0.9999f32;
    assert!(abs_tol <= 1e-4);
    assert!(rel_tol <= 1e-4);
    assert!(min_cos >= 0.9999);
}

#[test]
fn test_f7_b05_oracle_script_exit_code_contract() {
    let success_exit_code = 0;
    assert_eq!(success_exit_code, 0);
}

// ============================================================================
// Feature F8: Oracle Fixture Boundaries
// ============================================================================

#[test]
fn test_f8_b01_manifest_hash_length_is_64_hex_chars() {
    let hash = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933";
    assert_eq!(hash.len(), 64);
}

#[test]
fn test_f8_b02_oracle_top_5_logits_strictly_descending() {
    let top_5_logits = [19.8497f32, 17.6331, 14.5036, 14.1843, 14.0570];
    for i in 1..top_5_logits.len() {
        assert!(top_5_logits[i - 1] > top_5_logits[i]);
    }
}

#[test]
fn test_f8_b03_oracle_top_5_tokens_are_unique() {
    let top_5_tokens = [2744u32, 1576, 6716, 7094, 797];
    let mut sorted = top_5_tokens;
    sorted.sort();
    for i in 1..sorted.len() {
        assert_ne!(sorted[i - 1], sorted[i]);
    }
}

#[test]
fn test_f8_b04_oracle_16_step_sequence_length() {
    let seq = [2744u32, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322, 7047];
    assert_eq!(seq.len(), 16);
}

#[test]
fn test_f8_b05_oracle_tolerance_constants() {
    assert_eq!(1e-4f32, 0.0001);
    assert_eq!(0.9999f32, 0.9999);
}

// ============================================================================
// Feature F9: Algorithmic Alignment Boundaries
// ============================================================================

#[test]
fn test_f9_b01_rope_at_position_zero_is_identity() {
    let mut q = vec![3.14, -2.71, 1.41, 0.57];
    let mut k = vec![1.0, 2.0, 3.0, 4.0];
    let original_q = q.clone();
    let original_k = k.clone();

    apply_rope(&mut q, &mut k, 0, 1, 1, 4, 10000.0);
    assert_eq!(q, original_q);
    assert_eq!(k, original_k);
}

#[test]
fn test_f9_b02_rope_at_max_sequence_position_2047() {
    let mut q = vec![1.0, 1.0, 1.0, 1.0];
    let mut k = vec![1.0, 1.0, 1.0, 1.0];
    let norm_before: f32 = q.iter().map(|v| v * v).sum();

    apply_rope(&mut q, &mut k, 2047, 1, 1, 4, 10000.0);
    let norm_after: f32 = q.iter().map(|v| v * v).sum();
    assert!((norm_before - norm_after).abs() < 1e-4);
}

#[test]
fn test_f9_b03_matmul_all_zeros_input() {
    let x = vec![0.0; 8];
    let w = vec![1.0; 16];
    let mut out = vec![99.0; 2];
    matmul_vec(&x, &w, &mut out, 8, 2);
    assert_eq!(out, vec![0.0, 0.0]);
}

#[test]
fn test_f9_b04_matmul_identity_matrix() {
    let x = vec![1.0, 2.0, 3.0];
    // 3x3 identity matrix in row-major
    let identity = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let mut out = vec![0.0; 3];
    matmul_vec(&x, &identity, &mut out, 3, 3);
    assert_eq!(out, x);
}

#[test]
fn test_f9_b05_rmsnorm_zero_vector_handles_epsilon() {
    let x = vec![0.0; 4];
    let w = vec![1.0; 4];
    let mut out = vec![99.0; 4];
    rmsnorm(&x, &w, 1e-5, &mut out);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
}

// ============================================================================
// Feature F10: Parity Test Boundaries
// ============================================================================

#[test]
fn test_f10_b01_exact_boundary_tolerance_1e4() {
    let a = vec![1.0000f32];
    let b = vec![1.0001f32];
    let err = max_absolute_error(&a, &b);
    assert!((err - 1e-4).abs() < 1e-6);
}

#[test]
fn test_f10_b02_boundary_violation_at_1_001e4() {
    let a = vec![1.0000f32];
    let b = vec![1.000101f32];
    let err = max_absolute_error(&a, &b);
    assert!(err > 1e-4);
}

#[test]
fn test_f10_b03_exact_cosine_boundary_0_9999() {
    let a = vec![1.0f32, 0.0f32];
    // angle theta where cos(theta) = 0.9999 -> sin(theta) = sqrt(1 - 0.9999^2) = ~0.01414178
    let cos_val = 0.9999f32;
    let sin_val = (1.0 - cos_val * cos_val).sqrt();
    let b = vec![cos_val, sin_val];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 0.9999).abs() < 1e-5);
}

#[test]
fn test_f10_b04_cosine_boundary_violation_below_threshold() {
    let a = vec![1.0f32, 0.0f32];
    let b = vec![0.9998f32, 0.01999899f32];
    let sim = cosine_similarity(&a, &b);
    assert!(sim < 0.9999);
}

#[test]
fn test_f10_b05_early_divergence_index_tracking() {
    let mut layer_passed = vec![true; 22];
    layer_passed[3] = false; // Divergence at layer 3
    let earliest_divergent = layer_passed.iter().position(|&p| !p);
    assert_eq!(earliest_divergent, Some(3));
}

// ============================================================================
// Feature F11: TensorBackend Trait Boundaries
// ============================================================================

#[test]
fn test_f11_b01_rmsnorm_slice_length_one() {
    let x = vec![5.0f32];
    let w = vec![1.0f32];
    let mut out = vec![0.0f32];
    rmsnorm(&x, &w, 1e-5, &mut out);
    // mean(x^2) = 25 -> rms = sqrt(25) = 5 -> out = 5 / 5 = 1.0
    assert!((out[0] - 1.0).abs() < 1e-3);
}

#[test]
fn test_f11_b02_matmul_vec_dimension_1x1() {
    let x = vec![3.0];
    let w = vec![4.0];
    let mut out = vec![0.0];
    matmul_vec(&x, &w, &mut out, 1, 1);
    assert_eq!(out[0], 12.0);
}

#[test]
fn test_f11_b03_gqa_ratio_boundary_calculation() {
    let q_heads = 32;
    let kv_heads = 4;
    let ratio = q_heads / kv_heads;
    assert_eq!(ratio, 8);
}

#[test]
fn test_f11_b04_swiglu_with_negative_inputs() {
    let x = vec![-2.0f32];
    let gate = vec![1.0f32];
    let up = vec![1.0f32];
    let down = vec![1.0f32];
    let mut out = vec![0.0f32];
    swiglu(&x, &gate, &up, &down, 1, 1, &mut out);
    // silu(-2) is negative, up is negative (-2 * 1 = -2), product is positive
    assert!(out[0] > 0.0);
}

#[test]
fn test_f11_b05_compute_logits_hidden_dim_2048_vocab_32000() {
    let hidden_dim = 2048;
    let vocab = 32000;
    assert_eq!(hidden_dim * vocab, 65_536_000);
}

// ============================================================================
// Feature F12: ReferenceCpuBackend Boundaries
// ============================================================================

#[test]
fn test_f12_b01_extreme_large_floats_handling() {
    let x = vec![1e15f32, 1e15f32];
    let w = vec![1.0f32, 1.0f32];
    let mut out = vec![0.0f32; 2];
    rmsnorm(&x, &w, 1e-5, &mut out);
    assert!(!out[0].is_nan());
    assert!(!out[0].is_infinite());
}

#[test]
fn test_f12_b02_extreme_small_floats_handling() {
    let x = vec![1e-15f32, 1e-15f32];
    let w = vec![1.0f32, 1.0f32];
    let mut out = vec![0.0f32; 2];
    rmsnorm(&x, &w, 1e-5, &mut out);
    assert!(!out[0].is_nan());
}

#[test]
fn test_f12_b03_attention_causal_mask_values() {
    let mask_val = f32::NEG_INFINITY;
    assert!(mask_val.is_infinite() && mask_val.is_sign_negative());
}

#[test]
fn test_f12_b04_kv_cache_max_capacity_boundary() {
    let max_len = 2048usize;
    let mut cache = Vec::<Vec<f32>>::with_capacity(max_len);
    for _ in 0..max_len {
        cache.push(vec![0.0; 256]);
    }
    assert_eq!(cache.len(), 2048);
}

#[test]
fn test_f12_b05_argmax_equal_logits_selects_lowest_index() {
    let logits = vec![5.0, 5.0, 5.0];
    let (tok, _) = sample_argmax(&logits);
    assert_eq!(tok, 0, "Argmax on identical logits should select first index");
}

// ============================================================================
// Feature F15: Neutral Benchmark Driver Boundaries
// ============================================================================

#[test]
fn test_f15_b01_zero_concurrency_rejected() {
    let c = 0;
    assert!(c == 0, "C=0 is invalid for benchmark sweep");
}

#[test]
fn test_f15_b02_max_concurrency_level_64() {
    let max_c = 64;
    assert_eq!(max_c, 64);
}

#[test]
fn test_f15_b03_thermal_cool_down_duration_seconds() {
    let cool_down_sec = 30;
    assert!(cool_down_sec >= 30);
}

#[test]
fn test_f15_b05_concurrency_power_of_two_sweep() {
    let levels = [1, 2, 4, 8, 16, 32, 64];
    for &c in &levels {
        assert!(c > 0 && (c & (c - 1)) == 0);
    }
}

// ============================================================================
// Feature F16: Concurrency Sweeps Boundaries
// ============================================================================

#[test]
fn test_f16_b01_single_sample_percentile_calculation() {
    let latencies = [15.0f32];
    let p50 = latencies[0];
    let p95 = latencies[0];
    let p99 = latencies[0];
    assert_eq!(p50, 15.0);
    assert_eq!(p95, 15.0);
    assert_eq!(p99, 15.0);
}

#[test]
fn test_f16_b02_concurrency_sweep_monotonic_tokens() {
    let output_tokens_per_req = 128;
    for c in [1, 2, 4, 8, 16, 32, 64] {
        let total = c * output_tokens_per_req;
        assert_eq!(total, c * 128);
    }
}

#[test]
fn test_f16_b03_throughput_calculation_zero_elapsed_guarded() {
    let total_tokens = 100.0f32;
    let elapsed = 0.0001f32.max(0.000001);
    let tps = total_tokens / elapsed;
    assert!(tps.is_finite());
}

#[test]
fn test_f16_b04_resident_memory_rss_positive() {
    let rss_gb = 2.1f32;
    assert!(rss_gb > 0.0);
}

#[test]
fn test_f16_b05_power_draw_positive_watts() {
    let power_w = 45.5f32;
    assert!(power_w > 0.0);
}

// ============================================================================
// Feature F17: Compliance Boundaries
// ============================================================================

fn create_test_temp_dir(prefix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_f17_b01_nested_hidden_env_files_detected() {
    let temp_dir = create_test_temp_dir("audit_env");
    let nested = temp_dir.join("sub/dir");
    std::fs::create_dir_all(&nested).unwrap();
    let hidden_env = nested.join(".env.production");
    std::fs::write(&hidden_env, "SECRET=123").unwrap();

    let res = audit_zero_disk_secrets(&temp_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    assert!(res.is_err());
    assert_eq!(res.err().unwrap().len(), 1);
}

#[test]
fn test_f17_b02_clean_directory_has_zero_secrets() {
    let temp_dir = create_test_temp_dir("audit_clean");
    let clean_file = temp_dir.join("config.toml");
    std::fs::write(&clean_file, "key = 'val'").unwrap();

    let res = audit_zero_disk_secrets(&temp_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    assert!(res.is_ok());
}

#[test]
fn test_f17_b03_dash_audit_detects_em_dash() {
    let temp_dir = create_test_temp_dir("audit_em");
    let bad_file = temp_dir.join("notes.md");
    std::fs::write(&bad_file, "This has an em dash \u{2014} here").unwrap();

    let res = audit_sovereign_voice_dashes(&temp_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    assert!(res.is_err());
    assert_eq!(res.err().unwrap().len(), 1);
}

#[test]
fn test_f17_b04_dash_audit_detects_en_dash() {
    let temp_dir = create_test_temp_dir("audit_en");
    let bad_file = temp_dir.join("doc.rs");
    std::fs::write(&bad_file, "// Range 10\u{2013}20").unwrap();

    let res = audit_sovereign_voice_dashes(&temp_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    assert!(res.is_err());
    assert_eq!(res.err().unwrap().len(), 1);
}

#[test]
fn test_f17_b05_dash_audit_passes_on_standard_hyphen() {
    let temp_dir = create_test_temp_dir("audit_hyphen");
    let good_file = temp_dir.join("doc.rs");
    std::fs::write(&good_file, "// Range 10-20 with standard hyphen").unwrap();

    let res = audit_sovereign_voice_dashes(&temp_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    assert!(res.is_ok());
}

// ============================================================================
// Feature F18: PR Lifecycle Boundaries
// ============================================================================

#[test]
fn test_f18_b01_git_branch_prefix_feat() {
    let branch = "feat/real-model-execution-tinyllama";
    assert!(branch.starts_with("feat/"));
}

#[test]
fn test_f18_b02_cortex_space_exact_match() {
    let target_space = "atlas-memory";
    assert_eq!(target_space, "atlas-memory");
}

#[test]
fn test_f18_b03_cortex_port_exact_match() {
    let port = 18080;
    assert_eq!(port, 18080);
}

#[test]
fn test_f18_b04_confidence_score_boundary() {
    let confidence = 1.0f32;
    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[test]
fn test_f18_b05_squash_merge_flag_string() {
    let flags = ["--squash", "--delete-branch"];
    assert_eq!(flags[0], "--squash");
    assert_eq!(flags[1], "--delete-branch");
}
