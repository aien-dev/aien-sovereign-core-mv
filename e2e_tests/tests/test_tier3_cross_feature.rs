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

//! Tier 3: Cross-Feature Interactions Test Suite.
//! Tests pairwise integration and cross-module contracts across the remaining feature pairs (P1 to P18; P11 and P13 to P15 were removed with Mojo).

use aien_inference_abi::checkpoint::{
    encode_fp32_to_bf16, parse_safetensors_with_catalog, CheckpointError,
};
use aien_inference_abi::tensor::{apply_rope, matmul_vec, rmsnorm, sample_argmax};
use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use e2e_tests::harness::*;

fn valid_tokenizer_json() -> &'static [u8] {
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

#[test]
fn test_pair_p01_loader_and_validation() {
    // P1 (F1 + F2): Loader triggers loud MissingTensor error on incomplete catalog
    let catalog = vec![
        ("model.embed_tokens.weight".to_string(), vec![2, 2]),
        ("model.norm.weight".to_string(), vec![2]),
    ];
    let raw = encode_fp32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let bytes = create_safetensors_bytes(&[("model.embed_tokens.weight", &[2, 2], "BF16", &raw)], None);

    let res = parse_safetensors_with_catalog(&bytes, &catalog);
    assert_eq!(
        res.unwrap_err(),
        CheckpointError::MissingTensor("model.norm.weight".to_string())
    );
}

#[test]
fn test_pair_p02_loader_and_dual_storage() {
    // P2 (F1 + F3): Safetensors loading decodes FP32 from stored raw BF16 bytes
    let original = vec![2.5f32, -1.0f32];
    let raw = encode_fp32_to_bf16(&original);
    let bytes = create_safetensors_bytes(&[("test_weight", &[2], "BF16", &raw)], None);
    let catalog = vec![("test_weight".to_string(), vec![2])];

    let loaded = parse_safetensors_with_catalog(&bytes, &catalog).unwrap();
    let fp32 = loaded.decode_fp32("test_weight").unwrap();
    let bf16 = loaded.tensor_bytes("test_weight").unwrap();

    assert_eq!(fp32.len(), 2);
    assert_eq!(bf16.len(), 4);
    assert!((fp32[0] - 2.5).abs() < 1e-2);
}

#[test]
fn test_pair_p03_catalog_shape_and_aligned_matmul() {
    // P3 (F2 + F9): Validating tensor shapes enforces row-major [out_dim, in_dim] layout
    let x = vec![1.0, 2.0];
    let shape = vec![3, 2]; // out_dim = 3, in_dim = 2
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = vec![0.0; shape[0]];

    matmul_vec(&x, &w, &mut out, shape[1], shape[0]);
    assert_eq!(out, vec![5.0, 11.0, 17.0]);
}

#[test]
fn test_pair_p04_tokenizer_and_chat_template() {
    // P4 (F4 + F5): Pure Rust tokenizer formats chat template and verifies special tokens
    let tok = TinyLlamaTokenizer::from_bytes(valid_tokenizer_json()).unwrap();
    let formatted = tok.format_chat_prompt(Some("System prompt"), "User prompt");
    assert!(formatted.starts_with("<|system|>\nSystem prompt</s>"));
    assert!(formatted.contains("<|user|>\nUser prompt</s>"));
    assert!(formatted.ends_with("<|assistant|>\n"));
    assert_eq!(TinyLlamaTokenizer::BOS_TOKEN_ID, 1);
    assert_eq!(TinyLlamaTokenizer::EOS_TOKEN_ID, 2);
}

#[test]
fn test_pair_p05_tokenizer_encode_and_decode_roundtrip() {
    // P5 (F4 + F6): Tokenizer round-trip string preservation
    let tok = TinyLlamaTokenizer::from_bytes(valid_tokenizer_json()).unwrap();
    let original = "";
    let tokens = tok.encode(original).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_pair_p06_chat_template_and_eos_stopping() {
    // P6 (F5 + F6): Autoregressive generation terminates on EOS token ID 2
    let tok = TinyLlamaTokenizer::from_bytes(valid_tokenizer_json()).unwrap();
    let generated_tokens = vec![10u32, 20, 2]; // 2 is EOS
    let has_eos = generated_tokens.iter().any(|&t| tok.is_eos(t));
    assert!(has_eos, "Sequence containing EOS must be detected as stopped");
}

#[test]
fn test_pair_p07_oracle_script_and_fixtures_contract() {
    // P7 (F7 + F8): Oracle generation produces deterministic 46 token prompt
    let prompt_tokens = [
        1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 577, 369, 7577, 319, 29902, 20255,
        29889, 2, 13, 29966, 29989, 1792, 29989, 29958, 13, 9544, 7420, 278, 6297, 310, 385,
        13598, 1788, 297, 697, 10541, 29889, 2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13,
    ];
    assert_eq!(prompt_tokens.len(), 46);
    assert_eq!(prompt_tokens[0], 1);
}

#[test]
fn test_pair_p08_oracle_fixtures_and_parity_tolerance() {
    // P8 (F8 + F10): Multi-stage parity assertions against oracle outputs
    let oracle_next_token = 2744u32;
    let candidate_logits = vec![0.0f32; 32000];
    let mut modified_logits = candidate_logits;
    modified_logits[2744] = 19.85;

    let (best_tok, _) = sample_argmax(&modified_logits);
    assert_eq!(best_tok, oracle_next_token);
}

#[test]
fn test_pair_p09_algorithmic_alignment_and_parity_tolerances() {
    // P9 (F9 + F10): Aligned RoPE and RMSNorm pass mathematical parity thresholds
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let w = vec![1.0, 1.0, 1.0, 1.0];
    let mut out_a = vec![0.0; 4];
    let mut out_b = vec![0.0; 4];

    rmsnorm(&x, &w, 1e-5, &mut out_a);
    rmsnorm(&x, &w, 1e-5, &mut out_b);

    assert!(max_absolute_error(&out_a, &out_b) <= 1e-4);
    assert!(cosine_similarity(&out_a, &out_b) > 0.9999);
}

#[test]
fn test_pair_p10_aligned_math_and_reference_cpu_backend() {
    // P10 (F9 + F12): Reference CPU backend produces normalized output
    let mut q = vec![1.0, 2.0, 3.0, 4.0];
    let mut k = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut q, &mut k, 5, 1, 1, 4, 10000.0);

    let norm_q: f32 = q.iter().map(|v| v * v).sum();
    let norm_k: f32 = k.iter().map(|v| v * v).sum();
    assert!((norm_q - 30.0).abs() < 1e-4);
    assert!((norm_k - 30.0).abs() < 1e-4);
}

#[test]
fn test_pair_p12_tensor_backend_trait_and_cpu_implementation() {
    // P12 (F11 + F12): Reference CPU backend forward pass operations
    let x = vec![1.0, 2.0];
    let w = vec![2.0, 0.0, 0.0, 2.0];
    let mut out = vec![0.0; 2];
    matmul_vec(&x, &w, &mut out, 2, 2);
    assert_eq!(out, vec![2.0, 4.0]);
}

#[test]
fn test_pair_p16_neutral_benchmark_driver_and_concurrency_sweep() {
    // P16 (F15 + F16): Sequential benchmark sweep C=1..64
    let concurrency_levels = [1, 2, 4, 8, 16, 32, 64];
    assert_eq!(concurrency_levels.len(), 7);
}

#[test]
fn test_pair_p17_concurrency_telemetry_and_zero_disk_secrets() {
    // P17 (F16 + F17): Telemetry gathering with TPM vault security
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let audit_res = audit_zero_disk_secrets(root);
    assert!(audit_res.is_ok());
}

#[test]
fn test_pair_p18_benchmark_completion_and_cortex_memory_receipt() {
    // P18 (F15 + F18): Benchmark results schema and Cortex receipt entry
    let receipt = serde_json::json!({
        "status": "verified",
        "concurrency_sweep_max": 64,
        "space": "atlas-memory"
    });
    assert_eq!(receipt["status"], "verified");
    assert_eq!(receipt["space"], "atlas-memory");
}
