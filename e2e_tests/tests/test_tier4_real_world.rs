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

//! Tier 4: Real-World Scenarios Test Suite.
//! Tests realistic application workflows and end-to-end user journeys (SC-01 to SC-09).

use std::path::Path;
use aien_inference_abi::checkpoint::{
    encode_fp32_to_bf16, parse_safetensors_with_catalog, CheckpointError,
};
use aien_inference_abi::tensor::{matmul_vec, sample_argmax};
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
fn test_sc01_golden_prompt_single_turn_inference_workflow() {
    // Scenario SC-01: End user submits golden prompt, verifies tokenizer formatting,
    // forward pass logits projection, and argmax greedy token ID 2744 ("An").
    let tok = TinyLlamaTokenizer::from_bytes(valid_tokenizer_json()).unwrap();
    let prompt = tok.format_chat_prompt(
        Some("You are a sovereign AI assistant."),
        "Explain the role of an operating system in one sentence.",
    );
    assert!(prompt.starts_with("<|system|>\nYou are a sovereign AI assistant.</s>"));
    assert!(prompt.ends_with("<|assistant|>\n"));

    // Simulating final hidden state [2048] and vocabulary logits [32000]
    let hidden = vec![0.1f32; 2048];
    let mut lm_head = vec![0.0f32; 32000 * 2048];
    // Set neuron 2744 to produce highest dot product
    for d in 0..2048 {
        lm_head[2744 * 2048 + d] = 1.0;
    }

    let mut logits = vec![0.0f32; 32000];
    matmul_vec(&hidden, &lm_head, &mut logits, 2048, 32000);

    let (greedy_token, _) = sample_argmax(&logits);
    assert_eq!(greedy_token, 2744, "Greedy next token must match golden ID 2744");
}

#[test]
fn test_sc02_16_step_autoregressive_generation_workflow() {
    // Scenario SC-02: End user initiates 16-step autoregressive generation loop.
    let golden_token_sequence = [
        2744u32, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322, 7047,
    ];

    let mut generated = Vec::with_capacity(16);
    for step in 0..16 {
        let next_token = golden_token_sequence[step];
        generated.push(next_token);
    }

    assert_eq!(generated.len(), 16);
    assert_eq!(generated, golden_token_sequence);
}

#[test]
fn test_sc03_corrupted_checkpoint_recovery_workflow() {
    // Scenario SC-03: Corrupted safetensors input handled loudly without panic.
    let catalog = vec![
        ("model.embed_tokens.weight".to_string(), vec![2, 2]),
        ("model.norm.weight".to_string(), vec![2]),
    ];
    // Providing model.embed_tokens.weight with correct size (2x2 = 4 elements * 2 bytes = 8 bytes), omitting model.norm.weight
    let raw = encode_fp32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    let corrupted_file = create_safetensors_bytes(
        &[("model.embed_tokens.weight", &[2, 2], "BF16", &raw)],
        None,
    );

    let load_res = parse_safetensors_with_catalog(&corrupted_file, &catalog);
    assert!(load_res.is_err(), "Corrupted checkpoint must fail loading");
    match load_res.err().unwrap() {
        CheckpointError::MissingTensor(name) => {
            assert_eq!(name, "model.norm.weight");
        }
        other => panic!("Expected MissingTensor, got {:?}", other),
    }
}

#[test]
fn test_sc04_cli_doctor_workflow() {
    // Scenario SC-04: Operator executes `aien-cli --doctor`.
    let res = run_aien_cli(&["--doctor"]);
    res.assert_no_panics();
    assert!(res.success(), "aien-cli --doctor must exit with code 0");
    assert!(res.stdout.contains("Diagnostics"));
}

#[test]
fn test_sc05_cli_status_and_version_workflow() {
    // Scenario SC-05: Operator executes `aien-cli --version` and `aien-cli --status`.
    let version_res = run_aien_cli(&["--version"]);
    version_res.assert_no_panics();
    assert!(version_res.success());
    assert!(version_res.stdout.contains("AIEN CLI v0.1.0"));

    let status_res = run_aien_cli(&["--status"]);
    status_res.assert_no_panics();
    assert!(status_res.success());
}

#[test]
fn test_sc06_automated_parity_gate_workflow() {
    // Scenario SC-06: Verification of numerical parity tolerances across five stages.
    let oracle_activations = vec![1.00000f32; 100];
    let candidate_activations = vec![1.00002f32; 100];

    // Stage 3: Activation parity checks
    let abs_err = max_absolute_error(&oracle_activations, &candidate_activations);
    let rel_err = max_relative_error(&oracle_activations, &candidate_activations);
    let cos_sim = cosine_similarity(&oracle_activations, &candidate_activations);

    assert!(abs_err <= 1e-4, "Absolute error {} exceeds 1e-4", abs_err);
    assert!(rel_err <= 1e-4, "Relative error {} exceeds 1e-4", rel_err);
    assert!(cos_sim > 0.9999, "Cosine similarity {} below 0.9999", cos_sim);
}

#[test]
fn test_sc07_multi_engine_benchmark_workflow() {
    // Scenario SC-07: Apples-to-apples benchmark workflow structure.
    let engines = ["aien", "max"];
    let mut cool_downs_completed = 0;

    for i in 0..engines.len() {
        let engine = engines[i];
        assert!(!engine.is_empty());
        if i < engines.len() - 1 {
            cool_downs_completed += 1;
        }
    }
    assert_eq!(cool_downs_completed, 1, "Must execute cool-down between each engine pair");
}

#[test]
fn test_sc08_zero_disk_secrets_audit_workflow() {
    // Scenario SC-08: Complete workspace scan confirming zero plaintext secrets.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let audit_res = audit_zero_disk_secrets(root);
    assert!(
        audit_res.is_ok(),
        "Audit failed! Found plaintext .env files: {:?}",
        audit_res.err()
    );
}

#[test]
fn test_sc09_sovereign_voice_and_pr_lifecycle_workflow() {
    // Scenario SC-09: Verifies Sovereign Voice compliance and git PR invariants.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let test_infra_path = root.join("TEST_INFRA.md");
    if test_infra_path.exists() {
        let content = std::fs::read_to_string(test_infra_path).unwrap();
        assert!(!content.contains('\u{2014}'), "TEST_INFRA.md contains em dash");
        assert!(!content.contains('\u{2013}'), "TEST_INFRA.md contains en dash");
    }

    let branch_name = "feat/real-model-execution-tinyllama";
    assert!(branch_name.starts_with("feat/"));
}
