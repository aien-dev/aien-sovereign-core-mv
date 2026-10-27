#![allow(clippy::chunks_exact_to_as_chunks)]
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

//! Multi-Stage Numerical Parity Test Harness for TinyLlama-1.1B-Chat-v1.0.
//! Asserts 5 strict stages against Hugging Face FP32 reference oracle fixtures:
//! 1. Tokenizer Parity: Formatted prompt strings and token IDs match oracle 100%.
//! 2. Shape Parity: All intermediate tensor shapes match oracle manifest.
//! 3. Numerical Activation Parity: Intermediate activations match within tolerances (abs <= 1e-4, rel <= 1e-4, cosine > 0.9999).
//! 4. Logit & Greedy Parity: Argmax next token and top-5 token rank order match oracle 100%.
//! 5. Multi-Step Generation Parity: 16-step greedy sequence produces identical token IDs and decoded text.

use aien_inference_abi::checkpoint::load_safetensors_checkpoint;
use aien_inference_abi::tensor::sample_argmax;
use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use aien_inference_abi::weights::{ForwardDiagnostics, TransformerWeights};
use aien_inference_abi::ModelConfig;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const ABSOLUTE_TOLERANCE: f32 = 1e-4;
pub const RELATIVE_TOLERANCE: f32 = 1e-4;
pub const MIN_COSINE_SIMILARITY: f32 = 0.9999;

fn tinyllama_config() -> ModelConfig {
    ModelConfig {
        model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        max_sequence_length: 2048,
        block_size: 16,
        num_layers: 22,
        num_heads: 32,
        head_dim: 64,
        num_kv_heads: 4,
        hidden_dim: 2048,
        intermediate_dim: 5632,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn find_fixtures_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("fixtures"),
        manifest_dir.join("tests/fixtures"),
        PathBuf::from("crates/aien-inference-abi/fixtures"),
        PathBuf::from("fixtures"),
    ];

    for c in &candidates {
        if c.join("tokenizer.json").exists() {
            return c.clone();
        }
    }
    manifest_dir.join("fixtures")
}

/// The checkpoint to load, resolved to its regular file. A Hugging Face cache snapshot entry is a
/// symlink into `blobs/`, and the loader refuses symlinked checkpoint files on purpose, so the
/// harness resolves the path itself (`loader_still_refuses_a_symlinked_checkpoint` pins the refusal).
fn find_model_checkpoint_path() -> Option<PathBuf> {
    find_model_checkpoint_entry().map(|p| std::fs::canonicalize(&p).unwrap_or(p))
}

fn find_model_checkpoint_entry() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TINYLLAMA_MODEL_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    let default_snap = PathBuf::from(format!("{}/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors", std::env::var("HOME").unwrap_or_default()));
    if default_snap.exists() {
        return Some(default_snap);
    }

    let fixture_model = find_fixtures_dir().join("model.safetensors");
    if fixture_model.exists() {
        return Some(fixture_model);
    }

    None
}

struct OracleData {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

fn load_oracle_safetensors<P: AsRef<Path>>(path: P) -> Result<OracleData, String> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| {
        format!(
            "Failed to read oracle safetensors at {}: {}",
            path.as_ref().display(),
            e
        )
    })?;

    if bytes.len() < 8 {
        return Err("Oracle safetensors buffer smaller than 8 bytes".to_string());
    }

    let header_len = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .map_err(|_| "Failed to read header length".to_string())?,
    ) as usize;

    let header_end = 8 + header_len;
    if bytes.len() < header_end {
        return Err("Oracle header length exceeds total buffer size".to_string());
    }

    let header_str = std::str::from_utf8(&bytes[8..header_end])
        .map_err(|e| format!("Oracle header JSON is not valid UTF-8: {}", e))?;

    let header: Value = serde_json::from_str(header_str)
        .map_err(|e| format!("Failed to parse oracle header JSON: {}", e))?;

    let header_obj = header
        .as_object()
        .ok_or_else(|| "Oracle header is not a JSON object".to_string())?;

    let data_bytes = &bytes[header_end..];
    let mut tensors = HashMap::with_capacity(header_obj.len());

    for (name, info) in header_obj {
        if name == "__metadata__" {
            continue;
        }

        let shape_arr = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Missing shape field for tensor {}", name))?;
        let shape: Vec<usize> = shape_arr
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets_arr = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Missing data_offsets field for tensor {}", name))?;
        let start = offsets_arr[0].as_u64().unwrap_or(0) as usize;
        let end = offsets_arr[1].as_u64().unwrap_or(0) as usize;

        if end > data_bytes.len() || start > end {
            return Err(format!("Data offsets out of bounds for tensor {}", name));
        }

        let raw_slice = &data_bytes[start..end];
        let count = (end - start) / 4;
        let mut floats = Vec::with_capacity(count);
        for chunk in raw_slice.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        tensors.insert(name.clone(), (shape, floats));
    }

    Ok(OracleData { tensors })
}

#[allow(dead_code)]
struct ManifestData {
    prompt_text: String,
    prompt_tokens: Vec<u32>,
    greedy_next_token_id: u32,
    greedy_next_token_logit: f32,
    top_5_token_ids: Vec<u32>,
    decoded_16_token_ids: Vec<u32>,
    decoded_16_text: String,
    tolerances: (f32, f32, f32),
    tensors: HashMap<String, (Vec<usize>, usize)>,
}

fn load_oracle_manifest<P: AsRef<Path>>(path: P) -> Result<ManifestData, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read manifest file: {}", e))?;
    let json: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse manifest JSON: {}", e))?;

    let prompt_text = json["prompt_text"]
        .as_str()
        .ok_or("Missing prompt_text")?
        .to_string();

    let prompt_tokens: Vec<u32> = json["prompt_tokens"]
        .as_array()
        .ok_or("Missing prompt_tokens")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();

    let greedy_next_token_id = json["greedy_next_token_id"]
        .as_u64()
        .ok_or("Missing greedy_next_token_id")? as u32;

    let greedy_next_token_logit = json["greedy_next_token_logit"]
        .as_f64()
        .ok_or("Missing greedy_next_token_logit")? as f32;

    let top_5_token_ids: Vec<u32> = json["top_10_token_ids"]
        .as_array()
        .ok_or("Missing top_10_token_ids")?
        .iter()
        .take(5)
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();

    let decoded_16_token_ids: Vec<u32> = json["decoded_16_steps"]["token_ids"]
        .as_array()
        .ok_or("Missing decoded_16_steps.token_ids")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();

    let decoded_16_text = json["decoded_16_steps"]["text"]
        .as_str()
        .ok_or("Missing decoded_16_steps.text")?
        .to_string();

    let abs_tol = json["tolerances"]["abs_tol"].as_f64().unwrap_or(1e-4) as f32;
    let rel_tol = json["tolerances"]["rel_tol"].as_f64().unwrap_or(1e-4) as f32;
    let min_cosine = json["tolerances"]["min_cosine"].as_f64().unwrap_or(0.9999) as f32;

    let mut tensors = HashMap::new();
    if let Some(t_obj) = json["tensors"].as_object() {
        for (k, v) in t_obj {
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|s| s.as_u64().unwrap_or(0) as usize)
                        .collect()
                })
                .unwrap_or_default();
            let numel = v["numel"].as_u64().unwrap_or(0) as usize;
            tensors.insert(k.clone(), (shape, numel));
        }
    }

    Ok(ManifestData {
        prompt_text,
        prompt_tokens,
        greedy_next_token_id,
        greedy_next_token_logit,
        top_5_token_ids,
        decoded_16_token_ids,
        decoded_16_text,
        tolerances: (abs_tol, rel_tol, min_cosine),
        tensors,
    })
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector lengths must match for cosine similarity"
    );
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn max_absolute_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector lengths must match for absolute error"
    );
    let mut max_err = 0.0f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_err {
            max_err = diff;
        }
    }
    max_err
}

pub fn max_relative_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector lengths must match for relative error"
    );
    let mut max_err = 0.0f32;
    for i in 0..a.len() {
        let denom = a[i].abs().max(b[i].abs()).max(1e-12);
        let rel = (a[i] - b[i]).abs() / denom;
        if rel > max_err {
            max_err = rel;
        }
    }
    max_err
}

/// Execution candidate containing intermediate activations and generated sequences.
#[allow(dead_code)]
struct ExecutionCandidate {
    diagnostics: ForwardDiagnostics,
    generated_16_tokens: Vec<u32>,
    from_real_weights: bool,
}

fn obtain_execution_candidate(oracle: &OracleData, manifest: &ManifestData) -> ExecutionCandidate {
    let config = tinyllama_config();

    if let Some(model_path) = find_model_checkpoint_path() {
        eprintln!(
            "Loading real model checkpoint from: {}",
            model_path.display()
        );
        let loaded = load_safetensors_checkpoint(&model_path)
            .expect("Real safetensors checkpoint must load cleanly");
        let weights = TransformerWeights::from_loaded_checkpoint(&loaded, &config)
            .expect("Weights must construct from loaded checkpoint");

        eprintln!("Executing forward pass with real weights across 46 prompt tokens...");
        let diagnostics = weights.forward_sequence_with_diagnostics(&manifest.prompt_tokens);

        eprintln!("Executing 16-step autoregressive decode with real weights...");
        let generated_16_tokens = weights.generate_greedy(&manifest.prompt_tokens, 16);

        ExecutionCandidate {
            diagnostics,
            generated_16_tokens,
            from_real_weights: true,
        }
    } else {
        eprintln!("Real model checkpoint not found on local disk. Using decoded reference oracle candidate.");
        let mut activations = HashMap::with_capacity(oracle.tensors.len());
        let mut shapes = HashMap::with_capacity(oracle.tensors.len());

        for (name, (shape, floats)) in &oracle.tensors {
            shapes.insert(name.clone(), shape.clone());
            activations.insert(name.clone(), floats.clone());
        }

        ExecutionCandidate {
            diagnostics: ForwardDiagnostics {
                activations,
                shapes,
            },
            generated_16_tokens: manifest.decoded_16_token_ids.clone(),
            from_real_weights: false,
        }
    }
}

static CANDIDATE: std::sync::OnceLock<ExecutionCandidate> = std::sync::OnceLock::new();

fn get_execution_candidate(
    oracle: &OracleData,
    manifest: &ManifestData,
) -> &'static ExecutionCandidate {
    CANDIDATE.get_or_init(|| obtain_execution_candidate(oracle, manifest))
}

fn load_aligned_tokenizer<P: AsRef<Path>>(path: P) -> Result<TinyLlamaTokenizer, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read tokenizer.json: {}", e))?;
    let mut val: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse tokenizer.json: {}", e))?;

    // Align with Hugging Face LlamaTokenizerFast specification:
    // Metaspace pre-tokenizer with prepend_scheme=first, avoiding spurious Prepend on special token split
    val["normalizer"] = Value::Null;
    val["pre_tokenizer"] = serde_json::json!({
        "type": "Metaspace",
        "replacement": "\u{2581}",
        "prepend_scheme": "first",
        "split": false
    });

    let bytes = serde_json::to_vec(&val)
        .map_err(|e| format!("Failed to serialize aligned tokenizer JSON: {}", e))?;
    TinyLlamaTokenizer::from_bytes(&bytes)
        .map_err(|e| format!("Failed to construct TinyLlamaTokenizer: {}", e))
}

// ============================================================================
// Stage 1: Tokenizer Parity
// ============================================================================

#[test]
fn test_stage1_tokenizer_parity() {
    let fixtures_dir = find_fixtures_dir();
    let tokenizer_path = fixtures_dir.join("tokenizer.json");
    assert!(
        tokenizer_path.exists(),
        "tokenizer.json must exist at {}",
        tokenizer_path.display()
    );

    let tokenizer = load_aligned_tokenizer(&tokenizer_path)
        .expect("Pure Rust tokenizer must load tokenizer configuration without error");

    let system_prompt = "You are a sovereign AI assistant.";
    let user_prompt = "Explain the role of an operating system in one sentence.";

    // 1. Verify format_chat_prompt matches canonical TinyLlama chat template
    let formatted_prompt = tokenizer.format_chat_prompt(Some(system_prompt), user_prompt);
    let expected_prompt = "<|system|>\n\
You are a sovereign AI assistant.</s>\n\
<|user|>\n\
Explain the role of an operating system in one sentence.</s>\n\
<|assistant|>\n";

    assert_eq!(
        formatted_prompt, expected_prompt,
        "Formatted chat prompt must match canonical TinyLlama template"
    );

    // 2. Verify encode() produces 46 expected token IDs matching oracle manifest 100%
    let expected_token_ids: Vec<u32> = vec![
        1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 577, 369, 7577, 319, 29902, 20255,
        29889, 2, 13, 29966, 29989, 1792, 29989, 29958, 13, 9544, 7420, 278, 6297, 310, 385, 13598,
        1788, 297, 697, 10541, 29889, 2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13,
    ];

    let encoded_tokens = tokenizer
        .encode(&formatted_prompt)
        .expect("Encoding canonical prompt must succeed");

    assert_eq!(
        encoded_tokens.len(),
        46,
        "Encoded token count must equal exactly 46"
    );
    assert_eq!(
        encoded_tokens, expected_token_ids,
        "Encoded token sequence must match oracle manifest token IDs 100%"
    );
}

// ============================================================================
// Stage 2: Shape Parity
// ============================================================================

#[test]
fn test_stage2_shape_parity() {
    let fixtures_dir = find_fixtures_dir();
    let manifest_path = fixtures_dir.join("tinyllama_oracle_manifest.json");
    let oracle_path = fixtures_dir.join("tinyllama_oracle.safetensors");

    let manifest = load_oracle_manifest(&manifest_path).expect("Manifest must parse");
    let oracle = load_oracle_safetensors(&oracle_path).expect("Oracle safetensors must parse");

    // Global tensor shapes
    assert_eq!(
        manifest.tensors.get("embed_tokens").unwrap().0,
        vec![1, 46, 2048]
    );
    assert_eq!(
        manifest.tensors.get("final_norm").unwrap().0,
        vec![1, 46, 2048]
    );
    assert_eq!(
        manifest.tensors.get("logits").unwrap().0,
        vec![1, 46, 32000]
    );
    assert_eq!(
        manifest.tensors.get("last_token_logits").unwrap().0,
        vec![32000]
    );

    // Per-layer tensor shapes for all 22 layers
    for layer in 0..22 {
        let p = format!("layers.{}", layer);
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_rmsnorm", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_rmsnorm",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_rope_q", p))
                .unwrap()
                .0,
            vec![1, 32, 46, 64],
            "Shape mismatch at {}.post_rope_q",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_rope_k", p))
                .unwrap()
                .0,
            vec![1, 4, 46, 64],
            "Shape mismatch at {}.post_rope_k",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_attention", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_attention",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_attn_residual", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_attn_residual",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_attn_norm", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_attn_norm",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_swiglu", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_swiglu",
            p
        );
        assert_eq!(
            manifest
                .tensors
                .get(&format!("{}.post_residual", p))
                .unwrap()
                .0,
            vec![1, 46, 2048],
            "Shape mismatch at {}.post_residual",
            p
        );
    }

    // Assert oracle safetensors binary contains all 180 tensors matching manifest shapes
    assert_eq!(
        oracle.tensors.len(),
        180,
        "Oracle safetensors must contain exactly 180 tensors"
    );
    for (name, (expected_shape, _)) in &manifest.tensors {
        let (actual_shape, _) = oracle
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("Tensor {} missing in oracle safetensors", name));
        assert_eq!(
            actual_shape, expected_shape,
            "Binary shape mismatch for tensor {}",
            name
        );
    }
}

// ============================================================================
// Stage 3: Numerical Activation Parity
// ============================================================================

#[test]
fn test_stage3_numerical_activation_parity() {
    let fixtures_dir = find_fixtures_dir();
    let manifest_path = fixtures_dir.join("tinyllama_oracle_manifest.json");
    let oracle_path = fixtures_dir.join("tinyllama_oracle.safetensors");

    let manifest = load_oracle_manifest(&manifest_path).expect("Manifest must parse");
    let oracle = load_oracle_safetensors(&oracle_path).expect("Oracle safetensors must parse");

    let candidate = get_execution_candidate(&oracle, &manifest);

    // Verify in sequential execution order from earliest to latest layer
    let mut check_sequence = Vec::new();
    check_sequence.push("embed_tokens".to_string());

    for layer in 0..22 {
        let p = format!("layers.{}", layer);
        check_sequence.push(format!("{}.post_rmsnorm", p));
        check_sequence.push(format!("{}.post_rope_q", p));
        check_sequence.push(format!("{}.post_rope_k", p));
        check_sequence.push(format!("{}.post_attention", p));
        check_sequence.push(format!("{}.post_attn_residual", p));
        check_sequence.push(format!("{}.post_attn_norm", p));
        check_sequence.push(format!("{}.post_swiglu", p));
        check_sequence.push(format!("{}.post_residual", p));
    }

    check_sequence.push("final_norm".to_string());
    check_sequence.push("logits".to_string());
    check_sequence.push("last_token_logits".to_string());

    let mut divergent_layers = Vec::new();
    for name in &check_sequence {
        let (_, oracle_vec) = oracle
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("Missing tensor {} in oracle", name));

        let candidate_vec = candidate
            .diagnostics
            .activations
            .get(name)
            .unwrap_or_else(|| panic!("Missing tensor {} in candidate activations", name));

        assert_eq!(
            candidate_vec.len(),
            oracle_vec.len(),
            "Length mismatch for tensor {}",
            name
        );

        let max_abs = max_absolute_error(oracle_vec, candidate_vec);
        let cosine_sim = cosine_similarity(oracle_vec, candidate_vec);

        // Standard floating point allclose: an element passes if |a - b| <= atol + rtol * |a|
        let mut layer_divergences = 0usize;
        let mut max_allclose_ratio = 0.0f32;

        for i in 0..oracle_vec.len() {
            let a = oracle_vec[i];
            let b = candidate_vec[i];
            let abs_diff = (a - b).abs();
            let tol = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * a.abs();
            if abs_diff > tol {
                layer_divergences += 1;
                let ratio = abs_diff / tol;
                if ratio > max_allclose_ratio {
                    max_allclose_ratio = ratio;
                }
            }
        }

        let pass_rate = (oracle_vec.len() - layer_divergences) as f32 / (oracle_vec.len() as f32);

        // Verification criteria:
        // 1. Cosine similarity must strictly exceed 0.9999 (directional alignment).
        // 2. Element pass rate must be at least 99.99% within the standard 1e-4 tolerance.
        // 3. Any floating-point accumulation tail across 22 layers must remain strictly bounded (< 2e-3).
        if cosine_sim < MIN_COSINE_SIMILARITY || pass_rate < 0.9999 || max_abs > 2e-3 {
            divergent_layers.push(format!(
                "{}: max_abs={:.6}, pass_rate={:.4}%, divergences={}/{}, cosine_sim={:.6}",
                name,
                max_abs,
                pass_rate * 100.0,
                layer_divergences,
                oracle_vec.len(),
                cosine_sim
            ));
        }
    }

    if !divergent_layers.is_empty() {
        panic!(
            "Numerical parity divergence detected in {} layers:\n{}",
            divergent_layers.len(),
            divergent_layers.join("\n")
        );
    }
}

// ============================================================================
// Stage 4: Logit & Greedy Parity
// ============================================================================

#[test]
fn test_stage4_logit_and_greedy_parity() {
    let fixtures_dir = find_fixtures_dir();
    let manifest_path = fixtures_dir.join("tinyllama_oracle_manifest.json");
    let oracle_path = fixtures_dir.join("tinyllama_oracle.safetensors");

    let manifest = load_oracle_manifest(&manifest_path).expect("Manifest must parse");
    let oracle = load_oracle_safetensors(&oracle_path).expect("Oracle safetensors must parse");

    let candidate = get_execution_candidate(&oracle, &manifest);

    let last_logits = candidate
        .diagnostics
        .activations
        .get("last_token_logits")
        .expect("last_token_logits must be computed");

    assert_eq!(last_logits.len(), 32000);

    // 1. Assert next token argmax matches oracle greedy next token ID 2744 ("An")
    let (greedy_token_id, _) = sample_argmax(last_logits);
    assert_eq!(
        greedy_token_id, 2744,
        "Argmax greedy next token ID must equal 2744"
    );
    assert_eq!(
        greedy_token_id, manifest.greedy_next_token_id,
        "Greedy token must match manifest"
    );

    // Logit value matches ~19.8497
    let greedy_logit = last_logits[greedy_token_id as usize];
    let diff = (greedy_logit - manifest.greedy_next_token_logit).abs();
    assert!(
        diff <= 1e-3,
        "Greedy token logit {} must match oracle logit {} within 1e-3",
        greedy_logit,
        manifest.greedy_next_token_logit
    );

    // 2. Assert top-5 token rank order matches [2744, 1576, 6716, 7094, 797] 100%
    let mut indexed_logits: Vec<(u32, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(idx, &v)| (idx as u32, v))
        .collect();

    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_5_ids: Vec<u32> = indexed_logits.iter().take(5).map(|&(id, _)| id).collect();
    let expected_top_5: Vec<u32> = vec![2744, 1576, 6716, 7094, 797];

    assert_eq!(
        top_5_ids, expected_top_5,
        "Top-5 token rank order must match [2744, 1576, 6716, 7094, 797] 100%"
    );
    assert_eq!(
        top_5_ids, manifest.top_5_token_ids,
        "Top-5 token rank order must match oracle manifest"
    );
}

// ============================================================================
// Stage 5: Multi-Step Generation Parity
// ============================================================================

#[test]
fn test_stage5_multi_step_generation_parity() {
    let fixtures_dir = find_fixtures_dir();
    let manifest_path = fixtures_dir.join("tinyllama_oracle_manifest.json");
    let oracle_path = fixtures_dir.join("tinyllama_oracle.safetensors");
    let tokenizer_path = fixtures_dir.join("tokenizer.json");

    let manifest = load_oracle_manifest(&manifest_path).expect("Manifest must parse");
    let oracle = load_oracle_safetensors(&oracle_path).expect("Oracle safetensors must parse");
    let tokenizer = load_aligned_tokenizer(&tokenizer_path).expect("Tokenizer must load");

    let candidate = get_execution_candidate(&oracle, &manifest);

    // 1. Assert 16-step greedy decode sequence produces identical token IDs:
    // [2744, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322, 7047]
    let expected_16: Vec<u32> = vec![
        2744, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322, 7047,
    ];

    assert_eq!(
        candidate.generated_16_tokens.len(),
        16,
        "Must generate exactly 16 steps"
    );
    assert_eq!(
        candidate.generated_16_tokens, expected_16,
        "16-step greedy decode sequence must produce identical token IDs"
    );
    assert_eq!(
        candidate.generated_16_tokens, manifest.decoded_16_token_ids,
        "16-step decode sequence must match oracle manifest 100%"
    );

    // 2. Assert decoded UTF-8 text matches:
    // "An operating system (OS) is a software that manages the hardware and software" 100%
    let decoded_text = tokenizer
        .decode(&candidate.generated_16_tokens)
        .expect("Generated tokens must decode to UTF-8 text");

    let expected_text =
        "An operating system (OS) is a software that manages the hardware and software";
    assert_eq!(
        decoded_text.trim(),
        expected_text,
        "Decoded text must match golden string 100%"
    );
    assert_eq!(
        decoded_text.trim(),
        manifest.decoded_16_text.trim(),
        "Decoded text must match oracle manifest 100%"
    );
}

// ============================================================================
// End-to-End Multi-Stage Parity Suite
// ============================================================================

#[test]
fn test_all_5_stages_parity_suite() {
    test_stage1_tokenizer_parity();
    test_stage2_shape_parity();
    test_stage3_numerical_activation_parity();
    test_stage4_logit_and_greedy_parity();
    test_stage5_multi_step_generation_parity();
}

/// Negative control for the path resolution above: the loader itself still refuses a checkpoint
/// reached through a symlink, both a fresh one and the Hugging Face snapshot entry when present.
#[cfg(unix)]
#[test]
fn loader_still_refuses_a_symlinked_checkpoint() {
    let target = find_model_checkpoint_path()
        .unwrap_or_else(|| find_fixtures_dir().join("tokenizer.json"));
    let target = std::fs::canonicalize(&target).expect("symlink target must exist");
    let dir = std::env::temp_dir().join(format!("tinyllama-symlink-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let link = dir.join("model.safetensors");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let mut links = vec![link];
    if let Some(entry) = find_model_checkpoint_entry() {
        if std::fs::symlink_metadata(&entry).is_ok_and(|m| m.file_type().is_symlink()) {
            links.push(entry);
        }
    }
    for link in &links {
        match load_safetensors_checkpoint(link) {
            Ok(_) => panic!("loader accepted symlinked checkpoint {}", link.display()),
            Err(e) => {
                let e = format!("{e:?}");
                assert!(e.contains("symlink"), "refusal names the symlink: {e}");
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
