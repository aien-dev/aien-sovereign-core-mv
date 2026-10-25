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

//! Untied output head (`tie_word_embeddings=false`), as in Qwen3-8B (hidden 4096, intermediate
//! 12288, rope_theta 1e6, `lm_head.weight` in the last shard). CPU reference path, sc#384.
//!
//! Independent oracle: the tiny Qwen3 checkpoint's transformers logits (see
//! `fixtures/qwen3-tiny/PROVENANCE.md`, tied head). The test writes a copy of that checkpoint
//! WITH an extra `lm_head.weight` that is the embedding matrix with rows 3 and 5 swapped. The
//! hidden state does not depend on the head, so the right untied logits are the oracle logits
//! with entries 3 and 5 swapped. A loader that ignores `lm_head.weight` (and reuses the
//! embeddings) returns the unswapped oracle logits and fails; the negative control asserts the
//! swap is far outside the tolerance, so the check cannot pass vacuously. A second test shows
//! an untied config with no `lm_head.weight` is refused, never silently tied.
//!
//! The real Qwen3-8B shape check (config.json values and the `lm_head.weight` index entry) is
//! the ignored test at the bottom (`AIEN_QWEN3_8B_DIR`).

use aien_inference_abi::{load_model_config, ModelConfig, SequenceState, TransformerWeights};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const TOL: f32 = 1e-4;

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-tiny")
}

fn oracle_logits() -> (Vec<u32>, Vec<f32>) {
    let r: Value = serde_json::from_str(
        &std::fs::read_to_string(tiny_dir().join("forward_reference.json")).unwrap(),
    )
    .unwrap();
    let ids = r["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let logits = r["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    (ids, logits)
}

/// The tiny checkpoint plus an `lm_head.weight` (BF16 [64, 32]) = embeddings with rows 3, 5 swapped.
fn untied_checkpoint(dir: &Path, with_head: bool) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes = std::fs::read(tiny_dir().join("model.safetensors")).unwrap();
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
    let data = &bytes[8 + hlen..];
    let emb = &header["model.embed_tokens.weight"];
    assert_eq!(emb["dtype"], "BF16");
    assert_eq!(emb["shape"], json!([64, 32]));
    let start = emb["data_offsets"][0].as_u64().unwrap() as usize;
    let end = emb["data_offsets"][1].as_u64().unwrap() as usize;
    let row = 32 * 2;
    assert_eq!(end - start, 64 * row);
    let mut out = data.to_vec();
    if with_head {
        let mut head = data[start..end].to_vec();
        let (a, b) = (3 * row, 5 * row);
        for i in 0..row {
            head.swap(a + i, b + i);
        }
        header["lm_head.weight"] = json!({
            "dtype": "BF16", "shape": [64, 32],
            "data_offsets": [out.len(), out.len() + head.len()]
        });
        out.extend_from_slice(&head);
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut file = (h.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&h);
    file.extend_from_slice(&out);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

fn untied_config() -> ModelConfig {
    let mut c = load_model_config(&tiny_dir()).unwrap();
    assert!(c.tie_word_embeddings, "the tiny fixture is tied");
    c.tie_word_embeddings = false;
    c
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("aien-untied-head-{name}-{}", std::process::id()))
}

fn last_logits(w: &TransformerWeights, ids: &[u32]) -> Vec<f32> {
    let mut state = SequenceState {
        tokens: Vec::new(),
        layers: vec![Default::default(); w.config.num_layers],
        kv_dtype: Default::default(),
    };
    let mut last = Vec::new();
    for (pos, id) in ids.iter().enumerate() {
        last = w.forward_token(*id, pos, &mut state);
    }
    w.compute_logits(&last)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn untied_head_logits_use_lm_head_not_the_embeddings() {
    let dir = scratch("with");
    untied_checkpoint(&dir, true);
    let (ids, tied) = oracle_logits();
    let mut expected = tied.clone();
    expected.swap(3, 5);
    // Negative control: the swapped vector is far from the tied oracle.
    assert!(
        max_abs(&expected, &tied) > 100.0 * TOL,
        "rows 3 and 5 give nearly equal logits; the control would be vacuous"
    );
    let cfg = untied_config();
    let w = TransformerWeights::load_from_safetensors(dir.join("model.safetensors"), &cfg)
        .expect("untied checkpoint with lm_head.weight loads");
    assert!(w.lm_head.is_some(), "untied load must keep lm_head");
    let got = last_logits(&w, &ids);
    let d = max_abs(&got, &expected);
    eprintln!("untied head: max abs diff vs swapped oracle = {d:e}");
    assert!(d <= TOL, "max abs diff {d:e}");
    assert!(
        max_abs(&got, &tied) > 100.0 * TOL,
        "logits equal the tied (embedding) result: lm_head was ignored"
    );
    // Sequence path agrees with the token path.
    let seq = w.forward_sequence_with_diagnostics(&ids).activations["last_token_logits"].clone();
    assert!(max_abs(&seq, &expected) <= TOL);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn untied_config_without_lm_head_is_refused_not_tied() {
    let dir = scratch("without");
    untied_checkpoint(&dir, false);
    let r =
        TransformerWeights::load_from_safetensors(dir.join("model.safetensors"), &untied_config());
    let _ = std::fs::remove_dir_all(dir);
    let Err(e) = r else {
        panic!("untied config without lm_head.weight must be refused");
    };
    assert!(e.to_string().contains("lm_head.weight"), "{e}");
}

/// Real Qwen3-8B shapes from the downloaded files: `AIEN_QWEN3_8B_DIR=~/models/qwen3-8b cargo test
/// -p aien-inference-abi --test qwen3_untied_head -- --ignored`. Reads only config.json and the
/// shard index (no weights), fails when the variable is unset.
#[test]
#[ignore = "needs the Qwen3-8B model directory in AIEN_QWEN3_8B_DIR"]
fn real_qwen3_8b_config_is_untied_with_the_expected_shape() {
    let dir = PathBuf::from(std::env::var("AIEN_QWEN3_8B_DIR").expect("AIEN_QWEN3_8B_DIR unset"));
    let c = load_model_config(&dir).unwrap();
    assert!(!c.tie_word_embeddings);
    assert_eq!((c.hidden_dim, c.intermediate_dim), (4096, 12288));
    assert_eq!(c.rope_theta, 1_000_000.0);
    assert!(c.qk_norm);
    let idx: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert!(idx["weight_map"]["lm_head.weight"].is_string());
}
