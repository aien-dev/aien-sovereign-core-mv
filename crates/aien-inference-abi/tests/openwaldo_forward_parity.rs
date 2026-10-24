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

//! OpenWALDO tiny model (1 layer, hidden 32, 4 heads, 2 KV heads, vocab 259, tied
//! embeddings, bf16) loaded by the CPU reference engine from its Hugging Face directory,
//! compared with last-position logits from transformers 5.17.0 (CPU, float32 after
//! upcasting the bf16 weights). Input ids [1, 75, 104, 111] = bos + bytes "Hel".
//! Tolerance: absolute 1e-4, the ABSOLUTE_TOLERANCE of tinyllama_parity.rs (set before
//! the first comparison, not tuned).

use aien_inference_abi::{load_model_config, TransformerWeights};
use serde_json::Value;
use std::path::PathBuf;

const ABSOLUTE_TOLERANCE: f32 = 1e-4;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/openwaldo-byte")
}

#[test]
fn openwaldo_cpu_logits_match_transformers_reference() {
    let dir = dir();
    let config = load_model_config(&dir).expect("config.json");
    let weights = TransformerWeights::load_from_safetensors(dir.join("model.safetensors"), &config)
        .expect("weights");
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("forward_reference.json")).unwrap())
            .unwrap();
    let ids: Vec<u32> = r["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids, vec![1, 75, 104, 111]);
    let want: Vec<f32> = r["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let got =
        weights.forward_sequence_with_diagnostics(&ids).activations["last_token_logits"].clone();
    assert_eq!(got.len(), 259);
    assert_eq!(want.len(), 259);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max abs diff = {max_abs:e}");
    assert!(max_abs <= ABSOLUTE_TOLERANCE, "max abs diff {max_abs:e}");
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(&got) as u64, r["argmax"].as_u64().unwrap());
}
