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

//! Pins unsloth/Llama-3.2-1B-Instruct (the CAND-4 model, snapshot
//! 5a8abab4a5d6f164389b1079fb721cfab8d7126c) on the CPU reference path so the Qwen3 loader change
//! cannot alter a Llama-architecture model: composed prompt ids, and the exact last-position
//! logits (FNV-1a over every f32 bit pattern, plus the top-5 ids). Values were computed on
//! origin/main 845f58f (before the Qwen3 change). Set `AIEN_LLAMA3_DIR` to that snapshot
//! directory to run; skipped otherwise.

use aien_inference_abi::tokenizer::ChatTokenizer;
use aien_inference_abi::{load_model_config, TransformerWeights};
use std::path::Path;

const TURNS: &[(&str, &str)] = &[("system", "You are terse."), ("user", "  What is 2+2?\n")];

fn fnv1a(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in logits {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[test]
fn llama32_prompt_ids_and_forward_logits_are_pinned() {
    let Ok(dir) = std::env::var("AIEN_LLAMA3_DIR") else {
        eprintln!("AIEN_LLAMA3_DIR not set: Llama 3.2 pin skipped");
        return;
    };
    let dir = Path::new(&dir);
    let tok = ChatTokenizer::from_model_dir(dir, None).expect("tokenizer");
    let text = tok.template().render(TURNS);
    let ids = tok.encode(&text).expect("encode");
    println!("PIN llama32 ids={ids:?}");
    assert_eq!(ids, PIN_IDS);

    let config = load_model_config(dir).expect("config");
    let weights = TransformerWeights::load_from_safetensors(dir.join("model.safetensors"), &config)
        .expect("weights");
    let logits =
        weights.forward_sequence_with_diagnostics(&ids).activations["last_token_logits"].clone();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let top5: Vec<usize> = order[..5].to_vec();
    println!(
        "PIN llama32 len={} fnv={:#018x} top5={top5:?}",
        logits.len(),
        fnv1a(&logits)
    );
    assert_eq!(logits.len(), PIN_LEN);
    assert_eq!(top5, PIN_TOP5);
    assert_eq!(fnv1a(&logits), PIN_FNV);
}

const PIN_IDS: &[u32] = &[
    128000, 128006, 9125, 128007, 271, 2675, 527, 51637, 13, 128009, 128006, 882, 128007, 271,
    3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271,
];
const PIN_LEN: usize = 128256;
const PIN_TOP5: [usize; 5] = [18, 19, 40, 17, 16];
const PIN_FNV: u64 = 0x00d44eba4b6ca94b;
