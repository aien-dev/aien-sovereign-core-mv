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

//! Pins the composed chat prompt token ids for a fixed chat input, so the plain-model
//! change cannot alter how chat-template models build prompts. Ids were computed on
//! origin/main (86841ba) before the change.

use aien_inference_abi::tokenizer::{ChatTemplate, ChatTokenizer};
use std::path::Path;

const TURNS: &[(&str, &str)] = &[("system", "You are terse."), ("user", "  What is 2+2?\n")];

fn ids(dir: &Path) -> (ChatTemplate, String, Vec<u32>) {
    let tok = ChatTokenizer::from_model_dir(dir, None).expect("load");
    let text = tok.template().render(TURNS);
    let ids = tok.encode(&text).expect("encode");
    (tok.template(), text, ids)
}

#[test]
fn pin_tinyllama_zephyr_fixture_prompt_ids() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let (t, text, ids) = ids(&dir);
    println!("PIN zephyr text={text:?} ids={ids:?}");
    assert_eq!(t, ChatTemplate::Zephyr);
    assert_eq!(
        text,
        "<|system|>\nYou are terse.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
    );
    assert_eq!(ids, PIN_ZEPHYR);
}

/// CAND-4 model (Llama-3.2-1B-Instruct): set `AIEN_LLAMA3_DIR` to its model directory.
#[test]
fn pin_llama32_1b_prompt_ids() {
    let Ok(dir) = std::env::var("AIEN_LLAMA3_DIR") else {
        eprintln!("AIEN_LLAMA3_DIR not set: Llama 3.2 pin skipped");
        return;
    };
    let (t, text, ids) = ids(Path::new(&dir));
    println!("PIN llama3 text={text:?} ids={ids:?}");
    assert_eq!(t, ChatTemplate::Llama3);
    assert_eq!(ids, PIN_LLAMA32);
}

const PIN_ZEPHYR: &[u32] = &[
    1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 1935, 344, 29889, 2, 29871, 13, 29966, 29989,
    1792, 29989, 29958, 13, 5618, 338, 29871, 29906, 29974, 29906, 29973, 2, 29871, 13, 29966,
    29989, 465, 22137, 29989, 29958, 13,
];
const PIN_LLAMA32: &[u32] = &[
    128000, 128006, 9125, 128007, 271, 2675, 527, 51637, 13, 128009, 128006, 882, 128007, 271,
    3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271,
];
