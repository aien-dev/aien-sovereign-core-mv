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

//! Our BitNet tokenizer on the real `tokenizer.json` (ignored: needs the checkpoint).
//!
//!   BITNET_DIR=~/models/BitNet-b1.58-2B-4T-bf16 cargo test -p aien-inference-abi \
//!       --test bitnet_tokenizer_real -- --ignored
//!
//! The yardsticks are our own: the five acceptance id lists recorded in the ternary-oracle spec
//! (section 6), spot ranks from the reference `tokenizer.model` (spec section 6: "The" = 791,
//! " is" = 374, " France" = 9822), and exact round trips. No other tokenizer runs.

use std::path::PathBuf;

use aien_inference_abi::bitnet_omega::{sha256_file, BITNET_2B4T_TOKENIZER_SHA256};
use aien_inference_abi::bitnet_tokenizer::BitNetTokenizer;

fn dir() -> PathBuf {
    std::env::var_os("BITNET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("models/BitNet-b1.58-2B-4T-bf16")
        })
}

#[test]
#[ignore = "needs the BitNet 2B-4T checkpoint"]
fn real_tokenizer_matches_the_spec_ids_and_round_trips() {
    let path = dir().join("tokenizer.json");
    assert_eq!(
        sha256_file(&path).unwrap(),
        BITNET_2B4T_TOKENIZER_SHA256,
        "tokenizer.json is not the pinned file"
    );
    let t = BitNetTokenizer::from_file(&path).unwrap();
    assert_eq!(t.vocab_len(), 128_256);
    assert_eq!(t.bos, 128_000);

    let spec: [(&str, &[u32]); 5] = [
        (
            "The capital of France is",
            &[128000, 791, 6864, 315, 9822, 374],
        ),
        (
            "The capital of Germany is",
            &[128000, 791, 6864, 315, 10057, 374],
        ),
        (
            "Count from 6 to 10:",
            &[128000, 2568, 505, 220, 21, 311, 220, 605, 25],
        ),
        (
            "The boiling point of water in Celsius is",
            &[128000, 791, 50937, 1486, 315, 3090, 304, 62447, 374],
        ),
        (
            "The largest planet in the solar system is",
            &[128000, 791, 7928, 11841, 304, 279, 13238, 1887, 374],
        ),
    ];
    for (text, ids) in spec {
        assert_eq!(t.encode(text, true), ids, "{text:?}");
        assert_eq!(t.decode(&ids[1..]).unwrap(), text);
    }
    let chat = t.encode("<|begin_of_text|>User: hi<|eot_id|>", false);
    assert_eq!((chat[0], *chat.last().unwrap()), (128000, 128009));
    assert_eq!(
        t.decode(&chat).unwrap(),
        "<|begin_of_text|>User: hi<|eot_id|>"
    );
    assert_eq!(*t.encode("<|eot_id|>", false), [128009]);

    let stress = "Tab\there, CRLF\r\n\r\n  spaced   out \u{3000}x; I'M we'Ve it'ſ \
                  über-naïve 東京タワー 12345 ½ ٣٤٥ emoji 🙂🙂 \u{200d} trailing  ";
    let ids = t.encode(stress, false);
    assert_eq!(t.decode(&ids).unwrap(), stress);
    assert!(ids.iter().all(|&i| i < 128_000));
}
