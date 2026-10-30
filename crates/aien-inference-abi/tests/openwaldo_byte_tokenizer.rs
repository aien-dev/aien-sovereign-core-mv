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

//! OpenWALDO byte tokenizer as a plain `tokenizer.json`.
//!
//! OpenWALDO exports ship a custom Python tokenizer (`OpenWALDOByteTokenizer`):
//! `<pad>`=0, `<bos>`=1, `<eos>`=2, then byte `b` -> id `b + 3`. The fixture
//! `tokenizer.json` states the same mapping as a BPE model with no merges and
//! byte fallback, so AIEN loads it with no tokenizer code change. The
//! reference ids and decoded strings in `reference.json` were produced by the
//! exported Python class itself (transformers 5.17.0, CPU).

use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::Value;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/openwaldo-byte");

fn load() -> (ChatTokenizer, Vec<Value>) {
    let tok =
        ChatTokenizer::from_file(format!("{DIR}/tokenizer.json")).expect("load tokenizer.json");
    let text = std::fs::read_to_string(format!("{DIR}/reference.json")).expect("read reference");
    let cases: Vec<Value> = serde_json::from_str(&text).expect("parse reference");
    (tok, cases)
}

fn ids(v: &Value) -> Vec<u32> {
    v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn encode_matches_openwaldo_reference() {
    let (tok, cases) = load();
    let mut checked = 0;
    for case in cases.iter().filter(|c| c["text"].is_string()) {
        let text = case["text"].as_str().unwrap();
        let got = tok.encode_with_special(text, false).expect("encode");
        assert_eq!(got, ids(case), "encode mismatch for {text:?}");
        checked += 1;
    }
    assert_eq!(checked, 10);
}

#[test]
fn decode_matches_openwaldo_reference_on_valid_utf8() {
    let (tok, cases) = load();
    for case in cases.iter().filter(|c| c["text"].is_string()) {
        let got = tok.decode_opts(&ids(case), false).expect("decode");
        assert_eq!(
            got,
            case["decoded"].as_str().unwrap(),
            "decode mismatch for {:?}",
            ids(case)
        );
    }
}

#[test]
fn every_byte_maps_to_byte_plus_three() {
    let (tok, _) = load();
    for b in 0u8..=255 {
        let id = u32::from(b) + 3;
        let s = format!("<0x{b:02X}>");
        assert_eq!(tok.token_to_id(&s), Some(id), "{s}");
    }
}

/// Known limit: on an invalid UTF-8 byte run the two decoders disagree.
/// Python (`errors="replace"`) emits one U+FFFD per maximal invalid subpart and
/// keeps the following valid byte; the `tokenizers` ByteFallback decoder emits
/// one U+FFFD per byte token of the whole run. Token ids are identical, so
/// receipts that bind token ids are unaffected; decoded text is not.
/// This test pins today's behaviour so a decoder change is noticed.
#[test]
fn decode_diverges_on_invalid_utf8_run() {
    let (tok, cases) = load();
    let case = cases
        .iter()
        .find(|c| c["text"].is_null())
        .expect("invalid-run case");
    assert_eq!(case["decoded"].as_str().unwrap(), "\u{FFFD}A");
    let got = tok.decode_opts(&ids(case), false).expect("decode");
    assert_eq!(got, "\u{FFFD}\u{FFFD}\u{FFFD}");
}
