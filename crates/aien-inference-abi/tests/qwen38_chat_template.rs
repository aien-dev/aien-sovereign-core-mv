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

//! Qwen3.8-27B chat template (sc#642 serving). Pinned text: Hugging Face `Qwen/Qwen3.8-27B-FP8`
//! revision 017b9c7a (fixtures/chat-templates/qwen3.8-27b).
//!
//! Checks of the rendered bytes:
//! 1. JINJA ORACLE: `render_oracle.json`, the template rendered by jinja2 3.1.2 (scratch script
//!    outside the repository) for 8 conversations x 6 option sets, thinking on, off and default.
//! 2. GOLDEN PROMPT: with the checkpoint present (`AIEN_QWEN38_DIR`, ignored test), the
//!    thinking-off rendering of the golden question tokenizes to the host reference's
//!    `prompt_ids` (`fixtures/qwen38/gen-chat-max-24.json`), the ids the chip gate replays.
use aien_inference_abi::tokenizer::{ChatOptions, ChatTemplate, ChatTokenizer};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3.8-27b"
);

/// Host reference golden (greedy, 24 tokens, thinking off) shipped with the repository; the
/// chip gate `tools/qwen38_serve_gate.sh` replays the same file.
const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/qwen38/gen-chat-max-24.json"
));

#[test]
fn shipped_golden_is_well_formed() {
    let g: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    let ids = |k: &str| g[k].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(ids("generated_ids"), 24);
    assert!(ids("prompt_ids") > 0);
    // The reference answer opens a C code block (see PROVENANCE.md next to the file).
    assert!(g["generated_text"]
        .as_str()
        .unwrap()
        .starts_with("```c\nint max("));
}

/// The speed tool's references: 256 reference ids for its decode request, and a corpus long
/// enough for its 100,000-byte prompt (see fixtures/qwen38/PROVENANCE.md).
#[test]
fn shipped_speed_references_are_well_formed() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/qwen38");
    let r: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/decode-lighthouses-256.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(r["ids"].as_array().unwrap().len(), 256);
    assert_eq!(r["request"]["max_tokens"], 256);
    assert_eq!(r["request"]["temperature"], 0);
    assert_eq!(r["request"]["aien_return_token_ids"], true);
    let corpus = std::fs::read(format!("{dir}/speed-corpus.txt")).unwrap();
    assert!(corpus.len() >= 100_000, "corpus is {} bytes", corpus.len());
    assert!(std::str::from_utf8(&corpus).is_ok());
}

fn t() -> ChatTemplate {
    let text = std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap();
    ChatTemplate::detect(&text).expect("Qwen3.8 template accepted")
}

fn opts(v: &serde_json::Value) -> ChatOptions {
    ChatOptions {
        enable_thinking: v.get("enable_thinking").and_then(|x| x.as_bool()),
        reasoning_effort: v
            .get("reasoning_effort")
            .and_then(|x| x.as_str())
            .map(String::from),
    }
}

#[test]
fn pinned_template_is_selected() {
    assert_eq!(t(), ChatTemplate::Qwen38);
}

#[test]
fn rendering_matches_the_jinja_reference_rendering() {
    let oracle: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/render_oracle.json")).unwrap(),
    )
    .unwrap();
    let tpl = t();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for (i, case) in cases.iter().enumerate() {
        let turns: Vec<(String, String)> = case["turns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into()))
            .collect();
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let o = opts(&case["options"]);
        assert_eq!(
            tpl.try_render_with(&refs, &o).unwrap(),
            case["text"].as_str().unwrap(),
            "case {i} options {o:?}"
        );
        // The default is the template's default (thinking on, effort xhigh).
        if o == ChatOptions::default() {
            assert_eq!(
                tpl.try_render(&refs).unwrap(),
                case["text"].as_str().unwrap()
            );
        }
    }
}

#[test]
fn options_change_the_rendering_and_bad_ones_are_refused() {
    let tpl = t();
    let turns = [("user", "Hi")];
    let on = tpl
        .try_render_with(&turns, &ChatOptions::default())
        .unwrap();
    let off = tpl
        .try_render_with(
            &turns,
            &ChatOptions {
                enable_thinking: Some(false),
                reasoning_effort: None,
            },
        )
        .unwrap();
    assert_ne!(on, off);
    assert!(on.ends_with("<|im_start|>assistant\n<think>\n"));
    assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let bad = ChatOptions {
        enable_thinking: None,
        reasoning_effort: Some("max".into()),
    };
    assert!(tpl.try_render_with(&turns, &bad).is_err());
    assert!(tpl
        .try_render_with(
            &[("user", "a"), ("system", "late")],
            &ChatOptions::default()
        )
        .is_err());
    assert!(tpl
        .try_render_with(&[("assistant", "no user")], &ChatOptions::default())
        .is_err());
}

#[test]
#[ignore = "needs AIEN_QWEN38_DIR (the checkpoint)"]
fn golden_prompt_tokenizes_to_the_reference_ids() {
    let dir = std::env::var("AIEN_QWEN38_DIR").expect("AIEN_QWEN38_DIR");
    let g: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    let want: Vec<u32> = g["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tk = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).unwrap();
    assert_eq!(tk.template(), ChatTemplate::Qwen38);
    let off = ChatOptions {
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    let text = tk
        .try_format_chat_with(
            &[(
                "user",
                "Write a C function that returns the larger of two ints.",
            )],
            &off,
        )
        .unwrap();
    assert_eq!(tk.encode_with_special(&text, true).unwrap(), want);
    // Negative control: the thinking-on rendering of the same question must not match.
    let on = tk
        .try_format_chat_with(
            &[(
                "user",
                "Write a C function that returns the larger of two ints.",
            )],
            &ChatOptions::default(),
        )
        .unwrap();
    assert_ne!(tk.encode_with_special(&on, true).unwrap(), want);
    assert_eq!(tk.stop_token_ids(), &[248046, 248044]);
    // The limit comes from config.json text_config (262144), not the 2048 default: a 30k prompt
    // encodes (the server's 128k context needs it).
    let long = "word ".repeat(30_000);
    assert!(tk.encode_with_special(&long, true).unwrap().len() > 2048);
}
