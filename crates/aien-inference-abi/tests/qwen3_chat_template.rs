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

//! Qwen3-4B-Instruct-2507 chat template: the pinned template text is recognized, the engine's
//! rendering equals the reference Jinja rendering (fixtures/chat-templates/.../render_oracle.json)
//! for every recorded conversation, and any altered Qwen template is refused.

use aien_inference_abi::tokenizer::ChatTemplate;

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3-4b-instruct-2507"
);

fn template() -> String {
    std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap()
}

#[test]
fn pinned_template_is_qwen3_chatml_without_default_system() {
    let t = ChatTemplate::detect(&template()).expect("Qwen3-4B-Instruct-2507 template accepted");
    assert_eq!(t, ChatTemplate::ChatMlQwen3);
    assert_eq!(t.default_system(), None);
}

#[test]
fn rendering_matches_the_reference_jinja_rendering() {
    let oracle: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/render_oracle.json")).unwrap(),
    )
    .unwrap();
    let t = ChatTemplate::detect(&template()).unwrap();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for (i, case) in cases.iter().enumerate() {
        let turns: Vec<(String, String)> = case["turns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_str().unwrap().to_string(),
                    p[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let ends_in_assistant = refs
            .last()
            .is_some_and(|(r, c)| *r == "assistant" && !c.trim().is_empty());
        // The engine opens a generation prompt unless the last turn is a non-empty assistant
        // turn; the oracle case records which of the two it rendered.
        assert_eq!(
            case["add_generation_prompt"].as_bool().unwrap(),
            !ends_in_assistant,
            "case {i}: oracle generated with the other generation-prompt setting"
        );
        assert_eq!(
            t.try_render(&refs).unwrap(),
            case["text"].as_str().unwrap(),
            "case {i}"
        );
    }
}

#[test]
fn altered_qwen_templates_are_refused() {
    let base = template();
    let altered = [
        // Qwen3-4B (original) style thinking tail after the generation prompt.
        base.replace(
            "{{- '<|im_start|>assistant\\n' }}",
            "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}",
        ),
        // Contents trimmed.
        base.replace(
            "{%- set content = message.content %}",
            "{%- set content = message.content | trim %}",
        ),
        // A default system message added.
        base.replacen(
            "{%- else %}",
            "{%- else %}\n    {{- '<|im_start|>system\\nYou are Qwen.<|im_end|>\\n' }}",
            1,
        ),
        // One byte appended inside the template.
        format!("{base}{{# #}}"),
    ];
    for (i, text) in altered.iter().enumerate() {
        assert_ne!(text, &base, "alteration {i} did not change the template");
        assert!(
            ChatTemplate::detect(text).is_err(),
            "altered template {i} was accepted"
        );
    }
}

/// The real model directory (weights not needed): `AIEN_QWEN3_DIR=<dir> cargo test -p
/// aien-inference-abi --test qwen3_chat_template -- --ignored`. Fails, never skips, when unset.
#[test]
#[ignore = "needs the Qwen3-4B-Instruct-2507 model directory in AIEN_QWEN3_DIR"]
fn real_model_dir_loads_with_the_pinned_template() {
    use aien_inference_abi::tokenizer::ChatTokenizer;
    let dir = std::env::var("AIEN_QWEN3_DIR").expect("AIEN_QWEN3_DIR is not set");
    let tok = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).unwrap();
    assert_eq!(tok.template(), ChatTemplate::ChatMlQwen3);
    let im_end = tok.token_to_id("<|im_end|>").expect("<|im_end|> in vocab");
    assert!(
        tok.stop_token_ids().contains(&im_end),
        "<|im_end|> must stop generation"
    );
    let text = tok
        .try_format_chat(&[("system", "You are AIEN."), ("user", "List two primes.")])
        .unwrap();
    let ids = tok.encode(&text).unwrap();
    assert_eq!(
        ids.first().copied(),
        tok.token_to_id("<|im_start|>"),
        "no BOS is added"
    );
    assert_eq!(ids.iter().filter(|&&t| t == im_end).count(), 2);
}

/// Other ChatML models (SmolLM2, plain ChatML) keep their layout: role `tool` is still a plain
/// user turn there, and the non-tool roles render the same as for Qwen3.
#[test]
fn plain_chatml_is_unchanged_and_shares_non_tool_roles_with_qwen3() {
    let plain = ChatTemplate::ChatMl {
        default_system: None,
    };
    let turns = [("user", "q"), ("tool", "r"), ("tool", "s")];
    assert_eq!(
        plain.try_render(&turns).unwrap(),
        "<|im_start|>user\nq<|im_end|>\n<|im_start|>user\nr<|im_end|>\n<|im_start|>user\ns<|im_end|>\n<|im_start|>assistant\n"
    );
    let qwen = ChatTemplate::detect(&template()).unwrap();
    let no_tool = [
        ("system", "S"),
        ("user", "U"),
        ("assistant", "A"),
        ("user", "V"),
    ];
    assert_eq!(plain.try_render(&no_tool), qwen.try_render(&no_tool));
}
