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

//! Qwen3-8B chat template, thinking OFF (sc#384). Pinned text: Hugging Face `Qwen/Qwen3-8B`
//! revision b968826d9c46dd6066d109eabc6255188de91218 (fixtures/chat-templates/qwen3-8b).
//!
//! Three independent checks of the rendered bytes:
//! 1. HAND ORACLE (below): strings written by reading the template text branch by branch, not
//!    produced by the engine.
//! 2. JINJA ORACLE: `render_oracle.json`, the template rendered by jinja2 3.1.6 with
//!    `enable_thinking=False` (scratch script outside the repository), for 10 conversations.
//! 3. BYTE DIFF against the downloaded file: the ignored test extracts `chat_template` from
//!    `$AIEN_QWEN3_8B_DIR/tokenizer_config.json` and compares it to the pinned fixture byte for
//!    byte (and loads the model directory's tokenizer). Run it with
//!    `AIEN_QWEN3_8B_DIR=~/models/qwen3-8b cargo test -p aien-inference-abi --test
//!    qwen3_8b_chat_template -- --ignored`.
//!
//! Template branches taken (tools=None, add_generation_prompt=true, enable_thinking=false):
//! - `{%- if tools %}` false -> `{%- else %}`: system header only if `messages[0].role=='system'`:
//!   `<|im_start|>system\n{content}<|im_end|>\n`; the loop then skips that first system turn
//!   (`message.role == "system" and not loop.first` is false), so it appears once.
//! - user, or system after the first turn: `<|im_start|>{role}\n{content}<|im_end|>\n`, verbatim.
//! - assistant, no reasoning_content, no `</think>` in content, no tool_calls: `index <=
//!   last_query_index` -> `<|im_start|>assistant\n{content}`, then `<|im_end|>\n`. (An assistant
//!   turn after the last user turn that is not last takes the same plain text; one that is last
//!   takes the think-block branch and is refused by the engine.)
//! - generation prompt: `<|im_start|>assistant\n` then, because `enable_thinking is defined and
//!   is false`, `<think>\n\n</think>\n\n`.

use aien_inference_abi::tokenizer::{ChatTemplate, TokenizerError};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3-8b"
);
const GEN: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn template_text() -> String {
    std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap()
}

fn t() -> ChatTemplate {
    ChatTemplate::detect(&template_text()).expect("Qwen3-8B template accepted")
}

#[test]
fn pinned_template_is_selected_and_distinct_from_the_4b_layout() {
    assert_eq!(t(), ChatTemplate::ChatMlQwen3NoThink);
    assert_ne!(t(), ChatTemplate::ChatMlQwen3);
}

#[test]
fn hand_oracle_from_the_template_branches() {
    let tpl = t();
    // Branch: no system, one user.
    assert_eq!(
        tpl.try_render(&[("user", "Hi")]).unwrap(),
        format!("<|im_start|>user\nHi<|im_end|>\n{GEN}")
    );
    // Branch: leading system emitted once, contents verbatim (no trim).
    assert_eq!(
        tpl.try_render(&[("system", " S\n"), ("user", "  U \n")])
            .unwrap(),
        format!("<|im_start|>system\n S\n<|im_end|>\n<|im_start|>user\n  U \n<|im_end|>\n{GEN}")
    );
    // Branch: earlier assistant turn (index <= last_query_index) is plain, no think block.
    assert_eq!(
        tpl.try_render(&[("user", "a"), ("assistant", "b"), ("user", "c")])
            .unwrap(),
        format!(
            "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n{GEN}"
        )
    );
}

#[test]
fn rendering_matches_the_jinja_reference_rendering() {
    let oracle: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/render_oracle.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(oracle["enable_thinking"], false);
    let tpl = t();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(case["add_generation_prompt"], true);
        assert_eq!(case["enable_thinking"], false);
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
        assert_eq!(
            tpl.try_render(&refs).unwrap(),
            case["text"].as_str().unwrap(),
            "case {i}"
        );
        assert!(case["text"].as_str().unwrap().ends_with(GEN));
    }
}

fn layout_refused(r: Result<String, TokenizerError>) {
    assert!(
        matches!(r, Err(TokenizerError::UnsupportedChatLayout(_))),
        "{r:?}"
    );
}

#[test]
fn uncovered_combinations_are_refused_by_name() {
    let tpl = t();
    // Thinking enabled.
    layout_refused(tpl.try_render_thinking(&[("user", "q")], true));
    assert_eq!(
        tpl.try_render_thinking(&[("user", "q")], false).unwrap(),
        tpl.try_render(&[("user", "q")]).unwrap()
    );
    // Tool turns.
    assert!(matches!(
        tpl.try_render(&[("user", "q"), ("tool", "r")]),
        Err(TokenizerError::ToolTurnRefused(_))
    ));
    // Reasoning content / thinking markers inside an assistant turn.
    layout_refused(tpl.try_render(&[
        ("user", "q"),
        ("assistant", "<think>x</think>y"),
        ("user", "z"),
    ]));
    // Trailing assistant turn (continuation).
    layout_refused(tpl.try_render(&[("user", "q"), ("assistant", "partial")]));
    layout_refused(tpl.try_render(&[("user", "q"), ("assistant", "")]));
    // User text shaped like a tool response.
    layout_refused(tpl.try_render(&[("user", "<tool_response>\nx\n</tool_response>")]));
    // Unknown or differently cased roles, and an empty conversation.
    layout_refused(tpl.try_render(&[("developer", "q")]));
    layout_refused(tpl.try_render(&[("User", "q")]));
    layout_refused(tpl.try_render(&[]));
}

#[test]
fn altered_templates_are_refused() {
    let base = template_text();
    let altered = [
        // Thinking-on generation prompt (no enable_thinking switch).
        base.replace(
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{%- if false %}",
        ),
        base.replace("\\n</think>\\n\\n", "\\n</think>\\n"),
        format!("{base}{{# #}}"),
    ];
    for (i, text) in altered.iter().enumerate() {
        assert_ne!(text, &base, "alteration {i} changed nothing");
        assert!(ChatTemplate::detect(text).is_err(), "altered {i} accepted");
    }
}

#[test]
fn the_4b_layout_is_unchanged() {
    let four_b = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/chat-templates/qwen3-4b-instruct-2507/chat_template.jinja"
    ))
    .unwrap();
    let f = ChatTemplate::detect(&four_b).unwrap();
    assert_eq!(f, ChatTemplate::ChatMlQwen3);
    assert_eq!(
        f.try_render(&[("user", "Hi")]).unwrap(),
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
#[ignore = "needs the Qwen3-8B model directory in AIEN_QWEN3_8B_DIR"]
fn downloaded_tokenizer_config_matches_the_pin_byte_for_byte() {
    use aien_inference_abi::tokenizer::ChatTokenizer;
    let dir = std::env::var("AIEN_QWEN3_8B_DIR").expect("AIEN_QWEN3_8B_DIR is not set");
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/tokenizer_config.json")).unwrap(),
    )
    .unwrap();
    let downloaded = cfg["chat_template"].as_str().expect("chat_template string");
    assert_eq!(downloaded.as_bytes(), template_text().as_bytes());
    let tok = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).unwrap();
    assert_eq!(tok.template(), ChatTemplate::ChatMlQwen3NoThink);
    let im_end = tok.token_to_id("<|im_end|>").expect("<|im_end|> in vocab");
    assert!(tok.stop_token_ids().contains(&im_end));
    let text = tok
        .try_format_chat(&[("system", "You are AIEN."), ("user", "List two primes.")])
        .unwrap();
    assert!(text.ends_with(GEN));
    let ids = tok.encode(&text).unwrap();
    assert_eq!(ids.first().copied(), tok.token_to_id("<|im_start|>"));
    assert_eq!(ids.iter().filter(|&&i| i == im_end).count(), 2);
}
