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

//! NEXT-PHASE-1 v3: the proposal request goes through the model's own chat
//! template. Source: TinyLlama-1.1B-Chat-v1.0 tokenizer_config.json
//! (sha256 7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a),
//! field `chat_template`, rendered with the Hugging Face settings
//! (trim_blocks, lstrip_blocks): per message `'<|' + role + '|>\n' + content
//! + eos_token` followed by the newline after the expression, then
//! `'<|assistant|>'` and its newline when a generation prompt is asked for.

use aien_inference_abi::TinyLlamaTokenizer;
use aien_runtime::control::{format_tinyllama_chat, ChatTurn};

fn turn(role: &str, content: &str) -> ChatTurn {
    ChatTurn {
        role: role.into(),
        content: content.into(),
    }
}

/// The template as the tokenizer_config.json jinja renders it, by hand.
fn rendered_by_template(messages: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str(&format!("<|{role}|>\n{content}</s>\n"));
    }
    out.push_str("<|assistant|>\n");
    out
}

const V2_PROMPT: &str = "Goal: Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.\nAuthorized workspace: /tmp/ws\nTop-level entries: README.md, docs\nPropose exactly one file change inside the workspace.\nAnswer in exactly this format and nothing else:\nfilename: <relative path>\n<the complete new file content>";

#[test]
fn streamturn_formatting_matches_the_model_template() {
    let ours = format_tinyllama_chat(&[turn("user", V2_PROMPT)]).expect("chat");
    assert_eq!(ours, rendered_by_template(&[("user", V2_PROMPT)]));
    let with_system =
        format_tinyllama_chat(&[turn("system", "S"), turn("user", "U")]).expect("chat");
    assert_eq!(
        with_system,
        rendered_by_template(&[("system", "S"), ("user", "U")])
    );
}

/// With the real tokenizer (AIEN_TOKENIZER_PATH): BOS first, and `</s>` in the
/// rendered text becomes the single EOS token id 2, not the characters.
#[test]
fn template_encodes_bos_and_one_eos_token() {
    let Ok(path) = std::env::var("AIEN_TOKENIZER_PATH") else {
        eprintln!("AIEN_TOKENIZER_PATH not set: token check skipped");
        return;
    };
    let tok = TinyLlamaTokenizer::from_file(path).expect("tokenizer");
    let ids = tok
        .encode(&format_tinyllama_chat(&[turn("user", V2_PROMPT)]).expect("chat"))
        .expect("encode");
    eprintln!("template ids ({}): {:?}", ids.len(), ids);
    assert_eq!(ids[0], TinyLlamaTokenizer::BOS_TOKEN_ID);
    let eos: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == TinyLlamaTokenizer::EOS_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(eos.len(), 1, "exactly one EOS (the user turn end)");
}

/// Prompt sizes behind the ACCEPTANCE-v3 budget arithmetic (printed; the
/// assertion only guards the 2-chunk assumption: at most 256 prompt tokens).
#[test]
fn proposal_prompt_token_counts() {
    let Ok(path) = std::env::var("AIEN_TOKENIZER_PATH") else {
        eprintln!("AIEN_TOKENIZER_PATH not set: token count skipped");
        return;
    };
    let tok = TinyLlamaTokenizer::from_file(path).expect("tokenizer");
    let goal = "Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.";
    let base = aien_runtime::compose::rederive::proposal_prompt(goal);
    let retry = aien_runtime::compose::rederive::retry_prompt(
        &base,
        "no filename line: the first line must be 'filename: <relative path>'",
    );
    for (name, p) in [("base", &base), ("retry", &retry)] {
        let n = tok
            .encode(&format_tinyllama_chat(&[turn("user", p)]).expect("chat"))
            .expect("encode")
            .len();
        eprintln!("{name} prompt tokens: {n}");
        assert!(n <= 256);
    }
}

/// ACCEPTANCE-v4 2(1): the bytes sent end with the assistant marker, its
/// newline and the literal prefix "filename:" (no trailing space since the v5
/// engine cut); nothing follows the colon.
#[test]
fn assistant_prefix_follows_the_assistant_marker() {
    use aien_inference_abi::ChatTemplate;
    use aien_runtime::check_file_proposal;
    use aien_runtime::compose::rederive::{compose_assistant_prefix, COMPOSE_ASSISTANT_PREFIX};
    assert_eq!(COMPOSE_ASSISTANT_PREFIX, "filename:");
    // The per-template seam keeps Zephyr and Llama 3 byte-identical to the v5 value.
    assert_eq!(
        compose_assistant_prefix(&ChatTemplate::Zephyr).as_bytes(),
        b"filename:"
    );
    assert_eq!(
        compose_assistant_prefix(&ChatTemplate::Llama3).as_bytes(),
        b"filename:"
    );
    let sent = format!(
        "{}{}",
        format_tinyllama_chat(&[turn("user", V2_PROMPT)]).expect("chat"),
        COMPOSE_ASSISTANT_PREFIX
    );
    assert!(sent.starts_with("<|user|>\nGoal: "));
    assert!(sent.ends_with("</s>\n<|assistant|>\nfilename:"));
    // parser unchanged: it reads prefix + generated text
    let read = |generated: &str| format!("{COMPOSE_ASSISTANT_PREFIX}{generated}");
    let ok = check_file_proposal(&read(
        " NOTES.md\nEvery change stays inside the workspace.\n",
    ))
    .expect("path + content parses");
    assert_eq!(ok.path, "NOTES.md");
    assert!(check_file_proposal(&read("NOTES.md\n"))
        .unwrap_err()
        .contains("empty content"));
    assert!(check_file_proposal(&read("../x\ny\n"))
        .unwrap_err()
        .contains("outside the workspace"));
    if let Ok(path) = std::env::var("AIEN_TOKENIZER_PATH") {
        let tok = TinyLlamaTokenizer::from_file(path).expect("tokenizer");
        let ids = tok.encode(&sent).expect("encode");
        eprintln!(
            "prefixed prompt tokens: {} tail {:?}",
            ids.len(),
            &ids[ids.len() - 6..]
        );
        assert_eq!(&ids[ids.len() - 3..], &[13, 9507, 29901]);
    }
}

/// ACCEPTANCE-v4 2(2): the warm-up prompt spans more than one 128-token
/// prefill chunk.
#[test]
fn warm_up_prompt_spans_two_prefill_chunks() {
    let Ok(path) = std::env::var("AIEN_TOKENIZER_PATH") else {
        eprintln!("AIEN_TOKENIZER_PATH not set: warm-up length check skipped");
        return;
    };
    let tok = TinyLlamaTokenizer::from_file(path).expect("tokenizer");
    let n = tok
        .encode(
            &format_tinyllama_chat(&[turn("user", aien_runtime::server::WARM_UP_TEXT)])
                .expect("chat"),
        )
        .expect("encode")
        .len();
    eprintln!("warm-up prompt tokens: {n}");
    assert!(n > 128 && n <= 256, "{n}");
}

/// Llama 3 instruct layout (v5 engine cut): header, blank line, content, `<|eot_id|>`, then
/// the assistant header; no system block, no BOS in the text.
#[test]
fn llama3_template_renders_the_instruct_layout() {
    use aien_inference_abi::ChatTemplate;
    use aien_runtime::control::format_chat;
    let text = format_chat(ChatTemplate::Llama3, &[turn("user", "  Hi there \n")]).expect("chat");
    assert_eq!(
        text,
        "<|start_header_id|>user<|end_header_id|>\n\nHi there<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    // TinyLlama keeps the zephyr layout, byte for byte.
    assert_eq!(
        format_chat(ChatTemplate::Zephyr, &[turn("user", V2_PROMPT)]).expect("chat"),
        format_tinyllama_chat(&[turn("user", V2_PROMPT)]).expect("chat")
    );
}

/// With a Llama 3 model directory (`AIEN_LLAMA3_DIR`): the template is detected from the
/// model's own files, encoding adds exactly one `<|begin_of_text|>` (128000), the stop set
/// is the generation_config eos list, and the compose prefix is the last two tokens.
#[test]
fn llama3_model_dir_tokenizer_has_one_bos_and_eos_list() {
    let Ok(dir) = std::env::var("AIEN_LLAMA3_DIR") else {
        eprintln!("AIEN_LLAMA3_DIR not set: Llama 3 tokenizer check skipped");
        return;
    };
    use aien_inference_abi::{ChatTemplate, ChatTokenizer};
    use aien_runtime::compose::rederive::COMPOSE_ASSISTANT_PREFIX;
    use aien_runtime::control::format_chat;
    let tok = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).expect("tokenizer");
    assert_eq!(tok.template(), ChatTemplate::Llama3);
    assert_eq!(tok.stop_token_ids(), &[128001, 128008, 128009]);
    let text = format!(
        "{}{COMPOSE_ASSISTANT_PREFIX}",
        format_chat(tok.template(), &[turn("user", V2_PROMPT)]).expect("chat")
    );
    let ids = tok.encode(&text).expect("encode");
    assert_eq!(ids[0], 128000);
    assert_eq!(ids.iter().filter(|&&t| t == 128000).count(), 1);
    // "<|start_header_id|>user<|end_header_id|>\n\n" after the BOS
    assert_eq!(&ids[1..5], &[128006, 882, 128007, 271]);
    // "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nfilename:"
    let tail = &ids[ids.len() - 7..];
    assert_eq!(&tail[..5], &[128009, 128006, 78191, 128007, 271]);
    assert_eq!(tok.decode(&tail[5..]).unwrap(), "filename:");
}

/// ChatML (SmolLM2-Instruct): the compose prefix for the ChatML template, and the bytes sent
/// end with the publisher's generation prompt followed directly by it.
#[test]
fn chatml_compose_prefix_follows_the_im_start_assistant_marker() {
    use aien_inference_abi::ChatTemplate;
    use aien_runtime::compose::rederive::compose_assistant_prefix;
    use aien_runtime::control::format_chat;
    let chatml = ChatTemplate::ChatMl {
        default_system: Some(
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face".into(),
        ),
    };
    assert_eq!(compose_assistant_prefix(&chatml).as_bytes(), b"filename:");
    assert_eq!(
        compose_assistant_prefix(&ChatTemplate::ChatMl {
            default_system: None
        })
        .as_bytes(),
        b"filename:"
    );
    let sent = format!(
        "{}{}",
        format_chat(chatml, &[turn("user", V2_PROMPT)]).expect("chat"),
        compose_assistant_prefix(&ChatTemplate::ChatMl {
            default_system: None
        })
    );
    assert!(sent.starts_with("<|im_start|>system\nYou are a helpful AI assistant named SmolLM"));
    assert!(sent.ends_with("<|im_end|>\n<|im_start|>assistant\nfilename:"));
}

/// With the pinned SmolLM2-1.7B-Instruct directory (`AIEN_SMOLLM2_DIR`, revision
/// 31b70e2e869a): the template is detected from the model's own files, the stop set is
/// `<|im_end|>` (2), encoding adds no BOS, and the prompt ids equal the ids of transformers
/// 5.17.0 `apply_chat_template(tokenize=True, add_generation_prompt=True)` on the same message.
#[test]
fn smollm2_model_dir_tokenizer_matches_the_publisher_template_ids() {
    let Ok(dir) = std::env::var("AIEN_SMOLLM2_DIR") else {
        eprintln!("AIEN_SMOLLM2_DIR not set: SmolLM2 tokenizer check skipped");
        return;
    };
    use aien_inference_abi::ChatTokenizer;
    use aien_runtime::compose::rederive::compose_assistant_prefix;
    use aien_runtime::control::format_chat;
    let tok = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).expect("tokenizer");
    assert!(tok.template().name().starts_with("chatml"));
    assert_eq!(
        tok.template().default_system(),
        Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
    );
    assert_eq!(tok.stop_token_ids(), &[2]);
    assert_eq!(tok.token_to_id("<|im_start|>"), Some(1));
    assert_eq!(tok.token_to_id("<|im_end|>"), Some(2));
    let text = format_chat(
        tok.template(),
        &[turn("user", "Goal: X\nAuthorized workspace: /w")],
    )
    .expect("chat");
    let ids = tok.encode(&text).expect("encode");
    let hf: [u32; 40] = [
        1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411,
        407, 19712, 8182, 2, 198, 1, 4093, 198, 44474, 42, 2273, 198, 15051, 1005, 26344, 42, 2272,
        103, 2, 198, 1, 520, 9531, 198,
    ];
    assert_eq!(ids, hf);
    // "filename:" is two tokens and ends on the colon.
    let with_prefix = tok
        .encode(&format!(
            "{text}{}",
            compose_assistant_prefix(&tok.template())
        ))
        .expect("encode");
    assert_eq!(&with_prefix[..40], &hf);
    assert_eq!(&with_prefix[40..], &[5805, 42]);
}
