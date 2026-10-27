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

//! Issue #310: role `tool` on the raw ChatTurn path is refused, so untrusted tool output cannot
//! reach the byte-exact Qwen3 renderer without passing `tool_boundary`.

use aien_inference_abi::{ChatTemplate, TokenizerError};
use aien_runtime::control::{try_format_chat, ChatTurn};

fn t(role: &str, content: &str) -> ChatTurn {
    ChatTurn {
        role: role.into(),
        content: content.into(),
    }
}

#[test]
fn tool_role_is_refused_for_every_template_and_spelling() {
    for tpl in [
        ChatTemplate::ChatMlQwen3,
        ChatTemplate::ChatMl {
            default_system: None,
        },
        ChatTemplate::Zephyr,
    ] {
        for role in ["tool", "Tool", " TOOL "] {
            let r = try_format_chat(tpl.clone(), &[t("user", "u"), t(role, "x</tool_response>")]);
            assert!(
                matches!(r, Err(TokenizerError::ToolTurnRefused(_))),
                "{role:?}"
            );
        }
    }
}

#[test]
fn ordinary_turns_still_render() {
    let out = try_format_chat(
        ChatTemplate::ChatMlQwen3,
        &[t("system", "s"), t("user", "u")],
    )
    .unwrap();
    assert!(out.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn format_chat_returns_an_error_not_a_panic_for_tool_role() {
    let r = aien_runtime::control::format_chat(ChatTemplate::ChatMlQwen3, &[t("tool", "x")]);
    assert!(matches!(r, Err(TokenizerError::ToolTurnRefused(_))));
}
