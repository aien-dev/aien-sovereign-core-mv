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

//! Trusted boundary for tool output entering a Qwen3 conversation (issue #310).
//!
//! The Qwen3-Instruct-2507 chat template does not escape tool content, and
//! [`ChatTemplate::ChatMlQwen3`] matches it byte for byte (that is correct and stays so). A tool
//! result containing `<|im_end|>` or `</tool_response>` would therefore close the tool turn
//! early and let untrusted text forge the next message. The renderer is not the place to fix
//! this; the place is the code that turns tool output into a conversation turn.
//!
//! This module is that place. Honest status: UNVERIFIED that any production caller forwards tool
//! turns today (the daemon sends user/system only). So the boundary is the single entry point a
//! tool turn must pass: a tool turn can only be built as a [`ToolOutput`] (private fields, made
//! only by [`ToolGuard::admit`]) and only added to a [`ToolConversation`], which pairs it with an
//! assistant tool call. The runtime's `ChatTurn` path refuses role `tool` outright and points
//! here (`aien_runtime::control`).
//!
//! Policy: fail closed with a named [`ToolBoundaryError`]. Nothing is escaped or stripped.
//! Source of truth for the refusal list: the template's own structural strings (below, pinned by
//! a test against what the renderer actually emits) united with every added token of the loaded
//! tokenizer ([`ToolGuard::from_tokenizer`]), so a vocabulary change cannot drift past the check.

use crate::tokenizer::{ChatTemplate, ChatTokenizer};
use std::collections::HashSet;
use std::fmt;

/// Structural strings of the Qwen3-Instruct-2507 template and tokenizer (lowercase).
pub const QWEN3_CONTROL_STRINGS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_response>",
    "</tool_response>",
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "<think>",
    "</think>",
];

/// Tag names refused in any `<name`, `</name`, `< / name` spelling (spacing variants).
const REFUSED_TAG_NAMES: &[&str] = &["tool_call", "tool_response", "tools", "think"];

/// Why a tool turn was refused. Each variant is a distinct, named refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolBoundaryError {
    /// Tool text contains a template control string or structural tag. `pattern` names the
    /// refused pattern (never the hostile text itself).
    ControlSequence { pattern: String },
    /// `assistant_tool_calls` with no ids.
    NoToolCalls,
    /// Empty tool call id.
    EmptyCallId,
    /// The assistant turn declared the same call id twice (or reused an earlier call's id).
    DuplicateCallId(String),
    /// A tool result for an id no assistant turn declared.
    UnknownCallId(String),
    /// A second result for a call that already has one (replay).
    DuplicateToolResult(String),
    /// A tool result with no unanswered assistant tool call before it.
    ToolResultWithoutCall,
    /// A non-tool turn, or a render, while declared tool calls are still unanswered.
    ToolCallsPending(Vec<String>),
    /// The tool turn was asked of a template that does not define role `tool`.
    UnsupportedTemplate,
    /// The renderer itself refused (message from the renderer).
    Render(String),
}

impl fmt::Display for ToolBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ControlSequence { pattern } => {
                write!(
                    f,
                    "tool output refused: contains template control sequence {pattern}"
                )
            }
            Self::NoToolCalls => write!(f, "assistant tool-call turn declares no calls"),
            Self::EmptyCallId => write!(f, "tool call id is empty"),
            Self::DuplicateCallId(id) => write!(f, "tool call id {id:?} declared twice"),
            Self::UnknownCallId(id) => write!(f, "tool result for undeclared call id {id:?}"),
            Self::DuplicateToolResult(id) => {
                write!(f, "second tool result for call id {id:?} (replay refused)")
            }
            Self::ToolResultWithoutCall => {
                write!(
                    f,
                    "tool result with no unanswered assistant tool call before it"
                )
            }
            Self::ToolCallsPending(ids) => {
                write!(f, "tool calls still unanswered: {ids:?}")
            }
            Self::UnsupportedTemplate => {
                write!(f, "this chat template does not define role tool; refused")
            }
            Self::Render(e) => write!(f, "render refused: {e}"),
        }
    }
}

impl std::error::Error for ToolBoundaryError {}

/// Characters that render as nothing (or are control characters) and are dropped before
/// matching: soft hyphen, combining grapheme joiner, Arabic letter mark, Mongolian vowel
/// separator, zero-width and bidi marks, invisible operators, variation selectors, braille
/// blank, Hangul fillers, Khmer inherent vowels, tag characters, BOM, and all non-whitespace
/// control characters (NUL included).
fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{00AD}' | '\u{034F}' | '\u{061C}' | '\u{115F}' | '\u{1160}' | '\u{17B4}' | '\u{17B5}'
        | '\u{180B}'..='\u{180F}' | '\u{200B}'..='\u{200F}' | '\u{202A}'..='\u{202E}'
        | '\u{2060}'..='\u{206F}' | '\u{2800}' | '\u{3164}' | '\u{FE00}'..='\u{FE0F}'
        | '\u{FEFF}' | '\u{FFA0}' | '\u{FFF9}'..='\u{FFFB}' | '\u{E0000}'..='\u{E0FFF}')
        || (c.is_control() && !c.is_whitespace())
}

/// Folds text for matching: drops invisible characters, maps full-width ASCII and a small set
/// of angle/bar/slash and letter lookalikes (long s, dotless i) to ASCII, and lowercases (which
/// also maps the Kelvin sign to `k`). This is NOT full NFKC: no normalization crate is in
/// Cargo.lock (only `unicode-normalization-alignments`, unrelated) and the sovereignty rule
/// forbids adding one.
fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if is_invisible(c) {
            continue;
        }
        let c = match c {
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            '\u{2039}' | '\u{27E8}' | '\u{2329}' | '\u{3008}' => '<',
            '\u{203A}' | '\u{27E9}' | '\u{232A}' | '\u{3009}' => '>',
            '\u{2758}' | '\u{2502}' | '\u{2223}' | '\u{01C0}' => '|',
            '\u{29F8}' | '\u{2215}' => '/',
            '\u{017F}' => 's',
            '\u{0131}' => 'i',
            other => other,
        };
        out.extend(c.to_lowercase());
    }
    out
}

/// The refusal list. Construct with [`ToolGuard::from_tokenizer`] (the only public constructor).
#[derive(Debug, Clone)]
pub struct ToolGuard {
    patterns: Vec<String>,
}

impl ToolGuard {
    /// The template's structural strings only.
    pub(crate) fn qwen3() -> Self {
        Self {
            patterns: QWEN3_CONTROL_STRINGS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// The template's structural strings united with every added token (special or not) of
    /// `tok`, so the refusal list follows the real vocabulary.
    pub fn from_tokenizer(tok: &ChatTokenizer) -> Self {
        let mut g = Self::qwen3();
        for added in tok.inner().get_added_tokens_decoder().values() {
            let p = fold(&added.content);
            if !p.trim().is_empty() && !g.patterns.contains(&p) {
                g.patterns.push(p);
            }
        }
        g
    }

    fn scan(&self, text: &str) -> Result<(), ToolBoundaryError> {
        let f = fold(text);
        // Also match with every whitespace character removed, so `<tool _response>` is caught.
        let squeezed: String = f.chars().filter(|c| !c.is_whitespace()).collect();
        for p in &self.patterns {
            let squeezed_p: String = p.chars().filter(|c| !c.is_whitespace()).collect();
            if f.contains(p.as_str()) || squeezed.contains(squeezed_p.as_str()) {
                return Err(ToolBoundaryError::ControlSequence { pattern: p.clone() });
            }
        }
        // Spacing variants: `<` [ws] `|`, and `<` [ws] [`/`] [ws] tag-name (Unicode whitespace).
        let ch: Vec<char> = f.chars().collect();
        let skip_ws = |mut j: usize| {
            while j < ch.len() && ch[j].is_whitespace() {
                j += 1;
            }
            j
        };
        for i in (0..ch.len()).filter(|&i| ch[i] == '<') {
            let mut j = skip_ws(i + 1);
            if ch.get(j) == Some(&'|') {
                return Err(ToolBoundaryError::ControlSequence {
                    pattern: "<|".into(),
                });
            }
            if ch.get(j) == Some(&'/') {
                j = skip_ws(j + 1);
            }
            let rest: String = ch[j..].iter().take(16).collect();
            for name in REFUSED_TAG_NAMES {
                if rest.starts_with(name) {
                    return Err(ToolBoundaryError::ControlSequence {
                        pattern: format!("<{name}> (spacing variant)"),
                    });
                }
            }
        }
        Ok(())
    }

    /// The only way to make a [`ToolOutput`]: untrusted `text` for call `call_id`, or a named
    /// refusal.
    pub fn admit(&self, call_id: &str, text: &str) -> Result<ToolOutput, ToolBoundaryError> {
        if call_id.is_empty() {
            return Err(ToolBoundaryError::EmptyCallId);
        }
        self.scan(text)?;
        Ok(ToolOutput {
            call_id: call_id.to_string(),
            text: text.to_string(),
        })
    }
}

/// Tool text that passed the guard, bound to one call id. Private fields, no `Clone`: it cannot
/// be built around the check, and one value can be submitted once.
#[derive(Debug)]
pub struct ToolOutput {
    call_id: String,
    text: String,
}

impl ToolOutput {
    pub fn call_id(&self) -> &str {
        &self.call_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A conversation that can carry tool turns. Tool turns enter only through
/// [`ToolConversation::tool_result`], which enforces call/result pairing.
#[derive(Debug)]
pub struct ToolConversation {
    turns: Vec<(Role, String)>,
    declared: HashSet<String>,
    pending: Vec<String>,
}

impl Default for ToolConversation {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolConversation {
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            declared: HashSet::new(),
            pending: Vec::new(),
        }
    }

    fn plain(&mut self, role: Role, text: &str) -> Result<(), ToolBoundaryError> {
        if !self.pending.is_empty() {
            return Err(ToolBoundaryError::ToolCallsPending(self.pending.clone()));
        }
        self.turns.push((role, text.to_string()));
        Ok(())
    }

    /// Trusted system text.
    pub fn system(&mut self, text: &str) -> Result<(), ToolBoundaryError> {
        self.plain(Role::System, text)
    }

    /// Trusted user text.
    pub fn user(&mut self, text: &str) -> Result<(), ToolBoundaryError> {
        self.plain(Role::User, text)
    }

    /// An assistant turn with no tool calls.
    pub fn assistant(&mut self, text: &str) -> Result<(), ToolBoundaryError> {
        self.plain(Role::Assistant, text)
    }

    /// An assistant turn that calls tools with ids `call_ids`. Results are then expected, one
    /// per id, before any other turn. The assistant `text` is model output and is not scanned
    /// here (it is not tool output).
    pub fn assistant_tool_calls(
        &mut self,
        text: &str,
        call_ids: &[&str],
    ) -> Result<(), ToolBoundaryError> {
        if !self.pending.is_empty() {
            return Err(ToolBoundaryError::ToolCallsPending(self.pending.clone()));
        }
        let mut fresh = HashSet::new();
        for id in call_ids {
            if id.is_empty() {
                return Err(ToolBoundaryError::EmptyCallId);
            }
            if self.declared.contains(*id) || !fresh.insert(*id) {
                return Err(ToolBoundaryError::DuplicateCallId((*id).to_string()));
            }
        }
        if fresh.is_empty() {
            return Err(ToolBoundaryError::NoToolCalls);
        }
        for id in call_ids {
            self.declared.insert((*id).to_string());
            self.pending.push((*id).to_string());
        }
        self.turns.push((Role::Assistant, text.to_string()));
        Ok(())
    }

    /// Adds a guarded tool result. Refused unless an unanswered call with this id exists.
    pub fn tool_result(&mut self, out: ToolOutput) -> Result<(), ToolBoundaryError> {
        if self.pending.is_empty() && !self.declared.contains(&out.call_id) {
            return Err(ToolBoundaryError::ToolResultWithoutCall);
        }
        if !self.declared.contains(&out.call_id) {
            return Err(ToolBoundaryError::UnknownCallId(out.call_id));
        }
        match self.pending.iter().position(|p| *p == out.call_id) {
            Some(i) => {
                self.pending.remove(i);
                self.turns.push((Role::Tool, out.text));
                Ok(())
            }
            None => Err(ToolBoundaryError::DuplicateToolResult(out.call_id)),
        }
    }

    /// Renders with `template`. Refused while calls are unanswered, or if the conversation has
    /// tool turns and `template` is not [`ChatTemplate::ChatMlQwen3`] (other templates would
    /// silently downgrade role `tool` to `user`).
    pub fn render(&self, template: &ChatTemplate) -> Result<String, ToolBoundaryError> {
        if !self.pending.is_empty() {
            return Err(ToolBoundaryError::ToolCallsPending(self.pending.clone()));
        }
        let has_tool = self.turns.iter().any(|(r, _)| *r == Role::Tool);
        if has_tool && *template != ChatTemplate::ChatMlQwen3 {
            return Err(ToolBoundaryError::UnsupportedTemplate);
        }
        let turns: Vec<(&str, &str)> = self
            .turns
            .iter()
            .map(|(r, t)| (r.name(), t.as_str()))
            .collect();
        template
            .try_render(&turns)
            .map_err(|e| ToolBoundaryError::Render(e.to_string()))
    }
}
