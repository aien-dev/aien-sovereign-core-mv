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

//! Pure Rust tokenizer and chat template formatter for the Llama model family
//! (TinyLlama zephyr-style chat, Llama 3 instruct chat, ChatML as shipped by SmolLM2).
//! Wraps Hugging Face tokenizers crate without Python runtime dependencies.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Loud error enum for tokenizer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    LoadError(String),
    /// Chat rendering was asked of a plain (base) model that has no chat template. Not a
    /// load failure: the tokenizer loaded fine.
    NoChatTemplate(String),
    EncodeError(String),
    DecodeError(String),
    ContextLengthExceeded {
        len: usize,
        max: usize,
    },
    /// Role `tool` offered to the raw `ChatTurn` path; tool turns must go through
    /// `tool_boundary::ToolConversation` (issue #310).
    ToolTurnRefused(String),
    /// A message combination the pinned Qwen3-8B template layout does not cover (thinking
    /// enabled, tool turns, tool calls, reasoning content, unknown roles, an assistant turn
    /// to continue). Refused by name, never approximated (sc#384).
    UnsupportedChatLayout(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadError(e) => write!(f, "Failed to load tokenizer: {}", e),
            Self::NoChatTemplate(e) => write!(f, "{}", e),
            Self::EncodeError(e) => write!(f, "Tokenization encoding failed: {}", e),
            Self::DecodeError(e) => write!(f, "Tokenization decoding failed: {}", e),
            Self::ToolTurnRefused(e) => write!(f, "{}", e),
            Self::UnsupportedChatLayout(e) => write!(f, "unsupported chat layout: {}", e),
            Self::ContextLengthExceeded { len, max } => {
                write!(
                    f,
                    "Sequence length {} exceeds maximum context length {}",
                    len, max
                )
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// The chat layout a model was fine-tuned on. Selected from the model's own chat template
/// text (no template engine): each variant renders the one layout that template produces
/// for plain user/system/assistant turns ending in a generation prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// TinyLlama-Chat (zephyr): `<|user|>\n{content}</s>\n<|assistant|>\n`.
    Zephyr,
    /// Llama 3 instruct: `<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>`
    /// then `<|start_header_id|>assistant<|end_header_id|>\n\n`. No system block is added
    /// (the Hugging Face template's default "Cutting Knowledge Date / Today Date" block
    /// depends on the wall clock, so it is left out).
    Llama3,
    /// ChatML exactly as the publisher's template renders it (SmolLM2-Instruct, and
    /// Qwen3-4B-Instruct-2507 without tools): per turn `<|im_start|>{role}\n{content}<|im_end|>\n`,
    /// then `<|im_start|>assistant\n`. Contents are kept verbatim and empty turns are kept
    /// (the template neither trims nor skips).
    /// When the template carries a default system message and the first turn is not a
    /// system turn, that message is rendered first as a system turn, as the template does.
    /// No BOS is part of the text; the model's tokenizer decides whether `encode` adds one
    /// (SmolLM2's `tokenizer.json` has no post-processor, so it adds none).
    ChatMl {
        /// The default system message read from the model's own template, if it has one.
        default_system: Option<Arc<str>>,
    },
    /// Qwen3-4B-Instruct-2507: the plain ChatML layout above (no default system message) plus
    /// the publisher template's handling of role `tool`: a run of consecutive `tool` turns is
    /// one user turn, `<|im_start|>user`, then per turn `\n<tool_response>\n{content}\n</tool_response>`,
    /// then `<|im_end|>\n`. Selected only by the pinned template text in `QWEN3_INSTRUCT_2507`.
    ChatMlQwen3,
    /// Qwen3-8B (and any model shipping the same template text), thinking mode explicitly OFF:
    /// the pinned template `QWEN3_8B` rendered with `enable_thinking=false`. System and user
    /// turns are ChatML verbatim; earlier assistant turns are plain; the generation prompt is
    /// the assistant header plus an empty think block. Everything else (thinking on, tool
    /// turns, tool calls, reasoning content, unknown roles, a trailing assistant turn) is
    /// refused with `TokenizerError::UnsupportedChatLayout`. Selected only by the pinned text.
    ChatMlQwen3NoThink,
    /// The model directory has no chat template at all (a base model). Plain text encode
    /// and decode work; every chat render is refused (`try_render` errors with
    /// "no chat template"). Chosen only when neither `tokenizer_config.json` `chat_template`
    /// nor `chat_template.jinja` exists; a template that exists but is unrecognized is
    /// refused at load.
    None,
}

/// The ChatML layouts this engine renders, as the exact Jinja text a model ships in
/// `tokenizer_config.json` `chat_template` (the SmolLM2-Instruct template, and the same
/// without its default system block). `{SYSTEM}` stands for the default system message
/// (plain text: no quote, backslash, Jinja brace or ChatML marker). Any other ChatML
/// template (tools, trimming, another generation prompt) is refused, not approximated.
const CHATML_WITH_DEFAULT_SYSTEM: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n{SYSTEM}<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
const CHATML_PLAIN: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// The Qwen3-4B-Instruct-2507 chat template, byte for byte (revision cdbee75f, Apache-2.0; see
/// `fixtures/chat-templates/qwen3-4b-instruct-2507/PROVENANCE.md`). Without tools and without
/// assistant tool calls (this engine passes neither) it renders exactly the plain ChatML layout:
/// no default system message, contents verbatim, generation prompt `<|im_start|>assistant\n`.
/// Any other Qwen template text (tools in use, thinking blocks, another revision) is refused.
const QWEN3_INSTRUCT_2507: &str =
    include_str!("../fixtures/chat-templates/qwen3-4b-instruct-2507/chat_template.jinja");

/// The Qwen3-8B chat template, byte for byte (Hugging Face `Qwen/Qwen3-8B` revision
/// b968826d9c46dd6066d109eabc6255188de91218, Apache-2.0; see
/// `fixtures/chat-templates/qwen3-8b/PROVENANCE.md`). It has an `enable_thinking` switch; this
/// engine always renders the `enable_thinking=false` form ([`ChatTemplate::ChatMlQwen3NoThink`]).
const QWEN3_8B: &str = include_str!("../fixtures/chat-templates/qwen3-8b/chat_template.jinja");

/// Matches a chat template text against the ChatML layouts above. Returns the layout
/// with its default system message (None for the plain layout), or a refusal.
fn parse_chatml(template_text: &str) -> Result<ChatTemplate, TokenizerError> {
    let text = template_text.trim();
    if text == QWEN3_INSTRUCT_2507.trim() {
        return Ok(ChatTemplate::ChatMlQwen3);
    }
    if text == QWEN3_8B.trim() {
        return Ok(ChatTemplate::ChatMlQwen3NoThink);
    }
    if text == CHATML_PLAIN {
        return Ok(ChatTemplate::ChatMl {
            default_system: None,
        });
    }
    let (head, tail) = CHATML_WITH_DEFAULT_SYSTEM
        .split_once("{SYSTEM}")
        .expect("placeholder in the ChatML layout");
    let system = text
        .strip_prefix(head)
        .and_then(|rest| rest.strip_suffix(tail))
        .filter(|s| {
            !s.is_empty()
                && !s.contains(['\'', '\\', '{', '}'])
                && !s.contains("<|im_start|>")
                && !s.contains("<|im_end|>")
        });
    match system {
        Some(system) => Ok(ChatTemplate::ChatMl {
            default_system: Some(Arc::from(system)),
        }),
        None => Err(TokenizerError::LoadError(
            "unsupported ChatML chat template: only the plain ChatML layout (each turn \
             <|im_start|>role, newline, content, <|im_end|>, newline; generation prompt \
             <|im_start|>assistant and a newline), optionally with one plain-text default \
             system message, is rendered; this template differs (tools, trimming or another \
             layout)"
                .to_string(),
        )),
    }
}

/// The publisher's ChatML rendering for `(role, content)` turns ending in a generation
/// prompt (see [`ChatTemplate::ChatMl`]). Roles are normalized as for the other layouts.
fn render_chatml(default_system: Option<&str>, qwen_tools: bool, turns: &[(&str, &str)]) -> String {
    let role_of = |role: &str| match role.trim().to_ascii_lowercase().as_str() {
        "system" => "system",
        "assistant" => "assistant",
        "tool" if qwen_tools => "tool",
        _ => "user",
    };
    let mut out = String::new();
    if let (Some(system), Some((first_role, _))) = (default_system, turns.first()) {
        if role_of(first_role) != "system" {
            out.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
        }
    }
    for (i, (role, content)) in turns.iter().enumerate() {
        let role = role_of(role);
        if role == "tool" {
            // Qwen3 template: one user turn per run of tool turns.
            if i == 0 || role_of(turns[i - 1].0) != "tool" {
                out.push_str("<|im_start|>user");
            }
            out.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
            if turns
                .get(i + 1)
                .is_none_or(|(next, _)| role_of(next) != "tool")
            {
                out.push_str("<|im_end|>\n");
            }
        } else {
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
    }
    if !ends_in_assistant(turns) {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// True when the last turn is a non-empty assistant turn: generation then continues it
/// instead of opening a new assistant turn.
fn ends_in_assistant(turns: &[(&str, &str)]) -> bool {
    turns.last().is_some_and(|(role, content)| {
        role.trim().eq_ignore_ascii_case("assistant") && !content.trim().is_empty()
    })
}

/// Oracle-checked rendering of the pinned Qwen3-8B template with `enable_thinking=false`,
/// `tools=None`, `add_generation_prompt=true`. Branches taken (template `QWEN3_8B`):
/// 1. `tools` falsy: the system header is `<|im_start|>system\n{content}<|im_end|>\n` only when
///    `messages[0].role == 'system'`; in the loop that first system turn then matches no branch
///    (the `system and not loop.first` test fails), so it is emitted exactly once.
/// 2. user, and system after the first turn: `<|im_start|>{role}\n{content}<|im_end|>\n`.
/// 3. assistant, `reasoning_content` undefined and no `</think>` in content: content stays as is.
///    At `index <= last_query_index` the turn is `<|im_start|>assistant\n{content}`; at
///    `index > last_query_index` and not last the same text (no reasoning); the last message being
///    an assistant turn takes the `<think>` block branch and closes the turn, so it is refused.
///    No `tool_calls`, so the turn ends with `<|im_end|>\n`.
/// 4. generation prompt with `enable_thinking is false`:
///    `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
///
/// Tool turns, tool calls, reasoning, thinking-on, tool-response-shaped user text, unknown roles,
/// and an empty conversation are refused by name.
fn render_qwen3_nothink(turns: &[(&str, &str)]) -> Result<String, TokenizerError> {
    let refuse = |why: &str| Err(TokenizerError::UnsupportedChatLayout(why.to_string()));
    if turns.is_empty() {
        return refuse("empty conversation");
    }
    let mut out = String::new();
    for (i, (role, content)) in turns.iter().enumerate() {
        match *role {
            "system" | "user" => {
                if *role == "user"
                    && content.starts_with("<tool_response>")
                    && content.ends_with("</tool_response>")
                {
                    return refuse("user text shaped like a tool response (changes the template's last-query index)");
                }
                out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            "assistant" => {
                if i + 1 == turns.len() {
                    return refuse("trailing assistant turn (the template would add a think block and close it; continuation is not supported)");
                }
                if content.contains("<think>") || content.contains("</think>") {
                    return refuse("thinking markers inside an assistant turn (reasoning content)");
                }
                out.push_str(&format!("<|im_start|>assistant\n{content}<|im_end|>\n"));
            }
            "tool" => {
                return Err(TokenizerError::ToolTurnRefused(
                    "role tool refused: the pinned Qwen3-8B layout covers no tool turns".into(),
                ))
            }
            _ => {
                return refuse("role other than exactly system, user or assistant");
            }
        }
    }
    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok(out)
}

impl ChatTemplate {
    /// Qwen3-8B with `enable_thinking` requested explicitly. Only `false` renders; `true` is
    /// refused (the engine does not parse or budget a thinking block).
    pub fn try_render_thinking(
        &self,
        turns: &[(&str, &str)],
        enable_thinking: bool,
    ) -> Result<String, TokenizerError> {
        if enable_thinking {
            return Err(TokenizerError::UnsupportedChatLayout(
                "thinking enabled is not supported".into(),
            ));
        }
        self.try_render(turns)
    }
}
fn no_chat_template_error() -> TokenizerError {
    TokenizerError::NoChatTemplate(
        "no chat template: this is a plain (base) model, chat rendering is refused".to_string(),
    )
}

impl ChatTemplate {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Zephyr => "zephyr (TinyLlama chat)",
            Self::Llama3 => "llama3 (Llama 3 instruct)",
            Self::ChatMl { .. } => "chatml (<|im_start|> / <|im_end|>)",
            Self::ChatMlQwen3 => "chatml-qwen3 (<|im_start|> / <|im_end|>, tool responses)",
            Self::ChatMlQwen3NoThink => "chatml-qwen3-nothink (Qwen3-8B, thinking off)",
            Self::None => "none (plain text model)",
        }
    }

    /// Recognizes the layout from a Jinja chat template's text. A ChatML template must be
    /// one of the exact layouts this engine renders (see [`ChatTemplate::ChatMl`]).
    pub fn detect(template_text: &str) -> Result<Self, TokenizerError> {
        if template_text.contains("<|start_header_id|>") && template_text.contains("<|eot_id|>") {
            Ok(Self::Llama3)
        } else if template_text.contains("<|user|>") && template_text.contains("<|assistant|>") {
            Ok(Self::Zephyr)
        } else if template_text.contains("<|im_start|>") && template_text.contains("<|im_end|>") {
            parse_chatml(template_text)
        } else {
            Err(TokenizerError::LoadError(
                "unsupported chat template: no zephyr (<|user|>), llama3 \
                 (<|start_header_id|>) or chatml (<|im_start|>) markers found"
                    .to_string(),
            ))
        }
    }

    /// The default system message the model's template adds (ChatML only).
    pub fn default_system(&self) -> Option<&str> {
        match self {
            Self::ChatMl { default_system } => default_system.as_deref(),
            _ => None,
        }
    }

    /// Renders `(role, content)` turns; roles other than system and assistant are user
    /// turns. Generation continues from an assistant header unless the last turn is a
    /// non-empty assistant turn. Zephyr and Llama 3: contents are trimmed and empty turns
    /// skipped, and the BOS token is not part of the text (`encode` adds exactly one
    /// through the tokenizer's post-processor). ChatML: see [`ChatTemplate::ChatMl`].
    pub fn render(&self, turns: &[(&str, &str)]) -> String {
        self.try_render(turns).expect("no chat template")
    }

    /// Like [`ChatTemplate::render`], but a plain model (no chat template) is refused with
    /// an error containing "no chat template" instead of a layout being invented.
    pub fn try_render(&self, turns: &[(&str, &str)]) -> Result<String, TokenizerError> {
        if let Self::None = self {
            return Err(no_chat_template_error());
        }
        if let Self::ChatMl { default_system } = self {
            return Ok(render_chatml(default_system.as_deref(), false, turns));
        }
        if let Self::ChatMlQwen3 = self {
            return Ok(render_chatml(None, true, turns));
        }
        if let Self::ChatMlQwen3NoThink = self {
            return render_qwen3_nothink(turns);
        }
        let mut out = String::new();
        for (role, content) in turns {
            let body = content.trim();
            if body.is_empty() {
                continue;
            }
            let role = match role.trim().to_ascii_lowercase().as_str() {
                "system" => "system",
                "assistant" => "assistant",
                _ => "user",
            };
            match self {
                Self::Zephyr => out.push_str(&format!("<|{role}|>\n{body}</s>\n")),
                Self::Llama3 => out.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{body}<|eot_id|>"
                )),
                Self::ChatMl { .. } | Self::ChatMlQwen3 | Self::ChatMlQwen3NoThink => {
                    unreachable!("ChatML returned above")
                }
                Self::None => unreachable!("plain model refused above"),
            }
        }
        if !ends_in_assistant(turns) {
            out.push_str(match self {
                Self::Zephyr => "<|assistant|>\n",
                Self::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
                Self::ChatMl { .. } | Self::ChatMlQwen3 | Self::ChatMlQwen3NoThink => {
                    unreachable!("ChatML returned above")
                }
                Self::None => unreachable!("plain model refused above"),
            });
        }
        Ok(out)
    }
}

/// Role `tool` is refused on every `ChatTokenizer` formatting path (issue #310): untrusted tool
/// output must enter through `tool_boundary::ToolConversation`. `ChatTemplate::render` and
/// `try_render` stay upstream-exact (the #309 oracle uses them) and are not a safe route.
pub fn refuse_tool_role(turns: &[(&str, &str)]) -> Result<(), TokenizerError> {
    if turns
        .iter()
        .any(|(r, _)| r.trim().eq_ignore_ascii_case("tool"))
    {
        return Err(TokenizerError::ToolTurnRefused(
            "role tool refused: use tool_boundary::ToolConversation".into(),
        ));
    }
    Ok(())
}

/// Pure Rust wrapper around Hugging Face tokenizers with the model's chat template and
/// end-of-sequence set. `from_file`/`from_bytes` give the TinyLlama defaults;
/// `from_model_dir` reads them from the model directory.
#[derive(Clone)]
pub struct ChatTokenizer {
    inner: tokenizers::Tokenizer,
    template: ChatTemplate,
    stop_token_ids: Vec<u32>,
    max_context_len: usize,
}

/// The name this tokenizer had while it served TinyLlama only.
pub type TinyLlamaTokenizer = ChatTokenizer;

fn read_json(path: &Path) -> Result<Option<serde_json::Value>, TokenizerError> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map(Some).map_err(|e| {
            TokenizerError::LoadError(format!("{} is not valid JSON: {}", path.display(), e))
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(TokenizerError::LoadError(format!(
            "Failed to read {}: {}",
            path.display(),
            e
        ))),
    }
}

/// `eos_token_id` of a Hugging Face config as a list (it is a number or an array).
pub fn eos_token_ids_from_json(value: &serde_json::Value) -> Option<Vec<u32>> {
    match value.get("eos_token_id")? {
        serde_json::Value::Number(n) => n.as_u64().map(|id| vec![id as u32]),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| v.as_u64().map(|id| id as u32))
            .collect(),
        _ => None,
    }
}

impl ChatTokenizer {
    /// Beginning of Sequence token ID pinned for TinyLlama (<s>).
    pub const BOS_TOKEN_ID: u32 = 1;
    /// End of Sequence token ID pinned for TinyLlama (</s>).
    pub const EOS_TOKEN_ID: u32 = 2;
    /// Unknown / Padding token ID pinned for TinyLlama (<unk>).
    pub const UNK_TOKEN_ID: u32 = 0;
    /// Maximum context window supported by TinyLlama-1.1B.
    pub const MAX_CONTEXT_LEN: usize = 2048;

    /// Loads tokenizer directly from a local tokenizer.json file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let p = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(p).map_err(|e| {
            TokenizerError::LoadError(format!("Failed to load from {}: {}", p.display(), e))
        })?;
        Ok(Self::tinyllama(inner))
    }

    /// Loads tokenizer directly from a JSON byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| {
            TokenizerError::LoadError(format!("Failed to parse tokenizer bytes: {}", e))
        })?;
        Ok(Self::tinyllama(inner))
    }

    fn tinyllama(inner: tokenizers::Tokenizer) -> Self {
        Self {
            inner,
            template: ChatTemplate::Zephyr,
            stop_token_ids: vec![Self::EOS_TOKEN_ID],
            max_context_len: Self::MAX_CONTEXT_LEN,
        }
    }

    /// Loads `tokenizer_json` (default `<dir>/tokenizer.json`) with the chat template and
    /// end-of-sequence set of the model directory `dir`:
    /// - template: detected from `chat_template` in `tokenizer_config.json`, else from
    ///   `chat_template.jinja`;
    /// - stop set: `eos_token_id` of `generation_config.json`, else of `config.json`;
    /// - context limit: `max_position_embeddings` of `config.json` (default 2048).
    pub fn from_model_dir(
        dir: &Path,
        tokenizer_json: Option<&Path>,
    ) -> Result<Self, TokenizerError> {
        let default_json = dir.join("tokenizer.json");
        let mut tokenizer = Self::from_file(tokenizer_json.unwrap_or(&default_json))?;
        // A present `chat_template` key must be a string; any other form (the HF list
        // form, null, an object) is an unrecognized template and never falls back to plain.
        let config_template = match read_json(&dir.join("tokenizer_config.json"))?
            .as_ref()
            .and_then(|c| c.get("chat_template"))
        {
            None => None,
            Some(serde_json::Value::String(text)) => Some(text.clone()),
            Some(other) => {
                let found = match other {
                    serde_json::Value::Array(_) => "a list",
                    serde_json::Value::Object(_) => "an object",
                    serde_json::Value::Null => "null",
                    _ => "a non-string value",
                };
                return Err(TokenizerError::LoadError(format!(
                    "unrecognized chat template form in {}: expected a string, found {}",
                    dir.join("tokenizer_config.json").display(),
                    found
                )));
            }
        };
        let jinja_path = dir.join("chat_template.jinja");
        let template_text = match config_template {
            Some(text) => Some(text),
            None if jinja_path.exists() => {
                Some(std::fs::read_to_string(&jinja_path).map_err(|e| {
                    TokenizerError::LoadError(format!(
                        "unreadable chat template {}: {}",
                        jinja_path.display(),
                        e
                    ))
                })?)
            }
            None => None,
        };
        // Plain state only when neither source exists; an existing template must be recognized.
        tokenizer.template = match template_text {
            Some(text) => ChatTemplate::detect(&text)?,
            None => ChatTemplate::None,
        };
        let config = read_json(&dir.join("config.json"))?;
        let stop = read_json(&dir.join("generation_config.json"))?
            .as_ref()
            .and_then(eos_token_ids_from_json)
            .or_else(|| config.as_ref().and_then(eos_token_ids_from_json))
            .ok_or_else(|| {
                TokenizerError::LoadError(format!(
                    "no eos_token_id in {}/generation_config.json or config.json",
                    dir.display()
                ))
            })?;
        if stop.is_empty() {
            return Err(TokenizerError::LoadError(format!(
                "empty eos_token_id in {}",
                dir.display()
            )));
        }
        tokenizer.stop_token_ids = stop;
        if let Some(max) = config
            .as_ref()
            .and_then(|c| c.get("max_position_embeddings"))
            .and_then(|v| v.as_u64())
        {
            tokenizer.max_context_len = max as usize;
        }
        Ok(tokenizer)
    }

    /// The chat layout this tokenizer renders.
    pub fn template(&self) -> ChatTemplate {
        self.template.clone()
    }

    /// Token ids that end generation (the model's `eos_token_id` set).
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    /// Renders `(role, content)` turns with this model's chat template.
    pub fn format_chat(&self, turns: &[(&str, &str)]) -> String {
        refuse_tool_role(turns).expect("tool role refused");
        self.template.render(turns)
    }

    /// Fallible [`ChatTokenizer::format_chat`]: refuses a plain model ("no chat template").
    pub fn try_format_chat(&self, turns: &[(&str, &str)]) -> Result<String, TokenizerError> {
        refuse_tool_role(turns)?;
        self.template.try_render(turns)
    }

    /// Formats a conversation according to the canonical TinyLlama chat template:
    /// `<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n`
    pub fn format_chat_prompt(&self, system: Option<&str>, user: &str) -> String {
        Self::format_prompt(system, user)
    }

    /// Static formatter for the canonical TinyLlama chat template.
    pub fn format_prompt(system: Option<&str>, user: &str) -> String {
        match system {
            Some(sys) if !sys.trim().is_empty() => {
                format!(
                    "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                    sys.trim(),
                    user.trim()
                )
            }
            _ => {
                format!("<|user|>\n{}</s>\n<|assistant|>\n", user.trim())
            }
        }
    }

    /// Encodes text into token IDs including configured special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_with_special(text, true)
    }

    /// Encodes text with explicit control over special tokens insertion.
    pub fn encode_with_special(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
        let tokens = encoding.get_ids().to_vec();
        if tokens.len() > self.max_context_len {
            return Err(TokenizerError::ContextLengthExceeded {
                len: tokens.len(),
                max: self.max_context_len,
            });
        }
        Ok(tokens)
    }

    /// Decodes a sequence of token IDs back into recognizable UTF-8 text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        self.decode_opts(tokens, false)
    }

    /// Decodes tokens with option to skip special tokens.
    pub fn decode_opts(
        &self,
        tokens: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }

    /// Checks if a token ID signifies generation stopping (EOS).
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// Checks if a token ID signifies sequence beginning (BOS).
    pub fn is_bos(&self, token_id: u32) -> bool {
        token_id == Self::BOS_TOKEN_ID
    }

    /// Checks if a token is a terminal or stop token.
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == Self::EOS_TOKEN_ID || token_id == Self::UNK_TOKEN_ID
    }

    /// Returns the vocabulary size of the underlying model.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Returns the token ID for a given string token if present in vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Returns the string representation for a given token ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Returns a reference to the underlying tokenizers::Tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_special_tokens() {
        assert_eq!(TinyLlamaTokenizer::BOS_TOKEN_ID, 1);
        assert_eq!(TinyLlamaTokenizer::EOS_TOKEN_ID, 2);
        assert_eq!(TinyLlamaTokenizer::UNK_TOKEN_ID, 0);
        assert_eq!(TinyLlamaTokenizer::MAX_CONTEXT_LEN, 2048);
    }

    #[test]
    fn test_chat_template_with_system() {
        let system = "You are a sovereign AI assistant.";
        let user = "Explain the role of an operating system in one sentence.";
        let formatted = TinyLlamaTokenizer::format_prompt(Some(system), user);

        let expected = "<|system|>\nYou are a sovereign AI assistant.</s>\n<|user|>\nExplain the role of an operating system in one sentence.</s>\n<|assistant|>\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_chat_template_without_system() {
        let user = "Hello TinyLlama!";
        let formatted = TinyLlamaTokenizer::format_prompt(None, user);

        let expected = "<|user|>\nHello TinyLlama!</s>\n<|assistant|>\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_chat_template_with_empty_system() {
        let user = "Hello TinyLlama!";
        let formatted = TinyLlamaTokenizer::format_prompt(Some("   "), user);

        let expected = "<|user|>\nHello TinyLlama!</s>\n<|assistant|>\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_stop_and_eos_token_checks() {
        // We can construct a mock tokenizer instance or test static methods
        assert_eq!(TinyLlamaTokenizer::EOS_TOKEN_ID, 2);
        assert_eq!(TinyLlamaTokenizer::BOS_TOKEN_ID, 1);
        assert_eq!(TinyLlamaTokenizer::UNK_TOKEN_ID, 0);
    }

    /// HuggingFaceTB/SmolLM2-1.7B-Instruct tokenizer_config.json `chat_template` at revision
    /// 31b70e2e869a, verbatim (the decoded JSON string; its newlines are real newlines).
    const SMOLLM2_TEMPLATE: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn smollm2_template_is_chatml_with_its_default_system_message() {
        let t = ChatTemplate::detect(SMOLLM2_TEMPLATE).unwrap();
        assert_eq!(
            t.default_system(),
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
        );
        assert!(t.name().starts_with("chatml"));
        // Expected strings: transformers 5.17.0 apply_chat_template(add_generation_prompt=True)
        // on the same messages with the pinned tokenizer_config.json (scratch run, outside
        // the repository).
        assert_eq!(
            t.render(&[("user", "Goal: X\nAuthorized workspace: /w")]),
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
             <|im_start|>user\nGoal: X\nAuthorized workspace: /w<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            t.render(&[("system", "S"), ("user", "U")]),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
        // Contents verbatim: the template does not trim.
        assert!(t
            .render(&[("user", "  Hi \n")])
            .contains("<|im_start|>user\n  Hi \n<|im_end|>\n"));
    }

    #[test]
    fn chatml_without_default_system_and_refusals() {
        let plain = SMOLLM2_TEMPLATE.replace(
            "{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}",
            "",
        );
        let t = ChatTemplate::detect(&plain).unwrap();
        assert_eq!(t.default_system(), None);
        assert_eq!(
            t.render(&[("user", "U")]),
            "<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
        // A ChatML template with other logic (tools, trimming) is refused, not approximated.
        let tools = SMOLLM2_TEMPLATE.replace(
            "{% for message in messages %}",
            "{% if tools %}{{ tools | tojson }}{% endif %}{% for message in messages %}",
        );
        let err = ChatTemplate::detect(&tools).unwrap_err().to_string();
        assert!(err.contains("unsupported ChatML"), "{err}");
        let trimmed = SMOLLM2_TEMPLATE.replace("message['content']", "message['content'] | trim");
        assert!(ChatTemplate::detect(&trimmed).is_err());
        // A default system message with a quote cannot be read as plain text.
        let quoted = SMOLLM2_TEMPLATE.replace("named SmolLM", "named 'SmolLM'");
        assert!(ChatTemplate::detect(&quoted).is_err());
        // Zephyr and Llama 3 detection is unchanged.
        assert_eq!(
            ChatTemplate::detect("<|user|> ... <|assistant|>").unwrap(),
            ChatTemplate::Zephyr
        );
        assert_eq!(
            ChatTemplate::detect("<|start_header_id|> <|eot_id|>").unwrap(),
            ChatTemplate::Llama3
        );
        assert!(ChatTemplate::detect("{{ messages }}").is_err());
    }

    #[test]
    fn test_synthetic_tokenizer_roundtrip() {
        // Construct a minimal BPE tokenizer JSON for testing encode and decode
        let minimal_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "unk_token": "<unk>",
                "vocab": {
                    "<unk>": 0,
                    "<s>": 1,
                    "</s>": 2,
                    "hello": 3,
                    "world": 4
                }
            }
        }"#;

        let tokenizer = TinyLlamaTokenizer::from_bytes(minimal_json.as_bytes()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.token_to_id("hello"), Some(3));
        assert_eq!(tokenizer.token_to_id("world"), Some(4));
        assert_eq!(tokenizer.id_to_token(1), Some("<s>".to_string()));
        assert_eq!(tokenizer.id_to_token(2), Some("</s>".to_string()));

        let tokens = tokenizer.encode("hello world").unwrap();
        assert_eq!(tokens, vec![3, 4]);

        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, "hello world");
        assert!(tokenizer.is_eos(2));
        assert!(!tokenizer.is_eos(3));
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_stop_token(2));
        assert!(tokenizer.is_stop_token(0));
    }
}
