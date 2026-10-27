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

//! Text <-> token ids and chat rendering, behind a trait so the server's tests run
//! without a model directory.
//!
//! [`HfCodec`] wraps the workspace's `ChatTokenizer` (pure Rust Hugging Face
//! tokenizer plus the pinned chat templates): for the Qwen3-30B-A3B-Instruct-2507
//! checkpoint it detects the Qwen3-Instruct-2507 ChatML branch (no `<think>` block,
//! no tools) and the stop set `[151645, 151643]` from `generation_config.json`.
//! [`ByteCodec`] is the test double: one token per byte, stop id 256.

use std::fmt;
use std::path::Path;

use aien_inference_abi::tokenizer::{ChatOptions, ChatTemplate, ChatTokenizer, TokenizerError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The conversation cannot be rendered (tool role, unsupported layout, no template).
    Unrenderable(String),
    /// Encoding or decoding failed, or the text is longer than the context.
    Failed(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unrenderable(m) => write!(f, "{m}"),
            Self::Failed(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CodecError {}

impl From<TokenizerError> for CodecError {
    fn from(e: TokenizerError) -> Self {
        match e {
            TokenizerError::NoChatTemplate(_)
            | TokenizerError::ToolTurnRefused(_)
            | TokenizerError::UnsupportedChatLayout(_) => Self::Unrenderable(e.to_string()),
            other => Self::Failed(other.to_string()),
        }
    }
}

pub trait Codec: Send + Sync {
    /// Renders `(role, content)` turns ending in a generation prompt.
    fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError>;
    /// [`Self::render_chat`] with the request's template switches. Only templates that define
    /// them read them; the default ignores them.
    fn render_chat_with(
        &self,
        turns: &[(&str, &str)],
        _opts: &ChatOptions,
    ) -> Result<String, CodecError> {
        self.render_chat(turns)
    }
    /// The token that closes a thinking block, for a template whose generation prompt opens one
    /// (`<think>\n`). `None`: the model has no thinking block.
    fn reasoning_end_id(&self) -> Option<u32> {
        None
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError>;
    /// Decodes without special tokens. A sequence cut inside a multi-byte
    /// character must decode with a trailing U+FFFD, never fail.
    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError>;
    fn stop_token_ids(&self) -> Vec<u32>;
    /// The bytes one token stands for (`/v1/tokenize-text`). The default decodes the token
    /// alone, which is exact unless the token is part of a multi-byte character.
    fn token_bytes(&self, id: u32) -> Result<Vec<u8>, CodecError> {
        Ok(self.decode(&[id])?.into_bytes())
    }
}

/// The model's own tokenizer and chat template.
pub struct HfCodec {
    tokenizer: ChatTokenizer,
    template_name: &'static str,
    /// The vocabulary is GPT-2 byte-level (a space is the piece `Ġ`): pieces map to bytes.
    byte_level: bool,
}

impl HfCodec {
    pub fn from_model_dir(dir: &Path) -> Result<Self, CodecError> {
        let tokenizer = ChatTokenizer::from_model_dir(dir, None)?;
        let template_name = tokenizer.template().name();
        let byte_level = match tokenizer.encode(" a").ok().as_deref() {
            Some([first, ..]) => tokenizer
                .id_to_token(*first)
                .is_some_and(|p| p.starts_with('\u{120}')),
            _ => false,
        };
        Ok(Self {
            tokenizer,
            template_name,
            byte_level,
        })
    }

    /// The detected chat layout, e.g. `chatml-qwen3` (for the operator log).
    pub fn template_name(&self) -> &'static str {
        self.template_name
    }
}

impl Codec for HfCodec {
    fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError> {
        Ok(self.tokenizer.try_format_chat(turns)?)
    }

    fn render_chat_with(
        &self,
        turns: &[(&str, &str)],
        opts: &ChatOptions,
    ) -> Result<String, CodecError> {
        Ok(self.tokenizer.try_format_chat_with(turns, opts)?)
    }

    fn reasoning_end_id(&self) -> Option<u32> {
        if !matches!(
            self.tokenizer.template(),
            ChatTemplate::Qwen38 | ChatTemplate::Qwen36
        ) {
            return None;
        }
        match self
            .tokenizer
            .encode_with_special("</think>", true)
            .ok()?
            .as_slice()
        {
            [id] => Some(*id),
            _ => None,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
        Ok(self.tokenizer.encode_with_special(text, true)?)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
        Ok(self.tokenizer.decode_opts(tokens, true)?)
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        self.tokenizer.stop_token_ids().to_vec()
    }

    /// A byte-level BPE piece (`Ġ` for a space, ...) maps back to its exact bytes, even half a
    /// character; a piece outside the byte-level alphabet falls back to decoding the token.
    fn token_bytes(&self, id: u32) -> Result<Vec<u8>, CodecError> {
        if let Some(b) = self
            .byte_level
            .then(|| self.tokenizer.id_to_token(id))
            .flatten()
            .and_then(|p| byte_level_bytes(&p))
        {
            return Ok(b);
        }
        Ok(self.decode(&[id])?.into_bytes())
    }
}

/// Inverts GPT-2's `bytes_to_unicode`: bytes `!`..=`~`, `¡`..=`¬` and `®`..=`ÿ` stand for
/// themselves; the other 68 bytes, in order, are U+0100 onward.
pub fn byte_level_bytes(piece: &str) -> Option<Vec<u8>> {
    let direct = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let shifted: Vec<u8> = (0u32..=255).filter(|&b| !direct(b)).map(|b| b as u8).collect();
    piece
        .chars()
        .map(|c| {
            let c = c as u32;
            if direct(c) {
                Some(c as u8)
            } else {
                c.checked_sub(256)
                    .and_then(|i| shifted.get(i as usize).copied())
            }
        })
        .collect()
}

/// Test codec: token id = byte value, stop id 256. The chat layout is
/// `<role>content</role>\n` per turn then `<assistant>`.
#[derive(Debug, Default, Clone)]
pub struct ByteCodec;

impl ByteCodec {
    pub const STOP: u32 = 256;
}

impl Codec for ByteCodec {
    fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError> {
        if turns.iter().any(|(r, _)| r.eq_ignore_ascii_case("tool")) {
            return Err(CodecError::Unrenderable("role tool refused".into()));
        }
        let mut out = String::new();
        for (role, content) in turns {
            out.push_str(&format!("<{role}>{content}</{role}>\n"));
        }
        out.push_str("<assistant>");
        Ok(out)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter(|&&t| t < 256)
            .map(|&t| t as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        vec![Self::STOP]
    }

    fn token_bytes(&self, id: u32) -> Result<Vec<u8>, CodecError> {
        u8::try_from(id)
            .map(|b| vec![b])
            .map_err(|_| CodecError::Failed(format!("token {id} has no bytes")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_codec_round_trips_and_marks_partial_utf8() {
        let c = ByteCodec;
        let ids = c.encode("héllo").unwrap();
        assert_eq!(ids.len(), 6);
        assert_eq!(c.decode(&ids).unwrap(), "héllo");
        let cut = c.decode(&ids[..2]).unwrap();
        assert!(cut.ends_with('\u{FFFD}'), "{cut:?}");
    }

    #[test]
    fn byte_codec_renders_turns_and_refuses_tool() {
        let c = ByteCodec;
        let s = c.render_chat(&[("system", "s"), ("user", "u")]).unwrap();
        assert_eq!(s, "<system>s</system>\n<user>u</user>\n<assistant>");
        assert!(matches!(
            c.render_chat(&[("tool", "x")]),
            Err(CodecError::Unrenderable(_))
        ));
    }

    #[test]
    fn byte_level_pieces_map_back_to_bytes() {
        assert_eq!(byte_level_bytes("\u{120}a"), Some(b" a".to_vec()));
        assert_eq!(byte_level_bytes("\u{10A}"), Some(vec![b'\n']));
        assert_eq!(byte_level_bytes("<|im_end|>"), Some(b"<|im_end|>".to_vec()));
        // `é` split across two pieces: the high byte alone is exact, no U+FFFD.
        assert_eq!(byte_level_bytes("\u{C3}"), Some(vec![0xC3]));
        assert_eq!(byte_level_bytes("\u{2581}x"), None);
        assert_eq!(ByteCodec.token_bytes(0xC3).unwrap(), vec![0xC3]);
        assert!(ByteCodec.token_bytes(ByteCodec::STOP).is_err());
    }
}
