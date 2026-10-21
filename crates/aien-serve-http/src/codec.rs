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
}

/// The model's own tokenizer and chat template.
pub struct HfCodec {
    tokenizer: ChatTokenizer,
    template_name: &'static str,
}

impl HfCodec {
    pub fn from_model_dir(dir: &Path) -> Result<Self, CodecError> {
        let tokenizer = ChatTokenizer::from_model_dir(dir, None)?;
        let template_name = tokenizer.template().name();
        Ok(Self {
            tokenizer,
            template_name,
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
}
