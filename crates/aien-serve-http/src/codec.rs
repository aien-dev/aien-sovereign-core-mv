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

use aien_inference_abi::tokenizer::{ChatTokenizer, TokenizerError};

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
    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError>;
    /// Decodes without special tokens. A sequence cut inside a multi-byte
    /// character must decode with a trailing U+FFFD, never fail.
    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError>;
    fn stop_token_ids(&self) -> Vec<u32>;
    /// Byte length of the stable part of a rendered chat: everything except the closing of
    /// the last turn and the trailing generation prompt. `None` when the layout is not
    /// recognised (prefill-ahead then refuses rather than guess).
    fn stable_len(&self, _rendered: &str) -> Option<usize> {
        None
    }
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

    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
        Ok(self.tokenizer.encode_with_special(text, true)?)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
        Ok(self.tokenizer.decode_opts(tokens, true)?)
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        self.tokenizer.stop_token_ids().to_vec()
    }

    fn stable_len(&self, rendered: &str) -> Option<usize> {
        // ChatML: the last turn closes with `<|im_end|>\n`, then the assistant header.
        rendered.strip_suffix("<|im_end|>\n<|im_start|>assistant\n").map(str::len)
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

    fn stable_len(&self, rendered: &str) -> Option<usize> {
        // Layout `<role>content</role>\n` per turn then `<assistant>`.
        rendered.strip_suffix("<assistant>").and_then(|r| r.trim_end_matches('\n').rfind("</"))
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

#[cfg(test)]
mod stable_prefix_tests {
    use super::*;

    /// With the real Qwen3 tokenizer: the tokens of the stable text are exactly the head of
    /// the tokens of the full rendered prompt, for any draft of the last user message.
    /// Skipped when the checkpoint directory is not on this machine.
    #[test]
    fn real_tokenizer_stable_prefix_is_a_head_of_the_full_prompt() {
        let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("models/qwen3-30b-a3b-instruct-2507-fp8");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("skipped: no tokenizer at {}", dir.display());
            return;
        }
        let c = HfCodec::from_model_dir(&dir).expect("tokenizer");
        let sys = "You are Odysseus. Use the tools carefully.\n".repeat(20);
        for draft in ["", "What", "What is the capital of Fr", "What is the capital of France?", "list: \n- a\n- b ", "日本語のテスト"] {
            let turns = [("system", sys.as_str()), ("user", draft)];
            let text = c.render_chat(&turns).expect("render");
            let n = c.stable_len(&text).expect("ChatML layout recognised");
            assert!(!text[..n].contains("<|im_start|>assistant"), "generation header leaked");
            let stable = c.encode(&text[..n]).expect("encode stable");
            let full = c.encode(&text).expect("encode full");
            assert_eq!(&full[..stable.len()], &stable[..], "draft {draft:?}: stable tokens are not a head of the full prompt");
            assert!(full.len() - stable.len() <= 6, "tail after the draft should be a handful of tokens, got {}", full.len() - stable.len());
        }
    }
}
