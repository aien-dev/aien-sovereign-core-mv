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

//! The model vocabulary as byte strings, with each token's shape against Omegese's word
//! separators precomputed, and an encoder for the text the constraint forces.

use std::collections::HashMap;
use std::path::Path;

use crate::word::is_sep;

/// A token's text against the word separators (space, tab, newline, brackets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// No program text: a special or added token, or an empty one.
    Special,
    /// Word bytes only: the token grows the current word.
    Word,
    /// Word bytes `..a_end` (maybe none), one run of separators `a_end..sep_end`, then word
    /// bytes to the end (maybe none): one word boundary.
    OneBoundary { a_end: usize, sep_end: usize },
    /// A second separator run after word bytes: the token crosses two word boundaries.
    Multi,
}

fn shape_of(t: &[u8]) -> Shape {
    if t.is_empty() {
        return Shape::Special;
    }
    let Some(a_end) = t.iter().position(|&b| is_sep(b)) else {
        return Shape::Word;
    };
    let sep_end = a_end
        + t[a_end..]
            .iter()
            .position(|&b| !is_sep(b))
            .unwrap_or(t.len() - a_end);
    if t[sep_end..].iter().any(|&b| is_sep(b)) {
        Shape::Multi
    } else {
        Shape::OneBoundary { a_end, sep_end }
    }
}

enum Encoder {
    /// The model's own tokenizer: forced text is tokenized as the model would read it.
    Hf(Box<tokenizers::Tokenizer>),
    /// Longest match over the table (synthetic tables in tests).
    Greedy {
        by_text: HashMap<Vec<u8>, u32>,
        longest: usize,
    },
}

/// Every token id's bytes (`None` for special tokens) and the encoder for forced text.
pub struct TokenTable {
    texts: Vec<Option<Vec<u8>>>,
    shapes: Vec<Shape>,
    encoder: Encoder,
}

impl TokenTable {
    /// A synthetic table: `texts[id]`, `None` for a special token; forced text is encoded by
    /// longest match (the lowest id wins a duplicate).
    pub fn from_texts(texts: Vec<Option<Vec<u8>>>) -> Self {
        let mut by_text = HashMap::new();
        let mut longest = 0;
        for (id, t) in texts.iter().enumerate() {
            if let Some(t) = t.as_ref().filter(|t| !t.is_empty()) {
                by_text.entry(t.clone()).or_insert(id as u32);
                longest = longest.max(t.len());
            }
        }
        Self::build(texts, Encoder::Greedy { by_text, longest })
    }

    /// The vocabulary of a byte-level BPE `tokenizer.json` (Qwen's). Added tokens (the chat
    /// specials) have no program text. `Err` if any other token is not byte-level text.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, String> {
        let tk = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("load {}: {e}", path.display()))?;
        let added = tk.get_added_tokens_decoder();
        let inv = byte_decoder();
        let n = tk.get_vocab_size(true);
        let mut texts = Vec::with_capacity(n);
        for id in 0..n as u32 {
            if added.contains_key(&id) {
                texts.push(None);
                continue;
            }
            let Some(s) = tk.id_to_token(id) else {
                texts.push(None);
                continue;
            };
            let bytes = s
                .chars()
                .map(|c| inv.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| format!("token {id} ({s:?}) is not byte-level text"))?;
            texts.push(Some(bytes));
        }
        Ok(Self::build(texts, Encoder::Hf(Box::new(tk))))
    }

    fn build(texts: Vec<Option<Vec<u8>>>, encoder: Encoder) -> Self {
        let shapes = texts
            .iter()
            .map(|t| t.as_deref().map_or(Shape::Special, shape_of))
            .collect();
        Self {
            texts,
            shapes,
            encoder,
        }
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// The bytes of `id`, `None` for a special token or an id outside the table.
    pub fn text(&self, id: u32) -> Option<&[u8]> {
        self.texts.get(id as usize)?.as_deref()
    }

    pub fn shape(&self, id: u32) -> Shape {
        self.shapes
            .get(id as usize)
            .copied()
            .unwrap_or(Shape::Special)
    }

    /// A special token's id by its content (`<|im_end|>`); only for a `tokenizer.json` table.
    pub fn special_id(&self, content: &str) -> Option<u32> {
        match &self.encoder {
            Encoder::Hf(tk) => tk
                .token_to_id(content)
                .filter(|&id| self.text(id).is_none()),
            Encoder::Greedy { .. } => None,
        }
    }

    /// The bytes of a token sequence; `Err` on a special token.
    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(
                self.text(id)
                    .ok_or_else(|| format!("token {id} has no program text"))?,
            );
        }
        Ok(out)
    }

    /// Tokens for `text`, checked to decode back to exactly `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let ids = match &self.encoder {
            Encoder::Hf(tk) => tk
                .encode(text, false)
                .map_err(|e| format!("encode {text:?}: {e}"))?
                .get_ids()
                .to_vec(),
            Encoder::Greedy { by_text, longest } => {
                let b = text.as_bytes();
                let mut ids = Vec::new();
                let mut i = 0;
                while i < b.len() {
                    let id = (1..=(*longest).min(b.len() - i))
                        .rev()
                        .find_map(|n| by_text.get(&b[i..i + n]).map(|&id| (id, n)));
                    let (id, n) = id.ok_or_else(|| {
                        format!("no token in the table starts byte {i} of {text:?}")
                    })?;
                    ids.push(id);
                    i += n;
                }
                ids
            }
        };
        if self.decode(&ids)? != text.as_bytes() {
            return Err(format!(
                "{text:?} does not round-trip through the tokenizer"
            ));
        }
        Ok(ids)
    }
}

/// GPT-2's byte-level alphabet, inverted: each printable stand-in character to its byte.
fn byte_decoder() -> HashMap<char, u8> {
    let mut map = HashMap::new();
    let mut n = 0u32;
    for b in 0..=255u8 {
        let direct = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        let c = if direct {
            char::from(b)
        } else {
            n += 1;
            char::from_u32(255 + n).expect("a valid code point")
        };
        map.insert(c, b);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(shape_of(b"add"), Shape::Word);
        assert_eq!(shape_of(b"=-"), Shape::Word);
        assert_eq!(
            shape_of(b" add"),
            Shape::OneBoundary {
                a_end: 0,
                sep_end: 1
            }
        );
        assert_eq!(
            shape_of(b"end\n"),
            Shape::OneBoundary {
                a_end: 3,
                sep_end: 4
            }
        );
        assert_eq!(
            shape_of(b")\n"),
            Shape::OneBoundary {
                a_end: 0,
                sep_end: 2
            }
        );
        assert_eq!(
            shape_of(b"a b"),
            Shape::OneBoundary {
                a_end: 1,
                sep_end: 2
            }
        );
        assert_eq!(shape_of(b" a "), Shape::Multi);
        assert_eq!(shape_of(b"(x)"), Shape::Multi);
        assert_eq!(shape_of(b""), Shape::Special);
    }

    #[test]
    fn the_byte_alphabet_is_a_bijection() {
        let d = byte_decoder();
        assert_eq!(d.len(), 256);
        assert_eq!(d[&'Ġ'], b' ');
        assert_eq!(d[&'Ċ'], b'\n');
        assert_eq!(d[&'a'], b'a');
    }

    #[test]
    fn greedy_encoding_round_trips() {
        let t = TokenTable::from_texts(vec![
            Some(b"a".to_vec()),
            Some(b"ab".to_vec()),
            Some(b"b".to_vec()),
            None,
        ]);
        assert_eq!(t.encode("abab").unwrap(), vec![1, 1]);
        assert_eq!(t.encode("ba").unwrap(), vec![2, 0]);
        assert!(t.encode("c").is_err());
        assert!(t.decode(&[3]).is_err());
        assert_eq!(t.shape(3), Shape::Special);
        assert_eq!(t.shape(99), Shape::Special);
    }
}
