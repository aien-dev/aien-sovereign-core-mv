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

//! Turns the engine's token stream into text deltas: decodes incrementally
//! (holding back a token run that ends inside a multi-byte character) and
//! applies stop strings (holding back any text that could still grow into one).

use std::sync::Arc;

use crate::codec::Codec;

/// How many tokens may be held back waiting for a character to complete before
/// the run is flushed as-is (guards against a model emitting invalid bytes forever).
const MAX_PARTIAL_TOKENS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Push {
    /// Nothing to emit yet.
    Hold,
    /// Emit this text and keep going.
    Delta(String),
    /// Emit this text (possibly empty) and stop: a stop string matched.
    StopAt(String),
}

pub struct TextAssembler {
    codec: Arc<dyn Codec>,
    stop_strings: Vec<String>,
    tokens: Vec<u32>,
    /// Tokens before this index have been decoded into `pending` or emitted.
    decoded_upto: usize,
    /// Decoded text not yet emitted (may be a prefix of a stop string).
    pending: String,
    stopped: bool,
}

impl TextAssembler {
    pub fn new(codec: Arc<dyn Codec>, stop_strings: Vec<String>) -> Self {
        Self {
            codec,
            stop_strings,
            tokens: Vec::new(),
            decoded_upto: 0,
            pending: String::new(),
            stopped: false,
        }
    }

    pub fn push(&mut self, token: u32) -> Push {
        if self.stopped {
            return Push::Hold;
        }
        self.tokens.push(token);
        let window = &self.tokens[self.decoded_upto..];
        let text = match self.codec.decode(window) {
            Ok(t) => t,
            Err(_) => return Push::Hold,
        };
        let partial = text.ends_with('\u{FFFD}');
        if partial && window.len() < MAX_PARTIAL_TOKENS {
            return Push::Hold;
        }
        self.decoded_upto = self.tokens.len();
        self.pending.push_str(&text);
        self.drain()
    }

    /// Emits whatever is still held back. Call once when the engine is done.
    pub fn finish(&mut self) -> Option<String> {
        if self.stopped {
            return None;
        }
        // A run that never completed its character still has to be shown.
        if self.decoded_upto < self.tokens.len() {
            if let Ok(t) = self.codec.decode(&self.tokens[self.decoded_upto..]) {
                self.pending.push_str(&t);
            }
            self.decoded_upto = self.tokens.len();
        }
        if self.pending.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending))
        }
    }

    fn drain(&mut self) -> Push {
        if let Some(idx) = self
            .stop_strings
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
        {
            self.stopped = true;
            let out = self.pending[..idx].to_string();
            self.pending.clear();
            return Push::StopAt(out);
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        let keep_from = self
            .pending
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &self.pending[i..];
                self.stop_strings
                    .iter()
                    .any(|s| s.len() > tail.len() && s.starts_with(tail))
            })
            .unwrap_or(self.pending.len());
        if keep_from == 0 {
            return Push::Hold;
        }
        let out = self.pending[..keep_from].to_string();
        self.pending = self.pending[keep_from..].to_string();
        Push::Delta(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::ByteCodec;

    fn asm(stops: &[&str]) -> TextAssembler {
        TextAssembler::new(
            Arc::new(ByteCodec),
            stops.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn run(mut a: TextAssembler, text: &str) -> (String, bool) {
        let mut out = String::new();
        let mut stopped = false;
        for b in text.bytes() {
            match a.push(u32::from(b)) {
                Push::Hold => {}
                Push::Delta(d) => out.push_str(&d),
                Push::StopAt(d) => {
                    out.push_str(&d);
                    stopped = true;
                    break;
                }
            }
        }
        if let Some(rest) = a.finish() {
            out.push_str(&rest);
        }
        (out, stopped)
    }

    #[test]
    fn multibyte_characters_are_never_split() {
        let mut a = asm(&[]);
        let mut deltas = Vec::new();
        for b in "héllo wörld".bytes() {
            if let Push::Delta(d) = a.push(u32::from(b)) {
                deltas.push(d);
            }
        }
        assert!(deltas.iter().all(|d| !d.contains('\u{FFFD}')), "{deltas:?}");
        assert_eq!(deltas.concat(), "héllo wörld");
    }

    #[test]
    fn stop_string_truncates_and_holds_back_prefix() {
        assert_eq!(
            run(asm(&["world"]), "hello world!"),
            ("hello ".into(), true)
        );
        assert_eq!(run(asm(&["<|end|>"]), "a<|en"), ("a<|en".into(), false));
        assert_eq!(run(asm(&["xx", "b"]), "aaxxb"), ("aa".into(), true));
    }

    #[test]
    fn no_stops_passes_everything() {
        assert_eq!(run(asm(&[]), "plain text"), ("plain text".into(), false));
    }
}
