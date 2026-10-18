//! Our own tokenizer for BitNet b1.58 2B-4T (stage T2): the LLaMA-3 byte-level BPE read from the
//! checkpoint's `tokenizer.json`, with no tokenizer library in the path.
//!
//! What it implements, and where each rule comes from (spec section 4):
//! - Special tokens are cut out of the text first (the 256 `added_tokens`, all `special`,
//!   `normalized: false`), longest match at the leftmost position.
//! - The pre-tokenizer splits with the LLaMA-3 pattern (`tokenizer.json` pre_tokenizer, and
//!   `REF/tokenizer.py:43`), here a hand-written scanner that follows the regex's leftmost,
//!   first-alternative, greedy-with-backtracking semantics exactly:
//!   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
//!   `\p{L}`, `\p{N}`, `\s` and the case folds come from the Unicode tables of `regex-syntax`
//!   (data only; no regex engine runs).
//! - Each piece's UTF-8 bytes map to the GPT-2 byte alphabet; `ignore_merges: true` means a
//!   piece that is itself a vocabulary entry is one token; otherwise BPE merges the adjacent pair
//!   of lowest merge rank, leftmost first, until none applies.
//! - BOS 128000 is prepended when asked (the `TemplateProcessing` post-processor).

use std::collections::HashMap;
use std::path::Path;

use regex_syntax::hir::{Class, HirKind};

/// The split pattern this scanner implements; any other pattern is refused.
pub const LLAMA3_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The contractions of the first alternative, in pattern order.
const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];

/// A sorted, disjoint set of inclusive char ranges.
#[derive(Debug, Clone)]
struct CharSet(Vec<(char, char)>);

impl CharSet {
    fn from_class(pattern: &str) -> Result<Self, String> {
        let hir = regex_syntax::parse(pattern).map_err(|e| format!("class {pattern}: {e}"))?;
        match hir.kind() {
            HirKind::Class(Class::Unicode(c)) => Ok(Self(
                c.ranges().iter().map(|r| (r.start(), r.end())).collect(),
            )),
            HirKind::Literal(_) => {
                let c = pattern
                    .chars()
                    .last()
                    .ok_or_else(|| format!("class {pattern}: empty"))?;
                Ok(Self(vec![(c, c)]))
            }
            other => Err(format!(
                "class {pattern}: not a character class ({other:?})"
            )),
        }
    }

    fn contains(&self, c: char) -> bool {
        self.0
            .binary_search_by(|&(lo, hi)| {
                if hi < c {
                    std::cmp::Ordering::Less
                } else if lo > c {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

/// The character classes of the split pattern.
#[derive(Debug, Clone)]
struct Classes {
    letter: CharSet,
    number: CharSet,
    space: CharSet,
    /// Case-insensitive sets for the contraction letters, keyed by the lowercase ASCII letter.
    folds: Vec<(char, CharSet)>,
}

impl Classes {
    fn new() -> Result<Self, String> {
        let mut folds = Vec::new();
        for c in "strevmld".chars() {
            folds.push((c, CharSet::from_class(&format!("(?i:{c})"))?));
        }
        Ok(Self {
            letter: CharSet::from_class(r"\p{L}")?,
            number: CharSet::from_class(r"\p{N}")?,
            space: CharSet::from_class(r"\s")?,
            folds,
        })
    }

    fn l(&self, c: char) -> bool {
        if c.is_ascii() {
            c.is_ascii_alphabetic()
        } else {
            self.letter.contains(c)
        }
    }

    fn n(&self, c: char) -> bool {
        if c.is_ascii() {
            c.is_ascii_digit()
        } else {
            self.number.contains(c)
        }
    }

    fn s(&self, c: char) -> bool {
        self.space.contains(c)
    }

    /// `[^\s\p{L}\p{N}]`
    fn other(&self, c: char) -> bool {
        !self.s(c) && !self.l(c) && !self.n(c)
    }

    fn folds_to(&self, c: char, lower: char) -> bool {
        self.folds
            .iter()
            .find(|(k, _)| *k == lower)
            .is_some_and(|(_, set)| set.contains(c))
    }
}

/// Splits `text` into pre-tokens with the LLaMA-3 pattern; returns char-index ranges.
fn split_pieces(cl: &Classes, chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let at = |i: usize| chars.get(i).copied();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let run = |from: usize, f: &dyn Fn(char) -> bool| {
            let mut e = from;
            while e < n && f(chars[e]) {
                e += 1;
            }
            e
        };
        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        let alt1 = (c == '\'')
            .then(|| {
                CONTRACTIONS.iter().find_map(|w| {
                    let ok = w
                        .chars()
                        .enumerate()
                        .all(|(j, lc)| at(i + 1 + j).is_some_and(|x| cl.folds_to(x, lc)));
                    ok.then_some(i + 1 + w.len())
                })
            })
            .flatten();
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        let alt2 = || {
            if cl.l(c) {
                Some(run(i, &|x| cl.l(x)))
            } else if c != '\r' && c != '\n' && !cl.n(c) && at(i + 1).is_some_and(|x| cl.l(x)) {
                Some(run(i + 1, &|x| cl.l(x)))
            } else {
                None
            }
        };
        // 3. \p{N}{1,3}
        let alt3 = || {
            cl.n(c).then(|| {
                let mut e = i;
                while e < n && e < i + 3 && cl.n(chars[e]) {
                    e += 1;
                }
                e
            })
        };
        // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        let alt4 = || {
            let start = if c == ' ' && at(i + 1).is_some_and(|x| cl.other(x)) {
                i + 1
            } else if cl.other(c) {
                i
            } else {
                return None;
            };
            let e = run(start, &|x| cl.other(x));
            Some(run(e, &|x| x == '\r' || x == '\n'))
        };
        // 5-7 start on whitespace; `e` is the end of the whitespace run.
        let ws = || {
            if !cl.s(c) {
                return None;
            }
            let e = run(i, &|x| cl.s(x));
            // 5. \s*[\r\n]+ : greedy \s* backs off to the last newline of the run, and the
            //    newline run from there ends at the next non-newline.
            if let Some(last) = (i..e).rev().find(|&j| chars[j] == '\r' || chars[j] == '\n') {
                return Some(last + 1);
            }
            // 6. \s+(?!\S) : the whole run at the end of text, else all but its last char.
            if e == n {
                return Some(e);
            }
            if e - i >= 2 {
                return Some(e - 1);
            }
            // 7. \s+
            Some(e)
        };
        let end = alt1
            .or_else(alt2)
            .or_else(alt3)
            .or_else(alt4)
            .or_else(ws)
            // Every char is a letter, number, whitespace or other, so one alternative matched.
            .unwrap_or(i + 1);
        out.push((i, end));
        i = end;
    }
    out
}

/// GPT-2's byte-to-char alphabet: printable bytes stand for themselves, the rest move to 256+.
fn byte_alphabet() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let cp = if printable {
            b
        } else {
            let v = next;
            next += 1;
            v
        };
        map[b as usize] = char::from_u32(cp).expect("alphabet char");
    }
    map
}

/// The BitNet 2B-4T tokenizer, built from `tokenizer.json`.
#[derive(Debug, Clone)]
pub struct BitNetTokenizer {
    classes: Classes,
    /// Base vocabulary keyed by raw bytes.
    vocab: HashMap<Vec<u8>, u32>,
    /// `(left, right) -> (rank, merged)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Every id's bytes (specials as their UTF-8 text), for decoding.
    id_bytes: Vec<Vec<u8>>,
    /// Special tokens by text, longest first.
    specials: Vec<(String, u32)>,
    byte_id: [u32; 256],
    pub bos: u32,
}

impl BitNetTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    /// Reads `tokenizer.json`; every setting this scanner does not implement is refused by name.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let j: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("tokenizer.json: {e}"))?;
        if !j["normalizer"].is_null() {
            return Err("tokenizer.json: a normalizer is set (none is implemented)".into());
        }
        let m = &j["model"];
        let want = |k: &str, v: serde_json::Value| -> Result<(), String> {
            if m[k] != v {
                return Err(format!("tokenizer.json: model.{k} is {}, need {v}", m[k]));
            }
            Ok(())
        };
        want("type", "BPE".into())?;
        want("ignore_merges", true.into())?;
        want("byte_fallback", false.into())?;
        want("dropout", serde_json::Value::Null)?;
        want("continuing_subword_prefix", serde_json::Value::Null)?;
        want("end_of_word_suffix", serde_json::Value::Null)?;
        let pre = &j["pre_tokenizer"];
        let parts = pre["pretokenizers"].as_array();
        let ok_pre = pre["type"] == "Sequence"
            && parts.is_some_and(|p| {
                p.len() == 2
                    && p[0]["type"] == "Split"
                    && p[0]["pattern"]["Regex"] == LLAMA3_SPLIT_PATTERN
                    && p[0]["behavior"] == "Isolated"
                    && p[0]["invert"] == false
                    && p[1]["type"] == "ByteLevel"
                    && p[1]["add_prefix_space"] == false
                    && p[1]["use_regex"] == false
            });
        if !ok_pre {
            return Err(
                "tokenizer.json: pre_tokenizer is not the LLaMA-3 split then ByteLevel".into(),
            );
        }

        let alphabet = byte_alphabet();
        let mut unmap = HashMap::new();
        for (b, &c) in alphabet.iter().enumerate() {
            unmap.insert(c, b as u8);
        }
        let to_bytes = |s: &str| -> Result<Vec<u8>, String> {
            s.chars()
                .map(|c| {
                    unmap.get(&c).copied().ok_or_else(|| {
                        format!("vocab entry {s:?}: char {c:?} not in the byte alphabet")
                    })
                })
                .collect()
        };

        let vocab_json = m["vocab"]
            .as_object()
            .ok_or("tokenizer.json: model.vocab is not an object")?;
        let mut vocab = HashMap::with_capacity(vocab_json.len());
        let mut id_bytes: Vec<Option<Vec<u8>>> = Vec::new();
        for (s, id) in vocab_json {
            let id = id
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| format!("vocab entry {s:?}: id {id} is not a token id"))?;
            let bytes = to_bytes(s)?;
            if vocab.insert(bytes.clone(), id).is_some() {
                return Err(format!("vocab entry {s:?}: duplicate bytes"));
            }
            if id_bytes.len() <= id as usize {
                id_bytes.resize(id as usize + 1, None);
            }
            if id_bytes[id as usize].replace(bytes).is_some() {
                return Err(format!("vocab id {id} used twice"));
            }
        }
        let mut byte_id = [0u32; 256];
        for b in 0..=255u8 {
            byte_id[b as usize] = *vocab
                .get(&vec![b])
                .ok_or_else(|| format!("byte {b} has no vocabulary entry"))?;
        }

        let mut specials = Vec::new();
        for t in j["added_tokens"]
            .as_array()
            .ok_or("tokenizer.json: added_tokens is not a list")?
        {
            let id = t["id"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| format!("added token {t}: no id"))?;
            let content = t["content"]
                .as_str()
                .ok_or_else(|| format!("added token {id}: no content"))?;
            if t["special"] != true
                || t["normalized"] != false
                || t["lstrip"] != false
                || t["rstrip"] != false
                || t["single_word"] != false
                || !content.starts_with("<|")
            {
                return Err(format!(
                    "added token {id} {content:?}: only plain special tokens starting with <| are implemented"
                ));
            }
            if id_bytes.len() <= id as usize {
                id_bytes.resize(id as usize + 1, None);
            }
            if id_bytes[id as usize]
                .replace(content.as_bytes().to_vec())
                .is_some()
            {
                return Err(format!("added token {id} {content:?} reuses an id"));
            }
            specials.push((content.to_string(), id));
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let id_bytes = id_bytes
            .into_iter()
            .enumerate()
            .map(|(i, b)| b.ok_or_else(|| format!("token id {i} is missing")))
            .collect::<Result<Vec<_>, _>>()?;

        let merges_json = m["merges"]
            .as_array()
            .ok_or("tokenizer.json: model.merges is not a list")?;
        let mut merges = HashMap::with_capacity(merges_json.len());
        for (rank, mj) in merges_json.iter().enumerate() {
            let (a, b) = match mj {
                serde_json::Value::String(s) => s
                    .split_once(' ')
                    .ok_or_else(|| format!("merge {rank} {s:?}: no space"))?,
                serde_json::Value::Array(p) if p.len() == 2 => (
                    p[0].as_str()
                        .ok_or_else(|| format!("merge {rank}: not text"))?,
                    p[1].as_str()
                        .ok_or_else(|| format!("merge {rank}: not text"))?,
                ),
                other => return Err(format!("merge {rank}: {other} is not a pair")),
            };
            let (ba, bb) = (to_bytes(a)?, to_bytes(b)?);
            let id = |x: &Vec<u8>| {
                vocab.get(x).copied().ok_or_else(|| {
                    format!("merge {rank} {a:?} {b:?}: a side is not in the vocabulary")
                })
            };
            let joined = [ba.as_slice(), bb.as_slice()].concat();
            let key = (id(&ba)?, id(&bb)?);
            let merged = id(&joined)?;
            merges.entry(key).or_insert((rank as u32, merged));
        }

        let bos = j["post_processor"]["processors"]
            .as_array()
            .and_then(|ps| {
                ps.iter()
                    .find(|p| p["type"] == "TemplateProcessing")
                    .and_then(|p| p["special_tokens"]["<|begin_of_text|>"]["ids"][0].as_u64())
            })
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("tokenizer.json: no <|begin_of_text|> in the template post-processor")?;

        Ok(Self {
            classes: Classes::new()?,
            vocab,
            merges,
            id_bytes,
            specials,
            byte_id,
            bos,
        })
    }

    /// Vocabulary size including the special tokens.
    pub fn vocab_len(&self) -> usize {
        self.id_bytes.len()
    }

    /// Encodes `text`, BOS first when `bos`.
    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if bos {
            out.push(self.bos);
        }
        let mut rest = text;
        while !rest.is_empty() {
            // Leftmost special token, longest at that position.
            let found = rest.match_indices("<|").find_map(|(at, _)| {
                self.specials
                    .iter()
                    .find(|(s, _)| rest[at..].starts_with(s.as_str()))
                    .map(|(s, id)| (at, s.len(), *id))
            });
            match found {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    rest = "";
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = text.chars().collect();
        let mut buf = [0u8; 4];
        for (a, b) in split_pieces(&self.classes, &chars) {
            let mut bytes = Vec::new();
            for &c in &chars[a..b] {
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
            self.bpe(&bytes, out);
        }
    }

    /// One pre-token: whole if it is a vocabulary entry (`ignore_merges`), else ranked merges.
    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if let Some(&id) = self.vocab.get(bytes) {
            out.push(id);
            return;
        }
        let mut sym: Vec<u32> = bytes.iter().map(|&b| self.byte_id[b as usize]).collect();
        loop {
            let best = sym
                .windows(2)
                .enumerate()
                .filter_map(|(p, w)| self.merges.get(&(w[0], w[1])).map(|&(r, m)| (r, p, m)))
                .min_by_key(|&(r, p, _)| (r, p));
            match best {
                Some((_, p, m)) => {
                    sym[p] = m;
                    sym.remove(p + 1);
                }
                None => break,
            }
        }
        out.extend_from_slice(&sym);
    }

    /// The text of `ids` (invalid UTF-8 replaced); an unknown id is refused.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(
                self.id_bytes
                    .get(id as usize)
                    .ok_or_else(|| format!("token id {id} outside the vocabulary"))?,
            );
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<String> {
        let cl = Classes::new().unwrap();
        let chars: Vec<char> = text.chars().collect();
        split_pieces(&cl, &chars)
            .into_iter()
            .map(|(a, b)| chars[a..b].iter().collect())
            .collect()
    }

    #[test]
    fn split_follows_the_llama3_pattern() {
        assert_eq!(
            pieces("The capital of France is"),
            ["The", " capital", " of", " France", " is"]
        );
        assert_eq!(
            pieces("Count from 6 to 10:"),
            ["Count", " from", " ", "6", " to", " ", "10", ":"]
        );
        assert_eq!(pieces("12345"), ["123", "45"]);
        assert_eq!(pieces("I'm, they'LL"), ["I", "'m", ",", " they", "'LL"]);
        assert_eq!(pieces("x'ſ"), ["x", "'ſ"]);
        assert_eq!(pieces("'xy"), ["'xy"]);
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("a \n\n b"), ["a", " \n\n", " b"]);
        assert_eq!(pieces("a.\n\nB"), ["a", ".\n\n", "B"]);
        assert_eq!(pieces("x   "), ["x", "   "]);
        assert_eq!(pieces("x \t!"), ["x", " ", "\t", "!"]);
        assert_eq!(pieces("x  !!"), ["x", " ", " !!"]);
        assert_eq!(pieces("\tword"), ["\tword"]);
        assert_eq!(pieces("über 3½"), ["über", " ", "3½"]);
        assert_eq!(pieces("a\r\n \r\nb"), ["a", "\r\n \r\n", "b"]);
        assert_eq!(
            pieces("it'ſ \u{3000}x;  über-naïve 12345 ٣٤٥ 🙂🙂\u{200d} end  "),
            [
                "it",
                "'ſ",
                " ",
                "\u{3000}x",
                ";",
                " ",
                " über",
                "-naïve",
                " ",
                "123",
                "45",
                " ",
                "٣٤٥",
                " 🙂🙂\u{200d}",
                " end",
                "  "
            ]
        );
    }

    #[test]
    fn byte_alphabet_is_gpt2s() {
        let a = byte_alphabet();
        assert_eq!(a[b' ' as usize], 'Ġ');
        assert_eq!(a[b'\n' as usize], 'Ċ');
        assert_eq!(a[b'A' as usize], 'A');
        assert_eq!(a[0xAD], char::from_u32(256 + 33 + 34).unwrap());
        let mut seen: Vec<char> = a.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 256);
    }

    /// A toy vocabulary: bytes plus "ab", "abc", "Ġab"; merges a b, ab c, Ġ ab.
    fn toy() -> BitNetTokenizer {
        let a = byte_alphabet();
        let mut vocab = serde_json::Map::new();
        for (b, c) in a.iter().enumerate() {
            vocab.insert(c.to_string(), (b as u64).into());
        }
        vocab.insert("ab".into(), 256.into());
        vocab.insert("abc".into(), 257.into());
        vocab.insert("Ġab".into(), 258.into());
        vocab.insert("bc".into(), 259.into());
        let j = serde_json::json!({
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": LLAMA3_SPLIT_PATTERN}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
            "post_processor": {"type": "Sequence", "processors": [{"type": "TemplateProcessing",
                "special_tokens": {"<|begin_of_text|>": {"ids": [260]}}}]},
            "added_tokens": [
                {"id": 260, "content": "<|begin_of_text|>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
                {"id": 261, "content": "<|eot_id|>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false}],
            "model": {"type": "BPE", "ignore_merges": true, "byte_fallback": false, "dropout": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "vocab": vocab, "merges": ["b c", "a b", "ab c", "Ġ ab"]}
        });
        BitNetTokenizer::from_json(&j.to_string()).unwrap()
    }

    #[test]
    fn bpe_merges_by_rank_and_specials_split_out() {
        let t = toy();
        // "abc" is in the vocabulary: one token even though "b c" outranks "a b".
        assert_eq!(t.encode("abc", false), [257]);
        // "abd": "a b" merges; "abcc" would merge "b c" first (rank 0) then nothing joins "a".
        assert_eq!(t.encode("abd", false), [256, b'd' as u32]);
        assert_eq!(t.encode("abcc", false), [b'a' as u32, 259, b'c' as u32]);
        assert_eq!(t.encode("x ab", true), [260, b'x' as u32, 258]);
        assert_eq!(
            t.encode("a<|eot_id|>b", false),
            [b'a' as u32, 261, b'b' as u32]
        );
        assert_eq!(
            t.decode(&[260, 258, 261]).unwrap(),
            "<|begin_of_text|> ab<|eot_id|>"
        );
        assert!(t.decode(&[999]).is_err());
    }

    #[test]
    fn unimplemented_settings_are_refused_by_name() {
        let mut j: serde_json::Value = serde_json::json!({"normalizer": {"type": "NFC"}});
        assert!(BitNetTokenizer::from_json(&j.to_string())
            .unwrap_err()
            .contains("normalizer"));
        j = serde_json::json!({"normalizer": null, "model": {"type": "BPE", "ignore_merges": false}});
        assert!(BitNetTokenizer::from_json(&j.to_string())
            .unwrap_err()
            .contains("ignore_merges"));
    }
}
