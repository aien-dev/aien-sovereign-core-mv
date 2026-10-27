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

//! Word-level reading of one oracle answer: may the current word grow by some bytes, and is it a
//! whole item. Section 6.2's writer rule: text may be appended to the current word exactly when
//! U followed by it is still the start of some `word`, `newname` or `number` item.

use std::collections::HashSet;

use crate::oracle::Item;

/// The longest new name (OMEGESE.md section 6.2).
pub const NAME_MAX: usize = 63;

/// Space, tab, newline and the brackets end a word (section 6.2).
pub fn is_sep(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'(' | b')')
}

/// The word still being written: the bytes after the last separator.
pub fn current_word(text: &[u8]) -> &[u8] {
    match text.iter().rposition(|&b| is_sep(b)) {
        Some(i) => &text[i + 1..],
        None => text,
    }
}

/// One oracle answer, read against the current word U.
#[derive(Debug, Clone)]
pub struct ItemSet {
    u: Vec<u8>,
    /// Every non-empty prefix of `X[U.len()..]` over the `word X` items.
    word_prefixes: HashSet<Vec<u8>>,
    /// `X[U.len()..]` over the `word X` items.
    word_rests: HashSet<Vec<u8>>,
    /// `Some(exclusions)` when a `newname` item was printed.
    newname: Option<Vec<Vec<u8>>>,
    numbers: Vec<(i128, i128)>,
    open: bool,
    close: bool,
    space: bool,
    newline: bool,
    eof: bool,
}

impl ItemSet {
    /// `Err` when a `word` item does not start with U (the grammar promises it does).
    pub fn new(items: &[Item], u: &[u8]) -> Result<Self, String> {
        let mut s = Self {
            u: u.to_vec(),
            word_prefixes: HashSet::new(),
            word_rests: HashSet::new(),
            newname: None,
            numbers: Vec::new(),
            open: false,
            close: false,
            space: false,
            newline: false,
            eof: false,
        };
        for it in items {
            match it {
                Item::Word(x) => {
                    let rest = x.strip_prefix(u).ok_or_else(|| {
                        format!(
                            "oracle item `word {}` does not start with the current word {:?}",
                            String::from_utf8_lossy(x),
                            String::from_utf8_lossy(u)
                        )
                    })?;
                    for k in 1..=rest.len() {
                        s.word_prefixes.insert(rest[..k].to_vec());
                    }
                    s.word_rests.insert(rest.to_vec());
                }
                Item::NewName { exclude } => {
                    s.newname
                        .get_or_insert_with(Vec::new)
                        .extend(exclude.iter().cloned());
                }
                Item::Number { lo, hi } => s.numbers.push((*lo, *hi)),
                Item::Open => s.open = true,
                Item::Close => s.close = true,
                Item::Space => s.space = true,
                Item::Newline => s.newline = true,
                Item::Eof => s.eof = true,
            }
        }
        Ok(s)
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    /// The item that lets separator byte `b` come next.
    pub fn sep_offered(&self, b: u8) -> bool {
        match b {
            b' ' | b'\t' => self.space,
            b'\n' => self.newline,
            b'(' => self.open,
            b')' => self.close,
            _ => false,
        }
    }

    /// U followed by `a` (non-empty, no separator) is still the start of some word, new name or
    /// number item.
    pub fn extends(&self, a: &[u8]) -> bool {
        if self.word_prefixes.contains(a) {
            return true;
        }
        let mut buf = [0u8; 80];
        let Some(w) = joined(&mut buf, &self.u, a) else {
            return false;
        };
        if let Some(ex) = &self.newname {
            if name_prefix_ok(w, ex) {
                return true;
            }
        }
        self.numbers
            .iter()
            .any(|&(lo, hi)| number_prefix_ok(w, lo, hi))
    }

    /// U followed by `a` is a whole item, so a separator may end it (a cheap filter; the oracle
    /// decides what may follow). New names are not checked against the reserved words here.
    pub fn completes(&self, a: &[u8]) -> bool {
        if self.word_rests.contains(a) {
            return true;
        }
        let mut buf = [0u8; 80];
        let Some(w) = joined(&mut buf, &self.u, a) else {
            return false;
        };
        if let Some(ex) = &self.newname {
            if is_name(w) && w.len() <= NAME_MAX && !ex.iter().any(|e| e == w) {
                return true;
            }
        }
        number_value(w).is_some_and(|v| self.numbers.iter().any(|&(lo, hi)| lo <= v && v <= hi))
    }
}

fn joined<'a>(buf: &'a mut [u8; 80], u: &[u8], a: &[u8]) -> Option<&'a [u8]> {
    let n = u.len() + a.len();
    if n > buf.len() {
        return None;
    }
    buf[..u.len()].copy_from_slice(u);
    buf[u.len()..n].copy_from_slice(a);
    Some(&buf[..n])
}

/// A letter or `_`, then letters, digits or `_`.
fn is_name(w: &[u8]) -> bool {
    match w.split_first() {
        Some((&c, rest)) => {
            (c.is_ascii_alphabetic() || c == b'_')
                && rest.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_')
        }
        None => false,
    }
}

/// Some legal new name starts with `w`.
fn name_prefix_ok(w: &[u8], exclude: &[Vec<u8>]) -> bool {
    is_name(w) && (w.len() < NAME_MAX || (w.len() == NAME_MAX && !exclude.iter().any(|e| e == w)))
}

/// The value of a whole number word: decimal digits, an optional minus sign, no leading zeros,
/// not `-0`.
pub fn number_value(w: &[u8]) -> Option<i128> {
    let (neg, d) = match w.split_first() {
        Some((b'-', d)) => (true, d),
        _ => (false, w),
    };
    if d.is_empty() || d.len() > 20 || !d.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if d[0] == b'0' && (d.len() > 1 || neg) {
        return None;
    }
    let m: i128 = std::str::from_utf8(d).ok()?.parse().ok()?;
    Some(if neg { -m } else { m })
}

/// Some number `v` with `lo <= v <= hi` has a decimal text (as [`number_value`] reads it) that
/// starts with `p`.
pub fn number_prefix_ok(p: &[u8], lo: i128, hi: i128) -> bool {
    if p.is_empty() {
        return lo <= hi;
    }
    let (neg, d) = match p.split_first() {
        Some((b'-', d)) => (true, d),
        _ => (false, p),
    };
    if !d.iter().all(u8::is_ascii_digit) {
        return false;
    }
    // The magnitudes of the values of the sign `p` has.
    let (mlo, mhi) = if neg {
        if lo > -1 {
            return false;
        }
        ((-hi).max(1), -lo)
    } else {
        if hi < 0 {
            return false;
        }
        (lo.max(0), hi)
    };
    if mlo > mhi {
        return false;
    }
    if d.is_empty() {
        return true;
    }
    if d[0] == b'0' {
        // Only "0" itself, and never "-0".
        return d.len() == 1 && !neg && mlo == 0;
    }
    if d.len() > 20 {
        return false;
    }
    let Some(dv) = std::str::from_utf8(d)
        .ok()
        .and_then(|s| s.parse::<i128>().ok())
    else {
        return false;
    };
    // Values whose text has `len` digits and starts with `d`: [dv * 10^k, (dv + 1) * 10^k - 1].
    let mut scale: i128 = 1;
    for _ in d.len()..=20 {
        let a = dv * scale;
        let b = (dv + 1) * scale - 1;
        if a > mhi {
            return false;
        }
        if b >= mlo {
            return true;
        }
        scale *= 10;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force over every value in a small range.
    fn brute(p: &str, lo: i128, hi: i128) -> bool {
        (lo..=hi).any(|v| v.to_string().starts_with(p))
    }

    #[test]
    fn number_prefixes_match_brute_force() {
        let ranges = [
            (0, 0),
            (0, 255),
            (-128, 127),
            (-5, -1),
            (1, 14),
            (7, 1000),
            (-1000, 3),
        ];
        let mut prefixes = vec!["-".to_string(), "".to_string()];
        for n in 0..1200 {
            prefixes.push(n.to_string());
            prefixes.push(format!("-{n}"));
        }
        prefixes.extend(["00", "01", "-0", "-00", "0-", "1a"].map(String::from));
        for &(lo, hi) in &ranges {
            for p in &prefixes {
                assert_eq!(
                    number_prefix_ok(p.as_bytes(), lo, hi),
                    brute(p, lo, hi),
                    "{p:?} in {lo}..={hi}"
                );
            }
        }
    }

    #[test]
    fn number_prefixes_at_the_64_bit_edges() {
        let u = u64::MAX as i128;
        assert!(number_prefix_ok(b"18446744073709551615", 0, u));
        assert!(!number_prefix_ok(b"18446744073709551616", 0, u));
        assert!(number_prefix_ok(b"1844674407370955161", 0, u));
        assert!(!number_prefix_ok(b"28446744073709551615", 0, u));
        assert!(number_prefix_ok(b"9", 0, u));
        let (lo, hi) = (i64::MIN as i128, i64::MAX as i128);
        assert!(number_prefix_ok(b"-9223372036854775808", lo, hi));
        assert!(!number_prefix_ok(b"-9223372036854775809", lo, hi));
        assert!(!number_prefix_ok(b"123456789012345678901", 0, u));
    }

    #[test]
    fn number_values_are_canonical() {
        assert_eq!(number_value(b"0"), Some(0));
        assert_eq!(number_value(b"-7"), Some(-7));
        for bad in ["", "-", "-0", "007", "1x", "--1"] {
            assert_eq!(number_value(bad.as_bytes()), None, "{bad}");
        }
    }

    #[test]
    fn words_names_and_separators() {
        let items = vec![
            Item::Word(b"add".to_vec()),
            Item::Word(b"a".to_vec()),
            Item::NewName {
                exclude: vec![b"a".to_vec()],
            },
            Item::Space,
        ];
        let s = ItemSet::new(&items, b"a").unwrap();
        assert!(s.extends(b"d") && s.extends(b"dd") && s.extends(b"x9"));
        assert!(!s.extends(b"-") && !s.extends(b"="));
        assert!(s.completes(b"") && s.completes(b"dd") && s.completes(b"b"));
        assert!(s.sep_offered(b' ') && s.sep_offered(b'\t') && !s.sep_offered(b'\n'));
        assert!(
            ItemSet::new(&items, b"x").is_err(),
            "word a does not start with x"
        );
        let only = ItemSet::new(&[Item::Word(b"end".to_vec())], b"").unwrap();
        assert!(only.extends(b"en") && !only.extends(b"enx") && !only.completes(b"en"));
        assert_eq!(current_word(b"fn f a(uint"), b"uint");
        assert_eq!(current_word(b"ret a\n"), b"");
    }

    #[test]
    fn a_new_name_stops_at_63_characters() {
        let ex = vec![b"x".repeat(63)];
        assert!(name_prefix_ok(&b"y".repeat(63), &ex));
        assert!(!name_prefix_ok(&b"x".repeat(63), &ex));
        assert!(!name_prefix_ok(&b"y".repeat(64), &ex));
        assert!(!name_prefix_ok(b"9a", &ex));
    }
}
