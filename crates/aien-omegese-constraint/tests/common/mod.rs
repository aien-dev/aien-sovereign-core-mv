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

//! Shared by the host tests: the 64-token synthetic table, an exact one-program oracle, and the
//! real binary's path from the environment.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

use aien_omegese_constraint::constraint::canonical_seps;
use aien_omegese_constraint::word::{current_word, is_sep};
use aien_omegese_constraint::{Answer, Item, ItemOracle, Signature, TokenTable};

/// The synthetic table's stop token (a special token, no text).
pub const STOP: u32 = 63;

/// 64 tokens for the 64-token `EchoChainModel`: every single byte Omegese uses (so a mask is
/// never empty), the fence, and multi-item tokens with one boundary (` add`, `end\n`, `a b`,
/// `)\n`, and ` \n`, a blank run that reads as its newline) and with two (` a `, `(b)`).
pub fn synthetic_texts() -> Vec<Option<Vec<u8>>> {
    let mut t: Vec<Option<Vec<u8>>> = Vec::new();
    for c in b'a'..=b'z' {
        t.push(Some(vec![c]));
    }
    for c in b'0'..=b'9' {
        t.push(Some(vec![c]));
    }
    for s in [
        "_", "-", "=", " ", "\n", "(", ")", "`", "```", "end", "ret", "uint", " uint", " add",
        " a", " b", "\n\n", ")\n", " (", "end\n", "a b", " a ", "(b)", "if", " \n", "\n```", "fn",
    ] {
        t.push(Some(s.as_bytes().to_vec()));
    }
    t.push(None);
    assert_eq!(t.len(), 64);
    assert_eq!(t[STOP as usize], None);
    t
}

pub fn synthetic_table() -> TokenTable {
    TokenTable::from_texts(synthetic_texts())
}

pub fn sig_f() -> Signature {
    Signature::from_requirement("entry f\nparams u32 u32\nret u32\n").unwrap()
}

/// An exact oracle for the one-program language {`target`} (prefix-closed; `target` has single
/// blanks), blank runs read as Omegese reads them: what a writer may add is exactly the next
/// piece of `target`, or more blank space after blank space.
pub struct ExactText {
    pub target: Vec<u8>,
    pub calls: usize,
}

impl ExactText {
    pub fn new(target: &str) -> Self {
        Self {
            target: target.as_bytes().to_vec(),
            calls: 0,
        }
    }
}

impl ItemOracle for ExactText {
    fn next(&mut self, raw: &[u8]) -> Result<Answer, String> {
        self.calls += 1;
        // Blank runs read in canonical form, as Omegese reads them (the mask relies on it).
        let p = &normalize(raw);
        let t = &self.target;
        if !t.starts_with(p) {
            return Ok(Answer::Refused(
                "REFUSED omegese: line 1: not the target".into(),
            ));
        }
        let u = current_word(p);
        let mut items = Vec::new();
        let word_end = t[p.len()..]
            .iter()
            .position(|&b| is_sep(b))
            .map_or(t.len(), |k| p.len() + k);
        let start = p.len() - u.len();
        if word_end > start {
            items.push(Item::Word(t[start..word_end].to_vec()));
        }
        if p.len() == t.len() {
            items.push(Item::Eof);
        } else if word_end == p.len() {
            items.push(match t[p.len()] {
                b' ' | b'\t' => Item::Space,
                b'\n' => Item::Newline,
                b'(' => Item::Open,
                _ => Item::Close,
            });
        }
        // More blank space after blank space reads the same (the target has no two blanks in
        // a row, so these are new items).
        match p.last() {
            Some(b' ') => items.push(Item::Space),
            Some(b'\n') => items.extend([Item::Space, Item::Newline]),
            _ => {}
        }
        Ok(Answer::Items(items))
    }
}

/// Every separator run in `canonical_seps` form.
pub fn normalize(p: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < p.len() {
        if is_sep(p[i]) {
            let j = p[i..]
                .iter()
                .position(|&b| !is_sep(b))
                .map_or(p.len(), |k| i + k);
            out.extend(canonical_seps(&p[i..j]));
            i = j;
        } else {
            out.push(p[i]);
            i += 1;
        }
    }
    out
}

/// `OMEGESE_BIN`: the `omegese` binary built from omega main (`make omegese`). The tests that need
/// it are `#[ignore]` and fail, never skip, when it is missing.
pub fn omegese_bin() -> PathBuf {
    let p = std::env::var_os("OMEGESE_BIN")
        .map(PathBuf::from)
        .expect("set OMEGESE_BIN to the omegese binary built from omega main");
    assert!(p.is_file(), "OMEGESE_BIN {} is not a file", p.display());
    p
}

fn scratch(name: &str, body: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("omegese-constraint-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// `omegese emit` on `program`: `Ok(())` accepted, `Err(its line)` refused.
pub fn emit(bin: &PathBuf, program: &[u8]) -> Result<(), String> {
    let src = scratch("emit.omegese", program);
    let out = src.with_extension("json");
    let o = Command::new(bin)
        .arg("emit")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    match o.status.code() {
        Some(0) => Ok(()),
        _ => Err(String::from_utf8_lossy(&o.stdout).trim().to_string()),
    }
}

/// `omegese complete` on `prefix`: the text that finishes it.
pub fn complete(bin: &PathBuf, prefix: &[u8]) -> Result<Vec<u8>, String> {
    let src = scratch("complete.omegese", prefix);
    let o = Command::new(bin)
        .arg("complete")
        .arg(&src)
        .output()
        .unwrap();
    match o.status.code() {
        Some(0) => Ok(o.stdout),
        _ => Err(String::from_utf8_lossy(&o.stdout).trim().to_string()),
    }
}
