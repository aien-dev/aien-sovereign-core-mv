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

//! aien-verify: the independent verifier of WHOLE-SYSTEM-E2E v1 (lane L4).
//! Reads a run folder (chain receipts, SHA256SUMS, output files) and writes the verdict.
//! Standard library only; own sha256; shares no code with the executor or the harness.
//! Rules: docs/campaigns/whole-system-e2e/VERIFIER-v1.md.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// ---------------------------------------------------------------- sha256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }
    fn block(h: &mut [u32; 8], b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b_, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if !self.buf.is_empty() {
            let need = 64 - self.buf.len();
            let take = need.min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                Self::block(&mut self.h, &b);
            }
        }
        while data.len() >= 64 {
            Self::block(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        self.buf.extend_from_slice(data);
    }
    fn finish(mut self) -> String {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_be_bytes());
        let saved = self.len;
        self.update(&pad);
        self.len = saved;
        self.h.iter().map(|x| format!("{:08x}", x)).collect()
    }
}

fn sha256_bytes(d: &[u8]) -> String {
    let mut s = Sha256::new();
    s.update(d);
    s.finish()
}

fn sha256_file(p: &Path) -> Result<String, String> {
    fs::read(p).map(|d| sha256_bytes(&d)).map_err(|e| format!("{}: {}", p.display(), e))
}

fn selftest() -> Result<(), String> {
    let vecs: [(&[u8], &str); 4] = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        (
            b"The quick brown fox jumps over the lazy dog",
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
        ),
    ];
    for (i, (m, want)) in vecs.iter().enumerate() {
        let got = sha256_bytes(m);
        if got != *want {
            return Err(format!("sha256 vector {} wrong: {}", i, got));
        }
    }
    let million = vec![b'a'; 1_000_000];
    let got = sha256_bytes(&million);
    if got != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" {
        return Err(format!("sha256 million-a wrong: {}", got));
    }
    // chunked update must equal one-shot
    let mut s = Sha256::new();
    for c in million.chunks(37) {
        s.update(c);
    }
    if s.finish() != got {
        return Err("chunked sha256 differs".into());
    }
    Ok(())
}

// ---------------------------------------------------------------- minimal JSON

#[derive(Debug, Clone)]
enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, x)| x),
            _ => None,
        }
    }
    fn s(&self, k: &str) -> Option<&str> {
        match self.get(k) {
            Some(J::Str(s)) => Some(s),
            _ => None,
        }
    }
    fn b(&self, k: &str) -> Option<bool> {
        match self.get(k) {
            Some(J::Bool(b)) => Some(*b),
            _ => None,
        }
    }
    fn n(&self, k: &str) -> Option<f64> {
        match self.get(k) {
            Some(J::Num(n)) => Some(*n),
            _ => None,
        }
    }
    fn present_non_null(&self, k: &str) -> bool {
        !matches!(self.get(k), None | Some(J::Null))
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn lit(&mut self, s: &str, v: J) -> Result<J, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("short \\u escape".into());
        }
        let t = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        self.i += 4;
        u32::from_str_radix(t, 16).map_err(|e| e.to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3ff);
                            }
                            char::from_u32(cp).unwrap_or('\u{fffd}')
                        }
                        _ => return Err("unknown escape".into()),
                    };
                    let mut tmp = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| "invalid UTF-8 in string".to_string())
    }
    fn val(&mut self) -> Result<J, String> {
        self.ws();
        match self.b.get(self.i).ok_or("unexpected end")? {
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b.get(self.i) == Some(&b'}') {
                        self.i += 1;
                        break;
                    }
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(format!("expected key at {}", self.i));
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(format!("expected ':' at {}", self.i));
                    }
                    self.i += 1;
                    let x = self.val()?;
                    v.push((k, x));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {}
                        _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
                    }
                }
                Ok(J::Obj(v))
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b.get(self.i) == Some(&b']') {
                        self.i += 1;
                        break;
                    }
                    v.push(self.val()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {}
                        _ => return Err(format!("expected ',' or ']' at {}", self.i)),
                    }
                }
                Ok(J::Arr(v))
            }
            b'"' => Ok(J::Str(self.string()?)),
            b't' => self.lit("true", J::Bool(true)),
            b'f' => self.lit("false", J::Bool(false)),
            b'n' => self.lit("null", J::Null),
            _ => {
                let st = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[st..self.i]).map_err(|e| e.to_string())?;
                t.parse::<f64>().map(J::Num).map_err(|_| format!("bad number at {}", st))
            }
        }
    }
}

fn parse_json(b: &[u8]) -> Result<J, String> {
    let mut p = P { b, i: 0 };
    let v = p.val()?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("trailing bytes at {}", p.i));
    }
    Ok(v)
}

fn jesc(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ---------------------------------------------------------------- model

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum V {
    Pass,
    Fail,
    NotRun,
}

impl V {
    fn word(self) -> &'static str {
        match self {
            V::Pass => "PASS",
            V::Fail => "FAIL",
            V::NotRun => "NOT_RUN",
        }
    }
}

#[derive(Clone)]
struct Row {
    name: String,
    v: V,
    reason: String,
}

fn row(name: &str, v: V, reason: impl Into<String>) -> Row {
    Row { name: name.to_string(), v, reason: reason.into() }
}

struct Rec {
    step: String,
    j: J,
}

struct Ctx {
    dir: PathBuf,
    recs: Vec<Rec>,
}

impl Ctx {
    fn rec(&self, step: &str) -> Option<&Rec> {
        self.recs.iter().find(|r| r.step == step)
    }
    fn index(&self, step: &str) -> Option<usize> {
        self.recs.iter().position(|r| r.step == step)
    }
    fn artifact(&self, name: &str) -> PathBuf {
        self.dir.join("artifacts").join(name)
    }
}

// ---------------------------------------------------------------- chain checks

struct Chain {
    problems: Vec<String>,
    notes: Vec<String>,
    objective_id: String,
    run_id: String,
    status_words: BTreeMap<String, String>,
}

fn walk(dir: &Path, base: &Path, out: &mut Vec<String>, problems: &mut Vec<String>) {
    let rd = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => {
            problems.push(format!("UNREADABLE_DIR: {}: {}", dir.display(), e));
            return;
        }
    };
    for e in rd.flatten() {
        let p = e.path();
        let rel = p.strip_prefix(base).unwrap().to_string_lossy().to_string();
        match fs::symlink_metadata(&p) {
            Ok(m) if m.file_type().is_symlink() => problems.push(format!("SYMLINK: {}", rel)),
            Ok(m) if m.is_dir() => walk(&p, base, out, problems),
            Ok(_) => out.push(rel),
            Err(e) => problems.push(format!("UNREADABLE: {}: {}", rel, e)),
        }
    }
}

fn load_chain(dir: &Path, objective_arg: Option<&str>) -> (Ctx, Chain) {
    let mut ch = Chain {
        problems: vec![],
        notes: vec![],
        objective_id: String::new(),
        run_id: String::new(),
        status_words: BTreeMap::new(),
    };
    let mut ctx = Ctx { dir: dir.to_path_buf(), recs: vec![] };
    let cdir = dir.join("chain");
    let mut names: Vec<String> = match fs::read_dir(&cdir) {
        Ok(r) => r
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.len() > 9 && n.ends_with(".json") && n.as_bytes()[..3].iter().all(|c| c.is_ascii_digit()) && n.as_bytes()[3] == b'-')
            .collect(),
        Err(e) => {
            ch.problems.push(format!("NO_CHAIN: cannot read {}: {}", cdir.display(), e));
            return (ctx, ch);
        }
    };
    names.sort();
    if names.is_empty() {
        ch.problems.push("NO_CHAIN: no chain/NNN-*.json receipts".into());
    }
    let mut prev = "0".repeat(64);
    for (i, n) in names.iter().enumerate() {
        let want_no = format!("{:03}", i + 1);
        if !n.starts_with(&want_no) {
            ch.problems.push(format!("SEQUENCE_GAP: expected a receipt numbered {} but found {}", want_no, n));
        }
        let bytes = match fs::read(cdir.join(n)) {
            Ok(b) => b,
            Err(e) => {
                ch.problems.push(format!("UNREADABLE: chain/{}: {}", n, e));
                continue;
            }
        };
        let sha = sha256_bytes(&bytes);
        let j = match parse_json(&bytes) {
            Ok(j) => j,
            Err(e) => {
                ch.problems.push(format!("BAD_JSON: chain/{}: {}", n, e));
                prev = sha;
                continue;
            }
        };
        let step = j.s("step").unwrap_or("").to_string();
        let fname_step = n[4..n.len() - 5].to_string();
        if step != fname_step {
            ch.problems.push(format!("STEP_NAME: chain/{} carries step {:?}", n, step));
        }
        match j.s("prev_receipt_sha256") {
            Some(p) if p == prev => {}
            Some(p) => ch.problems.push(format!(
                "LINK_BROKEN: chain/{} prev_receipt_sha256 {} but the previous file hashes to {}",
                n, p, prev
            )),
            None => ch.problems.push(format!("LINK_BROKEN: chain/{} has no prev_receipt_sha256", n)),
        }
        for (field, slot) in [("objective_id", 0), ("run_id", 1)] {
            let cur = if slot == 0 { &mut ch.objective_id } else { &mut ch.run_id };
            match j.s(field) {
                Some(v) if cur.is_empty() => *cur = v.to_string(),
                Some(v) if v == cur => {}
                Some(v) => ch.problems.push(format!("ID_MISMATCH: chain/{} {} {} differs from {}", n, field, v, cur)),
                None => ch.problems.push(format!("ID_MISMATCH: chain/{} has no {}", n, field)),
            }
        }
        if let Some(st) = j.s("status") {
            ch.status_words.insert(step.clone(), st.to_string());
        }
        ctx.recs.push(Rec { step, j });
        prev = sha;
    }
    // manifest
    let mpath = cdir.join("SHA256SUMS");
    let mut listed: BTreeMap<String, String> = BTreeMap::new();
    match fs::read_to_string(&mpath) {
        Err(e) => ch.problems.push(format!("MANIFEST_MISSING: chain/SHA256SUMS: {}", e)),
        Ok(t) => {
            for (ln, line) in t.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                if line.len() > 66 && is_hex64(&line[..64]) && &line[64..66] == "  " {
                    let p = line[66..].trim_start_matches("./").to_string();
                    if listed.insert(p.clone(), line[..64].to_string()).is_some() {
                        ch.problems.push(format!("MANIFEST_DUPLICATE: {} listed twice", p));
                    }
                } else {
                    ch.problems.push(format!("MANIFEST_SYNTAX: SHA256SUMS line {}", ln + 1));
                }
            }
        }
    }
    let mut files = vec![];
    walk(dir, dir, &mut files, &mut ch.problems);
    files.sort();
    const SELF_OUT: [&str; 3] = ["chain/SHA256SUMS", "VERIFIER_VERDICT.txt", "VERIFIER_RECEIPT.json"];
    for f in &files {
        if SELF_OUT.contains(&f.as_str()) {
            continue;
        }
        match listed.get(f) {
            None => ch.problems.push(format!("MANIFEST_UNLISTED: {} is in the run folder but not in chain/SHA256SUMS", f)),
            Some(want) => match sha256_file(&dir.join(f)) {
                Ok(got) if &got == want => {}
                Ok(got) => ch.problems.push(format!("MANIFEST_DIGEST: {} hashes to {} but the manifest says {}", f, got, want)),
                Err(e) => ch.problems.push(format!("UNREADABLE: {}", e)),
            },
        }
    }
    for (p, _) in &listed {
        if !files.contains(p) {
            if p.ends_with(".log") {
                ch.notes.push(format!("manifest lists {} which is absent from the folder (logs are never an input to a row)", p));
            } else {
                ch.problems.push(format!("MANIFEST_ABSENT: {} is listed in chain/SHA256SUMS but missing from the run folder", p));
            }
        }
    }
    // objective id and run id recomputation
    if let Some(txt) = ctx.rec("E2").and_then(|r| r.j.s("objective_text")) {
        let want = sha256_bytes(format!("AIEN_E2E_OBJECTIVE_V1\n{}", txt).as_bytes());
        if want != ch.objective_id {
            ch.problems.push(format!("OBJECTIVE_ID: text in the E2 receipt hashes to {} but the chain says {}", want, ch.objective_id));
        }
        if let Some(a) = objective_arg {
            if a != want {
                ch.problems.push(format!("OBJECTIVE_ID: --objective-id {} differs from the recomputed {}", a, want));
            }
        }
    } else {
        ch.notes.push("no objective text in an E2 receipt: objective_id not recomputed".into());
        if let Some(a) = objective_arg {
            if a != ch.objective_id {
                ch.problems.push(format!("OBJECTIVE_ID: --objective-id {} differs from the chain's {}", a, ch.objective_id));
            }
        }
    }
    if let Some(pre) = ctx.rec("PRE") {
        if let (Some(d), Some(w), Some(h)) = (pre.j.s("daemon_sha256"), pre.j.s("model_weights_digest"), pre.j.s("host")) {
            let want = sha256_bytes(format!("{}\n{}\n{}\n{}", ch.objective_id, d, w, h).as_bytes());
            if want != ch.run_id {
                ch.problems.push(format!("RUN_ID: recomputed {} but the chain says {}", want, ch.run_id));
            }
        } else {
            ch.notes.push("PRE receipt lacks daemon, weights or host: run_id not recomputed".into());
        }
    }
    (ctx, ch)
}

// ---------------------------------------------------------------- row derivation

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

fn topics_from(obj: &str) -> Vec<String> {
    let Some(p) = obj.find("Mention ") else { return vec![] };
    let rest = &obj[p + 8..];
    let end = rest.rfind('.').unwrap_or(rest.len());
    rest[..end]
        .replace(" and ", ", ")
        .split(", ")
        .map(|t| t.trim().trim_start_matches("the ").to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn required_names(ctx: &Ctx, step: &str) -> Vec<String> {
    let r = match ctx.rec(step) {
        Some(r) => r,
        None => return vec![],
    };
    if let Some(J::Arr(a)) = r.j.get("required_names") {
        return a.iter().filter_map(|x| if let J::Str(s) = x { Some(s.clone()) } else { None }).collect();
    }
    if step == "E3m" {
        return r.j.s("objective_text").map(topics_from).unwrap_or_default();
    }
    match ctx.rec("PRE").and_then(|p| p.j.get("inbox_files")) {
        Some(J::Arr(a)) => a.iter().filter_map(|f| f.s("name").map(String::from)).collect(),
        _ => vec![],
    }
}

/// The E3m D4 rule of ACCEPTANCE-v1.3.1-e3m.md, as the harness writes it into the E3m receipt's `d4_rule`.
const D4_RULE_V131: &str = "case-insensitive whole word: LC_ALL=C grep -q -i -w (ACCEPTANCE-v1.3.1-e3m.md)";

/// GNU `grep -i -w` under `LC_ALL=C`: `word` occurs in `text` in any ASCII letter case with no ASCII letter, digit or
/// underscore directly before or after that occurrence (VERIFIER-v1.1-e3m.md).
fn contains_word_ci(text: &str, word: &str) -> bool {
    let (t, w) = (text.as_bytes(), word.as_bytes());
    if w.is_empty() || w.len() > t.len() {
        return false;
    }
    let wc = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    (0..=t.len() - w.len()).any(|i| {
        t[i..i + w.len()].eq_ignore_ascii_case(w) && (i == 0 || !wc(t[i - 1])) && (i + w.len() == t.len() || !wc(t[i + w.len()]))
    })
}

fn base(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

fn doc_row(ctx: &Ctx, step: &str) -> Row {
    let r = match ctx.rec(step) {
        Some(r) => r,
        None => return row(step, V::NotRun, "no receipt for this step in the chain"),
    };
    let j = &r.j;
    let Some(state) = j.s("effect_state") else {
        return row(step, V::NotRun, "receipt has no effect_state fact");
    };
    if state != "DONE" {
        return row(step, V::Fail, format!("effect_state {:?}: the file-writing effect did not happen", state));
    }
    let (Some(wp), Some(cs)) = (j.s("written_path"), j.s("content_sha256")) else {
        return row(step, V::NotRun, "effect_state DONE but no written_path or content_sha256 in the receipt");
    };
    if !is_hex64(cs) {
        return row(step, V::NotRun, "content_sha256 is not a sha256");
    }
    let rel = format!("{}-{}", step, base(wp));
    let path = ctx.artifact(&rel);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(_) => return row(step, V::Fail, format!("OUTPUT_ABSENT: artifacts/{} is not in the run folder", rel)),
    };
    let got = sha256_bytes(&bytes);
    if got != cs {
        return row(step, V::Fail, format!("DIGEST_MISMATCH: artifacts/{} hashes to {} but the receipt content_sha256 is {}", rel, got, cs));
    }
    if let Some(d) = j.s("disk_sha256") {
        if d != cs {
            return row(step, V::Fail, format!("DIGEST_MISMATCH: disk_sha256 {} differs from content_sha256 {}", d, cs));
        }
    }
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return row(step, V::Fail, "D1: output is not valid UTF-8"),
    };
    let mut failed = vec![];
    if !text.starts_with("# ") {
        failed.push("D2 first line does not start with \"# \"".to_string());
    }
    let wc = word_count(&text);
    if wc > 200 {
        failed.push(format!("D3 {} words, at most 200 allowed", wc));
    }
    let names = required_names(ctx, step);
    if names.is_empty() {
        return row(step, V::NotRun, "no required names or topics can be read from the receipts");
    }
    // D4: E3m topic words follow the rule its receipt was judged under (VERIFIER-v1.1-e3m.md); E3 file names stay exact.
    let whole_word = match (step, j.s("d4_rule")) {
        ("E3m", Some(D4_RULE_V131)) => true,
        ("E3m", Some(other)) => return row(step, V::NotRun, format!("unknown d4_rule {:?} in the receipt", other)),
        _ => false,
    };
    let missing: Vec<&String> =
        names.iter().filter(|n| if whole_word { !contains_word_ci(&text, n) } else { !text.contains(n.as_str()) }).collect();
    if !missing.is_empty() {
        failed.push(format!("D4 required name(s) missing: {:?}", missing));
    }
    if !failed.is_empty() {
        return row(step, V::Fail, format!("document rules: {}", failed.join("; ")));
    }
    if j.s("desk") != Some("required") {
        return row(step, V::Fail, "desk is not 'required'");
    }
    if j.n("grant_id").is_none() {
        return row(step, V::Fail, "no grant_id: no grant was spent");
    }
    match j.s("second_execute_refusal") {
        None => return row(step, V::NotRun, "no second_execute_refusal fact (spent-once not shown)"),
        Some("AlreadySpent") => {}
        Some(o) => return row(step, V::Fail, format!("second execute gave {:?}, not AlreadySpent", o)),
    }
    if let Some(cr) = j.get("compose_receipt") {
        if matches!(cr, J::Obj(_)) && cr.b("success") != Some(true) {
            return row(step, V::Fail, "compose_receipt does not record success true");
        }
    }
    row(step, V::Pass, format!("DONE, digest matches, {} words, names present, grant spent once", wc))
}

fn e1_row(ctx: &Ctx) -> Row {
    let Some(r) = ctx.rec("E1") else { return row("E1", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let sig = j.s("signature_check").unwrap_or("none");
    if sig == "none" || !j.present_non_null("artifact_sha256") {
        return row("E1", V::NotRun, "no release artifact or signature check in the receipt");
    }
    if sig != "verified" && sig != "dry-run-key" {
        return row("E1", V::Fail, format!("signature_check {:?}", sig));
    }
    let art_ok = j.s("artifact_sha256").map(is_hex64).unwrap_or(false);
    let files_ok = j.n("files_verified").map(|n| n > 0.0).unwrap_or(false);
    let bins_ok = j.s("daemon_sha256").map(is_hex64).unwrap_or(false) && j.s("cli_sha256").map(is_hex64).unwrap_or(false);
    if art_ok && files_ok && bins_ok {
        row("E1", V::Pass, format!("signature_check {}, artifact and binaries hashed, files verified", sig))
    } else {
        row("E1", V::NotRun, "artifact_sha256, files_verified, daemon_sha256 or cli_sha256 missing or malformed")
    }
}

fn e2_row(ctx: &Ctx) -> Row {
    let Some(r) = ctx.rec("E2") else { return row("E2", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let (Some(t), Some(h)) = (j.s("objective_text"), j.s("objective_text_sha256")) else {
        return row("E2", V::NotRun, "no objective text or text digest in the receipt");
    };
    if sha256_bytes(t.as_bytes()) != h {
        return row("E2", V::Fail, "objective_text_sha256 does not match the objective text");
    }
    if j.s("operator_surface") != Some("aien-cli") {
        return row("E2", V::NotRun, format!("operator_surface {:?} is not aien-cli", j.s("operator_surface").unwrap_or("")));
    }
    let inst = j.s("installation_record").unwrap_or("NOT_RUN");
    if inst.starts_with("NOT_RUN") {
        return row("E2", V::NotRun, "no installation-side record of the objective (harness-side only)");
    }
    if j.b("recorded_before_work") != Some(true) {
        return row("E2", V::Fail, "recorded_before_work is not true");
    }
    let me = ctx.index("E2").unwrap();
    for s in ["E3", "E3m", "E4", "E6"] {
        if let Some(i) = ctx.index(s) {
            if i < me {
                return row("E2", V::Fail, format!("effect receipt {} predates the objective record", s));
            }
        }
    }
    row("E2", V::Pass, "objective recorded through aien-cli before any effect receipt")
}

fn e4_row(ctx: &Ctx, e4_recalled: &mut bool) -> Row {
    let Some(r) = ctx.rec("E4") else { return row("E4", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let Some(rk) = j.s("restart_kind") else { return row("E4", V::NotRun, "receipt carries no restart facts") };
    if rk != "graceful" && rk != "sigkill" {
        return row("E4", V::Fail, format!("restart_kind {:?}", rk));
    }
    if let Some(st) = j.s("effect_state") {
        if st != "DONE" {
            return row("E4", V::Fail, format!("second objective produced no effect (effect_state {:?}); recall after restart not shown", st));
        }
    }
    match j.b("recall_after_restart") {
        None => return row("E4", V::NotRun, "no recall_after_restart fact"),
        Some(false) => return row("E4", V::Fail, "recall_after_restart is false"),
        Some(true) => {}
    }
    let (Some(_store), Some(item)) = (j.s("memory_store"), j.s("item_sha256")) else {
        return row("E4", V::NotRun, "memory_store or item_sha256 missing");
    };
    if !is_hex64(item) {
        return row("E4", V::NotRun, "item_sha256 is not a sha256");
    }
    if let Ok(got) = sha256_file(&ctx.artifact("E4-remember.json")) {
        if got != item {
            return row("E4", V::Fail, "DIGEST_MISMATCH: item_sha256 differs from artifacts/E4-remember.json");
        }
    }
    let (Some(ru), Some(cu)) = (j.s("restart_utc"), j.s("recall_utc")) else {
        return row("E4", V::NotRun, "restart_utc or recall_utc missing");
    };
    if cu < ru {
        return row("E4", V::Fail, "the recall receipt predates the restart");
    }
    let (base_step, first) = if ctx.rec("E3m").is_some() { ("E3m", required_names(ctx, "E3m")) } else { ("E3", required_names(ctx, "E3")) };
    let Some(first) = first.first().cloned() else { return row("E4", V::NotRun, "no first required name to look for") };
    let before_rel = format!("{}-report.md", base_step);
    let (Ok(before), Ok(after)) = (fs::read_to_string(ctx.artifact(&before_rel)), fs::read_to_string(ctx.artifact("E4-report.md"))) else {
        return row("E4", V::Fail, format!("OUTPUT_ABSENT: artifacts/{} or artifacts/E4-report.md missing", before_rel));
    };
    let nl = |s: &str| s.bytes().filter(|&c| c == b'\n').count();
    if !after.starts_with(&before) || nl(&after) != nl(&before) + 1 {
        return row("E4", V::Fail, "the report after the second objective is not the earlier report plus exactly one line");
    }
    let last = after.trim_end_matches('\n').rsplit('\n').next().unwrap_or("");
    if !last.to_lowercase().contains(&first.to_lowercase()) {
        return row("E4", V::Fail, format!("the appended line does not name {:?}", first));
    }
    *e4_recalled = true;
    row("E4", V::Pass, format!("{} restart, recall after restart, one line appended naming {:?}", rk, first))
}

fn e6_row(ctx: &Ctx) -> Row {
    let Some(r) = ctx.rec("E6") else { return row("E6", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let (Some(kp), Some(rc), Some(eb), Some(ea), Some(dup)) = (
        j.s("kill_point"),
        j.n("restart_count"),
        j.n("effects_before_kill"),
        j.n("effects_after_restart"),
        j.n("duplicates"),
    ) else {
        return row("E6", V::NotRun, "kill_point, restart_count, effects_before_kill, effects_after_restart or duplicates missing");
    };
    let Some(tr) = j.s("third_execute_refusal") else { return row("E6", V::NotRun, "no third_execute_refusal fact") };
    let mut bad = vec![];
    if kp.is_empty() {
        bad.push("empty kill_point".to_string());
    }
    if rc < 1.0 {
        bad.push(format!("restart_count {}", rc));
    }
    if eb != 0.0 {
        bad.push(format!("effects_before_kill {}", eb));
    }
    if ea != 1.0 {
        bad.push(format!("effects_after_restart {}", ea));
    }
    if dup != 0.0 {
        bad.push(format!("duplicates {}", dup));
    }
    if j.n("grant_id").is_none() {
        bad.push("no grant_id".into());
    }
    if tr != "AlreadySpent" {
        bad.push(format!("third execute gave {:?}", tr));
    }
    if bad.is_empty() {
        row("E6", V::Pass, format!("killed at {}, restarted, effect once, duplicates 0, third execute AlreadySpent", kp))
    } else {
        row("E6", V::Fail, bad.join("; "))
    }
}

fn ctrl_simple(ctx: &Ctx, step: &str, check: impl Fn(&J) -> Option<Result<String, String>>) -> Row {
    let Some(r) = ctx.rec(step) else { return row(step, V::NotRun, "no receipt for this step in the chain") };
    match check(&r.j) {
        None => row(step, V::NotRun, "receipt lacks the facts this control needs"),
        Some(Ok(m)) => row(step, V::Pass, m),
        Some(Err(m)) => row(step, V::Fail, m),
    }
}

fn contains_ci(h: Option<&str>, n: &str) -> bool {
    h.map(|h| h.to_lowercase().contains(&n.to_lowercase())).unwrap_or(false)
}

fn ctrl_e3a(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "CTRL-E3a", |j| {
        let ex = j.s("exit")?.trim().parse::<i64>().ok()?;
        let log = j.s("log_line")?;
        Some(if ex != 0 && contains_ci(Some(log), "NoDesk") {
            Ok(format!("daemon exited {} before serving and names NoDesk", ex))
        } else {
            Err(format!("exit {} and log does not name NoDesk", ex))
        })
    })
}

fn ctrl_e3b(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "CTRL-E3b", |j| {
        let r = j.s("refusal")?;
        let a = j.s("file_absent")?;
        Some(if r == "Revoked" && a == "yes" { Ok("grant revoked before spend: refused Revoked, no file".into()) } else { Err(format!("refusal {:?}, file_absent {:?}", r, a)) })
    })
}

fn ctrl_e4(ctx: &Ctx, e4_recalled: bool) -> Row {
    let Some(r) = ctx.rec("CTRL-E4") else { return row("CTRL-E4", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let (Some(st), Some(ch)) = (j.s("effect_state"), j.s("report_changed")) else {
        return row("CTRL-E4", V::NotRun, "receipt lacks effect_state or report_changed");
    };
    if st == "DONE" || ch != "no" {
        return row("CTRL-E4", V::Fail, format!("store removed yet effect_state {:?}, report_changed {:?}", st, ch));
    }
    let named = match fs::read(ctx.artifact("CTRL-E4-propose.json")).ok().and_then(|b| parse_json(&b).ok()) {
        Some(a) => a.s("error").map(|e| e.contains("E_") || e.contains("refused")).unwrap_or(false),
        None => contains_ci(j.s("evidence"), "refused"),
    };
    if !named {
        return row("CTRL-E4", V::Fail, "no named refusal for the removed store");
    }
    if !e4_recalled {
        return row("CTRL-E4", V::NotRun, "refusal named, but E4 did not show a recall, so the control tests store loss only, not memory loss");
    }
    row("CTRL-E4", V::Pass, "store removed: named refusal, report unchanged")
}

fn ctrl_e1(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "CTRL-E1", |j| {
        let ir = j.b("install_refused")?;
        let why = j.s("refusal_reason")?;
        Some(if ir && (contains_ci(Some(why), "checksum") || contains_ci(Some(why), "signature")) {
            Ok("mutated artifact refused, reason names checksum or signature".into())
        } else {
            Err(format!("install_refused {}, reason {:?}", ir, why))
        })
    })
}

fn ctrl_e2(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "CTRL-E2", |j| {
        let rb = j.b("refused_before_execution")?;
        let nm = j.s("refusal_name")?;
        Some(if rb && !nm.is_empty() { Ok(format!("refused before execution: {}", nm)) } else { Err(format!("refused_before_execution {}, refusal_name {:?}", rb, nm)) })
    })
}

// CTRL-E5 (v1.3-ctrl-e5): judged from the LAST CTRL-E5 receipt's facts. `pre_id` is the verdict id this verifier computes
// from the folder with CTRL-E5 and E5 forced to NOT_RUN; the receipt's original_verdict_id must equal it.
fn ctrl_e5(ctx: &Ctx, pre_id: &str) -> Row {
    let Some(r) = ctx.recs.iter().rev().find(|r| r.step == "CTRL-E5") else { return row("CTRL-E5", V::NotRun, "no receipt for this step in the chain") };
    let j = &r.j;
    let (Some(rej), Some(mcu), Some(orig), Some(mutd), Some(why)) =
        (j.b("mutated_output_rejected"), j.b("mutated_chain_unbroken"), j.s("original_verdict_id"), j.s("mutated_verdict_id"), j.s("verifier_reason"))
    else {
        return row("CTRL-E5", V::NotRun, "receipt lacks the facts this control needs");
    };
    let mut bad = vec![];
    if !rej {
        bad.push("mutated_output_rejected false".to_string());
    }
    if mcu {
        bad.push("mutated copy's chain still unbroken".into());
    }
    if !why.contains("DIGEST_MISMATCH") {
        bad.push(format!("reason {:?} does not name DIGEST_MISMATCH", why));
    }
    if !is_hex64(orig) || !is_hex64(mutd) || orig == mutd {
        bad.push("mutated verdict id is not a different, well-formed id".into());
    }
    if orig != pre_id {
        bad.push(format!("original_verdict_id {} is not the pre-CTRL-E5 id {} this verifier derives", orig, pre_id));
    }
    if bad.is_empty() {
        row("CTRL-E5", V::Pass, "one byte of an output changed on a copy: rejected on DIGEST_MISMATCH, chain broken, verdict id changed")
    } else {
        row("CTRL-E5", V::Fail, bad.join("; "))
    }
}

fn ctrl_e6(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "CTRL-E6", |j| {
        let nm = j.s("refusal_name")?;
        let ea = j.n("effects_after_restart")?;
        Some(if !nm.is_empty() && ea == 0.0 { Ok(format!("durable state removed: refused {}, no second effect", nm)) } else { Err(format!("refusal_name {:?}, effects_after_restart {}", nm, ea)) })
    })
}

fn net_row(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "NET", |j| {
        let n = j.n("daemon_established_tcp")?;
        Some(if n == 0.0 { Ok("daemon held no established TCP socket".into()) } else { Err(format!("{} established TCP socket(s)", n)) })
    })
}

fn pre_row(ctx: &Ctx) -> Row {
    ctrl_simple(ctx, "PRE", |j| {
        let w = j.s("model_weights_digest")?;
        let cw = j.s("candidate_weights_sha256")?;
        let t = j.s("tokenizer_sha256")?;
        let ct = j.s("candidate_tokenizer_sha256")?;
        let bl = j.s("backend_line")?;
        let h = j.s("host")?;
        Some(if is_hex64(w) && w == cw && t == ct && !bl.is_empty() && !h.is_empty() {
            Ok("model digests equal the candidate record; backend line and host present".into())
        } else {
            Err("model digests differ from the candidate record or backend line missing".into())
        })
    })
}

// ---------------------------------------------------------------- verdict

fn verdict_id(objective: &str, contract: &str, rows: &BTreeMap<String, V>) -> String {
    let mut lines = vec![format!("objective_id {}", objective), format!("contract_sha256 {}", contract)];
    for i in 1..=6 {
        let k = format!("E{}", i);
        lines.push(format!("{} {}", k, rows.get(&k).copied().unwrap_or(V::NotRun).word()));
    }
    for i in 1..=6 {
        let k = format!("CTRL-E{}", i);
        lines.push(format!("{} {}", k, rows.get(&k).copied().unwrap_or(V::NotRun).word()));
    }
    sha256_bytes(format!("AIEN_E2E_VERDICT_V1\n{}", lines.join("\n")).as_bytes())
}

struct Opts {
    dir: PathBuf,
    contract: String,
    objective: Option<String>,
    peer: Option<String>,
    write: bool,
}

fn parse_args(a: &[String]) -> Result<Opts, String> {
    let mut dir = None;
    let (mut contract, mut objective, mut peer) = (None, None, None);
    let mut write = false;
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--contract-sha256" | "--objective-id" | "--peer-verdict-id" => {
                let v = a.get(i + 1).ok_or(format!("{} needs a value", a[i]))?.clone();
                if !is_hex64(&v) {
                    return Err(format!("{} must be 64 lowercase hex characters", a[i]));
                }
                match a[i].as_str() {
                    "--contract-sha256" => contract = Some(v),
                    "--objective-id" => objective = Some(v),
                    _ => peer = Some(v),
                }
                i += 2;
            }
            "--write" => {
                write = true;
                i += 1;
            }
            s if !s.starts_with("--") && dir.is_none() => {
                dir = Some(PathBuf::from(s));
                i += 1;
            }
            s => return Err(format!("unknown argument {}", s)),
        }
    }
    Ok(Opts {
        dir: dir.ok_or("usage: aien-verify RUN_DIR --contract-sha256 HEX [--objective-id HEX] [--peer-verdict-id HEX] [--write]")?,
        contract: contract.ok_or("--contract-sha256 is required (the frozen contract digest, never compiled in)")?,
        objective,
        peer,
        write,
    })
}

fn build_host() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--selftest") {
        return match selftest() {
            Ok(()) => {
                println!("selftest ok: sha256 vectors (empty, abc, two-block, fox, million a)");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("selftest FAILED: {}", e);
                ExitCode::from(3)
            }
        };
    }
    let o = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aien-verify: {}", e);
            return ExitCode::from(2);
        }
    };
    let (ctx, chain) = load_chain(&o.dir, o.objective.as_deref());
    let chain_unbroken = chain.problems.is_empty();

    let mut e4_recalled = false;
    let mut rows: Vec<Row> = vec![];
    rows.push(e1_row(&ctx));
    rows.push(e2_row(&ctx));
    rows.push(doc_row(&ctx, "E3"));
    rows.push(e4_row(&ctx, &mut e4_recalled));
    let e5_idx = rows.len();
    rows.push(row("E5", V::NotRun, "placeholder"));
    rows.push(e6_row(&ctx));
    rows.push(ctrl_e1(&ctx));
    rows.push(ctrl_e2(&ctx));
    let a = ctrl_e3a(&ctx);
    let b = ctrl_e3b(&ctx);
    let c3 = match (a.v, b.v) {
        (V::Pass, V::Pass) => row("CTRL-E3", V::Pass, "CTRL-E3a and CTRL-E3b both PASS"),
        (V::Fail, _) | (_, V::Fail) => row("CTRL-E3", V::Fail, format!("CTRL-E3a {}, CTRL-E3b {}", a.v.word(), b.v.word())),
        _ => row("CTRL-E3", V::NotRun, format!("CTRL-E3a {} ({}); CTRL-E3b {} ({})", a.v.word(), a.reason, b.v.word(), b.reason)),
    };
    rows.push(c3);
    rows.push(ctrl_e4(&ctx, e4_recalled));
    let c5_idx = rows.len();
    rows.push(row("CTRL-E5", V::NotRun, "placeholder"));
    rows.push(ctrl_e6(&ctx));
    let mut supp = vec![pre_row(&ctx), doc_row(&ctx, "E3m"), a, b, net_row(&ctx)];

    // E5: base verdict id with E5 NOT_RUN, then the peer comparison
    let map = |rows: &Vec<Row>| rows.iter().map(|r| (r.name.clone(), r.v)).collect::<BTreeMap<_, _>>();
    let pre_id = verdict_id(&chain.objective_id, &o.contract, &map(&rows));
    rows[c5_idx] = ctrl_e5(&ctx, &pre_id);
    let base_id = verdict_id(&chain.objective_id, &o.contract, &map(&rows));
    rows[e5_idx] = match (&o.peer, chain_unbroken) {
        (None, _) => row("E5", V::NotRun, "no --peer-verdict-id: cross-machine agreement not shown"),
        (Some(_), false) => row("E5", V::Fail, "chain is not unbroken"),
        (Some(p), true) if *p == base_id => row("E5", V::Pass, "chain unbroken and the peer machine's base verdict id equals this machine's"),
        (Some(p), true) => row("E5", V::Fail, format!("peer base verdict id {} differs from this machine's {}", p, base_id)),
    };
    let vid = verdict_id(&chain.objective_id, &o.contract, &map(&rows));

    // status words in receipts that contradict the derived rows (never copied, only reported)
    let mut disagreements = vec![];
    let derived: BTreeMap<String, V> = rows.iter().chain(supp.iter()).map(|r| (r.name.clone(), r.v)).collect();
    for (step, st) in &chain.status_words {
        if let Some(v) = derived.get(step) {
            if v.word() != st && ["PASS", "FAIL", "NOT_RUN"].contains(&st.as_str()) && step != "E5" {
                disagreements.push(format!("status_disagreement: {} receipt says {} but the facts derive {}", step, st, v.word()));
            }
        }
    }

    let any_fail = rows.iter().chain(supp.iter()).any(|r| r.v == V::Fail);
    let ok = chain_unbroken && !any_fail;

    let mut verdict = String::from("AIEN_E2E_VERDICT_V1\n");
    verdict.push_str(&format!("objective_id {}\ncontract_sha256 {}\n", chain.objective_id, o.contract));
    for r in rows.iter().filter(|r| r.name != "CTRL-E3").filter(|r| r.name.starts_with('E')) {
        verdict.push_str(&format!("{} {}\n", r.name, r.v.word()));
    }
    for r in rows.iter().filter(|r| r.name.starts_with("CTRL-E")) {
        verdict.push_str(&format!("{} {}\n", r.name, r.v.word()));
    }
    verdict.push_str(&format!("verdict_id {}\nchain_unbroken {}\nbase_verdict_id {}\n", vid, chain_unbroken, base_id));
    for r in rows.iter() {
        verdict.push_str(&format!("# {} {}: {}\n", r.name, r.v.word(), r.reason));
    }
    for r in supp.drain(..) {
        verdict.push_str(&format!("# (supplementary) {} {}: {}\n", r.name, r.v.word(), r.reason));
        rows.push(Row { name: format!("supp:{}", r.name), ..r });
    }
    for p in &chain.problems {
        verdict.push_str(&format!("# CHAIN_PROBLEM {}\n", p));
    }
    for d in &disagreements {
        verdict.push_str(&format!("# {}\n", d));
    }
    for n in &chain.notes {
        verdict.push_str(&format!("# note: {}\n", n));
    }
    print!("{}", verdict);

    if o.write {
        let exe = std::env::current_exe().ok().or_else(|| args.first().map(PathBuf::from));
        let vsha = exe.and_then(|p| sha256_file(&p).ok());
        let mut j = String::from("{\n");
        j.push_str(&format!("  \"verifier_sha256\": {},\n", vsha.map(|s| jesc(&s)).unwrap_or("null".into())));
        j.push_str(&format!("  \"verifier_build_host\": {},\n", jesc(&build_host())));
        j.push_str(&format!("  \"chain_unbroken\": {},\n", chain_unbroken));
        j.push_str(&format!("  \"objective_id\": {},\n  \"contract_sha256\": {},\n", jesc(&chain.objective_id), jesc(&o.contract)));
        j.push_str("  \"rows\": [\n");
        let n = rows.len();
        for (i, r) in rows.iter().enumerate() {
            j.push_str(&format!(
                "    {{\"row\": {}, \"verdict\": {}, \"reason\": {}}}{}\n",
                jesc(&r.name),
                jesc(r.v.word()),
                jesc(&r.reason),
                if i + 1 < n { "," } else { "" }
            ));
        }
        j.push_str("  ],\n  \"chain_problems\": [");
        j.push_str(&chain.problems.iter().map(|p| jesc(p)).collect::<Vec<_>>().join(", "));
        j.push_str("],\n  \"status_disagreements\": [");
        j.push_str(&disagreements.iter().map(|p| jesc(p)).collect::<Vec<_>>().join(", "));
        j.push_str(&format!("],\n  \"base_verdict_id\": {},\n  \"verdict_id\": {}\n}}\n", jesc(&base_id), jesc(&vid)));
        if let Err(e) = fs::write(o.dir.join("VERIFIER_VERDICT.txt"), &verdict).and_then(|_| fs::write(o.dir.join("VERIFIER_RECEIPT.json"), &j)) {
            eprintln!("aien-verify: cannot write into the run folder: {}", e);
            return ExitCode::from(2);
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        eprintln!("aien-verify: NOT OK (chain_unbroken {}, FAIL rows: {})", chain_unbroken, rows.iter().filter(|r| r.v == V::Fail).count());
        ExitCode::from(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_vectors() {
        selftest().unwrap();
    }
    #[test]
    fn whole_word_ci_is_grep_i_w() {
        for yes in ["# Windmill\n", "WINDMILL", "the windmill.", "windmill,", "windmill's", "a windmill", "x-windmill"] {
            assert!(contains_word_ci(yes, "windmill"), "{yes:?}");
        }
        for no in ["windmills", "windmill_x", "wind-mill", "awindmill", "windmill9", "", "windmil"] {
            assert!(!contains_word_ci(no, "windmill"), "{no:?}");
        }
        assert!(contains_word_ci("windmills and a Windmill", "windmill"), "a later occurrence counts");
        assert!(contains_word_ci("caf\u{e9} windmill", "windmill"));
    }
    #[test]
    fn topics() {
        assert_eq!(
            topics_from("Write x. Mention the harbour, the orchard and the windmill."),
            vec!["harbour", "orchard", "windmill"]
        );
    }
}
