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

//! The manifest: one line per fact, `#` starts a comment.
//!
//! ```text
//! abi rxc_host 2                       name and contract version (informational)
//! header src/runtime/rxc_host_abi.h    header(s) the C side is read from, relative to omega
//! struct RxcHostInfo 80 8              name, sizeof, alignment
//! field RxcHostInfo machine_id 5 32 u8 32   struct, field, offset, size, kind, element count
//! const RXC_HOST_ABI_VERSION 2         mirrored constant and its value
//! fnptr RxcHostSkillFn fn(*mut c_void, u64, *mut u64) -> i32
//! fn rxc_host_open fn(*const c_char, u32, ...) -> i32
//! ```
//!
//! Kinds: `bool c_char i8 u8 i16 u16 i32 u32 i64 u64 f32 f64` (`size_t` is `u64` on the
//! LP64 targets this code runs on). Types in signatures use Rust spelling, canonicalised by
//! [`canon_type`] so `c_int`, `int` and `int32_t` all read `i32`.

use std::collections::BTreeSet;

#[derive(Debug, Default, Clone)]
pub struct Manifest {
    pub abi: Option<(String, String)>,
    pub headers: Vec<String>,
    /// (name, size, align)
    pub structs: Vec<(String, usize, usize)>,
    /// (struct, field, offset, size, kind, count), in declaration order per struct.
    pub fields: Vec<(String, String, usize, usize, String, usize)>,
    pub consts: Vec<(String, i128)>,
    pub fnptrs: Vec<(String, String)>,
    pub fns: Vec<(String, String)>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let mut m = Manifest::default();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let bad = |why: &str| format!("manifest line {}: {why}: {raw}", i + 1);
            let w: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| s.parse::<usize>().map_err(|_| bad("not a number"));
            match w[0] {
                "abi" if w.len() == 3 => m.abi = Some((w[1].into(), w[2].into())),
                "header" if w.len() == 2 => m.headers.push(w[1].into()),
                "struct" if w.len() == 4 => m.structs.push((w[1].into(), num(w[2])?, num(w[3])?)),
                "field" if w.len() == 7 => m.fields.push((
                    w[1].into(),
                    w[2].into(),
                    num(w[3])?,
                    num(w[4])?,
                    w[5].into(),
                    num(w[6])?,
                )),
                "const" if w.len() == 3 => m.consts.push((
                    w[1].into(),
                    w[2].parse::<i128>().map_err(|_| bad("not an integer"))?,
                )),
                "fnptr" | "fn" if w.len() >= 3 => {
                    let rest = line[w[0].len()..].trim_start()[w[1].len()..].trim();
                    let sig = canon_sig(rest).ok_or_else(|| bad("bad signature"))?;
                    if w[0] == "fn" {
                        m.fns.push((w[1].into(), sig));
                    } else {
                        m.fnptrs.push((w[1].into(), sig));
                    }
                }
                _ => return Err(bad("unknown or malformed line")),
            }
        }
        let mut seen = BTreeSet::new();
        for (s, ..) in &m.structs {
            if !seen.insert(s.clone()) {
                return Err(format!("manifest: struct {s} listed twice"));
            }
        }
        for (s, f, ..) in &m.fields {
            if !seen.contains(s) {
                return Err(format!("manifest: field {s}.{f} of an unlisted struct"));
            }
        }
        Ok(m)
    }

    /// The comparable facts (everything but `abi` and `header`), one canonical line each,
    /// in file order.
    pub fn fact_lines(&self) -> Vec<String> {
        let mut v = Vec::new();
        for (s, size, align) in &self.structs {
            v.push(format!("struct {s} {size} {align}"));
            for (fs, f, off, sz, kind, n) in &self.fields {
                if fs == s {
                    v.push(format!("field {s} {f} {off} {sz} {kind} {n}"));
                }
            }
        }
        for (n, val) in &self.consts {
            v.push(format!("const {n} {val}"));
        }
        for (n, sig) in &self.fnptrs {
            v.push(format!("fnptr {n} {sig}"));
        }
        for (n, sig) in &self.fns {
            v.push(format!("fn {n} {sig}"));
        }
        v
    }

    pub fn field_names(&self, s: &str) -> Vec<String> {
        self.fields
            .iter()
            .filter(|f| f.0 == s)
            .map(|f| f.1.clone())
            .collect()
    }
}

/// Canonical spelling of one Rust-syntax type: whitespace removed, `Option<..>` (a nullable
/// function pointer has the same ABI as the bare one) and `std::os::raw::` paths dropped,
/// C-alias integers mapped to their fixed-width names, then `*const `/`*mut ` re-spaced.
pub fn canon_type(t: &str) -> String {
    let mut s: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    for p in [
        "std::os::raw::",
        "core::ffi::",
        "std::ffi::",
        "::std::os::raw::",
    ] {
        s = s.replace(p, "");
    }
    while let Some(inner) = s.strip_prefix("Option<").and_then(|r| r.strip_suffix('>')) {
        s = inner.to_string();
    }
    let mut out = String::new();
    let mut rest = s.as_str();
    loop {
        if let Some(r) = rest.strip_prefix("*const") {
            out.push_str("*const ");
            rest = r;
        } else if let Some(r) = rest.strip_prefix("*mut") {
            out.push_str("*mut ");
            rest = r;
        } else {
            break;
        }
    }
    let base = match rest {
        "c_int" => "i32",
        "c_uint" => "u32",
        "c_long" | "c_longlong" => "i64",
        "c_ulong" | "c_ulonglong" => "u64",
        "c_short" => "i16",
        "c_ushort" => "u16",
        "c_float" => "f32",
        "c_double" => "f64",
        "c_schar" => "i8",
        "c_uchar" => "u8",
        "" | "()" => "()",
        other => other,
    };
    out.push_str(base);
    out
}

/// Canonical signature `fn(a, b) -> r` from either `fn(a, b) -> r`, `(a, b) -> r` or a
/// parameter list plus return type assembled by the callers.
pub fn canon_sig(s: &str) -> Option<String> {
    let s = s.trim();
    let s = s.strip_prefix("fn").unwrap_or(s).trim();
    let open = s.find('(')?;
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in s.char_indices().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let params = split_top(&s[open + 1..close]);
    let ret = s[close + 1..].trim();
    let ret = ret.strip_prefix("->").map(str::trim).unwrap_or("()");
    Some(make_sig(&params, ret))
}

pub(crate) fn make_sig<S: AsRef<str>>(params: &[S], ret: &str) -> String {
    let ps: Vec<String> = params
        .iter()
        .map(|p| canon_type(p.as_ref()))
        .filter(|p| p != "()")
        .collect();
    format!("fn({}) -> {}", ps.join(", "), canon_type(ret))
}

/// Split on commas that are not inside `<>`, `()` or `[]`.
pub(crate) fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_types() {
        assert_eq!(canon_type("* const c_char"), "*const c_char");
        assert_eq!(canon_type("*mut *mut RxcHost"), "*mut *mut RxcHost");
        assert_eq!(canon_type("Option<RxcHostVerifyFn>"), "RxcHostVerifyFn");
        assert_eq!(canon_type("c_int"), "i32");
        assert_eq!(canon_type(""), "()");
        assert_eq!(
            canon_sig("fn(*mut c_void,u64, * mut u64)->c_int").unwrap(),
            "fn(*mut c_void, u64, *mut u64) -> i32"
        );
        assert_eq!(canon_sig("fn()").unwrap(), "fn() -> ()");
    }

    #[test]
    fn parse_round_trip_and_refusals() {
        let t = "abi x 2\nheader a.h\nstruct S 8 4\nfield S a 0 4 u32 1\nfield S b 4 4 i32 1\nconst K -3\nfn f fn(u32) -> i32\n";
        let m = Manifest::parse(t).unwrap();
        assert_eq!(m.fact_lines().len(), 5);
        assert_eq!(m.field_names("S"), ["a", "b"]);
        assert!(Manifest::parse("field T a 0 4 u32 1\n").is_err());
        assert!(Manifest::parse("struct S 8\n").is_err());
        assert!(Manifest::parse("struct S 8 4\nstruct S 8 4\n").is_err());
    }
}
