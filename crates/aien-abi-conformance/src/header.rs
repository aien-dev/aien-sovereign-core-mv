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

//! Just enough C header reading for omega's plain ABI headers: comments and preprocessor
//! lines removed, then top-level declarations split on `;`. It reads
//! `typedef struct [Tag] { ... } Name;` (field names in order), prototypes and
//! `typedef R (*Name)(...);` callback types. Types come back in canonical Rust spelling.

use crate::manifest::{make_sig, split_top};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Header {
    /// struct typedef name -> field names in declaration order
    pub structs: BTreeMap<String, Vec<String>>,
    /// function name -> canonical signature
    pub fns: BTreeMap<String, String>,
    /// callback typedef name -> canonical signature
    pub fnptrs: BTreeMap<String, String>,
}

/// Remove comments, string-free preprocessor lines (with `\` continuations) and
/// `extern "C" {` / `}` wrappers.
pub fn strip(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            out.push(' ');
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    let mut kept = String::new();
    let mut cont = false;
    for line in out.lines() {
        let t = line.trim();
        if cont || t.starts_with('#') {
            cont = t.ends_with('\\');
            continue;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    kept.replace("extern \"C\" {", " ")
}

/// Split at top-level `;` (braces tracked) into whitespace-normalised statements.
fn statements(src: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in src.chars() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    // the closing brace of an extern "C" block
                    depth = 0;
                    continue;
                }
            }
            ';' if depth == 0 => {
                let s = cur.split_whitespace().collect::<Vec<_>>().join(" ");
                if !s.is_empty() {
                    v.push(s);
                }
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    v
}

fn tokens(s: &str) -> Vec<String> {
    let mut t = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                t.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                t.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        t.push(cur);
    }
    t
}

const BUILTIN: &[&str] = &[
    "void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "_Bool",
    "bool",
];

/// One C declaration (parameter, field or return type) -> (canonical Rust type, name).
pub fn c_decl(decl: &str) -> (String, Option<String>) {
    let toks = tokens(decl);
    let mut base_const = false;
    let mut words: Vec<String> = Vec::new();
    let mut i = 0;
    while i < toks.len() && toks[i] != "*" && toks[i] != "[" {
        match toks[i].as_str() {
            "const" => base_const = true,
            "volatile" | "struct" | "enum" | "restrict" | "__restrict" => {}
            w => words.push(w.to_string()),
        }
        i += 1;
    }
    // Pointer levels: each `*` may be followed by `const` (the pointer itself, ABI-irrelevant).
    let mut stars = 0;
    let mut name = None;
    while i < toks.len() {
        match toks[i].as_str() {
            "*" => stars += 1,
            "const" | "restrict" | "__restrict" => {}
            "[" => {
                // array parameter: decays to a pointer
                stars += 1;
                while i < toks.len() && toks[i] != "]" {
                    i += 1;
                }
            }
            w if w.chars().all(|c| c.is_alphanumeric() || c == '_') => name = Some(w.to_string()),
            _ => {}
        }
        i += 1;
    }
    let builtin: Vec<&str> = words
        .iter()
        .map(String::as_str)
        .filter(|w| BUILTIN.contains(w))
        .collect();
    let (base, rest_name) = if builtin.is_empty() {
        (
            words.first().cloned().unwrap_or_default(),
            words.get(1).cloned(),
        )
    } else {
        let others: Vec<&String> = words
            .iter()
            .filter(|w| !BUILTIN.contains(&w.as_str()))
            .collect();
        (builtin.join(" "), others.first().map(|s| s.to_string()))
    };
    let name = name.or(rest_name);
    let rust_base = match base.as_str() {
        "int" | "signed" | "signed int" | "int32_t" => "i32",
        "unsigned" | "unsigned int" | "uint32_t" => "u32",
        "long" | "long int" | "long long" | "int64_t" => "i64",
        "unsigned long" | "unsigned long long" | "uint64_t" => "u64",
        "short" | "int16_t" => "i16",
        "unsigned short" | "uint16_t" => "u16",
        "signed char" | "int8_t" => "i8",
        "unsigned char" | "uint8_t" => "u8",
        "char" => "c_char",
        "size_t" => "usize",
        "ssize_t" | "ptrdiff_t" => "isize",
        "float" => "f32",
        "double" => "f64",
        "bool" | "_Bool" => "bool",
        "void" if stars > 0 => "c_void",
        "void" => "()",
        other => other,
    }
    .to_string();
    let mut ty = String::new();
    for level in 0..stars {
        // The innermost pointer points at the (possibly const) base; outer ones point at
        // pointers whose own constness the C declaration placed after their `*`.
        let innermost = level == stars - 1;
        ty.push_str(if innermost && base_const {
            "*const "
        } else {
            "*mut "
        });
    }
    ty.push_str(&rust_base);
    (ty, name)
}

fn params(list: &str) -> Vec<String> {
    let l = list.trim();
    if l.is_empty() || l == "void" {
        return Vec::new();
    }
    split_top(l).iter().map(|p| c_decl(p).0).collect()
}

/// Parse the declarations of one header.
pub fn parse(src: &str) -> Header {
    let mut h = Header::default();
    for st in statements(&strip(src)) {
        if let Some(rest) = st.strip_prefix("typedef ") {
            if rest.starts_with("struct") && rest.contains('{') {
                let (Some(open), Some(close)) = (rest.find('{'), rest.rfind('}')) else {
                    continue;
                };
                let name = rest[close + 1..].trim().to_string();
                let mut fields = Vec::new();
                for decl in rest[open + 1..close].split(';') {
                    let d = decl.trim();
                    if d.is_empty() {
                        continue;
                    }
                    let parts = split_top(d);
                    // first declarator carries the base type; the rest are bare declarators
                    if let Some((_, Some(n))) = parts.first().map(|p| c_decl(p)) {
                        fields.push(n);
                    }
                    for p in parts.iter().skip(1) {
                        let t = tokens(p);
                        if let Some(n) = t
                            .iter()
                            .find(|w| w.chars().all(|c| c.is_alphanumeric() || c == '_'))
                        {
                            fields.push(n.clone());
                        }
                    }
                }
                h.structs.insert(name, fields);
            } else if let Some(p) = rest.find("(*") {
                // typedef R (*Name)(params)
                let ret = c_decl(&rest[..p]).0;
                let after = &rest[p + 2..];
                let Some(close) = after.find(')') else {
                    continue;
                };
                let name = after[..close].trim().to_string();
                let plist = after[close + 1..].trim();
                if let Some(inner) = plist.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                    h.fnptrs.insert(name, make_sig(&params(inner), &ret));
                }
            }
            continue;
        }
        // prototype: R name(params)
        let Some(open) = st.find('(') else { continue };
        if !st.ends_with(')') || st.contains('{') || st.contains('=') {
            continue;
        }
        let (ret, name) = c_decl(&st[..open]);
        let Some(name) = name else { continue };
        let inner = &st[open + 1..st.len() - 1];
        h.fns.insert(name, make_sig(&params(inner), &ret));
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations() {
        assert_eq!(c_decl("const char *dir").0, "*const c_char");
        assert_eq!(c_decl("RxcHost **out").0, "*mut *mut RxcHost");
        assert_eq!(c_decl("const uint64_t links[4]").0, "*const u64");
        assert_eq!(c_decl("uint32_t out[4]").0, "*mut u32");
        assert_eq!(c_decl("size_t").0, "usize");
        assert_eq!(c_decl("unsigned int x").0, "u32");
        assert_eq!(c_decl("const OmegaGpuTensor *t").0, "*const OmegaGpuTensor");
        assert_eq!(c_decl("void").0, "()");
        assert_eq!(c_decl("int").1, None);
        assert_eq!(c_decl("uint32_t m").1.as_deref(), Some("m"));
    }

    #[test]
    fn header_parts() {
        let src = r#"
/* comment with ; inside */
#ifndef X
#define X 1u   /* abi */
#define LONG(a) \
    ((a) + 1)
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Foo Foo;
typedef int (*CbFn)(void *ctx, uint64_t task, uint64_t *result);
typedef struct {
    uint32_t a, b; // two
    char name[64];
    bool ok;
} Info;
int open_it(const char *dir, uint32_t n, Foo **out, Info *info);
void close_it(Foo *h);
uint32_t layout(uint32_t out[4]);
int set_cb(Foo *h, CbFn fn, void *ctx);
const char *rc_name(int rc);
#ifdef __cplusplus
}
#endif
#endif
"#;
        let h = parse(src);
        assert_eq!(h.structs["Info"], ["a", "b", "name", "ok"]);
        assert_eq!(
            h.fns["open_it"],
            "fn(*const c_char, u32, *mut *mut Foo, *mut Info) -> i32"
        );
        assert_eq!(h.fns["close_it"], "fn(*mut Foo) -> ()");
        assert_eq!(h.fns["layout"], "fn(*mut u32) -> u32");
        assert_eq!(h.fns["set_cb"], "fn(*mut Foo, CbFn, *mut c_void) -> i32");
        assert_eq!(h.fns["rc_name"], "fn(i32) -> *const c_char");
        assert_eq!(h.fnptrs["CbFn"], "fn(*mut c_void, u64, *mut u64) -> i32");
    }
}
