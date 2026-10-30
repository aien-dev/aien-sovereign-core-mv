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

//! The entry function's public signature, as the task gives it (the `entry`, `params` and `ret`
//! lines the user message already contains), and the Omegese header line built from it.

/// Parameter names given to a signature that names none, in order (Omegese allows 6 parameters).
const PARAM_NAMES: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

/// The entry function: its name, its parameters (name, Omegese type word) and its return type
/// word (`None`: void).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub entry: String,
    pub params: Vec<(String, String)>,
    pub ret: Option<String>,
}

/// The IR type name the task writes, as the one-token Omegese word (OMEGESE.md section 3).
fn omegese_type(ir: &str) -> Option<&'static str> {
    Some(match ir {
        "bool" => "bool",
        "u8" => "uchar",
        "u16" => "ushort",
        "u32" => "uint",
        "u64" => "ulong",
        "i8" => "byte",
        "i16" => "short",
        "i32" => "int",
        "i64" => "long",
        _ => return None,
    })
}

impl Signature {
    /// From the task's signature lines: `entry <name>`, `params <ir type>*`, `ret <ir type>`
    /// (`ret` alone or `ret void`: void). Other lines are ignored. Parameters are named
    /// `a`, `b`, ... in order.
    pub fn from_requirement(text: &str) -> Result<Self, String> {
        let (mut entry, mut params, mut ret) = (None, None, None);
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("entry") => entry = Some(w.collect::<Vec<_>>()),
                Some("params") => params = Some(w.collect::<Vec<_>>()),
                Some("ret") => ret = Some(w.collect::<Vec<_>>()),
                _ => {}
            }
        }
        let entry = match entry.as_deref() {
            Some([name]) => name.to_string(),
            _ => return Err("the signature needs one `entry <name>` line".into()),
        };
        let params = params.ok_or("the signature has no `params` line")?;
        if params.len() > PARAM_NAMES.len() {
            return Err(format!("{} parameters; Omegese allows 6", params.len()));
        }
        let params = params
            .iter()
            .zip(PARAM_NAMES)
            .map(|(t, n)| {
                omegese_type(t)
                    .map(|t| (n.to_string(), t.to_string()))
                    .ok_or_else(|| format!("unknown parameter type {t:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ret = match ret.as_deref() {
            Some([]) | Some(["void"]) => None,
            Some([t]) => Some(
                omegese_type(t)
                    .ok_or_else(|| format!("unknown return type {t:?}"))?
                    .to_string(),
            ),
            _ => return Err("the signature needs one `ret <type>` line".into()),
        };
        if params.iter().any(|(n, _)| *n == entry) {
            return Err(format!(
                "the entry {entry:?} has the name of a parameter; name the parameters explicitly"
            ));
        }
        Ok(Self { entry, params, ret })
    }

    /// From a written header line, `fn <name> (<param> <type>)* to <type>|void` (an array type is
    /// `own|edit|view <type> <len>`), names and type words kept as written.
    pub fn from_header(line: &str) -> Result<Self, String> {
        let w: Vec<&str> = line.split_whitespace().collect();
        let bad = || format!("not a header line: {line:?}");
        let (Some(&"fn"), Some(entry)) = (w.first(), w.get(1)) else {
            return Err(bad());
        };
        let to = w.iter().position(|&x| x == "to").ok_or_else(bad)?;
        let (mut mid, tail) = (&w[2..to], &w[to + 1..]);
        if tail.len() != 1 {
            return Err(bad());
        }
        let mut params = Vec::new();
        while let Some((name, rest)) = mid.split_first() {
            // A type is one word, or `own|edit|view <type> <len>` for an array.
            let n = match rest.first() {
                Some(&"own" | &"edit" | &"view") => 3,
                Some(_) => 1,
                None => return Err(bad()),
            };
            if rest.len() < n {
                return Err(bad());
            }
            params.push((name.to_string(), rest[..n].join(" ")));
            mid = &rest[n..];
        }
        Ok(Self {
            entry: entry.to_string(),
            params,
            ret: (tail[0] != "void").then(|| tail[0].to_string()),
        })
    }

    /// `fn <entry> <name> <type> ... to <ret>` and a newline.
    pub fn header(&self) -> String {
        let mut s = format!("fn {}", self.entry);
        for (n, t) in &self.params {
            s.push(' ');
            s.push_str(n);
            s.push(' ');
            s.push_str(t);
        }
        s.push_str(" to ");
        s.push_str(self.ret.as_deref().unwrap_or("void"));
        s.push('\n');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirement_lines_make_the_header() {
        let s = Signature::from_requirement("entry clamp_sub\nparams u32 u32\nret u32\n").unwrap();
        assert_eq!(s.header(), "fn clamp_sub a uint b uint to uint\n");
        let s = Signature::from_requirement("entry go\nparams\nret\n").unwrap();
        assert_eq!(s.header(), "fn go to void\n");
        let s = Signature::from_requirement("x\nentry t\nparams i16 u8 bool\nret i64\n").unwrap();
        assert_eq!(s.header(), "fn t a short b uchar c bool to long\n");
        for bad in [
            "params u32\nret u32",
            "entry x\nret u32",
            "entry x\nparams u33\nret u32",
            "entry x\nparams u32\nret f32",
            "entry b\nparams u32 u32\nret u32",
            "entry x\nparams u8 u8 u8 u8 u8 u8 u8\nret u8",
        ] {
            assert!(Signature::from_requirement(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn header_lines_round_trip() {
        for h in [
            "fn gcd a uint b uint to uint\n",
            "fn dot u view long 3 w view long 3 to long\n",
            "fn go to void\n",
        ] {
            assert_eq!(Signature::from_header(h).unwrap().header(), h);
        }
        for bad in [
            "ret a",
            "fn f a to uint",
            "fn f a view uint to uint",
            "fn f a uint to",
        ] {
            assert!(Signature::from_header(bad).is_err(), "{bad}");
        }
    }
}
