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

//! `closure.toml`: strict, line based component manifest, in the style of the
//! aien-test `.gate` format. Despite the file name it is not general TOML.
//!
//! ```text
//! closure-manifest v1
//! component aien-scheduler
//! profile host-v1
//! receipt <64 hex>
//! source blake3:<64 hex>
//! dep aien-kv-cache <64 hex receipt id of that dependency>
//! dep aien-platform
//! ```
//!
//! Rules: first line is the header; `component`, `profile`, `receipt` and
//! `source` appear exactly once; `dep` lines are unique per name; no tabs, no
//! CR, no blank lines, no comments, no trailing whitespace, no unknown keys. A
//! `dep` line without a receipt id parses (it is reported as
//! DEPENDENCY_NOT_PINNED by the verifier, not hidden here).

use std::collections::BTreeMap;

pub const FILE_NAME: &str = "closure.toml";
pub const HEADER: &str = "closure-manifest v1";
pub const MAX_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub component: String,
    pub profile: String,
    /// Receipt id (lowercase hex) that qualifies this component.
    pub receipt: String,
    /// `blake3:<hex>` digest of the component source the receipt was taken at.
    pub source: String,
    /// Declared internal deps: name -> pinned receipt id of that dependency.
    pub deps: BTreeMap<String, Option<String>>,
}

pub fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

pub fn parse(text: &str) -> Result<Manifest, String> {
    if text.len() > MAX_BYTES {
        return Err("manifest too large".into());
    }
    if text.starts_with('\u{feff}') {
        return Err("BOM not allowed".into());
    }
    if text.contains('\r') || text.contains('\t') {
        return Err("tabs and CR are not allowed".into());
    }
    let body = text
        .strip_suffix('\n')
        .ok_or("file must end with a single newline")?;
    let mut lines = body.split('\n').enumerate();
    match lines.next() {
        Some((_, l)) if l == HEADER => {}
        _ => return Err(format!("line 1: expected header {HEADER:?}")),
    }
    let mut component = None;
    let mut profile = None;
    let mut receipt = None;
    let mut source = None;
    let mut deps: BTreeMap<String, Option<String>> = BTreeMap::new();
    for (i, line) in lines {
        let n = i + 1;
        if line.is_empty() || line != line.trim() {
            return Err(format!("line {n}: blank line or stray whitespace"));
        }
        let parts: Vec<&str> = line.split(' ').collect();
        if parts.iter().any(|p| p.is_empty()) {
            return Err(format!("line {n}: repeated spaces"));
        }
        match (parts[0], parts.len()) {
            ("component", 2) | ("profile", 2) => {
                if !is_name(parts[1]) {
                    return Err(format!("line {n}: bad name {:?}", parts[1]));
                }
                let slot = if parts[0] == "component" {
                    &mut component
                } else {
                    &mut profile
                };
                if slot.replace(parts[1].to_string()).is_some() {
                    return Err(format!("line {n}: duplicate {}", parts[0]));
                }
            }
            ("receipt", 2) => {
                if !is_hex64(parts[1]) {
                    return Err(format!("line {n}: receipt must be 64 lowercase hex"));
                }
                if receipt.replace(parts[1].to_string()).is_some() {
                    return Err(format!("line {n}: duplicate receipt"));
                }
            }
            ("source", 2) => {
                let ok = parts[1].strip_prefix("blake3:").is_some_and(is_hex64);
                if !ok {
                    return Err(format!(
                        "line {n}: source must be blake3:<64 lowercase hex>"
                    ));
                }
                if source.replace(parts[1].to_string()).is_some() {
                    return Err(format!("line {n}: duplicate source"));
                }
            }
            ("dep", 2) | ("dep", 3) => {
                if !is_name(parts[1]) {
                    return Err(format!("line {n}: bad dep name {:?}", parts[1]));
                }
                let pin = match parts.get(2) {
                    Some(p) if is_hex64(p) => Some(p.to_string()),
                    Some(_) => return Err(format!("line {n}: dep pin must be 64 lowercase hex")),
                    None => None,
                };
                if deps.insert(parts[1].to_string(), pin).is_some() {
                    return Err(format!("line {n}: duplicate dep {}", parts[1]));
                }
            }
            (key, _) => return Err(format!("line {n}: unknown or malformed key {key:?}")),
        }
    }
    Ok(Manifest {
        component: component.ok_or("missing component")?,
        profile: profile.ok_or("missing profile")?,
        receipt: receipt.ok_or("missing receipt")?,
        source: source.ok_or("missing source")?,
        deps,
    })
}

pub fn render(m: &Manifest) -> String {
    let mut s = format!(
        "{HEADER}\ncomponent {}\nprofile {}\nreceipt {}\nsource {}\n",
        m.component, m.profile, m.receipt, m.source
    );
    for (name, pin) in &m.deps {
        match pin {
            Some(p) => s.push_str(&format!("dep {name} {p}\n")),
            None => s.push_str(&format!("dep {name}\n")),
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        let mut deps = BTreeMap::new();
        deps.insert("b".to_string(), Some("1".repeat(64)));
        deps.insert("c".to_string(), None);
        Manifest {
            component: "a".into(),
            profile: "host-v1".into(),
            receipt: "2".repeat(64),
            source: format!("blake3:{}", "3".repeat(64)),
            deps,
        }
    }

    #[test]
    fn round_trips() {
        let m = sample();
        assert_eq!(parse(&render(&m)).unwrap(), m);
    }

    #[test]
    fn rejects_malformed() {
        let good = render(&sample());
        for bad in [
            good.replace("closure-manifest v1", "closure-manifest v2"),
            good.replace('\n', "\r\n"),
            good.trim_end().to_string(),
            format!("{good}\n"),
            format!("{good}extra thing\n"),
            format!("{good}component again\n"),
            good.replace("profile host-v1\n", ""),
            good.replace(&"2".repeat(64), "XYZ"),
            format!("{good}dep b\n"),
        ] {
            assert!(parse(&bad).is_err(), "should reject:\n{bad}");
        }
    }
}
