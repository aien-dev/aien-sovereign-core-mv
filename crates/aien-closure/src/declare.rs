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

//! Declaration checks: what a unit says it depends on against what it uses.
//!
//! One meaning per code, from `aien-protocols` `specs/verified-crumb/SPEC.md`
//! section 6.2:
//!
//! | Situation | Code |
//! |---|---|
//! | actual import or edge not declared in the dependency list | `UNDECLARED_IMPORT` |
//! | declared dependency that is not an actual import or edge | `UNVERIFIED_DEPENDENCY` |
//! | lock line for something the program never imports | `UNVERIFIED_DEPENDENCY` |
//! | import, edge or manifest entry without a (correct) lock pin | `DEPENDENCY_NOT_PINNED` |
//!
//! The first two rows are one set comparison, [`compare`]. The same function
//! serves SPEC step 8 (a VC's stored source imports against its
//! `dependencies[]`), which this crate does not run today because it reads no
//! VCs. The lock row is [`extra_needs`].

use crate::error::Code;
use std::collections::BTreeSet;

/// One disagreement between the actual set and the declared set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disagreement {
    pub code: Code,
    pub name: String,
}

/// Compare what a program actually uses with what its dependency list
/// declares. Actual but undeclared is `UNDECLARED_IMPORT` (the program does
/// more than it says). Declared but not actual is `UNVERIFIED_DEPENDENCY`
/// (nothing shows the declaration is an edge). Sorted by code then name.
pub fn compare(actual: &BTreeSet<String>, declared: &BTreeSet<String>) -> Vec<Disagreement> {
    let mut out: Vec<Disagreement> = actual
        .difference(declared)
        .map(|n| Disagreement {
            code: Code::UndeclaredImport,
            name: n.clone(),
        })
        .collect();
    out.extend(declared.difference(actual).map(|n| Disagreement {
        code: Code::UnverifiedDependency,
        name: n.clone(),
    }));
    out
}

fn needs_pair(line: &str) -> Option<(&str, &str)> {
    let mut it = line.split(' ');
    match (it.next(), it.next(), it.next()) {
        (Some("needs"), Some(c), Some(d)) => Some((c, d)),
        _ => None,
    }
}

/// `needs <component> <dep> <id>` lines of the lock on disk whose
/// (component, dep) pair is not in the rebuilt lock: pins for something the
/// component does not depend on. A line with a known pair but another id is
/// not extra; it is a wrong pin and stays `DEPENDENCY_NOT_PINNED`.
pub fn extra_needs(have: &str, want: &str) -> Vec<(String, String)> {
    let wanted: BTreeSet<(&str, &str)> = want.lines().filter_map(needs_pair).collect();
    let mut out: Vec<(String, String)> = have
        .lines()
        .filter_map(needs_pair)
        .filter(|p| !wanted.contains(p))
        .map(|(c, d)| (c.to_string(), d.to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The lock text with every extra `needs` line removed, so what remains can
/// be compared with the rebuilt lock for the pin rows.
pub fn without_extra_needs(have: &str, want: &str) -> String {
    let wanted: BTreeSet<(&str, &str)> = want.lines().filter_map(needs_pair).collect();
    let mut s = String::new();
    for line in have.lines() {
        if needs_pair(line).is_some_and(|p| !wanted.contains(&p)) {
            continue;
        }
        s.push_str(line);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn codes(d: &[Disagreement]) -> Vec<(Code, &str)> {
        d.iter().map(|x| (x.code, x.name.as_str())).collect()
    }

    #[test]
    fn agreement_has_no_findings() {
        assert!(compare(&set(&["a", "b"]), &set(&["a", "b"])).is_empty());
        assert!(compare(&set(&[]), &set(&[])).is_empty());
    }

    #[test]
    fn row_actual_not_declared_is_undeclared_import() {
        let d = compare(&set(&["a", "b"]), &set(&["a"]));
        assert_eq!(codes(&d), vec![(Code::UndeclaredImport, "b")]);
        // Negative: it is not the extra-declaration code.
        assert!(d.iter().all(|x| x.code != Code::UnverifiedDependency));
    }

    #[test]
    fn row_declared_not_actual_is_unverified_dependency() {
        let d = compare(&set(&["a"]), &set(&["a", "z"]));
        assert_eq!(codes(&d), vec![(Code::UnverifiedDependency, "z")]);
        assert!(d.iter().all(|x| x.code != Code::UndeclaredImport));
    }

    #[test]
    fn step8_stored_source_imports_against_dependencies() {
        // SPEC step 8: the VC's stored source imports are `actual`, the
        // `dependencies[]` ids are `declared`. Both directions at once.
        let d = compare(&set(&["id1", "id2"]), &set(&["id2", "id3"]));
        assert_eq!(
            codes(&d),
            vec![
                (Code::UndeclaredImport, "id1"),
                (Code::UnverifiedDependency, "id3")
            ]
        );
    }

    #[test]
    fn extra_needs_finds_only_unknown_pairs() {
        let want = "closure-lock v1\ncomponent a r s\nneeds a b ID1\n";
        let same = want.to_string();
        assert!(extra_needs(&same, want).is_empty());
        // Known pair, other id: a wrong pin, not an extra one.
        let wrong = "closure-lock v1\ncomponent a r s\nneeds a b OTHER\n";
        assert!(extra_needs(wrong, want).is_empty());
        // Unknown pair: extra.
        let extra = format!("{want}needs a z ID9\n");
        assert_eq!(extra_needs(&extra, want), vec![("a".into(), "z".into())]);
        assert_eq!(without_extra_needs(&extra, want), want);
        // Missing line is not extra.
        assert!(extra_needs("closure-lock v1\n", want).is_empty());
    }
}
