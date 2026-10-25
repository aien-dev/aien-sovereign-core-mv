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

//! Every `*.gate` manifest in the repository must parse under the strict
//! Slice A grammar (ADR 0028 Decision 3), and gate identifiers must be unique
//! (Decision 2, rule D2).
//!
//! `scan` finds gates the way the runner does (`aien_test::runner::discover`,
//! rule D1) and parses each one with `aien_test::manifest::parse`. The first
//! test runs it over this repository. The other tests run it over small
//! throwaway trees to show that a mistyped key, a duplicate gate id or a walk
//! that finds nothing cannot pass unnoticed. That last part is what the
//! mutants `discover_ignores_gate_files`, `discover_not_recursive` and
//! `unknown_child_key_allowed` in `mutations/aien-test.mutants` attack.

use aien_test::manifest::{self, Manifest};
use aien_test::runner::discover;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Scan {
    parsed: Vec<(String, Manifest)>,
    errors: Vec<String>,
}

/// Parse every gate file under `root`. Each unparsable file and each gate id
/// that appears in more than one file adds an entry to `errors`; paths are
/// shown relative to `root`.
fn scan(root: &Path) -> Scan {
    let mut parsed: Vec<(String, Manifest)> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for path in discover(root) {
        let shown = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        match fs::read(path) {
            Err(e) => errors.push(format!("{shown}: cannot read: {e}")),
            Ok(bytes) => match manifest::parse(&bytes) {
                Ok(m) => parsed.push((shown, m)),
                Err(e) => errors.push(format!("{shown}: BAD_MANIFEST {e}")),
            },
        }
    }
    let mut by_id: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (shown, m) in &parsed {
        by_id
            .entry(m.gate.as_str())
            .or_default()
            .push(shown.as_str());
    }
    for (id, files) in &by_id {
        if files.len() > 1 {
            let list = files.join(", ");
            errors.push(format!("duplicate gate id {id} in: {list}"));
        }
    }
    Scan { parsed, errors }
}

/// `tools/aien-test` sits two levels below the repository root.
fn repo_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::canonicalize(root).expect("repository root")
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A throwaway directory tree, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("aien-test-gates-{tag}-{pid}-{n}"));
        fs::create_dir_all(&path).expect("create temp dir");
        TempDir(fs::canonicalize(path).expect("canonicalize temp dir"))
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().expect("parent dir")).expect("create parent dir");
        fs::write(p, content).expect("write fixture");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The smallest valid manifest, with the gate id `FIXTURE-A` replaced.
fn fixture(gate: &str) -> String {
    concat!(
        "gate: FIXTURE-A\n",
        "manifest_version: 1\n",
        "owner: fixture\n",
        "requires:\n",
        "  - host\n",
        "run:\n",
        "  exec: x.sh\n",
        "expects:\n",
        "  verdict: PASS\n",
        "timeout: 5s\n",
    )
    .replace("FIXTURE-A", gate)
}

#[test]
fn every_committed_gate_parses() {
    let root = repo_root();
    let s = scan(&root);
    let problems = s.errors.join("\n");
    assert!(
        s.errors.is_empty(),
        "gate manifests that do not parse or collide:\n{problems}"
    );
    // A real checkout (it has crates/) carries gates; finding none means the
    // walk is broken, not that everything is fine.
    if root.join("crates").is_dir() {
        let shown = root.display();
        assert!(
            !s.parsed.is_empty(),
            "no .gate file found under {shown}; the discovery walk found nothing"
        );
    }
}

#[test]
fn a_typo_in_a_top_level_key_is_reported() {
    let t = TempDir::new("typo");
    t.write("tests/a/good.gate", &fixture("FIXTURE-GOOD"));
    t.write(
        "tests/b/typo.gate",
        &fixture("FIXTURE-TYPO").replace("manifest_version", "manifest_versoin"),
    );
    let s = scan(t.path());
    let errors = format!("{:?}", s.errors);
    assert_eq!(s.parsed.len(), 1, "the good gate must be found: {errors}");
    assert_eq!(s.errors.len(), 1, "exactly the typo gate fails: {errors}");
    assert!(s.errors[0].contains("tests/b/typo.gate"), "{errors}");
    assert!(
        s.errors[0].contains("unknown key manifest_versoin"),
        "{errors}"
    );
}

#[test]
fn a_typo_in_a_child_key_is_reported() {
    let t = TempDir::new("child");
    t.write(
        "tests/a/typo.gate",
        &fixture("FIXTURE-CHILD").replace("  exec: x.sh", "  exce: x.sh"),
    );
    let s = scan(t.path());
    let errors = format!("{:?}", s.errors);
    assert!(s.parsed.is_empty(), "{errors}");
    assert_eq!(s.errors.len(), 1, "{errors}");
    assert!(
        s.errors[0].contains("unknown child key exce under run"),
        "{errors}"
    );
}

#[test]
fn a_duplicate_gate_id_is_reported_for_both_files() {
    let t = TempDir::new("dup");
    t.write("tests/a/one.gate", &fixture("FIXTURE-DUP"));
    t.write("tests/b/two.gate", &fixture("FIXTURE-DUP"));
    let s = scan(t.path());
    let errors = format!("{:?}", s.errors);
    assert_eq!(s.errors.len(), 1, "{errors}");
    assert!(
        s.errors[0].contains("duplicate gate id FIXTURE-DUP"),
        "{errors}"
    );
    assert!(s.errors[0].contains("tests/a/one.gate"), "{errors}");
    assert!(s.errors[0].contains("tests/b/two.gate"), "{errors}");
}

#[test]
fn build_and_ignored_directories_are_not_scanned() {
    let t = TempDir::new("skip");
    t.write("tests/a/good.gate", &fixture("FIXTURE-GOOD"));
    t.write("target/skipped.gate", "this is not a manifest\n");
    t.write("vendored/.aien-test-ignore", "");
    t.write("vendored/bad.gate", "this is not a manifest\n");
    let s = scan(t.path());
    let errors = format!("{:?}", s.errors);
    assert!(s.errors.is_empty(), "{errors}");
    assert_eq!(s.parsed.len(), 1, "{errors}");
    assert_eq!(s.parsed[0].1.gate, "FIXTURE-GOOD");
}
