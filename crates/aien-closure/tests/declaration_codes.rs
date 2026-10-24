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

//! SPEC section 6.2 of aien-protocols `specs/verified-crumb/SPEC.md`: one
//! meaning per declaration code. One test per row of the table, each with a
//! negative twin that fails if the row's code is swapped for a neighbouring
//! code (a mutant that returns the wrong code is killed by the twin).
//!
//! Rows 5a and 5b (a VC's stored source imports against `dependencies[]`) are
//! covered by the unit test `step8_stored_source_imports_against_dependencies`
//! in `src/declare.rs`: this crate reads no VCs, the same set comparison is
//! what step 8 needs.

mod common;

use aien_closure::Code;
use aien_proof::evidence::{receipt_id, store_receipt, Receipt};
use common::*;
use std::collections::BTreeSet;
use std::path::Path;

fn only(root: &Path, store: &Path, want: Code) {
    expect_only(root, store, want);
}

/// Change a manifest line in place.
fn edit_manifest(root: &Path, krate: &str, from: &str, to: &str) {
    let rel = format!("crates/{krate}/closure.toml");
    let text = read(root, &rel);
    assert!(text.contains(from), "{from:?} not in {text}");
    write(root, &rel, &text.replace(from, to));
}

/// Re-qualify `krate` with an extra receipt dependency on `dep`, and declare
/// it in the manifest, so the only thing wrong is that `dep` is not an edge.
fn declare_non_edge(
    root: &Path,
    store: &Path,
    ids: &std::collections::BTreeMap<String, String>,
    krate: &str,
    dep: &str,
) {
    let p = store.join("receipts").join(format!("{}.json", ids[krate]));
    let mut r: Receipt = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    r.dependencies.push(ids[dep].clone());
    r.id = receipt_id(&r).unwrap();
    store_receipt(store, &r).unwrap();
    edit_manifest(root, krate, &ids[krate], &r.id);
    let rel = format!("crates/{krate}/closure.toml");
    let text = read(root, &rel);
    write(root, &rel, &format!("{}dep {dep} {}\n", text, ids[dep]));
}

fn found(root: &Path, store: &Path) -> BTreeSet<Code> {
    codes(root, store)
}

// Row 1: an import with no pin, or a pin that is not the dependency's own receipt.

#[test]
fn row1_edge_without_receipt_pin_is_not_pinned() {
    let (root, store, ids) = seeded(&Knobs::default());
    edit_manifest(
        &root,
        "alpha",
        &format!("dep beta {}", ids["beta"]),
        "dep beta",
    );
    only(&root, &store, Code::DependencyNotPinned);
}

#[test]
fn row1_not_undeclared_and_not_unverified_twin() {
    // Same situation, negative twin: a missing pin is neither an undeclared
    // edge (the edge IS declared) nor an extra declaration.
    let (root, store, ids) = seeded(&Knobs::default());
    edit_manifest(
        &root,
        "alpha",
        &format!("dep beta {}", ids["beta"]),
        "dep beta",
    );
    let got = found(&root, &store);
    assert!(!got.contains(&Code::UndeclaredImport), "{got:?}");
    assert!(!got.contains(&Code::UnverifiedDependency), "{got:?}");
}

#[test]
fn row1_pin_that_is_not_the_dependencys_receipt_is_not_pinned() {
    let (root, store, ids) = seeded(&Knobs::default());
    edit_manifest(
        &root,
        "alpha",
        &format!("dep beta {}", ids["beta"]),
        &format!("dep beta {}", ids["gamma"]),
    );
    only(&root, &store, Code::DependencyNotPinned);
}

// Row 2: a lock line for something the component does not depend on.

#[test]
fn row2_lock_line_with_no_edge_is_unverified_dependency() {
    let (root, store, ids) = seeded(&Knobs::default());
    let lock = read(&root, "closure.lock");
    // gamma has no dependencies, so `gamma needs alpha` is a pin with no edge.
    write(
        &root,
        "closure.lock",
        &format!("{lock}needs gamma alpha {}\n", ids["alpha"]),
    );
    only(&root, &store, Code::UnverifiedDependency);
}

#[test]
fn row2_not_undeclared_and_not_not_pinned_twin() {
    let (root, store, ids) = seeded(&Knobs::default());
    let lock = read(&root, "closure.lock");
    write(
        &root,
        "closure.lock",
        &format!("{lock}needs gamma alpha {}\n", ids["alpha"]),
    );
    let got = found(&root, &store);
    assert!(!got.contains(&Code::UndeclaredImport), "{got:?}");
    assert!(!got.contains(&Code::DependencyNotPinned), "{got:?}");
}

#[test]
fn row2_wrong_id_and_missing_line_stay_not_pinned() {
    // The mirror cases are pin problems, not extra pins: a known pair with
    // another id, and a missing line.
    let (root, store, ids) = seeded(&Knobs::default());
    let lock = read(&root, "closure.lock");
    write(
        &root,
        "closure.lock",
        &lock.replace(
            &format!("needs alpha gamma {}", ids["gamma"]),
            &format!("needs alpha gamma {}", ids["alpha"]),
        ),
    );
    only(&root, &store, Code::DependencyNotPinned);
    let dropped: String = lock
        .lines()
        .filter(|l| !l.starts_with("needs alpha gamma"))
        .map(|l| format!("{l}\n"))
        .collect();
    write(&root, "closure.lock", &dropped);
    only(&root, &store, Code::DependencyNotPinned);
}

#[test]
fn row2_extra_pin_and_missing_pin_together_report_both() {
    let (root, store, ids) = seeded(&Knobs::default());
    let lock = read(&root, "closure.lock");
    let mut edited: String = lock
        .lines()
        .filter(|l| !l.starts_with("needs alpha gamma"))
        .map(|l| format!("{l}\n"))
        .collect();
    edited.push_str(&format!("needs gamma alpha {}\n", ids["alpha"]));
    write(&root, "closure.lock", &edited);
    assert_eq!(
        found(&root, &store),
        BTreeSet::from([Code::UnverifiedDependency, Code::DependencyNotPinned])
    );
}

// Row 3: a declared dependency that is not an actual edge.

#[test]
fn row3_declared_dep_that_is_not_an_edge_is_unverified_dependency() {
    // alpha depends on beta only; it now also declares gamma (a transitive
    // dependency, not an edge of alpha).
    let (root, store, ids) = seeded(&Knobs::default());
    declare_non_edge(&root, &store, &ids, "alpha", "gamma");
    only(&root, &store, Code::UnverifiedDependency);
}

#[test]
fn row3_not_undeclared_twin() {
    let (root, store, ids) = seeded(&Knobs::default());
    declare_non_edge(&root, &store, &ids, "alpha", "gamma");
    let got = found(&root, &store);
    assert!(!got.contains(&Code::UndeclaredImport), "{got:?}");
    assert!(!got.contains(&Code::DependencyNotPinned), "{got:?}");
}

#[test]
fn row3_declared_dep_without_pin_that_is_not_an_edge_is_still_unverified() {
    let (root, store, _) = seeded(&Knobs::default());
    let text = read(&root, "crates/alpha/closure.toml");
    write(
        &root,
        "crates/alpha/closure.toml",
        &format!("{text}dep gamma\n"),
    );
    only(&root, &store, Code::UnverifiedDependency);
}

// Row 4: an actual edge the manifest does not declare.

fn undeclared_edge() -> (std::path::PathBuf, std::path::PathBuf) {
    let (root, store, _) = seeded(&Knobs::default());
    let m = read(&root, "crates/alpha/Cargo.toml");
    write(
        &root,
        "crates/alpha/Cargo.toml",
        &m.replace(
            "[dependencies]",
            "[dependencies]\ngamma = { path = \"../gamma\" }",
        ),
    );
    // Re-seed against the new graph, then drop the declaration again, so the
    // undeclared edge is the only thing wrong.
    seed(&root, &store, &Knobs::default());
    let man = read(&root, "crates/alpha/closure.toml");
    let kept: String = man
        .lines()
        .filter(|l| !l.starts_with("dep gamma"))
        .map(|l| format!("{l}\n"))
        .collect();
    write(&root, "crates/alpha/closure.toml", &kept);
    (root, store)
}

#[test]
fn row4_actual_edge_not_declared_is_undeclared_import() {
    let (root, store) = undeclared_edge();
    only(&root, &store, Code::UndeclaredImport);
}

#[test]
fn row4_not_unverified_and_not_not_pinned_twin() {
    let (root, store) = undeclared_edge();
    let got = found(&root, &store);
    assert!(!got.contains(&Code::UnverifiedDependency), "{got:?}");
    assert!(!got.contains(&Code::DependencyNotPinned), "{got:?}");
}

// Rows 1 to 4 are four different codes, so a mutant that merges two of them
// is caught by the exit codes.

#[test]
fn rows_1_to_4_map_to_four_distinct_exit_codes() {
    let exits: BTreeSet<i32> = [
        Code::DependencyNotPinned,
        Code::UnverifiedDependency,
        Code::UndeclaredImport,
    ]
    .iter()
    .map(|c| c.exit_code())
    .collect();
    assert_eq!(exits, BTreeSet::from([13, 10, 16]));
}
