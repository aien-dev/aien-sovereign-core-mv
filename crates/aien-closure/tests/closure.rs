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

mod common;

use aien_closure::error::ALL;
use aien_closure::Code;
use common::*;
use std::collections::BTreeSet;
use std::process::Command;

#[test]
fn pass_case() {
    let (root, store, ids) = seeded(&Knobs::default());
    let out = run(&root, &store);
    assert!(out.ok(), "{:?}", out.findings);
    assert_eq!(out.components, 3);
    assert_eq!(out.passed.len(), 3);
    let (exit, stdout) = bin(&root, &store);
    assert_eq!(exit, 0, "{stdout}");
    assert!(stdout.contains("SUMMARY status=PASS findings=0 components=3"));
    assert!(stdout.contains(&format!("PASS component=alpha receipt={}", ids["alpha"])));
    // closure.lock lists the transitive closure with ids.
    let lock = read(&root, "closure.lock");
    assert!(
        lock.contains(&format!("needs alpha gamma {}", ids["gamma"])),
        "{lock}"
    );
}

#[test]
fn committed_seeded_fixture_passes() {
    let root = fixture_src();
    let out = run(&root, &root.join("store"));
    assert!(out.ok(), "{:?}", out.findings);
    assert_eq!(out.components, 3);
}

/// `cargo test -- --ignored regenerate_committed_fixture` rewrites the
/// committed manifests, lock and store. Deterministic: same sources, same bytes.
#[test]
#[ignore]
fn regenerate_committed_fixture() {
    let root = fixture_src();
    let store = root.join("store");
    let _ = std::fs::remove_dir_all(&store);
    seed(&root, &store, &Knobs::default());
}

#[test]
fn unverified_dependency() {
    // beta depends on gamma, but gamma carries no manifest.
    let (root, store, _) = seeded(&Knobs::default());
    std::fs::remove_file(root.join("crates/gamma/closure.toml")).unwrap();
    expect_only(&root, &store, Code::UnverifiedDependency);
}

#[test]
fn missing_receipt() {
    let (root, store, ids) = seeded(&Knobs::default());
    std::fs::remove_file(
        store
            .join("receipts")
            .join(format!("{}.json", ids["alpha"])),
    )
    .unwrap();
    expect_only(&root, &store, Code::MissingReceipt);
}

#[test]
fn receipt_hash_mismatch() {
    // Same file name, different bytes: the content no longer hashes to its id.
    let (root, store, ids) = seeded(&Knobs::default());
    let p = store
        .join("receipts")
        .join(format!("{}.json", ids["gamma"]));
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("1786000000"));
    std::fs::write(&p, text.replace("1786000000", "1786000001")).unwrap();
    let got = codes(&root, &store);
    assert!(got.contains(&Code::ReceiptHashMismatch), "{got:?}");
    assert_eq!(got, BTreeSet::from([Code::ReceiptHashMismatch]));
    assert_eq!(bin(&root, &store).0, Code::ReceiptHashMismatch.exit_code());
}

#[test]
fn dependency_not_pinned() {
    let (root, store, ids) = seeded(&Knobs::default());
    let m = read(&root, "crates/alpha/closure.toml");
    write(
        &root,
        "crates/alpha/closure.toml",
        &m.replace(&format!("dep beta {}", ids["beta"]), "dep beta"),
    );
    expect_only(&root, &store, Code::DependencyNotPinned);
}

#[test]
fn dependency_not_pinned_when_lock_is_stale() {
    let (root, store, _) = seeded(&Knobs::default());
    let lock = read(&root, "closure.lock");
    write(
        &root,
        "closure.lock",
        &lock.replace("needs alpha gamma", "needs alpha beta"),
    );
    expect_only(&root, &store, Code::DependencyNotPinned);
    std::fs::remove_file(root.join("closure.lock")).unwrap();
    expect_only(&root, &store, Code::DependencyNotPinned);
}

#[test]
fn dependency_cycle() {
    // gamma now depends on alpha: alpha -> beta -> gamma -> alpha.
    let (root, store, _) = seeded(&Knobs::default());
    let m = read(&root, "crates/gamma/Cargo.toml");
    write(
        &root,
        "crates/gamma/Cargo.toml",
        &m.replace(
            "[dependencies]",
            "[dependencies]\nalpha = { path = \"../alpha\" }",
        ),
    );
    let got = codes(&root, &store);
    assert!(got.contains(&Code::DependencyCycle), "{got:?}");
    assert_eq!(bin(&root, &store).0, Code::DependencyCycle.exit_code());
}

#[test]
fn stale_receipt() {
    // Source changes after the receipt was taken.
    let (root, store, _) = seeded(&Knobs::default());
    write(
        &root,
        "crates/gamma/src/lib.rs",
        "pub fn gamma() -> u32 { 4 }\n",
    );
    let got = codes(&root, &store);
    assert_eq!(got, BTreeSet::from([Code::StaleReceipt]));
    assert_eq!(bin(&root, &store).0, Code::StaleReceipt.exit_code());
}

#[test]
fn stale_receipt_when_receipt_records_other_source() {
    // The manifest digest is current but the receipt was taken at a different one.
    let (root, store, ids) = seeded(&Knobs::default());
    let p = store
        .join("receipts")
        .join(format!("{}.json", ids["gamma"]));
    let text = std::fs::read_to_string(&p).unwrap();
    let r: aien_proof::evidence::Receipt = serde_json::from_str(&text).unwrap();
    let mut other = r.clone();
    other.input_artifacts = vec![format!("blake3:{}", "d".repeat(64))];
    other.id = aien_proof::evidence::receipt_id(&other).unwrap();
    aien_proof::evidence::store_receipt(&store, &other).unwrap();
    let m = read(&root, "crates/gamma/closure.toml");
    write(
        &root,
        "crates/gamma/closure.toml",
        &m.replace(&ids["gamma"], &other.id),
    );
    let got = codes(&root, &store);
    assert!(got.contains(&Code::StaleReceipt), "{got:?}");
}

#[test]
fn undeclared_import() {
    // alpha gains a direct edge to gamma that its manifest does not declare.
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
    expect_only(&root, &store, Code::UndeclaredImport);
}

#[test]
fn tainted_artifact_dirty() {
    let knobs = Knobs {
        gamma_dirty: true,
        ..Knobs::default()
    };
    let (root, store, _) = seeded(&knobs);
    expect_only(&root, &store, Code::TaintedArtifact);
}

#[test]
fn tainted_artifact_test_only_trust() {
    let knobs = Knobs {
        gamma_tier: aien_proof::evidence::Tier::TestOnlyTrust,
        ..Knobs::default()
    };
    let (root, store, _) = seeded(&knobs);
    let got = codes(&root, &store);
    assert!(got.contains(&Code::TaintedArtifact), "{got:?}");
}

#[test]
fn unknown_verifier_profile() {
    let knobs = Knobs {
        alpha_profile: "bogus-v9",
        ..Knobs::default()
    };
    let (root, store, _) = seeded(&knobs);
    expect_only(&root, &store, Code::UnknownVerifierProfile);
}

#[test]
fn nine_codes_have_distinct_exit_codes_and_names() {
    let exits: BTreeSet<i32> = ALL.iter().map(|c| c.exit_code()).collect();
    let names: BTreeSet<&str> = ALL.iter().map(|c| c.as_str()).collect();
    assert_eq!(ALL.len(), 9);
    assert_eq!(exits.len(), 9);
    assert_eq!(names.len(), 9);
    assert!(exits.iter().all(|e| (10..=18).contains(e)));
}

#[test]
fn no_manifests_fails_closed() {
    let root = temp_dir("bare");
    copy_sources(&fixture_src(), &root);
    let got = codes(&root, &root.join("store"));
    assert_eq!(got, BTreeSet::from([Code::UnverifiedDependency]));
    assert_eq!(
        bin(&root, &root.join("store")).0,
        Code::UnverifiedDependency.exit_code()
    );
}

#[test]
fn usage_error_is_exit_2() {
    let o = Command::new(env!("CARGO_BIN_EXE_verify-closure"))
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}
