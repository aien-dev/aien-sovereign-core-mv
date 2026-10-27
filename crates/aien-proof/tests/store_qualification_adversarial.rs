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

use aien_proof::chain::verify_chain;
use aien_proof::evidence::{
    receipt_id, store_receipt, verify_bytes, verify_with_store, Assertion, LedgerRef, Mutation,
    Receipt, Tier, Verdict, SCHEMA, SCHEMA_VERSION,
};
use aien_proof::gate::{evaluate, AssertionSource, Gate, MissingAs, Requirement};
use aien_proof::ledger;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

fn temp_store(tag: &str) -> PathBuf {
    loop {
        let n = N.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("aien-proof-adv-{tag}-{}-{n}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Ok(()) => return dir,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("cannot create temp store: {e}"),
        }
    }
}

fn pass_assertion(id: &str, source: &str) -> Assertion {
    Assertion {
        id: id.to_string(),
        expected: "yes".to_string(),
        observed: "yes".to_string(),
        pass: true,
        source: source.to_string(),
        note: String::new(),
    }
}

fn sample_receipt() -> Receipt {
    Receipt {
        schema: SCHEMA.to_string(),
        version: SCHEMA_VERSION,
        id: String::new(),
        kind: "store-qemu".to_string(),
        tier: Tier::Qemu,
        result: Verdict::Pass,
        timestamp: 1_786_000_000,
        repo: "https://github.com/aien-dev/aienux".to_string(),
        commit: "a".repeat(40),
        dirty: false,
        toolchain: "rustc 1.90.0".to_string(),
        procedure: "scripts/store_qualify.sh".to_string(),
        machine: "qemu-virt-aarch64".to_string(),
        env_class: "QEMU".to_string(),
        input_artifacts: vec!["blake3:".to_string() + &"b".repeat(64)],
        output_artifacts: vec![],
        assertions: vec![pass_assertion("store_previous_or_new_only", "store-tests")],
        dependencies: vec![],
        declared_mutation: Mutation::None,
        observed_mutation: Mutation::None,
        authority: String::new(),
        output_digest: "c".repeat(64),
        external_refs: vec![],
        ledger: None,
        lease: None,
        required_features: vec![],
        reserved: String::new(),
    }
}

fn sealed(mut r: Receipt) -> Receipt {
    let id = receipt_id(&r).expect("sample receipt must seal");
    r.id = id;
    r
}

fn put(store: &Path, r: &Receipt) -> String {
    store_receipt(store, r).expect("store_receipt must accept sealed receipt")
}

fn req(
    name: &str,
    kind: &str,
    min_tier: Tier,
    assertions: Vec<&str>,
    missing_as: MissingAs,
) -> Requirement {
    Requirement {
        name: name.to_string(),
        kind: kind.to_string(),
        gate_ref: None,
        resource: None,
        repo: None,
        machine: None,
        procedure_contains: None,
        min_input_artifacts: 0,
        dependency_kinds: vec![],
        min_tier,
        assertions: assertions.into_iter().map(str::to_string).collect(),
        assertion_sources: vec![],
        mutation: None,
        require_clean: false,
        missing_as,
        missing_reason: None,
    }
}

fn gate(name: &str, requires: Vec<Requirement>) -> Gate {
    Gate {
        gate: name.to_string(),
        description: String::new(),
        requires,
    }
}

fn stored_path(store: &Path, id: &str) -> PathBuf {
    store.join("receipts").join(format!("{id}.json"))
}

#[test]
fn host_receipt_substituted_for_qemu_is_blocked() {
    let store = temp_store("host-for-qemu");
    let mut r = sample_receipt();
    r.tier = Tier::HostTest;
    r.env_class = "HOST_TEST".to_string();
    put(&store, &sealed(r));
    let manifest = gate(
        "STORE-ADV",
        vec![req(
            "Store QEMU",
            "store-qemu",
            Tier::Qemu,
            vec!["store_previous_or_new_only"],
            MissingAs::Blocked,
        )],
    );
    let report = evaluate(&manifest, &store);
    assert_ne!(report.status, Verdict::Pass);
    let line = &report.lines[0];
    assert_eq!(line.status, Verdict::Blocked);
    assert!(
        line.detail.contains("cannot satisfy"),
        "unexpected detail: {}",
        line.detail
    );
}

#[test]
fn different_commit_substituted_fails_identity() {
    let r = sealed(sample_receipt());
    let mut value = serde_json::to_value(&r).expect("receipt serializes");
    value["commit"] = serde_json::Value::String("b".repeat(40));
    let bytes = serde_json::to_vec(&value).expect("forged bytes serialize");
    let err = verify_bytes(&bytes).expect_err("tampered commit must not verify");
    assert!(err.contains("id mismatch"), "unexpected error: {err}");
}

#[test]
fn dirty_source_substituted_is_blocked() {
    let store = temp_store("dirty");
    let mut r = sample_receipt();
    r.dirty = true;
    put(&store, &sealed(r));
    let mut requirement = req(
        "Store QEMU",
        "store-qemu",
        Tier::Qemu,
        vec!["store_previous_or_new_only"],
        MissingAs::Blocked,
    );
    requirement.require_clean = true;
    let report = evaluate(&gate("STORE-ADV", vec![requirement]), &store);
    assert_ne!(report.status, Verdict::Pass);
    let line = &report.lines[0];
    assert_eq!(line.status, Verdict::Blocked);
    assert!(
        line.detail.to_ascii_lowercase().contains("clean"),
        "unexpected detail: {}",
        line.detail
    );
}

#[test]
fn assertion_from_wrong_source_fails() {
    let store = temp_store("wrong-source");
    let mut r = sample_receipt();
    r.assertions = vec![pass_assertion(
        "store_previous_or_new_only",
        "state-machine-mock",
    )];
    put(&store, &sealed(r));
    let mut requirement = req(
        "Store QEMU",
        "store-qemu",
        Tier::Qemu,
        vec!["store_previous_or_new_only"],
        MissingAs::Incomplete,
    );
    requirement.assertion_sources = vec![AssertionSource {
        id: "store_previous_or_new_only".to_string(),
        source: "aavmf_variables".to_string(),
    }];
    let report = evaluate(&gate("STORE-ADV", vec![requirement]), &store);
    assert_ne!(report.status, Verdict::Pass);
    let line = &report.lines[0];
    assert_eq!(line.status, Verdict::Fail);
    assert!(
        line.detail.contains("source"),
        "unexpected detail: {}",
        line.detail
    );
}

#[test]
fn incomplete_crash_matrix_does_not_pass() {
    let store = temp_store("crash-matrix");
    let mut r = sample_receipt();
    r.assertions = vec![pass_assertion("store_previous_or_new_only", "store-tests")];
    put(&store, &sealed(r));
    let manifest = gate(
        "STORE-ADV",
        vec![req(
            "Store QEMU",
            "store-qemu",
            Tier::Qemu,
            vec![
                "store_previous_or_new_only",
                "store_conflict_fail_closed",
                "store_nospace_zero_writes",
            ],
            MissingAs::Incomplete,
        )],
    );
    let report = evaluate(&manifest, &store);
    assert_ne!(report.status, Verdict::Pass);
    assert_ne!(report.lines[0].status, Verdict::Pass);
}

#[test]
fn missing_dependency_yields_incomplete_chain() {
    let store = temp_store("missing-dep");
    let mut r = sample_receipt();
    r.dependencies = vec!["f".repeat(64)];
    let sealed_r = sealed(r);
    let id = put(&store, &sealed_r);
    let report = verify_chain(&stored_path(&store, &id), &store).expect("chain verifies");
    assert_ne!(report.status, Verdict::Pass);
    assert_eq!(report.status, Verdict::Incomplete);
}

#[test]
fn altered_output_digest_against_ledger_ref_fails() {
    let store = temp_store("digest-ledger");
    let event = ledger::append(
        &store,
        "tester",
        "store-qemu",
        "pass exit=0",
        b"real-output",
    )
    .expect("ledger append works");
    let mut r = sample_receipt();
    r.output_digest = "d".repeat(64);
    r.ledger = Some(LedgerRef {
        index: event.index,
        hash: ledger::hex(&event.hash),
    });
    let sealed_r = sealed(r);
    let bytes = serde_json::to_vec(&sealed_r).expect("receipt serializes");
    let (_, report) = verify_with_store(&bytes, &store).expect("verify runs");
    assert_eq!(report.status, Verdict::Fail);
    let joined = report.reasons.join("; ").to_ascii_lowercase();
    assert!(
        joined.contains("digest") || joined.contains("payload") || joined.contains("ledger"),
        "unexpected reasons: {}",
        report.reasons.join("; ")
    );
}

#[test]
fn machine1_claim_without_hold_is_incomplete() {
    let mut r = sample_receipt();
    r.tier = Tier::Machine1Attended;
    r.env_class = "MACHINE1_ATTENDED".to_string();
    r.machine = "machine-1".to_string();
    let sealed_r = sealed(r);
    let bytes = serde_json::to_vec(&sealed_r).expect("receipt serializes");
    let (_, report) = verify_bytes(&bytes).expect("verify runs");
    assert_eq!(report.status, Verdict::Incomplete);
}

#[test]
fn mutating_claim_with_wrong_mutation_class_fails() {
    let store = temp_store("mutation");
    put(&store, &sealed(sample_receipt()));
    let mut requirement = req(
        "Store QEMU",
        "store-qemu",
        Tier::Qemu,
        vec!["store_previous_or_new_only"],
        MissingAs::Incomplete,
    );
    requirement.mutation = Some(Mutation::BoundedTestRegionWrite);
    let report = evaluate(&gate("STORE-ADV", vec![requirement]), &store);
    assert_ne!(report.status, Verdict::Pass);
    let line = &report.lines[0];
    assert_eq!(line.status, Verdict::Fail);
    assert!(
        line.detail.to_ascii_lowercase().contains("mutation"),
        "unexpected detail: {}",
        line.detail
    );
}

#[test]
fn blocked_receipt_used_as_pass_prerequisite_is_blocked() {
    let store = temp_store("blocked-prereq");
    let mut r = sample_receipt();
    r.result = Verdict::Blocked;
    put(&store, &sealed(r));
    let manifest = gate(
        "STORE-ADV",
        vec![req(
            "Store QEMU",
            "store-qemu",
            Tier::Qemu,
            vec!["store_previous_or_new_only"],
            MissingAs::Blocked,
        )],
    );
    let report = evaluate(&manifest, &store);
    assert_ne!(report.status, Verdict::Pass);
    assert_eq!(report.lines[0].status, Verdict::Blocked);
}
