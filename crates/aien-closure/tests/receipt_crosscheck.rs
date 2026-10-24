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

//! Rust side of the receipt cross-check for the Omega C reader (VC1 carry
//! item 5). Mints receipts with aien-proof for the cases the C reader's three
//! fixtures (`omega/tests/resolve/receipts`, all `component-qualification`,
//! tier HOST_TEST, mutation NONE, no lease, no ledger) do not cover, and
//! commits them under `tests/fixtures/receipts/` with `expected.txt`.
//!
//! The committed bytes must equal what this test mints (deterministic), every
//! fixture must verify the way `expected.txt` says, and a change to any bound
//! field must break the id. To rewrite the fixtures:
//! `cargo test -p aien-closure --test receipt_crosscheck -- --ignored regenerate_receipt_fixtures`.

mod common;

use aien_closure::verify::profile_min_tier;
use aien_proof::chain::tier_satisfies;
use aien_proof::evidence::{
    receipt_id, store_receipt, verify_bytes, verify_with_store, Assertion, LeaseRef, LedgerRef,
    Mutation, Receipt, Tier, Verdict, SCHEMA, SCHEMA_VERSION,
};
use common::temp_dir;
use std::fs;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/receipts")
}

fn b3(s: &str) -> String {
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

fn assertion(id: &str, pass: bool) -> Assertion {
    Assertion {
        id: id.to_string(),
        expected: "pass".to_string(),
        observed: if pass { "pass" } else { "fail" }.to_string(),
        pass,
        source: "cross-check".to_string(),
        note: String::new(),
    }
}

/// An unsealed receipt of the given verifier-profile kind and tier. `name` makes
/// every fixture's inputs and output distinct.
fn base(name: &str, kind: &str, tier: Tier) -> Receipt {
    let r = Receipt {
        schema: SCHEMA.to_string(),
        version: SCHEMA_VERSION,
        id: String::new(),
        kind: kind.to_string(),
        tier,
        result: Verdict::Pass,
        timestamp: 1_786_000_100,
        repo: "https://github.com/aien-dev/aien-sovereign-core".to_string(),
        commit: "b".repeat(40),
        dirty: false,
        toolchain: "rustc fixture".to_string(),
        procedure: format!("receipt-crosscheck {name}"),
        machine: "host".to_string(),
        env_class: tier.as_str().to_string(),
        input_artifacts: vec![
            format!("sha256:{}", "1".repeat(64)),
            format!("sha256:{}", b3(&format!("{name}-source"))),
        ],
        output_artifacts: vec![],
        assertions: vec![assertion("crosscheck_pass", true)],
        dependencies: vec![],
        declared_mutation: Mutation::None,
        observed_mutation: Mutation::None,
        authority: String::new(),
        output_digest: b3(&format!("{name}-output")),
        external_refs: vec![],
        ledger: None,
        lease: None,
        required_features: vec![],
        reserved: String::new(),
    };
    r
}

fn seal(mut r: Receipt) -> Receipt {
    r.id = receipt_id(&r).unwrap();
    r
}

fn lease(name: &str) -> LeaseRef {
    LeaseRef {
        hold_id: b3(&format!("{name}-hold")),
        resource: "machine-1".to_string(),
    }
}

fn ledger(name: &str) -> LedgerRef {
    LedgerRef {
        index: 42,
        hash: b3(&format!("{name}-ledger-event")),
    }
}

struct Case {
    name: &'static str,
    receipt: Receipt,
    /// What a profile-aware reader must do: accept only if the identity and
    /// consistency check pass and the tier satisfies the kind's profile.
    accept: bool,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut add = |name: &'static str, receipt: Receipt, accept: bool| {
        v.push(Case {
            name,
            receipt,
            accept,
        });
    };

    // Accepted: non-NONE mutation on the lowest tier.
    let mut r = base("host-volatile", "host-v1", Tier::HostTest);
    r.declared_mutation = Mutation::VolatileOnly;
    r.observed_mutation = Mutation::VolatileOnly;
    add("host-v1-volatile-mutation", seal(r), true);

    // Accepted: observed strictly below declared, both non-NONE.
    let mut r = base("host-declared-above", "host-v1", Tier::HostTest);
    r.declared_mutation = Mutation::BoundedTestRegionWrite;
    r.observed_mutation = Mutation::OneTimeBootSelection;
    add("host-v1-declared-above-observed", seal(r), true);

    // Accepted: ledger binding (the event is not in any store here).
    let mut r = base("host-ledger", "host-v1", Tier::HostTest);
    r.ledger = Some(ledger("host-ledger"));
    add("host-v1-ledger-binding", seal(r), true);

    // Accepted: lists in non-canonical order with duplicates, mixed digest
    // spellings, several assertions. The id is over the sorted unique form.
    let mut r = base("host-lists", "host-v1", Tier::HostTest);
    r.input_artifacts = vec![
        format!("sha256:{}", "f".repeat(64)),
        format!("blake3:{}", "0".repeat(64)),
        format!("sha256:{}", "f".repeat(64)),
        "a".repeat(64),
    ];
    r.output_artifacts = vec![
        format!("sha512:{}", "9".repeat(128)),
        format!("blake3:{}", "1".repeat(64)),
    ];
    r.dependencies = vec!["e".repeat(64), "2".repeat(64), "e".repeat(64)];
    r.assertions = vec![
        assertion("zeta", true),
        assertion("alpha", true),
        assertion("mid", true),
    ];
    r.external_refs = vec![
        "ref-b".to_string(),
        "ref-a".to_string(),
        "ref-b".to_string(),
    ];
    add("host-v1-unsorted-duplicate-lists", seal(r), true);

    // Accepted: the QEMU profile at the QEMU tier.
    let mut r = base("qemu", "qemu-v1", Tier::Qemu);
    r.machine = "qemu-virt-aarch64".to_string();
    r.declared_mutation = Mutation::BoundedTestRegionWrite;
    r.observed_mutation = Mutation::BoundedTestRegionWrite;
    add("qemu-v1-qemu-tier-mutation", seal(r), true);

    // Accepted: a hardware-tier receipt satisfies the lower host-v1 profile;
    // it carries a lease and a ledger binding and a boot-level mutation.
    let mut r = base("hw-lease", "host-v1", Tier::Machine1Attended);
    r.machine = "machine-1".to_string();
    r.declared_mutation = Mutation::BootConfigurationChange;
    r.observed_mutation = Mutation::OneTimeBootSelection;
    r.lease = Some(lease("hw-lease"));
    r.ledger = Some(ledger("hw-lease"));
    add("host-v1-hardware-tier-lease-ledger", seal(r), true);

    // Accepted: PRODUCTION with an authority reference, lease, trust-root
    // mutation (the maximum severity pair that is not destructive).
    let mut r = base("production", "production-v1", Tier::Production);
    r.machine = "machine-1".to_string();
    r.authority = "owner-key:fixture".to_string();
    r.declared_mutation = Mutation::TrustRootChange;
    r.observed_mutation = Mutation::TrustRootChange;
    r.lease = Some(lease("production"));
    add("production-v1-authority-lease", seal(r), true);

    // Refused: hardware tier PASS with no lease (aien-proof reports
    // INCOMPLETE).
    let mut r = base("hw-no-lease", "host-v1", Tier::Machine1ReadOnly);
    r.machine = "machine-1".to_string();
    add("host-v1-hardware-without-lease", seal(r), false);

    // Refused: a recorded FAIL with a failed assertion.
    let mut r = base("fail", "host-v1", Tier::HostTest);
    r.result = Verdict::Fail;
    r.assertions = vec![assertion("crosscheck_pass", false)];
    add("host-v1-result-fail", seal(r), false);

    // Refused: TEST_ONLY_TRUST satisfies nothing but itself.
    let r = base("test-only", "host-v1", Tier::TestOnlyTrust);
    add("host-v1-test-only-trust", seal(r), false);

    // Refused: the production profile needs a physical tier; QEMU is not.
    let r = base("prod-on-qemu", "production-v1", Tier::Qemu);
    add("production-v1-on-qemu-tier", seal(r), false);

    v
}

fn mutation_text(r: &Receipt) -> String {
    format!(
        "{}/{}",
        r.declared_mutation.as_str(),
        r.observed_mutation.as_str()
    )
}

fn yes(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// What aien-proof says: single-receipt verification (identity and
/// consistency), then with a store that has no ledger and no hold.
fn rust_verdict(r: &Receipt) -> (String, String) {
    let text = serde_json::to_string_pretty(r).unwrap();
    let alone = match verify_bytes(text.as_bytes()) {
        Ok((_, rep)) => rep.status.as_str().to_string(),
        Err(e) => format!("ERROR({e})"),
    };
    let empty = temp_dir("crosscheck-empty-store");
    let with_store = match verify_with_store(text.as_bytes(), &empty) {
        Ok((_, rep)) => rep.status.as_str().to_string(),
        Err(_) => "ERROR".to_string(),
    };
    (alone, with_store)
}

fn expected_text(cases: &[Case]) -> String {
    let mut s = String::from(
        "# receipt cross-check expectations, one line per fixture (file <id>.json)\n\
         # accept = verify passes AND tier satisfies the profile named by kind\n\
         # verify = aien-proof verify_bytes status; with_store = verify_with_store against a store\n\
         # holding no ledger and no hold (the C reader does not check ledger or lease bindings)\n",
    );
    let mut sorted: Vec<&Case> = cases.iter().collect();
    sorted.sort_by(|a, b| a.receipt.id.cmp(&b.receipt.id));
    for c in sorted {
        let r = &c.receipt;
        let (verify, with_store) = rust_verdict(r);
        s.push_str(&format!(
            "{} name={} kind={} tier={} result={} mutation={} lease={} ledger={} verify={} with_store={} accept={}\n",
            r.id,
            c.name,
            r.kind,
            r.tier.as_str(),
            r.result.as_str(),
            mutation_text(r),
            yes(r.lease.is_some()),
            yes(r.ledger.is_some()),
            verify,
            with_store,
            if c.accept { "ACCEPT" } else { "REFUSE" },
        ));
    }
    s
}

/// The reader decision the table claims, recomputed from aien-proof.
fn decided_accept(r: &Receipt) -> bool {
    let (verify, _) = rust_verdict(r);
    let need = profile_min_tier(&r.kind).expect("fixture kinds are verifier profiles");
    verify == "PASS" && tier_satisfies(need, r.tier)
}

fn minted_bytes(r: &Receipt) -> String {
    let store = temp_dir("crosscheck-mint");
    let id = store_receipt(&store, r).unwrap();
    fs::read_to_string(store.join("receipts").join(format!("{id}.json"))).unwrap()
}

#[test]
#[ignore]
fn regenerate_receipt_fixtures() {
    let dir = fixture_dir();
    let cs = cases();
    fs::create_dir_all(&dir).unwrap();
    for e in fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "json") {
            fs::remove_file(p).unwrap();
        }
    }
    for c in &cs {
        fs::write(
            dir.join(format!("{}.json", c.receipt.id)),
            minted_bytes(&c.receipt),
        )
        .unwrap();
    }
    fs::write(dir.join("expected.txt"), expected_text(&cs)).unwrap();
}

#[test]
fn every_case_is_decided_as_the_table_says() {
    for c in cases() {
        assert_eq!(
            decided_accept(&c.receipt),
            c.accept,
            "{}: aien-proof decides otherwise",
            c.name
        );
    }
}

#[test]
fn cases_cover_what_the_c_fixtures_do_not() {
    let cs = cases();
    let accepted: Vec<&Case> = cs.iter().filter(|c| c.accept).collect();
    let has = |f: &dyn Fn(&Receipt) -> bool| accepted.iter().any(|c| f(&c.receipt));
    assert!(has(&|r| r.declared_mutation != Mutation::None));
    assert!(has(&|r| r.observed_mutation != Mutation::None));
    assert!(has(&|r| r.lease.is_some()));
    assert!(has(&|r| r.ledger.is_some()));
    assert!(has(&|r| r.kind == "host-v1"));
    assert!(has(&|r| r.kind == "qemu-v1"));
    assert!(has(&|r| r.kind == "production-v1"));
    assert!(has(&|r| !r.authority.is_empty()));
    assert!(cs.iter().any(|c| !c.accept));
    // None of them is the kind the C fixtures use.
    assert!(cs
        .iter()
        .all(|c| c.receipt.kind != "component-qualification"));
    // Ids are distinct.
    let ids: std::collections::BTreeSet<&str> = cs.iter().map(|c| c.receipt.id.as_str()).collect();
    assert_eq!(ids.len(), cs.len());
}

#[test]
fn committed_fixtures_are_exactly_what_is_minted() {
    let dir = fixture_dir();
    let cs = cases();
    let mut on_disk: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json"))
        .collect();
    on_disk.sort();
    let mut want: Vec<String> = cs
        .iter()
        .map(|c| format!("{}.json", c.receipt.id))
        .collect();
    want.sort();
    assert_eq!(on_disk, want, "fixture set differs; regenerate");
    for c in &cs {
        let path = dir.join(format!("{}.json", c.receipt.id));
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            minted_bytes(&c.receipt),
            "{} bytes differ",
            c.name
        );
    }
    assert_eq!(
        fs::read_to_string(dir.join("expected.txt")).unwrap(),
        expected_text(&cs),
        "expected.txt differs; regenerate"
    );
}

#[test]
fn committed_files_verify_by_their_own_file_name() {
    // Read the files the way the C reader will: by name, then check that the
    // name is the canonical identity of the bytes inside.
    let dir = fixture_dir();
    for c in cases() {
        let id = &c.receipt.id;
        let text = fs::read(dir.join(format!("{id}.json"))).unwrap();
        let (r, _) = verify_bytes(&text).unwrap();
        assert_eq!(&r.id, id);
        assert_eq!(&receipt_id(&r).unwrap(), id);
    }
}

#[test]
fn store_bound_cases_are_refused_without_a_store_and_the_rest_are_not() {
    // The C reader ignores ledger and lease bindings (stated limit), so the
    // fixtures record what aien-proof does with them when no store backs them.
    for c in cases() {
        let (verify, with_store) = rust_verdict(&c.receipt);
        let bound = c.receipt.lease.is_some() || c.receipt.ledger.is_some();
        if bound && verify == "PASS" {
            assert_eq!(with_store, "ERROR", "{}", c.name);
        } else if !bound {
            assert_eq!(with_store, verify, "{}", c.name);
        }
    }
}

fn set(v: &mut serde_json::Value, ptr: &str, to: serde_json::Value) -> bool {
    match v.pointer_mut(ptr) {
        Some(slot) => {
            *slot = to;
            true
        }
        None => false,
    }
}

#[test]
fn changing_any_bound_field_breaks_the_id() {
    // Mutant-style negatives: every field the C reader must hash. Each edit
    // keeps the old id, so a reader that skips the field would still accept.
    let dir = fixture_dir();
    let tampers: Vec<(&str, serde_json::Value)> = vec![
        ("/kind", "host-v2".into()),
        ("/timestamp", 7.into()),
        ("/machine", "other".into()),
        ("/procedure", "other".into()),
        ("/commit", "c".repeat(40).into()),
        ("/dirty", true.into()),
        ("/output_digest", "d".repeat(64).into()),
        ("/authority", "someone-else".into()),
        ("/declared_mutation", "DESTRUCTIVE_STORAGE".into()),
        ("/observed_mutation", "VOLATILE_ONLY".into()),
        ("/lease/resource", "machine-2".into()),
        ("/lease/hold_id", "0".repeat(64).into()),
        ("/ledger/index", 43.into()),
        ("/ledger/hash", "0".repeat(64).into()),
        ("/assertions/0/observed", "changed".into()),
        (
            "/input_artifacts/0",
            format!("sha256:{}", "7".repeat(64)).into(),
        ),
        ("/dependencies/0", "9".repeat(64).into()),
    ];
    let mut applied = 0usize;
    for c in cases() {
        let text = fs::read_to_string(dir.join(format!("{}.json", c.receipt.id))).unwrap();
        let original: serde_json::Value = serde_json::from_str(&text).unwrap();
        for (ptr, to) in &tampers {
            if original.pointer(ptr) == Some(to) {
                continue;
            }
            let mut v = original.clone();
            if !set(&mut v, ptr, to.clone()) {
                continue;
            }
            applied += 1;
            let tampered = serde_json::to_string(&v).unwrap();
            let got = verify_bytes(tampered.as_bytes());
            assert!(
                !matches!(&got, Ok((_, rep)) if rep.status == Verdict::Pass),
                "{}: changing {ptr} was not detected",
                c.name
            );
        }
    }
    assert!(applied > 60, "only {applied} tampers applied");
}
