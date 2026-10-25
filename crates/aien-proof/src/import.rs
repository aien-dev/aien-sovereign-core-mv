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

//! Import external qualification output as receipts.
//!
//! Other repositories (SEED-0B runs, rollback verifiers, Store v1 crash
//! tests) emit a small machine readable assertion file. `import` turns that
//! file plus explicit source, artifact, and dependency references into a
//! sealed EvidenceReceiptV1. Prose is never parsed; the input contract is
//! JSON and any deviation fails closed.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::evidence::{
    check_commit, Assertion, LeaseRef, LedgerRef, Mutation, Receipt, Tier, Verdict, SCHEMA,
    SCHEMA_VERSION,
};
use crate::lease::check_resource_name;

/// Machine readable assertion file produced by each test setup.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionFile {
    pub assertions: Vec<AssertionInput>,
    /// Optional top level result. When absent it is derived from the
    /// assertions. A claimed PASS with a failing assertion is refused.
    #[serde(default)]
    pub result: Option<String>,
}

/// One assertion line in the input file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionInput {
    pub id: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub observed: String,
    pub result: AssertionResult,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub note: String,
}

/// Accepts `pass`/`fail` strings or booleans.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AssertionResult {
    Bool(bool),
    Word(String),
}

impl AssertionResult {
    fn pass(&self) -> Result<bool, String> {
        match self {
            AssertionResult::Bool(b) => Ok(*b),
            AssertionResult::Word(w) => match w.to_ascii_lowercase().as_str() {
                "pass" => Ok(true),
                "fail" => Ok(false),
                _ => Err(format!("assertion result must be pass or fail, got {w:?}")),
            },
        }
    }
}

/// All inputs needed to mint a receipt from external output.
#[derive(Debug, Clone, Default)]
pub struct ImportRequest {
    pub repo: String,
    pub commit: String,
    pub dirty: bool,
    pub kind: String,
    pub tier: String,
    pub toolchain: String,
    pub procedure: String,
    pub machine: String,
    pub input_artifacts: Vec<String>,
    pub output_artifacts: Vec<String>,
    pub dependencies: Vec<String>,
    pub declared_mutation: Option<String>,
    pub observed_mutation: Option<String>,
    pub authority: String,
    pub output_digest: Option<String>,
    pub output_file: Option<String>,
    pub external_refs: Vec<String>,
    pub ledger_index: Option<u64>,
    pub ledger_hash: Option<String>,
    pub lease_hold: Option<String>,
    pub lease_resource: Option<String>,
    pub timestamp: Option<u64>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse and validate the assertion file. Never sniffs prose.
pub fn read_assertion_file(path: &Path) -> Result<(Vec<Assertion>, Verdict), String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read assertion file: {e}"))?;
    let file: AssertionFile =
        serde_json::from_str(&text).map_err(|e| format!("assertion file invalid: {e}"))?;
    if file.assertions.is_empty() {
        return Err("assertion file must carry at least one assertion".to_string());
    }
    let mut out = Vec::with_capacity(file.assertions.len());
    for input in &file.assertions {
        if input.id.trim().is_empty() {
            return Err("assertion id must not be empty".to_string());
        }
        out.push(Assertion {
            id: input.id.clone(),
            expected: input.expected.clone(),
            observed: input.observed.clone(),
            pass: input.result.pass()?,
            source: input.source.clone(),
            note: input.note.clone(),
        });
    }
    let derived = if out.iter().all(|a| a.pass) {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    if let Some(claimed) = file.result {
        let verdict = Verdict::parse(&claimed).ok_or_else(|| format!("bad result {claimed:?}"))?;
        if verdict == Verdict::Pass && derived != Verdict::Pass {
            return Err("assertion file claims PASS but an assertion failed".to_string());
        }
        // A setup may only downgrade (BLOCKED, INCOMPLETE, SKIPPED), never
        // upgrade a failing run. FAIL stays FAIL.
        if derived == Verdict::Fail && verdict == Verdict::Pass {
            return Err("cannot upgrade a failing run to PASS".to_string());
        }
        if derived == Verdict::Pass && verdict == Verdict::Fail {
            return Err("file claims FAIL but every assertion passed".to_string());
        }
        Ok((out, verdict))
    } else {
        Ok((out, derived))
    }
}

// Canonical store assertion IDs live in store_assert.rs; this table only
// groups them per qualification kind.
use crate::store_assert::*;
fn store_required_assertions(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "store-v1-format" => Some(&[
            STORE_FORMAT_VALID,
            STORE_OBJECT_IDS_VALID,
            STORE_CATALOG_VALID,
            STORE_COMMIT_VALID,
            STORE_FULL_GRAPH_VALID,
        ]),
        "store-v1-host" => Some(&[
            STORE_PREVIOUS_OR_NEW_ONLY,
            STORE_CONFLICT_FAIL_CLOSED,
            STORE_INCONSISTENT_HISTORY_FAIL_CLOSED,
            STORE_IO_ERROR_NOT_FALLBACK,
            STORE_NOSPACE_ZERO_WRITES,
            STORE_OPEN_SIDE_EFFECT_FREE,
        ]),
        "store-v1-qemu" => Some(&[
            STORE_PREVIOUS_OR_NEW_ONLY,
            STORE_FULL_GRAPH_VALID,
            STORE_QEMU_REBOOT_RECOVERED,
        ]),
        _ => None,
    }
}

/// Store qualification profile for kinds starting with "store-".
/// Checks source identity, tier allow list, artifact and assertion
/// presence, assertion origin, PASS consistency, mutation allow list,
/// and the per-kind required assertion set. Output digest presence for
/// PASS stays enforced by `build_receipt`.
pub fn validate_store_profile(
    req: &ImportRequest,
    assertions: &[Assertion],
    result: Verdict,
) -> Result<(), String> {
    if req.repo.trim().is_empty() {
        return Err("store qualification requires non-empty repo".to_string());
    }
    check_commit(&req.commit)?;
    if req.procedure.trim().is_empty() {
        return Err("store qualification requires non-empty procedure".to_string());
    }
    if req.procedure.trim() == "TBD" {
        return Err("store qualification requires a named procedure, got \"TBD\"".to_string());
    }
    let tier = Tier::parse(&req.tier).ok_or_else(|| {
        format!(
            "store qualification kind {:?} has unknown tier {:?}",
            req.kind, req.tier
        )
    })?;
    match tier {
        Tier::HostTest | Tier::Qemu | Tier::QemuSecurity => {}
        _ => {
            return Err(format!(
                "store qualification kind {:?} refuses tier {:?}",
                req.kind,
                tier.as_str()
            ));
        }
    }
    if req.input_artifacts.is_empty() {
        return Err("store qualification requires at least one input artifact digest".to_string());
    }
    if assertions.is_empty() {
        return Err("store qualification requires at least one assertion".to_string());
    }
    for a in assertions {
        if a.source.trim().is_empty() {
            return Err(format!(
                "store qualification requires assertion source for {:?}",
                a.id
            ));
        }
    }
    if result == Verdict::Pass && assertions.iter().any(|a| !a.pass) {
        return Err("store qualification claims PASS but an assertion failed".to_string());
    }
    let declared = match &req.declared_mutation {
        Some(s) => Mutation::parse(s).ok_or_else(|| format!("unknown mutation class {s:?}"))?,
        None => Mutation::None,
    };
    match declared {
        Mutation::None
        | Mutation::VolatileOnly
        | Mutation::RemovableMediaOnly
        | Mutation::BoundedTestRegionWrite => {}
        _ => {
            return Err(format!(
                "store qualification kind {:?} refuses declared mutation {:?}",
                req.kind,
                declared.as_str()
            ));
        }
    }
    if let Some(required) = store_required_assertions(req.kind.as_str()) {
        for id in required {
            let ok = assertions.iter().any(|a| a.id == *id && a.pass);
            if !ok {
                return Err(format!(
                    "store qualification kind {:?} missing required passing assertion {:?}",
                    req.kind, id
                ));
            }
        }
    }
    Ok(())
}

/// Build a sealed receipt from validated inputs. Hardware tiers require an
/// explicit declared mutation and a lease reference; PASS receipts require
/// an output digest (direct hex or hashed from an output file).
pub fn build_receipt(
    req: &ImportRequest,
    assertions: Vec<Assertion>,
    result: Verdict,
) -> Result<Receipt, String> {
    if req.kind.starts_with("store-") {
        validate_store_profile(req, &assertions, result)?;
    }
    if req.kind == crate::formal::KIND {
        crate::formal::validate_formal_profile(req, &assertions, result)?;
    }
    let tier = Tier::parse(&req.tier)
        .ok_or_else(|| format!("unknown qualification tier {:?}", req.tier))?;
    check_commit(&req.commit)?;
    if req.repo.trim().is_empty() {
        return Err("repo must not be empty".to_string());
    }
    if req.kind.trim().is_empty() {
        return Err("gate/kind must not be empty".to_string());
    }
    if req.procedure.trim().is_empty() {
        return Err("procedure identity must not be empty".to_string());
    }
    if req.machine.trim().is_empty() {
        return Err("machine identity must not be empty".to_string());
    }
    for dep in &req.dependencies {
        let low = dep.to_ascii_lowercase();
        if low.len() != 64 || !low.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("dependency {dep:?} is not a 64 digit receipt hash"));
        }
    }
    let declared = match &req.declared_mutation {
        Some(s) => Mutation::parse(s).ok_or_else(|| format!("unknown mutation class {s:?}"))?,
        None => {
            if tier.is_hardware() {
                return Err("hardware receipts require --declared-mutation".to_string());
            }
            Mutation::None
        }
    };
    let observed = match &req.observed_mutation {
        Some(s) => Mutation::parse(s).ok_or_else(|| format!("unknown mutation class {s:?}"))?,
        None => declared,
    };
    let output_digest = match (&req.output_digest, &req.output_file) {
        (Some(hex), _) => {
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("output digest must be 64 hex digits (BLAKE3)".to_string());
            }
            hex.to_ascii_lowercase()
        }
        (None, Some(path)) => {
            let bytes = std::fs::read(path).map_err(|e| format!("cannot read output file: {e}"))?;
            blake3::hash(&bytes).to_hex().to_string()
        }
        (None, None) => {
            if result == Verdict::Pass {
                return Err("PASS receipts require --output-digest or --output-file".to_string());
            }
            "0".repeat(64)
        }
    };
    let ledger = match (req.ledger_index, &req.ledger_hash) {
        (Some(index), Some(hash)) => {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("ledger hash must be 64 hex digits".to_string());
            }
            Some(LedgerRef {
                index,
                hash: hash.to_ascii_lowercase(),
            })
        }
        (None, None) => None,
        _ => return Err("ledger reference needs both index and hash".to_string()),
    };
    let lease = match (&req.lease_hold, &req.lease_resource) {
        (Some(hold), Some(resource)) => {
            check_resource_name(resource)?;
            if hold.len() != 64 || !hold.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("lease hold id must be 64 hex digits".to_string());
            }
            Some(LeaseRef {
                hold_id: hold.to_ascii_lowercase(),
                resource: resource.clone(),
            })
        }
        (None, None) => None,
        _ => return Err("lease binding needs both hold id and resource".to_string()),
    };
    if tier.is_hardware() && lease.is_none() {
        return Err("hardware receipts require --lease-hold and --lease-resource".to_string());
    }
    let mut receipt = Receipt {
        schema: SCHEMA.to_string(),
        version: SCHEMA_VERSION,
        id: String::new(),
        kind: req.kind.clone(),
        tier,
        result,
        timestamp: req.timestamp.unwrap_or_else(now),
        repo: req.repo.clone(),
        commit: req.commit.to_ascii_lowercase(),
        dirty: req.dirty,
        toolchain: req.toolchain.clone(),
        procedure: req.procedure.clone(),
        machine: req.machine.clone(),
        env_class: tier.as_str().to_string(),
        input_artifacts: req.input_artifacts.clone(),
        output_artifacts: req.output_artifacts.clone(),
        assertions,
        dependencies: req
            .dependencies
            .iter()
            .map(|d| d.to_ascii_lowercase())
            .collect(),
        declared_mutation: declared,
        observed_mutation: observed,
        authority: req.authority.clone(),
        output_digest,
        external_refs: req.external_refs.clone(),
        ledger,
        lease,
        required_features: vec![],
        reserved: String::new(),
    };
    let id = crate::evidence::receipt_id(&receipt)?;
    receipt.id = id;
    // Final self check: identity plus consistency, fail closed.
    let text = serde_json::to_vec(&receipt).map_err(|e| e.to_string())?;
    let (_, report) = crate::evidence::verify_bytes(&text)?;
    if report.status == Verdict::Fail {
        return Err(format!(
            "built receipt fails self check: {}",
            report.reasons.join("; ")
        ));
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{store_receipt, verify_bytes};

    fn write(path: &Path, text: &str) {
        std::fs::write(path, text).unwrap();
    }

    fn req() -> ImportRequest {
        ImportRequest {
            repo: "https://github.com/aien-dev/aienux".to_string(),
            commit: "a".repeat(40),
            dirty: false,
            kind: "seed-0b-qemu".to_string(),
            tier: "QEMU".to_string(),
            toolchain: "rustc 1.90.0".to_string(),
            procedure: "scripts/qemu_seed0b.sh".to_string(),
            machine: "qemu-virt-aarch64".to_string(),
            input_artifacts: vec!["blake3:".to_string() + &"b".repeat(64)],
            output_digest: Some("c".repeat(64)),
            ..Default::default()
        }
    }

    #[test]
    fn import_roundtrip() {
        let dir = crate::testutil::temp_dir("import-ok");
        let file = dir.join("assert.json");
        write(
            &file,
            r#"{"assertions": [{"id": "artifact_signature_valid", "expected": "valid", "observed": "valid", "result": "pass", "source": "qemu"}]}"#,
        );
        let (assertions, result) = read_assertion_file(&file).unwrap();
        assert_eq!(result, Verdict::Pass);
        let receipt = build_receipt(&req(), assertions, result).unwrap();
        let store = crate::testutil::temp_dir("import-store");
        let id = store_receipt(&store, &receipt).unwrap();
        assert_eq!(id, receipt.id);
        let text = std::fs::read(store.join("receipts").join(format!("{id}.json"))).unwrap();
        let (_, report) = verify_bytes(&text).unwrap();
        assert_eq!(report.status, Verdict::Pass);
    }

    #[test]
    fn prose_is_never_parsed() {
        let dir = crate::testutil::temp_dir("import-prose");
        let file = dir.join("assert.json");
        write(&file, "all tests passed, trust me");
        assert!(read_assertion_file(&file).is_err());
    }

    #[test]
    fn claimed_pass_with_failed_assertion_refused() {
        let dir = crate::testutil::temp_dir("import-liar");
        let file = dir.join("assert.json");
        write(
            &file,
            r#"{"result": "PASS", "assertions": [{"id": "x", "result": "fail"}]}"#,
        );
        assert!(read_assertion_file(&file).is_err());
    }

    #[test]
    fn hardware_import_needs_mutation_and_lease() {
        let dir = crate::testutil::temp_dir("import-hw");
        let file = dir.join("assert.json");
        write(&file, r#"{"assertions": [{"id": "x", "result": "pass"}]}"#);
        let (assertions, result) = read_assertion_file(&file).unwrap();
        let mut r = req();
        r.tier = "MACHINE1_ATTENDED".to_string();
        r.machine = "machine-1".to_string();
        assert!(build_receipt(&r, assertions.clone(), result).is_err());
        r.declared_mutation = Some("REMOVABLE_MEDIA_ONLY".to_string());
        assert!(build_receipt(&r, assertions.clone(), result).is_err());
        r.lease_hold = Some("e".repeat(64));
        r.lease_resource = Some("machine-1".to_string());
        r.procedure = "scripts/seed0b_machine1.sh".to_string();
        let receipt = build_receipt(&r, assertions, result).unwrap();
        assert_eq!(receipt.env_class, "MACHINE1_ATTENDED");
        let _ = dir;
    }

    #[test]
    fn unknown_tier_and_mutation_refused() {
        let dir = crate::testutil::temp_dir("import-bad-enum");
        let file = dir.join("assert.json");
        write(&file, r#"{"assertions": [{"id": "x", "result": "pass"}]}"#);
        let (assertions, result) = read_assertion_file(&file).unwrap();
        let mut r = req();
        r.tier = "REAL_HARDWARE".to_string();
        assert!(build_receipt(&r, assertions.clone(), result).is_err());
        r.tier = "QEMU".to_string();
        r.declared_mutation = Some("FORMAT_EVERYTHING".to_string());
        assert!(build_receipt(&r, assertions, result).is_err());
        let _ = dir;
    }

    fn store_req(kind: &str, tier: &str) -> ImportRequest {
        let mut r = req();
        r.kind = kind.to_string();
        r.tier = tier.to_string();
        r.procedure = "scripts/store_test.sh".to_string();
        r.machine = "store-test-runner".to_string();
        r
    }

    fn store_assertion(id: &str) -> Assertion {
        Assertion {
            id: id.to_string(),
            expected: String::new(),
            observed: String::new(),
            pass: true,
            source: "store run log".to_string(),
            note: String::new(),
        }
    }

    fn format_assertions() -> Vec<Assertion> {
        vec![
            store_assertion("store_format_valid"),
            store_assertion("store_object_ids_valid"),
            store_assertion("store_catalog_valid"),
            store_assertion("store_commit_valid"),
            store_assertion("store_full_graph_valid"),
        ]
    }

    fn host_assertions() -> Vec<Assertion> {
        vec![
            store_assertion("store_previous_or_new_only"),
            store_assertion("store_conflict_fail_closed"),
            store_assertion("store_inconsistent_history_fail_closed"),
            store_assertion("store_io_error_not_fallback"),
            store_assertion("store_nospace_zero_writes"),
            store_assertion("store_open_side_effect_free"),
        ]
    }

    #[test]
    fn store_format_valid_passes_profile() {
        let r = store_req("store-v1-format", "HOST_TEST");
        let assertions = format_assertions();
        validate_store_profile(&r, &assertions, Verdict::Pass).unwrap();
        let receipt = build_receipt(&r, assertions, Verdict::Pass).unwrap();
        assert_eq!(receipt.kind, "store-v1-format");
        assert_eq!(receipt.tier, Tier::HostTest);
    }

    #[test]
    fn store_format_missing_required_refused() {
        let r = store_req("store-v1-format", "HOST_TEST");
        let mut assertions = format_assertions();
        assertions.retain(|a| a.id != "store_commit_valid");
        let err = validate_store_profile(&r, &assertions, Verdict::Pass).unwrap_err();
        assert!(err.contains("store_commit_valid"), "got: {err}");
        assert!(err.contains("store-v1-format"), "got: {err}");
        assert!(build_receipt(&r, assertions, Verdict::Pass).is_err());
    }

    #[test]
    fn store_empty_source_refused() {
        let r = store_req("store-v1-format", "QEMU");
        let mut assertions = format_assertions();
        assertions[0].source.clear();
        let err = validate_store_profile(&r, &assertions, Verdict::Pass).unwrap_err();
        assert!(err.contains("source"), "got: {err}");
        assert!(err.contains("store_format_valid"), "got: {err}");
        assert!(build_receipt(&r, assertions, Verdict::Pass).is_err());
    }

    #[test]
    fn store_host_machine1_tier_refused() {
        let r = store_req("store-v1-host", "MACHINE1_ATTENDED");
        let assertions = host_assertions();
        let err = validate_store_profile(&r, &assertions, Verdict::Pass).unwrap_err();
        assert!(err.contains("MACHINE1_ATTENDED"), "got: {err}");
        assert!(build_receipt(&r, assertions, Verdict::Pass).is_err());
    }

    #[test]
    fn store_destructive_mutation_refused() {
        let mut r = store_req("store-v1-format", "QEMU");
        r.declared_mutation = Some("DESTRUCTIVE_STORAGE".to_string());
        let assertions = format_assertions();
        let err = validate_store_profile(&r, &assertions, Verdict::Pass).unwrap_err();
        assert!(err.contains("DESTRUCTIVE_STORAGE"), "got: {err}");
        assert!(build_receipt(&r, assertions, Verdict::Pass).is_err());
    }

    #[test]
    fn store_future_unknown_subkind_has_no_required_set() {
        assert!(store_required_assertions("store-v1-future").is_none());
        let r = store_req("store-v1-future", "QEMU");
        let assertions = vec![store_assertion("store_future_check")];
        validate_store_profile(&r, &assertions, Verdict::Pass).unwrap();
        let receipt = build_receipt(&r, assertions, Verdict::Pass).unwrap();
        assert_eq!(receipt.kind, "store-v1-future");
        let mut bad = store_req("store-v1-future", "QEMU");
        bad.procedure = "TBD".to_string();
        let assertions = vec![store_assertion("store_future_check")];
        assert!(build_receipt(&bad, assertions, Verdict::Pass).is_err());
    }
}
