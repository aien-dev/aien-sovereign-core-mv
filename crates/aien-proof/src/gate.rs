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

//! Gate manifests: small declarative descriptions of what closes a gate.
//!
//! The evidence system never decides architectural policy. It evaluates a
//! checked in manifest against stored receipts and reports exactly which
//! prerequisite is responsible for PASS, FAIL, BLOCKED, or INCOMPLETE.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

use crate::chain::tier_satisfies;
use crate::evidence::{list_receipts, load_receipt, Tier, Verdict};

/// Subdirectory of the proof store holding gate manifests.
pub const GATE_DIR: &str = "gates";

/// One prerequisite inside a gate manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirement {
    /// Human label used in status output, for example `SEED-0B-QEMU`.
    pub name: String,
    /// Receipt kind that satisfies this requirement, for example `m3`.
    pub kind: String,
    /// Optional nested gate requirement. When set, `kind` must be empty.
    #[serde(default)]
    pub gate_ref: Option<String>,
    /// Exact physical resource required for hardware evidence and its hold.
    #[serde(default)]
    pub resource: Option<String>,
    /// Exact repository identity when the prerequisite belongs to another repo.
    #[serde(default)]
    pub repo: Option<String>,
    /// Exact machine/resource identity for non-hardware tiers.
    #[serde(default)]
    pub machine: Option<String>,
    /// Required substring in the procedure identity, often carrying the test script version.
    #[serde(default)]
    pub procedure_contains: Option<String>,
    /// Minimum count of input artifact digests required on the receipt.
    #[serde(default)]
    pub min_input_artifacts: usize,
    /// Required receipt kinds reachable through this receipt's verified chain.
    #[serde(default)]
    pub dependency_kinds: Vec<String>,
    /// Minimum qualification tier. A QEMU receipt never satisfies a
    /// Machine 1 requirement.
    #[serde(default = "default_tier")]
    pub min_tier: Tier,
    /// Assertion IDs that must exist and pass on the chosen receipt.
    #[serde(default)]
    pub assertions: Vec<String>,
    /// Optional source constraints for assertions whose origin matters.
    #[serde(default)]
    pub assertion_sources: Vec<AssertionSource>,
    /// Exact mutation class required from the receipt, when specified.
    #[serde(default)]
    pub mutation: Option<crate::evidence::Mutation>,
    /// When true, the satisfying receipt must come from a clean tree.
    #[serde(default)]
    pub require_clean: bool,
    /// Status used when no satisfying receipt, or a required assertion on
    /// it, exists in the store.
    #[serde(default)]
    pub missing_as: MissingAs,
    /// Named external condition behind a known BLOCKED prerequisite.
    #[serde(default)]
    pub missing_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionSource {
    pub id: String,
    pub source: String,
}

/// Status reported when required proof is absent from the store.
/// BLOCKED names a known prerequisite that prevents the run from proceeding
/// legally or procedurally, for example the owner trust chain.
/// INCOMPLETE names proof that is simply absent so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MissingAs {
    #[serde(rename = "BLOCKED")]
    #[default]
    Blocked,
    #[serde(rename = "INCOMPLETE")]
    Incomplete,
}

impl MissingAs {
    fn verdict(self) -> Verdict {
        match self {
            MissingAs::Blocked => Verdict::Blocked,
            MissingAs::Incomplete => Verdict::Incomplete,
        }
    }
}

fn default_tier() -> Tier {
    Tier::HostTest
}

/// A named gate: the set of prerequisites that close it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub gate: String,
    #[serde(default)]
    pub description: String,
    pub requires: Vec<Requirement>,
}

impl Gate {
    /// Reject malformed manifests at load: unnamed or duplicated
    /// requirements, requirements naming both or neither of gate and kind,
    /// and source constraints on assertions the manifest never requires.
    pub fn validate(&self) -> Result<(), String> {
        if self.gate.trim().is_empty() {
            return Err("gate manifest has no gate name".to_string());
        }
        let mut names = BTreeSet::new();
        for req in &self.requires {
            if req.name.trim().is_empty() {
                return Err(format!("gate {} has an unnamed requirement", self.gate));
            }
            if req
                .gate_ref
                .as_ref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(format!(
                    "gate {} requirement {:?} has an empty gate reference",
                    self.gate, req.name
                ));
            }
            if !names.insert(req.name.clone()) {
                return Err(format!(
                    "gate {} lists requirement {:?} twice",
                    self.gate, req.name
                ));
            }
            match (&req.gate_ref, req.kind.trim().is_empty()) {
                (Some(_), false) => {
                    return Err(format!(
                        "gate {} requirement {:?} names both a gate and a receipt kind",
                        self.gate, req.name
                    ));
                }
                (None, true) => {
                    return Err(format!(
                        "gate {} requirement {:?} names neither a gate nor a receipt kind",
                        self.gate, req.name
                    ));
                }
                _ => {}
            }
            for constrained in &req.assertion_sources {
                if !req.assertions.iter().any(|a| a == &constrained.id) {
                    return Err(format!(
                        "gate {} requirement {:?} constrains source of assertion {:?} it never requires",
                        self.gate, req.name, constrained.id
                    ));
                }
            }
            if req.gate_ref.is_some() && !req.dependency_kinds.is_empty() {
                return Err(format!(
                    "gate {} nested gate requirement {:?} cannot declare receipt dependency kinds",
                    self.gate, req.name
                ));
            }
        }
        Ok(())
    }
}

/// Per requirement outcome.
#[derive(Debug, Clone)]
pub struct GateLine {
    pub name: String,
    pub status: Verdict,
    pub detail: String,
}

/// Whole gate outcome plus the responsible prerequisite lines.
#[derive(Debug, Clone)]
pub struct GateReport {
    pub gate: String,
    pub status: Verdict,
    pub lines: Vec<GateLine>,
}

fn worse(a: Verdict, b: Verdict) -> Verdict {
    fn rank(v: Verdict) -> u8 {
        match v {
            Verdict::Fail => 0,
            Verdict::Blocked => 1,
            Verdict::Incomplete => 2,
            Verdict::Skipped => 3,
            Verdict::Pass => 4,
        }
    }
    if rank(a) <= rank(b) {
        a
    } else {
        b
    }
}

/// Pick the best stored receipt of `kind`: PASS first, then highest tier.
pub fn best_candidate(store: &Path, kind: &str) -> Option<crate::evidence::Receipt> {
    let mut best: Option<crate::evidence::Receipt> = None;
    for id in list_receipts(store) {
        let receipt = load_receipt(store, &id).ok()?;
        if receipt.kind != kind {
            continue;
        }
        let take = match &best {
            None => true,
            Some(current) => {
                let rank = |r: &crate::evidence::Receipt| {
                    (r.result == Verdict::Pass, r.tier.rank(), r.timestamp)
                };
                rank(&receipt) > rank(current)
            }
        };
        if take {
            best = Some(receipt);
        }
    }
    best
}

/// Evaluate one requirement against the store.
fn eval_requirement(store: &Path, req: &Requirement, stack: &mut Vec<String>) -> GateLine {
    if let Some(child_name) = &req.gate_ref {
        if !req.kind.is_empty() {
            return GateLine {
                name: req.name.clone(),
                status: Verdict::Fail,
                detail: "manifest requirement cannot name both a gate and receipt kind".into(),
            };
        }
        if stack.contains(child_name) {
            return GateLine {
                name: req.name.clone(),
                status: Verdict::Fail,
                detail: format!("gate dependency cycle through {child_name}"),
            };
        }
        let child = match load_manifest(store, child_name, None) {
            Ok(g) => g,
            Err(e) => {
                return GateLine {
                    name: req.name.clone(),
                    status: Verdict::Blocked,
                    detail: format!("BLOCKED {child_name}: {e}"),
                }
            }
        };
        stack.push(child_name.clone());
        let report = evaluate_inner(&child, store, stack);
        stack.pop();
        let summary = report
            .lines
            .iter()
            .filter(|l| l.status != Verdict::Pass)
            .map(|l| format!("{}: {}", l.name, l.detail))
            .collect::<Vec<_>>()
            .join("; ");
        let summary = if summary.is_empty() {
            format!("all {} requirements PASS", report.lines.len())
        } else {
            summary
        };
        return GateLine {
            name: req.name.clone(),
            status: report.status,
            detail: format!("{}: {} ({})", child_name, report.status.as_str(), summary),
        };
    }
    if req.kind.is_empty() {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Fail,
            detail: "manifest requirement has neither gate_ref nor receipt kind".into(),
        };
    }
    let receipt = match best_candidate(store, &req.kind) {
        None => {
            let detail = req.missing_reason.as_ref().map_or_else(
                || format!("MISSING: no receipt of kind {:?} in store", req.kind),
                |reason| format!("MISSING: {reason}"),
            );
            return GateLine {
                name: req.name.clone(),
                status: req.missing_as.verdict(),
                detail,
            };
        }
        Some(r) => r,
    };
    if receipt.result != Verdict::Pass {
        return GateLine {
            name: req.name.clone(),
            status: receipt.result,
            detail: format!(
                "{}: receipt {} records {}",
                req.kind,
                &receipt.id[..16],
                receipt.result.as_str()
            ),
        };
    }
    if !tier_satisfies(req.min_tier, receipt.tier) {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Blocked,
            detail: format!(
                "{}: receipt tier {} cannot satisfy required {}",
                req.kind,
                receipt.tier.as_str(),
                req.min_tier.as_str()
            ),
        };
    }
    if let Some(resource) = &req.resource {
        if receipt.machine != *resource
            || receipt
                .lease
                .as_ref()
                .is_none_or(|lease| lease.resource != *resource)
        {
            return GateLine {
                name: req.name.clone(),
                status: Verdict::Incomplete,
                detail: format!(
                    "{}: receipt and exclusive hold must bind resource {:?}",
                    req.kind, resource
                ),
            };
        }
    }
    if req.repo.as_ref().is_some_and(|repo| receipt.repo != *repo) {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Fail,
            detail: format!(
                "{}: repository identity does not match required {:?}",
                req.kind,
                req.repo.as_deref().unwrap_or_default()
            ),
        };
    }
    if req
        .machine
        .as_ref()
        .is_some_and(|machine| receipt.machine != *machine)
    {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Fail,
            detail: format!(
                "{}: machine identity does not match required {:?}",
                req.kind,
                req.machine.as_deref().unwrap_or_default()
            ),
        };
    }
    if req
        .procedure_contains
        .as_ref()
        .is_some_and(|needle| !receipt.procedure.contains(needle))
    {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Fail,
            detail: format!(
                "{}: procedure identity must include {:?}",
                req.kind,
                req.procedure_contains.as_deref().unwrap_or_default()
            ),
        };
    }
    if receipt.input_artifacts.len() < req.min_input_artifacts {
        return GateLine {
            name: req.name.clone(),
            status: req.missing_as.verdict(),
            detail: format!(
                "{}: requires at least {} input artifact digests, receipt has {}",
                req.kind,
                req.min_input_artifacts,
                receipt.input_artifacts.len()
            ),
        };
    }
    if req.require_clean && receipt.dirty {
        return GateLine {
            name: req.name.clone(),
            status: Verdict::Blocked,
            detail: format!(
                "{}: gate requires a clean source tree but receipt {} is dirty",
                req.kind,
                &receipt.id[..16]
            ),
        };
    }
    let mut assertion_problems = Vec::new();
    let mut assertion_status = Verdict::Pass;
    for wanted in &req.assertions {
        match receipt.assertions.iter().find(|a| &a.id == wanted) {
            Some(a) if a.pass => {}
            Some(_) => {
                assertion_status = Verdict::Fail;
                assertion_problems.push(format!("assertion {wanted:?} failed"));
            }
            None => {
                if assertion_status != Verdict::Fail {
                    assertion_status = req.missing_as.verdict();
                }
                assertion_problems.push(format!("assertion {wanted:?} missing"));
            }
        }
    }
    for wanted in &req.assertion_sources {
        match receipt.assertions.iter().find(|a| a.id == wanted.id) {
            Some(a) if a.pass && a.source == wanted.source => {}
            Some(a) if !a.pass => {
                assertion_status = Verdict::Fail;
                assertion_problems.push(format!("assertion {:?} failed", wanted.id));
            }
            Some(a) => {
                assertion_status = Verdict::Fail;
                assertion_problems.push(format!(
                    "assertion {:?} source {:?} does not match required {:?}",
                    wanted.id, a.source, wanted.source
                ));
            }
            None => {
                if assertion_status != Verdict::Fail {
                    assertion_status = req.missing_as.verdict();
                }
                assertion_problems.push(format!("assertion {:?} missing", wanted.id));
            }
        }
    }
    if !assertion_problems.is_empty() {
        return GateLine {
            name: req.name.clone(),
            status: assertion_status,
            detail: format!("{}: {}", req.kind, assertion_problems.join("; ")),
        };
    }
    if let Some(mutation) = req.mutation {
        if receipt.declared_mutation != mutation || receipt.observed_mutation != mutation {
            return GateLine {
                name: req.name.clone(),
                status: Verdict::Fail,
                detail: format!(
                    "{}: declared/observed mutation must both be {}",
                    req.kind,
                    mutation.as_str()
                ),
            };
        }
    }
    if receipt.tier.is_hardware() {
        match crate::lease::lease_status(&receipt, store) {
            Some(r) if r.status != Verdict::Pass => {
                return GateLine {
                    name: req.name.clone(),
                    status: if r.status == Verdict::Fail {
                        Verdict::Fail
                    } else {
                        Verdict::Blocked
                    },
                    detail: format!("{}: {}", req.kind, r.reasons.join("; ")),
                };
            }
            _ => {}
        }
    }
    let chain_path = store
        .join(crate::evidence::RECEIPT_DIR)
        .join(format!("{}.json", receipt.id));
    match crate::chain::verify_chain(&chain_path, store) {
        Ok(chain) if chain.status == Verdict::Pass => {}
        Ok(chain) => {
            return GateLine {
                name: req.name.clone(),
                status: chain.status,
                detail: format!(
                    "{}: receipt dependency chain {}: {}",
                    req.kind,
                    chain.status.as_str(),
                    chain.reasons.join("; ")
                ),
            };
        }
        Err(e) => {
            return GateLine {
                name: req.name.clone(),
                status: Verdict::Incomplete,
                detail: format!("{}: cannot verify receipt dependency chain: {e}", req.kind),
            };
        }
    }
    if !req.dependency_kinds.is_empty() {
        let chain = match crate::chain::verify_chain(&chain_path, store) {
            Ok(chain) => chain,
            Err(e) => {
                return GateLine {
                    name: req.name.clone(),
                    status: Verdict::Incomplete,
                    detail: format!("{}: cannot resolve receipt dependencies: {e}", req.kind),
                }
            }
        };
        let missing: Vec<&str> = req
            .dependency_kinds
            .iter()
            .filter(|kind| !chain.lines.iter().any(|line| &line.kind == *kind))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return GateLine {
                name: req.name.clone(),
                status: req.missing_as.verdict(),
                detail: format!(
                    "{}: required receipt dependency kinds missing from verified chain: {}",
                    req.kind,
                    missing.join(", ")
                ),
            };
        }
    }
    GateLine {
        name: req.name.clone(),
        status: Verdict::Pass,
        detail: format!(
            "{}: receipt {} tier {}",
            req.kind,
            &receipt.id[..16],
            receipt.tier.as_str()
        ),
    }
}

/// Evaluate a gate manifest against the store.
pub fn evaluate(manifest: &Gate, store: &Path) -> GateReport {
    evaluate_inner(manifest, store, &mut vec![manifest.gate.clone()])
}

fn evaluate_inner(manifest: &Gate, store: &Path, stack: &mut Vec<String>) -> GateReport {
    let mut status = Verdict::Pass;
    let mut lines = Vec::new();
    for req in &manifest.requires {
        let line = eval_requirement(store, req, stack);
        status = worse(status, line.status);
        lines.push(line);
    }
    // No requirements means nothing is proven.
    if manifest.requires.is_empty() {
        status = Verdict::Incomplete;
    }
    GateReport {
        gate: manifest.gate.clone(),
        status,
        lines,
    }
}

/// Load a manifest by gate name: explicit path first, then the store's
/// gate directory.
pub fn load_manifest(store: &Path, gate: &str, explicit: Option<&Path>) -> Result<Gate, String> {
    if let Some(path) = explicit {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read gate file: {e}"))?;
        let manifest: Gate =
            serde_json::from_str(&text).map_err(|e| format!("gate manifest invalid: {e}"))?;
        if manifest.gate != gate {
            return Err(format!(
                "gate manifest name {:?} does not match requested {gate:?}",
                manifest.gate
            ));
        }
        manifest.validate()?;
        return Ok(manifest);
    }
    let text = if let Some(text) = embedded_manifest(gate) {
        text.to_string()
    } else {
        let path = store.join(GATE_DIR).join(format!("{gate}.json"));
        std::fs::read_to_string(&path)
            .map_err(|_| format!("no manifest for gate {gate:?}: pass --gate <file>"))?
    };
    let manifest: Gate =
        serde_json::from_str(&text).map_err(|e| format!("gate manifest invalid: {e}"))?;
    if manifest.gate != gate {
        return Err(format!(
            "gate manifest name {:?} does not match requested {gate:?}",
            manifest.gate
        ));
    }
    manifest.validate()?;
    Ok(manifest)
}

/// Built in example manifests. These are examples of the manifest shape,
/// not claims that any gate has passed.
pub const SEED_0B_EXAMPLE: &str = include_str!("../gates/SEED-0B.json");
pub const SEED_0B_QEMU: &str = include_str!("../gates/SEED0B_QEMU.json");
pub const STORE_V1_FORMAT_PR_MERGE: &str = include_str!("../gates/STORE_V1_FORMAT_PR_MERGE.json");
pub const P3_STORE_QEMU: &str = include_str!("../gates/P3_STORE_QEMU.json");
pub const P3_STORE_MACHINE1: &str = include_str!("../gates/P3_STORE_MACHINE1.json");
pub const P3_DEVELOPMENT_QEMU_ENTRY: &str = include_str!("../gates/P3_DEVELOPMENT_QEMU_ENTRY.json");
pub const P3_MACHINE1_WRITE_ENTRY: &str = include_str!("../gates/P3_MACHINE1_WRITE_ENTRY.json");
pub const STORE_V1_FORMAT: &str = include_str!("../gates/STORE_V1_FORMAT.json");
pub const STORE_V1_HOST: &str = include_str!("../gates/STORE_V1_HOST.json");
pub const STORE_V1_QEMU: &str = include_str!("../gates/STORE_V1_QEMU.json");

fn embedded_manifest(gate: &str) -> Option<&'static str> {
    match gate {
        "SEED-0B" => Some(SEED_0B_EXAMPLE),
        "SEED0B_QEMU" => Some(SEED_0B_QEMU),
        "STORE_V1_FORMAT_PR_MERGE" => Some(STORE_V1_FORMAT_PR_MERGE),
        "P3_STORE_QEMU" => Some(P3_STORE_QEMU),
        "P3_STORE_MACHINE1" => Some(P3_STORE_MACHINE1),
        "P3_DEVELOPMENT_QEMU_ENTRY" => Some(P3_DEVELOPMENT_QEMU_ENTRY),
        "P3_MACHINE1_WRITE_ENTRY" => Some(P3_MACHINE1_WRITE_ENTRY),
        "STORE_V1_FORMAT" => Some(STORE_V1_FORMAT),
        "STORE_V1_HOST" => Some(STORE_V1_HOST),
        "STORE_V1_QEMU" => Some(STORE_V1_QEMU),
        _ => None,
    }
}

/// Materialize the example manifests into a directory.
pub fn write_examples(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (name, text) in [
        ("SEED-0B.json", SEED_0B_EXAMPLE),
        ("SEED0B_QEMU.json", SEED_0B_QEMU),
        ("STORE_V1_FORMAT_PR_MERGE.json", STORE_V1_FORMAT_PR_MERGE),
        ("P3_STORE_QEMU.json", P3_STORE_QEMU),
        ("P3_STORE_MACHINE1.json", P3_STORE_MACHINE1),
        ("P3_DEVELOPMENT_QEMU_ENTRY.json", P3_DEVELOPMENT_QEMU_ENTRY),
        ("P3_MACHINE1_WRITE_ENTRY.json", P3_MACHINE1_WRITE_ENTRY),
        ("STORE_V1_FORMAT.json", STORE_V1_FORMAT),
        ("STORE_V1_HOST.json", STORE_V1_HOST),
        ("STORE_V1_QEMU.json", STORE_V1_QEMU),
    ] {
        // Validate the shipped shape before writing.
        let manifest: Gate =
            serde_json::from_str(text).map_err(|e| format!("built in manifest broken: {e}"))?;
        manifest
            .validate()
            .map_err(|e| format!("built in manifest invalid: {e}"))?;
        let path = dir.join(name);
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        out.push(path);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::fixtures::{sample, sealed};
    use crate::evidence::store_receipt;

    fn store_qemu(store: &Path, kind: &str, assertions: Vec<&str>) {
        let mut r = sample();
        r.kind = kind.to_string();
        let ids = if assertions.is_empty() {
            vec!["recorded"]
        } else {
            assertions
        };
        r.assertions = ids
            .into_iter()
            .map(|id| crate::evidence::Assertion {
                id: id.to_string(),
                expected: "yes".to_string(),
                observed: "yes".to_string(),
                pass: true,
                source: "example".to_string(),
                note: String::new(),
            })
            .collect();
        store_receipt(store, &sealed(r)).unwrap();
    }

    fn manifest() -> Gate {
        Gate {
            gate: "SEED-0B".to_string(),
            description: "example".to_string(),
            requires: vec![
                Requirement {
                    name: "M3".to_string(),
                    kind: "m3".to_string(),
                    gate_ref: None,
                    resource: None,
                    repo: None,
                    machine: None,
                    procedure_contains: None,
                    min_input_artifacts: 0,
                    dependency_kinds: vec![],
                    min_tier: Tier::HostTest,
                    assertions: vec![],
                    assertion_sources: vec![],
                    mutation: None,
                    require_clean: false,
                    missing_as: MissingAs::Blocked,
                    missing_reason: None,
                },
                Requirement {
                    name: "SEED-0B-QEMU".to_string(),
                    kind: "seed-0b-qemu".to_string(),
                    gate_ref: None,
                    resource: None,
                    repo: None,
                    machine: None,
                    procedure_contains: None,
                    min_input_artifacts: 0,
                    dependency_kinds: vec![],
                    min_tier: Tier::Qemu,
                    assertions: vec!["artifact_signature_valid".to_string()],
                    assertion_sources: vec![],
                    mutation: None,
                    require_clean: true,
                    missing_as: MissingAs::Blocked,
                    missing_reason: None,
                },
                Requirement {
                    name: "SEED-0B-MACHINE1".to_string(),
                    kind: "seed-0b-machine1".to_string(),
                    gate_ref: None,
                    resource: None,
                    repo: None,
                    machine: None,
                    procedure_contains: None,
                    min_input_artifacts: 0,
                    dependency_kinds: vec![],
                    min_tier: Tier::Machine1Attended,
                    assertions: vec![],
                    assertion_sources: vec![],
                    mutation: None,
                    require_clean: true,
                    missing_as: MissingAs::Blocked,
                    missing_reason: None,
                },
            ],
        }
    }

    #[test]
    fn gate_reports_missing() {
        let store = crate::testutil::temp_dir("gate-missing");
        let report = evaluate(&manifest(), &store);
        assert_eq!(report.status, Verdict::Blocked);
        assert!(report.lines.iter().all(|l| l.status == Verdict::Blocked));
        assert!(report.lines[0].detail.contains("MISSING"));
    }

    #[test]
    fn qemu_receipt_cannot_satisfy_machine1_requirement() {
        let store = crate::testutil::temp_dir("gate-tier");
        store_qemu(&store, "m3", vec![]);
        store_qemu(&store, "seed-0b-qemu", vec!["artifact_signature_valid"]);
        // A QEMU tier receipt filed under the machine1 kind must not close
        // the physical requirement.
        let mut r = sample();
        r.kind = "seed-0b-machine1".to_string();
        store_receipt(&store, &sealed(r)).unwrap();
        let report = evaluate(&manifest(), &store);
        assert_eq!(report.status, Verdict::Blocked);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "SEED-0B-MACHINE1")
            .unwrap();
        assert!(line.detail.contains("cannot satisfy"));
    }

    #[test]
    fn dirty_tree_fails_clean_requirement() {
        let store = crate::testutil::temp_dir("gate-dirty");
        let mut r = sample();
        r.kind = "m3".to_string();
        store_receipt(&store, &sealed(r)).unwrap();
        let mut q = sample();
        q.kind = "seed-0b-qemu".to_string();
        q.dirty = true;
        store_receipt(&store, &sealed(q)).unwrap();
        let report = evaluate(&manifest(), &store);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "SEED-0B-QEMU")
            .unwrap();
        assert_eq!(line.status, Verdict::Blocked);
        assert!(line.detail.contains("clean source tree"));
    }

    #[test]
    fn failed_assertion_fails_gate() {
        let store = crate::testutil::temp_dir("gate-assert");
        store_qemu(&store, "m3", vec![]);
        let mut q = sample();
        q.kind = "seed-0b-qemu".to_string();
        q.assertions[0].pass = false;
        store_receipt(&store, &sealed(q)).unwrap();
        let report = evaluate(&manifest(), &store);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "SEED-0B-QEMU")
            .unwrap();
        assert_eq!(line.status, Verdict::Fail);
    }

    #[test]
    fn blocked_receipt_cannot_satisfy_a_gate() {
        let store = crate::testutil::temp_dir("gate-blocked-receipt");
        let mut r = sample();
        r.kind = "seed-0b-qemu".to_string();
        r.result = Verdict::Blocked;
        store_receipt(&store, &sealed(r)).unwrap();
        let report = evaluate(&manifest(), &store);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "SEED-0B-QEMU")
            .unwrap();
        assert_eq!(line.status, Verdict::Blocked);
    }

    #[test]
    fn qemu_bootnext_mock_source_cannot_prove_firmware_variable_behavior() {
        let store = crate::testutil::temp_dir("gate-bootnext-source");
        let mut r = sample();
        r.kind = "rollback".to_string();
        r.assertions = vec![crate::evidence::Assertion {
            id: "bootnext_consumed".into(),
            expected: "unset".into(),
            observed: "unset".into(),
            pass: true,
            source: "rollback_state_machine_mock".into(),
            note: String::new(),
        }];
        store_receipt(&store, &sealed(r)).unwrap();
        let gate = Gate {
            gate: "rollback-source-test".into(),
            description: String::new(),
            requires: vec![Requirement {
                name: "firmware BootNext".into(),
                kind: "rollback".into(),
                gate_ref: None,
                resource: None,
                repo: None,
                machine: None,
                procedure_contains: None,
                min_input_artifacts: 0,
                dependency_kinds: vec![],
                min_tier: Tier::Qemu,
                assertions: vec!["bootnext_consumed".into()],
                assertion_sources: vec![AssertionSource {
                    id: "bootnext_consumed".into(),
                    source: "aavmf_variables".into(),
                }],
                mutation: None,
                require_clean: true,
                missing_as: MissingAs::Incomplete,
                missing_reason: None,
            }],
        };
        assert_eq!(evaluate(&gate, &store).status, Verdict::Fail);
    }

    #[test]
    fn nested_gate_cycle_fails_closed() {
        let store = crate::testutil::temp_dir("gate-cycle");
        let dir = store.join(GATE_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("A.json"),
            r#"{"gate":"A","requires":[{"name":"B","kind":"","gate_ref":"B"}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("B.json"),
            r#"{"gate":"B","requires":[{"name":"A","kind":"","gate_ref":"A"}]}"#,
        )
        .unwrap();
        let a = load_manifest(&store, "A", None).unwrap();
        assert_eq!(evaluate(&a, &store).status, Verdict::Fail);
    }

    #[test]
    fn p3_gates_preserve_qemu_and_machine1_distinction() {
        let store = crate::testutil::temp_dir("gate-p3-split");
        let qemu = load_manifest(&store, "P3_DEVELOPMENT_QEMU_ENTRY", None).unwrap();
        let physical = load_manifest(&store, "P3_MACHINE1_WRITE_ENTRY", None).unwrap();
        let qemu_report = evaluate(&qemu, &store);
        let physical_report = evaluate(&physical, &store);
        assert_eq!(qemu_report.status, Verdict::Incomplete);
        assert_eq!(physical_report.status, Verdict::Blocked);
        let trust = physical_report
            .lines
            .iter()
            .find(|line| line.name == "OWNER_SECURE_BOOT_TRUST_CHAIN")
            .unwrap();
        assert_eq!(trust.status, Verdict::Blocked);
        assert!(trust
            .detail
            .contains("HARDWARE_QUALIFICATION_BLOCKED_BY_TRUST_CHAIN"));
        let bounded = physical_report
            .lines
            .iter()
            .find(|line| line.name == "BOUNDED_MACHINE1_STORE_REGION")
            .unwrap();
        assert_eq!(bounded.status, Verdict::Incomplete);
        assert_eq!(
            physical.requires[0].gate_ref.as_deref(),
            Some("P3_DEVELOPMENT_QEMU_ENTRY")
        );
    }

    #[test]
    fn checked_in_gate_cannot_be_substituted_by_store_copy() {
        let store = crate::testutil::temp_dir("gate-no-substitution");
        let dir = store.join(GATE_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("P3_DEVELOPMENT_QEMU_ENTRY.json"),
            r#"{"gate":"P3_DEVELOPMENT_QEMU_ENTRY","requires":[]}"#,
        )
        .unwrap();
        let manifest = load_manifest(&store, "P3_DEVELOPMENT_QEMU_ENTRY", None).unwrap();
        assert_eq!(manifest.requires.len(), 4);
    }

    #[test]
    fn narrow_store_format_merge_gate_does_not_require_seed_or_rollback() {
        let store = crate::testutil::temp_dir("gate-format-merge");
        store_assertion_receipt(
            &store,
            "adr0015-format",
            Tier::HostTest,
            "host",
            "scripts/check_adr0015.sh",
            pass_assertions(&["adr0015_accepted_frozen"], "adr0015"),
        );
        store_assertion_receipt(
            &store,
            "store-v1-format-golden-vectors",
            Tier::HostTest,
            "host",
            "cargo test -p aien-system-store format_golden_vectors",
            pass_assertions(
                &["store_format_v1_valid", "store_golden_vectors_match"],
                "store-format-tests",
            ),
        );
        store_assertion_receipt(
            &store,
            "strict-main-verification",
            Tier::HostTest,
            "host",
            "scripts/verify_all.sh",
            pass_assertions(&["strict_main_pass"], "verify-all"),
        );
        let manifest = load_manifest(&store, "STORE_V1_FORMAT_PR_MERGE", None).unwrap();
        assert_eq!(evaluate(&manifest, &store).status, Verdict::Pass);
        assert!(manifest.requires.iter().all(|requirement| {
            requirement.kind != "seed0b-qemu" && requirement.kind != "m0-native-rollback-qemu"
        }));
    }

    #[test]
    fn gate_rejects_receipt_with_missing_chain_dependency() {
        let store = crate::testutil::temp_dir("gate-missing-chain");
        let mut r = sample();
        r.kind = "m3".to_string();
        r.dependencies = vec!["f".repeat(64)];
        store_receipt(&store, &sealed(r)).unwrap();
        let manifest = Gate {
            gate: "CHAIN-DEMO".to_string(),
            description: String::new(),
            requires: vec![req("M3", "m3", Tier::HostTest, vec![], MissingAs::Blocked)],
        };
        assert_eq!(evaluate(&manifest, &store).status, Verdict::Incomplete);
    }

    #[test]
    fn mutation_class_mismatch_fails_gate() {
        use crate::evidence::Mutation;
        let store = crate::testutil::temp_dir("gate-mutation-mismatch");
        store_qemu(&store, "store-qemu", vec!["store_previous_or_new_only"]);
        let mut requirement = req(
            "Store QEMU",
            "store-qemu",
            Tier::Qemu,
            vec!["store_previous_or_new_only"],
            MissingAs::Incomplete,
        );
        requirement.mutation = Some(Mutation::BoundedTestRegionWrite);
        let report = evaluate(
            &Gate {
                gate: "MUTATION-DEMO".into(),
                description: String::new(),
                requires: vec![requirement],
            },
            &store,
        );
        assert_eq!(report.status, Verdict::Fail);
        assert!(report.lines[0]
            .detail
            .contains("declared/observed mutation"));
    }

    #[test]
    fn hardware_gate_requires_the_machine1_hold() {
        let store = crate::testutil::temp_dir("gate-no-hold");
        store_assertion_receipt(
            &store,
            "machine-run",
            Tier::Machine1Attended,
            "machine-1",
            "scripts/qualify.sh",
            pass_assertions(&["operator_attended"], "operator"),
        );
        let mut requirement = req(
            "Machine 1 run",
            "machine-run",
            Tier::Machine1Attended,
            vec!["operator_attended"],
            MissingAs::Incomplete,
        );
        requirement.resource = Some("machine-1".into());
        let report = evaluate(
            &Gate {
                gate: "MACHINE-HOLD-DEMO".into(),
                description: String::new(),
                requires: vec![requirement],
            },
            &store,
        );
        assert_eq!(report.status, Verdict::Incomplete);
        assert!(report.lines[0].detail.contains("exclusive hold"));
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

    fn assertion(id: &str, source: &str, pass: bool) -> crate::evidence::Assertion {
        crate::evidence::Assertion {
            id: id.to_string(),
            expected: "yes".to_string(),
            observed: "yes".to_string(),
            pass,
            source: source.to_string(),
            note: String::new(),
        }
    }

    /// Materialize the embedded P3 manifests into a fresh store.
    fn p3_store(tag: &str) -> std::path::PathBuf {
        let store = crate::testutil::temp_dir(tag);
        write_examples(&store.join(GATE_DIR)).unwrap();
        store
    }

    #[test]
    fn machine1_gate_names_its_blockers_on_an_empty_store() {
        let store = p3_store("gate-m1-empty");
        let manifest = load_manifest(&store, "P3_MACHINE1_WRITE_ENTRY", None).unwrap();
        let report = evaluate(&manifest, &store);
        assert_eq!(report.status, Verdict::Blocked);
        let trust = report
            .lines
            .iter()
            .find(|l| l.name == "OWNER_SECURE_BOOT_TRUST_CHAIN")
            .unwrap();
        assert_eq!(trust.status, Verdict::Blocked);
        assert!(trust
            .detail
            .contains("HARDWARE_QUALIFICATION_BLOCKED_BY_TRUST_CHAIN"));
        let rollback = report
            .lines
            .iter()
            .find(|l| l.name == "M0_NATIVE_ROLLBACK_MACHINE1")
            .unwrap();
        assert_eq!(rollback.status, Verdict::Blocked);
        // The bounded region is proof simply absent so far, not a known
        // procedural block.
        let region = report
            .lines
            .iter()
            .find(|l| l.name == "BOUNDED_MACHINE1_STORE_REGION")
            .unwrap();
        assert_eq!(region.status, Verdict::Incomplete);
    }

    fn store_assertion_receipt(
        store: &Path,
        kind: &str,
        tier: Tier,
        machine: &str,
        procedure: &str,
        assertions: Vec<crate::evidence::Assertion>,
    ) {
        let mut r = sample();
        r.kind = kind.to_string();
        r.tier = tier;
        r.env_class = tier.as_str().to_string();
        r.machine = machine.to_string();
        r.procedure = procedure.to_string();
        r.input_artifacts = vec![
            "blake3:".to_string() + &"b".repeat(64),
            "sha256:".to_string() + &"c".repeat(64),
        ];
        r.assertions = assertions;
        store_receipt(store, &sealed(r)).unwrap();
    }

    const ROLLBACK_BRANCHES: &[&str] = &[
        "rollback_harness_version_recorded",
        "qemu_aavmf_environment_recorded",
        "rollback_normal_pass",
        "rollback_fault_pass",
        "rollback_timeout_pass",
        "rollback_rejected_pass",
        "rollback_absent_pass",
        "rollback_repeat_boot_pass",
        "m0_native_rollback_qemu_pass",
    ];

    const ROLLBACK_MACHINE: &str = "qemu-aarch64-aavmf";
    const ROLLBACK_PROCEDURE: &str = "scripts/qemu_native_rollback_test.sh@0.9.1";

    fn pass_assertions(ids: &[&str], source: &str) -> Vec<crate::evidence::Assertion> {
        ids.iter().map(|id| assertion(id, source, true)).collect()
    }

    /// Satisfy the QEMU development gate with real stored receipts, then
    /// prove the Machine 1 gate still reports BLOCKED for the trust chain.
    /// This is the done condition distinction in one test.
    #[test]
    fn qemu_gate_passes_while_machine1_gate_stays_blocked() {
        let store = p3_store("gate-qemu-vs-m1");
        store_assertion_receipt(
            &store,
            "m0-native-rollback-qemu",
            Tier::Qemu,
            ROLLBACK_MACHINE,
            ROLLBACK_PROCEDURE,
            [
                pass_assertions(ROLLBACK_BRANCHES, "rollback-verify"),
                vec![
                    assertion("bootnext_consumed", "aavmf_variables", true),
                    assertion("default_boot_unchanged", "aavmf_variables", true),
                ],
            ]
            .concat(),
        );
        store_assertion_receipt(
            &store,
            "seed0b-qemu",
            Tier::Qemu,
            "qemu-virt-aarch64",
            "scripts/qemu_seed0b.sh",
            pass_assertions(
                &[
                    "seed0b_signature_valid",
                    "granted_subset_requested",
                    "unauthorized_write_denied",
                    "forged_handle_denied",
                    "all_task_frames_reclaimed",
                ],
                "seed0b-qemu",
            ),
        );
        store_assertion_receipt(
            &store,
            "adr0015-format",
            Tier::HostTest,
            "host",
            "scripts/check_adr0015.sh",
            pass_assertions(&["adr0015_accepted_frozen"], "adr0015"),
        );
        store_assertion_receipt(
            &store,
            "strict-main-verification",
            Tier::HostTest,
            "host",
            "scripts/verify_all.sh",
            pass_assertions(&["strict_main_pass"], "verify-all"),
        );
        let qemu = load_manifest(&store, "P3_DEVELOPMENT_QEMU_ENTRY", None).unwrap();
        let qemu_report = evaluate(&qemu, &store);
        assert_eq!(qemu_report.status, Verdict::Pass, "{qemu_report:?}");
        let m1 = load_manifest(&store, "P3_MACHINE1_WRITE_ENTRY", None).unwrap();
        let m1_report = evaluate(&m1, &store);
        assert_eq!(m1_report.status, Verdict::Blocked);
        let sub = m1_report
            .lines
            .iter()
            .find(|l| l.name == "P3_DEVELOPMENT_QEMU_ENTRY")
            .unwrap();
        assert_eq!(sub.status, Verdict::Pass);
        let trust = m1_report
            .lines
            .iter()
            .find(|l| l.name == "OWNER_SECURE_BOOT_TRUST_CHAIN")
            .unwrap();
        assert_eq!(trust.status, Verdict::Blocked);
    }

    #[test]
    fn mock_source_cannot_prove_firmware_bootnext() {
        let store = p3_store("gate-mock");
        store_assertion_receipt(
            &store,
            "m0-native-rollback-qemu",
            Tier::Qemu,
            ROLLBACK_MACHINE,
            ROLLBACK_PROCEDURE,
            [
                pass_assertions(ROLLBACK_BRANCHES, "rollback-verify"),
                vec![
                    assertion("bootnext_consumed", "state-machine-mock", true),
                    assertion("default_boot_unchanged", "aavmf_variables", true),
                ],
            ]
            .concat(),
        );
        let manifest = load_manifest(&store, "P3_DEVELOPMENT_QEMU_ENTRY", None).unwrap();
        let report = evaluate(&manifest, &store);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "M0_NATIVE_ROLLBACK_QEMU")
            .unwrap();
        assert_eq!(line.status, Verdict::Fail);
        assert!(line.detail.contains("aavmf_variables"));
    }

    #[test]
    fn blocked_receipt_cannot_satisfy_a_requirement() {
        let store = crate::testutil::temp_dir("gate-blocked");
        let mut r = sample();
        r.kind = "m3".to_string();
        r.result = Verdict::Blocked;
        r.assertions = vec![assertion("recorded", "example", true)];
        let r = sealed(r);
        // BLOCKED receipts carry no chain burden here; store then evaluate.
        store_receipt(&store, &r).unwrap();
        let manifest = Gate {
            gate: "BLOCKED-DEMO".to_string(),
            description: String::new(),
            requires: vec![req("M3", "m3", Tier::HostTest, vec![], MissingAs::Blocked)],
        };
        let report = evaluate(&manifest, &store);
        assert_eq!(report.status, Verdict::Blocked);
        assert!(report.lines[0].detail.contains("records BLOCKED"));
    }

    #[test]
    fn qemu_receipt_cannot_pose_as_machine1_rollback() {
        let store = p3_store("gate-rollback-tier");
        store_assertion_receipt(
            &store,
            "m0-native-rollback-machine1",
            Tier::Qemu,
            "machine-1",
            "scripts/rollback_machine1.sh",
            pass_assertions(&["secure_boot_unchanged"], "rollback-verify"),
        );
        let manifest = load_manifest(&store, "P3_MACHINE1_WRITE_ENTRY", None).unwrap();
        let report = evaluate(&manifest, &store);
        let line = report
            .lines
            .iter()
            .find(|l| l.name == "M0_NATIVE_ROLLBACK_MACHINE1")
            .unwrap();
        assert_eq!(line.status, Verdict::Blocked);
        assert!(line.detail.contains("cannot satisfy"));
    }

    #[test]
    fn manifest_cycles_fail_closed() {
        let store = crate::testutil::temp_dir("gate-cycle");
        let dir = store.join(GATE_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, child) in [("GATE-A", "GATE-B"), ("GATE-B", "GATE-A")] {
            let manifest = Gate {
                gate: name.to_string(),
                description: String::new(),
                requires: vec![Requirement {
                    name: format!("child {child}"),
                    kind: String::new(),
                    gate_ref: Some(child.to_string()),
                    resource: None,
                    repo: None,
                    machine: None,
                    procedure_contains: None,
                    min_input_artifacts: 0,
                    dependency_kinds: vec![],
                    min_tier: Tier::HostTest,
                    assertions: vec![],
                    assertion_sources: vec![],
                    mutation: None,
                    require_clean: false,
                    missing_as: MissingAs::Blocked,
                    missing_reason: None,
                }],
            };
            std::fs::write(
                dir.join(format!("{name}.json")),
                serde_json::to_string_pretty(&manifest).unwrap(),
            )
            .unwrap();
        }
        let manifest = load_manifest(&store, "GATE-A", None).unwrap();
        let report = evaluate(&manifest, &store);
        assert_eq!(report.status, Verdict::Fail);
        assert!(report.lines[0].detail.contains("cycle"));
    }

    #[test]
    fn malformed_manifests_rejected_at_load() {
        let both = Gate {
            gate: "BAD".to_string(),
            description: String::new(),
            requires: vec![Requirement {
                name: "x".to_string(),
                kind: "m3".to_string(),
                gate_ref: Some("OTHER".to_string()),
                resource: None,
                repo: None,
                machine: None,
                procedure_contains: None,
                min_input_artifacts: 0,
                dependency_kinds: vec![],
                min_tier: Tier::HostTest,
                assertions: vec![],
                assertion_sources: vec![],
                mutation: None,
                require_clean: false,
                missing_as: MissingAs::Blocked,
                missing_reason: None,
            }],
        };
        assert!(both.validate().is_err());
        let stray_source = Gate {
            gate: "BAD".to_string(),
            description: String::new(),
            requires: vec![Requirement {
                name: "x".to_string(),
                kind: "m3".to_string(),
                gate_ref: None,
                resource: None,
                repo: None,
                machine: None,
                procedure_contains: None,
                min_input_artifacts: 0,
                dependency_kinds: vec![],
                min_tier: Tier::HostTest,
                assertions: vec!["a".to_string()],
                assertion_sources: vec![AssertionSource {
                    id: "b".to_string(),
                    source: "mock".to_string(),
                }],
                mutation: None,
                require_clean: false,
                missing_as: MissingAs::Blocked,
                missing_reason: None,
            }],
        };
        assert!(stray_source.validate().is_err());
    }

    #[test]
    fn shipped_examples_parse() {
        let seed: Gate = serde_json::from_str(SEED_0B_EXAMPLE).unwrap();
        assert_eq!(seed.gate, "SEED-0B");
        assert!(!seed.requires.is_empty());
        let seed_qemu: Gate = serde_json::from_str(SEED_0B_QEMU).unwrap();
        assert_eq!(seed_qemu.gate, "SEED0B_QEMU");
        let format: Gate = serde_json::from_str(STORE_V1_FORMAT_PR_MERGE).unwrap();
        assert_eq!(format.gate, "STORE_V1_FORMAT_PR_MERGE");
        let p3_qemu: Gate = serde_json::from_str(P3_STORE_QEMU).unwrap();
        assert_eq!(p3_qemu.gate, "P3_STORE_QEMU");
        let p3_machine: Gate = serde_json::from_str(P3_STORE_MACHINE1).unwrap();
        assert_eq!(p3_machine.gate, "P3_STORE_MACHINE1");
        let p3: Gate = serde_json::from_str(P3_DEVELOPMENT_QEMU_ENTRY).unwrap();
        assert_eq!(p3.gate, "P3_DEVELOPMENT_QEMU_ENTRY");
        assert!(!p3.requires.is_empty());
        let physical: Gate = serde_json::from_str(P3_MACHINE1_WRITE_ENTRY).unwrap();
        assert_eq!(physical.gate, "P3_MACHINE1_WRITE_ENTRY");
        let store_format: Gate = serde_json::from_str(STORE_V1_FORMAT).unwrap();
        assert_eq!(store_format.gate, "STORE_V1_FORMAT");
        store_format.validate().unwrap();
        let store_host: Gate = serde_json::from_str(STORE_V1_HOST).unwrap();
        assert_eq!(store_host.gate, "STORE_V1_HOST");
        store_host.validate().unwrap();
        let store_qemu: Gate = serde_json::from_str(STORE_V1_QEMU).unwrap();
        assert_eq!(store_qemu.gate, "STORE_V1_QEMU");
        store_qemu.validate().unwrap();
    }
}
