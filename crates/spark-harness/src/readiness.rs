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

//! Artifact readiness and operation classes (Headstart M2, issue #535).
//!
//! A producer task can make its output ready for consumers at two levels:
//!
//! - [`ReadinessLevel::InterfaceVerified`]: the producer published an
//!   immutable interface artifact through [`HarnessEngine::publish_interface`].
//!   The harness recomputes the digest itself, mints an aien-proof receipt of
//!   kind [`INTERFACE_KIND`] (distinct from the M1 completion kind) and records
//!   it as the task's current interface. Consumers may only run the reversible,
//!   sandboxed [`OperationClass`] entries against it.
//! - [`ReadinessLevel::FinalVerified`]: M1 [`verify_completion`] passes for the
//!   producer's current generation.
//!
//! [`ReadinessLevel::Revoked`] permits nothing.
//!
//! An [`ArtifactReadiness`] can only be built by [`verify_interface`] or
//! [`verify_final`], which re-read the receipt from the harness store. It has
//! no public constructor and no `Deserialize`, so a caller cannot hand the
//! scheduler a readiness value, a receipt or a pass flag.
//!
//! The same trust limits as M1 apply (see `completion.rs` module docs):
//! receipts are unsigned, so authenticity rests on ownership of the store
//! plus recomputation of the receipt identity and every binding.

use crate::completion::{
    artifact_digest, check_identifier, host_identity, require, single_ref, task_key,
    verify_completion, write_atomic, CompletionError,
};
use crate::HarnessEngine;
use aien_proof::evidence::{self, Assertion, Mutation, Receipt, Tier, Verdict};
use aien_proof::lock::FileLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::fmt;
use std::fs;
use std::path::Path;

/// Receipt `kind` for published interface artifacts.
pub const INTERFACE_KIND: &str = "spark-harness-interface";
/// Honest tier: host-executed deterministic checks.
pub const INTERFACE_TIER: Tier = Tier::HostTest;
/// Subdirectory of the harness store holding current interface records.
pub const INTERFACE_DIR: &str = "interfaces";
const ASSERTION_SOURCE: &str = "spark-harness";

/// Closed set of operation classes a scheduled task may ask to run.
///
/// Only [`OperationClass::is_reversible_preparation`] classes may run while a
/// dependency is merely interface-verified. Every match on this enum is
/// exhaustive on purpose: a new class must be classified explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationClass {
    /// Reading sources and artifacts, no writes outside scratch space.
    ReadOnlyAnalysis,
    /// Builds or tests in a throwaway sandbox whose output is discarded on
    /// revocation.
    SandboxedCheck,
    /// Committing work (World commit). Irreversible.
    Commit,
    /// Publishing an artifact others may consume. Irreversible.
    Publish,
    /// Mutating persistent state. Irreversible.
    PersistentMutation,
    /// Any effect outside the host (network, devices, people). Irreversible.
    ExternalEffect,
}

impl OperationClass {
    pub const ALL: [OperationClass; 6] = [
        OperationClass::ReadOnlyAnalysis,
        OperationClass::SandboxedCheck,
        OperationClass::Commit,
        OperationClass::Publish,
        OperationClass::PersistentMutation,
        OperationClass::ExternalEffect,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OperationClass::ReadOnlyAnalysis => "read_only_analysis",
            OperationClass::SandboxedCheck => "sandboxed_check",
            OperationClass::Commit => "commit",
            OperationClass::Publish => "publish",
            OperationClass::PersistentMutation => "persistent_mutation",
            OperationClass::ExternalEffect => "external_effect",
        }
    }

    /// Exact, case-sensitive match. Anything else is unknown and refused.
    pub fn parse(s: &str) -> Result<Self, UnknownOperation> {
        OperationClass::ALL
            .into_iter()
            .find(|op| op.as_str() == s)
            .ok_or_else(|| UnknownOperation(s.to_string()))
    }

    /// True only for classes whose effects stay inside a sandbox and can be
    /// thrown away when the interface they used is revoked.
    pub fn is_reversible_preparation(self) -> bool {
        match self {
            OperationClass::ReadOnlyAnalysis | OperationClass::SandboxedCheck => true,
            OperationClass::Commit
            | OperationClass::Publish
            | OperationClass::PersistentMutation
            | OperationClass::ExternalEffect => false,
        }
    }
}

/// An operation name that is not in [`OperationClass`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownOperation(pub String);

impl fmt::Display for UnknownOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown operation class {:?}, refused", self.0)
    }
}

impl std::error::Error for UnknownOperation {}

impl Serialize for OperationClass {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for OperationClass {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        OperationClass::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// How far a producer's output may be trusted by its consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReadinessLevel {
    InterfaceVerified,
    FinalVerified,
    Revoked,
}

impl ReadinessLevel {
    /// The single place that decides which operations a level allows.
    pub fn permits(self, op: OperationClass) -> bool {
        match self {
            ReadinessLevel::FinalVerified => true,
            ReadinessLevel::InterfaceVerified => op.is_reversible_preparation(),
            ReadinessLevel::Revoked => false,
        }
    }
}

/// Proof pointer carried by scheduler events: which harness receipt, for
/// which source commit and artifact digest. It is only a claim; the store is
/// re-read and every field re-checked before it is believed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofClaim {
    pub receipt_id: String,
    pub commit: String,
    pub digest: String,
}

/// Verified readiness of one producer artifact. Built only by
/// [`verify_interface`] and [`verify_final`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactReadiness {
    producer_task: String,
    source_commit: String,
    producer_generation: u64,
    artifact_digest: String,
    receipt_id: String,
    level: ReadinessLevel,
}

impl ArtifactReadiness {
    pub fn producer_task(&self) -> &str {
        &self.producer_task
    }
    pub fn source_commit(&self) -> &str {
        &self.source_commit
    }
    pub fn producer_generation(&self) -> u64 {
        self.producer_generation
    }
    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }
    pub fn receipt_id(&self) -> &str {
        &self.receipt_id
    }
    pub fn level(&self) -> ReadinessLevel {
        self.level
    }
    pub fn permits(&self, op: OperationClass) -> bool {
        self.level.permits(op)
    }
    /// The same bindings, downgraded to [`ReadinessLevel::Revoked`].
    pub fn revoked(mut self) -> Self {
        self.level = ReadinessLevel::Revoked;
        self
    }
}

/// What a producer submits to publish an interface. No receipt, digest or
/// flag field exists; unknown fields are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceSubmission {
    pub task_id: String,
    pub generation: u64,
    /// 40 or 64 lowercase hex, checked by aien-proof `check_commit`.
    pub source_commit: String,
    /// The interface artifact. Must be a non-empty JSON object.
    pub interface: Value,
}

/// Durable record naming the task's current interface receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceRecord {
    pub task_id: String,
    pub generation: u64,
    pub receipt_id: String,
    pub source_commit: String,
    pub artifact_digest: String,
}

impl InterfaceRecord {
    pub fn claim(&self) -> ProofClaim {
        ProofClaim {
            receipt_id: self.receipt_id.clone(),
            commit: self.source_commit.clone(),
            digest: self.artifact_digest.clone(),
        }
    }
}

fn record_path(store: &Path, task_id: &str) -> std::path::PathBuf {
    store
        .join(INTERFACE_DIR)
        .join(format!("{}.json", task_key(task_id)))
}

fn read_record(path: &Path, task_id: &str) -> Result<Option<InterfaceRecord>, CompletionError> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CompletionError::Store(format!(
                "cannot read interface record {}: {e}",
                path.display()
            )))
        }
    };
    let rec: InterfaceRecord = serde_json::from_slice(&bytes).map_err(|e| {
        CompletionError::Store(format!(
            "interface record {} is malformed, refusing: {e}",
            path.display()
        ))
    })?;
    if rec.task_id != task_id {
        return Err(CompletionError::Store(format!(
            "interface record {} belongs to task '{}', not '{task_id}'",
            path.display(),
            rec.task_id
        )));
    }
    Ok(Some(rec))
}

fn pass(id: &str, expected: &str, observed: String) -> Assertion {
    Assertion {
        id: id.to_string(),
        expected: expected.to_string(),
        observed,
        pass: true,
        source: ASSERTION_SOURCE.to_string(),
        note: String::new(),
    }
}

impl HarnessEngine {
    fn mint_interface_receipt(
        &self,
        sub: &InterfaceSubmission,
        digest: &str,
        keys: usize,
    ) -> Result<Receipt, CompletionError> {
        let assertions = vec![
            pass(
                "interface:shape",
                "interface is a non-empty JSON object",
                format!("object with {keys} key(s)"),
            ),
            pass(
                "interface:digest",
                "digest recomputed by the harness",
                format!("blake3:{digest}"),
            ),
        ];
        let assertion_bytes =
            serde_json::to_vec(&assertions).map_err(|e| CompletionError::Store(e.to_string()))?;
        let repo = if self.completion_repo.trim().is_empty() {
            crate::UNSPECIFIED_REPO.to_string()
        } else {
            self.completion_repo.clone()
        };
        let mut r = Receipt {
            schema: evidence::SCHEMA.to_string(),
            version: evidence::SCHEMA_VERSION,
            id: String::new(),
            kind: INTERFACE_KIND.to_string(),
            tier: INTERFACE_TIER,
            result: Verdict::Pass,
            timestamp: (self.completion_clock)(),
            repo,
            commit: sub.source_commit.clone(),
            dirty: false,
            toolchain: format!("spark-harness {}", env!("CARGO_PKG_VERSION")),
            procedure: "spark-harness publish_interface".to_string(),
            machine: host_identity(),
            env_class: INTERFACE_TIER.as_str().to_string(),
            input_artifacts: Vec::new(),
            output_artifacts: vec![format!("blake3:{digest}")],
            assertions,
            dependencies: Vec::new(),
            declared_mutation: Mutation::None,
            observed_mutation: Mutation::None,
            authority: String::new(),
            output_digest: blake3::hash(&assertion_bytes).to_hex().to_string(),
            external_refs: vec![
                format!("task:{}", sub.task_id),
                format!("generation:{}", sub.generation),
            ],
            ledger: None,
            lease: None,
            required_features: Vec::new(),
            reserved: String::new(),
        };
        r.id = evidence::receipt_id(&r).map_err(CompletionError::Store)?;
        Ok(r)
    }

    /// Publish an immutable interface artifact for `sub.task_id` at
    /// `sub.generation`. The harness computes the digest, mints and stores a
    /// receipt of kind [`INTERFACE_KIND`], and records it as the task's
    /// current interface.
    ///
    /// - A lower generation than the recorded one is refused as stale.
    /// - The same generation and digest returns the existing record and
    ///   mints nothing (idempotent).
    /// - The same generation with a different digest is an interface change:
    ///   a new receipt replaces the record, and the old receipt stops
    ///   verifying. The scheduler revokes consumers bound to the old one.
    pub fn publish_interface(
        &self,
        sub: &InterfaceSubmission,
    ) -> Result<InterfaceRecord, CompletionError> {
        check_identifier("task_id", &sub.task_id)?;
        evidence::check_commit(&sub.source_commit).map_err(CompletionError::InvalidSubmission)?;
        let keys = match &sub.interface {
            Value::Object(m) if !m.is_empty() => m.len(),
            _ => {
                return Err(CompletionError::InvalidSubmission(
                    "interface must be a non-empty JSON object".to_string(),
                ))
            }
        };
        let digest = artifact_digest(&sub.interface);
        let dir = self.completion_store.join(INTERFACE_DIR);
        fs::create_dir_all(&dir)
            .map_err(|e| CompletionError::Store(format!("cannot create {}: {e}", dir.display())))?;
        let _lock = FileLock::acquire(&dir.join(".lock"))
            .map_err(|e| CompletionError::Store(format!("cannot lock interface store: {e}")))?;
        let path = record_path(&self.completion_store, &sub.task_id);
        if let Some(prev) = read_record(&path, &sub.task_id)? {
            if sub.generation < prev.generation {
                return Err(CompletionError::StaleGeneration {
                    task_id: sub.task_id.clone(),
                    submitted: sub.generation,
                    recorded: prev.generation,
                });
            }
            if sub.generation == prev.generation
                && prev.artifact_digest == digest
                && prev.source_commit == sub.source_commit
            {
                return Ok(prev);
            }
        }
        let receipt = self.mint_interface_receipt(sub, &digest, keys)?;
        let receipt_id = evidence::store_receipt(&self.completion_store, &receipt)
            .map_err(CompletionError::Store)?;
        let rec = InterfaceRecord {
            task_id: sub.task_id.clone(),
            generation: sub.generation,
            receipt_id,
            source_commit: sub.source_commit.clone(),
            artifact_digest: digest,
        };
        let text =
            serde_json::to_vec_pretty(&rec).map_err(|e| CompletionError::Store(e.to_string()))?;
        write_atomic(&path, &text).map_err(|e| {
            CompletionError::Store(format!(
                "cannot write interface record {}: {e}",
                path.display()
            ))
        })?;
        Ok(rec)
    }
}

/// Read the task's current interface record from the store, if any.
pub fn current_interface(
    store: &Path,
    task_id: &str,
) -> Result<Option<InterfaceRecord>, CompletionError> {
    read_record(&record_path(store, task_id), task_id)
}

/// Re-read and check an interface receipt: identity recomputes, PASS,
/// kind [`INTERFACE_KIND`], tier, commit, digest, task and generation all
/// match, and the task's interface record names exactly this receipt.
pub fn verify_interface(
    store: &Path,
    task_id: &str,
    generation: u64,
    claim: &ProofClaim,
) -> Result<ArtifactReadiness, CompletionError> {
    let receipt = evidence::load_receipt(store, &claim.receipt_id)
        .map_err(CompletionError::ReceiptInvalid)?;
    let path = store
        .join(evidence::RECEIPT_DIR)
        .join(format!("{}.json", receipt.id));
    let bytes = fs::read(&path)
        .map_err(|e| CompletionError::ReceiptInvalid(format!("{}: {e}", path.display())))?;
    let (again, report) =
        evidence::verify_bytes(&bytes).map_err(CompletionError::ReceiptInvalid)?;
    if again != receipt {
        return Err(CompletionError::ReceiptInvalid(
            "receipt file changed while being verified".to_string(),
        ));
    }
    if receipt.result != Verdict::Pass || !report.ok() {
        return Err(CompletionError::NotPass(format!(
            "{} ({})",
            receipt.result.as_str(),
            report.reasons.join("; ")
        )));
    }
    require("kind", INTERFACE_KIND, &receipt.kind)?;
    require("tier", INTERFACE_TIER.as_str(), receipt.tier.as_str())?;
    require("commit", &claim.commit, &receipt.commit)?;
    let digest_hex = claim
        .digest
        .strip_prefix("blake3:")
        .unwrap_or(&claim.digest)
        .to_ascii_lowercase();
    require(
        "artifact_digest",
        &format!("blake3:{digest_hex}"),
        &receipt.output_artifacts.join(","),
    )?;
    require(
        "task",
        task_id,
        single_ref(&receipt.external_refs, "task:")?,
    )?;
    require(
        "generation",
        &generation.to_string(),
        single_ref(&receipt.external_refs, "generation:")?,
    )?;
    let current = current_interface(store, task_id)?.ok_or_else(|| {
        CompletionError::NotCurrentCompletion(format!(
            "no interface record for task '{task_id}'; receipt {} is an orphan",
            receipt.id
        ))
    })?;
    if current.receipt_id != receipt.id
        || current.generation != generation
        || current.artifact_digest != digest_hex
    {
        return Err(CompletionError::NotCurrentCompletion(format!(
            "task '{task_id}' current interface is generation {} receipt {}; receipt {} is not it",
            current.generation, current.receipt_id, receipt.id
        )));
    }
    Ok(ArtifactReadiness {
        producer_task: task_id.to_string(),
        source_commit: receipt.commit,
        producer_generation: generation,
        artifact_digest: digest_hex,
        receipt_id: receipt.id,
        level: ReadinessLevel::InterfaceVerified,
    })
}

/// M1 [`verify_completion`] for the producer's generation, as readiness.
pub fn verify_final(
    store: &Path,
    task_id: &str,
    generation: u64,
    claim: &ProofClaim,
) -> Result<ArtifactReadiness, CompletionError> {
    let v = verify_completion(
        store,
        &claim.receipt_id,
        task_id,
        generation,
        &claim.commit,
        &claim.digest,
    )?;
    Ok(ArtifactReadiness {
        producer_task: v.task_id,
        source_commit: v.source_commit,
        producer_generation: v.generation,
        artifact_digest: v.artifact_digest,
        receipt_id: v.receipt_id,
        level: ReadinessLevel::FinalVerified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_exact_and_closed() {
        for op in OperationClass::ALL {
            assert_eq!(OperationClass::parse(op.as_str()), Ok(op));
        }
        for bad in [
            "",
            "Commit",
            "read-only-analysis",
            " commit",
            "sandboxed_check ",
            "rm",
        ] {
            assert!(
                OperationClass::parse(bad).is_err(),
                "{bad:?} must be unknown"
            );
        }
        let err = serde_json::from_str::<OperationClass>("\"deploy\"");
        assert!(err.is_err());
    }

    #[test]
    fn levels_permit_only_what_they_should() {
        for op in OperationClass::ALL {
            assert!(ReadinessLevel::FinalVerified.permits(op));
            assert!(!ReadinessLevel::Revoked.permits(op));
            assert_eq!(
                ReadinessLevel::InterfaceVerified.permits(op),
                matches!(
                    op,
                    OperationClass::ReadOnlyAnalysis | OperationClass::SandboxedCheck
                )
            );
        }
    }
}
