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

use crate::extractor::ExtractedPair;
use aien_protocol_types::Digest32;
use aien_provenance::{GrantPermissions, ProvenanceError, SourceGrant};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BundleKind {
    Training,
    Distillation,
    Evidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedDatasetRecord {
    pub record_id: Uuid,
    pub bundle_kind: BundleKind,
    pub grant_id: Uuid,
    pub source_uri: String,
    pub license_spdx: String,
    pub pair: ExtractedPair,
    pub provenance_digest: Digest32,
}

impl GatedDatasetRecord {
    pub fn compute_digest(&self) -> Digest32 {
        let mut hasher = Sha256::new();
        hasher.update(self.record_id.as_bytes());
        hasher.update(self.grant_id.as_bytes());
        hasher.update(self.source_uri.as_bytes());
        hasher.update(self.license_spdx.as_bytes());
        hasher.update(self.pair.prompt.as_bytes());
        hasher.update(self.pair.completion.as_bytes());
        if let Some(ref r) = self.pair.reasoning {
            hasher.update(r.as_bytes());
        }
        Digest32(hasher.finalize().into())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedDatasetBundle {
    pub bundle_id: Uuid,
    pub bundle_kind: BundleKind,
    pub license_spdx: String,
    pub records: Vec<GatedDatasetRecord>,
    pub merkle_root: Digest32,
    pub created_at: String,
}

pub struct RightsGate;

impl RightsGate {
    /// Validates an array of extracted pairs against a SourceGrant and builds an immutable GatedDatasetBundle.
    /// Cryptographic signature verification against the authority verifying key is mandatory and fails closed.
    pub fn validate_and_bundle(
        grant: &SourceGrant,
        pairs: &[ExtractedPair],
        bundle_kind: BundleKind,
        allowed_licenses: &[&str],
        authority_verifying_key: &p256::ecdsa::VerifyingKey,
    ) -> Result<GatedDatasetBundle, ProvenanceError> {
        // 1. Verify temporal validity
        let now = chrono::Utc::now().timestamp() as u64;
        grant.verify_validity(now)?;

        // 2. Mandatory cryptographic signature verification
        grant.verify_signature(authority_verifying_key)?;

        // 3. Verify license compatibility
        grant.verify_license(allowed_licenses)?;

        // 4. Verify grant permissions based on bundle kind
        let required_perm = match bundle_kind {
            BundleKind::Training => GrantPermissions::TRAIN,
            BundleKind::Distillation => GrantPermissions::DISTILL,
            BundleKind::Evidence => GrantPermissions::EVALUATE,
        };
        grant.verify_permission(required_perm)?;

        // 5. Build records and compute individual provenance digests
        let mut records = Vec::with_capacity(pairs.len());
        let mut digests = Vec::with_capacity(pairs.len());

        for pair in pairs {
            let record_id = Uuid::new_v4();
            let mut record = GatedDatasetRecord {
                record_id,
                bundle_kind,
                grant_id: grant.grant_id,
                source_uri: grant.source_uri.clone(),
                license_spdx: grant.license_spdx.clone(),
                pair: pair.clone(),
                provenance_digest: Digest32::ZERO,
            };
            record.provenance_digest = record.compute_digest();
            digests.push(record.provenance_digest);
            records.push(record);
        }

        // 6. Compute Merkle root over records
        let merkle_root = Self::compute_merkle_root(&digests);

        Ok(GatedDatasetBundle {
            bundle_id: Uuid::new_v4(),
            bundle_kind,
            license_spdx: grant.license_spdx.clone(),
            records,
            merkle_root,
            created_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Exports a bundle to a JSONL file with strict segregation enforcement.
    /// Evidence bundles are barred from being written to training/sft/dpo dataset paths,
    /// and must reside within dedicated evidence or quarantine path structures.
    pub fn export_bundle(bundle: &GatedDatasetBundle, output_path: &Path) -> io::Result<usize> {
        if bundle.bundle_kind == BundleKind::Evidence {
            let normalized = if output_path.is_absolute() {
                output_path.to_path_buf()
            } else {
                std::env::current_dir()
                    .map(|c| c.join(output_path))
                    .unwrap_or_else(|_| output_path.to_path_buf())
            };

            // Structural check: Bar from training, sft, dpo, distill paths
            for component in normalized.components() {
                let name = component.as_os_str().to_string_lossy().to_lowercase();
                if name.contains("train")
                    || name.contains("sft")
                    || name.contains("dpo")
                    || name.contains("distill")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "Evidence bundles are strictly quarantined and cannot be written to training or distillation dataset paths",
                    ));
                }
            }

            // Structural check: Must reside within an evidence or quarantine directory or file
            let in_evidence_or_quarantine = normalized.components().any(|c| {
                let name = c.as_os_str().to_string_lossy().to_lowercase();
                name.contains("evidence") || name.contains("quarantine")
            });

            if !in_evidence_or_quarantine {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Evidence bundles must be quarantined inside a dedicated evidence or quarantine path",
                ));
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)?;
        let mut writer = io::BufWriter::new(file);

        let mut count = 0;
        for record in &bundle.records {
            let serialized = serde_json::to_string(record)?;
            writer.write_all(serialized.as_bytes())?;
            writer.write_all(b"\n")?;
            count += 1;
        }
        writer.flush()?;
        Ok(count)
    }

    fn compute_merkle_root(digests: &[Digest32]) -> Digest32 {
        if digests.is_empty() {
            return Digest32::ZERO;
        }
        let mut current = digests.to_vec();
        while current.len() > 1 {
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(chunk[0].0);
                if chunk.len() > 1 {
                    hasher.update(chunk[1].0);
                } else {
                    hasher.update(chunk[0].0);
                }
                next.push(Digest32(hasher.finalize().into()));
            }
            current = next;
        }
        current[0]
    }
}
