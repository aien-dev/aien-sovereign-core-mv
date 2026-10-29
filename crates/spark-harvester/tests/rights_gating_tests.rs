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

use aien_provenance::{GrantPermissions, ProvenanceError, SourceGrant};
use p256::ecdsa::SigningKey;
use spark_harvester::{BundleKind, ExtractedPair, RightsGate};
use std::io::ErrorKind;

fn create_test_pair() -> ExtractedPair {
    ExtractedPair {
        prompt: "Explain copy-on-write paged attention".to_string(),
        reasoning: Some("Analyze memory refcounting and block sharing mechanics".to_string()),
        completion:
            "CoW allocates physical pages only when divergent tokens are written to a shared block."
                .to_string(),
        provider: "sovereign-lab".to_string(),
        model: "atlas-omni-reasoning".to_string(),
        token_count: 32,
        quality_score: 0.98,
    }
}

fn create_authority_keys() -> (SigningKey, p256::ecdsa::VerifyingKey) {
    let signing_key = SigningKey::from_slice(&[42u8; 32]).expect("valid signing key");
    let verifying_key = *signing_key.verifying_key();
    (signing_key, verifying_key)
}

#[test]
fn test_rights_gated_training_bundle_admitted_with_valid_grant() {
    let (sk, vk) = create_authority_keys();
    let mut grant = SourceGrant::new(
        "https://hf.co/datasets/aien/sovereign-math".to_string(),
        "Apache-2.0".to_string(),
        GrantPermissions::TRAIN | GrantPermissions::DISTILL,
        "Atlas Rights Authority".to_string(),
    );
    grant.sign(&sk);

    let pairs = vec![create_test_pair()];
    let bundle = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    )
    .expect("Valid signed training grant must be accepted");

    assert_eq!(bundle.bundle_kind, BundleKind::Training);
    assert_eq!(bundle.records.len(), 1);
    assert_ne!(bundle.merkle_root, aien_protocol_types::Digest32::ZERO);
    assert_eq!(
        bundle.records[0].source_uri,
        "https://hf.co/datasets/aien/sovereign-math"
    );
}

#[test]
fn test_unsigned_grant_rejected_fails_closed() {
    let (_sk, vk) = create_authority_keys();
    let grant = SourceGrant::new(
        "https://hf.co/datasets/aien/unsigned-dataset".to_string(),
        "Apache-2.0".to_string(),
        GrantPermissions::TRAIN,
        "Unauthenticated Issuer".to_string(),
    );

    let pairs = vec![create_test_pair()];
    let res = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    );

    match res {
        Err(ProvenanceError::MissingSignature) => {}
        other => panic!(
            "Expected MissingSignature for unsigned grant, got {:?}",
            other
        ),
    }
}

#[test]
fn test_unapproved_license_rejected() {
    let (sk, vk) = create_authority_keys();
    let mut grant = SourceGrant::new(
        "https://example.com/proprietary-dataset".to_string(),
        "CC-BY-NC-4.0".to_string(),
        GrantPermissions::TRAIN,
        "Untrusted Third Party".to_string(),
    );
    grant.sign(&sk);

    let pairs = vec![create_test_pair()];
    let res = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    );

    match res {
        Err(ProvenanceError::IncompatibleLicense { license }) => {
            assert_eq!(license, "CC-BY-NC-4.0");
        }
        other => panic!("Expected IncompatibleLicense, got {:?}", other),
    }
}

#[test]
fn test_missing_permission_rejected() {
    let (sk, vk) = create_authority_keys();
    let mut grant = SourceGrant::new(
        "https://hf.co/datasets/aien/eval-set".to_string(),
        "MIT".to_string(),
        GrantPermissions::EVALUATE, // Only EVALUATE, not TRAIN
        "Atlas Rights Authority".to_string(),
    );
    grant.sign(&sk);

    let pairs = vec![create_test_pair()];
    let res = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    );

    match res {
        Err(ProvenanceError::PermissionDenied { required, granted }) => {
            assert_eq!(required, GrantPermissions::TRAIN);
            assert_eq!(granted, GrantPermissions::EVALUATE);
        }
        other => panic!("Expected PermissionDenied, got {:?}", other),
    }
}

#[test]
fn test_expired_grant_rejected() {
    let (sk, vk) = create_authority_keys();
    let mut grant = SourceGrant::new(
        "https://hf.co/datasets/aien/expired-set".to_string(),
        "MIT".to_string(),
        GrantPermissions::TRAIN,
        "Atlas Rights Authority".to_string(),
    );
    grant.expires_at_epoch_sec = Some(1000); // Far in the past
    grant.sign(&sk);

    let pairs = vec![create_test_pair()];
    let res = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    );

    match res {
        Err(ProvenanceError::GrantExpired(expires_at)) => {
            assert_eq!(expires_at, 1000);
        }
        other => panic!("Expected GrantExpired, got {:?}", other),
    }
}

#[test]
fn test_cryptographic_signature_verification() {
    let signing_key = SigningKey::from_slice(&[33u8; 32]).expect("valid key");
    let verifying_key = *signing_key.verifying_key();

    let mut grant = SourceGrant::new(
        "https://hf.co/datasets/aien/signed-set".to_string(),
        "MIT".to_string(),
        GrantPermissions::TRAIN,
        "Atlas Rights Authority".to_string(),
    );
    grant.sign(&signing_key);

    let pairs = vec![create_test_pair()];
    let bundle = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &verifying_key,
    )
    .expect("Signed grant must be accepted");
    assert_eq!(bundle.records.len(), 1);

    // Tampering with grant fails signature verification
    let mut tampered_grant = grant.clone();
    tampered_grant.license_spdx = "Apache-2.0".to_string();
    let tamper_res = RightsGate::validate_and_bundle(
        &tampered_grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &verifying_key,
    );
    match tamper_res {
        Err(ProvenanceError::SignatureVerificationFailed(_)) => {}
        other => panic!("Expected SignatureVerificationFailed, got {:?}", other),
    }

    // Wrong authority key fails signature verification
    let wrong_signing_key = SigningKey::from_slice(&[99u8; 32]).unwrap();
    let wrong_verifying_key = *wrong_signing_key.verifying_key();
    let wrong_key_res = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Training,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &wrong_verifying_key,
    );
    match wrong_key_res {
        Err(ProvenanceError::SignatureVerificationFailed(_)) => {}
        other => panic!(
            "Expected SignatureVerificationFailed with wrong key, got {:?}",
            other
        ),
    }
}

#[test]
fn test_evidence_bundle_quarantine_enforced() {
    let (sk, vk) = create_authority_keys();
    let mut grant = SourceGrant::new(
        "https://hf.co/datasets/aien/benchmark-evidence".to_string(),
        "MIT".to_string(),
        GrantPermissions::EVALUATE,
        "Atlas Rights Authority".to_string(),
    );
    grant.sign(&sk);

    let pairs = vec![create_test_pair()];
    let bundle = RightsGate::validate_and_bundle(
        &grant,
        &pairs,
        BundleKind::Evidence,
        &["Apache-2.0", "MIT", "SRCL-1.0"],
        &vk,
    )
    .expect("Evidence grant must be accepted");

    // The quarantine gate scans every path component for train/sft/dpo/distill,
    // so a random temp-dir name can trip it by accident. Draw until the name is clean.
    let tmp = loop {
        let t = tempfile::tempdir().unwrap();
        let name = t
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_lowercase();
        if !["train", "sft", "dpo", "distill"]
            .iter()
            .any(|b| name.contains(b))
        {
            break t;
        }
    };

    // 1. Attempting to export evidence bundle to training path fails closed
    let train_path = tmp.path().join("train_dataset.jsonl");
    let err = RightsGate::export_bundle(&bundle, &train_path).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::PermissionDenied);

    let sft_path = tmp.path().join("sft.jsonl");
    let err_sft = RightsGate::export_bundle(&bundle, &sft_path).unwrap_err();
    assert_eq!(err_sft.kind(), ErrorKind::PermissionDenied);

    let distill_path = tmp.path().join("distill_dataset.jsonl");
    let err_distill = RightsGate::export_bundle(&bundle, &distill_path).unwrap_err();
    assert_eq!(err_distill.kind(), ErrorKind::PermissionDenied);

    // 2. Exporting to path without quarantine/evidence directory or name fails closed
    let general_path = tmp.path().join("output.jsonl");
    let err_general = RightsGate::export_bundle(&bundle, &general_path).unwrap_err();
    assert_eq!(err_general.kind(), ErrorKind::PermissionDenied);

    // 3. Exporting to quarantine evidence path succeeds
    let evidence_path = tmp.path().join("evidence_audit.jsonl");
    let count =
        RightsGate::export_bundle(&bundle, &evidence_path).expect("Evidence export must succeed");
    assert_eq!(count, 1);
    assert!(evidence_path.exists());

    let quarantine_dir = tmp.path().join("quarantine");
    std::fs::create_dir_all(&quarantine_dir).unwrap();
    let q_path = quarantine_dir.join("records.jsonl");
    let q_count =
        RightsGate::export_bundle(&bundle, &q_path).expect("Quarantine export must succeed");
    assert_eq!(q_count, 1);
    assert!(q_path.exists());
}
