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

//! SourceGrant v2 framed-digest cutover (aien-protocols-mv#59).
//!
//! Both fixtures carry the same grant fields and are signed by the same
//! synthetic test authority (signing scalar `[42u8; 32]`, never a real key).
//! `source-grant-v1-legacy.json` is signed over the retired concatenated
//! transcript; `source-grant-v2-reissued.json` is the same grant reissued over
//! the `AIEN_SOURCE_GRANT_V2` transcript. Every consumer boundary must refuse
//! the first and accept the second. There is no legacy fallback.

use aien_provenance::{GrantPermissions, ProvenanceError, SourceGrant};
use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use spark_harvester::{BundleKind, ExtractedPair, HarvesterClient, RightsGate};
use std::path::PathBuf;
use std::process::Command;

const LEGACY_FIXTURE: &str = "tests/fixtures/source-grant-v1-legacy.json";
const V2_FIXTURE: &str = "tests/fixtures/source-grant-v2-reissued.json";
/// SEC1 compressed public key of the synthetic `[42u8; 32]` test authority.
const AUTHORITY_PUB_HEX: &str =
    "020c901d423c831ca85e27c73c263ba132721bb9d7a84c4f0380b2a6756fd60133";

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn load(rel: &str) -> SourceGrant {
    let text = std::fs::read_to_string(fixture_path(rel)).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses as SourceGrant")
}

fn authority() -> (SigningKey, VerifyingKey) {
    let sk = SigningKey::from_slice(&[42u8; 32]).expect("valid synthetic key");
    let vk = *sk.verifying_key();
    assert_eq!(
        hex::encode(vk.to_sec1_point(true).as_bytes()),
        AUTHORITY_PUB_HEX
    );
    (sk, vk)
}

/// The retired v1 transcript, reproduced here only to prove the legacy
/// fixture is a genuine old-format signature rather than random bytes.
fn retired_v1_digest(g: &SourceGrant) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(g.grant_id.as_bytes());
    h.update(g.source_uri.as_bytes());
    h.update(g.license_spdx.as_bytes());
    h.update(g.permissions.bits().to_le_bytes());
    h.update(g.issuer.as_bytes());
    match g.expires_at_epoch_sec {
        Some(exp) => {
            h.update(b"exp:present");
            h.update(exp.to_le_bytes());
        }
        None => h.update(b"exp:none"),
    }
    h.finalize().into()
}

fn pair() -> ExtractedPair {
    ExtractedPair {
        prompt: "Explain framed digest transcripts".to_string(),
        reasoning: Some("Length prefixes remove field-boundary ambiguity".to_string()),
        completion: "Each variable-length field is prefixed by its byte length.".to_string(),
        provider: "sovereign-lab".to_string(),
        model: "atlas-omni-reasoning".to_string(),
        token_count: 16,
        quality_score: 0.9,
    }
}

#[test]
fn fixtures_are_the_same_grant_under_two_transcripts() {
    let (_sk, vk) = authority();
    let legacy = load(LEGACY_FIXTURE);
    let v2 = load(V2_FIXTURE);

    let mut legacy_fields = legacy.clone();
    legacy_fields.issuer_signature = None;
    let mut v2_fields = v2.clone();
    v2_fields.issuer_signature = None;
    assert_eq!(legacy_fields, v2_fields, "only the signature may differ");

    // Positive control: the legacy signature is valid over the retired transcript.
    let sig = Signature::from_slice(legacy.issuer_signature.as_deref().unwrap()).unwrap();
    vk.verify(&retired_v1_digest(&legacy), &sig)
        .expect("legacy fixture must be a genuine v1 signature");
}

#[test]
fn reissued_fixture_matches_current_protocol_signing() {
    let (sk, vk) = authority();
    let v2 = load(V2_FIXTURE);
    let mut resigned = v2.clone();
    resigned.issuer_signature = None;
    resigned.sign(&sk); // RFC 6979 deterministic ECDSA
    assert_eq!(
        resigned, v2,
        "reissued fixture drifted from the pinned protocol"
    );
    v2.verify_signature(&vk).expect("v2 grant verifies");
}

#[test]
fn rights_gate_refuses_legacy_signed_grant() {
    let (_sk, vk) = authority();
    let legacy = load(LEGACY_FIXTURE);
    let err =
        RightsGate::validate_and_bundle(&legacy, &[pair()], BundleKind::Training, &["MIT"], &vk)
            .expect_err("legacy-transcript grant must be refused");
    assert!(
        matches!(err, ProvenanceError::SignatureVerificationFailed(_)),
        "expected SignatureVerificationFailed, got {err:?}"
    );
}

#[test]
fn rights_gate_accepts_reissued_v2_grant() {
    let (_sk, vk) = authority();
    let v2 = load(V2_FIXTURE);
    let bundle =
        RightsGate::validate_and_bundle(&v2, &[pair()], BundleKind::Training, &["MIT"], &vk)
            .expect("reissued v2 grant must be accepted");
    assert_eq!(bundle.records.len(), 1);
    assert_eq!(bundle.records[0].source_uri, v2.source_uri);
    assert!(v2.verify_permission(GrantPermissions::DISTILL).is_ok());
}

#[tokio::test]
async fn client_refuses_legacy_signed_grant_before_any_request() {
    let (_sk, vk) = authority();
    let legacy = load(LEGACY_FIXTURE);
    // Port 9 (discard) on loopback: the refusal must come before any request.
    let err = HarvesterClient::new()
        .query_with_grant("http://127.0.0.1:9/v1", "unused", "m", "p", &legacy, &vk)
        .await
        .expect_err("legacy-transcript grant must be refused");
    let msg = err.to_string();
    assert!(
        msg.starts_with("Rights verification failed: Signature verification failed"),
        "expected rights refusal, got {msg}"
    );
}

fn run_extract(grant_rel: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_spark-harvester"))
        .args([
            "extract",
            "--prompt",
            "p",
            "--input",
            "i",
            "--bundle-kind",
            "training",
        ])
        .arg("--grant-file")
        .arg(fixture_path(grant_rel))
        .args(["--authority-key", AUTHORITY_PUB_HEX])
        .output()
        .expect("spark-harvester runs")
}

#[test]
fn cli_refuses_legacy_grant_file() {
    let out = run_extract(LEGACY_FIXTURE);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "legacy grant admitted; stderr: {stderr}"
    );
    assert!(
        stderr.contains("SignatureVerificationFailed"),
        "expected signature refusal, stderr: {stderr}"
    );
}

#[test]
fn cli_accepts_reissued_v2_grant_file() {
    let out = run_extract(V2_FIXTURE);
    assert!(
        out.status.success(),
        "v2 grant refused; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bundle: serde_json::Value = serde_json::from_slice(
        out.stdout
            .split(|b| *b == b'\n')
            .skip_while(|l| !l.starts_with(b"{"))
            .collect::<Vec<_>>()
            .join(&b'\n')
            .as_slice(),
    )
    .expect("bundle JSON on stdout");
    assert_eq!(bundle["license_spdx"], "MIT");
    assert_eq!(bundle["records"].as_array().map(Vec::len), Some(1));
}
