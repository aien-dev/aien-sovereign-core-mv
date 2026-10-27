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

use crate::identity::{IdentityAnchor, KeyRecord};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SyncReport {
    pub anchor_did: String,
    pub entity_name: String,
    pub radicle_key_detected: Option<String>,
    pub ssh_key_detected: Option<String>,
    pub in_sync: bool,
    pub rotated: bool,
    pub message: String,
}

pub fn get_sha256_fingerprint(key_str: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key_str.trim().as_bytes());
    hex::encode(hasher.finalize())
}

/// Validate format of detected public keys.
pub fn validate_public_key_format(key_type: &str, key_str: &str) -> Result<(), String> {
    let trimmed = key_str.trim();
    if trimmed.is_empty() {
        return Err("Public key is empty".to_string());
    }

    match key_type {
        "ssh-ed25519" => {
            if !trimmed.starts_with("ssh-ed25519 ") {
                return Err(
                    "Invalid SSH Ed25519 key format: expected prefix ssh-ed25519".to_string(),
                );
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 2 {
                return Err("Malformed SSH key: missing base64 token payload".to_string());
            }
            if parts[1].len() < 32 {
                return Err("SSH key payload too short for Ed25519 public key".to_string());
            }
            Ok(())
        }
        "radicle-ed25519" => {
            if trimmed.starts_with("ssh-ed25519 ")
                || trimmed.starts_with("rad:")
                || trimmed.len() >= 32
            {
                Ok(())
            } else {
                Err("Invalid Radicle public key length or format".to_string())
            }
        }
        _ => Ok(()),
    }
}

pub fn discover_local_keys() -> (Option<String>, Option<String>) {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    // 1. Radicle Key
    let rad_pub_path = PathBuf::from(&home)
        .join(".radicle")
        .join("keys")
        .join("radicle.pub");
    let rad_key = if rad_pub_path.exists() {
        fs::read_to_string(rad_pub_path)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    };

    // 2. SSH Key
    let ssh_pub_path = PathBuf::from(&home).join(".ssh").join("id_ed25519.pub");
    let ssh_key = if ssh_pub_path.exists() {
        fs::read_to_string(ssh_pub_path)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    };

    (rad_key, ssh_key)
}

pub fn sync_identity_keys(
    mut anchor: IdentityAnchor,
) -> Result<(IdentityAnchor, SyncReport), String> {
    let (rad_key, ssh_key) = discover_local_keys();
    let mut rotated = false;
    let mut message = "Keys are already synchronized with Identity Anchor.".to_string();

    let now = chrono::Utc::now().to_rfc3339();

    // Check Radicle Key sync
    if let Some(ref r_key) = rad_key {
        validate_public_key_format("radicle-ed25519", r_key)?;
        if anchor.primary_radicle_key.as_ref() != Some(r_key) {
            if anchor.primary_radicle_key.is_some() {
                rotated = true;
                message =
                    "Detected rotated Radicle public key; updated Identity Anchor.".to_string();
            }
            anchor.primary_radicle_key = Some(r_key.clone());
            anchor.key_history.push(KeyRecord {
                key_type: "radicle-ed25519".to_string(),
                public_key: r_key.clone(),
                finger_print: get_sha256_fingerprint(r_key),
                added_at: now.clone(),
                status: "active".to_string(),
            });
        }
    }

    // Check SSH Key sync
    if let Some(ref s_key) = ssh_key {
        validate_public_key_format("ssh-ed25519", s_key)?;
        if anchor.primary_ssh_key.as_ref() != Some(s_key) {
            if anchor.primary_ssh_key.is_some() {
                rotated = true;
                message = "Detected updated SSH public key; updated Identity Anchor.".to_string();
            }
            anchor.primary_ssh_key = Some(s_key.clone());
            anchor.key_history.push(KeyRecord {
                key_type: "ssh-ed25519".to_string(),
                public_key: s_key.clone(),
                finger_print: get_sha256_fingerprint(s_key),
                added_at: now.clone(),
                status: "active".to_string(),
            });
        }
    }

    anchor.last_synced_at = now;
    anchor.save()?;

    let in_sync = anchor.primary_ssh_key.is_some() || anchor.primary_radicle_key.is_some();

    let report = SyncReport {
        anchor_did: anchor.did.clone(),
        entity_name: anchor.name.clone(),
        radicle_key_detected: rad_key,
        ssh_key_detected: ssh_key,
        in_sync,
        rotated,
        message,
    };

    Ok((anchor, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_fingerprint_deterministic() {
        let sample = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExamplePublicKey user@host";
        let fp1 = get_sha256_fingerprint(sample);
        let fp2 = get_sha256_fingerprint(sample);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 64);
    }

    #[test]
    fn test_sync_report_serialization() {
        let report = SyncReport {
            anchor_did: "did:rad:aien:spark-01".to_string(),
            entity_name: "AIEN".to_string(),
            radicle_key_detected: Some("ssh-ed25519 AAAAC...".to_string()),
            ssh_key_detected: Some("ssh-ed25519 AAAAC...".to_string()),
            in_sync: true,
            rotated: false,
            message: "All keys synchronized.".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serializable");
        assert!(json.contains("did:rad:aien:spark-01"));
        assert!(json.contains("All keys synchronized."));
    }

    #[test]
    fn test_validate_public_key_format() {
        assert!(validate_public_key_format(
            "ssh-ed25519",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleValidPayloadKey user@host"
        )
        .is_ok());
        assert!(validate_public_key_format("ssh-ed25519", "rsa-bad AAAAC3").is_err());
        assert!(validate_public_key_format("ssh-ed25519", "ssh-ed25519 short").is_err());
        assert!(
            validate_public_key_format("radicle-ed25519", "rad:z6MkuTf9Vd6F4sM6eS2y5fV9tF3p")
                .is_ok()
        );
        assert!(validate_public_key_format("radicle-ed25519", "").is_err());
    }
}
