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

use chrono::{DateTime, Duration, Utc};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRecord {
    pub id: String,
    pub source_kind: String,
    pub source_id: String,
    pub commitment: String,
    pub merkle_root: Option<String>,
    pub segment_id: Option<String>,
    pub retention_class: String,
    pub retain_until: Option<String>,
    pub created_at: String,
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn is_commitment(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()) && hash != "hash_ref"
}

pub fn retain_until(class: &str, now: DateTime<Utc>) -> Option<String> {
    match class {
        "permanent" => None,
        "ephemeral" => Some((now + Duration::days(1)).to_rfc3339()),
        _ => Some((now + Duration::days(365)).to_rfc3339()),
    }
}

pub fn merkle_root(leaves: &[impl AsRef<[u8]>]) -> String {
    if leaves.is_empty() {
        return format!("{:x}", Sha256::digest(b""));
    }
    let mut level: Vec<[u8; 32]> = leaves
        .iter()
        .map(|leaf| {
            let digest = Sha256::digest(leaf.as_ref());
            let mut out = [0u8; 32];
            out.copy_from_slice(&digest);
            out
        })
        .collect();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push(*level.last().unwrap());
        }
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(pair[0]);
            hasher.update(pair[1]);
            let digest = hasher.finalize();
            let mut out = [0u8; 32];
            out.copy_from_slice(&digest);
            next.push(out);
        }
        level = next;
    }
    hex_encode(&level[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_root_changes_when_a_leaf_changes() {
        let a = merkle_root(&[b"aaa", b"bbb"]);
        let b = merkle_root(&[b"aaa", b"ccc"]);
        assert_ne!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn ephemeral_retention_is_shorter_than_standard() {
        let now = Utc::now();
        let ephemeral = retain_until("ephemeral", now).unwrap();
        let standard = retain_until("standard", now).unwrap();
        assert!(ephemeral < standard);
        assert!(retain_until("permanent", now).is_none());
    }
}
