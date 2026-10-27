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

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    AutonomousAgent,
    HumanOperator,
    NodeBridge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRecord {
    pub key_type: String, // e.g. "ssh-ed25519", "radicle-ed25519"
    pub public_key: String,
    pub finger_print: String,
    pub added_at: String,
    pub status: String, // "active", "rotated", "revoked"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityAnchor {
    pub did: String,  // e.g. "did:rad:aien:spark-01"
    pub name: String, // "AIEN" or "Drake Stapleton"
    pub entity_type: EntityType,
    pub primary_radicle_key: Option<String>,
    pub primary_ssh_key: Option<String>,
    pub key_history: Vec<KeyRecord>,
    pub last_synced_at: String,
    pub metadata: std::collections::HashMap<String, String>,
}

impl IdentityAnchor {
    pub fn default_agent() -> Self {
        let mut meta = std::collections::HashMap::new();
        meta.insert("harness".to_string(), "aien-native-1.0".to_string());
        meta.insert("host".to_string(), "spark-gb10".to_string());

        Self {
            did: "did:rad:aien:spark-master".to_string(),
            name: "AIEN".to_string(),
            entity_type: EntityType::AutonomousAgent,
            primary_radicle_key: None,
            primary_ssh_key: None,
            key_history: Vec::new(),
            last_synced_at: chrono::Utc::now().to_rfc3339(),
            metadata: meta,
        }
    }

    pub fn anchor_file_path() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".radicle")
            .join("identity_anchor.json")
    }

    pub fn load_or_init() -> Result<Self, String> {
        let path = Self::anchor_file_path();
        if path.exists() {
            let data =
                fs::read_to_string(&path).map_err(|e| format!("Failed to read anchor: {}", e))?;
            serde_json::from_str(&data).map_err(|e| format!("Failed to parse anchor: {}", e))
        } else {
            let anchor = Self::default_agent();
            anchor.save()?;
            Ok(anchor)
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::anchor_file_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent dir: {}", e))?;
        }
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        fs::write(&path, data).map_err(|e| format!("Failed to write anchor: {}", e))
    }
}
