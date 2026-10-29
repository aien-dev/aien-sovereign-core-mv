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

// spark-aegis Live System and Workspace Confinement Audit Engine
// Inspects active listening sockets, workspace secrets, and host invariants

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketAudit {
    pub protocol: String,
    pub local_address: String,
    pub is_loopback: bool,
    pub process_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceAudit {
    pub path: PathBuf,
    pub dot_env_found: Vec<PathBuf>,
    pub exposed_keys_found: Vec<PathBuf>,
    pub secure_tpm_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveAuditReport {
    pub timestamp: DateTime<Utc>,
    pub sockets: Vec<SocketAudit>,
    pub non_loopback_sockets_count: usize,
    pub workspace: WorkspaceAudit,
    pub cortex_online: bool,
    pub cortex_version: Option<String>,
    pub blake3_digest: String,
    pub overall_safe: bool,
}

pub async fn audit_system() -> Result<LiveAuditReport> {
    let ts = Utc::now();

    // 1. Audit listening sockets
    let sockets = audit_listening_sockets();
    let non_loopback_count = sockets.iter().filter(|s| !s.is_loopback).count();

    // 2. Audit workspace confinement
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let workspace_dir = PathBuf::from(home).join("workspace");
    let workspace = audit_workspace_dir(&workspace_dir);

    // 3. Audit Cortex memory engine
    let (cortex_online, cortex_version) = check_cortex_health().await;

    // Overall safety: no plaintext secrets in workspace
    let overall_safe = workspace.secure_tpm_only;

    // Cryptographic hash of audit report
    let digest_seed = format!(
        "{}:{}:{}:{}",
        ts, non_loopback_count, workspace.secure_tpm_only, cortex_online
    );
    let blake3_digest = blake3::hash(digest_seed.as_bytes()).to_hex().to_string();

    Ok(LiveAuditReport {
        timestamp: ts,
        sockets,
        non_loopback_sockets_count: non_loopback_count,
        workspace,
        cortex_online,
        cortex_version,
        blake3_digest,
        overall_safe,
    })
}

fn audit_listening_sockets() -> Vec<SocketAudit> {
    let mut results = Vec::new();

    // Try `ss -tulpn`
    if let Ok(output) = Command::new("ss").args(["-tulpn"]).output() {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                let proto = parts[0].to_string();
                let addr_str = parts[4];
                let is_loop = addr_str.starts_with("127.") || addr_str.starts_with("[::1]");
                let proc_name = if parts.len() >= 7 {
                    Some(parts[6].to_string())
                } else {
                    None
                };

                results.push(SocketAudit {
                    protocol: proto,
                    local_address: addr_str.to_string(),
                    is_loopback: is_loop,
                    process_name: proc_name,
                });
            }
        }
    }

    results
}

fn audit_workspace_dir(dir: &Path) -> WorkspaceAudit {
    let mut dot_envs = Vec::new();
    let mut exposed_keys = Vec::new();

    if dir.exists() && dir.is_dir() {
        let mut stack = vec![dir.to_path_buf()];
        let mut files_checked = 0;

        while let Some(current) = stack.pop() {
            if files_checked > 2000 {
                break; // Boundary limit
            }
            if let Ok(entries) = fs::read_dir(&current) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let name = match p.file_name().and_then(|s| s.to_str()) {
                        Some(n) => n,
                        None => continue,
                    };

                    if name == ".git" || name == "target" || name == "node_modules" {
                        continue;
                    }

                    if p.is_dir() {
                        stack.push(p);
                    } else if p.is_file() {
                        files_checked += 1;
                        if name == ".env" || name.starts_with(".env.") {
                            dot_envs.push(p.clone());
                        }
                        if name.ends_with(".key") || name.ends_with(".pem") || name == "id_rsa" {
                            exposed_keys.push(p.clone());
                        }
                    }
                }
            }
        }
    }

    let secure_tpm_only = dot_envs.is_empty() && exposed_keys.is_empty();

    WorkspaceAudit {
        path: dir.to_path_buf(),
        dot_env_found: dot_envs,
        exposed_keys_found: exposed_keys,
        secure_tpm_only,
    }
}

async fn check_cortex_health() -> (bool, Option<String>) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1500))
        .build();

    if let Ok(c) = client {
        let cortex_url = std::env::var("CORTEX_HEALTH_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:18080/health".to_string());
        if let Ok(resp) = c.get(&cortex_url).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let ver = json
                        .get("version")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    return (true, ver);
                }
                return (true, None);
            }
        }
    }

    (false, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_audit_empty_tmp() {
        let tmp = tempfile::tempdir().unwrap();
        let audit = audit_workspace_dir(tmp.path());
        assert!(audit.secure_tpm_only);
        assert!(audit.dot_env_found.is_empty());
    }

    #[test]
    fn test_workspace_audit_flags_dotenv() {
        let tmp = tempfile::tempdir().unwrap();
        let env_file = tmp.path().join(".env");
        fs::write(&env_file, b"SECRET=true").unwrap();

        let audit = audit_workspace_dir(tmp.path());
        assert!(!audit.secure_tpm_only);
        assert_eq!(audit.dot_env_found.len(), 1);
    }
}
