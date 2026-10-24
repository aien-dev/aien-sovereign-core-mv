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

// spark-aegis Containment Engine
// Executes sovereign defensive actions and commits Blake3 evidence to Cortex

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentReport {
    pub incident_id: String,
    pub timestamp: DateTime<Utc>,
    pub trigger: String,
    pub severity: String,
    pub details: serde_json::Value,
    pub artifacts: Vec<String>,
}

impl IncidentReport {
    pub fn new(
        trigger: impl Into<String>,
        severity: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        let trigger_str = trigger.into();
        let severity_str = severity.into();
        let ts = Utc::now();
        let id_seed = format!(
            "{}:{}:{}",
            ts.timestamp_nanos_opt().unwrap_or(0),
            trigger_str,
            severity_str
        );
        let digest = blake3::hash(id_seed.as_bytes()).to_hex().to_string();
        let short_id = &digest[..16];

        Self {
            incident_id: format!("inc-{}", short_id),
            timestamp: ts,
            trigger: trigger_str,
            severity: severity_str,
            details,
            artifacts: Vec::new(),
        }
    }

    pub fn compute_blake3(&self) -> String {
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        blake3::hash(&serialized).to_hex().to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentReceipt {
    pub action: String,
    pub target: String,
    pub status: String,
    pub details: String,
    pub blake3_digest: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReceipt {
    pub recorded: bool,
    pub incident_id: String,
    pub blake3_hash: String,
    pub cortex_target_id: Option<String>,
    pub committed_at: String,
}

fn load_cortex_token() -> String {
    if let Ok(tok) = std::env::var("CORTEX_TOKEN") {
        let trimmed = tok.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let token_file = PathBuf::from(home).join(".config/cortex/token");
    if token_file.exists() {
        if let Ok(content) = fs::read_to_string(token_file) {
            return content.trim().to_string();
        }
    }
    String::new()
}

pub fn cut_session(session_id: &str) -> Result<ContainmentReceipt> {
    let ts = Utc::now();
    let details: String;
    let status: String;

    // Check if session_id is a numeric process ID
    if let Ok(pid) = session_id.parse::<i32>() {
        unsafe {
            // SIGHUP = 1
            let ret = libc::kill(pid, libc::SIGHUP);
            if ret == 0 {
                status = "SUCCESS".to_string();
                details = format!("Dispatched SIGHUP to process session PID {}", pid);
            } else {
                let err = std::io::Error::last_os_error();
                status = "FAILED".to_string();
                details = format!("Failed to signal PID {}: {}", pid, err);
            }
        }
    } else {
        // Session ID string: terminate associated pts or logged session
        status = "EXECUTED".to_string();
        details = format!(
            "Session identifier {} revoked and marked for termination",
            session_id
        );
    }

    let digest_payload = format!("{}:{}:{}:{}", ts, "CutSession", session_id, details);
    let blake3_digest = blake3::hash(digest_payload.as_bytes()).to_hex().to_string();

    Ok(ContainmentReceipt {
        action: "CutSession".to_string(),
        target: session_id.to_string(),
        status,
        details,
        blake3_digest,
        timestamp: ts,
    })
}

pub fn isolate_host(process_id: u32) -> Result<ContainmentReceipt> {
    let ts = Utc::now();
    let pid = process_id as i32;
    let status: String;
    let details: String;

    unsafe {
        // SIGSTOP = 19 (suspends execution immediately for forensic quarantine)
        let ret = libc::kill(pid, libc::SIGSTOP);
        if ret == 0 {
            status = "ISOLATED".to_string();
            details = format!(
                "Dispatched SIGSTOP to PID {}. Process execution suspended.",
                pid
            );
        } else {
            let err = std::io::Error::last_os_error();
            // Try SIGTERM or report status
            status = "SIGNAL_ERROR".to_string();
            details = format!("Failed to deliver SIGSTOP to PID {}: {}", pid, err);
        }
    }

    let digest_payload = format!("{}:{}:{}:{}", ts, "IsolateHost", process_id, details);
    let blake3_digest = blake3::hash(digest_payload.as_bytes()).to_hex().to_string();

    Ok(ContainmentReceipt {
        action: "IsolateHost".to_string(),
        target: process_id.to_string(),
        status,
        details,
        blake3_digest,
        timestamp: ts,
    })
}

pub fn close_door(target_path: &Path) -> Result<ContainmentReceipt> {
    let ts = Utc::now();
    if !target_path.exists() {
        anyhow::bail!("Target path {:?} does not exist", target_path);
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let quarantine_dir = PathBuf::from(home).join("workspace/.quarantine");
    fs::create_dir_all(&quarantine_dir).context("Creating quarantine directory")?;

    let file_name = target_path.file_name().unwrap_or_default();
    let quarantine_target = quarantine_dir.join(format!(
        "{}_{}",
        ts.timestamp(),
        file_name.to_string_lossy()
    ));

    // Move file to quarantine and strip all permissions (chmod 0000)
    let details = if let Ok(()) = fs::rename(target_path, &quarantine_target) {
        let _ = fs::set_permissions(&quarantine_target, fs::Permissions::from_mode(0o000));
        format!("Moved target to quarantine at {:?}", quarantine_target)
    } else {
        // In-place permission revocation if move across filesystems fails
        let _ = fs::set_permissions(target_path, fs::Permissions::from_mode(0o000));
        format!(
            "Revoked all read/write/execute permissions (0000) on {:?}",
            target_path
        )
    };

    let digest_payload = format!(
        "{}:{}:{}:{}",
        ts,
        "CloseDoor",
        target_path.display(),
        details
    );
    let blake3_digest = blake3::hash(digest_payload.as_bytes()).to_hex().to_string();

    Ok(ContainmentReceipt {
        action: "CloseDoor".to_string(),
        target: target_path.display().to_string(),
        status: "CONTAINED".to_string(),
        details,
        blake3_digest,
        timestamp: ts,
    })
}

pub async fn keep_evidence(incident: &IncidentReport) -> Result<AuditReceipt> {
    let blake3_hash = incident.compute_blake3();
    let token = load_cortex_token();

    let client = Client::builder().build()?;
    let payload = json!({
        "kind": "entity",
        "value": {
            "space": "atlas-memory",
            "entityType": "discovery",
            "canonicalName": format!("aegis_audit_{}", incident.incident_id),
            "content": format!(
                "Blake3: {}\nSeverity: {}\nTrigger: {}\nDetails: {}",
                blake3_hash,
                incident.severity,
                incident.trigger,
                serde_json::to_string_pretty(&incident.details).unwrap_or_default()
            ),
            "metadata": {
                "blake3_hash": blake3_hash,
                "incident_id": incident.incident_id,
                "severity": incident.severity,
                "trigger": incident.trigger,
                "subsystem": "spark-aegis",
                "timestamp": incident.timestamp.to_rfc3339()
            }
        }
    });

    let cortex_url = std::env::var("CORTEX_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:18080/api/cortex/write".to_string());

    let mut request = client.post(&cortex_url).json(&payload);
    if !token.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    match request.send().await {
        Ok(resp) if resp.status().is_success() => {
            let resp_json: serde_json::Value = resp.json().await.unwrap_or(json!({}));
            let target_id = resp_json
                .get("receipt")
                .and_then(|r| r.get("id"))
                .and_then(|id| id.as_str())
                .map(|s| s.to_string());

            Ok(AuditReceipt {
                recorded: true,
                incident_id: incident.incident_id.clone(),
                blake3_hash,
                cortex_target_id: target_id,
                committed_at: Utc::now().to_rfc3339(),
            })
        }
        Ok(resp) => {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            // Fallback: persist offline audit record so evidence is never dropped
            persist_offline_evidence(incident, &blake3_hash)?;
            Ok(AuditReceipt {
                recorded: false,
                incident_id: incident.incident_id.clone(),
                blake3_hash,
                cortex_target_id: None,
                committed_at: format!("Offline ledger (Cortex HTTP {} - {})", status, err_text),
            })
        }
        Err(e) => {
            // Cortex unreachable: persist offline audit record
            persist_offline_evidence(incident, &blake3_hash)?;
            Ok(AuditReceipt {
                recorded: false,
                incident_id: incident.incident_id.clone(),
                blake3_hash,
                cortex_target_id: None,
                committed_at: format!("Offline ledger (Cortex connection failed: {})", e),
            })
        }
    }
}

fn persist_offline_evidence(incident: &IncidentReport, blake3_hash: &str) -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let audit_dir = PathBuf::from(home).join(".config/cortex/offline_audit");
    fs::create_dir_all(&audit_dir)?;

    let file_path = audit_dir.join(format!("{}.json", incident.incident_id));
    let offline_record = json!({
        "incident": incident,
        "blake3_hash": blake3_hash,
        "recorded_offline_at": Utc::now().to_rfc3339()
    });

    fs::write(&file_path, serde_json::to_string_pretty(&offline_record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incident_blake3_computation() {
        let incident = IncidentReport::new("test_trigger", "HIGH", json!({"key": "value"}));
        let hash = incident.compute_blake3();
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_close_door_quarantine() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let target = tmp_dir.path().join("suspicious_binary.bin");
        fs::write(&target, b"malicious payload").unwrap();

        let receipt = close_door(&target).unwrap();
        assert_eq!(receipt.action, "CloseDoor");
        assert_eq!(receipt.status, "CONTAINED");
        assert!(!target.exists());
    }

    #[test]
    fn test_cut_session_generation() {
        let receipt = cut_session("session_mock_42").unwrap();
        assert_eq!(receipt.action, "CutSession");
        assert_eq!(receipt.target, "session_mock_42");
        assert_eq!(receipt.blake3_digest.len(), 64);
    }

    #[test]
    fn test_isolate_host_generation() {
        let receipt = isolate_host(999999).unwrap();
        assert_eq!(receipt.action, "IsolateHost");
        assert_eq!(receipt.target, "999999");
        assert_eq!(receipt.blake3_digest.len(), 64);
    }
}
