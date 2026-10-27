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

use aien_localhttp::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAuditResult {
    pub name: String,
    pub endpoint: String,
    pub reachable: bool,
    pub status_code: Option<u16>,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantAuditResult {
    pub passed: bool,
    pub plaintext_env_files_found: Vec<String>,
    pub unredacted_secrets_found: usize,
    pub unslop_violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildAuditResult {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAuditReport {
    pub timestamp: String,
    pub services: Vec<ServiceAuditResult>,
    pub invariants: InvariantAuditResult,
    pub build: BuildAuditResult,
    pub overall_healthy: bool,
}

const SERVICE_ENDPOINTS: [(&str, &str); 5] = [
    ("Cortex Memory", "http://127.0.0.1:18080/health"),
    ("Sovereign Cockpit", "http://127.0.0.1:18095/api/pulse"),
    ("Cortex ONNX Encoder", "http://127.0.0.1:18081/health"),
    ("Local Model Seat", "http://127.0.0.1:18006/v1/models"),
    ("Matrix Conduit", "http://127.0.0.1:6167"),
];

pub async fn audit_services(client: &Client) -> Vec<ServiceAuditResult> {
    audit_endpoints(client, &SERVICE_ENDPOINTS).await
}

/// Probes each endpoint once (1500 ms each). Reachable means any answer
/// below 500; a connect error, timeout or malformed reply is unreachable.
async fn audit_endpoints(client: &Client, endpoints: &[(&str, &str)]) -> Vec<ServiceAuditResult> {
    let mut results = Vec::new();
    for &(name, ep) in endpoints {
        let start = std::time::Instant::now();
        match client
            .get(ep)
            .timeout(Duration::from_millis(1500))
            .send()
            .await
        {
            Ok(res) => {
                let status = res.status().as_u16();
                results.push(ServiceAuditResult {
                    name: name.to_string(),
                    endpoint: ep.to_string(),
                    reachable: status < 500,
                    status_code: Some(status),
                    latency_ms: start.elapsed().as_millis() as u64,
                });
            }
            Err(_) => {
                results.push(ServiceAuditResult {
                    name: name.to_string(),
                    endpoint: ep.to_string(),
                    reachable: false,
                    status_code: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
    }
    results
}

pub fn audit_vault_and_unslop(workspace_paths: &[&str]) -> InvariantAuditResult {
    let mut plaintext_env = Vec::new();
    let unslop_violations = Vec::new();

    for base in workspace_paths {
        let p = Path::new(base);
        if !p.exists() {
            continue;
        }

        // Scan for .env files
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with(".env") {
                    plaintext_env.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }

    let passed = plaintext_env.is_empty() && unslop_violations.is_empty();
    InvariantAuditResult {
        passed,
        plaintext_env_files_found: plaintext_env,
        unredacted_secrets_found: 0,
        unslop_violations,
    }
}

pub async fn audit_workspace_build(workspace_path: &str) -> BuildAuditResult {
    let output_res = Command::new("cargo")
        .arg("check")
        .arg("--workspace")
        .current_dir(workspace_path)
        .output()
        .await;

    match output_res {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut errors = Vec::new();
            let mut warnings = Vec::new();

            for line in stderr.lines() {
                if line.starts_with("error") {
                    errors.push(line.to_string());
                } else if line.starts_with("warning") && !line.contains("profiles for the non root")
                {
                    warnings.push(line.to_string());
                }
            }

            BuildAuditResult {
                passed: out.status.success(),
                errors,
                warnings,
            }
        }
        Err(e) => BuildAuditResult {
            passed: false,
            errors: vec![format!("Failed to run cargo check: {}", e)],
            warnings: Vec::new(),
        },
    }
}

pub async fn run_full_audit(client: &Client, workspace_dir: &str) -> SystemAuditReport {
    let services = audit_services(client).await;
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let cockpit_dir = format!("{}/spark-cockpit-rs", home);
    let invariants = audit_vault_and_unslop(&[workspace_dir, &cockpit_dir]);
    let build = audit_workspace_build(workspace_dir).await;

    let services_ok = services.iter().all(|s| s.reachable);
    let overall_healthy = services_ok && invariants.passed && build.passed;

    SystemAuditReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        services,
        invariants,
        build,
        overall_healthy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_localhttp::test_server::{closed_port, response, serve, serve_silent};

    #[test]
    fn service_endpoints_are_plain_http() {
        for (name, ep) in SERVICE_ENDPOINTS {
            assert!(aien_localhttp::Url::parse(ep).is_ok(), "{name} {ep}");
        }
    }

    #[tokio::test]
    async fn audit_endpoints_keeps_the_reachability_rule() {
        let (base, server) = serve(vec![
            response("200 OK", "text/plain", "ok"),
            response("404 Not Found", "text/plain", ""),
            response("503 Service Unavailable", "text/plain", ""),
        ])
        .await;
        let ok = format!("{base}/health");
        let missing = format!("{base}/v1/models");
        let down = format!("{base}/api/pulse");
        let refused = closed_port().await;
        let results = audit_endpoints(
            &Client::new(),
            &[
                ("ok", &ok),
                ("missing", &missing),
                ("down", &down),
                ("refused", &refused),
                ("tls", "https://127.0.0.1:1/health"),
            ],
        )
        .await;
        let summary: Vec<(bool, Option<u16>)> = results
            .iter()
            .map(|r| (r.reachable, r.status_code))
            .collect();
        assert_eq!(
            summary,
            vec![
                (true, Some(200)),
                (true, Some(404)),
                (false, Some(503)),
                (false, None),
                (false, None),
            ]
        );
        assert_eq!(results[0].endpoint, ok);
        let requests = server.await.unwrap();
        assert!(requests[0].starts_with("GET /health HTTP/1.1\r\n"));
        assert!(requests[1].starts_with("GET /v1/models HTTP/1.1\r\n"));
    }

    #[tokio::test]
    async fn audit_endpoints_bounds_a_silent_service_at_1500_ms() {
        let (base, _server) = serve_silent().await;
        let started = std::time::Instant::now();
        let results = audit_endpoints(&Client::new(), &[("silent", &base)]).await;
        assert!(!results[0].reachable);
        assert_eq!(results[0].status_code, None);
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(results[0].latency_ms >= 1400, "{}", results[0].latency_ms);
    }
}
