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
use rusqlite::params;
use serde_json::json;
use tracing::{error, info};
use uuid::Uuid;

use crate::audit::SystemAuditReport;

const CORTEX_URL: &str = "http://127.0.0.1:18080";
fn debugger_cortex_token_path() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    home.join(".config/cortex/token")
}

fn debugger_hive_db_path() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    home.join(".config/cortex/hive.db")
}

/// POSTs the bug report to `{cortex_url}/api/cortex/write`. Only a transport
/// failure is an error; the response status is not checked (unchanged).
async fn post_bug_report(
    client: &Client,
    cortex_url: &str,
    token: &str,
    payload: &serde_json::Value,
) -> Result<(), aien_localhttp::Error> {
    client
        .post(format!("{}/api/cortex/write", cortex_url))
        .header("Authorization", format!("Bearer {}", token.trim()))
        .json(payload)
        .send()
        .await
        .map(|_| ())
}

pub async fn report_incident(client: &Client, report: &SystemAuditReport, failure_summary: &str) {
    info!("[DebuggerReporter] Reporting incident: {}", failure_summary);

    // 1. Commit bug report to Cortex memory
    if let Ok(token) = std::fs::read_to_string(debugger_cortex_token_path()) {
        let payload = json!({
            "space": "atlas-memory",
            "canonicalName": format!("incident-{}", Uuid::new_v4().to_string()[..8].to_string()),
            "entityType": "bug_report",
            "content": format!("Incident detected by spark-debugger: {}
        Report Details: {:?}", failure_summary, report)
        });

        if let Err(e) = post_bug_report(client, CORTEX_URL, &token, &payload).await {
            error!(
                "[DebuggerReporter] Failed to commit bug report to Cortex: {}",
                e
            );
        } else {
            info!("[DebuggerReporter] Committed bug report to Cortex.");
        }
    }

    // 2. Emit alert comb onto Honeycomb lattice if hive.db exists
    if debugger_hive_db_path().exists() {
        if let Ok(conn) = rusqlite::Connection::open(debugger_hive_db_path()) {
            let comb_id = format!("comb-alert-{}", &Uuid::new_v4().to_string()[..8]);
            let now = chrono::Utc::now().to_rfc3339();
            let content = format!("🚨 AUDIT ALERT: {}", failure_summary);

            // Find free coordinate or default to outer ring
            let res = conn.execute(
                "INSERT INTO hive_combs (id, q, r, author, role, content, intent, created_at)
                 VALUES (?1, -2, 2, ?2, ?3, ?4, ?5, ?6)",
                params![
                    comb_id,
                    "spark-debugger",
                    "watchdog",
                    content,
                    "branch",
                    now
                ],
            );
            if res.is_ok() {
                info!("[DebuggerReporter] Emitted alert comb to Honeycomb Wall.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_localhttp::test_server::{closed_port, response, serve_once};

    #[test]
    fn cortex_url_is_plain_http() {
        assert!(aien_localhttp::Url::parse(CORTEX_URL).is_ok());
    }

    #[tokio::test]
    async fn post_bug_report_sends_the_same_request() {
        let (base, server) = serve_once(response("200 OK", "application/json", "{}")).await;
        let payload = json!({"space": "atlas-memory", "entityType": "bug_report"});
        post_bug_report(&Client::new(), &base, "  tok\n", &payload)
            .await
            .unwrap();
        let request = server.await.unwrap();
        assert!(
            request.starts_with("POST /api/cortex/write HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request.contains("\r\nAuthorization: Bearer tok\r\n"),
            "{request}"
        );
        assert!(request.contains("\r\nContent-Type: application/json\r\n"));
        let body: serde_json::Value =
            serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body, payload);
    }

    #[tokio::test]
    async fn post_bug_report_ignores_status_but_reports_transport_failures() {
        let (base, _server) =
            serve_once(response("500 Internal Server Error", "text/plain", "")).await;
        assert!(post_bug_report(&Client::new(), &base, "t", &json!({}))
            .await
            .is_ok());

        let err = post_bug_report(&Client::new(), &closed_port().await, "t", &json!({}))
            .await
            .unwrap_err();
        assert!(err.is_connect(), "{err:?}");

        let err = post_bug_report(&Client::new(), "https://127.0.0.1:1", "t", &json!({}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, aien_localhttp::Error::HttpsRefused { .. }),
            "{err:?}"
        );
    }
}
