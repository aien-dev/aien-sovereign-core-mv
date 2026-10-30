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

use crate::models::EmailMessage;
use aien_localhttp::Client;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct CortexSync {
    cortex_url: String,
    client: Client,
    /// Tests only: read the token from here instead of ~/.config/cortex/token.
    token_path_override: Option<PathBuf>,
}

impl CortexSync {
    pub fn new(cortex_url: Option<&str>) -> Self {
        let url = cortex_url.unwrap_or("http://127.0.0.1:18080").to_string();
        let client = Client::builder().timeout(Duration::from_secs(3)).build();
        Self {
            cortex_url: url,
            client,
            token_path_override: None,
        }
    }

    fn get_cortex_token(&self) -> String {
        let token_path = match &self.token_path_override {
            Some(path) => path.clone(),
            None => std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".config/cortex/token"),
        };
        if token_path.exists() {
            fs::read_to_string(&token_path)
                .unwrap_or_default()
                .trim()
                .to_string()
        } else {
            String::new()
        }
    }

    pub async fn check_health(&self) -> bool {
        self.client
            .get(format!("{}/health", self.cortex_url))
            .timeout(Duration::from_millis(500))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub async fn commit_mail(&self, msg: &EmailMessage) -> Result<(), String> {
        let token = self.get_cortex_token();
        let body = if msg.untrusted {
            crate::effect::render_untrusted_body(&msg.body)
        } else {
            msg.body.clone()
        };
        let formatted_content = format!(
            "From: {}\nTo: {}\nSubject: {}\nDate: {}\nFolder: {}\n\n{}",
            msg.from,
            msg.to.join(", "),
            msg.subject,
            msg.received_at,
            msg.folder,
            body
        );

        let payload = json!({
            "kind": "entity",
            "value": {
                "canonicalName": format!("mail:{}", msg.id),
                "entityType": "correspondence",
                "content": formatted_content,
                "metadata": {
                    "mail_id": msg.id,
                    "subject": msg.subject,
                    "from": msg.from,
                    "to": msg.to,
                    "folder": msg.folder,
                    "received_at": msg.received_at,
                    "untrusted": msg.untrusted,
                    "tool_calls": [],
                    "instructions": null
                },
                "space": "atlas-memory"
            }
        });

        let mut req = self
            .client
            .post(format!("{}/api/cortex/write", self.cortex_url))
            .header("Content-Type", "application/json");

        if !token.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Cortex connection failed: {}", e))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Cortex write returned HTTP {}", resp.status()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_localhttp::test_server::{closed_port, response, serve_once, serve_silent};
    use std::collections::HashMap;

    fn sync_with_token(base: &str, token: Option<&str>) -> (CortexSync, PathBuf) {
        let path =
            std::env::temp_dir().join(format!("spark-mail-cortex-token-{}", uuid::Uuid::new_v4()));
        if let Some(token) = token {
            fs::write(&path, token).unwrap();
        }
        let mut sync = CortexSync::new(Some(base));
        sync.token_path_override = Some(path.clone());
        (sync, path)
    }

    fn message() -> EmailMessage {
        EmailMessage {
            id: "m1".into(),
            from: "a@example.test".into(),
            to: vec!["b@example.test".into()],
            subject: "Hi".into(),
            body: "hello".into(),
            headers: HashMap::new(),
            received_at: "2026-10-15T00:00:00Z".into(),
            folder: "INBOX".into(),
            cortex_indexed: false,
            untrusted: false,
        }
    }

    #[tokio::test]
    async fn check_health_is_true_only_for_2xx() {
        let (base, server) = serve_once(response("200 OK", "text/plain", "ok")).await;
        assert!(CortexSync::new(Some(&base)).check_health().await);
        assert!(
            server
                .await
                .unwrap()
                .starts_with("GET /health HTTP/1.1\r\n")
        );

        let (base, _server) =
            serve_once(response("503 Service Unavailable", "text/plain", "")).await;
        assert!(!CortexSync::new(Some(&base)).check_health().await);
        assert!(
            !CortexSync::new(Some(&closed_port().await))
                .check_health()
                .await
        );
        assert!(
            !CortexSync::new(Some("https://127.0.0.1:1"))
                .check_health()
                .await
        );
    }

    #[tokio::test]
    async fn check_health_gives_up_after_500_ms() {
        let (base, _server) = serve_silent().await;
        let started = std::time::Instant::now();
        assert!(!CortexSync::new(Some(&base)).check_health().await);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn commit_mail_sends_the_same_request_with_bearer_token() {
        let (base, server) = serve_once(response("200 OK", "application/json", "{}")).await;
        let (sync, path) = sync_with_token(&base, Some("tok\n"));
        sync.commit_mail(&message()).await.unwrap();
        let _ = fs::remove_file(&path);
        let request = server.await.unwrap();
        assert!(
            request.starts_with("POST /api/cortex/write HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request.contains("\r\nAuthorization: Bearer tok\r\n"),
            "{request}"
        );
        assert_eq!(request.matches("Content-Type: application/json").count(), 1);
        let body: serde_json::Value =
            serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["kind"], "entity");
        assert_eq!(body["value"]["canonicalName"], "mail:m1");
        assert_eq!(body["value"]["entityType"], "correspondence");
        assert_eq!(body["value"]["metadata"]["untrusted"], false);
    }

    #[tokio::test]
    async fn commit_mail_without_token_sends_no_authorization() {
        let (base, server) = serve_once(response("200 OK", "application/json", "{}")).await;
        let (sync, _path) = sync_with_token(&base, None);
        sync.commit_mail(&message()).await.unwrap();
        let request = server.await.unwrap();
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn commit_mail_reports_error_status_and_transport_failures() {
        let (base, _server) =
            serve_once(response("500 Internal Server Error", "text/plain", "")).await;
        let (sync, _path) = sync_with_token(&base, None);
        assert_eq!(
            sync.commit_mail(&message()).await.unwrap_err(),
            "Cortex write returned HTTP 500 Internal Server Error"
        );

        let (sync, _path) = sync_with_token(&closed_port().await, None);
        let err = sync.commit_mail(&message()).await.unwrap_err();
        assert!(err.starts_with("Cortex connection failed: "), "{err}");

        let (sync, _path) = sync_with_token("https://127.0.0.1:1", None);
        let err = sync.commit_mail(&message()).await.unwrap_err();
        assert!(err.contains("https is refused"), "{err}");
    }
}
