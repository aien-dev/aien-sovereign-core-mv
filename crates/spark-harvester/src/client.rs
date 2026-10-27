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

use aien_provenance::{GrantPermissions, SourceGrant};
use reqwest::Client;
use serde_json::{json, Value};
use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Network request failed: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Vault key resolution failed for {0}: {1}")]
    VaultKey(String, String),
    #[error("API returned error: {0}")]
    Api(String),
    #[error("Rights verification failed: {0}")]
    RightsDenied(String),
}

pub struct HarvesterClient {
    http: Client,
}

impl Default for HarvesterClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HarvesterClient {
    pub fn new() -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap(),
        }
    }

    pub fn resolve_vault_key(key_name: &str) -> Result<String, ClientError> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let vault_candidate = format!("{}/.local/bin/atlas-vault", home);
        let bin = if std::path::Path::new(&vault_candidate).is_file() {
            &vault_candidate
        } else {
            "atlas-vault"
        };
        let output = Command::new(bin).arg("get").arg(key_name).output();

        match output {
            Ok(out) if out.status.success() => {
                let secret = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if secret.is_empty() {
                    Err(ClientError::VaultKey(
                        key_name.to_string(),
                        "empty secret".to_string(),
                    ))
                } else {
                    Ok(secret)
                }
            }
            Ok(out) => Err(ClientError::VaultKey(
                key_name.to_string(),
                String::from_utf8_lossy(&out.stderr).trim().to_string(),
            )),
            Err(e) => Err(ClientError::VaultKey(key_name.to_string(), e.to_string())),
        }
    }

    pub(crate) async fn query_openai_compat(
        &self,
        endpoint_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<String, ClientError> {
        let payload = json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6
        });

        let res = self
            .http
            .post(endpoint_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            return Err(ClientError::Api(format!("{}: {}", status, text)));
        }

        let body: Value = res.json().await?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(content)
    }

    pub async fn query_with_grant(
        &self,
        endpoint_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
        grant: &SourceGrant,
        authority_verifying_key: &p256::ecdsa::VerifyingKey,
    ) -> Result<String, ClientError> {
        let now = chrono::Utc::now().timestamp() as u64;
        grant
            .verify_validity(now)
            .map_err(|e| ClientError::RightsDenied(e.to_string()))?;

        grant
            .verify_signature(authority_verifying_key)
            .map_err(|e| ClientError::RightsDenied(e.to_string()))?;

        grant
            .verify_permission(GrantPermissions::TRAIN | GrantPermissions::DISTILL)
            .or_else(|_| grant.verify_permission(GrantPermissions::EVALUATE))
            .map_err(|e| ClientError::RightsDenied(e.to_string()))?;

        self.query_openai_compat(endpoint_url, api_key, model, prompt)
            .await
    }
}
