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

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

use crate::models::CortexSessionEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityClass {
    DoNotRemember,
    Credential,
    Authentication,
    Financial,
    Health,
    ThirdPartyPersonalData,
}

impl SensitivityClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DoNotRemember => "do_not_remember",
            Self::Credential => "credential",
            Self::Authentication => "authentication",
            Self::Financial => "financial",
            Self::Health => "health",
            Self::ThirdPartyPersonalData => "third_party_personal_data",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "do_not_remember" => Some(Self::DoNotRemember),
            "credential" => Some(Self::Credential),
            "authentication" => Some(Self::Authentication),
            "financial" => Some(Self::Financial),
            "health" => Some(Self::Health),
            "third_party_personal_data" => Some(Self::ThirdPartyPersonalData),
            _ => None,
        }
    }
}

static DO_NOT_REMEMBER_RE: OnceLock<Regex> = OnceLock::new();
static CREDENTIAL_RE: OnceLock<Regex> = OnceLock::new();
static AUTH_RE: OnceLock<Regex> = OnceLock::new();
static FINANCIAL_RE: OnceLock<Regex> = OnceLock::new();
static HEALTH_RE: OnceLock<Regex> = OnceLock::new();
static PII_RE: OnceLock<Regex> = OnceLock::new();

fn get_do_not_remember_re() -> &'static Regex {
    DO_NOT_REMEMBER_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:do not|don't|dont|never)\s+(?:remember|store|save|memorize)\b")
            .unwrap()
    })
}

fn get_credential_re() -> &'static Regex {
    CREDENTIAL_RE.get_or_init(|| {
        Regex::new(r#"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|passwd|private[_ -]?key|client[_ -]?secret)\b\s*(?::|=|is|was)\s*([^\s,;"']+)"#)
            .unwrap()
    })
}

fn get_auth_re() -> &'static Regex {
    AUTH_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:authorization:\s*bearer\s+[^\s]+|session[_ -]?cookie|one[- ]time password|\botp\b|mfa code)\b")
            .unwrap()
    })
}

fn get_financial_re() -> &'static Regex {
    FINANCIAL_RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:credit|debit) card\b|\b(?:routing|account) number\b|\b(?:cvv|iban|swift)\b",
        )
        .unwrap()
    })
}

fn get_health_re() -> &'static Regex {
    HEALTH_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:diagnosis|medical record|prescription|patient|therapy notes?|health insurance)\b")
            .unwrap()
    })
}

fn get_pii_re() -> &'static Regex {
    PII_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:social security|ssn|passport number|driver'?s license|home address)\b|\b(?:his|her|their|client|customer|employee|friend|coworker)'?s?\s+(?:email|phone|address|birthday)\b")
            .unwrap()
    })
}

pub struct SanitizationResult {
    pub sanitized_content: Option<String>,
    pub sanitized_payload: serde_json::Value,
    pub sensitivity: Option<String>,
    pub secret_fingerprint: Option<String>,
    pub redacted: bool,
}

pub struct SecurityMembrane;

impl SecurityMembrane {
    /// HMAC-SHA256 (RFC 2104) over the secret. The key never enters the stored fingerprint.
    pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
        const BLOCK: usize = 64;
        let mut key_block = [0u8; BLOCK];
        if key.len() > BLOCK {
            let digest = Sha256::digest(key);
            key_block[..32].copy_from_slice(&digest);
        } else {
            key_block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0x36u8; BLOCK];
        let mut opad = [0x5cu8; BLOCK];
        for i in 0..BLOCK {
            ipad[i] ^= key_block[i];
            opad[i] ^= key_block[i];
        }
        let mut inner = Sha256::new();
        inner.update(ipad);
        inner.update(message);
        let inner_hash = inner.finalize();
        let mut outer = Sha256::new();
        outer.update(opad);
        outer.update(inner_hash);
        let digest = outer.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    pub fn compute_secret_fingerprint(secret: &str, hmac_key: &[u8]) -> String {
        format!(
            "HMAC-SHA256:{}",
            crate::evidence::hex_encode(&Self::hmac_sha256(hmac_key, secret.trim().as_bytes(),))
        )
    }

    fn mask_text(text: &str, hmac_key: &[u8]) -> (String, Option<String>, Option<String>, bool) {
        if get_do_not_remember_re().is_match(text) {
            return (
                "<WITHHELD_BY_MEMORY_POLICY>".to_string(),
                Some("do_not_remember".to_string()),
                None,
                true,
            );
        }

        let mut out = text.to_string();
        let mut sensitivity: Option<String> = None;
        let mut fingerprint = None;
        let mut redacted = false;

        if let Some(caps) = get_credential_re().captures(&out) {
            let secret = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
            if !secret.is_empty() {
                fingerprint = Some(Self::compute_secret_fingerprint(&secret, hmac_key));
                out = out.replace(&secret, "<SECRET:TOKEN>");
                redacted = true;
            }
            sensitivity = Some("credential".to_string());
        }
        if get_auth_re().is_match(&out) {
            if fingerprint.is_none() {
                if let Some(m) = get_auth_re().find(&out) {
                    fingerprint = Some(Self::compute_secret_fingerprint(m.as_str(), hmac_key));
                }
            }
            out = get_auth_re()
                .replace_all(&out, "<SECRET:AUTH_TOKEN>")
                .to_string();
            if sensitivity.is_none() {
                sensitivity = Some("authentication".to_string());
            }
            redacted = true;
        }
        if get_financial_re().is_match(&out) {
            out = get_financial_re()
                .replace_all(&out, "<REDACTED:FINANCIAL>")
                .to_string();
            if sensitivity.is_none() {
                sensitivity = Some("financial".to_string());
            }
            redacted = true;
        }
        if get_health_re().is_match(&out) {
            let diagnosis_only = get_health_re().find_iter(&out).all(|m| {
                m.as_str().eq_ignore_ascii_case("diagnosis") && out.contains("diagnosis-protocol")
            });
            if !diagnosis_only {
                out = get_health_re()
                    .replace_all(&out, "<REDACTED:HEALTH>")
                    .to_string();
                if sensitivity.is_none() {
                    sensitivity = Some("health".to_string());
                }
                redacted = true;
            }
        }
        if get_pii_re().is_match(&out) {
            out = get_pii_re().replace_all(&out, "<REDACTED:PII>").to_string();
            if sensitivity.is_none() {
                sensitivity = Some("third_party_personal_data".to_string());
            }
            redacted = true;
        }
        (out, sensitivity, fingerprint, redacted)
    }

    fn sanitize_payload(
        value: &serde_json::Value,
        hmac_key: &[u8],
    ) -> (serde_json::Value, Option<String>, Option<String>, bool) {
        match value {
            serde_json::Value::String(text) => {
                let (masked, sensitivity, fingerprint, redacted) = Self::mask_text(text, hmac_key);
                (
                    serde_json::Value::String(masked),
                    sensitivity,
                    fingerprint,
                    redacted,
                )
            }
            serde_json::Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                let mut sensitivity = None;
                let mut fingerprint = None;
                let mut redacted = false;
                for item in items {
                    let (child, child_sensitivity, child_fp, child_redacted) =
                        Self::sanitize_payload(item, hmac_key);
                    out.push(child);
                    if sensitivity.is_none() {
                        sensitivity = child_sensitivity;
                    }
                    if fingerprint.is_none() {
                        fingerprint = child_fp;
                    }
                    redacted |= child_redacted;
                }
                (
                    serde_json::Value::Array(out),
                    sensitivity,
                    fingerprint,
                    redacted,
                )
            }
            serde_json::Value::Object(map) => {
                let mut out = serde_json::Map::new();
                let mut sensitivity = None;
                let mut fingerprint = None;
                let mut redacted = false;
                for (k, v) in map {
                    let (child, child_sensitivity, child_fp, child_redacted) =
                        Self::sanitize_payload(v, hmac_key);
                    out.insert(k.clone(), child);
                    if sensitivity.is_none() {
                        sensitivity = child_sensitivity;
                    }
                    if fingerprint.is_none() {
                        fingerprint = child_fp;
                    }
                    redacted |= child_redacted;
                }
                (
                    serde_json::Value::Object(out),
                    sensitivity,
                    fingerprint,
                    redacted,
                )
            }
            other => (other.clone(), None, None, false),
        }
    }

    /// Inspect incoming text and payload, mask secrets, and enforce memory directives.
    pub fn inspect_and_sanitize(
        content: Option<&str>,
        payload: &serde_json::Value,
        explicit_sensitivity: Option<&str>,
        hmac_key: &[u8],
    ) -> SanitizationResult {
        if explicit_sensitivity == Some("do_not_remember")
            || content.is_some_and(|t| get_do_not_remember_re().is_match(t))
        {
            return SanitizationResult {
                sanitized_content: content.map(|_| "<WITHHELD_BY_MEMORY_POLICY>".to_string()),
                sanitized_payload: serde_json::json!({ "redacted": true }),
                sensitivity: Some("do_not_remember".to_string()),
                secret_fingerprint: None,
                redacted: true,
            };
        }

        let (sanitized_payload, payload_sensitivity, payload_fp, payload_redacted) =
            Self::sanitize_payload(payload, hmac_key);

        let (sanitized_content, content_sensitivity, content_fp, content_redacted) = match content {
            Some(text) => {
                let (masked, sensitivity, fingerprint, redacted) = Self::mask_text(text, hmac_key);
                (Some(masked), sensitivity, fingerprint, redacted)
            }
            None => (None, None, None, false),
        };

        let mut sensitivity = content_sensitivity.or(payload_sensitivity);
        if sensitivity.is_none() {
            sensitivity = explicit_sensitivity.map(|s| s.to_string());
        }
        if explicit_sensitivity == Some("health") && sensitivity.as_deref() != Some("health") {
            sensitivity = Some("health".to_string());
        }
        if explicit_sensitivity == Some("financial") && sensitivity.is_none() {
            sensitivity = Some("financial".to_string());
        }
        if explicit_sensitivity == Some("third_party_personal_data") && sensitivity.is_none() {
            sensitivity = Some("third_party_personal_data".to_string());
        }
        if explicit_sensitivity == Some("authentication") && sensitivity.is_none() {
            sensitivity = Some("authentication".to_string());
        }

        let redacted = content_redacted
            || payload_redacted
            || matches!(
                sensitivity.as_deref(),
                Some(
                    "do_not_remember"
                        | "credential"
                        | "authentication"
                        | "financial"
                        | "health"
                        | "third_party_personal_data"
                )
            );

        SanitizationResult {
            sanitized_content,
            sanitized_payload,
            sensitivity,
            secret_fingerprint: content_fp.or(payload_fp),
            redacted,
        }
    }

    /// Project a raw database event into a SafeEventView.
    /// Processors must only access events through this view.
    pub fn to_safe_view(event: &CortexSessionEvent, hmac_key: &[u8]) -> SafeEventView {
        let inspected = Self::inspect_and_sanitize(
            event.content.as_deref(),
            &event.payload,
            event.sensitivity.as_deref(),
            hmac_key,
        );
        let is_do_not_remember = inspected.sensitivity.as_deref() == Some("do_not_remember");
        let is_credential = matches!(
            inspected.sensitivity.as_deref(),
            Some("credential" | "authentication")
        );
        let extraction_allowed = !inspected.redacted && !is_do_not_remember && !is_credential;
        let safe_content = if inspected.redacted || is_do_not_remember {
            None
        } else {
            inspected.sanitized_content
        };
        let payload = if inspected.redacted {
            serde_json::json!({ "redacted": true })
        } else {
            inspected.sanitized_payload
        };

        SafeEventView {
            event_id: event.id.clone(),
            session_id: event.session_id.clone(),
            sequence: event.sequence,
            branch_id: event.branch_id.clone(),
            event_type: event.event_type.clone(),
            role: event.role.clone(),
            safe_content,
            payload,
            sensitivity: inspected.sensitivity.or_else(|| event.sensitivity.clone()),
            extraction_allowed,
        }
    }
}

/// The sanitized, policy-filtered view of an event provided to background workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafeEventView {
    pub event_id: String,
    pub session_id: String,
    pub sequence: i64,
    pub branch_id: String,
    pub event_type: String,
    pub role: Option<String>,
    pub safe_content: Option<String>,
    pub payload: serde_json::Value,
    pub sensitivity: Option<String>,
    pub extraction_allowed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_do_not_remember_directive() {
        let res = SecurityMembrane::inspect_and_sanitize(
            Some("Please do not remember this confidential exchange"),
            &serde_json::json!({}),
            None,
            b"test-hmac-key",
        );
        assert_eq!(res.sensitivity, Some("do_not_remember".to_string()));
        assert_eq!(
            res.sanitized_content,
            Some("<WITHHELD_BY_MEMORY_POLICY>".to_string())
        );
    }

    #[test]
    fn test_credential_masking_and_fingerprint() {
        let key = b"test-hmac-key";
        let res = SecurityMembrane::inspect_and_sanitize(
            Some("Connecting with api_key: sk_live_99418294719247192 to server"),
            &serde_json::json!({"token": "api_key: sk_live_99418294719247192"}),
            None,
            key,
        );
        assert_eq!(res.sensitivity, Some("credential".to_string()));
        assert!(res.sanitized_content.unwrap().contains("<SECRET:TOKEN>"));
        assert!(!res
            .sanitized_payload
            .to_string()
            .contains("sk_live_99418294719247192"));
        let fingerprint = res.secret_fingerprint.unwrap();
        assert!(fingerprint.starts_with("HMAC-SHA256:"));
        let salted = {
            let mut hasher = Sha256::new();
            hasher.update(b"cortex_secret_salt:");
            hasher.update(b"sk_live_99418294719247192");
            format!("HMAC-SHA256:{:x}", hasher.finalize())
        };
        assert_ne!(fingerprint, salted);
    }

    #[test]
    fn test_health_and_pii_classification() {
        let res1 = SecurityMembrane::inspect_and_sanitize(
            Some("Patient medical record notes: high blood pressure"),
            &serde_json::json!({}),
            None,
            b"test-hmac-key",
        );
        assert_eq!(res1.sensitivity, Some("health".to_string()));

        let res2 = SecurityMembrane::inspect_and_sanitize(
            Some("Her email is sarah@example.com for communication"),
            &serde_json::json!({}),
            None,
            b"test-hmac-key",
        );
        assert_eq!(
            res2.sensitivity,
            Some("third_party_personal_data".to_string())
        );
    }

    #[test]
    fn test_safe_event_view_quarantine() {
        let sensitive_ev = CortexSessionEvent {
            id: "ev-s1".to_string(),
            session_id: "sess-1".to_string(),
            sequence: 1,
            branch_id: "main".to_string(),
            parent_event_id: None,
            event_type: "user_message".to_string(),
            role: Some("user".to_string()),
            content: Some("<WITHHELD_BY_MEMORY_POLICY>".to_string()),
            payload: serde_json::json!({}),
            created_at: "2026-09-21T00:00:00Z".to_string(),
            content_hash: "hash".to_string(),
            sensitivity: Some("do_not_remember".to_string()),
            redacted: false,
            segment_id: None,
        };

        let view = SecurityMembrane::to_safe_view(&sensitive_ev, b"test-hmac-key");
        assert!(!view.extraction_allowed);
        assert_eq!(view.safe_content, None);
    }

    #[test]
    fn test_hmac_sha256_rfc4231_case1() {
        let key = [0x0bu8; 20];
        let mac = SecurityMembrane::hmac_sha256(&key, b"Hi There");
        assert_eq!(
            crate::evidence::hex_encode(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }
}
