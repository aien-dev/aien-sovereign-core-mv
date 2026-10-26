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
use std::collections::HashMap;
use std::process::Command;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

static VAULT_CACHE: OnceLock<RwLock<(Instant, Vec<String>)>> = OnceLock::new();
static RESOLVED_SECRETS_CACHE: OnceLock<RwLock<HashMap<String, (Instant, String)>>> =
    OnceLock::new();
static SCRUB_TARGETS_CACHE: OnceLock<RwLock<(Instant, Vec<String>)>> = OnceLock::new();
static TEST_SECRETS: OnceLock<RwLock<HashMap<String, String>>> = OnceLock::new();

/// How long vault key lists, resolved secrets and scrub targets are cached before the TPM vault
/// is asked again.
const CACHE_TTL: Duration = Duration::from_secs(60);

/// List all secret keys currently registered in the hardware TPM vault.
/// Results are cached for `CACHE_TTL` (60 seconds) to avoid repeated TPM queries.
pub fn list_vault_keys() -> Vec<String> {
    let cache_lock =
        VAULT_CACHE.get_or_init(|| RwLock::new((Instant::now() - CACHE_TTL, Vec::new())));

    if let Ok(read_guard) = cache_lock.read() {
        if read_guard.0.elapsed() < CACHE_TTL {
            return read_guard.1.clone();
        }
    }

    if let Ok(mut write_guard) = cache_lock.write() {
        if write_guard.0.elapsed() < CACHE_TTL {
            return write_guard.1.clone();
        }
        let keys = fetch_vault_keys_raw();
        *write_guard = (Instant::now(), keys.clone());
        return keys;
    }

    fetch_vault_keys_raw()
}

fn fetch_vault_keys_raw() -> Vec<String> {
    if let Ok(out) = Command::new("atlas-vault").arg("list").output() {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

fn fetch_secret_raw(key_name: &str) -> Option<String> {
    if let Ok(out) = Command::new("atlas-vault").args(["get", key_name]).output() {
        if out.status.success() {
            let val = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !val.is_empty() {
                return Some(val);
            }
        }
    }
    None
}

/// Dynamically resolve a secret token.
/// Checks process environment first, then in-memory test secrets, then cache (`CACHE_TTL`, 60 seconds), then hardware TPM vault.
pub fn resolve_secret(key_name: &str) -> Option<String> {
    if key_name.trim().is_empty() {
        return None;
    }

    // 1. Check process environment first
    if let Ok(env_val) = std::env::var(key_name) {
        let trimmed = env_val.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }

    // 2. Check in-memory test secrets store
    if let Some(test_lock) = TEST_SECRETS.get() {
        if let Ok(guard) = test_lock.read() {
            if let Some(val) = guard.get(key_name) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                } else {
                    return None;
                }
            }
        }
    }

    // 3. Cache lookup (`CACHE_TTL`)
    let cache_lock = RESOLVED_SECRETS_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(guard) = cache_lock.read() {
        if let Some((fetched_at, val)) = guard.get(key_name) {
            if fetched_at.elapsed() < CACHE_TTL {
                return Some(val.clone());
            }
        }
    }

    // 4. Query hardware TPM-bound vault if key is registered
    let keys = list_vault_keys();
    if keys.iter().any(|k| k == key_name) {
        if let Some(val) = fetch_secret_raw(key_name) {
            if let Ok(mut write_guard) = cache_lock.write() {
                write_guard.insert(key_name.to_string(), (Instant::now(), val.clone()));
            }
            return Some(val);
        }
    }

    None
}

/// Check if a secret key is registered without exposing its value.
pub fn is_secret_present(key_name: &str) -> bool {
    if key_name.trim().is_empty() {
        return false;
    }

    if let Ok(env_val) = std::env::var(key_name) {
        if !env_val.trim().is_empty() {
            return true;
        }
    }

    if let Some(test_lock) = TEST_SECRETS.get() {
        if let Ok(guard) = test_lock.read() {
            if let Some(val) = guard.get(key_name) {
                return !val.trim().is_empty();
            }
        }
    }

    let keys = list_vault_keys();
    keys.iter().any(|k| k == key_name)
}

/// Register a test secret in memory for test verification without physical TPM interaction.
pub fn register_test_secret(key_name: &str, value: &str) {
    let test_lock = TEST_SECRETS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(mut guard) = test_lock.write() {
        guard.insert(key_name.to_string(), value.to_string());
    }

    // Update resolved secrets cache
    let cache_lock = RESOLVED_SECRETS_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(mut guard) = cache_lock.write() {
        if value.trim().is_empty() {
            guard.remove(key_name);
        } else {
            guard.insert(key_name.to_string(), (Instant::now(), value.to_string()));
        }
    }

    // Incremental update to scrub targets cache so valid cached TPM keys are preserved
    if let Some(scrub_lock) = SCRUB_TARGETS_CACHE.get() {
        if let Ok(mut guard) = scrub_lock.write() {
            if guard.0.elapsed() < CACHE_TTL {
                let trimmed = value.trim();
                if trimmed.len() >= 6 {
                    if !guard.1.contains(&trimmed.to_string()) {
                        guard.1.push(trimmed.to_string());
                        guard.1.sort();
                        guard.1.dedup();
                        guard.1.sort_by_key(|a| std::cmp::Reverse(a.len()));
                    }
                } else {
                    guard.1.retain(|s| s != trimmed);
                }
                guard.0 = Instant::now();
            } else {
                guard.0 = Instant::now() - CACHE_TTL;
                guard.1.clear();
            }
        }
    }
}

/// Remove a test secret from memory and purge from caches.
pub fn remove_test_secret(key_name: &str) {
    if let Some(test_lock) = TEST_SECRETS.get() {
        if let Ok(mut guard) = test_lock.write() {
            guard.remove(key_name);
        }
    }
    if let Some(cache_lock) = RESOLVED_SECRETS_CACHE.get() {
        if let Ok(mut guard) = cache_lock.write() {
            guard.remove(key_name);
        }
    }
}

/// Retrieve all active secret values of length >= 6 for outbound scrubbing.
/// Cached in memory for `CACHE_TTL` (60 seconds).
pub fn get_scrub_targets() -> Vec<String> {
    let cache_lock =
        SCRUB_TARGETS_CACHE.get_or_init(|| RwLock::new((Instant::now() - CACHE_TTL, Vec::new())));

    if let Ok(read_guard) = cache_lock.read() {
        if read_guard.0.elapsed() < CACHE_TTL {
            return read_guard.1.clone();
        }
    }

    if let Ok(mut write_guard) = cache_lock.write() {
        if write_guard.0.elapsed() < CACHE_TTL {
            return write_guard.1.clone();
        }

        let mut targets = Vec::new();

        // 1. Environment secrets for known model providers and services
        let known_env_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
            "DEEPSEEK_API_KEY",
            "CHATGPT_SESSION_TOKEN",
            "CLAUDE_SESSION_KEY",
            "CORTEX_TOKEN",
        ];
        for k in known_env_keys {
            if let Ok(val) = std::env::var(k) {
                let trimmed = val.trim().to_string();
                if trimmed.len() >= 6 {
                    targets.push(trimmed);
                }
            }
        }

        // 2. Hardware TPM-bound vault keys
        let vault_keys = list_vault_keys();
        for k in vault_keys {
            if let Some(val) = resolve_secret(&k) {
                if val.len() >= 6 {
                    targets.push(val);
                }
            }
        }

        // 3. Test secrets
        if let Some(test_lock) = TEST_SECRETS.get() {
            if let Ok(guard) = test_lock.read() {
                for val in guard.values() {
                    if val.len() >= 6 {
                        targets.push(val.clone());
                    }
                }
            }
        }

        // Deduplicate and sort descending by length to prevent partial replacement bugs
        targets.sort();
        targets.dedup();
        targets.sort_by_key(|a| std::cmp::Reverse(a.len()));

        *write_guard = (Instant::now(), targets.clone());
        return targets;
    }

    Vec::new()
}

/// Scrub sensitive host paths, local subnets, vault secrets, and operator identity from outbound prompts.
pub fn sanitize_outbound_prompt(prompt: &str) -> String {
    static PATH_RE: OnceLock<Regex> = OnceLock::new();
    static IP_RE: OnceLock<Regex> = OnceLock::new();
    static HOST_RE: OnceLock<Regex> = OnceLock::new();
    static SECRET_RE: OnceLock<Regex> = OnceLock::new();
    static OPERATOR_RE: OnceLock<Regex> = OnceLock::new();

    let path_re = PATH_RE.get_or_init(|| {
        Regex::new(r"/(?:home|Users)/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*").unwrap()
    });
    let ip_re = IP_RE.get_or_init(|| {
        Regex::new(r"\b(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|0\.0\.0\.0)\b").unwrap()
    });
    let host_re = HOST_RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:spark-[0-9a-f]{4}(?:\.local)?|spark\.local|localhost)\b|(?i)\bspark([^\w-]|\z)",
        )
        .unwrap()
    });
    let secret_re = SECRET_RE.get_or_init(|| {
        Regex::new(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|\b(?:sk-ant-[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{30,}|hf_[a-zA-Z0-9_]{30,}|ghp_[a-zA-Z0-9]{20,}|gho_[a-zA-Z0-9]{20,}|ey[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})\b").unwrap()
    });
    let operator_re = OPERATOR_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:[a-zA-Z0-9._%+-]*(?:drake\.stapleton|drake\.aien|drakestapleton|ballentine|aien)[a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|michael\s+drake\s+ballentine|drake\s+ballentine|michael\s+ballentine|drake\s+stapleton|ballentine|drakestapleton|drake\.stapleton)\b").unwrap()
    });

    // 1. Filesystem paths
    let s1 = path_re.replace_all(prompt, "<WORKSPACE_PATH>");

    // 2. Private IP addresses
    let s2 = ip_re.replace_all(&s1, "<LOCAL_HOST>");

    // 3. Local hostnames (preserves compound names like spark-adapters, spark-distill)
    let s3 = host_re.replace_all(&s2, "<LOCAL_HOST>$1");

    // 4. Dynamic registered vault secrets (exact literal replacement)
    let mut s4 = s3.into_owned();
    for secret in get_scrub_targets() {
        if s4.contains(&secret) {
            s4 = s4.replace(&secret, "[REDACTED_BY_ATLAS_VAULT]");
        }
    }

    // 5. Well-known API key signature patterns & PEM keys
    let s5 = secret_re.replace_all(&s4, "[REDACTED_BY_ATLAS_VAULT]");

    // 6. Operator personal identifiers & emails
    let s6 = operator_re.replace_all(&s5, "<OPERATOR>");

    s6.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_paths_and_ips() {
        let raw = "Read /home/alice/workspace/secret.rs from 100.100.1.2 and send to 127.0.0.1";
        let clean = sanitize_outbound_prompt(raw);
        assert!(!clean.contains("/home/alice"));
        assert!(!clean.contains("100.100.1.2"));
        assert!(!clean.contains("127.0.0.1"));
        assert!(clean.contains("<WORKSPACE_PATH>"));
        assert!(clean.contains("<LOCAL_HOST>"));
    }

    #[test]
    fn test_sanitize_operator_variants() {
        assert_eq!(
            sanitize_outbound_prompt("Author: Drake Stapleton"),
            "Author: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Author: drake stapleton"),
            "Author: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Former: Michael Drake Ballentine"),
            "Former: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Former: Drake Ballentine"),
            "Former: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Surname: Ballentine"),
            "Surname: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("User: drakestapleton"),
            "User: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Handle: drake.stapleton"),
            "Handle: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Email: aien@aienos.com"),
            "Email: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Email: drake.stapleton@example.com"),
            "Email: <OPERATOR>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Bracket: <drake.stapleton@example.com>"),
            "Bracket: <<OPERATOR>>"
        );
    }

    #[test]
    fn test_sanitize_operator_negative_cases() {
        assert_eq!(
            sanitize_outbound_prompt(
                "The Drake equation calculates extraterrestrial probabilities"
            ),
            "The Drake equation calculates extraterrestrial probabilities"
        );
        assert_eq!(
            sanitize_outbound_prompt("Use a staple to bind documents"),
            "Use a staple to bind documents"
        );
    }

    #[test]
    fn test_sanitize_hostnames() {
        assert_eq!(
            sanitize_outbound_prompt("Connecting to spark-1a2b on port 18080"),
            "Connecting to <LOCAL_HOST> on port 18080"
        );
        assert_eq!(
            sanitize_outbound_prompt("Connecting to spark-1a2b.local"),
            "Connecting to <LOCAL_HOST>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Connecting to spark.local"),
            "Connecting to <LOCAL_HOST>"
        );
        assert_eq!(
            sanitize_outbound_prompt("Connecting to localhost:8080"),
            "Connecting to <LOCAL_HOST>:8080"
        );
        assert_eq!(
            sanitize_outbound_prompt("ssh drakestapleton@spark"),
            "ssh <OPERATOR>@<LOCAL_HOST>"
        );
        assert_eq!(
            sanitize_outbound_prompt("http://spark:18080/api"),
            "http://<LOCAL_HOST>:18080/api"
        );
        assert_eq!(
            sanitize_outbound_prompt("Run target on spark"),
            "Run target on <LOCAL_HOST>"
        );
    }

    #[test]
    fn test_preserve_crate_and_binary_names() {
        assert_eq!(
            sanitize_outbound_prompt("Update crates/spark-adapters/src/vault.rs"),
            "Update crates/spark-adapters/src/vault.rs"
        );
        assert_eq!(
            sanitize_outbound_prompt("Run spark-distill crawl --track systems"),
            "Run spark-distill crawl --track systems"
        );
        assert_eq!(
            sanitize_outbound_prompt(
                "Crates: spark-mask, spark-supervisor, spark-hive, spark-dream"
            ),
            "Crates: spark-mask, spark-supervisor, spark-hive, spark-dream"
        );
        assert_eq!(
            sanitize_outbound_prompt("Check spark_adapters module and sparkling water"),
            "Check spark_adapters module and sparkling water"
        );
    }

    #[test]
    fn test_sanitize_secrets() {
        assert_eq!(
            sanitize_outbound_prompt(
                "Anthropic: sk-ant-api03-123456789012345678901234567890123456"
            ),
            "Anthropic: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("OpenAI: sk-proj-123456789012345678901234567890"),
            "OpenAI: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("Gemini: AIzaSyD-1234567890123456789012345678901"),
            "Gemini: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("HuggingFace: hf_abcdefghijklmnopqrstuvwxyz12345678"),
            "HuggingFace: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("GitHub PAT: ghp_123456789012345678901234567890"),
            "GitHub PAT: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("GitHub OAuth: gho_123456789012345678901234567890"),
            "GitHub OAuth: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt("JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
            "JWT: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt(
                "Key: -----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0\n-----END RSA PRIVATE KEY-----"
            ),
            "Key: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt(
                "Key: -----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjE\n-----END OPENSSH PRIVATE KEY-----"
            ),
            "Key: [REDACTED_BY_ATLAS_VAULT]"
        );
        assert_eq!(
            sanitize_outbound_prompt(
                "Key: -----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
            ),
            "Key: [REDACTED_BY_ATLAS_VAULT]"
        );
    }

    #[test]
    fn test_complex_multimodal_sanitization() {
        let raw = "Operator Drake Stapleton running on spark-1a2b with key sk-ant-api03-abcdefghijklmnopqrstuvwxyz12345 accessing /home/alice/secret.txt at 192.168.0.50 and spark:18080";
        let clean = sanitize_outbound_prompt(raw);
        assert_eq!(
            clean,
            "Operator <OPERATOR> running on <LOCAL_HOST> with key [REDACTED_BY_ATLAS_VAULT] accessing <WORKSPACE_PATH> at <LOCAL_HOST> and <LOCAL_HOST>:18080"
        );
    }

    #[test]
    fn test_dynamic_vault_secret_scrubbing() {
        register_test_secret("SUPER_SECRET_KEY", "quantum_secure_vault_passphrase_99182");
        let raw = "Deploy using passphrase quantum_secure_vault_passphrase_99182 immediately.";
        let clean = sanitize_outbound_prompt(raw);
        assert!(!clean.contains("quantum_secure_vault_passphrase_99182"));
        assert!(clean.contains("[REDACTED_BY_ATLAS_VAULT]"));
    }

    #[test]
    fn test_short_secrets_not_scrubbed() {
        register_test_secret("SHORT_SECRET", "abcde");
        let raw = "The word abcde is normal text.";
        let clean = sanitize_outbound_prompt(raw);
        assert!(clean.contains("abcde"));
        assert!(!clean.contains("[REDACTED_BY_ATLAS_VAULT]"));
    }

    #[test]
    fn test_sanitize_operator_and_secrets() {
        let raw = "Operator drakestapleton deployed key sk-proj-123456789012345678901234 for aien@aienos.com";
        let clean = sanitize_outbound_prompt(raw);
        assert!(!clean.contains("drakestapleton"));
        assert!(!clean.contains("aien@aienos.com"));
        assert!(clean.contains("<OPERATOR>"));
        assert!(clean.contains("[REDACTED_BY_ATLAS_VAULT]"));
    }
}
