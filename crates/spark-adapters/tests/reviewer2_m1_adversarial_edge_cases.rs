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

// Reviewer 2 Milestone 1 Adversarial Edge Cases Suite
// Tests prompt sanitization under extreme boundary conditions:
// prompt containing only secrets, mixed unicodes, escaped characters,
// multiline PEM blocks, and nested whitespace.
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::vault::{register_test_secret, sanitize_outbound_prompt};

#[test]
fn test_edge_case_only_secrets() {
    let secret = "sk-ant-api03-123456789012345678901234567890123456";
    let output = sanitize_outbound_prompt(secret);
    assert_eq!(output, "[REDACTED_BY_ATLAS_VAULT]");

    // Only dynamic vault secret
    let dynamic_secret = "vault_token_alpha_numeric_9988776655";
    register_test_secret("DYN_EDGE_ONLY", dynamic_secret);
    let output_dyn = sanitize_outbound_prompt(dynamic_secret);
    assert_eq!(output_dyn, "[REDACTED_BY_ATLAS_VAULT]");

    // Multiple consecutive secrets with no separator
    let dual_secrets = format!("{secret}sk-proj-123456789012345678901234567890");
    let output_dual = sanitize_outbound_prompt(&dual_secrets);
    assert!(!output_dual.contains(secret));
    assert!(!output_dual.contains("sk-proj-"));
}

#[test]
fn test_edge_case_mixed_unicodes() {
    let secret = "sk-ant-api03-123456789012345678901234567890123456";
    let input = format!(
        "你好世界: {secret} 🔒 操作员: Drake Stapleton 路径: /home/alice/模型.bin 主机: spark-1a2b"
    );
    let output = sanitize_outbound_prompt(&input);

    assert!(!output.contains(secret));
    assert!(!output.contains("Drake Stapleton"));
    assert!(!output.contains("/home/alice"));
    assert!(!output.contains("spark-1a2b"));

    assert!(output.contains("[REDACTED_BY_ATLAS_VAULT]"));
    assert!(output.contains("<OPERATOR>"));
    assert!(output.contains("<WORKSPACE_PATH>"));
    assert!(output.contains("<LOCAL_HOST>"));
    assert!(output.contains("你好世界:"));
    assert!(output.contains("🔒"));
}

#[test]
fn test_edge_case_escaped_characters() {
    // JSON-escaped string with quotes and newlines
    let input = r#"{\"key\": \"sk-proj-123456789012345678901234567890\", \"path\": \"/home/alice/config.json\", \"host\": \"spark\"}"#;
    let output = sanitize_outbound_prompt(input);

    assert!(!output.contains("sk-proj-"));
    assert!(!output.contains("/home/alice"));
    assert!(output.contains(r#"{\"key\": \"[REDACTED_BY_ATLAS_VAULT]\""#));
    assert!(output.contains(r#"\"path\": \"<WORKSPACE_PATH>\""#));
    assert!(output.contains(r#"\"host\": \"<LOCAL_HOST>\""#));
}

#[test]
fn test_edge_case_multiline_pem_blocks() {
    let rsa_pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0m4g...sample...fake...key...\n-----END RSA PRIVATE KEY-----";
    let ec_pem = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIIz...sample...fake...key...\n-----END EC PRIVATE KEY-----";
    let openssh_pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNza...sample...fake...key...\n-----END OPENSSH PRIVATE KEY-----";
    let generic_pem = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBAD...sample...fake...key...\n-----END PRIVATE KEY-----";

    for pem in [rsa_pem, ec_pem, openssh_pem, generic_pem] {
        let text = format!(
            "Here is the server configuration:\n{}\nProceed with caution.",
            pem
        );
        let output = sanitize_outbound_prompt(&text);
        assert!(!output.contains("PRIVATE KEY"));
        assert!(output.contains("[REDACTED_BY_ATLAS_VAULT]"));
        assert_eq!(
            output,
            "Here is the server configuration:\n[REDACTED_BY_ATLAS_VAULT]\nProceed with caution."
        );
    }
}

#[test]
fn test_edge_case_nested_whitespace() {
    let input = "\t\t\n\n   \t  Drake   Stapleton   \t\n  /home/alice/repo  \t\n  sk-ant-api03-123456789012345678901234567890123456  \n\n\t";
    let output = sanitize_outbound_prompt(input);

    assert!(!output.contains("alice"));
    assert!(!output.contains("sk-ant-"));
    assert!(output.contains("<WORKSPACE_PATH>"));
    assert!(output.contains("[REDACTED_BY_ATLAS_VAULT]"));
}
