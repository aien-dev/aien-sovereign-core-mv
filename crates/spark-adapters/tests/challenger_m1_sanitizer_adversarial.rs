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

// Challenger Milestone 1 Adversarial Stress Suite:
// Operator Name Sanitization, Hostname Boundaries, Secret Redaction, and System Prompt Leakage
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
use spark_adapters::vault::{register_test_secret, sanitize_outbound_prompt};

// ============================================================================
// Challenge 1: Operator Name Boundaries and False Positive Resistance
// ============================================================================

#[test]
fn test_challenge1_operator_case_and_whitespace_variants() {
    let variants = [
        "Drake Stapleton",
        "drake stapleton",
        "DRAKE STAPLETON",
        "Drake   Stapleton",
        "Drake\tStapleton",
        "Michael Drake Ballentine",
        "michael drake ballentine",
        "Drake Ballentine",
        "Michael Ballentine",
        "Ballentine",
        "drakestapleton",
        "drake.stapleton",
        "drake.stapleton@example.com",
        "aien@aienos.com",
    ];

    for variant in variants {
        let input = format!("Task submitted by {}", variant);
        let cleaned = sanitize_outbound_prompt(&input);
        assert!(
            !cleaned.contains(variant),
            "Variant '{}' was not scrubbed in output: '{}'",
            variant,
            cleaned
        );
        assert!(
            cleaned.contains("<OPERATOR>"),
            "Variant '{}' did not produce <OPERATOR> tag in output: '{}'",
            variant,
            cleaned
        );
    }
}

#[test]
fn test_challenge1_operator_punctuation_and_delimiter_wrapping() {
    let inputs = [
        ("Assignee: (Drake Stapleton)", "Assignee: (<OPERATOR>)"),
        ("Contact: [drake.stapleton@example.com]", "Contact: [<OPERATOR>]"),
        ("Author: 'Drake Stapleton';", "Author: '<OPERATOR>';"),
        (
            "Sign-off by \"Michael Drake Ballentine\".",
            "Sign-off by \"<OPERATOR>\".",
        ),
        (
            "User <drakestapleton> authorized run",
            "User <<OPERATOR>> authorized run",
        ),
    ];

    for (raw, expected) in inputs {
        let cleaned = sanitize_outbound_prompt(raw);
        assert_eq!(
            cleaned, expected,
            "Punctuation wrapping mismatch for raw input '{}'",
            raw
        );
    }
}

#[test]
fn test_challenge1_operator_negative_preservation() {
    // English terms and technical nouns containing substrings must never be corrupted
    let benign_texts = [
        "The Drake equation calculates extraterrestrial communications probability",
        "Use a heavy-duty staple to bind the document bundle",
        "The stapler is located on the desk beside the printer",
        "Botanical specimens include mandrake roots and wildflowers",
        "The ballistics report confirmed the trajectory",
    ];

    for text in benign_texts {
        let cleaned = sanitize_outbound_prompt(text);
        assert_eq!(
            cleaned, text,
            "Benign text must not be corrupted by operator regex: '{}' -> '{}'",
            text, cleaned
        );
    }
}

// ============================================================================
// Challenge 2: Hostname and Network Isolation Boundaries
// ============================================================================

#[test]
fn test_challenge2_hostname_and_url_variations() {
    let cases = [
        (
            "Connect to http://spark:8080/api/v1",
            "Connect to http://<LOCAL_HOST>:8080/api/v1",
        ),
        (
            "Access grpc endpoint at spark.local:18080",
            "Access grpc endpoint at <LOCAL_HOST>:18080",
        ),
        (
            "Target machine is spark-1a2b:18095",
            "Target machine is <LOCAL_HOST>:18095",
        ),
        (
            "Local binding on http://localhost:3000",
            "Local binding on http://<LOCAL_HOST>:3000",
        ),
        (
            "Remote execution via ssh drakestapleton@spark",
            "Remote execution via ssh <OPERATOR>@<LOCAL_HOST>",
        ),
        (
            "Connecting to spark-1a2b.local now",
            "Connecting to <LOCAL_HOST> now",
        ),
    ];

    for (raw, expected) in cases {
        let cleaned = sanitize_outbound_prompt(raw);
        assert_eq!(
            cleaned, expected,
            "Hostname scrubbing mismatch for input '{}': got '{}'",
            raw, cleaned
        );
    }
}

#[test]
fn test_challenge2_preserve_compound_identifiers() {
    // System crates and names starting with 'spark' or containing 'spark' must remain untouched
    let compound_terms = [
        "Update crates/spark-adapters/src/vault.rs",
        "Execute spark-distill crawl --track systems",
        "Module spark-mask provides mask generation",
        "Background services: spark-supervisor, spark-hive, spark-dream",
        "Inspect spark-cockpit-rs dashboard",
        "Enjoying a glass of sparkling water while compiling",
    ];

    for term in compound_terms {
        let cleaned = sanitize_outbound_prompt(term);
        assert_eq!(
            cleaned, term,
            "Compound term must not be corrupted: '{}' -> '{}'",
            term, cleaned
        );
    }
}

#[test]
fn test_challenge2_private_ip_and_path_scrubbing() {
    let input = "Download model from 10.0.1.20 and 192.168.0.50 to /home/alice/models/weights.bin on 127.0.0.1";
    let cleaned = sanitize_outbound_prompt(input);

    assert!(!cleaned.contains("10.0.1.20"));
    assert!(!cleaned.contains("192.168.0.50"));
    assert!(!cleaned.contains("127.0.0.1"));
    assert!(!cleaned.contains("/home/alice"));
    assert!(cleaned.contains("<LOCAL_HOST>"));
    assert!(cleaned.contains("<WORKSPACE_PATH>"));
}

// ============================================================================
// Challenge 3: In-Memory Dynamic Secrets and Multiline Keys
// ============================================================================

#[test]
fn test_challenge3_dynamic_vault_secret_registration() {
    let secret_val = "quantum_crypto_dynamic_pass_77182";
    register_test_secret("M1_CHALLENGE_SECRET", secret_val);

    let prompt = format!(
        "Execute query with authorization header Bearer {}",
        secret_val
    );
    let cleaned = sanitize_outbound_prompt(&prompt);

    assert!(
        !cleaned.contains(secret_val),
        "Dynamic secret must be scrubbed: '{}'",
        cleaned
    );
    assert!(
        cleaned.contains("[REDACTED_BY_ATLAS_VAULT]"),
        "Dynamic secret must be replaced with [REDACTED_BY_ATLAS_VAULT]"
    );
}

#[test]
fn test_challenge3_multiline_pem_and_signature_patterns() {
    let pem =
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0\nXYZ789\n-----END RSA PRIVATE KEY-----";
    let prompt = format!("Deploy server certificate and key:\n{}", pem);
    let cleaned = sanitize_outbound_prompt(&prompt);

    assert!(!cleaned.contains("MIIEowIBAAKCAQEA0"));
    assert!(cleaned.contains("[REDACTED_BY_ATLAS_VAULT]"));
}

// ============================================================================
// Challenge 4: System Prompt and Distill Task Verification
// ============================================================================

#[test]
fn test_challenge4_system_prompt_sanitization_in_distill_messages() {
    let sys = "System instructions for host spark-1a2b running under /home/alice/workspace for Drake Stapleton";
    let cleaned = sanitize_outbound_prompt(sys);

    assert!(!cleaned.contains("spark-1a2b"));
    assert!(!cleaned.contains("/home/alice"));
    assert!(!cleaned.contains("Drake Stapleton"));
    assert!(cleaned.contains("<LOCAL_HOST>"));
    assert!(cleaned.contains("<WORKSPACE_PATH>"));
    assert!(cleaned.contains("<OPERATOR>"));
}

#[test]
fn test_challenge4_distill_task_struct_scrubbing_simulation() {
    let task = DistillTask {
        id: "m1-challenger-task-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Run benchmark on 192.168.0.5 for drakestapleton".to_string(),
        system_prompt: Some("You are an assistant located on spark:18080".to_string()),
        teacher_model: "mock-teacher".to_string(),
        student_model: None,
        verification_strategy: VerificationStrategy::UnslopStrict,
        commit_to_cortex: false,
    };

    let scrubbed_user = sanitize_outbound_prompt(&task.prompt);
    let scrubbed_sys = task
        .system_prompt
        .as_ref()
        .map(|s| sanitize_outbound_prompt(s));

    assert!(!scrubbed_user.contains("192.168.0.5"));
    assert!(!scrubbed_user.contains("drakestapleton"));
    assert!(scrubbed_user.contains("<LOCAL_HOST>"));
    assert!(scrubbed_user.contains("<OPERATOR>"));

    let sys_str = scrubbed_sys.unwrap();
    assert!(!sys_str.contains("spark:18080"));
    assert!(sys_str.contains("<LOCAL_HOST>:18080"));
}
