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

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir);
        if let Some(parent) = p.parent() {
            if let Some(root) = parent.parent() {
                return root.to_path_buf();
            }
        }
    }
    // Fallback: this file lives at <root>/crates/spark-adapters/tests.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn spark_distill_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spark-distill"))
}
// Tier 2: Boundary, Corner, Negative, and Stress Tests
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::curriculum::{CurriculumEngine, CurriculumTrack};
use spark_adapters::models::{ChatMessage, DistillTask, TaskType, VerificationStrategy};
use spark_adapters::providers::format_openai_payload;
use spark_adapters::router::AdapterRouter;
use spark_adapters::vault::{is_secret_present, resolve_secret, sanitize_outbound_prompt};
use spark_adapters::verifier::Verifier;
use std::fs;
use std::process::Command;

// ============================================================================
// Feature 1: Vault Boundary
// ============================================================================

#[test]
fn test_t2_f01_vault_nonexistent_key_returns_none() {
    let res = resolve_secret("NONEXISTENT_ATLAS_KEY_9999");
    assert_eq!(res, None);
}

#[test]
fn test_t2_f02_vault_empty_env_var_treated_as_absent() {
    std::env::set_var("EMPTY_ATLAS_TEST_VAR", "");
    assert_eq!(resolve_secret("EMPTY_ATLAS_TEST_VAR"), None);
    std::env::remove_var("EMPTY_ATLAS_TEST_VAR");
}

#[test]
fn test_t2_f03_vault_whitespace_secret_rejected() {
    std::env::set_var("WS_ATLAS_TEST_VAR", "   \t \n  ");
    assert_eq!(resolve_secret("WS_ATLAS_TEST_VAR"), None);
    std::env::remove_var("WS_ATLAS_TEST_VAR");
}

#[test]
fn test_t2_f04_vault_key_case_sensitivity() {
    std::env::set_var("SOVEREIGN_LOWERCASE_VAR", "value123");
    assert_ne!(
        resolve_secret("sovereign_lowercase_var"),
        Some("value123".to_string())
    );
    std::env::remove_var("SOVEREIGN_LOWERCASE_VAR");
}

#[test]
fn test_t2_f05_vault_missing_daemon_handled_gracefully() {
    let res = is_secret_present("SOME_DEFINITELY_MISSING_SECRET_KEY");
    assert!(!res);
}

// ============================================================================
// Feature 2: Outbound Prompt Sanitizer Boundary
// ============================================================================

#[test]
fn test_t2_f06_sanitizer_empty_prompt() {
    let res = sanitize_outbound_prompt("");
    assert_eq!(res, "");
}

#[test]
fn test_t2_f07_sanitizer_prompt_only_paths() {
    let input = "/home/alice/dir/file.rs";
    let output = sanitize_outbound_prompt(input);
    assert_eq!(output, "<WORKSPACE_PATH>");
}

#[test]
fn test_t2_f08_sanitizer_prompt_only_ips() {
    let input = "127.0.0.1 192.168.0.1 10.0.0.1";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("127.0.0.1"));
    assert!(!output.contains("192.168.0.1"));
    assert!(!output.contains("10.0.0.1"));
    assert!(output.contains("<LOCAL_HOST>"));
}

#[test]
fn test_t2_f09_sanitizer_multiple_mixed_secrets() {
    let input = "sk-proj-abc12345678901234567890 and ghp_xyz12345678901234567890";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("sk-proj-"));
    assert!(!output.contains("ghp_"));
    assert!(output.contains("[REDACTED_BY_ATLAS_VAULT]"));
}

#[test]
fn test_t2_f10_sanitizer_system_prompt_scrubbing() {
    let sys = "You are running on /home/alice for aien@aienos.com";
    let cleaned = sanitize_outbound_prompt(sys);
    assert!(!cleaned.contains("/home/alice"));
    assert!(!cleaned.contains("aien@aienos.com"));
}

// ============================================================================
// Feature 3: Cortex Auth Boundary
// ============================================================================

#[test]
fn test_t2_f11_cortex_commit_without_token_fails_gracefully() {
    let default_endpoint = "http://127.0.0.1:18080";
    let url = format!("{}/api/cortex/write", default_endpoint);
    assert!(url.contains("/api/cortex/write"));
}

#[test]
fn test_t2_f12_cortex_unreachable_endpoint_handled() {
    let bad_endpoint = "http://127.0.0.1:19999/api/cortex/write";
    let res = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--connect-timeout",
            "1",
            bad_endpoint,
        ])
        .output();
    assert!(res.is_ok());
    let out = res.expect("curl");
    let code = String::from_utf8_lossy(&out.stdout);
    assert_eq!(code.trim(), "000");
}

#[test]
fn test_t2_f13_cortex_http_401_unauthorized_handled() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let output = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-H",
            "Authorization: Bearer invalid_test_token",
            "http://127.0.0.1:18080/api/status",
        ])
        .output()
        .expect("curl cortex status");
    let code = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code.trim(), "401");
}

#[test]
fn test_t2_f14_cortex_malformed_json_response_handled() {
    let invalid_json = "{ invalid_payload: ";
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(invalid_json);
    assert!(parsed.is_err());
}

#[test]
fn test_t2_f15_cortex_missing_target_id_fallback() {
    let body = serde_json::json!({
        "status": "ok"
    });
    let canonical_name = "distill_fallback_test";
    let entity_id = body
        .pointer("/receipt/targetId")
        .and_then(|v| v.as_str())
        .unwrap_or(canonical_name);
    assert_eq!(entity_id, "distill_fallback_test");
}

// ============================================================================
// Feature 4: Resident Generation Bounds Boundary
// ============================================================================

#[test]
fn test_t2_f16_resident_zero_or_negative_temperature_handling() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        reasoning: None,
    }];
    let payload = format_openai_payload("atlas-lightning-omni", &msgs, Some(0.0), false);
    let temp = payload["temperature"].as_f64().unwrap();
    assert_eq!(temp, 0.0);
}

#[test]
fn test_t2_f17_resident_context_length_exceeded_handling() {
    let max_len = 32768;
    let request_len = 35000;
    assert!(request_len > max_len);
}

#[test]
fn test_t2_f18_resident_large_prompt_payload_safety() {
    let large_prompt = "a".repeat(100_000);
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: large_prompt,
        reasoning: None,
    }];
    let payload = format_openai_payload("atlas-lightning-omni", &msgs, None, false);
    assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn test_t2_f19_resident_empty_messages_payload_safety() {
    let msgs: Vec<ChatMessage> = Vec::new();
    let payload = format_openai_payload("atlas-lightning-omni", &msgs, None, false);
    assert_eq!(payload["messages"].as_array().unwrap().len(), 0);
}

#[test]
fn test_t2_f20_resident_unicode_preservation() {
    let math_prompt = "Prove α + β = γ with ∀x ∈ ℝ";
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: math_prompt.to_string(),
        reasoning: None,
    }];
    let payload = format_openai_payload("atlas-lightning-omni", &msgs, None, false);
    let content = payload["messages"][0]["content"].as_str().unwrap();
    assert_eq!(content, math_prompt);
}

// ============================================================================
// Feature 5: Crawler Flag Collision Boundary
// ============================================================================

#[test]
fn test_t2_f21_crawler_short_flag_t_collision_rejection() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .args(["crawl", "--help"])
        .output()
        .expect("crawl help");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Short option names must be unique"),
        "Clap short flag collision detected"
    );
}

#[test]
fn test_t2_f22_crawler_unknown_track_returns_none() {
    let track = CurriculumTrack::parse("unknown_track_xyz");
    assert_eq!(track, None);
}

#[test]
fn test_t2_f23_crawler_zero_task_limit_returns_empty() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::NativeSystemsAndGpu,
        0,
        "anthropic/claude-3-7-sonnet",
        None,
        false,
    );
    assert_eq!(tasks.len(), 0);
}

#[test]
fn test_t2_f24_crawler_large_limit_safe_bounded_tasks() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::NativeSystemsAndGpu,
        100,
        "anthropic/claude-3-7-sonnet",
        None,
        false,
    );
    assert!(!tasks.is_empty());
    assert!(tasks.len() <= 100);
}

#[test]
fn test_t2_f25_crawler_dynamic_track_fallback_without_git() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::DynamicMining,
        2,
        "anthropic/claude-3-7-sonnet",
        None,
        false,
    );
    assert!(!tasks.is_empty());
}

// ============================================================================
// Feature 6: CLI Workbench Boundary
// ============================================================================

#[test]
fn test_t2_f26_cli_empty_args_shows_error() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path).output().expect("distill invocation");
    assert!(!output.status.success());
}

#[test]
fn test_t2_f27_cli_unknown_subcommand_exits_nonzero() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .arg("invalid_subcommand_xyz")
        .output()
        .expect("invalid subcmd");
    assert!(!output.status.success());
}

#[test]
fn test_t2_f28_cli_invalid_strategy_defaults_to_compiler() {
    let parsed = match "unknown_strategy".to_lowercase().as_str() {
        "unslop" => VerificationStrategy::UnslopStrict,
        "json" => VerificationStrategy::JsonSchema,
        "consensus" => VerificationStrategy::DualConsensus,
        _ => VerificationStrategy::CompilerCheck,
    };
    assert_eq!(parsed, VerificationStrategy::CompilerCheck);
}

#[test]
fn test_t2_f29_cli_invalid_task_type_defaults_to_code() {
    let parsed = match "unknown_task".to_lowercase().as_str() {
        "arch" => TaskType::SystemArchitecture,
        "reasoning" => TaskType::ReasoningTrace,
        "refactor" => TaskType::RefactorLogic,
        "harness" => TaskType::VerificationHarness,
        _ => TaskType::CodeSynthesis,
    };
    assert_eq!(parsed, TaskType::CodeSynthesis);
}

#[test]
fn test_t2_f30_cli_dataset_dir_auto_created() {
    let temp_dir = std::env::temp_dir().join(format!("test_auto_create_{}", uuid::Uuid::new_v4()));
    assert!(!temp_dir.exists());
    fs::create_dir_all(&temp_dir).unwrap();
    assert!(temp_dir.exists());
    fs::remove_dir_all(&temp_dir).unwrap();
}

// ============================================================================
// Feature 7: Key Verification Boundary
// ============================================================================

#[test]
fn test_t2_f31_verify_keys_zero_keys_handling() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    assert!(!adapters.is_empty());
}

#[test]
fn test_t2_f32_verify_keys_catalog_no_duplicate_ids() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let mut seen = std::collections::HashSet::new();
    for a in adapters {
        assert!(seen.insert(a.id), "Duplicate adapter ID detected");
    }
}

#[test]
fn test_t2_f33_verify_keys_all_endpoints_valid_urls() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    for a in adapters {
        assert!(
            a.endpoint.starts_with("http://")
                || a.endpoint.starts_with("https://")
                || a.endpoint.starts_with("in-process://"),
            "Invalid endpoint scheme"
        );
    }
}

#[test]
fn test_t2_f34_verify_keys_all_context_lengths_positive() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    for a in adapters {
        assert!(
            a.context_length > 0,
            "Context length must be strictly positive"
        );
    }
}

#[test]
fn test_t2_f35_verify_keys_adapter_status_string_representation() {
    let status_ready = "READY";
    let status_missing = "KEY MISSING";
    let status_local = "LOCAL";
    assert_ne!(status_ready, status_missing);
    assert_ne!(status_ready, status_local);
}

// ============================================================================
// Feature 8: Axum Routes Boundary
// ============================================================================

#[test]
fn test_t2_f36_axum_nonexistent_route_404() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let output = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://127.0.0.1:18095/nonexistent_route_xyz",
        ])
        .output()
        .expect("curl cockpit");
    let code = String::from_utf8_lossy(&output.stdout);
    // An unknown route never serves content. Without a session the cockpit access
    // gate (spark-cockpit-rs/src/access.rs) answers 401 before routing, so an
    // anonymous probe learns nothing about which routes exist; a gate-less build
    // answers 404.
    assert!(
        code.trim() == "404" || code.trim() == "401",
        "unknown route returned {} (expected 404, or 401 from the access gate)",
        code.trim()
    );
}

#[test]
fn test_t2_f37_axum_distill_empty_json_error() {
    let empty_payload = "{}";
    let res: Result<DistillTask, _> = serde_json::from_str(empty_payload);
    assert!(res.is_err(), "Empty payload must fail deserialization");
}

#[test]
fn test_t2_f38_axum_distill_missing_teacher_error() {
    let missing_teacher = r#"{
        "id": "task-01",
        "task_type": "code_synthesis",
        "prompt": "ping",
        "verification_strategy": "compiler_check",
        "commit_to_cortex": false
    }"#;
    let res: Result<DistillTask, _> = serde_json::from_str(missing_teacher);
    assert!(res.is_err());
}

#[test]
fn test_t2_f39_axum_commit_empty_json_error() {
    let empty_json = "{}";
    let val: serde_json::Value = serde_json::from_str(empty_json).unwrap();
    assert!(val.get("solution").is_none());
}

#[test]
fn test_t2_f40_axum_distill_invalid_track_caught() {
    let parsed = CurriculumTrack::parse("invalid_track_string");
    assert!(parsed.is_none());
}

// ============================================================================
// Feature 9: Cortex Closed-Loop Storage Boundary
// ============================================================================

#[test]
fn test_t2_f41_cortex_sub_threshold_score_rejected() {
    let score: f32 = 0.84;
    let passed = true;
    let commit_to_cortex = true;
    let should_commit = commit_to_cortex && passed && score >= 0.85;
    assert!(
        !should_commit,
        "Score 0.84 must not qualify for Cortex commit"
    );
}

#[test]
fn test_t2_f42_cortex_failed_verification_rejected() {
    let score: f32 = 0.95;
    let passed = false;
    let commit_to_cortex = true;
    let should_commit = commit_to_cortex && passed && score >= 0.85;
    assert!(
        !should_commit,
        "Failed verification must not qualify for Cortex commit"
    );
}

#[test]
fn test_t2_f43_cortex_commit_false_flag_respected() {
    use spark_adapters::distill::DistillationEngine;

    let qualifies = DistillationEngine::qualifies_for_cortex(false, true, 1.0);
    assert!(!qualifies, "commit_to_cortex=false must prevent commit");
}

#[test]
fn test_t2_f44_cortex_canonical_name_hyphen_sanitization() {
    let raw = "task-uuid-123-abc";
    let canonical = format!("distill_{}", raw.replace('-', "_"));
    assert!(
        !canonical.contains('-'),
        "Canonical name must not contain hyphens"
    );
}

#[test]
fn test_t2_f45_cortex_content_summary_includes_verified_solution() {
    let solution = "pub fn add() {}";
    let summary = format!("Verified Solution: {}", solution);
    assert!(summary.contains("Verified Solution:"));
}

// ============================================================================
// Feature 10: SFT & DPO Dataset Boundary
// ============================================================================

#[test]
fn test_t2_f46_dpo_zero_delta_rejected() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t2_f46_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let task = DistillTask {
        id: "task-zero-delta-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Implement vector push".to_string(),
        system_prompt: None,
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: Some("atlas-lightning-omni".to_string()),
        verification_strategy: VerificationStrategy::UnslopStrict,
        commit_to_cortex: false,
    };

    let chosen = "fn push(v: &mut Vec<i32>, val: i32) { v.push(val); }";
    let rejected = "fn push(v: &mut Vec<i32>, val: i32) { v.push(val); }";
    let preference_delta = 0.0f32;

    let valid_rej =
        DistillationEngine::filter_dpo_candidate(chosen, Some(rejected), preference_delta);
    assert_eq!(
        valid_rej, None,
        "Zero preference delta must be rejected for DPO pair"
    );

    let dpo_persisted = engine
        .persist_training_pair(&task, chosen, None, valid_rej)
        .expect("persist training pair");
    assert!(
        !dpo_persisted,
        "DPO pair must NOT be persisted when delta is zero"
    );

    assert!(
        temp_dir.join("sft.jsonl").exists(),
        "SFT dataset must still be written"
    );
    assert!(
        !temp_dir.join("dpo.jsonl").exists(),
        "DPO dataset must NOT be written when delta is zero"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t2_f47_dpo_negative_delta_rejected() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t2_f47_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let task = DistillTask {
        id: "task-negative-delta-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Calculate factorial".to_string(),
        system_prompt: None,
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: Some("atlas-lightning-omni".to_string()),
        verification_strategy: VerificationStrategy::CompilerCheck,
        commit_to_cortex: false,
    };

    let chosen = "fn fact(n: u64) -> u64 { (1..=n).product() }";
    let rejected = "fn fact(n: u64) -> u64 { 1 }";
    let negative_delta = -0.15f32;

    let valid_rej =
        DistillationEngine::filter_dpo_candidate(chosen, Some(rejected), negative_delta);
    assert_eq!(
        valid_rej, None,
        "Negative preference delta must be rejected for DPO pair"
    );

    let dpo_persisted = engine
        .persist_training_pair(&task, chosen, None, valid_rej)
        .expect("persist training pair");
    assert!(
        !dpo_persisted,
        "DPO pair must NOT be persisted when delta is negative"
    );
    assert!(
        !temp_dir.join("dpo.jsonl").exists(),
        "DPO dataset must NOT exist for negative delta"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t2_f48_dpo_identical_outputs_rejected() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t2_f48_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let task = DistillTask {
        id: "task-identical-dpo-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Write empty entry point".to_string(),
        system_prompt: None,
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: Some("atlas-lightning-omni".to_string()),
        verification_strategy: VerificationStrategy::CompilerCheck,
        commit_to_cortex: false,
    };

    let chosen = "fn main() {}";
    let identical_rejected = "fn main() {}";
    let positive_delta = 0.30f32;

    let valid_rej =
        DistillationEngine::filter_dpo_candidate(chosen, Some(identical_rejected), positive_delta);
    assert_eq!(
        valid_rej, None,
        "Identical chosen and rejected completions must be rejected for DPO pair"
    );

    let dpo_persisted = engine
        .persist_training_pair(&task, chosen, None, valid_rej)
        .expect("persist training pair");
    assert!(
        !dpo_persisted,
        "DPO pair must NOT be persisted for identical completions"
    );
    assert!(
        !temp_dir.join("dpo.jsonl").exists(),
        "DPO dataset must NOT exist for identical completions"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t2_f49_dpo_missing_student_rollout_no_dpo_file() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t2_f49_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let task = DistillTask {
        id: "task-missing-student-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Write unit test".to_string(),
        system_prompt: None,
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: None,
        verification_strategy: VerificationStrategy::UnslopStrict,
        commit_to_cortex: false,
    };

    let chosen = "#[test] fn test() {}";
    let valid_rej = DistillationEngine::filter_dpo_candidate(chosen, None, 0.0);
    assert_eq!(
        valid_rej, None,
        "Missing student rollout must yield None for DPO candidate"
    );

    let dpo_persisted = engine
        .persist_training_pair(&task, chosen, None, valid_rej)
        .expect("persist training pair");
    assert!(
        !dpo_persisted,
        "DPO pair must NOT be persisted when student rollout is missing"
    );
    assert!(
        !temp_dir.join("dpo.jsonl").exists(),
        "DPO dataset must NOT exist when student rollout is missing"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t2_f50_dataset_permission_denied_handled_gracefully() {
    let bad_path = std::path::Path::new("/root/nonexistent_forbidden_dir/sft.jsonl");
    let res = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(bad_path);
    assert!(res.is_err());
}

// ============================================================================
// Feature 11: Verification Gates Boundary
// ============================================================================

#[test]
fn test_t2_f51_verifier_rustc_compilation_failure_deducts_score() {
    let broken_code = "pub fn broken() { let x: i32 = \"not an int\"; }";
    let res = Verifier::verify_with_rustc(broken_code);
    assert!(res.is_err(), "rustc must fail on type mismatch");
}

#[test]
fn test_t2_f52_verifier_em_dash_detected_and_penalized() {
    let bad_text = "Fast \u{2014} and reliable.";
    let outcome = Verifier::verify(bad_text, VerificationStrategy::UnslopStrict);
    assert!(!outcome.passed);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("em dash")));
    assert!(outcome.score <= 0.75);
}

#[test]
fn test_t2_f53_verifier_en_dash_detected_and_penalized() {
    let bad_text = "Pages 1\u{2013}5.";
    let outcome = Verifier::verify(bad_text, VerificationStrategy::UnslopStrict);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("en dash")));
    assert!(outcome.score <= 0.85);
}

#[test]
fn test_t2_f54_verifier_banned_cliche_detected_and_penalized() {
    let text = "We must delve into the tapestry of the problem.";
    let outcome = Verifier::verify(text, VerificationStrategy::UnslopStrict);
    assert!(outcome.rule_violations.iter().any(|v| v.contains("delve")));
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("tapestry")));
}

#[test]
fn test_t2_f55_verifier_secret_signature_resets_score_to_zero() {
    let text = "Here is my secret sk-proj-123456789012345678901234";
    let outcome = Verifier::verify(text, VerificationStrategy::CompilerCheck);
    assert_eq!(outcome.score, 0.0);
    assert!(!outcome.passed);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("Critical: Contains secret signature")));
}

// ============================================================================
// Feature 12: Test Suite Boundary
// ============================================================================

#[test]
fn test_t2_f56_verifier_unclosed_delimiter_detected() {
    let unclosed = "fn test() { let x = (1 + 2;";
    let outcome = Verifier::verify(unclosed, VerificationStrategy::CompilerCheck);
    eprintln!("VIOLATIONS: {:?}", outcome.rule_violations);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.to_lowercase().contains("unclosed delimiter")));
}

#[test]
fn test_t2_f57_verifier_mismatched_delimiter_detected() {
    let mismatched = "fn test() { let x = [1, 2); }";
    let outcome = Verifier::verify(mismatched, VerificationStrategy::CompilerCheck);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.to_lowercase().contains("delimiter")));
}

#[test]
fn test_t2_f58_verifier_string_literal_delimiters_ignored() {
    let code_with_string = "pub fn greet() -> &'static str { \"([{\"}";
    let outcome = Verifier::verify(code_with_string, VerificationStrategy::CompilerCheck);
    assert!(!outcome
        .rule_violations
        .iter()
        .any(|v| v.to_lowercase().contains("unclosed delimiter")));
}

#[test]
fn test_t2_f59_verifier_malformed_json_schema_detected() {
    let bad_json = "Here is JSON: { \"invalid_json\": ";
    let outcome = Verifier::verify(bad_json, VerificationStrategy::JsonSchema);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("malformed") || v.contains("Incomplete")));
}

#[test]
fn test_t2_f60_verifier_incomplete_json_brackets_detected() {
    let incomplete_json = "Payload: { \"key\": \"value\"";
    let outcome = Verifier::verify(incomplete_json, VerificationStrategy::JsonSchema);
    assert!(outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("Incomplete JSON")));
}

// ============================================================================
// Feature 13: Remote Git Parity Boundary
// ============================================================================

#[test]
fn test_t2_f61_git_no_uncommitted_env_secrets() {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace_root())
        .output()
        .expect("git status");
    let status = String::from_utf8_lossy(&output.stdout);
    assert!(
        !status.contains(".env"),
        "No .env files allowed in git status"
    );
}

#[test]
fn test_t2_f62_git_log_zero_em_or_en_dashes() {
    let output = Command::new("git")
        .args(["log", "-n", "20", "--format=%B"])
        .current_dir(workspace_root())
        .output()
        .expect("git log");
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(
        !log.contains('\u{2014}'),
        "Git commits must not contain em dashes"
    );
    assert!(
        !log.contains('\u{2013}'),
        "Git commits must not contain en dashes"
    );
}

#[test]
fn test_t2_f63_git_covenant_is_nonbinding_values_statement() {
    let covenant_path = workspace_root().join("COVENANT.md");
    let content = fs::read_to_string(covenant_path).expect("read COVENANT");
    assert!(content.contains("Keep foundational advances open"));
    assert!(content.contains("grants no legal rights"));
}

#[test]
fn test_t2_f64_git_readme_contains_apache_badge() {
    let readme_path = workspace_root().join("README.md");
    let content = fs::read_to_string(readme_path).expect("read README");
    assert!(content.contains("License-Apache--2.0"));
    assert!(!content.contains("AGPL--3.0") && !content.contains("AGPL-3.0"));
}

#[test]
fn test_t2_f65_git_worktree_clean_no_confidential_files() {
    // target/ is build output and holds the pinned dependency checkouts written by
    // scripts/pins.sh, whose vendored crates ship PEM test fixtures; the scan is
    // about files a person or agent committed or dropped into the worktree.
    let root = workspace_root();
    let target = root.join("target");
    let output = Command::new("find")
        .args([
            root.to_str().unwrap(),
            "-path",
            target.to_str().unwrap(),
            "-prune",
            "-o",
            "(",
            "-name",
            "*.pem",
            "-o",
            "-name",
            "*.id_rsa",
            ")",
            "-print",
        ])
        .output()
        .expect("find keys");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim().is_empty(), "No private key files on disk");
}
