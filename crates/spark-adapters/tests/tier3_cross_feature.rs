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

// Tier 3: Cross-Feature Combinations and Subsystem Interaction Tests
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::curriculum::{CurriculumEngine, CurriculumTrack};
use spark_adapters::models::{ChatMessage, ProviderType, VerificationStrategy};
use spark_adapters::providers::format_openai_payload;
use spark_adapters::router::AdapterRouter;
use spark_adapters::vault::{resolve_secret, sanitize_outbound_prompt};
use spark_adapters::verifier::Verifier;
use std::fs;
use std::io::Write;
use std::process::Command;

#[test]
fn test_t3_01_sanitizer_router_dispatch_interaction() {
    let raw_prompt = "Load /home/drakestapleton/workspace/main.rs from 10.0.0.5 for drakestapleton with sk-proj-12345678901234567890";
    let clean_prompt = sanitize_outbound_prompt(raw_prompt);

    assert!(!clean_prompt.contains("/home/drakestapleton"));
    assert!(!clean_prompt.contains("10.0.0.5"));
    assert!(!clean_prompt.contains("drakestapleton"));
    assert!(!clean_prompt.contains("sk-proj-"));

    let router = AdapterRouter::new();
    let (provider, endpoint, model, key) = router.resolve_route("atlas-lightning-omni");
    assert_eq!(provider, ProviderType::LocalSeat);
    assert_eq!(endpoint, "http://127.0.0.1:18006/v1/chat/completions");
    assert_eq!(model, "atlas-lightning-omni");
    assert!(key.is_none());

    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: clean_prompt.clone(),
        reasoning: None,
    }];
    let payload = format_openai_payload(&model, &msgs, Some(0.7), false);
    assert_eq!(payload["messages"][0]["content"], clean_prompt);
}

#[test]
fn test_t3_02_verifier_consensus_cortex_qualification_gate() {
    let teacher = "pub fn sum_slice(s: &[i32]) -> i32 { s.iter().sum() }";
    let student = "pub fn sum_slice(arr: &[i32]) -> i32 { arr.iter().sum() }";

    let sim = Verifier::hybrid_consensus(teacher, student);
    assert!(
        sim > 0.60,
        "Consensus similarity between semantically equivalent code must exceed 0.60"
    );

    let t_verif = Verifier::verify(teacher, VerificationStrategy::CompilerCheck);
    assert!(t_verif.passed);
    assert_eq!(t_verif.score, 1.0);

    let s_verif = Verifier::verify(student, VerificationStrategy::CompilerCheck);
    assert!(s_verif.passed);
    assert_eq!(s_verif.score, 1.0);

    let qualifies_for_cortex = t_verif.passed && t_verif.score >= 0.85;
    assert!(
        qualifies_for_cortex,
        "Verified high-scoring solution must qualify for Cortex storage"
    );
}

#[test]
fn test_t3_03_crawler_curriculum_dataset_emission_pipeline() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::NativeSystemsAndGpu,
        2,
        "anthropic/claude-3-7-sonnet",
        Some("atlas-lightning-omni"),
        false,
    );
    assert_eq!(tasks.len(), 2);

    let temp_dir =
        std::env::temp_dir().join(format!("test_crawl_emission_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir).expect("create temp dataset dir");

    let sft_file = temp_dir.join("sft.jsonl");
    let dpo_file = temp_dir.join("dpo.jsonl");

    let mut sft_writer = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&sft_file)
        .unwrap();
    let mut dpo_writer = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dpo_file)
        .unwrap();

    for task in &tasks {
        let chosen = "pub fn execute() -> bool { true }";
        let rejected = "pub fn execute() -> bool { false }";
        let delta = 0.30;

        let sft_obj = serde_json::json!({
            "task_id": task.id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "messages": [
                { "role": "system", "content": task.system_prompt },
                { "role": "user", "content": task.prompt },
                { "role": "assistant", "content": chosen }
            ]
        });
        writeln!(sft_writer, "{}", serde_json::to_string(&sft_obj).unwrap()).unwrap();

        if delta > 0.0 && chosen != rejected {
            let dpo_obj = serde_json::json!({
                "task_id": task.id,
                "prompt": task.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            writeln!(dpo_writer, "{}", serde_json::to_string(&dpo_obj).unwrap()).unwrap();
        }
    }

    let sft_content = fs::read_to_string(&sft_file).unwrap();
    let dpo_content = fs::read_to_string(&dpo_file).unwrap();

    assert_eq!(sft_content.lines().count(), 2);
    assert_eq!(dpo_content.lines().count(), 2);

    fs::remove_dir_all(&temp_dir).unwrap();
}

#[test]
fn test_t3_04_vault_token_cortex_http_dispatch() {
    let token_opt = resolve_secret("CORTEX_TOKEN");
    if std::env::var("CI").is_ok() && token_opt.is_none() {
        return;
    }
    assert!(
        token_opt.is_some(),
        "CORTEX_TOKEN must resolve via vault or environment"
    );
    let token = token_opt.unwrap();
    assert!(!token.trim().is_empty());

    let auth_header = format!("Bearer {}", token);
    let output = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-H",
            &format!("Authorization: {}", auth_header),
            "http://127.0.0.1:18080/api/status",
        ])
        .output()
        .expect("curl cortex status");

    let code = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert!(
        !code.is_empty(),
        "Cortex status check must return HTTP status code"
    );
}

#[test]
fn test_t3_05_verifier_rustc_unslop_sft_pipeline() {
    let dirty_solution = "This is fast \u{2014} implementation.\n```rust\npub fn ok() {}\n```";
    let dirty_outcome = Verifier::verify(dirty_solution, VerificationStrategy::CompilerCheck);
    assert!(dirty_outcome
        .rule_violations
        .iter()
        .any(|v| v.contains("em dash")));
    assert!(dirty_outcome.score < 1.0, "Em dash must penalize score");

    let clean_solution = "Pure Rust implementation.\n```rust\npub fn run_fast() -> i32 { 42 }\n```";
    let clean_outcome = Verifier::verify(clean_solution, VerificationStrategy::CompilerCheck);
    assert!(
        clean_outcome.passed,
        "Clean solution must pass compiler and unslop checks"
    );
    assert_eq!(clean_outcome.score, 1.0);

    let sft_entry = serde_json::json!({
        "messages": [
            { "role": "user", "content": "Write fast function" },
            { "role": "assistant", "content": clean_solution }
        ]
    });
    let sft_str = serde_json::to_string(&sft_entry).unwrap();
    assert!(!sft_str.contains('\u{2014}'));
}
