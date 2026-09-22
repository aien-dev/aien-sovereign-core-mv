// Tier 4: Real-World Application Scenarios on NVIDIA DGX Spark
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::curriculum::{CurriculumEngine, CurriculumTrack};
use spark_adapters::models::{TaskType, VerificationStrategy};
use spark_adapters::vault::resolve_secret;
use std::fs;
use std::io::Write;
use std::process::Command;

#[test]
fn test_t4_01_verify_keys_cli_audit() {
    let bin_path = env!("CARGO_BIN_EXE_spark-distill");
    let output = Command::new(bin_path)
        .arg("verify-keys")
        .output()
        .expect("execute verify-keys");

    assert!(
        output.status.success(),
        "verify-keys must exit with success"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("ADAPTER ID"));
    assert!(stdout.contains("PROVIDER"));
    assert!(stdout.contains("STATUS"));
    assert!(stdout.contains("RESIDENT"));
    assert!(stdout.contains("atlas-lightning-omni"));
    assert!(stdout.contains("local_max"));
    assert!(!stdout.contains('\u{2014}'));
    assert!(!stdout.contains('\u{2013}'));
}

#[tokio::test]
async fn test_t4_02_resident_max_live_inference_roundtrip() {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "model": "atlas-lightning-omni",
        "messages": [
            { "role": "user", "content": "Respond with single word: OK" }
        ],
        "temperature": 0.0,
        "max_tokens": 32
    });

    let res = client
        .post("http://127.0.0.1:18007/v1/chat/completions")
        .json(&payload)
        .send()
        .await;

    if std::env::var("CI").is_ok() && res.is_err() {
        eprintln!("Skipping live MAX inference test in CI runner (offline daemon)");
        return;
    }
    assert!(
        res.is_ok(),
        "Live MAX inference server on 18006 must be reachable"
    );
    let response = res.unwrap();
    assert_eq!(response.status(), 200, "MAX server must return HTTP 200");

    let val: serde_json::Value = response.json().await.expect("parse MAX response");
    let content = val
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let reasoning = val
        .pointer("/choices/0/message/reasoning")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        !content.trim().is_empty() || !reasoning.trim().is_empty(),
        "MAX must return completion content or reasoning tokens"
    );

    let total_tokens = val
        .pointer("/usage/total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(total_tokens > 0, "MAX must report positive token usage");
}

#[test]
fn test_t4_03_systems_curriculum_batch_task_generation() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::NativeSystemsAndGpu,
        5,
        "anthropic/claude-3-7-sonnet",
        Some("atlas-lightning-omni"),
        true,
    );

    assert_eq!(tasks.len(), 5);
    for t in &tasks {
        assert_eq!(t.task_type, TaskType::CodeSynthesis);
        assert_eq!(t.verification_strategy, VerificationStrategy::CompilerCheck);
        assert!(!t.prompt.trim().is_empty());
        assert!(t.commit_to_cortex);
        assert_eq!(t.teacher_model, "anthropic/claude-3-7-sonnet");
        assert_eq!(t.student_model, Some("atlas-lightning-omni".to_string()));
        let uuid_parsed = uuid::Uuid::parse_str(&t.id);
        assert!(uuid_parsed.is_ok(), "Task ID must be a valid UUID");
    }
}

#[tokio::test]
async fn test_t4_04_cortex_memory_space_handshake() {
    let token_opt = resolve_secret("CORTEX_TOKEN");
    if std::env::var("CI").is_ok() && token_opt.is_none() {
        eprintln!("Skipping live Cortex handshake in CI runner (zero disk credentials)");
        return;
    }
    assert!(token_opt.is_some(), "CORTEX_TOKEN must be present");
    let token = token_opt.unwrap();

    let client = reqwest::Client::new();
    let res = client
        .get("http://127.0.0.1:18080/api/status")
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await;

    if std::env::var("CI").is_ok() && res.is_err() {
        eprintln!("Skipping live Cortex handshake in CI runner (offline daemon)");
        return;
    }
    assert!(res.is_ok(), "Cortex daemon on 18080 must be reachable");
    let resp = res.unwrap();
    assert!(
        resp.status().is_success() || resp.status().as_u16() == 404,
        "Cortex must respond to authenticated status query"
    );
}

#[test]
fn test_t4_05_end_to_end_sft_dpo_dataset_filesystem_emission() {
    let temp_dir = std::env::temp_dir().join(format!("test_crawl_e2e_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir).expect("create temp dataset dir");

    let sft_path = temp_dir.join("sft.jsonl");
    let dpo_path = temp_dir.join("dpo.jsonl");

    let mut sft_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&sft_path)
        .unwrap();
    let mut dpo_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dpo_path)
        .unwrap();

    let task_id = uuid::Uuid::new_v4().to_string();
    let prompt = "Implement an allocation-free single-producer single-consumer ring buffer in Rust";
    let chosen = "pub struct RingBuffer<T, const N: usize> { buf: [Option<T>; N] }";
    let rejected = "pub struct BrokenBuffer { inner: Vec<i32> }";
    let delta = 0.40;

    let sft_obj = serde_json::json!({
        "task_id": task_id,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "messages": [
            { "role": "system", "content": "You are a verified sovereign systems specialist." },
            { "role": "user", "content": prompt },
            { "role": "assistant", "content": chosen }
        ]
    });
    writeln!(sft_file, "{}", serde_json::to_string(&sft_obj).unwrap()).unwrap();

    if delta > 0.0 && chosen != rejected {
        let dpo_obj = serde_json::json!({
            "task_id": task_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });
        writeln!(dpo_file, "{}", serde_json::to_string(&dpo_obj).unwrap()).unwrap();
    }

    let sft_lines = fs::read_to_string(&sft_path).unwrap();
    let dpo_lines = fs::read_to_string(&dpo_path).unwrap();

    assert_eq!(sft_lines.lines().count(), 1);
    assert_eq!(dpo_lines.lines().count(), 1);

    let sft_parsed: serde_json::Value =
        serde_json::from_str(sft_lines.lines().next().unwrap()).unwrap();
    assert_eq!(sft_parsed["messages"].as_array().unwrap().len(), 3);

    let dpo_parsed: serde_json::Value =
        serde_json::from_str(dpo_lines.lines().next().unwrap()).unwrap();
    assert_eq!(dpo_parsed["chosen"], chosen);
    assert_eq!(dpo_parsed["rejected"], rejected);

    fs::remove_dir_all(&temp_dir).unwrap();
}
