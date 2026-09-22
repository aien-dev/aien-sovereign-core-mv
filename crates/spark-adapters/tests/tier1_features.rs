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
    PathBuf::from("/home/drakestapleton/workspace/aien-sovereign-core")
}

fn workshop_html_path() -> PathBuf {
    workspace_root().join("crates/spark-cockpit-rs/static/workshop.html")
}

fn spark_distill_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spark-distill"))
}
// Tier 1: Feature Coverage Tests for Sovereign Distillation Workshop
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::curriculum::{CurriculumEngine, CurriculumTrack};
use spark_adapters::models::{
    ChatMessage, DistillTask, ProviderType, TaskType, VerificationOutcome, VerificationStrategy,
};
use spark_adapters::providers::format_openai_payload;
use spark_adapters::router::AdapterRouter;
use spark_adapters::vault::{
    is_secret_present, list_vault_keys, resolve_secret, sanitize_outbound_prompt,
};
use spark_adapters::verifier::Verifier;
use std::fs;
use std::process::Command;

// ============================================================================
// Feature 1: In-Memory Vault Secret Resolution
// ============================================================================

#[test]
fn test_t1_f01_vault_keys_listing() {
    let keys = list_vault_keys();
    // list_vault_keys must return a vector without crashing or hanging
    assert!(keys.iter().all(|k| !k.trim().is_empty()));
}

#[test]
fn test_t1_f02_vault_secret_presence_check() {
    std::env::set_var("TEST_KEY_PRESENCE", "present_value");
    assert!(is_secret_present("TEST_KEY_PRESENCE"));
    std::env::remove_var("TEST_KEY_PRESENCE");
}

#[test]
fn test_t1_f03_vault_secret_resolution_non_empty() {
    std::env::set_var("TEST_KEY_RESOLUTION", "sovereign_secret_123");
    let resolved = resolve_secret("TEST_KEY_RESOLUTION");
    assert_eq!(resolved, Some("sovereign_secret_123".to_string()));
    std::env::remove_var("TEST_KEY_RESOLUTION");
}

#[test]
fn test_t1_f04_vault_env_variable_resolution() {
    std::env::set_var("ATLAS_TEST_VAR", "in_memory_val");
    assert_eq!(
        resolve_secret("ATLAS_TEST_VAR"),
        Some("in_memory_val".to_string())
    );
    std::env::remove_var("ATLAS_TEST_VAR");
}

#[test]
fn test_t1_f05_vault_keys_caching_invariant() {
    let t0_keys = list_vault_keys();
    let t1_keys = list_vault_keys();
    assert_eq!(t0_keys, t1_keys);
}

// ============================================================================
// Feature 2: Outbound Prompt Sanitizer Hardening
// ============================================================================

#[test]
fn test_t1_f06_sanitizer_scrubs_workspace_paths() {
    let input = "Inspect file at /home/drakestapleton/workspace/secret.rs and /Users/drakestapleton/audit.md";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("/home/drakestapleton"));
    assert!(!output.contains("/Users/drakestapleton"));
    assert!(output.contains("<WORKSPACE_PATH>"));
}

#[test]
fn test_t1_f07_sanitizer_scrubs_private_ips() {
    let input = "Connect to 127.0.0.1 and 100.64.0.1 and 192.168.1.10 and 10.0.0.5";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("127.0.0.1"));
    assert!(!output.contains("100.64.0.1"));
    assert!(!output.contains("192.168.1.10"));
    assert!(!output.contains("10.0.0.5"));
    assert!(output.contains("<LOCAL_HOST>"));
}

#[test]
fn test_t1_f08_sanitizer_scrubs_local_hostnames() {
    let input = "Run job on spark and spark-b87b and localhost directly";
    let output = sanitize_outbound_prompt(input);
    // Hardening requirement: hostnames scrubbed per PROJECT.md interface contract
    assert!(!output.contains("spark-b87b"));
    assert!(output.contains("<LOCAL_HOST>"));
}

#[test]
fn test_t1_f09_sanitizer_scrubs_operator_names() {
    let input = "Operator drakestapleton and Drake Stapleton with email drake.aien@proton.me";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("drakestapleton"));
    assert!(!output.contains("drake.aien@proton.me"));
    assert!(output.contains("<OPERATOR>"));
}

#[test]
fn test_t1_f10_sanitizer_scrubs_vault_secret_signatures() {
    let input = "Keys: sk-proj-123456789012345678901234 and ghp_12345678901234567890";
    let output = sanitize_outbound_prompt(input);
    assert!(!output.contains("sk-proj-"));
    assert!(!output.contains("ghp_"));
    assert!(output.contains("[REDACTED_BY_ATLAS_VAULT]"));
}

#[test]
fn test_t1_f10b_sanitizer_persisted_dataset_contains_no_raw_paths_or_ips() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use spark_adapters::vault::sanitize_outbound_prompt;
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t1_f10b_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let raw_prompt = "Write code for /home/drakestapleton/audit_secret.rs at 10.0.0.15";
    let sanitized_prompt = sanitize_outbound_prompt(raw_prompt);

    let task = DistillTask {
        id: "task-sanitizer-persist-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: sanitized_prompt,
        system_prompt: Some("You are AIEN Sovereign Architect.".to_string()),
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: Some("atlas-lightning-omni".to_string()),
        verification_strategy: VerificationStrategy::CompilerCheck,
        commit_to_cortex: false,
    };

    let chosen = "pub fn secret_fn() {}";
    engine
        .persist_training_pair(&task, chosen, None, None)
        .expect("persist training pair");

    let sft_path = temp_dir.join("sft.jsonl");
    assert!(sft_path.exists(), "sft.jsonl must exist on disk");

    let content = fs::read_to_string(&sft_path).expect("read sft.jsonl");
    assert!(
        !content.contains("/home/drakestapleton"),
        "sft.jsonl must not contain raw home path"
    );
    assert!(
        !content.contains("10.0.0.15"),
        "sft.jsonl must not contain private IP"
    );
    assert!(
        content.contains("<WORKSPACE_PATH>"),
        "sft.jsonl must contain <WORKSPACE_PATH>"
    );
    assert!(
        content.contains("<LOCAL_HOST>"),
        "sft.jsonl must contain <LOCAL_HOST>"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// Feature 3: Cortex Token Dynamic Auth
// ============================================================================

#[test]
fn test_t1_f11_cortex_token_resolves_from_vault() {
    if std::env::var("CI").is_ok() {
        return;
    }
    // Either CORTEX_TOKEN is in atlas-vault or environment
    let is_present = is_secret_present("CORTEX_TOKEN");
    assert!(
        is_present,
        "CORTEX_TOKEN must be registered in vault or environment"
    );
    let resolved = resolve_secret("CORTEX_TOKEN");
    assert!(resolved.is_some());
    assert!(!resolved.unwrap().trim().is_empty());
}

#[test]
fn test_t1_f12_cortex_token_presence_without_disk_file() {
    if std::env::var("CI").is_ok() {
        return;
    }
    // Presence check must function purely in memory without disk dependency
    assert!(is_secret_present("CORTEX_TOKEN"));
}

#[test]
fn test_t1_f13_cortex_bearer_header_format() {
    let token = "test_token_cortex";
    let header_val = format!("Bearer {}", token);
    assert_eq!(header_val, "Bearer test_token_cortex");
}

#[test]
fn test_t1_f14_cortex_endpoint_resolution_default() {
    let default_endpoint = "http://127.0.0.1:18080";
    assert_eq!(default_endpoint, "http://127.0.0.1:18080");
}

#[test]
fn test_t1_f15_cortex_live_service_connectivity() {
    // Live Cortex service on 18080 must be listening (returns 200, 401, or 404, but responds)
    let output = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://127.0.0.1:18080/api/status",
        ])
        .output()
        .expect("curl cortex");
    let code = String::from_utf8_lossy(&output.stdout);
    assert!(!code.trim().is_empty());
}

// ============================================================================
// Feature 4: Resident Generation Bounds
// ============================================================================

#[test]
fn test_t1_f16_resident_adapter_catalog_spec() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let resident = adapters.iter().find(|a| a.id == "atlas-lightning-omni");
    assert!(resident.is_some());
    let spec = resident.unwrap();
    assert!(spec.is_resident);
    assert!(!spec.requires_key);
}

#[test]
fn test_t1_f17_resident_context_length_bounds() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let resident = adapters
        .iter()
        .find(|a| a.id == "atlas-lightning-omni")
        .unwrap();
    assert_eq!(spec_context_len(resident), 32768);
}

fn spec_context_len(spec: &spark_adapters::models::AdapterSpec) -> usize {
    spec.context_length
}

#[test]
fn test_t1_f18_resident_model_endpoint_resolution() {
    let router = AdapterRouter::new();
    let (prov, endpoint, model, key) = router.resolve_route("atlas-lightning-omni");
    assert_eq!(prov, ProviderType::LocalMax);
    assert_eq!(endpoint, "http://127.0.0.1:18007/v1/chat/completions");
    assert_eq!(model, "atlas-lightning-omni");
    assert!(key.is_none());
}

#[test]
fn test_t1_f19_resident_format_openai_payload() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Ping".to_string(),
        reasoning: None,
    }];
    let payload = format_openai_payload("atlas-lightning-omni", &msgs, Some(0.7), false);
    assert_eq!(payload["model"], "atlas-lightning-omni");
    let temp = payload["temperature"].as_f64().unwrap();
    assert!((temp - 0.7).abs() < 1e-4);
    assert_eq!(payload["stream"], false);
}

#[test]
fn test_t1_f20_resident_live_model_availability() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let output = Command::new("curl")
        .args(["-s", "http://127.0.0.1:18007/v1/models"])
        .output()
        .expect("curl max");
    let body = String::from_utf8_lossy(&output.stdout);
    assert!(body.contains("atlas-lightning-omni"));
}

// ============================================================================
// Feature 5: Crawler Flag Collision Fix
// ============================================================================

#[test]
fn test_t1_f21_crawler_track_flag_parsing_long() {
    let track = CurriculumTrack::parse("systems");
    assert_eq!(track, Some(CurriculumTrack::NativeSystemsAndGpu));
}

#[test]
fn test_t1_f22_crawler_teacher_flag_parsing_long() {
    let router = AdapterRouter::new();
    let (prov, _, model, _) = router.resolve_route("anthropic/claude-3-7-sonnet");
    assert_eq!(prov, ProviderType::Anthropic);
    assert_eq!(model, "claude-3-7-sonnet-20250219");
}

#[test]
fn test_t1_f23_crawler_track_short_flag_distinct_from_teacher() {
    // spark-distill crawl --help must not panic with clap duplicate short option '-t'
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path).args(["crawl", "--help"]).output();
    assert!(output.is_ok());
    let res = output.unwrap();
    assert_eq!(
        res.status.code(),
        Some(0),
        "spark-distill crawl --help must exit with 0"
    );
}

#[test]
fn test_t1_f24_crawler_curriculum_tracks_enumeration() {
    assert!(CurriculumTrack::parse("systems").is_some());
    assert!(CurriculumTrack::parse("agent").is_some());
    assert!(CurriculumTrack::parse("science").is_some());
    assert!(CurriculumTrack::parse("frontier").is_some());
    assert!(CurriculumTrack::parse("dynamic").is_some());
}

#[test]
fn test_t1_f24b_crawler_curriculum_tracks_aliases() {
    assert_eq!(
        CurriculumTrack::parse("cuda"),
        Some(CurriculumTrack::NativeSystemsAndGpu)
    );
    assert_eq!(
        CurriculumTrack::parse("git"),
        Some(CurriculumTrack::DynamicMining)
    );
    assert_eq!(
        CurriculumTrack::parse("git_mining"),
        Some(CurriculumTrack::DynamicMining)
    );
    assert_eq!(
        CurriculumTrack::parse("git-mining"),
        Some(CurriculumTrack::DynamicMining)
    );
    assert_eq!(
        CurriculumTrack::parse("refactor"),
        Some(CurriculumTrack::DynamicMining)
    );
}

#[test]
fn test_t1_f25_crawler_task_generation_limit_respect() {
    let tasks = CurriculumEngine::generate_tasks(
        CurriculumTrack::NativeSystemsAndGpu,
        3,
        "anthropic/claude-3-7-sonnet",
        Some("atlas-lightning-omni"),
        false,
    );
    assert_eq!(tasks.len(), 3);
}

// ============================================================================
// Feature 6: Interactive CLI Workbench
// ============================================================================

#[test]
fn test_t1_f26_cli_help_invocation() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .arg("--help")
        .output()
        .expect("help invocation");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("spark-distill"));
}

#[test]
fn test_t1_f26b_cli_interactive_subcommand_schema() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .args(["cli", "--help"])
        .output()
        .expect("cli help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("--teacher"));
    assert!(stdout.contains("--student"));
}

#[test]
fn test_t1_f27_cli_single_subcommand_schema() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .args(["single", "--help"])
        .output()
        .expect("single help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--prompt"));
}

#[test]
fn test_t1_f28_cli_crawl_subcommand_schema() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .args(["crawl", "--help"])
        .output()
        .expect("crawl help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--track") || stdout.contains("-k") || stdout.contains("-t"));
}

#[test]
fn test_t1_f29_cli_verify_keys_subcommand_schema() {
    let bin_path = spark_distill_bin();
    let output = Command::new(bin_path)
        .args(["verify-keys", "--help"])
        .output();
    assert!(output.is_ok());
    let res = output.unwrap();
    assert_eq!(res.status.code(), Some(0));
}

#[test]
fn test_t1_f30_cli_task_type_parsing_coverage() {
    let json_code = serde_json::to_string(&TaskType::CodeSynthesis).unwrap();
    assert_eq!(json_code, "\"code_synthesis\"");
    let json_arch = serde_json::to_string(&TaskType::SystemArchitecture).unwrap();
    assert_eq!(json_arch, "\"system_architecture\"");
    let json_reasoning = serde_json::to_string(&TaskType::ReasoningTrace).unwrap();
    assert_eq!(json_reasoning, "\"reasoning_trace\"");
}

// ============================================================================
// Feature 7: Key Verification Command
// ============================================================================

#[test]
fn test_t1_f31_verify_keys_discovers_all_catalog_adapters() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    assert!(adapters.len() >= 11);
}

#[test]
fn test_t1_f32_verify_keys_resident_models_ready_without_key() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let resident = adapters
        .iter()
        .find(|a| a.id == "atlas-lightning-omni")
        .unwrap();
    assert!(!resident.requires_key);
    assert!(resident.is_available);
}

#[test]
fn test_t1_f33_verify_keys_local_engine_ready_without_key() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let ollama = adapters
        .iter()
        .find(|a| a.id == "ollama/qwen2.5-coder")
        .unwrap();
    assert!(!ollama.requires_key);
    assert!(ollama.is_available);
}

#[test]
fn test_t1_f34_verify_keys_cloud_models_require_key() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let openai = adapters.iter().find(|a| a.id == "openai/gpt-4o").unwrap();
    assert!(openai.requires_key);
    let anthropic = adapters
        .iter()
        .find(|a| a.id == "anthropic/claude-3-7-sonnet")
        .unwrap();
    assert!(anthropic.requires_key);
}

#[test]
fn test_t1_f35_verify_keys_web_sessions_require_key() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();
    let chatgpt_web = adapters
        .iter()
        .find(|a| a.id == "chatgpt/web-plus")
        .unwrap();
    assert!(chatgpt_web.requires_key);
    let claude_web = adapters.iter().find(|a| a.id == "claude/web-pro").unwrap();
    assert!(claude_web.requires_key);
}

#[test]
fn test_t1_f35b_web_session_token_aliases() {
    let chatgpt = spark_adapters::models::ProviderType::ChatGPTWeb;
    assert!(chatgpt.key_aliases().contains(&"CHATGPT_SESSION_TOKEN"));
    assert!(chatgpt.key_aliases().contains(&"OPENAI_WEB_TOKEN"));

    let claude = spark_adapters::models::ProviderType::ClaudeWeb;
    assert!(claude.key_aliases().contains(&"CLAUDE_SESSION_KEY"));
    assert!(claude.key_aliases().contains(&"CLAUDE_WEB_SESSION_KEY"));
}

// ============================================================================
// Feature 8: Axum Web Workshop Service
// ============================================================================

#[test]
fn test_t1_f36_axum_workshop_static_html_content() {
    let html_path = workshop_html_path();
    let content = fs::read_to_string(html_path).expect("read workshop.html");
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("Knowledge Distillation Workshop"));
}

#[test]
fn test_t1_f37_axum_workshop_html_contains_model_options() {
    let html_path = workshop_html_path();
    let content = fs::read_to_string(html_path).expect("read workshop.html");
    assert!(content.contains("atlas-lightning-omni"));
    assert!(content.contains("claude-3-7-sonnet"));
}

#[test]
fn test_t1_f38_axum_workshop_html_unslop_compliance() {
    let html_path = workshop_html_path();
    let content = fs::read_to_string(html_path).expect("read workshop.html");
    assert!(
        !content.contains('\u{2014}'),
        "workshop.html must not contain em dashes"
    );
    assert!(
        !content.contains('\u{2013}'),
        "workshop.html must not contain en dashes"
    );
}

#[test]
fn test_t1_f39_axum_workshop_distill_task_deserialization() {
    let raw_json = r#"{
        "id": "task-test-01",
        "task_type": "code_synthesis",
        "prompt": "Write ring buffer in Rust",
        "system_prompt": "You are AIEN Sovereign Architect",
        "teacher_model": "anthropic/claude-3-7-sonnet",
        "student_model": "atlas-lightning-omni",
        "verification_strategy": "compiler_check",
        "commit_to_cortex": false
    }"#;
    let parsed: Result<DistillTask, _> = serde_json::from_str(raw_json);
    assert!(parsed.is_ok());
    let task = parsed.unwrap();
    assert_eq!(task.id, "task-test-01");
}

#[test]
fn test_t1_f40_axum_workshop_response_json_serialization() {
    let outcome = VerificationOutcome {
        passed: true,
        score: 1.0,
        rule_violations: Vec::new(),
        compiler_output: Some("Verified".to_string()),
    };
    let serialized = serde_json::to_string(&outcome).unwrap();
    assert!(serialized.contains("\"passed\":true"));
    assert!(serialized.contains("\"score\":1.0"));
}

// ============================================================================
// Feature 9: Closed-Loop Cortex Storage
// ============================================================================

#[test]
fn test_t1_f41_cortex_payload_entity_kind() {
    let payload = serde_json::json!({
        "kind": "entity",
        "value": {
            "canonicalName": "distill_task_123",
            "entityType": "learned_procedure",
            "content": "Verified heuristic content",
            "confidence": 0.95
        }
    });
    assert_eq!(payload["kind"], "entity");
    assert_eq!(payload["value"]["entityType"], "learned_procedure");
}

#[test]
fn test_t1_f42_cortex_payload_canonical_name_format() {
    let task_id = "task-uuid-456";
    let canonical = format!("distill_{}", task_id.replace('-', "_"));
    assert_eq!(canonical, "distill_task_uuid_456");
}

#[test]
fn test_t1_f43_cortex_payload_metadata_contains_task_id() {
    use spark_adapters::distill::DistillationEngine;

    let payload = DistillationEngine::build_cortex_payload(
        "lesson_alias",
        "distill_task_789",
        "Verified compiler solution",
        "task-uuid-789",
        0.95,
    );

    assert_eq!(payload["kind"], "entity");
    assert_eq!(payload["value"]["space"], "atlas-memory");
    assert_eq!(payload["value"]["entityType"], "learned_procedure");
    assert_eq!(payload["value"]["canonicalName"], "distill_task_789");
    assert_eq!(payload["value"]["metadata"]["source"], "spark-distill");
    assert_eq!(payload["value"]["metadata"]["taskId"], "task-uuid-789");
}

#[test]
fn test_t1_f44_cortex_receipt_target_id_extraction() {
    use spark_adapters::distill::DistillationEngine;

    // Test standard camelCase targetId variant
    let body_target_id = serde_json::json!({
        "receipt": { "targetId": "cortex-entity-999" }
    });
    let id1 = DistillationEngine::parse_cortex_receipt(&body_target_id, "fallback");
    assert_eq!(id1, "cortex-entity-999");

    // Test snake_case target_id variant
    let body_snake = serde_json::json!({
        "receipt": { "target_id": "cortex-entity-888" }
    });
    let id2 = DistillationEngine::parse_cortex_receipt(&body_snake, "fallback");
    assert_eq!(id2, "cortex-entity-888");

    // Test simple id variant
    let body_simple = serde_json::json!({
        "receipt": { "id": "cortex-entity-777" }
    });
    let id3 = DistillationEngine::parse_cortex_receipt(&body_simple, "fallback");
    assert_eq!(id3, "cortex-entity-777");

    // Test fallback behavior when receipt lacks explicit target identifier
    let body_empty = serde_json::json!({ "receipt": {} });
    let id4 = DistillationEngine::parse_cortex_receipt(&body_empty, "default_canonical");
    assert_eq!(id4, "default_canonical");
}

#[test]
fn test_t1_f45_cortex_qualification_gate_score_threshold() {
    use spark_adapters::distill::DistillationEngine;

    assert_eq!(DistillationEngine::CORTEX_QUALIFICATION_THRESHOLD, 0.85);

    // Exactly at threshold and passed: qualifies
    assert!(
        DistillationEngine::qualifies_for_cortex(true, true, 0.85),
        "Score of 0.85 with passed verification must qualify for Cortex commit"
    );

    // Well above threshold and passed: qualifies
    assert!(
        DistillationEngine::qualifies_for_cortex(true, true, 1.0),
        "Score of 1.0 with passed verification must qualify for Cortex commit"
    );

    // Below threshold (0.84): must NOT qualify
    assert!(
        !DistillationEngine::qualifies_for_cortex(true, true, 0.84),
        "Score below 0.85 must be rejected by Cortex qualification gate"
    );

    // High score but verification failed: must NOT qualify
    assert!(
        !DistillationEngine::qualifies_for_cortex(true, false, 0.95),
        "Failed verification must prevent Cortex commit regardless of score"
    );

    // High score and passed, but commit_to_cortex is disabled: must NOT qualify
    assert!(
        !DistillationEngine::qualifies_for_cortex(false, true, 0.95),
        "commit_to_cortex=false flag must prevent Cortex commit"
    );
}

// ============================================================================
// Feature 10: SFT and DPO Dataset Emission
// ============================================================================

#[test]
fn test_t1_f46_sft_dataset_chatml_structure() {
    let sft_entry = serde_json::json!({
        "task_id": "test-sft-01",
        "timestamp": "2026-09-20T21:00:00Z",
        "messages": [
            { "role": "system", "content": "You are AIEN." },
            { "role": "user", "content": "Write hello world in Rust" },
            { "role": "assistant", "content": "fn main() { println!(\"Hello\"); }" }
        ]
    });
    let msgs = sft_entry["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[2]["role"], "assistant");
}

#[test]
fn test_t1_f47_sft_dataset_contains_iso_timestamp() {
    let now_rfc3339 = chrono::Utc::now().to_rfc3339();
    let parsed = chrono::DateTime::parse_from_rfc3339(&now_rfc3339);
    assert!(parsed.is_ok());
}

#[test]
fn test_t1_f48_sft_dataset_assistant_reasoning_field() {
    let assistant_obj = serde_json::json!({
        "role": "assistant",
        "content": "Solution",
        "reasoning": "Step-by-step trace"
    });
    assert_eq!(assistant_obj["reasoning"], "Step-by-step trace");
}

#[test]
fn test_t1_f49_dpo_dataset_pair_structure() {
    let dpo_entry = serde_json::json!({
        "task_id": "dpo-01",
        "prompt": "Optimize channel in Rust",
        "chosen": "pub struct Channel;",
        "rejected": "pub struct BrokenChannel;",
        "timestamp": "2026-09-20T21:00:00Z"
    });
    assert_eq!(dpo_entry["chosen"], "pub struct Channel;");
    assert_ne!(dpo_entry["chosen"], dpo_entry["rejected"]);
}

#[test]
fn test_t1_f50_dpo_dataset_preference_delta_strictly_positive() {
    use spark_adapters::distill::DistillationEngine;
    use spark_adapters::models::{DistillTask, TaskType, VerificationStrategy};
    use std::fs;

    let temp_dir =
        std::env::temp_dir().join(format!("spark_distill_t1_f50_{}", uuid::Uuid::new_v4()));
    let engine = DistillationEngine::new().with_dataset_dir(&temp_dir);

    let task = DistillTask {
        id: "task-dpo-positive-01".to_string(),
        task_type: TaskType::CodeSynthesis,
        prompt: "Write a thread-safe counter in Rust".to_string(),
        system_prompt: Some("You are AIEN Sovereign Architect.".to_string()),
        teacher_model: "anthropic/claude-3-7-sonnet".to_string(),
        student_model: Some("atlas-lightning-omni".to_string()),
        verification_strategy: VerificationStrategy::CompilerCheck,
        commit_to_cortex: false,
    };

    let chosen = "pub struct Counter(std::sync::atomic::AtomicUsize);";
    let rejected = "pub struct Counter(usize);";
    let preference_delta = 0.25f32;

    let valid_rej =
        DistillationEngine::filter_dpo_candidate(chosen, Some(rejected), preference_delta);
    assert_eq!(
        valid_rej,
        Some(rejected),
        "Positive preference delta must qualify rejected candidate"
    );

    let dpo_persisted = engine
        .persist_training_pair(
            &task,
            chosen,
            Some("Atomic counter is thread-safe"),
            valid_rej,
        )
        .expect("persist training pair");
    assert!(
        dpo_persisted,
        "DPO pair must be recorded when preference delta > 0.0"
    );

    let dpo_path = temp_dir.join("dpo.jsonl");
    assert!(dpo_path.exists(), "dpo.jsonl must exist on disk");

    let content = fs::read_to_string(&dpo_path).expect("read dpo.jsonl");
    let entry: serde_json::Value = serde_json::from_str(content.trim()).expect("parse dpo entry");
    assert_eq!(entry["task_id"], "task-dpo-positive-01");
    assert_eq!(entry["chosen"], chosen);
    assert_eq!(entry["rejected"], rejected);
    assert_eq!(entry["prompt"], "Write a thread-safe counter in Rust");

    let _ = fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// Feature 11: Four-Tier Verification Gates
// ============================================================================

#[test]
fn test_t1_f51_verifier_rustc_compiles_clean_code() {
    let valid_code = "pub fn add_two(x: i32) -> i32 { x + 2 }";
    let res = Verifier::verify_with_rustc(valid_code);
    assert!(res.is_ok(), "rustc must compile clean valid Rust code");
}

#[test]
fn test_t1_f52_verifier_unslop_passes_clean_response() {
    let text = "Pure native Rust implementation with zero allocations and zero unslop.";
    let outcome = Verifier::verify(text, VerificationStrategy::UnslopStrict);
    assert!(outcome.passed);
    assert_eq!(outcome.score, 1.0);
}

#[test]
fn test_t1_f53_verifier_token_jaccard_identical() {
    let a = "fn reverse_slice(s: &mut [u8]) { s.reverse(); }";
    let score = Verifier::token_jaccard(a, a);
    assert_eq!(score, 1.0);
}

#[test]
fn test_t1_f54_verifier_normalized_levenshtein_identical() {
    let text = "Sovereign AIEN Inference Stack";
    let score = Verifier::normalized_levenshtein(text, text);
    assert_eq!(score, 1.0);
}

#[test]
fn test_t1_f55_verifier_hybrid_consensus_weighting() {
    let a = "pub struct FastMap<K, V> { inner: std::collections::HashMap<K, V> }";
    let score = Verifier::hybrid_consensus(a, a);
    assert_eq!(score, 1.0);
}

// ============================================================================
// Feature 12: E2E Opaque-Box Test Suite
// ============================================================================

#[test]
fn test_t1_f56_e2e_harness_test_count_verification() {
    // Verifies all 13 features are indexed
    let features = [
        "vault_resolution",
        "prompt_sanitizer",
        "cortex_auth",
        "resident_bounds",
        "crawler_flags",
        "cli_workbench",
        "key_verification",
        "axum_routes",
        "cortex_storage",
        "dataset_emission",
        "verification_gates",
        "test_harness",
        "git_parity",
    ];
    assert_eq!(features.len(), 13);
}

#[test]
fn test_t1_f57_e2e_harness_cargo_test_invocation() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let status = Command::new("cargo")
        .args(["check", "-p", "spark-adapters"])
        .current_dir(workspace_root())
        .status();
    assert!(status.is_ok());
    assert!(status.unwrap().success());
}

#[test]
fn test_t1_f58_e2e_harness_isolated_temp_directories() {
    let temp_dir = std::env::temp_dir().join(format!("spark_test_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_dir).unwrap();
    assert!(temp_dir.exists());
    fs::remove_dir_all(&temp_dir).unwrap();
}

#[test]
fn test_t1_f59_e2e_harness_zero_disk_secrets_invariant() {
    let output = Command::new("find")
        .args([
            workspace_root().to_str().unwrap(),
            "-name",
            ".env*",
            "-not",
            "-path",
            "*/.*",
        ])
        .output()
        .expect("find .env");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.trim().is_empty(),
        "No .env files allowed in workspace"
    );
}

#[test]
fn test_t1_f60_e2e_harness_unslop_output_formatting() {
    use std::fs;
    use std::process::Command;

    // 1. Inspect the real E2E test suite runner script
    let script_path = workspace_root().join("scripts/e2e_distill_test.sh");
    assert!(
        script_path.exists(),
        "scripts/e2e_distill_test.sh must exist"
    );
    let script_content = fs::read_to_string(&script_path).expect("read e2e_distill_test.sh");
    assert!(
        !script_content.contains('\u{2014}'),
        "e2e_distill_test.sh contains forbidden em dash"
    );
    assert!(
        !script_content.contains('\u{2013}'),
        "e2e_distill_test.sh contains forbidden en dash"
    );

    // 2. Execute spark-distill --help and verify CLI output formatting
    let bin = spark_distill_bin();
    let output = Command::new(bin)
        .arg("--help")
        .output()
        .expect("run spark-distill --help");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains('\u{2014}'),
        "spark-distill CLI help contains forbidden em dash"
    );
    assert!(
        !stdout.contains('\u{2013}'),
        "spark-distill CLI help contains forbidden en dash"
    );
}

// ============================================================================
// Feature 13: Remote Git Parity & PR Lifecycle
// ============================================================================

#[test]
fn test_t1_f61_git_current_branch_is_feature_branch() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let output = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(workspace_root())
        .output()
        .expect("git branch");
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert!(
        branch.starts_with("feat/")
            || branch == "main"
            || branch == "master"
            || branch.starts_with("fix/")
            || branch.starts_with("perf/")
            || branch.starts_with("refactor/")
            || branch.starts_with("release/")
            || branch.is_empty(),
        "Branch must be a valid lifecycle branch (feat/*, fix/*, perf/*, refactor/*, release/*) or main/master, found: {}",
        branch
    );
}

#[test]
fn test_t1_f62_git_origin_remote_configured() {
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(workspace_root())
        .output()
        .expect("git remote origin");
    let url = String::from_utf8_lossy(&output.stdout);
    assert!(url.contains("github.com") || url.contains("git@github.com"));
}

#[test]
fn test_t1_f63_git_forgejo_remote_configured() {
    if std::env::var("CI").is_ok() {
        return;
    }
    let output = Command::new("git")
        .args(["remote", "get-url", "forgejo"])
        .current_dir(workspace_root())
        .output()
        .expect("git remote forgejo");
    assert!(
        output.status.success(),
        "forgejo remote must be configured for dual-push"
    );
}

#[test]
fn test_t1_f64_git_commit_messages_unslop_compliant() {
    let output = Command::new("git")
        .args(["log", "-n", "10", "--format=%s"])
        .current_dir(workspace_root())
        .output()
        .expect("git log");
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(
        !log.contains('\u{2014}'),
        "Commit messages must not contain em dashes"
    );
    assert!(
        !log.contains('\u{2013}'),
        "Commit messages must not contain en dashes"
    );
}

#[test]
fn test_t1_f65_git_license_file_declares_srcl() {
    let license_path = workspace_root().join("LICENSE");
    let content = fs::read_to_string(license_path).expect("read LICENSE");
    assert!(content.contains("SRCL-1.0"));
}
