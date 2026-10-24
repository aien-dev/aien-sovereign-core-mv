// Challenger Milestone 1 Adversarial Stress Suite:
// Concurrency Caching, Fallback Behavior, Payload Bounds, and Dynamic Cortex Auth
// Strictly adheres to sovereign unslop invariants: zero em dashes, zero en dashes.

use spark_adapters::distill::DistillationEngine;
use spark_adapters::models::ChatMessage;
use spark_adapters::providers::openai::{
    format_openai_payload, format_openai_payload_bounded, DEFAULT_MAX_TOKENS,
};
use spark_adapters::vault::{
    get_scrub_targets, is_secret_present, list_vault_keys, register_test_secret,
    remove_test_secret, resolve_secret, sanitize_outbound_prompt,
};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Challenge 1: Concurrency and TTL Cache Invalidation (50 threads)
// ============================================================================

#[test]
fn test_challenge1_concurrency_50_threads_sub_millisecond_reads() {
    let _concurrency_guard = VAULT_CONCURRENCY_LOCK.lock().unwrap();
    // Prime and warm the cache with known secrets
    register_test_secret(
        "CONCURRENCY_TEST_KEY_1",
        "initial_concurrency_secret_value_1",
    );
    register_test_secret(
        "CONCURRENCY_TEST_KEY_2",
        "initial_concurrency_secret_value_2",
    );

    // Warm-up pass to ensure cached state is populated prior to concurrent execution
    let _ = resolve_secret("CONCURRENCY_TEST_KEY_1");
    let _ = resolve_secret("CONCURRENCY_TEST_KEY_2");
    let _ = get_scrub_targets();

    let num_threads = 50;
    // Each thread times many short batches instead of one long run. A thread that the OS
    // deschedules (busy judge, other builds) loses the time inside the batch it was paused in,
    // but a slow or lock-blocked read slows EVERY batch. The median batch tells the two apart.
    let batches_per_thread = 20;
    let iterations_per_batch = 5;
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = Vec::with_capacity(num_threads);

    for thread_idx in 0..num_threads {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            // Synchronize all 50 threads to fire at the exact same moment
            b.wait();

            let start = Instant::now();
            let mut batch_avg_micros = Vec::with_capacity(batches_per_thread);

            for _ in 0..batches_per_thread {
                let batch_start = Instant::now();
                for i in 0..iterations_per_batch {
                    let key = if i % 2 == 0 {
                        "CONCURRENCY_TEST_KEY_1"
                    } else {
                        "CONCURRENCY_TEST_KEY_2"
                    };

                    let secret = resolve_secret(key);
                    assert!(
                        secret.is_some(),
                        "Secret must resolve in thread {}",
                        thread_idx
                    );

                    let targets = get_scrub_targets();
                    assert!(
                        !targets.is_empty(),
                        "Scrub targets must not be empty in thread {}",
                        thread_idx
                    );
                }
                let batch_ops = iterations_per_batch * 2;
                batch_avg_micros.push(batch_start.elapsed().as_micros() as f64 / batch_ops as f64);
            }

            let elapsed = start.elapsed();
            batch_avg_micros.sort_by(|a, b| a.total_cmp(b));
            let median_op_micros = batch_avg_micros[batches_per_thread / 2];

            (thread_idx, elapsed, median_op_micros)
        }));
    }

    let mut total_duration = Duration::from_secs(0);
    let mut max_median_micros: f64 = 0.0;

    for h in handles {
        let (t_idx, elapsed, median_micros) = h.join().expect("Thread must not panic or deadlock");
        if elapsed > total_duration {
            total_duration = elapsed;
        }
        if median_micros > max_median_micros {
            max_median_micros = median_micros;
        }
        assert!(
            median_micros < 1000.0,
            "Thread {} median batch operation time ({:.2} us) must be sub-millisecond (< 1000 us)",
            t_idx,
            median_micros
        );
    }

    println!(
        "Challenge 1 Pass: 50 concurrent threads completed {} operations in {:.2?}. Max median latency: {:.2} us/op (sub-millisecond invariant verified).",
        num_threads * batches_per_thread * iterations_per_batch * 2,
        total_duration,
        max_median_micros
    );
}

#[test]
fn test_challenge1_concurrency_50_threads_mixed_read_write_invalidation() {
    let _concurrency_guard = VAULT_CONCURRENCY_LOCK.lock().unwrap();
    let num_threads = 50;
    let iterations = 20;
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = Vec::with_capacity(num_threads);

    for thread_idx in 0..num_threads {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();

            if thread_idx < 10 {
                // Writer thread: continuously registers new secrets and invalidates cache
                for i in 0..iterations {
                    let key = format!("DYN_KEY_{}_{}", thread_idx, i);
                    let val = format!("dyn_secret_token_val_{}_{}_secure", thread_idx, i);
                    register_test_secret(&key, &val);
                    thread::sleep(Duration::from_micros(100));
                }
            } else if thread_idx < 30 {
                // Reader thread: verifies resolve_secret under concurrent invalidation
                for i in 0..iterations {
                    let key = format!("DYN_KEY_{}_{}", thread_idx % 10, i);
                    let _ = resolve_secret(&key);
                    let _ = resolve_secret("CONCURRENCY_TEST_KEY_1");
                }
            } else {
                // Sanitizer thread: verifies prompt scrubbing under concurrent invalidation
                for i in 0..iterations {
                    let raw_prompt = format!(
                        "Host spark at 127.0.0.1 for Drake Stapleton with DYN_KEY_{}_{}",
                        thread_idx % 10,
                        i
                    );
                    let cleaned = sanitize_outbound_prompt(&raw_prompt);
                    assert!(
                        !cleaned.contains("Drake Stapleton"),
                        "Operator name must be scrubbed"
                    );
                    assert!(!cleaned.contains("127.0.0.1"), "Local IP must be scrubbed");
                }
            }
        }));
    }

    for h in handles {
        h.join()
            .expect("Concurrent read-write thread must not deadlock or panic");
    }

    // Verify final state consistency
    let final_targets = get_scrub_targets();
    assert!(
        final_targets.len() >= 10,
        "Scrub targets must incorporate dynamically registered secrets"
    );
    println!("Challenge 1 Mixed Read/Write Pass: 50 threads executed without deadlock or race conditions.");
}

// ============================================================================
// Challenge 2: Fallback Behavior (Present vs Missing / Absent)
// ============================================================================

#[test]
fn test_challenge2_fallback_graceful_missing_and_empty_keys() {
    // Missing key resolution must return None gracefully without panic
    let missing_secret = resolve_secret("DEFINITELY_NON_EXISTENT_KEY_998877");
    assert_eq!(missing_secret, None);

    // Empty key string must return None gracefully
    let empty_secret = resolve_secret("");
    assert_eq!(empty_secret, None);

    // Missing key presence check must return false gracefully
    let presence = is_secret_present("DEFINITELY_NON_EXISTENT_KEY_998877");
    assert!(!presence);

    // Empty key presence check must return false
    let empty_presence = is_secret_present("");
    assert!(!empty_presence);

    // list_vault_keys must succeed even if no keys match
    let keys = list_vault_keys();
    assert!(keys.iter().all(|k| !k.is_empty()));
    println!(
        "Challenge 2 Fallback Pass: Missing and empty keys handled gracefully with zero panics."
    );
}

#[test]
fn test_challenge2_fallback_physical_tpm_or_cached_keys() {
    let _guard = CORTEX_CHALLENGE_LOCK.blocking_lock();
    // Physical TPM key presence on DGX Spark
    let keys = list_vault_keys();
    let has_cortex = keys.iter().any(|k| k == "CORTEX_TOKEN");
    if has_cortex {
        let cortex_tok = resolve_secret("CORTEX_TOKEN");
        assert!(
            cortex_tok.is_some(),
            "CORTEX_TOKEN present in vault list must resolve"
        );
        let presence = is_secret_present("CORTEX_TOKEN");
        assert!(
            presence,
            "is_secret_present must return true for CORTEX_TOKEN"
        );
        println!(
            "Challenge 2 Physical TPM Pass: CORTEX_TOKEN successfully resolved from TPM vault."
        );
    } else {
        println!("Challenge 2 Physical TPM Note: CORTEX_TOKEN not currently registered in physical vault.");
    }
}

// ============================================================================
// Challenge 3: Generation Bounding on OpenAI and Reasoning Models
// ============================================================================

#[test]
fn test_challenge3_openai_payload_bounds_never_unconstrained() {
    let dummy_messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a compiler assistant.".to_string(),
            reasoning: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Write a high-performance hash map in Rust.".to_string(),
            reasoning: None,
        },
    ];

    // 1. Verify default bounds
    assert_eq!(DEFAULT_MAX_TOKENS, 4096);

    let standard_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "atlas-lightning-omni",
        "deepseek-chat",
        "claude-3-7-sonnet",
    ];

    for model in standard_models {
        // Default format
        let p_default = format_openai_payload(model, &dummy_messages, None, false);
        assert_eq!(
            p_default["max_tokens"], 4096,
            "Standard model {} must have max_tokens 4096 by default",
            model
        );
        assert!(
            p_default.get("max_completion_tokens").is_none(),
            "Standard model {} must NOT have max_completion_tokens",
            model
        );

        // Bounded with None (should fall back to DEFAULT_MAX_TOKENS, never unbounded)
        let p_none = format_openai_payload_bounded(model, &dummy_messages, None, None, false);
        assert_eq!(
            p_none["max_tokens"], 4096,
            "Standard model {} with None bound must fall back to default limit",
            model
        );

        // Bounded with custom override
        let p_custom =
            format_openai_payload_bounded(model, &dummy_messages, None, Some(1024), false);
        assert_eq!(
            p_custom["max_tokens"], 1024,
            "Standard model {} must respect custom max_tokens 1024",
            model
        );
    }

    // 2. Reasoning models: o1, o3 series
    let reasoning_models = ["o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o3-high"];

    for model in reasoning_models {
        // Default format
        let p_default = format_openai_payload(model, &dummy_messages, None, false);
        assert_eq!(
            p_default["max_completion_tokens"], 4096,
            "Reasoning model {} must have max_completion_tokens 4096 by default",
            model
        );
        assert!(
            p_default.get("max_tokens").is_none(),
            "Reasoning model {} must NOT have max_tokens field",
            model
        );

        // Bounded with None
        let p_none = format_openai_payload_bounded(model, &dummy_messages, None, None, false);
        assert_eq!(
            p_none["max_completion_tokens"], 4096,
            "Reasoning model {} with None bound must fall back to default limit",
            model
        );

        // Bounded with custom override
        let p_custom =
            format_openai_payload_bounded(model, &dummy_messages, None, Some(8192), false);
        assert_eq!(
            p_custom["max_completion_tokens"], 8192,
            "Reasoning model {} must respect custom max_completion_tokens 8192",
            model
        );
    }

    // 3. Extreme boundary values: Some(1), Some(65536)
    let p_boundary_low =
        format_openai_payload_bounded("gpt-4o", &dummy_messages, None, Some(1), false);
    assert_eq!(p_boundary_low["max_tokens"], 1);

    let p_boundary_high =
        format_openai_payload_bounded("o3-mini", &dummy_messages, None, Some(65536), false);
    assert_eq!(p_boundary_high["max_completion_tokens"], 65536);

    println!("Challenge 3 Payload Bounding Pass: All models bounded strictly; reasoning models correctly map to max_completion_tokens.");
}

// ============================================================================
// Challenge 4: Cortex Dynamic Auth and Zero Disk Touches
// ============================================================================

static CORTEX_CHALLENGE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static VAULT_CONCURRENCY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[tokio::test]
async fn test_challenge4_cortex_commit_missing_token_fails_gracefully() {
    let _guard = CORTEX_CHALLENGE_LOCK.lock().await;
    // Explicitly simulate empty/missing token in test registry
    register_test_secret("CORTEX_TOKEN", "");

    let engine = DistillationEngine::new();
    let res = engine
        .commit_to_cortex(
            "test_entity_missing",
            "canonical_missing",
            "content here",
            "task_missing_1",
            0.9,
        )
        .await;

    // Must return Err indicating token missing, with zero panics
    assert!(res.is_err(), "Must return error when token is missing");
    let err_msg = res.unwrap_err();
    assert!(
        err_msg.contains("CORTEX_TOKEN not found"),
        "Error must explicitly mention CORTEX_TOKEN not found, got: {}",
        err_msg
    );
    remove_test_secret("CORTEX_TOKEN");
    println!(
        "Challenge 4 Missing Token Pass: Gracefully returned error: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_challenge4_cortex_dynamic_auth_and_zero_disk_touches() {
    let _guard = CORTEX_CHALLENGE_LOCK.lock().await;
    let mock_vault_token = "atlas_dynamic_vault_token_verification_secret_7718";
    register_test_secret("CORTEX_TOKEN", mock_vault_token);

    // Bind local ephemeral TCP listener for mock Cortex server
    let listener = TcpListener::bind("127.0.0.1:0").expect("Bind ephemeral port for mock Cortex");
    let port = listener.local_addr().expect("Local addr").port();
    listener.set_nonblocking(false).expect("Blocking listener");

    // Spawn server thread to capture incoming request and serve valid JSON receipt
    let server_handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("Accept incoming connection");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("Read timeout");

        let mut raw_data = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = stream.read(&mut buf).expect("Read HTTP request");
            if n == 0 {
                break;
            }
            raw_data.extend_from_slice(&buf[..n]);
            if let Some(pos) = raw_data.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&raw_data[..pos]);
                let mut content_len = 0;
                for line in headers.lines() {
                    if line.to_ascii_lowercase().starts_with("content-length:") {
                        if let Some(val) = line.split_once(58 as char).map(|x| x.1) {
                            content_len = val.trim().parse::<usize>().unwrap_or(0);
                        }
                    }
                }
                if raw_data.len() >= pos + 4 + content_len {
                    break;
                }
            }
        }

        let req_text = String::from_utf8_lossy(&raw_data).to_string();

        let json_body = "{\"receipt\":{\"targetId\":\"cortex_entity_777\"}}";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            json_body.len(),
            json_body
        );
        stream
            .write_all(resp.as_bytes())
            .expect("Write HTTP response");
        stream.flush().expect("Flush response");

        req_text
    });

    // Create decoy disk file ~/.config/cortex/token with bogus content if not present
    let home = std::env::var("HOME")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let cortex_dir = PathBuf::from(&home).join(".config/cortex");
    let decoy_file = cortex_dir.join("token");
    let decoy_token = "BOGUS_DEPRECATED_DISK_TOKEN_NEVER_USE";

    let had_decoy = decoy_file.exists();
    if !had_decoy {
        let _ = fs::create_dir_all(&cortex_dir);
        let _ = fs::write(&decoy_file, decoy_token);
    }

    let mut engine = DistillationEngine::new();
    engine.cortex_endpoint = format!("http://127.0.0.1:{}", port);

    let result = engine
        .commit_to_cortex(
            "test_learned_entity",
            "test_canonical_procedure",
            "Optimized kernel execution with zero-copy KV blocks",
            "task_distill_9918",
            0.98,
        )
        .await;

    // Clean up decoy file immediately if we created it
    if !had_decoy && decoy_file.exists() {
        let _ = fs::remove_file(&decoy_file);
    }

    assert!(
        result.is_ok(),
        "commit_to_cortex must succeed against mock server: {:?}",
        result.err()
    );
    let entity_id = result.unwrap();
    assert_eq!(entity_id, "cortex_entity_777");

    let received_http_req = server_handle
        .join()
        .expect("Mock server thread must complete");

    // Verify 1: Dynamic vault token was sent, NOT decoy disk token
    let expected_auth_header = format!("authorization: bearer {}", mock_vault_token);
    assert!(
        received_http_req
            .to_lowercase()
            .contains(&expected_auth_header),
        "Request must contain dynamic vault token Authorization header. Received:\n{}",
        received_http_req
    );
    assert!(
        !received_http_req.contains(decoy_token),
        "Request must NEVER contain decoy disk token!"
    );

    // Verify 2: Space is atlas-memory
    assert!(
        received_http_req.contains("\"space\":\"atlas-memory\"")
            || received_http_req.contains("\"space\": \"atlas-memory\""),
        "Payload must target atlas-memory space"
    );

    // Verify 3: Payload attributes
    assert!(received_http_req.contains("test_canonical_procedure"));
    assert!(received_http_req.contains("learned_procedure"));

    remove_test_secret("CORTEX_TOKEN");
    println!("Challenge 4 Dynamic Auth Pass: Cortex request verified with dynamic vault token and atlas-memory space; zero disk token leakage.");
}
