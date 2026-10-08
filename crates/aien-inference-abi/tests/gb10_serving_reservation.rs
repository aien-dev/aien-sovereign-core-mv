//! sovereign-core#277 (cut toward it, not a fix): when the daemon serves Qwen3 on the GB10
//! (the already opt-in path), start-up reserves the serving buffers once, from declared
//! bounds, and refuses to serve by name when omega refuses. CPU only: the reservation call is
//! injected, nothing touches a driver.
use aien_inference_abi::gb10_serving::{reserve_gb10_serving_with, ServingLimits};
use aien_inference_abi::model_config_from_hf_json;
use aien_omega_gpu::ServingBounds;
use std::cell::RefCell;
use std::path::PathBuf;

fn qwen3_4b() -> aien_inference_abi::ModelConfig {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("qwen3-4b-instruct-2507-config");
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let gen = std::fs::read_to_string(dir.join("generation_config.json")).unwrap();
    model_config_from_hf_json("Qwen/Qwen3-4B-Instruct-2507", &cfg, Some(&gen)).unwrap()
}

/// The daemon's declared limits (aien-cli run_daemon_server: max_batch_size 256, prefill chunk
/// 128, arena 4096), with the KV pool planned for 4096 tokens.
fn daemon_limits() -> ServingLimits {
    ServingLimits {
        context_tokens: 4096,
        max_batch_rows: 256,
        prefill_chunk_rows: 128,
    }
}

#[test]
fn qwen3_on_gb10_reserves_once_before_serving_with_the_declared_bounds() {
    let calls: RefCell<Vec<ServingBounds>> = RefCell::new(Vec::new());
    let got = reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), true, true, &|b| {
        calls.borrow_mut().push(*b);
        Ok(())
    })
    .expect("reserved")
    .expect("Qwen3 on GB10 must reserve");
    let calls = calls.into_inner();
    assert_eq!(calls.len(), 1, "exactly one reservation call");
    let b = calls[0];
    assert_eq!(got.bounds, b);
    assert_eq!((b.num_q_heads, b.num_kv_heads, b.head_dim), (32, 8, 128));
    assert_eq!(b.max_context, 4096);
    assert_eq!(
        b.max_seqs, 1,
        "the GB10 backend runs attention one sequence at a time"
    );
    assert_eq!(b.kv_block_size as usize, qwen3_4b().block_size);
    assert_eq!(
        b.max_rows, 256,
        "decode batch rows (256) exceed the prefill chunk (128)"
    );
    assert_eq!(b.max_k, 9728, "down_proj input is the intermediate size");
    assert_eq!(b.max_n_one_row, 151936, "the logits projection");
    assert!(b.max_n >= 9728);
    assert!(
        b.kernel_slots >= 21,
        "21 distinct matmul shapes for Qwen3-4B (design note)"
    );
    assert!(got.bytes.total() > 0);
}

#[test]
fn context_is_the_smaller_of_the_kv_plan_and_the_attention_limit() {
    let seen = RefCell::new(Vec::new());
    let rec = |b: &ServingBounds| {
        seen.borrow_mut().push(b.max_context);
        Ok(())
    };
    let mut l = daemon_limits();
    l.context_tokens = 2048;
    reserve_gb10_serving_with(&qwen3_4b(), &l, true, true, &rec).unwrap();
    l.context_tokens = 262144; // the model's own declared context: omega's f32 attention stops at 4096
    reserve_gb10_serving_with(&qwen3_4b(), &l, true, true, &rec).unwrap();
    assert_eq!(*seen.borrow(), vec![2048, 4096]);
}

#[test]
fn a_refusal_stops_serving_by_name_with_no_fallback() {
    let err = reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), true, true, &|_| {
        Err("omega_gpu rc=-4 (CHIP_FAIL) (stage: nvrm_alloc scratch)".to_string())
    })
    .expect_err("a refused reservation must refuse to serve");
    assert!(err.contains("GB10_SERVING_RESERVATION refused"), "{err}");
    assert!(err.contains("CHIP_FAIL"), "names omega's reason: {err}");
    assert!(err.contains("sovereign-core#277"), "{err}");
    assert!(err.contains("no on-demand fallback"), "{err}");
}

#[test]
fn nothing_changes_off_the_opt_in_qwen3_gb10_path() {
    let never = |_: &ServingBounds| -> Result<(), String> { panic!("must not reserve") };
    // not native (CPU daemon): no call
    assert_eq!(
        reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), false, true, &never).unwrap(),
        None
    );
    // default refusal still in force (no opt-in): the reservation is not what serves it
    assert_eq!(
        reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), true, false, &never).unwrap(),
        None
    );
    // a non-Qwen3 model on the GB10 is untouched
    let mut llama = qwen3_4b();
    llama.qk_norm = false;
    assert_eq!(
        reserve_gb10_serving_with(&llama, &daemon_limits(), true, true, &never).unwrap(),
        None
    );
}
