//! GB10-4: strict real-model execution gate (AIEN 0.1 commissioning).
//!
//! Non-ignored tests prove the production build fails closed: no checkpoint
//! on disk means a refusal, not reference weights. The ignored gate runs the
//! real TinyLlama checkpoint on the GB10 backend and writes the acceptance
//! receipt (checkpoint sha256, tokenizer sha256, backend identity, model
//! config, fallback_count). Run on the Spark:
//!   AIEN_E2E_CHECKPOINT=~/models/TinyLlama-1.1B-Chat-v1.0 \
//!   AIEN_STRICT_RECEIPT=/path/receipt.json \
//!   cargo test -p aien-inference-runtime --release --test strict_real_model -- --ignored --nocapture

mod drift;
mod oracle;

use aien_inference_abi::backend::TensorBackend;
use aien_inference_abi::strict::{self, StrictModelReceipt};
use aien_inference_abi::ExecutionSurface;
use aien_inference_runtime::model::EmbeddedModel;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn sha256_file_hex(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

#[test]
fn production_build_is_strict_unless_opted_in() {
    if std::env::var(strict::DEV_FALLBACK_ENV).is_ok() {
        eprintln!(
            "dev fallback opted in by {}; strictness not asserted",
            strict::DEV_FALLBACK_ENV
        );
        return;
    }
    assert!(
        strict::production_strict(),
        "this build has dev-fallback on; the strict gate must run on a production build"
    );
}

#[test]
fn missing_checkpoint_is_refused_not_faked() {
    let missing = PathBuf::from("/nonexistent/aien-strict-gate/model.safetensors");
    let tok = PathBuf::from("/nonexistent/aien-strict-gate/tokenizer.json");
    let err = EmbeddedModel::load_checkpoint(&missing, &tok, true, true)
        .err()
        .expect("a missing checkpoint must be an error");
    assert!(
        !err.contains("reference"),
        "refusal must not route to reference weights: {err}"
    );
}

#[test]
fn receipt_rejects_fallbacks_and_dev_builds() {
    let mut r = StrictModelReceipt {
        checkpoint_path: "x".into(),
        checkpoint_sha256: "0".repeat(64),
        tokenizer_path: "y".into(),
        tokenizer_sha256: "0".repeat(64),
        backend_identity: "OmegaGb10Backend (native Omega engine, no CUDA, NVIDIA GB10 sm_121)"
            .into(),
        model_id: "tinyllama".into(),
        model_config: serde_json::json!({}),
        fallback_count: 0,
        op_report: None,
        dev_fallback_build: false,
        verdict: String::new(),
    };
    assert!(r.verify().is_ok(), "{}", r.verdict);
    r.fallback_count = 1;
    assert!(r.verify().is_err());
    r.fallback_count = 0;
    r.dev_fallback_build = true;
    assert!(r.verify().is_err());
    r.dev_fallback_build = false;
    r.backend_identity = "ReferenceCpuBackend".into();
    assert!(r.verify().is_err());
}

#[test]
#[ignore = "needs the real checkpoint on the Spark: AIEN_E2E_CHECKPOINT + AIEN_STRICT_RECEIPT"]
fn strict_real_model_gate() {
    let ckpt = PathBuf::from(std::env::var("AIEN_E2E_CHECKPOINT").expect("AIEN_E2E_CHECKPOINT"));
    let receipt_path =
        PathBuf::from(std::env::var("AIEN_STRICT_RECEIPT").expect("AIEN_STRICT_RECEIPT"));
    let (model_path, dir) = if ckpt.is_dir() {
        (ckpt.join("model.safetensors"), ckpt.clone())
    } else {
        (
            ckpt.clone(),
            ckpt.parent().map(Path::to_path_buf).unwrap_or_default(),
        )
    };
    let tokenizer_path = dir.join("tokenizer.json");
    assert!(
        model_path.is_file(),
        "STRICT: checkpoint {} missing",
        model_path.display()
    );
    assert!(
        tokenizer_path.is_file(),
        "STRICT: tokenizer {} missing",
        tokenizer_path.display()
    );

    let surface = ExecutionSurface::detect();
    println!("STRICT_GATE surface: {}", surface.display_name());
    let mut model = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, true, true)
        .unwrap_or_else(|e| panic!("STRICT: bind refused: {e}"));
    let backend_identity = model.transformer.tensor_backend.name().to_string();
    println!("STRICT_GATE backend: {backend_identity}");

    let prompt = "The DGX Spark is a small computer with a large";
    let out = model
        .generate(prompt, 8, 0.0)
        .unwrap_or_else(|e| panic!("STRICT: generate failed: {e}"));
    println!("STRICT_GATE output: {out:?}");
    let fallback_count = model.transformer.tensor_backend.fallback_count();

    println!(
        "STRICT_GATE {}",
        model.transformer.tensor_backend.op_report().line()
    );
    let mut receipt = StrictModelReceipt {
        checkpoint_path: model_path.display().to_string(),
        checkpoint_sha256: sha256_file_hex(&model_path),
        tokenizer_path: tokenizer_path.display().to_string(),
        tokenizer_sha256: sha256_file_hex(&tokenizer_path),
        backend_identity,
        model_id: model.config.model_id.clone(),
        model_config: serde_json::to_value(&model.config)
            .unwrap_or(serde_json::json!({"model_id": model.config.model_id})),
        fallback_count,
        op_report: Some(model.transformer.tensor_backend.op_report()),
        dev_fallback_build: strict::dev_fallback_active(),
        verdict: String::new(),
    };
    let verdict = receipt.verify();
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .expect("write receipt");
    println!(
        "STRICT_GATE verdict: {} (receipt {})",
        receipt.verdict,
        receipt_path.display()
    );
    assert!(!out.trim().is_empty(), "STRICT: empty output");
    verdict.unwrap_or_else(|e| panic!("{e}"));
}

/// FB-1 cut 3b gate: the same greedy generation on the CPU reference backend and on
/// the Omega backend (the default GPU path), same checkpoint. Records tok/s for
/// both, checks zero fallbacks and all nine ops native, and judges the backend by teacher-
/// forced logits (tests/drift): greedy text alone cannot tell a wrong backend from a
/// near-tie flip (cut 3d: "processing" led "memory" by 0.0005 logits). Text equality is
/// printed (tokens_match) but no longer asserted.
///   AIEN_E2E_CHECKPOINT=~/models/TinyLlama-1.1B-Chat-v1.0 AIEN_STRICT_RECEIPT=/path/receipt.json \
///   cargo test -p aien-inference-runtime --release --test strict_real_model \
///     omega_vs_reference_real_model -- --ignored --nocapture
#[test]
#[ignore = "needs the real checkpoint and the GB10 with a native Omega build; heavy queue only"]
fn omega_vs_reference_real_model() {
    let ckpt = PathBuf::from(std::env::var("AIEN_E2E_CHECKPOINT").expect("AIEN_E2E_CHECKPOINT"));
    let receipt_path =
        PathBuf::from(std::env::var("AIEN_STRICT_RECEIPT").expect("AIEN_STRICT_RECEIPT"));
    let (model_path, dir) = if ckpt.is_dir() {
        (ckpt.join("model.safetensors"), ckpt.clone())
    } else {
        (
            ckpt.clone(),
            ckpt.parent().map(Path::to_path_buf).unwrap_or_default(),
        )
    };
    let tokenizer_path = dir.join("tokenizer.json");
    let prompt = "The DGX Spark is a small computer with a large";
    let max_tokens = 32usize;

    let run = |model: &mut EmbeddedModel| -> (String, usize, f64) {
        let mut text = String::new();
        let mut n = 0usize;
        let t = std::time::Instant::now();
        model
            .generate_stream(prompt, max_tokens, 0.0, |piece| {
                text.push_str(piece);
                n += 1;
                true
            })
            .unwrap_or_else(|e| panic!("generate failed: {e}"));
        (text, n, t.elapsed().as_secs_f64())
    };

    // Reference (CPU) leg.
    let mut cpu = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, false, true)
        .unwrap_or_else(|e| panic!("reference load: {e}"));
    println!(
        "OMEGA_GATE ref backend: {}",
        cpu.transformer.tensor_backend.name()
    );
    let (cpu_text, cpu_n, cpu_s) = run(&mut cpu);
    println!(
        "OMEGA_GATE ref tokens={cpu_n} secs={cpu_s:.3} tok/s={:.3} text={cpu_text:?}",
        cpu_n as f64 / cpu_s
    );
    let ptoks = cpu.tokenizer.encode(prompt).expect("encode");
    let (ref_logits, ref_toks) =
        drift::logits_per_step(&mut cpu.transformer, &ptoks, max_tokens, &[]);
    drop(cpu);

    // Omega leg. No AIEN_GPU_BACKEND switch: since FB-1 cut 6 the GPU path is Omega by default.
    let mut om = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, true, true)
        .unwrap_or_else(|e| panic!("STRICT: omega bind refused: {e}"));
    let backend_identity = om.transformer.tensor_backend.name().to_string();
    println!("OMEGA_GATE omega backend: {backend_identity}");
    // Warm-up: first call uploads every weight matrix; reported separately.
    let t = std::time::Instant::now();
    let _ = om.generate(prompt, 1, 0.0).expect("omega warm-up");
    println!(
        "OMEGA_GATE omega warm-up (weight upload + 1 token) secs={:.3}",
        t.elapsed().as_secs_f64()
    );
    let (om_text, om_n, om_s) = run(&mut om);
    let (om_logits, _) = drift::logits_per_step(&mut om.transformer, &ptoks, max_tokens, &ref_toks);
    let cmp = drift::compare(&ref_logits, &om_logits);
    for c in cmp.iter().filter(|c| c.ref_tok != c.cand_tok) {
        println!(
            "OMEGA_GATE flip step={} ref_tok={} omega_tok={} ref_margin={:.4} max_abs_dlogit={:.4}",
            c.step, c.ref_tok, c.cand_tok, c.ref_margin, c.max_abs_dlogit
        );
    }
    let worst = cmp.iter().map(|c| c.max_abs_dlogit).fold(0.0f32, f32::max);
    let drift_verdict = drift::judge(&cmp);
    println!(
        "OMEGA_GATE teacher_forced steps={} worst_max_abs_dlogit={worst:.4} bound={} judge={drift_verdict:?}",
        cmp.len(),
        drift::MAX_ABS_DLOGIT
    );
    println!(
        "OMEGA_GATE omega tokens={om_n} secs={om_s:.3} tok/s={:.3} text={om_text:?}",
        om_n as f64 / om_s
    );
    let report = om.transformer.tensor_backend.op_report();
    println!("OMEGA_GATE {}", report.line());
    let fallback_count = om.transformer.tensor_backend.fallback_count();
    println!("OMEGA_GATE tokens_match={}", cpu_text == om_text);
    for op in [
        "matmul_vec",
        "matmul_batch",
        "compute_logits",
        "rmsnorm",
        "apply_rope",
        "swiglu",
        "gqa_attention",
        "paged_attention",
        "paged_attention_batch",
    ] {
        assert!(
            report.native_ops.iter().any(|n| n == op),
            "{op} not in the native mask: {}",
            report.line()
        );
    }
    assert!(
        report.native_fallbacks.is_empty(),
        "native fallbacks: {}",
        report.line()
    );

    let mut receipt = StrictModelReceipt {
        checkpoint_path: model_path.display().to_string(),
        checkpoint_sha256: sha256_file_hex(&model_path),
        tokenizer_path: tokenizer_path.display().to_string(),
        tokenizer_sha256: sha256_file_hex(&tokenizer_path),
        backend_identity,
        model_id: om.config.model_id.clone(),
        model_config: serde_json::to_value(&om.config)
            .unwrap_or(serde_json::json!({"model_id": om.config.model_id})),
        fallback_count,
        op_report: Some(report),
        dev_fallback_build: strict::dev_fallback_active(),
        verdict: String::new(),
    };
    let verdict = receipt.verify();
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .expect("write receipt");
    println!(
        "OMEGA_GATE verdict: {} (receipt {})",
        receipt.verdict,
        receipt_path.display()
    );
    if let Err(e) = drift_verdict {
        panic!("Omega backend logits disagree with the reference: {e}");
    }
    verdict.unwrap_or_else(|e| panic!("{e}"));
}

/// The oracle fixture is self-consistent: its recorded sha256 matches the file, the prompt is
/// 46 tokens, the greedy decode is 16 tokens and starts with the oracle's argmax. Host test.
#[test]
fn oracle_fixture_is_intact() {
    let o = oracle::load();
    assert_eq!(
        sha256_file_hex(&oracle::fixtures_dir().join("tinyllama_oracle.safetensors")),
        o.safetensors_sha256
    );
    assert_eq!(o.prompt_tokens.len(), 46);
    assert_eq!(o.greedy_tokens.len(), 16);
    assert_eq!(
        drift::top2(&o.last_token_logits).0 as u32,
        o.greedy_tokens[0]
    );
}

/// GB10 strict gate against the INDEPENDENT reference (Hugging Face FP32 oracle fixture), not
/// our own CPU backend. The backend is chosen by the production runtime path
/// (`EmbeddedModel::load_checkpoint`, use_gpu), then wrapped only to count calls per op.
/// Judged with the frozen drift rule (`drift::MAX_ABS_DLOGIT`, near-tie flips only):
/// step 0 compares all 32000 logits with the oracle's; steps 0..15 are teacher-forced with the
/// oracle's greedy tokens and every argmax must equal the oracle token unless the oracle token
/// is within 2 * MAX_ABS_DLOGIT of the Omega top logit (the same near-tie bound `drift::judge`
/// uses). Every op that ran must have run natively (zero fallbacks, zero reference runs).
///   AIEN_E2E_CHECKPOINT=<TinyLlama dir with model.safetensors + tokenizer.json> \
///   AIEN_STRICT_RECEIPT=/path/receipt.json cargo test -p aien-inference-runtime --release \
///     --test strict_real_model omega_vs_hf_oracle -- --ignored --nocapture
#[test]
#[ignore = "needs the real checkpoint and the GB10 with a native Omega build; heavy queue only"]
fn omega_vs_hf_oracle() {
    use aien_inference_abi::native_ops::TensorOp;
    use std::sync::Arc;

    let o = oracle::load();
    let ckpt = PathBuf::from(std::env::var("AIEN_E2E_CHECKPOINT").expect("AIEN_E2E_CHECKPOINT"));
    let receipt_path =
        PathBuf::from(std::env::var("AIEN_STRICT_RECEIPT").expect("AIEN_STRICT_RECEIPT"));
    let model_path = ckpt.join("model.safetensors");
    let tokenizer_path = ckpt.join("tokenizer.json");
    let model_sha = sha256_file_hex(&model_path);
    let tokenizer_sha = sha256_file_hex(&tokenizer_path);
    println!(
        "ORACLE_GATE checkpoint {} sha256={model_sha}",
        model_path.display()
    );
    println!(
        "ORACLE_GATE tokenizer {} sha256={tokenizer_sha}",
        tokenizer_path.display()
    );
    assert_eq!(
        model_sha, o.model_sha256,
        "checkpoint is not the oracle's model"
    );
    assert_eq!(
        tokenizer_sha, o.tokenizer_sha256,
        "tokenizer is not the oracle's"
    );

    let mut model = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, true, true)
        .unwrap_or_else(|e| panic!("STRICT: omega bind refused: {e}"));
    let backend_identity = model.transformer.tensor_backend.name().to_string();
    println!("ORACLE_GATE backend: {backend_identity}");
    println!(
        "ORACLE_GATE dev_fallback_active={}",
        strict::dev_fallback_active()
    );
    let counting = Arc::new(oracle::CountingBackend::new(
        model.transformer.tensor_backend.clone(),
    ));
    model.transformer.tensor_backend = counting.clone();

    let steps = o.greedy_tokens.len();
    let (logits, fed) = drift::logits_per_step(
        &mut model.transformer,
        &o.prompt_tokens,
        steps,
        &o.greedy_tokens,
    );
    assert_eq!(fed, o.greedy_tokens);
    assert_eq!(logits.len(), steps);

    // Step 0: the full logit vector against the oracle.
    let cmp0 = drift::compare(std::slice::from_ref(&o.last_token_logits), &logits[..1]);
    println!(
        "ORACLE_GATE step0 max_abs_dlogit={:.4} oracle_tok={} omega_tok={} oracle_margin={:.4} bound={}",
        cmp0[0].max_abs_dlogit,
        cmp0[0].ref_tok,
        cmp0[0].cand_tok,
        cmp0[0].ref_margin,
        drift::MAX_ABS_DLOGIT
    );
    let step0 = drift::judge(&cmp0);

    // Steps 0..15: argmax against the oracle's greedy tokens, near-tie rule from the frozen bound.
    let mut bad = Vec::new();
    let mut flips = 0usize;
    for (s, (l, &want)) in logits.iter().zip(&o.greedy_tokens).enumerate() {
        let (top, top_v, _, _) = drift::top2(l);
        let gap = top_v - l[want as usize];
        if top as u32 != want {
            println!("ORACLE_GATE flip step={s} oracle_tok={want} omega_tok={top} gap={gap:.4}");
            if gap.is_nan() || gap > 2.0 * drift::MAX_ABS_DLOGIT {
                bad.push(format!("step {s}: omega {top} vs oracle {want}, gap {gap}"));
            }
            flips += 1;
        }
    }
    let text = model.tokenizer.decode(&fed).unwrap_or_default();
    println!(
        "ORACLE_GATE teacher_forced steps={steps} flips={flips} oracle_text={:?} fed_text={text:?}",
        o.greedy_text
    );

    // Per-op accounting: every op that ran ran natively. "native" below is calls minus the
    // backend's counted fallbacks and reference runs, so it only means native on a backend that
    // counts them (OmegaGb10Backend); a CPU backend reports none and is refused by the receipt.
    let native_engine = backend_identity.contains("native Omega engine");
    println!(
        "ORACLE_GATE {} native_engine={native_engine}",
        counting.line()
    );
    let report = counting.op_report();
    println!("ORACLE_GATE {}", report.line());
    let mut op_problems = Vec::new();
    if !native_engine {
        op_problems.push(format!(
            "backend is not the native Omega engine: {backend_identity}"
        ));
    }
    for (op, calls, native) in counting.native_calls() {
        if calls != native {
            op_problems.push(format!("{}: {calls} calls, {native} native", op.name()));
        }
    }
    if !report.native_fallbacks.is_empty() || !report.reference_runs.is_empty() {
        op_problems.push(report.line());
    }
    for op in [
        TensorOp::Rmsnorm,
        TensorOp::ApplyRope,
        TensorOp::Swiglu,
        TensorOp::ComputeLogits,
    ] {
        if counting.calls(op) == 0 {
            op_problems.push(format!("{} never ran", op.name()));
        }
    }
    if counting.calls(TensorOp::MatmulVec) + counting.calls(TensorOp::MatmulBatch) == 0 {
        op_problems.push("no matmul ran".into());
    }
    if counting.calls(TensorOp::GqaAttention)
        + counting.calls(TensorOp::PagedAttention)
        + counting.calls(TensorOp::PagedAttentionBatch)
        == 0
    {
        op_problems.push("no attention ran".into());
    }

    let mut receipt = StrictModelReceipt {
        checkpoint_path: model_path.display().to_string(),
        checkpoint_sha256: model_sha,
        tokenizer_path: tokenizer_path.display().to_string(),
        tokenizer_sha256: tokenizer_sha,
        backend_identity,
        model_id: model.config.model_id.clone(),
        model_config: serde_json::to_value(&model.config)
            .unwrap_or(serde_json::json!({"model_id": model.config.model_id})),
        fallback_count: counting.fallback_count(),
        op_report: Some(report),
        dev_fallback_build: strict::dev_fallback_active(),
        verdict: String::new(),
    };
    let verdict = receipt.verify();
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .expect("write receipt");
    println!(
        "ORACLE_GATE receipt verdict: {} (receipt {})",
        receipt.verdict,
        receipt_path.display()
    );
    if let Err(e) = step0 {
        panic!("Omega step-0 logits disagree with the HF oracle: {e}");
    }
    assert!(
        bad.is_empty(),
        "argmax disagreements beyond the near-tie bound: {bad:?}"
    );
    assert!(
        op_problems.is_empty(),
        "per-op native accounting: {op_problems:?}"
    );
    verdict.unwrap_or_else(|e| panic!("{e}"));
    println!("ORACLE_GATE verdict: PASS");
}
