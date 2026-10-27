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

//! PREFILL-E2E C7: measure the TinyLlama noise floor that branch parity will face.
//!
//! Tolerance rule this measurement feeds (no number is fixed here):
//! - `docs/campaigns/c1/acceptance.toml:21-22`: tolerance = noise_multiplier_k (2.0) x measured floor.
//! - `docs/campaigns/c1/acceptance.toml:23-24`: relative error = |a - b| / max(|ref|, 1.0e-3).
//! - `docs/campaigns/c1/acceptance.toml:31-32`: tinyllama floor is `status = "unmeasured"`; a later
//!   cut freezes the number from the forge log of this test.
//! - aien-architecture b4fce15 `docs/plans/prefill/PREFILL-3-SPEC.md:127-131` (floor = maximum
//!   observed over mathematically equal summation orders, reported with its run count) and
//!   `docs/plans/prefill/PREFILL-0-SPEC.md:142,145` (G6: measured, not assumed, recorded).
//!
//! What is measured, on the CPU reference backend (`ReferenceCpuBackend`, no CUDA), per prompt:
//! - (a) one-shot prefill of prompt + k teacher-forced tokens, last-position logits;
//! - (b) prefill(prompt) then k single-token forward steps (teacher forced), last-position logits;
//! - (c) repeat-run determinism: (a) and (b) repeated `runs` times on fresh sequences.
//!
//! Floor = max over prompts and runs of |a_r - a_0|, |b_r - a_0| (abs and rel, reference a_0).
//! Prompts are kept at <= 120 tokens total so the chunked-prefill continuation defect (Cut 3,
//! 128-token daemon chunks) cannot touch this measurement: both paths here are single-call.
//!
//! The ignored test asserts only sanity (finite, non-constant logits, expected vocab width).
//! Optional `AIEN_NOISE_FLOOR_EXPECT_MIN_ABS=<x>` makes it also assert max_abs >= x; the forge
//! uses it with a perturbation mutant to prove the measurement sees a 1e-2 change.
//!
//! Run: AIEN_E2E_CHECKPOINT=$HOME/models/TinyLlama-1.1B-Chat-v1.0 \
//!   cargo test --release -p aien-inference-abi --test tinyllama_noise_floor -- --ignored --nocapture

use aien_inference_abi::tokenizer::TinyLlamaTokenizer;
use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::weights::TransformerWeights;
use aien_inference_abi::ModelConfig;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};

/// `docs/campaigns/c1/acceptance.toml:24`.
const RELATIVE_ERROR_EPSILON: f64 = 1.0e-3;
/// Teacher-forced tokens per prompt (brief: k >= 8).
const K_TEACHER: usize = 8;
/// Upper bound on prompt + teacher tokens (brief: <= 120, below the 128-token chunk size).
const MAX_TOTAL_TOKENS: usize = 120;
const DEFAULT_RUNS: usize = 3;

/// Fixed prompts and teacher continuations (R = 5). Token ids come from the in-repo
/// tokenizer fixture `crates/aien-inference-abi/tests/fixtures/tokenizer.json` via
/// `TinyLlamaTokenizer` (`src/tokenizer.rs:52,96`).
const PROMPTS: [(&str, &str); 5] = [
    (
        "The capital of France is",
        " Paris, which is also the largest city in the country.",
    ),
    (
        "Water boils at sea level when it reaches a temperature of",
        " one hundred degrees Celsius, or two hundred twelve Fahrenheit.",
    ),
    (
        "In a small village near the mountains, an old clockmaker repaired",
        " every watch that people brought to his narrow wooden shop.",
    ),
    (
        "To compute the area of a circle, multiply pi by",
        " the square of the radius, then round the result.",
    ),
    (
        "def add(a, b):\n    return",
        " a + b\n\ndef sub(a, b):\n    return a - b\n",
    ),
];

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Diff {
    max_abs: f64,
    max_rel: f64,
}

impl Diff {
    fn merge(self, other: Diff) -> Diff {
        Diff {
            max_abs: self.max_abs.max(other.max_abs),
            max_rel: self.max_rel.max(other.max_rel),
        }
    }
}

/// Element-wise difference of `other` against `reference`, relative error per
/// `acceptance.toml:23`: |a - b| / max(|ref|, epsilon).
fn compare(reference: &[f32], other: &[f32], epsilon: f64) -> Diff {
    assert_eq!(
        reference.len(),
        other.len(),
        "logit vectors differ in length"
    );
    let mut d = Diff::default();
    for (&r, &o) in reference.iter().zip(other.iter()) {
        let abs = (r as f64 - o as f64).abs();
        assert!(abs.is_finite(), "non-finite logit difference");
        let rel = abs / (r as f64).abs().max(epsilon);
        d.max_abs = d.max_abs.max(abs);
        d.max_rel = d.max_rel.max(rel);
    }
    d
}

// `!(x >= min)` on purpose: a NaN floor must fail the expected-minimum check.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn check_expected_min(max_abs: f64, expected_min: Option<f64>) -> Result<(), String> {
    match expected_min {
        Some(min) if !(max_abs >= min) => Err(format!(
            "NOISE_FLOOR max_abs={:.6e} is below expected minimum {:.6e}",
            max_abs, min
        )),
        _ => Ok(()),
    }
}

fn assert_sane(logits: &[f32], vocab: usize, what: &str) {
    assert_eq!(logits.len(), vocab, "{what}: logits width != vocab");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "{what}: non-finite logit"
    );
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max > min, "{what}: logits are constant ({min})");
}

fn sha256_file(path: &Path) -> String {
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open {} for hashing: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let mut hex = String::with_capacity(64);
    for b in hasher.finalize().iter() {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

fn resolve_checkpoint() -> PathBuf {
    let raw = std::env::var("AIEN_E2E_CHECKPOINT").unwrap_or_else(|_| {
        panic!(
            "AIEN_E2E_CHECKPOINT is unset: this measurement needs the real TinyLlama checkpoint \
             (directory with model.safetensors, or the file itself). Refusing to fall back."
        )
    });
    let p = PathBuf::from(raw);
    let file = if p.is_dir() {
        p.join("model.safetensors")
    } else {
        p
    };
    assert!(
        file.is_file(),
        "AIEN_E2E_CHECKPOINT does not resolve to a file: {}",
        file.display()
    );
    file
}

/// Path (a): one-shot prefill on a fresh sequence, last-position logits.
fn one_shot(be: &mut NativeTransformerBackend, seq_id: u64, tokens: &[u32]) -> Vec<f32> {
    let logits = be
        .prefill_sequence(seq_id, tokens)
        .unwrap_or_else(|e| panic!("one-shot prefill failed: {e}"));
    be.release_sequence(seq_id);
    logits
}

/// Path (b): prefill(prompt), then one forward step per teacher token at position
/// `tokens.len()` (same convention as `append_branch_token`, `transformer_backend.rs:286`).
fn incremental(
    be: &mut NativeTransformerBackend,
    seq_id: u64,
    prompt: &[u32],
    teacher: &[u32],
) -> Vec<f32> {
    let mut logits = be
        .prefill_sequence(seq_id, prompt)
        .unwrap_or_else(|e| panic!("prompt prefill failed: {e}"));
    for &tok in teacher {
        let NativeTransformerBackend {
            weights,
            sequences,
            tensor_backend,
            kv_manager,
            ..
        } = &mut *be;
        let seq = sequences
            .get_mut(&seq_id)
            .expect("sequence vanished after prefill");
        let pos = seq.tokens.len();
        let hidden = NativeTransformerBackend::forward_token_impl_paged(
            weights,
            &**tensor_backend,
            tok,
            pos,
            seq,
            seq_id,
            kv_manager.as_ref(),
        );
        seq.tokens.push(tok);
        logits = NativeTransformerBackend::compute_logits_impl(weights, &**tensor_backend, &hidden);
    }
    be.release_sequence(seq_id);
    logits
}

#[test]
#[ignore = "needs AIEN_E2E_CHECKPOINT (real TinyLlama); run by the forge with --ignored"]
fn tinyllama_noise_floor_cpu_reference() {
    let model_path = resolve_checkpoint();
    let runs: usize = std::env::var("AIEN_NOISE_FLOOR_RUNS")
        .ok()
        .map(|v| v.parse().expect("AIEN_NOISE_FLOOR_RUNS must be an integer"))
        .unwrap_or(DEFAULT_RUNS);
    assert!(runs >= 2, "need at least 2 runs for a repeat-run floor");
    let expected_min: Option<f64> =
        std::env::var("AIEN_NOISE_FLOOR_EXPECT_MIN_ABS")
            .ok()
            .map(|v| {
                v.parse()
                    .expect("AIEN_NOISE_FLOOR_EXPECT_MIN_ABS must be a number")
            });

    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer.json");
    let tokenizer = TinyLlamaTokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|e| panic!("in-repo tokenizer {}: {e}", tokenizer_path.display()));

    let model_sha256 = sha256_file(&model_path);
    let config = ModelConfig::tinyllama_1_1b();
    let vocab = config.vocab_size;
    let weights = TransformerWeights::load_from_safetensors(&model_path, &config)
        .unwrap_or_else(|e| panic!("load {}: {e}", model_path.display()));
    // CPU reference backend only (no CUDA, no AIEN_REQUIRE_BLACKWELL).
    let mut be = NativeTransformerBackend::new_reference(weights);
    let backend_name = be.tensor_backend.name();
    assert_eq!(backend_name, "ReferenceCpuBackend");

    let mut floor = Diff::default();
    let mut next_id: u64 = 0xC7_0000;
    for (idx, (prompt_text, cont_text)) in PROMPTS.iter().enumerate() {
        let prompt = tokenizer
            .encode_with_special(prompt_text, true)
            .expect("encode prompt");
        let cont = tokenizer
            .encode_with_special(cont_text, false)
            .expect("encode continuation");
        assert!(
            cont.len() >= K_TEACHER,
            "prompt {idx}: continuation has {} tokens, need {K_TEACHER}",
            cont.len()
        );
        let teacher = &cont[..K_TEACHER];
        let mut full = prompt.clone();
        full.extend_from_slice(teacher);
        assert!(
            prompt.len() >= 2 && full.len() <= MAX_TOTAL_TOKENS,
            "prompt {idx}: {} total tokens outside [2, {MAX_TOTAL_TOKENS}]",
            full.len()
        );

        let mut reference: Option<Vec<f32>> = None;
        let mut ab = Diff::default();
        let mut repeat = Diff::default();
        for run in 0..runs {
            next_id += 1;
            let a = one_shot(&mut be, next_id, &full);
            next_id += 1;
            let b = incremental(&mut be, next_id, &prompt, teacher);
            assert_sane(&a, vocab, &format!("prompt {idx} run {run} path a"));
            assert_sane(&b, vocab, &format!("prompt {idx} run {run} path b"));
            let a0: &[f32] = reference.get_or_insert_with(|| a.clone());
            repeat = repeat.merge(compare(a0, &a, RELATIVE_ERROR_EPSILON));
            ab = ab.merge(compare(a0, &b, RELATIVE_ERROR_EPSILON));
        }
        println!(
            "NOISE_FLOOR_PROMPT idx={idx} prompt_tokens={} k={K_TEACHER} ab_max_abs={:.6e} ab_max_rel={:.6e} repeat_max_abs={:.6e} repeat_max_rel={:.6e}",
            prompt.len(),
            ab.max_abs,
            ab.max_rel,
            repeat.max_abs,
            repeat.max_rel
        );
        floor = floor.merge(ab).merge(repeat);
    }

    println!(
        "NOISE_FLOOR_META tokenizer=tests/fixtures/tokenizer.json tokenizer_sha256={} relative_error_epsilon={RELATIVE_ERROR_EPSILON:e}",
        sha256_file(&tokenizer_path)
    );
    println!(
        "NOISE_FLOOR max_abs={:.6e} max_rel={:.6e} runs={runs} prompts={} k={K_TEACHER} backend={backend_name} model_sha256={model_sha256}",
        floor.max_abs,
        floor.max_rel,
        PROMPTS.len()
    );

    if let Err(e) = check_expected_min(floor.max_abs, expected_min) {
        panic!("{e}");
    }
}

#[test]
fn compare_sees_injected_perturbation() {
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 4.0).collect();
    assert_eq!(compare(&a, &a, RELATIVE_ERROR_EPSILON), Diff::default());
    let mut b = a.clone();
    b[17] += 1e-2;
    let d = compare(&a, &b, RELATIVE_ERROR_EPSILON);
    assert!((d.max_abs - 1e-2).abs() < 1e-5, "max_abs {}", d.max_abs);
    // a[16] == 0.0: relative error must use the epsilon denominator, not divide by zero.
    let mut c = a.clone();
    c[16] += 1e-4;
    let dc = compare(&a, &c, RELATIVE_ERROR_EPSILON);
    assert!((dc.max_rel - 0.1).abs() < 1e-3, "max_rel {}", dc.max_rel);
}

#[test]
fn expected_minimum_rejects_small_floor() {
    assert!(check_expected_min(5e-3, Some(1e-2)).is_err());
    assert!(check_expected_min(f64::NAN, Some(1e-2)).is_err());
    assert!(check_expected_min(1e-2, Some(1e-2)).is_ok());
    assert!(check_expected_min(0.0, None).is_ok());
}

#[test]
fn sanity_rejects_constant_and_nonfinite_logits() {
    let flat = vec![1.0f32; 8];
    assert!(std::panic::catch_unwind(|| assert_sane(&flat, 8, "flat")).is_err());
    let mut nan = vec![0.0f32, 1.0, 2.0, 3.0];
    nan[2] = f32::NAN;
    assert!(std::panic::catch_unwind(|| assert_sane(&nan, 4, "nan")).is_err());
    assert!(std::panic::catch_unwind(|| assert_sane(&[0.0, 1.0], 2, "ok")).is_ok());
}
