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

//! sc#294 (Qwen3 v4 review): greedy decoding must be OBSERVED, not only
//! documented. The real submission path the daemon uses (`submit_work` ->
//! scheduler -> `NativeTransformerBackend::execute_step`) reports, in
//! `CompletionEvent::Finished`, how the backend chose the tokens, counted in
//! the sampling branch it took. CPU reference tensors, micro model.
use aien_inference_abi::{
    DecodeObservation, MockInferenceBackend, ModelConfig, NativeTransformerBackend,
    ReferenceCpuBackend, SamplingParams, TransformerWeights,
};
use aien_kv_cache::{create_shared_kv_manager, SharedKvManager};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{ChannelCompletionSink, CompletionEvent, PromptHandle, SchedulerConfig};
use std::sync::Arc;

fn micro_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-micro-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        num_kv_heads: 2,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn spine() -> (AienRuntimeSpine, SharedKvManager) {
    let kv = create_shared_kv_manager(64, 16);
    let cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };
    (AienRuntimeSpine::new(64, cfg, kv.clone()), kv)
}

/// Runs one request to completion; returns the streamed tokens, the
/// `Finished` decoding report and the sequence id.
async fn run<B: aien_inference_abi::AienInferenceBackend>(
    spine: &mut AienRuntimeSpine,
    backend: &mut B,
    sampling: SamplingParams,
) -> (Vec<u32>, Option<DecodeObservation>, u64) {
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));
    let prompt: PromptHandle = Arc::from([1u32, 10, 25, 42].as_slice());
    let seq = spine
        .submit_work(prompt, sampling, 1, Some(sink_id))
        .unwrap();
    spine.run_until_complete(backend, 40).await.unwrap();
    let (mut tokens, mut decoding, mut finished) = (Vec::new(), None, false);
    while let Ok(event) = rx.try_recv() {
        match event {
            CompletionEvent::Token { token, .. } => tokens.push(token),
            CompletionEvent::Finished { decoding: d, .. } => {
                finished = true;
                decoding = d;
            }
            CompletionEvent::Error { message, .. } => panic!("{message}"),
        }
    }
    assert!(finished, "the request must finish");
    (tokens, decoding, seq)
}

fn native(kv: SharedKvManager) -> NativeTransformerBackend {
    NativeTransformerBackend::with_shared_kv_and_backend(
        TransformerWeights::reference_test_weights(&micro_config()),
        Arc::new(ReferenceCpuBackend::new()),
        kv,
    )
}

/// Every token the backend chose was streamed, plus at most the final one
/// (a stop token is chosen but not streamed).
fn assert_counts_cover(tokens: &[u32], chosen: u64) {
    let n = tokens.len() as u64;
    assert!(n >= 1, "the request produced no token");
    assert!(
        chosen == n || chosen == n + 1,
        "backend chose {chosen} tokens, {n} were streamed"
    );
}

#[tokio::test]
async fn greedy_request_reports_greedy_from_the_backend() {
    let (mut spine, kv) = spine();
    let mut backend = native(kv);
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 5,
        stop_token_ids: vec![],
    };
    let (tokens, d, _) = run(&mut spine, &mut backend, sampling).await;
    let d = d.expect("the native backend reports how it decoded");
    assert_eq!(d.mode(), "greedy", "{d:?}");
    assert_eq!(d.sampled_tokens, 0);
    assert_counts_cover(&tokens, d.greedy_tokens);
    assert_eq!(
        (d.temperature, d.top_p, d.seed_request_id),
        (None, None, None)
    );
}

#[tokio::test]
async fn sampled_request_reports_sampling_from_the_backend() {
    let (mut spine, kv) = spine();
    let mut backend = native(kv);
    let sampling = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 5,
        stop_token_ids: vec![],
    };
    let (tokens, d, seq) = run(&mut spine, &mut backend, sampling).await;
    let d = d.expect("the native backend reports how it decoded");
    assert_eq!(d.mode(), "sampled", "{d:?}");
    assert_eq!(d.greedy_tokens, 0);
    assert_counts_cover(&tokens, d.sampled_tokens);
    assert_eq!(d.temperature, Some(0.7));
    assert_eq!(d.top_p, Some(0.9));
    assert_eq!(d.seed_request_id, Some(seq));
}

/// A backend that does not observe its decoding makes no claim: None, never
/// a default "greedy".
#[tokio::test]
async fn a_backend_without_the_observation_makes_no_claim() {
    let (mut spine, _kv) = spine();
    let mut backend = MockInferenceBackend::new(1);
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 3,
        stop_token_ids: vec![],
    };
    let (_, d, _) = run(&mut spine, &mut backend, sampling).await;
    assert_eq!(d, None);
}

/// The record field: present with the backend's report, absent without one.
#[test]
fn record_field_matches_the_report_and_is_absent_without_one() {
    let id = aien_runtime::generation::ModelIdentity {
        model_sha256: "m".repeat(64),
        model_digest_kind: "file".into(),
        model_path: "/m".into(),
        tokenizer_sha256: "t".repeat(64),
        tokenizer_path: "/t".into(),
    };
    let report = DecodeObservation {
        greedy_tokens: 4,
        ..Default::default()
    };
    let ev = |d| aien_runtime::generation::TurnEvidence {
        prompt_ids: &[1, 2],
        output_ids: &[3, 4, 5],
        text: "abc",
        total_tokens: 5,
        finish_reason: "eos",
        request_id: 1,
        operation_id: 1,
        decoding: d,
        ops: None,
    };
    let start = aien_runtime::generation::DaemonStart(1);
    let with = aien_runtime::generation::build_record(&id, &ev(Some(&report)), start);
    assert_eq!(
        with["decoding"],
        serde_json::json!({"mode": "greedy", "greedy_tokens": 4, "sampled_tokens": 0})
    );
    let without = aien_runtime::generation::build_record(&id, &ev(None), start);
    assert!(without.get("decoding").is_none(), "{without}");
}
