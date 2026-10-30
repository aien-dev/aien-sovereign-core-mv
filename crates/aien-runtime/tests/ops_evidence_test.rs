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

//! sc#337: "no silent fallback" must be shown by what a run keeps, not by the
//! absence of a crash. The real submission path the daemon uses
//! (`submit_work` -> scheduler -> `NativeTransformerBackend::execute_step`)
//! reports, in `CompletionEvent::Finished`, the tensor backend and its op
//! counters; the generation record and the daemon's `OP_REPORT` log line carry
//! them. CPU reference tensors, micro model, no chip.
use aien_inference_abi::native_ops::{NativeOpMask, OpAccounting, TensorOp};
use aien_inference_abi::{
    MockInferenceBackend, ModelConfig, NativeTransformerBackend, OpEvidence, ReferenceCpuBackend,
    RopeParams, SamplingParams, TensorBackend, TransformerWeights,
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

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 3,
        stop_token_ids: vec![],
    }
}

/// Runs one request to completion; returns the `Finished` op evidence.
async fn run<B: aien_inference_abi::AienInferenceBackend>(
    backend_for: impl FnOnce(SharedKvManager) -> B,
) -> Option<OpEvidence> {
    let (mut spine, kv) = spine();
    let mut backend = backend_for(kv);
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));
    let prompt: PromptHandle = Arc::from([1u32, 10, 25, 42].as_slice());
    spine
        .submit_work(prompt, greedy(), 1, Some(sink_id))
        .unwrap();
    spine.run_until_complete(&mut backend, 40).await.unwrap();
    let mut ops = None;
    let mut finished = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            CompletionEvent::Finished { ops: o, .. } => {
                finished = true;
                ops = o;
            }
            CompletionEvent::Error { message, .. } => panic!("{message}"),
            CompletionEvent::Token { .. } => {}
        }
    }
    assert!(finished, "the request must finish");
    ops
}

fn native_with(
    tensors: Arc<dyn TensorBackend>,
) -> impl FnOnce(SharedKvManager) -> NativeTransformerBackend {
    move |kv| {
        NativeTransformerBackend::with_shared_kv_and_backend(
            TransformerWeights::reference_test_weights(&micro_config()),
            tensors,
            kv,
        )
    }
}

/// The reference CPU math, with every op claimed native and every `rmsnorm`
/// counted as a fallback: what a dev build (`AIEN_DEV_FALLBACK=1`) records
/// when a claimed-native op runs on the CPU. `record_reference` counts
/// without the strict check, so the test runs the same in any build.
struct ForcedFallback {
    cpu: ReferenceCpuBackend,
    acct: OpAccounting,
    /// sc#402: when Some, every `matmul_vec` counts as one chip call, the way the
    /// Omega GB10 backend counts `chip_calls`; None is a backend with no chip.
    chip_calls: Option<std::sync::atomic::AtomicU64>,
}

impl TensorBackend for ForcedFallback {
    fn name(&self) -> &'static str {
        "forced-fallback"
    }
    fn op_report(&self) -> aien_inference_abi::native_ops::OpReport {
        self.acct.report()
    }
    fn gpu_executions(&self) -> Option<u64> {
        self.chip_calls
            .as_ref()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.acct.record_reference(TensorOp::Rmsnorm);
        self.cpu.rmsnorm(out, x, weight, eps)
    }
    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        self.cpu.rmsnorm_heads(x, weight, head_dim, eps)
    }
    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope: &RopeParams,
    ) {
        self.cpu
            .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, rope)
    }
    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        if let Some(c) = &self.chip_calls {
            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.cpu.matmul_vec(out, x, weight, out_dim, in_dim)
    }
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        self.cpu.swiglu(out, gate, up)
    }
    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        self.cpu
            .compute_logits(logits, hidden, embed_weight, vocab_size, hidden_dim)
    }
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.cpu.gqa_attention(
            out,
            q,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
}

/// A production run on the CPU reference backend: the call reports the
/// backend by name and zero fallbacks.
#[tokio::test]
async fn a_production_run_reports_its_backend_and_zero_fallbacks() {
    let ops = run(native_with(Arc::new(ReferenceCpuBackend::new())))
        .await
        .expect("the native backend reports its op evidence");
    assert_eq!(ops.backend, "ReferenceCpuBackend");
    assert_eq!(ops.native_fallbacks, 0, "{ops:?}");
    assert!(ops.report.starts_with("OP_REPORT "), "{}", ops.report);
}

/// A forced fallback shows: a non-zero `native_fallbacks` and the op by name.
#[tokio::test]
async fn a_forced_fallback_shows_a_non_zero_count() {
    let tensors = Arc::new(ForcedFallback {
        cpu: ReferenceCpuBackend::new(),
        acct: OpAccounting::new(NativeOpMask::ALL),
        chip_calls: None,
    });
    let ops = run(native_with(tensors))
        .await
        .expect("the native backend reports its op evidence");
    assert_eq!(ops.backend, "forced-fallback");
    assert!(ops.native_fallbacks > 0, "{ops:?}");
    assert!(
        ops.report.contains("native_fallbacks=[rmsnorm:"),
        "{}",
        ops.report
    );
    let line = aien_runtime::generation::op_report_line(Some(&ops)).unwrap();
    assert!(
        line.starts_with("OP_REPORT ") && line.ends_with(" backend=forced-fallback"),
        "{line}"
    );
}

/// A backend that does not account its ops makes no claim: None, never zero.
#[tokio::test]
async fn a_backend_without_op_accounting_makes_no_claim() {
    assert_eq!(run(|_| MockInferenceBackend::new(1)).await, None);
    assert_eq!(aien_runtime::generation::op_report_line(None), None);
}

/// The record field: present with the backend's evidence, absent without it.
#[test]
fn record_field_matches_the_evidence_and_is_absent_without_it() {
    let id = aien_runtime::generation::ModelIdentity {
        model_sha256: "m".repeat(64),
        model_digest_kind: "file".into(),
        model_path: "/m".into(),
        tokenizer_sha256: "t".repeat(64),
        tokenizer_path: "/t".into(),
    };
    let ops = OpEvidence {
        backend: "omega-gb10".into(),
        native_fallbacks: 0,
        reference_runs: 7,
        report: "OP_REPORT native=[matmul_vec] reference=[rmsnorm] native_fallbacks=[] reference_runs=[rmsnorm:7]".into(),
        gpu_executions: None,
    };
    let ev = |o| aien_runtime::generation::TurnEvidence {
        prompt_ids: &[1, 2],
        output_ids: &[3, 4, 5],
        text: "abc",
        total_tokens: 5,
        finish_reason: "eos",
        request_id: 1,
        operation_id: 1,
        decoding: None,
        ops: o,
    };
    let start = aien_runtime::generation::DaemonStart(1);
    let with = aien_runtime::generation::build_record(&id, &ev(Some(&ops)), start);
    assert_eq!(
        with["ops"],
        serde_json::json!({
            "backend": "omega-gb10",
            "native_fallbacks": 0,
            "reference_runs": 7,
            "scope": "process",
            "report": ops.report,
        })
    );
    let without = aien_runtime::generation::build_record(&id, &ev(None), start);
    assert!(without.get("ops").is_none(), "{without}");
}

/// sc#402: a backend that counts chip calls reports them with the call
/// (`gpu_executions`, process total), in the record and in the log line; the
/// CPU reference backend has no chip and makes no claim (None, never zero).
#[tokio::test]
async fn chip_calls_are_reported_and_absent_without_a_chip() {
    let tensors = Arc::new(ForcedFallback {
        cpu: ReferenceCpuBackend::new(),
        acct: OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec])),
        chip_calls: Some(std::sync::atomic::AtomicU64::new(0)),
    });
    let ops = run(native_with(tensors.clone()))
        .await
        .expect("the native backend reports its op evidence");
    let counted = tensors.gpu_executions().unwrap();
    assert!(counted > 0, "the micro model runs matmul_vec");
    assert_eq!(ops.gpu_executions, Some(counted), "{ops:?}");
    assert_eq!(ops.native_fallbacks, 0, "{ops:?}");
    let line = aien_runtime::generation::op_report_line(Some(&ops)).unwrap();
    assert!(
        line.ends_with(&format!(
            " backend=forced-fallback gpu_executions={counted}"
        )),
        "{line}"
    );
    let json = serde_json::to_value(&ops).unwrap();
    assert_eq!(json["gpu_executions"], serde_json::json!(counted));

    let cpu = run(native_with(Arc::new(ReferenceCpuBackend::new())))
        .await
        .expect("the native backend reports its op evidence");
    assert_eq!(cpu.gpu_executions, None);
    let cpu_json = serde_json::to_value(&cpu).unwrap();
    assert!(cpu_json.get("gpu_executions").is_none(), "{cpu_json}");
    assert!(!aien_runtime::generation::op_report_line(Some(&cpu))
        .unwrap()
        .contains("gpu_executions"));
}

/// sc#402: the record carries `ops.gpu_executions` only when the evidence has it.
#[test]
fn record_carries_gpu_executions_only_when_counted() {
    let id = aien_runtime::generation::ModelIdentity {
        model_sha256: "m".repeat(64),
        model_digest_kind: "file".into(),
        model_path: "/m".into(),
        tokenizer_sha256: "t".repeat(64),
        tokenizer_path: "/t".into(),
    };
    let ops = OpEvidence {
        backend: "omega-gb10".into(),
        native_fallbacks: 0,
        reference_runs: 0,
        report: "OP_REPORT native=[matmul_vec] reference=[] native_fallbacks=[] reference_runs=[]"
            .into(),
        gpu_executions: Some(156),
    };
    let ev = aien_runtime::generation::TurnEvidence {
        prompt_ids: &[1, 2],
        output_ids: &[3, 4, 5],
        text: "abc",
        total_tokens: 5,
        finish_reason: "eos",
        request_id: 1,
        operation_id: 1,
        decoding: None,
        ops: Some(&ops),
    };
    let start = aien_runtime::generation::DaemonStart(1);
    let rec = aien_runtime::generation::build_record(&id, &ev, start);
    assert_eq!(rec["ops"]["gpu_executions"], serde_json::json!(156));
    assert_eq!(rec["ops"]["scope"], serde_json::json!("process"));
}

/// sc#384 (2026-10-11): the frozen OPEN-MODEL-QWEN3 v5 row `-OPR` parses the
/// daemon's OP_REPORT line as `... reference_runs=[..] backend=<rest>`. The chip
/// call count (sc#402) must come after `backend=`, or every qualification launch
/// fails that product row (v7 dry attempt 2 did).
#[test]
fn op_report_line_keeps_backend_right_after_reference_runs() {
    let ops = aien_abi_core::OpEvidence {
        report: "OP_REPORT native=[matmul_vec] reference=[] native_fallbacks=[] reference_runs=[]"
            .into(),
        backend: "OmegaGb10Backend (native Omega engine)".into(),
        gpu_executions: Some(7),
        ..Default::default()
    };
    let line = aien_runtime::generation::op_report_line(Some(&ops)).unwrap();
    // The v5 regex reads the four bracketed fields and then ` backend=` directly;
    // the same check without a regex dependency: the bracketed prefix is intact
    // and ` backend=` follows `reference_runs=[]` immediately.
    assert!(
        line.starts_with(
            "OP_REPORT native=[matmul_vec] reference=[] native_fallbacks=[] reference_runs=[] backend=OmegaGb10"
        ),
        "{line}"
    );
    assert!(
        !line[..line.find(" backend=").unwrap()].contains("gpu_executions"),
        "{line}"
    );
    assert!(line.ends_with(" gpu_executions=7"), "{line}");
}
