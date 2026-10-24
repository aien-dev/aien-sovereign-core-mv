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

//! Qwen3-8B correctness smoke driver (docs/campaigns/open-model-qwen3/SMOKE-8B-v1.md, sc#384).
//! Two fixed conversations (PROMPTS), thinking off, greedy, on the daemon's path (strict
//! safetensors load, shared-KV runtime, `ChatTokenizer::from_model_dir`, one warm-up turn).
//! Prints ONE JSON object: prompt text and ids, output ids, finish reason, fallback_count.
//!
//! usage: qwen3_8b_smoke <model_dir> <cpu|gb10>
//!   cpu  = `ReferenceCpuBackend`, host f32 weights (never touches the GPU).
//!   gb10 = the daemon's GB10 construction (session open with retry, resident bf16 load through
//!          `OmegaUploader`, `OmegaGb10Backend`). Refuses by name when the engine is not linked,
//!          the build is not production-strict, or the model is refused; never falls back.
//!          UNVERIFIED: compiled but not run on the chip when this file was written.
use aien_inference_abi::{ChatTokenizer, SamplingParams, TensorBackend};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use aien_scheduler::{ChannelCompletionSink, CompletionEvent};
use sha2::Digest;
use std::sync::Arc;

const SYSTEM: &str = "You are a careful assistant.";
const GEN_PROMPT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
/// (name, user text, max new tokens). Prompt 1 is the original one-token check; prompt 2 must reply with
/// at least 16 tokens before eos (SMOKE-8B-v1.md).
const PROMPTS: &[(&str, &str, usize)] = &[
    ("p1", "Reply with the single word OK.", 16),
    (
        "p2",
        "List the months of the year, one per line, nothing else.",
        48,
    ),
];

async fn turn(
    spine: &mut aien_runtime::AienRuntimeSpine,
    backend: &mut aien_inference_abi::NativeTransformerBackend,
    ids: &[u32],
    max_tokens: usize,
    stop: &[u32],
) -> Result<(Vec<u32>, String, f64), String> {
    let (sink, mut ev) = ChannelCompletionSink::channel();
    let sid = spine.register_completion_sink(Arc::new(sink));
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 0.95,
        max_tokens,
        stop_token_ids: stop.to_vec(),
    };
    spine.submit_work(Arc::from(ids), sampling, 2, Some(sid))?;
    let t0 = std::time::Instant::now();
    let mut out = Vec::new();
    loop {
        if spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0 {
            spine.step(backend).await.map_err(|e| format!("{e:?}"))?;
        }
        while let Ok(e) = ev.try_recv() {
            match e {
                CompletionEvent::Token { token, .. } => out.push(token),
                CompletionEvent::Finished { finish_reason, .. } => {
                    let label = aien_runtime::server::finish_reason_label(&finish_reason);
                    return Ok((out, label.into(), t0.elapsed().as_secs_f64() * 1e3));
                }
                CompletionEvent::Error { message, .. } => return Err(message),
            }
        }
    }
}

fn die(code: i32, msg: &str) -> ! {
    eprintln!("qwen3_8b_smoke: {msg}");
    std::process::exit(code)
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 || !matches!(a[2].as_str(), "cpu" | "gb10") {
        die(2, "usage: qwen3_8b_smoke <model_dir> <cpu|gb10>");
    }
    let dir = std::path::PathBuf::from(&a[1]);
    let gb10 = a[2] == "gb10";
    // The matmul CTA budget the daemon serves with (AIEN_OMEGA_CTA_BUDGET, else the AIEN default),
    // applied before the first chip call; omega's own library default (64) is never what the smoke
    // measures. Set after the GPU session opens, below.
    let cta_budget: Option<aien_omega_gpu::CtaBudget> = if gb10 {
        let raw = std::env::var(aien_omega_gpu::CTA_BUDGET_ENV).ok();
        Some(
            aien_omega_gpu::parse_cta_budget(raw.as_deref())
                .unwrap_or_else(|e| die(2, &e))
                .unwrap_or(aien_omega_gpu::DAEMON_DEFAULT_CTA_BUDGET),
        )
    } else {
        None
    };
    // The daemon's marker-wait spin window as well (AIEN_OMEGA_SPIN_US, else the daemon default):
    // without it every launch pays the 50 us sleep-poll floor and the smoke does not time the daemon's path.
    let spin_us: Option<u32> = if gb10 {
        let raw = std::env::var(aien_omega_gpu::SPIN_US_ENV).ok();
        Some(
            aien_omega_gpu::parse_spin_us(raw.as_deref())
                .unwrap_or_else(|e| die(2, &e))
                .unwrap_or(aien_omega_gpu::DAEMON_DEFAULT_SPIN_US),
        )
    } else {
        None
    };
    // AIEN_SMOKE_LONG=1 adds a third, longer prompt (not part of the golden; for decode timing only).
    // AIEN_SMOKE_CTX=1 adds a fourth with a long prompt (32 numbered rules, about 900 tokens: the
    // campaign's document tasks decode at 700 to 1000 tokens of context) and 64 new tokens, so the
    // timing table shows what attention costs at that context. Not part of the golden.
    let long_user: String = {
        let rules: Vec<String> = (1..=32)
            .map(|i| {
                format!(
                    "Rule {i}: a book borrowed on day {} of the month is due back after {} days, and the fine for a late return is {} cents per day.",
                    i, i % 7 + 3, i * 3 % 50 + 5
                )
            })
            .collect();
        format!(
            "Here are the rules of a small library.\n{}\nSummarize these rules in two sentences.",
            rules.join("\n")
        )
    };
    let mut prompts: Vec<(&str, &str, usize)> = PROMPTS.to_vec();
    if std::env::var("AIEN_SMOKE_LONG")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        prompts.push((
            "p3",
            "Describe how a bicycle works, in about 250 words.",
            256,
        ));
    }
    if std::env::var("AIEN_SMOKE_CTX")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        prompts.push(("p4", long_user.as_str(), 64));
    }
    let config = aien_inference_abi::load_model_config(&dir).expect("config.json");
    let (weights, tb): (_, Arc<dyn TensorBackend>) = if gb10 {
        if !aien_inference_abi::strict::production_strict() {
            die(
                3,
                "gb10 leg needs a production-strict build (no dev fallback)",
            );
        }
        let omega = aien_inference_abi::OmegaGb10Backend::new();
        if !omega.is_available() {
            die(3, "the Omega GB10 engine is not linked (stub build)");
        }
        if let Some(why) = aien_inference_abi::omega_model_refusal(&config) {
            die(3, &why);
        }
        aien_inference_abi::open_gpu_session_with_retry(
            aien_inference_abi::GPU_SESSION_OPEN_ATTEMPTS,
            aien_inference_abi::GPU_SESSION_RETRY_DELAY,
            aien_inference_abi::GPU_SESSION_OPEN_DEADLINE,
            &aien_runtime::shared_kv::read_mem_available,
        )
        .unwrap_or_else(|e| die(3, &format!("GPU session: {e}")));
        if let Some(b) = cta_budget {
            aien_omega_gpu::set_cta_budget(b)
                .unwrap_or_else(|e| die(3, &format!("CTA budget {}: {e}", b.get())));
            eprintln!("  matmul CTA budget {} (daemon setting)", b.get());
        }
        if let Some(us) = spin_us {
            aien_omega_gpu::set_spin_us(us)
                .unwrap_or_else(|e| die(3, &format!("spin window {us} us: {e}")));
            eprintln!("  marker spin window {us} us (daemon setting)");
        }
        let weights = aien_inference_abi::load_resident_weights(
            &dir,
            &config,
            &aien_inference_abi::OmegaUploader,
            &mut |_| {},
        )
        .unwrap_or_else(|e| die(3, &format!("resident load: {e}")));
        (weights, Arc::new(omega))
    } else {
        let weights = aien_inference_abi::TransformerWeights::load_from_safetensors(&dir, &config)
            .expect("weights");
        (
            weights,
            Arc::new(aien_inference_abi::ReferenceCpuBackend::new()),
        )
    };
    let tok = ChatTokenizer::from_model_dir(&dir, None).expect("tokenizer");
    let cfg = aien_scheduler::SchedulerConfig {
        max_batch_size: 256,
        max_batch_tokens: 16384,
        max_prefill_tokens: 8192,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 64,
    };
    let sizing = SharedKvSizing {
        arena_capacity: 4096,
        // 24k tokens of KV. Every block id then passes the resident pool's 32-bit rule (block
        // 1819 is the edge for this geometry), so the paged attention reads the pool in place.
        total_blocks: 1536,
    };
    let (mut spine, mut backend) =
        build_shared_kv_runtime(weights, tb.clone(), cfg, sizing).expect("shared KV");
    let stop = tok.stop_token_ids().to_vec();
    let wm = vec![aien_runtime::ChatTurn {
        role: "user".into(),
        content: aien_runtime::WARM_UP_TEXT.into(),
    }];
    let wids = tok
        .encode(&aien_runtime::format_chat(tok.template(), &wm).expect("chat"))
        .expect("encode");
    turn(&mut spine, &mut backend, &wids, 1, &stop)
        .await
        .expect("warm-up");
    let sha = |b: &[u8]| -> String {
        sha2::Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    let mut results = Vec::new();
    let mut p3_timing: Option<String> = None;
    for (name, user, max_tokens) in prompts.iter() {
        // The timing table covers one prompt at a time (its prompt read and its decode steps).
        tb.reset_op_timing();
        let msgs = vec![
            aien_runtime::ChatTurn {
                role: "system".into(),
                content: SYSTEM.into(),
            },
            aien_runtime::ChatTurn {
                role: "user".into(),
                content: (*user).into(),
            },
        ];
        let text = aien_runtime::format_chat(tok.template(), &msgs).expect("chat");
        let expected = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{GEN_PROMPT}"
        );
        if text != expected {
            die(
                4,
                &format!("rendered prompt {name} differs from the golden text: {text:?}"),
            );
        }
        let ids = tok.encode(&text).expect("encode");
        let (out, finish, ms) = turn(&mut spine, &mut backend, &ids, *max_tokens, &stop)
            .await
            .expect("turn");
        let reply = tok.decode_opts(&out, true).expect("decode");
        let prompt_timing = tb.op_timing_report();
        if let Some(t) = &prompt_timing {
            eprintln!("  {name} op timing: {t}");
        }
        if *name == "p3" {
            p3_timing = prompt_timing.clone();
        }
        results.push(serde_json::json!({
            "name": name, "prompt_sha256": sha(text.as_bytes()), "prompt_ids": ids,
            "max_tokens": max_tokens, "output_ids": out, "finish": finish, "ms": ms,
            "reply": reply, "op_timing": prompt_timing,
        }));
    }
    // Per-operation wall and chip time for p3 (AIEN_SMOKE_LONG=1), from the GPU backend's table;
    // every prompt also carries its own table.
    let op_timing = p3_timing;
    println!(
        "{}",
        serde_json::json!({
            "leg": a[2], "backend": TensorBackend::name(tb.as_ref()),
            "cta_budget": cta_budget.map(|b| b.get()), "spin_us": spin_us, "op_timing": op_timing,
            "model_dir": dir, "model_id": config.model_id, "template": tok.template().name(),
            "prompts": results, "fallback_count": tb.fallback_count(),
        })
    );
}
