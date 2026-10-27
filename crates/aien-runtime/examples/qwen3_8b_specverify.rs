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

//! Qwen3-8B draft-reuse speculative verification, development measurement (sc#384 speed lane,
//! preregistered in ~/workspace/investigations/2026-10-12-qwen3-8b-decode-speed/plan.md,
//! decision 14:30Z 2026-10-12). NOT a campaign run; nothing is promoted by it.
//!
//! The question: when the corrective second attempt of a document task regenerates the whole
//! answer, can the rejected first draft be reused as a free drafter (propose short stretches of
//! it, verify them with the unchanged model in one row-batched pass) without changing a single
//! output token, and does that save enough time to matter?
//!
//! M1 baseline: plain greedy decode of the real v8-style corrective prompt for task D1 (goal
//!     text fixed below, refused reply and reason read from the v7 run receipt).
//! M2 draft reuse: the same prompt, greedy, with blocks of B = 8, 4, 2 draft tokens proposed from
//!     the v7 refused reply (suffix match of the last three emitted tokens), verified by
//!     `NativeTransformerBackend::verify_block`; rejected K/V dropped by `truncate_kv`.
//! M3 primitive cost: one verify_block call of 2, 4, 8 and 16 rows against one plain decode step,
//!     at the baseline's final context.
//!
//! usage: qwen3_8b_specverify <model_dir> <v7 run receipt json> <cpu|gb10>
//! Prints ONE JSON object on stdout (ids, timings, counts, VmHWM). Exit 5 when the GPU backend
//! reports any CPU fallback (the leg is then not a chip measurement).
use aien_inference_abi::{sample_argmax, ChatTokenizer, TensorBackend};
use aien_runtime::shared_kv::{build_shared_kv_runtime, SharedKvSizing};
use std::sync::Arc;
use std::time::Instant;

const SYSTEM: &str = "You are a careful assistant.";
/// tasks-oq3-v5.json D1, verbatim.
const GOAL: &str = "Please create the file docs/knife-sharpening.md, a short how-to guide on sharpening a kitchen knife by hand. It should have at least 22 non-empty lines and three sections titled \"Gather Your Tools\", \"Sharpen the Edge\" and \"Test and Store\". Make sure the words whetstone and angle both appear somewhere in it.";
const MAX_TOKENS: usize = 1024;
const BLOCKS: &[usize] = &[8, 4, 2];
const SUFFIX: usize = 3;

fn die(code: i32, msg: &str) -> ! {
    eprintln!("qwen3_8b_specverify: {msg}");
    std::process::exit(code)
}

fn vm_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// Where the draft continues after `out`: the position just past the LAST place in `draft`
/// whose preceding `SUFFIX` tokens equal the last `SUFFIX` emitted tokens.
fn propose(draft: &[u32], out: &[u32], want: usize) -> Option<Vec<u32>> {
    let k = out.len().min(SUFFIX);
    if k == 0 || want == 0 {
        return None;
    }
    let tail = &out[out.len() - k..];
    let mut j = None;
    for i in k..draft.len() {
        if &draft[i - k..i] == tail {
            j = Some(i);
        }
    }
    let j = j?;
    let end = (j + want).min(draft.len());
    if end <= j {
        return None;
    }
    Some(draft[j..end].to_vec())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 || !matches!(a[3].as_str(), "cpu" | "gb10") {
        die(
            2,
            "usage: qwen3_8b_specverify <model_dir> <v7 run receipt json> <cpu|gb10>",
        );
    }
    let dir = std::path::PathBuf::from(&a[1]);
    let gb10 = a[3] == "gb10";
    let receipt: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&a[2]).unwrap_or_else(|e| die(2, &format!("receipt: {e}"))),
    )
    .unwrap_or_else(|e| die(2, &format!("receipt json: {e}")));
    let att = &receipt["runs"][0]["proposal_attempts"][0];
    let refused_text = att["text"]
        .as_str()
        .unwrap_or_else(|| die(2, "receipt .text"));
    let reason = att["reason"]
        .as_str()
        .unwrap_or_else(|| die(2, "receipt .reason"));
    let draft: Vec<u32> = att["token_ids"]
        .as_array()
        .unwrap_or_else(|| die(2, "receipt .token_ids"))
        .iter()
        .map(|v| v.as_u64().unwrap_or_else(|| die(2, "token id")) as u32)
        .collect();

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
        }
        if let Some(us) = spin_us {
            aien_omega_gpu::set_spin_us(us)
                .unwrap_or_else(|e| die(3, &format!("spin window {us} us: {e}")));
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
        total_blocks: 1536,
    };
    let (_spine, mut backend) =
        build_shared_kv_runtime(weights, tb.clone(), cfg, sizing).expect("shared KV");
    let stop = tok.stop_token_ids().to_vec();
    let hwm_loaded = vm_hwm_kb();

    // The prompt: exactly what the production second attempt sends (compose/bridge.rs, corrective path).
    let user = aien_runtime::compose::rederive::corrective_retry_prompt(
        &aien_runtime::compose::rederive::proposal_prompt(GOAL),
        reason,
        Some(refused_text),
    );
    let msgs = vec![
        aien_runtime::ChatTurn {
            role: "system".into(),
            content: SYSTEM.into(),
        },
        aien_runtime::ChatTurn {
            role: "user".into(),
            content: user.clone(),
        },
    ];
    let text = aien_runtime::format_chat(tok.template(), &msgs).expect("chat");
    let ids = tok.encode(&text).expect("encode");
    eprintln!(
        "  prompt {} tokens, draft {} tokens",
        ids.len(),
        draft.len()
    );

    // Warm-up (one token, discarded), same as serving.
    let wm = vec![aien_runtime::ChatTurn {
        role: "user".into(),
        content: aien_runtime::WARM_UP_TEXT.into(),
    }];
    let wids = tok
        .encode(&aien_runtime::format_chat(tok.template(), &wm).expect("chat"))
        .expect("encode");
    backend
        .generate_tokens(1, &wids, 1, 0.0, &stop)
        .expect("warm-up");

    // Tie check (AIEN_SPECVERIFY_TIE=<pos>, AIEN_SPECVERIFY_BASE=<earlier receipt>): no runs; the
    // prompt plus the earlier baseline's first `pos` tokens are read, then the next position is
    // scored by the plain one-row path and by the batched path at 1, 3, 5 and 9 rows. Prints the
    // top two token ids and scores of each, so a divergence between the paths can be told apart
    // from a fault: a near-tie flipped by summation order shows as a tiny gap.
    if let Ok(pos) = std::env::var("AIEN_SPECVERIFY_TIE") {
        let pos: usize = pos
            .parse()
            .unwrap_or_else(|_| die(2, "AIEN_SPECVERIFY_TIE=<pos>"));
        let prev: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("AIEN_SPECVERIFY_BASE").unwrap_or_default())
                .unwrap_or_else(|e| die(2, &format!("AIEN_SPECVERIFY_BASE: {e}"))),
        )
        .unwrap_or_else(|e| die(2, &format!("base json: {e}")));
        let base: Vec<u32> = prev["baseline"]["output_ids"]
            .as_array()
            .unwrap_or_else(|| die(2, "base .baseline.output_ids"))
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if pos + 9 > base.len() {
            die(2, "pos too close to the end of the baseline");
        }
        let top2 = |row: &[f32]| -> serde_json::Value {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            serde_json::json!({ "top": [idx[0], idx[1]], "scores": [row[idx[0]], row[idx[1]]],
                                "gap": row[idx[0]] - row[idx[1]] })
        };
        let seq = 400u64;
        let mut ctx_ids = ids.clone();
        ctx_ids.extend_from_slice(&base[..pos]);
        backend
            .prefill_sequence(seq, &ctx_ids)
            .unwrap_or_else(|e| die(4, &format!("tie prefill: {e}")));
        let offset = ctx_ids.len();
        let mut rows_out = Vec::new();
        backend
            .sequences
            .get_mut(&seq)
            .expect("seq")
            .tokens
            .push(base[pos]);
        let l1 = backend
            .decode_pending(seq)
            .unwrap_or_else(|e| die(4, &format!("tie decode: {e}")));
        backend.sequences.get_mut(&seq).expect("seq").tokens.pop();
        backend.truncate_kv(seq, offset).unwrap();
        rows_out.push(serde_json::json!({ "path": "plain m=1", "row0": top2(&l1) }));
        let vocab = l1.len();
        for &rows in &[1usize, 3, 5, 9] {
            let l = backend
                .verify_block_logits(seq, &base[pos..pos + rows])
                .unwrap_or_else(|e| die(4, &format!("tie verify {rows}: {e}")));
            backend.truncate_kv(seq, offset).unwrap();
            rows_out.push(serde_json::json!({ "path": format!("verify rows={rows}"), "row0": top2(&l[..vocab]) }));
        }
        backend.release_sequence(seq);
        println!(
            "{}",
            serde_json::json!({ "tie_check": { "position": pos, "expected_next": base[pos + 1],
                "context": offset, "paths": rows_out }, "fallback_count": tb.fallback_count() })
        );
        return;
    }

    // M1: plain greedy baseline.
    tb.reset_op_timing();
    let t0 = Instant::now();
    let base = backend
        .generate_tokens(101, &ids, MAX_TOKENS, 0.0, &stop)
        .unwrap_or_else(|e| die(4, &format!("baseline: {e}")));
    let base_ms = t0.elapsed().as_secs_f64() * 1e3;
    let base_timing = tb.op_timing_report();
    eprintln!("  M1 baseline: {} tokens in {:.0} ms", base.len(), base_ms);

    // M2: draft reuse, one run per block size.
    let mut spec_runs = Vec::new();
    for (i, &b) in BLOCKS.iter().enumerate() {
        let seq = 200 + i as u64;
        tb.reset_op_timing();
        let t0 = Instant::now();
        let logits = backend
            .prefill_sequence(seq, &ids)
            .unwrap_or_else(|e| die(4, &format!("spec B={b} prefill: {e}")));
        let (first, _) = sample_argmax(&logits);
        let mut out: Vec<u32> = Vec::new();
        let mut finished = stop.contains(&first);
        if !finished {
            out.push(first);
            backend
                .sequences
                .get_mut(&seq)
                .expect("seq")
                .tokens
                .push(first);
        }
        let (mut verify_calls, mut proposed, mut accepted, mut plain_steps) =
            (0usize, 0usize, 0usize, 0usize);
        let mut verify_ms: Vec<f64> = Vec::new();
        let mut plain_ms: Vec<f64> = Vec::new();
        while !finished && out.len() < MAX_TOKENS {
            let pending = *out.last().expect("pending");
            let want = b.min(MAX_TOKENS - out.len());
            let mut emitted: Vec<u32> = Vec::new();
            match propose(&draft, &out, want) {
                Some(d) => {
                    let mut block = Vec::with_capacity(d.len() + 1);
                    block.push(pending);
                    block.extend_from_slice(&d);
                    let offset = ids.len() + out.len() - 1;
                    let tv = Instant::now();
                    let preds = backend
                        .verify_block(seq, &block)
                        .unwrap_or_else(|e| die(4, &format!("spec B={b} verify: {e}")));
                    verify_ms.push(tv.elapsed().as_secs_f64() * 1e3);
                    verify_calls += 1;
                    proposed += d.len();
                    let mut acc = 0usize;
                    while acc < d.len() && preds[acc] == d[acc] {
                        acc += 1;
                    }
                    accepted += acc;
                    emitted.extend_from_slice(&d[..acc]);
                    emitted.push(preds[acc]);
                    backend
                        .truncate_kv(seq, offset + acc + 1)
                        .unwrap_or_else(|e| die(4, &format!("spec B={b} truncate: {e}")));
                }
                None => {
                    let tp = Instant::now();
                    let logits = backend
                        .decode_pending(seq)
                        .unwrap_or_else(|e| die(4, &format!("spec B={b} decode: {e}")));
                    plain_ms.push(tp.elapsed().as_secs_f64() * 1e3);
                    plain_steps += 1;
                    emitted.push(sample_argmax(&logits).0);
                }
            }
            // Same stop and cap rule as the plain generator: a stop token ends the reply unemitted.
            for t in emitted {
                if stop.contains(&t) || out.len() >= MAX_TOKENS {
                    finished = true;
                    break;
                }
                out.push(t);
                backend.sequences.get_mut(&seq).expect("seq").tokens.push(t);
            }
            if out.len() >= MAX_TOKENS {
                finished = true;
            }
        }
        // The verify K/V for emitted-but-not-forwarded tokens is already pruned; the pending
        // token's K/V is never computed (same as the plain path). Release the sequence.
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        backend.release_sequence(seq);
        let identical = out == base;
        let mean = |v: &[f64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        eprintln!(
            "  M2 B={b}: {} tokens in {:.0} ms ({:.2}x baseline), identical={identical}, verify {verify_calls} calls mean {:.1} ms, accepted {accepted}/{proposed}, plain {plain_steps} steps mean {:.1} ms",
            out.len(), ms, ms / base_ms, mean(&verify_ms), mean(&plain_ms)
        );
        spec_runs.push(serde_json::json!({
            "block": b, "rows": b + 1, "tokens": out.len(), "ms": ms, "ratio_to_baseline": ms / base_ms,
            "identical_to_baseline": identical,
            "first_divergence": out.iter().zip(base.iter()).position(|(x, y)| x != y),
            "verify_calls": verify_calls, "proposed": proposed, "accepted": accepted,
            "verify_ms_mean": mean(&verify_ms), "plain_steps": plain_steps, "plain_ms_mean": mean(&plain_ms),
            "output_ids": out, "op_timing": tb.op_timing_report(),
        }));
    }

    // M3: primitive cost at the baseline's final context (prompt + all but the last 17 tokens
    // re-read as one prompt, then the next tokens as the block), each size three times.
    let mut m3 = Vec::new();
    if base.len() > 20 {
        let seq = 300u64;
        let cut = base.len() - 17;
        let mut ctx_ids = ids.clone();
        ctx_ids.extend_from_slice(&base[..cut]);
        backend
            .prefill_sequence(seq, &ctx_ids)
            .unwrap_or_else(|e| die(4, &format!("M3 prefill: {e}")));
        let offset = ctx_ids.len();
        for &rows in &[1usize, 2, 4, 8, 16] {
            let mut ms = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                if rows == 1 {
                    backend
                        .sequences
                        .get_mut(&seq)
                        .expect("seq")
                        .tokens
                        .push(base[cut]);
                    backend
                        .decode_pending(seq)
                        .unwrap_or_else(|e| die(4, &format!("M3 decode: {e}")));
                    backend.sequences.get_mut(&seq).expect("seq").tokens.pop();
                } else {
                    backend
                        .verify_block(seq, &base[cut..cut + rows])
                        .unwrap_or_else(|e| die(4, &format!("M3 verify {rows}: {e}")));
                }
                ms.push(t.elapsed().as_secs_f64() * 1e3);
                backend
                    .truncate_kv(seq, offset)
                    .unwrap_or_else(|e| die(4, &format!("M3 truncate: {e}")));
            }
            eprintln!(
                "  M3 rows={rows} at ctx {offset}: {:?} ms",
                ms.iter().map(|v| v.round()).collect::<Vec<_>>()
            );
            m3.push(serde_json::json!({ "rows": rows, "context": offset, "ms": ms }));
        }
        backend.release_sequence(seq);
    }
    let fallback = tb.fallback_count();
    println!(
        "{}",
        serde_json::json!({
            "leg": a[3], "backend": TensorBackend::name(tb.as_ref()),
            "cta_budget": cta_budget.map(|b| b.get()), "spin_us": spin_us,
            "model_id": config.model_id, "prompt_tokens": ids.len(), "draft_tokens": draft.len(),
            "goal": GOAL, "reason": reason,
            "baseline": { "tokens": base.len(), "ms": base_ms, "output_ids": base, "op_timing": base_timing,
                           "reply": tok.decode_opts(&base, true).unwrap_or_default() },
            "spec_runs": spec_runs, "m3_primitive_cost": m3,
            "vm_hwm_kb_after_load": hwm_loaded, "vm_hwm_kb_end": vm_hwm_kb(),
            "fallback_count": fallback,
        })
    );
    if fallback > 0 {
        die(
            5,
            &format!("{fallback} CPU fallback(s): not a chip measurement"),
        );
    }
}
