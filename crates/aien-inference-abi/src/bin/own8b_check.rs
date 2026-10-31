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

//! OWN-8B stage 2 check: a dense Qwen3 checkpoint (Qwen3-8B FP8, Qwen3-4B bf16 tied) on Omega
//! against the host reference path (design `~/handoffs/2026-10-15-own8b-design.md`, section 2.4).
//!
//!   own8b_check golden <checkpoint> <golden.json> [--new N] [--prompts 1,2,3,4,5]
//!       Host only: the same engine code with host GEMVs and the host bf16 KV pool (oracle A),
//!       greedy continuations of the 5 fixed prompts (4 chat prompts + a 512-token prompt),
//!       top-2 margins, the shard SHA256s. Recorded once, then sealed.
//!   own8b_check chip <checkpoint> <golden.json> <receipt-dir> [--new N] [--prompts 1,2,3,4,5]
//!                    [--layers 0,17,35] [--parity-prompts 1,2,3,4] [--max-context 4096]
//!                    [--speed-tokens 128] [--upload-mutant swap-kv|transpose-scales]
//!                    [--kernel-mutant N] [--attention-mutant] [--no-host] [--force-memory]
//!       Omega: loads the resident model; oracle A per layer (chip traces against the host
//!       engine teacher-forced on the golden tokens, cosine >= 0.9999 and p95_rel <= 2e-3 per
//!       traced layer, thresholds fixed by the design); greedy tokens against the golden
//!       exactly; prefill (multi-row) against decode (row by row) on the chip; decode tok/s and
//!       prefill tok/s. Mutant flags turn a run into a negative control: it must FAIL.
//!       Writes <receipt-dir>/run.json and log.txt; the shell seals SHA256SUMS.
//!
//! Exit: 0 PASS, 1 FAIL, 2 usage, 3 refused (memory).

use std::path::{Path, PathBuf};
use std::time::Instant;

use aien_inference_abi::qwen3_dense_omega::engine::{
    DenseLayerTrace, DenseUploadMutant, HostDenseUploader, MutantUploader, OmegaDenseUploader,
    Qwen3DenseOmegaEngine,
};
use aien_inference_abi::qwen3_dense_omega::{DenseUploader, DenseWeightFormat, Qwen3DenseConfig};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::GenerateParams;
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;
use sha2::{Digest, Sha256};

/// Oracle A thresholds (design 2.4 A), fixed before any run.
const COSINE_MIN: f64 = 0.9999;
const P95_REL_MAX: f32 = 2e-3;

const PROMPT_TEXTS: [(&str, &str); 4] = [
    ("p1-one-token", "Hello"),
    ("p2-short", "What is the capital of France? Answer in one sentence."),
    (
        "p3-medium",
        "Summarise the following in three bullet points.\n\nThe DGX Spark is a compact workstation built around the GB10 Grace Blackwell superchip. It pairs an Arm CPU with a Blackwell GPU over a coherent link, sharing one pool of unified memory, so a model that is loaded once can be read by both sides without copies. Engineers who write their own kernels for this chip have to think about launch overhead, about which memory is cached on the GPU side, and about how the host polls for completion. The practical outcome is that bandwidth per token, not compute, sets the speed of decoding.",
    ),
    (
        "p4-code",
        "Write a Rust function `fn is_prime(n: u64) -> bool` that uses trial division up to the square root, with a short doc comment. Then explain in two sentences why checking up to the square root is enough.",
    ),
];
/// The fifth prompt: the first 512 ids of a long text (prefill at width, design 2.4 B).
const LONG_PROMPT_TOKENS: usize = 512;

fn chatml(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn prompts(tok: &Tokenizer) -> Vec<(String, Vec<u32>)> {
    let mut out: Vec<(String, Vec<u32>)> = PROMPT_TEXTS
        .iter()
        .map(|(name, text)| {
            let ids = if *name == "p1-one-token" {
                tok.encode_with_special(text, false).expect("encode")
            } else {
                tok.encode_with_special(&chatml(text), false)
                    .expect("encode")
            };
            (name.to_string(), ids)
        })
        .collect();
    let long: String = (0..12)
        .map(|i| format!("Section {i}. {}\n", PROMPT_TEXTS[2].1))
        .collect();
    let mut ids = tok.encode_with_special(&long, false).expect("encode");
    assert!(
        ids.len() >= LONG_PROMPT_TOKENS,
        "long prompt has {} ids",
        ids.len()
    );
    ids.truncate(LONG_PROMPT_TOKENS);
    out.push(("p5-512".into(), ids));
    out
}

#[derive(Debug, Clone)]
struct Args {
    mode: String,
    checkpoint: PathBuf,
    golden: PathBuf,
    receipt: Option<PathBuf>,
    new_tokens: usize,
    prompt_ids: Vec<usize>,
    parity_prompts: Vec<usize>,
    layers: Option<Vec<usize>>,
    max_context: usize,
    speed_tokens: usize,
    upload_mutant: Option<DenseUploadMutant>,
    kernel_mutant: i32,
    attention_mutant: bool,
    no_host: bool,
    force_memory: bool,
}

fn usage() -> ! {
    eprintln!("{}", USAGE);
    std::process::exit(2)
}

const USAGE: &str = "usage: own8b_check golden <checkpoint> <golden.json> [--new N] [--prompts 1,2,..]\n       own8b_check chip <checkpoint> <golden.json> <receipt-dir> [flags, see the source header]";

fn list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|x| x.trim().parse().unwrap_or_else(|_| usage()))
        .collect()
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        mode: argv.first().cloned().unwrap_or_else(|| usage()),
        checkpoint: PathBuf::new(),
        golden: PathBuf::new(),
        receipt: None,
        new_tokens: 32,
        prompt_ids: vec![1, 2, 3, 4, 5],
        parity_prompts: vec![1, 2, 3, 4],
        layers: None,
        max_context: 4096,
        speed_tokens: 128,
        upload_mutant: None,
        kernel_mutant: 0,
        attention_mutant: false,
        no_host: false,
        force_memory: false,
    };
    let mut pos = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        let v = || argv.get(i + 1).cloned().unwrap_or_else(|| usage());
        match argv[i].as_str() {
            "--new" => a.new_tokens = v().parse().unwrap_or_else(|_| usage()),
            "--prompts" => a.prompt_ids = list(&v()),
            "--parity-prompts" => a.parity_prompts = list(&v()),
            "--layers" => a.layers = Some(list(&v())),
            "--max-context" => a.max_context = v().parse().unwrap_or_else(|_| usage()),
            "--speed-tokens" => a.speed_tokens = v().parse().unwrap_or_else(|_| usage()),
            "--upload-mutant" => {
                a.upload_mutant = Some(DenseUploadMutant::parse(&v()).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    usage()
                }))
            }
            "--kernel-mutant" => a.kernel_mutant = v().parse().unwrap_or_else(|_| usage()),
            "--attention-mutant" => {
                a.attention_mutant = true;
                i += 1;
                continue;
            }
            "--no-host" => {
                a.no_host = true;
                i += 1;
                continue;
            }
            "--force-memory" => {
                a.force_memory = true;
                i += 1;
                continue;
            }
            f if f.starts_with("--") => usage(),
            p => {
                pos.push(p.to_string());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    if pos.len() < 2 {
        usage();
    }
    a.checkpoint = PathBuf::from(&pos[0]);
    a.golden = PathBuf::from(&pos[1]);
    a.receipt = pos.get(2).map(PathBuf::from);
    a
}

fn top2_margin(logits: &[f32]) -> (u32, f32) {
    let (mut b1, mut b2) = ((0usize, f32::NEG_INFINITY), f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > b1.1 {
            b2 = b1.1;
            b1 = (i, v);
        } else if v > b2 {
            b2 = v;
        }
    }
    (b1.0 as u32, b1.1 - b2)
}

fn shard_digests(dir: &Path) -> Vec<serde_json::Value> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("checkpoint dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let mut f = std::fs::File::open(dir.join(&n)).expect("shard");
            let mut h = Sha256::new();
            std::io::copy(&mut f, &mut h).expect("hash shard");
            serde_json::json!({"file": n, "sha256": format!("{:x}", h.finalize())})
        })
        .collect()
}

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1e6)
}

fn config_json(c: &Qwen3DenseConfig) -> serde_json::Value {
    serde_json::json!({
        "format": format!("{:?}", c.format), "layers": c.layers, "hidden": c.hidden,
        "q_heads": c.q_heads, "kv_heads": c.kv_heads, "head_dim": c.head_dim,
        "intermediate": c.intermediate, "vocab": c.vocab, "rope_theta": c.rope_theta,
        "tie_word_embeddings": c.tie_word_embeddings, "eos": c.eos_token_ids,
        "resident_bytes": c.model_resident_bytes(),
    })
}

fn golden(a: &Args) {
    let tok = Tokenizer::from_file(a.checkpoint.join("tokenizer.json")).expect("tokenizer");
    let t0 = Instant::now();
    let (mut e, report) = Qwen3DenseOmegaEngine::load_host(
        &a.checkpoint,
        &HostDenseUploader,
        a.max_context,
        &mut |_| {},
    )
    .expect("host load");
    let config = e.config().clone();
    eprintln!(
        "host reference loaded in {:.0}s ({:.2} GB)",
        t0.elapsed().as_secs_f64(),
        report.resident_bytes as f64 / 1e9
    );
    let mut out = Vec::new();
    for (idx, (name, ids)) in prompts(&tok).into_iter().enumerate() {
        if !a.prompt_ids.contains(&(idx + 1)) {
            continue;
        }
        let t1 = Instant::now();
        e.reset();
        let mut hidden = e.step_batch(&ids).expect("prefill");
        let (mut tokens, mut margins) = (Vec::new(), Vec::new());
        for _ in 0..a.new_tokens {
            let (next, margin) = top2_margin(&e.logits(&hidden).expect("logits"));
            tokens.push(next);
            margins.push(margin);
            if config.eos_token_ids.contains(&next) {
                break;
            }
            hidden = e.step(next).expect("step");
        }
        let secs = t1.elapsed().as_secs_f64();
        let text = tok.decode(&tokens).unwrap_or_default();
        eprintln!(
            "{name}: {} prompt tokens, {} new in {secs:.0}s: {text:?}",
            ids.len(),
            tokens.len()
        );
        out.push(serde_json::json!({
            "name": name, "prompt_tokens": ids, "golden": tokens, "margins": margins,
            "text": text, "host_seconds": secs,
        }));
    }
    let doc = serde_json::json!({
        "checkpoint": a.checkpoint.display().to_string(),
        "shards": shard_digests(&a.checkpoint),
        "config": config_json(&config),
        "max_new": a.new_tokens,
        "oracle": "Qwen3DenseOmegaEngine on the host (HostDenseUploader: HostFp8Gemv / HostBf16Gemv, f32 sums in column order; HostPoolKv bf16 KV), greedy, stop on eos",
        "prompts": out,
    });
    std::fs::write(&a.golden, serde_json::to_string_pretty(&doc).unwrap()).expect("write golden");
    eprintln!("wrote {}", a.golden.display());
}

/// Per traced layer: worst cosine and p95_rel of x_out and of the attention output.
#[derive(Debug, Default, Clone, Copy)]
struct LayerWorst {
    positions: usize,
    x_cos: f64,
    x_p95: f32,
    attn_cos: f64,
    mlp_cos: f64,
}

fn compare_traces(chip: &[DenseLayerTrace], host: &[DenseLayerTrace], layer: usize) -> LayerWorst {
    let mut w = LayerWorst {
        positions: 0,
        x_cos: 1.0,
        x_p95: 0.0,
        attn_cos: 1.0,
        mlp_cos: 1.0,
    };
    for c in chip.iter().filter(|t| t.layer == layer) {
        let Some(h) = host.iter().find(|t| t.layer == layer && t.pos == c.pos) else {
            continue;
        };
        let x = parity_stats(&c.x_out, &h.x_out);
        w.positions += 1;
        w.x_cos = w
            .x_cos
            .min(if x.non_finite > 0 { f64::NAN } else { x.cosine });
        w.x_p95 = w.x_p95.max(x.p95_rel);
        w.attn_cos = w.attn_cos.min(parity_stats(&c.attn, &h.attn).cosine);
        w.mlp_cos = w.mlp_cos.min(parity_stats(&c.mlp_out, &h.mlp_out).cosine);
    }
    w
}

fn chip(a: &Args) {
    let receipt = a.receipt.clone().unwrap_or_else(|| usage());
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let config = Qwen3DenseConfig::from_checkpoint(&a.checkpoint).expect("config");
    let need_gb =
        config.model_resident_bytes() as f64 / 1e9 * if a.no_host { 1.2 } else { 2.4 } + 4.0;
    let avail = mem_available_gb();
    say(format!(
        "mem available before load: {avail:.1} GB, this run needs about {need_gb:.1} GB"
    ));
    if avail < need_gb && !a.force_memory {
        say("REFUSED: not enough memory available; --force-memory to override".into());
        std::process::exit(3);
    }
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&a.golden).expect("golden file"))
            .expect("golden json");
    let gprompts = golden["prompts"]
        .as_array()
        .expect("golden prompts")
        .clone();
    let tok = Tokenizer::from_file(a.checkpoint.join("tokenizer.json")).expect("tokenizer");
    let ours = prompts(&tok);
    for (i, gp) in gprompts.iter().enumerate() {
        let ids: Vec<u32> = serde_json::from_value(gp["prompt_tokens"].clone()).unwrap();
        let name = gp["name"].as_str().unwrap_or("");
        let mine = ours.iter().find(|p| p.0 == name).map(|p| &p.1);
        assert_eq!(
            mine,
            Some(&ids),
            "golden prompt {i} ({name}) is not this binary's prompt"
        );
    }
    let (bf16_native, fp8_native) = OmegaDenseUploader::native();
    say(format!(
        "omega native: bf16 GEMV {bf16_native}, FP8 GEMV {fp8_native}, FP8 GEMM {}",
        aien_omega_fp8::has_gemm()
    ));
    let layers = a
        .layers
        .clone()
        .unwrap_or_else(|| vec![0, config.layers / 2 - 1, config.layers - 1]);

    // --- load (chip) ---
    let t0 = Instant::now();
    let omega = OmegaDenseUploader;
    let mutant_up;
    let up: &dyn DenseUploader = match a.upload_mutant {
        Some(m) => {
            mutant_up = MutantUploader::new(&omega, m, &config);
            &mutant_up
        }
        None => &omega,
    };
    let (mut e, report) =
        Qwen3DenseOmegaEngine::load_omega(&a.checkpoint, up, a.max_context, &mut |_| {})
            .unwrap_or_else(|err| {
                println!("LOAD FAIL {err}");
                std::process::exit(1)
            });
    let load_s = t0.elapsed().as_secs_f64();
    e.attention_mutant = a.attention_mutant;
    if a.kernel_mutant != 0 {
        if config.format == DenseWeightFormat::Fp8Block128 {
            aien_omega_fp8::test_set_mutant(a.kernel_mutant);
            aien_omega_fp8::gemm_test_set_mutant(a.kernel_mutant);
        } else {
            aien_omega_bf16::test_set_mutant(a.kernel_mutant);
        }
    }
    say(format!(
        "LOAD {} format {:?} tied {} layers {} resident {:.3} GB uploads {} in {load_s:.1}s, kv {} resident {} context {}, controls: upload_mutant {:?} kernel_mutant {} attention_mutant {}",
        a.checkpoint.display(),
        config.format,
        config.tie_word_embeddings,
        config.layers,
        report.resident_bytes as f64 / 1e9,
        report.uploads,
        e.attention_path(),
        e.kv_resident(),
        e.max_context_tokens(),
        a.upload_mutant,
        a.kernel_mutant,
        a.attention_mutant,
    ));
    let mut pass = true;
    let mut parity_json = Vec::new();

    // --- oracle A: per-layer parity against the host engine, teacher-forced ---
    if !a.no_host {
        let t1 = Instant::now();
        let (mut host, _) = Qwen3DenseOmegaEngine::load_host(
            &a.checkpoint,
            &HostDenseUploader,
            a.max_context,
            &mut |_| {},
        )
        .expect("host load");
        say(format!(
            "host reference loaded in {:.1}s",
            t1.elapsed().as_secs_f64()
        ));
        e.trace_layers = layers.clone();
        host.trace_layers = layers.clone();
        let mut worst: Vec<LayerWorst> = Vec::new();
        for (idx, gp) in gprompts.iter().enumerate() {
            if !a.parity_prompts.contains(&(idx + 1)) {
                continue;
            }
            let ids: Vec<u32> = serde_json::from_value(gp["prompt_tokens"].clone()).unwrap();
            let gold: Vec<u32> = serde_json::from_value(gp["golden"].clone()).unwrap();
            let steps = &gold[..gold.len().saturating_sub(1).min(16)];
            for eng in [&mut e, &mut host] {
                eng.reset();
                eng.step_batch(&ids).expect("prefill");
                for &t in steps {
                    eng.step(t).expect("step");
                }
            }
            for (li, &l) in layers.iter().enumerate() {
                let w = compare_traces(&e.traces, &host.traces, l);
                if worst.len() <= li {
                    worst.push(w);
                } else {
                    let o = &mut worst[li];
                    o.positions += w.positions;
                    o.x_cos = o.x_cos.min(w.x_cos);
                    o.x_p95 = o.x_p95.max(w.x_p95);
                    o.attn_cos = o.attn_cos.min(w.attn_cos);
                    o.mlp_cos = o.mlp_cos.min(w.mlp_cos);
                }
            }
            e.traces.clear();
            host.traces.clear();
        }
        for (li, &l) in layers.iter().enumerate() {
            let w = worst[li];
            let ok = w.x_cos >= COSINE_MIN && w.x_p95 <= P95_REL_MAX && w.attn_cos >= COSINE_MIN;
            pass &= ok;
            say(format!(
                "PARITY layer {l} over {} positions: x_out cosine(min) {:.6} p95_rel(max) {:.2e}, attention cosine(min) {:.6}, mlp cosine(min) {:.6} -> {}",
                w.positions, w.x_cos, w.x_p95, w.attn_cos, w.mlp_cos,
                if ok { "PASS" } else { "FAIL" }
            ));
            parity_json.push(serde_json::json!({"layer": l, "positions": w.positions, "x_cosine_min": w.x_cos, "x_p95_rel_max": w.x_p95, "attn_cosine_min": w.attn_cos, "mlp_cosine_min": w.mlp_cos, "pass": ok}));
        }
        e.trace_layers.clear();
        drop(host);
    }

    // --- prefill (multi-row) against decode (row by row) on the chip ---
    let long = gprompts
        .iter()
        .find(|p| p["name"] == "p5-512")
        .or_else(|| gprompts.last())
        .expect("a golden prompt");
    let long_ids: Vec<u32> = serde_json::from_value(long["prompt_tokens"].clone()).unwrap();
    e.reset();
    let tp = Instant::now();
    let h_batch = e.step_batch(&long_ids).expect("prefill");
    let prefill_s = tp.elapsed().as_secs_f64();
    let prefill_timing = e.last_timing;
    e.reset();
    let mut h_rows = Vec::new();
    for &t in &long_ids {
        h_rows = e.step(t).expect("step");
    }
    let exact = h_batch
        .iter()
        .zip(&h_rows)
        .all(|(x, y)| x.to_bits() == y.to_bits());
    let ps = parity_stats(&h_batch, &h_rows);
    say(format!(
        "PREFILL {} tokens in {prefill_s:.3}s ({} launches, chip {:.1} ms), last hidden vs token-at-a-time: bit-equal {exact}, cosine {:.9}",
        long_ids.len(),
        prefill_timing.launches,
        prefill_timing.chip_ns as f64 / 1e6,
        ps.cosine
    ));

    // --- greedy tokens against the golden ---
    let mut tokens_json = Vec::new();
    let mut tokens_all = true;
    for (idx, gp) in gprompts.iter().enumerate() {
        if !a.prompt_ids.contains(&(idx + 1)) {
            continue;
        }
        let name = gp["name"].as_str().unwrap_or("?");
        let ids: Vec<u32> = serde_json::from_value(gp["prompt_tokens"].clone()).unwrap();
        let gold: Vec<u32> = serde_json::from_value(gp["golden"].clone()).unwrap();
        let margins: Vec<f32> = serde_json::from_value(gp["margins"].clone()).unwrap_or_default();
        e.reset();
        let mut got = Vec::new();
        let params = GenerateParams::greedy(gold.len(), Vec::new());
        e.generate(&ids, &params, &mut |t| {
            got.push(t);
            true
        })
        .expect("generate");
        let first_diff = got.iter().zip(&gold).position(|(x, y)| x != y);
        let same = first_diff.is_none() && got.len() == gold.len();
        tokens_all &= same;
        let detail = match first_diff {
            None => "identical".to_string(),
            Some(p) => format!(
                "first divergence at {p} (golden margin there {:.4})",
                margins.get(p).copied().unwrap_or(f32::NAN)
            ),
        };
        say(format!(
            "TOKENS {name}: {} of {} golden tokens, {detail} -> {}",
            got.iter().zip(&gold).take_while(|(x, y)| x == y).count(),
            gold.len(),
            if same { "PASS" } else { "FAIL" }
        ));
        tokens_json.push(serde_json::json!({"name": name, "golden": gold, "chip": got, "first_divergence": first_diff, "pass": same}));
    }
    pass &= tokens_all;

    // --- decode speed ---
    let p4 = gprompts
        .iter()
        .find(|p| p["name"] == "p4-code")
        .unwrap_or(&gprompts[0]);
    let ids: Vec<u32> = serde_json::from_value(p4["prompt_tokens"].clone()).unwrap();
    e.reset();
    let mut n = 0usize;
    let o = e
        .generate(
            &ids,
            &GenerateParams::greedy(a.speed_tokens, Vec::new()),
            &mut |_| {
                n += 1;
                true
            },
        )
        .expect("speed run");
    let dt = o.decode_timing;
    let steps = o.emitted_tokens.max(1) as f64;
    let tok_s = o.emitted_tokens as f64 / (o.decode_ms as f64 / 1e3);
    say(format!(
        "SPEED decode {} tokens in {} ms = {tok_s:.2} tok/s; per token: chip {:.2} ms, host {:.2} ms, attention {:.2} ms, mlp {:.2} ms, lm_head {:.2} ms, {:.0} launches; prefill {} tokens in {} ms",
        o.emitted_tokens,
        o.decode_ms,
        dt.chip_ns as f64 / 1e6 / steps,
        dt.host_ns as f64 / 1e6 / steps,
        dt.attention_ns as f64 / 1e6 / steps,
        dt.experts_ns as f64 / 1e6 / steps,
        dt.lm_head_ns as f64 / 1e6 / steps,
        dt.launches as f64 / steps,
        o.prompt_tokens,
        o.prefill_ms,
    ));

    let verdict = if pass { "PASS" } else { "FAIL" };
    say(format!(
        "SUMMARY {verdict} (parity {} layers, tokens {}, prefill bit-equal {exact})",
        parity_json.len(),
        if tokens_all { "identical" } else { "differ" }
    ));
    let doc = serde_json::json!({
        "checkpoint": a.checkpoint.display().to_string(),
        "golden": a.golden.display().to_string(),
        "config": config_json(&config),
        "controls": {"upload_mutant": format!("{:?}", a.upload_mutant), "kernel_mutant": a.kernel_mutant, "attention_mutant": a.attention_mutant},
        "thresholds": {"cosine_min": COSINE_MIN, "p95_rel_max": P95_REL_MAX},
        "load_seconds": load_s, "resident_bytes": report.resident_bytes,
        "parity": parity_json, "tokens": tokens_json,
        "prefill": {"tokens": long_ids.len(), "seconds": prefill_s, "launches": prefill_timing.launches, "chip_ns": prefill_timing.chip_ns, "bit_equal_to_decode": exact, "cosine": ps.cosine},
        "speed": {"tokens": o.emitted_tokens, "decode_ms": o.decode_ms, "tok_s": tok_s, "chip_ns": dt.chip_ns, "host_ns": dt.host_ns, "attention_ns": dt.attention_ns, "mlp_ns": dt.experts_ns, "lm_head_ns": dt.lm_head_ns, "launches": dt.launches, "prefill_ms": o.prefill_ms, "prompt_tokens": o.prompt_tokens},
        "verdict": verdict,
    });
    std::fs::write(
        receipt.join("run.json"),
        serde_json::to_string_pretty(&doc).unwrap(),
    )
    .expect("run.json");
    std::fs::write(receipt.join("log.txt"), log.join("\n") + "\n").expect("log");
    std::process::exit(if pass { 0 } else { 1 });
}

fn main() {
    let a = parse();
    match a.mode.as_str() {
        "golden" => golden(&a),
        "chip" => chip(&a),
        _ => usage(),
    }
}
