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

//! Qwen3.8-27B host reference check (plan `~/handoffs/2026-10-18-qwen38-27b-on-omega.md`, step 2).
//!
//!   qwen38_check score    <checkpoint-dir> <passage.txt> <max-tokens> <out.json>
//!   qwen38_check generate <checkpoint-dir> <prompt.txt> <new-tokens> <out.json>
//!
//! Host only (no chip). `score` runs the passage's first `max-tokens` tokens through the
//! reference in one pass and reports teacher-forced perplexity and next-token accuracy (wrong
//! maths shows as a perplexity far above a few units). `generate` continues the prompt greedily
//! and writes the ids and each step's top five logits: the golden tokens later stages are held
//! to. Every pass streams the whole checkpoint from disk, so a generated token costs one pass.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, negative_log_likelihood, Logits, Qwen38Reference};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::json;

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.into_iter().take(5).map(|i| (i, logits[i])).collect()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage =
        "usage: qwen38_check score|generate <checkpoint-dir> <text-file> <count> <out.json>";
    if args.len() != 6 {
        return Err(usage.into());
    }
    let (mode, dir, text_path, out_path) = (&args[1], Path::new(&args[2]), &args[3], &args[5]);
    let count: usize = args[4].parse().map_err(|_| usage.to_string())?;
    let mut text = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;
    if mode == "score" {
        // The tokenizer refuses more than 2048 tokens; a token is at least one character, so
        // eight characters per wanted token leaves room and the ids are cut to `count` below.
        if let Some((cut, _)) = text.char_indices().nth(count * 8) {
            text.truncate(cut);
        }
    }
    let tokenizer =
        ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut ids = tokenizer.encode(&text).map_err(|e| e.to_string())?;

    let t0 = Instant::now();
    let mut model = Qwen38Reference::open(dir)?;
    let open_s = t0.elapsed().as_secs_f64();
    eprintln!("opened in {open_s:.1} s (embedding, final norm, lm_head)");
    let vocab = model.config.vocab;
    let mut state = model.new_state();

    let report = match mode.as_str() {
        "score" => {
            ids.truncate(count);
            if ids.len() < 2 {
                return Err("score: the passage needs at least two tokens".into());
            }
            let t1 = Instant::now();
            let logits = model.forward(&mut state, &ids, Logits::All)?;
            let pass_s = t1.elapsed().as_secs_f64();
            let mut nll = Vec::with_capacity(ids.len() - 1);
            let mut hits = 0usize;
            for (i, row) in logits.chunks_exact(vocab).take(ids.len() - 1).enumerate() {
                let next = ids[i + 1] as usize;
                nll.push(negative_log_likelihood(row, next));
                hits += usize::from(argmax(row) == next);
            }
            let mean = nll.iter().sum::<f64>() / nll.len() as f64;
            let ppl = mean.exp();
            let acc = hits as f64 / nll.len() as f64;
            eprintln!(
                "scored {} tokens in {pass_s:.1} s: perplexity {ppl:.3}, next-token accuracy {acc:.3}",
                ids.len()
            );
            json!({
                "mode": "score", "passage": text_path, "tokens": ids.len(), "ids": ids,
                "mean_nll": mean, "perplexity": ppl, "next_token_accuracy": acc,
                "nll": nll, "pass_seconds": pass_s, "open_seconds": open_s,
            })
        }
        "generate" => {
            let prompt = ids.clone();
            let mut steps = Vec::new();
            let t1 = Instant::now();
            let mut logits = model.forward(&mut state, &prompt, Logits::Last)?;
            let prefill_s = t1.elapsed().as_secs_f64();
            let mut out = Vec::new();
            let mut step_s = Vec::new();
            for step in 0..count {
                let next = argmax(&logits) as u32;
                steps.push(top5(&logits));
                out.push(next);
                let piece = tokenizer.decode(&[next]).unwrap_or_default();
                eprintln!("step {step}: {next} {piece:?}");
                if model.config.eos_token_ids.contains(&next) || step + 1 == count {
                    break;
                }
                let t2 = Instant::now();
                logits = model.forward(&mut state, &[next], Logits::Last)?;
                step_s.push(t2.elapsed().as_secs_f64());
            }
            let text_out = tokenizer.decode(&out).map_err(|e| e.to_string())?;
            eprintln!("continuation: {text_out:?}");
            json!({
                "mode": "generate", "prompt_file": text_path, "prompt_ids": prompt,
                "generated_ids": out, "generated_text": text_out, "top5": steps,
                "prefill_seconds": prefill_s, "step_seconds": step_s, "open_seconds": open_s,
            })
        }
        _ => return Err(usage.into()),
    };
    std::fs::write(
        out_path,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{out_path}: {e}"))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38_check: {e}");
        std::process::exit(1);
    }
}
