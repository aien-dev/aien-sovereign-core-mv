//! One prompt in, text and measured speed out: the public demo of Qwen3.8-27B on the Omega engine (ledger L182).
//!
//!   qwen38_say <checkpoint-dir> <prompt> [--gen N] [--receipt <out.json>]
//!
//! Loads the 4-bit store beside the checkpoint (`<checkpoint-dir>.mxfp4/aien-w4.store`, or
//! `AIEN_Q38_W4_STORE`), renders the prompt with the checkpoint's chat template, reads it one token
//! at a time, then decodes greedily until a stop token or N new tokens (default 128), printing the
//! text as it comes. Then the numbers, each named for what it measures:
//!
//! - tokens/s (wall clock): generated tokens over the wall time of the steps that made them.
//! - weight bytes per token: every layer projection plus the output head, read once per token
//!   (`Qwen38Omega::weight_bytes`). KV cache and recurrent state reads are not counted, so the
//!   GB/s derived from it is a floor on what the chip moved.
//! - share of the memory wall: that GB/s over 273 GB/s, the GB10's LPDDR5X peak (NVIDIA spec).
//! - chip time per token: median of the GPU's own stamps (first submit to last marker); the wall
//!   clock minus this is host time between tokens.
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::tokenizer::{eos_token_ids_from_json, ChatTokenizer};
use serde_json::json;

/// GB10 LPDDR5X peak, bytes per second (NVIDIA DGX Spark spec: 273 GB/s).
const WALL_BYTES_PER_S: f64 = 273e9;

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38_say: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: qwen38_say <checkpoint-dir> <prompt> [--gen N] [--receipt <out.json>]".into(),
        );
    }
    let dir = Path::new(&args[1]);
    let prompt = &args[2];
    let mut gen = 128usize;
    let mut receipt: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        let v = args
            .get(i + 1)
            .ok_or(format!("{} needs a value", args[i]))?;
        match args[i].as_str() {
            "--gen" => gen = v.parse().map_err(|_| format!("--gen {v}: not a number"))?,
            "--receipt" => receipt = Some(v.clone()),
            o => return Err(format!("unknown option {o}")),
        }
        i += 2;
    }
    if gen == 0 {
        return Err("--gen must be at least 1".into());
    }

    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    // What the checkpoint's chat_template.jinja renders for one user turn with enable_thinking=false and no system
    // message or tools (the shared ChatML renderer refuses this template because of its tool and reasoning branches).
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt.trim()
    );
    let ids = tk.encode(&text).map_err(|e| e.to_string())?;
    let gen_cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("generation_config.json"))
            .map_err(|e| format!("generation_config.json: {e}"))?,
    )
    .map_err(|e| format!("generation_config.json: {e}"))?;
    let mut stop =
        eos_token_ids_from_json(&gen_cfg).ok_or("generation_config.json: no eos_token_id")?;
    stop.extend(tk.token_to_id("<|im_end|>"));
    if ids.is_empty() {
        return Err("the prompt encodes to no tokens".into());
    }

    eprintln!("loading the 4-bit model onto the GPU ...");
    let t_open = Instant::now();
    let mut e = Qwen38Omega::open(dir, None, (ids.len() + gen + 1).next_multiple_of(256))?;
    let open_s = t_open.elapsed().as_secs_f64();
    if !e.w4_only() {
        return Err("no 4-bit store was loaded (the demo measures the 4-bit engine only)".into());
    }
    let weight_bytes = e.weight_bytes();

    let t_prefill = Instant::now();
    let mut next = 0u32;
    for &t in &ids {
        next = e.step(t)?.0;
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // `next` is the first answer token; each later step feeds the previous answer token.
    let mut out = vec![next];
    let mut shown = 0usize;
    let mut chip_ms = Vec::with_capacity(gen);
    let mut stdout = std::io::stdout();
    writeln!(stdout, "--- answer").ok();
    let t_gen = Instant::now();
    while out.len() < gen && !stop.contains(&next) {
        let (top, st) = e.step(next)?;
        chip_ms.push(st.chip_ns as f64 / 1e6);
        next = top;
        out.push(top);
        let s = tk.decode(&out).map_err(|e| e.to_string())?;
        // Print only whole characters: a partial UTF-8 sequence decodes as U+FFFD until complete.
        if s.len() > shown && !s.ends_with('\u{FFFD}') {
            write!(stdout, "{}", &s[shown..]).ok();
            stdout.flush().ok();
            shown = s.len();
        }
    }
    let gen_s = t_gen.elapsed().as_secs_f64();
    let s = tk.decode(&out).map_err(|e| e.to_string())?;
    if s.len() > shown {
        write!(stdout, "{}", &s[shown..]).ok();
    }
    writeln!(stdout).ok();

    let steps = chip_ms.len();
    if steps == 0 {
        return Err("the model stopped after one token: nothing to time".into());
    }
    let tok_s = steps as f64 / gen_s;
    let gbps = weight_bytes as f64 * tok_s / 1e9;
    let share = weight_bytes as f64 * tok_s / WALL_BYTES_PER_S;
    let mut sorted = chip_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let chip_med = sorted[steps / 2];
    let chip_tok_s = 1e3 / chip_med;

    println!("--- numbers (this run, this machine)");
    println!(
        "model load              {open_s:.1} s ({:.2} GB of 4-bit weights onto the GPU)",
        weight_bytes as f64 / 1e9
    );
    println!(
        "prompt                  {} tokens read in {prefill_s:.1} s",
        ids.len()
    );
    println!(
        "generation              {steps} tokens in {gen_s:.2} s = {tok_s:.2} tokens/s (wall clock)"
    );
    println!(
        "weight bytes per token  {:.2} GB -> {gbps:.1} GB/s = {:.1}% of the 273 GB/s memory wall",
        weight_bytes as f64 / 1e9,
        share * 100.0
    );
    println!("chip time per token     {chip_med:.2} ms median = {chip_tok_s:.2} tokens/s while the GPU works (the rest is host time between tokens)");

    if let Some(path) = receipt {
        let r = json!({
            "tool": "qwen38_say", "checkpoint": dir.display().to_string(),
            "w4_store": e.w4_store.as_ref().map(|p| p.display().to_string()),
            "prompt": prompt, "prompt_tokens": ids.len(), "gen_limit": gen,
            "answer": s, "answer_ids": out,
            "open_s": open_s, "prefill_s": prefill_s,
            "gen_steps": steps, "gen_s": gen_s, "tok_s_wall": tok_s,
            "weight_bytes_per_token": weight_bytes, "gb_s_floor": gbps,
            "wall_bytes_per_s": WALL_BYTES_PER_S, "share_of_wall": share,
            "chip_ms_median": chip_med, "chip_ms": chip_ms, "tok_s_chip_median": chip_tok_s,
        });
        let body = serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?;
        std::fs::write(&path, body).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}
