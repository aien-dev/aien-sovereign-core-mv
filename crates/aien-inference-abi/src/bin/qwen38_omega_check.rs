//! Qwen3.8-27B on the Omega engine against the host reference, over a model prefix (sc #642).
//!
//!   qwen38_omega_check <checkpoint-dir> <layers> <prompt.txt> <tokens> <out.json>
//!
//! Uploads the first `layers` layers (final norm and lm_head on top), feeds the prompt's first
//! `tokens` ids one decode step at a time through the one-list graph, and holds each step's
//! logits to the reference's (same prefix, one host pass, `Logits::All`): cosine, p95 relative
//! error and the greedy token. The negative control compares each step with the reference's NEXT
//! position, and with the reference ONE LAYER SHORTER (the last layer must matter); both must
//! disagree: a check that cannot fail proves nothing.
//!
//! PASS when every step has cosine >= 0.9999, the greedy tokens agree wherever the reference's
//! top-two margin is at least 0.1, and the control is caught.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Config, Qwen38Reference};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::json;

const COSINE_MIN: f64 = 0.9999;
const MARGIN_MIN: f32 = 0.1;

fn margin(logits: &[f32]) -> f32 {
    let top = argmax(logits);
    let second = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != top)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    logits[top] - second
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let usage =
        "usage: qwen38_omega_check <checkpoint-dir> <layers> <prompt.txt> <tokens> <out.json>";
    if args.len() != 6 {
        return Err(usage.into());
    }
    let dir = Path::new(&args[1]);
    let layers: usize = args[2].parse().map_err(|_| usage.to_string())?;
    let count: usize = args[4].parse().map_err(|_| usage.to_string())?;
    let out_path = &args[5];
    let text = std::fs::read_to_string(&args[3]).map_err(|e| format!("{}: {e}", args[3]))?;
    let tokenizer =
        ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut ids = tokenizer.encode(&text).map_err(|e| e.to_string())?;
    ids.truncate(count);
    if ids.len() < 2 {
        return Err("the prompt needs at least two tokens".into());
    }
    let t = ids.len();
    let config = Qwen38Config::from_checkpoint(dir)?
        .prefix(layers)
        .ok_or_else(|| format!("layers {layers} outside the model"))?;
    let attn_layers = config.full_attention.iter().filter(|&&a| a).count();
    eprintln!(
        "prefix: {layers} layers ({} DeltaNet, {attn_layers} attention), {t} tokens",
        layers - attn_layers
    );

    let t0 = Instant::now();
    let reference = {
        let mut r = Qwen38Reference::new(config.clone(), ShardSource::open(dir)?)?;
        let mut state = r.new_state();
        r.forward(&mut state, &ids, Logits::All)?
    };
    let short = match config.prefix(layers - 1) {
        Some(c) => {
            let mut r = Qwen38Reference::new(c, ShardSource::open(dir)?)?;
            let mut state = r.new_state();
            Some(r.forward(&mut state, &ids, Logits::All)?)
        }
        None => None,
    };
    let ref_s = t0.elapsed().as_secs_f64();
    eprintln!("reference: {t} positions in {ref_s:.1} s");
    let vocab = config.vocab;

    let t1 = Instant::now();
    let mut engine = Qwen38Omega::load(config, &mut ShardSource::open(dir)?, t.max(256))?;
    let load_s = t1.elapsed().as_secs_f64();
    eprintln!(
        "engine: uploaded in {load_s:.1} s, weights {} MB, KV {} MB",
        engine.weight_bytes() >> 20,
        engine.kv_bytes() >> 20
    );

    let mut steps = Vec::with_capacity(t);
    let mut gpu_logits = Vec::with_capacity(t);
    let (mut cos_min, mut p95_max, mut greedy_ok) = (f64::INFINITY, 0f32, true);
    let mut chip_ns = Vec::with_capacity(t);
    for (p, &id) in ids.iter().enumerate() {
        let (token, st) = engine.step(id)?;
        let logits = engine.logits()?;
        let expected = &reference[p * vocab..(p + 1) * vocab];
        let s = parity_stats(&logits, expected);
        let (ref_tok, m) = (argmax(expected), margin(expected));
        let agree = token as usize == ref_tok;
        let counts = m >= MARGIN_MIN;
        if counts && !agree {
            greedy_ok = false;
        }
        if s.non_finite > 0 {
            cos_min = f64::NEG_INFINITY;
        }
        cos_min = cos_min.min(s.cosine);
        p95_max = p95_max.max(s.p95_rel);
        chip_ns.push(st.chip_ns);
        println!(
            "step {p:3}: token {token:6} ref {ref_tok:6} margin {m:7.3} cosine {:.6} p95_rel {:.4} max_abs {:.4} chip {:.2} ms launches {} submits {}{}",
            s.cosine,
            s.p95_rel,
            s.max_abs,
            st.chip_ns as f64 / 1e6,
            st.launches,
            st.submits,
            if agree { "" } else if counts { "  GREEDY DIFFERS" } else { "  (differs, below margin)" }
        );
        steps.push(json!({"pos": p, "token": token, "ref_token": ref_tok, "ref_margin": m,
            "cosine": s.cosine, "p95_rel": s.p95_rel, "max_abs": s.max_abs, "non_finite": s.non_finite,
            "chip_ns": st.chip_ns, "launches": st.launches, "submits": st.submits}));
        gpu_logits.push(logits);
    }

    // Negative control: step p against the reference's position p + 1.
    let control_cos = (0..t - 1)
        .map(|p| parity_stats(&gpu_logits[p], &reference[(p + 1) * vocab..(p + 2) * vocab]).cosine)
        .fold(f64::NEG_INFINITY, f64::max);
    let short_cos = short.as_ref().map(|s| {
        (0..t)
            .map(|p| parity_stats(&gpu_logits[p], &s[p * vocab..(p + 1) * vocab]).cosine)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let caught = control_cos < COSINE_MIN && short_cos.is_none_or(|c| c < COSINE_MIN);
    println!(
        "CONTROL shifted-by-one: best cosine {control_cos:.6}; one layer short: best cosine {} -> {}",
        short_cos.map_or("n/a (1 layer)".to_string(), |c| format!("{c:.6}")),
        if caught { "caught" } else { "NOT CAUGHT" }
    );

    let mut sorted = chip_ns.clone();
    sorted.sort_unstable();
    let median_ms = sorted[sorted.len() / 2] as f64 / 1e6;
    let pass = cos_min >= COSINE_MIN && greedy_ok && caught;
    println!(
        "QWEN38_OMEGA_PARITY layers {layers} tokens {t}: cosine(min) {cos_min:.6} p95_rel(max) {p95_max:.4} greedy {} control {} chip median {median_ms:.2} ms/token -> {}",
        if greedy_ok { "agrees" } else { "DIFFERS" },
        if caught { "caught" } else { "NOT CAUGHT" },
        if pass { "PASS" } else { "FAIL" }
    );
    let report = json!({
        "checkpoint": dir.display().to_string(), "layers": layers, "attention_layers": attn_layers,
        "tokens": t, "ids": ids, "reference_s": ref_s, "upload_s": load_s,
        "weight_bytes": engine.weight_bytes(), "kv_bytes": engine.kv_bytes(),
        "cosine_min": cos_min, "p95_rel_max": p95_max, "greedy_agrees": greedy_ok,
        "control_best_cosine": control_cos, "control_short_best_cosine": short_cos, "control_caught": caught,
        "chip_median_ms": median_ms, "thresholds": {"cosine_min": COSINE_MIN, "margin_min": MARGIN_MIN},
        "steps": steps, "pass": pass,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("{out_path}: {e}"))?;
    Ok(pass)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_omega_check: {e}");
            std::process::exit(2);
        }
    }
}
