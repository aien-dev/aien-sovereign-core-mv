//! The whole Qwen3.8-27B (all 64 layers) on the Omega engine against the host reference, one
//! layer window at a time (sc #642, omega-mv #559).
//!
//!   qwen38_omega_full <checkpoint-dir> <window-layers> <golden.json> <out.json>
//!
//! `golden.json` is a `qwen38_check generate` record (`prompt_ids`, `generated_ids`: the host
//! reference's greedy continuation). The sequence is the prompt plus every generated token but the
//! last, fed one decode step at a time (teacher forcing with the reference's own tokens: if the chip
//! picks the reference's token at every step, free-running greedy decoding produces the same text).
//!
//! The device holds one window of `window-layers` layers (plus lm_head) at a time, because the
//! resident 30B server leaves less free memory than the whole 27B needs. Window 0 takes the token
//! embeddings; window w takes window w - 1's residual stream as the CHIP computed it (never the
//! reference's), so the last window's logits are a full 64-layer chip forward pass. Each window's
//! output residual is also held to the reference's residual at that layer boundary.
//!
//! PASS (the identity gate Drake set for the motor, 2026-10-21, L312) when every generated
//! position's chip token equals the golden token, except where the reference's own top-two margin
//! is below a band WRITTEN IN ADVANCE (`MARGIN_MIN`, 0.1 logit): two float orders cannot be asked
//! to agree on a decision the reference itself made by less than that, so such a flip is reported
//! as allowed, while a flip at a wider margin is a wrong kernel and fails. The band is never taken
//! from the run under test (2026-10-21 01:58Z: a band equal to the run's own max logit error let
//! the shared-expert mutant through with 22 flips at max error 23.8: a broken run excused itself).
//! The same band gates the chip's greedy choice at every position, prompt included. The logits'
//! cosine and max error are recorded at every step (the earlier 0.9999 cosine floor is an open
//! precision item, not a gate), and both controls must be caught: (1) each step against the
//! reference's NEXT position, (2) the final residual against the reference's residual one window
//! earlier (the last window must matter; n/a with one window). A check that cannot fail proves
//! nothing: the negative control (`AIEN_Q36_TEST_MUTANT`) must fail this rule.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Config, Qwen38Reference};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use serde_json::{json, Value};

const COSINE_MIN: f64 = 0.9999;
/// Free memory kept for everything else on the box (the resident 30B server shares this pool:
/// 2026-10-18 16:43:58Z a 32-layer window pushed the kernel into the OOM killer, which took the
/// server). A window is uploaded only when MemAvailable covers it plus this margin.
const KEEP_FREE: u64 = 12 << 30;

/// Device bytes a window takes: its FP8 weights (1 byte each; the 128x128 block scales add 1/4096),
/// the bf16 a|b rows of its DeltaNet layers, and the bf16 lm_head; from the config, not measured.
fn window_bytes(c: &Qwen38Config, first: usize, count: usize) -> u64 {
    let h = c.hidden as u64;
    // Qwen3.6-35B-A3B (sc L306): every layer holds 257 FP8 experts (256 routed + the shared one)
    // and a bf16 router of 257 rows.
    let mlp = match &c.moe {
        Some(m) => (m.experts as u64 + 1) * (3 * m.expert_inter as u64 * h + 2 * h),
        None => 3 * c.intermediate as u64 * h,
    };
    let o = h * (c.heads * c.head_dim) as u64;
    let layers: u64 = c.full_attention[first..first + count]
        .iter()
        .map(|&attn| {
            if attn {
                (2 * c.heads * c.head_dim + 2 * c.kv_dim()) as u64 * h + o + mlp
            } else {
                (c.conv_dim() + c.lin_value_dim()) as u64 * h
                    + o
                    + mlp
                    + 2 * 2 * c.lin_v_heads as u64 * h
            }
        })
        .sum();
    layers + layers / 4096 + 2 * c.vocab as u64 * h
}

/// MemAvailable from /proc/meminfo, in bytes.
fn mem_available() -> Result<u64, String> {
    let m = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    m.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb << 10)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}
/// The written band: the reference's top-two logit margin below which two float orders may
/// legitimately disagree. Fixed in advance, never taken from the run under test. Every correct run
/// held it (2026-10-21 p1..p4: 127/128 words identical, the one flip at margin 0.0096) and the
/// shared-expert mutant broke it (22 flips, smallest margin 0.214, 10 of them wider than even the
/// correct runs' largest vocabulary-wide logit error 2.425).
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

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("golden: no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("golden: {key} holds a non-id"))
        })
        .collect()
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let usage =
        "usage: qwen38_omega_full <checkpoint-dir> <window-layers> <golden.json> <out.json>";
    if args.len() != 5 {
        return Err(usage.into());
    }
    let dir = Path::new(&args[1]);
    let wl: usize = args[2].parse().map_err(|_| usage.to_string())?;
    let out_path = &args[4];
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(&args[3]).map_err(|e| format!("{}: {e}", args[3]))?,
    )
    .map_err(|e| format!("{}: {e}", args[3]))?;
    let prompt = ids(&golden, "prompt_ids")?;
    let generated = ids(&golden, "generated_ids")?;
    if prompt.is_empty() || generated.is_empty() {
        return Err("golden: empty prompt or continuation".into());
    }
    let mut seq = prompt.clone();
    seq.extend_from_slice(&generated[..generated.len() - 1]);
    let (t, first_gen) = (seq.len(), prompt.len() - 1);

    let config = Qwen38Config::from_checkpoint(dir)?;
    let layers = config.layers;
    if wl == 0 || layers % wl != 0 {
        return Err(format!("window {wl} does not divide {layers} layers"));
    }
    let windows = layers / wl;
    let (h, vocab) = (config.hidden, config.vocab);
    eprintln!(
        "model: {layers} layers in {windows} windows of {wl}; {t} positions ({} generated)",
        generated.len()
    );

    // The reference: logits at every position plus the residual at every window boundary.
    let t0 = Instant::now();
    let bounds: Vec<usize> = (1..=windows).map(|w| w * wl).collect();
    let mut taps = Vec::new();
    let reference = {
        let mut r = Qwen38Reference::new(config.clone(), ShardSource::open(dir)?)?;
        // L355: with AIEN_Q36_W4_STORE set the chip loads the 4-bit store, so the reference reads the same 4-bit bytes
        // (experts; trunk too with AIEN_Q36_W4=all): the gate then checks the chip's 4-bit maths, not the format's drift.
        if let Some(p) = aien_inference_abi::q36_w4_store::resolve(dir)? {
            let trunk = std::env::var("AIEN_Q36_W4").as_deref() == Ok("all");
            let store = aien_inference_abi::q36_w4_store::W4Store::open(&p, dir)?;
            r.set_w4(Some(aien_inference_abi::qwen38::W4Host { store: std::sync::Arc::new(store), experts: true, trunk, mutant: false }));
        }
        let mut state = r.new_state();
        r.forward_tapped(&mut state, &seq, Logits::All, &bounds, &mut taps)?
    };
    let ref_s = t0.elapsed().as_secs_f64();
    let ref_matches_golden = (0..generated.len()).all(|i| {
        argmax(&reference[(first_gen + i) * vocab..(first_gen + i + 1) * vocab])
            == generated[i] as usize
    });
    eprintln!(
        "reference: {t} positions in {ref_s:.1} s; its greedy tokens {} the golden record",
        if ref_matches_golden {
            "equal"
        } else {
            "DIFFER FROM"
        }
    );

    // The chip, window by window, each fed the previous window's chip residual.
    let mut x_in: Vec<Vec<f32>> = Vec::new();
    let mut gpu_logits: Vec<Vec<f32>> = Vec::new();
    let mut tokens = vec![0u32; t];
    let mut chip_ns = vec![0u64; t];
    let mut win_report = Vec::new();
    let mut upload_s = 0.0;
    let mut weight_bytes = 0usize;
    for (w, tap) in taps.iter().enumerate().take(windows) {
        let first = w * wl;
        let need = window_bytes(&config, first, wl);
        let avail = mem_available()?;
        if avail < need + KEEP_FREE {
            return Err(format!(
                "window {w}: needs {} MiB with {} MiB kept free, MemAvailable {} MiB; refusing (use a smaller window)",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let t1 = Instant::now();
        let mut engine = Qwen38Omega::open_window(dir, first, wl, t.max(256))?;
        let ls = t1.elapsed().as_secs_f64();
        upload_s += ls;
        weight_bytes += engine.weight_bytes();
        eprintln!(
            "window {w} (layers {first}..{}): uploaded in {ls:.1} s, weights {} MB, KV {} MB",
            first + wl - 1,
            engine.weight_bytes() >> 20,
            engine.kv_bytes() >> 20
        );
        let last = w + 1 == windows;
        let mut x_out = Vec::with_capacity(t);
        for p in 0..t {
            let (tok, st) = if w == 0 {
                engine.step(seq[p])?
            } else {
                engine.step_residual(&x_in[p])?
            };
            chip_ns[p] += st.chip_ns;
            x_out.push(engine.residual()?);
            if last {
                tokens[p] = tok;
                gpu_logits.push(engine.logits()?);
            }
        }

        let (mut cos_min, mut p95_max) = (f64::INFINITY, 0f32);
        for (p, x) in x_out.iter().enumerate() {
            let s = parity_stats(x, &tap[p * h..(p + 1) * h]);
            cos_min = cos_min.min(if s.non_finite > 0 {
                f64::NEG_INFINITY
            } else {
                s.cosine
            });
            p95_max = p95_max.max(s.p95_rel);
        }
        println!("window {w} layers {first}..{}: residual cosine(min) {cos_min:.6} p95_rel(max) {p95_max:.4}", first + wl - 1);
        win_report.push(json!({"window": w, "first_layer": first, "layers": wl, "upload_s": ls,
            "weight_bytes": engine.weight_bytes(), "residual_cosine_min": cos_min, "residual_p95_rel_max": p95_max}));
        x_in = x_out;
        drop(engine);
    }

    let mut steps = Vec::with_capacity(t);
    let (mut cos_min, mut p95_max, mut greedy_ok, mut golden_ok) =
        (f64::INFINITY, 0f32, true, true);
    let mut gen_tokens = Vec::new();
    // Golden flips inside the written margin band (allowed, reported) and beyond it (wrong kernel).
    let (mut flips_allowed, mut flips_beyond) = (Vec::new(), Vec::new());
    for p in 0..t {
        let logits = &gpu_logits[p];
        let expected = &reference[p * vocab..(p + 1) * vocab];
        let s = parity_stats(logits, expected);
        let (ref_tok, m) = (argmax(expected), margin(expected));
        let tok = tokens[p] as usize;
        let agree = tok == ref_tok && argmax(logits) == tok;
        if m >= MARGIN_MIN && !agree {
            greedy_ok = false;
        }
        let gold = (p >= first_gen).then(|| generated[p - first_gen] as usize);
        if let Some(g) = gold {
            gen_tokens.push(tok);
            if tok != g {
                golden_ok = false;
                let flip = json!({"pos": p, "token": tok, "golden": g, "ref_margin": m, "max_abs": s.max_abs});
                if m < MARGIN_MIN {
                    flips_allowed.push(flip);
                } else {
                    flips_beyond.push(flip);
                }
            }
        }
        cos_min = cos_min.min(if s.non_finite > 0 {
            f64::NEG_INFINITY
        } else {
            s.cosine
        });
        p95_max = p95_max.max(s.p95_rel);
        println!(
            "step {p:3}: token {tok:6} ref {ref_tok:6}{} margin {m:7.3} cosine {:.6} p95_rel {:.4} max_abs {:.4} chip {:.2} ms{}",
            gold.map_or(String::new(), |g| format!(" golden {g:6}")),
            s.cosine,
            s.p95_rel,
            s.max_abs,
            chip_ns[p] as f64 / 1e6,
            if gold.is_some_and(|g| g != tok) { "  GOLDEN DIFFERS" } else if agree { "" } else { "  (differs)" }
        );
        steps.push(json!({"pos": p, "token": tok, "ref_token": ref_tok, "golden": gold, "ref_margin": m,
            "cosine": s.cosine, "p95_rel": s.p95_rel, "max_abs": s.max_abs, "non_finite": s.non_finite,
            "chip_ns_all_windows": chip_ns[p]}));
    }

    // Controls.
    let shift_cos = (0..t - 1)
        .map(|p| parity_stats(&gpu_logits[p], &reference[(p + 1) * vocab..(p + 2) * vocab]).cosine)
        .fold(f64::NEG_INFINITY, f64::max);
    let short_cos = (windows >= 2).then(|| {
        let tap = &taps[windows - 2];
        (0..t)
            .map(|p| parity_stats(&x_in[p], &tap[p * h..(p + 1) * h]).cosine)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let caught = shift_cos < COSINE_MIN && short_cos.is_none_or(|c| c < COSINE_MIN);
    println!(
        "CONTROL shifted-by-one: best cosine {shift_cos:.6}; one window short: best cosine {} -> {}",
        short_cos.map_or("n/a".to_string(), |c| format!("{c:.6}")),
        if caught { "caught" } else { "NOT CAUGHT" }
    );
    let text_ok = gen_tokens.len() == generated.len();
    let words_ok = flips_beyond.is_empty();
    let cosine_floor_held = cos_min >= COSINE_MIN;
    let pass = words_ok && greedy_ok && text_ok && caught && ref_matches_golden;
    println!(
        "QWEN38_OMEGA_FULL layers {layers} windows {windows} positions {t}: generated {}/{} identical to golden ({} flips inside the written margin band {MARGIN_MIN}, {} beyond it), cosine(min) {cos_min:.6} (floor {COSINE_MIN} {}, recorded not gated) p95_rel(max) {p95_max:.4} greedy {} control {} -> {}",
        gen_tokens.iter().zip(&generated).filter(|(a, b)| **a == **b as usize).count(),
        generated.len(),
        flips_allowed.len(),
        flips_beyond.len(),
        if cosine_floor_held { "held" } else { "NOT held: open precision item" },
        if greedy_ok { "agrees" } else { "DIFFERS" },
        if caught { "caught" } else { "NOT CAUGHT" },
        if pass { "PASS" } else { "FAIL" }
    );
    let report = json!({
        "checkpoint": dir.display().to_string(), "layers": layers, "window_layers": wl, "windows": win_report,
        "positions": t, "prompt_ids": prompt, "golden_ids": generated, "chip_ids": gen_tokens,
        "reference_s": ref_s, "reference_matches_golden": ref_matches_golden, "upload_s": upload_s,
        "weight_bytes": weight_bytes, "cosine_min": cos_min, "p95_rel_max": p95_max,
        "greedy_agrees": greedy_ok, "golden_identical": golden_ok,
        "words_identical_outside_margin_band": words_ok,
        "flips_allowed": flips_allowed, "flips_beyond_error": flips_beyond,
        "cosine_floor_held": cosine_floor_held, "cosine_floor_gated": false,
        "control_shift_best_cosine": shift_cos, "control_window_short_best_cosine": short_cos,
        "control_caught": caught, "thresholds": {"cosine_min_recorded": COSINE_MIN, "margin_band_logits": MARGIN_MIN},
        "gate": "words identical to the golden except where the reference's top-two margin is below the written band (0.1 logit, fixed in advance, never from the run under test); chip greedy agrees at every position outside the band; controls caught; reference equals golden (Drake 2026-10-21, L312; band fixed 02:10Z after a self-measured band let mutant 4 through)",
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
            eprintln!("qwen38_omega_full: {e}");
            std::process::exit(2);
        }
    }
}
