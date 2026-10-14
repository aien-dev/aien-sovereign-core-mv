//! OWN-30B exact-verify probe (W-L4d). Real checkpoint weights, only the first N layers loaded
//! (so the 30 GB model is never resident; the output is not the model's, the numerics are).
//!
//!   own30b_exact_probe <checkpoint> <layers> <golden.json> <receipt-dir> [--reps 6] [--max-m 8] [--prompt p4-long]
//!                      [--paths verify,fast,wp] [--no-identity] [--decode-graph on] [--resident on|off] [--rows 1,3,5,9]
//!   W-L4f: paths `batch` (the batched exact verify; needs --decode-graph on), `batch-fp8` (its act-fp8 control) and
//!   `graph` (m one-list tokens in a row: the served plain step). Their IDENTITY reference is the one-list token, and
//!   LM-IDENTITY compares the host lm_head on each verified row with the token's in-list logits.
//!
//! After a plain prefill of one golden prompt, at the same KV position every time (rewind after each call):
//!   (W-L4e: paths also `mutant` = bf16 attention, `mutant-res` = bf16 residual stream, `mutant-fp8` = W8A8 activations; every IDENTITY line also
//!   reports the logit change, argmax flips and the reference rows' smallest top-1/top-2 margin.)
//!   1. IDENTITY: for m = 1..max_m rows, the hidden state of every row of each verify path is compared bit for bit
//!      with m plain one-token `step` calls on the same tokens. Prints `IDENTITY path=<p> m=<m> rows_equal=a/m
//!      words_differ=w`. The whole-prompt path (FP8 activations) is the instrument's negative control: it must
//!      show differing words, or the comparison cannot say no.
//!   2. TIMING: median wall of `step` (+ lm_head), and of each verify path for m = 1..max_m, with the engine's
//!      StepTiming breakdown of the last call.
//!
//! Run under flock /tmp/aien-gb10.lock. Writes probe.txt and probe.json into the receipt dir.

use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::spec::{ExactProf, NumericsMutant};
use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeOmegaEngine, StepTiming, WeightFormat};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn argmax(lg: &[f32]) -> usize {
    lg.iter().enumerate().fold(0usize, |b, (i, &v)| if v > lg[b] { i } else { b })
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn timing_line(t: &StepTiming) -> String {
    format!(
        "total {:.2} attention {:.2} experts {:.2} route {:.2} moe_host {:.2} host {:.2} chip {:.2} ms launches {}",
        ms(t.total_ns),
        ms(t.attention_ns),
        ms(t.experts_ns),
        ms(t.route_ns),
        ms(t.moe_host_ns),
        ms(t.host_ns),
        ms(t.chip_ns),
        t.launches
    )
}

fn run_path(engine: &mut Qwen3MoeOmegaEngine, path: &str, toks: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    match path {
        "verify" => engine.step_verify(toks),
        "fast" => engine.step_verify_exact(toks),
        "wp" => engine.step_verify_wp(toks),
        "mutant" => engine.step_verify_exact_inner(toks, &mut ExactProf::default(), NumericsMutant::AttnBf16),
        // W-L4e: the residual-stream bf16 mutant.
        "mutant-res" => engine.step_verify_exact_inner(toks, &mut ExactProf::default(), NumericsMutant::ResidualBf16),
        "mutant-fp8" => engine.step_verify_exact_inner(toks, &mut ExactProf::default(), NumericsMutant::ActFp8),
        // W-L4f: the batched exact verify on the one-list decode graph, its act-fp8 control, and m one-list tokens
        // one after another (the served plain step: what a round of m rows costs without speculation).
        "batch" => engine.step_verify_batch(toks, NumericsMutant::None),
        "batch-fp8" => engine.step_verify_batch(toks, NumericsMutant::ActFp8),
        "graph" => toks.iter().map(|&t| engine.decode_step_hidden(t, false).map(|(h, _)| h)).collect(),
        other => Err(format!("unknown path {other}")),
    }
}

fn main() {
    let mut it = std::env::args().skip(1);
    let mut pos: Vec<String> = Vec::new();
    let (mut reps, mut max_m, mut prompt_name) = (6usize, 8usize, "p4-long".to_string());
    let mut paths: Vec<String> = vec!["verify".into(), "fast".into(), "wp".into()];
    let mut identity = true;
    let (mut decode_graph, mut resident, mut rows_list): (bool, Option<bool>, Option<Vec<usize>>) = (false, None, None);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => reps = it.next().expect("--reps N").parse().expect("number"),
            "--max-m" => max_m = it.next().expect("--max-m N").parse().expect("number"),
            "--prompt" => prompt_name = it.next().expect("--prompt NAME"),
            "--paths" => paths = it.next().expect("--paths a,b").split(',').map(str::to_string).collect(),
            "--no-identity" => identity = false,
            "--decode-graph" => decode_graph = it.next().expect("--decode-graph on|off") == "on",
            "--resident" => resident = Some(it.next().expect("--resident on|off") == "on"),
            "--rows" => rows_list = Some(it.next().expect("--rows 1,3,5,9").split(',').map(|s| s.parse().expect("number")).collect()),
            _ => pos.push(a),
        }
    }
    if pos.len() != 4 {
        eprintln!("usage: own30b_exact_probe <checkpoint> <layers> <golden.json> <receipt-dir> [--reps 6] [--max-m 8] [--prompt NAME] [--paths verify,fast,wp] [--no-identity]");
        std::process::exit(2);
    }
    let checkpoint = PathBuf::from(&pos[0]);
    let layers: usize = pos[1].parse().expect("layers");
    let (golden_path, receipt) = (PathBuf::from(&pos[2]), PathBuf::from(&pos[3]));
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    say(format!("spin_us {:?}", aien_omega_gpu::spin_us()));
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&golden_path).expect("golden file")).expect("golden json");
    let p = golden["prompts"].as_array().expect("prompts").iter().find(|p| p["name"].as_str() == Some(&prompt_name)).expect("prompt name in golden file");
    let ids: Vec<u32> = p["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let cont: Vec<u32> = p["golden"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let rows_src: Vec<u32> = cont.iter().copied().cycle().take(max_m.max(1)).collect();

    let t = Instant::now();
    let (mut engine, _report) =
        Qwen3MoeOmegaEngine::load_omega_first_layers(&checkpoint, layers, 4096, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {})
            .expect("load on Omega");
    let wp_ok = engine.set_whole_prompt(true).is_ok();
    say(format!("LOAD {layers} layers {:.1}s prompt {prompt_name} ({} tokens) reps {reps} max_m {max_m} whole-prompt {wp_ok}", t.elapsed().as_secs_f64(), ids.len()));
    let paths: Vec<String> = paths.into_iter().filter(|p| p != "wp" || wp_ok).collect();

    // Plain prefill (step_batch chunks), leaving the KV holding the prompt; the first token is a real id.
    let _ = engine.set_whole_prompt(false);
    let params = GenerateParams::greedy(1, vec![]);
    let mut first = 0u32;
    engine
        .generate_with(&ids, &params, &mut |lg, _| lg.iter().enumerate().fold(0usize, |b, (i, &v)| if v > lg[b] { i } else { b }) as u32, &mut |t| {
            first = t;
            true
        })
        .expect("prefill");
    if wp_ok {
        engine.set_whole_prompt(true).expect("whole-prompt back on");
    }
    let base = ids.len();
    engine.rewind(base);
    say(format!("prefilled {base} positions; first token {first}"));
    // W-L4f: the one-list decode (the served plain step) after the plain prefill, so both start from the same KV.
    if let Some(r) = resident {
        aien_omega_dgraph::resident::set(aien_omega_dgraph::resident::Resident { on: r, ..aien_omega_dgraph::resident::GATED }).expect("resident decode switch");
    }
    if decode_graph {
        engine.set_decode_graph(true).expect("one-list decode on");
    }
    say(format!("decode_graph={} resident={:?}", engine.decode_graph_on(), aien_omega_dgraph::resident::get()));
    let ms_list: Vec<usize> = rows_list.clone().unwrap_or_else(|| (1..=max_m).collect());
    assert!(ms_list.iter().all(|&m| m >= 1 && m <= max_m), "--rows values must be 1..=--max-m");
    let mut json = serde_json::Map::new();
    let toks_for = |m: usize| -> Vec<u32> { std::iter::once(first).chain(rows_src.iter().copied()).take(m).collect() };

    // --- 1. identity ---
    if identity {
        let mut rows_json = Vec::new();
        let graph_paths = |p: &str| p.starts_with("batch") || p == "graph";
        for &m in &ms_list {
            let toks = toks_for(m);
            engine.rewind(base);
            let ref_step: Vec<Vec<f32>> = if paths.iter().any(|p| !graph_paths(p)) { toks.iter().map(|&t| engine.step(t).expect("step")).collect() } else { Vec::new() };
            // W-L4f: the batched verify's reference is the served plain step, the one-list token (and its in-list logits).
            let (ref_graph, ref_graph_logits): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if paths.iter().any(|p| graph_paths(p)) {
                engine.rewind(base);
                toks.iter().map(|&t| engine.decode_step_hidden(t, true).map(|(h, l)| (h, l.unwrap_or_default())).expect("one-list token")).unzip()
            } else {
                (Vec::new(), Vec::new())
            };
            for path in &paths {
                engine.rewind(base);
                let got = run_path(&mut engine, path, &toks).unwrap_or_else(|e| panic!("{path} x{m}: {e}"));
                let reference = if graph_paths(path) { &ref_graph } else { &ref_step };
                if graph_paths(path) {
                    // The spec loop runs the host lm_head on verified rows: it must equal the graph's in-list lm_head.
                    let mut lm_rows_equal = 0usize;
                    for (b, lg) in got.iter().zip(&ref_graph_logits) {
                        let lb = engine.logits(b).expect("lm_head");
                        lm_rows_equal += (lb.len() == lg.len() && lb.iter().zip(lg).all(|(x, y)| x.to_bits() == y.to_bits())) as usize;
                    }
                    say(format!("LM-IDENTITY path={path} m={m} host_lm_head_rows_equal_graph_logits={lm_rows_equal}/{m}"));
                }
                let mut rows_equal = 0usize;
                let mut words = 0usize;
                let mut first_bad: Option<usize> = None;
                for (r, (a, b)) in reference.iter().zip(&got).enumerate() {
                    let d = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    if d == 0 && a.len() == b.len() {
                        rows_equal += 1;
                    } else if first_bad.is_none() {
                        first_bad = Some(r);
                    }
                    words += d;
                }
                // W-L4e: what the difference does to the logits: largest |logit change|, rows whose argmax
                // flipped, and the smallest top-1/top-2 margin of the reference rows (a flip needs a change
                // of about the margin).
                let (mut dmax, mut flips, mut margin) = (0.0f32, 0usize, f32::INFINITY);
                for (a, b) in reference.iter().zip(&got) {
                    let la = engine.logits(a).expect("lm_head");
                    let lb = engine.logits(b).expect("lm_head");
                    dmax = la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).fold(dmax, f32::max);
                    let (ia, ib) = (argmax(&la), argmax(&lb));
                    flips += (ia != ib) as usize;
                    let second = la.iter().enumerate().filter(|(i, _)| *i != ia).map(|(_, &v)| v).fold(f32::NEG_INFINITY, f32::max);
                    margin = margin.min(la[ia] - second);
                }
                say(format!("IDENTITY path={path} m={m} rows_equal={rows_equal}/{m} words_differ={words} first_bad_row={first_bad:?} logit_max_abs_diff={dmax:.4e} argmax_flips={flips} min_ref_margin={margin:.4}"));
                rows_json.push(serde_json::json!({"path": path, "m": m, "rows_equal": rows_equal, "words_differ": words, "logit_max_abs_diff": dmax, "argmax_flips": flips, "min_ref_margin": margin}));
            }
        }
        json.insert("identity".into(), serde_json::Value::Array(rows_json));
    }

    // --- 2. timing ---
    let mut t_step = Vec::new();
    let mut t_lm = Vec::new();
    for r in 0..=reps {
        engine.rewind(base);
        let t = Instant::now();
        let h = engine.step(first).expect("step");
        let s = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let _ = engine.logits(&h).expect("lm_head");
        if r > 0 {
            t_step.push(s);
            t_lm.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let lt = engine.last_timing;
    let (step_ms, lm_ms) = (median(&mut t_step), median(&mut t_lm));
    say(format!("PLAIN step {step_ms:.2} ms ({:.3} ms/layer) lm_head {lm_ms:.2} ms | {}", step_ms / layers as f64, timing_line(&lt)));
    json.insert("plain".into(), serde_json::json!({"step_ms": step_ms, "lm_head_ms": lm_ms, "layers": layers}));
    let mut timing_rows = Vec::new();
    for path in &paths {
        for &m in &ms_list {
            let toks = toks_for(m);
            let mut walls = Vec::new();
            let mut last = StepTiming::default();
            for r in 0..=reps {
                engine.rewind(base);
                let t = Instant::now();
                run_path(&mut engine, path, &toks).unwrap_or_else(|e| panic!("{path} x{m}: {e}"));
                if r > 0 {
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                    last = engine.last_timing;
                }
            }
            let wall = median(&mut walls);
            say(format!("TIME path={path} m={m}: wall {wall:.2} ms ({:.3} ms/layer, {:.2}x plain step) | {}", wall / layers as f64, wall / step_ms, timing_line(&last)));
            if path == "fast" {
                let mut prof = ExactProf::default();
                for _ in 0..reps {
                    engine.rewind(base);
                    engine.step_verify_exact_prof(&toks, &mut prof).expect("exact prof");
                }
                say(format!("PROF m={m} per call (all {layers} layers): {}", prof.line()));
            }
            timing_rows.push(serde_json::json!({"path": path, "m": m, "wall_ms": wall, "per_layer_ms": wall / layers as f64}));
        }
    }
    json.insert("timing".into(), serde_json::Value::Array(timing_rows));
    engine.rewind(base);
    say("PROBE DONE".into());
    std::fs::write(receipt.join("probe.txt"), log.join("\n")).expect("write log");
    std::fs::write(receipt.join("probe.json"), serde_json::to_string_pretty(&serde_json::Value::Object(json)).unwrap()).expect("write json");
}
