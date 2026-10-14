//! OWN-30B speculative decoding chip gate (cut 1, greedy).
//!
//!   own30b_spec_check <checkpoint> <golden.json> <receipt-dir> [--runs 3] [--new 16] [--max-context 8192] [--force-memory]
//!
//! Loads the production configuration (FP8 everywhere, device router, bf16 lm_head GEMV) once, then
//! for each of the golden file's prompts and each run: generates with speculation OFF and ON and
//! compares the tokens (and against the golden list). Reports speculation acceptance and tok/s.
//! Negative control: with the extra-accept mutant on, at least one prompt must differ from OFF,
//! otherwise the identity check itself is not trustworthy and the run FAILS. A refusal or error of
//! any launch is a failure, never "caught". Exit 0 only when everything passes.
//!
//! Memory: refuses with less than 70 GB available (the live server holds a 30 GB model) unless --force-memory.
//! Run it under `flock /tmp/aien-gb10.lock`.

use std::path::PathBuf;

use aien_inference_abi::qwen3_moe_omega::spec::SpecConfig;
use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeConfig, Qwen3MoeOmegaEngine, WeightFormat};

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

fn argmax(lg: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in lg.iter().enumerate() {
        if v > lg[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let mut it = std::env::args().skip(1);
    let mut pos: Vec<String> = Vec::new();
    let (mut runs, mut new_tokens, mut max_context, mut force) = (3usize, 16usize, 8192usize, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--runs" => runs = it.next().expect("--runs N").parse().expect("number"),
            "--new" => new_tokens = it.next().expect("--new N").parse().expect("number"),
            "--max-context" => max_context = it.next().expect("--max-context N").parse().expect("number"),
            "--force-memory" => force = true,
            _ => pos.push(a),
        }
    }
    if pos.len() != 3 {
        eprintln!("usage: own30b_spec_check <checkpoint> <golden.json> <receipt-dir> [--runs 3] [--new 16] [--max-context 8192] [--force-memory]");
        std::process::exit(2);
    }
    let (checkpoint, golden_path, receipt) = (PathBuf::from(&pos[0]), PathBuf::from(&pos[1]), PathBuf::from(&pos[2]));
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !force {
        say("REFUSED: less than 70 GB available (another model may be loaded); --force-memory to override".into());
        let _ = std::fs::write(receipt.join("run.log"), log.join("\n"));
        std::process::exit(3);
    }
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&golden_path).expect("golden file")).expect("golden json");
    let config = Qwen3MoeConfig::from_checkpoint(&checkpoint).expect("config");
    let (mut engine, report) =
        Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, max_context, WeightFormat::Fp8All, AttnPath::Auto, DevicePaths::default(), &mut |_| {})
            .expect("load on Omega");
    say(format!(
        "LOAD {:.1}s attention={} prefill_chunk={} device_routing={} spec_possible={}",
        report.seconds,
        engine.attention_path(),
        engine.prefill_chunk,
        engine.device_routing,
        engine.spec_possible()
    ));
    if !engine.spec_possible() {
        say("FAIL: speculative verify path is not available on this engine (needs the prefill GEMM everywhere)".into());
        let _ = std::fs::write(receipt.join("run.log"), log.join("\n"));
        std::process::exit(1);
    }
    let off = SpecConfig::default();
    let on = SpecConfig { enabled: true, ..SpecConfig::default() };
    let mutant = SpecConfig { enabled: true, mutant_accept_extra: true, ..SpecConfig::default() };
    let params = GenerateParams::greedy(new_tokens, config.eos_token_ids.clone());
    let mut all_ok = true;
    let mut rows = Vec::new();
    let mut mutant_differs = false;
    for p in golden["prompts"].as_array().expect("prompts") {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let ids: Vec<u32> = p["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want: Vec<u32> =
            p["golden"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).filter(|t| !config.eos_token_ids.contains(t)).collect();
        for run in 1..=runs {
            let go = |cfg: &SpecConfig, engine: &mut Qwen3MoeOmegaEngine| {
                let mut got = Vec::new();
                let (o, st) = engine
                    .generate_spec_with(&ids, &params, cfg, &mut |lg, _| argmax(lg), &mut |t| {
                        got.push(t);
                        true
                    })
                    .expect("generate (an error is a failure, not a pass)");
                let tps = if o.decode_ms > 0 { o.emitted_tokens as f64 * 1000.0 / o.decode_ms as f64 } else { 0.0 };
                (got, st, tps, o.decode_ms)
            };
            let (t_off, _, tps_off, ms_off) = go(&off, &mut engine);
            let (t_on, st, tps_on, ms_on) = go(&on, &mut engine);
            let same = t_off == t_on;
            let golden_same = t_off == want;
            if !same {
                all_ok = false;
            }
            say(format!(
                "{name} run {run}: spec-on vs spec-off {} | off-vs-golden {} | off {tps_off:.1} tok/s ({ms_off} ms) on {tps_on:.1} tok/s ({ms_on} ms) | rounds {} plain {} drafted {} accepted {} acceptance {:.3}",
                if same { "IDENTICAL" } else { "DIFFERENT (FAIL)" },
                if golden_same { "IDENTICAL" } else { "differs (recorded; spec-off is the reference)" },
                st.rounds, st.plain_steps, st.drafted, st.accepted, st.acceptance()
            ));
            rows.push(serde_json::json!({"prompt": name, "run": run, "identical": same, "off_matches_golden": golden_same,
                "off_tok_s": tps_off, "on_tok_s": tps_on, "rounds": st.rounds, "plain": st.plain_steps, "drafted": st.drafted, "accepted": st.accepted,
                "acceptance": st.acceptance(), "tokens_off": t_off, "tokens_on": t_on}));
            if run == 1 {
                let (t_mut, _, _, _) = go(&mutant, &mut engine);
                if t_mut != t_off {
                    mutant_differs = true;
                }
                say(format!("{name} negative control (accept one extra wrong draft): {}", if t_mut != t_off { "output differs (CAUGHT)" } else { "output unchanged (no rejected draft to corrupt)" }));
            }
        }
    }
    if !mutant_differs {
        say("FAIL: the extra-accept mutant never changed the output; the identity check is not proven on this data".into());
        all_ok = false;
    }
    say(format!("RESULT: {}", if all_ok { "PASS" } else { "FAIL" }));
    std::fs::write(receipt.join("run.log"), log.join("\n")).expect("write log");
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&serde_json::json!({"runs": rows, "pass": all_ok})).unwrap()).expect("write json");
    std::process::exit(if all_ok { 0 } else { 1 });
}
