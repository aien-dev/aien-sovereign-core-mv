//! OWN-30B speculative verify cost probe (W-L4c cut 1).
//!
//!   own30b_spec_probe <checkpoint> <golden.json> <receipt-dir> [--reps 6] [--max-m 8] [--prompt p4-long] [--force-memory]
//!
//! Loads the production configuration once (FP8 everywhere, fused KV, whole-prompt driver), prefills
//! one golden prompt, then times, at the same KV position every time (rewind after each call):
//!   * plain decode: `step` + one lm_head (what spec-off pays per token);
//!   * one lm_head launch alone;
//!   * whole-prompt verify `step_verify_wp` of m = 1..max_m rows, host router/combine and device
//!     router/combine (`set_whole_prompt_device_host`), with the driver's per-phase breakdown;
//!   * decode-rounding verify `step_verify` of m rows (the identical path), fewer reps.
//! Median of `reps` after one warm-up call. Writes probe.txt and probe.json. Run under flock /tmp/aien-gb10.lock.

use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, Qwen3MoeOmegaEngine, WeightFormat};

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn main() {
    let mut it = std::env::args().skip(1);
    let mut pos: Vec<String> = Vec::new();
    let (mut reps, mut max_m, mut force, mut prompt_name) = (6usize, 8usize, false, "p4-long".to_string());
    let mut decode_verify = true;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => reps = it.next().expect("--reps N").parse().expect("number"),
            "--max-m" => max_m = it.next().expect("--max-m N").parse().expect("number"),
            "--prompt" => prompt_name = it.next().expect("--prompt NAME"),
            "--no-decode-verify" => decode_verify = false,
            "--force-memory" => force = true,
            _ => pos.push(a),
        }
    }
    if pos.len() != 3 {
        eprintln!("usage: own30b_spec_probe <checkpoint> <golden.json> <receipt-dir> [--reps 6] [--max-m 8] [--prompt NAME] [--no-decode-verify] [--force-memory]");
        std::process::exit(2);
    }
    let (checkpoint, golden_path, receipt) = (PathBuf::from(&pos[0]), PathBuf::from(&pos[1]), PathBuf::from(&pos[2]));
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    // W-L4c: the spin window from AIEN_OMEGA_SPIN_US, as own30b_check does (without it omega waits with
    // its 50 us default and plain decode measures ~39 ms/token instead of ~26).
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    say(format!("spin_us {:?}", aien_omega_gpu::spin_us()));
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !force {
        say("REFUSED: less than 70 GB available; --force-memory to override".into());
        let _ = std::fs::write(receipt.join("probe.txt"), log.join("\n"));
        std::process::exit(3);
    }
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&golden_path).expect("golden file")).expect("golden json");
    let p = golden["prompts"].as_array().expect("prompts").iter().find(|p| p["name"].as_str() == Some(&prompt_name)).expect("prompt name in golden file");
    let ids: Vec<u32> = p["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    // Draft-like tokens to verify: the golden continuation (real text), repeated if short.
    let cont: Vec<u32> = p["golden"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let rows_src: Vec<u32> = cont.iter().copied().cycle().take(max_m.max(1)).collect();

    let (mut engine, report) =
        Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, 8192, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {}).expect("load on Omega");
    engine.set_whole_prompt(true).expect("whole-prompt driver refused");
    say(format!("LOAD {:.1}s prompt {prompt_name} ({} tokens) reps {reps} max_m {max_m}", report.seconds, ids.len()));

    // Prefill through generate (whole-prompt), one token, leaving the KV holding the prompt.
    let params = aien_inference_abi::qwen3_moe_omega::GenerateParams::greedy(1, vec![]);
    let mut first = 0u32;
    engine
        .generate_with(&ids, &params, &mut |lg, _| {
            let mut b = 0usize;
            for (i, &v) in lg.iter().enumerate() {
                if v > lg[b] {
                    b = i;
                }
            }
            b as u32
        }, &mut |t| {
            first = t;
            true
        })
        .expect("prefill");
    let base = ids.len();
    engine.rewind(base);
    say(format!("prefilled {base} positions; first token {first}"));
    let mut json = serde_json::Map::new();

    // --- plain decode step + lm_head ---
    let mut t_step = Vec::new();
    let mut t_lm = Vec::new();
    for r in 0..=reps {
        engine.rewind(base);
        let t = Instant::now();
        let h = engine.step(first).expect("step");
        let s = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let _ = engine.logits(&h).expect("lm_head");
        let l = t.elapsed().as_secs_f64() * 1e3;
        if r > 0 {
            t_step.push(s);
            t_lm.push(l);
        }
    }
    let lt = engine.last_timing.clone();
    say(format!(
        "PLAIN last step+lm_head timing: attention {:.2} experts {:.2} lm_head {:.2} host {:.2} chip {:.2} ms, launches {}",
        ms(lt.attention_ns), ms(lt.experts_ns), ms(lt.lm_head_ns), ms(lt.host_ns), ms(lt.chip_ns), lt.launches
    ));
    let (step_ms, lm_ms) = (median(&mut t_step), median(&mut t_lm));
    say(format!("PLAIN step {step_ms:.2} ms + lm_head {lm_ms:.2} ms = {:.2} ms/token ({:.1} tok/s)", step_ms + lm_ms, 1000.0 / (step_ms + lm_ms)));
    json.insert("plain".into(), serde_json::json!({"step_ms": step_ms, "lm_head_ms": lm_ms}));

    // --- whole-prompt verify, host and device router/combine ---
    let mut wp_rows = Vec::new();
    for devhost in [false, true] {
        if let Err(e) = engine.set_whole_prompt_device_host(devhost) {
            say(format!("devhost={devhost}: refused ({e}); skipped"));
            continue;
        }
        for m in 1..=max_m {
            let toks: Vec<u32> = std::iter::once(first).chain(rows_src.iter().copied()).take(m).collect();
            let mut walls = Vec::new();
            let mut last = None;
            for r in 0..=reps {
                engine.rewind(base);
                let t = Instant::now();
                let _ = engine.step_verify_wp(&toks).expect("step_verify_wp");
                let w = t.elapsed().as_secs_f64() * 1e3;
                if r > 0 {
                    walls.push(w);
                    last = engine.whole_prompt_stats().cloned();
                }
            }
            let wall = median(&mut walls);
            let st = last.expect("stats");
            let pr = &st.prof;
            let host_other = ms(pr.embed_ns + pr.norm_in_ns + pr.resid_ns + pr.norm_post_ns + pr.final_ns);
            say(format!(
                "WP devhost={devhost} m={m}: wall {wall:.2} ms | dev calls {:.2} (chip {:.2} attn {:.2}) | router {:.2} plan {:.2} tiles {:.2} combine {:.2} norms/resid/embed {:.2} | launches {} router {} combine {}",
                ms(pr.dev_ns), ms(st.chip_ns), ms(st.attn_ns), ms(pr.router_ns), ms(pr.plan_ns), ms(pr.tiles_ns), ms(pr.combine_ns), host_other,
                st.launches(), st.router, st.combine
            ));
            wp_rows.push(serde_json::json!({"devhost": devhost, "m": m, "wall_ms": wall, "dev_ms": ms(pr.dev_ns), "chip_ms": ms(st.chip_ns), "attn_ms": ms(st.attn_ns),
                "router_ms": ms(pr.router_ns), "plan_ms": ms(pr.plan_ns), "tiles_ms": ms(pr.tiles_ns), "combine_ms": ms(pr.combine_ns), "host_other_ms": host_other,
                "launches": st.launches(), "router_launches": st.router, "combine_launches": st.combine}));
        }
    }
    json.insert("wp".into(), serde_json::Value::Array(wp_rows));
    let _ = engine.set_whole_prompt_device_host(false);

    // --- decode-rounding verify (the token-identical path) ---
    if decode_verify && engine.spec_possible() {
        let mut dv = Vec::new();
        for m in [1usize, 2, 4, 6, 8].into_iter().filter(|&m| m <= max_m) {
            let toks: Vec<u32> = std::iter::once(first).chain(rows_src.iter().copied()).take(m).collect();
            let mut walls = Vec::new();
            for r in 0..=reps.min(3) {
                engine.rewind(base);
                let t = Instant::now();
                let _ = engine.step_verify(&toks).expect("step_verify");
                if r > 0 {
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let wall = median(&mut walls);
            say(format!("DECODE-VERIFY m={m}: wall {wall:.2} ms"));
            dv.push(serde_json::json!({"m": m, "wall_ms": wall}));
        }
        json.insert("decode_verify".into(), serde_json::Value::Array(dv));
    }
    engine.rewind(base);
    say("PROBE DONE".into());
    std::fs::write(receipt.join("probe.txt"), log.join("\n")).expect("write log");
    std::fs::write(receipt.join("probe.json"), serde_json::to_string_pretty(&serde_json::Value::Object(json)).unwrap()).expect("write json");
}
