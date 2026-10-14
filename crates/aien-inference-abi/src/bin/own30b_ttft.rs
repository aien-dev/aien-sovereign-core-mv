//! OWN-30B time-to-first-token, whole-prompt prefill OFF vs ON, in one process (W-L1G-W).
//!
//!   own30b_ttft <checkpoint> <receipt-dir> [--lens 512,2048] [--new-tokens 32] [--reps 2]
//!               [--max-context 8192] [--force-memory]
//!               [--devhost]   (W-L1G-E: also ON with router logits + combine on the chip, and expert agreement)
//!
//! Loads the FP8 model once (fused attention), then for every prompt length runs, `--reps`
//! times, OFF (the engine's chunked prefill) then ON (`set_whole_prompt(true)`), each from an
//! empty KV (`reset`), greedy, no stop ids, so exactly `--new-tokens` tokens. TTFT = wall time
//! from the `generate_with` call to the first emitted token (prefill + the first lm_head).
//! Compares OFF vs ON tokens (first differing position) and the first-step logits (cosine, max
//! abs diff, top-2 margins). Writes `run.json` and `run.txt` into the receipt dir.
use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeOmegaEngine, StepTiming, WeightFormat};
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;
use aien_inference_abi::qwen3_moe_omega_prefill::WpTrace;

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

/// A chat prompt of exactly `n` tokens: a fixed head, numbered log lines (each differs), cut so
/// that head + body + tail = n. Encoded piecewise (the tokenizer refuses > 2048 tokens per call).
fn prompt_ids(tok: &Tokenizer, n: usize) -> Vec<u32> {
    let enc = |t: &str| tok.encode_with_special(t, false).expect("encode");
    let mut ids = enc("<|im_start|>user\nYou are a careful engineering assistant. Read the log below and say, in plain words, which block number has the largest decode time.\n\n");
    let tail = enc("\nAnswer with the block number only.<|im_end|>\n<|im_start|>assistant\n");
    assert!(n > ids.len() + tail.len() + 8, "prompt length {n} too short");
    let mut b = 0usize;
    while ids.len() + tail.len() < n {
        ids.extend(enc(&format!(
            "[block {b}] loaded 48 layers in {} s, resident {}.{} GB\n[block {b}] prompt {} tokens, prefill {}.{} s\n[block {b}] decode 16 tokens, {} ms/token\n",
            200 + (b * 7) % 23,
            58 + b % 5,
            (b * 3) % 10,
            20 + (b * 37) % 700,
            1 + (b * 5) % 40,
            (b * 7) % 10,
            50 + (b * 11) % 37,
        )));
        b += 1;
    }
    ids.truncate(n - tail.len());
    ids.extend(tail);
    assert_eq!(ids.len(), n);
    ids
}

struct Run {
    tokens: Vec<u32>,
    logits0: Vec<f32>,
    ttft_ms: f64,
    prefill_ms: u64,
    decode_ms: u64,
    pt: StepTiming,
}

fn run(engine: &mut Qwen3MoeOmegaEngine, ids: &[u32], new_tokens: usize) -> Run {
    engine.reset();
    let mut tokens = Vec::new();
    let mut logits0 = Vec::new();
    let mut first: Option<f64> = None;
    let t0 = Instant::now();
    let o = engine
        .generate_with(
            ids,
            &GenerateParams::greedy(new_tokens, Vec::new()),
            &mut |logits, step| {
                if step == 0 {
                    logits0 = logits.to_vec();
                }
                aien_inference_abi::tensor::sample_argmax(logits).0
            },
            &mut |t| {
                if first.is_none() {
                    first = Some(t0.elapsed().as_secs_f64() * 1e3);
                }
                tokens.push(t);
                true
            },
        )
        .expect("generate");
    Run { tokens, logits0, ttft_ms: first.expect("no token"), prefill_ms: o.prefill_ms, decode_ms: o.decode_ms, pt: o.prefill_timing }
}

fn top2(l: &[f32]) -> (usize, f32) {
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    (idx[0], l[idx[0]] - l[idx[1]])
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = PathBuf::from(args.get(1).expect("checkpoint dir"));
    let receipt = PathBuf::from(args.get(2).expect("receipt dir"));
    let (mut lens, mut new_tokens, mut reps, mut max_context, mut force, mut cached) = (vec![512usize, 2048], 32usize, 2usize, 8192usize, false, false);
    let (mut devhost, mut devhost_exact) = (false, true);
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lens" => lens = it.next().expect("--lens").split(',').map(|s| s.parse().expect("len")).collect(),
            "--new-tokens" => new_tokens = it.next().expect("--new-tokens").parse().expect("n"),
            "--reps" => reps = it.next().expect("--reps").parse().expect("n"),
            "--max-context" => max_context = it.next().expect("--max-context").parse().expect("n"),
            "--force-memory" => force = true,
            "--cached" => cached = true,
            "--devhost" => devhost = true,
            o => panic!("unknown argument {o}"),
        }
    }
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !force {
        say("REFUSED: less than 70 GB available; --force-memory to override".into());
        std::process::exit(3);
    }
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    say(format!(
        "build native={} pf={} pattn={} pattn_dev={} spin_us={:?}",
        aien_omega_gpu::is_native(),
        aien_omega_pf::is_native(),
        aien_omega_pf::has_attention(),
        aien_omega_pf::has_attention_dev(),
        aien_omega_gpu::spin_us()
    ));
    let tok = Tokenizer::from_file(checkpoint.join("tokenizer.json")).expect("tokenizer");
    let t0 = Instant::now();
    let (mut engine, report) =
        Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, max_context, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {})
            .expect("load on Omega");
    say(format!(
        "LOAD {:.1}s resident {:.2} GB attention {} prefill_gemm {} chunk {} mem_available_after {:.1} GB",
        t0.elapsed().as_secs_f64(),
        report.resident_bytes as f64 / 1e9,
        engine.attention_path(),
        engine.prefill_gemm,
        engine.prefill_chunk,
        mem_available_gb()
    ));
    let mut rows = Vec::new();
    let mut all_identical = true;
    let mut offs: Vec<(usize, Vec<u32>, Vec<f32>, f64)> = Vec::new();
    for &n in &lens {
        let ids = prompt_ids(&tok, n);
        for rep in 0..reps {
            engine.set_whole_prompt(false).expect("off");
            let off = run(&mut engine, &ids, new_tokens);
            engine.set_whole_prompt(true).expect("whole-prompt prefill refused");
            let on = run(&mut engine, &ids, new_tokens);
            let st = engine.whole_prompt_stats().cloned().unwrap_or_default();
            let diff = off.tokens.iter().zip(&on.tokens).position(|(a, b)| a != b).or((off.tokens.len() != on.tokens.len()).then_some(off.tokens.len().min(on.tokens.len())));
            let cos = cosine(&off.logits0, &on.logits0);
            let maxd = off.logits0.iter().zip(&on.logits0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let (ta, ma) = top2(&off.logits0);
            let (tb, mb) = top2(&on.logits0);
            all_identical &= diff.is_none();
            if rep + 1 == reps {
                offs.push((n, off.tokens.clone(), off.logits0.clone(), off.ttft_ms));
            }
            say(format!(
                "len {n} rep {rep}: TTFT off {:.1} ms on {:.1} ms (x{:.2}); prefill_ms off {} on {}; decode {} tokens off {} ms on {} ms; tokens {} ; logits0 cosine {cos:.7} maxdiff {maxd:.3e} top1 {ta}/{tb} margin {ma:.3}/{mb:.3}",
                off.ttft_ms,
                on.ttft_ms,
                off.ttft_ms / on.ttft_ms,
                off.prefill_ms,
                on.prefill_ms,
                new_tokens,
                off.decode_ms,
                on.decode_ms,
                match diff {
                    None => "IDENTICAL".to_string(),
                    Some(p) => format!("DIFFER at {p}"),
                }
            ));
            say(format!(
                "  on: launches {} chip(GEMM+quant) {:.1} ms attention chip {:.1} ms host {:.1} ms attn_dev {} | off: chip {:.1} ms launches {}",
                st.launches(),
                st.chip_ns as f64 / 1e6,
                st.attn_ns as f64 / 1e6,
                st.host_ns as f64 / 1e6,
                st.attn_dev,
                off.pt.chip_ns as f64 / 1e6,
                off.pt.launches
            ));
            let p = &st.prof;
            let ms = |v: u64| v as f64 / 1e6;
            say(format!(
                "  on profile ms: wall {:.1} dev(calls) {:.1} | host: embed {:.1} norm_in {:.1} resid {:.1} norm_post {:.1} router {:.1} plan {:.1} tiles {:.1} combine {:.1} final {:.1}",
                ms(p.wall_ns), ms(p.dev_ns), ms(p.embed_ns), ms(p.norm_in_ns), ms(p.resid_ns), ms(p.norm_post_ns), ms(p.router_ns), ms(p.plan_ns), ms(p.tiles_ns), ms(p.combine_ns), ms(p.final_ns)
            ));
            rows.push(serde_json::json!({
                "len": n, "rep": rep,
                "ttft_ms_off": off.ttft_ms, "ttft_ms_on": on.ttft_ms,
                "prefill_ms_off": off.prefill_ms, "prefill_ms_on": on.prefill_ms,
                "decode_ms_off": off.decode_ms, "decode_ms_on": on.decode_ms,
                "tokens_off": off.tokens, "tokens_on": on.tokens, "first_diff": diff,
                "logits0_cosine": cos, "logits0_maxdiff": maxd, "top1_off": ta, "top1_on": tb, "margin_off": ma, "margin_on": mb,
                "on_stats": {"launches": st.launches(), "chip_ns": st.chip_ns, "attn_ns": st.attn_ns, "host_ns": st.host_ns, "attn_dev": st.attn_dev},
                "off_timing": {"chip_ns": off.pt.chip_ns, "launches": off.pt.launches, "total_ns": off.pt.total_ns},
            }));
            if devhost {
                // Phase D (W-L1G-E): the same ON run with the router logits + MoE combine on the chip.
                engine.set_whole_prompt_device_host(true).expect("device router / combine refused");
                let dv = run(&mut engine, &ids, new_tokens);
                let sd = engine.whole_prompt_stats().cloned().unwrap_or_default();
                engine.set_whole_prompt_device_host(false).expect("device host work off");
                let same_tokens = dv.tokens == on.tokens;
                let bits_differ = dv.logits0.iter().zip(&on.logits0).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + dv.logits0.len().abs_diff(on.logits0.len());
                let diff_off = off.tokens.iter().zip(&dv.tokens).position(|(a, b)| a != b);
                all_identical &= diff_off.is_none();
                devhost_exact &= same_tokens && bits_differ == 0;
                say(format!(
                    "len {n} rep {rep} DEVHOST: TTFT off {:.1} ms on(host) {:.1} ms on(dev) {:.1} ms (x{:.2} vs off, x{:.2} vs on-host); prefill_ms on(dev) {}; vs on-host tokens {} logits0 bits differ {bits_differ}; vs off tokens {}",
                    off.ttft_ms,
                    on.ttft_ms,
                    dv.ttft_ms,
                    off.ttft_ms / dv.ttft_ms,
                    on.ttft_ms / dv.ttft_ms,
                    dv.prefill_ms,
                    if same_tokens { "IDENTICAL" } else { "DIFFER" },
                    diff_off.map_or("IDENTICAL".to_string(), |p| format!("DIFFER at {p}"))
                ));
                let p = &sd.prof;
                say(format!(
                    "  on(dev): launches {} + router {} + combine {} chip {:.1} ms attention {:.1} ms host {:.1} ms | profile ms: wall {:.1} dev(calls) {:.1} | host: embed {:.1} norm_in {:.1} resid {:.1} norm_post {:.1} router {:.1} plan {:.1} tiles {:.1} combine {:.1} final {:.1}",
                    sd.launches(), sd.router, sd.combine,
                    sd.chip_ns as f64 / 1e6, sd.attn_ns as f64 / 1e6, sd.host_ns as f64 / 1e6,
                    ms(p.wall_ns), ms(p.dev_ns), ms(p.embed_ns), ms(p.norm_in_ns), ms(p.resid_ns), ms(p.norm_post_ns), ms(p.router_ns), ms(p.plan_ns), ms(p.tiles_ns), ms(p.combine_ns), ms(p.final_ns)
                ));
                rows.push(serde_json::json!({"len": n, "rep": rep, "devhost": true, "ttft_ms_off": off.ttft_ms, "ttft_ms_on_host": on.ttft_ms, "ttft_ms_on_dev": dv.ttft_ms,
                    "prefill_ms_on_dev": dv.prefill_ms, "tokens_on_dev": dv.tokens, "tokens_equal_on_host": same_tokens, "logits0_bits_differ_vs_on_host": bits_differ, "first_diff_vs_off": diff_off,
                    "on_dev_stats": {"launches": sd.launches(), "router": sd.router, "combine": sd.combine, "chip_ns": sd.chip_ns, "attn_ns": sd.attn_ns, "host_ns": sd.host_ns, "wall_ns": p.wall_ns,
                        "router_ns": p.router_ns, "plan_ns": p.plan_ns, "combine_ns": p.combine_ns}}));
            }
        }
        if devhost {
            // Expert agreement (W-L1G-E), untimed: one traced ON run per routing path, every layer's expert ids compared.
            engine.set_whole_prompt(true).expect("on");
            engine.set_whole_prompt_device_host(false).expect("host");
            engine.whole_prompt_set_trace(Some(WpTrace::new(None))).expect("trace");
            let _ = run(&mut engine, &ids, 1);
            let th = engine.whole_prompt_take_trace().expect("host trace");
            engine.set_whole_prompt_device_host(true).expect("dev");
            engine.whole_prompt_set_trace(Some(WpTrace::new(None))).expect("trace");
            let _ = run(&mut engine, &ids, 1);
            let td = engine.whole_prompt_take_trace().expect("device trace");
            engine.set_whole_prompt_device_host(false).expect("host");
            let slots: usize = th.experts.iter().map(Vec::len).sum();
            let flips: usize = th.experts.iter().zip(&td.experts).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())).sum();
            let first_layer = th.experts.iter().zip(&td.experts).position(|(a, b)| a != b);
            let resid_bits: usize = th.x_out.iter().zip(&td.x_out).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()).sum();
            devhost_exact &= flips == 0 && resid_bits == 0 && th.experts.len() == td.experts.len() && !th.experts.is_empty();
            say(format!(
                "len {n} EXPERTS host vs device routing: {} layers, {slots} expert slots, flips {flips}, first differing layer {:?}; residual stream bits differ {resid_bits}",
                th.experts.len(),
                first_layer
            ));
            rows.push(serde_json::json!({"len": n, "experts_layers": th.experts.len(), "expert_slots": slots, "expert_flips": flips, "first_flip_layer": first_layer, "residual_bits_differ": resid_bits}));
        }
    }
    if cached {
        // Phase B: the whole-prompt GEMMs read a GPU-cached copy of the weights (the OFF runs above are the baseline).
        engine.set_whole_prompt(true).expect("on");
        let t = Instant::now();
        let bytes = engine.whole_prompt_cache_weights().expect("cached weight copy");
        say(format!("CACHED copy {:.2} GB in {:.2} s, mem_available {:.1} GB", bytes as f64 / 1e9, t.elapsed().as_secs_f64(), mem_available_gb()));
        for (n, off_tokens, off_logits, off_ttft) in &offs {
            let ids = prompt_ids(&tok, *n);
            for rep in 0..reps {
                let on = run(&mut engine, &ids, new_tokens);
                let st = engine.whole_prompt_stats().cloned().unwrap_or_default();
                let diff = off_tokens.iter().zip(&on.tokens).position(|(a, b)| a != b);
                let cos = cosine(off_logits, &on.logits0);
                let maxd = off_logits.iter().zip(&on.logits0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                all_identical &= diff.is_none();
                say(format!(
                    "len {n} rep {rep} CACHED: TTFT off {off_ttft:.1} ms on {:.1} ms (x{:.2}); prefill_ms on {}; tokens {} ; logits0 cosine {cos:.7} maxdiff {maxd:.3e}; launches {} chip(GEMM+quant) {:.1} ms attention {:.1} ms host {:.1} ms",
                    on.ttft_ms,
                    off_ttft / on.ttft_ms,
                    on.prefill_ms,
                    diff.map_or("IDENTICAL".to_string(), |p| format!("DIFFER at {p}")),
                    st.launches(),
                    st.chip_ns as f64 / 1e6,
                    st.attn_ns as f64 / 1e6,
                    st.host_ns as f64 / 1e6
                ));
                rows.push(serde_json::json!({"len": n, "rep": rep, "cached": true, "ttft_ms_on": on.ttft_ms, "prefill_ms_on": on.prefill_ms, "tokens_on": on.tokens, "first_diff": diff, "logits0_cosine": cos, "logits0_maxdiff": maxd,
                    "on_stats": {"launches": st.launches(), "chip_ns": st.chip_ns, "attn_ns": st.attn_ns, "host_ns": st.host_ns}}));
            }
        }
    }
    say(format!("SUMMARY tokens {}", if all_identical { "IDENTICAL" } else { "DIFFER" }));
    if devhost {
        say(format!("SUMMARY devhost vs host routing+combine {}", if devhost_exact { "BIT-IDENTICAL" } else { "DIFFER" }));
    }
    let doc = serde_json::json!({"checkpoint": checkpoint, "lens": lens, "new_tokens": new_tokens, "reps": reps, "rows": rows});
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&doc).unwrap()).expect("run.json");
    std::fs::write(receipt.join("run.txt"), log.join("\n") + "\n").expect("run.txt");
}
