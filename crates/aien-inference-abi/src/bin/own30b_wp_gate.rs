//! OWN-30B whole-prompt prefill correctness gate (W-L1G-C, 2026-10-14): is the ON path's token
//! drift activation rounding or a bug?
//!
//!   own30b_wp_gate <checkpoint> <receipt-dir> [--lens 512,2048] [--ops-layer 0] [--mutants]
//!                  (W-RC: --lens 1 is own30b_check p1 "Hello"; other short lens take the preamble head)
//!                  [--max-context 8192] [--force-memory]
//!                  [--devhost]  (W-L1G-E: ON and the mutants run with the router logits + combine on the chip)
//!
//! Loads the FP8 model once (fused attention). Per prompt length, three runs of the same prompt
//! from an empty KV, each recording every layer's residual after attention (`x_mid`) and after
//! the experts (`x_out`), the chosen experts, and every op of `--ops-layer`:
//!   OFF   the engine's per-token OFF kernels, f32 activations (`control_step`, mode Off);
//!   CTRL  the same, with the ON path's activation quantiser at every GEMM input (mode W8A8);
//!   ON    the whole-prompt path (`set_whole_prompt(true)`), recorder installed.
//! Then per layer the relative error (L2 over all tokens) of ON vs OFF, CTRL vs OFF and ON vs
//! CTRL, the expert-set flips, the first-step logits, and the per-op errors at `--ops-layer`.
//!
//! Pass rule (fixed before the ON numbers were looked at; see the receipt's NOTES):
//!   B = max over layers of relerr(CTRL, OFF) on x_out: the rounding budget, measured by the
//!       control (first-order e4m3 activation rounding plus the routing flips it causes).
//!   PASS iff max over layers of relerr(ON, CTRL) on x_out <= B, and at `--ops-layer` (0: the
//!   layer whose inputs are identical) relerr(ON, CTRL) <= relerr(CTRL, OFF) for qkv, oproj and
//!   moe_out (the three GEMM-fed outputs with a non-zero budget).
//!   The negative controls (`--mutants`: ON with the routing mutant, ON with the gate|up scale
//!   mutant) must FAIL that rule (their max relerr vs CTRL > B).
//! Writes run.json (every per-layer number) and run.txt into the receipt dir.
use std::path::PathBuf;
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};

/// `--devhost` (W-L1G-E): ON runs use the device router logits + MoE combine.
static DEVHOST: AtomicBool = AtomicBool::new(false);

use aien_inference_abi::qwen3_moe_omega::w8a8_control::ControlMode;
use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeOmegaEngine, WeightFormat};
use aien_inference_abi::qwen3_moe_omega_prefill::fake_quant::fake_quant_row;
use aien_inference_abi::qwen3_moe_omega_prefill::{WpMutant, WpTrace};
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

/// The TTFT tool's prompt of exactly `n` tokens (own30b_ttft.rs, copied so the gate sees the
/// same prompt the TTFT receipts used).
fn prompt_ids(tok: &Tokenizer, n: usize) -> Vec<u32> {
    let enc = |t: &str| tok.encode_with_special(t, false).expect("encode");
    let mut ids = enc("<|im_start|>user\nYou are a careful engineering assistant. Read the log below and say, in plain words, which block number has the largest decode time.\n\n");
    let tail = enc("\nAnswer with the block number only.<|im_end|>\n<|im_start|>assistant\n");
    // W-RC: short prompts (what the server prefills after prefix reuse). n = 1 is own30b_check's
    // p1 ("Hello", the prompt whose ON token 0 differs); 2.. are the first n tokens of the preamble.
    if n == 1 {
        let one = enc("Hello");
        assert_eq!(one.len(), 1, "\"Hello\" is one token");
        return one;
    }
    if n <= ids.len() + tail.len() + 8 {
        assert!(n <= ids.len(), "prompt length {n} too short for the preamble");
        ids.truncate(n);
        return ids;
    }
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
    name: String,
    trace: WpTrace,
    logits0: Vec<f32>,
    secs: f64,
}

/// Relative L2 error of `a` against the reference `b`, and the largest per-row relative error.
fn relerr(a: &[f32], b: &[f32], row: usize) -> (f64, f64) {
    let (e, w, _) = relerr_idx(a, b, row);
    (e, w)
}

/// As [`relerr`], plus the index of the worst row.
fn relerr_idx(a: &[f32], b: &[f32], row: usize) -> (f64, f64, usize) {
    assert_eq!(a.len(), b.len());
    let (mut d2, mut b2, mut worst, mut wi) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    if row > 0 {
        for (ri, (ra, rb)) in a.chunks(row).zip(b.chunks(row)).enumerate() {
            let (mut rd, mut rr) = (0.0f64, 0.0f64);
            for (x, y) in ra.iter().zip(rb) {
                let d = *x as f64 - *y as f64;
                rd += d * d;
                rr += *y as f64 * *y as f64;
            }
            d2 += rd;
            b2 += rr;
            if rr > 0.0 && (rd / rr).sqrt() > worst {
                worst = (rd / rr).sqrt();
                wi = ri;
            }
        }
    }
    (if b2 > 0.0 { (d2 / b2).sqrt() } else { 0.0 }, worst, wi)
}

/// Relative L2 error over only the rows whose expert sets agree (`same` marks them): the GEMM
/// outputs of tokens whose routing did not flip, i.e. the GEMMs alone, without routing effects.
fn relerr_rows(a: &[f32], b: &[f32], row: usize, same: &[bool]) -> (f64, usize) {
    let (mut d2, mut b2, mut n) = (0.0f64, 0.0f64, 0usize);
    for (ri, (ra, rb)) in a.chunks(row).zip(b.chunks(row)).enumerate() {
        if !same[ri] {
            continue;
        }
        n += 1;
        for (x, y) in ra.iter().zip(rb) {
            let d = *x as f64 - *y as f64;
            d2 += d * d;
            b2 += *y as f64 * *y as f64;
        }
    }
    (if b2 > 0.0 { (d2 / b2).sqrt() } else { 0.0 }, n)
}

/// Per token: the two expert sets are equal.
fn same_route(a: &[u32], b: &[u32], top_k: usize) -> Vec<bool> {
    a.chunks(top_k).zip(b.chunks(top_k)).map(|(ra, rb)| ra.iter().all(|x| rb.contains(x))).collect()
}

/// Tokens whose expert SET differs, and the number of ids of `a` not in `b`'s set for that token.
fn flips(a: &[u32], b: &[u32], top_k: usize) -> (usize, usize) {
    let (mut tokens, mut ids) = (0, 0);
    for (ra, rb) in a.chunks(top_k).zip(b.chunks(top_k)) {
        let missing = ra.iter().filter(|x| !rb.contains(x)).count();
        if missing > 0 {
            tokens += 1;
            ids += missing;
        }
    }
    (tokens, ids)
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

fn argmax(l: &[f32]) -> usize {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap_or(0)
}

fn control_run(engine: &mut Qwen3MoeOmegaEngine, name: &str, ids: &[u32], mode: ControlMode, ops_layer: usize) -> Run {
    let c = engine.weights().config.clone();
    engine.set_whole_prompt(false).expect("off");
    engine.reset();
    let mut trace = WpTrace::new(Some(ops_layer));
    trace.begin(ids.len(), c.layers, c.hidden, c.top_k);
    let t0 = Instant::now();
    let mut hidden = Vec::new();
    for (row, &tok) in ids.iter().enumerate() {
        hidden = engine.control_step(tok, mode, WpMutant::None, &mut trace, row).unwrap_or_else(|e| panic!("{name} token {row}: {e}"));
    }
    let logits0 = engine.logits(&hidden).expect("logits");
    Run { name: name.to_string(), trace, logits0, secs: t0.elapsed().as_secs_f64() }
}

fn on_run(engine: &mut Qwen3MoeOmegaEngine, name: &str, ids: &[u32], mutant: WpMutant, ops_layer: usize) -> Run {
    engine.set_whole_prompt(true).expect("whole-prompt prefill refused");
    engine.set_whole_prompt_device_host(DEVHOST.load(Ordering::Relaxed)).expect("device router / combine refused");
    engine.whole_prompt_set_mutant(mutant).expect("mutant");
    engine.whole_prompt_set_trace(Some(WpTrace::new(Some(ops_layer)))).expect("trace");
    engine.reset();
    let mut logits0 = Vec::new();
    let t0 = Instant::now();
    engine
        .generate_with(
            ids,
            &GenerateParams::greedy(1, Vec::new()),
            &mut |logits, step| {
                if step == 0 {
                    logits0 = logits.to_vec();
                }
                aien_inference_abi::tensor::sample_argmax(logits).0
            },
            &mut |_| true,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let secs = t0.elapsed().as_secs_f64();
    let trace = engine.whole_prompt_take_trace().expect("recorder");
    engine.whole_prompt_set_mutant(WpMutant::None).expect("mutant off");
    assert_eq!(trace.m, ids.len(), "{name}: the whole-prompt path did not run (recorder empty)");
    let st = engine.whole_prompt_stats().cloned().unwrap_or_default();
    let want = if DEVHOST.load(Ordering::Relaxed) { st.layers } else { 0 };
    assert!(st.router == want && st.combine == want, "{name}: device router / combine launches {} / {}, want {want} each", st.router, st.combine);
    println!("{name}: routing path {} (router launches {}, combine launches {})", if want > 0 { "DEVICE" } else { "HOST" }, st.router, st.combine);
    Run { name: name.to_string(), trace, logits0, secs }
}

/// Per-layer comparison of `a` against `b`: x_mid / x_out relative errors, worst token, flips.
fn compare_layers(a: &Run, b: &Run) -> Vec<serde_json::Value> {
    let (h, k) = (a.trace.hidden, a.trace.top_k);
    (0..a.trace.layers())
        .map(|l| {
            let (mid, mid_worst, mid_wi) = relerr_idx(&a.trace.x_mid[l], &b.trace.x_mid[l], h);
            let (out, out_worst, out_wi) = relerr_idx(&a.trace.x_out[l], &b.trace.x_out[l], h);
            let (ft, fi) = flips(&a.trace.experts[l], &b.trace.experts[l], k);
            let same = same_route(&a.trace.experts[l], &b.trace.experts[l], k);
            let (out_same, n_same) = relerr_rows(&a.trace.x_out[l], &b.trace.x_out[l], h, &same);
            serde_json::json!({"layer": l, "mid": mid, "mid_worst_token": mid_worst, "mid_worst_idx": mid_wi, "out": out, "out_worst_token": out_worst, "out_worst_idx": out_wi, "flip_tokens": ft, "flip_ids": fi, "out_same_route": out_same, "same_route_tokens": n_same})
        })
        .collect()
}

fn compare_ops(a: &Run, b: &Run) -> serde_json::Value {
    let m = a.trace.m;
    let one = |x: &[f32], y: &[f32]| if x.is_empty() || x.len() != y.len() { serde_json::Value::Null } else { serde_json::json!(relerr(x, y, x.len() / m).0) };
    let (oa, ob) = (&a.trace.ops, &b.trace.ops);
    let l = a.trace.ops_layer.unwrap_or(0);
    let same = same_route(&a.trace.experts[l], &b.trace.experts[l], a.trace.top_k);
    let (moe_same, n_same) = if oa.moe_out.len() == ob.moe_out.len() && !oa.moe_out.is_empty() { relerr_rows(&oa.moe_out, &ob.moe_out, oa.moe_out.len() / m, &same) } else { (f64::NAN, 0) };
    // How far apart the two runs' QUANTISED GEMM inputs are (host quantiser on the recorded pre-quant rows):
    // the rounding-decision flips a small input difference causes, the floor of any GEMM-output comparison.
    let quantised = |x: &[f32], y: &[f32]| {
        if x.is_empty() || x.len() != y.len() {
            return serde_json::Value::Null;
        }
        let (mut qx, mut qy) = (x.to_vec(), y.to_vec());
        let k = x.len() / m;
        qx.chunks_mut(k).for_each(fake_quant_row);
        qy.chunks_mut(k).for_each(fake_quant_row);
        serde_json::json!(relerr(&qx, &qy, k).0)
    };
    serde_json::json!({
        "moe_out_same_route": moe_same, "same_route_tokens": n_same,
        "post_quantised": quantised(&oa.post, &ob.post), "attn_quantised": quantised(&oa.attn, &ob.attn), "x_norm_quantised": quantised(&oa.x_norm, &ob.x_norm),
        "x_norm": one(&oa.x_norm, &ob.x_norm), "qkv": one(&oa.qkv, &ob.qkv), "attn": one(&oa.attn, &ob.attn), "oproj": one(&oa.oproj, &ob.oproj),
        "post": one(&oa.post, &ob.post), "logits": one(&oa.logits, &ob.logits), "moe_out": one(&oa.moe_out, &ob.moe_out),
    })
}

fn max_out(layers: &[serde_json::Value]) -> f64 {
    layers.iter().map(|v| v["out"].as_f64().unwrap_or(0.0)).fold(0.0, f64::max)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = PathBuf::from(args.get(1).expect("checkpoint dir"));
    let receipt = PathBuf::from(args.get(2).expect("receipt dir"));
    let (mut lens, mut ops_layer, mut mutants, mut max_context, mut force) = (vec![512usize, 2048], 0usize, false, 8192usize, false);
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lens" => lens = it.next().expect("--lens").split(',').map(|s| s.parse().expect("len")).collect(),
            "--ops-layer" => ops_layer = it.next().expect("--ops-layer").parse().expect("n"),
            "--mutants" => mutants = true,
            "--devhost" => DEVHOST.store(true, Ordering::Relaxed),
            "--max-context" => max_context = it.next().expect("--max-context").parse().expect("n"),
            "--force-memory" => force = true,
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
        "build native={} pf={} pattn={} pattn_dev={}",
        aien_omega_gpu::is_native(),
        aien_omega_pf::is_native(),
        aien_omega_pf::has_attention(),
        aien_omega_pf::has_attention_dev()
    ));
    let tok = Tokenizer::from_file(checkpoint.join("tokenizer.json")).expect("tokenizer");
    let t0 = Instant::now();
    let (mut engine, report) =
        Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, max_context, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {})
            .expect("load on Omega");
    say(format!("LOAD {:.1}s resident {:.2} GB attention {} mem_available_after {:.1} GB", t0.elapsed().as_secs_f64(), report.resident_bytes as f64 / 1e9, engine.attention_path(), mem_available_gb()));

    let mut docs = Vec::new();
    let mut all_pass = true;
    for &n in &lens {
        let ids = prompt_ids(&tok, n);
        let off = control_run(&mut engine, "OFF", &ids, ControlMode::Off, ops_layer);
        say(format!("len {n} OFF  (per-token OFF kernels, f32 activations): {:.1} s", off.secs));
        let ctrl = control_run(&mut engine, "CTRL", &ids, ControlMode::W8A8, ops_layer);
        say(format!("len {n} CTRL (OFF kernels + the ON quantiser at every GEMM input): {:.1} s", ctrl.secs));
        let on = on_run(&mut engine, "ON", &ids, WpMutant::None, ops_layer);
        say(format!("len {n} ON   (whole-prompt path): {:.1} s", on.secs));
        let mut runs: Vec<(&str, &Run, &Run, Vec<serde_json::Value>, serde_json::Value)> = Vec::new();
        for (a, b) in [(&on, &off), (&ctrl, &off), (&on, &ctrl)] {
            let name = format!("{}_vs_{}", a.name, b.name);
            let layers = compare_layers(a, b);
            let ops = compare_ops(a, b);
            runs.push((Box::leak(name.into_boxed_str()), a, b, layers, ops));
        }
        say(format!("len {n} per-layer relative error on x_out (L2 over all {n} tokens) | expert-set flips (tokens/ids):"));
        say(format!("{:>5} {:>12} {:>12} {:>12} | {:>11} {:>11} {:>11}", "layer", "ON-OFF", "CTRL-OFF", "ON-CTRL", "ON-OFF", "CTRL-OFF", "ON-CTRL"));
        for l in 0..off.trace.layers() {
            let v = |i: usize, key: &str| runs[i].3[l][key].as_f64().unwrap_or(0.0);
            let f = |i: usize| format!("{}/{}", runs[i].3[l]["flip_tokens"], runs[i].3[l]["flip_ids"]);
            say(format!("{l:>5} {:>12.3e} {:>12.3e} {:>12.3e} | {:>11} {:>11} {:>11}", v(0, "out"), v(1, "out"), v(2, "out"), f(0), f(1), f(2)));
        }
        say(format!("len {n} ops at layer {ops_layer} (relative error): {}", ["ON-OFF", "CTRL-OFF", "ON-CTRL"].iter().zip(&runs).map(|(nm, r)| format!("{nm} {}", r.4)).collect::<Vec<_>>().join(" ; ")));
        let (c_on_off, c_ctrl_off, c_on_ctrl) = (cosine(&on.logits0, &off.logits0), cosine(&ctrl.logits0, &off.logits0), cosine(&on.logits0, &ctrl.logits0));
        say(format!(
            "len {n} first-step logits cosine: ON-OFF {c_on_off:.6} CTRL-OFF {c_ctrl_off:.6} ON-CTRL {c_on_ctrl:.6}; top-1 OFF {} CTRL {} ON {}",
            argmax(&off.logits0),
            argmax(&ctrl.logits0),
            argmax(&on.logits0)
        ));
        let budget = max_out(&runs[1].3);
        let on_ctrl = max_out(&runs[2].3);
        let ops_ok: Vec<(String, bool)> = ["qkv", "oproj", "moe_out"]
            .iter()
            .map(|k| {
                let (a, b) = (runs[2].4[*k].as_f64().unwrap_or(f64::NAN), runs[1].4[*k].as_f64().unwrap_or(f64::NAN));
                (format!("{k} ON-CTRL {a:.3e} <= CTRL-OFF {b:.3e}"), a <= b)
            })
            .collect();
        let pass = on_ctrl <= budget && ops_ok.iter().all(|o| o.1);
        all_pass &= pass;
        say(format!("len {n} RULE: budget B = max_l relerr(CTRL,OFF) = {budget:.3e}; max_l relerr(ON,CTRL) = {on_ctrl:.3e} ({:.2} x B); ops: {}", on_ctrl / budget, ops_ok.iter().map(|o| format!("{} {}", o.0, if o.1 { "ok" } else { "VIOLATED" })).collect::<Vec<_>>().join(", ")));
        say(format!("len {n} VERDICT: {}", if pass { "PASS (ON within the control's rounding budget: drift = activation rounding)" } else { "FAIL (ON differs from the W8A8 control beyond the rounding budget)" }));
        let mut doc = serde_json::json!({
            "len": n, "ops_layer": ops_layer, "secs": {"off": off.secs, "ctrl": ctrl.secs, "on": on.secs},
            "layers": runs.iter().map(|r| (r.0, &r.3)).collect::<std::collections::BTreeMap<_, _>>(),
            "ops": runs.iter().map(|r| (r.0, &r.4)).collect::<std::collections::BTreeMap<_, _>>(),
            "logits0": {"cos_on_off": c_on_off, "cos_ctrl_off": c_ctrl_off, "cos_on_ctrl": c_on_ctrl, "top1_off": argmax(&off.logits0), "top1_ctrl": argmax(&ctrl.logits0), "top1_on": argmax(&on.logits0)},
            "rule": {"budget": budget, "on_ctrl": on_ctrl, "ratio": on_ctrl / budget, "ops": ops_ok, "pass": pass},
        });
        if mutants {
            let mut mres = Vec::new();
            for (mutant, name) in [(WpMutant::Routing, "MUT_ROUTING"), (WpMutant::GateUpScales, "MUT_GUSCALES")] {
                let mu = on_run(&mut engine, name, &ids, mutant, ops_layer);
                let layers = compare_layers(&mu, &ctrl);
                let worst = max_out(&layers);
                let caught = worst > budget;
                all_pass &= caught;
                let l0 = &layers[0];
                say(format!(
                    "len {n} NEGATIVE CONTROL {name} vs CTRL: max_l relerr {worst:.3e} ({:.1} x B), layer 0 out {:.3e} flips {}/{}, logits cosine vs OFF {:.6}: {}",
                    worst / budget,
                    l0["out"].as_f64().unwrap_or(0.0),
                    l0["flip_tokens"],
                    l0["flip_ids"],
                    cosine(&mu.logits0, &off.logits0),
                    if caught { "CAUGHT (rule fails it)" } else { "NOT CAUGHT: the rule is too loose" }
                ));
                mres.push(serde_json::json!({"mutant": name, "max_out_vs_ctrl": worst, "ratio": worst / budget, "caught": caught, "layers": layers, "ops_vs_ctrl": compare_ops(&mu, &ctrl), "cos_vs_off": cosine(&mu.logits0, &off.logits0)}));
            }
            doc["mutants"] = serde_json::Value::Array(mres);
        }
        docs.push(doc);
    }
    say(format!("SUMMARY {}", if all_pass { "PASS" } else { "FAIL" }));
    let out = serde_json::json!({"checkpoint": checkpoint, "lens": lens, "ops_layer": ops_layer, "mutants": mutants, "runs": docs});
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&out).unwrap()).expect("run.json");
    std::fs::write(receipt.join("run.txt"), log.join("\n") + "\n").expect("run.txt");
}
