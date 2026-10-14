//! OWN-30B W-L1I drift split (2026-10-14): what is left of the whole-prompt (W8A16) drift
//! against plain decode, and how much each source contributes.
//!
//!   own30b_drift_split <checkpoint> <receipt-dir> [--lens 512,2048] [--ops-layer 0] [--max-context 4096]
//!                      [--variants OFF,ULP,...] [--devhost] [--min-mem-gb 70]
//!
//! Method (the W8A8-control style of W-L1G-C, one variable at a time): every variant is the OFF
//! path (per-token decode kernels, f32 activations, `control_step_emul`) with ONE rounding of the
//! ON path emulated, plus the ON path itself; each is compared with OFF per layer (x_out relative
//! L2 error over all tokens, expert-set flips) and at `--ops-layer` per op.
//!
//!   OFF        the reference (plain decode)
//!   ULP        FLOOR CONTROL: OFF with a 1-ulp nudge of token 0's embedding (6e-8 relative)
//!   HATT       host attention emulator, f32 Q/P, one global max   (emulator floor vs the chip)
//!   HATT32     host attention, f32 Q/P, online softmax over 32-key blocks (the prefill's order)
//!   HQ         HATT32 + q' rounded to bf16
//!   HP         HATT32 + p rounded to bf16 in O += P V (l from f32 p)
//!   HQP        HATT32 + both (the prefill attention's numerics)
//!   MUTQS      NEGATIVE CONTROL: HATT32 with the score scale x1.02 (a real bug, must be caught)
//!   A16        bf16 at all four GEMV inputs (the W8A16 GEMM activation rounding), chip attention
//!   A16QKV / A16O / A16GU / A16DOWN   the same at one input point
//!   FULL       A16 + HQP: the full emulation of ON16
//!   FIXQ       FORECAST: FULL with f32 q' (what an f32-Q prefill attention would leave)
//!   FIXQKV     FORECAST: FIXQ with f32 q|k|v GEMM inputs too (f32 Q + f32 activations into the qkv GEMM)
//!   ON16       the whole-prompt path, W8A16
//!   ON8        the whole-prompt path, W8A8 (reference)
//!
//! Pre-registered rules (fixed before any number was seen; plan.md of the investigation):
//!   R1 floor   layer 0 x_out: relerr(ULP,OFF) and relerr(HATT32,OFF) <= 0.1 x relerr(ON16,OFF).
//!   R2 teeth   MUTQS attention-op error vs the chip >= 10 x HQP's, every layer.
//!   R3 repro   layer 0 x_out: relerr(ON16,FULL) <= 0.1 x relerr(ON16,OFF) (FULL reproduces ON16).
//!   Shares     layer 0 x_out: share_X = e_X^2 / sum e^2 over {A16, HQ, HP, HATT32}, e_X = relerr(X,OFF)
//!              (independent errors add in squares; additivity checked: e_FULL^2 vs the sum).
//!   Depth      per layer, relerr(X,OFF) for every variant; where ULP's curve reaches half of
//!              ON16's, the max-over-layers metric is declared saturated (routing chaos).
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::host_attn::{HostAttn, HostAttnMode};
use aien_inference_abi::qwen3_moe_omega::w8a8_control::{ActRound, Emul, PT_ALL, PT_DOWN, PT_GU, PT_O, PT_QKV};
use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeOmegaEngine, WeightFormat};
use aien_inference_abi::qwen3_moe_omega_prefill::{PrefillActs, WpMutant, WpTrace};
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;

const ALL: &str = "OFF,ULP,HATT,HATT32,HQ,HP,HQP,MUTQS,A16,A16QKV,A16O,A16GU,A16DOWN,FULL,FIXQ,FIXQKV,ON16,ON8";

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

/// own30b_wp_gate's prompt of exactly `n` tokens (copied: the gate and this split see the same prompt).
fn prompt_ids(tok: &Tokenizer, n: usize) -> Vec<u32> {
    let enc = |t: &str| tok.encode_with_special(t, false).expect("encode");
    let mut ids = enc("<|im_start|>user\nYou are a careful engineering assistant. Read the log below and say, in plain words, which block number has the largest decode time.\n\n");
    let tail = enc("\nAnswer with the block number only.<|im_end|>\n<|im_start|>assistant\n");
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
    trace: WpTrace,
    logits0: Vec<f32>,
    secs: f64,
    /// Host attention op error vs the chip kernel on the same inputs, per layer (host variants).
    attn_vs_chip: Option<Vec<f64>>,
}

fn relerr(a: &[f32], b: &[f32]) -> f64 {
    let (mut d2, mut b2) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        d2 += d * d;
        b2 += *y as f64 * *y as f64;
    }
    if b2 > 0.0 { (d2 / b2).sqrt() } else { 0.0 }
}

/// Median and max of the per-token relative errors (rows of `h`), and the worst token's index:
/// tells whether an L2 number is carried by a few outlier tokens or by all of them.
fn per_token(a: &[f32], b: &[f32], h: usize) -> (f64, f64, usize) {
    let mut v: Vec<(f64, usize)> = a.chunks(h).zip(b.chunks(h)).enumerate().map(|(i, (x, y))| (relerr(x, y), i)).collect();
    let worst = v.iter().copied().fold((0.0, 0), |m, e| if e.0 > m.0 { e } else { m });
    v.sort_by(|p, q| p.0.total_cmp(&q.0));
    (v[v.len() / 2].0, worst.0, worst.1)
}

/// relerr over only the tokens whose expert sets agree in both runs (routing flips removed).
fn relerr_same_route(a: &[f32], b: &[f32], h: usize, ea: &[u32], eb: &[u32], k: usize) -> f64 {
    let (mut d2, mut b2) = (0.0f64, 0.0f64);
    for ((x, y), (ra, rb)) in a.chunks(h).zip(b.chunks(h)).zip(ea.chunks(k).zip(eb.chunks(k))) {
        if ra.iter().any(|e| !rb.contains(e)) {
            continue;
        }
        for (p, q) in x.iter().zip(y) {
            d2 += (*p as f64 - *q as f64).powi(2);
            b2 += (*q as f64).powi(2);
        }
    }
    if b2 > 0.0 { (d2 / b2).sqrt() } else { 0.0 }
}

fn flip_tokens(a: &[u32], b: &[u32], k: usize) -> usize {
    a.chunks(k).zip(b.chunks(k)).filter(|(ra, rb)| ra.iter().any(|x| !rb.contains(x))).count()
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

/// The emulation and host-attention mode of a control variant; None for ON variants.
fn variant(name: &str) -> Option<(Emul, Option<HostAttnMode>)> {
    let f32a = HostAttnMode::f32();
    let on32 = HostAttnMode { online32: true, ..f32a };
    let a16 = |pts: u8| Emul { acts: ActRound::Bf16, act_points: pts, ..Emul::default() };
    let host = |m: HostAttnMode| (Emul { host_attn: true, ..Emul::default() }, Some(m));
    Some(match name {
        "OFF" => (Emul::default(), None),
        "ULP" => (Emul { ulp_nudge: true, ..Emul::default() }, None),
        "HATT" => host(f32a),
        "HATT32" => host(on32),
        "HQ" => host(HostAttnMode { q_bf16: true, ..on32 }),
        "HP" => host(HostAttnMode { p_bf16: true, ..on32 }),
        "HQP" => host(HostAttnMode { q_bf16: true, p_bf16: true, ..on32 }),
        "MUTQS" => host(HostAttnMode { qk_scale_mul: 1.02, ..on32 }),
        "A16" => (a16(PT_ALL), None),
        "A16QKV" => (a16(PT_QKV), None),
        "A16O" => (a16(PT_O), None),
        "A16GU" => (a16(PT_GU), None),
        "A16DOWN" => (a16(PT_DOWN), None),
        "FULL" => (Emul { host_attn: true, ..a16(PT_ALL) }, Some(HostAttnMode { q_bf16: true, p_bf16: true, ..on32 })),
        "FIXQ" => (Emul { host_attn: true, ..a16(PT_ALL) }, Some(HostAttnMode { p_bf16: true, ..on32 })),
        "FIXQKV" => (Emul { host_attn: true, ..a16(PT_O | PT_GU | PT_DOWN) }, Some(HostAttnMode { p_bf16: true, ..on32 })),
        "ON16" | "ON8" => return None,
        o => panic!("unknown variant {o} (known: {ALL})"),
    })
}

fn control_run(engine: &mut Qwen3MoeOmegaEngine, name: &str, ids: &[u32], emul: Emul, hmode: Option<HostAttnMode>, ops_layer: usize) -> Run {
    let c = engine.weights().config.clone();
    engine.set_whole_prompt(false).expect("off");
    engine.reset();
    let mut trace = WpTrace::new(Some(ops_layer));
    trace.begin(ids.len(), c.layers, c.hidden, c.top_k);
    let mut host = hmode.map(|m| HostAttn::new(m, c.layers, c.q_heads, c.kv_heads, c.head_dim, c.rope_theta));
    let t0 = Instant::now();
    let mut hidden = Vec::new();
    for (row, &tok) in ids.iter().enumerate() {
        hidden = engine.control_step_emul(tok, &emul, host.as_mut(), WpMutant::None, &mut trace, row).unwrap_or_else(|e| panic!("{name} token {row}: {e}"));
    }
    let logits0 = engine.logits(&hidden).expect("logits");
    Run { trace, logits0, secs: t0.elapsed().as_secs_f64(), attn_vs_chip: host.map(|h| h.chip_relerr()) }
}

fn on_run(engine: &mut Qwen3MoeOmegaEngine, name: &str, ids: &[u32], acts: PrefillActs, devhost: bool, ops_layer: usize) -> Run {
    engine.set_whole_prompt(false).expect("off");
    engine.set_whole_prompt(true).expect("whole-prompt prefill refused");
    engine.set_whole_prompt_device_host(devhost).expect("device router / combine refused");
    engine.set_whole_prompt_acts(acts).expect("acts refused");
    engine.whole_prompt_set_mutant(WpMutant::None).expect("mutant");
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
    assert_eq!(trace.m, ids.len(), "{name}: the whole-prompt path did not run (recorder empty)");
    Run { trace, logits0, secs, attn_vs_chip: None }
}

fn op(a: &Run, b: &Run, key: &str) -> f64 {
    let (oa, ob) = (&a.trace.ops, &b.trace.ops);
    let (x, y) = match key {
        "qkv" => (&oa.qkv, &ob.qkv),
        "attn" => (&oa.attn, &ob.attn),
        "oproj" => (&oa.oproj, &ob.oproj),
        "logits" => (&oa.logits, &ob.logits),
        "moe_out" => (&oa.moe_out, &ob.moe_out),
        _ => unreachable!(),
    };
    if x.is_empty() || x.len() != y.len() { f64::NAN } else { relerr(x, y) }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = PathBuf::from(args.get(1).expect("checkpoint dir"));
    let receipt = PathBuf::from(args.get(2).expect("receipt dir"));
    let (mut lens, mut ops_layer, mut max_context, mut min_mem, mut devhost) = (vec![512usize, 2048], 0usize, 4096usize, 70.0f64, false);
    let mut names: Vec<String> = ALL.split(',').map(String::from).collect();
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lens" => lens = it.next().expect("--lens").split(',').map(|s| s.parse().expect("len")).collect(),
            "--ops-layer" => ops_layer = it.next().expect("--ops-layer").parse().expect("n"),
            "--max-context" => max_context = it.next().expect("--max-context").parse().expect("n"),
            "--variants" => names = it.next().expect("--variants").split(',').map(String::from).collect(),
            "--min-mem-gb" => min_mem = it.next().expect("--min-mem-gb").parse().expect("gb"),
            "--devhost" => devhost = true,
            o => panic!("unknown argument {o}"),
        }
    }
    for n in &names {
        variant(n);
    }
    assert!(names.first().map(String::as_str) == Some("OFF"), "the first variant must be OFF (the reference)");
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB (minimum {min_mem})"));
    if avail < min_mem {
        say("REFUSED: not enough memory available".into());
        std::process::exit(3);
    }
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    let tok = Tokenizer::from_file(checkpoint.join("tokenizer.json")).expect("tokenizer");
    let t0 = Instant::now();
    let (mut engine, report) =
        Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, max_context, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {}).expect("load on Omega");
    let nl = engine.weights().config.layers;
    say(format!("LOAD {:.1}s layers {nl} resident {:.2} GB attention {} devhost {devhost}", t0.elapsed().as_secs_f64(), report.resident_bytes as f64 / 1e9, engine.attention_path()));

    let mut docs = Vec::new();
    for &n in &lens {
        let ids = prompt_ids(&tok, n);
        let mut runs: BTreeMap<String, Run> = BTreeMap::new();
        for name in &names {
            let r = match variant(name) {
                Some((emul, hm)) => control_run(&mut engine, name, &ids, emul, hm, ops_layer),
                None => on_run(&mut engine, name, &ids, if name == "ON16" { PrefillActs::W8A16 } else { PrefillActs::W8A8 }, devhost, ops_layer),
            };
            say(format!("len {n} {name}: {:.1} s", r.secs));
            runs.insert(name.clone(), r);
        }
        let off = &runs["OFF"];
        let k = off.trace.top_k;
        let out = |a: &str, b: &str, l: usize| relerr(&runs[a].trace.x_out[l], &runs[b].trace.x_out[l]);
        // per-layer table vs OFF
        let cols: Vec<&String> = names.iter().filter(|n| n.as_str() != "OFF").collect();
        say(format!("len {n} relerr(X,OFF) on x_out per layer [expert-set flip tokens]:"));
        say(format!("{:>5} {}", "layer", cols.iter().map(|c| format!("{c:>16}")).collect::<String>()));
        let mut per_layer = BTreeMap::new();
        for l in 0..nl {
            let mut line = format!("{l:>5}");
            for c in &cols {
                let e = out(c, "OFF", l);
                let f = flip_tokens(&runs[c.as_str()].trace.experts[l], &off.trace.experts[l], k);
                line += &format!(" {:>10.3e} [{f:>3}]", e);
                let (med, worst, wi) = per_token(&runs[c.as_str()].trace.x_out[l], &off.trace.x_out[l], off.trace.hidden);
                per_layer.entry(c.to_string()).or_insert_with(Vec::new).push(serde_json::json!({"out": e, "mid": relerr(&runs[c.as_str()].trace.x_mid[l], &off.trace.x_mid[l]), "flips": f, "token_median": med, "token_worst": worst, "token_worst_idx": wi}));
            }
            say(line);
        }
        say(format!("len {n} per-token relerr(X,OFF) on x_out: median / worst@token, per layer:"));
        for l in 0..nl {
            let mut line = format!("{l:>5}");
            for c in &cols {
                let (med, worst, wi) = per_token(&runs[c.as_str()].trace.x_out[l], &off.trace.x_out[l], off.trace.hidden);
                line += &format!(" {c} {med:.2e}/{worst:.2e}@{wi}");
            }
            say(line);
        }
        // attention op error vs the chip kernel, identical inputs, per layer
        let mut attn_chip = BTreeMap::new();
        for c in &cols {
            if let Some(v) = &runs[c.as_str()].attn_vs_chip {
                say(format!("len {n} {c} attention op vs chip decode kernel (same inputs) per layer: {}", v.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" ")));
                attn_chip.insert(c.to_string(), v.clone());
            }
        }
        // ops at ops_layer vs OFF
        let mut ops = BTreeMap::new();
        for c in &cols {
            let r = &runs[c.as_str()];
            let v: BTreeMap<&str, f64> = ["qkv", "attn", "oproj", "logits", "moe_out"].iter().map(|kk| (*kk, op(r, off, kk))).collect();
            say(format!("len {n} ops layer {ops_layer} {c} vs OFF: {}", v.iter().map(|(a, b)| format!("{a} {b:.3e}")).collect::<Vec<_>>().join(", ")));
            ops.insert(c.to_string(), v);
        }
        let mut doc = serde_json::json!({"len": n, "layers": nl, "per_layer_vs_off": per_layer, "attn_vs_chip": attn_chip, "ops_vs_off": ops,
            "secs": names.iter().map(|nm| (nm.clone(), runs[nm].secs)).collect::<BTreeMap<_, _>>()});
        let has = |v: &str| runs.contains_key(v);
        let l0 = |v: &str| out(v, "OFF", ops_layer);
        if has("ON16") {
            let on = l0("ON16");
            let mut rules = serde_json::Map::new();
            if has("ULP") && has("HATT32") {
                let (u, h) = (l0("ULP"), l0("HATT32"));
                let ok = u <= 0.1 * on && h <= 0.1 * on;
                say(format!("len {n} R1 floor: layer {ops_layer} ULP {u:.3e} HATT32 {h:.3e} vs 0.1 x ON16 {:.3e}: {}", 0.1 * on, if ok { "PASS" } else { "FAIL" }));
                rules.insert("R1".into(), serde_json::json!({"ulp": u, "hatt32": h, "on16": on, "pass": ok}));
            }
            if let (Some(m), Some(q)) = (runs.get("MUTQS").and_then(|r| r.attn_vs_chip.clone()), runs.get("HQP").and_then(|r| r.attn_vs_chip.clone())) {
                let ok = m.iter().zip(&q).all(|(a, b)| *a >= 10.0 * *b);
                say(format!("len {n} R2 teeth: MUTQS attention error >= 10 x HQP's at every layer: {}", if ok { "PASS (mutant caught)" } else { "FAIL" }));
                rules.insert("R2".into(), serde_json::json!({"pass": ok}));
            }
            if has("FULL") {
                let r = out("ON16", "FULL", ops_layer);
                let ok = r <= 0.1 * on;
                say(format!("len {n} R3 repro: layer {ops_layer} relerr(ON16,FULL) {r:.3e} vs 0.1 x relerr(ON16,OFF) {:.3e}: {}", 0.1 * on, if ok { "PASS" } else { "FAIL" }));
                let per: Vec<f64> = (0..nl).map(|l| out("ON16", "FULL", l)).collect();
                say(format!("len {n} relerr(ON16,FULL) per layer: {}", per.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" ")));
                let (a, b) = (&runs["ON16"].trace, &runs["FULL"].trace);
                let same: Vec<f64> = (0..nl).map(|l| relerr_same_route(&a.x_out[l], &b.x_out[l], a.hidden, &a.experts[l], &b.experts[l], k)).collect();
                let fl: Vec<usize> = (0..nl).map(|l| flip_tokens(&a.experts[l], &b.experts[l], k)).collect();
                say(format!("len {n} relerr(ON16,FULL) same-route tokens only per layer: {} (ON16-vs-FULL flip tokens {:?})", same.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" "), fl));
                rules.insert("R3".into(), serde_json::json!({"on16_full": r, "on16_off": on, "per_layer": per, "pass": ok}));
            }
            let parts = ["A16", "HQ", "HP", "HATT32"];
            if parts.iter().all(|p| has(p)) {
                let e2: Vec<f64> = parts.iter().map(|p| l0(p).powi(2)).collect();
                let sum: f64 = e2.iter().sum();
                say(format!(
                    "len {n} SHARES at layer {ops_layer} (x_out, squares): {} | sum {:.3e}{}; ON16 {:.3e}",
                    parts.iter().zip(&e2).map(|(p, e)| format!("{p} {:.1}%", 100.0 * e / sum)).collect::<Vec<_>>().join(", "),
                    sum.sqrt(),
                    if has("FULL") { format!(", FULL {:.3e}", l0("FULL")) } else { String::new() },
                    on
                ));
                rules.insert("shares".into(), serde_json::json!(parts.iter().zip(&e2).map(|(p, e)| (p.to_string(), e / sum)).collect::<BTreeMap<_, _>>()));
            }
            if has("ULP") {
                let sat = (0..nl).find(|&l| out("ULP", "OFF", l) >= 0.5 * out("ON16", "OFF", l));
                say(format!("len {n} DEPTH: ULP reaches half of ON16's drift at layer {}", sat.map_or("never (within these layers)".to_string(), |l| l.to_string())));
                rules.insert("saturation_layer".into(), serde_json::json!(sat));
            }
            let max_on = (0..nl).map(|l| out("ON16", "OFF", l)).fold(0.0, f64::max);
            say(format!("len {n} DRIFT max_l relerr(ON16,OFF) = {max_on:.3e}; per variant max_l: {}", cols.iter().map(|c| format!("{c} {:.3e}", (0..nl).map(|l| out(c, "OFF", l)).fold(0.0, f64::max))).collect::<Vec<_>>().join(", ")));
            say(format!("len {n} logits cosine vs OFF: {}; top-1 OFF {}", cols.iter().map(|c| format!("{c} {:.6}", cosine(&runs[c.as_str()].logits0, &off.logits0))).collect::<Vec<_>>().join(", "), argmax(&off.logits0)));
            doc["rules"] = serde_json::Value::Object(rules);
        }
        docs.push(doc);
    }
    let out = serde_json::json!({"checkpoint": checkpoint, "lens": lens, "ops_layer": ops_layer, "variants": names, "runs": docs});
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&out).unwrap()).expect("run.json");
    std::fs::write(receipt.join("run.txt"), log.join("\n") + "\n").expect("run.txt");
}
