//! The Qwen3.6-35B-A3B MXFP4 store (L355): convert once, check, survey the scale rules.
//!
//!   q36_w4_store write <checkpoint dir> <out store> <ocp|fit|best|nv16> [threads]
//!   q36_w4_store gptq <checkpoint dir> <out store> <calibration.txt> <tokens> [threads]
//!   q36_w4_store check <checkpoint dir> <store>
//!   q36_w4_store survey <checkpoint dir> [experts per layer, default 16]
//!   q36_w4_store split <checkpoint dir> [experts per layer, default 16]
//!
//! `survey` converts a fixed sample of each layer's routed experts (ids 0, 16, 32, .. for 16) plus the shared expert and
//! the trunk with every rule and prints one JSON line per rule: weights, saturated, relative RMS error. It picks no rule;
//! the choice is written in docs/inference/q36/w4-gate.md before a drift run.
use aien_inference_abi::q36_w4_store::{layer_projs, read_fused, write_store, W4Store};
use aien_inference_abi::qwen38::{text_prefix, Qwen38Config};
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use aien_omega_q38::w4::{from_fp8, W4Stats, W4_RULE_BEST, W4_RULE_FIT, W4_RULE_OCP};
use std::path::Path;
use std::time::Instant;

fn rule_of(s: &str) -> Result<i32, String> {
    match s {
        "ocp" => Ok(W4_RULE_OCP),
        "fit" => Ok(W4_RULE_FIT),
        "best" => Ok(W4_RULE_BEST),
        r => Err(format!("rule {r} (ocp, fit or best)")),
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get())
}

fn survey(dir: &Path, per_layer: usize) -> Result<(), String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let step = (s.experts / per_layer.max(1)).max(1);
    let rules = [("ocp", W4_RULE_OCP), ("fit", W4_RULE_FIT), ("best", W4_RULE_BEST)];
    let mut ex = [W4Stats::default(); 3];
    let mut tr = [W4Stats::default(); 3];
    for l in 0..c.layers {
        let projs = layer_projs(&c, &prefix, l, c.full_attention[l])?;
        for (pi, (name, parts)) in projs.iter().enumerate() {
            // experts: a sample of blocks (gate + up = 2 parts per expert, down = 1), always including the shared one
            let picked: Vec<_> = match pi {
                0 | 1 => parts.clone(),
                2 => parts.chunks(2).enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).flat_map(|(_, p)| p.to_vec()).collect(),
                _ => parts.iter().enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).map(|(_, p)| p.clone()).collect(),
            };
            for part in &picked {
                let (n, k, w, sc) = read_fused(&mut source, name, std::slice::from_ref(part))?;
                for (ri, (_, rule)) in rules.iter().enumerate() {
                    let (_, _, st) = from_fp8(n, k, &w, &sc, threads(), *rule).map_err(|e| format!("{name}: {e}"))?;
                    if pi < 2 { tr[ri].add(&st) } else { ex[ri].add(&st) }
                }
            }
        }
        eprintln!("survey: layer {l} of {}", c.layers);
    }
    for (ri, (rname, _)) in rules.iter().enumerate() {
        for (what, st) in [("experts", &ex[ri]), ("trunk", &tr[ri])] {
            println!(
                "{{\"rule\":\"{rname}\",\"tensors\":\"{what}\",\"weights\":{},\"saturated\":{},\"zero_groups\":{},\"rel_rms\":{:.6}}}",
                st.weights, st.saturated, st.zero_groups, st.rel_rms()
            );
        }
    }
    Ok(())
}

/// The survey's expert sample with BEST and nv16, gate|up and down apart (docs/inference/q36/w4-gate.md, NV16 arm item
/// 4): one JSON line per rule and matrix with sq_err and sq_ref, then the mixed go / no-go against 0.5 x BEST's sq.
fn split(dir: &Path, per_layer: usize) -> Result<(), String> {
    use aien_inference_abi::{q36_w4_gptq::widen_fp8, q36_w4_nv};
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let step = (s.experts / per_layer.max(1)).max(1);
    // [rule][matrix]: rule 0 = best, 1 = nv16; matrix 0 = gate|up, 1 = down
    let mut st = [[W4Stats::default(); 2]; 2];
    for l in 0..c.layers {
        let projs = layer_projs(&c, &prefix, l, c.full_attention[l])?;
        for (pi, (name, parts)) in projs.iter().enumerate().skip(2) {
            let picked: Vec<_> = match pi {
                2 => parts.chunks(2).enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).flat_map(|(_, p)| p.to_vec()).collect(),
                _ => parts.iter().enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).map(|(_, p)| p.clone()).collect(),
            };
            for part in &picked {
                // one part = one linear (gate, up or down), so nv16 gets one global per linear as in its store
                let (n, k, w, sc) = read_fused(&mut source, name, std::slice::from_ref(part))?;
                let (_, _, b) = from_fp8(n, k, &w, &sc, threads(), W4_RULE_BEST).map_err(|e| format!("{name}: {e}"))?;
                let (_, _, _, v) = q36_w4_nv::quantize(&widen_fp8(n, k, &w, &sc), n, k, threads()).map_err(|e| format!("{name}: {e}"))?;
                st[0][pi - 2].add(&b);
                st[1][pi - 2].add(&v);
            }
        }
        eprintln!("split: layer {l} of {}", c.layers);
    }
    for (ri, rname) in ["best", "nv16"].iter().enumerate() {
        for (mi, what) in ["gate_up", "down"].iter().enumerate() {
            let x = &st[ri][mi];
            println!(
                "{{\"rule\":\"{rname}\",\"matrix\":\"{what}\",\"weights\":{},\"saturated\":{},\"sq_err\":{:e},\"sq_ref\":{:e},\"rel_rms\":{:.6}}}",
                x.weights, x.saturated, x.sq_err, x.sq_ref, x.rel_rms()
            );
        }
    }
    let r = st[0][0].sq_ref + st[0][1].sq_ref;
    let limit = 0.5 * (st[0][0].sq_err + st[0][1].sq_err) / r;
    let (gu, dn) = (st[1][0].sq_err / r, st[1][1].sq_err / r);
    let v = |x: f64| if x <= limit { "GO" } else { "NO-GO" };
    println!(
        "{{\"best_sq\":{:.7},\"nv16_sq\":{:.7},\"limit_sq\":{limit:.7},\"m_gu_sq\":{gu:.7},\"m_gu\":\"{}\",\"m_dn_sq\":{dn:.7},\"m_dn\":\"{}\"}}",
        2.0 * limit,
        (st[1][0].sq_err + st[1][1].sq_err) / r,
        v(gu),
        v(dn)
    );
    Ok(())
}

/// Runs the FP8 host reference over the first `tokens` tokens of `text` with the calibration tap on, then writes the
/// error-feedback store (docs/inference/q36/w4-gate.md, GPTQ arm). The calibration text must not be gate text.
fn gptq(dir: &Path, out: &Path, text_path: &Path, tokens: usize, th: usize) -> Result<(), String> {
    use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
    use aien_inference_abi::tokenizer::ChatTokenizer;
    let t0 = Instant::now();
    let text = std::fs::read_to_string(text_path).map_err(|e| format!("{}: {e}", text_path.display()))?;
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    // the tokenizer takes at most 2048 tokens per call: encode pieces of about 4000 characters cut at line ends
    let mut ids = Vec::new();
    let mut piece = String::new();
    for line in text.split_inclusive('\n') {
        piece.push_str(line);
        if piece.len() >= 4000 {
            ids.extend(tok.encode(&piece).map_err(|e| e.to_string())?);
            piece.clear();
        }
    }
    if !piece.is_empty() {
        ids.extend(tok.encode(&piece).map_err(|e| e.to_string())?);
    }
    if ids.len() < tokens {
        return Err(format!("{}: {} tokens, {tokens} asked", text_path.display(), ids.len()));
    }
    ids.truncate(tokens);
    let mut model = Qwen38Reference::open(dir)?;
    let c = model.config.clone();
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    model.calib = Some(aien_inference_abi::q36_w4_gptq::Calib::new(c.layers, c.hidden, s.expert_inter, s.experts, th));
    // windows of 2048 tokens (a fresh context each, like separate conversations), fed in chunks of 512
    let mut st = model.new_state();
    for (i, chunk) in ids.chunks(512).enumerate() {
        if i % 4 == 0 {
            st = model.new_state();
        }
        model.forward(&mut st, chunk, Logits::Last)?;
        eprintln!("gptq calibration: {} of {tokens} tokens, {:.0} s", i * 512 + chunk.len(), t0.elapsed().as_secs_f64());
    }
    let calib = model.calib.take().ok_or("calibration lost")?;
    drop(model);
    let h = aien_inference_abi::q36_w4_gptq::write_store(dir, out, &calib, th, &mut |m| eprintln!("{m}"))?;
    println!(
        "{{\"store\":\"{}\",\"rule\":\"{}\",\"calibration\":\"{}\",\"tokens\":{tokens},\"tensors\":{},\"weights\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
        out.display(), h.rule, text_path.display(), h.tensors.len(), h.weights, h.rel_rms, t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t0 = Instant::now();
    let r = match a.get(1).map(String::as_str) {
        Some("write") if a.len() >= 5 && a[4] == "nv16" => {
            let th = a.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            aien_inference_abi::q36_w4_nv::write_store(Path::new(&a[2]), Path::new(&a[3]), th, &mut |m| eprintln!("{m}")).map(|h| {
                println!(
                    "{{\"store\":\"{}\",\"rule\":\"{}\",\"tensors\":{},\"weights\":{},\"saturated\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
                    a[3], h.rule, h.tensors.len(), h.weights, h.saturated, h.rel_rms, t0.elapsed().as_secs_f64()
                );
            })
        }
        Some("write") if a.len() >= 5 => rule_of(&a[4]).and_then(|rule| {
            let th = a.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            let h = write_store(Path::new(&a[2]), Path::new(&a[3]), rule, th, &mut |m| eprintln!("{m}"))?;
            println!(
                "{{\"store\":\"{}\",\"rule\":\"{}\",\"tensors\":{},\"weights\":{},\"saturated\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
                a[3], h.rule, h.tensors.len(), h.weights, h.saturated, h.rel_rms, t0.elapsed().as_secs_f64()
            );
            Ok(())
        }),
        Some("check") if a.len() >= 4 => W4Store::open(Path::new(&a[3]), Path::new(&a[2])).and_then(|s| {
            s.check_all()?;
            println!("{{\"store\":\"{}\",\"check\":\"PASS\",\"tensors\":{},\"seconds\":{:.1}}}", a[3], s.header.tensors.len(), t0.elapsed().as_secs_f64());
            Ok(())
        }),
        Some("gptq") if a.len() >= 6 => a[5].parse().map_err(|_| "tokens".to_string()).and_then(|n| {
            let th = a.get(6).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            gptq(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]), n, th)
        }),
        Some("survey") if a.len() >= 3 => survey(Path::new(&a[2]), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16)),
        Some("split") if a.len() >= 3 => split(Path::new(&a[2]), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16)),
        _ => Err("usage: q36_w4_store write <dir> <out> <ocp|fit|best|nv16> [threads] | gptq <dir> <out> <calibration.txt> <tokens> [threads] | check <dir> <store> | survey <dir> [n] | split <dir> [n]".into()),
    };
    if let Err(e) = r {
        eprintln!("q36_w4_store: {e}");
        std::process::exit(1);
    }
}
