//! The Qwen3.6-35B-A3B MXFP4 store (L355): convert once, check, survey the scale rules.
//!
//!   q36_w4_store write <checkpoint dir> <out store> <ocp|fit|best> [threads]
//!   q36_w4_store check <checkpoint dir> <store>
//!   q36_w4_store survey <checkpoint dir> [experts per layer, default 16]
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t0 = Instant::now();
    let r = match a.get(1).map(String::as_str) {
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
        Some("survey") if a.len() >= 3 => survey(Path::new(&a[2]), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16)),
        _ => Err("usage: q36_w4_store write <dir> <out> <ocp|fit|best> [threads] | check <dir> <store> | survey <dir> [n]".into()),
    };
    if let Err(e) = r {
        eprintln!("q36_w4_store: {e}");
        std::process::exit(1);
    }
}
