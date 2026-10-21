//! One window of the 35B on the chip fed the HOST reference's residual (L306 gate 1 probe).
//!
//!   q36_layer_probe <checkpoint-dir> <golden.json> <first> <count>...
//!
//! For each `count` window starting at `first`, `first + 1`, ... (one run per listed start, see
//! below) the chip runs layers `first .. first + count` over the golden sequence, each position fed
//! the reference's residual entering layer `first` (token embeddings when `first` is 0), and its
//! output residual is held to the reference's residual entering `first + count`. With host input
//! the error printed is that window's own, not carried from earlier layers.
//!
//! Arguments after `count` are more starts: `q36_layer_probe dir g.json 4 4 0 1 2 3 5` runs the
//! windows starting at 4, 0, 1, 2, 3 and 5, each `count` layers.
use std::path::Path;

use aien_inference_abi::qwen38::{Logits, Qwen38Config, Qwen38Reference};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use serde_json::Value;

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_layer_probe: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err(
            "usage: q36_layer_probe <checkpoint-dir> <golden.json> <first> <count> [more firsts]"
                .into(),
        );
    }
    let dir = Path::new(&args[1]);
    let g: Value = serde_json::from_str(
        &std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {e}", args[2]))?,
    )
    .map_err(|e| format!("{}: {e}", args[2]))?;
    let ids_of = |k: &str| -> Vec<u32> {
        g[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut seq = ids_of("prompt_ids");
    let gen = ids_of("generated_ids");
    seq.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
    let t = seq.len();
    let count: usize = args[4].parse().map_err(|_| "count")?;
    let mut firsts: Vec<usize> = vec![args[3].parse().map_err(|_| "first")?];
    for a in &args[5..] {
        firsts.push(a.parse().map_err(|_| format!("first {a}"))?);
    }
    let config = Qwen38Config::from_checkpoint(dir)?;
    let h = config.hidden;
    let mut bounds: Vec<usize> = firsts
        .iter()
        .flat_map(|&f| [f, f + count])
        .filter(|&b| b > 0)
        .collect();
    bounds.sort_unstable();
    bounds.dedup();
    let mut taps = Vec::new();
    {
        let mut r = Qwen38Reference::new(config.clone(), ShardSource::open(dir)?)?;
        let mut state = r.new_state();
        r.forward_tapped(&mut state, &seq, Logits::Last, &bounds, &mut taps)?;
    }
    let tap =
        |b: usize| -> &Vec<f32> { &taps[bounds.iter().position(|&x| x == b).expect("tapped")] };
    for &first in &firsts {
        let mut engine = Qwen38Omega::open_window(dir, first, count, t.max(256))?;
        let want = tap(first + count);
        let (mut cos_min, mut p95_max, mut worst) = (f64::INFINITY, 0f32, 0usize);
        let mut per = Vec::with_capacity(t);
        for p in 0..t {
            if first == 0 {
                engine.step(seq[p])?;
            } else {
                engine.step_residual(&tap(first)[p * h..(p + 1) * h])?;
            }
            let x = engine.residual()?;
            let s = parity_stats(&x, &want[p * h..(p + 1) * h]);
            if s.cosine < cos_min {
                worst = p;
            }
            cos_min = cos_min.min(s.cosine);
            p95_max = p95_max.max(s.p95_rel);
            per.push(format!("{:.6}", s.cosine));
        }
        println!(
            "Q36_LAYER layers {first}..{} (host-fed): residual cosine(min) {cos_min:.6} at position {worst}, p95_rel(max) {p95_max:.4}; per position {}",
            first + count - 1,
            per.join(" ")
        );
        drop(engine);
    }
    Ok(())
}
