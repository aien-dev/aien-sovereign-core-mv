//! The Qwen3.6-35B-A3B experts alone on the chip against the host reference (L306 gate 1 probe).
//!
//!   q36_expert_check <checkpoint-dir> <golden.json> <layers e.g. 0,1,4,5> <out.txt>
//!
//! The host reference runs the golden sequence and keeps each MoE layer's normed input rows and
//! routed ids; for each listed layer the chip's FP8 GEMV computes every used expert's gate|up and
//! down on exactly those rows (see `qwen38_omega::expert_probe`). Prints one line per layer.
use std::path::Path;

use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
use aien_inference_abi::qwen38_omega::expert_probe;
use serde_json::Value;

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_expert_check: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: q36_expert_check <checkpoint-dir> <golden.json> <layers> <out.txt>".into(),
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
    let mut toks = ids_of("prompt_ids");
    let gen = ids_of("generated_ids");
    toks.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
    let layers: Vec<usize> = args[3]
        .split(',')
        .map(|s| s.parse().map_err(|_| format!("layer {s}")))
        .collect::<Result<_, _>>()?;

    let mut model = Qwen38Reference::open(dir)?;
    model.moe_capture = true;
    let mut state = model.new_state();
    model.forward(&mut state, &toks, Logits::Last)?;
    let top_k = model
        .config
        .moe
        .as_ref()
        .ok_or("not a MoE checkpoint")?
        .top_k;
    eprintln!("reference: {} positions", toks.len());
    let lines = expert_probe(dir, &layers, &model.moe_xn, &model.routes, top_k)?;
    std::fs::write(&args[4], lines.join("\n") + "\n").map_err(|e| format!("{}: {e}", args[4]))?;
    for l in &lines {
        println!("{l}");
    }
    Ok(())
}
