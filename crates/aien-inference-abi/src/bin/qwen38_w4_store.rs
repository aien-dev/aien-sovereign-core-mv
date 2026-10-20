//! Converts the Qwen3.8-27B FP8 checkpoint to the MXFP4 store once, ahead of time (L125 SPEED-1), so the engine loads
//! 4-bit weights only (`aien_inference_abi::qwen38_w4_store`).
//!
//!   qwen38_w4_store write <checkpoint-dir> [<out-file>]   default out: <dir>.mxfp4/aien-w4.store (the served default)
//!   qwen38_w4_store check <checkpoint-dir> [<out-file>]   open it as the loader does, then read and check every tensor
//!
//! The scale rule comes from `AIEN_Q38_W4_RULE` (ocp default, fit, best), the same switch as the convert-at-load path.
//! Host only: no GPU. Prints one JSON line at the end.
use std::path::{Path, PathBuf};
use std::time::Instant;

use aien_inference_abi::qwen38_omega::w4_rule;
use aien_inference_abi::qwen38_w4_store::{default_path, write_store, W4Store};

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: qwen38_w4_store write|check <checkpoint-dir> [<out-file>]";
    if args.len() < 3 || args.len() > 4 {
        return Err(usage.into());
    }
    let dir = Path::new(&args[2]);
    let out = args.get(3).map_or_else(|| default_path(dir), PathBuf::from);
    let t0 = Instant::now();
    match args[1].as_str() {
        "write" => {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let h = write_store(dir, &out, w4_rule()?, 16, &mut |l| eprintln!("{l}"))?;
            let bytes = std::fs::metadata(&out).map_err(|e| e.to_string())?.len();
            println!(
                "{}",
                serde_json::json!({
                    "store": out.display().to_string(), "bytes": bytes, "tensors": h.tensors.len(),
                    "rule": h.rule, "source": h.source, "weights": h.weights, "saturated": h.saturated,
                    "rel_rms": h.rel_rms, "seconds": t0.elapsed().as_secs_f64(),
                })
            );
        }
        "check" => {
            let s = W4Store::open(&out, dir)?;
            for t in &s.header.tensors {
                s.read(&t.name, t.n, t.k)?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "store": out.display().to_string(), "tensors": s.header.tensors.len(), "rule": s.header.rule,
                    "source": s.header.source, "verdict": "PASS", "seconds": t0.elapsed().as_secs_f64(),
                })
            );
        }
        _ => return Err(usage.into()),
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38_w4_store: {e}");
        std::process::exit(1);
    }
}
