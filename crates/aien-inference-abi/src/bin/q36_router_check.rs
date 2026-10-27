//! The Qwen3.6-35B-A3B router alone on the chip against the host reference (L306 gate 1 probe).
//!
//!   q36_router_check <checkpoint-dir> <golden.json> <out.json>
//!
//! The host reference runs the golden sequence (prompt plus every generated token but the last)
//! and keeps each MoE layer's input residual; the chip's router (`omega_gpu_router`, the kernel the
//! decode graph records) then routes exactly those rows, layer by layer. Per row: the 8 routed ids
//! as sets, the routed weights and the shared expert gate (slot 8) against the reference. A pick
//! that differs is reported with the reference's gap between its 8th and 9th logit, so a near tie
//! (allowed) is told apart from a wrong pick (a bug). Control: the chip's ids for row r of layer l
//! must not match row r + 1's (the check can say no).
use std::path::Path;

use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
use aien_inference_abi::qwen38_omega::router_probe;
use serde_json::{json, Value};

/// A weight is right when it is within this of the reference's (f32 softmax over 256 logits).
const WEIGHT_TOL: f32 = 1e-4;
/// A differing pick is a near tie when the reference's 8th/9th logit gap is under this.
const TIE_GAP: f32 = 1e-3;

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_router_check: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: q36_router_check <checkpoint-dir> <golden.json> <out.json>".into());
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
    let t = toks.len();

    let mut model = Qwen38Reference::open(dir)?;
    model.moe_capture = true;
    let mut state = model.new_state();
    model.forward(&mut state, &toks, Logits::Last)?;
    let c = model.config.clone();
    let top_k = c.moe.as_ref().ok_or("not a MoE checkpoint")?.top_k;
    let h = c.hidden;
    eprintln!(
        "reference: {t} positions, {} layers captured",
        model.moe_x.len()
    );

    let (mut router, _) = router_probe(dir)?;
    let slots = router.slots();
    let (mut same, mut tie, mut wrong, mut wmax, mut gmax, mut ctl_same) =
        (0usize, 0usize, 0usize, 0f32, 0f32, 0usize);
    let mut bad = Vec::new();
    let mut prev: Option<Vec<u32>> = None;
    for l in 0..c.layers {
        for r in 0..t {
            let x = &model.moe_x[l][r * h..(r + 1) * h];
            let (mut ids, mut wts) = (vec![0u32; slots], vec![0f32; slots]);
            router
                .route(l, x, c.eps, true, 1.0, &mut ids, &mut wts)
                .map_err(|e| format!("layer {l} row {r}: router: {e:?}"))?;
            let want: Vec<u32> = model.routes[l][r * top_k..(r + 1) * top_k]
                .iter()
                .map(|&e| e as u32)
                .collect();
            let want_w = &model.route_wts[l][r * top_k..(r + 1) * top_k];
            let gap = model.route_gaps[l][r];
            let mut a = ids[..top_k].to_vec();
            let mut b = want.clone();
            a.sort_unstable();
            b.sort_unstable();
            let sg = (wts[top_k] - model.shared_g[l][r]).abs();
            gmax = gmax.max(sg);
            if let Some(p) = &prev {
                if *p == a {
                    ctl_same += 1;
                }
            }
            prev = Some(a.clone());
            if a == b {
                same += 1;
                for (k, &e) in want.iter().enumerate() {
                    let j = ids[..top_k].iter().position(|&v| v == e).unwrap_or(0);
                    wmax = wmax.max((wts[j] - want_w[k]).abs());
                }
            } else if gap < TIE_GAP {
                tie += 1;
            } else {
                wrong += 1;
                if bad.len() < 40 {
                    bad.push(
                        json!({"layer": l, "row": r, "chip": ids[..top_k], "chip_w": wts[..top_k],
                        "ref": want, "ref_w": want_w, "gap": gap}),
                    );
                }
            }
        }
    }
    let rows = c.layers * t;
    let pass = wrong == 0 && wmax < WEIGHT_TOL && gmax < WEIGHT_TOL && ctl_same < rows / 10;
    let line = format!(
        "Q36_ROUTER layers {} rows {rows}: same picks {same}, near ties {tie} (gap < {TIE_GAP}), wrong {wrong}; \
         routed weight max diff {wmax:.2e}, shared gate max diff {gmax:.2e}; control (row vs next row same) {ctl_same} -> {}",
        c.layers,
        if pass { "PASS" } else { "FAIL" }
    );
    println!("{line}");
    let out = json!({"golden": args[2], "positions": t, "layers": c.layers, "same": same, "near_ties": tie,
        "wrong": wrong, "weight_max_diff": wmax, "shared_gate_max_diff": gmax, "control_same": ctl_same,
        "pass": pass, "wrong_rows": bad, "summary": line});
    std::fs::write(
        &args[3],
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", args[3]))?;
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
