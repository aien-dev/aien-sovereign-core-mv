//! Host study of the 4-bit (MXFP4) decode path for Qwen3.8-27B (cascade/w4-gemv): is the drift the chip's 4-bit run
//! shows the weights' rounding, or the chip?
//!
//!   qwen38_w4_host <checkpoint-dir> <seq.json> <out.json> [--rules floor,ceil,mse] [--chip <run.logits>]...
//!
//! `seq.json` holds `ids` (or `prompt_ids` + `generated_ids`); every id but the last is fed (teacher forcing) and the
//! next id is scored. The host reference runs once with the checkpoint's FP8 weights ("off") and once per rule with
//! every projection rounded to MXFP4 and back ([`REF_W4_RULE`]); the bf16 lm_head is the same in all of them. Per
//! position against "off": final residual cosine (before the final norm), logits cosine, KL(off || rule), argmax
//! agreement, and each run's NLL of the next token. A `--chip` file (`qwen38_ab run` .logits, f32 vocab per step) is
//! held against every host run, so a chip 4-bit run that matches the host 4-bit run far better than the host FP8 run
//! shows the drift is the rounding, not the kernels.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

use aien_inference_abi::qwen38::{
    argmax, negative_log_likelihood, ref_w4_rule, Logits, Qwen38Config, Qwen38Reference,
    REF_W4_RULE,
};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use serde_json::{json, Value};

fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let lse = |l: &[f32]| {
        let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&a, &b)| {
            let la = a as f64 - lp;
            la.exp() * (la - (b as f64 - lq))
        })
        .sum()
}

fn margin(l: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in l {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn ids_of(v: &Value) -> Result<Vec<u32>, String> {
    let arr = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let ids = arr("ids");
    if !ids.is_empty() {
        return Ok(ids);
    }
    let mut p = arr("prompt_ids");
    p.extend(arr("generated_ids"));
    if p.len() < 2 {
        return Err("seq.json: no ids".into());
    }
    Ok(p)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38_w4_host: {e}");
        std::process::exit(2);
    }
}

#[allow(clippy::needless_range_loop)]
fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: qwen38_w4_host <checkpoint-dir> <seq.json> <out.json> [--rules floor,ceil,mse] [--chip <run.logits>]...".into());
    }
    let (dir, seq_path, out_path) = (Path::new(&a[1]), &a[2], &a[3]);
    let mut rules = vec!["floor".to_string(), "ceil".into(), "mse".into()];
    let mut chips: Vec<String> = Vec::new();
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--rules" if i + 1 < a.len() => {
                rules = a[i + 1].split(',').map(str::to_string).collect();
                i += 2;
            }
            "--chip" if i + 1 < a.len() => {
                chips.push(a[i + 1].clone());
                i += 2;
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    for r in &rules {
        ref_w4_rule(r)
            .ok_or_else(|| format!("unknown rule {r} (off, floor, ceil, mse, nvfp4, int3)"))?;
    }
    let seq: Value = serde_json::from_str(
        &std::fs::read_to_string(seq_path).map_err(|e| format!("{seq_path}: {e}"))?,
    )
    .map_err(|e| format!("{seq_path}: {e}"))?;
    let ids = ids_of(&seq)?;
    let fed = &ids[..ids.len() - 1];
    let t = fed.len();
    let config = Qwen38Config::from_checkpoint(dir)?;
    let (h, vocab, layers) = (config.hidden, config.vocab, config.layers);

    let mut names = vec!["off".to_string()];
    names.extend(rules.iter().cloned());
    let mut runs: Vec<(Vec<f32>, Vec<f32>, f64)> = Vec::new();
    for name in &names {
        REF_W4_RULE.store(ref_w4_rule(name).unwrap(), Ordering::Relaxed);
        let t0 = Instant::now();
        let mut r = Qwen38Reference::new(config.clone(), ShardSource::open(dir)?)?;
        let mut state = r.new_state();
        let mut taps = Vec::new();
        let logits = r.forward_tapped(&mut state, fed, Logits::All, &[layers], &mut taps)?;
        let s = t0.elapsed().as_secs_f64();
        eprintln!("host {name}: {t} positions in {s:.1} s");
        runs.push((logits, taps.remove(0), s));
    }
    REF_W4_RULE.store(0, Ordering::Relaxed);
    let row = |l: &[f32], p: usize| l[p * vocab..(p + 1) * vocab].to_vec();
    let nll_mean = |l: &[f32]| {
        (0..t)
            .map(|p| negative_log_likelihood(&l[p * vocab..(p + 1) * vocab], ids[p + 1] as usize))
            .sum::<f64>()
            / t as f64
    };

    let base = &runs[0];
    let mut variants = Vec::new();
    for (vi, name) in names.iter().enumerate().skip(1) {
        let v = &runs[vi];
        let mut steps = Vec::new();
        let (mut agree, mut rc_min, mut lc_min, mut kl_sum, mut kl_max) =
            (0usize, 1f64, 1f64, 0f64, 0f64);
        for p in 0..t {
            let (bl, vl) = (row(&base.0, p), row(&v.0, p));
            let rcos = parity_stats(&v.1[p * h..(p + 1) * h], &base.1[p * h..(p + 1) * h]).cosine;
            let lcos = parity_stats(&vl, &bl).cosine;
            let k = kl(&bl, &vl);
            let same = argmax(&bl) == argmax(&vl);
            agree += same as usize;
            rc_min = rc_min.min(rcos);
            lc_min = lc_min.min(lcos);
            kl_sum += k;
            kl_max = kl_max.max(k);
            let rn = |x: &[f32]| {
                x.iter()
                    .map(|&a| (a as f64) * (a as f64))
                    .sum::<f64>()
                    .sqrt()
            };
            steps.push(json!({"p": p, "fed": fed[p], "residual_cosine": rcos, "logits_cosine": lcos, "kl": k,
                "argmax_off": argmax(&bl), "argmax": argmax(&vl), "margin_off": margin(&bl),
                "residual_norm_off": rn(&base.1[p * h..(p + 1) * h])}));
            println!("{name} p {p:3} fed {:6} residual_cos {rcos:.6} logits_cos {lcos:.6} kl {k:.5} top {} {} margin_off {:.3}",
                fed[p], argmax(&bl), argmax(&vl), margin(&bl));
        }
        let (nb, nv) = (nll_mean(&base.0), nll_mean(&v.0));
        println!("{name}: argmax {agree}/{t}, residual cosine min {rc_min:.6}, logits cosine min {lc_min:.6}, KL mean {:.5} max {kl_max:.5}, NLL off {nb:.5} {name} {nv:.5}",
            kl_sum / t as f64);
        variants.push(json!({"rule": name, "host_s": v.2, "argmax_agree": agree, "positions": t,
            "residual_cosine_min": rc_min, "logits_cosine_min": lc_min, "kl_mean": kl_sum / t as f64, "kl_max": kl_max,
            "nll_mean_off": nb, "nll_mean": nv, "steps": steps}));
    }
    let mut chip_out = Vec::new();
    for c in &chips {
        let Ok(cl) = read_f32(c) else {
            println!("chip {c}: missing, skipped");
            continue;
        };
        if cl.len() < t * vocab {
            return Err(format!("{c}: {} floats, want {}", cl.len(), t * vocab));
        }
        let mut per = Vec::new();
        for (vi, name) in names.iter().enumerate() {
            let (mut cmin, mut csum, mut kmax, mut ksum, mut agree) =
                (1f64, 0f64, 0f64, 0f64, 0usize);
            for p in 0..t {
                let (hl, xl) = (row(&runs[vi].0, p), row(&cl, p));
                let cs = parity_stats(&xl, &hl).cosine;
                let k = kl(&hl, &xl);
                cmin = cmin.min(cs);
                csum += cs;
                kmax = kmax.max(k);
                ksum += k;
                agree += (argmax(&hl) == argmax(&xl)) as usize;
            }
            println!("chip {c} vs host {name}: logits cosine min {cmin:.6} mean {:.6}, KL mean {:.5} max {kmax:.5}, argmax {agree}/{t}",
                csum / t as f64, ksum / t as f64);
            per.push(json!({"host": name, "logits_cosine_min": cmin, "logits_cosine_mean": csum / t as f64,
                "kl_mean": ksum / t as f64, "kl_max": kmax, "argmax_agree": agree}));
        }
        chip_out.push(json!({"chip": c, "vs": per}));
    }
    let out = json!({"checkpoint": dir.display().to_string(), "seq": seq_path, "positions": t,
        "host_s_off": base.2, "variants": variants, "chip": chip_out});
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| format!("{out_path}: {e}"))?;
    Ok(())
}
