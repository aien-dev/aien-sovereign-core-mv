//! 4-bit drift on the Qwen3.6-35B-A3B host reference (L355), scored by docs/inference/q36/w4-gate.md.
//!
//!   q36_w4_drift forced <checkpoint dir> <ref.json> <steps> <arm> <out.json>
//!   q36_w4_drift score  <checkpoint dir> <passage.txt> <tokens> <arm> <out.json>
//!
//! arm: fp8 | w4e (experts from the store) | w4et (experts + trunk) | w4e-mut (experts, every e2m1 mantissa bit
//! cleared: the calibrated mutant). The store comes from AIEN_Q36_W4_STORE (a path, or default) for every arm but fp8.
//! forced: the prompt (`prompt_ids`) then `steps` tokens teacher-forced on the reference ids (`f32_ids` of an L313
//! drift file, else `ref_ids` of an fp8 forced output); with neither (a golden file) the arm runs greedy and its own picks
//! become `ref_ids` (a golden may give `prompt_text` instead of `prompt_ids`). Per step: the arm's argmax, its top-1 margin and the NLL of the reference id. score: NLL per token.
//!   q36_w4_drift mtp    <checkpoint dir> <ref.json> <steps> <arm> <out.json>   (AIEN_L355_MTP_K, default 4)
//! mtp: teacher-forced on the reference ids like forced; at every step the MTP head drafts a chain of up to K tokens
//! (each draft fed to the next) and the step records how many of them equal the reference continuation (the drafts a
//! verify pass would accept); the MTP cache holds every true position (the prompt included), chained drafts are
//! rolled back. Accepted tokens per verify pass at cap k = 1 + mean(min(accepted, k)).
//! One arm per process (one model in memory); the output is rewritten after every step (resumable by hand, not
//! automatically).
use aien_inference_abi::q36_w4_store::W4Store;
use aien_inference_abi::qwen38::{argmax, negative_log_likelihood, Logits, MtpState, Qwen38Reference, W4Host};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn margin(logits: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

fn ids_of(g: &Value, k: &str) -> Vec<u32> {
    g[k].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect()).unwrap_or_default()
}

fn write(out: &str, v: &Value) -> Result<(), String> {
    let tmp = format!("{out}.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, out).map_err(|e| format!("{out}: {e}"))
}

fn arm_host(arm: &str, dir: &Path) -> Result<Option<W4Host>, String> {
    let (experts, trunk, mutant) = match arm {
        "fp8" => return Ok(None),
        "w4e" => (true, false, false),
        "w4et" => (true, true, false),
        "w4e-mut" | "w4e-smut" | "w4e-mgu" | "w4e-mdn" => (true, false, arm == "w4e-mut"),
        a => return Err(format!("arm {a} (fp8, w4e, w4et, w4e-mut, w4e-smut, w4e-mgu, w4e-mdn)")),
    };
    let p = aien_inference_abi::q36_w4_store::resolve(dir)?
        .ok_or("AIEN_Q36_W4_STORE (store path or default) is not set")?;
    let store = W4Store::open(&p, dir)?;
    let fp8_layers = aien_inference_abi::q36_w4_store::fp8_layers_from_env()?;
    let shared_fp8 = std::env::var("AIEN_Q36_W4_SHARED_FP8").as_deref() == Ok("1");
    let scale_mutant = arm == "w4e-smut";
    if scale_mutant && store.group() != 16 {
        return Err("w4e-smut is the nv16 scale control (needs an nv16 store)".into());
    }
    // w4-gate.md "NV16 arm" item 4: m-gu keeps down at FP8, m-dn keeps gate and up at FP8
    let fp8_part = match arm {
        "w4e-mdn" => 1,
        "w4e-mgu" => 2,
        _ => 0,
    };
    Ok(Some(W4Host { store: Arc::new(store), experts, trunk, mutant, fp8_layers, shared_fp8, scale_mutant, fp8_part }))
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let usage = "usage: q36_w4_drift forced|score|mtp <checkpoint dir> <input> <count> <arm> <out.json>";
    if a.len() != 7 {
        return Err(usage.into());
    }
    let (mode, dir, input, arm, out) = (a[1].as_str(), Path::new(&a[2]), &a[3], a[5].as_str(), &a[6]);
    let count: usize = a[4].parse().map_err(|_| usage.to_string())?;
    let t0 = Instant::now();
    let mut model = Qwen38Reference::open(dir)?;
    model.set_w4(arm_host(arm, dir)?);
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    match mode {
        "forced" => {
            let g: Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
            let mut prompt = ids_of(&g, "prompt_ids");
            if let (true, Some(pt)) = (prompt.is_empty(), g["prompt_text"].as_str()) {
                // a held-out prompt written as text (w4-gate.md, GPTQ arm): the tokenizer's ids, special tokens included
                let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
                prompt = tok.encode(pt).map_err(|e| e.to_string())?;
            }
            let mut refs = ids_of(&g, "f32_ids");
            if refs.is_empty() {
                refs = ids_of(&g, "ref_ids");
            }
            let greedy = refs.is_empty();
            if prompt.is_empty() || (!greedy && refs.len() < count) {
                return Err(format!("{input}: prompt_ids empty or fewer than {count} reference ids"));
            }
            let ref_margin: Vec<f64> = g["f32_margin"].as_array().or(g["ref_margin"].as_array())
                .map(|v| v.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            // AIEN_L355_ROUTES=1 (diagnostic, w4-results.md): the routed expert ids of every layer at every step (the
            // top-k picks of the row whose logits are scored), to see whether the 4-bit excess sits on router flips
            let want_routes = std::env::var("AIEN_L355_ROUTES").as_deref() == Ok("1");
            let k_top = model.config.moe.as_ref().map_or(0, |m| m.top_k);
            let last_picks = |m: &Qwen38Reference<_>| -> Vec<Vec<u16>> {
                m.routes.iter().take(m.config.layers).map(|r| r[r.len().saturating_sub(k_top)..].to_vec()).collect()
            };
            let mut st = model.new_state();
            let mut logits = model.forward(&mut st, &prompt, Logits::Last)?;
            let mut routes: Vec<Vec<Vec<u16>>> = Vec::new();
            let (mut am, mut mg, mut nll, mut fed, mut secs) = (vec![], vec![], vec![], vec![], vec![]);
            for t in 0..count {
                let ts = Instant::now();
                if want_routes {
                    routes.push(last_picks(&model));
                }
                let pick = argmax(&logits) as u32;
                let want = if greedy { pick } else { refs[t] };
                am.push(pick);
                mg.push(margin(&logits) as f64);
                nll.push(negative_log_likelihood(&logits, want as usize));
                fed.push(want);
                if t + 1 < count {
                    logits = model.forward(&mut st, &[want], Logits::Last)?;
                }
                secs.push(ts.elapsed().as_secs_f64());
                let agree = am.iter().zip(&fed).filter(|(x, y)| x == y).count();
                let rep = json!({
                    "mode": "forced", "arm": arm, "input": input, "greedy_self_reference": greedy,
                    "complete": t + 1 == count, "steps": t + 1, "prompt_ids": prompt,
                    "ref_ids": fed, "ref_margin": if greedy { json!(mg) } else { json!(ref_margin[..ref_margin.len().min(t + 1)]) },
                    "argmax": am, "margin": mg, "nll": nll, "agree": agree,
                    "store": std::env::var("AIEN_Q36_W4_STORE").ok().filter(|_| arm != "fp8"),
                    "fp8_layers": std::env::var("AIEN_Q36_W4_FP8_LAYERS").ok(), "shared_fp8": std::env::var("AIEN_Q36_W4_SHARED_FP8").ok(),
                    "routes": if want_routes { json!(routes) } else { Value::Null },
                    "step_seconds": secs, "seconds": t0.elapsed().as_secs_f64(),
                });
                write(out, &rep)?;
            }
            Ok(())
        }
        "mtp" => {
            let kmax: usize = std::env::var("AIEN_L355_MTP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 16);
            let g: Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
            let prompt = ids_of(&g, "prompt_ids");
            let mut refs = ids_of(&g, "f32_ids");
            if refs.is_empty() {
                refs = ids_of(&g, "ref_ids");
            }
            if prompt.len() < 2 || refs.len() < count {
                return Err(format!("{input}: need a prompt and {count} reference ids"));
            }
            let (layers, h) = (model.config.layers, model.config.hidden);
            let cfg = model.config.clone();
            let mut st = model.new_state();
            let mut taps = Vec::new();
            model.forward_tapped(&mut st, &prompt, Logits::Last, &[layers], &mut taps)?;
            let xp = taps.pop().ok_or("no tap")?;
            let mut ms = MtpState::default();
            for t in 0..prompt.len() - 1 {
                model.mtp_step(&mut ms, &xp[t * h..(t + 1) * h], prompt[t + 1], t)?;
            }
            let mut h_last = xp[(prompt.len() - 1) * h..].to_vec();
            let mut pos = prompt.len() - 1;
            let (mut acc, mut d1s, mut secs) = (Vec::new(), Vec::new(), Vec::new());
            for s in 0..count {
                let ts = Instant::now();
                let (lg, hx) = model.mtp_step(&mut ms, &h_last, refs[s], pos)?;
                let keep = ms.rows(&cfg);
                let d1 = argmax(&lg) as u32;
                d1s.push(d1);
                let mut a = 0usize;
                if s + 1 < count && d1 == refs[s + 1] {
                    a = 1;
                    let (mut hc, mut dp) = (hx, d1);
                    for j in 2..=kmax {
                        if s + j >= count {
                            break;
                        }
                        let (lj, hj) = model.mtp_step(&mut ms, &hc, dp, pos + j - 1)?;
                        let dj = argmax(&lj) as u32;
                        if dj != refs[s + j] {
                            break;
                        }
                        a = j;
                        hc = hj;
                        dp = dj;
                    }
                    ms.truncate(&cfg, keep);
                }
                acc.push(a);
                if s + 1 < count {
                    taps.clear();
                    model.forward_tapped(&mut st, &[refs[s]], Logits::Last, &[layers], &mut taps)?;
                    h_last = taps.pop().ok_or("no tap")?;
                    pos += 1;
                }
                secs.push(ts.elapsed().as_secs_f64());
                let per_k: Vec<f64> = (1..=kmax).map(|k| 1.0 + acc.iter().map(|&x| x.min(k)).sum::<usize>() as f64 / acc.len() as f64).collect();
                write(out, &json!({
                    "mode": "mtp", "arm": arm, "input": input, "k_max": kmax, "complete": s + 1 == count, "steps": s + 1,
                    "accepted": acc, "draft1": d1s, "tokens_per_pass_by_k": per_k,
                    "options": "embed first, residual h (pre final norm), position of h (sc qwen38/mtp-spec-decode defaults)",
                    "step_seconds": secs, "seconds": t0.elapsed().as_secs_f64(),
                }))?;
            }
            Ok(())
        }
        "score" => {
            let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
            // the same cut as L313's q36_state_drift score (count * 8 characters, then count tokens), so the fp8 arm is comparable
            let mut text = text;
            if let Some((cut, _)) = text.char_indices().nth(count * 8) {
                text.truncate(cut);
            }
            let mut ids = tok.encode(&text).map_err(|e| e.to_string())?;
            ids.truncate(count);
            let mut st = model.new_state();
            let logits = model.forward(&mut st, &ids, Logits::All)?;
            let v = model.config.vocab;
            let nll: Vec<f64> = (0..ids.len() - 1).map(|i| negative_log_likelihood(&logits[i * v..(i + 1) * v], ids[i + 1] as usize)).collect();
            let hits = (0..ids.len() - 1).filter(|&i| argmax(&logits[i * v..(i + 1) * v]) == ids[i + 1] as usize).count();
            let mean = nll.iter().sum::<f64>() / nll.len() as f64;
            write(out, &json!({
                "mode": "score", "arm": arm, "passage": input, "tokens": ids.len(), "ids": ids, "nll": nll,
                "mean_nll": mean, "top1_hits": hits,
                "store": std::env::var("AIEN_Q36_W4_STORE").ok().filter(|_| arm != "fp8"),
                "fp8_layers": std::env::var("AIEN_Q36_W4_FP8_LAYERS").ok(), "shared_fp8": std::env::var("AIEN_Q36_W4_SHARED_FP8").ok(),
                "seconds": t0.elapsed().as_secs_f64(),
            }))
        }
        _ => Err(usage.into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_w4_drift: {e}");
        std::process::exit(1);
    }
}
