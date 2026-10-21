//! L313 DeltaNet state precision and chunked state update on the Qwen3.6-35B host reference.
//!
//!   q36_state_drift drift <checkpoint-dir> <golden.json> <new-tokens> <out.json>
//!   q36_state_drift score <checkpoint-dir> <passage.txt> <max-tokens> <out.json>
//!   q36_state_drift chunk <checkpoint-dir> <drift.json> <new-tokens> <out.json>
//!
//! Host only (no chip). Every mode runs several states in lockstep (one layer read, every state
//! in turn), all fed the same ids; state 0 is the reference.
//! `drift`: states f32 | bf16 | bf16 forced | fp8 | fp8 forced ("forced" takes the expert ids the
//! f32 state picked at the same layer and row, so it shows state drift alone; the free states add
//! router flips). The f32 state continues the golden's prompt greedily; the others are fed the
//! f32 ids (teacher-forced), so the first step whose argmax differs is the first divergence of
//! their own greedy run. Per step: argmax agreement, logits cosine, NLL of the f32 pick, state
//! relative error (all DeltaNet layers, and the worst layer), routed rows whose expert set flipped.
//! `score`: the same states teacher-forced over a passage in one call: perplexity per state.
//! `chunk`: the drift run's ids fed in blocks of 16 with every row's logits; states token-by-token
//! (reference) | chunk 2 | 4 | 8 | 16 | 16 forced: the chunked state update against the recurrence.

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{
    argmax, negative_log_likelihood, Logits, Qwen38Reference, Qwen38State, StateStore,
};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// Top-1 minus top-2 logit.
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

/// State relative error against the reference: over all DeltaNet layers, and the worst layer.
fn state_rel(s: &Qwen38State, r: &Qwen38State) -> (f64, f64) {
    let (mut dd, mut rr, mut worst) = (0f64, 0f64, 0f64);
    for (a, b) in s.delta_states().zip(r.delta_states()) {
        let (mut d, mut n) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            d += (x as f64 - y as f64).powi(2);
            n += (y as f64).powi(2);
        }
        dd += d;
        rr += n;
        worst = worst.max((d / n).sqrt());
    }
    ((dd / rr).sqrt(), worst)
}

fn ids_of(g: &Value, k: &str) -> Vec<u32> {
    g[k].as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default()
}

struct Variant {
    name: String,
    store: StateStore,
    forced: bool,
    chunk: usize,
}

fn variants(mode: &str) -> Vec<Variant> {
    let v = |name: &str, store, forced, chunk| Variant {
        name: name.into(),
        store,
        forced,
        chunk,
    };
    if mode == "chunk" {
        vec![
            v("token-by-token", StateStore::F32, false, 0),
            v("chunk2", StateStore::F32, false, 2),
            v("chunk4", StateStore::F32, false, 4),
            v("chunk8", StateStore::F32, false, 8),
            v("chunk16", StateStore::F32, false, 16),
            v("chunk16+forced", StateStore::F32, true, 16),
        ]
    } else {
        vec![
            v("f32", StateStore::F32, false, 0),
            v("bf16", StateStore::Bf16, false, 0),
            v("bf16+forced", StateStore::Bf16, true, 0),
            v("fp8-e4m3-row", StateStore::Fp8E4m3Row, false, 0),
            v("fp8-e4m3-row+forced", StateStore::Fp8E4m3Row, true, 0),
        ]
    }
}

/// Per-state, per-row record against state 0.
#[derive(Default)]
struct Track {
    agree: Vec<bool>,
    cos: Vec<f64>,
    nll_ref: Vec<f64>,
    flips: Vec<usize>,
    state_rel: Vec<f64>,
    state_rel_worst: Vec<f64>,
}

impl Track {
    fn summary(&self, v: &Variant, extra: Value) -> Value {
        let first = self.agree.iter().position(|&a| !a);
        let (cmin, cat) =
            self.cos.iter().enumerate().fold(
                (1f64, 0usize),
                |m, (i, &c)| if c < m.0 { (c, i) } else { m },
            );
        json!({
            "name": v.name, "store": v.store.name(), "forced_routes": v.forced, "chunk": v.chunk,
            "first_divergence": first, "agree": self.agree.iter().filter(|&&a| a).count(),
            "rows": self.agree.len(), "cos_min": cmin, "cos_min_at": cat,
            "flips_total": self.flips.iter().sum::<usize>(),
            "state_rel_last": self.state_rel.last(), "state_rel_max": self.state_rel.iter().cloned().fold(0f64, f64::max),
            "state_rel_worst_layer_max": self.state_rel_worst.iter().cloned().fold(0f64, f64::max),
            "extra": extra,
            "per_row": {"agree": self.agree, "cos": self.cos, "nll_ref": self.nll_ref, "flips": self.flips,
                        "state_rel": self.state_rel, "state_rel_worst_layer": self.state_rel_worst},
        })
    }
}

fn write(out_path: &str, report: &Value) -> Result<(), String> {
    let tmp = format!("{out_path}.part");
    std::fs::write(
        &tmp,
        serde_json::to_string_pretty(report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, out_path).map_err(|e| format!("{out_path}: {e}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage =
        "usage: q36_state_drift drift|score|chunk <checkpoint-dir> <input> <count> <out.json>";
    if args.len() != 6 {
        return Err(usage.into());
    }
    let (mode, dir, input, out_path) = (args[1].as_str(), Path::new(&args[2]), &args[3], &args[5]);
    let count: usize = args[4].parse().map_err(|_| usage.to_string())?;
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let t0 = Instant::now();
    let mut model = Qwen38Reference::open(dir)?;
    let vocab = model.config.vocab;
    let vs = variants(mode);
    let mut states: Vec<Qwen38State> = vs
        .iter()
        .map(|v| {
            let mut s = model.new_state();
            s.store = v.store;
            s.force_routes = v.forced;
            s
        })
        .collect();
    let mut tracks: Vec<Track> = vs.iter().map(|_| Track::default()).collect();
    // Compares one call's rows (`n` rows of logits per state) against state 0.
    let record = |tracks: &mut Vec<Track>,
                  states: &[Qwen38State],
                  rows: &[Vec<f32>],
                  flips: &[usize],
                  targets: Option<&[u32]>| {
        let n = rows[0].len() / vocab;
        for (i, tr) in tracks.iter_mut().enumerate() {
            for r in 0..n {
                let (a, b) = (
                    &rows[0][r * vocab..(r + 1) * vocab],
                    &rows[i][r * vocab..(r + 1) * vocab],
                );
                let want = argmax(a);
                tr.agree.push(argmax(b) == want);
                tr.cos.push(cosine(a, b));
                let tgt = targets.and_then(|t| t.get(r)).map_or(want, |&t| t as usize);
                tr.nll_ref.push(negative_log_likelihood(b, tgt));
            }
            tr.flips.push(flips.get(i).copied().unwrap_or(0));
            let (rel, worst) = state_rel(&states[i], &states[0]);
            tr.state_rel.push(rel);
            tr.state_rel_worst.push(worst);
        }
    };

    let report = match mode {
        "drift" => {
            let g: Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
            let prompt = ids_of(&g, "prompt_ids");
            let golden = ids_of(&g, "generated_ids");
            let mut logits = model.forward_lockstep(&mut states, &prompt, Logits::Last)?;
            let prefill_flips = model.lockstep_flips.clone();
            let mut ids = Vec::with_capacity(count);
            let mut margins = Vec::with_capacity(count);
            let mut step_s = Vec::with_capacity(count);
            let snapshot = |ids: &[u32],
                            margins: &[f32],
                            step_s: &[f64],
                            tracks: &[Track],
                            done: bool|
             -> Value {
                let gm = golden.iter().zip(ids).position(|(a, b)| a != b);
                json!({
                    "mode": "drift", "golden": input, "prompt_ids": prompt, "f32_ids": ids,
                    "golden_checked": golden.len().min(ids.len()), "golden_first_mismatch": gm,
                    "complete": done, "steps": ids.len(), "seconds": t0.elapsed().as_secs_f64(),
                    "step_seconds": step_s, "f32_margin": margins, "prefill_flips": prefill_flips,
                    "states": tracks.iter().zip(&vs).map(|(t, v)| t.summary(v, json!(null))).collect::<Vec<_>>(),
                })
            };
            for step in 0..count {
                let ts = Instant::now();
                let flips = if step == 0 {
                    vec![0; vs.len()]
                } else {
                    model.lockstep_flips.clone()
                };
                let tok = argmax(&logits[0]) as u32;
                margins.push(margin(&logits[0]));
                record(&mut tracks, &states, &logits, &flips, None);
                ids.push(tok);
                if step + 1 < count {
                    logits = model.forward_lockstep(&mut states, &[tok], Logits::Last)?;
                }
                step_s.push(ts.elapsed().as_secs_f64());
                if (step + 1) % 32 == 0 {
                    let firsts: Vec<String> = tracks
                        .iter()
                        .zip(&vs)
                        .skip(1)
                        .map(|(t, v)| format!("{} {:?}", v.name, t.agree.iter().position(|&a| !a)))
                        .collect();
                    eprintln!(
                        "step {} at {:.0} s; first divergence: {}",
                        step + 1,
                        t0.elapsed().as_secs_f64(),
                        firsts.join(", ")
                    );
                    write(out_path, &snapshot(&ids, &margins, &step_s, &tracks, false))?;
                }
            }
            snapshot(&ids, &margins, &step_s, &tracks, true)
        }
        "score" => {
            let mut text = text;
            if let Some((cut, _)) = text.char_indices().nth(count * 8) {
                text.truncate(cut);
            }
            let tok =
                ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
            let mut ids = tok.encode(&text).map_err(|e| e.to_string())?;
            ids.truncate(count);
            let logits = model.forward_lockstep(&mut states, &ids, Logits::All)?;
            let flips = model.lockstep_flips.clone();
            record(&mut tracks, &states, &logits, &flips, Some(&ids[1..]));
            let per: Vec<Value> = tracks
                .iter()
                .zip(&vs)
                .map(|(t, v)| {
                    let nll = &t.nll_ref[..ids.len() - 1];
                    let mean = nll.iter().sum::<f64>() / nll.len() as f64;
                    let rows = &logits[vs.iter().position(|x| x.name == v.name).unwrap_or(0)];
                    let hits = (0..ids.len() - 1)
                        .filter(|&r| {
                            argmax(&rows[r * vocab..(r + 1) * vocab]) == ids[r + 1] as usize
                        })
                        .count();
                    t.summary(
                        v,
                        json!({"mean_nll": mean, "perplexity": mean.exp(),
                        "next_token_accuracy": hits as f64 / nll.len() as f64}),
                    )
                })
                .collect();
            json!({"mode": "score", "passage": input, "tokens": ids.len(), "ids": ids,
                   "seconds": t0.elapsed().as_secs_f64(), "states": per})
        }
        "chunk" => {
            let g: Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
            let prompt = ids_of(&g, "prompt_ids");
            let gen = ids_of(&g, "f32_ids");
            let n = count.min(gen.len());
            // The prompt token by token (chunk 0) for every state, then the chunk sizes.
            model.forward_lockstep(&mut states, &prompt, Logits::Last)?;
            states
                .iter_mut()
                .zip(&vs)
                .for_each(|(s, v)| s.chunk = v.chunk);
            const STRIDE: usize = 16;
            let mut seq_ids = Vec::with_capacity(n);
            for block in gen[..n.saturating_sub(1)].chunks(STRIDE) {
                let rows = model.forward_lockstep(&mut states, block, Logits::All)?;
                let flips = model.lockstep_flips.clone();
                record(&mut tracks, &states, &rows, &flips, None);
                for r in rows[0].chunks_exact(vocab) {
                    seq_ids.push(argmax(r) as u32);
                }
                eprintln!(
                    "{} of {} rows at {:.0} s",
                    seq_ids.len(),
                    n - 1,
                    t0.elapsed().as_secs_f64()
                );
            }
            // The reference fed t = 16 rows against the drift run's one-row decode.
            let vs_decode = seq_ids
                .iter()
                .zip(&gen[1..])
                .filter(|(a, b)| a == b)
                .count();
            json!({"mode": "chunk", "drift": input, "stride": STRIDE, "rows": seq_ids.len(),
                   "reference_vs_decode_agree": vs_decode, "seconds": t0.elapsed().as_secs_f64(),
                   "states": tracks.iter().zip(&vs).map(|(t, v)| t.summary(v, json!(null))).collect::<Vec<_>>()})
        }
        _ => return Err(usage.into()),
    };
    write(out_path, &report)?;
    eprintln!("done in {:.0} s -> {out_path}", t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_state_drift: {e}");
        std::process::exit(1);
    }
}
