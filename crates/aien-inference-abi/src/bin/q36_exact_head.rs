//! Exact two-pass lm_head study on the host reference (HUM lever L314).
//!
//!   q36_exact_head hidden  <checkpoint-dir> <out.bin> <gen.json>...
//!   q36_exact_head passage <checkpoint-dir> <out.bin> <text-file> <count> <source-id>
//!   q36_exact_head analyze <checkpoint-dir> <hidden.bin> <out.json>
//!   q36_exact_head export  <checkpoint-dir> <out-dir>
//!   q36_exact_head gate    <checkpoint-dir> <w4|fp8> <row|group|linf|min> <scale> <out.json> <gen.json>...
//!
//! `hidden` replays each golden run (prompt_ids, then generated_ids token by token, exactly as
//! `qwen38_check generate` ran it) and appends the final normed hidden vector of every position;
//! `passage` appends the hidden vectors of a teacher-forced text. Records: `src pos next full` (u32,
//! `next` = the token that followed, u32::MAX when none) then `hidden` f32s.
//! `analyze` builds the W4 and FP8 screens of the bf16 lm_head and reports |C| for every bound,
//! the bytes per token, and checks the two-pass argmax against the full one on every record.
//! `gate` decodes greedily with the two-pass head choosing every token and compares the ids with
//! the golden run; `scale` < 1 is the mutant.
use aien_inference_abi::exact_head::{self, BoundKind, Screen, ScreenKind, GROUP};
use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Reference};
use aien_inference_abi::qwen3_moe_omega::{sample_next, GenerateParams};
use rayon::prelude::*;
use aien_inference_abi::tokenizer::ChatTokenizer;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const NONE: u32 = u32::MAX;
/// The 35B generation_config's top_k, and seeds per record for the sampled-draw check.
const TOP_K: usize = 20;
const SEEDS: u64 = 8;

fn ids(v: &serde_json::Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or(format!("golden json has no {key}"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as u32).ok_or(format!("{key}: not an id")))
        .collect()
}

fn golden(path: &str) -> Result<(Vec<u32>, Vec<u32>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok((ids(&v, "prompt_ids")?, ids(&v, "generated_ids")?))
}

fn append(out: &mut std::fs::File, src: u32, pos: u32, next: u32, full: u32, h: &[f32]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(16 + h.len() * 4);
    for v in [src, pos, next, full] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in h {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    out.write_all(&buf).map_err(|e| e.to_string())
}

/// The last position's normed hidden vector and the full argmax, from one tapped forward.
fn step(
    model: &mut Qwen38Reference<aien_inference_abi::qwen3_moe_omega::ShardSource>,
    state: &mut aien_inference_abi::qwen38::Qwen38State,
    tokens: &[u32],
) -> Result<(Vec<f32>, usize), String> {
    let mut taps = Vec::new();
    let layers = model.config.layers;
    let logits = model.forward_tapped(state, tokens, Logits::Last, &[layers], &mut taps)?;
    Ok((model.final_hidden(&taps[0]), argmax(&logits)))
}

fn open_out(path: &str) -> Result<std::fs::File, String> {
    std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{path}: {e}"))
}

fn hidden(dir: &Path, out_path: &str, goldens: &[String]) -> Result<(), String> {
    let mut model = Qwen38Reference::open(dir)?;
    let h = model.config.hidden;
    let mut out = open_out(out_path)?;
    for (src, path) in goldens.iter().enumerate() {
        let (prompt, gen) = golden(path)?;
        let t0 = Instant::now();
        let mut state = model.new_state();
        let (rows, full) = step(&mut model, &mut state, &prompt)?;
        // Prompt positions: the normed rows of every prompt token (all from one call).
        let t = prompt.len();
        for p in 0..t {
            let next = if p + 1 < t { prompt[p + 1] } else { gen[0] };
            let f = if p + 1 == t { full as u32 } else { NONE };
            append(&mut out, src as u32, p as u32, next, f, &rows[p * h..(p + 1) * h])?;
        }
        let mut mism = usize::from(full as u32 != gen[0]);
        for (i, &tok) in gen.iter().enumerate().take(gen.len() - 1) {
            let (row, full) = step(&mut model, &mut state, &[tok])?;
            mism += usize::from(full as u32 != gen[i + 1]);
            append(&mut out, src as u32, (t + i) as u32, gen[i + 1], full as u32, &row)?;
        }
        eprintln!(
            "{path}: {} prompt + {} generated positions in {:.1} s, full argmax != golden at {mism} steps",
            t,
            gen.len() - 1,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn passage(dir: &Path, out_path: &str, text_path: &str, count: usize, src: u32) -> Result<(), String> {
    let mut text = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;
    if let Some((cut, _)) = text.char_indices().nth(count * 8) {
        text.truncate(cut);
    }
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut ids = tok.encode(&text).map_err(|e| e.to_string())?;
    ids.truncate(count);
    let mut model = Qwen38Reference::open(dir)?;
    let h = model.config.hidden;
    let mut state = model.new_state();
    let t0 = Instant::now();
    let (rows, _) = step(&mut model, &mut state, &ids)?;
    let mut out = open_out(out_path)?;
    for p in 0..ids.len() {
        let next = ids.get(p + 1).copied().unwrap_or(NONE);
        append(&mut out, src, p as u32, next, NONE, &rows[p * h..(p + 1) * h])?;
    }
    eprintln!("{text_path}: {} positions in {:.1} s", ids.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

struct Record {
    src: u32,
    pos: u32,
    next: u32,
    full: u32,
    h: Vec<f32>,
}

fn read_records(path: &str, k: usize) -> Result<Vec<Record>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut bytes)).map_err(|e| format!("{path}: {e}"))?;
    let size = 16 + 4 * k;
    if bytes.len() % size != 0 {
        return Err(format!("{path}: {} bytes is not a whole number of records", bytes.len()));
    }
    let u = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    Ok(bytes
        .chunks_exact(size)
        .map(|r| Record {
            src: u(&r[0..4]),
            pos: u(&r[4..8]),
            next: u(&r[8..12]),
            full: u(&r[12..16]),
            h: r[16..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        })
        .collect())
}

fn quantiles(v: &mut [usize]) -> serde_json::Value {
    v.sort_unstable();
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
    serde_json::json!({ "mean": mean, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": v[v.len() - 1], "min": v[0] })
}

/// First-on-ties argmax over the candidate rows `c` (ascending) of the logits `l`.
fn pick(c: &[u32], l: &[f32]) -> usize {
    let mut best = c[0] as usize;
    for &r in c {
        if l[r as usize] > l[best] {
            best = r as usize;
        }
    }
    best
}

/// Mutant scales swept per bound: the largest one that loses an argmax is the bound's slack.
const SCALES: [f64; 12] = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0];

fn analyze(dir: &Path, hidden_path: &str, out_path: &str) -> Result<(), String> {
    let model = Qwen38Reference::open(dir)?;
    let head = model.lm_head();
    let (n, k) = (head.n(), head.k());
    let recs = read_records(hidden_path, k)?;
    eprintln!("{} records, head [{n}][{k}]", recs.len());
    let full_bytes = (n * k * 2) as u64;
    let row_bytes = (k * 2) as u64;
    // Exact logits once per record (today's host head: Matrix::matmul).
    let t0 = Instant::now();
    let full: Vec<(usize, Vec<f32>)> = recs
        .iter()
        .map(|r| {
            let l = head.matmul(&r.h, 1);
            (argmax(&l), l)
        })
        .collect();
    eprintln!("full head on every record in {:.1} s", t0.elapsed().as_secs_f64());
    let stored = recs.iter().zip(&full).filter(|(r, f)| r.full != NONE && r.full as usize != f.0).count();
    let mut variants = Vec::new();
    for kind in [ScreenKind::W4, ScreenKind::Fp8, ScreenKind::Fp8Block] {
        let t0 = Instant::now();
        let screen = Screen::build(head, kind)?;
        eprintln!("{kind:?} screen built in {:.1} s", t0.elapsed().as_secs_f64());
        let s: Vec<Vec<f32>> = recs.iter().map(|r| screen.logits(&r.h)).collect();
        let screen_top1 = s.iter().zip(&full).filter(|(s, f)| argmax(s) == f.0).count();
        for bk in [BoundKind::Row, BoundKind::Group, BoundKind::GroupLinf, BoundKind::Min] {
            let mut sizes = Vec::new();
            let mut sizes_tau = Vec::new();
            let mut wrong = 0;
            let mut wrong_tau = 0;
            let mut first_fail: Option<f64> = None;
            let mut fails_at = Vec::new();
            for (i, r) in recs.iter().enumerate() {
                let b = screen.bounds(&r.h, bk, 1.0);
                let c = exact_head::candidates(&s[i], &b);
                // The recheck's logits are the full head's bits (same widen + dot), so the
                // full logits stand in for it here; gate mode and the unit tests run the recheck.
                let got = pick(&c, &full[i].1);
                if i == 0 {
                    assert_eq!(exact_head::recheck(head, &r.h, &c).0, got, "recheck != full head bits");
                }
                wrong += usize::from(got != full[i].0);
                sizes.push(c.len());
                // Exact threshold: recheck the best upper bound first; its exact logit is a lower
                // bound on the winner's, so tau = max(that, max_j (s_j - B_j)).
                let ub = (0..n).max_by(|&a, &z| (s[i][a] as f64 + b[a]).total_cmp(&(s[i][z] as f64 + b[z]))).unwrap();
                let ct = exact_head::candidates_from(&s[i], &b, full[i].1[ub] as f64);
                wrong_tau += usize::from(pick(&ct, &full[i].1) != full[i].0);
                sizes_tau.push(ct.len() + usize::from(ct.binary_search(&(ub as u32)).is_err()));
            }
            // Mutant sweep on the plain threshold: scaled bounds until an argmax is lost.
            for &sc in &SCALES[1..] {
                let fails = recs
                    .iter()
                    .enumerate()
                    .filter(|(i, r)| {
                        let b = screen.bounds(&r.h, bk, sc);
                        pick(&exact_head::candidates(&s[*i], &b), &full[*i].1) != full[*i].0
                    })
                    .count();
                fails_at.push(serde_json::json!({ "scale": sc, "argmax_lost": fails }));
                if fails > 0 && first_fail.is_none() {
                    first_fail = Some(sc);
                }
            }
            // Sampling (the 35B's generation_config: temperature 1.0, top_k 20, top_p 0.95): the
            // door masks to the top 20 first, so candidates that certify the top 20 make the draw
            // a function of exact logits only. Checked bit for bit against the full head's draw.
            let topk: Vec<(usize, usize)> = recs
                .par_iter()
                .enumerate()
                .map(|(i, r)| {
                    let b = screen.bounds(&r.h, bk, 1.0);
                    let c = exact_head::candidates_topk(&s[i], &b, TOP_K, f64::NEG_INFINITY);
                    let mut masked = vec![f32::NEG_INFINITY; n];
                    for &row in &c {
                        masked[row as usize] = full[i].1[row as usize];
                    }
                    let differ = if bk == BoundKind::Min {
                        (0..SEEDS)
                            .filter(|&seed| {
                                let p = GenerateParams { max_new_tokens: 1, temperature: 1.0, top_p: 0.95, top_k: TOP_K, seed, stop_token_ids: Vec::new() };
                                sample_next(&full[i].1, &p, i) != sample_next(&masked, &p, i)
                            })
                            .count()
                    } else {
                        0
                    };
                    (c.len(), differ)
                })
                .collect();
            let mut qk: Vec<usize> = topk.iter().map(|v| v.0).collect();
            let draws_differ: usize = topk.iter().map(|v| v.1).sum();
            let fixed = kind.bytes(n, k) + bk.bytes(n, k);
            let mut q = sizes.clone();
            let mut qt = sizes_tau.clone();
            let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
            let bpt = |c: f64| fixed as f64 + c * row_bytes as f64;
            eprintln!(
                "{kind:?} {bk:?}: |C| mean {:.1} max {}, exact-tau mean {:.1} max {}, wrong {wrong}/{wrong_tau}, first mutant fail at x{first_fail:?}",
                mean(&sizes),
                sizes.iter().max().unwrap(),
                mean(&sizes_tau),
                sizes_tau.iter().max().unwrap()
            );
            variants.push(serde_json::json!({
                "screen": format!("{kind:?}"), "bound": format!("{bk:?}"),
                "screen_top1_equals_full": screen_top1, "records": recs.len(),
                "C": quantiles(&mut q), "C_exact_tau": quantiles(&mut qt),
                "argmax_wrong": wrong, "argmax_wrong_exact_tau": wrong_tau,
                "fixed_bytes": fixed, "bytes_per_token_mean": bpt(mean(&sizes)),
                "bytes_per_token_mean_exact_tau": bpt(mean(&sizes_tau)),
                "bytes_per_token_p99_exact_tau": bpt(qt[((qt.len() - 1) as f64 * 0.99).round() as usize] as f64),
                "mutant_sweep": fails_at, "largest_scale_that_fails": first_fail,
                "C_top20": quantiles(&mut qk), "bytes_per_token_mean_top20": bpt(qk.iter().sum::<usize>() as f64 / qk.len() as f64),
                "sampled_draws_checked": if bk == BoundKind::Min { recs.len() as u64 * SEEDS } else { 0 },
                "sampled_draws_differ": draws_differ,
                "C_per_record": sizes, "C_exact_tau_per_record": sizes_tau,
            }));
        }
    }
    let per: Vec<_> = recs
        .iter()
        .zip(&full)
        .map(|(r, f)| {
            let mut sorted = f.1.clone();
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            let hn = r.h.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            serde_json::json!({ "src": r.src, "pos": r.pos, "next": r.next, "full_argmax": f.0,
                                "top1": sorted[0], "gap12": sorted[0] - sorted[1], "hidden_norm": hn })
        })
        .collect();
    let report = serde_json::json!({
        "head": { "rows": n, "cols": k, "group": GROUP, "full_bytes": full_bytes, "row_bytes": row_bytes },
        "gamma": exact_head::gamma(k), "records": recs.len(),
        "stored_full_argmax_disagrees_with_recomputed": stored,
        "variants": variants, "per_record": per,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())
}

fn gate(dir: &Path, kind: ScreenKind, bk: BoundKind, scale: f64, out_path: &str, goldens: &[String]) -> Result<bool, String> {
    let mut model = Qwen38Reference::open(dir)?;
    let screen = Screen::build(model.lm_head(), kind)?;
    let mut results = Vec::new();
    let mut pass = true;
    for path in goldens {
        let (prompt, gen) = golden(path)?;
        let mut state = model.new_state();
        let (mut row, mut full) = step(&mut model, &mut state, &prompt)?;
        let mut out = Vec::new();
        let mut cs = Vec::new();
        let mut diverged = None;
        for i in 0..gen.len() {
            let (tok, c) = exact_head::argmax(model.lm_head(), &screen, &row, bk, scale);
            out.push(tok as u32);
            cs.push(c);
            if tok as u32 != gen[i] || tok != full {
                diverged = Some(i);
                break;
            }
            if i + 1 < gen.len() {
                (row, full) = step(&mut model, &mut state, &[tok as u32])?;
            }
        }
        pass &= diverged.is_none();
        eprintln!("{path}: {} ({} tokens, max |C| {})", if diverged.is_none() { "IDENTICAL" } else { "DIVERGED" }, out.len(), cs.iter().max().unwrap());
        results.push(serde_json::json!({ "golden": path, "identical": diverged.is_none(), "first_divergence": diverged,
                                         "ids": out, "golden_ids": gen, "C": cs }));
    }
    let report = serde_json::json!({ "screen": format!("{kind:?}"), "bound": format!("{bk:?}"), "scale": scale,
                                     "verdict": if pass { "PASS" } else { "FAIL" }, "runs": results });
    std::fs::write(out_path, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    Ok(pass)
}

/// Roundings per term the chip screen may add: k products and sums plus the 16 block partials.
const CHIP_GAMMA_LEN: usize = 2100;

/// The chip test's inputs (weights: outside git): head.bf16 [n][k] u16, fp8b.codes [n][k] e4m3,
/// fp8b.scales [n/128][k/128] bf16, coef.f32 [n] (the Row bound coefficient, gamma(2100)).
fn export(dir: &Path, out: &str) -> Result<(), String> {
    let model = Qwen38Reference::open(dir)?;
    let head = model.lm_head();
    let (n, k) = (head.n(), head.k());
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let put = |name: &str, bytes: &[u8]| std::fs::write(format!("{out}/{name}"), bytes).map_err(|e| format!("{name}: {e}"));
    let mut raw = Vec::with_capacity(n * k * 2);
    for r in 0..n {
        raw.extend(head.bf16_row(r).unwrap().iter().flat_map(|v| v.to_le_bytes()));
    }
    put("head.bf16", &raw)?;
    drop(raw);
    let scales = exact_head::fp8_block_scales(head);
    let codes = exact_head::fp8_block_codes(head, &scales);
    let screen = Screen::build(head, ScreenKind::Fp8Block)?;
    let bad = screen.codes_mismatch(&codes, &scales);
    if bad != 0 {
        return Err(format!("export: {bad} rows of codes do not reproduce the screen"));
    }
    put("fp8b.codes", &codes)?;
    let sb: Vec<u8> = scales.iter().flat_map(|&s| aien_inference_abi::qwen3_moe::f32_to_bf16(s).to_le_bytes()).collect();
    put("fp8b.scales", &sb)?;
    let coef = screen.row_coef(CHIP_GAMMA_LEN);
    put("coef.f32", &coef.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let manifest = serde_json::json!({ "n": n, "k": k, "group": GROUP, "gamma_len": CHIP_GAMMA_LEN,
        "files": ["head.bf16", "fp8b.codes", "fp8b.scales", "coef.f32"], "codes_reproduce_screen_rows": n });
    put("manifest.json", serde_json::to_string_pretty(&manifest).unwrap().as_bytes())?;
    eprintln!("export: {n} x {k} to {out}, codes reproduce the fp8b screen on every row");
    Ok(())
}

fn run() -> Result<i32, String> {
    let a: Vec<String> = std::env::args().collect();
    let usage = "usage: q36_exact_head hidden|passage|analyze|gate ... (see the file header)";
    match a.get(1).map(String::as_str) {
        Some("hidden") if a.len() >= 5 => hidden(Path::new(&a[2]), &a[3], &a[4..]).map(|_| 0),
        Some("passage") if a.len() == 7 => {
            let count = a[5].parse().map_err(|_| usage.to_string())?;
            let src = a[6].parse().map_err(|_| usage.to_string())?;
            passage(Path::new(&a[2]), &a[3], &a[4], count, src).map(|_| 0)
        }
        Some("analyze") if a.len() == 5 => analyze(Path::new(&a[2]), &a[3], &a[4]).map(|_| 0),
        Some("export") if a.len() == 4 => export(Path::new(&a[2]), &a[3]).map(|_| 0),
        Some("gate") if a.len() >= 8 => {
            let kind = ScreenKind::parse(&a[3]).ok_or(usage)?;
            let bk = BoundKind::parse(&a[4]).ok_or(usage)?;
            let scale = a[5].parse().map_err(|_| usage.to_string())?;
            gate(Path::new(&a[2]), kind, bk, scale, &a[6], &a[7..]).map(|p| if p { 0 } else { 1 })
        }
        _ => Err(usage.into()),
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("q36_exact_head: {e}");
            std::process::exit(2);
        }
    }
}
