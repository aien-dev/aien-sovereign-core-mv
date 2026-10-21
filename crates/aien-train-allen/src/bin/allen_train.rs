//! Small Allen trainer (L389), host CPU only.
//!
//!   allen_train parity <model-dir> <text-file> <tokens> <out.json>
//!       the trainer's forward against the host reference (`Qwen38Reference`) on the same weights
//!   allen_train bench <model-dir> <tokens> <reps>
//!       one training sequence (forward + backward), timed
//!   allen_train train <model-dir> <train.jsonl> <val.jsonl> <run-dir> [key=value ...]
//!       LoRA fine-tune; keys: rank alpha lr steps batch warmup max_len eval_every save_every seed
//!       resumes from <run-dir>/adapters.bin; writes <run-dir>/merged at the end
//!   allen_train export-gpu <model-dir> <out.bin>
//!       the weights as one flat file for the iMac GPU kernels (see `io::export_gpu`)
//!
//! Threads: RAYON_NUM_THREADS (run it under `nice -n 19`; the GB10 is not used at all).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, negative_log_likelihood, Logits, Qwen38Reference};
use aien_inference_abi::tokenizer::ChatTokenizer;
use aien_train_allen::{io, model::SplitMix, train};
use serde_json::json;

fn parity(dir: &Path, text: &str, n: usize, out: &str) -> Result<(), String> {
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut text = std::fs::read_to_string(text).map_err(|e| e.to_string())?;
    if let Some((cut, _)) = text.char_indices().nth(n * 8) {
        text.truncate(cut);
    }
    let mut ids = tok.encode(&text).map_err(|e| e.to_string())?;
    ids.truncate(n);
    let t0 = Instant::now();
    let mut reference = Qwen38Reference::open(dir)?;
    let mut state = reference.new_state();
    let want = reference.forward(&mut state, &ids, Logits::All)?;
    let ref_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let model = io::load(dir)?;
    let load_s = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let got = model.logits(&ids);
    let ours_s = t2.elapsed().as_secs_f64();
    let v = model.c.vocab;
    let (mut max_abs, mut argmax_diff, mut argmax_diff_close) = (0f32, 0usize, 0usize);
    let (mut nll_ref, mut nll_ours) = (0f64, 0f64);
    for i in 0..ids.len() {
        let (a, b) = (&want[i * v..(i + 1) * v], &got[i * v..(i + 1) * v]);
        for (x, y) in a.iter().zip(b) {
            max_abs = max_abs.max((x - y).abs());
        }
        if argmax(a) != argmax(b) {
            let mut s = a.to_vec();
            s.sort_by(|p, q| q.total_cmp(p));
            if s[0] - s[1] <= 0.02 {
                argmax_diff_close += 1;
            } else {
                argmax_diff += 1;
            }
        }
        if i + 1 < ids.len() {
            nll_ref += negative_log_likelihood(a, ids[i + 1] as usize);
            nll_ours += negative_log_likelihood(b, ids[i + 1] as usize);
        }
    }
    let m = (ids.len() - 1) as f64;
    let (nll_ref, nll_ours) = (nll_ref / m, nll_ours / m);
    let pass = max_abs <= 0.02 && argmax_diff == 0 && (nll_ref - nll_ours).abs() <= 1e-3;
    let report = json!({
        "tokens": ids.len(), "max_abs_logit_diff": max_abs, "argmax_differ_clear": argmax_diff,
        "argmax_differ_within_0.02_gap": argmax_diff_close, "mean_nll_reference": nll_ref,
        "mean_nll_trainer": nll_ours, "tolerance": "max_abs <= 0.02, no argmax change where the reference top-2 gap > 0.02, |mean nll diff| <= 1e-3 (pre-registered)",
        "pass": pass, "reference_s": ref_s, "trainer_load_s": load_s, "trainer_forward_s": ours_s,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!("{report}");
    if pass {
        Ok(())
    } else {
        Err("parity: outside the pre-registered tolerance".into())
    }
}

/// Multiply-adds per token of the linear maps (forward), for the speed receipt.
fn macs_per_token(m: &aien_train_allen::model::Model) -> (u64, u64) {
    let lin: u64 = m.lins().iter().map(|(_, l)| (l.n * l.k) as u64).sum();
    (lin, (m.c.vocab * m.c.hidden) as u64)
}

fn bench(dir: &Path, t: usize, reps: usize) -> Result<(), String> {
    let mut model = io::load(dir)?;
    model.add_lora(16, 32.0, 389, 0.0);
    let mut rng = SplitMix(1);
    let tokens: Vec<u32> = (0..t).map(|_| (rng.next_u64() % 200_000) as u32).collect();
    let weights: Vec<f32> = (0..t).map(|i| if i >= t / 2 { 1.0 } else { 0.0 }).collect();
    let (lin, head) = macs_per_token(&model);
    for r in 0..reps {
        let t0 = Instant::now();
        let l = model.train_seq(&tokens, &weights);
        let s = t0.elapsed().as_secs_f64();
        let loss_rows = weights.iter().filter(|&&w| w > 0.0).count() as u64;
        // forward + dX backward over every map (2 x 2 flops per MAC), the head forward and
        // backward on the loss rows only; adapters and the sequence mixers not counted.
        let flops = 4 * lin * t as u64 + 4 * head * loss_rows;
        println!(
            "{}",
            json!({"rep": r, "tokens": t, "loss_rows": loss_rows, "seconds": s, "tokens_per_s": t as f64 / s,
                   "counted_gflop": flops as f64 / 1e9, "gflop_per_s": flops as f64 / 1e9 / s,
                   "mean_nll": l.loss / l.weight, "threads": rayon::current_num_threads()})
        );
        for (_, p) in model.params_mut() {
            p.g.iter_mut().for_each(|g| *g = 0.0);
        }
    }
    Ok(())
}

struct Opts {
    rank: usize,
    alpha: f32,
    lr: f32,
    steps: u64,
    batch: usize,
    warmup: u64,
    max_len: usize,
    eval_every: u64,
    save_every: u64,
    seed: u64,
}

fn opts(kv: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        rank: 16,
        alpha: 32.0,
        lr: 2e-4,
        steps: 600,
        batch: 8,
        warmup: 20,
        max_len: 512,
        eval_every: 50,
        save_every: 25,
        seed: 389,
    };
    for a in kv {
        let (k, v) = a
            .split_once('=')
            .ok_or_else(|| format!("{a}: want key=value"))?;
        let bad = |e: std::fmt::Arguments| format!("{a}: bad value ({e})");
        match k {
            "rank" => o.rank = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "alpha" => o.alpha = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "lr" => o.lr = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "steps" => o.steps = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "batch" => o.batch = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "warmup" => o.warmup = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "max_len" => o.max_len = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "eval_every" => o.eval_every = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "save_every" => o.save_every = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            "seed" => o.seed = v.parse().map_err(|e| bad(format_args!("{e}")))?,
            _ => return Err(format!("{k}: unknown key")),
        }
    }
    Ok(o)
}

fn examples(
    tok: &ChatTokenizer,
    path: &Path,
    max_len: usize,
) -> Result<(Vec<train::Example>, usize), String> {
    let mut out = Vec::new();
    let mut dropped = 0;
    for line in train::read_lines(path)? {
        let e = train::Example::from_line(tok, &line)?;
        if e.tokens.len() > max_len || !e.weights.iter().any(|&w| w > 0.0) {
            dropped += 1;
        } else {
            out.push(e);
        }
    }
    Ok((out, dropped))
}

fn run_train(dir: &Path, train_p: &Path, val_p: &Path, run: &Path, o: Opts) -> Result<(), String> {
    std::fs::create_dir_all(run).map_err(|e| e.to_string())?;
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let (mut data, dropped) = examples(&tok, train_p, o.max_len)?;
    let (val, vdropped) = examples(&tok, val_p, o.max_len)?;
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.join("train.log.jsonl"))
        .map_err(|e| e.to_string())?;
    let mut emit = |v: serde_json::Value| {
        println!("{v}");
        let _ = writeln!(log, "{v}");
    };
    let t0 = Instant::now();
    let mut model = io::load(dir)?;
    model.add_lora(o.rank, o.alpha, o.seed, 0.0);
    let ad = run.join("adapters.bin");
    let mut done = if ad.exists() {
        train::load_adapters(&mut model, &ad)?
    } else {
        0
    };
    emit(
        json!({"event": "start", "train": data.len(), "train_dropped_over_max_len": dropped,
                "val": val.len(), "val_dropped": vdropped, "resume_step": done, "load_s": t0.elapsed().as_secs_f64(),
                "rank": o.rank, "alpha": o.alpha, "lr": o.lr, "steps": o.steps, "batch": o.batch,
                "max_len": o.max_len, "threads": rayon::current_num_threads()}),
    );
    // A fixed order per epoch from the seed; resuming replays the same order.
    let order = |epoch: u64, n: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = SplitMix(o.seed ^ (epoch.wrapping_mul(0x9e37_79b9)));
        for i in (1..n).rev() {
            idx.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        idx
    };
    let n = data.len();
    let per_epoch = (n / o.batch).max(1) as u64;
    let eval = |model: &aien_train_allen::model::Model| {
        let (mut l, mut w, mut h) = (0f64, 0f64, 0f64);
        for e in &val {
            let s = model.loss(&e.tokens, &e.weights);
            l += s.loss;
            w += s.weight;
            h += s.hits;
        }
        (l / w.max(1e-9), h / w.max(1e-9))
    };
    if done == 0 {
        let (vl, va) = eval(&model);
        emit(json!({"event": "val", "step": 0, "val_nll": vl, "val_acc": va}));
    }
    while done < o.steps {
        let step_no = done + 1;
        let epoch = done / per_epoch;
        let ord = order(epoch, n);
        let b0 = ((done % per_epoch) as usize) * o.batch;
        let lr = if step_no <= o.warmup {
            o.lr * step_no as f32 / o.warmup as f32
        } else {
            let p = (step_no - o.warmup) as f32 / (o.steps - o.warmup).max(1) as f32;
            o.lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()))
        };
        let t1 = Instant::now();
        let (mut loss, mut weight, mut hits, mut toks) = (0f64, 0f64, 0f64, 0usize);
        for &i in &ord[b0..(b0 + o.batch).min(n)] {
            let e = &mut data[i];
            let s = model.train_seq(&e.tokens, &e.weights);
            loss += s.loss;
            weight += s.weight;
            hits += s.hits;
            toks += e.tokens.len();
        }
        let mut params = model.params_mut();
        let gnorm = train::step(
            &mut params,
            weight,
            1.0,
            train::Adam {
                lr,
                b1: 0.9,
                b2: 0.999,
                eps: 1e-8,
                wd: 0.0,
            },
            step_no,
        );
        drop(params);
        done = step_no;
        let s = t1.elapsed().as_secs_f64();
        emit(
            json!({"event": "step", "step": done, "epoch": epoch, "lr": lr, "nll": loss / weight.max(1e-9),
                    "acc": hits / weight.max(1e-9), "grad_norm": gnorm, "tokens": toks, "loss_tokens": weight,
                    "seconds": s, "tokens_per_s": toks as f64 / s}),
        );
        if done % o.save_every == 0 || done == o.steps {
            train::save_adapters(&mut model, &ad, done)?;
        }
        if done % o.eval_every == 0 || done == o.steps {
            let (vl, va) = eval(&model);
            emit(json!({"event": "val", "step": done, "val_nll": vl, "val_acc": va}));
        }
    }
    let readme = format!(
        "# allen-small (L389), PRIVATE\n\nQwen3.5-0.8B (Qwen/Qwen3.5-0.8B @2fc06364, Apache-2.0) with a rank-{} LoRA (alpha {}) merged in, \
         trained by aien-train-allen on Drake's private teaching set (~/datasets/allen-drake). \
         Text tensors only. Never publish, never push: Drake's data shaped these weights.\n\nSteps {}, batch {}, lr {}.\n",
        o.rank, o.alpha, o.steps, o.batch, o.lr
    );
    let merged = run.join("merged");
    let k = io::save_merged(&model, dir, &merged, &readme)?;
    emit(json!({"event": "merged", "dir": merged.display().to_string(), "maps_replaced": k}));
    Ok(())
}

/// Greedy answers for prompt lines (`segs` without the answer), resumable by id.
fn generate(dir: &Path, prompts: &Path, out: &Path, max_new: usize) -> Result<(), String> {
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut done = std::collections::HashSet::new();
    if let Ok(t) = std::fs::read_to_string(out) {
        for l in t.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                if let Some(id) = v["id"].as_str() {
                    done.insert(id.to_string());
                }
            }
        }
    }
    let t0 = Instant::now();
    let model = io::load(dir)?;
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let mut stop: Vec<u32> = model.c.eos_token_ids.clone();
    for s in ["<|im_end|>", "<|endoftext|>"] {
        if let Ok(ids) = tok.encode_with_special(s, false) {
            if ids.len() == 1 {
                stop.push(ids[0]);
            }
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .map_err(|e| e.to_string())?;
    for line in train::read_lines(prompts)? {
        if done.contains(&line.id) {
            continue;
        }
        let e = train::Example::from_line(&tok, &line)?;
        let t1 = Instant::now();
        let ids = model.generate(&e.tokens, max_new, &stop);
        let s = t1.elapsed().as_secs_f64();
        let body: Vec<u32> = ids.iter().copied().filter(|t| !stop.contains(t)).collect();
        let text = tok.decode(&body).map_err(|e| e.to_string())?;
        let rec = json!({"id": line.id, "split": line.split, "text": text, "prompt_tokens": e.tokens.len(),
                         "new_tokens": ids.len(), "stopped": ids.last().map(|t| stop.contains(t)).unwrap_or(false),
                         "seconds": s, "tokens": ids});
        writeln!(f, "{rec}").map_err(|e| e.to_string())?;
        eprintln!(
            "{} {} tokens {:.1} s: {}",
            line.id,
            ids.len(),
            s,
            text.chars().take(80).collect::<String>()
        );
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("parity") if a.len() == 6 => {
            parity(Path::new(&a[2]), &a[3], a[4].parse().unwrap_or(64), &a[5])
        }
        Some("bench") if a.len() == 5 => bench(
            Path::new(&a[2]),
            a[3].parse().unwrap_or(256),
            a[4].parse().unwrap_or(2),
        ),
        Some("train") if a.len() >= 6 => opts(&a[6..]).and_then(|o| {
            run_train(
                Path::new(&a[2]),
                Path::new(&a[3]),
                Path::new(&a[4]),
                Path::new(&a[5]),
                o,
            )
        }),
        Some("generate") if a.len() >= 5 => generate(
            Path::new(&a[2]),
            Path::new(&a[3]),
            Path::new(&a[4]),
            a.get(5).and_then(|v| v.parse().ok()).unwrap_or(160),
        ),
        Some("export-gpu") if a.len() == 4 => {
            io::load(Path::new(&a[2])).and_then(|m| io::export_gpu(&m, Path::new(&a[3])))
        }
        Some("gemmbench") => {
            for &(m, k, n) in &[
                (256usize, 1024usize, 3584usize),
                (256, 3584, 1024),
                (256, 1024, 6144),
                (64, 1024, 248320),
            ] {
                let a = vec![0.5f32; m * k];
                let b = vec![0.25f32; k * n];
                let mut c = vec![0f32; m * n];
                aien_train_allen::gemm::gemm(&a, &b, &mut c, m, k, n, false);
                let t0 = Instant::now();
                for _ in 0..3 {
                    aien_train_allen::gemm::gemm(&a, &b, &mut c, m, k, n, false);
                }
                let s = t0.elapsed().as_secs_f64() / 3.0;
                println!(
                    "{}",
                    json!({"m": m, "k": k, "n": n, "seconds": s, "gflop_per_s": 2.0 * (m * k * n) as f64 / s / 1e9})
                );
            }
            Ok(())
        }
        _ => Err("usage: see the header of src/bin/allen_train.rs".into()),
    };
    if let Err(e) = r {
        eprintln!("allen_train: {e}");
        std::process::exit(1);
    }
}
