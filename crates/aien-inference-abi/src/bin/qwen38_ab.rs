//! A/B drift and speed tool for Qwen3.8-27B on the Omega decode graph (FP8 lm_head on/off,
//! 4-bit weights): every number comes from the whole model, all 64 layers resident.
//!
//!   qwen38_ab text <checkpoint-dir> <text-file> <n> <seq.json>
//!       the first `n` tokens of a plain text (no chat template), as `{"ids": [...]}`.
//!   qwen38_ab run <checkpoint-dir> <seq.json> <out-prefix> [--gen N] [--tail N]
//!       teacher-forced: feeds `ids` one decode step at a time; with `--gen N` the ids are a
//!       prompt and N more tokens follow greedily (free running). Writes `<out-prefix>.logits`
//!       (f32, vocab per step, step order) (with `--tail N` only the last N steps' logits, so a long
//!       text stays small) and `<out-prefix>.json` (per step: fed token, chip
//!       argmax, top-two margin, chip ms, and the negative log-likelihood of the next fed token).
//!       The engine variant comes from the environment (AIEN_Q38_FP8_HEAD=1, AIEN_Q38_W4=1,
//!       AIEN_Q38_KV8=1 for the 8-bit attention cache).
//!   qwen38_ab compare <a-prefix> <b-prefix> <out.json>
//!       position by position, A taken as the reference: argmax agreement, A's top-two margin
//!       where they differ, logits cosine, KL(A || B), and both texts' mean NLL. Two runs fed the
//!       same tokens see the same inputs, so with teacher forcing only the variant differs.
//!
//!   qwen38_ab speed <checkpoint-dir> <ctx> <steps> <seed> <out.json>
//!       BENCHMARK ONLY: decode speed at context `ctx` without reading a prompt. The attention
//!       cache is filled with a bounded pattern up to `ctx - steps`, then `steps` greedy steps
//!       run; the tokens are meaningless, the chip and wall times per step are the result.
//!
//! A run refuses to start unless MemAvailable covers the model, its KV cache and 12 GiB
//! for everything else on the box (the live server shares the pool).

use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, Qwen38Config};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};

const KEEP_FREE: u64 = 12 << 30;
/// Host copies held while loading (token embeddings, one converted layer, slack).
const HOST_SLACK: u64 = 6 << 30;
const TOKENIZE_PIECE: usize = 2000;

fn mem_available() -> Result<u64, String> {
    let m = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    m.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb << 10)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

/// Device bytes of the whole model at FP8 (one byte per projection weight, bf16 a|b and lm_head);
/// the 4-bit variant needs less, so this bound holds for every variant.
fn model_bytes(c: &Qwen38Config) -> u64 {
    let h = c.hidden as u64;
    let mlp = 3 * c.intermediate as u64 * h;
    let o = h * (c.heads * c.head_dim) as u64;
    let layers: u64 = c
        .full_attention
        .iter()
        .map(|&attn| {
            if attn {
                (2 * c.heads * c.head_dim + 2 * c.kv_dim()) as u64 * h + o + mlp
            } else {
                (c.conv_dim() + c.lin_value_dim()) as u64 * h
                    + o
                    + mlp
                    + 2 * 2 * c.lin_v_heads as u64 * h
            }
        })
        .sum();
    layers + layers / 4096 + 2 * c.vocab as u64 * h
}

/// The engine variant as the environment sets it, recorded in every output.
fn variant() -> Value {
    let v = |k: &str| std::env::var(k).ok();
    json!({
        "AIEN_Q38_FP8_HEAD": v("AIEN_Q38_FP8_HEAD"),
        "AIEN_Q38_W4": v("AIEN_Q38_W4"),
        "AIEN_Q38_W4_HEAD": v("AIEN_Q38_W4_HEAD"),
        "AIEN_Q38_W4_RULE": v("AIEN_Q38_W4_RULE"),
        "AIEN_Q38_KV8": v("AIEN_Q38_KV8"),
    })
}

/// Attention KV cache bytes at `max_pos`: 16 layers x K and V x 4 KV heads x 256 wide, bf16
/// (512 bytes a row) or 8-bit (256 bytes + a 4-byte scale).
fn kv_cache_bytes(c: &Qwen38Config, max_pos: usize) -> u64 {
    let n_attn = c.full_attention.iter().filter(|&&a| a).count() as u64;
    let row = if std::env::var("AIEN_Q38_KV8").as_deref() == Ok("1") {
        256 + 4
    } else {
        512
    };
    n_attn * 2 * 4 * max_pos as u64 * row
}

/// Refuses to start unless MemAvailable covers the model (its variant's copies), the KV cache at
/// `max_pos`, host slack and 12 GiB kept free. Returns MemAvailable.
fn memory_guard(config: &Qwen38Config, max_pos: usize) -> Result<u64, String> {
    // the MXFP4 copies sit beside the FP8 tensors (the prefill keeps FP8): + 17 / 32 of the projection bytes, + the head's
    let fp8 = model_bytes(config);
    let head = 2 * config.vocab as u64 * config.hidden as u64;
    let mut need = fp8 + HOST_SLACK + kv_cache_bytes(config, max_pos);
    if std::env::var("AIEN_Q38_W4").as_deref() == Ok("1") {
        need += (fp8 - head) * 17 / 32;
    }
    if std::env::var("AIEN_Q38_W4_HEAD").as_deref() == Ok("1") {
        need += head / 2 * 17 / 32;
    }
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "memory guard: MemAvailable {} MiB < model + KV cache {} MiB + 12 GiB kept free",
            avail >> 20,
            need >> 20
        ));
    }
    Ok(avail)
}

fn margin(logits: &[f32]) -> f32 {
    let top = argmax(logits);
    let second = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != top)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    logits[top] - second
}

/// log(sum exp(l)) in f64.
fn logsumexp(l: &[f32]) -> f64 {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

fn read_json(p: &str) -> Result<Value, String> {
    serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)
        .map_err(|e| format!("{p}: {e}"))
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{key} holds a non-id"))
        })
        .collect()
}

fn text(dir: &Path, file: &str, n: usize, out: &str) -> Result<(), String> {
    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let body = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let mut toks = Vec::with_capacity(n);
    let mut rest = body.as_str();
    while toks.len() < n && !rest.is_empty() {
        let mut cut = rest.len().min(TOKENIZE_PIECE);
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        toks.extend(tk.encode(&rest[..cut]).map_err(|e| e.to_string())?);
        rest = &rest[cut..];
    }
    if toks.len() < n {
        return Err(format!("{file}: only {} tokens", toks.len()));
    }
    toks.truncate(n);
    let v = json!({ "source": file, "ids": toks });
    std::fs::write(out, serde_json::to_string(&v).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{out}: {e}"))
}

fn run(dir: &Path, seq_path: &str, prefix: &str, gen: usize, tail: usize) -> Result<(), String> {
    let seq = ids(&read_json(seq_path)?, "ids")?;
    if seq.len() < 2 && gen == 0 {
        return Err("seq: fewer than 2 ids".into());
    }
    let config = Qwen38Config::from_checkpoint(dir)?;
    let (vocab, steps) = (
        config.vocab,
        if gen > 0 {
            seq.len() + gen - 1
        } else {
            seq.len() - 1
        },
    );
    // logits are written from this step on (`--tail N`: the last N steps; a 32k text would be 32 GB)
    let logits_from = if tail > 0 {
        steps.saturating_sub(tail)
    } else {
        0
    };
    let max_pos = (steps + 1).max(256).next_multiple_of(256);
    let avail = memory_guard(&config, max_pos)?;
    let variant = variant();
    eprintln!(
        "variant {variant}; {steps} steps; MemAvailable {} MiB",
        avail >> 20
    );
    let t_open = Instant::now();
    let mut e = Qwen38Omega::open(dir, None, max_pos)?;
    let open_s = t_open.elapsed().as_secs_f64();
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(format!("{prefix}.logits")).map_err(|e| format!("{prefix}: {e}"))?,
    );
    let mut rows = Vec::with_capacity(steps);
    let mut fed = seq[0];
    let mut generated = Vec::new();
    let mut nll_sum = 0f64;
    let mut nll_n = 0usize;
    let t_all = Instant::now();
    for p in 0..steps {
        let t = Instant::now();
        let (top, st) = e.step(fed)?;
        let l = e.logits()?;
        let wall_ms = t.elapsed().as_secs_f64() * 1e3;
        if l.len() != vocab || argmax(&l) != top as usize {
            return Err(format!(
                "step {p}: logits read back disagree with the chip argmax"
            ));
        }
        let next = if p + 1 < seq.len() {
            seq[p + 1]
        } else {
            generated.push(top);
            top
        };
        let nll = logsumexp(&l) - l[next as usize] as f64;
        if p + 1 < seq.len() {
            nll_sum += nll;
            nll_n += 1;
        }
        if p >= logits_from {
            for v in &l {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        rows.push(json!({
            "p": p, "fed": fed, "top": top, "margin": margin(&l), "next": next, "nll": nll,
            "chip_ms": st.chip_ns as f64 / 1e6, "wall_ms": wall_ms,
        }));
        if p % 32 == 0 {
            eprintln!(
                "step {p:4}: fed {fed} top {top} chip {:.2} ms",
                st.chip_ns as f64 / 1e6
            );
        }
        fed = next;
    }
    f.flush().map_err(|e| e.to_string())?;
    let mut chip: Vec<f64> = rows
        .iter()
        .map(|r| r["chip_ms"].as_f64().unwrap())
        .collect();
    chip.sort_by(|a, b| a.total_cmp(b));
    let median = chip[chip.len() / 2];
    let out = json!({
        "mode": "run", "seq": seq_path, "variant": variant, "steps": steps, "vocab": vocab, "logits_from": logits_from,
        "teacher_forced": seq.len() - 1, "generated_ids": generated,
        "mean_nll_fed": if nll_n > 0 { nll_sum / nll_n as f64 } else { f64::NAN },
        "chip_ms_median": median, "chip_ms_min": chip[0], "chip_ms_max": chip[chip.len() - 1],
        "open_s": open_s, "run_s": t_all.elapsed().as_secs_f64(),
        "w4_rel_rms": e.w4_stats.map(|s| s.rel_rms()), "w4_weights": e.w4_stats.map(|s| s.weights),
        "w4_zero_groups": e.w4_stats.map(|s| s.zero_groups), "w4_saturated": e.w4_stats.map(|s| s.saturated),
        "rows": rows,
    });
    std::fs::write(
        format!("{prefix}.json"),
        serde_json::to_string_pretty(&out).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "QWEN38_AB run {steps} steps: chip median {median:.2} ms (min {:.2} max {:.2}), mean NLL of fed text {:.5}, generated {}",
        chip[0],
        chip[chip.len() - 1],
        nll_sum / nll_n.max(1) as f64,
        generated.len()
    );
    Ok(())
}

fn compare(a: &str, b: &str, out: &str) -> Result<(), String> {
    let (ja, jb) = (
        read_json(&format!("{a}.json"))?,
        read_json(&format!("{b}.json"))?,
    );
    let vocab = ja["vocab"].as_u64().ok_or("a: no vocab")? as usize;
    let (ra, rb) = (
        ja["rows"].as_array().ok_or("a: rows")?,
        jb["rows"].as_array().ok_or("b: rows")?,
    );
    let fed_a: Vec<u64> = ra
        .iter()
        .map(|r| r["fed"].as_u64().unwrap_or(u64::MAX))
        .collect();
    let n = ra.len().min(rb.len());
    // runs made with `--tail` hold logits from this step on; the earlier steps are checked for
    // the same input only (their NLL is in the rows)
    let lf = ja["logits_from"].as_u64().unwrap_or(0) as usize;
    if jb["logits_from"].as_u64().unwrap_or(0) as usize != lf {
        return Err("a and b hold logits from different steps (--tail)".into());
    }
    let mut fa = std::fs::File::open(format!("{a}.logits")).map_err(|e| e.to_string())?;
    let mut fb = std::fs::File::open(format!("{b}.logits")).map_err(|e| e.to_string())?;
    let (mut ba, mut bb) = (vec![0u8; vocab * 4], vec![0u8; vocab * 4]);
    let (mut agree, mut same_input) = (0usize, 0usize);
    let (mut cos_min, mut cos_sum, mut kl_sum, mut kl_max) = (f64::INFINITY, 0f64, 0f64, 0f64);
    let mut flips = Vec::new();
    let mut first_input_diff = None;
    for p in 0..n {
        if p >= lf {
            fa.read_exact(&mut ba).map_err(|e| e.to_string())?;
            fb.read_exact(&mut bb).map_err(|e| e.to_string())?;
        }
        if rb[p]["fed"].as_u64() != Some(fed_a[p]) {
            first_input_diff.get_or_insert(p);
            continue; // free-run texts diverged: positions after it see different inputs
        }
        if first_input_diff.is_some() {
            continue;
        }
        if p < lf {
            continue;
        }
        same_input += 1;
        let la: Vec<f32> = ba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let lb: Vec<f32> = bb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let (ta, tb) = (argmax(&la), argmax(&lb));
        if ta == tb {
            agree += 1;
        } else {
            flips.push(
                json!({"p": p, "a": ta, "b": tb, "a_margin": margin(&la), "b_margin": margin(&lb)}),
            );
        }
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in la.iter().zip(&lb) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        cos_min = cos_min.min(cos);
        cos_sum += cos;
        let (za, zb) = (logsumexp(&la), logsumexp(&lb));
        let kl: f64 = la
            .iter()
            .zip(&lb)
            .map(|(&x, &y)| {
                let lpa = x as f64 - za;
                lpa.exp() * (lpa - (y as f64 - zb))
            })
            .sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
    }
    let m = same_input.max(1) as f64;
    let margin_min = ra[..n]
        .iter()
        .map(|r| r["margin"].as_f64().unwrap_or(f64::NAN))
        .fold(f64::INFINITY, f64::min);
    let v = json!({
        "mode": "compare", "a": a, "b": b, "positions": n, "logits_from": lf, "same_input_positions": same_input,
        "first_input_difference": first_input_diff,
        "argmax_agree": agree, "flips": flips,
        "cosine_min": cos_min, "cosine_mean": cos_sum / m,
        "kl_mean": kl_sum / m, "kl_max": kl_max, "a_margin_min": margin_min,
        "a_mean_nll_fed": ja["mean_nll_fed"], "b_mean_nll_fed": jb["mean_nll_fed"],
        "a_generated": ja["generated_ids"], "b_generated": jb["generated_ids"],
        "a_chip_ms_median": ja["chip_ms_median"], "b_chip_ms_median": jb["chip_ms_median"],
    });
    std::fs::write(out, serde_json::to_string_pretty(&v).unwrap()).map_err(|e| e.to_string())?;
    eprintln!(
        "QWEN38_AB compare: {same_input} positions with the same input of {n}; argmax agree {agree}/{same_input}; cosine min {cos_min:.6} mean {:.6}; KL mean {:.3e} max {kl_max:.3e}; first input difference {first_input_diff:?}; NLL a {} b {}",
        cos_sum / m,
        kl_sum / m,
        ja["mean_nll_fed"],
        jb["mean_nll_fed"]
    );
    Ok(())
}

/// `decode <ckpt> <run.json> <out.txt>`: the fed prompt (teacher-forced ids) and the generated ids as text.
fn decode(dir: &Path, run_json: &str, out: &str) -> Result<(), String> {
    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let v = read_json(run_json)?;
    let fed: Vec<u32> = v["rows"]
        .as_array()
        .ok_or("no rows")?
        .iter()
        .take(v["teacher_forced"].as_u64().unwrap_or(0) as usize + 1)
        .map(|r| r["fed"].as_u64().unwrap_or(0) as u32)
        .collect();
    let gen = ids(&v, "generated_ids")?;
    let prompt = tk.decode(&fed).map_err(|e| e.to_string())?;
    let text = tk.decode(&gen).map_err(|e| e.to_string())?;
    std::fs::write(
        out,
        format!(
            "=== prompt ({} ids)\n{prompt}\n=== generated ({} ids)\n{text}\n",
            fed.len(),
            gen.len()
        ),
    )
    .map_err(|e| format!("{out}: {e}"))
}

/// `timeline <ckpt> <seq.json> <out.json>`: GPU-stamped time per launch kind, per token (the graph's stamps), median over
/// the seq's tokens after the first; the span is the whole token on the chip.
fn timeline(dir: &Path, seq_path: &str, out: &str) -> Result<(), String> {
    let seq = ids(&read_json(seq_path)?, "ids")?;
    let mut e = Qwen38Omega::open(dir, None, (seq.len() + 1).max(256).next_multiple_of(256))?;
    e.graph_mut().set_stamps(true);
    let mut per_kind: std::collections::BTreeMap<u8, Vec<f64>> = Default::default();
    let mut counts: std::collections::BTreeMap<u8, usize> = Default::default();
    let mut span = Vec::new();
    for (p, &t) in seq.iter().enumerate() {
        e.step(t)?;
        if p == 0 {
            continue;
        }
        let (st, kd) = e.graph_mut().stamps().map_err(|x| x.to_string())?;
        let mut sum: std::collections::BTreeMap<u8, f64> = Default::default();
        for (i, &k) in kd.iter().enumerate() {
            *sum.entry(k).or_default() += (st[i + 1] - st[i]) as f64 / 1e6;
            if p == 1 {
                *counts.entry(k).or_default() += 1;
            }
        }
        for (k, v) in sum {
            per_kind.entry(k).or_default().push(v);
        }
        span.push((st[kd.len()] - st[0]) as f64 / 1e6);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut rows = Vec::new();
    for (k, v) in per_kind.iter_mut() {
        let m = med(v);
        let name = aien_omega_q38::kind_name(*k);
        eprintln!(
            "{name:>8} {:4} launches {m:8.3} ms",
            counts.get(k).copied().unwrap_or(0)
        );
        rows.push(json!({"kind": name, "launches": counts.get(k), "ms_median": m}));
    }
    let s = med(&mut span);
    eprintln!("    span {s:8.3} ms over {} tokens", span.len());
    let v = json!({"mode": "timeline", "seq": seq_path, "tokens": span.len(), "span_ms_median": s, "kinds": rows,
        "variant": variant()});
    std::fs::write(out, serde_json::to_string_pretty(&v).unwrap())
        .map_err(|e| format!("{out}: {e}"))
}

/// `speed <ckpt> <ctx> <steps> <seed> <out.json>`: decode speed at long context, BENCHMARK ONLY.
/// The attention cache is filled with a bounded pattern up to `ctx - steps` (no prompt is read),
/// then `steps` greedy steps run at positions `ctx - steps .. ctx`. The tokens are meaningless;
/// only the times count. Per step: chip ms (GPU work) and wall ms (the whole step call).
fn speed(dir: &Path, ctx: usize, steps: usize, seed: u32, out: &str) -> Result<(), String> {
    if steps == 0 || steps > ctx {
        return Err(format!("speed: {steps} steps at context {ctx}"));
    }
    let config = Qwen38Config::from_checkpoint(dir)?;
    let max_pos = ctx.max(256).next_multiple_of(256);
    let avail = memory_guard(&config, max_pos)?;
    let variant = variant();
    eprintln!(
        "variant {variant}; context {ctx}, {steps} steps; MemAvailable {} MiB",
        avail >> 20
    );
    let t_open = Instant::now();
    let mut e = Qwen38Omega::open(dir, None, max_pos)?;
    let open_s = t_open.elapsed().as_secs_f64();
    let start = ctx - steps;
    let t_fill = Instant::now();
    e.bench_seek(start, seed)?;
    let fill_s = t_fill.elapsed().as_secs_f64();
    // any in-vocab token; greedy from there
    let mut fed = 0u32;
    let (mut chip, mut wall) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    for _ in 0..steps {
        let t = Instant::now();
        let (top, st) = e.step(fed)?;
        wall.push(t.elapsed().as_secs_f64() * 1e3);
        chip.push(st.chip_ns as f64 / 1e6);
        fed = top;
    }
    let med = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    let (cm, c0, c1) = med(&chip);
    let (wm, w0, w1) = med(&wall);
    // attention rows read by the median step (position ctx - steps / 2), K and V of all heads
    let n_attn = config.full_attention.iter().filter(|&&a| a).count();
    let row = if e.kv8() { 256 + 4 } else { 512 };
    let kv_read = (n_attn * 2 * 4 * (start + steps / 2 + 1) * row) as u64;
    let v = json!({
        "mode": "speed", "benchmark_only": true, "variant": variant, "kv8": e.kv8(),
        "context": ctx, "steps": steps, "first_position": start, "seed": seed, "max_pos": max_pos,
        "kv_cache_bytes": e.kv_bytes(), "kv_read_bytes_median_step": kv_read,
        "weight_bytes_resident": e.weight_bytes(),
        "chip_ms_median": cm, "chip_ms_min": c0, "chip_ms_max": c1,
        "wall_ms_median": wm, "wall_ms_min": w0, "wall_ms_max": w1,
        "tok_s_wall_median": 1e3 / wm, "tok_s_chip_median": 1e3 / cm,
        "open_s": open_s, "fill_s": fill_s, "chip_ms": chip, "wall_ms": wall,
    });
    std::fs::write(out, serde_json::to_string_pretty(&v).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    eprintln!(
        "QWEN38_AB speed context {ctx} kv8 {}: wall median {wm:.3} ms ({:.2} tok/s), chip median {cm:.3} ms, {steps} steps",
        e.kv8(),
        1e3 / wm
    );
    Ok(())
}

/// `w4err <ckpt> <out.json>`: host only, every projection under the OCP, FIT and BEST MXFP4 scale rules.
fn w4err(dir: &Path, out: &str) -> Result<(), String> {
    use aien_omega_q38::{W4_RULE_BEST, W4_RULE_FIT, W4_RULE_OCP};
    let t0 = Instant::now();
    let st = aien_inference_abi::qwen38_omega::w4_error_survey(
        dir,
        &[W4_RULE_OCP, W4_RULE_FIT, W4_RULE_BEST],
        16,
    )?;
    let rows: Vec<Value> = ["ocp", "fit", "best"]
        .iter()
        .zip(&st)
        .map(|(n, s)| {
            eprintln!("{n:>4}: rel_rms {:.6} clipped {} of {} weights", s.rel_rms(), s.saturated, s.weights);
            json!({"rule": n, "rel_rms": s.rel_rms(), "saturated": s.saturated, "weights": s.weights, "groups": s.groups,
                   "zero_groups": s.zero_groups, "sq_err": s.sq_err, "sq_ref": s.sq_ref})
        })
        .collect();
    let v = json!({"mode": "w4err", "rules": rows, "seconds": t0.elapsed().as_secs_f64()});
    std::fs::write(out, serde_json::to_string_pretty(&v).unwrap())
        .map_err(|e| format!("{out}: {e}"))
}
/// `[--gen N] [--tail N]` of `run`.
fn run_args(a: &[String]) -> Result<(usize, usize), String> {
    let (mut gen, mut tail) = (0, 0);
    let mut it = a.iter();
    while let Some(k) = it.next() {
        let v = it.next().and_then(|v| v.parse().ok());
        match (k.as_str(), v) {
            ("--gen", Some(n)) => gen = n,
            ("--tail", Some(n)) => tail = n,
            _ => return Err(format!("run: {k}: expected --gen N or --tail N")),
        }
    }
    Ok((gen, tail))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("text") if a.len() == 6 => a[4]
            .parse()
            .map_err(|_| "n".to_string())
            .and_then(|n| text(Path::new(&a[2]), &a[3], n, &a[5])),
        Some("run") if a.len() >= 5 => run_args(&a[5..])
            .and_then(|(g, t)| run(Path::new(&a[2]), &a[3], &a[4], g, t)),
        Some("compare") if a.len() == 5 => compare(&a[2], &a[3], &a[4]),
        Some("decode") if a.len() == 5 => decode(Path::new(&a[2]), &a[3], &a[4]),
        Some("timeline") if a.len() == 5 => timeline(Path::new(&a[2]), &a[3], &a[4]),
        Some("w4err") if a.len() == 4 => w4err(Path::new(&a[2]), &a[3]),
        Some("speed") if a.len() == 7 => match (a[3].parse(), a[4].parse(), a[5].parse()) {
            (Ok(ctx), Ok(n), Ok(seed)) => speed(Path::new(&a[2]), ctx, n, seed, &a[6]),
            _ => Err("speed <ckpt> <ctx> <steps> <seed> <out.json>".into()),
        },
        _ => Err("usage: qwen38_ab text <ckpt> <text-file> <n> <seq.json> | run <ckpt> <seq.json> <out-prefix> [--gen N] [--tail N] | compare <a> <b> <out.json> | decode <ckpt> <run.json> <out.txt> | timeline <ckpt> <seq.json> <out.json> | w4err <ckpt> <out.json> | speed <ckpt> <ctx> <steps> <seed> <out.json>".into()),
    };
    if let Err(e) = r {
        eprintln!("qwen38_ab: {e}");
        std::process::exit(1);
    }
}
