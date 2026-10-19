//! A/B drift and speed tool for Qwen3.8-27B on the Omega decode graph (FP8 lm_head on/off,
//! 4-bit weights): every number comes from the whole model, all 64 layers resident.
//!
//!   qwen38_ab text <checkpoint-dir> <text-file> <n> <seq.json>
//!       the first `n` tokens of a plain text (no chat template), as `{"ids": [...]}`.
//!   qwen38_ab run <checkpoint-dir> <seq.json> <out-prefix> [--gen N]
//!       teacher-forced: feeds `ids` one decode step at a time; with `--gen N` the ids are a
//!       prompt and N more tokens follow greedily (free running). Writes `<out-prefix>.logits`
//!       (f32, vocab per step, step order) and `<out-prefix>.json` (per step: fed token, chip
//!       argmax, top-two margin, chip ms, and the negative log-likelihood of the next fed token).
//!       The engine variant comes from the environment (AIEN_Q38_FP8_HEAD=1, AIEN_Q38_W4=1).
//!   qwen38_ab compare <a-prefix> <b-prefix> <out.json>
//!       position by position, A taken as the reference: argmax agreement, A's top-two margin
//!       where they differ, logits cosine, KL(A || B), and both texts' mean NLL. Two runs fed the
//!       same tokens see the same inputs, so with teacher forcing only the variant differs.
//!
//! A run refuses to start unless MemAvailable covers the model plus 12 GiB for everything else
//! on the box (the live server shares the pool).

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

fn run(dir: &Path, seq_path: &str, prefix: &str, gen: usize) -> Result<(), String> {
    let seq = ids(&read_json(seq_path)?, "ids")?;
    if seq.len() < 2 && gen == 0 {
        return Err("seq: fewer than 2 ids".into());
    }
    let config = Qwen38Config::from_checkpoint(dir)?;
    // the MXFP4 copies sit beside the FP8 tensors (the prefill keeps FP8): + 17 / 32 of the projection bytes, + the head's
    let fp8 = model_bytes(&config);
    let head = 2 * config.vocab as u64 * config.hidden as u64;
    let mut need = fp8 + HOST_SLACK;
    if std::env::var("AIEN_Q38_W4").as_deref() == Ok("1") {
        need += (fp8 - head) * 17 / 32;
    }
    if std::env::var("AIEN_Q38_W4_HEAD").as_deref() == Ok("1") {
        need += head / 2 * 17 / 32;
    }
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "memory guard: MemAvailable {} MiB < model {} MiB + 12 GiB kept free",
            avail >> 20,
            need >> 20
        ));
    }
    let (vocab, steps) = (
        config.vocab,
        if gen > 0 {
            seq.len() + gen - 1
        } else {
            seq.len() - 1
        },
    );
    let variant = json!({
        "AIEN_Q38_FP8_HEAD": std::env::var("AIEN_Q38_FP8_HEAD").ok(),
        "AIEN_Q38_W4": std::env::var("AIEN_Q38_W4").ok(),
        "AIEN_Q38_W4_HEAD": std::env::var("AIEN_Q38_W4_HEAD").ok(),
    });
    eprintln!(
        "variant {variant}; {steps} steps; MemAvailable {} MiB",
        avail >> 20
    );
    let t_open = Instant::now();
    let mut e = Qwen38Omega::open(dir, None, (steps + 1).max(256).next_multiple_of(256))?;
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
        for v in &l {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
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
        "mode": "run", "seq": seq_path, "variant": variant, "steps": steps, "vocab": vocab,
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
    let mut fa = std::fs::File::open(format!("{a}.logits")).map_err(|e| e.to_string())?;
    let mut fb = std::fs::File::open(format!("{b}.logits")).map_err(|e| e.to_string())?;
    let (mut ba, mut bb) = (vec![0u8; vocab * 4], vec![0u8; vocab * 4]);
    let (mut agree, mut same_input) = (0usize, 0usize);
    let (mut cos_min, mut cos_sum, mut kl_sum, mut kl_max) = (f64::INFINITY, 0f64, 0f64, 0f64);
    let mut flips = Vec::new();
    let mut first_input_diff = None;
    for p in 0..n {
        fa.read_exact(&mut ba).map_err(|e| e.to_string())?;
        fb.read_exact(&mut bb).map_err(|e| e.to_string())?;
        if rb[p]["fed"].as_u64() != Some(fed_a[p]) {
            first_input_diff.get_or_insert(p);
            continue; // free-run texts diverged: positions after it see different inputs
        }
        if first_input_diff.is_some() {
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
        "mode": "compare", "a": a, "b": b, "positions": n, "same_input_positions": same_input,
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
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("text") if a.len() == 6 => a[4]
            .parse()
            .map_err(|_| "n".to_string())
            .and_then(|n| text(Path::new(&a[2]), &a[3], n, &a[5])),
        Some("run") if a.len() == 5 => run(Path::new(&a[2]), &a[3], &a[4], 0),
        Some("run") if a.len() == 7 && a[5] == "--gen" => a[6]
            .parse()
            .map_err(|_| "--gen N".to_string())
            .and_then(|g| run(Path::new(&a[2]), &a[3], &a[4], g)),
        Some("compare") if a.len() == 5 => compare(&a[2], &a[3], &a[4]),
        Some("decode") if a.len() == 5 => decode(Path::new(&a[2]), &a[3], &a[4]),
        _ => Err("usage: qwen38_ab text <ckpt> <text-file> <n> <seq.json> | run <ckpt> <seq.json> <out-prefix> [--gen N] | compare <a> <b> <out.json> | decode <ckpt> <run.json> <out.txt>".into()),
    };
    if let Err(e) = r {
        eprintln!("qwen38_ab: {e}");
        std::process::exit(1);
    }
}
