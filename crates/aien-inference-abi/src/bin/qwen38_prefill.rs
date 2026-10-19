//! Qwen3.8-27B whole-prompt prefill on the Omega engine (sc #642 step 9): parity and time to
//! first token.
//!
//!   qwen38_prefill golden <checkpoint-dir> <golden.json> <chunk> <out.json>
//!   qwen38_prefill score  <checkpoint-dir> <score.json>  <chunk> <out.json>
//!   qwen38_prefill ttft   <checkpoint-dir> <corpus.txt>  <len,len,...> <chunk> <out.json>
//!
//! `golden`: a `qwen38_check generate` record (host reference). The prompt is prefilled in chunks
//! of `chunk`, then the decode graph continues greedily; every generated token must equal the
//! reference's. The prefill's last logits are also held to the decode graph's logits for the same
//! prompt fed one token at a time (cosine), with a control: the decode graph's logits one position
//! earlier must fall below the bar.
//!
//! `score`: a `qwen38_check score` record (host reference, per-token NLL of a passage). The
//! passage is prefilled in chunks and every position's next-token NLL comes from the prefill's
//! residual rows; the mean NLL must be within 2% of the reference's and the next-token accuracy
//! within 2 points. Control: the NLL against the token two ahead must be at least 1 nat worse.
//!
//! `ttft`: for each length, a fresh sequence (state reset) prefilled with the first `len` tokens
//! of `corpus.txt` (tokenised in pieces of at most 2000 characters, the tokenizer's limit);
//! the time from the call to the first generated token, with the breakdown.
//!
//! The whole model is resident (about 25.7 GiB) plus the KV cache for the longest sequence. The
//! upload is refused unless MemAvailable covers that plus 12 GiB kept free (the resident 30B server
//! shares the memory; OOM lesson of 2026-10-18 16:43:58Z).

use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, Qwen38Config};
use aien_inference_abi::qwen38_omega::{layout, PrefillStats, Qwen38Omega};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};

const COSINE_MIN: f64 = 0.999;
const KEEP_FREE: u64 = 12 << 30;
const TOKENIZE_PIECE: usize = 2000;

fn mem_available() -> Result<u64, String> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

/// Device bytes for the whole model, its KV cache at `max_pos` and the prefill buffers for
/// `chunk` tokens; from the config, not measured.
fn need_bytes(c: &Qwen38Config, max_pos: usize, chunk: usize) -> u64 {
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
                (c.conv_dim() + c.lin_value_dim()) as u64 * h + o + mlp + 4 * 2 * 2 * 48 * h
            }
        })
        .sum();
    let weights = layers + layers / 4096 + 2 * c.vocab as u64 * h;
    let n_attn = c.full_attention.iter().filter(|&&a| a).count() as u64;
    let kv = n_attn * 2 * 4 * max_pos as u64 * 512 + max_pos as u64 * 1024;
    // xin, xq, xs, mix, act, out, plus the DeltaNet prefill scratch (omega_gpu_gdn_api.h).
    let m = chunk as u64;
    let bufs = m * (6144 * 4 + 17408 + 136 * 4 + 16384 * 4 + 17408 * 2 + 5120 * 4)
        + m * (40960 + 3 * 24576 + 384)
        + m.div_ceil(8) * 8 * 74496;
    weights + kv + bufs
}

fn open(dir: &Path, max_pos: usize, chunk: usize) -> Result<Qwen38Omega, String> {
    // The attention cache holds whole score passes of 256 positions (OMEGA_DATTN256_CHUNK).
    let max_pos = max_pos.next_multiple_of(256);
    let c = Qwen38Config::from_checkpoint(dir)?;
    let need = need_bytes(&c, max_pos, chunk);
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "needs {} MiB with {} MiB kept free, MemAvailable {} MiB; refusing",
            need >> 20,
            KEEP_FREE >> 20,
            avail >> 20
        ));
    }
    eprintln!(
        "upload: need {} MiB, MemAvailable {} MiB",
        need >> 20,
        avail >> 20
    );
    match PACK.get().and_then(Option::as_ref) {
        Some(p) => {
            Qwen38Omega::open_pack(dir, p, None, max_pos, layout::flip_from_env().as_deref())
        }
        None => Qwen38Omega::open(dir, None, max_pos),
    }
}

/// The layout pack from `--layout` (sc #676), opened once against the pinned id.
static PACK: std::sync::OnceLock<Option<layout::Pack>> = std::sync::OnceLock::new();

fn stats_json(s: &PrefillStats) -> Value {
    json!({
        "tokens": s.tokens, "chunks": s.chunks, "wall_ns": s.wall_ns,
        "gemm_chip_ns": s.gemm_chip_ns, "gemm_call_ns": s.gemm_call_ns,
        "gemm_launches": s.gemm_launches, "ab_call_ns": s.ab_call_ns,
        "gdn_chip_ns": s.gdn_chip_ns, "gdn_call_ns": s.gdn_call_ns,
        "attn_chip_ns": s.attn_chip_ns, "attn_call_ns": s.attn_call_ns,
        "host_ns": s.host_ns, "head_ns": s.head_ns,
    })
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
        .map(|x| x.as_u64().map(|t| t as u32).ok_or("bad id".to_string()))
        .collect()
}

fn golden(dir: &Path, golden: &str, chunk: usize) -> Result<(bool, Value), String> {
    let g = read_json(golden)?;
    let prompt = ids(&g, "prompt_ids")?;
    let want = ids(&g, "generated_ids")?;
    let mut e = open(dir, prompt.len() + want.len() + 1, chunk)?;

    // Decode graph, one token at a time: the logits after the last and second to last prompt token.
    e.reset()?;
    let mut dec_prev = Vec::new();
    for (i, &t) in prompt.iter().enumerate() {
        e.step(t)?;
        if i + 2 == prompt.len() {
            dec_prev = e.logits()?;
        }
    }
    let dec = e.logits()?;

    // Prefill, then greedy decode.
    e.reset()?;
    let (first, logits, st) = e.prefill(&prompt, chunk)?;
    let mut got = vec![first];
    while got.len() < want.len() {
        let (next, _) = e.step(*got.last().unwrap())?;
        got.push(next);
    }
    let p = parity_stats(&logits, &dec);
    let control = parity_stats(&logits, &dec_prev);
    let same = got == want;
    let pass = same
        && p.cosine >= COSINE_MIN
        && control.cosine < COSINE_MIN
        && argmax(&dec) as u32 == first;
    Ok((
        pass,
        json!({
            "mode": "golden", "prompt_tokens": prompt.len(), "chunk": chunk,
            "generated": got, "expected": want, "tokens_equal": same,
            "first_token_decode_graph": argmax(&dec), "first_token_prefill": first,
            "cosine_vs_decode_graph": p.cosine, "max_abs_vs_decode_graph": p.max_abs,
            "control_cosine_vs_previous_position": control.cosine,
            "cosine_min": COSINE_MIN, "stats": stats_json(&st), "pass": pass,
        }),
    ))
}

fn log_softmax_at(logits: &[f32], t: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    logits[t] as f64 - max - sum.ln()
}

fn score(dir: &Path, record: &str, chunk: usize) -> Result<(bool, Value), String> {
    let r = read_json(record)?;
    let toks = ids(&r, "ids")?;
    let ref_mean = r["mean_nll"].as_f64().ok_or("no mean_nll")?;
    let ref_acc = r["next_token_accuracy"]
        .as_f64()
        .ok_or("no next_token_accuracy")?;
    let ref_nll: Vec<f64> = r["nll"]
        .as_array()
        .ok_or("no nll")?
        .iter()
        .filter_map(Value::as_f64)
        .collect();
    let n = toks.len();
    let h = Qwen38Config::from_checkpoint(dir)?.hidden;
    let mut e = open(dir, n + 1, chunk)?;
    e.reset()?;
    let (rows, st) = e.prefill_rows(&toks, chunk, true)?;
    let (mut nll, mut ctrl, mut hits) = (Vec::with_capacity(n), 0f64, 0usize);
    for p in 0..n - 1 {
        let logits = e.head_logits(&rows[p * h..(p + 1) * h])?;
        let next = toks[p + 1] as usize;
        nll.push(-log_softmax_at(&logits, next));
        if argmax(&logits) == next {
            hits += 1;
        }
        if p + 2 < n {
            ctrl += -log_softmax_at(&logits, toks[p + 2] as usize);
        }
    }
    let mean = nll.iter().sum::<f64>() / nll.len() as f64;
    let acc = hits as f64 / nll.len() as f64;
    let ctrl_mean = ctrl / (n - 2) as f64;
    let (mut max_d, mut sum_d) = (0f64, 0f64);
    for (a, b) in nll.iter().zip(&ref_nll) {
        let d = (a - b).abs();
        max_d = max_d.max(d);
        sum_d += d;
    }
    let mean_ok = (mean - ref_mean).abs() <= 0.02 * ref_mean;
    let acc_ok = (acc - ref_acc).abs() <= 0.02;
    let ctrl_ok = ctrl_mean >= ref_mean + 1.0;
    let pass = mean_ok && acc_ok && ctrl_ok && ref_nll.len() == nll.len();
    Ok((
        pass,
        json!({
            "mode": "score", "tokens": n, "chunk": chunk,
            "mean_nll": mean, "reference_mean_nll": ref_mean,
            "perplexity": mean.exp(), "reference_perplexity": ref_mean.exp(),
            "next_token_accuracy": acc, "reference_next_token_accuracy": ref_acc,
            "max_abs_nll_diff": max_d, "mean_abs_nll_diff": sum_d / nll.len() as f64,
            "control_mean_nll_two_ahead": ctrl_mean,
            "stats": stats_json(&st), "pass": pass,
        }),
    ))
}

fn corpus_tokens(dir: &Path, corpus: &str, want: usize) -> Result<Vec<u32>, String> {
    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(corpus).map_err(|e| format!("{corpus}: {e}"))?;
    let mut out = encode_pieces(&tk, &text, want)?;
    if out.len() < want {
        return Err(format!("{corpus}: {} tokens, {want} wanted", out.len()));
    }
    out.truncate(want);
    Ok(out)
}

/// The tokenizer refuses sequences over 2048 tokens, so text goes in pieces of at most TOKENIZE_PIECE bytes,
/// stopping once `want` tokens are in hand.
fn encode_pieces(tk: &ChatTokenizer, text: &str, want: usize) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(want.min(text.len()));
    let mut rest = text;
    while out.len() < want && !rest.is_empty() {
        let mut cut = rest.len().min(TOKENIZE_PIECE);
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.extend(tk.encode(&rest[..cut]).map_err(|e| e.to_string())?);
        rest = &rest[cut..];
    }
    Ok(out)
}

fn ttft(dir: &Path, corpus: &str, lens: &[usize], chunk: usize) -> Result<(bool, Value), String> {
    let max = *lens.iter().max().ok_or("no lengths")?;
    let t = Instant::now();
    let toks = corpus_tokens(dir, corpus, max)?;
    eprintln!(
        "corpus: {} tokens in {:.1} s",
        toks.len(),
        t.elapsed().as_secs_f64()
    );
    let mut e = open(dir, max, chunk)?;
    let mut runs = Vec::new();
    for &n in lens {
        e.reset()?;
        let (first, _, st) = e.prefill(&toks[..n], chunk)?;
        let s = st.wall_ns as f64 / 1e9;
        eprintln!(
            "len {n}: TTFT {s:.3} s ({:.0} tok/s), first token {first}; gemm chip {:.1} s, gdn call {:.1} s, attn call {:.1} s, a|b {:.1} s, host {:.1} s",
            n as f64 / s,
            st.gemm_chip_ns as f64 / 1e9,
            st.gdn_call_ns as f64 / 1e9,
            st.attn_call_ns as f64 / 1e9,
            st.ab_call_ns as f64 / 1e9,
            st.host_ns as f64 / 1e9
        );
        runs.push(json!({
            "tokens": n, "ttft_s": s, "first_token": first, "stats": stats_json(&st),
        }));
    }
    Ok((
        true,
        json!({ "mode": "ttft", "chunk": chunk, "corpus": corpus, "runs": runs }),
    ))
}

/// Writes prompt-<len>.txt, the text of the first <len> corpus tokens, so other servers can be sent the same
/// bytes the ttft mode prefills. Re-encoding that text in one piece is timed: the tokenizer cost the
/// in-process ttft leaves out.
fn prompts(dir: &Path, corpus: &str, lens: &[usize], out: &Path) -> Result<(bool, Value), String> {
    let max = *lens.iter().max().ok_or("no lengths")?;
    let toks = corpus_tokens(dir, corpus, max)?;
    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut rows = Vec::new();
    let mut pass = true;
    for &n in lens {
        let text = tk.decode(&toks[..n]).map_err(|e| e.to_string())?;
        let t = Instant::now();
        let again = encode_pieces(&tk, &text, usize::MAX)?.len();
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;
        let path = out.join(format!("prompt-{n}.txt"));
        std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
        // The servers count the re-encoded text plus their chat template; 1% is the side-by-side's void line.
        let ok = again.abs_diff(n) * 100 <= n;
        pass &= ok;
        eprintln!(
            "len {n}: {} bytes, re-encoded {again} tokens in {encode_ms:.1} ms",
            text.len()
        );
        rows.push(json!({"tokens": n, "bytes": text.len(), "reencoded_tokens": again,
                         "encode_ms": encode_ms, "file": path.display().to_string(), "within_1pct": ok}));
    }
    Ok((
        pass,
        json!({"mode": "prompts", "corpus": corpus, "prompts": rows}),
    ))
}

fn main() {
    let mut a: Vec<String> = std::env::args().collect();
    let pack = layout::take_layout_arg(&mut a)
        .and_then(|p| p.map(|p| layout::open_for_tool(&p)).transpose());
    match pack {
        Ok(p) => {
            let _ = PACK.set(p);
        }
        Err(e) => {
            eprintln!("qwen38_prefill: {e}");
            std::process::exit(2);
        }
    }
    let usage = "usage: qwen38_prefill golden|score <dir> <record.json> <chunk> <out.json> | ttft <dir> <corpus.txt> <len,...> <chunk> <out.json> | prompts <dir> <corpus.txt> <len,...> <out-dir>; [--layout <pack>] loads from a pinned layout pack";
    let res = (|| -> Result<(bool, Value, String), String> {
        match a.get(1).map(String::as_str) {
            Some(mode @ ("golden" | "score")) if a.len() == 6 => {
                let chunk: usize = a[4].parse().map_err(|_| usage.to_string())?;
                let dir = Path::new(&a[2]);
                let (pass, v) = if mode == "golden" {
                    golden(dir, &a[3], chunk)?
                } else {
                    score(dir, &a[3], chunk)?
                };
                Ok((pass, v, a[5].clone()))
            }
            Some("ttft") if a.len() == 7 => {
                let lens: Vec<usize> = a[4]
                    .split(',')
                    .map(|s| s.parse().map_err(|_| usage.to_string()))
                    .collect::<Result<_, _>>()?;
                let chunk: usize = a[5].parse().map_err(|_| usage.to_string())?;
                let (pass, v) = ttft(Path::new(&a[2]), &a[3], &lens, chunk)?;
                Ok((pass, v, a[6].clone()))
            }
            Some("prompts") if a.len() == 6 => {
                let lens: Vec<usize> = a[4]
                    .split(',')
                    .map(|s| s.parse().map_err(|_| usage.to_string()))
                    .collect::<Result<_, _>>()?;
                let (pass, v) = prompts(Path::new(&a[2]), &a[3], &lens, Path::new(&a[5]))?;
                Ok((
                    pass,
                    v,
                    Path::new(&a[5]).join("prompts.json").display().to_string(),
                ))
            }
            _ => Err(usage.to_string()),
        }
    })();
    match res {
        Ok((pass, v, out)) => {
            let text = serde_json::to_string_pretty(&v).expect("json");
            if let Err(e) = std::fs::write(&out, &text) {
                eprintln!("{out}: {e}");
                std::process::exit(2);
            }
            println!("{text}");
            println!("{}", if pass { "PASS" } else { "FAIL" });
            std::process::exit(if pass { 0 } else { 1 });
        }
        Err(e) => {
            eprintln!("qwen38_prefill: {e}");
            std::process::exit(2);
        }
    }
}
