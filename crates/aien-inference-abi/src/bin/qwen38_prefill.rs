//! Qwen3.8-27B whole-prompt prefill on the Omega engine (sc #642 step 9): parity and time to
//! first token.
//!
//!   qwen38_prefill golden <checkpoint-dir> <golden.json> <chunk> <out.json>
//!   qwen38_prefill score  <checkpoint-dir> <score.json>  <chunk> <out.json>
//!   qwen38_prefill ttft   <checkpoint-dir> <corpus.txt>  <len,len,...> <chunk> <out.json>
//!   qwen38_prefill storage <checkpoint-dir> <layers> <tokens> <chunk> <out.json>
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
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, Qwen38Config};
use aien_inference_abi::qwen38_omega::{storage, PrefillStats, Qwen38Omega};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::storage_map::{ProcMemory, Receipt, Space};
use aien_inference_abi::tokenizer::ChatTokenizer;
use aien_omega_pf::{OmegaPfDevice, PfBuffer, PfMem, PrefillDevice};
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

/// Bytes the run needs at its peak: the storage map's plan for the whole model, its KV cache at
/// `max_pos` and the prefill for chunks of `chunk` tokens (sc #678: the same map the receipt
/// measures; the old formula left out the host embedding and omega's own scratch).
fn need_bytes(c: &Qwen38Config, max_pos: usize, chunk: usize) -> u64 {
    storage::plan(c, max_pos, Some((chunk, OmegaPfDevice.qgemm_bm()))).peak()
}

/// The process before its first allocation (set first thing in `main`).
static BASELINE: OnceLock<ProcMemory> = OnceLock::new();
/// Where the storage map receipts go: `<out>.storage-map-<stage>.{txt,json}`.
static MAP_OUT: OnceLock<String> = OnceLock::new();
/// The negative control's blocks (AIEN_STORAGE_MAP_LEAK_BYTES): made after the load, never
/// registered in the map, kept until the process ends.
static LEAK: Mutex<Vec<(PfMem, Vec<u8>)>> = Mutex::new(Vec::new());

/// Unaccounted bytes a run may hold, per space, before its storage check fails; set from the
/// measured clean runs (evidence/f6-storage-map-*/).
const UNACCOUNTED_DRIVER_MAX: i64 = 64 << 20;
const UNACCOUNTED_HEAP_MAX: i64 = 128 << 20;

/// AIEN_STORAGE_MAP_LEAK_BYTES=n: one omega host buffer and one heap vector of n bytes each,
/// every page written, that the map does not name. Both must show in the unaccounted numbers.
fn leak_control() -> Result<(), String> {
    let Ok(v) = std::env::var("AIEN_STORAGE_MAP_LEAK_BYTES") else {
        return Ok(());
    };
    let n: usize = v
        .parse()
        .map_err(|_| format!("AIEN_STORAGE_MAP_LEAK_BYTES={v}: not a byte count"))?;
    let mut d = PfMem::alloc_host(n).map_err(|e| format!("leak control: {e}"))?;
    d.host_mut().ok_or("leak control: no host view")?.fill(0x5a);
    let h = vec![0x5au8; n];
    eprintln!("NEGATIVE CONTROL: {n} unregistered bytes in each space");
    LEAK.lock().expect("leak lock").push((d, h));
    Ok(())
}

/// Holds the model's storage map against the process now; writes the receipt files and returns
/// the totals and whether the unaccounted numbers stay within the bars and the planned peak
/// covers the held one.
fn storage_receipt(e: &Qwen38Omega, stage: &str) -> Result<(bool, Value), String> {
    let base = *BASELINE.get().ok_or("no baseline")?;
    let r = Receipt::new(
        format!("qwen38_prefill {stage}"),
        e.storage_map().clone(),
        base,
        ProcMemory::read()?,
    );
    let ud = r.unaccounted(Space::Driver);
    let uh = r.unaccounted(Space::Heap);
    let peak_ok = r.map.peak() >= r.peak_held();
    let ok = ud.abs() <= UNACCOUNTED_DRIVER_MAX && uh.abs() <= UNACCOUNTED_HEAP_MAX && peak_ok;
    let mut text = r.to_text();
    text.push_str(&format!(
        "storage check ({stage}): |unaccounted driver| {} <= {UNACCOUNTED_DRIVER_MAX}, |unaccounted heap| {} <= {UNACCOUNTED_HEAP_MAX}, planned peak >= held peak {peak_ok}: {}\n",
        ud.abs(),
        uh.abs(),
        if ok { "PASS" } else { "FAIL" }
    ));
    let mut v = r.to_json();
    v["check"] = json!({"stage": stage, "pass": ok, "unaccounted_driver_max": UNACCOUNTED_DRIVER_MAX,
                        "unaccounted_heap_max": UNACCOUNTED_HEAP_MAX, "peak_covered": peak_ok});
    if let Some(out) = MAP_OUT.get() {
        for (ext, body) in [
            ("txt", text.clone()),
            ("json", serde_json::to_string_pretty(&v).expect("json")),
        ] {
            let p = format!("{out}.storage-map-{stage}.{ext}");
            std::fs::write(&p, body).map_err(|e| format!("{p}: {e}"))?;
        }
    }
    eprint!(
        "{}",
        text.lines()
            .rev()
            .take(9)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|l| format!("{l}\n"))
            .collect::<String>()
    );
    Ok((
        ok,
        json!({"stage": stage, "pass": ok, "totals": v["totals"], "mismatches": v["mismatches"]}),
    ))
}

fn open(dir: &Path, max_pos: usize, chunk: usize) -> Result<Qwen38Omega, String> {
    open_layers(dir, None, max_pos, chunk)
}

/// [`open`] for the first `layers` layers only (`None`: all).
fn open_layers(
    dir: &Path,
    layers: Option<usize>,
    max_pos: usize,
    chunk: usize,
) -> Result<Qwen38Omega, String> {
    // The attention cache holds whole score passes of 256 positions (OMEGA_DATTN256_CHUNK).
    let max_pos = max_pos.next_multiple_of(256);
    let full = Qwen38Config::from_checkpoint(dir)?;
    let c = match layers {
        None => full,
        Some(n) => full
            .prefix(n)
            .ok_or_else(|| format!("layers {n}: outside the model"))?,
    };
    let need = need_bytes(&c, max_pos, chunk.min(max_pos));
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
        "upload: need {} MiB (storage map peak), MemAvailable {} MiB",
        need >> 20,
        avail >> 20
    );
    let e = Qwen38Omega::open(dir, layers, max_pos)?;
    leak_control()?;
    storage_receipt(&e, "load")?;
    Ok(e)
}

/// The storage map's own check (sc #678): the first `layers` layers, a prefill of `tokens`
/// tokens in chunks of `chunk`, the host head and one decode step, each first use probed; PASS
/// when every probed call matches its plan, nothing planned is left unprobed and both
/// receipts pass. Small enough to run beside the live server.
fn storage_check(
    dir: &Path,
    layers: usize,
    tokens: usize,
    chunk: usize,
) -> Result<(bool, Value), String> {
    let mut e = open_layers(dir, Some(layers), tokens + 1, chunk)?;
    let ids: Vec<u32> = (0..tokens as u32).map(|i| 1000 + 7 * i).collect();
    e.reset()?;
    let (first, _, st) = e.prefill(&ids, chunk)?;
    e.step(first)?;
    let (ok, sm) = storage_receipt(&e, "end")?;
    let mismatches = e.storage_map().failing(storage::ALLOWANCE_SLACK).len();
    let unprobed = e.storage_map().unprobed().len();
    let pass = ok && mismatches == 0 && unprobed == 0;
    Ok((
        pass,
        json!({"mode": "storage", "layers": layers, "tokens": tokens, "chunk": chunk,
               "mismatched_calls": mismatches, "unprobed_calls": unprobed,
               "stats": stats_json(&st), "storage_map": sm, "pass": pass}),
    ))
}

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
    let (sm_ok, sm) = storage_receipt(&e, "end")?;
    let pass = pass && sm_ok;
    Ok((
        pass,
        json!({
            "mode": "golden", "prompt_tokens": prompt.len(), "chunk": chunk,
            "generated": got, "expected": want, "tokens_equal": same,
            "first_token_decode_graph": argmax(&dec), "first_token_prefill": first,
            "cosine_vs_decode_graph": p.cosine, "max_abs_vs_decode_graph": p.max_abs,
            "control_cosine_vs_previous_position": control.cosine,
            "cosine_min": COSINE_MIN, "stats": stats_json(&st), "pass": pass, "storage_map": sm,
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
    let (sm_ok, sm) = storage_receipt(&e, "end")?;
    let pass = pass && sm_ok;
    Ok((
        pass,
        json!({
            "mode": "score", "tokens": n, "chunk": chunk,
            "mean_nll": mean, "reference_mean_nll": ref_mean,
            "perplexity": mean.exp(), "reference_perplexity": ref_mean.exp(),
            "next_token_accuracy": acc, "reference_next_token_accuracy": ref_acc,
            "max_abs_nll_diff": max_d, "mean_abs_nll_diff": sum_d / nll.len() as f64,
            "control_mean_nll_two_ahead": ctrl_mean,
            "stats": stats_json(&st), "pass": pass, "storage_map": sm,
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
    let (sm_ok, sm) = storage_receipt(&e, "end")?;
    Ok((
        sm_ok,
        json!({ "mode": "ttft", "chunk": chunk, "corpus": corpus, "runs": runs, "storage_map": sm }),
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
    // Before anything allocates: the receipts count from here.
    match ProcMemory::read() {
        Ok(m) => {
            let _ = BASELINE.set(m);
        }
        Err(e) => {
            eprintln!("qwen38_prefill: {e}");
            std::process::exit(2);
        }
    }
    let a: Vec<String> = std::env::args().collect();
    if let Some(out) = a.last() {
        let _ = MAP_OUT.set(out.trim_end_matches(".json").to_string());
    }
    let usage = "usage: qwen38_prefill golden|score <dir> <record.json> <chunk> <out.json> | ttft <dir> <corpus.txt> <len,...> <chunk> <out.json> | prompts <dir> <corpus.txt> <len,...> <out-dir> | storage <dir> <layers> <tokens> <chunk> <out.json>";
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
            Some("storage") if a.len() == 7 => {
                let num = |i: usize| a[i].parse::<usize>().map_err(|_| usage.to_string());
                let (pass, v) = storage_check(Path::new(&a[2]), num(3)?, num(4)?, num(5)?)?;
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
