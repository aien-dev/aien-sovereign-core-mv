//! Prefill gate for the 35B (L369): a prompt read in chunks through the chip must leave the
//! conversation in the same state, bit for bit, as the same prompt fed token by token through the
//! decode graph (today's served path), and the replies after it must be the same.
//!
//! ```text
//!   qwen36_prefill_gate <checkpoint-dir> <corpus.txt> <len,len,...> <chunk> <decode> <out.json> [layers]
//! ```
//!
//! One graph with two head slots over one upload ([`Qwen38Omega::open_window_heads`]). For each
//! length, both slots are emptied, then the first `len` tokens of `corpus.txt`:
//! - slot 0 token by token ([`Qwen38Omega::step`], the decode graph): time A;
//! - slot 1 in chunks of `chunk` ([`Qwen38Omega::prefill_moe_head`]): time B.
//!
//! PASS needs, for every length: the last prompt token's greedy token and logits bit-identical;
//! every DeltaNet state and conv tail bit-identical ([`Qwen38Snapshot::first_diff`]); then
//! `decode` greedy steps with both slots in one stroke, each step's tokens and logits
//! bit-identical (this reads the KV caches the prompt wrote). Exit 0 PASS, 1 FAIL, 2 error.
//! Prompt tokens per second, A and B, are in the record (a receipt only inside a chip window).
//!
//! Negative controls, each must FAIL: `AIEN_OMEGA_Q36_PF_MUTANT=1` (the last two tokens' GDN
//! steps swapped), `=2` (every expert segment one row short) and `=3` (each token's DeltaNet output copied
//! after the whole chunk instead of right after its step).
//!
//! `layers`: a model prefix (the first `layers` layers) for a small load; the full 35B loads only
//! under `q36-extra` (one extra 35B beside the live server).
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38_omega::{Qwen38Omega, Qwen38Snapshot};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};

const TOKENIZE_PIECE: usize = 2000;

fn corpus_tokens(dir: &Path, corpus: &str, want: usize) -> Result<Vec<u32>, String> {
    let tk = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(corpus).map_err(|e| format!("{corpus}: {e}"))?;
    let mut out = Vec::with_capacity(want);
    let mut rest = text.as_str();
    while out.len() < want && !rest.is_empty() {
        let mut cut = rest.len().min(TOKENIZE_PIECE);
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.extend(tk.encode(&rest[..cut]).map_err(|e| e.to_string())?);
        rest = &rest[cut..];
    }
    if out.len() < want {
        return Err(format!("{corpus}: {} tokens, {want} wanted", out.len()));
    }
    out.truncate(want);
    Ok(out)
}

fn first_bit_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

/// Head 1's state as a snapshot: the snapshot reads slot 0, so the slots trade places around it.
fn snap_head(m: &mut Qwen38Omega, head: usize) -> Result<Qwen38Snapshot, String> {
    let mut s = Qwen38Snapshot::default();
    if head == 0 {
        m.snapshot(&mut s)?;
    } else {
        m.swap_heads(0, head)?;
        let r = m.snapshot(&mut s);
        m.swap_heads(0, head)?;
        r?;
    }
    Ok(s)
}

fn one(m: &mut Qwen38Omega, prompt: &[u32], chunk: usize, decode: usize) -> Result<Value, String> {
    let n = prompt.len();
    m.reset_head(0)?;
    m.reset_head(1)?;
    let t = Instant::now();
    let mut next_a = 0u32;
    for &tok in prompt {
        next_a = m.step(tok)?.0;
    }
    let ms_a = t.elapsed().as_secs_f64() * 1e3;
    let logits_a = m.logits_head(0)?;
    let s0 = m.q36_prefill_streamed();
    let t = Instant::now();
    let (next_b, chunks) = m.prefill_moe_head(1, prompt, chunk, true)?;
    let ms_b = t.elapsed().as_secs_f64() * 1e3;
    let streamed = m.q36_prefill_streamed() - s0;
    let next_b = next_b.ok_or("prefill gave no token")?;
    let logits_b = m.logits_head(1)?;
    let logits_diff = first_bit_diff(&logits_a, &logits_b);
    let (sa, sb) = (snap_head(m, 0)?, snap_head(m, 1)?);
    let state_diff = sa.first_diff(&sb);
    let (mut a, mut b) = (next_a, next_b);
    let mut decode_diff: Option<usize> = None;
    let (mut toks_a, mut toks_b) = (Vec::new(), Vec::new());
    for d in 0..decode {
        let (out, _) = m.stroke(&[a, b])?;
        let (la, lb) = (m.logits_head(0)?, m.logits_head(1)?);
        toks_a.push(out[0]);
        toks_b.push(out[1]);
        if decode_diff.is_none() && (out[0] != out[1] || first_bit_diff(&la, &lb).is_some()) {
            decode_diff = Some(d);
        }
        (a, b) = (out[0], out[1]);
    }
    let pass =
        next_a == next_b && logits_diff.is_none() && state_diff.is_none() && decode_diff.is_none();
    let rec = json!({
        "len": n, "chunk": chunk, "chunks": chunks, "pass": pass,
        "token_by_token_ms": ms_a, "chunked_ms": ms_b,
        "token_by_token_tok_s": n as f64 / (ms_a / 1e3), "chunked_tok_s": n as f64 / (ms_b / 1e3),
        "next": [next_a, next_b], "logits_first_diff": logits_diff, "state_first_diff": state_diff,
        "decode_first_diff": decode_diff, "decode_tokens": [toks_a, toks_b],
        "chunked_fp8_streamed_bytes": streamed, "chunked_fp8_gb_s": streamed as f64 / (ms_b * 1e6),
    });
    eprintln!(
        "len {n}: {} | token by token {ms_a:.0} ms ({:.1} tok/s), chunked {ms_b:.0} ms ({:.1} tok/s, {chunks} chunks, fp8 {:.1} GB/s) | logits {:?} state {:?} decode {:?}",
        if pass { "PASS" } else { "FAIL" },
        n as f64 / (ms_a / 1e3),
        n as f64 / (ms_b / 1e3),
        streamed as f64 / (ms_b * 1e6),
        logits_diff,
        state_diff,
        decode_diff
    );
    Ok(rec)
}

/// `L369_PROFILE=1`: one chunk on head 1 with the launch timestamps on, chip time per kind of launch
/// (the first launch of each list also carries the host gap before it). Not a receipt outside a window.
fn profile(m: &mut Qwen38Omega, prompt: &[u32], chunk: usize) -> Result<(), String> {
    use aien_omega_q38::q36;
    m.reset_head(1)?;
    m.q36_graph_mut().ok_or("no q36 graph")?.set_stamps(true);
    let t = Instant::now();
    let r = m.prefill_moe_head(1, prompt, chunk, true);
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let g = m.q36_graph_mut().ok_or("no q36 graph")?;
    let tl = g.timeline().map_err(|e| format!("timeline: {e}"));
    g.set_stamps(false);
    r?;
    let tl = tl?;
    let mut per: Vec<(String, u64, u32)> = Vec::new();
    for &(kind, d) in &tl {
        let name = q36::Q36Graph::kind_name(kind);
        match per.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += d;
                e.2 += 1
            }
            None => per.push((name, d, 1)),
        }
    }
    per.sort_by_key(|e| std::cmp::Reverse(e.1));
    let span: u64 = tl.iter().map(|e| e.1).sum();
    eprintln!(
        "profile: {} tokens, wall {wall:.1} ms, {} launches, stamp span {:.1} ms",
        prompt.len(),
        tl.len(),
        span as f64 / 1e6
    );
    for (name, ns, c) in per {
        eprintln!(
            "  {name:<12} {:>8.2} ms {:>5.1}%  {c} launches",
            ns as f64 / 1e6,
            100.0 * ns as f64 / span as f64
        );
    }
    Ok(())
}

fn run(args: &[String]) -> Result<bool, String> {
    if args.len() < 7 {
        return Err("usage: qwen36_prefill_gate <checkpoint-dir> <corpus.txt> <len,len,...> <chunk> <decode> <out.json> [layers]".into());
    }
    let dir = Path::new(&args[1]);
    let lens: Vec<usize> = args[3]
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("len {s}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    let chunk: usize = args[4].parse().map_err(|e| format!("chunk: {e}"))?;
    let decode: usize = args[5].parse().map_err(|e| format!("decode: {e}"))?;
    let out = &args[6];
    let max_len = *lens.iter().max().ok_or("no lengths")?;
    let prompt = corpus_tokens(dir, &args[2], max_len)?;
    let full = aien_inference_abi::qwen38::Qwen38Config::from_checkpoint(dir)?;
    let count = match args.get(7) {
        Some(l) => l.parse::<usize>().map_err(|e| format!("layers: {e}"))?,
        None => full.layers,
    };
    let t = Instant::now();
    // the attention cache holds whole score passes of 256 positions
    let max_pos = (max_len + decode + 1).next_multiple_of(256);
    let mut m = Qwen38Omega::open_window_heads(dir, 0, count, max_pos, 2)?;
    eprintln!(
        "loaded {count} layers in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let mut runs = Vec::new();
    let mut all = true;
    for &len in &lens {
        let r = one(&mut m, &prompt[..len], chunk, decode)?;
        all &= r["pass"].as_bool() == Some(true);
        runs.push(r);
    }
    if std::env::var("L369_PROFILE").is_ok_and(|v| v == "1") {
        profile(&mut m, &prompt[..chunk.min(max_len)], chunk)?;
    }
    let mutant = std::env::var("AIEN_OMEGA_Q36_PF_MUTANT").unwrap_or_default();
    let rec = json!({ "gate": "qwen36_prefill_gate", "pass": all, "layers": count, "mutant": mutant, "runs": runs });
    std::fs::write(
        out,
        serde_json::to_string_pretty(&rec).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("{out}: {e}"))?;
    eprintln!("{}", if all { "GATE PASS" } else { "GATE FAIL" });
    Ok(all)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen36_prefill_gate: {e}");
            std::process::exit(2);
        }
    }
}
