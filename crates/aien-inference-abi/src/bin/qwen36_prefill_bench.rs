//! Prefill bench for the 35B (P-DRA-4 Lane P, gates G1-G3): one prefill arm per process, so the
//! arm is the environment (path A: the legacy fp8 GEMV prefill, chunk 128; path B: the tensor-core
//! prefill, `AIEN_OMEGA_Q36_PF_QMMA=1` on a library that has it). The logits it writes are judged
//! offline by `om/prefill-logits` (A against B, same prompt).
//!
//! ```text
//!   qwen36_prefill_bench [--plan] <checkpoint-dir> <arm> <out-dir> <reps> <prompt>...
//!   prompt: NAME=corpus:FILE.txt:LEN   the first LEN tokens of FILE (tokenized as the L369 gate does)
//!           NAME=text:FILE.txt         every token of FILE
//!           NAME=ids:FILE.json         token ids: a JSON array, or an object with "prompt_ids" / "ids"
//! ```
//!
//! Chunk: `AIEN_Q36_PREFILL_CHUNK` read by [`q36_prefill_chunk`] (the server's knob, default the
//! graph's most rows); 0 = token by token through the decode graph (the GEMV reference). The bench
//! hands the prefill one chunk per call, so every chunk gets its own clock, byte count and timeline.
//!
//! Per prompt, `reps` timed passes (stamps off), then one pass with the launch timestamps on.
//! Written to `<out-dir>/<arm>/`: `NAME.logits.f32` (the last prompt token's logits, 248320 LE f32,
//! no header), `NAME.json`, and `bench.json` (every prompt). Clocks, never mixed:
//! - `ttft_inproc_host_ms`: host clock around the whole prompt (embedding, submit, wait, the last
//!   token's head and argmax) in this process. No HTTP, no tokenizer, no queue.
//! - `chip_only_ms`: the GPU timestamps, last minus first stamp of each chunk, summed over chunks.
//!   The first launch of each command list carries the host gap before it, so this is the device
//!   span, not busy time; `classes` sums the same per-launch spans by launch kind.
//!   `GraphStats::chip_ns` is not used: it is host clock from submit to the marker.
//! - `streamed_bytes`: the graph's fp8 GEMM weight byte count, per chunk.
//!
//! `AIEN_BENCH_LAYERS=N`: a model prefix (small load). `AIEN_BENCH_STAMPS=0`: no timestamp pass.
//! `--plan`: tokenize, write `NAME.ids.json` with the chunk plan, load nothing (host check).
//! Exit 0 done, 1 a rep's logits differ bit for bit from rep 0 (non-determinism), 2 error.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aien_inference_abi::qwen38_omega::{q36_prefill_chunk, Qwen38Omega};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const TOKENIZE_PIECE: usize = 2000;
const TOP_K: usize = 16;
/// The knobs a record carries: everything that picks a prefill or decode path in this process.
const KNOBS: &[&str] = &[
    "AIEN_Q36_PREFILL_CHUNK",
    "AIEN_OMEGA_Q36_PF_CHUNK",
    "AIEN_OMEGA_Q36_PF_QMMA",
    "AIEN_OMEGA_Q36_PF_MUTANT",
    "AIEN_OMEGA_PFQ_MUTANT",
    "AIEN_OMEGA_PFQ_WAIT",
    "AIEN_OMEGA_Q36_PF_GDN",
    "AIEN_OMEGA_Q36_PF_ATTN",
    "AIEN_OMEGA_FP8_GROUPED_WAVE",
    "AIEN_Q36_EXACT_HEAD",
    "AIEN_STROKE_IN_WINDOW",
    "AIEN_BENCH_LAYERS",
    "AIEN_BENCH_STAMPS",
];

struct Prompt {
    name: String,
    source: String,
    ids: Vec<u32>,
}

enum Spec {
    Corpus(PathBuf, usize),
    Text(PathBuf),
    Ids(PathBuf),
}

fn parse_spec(s: &str) -> Result<(String, Spec), String> {
    let (name, rest) = s
        .split_once('=')
        .ok_or_else(|| format!("prompt {s}: NAME=corpus:FILE:LEN | NAME=text:FILE | NAME=ids:FILE"))?;
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(format!("prompt {s}: NAME must be [A-Za-z0-9_-]+"));
    }
    let (kind, arg) = rest
        .split_once(':')
        .ok_or_else(|| format!("prompt {s}: no kind"))?;
    let spec = match kind {
        "corpus" => {
            let (file, len) = arg
                .rsplit_once(':')
                .ok_or_else(|| format!("prompt {s}: corpus:FILE:LEN"))?;
            let len: usize = len.parse().map_err(|e| format!("prompt {s}: len: {e}"))?;
            if len == 0 {
                return Err(format!("prompt {s}: len 0"));
            }
            Spec::Corpus(file.into(), len)
        }
        "text" => Spec::Text(arg.into()),
        "ids" => Spec::Ids(arg.into()),
        k => return Err(format!("prompt {s}: kind {k} (corpus | text | ids)")),
    };
    Ok((name.to_string(), spec))
}

/// The L369 gate's tokenization: pieces of at most [`TOKENIZE_PIECE`] bytes, cut on char boundaries.
fn tokenize(tk: &ChatTokenizer, text: &str, want: Option<usize>) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    let mut rest = text;
    while want.is_none_or(|w| out.len() < w) && !rest.is_empty() {
        let mut cut = rest.len().min(TOKENIZE_PIECE);
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.extend(tk.encode(&rest[..cut]).map_err(|e| e.to_string())?);
        rest = &rest[cut..];
    }
    if let Some(w) = want {
        if out.len() < w {
            return Err(format!("{} tokens, {w} wanted", out.len()));
        }
        out.truncate(w);
    }
    Ok(out)
}

fn ids_from_json(v: &Value) -> Result<Vec<u32>, String> {
    let arr = match v {
        Value::Array(a) => a,
        Value::Object(o) => o
            .get("prompt_ids")
            .or_else(|| o.get("ids"))
            .and_then(Value::as_array)
            .ok_or("no \"prompt_ids\" or \"ids\" array")?,
        _ => return Err("not an array or object".into()),
    };
    let ids = arr
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("id {x} is not a u32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("no ids".into());
    }
    Ok(ids)
}

fn tokenizer<'a>(tk: &'a mut Option<ChatTokenizer>, dir: &Path) -> Result<&'a ChatTokenizer, String> {
    if tk.is_none() {
        *tk = Some(ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?);
    }
    Ok(tk.as_ref().expect("set above"))
}

fn load_prompt(tk: &mut Option<ChatTokenizer>, dir: &Path, spec: &str) -> Result<Prompt, String> {
    let (name, s) = parse_spec(spec)?;
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    let ids = match &s {
        Spec::Corpus(f, len) => tokenize(tokenizer(tk, dir)?, &read(f)?, Some(*len)),
        Spec::Text(f) => tokenize(tokenizer(tk, dir)?, &read(f)?, None),
        Spec::Ids(f) => serde_json::from_str::<Value>(&read(f)?)
            .map_err(|e| e.to_string())
            .and_then(|v| ids_from_json(&v)),
    }
    .map_err(|e| format!("prompt {name}: {e}"))?;
    Ok(Prompt { name, source: spec.to_string(), ids })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn ids_sha256(ids: &[u32]) -> String {
    let b: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    sha256_hex(&b)
}

fn logits_bytes(l: &[f32]) -> Vec<u8> {
    l.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The `k` largest finite logits, descending; a tie keeps the lower id first (the judge's order).
fn top_k(l: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut t: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &x) in l.iter().enumerate() {
        if !x.is_finite() || (t.len() == k && x <= t[k - 1].1) {
            continue;
        }
        let p = t.iter().position(|e| x > e.1).unwrap_or(t.len());
        t.insert(p, (i as u32, x));
        t.truncate(k);
    }
    t
}

/// Chunk sizes the bench hands the prefill: `chunk` each, the last one shorter; chunk 0 = one
/// token per step.
fn chunk_plan(n: usize, chunk: usize) -> Vec<usize> {
    let c = chunk.max(1);
    (0..n.div_ceil(c)).map(|i| c.min(n - i * c)).collect()
}

/// The attention cache holds whole score passes of 256 positions (as the L369 gate).
fn max_pos_for(longest: usize) -> usize {
    (longest + 1).next_multiple_of(256)
}

struct Pass {
    host_ms: f64,
    chunk_host_ms: Vec<f64>,
    streamed: Vec<u64>,
    /// Per chunk: the device span (ns) or the timeline error.
    chip_ns: Vec<Result<u64, String>>,
    classes: BTreeMap<String, (u64, u64)>,
    next: u32,
    logits: Vec<f32>,
}

fn one_pass(m: &mut Qwen38Omega, ids: &[u32], chunk: usize, stamps: bool) -> Result<Pass, String> {
    m.reset_head(0)?;
    m.set_stamps(stamps)?;
    let plan = chunk_plan(ids.len(), chunk);
    let mut p = Pass {
        host_ms: 0.0,
        chunk_host_ms: Vec::with_capacity(plan.len()),
        streamed: Vec::with_capacity(plan.len()),
        chip_ns: Vec::new(),
        classes: BTreeMap::new(),
        next: 0,
        logits: Vec::new(),
    };
    let t_all = Instant::now();
    let mut at = 0;
    for (i, &len) in plan.iter().enumerate() {
        let c = &ids[at..at + len];
        at += len;
        let last = i + 1 == plan.len();
        let s0 = m.q36_prefill_streamed();
        let t = Instant::now();
        let r = if chunk == 0 {
            m.step(c[0]).map(|(tok, _)| Some(tok))
        } else {
            m.prefill_moe_head(0, c, chunk, last).map(|(tok, _)| tok)
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let tok = match r {
            Ok(t) => t,
            Err(e) => {
                let _ = m.set_stamps(false);
                return Err(e);
            }
        };
        p.chunk_host_ms.push(ms);
        p.streamed.push(m.q36_prefill_streamed() - s0);
        if stamps {
            p.chip_ns.push(m.timeline().map(|tl| {
                for (name, ns) in &tl {
                    let e = p.classes.entry(name.clone()).or_default();
                    e.0 += ns;
                    e.1 += 1;
                }
                tl.iter().map(|e| e.1).sum()
            }));
        }
        if last {
            p.next = tok.ok_or("prefill gave no token")?;
        }
    }
    p.host_ms = t_all.elapsed().as_secs_f64() * 1e3;
    m.set_stamps(false)?;
    p.logits = m.logits_head(0)?;
    Ok(p)
}

fn knobs() -> Value {
    let mut o = serde_json::Map::new();
    for k in KNOBS {
        o.insert((*k).into(), std::env::var(k).map(Value::from).unwrap_or(Value::Null));
    }
    Value::Object(o)
}

fn plan_record(p: &Prompt, chunk: usize) -> Value {
    let plan = chunk_plan(p.ids.len(), chunk);
    json!({
        "name": p.name, "source": p.source, "tokens": p.ids.len(), "ids_sha256": ids_sha256(&p.ids),
        "chunk": chunk, "chunks": plan.len(), "chunk_sizes_first_last": [plan[0], plan[plan.len() - 1]],
        "prompt_ids": p.ids,
    })
}

fn bench_one(m: &mut Qwen38Omega, p: &Prompt, chunk: usize, reps: usize, dir: &Path) -> Result<Value, String> {
    let n = p.ids.len();
    let mut passes = Vec::with_capacity(reps);
    for r in 0..reps {
        let ps = one_pass(m, &p.ids, chunk, false)?;
        eprintln!("{} rep {r}: {:.1} ms host ({:.1} tok/s), next {}", p.name, ps.host_ms, n as f64 / (ps.host_ms / 1e3), ps.next);
        passes.push(ps);
    }
    let stamped = if std::env::var("AIEN_BENCH_STAMPS").is_ok_and(|v| v == "0") {
        None
    } else {
        Some(one_pass(m, &p.ids, chunk, true)?)
    };
    let ref0 = &passes[0];
    let bits = |l: &[f32]| l.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let b0 = bits(&ref0.logits);
    let mut diff_reps: Vec<usize> = passes.iter().enumerate().skip(1).filter(|(_, q)| bits(&q.logits) != b0).map(|(i, _)| i).collect();
    if let Some(s) = &stamped {
        if bits(&s.logits) != b0 {
            diff_reps.push(reps);
        }
    }
    let lb = logits_bytes(&ref0.logits);
    let lpath = dir.join(format!("{}.logits.f32", p.name));
    std::fs::write(&lpath, &lb).map_err(|e| format!("{}: {e}", lpath.display()))?;
    let host: Vec<f64> = passes.iter().map(|q| q.host_ms).collect();
    let mut sorted = host.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let streamed_total: u64 = ref0.streamed.iter().sum();
    let chip = stamped.as_ref().map(|s| {
        let ok: Vec<u64> = s.chip_ns.iter().filter_map(|r| r.as_ref().ok().copied()).collect();
        let errs: Vec<String> = s.chip_ns.iter().filter_map(|r| r.as_ref().err().cloned()).collect();
        let complete = errs.is_empty();
        let ms: f64 = ok.iter().sum::<u64>() as f64 / 1e6;
        let classes: BTreeMap<&String, Value> = s
            .classes
            .iter()
            .map(|(k, (ns, c))| (k, json!({"ms": *ns as f64 / 1e6, "launches": c})))
            .collect();
        json!({
            "label": "CHIP_ONLY (GPU timestamps: device span per chunk incl. inter-list host gaps; not HTTP)",
            "complete": complete,
            "chip_only_ms": if complete { json!(ms) } else { Value::Null },
            "chip_only_tok_s": if complete { json!(n as f64 / (ms / 1e3)) } else { Value::Null },
            "chunk_chip_ms": s.chip_ns.iter().map(|r| r.as_ref().ok().map(|ns| *ns as f64 / 1e6)).collect::<Vec<_>>(),
            "timeline_errors": errs.iter().take(4).collect::<Vec<_>>(),
            "timeline_error_count": errs.len(),
            "classes_ms": classes,
            "stamped_host_ms": s.host_ms,
        })
    });
    let top: Vec<Value> = top_k(&ref0.logits, TOP_K).iter().map(|(i, x)| json!([i, x])).collect();
    let rec = json!({
        "name": p.name, "source": p.source, "tokens": n, "ids_sha256": ids_sha256(&p.ids),
        "chunk": chunk, "chunks": ref0.chunk_host_ms.len(), "reps": reps,
        "next": ref0.next, "top_k": top,
        "logits_file": lpath.file_name().map(|s| s.to_string_lossy().into_owned()),
        "logits_sha256": sha256_hex(&lb),
        "reps_bit_identical": diff_reps.is_empty(), "reps_differing": diff_reps,
        "ttft_inproc_host": {
            "label": "TTFT_INPROC_HOST (host clock around the whole prompt in this process; not HTTP, not chip-only)",
            "ms": host, "median_ms": median, "median_tok_s": n as f64 / (median / 1e3),
            "chunk_ms_rep0": ref0.chunk_host_ms,
        },
        "streamed": {
            "label": "fp8 GEMM weight bytes (omega_gpu_q36_graph_prefill_streamed deltas), rep 0",
            "total_bytes": streamed_total, "bytes_per_token": streamed_total as f64 / n as f64,
            "per_chunk_bytes": ref0.streamed,
        },
        "chip": chip,
    });
    std::fs::write(dir.join(format!("{}.json", p.name)), serde_json::to_string_pretty(&rec).map_err(|e| e.to_string())? + "\n")
        .map_err(|e| format!("{}: {e}", p.name))?;
    Ok(rec)
}

fn run(args: &[String]) -> Result<bool, String> {
    let plan_only = args.get(1).is_some_and(|a| a == "--plan");
    let a = &args[if plan_only { 2 } else { 1 }..];
    if a.len() < 5 {
        return Err("usage: qwen36_prefill_bench [--plan] <checkpoint-dir> <arm> <out-dir> <reps> <prompt>...".into());
    }
    let dir = Path::new(&a[0]);
    let arm = &a[1];
    if arm.is_empty() || !arm.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(format!("arm {arm}: [A-Za-z0-9_-]+"));
    }
    let out = Path::new(&a[2]).join(arm);
    let reps: usize = a[3].parse().map_err(|e| format!("reps: {e}"))?;
    if reps == 0 {
        return Err("reps: at least 1".into());
    }
    let chunk = q36_prefill_chunk()?;
    let mut tk = None;
    let prompts = a[4..].iter().map(|s| load_prompt(&mut tk, dir, s)).collect::<Result<Vec<_>, _>>()?;
    let mut names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err("two prompts with one NAME".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let longest = prompts.iter().map(|p| p.ids.len()).max().ok_or("no prompts")?;
    let max_pos = max_pos_for(longest);
    if plan_only {
        for p in &prompts {
            let r = plan_record(p, chunk);
            eprintln!("plan {}: {} tokens, ids {}, chunk {chunk}, {} chunks", p.name, p.ids.len(), r["ids_sha256"], r["chunks"]);
            std::fs::write(out.join(format!("{}.ids.json", p.name)), serde_json::to_string(&r).map_err(|e| e.to_string())? + "\n")
                .map_err(|e| format!("{}: {e}", p.name))?;
        }
        eprintln!("plan: arm {arm}, max_pos {max_pos}, knobs {}", knobs());
        return Ok(true);
    }
    let full = aien_inference_abi::qwen38::Qwen38Config::from_checkpoint(dir)?;
    let count = match std::env::var("AIEN_BENCH_LAYERS") {
        Ok(l) => l.parse::<usize>().map_err(|e| format!("AIEN_BENCH_LAYERS: {e}"))?,
        Err(_) => full.layers,
    };
    let t = Instant::now();
    let mut m = Qwen38Omega::open_window(dir, 0, count, max_pos)?;
    let load_s = t.elapsed().as_secs_f64();
    eprintln!("arm {arm}: loaded {count} layers in {load_s:.1} s, max_pos {max_pos}, chunk {chunk}");
    let mut runs = Vec::new();
    let mut all = true;
    for p in &prompts {
        let r = bench_one(&mut m, p, chunk, reps, &out)?;
        all &= r["reps_bit_identical"].as_bool() == Some(true);
        runs.push(r);
    }
    let rec = json!({
        "bench": "qwen36_prefill_bench", "arm": arm, "layers": count, "chunk": chunk, "max_pos": max_pos,
        "load_s": load_s, "knobs": knobs(), "deterministic": all, "runs": runs,
    });
    let path = out.join("bench.json");
    std::fs::write(&path, serde_json::to_string_pretty(&rec).map_err(|e| e.to_string())? + "\n")
        .map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("BENCH {} arm {arm}", if all { "DONE" } else { "NONDETERMINISTIC" });
    Ok(all)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen36_prefill_bench: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_plan_covers_the_prompt() {
        assert_eq!(chunk_plan(300, 128), vec![128, 128, 44]);
        assert_eq!(chunk_plan(256, 128), vec![128, 128]);
        assert_eq!(chunk_plan(5, 0), vec![1; 5]);
        assert_eq!(chunk_plan(8192, 8192), vec![8192]);
        for n in 1..600 {
            for c in [0, 1, 7, 128, 512] {
                let p = chunk_plan(n, c);
                assert_eq!(p.iter().sum::<usize>(), n);
                assert!(p.iter().all(|&x| x >= 1 && x <= c.max(1)));
            }
        }
    }

    #[test]
    fn max_pos_rounds_to_score_passes() {
        assert_eq!(max_pos_for(255), 256);
        assert_eq!(max_pos_for(256), 512);
        assert_eq!(max_pos_for(32768), 33024);
    }

    #[test]
    fn top_k_is_descending_lower_id_first_and_skips_nonfinite() {
        let l = [1.0, 3.0, f32::NAN, 3.0, f32::INFINITY, 2.0, -1.0];
        assert_eq!(top_k(&l, 3), vec![(1, 3.0), (3, 3.0), (5, 2.0)]);
        assert_eq!(top_k(&l, 16).len(), 5);
    }

    #[test]
    fn specs_parse_and_refuse() {
        assert!(matches!(parse_spec("c2k=corpus:/a/b.txt:2048"), Ok((n, Spec::Corpus(_, 2048))) if n == "c2k"));
        assert!(matches!(parse_spec("x=text:/a.txt"), Ok((_, Spec::Text(_)))));
        assert!(matches!(parse_spec("x=ids:/a.json"), Ok((_, Spec::Ids(_)))));
        for bad in ["x", "=corpus:a:1", "a/b=text:x", "x=corpus:a:0", "x=corpus:a", "x=pdf:a"] {
            assert!(parse_spec(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn ids_json_forms() {
        assert_eq!(ids_from_json(&json!([1, 2, 3])).unwrap(), vec![1, 2, 3]);
        assert_eq!(ids_from_json(&json!({"prompt_ids": [7]})).unwrap(), vec![7]);
        assert_eq!(ids_from_json(&json!({"ids": [8, 9]})).unwrap(), vec![8, 9]);
        assert!(ids_from_json(&json!([])).is_err());
        assert!(ids_from_json(&json!([-1])).is_err());
        assert!(ids_from_json(&json!([4294967296u64])).is_err());
        assert!(ids_from_json(&json!({"x": [1]})).is_err());
    }

    #[test]
    fn ids_sha_is_le_u32() {
        // sha256 of the 4 bytes 01 00 00 00
        assert_eq!(ids_sha256(&[1]), sha256_hex(&[1, 0, 0, 0]));
        assert_ne!(ids_sha256(&[1, 2]), ids_sha256(&[2, 1]));
    }
}
