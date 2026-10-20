//! wormhole30: L190 checks for the 27B + 30B wormhole (docs/WORMHOLE-30B.md). Host only, no timing claims
//! except the bridge's own host cost.
//!
//!   wormhole30 vocab TOK_27B.json TOK_30B.json OUT.json INPUT.json...
//!     W1 vocabulary receipt: sizes, tokens with the same bytes (and the same id), and on every text of the
//!     INPUT files (L080 inputs: 27B prompt ids + 27B chip continuation; the user message and the continuation
//!     are the texts) the token counts, the share of each model's token boundaries that are also the other's,
//!     exact round trips, and the bridge cost per block (drafter ids -> bytes -> verifier ids) for k = 1..16.
//!
//!   wormhole30 tensors DIR_27B DIR_30B OUT.json LIST.tsv
//!     M0 shared-weights receipt: every tensor of both checkpoints (name, dtype, shape, sha256), identical
//!     tensors, identical 64 KiB byte ranges anywhere, and shapes the two have in common.
//!
//!   wormhole30 oneread DIR_30B OUT.json [ROUNDS]
//!     M0b: cold-cache load of the 30B CPU reference plus the heartbeat manifest digests, two NVMe reads
//!     (Manifest::build, then load_qwen3_coder) against one (load_qwen3_coder_tapped digesting on the way).
//!
//!   wormhole30 accept DIR_30B TOK_27B TOK_30B OUT.json K,K,.. INPUT.json...
//!     W3 forward acceptance, host only: the 30B CPU reference drafts k words, the 27B (its recorded greedy
//!     chip output in INPUT) verifies through the text bridge; output must equal the record for every k, and
//!     an accept-one-extra control (k=4) must differ. Needs about 35 GB (refuses below 45 GB available).
//!
//!   wormhole30 ringcheck DIR_30B OUT.json N SLOTS INPUT.json
//!     W2 host ring: the 30B with experts through the heartbeat ring (SLOTS slots) vs the resident reference,
//!     teacher-forced over the first N ids; logits bit for bit; corrupted digests must be refused.
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use aien_wormhole::align::{ByteVocab, Vocab};
use serde_json::{json, Value};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("vocab") if a.len() >= 6 => vocab(&a[2], &a[3], &a[4], &a[5..]),
        Some("tensors") if a.len() == 6 => tensors(&a[2], &a[3], &a[4], &a[5]),
        Some("ringcheck") if a.len() == 7 => ringcheck(&a[2], &a[3], a[4].parse().map_err(|_| "N").unwrap_or(0), a[5].parse().unwrap_or(0), &a[6]),
        Some("accept") if a.len() >= 8 => accept(&a[2], &a[3], &a[4], &a[5], &a[6], &a[7..]),
        Some("oneread") if a.len() >= 4 => oneread(&a[2], &a[3], a.get(4).and_then(|r| r.parse().ok()).unwrap_or(2)),
        _ => Err("usage: see the file header".into()),
    };
    if let Err(e) = r {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn read_json(p: &str) -> Result<Value, String> {
    serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
}

/// Byte offsets where each token ends (the token boundaries inside the text).
fn ends(voc: &ByteVocab, ids: &[u32]) -> Vec<usize> {
    let mut at = 0;
    ids.iter()
        .map(|&i| {
            at += voc.token_bytes(i).len();
            at
        })
        .collect()
}

/// Mean host nanoseconds to carry blocks of `k` ids of `from` into `to` over the whole text.
fn bridge_ns(from: &ByteVocab, to: &ByteVocab, ids: &[u32], k: usize) -> Result<(f64, usize, f64), String> {
    let mut total = 0u128;
    let mut blocks = 0usize;
    let mut out_ids = 0usize;
    for _rep in 0..5 {
        for c in ids.chunks(k) {
            let t = Instant::now();
            let b = from.decode(c);
            let (o, _) = to.encode_bytes(&b)?;
            total += t.elapsed().as_nanos();
            blocks += 1;
            out_ids += o.len();
        }
    }
    Ok((total as f64 / blocks as f64, blocks / 5, out_ids as f64 / blocks as f64))
}

fn vocab(t27: &str, t30: &str, out: &str, inputs: &[String]) -> Result<(), String> {
    let v27 = ByteVocab::from_file(Path::new(t27))?;
    let v30 = ByteVocab::from_file(Path::new(t30))?;
    let sha = |p: &str| -> Result<String, String> {
        let b = std::fs::read(p).map_err(|e| e.to_string())?;
        Ok(blake3::hash(&b).to_hex().to_string())
    };
    // vocabulary level: same byte string, same id
    let mut by_bytes30: HashMap<&[u8], u32> = HashMap::new();
    for id in 0..v30.len() as u32 {
        if !v30.added[id as usize] && !v30.token_bytes(id).is_empty() {
            by_bytes30.insert(v30.token_bytes(id), id);
        }
    }
    let (mut normal27, mut same_bytes, mut same_id) = (0usize, 0usize, 0usize);
    for id in 0..v27.len() as u32 {
        if v27.added[id as usize] || v27.token_bytes(id).is_empty() {
            continue;
        }
        normal27 += 1;
        if let Some(&j) = by_bytes30.get(v27.token_bytes(id)) {
            same_bytes += 1;
            if j == id {
                same_id += 1;
            }
        }
    }
    let first_diff = (0..v30.len().min(v27.len()) as u32)
        .find(|&i| !v30.added[i as usize] && v27.token_bytes(i) != v30.token_bytes(i));
    let special = |v: &ByteVocab| -> Vec<Value> {
        (0..v.len())
            .filter(|&i| v.special[i])
            .take(8)
            .map(|i| json!([i, String::from_utf8_lossy(v.token_bytes(i as u32))]))
            .collect()
    };

    let mut texts = Vec::new();
    let mut agg = [0usize; 6]; // n27, n30, ends27 in 30, ends30 in 27, same spans, bytes
    for p in inputs {
        let j = read_json(p)?;
        let all = ids(&j["ids"]);
        let pl = j["prompt_len"].as_u64().ok_or("prompt_len")? as usize;
        let prompt = String::from_utf8_lossy(&v27.decode(&all[..pl])).to_string();
        let user = prompt
            .split_once("user\n")
            .and_then(|(_, r)| r.split_once("<|im_end|>"))
            .map(|(u, _)| u.to_string())
            .ok_or("no user turn in the prompt")?;
        let cont: Vec<u32> = all[pl..].iter().copied().take_while(|&i| !v27.special[i as usize]).collect();
        let cont_text = String::from_utf8(v27.decode(&cont)).map_err(|e| e.to_string())?;
        for (kind, text) in [("prompt", user), ("continuation", cont_text)] {
            let a = v27.encode_str(&text)?;
            let b = v30.encode_str(&text)?;
            let rt27 = v27.decode(&a) == text.as_bytes();
            let rt30 = v30.decode(&b) == text.as_bytes();
            let e27 = ends(&v27, &a);
            let e30 = ends(&v30, &b);
            let s30: std::collections::HashSet<usize> = e30.iter().copied().collect();
            let s27: std::collections::HashSet<usize> = e27.iter().copied().collect();
            let in30 = e27.iter().filter(|e| s30.contains(e)).count();
            let in27 = e30.iter().filter(|e| s27.contains(e)).count();
            // tokens whose byte span is one whole token in the other cut
            let starts27: Vec<usize> = std::iter::once(0).chain(e27.iter().copied()).collect();
            let spans27: std::collections::HashSet<(usize, usize)> = starts27.windows(2).map(|w| (w[0], w[1])).collect();
            let starts30: Vec<usize> = std::iter::once(0).chain(e30.iter().copied()).collect();
            let same_spans = starts30.windows(2).filter(|w| spans27.contains(&(w[0], w[1]))).count();
            // the 27B's own chip ids for its continuation vs a fresh cut of the same text
            let own_cut_equal = (kind == "continuation").then(|| a == cont);
            let mut cost = Vec::new();
            for k in [1usize, 2, 4, 8, 16] {
                let (f, nb, avg) = bridge_ns(&v30, &v27, &b, k)?;
                let (r, _, ravg) = bridge_ns(&v27, &v30, &a, k)?;
                cost.push(json!({"k": k, "blocks": nb, "30b_to_27b_ns": f, "27b_ids_per_block": avg,
                                 "27b_to_30b_ns": r, "30b_ids_per_block": ravg}));
            }
            let t = Instant::now();
            let _ = v30.encode_bytes(text.as_bytes())?;
            let reencode_30_ns = t.elapsed().as_nanos();
            let t = Instant::now();
            let _ = v27.encode_bytes(text.as_bytes())?;
            let reencode_27_ns = t.elapsed().as_nanos();
            for (i, x) in [a.len(), b.len(), in30, in27, same_spans, text.len()].iter().enumerate() {
                agg[i] += x;
            }
            texts.push(json!({
                "input": p, "kind": kind, "bytes": text.len(), "tokens_27b": a.len(), "tokens_30b": b.len(),
                "roundtrip_27b": rt27, "roundtrip_30b": rt30, "own_chip_ids_equal_fresh_cut": own_cut_equal,
                "ends_27b_also_30b": in30, "ends_30b_also_27b": in27, "same_token_spans": same_spans,
                "share_27b_ends_in_30b": in30 as f64 / a.len() as f64, "share_30b_ends_in_27b": in27 as f64 / b.len() as f64,
                "bridge_cost": cost, "whole_text_reencode_ns": {"30b": reencode_30_ns, "27b": reencode_27_ns},
            }));
        }
    }
    let r = json!({
        "what": "L190 W1 vocabulary receipt (host only)",
        "tokenizer_27b": {"path": t27, "blake3": sha(t27)?, "size_with_added": v27.len(), "normal": normal27,
                          "special_first": special(&v27)},
        "tokenizer_30b": {"path": t30, "blake3": sha(t30)?, "size_with_added": v30.len(), "normal": by_bytes30.len(),
                          "special_first": special(&v30)},
        "normal_27b_tokens_whose_bytes_are_a_30b_token": same_bytes,
        "of_those_same_id": same_id,
        "first_id_with_different_bytes": first_diff,
        "texts": texts,
        "all_texts": {"bytes": agg[5], "tokens_27b": agg[0], "tokens_30b": agg[1],
                      "share_27b_ends_in_30b": agg[2] as f64 / agg[0] as f64,
                      "share_30b_ends_in_27b": agg[3] as f64 / agg[1] as f64,
                      "same_token_spans_of_30b": agg[4] as f64 / agg[1] as f64,
                      "same_token_spans_of_27b": agg[4] as f64 / agg[0] as f64},
    });
    std::fs::write(out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    println!(
        "PASS vocab: 27B {} ids, 30B {} ids, same bytes {} (same id {}), texts {}, 27B ends in 30B {:.4}, 30B ends in 27B {:.4}",
        v27.len(), v30.len(), same_bytes, same_id, r["texts"].as_array().map_or(0, Vec::len),
        r["all_texts"]["share_27b_ends_in_30b"].as_f64().unwrap_or(0.0), r["all_texts"]["share_30b_ends_in_27b"].as_f64().unwrap_or(0.0)
    );
    Ok(())
}

/// One tensor of a safetensors file: absolute byte range in the file.
struct Tensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    file: std::path::PathBuf,
    start: u64,
    len: u64,
}

fn safetensors_index(dir: &str) -> Result<Vec<Tensor>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        use std::io::Read;
        let mut fh = std::fs::File::open(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        let mut n = [0u8; 8];
        fh.read_exact(&mut n).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        fh.read_exact(&mut h).map_err(|e| e.to_string())?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("{}: {e}", f.display()))?;
        for (name, t) in h.as_object().ok_or("header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let (a, b) = (
                t["data_offsets"][0].as_u64().ok_or("data_offsets")?,
                t["data_offsets"][1].as_u64().ok_or("data_offsets")?,
            );
            out.push(Tensor {
                name: name.clone(),
                dtype: t["dtype"].as_str().unwrap_or("?").to_string(),
                shape: t["shape"].as_array().map(|s| s.iter().filter_map(Value::as_u64).collect()).unwrap_or_default(),
                file: f.clone(),
                start: 8 + n + a,
                len: b - a,
            });
        }
    }
    Ok(out)
}

/// Chunk size for the byte-range comparison (aligned to each tensor's start).
const CHUNK: usize = 64 * 1024;

struct Hashed {
    sha256: String,
    chunks: Vec<[u8; 32]>,
}

/// sha256 of every tensor's bytes and blake3 of every 64 KiB chunk, one read of each file, files in parallel.
fn hash_all(ts: &[Tensor], threads: usize) -> Result<(Vec<Hashed>, u64, f64), String> {
    use sha2::Digest;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut by_file: HashMap<&Path, Vec<usize>> = HashMap::new();
    for (i, t) in ts.iter().enumerate() {
        by_file.entry(t.file.as_path()).or_default().push(i);
    }
    let jobs: Vec<Vec<usize>> = by_file.into_values().collect();
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let results: Vec<Result<Vec<(usize, Hashed)>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut got = Vec::new();
                    let mut buf = Vec::new();
                    loop {
                        let j = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(j) else { return Ok(got) };
                        let fh = std::fs::File::open(&ts[job[0]].file).map_err(|e| e.to_string())?;
                        let mut order = job.clone();
                        order.sort_by_key(|&i| ts[i].start);
                        for i in order {
                            let t = &ts[i];
                            buf.resize(t.len as usize, 0);
                            fh.read_exact_at(&mut buf, t.start).map_err(|e| format!("{}: {e}", t.name))?;
                            let sha256 = sha2::Sha256::digest(&buf).iter().map(|b| format!("{b:02x}")).collect();
                            let chunks = buf.chunks(CHUNK).map(|c| *blake3::hash(c).as_bytes()).collect();
                            got.push((i, Hashed { sha256, chunks }));
                        }
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("hash thread panicked".into()))).collect()
    });
    let secs = t0.elapsed().as_secs_f64();
    let mut out: Vec<Option<Hashed>> = (0..ts.len()).map(|_| None).collect();
    for r in results {
        for (i, h) in r? {
            out[i] = Some(h);
        }
    }
    let bytes = ts.iter().map(|t| t.len).sum();
    Ok((out.into_iter().map(|h| h.expect("every tensor hashed")).collect(), bytes, secs))
}

/// Drake 2026-10-20T17:20Z: do the 27B and the 30B share any weights that could be read from NVMe once for both?
fn tensors(d27: &str, d30: &str, out: &str, list: &str) -> Result<(), String> {
    let t27 = safetensors_index(d27)?;
    let t30 = safetensors_index(d30)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let (h27, b27, s27) = hash_all(&t27, threads)?;
    let (h30, b30, s30) = hash_all(&t30, threads)?;

    // whole tensors with the same bytes
    let mut by_sha30: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, h) in h30.iter().enumerate() {
        by_sha30.entry(h.sha256.as_str()).or_default().push(i);
    }
    let mut same_tensors = Vec::new();
    let mut same_tensor_bytes = 0u64;
    for (i, h) in h27.iter().enumerate() {
        if let Some(js) = by_sha30.get(h.sha256.as_str()) {
            same_tensor_bytes += t27[i].len;
            same_tensors.push(json!({"27b": t27[i].name, "30b": js.iter().map(|&j| t30[j].name.clone()).collect::<Vec<_>>(),
                                     "bytes": t27[i].len, "sha256": h.sha256}));
        }
    }
    // 64 KiB byte ranges with the same bytes (all-zero chunks counted apart: they say nothing about shared weights)
    let zero = *blake3::hash(&[0u8; CHUNK]).as_bytes();
    let mut c30: HashMap<[u8; 32], ()> = HashMap::new();
    for h in &h30 {
        for c in &h.chunks {
            c30.insert(*c, ());
        }
    }
    let (mut chunks27, mut shared_chunks, mut shared_zero) = (0u64, 0u64, 0u64);
    for h in &h27 {
        for c in &h.chunks {
            chunks27 += 1;
            if c30.contains_key(c) {
                if *c == zero {
                    shared_zero += 1;
                } else {
                    shared_chunks += 1;
                }
            }
        }
    }
    // shapes that match even if the bytes differ
    let mut shapes: std::collections::BTreeMap<String, (u64, u64, Vec<String>, Vec<String>)> = Default::default();
    for (ts, side) in [(&t27, 0), (&t30, 1)] {
        for t in ts.iter() {
            let e = shapes.entry(format!("{:?}", t.shape)).or_default();
            if side == 0 {
                e.0 += 1;
                if e.2.len() < 3 {
                    e.2.push(format!("{} {}", t.name, t.dtype));
                }
            } else {
                e.1 += 1;
                if e.3.len() < 3 {
                    e.3.push(format!("{} {}", t.name, t.dtype));
                }
            }
        }
    }
    let shape_matches: Vec<Value> = shapes
        .iter()
        .filter(|(_, v)| v.0 > 0 && v.1 > 0)
        .map(|(s, v)| json!({"shape": s, "count_27b": v.0, "count_30b": v.1, "e_g_27b": v.2, "e_g_30b": v.3}))
        .collect();
    let cfg = |d: &str| -> Result<Value, String> {
        let c = read_json(&format!("{d}/config.json"))?;
        let t = if c["text_config"].is_object() { c["text_config"].clone() } else { c };
        Ok(json!({"hidden_size": t["hidden_size"], "vocab_size": t["vocab_size"], "layers": t["num_hidden_layers"],
                  "head_dim": t["head_dim"], "heads": t["num_attention_heads"], "kv_heads": t["num_key_value_heads"],
                  "intermediate_size": t["intermediate_size"], "moe_intermediate_size": t["moe_intermediate_size"],
                  "experts": t["num_experts"]}))
    };
    let gb = |b: u64| b as f64 / 1e9;
    let r = json!({
        "what": "L190 M0: shared weights between the 27B and the 30B (Drake 2026-10-20T17:20Z), host only",
        "method": "every tensor of every *.safetensors file: name, dtype, shape, sha256 of its bytes; plus blake3 of every 64 KiB chunk (aligned to the tensor start) for byte ranges shared at any position",
        "model_27b": {"dir": d27, "tensors": t27.len(), "bytes": b27, "gb": gb(b27), "hash_seconds": s27, "config": cfg(d27)?},
        "model_30b": {"dir": d30, "tensors": t30.len(), "bytes": b30, "gb": gb(b30), "hash_seconds": s30, "config": cfg(d30)?},
        "identical_tensors": same_tensors.len(), "identical_tensor_bytes": same_tensor_bytes,
        "identical_tensor_list": same_tensors,
        "chunks_64k_27b": chunks27, "chunks_64k_27b_also_in_30b_nonzero": shared_chunks,
        "chunks_64k_27b_also_in_30b_all_zero": shared_zero,
        "shared_nonzero_bytes_upper_bound": shared_chunks * CHUNK as u64,
        "gb_saved_if_loaded_once": gb(same_tensor_bytes.max(shared_chunks * CHUNK as u64)),
        "shape_matches": shape_matches,
        "per_tensor_list": list,
    });
    std::fs::write(out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut tsv = String::from("model\tname\tdtype\tshape\tbytes\tsha256\n");
    for (m, ts, hs) in [("27b", &t27, &h27), ("30b", &t30, &h30)] {
        for (t, h) in ts.iter().zip(hs.iter()) {
            tsv += &format!("{m}\t{}\t{}\t{:?}\t{}\t{}\n", t.name, t.dtype, t.shape, t.len, h.sha256);
        }
    }
    std::fs::write(list, tsv).map_err(|e| e.to_string())?;
    println!(
        "PASS tensors: 27B {} tensors {:.2} GB ({:.1} s), 30B {} tensors {:.2} GB ({:.1} s); identical tensors {} ({} bytes); shared nonzero 64 KiB chunks {} (zero chunks {}); shapes in common {}",
        t27.len(), gb(b27), s27, t30.len(), gb(b30), s30, same_tensors.len(), same_tensor_bytes, shared_chunks, shared_zero,
        r["shape_matches"].as_array().map_or(0, Vec::len)
    );
    Ok(())
}

/// Storage bytes this process has read so far (`/proc/self/io` read_bytes: block-device reads, not page-cache hits).
fn io_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("read_bytes: ").and_then(|v| v.trim().parse().ok())))
        .unwrap_or(0)
}

fn meminfo_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix(key).and_then(|v| v.split_whitespace().next()?.parse().ok())))
        .unwrap_or(0)
}

/// Drop this checkpoint's pages from the page cache (POSIX_FADV_DONTNEED, no root needed) so the next load
/// reads the NVMe.
fn make_cold(dir: &str) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            let f = std::fs::File::open(&p).map_err(|e| e.to_string())?;
            // SAFETY: valid open fd, whole-file range.
            let rc = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            if rc != 0 {
                return Err(format!("{}: posix_fadvise {rc}", p.display()));
            }
        }
    }
    Ok(())
}

/// Before: the heartbeat manifest reads every expert to digest it, then the CPU reference reads them again.
fn load_two_reads(dir: &Path) -> Result<(Value, Vec<String>), String> {
    let io0 = io_read_bytes();
    let t = Instant::now();
    let m = aien_heartbeat::manifest::Manifest::build(dir).map_err(|e| e.to_string())?;
    let manifest_s = t.elapsed().as_secs_f64();
    let io1 = io_read_bytes();
    let w = aien_inference_abi::qwen3_coder::load_qwen3_coder(dir).map_err(|e| format!("{e:?}"))?;
    let total_s = t.elapsed().as_secs_f64();
    let io2 = io_read_bytes();
    drop(w);
    Ok((
        json!({"path": "two reads", "seconds": total_s, "manifest_seconds": manifest_s,
               "reference_seconds": total_s - manifest_s, "nvme_bytes": io2 - io0,
               "nvme_bytes_manifest": io1 - io0, "nvme_bytes_reference": io2 - io1}),
        m.experts.iter().map(|e| e.digest.clone()).collect(),
    ))
}

/// After: one read. The CPU reference loader hands each expert tensor's bytes to a tap that assembles the
/// expert's heartbeat payload (segments in file order) and, once complete, digests it on worker threads.
fn load_one_read(dir: &Path) -> Result<(Value, Vec<String>), String> {
    use std::sync::mpsc;
    let io0 = io_read_bytes();
    let t = Instant::now();
    let m = aien_heartbeat::manifest::Manifest::layout(dir).map_err(|e| e.to_string())?;
    // tensor name -> (expert index, offset in its payload)
    let mut at: HashMap<String, (usize, usize)> = HashMap::new();
    for (i, e) in m.experts.iter().enumerate() {
        for tr in &e.tensors {
            at.insert(tr.name.clone(), (i, tr.offset as usize));
        }
    }
    let mut payload: Vec<Option<(Vec<u8>, u64)>> = (0..m.experts.len()).map(|_| None).collect();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(workers * 4);
    let rx = std::sync::Mutex::new(rx);
    let (digests, w, tap_stray) = std::thread::scope(|s| -> Result<_, String> {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let job = rx.lock().map_err(|_| "rx poisoned".to_string())?.recv();
                        let Ok((i, b)) = job else { return Ok::<_, String>(got) };
                        got.push((i, blake3::hash(&b).to_hex().to_string()));
                    }
                })
            })
            .collect();
        let mut stray = 0usize;
        let mut send_err = None;
        let w = aien_inference_abi::qwen3_coder::load_qwen3_coder_tapped(dir, &mut |name, bytes| {
            let Some(&(i, off)) = at.get(name) else {
                stray += 1;
                return;
            };
            let len = m.experts[i].len;
            let slot = payload[i].get_or_insert_with(|| (vec![0u8; len as usize], 0));
            slot.0[off..off + bytes.len()].copy_from_slice(bytes);
            slot.1 += bytes.len() as u64;
            if slot.1 == len {
                let (b, _) = payload[i].take().expect("slot");
                if tx.send((i, b)).is_err() {
                    send_err = Some("digest workers gone");
                }
            }
        });
        drop(tx);
        let mut digests = vec![String::new(); m.experts.len()];
        for h in hs {
            for (i, d) in h.join().map_err(|_| "digest thread panicked".to_string())?? {
                digests[i] = d;
            }
        }
        if let Some(e) = send_err {
            return Err(e.into());
        }
        Ok((digests, w.map_err(|e| format!("{e:?}"))?, stray))
    })?;
    let total_s = t.elapsed().as_secs_f64();
    let io1 = io_read_bytes();
    let incomplete = payload.iter().filter(|p| p.is_some()).count();
    drop(w);
    Ok((
        json!({"path": "one read", "seconds": total_s, "nvme_bytes": io1 - io0,
               "expert_payloads_incomplete": incomplete, "tapped_tensors_not_in_manifest": tap_stray}),
        digests,
    ))
}

/// Drake 2026-10-20T17:20Z, the two-birds path when the two models share nothing: one NVMe read of the 30B
/// experts feeds both the heartbeat ring's manifest digests and the CPU reference. Alternates two-read and
/// one-read loads from a cold cache; the one-read digests must equal the manifest's own.
fn oneread(d30: &str, out: &str, rounds: usize) -> Result<(), String> {
    let avail_gb = meminfo_kb("MemAvailable:") as f64 / 1e6 * 1.024;
    if avail_gb < 45.0 {
        return Err(format!("MemAvailable {avail_gb:.1} GB < 45 GB: the CPU reference (about 35 GB) would crowd other tenants; refusing"));
    }
    let dir = Path::new(d30);
    let mut runs = Vec::new();
    let mut want: Option<Vec<String>> = None;
    let mut mismatches = 0usize;
    for r in 0..rounds {
        for one in [false, true] {
            make_cold(d30)?;
            let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
            let (mut v, digests) = if one { load_one_read(dir)? } else { load_two_reads(dir)? };
            match &want {
                None if !one => want = Some(digests),
                Some(w) => {
                    let bad = w.iter().zip(&digests).filter(|(a, b)| a != b).count() + w.len().abs_diff(digests.len());
                    mismatches += bad;
                    v["digest_mismatches_vs_manifest_build"] = json!(bad);
                }
                None => return Err("first run must be the two-read path".into()),
            }
            v["round"] = json!(r);
            v["loadavg"] = json!(load.trim());
            eprintln!("{v}");
            runs.push(v);
        }
    }
    let mean = |p: &str, k: &str| -> f64 {
        let xs: Vec<f64> = runs.iter().filter(|v| v["path"] == p).filter_map(|v| v[k].as_f64()).collect();
        xs.iter().sum::<f64>() / xs.len().max(1) as f64
    };
    let r = json!({
        "what": "L190 M0b: one NVMe read of the 30B experts feeds the heartbeat manifest digests and the CPU reference (host only)",
        "dir": d30, "rounds": rounds, "experts": want.as_ref().map_or(0, Vec::len),
        "cold_cache": "posix_fadvise DONTNEED on every shard before each load; nvme_bytes from /proc/self/io read_bytes",
        "mean_seconds_two_reads": mean("two reads", "seconds"), "mean_seconds_one_read": mean("one read", "seconds"),
        "mean_nvme_gb_two_reads": mean("two reads", "nvme_bytes") / 1e9, "mean_nvme_gb_one_read": mean("one read", "nvme_bytes") / 1e9,
        "digest_mismatches": mismatches,
        "runs": runs,
    });
    std::fs::write(out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if mismatches > 0 {
        return Err(format!("{mismatches} expert digests from the one-read path differ from Manifest::build"));
    }
    println!(
        "PASS oneread: two reads {:.1} s ({:.1} GB from NVMe), one read {:.1} s ({:.1} GB), {} experts, digests equal",
        r["mean_seconds_two_reads"].as_f64().unwrap_or(0.0), r["mean_nvme_gb_two_reads"].as_f64().unwrap_or(0.0),
        r["mean_seconds_one_read"].as_f64().unwrap_or(0.0), r["mean_nvme_gb_one_read"].as_f64().unwrap_or(0.0), r["experts"]
    );
    Ok(())
}

/// The 27B as verifier through its own recorded greedy output (the L080 inputs: prompt + the 27B chip
/// continuation). While the fed ids equal that record, its choice after position p is the record's id p+1;
/// after a wrong id its choice is unknown (u32::MAX), which the round never commits unless the bridge is
/// wrong, and then the output differs from the record and the run fails.
struct Recorded {
    known: Vec<u32>,
    ids: Vec<u32>,
}

impl aien_wormhole::align::Lm for Recorded {
    fn feed(&mut self, t: &[u32]) -> Result<Vec<u32>, String> {
        let mut r = Vec::with_capacity(t.len());
        for &x in t {
            self.ids.push(x);
            let n = self.ids.len();
            let on_record = n < self.known.len() && self.known[..n] == self.ids[..];
            r.push(if on_record { self.known[n] } else { u32::MAX });
        }
        Ok(r)
    }
    fn truncate(&mut self, len: usize) -> Result<(), String> {
        self.ids.truncate(len);
        Ok(())
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// The 30B CPU reference (`load_qwen3_coder`, experts exact) as a greedy model with a cut-back KV cache.
struct Moe30Ref<'a> {
    w: &'a aien_inference_abi::qwen3_coder::Qwen3CoderWeights,
    st: aien_inference_abi::qwen3_coder::Qwen3CoderState,
    len: usize,
    steps: usize,
    ns: u128,
}

impl aien_wormhole::align::Lm for Moe30Ref<'_> {
    fn feed(&mut self, t: &[u32]) -> Result<Vec<u32>, String> {
        use aien_inference_abi::qwen3_coder::{compute_logits, forward_token};
        let mut r = Vec::with_capacity(t.len());
        for &x in t {
            let t0 = Instant::now();
            let h = forward_token(self.w, x, self.len, &mut self.st);
            let l = compute_logits(self.w, &h);
            self.ns += t0.elapsed().as_nanos();
            self.len += 1;
            self.steps += 1;
            // first index of the largest logit (the greedy rule every L080 tool uses)
            let mut best = 0usize;
            for (i, &v) in l.iter().enumerate() {
                if v > l[best] {
                    best = i;
                }
            }
            r.push(best as u32);
        }
        Ok(r)
    }
    fn truncate(&mut self, len: usize) -> Result<(), String> {
        for kv in &mut self.st.layers {
            kv.keys.truncate(len);
            kv.values.truncate(len);
        }
        self.len = len.min(self.len);
        Ok(())
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// W3 forward acceptance on the host: the 30B drafts, the 27B (its recorded greedy output) verifies, through
/// the text bridge, for every k; gate: output equal to the 27B's record, and the accept-one-extra control
/// must differ on at least one text.
fn accept(d30: &str, t27: &str, t30: &str, out: &str, ks: &str, inputs: &[String]) -> Result<(), String> {
    use aien_wormhole::align::generate_bridged;
    use aien_wormhole::spec::Fault;
    let ks: Vec<usize> = ks.split(',').map(|k| k.parse().map_err(|_| format!("bad k {k}"))).collect::<Result<_, _>>()?;
    let v27 = ByteVocab::from_file(Path::new(t27))?;
    let v30 = ByteVocab::from_file(Path::new(t30))?;
    let avail_gb = meminfo_kb("MemAvailable:") as f64 * 1.024e-6;
    if avail_gb < 45.0 {
        return Err(format!("MemAvailable {avail_gb:.1} GB < 45 GB for the 30B CPU reference (about 35 GB); refusing"));
    }
    let t = Instant::now();
    let w = aien_inference_abi::qwen3_coder::load_qwen3_coder(Path::new(d30)).map_err(|e| format!("{e:?}"))?;
    let load_s = t.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let mut fails = Vec::new();
    let mut control_differs = false;
    for p in inputs {
        let j = read_json(p)?;
        let all = ids(&j["ids"]);
        let pl = j["prompt_len"].as_u64().ok_or("prompt_len")? as usize;
        let n = all[pl..].iter().take_while(|&&i| !v27.special[i as usize]).count();
        let known = all[..pl + n].to_vec();
        let prompt27 = String::from_utf8_lossy(&v27.decode(&all[..pl])).to_string();
        let user = prompt27
            .split_once("user\n")
            .and_then(|(_, r)| r.split_once("<|im_end|>"))
            .map(|(u, _)| u.to_string())
            .ok_or("no user turn in the prompt")?;
        let d_prompt = v30.encode_str(&format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"))?;
        let runs: Vec<(usize, Fault)> = ks
            .iter()
            .map(|&k| (k, Fault::None))
            .chain(std::iter::once((4, Fault::AcceptOneExtra)))
            .collect();
        for (k, fault) in runs {
            let mut d = Moe30Ref { w: &w, st: aien_inference_abi::qwen3_coder::Qwen3CoderState::new(), len: 0, steps: 0, ns: 0 };
            let mut v = Recorded { known: known.clone(), ids: Vec::new() };
            let t = Instant::now();
            let (got, s) = match generate_bridged(&mut d, &v30, &mut v, &v27, &d_prompt, &all[..pl], n, k, &[], fault) {
                Ok(x) => x,
                // a control that breaks the round outright also went red
                Err(e) if fault != Fault::None => {
                    eprintln!("control {fault:?} k={k} {p}: {e}");
                    control_differs = true;
                    continue;
                }
                Err(e) => return Err(e),
            };
            let equal = got[..] == known[pl..];
            let first_diff = got.iter().zip(&known[pl..]).position(|(a, b)| a != b);
            let e = s.accepted as f64 / s.rounds.max(1) as f64 + 1.0;
            let r = json!({"input": p, "k": k, "fault": format!("{fault:?}"), "words": n, "equal_to_verifier": equal,
                           "first_diff": first_diff, "accepted": s.accepted, "drafted_27b_ids": s.drafted_v,
                           "drafted_30b_ids": s.drafted_d, "rounds": s.rounds, "words_per_round": n as f64 / s.rounds.max(1) as f64,
                           "accepted_per_round_plus_one": e, "drafter_steps": d.steps, "drafter_refeed": s.drafter_refeed,
                           "cut_tails": s.cut_tails, "bridge_ns": s.bridge_ns, "drafter_s": d.ns as f64 / 1e9,
                           "wall_s": t.elapsed().as_secs_f64()});
            eprintln!("{r}");
            match fault {
                Fault::None if !equal => fails.push(format!("{p} k={k}: output differs from the 27B at {first_diff:?}")),
                Fault::None => {}
                _ => control_differs |= !equal,
            }
            results.push(r);
            std::fs::write(out, serde_json::to_string_pretty(&json!({"partial": true, "runs": results})).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        }
    }
    let r = json!({
        "what": "L190 W3 forward acceptance (host only): 30B CPU reference drafts, the 27B's recorded chip greedy output verifies, through the text bridge",
        "dir_30b": d30, "load_seconds": load_s, "ks": ks,
        "gate_output_equal_to_verifier": fails.is_empty(), "fails": fails, "control_accept_one_extra_differs": control_differs,
        "runs": results,
    });
    std::fs::write(out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if !r["fails"].as_array().is_some_and(Vec::is_empty) {
        return Err(format!("FAIL gate: {}", r["fails"]));
    }
    if !control_differs {
        return Err("FAIL control: accepting one refused draft did not change any output".into());
    }
    println!("PASS accept: every k equal to the 27B's own output; control differs");
    Ok(())
}

fn logits_digest(l: &[f32]) -> String {
    let mut b = Vec::with_capacity(l.len() * 4);
    for v in l {
        b.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    blake3::hash(&b).to_hex()[..16].to_string()
}

/// W2 host ring exactness: the 30B with its experts through the heartbeat ring (SLOTS slots) against the
/// resident CPU reference, teacher-forced over the first N ids of INPUT; logits must match bit for bit.
/// Then the control: every expert digest of layer 0 corrupted, the next step must refuse (fail closed).
/// Ring first (about 6 GB + slots), resident after (about 35 GB), so the peak is the resident one.
fn ringcheck(d30: &str, out: &str, n: usize, slots: usize, input: &str) -> Result<(), String> {
    use aien_wormhole::moe30::{argmax, Moe30Ring};
    let j = read_json(input)?;
    let all = ids(&j["ids"]);
    let feed: Vec<u32> = all.iter().copied().take(n).collect();
    let avail0 = meminfo_kb("MemAvailable:");
    let t = Instant::now();
    let manifest = aien_heartbeat::manifest::Manifest::build(Path::new(d30)).map_err(|e| e.to_string())?;
    let manifest_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut ring = Moe30Ring::open(Path::new(d30), manifest, slots, 8, true)?;
    let open_s = t.elapsed().as_secs_f64();
    let avail1 = meminfo_kb("MemAvailable:");
    let mut ring_rows = Vec::new();
    for &x in &feed {
        let l = ring.step(x)?;
        ring_rows.push((argmax(&l), logits_digest(&l)));
    }
    let ring_ms = ring.ns as f64 / 1e6 / feed.len() as f64;
    let stats = serde_json::to_value(&ring.cache.stats).map_err(|e| e.to_string())?;
    let dense = ring.dense_bytes();
    let slot_bytes = aien_heartbeat::stream::Streamer::slot_bytes(&ring.cache.manifest);
    for e in 0..128u16 {
        ring.cache.corrupt_digest(0, e);
    }
    ring.cache.evict_all();
    let control = (0..4).map(|k| ring.step(feed[k % feed.len()])).find_map(Result::err);
    drop(ring);
    if meminfo_kb("MemAvailable:") as f64 * 1.024e-6 < 45.0 {
        return Err("MemAvailable < 45 GB for the resident reference; refusing".into());
    }
    let w = aien_inference_abi::qwen3_coder::load_qwen3_coder(Path::new(d30)).map_err(|e| format!("{e:?}"))?;
    let mut st = aien_inference_abi::qwen3_coder::Qwen3CoderState::new();
    let mut same = 0usize;
    let mut first_diff = None;
    for (p, &x) in feed.iter().enumerate() {
        let h = aien_inference_abi::qwen3_coder::forward_token(&w, x, p, &mut st);
        let l = aien_inference_abi::qwen3_coder::compute_logits(&w, &h);
        if (argmax(&l), logits_digest(&l)) == ring_rows[p] {
            same += 1;
        } else if first_diff.is_none() {
            first_diff = Some(p);
        }
    }
    let r = json!({
        "what": "L190 W2 host ring exactness: 30B experts through the heartbeat ring vs the resident CPU reference",
        "input": input, "tokens": feed.len(), "slots": slots, "slot_bytes": slot_bytes,
        "ring_bytes": slots * slot_bytes, "dense_bytes": dense, "manifest_build_s": manifest_s, "ring_open_s": open_s,
        "mem_available_kb_before": avail0, "mem_available_kb_after_open": avail1,
        "ring_ms_per_token": ring_ms, "cache": stats,
        "logits_bit_equal": same, "first_diff": first_diff,
        "control_corrupt_digest_refused": control.is_some(), "control_error": control,
    });
    std::fs::write(out, serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if same != feed.len() {
        return Err(format!("FAIL: {same}/{} tokens bit-equal, first diff at {first_diff:?}", feed.len()));
    }
    if control.is_none() {
        return Err("FAIL control: corrupted digests were not refused".into());
    }
    println!("PASS ringcheck: {same}/{} tokens logits bit-equal, {ring_ms:.0} ms/token on the ring, corrupt digest refused", feed.len());
    Ok(())
}
