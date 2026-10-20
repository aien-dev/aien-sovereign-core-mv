//! L308 typewriter experts: which experts a MoE model uses per task domain (the "type head").
//!
//!   typewriter routes <checkpoint-dir> <text> <max-tokens> <out.json> [head.json]
//!   typewriter heads  <coverage> <out.json> <domain>=<fit.json>,<held.json> ...
//!
//! `routes` runs the text through the host reference in one teacher-forced pass and writes, per
//! layer, the routed expert ids and their renormalised weights for every token, plus each token's
//! negative log-likelihood. With `head.json` (written by `heads`) routing is restricted to that
//! head's experts per layer: the lossy typewriter (H5), whose output differs from the model.
//!
//! `heads` reads the fit and held-out routes of each domain and reports, per layer, how many
//! experts cover 90 and 99 percent of the routed slots, the overlap between domains, and how much
//! of each held-out text a fit head covers (its own domain and the others). It writes, per domain,
//! the smallest per-layer set covering `coverage` of the fit slots: the head.
//!
//! Host only (no chip). Model-independent: any checkpoint `Qwen38Reference` opens with a MoE block.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::{argmax, negative_log_likelihood, Logits, Qwen38Reference};
use aien_inference_abi::tokenizer::ChatTokenizer;
use serde_json::{json, Value};

const USAGE: &str = "usage: typewriter routes <dir> <text> <max-tokens> <out.json> [head.json]\n       typewriter heads <coverage> <out.json> <domain>=<fit.json>,<held.json> ...";

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))
}

fn write_json(path: &str, v: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec(v).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{path}: {e}"))
}

/// Per layer allowed-expert mask from a `heads` file entry `{"layers": [[ids...], ...]}`.
fn head_mask(head: &Value, experts: usize) -> Result<Vec<Vec<bool>>, String> {
    let layers = head["layers"].as_array().ok_or("head: no layers")?;
    layers
        .iter()
        .map(|ids| {
            let mut allow = vec![false; experts];
            for id in ids.as_array().ok_or("head: layer is not a list")? {
                let e = id.as_u64().ok_or("head: id")? as usize;
                *allow.get_mut(e).ok_or("head: id out of range")? = true;
            }
            Ok(allow)
        })
        .collect()
}

fn routes(args: &[String]) -> Result<(), String> {
    if args.len() != 4 && args.len() != 5 {
        return Err(USAGE.into());
    }
    let (dir, text_path, out_path) = (Path::new(&args[0]), &args[1], &args[3]);
    let count: usize = args[2].parse().map_err(|_| USAGE.to_string())?;
    let full = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;
    let tokenizer =
        ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    // The tokenizer refuses more than 2048 tokens: cut the text at 8, then fewer, characters per
    // wanted token until it encodes (code packs about 3 characters per token, prose about 4).
    let mut ids = Vec::new();
    for per in [8usize, 6, 4, 3, 2] {
        let mut text = full.clone();
        if let Some((cut, _)) = full.char_indices().nth(count * per) {
            text.truncate(cut);
        }
        if let Ok(v) = tokenizer.encode(&text) {
            ids = v;
            break;
        }
    }
    ids.truncate(count);
    if ids.len() < 2 {
        return Err("routes: the text needs at least two tokens".into());
    }
    let mut model = Qwen38Reference::open(dir)?;
    let moe = model.config.moe.clone().ok_or("routes: the checkpoint has no MoE block")?;
    let head_path = args.get(4);
    if let Some(p) = head_path {
        model.expert_mask = Some(head_mask(&read_json(p)?, moe.experts)?);
    }
    let vocab = model.config.vocab;
    let mut state = model.new_state();
    let t0 = Instant::now();
    let logits = model.forward(&mut state, &ids, Logits::All)?;
    let pass_s = t0.elapsed().as_secs_f64();
    let mut nll = Vec::with_capacity(ids.len() - 1);
    let mut argmaxes = Vec::with_capacity(ids.len());
    let mut hits = 0usize;
    for (i, row) in logits.chunks_exact(vocab).enumerate() {
        let top = argmax(row);
        argmaxes.push(top as u32);
        if i + 1 < ids.len() {
            nll.push(negative_log_likelihood(row, ids[i + 1] as usize));
            hits += usize::from(top == ids[i + 1] as usize);
        }
    }
    let mean = nll.iter().sum::<f64>() / nll.len() as f64;
    eprintln!(
        "{text_path}: {} tokens in {pass_s:.1} s, mean nll {mean:.4}, accuracy {:.3}",
        ids.len(),
        hits as f64 / nll.len() as f64
    );
    let report = json!({
        "mode": "routes", "text": text_path, "checkpoint": dir, "head": head_path,
        "experts": moe.experts, "top_k": moe.top_k, "layers": model.routes.len(),
        "tokens": ids.len(), "ids": ids, "argmax": argmaxes,
        "nll": nll, "mean_nll": mean, "next_token_accuracy": hits as f64 / nll.len() as f64,
        "pass_seconds": pass_s, "routes": model.routes, "weights": model.route_weights,
    });
    write_json(out_path, &report)
}

/// One routes file: per layer, the routed ids `[t][k]` flattened.
struct Trace {
    experts: usize,
    top_k: usize,
    routes: Vec<Vec<u16>>,
}

fn trace(path: &str) -> Result<Trace, String> {
    let v = read_json(path)?;
    let experts = v["experts"].as_u64().ok_or(format!("{path}: experts"))? as usize;
    let top_k = v["top_k"].as_u64().ok_or(format!("{path}: top_k"))? as usize;
    let routes = v["routes"]
        .as_array()
        .ok_or(format!("{path}: routes"))?
        .iter()
        .map(|l| {
            l.as_array()
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(u64::MAX) as u16).collect())
                .ok_or(format!("{path}: layer"))
        })
        .collect::<Result<Vec<Vec<u16>>, String>>()?;
    Ok(Trace { experts, top_k, routes })
}

/// Experts of one layer by routed-slot count, most used first (lower id on ties).
fn ranked(slots: &[u16], experts: usize) -> Vec<(usize, usize)> {
    let mut count = vec![0usize; experts];
    for &e in slots {
        count[e as usize] += 1;
    }
    let mut r: Vec<(usize, usize)> = count.into_iter().enumerate().collect();
    r.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    r
}

/// The smallest prefix of `ranked` that covers `cov` of the slots.
fn cover(ranked: &[(usize, usize)], cov: f64) -> Vec<usize> {
    let total: usize = ranked.iter().map(|r| r.1).sum();
    let mut acc = 0usize;
    let mut set = Vec::new();
    for &(e, c) in ranked {
        if acc as f64 >= cov * total as f64 {
            break;
        }
        acc += c;
        set.push(e);
    }
    set
}

fn frac_inside(slots: &[u16], set: &[usize], experts: usize) -> f64 {
    let mut inside = vec![false; experts];
    set.iter().for_each(|&e| inside[e] = true);
    slots.iter().filter(|&&e| inside[e as usize]).count() as f64 / slots.len().max(1) as f64
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    let inter = a.iter().filter(|e| b.contains(e)).count();
    let union = a.len() + b.len() - inter;
    inter as f64 / union.max(1) as f64
}

fn stats(xs: &[f64]) -> Value {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    json!({"mean": mean, "min": min, "max": max})
}

fn heads(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(USAGE.into());
    }
    let cov: f64 = args[0].parse().map_err(|_| USAGE.to_string())?;
    let out_path = &args[1];
    let mut fit: BTreeMap<String, Trace> = BTreeMap::new();
    let mut held: BTreeMap<String, Trace> = BTreeMap::new();
    for spec in &args[2..] {
        let (name, files) = spec.split_once('=').ok_or(USAGE)?;
        let (f, h) = files.split_once(',').ok_or(USAGE)?;
        fit.insert(name.into(), trace(f)?);
        held.insert(name.into(), trace(h)?);
    }
    let names: Vec<String> = fit.keys().cloned().collect();
    let first = &fit[&names[0]];
    let (experts, layers) = (first.experts, first.routes.len());
    let mut report = serde_json::Map::new();
    let mut head_sets: BTreeMap<String, Vec<Vec<usize>>> = BTreeMap::new();
    for name in &names {
        let t = &fit[name];
        let (mut n90, mut n99, mut ncov) = (Vec::new(), Vec::new(), Vec::new());
        let mut sets = Vec::with_capacity(layers);
        for l in 0..layers {
            let r = ranked(&t.routes[l], experts);
            n90.push(cover(&r, 0.90).len() as f64);
            n99.push(cover(&r, 0.99).len() as f64);
            let set = cover(&r, cov);
            ncov.push(set.len() as f64);
            sets.push(set);
        }
        // How much of each held-out text this fit head covers, per layer then averaged.
        let mut on_held = serde_json::Map::new();
        for other in &names {
            let h = &held[other];
            let fr: Vec<f64> =
                (0..layers).map(|l| frac_inside(&h.routes[l], &sets[l], experts)).collect();
            on_held.insert(other.clone(), stats(&fr));
        }
        report.insert(
            name.clone(),
            json!({
                "tokens_fit": t.routes[0].len() / t.top_k.max(1),
                "experts_for_90": stats(&n90), "experts_for_99": stats(&n99),
                "experts_in_head": stats(&ncov), "head_experts_total": ncov.iter().sum::<f64>(),
                "per_layer_90": n90, "per_layer_99": n99,
                "head_covers_held_out": Value::Object(on_held),
            }),
        );
        head_sets.insert(name.clone(), sets);
    }
    // Controls (equal size per layer): a general-purpose head (the most used experts over every
    // domain's fit text pooled) and a random head (fixed seed). A domain head is only a "type
    // head" if it covers its own held-out text better than both.
    let mut controls = serde_json::Map::new();
    let mut control_sets: BTreeMap<String, Vec<Vec<usize>>> = BTreeMap::new();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for name in &names {
        let (mut own, mut glob, mut rand) = (Vec::new(), Vec::new(), Vec::new());
        let (mut gsets, mut rsets) = (Vec::new(), Vec::new());
        for l in 0..layers {
            let n = head_sets[name][l].len();
            let pooled: Vec<u16> = names.iter().flat_map(|d| fit[d].routes[l].iter().copied()).collect();
            let g: Vec<usize> = ranked(&pooled, experts).into_iter().take(n).map(|r| r.0).collect();
            let mut perm: Vec<usize> = (0..experts).collect();
            for i in (1..experts).rev() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                perm.swap(i, (seed % (i as u64 + 1)) as usize);
            }
            let r: Vec<usize> = perm.into_iter().take(n).collect();
            let h = &held[name].routes[l];
            own.push(frac_inside(h, &head_sets[name][l], experts));
            glob.push(frac_inside(h, &g, experts));
            rand.push(frac_inside(h, &r, experts));
            gsets.push(g);
            rsets.push(r);
        }
        controls.insert(
            name.clone(),
            json!({"own_head_on_held": stats(&own), "general_head_on_held": stats(&glob), "random_head_on_held": stats(&rand)}),
        );
        control_sets.insert(format!("general-for-{name}"), gsets);
        control_sets.insert(format!("random-for-{name}"), rsets);
    }
    // Overlap of the heads between domains (Jaccard per layer, averaged).
    let mut overlap = serde_json::Map::new();
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            let j: Vec<f64> =
                (0..layers).map(|l| jaccard(&head_sets[a][l], &head_sets[b][l])).collect();
            overlap.insert(format!("{a}~{b}"), stats(&j));
        }
    }
    let heads_out: serde_json::Map<String, Value> = head_sets
        .iter()
        .chain(control_sets.iter())
        .map(|(n, s)| (n.clone(), json!({"coverage": cov, "layers": s})))
        .collect();
    let out = json!({
        "mode": "heads", "coverage": cov, "experts": experts, "layers": layers,
        "domains": Value::Object(report), "overlap_jaccard": Value::Object(overlap), "controls": Value::Object(controls),
        "heads": Value::Object(heads_out),
    });
    write_json(out_path, &out)?;
    eprintln!("{}", serde_json::to_string_pretty(&out["overlap_jaccard"]).unwrap_or_default());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = match args.get(1).map(String::as_str) {
        Some("routes") => routes(&args[2..]),
        Some("heads") => heads(&args[2..]),
        Some("plan") => plan(&args[2..]),
        Some("load") => load(&args[2..]),
        Some("trace") => trace_cmd(&args[2..]),
        _ => Err(USAGE.into()),
    };
    if let Err(e) = r {
        eprintln!("typewriter: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------------------------
// Placement plan (rip the model apart by its own structure) and head load timing (H2).
// ---------------------------------------------------------------------------------------------

/// One tensor's place on disk: shard file, absolute byte offset, length.
#[derive(Clone)]
struct TensorPlace {
    name: String,
    file: std::path::PathBuf,
    offset: u64,
    len: u64,
}

/// Every tensor of a sharded safetensors checkpoint, from the index and each shard's header.
fn tensor_places(dir: &Path) -> Result<Vec<TensorPlace>, String> {
    use std::io::Read;
    let index = read_json(&dir.join("model.safetensors.index.json").to_string_lossy())?;
    let map = index["weight_map"].as_object().ok_or("index: weight_map")?;
    let mut shards: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
    shards.sort();
    shards.dedup();
    let mut out = Vec::new();
    for shard in shards {
        let file = dir.join(&shard);
        let mut f = std::fs::File::open(&file).map_err(|e| format!("{shard}: {e}"))?;
        let mut n = [0u8; 8];
        f.read_exact(&mut n).map_err(|e| format!("{shard}: {e}"))?;
        let hlen = u64::from_le_bytes(n);
        let mut h = vec![0u8; hlen as usize];
        f.read_exact(&mut h).map_err(|e| format!("{shard}: {e}"))?;
        let header: Value = serde_json::from_slice(&h).map_err(|e| format!("{shard}: {e}"))?;
        for (name, t) in header.as_object().ok_or("header")? {
            if name == "__metadata__" {
                continue;
            }
            let off = t["data_offsets"].as_array().ok_or(format!("{name}: data_offsets"))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            out.push(TensorPlace { name: name.clone(), file: file.clone(), offset: 8 + hlen + a, len: b - a });
        }
    }
    Ok(out)
}

/// The model's own structure: (class, layer, expert) from a tensor name.
fn part_of(name: &str) -> (&'static str, Option<usize>, Option<usize>) {
    let num_after = |key: &str| -> Option<usize> {
        let i = name.find(key)? + key.len();
        name[i..].split('.').next()?.parse().ok()
    };
    let layer = num_after(".layers.");
    if name.starts_with("mtp") {
        return ("mtp", layer, None);
    }
    if name.contains("visual") || name.contains("vision") {
        return ("vision", layer, None);
    }
    if let Some(e) = num_after(".experts.") {
        return ("expert", layer, Some(e));
    }
    let class = if name.contains("shared_expert") {
        "shared_expert"
    } else if name.contains(".mlp.gate.") {
        "router"
    } else if name.contains("_attn.") {
        "attention"
    } else if name.contains("layernorm") || name.contains("norm.weight") && layer.is_some() {
        "norm"
    } else if name.contains("embed_tokens") {
        "embedding"
    } else if name.starts_with("lm_head") {
        "lm_head"
    } else if layer.is_none() && name.contains("norm") {
        "final_norm"
    } else if name.contains("visual") || name.contains("vision") {
        "vision"
    } else {
        "other"
    };
    (class, layer, None)
}

/// `typewriter plan <dir> <tiers.json> <heads.json> <domain> <out.json>`
fn plan(args: &[String]) -> Result<(), String> {
    if args.len() != 5 {
        return Err("usage: typewriter plan <dir> <tiers.json> <heads.json> <domain> <out.json>".into());
    }
    let dir = Path::new(&args[0]);
    let tiers = read_json(&args[1])?;
    let heads = read_json(&args[2])?;
    let head = &heads["heads"][&args[3]]["layers"];
    let config = read_json(&dir.join("config.json").to_string_lossy())?;
    let text = if config["text_config"].is_object() { &config["text_config"] } else { &config };
    let top_k = text["num_experts_per_tok"].as_u64().ok_or("config: num_experts_per_tok")? as usize;
    let places = tensor_places(dir)?;
    let mut by_class: BTreeMap<&str, (u64, usize)> = BTreeMap::new();
    let mut expert_bytes: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    for p in &places {
        let (c, l, e) = part_of(&p.name);
        let s = by_class.entry(c).or_insert((0, 0));
        s.0 += p.len;
        s.1 += 1;
        if let (Some(l), Some(e)) = (l, e) {
            *expert_bytes.entry((l, e)).or_insert(0) += p.len;
        }
    }
    let in_head = |l: usize, e: usize| -> bool {
        head[l].as_array().map(|a| a.iter().any(|x| x.as_u64() == Some(e as u64))).unwrap_or(false)
    };
    let (mut head_b, mut rest_b, mut head_n) = (0u64, 0u64, 0usize);
    let layers = expert_bytes.keys().map(|k| k.0).max().map_or(0, |m| m + 1);
    for (&(l, e), &b) in &expert_bytes {
        if in_head(l, e) {
            head_b += b;
            head_n += 1;
        } else {
            rest_b += b;
        }
    }
    let one_expert = expert_bytes.values().next().copied().unwrap_or(0);
    let trunk_classes = ["attention", "norm", "router", "shared_expert", "final_norm"];
    let trunk_b: u64 = trunk_classes.iter().map(|c| by_class.get(c).map_or(0, |s| s.0)).sum();
    let lm_b = by_class.get("lm_head").map_or(0, |s| s.0);
    let embed_row = by_class.get("embedding").map_or(0, |s| s.0)
        / text["vocab_size"].as_u64().unwrap_or(1).max(1);
    // Bytes one decode beat (one token) must read: the whole trunk, the lm_head, one embedding
    // row and top_k experts per layer.
    let beat_b = trunk_b + lm_b + embed_row + (top_k * layers) as u64 * one_expert;
    let mut tier_rows = Vec::new();
    for t in tiers["tiers"].as_array().ok_or("tiers: list")? {
        let gbs = t["read_gb_s"].as_f64().unwrap_or(0.0);
        let cap = t["bytes"].as_u64().unwrap_or(0);
        let us = |b: u64| if gbs > 0.0 { b as f64 / (gbs * 1e3) } else { f64::NAN };
        tier_rows.push(json!({
            "tier": t["name"], "bytes": cap, "read_gb_s": gbs, "source": t["source"],
            "beat_us_if_all_read_here": us(beat_b), "head_load_us": us(head_b),
            "fits_trunk": cap >= trunk_b + lm_b, "fits_head": cap >= head_b,
            "fits_one_layer_active_experts": cap >= top_k as u64 * one_expert,
        }));
    }
    let classes: serde_json::Map<String, Value> = by_class
        .iter()
        .map(|(c, (b, n))| (c.to_string(), json!({"bytes": b, "tensors": n})))
        .collect();
    let out = json!({
        "mode": "plan", "checkpoint": dir, "domain": args[3], "layers": layers, "top_k": top_k,
        "classes": Value::Object(classes),
        "one_expert_bytes": one_expert, "experts_total": expert_bytes.len(),
        "trunk_bytes": trunk_b, "lm_head_bytes": lm_b, "embedding_row_bytes": embed_row,
        "head_experts": head_n, "head_bytes": head_b, "rest_expert_bytes": rest_b,
        "beat_bytes": beat_b, "tiers": tier_rows,
        "placement": {
            "math_cores": "per-layer kernels and launch plan built once at load (no compile, no allocation per beat)",
            "fast_memory_now": "trunk + lm_head + current head, resident in the GPU-visible DRAM region",
            "fast_memory_next": "next head, streamed by the Heartbeat ring into a second resident region while the current head types",
            "big_base": "every other expert: NVMe, packed in beat order (layer by layer) so the ring reads it sequentially",
        },
    });
    write_json(&args[4], &out)?;
    eprintln!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    Ok(())
}

#[cfg(target_arch = "aarch64")]
const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))]
const O_DIRECT: i32 = 0o40000;
const ALIGN: usize = 4096;

/// A heap buffer aligned for O_DIRECT.
struct Aligned {
    ptr: *mut u8,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let len = len.div_ceil(ALIGN) * ALIGN;
        let layout = std::alloc::Layout::from_size_align(len.max(ALIGN), ALIGN).expect("layout");
        // SAFETY: non-zero size, valid alignment; zeroed so every page is faulted in up front.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory");
        Self { ptr, len: len.max(ALIGN) }
    }
    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr owns len bytes for the life of self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN).expect("layout");
        // SAFETY: allocated with this layout in new().
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

fn open_direct(path: &Path) -> Result<std::fs::File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Reads `[offset, offset + len)` of `f` with O_DIRECT into `buf` (whole aligned pages; the
/// bytes read are returned, which can exceed `len` by the alignment slack).
fn read_direct(f: &std::fs::File, offset: u64, len: u64, buf: &mut Aligned) -> Result<u64, String> {
    use std::os::unix::fs::FileExt;
    let start = offset / ALIGN as u64 * ALIGN as u64;
    let end = (offset + len).div_ceil(ALIGN as u64) * ALIGN as u64;
    let want = (end - start) as usize;
    let mut done = 0usize;
    while done < want {
        let chunk = (want - done).min(buf.len);
        let n = f
            .read_at(&mut buf.slice()[..chunk], start + done as u64)
            .map_err(|e| format!("read_at: {e}"))?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done as u64)
}

/// `typewriter load <dir> <heads.json> <domain> <packed-file> <rounds> <out.json>`: H2 timing.
/// Per round: (a) the head's expert tensors read from the checkpoint where they lie (O_DIRECT,
/// one read per tensor); (b) the same bytes read from one file packed in beat order (layer by
/// layer, O_DIRECT, 64 MiB reads); (c) the head copied from host memory into a separate resident
/// region (pre-faulted). The packed file is written once (first run) and checked by length.
fn load(args: &[String]) -> Result<(), String> {
    use std::io::Write;
    if args.len() != 6 {
        return Err("usage: typewriter load <dir> <heads.json> <domain> <packed-file> <rounds> <out.json>".into());
    }
    let dir = Path::new(&args[0]);
    let heads = read_json(&args[1])?;
    let head = &heads["heads"][&args[2]]["layers"];
    let packed = Path::new(&args[3]);
    let rounds: usize = args[4].parse().map_err(|_| "rounds".to_string())?;
    let mut parts: Vec<(usize, usize, TensorPlace)> = tensor_places(dir)?
        .into_iter()
        .filter_map(|p| match part_of(&p.name) {
            ("expert", Some(l), Some(e)) => {
                let ok = head[l].as_array().is_some_and(|a| a.iter().any(|x| x.as_u64() == Some(e as u64)));
                ok.then_some((l, e, p))
            }
            _ => None,
        })
        .collect();
    parts.sort_by(|a, b| (a.0, a.1, &a.2.name).cmp(&(b.0, b.1, &b.2.name)));
    let total: u64 = parts.iter().map(|p| p.2.len).sum();
    let experts = {
        let mut v: Vec<(usize, usize)> = parts.iter().map(|p| (p.0, p.1)).collect();
        v.dedup();
        v.len()
    };
    // Packed file in beat order, written once (tensor bytes back to back, padded to ALIGN).
    let padded = total.div_ceil(ALIGN as u64) * ALIGN as u64;
    if std::fs::metadata(packed).map(|m| m.len()).ok() != Some(padded) {
        let mut w = std::io::BufWriter::with_capacity(64 << 20, std::fs::File::create(packed).map_err(|e| e.to_string())?);
        for (_, _, p) in &parts {
            let mut f = std::fs::File::open(&p.file).map_err(|e| e.to_string())?;
            use std::io::{Read, Seek, SeekFrom};
            f.seek(SeekFrom::Start(p.offset)).map_err(|e| e.to_string())?;
            let mut b = vec![0u8; p.len as usize];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            w.write_all(&b).map_err(|e| e.to_string())?;
        }
        w.write_all(&vec![0u8; (padded - total) as usize]).map_err(|e| e.to_string())?;
        w.into_inner().map_err(|e| e.to_string())?.sync_all().map_err(|e| e.to_string())?;
    }
    let mut buf = Aligned::new(64 << 20);
    let mut host = Aligned::new(padded as usize);
    let mut resident = Aligned::new(padded as usize);
    let mut rows = Vec::new();
    for round in 0..rounds {
        // (a) scattered: one O_DIRECT read per tensor where the checkpoint keeps it.
        let mut files: BTreeMap<std::path::PathBuf, std::fs::File> = BTreeMap::new();
        let t = Instant::now();
        let mut read_a = 0u64;
        for (_, _, p) in &parts {
            if !files.contains_key(&p.file) {
                files.insert(p.file.clone(), open_direct(&p.file)?);
            }
            read_a += read_direct(&files[&p.file], p.offset, p.len, &mut buf)?;
        }
        let scattered_s = t.elapsed().as_secs_f64();
        // (b) packed in beat order: sequential O_DIRECT reads straight into host memory.
        let f = open_direct(packed)?;
        let t = Instant::now();
        let mut off = 0u64;
        while off < padded {
            let n = ((padded - off) as usize).min(64 << 20);
            use std::os::unix::fs::FileExt;
            let got = f.read_at(&mut host.slice()[off as usize..off as usize + n], off).map_err(|e| e.to_string())?;
            if got == 0 {
                return Err("packed file short".into());
            }
            off += got as u64;
        }
        let packed_s = t.elapsed().as_secs_f64();
        // (c) host memory -> resident region (same DRAM pool on GB10: a memory-to-memory copy).
        let t = Instant::now();
        resident.slice()[..padded as usize].copy_from_slice(&host.slice()[..padded as usize]);
        let copy_s = t.elapsed().as_secs_f64();
        let check = resident.slice()[(padded as usize) / 2] ^ host.slice()[(padded as usize) / 2];
        let gbs = |b: u64, s: f64| b as f64 / s / 1e9;
        eprintln!(
            "round {round}: scattered {scattered_s:.3} s ({:.2} GB/s), packed {packed_s:.3} s ({:.2} GB/s), copy {copy_s:.3} s ({:.2} GB/s)",
            gbs(read_a, scattered_s), gbs(padded, packed_s), gbs(padded, copy_s)
        );
        rows.push(json!({
            "round": round, "scattered_seconds": scattered_s, "scattered_bytes_read": read_a,
            "packed_seconds": packed_s, "packed_bytes": padded, "copy_seconds": copy_s,
            "copy_check_xor": check,
        }));
    }
    let out = json!({
        "mode": "load", "checkpoint": dir, "domain": args[2], "head_experts": experts,
        "tensors": parts.len(), "head_bytes": total, "packed_file": packed, "rounds": rows,
    });
    write_json(&args[5], &out)
}

/// `typewriter trace <heads.json> <one-expert-bytes> <out.json> <domain>=<fit.json>,<held.json> ...`:
/// offline analyses from recorded routes (no model): H4 prefetch hit rates one token ahead and
/// task-switch detection delay; H6 growth of the expert union when N sequences share one beat.
fn trace_cmd(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err("usage: typewriter trace <heads.json> <one-expert-bytes> <out.json> <domain>=<fit.json>,<held.json> ...".into());
    }
    let heads = read_json(&args[0])?;
    let expert_b: f64 = args[1].parse().map_err(|_| "one-expert-bytes".to_string())?;
    let mut fit: BTreeMap<String, Trace> = BTreeMap::new();
    let mut held: BTreeMap<String, Trace> = BTreeMap::new();
    for spec in &args[3..] {
        let (name, files) = spec.split_once('=').ok_or("domain=fit,held")?;
        let (f, h) = files.split_once(',').ok_or("domain=fit,held")?;
        fit.insert(name.into(), trace(f)?);
        held.insert(name.into(), trace(h)?);
    }
    let names: Vec<String> = held.keys().cloned().collect();
    let t0 = &held[&names[0]];
    let (experts, k, layers) = (t0.experts, t0.top_k, t0.routes.len());
    let head_mask = |d: &str| -> Vec<Vec<bool>> {
        (0..layers)
            .map(|l| {
                let mut m = vec![false; experts];
                if let Some(a) = heads["heads"][d]["layers"][l].as_array() {
                    a.iter().filter_map(|x| x.as_u64()).for_each(|e| m[e as usize] = true);
                }
                m
            })
            .collect()
    };

    // H4a: predictors of token i+1's experts per layer, scored as hits / k, with the bytes each
    // predictor would have had in flight (the predicted set size).
    let mut h4 = serde_json::Map::new();
    for d in &names {
        let t = &held[d];
        let n = t.routes[0].len() / k;
        let mask = head_mask(d);
        let mut rows = serde_json::Map::new();
        for w in [1usize, 2, 4, 8, 16] {
            // "recent w": the union of the last w tokens' experts (a cache of the last w beats).
            let (mut hit, mut tot, mut set_sz) = (0usize, 0usize, 0usize);
            let (mut hit_h, mut set_h) = (0usize, 0usize);
            for l in 0..layers {
                for i in w..n {
                    let mut pred = vec![false; experts];
                    for j in i - w..i {
                        tok(t, k, l, j).iter().for_each(|&e| pred[e as usize] = true);
                    }
                    let sz = pred.iter().filter(|&&p| p).count();
                    let act = tok(t, k, l, i);
                    hit += act.iter().filter(|&&e| pred[e as usize]).count();
                    hit_h += act.iter().filter(|&&e| pred[e as usize] || mask[l][e as usize]).count();
                    set_h += (0..experts).filter(|&e| pred[e] || mask[l][e]).count();
                    tot += k;
                    set_sz += sz;
                }
            }
            let beats = (tot / k).max(1) as f64;
            rows.insert(format!("recent_{w}"), json!({
                "hit_rate": hit as f64 / tot as f64,
                "experts_in_flight_per_layer": set_sz as f64 / beats,
                "with_head_hit_rate": hit_h as f64 / tot as f64,
                "with_head_experts_per_layer": set_h as f64 / beats,
            }));
        }
        h4.insert(d.clone(), Value::Object(rows));
    }
    // H4b: task-switch detection. Stream domain A's held-out routes then domain B's; every token,
    // score each domain by the log-probability of the last `win` tokens' routed experts under
    // that domain's fit-text usage (per layer, add-one smoothed); the detector says the domain
    // with the highest score. Delay = tokens after the switch until it says B and keeps saying B
    // to the end of B's first 128 tokens. Fit and held-out texts are different documents.
    let logp: BTreeMap<String, Vec<Vec<f64>>> = names
        .iter()
        .map(|d| {
            let t = &fit[d];
            let tabs = (0..layers)
                .map(|l| {
                    let mut c = vec![1f64; experts];
                    t.routes[l].iter().for_each(|&e| c[e as usize] += 1.0);
                    let s: f64 = c.iter().sum();
                    c.iter().map(|v| (v / s).ln()).collect()
                })
                .collect();
            (d.clone(), tabs)
        })
        .collect();
    let share = |t: &Trace, i: usize, d: &str| -> f64 {
        let lp = &logp[d];
        (0..layers).map(|l| tok(t, k, l, i).iter().map(|&e| lp[l][e as usize]).sum::<f64>()).sum()
    };
    let mut h4b = serde_json::Map::new();
    for win in [1usize, 4, 8, 16, 32] {
        let mut delays = serde_json::Map::new();
        let mut correct = 0usize;
        let mut total = 0usize;
        for a in &names {
            for b in &names {
                if a == b {
                    continue;
                }
                let (ta, tb) = (&held[a], &held[b]);
                let na = (ta.routes[0].len() / k).min(256);
                let nb = (tb.routes[0].len() / k).min(128);
                let seq: Vec<(&Trace, usize)> = (na - win.min(na)..na).map(|i| (ta, i)).chain((0..nb).map(|i| (tb, i))).collect();
                let off = win.min(na);
                let mut says_b = Vec::with_capacity(nb);
                for p in off..seq.len() {
                    let lo = p + 1 - win.min(p + 1);
                    let best = names
                        .iter()
                        .map(|d| (d, seq[lo..=p].iter().map(|&(t, i)| share(t, i, d)).sum::<f64>()))
                        .max_by(|x, y| x.1.total_cmp(&y.1))
                        .map(|x| x.0.clone())
                        .unwrap_or_default();
                    says_b.push(best == *b);
                }
                let delay = (0..says_b.len()).find(|&s| says_b[s..].iter().all(|&x| x)).unwrap_or(says_b.len());
                delays.insert(format!("{a}->{b}"), json!(delay));
                correct += says_b[says_b.len() / 2..].iter().filter(|&&x| x).count();
                total += says_b.len() - says_b.len() / 2;
            }
        }
        h4b.insert(format!("window_{win}"), json!({"delay_tokens": Value::Object(delays), "steady_accuracy": correct as f64 / total.max(1) as f64}));
    }
    // H6: N independent sequences on one beat. Draw N token positions from the pooled held-out
    // routes (fixed seed, 64 draws per N); per layer the union of their experts is what one weight
    // stroke must read. Bytes per beat for the routed part = union x one expert.
    let pool: Vec<(&Trace, usize)> = names.iter().flat_map(|d| { let t = &held[d]; (0..t.routes[0].len() / k).map(move |i| (t, i)) }).collect();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut h6 = Vec::new();
    for nseq in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let mut uni = 0f64;
        let draws = 64;
        for _ in 0..draws {
            let mut picks = Vec::with_capacity(nseq);
            for _ in 0..nseq {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                picks.push(pool[(seed % pool.len() as u64) as usize]);
            }
            for l in 0..layers {
                let mut u = vec![false; experts];
                picks.iter().for_each(|&(t, i)| tok(t, k, l, i).iter().for_each(|&e| u[e as usize] = true));
                uni += u.iter().filter(|&&x| x).count() as f64;
            }
        }
        let per_layer = uni / (draws * layers) as f64;
        h6.push(json!({
            "sequences": nseq, "union_experts_per_layer": per_layer,
            "routed_bytes_per_beat": per_layer * layers as f64 * expert_b,
            "routed_bytes_per_token": per_layer * layers as f64 * expert_b / nseq as f64,
        }));
    }
    let out = json!({"mode": "trace", "experts": experts, "top_k": k, "layers": layers, "one_expert_bytes": expert_b,
        "h4_prefetch": Value::Object(h4), "h4_switch": Value::Object(h4b), "h6_union": h6});
    write_json(&args[2], &out)?;
    eprintln!("{}", serde_json::to_string_pretty(&out["h6_union"]).unwrap_or_default());
    Ok(())
}

/// Token `i`'s routed experts at layer `l`.
fn tok(t: &Trace, k: usize, l: usize, i: usize) -> &[u16] {
    &t.routes[l][i * k..(i + 1) * k]
}
