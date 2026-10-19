//! READ DRUM tool (sc #677): Qwen3.8-27B run from the NVMe in layer groups under a memory budget.
//!
//!   qwen38_drum build <checkpoint-dir> <drum>
//!   qwen38_drum verify <drum> [threads]
//!   qwen38_drum read <drum> <threads,...> <group-layers> <out.json>
//!   qwen38_drum reference <checkpoint-dir> <golden.json> <out-prefix>
//!   qwen38_drum run <checkpoint-dir> <drum> <golden.json> <budget-GiB> <group-layers> <batch>
//!                   <reference-prefix> <out.json> [--threads N] [--control flip|stale]
//!
//! `reference` runs the production whole-model upload ([`Qwen38Omega::open`]) teacher-forced on the
//! golden record and writes `<prefix>.json` + `<prefix>.logits` (f32 logits of every generated
//! position). `run` does the same from the drum under the budget, for `batch` conversations in
//! lockstep, and is PASS only when every conversation's tokens equal the golden record and its
//! logits equal the reference bit for bit, and the process's peak resident memory stayed within
//! the budget. With `--control` the run must come out FAIL (a check that cannot fail proves
//! nothing). Memory guard: refuses unless MemAvailable covers the need plus 12 GiB.
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38_omega::drum::{self, Control, DrumFile, DrumIndex, DrumModel};
use aien_inference_abi::qwen38_omega::Qwen38Omega;
use serde_json::{json, Value};

const KEEP_FREE: u64 = 12 << 30;
const GIB: f64 = (1u64 << 30) as f64;

fn mem_available() -> Result<u64, String> {
    let m = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    m.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb << 10)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

/// `VmHWM` / `VmRSS` of this process in bytes (omega's driver mappings count as file pages).
fn vm(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb << 10)
}

fn guard(need: u64, what: &str) -> Result<(), String> {
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "{what}: needs {:.1} GiB with 12 GiB kept free, MemAvailable {:.1} GiB; refusing",
            need as f64 / GIB,
            avail as f64 / GIB
        ));
    }
    Ok(())
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("golden: no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("golden: {key} holds a non-id"))
        })
        .collect()
}

/// (fed tokens, first generated step, golden generated ids).
fn golden(path: &str) -> Result<(Vec<u32>, usize, Vec<u32>), String> {
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
            .map_err(|e| format!("{path}: {e}"))?;
    let prompt = ids(&g, "prompt_ids")?;
    let gen = ids(&g, "generated_ids")?;
    if prompt.is_empty() || gen.is_empty() {
        return Err("golden: empty prompt or continuation".into());
    }
    let mut feed = prompt.clone();
    feed.extend_from_slice(&gen[..gen.len() - 1]);
    Ok((feed, prompt.len() - 1, gen))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn cmd_build(a: &[String]) -> Result<bool, String> {
    let [dir, out] = a else {
        return Err("usage: build <checkpoint-dir> <drum>".into());
    };
    let t = Instant::now();
    let ix = drum::build(Path::new(dir), Path::new(out))?;
    let bytes: u64 = ix.layer_bytes().iter().sum();
    println!(
        "DRUM_BUILD {} layers, {bytes} B, {:.1} s -> {} (+ .json)",
        ix.layers.len(),
        t.elapsed().as_secs_f64(),
        out
    );
    Ok(true)
}

fn cmd_verify(a: &[String]) -> Result<bool, String> {
    let (drum_p, threads) = match a {
        [d] => (d, 8),
        [d, t] => (d, t.parse().map_err(|_| "threads: a number")?),
        _ => return Err("usage: verify <drum> [threads]".into()),
    };
    let ix = DrumIndex::load(&drum::manifest_path(Path::new(drum_p)))?;
    let bad = drum::verify(Path::new(drum_p), &ix, threads)?;
    println!(
        "DRUM_VERIFY {} layers, {} differ from the manifest {:?} -> {}",
        ix.layers.len(),
        bad.len(),
        bad,
        if bad.is_empty() { "PASS" } else { "FAIL" }
    );
    Ok(bad.is_empty())
}

/// Streams the whole drum group by group (no GPU): NVMe bytes/s per thread count.
fn cmd_read(a: &[String]) -> Result<bool, String> {
    let [drum_p, threads, group, out] = a else {
        return Err("usage: read <drum> <threads,...> <group-layers> <out.json>".into());
    };
    let ix = DrumIndex::load(&drum::manifest_path(Path::new(drum_p)))?;
    let group: usize = group.parse().map_err(|_| "group: a number")?;
    let ranges: Vec<(u64, usize)> = ix
        .layers
        .chunks(group)
        .map(|c| {
            let s = c[0].start;
            let e = c[c.len() - 1].start + c[c.len() - 1].len;
            (s, (e - s) as usize)
        })
        .collect();
    let max = ranges.iter().map(|r| r.1).max().unwrap_or(0);
    let mut buf = drum::AlignedBuf::new(max);
    let total: u64 = ranges.iter().map(|r| r.1 as u64).sum();
    let mut rows = Vec::new();
    for t in threads.split(',') {
        let n: usize = t.parse().map_err(|_| "threads: numbers")?;
        let f = DrumFile::open(Path::new(drum_p), n)?;
        let t0 = Instant::now();
        let mut per = Vec::new();
        for &(s, l) in &ranges {
            let t1 = Instant::now();
            f.read(s, l, &mut buf)?;
            per.push(l as f64 / t1.elapsed().as_secs_f64() / 1e9);
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "DRUM_READ threads {n}: {total} B in {secs:.3} s = {:.3} GB/s (group median {:.3} GB/s)",
            total as f64 / secs / 1e9,
            median(per.clone())
        );
        rows.push(json!({"threads": n, "bytes": total, "secs": secs, "group_gbps": per}));
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&json!({"group_layers": group, "runs": rows})).unwrap(),
    )
    .map_err(|e| format!("{out}: {e}"))?;
    Ok(true)
}

fn cmd_reference(a: &[String]) -> Result<bool, String> {
    let [dir, gold, prefix] = a else {
        return Err("usage: reference <checkpoint-dir> <golden.json> <out-prefix>".into());
    };
    let (feed, first, gen) = golden(gold)?;
    let max_pos = feed.len().div_ceil(256) * 256;
    guard(31 << 30, "reference (whole model)")?;
    let t = Instant::now();
    let mut m = Qwen38Omega::open(Path::new(dir), None, max_pos)?;
    let upload_s = t.elapsed().as_secs_f64();
    let (mut toks, mut shas, mut ms, mut logits_out) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (p, &tok) in feed.iter().enumerate() {
        let t = Instant::now();
        let (next, _) = m.step(tok)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if p >= first {
            let l = m.logits()?;
            let b = f32_bytes(&l);
            shas.push(drum::sha256_hex(&b));
            logits_out.extend_from_slice(&b);
            toks.push(next);
        }
    }
    let identical = toks == gen;
    std::fs::write(format!("{prefix}.logits"), &logits_out).map_err(|e| e.to_string())?;
    let r = json!({"mode": "reference", "upload_s": upload_s, "positions": feed.len(), "first_generated_step": first,
        "tokens": toks, "golden": gen, "golden_identical": identical, "logits_sha256": shas, "step_ms": ms,
        "vm_hwm": vm("VmHWM:"), "weight_bytes": m.weight_bytes()});
    std::fs::write(
        format!("{prefix}.json"),
        serde_json::to_string_pretty(&r).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "DRUM_REFERENCE {} positions, generated {}/{} equal to golden, median step {:.2} ms, VmHWM {:.2} GiB -> {}",
        feed.len(),
        toks.iter().zip(&gen).filter(|(a, b)| a == b).count(),
        gen.len(),
        median(ms),
        vm("VmHWM:") as f64 / GIB,
        if identical { "PASS" } else { "FAIL" }
    );
    Ok(identical)
}

fn cmd_run(a: &[String]) -> Result<bool, String> {
    let (pos, opts): (Vec<&String>, Vec<&String>) = {
        let mut p = Vec::new();
        let mut o = Vec::new();
        let mut it = a.iter();
        while let Some(x) = it.next() {
            if x.starts_with("--") {
                o.push(x);
                o.push(it.next().ok_or("an option without a value")?);
            } else {
                p.push(x);
            }
        }
        (p, o)
    };
    let [dir, drum_p, gold, budget, group, batch, refp, out] = pos[..] else {
        return Err("usage: run <checkpoint-dir> <drum> <golden.json> <budget-GiB> <group-layers> <batch> <reference-prefix> <out.json> [--threads N] [--control flip|stale]".into());
    };
    let (mut threads, mut control) = (8usize, Control::None);
    for kv in opts.chunks(2) {
        match (kv[0].as_str(), kv[1].as_str()) {
            ("--threads", n) => threads = n.parse().map_err(|_| "--threads: a number")?,
            ("--control", "flip") => control = Control::FlipByte,
            ("--control", "stale") => control = Control::StaleSlot,
            (k, v) => return Err(format!("unknown option {k} {v}")),
        }
    }
    let budget_gib: f64 = budget.parse().map_err(|_| "budget: GiB")?;
    let budget = (budget_gib * GIB) as u64;
    let group: usize = group.parse().map_err(|_| "group: a number")?;
    let batch: usize = batch.parse().map_err(|_| "batch: a number")?;
    let (feed, first, gen) = golden(gold)?;
    let max_pos = feed.len().div_ceil(256) * 256;
    let rf: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{refp}.json"))
            .map_err(|e| format!("{refp}.json: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let ref_logits =
        std::fs::read(format!("{refp}.logits")).map_err(|e| format!("{refp}.logits: {e}"))?;
    guard(budget, "run")?;

    let t = Instant::now();
    let mut m = DrumModel::open(
        Path::new(dir),
        Path::new(drum_p),
        budget,
        group,
        batch,
        max_pos,
        1 << 30,
        threads,
        control,
    )?;
    let open_s = t.elapsed().as_secs_f64();
    let plan = m.plan.clone();
    println!(
        "DRUM_PLAN budget {:.2} GiB: {} groups of {group}, {} pinned ({:.2} GiB), slot {:.2} GiB, streamed {:.2} GiB/pass, fixed {:.2} GiB, planned peak {:.2} GiB; opened in {open_s:.1} s, VmRSS {:.2} GiB",
        budget as f64 / GIB,
        plan.groups.len(),
        plan.pinned,
        plan.pinned_bytes as f64 / GIB,
        plan.slot_bytes as f64 / GIB,
        plan.streamed_bytes as f64 / GIB,
        plan.fixed as f64 / GIB,
        plan.peak as f64 / GIB,
        vm("VmRSS:") as f64 / GIB
    );
    let vocab = m.config.vocab;
    let (mut toks, mut passes) = (vec![Vec::new(); batch], Vec::new());
    let (mut equal_logits, mut max_abs) = (vec![0usize; batch], 0f32);
    for (p, &tok) in feed.iter().enumerate() {
        let want = p >= first;
        let (next, logits, st) = m.step(&vec![tok; batch], want)?;
        if want {
            let i = p - first;
            let r = &ref_logits[i * vocab * 4..(i + 1) * vocab * 4];
            for c in 0..batch {
                toks[c].push(next[c]);
                let b = f32_bytes(&logits[c]);
                if b == r {
                    equal_logits[c] += 1;
                } else {
                    for (x, y) in logits[c].iter().zip(r.as_chunks::<4>().0) {
                        let y = f32::from_le_bytes(*y);
                        max_abs = max_abs.max((x - y).abs());
                    }
                }
            }
        }
        println!(
            "pass {p:3}: wall {:8.1} ms  wait {:7.1}  upload {:7.1}  compute {:7.1}  read {:7.1} ms {:.2} GB",
            st.wall_ns as f64 / 1e6,
            st.wait_ns as f64 / 1e6,
            st.upload_ns as f64 / 1e6,
            st.compute_ns as f64 / 1e6,
            st.read_ns as f64 / 1e6,
            st.read_bytes as f64 / 1e9
        );
        passes.push(st);
    }
    let hwm = vm("VmHWM:");
    let n_gen = gen.len();
    let tokens_ok = toks.iter().all(|t| *t == gen);
    let logits_ok = equal_logits.iter().all(|&e| e == n_gen);
    let within = hwm <= budget;
    let ref_ok = rf["golden_identical"].as_bool() == Some(true);
    let pass = tokens_ok && logits_ok && within && ref_ok;
    // Timing from pass 2 on (the first passes generate kernels).
    let warm = &passes[2.min(passes.len() - 1)..];
    let med = |f: &dyn Fn(&drum::PassStats) -> u64| {
        median(warm.iter().map(|s| f(s) as f64 / 1e6).collect())
    };
    let (wall, wait, upload, compute, read) = (
        med(&|s| s.wall_ns),
        med(&|s| s.wait_ns),
        med(&|s| s.upload_ns),
        med(&|s| s.compute_ns),
        med(&|s| s.read_ns),
    );
    let read_gbps = median(
        warm.iter()
            .filter(|s| s.read_ns > 0)
            .map(|s| s.read_bytes as f64 / s.read_ns as f64)
            .collect(),
    );
    println!(
        "DRUM_RUN budget {:.2} GiB group {group} batch {batch} control {control:?}: tokens {} logits bit-identical {}/{} per conversation (max |diff| {max_abs}), VmHWM {:.2} GiB {} budget; median pass {wall:.1} ms (wait {wait:.1}, upload {upload:.1}, compute {compute:.1}, reader {read:.1} ms at {read_gbps:.2} GB/s) = {:.3} tok/s per conversation, {:.3} tok/s total -> {}",
        budget as f64 / GIB,
        if tokens_ok { "== golden" } else { "DIFFER" },
        equal_logits.iter().min().unwrap_or(&0),
        n_gen,
        hwm as f64 / GIB,
        if within { "<=" } else { ">" },
        1e3 / wall,
        batch as f64 * 1e3 / wall,
        if pass { "PASS" } else { "FAIL" }
    );
    let r = json!({"mode": "run", "budget": budget, "group": group, "batch": batch, "threads": threads,
        "control": format!("{control:?}"), "plan": plan, "open_s": open_s, "positions": feed.len(),
        "tokens": toks, "golden": gen, "tokens_identical": tokens_ok, "logits_equal_per_conversation": equal_logits,
        "logits_bit_identical": logits_ok, "max_abs_diff": max_abs, "reference_golden_identical": ref_ok,
        "vm_hwm": hwm, "within_budget": within,
        "median_ms": {"wall": wall, "wait": wait, "upload": upload, "compute": compute, "reader": read},
        "reader_gbps_median": read_gbps, "passes": passes, "pass": pass});
    std::fs::write(out, serde_json::to_string_pretty(&r).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    Ok(pass)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(String::as_str) {
        Some("build") => cmd_build(&a[1..]),
        Some("verify") => cmd_verify(&a[1..]),
        Some("read") => cmd_read(&a[1..]),
        Some("reference") => cmd_reference(&a[1..]),
        Some("run") => cmd_run(&a[1..]),
        _ => Err("usage: qwen38_drum build|verify|read|reference|run ...".into()),
    };
    match r {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_drum: {e}");
            std::process::exit(2);
        }
    }
}
