//! `engine`: the L312 command line. Every subcommand writes its receipt as JSON next to its outputs.
//!
//!   engine manifest MODEL_DIR OUT.json                       hash every expert once (heartbeat Manifest::build)
//!   engine trace MODEL_DIR MANIFEST OUT.json --prompt T --gen N [--slots S]
//!                                                           teacher-forced host pass (ring-backed, L190 code): routes per
//!                                                           layer + argmax per position = the identity gate's reference
//!   engine plan trace TRACE MODEL_DIR OUT.json [--buffers B] synthetic plan: head(layer) = union of routed experts
//!   engine plan union MODEL_DIR OUT.json [--buffers B] TRACE...   same over several traces (N heads on one stroke)
//!   engine plan fixed MODEL_DIR OUT.json --layers K --experts E [--buffers B]
//!   engine schedule PLAN MANIFEST TOKENS OUT.json            build + host proof (complete, acyclic, no collision, edges)
//!   engine move SCHEDULE MANIFEST OUT_DIR [--workers W] [--fire-us U] [--direct 0|1] [--engines 1|2]
//!               [--coupling brigade|clock|reservoir] [--period-us T]
//!                                                           movement engines alone: a stand-in piston that only holds U us
//!   engine run SCHEDULE MANIFEST TRACE MODEL_DIR OUT_DIR [--workers W] [--direct 0|1] [--engines 1|2]
//!                                                           both engines on the host: identical tokens gate + timing diagram
//!   engine stroke SCHEDULE MANIFEST MODEL_DIR OUT_DIR --n N --traces A.json,B.json,... [--positions P] [--workers W] [--engines E]
//!                                                           H6: N heads on one stroke; gate per head vs its own trace
//!   engine union OUT.json [--positions P] TRACE...          H6 offline: expert union vs N from the traces
//!   engine today URL MODEL_DIR TRACE OUT.json               today's path (the live server) on the same prompt: agreement count
use aien_engine::movement::{Engines, FillStamp, MovementEngine};
use aien_engine::piston::{argmax, gate_identical, Crank, HostPiston, HostPistonBank, StrokeBeat};
use aien_engine::plan::{PlacementPlan, RouteTrace};
use aien_engine::schedule::Schedule;
use aien_engine::timing::{png_diagram, series, summarize, text_diagram, BeatTrace};
use aien_engine::EXPERTS;
use aien_heartbeat::manifest::Manifest;
use aien_wormhole::moe30::Moe30Ring;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

fn opt(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn opt_usize(args: &[String], key: &str, default: usize) -> usize {
    opt(args, key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Positional arguments: everything that is not an `--option VALUE` pair.
fn positionals(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += 2;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    out
}

const TEMPLATE: (&str, &str) = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

fn tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, String> {
    tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| e.to_string())
}

fn write_json(p: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", p.display()))
}

fn cmd_manifest(a: &[String]) -> Result<(), String> {
    let t = Instant::now();
    let m = Manifest::build(Path::new(&a[0])).map_err(|e| e.to_string())?;
    m.save(Path::new(&a[1])).map_err(|e| e.to_string())?;
    println!("manifest: {} experts, {} bytes, {:.1} s -> {}", m.experts.len(), m.total_bytes(), t.elapsed().as_secs_f64(), a[1]);
    Ok(())
}

fn cmd_trace(a: &[String]) -> Result<(), String> {
    let (dir, mpath, out) = (Path::new(&a[0]), Path::new(&a[1]), Path::new(&a[2]));
    let prompt = opt(a, "--prompt").ok_or("--prompt TEXT")?;
    let gen = opt_usize(a, "--gen", 8);
    let slots = opt_usize(a, "--slots", 512);
    let tk = tokenizer(dir)?;
    let text = format!("{}{}{}", TEMPLATE.0, prompt, TEMPLATE.1);
    let ids: Vec<u32> = tk.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let m = Manifest::load(mpath).map_err(|e| e.to_string())?;
    let mut lm = Moe30Ring::open(dir, m, slots, 8, true)?;
    let mut tr = RouteTrace { schema: "aien-engine-trace/1".into(), model: dir.file_name().unwrap().to_string_lossy().into(), prompt_tokens: ids.clone(), ..Default::default() };
    let t = Instant::now();
    let mut seq = ids.clone();
    let mut i = 0;
    while i < seq.len() {
        let logits = lm.step(seq[i])?;
        let next = argmax(&logits);
        tr.argmax.push(next);
        tr.routes.push(lm.last_routes.clone());
        i += 1;
        if i == seq.len() && seq.len() < ids.len() + gen {
            seq.push(next);
        }
    }
    tr.host_ns_per_token = (t.elapsed().as_nanos() / seq.len() as u128) as u64;
    let gen_ids = &seq[ids.len()..];
    let gen_text = tk.decode(gen_ids, true).map_err(|e| e.to_string())?;
    tr.save(out)?;
    let side = json!({"prompt": prompt, "prompt_tokens": ids.len(), "generated": gen_ids, "generated_text": gen_text, "positions": seq.len(),
        "host_ms_per_token": tr.host_ns_per_token as f64 / 1e6, "ring_slots": slots, "cache": format!("{:?}", lm.cache.stats)});
    write_json(&out.with_extension("side.json"), &side)?;
    println!("trace: {} positions, generated {:?}", seq.len(), gen_text);
    Ok(())
}

fn cmd_plan(a: &[String]) -> Result<(), String> {
    let buffers = opt_usize(a, "--buffers", 2);
    let pos = positionals(a);
    let (plan, out) = match pos[0].as_str() {
        "trace" => {
            let tr = RouteTrace::load(Path::new(&pos[1]))?;
            let dir = Path::new(&pos[2]);
            let k = tr.routes.first().map(|r| r.len()).ok_or("empty trace")?;
            (PlacementPlan::synthetic_from_trace(&tr.model, dir, &tr, k, buffers), pos[3].clone())
        }
        "union" => {
            let dir = Path::new(&pos[1]);
            let traces: Vec<RouteTrace> = pos[3..].iter().map(|p| RouteTrace::load(Path::new(p))).collect::<Result<_, _>>()?;
            let k = traces.first().and_then(|t| t.routes.first()).map(|r| r.len()).ok_or("empty traces")?;
            (PlacementPlan::synthetic_from_traces(&traces[0].model, dir, &traces, k, buffers), pos[2].clone())
        }
        "fixed" => {
            let dir = Path::new(&pos[1]);
            (PlacementPlan::synthetic_fixed(&dir.file_name().unwrap().to_string_lossy(), dir, opt_usize(a, "--layers", 4), opt_usize(a, "--experts", 8) as u16, buffers), pos[2].clone())
        }
        x => return Err(format!("plan {x}: trace|union|fixed")),
    };
    plan.save(Path::new(&out))?;
    let sizes: Vec<usize> = plan.heads.iter().map(|h| h.experts.len()).collect();
    println!("plan: {} layers, head sizes {:?} (min {} max {}), buffers {}, synthetic {}", plan.layers.len(), sizes, sizes.iter().min().unwrap(), sizes.iter().max().unwrap(), plan.buffers_in_flight, plan.synthetic);
    Ok(())
}

fn cmd_schedule(a: &[String]) -> Result<(), String> {
    let plan = PlacementPlan::load(Path::new(&a[0]))?;
    let m = Manifest::load(Path::new(&a[1])).map_err(|e| e.to_string())?;
    let tokens: u32 = a[2].parse().map_err(|_| "TOKENS")?;
    let s = Schedule::build(&plan, &m, tokens)?;
    let r = s.check();
    let edges = s.edges();
    let expected = s.expected_edges();
    let edges_ok = edges == expected;
    s.save(Path::new(&a[3]))?;
    let mut rep = serde_json::to_value(&r).unwrap();
    rep["edges"] = serde_json::to_value(&edges).unwrap();
    rep["edges_expected"] = serde_json::to_value(&expected).unwrap();
    rep["edges_exact"] = json!(edges_ok);
    write_json(&Path::new(&a[3]).with_extension("check.json"), &rep)?;
    println!("schedule: {} beats, {} parts, {} buffers x {} slots x {} bytes; check {}: complete {} acyclic {} no_collision {} edges_exact {} (fuel {} residual {} token {} release {} free_parts {}) bytes/beat {}..{} total {}",
        s.beats.len(), s.parts.len(), s.buffers, s.slots, s.slot_bytes, if r.pass() && edges_ok { "PASS" } else { "FAIL" }, r.complete, r.acyclic, r.no_collision, edges_ok, edges.fuel, edges.residual, edges.token, edges.release, edges.free_parts, r.bytes_per_beat_min, r.bytes_per_beat_max, r.bytes_total);
    for p in r.problems.iter().chain(edges.violations.iter()) {
        println!("  problem: {p}");
    }
    if r.pass() && edges_ok { Ok(()) } else { Err("schedule check FAIL".into()) }
}

fn engines_of(a: &[String]) -> Result<Engines, String> {
    Engines::parse(&opt(a, "--engines").unwrap_or_else(|| "1".into()))
}

fn start_engines(s: Arc<Schedule>, m: &Manifest, workers: usize, direct: bool, engines: Engines, t0: Instant) -> Result<(Crank, std::thread::JoinHandle<()>), String> {
    let me = MovementEngine::open(s.clone(), m, workers, direct, engines, t0)?;
    let ring = me.ring();
    let (jtx, jrx) = mpsc::channel();
    let (stx, srx) = mpsc::channel();
    let h = std::thread::spawn(move || me.serve(jrx, stx));
    Ok((Crank::new(s, ring, jtx, srx, t0), h))
}

/// Per-stage series from the fill stamps: NVMe engine time, RAM engine time, the hand-off wait between them.
fn stage_series(stamps: &[FillStamp]) -> Value {
    let us = |x: i64| x as f64 / 1e3;
    let nvme: Vec<f64> = stamps.iter().map(|s| us(s.nvme_done_ns - s.issue_ns)).collect();
    let ram: Vec<f64> = stamps.iter().map(|s| us(s.done_ns - s.ram_start_ns)).collect();
    let handoff: Vec<f64> = stamps.iter().map(|s| us(s.ram_start_ns - s.nvme_done_ns)).collect();
    let digest: Vec<f64> = stamps.iter().map(|s| s.digest_ns_sum as f64 / 1e3).collect();
    json!({"nvme_us": series(&nvme), "ram_us": series(&ram), "handoff_wait_us": series(&handoff), "digest_us_sum": series(&digest)})
}

fn write_receipts(dir: &Path, traces: &[BeatTrace], extra: Value) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("beats.json"), serde_json::to_vec(traces).unwrap()).map_err(|e| e.to_string())?;
    let sum = summarize(traces);
    let mut v = serde_json::to_value(&sum).unwrap();
    if let (Value::Object(o), Value::Object(e)) = (&mut v, extra) {
        o.extend(e);
    }
    write_json(&dir.join("summary.json"), &v)?;
    std::fs::write(dir.join("diagram.txt"), text_diagram(traces, 100, 120)).map_err(|e| e.to_string())?;
    png_diagram(traces, &dir.join("diagram.png"))?;
    println!("{}", serde_json::to_string(&v).unwrap());
    Ok(())
}

fn spin_until(crank: &Crank, until_ns: i64) {
    while crank.now_ns() < until_ns {
        std::hint::spin_loop();
    }
}

fn cmd_move(a: &[String]) -> Result<(), String> {
    let s = Arc::new(Schedule::load(Path::new(&a[0]))?);
    let m = Manifest::load(Path::new(&a[1])).map_err(|e| e.to_string())?;
    let out = PathBuf::from(&a[2]);
    let workers = opt_usize(a, "--workers", 8);
    let fire_us = opt_usize(a, "--fire-us", opt_usize(a, "--beat-us", 0)) as i64;
    let direct = opt_usize(a, "--direct", 1) == 1;
    let engines = engines_of(a)?;
    let coupling = opt(a, "--coupling").unwrap_or_else(|| "brigade".into());
    let period_us = opt_usize(a, "--period-us", 10_000) as i64;
    if !s.check().pass() {
        return Err("schedule check FAIL".into());
    }
    let t0 = Instant::now();
    let (mut crank, h) = start_engines(s.clone(), &m, workers, direct, engines, t0)?;
    let mut traces = Vec::with_capacity(s.beats.len());
    let mut stamps = Vec::with_capacity(s.beats.len());
    // reservoir: the movement engines run as far ahead as free buffers allow; the piston only reports buffers free
    let (free_tx, free_rx) = mpsc::channel::<u32>();
    let issuer = if coupling == "reservoir" {
        let (jobs, s2) = (crank.issuer(), s.clone());
        Some(std::thread::spawn(move || {
            let mut free = vec![1u32; s2.buffers as usize];
            let fills: Vec<_> = s2.prime.iter().cloned().chain(s2.beats.iter().filter_map(|b| b.fill.clone())).collect();
            for f in fills {
                while free[f.buffer as usize] == 0 {
                    match free_rx.recv() {
                        Ok(u) => free[u as usize] += 1,
                        Err(_) => return,
                    }
                }
                free[f.buffer as usize] -= 1;
                if jobs.send(f).is_err() {
                    return;
                }
            }
        }))
    } else {
        crank.prime()?;
        None
    };
    let t_start = crank.now_ns();
    for beat in &s.beats {
        if coupling == "clock" {
            spin_until(&crank, t_start + beat.b as i64 * period_us * 1000);
        }
        if coupling != "reservoir" {
            if let Some(f) = &beat.fill {
                crank.issue(f)?;
            }
        }
        let stamp = crank.wait_fill(beat.b)?;
        let fire_start = crank.now_ns();
        if fire_us > 0 {
            spin_until(&crank, fire_start + fire_us * 1000);
        }
        let fire_done = crank.now_ns();
        if coupling == "reservoir" {
            let _ = free_tx.send(beat.fire_buffer);
        }
        traces.push(BeatTrace { b: beat.b, token: beat.token, layer: beat.layer, buffer: beat.fire_buffer, bytes: stamp.bytes, fill_issue_ns: stamp.issue_ns, fill_done_ns: stamp.done_ns, fire_start_ns: fire_start, fire_done_ns: fire_done, misfire: false });
        stamps.push(stamp);
    }
    drop(free_tx);
    drop(crank);
    if let Some(i) = issuer {
        let _ = i.join();
    }
    h.join().map_err(|_| "movement thread panicked")?;
    write_receipts(&out, &traces, json!({"mode": "movement-only", "model": s.model, "workers": workers, "direct": direct, "fire_us_setting": fire_us, "engines": engines.count(), "coupling": coupling, "period_us_setting": if coupling == "clock" { period_us } else { 0 }, "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(), "tokens": s.tokens, "stages": stage_series(&stamps)}))
}

fn replay_sequence(tr: &RouteTrace) -> Vec<u32> {
    tr.prompt_tokens.iter().copied().chain(tr.argmax[tr.prompt_tokens.len() - 1..tr.argmax.len() - 1].iter().copied()).collect()
}

fn cmd_run(a: &[String]) -> Result<(), String> {
    let s = Arc::new(Schedule::load(Path::new(&a[0]))?);
    let m = Manifest::load(Path::new(&a[1])).map_err(|e| e.to_string())?;
    let tr = RouteTrace::load(Path::new(&a[2]))?;
    let dir = Path::new(&a[3]);
    let out = PathBuf::from(&a[4]);
    let workers = opt_usize(a, "--workers", 8);
    let direct = opt_usize(a, "--direct", 1) == 1;
    let engines = engines_of(a)?;
    if !s.check().pass() {
        return Err("schedule check FAIL".into());
    }
    if s.layers.len() != tr.routes.first().map(|r| r.len()).unwrap_or(0) {
        return Err(format!("schedule has {} layers per token, the trace {}: the host piston fires every layer of the model", s.layers.len(), tr.routes.first().map(|r| r.len()).unwrap_or(0)));
    }
    let seq = replay_sequence(&tr);
    let n = (s.tokens as usize).min(seq.len());
    let mut piston = HostPiston::open(dir)?;
    let t0 = Instant::now();
    let (mut crank, h) = start_engines(s.clone(), &m, workers, direct, engines, t0)?;
    crank.prime()?;
    let mut traces = Vec::new();
    let mut got = Vec::new();
    let mut routes_seen: Vec<Vec<Vec<u16>>> = Vec::new();
    let k = s.layers.len() as u32;
    for (t, &tok) in seq.iter().take(n).enumerate() {
        let mut routes = Vec::new();
        let (logits, bt) = piston.run_token(&mut crank, tok, t as u32 * k, Some(&mut routes))?;
        got.push(argmax(&logits));
        traces.extend(bt);
        routes_seen.push(routes);
    }
    drop(crank);
    h.join().map_err(|_| "movement thread panicked")?;
    let (same, total) = gate_identical(&tr, &got);
    let routes_same = routes_seen.iter().zip(&tr.routes).filter(|(a, b)| a == b).count();
    let gate = json!({"identical_tokens": same == total && total > 0, "same": same, "of": total, "routes_identical_positions": routes_same, "got": got, "expected": &tr.argmax[..total]});
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    write_json(&out.join("gate.json"), &gate)?;
    write_receipts(&out, &traces, json!({"mode": "host-loop", "model": s.model, "workers": workers, "direct": direct, "engines": engines.count(), "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(), "tokens": n, "gate": gate}))?;
    if same != total {
        return Err(format!("identity gate FAIL: {same} of {total}"));
    }
    println!("identity gate PASS: {same}/{total} tokens, routes identical at {routes_same}/{total} positions");
    Ok(())
}

fn cmd_stroke(a: &[String]) -> Result<(), String> {
    let pos = positionals(a);
    let s = Arc::new(Schedule::load(Path::new(&pos[0]))?);
    let m = Manifest::load(Path::new(&pos[1])).map_err(|e| e.to_string())?;
    let dir = Path::new(&pos[2]);
    let out = PathBuf::from(&pos[3]);
    let n = opt_usize(a, "--n", 1);
    let workers = opt_usize(a, "--workers", 8);
    let engines = engines_of(a)?;
    let traces: Vec<RouteTrace> = opt(a, "--traces").ok_or("--traces A.json,B.json,...")?.split(',').map(|p| RouteTrace::load(Path::new(p))).collect::<Result<_, _>>()?;
    if traces.len() < n {
        return Err(format!("{} traces for {n} heads", traces.len()));
    }
    let traces = &traces[..n];
    if !s.check().pass() {
        return Err("schedule check FAIL".into());
    }
    let seqs: Vec<Vec<u32>> = traces.iter().map(replay_sequence).collect();
    let p = opt_usize(a, "--positions", usize::MAX).min(seqs.iter().map(|q| q.len()).min().unwrap_or(0)).min(s.tokens as usize);
    if p == 0 {
        return Err("no positions".into());
    }
    let mut bank = HostPistonBank::open(dir, n)?;
    let t0 = Instant::now();
    let (mut crank, h) = start_engines(s.clone(), &m, workers, true, engines, t0)?;
    crank.prime()?;
    let k = s.layers.len() as u32;
    let mut traces_out: Vec<BeatTrace> = Vec::new();
    let mut strokes: Vec<StrokeBeat> = Vec::new();
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut routes_same = vec![0usize; n];
    for t in 0..p {
        let toks: Vec<u32> = seqs.iter().map(|q| q[t]).collect();
        let mut routes: Vec<Vec<Vec<u16>>> = Vec::new();
        let (logits, bt, sb) = bank.run_stroke(&mut crank, &toks, t as u32 * k, Some(&mut routes))?;
        for sidx in 0..n {
            got[sidx].push(argmax(&logits[sidx]));
            if routes[sidx] == traces[sidx].routes[t] {
                routes_same[sidx] += 1;
            }
        }
        traces_out.extend(bt);
        strokes.extend(sb);
    }
    drop(crank);
    h.join().map_err(|_| "movement thread panicked")?;
    let per_head: Vec<Value> = (0..n).map(|sidx| {
        let (same, total) = gate_identical(&traces[sidx], &got[sidx]);
        json!({"head": sidx, "same": same, "of": total, "routes_identical_positions": routes_same[sidx], "identical": same == total})
    }).collect();
    let all_identical = per_head.iter().all(|h| h["identical"].as_bool() == Some(true));
    let union: Vec<f64> = strokes.iter().map(|x| x.union as f64).collect();
    let union_bytes: Vec<f64> = strokes.iter().map(|x| x.union_bytes as f64).collect();
    let attn: Vec<f64> = strokes.iter().map(|x| x.attn_ns as f64 / 1e3).collect();
    let moe: Vec<f64> = strokes.iter().map(|x| x.moe_ns as f64 / 1e3).collect();
    let span_s = traces_out.last().map(|l| (l.fire_done_ns - traces_out[0].fire_start_ns) as f64 / 1e9).unwrap_or(0.0);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("strokes.json"), serde_json::to_vec(&strokes).unwrap()).map_err(|e| e.to_string())?;
    let gate = json!({"identical_tokens_every_head": all_identical, "heads": per_head});
    write_json(&out.join("gate.json"), &gate)?;
    write_receipts(&out, &traces_out, json!({"mode": "stroke", "model": s.model, "heads": n, "tokens_per_stroke": n, "strokes": p, "workers": workers, "engines": engines.count(), "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(),
        "union_experts_per_beat": series(&union), "union_bytes_per_beat": series(&union_bytes), "attn_us": series(&attn), "moe_us": series(&moe),
        "tokens_per_s_total": if span_s > 0.0 { (n * p) as f64 / span_s } else { 0.0 }, "gate": gate}))?;
    if !all_identical {
        return Err("stroke identity gate FAIL (see gate.json)".into());
    }
    println!("stroke identity gate PASS: {n} heads x {p} strokes, every head equals its own trace");
    Ok(())
}

fn cmd_union(a: &[String]) -> Result<(), String> {
    let pos = positionals(a);
    let out = Path::new(&pos[0]);
    let traces: Vec<RouteTrace> = pos[1..].iter().map(|p| RouteTrace::load(Path::new(p))).collect::<Result<_, _>>()?;
    let p = opt_usize(a, "--positions", usize::MAX).min(traces.iter().map(|t| t.routes.len()).min().unwrap_or(0));
    let k = traces[0].routes[0].len();
    let top_k = traces[0].routes[0][0].len();
    let mut rows = Vec::new();
    let mut n = 1usize;
    while n <= traces.len() {
        let mut unions = Vec::with_capacity(p * k);
        for t in 0..p {
            for l in 0..k {
                let mut set: Vec<u16> = Vec::new();
                for tr in &traces[..n] {
                    for &e in &tr.routes[t][l] {
                        if !set.contains(&e) {
                            set.push(e);
                        }
                    }
                }
                unions.push(set.len() as f64);
            }
        }
        let indep = EXPERTS as f64 * (1.0 - (1.0 - top_k as f64 / EXPERTS as f64).powi(n as i32));
        let measured = series(&unions);
        rows.push(json!({"n": n, "measured_union": measured, "independent_draw_expectation": indep, "below_independent": measured.mean < indep, "bytes_per_stroke_routed_mean": measured.mean * 4_719_168.0}));
        n *= 2;
    }
    let rec = json!({"traces": traces.len(), "positions": p, "layers": k, "top_k": top_k, "experts": EXPERTS, "formula": "experts * (1 - (1 - top_k/experts)^N)", "rows": rows});
    write_json(out, &rec)?;
    println!("{}", serde_json::to_string_pretty(&rec["rows"]).unwrap());
    Ok(())
}

fn http(url: &str, method: &str, body: &str) -> Result<String, String> {
    let u = url.strip_prefix("http://").ok_or("http:// URL only")?;
    let (host, path) = u.split_once('/').map(|(h, p)| (h, format!("/{p}"))).unwrap_or((u, "/".into()));
    let mut s = std::net::TcpStream::connect(host).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(600))).ok();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).map_err(|e| e.to_string())?;
    let mut resp = String::new();
    s.read_to_string(&mut resp).map_err(|e| e.to_string())?;
    let (_, b) = resp.split_once("\r\n\r\n").ok_or("bad HTTP response")?;
    Ok(b.to_string())
}

fn parse_body(resp: &str) -> Result<Value, String> {
    serde_json::from_str(resp.trim_start_matches(|c: char| c.is_ascii_hexdigit() || c == '\r' || c == '\n')).or_else(|_| serde_json::from_str(resp)).map_err(|e| format!("{e}: {}", &resp[..resp.len().min(200)]))
}

fn cmd_today(a: &[String]) -> Result<(), String> {
    let (url, dir, tr, out) = (&a[0], Path::new(&a[1]), RouteTrace::load(Path::new(&a[2]))?, Path::new(&a[3]));
    let tk = tokenizer(dir)?;
    let prompt_full = tk.decode(&tr.prompt_tokens, false).map_err(|e| e.to_string())?;
    let prompt = prompt_full.strip_prefix(TEMPLATE.0).and_then(|p| p.strip_suffix(TEMPLATE.1)).ok_or("trace prompt is not in the chat template")?;
    let base = url.trim_end_matches('/').trim_end_matches("/v1/chat/completions");
    let models = parse_body(&http(&format!("{base}/v1/models"), "GET", "")?)?;
    let model = opt(a, "--model").or_else(|| models["data"][0]["id"].as_str().map(String::from)).ok_or("no model id from /v1/models; pass --model")?;
    let gen: Vec<u32> = tr.argmax[tr.prompt_tokens.len() - 1..].to_vec();
    let body = json!({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen.len(), "temperature": 0}).to_string();
    let t = Instant::now();
    let resp = http(&format!("{base}/v1/chat/completions"), "POST", &body)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let v = parse_body(&resp)?;
    let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    let ids: Vec<u32> = tk.encode(text.clone(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let same = ids.iter().zip(&gen).take_while(|(a, b)| a == b).count();
    let host_text = tk.decode(&gen, true).map_err(|e| e.to_string())?;
    let rec = json!({"url": url, "model": model, "prompt": prompt, "today_text": text, "today_ids": ids, "host_text": host_text, "host_ids": gen, "same_prefix_ids": same, "of": gen.len(), "today_ms": ms,
        "error": if text.is_empty() { Some(resp.chars().take(300).collect::<String>()) } else { None },
        "note": "agreement (longest common prefix) between today's chip path and the host trace; the loop's own gate is host loop == host trace"});
    write_json(out, &rec)?;
    println!("today: model {model}, {same}/{} ids agree, {:.0} ms, text {:?}", gen.len(), ms, text.chars().take(80).collect::<String>());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("manifest") => cmd_manifest(&args[1..]),
        Some("trace") => cmd_trace(&args[1..]),
        Some("plan") => cmd_plan(&args[1..]),
        Some("schedule") => cmd_schedule(&args[1..]),
        Some("move") => cmd_move(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("stroke") => cmd_stroke(&args[1..]),
        Some("union") => cmd_union(&args[1..]),
        Some("today") => cmd_today(&args[1..]),
        _ => Err("engine manifest|trace|plan|schedule|move|run|stroke|union|today (see the file header)".to_string()),
    };
    if let Err(e) = r {
        eprintln!("engine: {e}");
        std::process::exit(1);
    }
}
