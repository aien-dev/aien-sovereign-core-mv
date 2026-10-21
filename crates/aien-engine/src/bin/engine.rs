//! `engine`: the L312 command line. Every subcommand writes its receipt as JSON next to its outputs.
//!
//!   engine manifest MODEL_DIR OUT.json                       hash every expert once (heartbeat Manifest::build)
//!   engine trace MODEL_DIR MANIFEST OUT.json --prompt T --gen N [--slots S]
//!                                                           teacher-forced host pass (ring-backed, L190 code): routes per
//!                                                           layer + argmax per position = the identity gate's reference
//!   engine plan trace TRACE MODEL_DIR OUT.json [--buffers B] synthetic plan: head(layer) = union of routed experts
//!   engine plan union MODEL_DIR OUT.json [--buffers B] TRACE...   same over several traces (N heads on one stroke)
//!   engine plan fixed MODEL_DIR OUT.json --layers K --experts E [--buffers B]
//!   engine plan l308 MODEL_DIR HEADS.json DOMAIN OUT.json [--buffers B] [--layers K]   L308 measured heads file -> plan (synthetic false)
//!   engine schedule PLAN MANIFEST TOKENS OUT.json            build + host proof (complete, acyclic, no collision, edges)
//!   engine move SCHEDULE MANIFEST OUT_DIR [--workers W] [--fire-us U] [--direct 0|1] [--engines 1|2]
//!               [--coupling brigade|clock|reservoir] [--period-us T]
//!                                                           movement engines alone: a stand-in piston that only holds U us
//!   engine run SCHEDULE MANIFEST TRACE MODEL_DIR OUT_DIR [--workers W] [--direct 0|1] [--engines 1|2]
//!   engine trace35 GOLDEN.json MODEL_DIR OUT.json          L306 golden (prompt_ids, generated_ids, routes) -> trace
//!   engine run35 SCHEDULE MANIFEST TRACE MODEL_DIR OUT_DIR [--workers W] [--direct 0|1] [--engines 1|2]
//!                                                           the 35B motor on the host: L306 exact forward, experts from the ring
//!                                                           both engines on the host: identical tokens gate + timing diagram
//!   engine stroke SCHEDULE MANIFEST MODEL_DIR OUT_DIR --n N --traces A.json,B.json,... [--positions P] [--workers W] [--engines E]
//!                                                           H6: N heads on one stroke; gate per head vs its own trace
//!   engine swap PLAN_A PLAN_B|shift:N MANIFEST OUT_DIR [--tokens T] [--swap-at K] [--lead L] [--fire-us U] [--workers W]
//!                                                           H4: head A resident in RAM, head B from token K, loader L tokens early
//!   engine union OUT.json [--positions P] TRACE...          H6 offline: expert union vs N from the traces
//!   engine today URL MODEL_DIR TRACE OUT.json               today's path (the live server) on the same prompt: agreement count
//!   engine stage35 --model DIR --goldens A,B,.. --index I --of N --first F --last L --convs C [--steps S] --prev [listen:|dial:]ADDR
//!                  --next [listen:|dial:]ADDR --source ring|shards [--manifest M] [--no-boundary-check] --out DIR
//!                                                           L258: one stage of the 35B split (layers F..=L), residual over TCP
//!   engine pipe-report OUT STEPS STAGE_DIR... --goldens A,B,..  L258 gate over a split run + cross-stage timing diagram
//!   engine pack35 MODEL_DIR FIRST LAST OUT_DIR              L258: the layer files of one stage, for a machine holding only it
use aien_engine::movement::{Engines, FillStamp, MovementEngine};
use aien_engine::piston35::{golden_to_trace, RingSource, Shared};
use aien_inference_abi::qwen38::{argmax as argmax38, Logits, Qwen38Config, Qwen38Reference};
use aien_engine::piston::{argmax, gate_identical, Crank, HostPiston, HostPistonBank, StrokeBeat};
use aien_engine::plan::{PlacementPlan, RouteTrace};
use aien_engine::pipeline::{overlap, pipe_diagram, run_stage, stage_plan, stage_schedule, End, StageReport, Wire};
use aien_engine::schedule::{Schedule, Stage};
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
        "l308" => {
            let dir = Path::new(&pos[1]);
            let model = dir.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
            (PlacementPlan::from_l308_heads(&model, dir, Path::new(&pos[2]), &pos[3], opt_usize(a, "--layers", 0), buffers)?, pos[4].clone())
        }
        x => return Err(format!("plan {x}: trace|union|fixed|l308")),
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


/// H4 head swap mid-run (movement engines + stand-in piston, like `move`): head A is resident in host RAM before the run;
/// tokens from `--swap-at K` fire on head B. A loader brings B into the resident tier starting `--lead L` tokens before
/// the swap (L = -1: never, every B part comes from the disk). Receipt: per-token time, the extra time the user sees
/// after the swap against the steady token before it, and the hit rate (parts served from RAM) after the swap.
fn cmd_swap(a: &[String]) -> Result<(), String> {
    let pa = PlacementPlan::load(Path::new(&a[0]))?;
    let pb = if let Some(sh) = a[1].strip_prefix("shift:") {
        // head B = head A with every expert id moved by `sh` (mod the expert count): a disjoint head of the same size
        let sh: u16 = sh.parse().map_err(|_| "shift:N")?;
        let n = opt_usize(a, "--experts-per-layer", EXPERTS) as u16;
        let mut b = pa.clone();
        for h in &mut b.heads {
            h.experts = h.experts.iter().map(|x| (x + sh) % n).collect();
        }
        b.source = format!("{} shifted by {sh} of {n}", pa.source);
        b
    } else {
        PlacementPlan::load(Path::new(&a[1]))?
    };
    let m = Manifest::load(Path::new(&a[2])).map_err(|e| e.to_string())?;
    let out = PathBuf::from(&a[3]);
    let tokens = opt_usize(a, "--tokens", 24) as u32;
    let swap_at = opt_usize(a, "--swap-at", 12) as u32;
    let lead: i64 = opt(a, "--lead").map(|s| s.parse().unwrap_or(-1)).unwrap_or(-1);
    let workers = opt_usize(a, "--workers", 8);
    let fire_us = opt_usize(a, "--fire-us", 2000) as i64;
    let s = Arc::new(Schedule::build_swap(&pa, &pb, &m, tokens, swap_at)?);
    let rep = s.check();
    if !rep.pass() {
        return Err(format!("swap schedule check FAIL: {:?}", rep.problems));
    }
    let k = s.layers.len() as u32;
    let part_ids = |pl: &PlacementPlan| -> Vec<u32> {
        let mut v: Vec<u32> = pl.heads.iter().flat_map(|h| h.experts.iter().map(move |x| (h.layer, *x))).filter_map(|(l, x)| s.parts.iter().find(|p| p.layer == l && p.expert == x).map(|p| p.id)).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let (ia, ib) = (part_ids(&pa), part_ids(&pb));
    let slot_bytes = aien_heartbeat::stream::Streamer::slot_bytes(&m);
    let resident = Arc::new(aien_heartbeat::ring::Ring::new(ia.len() + ib.len(), slot_bytes));
    let cfg = aien_heartbeat::stream::StreamConfig { slots: ia.len() + ib.len(), workers, direct: true, publish: true, lease_wait: std::time::Duration::from_secs(60) };
    let loader = Arc::new(aien_heartbeat::stream::Streamer::open(&m.shard_paths(), cfg).map_err(|e| e.to_string())?);
    let entry = |pid: u32| -> Result<aien_heartbeat::manifest::ExpertEntry, String> {
        let p = &s.parts[pid as usize];
        m.experts.iter().find(|e| e.layer == p.layer && e.expert == p.expert).cloned().ok_or(format!("part {pid}: not in the manifest"))
    };
    let ea: Vec<_> = ia.iter().map(|&p| entry(p)).collect::<Result<_, _>>()?;
    let eb: Vec<_> = ib.iter().map(|&p| entry(p)).collect::<Result<_, _>>()?;
    let load_head = |es: &[aien_heartbeat::manifest::ExpertEntry], base: usize, ld: &aien_heartbeat::stream::Streamer, res: &aien_heartbeat::ring::Ring| -> Result<(), String> {
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..workers.max(1))
                .map(|wi| {
                    sc.spawn(move || -> Result<(), String> {
                        for (i, e) in es.iter().enumerate().filter(|(i, _)| i % workers.max(1) == wi) {
                            ld.load_one(res, base + i, e).map_err(|x| x.to_string())?;
                        }
                        Ok(())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(Err("loader panicked".into()))).collect::<Result<Vec<()>, String>>()
        })?;
        Ok(())
    };
    let t_pre = Instant::now();
    load_head(&ea, 0, &loader, &resident)?;
    let head_a_load_ms = t_pre.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let mut me = MovementEngine::open(s.clone(), &m, workers, true, Engines::Direct, t0)?;
    me.set_resident(resident.clone());
    let ring = me.ring();
    let (jtx, jrx) = mpsc::channel();
    let (stx, srx) = mpsc::channel();
    let h = std::thread::spawn(move || me.serve(jrx, stx));
    let mut crank = Crank::new(s.clone(), ring, jtx, srx, t0);
    // the loader: waits for the trigger beat, then brings head B into the resident tier
    let beat_now = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let trigger: Option<u32> = (lead >= 0).then(|| (swap_at as i64 - lead).max(0) as u32 * k);
    let lt = {
        let (beat_now, loader, resident, eb) = (beat_now.clone(), loader.clone(), resident.clone(), eb.clone());
        let base = ia.len();
        std::thread::spawn(move || -> Result<(i64, i64), String> {
            let Some(tb) = trigger else { return Ok((-1, -1)) };
            while beat_now.load(std::sync::atomic::Ordering::Acquire) < tb {
                std::hint::spin_loop();
            }
            let st = t0.elapsed().as_nanos() as i64;
            let w = workers.max(1);
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..w)
                    .map(|wi| {
                        let (eb, loader, resident) = (&eb, &loader, &resident);
                        sc.spawn(move || -> Result<(), String> {
                            for (i, e) in eb.iter().enumerate().filter(|(i, _)| i % w == wi) {
                                loader.load_one(resident, base + i, e).map_err(|x| x.to_string())?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or(Err("loader panicked".into()))).collect::<Result<Vec<()>, String>>()
            })?;
            Ok((st, t0.elapsed().as_nanos() as i64))
        })
    };
    crank.prime()?;
    let mut traces = Vec::with_capacity(s.beats.len());
    let mut hits = Vec::with_capacity(s.beats.len());
    for beat in &s.beats {
        beat_now.store(beat.b, std::sync::atomic::Ordering::Release);
        if let Some(f) = &beat.fill {
            crank.issue(f)?;
        }
        let stamp = crank.wait_fill(beat.b)?;
        let fire_start = crank.now_ns();
        if fire_us > 0 {
            spin_until(&crank, fire_start + fire_us * 1000);
        }
        let fire_done = crank.now_ns();
        hits.push((stamp.hits, beat.fire_parts.len() as u32));
        traces.push(BeatTrace { b: beat.b, token: beat.token, layer: beat.layer, buffer: beat.fire_buffer, bytes: stamp.bytes, fill_issue_ns: stamp.issue_ns, fill_done_ns: stamp.done_ns, fire_start_ns: fire_start, fire_done_ns: fire_done, misfire: false });
    }
    beat_now.store(u32::MAX, std::sync::atomic::Ordering::Release);
    drop(crank);
    h.join().map_err(|_| "movement thread panicked")?;
    let (load_b_start, load_b_done) = lt.join().map_err(|_| "loader thread panicked")??;
    // per token: time from the previous token's last fire done to this token's last fire done (token 0: from the first fire start)
    let mut tok_ms = Vec::with_capacity(tokens as usize);
    for t in 0..tokens as usize {
        let end = traces[(t + 1) * k as usize - 1].fire_done_ns;
        let start = if t == 0 { traces[0].fire_start_ns } else { traces[t * k as usize - 1].fire_done_ns };
        tok_ms.push((end - start) as f64 / 1e6);
    }
    let mut before: Vec<f64> = tok_ms[1..swap_at as usize].to_vec();
    before.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let steady = before.get(before.len() / 2).copied().unwrap_or(0.0);
    let extra_ms: f64 = tok_ms[swap_at as usize..].iter().map(|t| (t - steady).max(0.0)).sum();
    let first_extra_ms = tok_ms.get(swap_at as usize).map(|t| t - steady).unwrap_or(0.0);
    let sb = (swap_at * k) as usize;
    let (h_after, n_after): (u32, u32) = hits[sb..].iter().fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
    let (h_before, n_before): (u32, u32) = hits[..sb].iter().fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
    let first_all_hit = (swap_at as usize..tokens as usize).find(|&t| hits[t * k as usize..(t + 1) * k as usize].iter().all(|x| x.0 == x.1));
    let swap_beat_ns = traces[sb].fill_issue_ns;
    write_receipts(
        &out,
        &traces,
        json!({"mode": "head-swap", "model": s.model, "workers": workers, "fire_us_setting": fire_us, "tokens": tokens, "swap_at_token": swap_at, "lead_tokens": lead,
               "layers": k, "buffers": s.buffers, "head_a_parts": ia.len(), "head_b_parts": ib.len(), "head_b_source": pb.source, "head_a_load_ms": head_a_load_ms,
               "head_b_load_ms": if load_b_done > 0 { (load_b_done - load_b_start) as f64 / 1e6 } else { -1.0 },
               "head_b_load_start_vs_swap_ms": if load_b_start > 0 { (load_b_start - swap_beat_ns) as f64 / 1e6 } else { 0.0 },
               "token_ms": tok_ms, "steady_token_ms_before_swap": steady, "first_token_after_swap_extra_ms": first_extra_ms, "user_visible_extra_ms_total": extra_ms,
               "hit_rate_before_swap": h_before as f64 / n_before.max(1) as f64, "hit_rate_after_swap": h_after as f64 / n_after.max(1) as f64,
               "first_all_hit_token_after_swap": first_all_hit}),
    )?;
    println!("swap lead {lead}: steady {steady:.2} ms/token, first token after swap +{first_extra_ms:.2} ms, total extra {extra_ms:.1} ms, hit rate after {:.3}, head B load {:.1} ms", h_after as f64 / n_after.max(1) as f64, if load_b_done > 0 { (load_b_done - load_b_start) as f64 / 1e6 } else { -1.0 });
    Ok(())
}

/// L306's golden -> my trace format (token 0 = prefill with the union head, then one decode step per generated id).
fn cmd_trace35(a: &[String]) -> Result<(), String> {
    let g: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[0]).map_err(|e| format!("{}: {e}", a[0]))?).map_err(|e| e.to_string())?;
    let dir = Path::new(&a[1]);
    let model = dir.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let tr = golden_to_trace(&g, &model)?;
    tr.save(Path::new(&a[2]))?;
    println!("trace35: {} steps ({} prompt ids, {} generated), {} layers, routed per layer at step 1: {:?}", tr.routes.len(), tr.prompt_tokens.len(), tr.argmax.len(), tr.routes[0].len(), tr.routes.get(1).map(|r| r.iter().map(|x| x.len()).collect::<Vec<_>>()));
    Ok(())
}

/// The 35B motor on the host: L306's exact forward with every routed expert served from the ring along the schedule.
/// Gate: tokens and routes identical to the golden the trace came from. Beats are drawn inside the source.
fn cmd_run35(a: &[String]) -> Result<(), String> {
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
    let k = tr.routes.first().map(|r| r.len()).unwrap_or(0);
    if s.layers.len() != k {
        return Err(format!("schedule has {} layers per token, the trace {}: the 35B piston fires every layer", s.layers.len(), k));
    }
    let steps = (s.tokens as usize).min(tr.routes.len()).min(tr.argmax.len());
    if steps == 0 {
        return Err("no steps".into());
    }
    let t0 = Instant::now();
    let (crank, h) = start_engines(s.clone(), &m, workers, direct, engines, t0)?;
    let ring = crank.ring();
    let shared = Arc::new(Shared::default());
    let source = RingSource::open(dir, ring, s.clone(), crank, shared.clone())?;
    source.prime()?;
    let t_open = Instant::now();
    let config = Qwen38Config::from_checkpoint(dir)?;
    let mut model = Qwen38Reference::new(config, source)?;
    let open_s = t_open.elapsed().as_secs_f64();
    shared.armed.store(true, std::sync::atomic::Ordering::Release);
    let mut state = model.new_state();
    let mut got: Vec<u32> = Vec::with_capacity(steps);
    let mut routes_seen: Vec<Vec<Vec<u16>>> = Vec::with_capacity(steps);
    let mut step_s: Vec<f64> = Vec::with_capacity(steps);
    let mut failure: Option<String> = None;
    for step in 0..steps {
        shared.token.store(step as u32, std::sync::atomic::Ordering::Release);
        let ids: Vec<u32> = if step == 0 { tr.prompt_tokens.clone() } else { vec![tr.argmax[step - 1]] };
        let ts = Instant::now();
        match model.forward(&mut state, &ids, Logits::Last) {
            Ok(logits) => {
                step_s.push(ts.elapsed().as_secs_f64());
                got.push(argmax38(&logits) as u32);
                let mut r = model.routes.clone();
                for v in &mut r {
                    v.sort_unstable();
                    v.dedup();
                }
                routes_seen.push(r);
            }
            Err(e) => {
                failure = Some(format!("step {step}: {e}"));
                break;
            }
        }
    }
    shared.close(t0.elapsed().as_nanos() as i64);
    drop(model);
    h.join().map_err(|_| "movement thread panicked")?;
    let (same, total) = gate_identical(&tr, &got);
    let routes_same = routes_seen.iter().zip(&tr.routes).filter(|(a, b)| a == b).count();
    let eager = shared.eager_expert_reads.load(std::sync::atomic::Ordering::Relaxed);
    let traces = shared.traces.lock().unwrap().clone();
    let misfires = traces.iter().filter(|t| t.misfire).count();
    let gate = json!({"identical_tokens": failure.is_none() && same == total && total == steps && steps > 0, "same": same, "of": total, "steps": steps, "routes_identical_steps": routes_same, "got": got, "expected": &tr.argmax[..steps], "failure": failure, "eager_expert_reads_at_open": eager, "misfires": misfires});
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    write_json(&out.join("gate.json"), &gate)?;
    write_receipts(&out, &traces, json!({"mode": "host-loop-35b", "model": s.model, "workers": workers, "direct": direct, "engines": engines.count(), "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(), "tokens": s.tokens, "steps": steps, "open_seconds": open_s, "step_seconds": step_s, "eager_expert_reads_at_open": eager, "gate": gate.clone()}))?;
    if let Some(f) = failure {
        return Err(format!("35B loop FAILED closed: {f} (tokens identical so far {same}/{total})"));
    }
    if same != total || total != steps {
        return Err(format!("identity gate FAIL: {same} of {total} (steps {steps})"));
    }
    println!("identity gate PASS (35B): {same}/{total} tokens, routes identical at {routes_same}/{steps} steps, eager expert reads at open {eager}, {} beats, {misfires} misfires", traces.len());
    Ok(())
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
        Some("swap") => cmd_swap(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("trace35") => cmd_trace35(&args[1..]),
        Some("run35") => cmd_run35(&args[1..]),
        Some("stroke") => cmd_stroke(&args[1..]),
        Some("union") => cmd_union(&args[1..]),
        Some("today") => cmd_today(&args[1..]),
        Some("stage35") => cmd_stage35(&args[1..]),
        Some("pipe-report") => cmd_pipe_report(&args[1..]),
        Some("pack35") => cmd_pack35(&args[1..]),
        _ => Err("engine manifest|trace|plan|schedule|move|run|stroke|union|today (see the file header)".to_string()),
    };
    if let Err(e) = r {
        eprintln!("engine: {e}");
        std::process::exit(1);
    }
}

/// L258 goldens (L306's gen-p*.json) as traces, in conversation order.
fn goldens_of(a: &[String], model: &str) -> Result<Vec<RouteTrace>, String> {
    opt(a, "--goldens").ok_or("--goldens a.json,b.json,...")?.split(',').map(|p| {
        let g: Value = serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
        golden_to_trace(&g, model)
    }).collect()
}

fn stage_of(a: &[String], model_layers: u16) -> Result<Stage, String> {
    let n = |k: &str| opt(a, k).ok_or(format!("{k} N")).and_then(|v| v.parse::<u32>().map_err(|e| format!("{k}: {e}")));
    Ok(Stage { index: n("--index")?, of: n("--of")?, first: n("--first")? as u16, last: n("--last")? as u16, model_layers, convs: n("--convs")? })
}

/// One stage of the 35B split (L258). `--source ring`: experts from the heartbeat ring along the stage's own schedule
/// (the motor); `--source shards`: the exact reference reading its layer files directly (a machine without the
/// manifest, e.g. the Pi with its layers packed in RAM). Writes OUT/stage.json (+ the ring's beats and diagram).
fn cmd_stage35(a: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(opt(a, "--model").ok_or("--model DIR")?);
    let out = PathBuf::from(opt(a, "--out").ok_or("--out DIR")?);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let config = Qwen38Config::from_checkpoint(&dir)?;
    let st = stage_of(a, config.layers as u16)?;
    let steps = opt_usize(a, "--steps", 32) as u32;
    let model_name = dir.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let goldens = goldens_of(a, &model_name)?;
    let check = !a.iter().any(|x| x == "--no-boundary-check");
    let source = opt(a, "--source").unwrap_or_else(|| "ring".into());
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim().to_string();
    let (embed, head) = (st.index == 0, st.index + 1 == st.of);
    let wire = |st: &Stage| -> Result<Option<Wire>, String> {
        if st.of == 1 {
            return Ok(None);
        }
        let prev = End::parse(&opt(a, "--prev").ok_or("--prev [listen:|dial:]ADDR")?, true);
        let next = End::parse(&opt(a, "--next").ok_or("--next [listen:|dial:]ADDR")?, false);
        Ok(Some(Wire::open(st, prev, next, check)?))
    };
    let t_open = Instant::now();
    let mut rep = match source.as_str() {
        "shards" => {
            let mut model = Qwen38Reference::new_stage(config, aien_inference_abi::qwen3_moe_omega::ShardSource::open(&dir)?, embed, head)?;
            let open_s = t_open.elapsed().as_secs_f64();
            let w = wire(&st)?;
            let mut r = run_stage(&mut model, &st, &goldens, steps, w, check, &mut |_| {});
            r.open_seconds = open_s;
            r
        }
        "ring" => {
            let m = Manifest::load(Path::new(&opt(a, "--manifest").ok_or("--manifest M")?)).map_err(|e| e.to_string())?;
            let plan = stage_plan(&model_name, &dir, &goldens[..st.convs as usize], st.first, st.last, opt_usize(a, "--buffers", 3));
            let s = Arc::new(stage_schedule(&plan, &m, &st, steps)?);
            s.save(&out.join("schedule.json"))?;
            let t0 = Instant::now();
            let (crank, h) = start_engines(s.clone(), &m, opt_usize(a, "--workers", 8), opt_usize(a, "--direct", 1) == 1, engines_of(a)?, t0)?;
            let shared = Arc::new(Shared::default());
            let src = RingSource::open(&dir, crank.ring(), s.clone(), crank, shared.clone())?;
            src.prime()?;
            let mut model = Qwen38Reference::new_stage(config, src, embed, head)?;
            shared.armed.store(true, std::sync::atomic::Ordering::Release);
            let open_s = t_open.elapsed().as_secs_f64();
            let w = wire(&st)?;
            let sh = shared.clone();
            let mut r = run_stage(&mut model, &st, &goldens, steps, w, check, &mut |p| sh.token.store(p, std::sync::atomic::Ordering::Release));
            shared.close(t0.elapsed().as_nanos() as i64);
            drop(model);
            h.join().map_err(|_| "movement thread panicked")?;
            r.open_seconds = open_s;
            let traces = shared.traces.lock().unwrap().clone();
            let misfires = traces.iter().filter(|t| t.misfire).count();
            write_receipts(&out.join("beats"), &traces, json!({"mode": "stage35-ring", "stage": st, "misfires": misfires, "eager_expert_reads_at_open": shared.eager_expert_reads.load(std::sync::atomic::Ordering::Relaxed)}))?;
            r
        }
        other => return Err(format!("--source {other}: ring or shards")),
    };
    rep.host = host;
    rep.source = source;
    write_json(&out.join("stage.json"), &serde_json::to_value(&rep).unwrap())?;
    let n = rep.events.len();
    match &rep.failure {
        Some(f) => Err(format!("stage {} FAILED closed after {n} passes: {f}", st.index)),
        None => {
            println!("stage {} of {} layers {}..={} ({}): {n} passes in {:.2} s, routes identical {:?}", st.index, st.of, st.first, st.last, rep.source, rep.wall_seconds, rep.routes_identical);
            Ok(())
        }
    }
}

/// L258 gate over a split run: stage 0's tokens identical to every golden, routes identical in every stage, plus the
/// cross-stage timing diagram, the overlap and tokens per second. `engine pipe-report OUT STEPS STAGE_DIR...`
fn cmd_pipe_report(a: &[String]) -> Result<(), String> {
    let out = PathBuf::from(&a[0]);
    let steps: usize = a[1].parse().map_err(|_| "STEPS")?;
    let reps: Vec<StageReport> = a[2..].iter().map(|d| serde_json::from_slice(&std::fs::read(Path::new(d).join("stage.json")).map_err(|e| format!("{d}: {e}"))?).map_err(|e| format!("{d}: {e}"))).collect::<Result<_, String>>()?;
    let goldens = goldens_of(a, "")?;
    let s0 = reps.first().ok_or("no stages")?;
    let convs = s0.tokens.len();
    let mut same = 0;
    let mut of = 0;
    let mut per_conv = Vec::new();
    for (c, got) in s0.tokens.iter().enumerate() {
        let want = &goldens[c].argmax[..steps.min(goldens[c].argmax.len())];
        let (s, n) = (got.iter().zip(want).filter(|(x, y)| x == y).count(), want.len());
        same += s;
        of += n;
        per_conv.push(json!({"conv": c, "same": s, "of": n, "got": got, "expected": want}));
    }
    let failures: Vec<Value> = reps.iter().filter_map(|r| r.failure.clone().map(|f| json!({"stage": r.stage.as_ref().map(|s| s.index), "failure": f}))).collect();
    let routes_ok = reps.iter().all(|r| r.routes_identical.iter().all(|&n| n as usize == steps));
    let identical = failures.is_empty() && same == of && of == convs * steps && s0.tokens.iter().all(|t| t.len() == steps);
    // wall time on the clocks of stage 0 and the last stage (the Spark); a remote middle stage may be offset
    let t0 = s0.events.iter().map(|e| e.start_ns).min().unwrap_or(0);
    let t_end = s0.events.iter().map(|e| e.sent_ns).chain(reps.last().unwrap().events.iter().map(|e| e.sent_ns)).max().unwrap_or(t0);
    let wall = (t_end - t0) as f64 / 1e9;
    let (busy, both) = overlap(&reps);
    // per-token latency at stage 0: token arrival minus the send of the pass that produced it
    let lat: Vec<f64> = s0.events.iter().filter(|e| e.step > 0).filter_map(|e| s0.events.iter().find(|x| x.conv == e.conv && x.step + 1 == e.step).map(|x| (e.recv_ns - x.start_ns) as f64 / 1e6)).collect();
    let gate = json!({"identical_tokens": identical, "same": same, "of": of, "convs": convs, "steps": steps, "routes_identical_all_stages": routes_ok, "routes_identical": reps.iter().map(|r| r.routes_identical.clone()).collect::<Vec<_>>(), "failures": failures, "per_conv": per_conv});
    let summary = json!({"stages": reps.iter().map(|r| json!({"stage": r.stage, "host": r.host, "source": r.source, "boundary_check": r.boundary_check, "passes": r.events.len(), "open_s": r.open_seconds, "wall_s": r.wall_seconds, "fire_ms": series(&r.events.iter().map(|e| (e.done_ns - e.start_ns) as f64 / 1e6).collect::<Vec<_>>())})).collect::<Vec<_>>(),
        "wall_s": wall, "tokens": of, "tok_per_s": of as f64 / wall, "busy_fraction": busy, "two_or_more_stages_busy_fraction": both, "token_latency_ms": series(&lat), "gate": gate.clone()});
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    write_json(&out.join("gate.json"), &gate)?;
    write_json(&out.join("summary.json"), &summary)?;
    let d = pipe_diagram(&reps, 160);
    std::fs::write(out.join("diagram.txt"), &d).map_err(|e| e.to_string())?;
    print!("{d}");
    println!("pipe gate {}: tokens {same}/{of} over {convs} conversations x {steps} steps, routes identical in every stage {routes_ok}, {:.3} tok/s, overlap {:.3}", if identical && routes_ok { "PASS" } else { "FAIL" }, of as f64 / wall, both);
    if identical && routes_ok { Ok(()) } else { Err("pipe gate FAIL".into()) }
}

/// L258 stage pack: the layer files of `first..=last` + config + an index naming only them (a machine that holds one
/// stage gets only its own weights). `engine pack35 MODEL_DIR FIRST LAST OUT_DIR`
fn cmd_pack35(a: &[String]) -> Result<(), String> {
    let (dir, out) = (Path::new(&a[0]), Path::new(&a[3]));
    let (first, last): (usize, usize) = (a[1].parse().map_err(|_| "FIRST")?, a[2].parse().map_err(|_| "LAST")?);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let idx: Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let map = idx["weight_map"].as_object().ok_or("index: no weight_map")?;
    let keep = |name: &str| -> bool {
        name.find("layers.").and_then(|i| name[i + 7..].split('.').next()?.parse::<usize>().ok()).map(|l| (first..=last).contains(&l) && !name.contains("visual") && !name.starts_with("mtp")).unwrap_or(false)
    };
    let mut files = std::collections::BTreeSet::new();
    let mut kept = serde_json::Map::new();
    for (k, v) in map {
        if keep(k) {
            files.insert(v.as_str().ok_or("index: non-string file")?.to_string());
            kept.insert(k.clone(), v.clone());
        }
    }
    let mut bytes = 0u64;
    for f in &files {
        bytes += std::fs::copy(dir.join(f), out.join(f)).map_err(|e| format!("{f}: {e}"))?;
    }
    for f in ["config.json", "generation_config.json"] {
        std::fs::copy(dir.join(f), out.join(f)).map_err(|e| format!("{f}: {e}"))?;
    }
    write_json(&out.join("model.safetensors.index.json"), &json!({"metadata": {"l258_stage_pack": format!("layers {first}..={last} of {}", dir.display())}, "weight_map": kept}))?;
    println!("pack35: layers {first}..={last}: {} tensors in {} files, {bytes} bytes", kept.len(), files.len());
    Ok(())
}
