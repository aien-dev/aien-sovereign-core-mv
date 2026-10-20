//! `engine`: the L312 command line. Every subcommand writes its receipt as JSON next to its outputs.
//!
//!   engine manifest MODEL_DIR OUT.json                       hash every expert once (heartbeat Manifest::build)
//!   engine trace MODEL_DIR MANIFEST OUT.json --prompt T --gen N [--slots S]
//!                                                           teacher-forced host pass (ring-backed, L190 code): routes per
//!                                                           layer + argmax per position = the identity gate's reference
//!   engine plan trace TRACE MODEL_DIR OUT.json [--buffers B] synthetic plan: head(layer) = union of routed experts
//!   engine plan fixed MODEL_DIR OUT.json --layers K --experts E [--buffers B]
//!   engine schedule PLAN MANIFEST TOKENS OUT.json            build + host check (complete, acyclic, no collision)
//!   engine move SCHEDULE MANIFEST OUT_DIR [--workers W] [--beat-us U] [--direct 0|1]
//!                                                           movement engine alone: a stand-in piston that only waits U us
//!   engine run SCHEDULE MANIFEST TRACE MODEL_DIR OUT_DIR [--workers W] [--direct 0|1]
//!                                                           both engines on the host: identical tokens gate + timing diagram
//!   engine today URL MODEL_DIR TRACE OUT.json                today's path (the live server) on the same prompt: agreement count
use aien_engine::movement::MovementEngine;
use aien_engine::piston::{argmax, gate_identical, Crank, HostPiston};
use aien_engine::plan::{PlacementPlan, RouteTrace};
use aien_engine::schedule::Schedule;
use aien_engine::timing::{png_diagram, summarize, text_diagram, BeatTrace};
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
    let mut next = 0u32;
    let mut seq = ids.clone();
    let mut i = 0;
    while i < seq.len() {
        let logits = lm.step(seq[i])?;
        next = argmax(&logits);
        tr.argmax.push(next);
        tr.routes.push(lm.last_routes.clone());
        i += 1;
        if i == seq.len() && seq.len() < ids.len() + gen {
            seq.push(next);
        }
    }
    let _ = next;
    tr.host_ns_per_token = (t.elapsed().as_nanos() / seq.len() as u128) as u64;
    let gen_ids = &seq[ids.len()..];
    let gen_text = tk.decode(gen_ids, true).map_err(|e| e.to_string())?;
    tr.save(out)?;
    let side = json!({"prompt_tokens": ids.len(), "generated": gen_ids, "generated_text": gen_text, "positions": seq.len(),
        "host_ms_per_token": tr.host_ns_per_token as f64 / 1e6, "ring_slots": slots, "cache": format!("{:?}", lm.cache.stats)});
    write_json(&out.with_extension("side.json"), &side)?;
    println!("trace: {} positions, generated {:?}\n{}", seq.len(), gen_text, serde_json::to_string(&side).unwrap());
    Ok(())
}

fn cmd_plan(a: &[String]) -> Result<(), String> {
    let buffers = opt_usize(a, "--buffers", 2);
    let plan = match a[0].as_str() {
        "trace" => {
            let tr = RouteTrace::load(Path::new(&a[1]))?;
            let dir = Path::new(&a[2]);
            let k = tr.routes.first().map(|r| r.len()).ok_or("empty trace")?;
            PlacementPlan::synthetic_from_trace(&tr.model, dir, &tr, k, buffers)
        }
        "fixed" => {
            let dir = Path::new(&a[1]);
            PlacementPlan::synthetic_fixed(&dir.file_name().unwrap().to_string_lossy(), dir, opt_usize(a, "--layers", 4), opt_usize(a, "--experts", 8) as u16, buffers)
        }
        x => return Err(format!("plan {x}: trace|fixed")),
    };
    let out = Path::new(if a[0] == "trace" { &a[3] } else { &a[2] });
    plan.save(out)?;
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
    s.save(Path::new(&a[3]))?;
    let rep = serde_json::to_value(&r).unwrap();
    write_json(&Path::new(&a[3]).with_extension("check.json"), &rep)?;
    println!("schedule: {} beats, {} parts, {} buffers x {} slots x {} bytes; check {}: complete {} acyclic {} no_collision {} bytes/beat {}..{} total {}",
        s.beats.len(), s.parts.len(), s.buffers, s.slots, s.slot_bytes, if r.pass() { "PASS" } else { "FAIL" }, r.complete, r.acyclic, r.no_collision, r.bytes_per_beat_min, r.bytes_per_beat_max, r.bytes_total);
    for p in &r.problems {
        println!("  problem: {p}");
    }
    if r.pass() { Ok(()) } else { Err("schedule check FAIL".into()) }
}

fn start_engines(s: Arc<Schedule>, m: &Manifest, workers: usize, direct: bool, t0: Instant) -> Result<(Crank, std::thread::JoinHandle<()>), String> {
    let me = MovementEngine::open(s.clone(), m, workers, direct, t0)?;
    let ring = me.ring();
    let (jtx, jrx) = mpsc::channel();
    let (stx, srx) = mpsc::channel();
    let h = std::thread::spawn(move || me.serve(jrx, stx));
    Ok((Crank::new(s, ring, jtx, srx, t0), h))
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
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    Ok(())
}

fn cmd_move(a: &[String]) -> Result<(), String> {
    let s = Arc::new(Schedule::load(Path::new(&a[0]))?);
    let m = Manifest::load(Path::new(&a[1])).map_err(|e| e.to_string())?;
    let out = PathBuf::from(&a[2]);
    let workers = opt_usize(a, "--workers", 8);
    let beat_us = opt_usize(a, "--beat-us", 0) as u64;
    let direct = opt_usize(a, "--direct", 1) == 1;
    if !s.check().pass() {
        return Err("schedule check FAIL".into());
    }
    let t0 = Instant::now();
    let (mut crank, h) = start_engines(s.clone(), &m, workers, direct, t0)?;
    crank.prime()?;
    let mut traces = Vec::with_capacity(s.beats.len());
    for beat in &s.beats {
        if let Some(f) = &beat.fill {
            crank.issue(f)?;
        }
        let stamp = crank.wait_fill(beat.b)?;
        let fire_start = crank.now_ns();
        // stand-in piston: touch nothing, hold the beat for beat_us (the clock under test)
        if beat_us > 0 {
            let until = fire_start + beat_us as i64 * 1000;
            while crank.now_ns() < until {
                std::hint::spin_loop();
            }
        }
        let fire_done = crank.now_ns();
        traces.push(BeatTrace { b: beat.b, token: beat.token, layer: beat.layer, buffer: beat.fire_buffer, bytes: stamp.bytes, fill_issue_ns: stamp.issue_ns, fill_done_ns: stamp.done_ns, fire_start_ns: fire_start, fire_done_ns: fire_done, misfire: false });
    }
    drop(crank);
    h.join().map_err(|_| "movement thread panicked")?;
    write_receipts(&out, &traces, json!({"mode": "movement-only", "model": s.model, "workers": workers, "direct": direct, "beat_us_setting": beat_us, "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(), "tokens": s.tokens}))
}

fn cmd_run(a: &[String]) -> Result<(), String> {
    let s = Arc::new(Schedule::load(Path::new(&a[0]))?);
    let m = Manifest::load(Path::new(&a[1])).map_err(|e| e.to_string())?;
    let tr = RouteTrace::load(Path::new(&a[2]))?;
    let dir = Path::new(&a[3]);
    let out = PathBuf::from(&a[4]);
    let workers = opt_usize(a, "--workers", 8);
    let direct = opt_usize(a, "--direct", 1) == 1;
    if !s.check().pass() {
        return Err("schedule check FAIL".into());
    }
    if s.layers.len() != tr.routes.first().map(|r| r.len()).unwrap_or(0) {
        return Err(format!("schedule has {} layers per token, the trace {}: the host piston fires every layer of the model", s.layers.len(), tr.routes.first().map(|r| r.len()).unwrap_or(0)));
    }
    let seq: Vec<u32> = tr.prompt_tokens.iter().copied().chain(tr.argmax[tr.prompt_tokens.len() - 1..tr.argmax.len() - 1].iter().copied()).collect();
    let n = (s.tokens as usize).min(seq.len());
    let mut piston = HostPiston::open(dir)?;
    let t0 = Instant::now();
    let (mut crank, h) = start_engines(s.clone(), &m, workers, direct, t0)?;
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
    write_json(&out.join("gate.json"), &gate).ok();
    std::fs::create_dir_all(&out).ok();
    write_json(&out.join("gate.json"), &gate)?;
    write_receipts(&out, &traces, json!({"mode": "host-loop", "model": s.model, "workers": workers, "direct": direct, "buffers": s.buffers, "slots": s.slots, "layers": s.layers.len(), "tokens": n, "gate": gate}))?;
    if same != total {
        return Err(format!("identity gate FAIL: {same} of {total}"));
    }
    println!("identity gate PASS: {same}/{total} tokens, routes identical at {routes_same}/{total} positions");
    Ok(())
}

fn http_post_json(url: &str, body: &str) -> Result<String, String> {
    let u = url.strip_prefix("http://").ok_or("http:// URL only")?;
    let (host, path) = u.split_once('/').map(|(h, p)| (h, format!("/{p}"))).unwrap_or((u, "/".into()));
    let mut s = std::net::TcpStream::connect(host).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(600))).ok();
    write!(s, "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).map_err(|e| e.to_string())?;
    let mut resp = String::new();
    s.read_to_string(&mut resp).map_err(|e| e.to_string())?;
    let (_, b) = resp.split_once("\r\n\r\n").ok_or("bad HTTP response")?;
    Ok(b.to_string())
}

fn cmd_today(a: &[String]) -> Result<(), String> {
    let (url, dir, tr, out) = (&a[0], Path::new(&a[1]), RouteTrace::load(Path::new(&a[2]))?, Path::new(&a[3]));
    let tk = tokenizer(dir)?;
    let prompt_full = tk.decode(&tr.prompt_tokens, false).map_err(|e| e.to_string())?;
    let prompt = prompt_full.strip_prefix(TEMPLATE.0).and_then(|p| p.strip_suffix(TEMPLATE.1)).ok_or("trace prompt is not in the chat template")?;
    let gen: Vec<u32> = tr.argmax[tr.prompt_tokens.len() - 1..].to_vec();
    let body = json!({"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": gen.len(), "temperature": 0}).to_string();
    let t = Instant::now();
    let resp = http_post_json(url, &body)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let v: Value = serde_json::from_str(resp.trim_start_matches(|c: char| c.is_ascii_hexdigit() || c == '\r' || c == '\n')).or_else(|_| serde_json::from_str(&resp)).map_err(|e| format!("{e}: {}", &resp[..resp.len().min(200)]))?;
    let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    let ids: Vec<u32> = tk.encode(text.clone(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let same = ids.iter().zip(&gen).filter(|(a, b)| a == b).count();
    let host_text = tk.decode(&gen, true).map_err(|e| e.to_string())?;
    let rec = json!({"url": url, "prompt": prompt, "today_text": text, "today_ids": ids, "host_text": host_text, "host_ids": gen, "same_prefix_ids": same, "of": gen.len(), "today_ms": ms, "note": "agreement count between today's chip path and the host trace; the loop's own gate is host loop == host trace"});
    write_json(out, &rec)?;
    println!("{}", serde_json::to_string_pretty(&rec).unwrap());
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
        Some("today") => cmd_today(&args[1..]),
        _ => Err("engine manifest|trace|plan|schedule|move|run|today (see the file header)".to_string()),
    };
    if let Err(e) = r {
        eprintln!("engine: {e}");
        std::process::exit(1);
    }
}
