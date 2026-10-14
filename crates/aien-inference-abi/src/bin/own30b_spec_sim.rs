//! Host-only drafter replay (W-L4c cut 2). NOT a hardware result.
//!
//!   own30b_spec_sim <golden.json> <run.json> [<run.json> ..] [--k 16] [--global]
//!
//! Replays each prompt's recorded spec-off output (`tokens_off` of an own30b_spec_check run.json,
//! prompt ids from `prompt_tokens` or the golden file) through the prompt-lookup drafter and the
//! suffix drafter exactly as `generate_spec_drafter` drives them: the round's first token is the
//! model's, the draft is accepted while it equals the recorded output. With a verify pass that is
//! token-identical to plain decode (the decode path) the chip produces exactly these rounds, so
//! acceptance here is exact for that path; the time is the cost model's (the suffix config's
//! CostModel, AIEN_OWN30B_SPEC_COST_* to change it), i.e. a projection.
//! `--global`: each prompt's suffix drafter also holds the OTHER prompts' sequences (past conversations).

use std::collections::BTreeMap;

use aien_inference_abi::qwen3_moe_omega::spec::suffix::{SuffixConfig, SuffixDrafter};
use aien_inference_abi::qwen3_moe_omega::spec::{find_draft, SpecConfig};

#[derive(Default, Debug)]
struct Sim {
    rounds: usize,
    plain: usize,
    drafted: usize,
    accepted: usize,
    ms: f64,
}

fn simulate(prompt: &[u32], out: &[u32], k: usize, mut draft: impl FnMut(&[u32], usize) -> Vec<u32>, mut push: impl FnMut(u32), cfg: &SuffixConfig) -> Sim {
    let c = &cfg.cost;
    let mut s = Sim::default();
    let mut ctx = prompt.to_vec();
    let mut i = 0usize;
    while i < out.len() {
        let t_first = out[i];
        ctx.push(t_first);
        push(t_first);
        let left = out.len() - i - 1;
        let kk = k.min(left);
        let d = if kk == 0 { Vec::new() } else { draft(&ctx, kk) };
        if d.is_empty() {
            s.plain += 1;
            s.ms += c.plain_ms as f64;
            i += 1;
            continue;
        }
        let acc = d.iter().zip(&out[i + 1..]).take_while(|(a, b)| a == b).count();
        s.rounds += 1;
        s.drafted += d.len();
        s.accepted += acc;
        s.ms += (c.verify_ms(d.len() + 1) + c.lm_ms * (acc + 1) as f32) as f64;
        for &t in &out[i + 1..i + 1 + acc] {
            ctx.push(t);
            push(t);
        }
        i += 1 + acc;
    }
    s
}

fn main() {
    let mut it = std::env::args().skip(1);
    let mut pos = Vec::new();
    let (mut k, mut global) = (16usize, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--k" => k = it.next().expect("--k N").parse().expect("number"),
            "--global" => global = true,
            _ => pos.push(a),
        }
    }
    if pos.len() < 2 {
        eprintln!("usage: own30b_spec_sim <golden.json> <run.json>.. [--k 16] [--global]");
        std::process::exit(2);
    }
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&pos[0]).expect("golden")).expect("json");
    let mut gold_prompts: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for p in golden["prompts"].as_array().expect("prompts") {
        let ids = p["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        gold_prompts.insert(p["name"].as_str().unwrap().to_string(), ids);
    }
    // name -> (prompt, longest recorded spec-off output)
    let mut data: BTreeMap<String, (Vec<u32>, Vec<u32>)> = BTreeMap::new();
    for rp in &pos[1..] {
        let run: serde_json::Value = serde_json::from_slice(&std::fs::read(rp).expect("run.json")).expect("json");
        for r in run["runs"].as_array().expect("runs") {
            let name = r["prompt"].as_str().unwrap().to_string();
            let off: Vec<u32> = r["tokens_off"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let prompt: Vec<u32> = match r.get("prompt_tokens").and_then(|v| v.as_array()) {
                Some(a) => a.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
                None => gold_prompts.get(&name).cloned().unwrap_or_default(),
            };
            if prompt.is_empty() {
                continue;
            }
            let e = data.entry(name).or_insert((prompt, Vec::new()));
            if off.len() > e.1.len() {
                e.1 = off;
            }
        }
    }
    let sc = SuffixConfig::from_env();
    let lk = SpecConfig::default();
    println!("HOST REPLAY (not hardware): k={k} global={global} cost {:?}", sc.cost);
    println!("{:<14} {:>5} | {:>28} | {:>28} | {:>8}", "prompt", "out", "lookup: rounds acc tok/rnd t/s", "suffix: rounds acc tok/rnd t/s", "plain t/s");
    let names: Vec<String> = data.keys().cloned().collect();
    for name in &names {
        let (prompt, out) = &data[name];
        let plain_tps = 1000.0 / sc.cost.plain_ms as f64;
        let l = simulate(prompt, out, lk.max_draft.min(k), |c, kk| find_draft(c, lk.max_ngram, lk.min_ngram, kk), |_| {}, &sc);
        let mut d = SuffixDrafter::new(sc.clone());
        if global {
            for other in names.iter().filter(|n| *n != name) {
                let (p, o) = &data[other];
                d.finish(&p.iter().chain(o.iter()).copied().collect::<Vec<_>>());
            }
        }
        d.begin(prompt);
        let dc = std::cell::RefCell::new(d);
        let s = simulate(prompt, out, k, |c, kk| dc.borrow().draft(c, kk).tokens, |t| dc.borrow_mut().push(t), &sc);
        let f = |x: &Sim| {
            let tpr = if x.rounds > 0 { (x.accepted + x.rounds) as f64 / x.rounds as f64 } else { 0.0 };
            format!("{:>6} {:>4} {:>6.2} {:>7.1}", x.rounds, x.accepted, tpr, out.len() as f64 * 1000.0 / x.ms)
        };
        println!("{:<14} {:>5} | {:>28} | {:>28} | {:>8.1}", name, out.len(), f(&l), f(&s), plain_tps);
    }
}
