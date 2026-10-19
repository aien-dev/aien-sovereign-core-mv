//! wormhole_run: host (CPU) checks of the Wormhole streamer and round (docs/WORMHOLE.md). No timing claims.
//!
//!   wormhole_run manifest   FN_DIR OUT.json
//!     Heartbeat manifest of Flash-Next's routed experts (byte spans + BLAKE3 per expert, reads every expert once).
//!   wormhole_run ring-check FN_DIR MANIFEST LAYERS SLOTS OUT.json ID...
//!     Flash-Next with its routed experts served from the Heartbeat ring (SLOTS versioned slots, demand fill, LRU)
//!     against the plain checkpoint reader: whole prompt plain vs one word at a time through the ring; every logit
//!     must be bit for bit equal. WORMHOLE_CORRUPT=1 adds the negative control (one used expert's digest replaced:
//!     the ring must fail closed).
//!   wormhole_run pair Q38_DIR FN_DIR MANIFEST SLOTS K N OUT.json ID...
//!     The Wormhole round with real weights: Flash-Next drafts (experts through the ring), the 27B verifies, K
//!     words per round; and the 27B alone. The two outputs must be identical. WORMHOLE_DIRECTION=reverse swaps
//!     roles (27B drafts, Flash-Next verifies; then Flash-Next alone is the reference). WORMHOLE_FAULT=extra|rowshift
//!     runs a negative control (must differ).
use std::path::Path;
use std::time::Instant;

use aien_heartbeat::Manifest;
use aien_inference_abi::flashnext::{FlashNextConfig, FlashNextReference, FlashNextState};
use aien_inference_abi::qwen38::{Logits, Qwen38Reference, Qwen38State};
use aien_inference_abi::qwen3_moe_omega::{ShardSource, TensorSource};
use aien_wormhole::cache::ExpertCache;
use aien_wormhole::source::RingSource;
use aien_wormhole::spec::{generate_alone, generate_wormhole, Fault, GreedyModel};
use serde_json::json;

const EOS: [u32; 2] = [248044, 248046];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("manifest") if a.len() == 4 => manifest(&a[2], &a[3]),
        Some("ring-check") if a.len() >= 8 => ring_check(&a[2], &a[3], &a[4], &a[5], &a[6], &a[7..]),
        Some("pair") if a.len() >= 10 => pair(&a[2], &a[3], &a[4], &a[5], &a[6], &a[7], &a[8], &a[9..]),
        _ => Err("usage: see the file header".into()),
    };
    if let Err(e) = r {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn ids_of(v: &[String]) -> Result<Vec<u32>, String> {
    v.iter()
        .flat_map(|s| s.split_whitespace())
        .map(|s| s.parse().map_err(|_| format!("id {s}")))
        .collect()
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim().to_string()
}

fn argmax(row: &[f32]) -> u32 {
    let mut b = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x > row[b] {
            b = i;
        }
    }
    b as u32
}

fn manifest(dir: &str, out: &str) -> Result<(), String> {
    let t0 = Instant::now();
    let m = Manifest::build(Path::new(dir)).map_err(|e| e.to_string())?;
    m.save(Path::new(out)).map_err(|e| e.to_string())?;
    println!(
        "PASS manifest: {} experts ({} layers x {}), {} bytes, max expert {} bytes, {:.0}s",
        m.experts.len(),
        m.num_layers,
        m.experts_per_layer,
        m.total_bytes(),
        m.max_expert_len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn fn_ring(dir: &str, man: &str, layers: usize, slots: usize) -> Result<FlashNextReference<RingSource<ShardSource>>, String> {
    let dirp = Path::new(dir);
    let mut c = FlashNextConfig::from_checkpoint(dirp)?;
    if layers != c.layers {
        c = c.prefix(layers).ok_or("layers out of range")?;
    }
    let m = Manifest::load(Path::new(man)).map_err(|e| e.to_string())?;
    let cache = ExpertCache::new(m, slots, 8, true)?;
    let mut r = FlashNextReference::new(c, RingSource { inner: ShardSource::open(dirp)?, cache });
    r.evict_experts = true;
    Ok(r)
}

fn fn_plain(dir: &str, layers: usize) -> Result<FlashNextReference<ShardSource>, String> {
    let mut r = FlashNextReference::open(Path::new(dir), if layers == 48 { None } else { Some(layers) })?;
    r.evict_experts = true;
    Ok(r)
}

fn ring_check(dir: &str, man: &str, layers: &str, slots: &str, out: &str, ids: &[String]) -> Result<(), String> {
    let layers: usize = layers.parse().map_err(|_| "layers")?;
    let slots: usize = slots.parse().map_err(|_| "slots")?;
    let ids = ids_of(ids)?;
    let l0 = loadavg();
    let mut plain = fn_plain(dir, layers)?;
    let v = plain.config.vocab;
    let t0 = Instant::now();
    let mut st = plain.new_state();
    let (_, la) = plain.forward(&mut st, &ids, None)?;
    let wall_plain = t0.elapsed().as_secs_f64();
    let first_used: Vec<(u16, u16)> = plain.traces.iter().enumerate().map(|(l, t)| (l as u16, t.ids[0][0] as u16)).collect();
    drop(plain);

    let mut ring = fn_ring(dir, man, layers, slots)?;
    let t1 = Instant::now();
    let mut st = ring.new_state();
    let mut lb: Vec<f32> = Vec::new();
    for &t in &ids {
        let (_, lg) = ring.forward(&mut st, &[t], None)?;
        lb.extend(lg);
    }
    let wall_ring = t1.elapsed().as_secs_f64();
    let same_bits = la.len() == lb.len() && la.iter().zip(&lb).all(|(a, b)| a.to_bits() == b.to_bits());
    let max_abs = la.iter().zip(&lb).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let argmax_same = (0..ids.len()).filter(|&i| argmax(&la[i * v..(i + 1) * v]) == argmax(&lb[i * v..(i + 1) * v])).count();
    let stats = ring.source.cache.stats.clone();
    drop(ring);

    // negative control: one expert the prompt routes to gets a wrong expected digest -> fail closed
    let control = if std::env::var("WORMHOLE_CORRUPT").as_deref() == Ok("1") {
        let mut bad = fn_ring(dir, man, layers, slots)?;
        let (l, e) = first_used[0];
        bad.source.cache.corrupt_digest(l, e);
        let mut st = bad.new_state();
        let r = bad.forward(&mut st, &ids, None);
        Some(match r {
            Ok(_) => json!({"corrupted": [l, e], "result": "NOT CAUGHT (ran to the end)"}),
            Err(err) => json!({"corrupted": [l, e], "result": "caught (failed closed)", "error": err}),
        })
    } else {
        None
    };
    let caught = control.as_ref().map(|c| c["result"] == "caught (failed closed)");
    let pass = same_bits && caught != Some(false);
    let doc = json!({"what": "Flash-Next routed experts through the Heartbeat ring vs the plain checkpoint reader (CPU reference)",
        "model": dir, "manifest": man, "layers": layers, "slots": slots, "ids": ids,
        "plain": "whole prompt in one call", "ring": "one word at a time, experts from published digest-checked slots",
        "all_logits_bit_identical": same_bits, "max_abs_logit_diff": max_abs, "argmax_same_positions": argmax_same,
        "positions": ids.len(), "ring_stats": stats, "control": control, "pass": pass,
        "wall_s": {"plain": wall_plain, "ring": wall_ring}, "loadavg_start": l0, "loadavg_end": loadavg()});
    std::fs::write(out, serde_json::to_vec_pretty(&doc).unwrap()).map_err(|e| e.to_string())?;
    println!("{} ring-check: bits identical {same_bits}, max|d| {max_abs}, argmax {argmax_same}/{}, lookups {} hits {} misses {}, control {:?}",
        if pass { "PASS" } else { "FAIL" }, ids.len(), stats.lookups, stats.hits, stats.misses, caught);
    if pass { Ok(()) } else { Err("ring-check failed".into()) }
}

struct FnModel(FlashNextReference<RingSource<ShardSource>>);
impl GreedyModel for FnModel {
    type State = FlashNextState;
    fn new_state(&self) -> FlashNextState {
        self.0.new_state()
    }
    fn argmax_rows(&mut self, st: &mut FlashNextState, tokens: &[u32]) -> Result<Vec<u32>, String> {
        let v = self.0.config.vocab;
        let (_, lg) = self.0.forward(st, tokens, None)?;
        Ok(lg.chunks_exact(v).map(argmax).collect())
    }
}

struct Q38Model<S: TensorSource>(Qwen38Reference<S>, usize);
impl<S: TensorSource> GreedyModel for Q38Model<S> {
    type State = Qwen38State;
    fn new_state(&self) -> Qwen38State {
        self.0.new_state()
    }
    fn argmax_rows(&mut self, st: &mut Qwen38State, tokens: &[u32]) -> Result<Vec<u32>, String> {
        let v = self.0.config.vocab;
        self.1 += 1;
        let lg = self.0.forward(st, tokens, Logits::All)?;
        Ok(lg.chunks_exact(v).map(argmax).collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn pair(q38: &str, fnd: &str, man: &str, slots: &str, k: &str, n: &str, out: &str, ids: &[String]) -> Result<(), String> {
    let slots: usize = slots.parse().map_err(|_| "slots")?;
    let k: usize = k.parse().map_err(|_| "k")?;
    let n: usize = n.parse().map_err(|_| "n")?;
    let prompt = ids_of(ids)?;
    let fault = match std::env::var("WORMHOLE_FAULT").as_deref() {
        Ok("extra") => Fault::AcceptOneExtra,
        Ok("rowshift") => Fault::RowShift,
        _ => Fault::None,
    };
    let reverse = std::env::var("WORMHOLE_DIRECTION").as_deref() == Ok("reverse");
    let l0 = loadavg();
    let mut fm = FnModel(fn_ring(fnd, man, 48, slots)?);
    let mut qm = Q38Model(Qwen38Reference::open(Path::new(q38))?, 0);
    let t0 = Instant::now();
    let alone = if reverse { generate_alone(&mut fm, &prompt, n, &EOS)? } else { generate_alone(&mut qm, &prompt, n, &EOS)? };
    let wall_alone = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (worm, stats) = if reverse {
        generate_wormhole(&mut qm, &mut fm, &prompt, n, k, &EOS, fault)?
    } else {
        generate_wormhole(&mut fm, &mut qm, &prompt, n, k, &EOS, fault)?
    };
    let wall_worm = t1.elapsed().as_secs_f64();
    let identical = worm == alone;
    let expect_identical = fault == Fault::None;
    let pass = identical == expect_identical;
    let doc = json!({"what": "Wormhole round with real weights on the CPU references (quality and exactness only)",
        "verifier": if reverse { "Flash-Next (experts via the Heartbeat ring)" } else { "Qwen3.8-27B" },
        "drafter": if reverse { "Qwen3.8-27B" } else { "Flash-Next (experts via the Heartbeat ring)" },
        "k": k, "n": n, "slots": slots, "fault": format!("{fault:?}"), "prompt_ids": prompt,
        "verifier_alone": alone, "wormhole": worm, "identical": identical, "pass": pass,
        "stats": stats, "ring_stats": fm.0.source.cache.stats.clone(),
        "wall_s_cpu_not_a_speed_claim": {"alone": wall_alone, "wormhole": wall_worm},
        "loadavg_start": l0, "loadavg_end": loadavg()});
    std::fs::write(out, serde_json::to_vec_pretty(&doc).unwrap()).map_err(|e| e.to_string())?;
    println!("{} pair (reverse {reverse}, fault {fault:?}, k {k}): identical {identical}, rounds {} drafted {} accepted {}",
        if pass { "PASS" } else { "FAIL" }, stats.rounds, stats.drafted, stats.accepted);
    if pass { Ok(()) } else { Err("pair check failed".into()) }
}
