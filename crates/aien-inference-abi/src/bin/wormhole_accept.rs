//! wormhole_accept: quality-only measurements for the Wormhole (27B + Flash-Next draft and block verify),
//! design in docs/WORMHOLE.md. Host only (CPU references); no chip, no timing claims.
//!
//!   wormhole_accept fn-score  FN_DIR  OUT.json PROMPT_LEN ID...
//!     Flash-Next teacher-forced over the ids (prompt + a continuation): its greedy choice after every position,
//!     the top-two logit margin, and the routed experts of every layer for every position (the expert trace).
//!   wormhole_accept fn-gen    FN_DIR  OUT.json N ID...
//!     Flash-Next greedy continuation of the prompt ids (stops at an end mark), the routed experts of every step.
//!     Rewrites OUT.json every 8 tokens so a stopped run keeps what it made.
//!   wormhole_accept q38-score Q38_DIR OUT.json PROMPT_LEN ID...
//!     Qwen3.8-27B (host reference) teacher-forced over the ids: greedy choice and margin after every position.
//!   wormhole_accept accept    VERIFIER.json DRAFTER.json OUT.json
//!     Exact greedy block acceptance. VERIFIER has `ids` (prompt + the verifier's own greedy continuation) and
//!     `prompt_len`; DRAFTER has `ids` (the same sequence) and `argmax` (its greedy choice after every position,
//!     teacher-forced on that sequence). While a drafter's drafts match, its context IS the verifier's prefix,
//!     so its next draft equals its teacher-forced choice there: the round simulation below is exact for any
//!     autoregressive greedy drafter, for every block size, from one teacher-forced pass.
//!   wormhole_accept hitrate   OUT.json EXPERT_BYTES CAP_GB,... TRACE.json...
//!     Expert cache hit rate on the traces (fn-score or fn-gen files, in the order given), LRU and pinned-hot
//!     policies, each prompt's prefill warming the cache, hits counted on generated tokens only.
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::flashnext::FlashNextReference;
use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
use serde_json::{json, Value};

const FN_EOS: [u32; 2] = [248044, 248046];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("fn-score") if a.len() >= 6 => fn_score(&a[2], &a[3], &a[4], &a[5..]),
        Some("fn-gen") if a.len() >= 6 => fn_gen(&a[2], &a[3], &a[4], &a[5..]),
        Some("q38-score") if a.len() >= 6 => q38_score(&a[2], &a[3], &a[4], &a[5..]),
        Some("accept") if a.len() == 5 => accept(&a[2], &a[3], &a[4]),
        Some("hitrate") if a.len() >= 6 => hitrate(&a[2], &a[3], &a[4], &a[5..]),
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

fn mem_avail_kb() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}

/// greedy choice (first index of the maximum, the references' own tie rule) and top-two margin
fn choice(row: &[f32]) -> (u32, f32) {
    let mut b = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x > row[b] {
            b = i;
        }
    }
    let second = row
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != b)
        .map(|(_, &x)| x)
        .fold(f32::NEG_INFINITY, f32::max);
    (b as u32, row[b] - second)
}

fn write(out: &str, v: &Value) -> Result<(), String> {
    let tmp = format!("{out}.tmp");
    std::fs::write(&tmp, serde_json::to_vec(v).unwrap()).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, out).map_err(|e| format!("{out}: {e}"))
}

fn fn_score(dir: &str, out: &str, plen: &str, ids: &[String]) -> Result<(), String> {
    let plen: usize = plen.parse().map_err(|_| "prompt_len")?;
    let ids = ids_of(ids)?;
    let mut r = FlashNextReference::open(Path::new(dir), None)?;
    r.evict_experts = true;
    let v = r.config.vocab;
    let (l0, m0) = (loadavg(), mem_avail_kb());
    let t0 = Instant::now();
    let mut st = r.new_state();
    let (_, logits) = r.forward(&mut st, &ids, None)?;
    let wall = t0.elapsed().as_secs_f64();
    let (mut am, mut mg) = (Vec::new(), Vec::new());
    for row in logits.chunks_exact(v) {
        let (c, m) = choice(row);
        am.push(c);
        mg.push(m);
    }
    // routing[pos][layer] = the 10 routed experts
    let routing: Vec<Vec<Vec<usize>>> = (0..ids.len())
        .map(|p| r.traces.iter().map(|t| t.ids[p].clone()).collect())
        .collect();
    let hits = (plen..ids.len()).filter(|&j| am[j - 1] == ids[j]).count();
    write(
        out,
        &json!({"what": "Flash-Next CPU reference (real NVFP4), teacher-forced", "model": dir,
            "ids": ids, "prompt_len": plen, "argmax": am, "margin": mg, "routing": routing,
            "continuation_top1_hits": hits, "continuation_len": ids.len() - plen,
            "wall_s": wall, "loadavg_start": l0, "loadavg_end": loadavg(),
            "mem_avail_kb_start": m0, "mem_avail_kb_end": mem_avail_kb()}),
    )?;
    println!("PASS fn-score {} tokens, continuation top-1 {hits}/{} in {wall:.0}s", ids.len(), ids.len() - plen);
    Ok(())
}

fn fn_gen(dir: &str, out: &str, n: &str, ids: &[String]) -> Result<(), String> {
    let n: usize = n.parse().map_err(|_| "n")?;
    let prompt = ids_of(ids)?;
    let mut r = FlashNextReference::open(Path::new(dir), None)?;
    r.evict_experts = true;
    let v = r.config.vocab;
    let (l0, m0) = (loadavg(), mem_avail_kb());
    let t0 = Instant::now();
    let mut st = r.new_state();
    let (_, logits) = r.forward(&mut st, &prompt, None)?;
    let mut routing: Vec<Vec<Vec<usize>>> = (0..prompt.len())
        .map(|p| r.traces.iter().map(|t| t.ids[p].clone()).collect())
        .collect();
    let mut am: Vec<u32> = Vec::new();
    let mut mg: Vec<f32> = Vec::new();
    for row in logits.chunks_exact(v) {
        let (c, m) = choice(row);
        am.push(c);
        mg.push(m);
    }
    let prefill = t0.elapsed().as_secs_f64();
    let mut seq = prompt.clone();
    let mut step_s = Vec::new();
    let doc = |seq: &Vec<u32>, am: &Vec<u32>, mg: &Vec<f32>, routing: &Vec<Vec<Vec<usize>>>, step_s: &Vec<f64>, done: bool| {
        json!({"what": "Flash-Next CPU reference (real NVFP4), greedy generation", "model": dir,
            "ids": seq, "prompt_len": prompt.len(), "argmax": am, "margin": mg, "routing": routing,
            "finished": done, "prefill_s": prefill, "step_s": step_s,
            "loadavg_start": l0, "loadavg_now": loadavg(), "mem_avail_kb_start": m0, "mem_avail_kb_now": mem_avail_kb()})
    };
    for i in 0..n {
        let next = *am.last().unwrap();
        seq.push(next);
        if FN_EOS.contains(&next) || i + 1 == n {
            break;
        }
        let t1 = Instant::now();
        let (_, lg) = r.forward(&mut st, &[next], None)?;
        step_s.push(t1.elapsed().as_secs_f64());
        routing.push(r.traces.iter().map(|t| t.ids[0].clone()).collect());
        let (c, m) = choice(&lg);
        am.push(c);
        mg.push(m);
        if i % 8 == 7 {
            write(out, &doc(&seq, &am, &mg, &routing, &step_s, false))?;
        }
    }
    // am has one entry per position of seq except possibly the last: pad so argmax[i] = choice after seq[i]
    write(out, &doc(&seq, &am, &mg, &routing, &step_s, true))?;
    println!("PASS fn-gen {} new tokens, prefill {prefill:.0}s, {} steps", seq.len() - prompt.len(), step_s.len());
    Ok(())
}

fn q38_score(dir: &str, out: &str, plen: &str, ids: &[String]) -> Result<(), String> {
    let plen: usize = plen.parse().map_err(|_| "prompt_len")?;
    let ids = ids_of(ids)?;
    let mut m = Qwen38Reference::open(Path::new(dir))?;
    let v = m.config.vocab;
    let (l0, m0) = (loadavg(), mem_avail_kb());
    let t0 = Instant::now();
    let mut st = m.new_state();
    let logits = m.forward(&mut st, &ids, Logits::All)?;
    let wall = t0.elapsed().as_secs_f64();
    let (mut am, mut mg) = (Vec::new(), Vec::new());
    for row in logits.chunks_exact(v) {
        let (c, g) = choice(row);
        am.push(c);
        mg.push(g);
    }
    let hits = (plen..ids.len()).filter(|&j| am[j - 1] == ids[j]).count();
    write(
        out,
        &json!({"what": "Qwen3.8-27B host reference (FP8 checkpoint), teacher-forced", "model": dir,
            "ids": ids, "prompt_len": plen, "argmax": am, "margin": mg,
            "continuation_top1_hits": hits, "continuation_len": ids.len() - plen,
            "wall_s": wall, "loadavg_start": l0, "loadavg_end": loadavg(),
            "mem_avail_kb_start": m0, "mem_avail_kb_end": mem_avail_kb()}),
    )?;
    println!("PASS q38-score {} tokens, continuation top-1 {hits}/{} in {wall:.0}s", ids.len(), ids.len() - plen);
    Ok(())
}

fn read_json(p: &str) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))
}

fn u32s(v: &Value, k: &str) -> Result<Vec<u32>, String> {
    v[k].as_array()
        .ok_or(format!("missing {k}"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as u32).ok_or(format!("{k}: not an id")))
        .collect()
}

/// Greedy block rounds: from position p the drafter proposes min(k, left) tokens; the verifier keeps the
/// matching prefix m and adds its own token (the correction, or the bonus after a full match).
pub fn rounds(ids: &[u32], am: &[u32], plen: usize, k: usize) -> (usize, usize, usize, usize) {
    let (mut p, mut rounds, mut drafted, mut accepted) = (plen, 0, 0, 0);
    let end = ids.len();
    while p < end {
        let left = end - p;
        let kk = k.min(left);
        let mut m = 0;
        while m < kk && am[p + m - 1] == ids[p + m] {
            m += 1;
        }
        rounds += 1;
        drafted += kk;
        accepted += m;
        p += (m + 1).min(left);
    }
    (rounds, drafted, accepted, end - plen)
}

fn accept(ver: &str, dr: &str, out: &str) -> Result<(), String> {
    let (vj, dj) = (read_json(ver)?, read_json(dr)?);
    let ids = u32s(&vj, "ids")?;
    let plen = vj["prompt_len"].as_u64().ok_or("prompt_len")? as usize;
    let dids = u32s(&dj, "ids")?;
    let am = u32s(&dj, "argmax")?;
    if dids != ids {
        return Err("drafter file was not scored on the verifier's sequence".into());
    }
    if am.len() + 1 < ids.len() {
        return Err("drafter argmax shorter than the sequence".into());
    }
    let mut rows = Vec::new();
    for k in [1usize, 2, 3, 4, 6, 8, 12, 16] {
        let (r, d, a, n) = rounds(&ids, &am, plen, k);
        rows.push(json!({"k": k, "rounds": r, "drafted": d, "accepted": a, "tokens": n,
            "accept_rate": a as f64 / d as f64, "tokens_per_round": n as f64 / r as f64}));
        println!("k {k:2}: rounds {r:4} drafted {d:5} accepted {a:5} rate {:.4} tokens/round {:.3}", a as f64 / d as f64, n as f64 / r as f64);
    }
    let top1 = (plen..ids.len()).filter(|&j| am[j - 1] == ids[j]).count();
    write(out, &json!({"verifier_file": ver, "drafter_file": dr, "prompt_len": plen, "tokens": ids.len() - plen,
        "drafter_top1_on_verifier_text": top1, "blocks": rows}))?;
    Ok(())
}

struct Lru {
    cap: usize,
    tick: u64,
    last: HashMap<(u16, u16), u64>,
    order: VecDeque<((u16, u16), u64)>,
}
impl Lru {
    fn new(cap: usize) -> Self {
        Self { cap, tick: 0, last: HashMap::new(), order: VecDeque::new() }
    }
    fn touch(&mut self, k: (u16, u16)) -> bool {
        self.tick += 1;
        let hit = self.last.contains_key(&k);
        self.last.insert(k, self.tick);
        self.order.push_back((k, self.tick));
        while self.last.len() > self.cap {
            let (old, t) = self.order.pop_front().unwrap();
            if self.last.get(&old) == Some(&t) {
                self.last.remove(&old);
            }
        }
        if self.order.len() > 8 * self.cap.max(1024) {
            let last = &self.last;
            self.order.retain(|(k, t)| last.get(k) == Some(t));
        }
        hit
    }
}

fn hitrate(out: &str, ebytes: &str, caps: &str, files: &[String]) -> Result<(), String> {
    let eb: f64 = ebytes.parse().map_err(|_| "EXPERT_BYTES")?;
    let caps: Vec<f64> = caps.split(',').map(|s| s.parse().map_err(|_| "CAP_GB")).collect::<Result<_, _>>()?;
    // traces: per file, (prompt_len, routing[pos][layer][10])
    let mut traces = Vec::new();
    for f in files {
        let j = read_json(f)?;
        let plen = j["prompt_len"].as_u64().ok_or("prompt_len")? as usize;
        let rt: Vec<Vec<Vec<u16>>> = serde_json::from_value(j["routing"].clone()).map_err(|e| e.to_string())?;
        traces.push((f.clone(), plen, rt));
    }
    let mut res = Vec::new();
    // distinct experts per block of k generated tokens (what one verify or draft block must have resident)
    let mut blocks = Vec::new();
    for k in [1usize, 4, 8, 16] {
        let (mut sum, mut cnt) = (0usize, 0usize);
        for (_, plen, rt) in &traces {
            let gen = &rt[*plen..];
            for ch in gen.chunks(k) {
                if ch.len() < k {
                    continue;
                }
                let s: HashSet<(usize, u16)> =
                    ch.iter().flat_map(|pos| pos.iter().enumerate().flat_map(|(l, ex)| ex.iter().map(move |&e| (l, e)))).collect();
                sum += s.len();
                cnt += 1;
            }
        }
        blocks.push(json!({"k": k, "blocks": cnt, "mean_distinct_experts": sum as f64 / cnt.max(1) as f64,
            "mean_bytes": sum as f64 / cnt.max(1) as f64 * eb}));
    }
    for &gb in &caps {
        let cap = ((gb * 1e9) / eb).floor() as usize;
        // LRU, cold at the first prompt, warm across prompts in the order given
        let mut lru = Lru::new(cap);
        let (mut h, mut n) = (0usize, 0usize);
        let mut per = Vec::new();
        for (f, plen, rt) in &traces {
            let (mut ph, mut pn) = (0usize, 0usize);
            for (pos, layers) in rt.iter().enumerate() {
                for (l, ex) in layers.iter().enumerate() {
                    for &e in ex {
                        let hit = lru.touch((l as u16, e));
                        if pos >= *plen {
                            ph += hit as usize;
                            pn += 1;
                        }
                    }
                }
            }
            h += ph;
            n += pn;
            per.push(json!({"file": f, "lru_hits": ph, "lookups": pn, "lru_hit_rate": ph as f64 / pn.max(1) as f64}));
        }
        // pinned hot: per held-out prompt, pin the cap most-used experts of the OTHER traces (no peeking), plus
        // LRU over the rest of the room is not modelled (pinned only, the cautious number)
        let mut pin_h = 0usize;
        let mut pin_n = 0usize;
        for (i, (_, plen, rt)) in traces.iter().enumerate() {
            let mut freq: HashMap<(u16, u16), usize> = HashMap::new();
            for (j, (_, _, rt2)) in traces.iter().enumerate() {
                if j == i {
                    continue;
                }
                for layers in rt2 {
                    for (l, ex) in layers.iter().enumerate() {
                        for &e in ex {
                            *freq.entry((l as u16, e)).or_default() += 1;
                        }
                    }
                }
            }
            let mut f: Vec<_> = freq.into_iter().collect();
            f.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let pinned: HashSet<(u16, u16)> = f.into_iter().take(cap).map(|x| x.0).collect();
            for layers in &rt[*plen..] {
                for (l, ex) in layers.iter().enumerate() {
                    for &e in ex {
                        pin_h += pinned.contains(&(l as u16, e)) as usize;
                        pin_n += 1;
                    }
                }
            }
        }
        let total_experts = 48 * 512;
        res.push(json!({"cache_gb": gb, "cache_experts": cap, "share_of_all_experts": cap as f64 / total_experts as f64,
            "lru_hit_rate_warm_across_prompts": h as f64 / n.max(1) as f64, "lookups": n, "per_prompt": per,
            "pinned_hot_heldout_hit_rate": pin_h as f64 / pin_n.max(1) as f64}));
        println!("cache {gb} GB = {cap} experts: LRU hit {:.4}, pinned-hot (held out) {:.4}", h as f64 / n.max(1) as f64, pin_h as f64 / pin_n.max(1) as f64);
    }
    write(out, &json!({"expert_bytes": eb, "files": files, "distinct_per_block": blocks, "caches": res}))
}

#[cfg(test)]
mod tests {
    use super::rounds;
    #[test]
    fn rounds_exact_on_a_hand_case() {
        // prompt [9], continuation 1..=6; drafter right on 1,2, wrong on 3, right on 4,5,6
        let ids = [9u32, 1, 2, 3, 4, 5, 6];
        let am = [1u32, 2, 7, 4, 5, 6, 0];
        // k 2: round1 drafts 1,2 (both ok) +bonus 3 -> p=4; round2 drafts 4,5 ok +bonus 6 -> end
        assert_eq!(rounds(&ids, &am, 1, 2), (2, 4, 4, 6));
        // k 4: round1 drafts 1,2,3x -> m 2, +3 -> p 4; round2 drafts 4,5,6 (3 left) all ok -> end
        assert_eq!(rounds(&ids, &am, 1, 4), (2, 7, 5, 6));
        // a drafter that is always wrong: one token per round
        let bad = [0u32; 7];
        assert_eq!(rounds(&ids, &bad, 1, 4), (6, 4 + 4 + 4 + 3 + 2 + 1, 0, 6));
    }
}
