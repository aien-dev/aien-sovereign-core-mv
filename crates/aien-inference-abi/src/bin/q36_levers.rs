//! Speed levers on the served Qwen3.6-35B-A3B decode graph (sc L356): identity and timing in one run.
//!
//!   q36_levers <checkpoint-dir> <out.json> <golden.json>...
//!
//! 1. Identity: every golden (`qwen38_check generate` record: `prompt_ids`, `generated_ids`) is fed
//!    token by token through the decode graph and decoded greedily for as many tokens as it holds.
//!    The chip's ids must equal the golden's; every decoded step's logits are also hashed (FNV-1a
//!    over the raw f32 bits) so two builds can be held to bit-identical logits, not only ids.
//! 2. Timing: the first golden's prompt, then greedy decoding up to `Q36_LEVERS_POS` (default 1100)
//!    with no logits read. Chip and host time per token, averaged per position band. Tokens at the
//!    positions in `Q36_LEVERS_STAMPS` (default 40,300,1000) run with the GPU timeline on and are
//!    left out of the averages: their launches are summed per kind and set against the bytes each
//!    kind moves at `BW` (the 228 GB/s stream rate), the gap a lever can close.
//!
//! Exit 0 when every golden is identical, 1 otherwise.
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega};
use aien_omega_q38::q36;
use serde_json::{json, Value};

const KEEP_FREE: u64 = 12 << 30;
/// Stream rate the byte times are taken at (GB/s, MACHINE_CEILING).
const BW: f64 = 228e9;

fn mem_available() -> Result<u64, String> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

fn fnv(l: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in l {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn ids(v: &Value, k: &str) -> Result<Vec<u32>, String> {
    v[k].as_array()
        .ok_or(format!("golden: no {k}"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as u32).ok_or(format!("golden: bad {k}")))
        .collect()
}

/// Bytes one launch kind moves per token at position `pos` (weights, state, KV; glue kernels
/// counted as their inputs plus outputs in f32). From the shapes in omega_gpu_q36_graph.h.
fn kind_bytes(c: &Qwen38Config, kind: &str, pos: usize) -> f64 {
    let h = c.hidden as f64;
    let gdn = c.full_attention.iter().filter(|a| !**a).count() as f64;
    let attn = c.layers as f64 - gdn;
    let mixer = q36::MIXER_OUT as f64;
    match kind {
        "mix" => gdn * q36::GDN_MIX as f64 * h + attn * q36::ATT_MIX as f64 * h,
        "ab" => gdn * q36::AB as f64 * h * 2.0,
        // DeltaNet state f32 [32 v heads][128][128], read and written.
        "gdn" => gdn * 2.0 * 32.0 * 128.0 * 128.0 * 4.0,
        // K and V, bf16, 2 kv heads x 256, positions 0..=pos.
        "attn" => attn * 2.0 * q36::KV_HEADS as f64 * 256.0 * 2.0 * (pos + 1) as f64,
        "o_proj" => c.layers as f64 * h * mixer,
        "router" => c.layers as f64 * q36::EXPERT_ROWS as f64 * h * 2.0,
        "gate_up" => c.layers as f64 * q36::GROUPS as f64 * 2.0 * q36::EXPERT_INTER as f64 * h,
        "down" => c.layers as f64 * q36::GROUPS as f64 * h * q36::EXPERT_INTER as f64,
        "lm_head" => c.vocab as f64 * h * 2.0,
        _ => 0.0,
    }
}

/// Env vars worth recording in the evidence JSON: the AIEN_/OMEGA/Q36_ ones, minus anything whose name
/// looks like a secret (KEY, TOKEN, SECRET, PASS, CRED, any case). Evidence files get committed.
fn evidence_env(vars: impl Iterator<Item = (String, String)>) -> Vec<(String, String)> {
    const SECRET_WORDS: [&str; 5] = ["KEY", "TOKEN", "SECRET", "PASS", "CRED"];
    vars.filter(|(k, _)| k.starts_with("AIEN_") || k.starts_with("OMEGA") || k.starts_with("Q36_"))
        .filter(|(k, _)| {
            let up = k.to_ascii_uppercase();
            !SECRET_WORDS.iter().any(|w| up.contains(w))
        })
        .collect()
}

fn run() -> Result<bool, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: q36_levers <checkpoint-dir> <out.json> <golden.json>...".into());
    }
    let dir = Path::new(&a[1]);
    let env_list = |k: &str, d: &str| -> Vec<usize> {
        std::env::var(k)
            .unwrap_or_else(|_| d.into())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    };
    let end_pos = env_list("Q36_LEVERS_POS", "1100")[0];
    let stamp_at = env_list("Q36_LEVERS_STAMPS", "40,300,1000");
    let max_pos = 2048.max(end_pos + 1);
    let c = Qwen38Config::from_checkpoint(dir)?;
    let need = need_bytes(&c, max_pos, 4096);
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "needs {} MiB + {} MiB free, MemAvailable {} MiB; refusing",
            need >> 20,
            KEEP_FREE >> 20,
            avail >> 20
        ));
    }
    let env = evidence_env(std::env::vars());
    eprintln!("upload: need {} MiB, MemAvailable {} MiB, env {env:?}", need >> 20, avail >> 20);
    let t = Instant::now();
    let mut m = Qwen38Omega::open(dir, None, max_pos)?;
    let open_s = t.elapsed().as_secs_f64();
    if !m.is_moe() {
        return Err("not the 35B (sparse) model".into());
    }

    // 1. identity
    let mut all_ok = true;
    let mut goldens = Vec::new();
    let mut first_prompt = Vec::new();
    for gpath in &a[3..] {
        let g: Value = serde_json::from_str(&std::fs::read_to_string(gpath).map_err(|e| format!("{gpath}: {e}"))?)
            .map_err(|e| format!("{gpath}: {e}"))?;
        let (prompt, want) = (ids(&g, "prompt_ids")?, ids(&g, "generated_ids")?);
        if first_prompt.is_empty() {
            first_prompt = prompt.clone();
        }
        m.reset()?;
        let mut tok = 0;
        for &p in &prompt {
            tok = m.step(p)?.0;
        }
        let mut got = vec![tok];
        let mut hashes = vec![format!("{:016x}", fnv(&m.logits()?))];
        while got.len() < want.len() {
            tok = m.step(tok)?.0;
            got.push(tok);
            hashes.push(format!("{:016x}", fnv(&m.logits()?)));
        }
        let ok = got == want;
        all_ok &= ok;
        let first_diff = got.iter().zip(&want).position(|(x, y)| x != y);
        eprintln!("identity {gpath}: {} ({} tokens)", if ok { "IDENTICAL" } else { "DIFFER" }, want.len());
        goldens.push(json!({"golden": gpath, "identical": ok, "first_diff": first_diff, "got": got, "logits_fnv": hashes}));
    }

    // 2. timing
    m.reset()?;
    let mut tok = 0;
    for &p in &first_prompt {
        tok = m.step(p)?.0;
    }
    let bands: [(usize, usize); 4] = [(0, 64), (64, 288), (288, 800), (800, usize::MAX)];
    let mut sums = vec![(0u64, 0u64, 0u64, 0usize); bands.len()]; // chip, host total, wall, n
    let mut stamped = Vec::new();
    let mut pos = first_prompt.len();
    while pos < end_pos {
        let stamp = stamp_at.contains(&pos);
        if stamp {
            m.q36_graph_mut().ok_or("no q36 graph")?.set_stamps(true);
        }
        let t = Instant::now();
        let (next, st) = m.step(tok)?;
        let wall = t.elapsed().as_nanos() as u64;
        if stamp {
            let g = m.q36_graph_mut().ok_or("no q36 graph")?;
            let (s, k) = g.stamps().map_err(|e| format!("stamps: {e:?}"))?;
            g.set_stamps(false);
            let mut per: Vec<(String, u64, u32)> = Vec::new();
            for (i, kind) in k.iter().enumerate() {
                let name = q36::kind_name(*kind);
                let d = s[i + 1].saturating_sub(s[i]);
                match per.iter_mut().find(|e| e.0 == name) {
                    Some(e) => {
                        e.1 += d;
                        e.2 += 1
                    }
                    None => per.push((name, d, 1)),
                }
            }
            let span = s[s.len() - 1] - s[0];
            let mut rows = Vec::new();
            let mut byte_total = 0.0;
            eprintln!("timeline pos {pos}: {} launches, span {:.1} us, chip {:.1} us", k.len(), span as f64 / 1e3, st.chip_ns as f64 / 1e3);
            for (name, ns, n) in &per {
                let b = kind_bytes(&c, name, pos);
                byte_total += b;
                let byte_us = b / BW * 1e6;
                eprintln!(
                    "  {name:>8}: {n:4} launches {:9.1} us ({:5.1}%), {:8.2} MB, byte time {:8.1} us, gap {:8.1} us, {:6.2} us/launch",
                    *ns as f64 / 1e3,
                    *ns as f64 * 100.0 / span as f64,
                    b / 1e6,
                    byte_us,
                    *ns as f64 / 1e3 - byte_us,
                    *ns as f64 / 1e3 / *n as f64
                );
                rows.push(json!({"kind": name, "launches": n, "ns": ns, "bytes": b, "byte_ns": b / BW * 1e9}));
            }
            eprintln!("  total bytes {:.1} MB, byte time {:.1} us", byte_total / 1e6, byte_total / BW * 1e6);
            stamped.push(json!({"pos": pos, "launches": k.len(), "span_ns": span, "chip_ns": st.chip_ns, "wall_ns": wall,
                                "bytes": byte_total, "byte_ns": byte_total / BW * 1e9, "kinds": rows}));
        } else {
            let b = bands.iter().position(|&(lo, hi)| pos >= lo && pos < hi).unwrap_or(0);
            let e = &mut sums[b];
            e.0 += st.chip_ns;
            e.1 += st.total_ns;
            e.2 += wall;
            e.3 += 1;
        }
        tok = next;
        pos += 1;
    }
    let band_rows: Vec<Value> = bands
        .iter()
        .zip(&sums)
        .filter(|(_, s)| s.3 > 0)
        .map(|(&(lo, hi), &(chip, tot, wall, n))| {
            let n64 = n as f64;
            eprintln!(
                "band pos [{lo}, {}): {n} tokens, chip {:.3} ms, graph total {:.3} ms, wall {:.3} ms, {:.2} tok/s (wall)",
                if hi == usize::MAX { end_pos } else { hi },
                chip as f64 / n64 / 1e6,
                tot as f64 / n64 / 1e6,
                wall as f64 / n64 / 1e6,
                n64 * 1e9 / wall as f64
            );
            json!({"lo": lo, "hi": hi.min(end_pos), "tokens": n, "chip_ms": chip as f64 / n64 / 1e6,
                   "graph_total_ms": tot as f64 / n64 / 1e6, "wall_ms": wall as f64 / n64 / 1e6, "tok_s": n64 * 1e9 / wall as f64})
        })
        .collect();
    let out = json!({
        "tool": "q36_levers", "model": a[1], "open_s": open_s, "env": env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
        "all_identical": all_ok, "goldens": goldens, "bands": band_rows, "timelines": stamped,
    });
    std::fs::write(&a[2], serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    eprintln!("identity: {}", if all_ok { "ALL IDENTICAL" } else { "DIFFER" });
    Ok(all_ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("q36_levers: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::evidence_env;

    fn kv(k: &str) -> (String, String) {
        (k.to_string(), "v".to_string())
    }

    #[test]
    fn secret_looking_names_are_dropped_and_the_rest_kept() {
        let input = [
            "AIEN_COORD_KEY", "AIEN_api_token", "OMEGA_SECRET_X", "Q36_PASSWORD", "AIEN_DB_CRED",
            "AIEN_PASSPHRASE", "AIEN_GRAPH", "OMEGA_FLAGS", "Q36_LEVERS_POS", "HOME", "PATH",
        ];
        let kept: Vec<String> = evidence_env(input.iter().map(|k| kv(k))).into_iter().map(|(k, _)| k).collect();
        assert_eq!(kept, ["AIEN_GRAPH", "OMEGA_FLAGS", "Q36_LEVERS_POS"]);
    }
}
