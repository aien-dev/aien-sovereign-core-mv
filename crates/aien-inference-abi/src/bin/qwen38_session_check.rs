// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Next-turn resume check for the Qwen3.8 serving session (sc #642), on the chip.
//!
//!   qwen38_session_check <checkpoint-dir> <layers|all> <new-tokens> <out.json>
//!
//! Two-turn chat, thinking off, greedy. Turn 2 is served three ways and compared:
//! - resumed: the session restores the snapshot taken before turn 1's last `<|im_start|>` and
//!   feeds only the new turns;
//! - cold: the session forgets everything and feeds the whole turn 2 prompt;
//! - live: turn 3 extends turn 2's prompt plus its reply tokens exactly, so the live state goes on.
//!
//! Pass: resumed == cold token for token (and live == its cold twin), the first logits close
//! (cosine >= 0.999), and both controls caught: a different second question must give different
//! tokens, and a stale state (turn 2's new tokens fed on top of turn 1's live state, the bug a
//! wrong restore would be) must give different greedy tokens than cold. `layers` below 64 runs the first layers only
//! (fits beside another model; the tokens are then not English, the comparison still holds).
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Session, Reuse};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::tokenizer::{ChatOptions, ChatTokenizer};
use serde_json::json;

const KEEP_FREE: u64 = 12 << 30;
const COSINE_MIN: f64 = 0.999;
const CHUNK: usize = 4096;

fn mem_available() -> Result<u64, String> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

/// Feeds `prompt`, then `n` greedy tokens: (ids, first logits, reuse).
fn turn(
    s: &mut Qwen38Session,
    prompt: &[u32],
    n: usize,
) -> Result<(Vec<u32>, Vec<f32>, Reuse, f64), String> {
    let t = Instant::now();
    let (mut logits, st) = s.start(prompt)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let first = logits.clone();
    let mut ids = Vec::new();
    while ids.len() < n {
        let tok = argmax(&logits);
        ids.push(tok);
        if ids.len() < n {
            logits = s.advance(tok)?.0;
        }
    }
    Ok((ids, first, st.reuse.unwrap_or(Reuse::Cold), ms))
}

fn run() -> Result<bool, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        return Err(
            "usage: qwen38_session_check <checkpoint-dir> <layers|all> <new-tokens> <out.json>"
                .into(),
        );
    }
    let dir = Path::new(&a[1]);
    let layers = if a[2] == "all" {
        None
    } else {
        Some(a[2].parse::<usize>().map_err(|e| e.to_string())?)
    };
    let n: usize = a[3]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let max_pos = 2048;
    let c = Qwen38Config::from_checkpoint(dir)?;
    let c = match layers {
        Some(l) => c.prefix(l).ok_or("layers outside the model")?,
        None => c,
    };
    let need = need_bytes(&c, max_pos, CHUNK);
    let avail = mem_available()?;
    if avail < need + KEEP_FREE {
        return Err(format!(
            "needs {} MiB + {} MiB free, MemAvailable {} MiB; refusing",
            need >> 20,
            KEEP_FREE >> 20,
            avail >> 20
        ));
    }
    eprintln!(
        "upload: {} layers, need {} MiB, MemAvailable {} MiB",
        c.layers,
        need >> 20,
        avail >> 20
    );
    let tk = ChatTokenizer::from_model_dir(dir, None).map_err(|e| e.to_string())?;
    let off = ChatOptions {
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    let enc = |turns: &[(&str, &str)]| -> Result<Vec<u32>, String> {
        let text = tk
            .try_format_chat_with(turns, &off)
            .map_err(|e| e.to_string())?;
        tk.encode_with_special(&text, true)
            .map_err(|e| e.to_string())
    };
    let q1 = "Write a C function that returns the larger of two ints.";
    let q2 = "Now make it work for three ints.";
    let q2x = "Explain what a pointer is in one sentence.";
    let p1 = enc(&[("user", q1)])?;

    let model = Qwen38Omega::open(dir, layers, max_pos)?;
    let mut s = Qwen38Session::new(model, CHUNK);

    let (a1, _, r1, ms1) = turn(&mut s, &p1, n)?;
    let reply1 = tk.decode_opts(&a1, true).map_err(|e| e.to_string())?;
    let p2 = enc(&[("user", q1), ("assistant", &reply1), ("user", q2)])?;
    let snap_len = s.snapshot_tokens().map_or(0, <[u32]>::len);

    // Control 2 (stale state): turn 2's new tokens fed on top of turn 1's live state.
    let (stale, stale_ids) = {
        let m = s.model_mut();
        let rest = &p2[snap_len..];
        let (_, logits, _) = m.prefill(rest, CHUNK)?;
        let mut ids = vec![argmax(&logits)];
        while ids.len() < n {
            m.step(*ids.last().unwrap())?;
            ids.push(argmax(&m.logits()?));
        }
        (logits, ids)
    };
    s.forget();
    // Re-run turn 1 so the snapshot exists again, then turn 2 resumed.
    let (a1b, _, _, _) = turn(&mut s, &p1, n)?;
    let (b_res, f_res, r2, ms2) = turn(&mut s, &p2, n)?;
    // Live: turn 3 = turn 2's prompt + its reply tokens fed so far + a closing and a new question.
    let mut p3 = s.history().to_vec();
    p3.extend(tk.encode_with_special("<|im_end|>\n<|im_start|>user\nThanks.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true).map_err(|e| e.to_string())?);
    let (c_live, f_live, r3, ms3) = turn(&mut s, &p3, n)?;

    s.forget();
    let (b_cold, f_cold, r2c, ms2c) = turn(&mut s, &p2, n)?;
    s.forget();
    let (c_cold, f_ccold, _, _) = turn(&mut s, &p3, n)?;
    s.forget();
    let p2x = enc(&[("user", q1), ("assistant", &reply1), ("user", q2x)])?;
    let (b_x, _, _, _) = turn(&mut s, &p2x, n)?;

    let pr = parity_stats(&f_res, &f_cold);
    let pl = parity_stats(&f_live, &f_ccold);
    let ps = parity_stats(&stale, &f_cold);
    let checks = json!({
        "turn1_repeatable": a1 == a1b,
        "turn1_cold": r1 == Reuse::Cold,
        "turn2_used_snapshot": matches!(r2, Reuse::Snapshot(k) if k == snap_len),
        "turn3_used_live": matches!(r3, Reuse::Live(_)),
        "cold_was_cold": r2c == Reuse::Cold,
        "resumed_equals_cold": b_res == b_cold,
        "resumed_cosine_ok": pr.cosine >= COSINE_MIN,
        "live_equals_cold": c_live == c_cold,
        "live_cosine_ok": pl.cosine >= COSINE_MIN,
        "control_other_question_differs": b_x != b_cold,
        "control_stale_state_caught": stale_ids != b_cold,
    });
    let pass = checks
        .as_object()
        .unwrap()
        .values()
        .all(|v| v.as_bool() == Some(true));
    let out = json!({
        "layers": c.layers, "new_tokens": n, "pass": pass, "checks": checks,
        "prompt_tokens": [p1.len(), p2.len(), p3.len()], "snapshot_tokens": snap_len,
        "reuse": [format!("{r1:?}"), format!("{r2:?}"), format!("{r3:?}"), format!("{r2c:?}")],
        "start_ms": {"turn1_cold": ms1, "turn2_resumed": ms2, "turn3_live": ms3, "turn2_cold": ms2c},
        "cosine": {"resumed_vs_cold": pr.cosine, "live_vs_cold": pl.cosine, "stale_vs_cold": ps.cosine},
        "max_abs": {"resumed_vs_cold": pr.max_abs, "live_vs_cold": pl.max_abs, "stale_vs_cold": ps.max_abs},
        "ids": {"turn1": a1, "turn2_resumed": b_res, "turn2_cold": b_cold, "turn3_live": c_live,
                "turn3_cold": c_cold, "turn2_other_question": b_x, "turn2_stale_state": stale_ids},
    });
    std::fs::write(&a[4], serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string(&json!({"pass": pass, "checks": out["checks"], "cosine": out["cosine"], "start_ms": out["start_ms"], "reuse": out["reuse"]})).unwrap());
    Ok(pass)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_session_check: {e}");
            std::process::exit(2);
        }
    }
}
