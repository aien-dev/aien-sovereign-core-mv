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

//! `q36_mtp_check <checkpoint> <out.json> <golden.json>...` (DRA-70, sc#866). Host only, no chip.
//!
//! Per golden (`prompt_ids` + `generated_ids` from `qwen38_check`): one teacher-forced target pass
//! over the whole sequence gives the target argmax G[i] and post-final-norm hidden H[i]; one
//! teacher-forced MTP pass over (H[i], S[i + 1]) gives the depth-1 draft D[i] for S[i + 2].
//! Depth-1 acceptance over the generated region counts D[i] == G[i + 1]. Also checks that the
//! target argmax reproduces the golden (sanity) and times one single-row MTP draft step (the CPU
//! reference streams the routed experts from the shards, so this cost includes that read).

use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen38::mtp::MtpState;
use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Reference};

fn ids(v: &serde_json::Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or(format!("no {key}"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as u32).ok_or(format!("bad {key}")))
        .collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("q36_mtp_check: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: q36_mtp_check <checkpoint> <out.json> <golden.json>...".into());
    }
    let (ckpt, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    let t0 = Instant::now();
    let mut r = Qwen38Reference::open(&ckpt)?;
    let w = r.read_mtp()?;
    let open_s = t0.elapsed().as_secs_f64();
    let (layers, h, vocab) = (r.config.layers, r.config.hidden, r.config.vocab);
    let top_k = r.config.moe.as_ref().map_or(0, |m| m.top_k);
    let mut items = Vec::new();
    let (mut acc_all, mut n_all) = (0usize, 0usize);
    for g in &args[2..] {
        let text = std::fs::read_to_string(g).map_err(|e| format!("{g}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{g}: {e}"))?;
        let (prompt, gen) = (ids(&v, "prompt_ids")?, ids(&v, "generated_ids")?);
        let seq: Vec<u32> = prompt.iter().chain(&gen).copied().collect();
        let (pl, l) = (prompt.len(), seq.len());
        let t1 = Instant::now();
        let mut st = r.new_state();
        let mut taps = Vec::new();
        let logits = r.forward_tapped(&mut st, &seq, Logits::All, &[layers], &mut taps)?;
        let target_s = t1.elapsed().as_secs_f64();
        let gt: Vec<u32> = logits.chunks_exact(vocab).map(|x| argmax(x) as u32).collect();
        // Sanity: the target argmax after S[..pl + j] is the golden's generated token j.
        let golden_match = (0..gen.len()).filter(|&j| gt[pl - 1 + j] == gen[j]).count();
        let hid = r.final_hidden(&taps[0]);
        // MTP rows i = 0..l-1 pair H[i] with S[i + 1]; the last row has no next token.
        let t2 = Instant::now();
        let mut ms = MtpState::default();
        let rows = r.mtp_forward(&w, &mut ms, &hid[..(l - 1) * h], &seq[1..])?;
        let mtp_s = t2.elapsed().as_secs_f64();
        let dl = r.lm_head().matmul(&rows, l - 1);
        let d: Vec<u32> = dl.chunks_exact(vocab).map(|x| argmax(x) as u32).collect();
        // D[i] drafts S[i + 2]; the target's own token there is G[i + 1]. Generated region:
        // draft positions where S[i + 2] is a generated token, i + 2 >= pl.
        let (mut acc, mut n) = (0usize, 0usize);
        for i in pl.saturating_sub(2)..l - 2 {
            n += 1;
            acc += usize::from(d[i] == gt[i + 1]);
        }
        // One single-row draft step after the cache holds l - 1 positions (rows the real
        // decode would see): MTP layer + full bf16 head.
        let mut one = ms.clone();
        one.truncate(l - 2, r.config.kv_dim());
        let t3 = Instant::now();
        let row = r.mtp_forward(&w, &mut one, &hid[(l - 2) * h..(l - 1) * h], &seq[l - 1..])?;
        let step_layer_s = t3.elapsed().as_secs_f64();
        let t4 = Instant::now();
        let _ = r.lm_head().matmul(&row, 1);
        let step_head_s = t4.elapsed().as_secs_f64();
        acc_all += acc;
        n_all += n;
        eprintln!(
            "{g}: depth-1 {acc}/{n}, golden {golden_match}/{}, target {target_s:.1}s, mtp {mtp_s:.1}s, step {:.0}+{:.0} ms",
            gen.len(),
            step_layer_s * 1e3,
            step_head_s * 1e3
        );
        items.push(serde_json::json!({
            "golden": g, "prompt_len": pl, "generated": gen.len(),
            "golden_match": golden_match, "depth1_accepted": acc, "depth1_positions": n,
            "target_seconds": target_s, "mtp_seconds": mtp_s,
            "draft_step_layer_ms": step_layer_s * 1e3, "draft_step_head_ms": step_head_s * 1e3,
            "drafts": d, "target_argmax": gt,
        }));
    }
    let report = serde_json::json!({
        "tool": "q36_mtp_check", "checkpoint": ckpt.display().to_string(),
        "open_seconds": open_s,
        "mtp_draft_bytes_per_token": w.draft_bytes(top_k, 3_146_112),
        "depth1_accepted": acc_all, "depth1_positions": n_all,
        "depth1_rate": acc_all as f64 / n_all.max(1) as f64,
        "items": items,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "depth-1 acceptance {acc_all}/{n_all} = {:.3}",
        acc_all as f64 / n_all.max(1) as f64
    );
    Ok(())
}
