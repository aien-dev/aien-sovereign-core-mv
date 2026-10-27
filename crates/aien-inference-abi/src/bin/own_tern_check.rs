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

//! OWN-TERN check, stage T2: the sealed golden of our CPU reference for BitNet b1.58 2B-4T.
//!
//!   own_tern_check golden <checkpoint-dir> <passage.txt|passage.json> <out-dir>
//!
//! Host only (no chip). Loads the bf16 master with our loader into i8 trit matrices, then:
//! - teacher-forced loss and perplexity over the whole passage (BOS first), for the reference and
//!   the three negative controls (wrong norm order, skipped sub-norms, q and v scales swapped);
//! - greedy continuations (32 tokens) of our five prompts, and of the spec's five acceptance
//!   prompts for information;
//! - hidden-state dumps after layers 0, 1, 15, 29 and after the final norm, plus the last
//!   position's logits, for every prompt and for a 512-token prompt (BOS and the passage's
//!   first 511 tokens).
//!
//! Writes `<out-dir>/golden.json` (ids, perplexity receipt, decode timing, dump sha256s),
//! `<out-dir>/passage-nll-<variant>.f64` (per-token losses, little-endian f64) and
//! `<out-dir>/dumps/*.f32` (little-endian f32, `[tokens][hidden]`; logits `[vocab]`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use aien_inference_abi::bitnet_forward::{
    argmax, HostBitNetForward, HostRefUploader, KvCache, Variant,
};
use aien_inference_abi::bitnet_omega::{
    load_bitnet_omega, sha256_file, BITNET_2B4T_SAFETENSORS_SHA256, BITNET_2B4T_TOKENIZER_SHA256,
    TERN_LAYOUT_ID,
};
use aien_inference_abi::bitnet_tokenizer::BitNetTokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const OWN_PROMPTS: [(&str, &str); 5] = [
    (
        "own1-flat-tyre",
        "The first thing to do when a bicycle tyre goes flat is",
    ),
    (
        "own2-mountain-water",
        "Water boils at a lower temperature high on a mountain because",
    ),
    (
        "own3-bread",
        "A simple loaf of bread needs only four ingredients:",
    ),
    ("own4-light", "The three primary colours of light are"),
    (
        "own5-lighthouse",
        "Once upon a time, an old lighthouse keeper",
    ),
];

/// Spec section 6 acceptance prompts: recorded for information, not a gate.
const SPEC_PROMPTS: [(&str, &str); 5] = [
    ("spec1-france", "The capital of France is"),
    ("spec2-germany", "The capital of Germany is"),
    ("spec3-count", "Count from 6 to 10:"),
    ("spec4-boiling", "The boiling point of water in Celsius is"),
    ("spec5-planet", "The largest planet in the solar system is"),
];

const TAP_LAYERS: [usize; 4] = [0, 1, 15, 29];
const NEW_TOKENS: usize = 32;
const LONG_PROMPT: usize = 512;
const MIN_PASSAGE_TOKENS: usize = 2000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = match args.get(1).map(String::as_str) {
        Some("golden") if args.len() == 5 => golden(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        ),
        _ => Err(
            "usage: own_tern_check golden <checkpoint-dir> <passage.txt|passage.json> <out-dir>"
                .into(),
        ),
    };
    if let Err(e) = r {
        eprintln!("own_tern_check: {e}");
        std::process::exit(1);
    }
}

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_f32(path: &Path, v: &[f32]) -> Result<String, String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha_hex(&bytes))
}

fn golden(ckpt: &Path, passage_path: &Path, out: &Path) -> Result<(), String> {
    let t_all = Instant::now();
    let dumps = out.join("dumps");
    std::fs::create_dir_all(&dumps).map_err(|e| format!("{}: {e}", dumps.display()))?;

    for (file, want) in [
        ("model.safetensors", BITNET_2B4T_SAFETENSORS_SHA256),
        ("tokenizer.json", BITNET_2B4T_TOKENIZER_SHA256),
    ] {
        let got = sha256_file(&ckpt.join(file))?;
        if got != want {
            return Err(format!("{file} sha256 {got} is not the pinned {want}"));
        }
    }
    let tok = BitNetTokenizer::from_file(&ckpt.join("tokenizer.json"))?;
    let passage = read_passage(passage_path)?;
    let passage_sha = sha_hex(passage.as_bytes());
    let ids = tok.encode(&passage, true);
    if ids.len() - 1 < MIN_PASSAGE_TOKENS {
        return Err(format!(
            "passage has {} tokens, the gate needs at least {MIN_PASSAGE_TOKENS}",
            ids.len() - 1
        ));
    }
    if tok.decode(&ids[1..])? != passage {
        return Err("passage does not round-trip through the tokenizer".into());
    }
    eprintln!(
        "passage: {} tokens after BOS, sha256 {passage_sha}",
        ids.len() - 1
    );

    let t_load = Instant::now();
    let (w, report) = load_bitnet_omega(ckpt, &HostRefUploader, &mut |m| eprintln!("load: {m}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let eos = w.config.eos_token_ids.clone();
    let mut f = HostBitNetForward::new(&w);
    let h = w.config.hidden;

    // Perplexity: row i predicts ids[i + 1].
    let mut ppl = serde_json::Map::new();
    for v in [
        Variant::Reference,
        Variant::SubNormAfterO,
        Variant::SkipSubNorms,
        Variant::SwapQvScales,
    ] {
        f.variant = v;
        let t0 = Instant::now();
        let normed = f.forward(&ids, &mut KvCache::new(&w.config), &[], &mut Vec::new())?;
        let n = ids.len() - 1;
        let nll = f.nll(&normed[..n * h], &ids[1..])?;
        let sum: f64 = nll.iter().sum();
        let bytes: Vec<u8> = nll.iter().flat_map(|x| x.to_le_bytes()).collect();
        let file = format!("passage-nll-{}.f64", v.name());
        std::fs::write(out.join(&file), &bytes).map_err(|e| format!("{file}: {e}"))?;
        let mean = sum / n as f64;
        eprintln!(
            "{}: {n} tokens, sum nll {sum:.17e}, perplexity {:.6} ({:.1} s)",
            v.name(),
            mean.exp(),
            t0.elapsed().as_secs_f64()
        );
        ppl.insert(
            v.name().into(),
            json!({
                "predicted_tokens": n,
                "sum_nll_nats": format!("{sum:.17e}"),
                "mean_nll_nats": format!("{mean:.17e}"),
                "perplexity": format!("{:.9}", mean.exp()),
                "nll_file": file,
                "nll_sha256": sha_hex(&bytes),
                "seconds": t0.elapsed().as_secs_f64(),
            }),
        );
    }
    f.variant = Variant::Reference;

    // Prompts: dumps, then greedy decode with the KV cache.
    let mut prompts = Vec::new();
    let mut step_ms = Vec::new();
    let mut texts: Vec<(String, String, Vec<u32>, bool)> = OWN_PROMPTS
        .iter()
        .map(|(n, t)| (n.to_string(), t.to_string(), tok.encode(t, true), true))
        .collect();
    texts.extend(
        SPEC_PROMPTS
            .iter()
            .map(|(n, t)| (n.to_string(), t.to_string(), tok.encode(t, true), true)),
    );
    texts.push((
        format!("long{LONG_PROMPT}-passage-head"),
        format!("BOS and the first {} passage tokens", LONG_PROMPT - 1),
        ids[..LONG_PROMPT].to_vec(),
        false,
    ));
    for (name, text, pids, generate) in texts {
        let mut cache = KvCache::new(&w.config);
        let mut taps = Vec::new();
        let normed = f.forward(&pids, &mut cache, &TAP_LAYERS, &mut taps)?;
        let t = pids.len();
        let mut files = Vec::new();
        for (layer, rows) in &taps {
            let file = format!("{name}.layer{layer:02}.f32");
            let sha = write_f32(&dumps.join(&file), rows)?;
            files.push(json!({"tap": format!("layer{layer:02}"), "file": file, "shape": [t, h], "sha256": sha}));
        }
        let file = format!("{name}.final-norm.f32");
        let sha = write_f32(&dumps.join(&file), &normed)?;
        files.push(json!({"tap": "final_norm", "file": file, "shape": [t, h], "sha256": sha}));
        let last = f.logits(&normed[(t - 1) * h..]);
        let file = format!("{name}.last-logits.f32");
        let sha = write_f32(&dumps.join(&file), &last)?;
        files.push(
            json!({"tap": "last_logits", "file": file, "shape": [w.config.vocab], "sha256": sha}),
        );

        let mut entry = json!({"name": name, "text": text, "ids": pids, "dumps": files});
        if generate {
            let mut next = argmax(&last);
            let mut gen = Vec::new();
            while gen.len() < NEW_TOKENS {
                gen.push(next);
                if eos.contains(&next) || gen.len() == NEW_TOKENS {
                    break;
                }
                let t0 = Instant::now();
                let row = f.forward(&[next], &mut cache, &[], &mut Vec::new())?;
                next = argmax(&f.logits(&row));
                step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let cont = tok.decode(&gen)?;
            eprintln!("{name}: {text:?} -> {cont:?}");
            entry["greedy_ids"] = json!(gen);
            entry["greedy_text"] = json!(cont);
        }
        prompts.push(entry);
    }
    step_ms.sort_by(f64::total_cmp);
    let median = step_ms.get(step_ms.len() / 2).copied().unwrap_or(0.0);

    let golden: Value = json!({
        "stage": "bitnet-t2-host-reference",
        "layout_id": TERN_LAYOUT_ID,
        "model_sha256": BITNET_2B4T_SAFETENSORS_SHA256,
        "tokenizer_sha256": BITNET_2B4T_TOKENIZER_SHA256,
        "arithmetic": "f32; exact int32 ternary products; per-token int8 activation absmax, round half to even; RMSNorm square sums in f64; half-split RoPE, f64 angles; tied bf16 table widened to f32",
        "bos": "every prompt and the passage start with BOS 128000 at position 0",
        "passage": {"file": passage_path.file_name().and_then(|s| s.to_str()), "sha256": passage_sha, "tokens_after_bos": ids.len() - 1},
        "passage_ids_sha256": sha_hex(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()),
        "perplexity": ppl,
        "prompts": prompts,
        "greedy": {"new_tokens_max": NEW_TOKENS, "stop_ids": eos},
        "decode": {"steps": step_ms.len(), "median_ms_per_token": median, "note": "host reference: one-token forward plus the full-vocabulary logits, i8 trits and f32 table"},
        "load": {"seconds": load_s, "resident_bytes": report.resident_bytes},
        "total_seconds": t_all.elapsed().as_secs_f64(),
    });
    let text = serde_json::to_string_pretty(&golden).map_err(|e| e.to_string())?;
    let path: PathBuf = out.join("golden.json");
    std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// The passage text: a plain file, or the `text` field of a JSON file whose `text_sha256` must
/// match (the evidence keeps the passage as JSON data, dashes and all).
fn read_passage(path: &Path) -> Result<String, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if path.extension().and_then(|e| e.to_str()) != Some("json") {
        return Ok(raw);
    }
    let j: Value = serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = j["text"]
        .as_str()
        .ok_or_else(|| format!("{}: no text field", path.display()))?;
    let sha = sha_hex(text.as_bytes());
    if j["text_sha256"] != sha.as_str() {
        return Err(format!(
            "{}: text sha256 {sha} is not the recorded {}",
            path.display(),
            j["text_sha256"]
        ));
    }
    Ok(text.to_string())
}
