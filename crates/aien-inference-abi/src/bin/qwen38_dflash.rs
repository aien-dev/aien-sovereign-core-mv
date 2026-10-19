//! Qwen3.8-27B with the DFlash 2 drafter as a draft source (cascade/dflash-drafter).
//!
//!   qwen38_dflash gen <27B-dir> <drafter-dir> <out.json> [--prompt code|codelong|prose|tune]
//!                     [--n 256] [--live 1] [--mtp-k 7] [--mutant none|convswap|nopair|nocontext]
//!                     [--norollback 1]
//!     On the chip, exact rows path:
//!     1. capture: greedy, one token at a time (k = 0), the residual after the drafter's tap layers
//!        recorded for every position. These tokens are the reference.
//!     2. replay: the DFlash round loop over the record (drafter on the host, the 27B's lm_head
//!        on the chip): accepted drafts per round, words per verify pass.
//!     3. live (--live 1): the same loop with a real verify pass per round (the anchor and 7 drafts
//!        as one exact rows pass, rollback and replay of the DeltaNet state). PASS needs its tokens
//!        equal to the capture's and every round's acceptance equal to the replay's.
//!     4. MTP (--mtp-k K > 0): the checkpoint's own MTP chain of K on the same path, its tokens
//!        equal to the capture's, its words per pass.
//!     --mutant changes the drafter (acceptance must drop, tokens must not change); --norollback 1
//!     leaves the DeltaNet state of refused drafts in place (tokens must change: the control).
//!
//! The GPU lock is the caller's (flock /tmp/aien-gb10.lock). Timings in the output are taken on a
//! shared machine unless the receipt says the quiet lock was held exclusively.
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_dflash::{DraftMutant, Drafter};
use aien_inference_abi::qwen38_omega::dflash::{self, DflashRun};
use aien_inference_abi::qwen38_omega::spec::{generate, Mtp, MtpOptions, RowStats, SpecMutant};
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega};
use aien_inference_abi::tokenizer::{ChatOptions, ChatTokenizer};
use serde_json::{json, Value};

const KEEP_FREE: u64 = 12 << 30;
const MTP_BYTES: u64 = 1 << 30;
/// The drafter on the host: 3.85 GB of bf16 plus working rows.
const DRAFTER_BYTES: u64 = 4 << 30;

const PROMPT_CODE: &str = "Write a C function that returns the larger of two ints.";
const PROMPT_PROSE: &str = "Write a short story, about three paragraphs, about a lighthouse keeper who finds a message in a bottle.";
const PROMPT_TUNE: &str = "Explain how the gears on a bicycle work and when a rider should shift.";
const PROMPT_CODE_LONG: &str = "Write a C program that reads lines from standard input, counts how often each word appears using a hash table you implement yourself, and prints the 10 most frequent words. Include comments.";

fn mem_available() -> Result<u64, String> {
    let m = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    m.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb << 10)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}

fn guard(need: u64, what: &str) -> Result<(), String> {
    let avail = mem_available()?;
    eprintln!(
        "memory: {what} needs {} MiB, keep {} MiB free, MemAvailable {} MiB",
        need >> 20,
        KEEP_FREE >> 20,
        avail >> 20
    );
    if avail < need + KEEP_FREE {
        return Err(format!(
            "{what}: needs {} MiB with {} MiB kept free, MemAvailable {} MiB; refusing",
            need >> 20,
            KEEP_FREE >> 20,
            avail >> 20
        ));
    }
    Ok(())
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn stats_json(s: &RowStats) -> Value {
    json!({"rows": s.rows, "passes": s.passes, "gemm_ms": ms(s.gemm_ns), "gemm_calls": s.gemm_calls,
           "ab_ms": ms(s.ab_ns), "gdn_ms": ms(s.gdn_ns), "gdn_calls": s.gdn_calls, "attn_ms": ms(s.attn_ns),
           "attn_calls": s.attn_calls, "host_ms": ms(s.host_ns), "head_ms": ms(s.head_ns), "head_calls": s.head_calls})
}

fn prompt_ids(dir: &Path, which: &str) -> Result<Vec<u32>, String> {
    let text = match which {
        "code" => PROMPT_CODE,
        "prose" => PROMPT_PROSE,
        "tune" => PROMPT_TUNE,
        "codelong" => PROMPT_CODE_LONG,
        other => return Err(format!("--prompt {other}: code|codelong|prose|tune")),
    };
    let tok = ChatTokenizer::from_model_dir(dir, None).map_err(|e| e.to_string())?;
    let opts = ChatOptions {
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    let s = tok
        .try_format_chat_with(&[("user", text)], &opts)
        .map_err(|e| e.to_string())?;
    tok.encode_with_special(&s, false)
        .map_err(|e| e.to_string())
}

fn run_json(r: &DflashRun) -> Value {
    json!({
        "rounds": r.rounds, "drafted": r.drafted, "accepted": r.accepted,
        "words_per_pass": r.words_per_pass(),
        "accept_rate": if r.drafted > 0 { r.accepted as f64 / r.drafted as f64 } else { 0.0 },
        "accepted_hist": r.accepted_hist, "rollbacks": r.rollbacks,
        "tokens_n": r.tokens.len(),
        "per_round_start_accepted": r.per_round,
        "ms": {"decode": ms(r.decode_ns), "prompt": ms(r.prompt_ns), "draft": ms(r.draft_ns),
               "context": ms(r.context_ns), "verify": ms(r.verify_ns), "snapshot": ms(r.snapshot_ns),
               "rollback": ms(r.rollback_ns), "head": ms(r.head_ns)},
        "head_calls": r.head_calls,
        "main": stats_json(&r.main),
    })
}

fn gen(args: &[String]) -> Result<bool, String> {
    let [dir, ddir, out] = [args.first(), args.get(1), args.get(2)]
        .map(|a| a.map(String::as_str))
        .map(|a| a.ok_or("gen <27B-dir> <drafter-dir> <out.json> [flags]"));
    let (dir, ddir, out) = (Path::new(dir?), Path::new(ddir?), out?);
    let n: usize = flag(args, "--n")
        .unwrap_or("256")
        .parse()
        .map_err(|e| format!("--n: {e}"))?;
    let which = flag(args, "--prompt").unwrap_or("code");
    let live = flag(args, "--live").unwrap_or("1") == "1";
    let mtp_k: usize = flag(args, "--mtp-k")
        .unwrap_or("7")
        .parse()
        .map_err(|e| format!("--mtp-k: {e}"))?;
    let mutant = DraftMutant::parse(flag(args, "--mutant").unwrap_or("none"))?;
    let norollback = flag(args, "--norollback") == Some("1");

    let prompt = prompt_ids(dir, which)?;
    let tok = ChatTokenizer::from_model_dir(dir, None).map_err(|e| e.to_string())?;
    let stop = tok.stop_token_ids().to_vec();
    let c = Qwen38Config::from_checkpoint(dir)?;
    let max_pos = (prompt.len() + n + 16).div_ceil(256) * 256;
    guard(
        need_bytes(&c, max_pos, 0) + MTP_BYTES + DRAFTER_BYTES,
        "whole model + MTP head + drafter",
    )?;
    let t = Instant::now();
    let drafter = Drafter::open(ddir)?;
    let load_drafter_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut model = Qwen38Omega::open(dir, None, max_pos)?;
    let load_model_s = t.elapsed().as_secs_f64();
    eprintln!(
        "drafter {} MiB in {load_drafter_s:.1} s, model in {load_model_s:.1} s, prompt {which} {} tokens",
        drafter.weight_bytes() >> 20,
        prompt.len()
    );

    let t = Instant::now();
    let cap = dflash::capture(&mut model, &drafter.cfg.taps, &prompt, n, &stop, 64)?;
    let capture_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "capture: {} tokens in {capture_ms:.0} ms",
        cap.generated().len()
    );
    let rep = dflash::replay_rounds(&drafter, &mut model, &cap, mutant)?;
    eprintln!(
        "replay: {} rounds, {} accepted of {} drafted, {:.3} words per pass",
        rep.rounds,
        rep.accepted,
        rep.drafted,
        rep.words_per_pass()
    );
    let text = tok.decode(cap.generated()).unwrap_or_default();
    let mut rec = json!({
        "schema": "QWEN38_DFLASH_GEN_V1",
        "prompt": which, "prompt_len": prompt.len(), "n": n, "mutant": format!("{mutant:?}"),
        "norollback": norollback,
        "drafter_dir": ddir.display().to_string(), "model_dir": dir.display().to_string(),
        "drafter_config": format!("{:?}", drafter.cfg),
        "capture": {"tokens": cap.generated(), "text": text, "ms": capture_ms, "main": stats_json(&cap.main)},
        "replay": run_json(&rep),
        "load_s": {"drafter": load_drafter_s, "model": load_model_s},
    });
    let mut pass = true;
    if live {
        let r = dflash::generate(
            &mut model, &drafter, &prompt, n, &stop, 64, mutant, norollback,
        )?;
        let same_tokens = r.tokens == cap.generated();
        let same_rounds = r.per_round == rep.per_round;
        eprintln!(
            "live: {} rounds, {:.3} words per pass, tokens identical to k = 0: {same_tokens}, rounds identical to replay: {same_rounds}",
            r.rounds,
            r.words_per_pass()
        );
        let mut j = run_json(&r);
        j["identical_to_k0"] = json!(same_tokens);
        j["rounds_identical_to_replay"] = json!(same_rounds);
        if !same_tokens {
            j["first_diff"] = json!(r
                .tokens
                .iter()
                .zip(cap.generated())
                .position(|(a, b)| a != b));
        }
        rec["live"] = j;
        pass &= same_tokens && same_rounds;
    }
    if mtp_k > 0 {
        let mut mtp = Mtp::open(dir, &model, MtpOptions::default())?;
        let r = generate(
            &mut model,
            Some(&mut mtp),
            &prompt,
            n,
            mtp_k,
            &stop,
            64,
            SpecMutant::None,
        )?;
        let same = r.tokens == cap.generated();
        let wpp = if r.rounds > 0 {
            (r.accepted + r.rounds) as f64 / r.rounds as f64
        } else {
            0.0
        };
        eprintln!(
            "mtp k {mtp_k}: {} rounds, {wpp:.3} words per pass, tokens identical: {same}",
            r.rounds
        );
        rec["mtp"] = json!({"k": mtp_k, "rounds": r.rounds, "drafted": r.drafted, "accepted": r.accepted,
            "words_per_pass": wpp, "rollbacks": r.rollbacks, "identical_to_k0": same,
            "ms": {"decode": ms(r.decode_ns), "draft": ms(r.draft_ns), "verify": ms(r.verify_ns),
                   "snapshot": ms(r.snapshot_ns), "rollback": ms(r.rollback_ns)},
            "main": stats_json(&r.main), "mtp": stats_json(&r.mtp)});
        pass &= same;
    }
    rec["pass"] = json!(pass);
    std::fs::write(out, serde_json::to_string_pretty(&rec).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    println!(
        "QWEN38_DFLASH prompt {which} n {n} mutant {mutant:?} norollback {norollback}: replay {:.3} words/pass{}{} -> {}",
        rep.words_per_pass(),
        rec.get("live")
            .map(|l| format!(", live {:.3}", l["words_per_pass"].as_f64().unwrap_or(0.0)))
            .unwrap_or_default(),
        rec.get("mtp")
            .map(|m| format!(", mtp k{mtp_k} {:.3}", m["words_per_pass"].as_f64().unwrap_or(0.0)))
            .unwrap_or_default(),
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("gen") => gen(&args[1..]),
        _ => Err("usage: qwen38_dflash gen <27B-dir> <drafter-dir> <out.json> [flags]".into()),
    };
    match r {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_dflash: {e}");
            std::process::exit(2);
        }
    }
}
