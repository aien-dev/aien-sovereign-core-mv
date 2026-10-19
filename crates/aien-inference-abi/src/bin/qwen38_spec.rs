//! Qwen3.8-27B speculative decoding with the MTP head on the chip (sc #679 task A).
//!
//!   qwen38_spec probe <checkpoint-dir> <layers> <out.json>
//!     The first `layers` layers only. Checks, each with a control that must fail:
//!     - rows identity: an m-row exact pass equals m one-row passes bit for bit (residual rows and
//!       head tokens), at m = 2..5 from a mid-sequence position, and the whole prompt as one pass;
//!     - rollback: snapshot, a 5-row pass, restore, replay of j rows leaves the DeltaNet state
//!       bit for bit where j one-row passes leave it (control: replay j + 1 rows);
//!     - costs: snapshot, restore, exact pass at m = 1 and 5, decode graph token.
//!   qwen38_spec gen <checkpoint-dir> <out.json> [--k 0,1,2,3,4] [--n 256] [--prompt code|codelong|codeedit|quote|prose|tune|golden]
//!                   [--order embed|hidden] [--normed 0|1] [--offset 0|1] [--mutant none|extra|norollback|rowshift]
//!                   [--graph 0|1] [--golden <gen-chat-max-24.json>] [--weights fp8|mxfp4]
//!                   [--rows all|accepted] [--source mtp|ngram|ngram-first|agree|merge] [--ngram <max>,<min>]
//!                   [--tree <w1>,<w2>,<side>] [--mutant ...|treenorebuild|treenokv]
//!     --tree replaces the chain with a small tree: the MTP chain of k as the trunk, the next
//!     w1 - 1 best first words and w2 - 1 best second words as side branches, each continued
//!     by an MTP chain of <side> words; one tree pass, rebuild-only-accepted state.
//!     k up to 8 (the cascade's chain). --rows all (default) runs the lm_head and the MTP head on every
//!     row of the verify pass (the cascade shape); accepted runs them on the accepted rows only (the
//!     pre-cascade shape, fewer lm_head calls today). --source picks the draft source: the MTP chain,
//!     the prompt-lookup n-gram drafter (spans of <max> down to <min> tokens, default 6,2), or a mix.
//!     The whole model. For each k: greedy tokens, acceptance, wall per token and its parts; PASS when
//!     every k > 0 run's tokens equal the k = 0 run's (the same exact path one token at a time) and,
//!     with --golden on the golden prompt, the first 24 equal the golden record. --graph 1 also runs
//!     the decode graph (the live server's path) for its tokens and chip time per token.
//!   qwen38_spec drift <checkpoint-dir> <out.json> --record <score-license-256.json> [--weights fp8|mxfp4]
//!                     [--chunk 8] [--dump <logits.f32>] [--against <fp8-logits.f32>]
//!     Task B. The record's tokens scored teacher-forced on the exact rows path: per-position NLL,
//!     argmax, top-two margin, perplexity, and the two-ahead control (must score >= 1 nat worse).
//!     --weights mxfp4 converts every FP8 projection to MXFP4 at load (qwen38_omega::w4).
//!     --dump writes every position's logits; --against reads an FP8 dump and reports argmax
//!     agreement, KL(fp8 || this) and the FP8 top-two margin where the argmax flipped. The FP8 run
//!     PASSes when it reproduces the record (mean NLL within 2 percent, accuracy within 0.02).
//!
//! The GPU lock is the caller's (flock /tmp/aien-gb10.lock). A model is uploaded only when
//! MemAvailable covers it plus 12 GiB.
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::spec::argmax;
use aien_inference_abi::qwen38_omega::spec::{
    generate, generate_tree, DraftSource, FcOrder, GdnRows, HeadRows, Mtp, MtpOptions, NgramDraft,
    RowStats, SpecMutant, SpecOptions, SpecRun, TreeSpec, VerifyKernel,
};
use aien_inference_abi::qwen38_omega::{need_bytes, w4, Qwen38Omega, Qwen38Snapshot};
use aien_inference_abi::tokenizer::{ChatOptions, ChatTokenizer};
use serde_json::{json, Value};

const KEEP_FREE: u64 = 12 << 30;
/// The MTP head: fc bf16 + one attention layer at FP8 + its KV (counted generously).
const MTP_BYTES: u64 = 1 << 30;

const PROMPT_CODE: &str = "Write a C function that returns the larger of two ints.";
const PROMPT_PROSE: &str = "Write a short story, about three paragraphs, about a lighthouse keeper who finds a message in a bottle.";
const PROMPT_TUNE: &str = "Explain how the gears on a bicycle work and when a rider should shift.";
const PROMPT_CODE_LONG: &str = "Write a C program that reads lines from standard input, counts how often each word appears using a hash table you implement yourself, and prints the 10 most frequent words. Include comments.";
const PROMPT_CODE_EDIT: &str = "Here is a C function:

```c
#include <stdio.h>
#include <string.h>

static int count_words(const char *line) {
    int count = 0;
    int in_word = 0;
    for (const char *p = line; *p; p++) {
        if (*p == ' ' || *p == '\t' || *p == '\n') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
    }
    return count;
}

int main(void) {
    char line[1024];
    int total = 0;
    while (fgets(line, sizeof line, stdin)) {
        total += count_words(line);
    }
    printf(\"%d\\n\", total);
    return 0;
}
```

Change it so that it also counts lines and characters, and prints all three numbers on one line separated by spaces. Return the complete modified program and nothing else.";
const PROMPT_QUOTE: &str = "Read this passage:

The lighthouse at Point Harrow was built in 1847 from local granite, and for the first sixty years its lamp burned whale oil carried up the spiral stair by hand every evening. The keeper, a position held by the same family for three generations, lived in the small cottage at its base with a vegetable garden walled against the wind. In 1908 the oil lamp was replaced by an acetylene burner that could be left to run through the night, and the keeper's duties shrank to cleaning the lens and keeping the log. The lens itself, a second-order Fresnel made in Paris, weighs nearly two tonnes and floats on a bath of mercury so that a clockwork motor can turn it with one finger. Ships approaching from the south see one long flash every fifteen seconds; from the north, where the rocks are worse, the pattern is two short flashes. The station was automated in 1974 and the cottage is now a museum open on summer weekends. Visitors who climb the one hundred and twelve steps can still see the brass plate recording the names of the eleven keepers.

First, quote word for word the sentence about the lens. Then quote word for word the sentence about what ships see. Then say in one sentence why the two flash patterns differ.";

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

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn first_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

fn stats_json(s: &RowStats) -> Value {
    json!({"rows": s.rows, "passes": s.passes, "gemm_ms": ms(s.gemm_ns), "gemm_calls": s.gemm_calls,
           "ab_ms": ms(s.ab_ns), "gdn_ms": ms(s.gdn_ns), "gdn_calls": s.gdn_calls, "attn_ms": ms(s.attn_ns),
           "attn_calls": s.attn_calls, "host_ms": ms(s.host_ns), "head_ms": ms(s.head_ns), "head_calls": s.head_calls})
}

fn prompt_ids(dir: &Path, which: &str, golden: Option<&Value>) -> Result<Vec<u32>, String> {
    if which == "golden" {
        let g = golden.ok_or("--prompt golden needs --golden")?;
        return ids(g, "prompt_ids");
    }
    let text = match which {
        "code" => PROMPT_CODE,
        "prose" => PROMPT_PROSE,
        "tune" => PROMPT_TUNE,
        "codelong" => PROMPT_CODE_LONG,
        "codeedit" => PROMPT_CODE_EDIT,
        "quote" => PROMPT_QUOTE,
        other => {
            return Err(format!(
                "--prompt {other}: code|codelong|codeedit|quote|prose|tune|golden"
            ))
        }
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

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("golden: no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as u32)
                .ok_or_else(|| format!("golden: {key} holds a non-id"))
        })
        .collect()
}

/// `tokens` one at a time through the exact path: every row's residual.
fn singles(m: &mut Qwen38Omega, tokens: &[u32], st: &mut RowStats) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for &t in tokens {
        out.extend(m.exact_rows(&[t], None, st)?);
    }
    Ok(out)
}

fn snap_equal(a: &Qwen38Snapshot, b: &Qwen38Snapshot) -> bool {
    a.bits_equal(b)
}

fn probe(args: &[String]) -> Result<bool, String> {
    let [dir, layers, out] = args else {
        return Err("usage: qwen38_spec probe <checkpoint-dir> <layers> <out.json>".into());
    };
    let dir = Path::new(dir);
    let layers: usize = layers.parse().map_err(|e| format!("layers: {e}"))?;
    let full = Qwen38Config::from_checkpoint(dir)?;
    let c = full.prefix(layers).ok_or("layers out of range")?;
    let max_pos = 256;
    guard(need_bytes(&c, max_pos, 0), "probe model")?;
    let t = Instant::now();
    let mut m = Qwen38Omega::open(dir, Some(layers), max_pos)?;
    eprintln!(
        "uploaded {layers} layers in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("HOME").unwrap_or_default() + "/handoffs/qwen38-ref/gen-chat-max-24.json",
        )
        .map_err(|e| format!("golden: {e}"))?,
    )
    .map_err(|e| format!("golden: {e}"))?;
    let mut toks = ids(&golden, "prompt_ids")?;
    toks.extend(ids(&golden, "generated_ids")?);
    let h = c.hidden;
    let mut st = RowStats::default();
    let mut pass = true;
    let mut checks = Vec::new();

    // 1. Whole sequence: one pass of n rows against n one-row passes.
    let n = toks.len();
    m.reset()?;
    let t = Instant::now();
    let one = m.exact_rows(&toks, None, &mut st)?;
    let one_ms = t.elapsed().as_secs_f64() * 1e3;
    m.reset()?;
    let mut s1 = RowStats::default();
    let t = Instant::now();
    let many = singles(&mut m, &toks, &mut s1)?;
    let many_ms = t.elapsed().as_secs_f64() * 1e3;
    let ok = bits_equal(&one, &many);
    pass &= ok;
    eprintln!(
        "rows identity, {n} rows in one pass vs {n} one-row passes: {} ({one_ms:.1} ms vs {many_ms:.1} ms)",
        if ok { "IDENTICAL" } else { "DIFFER" }
    );
    checks.push(
        json!({"check": "whole_sequence_rows", "rows": n, "identical": ok,
        "first_diff": first_diff(&one, &many), "pass_ms": one_ms, "singles_ms": many_ms}),
    );

    // 2. Mid-sequence m-row passes, m = 2..5, each against m one-row passes from the same state.
    let base = 20;
    let mut snap = Qwen38Snapshot::default();
    for mm in 2..=5usize {
        m.reset()?;
        singles(&mut m, &toks[..base], &mut st)?;
        m.snapshot(&mut snap)?;
        let rows = m.exact_rows(&toks[base..base + mm], None, &mut st)?;
        let mut heads_a = Vec::new();
        for r in 0..mm {
            heads_a.push(m.head_token(&rows[r * h..(r + 1) * h], &mut st)?);
        }
        m.restore(&snap)?;
        let single = singles(&mut m, &toks[base..base + mm], &mut st)?;
        let mut heads_b = Vec::new();
        for r in 0..mm {
            heads_b.push(m.head_token(&single[r * h..(r + 1) * h], &mut st)?);
        }
        let ok = bits_equal(&rows, &single) && heads_a == heads_b;
        pass &= ok;
        eprintln!(
            "rows identity m={mm} at position {base}: {}",
            if ok { "IDENTICAL" } else { "DIFFER" }
        );
        checks.push(json!({"check": "mid_rows", "m": mm, "identical": ok, "heads": heads_a}));
    }

    // 3. Rollback: 5-row pass with kept inputs, restore, replay j rows == j one-row passes.
    for j in 1..=4usize {
        m.reset()?;
        singles(&mut m, &toks[..base], &mut st)?;
        m.snapshot(&mut snap)?;
        let mut keep: Vec<GdnRows> = Vec::new();
        m.exact_rows(&toks[base..base + 5], Some(&mut keep), &mut st)?;
        m.restore(&snap)?;
        m.replay_gdn(&keep, j, &mut st)?;
        let mut after_replay = Qwen38Snapshot::default();
        m.snapshot(&mut after_replay)?;
        // Control: one row too many.
        m.restore(&snap)?;
        m.replay_gdn(&keep, j + 1, &mut st)?;
        let mut after_wrong = Qwen38Snapshot::default();
        m.snapshot(&mut after_wrong)?;
        m.restore(&snap)?;
        singles(&mut m, &toks[base..base + j], &mut st)?;
        let mut after_single = Qwen38Snapshot::default();
        m.snapshot(&mut after_single)?;
        let ok = snap_equal(&after_replay, &after_single);
        let control_caught = !snap_equal(&after_wrong, &after_single);
        pass &= ok && control_caught;
        eprintln!(
            "rollback j={j}: replay {} the one-row state; control (j+1 rows) {}",
            if ok { "EQUALS" } else { "DIFFERS FROM" },
            if control_caught {
                "caught"
            } else {
                "NOT CAUGHT"
            }
        );
        checks.push(
            json!({"check": "rollback", "j": j, "identical": ok, "control_caught": control_caught}),
        );
    }

    // 4. Costs.
    m.reset()?;
    singles(&mut m, &toks[..base], &mut st)?;
    let reps = 5;
    let (mut snap_ns, mut restore_ns) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        m.snapshot(&mut snap)?;
        snap_ns.push(t.elapsed().as_nanos() as u64);
        let t = Instant::now();
        m.restore(&snap)?;
        restore_ns.push(t.elapsed().as_nanos() as u64);
    }
    let mut pass_ns = [Vec::new(), Vec::new()];
    for _ in 0..reps {
        for (i, mm) in [1usize, 5].into_iter().enumerate() {
            m.restore(&snap)?;
            let t = Instant::now();
            m.exact_rows(&toks[base..base + mm], None, &mut RowStats::default())?;
            pass_ns[i].push(t.elapsed().as_nanos() as u64);
        }
    }
    let mut s5 = RowStats::default();
    m.restore(&snap)?;
    m.exact_rows(&toks[base..base + 5], None, &mut s5)?;
    let mut s1b = RowStats::default();
    m.restore(&snap)?;
    m.exact_rows(&toks[base..base + 1], None, &mut s1b)?;
    // The decode graph over the same history, token by token (its own state path).
    m.reset()?;
    let mut graph_ns = Vec::new();
    let mut graph_tokens = Vec::new();
    for &t in &toks[..base + 5] {
        let (g, gs) = m.step(t)?;
        graph_ns.push(gs.total_ns);
        graph_tokens.push(g);
    }
    m.reset()?;
    let mut s = RowStats::default();
    let rows = m.exact_rows(&toks[..base + 5], None, &mut s)?;
    let mut exact_tokens = Vec::new();
    for r in 0..base + 5 {
        exact_tokens.push(m.head_token(&rows[r * h..(r + 1) * h], &mut s)?);
    }
    let agree = graph_tokens
        .iter()
        .zip(&exact_tokens)
        .filter(|(a, b)| a == b)
        .count();
    let med = |v: &mut Vec<u64>| {
        v.sort_unstable();
        ms(v[v.len() / 2])
    };
    let costs = json!({
        "snapshot_ms_median": med(&mut snap_ns), "restore_ms_median": med(&mut restore_ns),
        "exact_pass_m1_ms_median": med(&mut pass_ns[0]), "exact_pass_m5_ms_median": med(&mut pass_ns[1]),
        "exact_pass_m1_parts": stats_json(&s1b), "exact_pass_m5_parts": stats_json(&s5),
        "graph_token_total_ms_median": med(&mut graph_ns.clone()),
        "graph_vs_exact_head_tokens_agree": agree, "positions": base + 5,
    });
    eprintln!("costs ({layers} layers): {costs}");
    let rec = json!({"tool": "qwen38_spec probe", "layers": layers, "pass": pass, "checks": checks, "costs": costs});
    std::fs::write(out, serde_json::to_string_pretty(&rec).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    println!(
        "QWEN38_SPEC_PROBE layers {layers}: {}",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn run_json(k: usize, r: &SpecRun) -> Value {
    let gen = r.tokens.len() as f64;
    json!({
        "k": k, "tokens": r.tokens, "generated": r.tokens.len(), "rounds": r.rounds,
        "drafted": r.drafted, "accepted": r.accepted, "rollbacks": r.rollbacks,
        "accept_rate": if r.drafted > 0 { r.accepted as f64 / r.drafted as f64 } else { 0.0 },
        "tokens_per_round": if r.rounds > 0 { (gen - 1.0) / r.rounds as f64 } else { 0.0 },
        "prompt_ms": ms(r.prompt_ns), "decode_ms": ms(r.decode_ns),
        "decode_ms_per_token": ms(r.decode_ns) / (gen - 1.0).max(1.0),
        "draft_ms": ms(r.draft_ns), "verify_ms": ms(r.verify_ns), "snapshot_ms": ms(r.snapshot_ns),
        "rollback_ms": ms(r.rollback_ns), "main": stats_json(&r.main), "mtp": stats_json(&r.mtp),
        "ngram_rounds": r.ngram_rounds, "ngram_drafted": r.ngram_drafted, "ngram_accepted": r.ngram_accepted,
        "accepted_hist": r.accepted_hist,
        "accepted_per_round": if r.rounds > 0 { r.accepted as f64 / r.rounds as f64 } else { 0.0 },
        "words_per_pass": if r.rounds > 0 { (r.accepted + r.rounds) as f64 / r.rounds as f64 } else { 0.0 },
        "rows_per_pass": if r.rounds > 0 { (r.drafted + r.rounds) as f64 / r.rounds as f64 } else { 0.0 },
        "verify_ms_per_round": if r.rounds > 0 { ms(r.verify_ns) / r.rounds as f64 } else { 0.0 },
        "draft_ms_per_round": if r.rounds > 0 { ms(r.draft_ns) / r.rounds as f64 } else { 0.0 },
        "state_ms_per_round": if r.rounds > 0 { ms(r.snapshot_ns + r.rollback_ns) / r.rounds as f64 } else { 0.0 },
    })
}

fn gen(args: &[String]) -> Result<bool, String> {
    let (Some(dir), Some(out)) = (args.first(), args.get(1)) else {
        return Err("usage: qwen38_spec gen <checkpoint-dir> <out.json> [flags]".into());
    };
    let dir = Path::new(dir);
    let ks: Vec<usize> = flag(args, "--k")
        .unwrap_or("0,1,2,3,4")
        .split(',')
        .map(|s| s.parse().map_err(|e| format!("--k: {e}")))
        .collect::<Result<_, String>>()?;
    let n: usize = flag(args, "--n")
        .unwrap_or("256")
        .parse()
        .map_err(|e| format!("--n: {e}"))?;
    let which = flag(args, "--prompt").unwrap_or("code");
    let opts = MtpOptions {
        order: match flag(args, "--order").unwrap_or("embed") {
            "embed" => FcOrder::EmbedFirst,
            "hidden" => FcOrder::HiddenFirst,
            o => return Err(format!("--order {o}")),
        },
        normed_hidden: flag(args, "--normed").unwrap_or("0") == "1",
        pos_offset: flag(args, "--offset")
            .unwrap_or("0")
            .parse()
            .map_err(|e| format!("--offset: {e}"))?,
    };
    let mutant = match flag(args, "--mutant").unwrap_or("none") {
        "none" => SpecMutant::None,
        "extra" => SpecMutant::ExtraAccept,
        "norollback" => SpecMutant::NoRollback,
        "rowshift" => SpecMutant::RowShift,
        "treenorebuild" => SpecMutant::TreeNoRebuild,
        "treenokv" => SpecMutant::TreeNoKvFix,
        o => return Err(format!("--mutant {o}")),
    };
    let tree = match flag(args, "--tree") {
        None => None,
        Some(s) => {
            let v: Vec<usize> = s
                .split(',')
                .map(|x| x.parse().map_err(|e| format!("--tree: {e}")))
                .collect::<Result<_, _>>()?;
            let [w1, w2, side] = v[..] else {
                return Err("--tree w1,w2,side".into());
            };
            Some(TreeSpec { w1, w2, side })
        }
    };
    // Task B: the same generation with every FP8 projection converted to MXFP4 at load.
    match flag(args, "--weights").unwrap_or("fp8") {
        "fp8" => {}
        "mxfp4" => w4::set_mxfp4(true),
        o => return Err(format!("--weights {o}: fp8|mxfp4")),
    }
    let rows = match flag(args, "--rows").unwrap_or("all") {
        "all" => HeadRows::All,
        "accepted" => HeadRows::Accepted,
        o => return Err(format!("--rows {o}: all|accepted")),
    };
    let kernel = match flag(args, "--verify").unwrap_or("exact") {
        "exact" => VerifyKernel::ExactRows,
        "tree" => VerifyKernel::TreeGemm,
        o => return Err(format!("--verify {o}: exact|tree")),
    };
    let source = match flag(args, "--source").unwrap_or("mtp") {
        "mtp" => DraftSource::Mtp,
        "ngram" => DraftSource::Ngram,
        "ngram-first" => DraftSource::NgramFirst,
        "agree" => DraftSource::Agree,
        "merge" => DraftSource::Merge,
        o => return Err(format!("--source {o}: mtp|ngram|ngram-first|agree|merge")),
    };
    let ngram = match flag(args, "--ngram") {
        None => NgramDraft::default(),
        Some(s) => {
            let (a, b) = s.split_once(',').ok_or("--ngram max,min")?;
            NgramDraft {
                max_n: a.parse().map_err(|e| format!("--ngram: {e}"))?,
                min_n: b.parse().map_err(|e| format!("--ngram: {e}"))?,
            }
        }
    };
    let sopts = SpecOptions {
        kernel,
        rows,
        source,
        ngram,
        mutant,
        tree,
    };
    let graph = flag(args, "--graph").unwrap_or("0") == "1";
    let golden: Option<Value> = match flag(args, "--golden") {
        Some(p) => Some(
            serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)
                .map_err(|e| format!("{p}: {e}"))?,
        ),
        None => None,
    };
    let prompt = prompt_ids(dir, which, golden.as_ref())?;
    if let Some(g) = &golden {
        let gp = ids(g, "prompt_ids")?;
        eprintln!(
            "prompt {which}: {} tokens; golden prompt {} the rendered code prompt",
            prompt.len(),
            if prompt_ids(dir, "code", None)? == gp {
                "equals"
            } else {
                "DIFFERS FROM"
            }
        );
    }
    let tok = ChatTokenizer::from_model_dir(dir, None).map_err(|e| e.to_string())?;
    let stop = tok.stop_token_ids().to_vec();
    let c = Qwen38Config::from_checkpoint(dir)?;
    let max_pos = (prompt.len() + n + 16).div_ceil(256) * 256;
    guard(
        need_bytes(&c, max_pos, 0) + MTP_BYTES,
        "whole model + MTP head",
    )?;
    let t = Instant::now();
    let mut model = Qwen38Omega::open(dir, None, max_pos)?;
    let mut mtp = if ks.iter().any(|&k| k > 0) {
        let m = Mtp::open(dir, &model, opts)?;
        eprintln!(
            "MTP head: {} MiB, raw norm means (embed, hidden, final) {:?}, options {opts:?}",
            m.weight_bytes() >> 20,
            m.raw_norm_means
        );
        Some(m)
    } else {
        None
    };
    eprintln!("uploaded in {:.1} s", t.elapsed().as_secs_f64());

    if flag(args, "--sweep-mtp") == Some("1") {
        // Draft quality of every MTP wiring at k = 1 (the tuning prompt; output is the k = 0 path's
        // whatever the wiring, so only acceptance is compared).
        let m = mtp.as_mut().ok_or("--sweep-mtp needs k > 0 in --k")?;
        let mut rows = Vec::new();
        let orders = match flag(args, "--sweep-order") {
            Some("embed") => vec![FcOrder::EmbedFirst],
            _ => vec![FcOrder::EmbedFirst, FcOrder::HiddenFirst],
        };
        for order in orders {
            for normed_hidden in [false, true] {
                for pos_offset in [0usize, 1] {
                    m.opts = MtpOptions {
                        order,
                        normed_hidden,
                        pos_offset,
                    };
                    let r = generate(&mut model, Some(&mut *m), &prompt, n, 1, &stop, 64, sopts)?;
                    let rate = r.accepted as f64 / r.drafted.max(1) as f64;
                    eprintln!(
                        "mtp {:?}: accept {}/{} = {rate:.3}",
                        m.opts, r.accepted, r.drafted
                    );
                    rows.push(
                        json!({"options": format!("{:?}", m.opts), "accepted": r.accepted,
                        "drafted": r.drafted, "accept_rate": rate, "tokens": r.tokens}),
                    );
                }
            }
        }
        let rec = json!({"tool": "qwen38_spec gen --sweep-mtp", "prompt": which, "n": n, "variants": rows});
        std::fs::write(out, serde_json::to_string_pretty(&rec).unwrap())
            .map_err(|e| format!("{out}: {e}"))?;
        println!("QWEN38_SPEC_SWEEP prompt {which} n {n}: done");
        return Ok(true);
    }
    let mut runs = Vec::new();
    let mut base: Option<Vec<u32>> = None;
    let mut pass = true;
    for &k in &ks {
        let r = match (tree, k > 0) {
            (Some(spec), true) => generate_tree(
                &mut model,
                mtp.as_mut().ok_or("--tree needs the MTP head")?,
                &prompt,
                n,
                k,
                &stop,
                64,
                sopts,
                spec,
            )?,
            _ => generate(
                &mut model,
                if k > 0 { mtp.as_mut() } else { None },
                &prompt,
                n,
                k,
                &stop,
                64,
                sopts,
            )?,
        };
        let mut j = run_json(k, &r);
        let same = match &base {
            None => {
                if k == 0 {
                    base = Some(r.tokens.clone());
                }
                None
            }
            Some(b) => Some(*b == r.tokens),
        };
        if let Some(s) = same {
            pass &= s;
            j["identical_to_k0"] = json!(s);
            if !s {
                let d = base
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(&r.tokens)
                    .position(|(a, b)| a != b);
                j["first_diff"] = json!(d);
            }
        }
        if let Some(g) = &golden {
            if which == "golden" {
                let gi = ids(g, "generated_ids")?;
                let gl = gi.len().min(r.tokens.len());
                let ok = r.tokens[..gl] == gi[..gl] && gl == gi.len();
                pass &= ok;
                j["golden_identical"] = json!(ok);
            }
        }
        eprintln!(
            "k={k}: {} tokens, {:.2} ms/token decode, accept {}/{} ({:.3}), {:.2} tokens/round, ngram rounds {} accept {}/{}, identical_to_k0 {:?}",
            r.tokens.len(),
            j["decode_ms_per_token"].as_f64().unwrap_or(0.0),
            r.accepted,
            r.drafted,
            j["accept_rate"].as_f64().unwrap_or(0.0),
            j["tokens_per_round"].as_f64().unwrap_or(0.0),
            r.ngram_rounds,
            r.ngram_accepted,
            r.ngram_drafted,
            same
        );
        runs.push(j);
    }
    let mut graph_rec = Value::Null;
    if graph {
        model.reset()?;
        let t = Instant::now();
        let mut next = 0;
        for &p in &prompt {
            next = model.step(p)?.0;
        }
        let prompt_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut toks = vec![next];
        let (mut chip, mut total) = (Vec::new(), Vec::new());
        let t = Instant::now();
        while toks.len() < n && !stop.contains(toks.last().unwrap()) {
            let (g, s) = model.step(*toks.last().unwrap())?;
            chip.push(s.chip_ns);
            total.push(s.total_ns);
            toks.push(g);
        }
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let steps = chip.len().max(1) as f64;
        let agree = base
            .as_ref()
            .map(|b| b.iter().zip(&toks).take_while(|(a, b)| a == b).count());
        eprintln!(
            "graph: {} tokens, chip {:.2} ms/token, wall {:.2} ms/token, agrees with the exact path for the first {agree:?} tokens",
            toks.len(),
            ms(chip.iter().sum::<u64>()) / steps,
            wall / steps
        );
        graph_rec = json!({"tokens": toks, "prompt_ms": prompt_ms, "decode_ms": wall,
            "chip_ms_per_token": ms(chip.iter().sum::<u64>()) / steps, "wall_ms_per_token": wall / steps,
            "chip_ns": chip, "total_ns": total, "prefix_agreeing_with_exact_k0": agree});
    }
    let text = base
        .as_ref()
        .map(|b| tok.decode(b).unwrap_or_default())
        .unwrap_or_default();
    let rec = json!({"tool": "qwen38_spec gen", "prompt": which, "prompt_ids": prompt, "n": n,
        "mtp_options": format!("{opts:?}"), "mutant": format!("{mutant:?}"), "spec_options": format!("{sopts:?}"), "pass": pass,
        "runs": runs, "graph": graph_rec, "k0_text": text});
    std::fs::write(out, serde_json::to_string_pretty(&rec).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    println!(
        "QWEN38_SPEC_GEN prompt {which} n {n} k {ks:?} rows {rows:?} source {source:?} tree {tree:?} mutant {mutant:?}: {}",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// `log_softmax(logits)` in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = logits
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    logits.iter().map(|&v| v as f64 - lse).collect()
}

/// Largest minus second largest logit.
fn top2_margin(logits: &[f32]) -> f64 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    (a - b) as f64
}

fn drift(args: &[String]) -> Result<bool, String> {
    use std::io::{Read, Write};
    let (Some(dir), Some(out)) = (args.first(), args.get(1)) else {
        return Err(
            "usage: qwen38_spec drift <checkpoint-dir> <out.json> --record <score.json> [flags]"
                .into(),
        );
    };
    let dir = Path::new(dir);
    let record_path = flag(args, "--record").ok_or("drift: --record <score-license-256.json>")?;
    let record: Value = serde_json::from_str(
        &std::fs::read_to_string(record_path).map_err(|e| format!("{record_path}: {e}"))?,
    )
    .map_err(|e| format!("{record_path}: {e}"))?;
    let toks = ids(&record, "ids")?;
    let ref_mean = record["mean_nll"].as_f64().ok_or("record: no mean_nll")?;
    let ref_acc = record["next_token_accuracy"]
        .as_f64()
        .ok_or("record: no next_token_accuracy")?;
    let weights = flag(args, "--weights").unwrap_or("fp8");
    match weights {
        "fp8" => {}
        "mxfp4" => w4::set_mxfp4(true),
        o => return Err(format!("--weights {o}: fp8|mxfp4")),
    }
    let chunk: usize = flag(args, "--chunk")
        .unwrap_or("8")
        .parse()
        .map_err(|e| format!("--chunk: {e}"))?;
    let n = toks.len();
    let c = Qwen38Config::from_checkpoint(dir)?;
    let max_pos = (n + 16).div_ceil(256) * 256;
    guard(need_bytes(&c, max_pos, 0), "whole model")?;
    let t = Instant::now();
    let mut model = Qwen38Omega::open(dir, None, max_pos)?;
    let w4s = w4::stats();
    eprintln!(
        "uploaded in {:.1} s, weights {weights}, {} tensors converted, {} weights below the block range",
        t.elapsed().as_secs_f64(),
        w4s.tensors,
        w4s.underflow
    );
    let mut st = RowStats::default();
    model.reset()?;
    let t = Instant::now();
    let mut rows = Vec::with_capacity(n * c.hidden);
    for part in toks.chunks(chunk.max(1)) {
        rows.extend(model.exact_rows(part, None, &mut st)?);
    }
    let pass_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut dump = match flag(args, "--dump") {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?,
        )),
        None => None,
    };
    let mut against = match flag(args, "--against") {
        Some(p) => Some(std::io::BufReader::new(
            std::fs::File::open(p).map_err(|e| format!("{p}: {e}"))?,
        )),
        None => None,
    };
    let v = c.vocab;
    let h = c.hidden;
    let (mut nll, mut hits, mut ctrl) = (Vec::with_capacity(n), 0usize, 0f64);
    let (mut agree, mut kl_sum, mut kl_max) = (0usize, 0f64, 0f64);
    let (mut margins, mut ref_margins, mut flips) = (Vec::new(), Vec::new(), Vec::new());
    let mut ref_logits = vec![0f32; v];
    let mut bytes = vec![0u8; v * 4];
    for p in 0..n - 1 {
        let logits = model.head_logits(&rows[p * h..(p + 1) * h])?;
        let lp = log_softmax(&logits);
        let next = toks[p + 1] as usize;
        nll.push(-lp[next]);
        let top = argmax(&logits) as usize;
        if top == next {
            hits += 1;
        }
        if p + 2 < n {
            ctrl += -lp[toks[p + 2] as usize];
        }
        margins.push(top2_margin(&logits));
        if let Some(d) = dump.as_mut() {
            for (b, x) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(&logits) {
                *b = x.to_le_bytes();
            }
            d.write_all(&bytes).map_err(|e| format!("dump: {e}"))?;
        }
        if let Some(a) = against.as_mut() {
            a.read_exact(&mut bytes)
                .map_err(|e| format!("against: {e}"))?;
            for (x, b) in ref_logits.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *x = f32::from_le_bytes(*b);
            }
            let rlp = log_softmax(&ref_logits);
            let kl: f64 = rlp.iter().zip(&lp).map(|(&r, &q)| r.exp() * (r - q)).sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let rtop = argmax(&ref_logits) as usize;
            let rm = top2_margin(&ref_logits);
            ref_margins.push(rm);
            if rtop == top {
                agree += 1;
            } else {
                flips.push(json!({"pos": p, "ref_argmax": rtop, "argmax": top,
                                  "ref_margin": rm, "ref_is_next": rtop == next, "is_next": top == next}));
            }
        }
    }
    if let Some(d) = dump.as_mut() {
        d.flush().map_err(|e| format!("dump: {e}"))?;
    }
    let scored = nll.len();
    let mean = nll.iter().sum::<f64>() / scored as f64;
    let acc = hits as f64 / scored as f64;
    let ctrl_mean = ctrl / (n - 2) as f64;
    let ctrl_ok = ctrl_mean >= mean + 1.0;
    // The FP8 run must reproduce the record (same bars as qwen38_prefill score); the MXFP4 run
    // is the study and has no bar, only the control.
    let ref_ok = (mean - ref_mean).abs() <= 0.02 * ref_mean && (acc - ref_acc).abs() <= 0.02;
    let pass = ctrl_ok && (weights != "fp8" || ref_ok);
    let mut sorted = ref_margins.clone();
    sorted.sort_by(f64::total_cmp);
    let mut j = json!({
        "mode": "drift", "weights": weights, "record": record_path, "tokens": n, "scored": scored,
        "chunk": chunk, "pass_ms": pass_ms, "stats": stats_json(&st),
        "mean_nll": mean, "perplexity": mean.exp(), "next_token_accuracy": acc,
        "record_mean_nll": ref_mean, "record_perplexity": ref_mean.exp(),
        "record_next_token_accuracy": ref_acc, "control_mean_nll_two_ahead": ctrl_mean,
        "control_ok": ctrl_ok, "record_ok": ref_ok,
        "top2_margin_min": margins.iter().copied().fold(f64::INFINITY, f64::min),
        "nll": nll, "pass": pass,
    });
    if weights == "mxfp4" {
        j["mxfp4"] = json!({"group": w4::GROUP, "tensors": w4s.tensors, "weights": w4s.weights,
            "groups": w4s.groups, "underflow": w4s.underflow,
            "relative_rms_weight_error": (w4s.sq_err / w4s.sq_ref).sqrt()});
    }
    if against.is_some() {
        j["against"] = json!({
            "file": flag(args, "--against"), "argmax_agree": agree, "argmax_agree_rate": agree as f64 / scored as f64,
            "kl_mean": kl_sum / scored as f64, "kl_max": kl_max, "flips": flips,
            "ref_margin_min": sorted.first(), "ref_margin_median": sorted.get(sorted.len() / 2),
        });
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&j).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{out}: {e}"))?;
    println!(
        "QWEN38_SPEC_DRIFT weights {weights}: mean NLL {mean:.5} (record {ref_mean:.5}), perplexity {:.5}, accuracy {acc:.4}, control {ctrl_mean:.3}{}: {}",
        mean.exp(),
        if against.is_some() {
            format!(", argmax agree {agree}/{scored}, KL mean {:.3e} max {kl_max:.3e}", kl_sum / scored as f64)
        } else {
            String::new()
        },
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("probe") => probe(&args[1..]),
        Some("gen") => gen(&args[1..]),
        Some("drift") => drift(&args[1..]),
        _ => Err("usage: qwen38_spec probe|gen|drift ... (see the file header)".into()),
    };
    match r {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen38_spec: {e}");
            std::process::exit(2);
        }
    }
}
