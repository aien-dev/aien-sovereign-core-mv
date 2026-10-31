//! OWN-30B self-draft acceptance (aienix#4, Heartbeat H0): the 30B drafts with only a resident
//! subset of experts per layer, the exact full-router 30B verifies.
//!
//!   own30b_selfdraft <checkpoint> <receipt-dir> [--new 256] [--rs 16,32,64] [--ks 4,8,16,32]
//!                    [--max-context 4096] [--cross-check 3] [--force-memory]
//!
//! 1. Calibration: greedy decode of the CALIB prompts (distinct from the EVAL prompts) on the
//!    full router with `expert_counts` on; per layer the R most chosen experts (ties: lower id)
//!    form the resident set.
//! 2. Reference: per EVAL prompt, the full model's greedy continuation T (token-at-a-time
//!    `step`, stop at EOS or `--new`).
//! 3. Run lengths: at every position p of T, from the full-model KV of prompt + T[..p], the
//!    masked model drafts greedily (`set_draft_experts`) until it misses T or 32 tokens; L_R(p)
//!    = matched tokens. Greedy verify is exact, so a block of k drafted at p accepts
//!    a = min(L_R(p), k) and advances a + 1 tokens (the verifier's own next token).
//! 4. Simulation per (R, k): p = 0; while p < |T|: blocks += 1; p += min(L, k) + 1.
//!    Tokens per block = |T| / blocks.
//! 5. Cross-check on the first `--cross-check` prompts at k = 8: the literal loop (draft k with
//!    the mask, rewind, full-model step + logits over the drafted tokens, accept the matching
//!    prefix + 1, rewind) must give the same blocks and the same tokens as T.
//!
//! Writes run.txt and run.json into the receipt dir.
use std::path::PathBuf;
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::{
    sample_next, AttnPath, DevicePaths, GenerateParams, Qwen3MoeOmegaEngine, WeightFormat,
};
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;

const CALIB: [&str; 8] = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator moves heat from inside to outside, step by step.",
    "Write a Rust function that parses a CSV line with quoted fields, and explain it.",
    "List ten practical tips for keeping a vegetable garden healthy through a dry summer.",
    "Describe the causes and consequences of the fall of the Western Roman Empire.",
    "Solve: a train leaves at 3 pm at 80 km/h, another at 4 pm at 100 km/h on the same track. When does the second catch the first? Show the working.",
    "Translate into French and then explain the grammar: 'We would have gone if it had not rained.'",
    "Write a product description for a solar-powered camping lantern.",
];

const EVAL: [&str; 20] = [
    "Write a poem about the ocean at night.",
    "Explain what a hash table is and when you would use one.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with comments.",
    "What are the main differences between a diesel and a petrol engine?",
    "Give me a recipe for a simple tomato soup.",
    "Summarize the plot of Romeo and Juliet in plain words.",
    "How does a bicycle stay upright while moving?",
    "Write a polite email asking a landlord to fix a leaking tap.",
    "Explain the Pythagorean theorem with an example.",
    "What should I check before buying a used car?",
    "Write a short dialogue between a robot and a child who meet for the first time.",
    "Describe how photosynthesis works.",
    "Write a C function that reverses a string in place, and explain each line.",
    "What are good habits for getting better sleep?",
    "Explain inflation to a twelve-year-old.",
    "Compare TCP and UDP and say when each is used.",
    "Write a cover letter opening paragraph for a mechanical engineering job.",
    "Explain how a transistor works as a switch.",
    "Plan a three-day walking trip in the mountains, with a packing list.",
    "Write a JSON object describing a library book, then explain each field.",
];

/// Draft run lengths are counted up to this many tokens (the largest block size).
const L_CAP: usize = 32;

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

fn chat(tok: &Tokenizer, user: &str) -> Vec<u32> {
    tok.encode_with_special(
        &format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
        false,
    )
    .expect("encode")
}

struct Ctx {
    greedy: GenerateParams,
    eos: Vec<u32>,
}

impl Ctx {
    fn next(&self, e: &mut Qwen3MoeOmegaEngine, token: u32) -> u32 {
        let h = e.step(token).expect("step");
        let lg = e.logits(&h).expect("logits");
        sample_next(&lg, &self.greedy, 0)
    }
}

/// Prefill all but the last prompt token (KV only); returns the last token (not yet stepped).
fn prefill(e: &mut Qwen3MoeOmegaEngine, prompt: &[u32]) -> u32 {
    e.reset();
    let body = &prompt[..prompt.len() - 1];
    for chunk in body.chunks(e.prefill_chunk.max(1)) {
        e.step_batch(chunk).expect("prefill");
    }
    *prompt.last().unwrap()
}

/// Full-model greedy continuation, token at a time, stop at EOS (not included) or `n`.
fn reference(e: &mut Qwen3MoeOmegaEngine, cx: &Ctx, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut last = prefill(e, prompt);
    let mut t = Vec::with_capacity(n);
    while t.len() < n {
        let nx = cx.next(e, last);
        if cx.eos.contains(&nx) {
            break;
        }
        t.push(nx);
        last = nx;
    }
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = PathBuf::from(args.get(1).expect("checkpoint dir"));
    let receipt = PathBuf::from(args.get(2).expect("receipt dir"));
    let (mut new, mut max_context, mut force, mut cross) = (256usize, 4096usize, false, 3usize);
    let (mut rs, mut ks) = (vec![16usize, 32, 64], vec![4usize, 8, 16, 32]);
    let list = |s: &str| -> Vec<usize> { s.split(',').map(|v| v.parse().expect("n")).collect() };
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--new" => new = it.next().expect("--new").parse().expect("n"),
            "--rs" => rs = list(it.next().expect("--rs")),
            "--ks" => ks = list(it.next().expect("--ks")),
            "--max-context" => max_context = it.next().expect("n").parse().expect("n"),
            "--cross-check" => cross = it.next().expect("n").parse().expect("n"),
            "--force-memory" => force = true,
            o => panic!("unknown argument {o}"),
        }
    }
    assert!(ks.iter().all(|&k| k <= L_CAP), "block size above {L_CAP}");
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !force {
        say("REFUSED: less than 70 GB available; --force-memory to override".into());
        std::process::exit(3);
    }
    if let Some(us) =
        aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref())
            .expect("spin env")
    {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    say(format!(
        "build native={} spin_us={:?}",
        aien_omega_gpu::is_native(),
        aien_omega_gpu::spin_us()
    ));
    let tok = Tokenizer::from_file(checkpoint.join("tokenizer.json")).expect("tokenizer");
    let t0 = Instant::now();
    let (mut e, report) = Qwen3MoeOmegaEngine::load_omega_paths(
        &checkpoint,
        max_context,
        WeightFormat::Fp8All,
        AttnPath::Fused,
        DevicePaths::default(),
        &mut |_| {},
    )
    .expect("load on Omega");
    let (layers, experts) = (e.config().layers, e.config().experts);
    let cx = Ctx {
        greedy: GenerateParams::greedy(new, Vec::new()),
        eos: e.config().eos_token_ids.clone(),
    };
    say(format!(
        "LOAD {:.1}s resident {:.2} GB attention {} device_routing {}",
        t0.elapsed().as_secs_f64(),
        report.resident_bytes as f64 / 1e9,
        e.attention_path(),
        e.device_routing
    ));

    // 1. Calibration trace.
    let t1 = Instant::now();
    e.expert_counts = Some(vec![0; layers * experts]);
    let mut calib_tokens = 0usize;
    for p in CALIB {
        calib_tokens += reference(&mut e, &cx, &chat(&tok, p), new).len();
    }
    let counts = e.expert_counts.take().unwrap();
    let mut residents: Vec<(usize, Vec<Vec<u32>>, f64)> = Vec::new();
    for &r in &rs {
        let mut sets = Vec::with_capacity(layers);
        let (mut covered, mut total) = (0u64, 0u64);
        for l in 0..layers {
            let row = &counts[l * experts..(l + 1) * experts];
            let mut ids: Vec<u32> = (0..experts as u32).collect();
            ids.sort_by(|&a, &b| row[b as usize].cmp(&row[a as usize]).then(a.cmp(&b)));
            ids.truncate(r);
            covered += ids.iter().map(|&x| row[x as usize]).sum::<u64>();
            total += row.iter().sum::<u64>();
            sets.push(ids);
        }
        let cov = covered as f64 / total.max(1) as f64;
        say(format!(
            "CALIB R={r} trace-coverage {:.4} ({covered}/{total} routed picks)",
            cov
        ));
        residents.push((r, sets, cov));
    }
    say(format!(
        "CALIB {} prompts, {calib_tokens} decode tokens traced, {:.1}s",
        CALIB.len(),
        t1.elapsed().as_secs_f64()
    ));

    // 2 + 3. Reference and draft run lengths per position.
    // runs[ri][prompt] = L_R(p) for every p in T.
    let mut runs: Vec<Vec<Vec<usize>>> = vec![Vec::new(); residents.len()];
    let mut refs: Vec<Vec<u32>> = Vec::new();
    let t2 = Instant::now();
    for (pi, text) in EVAL.iter().enumerate() {
        let prompt = chat(&tok, text);
        let t = reference(&mut e, &cx, &prompt, new);
        let base = prompt.len() - 1; // KV positions holding the committed prefix minus its last token
        let mut per_r: Vec<Vec<usize>> = vec![Vec::with_capacity(t.len()); residents.len()];
        let mut last = prefill(&mut e, &prompt);
        for p in 0..t.len() {
            for (ri, (_, sets, _)) in residents.iter().enumerate() {
                e.set_draft_experts(Some(sets.clone())).expect("mask");
                let mut cur = last;
                let mut l = 0usize;
                while l < L_CAP && p + l < t.len() {
                    let d = cx.next(&mut e, cur);
                    if d != t[p + l] {
                        break;
                    }
                    cur = d;
                    l += 1;
                }
                per_r[ri].push(l);
                e.set_draft_experts(None).expect("unmask");
                e.rewind(base + p);
            }
            // Advance the full-model KV by the committed token and check it reproduces T.
            let v = cx.next(&mut e, last);
            assert_eq!(
                v, t[p],
                "prompt {pi} position {p}: full model not deterministic"
            );
            last = v;
        }
        say(format!(
            "EVAL prompt {pi:02} prompt_tokens {} generated {} mean L {}",
            prompt.len(),
            t.len(),
            per_r
                .iter()
                .map(|v| format!(
                    "{:.2}",
                    v.iter().sum::<usize>() as f64 / v.len().max(1) as f64
                ))
                .collect::<Vec<_>>()
                .join("/")
        ));
        for (ri, v) in per_r.into_iter().enumerate() {
            runs[ri].push(v);
        }
        refs.push(t);
    }
    say(format!(
        "EVAL run lengths {:.1}s",
        t2.elapsed().as_secs_f64()
    ));

    // 4. Simulation.
    let simulate = |l: &[usize], k: usize| -> usize {
        let (mut p, mut blocks) = (0usize, 0usize);
        while p < l.len() {
            blocks += 1;
            p += l[p].min(k) + 1;
        }
        blocks
    };
    let total_tokens: usize = refs.iter().map(|t| t.len()).sum();
    let mut table = Vec::new();
    for (ri, (r, _, cov)) in residents.iter().enumerate() {
        for &k in &ks {
            let blocks: usize = runs[ri].iter().map(|l| simulate(l, k)).sum();
            let accepted: usize = total_tokens - blocks;
            let tpb = total_tokens as f64 / blocks as f64;
            say(format!(
                "ACCEPT R={r} k={k} tokens {total_tokens} blocks {blocks} accepted_draft {accepted} tokens_per_block {tpb:.3} accept_rate {:.4}",
                accepted as f64 / (blocks * k) as f64
            ));
            table.push(serde_json::json!({
                "R": r, "k": k, "tokens": total_tokens, "blocks": blocks,
                "accepted_draft_tokens": accepted, "trace_coverage": cov,
            }));
        }
    }

    // 5. Literal cross-check at k = 8.
    let k = 8usize;
    let mut cross_ok = true;
    for (pi, text) in EVAL.iter().enumerate().take(cross) {
        let prompt = chat(&tok, text);
        let base = prompt.len() - 1;
        for (ri, (r, sets, _)) in residents.iter().enumerate() {
            let t = &refs[pi];
            let mut last = prefill(&mut e, &prompt);
            let (mut out, mut blocks) = (Vec::<u32>::new(), 0usize);
            while out.len() < t.len() {
                blocks += 1;
                let committed = base + out.len();
                // Draft k with the mask.
                e.set_draft_experts(Some(sets.clone())).expect("mask");
                let mut draft = Vec::with_capacity(k);
                let mut cur = last;
                for _ in 0..k {
                    cur = cx.next(&mut e, cur);
                    draft.push(cur);
                }
                e.set_draft_experts(None).expect("unmask");
                e.rewind(committed);
                // Verify exactly: full-model step + logits over last, d1, ..., stop at a miss.
                let mut cur = last;
                let mut i = 0usize;
                loop {
                    let v = cx.next(&mut e, cur);
                    out.push(v);
                    if i < k && draft[i] == v && out.len() < t.len() {
                        cur = v;
                        i += 1;
                    } else {
                        break;
                    }
                }
                last = *out.last().unwrap();
                // KV holds committed + accepted + 1 positions; the new last token is not stepped.
                e.rewind(base + out.len());
            }
            out.truncate(t.len());
            let want = simulate(&runs[ri][pi], k);
            let ok = &out == t && blocks == want;
            cross_ok &= ok;
            say(format!(
                "CROSS prompt {pi:02} R={r} k={k} literal blocks {blocks} simulated {want} tokens_match {} {}",
                &out == t,
                if ok { "OK" } else { "MISMATCH" }
            ));
        }
    }
    say(format!(
        "SUMMARY eval_prompts {} tokens {total_tokens} cross_check {}",
        EVAL.len(),
        if cross_ok { "PASS" } else { "FAIL" }
    ));
    let json = serde_json::json!({
        "calib_prompts": CALIB.len(), "calib_decode_tokens": calib_tokens,
        "eval_prompts": EVAL.len(), "new": new, "table": table, "cross_check": cross_ok,
        "resident_sets": residents.iter().map(|(r, s, _)| serde_json::json!({"R": r, "sets": s})).collect::<Vec<_>>(),
        "runs": runs,
    });
    std::fs::write(
        receipt.join("run.json"),
        serde_json::to_string(&json).unwrap(),
    )
    .unwrap();
    std::fs::write(receipt.join("run.txt"), log.join("\n") + "\n").unwrap();
    if !cross_ok {
        std::process::exit(1);
    }
}
