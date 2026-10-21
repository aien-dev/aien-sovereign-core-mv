//! Stroke gate: N conversations per stroke on the chip, judged word for word (L312, the motor).
//!
//! ```text
//!   qwen36_stroke_gate <checkpoint-dir> <out.json> <golden.json>...   (2..=4 goldens, one per head)
//! ```
//!
//! Every golden is a `qwen38_check generate` record (`prompt_ids`, `generated_ids`); conversation
//! `c` lives in head slot `c` of ONE multi-head graph ([`Qwen38Omega::open_window_heads`]). Each
//! step is one [`Qwen38Omega::stroke`]: every head takes its own next token (teacher forcing with
//! the reference's own tokens, as `qwen38_omega_full`) in one command list, one submit. A head
//! whose sequence has ended keeps taking its last token; nothing of it is judged after its end.
//!
//! The rule is `qwen38_omega_full`'s, applied to every conversation: words identical to its golden
//! except where the host reference's top-two margin is below the written band (`MARGIN_MIN`, fixed
//! in advance, never from the run under test); the chip's greedy choice agrees with the reference
//! at every position outside the band; the shifted-by-one control is caught; the reference equals
//! the golden. Logit cosine and max error are recorded, not gated (open precision item).
//!
//! Negative controls, both must FAIL this gate:
//! - `AIEN_Q36_TEST_MUTANT=4` (the shared expert dropped), as gate 1;
//! - `AIEN_STROKE_SWAP=control`: half way, slots 0 and 1 exchange their state objects
//!   ([`Qwen38Omega::swap_heads`]) while the feeds stay: each continues the OTHER conversation.
//!
//! Head swap as a feature, must PASS: `AIEN_STROKE_SWAP=feature` does the same exchange AND moves
//! the feeds with it (conversation 0 now lives in slot 1 and is fed and read there).
//!
//! The tool takes the hive's chip lock itself around the chip part. Inside a `spark-window` (whose
//! holder already owns that lock) set `AIEN_STROKE_IN_WINDOW=1` to skip it; never elsewhere.

use std::path::Path;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Config, Qwen38Reference};
use aien_inference_abi::qwen38_omega::{mem_available, window_bytes, Qwen38Omega, KEEP_FREE};
use aien_inference_abi::qwen3_moe::parity_stats;
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use serde_json::{json, Value};

/// Recorded, not gated (see the crate's `qwen38_omega_full`).
const COSINE_MIN: f64 = 0.9999;
/// The written band (`qwen38_omega_full::MARGIN_MIN`): fixed 2026-10-21 02:10Z, never from the run.
const MARGIN_MIN: f32 = 0.1;
/// How long the memory wait may go on, with or without the lock, before the tool refuses.
const MEMORY_WAIT: Duration = Duration::from_secs(40 * 60);

fn margin(logits: &[f32]) -> f32 {
    let top = argmax(logits);
    let second = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != top)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    logits[top] - second
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("golden: no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("golden: {key} holds a non-id"))
        })
        .collect()
}

/// One conversation: its golden, the teacher-forced sequence and where generation starts.
struct Conv {
    name: String,
    prompt: Vec<u32>,
    generated: Vec<u32>,
    /// `prompt ++ generated[..n-1]`: what is fed; position `p` predicts `seq[p + 1]`.
    seq: Vec<u32>,
    /// The position whose prediction is the first generated token.
    first_gen: usize,
}

fn load_conv(path: &str) -> Result<Conv, String> {
    let golden: Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
            .map_err(|e| format!("{path}: {e}"))?;
    let prompt = ids(&golden, "prompt_ids")?;
    let generated = ids(&golden, "generated_ids")?;
    if prompt.is_empty() || generated.is_empty() {
        return Err(format!("{path}: empty prompt or continuation"));
    }
    let mut seq = prompt.clone();
    seq.extend_from_slice(&generated[..generated.len() - 1]);
    let first_gen = prompt.len() - 1;
    let name = Path::new(path)
        .file_stem()
        .map_or(path.to_string(), |s| s.to_string_lossy().into_owned());
    Ok(Conv {
        name,
        prompt,
        generated,
        seq,
        first_gen,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Swap {
    None,
    /// Exchange the states of slots 0 and 1, feeds unchanged: must be caught.
    Control,
    /// Exchange the states and move the feeds with them: must pass.
    Feature,
}

/// The hive's chip lock (`/tmp/aien-gb10.lock`, taken as aien-serve-http takes it): held from the
/// upload to the end of the strokes and released when the file drops. The host reference runs
/// before it, so the chip is never held idle; the memory wait runs before it too.
/// This binary's sha256, stamped on every receipt: the script's start line is not enough once the
/// binary is rebuilt while a run is queued.
fn self_sha256() -> String {
    use sha2::Digest;
    match std::fs::read("/proc/self/exe") {
        Ok(b) => format!("{:x}", sha2::Sha256::digest(&b)),
        Err(e) => format!("unreadable: {e}"),
    }
}

fn chip_lock() -> Result<std::fs::File, String> {
    use std::os::fd::AsRawFd;
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open("/tmp/aien-gb10.lock")
        .map_err(|e| format!("/tmp/aien-gb10.lock: {e}"))?;
    // SAFETY: a live fd; flock blocks until the lock is ours and the kernel releases it on close.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(format!(
            "/tmp/aien-gb10.lock: flock: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(f)
}

/// Waits (no lock held) until the box has `need` plus KEEP_FREE available, at most MEMORY_WAIT:
/// test servers on 18093 come and go, and an upload into a full box reaches the OOM killer.
fn wait_for_memory(need: u64) -> Result<u64, String> {
    let t0 = Instant::now();
    let mut said = 0u64;
    loop {
        let avail = mem_available()?;
        if avail >= need + KEEP_FREE {
            return Ok(avail);
        }
        let s = t0.elapsed().as_secs();
        if t0.elapsed() >= MEMORY_WAIT {
            return Err(format!(
                "the whole stack needs {} MiB with {} MiB kept free, MemAvailable {} MiB after {s} s; refusing",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        if s >= said {
            eprintln!(
                "waiting for memory: need {} MiB + {} MiB free, MemAvailable {} MiB ({s} s)",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            );
            said = s + 120;
        }
        std::thread::sleep(Duration::from_secs(10));
    }
}

fn median(v: &mut [u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: qwen36_stroke_gate <checkpoint-dir> <out.json> <golden.json>... (2..=4)";
    if args.len() < 5 || args.len() > 7 {
        return Err(usage.into());
    }
    let dir = Path::new(&args[1]);
    let out_path = &args[2];
    let convs: Vec<Conv> = args[3..]
        .iter()
        .map(|p| load_conv(p))
        .collect::<Result<_, _>>()?;
    let heads = convs.len();
    let bin_sha = self_sha256();
    eprintln!("binary sha256 {bin_sha}");
    let swap = match std::env::var("AIEN_STROKE_SWAP").ok().as_deref() {
        None | Some("") => Swap::None,
        Some("control") => Swap::Control,
        Some("feature") => Swap::Feature,
        Some(v) => return Err(format!("AIEN_STROKE_SWAP={v}: control, feature or unset")),
    };

    let config = Qwen38Config::from_checkpoint(dir)?;
    let layers = config.layers;
    let vocab = config.vocab;
    let t_max = convs.iter().map(|c| c.seq.len()).max().unwrap();
    let t_min = convs.iter().map(|c| c.seq.len()).min().unwrap();
    // The swap lands half way through the shortest conversation: both are mid-sequence.
    let swap_at = t_min / 2;
    eprintln!(
        "model: {layers} layers, {heads} heads; positions {}; swap {}",
        convs
            .iter()
            .map(|c| format!("{} {}", c.name, c.seq.len()))
            .collect::<Vec<_>>()
            .join(", "),
        match swap {
            Swap::None => "none".to_string(),
            Swap::Control => format!("CONTROL at step {swap_at} (slots 0<->1, feeds stay)"),
            Swap::Feature => format!("feature at step {swap_at} (slots 0<->1 with their feeds)"),
        }
    );

    // The host reference of every conversation: logits at every position.
    let t0 = Instant::now();
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(heads);
    {
        let mut r = Qwen38Reference::new(config.clone(), ShardSource::open(dir)?)?;
        for c in &convs {
            let mut state = r.new_state();
            reference.push(r.forward(&mut state, &c.seq, Logits::All)?);
        }
    }
    let ref_s = t0.elapsed().as_secs_f64();
    let ref_matches_golden: Vec<bool> = convs
        .iter()
        .zip(&reference)
        .map(|(c, r)| {
            (0..c.generated.len()).all(|i| {
                argmax(&r[(c.first_gen + i) * vocab..(c.first_gen + i + 1) * vocab])
                    == c.generated[i] as usize
            })
        })
        .collect();
    eprintln!(
        "reference: {heads} conversations in {ref_s:.1} s; greedy tokens {} the golden records",
        if ref_matches_golden.iter().all(|&b| b) {
            "equal"
        } else {
            "DIFFER FROM"
        }
    );

    // The chip: every conversation in its own head slot, one stroke per step. Uploaded only when
    // the box keeps KEEP_FREE after it (the 18091 server shares the pool; never starve it), and
    // only under the hive's chip lock.
    let need = window_bytes(&config, 0, layers);
    let in_window = std::env::var("AIEN_STROKE_IN_WINDOW").is_ok_and(|v| v == "1");
    // Memory moves between the wait and the lock (another lane's server loads while this process
    // queues for the lock), so: wait, lock, look again under the lock; if short, let the lock go
    // (the lane holding the memory may need the chip to finish) and wait on, up to MEMORY_WAIT.
    let t_mem = Instant::now();
    let lock = loop {
        let avail = wait_for_memory(need)?;
        if in_window {
            eprintln!(
                "AIEN_STROKE_IN_WINDOW=1: the spark-window holds the chip lock; not taking it"
            );
            break None;
        }
        eprintln!(
            "memory: MemAvailable {} MiB for {} MiB; taking the chip lock",
            avail >> 20,
            need >> 20
        );
        let lock = chip_lock()?;
        let avail = mem_available()?;
        if avail >= need + KEEP_FREE {
            break Some(lock);
        }
        eprintln!(
            "under the lock MemAvailable is {} MiB (need {} + {} MiB); letting the lock go, waiting on",
            avail >> 20,
            need >> 20,
            KEEP_FREE >> 20
        );
        drop(lock);
        if t_mem.elapsed() >= MEMORY_WAIT {
            return Err(format!(
                "memory never stayed free under the lock within {} min; refusing",
                MEMORY_WAIT.as_secs() / 60
            ));
        }
        std::thread::sleep(Duration::from_secs(10));
    };
    let t1 = Instant::now();
    let mut engine = Qwen38Omega::open_window_heads(dir, 0, layers, t_max.max(256), heads)?;
    let upload_s = t1.elapsed().as_secs_f64();
    eprintln!(
        "graph: {heads} heads uploaded in {upload_s:.1} s, weights {} MB, KV {} MB per head",
        engine.weight_bytes() >> 20,
        engine.kv_bytes() >> 20
    );
    let mut slot_of: Vec<usize> = (0..heads).collect();
    let mut tokens: Vec<Vec<u32>> = convs
        .iter()
        .map(|c| Vec::with_capacity(c.seq.len()))
        .collect();
    let mut gpu_logits: Vec<Vec<Vec<f32>>> = (0..heads).map(|_| Vec::new()).collect();
    let mut stroke_ns = Vec::with_capacity(t_max);
    let mut swapped_at = None;
    for p in 0..t_max {
        if p == swap_at && swap != Swap::None {
            engine.swap_heads(0, 1)?;
            if swap == Swap::Feature {
                slot_of.swap(0, 1);
            }
            swapped_at = Some(p);
        }
        let mut feed = vec![0u32; heads];
        for (c, conv) in convs.iter().enumerate() {
            feed[slot_of[c]] = conv.seq[p.min(conv.seq.len() - 1)];
        }
        let (out, st) = engine.stroke(&feed)?;
        stroke_ns.push(st.chip_ns);
        for (c, conv) in convs.iter().enumerate() {
            if p < conv.seq.len() {
                tokens[c].push(out[slot_of[c]]);
                gpu_logits[c].push(engine.logits_head(slot_of[c])?);
            }
        }
    }
    drop(engine);
    drop(lock);

    // Judge every conversation by the written rule.
    let mut convs_report = Vec::with_capacity(heads);
    let mut all_pass = true;
    for (c, conv) in convs.iter().enumerate() {
        let t = conv.seq.len();
        let r = &reference[c];
        let (mut cos_min, mut p95_max, mut greedy_ok) = (f64::INFINITY, 0f32, true);
        let (mut flips_allowed, mut flips_beyond) = (Vec::new(), Vec::new());
        let mut gen_tokens = Vec::new();
        let mut steps = Vec::with_capacity(t);
        for (p, (logits, &tok)) in gpu_logits[c].iter().zip(&tokens[c]).enumerate() {
            let expected = &r[p * vocab..(p + 1) * vocab];
            let s = parity_stats(logits, expected);
            let (ref_tok, m) = (argmax(expected), margin(expected));
            let tok = tok as usize;
            let agree = tok == ref_tok && argmax(logits) == tok;
            if m >= MARGIN_MIN && !agree {
                greedy_ok = false;
            }
            let gold = (p >= conv.first_gen).then(|| conv.generated[p - conv.first_gen] as usize);
            if let Some(g) = gold {
                gen_tokens.push(tok);
                if tok != g {
                    let flip = json!({"pos": p, "token": tok, "golden": g, "ref_margin": m, "max_abs": s.max_abs});
                    if m < MARGIN_MIN {
                        flips_allowed.push(flip);
                    } else {
                        flips_beyond.push(flip);
                    }
                }
            }
            cos_min = cos_min.min(if s.non_finite > 0 {
                f64::NEG_INFINITY
            } else {
                s.cosine
            });
            p95_max = p95_max.max(s.p95_rel);
            println!(
                "{} step {p:3}: token {tok:6} ref {ref_tok:6}{} margin {m:7.3} cosine {:.6} p95_rel {:.4} max_abs {:.4}{}",
                conv.name,
                gold.map_or(String::new(), |g| format!(" golden {g:6}")),
                s.cosine,
                s.p95_rel,
                s.max_abs,
                if gold.is_some_and(|g| g != tok) { "  GOLDEN DIFFERS" } else if agree { "" } else { "  (differs)" }
            );
            steps.push(json!({"pos": p, "token": tok, "ref_token": ref_tok, "golden": gold, "ref_margin": m,
                "cosine": s.cosine, "p95_rel": s.p95_rel, "max_abs": s.max_abs, "non_finite": s.non_finite}));
        }
        let shift_cos = (0..t - 1)
            .map(|p| parity_stats(&gpu_logits[c][p], &r[(p + 1) * vocab..(p + 2) * vocab]).cosine)
            .fold(f64::NEG_INFINITY, f64::max);
        let caught = shift_cos < COSINE_MIN;
        let words_ok = flips_beyond.is_empty();
        let text_ok = gen_tokens.len() == conv.generated.len();
        let cosine_floor_held = cos_min >= COSINE_MIN;
        let pass = words_ok && greedy_ok && text_ok && caught && ref_matches_golden[c];
        all_pass &= pass;
        let identical = gen_tokens
            .iter()
            .zip(&conv.generated)
            .filter(|(a, b)| **a == **b as usize)
            .count();
        println!(
            "{} (slot {} -> {}): {identical}/{} identical to golden ({} flips inside the written band {MARGIN_MIN}, {} beyond it); greedy {} the reference outside the band; cosine min {cos_min:.6} ({}); shifted-by-one {} -> {}",
            conv.name,
            c,
            slot_of[c],
            conv.generated.len(),
            flips_allowed.len(),
            flips_beyond.len(),
            if greedy_ok { "agrees with" } else { "DIFFERS FROM" },
            if cosine_floor_held { "held" } else { "NOT held: open precision item" },
            if caught { "caught" } else { "NOT CAUGHT" },
            if pass { "PASS" } else { "FAIL" }
        );
        convs_report.push(json!({
            "name": conv.name, "slot_start": c, "slot_end": slot_of[c], "positions": t,
            "prompt_ids": conv.prompt, "golden_ids": conv.generated, "chip_ids": gen_tokens,
            "reference_matches_golden": ref_matches_golden[c],
            "cosine_min": cos_min, "p95_rel_max": p95_max, "greedy_agrees": greedy_ok,
            "words_identical_outside_margin_band": words_ok,
            "flips_allowed": flips_allowed, "flips_beyond_error": flips_beyond,
            "cosine_floor_held": cosine_floor_held, "control_shift_best_cosine": shift_cos,
            "control_caught": caught, "steps": steps, "pass": pass,
        }));
    }

    let mut ns = stroke_ns.clone();
    let med = median(&mut ns);
    println!(
        "QWEN36_STROKE_GATE heads {heads} strokes {}: chip per stroke median {:.3} ms ({} tokens per stroke; shared chip, a receipt only under spark-window); swap {}; {}",
        stroke_ns.len(),
        med as f64 / 1e6,
        heads,
        match (swap, swapped_at) {
            (Swap::None, _) => "none".to_string(),
            (Swap::Control, Some(p)) => format!("CONTROL at step {p} (expected FAIL)"),
            (Swap::Feature, Some(p)) => format!("feature at step {p} (expected PASS)"),
            (_, None) => "requested but never reached".to_string(),
        },
        if all_pass { "PASS" } else { "FAIL" }
    );
    let report = json!({
        "checkpoint": dir.display().to_string(), "layers": layers, "heads": heads,
        "reference_s": ref_s, "upload_s": upload_s,
        "bin_sha256": bin_sha,
        "in_window": in_window,
        "swap_mode": match swap { Swap::None => "none", Swap::Control => "control", Swap::Feature => "feature" },
        "swapped_at_step": swapped_at,
        "stroke_chip_ns": stroke_ns, "stroke_chip_ns_median": med,
        "thresholds": {"cosine_min_recorded": COSINE_MIN, "margin_band_logits": MARGIN_MIN},
        "gate": "every conversation: words identical to its golden except where the reference's top-two margin is below the written band (0.1 logit, fixed in advance, never from the run under test); chip greedy agrees at every position outside the band; shifted-by-one caught; reference equals golden. Controls: AIEN_Q36_TEST_MUTANT=4 and AIEN_STROKE_SWAP=control must FAIL; AIEN_STROKE_SWAP=feature must PASS (L312 stroke v1)",
        "conversations": convs_report, "pass": all_pass,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("{out_path}: {e}"))?;
    Ok(all_pass)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("qwen36_stroke_gate: {e}");
            std::process::exit(2);
        }
    }
}
