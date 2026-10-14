//! OWN-30B speculative decoding chip gate (cut 1, greedy).
//!
//!   own30b_spec_check <checkpoint> <golden.json> <receipt-dir> [--runs 3] [--new 16] [--max-context 8192] [--force-memory]
//!
//! Loads the production configuration (FP8 everywhere, device router, bf16 lm_head GEMV) once, then
//! for each of the golden file's prompts and each run: generates with speculation OFF and ON and
//! compares the tokens (and against the golden list). Reports speculation acceptance and tok/s.
//! Negative control: with the extra-accept mutant on, at least one prompt must differ from OFF,
//! otherwise the identity check itself is not trustworthy and the run FAILS. A refusal or error of
//! any launch is a failure, never "caught". Exit 0 only when everything passes.
//!
//! Memory: refuses with less than 70 GB available (the live server holds a 30 GB model) unless --force-memory.
//! Run it under `flock /tmp/aien-gb10.lock`.
//!
//! W-L4b: `--verify decode|wp|both` picks the verify pass (decode rounding, must be identical; whole-prompt
//! FP8-activation pass, identity reported with the first diverging token). `--wp-prefill` turns the
//! whole-prompt prefill on for every generation (off and on start from the same KV); `--wp-cached`
//! adds its GPU-cached weight copy. Negative control per path: mutant vs the same path's spec-on output.
//!
//! W-L4d: `--verify exact` (or `decode,exact`): the fast exact verify (`AIEN_OWN30B_SPEC_VERIFY=exact`); it must be
//! token-identical like `decode`. Extra numerics control: the exact verify with bf16-truncated attention must differ
//! from OFF on at least one prompt, else FAIL.
//!
//! W-L4e: `--numerics-mutant act-fp8` (default: W8A8 activations inside the verify, the whole-prompt
//! rounding class), `residual-bf16` (residual stream truncated to bf16) or `attn-bf16` (the W-L4d control, never caught
//! on the full model). The golden compare looks only at the golden list's length (16 tokens; never past it). `--require-golden`
//! makes a spec-off difference inside that length a FAIL (default: recorded).

use std::path::PathBuf;

use aien_inference_abi::qwen3_moe_omega::spec::suffix::SuffixDrafter;
use aien_inference_abi::qwen3_moe_omega::spec::{DraftSource, NumericsMutant, SpecConfig, VerifyPath};
use aien_inference_abi::tokenizer::ChatTokenizer;
use aien_inference_abi::qwen3_moe_omega::{AttnPath, DevicePaths, GenerateParams, Qwen3MoeConfig, Qwen3MoeOmegaEngine, WeightFormat};

fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

fn argmax(lg: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in lg.iter().enumerate() {
        if v > lg[best] {
            best = i;
        }
    }
    best as u32
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
}

/// W-L4e cut 4: the first token where `off` differs from the golden list, looking ONLY at the golden
/// length (the golden file holds 16 tokens; a longer generation is not judged past it). `off` ending
/// before the golden list does is a difference at its end.
fn golden_diff(off: &[u32], golden: &[u32]) -> Option<usize> {
    (0..golden.len()).find(|&i| off.get(i) != golden.get(i))
}

fn main() {
    let mut it = std::env::args().skip(1);
    let mut pos: Vec<String> = Vec::new();
    let (mut runs, mut new_tokens, mut max_context, mut force) = (3usize, 16usize, 8192usize, false);
    let (mut verify, mut wp_prefill, mut wp_cached) = ("decode".to_string(), false, false);
    // W-L4c: --drafter lookup|suffix, --suffix-global (past conversations = the OTHER prompts' spec-off
    // outputs), --extra-prompt NAME FILE (a chat prompt tokenized here; repeatable), --prompts a,b (subset).
    let (mut drafter, mut suffix_global) = ("lookup".to_string(), false);
    let mut layers: Option<usize> = None;
    let mut extra: Vec<(String, String)> = Vec::new();
    let mut only: Option<Vec<String>> = None;
    let mut new_extra = 0usize;
    // W-L4e: --require-golden: spec-off differing from the golden list (on the golden length only) is a FAIL.
    let mut require_golden = false;
    // W-L4e cut 1: the numerics control (default act-fp8, the whole-prompt rounding class; attn-bf16 is the W-L4d one, never caught on the full model).
    let mut numerics = NumericsMutant::ActFp8;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--runs" => runs = it.next().expect("--runs N").parse().expect("number"),
            "--new" => new_tokens = it.next().expect("--new N").parse().expect("number"),
            "--max-context" => max_context = it.next().expect("--max-context N").parse().expect("number"),
            "--force-memory" => force = true,
            "--require-golden" => require_golden = true,
            "--numerics-mutant" => numerics = NumericsMutant::parse(&it.next().expect("--numerics-mutant KIND")).unwrap_or_else(|e| panic!("{e}")),
            "--verify" => verify = it.next().expect("--verify decode|wp|both"),
            "--wp-prefill" => wp_prefill = true,
            "--wp-cached" => wp_cached = true,
            "--drafter" => drafter = it.next().expect("--drafter lookup|suffix"),
            "--suffix-global" => suffix_global = true,
            "--layers" => layers = Some(it.next().expect("--layers N").parse().expect("number")),
            "--new-extra" => new_extra = it.next().expect("--new-extra N").parse().expect("number"),
            "--extra-prompt" => {
                let n = it.next().expect("--extra-prompt NAME FILE");
                let fpath = it.next().expect("--extra-prompt NAME FILE");
                extra.push((n, fpath));
            }
            "--prompts" => only = Some(it.next().expect("--prompts a,b").split(',').map(str::to_string).collect()),
            _ => pos.push(a),
        }
    }
    let paths: Vec<VerifyPath> = match verify.as_str() {
        "decode" => vec![VerifyPath::Decode],
        "wp" => vec![VerifyPath::WholePrompt],
        "exact" => vec![VerifyPath::Exact],
        "decode,exact" => vec![VerifyPath::Decode, VerifyPath::Exact],
        "both" => vec![VerifyPath::Decode, VerifyPath::WholePrompt],
        _ => {
            eprintln!("--verify must be decode, wp, both, exact or decode,exact");
            std::process::exit(2);
        }
    };
    let needs_wp = wp_prefill || wp_cached || paths.contains(&VerifyPath::WholePrompt);
    if pos.len() != 3 {
        eprintln!("usage: own30b_spec_check <checkpoint> <golden.json> <receipt-dir> [--runs 3] [--new 16] [--max-context 8192] [--force-memory] [--verify decode|wp|both] [--wp-prefill] [--wp-cached]");
        std::process::exit(2);
    }
    let (checkpoint, golden_path, receipt) = (PathBuf::from(&pos[0]), PathBuf::from(&pos[1]), PathBuf::from(&pos[2]));
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    // W-L4c: the spin window from AIEN_OMEGA_SPIN_US, as own30b_check does (without it omega waits with
    // its 50 us default and plain decode measures ~39 ms/token instead of ~26).
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    say(format!("spin_us {:?}", aien_omega_gpu::spin_us()));
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !force && layers.is_none() {
        say("REFUSED: less than 70 GB available (another model may be loaded); --force-memory to override".into());
        let _ = std::fs::write(receipt.join("run.log"), log.join("\n"));
        std::process::exit(3);
    }
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&golden_path).expect("golden file")).expect("golden json");
    let config = Qwen3MoeConfig::from_checkpoint(&checkpoint).expect("config");
    // The whole-prompt driver needs omega's fused KV cache.
    let attn = if needs_wp { AttnPath::Fused } else { AttnPath::Auto };
    // --layers N (W-L4d rehearsal): only the first N layers (a few GB, safe beside the live server); the tokens are
    // not the model's, but OFF vs ON identity and the controls are still meaningful. Fused KV (the exact verify needs it).
    let (mut engine, report) = match layers {
        Some(n) => Qwen3MoeOmegaEngine::load_omega_first_layers(&checkpoint, n, max_context, WeightFormat::Fp8All, AttnPath::Fused, DevicePaths::default(), &mut |_| {}),
        None => Qwen3MoeOmegaEngine::load_omega_paths(&checkpoint, max_context, WeightFormat::Fp8All, attn, DevicePaths::default(), &mut |_| {}),
    }
    .expect("load on Omega");
    if needs_wp {
        engine.set_whole_prompt(true).expect("whole-prompt driver refused");
        if wp_cached {
            let bytes = engine.whole_prompt_cache_weights().expect("cached weight copy");
            say(format!("whole-prompt cached weight copy: {:.1} GB", bytes as f64 / 1e9));
        }
    }
    say(format!(
        "LOAD {:.1}s attention={} prefill_chunk={} device_routing={} spec_possible={} whole_prompt={} (prefill and wp verify) verify={verify} new={new_tokens}",
        report.seconds,
        engine.attention_path(),
        engine.prefill_chunk,
        engine.device_routing,
        engine.spec_possible(),
        engine.whole_prompt_on(),
    ));
    for &p in &paths {
        if !engine.spec_verify_possible(p) {
            say(format!("FAIL: verify path {p:?} is not available on this engine"));
            let _ = std::fs::write(receipt.join("run.log"), log.join("\n"));
            std::process::exit(1);
        }
    }
    let off = SpecConfig::default();
    // Tuning (K, MAX_N, MIN_N, MIN_ACCEPT) from the AIEN_OWN30B_SPEC_* environment; enabled / verify / mutant set here.
    let mut base = SpecConfig { enabled: false, mutant_accept_extra: false, ..SpecConfig::from_env() };
    base.source = match drafter.as_str() {
        "lookup" => DraftSource::PromptLookup,
        "suffix" => DraftSource::Suffix,
        other => panic!("--drafter {other}: lookup or suffix"),
    };
    say(format!("spec tuning: max_draft={} max_ngram={} min_ngram={} min_accept={} drafter={:?} suffix_global={suffix_global}", base.max_draft, base.max_ngram, base.min_ngram, base.min_accept, base.source));
    if base.source == DraftSource::Suffix {
        say(format!("suffix config: {:?}", base.suffix));
    }
    // Prompts: the golden file's, then --extra-prompt chat prompts (no golden list: spec-off is the reference).
    let mut prompts: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
    for p in golden["prompts"].as_array().expect("prompts") {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let ids: Vec<u32> = p["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want: Vec<u32> =
            p["golden"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).filter(|t| !config.eos_token_ids.contains(t)).collect();
        prompts.push((name, ids, want));
    }
    let tok = if extra.is_empty() { None } else { Some(ChatTokenizer::from_model_dir(&checkpoint, None).expect("tokenizer")) };
    for (n, fpath) in &extra {
        let text = std::fs::read_to_string(fpath).expect("extra prompt file");
        let t = tok.as_ref().unwrap();
        let ids = t.encode_with_special(&t.format_chat(&[("user", text.trim())]), false).expect("tokenize");
        say(format!("extra prompt {n}: {} tokens from {fpath}", ids.len()));
        prompts.push((n.clone(), ids, Vec::new()));
    }
    if let Some(o) = &only {
        prompts.retain(|(n, _, _)| o.iter().any(|x| n.starts_with(x.as_str())));
    }
    // Finished spec-off sequences (prompt + output) per prompt: the "past conversations" for --suffix-global.
    let mut past: Vec<(String, Vec<u32>)> = Vec::new();
    let params = GenerateParams::greedy(new_tokens, config.eos_token_ids.clone());
    let mut all_ok = true;
    let mut rows = Vec::new();
    let mut mutant_caught = vec![false; paths.len()];
    let mut numerics_caught = false;
    let mut numerics_tally = (0usize, 0usize);
    let mut wp_identical = (0usize, 0usize);
    for (name, ids, want) in prompts.iter().cloned() {
        // --new-extra: a longer generation for the --extra-prompt prompts (no golden list).
        let params = if want.is_empty() && new_extra > 0 { GenerateParams::greedy(new_extra, config.eos_token_ids.clone()) } else { params.clone() };
        for run in 1..=runs {
            let go = |cfg: &SpecConfig, engine: &mut Qwen3MoeOmegaEngine| {
                let mut got = Vec::new();
                // --suffix-global: a drafter whose global tree holds the OTHER prompts' finished
                // spec-off sequences (never this prompt's own earlier runs).
                let mut d = (suffix_global && cfg.enabled && cfg.source == DraftSource::Suffix).then(|| {
                    let mut d = SuffixDrafter::new(cfg.suffix.clone());
                    for (n, s) in past.iter().filter(|(n, _)| n != &name) {
                        let _ = n;
                        d.finish(s);
                    }
                    d
                });
                let (o, st) = engine
                    .generate_spec_drafter(&ids, &params, cfg, d.as_mut(), &mut |lg, _| argmax(lg), &mut |t| {
                        got.push(t);
                        true
                    })
                    .expect("generate (an error is a failure, not a pass)");
                let tps = if o.decode_ms > 0 { o.emitted_tokens as f64 * 1000.0 / o.decode_ms as f64 } else { 0.0 };
                (got, st, tps, o.decode_ms)
            };
            let (t_off, _, tps_off, ms_off) = go(&off, &mut engine);
            let golden_cmp = match (want.is_empty(), golden_diff(&t_off, &want)) {
                (true, _) => "no golden list".to_string(),
                (false, None) => format!("IDENTICAL on the golden length {}", want.len()),
                (false, Some(i)) => {
                    if require_golden {
                        all_ok = false;
                    }
                    format!("differs at token {i} of the golden {} ({})", want.len(), if require_golden { "FAIL: --require-golden" } else { "recorded; spec-off is the reference" })
                }
            };
            say(format!("{name} run {run}: OFF {tps_off:.1} tok/s ({ms_off} ms, {} tokens) | off-vs-golden {golden_cmp}", t_off.len()));
            if run == 1 && want.is_empty() {
                if let Some(t) = tok.as_ref() {
                    let text = t.decode(&t_off).unwrap_or_default();
                    say(format!("{name} OFF text (first 400 chars): {:?}", text.chars().take(400).collect::<String>()));
                }
            }
            for (pi, &path) in paths.iter().enumerate() {
                let on = SpecConfig { enabled: true, verify: path, ..base.clone() };
                let (t_on, st, tps_on, ms_on) = go(&on, &mut engine);
                let diff = first_diff(&t_off, &t_on);
                let tokens_per_round = if st.rounds > 0 { (st.accepted + st.rounds) as f64 / st.rounds as f64 } else { 0.0 };
                let verify_ms = if st.rounds > 0 { st.verify_ns as f64 / 1e6 / st.rounds as f64 } else { 0.0 };
                let verdict = match (path, diff) {
                    (_, None) => "IDENTICAL".to_string(),
                    // Decode-rounding verify must be token-identical: a difference is a failure.
                    (VerifyPath::Decode | VerifyPath::Exact, Some(i)) => {
                        all_ok = false;
                        format!("DIFFERENT at token {i} (FAIL)")
                    }
                    // Whole-prompt verify rounds activations to FP8: a difference is reported, not hidden.
                    (VerifyPath::WholePrompt, Some(i)) => format!("DIVERGES at token {i} (FP8-activation verify; reported)"),
                };
                if path == VerifyPath::WholePrompt {
                    wp_identical.1 += 1;
                    if diff.is_none() {
                        wp_identical.0 += 1;
                    }
                }
                say(format!(
                    "{name} run {run}: ON verify={path:?} {tps_on:.1} tok/s ({ms_on} ms) vs off {verdict} | rounds {} wp_rounds {} plain {} backoff {} drafted {} accepted {} acceptance {:.3} | tokens/round {:.2} verify {:.1} ms/round global {}",
                    st.rounds, st.wp_rounds, st.plain_steps, st.backoff_steps, st.drafted, st.accepted, st.acceptance(), tokens_per_round, verify_ms, st.global_rounds
                ));
                rows.push(serde_json::json!({"prompt": name, "run": run, "verify": format!("{path:?}"), "identical": diff.is_none(), "first_diff": diff,
                    "off_matches_golden": !want.is_empty() && golden_diff(&t_off, &want).is_none(), "golden_len": want.len(), "off_tok_s": tps_off, "on_tok_s": tps_on, "off_ms": ms_off, "on_ms": ms_on,
                    "rounds": st.rounds, "wp_rounds": st.wp_rounds, "plain": st.plain_steps, "backoff": st.backoff_steps, "drafted": st.drafted, "accepted": st.accepted,
                    "acceptance": st.acceptance(), "tokens_per_round": tokens_per_round, "verify_ms_per_round": verify_ms, "global_rounds": st.global_rounds,
                    "drafter": format!("{:?}", on.source), "prompt_tokens": ids, "tokens_off": t_off, "tokens_on": t_on}));
                if run == 1 {
                    // Negative control: compared with the SAME path's honest spec-on output.
                    let mutant = SpecConfig { mutant_accept_extra: true, ..on.clone() };
                    let (t_mut, _, _, _) = go(&mutant, &mut engine);
                    let caught = t_mut != t_on;
                    mutant_caught[pi] |= caught;
                    say(format!(
                        "{name} negative control verify={path:?} (accept one extra wrong draft): {}",
                        if caught { "output differs (CAUGHT)" } else { "output unchanged (no rejected draft to corrupt)" }
                    ));
                    if path == VerifyPath::Exact {
                        // W-L4d numerics control: the exact verify with attention outputs truncated to bf16.
                        // Compared with spec-off; it must differ on at least one prompt.
                        let nm = SpecConfig { mutant_numerics: numerics, ..on.clone() };
                        let (t_nm, _, _, _) = go(&nm, &mut engine);
                        let c = t_nm != t_off;
                        numerics_caught |= c;
                        numerics_tally.1 += 1;
                        numerics_tally.0 += c as usize;
                        say(format!(
                            "{name} numerics control verify=Exact ({numerics:?} in verify): {}",
                            if c { format!("output differs from OFF at token {} (CAUGHT)", first_diff(&t_off, &t_nm).unwrap_or(0)) } else { "output unchanged".to_string() }
                        ));
                    }
                }
            }
            if run == 1 {
                past.push((name.clone(), ids.iter().copied().chain(t_off.iter().copied()).collect()));
            }
        }
    }
    for (pi, &path) in paths.iter().enumerate() {
        if !mutant_caught[pi] {
            say(format!("FAIL: verify={path:?}: the extra-accept mutant never changed the output; the check is not proven on this data"));
            all_ok = false;
        }
    }
    if paths.contains(&VerifyPath::Exact) {
        say(format!("NUMERICS CONTROL ({numerics:?}): caught on {} of {} prompts", numerics_tally.0, numerics_tally.1));
    }
    if paths.contains(&VerifyPath::Exact) && !numerics_caught {
        say("FAIL: verify=Exact: the numerics mutant never changed the output; the identity check cannot see a rounding change on this data".into());
        all_ok = false;
    }
    if paths.contains(&VerifyPath::WholePrompt) {
        say(format!("WP-VERIFY IDENTITY vs spec-off: {} of {} runs identical", wp_identical.0, wp_identical.1));
    }
    say(format!("RESULT: {}", if all_ok { "PASS" } else { "FAIL" }));
    std::fs::write(receipt.join("run.log"), log.join("\n")).expect("write log");
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&serde_json::json!({"runs": rows, "pass": all_ok})).unwrap()).expect("write json");
    std::process::exit(if all_ok { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::golden_diff;

    #[test]
    fn golden_compare_stops_at_the_golden_length() {
        let golden = [1u32, 2, 3, 4];
        // A 64-token generation that matches the 4 golden tokens is identical, whatever follows.
        let long: Vec<u32> = [1u32, 2, 3, 4].iter().copied().chain(100..160).collect();
        assert_eq!(golden_diff(&long, &golden), None);
        assert_eq!(golden_diff(&golden, &golden), None);
        // A real difference inside the golden length is still reported at its index.
        assert_eq!(golden_diff(&[1, 2, 9, 4, 5], &golden), Some(2));
        // Stopping early (fewer tokens than the golden list) differs where the output ends.
        assert_eq!(golden_diff(&[1, 2], &golden), Some(2));
        // No golden list: nothing to compare.
        assert_eq!(golden_diff(&long, &[]), None);
    }
}
