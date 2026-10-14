//! OWN-30B check: Qwen3-30B-A3B on Omega against the CPU oracle (design section 4).
//!
//!   own30b_check golden <checkpoint> <golden.json> [--new N]
//!       CPU only: loads the f32/FP8 reference model (`load_qwen3_coder`, about 35 GB host),
//!       runs greedy continuations of the 4 fixed prompts, writes tokens and top-2 margins.
//!   own30b_check chip <checkpoint> <golden.json> <receipt-dir> [--new N] [--layers 0,23,47]
//!                     [--max-context 8192] [--prompts 1,2,3,4] [--force-memory]
//!                     [--weights bf16|fp8-experts|fp8]
//!                     [--router auto|host|device] [--lm-head auto|dense|gemv] [--router-mutants 1,2,3,4]
//!                     [--prefill auto|gemv|gemm] [--prefill-routing host|device] [--prefill-chunk N]
//!                     [--gemm-mutants 1,2,3] [--long-prompt N] [--tokens-only]
//!                     [--whole-prompt] [--prefill-device-host]
//!       Omega: loads the resident model, per-layer parity (attention replay + exact MoE oracle)
//!       with the routing mutant and the chip-side window-skew mutant as negative controls,
//!       whole-model greedy tokens against the golden file, decode tok/s and load time.
//!       Writes <receipt-dir>/run.json and a plain text log; the shell seals SHA256SUMS.
//!
//! Memory: the chip load refuses to start with less than 70 GB available (another 30 GB model
//! may be running) unless --force-memory.

use std::path::{Path, PathBuf};
use std::time::Instant;

use aien_inference_abi::qwen3_coder::{compute_logits, forward_token, load_qwen3_coder, Qwen3CoderState};
use aien_inference_abi::qwen3_moe::load_qwen3_fp8_moe_layer;
use aien_inference_abi::qwen3_moe_omega::{
    current_rss_bytes, load_attention_layer_f32, peak_rss_bytes, replay_attention, replay_moe, AttnPath, DevicePaths, GenerateParams, LmHead,
    OmegaModelUploader, PathChoice, Qwen3MoeConfig, Qwen3MoeOmegaEngine, StepTiming, WeightFormat,
};
use aien_inference_abi::qwen3_moe_omega::dgraph;
use aien_inference_abi::tokenizer::ChatTokenizer as Tokenizer;

const PROMPT_TEXTS: [(&str, &str); 4] = [
    ("p1-one-token", "Hello"),
    ("p2-short", "What is the capital of France? Answer in one sentence."),
    (
        "p3-medium",
        "Summarise the following in three bullet points.\n\nThe DGX Spark is a compact workstation built around the GB10 Grace Blackwell superchip. It pairs an Arm CPU with a Blackwell GPU over a coherent link, sharing one pool of unified memory, so a model that is loaded once can be read by both sides without copies. The design trades peak bandwidth for capacity: a large mixture-of-experts model fits in memory whole, and each generated token only touches the experts the router selects. Engineers who write their own kernels for this chip have to think about launch overhead, about which memory is cached on the GPU side, and about how the host polls for completion. Small launches add up quickly when a layer needs many of them, so grouping work per layer matters more than raw arithmetic throughput. The practical outcome is that bandwidth per token, not compute, sets the speed of decoding, and that the fastest path keeps weights in the narrowest format the kernels can read directly.",
    ),
    (
        "p4-long",
        "You are a careful engineering assistant. Read the log below and say, in plain words, what went wrong and what to check first.\n\n[run 1] loaded 48 layers in 212 s, resident 60.4 GB\n[run 1] prompt 23 tokens, prefill 1.9 s\n[run 1] decode 16 tokens, 58 ms/token\n[run 1] parity layer 0 cosine 0.99998 p95_rel 0.0021\n[run 1] parity layer 23 cosine 0.99997 p95_rel 0.0024\n[run 1] parity layer 47 cosine 0.99995 p95_rel 0.0031\n[run 2] loaded 48 layers in 208 s, resident 60.4 GB\n[run 2] prompt 256 tokens, prefill 15.1 s\n[run 2] decode 16 tokens, 61 ms/token\n[run 2] token 9 differs from golden (margin 0.0007)\n[run 3] loaded 48 layers in 215 s\n[run 3] prompt 700 tokens, prefill 41 s\n[run 3] decode 16 tokens, 66 ms/token\n[run 3] parity layer 23 cosine 0.99996\n[run 4] mutant control: routing shifted, layer 23 cosine 0.41 (caught)\n[run 4] mutant control: window skew 1 tile, layer 23 cosine 0.12 (caught)\n[run 5] kv pool resident, attention 1.1 ms/layer at 700 context\n[run 5] experts 0.61 ms/layer, lm_head 3.2 ms\n[run 5] host ops 1.4 ms/token\n[run 6] free memory before load 101 GB, after 39 GB\n[run 6] peak host rss 64.1 GB\n[run 7] seal on: unprepared kernel refused by name\n[run 7] seal off: same tokens\n[run 8] spin window 2000 us, launch floor 6 us\n[run 8] 913 launches per token, 14 ms launch tax\n[run 9] prompt 1 token, prefill 0.07 s\n[run 9] decode 16 tokens, 57 ms/token\n[run 10] golden tokens identical on p1, p2, p3; p4 diverges at 12 (margin 0.0003)\n\nThe log repeats the same measurements across several days; treat each run as independent. Mention the numbers that matter and nothing else.",
    ),
];

fn chatml(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn prompts(tok: &Tokenizer) -> Vec<(String, Vec<u32>)> {
    PROMPT_TEXTS
        .iter()
        .map(|(name, text)| {
            let ids = if *name == "p1-one-token" {
                tok.encode_with_special(text, false).expect("encode")
            } else {
                tok.encode_with_special(&chatml(text), false).expect("encode")
            };
            (name.to_string(), ids)
        })
        .collect()
}

/// One greedy run from an empty engine at a given prefill chunk: tokens, the first step's logits
/// (captured by the sampler), and the prefill wall time and timing.
struct PromptRun {
    tokens: Vec<u32>,
    logits0: Vec<f32>,
    prefill_ms: u64,
    pt: StepTiming,
}

fn run_prompt(engine: &mut Qwen3MoeOmegaEngine, ids: &[u32], chunk: usize, new_tokens: usize, eos: &[u32]) -> Result<PromptRun, String> {
    engine.reset();
    engine.prefill_chunk = chunk;
    let mut tokens = Vec::new();
    let mut logits0: Vec<f32> = Vec::new();
    let o = engine.generate_with(
        ids,
        &GenerateParams::greedy(new_tokens, eos.to_vec()),
        &mut |logits, step| {
            if step == 0 {
                logits0 = logits.to_vec();
            }
            aien_inference_abi::tensor::sample_argmax(logits).0
        },
        &mut |t| {
            tokens.push(t);
            true
        },
    )?;
    Ok(PromptRun { tokens, logits0, prefill_ms: o.prefill_ms, pt: o.prefill_timing })
}

/// Cosine and largest absolute difference between two logits vectors.
fn logits_agreement(a: &[f32], b: &[f32]) -> (f64, f32) {
    let s = aien_inference_abi::qwen3_moe::parity_stats(a, b);
    let maxd = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    (s.cosine, maxd)
}

fn prefill_line(head: &str, n_tokens: usize, r: &PromptRun) -> String {
    let n = n_tokens as f64;
    let pt = &r.pt;
    format!(
        "{head}, {n_tokens} tokens): {:.2}s = {:.1} ms/token; per token: attention {:.1} ms experts {:.1} ms (routing {:.2} ms, sort+swiglu+combine {:.2} ms) host {:.1} ms chip-reported {:.1} ms launches {:.1}; first tokens {:?}",
        r.prefill_ms as f64 / 1000.0,
        r.prefill_ms as f64 / n,
        pt.attention_ns as f64 / 1e6 / n,
        pt.experts_ns as f64 / 1e6 / n,
        pt.route_ns as f64 / 1e6 / n,
        pt.moe_host_ns as f64 / 1e6 / n,
        pt.host_ns as f64 / 1e6 / n,
        pt.chip_ns as f64 / 1e6 / n,
        pt.launches as f64 / n,
        r.tokens
    )
}

/// A chat prompt of at least `min_tokens` tokens: numbered log blocks (each block differs) and a
/// final question. The tokenizer refuses more than 2048 tokens per call, so the prompt is encoded
/// piece by piece (each piece ends on a newline) and the ids are joined.
fn long_prompt_ids(tok: &Tokenizer, min_tokens: usize) -> Vec<u32> {
    let enc = |t: &str| tok.encode_with_special(t, false).expect("encode");
    let mut ids = enc("<|im_start|>user\nYou are a careful engineering assistant. Read the log below and say, in plain words, which block number has the largest decode time.\n\n");
    let tail = enc("\nAnswer with the block number only.<|im_end|>\n<|im_start|>assistant\n");
    let mut b = 0usize;
    while ids.len() + tail.len() < min_tokens {
        ids.extend(enc(&format!(
            "[block {b}] loaded 48 layers in {} s, resident {}.{} GB\n[block {b}] prompt {} tokens, prefill {}.{} s\n[block {b}] decode 16 tokens, {} ms/token\n[block {b}] parity layer {} cosine 0.9999{}\n",
            200 + (b * 7) % 23,
            58 + b % 5,
            (b * 3) % 10,
            20 + (b * 37) % 700,
            1 + (b * 5) % 40,
            (b * 7) % 10,
            50 + (b * 11) % 37,
            (b * 5) % 48,
            b % 10
        )));
        b += 1;
    }
    ids.extend(tail);
    ids
}

fn top2_margin(logits: &[f32]) -> (u32, f32) {
    let (mut b0, mut v0, mut v1) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > v0 {
            v1 = v0;
            v0 = v;
            b0 = i;
        } else if v > v1 {
            v1 = v;
        }
    }
    (b0 as u32, v0 - v1)
}

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

struct Args {
    mode: String,
    checkpoint: PathBuf,
    golden: PathBuf,
    receipt: Option<PathBuf>,
    new_tokens: usize,
    layers: Vec<usize>,
    max_context: usize,
    prompt_ids: Vec<usize>,
    force_memory: bool,
    format: WeightFormat,
    attn: AttnPath,
    /// Chip-side decode attention mutant for the attention control (omega_gpu_decode_attn_api.h).
    attn_mutant: i32,
    /// Prompt tokens per prefill step for the TOKENS runs (None = the engine default).
    prefill_chunk: Option<usize>,
    /// Router and lm_head implementations (auto = device ones when linked).
    paths: DevicePaths,
    /// Device router mutants run as negative controls (omega_gpu_router_api.h: 1 RMSNorm without
    /// the weight, 2 wrong mask, 3 wrong directory entry, 4 weights not normalised).
    router_mutants: Vec<i32>,
    /// Skip parity, controls and the prefill check: tokens against the golden and timing only.
    tokens_only: bool,
    /// Whole-prompt prefill (W-RC: the RC server default) and its device router/combine.
    whole_prompt: bool,
    wp_device_host: bool,
    /// Batched prefill kernel: auto (GEMM when linked), gemv (the grouped GEMV) or gemm (required).
    prefill: PathChoice,
    /// Batched prefill routing: false = host (default), true = the device router per token.
    prefill_device_routing: bool,
    /// Prefill GEMM mutants run as negative controls through the real prefill path (omega_gpu_fp8gemm_api.h:
    /// 1 scale dropped, 2 scale pointer never advances, 3 neighbour row's x).
    gemm_mutants: Vec<i32>,
    /// Also run a prompt of at least this many tokens: chunk 1 against the batched chunk, tokens and hidden state.
    long_prompt: usize,
}

fn parse_choice(v: &str, host_word: &str, flag: &str) -> PathChoice {
    match v {
        "auto" => PathChoice::Auto,
        w if w == host_word => PathChoice::Host,
        "device" | "gemv" => PathChoice::Device,
        other => panic!("{flag} {other}: auto, {host_word} or device"),
    }
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mode = it.next().unwrap_or_default();
    let mut pos: Vec<String> = Vec::new();
    let mut a = Args {
        mode: mode.clone(),
        checkpoint: PathBuf::new(),
        golden: PathBuf::new(),
        receipt: None,
        new_tokens: 16,
        layers: vec![0, 23, 47],
        max_context: 8192,
        prompt_ids: vec![1, 2, 3, 4],
        force_memory: false,
        format: WeightFormat::Bf16,
        attn: AttnPath::Auto,
        attn_mutant: 1,
        prefill_chunk: None,
        paths: DevicePaths::default(),
        router_mutants: vec![1, 2, 3, 4],
        tokens_only: false,
        whole_prompt: false,
        wp_device_host: false,
        prefill: PathChoice::Auto,
        prefill_device_routing: false,
        gemm_mutants: vec![1, 2, 3],
        long_prompt: 0,
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--new" => a.new_tokens = it.next().expect("--new N").parse().expect("number"),
            "--max-context" => a.max_context = it.next().expect("--max-context N").parse().expect("number"),
            "--layers" => a.layers = it.next().expect("--layers a,b").split(',').map(|s| s.parse().expect("layer")).collect(),
            "--prompts" => a.prompt_ids = it.next().expect("--prompts a,b").split(',').map(|s| s.parse().expect("prompt")).collect(),
            "--force-memory" => a.force_memory = true,
            "--weights" => {
                a.format = match it.next().expect("--weights bf16|fp8-experts|fp8").as_str() {
                    "bf16" => WeightFormat::Bf16,
                    "fp8-experts" => WeightFormat::Fp8Experts,
                    "fp8" => WeightFormat::Fp8All,
                    other => panic!("--weights {other}: bf16, fp8-experts or fp8"),
                }
            }
            "--attn" => {
                a.attn = match it.next().expect("--attn auto|split|fused").as_str() {
                    "auto" => AttnPath::Auto,
                    "split" => AttnPath::Split,
                    "fused" => AttnPath::Fused,
                    other => panic!("--attn {other}: auto, split or fused"),
                }
            }
            "--attn-mutant" => a.attn_mutant = it.next().expect("--attn-mutant N").parse().expect("number"),
            "--prefill-chunk" => a.prefill_chunk = Some(it.next().expect("--prefill-chunk N").parse().expect("number")),
            "--router" => a.paths.router = parse_choice(&it.next().expect("--router auto|host|device"), "host", "--router"),
            "--lm-head" => a.paths.lm_head = parse_choice(&it.next().expect("--lm-head auto|dense|gemv"), "dense", "--lm-head"),
            "--tokens-only" => a.tokens_only = true,
            "--whole-prompt" => a.whole_prompt = true,
            "--prefill-device-host" => a.wp_device_host = true,
            "--prefill" => a.prefill = parse_choice(&it.next().expect("--prefill auto|gemv|gemm"), "gemv", "--prefill"),
            "--prefill-routing" => {
                a.prefill_device_routing = match it.next().expect("--prefill-routing host|device").as_str() {
                    "host" => false,
                    "device" => true,
                    other => panic!("--prefill-routing {other}: host or device"),
                }
            }
            "--gemm-mutants" => {
                let v = it.next().expect("--gemm-mutants a,b (empty for none)");
                a.gemm_mutants = v.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().expect("mutant")).collect();
            }
            "--long-prompt" => a.long_prompt = it.next().expect("--long-prompt N").parse().expect("number"),
            "--router-mutants" => {
                let v = it.next().expect("--router-mutants a,b (empty for none)");
                a.router_mutants = v.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().expect("mutant")).collect();
            }
            other => pos.push(other.to_string()),
        }
    }
    if mode != "golden" && mode != "chip" || pos.len() < 2 {
        eprintln!("usage: own30b_check golden <checkpoint> <golden.json> | chip <checkpoint> <golden.json> <receipt-dir>");
        std::process::exit(2);
    }
    a.checkpoint = PathBuf::from(&pos[0]);
    a.golden = PathBuf::from(&pos[1]);
    a.receipt = pos.get(2).map(PathBuf::from);
    a
}

fn golden(a: &Args) {
    let tok = Tokenizer::from_file(a.checkpoint.join("tokenizer.json")).expect("tokenizer");
    let config = Qwen3MoeConfig::from_checkpoint(&a.checkpoint).expect("config");
    let t0 = Instant::now();
    let weights = load_qwen3_coder(&a.checkpoint).expect("load CPU reference");
    eprintln!("cpu reference loaded in {:.0}s, rss {:.1} GB", t0.elapsed().as_secs_f64(), current_rss_bytes().unwrap_or(0) as f64 / 1e9);
    let mut out = Vec::new();
    for (idx, (name, ids)) in prompts(&tok).into_iter().enumerate() {
        if !a.prompt_ids.contains(&(idx + 1)) {
            continue;
        }
        let t1 = Instant::now();
        let mut state = Qwen3CoderState::new();
        let mut hidden = Vec::new();
        for (pos, &t) in ids.iter().enumerate() {
            hidden = forward_token(&weights, t, pos, &mut state);
        }
        let mut tokens = Vec::new();
        let mut margins = Vec::new();
        for pos in (ids.len()..).take(a.new_tokens) {
            let logits = compute_logits(&weights, &hidden);
            let (next, margin) = top2_margin(&logits);
            tokens.push(next);
            margins.push(margin);
            if config.eos_token_ids.contains(&next) {
                break;
            }
            hidden = forward_token(&weights, next, pos, &mut state);
        }
        let secs = t1.elapsed().as_secs_f64();
        let text = tok.decode(&tokens).unwrap_or_default();
        eprintln!("{name}: {} prompt tokens, {} new in {secs:.0}s: {text:?}", ids.len(), tokens.len());
        out.push(serde_json::json!({
            "name": name, "prompt_tokens": ids, "golden": tokens, "margins": margins,
            "text": text, "cpu_seconds": secs,
        }));
    }
    let doc = serde_json::json!({
        "checkpoint": a.checkpoint.display().to_string(), "max_new": a.new_tokens,
        "oracle": "qwen3_coder::forward_token (f32 attention, exact FP8 MoE) + compute_logits, greedy",
        "prompts": out,
    });
    std::fs::write(&a.golden, serde_json::to_string_pretty(&doc).unwrap()).expect("write golden");
    eprintln!("wrote {}", a.golden.display());
}

fn chip(a: &Args) {
    let receipt = a.receipt.clone().expect("receipt dir");
    std::fs::create_dir_all(&receipt).expect("receipt dir");
    let mut log: Vec<String> = Vec::new();
    let mut say = |s: String| {
        println!("{s}");
        log.push(s);
    };
    let avail = mem_available_gb();
    say(format!("mem available before load: {avail:.1} GB"));
    if avail < 70.0 && !a.force_memory {
        say("REFUSED: less than 70 GB available (another model may be loaded); --force-memory to override".into());
        std::process::exit(3);
    }
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(&a.golden).expect("golden file")).expect("golden json");
    let tok = Tokenizer::from_file(a.checkpoint.join("tokenizer.json")).expect("tokenizer");
    // Spin window: honour AIEN_OMEGA_SPIN_US (one knob, one variable per run); unset keeps
    // whatever the session default is, and the value actually in force is printed below.
    if let Some(us) = aien_omega_gpu::parse_spin_us(std::env::var(aien_omega_gpu::SPIN_US_ENV).ok().as_deref()).expect("spin env") {
        aien_omega_gpu::set_spin_us(us).expect("set spin window");
    }
    let config = Qwen3MoeConfig::from_checkpoint(&a.checkpoint).expect("config");
    say(format!(
        "engine native={} window_api={} fp8_api={} dattn_api={} weights={:?} attn={:?} spin_us={:?} cta_budget={:?}",
        aien_omega_gpu::is_native(),
        OmegaModelUploader::uses_window_api(),
        OmegaModelUploader::uses_fp8_api(),
        OmegaModelUploader::uses_dattn_api(),
        a.format,
        a.attn,
        aien_omega_gpu::spin_us(),
        aien_omega_gpu::cta_budget()
    ));

    // ---- load ----
    let t0 = Instant::now();
    let mut n = 0usize;
    let (mut engine, report) = Qwen3MoeOmegaEngine::load_omega_paths(&a.checkpoint, a.max_context, a.format, a.attn, a.paths, &mut |s| {
        n += 1;
        if n.is_multiple_of(20) || s.starts_with("kv") || s == "lm_head" {
            eprintln!("  [{:.0}s] {s}", t0.elapsed().as_secs_f64());
        }
    })
    .expect("load on Omega");
    match a.prefill {
        PathChoice::Host => engine.prefill_gemm = false,
        PathChoice::Device => assert!(engine.prefill_gemm, "--prefill gemm: the prefill GEMM is not linked or not offered for every projection"),
        PathChoice::Auto => {}
    }
    engine.prefill_device_routing = a.prefill_device_routing;
    if !engine.prefill_gemm {
        engine.prefill_chunk = aien_inference_abi::qwen3_moe_omega::PREFILL_CHUNK;
    }
    if let Some(chunk) = a.prefill_chunk {
        engine.prefill_chunk = chunk;
    }
    say(format!(
        "prefill path {} routing {} (gemm_api={})",
        if engine.prefill_gemm { "gemm" } else { "gemv" },
        if engine.prefill_device_routing { "device" } else { "host" },
        aien_omega_fp8::has_gemm()
    ));
    say(format!("attention path {} prefill_chunk {} batched_prefill_possible {}", engine.attention_path(), engine.prefill_chunk, engine.batched_prefill_possible()));
    let router_path = if engine.device_routing { "device" } else { "host" };
    let lm_head_path = if matches!(engine.weights().lm_head, LmHead::Gemv(_)) { "gemv" } else { "dense" };
    say(format!("router path {router_path} lm_head path {lm_head_path} (router_api={} bf16_api={})", OmegaModelUploader::uses_router_api(), OmegaModelUploader::uses_bf16_api()));
    // One command list per decode token (OWN30B_DECODE_GRAPH=1): built now so a load that cannot
    // run it is refused before any generation, never run on the old path in its place.
    if dgraph::stamps_requested() && !engine.decode_graph_on() {
        say(format!("REFUSED: {}=1 measures the one-list decode; it needs {}=1 as well", dgraph::STAMPS_ENV, dgraph::ENV));
        std::process::exit(4);
    }
    if engine.decode_graph_on() {
        if let Err(e) = engine.prepare_decode_graph() {
            say(format!("REFUSED: {e}"));
            std::process::exit(4);
        }
    }
    say(format!(
        "decode graph {} (dgraph_api={}, {}=1 to turn on) mutant {}",
        if engine.decode_graph_on() { "ON" } else { "OFF" },
        aien_omega_dgraph::is_native(),
        aien_inference_abi::qwen3_moe_omega::dgraph::ENV,
        engine.decode_graph_mutant()
    ));
    // Whole-prompt prefill (W-RC): the release-candidate server turns both on. The per-layer
    // parity replays set trace layers, which the whole-prompt path declines (chunked there).
    if a.whole_prompt {
        if let Err(e) = engine.set_whole_prompt(true) {
            say(format!("REFUSED: whole-prompt prefill: {e}"));
            std::process::exit(4);
        }
        if a.wp_device_host {
            if let Err(e) = engine.set_whole_prompt_device_host(true) {
                say(format!("REFUSED: whole-prompt device router/combine: {e}"));
                std::process::exit(4);
            }
        }
    }
    say(format!(
        "whole-prompt prefill {} device router/combine {}",
        if engine.whole_prompt_on() { "ON" } else { "OFF" },
        if engine.whole_prompt_device_host_on() { "ON" } else { "OFF" }
    ));
    let peak = report.peak_rss_bytes.unwrap_or(0) as f64 / 1e9;
    say(format!(
        "LOAD: {:.1}s resident {:.2} GB host {:.2} GB peak_rss {:.2} GB (before {:.2} GB) kv_resident={} mem_available_after {:.1} GB",
        report.seconds,
        report.resident_bytes as f64 / 1e9,
        report.host_bytes as f64 / 1e9,
        peak,
        report.peak_rss_before_bytes.unwrap_or(0) as f64 / 1e9,
        engine.kv_resident(),
        mem_available_gb()
    ));
    // Transient budget: whatever the resident mapping counts, the extra over resident + host must
    // stay within one layer's transients plus slack (design: peak near the 60 GB resident size).
    let transient = peak - (report.resident_bytes + report.host_bytes) as f64 / 1e9;
    say(format!("LOAD peak over resident+host: {transient:.2} GB (budget 6 GB) -> {}", if transient <= 6.0 { "PASS" } else { "FAIL" }));

    let ps = prompts(&tok);
    let mut results = Vec::new();

    let mut parity_json = Vec::new();
    let mut parity_pass = true;
    let mut controls = Vec::new();
    let mut tokens_pass_prefill = true;
    // --tokens-only skips parity, controls and the prefill check (speed and token runs only).
    if !a.tokens_only {
    // ---- layer parity on the short prompt (prompt 2), chip traces replayed on the host ----
    let (pname, pids) = &ps[1];
    engine.trace_layers = a.layers.clone();
    engine.reset();
    for &t in pids {
        engine.step(t).expect("step");
    }
    let traces = std::mem::take(&mut engine.traces);
    for &l in &a.layers {
        let lt: Vec<_> = traces.iter().filter(|t| t.layer == l).cloned().collect();
        let attn = load_attention_layer_f32(&a.checkpoint, &config, l).expect("attention weights");
        let host = replay_attention(&config, &attn, &lt);
        let capsule = load_qwen3_fp8_moe_layer(&a.checkpoint, l).expect("capsule");
        let mut worst_attn = 1.0f64;
        let mut worst_moe = 1.0f64;
        let mut worst_p95 = 0.0f32;
        let mut routing_ok = true;
        let mut worst_wt = 0.0f32;
        let mut worst_xprime = 0.0f32;
        for (t, h) in lt.iter().zip(&host) {
            // The device router's own x' against the host RMSNorm of the same residual row.
            if !t.post_device.is_empty() {
                worst_xprime = t.post_device.iter().zip(&t.post).map(|(d, p)| (d - p).abs() / (1.0 + p.abs())).fold(worst_xprime, f32::max);
            }
            let s = aien_inference_abi::qwen3_moe::parity_stats(&t.attn_proj, h);
            worst_attn = worst_attn.min(s.cosine);
            let m = replay_moe(&capsule, t, config.top_k);
            worst_moe = worst_moe.min(m.stats.cosine);
            worst_p95 = worst_p95.max(m.stats.p95_rel);
            routing_ok &= m.routing_agrees;
            worst_wt = worst_wt.max(m.max_weight_err);
        }
        let pass = worst_attn >= 0.999 && worst_moe >= 0.999 && routing_ok && worst_wt <= 1e-4 && worst_xprime <= 1e-4;
        parity_pass &= pass;
        say(format!(
            "PARITY layer {l} over {} positions ({pname}): attention cosine(min) {worst_attn:.6} moe cosine(min) {worst_moe:.6} moe p95_rel(max) {worst_p95:.4} routing_agrees={routing_ok} routing weights err(max) {worst_wt:.2e} router x' rel(max) {worst_xprime:.2e} -> {}",
            lt.len(),
            if pass { "PASS" } else { "FAIL" }
        ));
        parity_json.push(serde_json::json!({"layer": l, "positions": lt.len(), "attn_cosine_min": worst_attn, "moe_cosine_min": worst_moe, "moe_p95_rel_max": worst_p95, "routing_agrees": routing_ok, "routing_weight_err_max": worst_wt, "router_xprime_rel_max": worst_xprime, "pass": pass}));
    }
    // Negative controls on the first traced layer: host routing mutant, then the chip-side mutant
    // of the expert kernel in use (bf16: window skew of one tile; FP8: scales dropped, kernel
    // mutant 1 of omega_gpu_fp8_api.h).
    let l0 = a.layers[0];
    let capsule = load_qwen3_fp8_moe_layer(&a.checkpoint, l0).expect("capsule");
    let fp8 = a.format.fp8_experts();
    let chip_control = if fp8 { "fp8-scales-dropped" } else { "window-skew-1-tile" };
    for (name, set) in [("routing-mutant", 0), (chip_control, 1)] {
        engine.reset();
        engine.trace_layers = vec![l0];
        engine.routing_mutant = set == 0;
        if set == 1 && fp8 {
            aien_omega_fp8::test_set_mutant(1);
        } else if set == 1 {
            aien_omega_moe::test_set_window_skew(1);
        }
        let stepped: Result<Vec<_>, _> = pids[..pids.len().min(4)].iter().map(|&t| engine.step(t)).collect();
        aien_omega_fp8::test_set_mutant(0);
        aien_omega_moe::test_set_window_skew(0);
        engine.routing_mutant = false;
        if let Err(e) = stepped {
            // A mutant kernel the chip refuses (poison survived) is also caught, by name.
            say(format!("CONTROL {name}: layer {l0} step refused: {e} -> CAUGHT (refused)"));
            controls.push(serde_json::json!({"control": name, "refused": e, "caught": true}));
            engine.traces.clear();
            continue;
        }
        let worst = engine.traces.iter().map(|t| replay_moe(&capsule, t, config.top_k).stats.cosine).fold(1.0f64, f64::min);
        let not_applicable = set == 1 && !fp8 && !OmegaModelUploader::uses_window_api();
        let caught = worst < 0.99 || not_applicable;
        let note = if not_applicable { " (window API not linked: control not applicable)" } else { "" };
        say(format!("CONTROL {name}: layer {l0} moe cosine(min) {worst:.4} -> {}{note}", if caught { "CAUGHT" } else { "NOT CAUGHT (FAIL)" }));
        controls.push(serde_json::json!({"control": name, "moe_cosine_min": worst, "caught": caught}));
        parity_pass &= caught;
        engine.traces.clear();
    }
    // Third control on the fused path: the chip-side decode attention mutant (default 1 = no
    // RoPE on q and k; omega_gpu_decode_attn_api.h) over the parity prompt; the host attention
    // replay of the first traced layer must then disagree.
    if engine.attention_path() == "fused" {
        let name = format!("attention-mutant-{}", a.attn_mutant);
        engine.reset();
        engine.trace_layers = vec![l0];
        aien_omega_dattn::test_set_mutant(a.attn_mutant);
        let stepped: Result<Vec<_>, _> = pids.iter().map(|&t| engine.step(t)).collect();
        aien_omega_dattn::test_set_mutant(0);
        match stepped {
            Err(e) => {
                say(format!("CONTROL {name}: layer {l0} step refused: {e} -> CAUGHT (refused)"));
                controls.push(serde_json::json!({"control": name, "refused": e, "caught": true}));
            }
            Ok(_) => {
                let attn = load_attention_layer_f32(&a.checkpoint, &config, l0).expect("attention weights");
                let host = replay_attention(&config, &attn, &engine.traces);
                let worst = engine
                    .traces
                    .iter()
                    .zip(&host)
                    .map(|(t, h)| aien_inference_abi::qwen3_moe::parity_stats(&t.attn_proj, h).cosine)
                    .fold(1.0f64, f64::min);
                let caught = worst < 0.99;
                say(format!(
                    "CONTROL {name}: layer {l0} attention cosine(min) {worst:.4} over {} positions -> {}",
                    engine.traces.len(),
                    if caught { "CAUGHT" } else { "NOT CAUGHT (FAIL)" }
                ));
                controls.push(serde_json::json!({"control": name, "attn_cosine_min": worst, "caught": caught}));
                parity_pass &= caught;
            }
        }
        engine.traces.clear();
    }
    // Router controls on the device path: each deliberately wrong router kernel (omega_gpu_router_api.h)
    // over the parity prompt must make the first traced layer disagree with the host oracle: the
    // oracle input is the host's own RMSNorm of the same residual row, the chip's choice and MoE
    // output come from the wrong router.
    if engine.device_routing {
        for &m in &a.router_mutants {
            let name = format!("router-mutant-{m}");
            engine.reset();
            engine.trace_layers = vec![l0];
            aien_omega_router::test_set_mutant(m);
            let stepped: Result<Vec<_>, _> = pids[..pids.len().min(4)].iter().map(|&t| engine.step(t)).collect();
            aien_omega_router::test_set_mutant(0);
            match stepped {
                Err(e) => {
                    say(format!("CONTROL {name}: layer {l0} step refused: {e} -> CAUGHT (refused)"));
                    controls.push(serde_json::json!({"control": name, "refused": e, "caught": true}));
                }
                Ok(_) => {
                    let mut worst = 1.0f64;
                    let mut routing_ok = true;
                    let mut xprime = 0.0f32;
                    let mut wt_err = 0.0f32;
                    for t in &engine.traces {
                        let r = replay_moe(&capsule, t, config.top_k);
                        worst = worst.min(r.stats.cosine);
                        routing_ok &= r.routing_agrees;
                        wt_err = wt_err.max(r.max_weight_err);
                        xprime = t.post_device.iter().zip(&t.post).map(|(d, p)| (d - p).abs() / (1.0 + p.abs())).fold(xprime, f32::max);
                    }
                    let caught = worst < 0.99 || !routing_ok || wt_err > 1e-3 || xprime > 1e-4;
                    say(format!(
                        "CONTROL {name}: layer {l0} moe cosine(min) {worst:.4} routing_agrees={routing_ok} routing weights err(max) {wt_err:.2e} router x' rel(max) {xprime:.2e} over {} positions -> {}",
                        engine.traces.len(),
                        if caught { "CAUGHT" } else { "NOT CAUGHT (FAIL)" }
                    ));
                    controls.push(serde_json::json!({"control": name, "moe_cosine_min": worst, "routing_agrees": routing_ok, "routing_weight_err_max": wt_err, "router_xprime_rel_max": xprime, "caught": caught}));
                    parity_pass &= caught;
                }
            }
            engine.traces.clear();
        }
    }
    engine.trace_layers.clear();
    }

    // ---- batched prefill against token-at-a-time on the 204-token prompt: same logits, same tokens ----
    let mut prefill_json = serde_json::Value::Null;
    if !a.tokens_only && engine.batched_prefill_possible() && engine.prefill_chunk > 1 {
        let (pname, pids) = &ps[2];
        let chunk = engine.prefill_chunk;
        let mut outs = Vec::new();
        for c in [1usize, chunk] {
            let r = run_prompt(&mut engine, pids, c, 8, &config.eos_token_ids).expect("generate");
            say(prefill_line(&format!("PREFILL chunk {c} ({pname}"), pids.len(), &r));
            outs.push(r);
        }
        engine.prefill_chunk = chunk;
        let same = outs[0].tokens == outs[1].tokens;
        let (cos, maxd) = logits_agreement(&outs[1].logits0, &outs[0].logits0);
        let ok = same && cos > 0.99999;
        say(format!(
            "PREFILL chunk {chunk} vs 1: tokens {} ; first-step logits cosine {cos:.7} max abs diff {maxd:.2e} ; {:.1} -> {:.1} ms/token -> {}",
            if same { "IDENTICAL" } else { "DIFFER (FAIL)" },
            outs[0].prefill_ms as f64 / pids.len() as f64,
            outs[1].prefill_ms as f64 / pids.len() as f64,
            if ok { "PASS" } else { "FAIL" }
        ));
        tokens_pass_prefill = ok;
        prefill_json = serde_json::json!({"prompt": pname, "prompt_tokens": pids.len(), "chunk": chunk,
            "prefill_ms_chunk1": outs[0].prefill_ms, "prefill_ms_chunk": outs[1].prefill_ms,
            "launches_per_token_chunk1": outs[0].pt.launches as f64 / pids.len() as f64, "launches_per_token_chunk": outs[1].pt.launches as f64 / pids.len() as f64,
            "tokens_identical": same, "logits_cosine": cos, "logits_max_abs_diff": maxd});
    }

    // ---- prefill GEMM negative controls through the real batched path ----
    // Each deliberately wrong GEMM kernel (omega_gpu_fp8gemm_api.h) runs a whole chunked prefill; the
    // first-step logits must then disagree with the clean chunked prefill of the same prompt.
    let mut gemm_controls = Vec::new();
    if !a.tokens_only && engine.prefill_gemm && engine.batched_prefill_possible() && !a.gemm_mutants.is_empty() {
        let (pname, pids) = &ps[2];
        let ids = &pids[..pids.len().min(80)];
        let chunk = engine.prefill_chunk.max(2);
        let clean = run_prompt(&mut engine, ids, chunk, 1, &config.eos_token_ids).expect("clean gemm prefill");
        for &m in &a.gemm_mutants {
            let name = format!("gemm-mutant-{m}");
            aien_omega_fp8::gemm_test_set_mutant(m);
            let r = run_prompt(&mut engine, ids, chunk, 1, &config.eos_token_ids);
            aien_omega_fp8::gemm_test_set_mutant(0);
            match r {
                Err(e) => {
                    say(format!("CONTROL {name}: prefill of {} tokens ({pname}) refused: {e} -> CAUGHT (refused)", ids.len()));
                    gemm_controls.push(serde_json::json!({"control": name, "refused": e, "caught": true}));
                }
                Ok(r) => {
                    let (cos, maxd) = logits_agreement(&r.logits0, &clean.logits0);
                    let caught = !cos.is_finite() || cos < 0.999 || r.tokens != clean.tokens;
                    say(format!(
                        "CONTROL {name}: prefill of {} tokens ({pname}, chunk {chunk}) first-step logits cosine {cos:.4} max abs diff {maxd:.2e} tokens {:?} vs clean {:?} -> {}",
                        ids.len(), r.tokens, clean.tokens, if caught { "CAUGHT" } else { "NOT CAUGHT (FAIL)" }
                    ));
                    gemm_controls.push(serde_json::json!({"control": name, "logits_cosine": cos, "max_abs_diff": maxd, "caught": caught}));
                    parity_pass &= caught;
                }
            }
        }
        // The kernel cache is back to the clean kernel: the same prefill again is bit-identical.
        let again = run_prompt(&mut engine, ids, chunk, 1, &config.eos_token_ids).expect("clean gemm prefill again");
        let restored = again.logits0 == clean.logits0;
        say(format!("CONTROL gemm-mutants-off: the clean prefill after the mutants is bit-identical to before -> {}", if restored { "PASS" } else { "FAIL" }));
        parity_pass &= restored;
        controls.extend(gemm_controls.iter().cloned());
        engine.prefill_chunk = chunk;
    }

    // ---- whole-model greedy tokens vs golden, with timing ----
    let mut tokens_pass = tokens_pass_prefill;
    for (idx, (name, ids)) in ps.iter().enumerate() {
        if !a.prompt_ids.contains(&(idx + 1)) {
            continue;
        }
        let g = golden["prompts"].as_array().and_then(|arr| arr.iter().find(|p| p["name"] == *name));
        let Some(g) = g else {
            say(format!("{name}: no golden entry, skipped"));
            continue;
        };
        let want: Vec<u32> = g["golden"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let margins: Vec<f32> = g["margins"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let golden_prompt: Vec<u32> = g["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        if golden_prompt != *ids {
            say(format!("{name}: prompt tokens differ from the golden file's ({} vs {}), FAIL", ids.len(), golden_prompt.len()));
            tokens_pass = false;
            continue;
        }
        let mut got = Vec::new();
        let params = GenerateParams::greedy(a.new_tokens, config.eos_token_ids.clone());
        // Start this prompt's timeline from zero (earlier sections also ran graph tokens).
        let _ = engine.take_decode_timeline();
        let o = engine
            .generate(ids, &params, &mut |t| {
                got.push(t);
                true
            })
            .expect("generate");
        // The golden list includes a final stop id when the CPU stopped; ours consumed it.
        let want_cmp: Vec<u32> = want.iter().copied().filter(|t| !config.eos_token_ids.contains(t)).collect();
        let first_diff = got.iter().zip(&want_cmp).position(|(x, y)| x != y).or(if got.len() != want_cmp.len() { Some(got.len().min(want_cmp.len())) } else { None });
        let verdict = match first_diff {
            None => "IDENTICAL".to_string(),
            Some(i) => {
                let m = margins.get(i).copied().unwrap_or(f32::NAN);
                format!("DIVERGES at {i} (cpu top-2 margin {m:.4}; {})", if m < 1e-3 { "below the 1e-3 floor: recorded, not waved" } else { "FAIL" })
            }
        };
        if let Some(i) = first_diff {
            if margins.get(i).copied().unwrap_or(0.0) >= 1e-3 {
                tokens_pass = false;
            }
        }
        let dt = &o.decode_timing;
        let steps = o.emitted_tokens.max(1) as f64;
        let tok_s = if o.decode_ms > 0 { o.emitted_tokens as f64 * 1000.0 / o.decode_ms as f64 } else { 0.0 };
        let pt = &o.prefill_timing;
        say(format!(
            "TOKENS {name}: {} prompt tokens, prefill {:.2}s ({:.1} ms/token; chunk {}, per token attention {:.1} ms experts {:.1} ms host {:.1} ms launches {:.1}), decode {} tokens in {} ms = {:.1} ms/token = {tok_s:.1} tok/s; per step: attention {:.1} ms experts {:.1} ms lm_head {:.1} ms host {:.1} ms chip-reported {:.1} ms launches {}; {verdict}",
            ids.len(),
            o.prefill_ms as f64 / 1000.0,
            o.prefill_ms as f64 / ids.len() as f64,
            engine.prefill_chunk,
            pt.attention_ns as f64 / 1e6 / ids.len() as f64,
            pt.experts_ns as f64 / 1e6 / ids.len() as f64,
            pt.host_ns as f64 / 1e6 / ids.len() as f64,
            pt.launches as f64 / ids.len() as f64,
            o.emitted_tokens,
            o.decode_ms,
            o.decode_ms as f64 / steps,
            dt.attention_ns as f64 / 1e6 / steps,
            dt.experts_ns as f64 / 1e6 / steps,
            dt.lm_head_ns as f64 / 1e6 / steps,
            dt.host_ns as f64 / 1e6 / steps,
            dt.chip_ns as f64 / 1e6 / steps,
            dt.launches as f64 / steps,
        ));
        // Every old-path launch is its own synchronous submission; a one-list token is one
        // submission for all its launches.
        let submits = dt.launches - dt.graph_launches + dt.graph_submits;
        let per_graph = |v: u64| if dt.graph_tokens > 0 { v as f64 / dt.graph_tokens as f64 } else { 0.0 };
        say(format!(
            "  DECODE GRAPH {}: {} of {} decode steps as one list (the first step is the prefill's lm_head); per step launches {:.1} submits {:.1}; per graph token launches {:.1} submits {:.1} record {:.1} us",
            if engine.decode_graph_on() { "ON" } else { "OFF" },
            dt.graph_tokens,
            o.emitted_tokens,
            dt.launches as f64 / steps,
            submits as f64 / steps,
            per_graph(dt.graph_launches),
            per_graph(dt.graph_submits),
            per_graph(dt.graph_record_ns) / 1e3,
        ));
        let mut timeline_json = serde_json::Value::Null;
        if let Some(t) = engine.take_decode_timeline() {
            let lines = timeline_table(&t, &config);
            for l in &lines {
                say(format!("  {l}"));
            }
            timeline_json = serde_json::json!({"steps": t.steps, "launches": t.launches, "ns": t.ns, "span_ns": t.span_ns, "chip_ns": t.chip_ns, "table": lines});
        }
        say(format!("  chip text: {:?}", tok.decode(&got).unwrap_or_default()));
        say(format!("  cpu  text: {:?}", g["text"].as_str().unwrap_or("")));
        results.push(serde_json::json!({
            "name": name, "prompt_tokens": ids.len(), "chip": got, "golden": want, "first_diff": first_diff,
            "prefill_ms": o.prefill_ms, "decode_ms": o.decode_ms, "emitted": o.emitted_tokens, "tok_s": tok_s,
            "per_step_ms": {"attention": dt.attention_ns as f64 / 1e6 / steps, "experts": dt.experts_ns as f64 / 1e6 / steps,
                "lm_head": dt.lm_head_ns as f64 / 1e6 / steps, "host": dt.host_ns as f64 / 1e6 / steps, "chip_reported": dt.chip_ns as f64 / 1e6 / steps,
                "launches": dt.launches as f64 / steps},
            "decode_graph": {"on": engine.decode_graph_on(), "graph_tokens": dt.graph_tokens, "graph_launches": dt.graph_launches,
                "graph_submits": dt.graph_submits, "graph_record_ns": dt.graph_record_ns, "submits": submits, "mutant": engine.decode_graph_mutant(), "timeline": timeline_json},
        }));
    }
    // ---- a long prompt (new territory past 673 tokens): chunk 1 against the batched chunk ----
    let mut long_json = serde_json::Value::Null;
    if a.long_prompt > 0 && engine.batched_prefill_possible() {
        let ids = long_prompt_ids(&tok, a.long_prompt);
        let chunk = engine.prefill_chunk.max(2);
        let mut outs = Vec::new();
        for c in [chunk, 1usize] {
            let r = run_prompt(&mut engine, &ids, c, a.new_tokens, &config.eos_token_ids).expect("long prompt generate");
            say(prefill_line(&format!("LONG chunk {c}"), ids.len(), &r));
            outs.push(r);
        }
        engine.prefill_chunk = chunk;
        let same = outs[0].tokens == outs[1].tokens;
        let (cos, maxd) = logits_agreement(&outs[0].logits0, &outs[1].logits0);
        let (_, margin) = top2_margin(&outs[1].logits0);
        // The batched path routes on the host and chunk 1 on the device, so their logits drift a little
        // with context (tokens must still agree); the GEMM itself must equal the grouped GEMV bit for bit.
        let mut gemm_eq_gemv = true;
        if engine.prefill_gemm {
            engine.prefill_gemm = false;
            let g = run_prompt(&mut engine, &ids, aien_inference_abi::qwen3_moe_omega::PREFILL_CHUNK, a.new_tokens, &config.eos_token_ids).expect("long prompt gemv");
            engine.prefill_gemm = true;
            say(prefill_line("LONG gemv chunk 8", ids.len(), &g));
            gemm_eq_gemv = g.logits0 == outs[0].logits0 && g.tokens == outs[0].tokens;
            say(format!("LONG GEMM chunk {chunk} vs GEMV chunk 8: first-step logits and tokens bit-identical -> {}", if gemm_eq_gemv { "PASS" } else { "FAIL" }));
        }
        let ok = same && cos > 0.999 && gemm_eq_gemv;
        say(format!(
            "LONG chunk {chunk} vs 1 ({} tokens): tokens {} ; first-step logits cosine {cos:.7} max abs diff {maxd:.2e} (chunk-1 top-2 margin {margin:.4}) ; {:.1} -> {:.1} ms/token -> {}",
            ids.len(),
            if same { "IDENTICAL" } else { "DIFFER (FAIL)" },
            outs[1].prefill_ms as f64 / ids.len() as f64,
            outs[0].prefill_ms as f64 / ids.len() as f64,
            if ok { "PASS" } else { "FAIL" }
        ));
        say(format!("  long chip text: {:?}", tok.decode(&outs[0].tokens).unwrap_or_default()));
        tokens_pass &= ok;
        long_json = serde_json::json!({"prompt_tokens": ids.len(), "chunk": chunk, "tokens_identical": same, "logits_cosine": cos, "logits_max_abs_diff": maxd,
            "prefill_ms_chunk": outs[0].prefill_ms, "prefill_ms_chunk1": outs[1].prefill_ms, "top2_margin_chunk1": margin});
    }
    say(format!("SUMMARY parity={} tokens={} peak_rss={:.2} GB", if parity_pass { "PASS" } else { "FAIL" }, if tokens_pass { "PASS" } else { "FAIL" }, peak_rss_bytes().unwrap_or(0) as f64 / 1e9));
    let doc = serde_json::json!({
        "checkpoint": a.checkpoint.display().to_string(),
        "golden": a.golden.display().to_string(),
        "engine": {"native": aien_omega_gpu::is_native(), "window_api": OmegaModelUploader::uses_window_api(),
            "fp8_api": OmegaModelUploader::uses_fp8_api(), "router_api": OmegaModelUploader::uses_router_api(), "bf16_api": OmegaModelUploader::uses_bf16_api(), "dattn_api": OmegaModelUploader::uses_dattn_api(),
            "weights": format!("{:?}", a.format), "attention": engine.attention_path(), "router": router_path, "lm_head": lm_head_path, "prefill_chunk": engine.prefill_chunk, "prefill_path": if engine.prefill_gemm { "gemm" } else { "gemv" }, "prefill_routing": if engine.prefill_device_routing { "device" } else { "host" },
            "spin_us": aien_omega_gpu::spin_us(), "cta_budget": aien_omega_gpu::cta_budget(), "kv_resident": engine.kv_resident(),
            "max_context": engine.max_context_tokens()},
        "prefill": prefill_json, "long_prompt": long_json,
        "load": {"seconds": report.seconds, "resident_bytes": report.resident_bytes, "host_bytes": report.host_bytes,
            "peak_rss_bytes": report.peak_rss_bytes, "peak_rss_before_bytes": report.peak_rss_before_bytes, "transient_gb": transient},
        "parity": parity_json, "controls": controls, "prompts": results,
        "pass": parity_pass && tokens_pass,
    });
    std::fs::write(receipt.join("run.json"), serde_json::to_string_pretty(&doc).unwrap()).expect("write run.json");
    std::fs::write(receipt.join("run.log"), log.join("\n") + "\n").expect("write run.log");
    println!("receipt: {}", receipt.display());
    if !(parity_pass && tokens_pass) {
        std::process::exit(1);
    }
}

fn main() {
    let a = parse_args();
    if !Path::new(&a.checkpoint).join("config.json").is_file() {
        eprintln!("no config.json in {}", a.checkpoint.display());
        std::process::exit(2);
    }
    match a.mode.as_str() {
        "golden" => golden(&a),
        _ => chip(&a),
    }
}

/// The one-list decode's per-kind timeline as table lines: kind, launches/step, us/step, FP8 (or
/// bf16 for the lm_head) weight bytes/step and the achieved GB/s. Router, attention and the glue
/// kernels get "-" for bytes (their traffic is not a fixed weight read; KV bytes are not guessed).
fn timeline_table(t: &dgraph::Timeline, c: &Qwen3MoeConfig) -> Vec<String> {
    let steps = t.steps.max(1) as f64;
    let l = c.layers as f64;
    let bytes = |kind: &str| -> Option<f64> {
        match kind {
            "qkv" => Some(5120.0 * 2048.0 * l),
            "o_proj" => Some(2048.0 * 4096.0 * l),
            "gate_up" => Some(8.0 * 1536.0 * 2048.0 * l),
            "down" => Some(8.0 * 2048.0 * 768.0 * l),
            "lm_head" => Some(151936.0 * 2048.0 * 2.0),
            _ => None,
        }
    };
    let mut out = vec![format!("TIMELINE over {} graph tokens: {:<12} {:>9} {:>10} {:>14} {:>9}", t.steps, "kind", "launches", "us/step", "bytes/step", "GB/s")];
    let mut sum_ns = 0u64;
    for k in 0..12 {
        let name = aien_omega_dgraph::kind_name(k as u8);
        sum_ns += t.ns[k];
        let us = t.ns[k] as f64 / 1e3 / steps;
        let (b, gbs) = match bytes(&name) {
            // bytes per ns = GB/s
            Some(b) if t.ns[k] > 0 => (format!("{b:.0}"), format!("{:.1}", b * steps / t.ns[k] as f64)),
            Some(b) => (format!("{b:.0}"), "-".into()),
            None => ("-".into(), "-".into()),
        };
        out.push(format!("TIMELINE {:<12} {:>9.1} {:>10.1} {:>14} {:>9}", name, t.launches[k] as f64 / steps, us, b, gbs));
    }
    out.push(format!(
        "TIMELINE span {:.1} us/step (sum of kinds {:.1}) vs graph chip_ns {:.1} us/step",
        t.span_ns as f64 / 1e3 / steps,
        sum_ns as f64 / 1e3 / steps,
        t.chip_ns as f64 / 1e3 / steps
    ));
    out
}
