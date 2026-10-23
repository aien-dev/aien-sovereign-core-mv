//! `aien-serve-http`: OpenAI-compatible door on 127.0.0.1 for the OWN-30B model.
//!
//!   aien-serve-http [--engine cpu|stub|omega] [--weights bf16|fp8-experts|fp8] [--spin-us N]
//!                   [--whole-prompt on|off] [--prefill-device-host on|off]
//!                   [--model-dir DIR] [--port 18091]
//!                   [--bind 127.0.0.1] [--model-id ID] [--max-context N]
//!                   [--max-tokens-cap N] [--stub-reply TEXT]
//!                   [--mem-budget-gib G] [--budget-policy shrink|refuse] [--plan-only]
//!
//! Dense Qwen3 (aien-rig): a checkpoint whose `config.json` says `model_type: qwen3` (Qwen3-8B,
//! Qwen3-8B-FP8, Qwen3-4B-Instruct-2507) loads onto the dense Omega engine
//! ([`aien_serve_http::dense_engine`]). `--weights w4` converts Qwen's FP8 checkpoint to 4-bit
//! MXFP4 at load (omega's W4 GEMV). `--mem-budget-gib 12` plans weights + KV + scratch inside
//! 12 GiB before the upload: a context that does not fit is shrunk to the largest that does
//! (`--budget-policy shrink`, the default) or refused (`refuse`); a model that cannot hold 2048
//! tokens is refused. The bytes omega allocated are checked against the budget after the load.
//! `--plan-only` prints the plan and exits without touching the chip.
//!
//! `cpu` (default) loads the checkpoint into the CPU reference forward (slow, real).
//! `omega` loads the checkpoint onto AIEN's own GPU engine (FP8 by default, spin 2000 us
//! unless `--spin-us` or `AIEN_OMEGA_SPIN_US` says otherwise) and takes the shared chip lock
//! only around each generation. Its prefill runs the whole prompt in one pass with the router
//! and MoE combine on the chip (ON by default; `--whole-prompt off`, `--prefill-device-host off`
//! or `AIEN_SERVE_WHOLE_PROMPT=0` / `AIEN_SERVE_WP_DEVICE_HOST=0` turn them off).
//! Persistent KV (omega only, opt in): with `AIEN_KV_STORE_DIR` set (and `AIEN_KV_STORE_KEY`,
//! e.g. `atlas-vault run AIEN_KV_STORE_KEY -- aien-serve-http ...`) the door parks each turn's KV
//! in the sealed store and restores the longest stored prefix after a restart; any problem
//! refuses with a startup line and the door runs plain. Unset: nothing changes.
//! SIGTERM/SIGINT: stop accepting, finish the request in flight, drain the parks, exit 0.
//! `AIEN_KV_TEST_FAULT` (chip acceptance checks only) stalls one store step; see `kv_persist`.
//! `stub` replays `--stub-reply` for every request, using the model directory's
//! tokenizer when it exists (so Odysseus can be wired before the weights load).

use std::net::IpAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use aien_inference_abi::qwen3_dense_budget::{BudgetPolicy, ProjFormat};
use aien_inference_abi::qwen3_moe_omega::WeightFormat;
use aien_serve_http::codec::{ByteCodec, Codec, HfCodec};
use aien_serve_http::cpu_engine::CpuReferenceEngine;
use aien_serve_http::dense_engine;
use aien_serve_http::engine::{Engine, StubEngine};
use aien_serve_http::kv_persist;
use aien_serve_http::omega_engine::{
    parse_switch, OmegaDoor, PrefillMode, WHOLE_PROMPT_ENV, WP_DEVICE_HOST_ENV,
};
use aien_serve_http::server::{serve_until, ServerConfig};

const DEFAULT_MODEL_DIR: &str = "models/qwen3-30b-a3b-instruct-2507-fp8";

struct Args {
    engine: String,
    model_dir: PathBuf,
    cfg: ServerConfig,
    stub_reply: String,
    weights: WeightFormat,
    spin_us: Option<u32>,
    whole_prompt: Option<bool>,
    wp_device_host: Option<bool>,
    prefill: PrefillMode,
    /// `--weights w4` (dense Qwen3 only): MXFP4 converted from Qwen's FP8 at load.
    dense_w4: bool,
    /// `--mem-budget-gib`: weights + KV + scratch must fit (dense Qwen3 only).
    budget_bytes: Option<u64>,
    budget_policy: BudgetPolicy,
    plan_only: bool,
}

fn usage() -> String {
    "usage: aien-serve-http [--engine cpu|stub|omega] [--weights bf16|fp8-experts|fp8|w4] [--spin-us N] \
     [--model-dir DIR] [--port N] [--bind ADDR] [--model-id ID] [--max-context N] \
     [--max-tokens-cap N] [--stub-reply TEXT] [--whole-prompt on|off] [--prefill-device-host on|off] \
     [--mem-budget-gib G] [--budget-policy shrink|refuse] [--plan-only]"
        .to_string()
}

fn parse_args() -> Result<Args, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let mut a = Args {
        engine: "cpu".into(),
        model_dir: PathBuf::from(home).join(DEFAULT_MODEL_DIR),
        cfg: ServerConfig {
            max_context_tokens: Some(32768),
            ..ServerConfig::default()
        },
        stub_reply: "Hello from the AIEN stub engine.".into(),
        weights: WeightFormat::Fp8All,
        spin_us: None,
        whole_prompt: None,
        wp_device_host: None,
        prefill: PrefillMode::default(),
        dense_w4: false,
        budget_bytes: None,
        budget_policy: BudgetPolicy::Shrink,
        plan_only: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| format!("{flag} needs a value\n{}", usage()))
        };
        match flag.as_str() {
            "--engine" => a.engine = val()?,
            "--model-dir" => a.model_dir = PathBuf::from(val()?),
            "--port" => a.cfg.port = val()?.parse().map_err(|e| format!("--port: {e}"))?,
            "--bind" => {
                a.cfg.bind = val()?
                    .parse::<IpAddr>()
                    .map_err(|e| format!("--bind: {e}"))?
            }
            "--model-id" => a.cfg.model_id = val()?,
            "--max-context" => {
                a.cfg.max_context_tokens =
                    Some(val()?.parse().map_err(|e| format!("--max-context: {e}"))?)
            }
            "--max-tokens-cap" => {
                a.cfg.max_new_tokens_cap = val()?
                    .parse()
                    .map_err(|e| format!("--max-tokens-cap: {e}"))?
            }
            "--stub-reply" => a.stub_reply = val()?,
            "--weights" => {
                a.weights = match val()?.as_str() {
                    "bf16" => WeightFormat::Bf16,
                    "fp8-experts" => WeightFormat::Fp8Experts,
                    "fp8" => WeightFormat::Fp8All,
                    "w4" => {
                        a.dense_w4 = true;
                        WeightFormat::Fp8All
                    }
                    other => {
                        return Err(format!(
                            "--weights must be bf16, fp8-experts, fp8 or w4, got {other}"
                        ))
                    }
                }
            }
            "--mem-budget-gib" => {
                let g: f64 = val()?
                    .parse()
                    .map_err(|e| format!("--mem-budget-gib: {e}"))?;
                if !(g.is_finite() && g > 0.0) {
                    return Err(format!(
                        "--mem-budget-gib must be a positive number, got {g}"
                    ));
                }
                a.budget_bytes = Some((g * (1u64 << 30) as f64) as u64);
            }
            "--budget-policy" => a.budget_policy = BudgetPolicy::parse(&val()?)?,
            "--plan-only" => a.plan_only = true,
            "--spin-us" => a.spin_us = Some(val()?.parse().map_err(|e| format!("--spin-us: {e}"))?),
            "--whole-prompt" => a.whole_prompt = Some(parse_switch("--whole-prompt", &val()?)?),
            "--prefill-device-host" => {
                a.wp_device_host = Some(parse_switch("--prefill-device-host", &val()?)?)
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown flag {other}\n{}", usage())),
        }
    }
    if !matches!(a.engine.as_str(), "cpu" | "stub" | "omega") {
        return Err(format!(
            "--engine must be cpu, stub or omega, got {}",
            a.engine
        ));
    }
    if a.spin_us.is_none() {
        // Flag beats environment beats the omega default (2000 us, the cut 7 number).
        let env = std::env::var("AIEN_OMEGA_SPIN_US").ok();
        a.spin_us = Some(aien_omega_gpu::parse_spin_us(env.as_deref())?.unwrap_or(2000));
    }
    // Prefill path: flag beats environment beats the default (both on; see PrefillMode).
    let env_wp = std::env::var(WHOLE_PROMPT_ENV).ok();
    let env_dev = std::env::var(WP_DEVICE_HOST_ENV).ok();
    a.prefill = PrefillMode::from_env(env_wp.as_deref(), env_dev.as_deref())?;
    if let Some(v) = a.whole_prompt {
        a.prefill.whole_prompt = v;
    }
    if let Some(v) = a.wp_device_host {
        a.prefill.device_host = v;
    }
    Ok(a)
}

/// The dense door's load options from the command line (aien-rig).
fn dense_options(a: &Args) -> dense_engine::DenseOptions {
    dense_engine::DenseOptions {
        format: if a.dense_w4 {
            ProjFormat::W4
        } else {
            ProjFormat::Checkpoint
        },
        max_context: a.cfg.max_context_tokens.unwrap_or(32768),
        budget_bytes: a.budget_bytes,
        policy: a.budget_policy,
    }
}

/// Sampling defaults from the checkpoint's `generation_config.json`, when present.
fn apply_generation_defaults(cfg: &mut ServerConfig, model_dir: &std::path::Path) {
    let path = model_dir.join("generation_config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        eprintln!(
            "aien-serve-http: {} is not valid JSON; keeping built-in sampling defaults",
            path.display()
        );
        return;
    };
    if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) {
        cfg.default_temperature = t as f32;
    }
    if let Some(p) = v.get("top_p").and_then(|x| x.as_f64()) {
        cfg.default_top_p = p as f32;
    }
    if let Some(k) = v.get("top_k").and_then(|x| x.as_u64()) {
        cfg.default_top_k = k as usize;
    }
}

fn main() -> ExitCode {
    let mut args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    apply_generation_defaults(&mut args.cfg, &args.model_dir);
    if args.plan_only {
        // aien-rig: print the memory plan for this checkpoint and exit; nothing touches the chip.
        if !dense_engine::is_dense_qwen3_dir(&args.model_dir) {
            eprintln!(
                "aien-serve-http: --plan-only needs a dense Qwen3 checkpoint (model_type qwen3)"
            );
            return ExitCode::from(2);
        }
        let opts = dense_options(&args);
        return match dense_engine::plan_for(&args.model_dir, &opts) {
            Ok((_, Some(p))) => {
                println!("{}", p.line());
                ExitCode::SUCCESS
            }
            Ok((_, None)) => {
                println!(
                    "no --mem-budget-gib given: context {} as asked",
                    opts.max_context
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                println!("refused: {e}");
                ExitCode::from(1)
            }
        };
    }
    // OWN-8B stage S negative controls: read once, refused outright without debug assertions.
    match aien_serve_http::sampling::install_mutant_from_env() {
        Ok(None) => {}
        Ok(Some(m)) => {
            for _ in 0..3 {
                eprintln!(
                    "aien-serve-http: !!! SAMPLING MUTANT {} ACTIVE: this server samples WRONG on purpose (negative control) !!!",
                    m.name()
                );
            }
        }
        Err(e) => {
            eprintln!("aien-serve-http: {e}");
            return ExitCode::from(2);
        }
    }

    let codec: Arc<dyn Codec> = if args.model_dir.join("tokenizer.json").exists() {
        match HfCodec::from_model_dir(&args.model_dir) {
            Ok(c) => {
                eprintln!(
                    "aien-serve-http: tokenizer from {} (chat layout {}, stop ids {:?})",
                    args.model_dir.display(),
                    c.template_name(),
                    c.stop_token_ids()
                );
                Arc::new(c)
            }
            Err(e) => {
                eprintln!("aien-serve-http: tokenizer load failed: {e}");
                return ExitCode::from(1);
            }
        }
    } else if args.engine == "stub" {
        eprintln!(
            "aien-serve-http: no tokenizer.json in {}; stub engine uses the byte codec",
            args.model_dir.display()
        );
        Arc::new(ByteCodec)
    } else {
        eprintln!(
            "aien-serve-http: no tokenizer.json in {}",
            args.model_dir.display()
        );
        return ExitCode::from(1);
    };

    let engine: Arc<dyn Engine> = if args.engine == "stub" {
        let reply = match codec.encode(&args.stub_reply) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("aien-serve-http: cannot encode --stub-reply: {e}");
                return ExitCode::from(1);
            }
        };
        Arc::new(
            StubEngine::new(reply).with_max_context(args.cfg.max_context_tokens.unwrap_or(4096)),
        )
    } else if args.engine == "omega" && dense_engine::is_dense_qwen3_dir(&args.model_dir) {
        if let Some(us) = args.spin_us {
            if let Err(e) = aien_omega_gpu::set_spin_us(us) {
                eprintln!("aien-serve-http: spin: {e}");
                return ExitCode::from(1);
            }
        }
        if args.cfg.model_id == ServerConfig::default().model_id {
            args.cfg.model_id = dense_engine::DEFAULT_MODEL_ID.into();
        }
        let opts = dense_options(&args);
        eprintln!(
            "aien-serve-http: loading dense Qwen3 ({:?}) onto the Omega engine from {}",
            opts.format,
            args.model_dir.display()
        );
        match dense_engine::DenseDoor::load(&args.model_dir, opts) {
            Ok(door) => Arc::new(door),
            Err(e) => {
                eprintln!("aien-serve-http: dense load refused: {e}");
                return ExitCode::from(1);
            }
        }
    } else if args.engine == "omega" {
        if args.dense_w4 || args.budget_bytes.is_some() {
            eprintln!("aien-serve-http: --weights w4 and --mem-budget-gib apply to dense Qwen3 checkpoints (model_type qwen3) only");
            return ExitCode::from(2);
        }
        eprintln!(
            "aien-serve-http: loading {:?} weights onto the Omega engine from {} (spin {:?} us)",
            args.weights,
            args.model_dir.display(),
            args.spin_us
        );
        match OmegaDoor::load(
            &args.model_dir,
            args.cfg.max_context_tokens.unwrap_or(32768),
            args.weights,
            args.spin_us,
            args.prefill,
        ) {
            Ok(mut door) => {
                // Persistent KV (sc#505): only when AIEN_KV_STORE_DIR is set; otherwise nothing else is read.
                if let Some(settings) = kv_persist::settings_from_lookup(|n| std::env::var(n).ok())
                {
                    // The key lives on only in the settings; no child inherits it.
                    std::env::remove_var(aien_kv_store::seal::KEY_ENV);
                    let line = enable_persistence(&mut door, settings, &args);
                    eprintln!("aien-serve-http: {line}");
                }
                Arc::new(door)
            }
            Err(e) => {
                eprintln!("aien-serve-http: omega load failed: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        let t0 = std::time::Instant::now();
        eprintln!(
            "aien-serve-http: loading CPU reference weights from {} (this takes minutes)",
            args.model_dir.display()
        );
        match CpuReferenceEngine::load(
            &args.model_dir,
            args.cfg.max_context_tokens.unwrap_or(32768),
        ) {
            Ok(e) => {
                eprintln!(
                    "aien-serve-http: weights loaded in {:.1} s",
                    t0.elapsed().as_secs_f64()
                );
                Arc::new(e)
            }
            Err(e) => {
                eprintln!("aien-serve-http: {e}");
                return ExitCode::from(1);
            }
        }
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("aien-serve-http: runtime: {e}");
            return ExitCode::from(1);
        }
    };
    rt.block_on(async move {
        let model_id = args.cfg.model_id.clone();
        // SIGTERM or SIGINT: stop accepting, let the request in flight finish, then drop the
        // engine, which drains the persistent KV parks before the process exits (sc#509).
        let keep = Arc::clone(&engine);
        // The handlers are installed here, before "serving" is printed, so an early SIGTERM drains too.
        let signals = shutdown_signals();
        match serve_until(args.cfg, engine, codec, signals).await {
            Ok((addr, handle)) => {
                eprintln!(
                    "aien-serve-http: serving {model_id} at http://{addr}/v1 (engine {})",
                    args.engine
                );
                let _ = handle.await;
                let t0 = std::time::Instant::now();
                eprintln!("aien-serve-http: shutting down: no new requests; finishing the one in flight and draining persistent KV");
                while Arc::strong_count(&keep) > 1 && t0.elapsed() < std::time::Duration::from_secs(600) {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                drop(keep);
                eprintln!("aien-serve-http: shutdown complete in {} ms", t0.elapsed().as_millis());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("aien-serve-http: bind failed: {e}");
                ExitCode::from(1)
            }
        }
    })
}

/// Opens the persistent KV store for the loaded omega engine (the checkpoint digest reads every
/// weight file once). Returns the startup line; any problem refuses and the door runs plain.
fn enable_persistence(
    door: &mut OmegaDoor,
    settings: Result<aien_runtime::shared_kv::KvPersistSettings, String>,
    args: &Args,
) -> String {
    let t0 = std::time::Instant::now();
    let build = kv_persist::OmegaBuild::linked();
    let env: Vec<(String, String)> = std::env::vars().collect();
    let config = door.config();
    let identity = settings.as_ref().map_err(|e| e.clone()).and_then(|s| {
        // No digest (minutes of reading) when the key already refuses: `open` reports the key.
        if !matches!(s.store_key, Some(Ok(_))) {
            return Err("the store key refused".to_string());
        }
        let rope_scaling = kv_persist::config_has_rope_scaling(&args.model_dir)?;
        let (sha, kind) = kv_persist::checkpoint_digest(&args.model_dir)?;
        kv_persist::OmegaKvIdentity::new(&kv_persist::IdentityInputs {
            model_id: &args.cfg.model_id,
            model_sha256: sha,
            model_digest_kind: kind,
            config: &config,
            rope_scaling,
            weights: args.weights,
            switches: door.kv_switches(),
            build: &build,
            env: &env,
        })
    });
    let fault = kv_persist::TestFault::from_lookup(|n| std::env::var(n).ok());
    let (state, line) = kv_persist::open_with_test_fault(settings, identity, fault);
    door.set_persistence(state);
    format!("{line} [identity in {:.1} s]", t0.elapsed().as_secs_f64())
}

/// Installs the SIGTERM and SIGINT handlers now; the returned future resolves on the first one.
fn shutdown_signals() -> impl std::future::Future<Output = ()> + Send + 'static {
    use tokio::signal::unix::{signal, SignalKind};
    let handlers = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    );
    async move {
        let (Ok(mut term), Ok(mut int)) = handlers else {
            eprintln!("aien-serve-http: cannot install the shutdown signal handlers; SIGTERM kills without draining");
            return std::future::pending().await;
        };
        tokio::select! {
            _ = term.recv() => eprintln!("aien-serve-http: SIGTERM received"),
            _ = int.recv() => eprintln!("aien-serve-http: SIGINT received"),
        }
    }
}
