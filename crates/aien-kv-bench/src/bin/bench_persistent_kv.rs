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

//! `bench_persistent_kv`: persistent KV benchmark on the host CPU reference
//! backend (cut C9, sc#485). Writes one JSON report (schema
//! `aien.pkv.bench.v1`) and prints the generated Markdown table.
//!
//! ```text
//! bench_persistent_kv --modes resident-only,cold-prefill,restore-from-page-cache,restore-from-device \
//!     --out benchmarks/data/persistent_kv_host_<stamp>.json \
//!     [--concurrency 1,2,4,8] [--prompt-tokens 512,2048] [--rounds 3] [--store-root DIR]
//! bench_persistent_kv --render <report.json>
//! ```
//!
//! There is no default mode: the operator names every mode, and each row is
//! labelled with its mode, platform and cache_state. `restore-from-device`
//! and the native physical rows are always NOT_RUN here (no root, no native
//! hardware). Never touches a GPU.

use aien_kv_bench::report::{
    build_report, check_report, render_markdown, Machine, Report, ReportMeta, Row, Workload, HOSTED,
};
use aien_kv_bench::run::{model_label, run_row, Mode, RunSpec, DEVICE_NOT_RUN_REASON};
use std::path::PathBuf;
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: bench_persistent_kv --modes <m1,m2,...> --out <file.json> [--concurrency 1,2,4,8] [--prompt-tokens 512,2048] [--rounds N] [--store-root DIR]\n       bench_persistent_kv --render <file.json>\nmodes: resident-only, cold-prefill, restore-from-page-cache, restore-from-device";

struct Args {
    modes: Vec<Mode>,
    out: PathBuf,
    concurrency: Vec<u32>,
    prompt_tokens: Vec<u32>,
    rounds: u32,
    store_root: PathBuf,
}

fn parse_list(flag: &str, v: &str) -> Result<Vec<u32>, String> {
    v.split(',')
        .map(|s| {
            s.trim()
                .parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| format!("{flag}: {s:?} is not a positive integer"))
        })
        .collect()
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut modes: Option<Vec<Mode>> = None;
    let mut out = None;
    let mut concurrency = vec![1, 2, 4, 8];
    let mut prompt_tokens = vec![512, 2048];
    let mut rounds = 3;
    let mut store_root = PathBuf::from("target/pkv-bench-store");
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--modes" => {
                let v = value()?;
                let mut list = Vec::new();
                for label in v.split(',').map(str::trim) {
                    let mode = Mode::parse(label).ok_or_else(|| {
                        format!("refusing to report: unknown mode label {label:?}")
                    })?;
                    if !list.contains(&mode) {
                        list.push(mode);
                    }
                }
                modes = Some(list);
            }
            "--out" => out = Some(PathBuf::from(value()?)),
            "--concurrency" => concurrency = parse_list(flag, &value()?)?,
            "--prompt-tokens" => prompt_tokens = parse_list(flag, &value()?)?,
            "--rounds" => rounds = parse_list(flag, &value()?)?.first().copied().unwrap_or(0),
            "--store-root" => store_root = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let modes = modes.filter(|m| !m.is_empty()).ok_or(
        "refusing to report: --modes is required (no default mode; every row is labelled by the mode the operator named)",
    )?;
    let out = out.ok_or("--out <file.json> is required")?;
    if out.exists() {
        return Err(format!(
            "{} exists; a report is never overwritten",
            out.display()
        ));
    }
    Ok(Args {
        modes,
        out,
        concurrency,
        prompt_tokens,
        rounds,
        store_root,
    })
}

fn sh(cmd: &str, args: &[&str]) -> Result<String, String> {
    let o = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("{cmd}: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "{cmd} {args:?}: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn read_trim(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn machine() -> Machine {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            matches!(k.trim(), "model name" | "Model name" | "Hardware").then(|| v.trim())
        })
        .unwrap_or("unknown (no model name in /proc/cpuinfo)")
        .to_string();
    // Field 19 of /proc/self/stat is the nice value (after the ")" of comm).
    let nice = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let rest = s.rsplit_once(')')?.1.to_string();
            rest.split_whitespace().nth(16)?.parse::<i64>().ok()
        })
        .unwrap_or(0);
    Machine {
        hostname: read_trim("/proc/sys/kernel/hostname"),
        kernel: read_trim("/proc/sys/kernel/osrelease"),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model,
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        nice,
    }
}

/// UTC civil time from the Unix clock (Howard Hinnant's days-to-civil).
fn utc_now() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let (hh, mm, ss) = (rem / 3600, rem % 3600 / 60, rem % 60);
    (
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z"),
        format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z"),
    )
}

fn render(path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let report: Report = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    check_report(&report).map_err(|e| e.to_string())?;
    print!("{}", render_markdown(&report).map_err(|e| e.to_string())?);
    Ok(())
}

async fn run(args: Args, argv: &[String]) -> Result<(), String> {
    let git_sha = sh("git", &["rev-parse", "HEAD"])?;
    let git_dirty = !sh("git", &["status", "--porcelain"])?.is_empty();
    let (started_utc, stamp) = utc_now();
    let spec = RunSpec {
        modes: args.modes.clone(),
        concurrency: args.concurrency.clone(),
        prompt_tokens: args.prompt_tokens.clone(),
        rounds: args.rounds,
        turn1_max_tokens: 8,
        turn2_suffix_tokens: 16,
        turn2_max_tokens: 16,
        block_size: 16,
        chunk_blocks: 16,
        store_root: args.store_root.clone(),
    };
    if spec.rounds == 0 {
        return Err("--rounds must be at least 1".into());
    }
    let mut rows = Vec::new();
    for &mode in &spec.modes {
        if mode == Mode::RestoreDevice {
            rows.push(Row::not_run(
                mode.label(),
                HOSTED,
                mode.cache_state(),
                DEVICE_NOT_RUN_REASON,
            ));
            continue;
        }
        for &prompt in &spec.prompt_tokens {
            for &c in &spec.concurrency {
                eprintln!("bench: {} c={c} prompt={prompt} ...", mode.label());
                let row = run_row(&spec, mode, c, prompt)
                    .await
                    .map_err(|e| format!("{} c={c} prompt={prompt}: {e}", mode.label()))?;
                eprintln!(
                    "bench:   p50 {} us, p95 {} us, ttft p50 {} us",
                    row.p50_us.unwrap_or(0),
                    row.p95_us.unwrap_or(0),
                    row.ttft_p50_us.unwrap_or(0)
                );
                rows.push(row);
            }
        }
    }
    let meta = ReportMeta {
        measurement_id: format!(
            "BENCH-PKV-HOST-{stamp}-{}",
            &git_sha[..git_sha.len().min(7)]
        ),
        git_sha,
        git_dirty,
        command: format!("bench_persistent_kv {}", argv.join(" ")),
        started_utc,
        platform: HOSTED.to_string(),
        machine: machine(),
        backend: "NativeTransformerBackend over ReferenceCpuBackend (host CPU, FP32 arithmetic, bf16 KV pool)".into(),
        model: model_label(),
        workload: Workload {
            concurrency: spec.concurrency.clone(),
            prompt_tokens: spec.prompt_tokens.clone(),
            rounds: spec.rounds,
            turn1_max_tokens: spec.turn1_max_tokens as u32,
            turn2_suffix_tokens: spec.turn2_suffix_tokens as u32,
            turn2_max_tokens: spec.turn2_max_tokens as u32,
            block_size: spec.block_size as u32,
            chunk_blocks: spec.chunk_blocks,
            dispatch_depth: aien_kv_store::worker::WorkerConfig::default().dispatch_depth,
        },
    };
    let report = build_report(meta, rows).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(&args.out, json + "\n").map_err(|e| format!("{}: {e}", args.out.display()))?;
    eprintln!("bench: wrote {}", args.out.display());
    print!("{}", render_markdown(&report).map_err(|e| e.to_string())?);
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--render") {
        return match argv
            .get(1)
            .ok_or("--render needs a file".to_string())
            .and_then(|p| render(p))
        {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("bench_persistent_kv: {e}");
                ExitCode::from(2)
            }
        };
    }
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bench_persistent_kv: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("bench_persistent_kv: tokio: {e}");
            return ExitCode::from(1);
        }
    };
    match rt.block_on(run(args, &argv)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench_persistent_kv: {e}");
            ExitCode::from(1)
        }
    }
}
