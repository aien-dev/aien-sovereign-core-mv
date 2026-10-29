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

//! Standalone benchmark driver for the AIEN native reference engine on DGX Spark GB10:
//! concurrency sweeps with thermal cooldown, telemetry and a token parity oracle.

mod engine_aien;
mod metrics;
mod oracle;
mod telemetry;
mod types;

use clap::Parser;
use engine_aien::{run_aien_concurrency_sweep, AienEngineHandle};
use oracle::ORACLE_BENCHMARK_128_TOKENS;
use std::fs;
use std::path::Path;
use telemetry::enforce_thermal_cooldown;
use types::{BenchmarkConfig, BenchmarkReport, ConcurrencyRunResult};

#[derive(Parser, Debug)]
#[command(
    name = "bench_apples_to_apples",
    about = "AIEN native reference LLM inference benchmark on DGX Spark GB10"
)]
struct Args {
    /// Comma-separated list of engines to benchmark: aien
    #[arg(short, long, default_value = "aien")]
    engines: String,

    /// Comma-separated list of concurrency levels to test
    #[arg(short, long, default_value = "1,2,4,8,16,32,64")]
    concurrency: String,

    /// Target output tokens per request
    #[arg(long, default_value_t = 128)]
    output_tokens: usize,

    /// Directory where JSON benchmark artifacts are written
    #[arg(long, default_value = "benchmarks/data")]
    output_dir: String,

    /// Idle GPU power threshold for thermal cooldown envelope in Watts
    #[arg(long, default_value_t = 12.0)]
    target_power_watts: f64,

    /// Max GPU temperature for thermal cooldown envelope in Celsius
    #[arg(long, default_value_t = 42.0)]
    max_temp_c: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let concurrency_levels: Vec<usize> = args
        .concurrency
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let config = BenchmarkConfig {
        output_token_count: args.output_tokens,
        cool_down_target_power_watts: args.target_power_watts,
        cool_down_max_temp_c: args.max_temp_c,
        concurrency_levels: concurrency_levels.clone(),
        ..Default::default()
    };

    let target_engines: Vec<String> = args
        .engines
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    println!("================================================================================");
    println!("   AIEN SOVEREIGN CORE: NATIVE REFERENCE BENCHMARK SUITE (DGX SPARK)   ");
    println!("================================================================================");
    println!("Hardware: NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB Unified LPDDR5x)");
    println!("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (FP32, AIEN Native Reference)");
    println!("Workload: 128 input tokens -> 128 output tokens (Greedy, Temp=0.0)");
    println!("Engines: {:?}", target_engines);
    println!("Concurrency levels: {:?}", concurrency_levels);
    println!(
        "Thermal cooldown envelope: <= {:.1}W, <= {:.1}C",
        config.cool_down_target_power_watts, config.cool_down_max_temp_c
    );
    println!("================================================================================");

    let mut all_results: Vec<ConcurrencyRunResult> = Vec::new();

    for engine in &target_engines {
        println!("\n>>> Preparing engine: {} <<<", engine.to_uppercase());

        // Enforce thermal cooldown and memory cache flush before each engine run
        enforce_thermal_cooldown(
            config.cool_down_target_power_watts,
            config.cool_down_max_temp_c,
        )
        .await;

        match engine.as_str() {
            "aien" => {
                let mut handle = match AienEngineHandle::init(&config) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("Failed to initialize AIEN engine: {}", e);
                        continue;
                    }
                };

                for &c in &concurrency_levels {
                    match run_aien_concurrency_sweep(&mut handle, &config, c).await {
                        Ok(res) => {
                            print_sweep_result_line(&res);
                            all_results.push(res);
                        }
                        Err(e) => eprintln!("AIEN C={} sweep failed: {}", c, e),
                    }
                }
            }

            other => {
                eprintln!("Unknown engine: '{}'. Skipping.", other);
            }
        }
    }

    println!("\n================================================================================");
    println!("                           FINAL BENCHMARK COMPARISON                           ");
    println!("================================================================================");
    print_summary_table(&all_results);
    println!("================================================================================");

    // Save JSON report
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let report = BenchmarkReport {
        benchmark_timestamp: chrono::Utc::now().to_rfc3339(),
        hardware: "NVIDIA DGX Spark GB10 Grace Blackwell 128GB Unified Memory".to_string(),
        model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        precision: "FP32 (AIEN Native Reference)".to_string(),
        prompt_tokens: config.prompt_token_count,
        target_output_tokens: config.output_token_count,
        results: all_results,
        parity_oracle_sequence: ORACLE_BENCHMARK_128_TOKENS.to_vec(),
    };

    let out_dir = Path::new(&args.output_dir);
    fs::create_dir_all(out_dir)?;

    let report_filename = format!("benchmark_results_{}.json", timestamp);
    let report_path = out_dir.join(&report_filename);
    let latest_path = out_dir.join("latest.json");

    let json_bytes = serde_json::to_string_pretty(&report)?;
    fs::write(&report_path, &json_bytes)?;
    fs::write(&latest_path, &json_bytes)?;

    println!("Benchmark report saved to: {}", report_path.display());
    println!("Latest benchmark updated:  {}", latest_path.display());

    Ok(())
}

fn print_sweep_result_line(res: &ConcurrencyRunResult) {
    println!(
        "[{}] C={:<2} | Throughput: {:>7.2} tok/s | TTFT p50: {:>6.2} ms | ITL p50: {:>5.2} ms | Peak RSS: {:>5.2} GB | Power: {:>5.1} W | J/tok: {:>5.3} | Parity: {:>5.1}%",
        res.engine,
        res.concurrency,
        res.throughput_tokens_sec,
        res.ttft_p50_ms,
        res.itl_p50_ms,
        res.peak_rss_gb,
        res.avg_gpu_power_watts,
        res.energy_joules_per_token,
        res.parity_match_rate_pct,
    );
}

fn print_summary_table(results: &[ConcurrencyRunResult]) {
    println!(
        "| {:<24} | {:>3} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9} | {:>8} | {:>9} | {:>8} |",
        "Engine",
        "C",
        "Throughput",
        "TTFT p50",
        "TTFT p95",
        "ITL p50",
        "ITL p95",
        "Peak RSS",
        "Avg Power",
        "Parity"
    );
    println!(
        "|--------------------------|-----|------------|------------|------------|-----------|-----------|----------|-----------|----------|"
    );

    for r in results {
        println!(
            "| {:<24} | {:>3} | {:>8.2} t/s | {:>7.2} ms | {:>7.2} ms | {:>6.2} ms | {:>6.2} ms | {:>6.2} GB | {:>7.1} W | {:>6.1} % |",
            r.engine,
            r.concurrency,
            r.throughput_tokens_sec,
            r.ttft_p50_ms,
            r.ttft_p95_ms,
            r.itl_p50_ms,
            r.itl_p95_ms,
            r.peak_rss_gb,
            r.avg_gpu_power_watts,
            r.parity_match_rate_pct,
        );
    }
}
