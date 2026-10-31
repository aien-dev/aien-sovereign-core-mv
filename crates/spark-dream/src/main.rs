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

use clap::{Parser, Subcommand};
use spark_dream::DreamEngine;
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(
    name = "spark-dream",
    about = "Native Rust Dynamic Dream Cycle Engine for SparkOS"
)]
struct Cli {
    #[arg(long, default_value_os_t = default_dream_basecamp())]
    basecamp: PathBuf,

    #[arg(long)]
    daemon: bool,

    #[arg(long, default_value_t = 900)]
    idle_threshold: u64,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Execute one immediate dream consolidation cycle
    Cycle,
    /// Run as continuous background daemon
    Daemon,
    /// Check current dream cycle status and telemetry
    Status,
}

fn default_dream_basecamp() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join("basecamp")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let engine = DreamEngine::new(cli.basecamp.clone());

    let command = cli.command.unwrap_or({
        if cli.daemon {
            Commands::Daemon
        } else {
            Commands::Cycle
        }
    });

    match command {
        Commands::Cycle => {
            println!("🌌 Executing immediate sovereign dream consolidation cycle on DGX Spark...");
            let start = std::time::Instant::now();
            let state = engine.execute_dream_cycle().await?;
            let elapsed = start.elapsed();
            println!("✅ Dream Cycle Completed in {:.2?}:", elapsed);
            println!("   - Facts Committed: {}", state.facts_committed);
            println!("   - Proposals Staged: {}", state.proposals_staged);
            println!("   - Rejected:         {}", state.facts_rejected);
        }

        Commands::Status => {
            let state_file = cli.basecamp.join("aien-dream/state.json");
            if state_file.exists() {
                let content = std::fs::read_to_string(&state_file)?;
                println!("{}", content);
            } else {
                println!("No dream cycle state found at {:?}", state_file);
            }
        }

        Commands::Daemon => {
            println!(
                "🌌 Starting native sovereign dream daemon (idle threshold: {}s)...",
                cli.idle_threshold
            );
            let mut ticker = tokio::time::interval(Duration::from_secs(60));

            loop {
                ticker.tick().await;

                // 1. Check GPU utilization
                let gpu_util = engine.get_gpu_utilization().await;
                if gpu_util > 5 {
                    // Operator or training load is active, skip cycle
                    continue;
                }

                // 2. Check operator idle time
                if let Some(last_act) = engine.get_last_session_activity() {
                    let now = chrono::Utc::now();
                    let idle_secs = (now - last_act).num_seconds().max(0) as u64;

                    if idle_secs >= cli.idle_threshold {
                        tracing::info!(
                            "Operator idle for {}s with 0% GPU load. Initiating dream cycle.",
                            idle_secs
                        );
                        if let Err(e) = engine.execute_dream_cycle().await {
                            tracing::warn!("Dream cycle error: {}", e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
