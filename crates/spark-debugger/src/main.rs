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

use aien_localhttp::Client;
use clap::{Parser, Subcommand};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod audit;
mod daemon;
mod mcp;
mod reporter;

use audit::run_full_audit;
use daemon::run_watchdog_daemon;
use mcp::run_mcp_server;

#[derive(Parser, Debug)]
#[command(
    name = "spark-debugger",
    about = "Autonomous Continuous Watchdog and MCP Debugger for SparkOS"
)]
struct Cli {
    #[arg(long, default_value_t = default_debugger_workspace())]
    workspace: String,

    #[arg(long)]
    daemon: bool,

    #[arg(long)]
    mcp: bool,

    #[arg(long, default_value_t = 30)]
    interval: u64,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Check,
    Daemon,
    Mcp,
}

fn default_debugger_workspace() -> String {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    format!("{}/workspace/aien-sovereign-core", home)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // aien-localhttp always sets TCP_NODELAY, so the old tcp_nodelay(true) is kept.
    let client = Client::builder().timeout(Duration::from_secs(5)).build();

    if cli.mcp || matches!(cli.command, Some(Commands::Mcp)) {
        // Run MCP server over stdio without standard logging interfering with stdout
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
            .init();
        return run_mcp_server(client, cli.workspace).await;
    }

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    if cli.daemon || matches!(cli.command, Some(Commands::Daemon)) {
        return run_watchdog_daemon(client, cli.workspace, cli.interval).await;
    }

    // Default: one-shot check
    println!();
    println!("=== SparkOS Sovereign Watchdog: System Diagnostic Audit ===");
    let report = run_full_audit(&client, &cli.workspace).await;

    println!(
        "{:<22} {:<38} {:<10} {:<10}",
        "SERVICE", "ENDPOINT", "STATUS", "LATENCY"
    );
    println!("{}", "-".repeat(82));
    for s in &report.services {
        let status_str = if s.reachable { "ONLINE" } else { "OFFLINE" };
        let code_str = s
            .status_code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<22} {:<38} {:<10} {:<10}",
            s.name,
            s.endpoint,
            format!("{} ({})", status_str, code_str),
            format!("{}ms", s.latency_ms)
        );
    }
    println!("{}", "-".repeat(82));

    println!(
        "
--- Invariant Checks ---"
    );
    println!(
        "TPM Vault Clean (Zero .env files): {}",
        if report.invariants.plaintext_env_files_found.is_empty() {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    if !report.invariants.plaintext_env_files_found.is_empty() {
        for f in &report.invariants.plaintext_env_files_found {
            println!("  Leaked file: {}", f);
        }
    }
    println!(
        "Unslop Standard (Zero em/en dashes): {}",
        if report.invariants.unslop_violations.is_empty() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    println!(
        "
--- Workspace Compile Check ---"
    );
    println!(
        "Cargo Check: {}",
        if report.build.passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    if !report.build.errors.is_empty() {
        for e in &report.build.errors {
            println!("  {}", e);
        }
    }

    println!(
        "
Overall Diagnostic Status: {}",
        if report.overall_healthy {
            "ALL SYSTEMS GREEN"
        } else {
            "ATTENTION REQUIRED"
        }
    );
    println!();

    Ok(())
}
