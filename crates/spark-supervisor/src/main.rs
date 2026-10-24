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

fn default_supervisor_config_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config/aien/supervisor.toml")
}

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use spark_supervisor::config::SupervisorConfig;
use spark_supervisor::supervisor::Supervisor;

/// Exit code for `status`: live state is only held inside the running daemon.
const EXIT_STATUS_UNAVAILABLE: i32 = 2;

const STATUS_UNAVAILABLE_MSG: &str = "spark-supervisor status: live status is not available from a separate process. \
The supervisor keeps service state in the memory of the `run` process and exposes no IPC or status endpoint, \
so a new process cannot see it. Check the daemon's log output instead.";

#[derive(Parser, Debug)]
#[command(
    name = "spark-supervisor",
    about = "Sovereign Native Process Supervisor for SparkOS"
)]
struct Cli {
    #[arg(long, default_value_os_t = default_supervisor_config_path())]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run,
    /// Refuses with a non-zero exit: there is no channel to a running daemon.
    Status,
    Check,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Run);

    // Refuse before touching the config so `status` never writes a default file.
    if matches!(command, Commands::Status) {
        eprintln!("{}", STATUS_UNAVAILABLE_MSG);
        std::process::exit(EXIT_STATUS_UNAVAILABLE);
    }

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    if !cli.config.exists() {
        eprintln!(
            "Config file not found at {:?}. Generating default supervisor.toml...",
            cli.config
        );
        if let Some(p) = cli.config.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let default_toml = format!(
            "[services.cortex]\ncommand = \"{}/.local/bin/cortex-rs\"\nargs = [\"--port\", \"18080\"]\nrestart = \"always\"\nhealth_url = \"http://127.0.0.1:18080/health\"\n\n[services.cockpit]\ncommand = \"{}/.local/bin/spark-cockpit-rs\"\nrestart = \"always\"\nhealth_url = \"http://127.0.0.1:18095/health\"\n",
            home, home
        );
        std::fs::write(&cli.config, default_toml)?;
    }

    let config = SupervisorConfig::load_from_file(&cli.config)?;

    match command {
        Commands::Run => {
            Supervisor::new(config).run_daemon().await?;
        }
        Commands::Status => unreachable!("handled above"),
        Commands::Check => {
            println!("Configuration {:?} is valid.", cli.config);
        }
    }

    Ok(())
}
