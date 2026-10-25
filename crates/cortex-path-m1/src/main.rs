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

use chrono::{DateTime, Utc};
use clap::Parser;
use cortex_path_m1::run::{self, RunOutcome};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "cortex-path-m1")]
struct Args {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// RFC3339 timestamp stored with the snapshot. Freshness is measured against this, not the wall clock.
    #[arg(long)]
    snapshot_time: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match execute(&args) {
        Ok(ExitCode::SUCCESS) => ExitCode::SUCCESS,
        Ok(code) => code,
        Err(err) => {
            eprintln!("cortex-path-m1: {err}");
            ExitCode::from(1)
        }
    }
}

fn execute(args: &Args) -> Result<ExitCode, String> {
    let snapshot_time = DateTime::parse_from_rfc3339(&args.snapshot_time)
        .map_err(|err| format!("snapshot-time: {err}"))?
        .with_timezone(&Utc);
    let fixture_bytes =
        fs::read(&args.fixture).map_err(|err| format!("read {}: {err}", args.fixture.display()))?;
    let binary_bytes = fs::read(std::env::current_exe().map_err(|err| err.to_string())?)
        .map_err(|err| format!("read experiment binary: {err}"))?;
    match run::run_experiment(&args.snapshot, &fixture_bytes, snapshot_time, &binary_bytes)? {
        RunOutcome::Rejected(rejection) => {
            eprintln!("scientific verdict refused");
            for reason in rejection.reasons {
                eprintln!("- {reason}");
            }
            Ok(ExitCode::from(2))
        }
        RunOutcome::Report(report) => {
            let json = serde_json::to_string_pretty(&report).map_err(|err| err.to_string())?;
            fs::write(&args.out, json)
                .map_err(|err| format!("write {}: {err}", args.out.display()))?;
            println!(
                "wrote {} pass={} strong_pass={} fail={}",
                args.out.display(),
                report.pass,
                report.strong_pass,
                report.fail
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}
