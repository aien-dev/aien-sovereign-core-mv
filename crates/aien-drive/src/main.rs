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

use aien_drive::*;
use clap::{Parser, ValueEnum};
use std::time::Duration;

#[derive(Clone, ValueEnum)]
enum Seat {
    Lightning,
    Judge,
}

/// AIEN Autonomous API Driver
#[derive(Parser)]
struct Args {
    /// Task directive for AIEN
    #[arg(required = true)]
    directive: Vec<String>,
    /// Inference seat on Spark
    #[arg(long, value_enum, default_value = "lightning")]
    seat: Seat,
    /// Maximum execution turns
    #[arg(long, default_value_t = 15)]
    max_turns: u32,
}

fn main() {
    let a = Args::parse();
    let (ep, model) = match a.seat {
        Seat::Lightning => (ENDPOINT_LIGHTNING, MODEL_LIGHTNING),
        Seat::Judge => (ENDPOINT_JUDGE, MODEL_JUDGE),
    };
    // AIEN_MODEL_ENDPOINT / AIEN_MODEL_NAME override the seat defaults (the
    // Python original read them but then overwrote them from --seat).
    let env = |k: &str, d: &str| {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| d.to_string())
    };
    let cfg = Config {
        endpoint: env("AIEN_MODEL_ENDPOINT", ep),
        model: env("AIEN_MODEL_NAME", model),
        cortex_endpoint: env("CORTEX_ENDPOINT", DEFAULT_CORTEX),
        cortex_space: env("AIEN_CORTEX_SPACE", "aien-sources"),
        request_timeout: Duration::from_secs(180),
    };
    match run_directive(&cfg, &a.directive.join(" "), a.max_turns) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
