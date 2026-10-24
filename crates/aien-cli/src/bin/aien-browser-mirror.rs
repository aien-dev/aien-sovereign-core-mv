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

//! `aien-browser-mirror`: headless-Chrome Cockpit mirror over the Chrome
//! DevTools Protocol. Rust replacement for the removed Python script
//! `basecamp/scripts/browser_mirror_test.py` (sovereign-core #180), same flags.
//!
//! ```text
//! aien-browser-mirror --test                      Cockpit self-test (default)
//! aien-browser-mirror --mentor "PROMPT"           one chat prompt, wait for the reply
//! aien-browser-mirror --screenshot [FILE.png]     screenshot of --url
//! aien-browser-mirror --eval "EXPR"               evaluate JavaScript on --url
//!   --url URL        Cockpit or any page (default http://127.0.0.1:18095)
//!   --out-dir DIR    where screenshots and reports go (default ~/basecamp/ui-tests)
//! ```
//!
//! Prints the JSON report on stdout. Exit status 0 when the report says
//! PASSED, COMPLETED or ok; 1 otherwise; 2 for a usage error.

use aien_cli::browser_mirror as bm;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: aien-browser-mirror [--test | --mentor PROMPT | --screenshot [FILE] | --eval EXPR]\n\
         \x20                          [--url URL] [--out-dir DIR]"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut action = "test".to_string();
    let mut arg: Option<String> = None;
    let mut url = std::env::var("AIEN_COCKPIT_URL").unwrap_or_else(|_| bm::DEFAULT_URL.to_string());
    let mut out_dir: PathBuf = bm::default_out_dir();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--test" => action = "test".to_string(),
            "--mentor" => {
                action = "mentor".to_string();
                if let Some(p) = args.get(i + 1).filter(|p| !p.starts_with("--")) {
                    arg = Some(p.clone());
                    i += 1;
                }
            }
            "--screenshot" => {
                action = "screenshot".to_string();
                if let Some(p) = args.get(i + 1).filter(|p| !p.starts_with("--")) {
                    arg = Some(p.clone());
                    i += 1;
                }
            }
            "--eval" => {
                action = "eval".to_string();
                arg = Some(args.get(i + 1).cloned().unwrap_or_else(|| usage()));
                i += 1;
            }
            "--url" => {
                url = args.get(i + 1).cloned().unwrap_or_else(|| usage());
                i += 1;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.get(i + 1).cloned().unwrap_or_else(|| usage()));
                i += 1;
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
        i += 1;
    }
    let report = bm::dispatch(&action, arg.as_deref(), &url, &out_dir);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    let status = report.get("status").and_then(|s| s.as_str()).unwrap_or("");
    let ok = matches!(status, "PASSED" | "COMPLETED" | "ok");
    std::process::exit(if ok { 0 } else { 1 });
}
