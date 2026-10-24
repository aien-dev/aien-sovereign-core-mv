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

use crate::telemetry::get_gpu_telemetry;
use colored::*;
use std::io::{stdout, Write};

/// Second banner line (sc#414): the runtime is the native Rust daemon. Modular
/// MAX was left on 2026-10-02 and no model seat listens on port 18006 here.
pub fn banner_runtime_line() -> String {
    let text = "Grace Blackwell GB10 • Native Rust runtime • Local model, no cloud";
    let inner = 76;
    let pad = inner - 2 - text.chars().count();
    format!("│  {text}{}│", " ".repeat(pad))
}

pub fn print_banner() {
    println!(
        "{}",
        "╭────────────────────────────────────────────────────────────────────────────╮"
            .cyan()
            .bold()
    );
    println!(
        "{}",
        "│  ⚡ AIEN • Autonomous Intelligence & Execution Node (NVIDIA DGX Spark)    │"
            .cyan()
            .bold()
    );
    println!("{}", banner_runtime_line().cyan());
    println!(
        "{}",
        "╰────────────────────────────────────────────────────────────────────────────╯"
            .cyan()
            .bold()
    );
    print_status_bar();
    println!();
}

pub fn print_status_bar() {
    let gpu = get_gpu_telemetry();
    let gpu_status = if gpu.available {
        let temp = gpu.display_temp();
        let used_gb = gpu
            .used_mb
            .map(|u| format!("{:.1} GB", u as f64 / 1024.0))
            .unwrap_or_else(|| "? GB".to_string());
        let total_gb = gpu
            .total_mb
            .map(|t| format!("{:.1} GB", t as f64 / 1024.0))
            .unwrap_or_else(|| "? GB".to_string());
        format!("GPU: {} | {}/{}", temp, used_gb, total_gb)
    } else {
        "GPU: Offline / Unavailable".to_string()
    };

    let status = format!(
        "[{}] • Runtime: native AIEN daemon • Mask: AIEN • Swarm: aien-hive",
        gpu_status
    );
    println!("{}", status.black().on_cyan().bold());
}

pub fn print_tool_start(tool_name: &str, summary: &str) {
    print!(
        "{} {}",
        "⚙ [Executing:".yellow().bold(),
        format!("{}({})]...", tool_name, summary).yellow()
    );
    let _ = stdout().flush();
}

pub fn print_tool_done(tool_name: &str, elapsed_ms: u128, success: bool) {
    if success {
        println!(
            "\r{}",
            format!("✓ [{}] Done ({}ms)", tool_name, elapsed_ms)
                .green()
                .bold()
        );
    } else {
        println!(
            "\r{}",
            format!("✗ [{}] Failed ({}ms)", tool_name, elapsed_ms)
                .red()
                .bold()
        );
    }
    let _ = stdout().flush();
}

pub fn print_safety_prompt(cmd: &str) -> bool {
    println!(
        "\n{}",
        "⚠ [SAFETY GATE] AIEN requested high-risk operation:"
            .red()
            .bold()
    );
    println!("  {}", cmd.yellow());
    print!("{}", "Do you approve execution? [y/N]: ".bold());
    let _ = stdout().flush();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_ok() {
        let trimmed = input.trim().to_lowercase();
        return trimmed == "y" || trimmed == "yes";
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_describes_native_runtime_not_max() {
        let l = banner_runtime_line();
        assert!(l.contains("Native Rust runtime"), "{l}");
        for stale in ["MAX", "26.5", "18006"] {
            assert!(!l.contains(stale), "stale banner text {stale}: {l}");
        }
        assert_eq!(l.chars().count(), 78, "box width: {l}");
    }
}
