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

use chrono::Utc;
use colored::*;
use reqwest::Client;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process::Command;

use crate::crumbs::{
    discover_workspace_crumbs_summary, load_or_init_dir_crumb, record_directory_crumb,
};
use crate::hive::hive_roster;
use crate::platform::PlatformContext;
use crate::telemetry::get_gpu_telemetry;

#[allow(dead_code)]
pub struct NestingReport {
    pub hostname: String,
    pub user: String,
    pub gpu_info: String,
    pub disk_free: String,
    pub seats_status: Vec<(String, bool)>,
    pub hive_active_count: usize,
    pub pending_parks_count: usize,
    pub scent_marked: bool,
    pub grounding_context: String,
}

pub async fn perform_nesting_ritual(session_id: &str, print_tui: bool) -> NestingReport {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let platform = PlatformContext::detect();
    let hostname = platform.hostname.clone();
    let user = platform.current_user.clone();

    // 1. Circle & Survey Surroundings (Hardware, user, disk, seats)
    let gpu = get_gpu_telemetry();
    let gpu_info = if gpu.available {
        format!(
            "{} ({}, {}, {} util)",
            gpu.display_name(),
            gpu.display_temp(),
            gpu.display_vram(),
            gpu.display_util()
        )
    } else {
        "Unavailable / Sensor Offline (CPU Execution Mode)".to_string()
    };

    let home_str = platform.home_dir.to_string_lossy().to_string();
    let disk_free = Command::new("df")
        .args(["-h", &home_str])
        .output()
        .map(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines()
                .nth(1)
                .and_then(|l| l.split_whitespace().nth(3))
                .unwrap_or("unknown")
                .to_string()
        })
        .unwrap_or_else(|_| "unknown".to_string());

    // Check service seats
    let mut seats_status = Vec::new();
    let seat_18006 = client
        .get("http://127.0.0.1:18006/v1/models")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    seats_status.push(("Model (18006)".to_string(), seat_18006));

    let judge_18082 = client
        .get("http://127.0.0.1:18082/v1/models")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    seats_status.push(("Judge (18082)".to_string(), judge_18082));

    let cortex_18080 = client
        .get("http://127.0.0.1:18080/health")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    seats_status.push(("Cortex (18080)".to_string(), cortex_18080));

    let dream_18085 = client
        .get("http://127.0.0.1:18085/health")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    seats_status.push(("Dream (18085)".to_string(), dream_18085));

    // Environmental verification: if running under dedicated DGX Spark profile, enforce strict anchors
    if platform.is_dgx_spark() {
        let mut critical_failures = Vec::new();
        if user != "drakestapleton" && user != "root" {
            critical_failures.push(format!("DGX Spark workstation policy: running as {}, must operate as drakestapleton on spark hardware", user));
        }
        if !seat_18006 {
            critical_failures.push(
                "Critical Seat Down: Model Seat (port 18006 / Nemotron-3.5) is offline".to_string(),
            );
        }
        if !judge_18082 {
            critical_failures.push(
                "Critical Seat Down: JSpace Truth Judge (port 18082 / Llama-3.2) is offline"
                    .to_string(),
            );
        }
        if !cortex_18080 {
            critical_failures.push(
                "Critical Seat Down: Spark Cortex Memory (port 18080) is offline".to_string(),
            );
        }

        if !critical_failures.is_empty() {
            eprintln!(
                "\n{}",
                "❌ NESTING RITUAL FAILED: CRITICAL HARD-BLOCK TRIGGERED"
                    .red()
                    .bold()
            );
            eprintln!("{}", "The agent cannot settle or take ungrounded actions because required environmental anchors are down:".red());
            for err in &critical_failures {
                eprintln!("  ✖ {}", err.bold().red());
            }
            eprintln!(
                "\n{}",
                "Aborting startup to prevent ungrounded or misdirected execution.".yellow()
            );
            std::process::exit(1);
        }
    }

    // 2. Check the Wind (Hive swarm activity, pulse, vibe & parked notes)
    let pulse_path = platform.home_dir.join("basecamp").join("hive-pulse.json");
    let (pulse_coherence, pulse_vibe, _pulse_status) = if pulse_path.exists() {
        if let Ok(c) = fs::read_to_string(&pulse_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c) {
                let coh = v
                    .get("coherence_score")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(95);
                let vibe = v
                    .get("vibe_summary")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("Harmonious forward vector")
                    .to_string();
                let st = v
                    .get("pulse_status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("harmonious")
                    .to_string();
                (Some(coh), Some(vibe), Some(st))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    let roster_text = hive_roster();
    let hive_active_count = if roster_text.contains("idle")
        || roster_text.contains("Empty")
        || roster_text.contains("0")
    {
        0
    } else {
        roster_text
            .lines()
            .filter(|l| l.contains("running") || l.contains("busy"))
            .count()
    };

    let park_dir = platform.home_dir.join("atlas-prime-workspace").join("park");
    let pending_parks_count = if park_dir.exists() {
        fs::read_dir(&park_dir).map(|r| r.count()).unwrap_or(0)
    } else {
        0
    };

    // 3. Hierarchical Directory Crumb Inspection & Scent-Drop
    let cwd = std::env::current_dir().unwrap_or_else(|_| platform.home_dir.clone());
    let dir_crumb = load_or_init_dir_crumb(&cwd, None);
    let crumb_topography = discover_workspace_crumbs_summary(&cwd);

    // Record presence into directory crumb
    record_directory_crumb(
        "AIEN",
        session_id,
        &cwd,
        "signin",
        "Operator session initiation",
        "Awaiting operator command",
    );

    // 4. Leave Scent (Record presence in basecamp/events.jsonl)
    let events_path = platform.home_dir.join("basecamp").join("events.jsonl");
    let mut scent_marked = false;
    let event_entry = json!({
        "event": "agent_nesting",
        "timestamp": Utc::now().to_rfc3339(),
        "session_id": session_id,
        "actor": "AIEN",
        "mask": "AIEN",
        "user": user,
        "host": hostname,
        "cwd": cwd.to_string_lossy().to_string(),
        "status": "grounded",
        "gpu_available": gpu.available,
        "gpu_temp_c": gpu.temp_c,
        "disk_free": disk_free
    });

    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&events_path)
    {
        if writeln!(f, "{}", event_entry).is_ok() {
            scent_marked = true;
        }
    }

    // 5. Flatten the Terrain (Build Context & Render TUI)
    let grounding_context = format!(
        "[NESTING RITUAL RECORD - GROUNDING SEQUENCE]\n\
         Host: {} | Operator: {} | Disk Free: {}\n\
         GPU: {}\n\
         Seats Verified: Model(18006)={}, Judge(18082)={}, Cortex(18080)={}, Dream(18085)={}\n\
         Swarm Status: {} active peer cell(s) | Pending Park Notes: {}\n\
         Hive Pulse & Vibe: Coherence={}%, Vibe={}\n\
         Scent Registered: basecamp/events.jsonl ({})\n\n\
         {}\n\n\
         Operating Rule: Consult the directory crumb topography above. Before modifying files, declare your vector and verify boundaries.",
        hostname, user, disk_free, gpu_info,
        seat_18006, judge_18082, cortex_18080, dream_18085,
        hive_active_count, pending_parks_count,
        pulse_coherence.unwrap_or(95), pulse_vibe.as_deref().unwrap_or("Aligned and focused"),
        session_id, crumb_topography
    );

    if print_tui {
        println!(
            "{}",
            "╭─────────────────────────────────────────────────────────────╮".cyan()
        );
        println!(
            "│  {}                  │",
            "🐕 NESTING RITUAL (Agent Grounding Initiation)"
                .bold()
                .yellow()
        );
        println!(
            "{}",
            "├─────────────────────────────────────────────────────────────┤".cyan()
        );
        println!(
            "│  {} {} ({})",
            "Circle Terrain:".bold().white(),
            hostname.green(),
            user.cyan()
        );
        println!("│  ├─ Storage Free:     {} available", disk_free.green());
        if gpu.available {
            println!(
                "│  ├─ GPU Telemetry:   {} | {} | {}",
                gpu.display_name(),
                gpu.display_temp(),
                gpu.display_vram()
            );
        } else {
            println!(
                "│  ├─ GPU Telemetry:   {}",
                "Unavailable / Sensor Offline (CPU Mode)".yellow()
            );
        }
        print!("│  └─ Seats Online:     ");
        for (name, ok) in &seats_status {
            if *ok {
                print!("{}✓ ", name.green());
            } else {
                print!("{}✗ ", name.red());
            }
        }
        println!();
        println!(
            "│  {}     {} active hive cell(s) | {} park note(s)",
            "Check the Wind:".bold().white(),
            hive_active_count,
            pending_parks_count
        );
        if let (Some(coh), Some(vibe)) = (&pulse_coherence, &pulse_vibe) {
            println!(
                "│  ├─ Hive Pulse:        {} (Coherence: {}%)",
                "💚 Heartbeat Active".bold().green(),
                coh
            );
            println!("│  ├─ Hive Vibe:         {}", vibe.italic().magenta());
        }

        let above_str = dir_crumb
            .above
            .as_ref()
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "Root".to_string());
        let subdirs: Vec<&str> = dir_crumb
            .below
            .iter()
            .filter(|i| i.item_type == "dir")
            .map(|i| i.name.as_str())
            .collect();
        let subdirs_str = if subdirs.is_empty() {
            "none".to_string()
        } else {
            subdirs.join(", ")
        };

        println!(
            "│  {}  Directory: {} (.crumb)",
            "Directory Crumb:".bold().white(),
            dir_crumb.dir_name.cyan()
        );
        if let Some(p) = &dir_crumb.purpose {
            println!("│  ├─ Purpose:          {}", p.statement.italic().cyan());
        }
        println!("│  ├─ Above:            {}", above_str.yellow());
        println!("│  ├─ Below:            [{}]", subdirs_str.white());
        println!(
            "│  └─ Local History:    {} recorded action(s)",
            dir_crumb.history.len()
        );

        println!(
            "│  {}       Presence signed into directory .crumb & basecamp",
            "Leave Scent:".bold().white()
        );
        println!(
            "│  {}    Workspace anchored, topography loaded, vector set.",
            "Flatten Terrain:".bold().white()
        );
        println!(
            "{}",
            "╰─────────────────────────────────────────────────────────────╯".cyan()
        );
        println!(
            "{}",
            "✓ Grounding complete. Directory crumb topography active. Ready.\n".dimmed()
        );
    }

    NestingReport {
        hostname,
        user,
        gpu_info,
        disk_free,
        seats_status,
        hive_active_count,
        pending_parks_count,
        scent_marked,
        grounding_context,
    }
}
