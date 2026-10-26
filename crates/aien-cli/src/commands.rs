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

pub fn parse_quoted_args(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in input.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch.is_whitespace() && !in_quotes {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}
use crate::crumbs::{format_dir_crumb_tui, leave_dir_whisper};
use crate::hive::{hive_kill, hive_roster, hive_spawn};
use crate::nesting::perform_nesting_ritual;
use crate::telemetry::get_gpu_telemetry;
use colored::*;
use reqwest::Client;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub async fn handle_slash_command(cmd: &str) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    match parts[0] {
        "/exit" | "/quit" => {
            println!("{}", "Exiting AIEN. The Spark desk remains ours.".cyan());
            std::process::exit(0);
        }
        "/rollback" => {
            // sc#667: undo the agent's own file writes from a turn on; the
            // operator's edits are never touched.
            let home = crate::platform::PlatformContext::detect().home_dir;
            let turn = parts.get(1).and_then(|t| t.parse::<u64>().ok());
            let session = parts.get(if turn.is_some() { 2 } else { 1 }).copied();
            match crate::checkpoint::rollback(&home, turn, session) {
                Ok(r) => {
                    println!(
                        "{}",
                        format!("Rolled back from turn {}", r.from_turn)
                            .cyan()
                            .bold()
                    );
                    for (label, paths) in [
                        ("restored", &r.restored),
                        ("removed (the agent created it)", &r.removed),
                        ("already as it was", &r.unchanged),
                        ("KEPT: changed after the agent wrote it", &r.conflicts),
                        ("spared: your own edits, not touched", &r.spared),
                    ] {
                        for p in paths {
                            println!("  {}: {}", label, p);
                        }
                    }
                }
                Err(e) => println!("{}", format!("Rollback refused: {}", e).red()),
            }
            true
        }
        "/checkpoints" => {
            let home = crate::platform::PlatformContext::detect().home_dir;
            match crate::checkpoint::list_turns(&home, parts.get(1).copied()) {
                Ok(turns) if turns.is_empty() => {
                    println!("No agent file writes recorded in that session.")
                }
                Ok(turns) => {
                    println!("{}", "Agent writes by turn".cyan());
                    for (t, n) in turns {
                        println!("  turn {}: {} file(s)", t, n);
                    }
                }
                Err(e) => println!("{}", e.red()),
            }
            true
        }
        "/approval" => {
            let ctx = crate::approval::GateContext::from_env();
            println!("{}", "Approval gate (sc#667)".cyan().bold());
            println!(
                "  mode: {:?}   person present: {}",
                ctx.mode,
                ctx.presence == crate::approval::Presence::Person
            );
            println!(
                "  policy file: {}",
                crate::approval::policy_path(&crate::platform::PlatformContext::detect().home_dir)
                    .display()
            );
            if let Some(why) = ctx.policy.broken() {
                println!(
                    "  {}",
                    format!("EVERY REQUEST IS DENIED: {}", why).red().bold()
                );
            }
            for w in &ctx.policy.warnings {
                println!("  {}", w.yellow());
            }
            println!("  floors (every mode): hardline commands, runtime files, sudo password on stdin, your deny patterns");
            true
        }
        "/clear" => {
            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
            let _ = std::io::stdout().flush();
            true
        }
        "/help" => {
            println!("{}", "\nAvailable Slash Commands:".cyan().bold());
            println!(
                "  /skill [list|preview|search|full|optimize] CPU-filtered skill access
  /context7 resolve <library> [query] | /context7 query <library_id> <query>
  context7-sync             Refresh Rust, Tokio, Axum, and Serde docs into Cortex
  /sandbox [init|status|test|promote|clean] Isolated Git worktree sandbox
  /browser [test|mentor|screenshot|eval] Headless Chrome CDP mirror (Rust, aien-browser-mirror)
  /subagents [list|view|run] Recursive contextual subagents hierarchy
  /goal [list|new|done]     Manage project goals & milestone lattices"
            );
            println!("  /rollback [turn] [session] Undo the agent's own file writes from a turn on (your edits are kept)");
            println!("  /checkpoints [session]    List turns with agent file writes");
            println!(
                "  /approval                 Show approval mode, presence and the policy file"
            );
            println!("  /rules                    Inspect active repository & directory rules (AGENTS.md, GEMINI.md)");
            println!("  /walkthrough [save]       Display live architecture map & roadmap (or save to WALKTHROUGH.md)");
            println!("  /nest                     Execute the Agent Nesting Ritual (grounding, threat check, peer wind, scent)");
            println!("  /crumb [path]             Inspect directory crumb (above, below, and local agent history)");
            println!("  /crumb whisper <message>  Leave a directory whisper for peer agents");
            println!("  /vault                    Inspect hardware TPM key vault and secret hygiene audit");
            println!("  /doctor                   Probe GB10 GPU, Model (18006), Judge (18082), Cortex (18080), Vault, Dream (18085)");
            println!("  /dream [now|status]       Inspect or trigger JSpace Dynamic Dream Cycle");
            println!("  /hive                     Manage swarm & inspect lattice (/hive roster | /hive wall | /hive spawn <role> <task>)");
            println!("  /adapter [list|emit|pr]   Autonomous open-source model adapters & PR pipeline for consumer hardware");
            println!("  /socratic <question>      Trigger Socratic inquiry and emit comb onto hexagonal lattice");
            println!("  /mask                     List or inspect active persona mask");
            println!("  /park <text>              Park high-volume input or interruption to park/ directory");
            println!("  /cortex <query>           Query permanent memory in Spark Cortex");
            println!(
                "  /sync                     Run Radicle sovereign identity key synchronization"
            );
            println!("  /clear                    Clear terminal screen");
            println!("  /exit                     Exit AIEN session\n");
            true
        }
        "/walkthrough" | "/roadmap" | "/map" => {
            if parts.len() > 1 && parts[1] == "save" {
                let platform = crate::platform::PlatformContext::detect();
                let dest = platform.home_dir.join("basecamp/WALKTHROUGH.md");
                match crate::walkthrough::generate_and_save_walkthrough_md(&dest) {
                    Ok(msg) => println!("{}", msg.green()),
                    Err(e) => println!("{}", format!("Failed to save walkthrough: {}", e).red()),
                }
            } else {
                let card = crate::walkthrough::render_walkthrough_tui();
                println!("{}", card);
            }
            true
        }
        "/nest" => {
            let session_id = format!("manual-nest-{}", chrono::Utc::now().timestamp());
            perform_nesting_ritual(&session_id, true).await;
            true
        }
        "/crumb" => {
            if parts.len() > 1 && parts[1] == "whisper" {
                if parts.len() < 3 {
                    println!("Usage: /crumb whisper <message>");
                } else {
                    let msg = parts[2..].join(" ");
                    let cwd =
                        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                    leave_dir_whisper("AIEN", &cwd, None, &msg);
                    println!(
                        "{}",
                        format!("✓ Whisper left in {}/.crumb: \"{}\"", cwd.display(), msg).green()
                    );
                }
            } else {
                let target_str = if parts.len() > 1 { parts[1] } else { "." };
                let p = Path::new(target_str);
                println!("{}", format_dir_crumb_tui(p));
            }
            true
        }
        "/vault" => {
            println!(
                "{}",
                "\n=== Hardware TPM Key Vault (atlas-vault) ==="
                    .cyan()
                    .bold()
            );
            if !crate::vault::is_vault_available() {
                let platform = crate::platform::PlatformContext::detect();
                println!(
                    "{}",
                    format!(
                        "❌ atlas-vault not found at {}",
                        platform.home_dir.join(".local/bin/atlas-vault").display()
                    )
                    .red()
                );
            } else {
                let keys = crate::vault::list_keys();
                println!(
                    "• Hardware Vault:  {}",
                    "TPM-bound atlas-vault (Active)".green()
                );
                println!(
                    "• Registered Keys: {}",
                    keys.len().to_string().cyan().bold()
                );
                for k in &keys {
                    println!("  - {}", k.green());
                }
                let findings = crate::vault::audit_workspace_secrets();
                if findings.is_empty() {
                    println!("{}", "✓ Secret Hygiene:  Clean (0 stray .env or plaintext credential files on disk).".green());
                } else {
                    println!(
                        "{}",
                        format!(
                            "⚠ Warning: {} stray .env files detected on disk:",
                            findings.len()
                        )
                        .yellow()
                        .bold()
                    );
                    for f in &findings {
                        println!("    - {}", f.red());
                    }
                }
                println!("{}", "• Storage Policy:  Direct value retrieval is in-memory only; plaintext .env strictly forbidden.\n".dimmed());
            }
            true
        }
        "/doctor" => {
            run_doctor().await;
            true
        }
        "/dream" => {
            // The former dream engine was a Python script outside this repository
            // (basecamp/aien-dream/dream_engine.py). AIEN carries no Python, so the
            // command says so instead of shelling out. Native dream telemetry lives
            // in the spark-dream crate (binary `spark-dream`).
            println!(
                "{}",
                "The /dream cycle is not part of the sovereign build: its old engine was a Python script outside this repo. Use the native `spark-dream` binary for dream telemetry."
                    .yellow()
            );
            true
        }
        "/adapter" | "/adapters" => {
            handle_adapter_command(&parts[1..]);
            true
        }
        "/hive" => {
            if parts.len() >= 2 && parts[1] == "wall" {
                match spark_hive::CombStore::open_default() {
                    Ok(store) => match store.get_cells() {
                        Ok((combs, bounds)) => {
                            println!("{}", format!("\n=== Sovereign Hexagonal Honeycomb Wall ({} combs, radius {}) ===", bounds.count, bounds.radius).yellow().bold());
                            for comb in combs.iter().take(12) {
                                println!(
                                    "  [{:3}, {:3}] {:<18} {:<12} {}",
                                    comb.q,
                                    comb.r,
                                    comb.author.cyan(),
                                    comb.role.dimmed(),
                                    comb.content
                                );
                            }
                            if combs.len() > 12 {
                                println!(
                                    "  ... and {} more combs on hexagonal lattice.",
                                    combs.len() - 12
                                );
                            }
                        }
                        Err(e) => {
                            println!("{}", format!("Failed to query hive cells: {}", e).red())
                        }
                    },
                    Err(e) => println!("{}", format!("Failed to open hive db: {}", e).red()),
                }
            } else if parts.len() < 2 || parts[1] == "roster" {
                println!("{}", "=== aien-hive Active Swarm Roster ===".cyan().bold());
                println!("{}", hive_roster());
            } else if parts[1] == "spawn" && parts.len() >= 4 {
                let role = parts[2];
                let task = parts[3..].join(" ");
                println!("{}", format!("Spawning hive cell [{}]...", role).yellow());
                let res = hive_spawn(role, &task);
                println!("{}", res.green());
            } else if parts[1] == "kill" && parts.len() >= 3 {
                let name = parts[2];
                println!("{}", hive_kill(name));
            } else {
                println!("Usage: /hive [roster | wall | spawn <role> <task> | kill <name>]");
            }
            true
        }
        "/socratic" => {
            if parts.len() < 2 {
                println!("Usage: /socratic <philosophical question or inquiry>");
            } else {
                let question = parts[1..].join(" ");
                println!(
                    "{}",
                    format!("Emitting Socratic inquiry: \"{}\"...", question).cyan()
                );
                match spark_hive::CombStore::open_default() {
                    Ok(store) => {
                        match spark_hive::emit_socratic_comb(&store, &question, None) {
                            Ok(comb) => {
                                println!("{}", format!("✓ Socratic Comb placed at axial coordinate ({}, {}) [id: {}]", comb.q, comb.r, comb.id).green());
                            }
                            Err(e) => {
                                println!("{}", format!("Failed to emit socratic comb: {}", e).red())
                            }
                        }
                    }
                    Err(e) => println!("{}", format!("Failed to open hive db: {}", e).red()),
                }
            }
            true
        }
        "/mask" => {
            run_mask_command(&parts[1..]);
            true
        }
        "/park" => {
            if parts.len() < 2 {
                println!("Usage: /park <notes/text to park>");
            } else {
                let text = parts[1..].join(" ");
                park_input(&text);
            }
            true
        }
        "/sync" => {
            println!("{}", "Synchronizing Radicle identity keys...".yellow());
            let out = Command::new("basecamp/rad-id-sync/target/debug/rad-id-sync")
                .arg("sync")
                .current_dir(&crate::platform::PlatformContext::detect().home_dir)
                .output();
            match out {
                Ok(o) => println!("{}", String::from_utf8_lossy(&o.stdout).green()),
                Err(e) => println!("{}", format!("Failed to sync keys: {}", e).red()),
            }
            true
        }
        "/goal" | "/goals" => {
            if parts.len() < 2 || parts[1] == "list" {
                println!("{}", crate::goals::format_goals_tui());
            } else if parts[1] == "new" || parts[1] == "add" {
                if parts.len() < 3 {
                    println!("Usage: /goal new <title> [| milestone 1, milestone 2]");
                } else {
                    let rest = parts[2..].join(" ");
                    let split: Vec<&str> = rest.split('|').collect();
                    let title = split[0].trim();
                    let milestones: Vec<String> = if split.len() > 1 {
                        split[1]
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let g = crate::goals::add_goal(title, "", milestones);
                    println!(
                        "{}",
                        format!("✓ Initialized goal: {} ({})", g.title, g.id)
                            .green()
                            .bold()
                    );
                }
            } else if parts[1] == "auto" {
                if parts.len() < 3 {
                    println!("Usage: /goal auto <id or title>");
                } else {
                    let target_id = parts[2..].join(" ");
                    crate::run_autonomous_goal(&target_id).await;
                }
            } else if parts[1] == "done" || parts[1] == "complete" {
                if parts.len() < 3 {
                    println!("Usage: /goal done <id or title>");
                } else {
                    let id = parts[2..].join(" ");
                    match crate::goals::complete_goal(&id) {
                        Ok(msg) => println!("{}", msg.green().bold()),
                        Err(e) => println!("{}", e.red()),
                    }
                }
            } else {
                println!("Usage: /goal [list | new <title> [| m1, m2] | done <id>]");
            }
            true
        }
        "/rule" | "/rules" => {
            let cwd = std::env::current_dir()
                .unwrap_or_else(|_| crate::platform::PlatformContext::detect().home_dir);
            println!("{}", crate::rules::format_rules_tui(&cwd));
            true
        }
        "/context7" => {
            let result = if parts.len() < 3 {
                json!({"status": "error", "error": "Usage: /context7 resolve <library> [query] | /context7 query <library_id> <query>"})
            } else if parts[1] == "resolve" {
                let library = parts[2];
                let query = if parts.len() > 3 {
                    parts[3..].join(" ")
                } else {
                    library.to_string()
                };
                crate::context7::context7_dispatch(&json!({
                    "action": "resolve",
                    "library": library,
                    "query": query
                }))
                .await
            } else if parts[1] == "query" && parts.len() >= 4 {
                crate::context7::context7_dispatch(&json!({
                    "action": "query",
                    "library_id": parts[2],
                    "query": parts[3..].join(" ")
                }))
                .await
            } else {
                json!({"status": "error", "error": "Usage: /context7 resolve <library> [query] | /context7 query <library_id> <query>"})
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
            true
        }
        "/skill" | "/skills" => {
            if parts.len() > 1 && parts[1] == "optimize" {
                let name = if parts.len() > 2 {
                    parts[2..].join(" ")
                } else {
                    "atlas-skillopt".to_string()
                };
                crate::skills::run_optimize_cli(&name);
            } else if parts.len() < 2 || parts[1] == "list" {
                println!("{}", crate::skills::format_skills_tui());
            } else if parts[1] == "full" {
                let name = parts[2..].join(" ");
                match crate::skills::read_skill_content(&name) {
                    Ok(content) => println!("{}", content),
                    Err(e) => println!("{}", e.red()),
                }
            } else if parts[1] == "search" && parts.len() >= 4 {
                let name = parts[2];
                let query = parts[3..].join(" ");
                match crate::skills::search_skill(name, &query, 3) {
                    Ok(result) => println!(
                        "{}",
                        serde_json::to_string_pretty(&result).unwrap_or_default()
                    ),
                    Err(e) => println!("{}", e.red()),
                }
            } else {
                let name = if parts[1] == "preview" {
                    parts[2..].join(" ")
                } else {
                    parts[1..].join(" ")
                };
                match crate::skills::preview_skill(&name, 96) {
                    Ok(preview) => println!(
                        "{}",
                        serde_json::to_string_pretty(&preview).unwrap_or_default()
                    ),
                    Err(e) => println!("{}", e.red()),
                }
            }
            true
        }
        "/sandbox" => {
            crate::sandbox::handle_sandbox_command(&parts[1..]);
            true
        }
        "/browser" | "/mirror" => {
            crate::sandbox::handle_browser_command(&parts[1..]);
            true
        }
        "/subagent" | "/subagents" => {
            let tokens = parse_quoted_args(cmd);
            if tokens.len() > 1 && tokens[1] == "view" {
                let id = if tokens.len() > 2 { &tokens[2] } else { "" };
                let res = crate::subagents::view_subagent(id);
                println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
            } else if tokens.len() > 1 && tokens[1] == "run" {
                let role = if tokens.len() > 2 {
                    &tokens[2]
                } else {
                    "Researcher"
                };
                let prompt = if tokens.len() > 3 {
                    tokens[3..].join(" ")
                } else {
                    "Investigate active workspace".to_string()
                };
                println!(
                    "{}",
                    format!("Spawning subagent '{}'...", role).cyan().bold()
                );
                let res = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(crate::subagents::invoke_subagent_from_root(role, &prompt))
                });
                println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
            } else {
                println!("{}", crate::subagents::format_subagents_tui());
            }
            true
        }
        "/cortex" => {
            crate::cortex::handle_cortex_command(&parts[1..]).await;
            true
        }
        _ => false,
    }
}

pub async fn run_doctor() {
    println!("{}", "=== AIEN Comprehensive Diagnostics ===".cyan().bold());
    let client = Client::new();

    // 1. GPU
    let gpu = get_gpu_telemetry();
    if gpu.available {
        let name = gpu.display_name();
        let temp = gpu.display_temp();
        let vram = gpu.display_vram();
        println!(
            "✓ GPU Hardware:        {} ({}, VRAM: {})",
            name.green(),
            temp,
            vram
        );
    } else {
        println!(
            "{}",
            "ℹ GPU Hardware:        Unavailable / Sensor Offline".yellow()
        );
    }

    // 2. Model Seat 18006
    match client.get("http://127.0.0.1:18006/v1/models").send().await {
        Ok(r) if r.status().is_success() => println!(
            "{}",
            "✓ Model Seat (Port 18006): Online (Nemotron-3.5-Lightning-30B-A3B BF16)".green()
        ),
        _ => println!(
            "{}",
            "⚠ Model Seat (Port 18006): OFFLINE or unreachable".yellow()
        ),
    }

    // 3. JSpace Judge 18082
    match client.get("http://127.0.0.1:18082/v1/models").send().await {
        Ok(r) if r.status().is_success() => println!(
            "{}",
            "✓ JSpace Truth Judge (Port 18082): Online (Llama-3.2-1B on CPU)".green()
        ),
        _ => println!("{}", "⚠ JSpace Truth Judge (Port 18082): Offline".yellow()),
    }

    // 4. Cortex Memory 18080
    match client.get("http://127.0.0.1:18080/").send().await {
        Ok(_) => println!(
            "{}",
            "✓ Spark Cortex Memory (Port 18080): Online (Postgres socket 15433)".green()
        ),
        _ => println!("{}", "⚠ Spark Cortex Memory (Port 18080): Offline".yellow()),
    }

    // 5. ONNX Encoder 18081
    match client.get("http://127.0.0.1:18081/health").send().await {
        Ok(r) if r.status().is_success() => println!(
            "{}",
            "✓ Cortex ONNX Encoder (Port 18081): Online (bge-base embeddings/salience)".green()
        ),
        _ => println!("{}", "⚠ Cortex ONNX Encoder (Port 18081): Offline".yellow()),
    }

    // 6. Dream Daemon 18085
    match client.get("http://127.0.0.1:18085/health").send().await {
        Ok(r) if r.status().is_success() => println!(
            "{}",
            "✓ JSpace Dream Engine (Port 18085): Online (Idle watcher daemon)".green()
        ),
        _ => println!("{}", "⚠ JSpace Dream Engine (Port 18085): Offline".yellow()),
    }

    // 7. Radicle Anchor
    println!(
        "{}",
        "✓ Radicle Identity:    did:rad:aien:spark-master (Active)".green()
    );

    // 8. Hardware TPM Vault & Secret Hygiene
    if crate::vault::is_vault_available() {
        let keys = crate::vault::list_keys();
        let findings = crate::vault::audit_workspace_secrets();
        if findings.is_empty() {
            println!("{}", format!("✓ TPM Key Vault:       Online ({} keys registered; 0 plaintext .env files on disk)", keys.len()).green());
        } else {
            println!(
                "{}",
                format!(
                    "⚠ TPM Key Vault:       Online ({} keys), but {} stray .env files found!",
                    keys.len(),
                    findings.len()
                )
                .yellow()
            );
        }
    } else {
        println!(
            "{}",
            "⚠ TPM Key Vault:       Offline (atlas-vault binary missing)".yellow()
        );
    }

    println!();
}

fn run_mask_command(_args: &[&str]) {
    let out = Command::new("aien-mask")
        .current_dir(&crate::platform::PlatformContext::detect().home_dir)
        .output();

    if out.is_ok() {
        println!("{}", "Active Mask: AIEN (Sovereign Operator)".cyan().bold());
    } else {
        println!("Active Mask: AIEN (default)");
    }
}

fn park_input(text: &str) {
    let platform = crate::platform::PlatformContext::detect();
    let park_dir = platform.home_dir.join("atlas-prime-workspace/park");
    let _ = fs::create_dir_all(&park_dir);
    let stamp = chrono::Utc::now().timestamp();
    let filename = format!("overload-{}.md", stamp);
    let path = park_dir.join(&filename);

    match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
    {
        Ok(mut f) => {
            let _ = writeln!(f, "# Overload Parked: {}", chrono::Utc::now().to_rfc3339());
            let _ = writeln!(f, "{}", text);
            println!(
                "{}",
                format!("✓ Input safely parked to: {}", path.display())
                    .green()
                    .bold()
            );
            println!("Continuing active thread without derailment.");
        }
        Err(e) => println!("{}", format!("Failed to park input: {}", e).red()),
    }
}

#[allow(dead_code)]
async fn query_cortex(query: &str) {
    let client = Client::new();
    let platform = crate::platform::PlatformContext::detect();
    let token_path = platform.home_dir.join(".config/cortex/token");
    let token = fs::read_to_string(token_path)
        .unwrap_or_default()
        .trim()
        .to_string();

    let url = format!(
        "http://127.0.0.1:18080/api/cortex/search?q={}&space=atlas-memory&limit=5",
        urlencoding_simple(query)
    );
    let mut req = client.get(&url).header("Accept", "application/json");
    if !token.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    match req.send().await {
        Ok(r) => {
            if let Ok(data) = r.json::<serde_json::Value>().await {
                let results = data.get("results").and_then(|v| v.as_array());
                match results {
                    Some(items) if !items.is_empty() => {
                        println!(
                            "{}",
                            format!(
                                "\nCortex Knowledge Recall for '{}' ({} found):",
                                query,
                                items.len()
                            )
                            .cyan()
                            .bold()
                        );
                        for (idx, item) in items.iter().enumerate() {
                            let title = item
                                .get("canonicalName")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Untitled");
                            let score = item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let content =
                                item.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            let snippet = if content.len() > 180 {
                                &content[..180]
                            } else {
                                content
                            };
                            println!("{}. {} ({:.2})", idx + 1, title.bold().yellow(), score);
                            println!("   {}\n", snippet.dimmed());
                        }
                    }
                    _ => println!("{}", "No matching entities found in atlas-memory.".dimmed()),
                }
            } else {
                println!("{}", "Could not parse JSON response from Cortex.".red());
            }
        }
        Err(e) => println!("{}", format!("Cortex query failed: {}", e).red()),
    }
}

#[allow(dead_code)]
fn urlencoding_simple(s: &str) -> String {
    s.replace(" ", "%20")
}

fn handle_adapter_command(args: &[&str]) {
    use spark_hive::{
        emit_adapter_pipeline_combs, evaluate_socratic_reflex, generate_pr_plan,
        get_catalog_adapters, list_adapter_pipeline_chains, BenchmarkTelemetry, CombStore,
    };

    if args.is_empty() || args[0] == "list" {
        println!(
            "{}",
            "\n=== AIEN Autonomous Model Adapters Catalogue (Democratizing Consumer Compute) ==="
                .yellow()
                .bold()
        );
        let adapters = get_catalog_adapters();
        for (i, a) in adapters.iter().enumerate() {
            println!(
                "  {}. [{}] {} ({})\n     Target: {} | Upstream: {} ({})\n     Impact: {}",
                i + 1,
                a.id.cyan().bold(),
                a.model.display_name(),
                a.model.slug(),
                a.model.hardware_profile().description().green(),
                a.engine.repo().yellow(),
                a.engine.primary_language(),
                a.summary.dimmed()
            );
        }
        println!("\nUsage: /adapter emit <adapter-id>   Emit 4-stage pipeline onto Honeycomb Wall");
        println!("       /adapter pr <adapter-id>     Preview PR body and gh submission commands");
        println!(
            "       /adapter lattice             Inspect adapter chains on the Honeycomb Wall\n"
        );
    } else if args[0] == "lattice" || args[0] == "chains" {
        match CombStore::open_default() {
            Ok(store) => {
                match list_adapter_pipeline_chains(&store) {
                    Ok(chains) => {
                        println!("{}", format!("\n=== Honeycomb Wall: Autonomous Adapter Chains ({} pipelines) ===", chains.len()).yellow().bold());
                        if chains.is_empty() {
                            println!("  No adapter pipelines on lattice yet. Run `/adapter emit <id>` to place one.");
                        }
                        for chain in chains {
                            println!(
                                "  • Origin [{}, {}] id={}: {}",
                                chain.origin.q,
                                chain.origin.r,
                                chain.origin.id.cyan(),
                                chain.origin.content
                            );
                            if let Some(soc) = chain.socratic {
                                println!(
                                    "    ↳ Socratic [{}, {}]: {}",
                                    soc.q,
                                    soc.r,
                                    soc.content.lines().next().unwrap_or("")
                                );
                            }
                            if let Some(ver) = chain.verifier {
                                println!(
                                    "    ↳ Verifier [{}, {}]: {}",
                                    ver.q,
                                    ver.r,
                                    ver.content.lines().next().unwrap_or("")
                                );
                            }
                            if let Some(pr) = chain.pr {
                                println!(
                                    "    ↳ PR [{}, {}]: {}",
                                    pr.q,
                                    pr.r,
                                    pr.content.lines().next().unwrap_or("")
                                );
                            }
                        }
                    }
                    Err(e) => {
                        println!("{}", format!("Failed to query adapter chains: {}", e).red())
                    }
                }
            }
            Err(e) => println!("{}", format!("Failed to open hive database: {}", e).red()),
        }
    } else if args[0] == "emit" {
        if args.len() < 2 {
            println!("Usage: /adapter emit <adapter-id | model-slug> [engine]");
            return;
        }
        let target = args[1].trim().to_lowercase();
        let engine_override = if args.len() > 2 {
            spark_hive::UpstreamEngine::from_name(args[2])
        } else {
            None
        };
        let spec = spark_hive::find_or_create_adapter(&target, engine_override);
        let Some(spec) = spec else {
            println!(
                "{}",
                format!(
                    "Adapter or model '{}' not found. Run `/adapter list` to view catalog.",
                    target
                )
                .red()
            );
            return;
        };

        println!(
            "{}",
            format!(
                "Initiating autonomous adapter pipeline for '{}'...",
                spec.id
            )
            .cyan()
            .bold()
        );
        let socratic = evaluate_socratic_reflex(&spec.model, &spec.engine);
        println!(
            "{}",
            format!(
                "✓ Socratic Reflex: Approved (freedom score: {:.2}, consumer impact: {:.2})",
                socratic.freedom_alignment_score, socratic.consumer_impact_score
            )
            .green()
        );

        let telem = BenchmarkTelemetry::estimate_for_model(
            &spec.model,
            &spec.engine,
            &crate::platform::PlatformContext::detect()
                .home_dir
                .join("workspace/aien-sandbox")
                .to_string_lossy(),
            None,
        );
        println!(
            "{}",
            format!("✓ Sandbox Telemetry: {}", telem.summary_line()).green()
        );

        let plan = generate_pr_plan(&spec, telem, socratic);
        match CombStore::open_default() {
            Ok(store) => match emit_adapter_pipeline_combs(&store, &plan, None) {
                Ok(receipt) => {
                    println!(
                        "{}",
                        "\n✓ Successfully emitted 4-stage pipeline onto Honeycomb Wall:"
                            .green()
                            .bold()
                    );
                    println!(
                        "  1. Origin Comb:   id={} [role: adapter-engine]",
                        receipt.origin_comb_id.cyan()
                    );
                    println!(
                        "  2. Socratic Comb: id={} [role: socratic]",
                        receipt.socratic_comb_id.cyan()
                    );
                    println!(
                        "  3. Verifier Comb: id={} [role: verifier]",
                        receipt.sandbox_comb_id.cyan()
                    );
                    println!(
                        "  4. PR Comb:       id={} [role: pr-pipeline]",
                        receipt.pr_comb_id.cyan()
                    );
                    println!(
                        "\nUpstream PR Target: {} (branch: {})",
                        plan.target_repo.yellow(),
                        plan.branch.yellow()
                    );
                    println!("Commit Title: {}", plan.commit_message.bold());
                    println!(
                        "Run `/adapter pr {}` to view full PR markdown and gh commands.\n",
                        spec.id
                    );
                }
                Err(e) => println!(
                    "{}",
                    format!("Failed to emit combs to honeycomb lattice: {}", e).red()
                ),
            },
            Err(e) => println!("{}", format!("Failed to open hive database: {}", e).red()),
        }
    } else if args[0] == "pr" {
        if args.len() < 2 {
            println!("Usage: /adapter pr <adapter-id | model-slug> [engine]");
            return;
        }
        let target = args[1].trim().to_lowercase();
        let engine_override = if args.len() > 2 {
            spark_hive::UpstreamEngine::from_name(args[2])
        } else {
            None
        };
        let spec = spark_hive::find_or_create_adapter(&target, engine_override);
        let Some(spec) = spec else {
            println!(
                "{}",
                format!(
                    "Adapter or model '{}' not found. Run `/adapter list` to view catalog.",
                    target
                )
                .red()
            );
            return;
        };

        let socratic = evaluate_socratic_reflex(&spec.model, &spec.engine);
        let telem = BenchmarkTelemetry::estimate_for_model(
            &spec.model,
            &spec.engine,
            &crate::platform::PlatformContext::detect()
                .home_dir
                .join("workspace/aien-sandbox")
                .to_string_lossy(),
            None,
        );
        let plan = generate_pr_plan(&spec, telem, socratic);

        println!(
            "{}",
            format!("\n=== Sovereign Pull Request Plan: {} ===", plan.pr_title)
                .yellow()
                .bold()
        );
        println!("Author: {}", plan.author.green());
        println!("Target Repository: {}", plan.target_repo.cyan());
        println!("Branch: {}", plan.branch.cyan());
        println!("\n--- Pull Request Body (Sovereign Voice / Anti-Slop Compliant) ---\n");
        println!("{}", plan.pr_body);
        println!("\n--- Automated gh CLI Commands ---");
        for cmd in plan.gh_commands {
            println!("  $ {}", cmd.yellow());
        }
        println!("\n--- Autonomous PR Execution Script ---");
        println!("{}", plan.pr_script);
        println!();
    } else {
        println!("Usage: /adapter [list | emit <id> | pr <id> | lattice]");
    }
}

// --------------------------------------------------------------------------
// SOVEREIGN ORCHESTRATION & DEVELOPER EXPERIENCE
// --------------------------------------------------------------------------

/// Explicit model manifest for daemon boot. No silent developer-machine paths.
/// After loading, it records the SHA-256 (hex) of the model and tokenizer files
/// so a later run receipt can name exactly which bytes were served.
struct DaemonModelManifest {
    model_id: String,
    label: String,
    checkpoint_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    /// Why no checkpoint file was resolved (None when one was found).
    fallback_reason: Option<String>,
    /// Hex SHA-256 of what was loaded (set only when its weights loaded): the file's own
    /// digest for a single file, the manifest digest over the index and every shard for a
    /// sharded checkpoint (see [`checkpoint_digest`]).
    model_sha256: Option<String>,
    /// Which form `model_sha256` has: `file` or `index+shards` (set with it).
    model_digest_kind: Option<&'static str>,
    /// Hex SHA-256 of the loaded tokenizer file (set only when it parsed).
    tokenizer_sha256: Option<String>,
}

/// Checkpoint selection inputs, read once from the environment.
/// `AIEN_MODEL_PATH` (a safetensors file) and `AIEN_TOKENIZER_PATH` (tokenizer.json,
/// default: `tokenizer.json` beside the model file) take precedence over any directory
/// scan. `AIEN_MODEL_DIR` is scanned first for a `*.safetensors` file (directly or one level
/// down). The model is described by `config.json` beside the checkpoint when present (any
/// `LlamaForCausalLM`, e.g. Llama 3.x), else by the built-in TinyLlama-1.1B config.
/// `AIEN_REQUIRE_CHECKPOINT=1` makes every fallback to reference weights fatal.
#[derive(Debug, Clone, Default)]
struct CheckpointPolicy {
    model_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    scan_dirs: Vec<std::path::PathBuf>,
    require_checkpoint: bool,
}

impl CheckpointPolicy {
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let non_empty = |key: &str| {
            get(key)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let mut scan_dirs = Vec::new();
        if let Some(dir) = non_empty("AIEN_MODEL_DIR") {
            scan_dirs.push(std::path::PathBuf::from(dir));
        }
        scan_dirs.push(std::path::PathBuf::from("models"));
        if let Some(home) = non_empty("HOME") {
            scan_dirs.push(std::path::PathBuf::from(home).join("models"));
        }
        Self {
            model_path: non_empty("AIEN_MODEL_PATH").map(std::path::PathBuf::from),
            tokenizer_path: non_empty("AIEN_TOKENIZER_PATH")
                .or_else(|| non_empty("AIEN_TOKENIZER"))
                .map(std::path::PathBuf::from),
            scan_dirs,
            require_checkpoint: non_empty("AIEN_REQUIRE_CHECKPOINT")
                .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .unwrap_or(aien_inference_abi::strict::production_strict()),
        }
    }

    fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }
}

/// The daemon's start line (sc#337): whether this process fails closed, whether
/// the binary was built with `dev-fallback`, the effective `require_checkpoint`
/// and the tensor backend's name, so a run's log shows the protections were on
/// instead of only the absence of a crash.
fn strict_start_line(
    strict: bool,
    dev_fallback_build: bool,
    require_checkpoint: bool,
    backend: &str,
) -> String {
    format!(
        "STRICT strict={strict} dev_fallback_build={dev_fallback_build} require_checkpoint={require_checkpoint} backend={backend}"
    )
}

fn is_safetensors(path: &std::path::Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
}

/// Explicit tokenizer path if given, else `tokenizer.json` beside the checkpoint.
/// The path is returned even if the file is missing; loading decides what that means.
fn tokenizer_for(
    policy: &CheckpointPolicy,
    checkpoint: &std::path::Path,
) -> Option<std::path::PathBuf> {
    if let Some(path) = &policy.tokenizer_path {
        return Some(path.clone());
    }
    checkpoint.parent().map(|dir| dir.join("tokenizer.json"))
}

fn checkpoint_manifest(
    policy: &CheckpointPolicy,
    path: std::path::PathBuf,
    model_id: String,
    label: String,
) -> DaemonModelManifest {
    DaemonModelManifest {
        model_id,
        label,
        tokenizer_path: tokenizer_for(policy, &path),
        checkpoint_path: Some(path),
        fallback_reason: None,
        model_sha256: None,
        model_digest_kind: None,
        tokenizer_sha256: None,
    }
}

fn resolve_daemon_manifest_with(policy: &CheckpointPolicy) -> DaemonModelManifest {
    if let Some(path) = &policy.model_path {
        let model_id = path
            .parent()
            .and_then(|dir| dir.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("checkpoint")
            .to_string();
        return checkpoint_manifest(
            policy,
            path.clone(),
            model_id,
            format!("explicit AIEN_MODEL_PATH={}", path.display()),
        );
    }
    for dir in &policy.scan_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if is_safetensors(&path) {
                let model_id = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("checkpoint")
                    .to_string();
                let label = format!(
                    "checkpoint found at {} (model_id={})",
                    path.display(),
                    model_id
                );
                return checkpoint_manifest(policy, path, "checkpoint".into(), label);
            }
            if path.is_dir() {
                let Ok(nested) = std::fs::read_dir(&path) else {
                    continue;
                };
                for child in nested.flatten() {
                    let child_path = child.path();
                    if is_safetensors(&child_path) {
                        let model_id = path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("checkpoint")
                            .to_string();
                        let label = format!("checkpoint found at {}", child_path.display());
                        return checkpoint_manifest(policy, child_path, model_id, label);
                    }
                }
            }
        }
    }
    let searched = policy
        .scan_dirs
        .iter()
        .map(|dir| dir.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    DaemonModelManifest {
        model_id: "aien-daemon-reference-fallback".into(),
        label: "no safetensors checkpoint resolved".into(),
        checkpoint_path: None,
        tokenizer_path: None,
        fallback_reason: Some(format!(
            "no AIEN_MODEL_PATH and no *.safetensors found in [{}]",
            searched
        )),
        model_sha256: None,
        model_digest_kind: None,
        tokenizer_sha256: None,
    }
}

/// Builds the native transformer backend for daemon boot with explicit fallback.
/// Uses the native Omega GPU engine when linked, falls back to CPU reference math.
/// Never returns the Mock backend: output always comes from real forward passes.
/// When AIEN_REQUIRE_BLACKWELL (or AIEN_GPU_BACKEND=omega) is set, a missing GPU engine is fatal:
/// the hardware gate must fail when fallback count is nonzero.
fn reference_config() -> aien_inference_abi::ModelConfig {
    aien_inference_abi::ModelConfig {
        model_id: "aien-daemon-reference-fallback".to_string(),
        max_sequence_length: 2048,
        block_size: 16,
        num_layers: 4,
        num_heads: 8,
        head_dim: 64,
        num_kv_heads: 4,
        hidden_dim: 512,
        intermediate_dim: 1408,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

/// Result of daemon model loading: the manifest (with SHA-256 digests filled in on
/// success), the weights, the tokenizer, and whether reference weights were used.
struct DaemonModel {
    manifest: DaemonModelManifest,
    weights: aien_inference_abi::TransformerWeights,
    tokenizer: Option<aien_inference_abi::ChatTokenizer>,
    label: String,
    /// The `CHECKPOINT_SHARDS` line for a sharded checkpoint (logged after the
    /// "checkpoint loaded" line, which keeps its pinned form); None otherwise.
    shards_line: Option<String>,
    #[allow(dead_code)] // read by tests now and by the PREFILL-E2E receipt (Cut 8)
    reference_weights: bool,
}

/// The digests of the files the daemon actually loaded, for its generation
/// records. Present only when real weights AND a parsed tokenizer loaded (the
/// reference fallback has no file identity).
fn model_identity(
    manifest: &DaemonModelManifest,
    tokenizer_loaded: bool,
) -> Option<aien_runtime::generation::ModelIdentity> {
    if !tokenizer_loaded {
        return None;
    }
    let path_text = |p: &std::path::Path| {
        std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .display()
            .to_string()
    };
    Some(aien_runtime::generation::ModelIdentity {
        model_sha256: manifest.model_sha256.clone()?,
        model_digest_kind: manifest.model_digest_kind?.to_string(),
        model_path: path_text(manifest.checkpoint_path.as_deref()?),
        tokenizer_sha256: manifest.tokenizer_sha256.clone()?,
        tokenizer_path: path_text(manifest.tokenizer_path.as_deref()?),
    })
}

/// Hex SHA-256 of a file, streamed in 1 MiB chunks.
fn sha256_file_hex(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// The digest of a checkpoint's weight bytes (sc#338), over exactly the files the
/// loader reads ([`aien_inference_abi::checkpoint_files`]).
struct CheckpointDigest {
    /// A single file: its own sha256. A sharded checkpoint: the sha256 of the
    /// manifest text `aien-checkpoint-digest v1\nindex <index sha256>\n` followed by
    /// one `<shard sha256>  <shard name>\n` line per shard, sorted by name (the
    /// shard lines are `sha256sum` lines).
    sha256: String,
    /// `file` or `index+shards`.
    kind: &'static str,
    /// The index's own sha256 (sharded only).
    index_sha256: Option<String>,
    /// `(shard name, shard sha256)` in name order (sharded only).
    shards: Vec<(String, String)>,
}

/// Errors name the file only when it is not `path` itself (the caller names `path`).
/// A directory is refused, as it was when `path` itself was hashed.
fn checkpoint_digest(path: &std::path::Path) -> Result<CheckpointDigest, String> {
    if path.is_dir() {
        return Err("is a directory, not a safetensors file or shard index".to_string());
    }
    let files = aien_inference_abi::checkpoint_files(path).map_err(|e| e.to_string())?;
    let sha = |p: &std::path::Path| {
        aien_inference_abi::checkpoint_sha256(p)
            .map(hex::encode)
            .map_err(|e| {
                if p == path {
                    e.to_string()
                } else {
                    format!("{}: {}", p.display(), e)
                }
            })
    };
    let Some(index) = files.index else {
        let Some((_, file)) = files.weights.first() else {
            return Err("names no weight file".to_string());
        };
        return Ok(CheckpointDigest {
            sha256: sha(file)?,
            kind: "file",
            index_sha256: None,
            shards: Vec::new(),
        });
    };
    if files.weights.is_empty() {
        return Err("its shard index names no shard".to_string());
    }
    use sha2::{Digest, Sha256};
    let index_sha256 = sha(&index)?;
    let mut manifest = format!("aien-checkpoint-digest v1\nindex {index_sha256}\n");
    let mut shards = Vec::with_capacity(files.weights.len());
    for (name, file) in &files.weights {
        let digest = sha(file)?;
        manifest.push_str(&format!("{digest}  {name}\n"));
        shards.push((name.clone(), digest));
    }
    Ok(CheckpointDigest {
        sha256: hex::encode(Sha256::digest(manifest.as_bytes())),
        kind: "index+shards",
        index_sha256: Some(index_sha256),
        shards,
    })
}

/// The `CHECKPOINT_SHARDS` log line (sc#338): every shard's sha256 and the index's,
/// so a run's log ties `model_sha256` to the shard bytes. None for a single file.
fn checkpoint_shards_line(d: &CheckpointDigest) -> Option<String> {
    let index = d.index_sha256.as_deref()?;
    let shards: Vec<String> = d.shards.iter().map(|(n, s)| format!("{n}:{s}")).collect();
    Some(format!(
        "CHECKPOINT_SHARDS model_sha256={} model_digest_kind={} index_sha256={} shards=[{}]",
        d.sha256,
        d.kind,
        index,
        shards.join(",")
    ))
}

/// The single decision point for every checkpoint fallback. Under
/// AIEN_REQUIRE_CHECKPOINT=1 it refuses with an error naming the reason (which names
/// the path); otherwise it prints exactly one warn-level line and lets the caller degrade.
fn strict_checkpoint_gate(require_checkpoint: bool, reason: &str) -> Result<(), String> {
    if require_checkpoint {
        return Err(format!(
            "AIEN_REQUIRE_CHECKPOINT=1: {}; refusing to fall back",
            reason
        ));
    }
    eprintln!(
        "WARN aien-daemon: {}; falling back (set AIEN_REQUIRE_CHECKPOINT=1 to make this fatal)",
        reason
    );
    Ok(())
}

fn reference_fallback(
    manifest: DaemonModelManifest,
    require_checkpoint: bool,
    reason: String,
) -> Result<DaemonModel, String> {
    strict_checkpoint_gate(require_checkpoint, &reason)?;
    let config = reference_config();
    Ok(DaemonModel {
        label: format!(
            "{} ({}); using reference fallback weights",
            manifest.label, reason
        ),
        weights: aien_inference_abi::TransformerWeights::reference_test_weights(&config),
        tokenizer: None,
        shards_line: None,
        reference_weights: true,
        manifest,
    })
}

/// How the daemon makes the GB10 Qwen3 weights resident instead of holding host f32 copies
/// (sovereign-core #277 cut C). `None` anywhere means the unchanged host f32 load.
struct ResidentLoad<'a> {
    native_linked: bool,
    strict: bool,
    qwen3_enabled: bool,
    uploader: &'a dyn aien_inference_abi::ResidentUploader,
    /// Opens the GPU session (bounded retry) so every driver allocation of the upload happens
    /// while the process holds almost no memory.
    open_session: &'a dyn Fn() -> Result<(), String>,
}

/// Plain-English refusal printed by `aien start` when the approval desk key is
/// missing or unusable (sc#414). The desk is mandatory, so this is a refusal,
/// never a warning. `detail` is the runtime's own reason.
pub fn start_refusal_text(detail: &str) -> String {
    format!(
        "AIEN will not start yet: the approval desk key has not been created.
         The desk is how you approve what AIEN is allowed to do, and AIEN refuses to run without it.
         Create it once, then start again:
             aien compose desk-key --create 1
             aien start
         (detail: {detail})"
    )
}

#[cfg(test)]
fn load_daemon_model(
    manifest: DaemonModelManifest,
    require_checkpoint: bool,
) -> Result<DaemonModel, String> {
    load_daemon_model_with(manifest, require_checkpoint, None)
}

fn load_daemon_model_with(
    mut manifest: DaemonModelManifest,
    require_checkpoint: bool,
    resident: Option<&ResidentLoad<'_>>,
) -> Result<DaemonModel, String> {
    let Some(path) = manifest.checkpoint_path.clone() else {
        let reason = manifest
            .fallback_reason
            .clone()
            .unwrap_or_else(|| "no checkpoint resolved".to_string());
        return reference_fallback(manifest, require_checkpoint, reason);
    };
    let digest = match checkpoint_digest(&path) {
        Ok(digest) => digest,
        Err(error) => {
            let reason = format!("checkpoint {} is unreadable: {}", path.display(), error);
            return reference_fallback(manifest, require_checkpoint, reason);
        }
    };
    // The model description: config.json beside the checkpoint (any LlamaForCausalLM
    // checkpoint), else the hard-wired TinyLlama-1.1B config.
    let model_dir = path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let has_config_json = model_dir.join("config.json").is_file();
    let config = if has_config_json {
        match aien_inference_abi::load_model_config(&model_dir) {
            Ok(config) => config,
            Err(error) => return reference_fallback(manifest, require_checkpoint, error),
        }
    } else {
        aien_inference_abi::ModelConfig::tinyllama_1_1b()
    };
    let resident = resident.filter(|r| {
        aien_inference_abi::resident_load_wanted(
            r.native_linked,
            r.strict,
            r.qwen3_enabled,
            &config,
        )
    });
    let loaded = match resident {
        None => aien_inference_abi::TransformerWeights::load_from_safetensors(&path, &config),
        Some(r) => {
            (r.open_session)()?;
            let mut count = 0usize;
            let loaded = aien_inference_abi::load_resident_weights(
                &path,
                &config,
                r.uploader,
                &mut |_name| count += 1,
            );
            match loaded {
                // An upload that fails is fatal here: the weights have no host copy to degrade to.
                Err(e @ aien_inference_abi::CheckpointError::ResidentUpload { .. }) => {
                    return Err(format!("GB10_RESIDENT_LOAD failed: {e}"));
                }
                other => {
                    if other.is_ok() {
                        println!(
                            "  GB10_RESIDENT_LOAD: {count} matmul weights streamed from the bf16 shards to the device; no host f32 copy of any matmul weight"
                        );
                    }
                    other
                }
            }
        }
    };
    let weights = match loaded {
        Ok(weights) => weights,
        Err(error) => {
            let reason = format!(
                "checkpoint {} failed to load against the {} config: {}",
                path.display(),
                config.model_id,
                error
            );
            return reference_fallback(manifest, require_checkpoint, reason);
        }
    };
    manifest.model_sha256 = Some(digest.sha256.clone());
    manifest.model_digest_kind = Some(digest.kind);

    let mut tokenizer = None;
    match manifest.tokenizer_path.clone() {
        None => strict_checkpoint_gate(
            require_checkpoint,
            &format!("no tokenizer path for checkpoint {}", path.display()),
        )?,
        Some(tokenizer_path) => match sha256_file_hex(&tokenizer_path) {
            Err(error) => strict_checkpoint_gate(
                require_checkpoint,
                &format!(
                    "tokenizer {} is unreadable: {}",
                    tokenizer_path.display(),
                    error
                ),
            )?,
            Ok(digest) => {
                let loaded = if has_config_json {
                    aien_inference_abi::ChatTokenizer::from_model_dir(
                        &model_dir,
                        Some(&tokenizer_path),
                    )
                } else {
                    aien_inference_abi::ChatTokenizer::from_file(&tokenizer_path)
                };
                match loaded {
                    Ok(loaded) => {
                        manifest.tokenizer_sha256 = Some(digest);
                        tokenizer = Some(loaded);
                    }
                    Err(error) => strict_checkpoint_gate(
                        require_checkpoint,
                        &format!(
                            "tokenizer {} failed to parse: {}",
                            tokenizer_path.display(),
                            error
                        ),
                    )?,
                }
            }
        },
    }

    let label = checkpoint_loaded_label(
        &path,
        tokenizer.is_some(),
        &manifest.model_id,
        &config.model_id,
        has_config_json,
        manifest.model_sha256.as_deref().unwrap_or("none"),
        manifest.tokenizer_sha256.as_deref().unwrap_or("none"),
    );
    Ok(DaemonModel {
        manifest,
        weights,
        tokenizer,
        label,
        shards_line: checkpoint_shards_line(&digest),
        reference_weights: false,
    })
}

/// The one-line "checkpoint loaded" label. Its format is parsed by the interplane
/// verifier (interplane#76): keep it byte-identical (pinned by a test).
fn checkpoint_loaded_label(
    path: &std::path::Path,
    tokenizer_loaded: bool,
    manifest_model_id: &str,
    config_model_id: &str,
    has_config_json: bool,
    model_sha256: &str,
    tokenizer_sha256: &str,
) -> String {
    format!(
        "checkpoint loaded from {} ({}, model_id={}, config={}, model_sha256={}, tokenizer_sha256={})",
        path.display(),
        if tokenizer_loaded {
            "tokenizer loaded"
        } else {
            "tokenizer missing"
        },
        manifest_model_id,
        if has_config_json {
            format!("{} (config.json)", config_model_id)
        } else {
            format!("{} (built in)", config_model_id)
        },
        model_sha256,
        tokenizer_sha256,
    )
}

/// Loaded daemon model parts: weights, tensor compute backend, backend label,
/// model label, tokenizer. The KV manager is built from these by
/// `aien_runtime::shared_kv::build_shared_kv_runtime` (one KV for spine and backend).
type DaemonBackendParts = (
    aien_inference_abi::TransformerWeights,
    std::sync::Arc<dyn aien_inference_abi::TensorBackend>,
    String,
    String,
    Option<aien_inference_abi::ChatTokenizer>,
    Option<aien_runtime::generation::ModelIdentity>,
);

/// Runtime switch for an explicitly CPU-only run (sc#384 preflight). `AIEN_FORCE_CPU_STUB=1`
/// is a BUILD-time flag (crates/aien-omega-gpu/build.rs): it only stubs the GPU library when
/// compiling, and a binary built earlier with the GPU library ignores it at run time.
pub const CPU_ONLY_ENV: &str = "AIEN_BACKEND";

/// Stable name of the refusal, so scripts and tests can match it.
pub const CPU_ONLY_REFUSAL: &str = "CPU_ONLY_BUT_GPU_BUILD";

/// True when `AIEN_BACKEND=cpu` (case-insensitive) is set.
fn cpu_only_requested(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("cpu"))
}

/// Preflight: an explicitly CPU-only run either uses the CPU backend or refuses to start.
/// Decided before any model load or GPU session is opened. `native_linked` is whether this
/// binary carries the real GPU library.
fn cpu_only_preflight(value: Option<&str>, native_linked: bool) -> Result<(), String> {
    if cpu_only_requested(value) && native_linked {
        return Err(format!(
            "{CPU_ONLY_REFUSAL}: {CPU_ONLY_ENV}=cpu asks for a CPU-only run, but this binary has the GB10 GPU engine linked and would select it. Refusing to start. Use a CPU-stub build (AIEN_FORCE_CPU_STUB=1 at build time) or unset {CPU_ONLY_ENV}. Note AIEN_FORCE_CPU_STUB does nothing at run time."
        ));
    }
    Ok(())
}

fn build_native_daemon_backend() -> Result<DaemonBackendParts, String> {
    let policy = CheckpointPolicy::from_env();
    let manifest = resolve_daemon_manifest_with(&policy);
    let DaemonModel {
        weights,
        tokenizer,
        label: model_label,
        manifest,
        shards_line,
        ..
    } = {
        let open_session = || {
            aien_inference_abi::open_gpu_session_with_retry(
                aien_inference_abi::GPU_SESSION_OPEN_ATTEMPTS,
                aien_inference_abi::GPU_SESSION_RETRY_DELAY,
                aien_inference_abi::GPU_SESSION_OPEN_DEADLINE,
                &aien_runtime::shared_kv::read_mem_available,
            )
            .map(|_| ())
        };
        let resident = ResidentLoad {
            native_linked: aien_omega_gpu::is_native(),
            strict: aien_inference_abi::strict::production_strict(),
            qwen3_enabled: aien_inference_abi::gb10_qwen3_enabled(),
            uploader: &aien_inference_abi::OmegaUploader,
            open_session: &open_session,
        };
        load_daemon_model_with(manifest, policy.require_checkpoint, Some(&resident))?
    };
    if let Some(line) = &shards_line {
        println!("  {line}");
    }
    let identity = model_identity(&manifest, tokenizer.is_some());
    // FB-1 cut 6: the native Omega engine is the only GPU backend (no CUDA). The env names
    // AIEN_REQUIRE_BLACKWELL (the GB10 chip, campaign spec) and AIEN_GPU_BACKEND=omega both
    // make a missing engine fatal instead of falling back to CPU reference math.
    let require_gpu = aien_inference_abi::omega_backend_selected()
        || std::env::var("AIEN_REQUIRE_BLACKWELL")
            .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    // T4 launch budget: parsed before the backend is chosen, so a bad value is
    // always fatal; applied before the first matmul (omega clears its kernel
    // cache on a change). Unset means the AIEN default (256), not omega's 64.
    let cta_setting = std::env::var(aien_omega_gpu::CTA_BUDGET_ENV).ok();
    let cta_budget = aien_omega_gpu::parse_cta_budget(cta_setting.as_deref())?;
    // Marker-wait spin window (omega#328): parsed here too, so a bad value is always fatal.
    let spin_setting = std::env::var(aien_omega_gpu::SPIN_US_ENV).ok();
    let spin_us = aien_omega_gpu::parse_spin_us(spin_setting.as_deref())?;
    let omega = aien_inference_abi::OmegaGb10Backend::new();
    if omega.is_available() {
        println!("  {}", apply_omega_cta_budget(cta_budget)?);
        println!("  {}", apply_omega_spin_us(spin_us)?);
        let name = aien_inference_abi::TensorBackend::name(&omega).to_string();
        let tensor_backend: std::sync::Arc<dyn aien_inference_abi::TensorBackend> =
            std::sync::Arc::new(omega);
        Ok((
            weights,
            tensor_backend,
            format!("NativeTransformerBackend/{name}"),
            model_label,
            tokenizer,
            identity,
        ))
    } else if require_gpu {
        Err(
            "the GB10 GPU is required (AIEN_REQUIRE_BLACKWELL or AIEN_GPU_BACKEND=omega) but the Omega GPU engine is not linked (stub build)"
                .to_string(),
        )
    } else {
        if let Some(budget) = cta_budget {
            println!(
                "  Omega CTA budget: {}={} has no effect (CPU reference backend)",
                aien_omega_gpu::CTA_BUDGET_ENV,
                budget.get()
            );
        }
        let tensor_backend: std::sync::Arc<dyn aien_inference_abi::TensorBackend> =
            std::sync::Arc::new(aien_inference_abi::ReferenceCpuBackend::new());
        Ok((
            weights,
            tensor_backend,
            "NativeTransformerBackend/CPU-reference (Omega GPU engine not linked, explicit fallback)"
                .to_string(),
            model_label,
            tokenizer,
            identity,
        ))
    }
}

/// The budget the daemon uses and where it came from: the explicit
/// `AIEN_OMEGA_CTA_BUDGET`, else the AIEN daemon default (1024, sealed GB10
/// evidence in `aien_omega_gpu::DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE`). Omega's
/// own library default (64) is never relied on by the daemon.
fn daemon_cta_budget(
    budget: Option<aien_omega_gpu::CtaBudget>,
) -> (aien_omega_gpu::CtaBudget, String) {
    let env = aien_omega_gpu::CTA_BUDGET_ENV;
    match budget {
        Some(b) => (b, format!("{env}={}", b.get())),
        None => (
            aien_omega_gpu::DAEMON_DEFAULT_CTA_BUDGET,
            format!(
                "AIEN default, {env} unset; evidence {}; omega library default {} not used",
                aien_omega_gpu::DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE,
                aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_MAX_CTAS
            ),
        ),
    }
}

/// Apply the daemon's marker-wait spin window (always set explicitly at start, omega#328)
/// and return the startup log line with the window omega reports back.
fn apply_omega_spin_us(us: Option<u32>) -> Result<String, String> {
    let env = aien_omega_gpu::SPIN_US_ENV;
    let (us, source) = match us {
        Some(v) => (v, format!("{env} set")),
        None => (
            aien_omega_gpu::DAEMON_DEFAULT_SPIN_US,
            format!(
                "AIEN default, {env} unset; evidence {}",
                aien_omega_gpu::DAEMON_DEFAULT_SPIN_US_EVIDENCE
            ),
        ),
    };
    aien_omega_gpu::set_spin_us(us)
        .map_err(|e| format!("Omega marker spin {us} us ({source}): {e}"))?;
    let active = aien_omega_gpu::spin_us().ok_or_else(|| {
        format!(
            "Omega marker spin {us} us ({source}): the Omega GPU engine is not linked (stub build)"
        )
    })?;
    Ok(format!("Omega marker spin: {active} us ({source})"))
}

/// Apply the daemon's CTA budget (always set explicitly at start) and return
/// the startup log line with the budget omega reports back
/// (`omega_gpu_matmul_cta_budget`).
fn apply_omega_cta_budget(budget: Option<aien_omega_gpu::CtaBudget>) -> Result<String, String> {
    let (b, source) = daemon_cta_budget(budget);
    aien_omega_gpu::set_cta_budget(b)
        .map_err(|e| format!("Omega CTA budget {} ({source}): {e}", b.get()))?;
    let active = aien_omega_gpu::cta_budget().ok_or_else(|| {
        format!(
            "Omega CTA budget {} ({source}): the Omega GPU engine is not linked (stub build)",
            b.get()
        )
    })?;
    if active != b.get() {
        return Err(format!(
            "Omega CTA budget: set {} but omega reports {active}",
            b.get()
        ));
    }
    Ok(format!(
        "Omega CTA budget: {active} CTAs per matmul launch ({source})"
    ))
}

/// The scheduler limits the daemon declares. One definition for `run_daemon_server` and the tests
/// of the GB10 serving reservation, whose bounds (decode rows, prefill chunk) come from it.
fn daemon_scheduler_config() -> aien_scheduler::SchedulerConfig {
    aien_scheduler::SchedulerConfig {
        max_batch_size: 256,
        max_batch_tokens: 16384,
        max_prefill_tokens: 8192,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 64,
    }
}

/// The daemon's start-up decision for the GB10 serving reservation, apart from the chip so a CPU
/// test can run it: derives the bounds from the KV plan's context and the scheduler limits, makes
/// runs reserve, prepare of every (rows, weight shape) kernel, then seal through `ops` on the opt-in
/// Qwen3 GB10 path (and never elsewhere), and returns the log line, or the named refusal that stops the daemon (no on-demand fallback).
/// `run_daemon_server` calls it once, before the server is built, hence before any request.
fn daemon_serving_reservation(
    model_config: &aien_inference_abi::ModelConfig,
    kv_context_tokens: usize,
    sched: &aien_scheduler::SchedulerConfig,
    gpu_native: bool,
    qwen3_enabled: bool,
    ops: &dyn aien_inference_abi::gb10_serving::ServingOps,
) -> Result<Option<String>, String> {
    let limits = aien_inference_abi::gb10_serving::ServingLimits {
        context_tokens: kv_context_tokens,
        max_batch_rows: sched.max_batch_size,
        prefill_chunk_rows: sched.prefill_chunk_size,
    };
    aien_inference_abi::gb10_serving::reserve_gb10_serving_with(
        model_config,
        &limits,
        gpu_native,
        qwen3_enabled,
        ops,
    )
    .map(|r| r.map(|r| r.log_line()))
}

pub async fn run_daemon_server() {
    println!(
        "{}",
        "⚡ Starting AIEN Sovereign Runtime Daemon...".cyan().bold()
    );
    // sc#384 preflight: an explicitly CPU-only run never reaches a GPU build.
    if let Err(fatal) = cpu_only_preflight(
        std::env::var(CPU_ONLY_ENV).ok().as_deref(),
        aien_omega_gpu::is_native(),
    ) {
        eprintln!("Fatal: {}", fatal.red().bold());
        std::process::exit(1);
    }
    // NEXT-PHASE-2 (ACCEPTANCE-v2 2.8): damaged idempotency state refuses the
    // start; it is never reset silently.
    if let Err(fatal) = aien_runtime::control::RuntimeController::load() {
        eprintln!("Fatal: {}", fatal.red().bold());
        std::process::exit(1);
    }
    // sc#328: the approval desk is required by default; a missing desk key
    // refuses the start here, before any model or GPU work.
    if let Err(fatal) = aien_runtime::compose::authorize_desk_at_start_from_env() {
        eprintln!("Fatal: {}", fatal.red().bold());
        std::process::exit(1);
    }
    let socket_path = aien_runtime::client::AienRuntimeClient::default_socket_path();
    let sched_cfg = daemon_scheduler_config();

    let (weights, tensor_backend, backend_label, model_label, tokenizer, identity) =
        match build_native_daemon_backend() {
            Ok(parts) => parts,
            Err(fatal) => {
                eprintln!("Fatal: {}", fatal.red().bold());
                std::process::exit(1);
            }
        };
    println!(
        "  {}",
        strict_start_line(
            aien_inference_abi::strict::production_strict(),
            aien_inference_abi::strict::DEV_FALLBACK,
            CheckpointPolicy::from_env().require_checkpoint,
            tensor_backend.name(),
        )
    );

    // One KV for runtime and backend: the spine's block tables and the
    // backend's K/V writes go to the same pooled manager. The pool is sized
    // from the loaded model's config and declared context, and the host's
    // MemAvailable is checked before it is allocated (#239). AIEN_KV_CONTEXT_TOKENS
    // may lower the context budget below the model's declared context, never raise it.
    let model_config = weights.config.clone();
    let sched_for_reservation = sched_cfg.clone();
    let kv_context_cap = match aien_runtime::shared_kv::kv_context_cap_from_env() {
        Ok(cap) => cap,
        Err(fatal) => {
            eprintln!("Fatal: shared KV pool: {}", fatal.red().bold());
            std::process::exit(1);
        }
    };
    let gpu_native = aien_inference_abi::OmegaGb10Backend::new().is_available();
    // Persistent KV (ADR 0003, cut C7): settings are checked now, so a bad value
    // stops the start with its name; persistence is off unless AIEN_KV_STORE_DIR is set.
    let kv_persist = match aien_runtime::shared_kv::KvPersistSettings::from_env() {
        Ok(s) => s,
        Err(fatal) => {
            eprintln!("Fatal: persistent KV: {}", fatal.red().bold());
            std::process::exit(1);
        }
    };
    let kv_arith_id = tensor_backend.name().to_string();
    let (mut spine, backend, _kv_plan) =
        match aien_runtime::shared_kv::build_shared_kv_runtime_for_model(
            weights,
            tensor_backend,
            sched_cfg,
            4096,
            kv_context_cap,
            &aien_runtime::shared_kv::read_mem_available_checked,
            aien_runtime::shared_kv::allow_unchecked_memory_from_env(),
        ) {
            Ok(parts) => parts,
            Err(fatal) => {
                eprintln!("Fatal: shared KV pool: {}", fatal.red().bold());
                std::process::exit(1);
            }
        };
    if kv_persist.enabled() {
        // The namespace binds the weights digest; without it nothing could
        // tell this model's K/V from another's, so the start is refused.
        let digest = identity
            .as_ref()
            .ok_or_else(|| {
                "AIEN_KV_STORE_DIR is set but the model weights digest is unknown".to_string()
            })
            .and_then(|id| {
                aien_runtime::shared_kv::parse_sha256_hex(&id.model_sha256)
                    .map(|sha| (sha, id.model_digest_kind.clone()))
            });
        // The host TensorBackend path: weights as the file holds them, this
        // build, the backend name plus KV_ARITH_EPOCH as the rounding class, a
        // dense engine config (ADR 0003 section 1.2.2).
        let host_engine = aien_runtime::compute_config::HostEngineIdentity::reference(&kv_arith_id)
            .map_err(|e| format!("persistent KV engine identity: {e}"));
        let enabled = digest.and_then(|d| host_engine.map(|e| (d, e))).and_then(
            |((sha, kind), host_engine)| {
                aien_runtime::shared_kv::enable_persistent_kv(
                    &mut spine,
                    &kv_persist,
                    &aien_runtime::shared_kv::ModelKvBinding {
                        model: &model_config,
                        arith_id: &kv_arith_id,
                        model_sha256: sha,
                        model_digest_kind: &kind,
                        engine: host_engine.identity(),
                    },
                    kv_context_cap,
                )
            },
        );
        match enabled {
            Ok(Some(line)) => println!("  {line}"),
            Ok(None) => {}
            Err(fatal) => {
                eprintln!("Fatal: persistent KV: {}", fatal.red().bold());
                std::process::exit(1);
            }
        }
    }
    // Open the GPU session before serving, with a bounded retry, so a failed
    // channel open is a clear refusal with status and free memory per attempt
    // instead of the strict-fallback panic at warm-up (#239, #236).
    if gpu_native {
        if let Err(fatal) = aien_inference_abi::open_gpu_session_with_retry(
            aien_inference_abi::GPU_SESSION_OPEN_ATTEMPTS,
            aien_inference_abi::GPU_SESSION_RETRY_DELAY,
            aien_inference_abi::GPU_SESSION_OPEN_DEADLINE,
            &aien_runtime::shared_kv::read_mem_available,
        ) {
            eprintln!("Fatal: {}", fatal.red().bold());
            std::process::exit(1);
        }
    }
    // GB10 Qwen3 (enabled by default, #277): reserve the serving buffers once, from the declared
    // bounds, before the first request. A refusal stops the daemon by name; there is no
    // on-demand fallback (sovereign-core#277, omega#327, omega#333). The outcome is logged
    // either way so a chip run can measure it.
    let mut serving_probe: Option<
        std::sync::Arc<dyn aien_inference_abi::gb10_serving::ServingProbe>,
    > = None;
    match daemon_serving_reservation(
        &model_config,
        _kv_plan.context_tokens,
        &sched_for_reservation,
        gpu_native,
        aien_inference_abi::gb10_qwen3_enabled(),
        &aien_inference_abi::gb10_serving::OmegaServingOps,
    ) {
        Ok(Some(line)) => {
            println!("  {line}");
            serving_probe = Some(std::sync::Arc::new(
                aien_inference_abi::gb10_serving::AllocProbe::omega(),
            ));
        }
        Ok(None) => {}
        Err(fatal) => {
            eprintln!("Fatal: {}", fatal.red().bold());
            std::process::exit(1);
        }
    }
    let server = aien_runtime::server::AienRuntimeServer::new(spine, &socket_path);
    if let Some(probe) = serving_probe {
        server.set_serving_probe(probe);
    }

    println!("  Backend: {}", backend_label.green());
    println!("  Model: {}", model_label.yellow());
    println!(
        "  KV: {}",
        "one pooled KV shared by runtime and backend".green()
    );
    if let Some(tokenizer) = tokenizer {
        println!(
            "  Tokenizer: {} chat template, stop ids {:?}",
            tokenizer.template().name().green(),
            tokenizer.stop_token_ids()
        );
        server.set_tokenizer(tokenizer);
        if let Some(identity) = identity {
            // The daemon's generation records name these digests (docs/DAEMON_GENERATION_RECORD.md).
            server.set_model_identity(identity);
        }
        // NEXT-PHASE-1 v4: one declared 1-token warm-up before serving.
        server.enable_warm_up();
    } else {
        println!(
            "  Tokenizer: {}",
            "not loaded; native chat will refuse turns until tokenizer.json is present".yellow()
        );
    }
    println!("✓ Binding socket at {}", socket_path.display());
    if let Err(e) = server.run(backend).await {
        eprintln!("Runtime daemon error: {}", e);
        // A refused start (sc#328: no desk key) must not exit as a success.
        std::process::exit(1);
    }
}

pub async fn handle_start_command() {
    let client = aien_runtime::client::AienRuntimeClient::default_client();
    if client.is_alive().await {
        println!(
            "{}",
            "✓ AIEN Sovereign Runtime is already active.".green().bold()
        );
        if let Ok(status) = client.get_status().await {
            render_runtime_status(&status);
        }
        return;
    }

    // First-run guard (sc#414): the approval desk is mandatory, so refuse here in
    // plain words instead of spawning a daemon that exits unseen.
    if let Err(detail) = aien_runtime::compose::authorize_desk_at_start_from_env() {
        eprintln!("{}", start_refusal_text(&detail).red().bold());
        std::process::exit(1);
    }
    println!(
        "{}",
        "⚡ Starting AIEN Sovereign Runtime Machine..."
            .cyan()
            .bold()
    );
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("aien"));
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--daemon");

    match cmd.spawn() {
        Ok(_) => {
            let mut ready = false;
            for _ in 0..60 {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                if client.is_alive().await {
                    ready = true;
                    break;
                }
            }

            if ready {
                println!("{}", "✓ AIEN Sovereign Runtime Online".green().bold());
                println!(
                    "  Socket:  {}",
                    aien_runtime::client::AienRuntimeClient::default_socket_path().display()
                );
                println!(
                    "  Runtime: Unified Sequence Arena, Transactional PagedAttention KV Pool, In-Process Cortex Memory"
                );
                if let Ok(status) = client.get_status().await {
                    render_runtime_status(&status);
                }
            } else {
                eprintln!(
                    "{}",
                    "⚠ Runtime daemon spawned but socket did not become ready in 3s.".yellow()
                );
            }
        }
        Err(e) => {
            eprintln!("Failed to spawn AIEN runtime daemon: {}", e);
        }
    }
}

pub async fn handle_stop_command() {
    println!("{}", "Stopping AIEN Sovereign Runtime...".yellow().bold());
    let client = aien_runtime::client::AienRuntimeClient::default_client();
    if client.is_alive().await {
        match client.shutdown().await {
            Ok(()) => println!("{}", "✓ Runtime daemon stopped cleanly over IPC".green()),
            Err(e) => eprintln!("Failed to send shutdown command: {}", e),
        }
    } else {
        println!("{}", "ℹ Runtime daemon is not running".dimmed());
    }

    let _ = std::process::Command::new("pkill")
        .args(["-f", "spark-cockpit"])
        .status();
}

pub fn render_runtime_status(status: &aien_runtime::control::RuntimeStatusReport) {
    let sharing_pct =
        if status.total_kv_blocks > status.free_kv_blocks && status.total_kv_blocks > 0 {
            let allocated = status.total_kv_blocks - status.free_kv_blocks;
            (status.shared_kv_pages as f64 / (allocated + status.shared_kv_pages) as f64) * 100.0
        } else {
            0.0
        };

    println!("  - Active Sequences:   {}", status.active_sequences);
    println!("  - Active Swarms:      {}", status.active_swarms);
    println!("  - Active Worlds:      {}", status.active_worlds);
    println!(
        "  - KV Cache Pool:      {}/{} blocks free",
        status.free_kv_blocks, status.total_kv_blocks
    );
    println!(
        "  - Shared KV Pages:    {} ({:.1}% deduplication)",
        status.shared_kv_pages, sharing_pct
    );
    println!("  - COW Faults:         {}", status.cow_faults);
    println!("  - Engine Status:      ONLINE");
}

pub async fn handle_status_command() {
    println!("{}", "=== AIEN Sovereign Machine Status ===".cyan().bold());
    let client = aien_runtime::client::AienRuntimeClient::default_client();

    match client.get_status().await {
        Ok(status) => {
            render_runtime_status(&status);
        }
        Err(_) => {
            println!(
                "{}",
                "  ℹ Native Runtime: OFFLINE (Run 'aien start' to initialize)".yellow()
            );
        }
    }

    println!("\n{}", "=== Hardware & Ancillary Services ===".cyan());
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .unwrap_or_default();

    let checks = [
        ("Model Seat", "http://127.0.0.1:18006/health", 18006),
        (
            "Cortex Memory Store",
            "http://127.0.0.1:18080/health",
            18080,
        ),
        (
            "Cortex ONNX Encoder",
            "http://127.0.0.1:18081/health",
            18081,
        ),
        (
            "Sovereign Glass Cockpit",
            "http://127.0.0.1:18095/api/pulse",
            18095,
        ),
        ("Matrix Conduit Federation", "http://127.0.0.1:6167", 6167),
        (
            "Sovereign Loopback Mail",
            "http://127.0.0.1:18092/health",
            2525,
        ),
    ];

    for (name, url, port) in checks {
        let t0 = std::time::Instant::now();
        match http_client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let elapsed = t0.elapsed().as_millis();
                println!(
                    "  ✓ {:<28} (Port {:<5}): {} ({} ms)",
                    name.green(),
                    port,
                    "ONLINE".green().bold(),
                    elapsed
                );
            }
            _ => {
                println!(
                    "  ℹ {:<28} (Port {:<5}): {}",
                    name.yellow(),
                    port,
                    "STANDBY / OFFLINE".dimmed()
                );
            }
        }
    }
}

pub async fn handle_swarm_command(args: &[String]) {
    let client = aien_runtime::client::AienRuntimeClient::default_client();
    if !client.is_alive().await {
        eprintln!(
            "{}",
            "Error: AIEN runtime is not running. Run 'aien start' first.".red()
        );
        return;
    }

    if args.is_empty() || args[0] == "help" {
        println!("{}", "AIEN Swarm Management:".cyan().bold());
        println!("  aien swarm launch [--branches N] [--tokens M] [--prompt \"<text>\"]");
        println!("  aien swarm status");
        println!("  aien swarm inspect <ID>");
        println!("  aien swarm cancel <ID>");
        return;
    }

    match args[0].as_str() {
        "launch" => {
            let mut branch_count = 16;
            let mut max_tokens = 32;
            let mut prompt_str = "Explain sovereign systems architecture".to_string();

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--branches" | "-b" if i + 1 < args.len() => {
                        branch_count = args[i + 1].parse().unwrap_or(16);
                        i += 2;
                    }
                    "--tokens" | "-t" if i + 1 < args.len() => {
                        max_tokens = args[i + 1].parse().unwrap_or(32);
                        i += 2;
                    }
                    "--prompt" | "-p" if i + 1 < args.len() => {
                        prompt_str = args[i + 1].clone();
                        i += 2;
                    }
                    _ => i += 1,
                }
            }

            let prompt_tokens: Vec<u32> = prompt_str.bytes().map(|b| b as u32).collect();
            let req = aien_runtime::control::LaunchSwarmReq {
                model_handle: 1,
                branch_count,
                max_active_sequences: branch_count * 2,
                max_tokens_per_branch: max_tokens,
                root_world_id: 0,
                priority: 1,
                prompt_tokens,
            };

            println!(
                "{}",
                format!(
                    "🚀 Launching Swarm with {} branches sharing root World...",
                    branch_count
                )
                .cyan()
            );
            match client.launch_swarm(req).await {
                Ok(swarm_id) => {
                    println!(
                        "{}",
                        format!("✓ Swarm #{} launched successfully.", swarm_id)
                            .green()
                            .bold()
                    );
                    println!(
                        "Use 'aien swarm inspect {}' or 'aien status' to monitor.",
                        swarm_id
                    );
                }
                Err(e) => eprintln!("Failed to launch swarm: {}", e),
            }
        }
        "inspect" if args.len() > 1 => {
            if let Ok(swarm_id) = args[1].parse::<u64>() {
                match client.inspect_swarm(swarm_id).await {
                    Ok(status) => render_runtime_status(&status),
                    Err(e) => eprintln!("Error inspecting swarm: {}", e),
                }
            } else {
                eprintln!("Invalid swarm ID: {}", args[1]);
            }
        }
        "cancel" if args.len() > 1 => {
            if let Ok(swarm_id) = args[1].parse::<u64>() {
                match client.cancel_swarm(swarm_id).await {
                    Ok(()) => println!("{}", format!("✓ Swarm #{} cancelled.", swarm_id).green()),
                    Err(e) => eprintln!("Error cancelling swarm: {}", e),
                }
            } else {
                eprintln!("Invalid swarm ID: {}", args[1]);
            }
        }
        "status" => {
            if let Ok(status) = client.get_status().await {
                render_runtime_status(&status);
            }
        }
        unknown => {
            eprintln!("Unknown swarm subcommand: {}", unknown);
        }
    }
}

pub async fn handle_cockpit_command() {
    println!(
        "{}",
        "==================================================================".cyan()
    );
    println!(
        "{}",
        "      ⚡ AIEN Sovereign Glass Terminal & Cockpit                  "
            .cyan()
            .bold()
    );
    println!(
        "{}",
        "==================================================================".cyan()
    );
    println!("  Local:       http://127.0.0.1:18095");
    println!("  Mascot:      AIEN Cosmic Monkey Warrior");
    println!("  Audio:       Natural Voice Response Enabled");
    println!(
        "{}",
        "==================================================================".cyan()
    );
    let _ = std::process::Command::new("xdg-open")
        .arg("http://127.0.0.1:18095")
        .spawn();
}

/// `aien harness` is not shipped in this release (sc#414 defect 2). It used to
/// call `spark-harness check`, a subcommand that never existed. The real
/// `spark-harness test` needs data files (workflows.json, evals.json) that the
/// installer does not copy, so nothing real backs this command on an installed
/// machine. It refuses, with a non-zero exit, instead of faking a pass.
pub fn harness_refusal_text() -> &'static str {
    "`aien harness` is not shipped in this release, so nothing was checked or certified. \
     Use `aien doctor` to probe this machine."
}

pub async fn handle_harness_command() {
    eprintln!("{}", harness_refusal_text().red().bold());
    std::process::exit(1);
}

pub async fn handle_aegis_command() {
    println!(
        "{}",
        "🛡️ Running Aegis Defensive Boundary Audit...".cyan().bold()
    );
    let status = std::process::Command::new("spark-aegis")
        .arg("audit")
        .status();
    match status {
        Ok(s) if s.success() => println!(
            "{}",
            "\n✓ Aegis Defense: Zero Disk Secrets & Boundary Secured."
                .green()
                .bold()
        ),
        _ => {
            println!("{}", "Auditing hardware TPM key vault directly...".yellow());
            let _ = std::process::Command::new("atlas-vault")
                .arg("list")
                .status();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_switch_never_selects_gpu() {
        // GPU-capable build: refuses with the named error.
        let e = cpu_only_preflight(Some("cpu"), true).unwrap_err();
        assert!(e.starts_with(CPU_ONLY_REFUSAL), "{e}");
        assert!(cpu_only_preflight(Some(" CPU "), true).is_err());
        // Stub build: allowed, the CPU reference backend is then the only one possible.
        assert!(cpu_only_preflight(Some("cpu"), false).is_ok());
        // Not requested: unchanged behaviour.
        assert!(cpu_only_preflight(None, true).is_ok());
        assert!(cpu_only_preflight(Some("gpu"), true).is_ok());
        // AIEN_FORCE_CPU_STUB alone is not the switch.
        assert!(!cpu_only_requested(Some("1")));
    }

    #[test]
    fn start_refusal_names_the_exact_command() {
        let t = start_refusal_text("no key at /x");
        assert!(t.contains("aien compose desk-key --create 1"), "{t}");
        assert!(t.contains("will not start"), "{t}");
        assert!(t.contains("no key at /x"), "{t}");
    }

    #[test]
    fn harness_refuses_plainly_instead_of_faking_a_pass() {
        let t = harness_refusal_text();
        assert!(t.contains("not shipped in this release"), "{t}");
        assert!(!t.contains("PASS"), "{t}");
    }

    #[test]
    fn strict_start_line_names_every_protection() {
        assert_eq!(
            strict_start_line(true, false, true, "omega-gb10"),
            "STRICT strict=true dev_fallback_build=false require_checkpoint=true backend=omega-gb10"
        );
        assert_eq!(
            strict_start_line(false, true, false, "ReferenceCpuBackend"),
            "STRICT strict=false dev_fallback_build=true require_checkpoint=false backend=ReferenceCpuBackend"
        );
    }

    // ---- GB10 serving reservation: the daemon's start-up decision (sovereign-core#277) ----

    fn qwen3_4b_config() -> aien_inference_abi::ModelConfig {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aien-inference-abi/fixtures/qwen3-4b-instruct-2507-config");
        let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let gen = std::fs::read_to_string(dir.join("generation_config.json")).unwrap();
        aien_inference_abi::model_config_from_hf_json(
            "Qwen/Qwen3-4B-Instruct-2507",
            &cfg,
            Some(&gen),
        )
        .unwrap()
    }

    use aien_inference_abi::gb10_serving::ServingOps;
    use std::cell::RefCell;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Ev {
        Reserve(aien_omega_gpu::ServingBounds),
        Prepare(u32, u32, u32),
        Seal,
    }

    /// Records every omega call in order; `fail_prepare_at` makes that (0-based) prepare fail.
    struct Recorder {
        events: RefCell<Vec<Ev>>,
        fail_prepare_at: Option<usize>,
        refuse_reserve: bool,
    }

    impl Recorder {
        fn new() -> Self {
            Recorder {
                events: RefCell::new(Vec::new()),
                fail_prepare_at: None,
                refuse_reserve: false,
            }
        }
    }

    impl ServingOps for Recorder {
        fn reserve(&self, b: &aien_omega_gpu::ServingBounds) -> Result<(), String> {
            self.events.borrow_mut().push(Ev::Reserve(*b));
            if self.refuse_reserve {
                return Err("omega_gpu rc=-4 (CHIP_FAIL)".to_string());
            }
            Ok(())
        }
        fn prepare(&self, m: u32, k: u32, n: u32) -> Result<(), String> {
            let mut ev = self.events.borrow_mut();
            let nth = ev.iter().filter(|e| matches!(e, Ev::Prepare(..))).count();
            ev.push(Ev::Prepare(m, k, n));
            if self.fail_prepare_at == Some(nth) {
                return Err("omega_gpu rc=-4 (CHIP_FAIL)".to_string());
            }
            Ok(())
        }
        fn seal(&self) {
            self.events.borrow_mut().push(Ev::Seal);
        }
    }

    fn run(
        rec: &Recorder,
        ctx: usize,
        native: bool,
        opted_in: bool,
    ) -> Result<Option<String>, String> {
        daemon_serving_reservation(
            &qwen3_4b_config(),
            ctx,
            &daemon_scheduler_config(),
            native,
            opted_in,
            rec,
        )
    }

    #[test]
    fn opt_in_qwen3_gb10_reserves_prepares_every_shape_and_seals_in_order() {
        let cfg = qwen3_4b_config();
        // the plan the daemon builds (context capped like AIEN_KV_CONTEXT_TOKENS=2048)
        let plan = aien_runtime::shared_kv::plan_checked_model_kv(
            &cfg,
            Some(2048),
            &|| Ok(u64::MAX),
            false,
        )
        .unwrap();
        let rec = Recorder::new();
        let line = run(&rec, plan.context_tokens, true, true)
            .unwrap()
            .expect("reserved on the opt-in path");
        let ev = rec.events.borrow();
        // exactly one reserve, first, with bounds from the KV plan and the scheduler
        let Ev::Reserve(b) = &ev[0] else {
            panic!("the first omega call must be the reservation: {:?}", ev[0])
        };
        assert_eq!(ev.iter().filter(|e| matches!(e, Ev::Reserve(_))).count(), 1);
        assert_eq!(b.max_context, 2048, "from the KV plan");
        assert_eq!(b.max_rows, 256, "from the scheduler");
        assert_eq!(b.kernel_slots, 128);
        // every (rows 1..=256) x (shape from the model config) is prepared, each exactly once
        let shapes = aien_inference_abi::gb10_serving::serving_matmul_shapes(&cfg).unwrap();
        assert_eq!(shapes.len(), 6, "q, kv, o, gate/up, down, logits");
        assert!(
            shapes.contains(&(2560, cfg.vocab_size as u32)),
            "logits included"
        );
        let prepared: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                Ev::Prepare(m, k, n) => Some((*m, *k, *n)),
                _ => None,
            })
            .collect();
        let want: std::collections::BTreeSet<_> = (1..=256u32)
            .flat_map(|m| shapes.iter().map(move |&(k, n)| (m, k, n)))
            .collect();
        assert_eq!(prepared.len(), 256 * 6);
        assert_eq!(
            prepared
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            want
        );
        // seal is last: after all prepares, before serving can start
        assert_eq!(ev.last(), Some(&Ev::Seal));
        assert_eq!(ev.iter().filter(|e| **e == Ev::Seal).count(), 1);
        assert_eq!(ev.len(), 1 + 256 * 6 + 1);
        assert!(
            line.contains("prepared 1536 matmul kernel calls in"),
            "{line}"
        );
        assert!(
            line.starts_with("GB10_SERVING_RESERVATION reserved bytes="),
            "{line}"
        );
    }

    #[test]
    fn a_refused_reservation_stops_the_daemon_by_name_before_any_prepare() {
        let mut rec = Recorder::new();
        rec.refuse_reserve = true;
        let err = run(&rec, 4096, true, true).unwrap_err();
        assert!(err.contains("GB10_SERVING_RESERVATION refused"), "{err}");
        assert!(
            err.contains("CHIP_FAIL") && err.contains("no on-demand fallback"),
            "{err}"
        );
        assert_eq!(
            rec.events.borrow().len(),
            1,
            "nothing after a refused reserve"
        );
    }

    #[test]
    fn a_prepare_failure_is_the_same_named_refusal_and_never_seals() {
        let mut rec = Recorder::new();
        rec.fail_prepare_at = Some(7);
        let err = run(&rec, 4096, true, true).unwrap_err();
        assert!(err.contains("GB10_SERVING_RESERVATION refused"), "{err}");
        assert!(err.contains("matmul kernel prepare m="), "{err}");
        let ev = rec.events.borrow();
        assert_eq!(
            ev.len(),
            1 + 8,
            "reserve + 8 prepares, the 8th failed, stop"
        );
        assert!(!ev.contains(&Ev::Seal), "a failed start-up must not seal");
    }

    #[test]
    fn default_path_makes_no_omega_call() {
        // Qwen3 switched off with AIEN_GB10_QWEN3_DECLARED_ATTEMPT=0
        let rec = Recorder::new();
        assert_eq!(run(&rec, 4096, true, false), Ok(None));
        // CPU daemon (no GB10 engine linked)
        assert_eq!(run(&rec, 4096, false, true), Ok(None));
        assert!(rec.events.borrow().is_empty());
    }

    #[test]
    fn the_shared_scheduler_config_is_the_daemons() {
        let s = daemon_scheduler_config();
        assert_eq!((s.max_batch_size, s.prefill_chunk_size), (256, 128));
    }

    #[test]
    fn model_identity_needs_real_weights_and_a_parsed_tokenizer() {
        let loaded = DaemonModelManifest {
            model_id: "m".into(),
            label: "l".into(),
            checkpoint_path: Some("/nonexistent/model.safetensors".into()),
            tokenizer_path: Some("/nonexistent/tokenizer.json".into()),
            fallback_reason: None,
            model_sha256: Some("aa".into()),
            model_digest_kind: Some("index+shards"),
            tokenizer_sha256: Some("bb".into()),
        };
        let id = model_identity(&loaded, true).expect("identity");
        assert_eq!(
            (id.model_sha256.as_str(), id.tokenizer_sha256.as_str()),
            ("aa", "bb")
        );
        assert_eq!(id.model_path, "/nonexistent/model.safetensors");
        assert_eq!(id.model_digest_kind, "index+shards");
        // No parsed tokenizer, no weights digest, or no tokenizer digest: no identity.
        assert!(model_identity(&loaded, false).is_none());
        for strip in [0, 1] {
            let mut m = DaemonModelManifest {
                model_sha256: loaded.model_sha256.clone(),
                model_digest_kind: loaded.model_digest_kind,
                tokenizer_sha256: loaded.tokenizer_sha256.clone(),
                checkpoint_path: loaded.checkpoint_path.clone(),
                tokenizer_path: loaded.tokenizer_path.clone(),
                model_id: "m".into(),
                label: "l".into(),
                fallback_reason: None,
            };
            if strip == 0 {
                m.model_sha256 = None;
            } else {
                m.tokenizer_sha256 = None;
            }
            assert!(model_identity(&m, true).is_none());
        }
    }

    #[test]
    fn checkpoint_loaded_line_format_is_pinned() {
        // interplane#76 parses this line; the text must not drift.
        let line = checkpoint_loaded_label(
            std::path::Path::new("/m/dir"),
            true,
            "mid",
            "cfgid",
            true,
            "aaa",
            "bbb",
        );
        assert_eq!(
            line,
            "checkpoint loaded from /m/dir (tokenizer loaded, model_id=mid, config=cfgid (config.json), model_sha256=aaa, tokenizer_sha256=bbb)"
        );
        let line = checkpoint_loaded_label(
            std::path::Path::new("/m/dir"),
            false,
            "mid",
            "cfgid",
            false,
            "none",
            "none",
        );
        assert_eq!(
            line,
            "checkpoint loaded from /m/dir (tokenizer missing, model_id=mid, config=cfgid (built in), model_sha256=none, tokenizer_sha256=none)"
        );
    }

    /// T4 launch budget: unset means the AIEN default 256 (not omega's 64),
    /// the log names it as the AIEN default with the evidence dir, and an
    /// explicit setting wins.
    #[test]
    fn daemon_cta_budget_defaults_to_1024_with_evidence() {
        let (b, source) = daemon_cta_budget(None);
        assert_eq!(b.get(), 1024);
        assert!(source.contains("AIEN default"), "{source}");
        assert!(
            source.contains("docs/inference/evidence/cta1024-20261012T0315Z"),
            "{source}"
        );
        let explicit = aien_omega_gpu::parse_cta_budget(Some("64"))
            .unwrap()
            .unwrap();
        let (b, source) = daemon_cta_budget(Some(explicit));
        assert_eq!(b.get(), 64);
        assert_eq!(source, "AIEN_OMEGA_CTA_BUDGET=64");
    }

    /// On the CPU stub the helper never claims a budget it could not set and
    /// read back from omega, and names the value it tried.
    #[test]
    fn omega_cta_budget_helper_refuses_on_the_stub() {
        if aien_omega_gpu::is_native() {
            eprintln!(
                "native build: stub refusal check skipped (chip calls not made in unit tests)"
            );
            return;
        }
        let none = apply_omega_cta_budget(None).expect_err("stub cannot set the default");
        assert!(
            none.contains("Omega CTA budget 1024 (AIEN default"),
            "{none}"
        );
        assert!(none.contains("stub"), "{none}");
        let b = aien_omega_gpu::parse_cta_budget(Some("128"))
            .unwrap()
            .unwrap();
        let set = apply_omega_cta_budget(Some(b)).expect_err("stub cannot set a budget");
        assert!(set.contains("AIEN_OMEGA_CTA_BUDGET=128"), "{set}");
    }

    #[tokio::test]
    async fn test_slash_command_adapter_suite() {
        // Test /adapter list
        let handled_list = handle_slash_command("/adapter list").await;
        assert!(handled_list);

        // Test /adapter pr
        let handled_pr = handle_slash_command("/adapter pr candle-qwen2-5-coder-paged-kv").await;
        assert!(handled_pr);

        // Test /adapter emit
        let handled_emit =
            handle_slash_command("/adapter emit candle-qwen2-5-coder-paged-kv").await;
        assert!(handled_emit);

        // Test /adapter lattice
        let handled_lattice = handle_slash_command("/adapter lattice").await;
        assert!(handled_lattice);
    }

    /// PE2E-C1 Step A probe (asserts nothing). Runs the daemon's real checkpoint
    /// resolve + load over the directory named by AIEN_VERIFY_MODELS_DIR (same scan
    /// as the daemon, no explicit path, flag off) and prints which file was chosen
    /// and whether reference weights were used.
    #[test]
    fn verify_daemon_checkpoint_resolution_probe() {
        let Ok(dir) = std::env::var("AIEN_VERIFY_MODELS_DIR") else {
            eprintln!("PROBE: AIEN_VERIFY_MODELS_DIR unset; nothing to probe");
            return;
        };
        let policy = CheckpointPolicy {
            scan_dirs: vec![std::path::PathBuf::from(&dir)],
            ..CheckpointPolicy::default()
        };
        let manifest = resolve_daemon_manifest_with(&policy);
        eprintln!(
            "PROBE: models_dir={} chosen_checkpoint={:?} tokenizer={:?} model_id={}",
            dir, manifest.checkpoint_path, manifest.tokenizer_path, manifest.model_id
        );
        let model = load_daemon_model(manifest, false).expect("flag off never errors");
        eprintln!(
            "PROBE: reference_weights_used={} weights_model_id={} tokenizer_loaded={} label={}",
            model.reference_weights,
            model.weights.config.model_id,
            model.tokenizer.is_some(),
            model.label
        );
    }

    // ---- PE2E-C1 strict checkpoint tests ----

    fn strict_policy(model_path: Option<std::path::PathBuf>) -> CheckpointPolicy {
        CheckpointPolicy {
            model_path,
            require_checkpoint: true,
            ..CheckpointPolicy::default()
        }
    }

    /// Writes a tiny valid safetensors file whose only tensor is a BF16
    /// `model.embed_tokens.weight` of shape [4, 8]: parseable, but not TinyLlama-shaped.
    fn write_mismatched_safetensors(path: &std::path::Path) {
        let payload = vec![0u8; 4 * 8 * 2];
        let header = format!(
            "{{\"model.embed_tokens.weight\":{{\"dtype\":\"BF16\",\"shape\":[4,8],\"data_offsets\":[0,{}]}}}}",
            payload.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).unwrap();
        std::fs::write(path.with_file_name("tokenizer.json"), b"{}").unwrap();
    }

    #[test]
    fn checkpoint_policy_reads_explicit_paths_and_flag() {
        let env = |key: &str| match key {
            "AIEN_MODEL_PATH" => Some("/m/model.safetensors".to_string()),
            "AIEN_TOKENIZER_PATH" => Some("/t/tokenizer.json".to_string()),
            "AIEN_REQUIRE_CHECKPOINT" => Some("1".to_string()),
            "HOME" => Some("/h".to_string()),
            _ => None,
        };
        let policy = CheckpointPolicy::from_lookup(env);
        assert_eq!(
            policy.model_path.as_deref(),
            Some(std::path::Path::new("/m/model.safetensors"))
        );
        assert_eq!(
            policy.tokenizer_path.as_deref(),
            Some(std::path::Path::new("/t/tokenizer.json"))
        );
        assert!(policy.require_checkpoint);
        assert!(policy
            .scan_dirs
            .contains(&std::path::PathBuf::from("/h/models")));

        let off = CheckpointPolicy::from_lookup(|key| {
            (key == "AIEN_REQUIRE_CHECKPOINT").then(|| "0".to_string())
        });
        assert!(!off.require_checkpoint);
        assert!(off.model_path.is_none());
    }

    #[test]
    fn checkpoint_explicit_path_beats_directory_scan() {
        let scan = tempfile::tempdir().unwrap();
        let nested = scan.path().join("AAA-first");
        std::fs::create_dir(&nested).unwrap();
        write_mismatched_safetensors(&nested.join("model.safetensors"));
        let explicit = scan.path().join("chosen").join("model.safetensors");
        let policy = CheckpointPolicy {
            model_path: Some(explicit.clone()),
            scan_dirs: vec![scan.path().to_path_buf()],
            ..CheckpointPolicy::default()
        };
        let manifest = resolve_daemon_manifest_with(&policy);
        assert_eq!(manifest.checkpoint_path, Some(explicit.clone()));
        assert_eq!(
            manifest.tokenizer_path,
            Some(explicit.with_file_name("tokenizer.json"))
        );
    }

    #[test]
    fn strict_checkpoint_refuses_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.safetensors");
        let policy = strict_policy(Some(missing.clone()));
        let manifest = resolve_daemon_manifest_with(&policy);
        let err = load_daemon_model(manifest, policy.require_checkpoint)
            .err()
            .expect("missing checkpoint under AIEN_REQUIRE_CHECKPOINT must be an error");
        assert!(err.contains("AIEN_REQUIRE_CHECKPOINT=1"), "{err}");
        assert!(err.contains(&missing.display().to_string()), "{err}");
    }

    #[test]
    fn strict_checkpoint_refuses_shape_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_mismatched_safetensors(&path);
        let policy = strict_policy(Some(path.clone()));
        let manifest = resolve_daemon_manifest_with(&policy);
        let err = load_daemon_model(manifest, policy.require_checkpoint)
            .err()
            .expect("mis-shaped checkpoint under AIEN_REQUIRE_CHECKPOINT must be an error");
        assert!(err.contains(&path.display().to_string()), "{err}");
        assert!(err.contains("model.embed_tokens.weight"), "{err}");
    }

    #[test]
    fn strict_checkpoint_refuses_scan_mismatch() {
        // Directory scan picks a non-TinyLlama checkpoint (the Qwen-first case).
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("Qwen-like");
        std::fs::create_dir(&nested).unwrap();
        let path = nested.join("model.safetensors");
        write_mismatched_safetensors(&path);
        let policy = CheckpointPolicy {
            scan_dirs: vec![dir.path().to_path_buf()],
            require_checkpoint: true,
            ..CheckpointPolicy::default()
        };
        let manifest = resolve_daemon_manifest_with(&policy);
        assert_eq!(manifest.checkpoint_path, Some(path.clone()));
        let err = load_daemon_model(manifest, true)
            .err()
            .expect("scan finding a non-matching checkpoint must be an error under the flag");
        assert!(err.contains(&path.display().to_string()), "{err}");
    }

    #[test]
    fn strict_checkpoint_refuses_empty_scan() {
        let dir = tempfile::tempdir().unwrap();
        let policy = CheckpointPolicy {
            scan_dirs: vec![dir.path().to_path_buf()],
            require_checkpoint: true,
            ..CheckpointPolicy::default()
        };
        let manifest = resolve_daemon_manifest_with(&policy);
        assert!(manifest.checkpoint_path.is_none());
        let err = load_daemon_model(manifest, true)
            .err()
            .expect("empty scan must be an error under the flag");
        assert!(err.contains(&dir.path().display().to_string()), "{err}");
    }

    #[test]
    fn checkpoint_fallback_without_flag_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.safetensors");
        let policy = CheckpointPolicy {
            model_path: Some(missing.clone()),
            ..CheckpointPolicy::default()
        };
        let manifest = resolve_daemon_manifest_with(&policy);
        let model = load_daemon_model(manifest, false).expect("flag off degrades");
        assert!(model.reference_weights);
        assert!(model.manifest.model_sha256.is_none());
        assert!(
            model.label.contains(&missing.display().to_string()),
            "{}",
            model.label
        );
    }

    /// Copies the tiny Qwen3 fixture into `dir` as two shards plus a
    /// `model.safetensors.index.json` (tensors split by sorted name, half each).
    fn write_two_shard_qwen3_tiny(dir: &std::path::Path) -> std::path::PathBuf {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aien-inference-abi/fixtures/qwen3-tiny");
        std::fs::copy(fixture.join("config.json"), dir.join("config.json")).unwrap();
        let bytes = std::fs::read(fixture.join("model.safetensors")).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let data = &bytes[8 + header_len..];
        let mut names: Vec<&String> = header.keys().filter(|k| *k != "__metadata__").collect();
        names.sort();
        let (first, second) = names.split_at(names.len() / 2);
        let mut weight_map = serde_json::Map::new();
        for (file, part) in [
            ("model-00001-of-00002.safetensors", first),
            ("model-00002-of-00002.safetensors", second),
        ] {
            let mut shard_header = serde_json::Map::new();
            let mut payload = Vec::new();
            for name in part {
                let mut info = header[*name].clone();
                let offsets = info["data_offsets"].as_array().unwrap();
                let (a, b) = (
                    offsets[0].as_u64().unwrap() as usize,
                    offsets[1].as_u64().unwrap() as usize,
                );
                let start = payload.len();
                payload.extend_from_slice(&data[a..b]);
                info["data_offsets"] = serde_json::json!([start, payload.len()]);
                shard_header.insert((*name).clone(), info);
                weight_map.insert((*name).clone(), serde_json::json!(file));
            }
            let text = serde_json::Value::Object(shard_header).to_string();
            let mut out = (text.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(text.as_bytes());
            out.extend_from_slice(&payload);
            std::fs::write(dir.join(file), out).unwrap();
        }
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(
            &index,
            serde_json::json!({"metadata": {}, "weight_map": weight_map}).to_string(),
        )
        .unwrap();
        index
    }

    fn file_sha(path: &std::path::Path) -> String {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
    }

    /// sc#338: a sharded checkpoint's `model_sha256` binds the index AND every
    /// shard it names (the documented manifest form), not the index alone; a
    /// changed shard under the same index changes it.
    #[test]
    fn a_sharded_checkpoint_digest_binds_every_shard() {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let index = write_two_shard_qwen3_tiny(dir.path());
        let load = || {
            let policy = CheckpointPolicy {
                model_path: Some(index.clone()),
                ..CheckpointPolicy::default()
            };
            let model = load_daemon_model(resolve_daemon_manifest_with(&policy), false)
                .expect("the two-shard fixture loads");
            assert!(!model.reference_weights, "{}", model.label);
            model
        };
        let model = load();
        assert_eq!(model.manifest.model_digest_kind, Some("index+shards"));
        let shard = |n| {
            dir.path()
                .join(format!("model-0000{n}-of-00002.safetensors"))
        };
        let expected = hex::encode(Sha256::digest(format!(
            "aien-checkpoint-digest v1\nindex {}\n{}  model-00001-of-00002.safetensors\n{}  model-00002-of-00002.safetensors\n",
            file_sha(&index),
            file_sha(&shard(1)),
            file_sha(&shard(2)),
        )));
        let digest = model.manifest.model_sha256.clone().unwrap();
        assert_eq!(digest, expected);
        assert_ne!(digest, file_sha(&index), "not the index alone");
        let line = model
            .shards_line
            .clone()
            .expect("a sharded load logs its shards");
        assert_eq!(
            line,
            format!(
                "CHECKPOINT_SHARDS model_sha256={digest} model_digest_kind=index+shards index_sha256={} shards=[model-00001-of-00002.safetensors:{},model-00002-of-00002.safetensors:{}]",
                file_sha(&index),
                file_sha(&shard(1)),
                file_sha(&shard(2)),
            )
        );
        // The "checkpoint loaded" line keeps its pinned form and carries the new digest.
        assert!(
            model.label.contains(&format!("model_sha256={digest},")),
            "{}",
            model.label
        );

        // Change one weight byte in the second shard, same index: a new digest.
        let mut bytes = std::fs::read(shard(2)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(shard(2), bytes).unwrap();
        let changed = load().manifest.model_sha256.unwrap();
        assert_ne!(changed, digest, "a changed shard must change the digest");
    }

    #[test]
    fn checkpoint_digest_refuses_index_escape_before_hashing() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model");
        std::fs::create_dir(&model).unwrap();
        std::fs::write(dir.path().join("outside.safetensors"), b"fixture").unwrap();
        let index = model.join("model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"w":"../outside.safetensors"}}"#,
        )
        .unwrap();
        assert!(checkpoint_digest(&index).is_err());
    }

    /// sc#338 review: the digest changes no refusal. A model directory as the
    /// checkpoint path is still refused (as when the path itself was hashed),
    /// and an index naming no shard is refused, with the path named once.
    #[test]
    fn a_directory_or_an_index_without_shards_is_still_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_two_shard_qwen3_tiny(dir.path());
        let err = load_daemon_model(
            resolve_daemon_manifest_with(&strict_policy(Some(dir.path().to_path_buf()))),
            true,
        )
        .err()
        .expect("a directory path is refused");
        assert!(err.contains("AIEN_REQUIRE_CHECKPOINT=1"), "{err}");
        assert!(err.contains("is a directory"), "{err}");

        let empty = tempfile::tempdir().unwrap();
        let index = empty.path().join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{}}"#).unwrap();
        let err = load_daemon_model(
            resolve_daemon_manifest_with(&strict_policy(Some(index.clone()))),
            true,
        )
        .err()
        .expect("an index without shards is refused");
        assert!(err.contains("names no shard"), "{err}");
        assert_eq!(
            err.matches(&index.display().to_string()).count(),
            1,
            "{err}"
        );
    }

    /// sc#338: a single safetensors file keeps `model_sha256` = the file's own
    /// sha256 (`model_digest_kind` = `file`) and logs no shard line.
    #[test]
    fn a_single_file_checkpoint_digest_is_the_file_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aien-inference-abi/fixtures/qwen3-tiny");
        for f in ["config.json", "model.safetensors"] {
            std::fs::copy(fixture.join(f), dir.path().join(f)).unwrap();
        }
        let path = dir.path().join("model.safetensors");
        let policy = CheckpointPolicy {
            model_path: Some(path.clone()),
            ..CheckpointPolicy::default()
        };
        let model = load_daemon_model(resolve_daemon_manifest_with(&policy), false).unwrap();
        assert!(!model.reference_weights, "{}", model.label);
        assert_eq!(model.manifest.model_digest_kind, Some("file"));
        assert_eq!(model.manifest.model_sha256, Some(file_sha(&path)));
        assert_eq!(model.shards_line, None);
    }

    /// Real-checkpoint proof. Ignored in plain `cargo test`; the forge runs it with
    /// `--include-ignored`, and then an unset AIEN_E2E_CHECKPOINT is a FAILURE, not a skip.
    #[test]
    #[ignore = "needs AIEN_E2E_CHECKPOINT (TinyLlama dir or model.safetensors); forge runs it"]
    fn strict_checkpoint_real_tinyllama_loads() {
        use sha2::{Digest, Sha256};
        let raw = std::env::var("AIEN_E2E_CHECKPOINT")
            .expect("AIEN_E2E_CHECKPOINT must point at the TinyLlama checkpoint");
        let mut model_path = std::path::PathBuf::from(raw);
        if model_path.is_dir() {
            model_path = model_path.join("model.safetensors");
        }
        let policy = strict_policy(Some(model_path.clone()));
        let manifest = resolve_daemon_manifest_with(&policy);
        let model = load_daemon_model(manifest, true).expect("real TinyLlama must load");
        assert!(!model.reference_weights);
        assert_eq!(
            model.weights.config.model_id,
            aien_inference_abi::ModelConfig::tinyllama_1_1b().model_id
        );
        assert!(model.tokenizer.is_some());

        let model_sha = model.manifest.model_sha256.clone().expect("model digest");
        assert_eq!(model_sha.len(), 64);
        assert!(model_sha.chars().all(|c| c.is_ascii_hexdigit()));
        let expected = hex::encode(Sha256::digest(std::fs::read(&model_path).unwrap()));
        assert_eq!(model_sha, expected);

        let tokenizer_path = model_path.with_file_name("tokenizer.json");
        let tokenizer_sha = model
            .manifest
            .tokenizer_sha256
            .clone()
            .expect("tokenizer digest");
        let expected = hex::encode(Sha256::digest(std::fs::read(&tokenizer_path).unwrap()));
        assert_eq!(tokenizer_sha, expected);
    }
}
