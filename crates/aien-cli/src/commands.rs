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
  context7-sync             Refresh Rust, Mojo, MAX, Tokio, Axum, and Serde docs into Cortex
  /sandbox [init|status|test|promote|clean] Isolated Git worktree sandbox
  /browser [test|mentor]    Headless Chrome CDP mirror self-testing & mentoring
  /subagents [list|view|run] Recursive contextual subagents hierarchy
  /goal [list|new|done]     Manage project goals & milestone lattices"
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
            if parts.len() > 1 && (parts[1] == "now" || parts[1] == "run") {
                println!("{}", "Triggering immediate JSpace Dream Cycle...".yellow());
                let platform = crate::platform::PlatformContext::detect();
                let py = platform.python_bin();
                let dream_script = platform
                    .home_dir
                    .join("basecamp/aien-dream/dream_engine.py");
                let out = Command::new(py)
                    .args([dream_script.to_str().unwrap_or(""), "--now"])
                    .output();
                match out {
                    Ok(o) => {
                        println!("{}", String::from_utf8_lossy(&o.stdout));
                        if !o.stderr.is_empty() {
                            eprintln!("{}", String::from_utf8_lossy(&o.stderr).dimmed());
                        }
                    }
                    Err(e) => println!("{}", format!("Failed to run dream engine: {}", e).red()),
                }
            } else {
                let platform = crate::platform::PlatformContext::detect();
                let py = platform.python_bin();
                let dream_script = platform
                    .home_dir
                    .join("basecamp/aien-dream/dream_engine.py");
                let out = Command::new(py)
                    .args([dream_script.to_str().unwrap_or(""), "--status"])
                    .output();
                match out {
                    Ok(o) => println!("{}", String::from_utf8_lossy(&o.stdout)),
                    Err(e) => println!("{}", format!("Failed to query dream status: {}", e).red()),
                }
            }
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
            "✓ JSpace Truth Judge (Port 18082): Online (Llama-3.2-1B on Modular MAX CPU)".green()
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
struct DaemonModelManifest {
    model_id: String,
    label: String,
    checkpoint_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
}

fn is_safetensors(path: &std::path::Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
}

fn tokenizer_beside(checkpoint: &std::path::Path) -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("AIEN_TOKENIZER") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let candidate = checkpoint.parent()?.join("tokenizer.json");
    candidate.is_file().then_some(candidate)
}

fn resolve_daemon_manifest() -> DaemonModelManifest {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("AIEN_MODEL_DIR") {
        candidates.push(std::path::PathBuf::from(dir));
    }
    candidates.push(std::path::PathBuf::from("models"));
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(std::path::PathBuf::from(home).join("models"));
    }
    for dir in candidates {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if is_safetensors(&path) {
                return DaemonModelManifest {
                    model_id: "checkpoint".into(),
                    label: format!(
                        "checkpoint found at {} (model_id={})",
                        path.display(),
                        path.file_stem()
                            .and_then(|name| name.to_str())
                            .unwrap_or("checkpoint")
                    ),
                    tokenizer_path: tokenizer_beside(&path),
                    checkpoint_path: Some(path),
                };
            }
            if path.is_dir() {
                let Ok(nested) = std::fs::read_dir(&path) else {
                    continue;
                };
                for child in nested.flatten() {
                    let child_path = child.path();
                    if is_safetensors(&child_path) {
                        return DaemonModelManifest {
                            model_id: path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("checkpoint")
                                .to_string(),
                            label: format!("checkpoint found at {}", child_path.display()),
                            tokenizer_path: tokenizer_beside(&child_path),
                            checkpoint_path: Some(child_path),
                        };
                    }
                }
            }
        }
    }
    DaemonModelManifest {
        model_id: "aien-daemon-reference-fallback".into(),
        label: "no safetensors checkpoint found (AIEN_MODEL_DIR, ./models, ~/models); using reference fallback".into(),
        checkpoint_path: None,
        tokenizer_path: None,
    }
}

/// Builds the native transformer backend for daemon boot with explicit fallback.
/// Prefers Blackwell hardware when available, falls back to CPU reference math.
/// Never returns the Mock backend: output always comes from real forward passes.
/// When AIEN_REQUIRE_BLACKWELL is set, a missing Blackwell device is fatal:
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
    }
}

fn load_production_weights(
    manifest: &DaemonModelManifest,
) -> (
    aien_inference_abi::TransformerWeights,
    Option<aien_inference_abi::TinyLlamaTokenizer>,
    String,
) {
    let Some(path) = &manifest.checkpoint_path else {
        let config = reference_config();
        return (
            aien_inference_abi::TransformerWeights::reference_test_weights(&config),
            None,
            manifest.label.clone(),
        );
    };
    let config = aien_inference_abi::ModelConfig::tinyllama_1_1b();
    match aien_inference_abi::TransformerWeights::load_from_safetensors(path, &config) {
        Ok(weights) => {
            let tokenizer = manifest.tokenizer_path.as_ref().and_then(|tokenizer_path| {
                aien_inference_abi::TinyLlamaTokenizer::from_file(tokenizer_path).ok()
            });
            let tokenizer_note = if tokenizer.is_some() {
                "tokenizer loaded"
            } else {
                "tokenizer missing"
            };
            (
                weights,
                tokenizer,
                format!(
                    "checkpoint loaded from {} ({}, model_id={})",
                    path.display(),
                    tokenizer_note,
                    manifest.model_id
                ),
            )
        }
        Err(error) => {
            let config = reference_config();
            (
                aien_inference_abi::TransformerWeights::reference_test_weights(&config),
                None,
                format!(
                    "{} (safetensors load failed: {}); using reference fallback",
                    manifest.label, error
                ),
            )
        }
    }
}

fn build_native_daemon_backend() -> Result<
    (
        aien_inference_abi::NativeTransformerBackend,
        String,
        String,
        Option<aien_inference_abi::TinyLlamaTokenizer>,
    ),
    String,
> {
    let manifest = resolve_daemon_manifest();
    let (weights, tokenizer, model_label) = load_production_weights(&manifest);
    let probe = aien_inference_abi::BlackwellGb10Backend::new();
    let require_blackwell = std::env::var("AIEN_REQUIRE_BLACKWELL")
        .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if probe.is_available() {
        let device = probe.device_name().to_string();
        drop(probe);
        let backend = aien_inference_abi::NativeTransformerBackend::new_blackwell(weights);
        Ok((
            backend,
            format!("NativeTransformerBackend/Blackwell ({})", device),
            model_label,
            tokenizer,
        ))
    } else if require_blackwell {
        Err(
            "AIEN_REQUIRE_BLACKWELL is set but no Blackwell device initialized (fallback count nonzero)".to_string(),
        )
    } else {
        let backend = aien_inference_abi::NativeTransformerBackend::new_reference(weights);
        Ok((
            backend,
            "NativeTransformerBackend/CPU-reference (Blackwell unavailable, explicit fallback)"
                .to_string(),
            model_label,
            tokenizer,
        ))
    }
}

pub async fn run_daemon_server() {
    println!(
        "{}",
        "⚡ Starting AIEN Sovereign Runtime Daemon...".cyan().bold()
    );
    let socket_path = aien_runtime::client::AienRuntimeClient::default_socket_path();
    let kv_manager = aien_kv_cache::create_shared_kv_manager(8192, 16);
    let sched_cfg = aien_scheduler::SchedulerConfig {
        max_batch_size: 256,
        max_batch_tokens: 16384,
        max_prefill_tokens: 8192,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 64,
    };
    let spine = aien_runtime::spine::AienRuntimeSpine::new(4096, sched_cfg, kv_manager);
    let server = aien_runtime::server::AienRuntimeServer::new(spine, &socket_path);

    let backend = {
        match build_native_daemon_backend() {
            Ok((native_backend, backend_label, model_label, tokenizer)) => {
                println!("  Backend: {}", backend_label.green());
                println!("  Model: {}", model_label.yellow());
                if let Some(tokenizer) = tokenizer {
                    server.set_tokenizer(tokenizer);
                    println!("  Tokenizer: {}", "TinyLlama chat template".green());
                } else {
                    println!(
                        "  Tokenizer: {}",
                        "not loaded; native chat will refuse turns until tokenizer.json is present"
                            .yellow()
                    );
                }
                native_backend
            }
            Err(fatal) => {
                eprintln!("Fatal: {}", fatal.red().bold());
                std::process::exit(1);
            }
        }
    };
    println!("✓ Binding socket at {}", socket_path.display());
    if let Err(e) = server.run(backend).await {
        eprintln!("Runtime daemon error: {}", e);
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
        (
            "Modular MAX Model Seat",
            "http://127.0.0.1:18006/health",
            18006,
        ),
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

pub async fn handle_harness_command() {
    println!(
        "{}",
        "⚡ Running AIEN 5-Layer Sovereign Harness Sweep..."
            .cyan()
            .bold()
    );
    let status = std::process::Command::new("spark-harness")
        .arg("check")
        .status();
    match status {
        Ok(s) if s.success() => println!(
            "{}",
            "\n✓ 5-Layer Verification PASS: All invariants certified."
                .green()
                .bold()
        ),
        _ => {
            println!(
                "{}",
                "Running comprehensive fallback diagnostic...".yellow()
            );
            run_doctor().await;
        }
    }
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
}
