#![allow(clippy::type_complexity, clippy::needless_range_loop)]
mod allen;
mod client;
mod commands;
pub mod compaction;
mod compose;
mod context7;
pub mod cortex;
mod crumbs;
pub mod goals;
mod hive;
pub mod hooks;
mod nesting;
mod objective;
pub mod platform;
pub mod rules;
mod safety;
pub mod sandbox;
pub mod science_context;
pub mod skills;
pub mod subagents;
mod telemetry;
pub mod tool_batch;
pub mod tool_plan;
pub mod tool_spill;
mod tools;
pub mod turn_journal;
pub mod turn_recovery;
mod ui;
pub mod vault;
mod walkthrough;

use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::{json, Value};
use std::env;
use std::fs;

use client::{extract_tool_calls, get_system_prompt, ChatClient};
use commands::{handle_slash_command, run_doctor};
use nesting::perform_nesting_ritual;
use tools::dispatch_tool;
use ui::print_banner;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        if args[1] == "--daemon" || args[1] == "daemon" {
            commands::run_daemon_server().await;
            return;
        }
        if args[1] == "--swarm" || args[1] == "swarm" {
            commands::handle_swarm_command(&args[2..]).await;
            return;
        }
        if args[1] == "--start" || args[1] == "start" {
            commands::handle_start_command().await;
            return;
        }
        if args[1] == "--stop" || args[1] == "stop" {
            commands::handle_stop_command().await;
            return;
        }
        if args[1] == "allen" {
            allen::handle_allen_command(&args[2..]).await;
            return;
        }
        if args[1] == "objective" {
            objective::handle_objective_command(&args[2..]);
            return;
        }
        if args[1] == "compose" {
            compose::handle_compose_command(&args[2..]).await;
            return;
        }
        if args[1] == "--status" || args[1] == "status" {
            commands::handle_status_command().await;
            return;
        }
        if args[1] == "--cockpit" || args[1] == "cockpit" {
            commands::handle_cockpit_command().await;
            return;
        }
        if args[1] == "--harness" || args[1] == "harness" {
            commands::handle_harness_command().await;
            return;
        }
        if args[1] == "--aegis" || args[1] == "aegis" {
            commands::handle_aegis_command().await;
            return;
        }
        if args[1] == "--adapter" || args[1] == "adapter" || args[1] == "--adapters" {
            let cmd = if args.len() > 2 {
                format!("/adapter {}", args[2..].join(" "))
            } else {
                "/adapter".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "--skill" || args[1] == "skill" || args[1] == "--skills" {
            let cmd = if args.len() > 2 {
                format!("/skill {}", args[2..].join(" "))
            } else {
                "/skill".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if (args[1] == "--auto" || args[1] == "auto") && args.len() > 2 {
            run_autonomous_goal(&args[2]).await;
            return;
        }
        if (args[1] == "--goal" || args[1] == "goal") && args.len() > 3 && args[2] == "auto" {
            run_autonomous_goal(&args[3]).await;
            return;
        }
        if args[1] == "--goal" || args[1] == "goal" || args[1] == "--goals" {
            let cmd = if args.len() > 2 {
                format!("/goal {}", args[2..].join(" "))
            } else {
                "/goal".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "context7-sync" || args[1] == "--context7-sync" {
            let code = context7::sync_to_cortex().await;
            std::process::exit(code);
        }
        if args[1] == "--cortex" || args[1] == "cortex" {
            let cmd = if args.len() > 2 {
                format!("/cortex {}", args[2..].join(" "))
            } else {
                "/cortex".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "--crumb" || args[1] == "crumb" {
            let target = if args.len() > 2 {
                std::path::Path::new(&args[2])
            } else {
                std::path::Path::new(".")
            };
            println!("{}", crumbs::format_dir_crumb_tui(target));
            return;
        }
        if args[1] == "--doctor" || args[1] == "doctor" {
            run_doctor().await;
            return;
        }
        if args[1] == "--rules" || args[1] == "rules" {
            let _ = handle_slash_command("/rules").await;
            return;
        }
        if args[1] == "--vault" || args[1] == "vault" {
            let _ = handle_slash_command("/vault").await;
            return;
        }
        if args[1] == "--nest" || args[1] == "nest" {
            perform_nesting_ritual("manual-nest", true).await;
            return;
        }
        if args[1] == "--help" || args[1] == "-h" || args[1] == "help" {
            let _ = handle_slash_command("/help").await;
            return;
        }
        if args[1] == "--version" || args[1] == "-v" {
            let surface = aien_inference_abi::ExecutionSurface::detect();
            println!("AIEN CLI v0.1.0 (Host: {})", surface.display_name());
            return;
        }
        if args[1] == "--walkthrough"
            || args[1] == "-w"
            || args[1] == "--roadmap"
            || args[1] == "walkthrough"
        {
            let card = walkthrough::render_walkthrough_tui();
            println!("{}", card);
            let platform = crate::platform::PlatformContext::detect();
            let dest = platform.home_dir.join("basecamp/WALKTHROUGH.md");
            let _ = walkthrough::generate_and_save_walkthrough_md(&dest);
            return;
        }
        if args[1] == "--sandbox" || args[1] == "sandbox" {
            let cmd = if args.len() > 2 {
                format!("/sandbox {}", args[2..].join(" "))
            } else {
                "/sandbox".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "--browser" || args[1] == "browser" || args[1] == "--mirror" {
            let cmd = if args.len() > 2 {
                format!("/browser {}", args[2..].join(" "))
            } else {
                "/browser".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "--subagents"
            || args[1] == "subagents"
            || args[1] == "--subagent"
            || args[1] == "subagent"
        {
            let cmd = if args.len() > 2 {
                format!("/subagents {}", args[2..].join(" "))
            } else {
                "/subagents".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if args[1] == "--skill" || args[1] == "skill" {
            let cmd = if args.len() > 2 {
                format!("/skill {}", args[2..].join(" "))
            } else {
                "/skill".to_string()
            };
            let _ = handle_slash_command(&cmd).await;
            return;
        }
        if (args[1] == "-p" || args[1] == "--prompt") && args.len() > 2 {
            let prompt = args[2..].join(" ");
            if prompt.starts_with("/") {
                let _ = handle_slash_command(&prompt).await;
                return;
            }
            run_single_prompt(&prompt).await;
            return;
        }
    }

    // Interactive REPL Mode
    // sc#665: `aien-cli --resume <session-id>` continues a session from its journal.
    let resume_id: Option<String> =
        if args.len() > 2 && (args[1] == "--resume" || args[1] == "resume") {
            Some(args[2].clone())
        } else {
            None
        };
    print_banner();

    let platform = crate::platform::PlatformContext::detect();
    let sessions_dir = platform.home_dir.join("basecamp/sessions");
    let _ = fs::create_dir_all(&sessions_dir);

    // sc#665: every message and tool request is on disk before the loop acts on it.
    let (session_id, mut journal, resumed) = match &resume_id {
        Some(id) => {
            let path = sessions_dir.join(format!("{}.journal.jsonl", id));
            match turn_journal::Journal::resume(&path) {
                Ok((j, rep)) => (id.clone(), j, Some(rep)),
                Err(e) => {
                    eprintln!("{}", format!("Cannot resume {}: {}", id, e).red().bold());
                    return;
                }
            }
        }
        None => {
            let id = format!("session-{}", chrono::Utc::now().timestamp());
            let path = sessions_dir.join(format!("{}.journal.jsonl", id));
            match turn_journal::Journal::create(&path) {
                Ok(j) => (id, j, None),
                Err(e) => {
                    eprintln!(
                        "{}",
                        format!(
                            "Cannot start the session journal, so no tool could run safely: {}",
                            e
                        )
                        .red()
                        .bold()
                    );
                    return;
                }
            }
        }
    };

    let client = ChatClient::new(None, None);
    let mut messages: Vec<Value> = Vec::new();

    match resumed {
        Some(rep) if !rep.messages.is_empty() => {
            messages = rep.messages;
            println!(
                "{}",
                format!("Resumed {}: {} messages.", session_id, messages.len()).cyan()
            );
            for reason in &rep.stops {
                println!("{}", format!("  earlier: {}", reason).dimmed());
            }
            if rep.torn_tail {
                println!(
                    "{}",
                    "  The last line of the journal was cut off mid-write and was dropped."
                        .dimmed()
                );
            }
            if !rep.in_doubt.is_empty() {
                match turn_journal::settle_in_doubt(
                    &mut journal,
                    &mut messages,
                    &rep.in_doubt,
                    dispatch_tool,
                ) {
                    Ok(steps) => {
                        for step in steps {
                            match step {
                                turn_journal::ResumeStep::Rerun(i) => println!(
                                    "{}",
                                    format!("  {} ({}) was a read; ran it again.", i.name, i.call_id)
                                        .dimmed()
                                ),
                                turn_journal::ResumeStep::ReportUnknown { intent, .. } => println!(
                                    "{}",
                                    format!(
                                        "  {} ({}) was recorded but its result was not; not run again. Allen has been told it is not known whether it happened.",
                                        intent.name, intent.call_id
                                    )
                                    .yellow()
                                ),
                            }
                        }
                        if let Err(e) = journal.snapshot(&messages) {
                            eprintln!("{}", format!("warning: {}", e).yellow());
                        }
                    }
                    Err(stop) => {
                        eprintln!("{}", stop.describe().red().bold());
                        return;
                    }
                }
            }
        }
        _ => {
            // Mandatory Nesting Ritual: Circle, Check Wind, Leave Scent, Flatten Terrain
            let nesting = perform_nesting_ritual(&session_id, true).await;

            let cwd = std::env::current_dir().unwrap_or_else(|_| platform.home_dir.clone());
            let rules = crate::rules::discover_rules(&cwd);
            let rules_prompt = crate::rules::format_rules_for_prompt(&rules);

            let mut full_system_prompt = get_system_prompt();
            if !rules_prompt.is_empty() {
                full_system_prompt.push_str(&rules_prompt);
            }

            let opening = [
                json!({"role": "system", "content": full_system_prompt}),
                json!({"role": "user", "content": format!("<system_grounding>\n{}\n</system_grounding>\nPerform your mandatory nesting initiation. Confirm your verified terrain, wind, scent mark, and operational readiness before taking commands.", nesting.grounding_context)}),
            ];
            for m in opening {
                if let Err(stop) = journal.push(&mut messages, m) {
                    eprintln!("{}", stop.describe().red().bold());
                    return;
                }
            }

            print!("{}", "AIEN Initiation: ".magenta().bold());
            let init_resp = match client.stream_turn(&messages, true).await {
                Ok(text) => text,
                Err(e) => {
                    eprintln!(
                        "\n{}",
                        format!("Failed during model nesting initiation: {}", e)
                            .red()
                            .bold()
                    );
                    return;
                }
            };
            if let Err(stop) = journal.push(
                &mut messages,
                json!({"role": "assistant", "content": init_resp}),
            ) {
                eprintln!("{}", stop.describe().red().bold());
                return;
            }
        }
    }
    println!(
        "{}",
        format!(
            "Session {} (continue it later with: aien-cli --resume {})",
            session_id, session_id
        )
        .dimmed()
    );

    let mut rl = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("Failed to initialize readline: {}", e);
            return;
        }
    };

    let history_path = platform.home_dir.join("basecamp/.aien_history");
    let history_file = history_path.to_str().unwrap_or(".aien_history");
    let _ = rl.load_history(history_file);

    println!(
        "{}",
        "Type your request, or /help for slash commands. Press Ctrl+C or /exit to exit.\n".dimmed()
    );

    // sc#397: one compactor per session. Originals of every altered or removed
    // message are written under the session store before the list changes.
    let compactor = crate::compaction::ContextCompactor::default()
        .with_store(crate::compaction::CompactionStore::new(
            sessions_dir.join(format!("{}-compaction", session_id)),
        ))
        .with_redactor(crate::vault::redact_secrets)
        .with_counter_from_env();
    let micro_compact = crate::compaction::micro_compact_enabled_from_env();
    // sc#666: tool batches; large results spill into the session's own folder.
    let batch = tool_batch::BatchConfig::new(
        compactor.max_tokens,
        Some(sessions_dir.join(format!("{}-spill", session_id))),
    );

    loop {
        let prompt_str = format!("{}{}", "AIEN".magenta().bold(), " ❯ ".cyan().bold());
        match rl.readline(&prompt_str) {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(trimmed);

                if trimmed.starts_with("/") && handle_slash_command(trimmed).await {
                    continue;
                }

                let mut user_turn = trimmed.to_string();
                let hook_registry = crate::hooks::HookRegistry::default_sovereign_registry();
                let _ = hook_registry.run_pre_turn(&mut user_turn).await;

                if let Err(stop) =
                    journal.push(&mut messages, json!({"role": "user", "content": user_turn}))
                {
                    println!("\n{}", stop.describe().red().bold());
                    continue;
                }

                let report = compactor.compact_if_needed(&mut messages);
                record_compaction(&report, &messages, &mut journal);

                // Multi-turn tool execution loop (sc#665: persist-before-act, one-shot recovery)
                if let Some(stop) = run_model_steps(
                    &client,
                    &mut messages,
                    &mut journal,
                    Some(&compactor),
                    &hook_registry,
                    true,
                    &batch,
                )
                .await
                {
                    println!("\n{}", stop.describe().red().bold());
                    journal.stop(&stop);
                }

                // sc#664: opt-in per-turn folding of one exchange
                // (AIEN_MICRO_COMPACT=1). The operator's words are never folded.
                if micro_compact {
                    let report = compactor.absorb_oldest_exchange(&mut messages);
                    record_compaction(&report, &messages, &mut journal);
                }

                // The end of the turn: the journal snapshot is the record of this
                // session; the plain JSON file is kept for readers that expect it.
                if let Err(e) = journal.snapshot(&messages) {
                    eprintln!(
                        "{}",
                        format!("warning: could not record the end of this turn: {}", e).yellow()
                    );
                }
                write_session_file(
                    &sessions_dir.join(format!("{}.json", session_id)),
                    &messages,
                );
                println!();
            }
            Err(ReadlineError::Interrupted) => {
                println!("\n{}", "^C (type /exit to quit)".dimmed());
            }
            Err(ReadlineError::Eof) => {
                println!("\n{}", "Exiting AIEN.".cyan());
                break;
            }
            Err(err) => {
                eprintln!("Readline error: {}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(history_file);
}

async fn run_single_prompt(prompt: &str) {
    let session_id = format!("session-single-{}", chrono::Utc::now().timestamp());
    let platform = crate::platform::PlatformContext::detect();
    let sessions_dir = platform.home_dir.join("basecamp/sessions");
    let _ = fs::create_dir_all(&sessions_dir);

    // Mandatory Nesting Ritual: Required at startup across all modes
    let nesting = perform_nesting_ritual(&session_id, true).await;

    let mut turn_prompt = prompt.to_string();
    let hook_registry = crate::hooks::HookRegistry::default_sovereign_registry();
    let _ = hook_registry.run_pre_turn(&mut turn_prompt).await;

    let cwd = std::env::current_dir().unwrap_or_else(|_| platform.home_dir.clone());
    let rules = crate::rules::discover_rules(&cwd);
    let rules_prompt = crate::rules::format_rules_for_prompt(&rules);

    let mut full_system_prompt = get_system_prompt();
    if !rules_prompt.is_empty() {
        full_system_prompt.push_str(&rules_prompt);
    }

    let client = ChatClient::new(None, None);
    let mut messages: Vec<Value> = Vec::new();

    // sc#665: same persist-before-act discipline as the interactive loop.
    let mut journal = match turn_journal::Journal::create(
        &sessions_dir.join(format!("{}.journal.jsonl", session_id)),
    ) {
        Ok(j) => j,
        Err(e) => {
            eprintln!(
                "Error: cannot start the session journal, so no tool could run safely: {}",
                e
            );
            return;
        }
    };
    let batch = tool_batch::BatchConfig::new(
        crate::compaction::ContextCompactor::default().max_tokens,
        Some(sessions_dir.join(format!("{}-spill", session_id))),
    );
    let opening = [
        json!({"role": "system", "content": full_system_prompt}),
        json!({"role": "user", "content": format!("<system_grounding>\n{}\n</system_grounding>\n{}", nesting.grounding_context, turn_prompt)}),
    ];
    for m in opening {
        if let Err(stop) = journal.push(&mut messages, m) {
            eprintln!("Error: {}", stop.describe());
            return;
        }
    }

    if let Some(stop) = run_model_steps(
        &client,
        &mut messages,
        &mut journal,
        None,
        &hook_registry,
        false,
        &batch,
    )
    .await
    {
        eprintln!("Error: {}", stop.describe());
        journal.stop(&stop);
    }

    let _ = journal.snapshot(&messages);
    write_session_file(
        &sessions_dir.join(format!("{}.json", session_id)),
        &messages,
    );
}

/// Writes the plain session file whole or not at all (temp file, then rename).
fn write_session_file(path: &std::path::Path, messages: &[Value]) {
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(messages).unwrap_or_default();
    if fs::write(&tmp, body).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

/// sc#665: one operator turn of model calls and tool rounds.
///
/// Each assistant message is on disk before its tool calls run, and each tool
/// request is on disk before the tool runs (`turn_journal`). A failed model call
/// gets at most one recovery per failure class (`turn_recovery`), so a turn can
/// never loop on recovery. A reply cut off inside a tool call is never executed:
/// it is asked again once with a higher output limit.
async fn run_model_steps(
    client: &ChatClient,
    messages: &mut Vec<Value>,
    journal: &mut turn_journal::Journal,
    compactor: Option<&crate::compaction::ContextCompactor>,
    hook_registry: &crate::hooks::HookRegistry,
    label: bool,
    batch: &tool_batch::BatchConfig,
) -> Option<turn_journal::TurnStop> {
    use turn_journal::TurnStop;
    use turn_recovery::{
        classify, tool_call_truncated, FailureClass, Recovery, RecoveryFlags, BASE_OUTPUT_CAP,
    };

    let mut flags = RecoveryFlags::default();
    let mut cap = BASE_OUTPUT_CAP;
    let mut steps_left = 10;
    while steps_left > 0 {
        if label {
            print!("\n{}", "AIEN: ".magenta().bold());
        }
        let mut assistant_resp = match client.stream_turn_with_limit(messages, true, cap).await {
            Ok(text) => text,
            Err(e) => match flags.next(classify(&e)) {
                Recovery::Retry { wait } => {
                    println!(
                        "\n{}",
                        format!("(model call failed: {}; trying once more)", e).dimmed()
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }
                Recovery::CompactThenRetry => match compactor {
                    Some(c) => {
                        println!(
                            "\n{}",
                            "(the conversation is too long for the model; compacting and trying once more)"
                                .dimmed()
                        );
                        let report = c.compact_if_needed(messages);
                        record_compaction(&report, messages, journal);
                        continue;
                    }
                    None => {
                        println!(
                            "\n{}",
                            format!("Error during inference: {}", e).red().bold()
                        );
                        return Some(TurnStop::Recovery(
                            "context-overflow: too long, and this mode has no compactor",
                        ));
                    }
                },
                Recovery::Stop { reason } => {
                    println!(
                        "\n{}",
                        format!("Error during inference: {}", e).red().bold()
                    );
                    return Some(TurnStop::Recovery(reason));
                }
                Recovery::RaiseCapThenRetry { .. } => {
                    // classify() never yields OutputTruncated for a transport error.
                    println!(
                        "\n{}",
                        format!("Error during inference: {}", e).red().bold()
                    );
                    return Some(TurnStop::Recovery(
                        "model-error: an error with no safe recovery",
                    ));
                }
            },
        };

        if tool_call_truncated(&assistant_resp) {
            match flags.next(FailureClass::OutputTruncated) {
                Recovery::RaiseCapThenRetry { max_tokens } => {
                    println!(
                        "\n{}",
                        "(the reply was cut off inside a tool call, so that call was not run; asking again with a higher output limit)"
                            .dimmed()
                    );
                    cap = max_tokens;
                    continue;
                }
                Recovery::Stop { reason } => return Some(TurnStop::Recovery(reason)),
                _ => return Some(TurnStop::Recovery("tool-call-truncated")),
            }
        }
        steps_left -= 1;

        hook_registry.run_post_turn(&mut assistant_resp).await;
        if let Err(stop) = journal.push(
            messages,
            json!({"role": "assistant", "content": assistant_resp.clone()}),
        ) {
            return Some(stop);
        }

        if let Some(c) = compactor {
            let report = c.compact_if_needed(messages);
            record_compaction(&report, messages, journal);
        }

        let tool_calls = extract_tool_calls(&assistant_resp);
        if tool_calls.is_empty() {
            // Completed turn
            return None;
        }
        match run_tools(journal, messages, tool_calls, batch) {
            Ok(r) => report_batch(&r),
            Err(stop) => return Some(stop),
        }
    }
    None
}

/// sc#666: one reply's tool calls as a planned batch. Reads of the same or
/// different paths run together; a write never runs beside a call on an
/// overlapping path; shell commands, unknown tools and any call that would ask
/// the operator run alone, so two approval prompts never share the screen.
fn run_tools(
    journal: &mut turn_journal::Journal,
    messages: &mut Vec<Value>,
    calls: Vec<(String, Value)>,
    batch: &tool_batch::BatchConfig,
) -> Result<tool_batch::BatchReport, turn_journal::TurnStop> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let safety = crate::safety::SafetyEngine::default_sovereign_engine();
    let access = |name: &str, args: &Value| match safety.evaluate(name, args) {
        crate::safety::SafetyDecision::AskUser(_) => tool_plan::Access::Barrier,
        _ => tool_plan::access_of(name, args, &cwd),
    };
    // Tools reach the async runtime from their own threads through the handle.
    let handle = tokio::runtime::Handle::current();
    let dispatch = move |name: &str, args: &Value| {
        let _runtime = handle.enter();
        dispatch_tool(name, args)
    };
    tool_batch::run_batch(journal, messages, calls, batch, access, dispatch)
}

fn report_batch(r: &tool_batch::BatchReport) {
    if r.ran_together > 0 {
        println!(
            "{}",
            format!(
                "  ⚡ tools: {} ran side by side ({} steps)",
                r.ran_together, r.segments
            )
            .cyan()
            .dimmed()
        );
    }
    for p in &r.spilled {
        println!(
            "{}",
            format!(
                "  ⚡ tools: a large result was saved to {} (Allen sees a preview)",
                p.display()
            )
            .cyan()
            .dimmed()
        );
    }
}

pub async fn run_autonomous_goal(goal_id: &str) {
    let manifest = goals::load_goals();
    let goal = match manifest
        .goals
        .iter()
        .find(|g| g.id == goal_id || g.title.to_lowercase().contains(&goal_id.to_lowercase()))
    {
        Some(g) => g.clone(),
        None => {
            eprintln!(
                "{}",
                format!("Goal '{}' not found in goals.json", goal_id).red()
            );
            return;
        }
    };

    println!(
        "{}",
        format!(
            "\n🚀 Launching Autonomous Execution Loop for Goal: {}",
            goal.title
        )
        .cyan()
        .bold()
    );
    println!(
        "{}",
        format!(
            "ID: {} | Total Milestones: {}",
            goal.id,
            goal.milestones.len()
        )
        .dimmed()
    );

    let session_id = format!("auto-goal-{}", chrono::Utc::now().timestamp());
    let nesting = perform_nesting_ritual(&session_id, true).await;
    let client = ChatClient::new(None, None);

    for m in &goal.milestones {
        if m.completed {
            println!(
                "{}",
                format!(
                    "  ✓ Milestone {} already completed: {}",
                    m.id, m.description
                )
                .green()
            );
            continue;
        }

        println!(
            "{}",
            format!("\n▶ Commencing Milestone {}: {}", m.id, m.description)
                .yellow()
                .bold()
        );

        let mut task_directive = format!(
            "<system_grounding>\n{}\n</system_grounding>\nAUTONOMOUS TASK DIRECTIVE:\nGoal: '{}' (ID: {})\nActive Milestone {}: {}\n\nExecute the necessary commands, write/edit files, run verifications, and accomplish this milestone.\nWhen verified, use the 'goal' tool with action 'milestone_done' (id: \"{}\", milestone_id: {}) and write any durable lesson learned with the 'cortex' tool.\nBegin now.",
            nesting.grounding_context,
            goal.title,
            goal.id,
            m.id,
            m.description,
            goal.id,
            m.id
        );
        let hook_registry = crate::hooks::HookRegistry::default_sovereign_registry();
        let _ = hook_registry.run_pre_turn(&mut task_directive).await;

        let mut messages = vec![
            json!({"role": "system", "content": get_system_prompt()}),
            json!({
                "role": "user",
                "content": task_directive
            }),
        ];

        let mut max_tool_steps = 15;
        let mut milestone_done = false;

        while max_tool_steps > 0 {
            max_tool_steps -= 1;
            print!(
                "\n{}",
                format!("AIEN [Milestone {}]: ", m.id).magenta().bold()
            );
            let mut assistant_resp = match client.stream_turn(&messages, true).await {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("Inference error: {}", e);
                    break;
                }
            };

            hook_registry.run_post_turn(&mut assistant_resp).await;
            messages.push(json!({"role": "assistant", "content": assistant_resp.clone()}));

            let tool_calls = extract_tool_calls(&assistant_resp);
            if !tool_calls.is_empty() {
                for (tool_name, tool_args) in tool_calls {
                    if tool_name == "goal" {
                        let action = tool_args
                            .get("action")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if action == "milestone_done" || action == "done" {
                            milestone_done = true;
                        }
                    }
                    let tool_result = dispatch_tool(&tool_name, &tool_args);
                    let result_str = serde_json::to_string(&tool_result).unwrap_or_default();
                    messages.push(json!({
                        "role": "user",
                        "content": format!("<tool_response name=\"{}\">\n{}\n</tool_response>", tool_name, result_str)
                    }));
                }
            } else {
                break;
            }
            if milestone_done {
                break;
            }
        }

        let _ = goals::complete_milestone(&goal.id, m.id);
        println!(
            "{}",
            format!("  ✓ Milestone {} accomplished: {}", m.id, m.description)
                .green()
                .bold()
        );
    }

    println!(
        "{}",
        format!(
            "\n🏆 All milestones for goal '{}' completed successfully!",
            goal.title
        )
        .green()
        .bold()
    );
}

/// Prints the outcome of a compaction pass. Skips are reported too, so a
/// refusal (detached tool result, store failure) is never silent.
/// sc#664: prints the compaction report and, when the history changed, records
/// the new history in the session journal at once. The journal line is written
/// whole or not at all, so a resumed session loads either the old history or
/// the compacted one, never both the recap and the messages it replaced.
fn record_compaction(
    report: &crate::compaction::CompactionReport,
    messages: &[Value],
    journal: &mut turn_journal::Journal,
) {
    report_compaction(report);
    if report.applied {
        if let Err(e) = journal.snapshot(messages) {
            eprintln!(
                "{}",
                format!(
                    "warning: the compacted history could not be recorded ({e}); a resumed session would load the history from before this compaction"
                )
                .yellow()
            );
        }
    }
}

fn report_compaction(report: &crate::compaction::CompactionReport) {
    use crate::compaction::SkipReason;
    if let Some(stats) = &report.stats {
        println!(
            "{}",
            format!(
                "  ⚡ ContextCompactor: {} -> {} tokens ({}), {} messages recapped, {} tool bytes pruned, {} oversized, record {}",
                stats.initial_tokens,
                stats.compacted_tokens,
                stats.token_source,
                stats.pruned_messages_count,
                stats.pruned_tool_bytes,
                stats.oversized_results,
                stats.recovery_pointer
            )
            .cyan()
            .dimmed()
        );
        if stats.still_over_budget {
            println!(
                "{}",
                "  ⚡ ContextCompactor: still over the hard token maximum after compaction (everything left is protected: recent turns or an approval wait)"
                    .yellow()
                    .dimmed()
            );
        }
        return;
    }
    match &report.skipped {
        None | Some(SkipReason::NotNeeded) | Some(SkipReason::TooShort) => {}
        Some(reason) => println!(
            "{}",
            format!(
                "  ⚡ ContextCompactor: skipped ({:?}); history left unchanged",
                reason
            )
            .yellow()
            .dimmed()
        ),
    }
}

#[cfg(test)]
mod compaction_journal_tests {
    use super::*;
    use crate::compaction::{CompactionStore, ContextCompactor};

    fn user(t: &str) -> Value {
        json!({"role": "user", "content": t})
    }

    /// sc#664 acceptance: a compaction is on disk before anything else
    /// happens, so a process killed right after it resumes with the compacted
    /// history, once: one recap, not the recap plus the messages it replaced.
    #[test]
    fn resume_after_a_compaction_does_not_double_load() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("s.journal.jsonl");
        let mut journal = turn_journal::Journal::create(&path).unwrap();
        let compactor = ContextCompactor::new(2000, 1000, 500)
            .with_store(CompactionStore::new(d.path().join("compaction")));
        let mut messages = Vec::new();
        let opening = [
            json!({"role": "system", "content": "sys"}),
            user("grounding"),
        ];
        for m in opening {
            journal.push(&mut messages, m).unwrap();
        }
        for k in 0..4 {
            journal
                .push(&mut messages, user(&format!("ask {k}")))
                .unwrap();
            journal
                .push(
                    &mut messages,
                    json!({"role": "assistant", "content": "<tool_call>\n{\"name\": \"view_file\", \"arguments\": {}}\n</tool_call>"}),
                )
                .unwrap();
            let big = format!(
                "<tool_response name=\"view_file\">\n{}\n</tool_response>",
                "Z".repeat(4000)
            );
            journal.push(&mut messages, user(&big)).unwrap();
            journal
                .push(
                    &mut messages,
                    json!({"role": "assistant", "content": format!("answer {k}")}),
                )
                .unwrap();
        }
        let before = messages.len();
        let report = compactor.compact_if_needed(&mut messages);
        assert!(report.applied, "{report:?}");
        assert!(messages.len() < before);
        record_compaction(&report, &messages, &mut journal);
        // Killed here: nothing else is written.
        drop(journal);

        let (_j, rep) = turn_journal::Journal::resume(&path).unwrap();
        assert_eq!(rep.messages, messages);
        let recaps = rep
            .messages
            .iter()
            .filter(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|c| c.starts_with("<context_compaction_recap"))
            })
            .count();
        assert_eq!(recaps, 1);
        for k in 0..4 {
            assert!(
                rep.messages.contains(&user(&format!("ask {k}"))),
                "ask {k} lost"
            );
        }
    }
}

#[cfg(test)]
mod tool_batch_runtime_tests {
    use super::*;

    /// sc#666: the real tools, run side by side from plain threads inside the
    /// real multi-threaded runtime. `dispatch_tool` reaches the runtime
    /// (`block_in_place`, `Handle::current`) for its hooks; the batch runner
    /// must hand each thread the runtime handle or this panics.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_reads_run_side_by_side_inside_the_runtime() {
        let d = tempfile::Builder::new()
            .prefix("aien-666-")
            .tempdir_in("/tmp")
            .unwrap();
        let a = d.path().join("a.txt");
        let b = d.path().join("b.txt");
        std::fs::write(&a, "alpha\n").unwrap();
        std::fs::write(&b, "beta\n").unwrap();
        let mut journal = turn_journal::Journal::create(&d.path().join("s.journal.jsonl")).unwrap();
        let mut messages = Vec::new();
        let batch = tool_batch::BatchConfig::new(24_000, Some(d.path().join("spill")));
        let calls = vec![
            (
                "view_file".to_string(),
                json!({"path": a.display().to_string()}),
            ),
            (
                "view_file".to_string(),
                json!({"path": b.display().to_string()}),
            ),
        ];
        let report = run_tools(&mut journal, &mut messages, calls, &batch).unwrap();
        assert_eq!((report.segments, report.ran_together), (1, 2));
        assert_eq!(messages.len(), 2);
        let first = messages[0]["content"].as_str().unwrap();
        let second = messages[1]["content"].as_str().unwrap();
        assert!(
            first.contains("alpha") && !first.contains("CONFINEMENT"),
            "{first}"
        );
        assert!(second.contains("beta"), "{second}");
    }
}
