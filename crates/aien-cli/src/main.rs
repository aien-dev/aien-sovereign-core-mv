#![allow(clippy::type_complexity, clippy::needless_range_loop)]
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

mod allen;
mod approval;
mod checkpoint;
mod client;
mod commands;
pub mod compaction;
mod compose;
mod context7;
pub mod cortex;
mod crumbs;
mod fs_bind;
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
    // sc#667: open the checkpoint store once the session id is known, before
    // nesting, resume settlement, or any tool. A store that cannot open is
    // reported inside start_checkpoints; the session still runs.
    start_checkpoints(&platform.home_dir, &session_id);

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

                // sc#667: a turn boundary before the message is recorded, so later
                // agent writes belong to this turn and `/rollback` can undo them.
                crate::checkpoint::begin_turn();
                if let Err(stop) =
                    journal.push(&mut messages, json!({"role": "user", "content": user_turn}))
                {
                    println!("\n{}", stop.describe().red().bold());
                    continue;
                }

                let report = compactor.compact_if_needed(&mut messages);
                if let Err(stop) = record_compaction(&report, &messages, &mut journal) {
                    println!("\n{}", stop.describe().red().bold());
                    journal.stop(&stop);
                    break;
                }

                // Multi-turn tool execution loop (sc#665: persist-before-act, one-shot recovery)
                if let Some(stop) = run_model_steps(
                    &client,
                    &mut messages,
                    &mut journal,
                    Some(&compactor),
                    &hook_registry,
                    true,
                )
                .await
                {
                    println!("\n{}", stop.describe().red().bold());
                    journal.stop(&stop);
                    // The compacted transcript is only in memory. Another step
                    // would act on a history the journal does not hold.
                    if matches!(stop, turn_journal::TurnStop::CompactionUnrecorded(_)) {
                        break;
                    }
                }

                // sc#664: opt-in per-turn folding of one exchange
                // (AIEN_MICRO_COMPACT=1). The operator's words are never folded.
                if micro_compact {
                    let report = compactor.absorb_oldest_exchange(&mut messages);
                    if let Err(stop) = record_compaction(&report, &messages, &mut journal) {
                        println!("\n{}", stop.describe().red().bold());
                        journal.stop(&stop);
                        break;
                    }
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
    start_checkpoints(&platform.home_dir, &session_id);
    crate::checkpoint::begin_turn();

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
                        if let Err(stop) = record_compaction(&report, messages, journal) {
                            return Some(stop);
                        }
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
            if let Err(stop) = record_compaction(&report, messages, journal) {
                return Some(stop);
            }
        }

        let tool_calls = extract_tool_calls(&assistant_resp);
        if tool_calls.is_empty() {
            // Completed turn
            return None;
        }
        if let Err(stop) =
            turn_journal::run_tool_calls(journal, messages, tool_calls, dispatch_tool)
        {
            return Some(stop);
        }
    }
    None
}

/// What one autonomous assistant reply did, after it was recorded.
#[derive(Debug)]
enum AutonomousStep {
    /// The reply had no tool call.
    StoppedTalking,
    /// Tools ran. `milestone_done` is the goal tool's signal.
    Acted { milestone_done: bool },
}

/// One autonomous assistant reply under persist-before-act.
///
/// The reply is on disk before any tool runs. Each tool request is on disk
/// before that tool runs. A failed journal write returns `Err` and does not
/// call `dispatch`.
fn apply_autonomous_reply<F>(
    journal: &mut turn_journal::Journal,
    messages: &mut Vec<Value>,
    assistant_resp: &str,
    dispatch: F,
) -> Result<AutonomousStep, turn_journal::TurnStop>
where
    F: FnMut(&str, &Value) -> Value,
{
    journal.push(
        messages,
        json!({"role": "assistant", "content": assistant_resp}),
    )?;
    let tool_calls = extract_tool_calls(assistant_resp);
    if tool_calls.is_empty() {
        return Ok(AutonomousStep::StoppedTalking);
    }
    let milestone_done = tool_calls.iter().any(|(name, args)| {
        name == "goal"
            && matches!(
                args.get("action").and_then(Value::as_str),
                Some("milestone_done") | Some("done")
            )
    });
    turn_journal::run_tool_calls(journal, messages, tool_calls, dispatch)?;
    Ok(AutonomousStep::Acted { milestone_done })
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
    let platform = crate::platform::PlatformContext::detect();
    let sessions_dir = platform.home_dir.join("basecamp/sessions");
    let _ = fs::create_dir_all(&sessions_dir);
    // Same persist-before-act rule as the chat loop: no tool runs unless its
    // request is on disk, and a failed write ends the goal.
    let mut journal = match turn_journal::Journal::create(
        &sessions_dir.join(format!("{}.journal.jsonl", session_id)),
    ) {
        Ok(j) => j,
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
    };
    println!(
        "{}",
        format!(
            "Session {} (continue it later with: aien-cli --resume {})",
            session_id, session_id
        )
        .dimmed()
    );
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

        let mut messages = Vec::new();
        let opening = [
            json!({"role": "system", "content": get_system_prompt()}),
            json!({"role": "user", "content": task_directive}),
        ];
        for message in opening {
            if let Err(stop) = journal.push(&mut messages, message) {
                eprintln!("{}", stop.describe().red().bold());
                journal.stop(&stop);
                return;
            }
        }

        let mut max_tool_steps = 15;
        let mut milestone_done = false;

        while max_tool_steps > 0 && !milestone_done {
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
            match apply_autonomous_reply(
                &mut journal,
                &mut messages,
                &assistant_resp,
                dispatch_tool,
            ) {
                Ok(AutonomousStep::StoppedTalking) => break,
                Ok(AutonomousStep::Acted {
                    milestone_done: done,
                }) => milestone_done = done,
                Err(stop) => {
                    eprintln!("{}", stop.describe().red().bold());
                    journal.stop(&stop);
                    return;
                }
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
///
/// `Err(CompactionUnrecorded)` means the compacted transcript is only in
/// memory. The caller stops the turn: the journal already holds the
/// pre-compaction history, and that is what resume loads. A marker cannot
/// stand in for the snapshot, because the write that failed is the one that
/// would have stored the compacted transcript.
fn record_compaction(
    report: &crate::compaction::CompactionReport,
    messages: &[Value],
    journal: &mut turn_journal::Journal,
) -> Result<(), turn_journal::TurnStop> {
    report_compaction(report);
    if report.applied {
        journal
            .snapshot(messages)
            .map_err(|e| turn_journal::TurnStop::CompactionUnrecorded(e.to_string()))?;
    }
    Ok(())
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

/// sc#667: record the agent's file writes so `/rollback` can undo them
/// without touching the operator's own edits. A store that cannot open is
/// reported; the session still runs, with writes refused only if a later
/// snapshot fails.
fn start_checkpoints(home: &std::path::Path, session_id: &str) {
    if let Err(err) = crate::checkpoint::start_session(home, session_id) {
        eprintln!(
            "{}",
            format!("Checkpoints off for this session: {}", err).yellow()
        );
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
        record_compaction(&report, &messages, &mut journal).unwrap();
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

    /// The snapshot after compaction is the compacted transcript. If it does
    /// not land, the turn does not take another step, and resume still has the
    /// history from before the compaction.
    #[test]
    fn a_failed_compaction_snapshot_stops_before_another_step() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("s.journal.jsonl");
        let mut journal = turn_journal::Journal::create(&path).unwrap();
        let compactor = ContextCompactor::new(2000, 1000, 500)
            .with_store(CompactionStore::new(d.path().join("compaction")));
        let mut messages = Vec::new();
        for m in [
            json!({"role": "system", "content": "sys"}),
            user("grounding"),
        ] {
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
        let durable = messages.clone();
        let report = compactor.compact_if_needed(&mut messages);
        assert!(report.applied, "{report:?}");
        assert_ne!(messages, durable);
        journal.accept_only_n_writes_for_test(0);

        let mut stepped = false;
        match record_compaction(&report, &messages, &mut journal) {
            Ok(()) => stepped = true,
            Err(stop) => {
                assert!(matches!(
                    stop,
                    turn_journal::TurnStop::CompactionUnrecorded(_)
                ));
                assert!(stop.describe().contains("compaction-unrecorded"));
                journal.stop(&stop);
            }
        }
        assert!(
            !stepped,
            "the turn continued after the compaction snapshot failed"
        );
        drop(journal);

        let (_j, rep) = turn_journal::Journal::resume(&path).unwrap();
        assert_eq!(
            rep.messages, durable,
            "resume loads the history from before this compaction"
        );
        assert!(
            !rep.messages.iter().any(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|c| c.starts_with("<context_compaction_recap"))
            }),
            "the unrecorded recap must not be in the journal"
        );
    }

    #[test]
    fn a_skipped_compaction_does_not_stop_the_turn() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("s.journal.jsonl");
        let mut journal = turn_journal::Journal::create(&path).unwrap();
        journal.accept_only_n_writes_for_test(0);
        let report = crate::compaction::CompactionReport {
            applied: false,
            skipped: Some(crate::compaction::SkipReason::NotNeeded),
            stats: None,
        };
        assert!(record_compaction(&report, &[], &mut journal).is_ok());
    }
}

#[cfg(test)]
mod autonomous_journal_tests {
    use super::*;

    fn reply_with(name: &str, args: serde_json::Value) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>",
            args = args
        )
    }

    /// The one allowed journal write records the assistant reply. The tool
    /// request is the write that fails, and dispatch does not run.
    #[test]
    fn journal_write_failure_stops_an_autonomous_turn_before_dispatch() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("s.journal.jsonl");
        let mut journal = turn_journal::Journal::create(&path).unwrap();
        let mut messages = Vec::new();
        journal
            .push(
                &mut messages,
                json!({"role": "user", "content": "do the milestone"}),
            )
            .unwrap();
        journal.accept_only_n_writes_for_test(1);
        let assistant = reply_with("run_command", json!({"command": "rm x"}));
        let mut ran = false;
        let result = apply_autonomous_reply(&mut journal, &mut messages, &assistant, |_, _| {
            ran = true;
            json!({})
        });
        assert!(
            matches!(result, Err(turn_journal::TurnStop::PersistFailed(_))),
            "{result:?}"
        );
        assert!(!ran, "dispatch ran after the journal write failed");
        assert!(
            !messages.iter().any(|m| m["content"]
                .as_str()
                .unwrap_or("")
                .contains("<tool_response")),
            "a tool result was added after the write failed"
        );
        drop(journal);
        let rep = turn_journal::replay(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(rep.in_doubt.is_empty(), "the tool request was not recorded");
        assert_eq!(rep.messages.len(), 2, "user message and assistant reply");
        assert!(rep.messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("run_command"));
    }

    #[test]
    fn an_autonomous_tool_request_is_on_disk_before_it_runs() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("s.journal.jsonl");
        let mut journal = turn_journal::Journal::create(&path).unwrap();
        let mut messages = Vec::new();
        journal
            .push(
                &mut messages,
                json!({"role": "user", "content": "milestone"}),
            )
            .unwrap();
        let assistant = reply_with(
            "goal",
            json!({"action": "milestone_done", "id": "g", "milestone_id": 1}),
        );
        let path_for_dispatch = path.clone();
        let step = apply_autonomous_reply(&mut journal, &mut messages, &assistant, |name, args| {
            let rep = turn_journal::replay(&std::fs::read_to_string(&path_for_dispatch).unwrap())
                .unwrap();
            assert_eq!(
                rep.in_doubt.len(),
                1,
                "the request must be durable before dispatch"
            );
            assert_eq!(rep.in_doubt[0].name, name);
            assert_eq!(rep.in_doubt[0].effect, turn_journal::Effect::Effectful);
            assert_eq!(args["action"], "milestone_done");
            json!({"ok": true})
        })
        .unwrap();
        assert!(matches!(
            step,
            AutonomousStep::Acted {
                milestone_done: true
            }
        ));
        assert!(messages.iter().any(|m| {
            m["content"]
                .as_str()
                .is_some_and(|c| c.contains("<tool_response name=\"goal\">"))
        }));
        drop(journal);
        let rep = turn_journal::replay(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(rep.in_doubt.is_empty());
    }
}
