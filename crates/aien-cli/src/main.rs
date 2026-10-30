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
    print_banner();

    let session_id = format!("session-{}", chrono::Utc::now().timestamp());
    let platform = crate::platform::PlatformContext::detect();
    let sessions_dir = platform.home_dir.join("basecamp/sessions");
    let _ = fs::create_dir_all(&sessions_dir);
    start_checkpoints(&platform.home_dir, &session_id);

    // Mandatory Nesting Ritual: Circle, Check Wind, Leave Scent, Flatten Terrain
    let nesting = perform_nesting_ritual(&session_id, true).await;

    let cwd = std::env::current_dir().unwrap_or_else(|_| platform.home_dir.clone());
    let rules = crate::rules::discover_rules(&cwd);
    let rules_prompt = crate::rules::format_rules_for_prompt(&rules);

    let mut full_system_prompt = get_system_prompt();
    if !rules_prompt.is_empty() {
        full_system_prompt.push_str(&rules_prompt);
    }

    let client = ChatClient::new(None, None);
    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": full_system_prompt}),
        json!({"role": "user", "content": format!("<system_grounding>\n{}\n</system_grounding>\nPerform your mandatory nesting initiation. Confirm your verified terrain, wind, scent mark, and operational readiness before taking commands.", nesting.grounding_context)}),
    ];

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
    messages.push(json!({"role": "assistant", "content": init_resp}));

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

                crate::checkpoint::begin_turn();
                messages.push(json!({"role": "user", "content": user_turn}));

                report_compaction(&compactor.compact_if_needed(&mut messages));

                // Multi-turn tool execution loop
                let mut max_tool_steps = 10;
                while max_tool_steps > 0 {
                    max_tool_steps -= 1;

                    print!("\n{}", "AIEN: ".magenta().bold());
                    let mut assistant_resp = match client.stream_turn(&messages, true).await {
                        Ok(text) => text,
                        Err(e) => {
                            println!(
                                "\n{}",
                                format!("Error during inference: {}", e).red().bold()
                            );
                            break;
                        }
                    };

                    hook_registry.run_post_turn(&mut assistant_resp).await;
                    messages.push(json!({"role": "assistant", "content": assistant_resp.clone()}));

                    report_compaction(&compactor.compact_if_needed(&mut messages));

                    let tool_calls = match extract_tool_calls(&assistant_resp) {
                        Ok(calls) => calls,
                        Err(error) => {
                            eprintln!("{}", error);
                            break;
                        }
                    };
                    if !tool_calls.is_empty() {
                        for (tool_name, tool_args) in tool_calls {
                            let tool_result = dispatch_tool(&tool_name, &tool_args);
                            let result_str =
                                serde_json::to_string(&tool_result).unwrap_or_default();
                            messages.push(json!({
                                "role": "user",
                                "content": format!("<tool_response name=\"{}\">\n{}\n</tool_response>", tool_name, result_str)
                            }));
                        }
                    } else {
                        // Completed turn
                        break;
                    }
                }

                // Save session turn to sessions/
                let session_path = sessions_dir.join(format!("{}.json", session_id));
                let _ = fs::write(
                    &session_path,
                    serde_json::to_string_pretty(&messages).unwrap_or_default(),
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
    let mut messages = vec![
        json!({"role": "system", "content": full_system_prompt}),
        json!({"role": "user", "content": format!("<system_grounding>\n{}\n</system_grounding>\n{}", nesting.grounding_context, turn_prompt)}),
    ];

    let mut max_tool_steps = 10;
    while max_tool_steps > 0 {
        max_tool_steps -= 1;
        let mut assistant_resp = match client.stream_turn(&messages, true).await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Error: {}", e);
                return;
            }
        };

        hook_registry.run_post_turn(&mut assistant_resp).await;
        messages.push(json!({"role": "assistant", "content": assistant_resp.clone()}));

        let tool_calls = match extract_tool_calls(&assistant_resp) {
            Ok(calls) => calls,
            Err(error) => {
                eprintln!("{}", error);
                return;
            }
        };
        if !tool_calls.is_empty() {
            for (tool_name, tool_args) in tool_calls {
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
    }

    let session_path = sessions_dir.join(format!("{}.json", session_id));
    let _ = fs::write(
        &session_path,
        serde_json::to_string_pretty(&messages).unwrap_or_default(),
    );
}

pub async fn run_autonomous_goal(goal_id: &str) {
    let manifest = goals::load_goals();
    let goal = match goals::goal_index(&manifest.goals, goal_id).map(|index| &manifest.goals[index])
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

        if let Err(error) = execute_autonomous_milestone(
            &client,
            &mut messages,
            &goal.id,
            m.id,
            &hook_registry,
            dispatch_tool,
        )
        .await
        {
            eprintln!("Milestone {} not completed: {}", m.id, error);
            return;
        }

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

trait MilestoneChat {
    async fn reply(&self, messages: &[Value]) -> Result<String, String>;
}

impl MilestoneChat for ChatClient {
    async fn reply(&self, messages: &[Value]) -> Result<String, String> {
        self.stream_turn(messages, true).await
    }
}

async fn execute_autonomous_milestone(
    client: &impl MilestoneChat,
    messages: &mut Vec<Value>,
    goal_id: &str,
    milestone_id: usize,
    hook_registry: &crate::hooks::HookRegistry,
    mut dispatch: impl FnMut(&str, &Value) -> Value,
) -> Result<(), String> {
    for _ in 0..15 {
        print!(
            "\n{}",
            format!("AIEN [Milestone {}]: ", milestone_id)
                .magenta()
                .bold()
        );
        let mut response = client.reply(messages).await?;
        hook_registry.run_post_turn(&mut response).await;
        messages.push(json!({"role":"assistant","content":response}));
        let calls = extract_tool_calls(&response)?;
        if calls.is_empty() {
            return Err("reply did not request milestone completion".into());
        }
        for (name, args) in calls {
            let action = args["action"].as_str();
            let completion_request = name == "goal"
                && matches!(
                    action,
                    Some("milestone_done" | "advance" | "done" | "complete")
                );
            if completion_request
                && (!matches!(action, Some("milestone_done" | "advance"))
                    || args["id"].as_str() != Some(goal_id)
                    || args["milestone_id"].as_u64() != Some(milestone_id as u64))
            {
                return Err(
                    "completion request does not identify the active goal/milestone".into(),
                );
            }
            let result = dispatch(&name, &args);
            messages.push(json!({"role":"user","content":format!("<tool_response name=\"{}\">\n{}\n</tool_response>",name,result)}));
            if completion_request {
                if result["status"].as_str() == Some("ok") {
                    return Ok(());
                }
                return Err(format!("milestone completion refused: {}", result));
            }
        }
    }
    Err("milestone exhausted its 15 tool-bearing turns without completion".into())
}

#[cfg(test)]
mod autonomous_tests {
    use super::*;
    struct FakeChat(std::cell::RefCell<std::collections::VecDeque<String>>);
    impl MilestoneChat for FakeChat {
        async fn reply(&self, _: &[Value]) -> Result<String, String> {
            Ok(self
                .0
                .borrow_mut()
                .pop_front()
                .expect("unexpected inference"))
        }
    }
    fn request(args: Value) -> String {
        format!(
            "<tool_call>{}</tool_call>",
            json!({"name":"goal","arguments":args})
        )
    }
    async fn run(
        replies: Vec<String>,
        dispatch: impl FnMut(&str, &Value) -> Value,
    ) -> Result<(), String> {
        execute_autonomous_milestone(
            &FakeChat(std::cell::RefCell::new(replies.into())),
            &mut Vec::new(),
            "active",
            7,
            &crate::hooks::HookRegistry::default_sovereign_registry(),
            dispatch,
        )
        .await
    }
    #[tokio::test]
    async fn round3_milestone_exhaustion_refuses_completion() {
        let reply = request(json!({"action":"list"}));
        let mut count = 0;
        let result = run(vec![reply; 15], |_, _| {
            count += 1;
            json!({"status":"ok"})
        })
        .await;
        assert!(result.is_err(), "exhaustion completed");
        assert_eq!(count, 15);
    }
    #[tokio::test]
    async fn round3_milestone_empty_and_final_text_refuse_completion() {
        for reply in ["", "I think the milestone is done."] {
            assert!(run(vec![reply.into()], |_, _| panic!("no dispatch"))
                .await
                .is_err());
        }
    }
    #[tokio::test]
    async fn round3_milestone_denied_request_refuses_completion() {
        for refusal in [
            json!({"error":"SECURITY DENIAL: policy refused"}),
            json!({"status":"error","error":"write goals failed"}),
        ] {
            let result = run(
                vec![request(
                    json!({"action":"milestone_done","id":"active","milestone_id":7}),
                )],
                |_, _| refusal.clone(),
            )
            .await;
            assert!(
                result.is_err(),
                "denied/failed completion request succeeded"
            );
        }
    }
    #[tokio::test]
    async fn round3_milestone_invalid_request_refuses_completion() {
        for args in [
            json!({"action":"milestone_done","id":"other","milestone_id":7}),
            json!({"action":"milestone_done","id":"active","milestone_id":8}),
            json!({"action":"milestone_done","id":"active"}),
            json!({"action":"done","id":"active"}),
        ] {
            let mut dispatched = false;
            assert!(run(vec![request(args)], |_, _| {
                dispatched = true;
                json!({"status":"ok"})
            })
            .await
            .is_err());
            assert!(!dispatched, "invalid completion request reached its effect");
        }
    }
    #[tokio::test]
    async fn round3_milestone_valid_request_completes() {
        let mut count = 0;
        run(
            vec![request(
                json!({"action":"milestone_done","id":"active","milestone_id":7}),
            )],
            |_, _| {
                count += 1;
                json!({"status":"ok"})
            },
        )
        .await
        .unwrap();
        assert_eq!(count, 1);
    }
}

/// Prints the outcome of a compaction pass. Skips are reported too, so a
/// refusal (detached tool result, store failure) is never silent.
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
