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

use colored::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use crate::crumbs::{
    format_dir_crumb_tui, leave_dir_whisper, load_or_init_dir_crumb, record_directory_crumb,
    save_dir_crumb, sniff_dir_crumb, CrumbPurpose,
};
use crate::safety::{contains_unmasked_secret, is_secret_leak_path};
use crate::ui::{print_safety_prompt, print_tool_done, print_tool_start};
use crate::vault::{redact_secrets, vault_dispatch_tool};
use chrono::Utc;

pub fn dispatch_tool(name: &str, args: &Value) -> Value {
    dispatch_tool_with(name, args, &crate::approval::GateContext::from_env())
}

pub fn dispatch_tool_with(name: &str, args: &Value, ctx: &crate::approval::GateContext) -> Value {
    let start = Instant::now();
    let summary = match name {
        "run_command" => args.get("command").and_then(Value::as_str).unwrap_or(""),
        "view_file" => args.get("path").and_then(Value::as_str).unwrap_or(""),
        "write_to_file" => args.get("path").and_then(Value::as_str).unwrap_or(""),
        "replace_file_content" => args.get("path").and_then(Value::as_str).unwrap_or(""),
        "list_dir" => args.get("path").and_then(Value::as_str).unwrap_or(""),
        "grep_search" | "search_directory" => args
            .get("query")
            .or_else(|| args.get("Query"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        "find_by_name" | "find_file" => args
            .get("pattern")
            .or_else(|| args.get("name"))
            .or_else(|| args.get("Pattern"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        "ask_question" => args.get("question").and_then(Value::as_str).unwrap_or(""),
        "create_dir" => args.get("path").and_then(Value::as_str).unwrap_or(""),
        "crumb" => args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("survey"),
        "vault" => args.get("action").and_then(Value::as_str).unwrap_or("list"),
        "goal" => args.get("action").and_then(Value::as_str).unwrap_or("list"),
        "skill" | "skills" => args.get("action").and_then(Value::as_str).unwrap_or("list"),
        "context7" => args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("resolve"),
        "cortex" => args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("search"),
        "sandbox" => args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("status"),
        "browser" => args.get("action").and_then(Value::as_str).unwrap_or("test"),
        "invoke_subagent" => args.get("role").and_then(Value::as_str).unwrap_or(""),
        "subagent" | "subagents" => args.get("action").and_then(Value::as_str).unwrap_or("list"),
        "socratic" | "socratic_inquiry" => args
            .get("question")
            .or_else(|| args.get("inquiry"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        "adapter" | "model_adapter" | "model_adapters" => {
            args.get("action").and_then(Value::as_str).unwrap_or("list")
        }
        _ => "",
    };

    // 1. The approval gate (sc#667): floors first in every mode, then the
    // nine-tier SafetyEngine's denials, mode, allowlist, presence, danger,
    // guardian and person. See approval.rs for the ordering rule.
    if let Err(res) = gate_tool(name, args, summary, ctx) {
        print_tool_start(name, summary);
        print_tool_done(name, start.elapsed().as_millis(), false);
        return res;
    }

    print_tool_start(name, summary);

    // 2. Lifecycle Pre-tool Hook
    let hook_registry = crate::hooks::HookRegistry::default_sovereign_registry();
    let mut mutable_args = args.clone();
    let short_circuit = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current()
            .block_on(hook_registry.run_pre_tool(name, &mut mutable_args))
    });
    if let Ok(Some(sc_result)) = short_circuit {
        print_tool_done(name, start.elapsed().as_millis(), true);
        return sc_result;
    }

    let (res, success) = match name {
        "run_command" => {
            let cmd = args
                .get("command")
                .or_else(|| args.get("CommandLine"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let cwd = args
                .get("cwd")
                .or_else(|| args.get("Cwd"))
                .and_then(Value::as_str)
                .unwrap_or(".");

            // The approval gate decided before this point (sc#667); a second
            // prompt here would re-ask the person and override yolo/off.
            exec_command(cmd, cwd)
        }
        "view_file" => {
            let path = args
                .get("path")
                .or_else(|| args.get("AbsolutePath"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let start_line = args
                .get("start_line")
                .or_else(|| args.get("StartLine"))
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            let end_line = args
                .get("end_line")
                .or_else(|| args.get("EndLine"))
                .and_then(Value::as_u64)
                .unwrap_or(500) as usize;
            exec_view_file(path, start_line, end_line)
        }
        "write_to_file" | "create_file" => {
            let path = args
                .get("path")
                .or_else(|| args.get("target_file"))
                .or_else(|| args.get("TargetFile"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = args
                .get("content")
                .or_else(|| args.get("CodeContent"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let overwrite = args
                .get("overwrite")
                .or_else(|| args.get("Overwrite"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let intent = args
                .get("intent")
                .and_then(Value::as_str)
                .unwrap_or("Writing file");
            let vector = args
                .get("vector")
                .and_then(Value::as_str)
                .unwrap_or("Next action");

            if is_secret_leak_path(path) {
                (
                    json!({"error": "REJECTED BY SOVEREIGN VAULT POLICY: Storing secrets or API keys in .env or plaintext credential files is strictly forbidden. Use 'atlas-vault add <KEY>' to store keys in the TPM-bound hardware vault, and retrieve them dynamically in memory via 'atlas-vault get <KEY>'."}),
                    false,
                )
            } else if let Some(reason) = contains_unmasked_secret(content) {
                (
                    json!({"error": format!("REJECTED BY SOVEREIGN VAULT POLICY: {}. Never commit or write plaintext keys into files. Use 'atlas-vault add <KEY>'.", reason)}),
                    false,
                )
            } else {
                // System paths were already put to the person by the
                // approval gate (sc#667); one decision, never two.
                recorded_write("write_to_file", path, || {
                    exec_write_file(path, content, overwrite, intent, vector)
                })
            }
        }
        "replace_file_content" | "edit_file" => {
            let path = args
                .get("path")
                .or_else(|| args.get("target_file"))
                .or_else(|| args.get("TargetFile"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let target = args
                .get("target")
                .or_else(|| args.get("target_content"))
                .or_else(|| args.get("TargetContent"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let replacement = args
                .get("replacement")
                .or_else(|| args.get("replacement_content"))
                .or_else(|| args.get("ReplacementContent"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let intent = args
                .get("intent")
                .and_then(Value::as_str)
                .unwrap_or("Editing file content");
            let vector = args
                .get("vector")
                .and_then(Value::as_str)
                .unwrap_or("Next action");

            if is_secret_leak_path(path) {
                (
                    json!({"error": "REJECTED BY SOVEREIGN VAULT POLICY: Modifying .env or secret files on disk is strictly forbidden. Use atlas-vault."}),
                    false,
                )
            } else if let Some(reason) = contains_unmasked_secret(replacement) {
                (
                    json!({"error": format!("REJECTED BY SOVEREIGN VAULT POLICY: {}. Never commit or write plaintext keys into files. Use 'atlas-vault add <KEY>'.", reason)}),
                    false,
                )
            } else {
                recorded_write("replace_file_content", path, || {
                    exec_replace_file(path, target, replacement, intent, vector)
                })
            }
        }
        "list_dir" | "list_directory" => {
            let path = args
                .get("path")
                .or_else(|| args.get("directory_path"))
                .or_else(|| args.get("DirectoryPath"))
                .and_then(Value::as_str)
                .unwrap_or(".");
            exec_list_dir(path)
        }
        "grep_search" | "search_directory" => {
            let query = args
                .get("query")
                .or_else(|| args.get("Query"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let path = args
                .get("path")
                .or_else(|| args.get("search_path"))
                .or_else(|| args.get("SearchPath"))
                .and_then(Value::as_str)
                .unwrap_or(".");
            exec_grep_search(query, path)
        }
        "find_by_name" | "find_file" => {
            let path = args
                .get("path")
                .or_else(|| args.get("search_directory"))
                .or_else(|| args.get("SearchDirectory"))
                .and_then(Value::as_str)
                .unwrap_or(".");
            let pattern = args
                .get("pattern")
                .or_else(|| args.get("name"))
                .or_else(|| args.get("Pattern"))
                .and_then(Value::as_str)
                .unwrap_or("*");
            exec_find_by_name(path, pattern)
        }
        "ask_question" => {
            let q = args.get("question").and_then(Value::as_str).unwrap_or("");
            let options = args.get("options").and_then(Value::as_array);
            exec_ask_question(q, options)
        }
        "create_dir" => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            let purpose = args
                .get("purpose")
                .and_then(Value::as_str)
                .unwrap_or("Workspace directory");
            let lifecycle = args.get("lifecycle").and_then(Value::as_str);
            exec_create_dir(path, purpose, lifecycle)
        }
        "crumb" => exec_crumb_tool(args),
        "skill" | "skills" => (crate::skills::skills_dispatch_tool(args), true),
        "context7" => {
            let res = crate::context7::context7_dispatch_tool(args);
            let ok = res.get("status").and_then(Value::as_str) == Some("ok");
            (res, ok)
        }
        "goal" => goal_tool_result(crate::goals::goals_dispatch_tool(args)),
        "cortex" => {
            let res = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(crate::cortex::cortex_dispatch_tool(args))
            });
            (res, true)
        }
        "vault" => (vault_dispatch_tool(args), true),
        "walkthrough" => {
            let action = args
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("render");
            if action == "save" {
                let platform = crate::platform::PlatformContext::detect();
                let dest = platform.home_dir.join("basecamp/WALKTHROUGH.md");
                match crate::walkthrough::generate_and_save_walkthrough_md(&dest) {
                    Ok(msg) => (
                        json!({"status": "ok", "message": msg, "path": &dest.display().to_string()}),
                        true,
                    ),
                    Err(e) => (json!({"error": e}), false),
                }
            } else {
                let card = crate::walkthrough::render_walkthrough_tui();
                (json!({"status": "ok", "walkthrough": card}), true)
            }
        }
        "hive" => (crate::hive::hive_dispatch_tool(args), true),
        "socratic" | "socratic_inquiry" => {
            let question = args
                .get("question")
                .or_else(|| args.get("inquiry"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let parent_id = args.get("parent_id").and_then(Value::as_str);
            if question.is_empty() {
                (json!({"error": "Socratic question cannot be empty"}), false)
            } else {
                match spark_hive::CombStore::open_default() {
                    Ok(store) => {
                        match spark_hive::emit_socratic_comb(&store, question, parent_id) {
                            Ok(comb) => (
                                json!({
                                    "status": "emitted",
                                    "comb_id": comb.id,
                                    "q": comb.q,
                                    "r": comb.r,
                                    "role": "socratic",
                                    "intent": "branch",
                                    "question": question,
                                    "message": "Socratic comb placed on hexagonal honeycomb lattice"
                                }),
                                true,
                            ),
                            Err(e) => (json!({"error": e.to_string()}), false),
                        }
                    }
                    Err(e) => (
                        json!({"error": format!("Failed to open hive database: {}", e)}),
                        false,
                    ),
                }
            }
        }
        "adapter" | "model_adapter" | "model_adapters" => (adapter_dispatch_tool(args), true),
        "sandbox" => (crate::sandbox::sandbox_dispatch_tool(args), true),
        "browser" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("test");
            let prompt = args.get("prompt").and_then(Value::as_str);
            (crate::sandbox::browser_dispatch_tool(action, prompt), true)
        }
        "invoke_subagent" => {
            let role = args
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("Subagent");
            let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("");
            let res = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(crate::subagents::invoke_subagent_from_root(role, prompt))
            });
            subagent_tool_result(res)
        }
        "subagent" | "subagents" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("list");
            match action {
                "list" => (crate::subagents::list_subagents(), true),
                "view" => {
                    let id = args.get("id").and_then(Value::as_str).unwrap_or("");
                    (crate::subagents::view_subagent(id), true)
                }
                _ => (
                    json!({"error": format!("Unknown subagent action: {}", action)}),
                    false,
                ),
            }
        }
        other => (json!({"error": format!("Unknown tool: {}", other)}), false),
    };

    // 3. Lifecycle Post-tool Hook
    let mut final_res = res;
    let hook_registry = crate::hooks::HookRegistry::default_sovereign_registry();
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(hook_registry.run_post_tool(
            name,
            args,
            &mut final_res,
        ))
    });

    // 4. Lifecycle On-tool-error Hook
    if !success {
        if let Some(err_str) = final_res.get("error").and_then(Value::as_str) {
            let recovery = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(hook_registry.run_on_tool_error(name, args, err_str))
            });
            if let Some(rec) = recovery {
                final_res = rec;
            }
        }
    }

    let _ = record_effect_receipt(name, args, &final_res, success);

    let elapsed = start.elapsed().as_millis();
    print_tool_done(name, elapsed, success);
    final_res
}

fn goal_tool_result(result: Value) -> (Value, bool) {
    let success = result["status"].as_str() == Some("ok");
    (result, success)
}

pub(crate) fn subagent_tool_result(res: Value) -> (Value, bool) {
    let success = res.get("status").and_then(Value::as_str) == Some("completed");
    (res, success)
}

/// Persist a content-addressed, secret-free receipt for every tool effect.
/// Arguments and results are hashed, never copied into the receipt.
/// Writes one effect receipt under `$AIEN_PROVENANCE_DIR` and returns its path
/// (None when it could not be written). NEXT-PHASE-1 `aien compose` cites the
/// returned file by its sha256.
pub(crate) fn record_effect_receipt(
    tool: &str,
    args: &Value,
    result: &Value,
    success: bool,
) -> Option<std::path::PathBuf> {
    let digest = |value: &Value| {
        let bytes = serde_json::to_vec(value).unwrap_or_default();
        format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
    };
    let receipt = json!({
        "version": 1,
        "tool": tool,
        "success": success,
        "arguments_digest": digest(args),
        "result_digest": digest(result),
        "policy": "default_sovereign_engine",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    let dir = aien_runtime::objective::provenance_root();
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let id = format!(
        "{}-{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        std::process::id()
    );
    let path = dir.join(format!("{}.json", id));
    let bytes = serde_json::to_vec_pretty(&receipt).ok()?;
    std::fs::write(&path, bytes).ok()?;
    Some(path)
}

fn validate_tool_path(path: &str) -> Result<std::path::PathBuf, String> {
    let platform = crate::platform::PlatformContext::detect();
    let mut allowed = platform.default_workspaces;
    allowed.push(std::path::PathBuf::from("/tmp"));
    crate::safety::validate_workspace_path(path, &allowed)
}

fn exec_command(cmd: &str, cwd: &str) -> (Value, bool) {
    if let Err(err) = validate_tool_path(cwd) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: Invalid command working directory: {}", err)}),
            false,
        );
    }

    let out = Command::new("bash")
        .args(["-c", cmd])
        .current_dir(cwd)
        .output();

    match out {
        Ok(output) => {
            let raw_stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let raw_stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let stdout = redact_secrets(&raw_stdout);
            let stderr = redact_secrets(&raw_stderr);
            let code = output.status.code().unwrap_or(-1);
            let success = output.status.success();
            (
                json!({
                    "exit_code": code,
                    "stdout": stdout,
                    "stderr": stderr
                }),
                success,
            )
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_view_file(path: &str, start_line: usize, end_line: usize) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let p = Path::new(path);
    // Sniff directory breadcrumbs on this file
    let crumb_notice = sniff_dir_crumb(p, "AIEN");

    // Record view in directory crumb
    record_directory_crumb(
        "AIEN",
        "active-session",
        p,
        "view",
        "Inspected file lines",
        "",
    );

    match fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();
            let start = if start_line > 0 { start_line - 1 } else { 0 };
            let end = std::cmp::min(end_line, total_lines);

            if start >= total_lines && total_lines > 0 {
                return (
                    json!({"error": "start_line exceeds file length", "total_lines": total_lines}),
                    false,
                );
            }

            let slice = &lines[start..end];
            let numbered: Vec<String> = slice
                .iter()
                .enumerate()
                .map(|(idx, line)| {
                    let redacted = redact_secrets(line);
                    format!("{}: {}", start + idx + 1, redacted)
                })
                .collect();

            let mut resp = json!({
                "path": path,
                "lines": numbered.join("\n"),
                "total_lines": total_lines
            });

            if let Some(notice) = crumb_notice {
                resp["peer_breadcrumb_notice"] = json!(notice);
            }

            (resp, true)
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

/// Run the approval gate for one request. `Err` carries the tool result to
/// hand back to the model when the request does not run.
fn gate_tool(
    name: &str,
    args: &Value,
    summary: &str,
    ctx: &crate::approval::GateContext,
) -> Result<(), Value> {
    use crate::approval::{after_guardian, decide, Gate, Verdict};
    let base = crate::safety::SafetyEngine::default_sovereign_engine().evaluate(name, args);
    let mut verdict = decide(&Gate {
        tool: name,
        args,
        base: &base,
        mode: ctx.mode,
        presence: ctx.presence,
        policy: &ctx.policy,
        facts: &ctx.facts,
        guardian_ready: crate::approval::guardian_ready(),
    });
    if let Verdict::AskGuardian { reason } = verdict {
        let answer = ask_guardian(name, summary, &reason);
        crate::approval::record_guardian(&answer);
        verdict = after_guardian(&answer, reason);
    }
    match verdict {
        Verdict::Allow { .. } => Ok(()),
        Verdict::Deny { by, reason } => Err(json!({
            "error": format!("SECURITY DENIAL [{:?}]: {}", by, reason)
        })),
        Verdict::AskGuardian { .. } => unreachable!("guardian answered above"),
        Verdict::AskPerson { reason } => {
            println!("{}", format!("⚠️  SAFETY GATE: {}", reason).yellow().bold());
            if print_safety_prompt(&format!("Tool: '{}' with summary: '{}'", name, summary)) {
                Ok(())
            } else {
                Err(json!({"error": "Operation aborted by operator safety gate"}))
            }
        }
    }
}

/// Ask the guardian model (smart mode). Any transport failure, timeout or
/// unparseable reply is `Failed`, which escalates to the person and counts
/// against the breaker.
fn ask_guardian(name: &str, summary: &str, reason: &str) -> crate::approval::GuardianAnswer {
    use crate::approval::GuardianAnswer;
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return GuardianAnswer::Failed("no async runtime for the guardian".into());
    };
    let messages = vec![
        json!({"role": "system", "content": crate::approval::GUARDIAN_PROMPT}),
        json!({"role": "user", "content": format!(
            "tool: {name}\nrequest: {}\nflagged because: {reason}",
            redact_secrets(summary)
        )}),
    ];
    let client = crate::client::ChatClient::new(None, None);
    let reply = tokio::task::block_in_place(|| {
        handle.block_on(client.stream_turn_with_limit(&messages, false, 32))
    });
    match reply {
        Ok(text) => crate::approval::parse_guardian_reply(&text),
        Err(err) => GuardianAnswer::Failed(err),
    }
}

/// Run a file tool's write as a recorded agent write (sc#667): prior content
/// kept, one ledger line after. Without an active chat session this is a
/// plain call.
fn recorded_write(tool: &str, path: &str, write: impl FnOnce() -> (Value, bool)) -> (Value, bool) {
    // Check confinement before checkpointing: before_write reads the target
    // and its directory, so even a later refused write must not reach it.
    let target = match validate_tool_path(path) {
        Ok(target) => target,
        Err(err) => {
            return (
                json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
                false,
            );
        }
    };
    // Keep the checkpoint identity consistent with the validated filesystem
    // target, rather than applying separate lexical normalization.
    // This remains pathname-based; descriptor-bound race handling is #754.
    match crate::checkpoint::record_agent_write(&target, tool, write) {
        Ok(((mut res, ok), warning)) => {
            if let (Some(w), Some(obj)) = (warning, res.as_object_mut()) {
                obj.insert("checkpoint_warning".into(), json!(w));
            }
            (res, ok)
        }
        Err(refusal) => (json!({"error": refusal}), false),
    }
}

fn exec_write_file(
    path: &str,
    content: &str,
    overwrite: bool,
    intent: &str,
    vector: &str,
) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let p = Path::new(path);
    if p.exists() && !overwrite {
        return (
            json!({"error": format!("File already exists: {}. Pass overwrite: true to replace.", path)}),
            false,
        );
    }

    if let Some(parent) = p.parent() {
        if !parent.exists() {
            let _ = fs::create_dir_all(parent);
            let file_name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let parent_purpose = CrumbPurpose {
                statement: format!("Auto-created to house file: {}", file_name),
                created_by: "AIEN".to_string(),
                session_id: "active-session".to_string(),
                created_at: Utc::now().to_rfc3339(),
                lifecycle: "permanent".to_string(),
            };
            let crumb = load_or_init_dir_crumb(parent, Some(parent_purpose));
            save_dir_crumb(&crumb);
        }
    }

    let crumb_notice = sniff_dir_crumb(p, "AIEN");

    match fs::write(path, content) {
        Ok(()) => {
            // Drop directory breadcrumb
            record_directory_crumb("AIEN", "active-session", p, "write", intent, vector);

            let mut resp = json!({
                "status": "ok",
                "path": path,
                "bytes": content.len()
            });
            if let Some(notice) = crumb_notice {
                resp["peer_breadcrumb_notice"] = json!(notice);
            }
            (resp, true)
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_replace_file(
    path: &str,
    target: &str,
    replacement: &str,
    intent: &str,
    vector: &str,
) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let p = Path::new(path);
    let crumb_notice = sniff_dir_crumb(p, "AIEN");

    match fs::read_to_string(path) {
        Ok(content) => {
            if !content.contains(target) {
                return (json!({"error": "target string not found in file"}), false);
            }
            let count = content.matches(target).count();
            if count > 1 {
                return (
                    json!({"error": format!("target string occurred {} times, must be unique", count)}),
                    false,
                );
            }
            let updated = content.replace(target, replacement);
            match fs::write(path, updated) {
                Ok(()) => {
                    // Record edit in directory crumb
                    record_directory_crumb("AIEN", "active-session", p, "edit", intent, vector);

                    let mut resp = json!({
                        "status": "ok",
                        "path": path,
                        "replaced": true
                    });
                    if let Some(notice) = crumb_notice {
                        resp["peer_breadcrumb_notice"] = json!(notice);
                    }
                    (resp, true)
                }
                Err(e) => (json!({"error": e.to_string()}), false),
            }
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_list_dir(path: &str) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let p = Path::new(path);
    let crumb_overview = format_dir_crumb_tui(p);

    match fs::read_dir(path) {
        Ok(entries) => {
            let mut list = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                list.push(json!({
                    "name": name,
                    "is_dir": is_dir,
                    "size_bytes": size
                }));
            }
            (
                json!({
                    "path": path,
                    "entries": list,
                    "directory_crumb": crumb_overview
                }),
                true,
            )
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_grep_search(query: &str, path: &str) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let out = Command::new("grep").args(["-rnI", query, path]).output();

    match out {
        Ok(output) => {
            let raw_stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let redacted_stdout = redact_secrets(&raw_stdout);
            let matches: Vec<String> = redacted_stdout
                .lines()
                .take(50)
                .map(|s| s.to_string())
                .collect();
            (
                json!({
                    "query": query,
                    "matches": matches,
                    "count": matches.len()
                }),
                true,
            )
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_crumb_tool(args: &Value) -> (Value, bool) {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("survey");
    let path_str = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let p = Path::new(path_str);

    match action {
        "survey" => {
            let view = format_dir_crumb_tui(p);
            (json!({"status": "ok", "crumb_view": view}), true)
        }
        "whisper" => {
            let msg = args.get("message").and_then(Value::as_str).unwrap_or("");
            let target_file = args.get("target_file").and_then(Value::as_str);
            let dir = if p.is_dir() {
                p
            } else {
                p.parent().unwrap_or(p)
            };
            leave_dir_whisper("AIEN", dir, target_file, msg);
            (
                json!({"status": "ok", "message": "Whisper recorded to directory .crumb"}),
                true,
            )
        }
        "init" => {
            let purpose_str = args
                .get("purpose")
                .and_then(Value::as_str)
                .unwrap_or("Workspace directory");
            let lifecycle = args
                .get("lifecycle")
                .and_then(Value::as_str)
                .unwrap_or("permanent");
            let p_dir = if p.is_dir() {
                p
            } else {
                p.parent().unwrap_or(p)
            };
            let _ = fs::create_dir_all(p_dir);
            let purpose_obj = CrumbPurpose {
                statement: purpose_str.to_string(),
                created_by: "AIEN".to_string(),
                session_id: "active-session".to_string(),
                created_at: Utc::now().to_rfc3339(),
                lifecycle: lifecycle.to_string(),
            };
            let mut crumb = load_or_init_dir_crumb(p_dir, Some(purpose_obj));
            crumb.description = purpose_str.to_string();
            save_dir_crumb(&crumb);
            (
                json!({"status": "ok", "message": format!("Initialized .crumb with purpose for {}", p_dir.display())}),
                true,
            )
        }
        "record" => {
            let intent = args
                .get("intent")
                .and_then(Value::as_str)
                .unwrap_or("Working in directory");
            let vector = args
                .get("vector")
                .and_then(Value::as_str)
                .unwrap_or("Next milestone");
            record_directory_crumb("AIEN", "tool-call", p, "record", intent, vector);
            (
                json!({"status": "ok", "message": "Directory crumb recorded"}),
                true,
            )
        }
        other => (
            json!({"error": format!("Unknown crumb action: {}", other)}),
            false,
        ),
    }
}

fn exec_create_dir(path: &str, purpose: &str, lifecycle: Option<&str>) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let p = Path::new(path);
    if path.is_empty() {
        return (json!({"error": "Path cannot be empty"}), false);
    }

    if let Err(e) = fs::create_dir_all(p) {
        return (
            json!({"error": format!("Failed to create directory {}: {}", path, e)}),
            false,
        );
    }

    let purpose_obj = CrumbPurpose {
        statement: purpose.to_string(),
        created_by: "AIEN".to_string(),
        session_id: "active-session".to_string(),
        created_at: Utc::now().to_rfc3339(),
        lifecycle: lifecycle.unwrap_or("permanent").to_string(),
    };

    let mut crumb = load_or_init_dir_crumb(p, Some(purpose_obj));
    crumb.description = purpose.to_string();
    save_dir_crumb(&crumb);

    // Record scent mark into crumb
    record_directory_crumb(
        "AIEN",
        "active-session",
        p,
        "create_dir",
        purpose,
        "Directory initialized with purpose",
    );

    let crumb_overview = format_dir_crumb_tui(p);

    (
        json!({
            "status": "ok",
            "path": path,
            "purpose": purpose,
            "crumb_created": true,
            "directory_crumb": crumb_overview
        }),
        true,
    )
}

pub fn adapter_dispatch_tool(args: &Value) -> Value {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("list");
    let target = args
        .get("target")
        .or_else(|| args.get("model"))
        .or_else(|| args.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let engine_str = args.get("engine").and_then(Value::as_str);
    let engine_override = engine_str.and_then(spark_hive::UpstreamEngine::from_name);

    match action {
        "list" | "catalog" => {
            let adapters = spark_hive::get_catalog_adapters();
            let models = spark_hive::ConsumerModel::all();
            let engines = spark_hive::UpstreamEngine::all();
            json!({
                "status": "ok",
                "total_adapters": adapters.len(),
                "adapters": adapters,
                "models": models.iter().map(|m| json!({
                    "slug": m.slug(),
                    "name": m.display_name(),
                    "params_b": m.param_count_billions(),
                    "hardware": m.hardware_profile().description(),
                    "quantization": m.recommended_quantization(),
                })).collect::<Vec<_>>(),
                "engines": engines.iter().map(|e| json!({
                    "repo": e.repo(),
                    "language": e.primary_language(),
                    "description": e.description(),
                })).collect::<Vec<_>>()
            })
        }
        "evaluate" | "socratic" => {
            if target.is_empty() {
                return json!({"error": "Target model or adapter id must be provided for evaluate"});
            }
            let spec = spark_hive::find_or_create_adapter(target, engine_override);
            let Some(spec) = spec else {
                return json!({"error": format!("Model or adapter {} not found", target)});
            };
            let eval = spark_hive::evaluate_socratic_reflex(&spec.model, &spec.engine);
            json!({
                "status": "ok",
                "target": spec.id,
                "model": spec.model.slug(),
                "engine": spec.engine.repo(),
                "evaluation": eval
            })
        }
        "plan" | "pr" => {
            if target.is_empty() {
                return json!({"error": "Target model or adapter id must be provided for plan"});
            }
            let spec = spark_hive::find_or_create_adapter(target, engine_override);
            let Some(spec) = spec else {
                return json!({"error": format!("Model or adapter {} not found", target)});
            };
            let socratic = spark_hive::evaluate_socratic_reflex(&spec.model, &spec.engine);
            let default_sb = crate::platform::PlatformContext::detect()
                .home_dir
                .join("workspace/aien-sandbox");
            let default_sb_str = default_sb.to_string_lossy();
            let sandbox_path = args
                .get("sandbox_path")
                .and_then(Value::as_str)
                .unwrap_or(&default_sb_str);
            let telem = spark_hive::BenchmarkTelemetry::estimate_for_model(
                &spec.model,
                &spec.engine,
                sandbox_path,
                None,
            );
            let plan = spark_hive::generate_pr_plan(&spec, telem, socratic);
            json!({
                "status": "ok",
                "plan": plan
            })
        }
        "emit" => {
            if target.is_empty() {
                return json!({"error": "Target model or adapter id must be provided for emit"});
            }
            let spec = spark_hive::find_or_create_adapter(target, engine_override);
            let Some(spec) = spec else {
                return json!({"error": format!("Model or adapter {} not found", target)});
            };
            let socratic = spark_hive::evaluate_socratic_reflex(&spec.model, &spec.engine);
            let default_sb = crate::platform::PlatformContext::detect()
                .home_dir
                .join("workspace/aien-sandbox");
            let default_sb_str = default_sb.to_string_lossy();
            let sandbox_path = args
                .get("sandbox_path")
                .and_then(Value::as_str)
                .unwrap_or(&default_sb_str);
            let telem = spark_hive::BenchmarkTelemetry::estimate_for_model(
                &spec.model,
                &spec.engine,
                sandbox_path,
                None,
            );
            let plan = spark_hive::generate_pr_plan(&spec, telem, socratic);
            let parent_id = args.get("parent_id").and_then(Value::as_str);
            match spark_hive::CombStore::open_default() {
                Ok(store) => {
                    match spark_hive::emit_adapter_pipeline_combs(&store, &plan, parent_id) {
                        Ok(receipt) => json!({
                            "status": "emitted",
                            "receipt": receipt,
                            "plan": {
                                "adapter_id": plan.adapter_id,
                                "target_repo": plan.target_repo,
                                "branch": plan.branch,
                                "commit_message": plan.commit_message,
                                "pr_title": plan.pr_title,
                            }
                        }),
                        Err(e) => json!({"error": format!("Failed to emit combs: {}", e)}),
                    }
                }
                Err(e) => json!({"error": format!("Failed to open hive database: {}", e)}),
            }
        }
        "chains" | "lattice" => match spark_hive::CombStore::open_default() {
            Ok(store) => match spark_hive::list_adapter_pipeline_chains(&store) {
                Ok(chains) => json!({
                    "status": "ok",
                    "total_chains": chains.len(),
                    "chains": chains
                }),
                Err(e) => json!({"error": format!("Failed to query chains: {}", e)}),
            },
            Err(e) => json!({"error": format!("Failed to open hive database: {}", e)}),
        },
        _ => {
            json!({"error": format!("Unknown adapter action {}. Valid actions: list, evaluate, plan, emit, chains", action)})
        }
    }
}

fn exec_find_by_name(path: &str, pattern: &str) -> (Value, bool) {
    if let Err(err) = validate_tool_path(path) {
        return (
            json!({"error": format!("CONFINEMENT DENIAL: {}", err)}),
            false,
        );
    }
    let out = Command::new("find").args([path, "-name", pattern]).output();
    match out {
        Ok(output) => {
            let raw_stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let redacted = redact_secrets(&raw_stdout);
            let matches: Vec<String> = redacted.lines().take(50).map(|s| s.to_string()).collect();
            (
                json!({
                    "path": path,
                    "pattern": pattern,
                    "matches": matches,
                    "count": matches.len()
                }),
                true,
            )
        }
        Err(e) => (json!({"error": e.to_string()}), false),
    }
}

fn exec_ask_question(question: &str, options: Option<&Vec<Value>>) -> (Value, bool) {
    println!(
        "\n{}",
        format!("❓ Question from AIEN: {}", question)
            .yellow()
            .bold()
    );
    if let Some(opts) = options {
        for (i, opt) in opts.iter().enumerate() {
            if let Some(opt_str) = opt.as_str() {
                println!("  {}. {}", i + 1, opt_str.cyan());
            }
        }
    }
    print!("{}", "Your Answer ❯ ".cyan().bold());
    let _ = std::io::stdout().flush();
    let mut input = String::new();
    let _ = std::io::stdin().read_line(&mut input);
    let trimmed = input.trim().to_string();
    (
        json!({
            "status": "answered",
            "question": question,
            "answer": trimmed
        }),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round3_goal_tool_failure_flag_is_false() {
        for result in [
            json!({"status":"error","error":"write goals failed"}),
            json!({"error":"denied"}),
        ] {
            assert!(!goal_tool_result(result).1);
        }
        assert!(goal_tool_result(json!({"status":"ok"})).1);
    }

    // Tests that install the process-wide checkpoint session must serialize.
    static CHECKPOINT_SESSION_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    struct StopCheckpointSession;
    impl Drop for StopCheckpointSession {
        fn drop(&mut self) {
            crate::checkpoint::stop_session();
        }
    }

    #[cfg(unix)]
    #[test]
    fn denied_recorded_write_does_not_snapshot_or_invoke_write() {
        use std::os::unix::fs::symlink;
        let _serial = CHECKPOINT_SESSION_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _stop = StopCheckpointSession;
        let home = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir_in("/tmp").unwrap();
        let work = work.path().canonicalize().unwrap();
        std::fs::write(work.join("sentinel"), "SYNTHETIC_CHECKPOINT_SENTINEL").unwrap();
        let denied = work.join("dangling");
        symlink(work.join("absent-target"), &denied).unwrap();
        crate::checkpoint::start_scoped_session(home.path(), "denied-path", work.clone()).unwrap();
        crate::checkpoint::begin_turn();
        let called = std::cell::Cell::new(false);
        let (result, ok) = recorded_write("write_to_file", denied.to_str().unwrap(), || {
            called.set(true);
            (json!({"error": "synthetic later denial"}), false)
        });
        assert!(!ok);
        assert!(result["error"]
            .as_str()
            .unwrap()
            .contains("CONFINEMENT DENIAL"));
        assert!(!called.get(), "write closure ran before confinement denial");
        let store = crate::checkpoint::store_root(home.path());
        assert!(
            !store.join("objects").exists(),
            "denied write copied checkpoint content"
        );
        assert!(
            !store.join("snapshots").exists(),
            "denied write created a snapshot"
        );
        assert!(
            !store.join("ledger").exists(),
            "denied write created a ledger"
        );
        assert!(!work.join("absent-target").exists());
    }

    #[cfg(unix)]
    #[test]
    fn recorded_write_checkpoint_uses_validated_target_identity() {
        use std::os::unix::fs::symlink;
        let _serial = CHECKPOINT_SESSION_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _stop = StopCheckpointSession;
        let home = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir_in("/tmp").unwrap();
        let work = work.path().canonicalize().unwrap();
        let target = work.join("target");
        std::fs::write(&target, "before").unwrap();
        let alias = work.join("alias");
        symlink(&target, &alias).unwrap();
        crate::checkpoint::start_scoped_session(home.path(), "canonical-target", work).unwrap();
        crate::checkpoint::begin_turn();
        let (_, ok) = recorded_write("write_to_file", alias.to_str().unwrap(), || {
            std::fs::write(&alias, "after").unwrap();
            (json!({"status": "ok"}), true)
        });
        assert!(ok);
        let store =
            crate::checkpoint::CheckpointStore::new(crate::checkpoint::store_root(home.path()));
        let (entries, _) = store.load_ledger("canonical-target").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, target.to_str().unwrap());
    }

    #[test]
    fn effect_receipt_is_hashed_and_secret_free() {
        let _env = crate::objective::PROVENANCE_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("AIEN_PROVENANCE_DIR", dir.path());
        let _ = record_effect_receipt(
            "write_to_file",
            &json!({"path":"/tmp/receipt-test","content":"secret-value"}),
            &json!({"status":"ok"}),
            true,
        );
        let entry = std::fs::read_dir(dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let text = std::fs::read_to_string(entry.path()).unwrap();
        assert!(text.contains("arguments_digest"));
        assert!(!text.contains("secret-value"));
        std::env::remove_var("AIEN_PROVENANCE_DIR");
    }

    #[test]
    fn test_adapter_tool_dispatch_list() {
        let args = json!({"action": "list"});
        let res = adapter_dispatch_tool(&args);
        assert_eq!(res.get("status").unwrap().as_str().unwrap(), "ok");
        assert!(res.get("adapters").unwrap().as_array().unwrap().len() >= 7);
        assert_eq!(res.get("models").unwrap().as_array().unwrap().len(), 9);
    }

    #[test]
    fn test_adapter_tool_dispatch_evaluate() {
        let args = json!({"action": "evaluate", "target": "deepseek-r1-distill-qwen-7b"});
        let res = adapter_dispatch_tool(&args);
        assert_eq!(res.get("status").unwrap().as_str().unwrap(), "ok");
        let eval = res.get("evaluation").unwrap();
        assert!(eval.get("approved").unwrap().as_bool().unwrap());
    }

    #[test]
    fn test_adapter_tool_dispatch_plan() {
        let args = json!({"action": "plan", "target": "qwen2.5-coder-7b", "engine": "vllm"});
        let res = adapter_dispatch_tool(&args);
        assert_eq!(res.get("status").unwrap().as_str().unwrap(), "ok");
        let plan = res.get("plan").unwrap();
        assert_eq!(
            plan.get("target_repo").unwrap().as_str().unwrap(),
            "vllm-project/vllm"
        );
        assert!(plan
            .get("pr_script")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("gh repo fork"));
    }

    #[test]
    fn test_safety_engine_allows_adapter_and_socratic() {
        let safety = crate::safety::SafetyEngine::default_sovereign_engine();
        let adapter_args = json!({"action": "list"});
        let socratic_args = json!({"question": "Is local offline execution sovereign?"});

        assert_eq!(
            safety.evaluate("adapter", &adapter_args),
            crate::safety::SafetyDecision::Allow
        );
        assert_eq!(
            safety.evaluate("socratic", &socratic_args),
            crate::safety::SafetyDecision::Allow
        );
    }

    #[test]
    fn test_path_confinement_in_tool_dispatch() {
        let bad_view = json!({"path": "/etc/shadow"});
        let res = dispatch_tool("view_file", &bad_view);
        assert!(res
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("CONFINEMENT DENIAL"));

        let bad_write = json!({"path": "/usr/bin/evil_binary", "content": "malicious"});
        let res = dispatch_tool("write_to_file", &bad_write);
        assert!(res
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("CONFINEMENT DENIAL"));

        let bad_cmd = json!({"command": "ls", "cwd": "/etc"});
        let res = dispatch_tool("run_command", &bad_cmd);
        assert!(res
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("CONFINEMENT DENIAL"));
    }

    #[test]
    fn test_policy_approved_write_reaches_effect_handler() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let path = format!("/tmp/aien-policy-effect-{}", std::process::id());
            let args = json!({"path": path, "content": "approved", "overwrite": true});
            let result = dispatch_tool("write_to_file", &args);
            assert!(
                result.get("error").is_none(),
                "approved write failed: {result}"
            );
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "approved");
            let _ = std::fs::remove_file(path);
        });
    }

    fn gate_ctx(
        mode: crate::approval::ApprovalMode,
        deny: &[&str],
    ) -> crate::approval::GateContext {
        use crate::approval::*;
        GateContext {
            mode,
            presence: Presence::Person,
            policy: CompiledPolicy::compile(&OperatorPolicy {
                mode: None,
                deny: deny.iter().map(|s| s.to_string()).collect(),
                allow: Vec::new(),
            }),
            facts: RuntimeFacts {
                protected: Vec::new(),
                home: std::path::PathBuf::from("/tmp"),
                cwd: std::path::PathBuf::from("/tmp"),
            },
        }
    }

    /// sc#667 acceptance through the real dispatcher: an operator deny
    /// pattern stops a real command in every mode; without the pattern the
    /// same command runs (negative control).
    #[test]
    fn operator_deny_pattern_blocks_a_real_command_in_every_mode() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            for mode in crate::approval::ALL_MODES {
                let dir = tempfile::Builder::new()
                    .prefix("aien-sc667-deny-")
                    .tempdir_in("/tmp")
                    .unwrap();
                let marker = dir.path().join("marker");
                let args = json!({
                    "command": format!("touch {}", marker.display()),
                    "cwd": dir.path().to_string_lossy(),
                });
                let res =
                    dispatch_tool_with("run_command", &args, &gate_ctx(mode, &[r"touch .*marker"]));
                let err = res.get("error").and_then(Value::as_str).unwrap_or("");
                assert!(err.contains("OperatorDeny"), "{mode:?}: {res}");
                assert!(!marker.exists(), "{mode:?}: command ran despite the floor");

                let res = dispatch_tool_with("run_command", &args, &gate_ctx(mode, &[]));
                assert!(res.get("error").is_none(), "{mode:?}: {res}");
                assert!(marker.exists(), "{mode:?}: control command did not run");
            }
        });
    }

    /// sc#667 acceptance through the real dispatcher: agent writes go to the
    /// ledger; a rollback after mixed operator and agent edits restores only
    /// the agent's files.
    #[test]
    fn rollback_through_the_dispatcher_spares_operator_edits() {
        let _serial = CHECKPOINT_SESSION_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _stop = StopCheckpointSession;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let home = tempfile::tempdir().unwrap();
            let work = tempfile::Builder::new()
                .prefix("aien-sc667-work-")
                .tempdir_in("/tmp")
                .unwrap();
            let work = work.path().canonicalize().unwrap();
            let agent_file = work.join("agent.txt");
            let created = work.join("created.txt");
            let operator_file = work.join("operator.txt");
            std::fs::write(&agent_file, "original").unwrap();
            std::fs::write(&operator_file, "operator v0").unwrap();

            let session = format!("sc667-dispatch-{}", std::process::id());
            crate::checkpoint::start_scoped_session(home.path(), &session, work.clone()).unwrap();
            crate::checkpoint::begin_turn();
            let ctx = gate_ctx(crate::approval::ApprovalMode::Manual, &[]);
            for (p, text) in [(&agent_file, "agent edit"), (&created, "agent new")] {
                let args = json!({"path": p.to_string_lossy(), "content": text, "overwrite": true});
                let res = dispatch_tool_with("write_to_file", &args, &ctx);
                assert!(res.get("error").is_none(), "{res}");
            }
            let args = json!({
                "path": agent_file.to_string_lossy(),
                "target": "edit",
                "replacement": "edit twice",
            });
            let res = dispatch_tool_with("replace_file_content", &args, &ctx);
            assert!(res.get("error").is_none(), "{res}");
            crate::checkpoint::stop_session();

            std::fs::write(&operator_file, "operator v1").unwrap();

            let report = crate::checkpoint::rollback(home.path(), Some(1), Some(&session)).unwrap();
            assert_eq!(std::fs::read_to_string(&agent_file).unwrap(), "original");
            assert!(!created.exists());
            assert_eq!(
                std::fs::read_to_string(&operator_file).unwrap(),
                "operator v1"
            );
            assert!(report
                .spared
                .contains(&operator_file.to_string_lossy().into_owned()));
        });
    }
}
