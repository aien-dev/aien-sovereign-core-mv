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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::SystemTime;

use crate::client::{extract_tool_calls, ChatClient};
use crate::tools::dispatch_tool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentSpec {
    pub id: String,
    pub role: String,
    pub prompt: String,
    pub depth: usize,
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub id: String,
    pub role: String,
    pub depth: usize,
    pub parent_id: Option<String>,
    pub status: String,
    pub summary: String,
    pub transcript_path: Option<String>,
    pub transcript_error: Option<String>,
    pub steps_executed: usize,
    pub children_spawned: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentStepLog {
    pub step: usize,
    pub assistant_response: String,
    pub tool_calls: Vec<(String, Value)>,
    pub tool_results: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTranscript {
    pub spec: SubagentSpec,
    pub start_time: String,
    pub end_time: String,
    pub status: String,
    pub summary: String,
    pub steps: Vec<SubagentStepLog>,
    pub children: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubagentOutcome {
    Completed,
    Failed,
    Exhausted,
}

impl SubagentOutcome {
    fn status(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Exhausted => "exhausted",
        }
    }
}

pub fn get_subagent_system_prompt(spec: &SubagentSpec) -> String {
    let mut p = String::new();
    p.push_str(&format!(
        "You are an autonomous AIEN Subagent with specialized role: '{}'.\n",
        spec.role
    ));
    p.push_str(&format!(
        "Execution Depth: {} (Max allowed: 3). Assigned by Parent Coordinator: {}.\n",
        spec.depth,
        spec.parent_id.as_deref().unwrap_or("Root")
    ));
    p.push_str("Host: NVIDIA DGX Spark (Grace Blackwell GB10).\n");
    let ws = crate::platform::PlatformContext::detect()
        .home_dir
        .join("workspace/aien-sovereign-core");
    p.push_str(&format!(
        "Primary Workspace: {}

",
        ws.display()
    ));
    p.push_str("OBJECTIVE:\n");
    p.push_str(&format!(
        "Execute the assigned task with high technical precision:\n{}\n\n",
        spec.prompt
    ));
    p.push_str("Available Tools:\n");
    p.push_str("- run_command: {\"command\": \"string\", \"cwd\": \"string\"}\n");
    p.push_str("- view_file: {\"path\": \"string\", \"start_line\": 1, \"end_line\": 100}\n");
    p.push_str(
        "- write_to_file: {\"path\": \"string\", \"content\": \"string\", \"overwrite\": true}\n",
    );
    p.push_str("- replace_file_content: {\"path\": \"string\", \"target\": \"string\", \"replacement\": \"string\"}\n");
    p.push_str("- create_dir: {\"path\": \"string\", \"purpose\": \"string\"}\n");
    p.push_str("- list_dir: {\"path\": \"string\"}\n");
    p.push_str("- grep_search: {\"query\": \"string\", \"path\": \"string\"}\n");
    p.push_str("- crumb: {\"action\": \"survey|whisper|record|init\", \"path\": \"string\"}\n");
    p.push_str("- vault: {\"action\": \"list|check|audit\", \"key\": \"string\"}\n");
    p.push_str("- cortex: {\"action\": \"search|write\", \"query\": \"string\", \"name\": \"string\", \"content\": \"string\", \"kind\": \"lesson|discovery|procedure\"}\n");
    p.push_str("- sandbox: {\"action\": \"init|status|test|promote|clean\"}\n");
    p.push_str("- browser: {\"action\": \"test|mentor\", \"prompt\": \"string\"}\n");
    if spec.depth < 3 {
        p.push_str("- invoke_subagent: {\"role\": \"string\", \"prompt\": \"string\"}\n");
    }
    p.push_str("\nPROTOCOL & CONTEXT HYGIENE:\n");
    p.push_str("1. ZERO EM DASHES AND EN DASHES: Strict invariant. Use standard commas, colons, or parentheses.\n");
    p.push_str("2. NO UNSLOP OR FLUFF: Lead directly with technical output and code.\n");
    p.push_str("3. ZERO PLAINTEXT SECRETS: Hardware TPM vault only.\n");
    if spec.depth < 3 {
        p.push_str("4. RECURSIVE SUBAGENTS: If a subtask requires substantial exploratory search, isolated testing, or deep verification, invoke a child subagent to keep your own context focused.\n");
    } else {
        p.push_str("4. LEAF SUBAGENT: Maximum recursion depth reached. Complete all remaining steps directly without invoking further subagents.\n");
    }
    p.push_str("5. SYNTHESIS: When your work is done, output your final response WITHOUT tool calls. Summarize your technical findings, exact changes, and verification proof so your parent coordinator receives a distilled, high-signal report.\n\n");
    p.push_str("To call a tool, output:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {}}\n</tool_call>\n");
    p
}

fn generate_short_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345);
    format!("{:04x}", nanos % 0xffff)
}

trait SubagentChat: Sync {
    fn reply(&self, messages: &[Value]) -> impl Future<Output = Result<String, String>> + Send;
}

impl SubagentChat for ChatClient {
    async fn reply(&self, messages: &[Value]) -> Result<String, String> {
        self.stream_turn_with_limit(messages, false, 1024).await
    }
}

pub fn execute_subagent(
    spec: SubagentSpec,
) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
    Box::pin(async move {
        let client = ChatClient::new(None, None);
        let platform = crate::platform::PlatformContext::detect();
        let transcript_dir = platform.home_dir.join("basecamp/sessions/subagents");
        execute_subagent_with(spec, &client, &transcript_dir, emit_completion_comb).await
    })
}

async fn execute_subagent_with(
    spec: SubagentSpec,
    client: &impl SubagentChat,
    transcript_dir: &Path,
    emit_comb: impl Fn(&SubagentSpec, &str),
) -> SubagentResult {
    let start_time = Utc::now().to_rfc3339();
    let indent = "  ".repeat(spec.depth);
    println!(
        "{}",
        format!(
            "{}┌─ [Subagent Spawning: '{}' | Depth: {} | ID: {}]",
            indent, spec.role, spec.depth, spec.id
        )
        .cyan()
        .bold()
    );
    println!(
        "{}",
        format!("{}│ Directive: {}", indent, spec.prompt).dimmed()
    );

    let mut messages = vec![
        json!({"role": "system", "content": get_subagent_system_prompt(&spec)}),
        json!({"role": "user", "content": format!("TASK ASSIGNMENT:\nRole: {}\nPrompt: {}\n\nExecute all necessary steps using tools, verify outcomes, and provide your distilled technical summary when complete.", spec.role, spec.prompt)}),
    ];

    let mut steps_log = Vec::new();
    let mut children_spawned = Vec::new();
    let mut max_steps = 10;
    let mut step_count = 0;
    let mut final_summary = String::new();
    let mut outcome = SubagentOutcome::Exhausted;

    while max_steps > 0 {
        max_steps -= 1;
        step_count += 1;

        let assistant_resp = match client.reply(&messages).await {
            Ok(text) => text,
            Err(e) => {
                let err_msg = format!("Subagent inference failed: {}", e);
                println!("{}", format!("{}│ [Error] {}", indent, err_msg).red());
                final_summary = err_msg;
                outcome = SubagentOutcome::Failed;
                break;
            }
        };

        messages.push(json!({"role": "assistant", "content": assistant_resp.clone()}));

        let tool_calls = match extract_tool_calls(&assistant_resp) {
            Ok(calls) => calls,
            Err(error) => {
                final_summary = error;
                outcome = SubagentOutcome::Failed;
                break;
            }
        };
        if tool_calls.is_empty() {
            // Only a real, nonempty model reply is a final synthesis.
            if assistant_resp.trim().is_empty() {
                outcome = SubagentOutcome::Failed;
                final_summary = "Subagent inference returned an empty reply.".to_string();
            } else {
                outcome = SubagentOutcome::Completed;
                final_summary = assistant_resp;
            }
            break;
        }

        let mut step_results = Vec::new();
        for (tool_name, tool_args) in &tool_calls {
            println!(
                "{}",
                format!(
                    "{}│ [Step {}] Executing tool: {}",
                    indent, step_count, tool_name
                )
                .yellow()
            );

            let tool_result = if tool_name == "invoke_subagent" {
                if spec.depth >= 3 {
                    json!({"error": "Maximum subagent recursion depth (3) reached. You must complete the task directly without further delegation."})
                } else {
                    let child_role = tool_args
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("Child Subagent");
                    let child_prompt = tool_args
                        .get("prompt")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let child_spec = SubagentSpec {
                        id: format!("sub-{}-{}", Utc::now().timestamp(), generate_short_id()),
                        role: child_role.to_string(),
                        prompt: child_prompt.to_string(),
                        depth: spec.depth + 1,
                        parent_id: Some(spec.id.clone()),
                    };
                    children_spawned.push(child_spec.id.clone());
                    let child_res = execute_subagent(child_spec).await;
                    serde_json::to_value(&child_res).unwrap_or_default()
                }
            } else {
                dispatch_tool(tool_name, tool_args)
            };

            let result_str = serde_json::to_string(&tool_result).unwrap_or_default();
            messages.push(json!({
                    "role": "user",
                    "content": format!("<tool_response name=\"{}\">\n{}\n</tool_response>", tool_name, result_str)
                }));
            step_results.push(tool_result);
        }

        steps_log.push(SubagentStepLog {
            step: step_count,
            assistant_response: assistant_resp,
            tool_calls: tool_calls.clone(),
            tool_results: step_results,
        });
    }

    if outcome == SubagentOutcome::Exhausted {
        final_summary = "Subagent exhausted its step limit without a final reply.".to_string();
    }

    let end_time = Utc::now().to_rfc3339();
    let transcript_path = transcript_dir.join(format!("{}.json", spec.id));

    let transcript = SubagentTranscript {
        spec: spec.clone(),
        start_time,
        end_time,
        status: outcome.status().to_string(),
        summary: final_summary.clone(),
        steps: steps_log,
        children: children_spawned.clone(),
    };

    let persisted = serde_json::to_vec_pretty(&transcript)
        .map_err(|e| e.to_string())
        .and_then(|bytes| {
            fs::create_dir_all(transcript_dir)
                .and_then(|_| fs::write(&transcript_path, bytes))
                .map_err(|e| e.to_string())
        });
    let (transcript_path, transcript_error) = match persisted {
        Ok(()) => (Some(transcript_path.to_string_lossy().to_string()), None),
        Err(error) => {
            outcome = SubagentOutcome::Failed;
            let message = format!("Subagent transcript write failed: {}", error);
            eprintln!("{}", format!("{}│ [Error] {}", indent, message).red());
            (None, Some(message))
        }
    };

    if outcome == SubagentOutcome::Completed {
        emit_comb(&spec, &final_summary);
    }

    let terminal_line = format!(
        "{}└─ [Subagent {} {} (depth: {}, steps: {}, children: {})]",
        indent,
        spec.id,
        outcome.status(),
        spec.depth,
        step_count,
        children_spawned.len()
    );
    let terminal_line = if outcome == SubagentOutcome::Completed {
        terminal_line.green()
    } else {
        terminal_line.red()
    };
    println!("{}", terminal_line.bold());

    SubagentResult {
        id: spec.id,
        role: spec.role,
        depth: spec.depth,
        parent_id: spec.parent_id,
        status: outcome.status().to_string(),
        summary: final_summary,
        transcript_path,
        transcript_error,
        steps_executed: step_count,
        children_spawned,
    }
}

fn emit_completion_comb(spec: &SubagentSpec, final_summary: &str) {
    let indent = "  ".repeat(spec.depth);
    if let Ok(store) = spark_hive::CombStore::open_default() {
        let summary_text = if final_summary.chars().count() > 280 {
            let trimmed: String = final_summary.chars().take(280).collect();
            format!("{}...", trimmed)
        } else {
            final_summary.to_string()
        };
        if let Ok(comb) = spark_hive::emit_subagent_comb(
            &store,
            &spec.role,
            &summary_text,
            spec.parent_id.as_deref(),
        ) {
            println!(
                "{}",
                format!(
                    "{}  [Hive Lattice] Comb placed at ({}, {}) [id: {}]",
                    indent, comb.q, comb.r, comb.id
                )
                .yellow()
            );
        }
    }
}

pub async fn invoke_subagent_from_root(role: &str, prompt: &str) -> Value {
    let spec = SubagentSpec {
        id: format!("sub-{}-{}", Utc::now().timestamp(), generate_short_id()),
        role: role.to_string(),
        prompt: prompt.to_string(),
        depth: 1,
        parent_id: Some("root-coordinator".to_string()),
    };
    let res = execute_subagent(spec).await;
    serde_json::to_value(&res).unwrap_or_default()
}

pub fn list_subagents() -> Value {
    let platform = crate::platform::PlatformContext::detect();
    let dir = platform.home_dir.join("basecamp/sessions/subagents");
    if !dir.exists() {
        return json!({"subagents": [], "count": 0});
    }

    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(transcript) = serde_json::from_str::<SubagentTranscript>(&content) {
                        items.push(json!({
                            "id": transcript.spec.id,
                            "role": transcript.spec.role,
                            "depth": transcript.spec.depth,
                            "parent_id": transcript.spec.parent_id,
                            "steps": transcript.steps.len(),
                            "status": transcript.status,
                            "start_time": transcript.start_time,
                            "summary_preview": transcript.summary.chars().take(120).collect::<String>(),
                            "children": transcript.children,
                            "transcript_path": path.to_string_lossy()
                        }));
                    }
                }
            }
        }
    }

    items.sort_by(|a, b| {
        let sa = a.get("start_time").and_then(Value::as_str).unwrap_or("");
        let sb = b.get("start_time").and_then(Value::as_str).unwrap_or("");
        sb.cmp(sa)
    });

    let count = items.len();
    json!({"subagents": items, "count": count})
}

pub fn view_subagent(id: &str) -> Value {
    let platform = crate::platform::PlatformContext::detect();
    let path = platform
        .home_dir
        .join("basecamp/sessions/subagents")
        .join(format!("{}.json", id));
    if !path.exists() {
        return json!({"error": format!("Subagent session '{}' not found", id)});
    }
    match fs::read_to_string(&path) {
        Ok(c) => serde_json::from_str::<Value>(&c).unwrap_or_else(|_| json!({"raw": c})),
        Err(e) => json!({"error": e.to_string()}),
    }
}

pub fn format_subagents_tui() -> String {
    let list = list_subagents();
    let mut out = String::new();
    out.push_str("\n=== AIEN Recursive Subagents Hierarchy ===\n");
    if let Some(items) = list.get("subagents").and_then(Value::as_array) {
        if items.is_empty() {
            out.push_str("No active or historical subagent sessions found.\n");
        } else {
            for item in items.iter().take(15) {
                let id = item.get("id").and_then(Value::as_str).unwrap_or("?");
                let role = item.get("role").and_then(Value::as_str).unwrap_or("?");
                let depth = item.get("depth").and_then(Value::as_u64).unwrap_or(1);
                let steps = item.get("steps").and_then(Value::as_u64).unwrap_or(0);
                let status = item
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let parent = item
                    .get("parent_id")
                    .and_then(Value::as_str)
                    .unwrap_or("Root");
                let preview = item
                    .get("summary_preview")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let indent = "  ".repeat(depth as usize);
                out.push_str(&format!(
                    "{}[depth {}] {} ({}) <- {}\n",
                    indent,
                    depth,
                    role.cyan().bold(),
                    id.yellow(),
                    parent.dimmed()
                ));
                out.push_str(&format!(
                    "{}  Status: {} | Steps: {} | Summary: {}\n",
                    indent,
                    status,
                    steps,
                    preview.dimmed()
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    struct FakeChat(std::sync::Mutex<std::collections::VecDeque<Result<String, String>>>);

    impl SubagentChat for FakeChat {
        async fn reply(&self, _messages: &[Value]) -> Result<String, String> {
            self.0
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model call")
        }
    }

    async fn run_replies(
        replies: Vec<Result<String, String>>,
        expected: &str,
        steps: usize,
        summary: &str,
    ) {
        let client = FakeChat(std::sync::Mutex::new(replies.into()));
        run_chat(&client, expected, steps, summary).await;
    }

    async fn run_chat(client: &impl SubagentChat, expected: &str, steps: usize, summary: &str) {
        let scratch = tempfile::tempdir().unwrap();
        let emissions = std::cell::RefCell::new(Vec::new());
        let result = execute_subagent_with(
            SubagentSpec {
                id: "outcome-test".into(),
                role: "Verifier".into(),
                prompt: "Test outcome classification".into(),
                depth: 3,
                parent_id: Some("test-parent".into()),
            },
            client,
            scratch.path(),
            |_, summary| emissions.borrow_mut().push(summary.to_string()),
        )
        .await;
        let transcript: SubagentTranscript = serde_json::from_str(
            &fs::read_to_string(result.transcript_path.as_ref().unwrap()).unwrap(),
        )
        .unwrap();
        let (tool_result, success) =
            crate::tools::subagent_tool_result(serde_json::to_value(&result).unwrap());
        assert_eq!(result.status, expected);
        assert_eq!(transcript.status, expected);
        assert_eq!(tool_result["status"], expected);
        assert_eq!(success, expected == "completed");
        assert_eq!(
            emissions.borrow().len(),
            usize::from(expected == "completed")
        );
        assert_eq!(result.steps_executed, steps);
        assert!(result.transcript_error.is_none());
        assert!(result.summary.contains(summary));
        assert_eq!(transcript.summary, result.summary);
    }

    #[tokio::test]
    async fn inference_error_is_failed() {
        run_replies(
            vec![Err("injected inference error".into())],
            "failed",
            1,
            "injected inference error",
        )
        .await;
    }

    #[tokio::test]
    async fn ten_tool_replies_are_exhausted() {
        let tool_reply = "<tool_call>{\"name\":\"invoke_subagent\",\"arguments\":{}}</tool_call>";
        run_replies(
            vec![Ok(tool_reply.into()); 10],
            "exhausted",
            10,
            "step limit",
        )
        .await;
    }

    #[tokio::test]
    async fn final_reply_is_completed() {
        run_replies(
            vec![Ok("Verified final synthesis.".into())],
            "completed",
            1,
            "Verified final synthesis.",
        )
        .await;
    }

    #[tokio::test]
    async fn empty_reply_is_failed() {
        run_replies(vec![Ok(" \n\t".into())], "failed", 1, "empty reply").await;
    }

    #[tokio::test]
    async fn error_after_tool_reply_is_failed() {
        run_replies(
            vec![
                Ok("<tool_call>{\"name\":\"invoke_subagent\",\"arguments\":{}}</tool_call>".into()),
                Err("injected inference error after tools".into()),
            ],
            "failed",
            2,
            "injected inference error after tools",
        )
        .await;
    }

    #[tokio::test]
    async fn final_reply_on_last_turn_is_completed() {
        let tool_reply = "<tool_call>{\"name\":\"invoke_subagent\",\"arguments\":{}}</tool_call>";
        let mut replies = vec![Ok(tool_reply.into()); 9];
        replies.push(Ok("Final synthesis on the last permitted turn.".into()));
        run_replies(replies, "completed", 10, "last permitted turn").await;
    }

    async fn persistence_failure(fail_directory_creation: bool) {
        for reply in [
            Ok::<String, String>("Final synthesis.".into()),
            Err("injected inference error".into()),
        ] {
            let scratch = tempfile::tempdir().unwrap();
            let transcript_dir = scratch.path().join("transcripts");
            if fail_directory_creation {
                fs::write(&transcript_dir, "not a directory").unwrap();
            } else {
                fs::create_dir_all(transcript_dir.join("outcome-test.json")).unwrap();
            }
            let summary = match &reply {
                Ok(text) => text.clone(),
                Err(error) => error.clone(),
            };
            let client = FakeChat(std::sync::Mutex::new(vec![reply].into()));
            let emissions = std::cell::Cell::new(0);
            let result = execute_subagent_with(
                SubagentSpec {
                    id: "outcome-test".into(),
                    role: "Verifier".into(),
                    prompt: "Test persistence failure".into(),
                    depth: 3,
                    parent_id: None,
                },
                &client,
                &transcript_dir,
                |_, _| emissions.set(emissions.get() + 1),
            )
            .await;
            assert_eq!(result.status, "failed");
            assert!(result.transcript_path.is_none());
            assert!(result
                .transcript_error
                .as_ref()
                .unwrap()
                .contains("transcript write failed"));
            assert!(result.summary.contains(&summary));
            let (value, success) =
                crate::tools::subagent_tool_result(serde_json::to_value(&result).unwrap());
            assert!(value["transcript_path"].is_null());
            assert!(!success);
            assert_eq!(emissions.get(), 0);
        }
    }

    #[tokio::test]
    async fn round2_persistence_directory_failure() {
        persistence_failure(true).await;
    }

    #[tokio::test]
    async fn round2_persistence_file_failure() {
        persistence_failure(false).await;
    }

    #[tokio::test]
    async fn round2_malformed_tool_call_is_failed() {
        run_replies(
            vec![Ok("<tool_call>{\"name\":</tool_call>".into())],
            "failed",
            1,
            "Malformed tool call",
        )
        .await;
    }

    #[tokio::test]
    async fn round3_unclosed_named_json_fence_is_failed() {
        run_replies(
            vec![Ok("```json\n{\"name\":\"goal\",\"arguments\":{\"action\":\"list\"}}".into())],
            "failed", 1, "Malformed tool call",
        ).await;
    }

    #[tokio::test]
    async fn round3_unclosed_spaced_tool_fence_is_failed() {
        for prefix in [" tool_call", "\n tool_call", "\njson"] {
            run_replies(
                vec![Ok(format!("```{prefix}\n{{\"name\":\"goal\",\"arguments\":{{}}}}"))],
                "failed", 1, "Malformed tool call",
            ).await;
        }
    }

    async fn remote_fixture(body: String) -> (ChatClient, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut bytes = [0; 1024];
            loop {
                let count = socket.read(&mut bytes).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&bytes[..count]);
                if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&request[..pos]).unwrap();
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if request.len() >= pos + 4 + length {
                        break;
                    }
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        });
        (
            ChatClient::new(Some(endpoint), Some("test-model".into())),
            server,
        )
    }

    fn remote_body(reason: Option<&str>, done: bool) -> String {
        let mut body = format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content":"Partial final synthesis."}}]})
        );
        if let Some(reason) = reason {
            body.push_str(&format!(
                "data: {}\n\n",
                json!({"choices":[{"delta":{},"finish_reason":reason}]})
            ));
        }
        if done {
            body.push_str("data: [DONE]\n\n");
        }
        body
    }

    #[tokio::test]
    async fn round2_remote_eof_is_failed() {
        let (client, server) = remote_fixture(remote_body(None, false)).await;
        run_chat(&client, "failed", 1, "before [DONE]").await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn round2_remote_length_is_failed() {
        let (client, server) = remote_fixture(remote_body(Some("length"), true)).await;
        run_chat(&client, "failed", 1, "length").await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_stop_is_completed() {
        let (client, server) = remote_fixture(remote_body(Some("stop"), true)).await;
        run_chat(&client, "completed", 1, "Partial final synthesis.").await;
        server.await.unwrap();
    }

    struct NativeChat(aien_runtime::client::AienRuntimeClient);
    impl SubagentChat for NativeChat {
        async fn reply(&self, _messages: &[Value]) -> Result<String, String> {
            self.0.stream_turn(Vec::new(), 1024, 0.2).await
        }
    }

    async fn native_fixture(
        finish: bool,
    ) -> (tempfile::TempDir, NativeChat, tokio::task::JoinHandle<()>) {
        use aien_runtime::control::ControlResponse;
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("runtime.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = String::new();
            tokio::io::BufReader::new(&mut socket)
                .read_line(&mut request)
                .await
                .unwrap();
            serde_json::from_str::<aien_runtime::control::ControlEnvelope>(&request).unwrap();
            let delta = ControlResponse::TurnDelta {
                text: "Partial final synthesis.".into(),
            };
            socket
                .write_all(format!("{}\n", serde_json::to_string(&delta).unwrap()).as_bytes())
                .await
                .unwrap();
            if finish {
                let terminal = ControlResponse::TurnFinished {
                    text: "Complete native synthesis.".into(),
                    total_tokens: 4,
                    generation_record: Some(9),
                };
                socket
                    .write_all(
                        format!("{}\n", serde_json::to_string(&terminal).unwrap()).as_bytes(),
                    )
                    .await
                    .unwrap();
            }
            socket.shutdown().await.unwrap();
        });
        let client = NativeChat(aien_runtime::client::AienRuntimeClient::new(&path));
        (scratch, client, server)
    }

    #[tokio::test]
    async fn round2_native_eof_is_failed() {
        let (_scratch, client, server) = native_fixture(false).await;
        run_chat(&client, "failed", 1, "before the turn finished").await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_finished_is_completed() {
        let (_scratch, client, server) = native_fixture(true).await;
        run_chat(&client, "completed", 1, "Complete native synthesis.").await;
        server.await.unwrap();
    }

    #[test]
    fn tool_flag_rejects_missing_and_unknown_status() {
        for value in [
            json!({}),
            json!({"status": "unknown"}),
            json!({"error": "serialization failed"}),
        ] {
            assert!(!crate::tools::subagent_tool_result(value).1);
        }
    }

    #[test]
    fn test_subagent_spec_serialization() {
        let spec = SubagentSpec {
            id: "sub-test-01".to_string(),
            role: "Codebase Researcher".to_string(),
            prompt: "Survey crates and verify Cargo.toml".to_string(),
            depth: 1,
            parent_id: Some("root-coord".to_string()),
        };
        let json_val = serde_json::to_value(&spec).expect("serialize spec");
        assert_eq!(json_val["id"], "sub-test-01");
        assert_eq!(json_val["depth"], 1);
        assert_eq!(json_val["role"], "Codebase Researcher");

        let deserialized: SubagentSpec =
            serde_json::from_value(json_val).expect("deserialize spec");
        assert_eq!(deserialized.id, spec.id);
        assert_eq!(deserialized.role, spec.role);
    }

    #[test]
    fn test_subagent_depth_guard() {
        let max_depth_spec = SubagentSpec {
            id: "sub-leaf-03".to_string(),
            role: "Leaf Verifier".to_string(),
            prompt: "Perform final check".to_string(),
            depth: 3,
            parent_id: Some("sub-parent-02".to_string()),
        };
        let prompt = get_subagent_system_prompt(&max_depth_spec);
        assert!(prompt.contains("LEAF SUBAGENT: Maximum recursion depth reached"));
        assert!(!prompt.contains("- invoke_subagent:"));
    }

    #[test]
    fn test_subagent_root_spec() {
        let root_spec = SubagentSpec {
            id: "sub-root-01".to_string(),
            role: "Researcher".to_string(),
            prompt: "Initial task".to_string(),
            depth: 1,
            parent_id: None,
        };
        let prompt = get_subagent_system_prompt(&root_spec);
        assert!(prompt.contains("- invoke_subagent:"));
        assert!(prompt.contains("RECURSIVE SUBAGENTS"));
        assert!(!prompt.is_empty());
        // verified hyphen use
    }

    #[test]
    fn test_subagents_list_and_view() {
        let list = list_subagents();
        assert!(list.get("subagents").is_some());
    }
}
