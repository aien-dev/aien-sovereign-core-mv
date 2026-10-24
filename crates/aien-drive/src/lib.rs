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

//! AIEN autonomous API driver (Rust port of the removed `scripts/aien_drive.py`).
//!
//! Loops a chat model at an OpenAI-style streaming endpoint with a system
//! prompt, extracts `<tool_call>` blocks from its reply, dispatches them, and
//! feeds the results back as `<tool_response>` user messages.

pub mod http;

use regex::Regex;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const ENDPOINT_LIGHTNING: &str = "http://127.0.0.1:18006/v1/chat/completions";
pub const MODEL_LIGHTNING: &str = "atlas-lightning-omni";
pub const ENDPOINT_JUDGE: &str = "http://127.0.0.1:18082/v1/chat/completions";
pub const MODEL_JUDGE: &str = "unsloth/Llama-3.2-1B-Instruct";
pub const DEFAULT_CORTEX: &str = "http://127.0.0.1:18080";

pub const SYSTEM_PROMPT: &str = r#"You are AIEN. Drake is the operator. This Spark desk is ours.
You are an autonomous operator-builder running on NVIDIA DGX Spark Grace Blackwell GB10 hardware.
Always be concise, precise, and lead with verified results.

Available Tools:
- run_command: {"command": "string", "cwd": "string"}
- view_file: {"path": "string", "start_line": 1, "end_line": 100}
- write_to_file: {"path": "string", "content": "string", "overwrite": true}
- replace_file_content: {"path": "string", "target": "string", "replacement": "string"}
- create_dir: {"path": "string", "purpose": "string"}
- list_dir: {"path": "string"}
- grep_search: {"query": "string", "path": "string"}
- goal: {"action": "milestone_done|done", "id": "string", "milestone_id": 1}
- cortex: {"action": "write", "name": "string", "content": "string", "kind": "lesson|discovery|procedure"}

MANDATORY RULES:
1. ZERO EM DASHES AND EN DASHES: Use plain commas, periods, or standard hyphens (-).
2. BAN FORMULAIC AI CLICHES: Never use 'It is not X, it is Y', 'delve', 'tapestry', 'testament', 'crucial', 'pivotal'.
3. LEAD WITH TECHNICAL PROOF: Output factual diffs, test outputs, and verified receipts.
4. TOOL EXECUTION FORMAT:
To execute a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "val"}}
</tool_call>

When all work for the milestone is verified, call goal with action 'milestone_done'.
"#;

/// Where a driver run talks to.
#[derive(Clone, Debug)]
pub struct Config {
    pub endpoint: String,
    pub model: String,
    pub cortex_endpoint: String,
    /// Cortex area the `cortex` tool writes to.
    pub cortex_space: String,
    pub request_timeout: Duration,
}

fn last_chars(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    if c.len() > n {
        c[c.len() - n..].iter().collect()
    } else {
        s.to_string()
    }
}

fn first_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Pull tool calls out of model text. Pattern 1: `<tool_call>{json}</tool_call>`.
/// Pattern 2 (only if pattern 1 found nothing): bare `{name, arguments}` objects.
pub fn extract_tool_calls(text: &str) -> Vec<Value> {
    let p1 = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();
    let calls: Vec<Value> = p1
        .captures_iter(text)
        .filter_map(|c| serde_json::from_str(c[1].trim()).ok())
        .collect();
    if !calls.is_empty() {
        return calls;
    }
    let p2 = Regex::new(
        r#"(?s)(\{\s*"?name"?\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"?arguments"?\s*:\s*\{.*?\}\s*\})"#,
    )
    .unwrap();
    p2.captures_iter(text)
        .filter_map(|c| serde_json::from_str(&c[1].replace('\'', "\"")).ok())
        .collect()
}

fn s_arg<'a>(a: &'a Value, k: &str, d: &'a str) -> &'a str {
    a.get(k).and_then(Value::as_str).unwrap_or(d)
}

fn run_with_timeout(
    mut cmd: Command,
    timeout: Duration,
    label: &str,
) -> Result<(i32, String, String), String> {
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = std::io::Read::read_to_end(&mut out, &mut b);
        b
    });
    let t_err = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = std::io::Read::read_to_end(&mut err, &mut b);
        b
    });
    let start = Instant::now();
    loop {
        if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
            let o = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
            let e = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
            return Ok((st.code().unwrap_or(-1), o, e));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Command '{label}' timed out after {} seconds",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run one tool. Never panics; failures come back as `{"error": ...}`.
pub fn dispatch_tool(cfg: &Config, name: &str, args: &Value) -> Value {
    let started = Instant::now();
    let shown = first_chars(&args.to_string(), 120);
    eprintln!("  [TOOL RUN] {name}: {shown}...");
    let res = dispatch_inner(cfg, name, args).unwrap_or_else(|e| json!({ "error": e }));
    eprintln!(
        "  [TOOL DONE] {name} in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    res
}

fn dispatch_inner(cfg: &Config, name: &str, a: &Value) -> Result<Value, String> {
    let io = |e: std::io::Error| e.to_string();
    Ok(match name {
        "run_command" => {
            let command = s_arg(a, "command", "");
            let mut c = Command::new("sh");
            c.arg("-c").arg(command).current_dir(s_arg(a, "cwd", "."));
            let (code, o, e) = run_with_timeout(c, Duration::from_secs(120), command)?;
            json!({"exit_code": code, "stdout": last_chars(&o, 4000), "stderr": last_chars(&e, 2000)})
        }
        "view_file" => {
            let path = s_arg(a, "path", "");
            let start = a.get("start_line").and_then(Value::as_i64).unwrap_or(1);
            let end = a.get("end_line").and_then(Value::as_i64).unwrap_or(100);
            let text = String::from_utf8_lossy(&std::fs::read(path).map_err(io)?).into_owned();
            let lines: Vec<&str> = text.split_inclusive('\n').collect();
            let total = lines.len() as i64;
            let lo = (start - 1).max(0).min(total) as usize;
            let hi = end.min(total).max(0) as usize;
            let sel = if lo < hi {
                lines[lo..hi].concat()
            } else {
                String::new()
            };
            json!({"path": path, "total_lines": total, "lines": sel})
        }
        "write_to_file" => {
            let path = s_arg(a, "path", "");
            let content = s_arg(a, "content", "");
            let overwrite = a.get("overwrite").and_then(Value::as_bool).unwrap_or(true);
            if std::path::Path::new(path).exists() && !overwrite {
                json!({"error": format!("File {path} exists and overwrite=false")})
            } else {
                if let Some(dir) = std::path::Path::new(path).parent() {
                    if !dir.as_os_str().is_empty() {
                        std::fs::create_dir_all(dir).map_err(io)?;
                    }
                }
                std::fs::write(path, content).map_err(io)?;
                json!({"status": "ok", "bytes_written": content.len()})
            }
        }
        "replace_file_content" => {
            let path = s_arg(a, "path", "");
            let target = s_arg(a, "target", "");
            let content = std::fs::read_to_string(path).map_err(io)?;
            if !content.contains(target) {
                json!({"error": format!("Target string not found in {path}")})
            } else {
                let new = content.replacen(target, s_arg(a, "replacement", ""), 1);
                std::fs::write(path, new).map_err(io)?;
                json!({"status": "ok", "replaced": true})
            }
        }
        "list_dir" => {
            let path = s_arg(a, "path", ".");
            let mut names: Vec<String> = std::fs::read_dir(path)
                .map_err(io)?
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect();
            names.sort();
            names.truncate(100);
            json!({"path": path, "entries": names})
        }
        "grep_search" => {
            let mut c = Command::new("grep");
            c.args(["-rnI", s_arg(a, "query", ""), s_arg(a, "path", ".")]);
            let (_, o, _) = run_with_timeout(c, Duration::from_secs(30), "grep")?;
            json!({"matches": first_chars(&o, 4000)})
        }
        "create_dir" => {
            let path = s_arg(a, "path", "");
            std::fs::create_dir_all(path).map_err(io)?;
            json!({"status": "ok", "created": path})
        }
        "goal" => {
            json!({"status": "ok", "goal_action": s_arg(a, "action", ""), "milestone_done": true})
        }
        "cortex" => {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let payload = json!({
                "kind": "entity",
                "value": {
                    "space": cfg.cortex_space,
                    "canonicalName": s_arg(a, "name", "lesson"),
                    "content": s_arg(a, "content", ""),
                    "confidence": 0.95,
                    "metadata": {"kind": s_arg(a, "kind", "lesson"), "timestamp": ts}
                }
            });
            let url = format!("{}/api/cortex/write", cfg.cortex_endpoint);
            match http::post_json(&url, &payload.to_string(), Duration::from_secs(10)) {
                Ok(r) if r.status < 400 => json!({"status": "ok", "cortex_code": r.status}),
                Ok(r) => json!({"warning": format!("Cortex write notice: HTTP {}", r.status)}),
                Err(e) => json!({"warning": format!("Cortex write notice: {e}")}),
            }
        }
        other => json!({"error": format!("Unknown tool: {other}")}),
    })
}

fn sanitize(s: &str) -> String {
    s.replace('\u{2014}', ", ").replace('\u{2013}', "-")
}

/// Read an SSE stream of chat-completion chunks. Returns the assistant text
/// (falling back to the reasoning text if it holds a tool call and content is empty).
pub fn read_stream(body: impl BufRead, out: &mut dyn Write) -> String {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut in_reasoning = false;
    for line in body.split(b'\n') {
        let Ok(line) = line else { break };
        let line = String::from_utf8_lossy(&line).trim().to_string();
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let delta = &chunk["choices"][0]["delta"];
        if let Some(r) = delta["reasoning"].as_str().filter(|s| !s.is_empty()) {
            if !in_reasoning {
                let _ = write!(out, "\n[AIEN THINKING] ");
                in_reasoning = true;
            }
            let r = sanitize(r);
            let _ = write!(out, "{r}");
            reasoning.push_str(&r);
        }
        if let Some(c) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            if in_reasoning {
                let _ = write!(out, "\n[AIEN OUTPUT]\n");
                in_reasoning = false;
            }
            let c = sanitize(c);
            let _ = write!(out, "{c}");
            content.push_str(&c);
        }
        let _ = out.flush();
    }
    let _ = writeln!(out, "\n");
    if content.is_empty() && reasoning.contains("<tool_call>") {
        reasoning
    } else {
        content
    }
}

fn call_model(cfg: &Config, messages: &[Value], out: &mut dyn Write) -> Result<String, String> {
    let payload = json!({
        "model": cfg.model, "messages": messages, "stream": true,
        "temperature": 0.2, "max_tokens": 2048
    });
    let r = http::post_json(&cfg.endpoint, &payload.to_string(), cfg.request_timeout)?;
    if r.status >= 400 {
        return Err(format!("model endpoint returned HTTP {}", r.status));
    }
    Ok(read_stream(r.body, out))
}

/// Drive the loop. Ok(true) when the model signalled `milestone_done`/`done`.
pub fn run_directive(cfg: &Config, directive: &str, max_turns: u32) -> Result<bool, String> {
    let mut out = std::io::stdout();
    println!("{}", "=".repeat(60));
    println!("DRIVING AIEN AUTONOMOUS EXECUTION LOOP");
    println!("Directive: {}...", first_chars(directive, 120));
    println!("{}", "=".repeat(60));
    let mut messages = vec![
        json!({"role": "system", "content": SYSTEM_PROMPT}),
        json!({"role": "user", "content": directive}),
    ];
    for turn in 1..=max_turns {
        println!("\n--- Turn {turn}/{max_turns} ---");
        let text = call_model(cfg, &messages, &mut out)?;
        messages.push(json!({"role": "assistant", "content": text}));
        let calls = extract_tool_calls(&text);
        if calls.is_empty() {
            println!("[AIEN IDLE] No tool calls emitted. Turn sequence concluded.");
            break;
        }
        let mut done = false;
        for tc in calls {
            let name = tc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let args = tc.get("arguments").cloned().unwrap_or_else(|| json!({}));
            if name == "goal" && matches!(s_arg(&args, "action", ""), "milestone_done" | "done") {
                done = true;
            }
            let res = dispatch_tool(cfg, &name, &args);
            let body = serde_json::to_string_pretty(&res).unwrap_or_default();
            messages.push(json!({
                "role": "user",
                "content": format!("<tool_response name=\"{name}\">\n{body}\n</tool_response>")
            }));
        }
        if done {
            println!("AIEN signaled milestone_done! Directive complete.");
            return Ok(true);
        }
    }
    Ok(false)
}
