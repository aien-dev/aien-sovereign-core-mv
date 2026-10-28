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

//! The other request shapes xAI's API accepts, as conversions to and from the chat-completions
//! shapes the server already serves: `POST /v1/responses` (OpenAI / xAI Responses), the legacy
//! `POST /v1/completions` and `POST /v1/messages` (Anthropic Messages). A request becomes a chat
//! request, so it passes the same validation; the chat-completion body becomes the reply. Nothing
//! is stored: `previous_response_id` and `background` are refused, never ignored.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

/// The part of a chat id after `chatcmpl-`, shared by the ids of every other surface.
pub fn id_suffix(chat_id: &str) -> &str {
    chat_id.strip_prefix("chatcmpl-").unwrap_or(chat_id)
}

fn copy(from: &Map<String, Value>, to: &mut Map<String, Value>, keys: &[&str]) {
    for k in keys {
        if let Some(v) = from.get(*k).filter(|v| !v.is_null()) {
            to.insert((*k).to_string(), v.clone());
        }
    }
}

fn object(raw: &Value) -> Result<&Map<String, Value>, String> {
    raw.as_object()
        .ok_or_else(|| "the request must be a JSON object".to_string())
}

// ---------------------------------------------------------------------------------------------
// Responses

/// A Responses request as a chat request, plus what the reply echoes back (`echo`).
pub fn responses_to_chat(raw: &Value) -> Result<(Value, Value), String> {
    let o = object(raw)?;
    if o.get("previous_response_id").is_some_and(|v| !v.is_null()) {
        return Err("previous_response_id is not supported: this server stores no responses; send the whole conversation in input".into());
    }
    if o.get("background").and_then(Value::as_bool) == Some(true) {
        return Err("background responses are not supported: this server stores no responses".into());
    }
    if o.get("store").and_then(Value::as_bool) == Some(true) {
        return Err("store: true is not supported: this server stores no responses".into());
    }
    let mut messages = Vec::new();
    if let Some(i) = o.get("instructions").filter(|v| !v.is_null()) {
        let s = i
            .as_str()
            .ok_or("instructions must be a string")?;
        messages.push(json!({"role": "system", "content": s}));
    }
    match o.get("input") {
        Some(Value::String(s)) => messages.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for item in items {
                input_item(item, &mut messages)?;
            }
        }
        _ => return Err("input must be a string or an array of input items".into()),
    }
    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(messages));
    copy(
        o,
        &mut chat,
        &[
            "model",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stream",
            "parallel_tool_calls",
            "reasoning_effort",
            "chat_template_kwargs",
            "aien_test",
            "aien_return_token_ids",
        ],
    );
    if let Some(m) = o.get("max_output_tokens").filter(|v| !v.is_null()) {
        chat.insert("max_completion_tokens".into(), m.clone());
    }
    if let Some(e) = o
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .filter(|v| !v.is_null())
    {
        chat.insert("reasoning_effort".into(), e.clone());
    }
    let tools = match o.get("tools").filter(|v| !v.is_null()) {
        None => Vec::new(),
        Some(Value::Array(t)) => t.iter().map(response_tool).collect::<Result<_, _>>()?,
        Some(_) => return Err("tools must be an array".into()),
    };
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = o.get("tool_choice").filter(|v| !v.is_null()) {
        chat.insert("tool_choice".into(), response_tool_choice(tc));
    }
    if let Some(f) = o
        .get("text")
        .and_then(|t| t.get("format"))
        .filter(|v| !v.is_null())
    {
        chat.insert("response_format".into(), f.clone());
    }
    let echo = json!({
        "instructions": o.get("instructions").cloned().unwrap_or(Value::Null),
        "max_output_tokens": o.get("max_output_tokens").cloned().unwrap_or(Value::Null),
        "temperature": o.get("temperature").cloned().unwrap_or(Value::Null),
        "top_p": o.get("top_p").cloned().unwrap_or(Value::Null),
        "tools": o.get("tools").filter(|v| v.is_array()).cloned().unwrap_or(json!([])),
        "tool_choice": o.get("tool_choice").filter(|v| !v.is_null()).cloned().unwrap_or(json!("auto")),
        "parallel_tool_calls": o.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true),
        "metadata": o.get("metadata").filter(|v| v.is_object()).cloned().unwrap_or(json!({})),
    });
    Ok((Value::Object(chat), echo))
}

fn input_item(item: &Value, messages: &mut Vec<Value>) -> Result<(), String> {
    let kind = item.get("type").and_then(Value::as_str).unwrap_or("message");
    match kind {
        "message" => {
            let role = item
                .get("role")
                .and_then(Value::as_str)
                .ok_or("a message input item needs a role")?;
            let content = match item.get("content") {
                Some(Value::String(s)) => Value::String(s.clone()),
                Some(Value::Array(parts)) => Value::Array(
                    parts
                        .iter()
                        .map(|p| match p.get("type").and_then(Value::as_str) {
                            Some("input_text" | "output_text" | "text") => {
                                json!({"type": "text", "text": p.get("text").cloned().unwrap_or(Value::Null)})
                            }
                            // The chat path names the part it refuses.
                            other => json!({"type": other.unwrap_or("missing")}),
                        })
                        .collect(),
                ),
                _ => return Err("a message input item needs string or array content".into()),
            };
            messages.push(json!({"role": role, "content": content}));
        }
        "function_call" => {
            let call = json!({
                "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                "type": "function",
                "function": {
                    "name": item.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                },
            });
            match messages.last_mut() {
                Some(m) if m["role"] == "assistant" => {
                    if !m["tool_calls"].is_array() {
                        m["tool_calls"] = json!([]);
                    }
                    m["tool_calls"]
                        .as_array_mut()
                        .expect("array")
                        .push(call);
                }
                _ => messages.push(json!({"role": "assistant", "content": null, "tool_calls": [call]})),
            }
        }
        "function_call_output" => {
            let output = match item.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect(),
                _ => return Err("a function_call_output item needs output".into()),
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                "content": output,
            }));
        }
        // The model's own earlier reasoning: not re-rendered (the chat template drops it too).
        "reasoning" => {}
        other => {
            return Err(format!(
                "input item type '{other}' is not supported: only message, function_call, function_call_output and reasoning items"
            ))
        }
    }
    Ok(())
}

fn response_tool(t: &Value) -> Result<Value, String> {
    match t.get("type").and_then(Value::as_str) {
        Some("function") if t.get("function").is_some() => Ok(t.clone()),
        Some("function") => {
            let mut f = Map::new();
            copy(
                t.as_object().expect("tool object"),
                &mut f,
                &["name", "description", "parameters"],
            );
            Ok(json!({"type": "function", "function": f}))
        }
        other => Err(format!(
            "tool type '{}' is not supported: only function tools; server-side tools (web_search, x_search, code_execution, ...) run at xAI, not on this server",
            other.unwrap_or("missing")
        )),
    }
}

fn response_tool_choice(tc: &Value) -> Value {
    match (tc.get("type").and_then(Value::as_str), tc.get("name")) {
        (Some("function"), Some(name)) => json!({"type": "function", "function": {"name": name}}),
        _ => tc.clone(),
    }
}

pub fn reasoning_item(chat_id: &str, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": format!("rs_{}", id_suffix(chat_id)),
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "status": "completed",
    })
}

pub fn message_item(chat_id: &str, text: &str, status: &str) -> Value {
    let content = if status == "in_progress" {
        json!([])
    } else {
        json!([output_text(text)])
    };
    json!({
        "type": "message",
        "id": format!("msg_{}", id_suffix(chat_id)),
        "role": "assistant",
        "status": status,
        "content": content,
    })
}

pub fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

/// A chat `tool_calls` entry as a Responses `function_call` item.
pub fn function_call_item(call: &Value, status: &str) -> Value {
    let call_id = call["id"].as_str().unwrap_or("call");
    json!({
        "type": "function_call",
        "id": format!("fc_{call_id}"),
        "call_id": call_id,
        "name": call["function"]["name"],
        "arguments": if status == "in_progress" { json!("") } else { call["function"]["arguments"].clone() },
        "status": status,
    })
}

/// Chat usage as Responses usage (the `aien_*` extras ride along).
pub fn response_usage(u: &Value) -> Value {
    let mut v = json!({
        "input_tokens": u["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": u["completion_tokens"],
        "output_tokens_details": {"reasoning_tokens": u["completion_tokens_details"]["reasoning_tokens"]},
        "total_tokens": u["total_tokens"],
        "num_sources_used": 0,
        "num_server_side_tools_used": 0,
    });
    if let (Some(from), Some(to)) = (u.as_object(), v.as_object_mut()) {
        for (k, x) in from.iter().filter(|(k, _)| k.starts_with("aien_")) {
            to.insert(k.clone(), x.clone());
        }
    }
    v
}

/// The `response` object: `status` `in_progress` (no output, no usage yet), `completed` or
/// `incomplete` (the reply hit `max_output_tokens`).
pub fn response_object(
    echo: &Value,
    chat_id: &str,
    created: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Option<Value>,
) -> Value {
    let mut r = json!({
        "id": format!("resp_{}", id_suffix(chat_id)),
        "object": "response",
        "created_at": created,
        "model": model,
        "status": status,
        "output": output,
        "usage": usage,
        "error": null,
        "incomplete_details": if status == "incomplete" { json!({"reason": "max_output_tokens"}) } else { Value::Null },
        "previous_response_id": null,
        "store": false,
        "background": false,
        "truncation": "disabled",
        "text": {"format": {"type": "text"}},
        "reasoning": null,
    });
    if status == "completed" || status == "incomplete" {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(created, |d| d.as_secs());
        r["completed_at"] = json!(now);
    }
    if let (Some(from), Some(to)) = (echo.as_object(), r.as_object_mut()) {
        for (k, v) in from {
            to.insert(k.clone(), v.clone());
        }
    }
    r
}

/// The output items of one chat choice, in order: reasoning, message, function calls.
pub fn output_items(chat_id: &str, message: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(r) = message["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
        out.push(reasoning_item(chat_id, r));
    }
    let calls = message["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
    let text = message["content"].as_str().unwrap_or("");
    if !text.is_empty() || calls.is_empty() {
        out.push(message_item(chat_id, text, "completed"));
    }
    out.extend(calls.iter().map(|c| function_call_item(c, "completed")));
    out
}

pub fn status_of(finish: &str) -> &'static str {
    if finish == "length" {
        "incomplete"
    } else {
        "completed"
    }
}

/// A chat completion as a `response` object.
pub fn chat_to_response(chat: &Value, echo: &Value) -> Value {
    let id = chat["id"].as_str().unwrap_or("");
    let choice = &chat["choices"][0];
    let mut r = response_object(
        echo,
        id,
        chat["created"].as_u64().unwrap_or(0),
        chat["model"].as_str().unwrap_or(""),
        status_of(choice["finish_reason"].as_str().unwrap_or("stop")),
        output_items(id, &choice["message"]),
        Some(response_usage(&chat["usage"])),
    );
    if let Some(info) = chat.get("aien_readers") {
        r["aien_readers"] = info.clone();
    }
    r
}

// ---------------------------------------------------------------------------------------------
// Legacy completions

/// A legacy completions request as chat fields plus its raw prompt (no chat template).
pub fn completions_to_chat(raw: &Value) -> Result<(Value, String), String> {
    let o = object(raw)?;
    let prompt = match o.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().expect("string").to_string()
        }
        Some(Value::Array(_)) => {
            return Err("prompt must be a string or an array of one string (batched prompts are not supported)".into())
        }
        _ => return Err("prompt must be a string".into()),
    };
    if o.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err("stream is not supported on /v1/completions; use /v1/chat/completions or /v1/responses".into());
    }
    if o.get("echo").and_then(Value::as_bool) == Some(true) {
        return Err("echo is not supported".into());
    }
    if o.get("suffix").is_some_and(|v| !v.is_null()) {
        return Err("suffix is not supported".into());
    }
    if o.get("logprobs").is_some_and(|v| !v.is_null() && v != &json!(false) && v != &json!(0)) {
        return Err("logprobs is not supported".into());
    }
    if o.get("best_of").and_then(Value::as_i64).is_some_and(|b| b > 1) {
        return Err("best_of > 1 is not supported".into());
    }
    let mut chat = Map::new();
    copy(
        o,
        &mut chat,
        &[
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "n",
            "aien_test",
            "aien_return_token_ids",
        ],
    );
    Ok((Value::Object(chat), prompt))
}

pub fn chat_to_text_completion(chat: &Value) -> Value {
    let choices: Vec<Value> = chat["choices"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .map(|c| {
            json!({
                "index": c["index"],
                "text": c["message"]["content"].as_str().unwrap_or(""),
                "finish_reason": c["finish_reason"],
                "logprobs": null,
            })
        })
        .collect();
    json!({
        "id": format!("cmpl-{}", id_suffix(chat["id"].as_str().unwrap_or(""))),
        "object": "text_completion",
        "created": chat["created"],
        "model": chat["model"],
        "choices": choices,
        "usage": chat["usage"],
    })
}

// ---------------------------------------------------------------------------------------------
// Anthropic Messages

fn block_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => Ok(b["text"].as_str().unwrap_or("").to_string()),
                other => Err(format!(
                    "content block type '{}' is not supported here: only text",
                    other.unwrap_or("missing")
                )),
            })
            .collect(),
        Value::Null => Ok(String::new()),
        _ => Err("content must be a string or an array of blocks".into()),
    }
}

/// A Messages request as a chat request.
pub fn messages_to_chat(raw: &Value) -> Result<Value, String> {
    let o = object(raw)?;
    if o.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err("stream is not supported on /v1/messages; use /v1/chat/completions or /v1/responses".into());
    }
    let max_tokens = o
        .get("max_tokens")
        .filter(|v| v.is_i64() || v.is_u64())
        .ok_or("max_tokens is required")?;
    let mut messages = Vec::new();
    if let Some(s) = o.get("system").filter(|v| !v.is_null()) {
        messages.push(json!({"role": "system", "content": block_text(s)?}));
    }
    let turns = o
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    for m in turns {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("every message needs a role")?;
        if role != "user" && role != "assistant" {
            return Err(format!("message role '{role}' is not user or assistant"));
        }
        let blocks = match m.get("content") {
            Some(Value::String(s)) => {
                messages.push(json!({"role": role, "content": s}));
                continue;
            }
            Some(Value::Array(b)) => b,
            _ => return Err("message content must be a string or an array of blocks".into()),
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
                Some("tool_use") if role == "assistant" => calls.push(json!({
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": b.get("input").cloned().unwrap_or(json!({})).to_string()},
                })),
                Some("tool_result") if role == "user" => messages.push(json!({
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": block_text(b.get("content").unwrap_or(&Value::Null))?,
                })),
                Some("thinking" | "redacted_thinking") => {}
                other => {
                    return Err(format!(
                        "content block type '{}' is not supported in a {role} message: only text, tool_use (assistant) and tool_result (user)",
                        other.unwrap_or("missing")
                    ))
                }
            }
        }
        if !calls.is_empty() {
            messages.push(json!({"role": role, "content": text, "tool_calls": calls}));
        } else if !text.is_empty() {
            messages.push(json!({"role": role, "content": text}));
        }
    }
    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(messages));
    chat.insert("max_tokens".into(), max_tokens.clone());
    copy(o, &mut chat, &["model", "temperature", "top_p", "top_k", "aien_test"]);
    if let Some(s) = o.get("stop_sequences").filter(|v| !v.is_null()) {
        chat.insert("stop".into(), s.clone());
    }
    if let Some(Value::Array(tools)) = o.get("tools") {
        let tools = tools
            .iter()
            .map(|t| match t.get("type").and_then(Value::as_str) {
                None | Some("custom") => Ok(json!({"type": "function", "function": {
                    "name": t["name"],
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("input_schema").cloned().unwrap_or(json!({})),
                }})),
                Some(other) => Err(format!(
                    "tool type '{other}' is not supported: only custom tools with an input_schema"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !tools.is_empty() {
            chat.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(tc) = o.get("tool_choice").filter(|v| !v.is_null()) {
        let mapped = match tc.get("type").and_then(Value::as_str) {
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some("any") => json!("required"),
            Some("tool") => json!({"type": "function", "function": {"name": tc["name"]}}),
            _ => return Err("tool_choice.type must be auto, none, any or tool".into()),
        };
        chat.insert("tool_choice".into(), mapped);
        if tc.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
            chat.insert("parallel_tool_calls".into(), json!(false));
        }
    }
    Ok(Value::Object(chat))
}

/// A chat completion as a Messages reply.
pub fn chat_to_message(chat: &Value) -> Value {
    let choice = &chat["choices"][0];
    let msg = &choice["message"];
    let mut content = Vec::new();
    if let Some(r) = msg["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
        content.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
    }
    if let Some(t) = msg["content"].as_str().filter(|t| !t.is_empty()) {
        content.push(json!({"type": "text", "text": t}));
    }
    for c in msg["tool_calls"].as_array().map_or(&[][..], Vec::as_slice) {
        let input: Value = c["function"]["arguments"]
            .as_str()
            .and_then(|a| serde_json::from_str(a).ok())
            .unwrap_or(json!({}));
        content.push(json!({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": input}));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let stop_reason = match choice["finish_reason"].as_str() {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    };
    json!({
        "id": format!("msg_{}", id_suffix(chat["id"].as_str().unwrap_or(""))),
        "type": "message",
        "role": "assistant",
        "model": chat["model"],
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": chat["usage"]["prompt_tokens"],
            "output_tokens": chat["usage"]["completion_tokens"],
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_input_items_become_chat_turns() {
        let raw = json!({
            "model": "m",
            "instructions": "be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "get", "arguments": "{\"city\":\"x\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            ],
            "tools": [{"type": "function", "name": "get", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "get"},
            "max_output_tokens": 9,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "text"}},
        });
        let (chat, echo) = responses_to_chat(&raw).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[0], json!({"role": "system", "content": "be brief"}));
        assert_eq!(m[1]["content"][0], json!({"type": "text", "text": "weather?"}));
        assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "get");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"}));
        assert_eq!(chat["tools"][0]["function"]["name"], "get");
        assert_eq!(chat["tool_choice"], json!({"type": "function", "function": {"name": "get"}}));
        assert_eq!(chat["max_completion_tokens"], 9);
        assert_eq!(chat["reasoning_effort"], "low");
        assert_eq!(chat["response_format"], json!({"type": "text"}));
        assert_eq!(echo["max_output_tokens"], 9);
    }

    #[test]
    fn responses_refuses_what_it_cannot_do() {
        for raw in [
            json!({"input": "x", "previous_response_id": "resp_1"}),
            json!({"input": "x", "background": true}),
            json!({"input": "x", "store": true}),
            json!({"input": "x", "tools": [{"type": "web_search"}]}),
            json!({"input": [{"type": "computer_call"}]}),
            json!({"input": 3}),
        ] {
            assert!(responses_to_chat(&raw).is_err(), "{raw}");
        }
    }

    #[test]
    fn chat_reply_becomes_a_response() {
        let chat = json!({
            "id": "chatcmpl-ab", "created": 5, "model": "m",
            "choices": [{"index": 0, "finish_reason": "length", "message": {
                "role": "assistant", "content": "hi", "reasoning_content": "think",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            }}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 2}},
        });
        let r = chat_to_response(&chat, &json!({"tool_choice": "auto"}));
        assert_eq!(r["id"], "resp_ab");
        assert_eq!(r["object"], "response");
        assert_eq!(r["status"], "incomplete");
        assert_eq!(r["incomplete_details"]["reason"], "max_output_tokens");
        let out = r["output"].as_array().unwrap();
        assert_eq!(out[0]["type"], "reasoning");
        assert_eq!(out[1]["content"][0]["text"], "hi");
        assert_eq!(out[2]["call_id"], "call_1");
        assert_eq!(r["usage"]["output_tokens_details"]["reasoning_tokens"], 2);
        assert_eq!(r["usage"]["total_tokens"], 7);

        let m = chat_to_message(&chat);
        assert_eq!(m["id"], "msg_ab");
        assert_eq!(m["stop_reason"], "max_tokens");
        assert_eq!(m["content"][1], json!({"type": "text", "text": "hi"}));
        assert_eq!(m["content"][2]["type"], "tool_use");

        let c = chat_to_text_completion(&chat);
        assert_eq!(c["id"], "cmpl-ab");
        assert_eq!(c["choices"][0]["text"], "hi");
    }

    #[test]
    fn messages_and_completions_requests_convert_or_refuse() {
        let raw = json!({
            "model": "m", "max_tokens": 5, "system": [{"type": "text", "text": "s"}],
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
            ],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "stop_sequences": ["END"],
        });
        let chat = messages_to_chat(&raw).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m[0]["content"], "s");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(m[3]["role"], "tool");
        assert_eq!(chat["stop"], json!(["END"]));
        assert_eq!(chat["tool_choice"], "auto");
        assert!(messages_to_chat(&json!({"messages": [], "max_tokens": 1, "stream": true})).is_err());
        assert!(messages_to_chat(&json!({"messages": []})).is_err());

        let (chat, prompt) = completions_to_chat(&json!({"prompt": ["once"], "max_tokens": 3})).unwrap();
        assert_eq!(prompt, "once");
        assert_eq!(chat["max_tokens"], 3);
        for raw in [
            json!({"prompt": "x", "stream": true}),
            json!({"prompt": "x", "echo": true}),
            json!({"prompt": "x", "suffix": "y"}),
            json!({"prompt": "x", "best_of": 2}),
            json!({"prompt": "x", "logprobs": 2}),
            json!({"prompt": ["a", "b"]}),
        ] {
            assert!(completions_to_chat(&raw).is_err(), "{raw}");
        }
    }
}
