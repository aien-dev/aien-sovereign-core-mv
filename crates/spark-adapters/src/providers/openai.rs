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

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde_json::{json, Value};

use crate::models::{ChatMessage, NormalizedChunk};

pub const DEFAULT_MAX_TOKENS: usize = 4096;

pub fn build_headers(
    api_key: Option<&str>,
    custom_headers: Option<Vec<(String, String)>>,
) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(key) = api_key {
        let auth_val = format!("Bearer {}", key.trim());
        if let Ok(hv) = HeaderValue::from_str(&auth_val) {
            headers.insert(AUTHORIZATION, hv);
        }
    }

    if let Some(custom) = custom_headers {
        for (k, v) in custom {
            if let (Ok(hk), Ok(hv)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                HeaderValue::from_str(&v),
            ) {
                headers.insert(hk, hv);
            }
        }
    }

    headers
}

pub fn format_openai_payload(
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f32>,
    stream: bool,
) -> Value {
    format_openai_payload_bounded(
        model,
        messages,
        temperature,
        Some(DEFAULT_MAX_TOKENS),
        stream,
    )
}

pub fn format_openai_payload_bounded(
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    stream: bool,
) -> Value {
    let formatted_msgs: Vec<Value> = messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role,
                "content": m.content
            })
        })
        .collect();

    let temp_f64: f64 = temperature
        .map(|t| (t as f64 * 1000.0).round() / 1000.0)
        .unwrap_or(0.7);

    let limit = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut payload = json!({
        "model": model,
        "messages": formatted_msgs,
        "temperature": temp_f64,
        "stream": stream,
    });

    if model.starts_with("o1") || model.starts_with("o3") {
        payload["max_completion_tokens"] = json!(limit);
    } else {
        payload["max_tokens"] = json!(limit);
    }

    payload
}

pub fn parse_openai_sse_line(line: &str) -> Option<NormalizedChunk> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return None;
    }

    let payload = trimmed.strip_prefix("data:")?.trim();
    if payload == "[DONE]" {
        return Some(NormalizedChunk {
            content: String::new(),
            reasoning: String::new(),
            is_done: true,
        });
    }

    let val: Value = serde_json::from_str(payload).ok()?;
    let choice = val.get("choices")?.as_array()?.first()?;
    let delta = choice.get("delta")?;

    let content = delta
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let reasoning = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    Some(NormalizedChunk {
        content,
        reasoning,
        is_done: false,
    })
}

pub async fn call_openai_unary(
    client: &Client,
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    messages: &[ChatMessage],
    temperature: Option<f32>,
) -> Result<(String, Option<String>, usize), String> {
    call_openai_unary_bounded(
        client,
        endpoint,
        model,
        api_key,
        messages,
        temperature,
        None,
    )
    .await
}

pub async fn call_openai_unary_bounded(
    client: &Client,
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    messages: &[ChatMessage],
    temperature: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<(String, Option<String>, usize), String> {
    let payload = format_openai_payload_bounded(model, messages, temperature, max_tokens, false);
    let headers = build_headers(api_key, None);

    let res = client
        .post(endpoint)
        .headers(headers)
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Failed to send request to {}: {}", endpoint, e))?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(format!("Provider returned HTTP {}: {}", status, body));
    }

    let val: Value = res
        .json()
        .await
        .map_err(|e| format!("Failed to parse JSON response: {}", e))?;

    let choice = val
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "Malformed response: missing choices array".to_string())?;

    let message = choice
        .get("message")
        .ok_or_else(|| "Missing message object in choice".to_string())?;

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let reasoning = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let token_count = val
        .get("usage")
        .and_then(|u| u.get("total_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0) as usize;

    Ok((content, reasoning, token_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_openai_payload_default_max_tokens() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "ping".to_string(),
            reasoning: None,
        }];
        let payload = format_openai_payload("atlas-lightning-omni", &msgs, Some(0.5), false);

        assert_eq!(payload["max_tokens"], 4096);
        assert_eq!(payload["temperature"], 0.5);
        assert_eq!(payload["model"], "atlas-lightning-omni");
    }

    #[test]
    fn test_format_openai_payload_reasoning_models_use_completion_tokens() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "analyze".to_string(),
            reasoning: None,
        }];
        let payload = format_openai_payload("o3-mini", &msgs, None, false);

        assert_eq!(payload["max_completion_tokens"], 4096);
        assert!(payload.get("max_tokens").is_none());
    }

    #[test]
    fn test_format_openai_payload_custom_bound() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "short response".to_string(),
            reasoning: None,
        }];
        let payload = format_openai_payload_bounded("gpt-4o", &msgs, None, Some(512), false);

        assert_eq!(payload["max_tokens"], 512);
    }
}
