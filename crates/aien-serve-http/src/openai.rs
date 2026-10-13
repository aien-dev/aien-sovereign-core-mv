//! OpenAI chat-completions wire types: the subset Odysseus and the OpenAI SDKs
//! send and read. Unknown request fields are ignored (llama.cpp extras such as
//! `session_id` / `cache_prompt` arrive from Odysseus).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub n: Option<i64>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub functions: Option<Vec<Value>>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Prefill-ahead session key (`session`, or Odysseus's `session_id`).
    #[serde(default, alias = "session_id")]
    pub session: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    /// A string, or an array of `{type: "text", text}` parts, or null.
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ErrorBody {
    pub fn new(kind: &str, message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                kind: kind.to_string(),
                param: None,
                code: None,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// OpenAI-style detail: how many prompt tokens were served from resident KV.
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

/// Builds a non-streaming `chat.completion` object.
pub fn completion_json(
    id: &str,
    created: u64,
    model: &str,
    content: &str,
    finish_reason: &str,
    usage: &Usage,
) -> Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    })
}

/// Builds one `chat.completion.chunk` object. `role` is set on the first chunk,
/// `finish_reason` and `usage` on the last.
pub fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
    usage: Option<&Usage>,
) -> Value {
    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".into(), Value::String(r.into()));
    }
    if let Some(c) = content {
        delta.insert("content".into(), Value::String(c.into()));
    }
    let mut v = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": Value::Object(delta),
            "finish_reason": finish_reason,
        }],
    });
    if let Some(u) = usage {
        v["usage"] = serde_json::to_value(u).expect("usage serializes");
    }
    v
}
