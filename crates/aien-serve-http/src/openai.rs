//! OpenAI chat-completions wire types: the subset Odysseus and the OpenAI SDKs
//! send and read. Unknown request fields are ignored (llama.cpp extras such as
//! `session_id` / `cache_prompt` arrive from Odysseus).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sampling::SamplingRecord;

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
    /// Template switches (`{"enable_thinking": bool, "reasoning_effort": "low|medium|xhigh",
    /// "preserve_thinking": bool}`), read only by templates that define them (Qwen3.8-27B:
    /// thinking and effort; Qwen3.6-35B-A3B: thinking and preserve).
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    /// Same as `chat_template_kwargs.reasoning_effort` (the OpenAI field name).
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// AIEN extra: return the generated token ids in `usage.aien_token_ids` (correctness gates).
    #[serde(default)]
    pub aien_return_token_ids: bool,
    /// AIEN extra, the merge train (L384): per-request test switches
    /// (`{"stroke": bool, "exact_head": bool, "mutant": k, "row_swap": k, ...}`, see
    /// [`crate::engine::TestSwitches`]). Honored only by a build with the cargo feature
    /// `aien-test-switches`; every other build answers 400 when the field is present.
    #[serde(default)]
    pub aien_test: Option<Value>,
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
    /// Assistant turn: the calls it made (OpenAI shape).
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
    /// Tool turn: which call it answers (not rendered by the Qwen template).
    #[serde(default)]
    pub tool_call_id: Option<String>,
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
    /// How the reply was sampled (OWN-8B stage S): an extra field, ignored by OpenAI clients.
    /// Its `seed` replays the request. Absent for an engine that does not sample.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aien_sampling: Option<SamplingRecord>,
    /// The generated token ids, when the request set `aien_return_token_ids`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aien_token_ids: Option<Vec<u32>>,
    /// The merge train's switches this request set (`{"applied": [names]}`), when it set any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aien_test: Option<Value>,
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
    reasoning: Option<&str>,
    finish_reason: &str,
    usage: &Usage,
) -> Value {
    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if let Some(r) = reasoning {
        message["reasoning_content"] = Value::String(r.into());
    }
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
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
    chunk_json_with(
        id,
        created,
        model,
        role,
        content,
        None,
        finish_reason,
        usage,
    )
}

/// [`chunk_json`] with a `reasoning_content` delta (the thinking block of a thinking model).
#[allow(clippy::too_many_arguments)]
pub fn chunk_json_with(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
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
    if let Some(r) = reasoning {
        delta.insert("reasoning_content".into(), Value::String(r.into()));
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
