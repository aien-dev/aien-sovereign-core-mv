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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    OpenRouter,
    Gemini,
    Groq,
    Ollama,
    DeepSeek,
    Together,
    Mistral,
    ChatGPTWeb,
    ClaudeWeb,
    /// Local OpenAI-compatible model seat on port 18006. Accepts the old `local_max` id.
    #[serde(alias = "local_max")]
    LocalSeat,
    EmbeddedNative,
}

impl ProviderType {
    pub fn slug(&self) -> &'static str {
        match self {
            ProviderType::OpenAI => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::OpenRouter => "openrouter",
            ProviderType::Gemini => "gemini",
            ProviderType::Groq => "groq",
            ProviderType::Ollama => "ollama",
            ProviderType::DeepSeek => "deepseek",
            ProviderType::Together => "together",
            ProviderType::Mistral => "mistral",
            ProviderType::ChatGPTWeb => "chatgpt_web",
            ProviderType::ClaudeWeb => "claude_web",
            ProviderType::LocalSeat => "local_seat",
            ProviderType::EmbeddedNative => "embedded_native",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ProviderType::OpenAI => "OpenAI",
            ProviderType::Anthropic => "Anthropic",
            ProviderType::OpenRouter => "OpenRouter (Multi-Model Gateway)",
            ProviderType::Gemini => "Google Gemini",
            ProviderType::Groq => "Groq LPU Engine",
            ProviderType::Ollama => "Ollama Local Engine",
            ProviderType::DeepSeek => "DeepSeek Cloud",
            ProviderType::Together => "Together AI",
            ProviderType::Mistral => "Mistral AI",
            ProviderType::ChatGPTWeb => "ChatGPT Plus (Web Session)",
            ProviderType::ClaudeWeb => "Claude Pro (Web Session)",
            ProviderType::LocalSeat => "Local Model Seat (GB10, port 18006)",
            ProviderType::EmbeddedNative => "Embedded Native (In-Process GB10/CPU)",
        }
    }

    pub fn default_endpoint(&self) -> &'static str {
        match self {
            ProviderType::OpenAI => "https://api.openai.com/v1/chat/completions",
            ProviderType::Anthropic => "https://api.anthropic.com/v1/messages",
            ProviderType::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
            ProviderType::Gemini => {
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            }
            ProviderType::Groq => "https://api.groq.com/openai/v1/chat/completions",
            ProviderType::Ollama => "http://127.0.0.1:11434/v1/chat/completions",
            ProviderType::DeepSeek => "https://api.deepseek.com/v1/chat/completions",
            ProviderType::Together => "https://api.together.xyz/v1/chat/completions",
            ProviderType::Mistral => "https://api.mistral.ai/v1/chat/completions",
            ProviderType::ChatGPTWeb => "https://chatgpt.com/backend-api/conversation",
            ProviderType::ClaudeWeb => "https://claude.ai/api/organizations/chat_conversations",
            ProviderType::LocalSeat => "http://127.0.0.1:18006/v1/chat/completions",
            ProviderType::EmbeddedNative => "in-process://native-transformer",
        }
    }

    pub fn key_name(&self) -> Option<&'static str> {
        match self {
            ProviderType::OpenAI => Some("OPENAI_API_KEY"),
            ProviderType::Anthropic => Some("ANTHROPIC_API_KEY"),
            ProviderType::OpenRouter => Some("OPENROUTER_API_KEY"),
            ProviderType::Gemini => Some("GEMINI_API_KEY"),
            ProviderType::Groq => Some("GROQ_API_KEY"),
            ProviderType::Ollama => None,
            ProviderType::DeepSeek => Some("DEEPSEEK_API_KEY"),
            ProviderType::Together => Some("TOGETHER_API_KEY"),
            ProviderType::Mistral => Some("MISTRAL_API_KEY"),
            ProviderType::ChatGPTWeb => Some("CHATGPT_SESSION_TOKEN"),
            ProviderType::ClaudeWeb => Some("CLAUDE_SESSION_KEY"),
            ProviderType::LocalSeat => None,
            ProviderType::EmbeddedNative => None,
        }
    }

    pub fn key_aliases(&self) -> &'static [&'static str] {
        match self {
            ProviderType::OpenAI => &["OPENAI_API_KEY"],
            ProviderType::Anthropic => &["ANTHROPIC_API_KEY"],
            ProviderType::OpenRouter => &["OPENROUTER_API_KEY"],
            ProviderType::Gemini => &["GEMINI_API_KEY"],
            ProviderType::Groq => &["GROQ_API_KEY"],
            ProviderType::Ollama => &[],
            ProviderType::DeepSeek => &["DEEPSEEK_API_KEY"],
            ProviderType::Together => &["TOGETHER_API_KEY"],
            ProviderType::Mistral => &["MISTRAL_API_KEY"],
            ProviderType::ChatGPTWeb => &["CHATGPT_SESSION_TOKEN", "OPENAI_WEB_TOKEN"],
            ProviderType::ClaudeWeb => &["CLAUDE_SESSION_KEY", "CLAUDE_WEB_SESSION_KEY"],
            ProviderType::LocalSeat => &[],
            ProviderType::EmbeddedNative => &[],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub id: String,
    pub name: String,
    pub provider: ProviderType,
    pub endpoint: String,
    pub model_id: String,
    pub context_length: usize,
    pub is_available: bool,
    pub requires_key: bool,
    pub is_resident: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedChunk {
    pub content: String,
    pub reasoning: String,
    pub is_done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    CodeSynthesis,
    SystemArchitecture,
    ReasoningTrace,
    RefactorLogic,
    VerificationHarness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStrategy {
    CompilerCheck,
    JsonSchema,
    UnslopStrict,
    DualConsensus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationOutcome {
    pub passed: bool,
    pub score: f32,
    pub rule_violations: Vec<String>,
    pub compiler_output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rollout {
    pub provider: ProviderType,
    pub model_id: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub duration_ms: u64,
    pub token_count: usize,
    pub verification: VerificationOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTask {
    pub id: String,
    pub task_type: TaskType,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub teacher_model: String,
    pub student_model: Option<String>,
    pub verification_strategy: VerificationStrategy,
    pub commit_to_cortex: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationRecord {
    pub task_id: String,
    pub task_type: TaskType,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub teacher_rollout: Rollout,
    pub student_rollout: Option<Rollout>,
    pub chosen: String,
    pub chosen_reasoning: Option<String>,
    pub rejected: Option<String>,
    pub rejected_reasoning: Option<String>,
    pub preference_delta: f32,
    pub durable_memory_committed: bool,
    pub cortex_entity_id: Option<String>,
    pub created_at: String,
}

impl DistillationRecord {
    pub fn has_dpo_pair(&self) -> bool {
        self.preference_delta > 0.0
            && self
                .rejected
                .as_deref()
                .map(|rej| rej != self.chosen.as_str())
                .unwrap_or(false)
    }
}
