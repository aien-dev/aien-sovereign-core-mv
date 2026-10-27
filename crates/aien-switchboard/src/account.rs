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

//! Accounts. An account names where its program keeps its own login, never
//! the login itself.

use serde::Deserialize;
use std::fmt;
use std::path::PathBuf;

/// Short operator-chosen name such as `anthropic-1`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(transparent)]
pub struct AccountId(pub String);

impl AccountId {
    pub fn new(s: impl Into<String>) -> Self {
        AccountId(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AccountId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Anthropic,
    #[serde(rename = "openai")]
    OpenAI,
    Google,
    Xai,
    #[serde(rename = "openrouter")]
    OpenRouter,
    Local,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
            Provider::Google => "google",
            Provider::Xai => "xai",
            Provider::OpenRouter => "openrouter",
            Provider::Local => "local",
        }
    }
}

/// How an account proves who it is. Both variants hold a pointer, never a
/// secret value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    /// The company's own program keeps its login in this settings folder
    /// (one folder per account). The router never opens it.
    SettingsFolder(PathBuf),
    /// Name of a key held in the vault (for example an OpenRouter key). The
    /// value is resolved elsewhere, in process memory only.
    KeyRef(String),
    /// Nothing needed (the local model).
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Account {
    pub id: AccountId,
    pub provider: Provider,
    /// Name of the program that serves this account, such as `claude`.
    /// UNKNOWN for `agy` and `grok`: how a second login is kept side by side
    /// is not confirmed (see ADR 0030, open UNKNOWNs).
    pub program: String,
    pub credential: Credential,
}

impl Account {
    pub fn is_local(&self) -> bool {
        self.provider == Provider::Local
    }
}
