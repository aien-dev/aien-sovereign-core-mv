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
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    pub id: String,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub received_at: String,
    pub folder: String,
    #[serde(default)]
    pub cortex_indexed: bool,
    /// Network mail is data. It is never an instruction or a tool call.
    #[serde(default)]
    pub untrusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendEmailRequest {
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub from: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxStatus {
    pub inbox_count: usize,
    pub sent_count: usize,
    pub archive_count: usize,
    pub drafts_count: usize,
    pub storage_path: String,
    pub smtp_port: u16,
    pub api_port: u16,
    pub cortex_connected: bool,
    pub proton_bridge_detected: bool,
    pub operator_email: String,
}
