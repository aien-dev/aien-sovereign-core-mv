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

#[allow(dead_code)]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DiscordUser {
    pub id: String,
    pub username: String,
    pub bot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DiscordMessage {
    pub id: String,
    pub channel_id: String,
    pub content: String,
    pub author: DiscordUser,
    pub guild_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateMessagePayload {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_reference: Option<MessageReference>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageReference {
    pub message_id: String,
}
