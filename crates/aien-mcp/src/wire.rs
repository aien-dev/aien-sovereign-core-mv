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

use aien_capability::ToolDescriptor;
use serde_json::Value;

use crate::Error;

/// What the session observed from one `tools/call`.
///
/// `Uncertain` means the request may have reached the server. The broker must
/// not send it again.
#[derive(Clone, Debug, PartialEq)]
pub enum CallOutcome {
    Finished(Value),
    Rejected(String),
    Uncertain,
}

/// Provider face used by an admitted session.
///
/// The methods are async so an `rmcp` peer can sit behind the same seam as an
/// in-process wire. The trait stays object-safe: the session stores
/// `Arc<dyn McpWire>` and never hands that handle to J-Space.
#[async_trait::async_trait]
pub trait McpWire: Send + Sync {
    async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, Error>;
    async fn call_tool(&self, name: &str, arguments: &Value) -> Result<CallOutcome, Error>;
}
