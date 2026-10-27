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

use std::sync::Arc;

use aien_capability::ToolDescriptor;
use serde_json::Value;

use crate::{CallOutcome, Error, McpWire};

type ToolHandler = dyn Fn(&str, &Value) -> CallOutcome + Send + Sync;

/// In-process provider used by tests and by local dry runs of the lanes.
pub struct MemoryWire {
    tools: Vec<ToolDescriptor>,
    call: Arc<ToolHandler>,
}

impl MemoryWire {
    pub fn new<F>(tools: Vec<ToolDescriptor>, call: F) -> Self
    where
        F: Fn(&str, &Value) -> CallOutcome + Send + Sync + 'static,
    {
        Self {
            tools,
            call: Arc::new(call),
        }
    }
}

#[async_trait::async_trait]
impl McpWire for MemoryWire {
    async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, Error> {
        Ok(self.tools.clone())
    }

    async fn call_tool(&self, name: &str, arguments: &Value) -> Result<CallOutcome, Error> {
        Ok((self.call)(name, arguments))
    }
}
