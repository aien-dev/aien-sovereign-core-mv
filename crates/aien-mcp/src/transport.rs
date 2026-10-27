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

//! `rmcp` client and server transports behind [`crate::McpWire`].
//!
//! Effect class is not read from the server. Enrollment supplies it. A listed
//! tool with no enrolled class fails discovery instead of being treated as safe.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use aien_capability::{Digest32, ToolDescriptor, ToolEffects};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    JsonObject, ListToolsResult, ServerCapabilities, ServerConfig,
};
use rmcp::service::{ClientCacheConfig, RequestContext, RoleClient, RoleServer, RunningService};
use rmcp::transport::IntoTransport;
use rmcp::{Peer, ServerHandler, ServiceExt};
use serde_json::Value;

use crate::{CallOutcome, Error, McpWire};

/// Client peer for one admitted provider. The `rmcp` session stays inside this type.
pub(crate) struct RmcpClientWire {
    peer: Peer<RoleClient>,
    service: Mutex<Option<RunningService<RoleClient, ()>>>,
    effects: HashMap<String, ToolEffects>,
}

impl RmcpClientWire {
    pub(crate) async fn connect<T, E, A>(
        transport: T,
        effects: HashMap<String, ToolEffects>,
    ) -> Result<Self, Error>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ().serve(transport).await.map_err(|err| Error::Transport(err.to_string()))?;
        service
            .set_response_cache_config(ClientCacheConfig::disabled())
            .await;
        let peer = service.peer().clone();
        Ok(Self {
            peer,
            service: Mutex::new(Some(service)),
            effects,
        })
    }
}

impl Drop for RmcpClientWire {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.service.lock() {
            drop(guard.take());
        }
    }
}

#[async_trait::async_trait]
impl McpWire for RmcpClientWire {
    async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, Error> {
        let listed = self
            .peer
            .list_all_tools()
            .await
            .map_err(|err| Error::Transport(err.to_string()))?;
        let mut descriptors = Vec::with_capacity(listed.len());
        let mut seen = std::collections::HashSet::new();
        for tool in listed {
            let name = tool.name.into_owned();
            if !seen.insert(name.clone()) {
                return Err(Error::Transport(format!("duplicate tool `{name}`")));
            }
            let Some(effects) = self.effects.get(&name).copied() else {
                return Err(Error::UnclassifiedTool(name));
            };
            let schema = Value::Object((*tool.input_schema).clone());
            descriptors.push(ToolDescriptor::new(
                name,
                effects,
                Digest32::of(&canonical_bytes(&schema)),
            ));
        }
        Ok(descriptors)
    }

    async fn call_tool(&self, name: &str, arguments: &Value) -> Result<CallOutcome, Error> {
        let arguments = match arguments {
            Value::Null => None,
            Value::Object(map) => Some(map.clone()),
            _ => {
                return Ok(CallOutcome::Rejected(
                    "tool arguments must be a JSON object".into(),
                ));
            }
        };
        let mut params = CallToolRequestParams::new(name.to_owned());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        match self.peer.call_tool(params).await {
            Ok(result) => Ok(outcome_from(result)),
            Err(err) => Err(Error::Transport(err.to_string())),
        }
    }
}

fn outcome_from(result: CallToolResult) -> CallOutcome {
    if result.is_error.unwrap_or(false) {
        let reason = result
            .content
            .iter()
            .find_map(|block| block.as_text().map(|text| text.text.clone()))
            .unwrap_or_else(|| "tool call failed".into());
        return CallOutcome::Rejected(reason);
    }
    if let Some(value) = result.structured_content {
        return CallOutcome::Finished(value);
    }
    let text = result
        .content
        .iter()
        .find_map(|block| block.as_text().map(|text| text.text.clone()))
        .unwrap_or_default();
    CallOutcome::Finished(Value::String(text))
}

type ToolCall = Arc<dyn Fn(&Value) -> Result<Value, String> + Send + Sync>;

/// One tool the in-process server exposes over `rmcp`.
pub struct LocalTool {
    name: String,
    description: String,
    input_schema: JsonObject,
    call: ToolCall,
}

impl LocalTool {
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonObject,
        call: F,
    ) -> Self
    where
        F: Fn(&Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            call: Arc::new(call) as ToolCall,
        }
    }
}

/// Server transport. It speaks `tools/list` and `tools/call` and does not mint
/// [`crate::AuthorizedEffect`].
#[derive(Clone)]
pub struct LocalToolServer {
    tools: Arc<Mutex<Vec<LocalTool>>>,
}

impl LocalToolServer {
    pub fn new(tools: Vec<LocalTool>) -> Self {
        Self {
            tools: Arc::new(Mutex::new(tools)),
        }
    }

    /// Replace the advertised catalog. The next admitted refresh observes it.
    pub fn replace_tools(&self, tools: Vec<LocalTool>) {
        *self
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = tools;
    }

    pub async fn serve<T, E, A>(
        &self,
        transport: T,
    ) -> Result<RunningService<RoleServer, Self>, Error>
    where
        T: IntoTransport<RoleServer, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        self.clone()
            .serve(transport)
            .await
            .map_err(|err| Error::Transport(err.to_string()))
    }
}

impl ServerHandler for LocalToolServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("aien-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions("AIEN MCP transport. Effect class is enrolled separately.")
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let tools = self
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let listed = tools
            .iter()
            .map(|tool| {
                rmcp::model::Tool::new(
                    tool.name.clone(),
                    tool.description.clone(),
                    Arc::new(tool.input_schema.clone()),
                )
            })
            .collect();
        Ok(ListToolsResult::with_all_items(listed).with_ttl_ms(0))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, rmcp::ErrorData> {
        let tools = self
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(tool) = tools.iter().find(|tool| tool.name == request.name.as_ref()) else {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "unknown tool {}",
                request.name
            ))])
            .into());
        };
        let call = tool.call.clone();
        drop(tools);
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        match call(&arguments) {
            Ok(value) => Ok(CallToolResult::structured(value).into()),
            Err(reason) => Ok(CallToolResult::error(vec![ContentBlock::text(reason)]).into()),
        }
    }
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
        Value::String(text) => {
            if let Ok(encoded) = serde_json::to_vec(text) {
                out.extend_from_slice(&encoded);
            }
        }
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                if let Ok(encoded) = serde_json::to_vec(key) {
                    out.extend_from_slice(&encoded);
                }
                out.push(b':');
                write_canonical(&map[*key], out);
            }
            out.push(b'}');
        }
    }
}
