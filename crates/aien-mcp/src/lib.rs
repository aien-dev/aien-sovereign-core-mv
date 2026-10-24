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

//! MCP session owner for AIEN.
//!
//! `aien-mcp` keeps the live session. J-Space receives a [`CapabilitySnapshot`]
//! and may invoke only speculation-safe tools. An irreversible `tools/call`
//! runs only when the Effect Broker presents an [`AuthorizedEffect`].

mod approval;
mod authority;
mod broker;
mod effect;
mod error;
pub mod memory;
mod session;
mod snapshot;
mod transport;
mod wire;

#[cfg(test)]
mod lane_tests;

#[cfg(test)]
mod transport_tests;

#[cfg(test)]
pub(crate) use effect::authorize_for_test;

pub use approval::{ApprovalDesk, ApprovalError, ApprovalGrant, GrantStatus};
pub use authority::{
    intent_digest, AuthorityContext, AuthorityDecision, AuthorityOutcome, EffectAuthority,
    EffectClassAuthority, EffectScope, Exposure, TrustLevel,
};
pub use broker::{
    receipt_digest, Clock, EffectLane, McpBroker, SpeculativeLane, SpeculativeToolCall,
};
pub use effect::{AuthorizedEffect, EffectIntent, EffectReceipt, ToolResult};
pub use error::Error;
pub use session::SessionManager;
pub use snapshot::CapabilitySnapshot;
pub use transport::{LocalTool, LocalToolServer};
pub use wire::{CallOutcome, McpWire};
