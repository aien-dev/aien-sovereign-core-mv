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

use aien_capability::ProviderId;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("provider `{0}` is not admitted to the capability graph")]
    NotAdmitted(ProviderId),
    #[error("provider `{provider}` has no tool `{tool}`")]
    UnknownTool { provider: ProviderId, tool: String },
    #[error("tool effect requires a committed winning world")]
    EffectRequiresCommit,
    #[error("local ephemeral tool requires a sandbox")]
    SandboxRequired,
    #[error("tool is speculation-safe and is not an external effect")]
    SpeculationSafe,
    #[error("authorized capability digest does not match the live catalog")]
    StaleCapability,
    #[error("effect outcome is uncertain and must be reconciled, not retried")]
    ReconciliationRequired,
    #[error("effect is already in flight")]
    EffectInFlight,
    #[error("idempotency key already belongs to a different effect")]
    IdempotencyConflict,
    #[error("tool call rejected: {0}")]
    Rejected(String),
    #[error("tool `{0}` has no enrolled effect class")]
    UnclassifiedTool(String),
    #[error("mcp transport failed: {0}")]
    Transport(String),
    #[error("approval is no longer reserved for this effect (it was cancelled)")]
    ApprovalNotReserved,
    #[error("approval expired before the effect could run; nothing was called")]
    ApprovalExpired,
    #[error(
        "approved effect needs a clock to check expiry; build the lane with EffectLane::with_clock"
    )]
    ApprovalClockMissing,
}
