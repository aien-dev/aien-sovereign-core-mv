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

use crate::approval::Reservation;
use aien_capability::{Digest32, EffectId, JNodeId, ProviderId, WorldId};
use serde_json::Value;

/// Irreversible work J-Space may record. Staging does not call the provider.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectIntent {
    pub provider: ProviderId,
    pub tool_name: String,
    pub arguments: Value,
    pub capability_digest: Digest32,
}

/// Result of one speculative-safe `tools/call`.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub output: Value,
}

/// Proof that a committed effect finished. The output is the provider result,
/// not a credential.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectReceipt {
    pub effect_id: EffectId,
    pub world_id: WorldId,
    pub winning_jnode: JNodeId,
    pub policy_digest: Digest32,
    pub provider: ProviderId,
    pub tool_name: String,
    pub capability_digest: Digest32,
    pub output: Value,
}

/// An effect the policy boundary has allowed.
///
/// The only production mint is crate-private and reachable solely through
/// [`crate::EffectLane::authorize`], which asks an [`crate::EffectAuthority`]. Fields stay
/// private, so another crate cannot build the value with a struct literal:
///
/// ```compile_fail
/// use aien_capability::{Digest32, EffectId, JNodeId, WorldId};
/// use aien_mcp::AuthorizedEffect;
/// let _ = AuthorizedEffect {
///     intent: (),
///     world_id: WorldId(1),
///     winning_jnode: JNodeId(1),
///     policy_digest: Digest32::of(b"x"),
///     capability_digest: Digest32::of(b"x"),
///     idempotency_key: EffectId::from_label("x"),
/// };
/// ```
///
/// A bare intent cannot be executed:
///
/// ```compile_fail
/// use aien_mcp::{EffectIntent, EffectLane, McpBroker};
/// async fn f(lane: EffectLane, intent: EffectIntent) {
///     let _ = lane.execute_effect(intent).await;
/// }
/// ```
#[derive(Clone, Debug)]
pub struct AuthorizedEffect<T> {
    intent: T,
    world_id: WorldId,
    winning_jnode: JNodeId,
    policy_digest: Digest32,
    capability_digest: Digest32,
    idempotency_key: EffectId,
    reservation: Option<Arc<Reservation>>,
}

impl<T> AuthorizedEffect<T> {
    /// Id of the approval grant this effect holds, if it was minted by `authorize_approved`.
    pub fn approval_id(&self) -> Option<Digest32> {
        self.reservation.as_ref().map(|r| r.grant_id())
    }

    /// Give up this effect without running it. If it holds an approval grant, the grant is
    /// released (usable again once re-checked at the next mint) and `reason` is recorded
    /// ([`crate::ApprovalDesk::release_reason`]). Dropping the effect has the same release with a
    /// fixed reason. Clones share the hold: after a cancel, a clone cannot be executed with the
    /// grant (`Error::ApprovalNotReserved`).
    pub fn cancel(self, reason: &str) {
        if let Some(r) = &self.reservation {
            r.release(reason);
        }
    }

    pub(crate) fn reservation(&self) -> Option<&Reservation> {
        self.reservation.as_deref()
    }

    pub fn intent(&self) -> &T {
        &self.intent
    }

    pub fn world_id(&self) -> WorldId {
        self.world_id
    }

    pub fn winning_jnode(&self) -> JNodeId {
        self.winning_jnode
    }

    pub fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub fn capability_digest(&self) -> Digest32 {
        self.capability_digest
    }

    pub fn idempotency_key(&self) -> EffectId {
        self.idempotency_key
    }
}

/// Test-only mint. Absent from production builds.
#[cfg(test)]
pub(crate) fn authorize_for_test<T>(
    intent: T,
    world_id: WorldId,
    winning_jnode: JNodeId,
    policy_digest: Digest32,
    capability_digest: Digest32,
    idempotency_key: EffectId,
) -> AuthorizedEffect<T> {
    AuthorizedEffect {
        intent,
        world_id,
        winning_jnode,
        policy_digest,
        capability_digest,
        idempotency_key,
        reservation: None,
    }
}

/// Production mint. Crate-private: reached only from `EffectLane::authorize` after an
/// `EffectAuthority` returned `Allow` or an empty `AllowRestricted`.
pub(crate) fn mint<T>(
    intent: T,
    world_id: WorldId,
    winning_jnode: JNodeId,
    policy_digest: Digest32,
    capability_digest: Digest32,
    idempotency_key: EffectId,
) -> AuthorizedEffect<T> {
    AuthorizedEffect {
        intent,
        world_id,
        winning_jnode,
        policy_digest,
        capability_digest,
        idempotency_key,
        reservation: None,
    }
}

/// Production mint for an effect that holds a reserved approval grant.
pub(crate) fn mint_reserved<T>(
    intent: T,
    world_id: WorldId,
    winning_jnode: JNodeId,
    policy_digest: Digest32,
    capability_digest: Digest32,
    idempotency_key: EffectId,
    reservation: Reservation,
) -> AuthorizedEffect<T> {
    let mut effect = mint(
        intent,
        world_id,
        winning_jnode,
        policy_digest,
        capability_digest,
        idempotency_key,
    );
    effect.reservation = Some(Arc::new(reservation));
    effect
}
