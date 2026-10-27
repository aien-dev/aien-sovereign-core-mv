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

//! Single-use approvals for the host-owned authority seam.
//!
//! When an [`crate::EffectAuthority`] answers `RequireApproval`, a host approver (a human or an
//! approval service, holding an [`ApprovalDesk`]) may issue one [`ApprovalGrant`]. The grant is
//! bound to one effect identity: the intent digest (tool, arguments, catalog), the idempotency
//! key, the world and the winning J-node. Spending is two-phase:
//!
//! 1. [`crate::EffectLane::authorize_approved`] *reserves* the grant when it mints the
//!    `AuthorizedEffect`. While reserved, no other mint can use it ([`ApprovalError::Reserved`]).
//! 2. [`crate::EffectLane::execute_effect`] *commits* the reservation just before the provider is
//!    called: the grant is now spent for good ([`ApprovalError::Consumed`]).
//! 3. If the `AuthorizedEffect` is dropped without running, or cancelled with
//!    [`crate::AuthorizedEffect::cancel`], the reservation is *released*: the grant is available
//!    again, and expiry, revocation and the effect binding are re-checked at the next mint.
//!
//! Limit: the broker keeps grant state in memory only. A restart forgets every grant (each is
//! `Unknown` afterwards), so a reserved or spent grant can never run twice across a restart; the
//! approver must issue a new one.
//!
//! Replay of a finished request does not need a second approval: the existing effect ledger
//! (keyed by the idempotency key) already returns the original receipt, and
//! [`crate::EffectLane::authorize_and_execute_approved`] consults it for a spent grant that is
//! bound to the same effect.
//!
//! A grant that has not been spent can be withdrawn with [`ApprovalDesk::revoke`]; after that
//! it is refused with [`ApprovalError::Revoked`] and cannot be spent.
//!
//! Expiry uses a host-supplied `now` and `expires_at` in the same unit. The crate reads no clock.

use std::sync::atomic::{AtomicBool, Ordering};

use aien_capability::{Digest32, EffectId, JNodeId, WorldId};

use crate::authority::{intent_digest, EffectScope};
use crate::broker::{Inner, McpBroker};
use crate::EffectIntent;

/// Why an approval could not be spent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalError {
    /// The grant was not issued by this broker's desk.
    Unknown,
    /// The grant is bound to a different effect than the one presented. Nothing is spent.
    Mismatch,
    /// The grant was already spent: an effect minted from it reached `execute_effect`.
    Consumed,
    /// The grant is reserved by a live `AuthorizedEffect` that has not run yet. It becomes
    /// usable again if that effect is dropped or cancelled un-executed.
    Reserved,
    /// `now` is at or past `expires_at`. Nothing is spent.
    Expired,
    /// The approver withdrew the grant with [`ApprovalDesk::revoke`] before it was spent.
    Revoked,
}

/// Proof that a host approver approved exactly one effect. Fields are private and only an
/// [`ApprovalDesk`] can create one. Cloning does not duplicate it: spending is tracked by the
/// broker, not by the value.
#[derive(Clone, Debug)]
pub struct ApprovalGrant {
    id: Digest32,
}

#[derive(Debug)]
pub(crate) struct ApprovalRecord {
    pub(crate) intent_digest: Digest32,
    pub(crate) idempotency_key: EffectId,
    pub(crate) world_id: WorldId,
    pub(crate) winning_jnode: JNodeId,
    pub(crate) expires_at: u64,
    pub(crate) state: GrantState,
    pub(crate) revoked: bool,
    pub(crate) release_reason: Option<String>,
}

impl ApprovalRecord {
    pub(crate) fn binds(&self, intent: &EffectIntent, scope: &EffectScope) -> bool {
        self.intent_digest == intent_digest(intent)
            && self.idempotency_key == scope.idempotency_key
            && self.world_id == scope.world_id
            && self.winning_jnode == scope.winning_jnode
    }
}

impl ApprovalGrant {
    pub(crate) fn id(&self) -> Digest32 {
        self.id
    }
}

/// The approver's handle. Held by the host's approval service, not by J-Space or the effect
/// lane's callers. Limit: any code holding a `McpBroker` or `SessionManager` can obtain one, so
/// the separation is by handle discipline, not by type.
#[derive(Clone)]
pub struct ApprovalDesk {
    broker: McpBroker,
}

impl ApprovalDesk {
    pub fn new(broker: McpBroker) -> Self {
        Self { broker }
    }

    /// Issue one grant for exactly this intent and scope, valid while `now < expires_at`.
    pub fn issue(
        &self,
        intent: &EffectIntent,
        scope: EffectScope,
        expires_at: u64,
    ) -> ApprovalGrant {
        let mut inner = self.broker.lock();
        inner.next_approval = inner.next_approval.saturating_add(1);
        let digest = intent_digest(intent);
        let mut bytes = inner.next_approval.to_le_bytes().to_vec();
        bytes.extend_from_slice(&digest.0);
        bytes.extend_from_slice(&scope.idempotency_key.0);
        let id = Digest32::of(&bytes);
        inner.approvals.insert(
            id,
            ApprovalRecord {
                intent_digest: digest,
                idempotency_key: scope.idempotency_key,
                world_id: scope.world_id,
                winning_jnode: scope.winning_jnode,
                expires_at,
                state: GrantState::Available,
                revoked: false,
                release_reason: None,
            },
        );
        ApprovalGrant { id }
    }

    /// Withdraw a grant that has not been spent. Returns `true` if the grant is now dead.
    /// A grant that was already spent (an effect may have been minted or run), or that this
    /// desk never issued, returns `false` and nothing changes. Revoking twice is `true` both times.
    pub fn revoke(&self, grant: &ApprovalGrant) -> bool {
        let mut inner = self.broker.lock();
        match inner.approvals.get_mut(&grant.id()) {
            Some(record) if record.state == GrantState::Available => {
                record.revoked = true;
                true
            }
            _ => false,
        }
    }
}

/// Where a grant is in its life. Private: hosts see [`GrantStatus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GrantState {
    Available,
    /// Held by the `AuthorizedEffect` carrying this reservation token.
    Reserved(u64),
    Spent,
}

/// What a host can ask about a grant with [`ApprovalDesk::status`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantStatus {
    /// Unspent and unreserved. It may still be expired; expiry is judged at mint against the
    /// host's `now`.
    Available,
    /// A live `AuthorizedEffect` holds it and has not run.
    Reserved,
    /// An effect minted from it reached `execute_effect`. Final.
    Spent,
    /// The approver withdrew it before it was spent.
    Revoked,
}

/// The broker-side hold a minted effect carries. Shared by clones of the effect (`Arc`), so the
/// grant is released when the last clone is dropped un-executed, or when any clone cancels.
pub(crate) struct Reservation {
    broker: McpBroker,
    grant: Digest32,
    token: u64,
    finished: AtomicBool,
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("token", &self.token)
            .finish_non_exhaustive()
    }
}

impl Reservation {
    pub(crate) fn new(broker: McpBroker, grant: Digest32, token: u64) -> Self {
        Self {
            broker,
            grant,
            token,
            finished: AtomicBool::new(false),
        }
    }

    pub(crate) fn grant_id(&self) -> Digest32 {
        self.grant
    }

    /// Make the spend final. The caller already holds the broker lock (`inner`). Fails if the
    /// reservation was released (cancelled) first.
    pub(crate) fn commit_locked(
        &self,
        inner: &mut Inner,
        now: Option<u64>,
    ) -> Result<(), crate::Error> {
        match inner.approvals.get_mut(&self.grant) {
            Some(r) if r.state == GrantState::Reserved(self.token) => {
                // Fail closed: without a clock expiry cannot be checked, so an approved effect
                // does not run. The grant is given back.
                let Some(now) = now else {
                    r.state = GrantState::Available;
                    r.release_reason = Some("no clock: expiry cannot be checked".to_string());
                    self.finished.store(true, Ordering::SeqCst);
                    return Err(crate::Error::ApprovalClockMissing);
                };
                if now >= r.expires_at {
                    // Fail closed: give the grant back (a later mint refuses it as Expired).
                    r.state = GrantState::Available;
                    r.release_reason = Some("expired before execute".to_string());
                    self.finished.store(true, Ordering::SeqCst);
                    return Err(crate::Error::ApprovalExpired);
                }
                r.state = GrantState::Spent;
                self.finished.store(true, Ordering::SeqCst);
                Ok(())
            }
            _ => Err(crate::Error::ApprovalNotReserved),
        }
    }

    /// Give the grant back. Idempotent; a no-op after commit or an earlier release.
    pub(crate) fn release(&self, reason: &str) {
        if self.finished.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut inner = self.broker.lock();
        if let Some(r) = inner.approvals.get_mut(&self.grant) {
            if r.state == GrantState::Reserved(self.token) {
                r.state = GrantState::Available;
                r.release_reason = Some(reason.to_string());
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.release("dropped before execute_effect");
    }
}

impl ApprovalGrant {
    /// Stable id of this grant (a digest). Hosts may use it as a correlation handle; it carries
    /// no authority, since spending needs the broker's own record.
    pub fn approval_id(&self) -> Digest32 {
        self.id
    }
}

impl ApprovalDesk {
    /// Current status of `grant`, or `None` if this desk never issued it.
    pub fn status(&self, grant: &ApprovalGrant) -> Option<GrantStatus> {
        let inner = self.broker.lock();
        let r = inner.approvals.get(&grant.id())?;
        Some(match r.state {
            GrantState::Spent => GrantStatus::Spent,
            GrantState::Reserved(_) => GrantStatus::Reserved,
            GrantState::Available if r.revoked => GrantStatus::Revoked,
            GrantState::Available => GrantStatus::Available,
        })
    }

    /// Why the grant was last released (`None` if it never was, or this desk did not issue it).
    pub fn release_reason(&self, grant: &ApprovalGrant) -> Option<String> {
        let inner = self.broker.lock();
        inner.approvals.get(&grant.id())?.release_reason.clone()
    }
}
