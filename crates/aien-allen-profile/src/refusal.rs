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

//! Every way a profile operation can say no, in plain words.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileRefusal {
    /// ALLEN is not engaged on this machine: there is no identity to attach a profile to.
    NotEngaged,
    /// `expected` revision differs from the current one (or another writer won the race).
    StaleUpdate {
        expected: u64,
        current: u64,
    },
    /// A stored revision names another agent or root than the engaged identity.
    ForeignProfile(String),
    /// A stored file is damaged, has an unknown schema/field, is oversize, or breaks the chain.
    Damaged(String),
    /// A requested value is not allowed.
    Invalid(String),
    /// Permissions are not profile fields.
    PermissionKey(String),
    /// The change would leave the profile exactly as it is.
    NothingToChange,
    /// `revert --to N` names a revision that does not exist.
    UnknownRevision(u64),
    /// Reset or revert with no profile yet.
    NoProfile,
    Io(String),
    /// Test seam only (feature `fault-injection`): the write was stopped on purpose.
    Crashed(String),
}

impl ProfileRefusal {
    /// Short stable code for scripts.
    pub fn code(&self) -> &'static str {
        use ProfileRefusal::*;
        match self {
            NotEngaged => "not_engaged",
            StaleUpdate { .. } => "stale_update",
            ForeignProfile(_) => "foreign_profile",
            Damaged(_) => "damaged",
            Invalid(_) => "invalid",
            PermissionKey(_) => "permission_key",
            NothingToChange => "nothing_to_change",
            UnknownRevision(_) => "unknown_revision",
            NoProfile => "no_profile",
            Io(_) => "io",
            Crashed(_) => "crashed",
        }
    }
}

impl fmt::Display for ProfileRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ProfileRefusal::*;
        match self {
            NotEngaged => write!(
                f,
                "ALLEN is not engaged on this machine, so there is no identity to attach a profile to. Nothing was changed."
            ),
            StaleUpdate { expected, current } => write!(
                f,
                "The profile is at revision {current}, not {expected}. Someone or something changed it since you looked. Show it again, then retry with --expect {current}. Nothing was changed."
            ),
            ForeignProfile(w) => write!(
                f,
                "The saved profile belongs to a different ALLEN identity ({w}). It was not used or changed."
            ),
            Damaged(w) => write!(
                f,
                "The saved profile is damaged or not understood ({w}). It was not replaced; ask the operator to inspect the profile folder."
            ),
            Invalid(w) => write!(f, "Not allowed: {w}. Nothing was changed."),
            PermissionKey(k) => write!(
                f,
                "Not allowed: \"{k}\" looks like a permission. Permissions are not profile fields; approvals go through the approval desk. Nothing was changed."
            ),
            NothingToChange => write!(f, "That would leave the profile exactly as it is. Nothing was changed."),
            UnknownRevision(n) => write!(f, "There is no revision {n} in this profile's history. Nothing was changed."),
            NoProfile => write!(f, "No profile has been saved yet, so there is nothing to revert or reset. Nothing was changed."),
            Io(w) => write!(f, "Could not read or write the profile store: {w}. Nothing was changed."),
            Crashed(w) => write!(f, "write stopped on purpose (test): {w}"),
        }
    }
}

impl std::error::Error for ProfileRefusal {}
