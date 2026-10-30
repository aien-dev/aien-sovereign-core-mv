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

use sha2::{Digest, Sha256};

/// SHA-256 digest used for catalogs, policies, and capability identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Digest32(pub [u8; 32]);

impl Digest32 {
    pub fn of(bytes: &[u8]) -> Self {
        let hashed = Sha256::digest(bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&hashed);
        Self(out)
    }
}

/// Stable idempotency key for one external effect attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId(pub [u8; 16]);

impl EffectId {
    pub fn from_label(label: &str) -> Self {
        let digest = Digest32::of(label.as_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&digest.0[..16]);
        Self(id)
    }
}

/// Admitted MCP (or other) provider. Enrollment lives outside J-Space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Branchable execution state that a winning plan commits into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldId(pub u64);

/// J-Space node that produced a staged effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JNodeId(pub u64);
