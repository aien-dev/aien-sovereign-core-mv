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

use crate::{Digest32, ToolEffects};
use sha2::{Digest, Sha256};

/// Atomic operation the capability graph names. An MCP tool name is a binding
/// under this descriptor, not a second identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDescriptor {
    name: String,
    effects: ToolEffects,
    schema_digest: Digest32,
}

impl ToolDescriptor {
    pub fn new(name: impl Into<String>, effects: ToolEffects, schema_digest: Digest32) -> Self {
        Self {
            name: name.into(),
            effects,
            schema_digest,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn effects(&self) -> ToolEffects {
        self.effects
    }

    pub fn capability_digest(&self) -> Digest32 {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update(self.effects.bits().to_le_bytes());
        hasher.update(self.schema_digest.0);
        let hashed = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&hashed);
        Digest32(out)
    }
}

/// Digest of a provider catalog. Order of declaration does not change the digest.
pub fn catalog_digest(tools: &[ToolDescriptor]) -> Digest32 {
    let mut ordered: Vec<&ToolDescriptor> = tools.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    let mut hasher = Sha256::new();
    for tool in ordered {
        hasher.update(tool.capability_digest().0);
    }
    let hashed = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&hashed);
    Digest32(out)
}
