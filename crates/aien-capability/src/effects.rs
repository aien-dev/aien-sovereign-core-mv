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

/// Operations a tool declares. A tool may set more than one bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToolEffects(u32);

impl ToolEffects {
    pub const PURE: Self = Self(0);
    pub const READ_FILESYSTEM: Self = Self(1 << 0);
    pub const READ_NETWORK: Self = Self(1 << 1);
    pub const SPAWN_PROCESS: Self = Self(1 << 2);
    pub const WORLD_MUTATION: Self = Self(1 << 3);
    pub const LOCAL_EPHEMERAL: Self = Self(1 << 4);
    pub const EXTERNAL_WRITE: Self = Self(1 << 5);
    pub const EXTERNAL_IRREVERSIBLE: Self = Self(1 << 6);
    pub const SECRET_BEARING: Self = Self(1 << 7);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub const fn contains(self, other: Self) -> bool {
        other.0 != 0 && self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for ToolEffects {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Single routing class derived from the highest declared effect bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectClass {
    Pure,
    ReadOnly,
    LocalEphemeral,
    WorldMutation,
    ExternalWrite,
    ExternalIrreversible,
}

const UNSAFE_FOR_SPECULATION: ToolEffects = ToolEffects(
    ToolEffects::WORLD_MUTATION.0
        | ToolEffects::EXTERNAL_WRITE.0
        | ToolEffects::EXTERNAL_IRREVERSIBLE.0,
);

pub fn routing_class(effects: ToolEffects) -> EffectClass {
    if effects.contains(ToolEffects::EXTERNAL_IRREVERSIBLE) {
        EffectClass::ExternalIrreversible
    } else if effects.contains(ToolEffects::EXTERNAL_WRITE) {
        EffectClass::ExternalWrite
    } else if effects.contains(ToolEffects::WORLD_MUTATION) {
        EffectClass::WorldMutation
    } else if effects.intersects(ToolEffects::SPAWN_PROCESS | ToolEffects::LOCAL_EPHEMERAL) {
        EffectClass::LocalEphemeral
    } else if effects.intersects(ToolEffects::READ_FILESYSTEM | ToolEffects::READ_NETWORK) {
        EffectClass::ReadOnly
    } else {
        EffectClass::Pure
    }
}

/// True when J-Space may request the call before a winning World exists.
///
/// `SECRET_BEARING` does not by itself make a call unsafe. The session owner
/// materializes credentials. The branch receives the tool result.
pub fn speculation_safe(effects: ToolEffects) -> bool {
    !effects.intersects(UNSAFE_FOR_SPECULATION)
}
