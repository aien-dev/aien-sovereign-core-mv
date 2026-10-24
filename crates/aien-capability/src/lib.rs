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

//! Tool descriptors and effect classification for the AIEN capability graph.
//!
//! A tool declares what it can do. `routing_class` is the highest declared
//! effect. `speculation_safe` is true only when that class may run before a
//! winning World is chosen.

mod effects;
mod ids;
mod tool;

pub use effects::{routing_class, speculation_safe, EffectClass, ToolEffects};
pub use ids::{Digest32, EffectId, JNodeId, ProviderId, WorldId};
pub use tool::{catalog_digest, ToolDescriptor};
