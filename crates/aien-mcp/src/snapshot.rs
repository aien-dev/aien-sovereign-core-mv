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
use std::time::SystemTime;

use aien_capability::{Digest32, ProviderId, ToolDescriptor};

/// Immutable view J-Space plans against. It is not a session handle.
#[derive(Clone, Debug)]
pub struct CapabilitySnapshot {
    pub provider: ProviderId,
    pub session_epoch: u64,
    pub catalog_digest: Digest32,
    pub tools: Arc<[ToolDescriptor]>,
    pub generated_at: SystemTime,
    pub expires_at: SystemTime,
}
