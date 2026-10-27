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

//! Resource permits: CPU slots, memory bytes and a concurrency count.
//!
//! A task acquires all three at dispatch or none of them. Permits are held
//! exactly while the task is `Provisional` (see `execution.rs`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimits {
    pub cpu_slots: u32,
    pub memory_bytes: u64,
    pub max_concurrency: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        ResourceLimits {
            cpu_slots: 4,
            memory_bytes: 8 << 30,
            max_concurrency: 4,
        }
    }
}

/// What one task needs while it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequest {
    pub cpu_slots: u32,
    pub memory_bytes: u64,
}

impl ResourceLimits {
    /// Could this request ever run, with the whole budget free?
    pub fn admits(&self, req: ResourceRequest) -> bool {
        self.max_concurrency >= 1
            && req.cpu_slots <= self.cpu_slots
            && req.memory_bytes <= self.memory_bytes
    }
}

/// Permits currently handed out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PermitPool {
    pub cpu_in_use: u32,
    pub memory_in_use: u64,
    pub running: u32,
}

impl PermitPool {
    /// Take all permits for `req`, or none.
    pub fn try_acquire(&mut self, limits: &ResourceLimits, req: ResourceRequest) -> bool {
        let fits = self.running < limits.max_concurrency
            && self.cpu_in_use.saturating_add(req.cpu_slots) <= limits.cpu_slots
            && self.memory_in_use.saturating_add(req.memory_bytes) <= limits.memory_bytes;
        if fits {
            self.running += 1;
            self.cpu_in_use += req.cpu_slots;
            self.memory_in_use += req.memory_bytes;
        }
        fits
    }

    pub fn release(&mut self, req: ResourceRequest) {
        self.running = self.running.saturating_sub(1);
        self.cpu_in_use = self.cpu_in_use.saturating_sub(req.cpu_slots);
        self.memory_in_use = self.memory_in_use.saturating_sub(req.memory_bytes);
    }
}
