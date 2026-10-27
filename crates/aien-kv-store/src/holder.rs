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

//! Holder generation allocation (ADR 0003 section 5.2).
//!
//! A chunk holder is a real block table in the cache manager under the
//! reserved sequence slot `u32::MAX`, with a nonzero 32-bit generation. The
//! first draft derived the generation as `job_id as u32`. That wraps silently
//! after 2^32 jobs, aliases job `2^32 + n` onto job `n`, and maps job `2^32`
//! onto the reserved zero. Two live holders could then share one id and one
//! would free the other's blocks.
//!
//! Two rules replace it:
//!
//! * [`HolderGenerations`] hands out a nonzero generation that no live holder
//!   uses, skipping live values when the counter wraps, and refuses
//!   (`Exhausted`) instead of reusing a live one.
//! * [`generation_for_job`] is the checked form of the old mapping for callers
//!   that must derive it from a `JobId`: it refuses any id that does not fit.

use crate::store::JobId;
use alloc::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolderError {
    /// Every generation is held by a live holder.
    Exhausted,
    /// The job id is zero or does not fit in 32 bits.
    JobIdOutOfRange,
}

/// The checked job-id mapping: `Ok` only when the plain cast would be exact
/// and nonzero.
pub fn generation_for_job(job: JobId) -> Result<u32, HolderError> {
    match u32::try_from(job.0) {
        Ok(g) if g != 0 => Ok(g),
        _ => Err(HolderError::JobIdOutOfRange),
    }
}

/// Collision-safe generation allocator. Not thread-safe by itself; it lives
/// with the cache owner that already serialises holder creation.
#[derive(Debug, Clone)]
pub struct HolderGenerations {
    next: u32,
    max: u32,
    live: BTreeSet<u32>,
}

impl Default for HolderGenerations {
    fn default() -> Self {
        Self::new()
    }
}

impl HolderGenerations {
    /// Generations `1..=u32::MAX`.
    pub fn new() -> Self {
        Self::with_max(u32::MAX)
    }

    /// Generations `1..=max` (tests use a small range to reach the wrap).
    pub fn with_max(max: u32) -> Self {
        Self::starting_at(1, max)
    }

    /// As `with_max`, with the counter starting at `first` (clamped into range).
    pub fn starting_at(first: u32, max: u32) -> Self {
        let max = max.max(1);
        Self {
            next: first.clamp(1, max),
            max,
            live: BTreeSet::new(),
        }
    }

    pub fn live(&self) -> usize {
        self.live.len()
    }

    pub fn alloc(&mut self) -> Result<u32, HolderError> {
        if self.live.len() as u64 >= self.max as u64 {
            return Err(HolderError::Exhausted);
        }
        // At least one value is free, so this loop ends within `live + 1` steps.
        loop {
            let g = self.next;
            self.next = if g >= self.max { 1 } else { g + 1 };
            if self.live.insert(g) {
                return Ok(g);
            }
        }
    }

    /// Returns false if `generation` was not live.
    pub fn release(&mut self, generation: u32) -> bool {
        self.live.remove(&generation)
    }
}
