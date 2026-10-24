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

//! SequenceArena and Generational Sequence Allocation
//! Enforces single-mutable-owner discipline for the scheduler thread.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceId {
    pub slot: u32,
    pub generation: u32,
}

impl SequenceId {
    pub fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    #[inline]
    pub fn as_u64(&self) -> u64 {
        ((self.slot as u64) << 32) | (self.generation as u64)
    }

    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self {
            slot: (val >> 32) as u32,
            generation: (val & 0xFFFF_FFFF) as u32,
        }
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq_{}:{}", self.slot, self.generation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceState {
    Ready,
    Prefill,
    Decode,
    BlockedOnTool,
    Cancelled,
    Completed,
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRecord {
    pub id: SequenceId,
    pub parent_id: Option<SequenceId>,
    pub world_id: u64,
    pub priority: u8,
    pub state: SequenceState,
    pub context_id: u64,
    pub block_table_id: u32,
    pub prefill_cursor: usize,
    pub generated_tokens: usize,
    pub arrival_ticks: u64,
    pub deadline_ticks: Option<u64>,
}

pub struct SequenceSlot {
    pub generation: u32,
    /// Set when the generation reached u32::MAX and was freed. A retired slot is never reused,
    /// so a stale id can never match again (same policy as the scheduler arena).
    pub retired: bool,
    pub record: Option<SequenceRecord>,
}

/// Contiguous SequenceArena owned exclusively by the Scheduler thread.
pub struct SequenceArena {
    slots: Vec<SequenceSlot>,
    free_slots: Vec<u32>,
    active_count: usize,
    capacity: usize,
    retired_slots_count: usize,
}

impl SequenceArena {
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_slots = Vec::with_capacity(capacity);

        for i in 0..capacity {
            slots.push(SequenceSlot {
                generation: 1,
                retired: false,
                record: None,
            });
            free_slots.push(i as u32);
        }
        free_slots.reverse();

        Self {
            slots,
            free_slots,
            active_count: 0,
            capacity,
            retired_slots_count: 0,
        }
    }

    /// Test-only: start every slot at `generation` so the u32::MAX boundary is reachable.
    #[cfg(test)]
    pub(crate) fn with_start_generation(capacity: usize, generation: u32) -> Self {
        let mut arena = Self::new(capacity);
        for slot in &mut arena.slots {
            slot.generation = generation;
        }
        arena
    }

    /// Number of slots permanently retired after exhausting their generation space.
    pub fn retired_slots(&self) -> usize {
        self.retired_slots_count
    }

    pub fn allocate(
        &mut self,
        world_id: u64,
        context_id: u64,
        priority: u8,
        arrival_ticks: u64,
    ) -> Result<SequenceId, String> {
        let slot_idx = self
            .free_slots
            .pop()
            .ok_or_else(|| format!("SequenceArena capacity exhausted ({})", self.capacity))?;

        let slot = &mut self.slots[slot_idx as usize];
        let id = SequenceId::new(slot_idx, slot.generation);

        slot.record = Some(SequenceRecord {
            id,
            parent_id: None,
            world_id,
            priority,
            state: SequenceState::Ready,
            context_id,
            block_table_id: slot_idx,
            prefill_cursor: 0,
            generated_tokens: 0,
            arrival_ticks,
            deadline_ticks: None,
        });

        self.active_count += 1;
        Ok(id)
    }

    pub fn fork(
        &mut self,
        parent_id: SequenceId,
        child_world_id: u64,
        arrival_ticks: u64,
    ) -> Result<SequenceId, String> {
        let (parent_context_id, parent_priority, parent_prefill, parent_generated) = {
            let parent = self
                .get(parent_id)
                .ok_or_else(|| format!("Parent sequence {} not found or stale", parent_id))?;
            (
                parent.context_id,
                parent.priority,
                parent.prefill_cursor,
                parent.generated_tokens,
            )
        };

        let child_slot = self.free_slots.pop().ok_or_else(|| {
            format!(
                "SequenceArena capacity exhausted on fork ({})",
                self.capacity
            )
        })?;

        let slot = &mut self.slots[child_slot as usize];
        let child_id = SequenceId::new(child_slot, slot.generation);

        slot.record = Some(SequenceRecord {
            id: child_id,
            parent_id: Some(parent_id),
            world_id: child_world_id,
            priority: parent_priority,
            state: SequenceState::Decode,
            context_id: parent_context_id,
            block_table_id: child_slot,
            prefill_cursor: parent_prefill,
            generated_tokens: parent_generated,
            arrival_ticks,
            deadline_ticks: None,
        });

        self.active_count += 1;
        Ok(child_id)
    }

    #[inline]
    pub fn get(&self, id: SequenceId) -> Option<&SequenceRecord> {
        let slot = self.slots.get(id.slot as usize)?;
        if slot.generation == id.generation {
            slot.record.as_ref()
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceRecord> {
        let slot = self.slots.get_mut(id.slot as usize)?;
        if slot.generation == id.generation {
            slot.record.as_mut()
        } else {
            None
        }
    }

    pub fn free(&mut self, id: SequenceId) -> bool {
        if let Some(slot) = self.slots.get_mut(id.slot as usize) {
            if slot.generation == id.generation && slot.record.is_some() {
                slot.record = None;
                // Advance the generation to invalidate in-flight references (ABA protection).
                // At u32::MAX the slot is retired instead of wrapping, so no id can ever repeat.
                if slot.generation == u32::MAX {
                    slot.retired = true;
                    self.retired_slots_count += 1;
                } else {
                    slot.generation += 1;
                    self.free_slots.push(id.slot);
                }
                self.active_count = self.active_count.saturating_sub(1);
                return true;
            }
        }
        false
    }

    pub fn validate(&self, id: SequenceId) -> bool {
        self.get(id).is_some()
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn iter_active(&self) -> impl Iterator<Item = &SequenceRecord> {
        self.slots.iter().filter_map(|s| s.record.as_ref())
    }

    pub fn iter_active_mut(&mut self) -> impl Iterator<Item = &mut SequenceRecord> {
        self.slots.iter_mut().filter_map(|s| s.record.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_below_max_advances_generation_and_reuses_slot() {
        let mut arena = SequenceArena::with_start_generation(1, u32::MAX - 1);
        let id = arena.allocate(1, 1, 0, 0).unwrap();
        assert_eq!(id.generation, u32::MAX - 1);
        assert!(arena.free(id));
        assert_eq!(arena.retired_slots(), 0);
        let next = arena.allocate(1, 1, 0, 0).unwrap();
        assert_eq!(next.slot, id.slot);
        assert_eq!(next.generation, u32::MAX);
        assert!(!arena.validate(id));
    }

    #[test]
    fn free_at_max_retires_slot_and_never_resurrects_stale_id() {
        let mut arena = SequenceArena::with_start_generation(2, u32::MAX);
        let id = arena.allocate(1, 1, 0, 0).unwrap();
        assert_eq!(id.generation, u32::MAX);
        assert!(arena.free(id));
        assert_eq!(arena.retired_slots(), 1);
        assert!(!arena.validate(id));
        assert!(!arena.free(id));

        // Retired slot is not handed out again; the other slot still works.
        let other = arena.allocate(1, 1, 0, 0).unwrap();
        assert_ne!(other.slot, id.slot);
        assert!(arena.validate(other));
        assert!(!arena.validate(id));
    }

    #[test]
    fn retired_only_slot_exhausts_capacity() {
        let mut arena = SequenceArena::with_start_generation(1, u32::MAX);
        let id = arena.allocate(1, 1, 0, 0).unwrap();
        assert!(arena.free(id));
        assert!(arena.allocate(1, 1, 0, 0).is_err());
        assert_eq!(arena.active_count(), 0);
    }
}
