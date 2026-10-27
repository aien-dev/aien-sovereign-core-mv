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

//! SwarmManager and Branch-Native Swarm Coordination
//! Manages first-class SwarmRecord entities and atomic branch operations.

use crate::sequence::{SequenceArena, SequenceId, SequenceState};
use crate::world::WorldStore;
use aien_kv_cache::{AienKvManager, PrefillGateError};
use aien_trace::{
    CorrelationIds, EventKind, EventStatus, EvidenceRefs, NullSink, TraceEvent, TraceId, TraceSink,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmState {
    Created,
    Running,
    Paused,
    Cancelling,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub model_handle: u64,
    pub branch_count: usize,
    pub max_active_sequences: usize,
    pub max_tokens_per_branch: usize,
    pub root_world_id: u64,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmRecord {
    pub id: u64,
    pub config: SwarmConfig,
    pub root_sequence_id: SequenceId,
    pub branch_sequences: Vec<SequenceId>,
    /// World IDs recorded at fork time. Cancel must drop these even when the
    /// arena records were already freed by natural completion.
    pub branch_worlds: Vec<u64>,
    /// Branches that finished naturally. Completion of all branches triggers
    /// swarm-wide reclamation: root arena slot, root KV, and branch worlds.
    pub finished_branches: Vec<SequenceId>,
    pub state: SwarmState,
    pub created_at: u64,
    /// Id of the `swarm_launched` trace event (the parent of this swarm's
    /// other trace events). Correlation only; `None` in old records.
    #[serde(default)]
    pub trace_root_event: Option<u64>,
}

pub struct SwarmManager {
    swarms: HashMap<u64, SwarmRecord>,
    /// Correlation trace sink. `NullSink` by default (tracing off).
    trace: Arc<dyn TraceSink>,
    /// swarm id -> trace id, derived at launch from (root world, swarm id).
    trace_ids: HashMap<u64, TraceId>,
    next_swarm_id: u64,
    /// PREFILL-GATE: swarms whose root prompt has not been prefilled yet
    /// (swarm id -> root prompt). Branch KV is forked from the root only after
    /// the root's prefill completion fence; see `fork_branches_from_ready_root`.
    pending_root_prefill: HashMap<u64, Vec<u32>>,
}

impl Default for SwarmManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmManager {
    pub fn new() -> Self {
        Self {
            swarms: HashMap::new(),
            trace: Arc::new(NullSink),
            trace_ids: HashMap::new(),
            next_swarm_id: 1,
            pending_root_prefill: HashMap::new(),
        }
    }

    /// Installs the correlation trace sink. The default is `NullSink`.
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.trace = sink;
    }

    /// Trace id of a launched swarm: `TraceId::derive(root_world_id, swarm_id, 0)`.
    pub fn trace_id(&self, swarm_id: u64) -> Option<TraceId> {
        self.trace_ids.get(&swarm_id).copied()
    }

    fn trace_event(
        &self,
        trace_id: TraceId,
        parent: Option<u64>,
        kind: EventKind,
        status: EventStatus,
        ids: CorrelationIds,
    ) -> u64 {
        if !self.trace.enabled() {
            return 0;
        }
        let event_id = self.trace.next_event_id();
        self.trace.emit(TraceEvent {
            trace_id,
            event_id,
            parent_event_id: parent,
            at_unix_ms: aien_trace::now_unix_ms(),
            kind,
            status,
            effect_certainty: None,
            ids,
            refs: EvidenceRefs::default(),
            note: None,
        });
        event_id
    }

    /// Launches a swarm: creates root sequence, preallocates root KV blocks, and forks N branches.
    pub fn launch_swarm(
        &mut self,
        config: SwarmConfig,
        arena: &mut SequenceArena,
        kv: &mut AienKvManager,
        world_store: &mut WorldStore,
        prompt_tokens: &[u32],
        timestamp: u64,
    ) -> Result<u64, String> {
        let swarm_id = self.next_swarm_id;
        self.next_swarm_id += 1;

        // Ensure a root world exists
        let root_world_id = if world_store.get_world(config.root_world_id).is_some() {
            config.root_world_id
        } else {
            world_store.create_root_world(1, 1, 1, timestamp)
        };

        // 1. Allocate root sequence in arena
        let root_seq = arena.allocate(
            root_world_id,
            1, // Initial root context revision
            config.priority,
            timestamp,
        )?;

        // 2. Allocate root prompt blocks in physical KV manager. This reserves
        // and zeroes blocks only (PrefillState::Allocated); no K/V is computed.
        let trace_id = TraceId::derive(root_world_id, swarm_id, 0);
        let root_event = self.trace_event(
            trace_id,
            None,
            EventKind::SwarmLaunched,
            EventStatus::Ok,
            CorrelationIds {
                swarm_id: Some(swarm_id),
                world_id: Some(root_world_id),
                root_sequence_id: Some(root_seq.as_u64()),
                ..Default::default()
            },
        );
        let root_u64 = root_seq.as_u64();
        kv.allocate_sequence(root_u64, prompt_tokens)?;

        if let Some(record) = arena.get_mut(root_seq) {
            // Nothing is prefilled yet: the cursor advances at the completion fence.
            record.prefill_cursor = 0;
            record.state = SequenceState::Prefill;
        }

        // 3. Fork N branch records and worlds. Branch KV is NOT forked here:
        // the root blocks are not computed, and sharing them would let branches
        // decode over zeroed K/V. `fork_branches_from_ready_root` shares the
        // root blocks after the root prefill completion fence.
        let mut branch_sequences = Vec::with_capacity(config.branch_count);
        let mut branch_worlds = Vec::with_capacity(config.branch_count);
        for _ in 0..config.branch_count {
            let child_world = world_store.fork_world(root_world_id, timestamp)?;
            let child_seq = arena.fork(root_seq, child_world, timestamp)?;
            if let Some(child) = arena.get_mut(child_seq) {
                // Waiting on the root prefill, not decoding.
                child.state = SequenceState::Prefill;
            }
            self.trace_event(
                trace_id,
                Some(root_event),
                EventKind::BranchForked,
                EventStatus::Ok,
                CorrelationIds {
                    swarm_id: Some(swarm_id),
                    world_id: Some(child_world),
                    root_sequence_id: Some(root_seq.as_u64()),
                    sequence_id: Some(child_seq.as_u64()),
                    ..Default::default()
                },
            );
            branch_sequences.push(child_seq);
            branch_worlds.push(child_world);
        }
        self.pending_root_prefill
            .insert(swarm_id, prompt_tokens.to_vec());

        let record = SwarmRecord {
            id: swarm_id,
            config,
            root_sequence_id: root_seq,
            branch_sequences,
            branch_worlds,
            finished_branches: Vec::new(),
            state: SwarmState::Running,
            created_at: timestamp,
            trace_root_event: Some(root_event),
        };
        self.trace_ids.insert(swarm_id, trace_id);

        self.swarms.insert(swarm_id, record);
        Ok(swarm_id)
    }

    pub fn get_swarm(&self, swarm_id: u64) -> Option<&SwarmRecord> {
        self.swarms.get(&swarm_id)
    }

    /// Swarms whose root prompt still needs a model prefill:
    /// (swarm id, root KV sequence id, root prompt).
    pub fn pending_root_prefills(&self) -> Vec<(u64, u64, Vec<u32>)> {
        let mut pending: Vec<(u64, u64, Vec<u32>)> = self
            .pending_root_prefill
            .iter()
            .filter_map(|(&swarm_id, prompt)| {
                self.swarms
                    .get(&swarm_id)
                    .filter(|s| s.state == SwarmState::Running)
                    .map(|s| (swarm_id, s.root_sequence_id.as_u64(), prompt.clone()))
            })
            .collect();
        pending.sort_by_key(|p| p.0);
        pending
    }

    /// True while the swarm's root prompt has not passed its prefill fence.
    pub fn is_root_prefill_pending(&self, swarm_id: u64) -> bool {
        self.pending_root_prefill.contains_key(&swarm_id)
    }

    /// Shares the root KV blocks with every branch once the root prefill
    /// completion fence has fired. Refuses (typed `PrefillGateError`, as a
    /// String) while the root is not PrefillReady; nothing is forked then.
    pub fn fork_branches_from_ready_root(
        &mut self,
        swarm_id: u64,
        arena: &mut SequenceArena,
        kv: &mut AienKvManager,
    ) -> Result<(), String> {
        let swarm = self
            .swarms
            .get(&swarm_id)
            .ok_or_else(|| format!("Swarm {} not found", swarm_id))?;
        let root_u64 = swarm.root_sequence_id.as_u64();

        // Check readiness once up front so a refusal forks nothing.
        match kv.prefill_state(root_u64) {
            Some(state) if state.is_ready() => {}
            Some(state) => {
                return Err(PrefillGateError::NotReady {
                    seq_id: root_u64,
                    state,
                }
                .to_string())
            }
            None => return Err(PrefillGateError::UnknownSequence { seq_id: root_u64 }.to_string()),
        }

        let prompt_len = kv
            .get_block_table(root_u64)
            .map(|t| t.total_tokens)
            .unwrap_or(0);
        for &child in &swarm.branch_sequences {
            kv.fork_prefilled(root_u64, child.as_u64())
                .map_err(|e| e.to_string())?;
            if let Some(rec) = arena.get_mut(child) {
                rec.prefill_cursor = prompt_len;
                rec.state = SequenceState::Decode;
            }
        }
        if let Some(root) = arena.get_mut(swarm.root_sequence_id) {
            root.prefill_cursor = prompt_len;
        }
        self.pending_root_prefill.remove(&swarm_id);
        Ok(())
    }

    /// Cancels a swarm and reclaims its resources: KV sequences, arena slots,
    /// and branch worlds. The root world is retained; branch worlds are dropped.
    /// Returns the sequence ids (root first, then every branch) whose backend
    /// per-sequence state the caller must release (PREFILL-E2E C5); the
    /// swarm manager has no backend handle.
    pub fn cancel_swarm(
        &mut self,
        swarm_id: u64,
        arena: &mut SequenceArena,
        kv: &mut AienKvManager,
        worlds: &mut WorldStore,
    ) -> Result<Vec<u64>, String> {
        let swarm = self
            .swarms
            .get_mut(&swarm_id)
            .ok_or_else(|| format!("Swarm {} not found", swarm_id))?;
        self.pending_root_prefill.remove(&swarm_id);

        swarm.state = SwarmState::Cancelling;

        if let Some(root_rec) = arena.get_mut(swarm.root_sequence_id) {
            root_rec.state = SequenceState::Cancelled;
        }
        let _ = kv.free_sequence(swarm.root_sequence_id.as_u64());

        for &child_id in &swarm.branch_sequences {
            if let Some(child_rec) = arena.get_mut(child_id) {
                child_rec.state = SequenceState::Cancelled;
            }
            let _ = kv.free_sequence(child_id.as_u64());
        }

        for &child_id in &swarm.branch_sequences {
            arena.free(child_id);
        }
        arena.free(swarm.root_sequence_id);

        // Drop from the launch-time ledger, not from arena lookups: branch
        // sequences may have completed naturally and already left the arena.
        let branch_worlds = swarm.branch_worlds.clone();
        for world_id in branch_worlds {
            worlds.drop_world(world_id);
        }
        worlds.collect_garbage();

        swarm.state = SwarmState::Completed;
        let mut released = Vec::with_capacity(swarm.branch_sequences.len() + 1);
        released.push(swarm.root_sequence_id.as_u64());
        released.extend(swarm.branch_sequences.iter().map(|b| b.as_u64()));
        let root_event = swarm.trace_root_event;
        if let Some(trace_id) = self.trace_ids.get(&swarm_id).copied() {
            self.trace_event(
                trace_id,
                root_event,
                EventKind::SwarmCancelled,
                EventStatus::Cancelled,
                CorrelationIds {
                    swarm_id: Some(swarm_id),
                    ..Default::default()
                },
            );
        }
        Ok(released)
    }

    /// Records a branch that finished naturally. Once every branch of a
    /// running swarm has finished, reclaims the whole swarm: root arena slot,
    /// root KV blocks, and all branch worlds. Without this, the root sequence
    /// pins the arena forever and finished branch worlds leak.
    /// Returns the root sequence id when this call reclaimed the swarm, so the
    /// caller can release the root's backend state (PREFILL-E2E C5).
    pub fn note_sequence_finished(
        &mut self,
        seq_id: SequenceId,
        arena: &mut SequenceArena,
        kv: &mut AienKvManager,
        worlds: &mut WorldStore,
    ) -> Option<SequenceId> {
        let swarm_id = self.swarms.iter().find_map(|(id, swarm)| {
            (swarm.state == SwarmState::Running
                && swarm.branch_sequences.contains(&seq_id)
                && !swarm.finished_branches.contains(&seq_id))
            .then_some(*id)
        });
        let swarm_id = swarm_id?;
        if let Some(trace_id) = self.trace_ids.get(&swarm_id).copied() {
            let root_event = self.swarms.get(&swarm_id).and_then(|s| s.trace_root_event);
            self.trace_event(
                trace_id,
                root_event,
                EventKind::SequenceFinished,
                EventStatus::Ok,
                CorrelationIds {
                    swarm_id: Some(swarm_id),
                    sequence_id: Some(seq_id.as_u64()),
                    ..Default::default()
                },
            );
        }

        let swarm = self
            .swarms
            .get_mut(&swarm_id)
            .expect("swarm id from live iterator must exist");
        swarm.finished_branches.push(seq_id);
        if swarm.finished_branches.len() < swarm.branch_sequences.len() {
            return None;
        }

        if let Some(root_rec) = arena.get_mut(swarm.root_sequence_id) {
            root_rec.state = SequenceState::Completed;
        }
        let _ = kv.free_sequence(swarm.root_sequence_id.as_u64());
        arena.free(swarm.root_sequence_id);
        self.pending_root_prefill.remove(&swarm_id);

        for world_id in swarm.branch_worlds.clone() {
            worlds.drop_world(world_id);
        }
        worlds.collect_garbage();
        swarm.state = SwarmState::Completed;
        Some(swarm.root_sequence_id)
    }

    pub fn active_swarm_count(&self) -> usize {
        self.swarms
            .values()
            .filter(|s| s.state == SwarmState::Running)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::WorldStore;

    fn test_config(branch_count: usize) -> SwarmConfig {
        SwarmConfig {
            model_handle: 1,
            branch_count,
            max_active_sequences: branch_count,
            max_tokens_per_branch: 8,
            root_world_id: 0,
            priority: 1,
        }
    }

    #[test]
    fn cancel_drops_branch_worlds_after_natural_completion() {
        let mut arena = SequenceArena::new(256);
        let kv_shared = aien_kv_cache::create_shared_kv_manager(256, 16);
        let mut kv = kv_shared.write();
        let mut worlds = WorldStore::new();
        let mut manager = SwarmManager::new();

        let prompt = vec![1u32, 2, 3];
        let swarm_id = manager
            .launch_swarm(test_config(4), &mut arena, &mut kv, &mut worlds, &prompt, 1)
            .expect("swarm launch must succeed");
        assert_eq!(worlds.active_world_count(), 5, "root plus 4 branch worlds");

        // Branches finish naturally: arena records leave the arena, worlds stay.
        let record = manager.get_swarm(swarm_id).expect("record").clone();
        for &child in &record.branch_sequences {
            arena.free(child);
        }
        arena.free(record.root_sequence_id);

        manager
            .cancel_swarm(swarm_id, &mut arena, &mut kv, &mut worlds)
            .expect("cancel must succeed");

        assert_eq!(
            worlds.active_world_count(),
            1,
            "branch worlds must drop even after arena records were freed"
        );
    }

    #[test]
    fn natural_completion_reclaims_root_and_branch_worlds() {
        let mut arena = SequenceArena::new(256);
        let kv_shared = aien_kv_cache::create_shared_kv_manager(256, 16);
        let mut kv = kv_shared.write();
        let mut worlds = WorldStore::new();
        let mut manager = SwarmManager::new();

        let prompt = vec![7u32, 8, 9];
        let swarm_id = manager
            .launch_swarm(test_config(3), &mut arena, &mut kv, &mut worlds, &prompt, 1)
            .expect("swarm launch must succeed");
        assert_eq!(worlds.active_world_count(), 4, "root plus 3 branch worlds");

        // Every branch finishes through the scheduler; the spine reports each
        // finish to the swarm manager.
        let record = manager.get_swarm(swarm_id).expect("record").clone();
        for &child in &record.branch_sequences {
            manager.note_sequence_finished(child, &mut arena, &mut kv, &mut worlds);
        }

        assert_eq!(manager.active_swarm_count(), 0, "swarm must complete");
        assert_eq!(
            worlds.active_world_count(),
            1,
            "only the root world remains"
        );
        assert!(
            arena.get(record.root_sequence_id).is_none(),
            "root arena slot must be reclaimed on natural completion"
        );
    }

    #[test]
    fn prefill_gate_branches_do_not_share_uncomputed_root_blocks() {
        let mut arena = SequenceArena::new(256);
        let kv_shared = aien_kv_cache::create_shared_kv_manager(256, 16);
        let mut kv = kv_shared.write();
        let mut worlds = WorldStore::new();
        let mut manager = SwarmManager::new();

        let prompt = vec![4u32, 5, 6, 7];
        let swarm_id = manager
            .launch_swarm(test_config(3), &mut arena, &mut kv, &mut worlds, &prompt, 1)
            .expect("swarm launch must succeed");
        let record = manager.get_swarm(swarm_id).expect("record").clone();
        let root = record.root_sequence_id.as_u64();

        assert!(manager.is_root_prefill_pending(swarm_id));
        assert_eq!(
            kv.prefill_state(root),
            Some(aien_kv_cache::PrefillState::Allocated),
            "launch allocates root blocks; it computes nothing"
        );
        for child in &record.branch_sequences {
            assert!(
                kv.get_block_table(child.as_u64()).is_none(),
                "no branch may hold root blocks before the root prefill fence"
            );
        }

        // Typed refusal while the root is not PrefillReady; nothing is forked.
        let err = manager
            .fork_branches_from_ready_root(swarm_id, &mut arena, &mut kv)
            .unwrap_err();
        assert!(
            err.contains("not PrefillReady"),
            "unexpected error: {}",
            err
        );
        for child in &record.branch_sequences {
            assert!(kv.get_block_table(child.as_u64()).is_none());
        }
        assert!(manager.is_root_prefill_pending(swarm_id));

        // After the completion fence, branches share the frozen root blocks.
        kv.mark_prefill_pending(root).unwrap();
        kv.complete_prefill(root).unwrap();
        manager
            .fork_branches_from_ready_root(swarm_id, &mut arena, &mut kv)
            .expect("fork after fence");
        assert!(!manager.is_root_prefill_pending(swarm_id));
        let root_blocks = kv.get_block_table(root).unwrap().block_ids.clone();
        for child in &record.branch_sequences {
            let table = kv.get_block_table(child.as_u64()).expect("branch table");
            assert_eq!(table.block_ids, root_blocks);
            assert!(table.is_prefill_ready());
        }
    }
}
