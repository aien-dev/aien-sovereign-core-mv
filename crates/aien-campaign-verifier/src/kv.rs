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

//! KV conservation: rebuild allocator state from the mutation journal and compare it with the
//! raw census and with AIEN's reported counters (SPEC.md Section 7).
//!
//! Gate 4 adds prefix-lease events; lease references then count toward `refcount(block)`.

use crate::ids::SequenceId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type BlockId = u32;

/// One line of `kv_allocations.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum KvEvent {
    SequenceAllocated {
        seq: SequenceId,
        blocks: Vec<BlockId>,
    },
    Forked {
        parent: SequenceId,
        child: SequenceId,
    },
    BlockAppended {
        seq: SequenceId,
        block: BlockId,
    },
    CowFault {
        seq: SequenceId,
        index: usize,
        old_block: BlockId,
        new_block: BlockId,
    },
    /// Logged at dispatch with the resolved target block (SPEC.md invariant I2).
    BlockWrite {
        seq: SequenceId,
        block: BlockId,
    },
    Released {
        seq: SequenceId,
    },
}

/// Allocator state as reconstructed by the verifier. Blocks absent from `refcounts` are free.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvState {
    pub pool_total: u32,
    pub refcounts: BTreeMap<BlockId, u32>,
    pub tables: BTreeMap<SequenceId, Vec<BlockId>>,
    pub cow_faults: u64,
}

impl KvState {
    pub fn new(pool_total: u32) -> Self {
        Self {
            pool_total,
            ..Default::default()
        }
    }

    fn take_free(&mut self, block: BlockId, at: usize, violations: &mut Vec<String>) -> bool {
        if block >= self.pool_total {
            violations.push(format!(
                "event {at}: block {block} outside pool of {}",
                self.pool_total
            ));
            return false;
        }
        if self.refcounts.contains_key(&block) {
            violations.push(format!(
                "event {at}: block {block} allocated while still referenced"
            ));
            return false;
        }
        self.refcounts.insert(block, 1);
        true
    }

    fn decref(&mut self, block: BlockId, at: usize, violations: &mut Vec<String>) {
        let Some(rc) = self.refcounts.get(&block).copied() else {
            violations.push(format!("event {at}: refcount underflow on block {block}"));
            return;
        };
        if rc > 1 {
            self.refcounts.insert(block, rc - 1);
        } else {
            self.refcounts.remove(&block);
        }
    }
}

/// Replays the journal. Invalid events are reported and not applied, so one defect does not
/// cascade into unrelated violations.
pub fn replay(pool_total: u32, events: &[KvEvent]) -> (KvState, Vec<String>) {
    let mut s = KvState::new(pool_total);
    let mut v = Vec::new();

    for (at, event) in events.iter().enumerate() {
        match event {
            KvEvent::SequenceAllocated { seq, blocks } => {
                if s.tables.contains_key(seq) {
                    v.push(format!("event {at}: sequence {seq} allocated while live"));
                    continue;
                }
                let mut table = Vec::with_capacity(blocks.len());
                for &b in blocks {
                    if s.take_free(b, at, &mut v) {
                        table.push(b);
                    }
                }
                s.tables.insert(*seq, table);
            }
            KvEvent::Forked { parent, child } => {
                if s.tables.contains_key(child) {
                    v.push(format!("event {at}: fork into existing sequence {child}"));
                    continue;
                }
                let Some(table) = s.tables.get(parent).cloned() else {
                    v.push(format!("event {at}: fork from unknown parent {parent}"));
                    continue;
                };
                for &b in &table {
                    *s.refcounts.entry(b).or_insert(0) += 1;
                }
                s.tables.insert(*child, table);
            }
            KvEvent::BlockAppended { seq, block } => {
                if !s.tables.contains_key(seq) {
                    v.push(format!("event {at}: append to unknown sequence {seq}"));
                    continue;
                }
                if s.take_free(*block, at, &mut v) {
                    if let Some(t) = s.tables.get_mut(seq) {
                        t.push(*block);
                    }
                }
            }
            KvEvent::CowFault {
                seq,
                index,
                old_block,
                new_block,
            } => {
                let current = s.tables.get(seq).and_then(|t| t.get(*index)).copied();
                if current != Some(*old_block) {
                    v.push(format!(
                        "event {at}: COW on {seq} index {index} expected block {old_block}, table has {current:?}"
                    ));
                    continue;
                }
                if s.refcounts.get(old_block).copied().unwrap_or(0) < 2 {
                    v.push(format!(
                        "event {at}: COW copy of private block {old_block} by {seq}"
                    ));
                }
                if !s.take_free(*new_block, at, &mut v) {
                    continue;
                }
                s.decref(*old_block, at, &mut v);
                if let Some(t) = s.tables.get_mut(seq) {
                    t[*index] = *new_block;
                }
                s.cow_faults += 1;
            }
            KvEvent::BlockWrite { seq, block } => {
                let owns = s.tables.get(seq).is_some_and(|t| t.contains(block));
                if !owns {
                    v.push(format!(
                        "event {at}: {seq} wrote block {block} it does not reference"
                    ));
                    continue;
                }
                let rc = s.refcounts.get(block).copied().unwrap_or(0);
                if rc > 1 {
                    v.push(format!(
                        "event {at}: {seq} wrote shared block {block} (refcount {rc}) without COW"
                    ));
                }
            }
            KvEvent::Released { seq } => {
                let Some(table) = s.tables.remove(seq) else {
                    v.push(format!(
                        "event {at}: release of unknown or already released sequence {seq}"
                    ));
                    continue;
                };
                for b in table {
                    s.decref(b, at, &mut v);
                }
            }
        }
    }

    (s, v)
}

#[derive(Debug, Clone, Deserialize)]
pub struct CensusBlock {
    pub block_id: BlockId,
    pub ref_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CensusSequence {
    pub seq: SequenceId,
    pub block_ids: Vec<BlockId>,
}

/// `kv_census.json`: raw allocator state dumped at the end of a run.
#[derive(Debug, Clone, Deserialize)]
pub struct KvCensus {
    pub pool_total: u32,
    pub free_blocks: Vec<BlockId>,
    pub blocks: Vec<CensusBlock>,
    pub sequences: Vec<CensusSequence>,
}

/// `kv_metrics.json`: AIEN's own `KvMetrics`. Unknown fields are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct ReportedKvMetrics {
    pub physical_pages: usize,
    pub logical_pages: usize,
    pub shared_pages: usize,
    pub private_pages: usize,
    pub cow_faults: usize,
    pub used_blocks: usize,
}

/// Conservation equations evaluated on the census alone.
pub fn check_census_conservation(c: &KvCensus) -> Vec<String> {
    let mut v = Vec::new();

    let mut free = BTreeSet::new();
    for &b in &c.free_blocks {
        if b >= c.pool_total {
            v.push(format!("free block {b} outside pool of {}", c.pool_total));
        }
        if !free.insert(b) {
            v.push(format!("block {b} appears twice in the free set"));
        }
    }

    let mut recorded = BTreeMap::new();
    for blk in c.blocks.iter().filter(|b| b.ref_count > 0) {
        if recorded.insert(blk.block_id, blk.ref_count).is_some() {
            v.push(format!("block {} has two census entries", blk.block_id));
        }
    }

    let mut live = BTreeSet::new();
    let mut referenced: BTreeMap<BlockId, u32> = BTreeMap::new();
    for s in &c.sequences {
        if !live.insert(s.seq) {
            v.push(format!("duplicate live sequence {}", s.seq));
        }
        for &b in &s.block_ids {
            *referenced.entry(b).or_insert(0) += 1;
        }
    }

    if recorded.len() + free.len() != c.pool_total as usize {
        v.push(format!(
            "allocated {} + free {} != pool_total {}",
            recorded.len(),
            free.len(),
            c.pool_total
        ));
    }
    for b in referenced.keys() {
        if free.contains(b) {
            v.push(format!("referenced block {b} is in the free set"));
        } else if !recorded.contains_key(b) {
            v.push(format!("referenced block {b} has no refcount"));
        }
    }
    for (b, rc) in &recorded {
        let refs = referenced.get(b).copied().unwrap_or(0);
        if refs == 0 {
            v.push(format!("unreferenced block {b} is outside the free set"));
        } else if refs != *rc {
            v.push(format!(
                "block {b}: refcount {rc} but {refs} table references"
            ));
        }
    }

    v
}

/// Journal replay must equal the raw census.
pub fn check_replay_matches_census(r: &KvState, c: &KvCensus) -> Vec<String> {
    let mut v = Vec::new();

    if r.pool_total != c.pool_total {
        v.push(format!(
            "pool_total: contract {} vs census {}",
            r.pool_total, c.pool_total
        ));
    }

    let census_tables: BTreeMap<SequenceId, &Vec<BlockId>> =
        c.sequences.iter().map(|s| (s.seq, &s.block_ids)).collect();
    for (seq, table) in &r.tables {
        match census_tables.get(seq) {
            None => v.push(format!(
                "sequence {seq} live in journal replay, absent from census"
            )),
            Some(t) if *t != table => v.push(format!(
                "sequence {seq} block table differs: replay {table:?}, census {t:?}"
            )),
            Some(_) => {}
        }
    }
    for seq in census_tables.keys() {
        if !r.tables.contains_key(seq) {
            v.push(format!(
                "sequence {seq} live in census, absent from journal replay"
            ));
        }
    }

    let census_refs: BTreeMap<BlockId, u32> = c
        .blocks
        .iter()
        .filter(|b| b.ref_count > 0)
        .map(|b| (b.block_id, b.ref_count))
        .collect();
    let keys: BTreeSet<BlockId> = census_refs
        .keys()
        .chain(r.refcounts.keys())
        .copied()
        .collect();
    for b in keys {
        let replayed = r.refcounts.get(&b).copied().unwrap_or(0);
        let census = census_refs.get(&b).copied().unwrap_or(0);
        if replayed != census {
            v.push(format!(
                "block {b}: replay refcount {replayed}, census refcount {census}"
            ));
        }
    }

    let census_free: BTreeSet<BlockId> = c.free_blocks.iter().copied().collect();
    let replay_free: BTreeSet<BlockId> = (0..r.pool_total)
        .filter(|b| !r.refcounts.contains_key(b))
        .collect();
    for b in replay_free.difference(&census_free) {
        v.push(format!("block {b} free in replay, not free in census"));
    }
    for b in census_free.difference(&replay_free) {
        v.push(format!("block {b} free in census, not free in replay"));
    }

    v
}

/// Journal replay must equal AIEN's reported counters.
pub fn check_replay_matches_metrics(r: &KvState, m: &ReportedKvMetrics) -> Vec<String> {
    let physical = r.refcounts.len();
    let logical: usize = r.tables.values().map(Vec::len).sum();
    let shared = r.refcounts.values().filter(|&&rc| rc > 1).count();
    let private = r.refcounts.values().filter(|&&rc| rc == 1).count();

    let pairs = [
        ("physical_pages", physical, m.physical_pages),
        ("used_blocks", physical, m.used_blocks),
        ("logical_pages", logical, m.logical_pages),
        ("shared_pages", shared, m.shared_pages),
        ("private_pages", private, m.private_pages),
        ("cow_faults", r.cow_faults as usize, m.cow_faults),
    ];
    pairs
        .iter()
        .filter(|(_, replayed, reported)| replayed != reported)
        .map(|(name, replayed, reported)| format!("{name}: replay {replayed}, reported {reported}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(slot: u32) -> SequenceId {
        SequenceId {
            boot_epoch: 1,
            slot,
            generation: 1,
        }
    }

    fn census_of(s: &KvState) -> KvCensus {
        KvCensus {
            pool_total: s.pool_total,
            free_blocks: (0..s.pool_total)
                .filter(|b| !s.refcounts.contains_key(b))
                .collect(),
            blocks: s
                .refcounts
                .iter()
                .map(|(&block_id, &ref_count)| CensusBlock {
                    block_id,
                    ref_count,
                })
                .collect(),
            sequences: s
                .tables
                .iter()
                .map(|(&seq, t)| CensusSequence {
                    seq,
                    block_ids: t.clone(),
                })
                .collect(),
        }
    }

    #[test]
    fn fork_diverge_release_conserves() {
        let (p, a, b) = (sid(0), sid(1), sid(2));
        let events = vec![
            KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0, 1],
            },
            KvEvent::Forked {
                parent: p,
                child: a,
            },
            KvEvent::Forked {
                parent: p,
                child: b,
            },
            KvEvent::CowFault {
                seq: a,
                index: 1,
                old_block: 1,
                new_block: 2,
            },
            KvEvent::BlockWrite { seq: a, block: 2 },
        ];
        let (state, violations) = replay(8, &events);
        assert!(violations.is_empty(), "{violations:?}");
        assert_eq!(state.refcounts.get(&0), Some(&3));
        assert_eq!(state.refcounts.get(&1), Some(&2));
        assert_eq!(state.refcounts.get(&2), Some(&1));

        let census = census_of(&state);
        assert!(check_census_conservation(&census).is_empty());
        assert!(check_replay_matches_census(&state, &census).is_empty());

        let mut events = events;
        events.extend([
            KvEvent::Released { seq: a },
            KvEvent::Released { seq: b },
            KvEvent::Released { seq: p },
        ]);
        let (state, violations) = replay(8, &events);
        assert!(violations.is_empty(), "{violations:?}");
        assert!(state.refcounts.is_empty());
        assert!(state.tables.is_empty());
    }

    #[test]
    fn duplicate_fork_is_a_violation_and_changes_nothing() {
        let (p, a) = (sid(0), sid(1));
        let fork = KvEvent::Forked {
            parent: p,
            child: a,
        };
        let events = vec![
            KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0],
            },
            fork.clone(),
            fork,
        ];
        let (state, violations) = replay(4, &events);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("fork into existing sequence"));
        assert_eq!(state.refcounts.get(&0), Some(&2));
    }

    #[test]
    fn double_release_is_a_violation_without_underflow() {
        let p = sid(0);
        let events = vec![
            KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0],
            },
            KvEvent::Released { seq: p },
            KvEvent::Released { seq: p },
        ];
        let (state, violations) = replay(4, &events);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("already released"));
        assert!(state.refcounts.is_empty());
    }

    #[test]
    fn write_to_shared_block_is_an_isolation_violation() {
        let (p, a) = (sid(0), sid(1));
        let events = vec![
            KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0],
            },
            KvEvent::Forked {
                parent: p,
                child: a,
            },
            KvEvent::BlockWrite { seq: a, block: 0 },
        ];
        let (_, violations) = replay(4, &events);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("without COW"));
    }

    #[test]
    fn cow_of_private_block_is_flagged() {
        let p = sid(0);
        let events = vec![
            KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0],
            },
            KvEvent::CowFault {
                seq: p,
                index: 0,
                old_block: 0,
                new_block: 1,
            },
        ];
        let (_, violations) = replay(4, &events);
        assert!(violations.iter().any(|v| v.contains("private block")));
    }

    #[test]
    fn journal_omitting_a_release_disagrees_with_census() {
        let p = sid(0);
        let journal = vec![KvEvent::SequenceAllocated {
            seq: p,
            blocks: vec![0, 1],
        }];
        let (state, _) = replay(4, &journal);
        // The allocator actually released p, but the journal never logged it.
        let census = census_of(&KvState::new(4));
        let violations = check_replay_matches_census(&state, &census);
        assert!(violations.iter().any(|v| v.contains("absent from census")));
    }

    #[test]
    fn referenced_block_in_free_set_breaks_conservation() {
        let p = sid(0);
        let (state, _) = replay(
            4,
            &[KvEvent::SequenceAllocated {
                seq: p,
                blocks: vec![0],
            }],
        );
        let mut census = census_of(&state);
        census.free_blocks.push(0);
        let violations = check_census_conservation(&census);
        assert!(violations
            .iter()
            .any(|v| v.contains("referenced block 0 is in the free set")));
    }

    #[test]
    fn reported_metrics_must_match_replay() {
        let (p, a) = (sid(0), sid(1));
        let (state, _) = replay(
            4,
            &[
                KvEvent::SequenceAllocated {
                    seq: p,
                    blocks: vec![0, 1],
                },
                KvEvent::Forked {
                    parent: p,
                    child: a,
                },
            ],
        );
        let honest = ReportedKvMetrics {
            physical_pages: 2,
            logical_pages: 4,
            shared_pages: 2,
            private_pages: 0,
            cow_faults: 0,
            used_blocks: 2,
        };
        assert!(check_replay_matches_metrics(&state, &honest).is_empty());

        let buggy = ReportedKvMetrics {
            shared_pages: 1,
            private_pages: 1,
            ..honest
        };
        assert_eq!(check_replay_matches_metrics(&state, &buggy).len(), 2);
    }
}
