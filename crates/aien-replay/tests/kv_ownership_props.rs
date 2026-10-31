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

//! Property tests: KV block ownership under fork / append / free / reserve.
//!
//! A random program of KV operations runs against `AienKvManager` and against
//! a tiny reference model (each live sequence = the list of token values it
//! should read back). After every step the checks below must hold. Each check
//! has a named mutation in `mutations/kv.mutants` that it is known to catch;
//! `mutations/run_mutants.sh` proves that.
//!
//! - ownership: every block's ref_count equals the number of block tables
//!   that hold it; `is_shared` is true exactly when ref_count > 1.
//! - conservation: free blocks + referenced blocks == pool size (no leak, no
//!   double free), and every block a sequence holds is not on the free list.
//! - content: every sequence reads back exactly the values written to it, so
//!   a write to one branch never shows up in another (copy-on-write isolation).
//! - rollback: reserve_token followed by rollback leaves the manager state
//!   identical to before the reservation.
//!
//! Forking onto a sequence id that already exists replaces it and releases its
//! old blocks (issue #142; see `fork_onto_live_child_leaks_blocks` below).

use aien_kv_cache::{AienKvManager, HeapUnifiedBuffer, KvDType, KvPoolConfig};
use proptest::prelude::*;
use std::collections::BTreeMap;

const TOTAL_BLOCKS: usize = 24;
const BLOCK_SIZE: usize = 4;
const SEQ_IDS: u64 = 6;
const KV_DIM: usize = 2;

type Mgr = AienKvManager<HeapUnifiedBuffer>;
type Model = BTreeMap<u64, Vec<f32>>;

fn new_mgr() -> Mgr {
    let cfg = KvPoolConfig {
        num_blocks: TOTAL_BLOCKS,
        block_size: BLOCK_SIZE,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: KV_DIM,
        dtype: KvDType::Fp32,
    };
    let buf = HeapUnifiedBuffer::allocate(cfg.total_bytes(), 64).expect("heap buffer");
    AienKvManager::with_buffer(TOTAL_BLOCKS, BLOCK_SIZE, cfg, buf).expect("manager")
}

#[derive(Debug, Clone)]
enum Op {
    Alloc { seq: u64, n: usize },
    Fork { parent: u64, child: u64 },
    Append { seq: u64 },
    Free { seq: u64 },
    ReserveRollback { seq: u64 },
    ReserveCommit { seq: u64 },
}

fn op() -> impl Strategy<Value = Op> {
    let s = 0..SEQ_IDS;
    prop_oneof![
        1 => (s.clone(), 0usize..10).prop_map(|(seq, n)| Op::Alloc { seq, n }),
        3 => (s.clone(), s.clone()).prop_map(|(parent, child)| Op::Fork { parent, child }),
        4 => s.clone().prop_map(|seq| Op::Append { seq }),
        1 => s.clone().prop_map(|seq| Op::Free { seq }),
        2 => s.clone().prop_map(|seq| Op::ReserveRollback { seq }),
        1 => s.prop_map(|seq| Op::ReserveCommit { seq }),
    ]
}

/// Everything about the manager that a caller can observe, for equality.
#[derive(Debug, PartialEq)]
struct Observed {
    tables: Vec<(u64, Vec<usize>, usize)>,
    blocks: Vec<(usize, usize, bool)>,
    free: usize,
    content: Vec<(u64, Vec<f32>)>,
}

fn observe(mgr: &Mgr, model: &Model) -> Observed {
    let mut tables = Vec::new();
    let mut content = Vec::new();
    for &seq in model.keys() {
        if let Some(t) = mgr.get_block_table(seq) {
            tables.push((seq, t.block_ids.clone(), t.total_tokens));
            content.push((
                seq,
                mgr.gather_layer_kv(seq, 0)
                    .map(|kv| kv.0)
                    .unwrap_or_default(),
            ));
        }
    }
    let blocks = (0..TOTAL_BLOCKS)
        .map(|b| {
            let blk = mgr.get_block(b).expect("block exists");
            (blk.ref_count, blk.num_tokens, blk.is_shared)
        })
        .collect();
    Observed {
        tables,
        blocks,
        free: mgr.free_block_count(),
        content,
    }
}

fn token_value(counter: &mut u32) -> f32 {
    *counter += 1;
    *counter as f32
}

fn write_token(mgr: &mut Mgr, block: usize, slot: usize, val: f32) {
    let k = [val; KV_DIM];
    let v = [-val; KV_DIM];
    mgr.write_explicit_token_kv(block, 0, slot, &k, &v)
        .expect("pool attached");
}

fn check(mgr: &Mgr, model: &Model) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        mgr.active_sequence_count(),
        model.len(),
        "live sequence count"
    );

    let mut holders = vec![0usize; TOTAL_BLOCKS];
    for (&seq, values) in model {
        let t = mgr
            .get_block_table(seq)
            .ok_or_else(|| TestCaseError::fail(format!("seq {seq} has no table")))?;
        prop_assert_eq!(t.total_tokens, values.len(), "seq {} token count", seq);
        prop_assert_eq!(
            t.block_ids.len(),
            values.len().div_ceil(BLOCK_SIZE),
            "seq {} block count",
            seq
        );
        let mut in_blocks = 0;
        for (i, &b) in t.block_ids.iter().enumerate() {
            holders[b] += 1;
            let n = mgr.get_block(b).expect("block").num_tokens;
            if i + 1 < t.block_ids.len() {
                prop_assert_eq!(n, BLOCK_SIZE, "seq {} non-tail block {} not full", seq, b);
            }
            in_blocks += n;
        }
        prop_assert_eq!(
            in_blocks,
            values.len(),
            "seq {} tokens recorded in blocks",
            seq
        );

        // content: what the sequence reads back is exactly what was written to it
        let (k, v) = mgr.gather_layer_kv(seq, 0).map_err(TestCaseError::fail)?;
        let want_k: Vec<f32> = values.iter().flat_map(|&x| [x; KV_DIM]).collect();
        let want_v: Vec<f32> = values.iter().flat_map(|&x| [-x; KV_DIM]).collect();
        prop_assert_eq!(k, want_k, "seq {} K content", seq);
        prop_assert_eq!(v, want_v, "seq {} V content", seq);
    }

    let mut referenced = 0;
    for (b, &h) in holders.iter().enumerate() {
        let blk = mgr.get_block(b).expect("block");
        prop_assert_eq!(blk.ref_count, h, "block {} ref_count vs holders", b);
        if blk.ref_count > 0 {
            referenced += 1;
            prop_assert_eq!(
                blk.is_shared,
                blk.ref_count > 1,
                "block {} is_shared flag",
                b
            );
        }
    }
    prop_assert_eq!(
        mgr.free_block_count() + referenced,
        TOTAL_BLOCKS,
        "free + referenced must equal pool size (leak or double free)"
    );
    Ok(())
}

fn apply(
    mgr: &mut Mgr,
    model: &mut Model,
    counter: &mut u32,
    op: &Op,
) -> Result<(), TestCaseError> {
    match *op {
        Op::Alloc { seq, n } => {
            if model.contains_key(&seq) {
                // existing sequence: the call returns its table and changes nothing
                mgr.allocate_sequence(seq, &[0; 3])
                    .map_err(TestCaseError::fail)?;
                return Ok(());
            }
            let prompt: Vec<u32> = (0..n as u32).collect();
            if mgr.allocate_sequence(seq, &prompt).is_ok() {
                let mut vals = Vec::with_capacity(n);
                for pos in 0..n {
                    let val = token_value(counter);
                    mgr.write_token_kv(seq, pos, 0, &[val; KV_DIM], &[-val; KV_DIM])
                        .map_err(TestCaseError::fail)?;
                    vals.push(val);
                }
                model.insert(seq, vals);
            }
        }
        Op::Fork { parent, child } => {
            if parent == child {
                return Ok(()); // self-fork is refused, see fork_onto_self_is_refused_and_harmless
            }
            match mgr.fork_sequence(parent, child) {
                Ok(_) => {
                    let vals = model
                        .get(&parent)
                        .cloned()
                        .ok_or_else(|| TestCaseError::fail("fork of missing parent succeeded"))?;
                    model.insert(child, vals);
                }
                Err(_) => prop_assert!(!model.contains_key(&parent), "fork of live parent refused"),
            }
        }
        Op::Append { seq } => {
            if let Ok((block, slot)) = mgr.append_token_with_slot(seq) {
                let vals = model
                    .get_mut(&seq)
                    .ok_or_else(|| TestCaseError::fail("append to missing seq succeeded"))?;
                prop_assert_eq!(slot, vals.len() % BLOCK_SIZE, "slot for seq {}", seq);
                let val = token_value(counter);
                write_token(mgr, block, slot, val);
                vals.push(val);
            }
        }
        Op::Free { seq } => {
            let ok = mgr.free_sequence(seq).is_ok();
            prop_assert_eq!(
                ok,
                model.remove(&seq).is_some(),
                "free result for seq {}",
                seq
            );
        }
        Op::ReserveRollback { seq } => {
            let before = observe(mgr, model);
            if let Ok(r) = mgr.reserve_token(seq) {
                prop_assert!(model.contains_key(&seq), "reserve on missing seq succeeded");
                mgr.rollback(r).map_err(TestCaseError::fail)?;
            }
            let after = observe(mgr, model);
            prop_assert_eq!(before, after, "reserve+rollback must restore state");
        }
        Op::ReserveCommit { seq } => {
            if let Ok(r) = mgr.reserve_token(seq) {
                let (block, slot) = (r.block_id, r.slot);
                mgr.commit(r).map_err(TestCaseError::fail)?;
                let vals = model
                    .get_mut(&seq)
                    .ok_or_else(|| TestCaseError::fail("reserve on missing seq succeeded"))?;
                let val = token_value(counter);
                write_token(mgr, block, slot, val);
                vals.push(val);
            }
        }
    }
    Ok(())
}

/// Case count: PROPTEST_CASES if set, else 256 (8 under Miri, which is slow).
fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if cfg!(miri) { 8 } else { 256 })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn kv_ownership_holds_under_random_programs(ops in prop::collection::vec(op(), 1..60)) {
        let mut mgr = new_mgr();
        let mut model = Model::new();
        let mut counter = 0u32;
        for op in &ops {
            apply(&mut mgr, &mut model, &mut counter, op)?;
            check(&mgr, &model)?;
        }
        // drain: freeing every live sequence returns the whole pool
        let live: Vec<u64> = model.keys().copied().collect();
        for seq in live {
            mgr.free_sequence(seq).map_err(TestCaseError::fail)?;
            model.remove(&seq);
            check(&mgr, &model)?;
        }
        prop_assert_eq!(mgr.free_block_count(), TOTAL_BLOCKS);
    }
}

/// Issue #142: forking onto a sequence id that is still live replaces it and
/// must release its old blocks back to the pool.
#[test]
fn fork_onto_live_child_leaks_blocks() {
    let mut mgr = new_mgr();
    mgr.allocate_sequence(1, &[1, 2, 3, 4, 5]).unwrap();
    mgr.allocate_sequence(2, &[1, 2, 3]).unwrap();
    // fork_sequence either refuses or releases seq 2's old block
    let _ = mgr.fork_sequence(1, 2);
    mgr.free_sequence(1).unwrap();
    mgr.free_sequence(2).unwrap();
    assert_eq!(mgr.free_block_count(), TOTAL_BLOCKS, "blocks leaked");
}

/// Re-forking onto a child that already shares the parent's blocks must keep
/// the counts right: nothing leaks and nothing is freed while still in use.
#[test]
fn refork_onto_existing_child_keeps_counts() {
    let mut mgr = new_mgr();
    mgr.allocate_sequence(1, &[1, 2, 3, 4, 5]).unwrap();
    mgr.fork_sequence(1, 2).unwrap();
    mgr.fork_sequence(1, 2).unwrap();
    mgr.free_sequence(1).unwrap();
    assert!(mgr.get_block_table(2).is_some());
    mgr.free_sequence(2).unwrap();
    assert_eq!(mgr.free_block_count(), TOTAL_BLOCKS, "blocks leaked");
}

/// Forking a sequence onto itself is refused and leaves it untouched.
#[test]
fn fork_onto_self_is_refused_and_harmless() {
    let mut mgr = new_mgr();
    mgr.allocate_sequence(1, &[1, 2, 3, 4, 5]).unwrap();
    let before = mgr.get_block_table(1).unwrap().block_ids.clone();
    assert!(mgr.fork_sequence(1, 1).is_err());
    assert_eq!(mgr.get_block_table(1).unwrap().block_ids, before);
    mgr.free_sequence(1).unwrap();
    assert_eq!(mgr.free_block_count(), TOTAL_BLOCKS);
}
