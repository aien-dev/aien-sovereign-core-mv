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

//! Holder ownership primitives (persistent-KV cut C3, ADR 0003 section 5.2).
//!
//! A holder is an ordinary block table whose id lives in the reserved slot
//! `HOLDER_SLOT` (`u32::MAX`, the high 32 bits of the u64 sequence id). A
//! storage job forks a sequence into a holder to keep its blocks alive after
//! the sequence frees them, and `adopt_table` moves a holder's table to a real
//! sequence id without touching reference counts. Every test here uses only the
//! existing refcount, copy-on-write and free-list paths plus the new primitives.

use aien_kv_cache::{
    holder_id, is_holder_id, AienKvManager, HeapUnifiedBuffer, KvDType, KvPoolConfig, PrefillState,
    SuffixPolicy, HOLDER_SLOT,
};

const BS: usize = 4;
const LAYERS: usize = 2;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 2;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;

type Mgr = AienKvManager<HeapUnifiedBuffer>;

fn new_mgr(total_blocks: usize) -> Mgr {
    let cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size: BS,
        num_layers: LAYERS,
        num_kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        dtype: KvDType::Fp32,
    };
    let buf = HeapUnifiedBuffer::allocate(cfg.total_bytes(), 64).expect("heap buffer");
    AienKvManager::with_buffer(total_blocks, BS, cfg, buf).expect("manager")
}

fn kval(val: f32, layer: usize) -> f32 {
    val + layer as f32 * 0.25
}

/// Raw read of every slot of every layer of `block`, whatever the tables say.
fn raw_block(m: &Mgr, block: usize) -> Vec<f32> {
    let pool = m.tensor_pool().expect("pool attached");
    let mut out = Vec::new();
    for layer in 0..LAYERS {
        for slot in 0..BS {
            let mut k = [0.0f32; KV_DIM];
            let mut v = [0.0f32; KV_DIM];
            pool.read_token_kv(block, layer, slot, &mut k, &mut v);
            out.extend_from_slice(&k);
            out.extend_from_slice(&v);
        }
    }
    out
}

/// Allocate `seq`, write one value per token in every layer, fire the prefill fence.
fn admit(m: &mut Mgr, seq: u64, vals: &[f32]) {
    let prompt: Vec<u32> = (0..vals.len() as u32).collect();
    m.allocate_sequence(seq, &prompt).expect("allocate");
    for (pos, &val) in vals.iter().enumerate() {
        for layer in 0..LAYERS {
            m.write_token_kv(
                seq,
                pos,
                layer,
                &[kval(val, layer); KV_DIM],
                &[-kval(val, layer); KV_DIM],
            )
            .expect("write");
        }
    }
    m.mark_prefill_pending(seq).expect("pending");
    m.complete_prefill(seq).expect("fence");
}

fn table_blocks(m: &Mgr, seq: u64) -> Vec<usize> {
    m.get_block_table(seq).expect("table").block_ids.clone()
}

fn ref_count(m: &Mgr, block: usize) -> usize {
    m.get_block(block).expect("block").ref_count
}

fn vals(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| base + i as f32).collect()
}

#[test]
fn holder_fork_keeps_blocks_after_owner_free() {
    let total = 16;
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(10, 1.0)); // 3 blocks, partial tail
    let blocks = table_blocks(&m, 1);
    assert_eq!(blocks.len(), 3);
    let before: Vec<Vec<f32>> = blocks.iter().map(|&b| raw_block(&m, b)).collect();

    let holder = holder_id(7).expect("holder id");
    m.fork_sequence(1, holder).expect("fork into holder");
    assert_eq!(m.holder_block_count(), 3);

    m.free_sequence(1).expect("owner free");

    // The owner is gone, the holder keeps every block alive.
    assert_eq!(m.free_block_count(), total - 3);
    assert_eq!(m.holder_block_count(), 3);
    assert_eq!(table_blocks(&m, holder), blocks);
    for &b in &blocks {
        assert_eq!(ref_count(&m, b), 1, "block {b} held by the holder only");
    }
    // None of the held blocks is handed to a new sequence, even when the rest
    // of the pool is drained.
    let drain: Vec<u32> = (0..((total - 3) * BS) as u32).collect();
    let drained = m.allocate_sequence(2, &drain).expect("drain free pool");
    assert_eq!(m.free_block_count(), 0);
    for b in &drained {
        assert!(!blocks.contains(b), "held block {b} was reused");
    }
    // Held bytes are untouched.
    for (i, &b) in blocks.iter().enumerate() {
        assert_eq!(raw_block(&m, b), before[i], "block {b} bytes changed");
    }
}

#[test]
fn holder_free_returns_blocks_only_at_zero_refcount() {
    let total = 8;
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(8, 1.0)); // 2 full blocks
    let blocks = table_blocks(&m, 1);
    let holder = holder_id(1).unwrap();
    m.fork_sequence(1, holder).unwrap();
    for &b in &blocks {
        assert_eq!(ref_count(&m, b), 2);
    }

    // Holder released first: the owner still has the blocks.
    m.free_sequence(holder).expect("holder free");
    assert_eq!(m.holder_block_count(), 0);
    assert_eq!(m.free_block_count(), total - 2);
    for &b in &blocks {
        assert_eq!(ref_count(&m, b), 1);
    }

    // Park again, owner frees first, holder free is the last reference.
    m.fork_sequence(1, holder).unwrap();
    m.free_sequence(1).unwrap();
    assert_eq!(m.free_block_count(), total - 2, "holder still holds them");
    m.free_sequence(holder).unwrap();
    assert_eq!(m.free_block_count(), total, "last reference returns blocks");
    for &b in &blocks {
        assert_eq!(ref_count(&m, b), 0);
    }
    assert_eq!(m.holder_block_count(), 0);
    assert_eq!(m.active_sequence_count(), 0);
}

#[test]
fn adopt_table_moves_table_and_keeps_refcounts() {
    let total = 12;
    let mut m = new_mgr(total);
    // Restore shape: the holder allocates its own blocks, the load fills them,
    // the fence fires, then the table is adopted by the request's sequence id.
    let holder = holder_id(42).unwrap();
    let prompt: Vec<u32> = (0..10).collect();
    m.allocate_sequence(holder, &prompt).unwrap();
    m.mark_prefill_pending(holder).unwrap();
    m.complete_prefill(holder).unwrap();
    let blocks = table_blocks(&m, holder);
    assert_eq!(blocks.len(), 3);
    assert_eq!(m.holder_block_count(), 3);

    let free_before = m.free_block_count();
    let counts_before: Vec<usize> = blocks.iter().map(|&b| ref_count(&m, b)).collect();
    let active_before = m.debug_active_blocks();
    let seq_count_before = m.active_sequence_count();

    let seq = 5u64;
    m.adopt_table(holder, seq).expect("adopt");

    assert!(m.get_block_table(holder).is_none(), "holder table is gone");
    let t = m.get_block_table(seq).expect("seq has the table");
    assert_eq!(t.sequence_id, seq);
    assert_eq!(t.block_ids, blocks);
    assert_eq!(t.total_tokens, 10);
    assert_eq!(t.prefill_state, PrefillState::PrefillReady);
    assert_eq!(m.free_block_count(), free_before);
    assert_eq!(m.active_sequence_count(), seq_count_before);
    assert_eq!(m.debug_active_blocks(), active_before);
    let counts_after: Vec<usize> = blocks.iter().map(|&b| ref_count(&m, b)).collect();
    assert_eq!(counts_after, counts_before);
    assert_eq!(m.holder_block_count(), 0, "no longer a holder block");

    // The adopted sequence is a normal one: it frees to the pool.
    m.free_sequence(seq).unwrap();
    assert_eq!(m.free_block_count(), total);
}

#[test]
fn adopt_table_refusals_change_nothing() {
    let mut m = new_mgr(12);
    let holder = holder_id(1).unwrap();
    m.allocate_sequence(holder, &[0, 1, 2, 3, 4]).unwrap();
    admit(&mut m, 9, &vals(3, 1.0));
    let free = m.free_block_count();
    let active = m.debug_active_blocks();
    let holder_blocks = table_blocks(&m, holder);
    let nine_blocks = table_blocks(&m, 9);

    // Unknown holder.
    assert!(m.adopt_table(holder_id(2).unwrap(), 5).is_err());
    // Destination already has a table: not replaced.
    assert!(m.adopt_table(holder, 9).is_err());
    // Source is not a holder id: a real sequence cannot be stolen this way.
    assert!(m.adopt_table(9, 5).is_err());
    // Destination is a holder id.
    assert!(m.adopt_table(holder, holder_id(3).unwrap()).is_err());
    // Self.
    assert!(m.adopt_table(holder, holder).is_err());

    assert_eq!(m.free_block_count(), free);
    assert_eq!(m.debug_active_blocks(), active);
    assert_eq!(table_blocks(&m, holder), holder_blocks);
    assert_eq!(table_blocks(&m, 9), nine_blocks);
    assert_eq!(m.get_block_table(9).unwrap().sequence_id, 9);
}

#[test]
fn owner_append_on_shared_partial_block_copies_not_holder() {
    let mut m = new_mgr(12);
    admit(&mut m, 1, &vals(6, 1.0)); // block 0 full, block 1 holds 2 of 4
    let blocks = table_blocks(&m, 1);
    assert_eq!(blocks.len(), 2);
    let holder = holder_id(3).unwrap();
    m.fork_sequence(1, holder).unwrap();
    let held_bytes: Vec<Vec<f32>> = blocks.iter().map(|&b| raw_block(&m, b)).collect();
    let faults_before = m.cow_faults();

    // The owner resumes decoding while the park is in flight.
    let (new_block, slot) = m.append_token_with_slot(1).expect("append");
    for layer in 0..LAYERS {
        m.write_explicit_token_kv(new_block, layer, slot, &[99.0; KV_DIM], &[-99.0; KV_DIM])
            .unwrap();
    }

    assert_eq!(m.cow_faults(), faults_before + 1, "exactly one COW fault");
    assert_ne!(
        new_block, blocks[1],
        "owner copied the shared partial block"
    );
    let owner_blocks = table_blocks(&m, 1);
    assert_eq!(owner_blocks[0], blocks[0], "full prefix block stays shared");
    assert_eq!(owner_blocks[1], new_block);
    // The holder still points at the frozen originals, bytes unchanged.
    assert_eq!(table_blocks(&m, holder), blocks);
    assert_eq!(m.get_block_table(holder).unwrap().total_tokens, 6);
    for (i, &b) in blocks.iter().enumerate() {
        assert_eq!(raw_block(&m, b), held_bytes[i], "held block {b} changed");
    }
    assert_eq!(ref_count(&m, blocks[0]), 2);
    assert_eq!(ref_count(&m, blocks[1]), 1, "held only by the holder now");
}

#[test]
fn shared_prefix_sibling_valid_after_holder_free() {
    let total = 16;
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(8, 1.0)); // 2 full blocks
    m.fork_prefilled(1, 2).expect("sibling shares the prefix");
    let prefix = table_blocks(&m, 2);
    assert_eq!(prefix, table_blocks(&m, 1));
    let bytes: Vec<Vec<f32>> = prefix.iter().map(|&b| raw_block(&m, b)).collect();

    // Sequence 1 parks (holder) and finishes; then the park is published and
    // the holder released.
    let holder = holder_id(11).unwrap();
    m.fork_sequence(1, holder).unwrap();
    m.free_sequence(1).unwrap();
    assert_eq!(m.holder_block_count(), 2);
    m.free_sequence(holder).unwrap();
    assert_eq!(m.holder_block_count(), 0);

    // Sequence 2 still owns the prefix and can decode: it reads its blocks and
    // appends a new one.
    assert_eq!(table_blocks(&m, 2), prefix);
    assert!(m.get_block_table(2).unwrap().is_prefill_ready());
    assert_eq!(m.free_block_count(), total - 2);
    for (i, &b) in prefix.iter().enumerate() {
        assert_eq!(ref_count(&m, b), 1);
        assert_eq!(raw_block(&m, b), bytes[i], "prefix block {b} changed");
    }
    let mut k = Vec::new();
    let mut v = Vec::new();
    m.gather_sequence_layer_kv(2, 0, &mut k, &mut v)
        .expect("sibling gathers its prefix");
    assert_eq!(k.len(), 8 * KV_DIM);
    assert_eq!(k[0], kval(1.0, 0));
    assert_eq!(k[7 * KV_DIM], kval(8.0, 0));
    let (b, slot) = m.append_token_with_slot(2).expect("sibling appends");
    assert!(!prefix.contains(&b));
    assert_eq!(slot, 0);

    m.free_sequence(2).unwrap();
    assert_eq!(m.free_block_count(), total);
}

#[test]
fn holder_id_never_collides_with_arena_ids() {
    // Arena ids are `slot << 32 | generation` with generation >= 1
    // (aien-scheduler SequenceId::to_u64). The arena hands out slots from 0 and
    // never reaches u32::MAX in practice; the holder range is exactly the ids
    // whose slot IS u32::MAX.
    assert_eq!(HOLDER_SLOT, u32::MAX);
    for job in [1u64, 2, 255, 65_536, u32::MAX as u64 - 1, u32::MAX as u64] {
        let id = holder_id(job).expect("valid job id");
        assert_eq!(id >> 32, HOLDER_SLOT as u64, "slot is the reserved one");
        assert_eq!(id & 0xFFFF_FFFF, job, "generation is the job id");
        assert!(is_holder_id(id));
    }
    // Distinct jobs give distinct ids.
    assert_ne!(holder_id(1).unwrap(), holder_id(2).unwrap());

    // Every arena-shaped id below the reserved slot is not a holder id, for
    // the first generations and for the generation boundary.
    for slot in (0u32..70_000).chain([u32::MAX - 2, u32::MAX - 1]) {
        for generation in [1u32, 2, 3, u32::MAX - 1, u32::MAX] {
            let arena_id = ((slot as u64) << 32) | generation as u64;
            assert!(!is_holder_id(arena_id), "slot {slot} gen {generation}");
            assert_ne!(Some(arena_id), holder_id(generation as u64).ok());
        }
    }

    // Refusals instead of silent truncation: generation zero is invalid, and a
    // job id past u32 would wrap onto another live holder.
    assert!(holder_id(0).is_err());
    assert!(holder_id(u32::MAX as u64 + 1).is_err());
    assert!(holder_id(u64::MAX).is_err());
}

#[test]
fn holder_block_count_counts_distinct_blocks_of_holders_only() {
    let mut m = new_mgr(16);
    assert_eq!(m.holder_block_count(), 0);
    admit(&mut m, 1, &vals(8, 1.0)); // 2 blocks
    admit(&mut m, 2, &vals(4, 5.0)); // 1 block
    assert_eq!(
        m.holder_block_count(),
        0,
        "ordinary sequences are not holders"
    );

    let h1 = holder_id(1).unwrap();
    let h2 = holder_id(2).unwrap();
    m.fork_sequence(1, h1).unwrap();
    m.fork_sequence(1, h2).unwrap(); // second holder over the same blocks
    assert_eq!(m.holder_block_count(), 2, "shared blocks counted once");
    m.fork_sequence(2, holder_id(3).unwrap()).unwrap();
    assert_eq!(m.holder_block_count(), 3);

    m.free_sequence(h1).unwrap();
    assert_eq!(m.holder_block_count(), 3, "h2 still holds the first two");
    m.free_sequence(h2).unwrap();
    assert_eq!(m.holder_block_count(), 1);
    m.free_sequence(holder_id(3).unwrap()).unwrap();
    assert_eq!(m.holder_block_count(), 0);
}

#[test]
fn holder_partial_failure_keeps_inflight_destinations() {
    // One restore allocates a holder over four chunks (one block each). Each
    // chunk load gets its own sub-holder over exactly its destination block.
    // Chunk 2 fails and its sub-holder is freed; the parent holder is freed too.
    // Chunks 0, 1 and 3 are still being written: their blocks must stay out of
    // the free list and must not be handed to another sequence.
    let total = 8;
    let mut m = new_mgr(total);
    let parent = holder_id(100).unwrap();
    let prompt: Vec<u32> = (0..16).collect();
    m.allocate_sequence(parent, &prompt).unwrap();
    m.mark_prefill_pending(parent).unwrap();
    let blocks = table_blocks(&m, parent);
    assert_eq!(blocks.len(), 4);

    let jobs: Vec<u64> = (0..4).map(|i| holder_id(101 + i).unwrap()).collect();
    for (i, &job) in jobs.iter().enumerate() {
        let got = m.fork_range(parent, i, 1, job).expect("sub-holder");
        assert_eq!(got, vec![blocks[i]]);
        assert_eq!(m.get_block_table(job).unwrap().total_tokens, BS);
    }
    for &b in &blocks {
        assert_eq!(ref_count(&m, b), 2, "parent holder plus its job holder");
    }

    // Chunk 2 fails: its job holder goes. Then the whole-restore holder goes.
    m.free_sequence(jobs[2]).unwrap();
    m.free_sequence(parent).unwrap();

    assert_eq!(
        m.free_block_count(),
        total - 3,
        "only the failed chunk returned"
    );
    for (i, &b) in blocks.iter().enumerate() {
        let expect = if i == 2 { 0 } else { 1 };
        assert_eq!(ref_count(&m, b), expect, "block {i}");
    }
    // Drain the pool: the three in-flight destinations are never reused.
    let drain: Vec<u32> = (0..((total - 3) * BS) as u32).collect();
    let got = m.allocate_sequence(50, &drain).unwrap();
    for (i, b) in blocks.iter().enumerate() {
        if i != 2 {
            assert!(!got.contains(b), "in-flight destination {b} was reused");
        }
    }
    assert_eq!(m.holder_block_count(), 3);

    // The surviving jobs finish and release; now everything is returned.
    for &i in &[0usize, 1, 3] {
        m.free_sequence(jobs[i]).unwrap();
    }
    assert_eq!(m.holder_block_count(), 0);
    m.free_sequence(50).unwrap();
    assert_eq!(m.free_block_count(), total);
}

#[test]
fn fork_of_prefix_chunk_does_not_share_suffix_blocks() {
    // Coalescing two requests whose common part is only the first chunk must
    // not share the whole table: the suffix blocks stay owned by one side.
    let total = 12;
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(14, 1.0)); // 4 blocks, last one partial
    let blocks = table_blocks(&m, 1);
    assert_eq!(blocks.len(), 4);

    let child = 2u64;
    let got = m.fork_range(1, 0, 2, child).expect("prefix fork");
    assert_eq!(got, blocks[..2].to_vec());
    let t = m.get_block_table(child).unwrap();
    assert_eq!(t.block_ids, blocks[..2].to_vec());
    assert_eq!(t.total_tokens, 2 * BS);
    assert!(t.is_prefill_ready());
    assert_eq!(ref_count(&m, blocks[0]), 2);
    assert_eq!(ref_count(&m, blocks[1]), 2);
    assert_eq!(ref_count(&m, blocks[2]), 1, "suffix not shared");
    assert_eq!(ref_count(&m, blocks[3]), 1, "suffix not shared");

    // Freeing the owner returns the suffix and keeps the shared prefix.
    m.free_sequence(1).unwrap();
    assert_eq!(m.free_block_count(), total - 2);
    assert_eq!(table_blocks(&m, child), blocks[..2].to_vec());

    // A range ending at the tail block carries the partial token count.
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(14, 1.0));
    m.fork_range(1, 2, 2, 3).unwrap();
    assert_eq!(m.get_block_table(3).unwrap().total_tokens, 14 - 2 * BS);
}

#[test]
fn fork_range_refusals_change_nothing() {
    let mut m = new_mgr(12);
    admit(&mut m, 1, &vals(8, 1.0));
    admit(&mut m, 2, &vals(4, 5.0));
    let free = m.free_block_count();
    let active = m.debug_active_blocks();
    assert!(m.fork_range(1, 0, 0, 3).is_err(), "empty range");
    assert!(m.fork_range(1, 1, 2, 3).is_err(), "past the table");
    assert!(
        m.fork_range(1, usize::MAX, 2, 3).is_err(),
        "overflowing range"
    );
    assert!(m.fork_range(9, 0, 1, 3).is_err(), "unknown parent");
    assert!(m.fork_range(1, 0, 1, 1).is_err(), "onto itself");
    assert!(m.fork_range(1, 0, 1, 2).is_err(), "live child kept");
    assert_eq!(m.free_block_count(), free);
    assert_eq!(m.debug_active_blocks(), active);
    // Readiness travels with the blocks, like fork_sequence: an unready parent
    // (a restore destination still loading) gives an unready sub-holder.
    m.allocate_sequence(7, &[0, 1, 2, 3]).unwrap();
    m.fork_range(7, 0, 1, 8).unwrap();
    assert_eq!(m.prefill_state(8), Some(PrefillState::Allocated));
    assert_eq!(m.prefill_state(7), Some(PrefillState::Allocated));
}

#[test]
fn fork_range_forks_only_the_named_blocks_and_does_not_replace_a_live_child() {
    let total = 16;
    let mut m = new_mgr(total);
    admit(&mut m, 1, &vals(16, 1.0)); // 4 full blocks
    let blocks = table_blocks(&m, 1);
    assert_eq!(m.prefill_state(1), Some(PrefillState::PrefillReady));

    // Only the two middle blocks, and the parent's state is left alone.
    let h = holder_id(1).unwrap();
    let got = m.fork_range(1, 1, 2, h).expect("middle range");
    assert_eq!(got, blocks[1..3].to_vec());
    assert_eq!(table_blocks(&m, h), blocks[1..3].to_vec());
    assert_eq!(m.get_block_table(h).unwrap().total_tokens, 2 * BS);
    assert_eq!(ref_count(&m, blocks[0]), 1);
    assert_eq!(ref_count(&m, blocks[1]), 2);
    assert_eq!(ref_count(&m, blocks[2]), 2);
    assert_eq!(ref_count(&m, blocks[3]), 1);
    assert_eq!(
        m.prefill_state(1),
        Some(PrefillState::PrefillReady),
        "fork_range leaves the parent's state alone"
    );

    // A live child is refused and untouched (fork_sequence would free it).
    admit(&mut m, 2, &vals(8, 50.0));
    let own = table_blocks(&m, 2);
    let free = m.free_block_count();
    let before: Vec<Vec<f32>> = own.iter().map(|&b| raw_block(&m, b)).collect();
    assert!(m.fork_range(1, 0, 1, 2).is_err());
    assert_eq!(table_blocks(&m, 2), own);
    assert_eq!(m.free_block_count(), free);
    for (i, &b) in own.iter().enumerate() {
        assert_eq!(ref_count(&m, b), 1);
        assert_eq!(raw_block(&m, b), before[i]);
    }
    // A live holder is refused the same way.
    assert!(m.fork_range(1, 0, 1, h).is_err());
    assert_eq!(table_blocks(&m, h), blocks[1..3].to_vec());
    assert_eq!(ref_count(&m, blocks[0]), 1);
}

/// Three chunk holders over a 14-token prefix: 8 tokens (2 blocks), 4 tokens
/// (1 block), 2 tokens (1 partial block). Returns (holders, their blocks).
fn chunk_holders(m: &mut Mgr) -> (Vec<u64>, Vec<Vec<usize>>) {
    let specs = [(1u64, 8usize), (2, 4), (3, 2)];
    let mut holders = Vec::new();
    let mut blocks = Vec::new();
    for (job, n) in specs {
        let h = holder_id(job).unwrap();
        let prompt: Vec<u32> = (0..n as u32).collect();
        m.allocate_sequence(h, &prompt).unwrap();
        m.mark_prefill_pending(h).unwrap();
        m.complete_prefill(h).unwrap();
        blocks.push(table_blocks(m, h));
        holders.push(h);
    }
    (holders, blocks)
}

#[test]
fn assemble_table_orders_holders_and_refcounts() {
    let total = 16;
    let mut m = new_mgr(total);
    let (holders, blocks) = chunk_holders(&mut m);
    let free = m.free_block_count();

    // Holders given out of storage order are laid out in the given order.
    let order = [holders[0], holders[1], holders[2]];
    let got = m
        .assemble_table(5, &order, SuffixPolicy::AllowPartialTail)
        .expect("assemble");
    let expect: Vec<usize> = blocks.iter().flatten().copied().collect();
    assert_eq!(got, expect);
    let t = m.get_block_table(5).unwrap();
    assert_eq!(t.sequence_id, 5);
    assert_eq!(t.block_ids, expect);
    assert_eq!(t.total_tokens, 14);
    assert_eq!(t.prefill_state, PrefillState::PrefillPending);
    assert_eq!(m.free_block_count(), free, "no block allocated or freed");
    for &b in &expect {
        assert_eq!(ref_count(&m, b), 2, "holder plus the new table");
    }
    // Holders are untouched, and still hold their blocks.
    for (h, bl) in holders.iter().zip(&blocks) {
        assert_eq!(&table_blocks(&m, *h), bl);
    }

    // Two requests assemble from the same chunk holder: refcounts add up, and
    // each table is independent.
    m.assemble_table(6, &holders[..2], SuffixPolicy::FullBlocksOnly)
        .expect("second request, full blocks only");
    assert_eq!(ref_count(&m, blocks[0][0]), 3);
    assert_eq!(m.get_block_table(6).unwrap().total_tokens, 12);

    // Freeing holders then tables returns everything exactly once.
    for &h in &holders {
        m.free_sequence(h).unwrap();
    }
    assert_eq!(m.free_block_count(), free);
    m.free_sequence(5).unwrap();
    m.free_sequence(6).unwrap();
    assert_eq!(m.free_block_count(), total);
}

#[test]
fn assemble_table_failure_leaves_every_holder_untouched() {
    let mut m = new_mgr(16);
    let (holders, blocks) = chunk_holders(&mut m);
    admit(&mut m, 9, &vals(4, 1.0));
    let counts: Vec<usize> = (0..16).map(|b| ref_count(&m, b)).collect();
    let unknown = holder_id(77).unwrap();
    let pending = holder_id(78).unwrap();
    m.allocate_sequence(pending, &[0, 1, 2, 3]).unwrap(); // still loading
    let free = m.free_block_count();
    let active = m.debug_active_blocks();
    let counts_p: Vec<usize> = (0..16).map(|b| ref_count(&m, b)).collect();

    let allow = SuffixPolicy::AllowPartialTail;
    // Seq already has a table: never replaced.
    assert!(m.assemble_table(9, &holders, allow).is_err());
    // Unknown holder in the last position: the good ones must not be touched.
    assert!(m
        .assemble_table(5, &[holders[0], holders[1], unknown], allow)
        .is_err());
    // A real sequence id is not a holder.
    assert!(m.assemble_table(5, &[holders[0], 9], allow).is_err());
    // A holder that is not complete (prefill never fenced).
    assert!(m.assemble_table(5, &[holders[0], pending], allow).is_err());
    // A holder listed twice would double its blocks.
    assert!(m
        .assemble_table(5, &[holders[0], holders[0]], allow)
        .is_err());
    // A partial block in the middle leaves a gap in positions.
    assert!(m
        .assemble_table(5, &[holders[2], holders[0]], allow)
        .is_err());
    // The suffix policy: a partial tail is refused under FullBlocksOnly.
    assert!(m
        .assemble_table(5, &holders, SuffixPolicy::FullBlocksOnly)
        .is_err());
    // Empty list and a holder id as the destination.
    assert!(m.assemble_table(5, &[], allow).is_err());
    assert!(m.assemble_table(unknown, &holders, allow).is_err());

    assert!(m.get_block_table(5).is_none());
    assert!(m.get_block_table(unknown).is_none());
    assert_eq!(m.free_block_count(), free);
    assert_eq!(m.debug_active_blocks(), active);
    let after: Vec<usize> = (0..16).map(|b| ref_count(&m, b)).collect();
    assert_eq!(after, counts_p);
    assert_ne!(counts, counts_p);
    for (h, bl) in holders.iter().zip(&blocks) {
        assert_eq!(&table_blocks(&m, *h), bl);
    }
    assert_eq!(m.get_block_table(9).unwrap().sequence_id, 9);
}
