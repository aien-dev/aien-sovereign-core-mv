//! KV lifecycle qualification (overnight CAND-0 campaign, lane L6-KV).
//!
//! Host-only checks on `AienKvManager` with a heap pool, no GPU, no model:
//!
//! - `freed_block_reuse_never_exposes_previous_sequence_kv`: a block freed by
//!   one sequence and handed to another (fresh allocation, append into a new
//!   block, copy-on-write destination) reads back zeros or the new owner's
//!   own values in every slot of every layer, never the old owner's; the
//!   freed sequence id is a stale identity that can neither read nor write.
//! - `sibling_appends_do_not_alter_parent_or_siblings`: copy-on-write isolation
//!   between a parent and two forked children that all keep appending.
//! - `exhaustion_refuses_without_changing_state`: allocation, append,
//!   copy-on-write and reservation on an empty pool all return `Err` and leave
//!   every observable counter unchanged, including `cow_faults` (a refused
//!   copy-on-write is not a fault that happened; the campaign verifier's
//!   conservation model counts a fault only when the copy took a block).
//! - `device_cow_copy_failure_releases_the_new_block`: when the device copy of
//!   a copy-on-write fails, the block taken for it goes back to the pool.
//! - `fixed_seed_random_lifecycle_keeps_accounting_consistent`: a fixed-seed
//!   random program of admit (allocate + prefill fence), gated fork, append,
//!   cancel, finish and reserve/rollback, checked after every step against a
//!   reference model: ownership, conservation, content, admission prediction
//!   (`blocks_needed_for_appends` equals the blocks an append really took),
//!   copy-on-write count, atomic refusal on exhaustion, stale ids refused.
//!
//! The randomized property test over the same manager without the fixed seed
//! lives in crates/aien-replay/tests/kv_ownership_props.rs.

use std::collections::BTreeMap;

use aien_kv_cache::{
    AienKvManager, BufferRegion, HeapUnifiedBuffer, KvDType, KvPoolConfig, MemoryDevice,
    PrefillState,
};
use aien_platform::{Fence, PlatformError};

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

/// K value of token `val` in `layer`; V is its negation.
fn kval(val: f32, layer: usize) -> f32 {
    val + layer as f32 * 0.25
}

fn write_all_layers(m: &mut Mgr, block: usize, slot: usize, val: f32) {
    for layer in 0..LAYERS {
        let k = [kval(val, layer); KV_DIM];
        let v = [-kval(val, layer); KV_DIM];
        m.write_explicit_token_kv(block, layer, slot, &k, &v)
            .expect("pool attached");
    }
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

/// Prefill a fresh sequence through the gate: allocate, write every prompt
/// token in every layer, mark pending, fire the completion fence.
fn admit(m: &mut Mgr, seq: u64, vals: &[f32]) -> Result<(), String> {
    let prompt: Vec<u32> = (0..vals.len() as u32).collect();
    m.allocate_sequence(seq, &prompt)?;
    for (pos, &val) in vals.iter().enumerate() {
        for layer in 0..LAYERS {
            m.write_token_kv(
                seq,
                pos,
                layer,
                &[kval(val, layer); KV_DIM],
                &[-kval(val, layer); KV_DIM],
            )?;
        }
    }
    m.mark_prefill_pending(seq).map_err(|e| e.to_string())?;
    m.complete_prefill(seq).map_err(|e| e.to_string())?;
    Ok(())
}

fn append_val(m: &mut Mgr, seq: u64, val: f32) -> Result<(), String> {
    let (block, slot) = m.append_token_with_slot(seq)?;
    write_all_layers(m, block, slot, val);
    Ok(())
}

/// Check that `seq` reads back exactly `vals` in every layer.
fn assert_content(m: &Mgr, seq: u64, vals: &[f32], what: &str) {
    for layer in 0..LAYERS {
        let (k, v) = m
            .gather_layer_kv(seq, layer)
            .unwrap_or_else(|e| panic!("{what}: gather seq {seq} layer {layer}: {e}"));
        let want_k: Vec<f32> = vals
            .iter()
            .flat_map(|&x| [kval(x, layer); KV_DIM])
            .collect();
        let want_v: Vec<f32> = vals
            .iter()
            .flat_map(|&x| [-kval(x, layer); KV_DIM])
            .collect();
        assert_eq!(k, want_k, "{what}: seq {seq} layer {layer} K");
        assert_eq!(v, want_v, "{what}: seq {seq} layer {layer} V");
    }
}

/// Everything a caller can observe about the manager, for atomicity checks.
#[derive(Debug, PartialEq)]
struct Snap {
    free: usize,
    allocated: usize,
    tables: Vec<(u64, Vec<usize>, usize, PrefillState)>,
    blocks: Vec<(usize, usize, bool)>,
    cow_faults: usize,
    metrics: (usize, usize, usize, usize, usize, usize),
}

fn snap(m: &Mgr, seqs: impl IntoIterator<Item = u64>) -> Snap {
    let mut tables = Vec::new();
    for seq in seqs {
        if let Some(t) = m.get_block_table(seq) {
            tables.push((seq, t.block_ids.clone(), t.total_tokens, t.prefill_state));
        }
    }
    let blocks = (0..m.total_block_count())
        .map(|b| {
            let blk = m.get_block(b).expect("block");
            (blk.ref_count, blk.num_tokens, blk.is_shared)
        })
        .collect();
    let km = m.metrics();
    Snap {
        free: m.free_block_count(),
        allocated: m.allocated_block_count(),
        tables,
        blocks,
        cow_faults: m.cow_faults(),
        metrics: (
            km.physical_pages,
            km.logical_pages,
            km.shared_pages,
            km.private_pages,
            km.cow_faults,
            km.used_blocks,
        ),
    }
}

#[test]
fn freed_block_reuse_never_exposes_previous_sequence_kv() {
    // The whole pool (4 blocks) belongs to sequence A first, every slot written.
    let mut m = new_mgr(4);
    let a_vals: Vec<f32> = (0..16).map(|i| 1000.0 + i as f32).collect();
    admit(&mut m, 1, &a_vals).unwrap();
    let a_blocks = m.get_block_table(1).unwrap().block_ids.clone();
    assert_eq!(a_blocks.len(), 4);
    for &b in &a_blocks {
        assert!(raw_block(&m, b).iter().all(|&x| x.abs() >= 1000.0));
    }
    m.free_sequence(1).unwrap();
    assert_eq!(m.free_block_count(), 4);

    // Stale identity: the freed id can neither read nor write nor append.
    assert!(m.get_block_table(1).is_none());
    assert!(
        m.gather_layer_kv(1, 0).is_err(),
        "stale id read must be refused"
    );
    assert!(
        m.write_token_kv(1, 0, 0, &[7.0; KV_DIM], &[7.0; KV_DIM])
            .is_err(),
        "stale id write must be refused"
    );
    assert!(
        m.append_token_with_slot(1).is_err(),
        "stale id append must be refused"
    );
    assert_eq!(m.prefill_state(1), None);
    assert_eq!(
        m.free_block_count(),
        4,
        "stale-id calls must not take blocks"
    );

    // Fresh allocation on recycled blocks: every slot of every layer is zero
    // before the new owner writes anything.
    let prompt: Vec<u32> = (0..6).collect();
    let b_blocks = m.allocate_sequence(2, &prompt).unwrap();
    assert_eq!(b_blocks.len(), 2);
    for &b in &b_blocks {
        assert!(
            a_blocks.contains(&b),
            "test precondition: block {b} is recycled"
        );
        assert!(
            raw_block(&m, b).iter().all(|&x| x == 0.0),
            "STALE_KV_LEAK: recycled block {b} still holds the previous owner's KV at allocation"
        );
    }
    for layer in 0..LAYERS {
        let (k, v) = m.gather_layer_kv(2, layer).unwrap();
        assert_eq!(k, vec![0.0; 6 * KV_DIM]);
        assert_eq!(v, vec![0.0; 6 * KV_DIM]);
    }

    // B prefills its own values, then appends across a block boundary into a
    // third recycled block: zero before B writes it.
    let b_vals: Vec<f32> = (0..6).map(|i| 2000.0 + i as f32).collect();
    for (pos, &val) in b_vals.iter().enumerate() {
        for layer in 0..LAYERS {
            m.write_token_kv(
                2,
                pos,
                layer,
                &[kval(val, layer); KV_DIM],
                &[-kval(val, layer); KV_DIM],
            )
            .unwrap();
        }
    }
    m.mark_prefill_pending(2).unwrap();
    m.complete_prefill(2).unwrap();
    let mut b_all = b_vals.clone();
    for i in 0..3 {
        let (block, slot) = m.append_token_with_slot(2).unwrap();
        if slot == 0 {
            assert!(a_blocks.contains(&block));
            assert!(
                raw_block(&m, block).iter().all(|&x| x == 0.0),
                "STALE_KV_LEAK: block {block} taken by append holds old KV"
            );
        }
        let val = 2100.0 + i as f32;
        write_all_layers(&mut m, block, slot, val);
        b_all.push(val);
    }
    assert_content(&m, 2, &b_all, "B after appends");

    // Copy-on-write into the last recycled block: the copy holds B's tokens in
    // the used slots and zeros in the rest, never A's values.
    m.fork_prefilled(2, 3).unwrap();
    let (cow_block, cow_slot) = m.append_token_with_slot(3).unwrap();
    assert!(a_blocks.contains(&cow_block));
    assert_eq!(cow_slot, b_all.len() % BS);
    let raw = raw_block(&m, cow_block);
    assert!(
        raw.iter()
            .all(|&x| x == 0.0 || (x.abs() >= 2000.0 && x.abs() < 3000.0)),
        "STALE_KV_LEAK: copy-on-write destination {cow_block} exposes previous-owner KV: {raw:?}"
    );
    write_all_layers(&mut m, cow_block, cow_slot, 3000.0);
    let mut c_all = b_all.clone();
    c_all.push(3000.0);
    assert_content(&m, 3, &c_all, "C after copy-on-write");
    assert_content(&m, 2, &b_all, "B after C's copy-on-write");

    m.free_sequence(2).unwrap();
    m.free_sequence(3).unwrap();
    assert_eq!(m.free_block_count(), 4);
}

#[test]
fn sibling_appends_do_not_alter_parent_or_siblings() {
    let mut m = new_mgr(32);
    let parent: Vec<f32> = (0..6).map(|i| 10.0 + i as f32).collect(); // ends mid-block
    admit(&mut m, 1, &parent).unwrap();
    m.fork_prefilled(1, 2).unwrap();
    m.fork_prefilled(1, 3).unwrap();
    assert_eq!(m.prefill_state(1), Some(PrefillState::SharedFrozen));
    assert_eq!(m.prefill_state(2), Some(PrefillState::SharedFrozen));
    // children inherit the computed prefix
    assert_content(&m, 2, &parent, "child 2 inherits prefix");
    assert_content(&m, 3, &parent, "child 3 inherits prefix");

    let mut c2 = parent.clone();
    let mut c3 = parent.clone();
    let mut p = parent.clone();
    for i in 0..7 {
        let v = 200.0 + i as f32;
        append_val(&mut m, 2, v).unwrap();
        c2.push(v);
        assert_content(&m, 1, &p, "parent while child 2 appends");
        assert_content(&m, 3, &c3, "child 3 while child 2 appends");
    }
    for i in 0..3 {
        let v = 300.0 + i as f32;
        append_val(&mut m, 3, v).unwrap();
        c3.push(v);
    }
    for i in 0..5 {
        let v = 100.0 + i as f32;
        append_val(&mut m, 1, v).unwrap();
        p.push(v);
    }
    assert_content(&m, 1, &p, "parent after all appends");
    assert_content(&m, 2, &c2, "child 2 after all appends");
    assert_content(&m, 3, &c3, "child 3 after all appends");
    // the full prefix block is still shared by all three
    let first = m.get_block_table(1).unwrap().block_ids[0];
    assert_eq!(m.get_block(first).unwrap().ref_count, 3);
    for s in 1..=3 {
        m.free_sequence(s).unwrap();
    }
    assert_eq!(m.free_block_count(), 32);
}

#[test]
fn exhaustion_refuses_without_changing_state() {
    // 3 blocks: root (6 tokens, 2 blocks, tail partial) forked to a child,
    // and a third sequence holding the last block, full.
    let mut m = new_mgr(3);
    admit(&mut m, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    m.fork_prefilled(1, 2).unwrap();
    admit(&mut m, 3, &[7.0, 8.0, 9.0, 10.0]).unwrap();
    assert_eq!(m.free_block_count(), 0);
    let seqs = [1u64, 2, 3, 4];
    let before = snap(&m, seqs);

    assert!(
        m.allocate_sequence(4, &[0; 4]).is_err(),
        "allocation on an empty pool"
    );
    assert_eq!(snap(&m, seqs), before, "refused allocation changed state");

    assert_eq!(m.blocks_needed_for_appends(3, 1), 1);
    assert!(
        m.append_token_with_slot(3).is_err(),
        "append past a full private tail"
    );
    assert_eq!(
        snap(&m, seqs),
        before,
        "refused fresh-block append changed state"
    );

    assert_eq!(m.blocks_needed_for_appends(2, 1), 1);
    assert!(
        m.append_token_with_slot(2).is_err(),
        "copy-on-write on an empty pool"
    );
    assert_eq!(
        snap(&m, seqs),
        before,
        "COW_COUNTER_VIOLATION: refused copy-on-write changed state (cow_faults counts a fault that never happened)"
    );

    assert!(m.reserve_token(2).is_err(), "reservation on an empty pool");
    assert_eq!(snap(&m, seqs), before, "refused reservation changed state");

    // content survived every refusal
    assert_content(
        &m,
        1,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "root after refusals",
    );
    assert_content(
        &m,
        2,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "child after refusals",
    );
    assert_content(&m, 3, &[7.0, 8.0, 9.0, 10.0], "third after refusals");

    // once a block is free the same copy-on-write succeeds and counts once
    m.free_sequence(3).unwrap();
    append_val(&mut m, 2, 99.0).unwrap();
    assert_eq!(m.cow_faults(), 1, "exactly one copy-on-write happened");
    assert_eq!(m.metrics().cow_faults, 1);
    assert_content(
        &m,
        2,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0],
        "child after retry",
    );
    assert_content(
        &m,
        1,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "root after child retry",
    );
}

/// A memory device whose every operation fails.
struct FailingDevice;

impl MemoryDevice for FailingDevice {
    fn copy(&self, _src: BufferRegion, _dst: BufferRegion) -> Result<Fence, PlatformError> {
        Err(PlatformError::DeviceUnavailable)
    }
    fn zero(&self, _dst: BufferRegion) -> Result<Fence, PlatformError> {
        Err(PlatformError::DeviceUnavailable)
    }
}

#[test]
fn device_cow_copy_failure_releases_the_new_block() {
    let mut m = new_mgr(4);
    admit(&mut m, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    m.fork_prefilled(1, 2).unwrap();
    let seqs = [1u64, 2];
    let before = snap(&m, seqs);
    assert!(
        m.append_token_with_slot_device(&FailingDevice, 2).is_err(),
        "a failed device copy must fail the append"
    );
    assert_eq!(
        snap(&m, seqs),
        before,
        "COW_DEVICE_LEAK: failed device copy changed state (block taken and never returned, or cow_faults bumped)"
    );
    m.free_sequence(1).unwrap();
    m.free_sequence(2).unwrap();
    assert_eq!(m.free_block_count(), 4, "COW_DEVICE_LEAK: block leaked");
}

// ---------------------------------------------------------------------------
// Fixed-seed random lifecycle
// ---------------------------------------------------------------------------

const RAND_BLOCKS: usize = 12;
const RAND_IDS: u64 = 8;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Reference model: each live sequence's token values.
type Model = BTreeMap<u64, Vec<f32>>;

fn check_invariants(m: &Mgr, model: &Model, cow_model: usize, ctx: &str) {
    assert_eq!(
        m.active_sequence_count(),
        model.len(),
        "{ctx}: live sequences"
    );
    let mut holders = [0usize; RAND_BLOCKS];
    let mut logical = 0usize;
    for (&seq, vals) in model {
        let t = m
            .get_block_table(seq)
            .unwrap_or_else(|| panic!("{ctx}: seq {seq} has no table"));
        assert!(t.is_prefill_ready(), "{ctx}: admitted seq {seq} not ready");
        assert_eq!(t.total_tokens, vals.len(), "{ctx}: seq {seq} tokens");
        assert_eq!(
            t.block_ids.len(),
            vals.len().div_ceil(BS),
            "{ctx}: seq {seq} blocks"
        );
        let mut in_blocks = 0;
        for (i, &b) in t.block_ids.iter().enumerate() {
            holders[b] += 1;
            let n = m.get_block(b).unwrap().num_tokens;
            if i + 1 < t.block_ids.len() {
                assert_eq!(n, BS, "{ctx}: seq {seq} non-tail block {b} not full");
            }
            in_blocks += n;
        }
        assert_eq!(in_blocks, vals.len(), "{ctx}: seq {seq} tokens in blocks");
        logical += t.block_ids.len();
        assert_content(m, seq, vals, ctx);
    }
    let mut referenced = 0;
    for (b, &h) in holders.iter().enumerate() {
        let blk = m.get_block(b).unwrap();
        assert_eq!(blk.ref_count, h, "{ctx}: block {b} ref_count vs holders");
        if h > 0 {
            referenced += 1;
            assert_eq!(blk.is_shared, h > 1, "{ctx}: block {b} is_shared");
        } else {
            assert_eq!(
                blk.num_tokens, 0,
                "{ctx}: free block {b} keeps a token count"
            );
        }
    }
    assert_eq!(
        m.free_block_count() + referenced,
        RAND_BLOCKS,
        "{ctx}: free + referenced != pool (leak or double free)"
    );
    assert_eq!(
        m.allocated_block_count(),
        referenced,
        "{ctx}: allocated count"
    );
    let km = m.metrics();
    assert_eq!(km.physical_pages, referenced, "{ctx}: metrics physical");
    assert_eq!(km.logical_pages, logical, "{ctx}: metrics logical");
    assert_eq!(
        km.shared_pages + km.private_pages,
        referenced,
        "{ctx}: shared + private"
    );
    assert_eq!(
        m.cow_faults(),
        cow_model,
        "{ctx}: cow_faults vs successful copy-on-writes"
    );
}

fn run_program(seed: u64, steps: usize) -> (usize, usize, usize) {
    let mut rng = Lcg(seed);
    let mut m = new_mgr(RAND_BLOCKS);
    let mut model = Model::new();
    let mut retired: Vec<u64> = Vec::new();
    let mut counter = 0u32;
    let mut cow_model = 0usize;
    let (mut refusals, mut cows, mut forks) = (0usize, 0usize, 0usize);
    let all_ids = 0..RAND_IDS;
    for step in 0..steps {
        let op = rng.below(100);
        let seq = rng.below(RAND_IDS);
        let ctx = format!("seed {seed:#x} step {step} op {op} seq {seq}");
        let before = snap(&m, all_ids.clone());
        match op {
            // admit a new sequence (allocate + prefill fence)
            0..=14 => {
                if model.contains_key(&seq) {
                    continue;
                }
                let n = 1 + rng.below(9) as usize;
                let need = m.incremental_blocks_needed(seq, n);
                assert_eq!(need, n.div_ceil(BS), "{ctx}: admission need for a new id");
                let vals: Vec<f32> = (0..n)
                    .map(|_| {
                        counter += 1;
                        counter as f32
                    })
                    .collect();
                match admit(&mut m, seq, &vals) {
                    Ok(()) => {
                        assert!(need <= before.free, "{ctx}: admitted beyond free blocks");
                        model.insert(seq, vals);
                    }
                    Err(_) => {
                        assert!(need > before.free, "{ctx}: refused admission with room");
                        assert_eq!(
                            snap(&m, all_ids.clone()),
                            before,
                            "{ctx}: refused admit changed state"
                        );
                        refusals += 1;
                    }
                }
            }
            // gated fork onto a fresh or live child id
            15..=34 => {
                let child = rng.below(RAND_IDS);
                if child == seq {
                    continue;
                }
                match m.fork_prefilled(seq, child) {
                    Ok(_) => {
                        let vals = model
                            .get(&seq)
                            .unwrap_or_else(|| panic!("{ctx}: fork of missing parent succeeded"))
                            .clone();
                        model.insert(child, vals);
                        forks += 1;
                    }
                    Err(_) => {
                        assert!(
                            !model.contains_key(&seq),
                            "{ctx}: fork of live ready parent refused"
                        );
                        assert_eq!(
                            snap(&m, all_ids.clone()),
                            before,
                            "{ctx}: refused fork changed state"
                        );
                    }
                }
            }
            // append (decode one token)
            35..=74 => {
                if !model.contains_key(&seq) {
                    assert!(
                        m.append_token_with_slot(seq).is_err(),
                        "{ctx}: append to dead id"
                    );
                    assert_eq!(
                        snap(&m, all_ids.clone()),
                        before,
                        "{ctx}: dead-id append changed state"
                    );
                    continue;
                }
                let predicted = m.blocks_needed_for_appends(seq, 1);
                let tail = *m.get_block_table(seq).unwrap().block_ids.last().unwrap();
                let tb = m.get_block(tail).unwrap();
                let will_cow = tb.is_shared && tb.num_tokens < BS;
                counter += 1;
                let val = counter as f32;
                match append_val(&mut m, seq, val) {
                    Ok(()) => {
                        assert_eq!(
                            before.free - m.free_block_count(),
                            predicted,
                            "{ctx}: blocks_needed_for_appends disagrees with the real append"
                        );
                        if will_cow {
                            cow_model += 1;
                            cows += 1;
                        }
                        model.get_mut(&seq).unwrap().push(val);
                    }
                    Err(_) => {
                        assert!(predicted > before.free, "{ctx}: append refused with room");
                        assert_eq!(
                            snap(&m, all_ids.clone()),
                            before,
                            "{ctx}: refused append changed state"
                        );
                        refusals += 1;
                    }
                }
            }
            // cancel or finish: both release the sequence
            75..=89 => {
                let live = model.remove(&seq).is_some();
                assert_eq!(m.free_sequence(seq).is_ok(), live, "{ctx}: free result");
                if live {
                    retired.push(seq);
                    // the released id is stale: no table, no read, no write
                    assert!(
                        m.get_block_table(seq).is_none(),
                        "{ctx}: freed table survives"
                    );
                    assert!(
                        m.gather_layer_kv(seq, 0).is_err(),
                        "{ctx}: stale read allowed"
                    );
                    assert!(
                        m.write_token_kv(seq, 0, 0, &[1.0; KV_DIM], &[1.0; KV_DIM])
                            .is_err(),
                        "{ctx}: stale write allowed"
                    );
                }
            }
            // reserve then roll back: no observable change
            90..=95 => {
                if let Ok(r) = m.reserve_token(seq) {
                    assert!(model.contains_key(&seq), "{ctx}: reserve on dead id");
                    m.rollback(r).unwrap();
                }
                assert_eq!(
                    snap(&m, all_ids.clone()),
                    before,
                    "{ctx}: reserve+rollback changed state"
                );
            }
            // reserve then commit: same as an append
            _ => {
                if !model.contains_key(&seq) {
                    continue;
                }
                let predicted = m.blocks_needed_for_appends(seq, 1);
                let tail = *m.get_block_table(seq).unwrap().block_ids.last().unwrap();
                let tb = m.get_block(tail).unwrap();
                let will_cow = tb.is_shared && tb.num_tokens < BS;
                match m.reserve_token(seq) {
                    Ok(r) => {
                        let (block, slot) = (r.block_id, r.slot);
                        m.commit(r).unwrap();
                        assert_eq!(
                            before.free - m.free_block_count(),
                            predicted,
                            "{ctx}: reserve need"
                        );
                        if will_cow {
                            cow_model += 1;
                            cows += 1;
                        }
                        counter += 1;
                        let val = counter as f32;
                        write_all_layers(&mut m, block, slot, val);
                        model.get_mut(&seq).unwrap().push(val);
                    }
                    Err(_) => {
                        assert!(predicted > before.free, "{ctx}: reserve refused with room");
                        assert_eq!(
                            snap(&m, all_ids.clone()),
                            before,
                            "{ctx}: refused reserve changed state"
                        );
                        refusals += 1;
                    }
                }
            }
        }
        check_invariants(&m, &model, cow_model, &ctx);
    }
    // drain: finishing everything returns the whole pool
    let live: Vec<u64> = model.keys().copied().collect();
    for seq in live {
        m.free_sequence(seq).unwrap();
        model.remove(&seq);
        check_invariants(
            &m,
            &model,
            cow_model,
            &format!("seed {seed:#x} drain {seq}"),
        );
    }
    assert_eq!(m.free_block_count(), RAND_BLOCKS);
    assert!(!retired.is_empty());
    (refusals, cows, forks)
}

#[test]
fn fixed_seed_random_lifecycle_keeps_accounting_consistent() {
    let mut totals = (0usize, 0usize, 0usize);
    for seed in [0x4c36_4b56u64, 1, 2, 3, 0xdead_beef] {
        let (r, c, f) = run_program(seed, 600);
        totals = (totals.0 + r, totals.1 + c, totals.2 + f);
    }
    // the program must actually reach exhaustion, copy-on-write and forks,
    // or the checks above proved nothing about them
    assert!(totals.0 > 0, "no exhaustion refusal was exercised");
    assert!(totals.1 > 0, "no copy-on-write was exercised");
    assert!(totals.2 > 0, "no fork was exercised");
    println!(
        "fixed-seed lifecycle: {} refusals, {} copy-on-writes, {} forks over 5 x 600 steps",
        totals.0, totals.1, totals.2
    );
}
