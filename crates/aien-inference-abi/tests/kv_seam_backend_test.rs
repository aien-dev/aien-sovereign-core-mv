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

//! C6a (sc#484, ADR 0003 sections 6.2 and 6.3): the backend side of the
//! persistent-KV seam.
//!
//! - `computed_prefix` names only the tokens whose K/V is written in the pool
//!   (review finding 3): never the last token after a decode step, never the
//!   token prefill sampled but did not run.
//! - `restore_sequence` gives a sequence backend state for a prefix whose K/V a
//!   restore already put in its block table, with the same refusals as
//!   `fork_sequence`, so the next prefill chunk lines up at the restored
//!   position.
//!
//! Reference CPU backend with synthetic reference weights and a small paged
//! pool: no checkpoint, no GPU, runs in seconds.

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, ModelConfig, NativeTransformerBackend, ReferenceCpuBackend,
    SamplingParams, ScheduledBatch, SequenceRequest, TransformerWeights,
};
use aien_kv_cache::{
    create_shared_kv_manager, create_shared_kv_manager_with_pool, holder_id, KvDType, KvPoolConfig,
    SharedKvManager, SuffixPolicy,
};
use std::collections::HashMap;
use std::sync::Arc;

const BLOCK: usize = 16;
const POOL_BLOCKS: usize = 64;

fn config() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        block_size: BLOCK,
        ..Default::default()
    }
}

fn weights() -> TransformerWeights {
    TransformerWeights::reference_test_weights(&config())
}

fn shared_pool() -> SharedKvManager {
    let c = config();
    let pool = KvPoolConfig::for_model(
        POOL_BLOCKS,
        BLOCK,
        c.num_layers,
        c.num_kv_heads,
        c.head_dim,
        KvDType::Fp32,
    );
    create_shared_kv_manager_with_pool(POOL_BLOCKS, BLOCK, pool).expect("pool")
}

fn backend_on(kv: &SharedKvManager) -> NativeTransformerBackend {
    NativeTransformerBackend::with_shared_kv_and_backend(
        weights(),
        Arc::new(ReferenceCpuBackend::new()),
        kv.clone(),
    )
}

/// Deterministic prompt avoiding token ids 0..=2.
fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % 253 + 3) as u32).collect()
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 64,
        stop_token_ids: vec![],
    }
}

fn prefill_batch(id: u64, tokens: &[u32]) -> ScheduledBatch {
    ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: id,
            prompt_tokens: tokens.to_vec(),
            sampling_params: greedy(),
            arrival_time_ns: 0,
            priority: 1,
        }],
        decode_requests: Vec::new(),
        block_tables: HashMap::new(),
        step_id: 1,
    }
}

fn decode_batch(id: u64) -> ScheduledBatch {
    ScheduledBatch {
        prefill_requests: Vec::new(),
        decode_requests: vec![id],
        block_tables: HashMap::new(),
        step_id: 2,
    }
}

fn token_of(outputs: &[DecodeOutput], id: u64) -> u32 {
    outputs
        .iter()
        .find_map(|o| match o {
            DecodeOutput::Token {
                request_id,
                token_id,
                ..
            } if *request_id == id => Some(*token_id),
            _ => None,
        })
        .expect("a token for the request")
}

/// Sum of |K| and |V| at one position, layer 0, read straight from the pool.
fn kv_magnitude(kv: &SharedKvManager, seq: u64, pos: usize) -> f32 {
    let r = kv.read();
    let table = r.get_block_table(seq).expect("table");
    let pool = r.tensor_pool().expect("pool");
    let kv_dim = config().num_kv_heads * config().head_dim;
    let (mut k, mut v) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
    let (blk, slot) = (table.block_ids[pos / BLOCK], pos % BLOCK);
    pool.read_token_kv(blk, 0, slot, &mut k, &mut v);
    k.iter().chain(v.iter()).map(|x| x.abs()).sum()
}

#[tokio::test]
async fn computed_prefix_excludes_uncomputed_last_token() {
    let kv = shared_pool();
    let mut b = backend_on(&kv);
    let id = 0x1_0000_0001u64;
    let p = prompt(20);

    // After prefill: the sampled token is held apart, every prompt token has K/V.
    let (out, _) = b.execute_step(&prefill_batch(id, &p)).await.unwrap();
    let tok1 = token_of(&out, id);
    assert_eq!(b.sequences[&id].tokens.len(), 20);
    assert_eq!(
        b.computed_prefix(id).expect("prefilled sequence"),
        p,
        "prefill: computed = the prompt; the sampled token is not in it"
    );
    assert_eq!(kv.read().total_tokens(id), Some(20));
    assert!(kv_magnitude(&kv, id, 19) > 0.0, "position 19 holds K/V");

    // After one decode step: the token fed (tok1) now has K/V, the newly
    // sampled one (tok2) is the last entry of `tokens` and has none.
    let (out, _) = b.execute_step(&decode_batch(id)).await.unwrap();
    let tok2 = token_of(&out, id);
    assert_eq!(
        b.sequences[&id].tokens.len(),
        22,
        "the naive token list is one longer than the computed prefix"
    );
    let mut expected = p.clone();
    expected.push(tok1);
    let computed = b.computed_prefix(id).expect("decoding sequence");
    assert_eq!(computed, expected, "decode: the last token is excluded");
    assert_ne!(computed.last(), Some(&tok2));
    assert_eq!(kv.read().total_tokens(id), Some(21));
    assert!(
        kv_magnitude(&kv, id, 20) > 0.0,
        "the last computed position (20) is written"
    );
    // Position 21 is the uncomputed last token: its slot in the pool is zero.
    // The table stops at 21 tokens but the block has room, so read the raw slot.
    let blk = kv.read().get_block_table(id).unwrap().block_ids[21 / BLOCK];
    let mut k = vec![0.0f32; 32];
    let mut v = vec![0.0f32; 32];
    kv.read()
        .tensor_pool()
        .unwrap()
        .read_token_kv(blk, 0, 21 % BLOCK, &mut k, &mut v);
    assert!(
        k.iter().chain(v.iter()).all(|x| *x == 0.0),
        "position `computed` (21) is unwritten in the pool"
    );

    // A second decode step moves the boundary by exactly one.
    b.execute_step(&decode_batch(id)).await.unwrap();
    assert_eq!(b.computed_prefix(id).unwrap().len(), 22);
    assert_eq!(b.sequences[&id].tokens.len(), 23);
}

#[test]
fn computed_prefix_is_none_for_unknown_sequence_and_unpaged_backend() {
    let kv = shared_pool();
    let b = backend_on(&kv);
    assert_eq!(b.computed_prefix(0x1_0000_0009), None, "unknown sequence");

    // No shared pool: K/V lives in the backend's dense per-sequence cache, not
    // in blocks a holder could pin, so there is nothing to park.
    let mut dense = NativeTransformerBackend::new(weights());
    dense.prefill_sequence(0x1_0000_0001, &prompt(8)).unwrap();
    assert_eq!(dense.computed_prefix(0x1_0000_0001), None);
}

fn pending_table(kv: &SharedKvManager, seq: u64, tokens: &[u32]) {
    let mut w = kv.write();
    w.allocate_sequence(seq, tokens).unwrap();
    w.mark_prefill_pending(seq).unwrap();
}

#[test]
fn restore_sequence_refuses_like_fork_sequence() {
    let seq = 0x2_0000_0001u64;
    let p = prompt(32);

    // No KV manager at all.
    let mut no_kv = NativeTransformerBackend::new(weights());
    let e = no_kv.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("no shared KV manager"), "{e}");

    // A manager with no physical pool.
    let bare = create_shared_kv_manager(POOL_BLOCKS, BLOCK);
    let mut no_pool = backend_on(&bare);
    pending_table(&bare, seq, &p);
    let e = no_pool.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("no physical tensor pool"), "{e}");

    let kv = shared_pool();
    let mut b = backend_on(&kv);

    // No block table for the sequence.
    let e = b.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("no KV block table"), "{e}");

    // Table exists but was never marked pending: allocation is not restoration.
    kv.write().allocate_sequence(seq, &p).unwrap();
    let e = b.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("prefill state"), "{e}");

    // A ready table could enter decode: refused.
    kv.write().mark_prefill_pending(seq).unwrap();
    kv.write().complete_prefill(seq).unwrap();
    let e = b.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("prefill state"), "{e}");
    assert!(
        b.sequences.is_empty(),
        "every refusal leaves no backend state"
    );
}

#[test]
fn restore_sequence_refuses_wrong_token_counts_and_double_restore() {
    let kv = shared_pool();
    let mut b = backend_on(&kv);
    let seq = 0x2_0000_0001u64;
    let p = prompt(32);
    pending_table(&kv, seq, &p);

    let e = b.restore_sequence(seq, &[]).unwrap_err();
    assert!(e.contains("empty"), "{e}");
    let e = b.restore_sequence(seq, &p[..31]).unwrap_err();
    assert!(
        e.contains("covers 32"),
        "token count must equal the table: {e}"
    );
    assert!(b.sequences.is_empty());

    b.restore_sequence(seq, &p).expect("matching restore");
    assert_eq!(b.sequences[&seq].tokens, p);
    assert!(
        !b.pending_prefill_token.contains_key(&seq),
        "a restored prefix has no pending sampled token"
    );
    let e = b.restore_sequence(seq, &p).unwrap_err();
    assert!(e.contains("already has backend state"), "{e}");
    assert_eq!(
        b.sequences[&seq].tokens, p,
        "second attempt changed nothing"
    );
}

/// The A2 prerequisite at the backend: a prefix parked behind a holder and
/// restored into a new sequence continues exactly like an uninterrupted prefill.
/// `restored` is the parked prefix length; a value inside a block exercises the
/// partial tail, which the first suffix append copies on write.
async fn restore_and_continue(restored: usize, policy: SuffixPolicy) {
    let p = prompt(40);

    // Control: one backend, one pool, the whole prompt in one chunk.
    let control_kv = shared_pool();
    let mut control = backend_on(&control_kv);
    let cid = 0x3_0000_0001u64;
    let (out, _) = control.execute_step(&prefill_batch(cid, &p)).await.unwrap();
    let control_token = token_of(&out, cid);

    // Cold run of the first two blocks, then park behind a holder and free.
    let kv = shared_pool();
    let mut cold = backend_on(&kv);
    let pid = 0x4_0000_0001u64;
    cold.execute_step(&prefill_batch(pid, &p[..restored]))
        .await
        .unwrap();
    let holder = holder_id(1).unwrap();
    {
        let mut w = kv.write();
        w.mark_prefill_pending(pid).unwrap();
        w.complete_prefill(pid).unwrap();
        w.fork_range(pid, 0, restored.div_ceil(BLOCK), holder)
            .unwrap();
        w.free_sequence(pid).unwrap();
    }
    AienInferenceBackend::release_sequence(&mut cold, pid).unwrap();

    // Restore: assemble the request's own table from the holder, give the
    // backend the restored tokens, prefill only the suffix.
    let rid = 0x5_0000_0001u64;
    kv.write().assemble_table(rid, &[holder], policy).unwrap();
    let mut warm = backend_on(&kv);
    warm.restore_sequence(rid, &p[..restored]).unwrap();
    let (out, _) = warm
        .execute_step(&prefill_batch(rid, &p[restored..]))
        .await
        .unwrap();
    assert_eq!(
        token_of(&out, rid),
        control_token,
        "greedy token after restore equals the uninterrupted run"
    );
    assert_eq!(kv.read().total_tokens(rid), Some(40));
    if !restored.is_multiple_of(BLOCK) {
        assert!(
            kv.read().cow_faults() >= 1,
            "the shared partial tail was copied on write, not mutated in place"
        );
    }

    // K/V of every position matches the control within the chunking tolerance.
    for layer in 0..2 {
        let (rk, rv) = kv.read().gather_layer_kv(rid, layer).unwrap();
        let (ck, cv) = control_kv.read().gather_layer_kv(cid, layer).unwrap();
        assert_eq!(rk.len(), ck.len());
        let worst = rk
            .iter()
            .zip(&ck)
            .chain(rv.iter().zip(&cv))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-4, "layer {layer}: K/V differs by {worst}");
    }
}

#[tokio::test]
async fn restored_prefix_continues_like_uninterrupted_prefill() {
    restore_and_continue(32, SuffixPolicy::FullBlocksOnly).await;
}

#[tokio::test]
async fn restored_partial_tail_continues_like_uninterrupted_prefill() {
    restore_and_continue(20, SuffixPolicy::AllowPartialTail).await;
}
