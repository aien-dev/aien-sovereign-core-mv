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

use aien_kv_cache::{AienKvManager, KvDType, KvPoolConfig};

#[test]
fn test_prefix_refcount_32_branches() {
    let block_size = 16;
    let total_blocks = 5000;
    let pool_cfg = KvPoolConfig::for_tinyllama(total_blocks, block_size, KvDType::Fp32);
    let bytes_per_block = pool_cfg.bytes_per_block();

    let mut mgr = AienKvManager::with_tensor_pool(total_blocks, block_size, pool_cfg).unwrap();

    // 1 parent sequence with 512 tokens (exactly 32 blocks of 16 tokens)
    let prompt_tokens: Vec<u32> = (0..512).collect();
    let parent_blocks = mgr.allocate_sequence(1, &prompt_tokens).unwrap();
    assert_eq!(parent_blocks.len(), 32);

    let initial_available = mgr.available_blocks();
    assert_eq!(initial_available, total_blocks - 32);

    // Fork 32 branches from parent 1
    for child_id in 2..=33 {
        let child_blocks = mgr.fork_context(1, child_id).unwrap();
        assert_eq!(child_blocks.len(), 32);
        assert_eq!(child_blocks, parent_blocks);
    }

    // Zero new physical blocks allocated during fork
    assert_eq!(mgr.available_blocks(), initial_available);

    let m = mgr.metrics();
    assert_eq!(m.physical_pages, 32);
    assert_eq!(m.logical_pages, 33 * 32); // 1056 logical references
    assert_eq!(m.shared_pages, 32);
    assert_eq!(m.private_pages, 0);
    assert_eq!(m.cow_faults, 0);
    assert_eq!(m.bytes_saved_vs_full_copy, 32 * 32 * bytes_per_block);
}

#[test]
fn test_tail_divergence_cow() {
    let block_size = 16;
    let total_blocks = 2000;
    let pool_cfg = KvPoolConfig::for_tinyllama(total_blocks, block_size, KvDType::Fp32);
    let mut mgr = AienKvManager::with_tensor_pool(total_blocks, block_size, pool_cfg).unwrap();

    // Parent has 30 tokens: block 0 (16 full) + block 1 (14 partial)
    let prompt: Vec<u32> = (0..30).collect();
    let p_blocks = mgr.allocate_sequence(1, &prompt).unwrap();
    assert_eq!(p_blocks.len(), 2);

    // Fork 32 child branches
    for child_id in 2..=33 {
        mgr.fork_context(1, child_id).unwrap();
    }

    let m_before = mgr.metrics();
    assert_eq!(m_before.physical_pages, 2);
    assert_eq!(m_before.shared_pages, 2);
    assert_eq!(m_before.cow_faults, 0);

    // Each child appends a token into the partial tail block (slot 14)
    // Every child triggers Copy-on-Write for block 1
    for child_id in 2..=33 {
        let (block_id, slot) = mgr.append_token_with_slot(child_id).unwrap();
        assert_eq!(slot, 14);
        assert_ne!(block_id, p_blocks[1]);
    }

    let m_after = mgr.metrics();
    // 32 COW faults triggered (one per child branch)
    assert_eq!(m_after.cow_faults, 32);
    // Block 0 is still shared by all 33 sequences (parent + 32 children)
    assert_eq!(m_after.shared_pages, 1);
    // 32 private blocks for children + 1 private block for parent (refcount dropped to 1)
    assert_eq!(m_after.private_pages, 33);
    // Total physical pages: 1 shared block + 33 private blocks = 34
    assert_eq!(m_after.physical_pages, 34);
    // Logical pages: 33 sequences * 2 blocks = 66
    assert_eq!(m_after.logical_pages, 66);
}

#[test]
fn test_parent_drop_before_children() {
    let block_size = 16;
    let total_blocks = 500;
    let mut mgr = AienKvManager::new(total_blocks, block_size);

    let prompt: Vec<u32> = (0..48).collect(); // 3 full blocks
    mgr.allocate_sequence(1, &prompt).unwrap();

    // Fork 10 children
    for i in 2..=11 {
        mgr.fork_context(1, i).unwrap();
    }

    assert_eq!(mgr.available_blocks(), total_blocks - 3);
    assert_eq!(mgr.metrics().shared_pages, 3);

    // Drop parent sequence 1
    let _ = mgr.free_sequence(1);

    // Blocks must remain allocated and shared among the 10 children
    assert_eq!(mgr.available_blocks(), total_blocks - 3);
    assert_eq!(mgr.metrics().shared_pages, 3);
    assert_eq!(mgr.active_sequence_count(), 10);

    // Children can still append tokens without failure
    for i in 2..=11 {
        mgr.append_token(i).unwrap();
    }
}

#[test]
fn test_reclamation_zero_leak() {
    let block_size = 16;
    let total_blocks = 500;
    let mut mgr = AienKvManager::new(total_blocks, block_size);

    let prompt: Vec<u32> = (0..64).collect(); // 4 blocks
    mgr.allocate_sequence(100, &prompt).unwrap();

    for i in 1..=32 {
        mgr.fork_context(100, 100 + i).unwrap();
        // Append tokens to create private blocks
        mgr.append_token(100 + i).unwrap();
    }

    // Release all 32 branches
    for i in 1..=32 {
        let _ = mgr.release_branch(100 + i);
    }

    // Only parent remains
    assert_eq!(mgr.active_sequence_count(), 1);
    let m = mgr.metrics();
    assert_eq!(m.physical_pages, 4);
    assert_eq!(m.private_pages, 4);
    assert_eq!(m.shared_pages, 0);

    // Release parent
    let _ = mgr.release_branch(100);
    assert_eq!(mgr.active_sequence_count(), 0);
    assert_eq!(mgr.available_blocks(), total_blocks);
    let m_final = mgr.metrics();
    assert_eq!(m_final.physical_pages, 0);
    assert_eq!(m_final.used_blocks, 0);
}

#[test]
fn test_tensor_pool_read_write_cow_parity() {
    let block_size = 16;
    let total_blocks = 100;
    let pool_cfg = KvPoolConfig::for_tinyllama(total_blocks, block_size, KvDType::Fp32);
    let kv_dim = pool_cfg.num_kv_heads * pool_cfg.head_dim; // 4 * 64 = 256
    let mut mgr = AienKvManager::with_tensor_pool(total_blocks, block_size, pool_cfg).unwrap();

    // Allocate parent with 15 tokens (1 partial block)
    let prompt: Vec<u32> = (0..15).collect();
    let p_blocks = mgr.allocate_sequence(1, &prompt).unwrap();
    let p_block = p_blocks[0];

    // Write distinctive FP32 patterns into all 15 prompt tokens across layer 0
    let mut expected_k = vec![0.0f32; 15 * kv_dim];
    let mut expected_v = vec![0.0f32; 15 * kv_dim];

    for t in 0..15 {
        let k_val = (t as f32 + 1.0) * 10.0;
        let v_val = (t as f32 + 1.0) * 20.0;
        let k_tok = vec![k_val; kv_dim];
        let v_tok = vec![v_val; kv_dim];

        mgr.write_explicit_token_kv(p_block, 0, t, &k_tok, &v_tok)
            .unwrap();
        expected_k[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&k_tok);
        expected_v[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&v_tok);
    }

    // Fork child 2
    mgr.fork_context(1, 2).unwrap();

    // Verify child reads exact same initial 15 tokens
    let mut child_k = Vec::new();
    let mut child_v = Vec::new();
    mgr.gather_sequence_layer_kv(2, 0, &mut child_k, &mut child_v)
        .unwrap();
    assert_eq!(child_k, expected_k);
    assert_eq!(child_v, expected_v);

    // Child appends 16th token (slot 15 in partial block) -> triggers Copy-on-Write
    let (child_block, slot) = mgr.append_token_with_slot(2).unwrap();
    assert_eq!(slot, 15);
    assert_ne!(child_block, p_block);

    // Write divergent token to child
    let child_k15 = vec![999.0f32; kv_dim];
    let child_v15 = vec![888.0f32; kv_dim];
    mgr.write_explicit_token_kv(child_block, 0, slot, &child_k15, &child_v15)
        .unwrap();

    // Verify parent KV remains 100% untouched
    let mut parent_k = Vec::new();
    let mut parent_v = Vec::new();
    mgr.gather_sequence_layer_kv(1, 0, &mut parent_k, &mut parent_v)
        .unwrap();
    assert_eq!(parent_k.len(), 15 * kv_dim);
    assert_eq!(parent_k, expected_k);
    assert_eq!(parent_v, expected_v);

    // Verify child KV has all 15 prefix tokens + divergent 16th token
    mgr.gather_sequence_layer_kv(2, 0, &mut child_k, &mut child_v)
        .unwrap();
    assert_eq!(child_k.len(), 16 * kv_dim);
    assert_eq!(&child_k[0..15 * kv_dim], &expected_k[..]);
    assert_eq!(&child_k[15 * kv_dim..16 * kv_dim], &child_k15[..]);
    assert_eq!(&child_v[15 * kv_dim..16 * kv_dim], &child_v15[..]);
}
