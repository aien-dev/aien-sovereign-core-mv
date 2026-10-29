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

//! GB10-2: `incremental_blocks_needed` is the one authority for new blocks.
//! Every prediction is checked against what the real append path allocates.

use aien_kv_cache::AienKvManager;

const BS: usize = 16;

fn prefilled(mgr: &mut AienKvManager, id: u64, tokens: usize) {
    let prompt: Vec<u32> = (0..tokens as u32).collect();
    mgr.allocate_sequence(id, &prompt).unwrap();
    mgr.mark_prefill_pending(id).unwrap();
    mgr.complete_prefill(id).unwrap();
}

/// Appends `n` tokens and returns how many free blocks that consumed.
fn consumed_by_appends(mgr: &mut AienKvManager, id: u64, n: usize) -> usize {
    let before = mgr.available_blocks();
    for _ in 0..n {
        mgr.append_token(id).unwrap();
    }
    before - mgr.available_blocks()
}

#[test]
fn no_table_needs_a_fresh_allocation_at_the_pool_block_size() {
    let mgr = AienKvManager::new(64, 32);
    assert_eq!(mgr.incremental_blocks_needed(1, 40), 2);
    assert_eq!(mgr.incremental_blocks_needed(1, 64), 2);
    assert_eq!(mgr.incremental_blocks_needed(1, 65), 3);
    assert_eq!(mgr.incremental_blocks_needed(1, 0), 0);
}

#[test]
fn fork_child_of_a_partial_prefix_needs_one_copy_on_write_block() {
    let mut mgr = AienKvManager::new(64, BS);
    prefilled(&mut mgr, 1, 40); // 3 blocks, last holds 8 tokens
    mgr.fork_prefilled(1, 2).unwrap();
    assert_eq!(mgr.incremental_blocks_needed(2, 40), 1);
    assert_eq!(
        consumed_by_appends(&mut mgr, 2, 1),
        1,
        "prediction == reality"
    );
}

#[test]
fn fork_child_of_a_full_prefix_needs_one_fresh_block() {
    let mut mgr = AienKvManager::new(64, BS);
    prefilled(&mut mgr, 1, 48); // exactly 3 full blocks
    mgr.fork_prefilled(1, 2).unwrap();
    assert_eq!(mgr.incremental_blocks_needed(2, 48), 1);
    assert_eq!(consumed_by_appends(&mut mgr, 2, 1), 1);
}

#[test]
fn private_partial_tail_absorbs_the_first_token() {
    let mut mgr = AienKvManager::new(64, BS);
    prefilled(&mut mgr, 1, 40); // private tail with 8 of 16 slots used
    assert_eq!(mgr.incremental_blocks_needed(1, 40), 0);
    assert_eq!(consumed_by_appends(&mut mgr, 1, 1), 0);
}

#[test]
fn prediction_matches_real_appends_across_shapes() {
    for prompt in [1usize, 15, 16, 17, 40, 47, 48, 100] {
        for appends in [1usize, 2, 8, 9, 16, 17, 40] {
            for fork in [false, true] {
                let mut mgr = AienKvManager::new(256, BS);
                prefilled(&mut mgr, 1, prompt);
                let id = if fork {
                    mgr.fork_prefilled(1, 2).unwrap();
                    2
                } else {
                    1
                };
                let predicted = mgr.blocks_needed_for_appends(id, appends);
                let actual = consumed_by_appends(&mut mgr, id, appends);
                assert_eq!(
                    predicted, actual,
                    "prompt {prompt}, appends {appends}, fork {fork}"
                );
            }
        }
    }
}

#[test]
fn unready_table_counts_a_fresh_allocation_minus_blocks_the_drop_returns() {
    let mut mgr = AienKvManager::new(64, BS);
    let prompt: Vec<u32> = (0..40).collect();
    mgr.allocate_sequence(1, &prompt).unwrap(); // Allocated, not computed
    assert_eq!(
        mgr.incremental_blocks_needed(1, 40),
        0,
        "own blocks come back"
    );
}
