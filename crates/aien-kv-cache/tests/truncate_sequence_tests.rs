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

//! `truncate_sequence` (draft-verify lane, 2026-10-12): drops the K/V past a position so a
//! speculative verify block can be rolled back; the next append continues at the cut.
use aien_kv_cache::AienKvManager;

const BS: usize = 16;

#[test]
fn truncate_frees_trailing_blocks_and_appends_continue_at_the_cut() {
    let mut mgr = AienKvManager::new(64, BS);
    let prompt: Vec<u32> = (0..40).collect();
    mgr.allocate_sequence(1, &prompt).unwrap();
    let free0 = mgr.free_block_count();
    // Eight appends: 48 tokens, 3 blocks full.
    let mut slots = Vec::new();
    for _ in 0..8 {
        slots.push(mgr.append_token_with_slot(1).unwrap());
    }
    assert_eq!(mgr.get_block_table(1).unwrap().total_tokens, 48);
    // Roll back to 41: the third block keeps 9 tokens, nothing freed.
    mgr.truncate_sequence(1, 41).unwrap();
    let t = mgr.get_block_table(1).unwrap();
    assert_eq!(t.total_tokens, 41);
    assert_eq!(t.block_ids.len(), 3);
    assert_eq!(mgr.free_block_count(), free0);
    // The next append lands where token 41 was.
    assert_eq!(mgr.append_token_with_slot(1).unwrap(), slots[1]);
    // Roll back to a block boundary: the third block is freed, the next append opens a new one.
    mgr.truncate_sequence(1, 32).unwrap();
    let t = mgr.get_block_table(1).unwrap();
    assert_eq!(t.total_tokens, 32);
    assert_eq!(t.block_ids.len(), 2);
    assert_eq!(mgr.free_block_count(), free0 + 1);
    let (blk, slot) = mgr.append_token_with_slot(1).unwrap();
    assert_eq!(slot, 0);
    assert_eq!(mgr.get_block_table(1).unwrap().block_ids[2], blk);
    assert_eq!(mgr.get_block_table(1).unwrap().total_tokens, 33);
    // Past the end and unknown sequences are refused; no-op at the current length.
    assert!(mgr.truncate_sequence(1, 34).is_err());
    assert!(mgr.truncate_sequence(2, 0).is_err());
    mgr.truncate_sequence(1, 33).unwrap();
    assert_eq!(mgr.get_block_table(1).unwrap().total_tokens, 33);
}

#[test]
fn truncate_refuses_shared_blocks() {
    let mut mgr = AienKvManager::new(64, BS);
    let prompt: Vec<u32> = (0..40).collect();
    mgr.allocate_sequence(1, &prompt).unwrap();
    mgr.fork_sequence(1, 2).unwrap();
    // The forked prefix is shared: a cut inside it must be refused, a cut that only touches
    // the child's own tail is fine once it has one.
    assert!(mgr.truncate_sequence(2, 20).is_err());
    assert!(mgr.truncate_sequence(2, 40).is_ok());
}
