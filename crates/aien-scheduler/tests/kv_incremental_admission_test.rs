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

//! GB10-2: the KV manager is the sole authority for how many NEW physical
//! blocks a sequence needs, and the scheduler admits by that number.
//!
//! On main (c75c583) `build_scheduled_batch` step 3 computes
//! `required_blocks = prompt_len.div_ceil(16)` (a literal 16, not the pool's
//! block size) and breaks when the free pool is below it, BEFORE it looks at
//! the sequence's block table. A fork child that already shares its parent's
//! prefix blocks is therefore refused unless the pool could hold the WHOLE
//! prompt again, and a pool with a block size other than 16 is mis-counted.
//! These tests use only APIs that exist on main, so they run (and FAIL) there.

use aien_inference_abi::{SamplingParams, SequenceRequest, KV_POOL_EXHAUSTED_PREFIX};
use aien_kv_cache::create_shared_kv_manager;
use aien_scheduler::{AienScheduler, SchedulerConfig, SequenceId};

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 4096,
        max_prefill_tokens: 4096,
        prefill_chunk_size: 4096,
        chunk_prefill: true,
        watermark_blocks: 1,
    }
}

fn request(id: u64, prompt: &[u32]) -> SequenceRequest {
    SequenceRequest {
        request_id: id,
        prompt_tokens: prompt.to_vec(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 4,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority: 1,
    }
}

/// Parent with a computed 40-token prefix (3 blocks of 16, last one partial),
/// forked to a child that shares all three blocks.
fn parent_and_forked_child(
    total_blocks: usize,
    block_size: usize,
) -> (aien_kv_cache::SharedKvManager, AienScheduler, u64, Vec<u32>) {
    let kv = create_shared_kv_manager(total_blocks, block_size);
    let scheduler = AienScheduler::new(config(), kv.clone());
    let parent = SequenceId::new(2, 1).unwrap().to_u64();
    let child = SequenceId::new(3, 1).unwrap().to_u64();
    let prompt: Vec<u32> = (10..50).collect();
    {
        let mut w = kv.write();
        w.allocate_sequence(parent, &prompt).expect("allocate");
        w.mark_prefill_pending(parent).expect("pending");
        w.complete_prefill(parent).expect("fence");
        w.fork_prefilled(parent, child).expect("fork");
    }
    (kv, scheduler, child, prompt)
}

#[test]
fn fork_child_is_admitted_on_incremental_blocks_only() {
    // 5-block pool. Parent holds 3, child shares them: 2 blocks are free.
    // The child's full prompt is 3 blocks (> 2 free) but it only needs ONE new
    // block (copy-on-write of the shared partial tail on its first decode).
    let (kv, mut scheduler, child, prompt) = parent_and_forked_child(5, 16);
    assert_eq!(kv.read().available_blocks(), 2, "setup: 2 free blocks");
    scheduler.submit_request(request(child, &prompt));

    let batch = scheduler.build_scheduled_batch().expect("schedule");
    let admitted = batch
        .as_ref()
        .is_some_and(|b| b.decode_requests.contains(&child));
    assert!(
        admitted,
        "KV_ADMISSION_TRAP: forked child {child} shares its parent's 3 prefix blocks and needs \
         1 new block, the pool has 2 free, but admission demanded a whole prompt's blocks (3); \
         batch = {batch:?}"
    );
}

#[test]
fn admission_uses_the_pool_block_size_not_a_literal_16() {
    // Block size 32: a 40-token prompt needs 2 blocks and the pool has 2.
    // Main counts 40.div_ceil(16) = 3 and refuses it.
    let kv = create_shared_kv_manager(2, 32);
    let mut scheduler = AienScheduler::new(config(), kv.clone());
    let prompt: Vec<u32> = (10..50).collect();
    let id = SequenceId::new(4, 1).unwrap().to_u64();
    scheduler.submit_request(request(id, &prompt));

    let batch = scheduler.build_scheduled_batch().expect("schedule");
    let admitted = batch
        .as_ref()
        .is_some_and(|b| b.prefill_requests.iter().any(|r| r.request_id == id));
    assert!(
        admitted,
        "KV_ADMISSION_BLOCK_SIZE: 40 tokens at block size 32 need 2 blocks and 2 are free, but \
         admission used a hard-coded 16 (3 blocks); batch = {batch:?}"
    );
}

#[test]
fn exhausted_pool_with_nothing_running_fails_loud() {
    // Negative control. 3-block pool, all held by the parent, child shares them.
    // The child needs 1 new block (copy-on-write), 0 are free, nothing is running
    // that could ever free one: the scheduler must say so, not wait forever.
    let (kv, mut scheduler, child, prompt) = parent_and_forked_child(3, 16);
    assert_eq!(kv.read().available_blocks(), 0, "setup: pool is full");
    scheduler.submit_request(request(child, &prompt));

    let result = scheduler.build_scheduled_batch();
    let err = result.expect_err(
        "KV_ADMISSION_SILENT_WAIT: pool cannot admit the child and nothing is running, yet \
         build_scheduled_batch returned Ok",
    );
    assert!(
        err.starts_with(KV_POOL_EXHAUSTED_PREFIX),
        "error must start with the shared marker, got: {err}"
    );
    assert!(
        err.contains(&format!("sequence {child}")),
        "error must name the sequence, got: {err}"
    );
    assert!(
        err.contains("needs 1 new KV blocks") && err.contains("only 0 are free"),
        "error must say needed vs available, got: {err}"
    );
}

#[test]
fn shortfall_while_others_run_waits_without_error() {
    // Control (must pass before and after): when a running sequence may still
    // free blocks, a shortfall is ordinary queueing, not an error.
    let kv = create_shared_kv_manager(3, 16);
    let mut scheduler = AienScheduler::new(config(), kv.clone());
    let first: Vec<u32> = (10..58).collect(); // 48 tokens = 3 blocks
    let second: Vec<u32> = (10..26).collect(); // 16 tokens = 1 block
    let a = SequenceId::new(5, 1).unwrap().to_u64();
    let b = SequenceId::new(6, 1).unwrap().to_u64();
    scheduler.submit_request(request(a, &first));
    scheduler.submit_request(request(b, &second));

    let batch = scheduler
        .build_scheduled_batch()
        .expect("a shortfall with a running sequence must not be an error")
        .expect("first sequence is admitted");
    assert!(batch.prefill_requests.iter().any(|r| r.request_id == a));
    assert!(!batch.prefill_requests.iter().any(|r| r.request_id == b));
    assert_eq!(scheduler.waiting_count(), 1, "second request keeps waiting");
}

#[test]
fn same_step_fork_admissions_reserve_their_copy_on_write_blocks() {
    // 4-block pool: parent holds 3, ONE block free. Two children each need one
    // new block on their first token. Admitting both in one step would promise
    // the same free block twice; only the first may enter this batch.
    let (kv, mut scheduler, first_child, prompt) = parent_and_forked_child(4, 16);
    let second_child = SequenceId::new(7, 1).unwrap().to_u64();
    let parent = SequenceId::new(2, 1).unwrap().to_u64();
    kv.write()
        .fork_prefilled(parent, second_child)
        .expect("fork");
    assert_eq!(kv.read().available_blocks(), 1, "setup: 1 free block");
    scheduler.submit_request(request(first_child, &prompt));
    scheduler.submit_request(request(second_child, &prompt));

    let batch = scheduler
        .build_scheduled_batch()
        .expect("second child waits, it is not an error: the first is scheduled")
        .expect("first child admitted");
    assert_eq!(batch.decode_requests, vec![first_child]);
    assert_eq!(scheduler.waiting_count(), 1, "second child keeps waiting");
}
