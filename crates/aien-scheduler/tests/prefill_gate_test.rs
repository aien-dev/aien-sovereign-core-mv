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

//! PREFILL-GATE: allocated != computed.
//!
//! Uses only APIs that already exist on main (cfd9982), so this file also runs
//! against main, where it must FAIL (see scripts/prefill-gate/fails-on-main.sh).
//! On main the admission rule (crates/aien-scheduler/src/lib.rs:475-492 at
//! cfd9982) treats any existing block table as prefilled and decodes it.

use aien_inference_abi::{SamplingParams, SequenceRequest};
use aien_kv_cache::create_shared_kv_manager;
use aien_scheduler::{AienScheduler, SchedulerConfig, SequenceId};

#[test]
fn prefill_gate_allocated_but_unprefilled_table_is_prefilled_not_decoded() {
    let kv_manager = create_shared_kv_manager(64, 16);
    let config = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 1024,
        max_prefill_tokens: 512,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 2,
    };
    let mut scheduler = AienScheduler::new(config, kv_manager.clone());

    // A swarm-style branch id (slot > 0) whose blocks were allocated and
    // zeroed by allocate_sequence, but never run through the model.
    let branch = SequenceId::new(3, 1).unwrap().to_u64();
    let prompt: Vec<u32> = (10..50).collect();
    kv_manager
        .write()
        .allocate_sequence(branch, &prompt)
        .expect("allocate");

    scheduler.submit_request(SequenceRequest {
        request_id: branch,
        prompt_tokens: prompt.clone(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 4,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority: 1,
    });

    let batch = scheduler
        .build_scheduled_batch()
        .expect("schedule")
        .expect("non-empty batch");

    assert!(
        !batch.decode_requests.contains(&branch),
        "PREFILL_GATE_VIOLATION: sequence with allocated-but-uncomputed KV blocks was scheduled for decode"
    );
    assert!(
        batch
            .prefill_requests
            .iter()
            .any(|r| r.request_id == branch && r.prompt_tokens == prompt),
        "PREFILL_GATE_VIOLATION: sequence with uncomputed KV blocks was not scheduled for prompt prefill"
    );
}
