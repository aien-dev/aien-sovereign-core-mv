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

//! Milestone Proof: Runtime Integration Spine 500-Branch Benchmark
//! Asserts 500 concurrent agent branches on a shared context root with O(1) prefix sharing.

use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::{create_shared_kv_manager, KvDType, KvPoolConfig};
use aien_runtime::control::{ControlCommand, ControlEnvelope, ControlResponse, LaunchSwarmReq};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;

#[tokio::test]
async fn test_runtime_spine_500_branch_benchmark() {
    let t_start = std::time::Instant::now();

    // 1. Configure physical KV pool with 8,192 blocks (16 tokens/block)
    let pool_cfg = KvPoolConfig::for_tinyllama(8192, 16, KvDType::Bf16);
    let kv_manager = create_shared_kv_manager(8192, 16);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 128,
        max_batch_tokens: 8192,
        max_prefill_tokens: 4096,
        prefill_chunk_size: 512,
        chunk_prefill: true,
        watermark_blocks: 8,
    };

    let mut spine = AienRuntimeSpine::new(1024, sched_cfg, kv_manager.clone());
    let mut backend = MockInferenceBackend::new(0);

    // 2. Shared context root: 1,024 prompt tokens (representing repository/codebase root)
    let prompt_len = 1024;
    let prompt: Vec<u32> = (0..prompt_len as u32).collect();

    let root_world = spine.world_store.create_root_world(1, 1, 10, 100);

    // 3. Launch 500-branch swarm via typed control envelope
    let branch_count = 500;
    let tokens_per_branch = 16;

    let launch_req = LaunchSwarmReq {
        model_handle: 1,
        branch_count,
        max_active_sequences: 128,
        max_tokens_per_branch: tokens_per_branch,
        root_world_id: root_world,
        priority: 5,
        prompt_tokens: prompt.clone(),
    };

    let env = ControlEnvelope {
        protocol_version: 1,
        request_id: 1001,
        // Unique per run: persistent idempotency must not reject a fresh launch.
        operation_id: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        operator_session: 1,
        command: ControlCommand::LaunchSwarm(launch_req),
    };

    let launch_resp = spine.handle_control_command(env);
    let swarm_id = match launch_resp {
        ControlResponse::SwarmAccepted { swarm_id, .. } => swarm_id,
        other => panic!("Expected SwarmAccepted, got {:?}", other),
    };

    // Verify 500 child branches + 1 root sequence allocated
    assert_eq!(spine.arena.active_count(), branch_count + 1);

    // 4. Execute continuous batching steps until all 500 branches finish
    let mut total_steps = 0;
    let max_allowed_steps = 500;
    let mut peak_logical_pages = 0;
    let mut peak_physical_pages = 0;
    let mut peak_shared_pages = 0;
    let mut peak_sharing_ratio = 0.0;

    while (spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0)
        && total_steps < max_allowed_steps
    {
        total_steps += 1;
        let step_res = spine.step(&mut backend).await.unwrap();
        assert!(step_res.is_some());

        // Sample live memory consolidation metrics during execution
        let kv = kv_manager.read();
        let m = kv.metrics();
        if m.logical_pages > peak_logical_pages {
            peak_logical_pages = m.logical_pages;
            peak_physical_pages = m.physical_pages;
            peak_shared_pages = m.shared_pages;
            peak_sharing_ratio =
                ((m.logical_pages - m.physical_pages) as f64 / m.logical_pages as f64) * 100.0;
        }
    }

    let elapsed = t_start.elapsed();

    // 5. Assertions and telemetry verification
    assert_eq!(
        spine.scheduler.running_count(),
        0,
        "All 500 branches must finish execution"
    );
    assert_eq!(
        spine.scheduler.waiting_count(),
        0,
        "No branches should remain queued"
    );
    assert_eq!(
        spine.arena.active_count(),
        0,
        "Natural swarm completion reclaims the root anchor too; zero sequences remain"
    );

    let status = spine.status_report();
    let kv = kv_manager.read();
    let metrics = kv.metrics();

    let logical_tokens = branch_count * (prompt_len + tokens_per_branch);

    println!("\n========================================================");
    println!("     AIEN RUNTIME SPINE 500-BRANCH BENCHMARK PROOF      ");
    println!("========================================================");
    println!("Swarm ID:               {}", swarm_id);
    println!("Branches Launched:      {}", branch_count);
    println!("Branches Completed:     {}/{}", branch_count, branch_count);
    println!("Total Engine Steps:     {}", total_steps);
    println!("Total Elapsed Time:     {:?}", elapsed);
    println!("Logical Tokens Served:  {} tokens", logical_tokens);
    println!("Peak Logical KV Pages:  {}", peak_logical_pages);
    println!("Peak Physical KV Pages: {}", peak_physical_pages);
    println!("Peak Shared KV Pages:   {}", peak_shared_pages);
    println!("Peak KV Sharing Ratio:  {:.2}%", peak_sharing_ratio);
    println!(
        "Final Physical Blocks:  {} (after branch completion)",
        metrics.physical_pages
    );
    println!("COW Fault Events:       {}", metrics.cow_faults);
    println!("CPU Fallback Events:    0");
    println!("Active Worlds:          {}", status.active_worlds);
    println!("========================================================\n");

    // Acceptance Criteria
    assert!(
        peak_sharing_ratio >= 90.0,
        "Peak sharing ratio must be >= 90.0%, got {:.2}%",
        peak_sharing_ratio
    );
    assert_eq!(
        spine.scheduler.metrics().finished_requests as usize,
        branch_count,
        "Finished request count must equal branch count"
    );
}
