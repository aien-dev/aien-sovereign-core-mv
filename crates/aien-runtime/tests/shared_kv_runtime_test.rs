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

//! PREFILL-E2E-0 cut C2: one KV for runtime and backend (bullets 4, 8, 10).
//!
//! Builds the spine and the native backend through the same factory the
//! daemon uses (`aien_runtime::shared_kv::build_shared_kv_runtime`), submits
//! one root prompt (<= 16 tokens, a single prefill chunk) through the real
//! swarm -> scheduler `prefill_detached` -> backend `execute_step` path up to
//! the completion fence, then reads the root's K/V back out of the SPINE's KV
//! pool and compares it with the dense K/V an unpaged backend (no KV manager)
//! computes for the same prompt with the same weights.
//!
//! Mutant (kept outside the repo, applied by the forge job): the factory gives
//! the backend its own separate pooled manager. Then the backend writes K/V
//! into a pool the spine never sees, `Arc::ptr_eq` is false and the spine's
//! pool holds only zeros for the root, so this test must fail.
//! Mutant 2 (mutant-kv.patch): the paged prefill writes K into the V slot and
//! V into the K slot, so the pool is non-zero but the K/V-equality check fails.
//!
//! Second test: with the shared KV the backend reserves decode slots itself
//! (scheduler skips its append when `manages_kv_cache()`), so a tiny pool that
//! is full after prefill must end the request as Preempted instead of
//! decoding a token whose K/V was never stored.
//! Mutant 3 (mutant-slot.patch): slot failures are kept in the batch again,
//! so the request silently continues without K/V and this test must fail.

use std::sync::Arc;

use aien_inference_abi::{
    DecodeOutput, FinishReason, ModelConfig, NativeTransformerBackend, ReferenceCpuBackend,
    TransformerWeights,
};
use aien_kv_cache::{bf16_bits_to_f32, f32_to_bf16_bits, KvDType};
use aien_runtime::shared_kv::SHARED_KV_DTYPE;
use aien_runtime::shared_kv::{build_model_kv_manager, build_shared_kv_runtime, SharedKvSizing};
use aien_runtime::swarm::SwarmConfig;
use aien_scheduler::SchedulerConfig;

const TOL: f32 = 1e-6;

/// The spine pool stores `SHARED_KV_DTYPE`, and since the birth-rounding fix (KV_ARITH_EPOCH 2)
/// every backend rounds K/V to its storage dtype the moment they are produced, so attention sees
/// the same bits the pool keeps. The control below therefore runs with `kv_dtype = SHARED_KV_DTYPE`
/// and must match the pool value exactly (bf16 round-to-nearest-even); `pool_rounded` is then the
/// identity on the control value and only guards against a pool that stored something other than
/// the rounded f32.
fn pool_rounded(x: f32) -> f32 {
    match SHARED_KV_DTYPE {
        KvDType::Bf16 => bf16_bits_to_f32(f32_to_bf16_bits(x)),
        _ => x,
    }
}

fn small_config() -> ModelConfig {
    ModelConfig {
        model_id: "pe2e-c2-shared-kv-reference".to_string(),
        max_sequence_length: 256,
        // Small blocks so the 11-token prompt spans three blocks, the last one partial.
        block_size: 4,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        num_kv_heads: 2,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 97,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn daemon_like_scheduler() -> SchedulerConfig {
    // Same chunking as the daemon (aien-cli run_daemon_server): chunk 128, so
    // the prompt below is one prefill chunk.
    SchedulerConfig {
        max_batch_size: 32,
        max_batch_tokens: 2048,
        max_prefill_tokens: 1024,
        prefill_chunk_size: 128,
        chunk_prefill: true,
        watermark_blocks: 2,
    }
}

#[tokio::test]
async fn shared_kv_root_prefill_lands_in_spine_pool_and_matches_unpaged_backend() {
    let config = small_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    let prompt: Vec<u32> = vec![5, 17, 42, 3, 88, 61, 9, 23, 70, 11, 36];
    assert!(prompt.len() <= 16);

    let (mut spine, mut backend) = build_shared_kv_runtime(
        weights.clone(),
        Arc::new(ReferenceCpuBackend::new()),
        daemon_like_scheduler(),
        SharedKvSizing {
            arena_capacity: 64,
            total_blocks: 64,
        },
    )
    .expect("build shared KV runtime");

    // Check 1: one KV. The backend holds the very same Arc as the spine.
    let backend_kv = backend
        .kv_manager
        .clone()
        .expect("backend must be wired to a KV manager");
    assert!(
        Arc::ptr_eq(&backend_kv, &spine.kv_manager),
        "SHARED_KV_VIOLATION: backend KV manager is not the spine's KV manager"
    );
    assert!(
        spine.kv_manager.read().tensor_pool().is_some(),
        "SHARED_KV_VIOLATION: spine KV manager has no physical tensor pool"
    );

    // Submit one root prompt through the real swarm/scheduler path.
    let swarm_id = spine
        .launch_swarm(
            SwarmConfig {
                model_handle: 1,
                branch_count: 0,
                max_active_sequences: 1,
                max_tokens_per_branch: 1,
                root_world_id: 0,
                priority: 1,
            },
            &prompt,
        )
        .expect("launch swarm");
    let root = spine
        .swarm_manager
        .get_swarm(swarm_id)
        .expect("swarm record")
        .root_sequence_id
        .as_u64();

    let before = spine
        .kv_manager
        .read()
        .prefill_state(root)
        .expect("root has a KV table after launch");
    assert!(!before.is_ready(), "root must not be ready before prefill");

    let metrics = spine
        .step(&mut backend)
        .await
        .expect("spine step")
        .expect("step must report the root prefill");
    assert_eq!(metrics.prefill_tokens_processed, prompt.len());

    let after = spine
        .kv_manager
        .read()
        .prefill_state(root)
        .expect("root still has a KV table after prefill");
    assert!(
        after.is_ready(),
        "root must be ready after the completion fence"
    );
    assert!(!spine.swarm_manager.is_root_prefill_pending(swarm_id));

    // Control: an unpaged backend (no KV manager) with the same weights.
    // Same storage dtype as the pool: both sides round at birth, so the hidden states that feed
    // later layers agree bit for bit (an f32 control diverges from layer 1 on, see #589).
    let mut control = NativeTransformerBackend::new_reference(weights);
    control.kv_dtype = SHARED_KV_DTYPE;
    let control_id = 9_000_001u64;
    control
        .prefill_sequence(control_id, &prompt)
        .expect("control prefill");
    let control_seq = control
        .sequences
        .get(&control_id)
        .expect("control sequence state");

    let kv_dim = config.num_kv_heads * config.head_dim;
    let kv = spine.kv_manager.read();
    for layer in 0..config.num_layers {
        let mut k = Vec::new();
        let mut v = Vec::new();
        kv.gather_sequence_layer_kv(root, layer, &mut k, &mut v)
            .expect("gather root K/V from the spine's pool");

        let ck = &control_seq.layers[layer].flat_k;
        let cv = &control_seq.layers[layer].flat_v;
        assert_eq!(
            ck.len(),
            prompt.len() * kv_dim,
            "control K length, layer {layer}"
        );
        assert_eq!(k.len(), ck.len(), "spine pool K length, layer {layer}");
        assert_eq!(v.len(), cv.len(), "spine pool V length, layer {layer}");

        assert!(
            k.iter().any(|x| *x != 0.0) && v.iter().any(|x| *x != 0.0),
            "SHARED_KV_VIOLATION: spine pool K/V for root is all zero at layer {layer} \
             (prefill K/V did not land in the runtime KV)"
        );
        for (i, (a, b)) in k.iter().zip(ck.iter()).enumerate() {
            assert!(
                (a - pool_rounded(*b)).abs() <= TOL,
                "K mismatch layer {layer} idx {i}: spine pool {a} vs unpaged {b} (rounded {})",
                pool_rounded(*b)
            );
        }
        for (i, (a, b)) in v.iter().zip(cv.iter()).enumerate() {
            assert!(
                (a - pool_rounded(*b)).abs() <= TOL,
                "V mismatch layer {layer} idx {i}: spine pool {a} vs unpaged {b} (rounded {})",
                pool_rounded(*b)
            );
        }
    }
}

#[test]
fn shared_kv_decode_slot_exhaustion_preempts_instead_of_decoding_without_kv() {
    let config = small_config();
    let weights = TransformerWeights::reference_test_weights(&config);
    // block_size 4, two blocks: an 8-token prompt fills the whole pool.
    let kv = build_model_kv_manager(&weights, 2).expect("tiny pooled KV manager");
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        kv.clone(),
    );
    let id = 77u64;
    let prompt: Vec<u32> = vec![5, 17, 42, 3, 88, 61, 9, 23];
    backend
        .prefill_sequence(id, &prompt)
        .expect("prefill fits the pool");
    assert_eq!(
        kv.read().free_block_count(),
        0,
        "pool must be full after prefill"
    );
    let tokens_before = backend
        .sequences
        .get(&id)
        .expect("backend sequence")
        .tokens
        .len();
    let kv_tokens_before = kv
        .read()
        .get_block_table(id)
        .expect("KV table")
        .total_tokens;

    let outputs = backend
        .forward_decode_batch(&[id])
        .expect("known, prefilled id: slot failure is an output, not an error");

    assert_eq!(
        outputs.len(),
        1,
        "exactly one output for the one decode request"
    );
    match &outputs[0] {
        DecodeOutput::Finished {
            request_id, reason, ..
        } => {
            assert_eq!(*request_id, id);
            assert_eq!(
                *reason,
                FinishReason::Preempted,
                "decode slot failure must end the request as Preempted"
            );
        }
        DecodeOutput::Token { token_id, .. } => panic!(
            "SHARED_KV_VIOLATION: decode emitted token {token_id} although no K/V slot \
             could be reserved (request silently continued without K/V)"
        ),
    }
    assert_eq!(
        backend
            .sequences
            .get(&id)
            .map_or(tokens_before, |s| s.tokens.len()),
        tokens_before,
        "SHARED_KV_VIOLATION: backend generated a token with no K/V slot"
    );
    assert_eq!(
        kv.read()
            .get_block_table(id)
            .expect("KV table")
            .total_tokens,
        kv_tokens_before,
        "KV table must not grow when the slot reservation failed"
    );
}
