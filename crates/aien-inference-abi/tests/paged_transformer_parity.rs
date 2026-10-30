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

use aien_inference_abi::{ModelConfig, NativeTransformerBackend, TransformerWeights};

#[test]
fn test_paged_vs_contiguous_numerical_parity() {
    let config = ModelConfig {
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        hidden_dim: 128,
        intermediate_dim: 256,
        vocab_size: 512,
        block_size: 16,
        ..Default::default()
    };

    let weights = TransformerWeights::reference_test_weights(&config);

    // 1. Contiguous reference backend
    let mut contiguous_backend = NativeTransformerBackend::new_reference(weights.clone());

    // 2. Paged physical COW backend
    let mut paged_backend = NativeTransformerBackend::with_paged_kv(weights, 500, 16).unwrap();

    let prompt: Vec<u32> = (1..32).collect();

    // Prefill both
    let _contiguous_logits = contiguous_backend.prefill_sequence(1, &prompt).unwrap();
    let paged_parent = paged_backend.create_context(&prompt).unwrap();
    let paged_branch = paged_backend.fork_context(paged_parent).unwrap();

    // Autoregressive generation parity across 10 steps
    for step in 0..10 {
        let (paged_tok, paged_logits) = paged_backend.decode_branch_step(paged_branch).unwrap();

        let (contig_tok, contig_logits) = {
            let h = {
                let seq = contiguous_backend.sequences.get_mut(&1).unwrap();
                let p = seq.tokens.len() - 1;
                let lt = *seq.tokens.last().unwrap();
                NativeTransformerBackend::forward_token_impl_paged(
                    &contiguous_backend.weights,
                    &*contiguous_backend.tensor_backend,
                    lt,
                    p,
                    seq,
                    1,
                    None,
                )
            };
            let l = contiguous_backend.compute_logits(&h);
            let (sampled, _) = aien_inference_abi::sample_argmax(&l);
            contiguous_backend
                .sequences
                .get_mut(&1)
                .unwrap()
                .tokens
                .push(sampled);
            (sampled, l)
        };

        let max_diff = paged_logits
            .iter()
            .zip(contig_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "Step {}: paged_tok={}, contig_tok={}, max_diff={}",
            step, paged_tok, contig_tok, max_diff
        );

        assert_eq!(
            paged_tok, contig_tok,
            "Token divergence at step {}: paged={}, contiguous={}",
            step, paged_tok, contig_tok
        );
    }
}

#[test]
fn test_32_branch_fork_and_cow_acceptance() {
    let config = ModelConfig {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        block_size: 16,
        ..Default::default()
    };

    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_paged_kv(weights, 2000, 16).unwrap();

    let prompt: Vec<u32> = (1..=128).collect();
    let parent = backend.create_context(&prompt).unwrap();

    let mut branches = Vec::new();
    for _ in 0..32 {
        let b = backend.fork_context(parent).unwrap();
        branches.push(b);
    }

    let r_init = backend.get_usage_receipt(branches[0]).unwrap();
    assert_eq!(r_init.shared_pages, 8);
    assert_eq!(r_init.private_pages, 0);
    assert_eq!(r_init.cow_faults, 0);
    assert_eq!(r_init.logical_pages, 8);
    assert!(r_init.bytes_saved_vs_full_copy > 0);

    for &b in &branches {
        let (tok, _) = backend.decode_branch_step(b).unwrap();
        assert!(tok > 0);
    }

    let r_branch = backend.get_usage_receipt(branches[0]).unwrap();
    assert_eq!(r_branch.shared_pages, 8);
    assert_eq!(r_branch.private_pages, 1);
    assert_eq!(r_branch.logical_pages, 9);

    for b in branches {
        backend.release_branch(b).unwrap();
    }

    let kv_mgr = backend.kv_manager.as_ref().unwrap().read();
    assert_eq!(kv_mgr.active_sequence_count(), 1);
    let m = kv_mgr.metrics();
    assert_eq!(m.physical_pages, 8);
    assert_eq!(m.shared_pages, 0);
    assert_eq!(m.private_pages, 8);
}
