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

//! Draft-verify lane (2026-10-12): `NativeTransformerBackend::verify_block` must predict, for a
//! block of already-known greedy tokens, exactly what the plain one-token path predicts, and a
//! rejected block must roll back cleanly so decoding continues identically. CPU reference
//! tensors, micro model, paged pool.
use aien_inference_abi::{
    sample_argmax, ModelConfig, NativeTransformerBackend, ReferenceCpuBackend, TensorBackend,
    TransformerWeights,
};
use aien_runtime::shared_kv::build_model_kv_manager_for;
use std::sync::Arc;

fn micro_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-micro-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        num_kv_heads: 2,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        eos_token_ids: Vec::new(),
        qk_norm: false,
    }
}

fn backend() -> NativeTransformerBackend {
    let weights = TransformerWeights::reference_test_weights(&micro_config());
    let cpu: Arc<dyn TensorBackend> = Arc::new(ReferenceCpuBackend::new());
    let kv = build_model_kv_manager_for(&weights, 64, cpu.as_ref()).unwrap();
    NativeTransformerBackend::with_shared_kv_and_backend(weights, cpu, kv)
}

const PROMPT: [u32; 4] = [1, 10, 25, 42];

#[test]
fn verify_block_matches_plain_greedy_and_rolls_back_rejections() {
    let mut b = backend();
    let base = b.generate_tokens(1, &PROMPT, 24, 0.0, &[]).unwrap();
    assert_eq!(base.len(), 24);

    // A fully correct block of 6 rows: every row predicts the next plain token.
    let logits = b.prefill_sequence(2, &PROMPT).unwrap();
    let first = sample_argmax(&logits).0;
    assert_eq!(first, base[0]);
    let offset = PROMPT.len();
    let preds = b.verify_block(2, &base[0..6]).unwrap();
    assert_eq!(preds, base[1..7].to_vec(), "row t must predict token t+1");
    b.truncate_kv(2, offset).unwrap();

    // A block with a wrong draft token in row 2: rows before it still predict the plain tokens,
    // rows after it are unconstrained. Accept 2, then continue plainly from the cut.
    let block = [base[0], base[1], 999, base[3]];
    let preds = b.verify_block(2, &block).unwrap();
    assert_eq!(preds[0], base[1]);
    assert_eq!(preds[1], base[2]);
    let accepted = 1; // draft tokens base[1] accepted; 999 rejected; bonus = preds[1] = base[2]
    b.truncate_kv(2, offset + accepted + 1).unwrap();
    let seq = b.sequences.get_mut(&2).unwrap();
    seq.tokens.extend_from_slice(&[base[0], base[1], base[2]]);
    let mut out = vec![base[0], base[1], base[2]];
    while out.len() < base.len() {
        let logits = b.decode_pending(2).unwrap();
        let t = sample_argmax(&logits).0;
        out.push(t);
        b.sequences.get_mut(&2).unwrap().tokens.push(t);
    }
    assert_eq!(
        out, base,
        "decoding after a rolled-back block must match the plain path"
    );

    // The pool is back to the plain path's footprint after release.
    b.release_sequence(2);
    let free_after = b.kv_manager.as_ref().unwrap().read().free_block_count();
    b.release_sequence(1);
    assert_eq!(
        b.kv_manager.as_ref().unwrap().read().free_block_count(),
        free_after
    );
}

#[test]
fn verify_block_errors_leave_the_pool_unchanged() {
    let mut b = backend();
    b.prefill_sequence(3, &PROMPT).unwrap();
    let free0 = b.kv_manager.as_ref().unwrap().read().free_block_count();
    assert!(b.verify_block(3, &[]).is_err());
    assert!(b.verify_block(4, &[1, 2]).is_err());
    assert_eq!(
        b.kv_manager.as_ref().unwrap().read().free_block_count(),
        free0
    );
    assert_eq!(
        b.kv_manager
            .as_ref()
            .unwrap()
            .read()
            .get_block_table(3)
            .unwrap()
            .total_tokens,
        PROMPT.len()
    );
}
