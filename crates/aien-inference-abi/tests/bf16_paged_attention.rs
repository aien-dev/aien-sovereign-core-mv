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

//! CPU paged attention must decode the pool dtype. A bf16 pool read as f32
//! diverges from the GPU path on the first decode step.

use aien_inference_abi::{ReferenceCpuBackend, TensorBackend};
use aien_kv_cache::{KvDType, KvPoolConfig, UnifiedKvTensorPool};

fn attention_from_decoded(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_q_heads * head_dim];
    let gqa = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let kv_dim = num_kv_heads * head_dim;
    for h in 0..num_q_heads {
        let kv_head = h / gqa;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut m_prev = f64::NEG_INFINITY;
        let mut l_prev = 0.0f64;
        let mut acc = vec![0.0f64; head_dim];
        for t in 0..context_len {
            let k_head = &k[t * kv_dim + kv_head * head_dim..t * kv_dim + (kv_head + 1) * head_dim];
            let v_head = &v[t * kv_dim + kv_head * head_dim..t * kv_dim + (kv_head + 1) * head_dim];
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += q_head[d] as f64 * k_head[d] as f64;
            }
            let score = dot * scale;
            let m_new = m_prev.max(score);
            let alpha = (m_prev - m_new).exp();
            let beta = (score - m_new).exp();
            let l_new = l_prev * alpha + beta;
            for d in 0..head_dim {
                acc[d] = acc[d] * alpha + beta * v_head[d] as f64;
            }
            m_prev = m_new;
            l_prev = l_new;
        }
        let inv_l = if l_prev > 0.0 { 1.0 / l_prev } else { 0.0 };
        for d in 0..head_dim {
            out[h * head_dim + d] = (acc[d] * inv_l) as f32;
        }
    }
    out
}

#[test]
fn cpu_paged_attention_matches_decoded_bf16() {
    let num_kv_heads = 2;
    let head_dim = 8;
    let num_q_heads = 4;
    let context_len = 3;
    let cfg = KvPoolConfig {
        num_blocks: 2,
        block_size: 8,
        num_layers: 1,
        num_kv_heads,
        head_dim,
        dtype: KvDType::Bf16,
    };
    let mut pool = UnifiedKvTensorPool::allocate(cfg).unwrap();
    let kv_dim = num_kv_heads * head_dim;
    let mut k_rows = Vec::new();
    let mut v_rows = Vec::new();
    for t in 0..context_len {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t + 1) as f32) * 0.15 + i as f32 * 0.01)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t + 1) as f32) * -0.2 + i as f32 * 0.02)
            .collect();
        pool.write_token_kv(0, 0, t, &k, &v);
        let mut k_back = vec![0.0; kv_dim];
        let mut v_back = vec![0.0; kv_dim];
        pool.read_token_kv(0, 0, t, &mut k_back, &mut v_back);
        k_rows.extend(k_back);
        v_rows.extend(v_back);
    }
    let q: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let expected = attention_from_decoded(
        &q,
        &k_rows,
        &v_rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        context_len,
    );
    let backend = ReferenceCpuBackend::new();
    let mut actual = vec![0.0; q.len()];
    backend.paged_attention(
        &mut actual,
        &q,
        &pool,
        &[0],
        context_len,
        0,
        num_q_heads,
        num_kv_heads,
        head_dim,
    );
    for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "head value {i}: cpu {got} decoded-bf16 {want}"
        );
    }
}
