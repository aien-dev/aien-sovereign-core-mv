#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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

//! Paged KV pool layout certification (host only).
//!
//! Enforces:
//! 1. Pure layout stride invariants against independent calculation.
//! 2. Coordinate-encoded byte pattern integrity in the physical pool.
//!
//! The GPU parity layers of this suite exercised the CUDA bf16 kernel, which FB-1 cut 6
//! removed. Omega's paged attention is checked against the CPU reference by the
//! aien-inference-runtime parity and strict real-model gates.

use aien_kv_cache::{KvDType, KvLayout, KvPoolConfig, UnifiedKvTensorPool};

/// Independent byte-offset formula calculating addresses without any production code.
fn oracle_byte_offset(
    block: usize,
    layer: usize,
    is_value: bool,
    token: usize,
    head: usize,
    dim: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    elem_bytes: usize,
) -> usize {
    let head_stride = head_dim * elem_bytes;
    let token_stride = num_kv_heads * head_stride;
    let plane_stride = block_size * token_stride;
    let layer_stride = 2 * plane_stride;
    let block_stride = num_layers * layer_stride;

    block * block_stride
        + layer * layer_stride
        + (if is_value { plane_stride } else { 0 })
        + token * token_stride
        + head * head_stride
        + dim * elem_bytes
}
#[test]
fn test_layer1_layout_strides_independent() {
    let config = KvPoolConfig {
        num_blocks: 16,
        block_size: 16,
        num_layers: 22,
        num_kv_heads: 4,
        head_dim: 64,
        dtype: KvDType::Bf16,
    };

    let layout = KvLayout::for_bf16(&config).unwrap();
    let desc = layout.to_desc();

    let elem_bytes = 2;
    let expected_head_stride = 64 * elem_bytes;
    let expected_token_stride = 4 * expected_head_stride;
    let expected_plane_stride = 16 * expected_token_stride;
    let expected_layer_stride = 2 * expected_plane_stride;
    let expected_block_stride = 22 * expected_layer_stride;

    assert_eq!(desc.head_stride_bytes, expected_head_stride as u64);
    assert_eq!(desc.token_stride_bytes, expected_token_stride as u64);
    assert_eq!(desc.kv_plane_stride_bytes, expected_plane_stride as u64);
    assert_eq!(desc.layer_stride_bytes, expected_layer_stride as u64);
    assert_eq!(desc.block_stride_bytes, expected_block_stride as u64);

    for b in [0, 5, 15] {
        for l in [0, 11, 21] {
            for is_v in [false, true] {
                for tok in [0, 8, 15] {
                    for h in [0, 2, 3] {
                        for d in [0, 31, 63] {
                            let prod_offset = layout.element_offset(b, l, is_v, tok, h, d);
                            let oracle_off = oracle_byte_offset(
                                b, l, is_v, tok, h, d, 22, 4, 64, 16, elem_bytes,
                            );
                            assert_eq!(
                                prod_offset, oracle_off,
                                "Mismatch at ({}, {}, {}, {}, {}, {})",
                                b, l, is_v, tok, h, d
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_layer2_physical_coordinate_byte_pattern() {
    let config = KvPoolConfig {
        num_blocks: 4,
        block_size: 16,
        num_layers: 4,
        num_kv_heads: 2,
        head_dim: 32,
        dtype: KvDType::Bf16,
    };

    let mut pool = UnifiedKvTensorPool::allocate(config.clone()).unwrap();

    // Write unique coordinate patterns
    for b in 0..4 {
        for l in 0..4 {
            for tok in 0..16 {
                let mut k_vec = vec![0.0f32; 2 * 32];
                let mut v_vec = vec![0.0f32; 2 * 32];
                for h in 0..2 {
                    for d in 0..32 {
                        let idx = h * 32 + d;
                        let val = (b * 32 + l * 8 + tok + h * 20 + 1) as f32;
                        k_vec[idx] = val;
                        v_vec[idx] = -val;
                    }
                }
                pool.write_token_kv(b, l, tok, &k_vec, &v_vec);
            }
        }
    }

    // Read back and verify exact recovery
    for b in 0..4 {
        for l in 0..4 {
            for tok in 0..16 {
                let mut k_out = vec![0.0f32; 2 * 32];
                let mut v_out = vec![0.0f32; 2 * 32];
                pool.read_token_kv(b, l, tok, &mut k_out, &mut v_out);
                for h in 0..2 {
                    for d in 0..32 {
                        let idx = h * 32 + d;
                        let expected_val = (b * 32 + l * 8 + tok + h * 20 + 1) as f32;
                        assert_eq!(
                            k_out[idx], expected_val,
                            "K mismatch at ({}, {}, {}, {}, {}): expected {}, got {}",
                            b, l, tok, h, d, expected_val, k_out[idx]
                        );
                        assert_eq!(
                            v_out[idx], -expected_val,
                            "V mismatch at ({}, {}, {}, {}, {}): expected {}, got {}",
                            b, l, tok, h, d, -expected_val, v_out[idx]
                        );
                    }
                }
            }
        }
    }
}
