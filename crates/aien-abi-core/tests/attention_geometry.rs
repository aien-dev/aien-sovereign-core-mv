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

//! Contract tests for `AttentionGeometry` (Drake's attention hardening cut, phase 1).
//! Numbers come from the model configs in this repo, not from guesses.

use aien_abi_core::{AttentionGeometry, AttentionGeometryError, ModelConfig};

#[test]
fn qwen3_coder_q_dim_differs_from_hidden_dim() {
    // crates/aien-inference-abi/src/qwen3_coder.rs: 32 q heads, 4 kv heads, head_dim 128,
    // hidden 2048. q_dim (4096) is NOT hidden_dim and the contract must not care.
    let g = AttentionGeometry::new(32, 4, 128).expect("Qwen3-Coder geometry");
    assert_eq!(g.q_dim(), 4096);
    assert_eq!(g.kv_dim(), 512);
    assert_eq!(g.gqa_ratio(), 8);
    assert_ne!(
        g.q_dim(),
        2048,
        "q_dim must be allowed to differ from hidden_dim"
    );
    assert_eq!(g.kv_head_of(0), Some(0));
    assert_eq!(g.kv_head_of(7), Some(0));
    assert_eq!(g.kv_head_of(8), Some(1));
    assert_eq!(g.kv_head_of(31), Some(3));
    assert_eq!(g.kv_head_of(32), None);
}

#[test]
fn tinyllama_and_default_configs_build() {
    let t = ModelConfig::tinyllama_1_1b()
        .attention_geometry()
        .expect("TinyLlama geometry");
    assert_eq!(
        (t.num_q_heads(), t.num_kv_heads(), t.head_dim()),
        (32, 4, 64)
    );
    assert_eq!(t.q_dim(), 2048);
    assert_eq!(t.kv_dim(), 256);
    let d = ModelConfig::default()
        .attention_geometry()
        .expect("default geometry");
    assert_eq!((d.q_dim(), d.kv_dim(), d.gqa_ratio()), (4096, 1024, 4));
}

#[test]
fn multi_head_attention_is_ratio_one() {
    let g = AttentionGeometry::new(16, 16, 64).unwrap();
    assert_eq!(g.gqa_ratio(), 1);
    for h in 0..16 {
        assert_eq!(g.kv_head_of(h), Some(h));
    }
    assert_eq!(g.q_dim(), g.kv_dim());
}

#[test]
fn non_power_of_two_ratio_is_accepted() {
    // The contract describes the model, not one kernel's envelope.
    let g = AttentionGeometry::new(24, 8, 64).unwrap();
    assert_eq!(g.gqa_ratio(), 3);
    assert_eq!(g.kv_head_of(5), Some(1));
    let g = AttentionGeometry::new(40, 8, 128).unwrap();
    assert_eq!(g.gqa_ratio(), 5);
    assert_eq!(g.kv_head_of(39), Some(7));
}

#[test]
fn zero_counts_are_refused() {
    assert_eq!(
        AttentionGeometry::new(0, 4, 64),
        Err(AttentionGeometryError::ZeroQueryHeads)
    );
    assert_eq!(
        AttentionGeometry::new(32, 0, 64),
        Err(AttentionGeometryError::ZeroKvHeads)
    );
    assert_eq!(
        AttentionGeometry::new(32, 4, 0),
        Err(AttentionGeometryError::ZeroHeadDim)
    );
}

#[test]
fn non_divisible_heads_are_refused() {
    assert_eq!(
        AttentionGeometry::new(10, 4, 64),
        Err(AttentionGeometryError::QueryHeadsNotDivisible {
            num_q_heads: 10,
            num_kv_heads: 4
        })
    );
    // more kv heads than query heads is also not a whole ratio
    assert_eq!(
        AttentionGeometry::new(4, 8, 64),
        Err(AttentionGeometryError::QueryHeadsNotDivisible {
            num_q_heads: 4,
            num_kv_heads: 8
        })
    );
}

#[test]
fn overflow_is_refused_not_wrapped() {
    let heads = usize::MAX / 2 + 1;
    assert_eq!(
        AttentionGeometry::new(heads, 1, 4),
        Err(AttentionGeometryError::Overflow { heads, head_dim: 4 })
    );
    // largest product that still fits is accepted
    let g = AttentionGeometry::new(usize::MAX / 64, 1, 64).unwrap();
    assert_eq!(g.q_dim(), (usize::MAX / 64) * 64);
}

#[test]
fn kv_pool_mismatch_is_named() {
    let g = AttentionGeometry::new(32, 4, 64).unwrap();
    assert_eq!(g.check_kv_pool(4, 64), Ok(()));
    assert_eq!(
        g.check_kv_pool(8, 64),
        Err(AttentionGeometryError::KvPoolMismatch {
            geometry_kv_heads: 4,
            geometry_head_dim: 64,
            pool_kv_heads: 8,
            pool_head_dim: 64,
        })
    );
    let msg = g.check_kv_pool(4, 128).unwrap_err().to_string();
    assert!(msg.contains("4 kv heads x head_dim 64") && msg.contains("4 kv heads x head_dim 128"));
}

#[test]
fn model_config_with_bad_heads_is_refused() {
    let mut c = ModelConfig::tinyllama_1_1b();
    c.num_kv_heads = 6;
    assert_eq!(
        c.attention_geometry(),
        Err(AttentionGeometryError::QueryHeadsNotDivisible {
            num_q_heads: 32,
            num_kv_heads: 6
        })
    );
}

#[test]
fn serde_round_trip_rechecks() {
    let g = AttentionGeometry::new(32, 4, 128).unwrap();
    let json = serde_json::to_string(&g).unwrap();
    let back: AttentionGeometry = serde_json::from_str(&json).unwrap();
    assert_eq!(back, g);
    let bad = serde_json::from_str::<AttentionGeometry>(
        r#"{"num_q_heads":10,"num_kv_heads":4,"head_dim":64}"#,
    );
    assert!(bad.is_err(), "deserialising must re-run the checks");
}
