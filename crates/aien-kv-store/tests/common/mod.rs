#![allow(dead_code)]
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

use aien_kv_store::{
    BackendFormat, ChunkHeader, ChunkKey, ComputeConfig, EngineConfig, EngineIdentity, HiddenAct,
    KvDtype, LayoutStrides, NamespaceFields, RopeScaling, WeightFormat,
};

/// Llama-3.2-1B-like compute configuration (ADR 1.2.1), the base of every
/// `sample()`.
pub fn sample_compute() -> ComputeConfig<'static> {
    ComputeConfig {
        num_heads: 32,
        hidden_dim: 2048,
        intermediate_dim: 8192,
        vocab_size: 128_256,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
        hidden_act: HiddenAct::Silu,
        arith_id: "ReferenceCpuBackend",
        arith_epoch: 1,
    }
}

/// A dense model on the host reference path (ADR 1.2.2): no experts, every
/// run-time switch off.
pub fn sample_engine_config() -> EngineConfig<'static> {
    EngineConfig {
        num_experts: 0,
        experts_per_token: 0,
        expert_intermediate_dim: 0,
        whole_prompt_prefill: false,
        device_router: false,
        matmul_path: "host-tensor-backend",
    }
}

/// The engine identity of every `sample()` (ADR 1.2.2).
pub fn sample_engine() -> EngineIdentity<'static> {
    EngineIdentity {
        weight_format: WeightFormat::AsFile,
        engine_build_id: "aien-runtime/0.1.0+git:test",
        rounding_class: "ReferenceCpuBackend@kv-arith-epoch-1",
        engine_config_digest: sample_engine_config().digest().unwrap(),
    }
}

/// Llama-3.2-1B-like host geometry (16 layers, 8 kv heads, head_dim 64, bf16,
/// 16-token blocks, 16-block chunks = 256 tokens).
pub fn sample() -> NamespaceFields<'static> {
    NamespaceFields {
        model_sha256: [0xAB; 32],
        model_digest_kind: "sha256-file",
        model_id: "llama-3.2-1b-instruct",
        num_layers: 16,
        num_kv_heads: 8,
        head_dim: 64,
        block_size: 16,
        rope_theta: 500000.0,
        rope_scaling: Some(RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }),
        qk_norm: false,
        dtype: KvDtype::Bf16,
        layout: LayoutStrides {
            head_stride: 128,
            token_stride: 1024,
            kv_plane_stride: 16384,
            layer_stride: 32768,
            block_stride: 524288,
            element_bytes: 2,
        },
        backend_format: BackendFormat::HostPagedPoolV1,
        chunk_tokens: 256,
        sharing_namespace: "default",
        compute_config_digest: sample_compute()
            .digest_with_engine(&sample_engine())
            .unwrap(),
    }
}

/// Small geometry for chain tests: 4-token blocks, 8-token chunks.
pub fn small() -> NamespaceFields<'static> {
    let mut f = sample();
    f.block_size = 4;
    f.chunk_tokens = 8;
    f.layout.kv_plane_stride = 4 * 1024;
    f.layout.layer_stride = 8 * 1024;
    f.layout.block_stride = 16 * 8 * 1024;
    f
}

pub fn hex(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 15) as usize] as char);
    }
    s
}

pub fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

pub fn unhex32(s: &str) -> [u8; 32] {
    unhex(s).try_into().expect("32 bytes")
}

/// A fully populated, internally consistent header for the `sample()` geometry
/// (first chunk, full 256 tokens).
pub fn sample_chunk_header() -> ChunkHeader {
    let f = sample();
    let ns = f.namespace_id().unwrap();
    let tokens: Vec<u32> = (0..256).collect();
    let key = aien_kv_store::chunk_key(&ns, None, 0, &tokens).unwrap();
    f.chunk_header(&ns, key, None, 0, 256, 1_700_000_000_000_000_000)
        .unwrap()
}

pub fn key(b: u8) -> ChunkKey {
    ChunkKey(aien_platform::ObjectId([b; 32]))
}

/// Deterministic segment digests: SHA-256 of "<tag>/seg/<j>".
pub fn seg_digests(tag: &str, n: u32) -> Vec<[u8; 32]> {
    use sha2::{Digest, Sha256};
    (0..n)
        .map(|j| Sha256::digest(format!("{tag}/seg/{j}").as_bytes()).into())
        .collect()
}

/// A consistent record for the `sample()` geometry: 16 segments of one pool
/// block each, `payload_sha256` = the root over them.
pub fn sample_record() -> (aien_kv_store::RecordHeader, Vec<[u8; 32]>) {
    let chunk = sample_chunk_header();
    let n = sample().segment_count(chunk.token_count).unwrap();
    let digests = seg_digests("sample", n);
    let rec = aien_kv_store::RecordHeader {
        chunk,
        payload_sha256: aien_kv_store::payload_root(&digests),
    };
    (rec, digests)
}
#[cfg(feature = "std")]
pub mod rig;
