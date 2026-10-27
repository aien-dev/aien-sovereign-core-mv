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

//! C6a (sc#484, ADR 0003 section 1.2.1, review finding 1): what the stored
//! bytes were computed with must be part of their identity.
//!
//! - `modelconfig_field_mutation_changes_compute_digest`: every `ModelConfig`
//!   field is mutated one at a time. The test destructures `ModelConfig`
//!   without `..`, so a new field is a compile error here until its author
//!   decides "bound" or "excluded with a reason" (the production conversion in
//!   `compute_config.rs` has the same rule).
//! - `arith_epoch_pinned_bytes`: a fixed tiny model and prompt run on the
//!   reference CPU backend; the SHA-256 of the K/V it writes is pinned next to
//!   `KV_ARITH_EPOCH`. A numeric change fails this test until its author bumps
//!   the epoch and re-pins both values, which changes every namespace id.

use aien_abi_core::{Llama3RopeScaling, ModelConfig};
use aien_inference_abi::{
    f32_to_bf16_bits, AienInferenceBackend, DecodeOutput, KvDType, KvPoolConfig,
    NativeTransformerBackend, ReferenceCpuBackend, SamplingParams, ScheduledBatch, SequenceRequest,
    TransformerWeights, KV_ARITH_EPOCH,
};
use aien_kv_cache::create_shared_kv_manager_with_pool;
use aien_kv_store::{BackendFormat, KvDtype, LayoutStrides};
use aien_runtime::compute_config::{
    compute_config_from, HostEngineIdentity, ModelBinding, StoreInputs,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

const ARITH_ID: &str = "ReferenceCpuBackend";

fn base() -> ModelConfig {
    ModelConfig {
        model_id: "tiny-test".to_string(),
        max_sequence_length: 2048,
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
        rope_scaling: Some(Llama3RopeScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }),
        tie_word_embeddings: false,
        eos_token_ids: vec![2],
        qk_norm: false,
    }
}

fn store() -> StoreInputs<'static> {
    StoreInputs {
        model_sha256: [0xAB; 32],
        model_digest_kind: "sha256-file",
        dtype: KvDtype::Bf16,
        layout: LayoutStrides {
            head_stride: 32,
            token_stride: 64,
            kv_plane_stride: 1024,
            layer_stride: 2048,
            block_stride: 4096,
            element_bytes: 2,
        },
        backend_format: BackendFormat::HostPagedPoolV1,
        chunk_tokens: 256,
        sharing_namespace: "default",
    }
}

/// (compute digest, namespace id) for a config.
fn ids(cfg: &ModelConfig, arith_id: &str) -> ([u8; 32], [u8; 32]) {
    let binding = ModelBinding::from_model(cfg, arith_id).expect("binding");
    let digest = binding.compute.digest().expect("digest");
    let engine = HostEngineIdentity::reference(arith_id).expect("engine identity");
    let fields = binding
        .namespace_fields(&store(), &engine.identity())
        .expect("fields");
    assert_eq!(
        fields.compute_config_digest,
        binding
            .compute
            .digest_with_engine(&engine.identity())
            .expect("bound digest"),
        "compute config plus engine identity is field 14"
    );
    (digest, fields.namespace_id().expect("namespace").0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Effect {
    /// In the compute-config digest (ADR 1.2.1 table); the namespace id moves too.
    Digest,
    /// Carried by namespace fields 4 to 8; the digest does not move.
    NamespaceOnly,
    /// Written exclusion (ADR 1.2.1 "Fields not listed and why"): neither moves.
    Excluded,
}

#[test]
fn modelconfig_field_mutation_changes_compute_digest() {
    let b = base();
    // Exhaustive: no `..`. A new ModelConfig field fails to compile here.
    let ModelConfig {
        model_id,
        max_sequence_length,
        block_size,
        num_layers,
        num_heads,
        head_dim,
        num_kv_heads,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        rope_scaling,
        tie_word_embeddings,
        eos_token_ids,
        qk_norm,
    } = b.clone();
    let scaling = rope_scaling.expect("base has llama3 scaling");

    type Mutation = (&'static str, Effect, Box<dyn Fn(&mut ModelConfig)>);
    let mut cases: Vec<Mutation> = vec![
        ("model_id", Effect::NamespaceOnly, {
            let v = format!("{model_id}-x");
            Box::new(move |c| c.model_id = v.clone())
        }),
        ("max_sequence_length", Effect::Excluded, {
            Box::new(move |c| c.max_sequence_length = max_sequence_length * 2)
        }),
        ("block_size", Effect::NamespaceOnly, {
            Box::new(move |c| c.block_size = block_size * 2)
        }),
        ("num_layers", Effect::NamespaceOnly, {
            Box::new(move |c| c.num_layers = num_layers + 1)
        }),
        ("num_heads", Effect::Digest, {
            Box::new(move |c| c.num_heads = num_heads + 1)
        }),
        ("head_dim", Effect::NamespaceOnly, {
            Box::new(move |c| c.head_dim = head_dim + 1)
        }),
        ("num_kv_heads", Effect::NamespaceOnly, {
            Box::new(move |c| c.num_kv_heads = num_kv_heads + 1)
        }),
        ("hidden_dim", Effect::Digest, {
            Box::new(move |c| c.hidden_dim = hidden_dim + 1)
        }),
        ("intermediate_dim", Effect::Digest, {
            Box::new(move |c| c.intermediate_dim = intermediate_dim + 1)
        }),
        ("vocab_size", Effect::Digest, {
            Box::new(move |c| c.vocab_size = vocab_size + 1)
        }),
        ("rms_norm_eps", Effect::Digest, {
            Box::new(move |c| c.rms_norm_eps = rms_norm_eps / 10.0)
        }),
        ("rope_theta", Effect::NamespaceOnly, {
            Box::new(move |c| c.rope_theta = rope_theta * 2.0)
        }),
        ("rope_scaling (removed)", Effect::NamespaceOnly, {
            Box::new(|c| c.rope_scaling = None)
        }),
        ("rope_scaling.factor", Effect::NamespaceOnly, {
            Box::new(move |c| {
                c.rope_scaling = Some(Llama3RopeScaling {
                    factor: scaling.factor + 1.0,
                    ..scaling
                })
            })
        }),
        ("rope_scaling.low_freq_factor", Effect::NamespaceOnly, {
            Box::new(move |c| {
                c.rope_scaling = Some(Llama3RopeScaling {
                    low_freq_factor: scaling.low_freq_factor + 1.0,
                    ..scaling
                })
            })
        }),
        ("rope_scaling.high_freq_factor", Effect::NamespaceOnly, {
            Box::new(move |c| {
                c.rope_scaling = Some(Llama3RopeScaling {
                    high_freq_factor: scaling.high_freq_factor + 1.0,
                    ..scaling
                })
            })
        }),
        (
            "rope_scaling.original_max_position_embeddings",
            Effect::NamespaceOnly,
            Box::new(move |c| {
                c.rope_scaling = Some(Llama3RopeScaling {
                    original_max_position_embeddings: scaling.original_max_position_embeddings + 1,
                    ..scaling
                })
            }),
        ),
        ("tie_word_embeddings", Effect::Digest, {
            Box::new(move |c| c.tie_word_embeddings = !tie_word_embeddings)
        }),
        ("eos_token_ids", Effect::Excluded, {
            let mut v = eos_token_ids.clone();
            v.push(99);
            Box::new(move |c| c.eos_token_ids = v.clone())
        }),
        ("qk_norm", Effect::NamespaceOnly, {
            Box::new(move |c| c.qk_norm = !qk_norm)
        }),
    ];
    // Every ModelConfig field name appears in the table (the destructure above
    // guarantees the list of names is complete; this guards the table).
    for field in [
        "model_id",
        "max_sequence_length",
        "block_size",
        "num_layers",
        "num_heads",
        "head_dim",
        "num_kv_heads",
        "hidden_dim",
        "intermediate_dim",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling (removed)",
        "tie_word_embeddings",
        "eos_token_ids",
        "qk_norm",
    ] {
        assert!(
            cases.iter().any(|c| c.0 == field),
            "no mutation for {field}"
        );
    }

    let (base_digest, base_ns) = ids(&b, ARITH_ID);
    for (name, effect, mutate) in cases.drain(..) {
        let mut c = b.clone();
        mutate(&mut c);
        assert_ne!(c, b, "{name}: the mutation did not change the config");
        let (digest, ns) = ids(&c, ARITH_ID);
        match effect {
            Effect::Digest => {
                assert_ne!(digest, base_digest, "{name} must change the compute digest");
                assert_ne!(ns, base_ns, "{name} must change the namespace id");
            }
            Effect::NamespaceOnly => {
                assert_eq!(digest, base_digest, "{name} is not a compute-digest field");
                assert_ne!(ns, base_ns, "{name} must change the namespace id");
            }
            Effect::Excluded => {
                assert_eq!(digest, base_digest, "{name} is written-excluded");
                assert_eq!(ns, base_ns, "{name} is written-excluded");
            }
        }
    }
}

#[test]
fn arith_identity_and_epoch_change_the_compute_digest() {
    let b = base();
    let (d0, n0) = ids(&b, ARITH_ID);
    let (d1, n1) = ids(&b, "NativeCpuBackend (Rayon Multi-Core)");
    assert_ne!(d0, d1, "a different tensor backend is a different digest");
    assert_ne!(n0, n1);
    let mut cc = compute_config_from(&b, ARITH_ID).unwrap();
    assert_eq!(
        cc.arith_epoch, KV_ARITH_EPOCH,
        "the epoch comes from the constant"
    );
    let before = cc.digest().unwrap();
    cc.arith_epoch += 1;
    assert_ne!(
        cc.digest().unwrap(),
        before,
        "bumping the epoch changes the digest"
    );
}

#[test]
fn conversion_refuses_values_that_do_not_fit_the_wire_width() {
    let mut c = base();
    // 2^32 heads cannot be written as the u32 of ADR 1.2.1 field 1.
    c.num_heads = u32::MAX as usize + 1;
    assert!(compute_config_from(&c, ARITH_ID).is_err());
    let mut c = base();
    c.num_layers = u32::MAX as usize + 1;
    assert!(ModelBinding::from_model(&c, ARITH_ID).is_err());
}

// --- arith_epoch_pinned_bytes ------------------------------------------------

/// The epoch the pin below was taken at, and the SHA-256 of the K/V bytes.
const PINNED_EPOCH: u32 = 2;
const PINNED_KV_SHA256: &str = "48340604c2513464fde46789d95b2473e87fdb057201822ff207b5e1acf4872c";

fn pin_model() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        block_size: 16,
        ..Default::default()
    }
}

/// Runs the fixed prompt on the reference backend in a bf16 pool (the daemon's
/// pool dtype) and hashes every written K and V value as bf16 bits, in
/// (layer, position, K then V) order.
async fn kv_bytes_sha256() -> String {
    let cfg = pin_model();
    let pool = KvPoolConfig::for_model(
        16,
        16,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        KvDType::Bf16,
    );
    let kv = create_shared_kv_manager_with_pool(16, 16, pool).expect("pool");
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        TransformerWeights::reference_test_weights(&cfg),
        Arc::new(ReferenceCpuBackend::new()),
        kv.clone(),
    );
    let id = 0x1_0000_0001u64;
    let prompt: Vec<u32> = (0..40u32).map(|i| (i * 37 + 11) % 253 + 3).collect();
    let prefill = ScheduledBatch {
        prefill_requests: vec![SequenceRequest {
            request_id: id,
            prompt_tokens: prompt,
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 8,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        }],
        decode_requests: vec![],
        block_tables: HashMap::new(),
        step_id: 1,
    };
    let (out, _) = backend.execute_step(&prefill).await.unwrap();
    assert!(matches!(out[0], DecodeOutput::Token { .. }));
    for step in 2..4 {
        let decode = ScheduledBatch {
            prefill_requests: vec![],
            decode_requests: vec![id],
            block_tables: HashMap::new(),
            step_id: step,
        };
        backend.execute_step(&decode).await.unwrap();
    }

    let r = kv.read();
    let table = r.get_block_table(id).expect("table");
    let pool = r.tensor_pool().expect("pool");
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let mut h = Sha256::new();
    for layer in 0..cfg.num_layers {
        for pos in 0..table.total_tokens {
            let (mut k, mut v) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
            pool.read_token_kv(table.block_ids[pos / 16], layer, pos % 16, &mut k, &mut v);
            for x in k.iter().chain(v.iter()) {
                h.update(f32_to_bf16_bits(*x).to_le_bytes());
            }
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[tokio::test]
async fn arith_epoch_pinned_bytes() {
    let actual = kv_bytes_sha256().await;
    assert_eq!(
        KV_ARITH_EPOCH, PINNED_EPOCH,
        "KV_ARITH_EPOCH changed: re-run, then pin the new epoch and hash together"
    );
    assert_eq!(
        actual, PINNED_KV_SHA256,
        "the forward pass now writes different K/V bytes. If this change is intended, \
         bump KV_ARITH_EPOCH in aien-inference-abi (every stored chunk becomes a different \
         namespace) and pin the new epoch and hash here."
    );
}

/// sc#504 (ADR 1.2.2): the host reference path fills the engine identity
/// honestly: weights used as the file holds them, a build id naming this
/// crate's version and commit, a rounding class naming the backend and
/// `KV_ARITH_EPOCH`, and a dense engine config with every switch off.
#[test]
fn host_reference_engine_identity_names_backend_epoch_and_build() {
    let e = HostEngineIdentity::reference(ARITH_ID).expect("engine identity");
    let id = e.identity();
    assert_eq!(id.weight_format, aien_kv_store::WeightFormat::AsFile);
    assert!(
        id.engine_build_id
            .starts_with(&format!("aien-runtime/{}+git:", env!("CARGO_PKG_VERSION"))),
        "build id {:?}",
        id.engine_build_id
    );
    assert_eq!(
        id.rounding_class,
        format!("{ARITH_ID}@kv-arith-epoch-{KV_ARITH_EPOCH}")
    );
    let dense = aien_kv_store::EngineConfig {
        num_experts: 0,
        experts_per_token: 0,
        expert_intermediate_dim: 0,
        whole_prompt_prefill: false,
        device_router: false,
        matmul_path: "host-tensor-backend",
    };
    assert_eq!(id.engine_config_digest, dense.digest().unwrap());
    // Another backend is another rounding class.
    let other = HostEngineIdentity::reference("NativeCpuBackend (Rayon Multi-Core)").unwrap();
    assert_ne!(other.identity().rounding_class, id.rounding_class);
}

/// Each engine input changed alone moves the namespace bound by
/// `ModelBinding::namespace_fields` (the daemon path).
#[test]
fn each_engine_input_changes_the_bound_namespace() {
    let b = base();
    let binding = ModelBinding::from_model(&b, ARITH_ID).expect("binding");
    let host = HostEngineIdentity::reference(ARITH_ID).expect("engine identity");
    let ns = |e: &aien_kv_store::EngineIdentity| {
        binding
            .namespace_fields(&store(), e)
            .expect("fields")
            .namespace_id()
            .expect("namespace")
    };
    let base_ns = ns(&host.identity());
    let mut wf = host.identity();
    wf.weight_format = aien_kv_store::WeightFormat::Fp8All;
    let mut build = host.identity();
    build.engine_build_id = "aien-runtime/0.0.0+git:other";
    let mut round = host.identity();
    round.rounding_class = "ReferenceCpuBackend@kv-arith-epoch-999";
    let mut cfg = host.identity();
    cfg.engine_config_digest[0] ^= 1;
    let mut seen = vec![base_ns];
    for (name, e) in [
        ("weight_format", wf),
        ("engine_build_id", build),
        ("rounding_class", round),
        ("engine_config_digest", cfg),
    ] {
        let got = ns(&e);
        assert!(!seen.contains(&got), "{name} did not move the namespace");
        seen.push(got);
    }
}
