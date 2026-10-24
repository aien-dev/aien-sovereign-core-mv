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

// Test cases adapted (re-expressed in Rust, none copied) from LMCache
// tests/v1/distributed/test_nixl_store_dynamic_l2_adapter.py at 4d69cdc4,
// Apache-2.0, Copyright LMCache contributors:
//   test_salted_objects_recover_independently_after_restart (line 745)
// See THIRD_PARTY.md and ADR 0003 context item 24.

mod common;

use aien_kv_store::{
    chunk_chain, chunk_key, validate_record, BackendFormat, ComputeConfig, EngineConfig,
    EngineIdentity, KvDtype, KvError, NamespaceFields, NamespaceId, RecordHeader, RopeScaling,
    WeightFormat,
};
use common::*;
use sha2::{Digest, Sha256};

type Mutator = (&'static str, fn(&mut NamespaceFields<'static>));

fn mutators() -> Vec<Mutator> {
    vec![
        ("model_sha256", |f| f.model_sha256[31] ^= 1),
        ("model_digest_kind", |f| f.model_digest_kind = "blake3-file"),
        ("model_id", |f| f.model_id = "llama-3.2-3b-instruct"),
        ("num_layers", |f| f.num_layers += 1),
        ("num_kv_heads", |f| f.num_kv_heads += 1),
        ("head_dim", |f| f.head_dim += 1),
        ("block_size", |f| f.block_size = 8),
        ("rope_theta", |f| f.rope_theta = 10000.0),
        ("rope_scaling_none", |f| f.rope_scaling = None),
        ("rope_factor", |f| {
            f.rope_scaling.as_mut().unwrap().factor = 8.0
        }),
        ("rope_low", |f| {
            f.rope_scaling.as_mut().unwrap().low_freq_factor = 2.0
        }),
        ("rope_high", |f| {
            f.rope_scaling.as_mut().unwrap().high_freq_factor = 8.0
        }),
        ("rope_original_max", |f| {
            f.rope_scaling
                .as_mut()
                .unwrap()
                .original_max_position_embeddings = 4096
        }),
        ("qk_norm", |f| f.qk_norm = true),
        ("dtype", |f| f.dtype = KvDtype::Fp16),
        ("layout_head_stride", |f| f.layout.head_stride += 2),
        ("layout_token_stride", |f| f.layout.token_stride += 2),
        ("layout_kv_plane_stride", |f| f.layout.kv_plane_stride += 2),
        ("layout_layer_stride", |f| f.layout.layer_stride += 2),
        ("layout_block_stride", |f| f.layout.block_stride += 2),
        ("layout_element_bytes", |f| f.layout.element_bytes = 4),
        ("backend_format", |f| {
            f.backend_format = BackendFormat::NativeF32DecodeStateV1
        }),
        ("chunk_tokens", |f| f.chunk_tokens = 512),
        ("sharing_namespace", |f| f.sharing_namespace = "tenant-b"),
        ("compute_config_digest", |f| f.compute_config_digest[0] ^= 1),
    ]
}

#[test]
fn namespace_changes_when_any_field_changes() {
    let base = sample().namespace_id().unwrap();
    let mut seen: Vec<(&str, NamespaceId)> = vec![("base", base)];
    for (name, mutate) in mutators() {
        let mut f = sample();
        mutate(&mut f);
        let id = f.namespace_id().unwrap();
        for (other, other_id) in &seen {
            assert_ne!(id, *other_id, "{name} collides with {other}");
        }
        seen.push((name, id));
    }
    // Same inputs, same id (deterministic).
    assert_eq!(base, sample().namespace_id().unwrap());
    // Every ADR 1.2 field is covered: 14 field rows expand to 25 mutators (the
    // 24 of C1 plus field 14), and the base is the 26th id.
    assert_eq!(seen.len(), 26);
}

#[test]
fn rope_scaling_presence_is_distinct_from_zeroed_scaling() {
    let mut a = sample();
    a.rope_scaling = None;
    let mut b = sample();
    b.rope_scaling = Some(RopeScaling {
        factor: 0.0,
        low_freq_factor: 0.0,
        high_freq_factor: 0.0,
        original_max_position_embeddings: 0,
    });
    assert_ne!(a.namespace_id().unwrap(), b.namespace_id().unwrap());
}

#[test]
fn namespace_refuses_oversized_or_inconsistent_fields() {
    let mut f = sample();
    let long = "x".repeat(65);
    // Leak is fine in a test: NamespaceFields borrows with a 'static lifetime here.
    f.sharing_namespace = Box::leak(long.into_boxed_str());
    assert_eq!(
        f.namespace_id(),
        Err(KvError::FieldTooLong("sharing_namespace"))
    );

    let mut f = sample();
    f.model_digest_kind = Box::leak("k".repeat(256).into_boxed_str());
    assert_eq!(
        f.namespace_id(),
        Err(KvError::FieldTooLong("model_digest_kind"))
    );

    let mut f = sample();
    f.chunk_tokens = 250; // not a multiple of block_size 16
    assert_eq!(f.namespace_id(), Err(KvError::InvalidGeometry));
    let mut f = sample();
    f.block_size = 0;
    assert_eq!(f.namespace_id(), Err(KvError::InvalidGeometry));
    let mut f = sample();
    f.chunk_tokens = 0;
    assert_eq!(f.namespace_id(), Err(KvError::InvalidGeometry));
}

#[test]
fn chunk_key_chain_depends_on_parent_and_position() {
    let ns = small().namespace_id().unwrap();
    let toks: Vec<u32> = (100..108).collect();
    let k0 = chunk_key(&ns, None, 0, &toks).unwrap();

    // Parent matters.
    let k_parent_a = chunk_key(&ns, Some(&key(1)), 8, &toks).unwrap();
    let k_parent_b = chunk_key(&ns, Some(&key(2)), 8, &toks).unwrap();
    assert_ne!(k_parent_a, k_parent_b);
    // Position matters even with identical tokens and parent.
    let k_pos_8 = chunk_key(&ns, Some(&key(1)), 8, &toks).unwrap();
    let k_pos_16 = chunk_key(&ns, Some(&key(1)), 16, &toks).unwrap();
    assert_ne!(k_pos_8, k_pos_16);
    // Namespace matters.
    let other_ns = {
        let mut f = small();
        f.sharing_namespace = "tenant-b";
        f.namespace_id().unwrap()
    };
    assert_ne!(k0, chunk_key(&other_ns, None, 0, &toks).unwrap());
    // Tokens matter, and so does token count (partial vs full prefix).
    let mut toks2 = toks.clone();
    toks2[7] += 1;
    assert_ne!(k0, chunk_key(&ns, None, 0, &toks2).unwrap());
    assert_ne!(k0, chunk_key(&ns, None, 0, &toks[..7]).unwrap());
    // First chunk uses an all-zero parent.
    assert_eq!(
        k0,
        chunk_key(
            &ns,
            Some(&aien_kv_store::ChunkKey(aien_platform::ObjectId([0; 32]))),
            0,
            &toks
        )
        .unwrap()
    );
}

#[test]
fn chunk_chain_links_each_chunk_to_all_earlier_tokens() {
    let ns = small().namespace_id().unwrap();
    let toks: Vec<u32> = (0..20).collect(); // chunks of 8: 8, 8, 4 (partial tail)
    let chain = chunk_chain(&ns, 8, &toks).unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0], chunk_key(&ns, None, 0, &toks[0..8]).unwrap());
    assert_eq!(
        chain[1],
        chunk_key(&ns, Some(&chain[0]), 8, &toks[8..16]).unwrap()
    );
    assert_eq!(
        chain[2],
        chunk_key(&ns, Some(&chain[1]), 16, &toks[16..20]).unwrap()
    );

    // Changing one token in chunk 0 changes every later key.
    let mut changed = toks.clone();
    changed[0] = 999;
    let chain2 = chunk_chain(&ns, 8, &changed).unwrap();
    assert!(chain.iter().zip(&chain2).all(|(a, b)| a != b));

    // Changing a token in the tail only changes the tail.
    let mut tail = toks.clone();
    tail[19] = 999;
    let chain3 = chunk_chain(&ns, 8, &tail).unwrap();
    assert_eq!(chain[..2], chain3[..2]);
    assert_ne!(chain[2], chain3[2]);

    // A longer prompt shares the full-chunk prefix keys with a shorter one.
    let longer: Vec<u32> = (0..28).collect();
    let chain4 = chunk_chain(&ns, 8, &longer).unwrap();
    assert_eq!(chain[..2], chain4[..2]);
    // The 4-token tail of the short prompt does not match a full 8-token chunk.
    assert_ne!(chain[2], chain4[2]);

    assert!(chunk_chain(&ns, 8, &[]).unwrap().is_empty());
    assert_eq!(chunk_chain(&ns, 0, &toks), Err(KvError::InvalidGeometry));
}

/// Adapted idea: LMCache `test_salted_objects_recover_independently_after_restart`.
/// Two salts over the same tokens never share keys, and recomputing after a
/// "restart" (fresh derivation from scratch) reproduces each salt's own keys.
#[test]
fn salted_namespaces_stay_independent_after_recomputation() {
    let toks: Vec<u32> = (0..16).collect();
    let derive = |salt: &'static str| {
        let mut f = small();
        f.sharing_namespace = salt;
        let ns = f.namespace_id().unwrap();
        chunk_chain(&ns, 8, &toks).unwrap()
    };
    let a1 = derive("salt-a");
    let b1 = derive("salt-b");
    let a2 = derive("salt-a");
    let b2 = derive("salt-b");
    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    for (x, y) in a1.iter().zip(&b1) {
        assert_ne!(x, y);
    }
}

type CfgMutator = (&'static str, fn(&mut ComputeConfig<'static>));

/// The eight `ComputeConfig` fields that have more than one legal value today.
/// The ninth, `hidden_act`, has the single value `Silu` (the loader refuses
/// every other activation, ADR 1.2.1), so it is mutated at the byte level in
/// `compute_config_digest_changes_with_each_field`.
fn compute_mutators() -> Vec<CfgMutator> {
    vec![
        ("num_heads", |c| c.num_heads = 24),
        ("hidden_dim", |c| c.hidden_dim = 3072),
        ("intermediate_dim", |c| c.intermediate_dim = 8193),
        ("vocab_size", |c| c.vocab_size = 128_257),
        ("rms_norm_eps", |c| c.rms_norm_eps = 1e-6),
        ("tie_word_embeddings", |c| c.tie_word_embeddings = false),
        ("arith_id", |c| {
            c.arith_id = "NativeCpuBackend (Rayon Multi-Core)"
        }),
        ("arith_epoch", |c| c.arith_epoch = 2),
    ]
}

/// The digest bytes assembled by hand from ADR 1.2.1 / section 12 I3, with the
/// activation byte given explicitly so the test can flip it.
fn manual_digest(c: &ComputeConfig, act_byte: u8) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIEN-PKV-CFG-V1\0");
    h.update(c.num_heads.to_le_bytes());
    h.update(c.hidden_dim.to_le_bytes());
    h.update(c.intermediate_dim.to_le_bytes());
    h.update(c.vocab_size.to_le_bytes());
    h.update(c.rms_norm_eps.to_bits().to_le_bytes());
    h.update([u8::from(c.tie_word_embeddings)]);
    h.update([act_byte]);
    h.update([c.arith_id.len() as u8]);
    h.update(c.arith_id.as_bytes());
    h.update(c.arith_epoch.to_le_bytes());
    h.finalize().into()
}

#[test]
fn compute_config_digest_changes_with_each_field() {
    let base = sample_compute();
    let base_digest = base.digest().unwrap();
    // The implementation equals the byte-by-byte formula of the ADR.
    assert_eq!(base_digest, manual_digest(&base, 1));
    assert_eq!(base_digest, sample_compute().digest().unwrap());

    let mut seen = vec![("base", base_digest)];
    for (name, mutate) in compute_mutators() {
        let mut c = sample_compute();
        mutate(&mut c);
        let d = c.digest().unwrap();
        assert_eq!(d, manual_digest(&c, 1), "{name}: matches the formula");
        for (other, other_d) in &seen {
            assert_ne!(d, *other_d, "{name} collides with {other}");
        }
        seen.push((name, d));
    }
    // Ninth field: the activation byte is part of the digest (1 = silu).
    let act2 = manual_digest(&base, 2);
    assert!(seen.iter().all(|(_, d)| *d != act2));
    assert_eq!(
        seen.len(),
        9,
        "8 struct mutators + base; hidden_act is the 9th"
    );

    // A length-prefixed string cannot alias its neighbour: "ab"+epoch vs "a"+...
    let mut a = sample_compute();
    a.arith_id = "";
    let mut b = sample_compute();
    b.arith_id = "x";
    assert_ne!(a.digest().unwrap(), b.digest().unwrap());
}

#[test]
fn compute_config_refuses_oversized_arith_id() {
    let mut c = sample_compute();
    c.arith_id = Box::leak("a".repeat(256).into_boxed_str());
    assert_eq!(c.digest(), Err(KvError::FieldTooLong("arith_id")));
    c.arith_id = Box::leak("a".repeat(255).into_boxed_str());
    assert!(c.digest().is_ok());
}

/// Review finding 1: two model directories with the same weights and the same
/// fields 1 to 13 but a different `rms_norm_eps` must not share objects.
#[test]
fn rms_norm_eps_alone_changes_the_namespace() {
    let ns_for = |eps: f32| {
        let mut c = sample_compute();
        c.rms_norm_eps = eps;
        let mut f = sample();
        f.compute_config_digest = c.digest_with_engine(&sample_engine()).unwrap();
        f.namespace_id().unwrap()
    };
    let a = ns_for(1e-5);
    let b = ns_for(1e-6);
    assert_ne!(a, b);
    assert_eq!(a, ns_for(1e-5));
    // Everything else about the two fields structs is equal.
    let mut f1 = sample();
    f1.compute_config_digest = {
        let mut c = sample_compute();
        c.rms_norm_eps = 1e-5;
        c.digest_with_engine(&sample_engine()).unwrap()
    };
    assert_eq!(f1, sample());
    // And the chunk keys derived from them differ, so no object can be found.
    let toks: Vec<u32> = (0..8).collect();
    assert_ne!(
        chunk_key(&a, None, 0, &toks).unwrap(),
        chunk_key(&b, None, 0, &toks).unwrap()
    );
}

/// ADR 1.2: field 14 is appended after field 13 and every earlier field keeps
/// its position, so the namespace id equals a hand-assembled hash.
fn manual_namespace(f: &NamespaceFields, domain: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(1u16.to_le_bytes());
    h.update(f.model_sha256);
    h.update([f.model_digest_kind.len() as u8]);
    h.update(f.model_digest_kind.as_bytes());
    h.update((f.model_id.len() as u16).to_le_bytes());
    h.update(f.model_id.as_bytes());
    for v in [f.num_layers, f.num_kv_heads, f.head_dim, f.block_size] {
        h.update(v.to_le_bytes());
    }
    h.update(f.rope_theta.to_bits().to_le_bytes());
    let r = f.rope_scaling.unwrap();
    h.update([1u8]);
    h.update(r.factor.to_bits().to_le_bytes());
    h.update(r.low_freq_factor.to_bits().to_le_bytes());
    h.update(r.high_freq_factor.to_bits().to_le_bytes());
    h.update(r.original_max_position_embeddings.to_le_bytes());
    h.update([u8::from(f.qk_norm)]);
    h.update([f.dtype as u8]);
    let l = f.layout;
    for v in [
        l.head_stride,
        l.token_stride,
        l.kv_plane_stride,
        l.layer_stride,
        l.block_stride,
        l.element_bytes,
    ] {
        h.update(v.to_le_bytes());
    }
    h.update([f.backend_format as u8]);
    h.update(f.chunk_tokens.to_le_bytes());
    h.update((f.sharing_namespace.len() as u16).to_le_bytes());
    h.update(f.sharing_namespace.as_bytes());
    h.update(f.compute_config_digest); // field 14, raw 32 bytes
    h.finalize().into()
}

#[test]
fn namespace_id_appends_compute_digest_after_sharing_namespace() {
    let f = sample();
    let want = manual_namespace(&f, b"AIEN-PKV-NS-V2\0");
    assert_eq!(f.namespace_id().unwrap().0, want);
}

#[test]
fn segments_follow_the_backend_format() {
    // Host: one segment per pool block.
    let f = sample();
    assert_eq!(f.segment_bytes(256).unwrap(), 524_288);
    assert_eq!(f.segment_count(256).unwrap(), 16);
    assert_eq!(f.segment_count(17).unwrap(), 2); // 17 tokens = 2 blocks
    assert_eq!(f.segment_count(1).unwrap(), 1);
    assert_eq!(f.segment_count(0), Err(KvError::TokenCountOutOfRange));
    // Native: the whole payload is one segment (ADR 2.1 item 3).
    let mut n = sample();
    n.backend_format = BackendFormat::NativeF32DecodeStateV1;
    n.dtype = KvDtype::Fp32;
    n.layout.element_bytes = 4;
    n.chunk_tokens = 64;
    assert_eq!(
        n.segment_bytes(64).unwrap(),
        n.expected_payload_len(64).unwrap()
    );
    assert_eq!(n.segment_count(64).unwrap(), 1);
    assert_eq!(n.segment_count(3).unwrap(), 1);
}

#[test]
fn host_chunk_blocks_are_capped_at_119() {
    // (4064 - 256) / 32: the digest table ends where the seal slot starts.
    let mut f = sample();
    f.chunk_tokens = 119 * f.block_size;
    assert!(f.namespace_id().is_ok());
    assert_eq!(f.segment_count(f.chunk_tokens).unwrap(), 119);
    f.chunk_tokens = 120 * f.block_size;
    assert_eq!(f.namespace_id(), Err(KvError::TooManySegments));
    // Native has one segment, so no block cap applies.
    f.backend_format = BackendFormat::NativeF32DecodeStateV1;
    assert!(f.namespace_id().is_ok());
}

#[test]
fn zero_byte_factors_in_the_layout_are_refused() {
    let mut f = sample();
    f.backend_format = BackendFormat::HostPagedPoolV1;
    f.layout.block_stride = 0;
    assert_eq!(f.namespace_id(), Err(KvError::InvalidGeometry));
    assert_eq!(f.expected_payload_len(16), Err(KvError::InvalidGeometry));

    let mut f = sample();
    f.backend_format = BackendFormat::NativeF32DecodeStateV1;
    f.layout.element_bytes = 0;
    assert_eq!(f.namespace_id(), Err(KvError::InvalidGeometry));
    assert_eq!(f.expected_payload_len(16), Err(KvError::InvalidGeometry));
}

// ---------------------------------------------------------------------------
// sc#504: the engine that computed the bytes is part of their identity
// (ADR 1.2.2). Each engine field changed alone gives a different namespace,
// and a record written under the old namespace is refused by validation.
// ---------------------------------------------------------------------------

type EngineMutator = (&'static str, fn(&mut EngineIdentity<'static>));

/// One mutator per `EngineIdentity` field.
fn engine_mutators() -> Vec<EngineMutator> {
    vec![
        ("weight_format", |e| {
            e.weight_format = WeightFormat::Fp8Experts
        }),
        ("engine_build_id", |e| {
            e.engine_build_id = "omega@0123abcd+kernels:feedf00d"
        }),
        ("rounding_class", |e| {
            e.rounding_class = "omega-gb10@whole-prompt-v1"
        }),
        ("engine_config_digest", |e| e.engine_config_digest[0] ^= 1),
    ]
}

fn fields_with_engine(e: &EngineIdentity) -> NamespaceFields<'static> {
    let mut f = sample();
    f.compute_config_digest = sample_compute().digest_with_engine(e).unwrap();
    f
}

/// A record written under `old` and read back by a runtime whose identity is
/// `new` must be refused: same tokens, same position, other namespace.
fn assert_old_record_refused(old: &NamespaceFields, new: &NamespaceFields, name: &str) {
    let toks: Vec<u32> = (0..old.chunk_tokens).collect();
    let old_ns = old.namespace_id().unwrap();
    let new_ns = new.namespace_id().unwrap();
    let old_key = chunk_key(&old_ns, None, 0, &toks).unwrap();
    let new_key = chunk_key(&new_ns, None, 0, &toks).unwrap();
    assert_ne!(old_key, new_key, "{name}: chunk key must differ");
    let stored = old
        .chunk_header(&old_ns, old_key, None, 0, old.chunk_tokens, 1)
        .unwrap();
    let n_seg = old.segment_count(old.chunk_tokens).unwrap() as usize;
    let digests = vec![[0x5Au8; 32]; n_seg];
    let rec = RecordHeader {
        chunk: stored,
        payload_sha256: aien_kv_store::payload_root(&digests),
    };
    let region = rec.encode_region(&digests).unwrap();
    let file_size = 4096 + stored.payload_len;
    let seg = old.segment_bytes(old.chunk_tokens).unwrap();
    // Positive control: the writer's own identity accepts the record.
    validate_record(&region, file_size, &stored, seg).unwrap();
    // The new identity expects its own namespace and key: refused.
    let expected = new
        .chunk_header(&new_ns, old_key, None, 0, new.chunk_tokens, 1)
        .unwrap();
    assert_eq!(
        validate_record(&region, file_size, &expected, seg),
        Err(KvError::NamespaceMismatch),
        "{name}: a record of the old namespace was accepted"
    );
}

#[test]
fn each_engine_field_alone_changes_the_namespace_and_refuses_old_records() {
    let base = fields_with_engine(&sample_engine());
    assert_eq!(base, sample(), "sample() carries sample_engine()");
    let mut seen = vec![("base", base.namespace_id().unwrap())];
    for (name, mutate) in engine_mutators() {
        let mut e = sample_engine();
        mutate(&mut e);
        let f = fields_with_engine(&e);
        let id = f.namespace_id().unwrap();
        for (other, other_id) in &seen {
            assert_ne!(id, *other_id, "{name} collides with {other}");
        }
        seen.push((name, id));
        assert_old_record_refused(&base, &f, name);
        assert_old_record_refused(&f, &base, name);
    }
    assert_eq!(seen.len(), 5, "4 engine fields + base");
}

#[test]
fn weight_format_alone_changes_the_namespace() {
    let mut ids = Vec::new();
    for wf in [
        WeightFormat::AsFile,
        WeightFormat::Bf16,
        WeightFormat::Fp8Experts,
        WeightFormat::Fp8All,
    ] {
        let mut e = sample_engine();
        e.weight_format = wf;
        let id = fields_with_engine(&e).namespace_id().unwrap();
        assert!(!ids.contains(&id), "{wf:?} aliases another weight format");
        ids.push(id);
        assert_eq!(WeightFormat::from_u8(wf as u8), Some(wf));
    }
    assert_eq!(WeightFormat::Fp8Experts.as_str(), "fp8-experts");
    assert_eq!(WeightFormat::from_u8(0), None);
}

#[test]
fn engine_build_id_alone_changes_the_namespace() {
    let mut e = sample_engine();
    e.engine_build_id = "aien-runtime/0.1.0+git:other";
    let f = fields_with_engine(&e);
    assert_ne!(f.namespace_id().unwrap(), sample().namespace_id().unwrap());
    assert_old_record_refused(&sample(), &f, "engine_build_id");
}

#[test]
fn rounding_class_alone_changes_the_namespace() {
    let mut e = sample_engine();
    e.rounding_class = "ReferenceCpuBackend@kv-arith-epoch-2";
    let f = fields_with_engine(&e);
    assert_ne!(f.namespace_id().unwrap(), sample().namespace_id().unwrap());
    assert_old_record_refused(&sample(), &f, "rounding_class");
}

#[test]
fn engine_config_digest_alone_changes_the_namespace() {
    let mut e = sample_engine();
    e.engine_config_digest[31] ^= 0x80;
    let f = fields_with_engine(&e);
    assert_ne!(f.namespace_id().unwrap(), sample().namespace_id().unwrap());
    assert_old_record_refused(&sample(), &f, "engine_config_digest");
}

type EngCfgMutator = (&'static str, fn(&mut EngineConfig<'static>));

/// MoE shape and run-time numeric switches (ADR 1.2.2): whole-prompt prefill
/// changes activation rounding, so it changes the K/V bytes.
fn engine_config_mutators() -> Vec<EngCfgMutator> {
    vec![
        ("num_experts", |c| c.num_experts = 128),
        ("experts_per_token", |c| c.experts_per_token = 8),
        ("expert_intermediate_dim", |c| {
            c.expert_intermediate_dim = 768
        }),
        ("whole_prompt_prefill", |c| c.whole_prompt_prefill = true),
        ("device_router", |c| c.device_router = true),
        ("matmul_path", |c| c.matmul_path = "gemv"),
    ]
}

fn manual_engine_config_digest(c: &EngineConfig) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIEN-PKV-ENG-V1\0");
    h.update(c.num_experts.to_le_bytes());
    h.update(c.experts_per_token.to_le_bytes());
    h.update(c.expert_intermediate_dim.to_le_bytes());
    h.update([u8::from(c.whole_prompt_prefill)]);
    h.update([u8::from(c.device_router)]);
    h.update([c.matmul_path.len() as u8]);
    h.update(c.matmul_path.as_bytes());
    h.finalize().into()
}

#[test]
fn engine_config_digest_changes_with_each_moe_field_and_switch() {
    let base = sample_engine_config();
    let base_d = base.digest().unwrap();
    assert_eq!(base_d, manual_engine_config_digest(&base));
    let mut seen = vec![("base", base_d)];
    for (name, mutate) in engine_config_mutators() {
        let mut c = sample_engine_config();
        mutate(&mut c);
        let d = c.digest().unwrap();
        assert_eq!(d, manual_engine_config_digest(&c), "{name}: formula");
        for (other, od) in &seen {
            assert_ne!(d, *od, "{name} collides with {other}");
        }
        seen.push((name, d));
        // And through field 14 the namespace differs too.
        let mut e = sample_engine();
        e.engine_config_digest = d;
        assert_ne!(
            fields_with_engine(&e).namespace_id().unwrap(),
            sample().namespace_id().unwrap(),
            "{name}: namespace ignores the switch"
        );
    }
    assert_eq!(seen.len(), 7);
}

/// Field 14 equals the byte-by-byte formula of ADR 1.2.2.
#[test]
fn field_14_binds_compute_config_then_engine_identity() {
    let c = sample_compute();
    let e = sample_engine();
    let mut h = Sha256::new();
    h.update(b"AIEN-PKV-CFG-V2\0");
    h.update(c.num_heads.to_le_bytes());
    h.update(c.hidden_dim.to_le_bytes());
    h.update(c.intermediate_dim.to_le_bytes());
    h.update(c.vocab_size.to_le_bytes());
    h.update(c.rms_norm_eps.to_bits().to_le_bytes());
    h.update([u8::from(c.tie_word_embeddings)]);
    h.update([1u8]);
    h.update([c.arith_id.len() as u8]);
    h.update(c.arith_id.as_bytes());
    h.update(c.arith_epoch.to_le_bytes());
    h.update([e.weight_format as u8]);
    h.update([e.engine_build_id.len() as u8]);
    h.update(e.engine_build_id.as_bytes());
    h.update([e.rounding_class.len() as u8]);
    h.update(e.rounding_class.as_bytes());
    h.update(e.engine_config_digest);
    let want: [u8; 32] = h.finalize().into();
    assert_eq!(c.digest_with_engine(&e).unwrap(), want);
    assert_eq!(sample().compute_config_digest, want);
    // The compute-only digest is not field 14 any more.
    assert_ne!(c.digest().unwrap(), want);
}

/// The engine build id and the rounding class are required: an empty string
/// would let every engine alias, so it is refused, as is an oversized one.
#[test]
fn empty_or_oversized_engine_strings_are_refused() {
    let c = sample_compute();
    let mut e = sample_engine();
    e.engine_build_id = "";
    assert_eq!(
        c.digest_with_engine(&e),
        Err(KvError::FieldEmpty("engine_build_id"))
    );
    let mut e = sample_engine();
    e.rounding_class = "";
    assert_eq!(
        c.digest_with_engine(&e),
        Err(KvError::FieldEmpty("rounding_class"))
    );
    let mut e = sample_engine();
    e.engine_build_id = Box::leak("b".repeat(256).into_boxed_str());
    assert_eq!(
        c.digest_with_engine(&e),
        Err(KvError::FieldTooLong("engine_build_id"))
    );
    let mut e = sample_engine();
    e.rounding_class = Box::leak("r".repeat(256).into_boxed_str());
    assert_eq!(
        c.digest_with_engine(&e),
        Err(KvError::FieldTooLong("rounding_class"))
    );
    let mut ec = sample_engine_config();
    ec.matmul_path = "";
    assert_eq!(ec.digest(), Err(KvError::FieldEmpty("matmul_path")));
}

/// The namespace domain moved to V2 with the engine identity: the V1
/// derivation over the very same 14 field values no longer matches, so no
/// object written before sc#504 can be found.
#[test]
fn v1_namespace_domain_no_longer_matches() {
    let f = sample();
    let v1 = manual_namespace(&f, b"AIEN-PKV-NS-V1\0");
    let v2 = manual_namespace(&f, b"AIEN-PKV-NS-V2\0");
    let got = f.namespace_id().unwrap().0;
    assert_ne!(got, v1, "the V1 domain still matches");
    assert_eq!(got, v2);
    // Also with the old (compute-only) field 14 the V1 id is not reproduced.
    let mut old = sample();
    old.compute_config_digest = sample_compute().digest().unwrap();
    assert_ne!(
        old.namespace_id().unwrap().0,
        manual_namespace(&old, b"AIEN-PKV-NS-V1\0")
    );
}
