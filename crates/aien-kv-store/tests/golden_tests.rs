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

//! Golden vectors: `tests/golden/kv_record_v1.json`. The file pins the
//! compute-config digest, the namespace derivation, the chunk key chain, the
//! 256-byte header bytes and the 4096-byte header region (segment digest table
//! and root) so the native C implementation (aienos N1) can check itself
//! against the same file (CUTS "Pin"). Regenerate deliberately with
//! `AIEN_KV_REGEN_GOLDEN=1 cargo test -p aien-kv-store --test golden_tests regenerate_golden`;
//! then bump `tests/golden/README.md` and the aienos `CONTRACT.lock`.

mod common;

use aien_kv_store::{
    chunk_chain, validate_record, BackendFormat, ChunkKey, ComputeConfig, EngineConfig,
    EngineIdentity, HiddenAct, KvDtype, LayoutStrides, NamespaceFields, RecordHeader, RopeScaling,
    WeightFormat,
};
use common::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/kv_record_v1.json"
);

fn cc_to_json(c: &ComputeConfig) -> Value {
    json!({
        "num_heads": c.num_heads,
        "hidden_dim": c.hidden_dim,
        "intermediate_dim": c.intermediate_dim,
        "vocab_size": c.vocab_size,
        "rms_norm_eps_f32_bits": c.rms_norm_eps.to_bits(),
        "tie_word_embeddings": c.tie_word_embeddings,
        "hidden_act": c.hidden_act as u8,
        "arith_id": c.arith_id,
        "arith_epoch": c.arith_epoch,
    })
}

fn parse_cc(v: &Value) -> ComputeConfig<'_> {
    assert_eq!(u(v, "hidden_act"), 1, "only silu exists");
    ComputeConfig {
        num_heads: u(v, "num_heads") as u32,
        hidden_dim: u(v, "hidden_dim"),
        intermediate_dim: u(v, "intermediate_dim"),
        vocab_size: u(v, "vocab_size"),
        rms_norm_eps: f32::from_bits(u(v, "rms_norm_eps_f32_bits") as u32),
        tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap(),
        hidden_act: HiddenAct::Silu,
        arith_id: v["arith_id"].as_str().unwrap(),
        arith_epoch: u(v, "arith_epoch") as u32,
    }
}

fn ec_to_json(c: &EngineConfig) -> Value {
    json!({
        "num_experts": c.num_experts,
        "experts_per_token": c.experts_per_token,
        "expert_intermediate_dim": c.expert_intermediate_dim,
        "whole_prompt_prefill": c.whole_prompt_prefill,
        "device_router": c.device_router,
        "matmul_path": c.matmul_path,
    })
}

fn parse_ec(v: &Value) -> EngineConfig<'_> {
    EngineConfig {
        num_experts: u(v, "num_experts") as u32,
        experts_per_token: u(v, "experts_per_token") as u32,
        expert_intermediate_dim: u(v, "expert_intermediate_dim"),
        whole_prompt_prefill: v["whole_prompt_prefill"].as_bool().unwrap(),
        device_router: v["device_router"].as_bool().unwrap(),
        matmul_path: v["matmul_path"].as_str().unwrap(),
    }
}

fn engine_to_json(e: &EngineIdentity, ec: &EngineConfig) -> Value {
    assert_eq!(e.engine_config_digest, ec.digest().unwrap());
    json!({
        "weight_format": e.weight_format as u8,
        "weight_format_name": e.weight_format.as_str(),
        "engine_build_id": e.engine_build_id,
        "rounding_class": e.rounding_class,
        "engine_config": ec_to_json(ec),
        "engine_config_digest": hex(&e.engine_config_digest),
    })
}

fn parse_engine(v: &Value) -> EngineIdentity<'_> {
    EngineIdentity {
        weight_format: WeightFormat::from_u8(u(v, "weight_format") as u8).unwrap(),
        engine_build_id: v["engine_build_id"].as_str().unwrap(),
        rounding_class: v["rounding_class"].as_str().unwrap(),
        engine_config_digest: unhex32(v["engine_config_digest"].as_str().unwrap()),
    }
}

fn ns_to_json(
    f: &NamespaceFields,
    c: &ComputeConfig,
    e: &EngineIdentity,
    ec: &EngineConfig,
) -> Value {
    json!({
        "model_sha256": hex(&f.model_sha256),
        "model_digest_kind": f.model_digest_kind,
        "model_id": f.model_id,
        "num_layers": f.num_layers,
        "num_kv_heads": f.num_kv_heads,
        "head_dim": f.head_dim,
        "block_size": f.block_size,
        "rope_theta_f32_bits": f.rope_theta.to_bits(),
        "rope_scaling": f.rope_scaling.map(|r| json!({
            "factor_f64_bits": r.factor.to_bits(),
            "low_freq_factor_f64_bits": r.low_freq_factor.to_bits(),
            "high_freq_factor_f64_bits": r.high_freq_factor.to_bits(),
            "original_max_position_embeddings": r.original_max_position_embeddings,
        })),
        "qk_norm": f.qk_norm,
        "dtype": f.dtype as u8,
        "layout": {
            "head_stride": f.layout.head_stride,
            "token_stride": f.layout.token_stride,
            "kv_plane_stride": f.layout.kv_plane_stride,
            "layer_stride": f.layout.layer_stride,
            "block_stride": f.layout.block_stride,
            "element_bytes": f.layout.element_bytes,
        },
        "backend_format": f.backend_format as u8,
        "chunk_tokens": f.chunk_tokens,
        "sharing_namespace": f.sharing_namespace,
        "compute_config": cc_to_json(c),
        "engine": engine_to_json(e, ec),
        "compute_config_digest": hex(&f.compute_config_digest),
    })
}

fn u(v: &Value, k: &str) -> u64 {
    v[k].as_u64().unwrap_or_else(|| panic!("missing u64 {k}"))
}

fn parse_ns(v: &Value) -> NamespaceFields<'_> {
    let l = &v["layout"];
    NamespaceFields {
        model_sha256: unhex32(v["model_sha256"].as_str().unwrap()),
        model_digest_kind: v["model_digest_kind"].as_str().unwrap(),
        model_id: v["model_id"].as_str().unwrap(),
        num_layers: u(v, "num_layers") as u32,
        num_kv_heads: u(v, "num_kv_heads") as u32,
        head_dim: u(v, "head_dim") as u32,
        block_size: u(v, "block_size") as u32,
        rope_theta: f32::from_bits(u(v, "rope_theta_f32_bits") as u32),
        rope_scaling: match &v["rope_scaling"] {
            Value::Null => None,
            r => Some(RopeScaling {
                factor: f64::from_bits(u(r, "factor_f64_bits")),
                low_freq_factor: f64::from_bits(u(r, "low_freq_factor_f64_bits")),
                high_freq_factor: f64::from_bits(u(r, "high_freq_factor_f64_bits")),
                original_max_position_embeddings: u(r, "original_max_position_embeddings"),
            }),
        },
        qk_norm: v["qk_norm"].as_bool().unwrap(),
        dtype: KvDtype::from_u8(u(v, "dtype") as u8).unwrap(),
        layout: LayoutStrides {
            head_stride: u(l, "head_stride"),
            token_stride: u(l, "token_stride"),
            kv_plane_stride: u(l, "kv_plane_stride"),
            layer_stride: u(l, "layer_stride"),
            block_stride: u(l, "block_stride"),
            element_bytes: u(l, "element_bytes"),
        },
        backend_format: BackendFormat::from_u8(u(v, "backend_format") as u8).unwrap(),
        chunk_tokens: u(v, "chunk_tokens") as u32,
        sharing_namespace: v["sharing_namespace"].as_str().unwrap(),
        compute_config_digest: unhex32(v["compute_config_digest"].as_str().unwrap()),
    }
}

fn tokens_of(v: &Value) -> Vec<u32> {
    v["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

fn written_ns(i: usize) -> u64 {
    1_700_000_000_000_000_000 + i as u64
}

/// Derive every output of one vector from its inputs (the library under test).
fn outputs(name: &str, ns_v: &Value, tokens: &[u32]) -> Value {
    let f = parse_ns(ns_v);
    let ns = f.namespace_id().unwrap();
    let chain = chunk_chain(&ns, f.chunk_tokens, tokens).unwrap();
    let mut chunks = Vec::new();
    for (i, key) in chain.iter().enumerate() {
        let start = i * f.chunk_tokens as usize;
        let end = (start + f.chunk_tokens as usize).min(tokens.len());
        let parent = if i == 0 { None } else { Some(&chain[i - 1]) };
        let token_count = (end - start) as u32;
        let h = f
            .chunk_header(&ns, *key, parent, start as u32, token_count, written_ns(i))
            .unwrap();
        let n_seg = f.segment_count(token_count).unwrap();
        let prefix = format!("{name}/{i}");
        let digests = seg_digests(&prefix, n_seg);
        let rec = RecordHeader {
            chunk: h,
            payload_sha256: aien_kv_store::payload_root(&digests),
        };
        let region = rec.encode_region(&digests).unwrap();
        chunks.push(json!({
            "index": i,
            "first_position": start,
            "token_count": end - start,
            "parent_key": hex(&h.parent_key),
            "chunk_key": hex(key.as_bytes()),
            "payload_len": h.payload_len,
            "segment_bytes": f.segment_bytes(token_count).unwrap(),
            "segment_count": n_seg,
            "segment_digests_of_ascii": format!("{prefix}/seg/<j>"),
            "segment_digests": digests.iter().map(|d| hex(d)).collect::<Vec<_>>(),
            "payload_sha256": hex(&rec.payload_sha256),
            "written_unix_ns": h.written_unix_ns,
            "header_hex": hex(&rec.encode()),
            "region_sha256": hex(&Sha256::digest(region)),
        }));
    }
    json!({ "namespace_id": hex(&ns.0), "chunks": chunks })
}

type VectorInput = (
    &'static str,
    NamespaceFields<'static>,
    ComputeConfig<'static>,
    (EngineIdentity<'static>, EngineConfig<'static>),
    Vec<u32>,
);

fn vector_inputs() -> Vec<VectorInput> {
    let mut small_f = sample();
    small_f.block_size = 4;
    small_f.chunk_tokens = 8;
    small_f.layout = LayoutStrides {
        head_stride: 128,
        token_stride: 1024,
        kv_plane_stride: 4096,
        layer_stride: 8192,
        block_stride: 131072,
        element_bytes: 2,
    };

    let mut native = sample();
    native.model_id = "llama-3.2-1b-instruct-q4km";
    native.model_digest_kind = "sha256-gguf";
    native.rope_scaling = None;
    native.dtype = KvDtype::Fp32;
    native.backend_format = BackendFormat::NativeF32DecodeStateV1;
    native.block_size = 16;
    native.chunk_tokens = 64;
    native.layout = LayoutStrides {
        head_stride: 256,
        token_stride: 2048,
        kv_plane_stride: 32768,
        layer_stride: 65536,
        block_stride: 1048576,
        element_bytes: 4,
    };

    let mut qk = small_f;
    qk.model_id = "qwen3-4b-instruct-2507";
    qk.rope_scaling = None;
    qk.qk_norm = true;
    qk.sharing_namespace = "";

    let llama = sample_compute();
    let qwen = ComputeConfig {
        num_heads: 32,
        hidden_dim: 2560,
        intermediate_dim: 9728,
        vocab_size: 151_936,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        hidden_act: HiddenAct::Silu,
        arith_id: "NativeCpuBackend (Rayon Multi-Core)",
        arith_epoch: 1,
    };
    let native_cc = ComputeConfig {
        arith_id: "aienos-infer-kernel",
        ..llama
    };
    let eps_1e6 = ComputeConfig {
        rms_norm_eps: 1e-6,
        ..llama
    };
    let epoch_2 = ComputeConfig {
        arith_epoch: 2,
        ..llama
    };

    let real_tokens: Vec<u32> = (0..300u32).map(|i| (i * 37 + 11) % 128_256).collect();
    let engine = (sample_engine(), sample_engine_config());
    let fp8_experts = (
        EngineIdentity {
            weight_format: WeightFormat::Fp8Experts,
            ..sample_engine()
        },
        sample_engine_config(),
    );
    let whole_prompt_cfg = EngineConfig {
        whole_prompt_prefill: true,
        ..sample_engine_config()
    };
    let whole_prompt = (
        EngineIdentity {
            engine_config_digest: whole_prompt_cfg.digest().unwrap(),
            ..sample_engine()
        },
        whole_prompt_cfg,
    );
    let with_digest =
        |mut f: NamespaceFields<'static>, c: &ComputeConfig, e: &(EngineIdentity, EngineConfig)| {
            f.compute_config_digest = c.digest_with_engine(&e.0).unwrap();
            f
        };
    vec![
        (
            "host-small-partial-tail",
            with_digest(small_f, &llama, &engine),
            llama,
            engine,
            (0..20).collect(),
        ),
        (
            "host-llama32-1b-full-and-partial",
            with_digest(sample(), &llama, &engine),
            llama,
            engine,
            real_tokens,
        ),
        (
            "native-f32-partial-tail",
            with_digest(native, &native_cc, &engine),
            native_cc,
            engine,
            (1000..1070).collect(),
        ),
        (
            "qk-norm-empty-sharing-one-full-chunk",
            with_digest(qk, &qwen, &engine),
            qwen,
            engine,
            (50..58).collect(),
        ),
        // Review finding 1: identical to `host-small-partial-tail` except
        // rms_norm_eps (1e-5 there, 1e-6 here); the namespace id must differ.
        (
            "rms-norm-eps-1e-6-only",
            with_digest(small_f, &eps_1e6, &engine),
            eps_1e6,
            engine,
            (0..20).collect(),
        ),
        // Identical to `host-small-partial-tail` except arith_epoch 2.
        (
            "arith-epoch-2",
            with_digest(small_f, &epoch_2, &engine),
            epoch_2,
            engine,
            (0..20).collect(),
        ),
        // sc#504: identical to `host-small-partial-tail` except the weight
        // format as loaded (same file, fp8 experts).
        (
            "weight-format-fp8-experts-only",
            with_digest(small_f, &llama, &fp8_experts),
            llama,
            fp8_experts,
            (0..20).collect(),
        ),
        // sc#504: identical except the whole-prompt prefill switch of the
        // engine config (it changes activation rounding).
        (
            "whole-prompt-prefill-only",
            with_digest(small_f, &llama, &whole_prompt),
            llama,
            whole_prompt,
            (0..20).collect(),
        ),
    ]
}

#[test]
fn regenerate_golden() {
    if std::env::var("AIEN_KV_REGEN_GOLDEN").as_deref() != Ok("1") {
        return;
    }
    let vectors: Vec<Value> = vector_inputs()
        .into_iter()
        .map(|(name, f, c, (e, ec), tokens)| {
            let ns_v = ns_to_json(&f, &c, &e, &ec);
            let out = outputs(name, &ns_v, &tokens);
            json!({
                "name": name,
                "namespace": ns_v,
                "tokens": tokens,
                "expected": out,
            })
        })
        .collect();
    let doc = json!({
        "format_version": 1,
        "about": "ADR 0003 sections 1 (1.2.2, sc#504), 2.1 and 12 (C1b). engine_config_digest = SHA-256('AIEN-PKV-ENG-V1\\0' || num_experts u32le || experts_per_token u32le || expert_intermediate_dim u64le || whole_prompt_prefill u8 || device_router u8 || matmul_path u8-len + bytes); compute_config_digest (field 14) = SHA-256('AIEN-PKV-CFG-V2\\0' || num_heads u32le || hidden_dim u64le || intermediate_dim u64le || vocab_size u64le || rms_norm_eps f32 bits u32le || tie u8 || hidden_act u8 || arith_id u8-len + bytes || arith_epoch u32le || weight_format u8 (1 as-file, 2 bf16, 3 fp8-experts, 4 fp8-all) || engine_build_id u8-len + bytes || rounding_class u8-len + bytes || engine_config_digest); namespace_id = SHA-256('AIEN-PKV-NS-V2\\0' || fields 1 to 13 || compute_config_digest); chunk_key = SHA-256('AIEN-PKV-CHUNK-V1\\0' || namespace_id || parent_key || first_position u32le || token_count u32le || tokens u32le). header_hex is the 256-byte record header; the record file starts with a 4096-byte region = header_hex || segment digests (32 bytes each, segment_count of them) || zeros; the payload starts at file offset 4096. payload_sha256 = SHA-256('AIEN-PKV-PAYLOAD-V1\\0' || segment digests). Segment digests are SHA-256 of the ASCII string segment_digests_of_ascii with <j> replaced by the segment index. region_sha256 is the SHA-256 of the whole 4096-byte region. f32/f64 inputs are given as IEEE bit patterns.",
        "vectors": vectors,
    });
    let mut text = serde_json::to_string_pretty(&doc).unwrap();
    text.push('\n');
    std::fs::write(GOLDEN, text).unwrap();
}

fn golden() -> Value {
    let text = std::fs::read_to_string(GOLDEN).expect("golden file present");
    serde_json::from_str(&text).unwrap()
}

#[test]
fn identity_vectors_match_golden() {
    let doc = golden();
    assert_eq!(doc["format_version"], 1);
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(
        vectors.len() >= 8,
        "golden file must carry the six vectors of C1b and the two engine vectors of sc#504"
    );
    let mut checked_headers = 0;
    for v in vectors {
        let name = v["name"].as_str().unwrap();
        let tokens = tokens_of(v);
        let got = outputs(name, &v["namespace"], &tokens);
        assert_eq!(
            got["namespace_id"], v["expected"]["namespace_id"],
            "{name}: namespace_id"
        );
        let want_chunks = v["expected"]["chunks"].as_array().unwrap();
        let got_chunks = got["chunks"].as_array().unwrap();
        assert!(!want_chunks.is_empty(), "{name}: no chunks");
        assert_eq!(
            got_chunks, want_chunks,
            "{name}: chunk keys, header bytes or region differ"
        );

        // The stored bytes decode back to the same fields and validate.
        let f = parse_ns(&v["namespace"]);
        let ns = f.namespace_id().unwrap();
        for c in want_chunks {
            let bytes = unhex(c["header_hex"].as_str().unwrap());
            let rec = RecordHeader::decode_struct(&bytes).unwrap();
            assert_eq!(
                rec.encode().as_slice(),
                bytes.as_slice(),
                "{name}: re-encode"
            );
            assert_eq!(rec.chunk.namespace_id, ns);
            assert_eq!(
                rec.chunk.chunk_key,
                ChunkKey(aien_platform::ObjectId(unhex32(
                    c["chunk_key"].as_str().unwrap()
                )))
            );
            let digests: Vec<[u8; 32]> = c["segment_digests"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| unhex32(d.as_str().unwrap()))
                .collect();
            assert_eq!(digests.len() as u64, u(c, "segment_count"));
            let region = rec.encode_region(&digests).unwrap();
            assert_eq!(
                hex(&Sha256::digest(region)),
                c["region_sha256"].as_str().unwrap(),
                "{name}: region bytes"
            );
            let file_size = 4096 + c["payload_len"].as_u64().unwrap();
            validate_record(&region, file_size, &rec.chunk, u(c, "segment_bytes")).unwrap();
            checked_headers += 1;
        }
    }
    assert!(checked_headers >= 12);
}

#[test]
fn golden_vectors_include_compute_config_digest() {
    let doc = golden();
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 8);
    for v in vectors {
        let name = v["name"].as_str().unwrap();
        let ns_v = &v["namespace"];
        // All nine inputs and the digest are present and consistent.
        let cc_v = &ns_v["compute_config"];
        for k in [
            "num_heads",
            "hidden_dim",
            "intermediate_dim",
            "vocab_size",
            "rms_norm_eps_f32_bits",
            "tie_word_embeddings",
            "hidden_act",
            "arith_id",
            "arith_epoch",
        ] {
            assert!(!cc_v[k].is_null(), "{name}: compute_config.{k} missing");
        }
        let cc = parse_cc(cc_v);
        let engine = parse_engine(&ns_v["engine"]);
        let ec = parse_ec(&ns_v["engine"]["engine_config"]);
        assert_eq!(
            hex(&ec.digest().unwrap()),
            ns_v["engine"]["engine_config_digest"].as_str().unwrap(),
            "{name}: engine config digest"
        );
        let digest = hex(&cc.digest_with_engine(&engine).unwrap());
        assert_eq!(
            ns_v["compute_config_digest"].as_str().unwrap(),
            digest,
            "{name}: digest"
        );
        // And it is what the namespace id commits to.
        let f = parse_ns(ns_v);
        let mut other = f;
        other.compute_config_digest[0] ^= 1;
        assert_ne!(
            f.namespace_id().unwrap(),
            other.namespace_id().unwrap(),
            "{name}: namespace ignores the digest"
        );
    }
    // The two new vectors differ from their sibling in exactly one field.
    let by_name = |n: &str| vectors.iter().find(|v| v["name"] == n).unwrap();
    let base = by_name("host-small-partial-tail");
    for (sibling, field_bits) in [
        ("rms-norm-eps-1e-6-only", "rms_norm_eps_f32_bits"),
        ("arith-epoch-2", "arith_epoch"),
    ] {
        let s = by_name(sibling);
        assert_eq!(base["tokens"], s["tokens"], "{sibling}: same tokens");
        assert_ne!(
            base["expected"]["namespace_id"], s["expected"]["namespace_id"],
            "{sibling}: namespace must differ"
        );
        let mut a = base["namespace"].clone();
        let mut b = s["namespace"].clone();
        assert_ne!(
            a["compute_config"][field_bits],
            b["compute_config"][field_bits]
        );
        for side in [&mut a, &mut b] {
            side["compute_config"][field_bits] = Value::Null;
            side["compute_config_digest"] = Value::Null;
        }
        assert_eq!(
            a, b,
            "{sibling}: differs from the base in {field_bits} only"
        );
    }
    assert_eq!(
        u(
            &by_name("host-small-partial-tail")["namespace"]["compute_config"],
            "rms_norm_eps_f32_bits"
        ),
        u64::from(1e-5f32.to_bits())
    );
    assert_eq!(
        u(
            &by_name("rms-norm-eps-1e-6-only")["namespace"]["compute_config"],
            "rms_norm_eps_f32_bits"
        ),
        u64::from(1e-6f32.to_bits())
    );
}

/// sc#504: the two engine vectors differ from `host-small-partial-tail` in one
/// engine input only, and their namespace ids differ.
#[test]
fn golden_engine_vectors_differ_in_one_engine_field_only() {
    let doc = golden();
    let vectors = doc["vectors"].as_array().unwrap();
    let by_name = |n: &str| vectors.iter().find(|v| v["name"] == n).unwrap();
    let base = by_name("host-small-partial-tail");
    for (sibling, path) in [
        ("weight-format-fp8-experts-only", &["weight_format"][..]),
        (
            "whole-prompt-prefill-only",
            &["engine_config", "whole_prompt_prefill"][..],
        ),
    ] {
        let s = by_name(sibling);
        assert_eq!(base["tokens"], s["tokens"], "{sibling}: same tokens");
        assert_ne!(
            base["expected"]["namespace_id"], s["expected"]["namespace_id"],
            "{sibling}: namespace must differ"
        );
        let mut a = base["namespace"].clone();
        let mut b = s["namespace"].clone();
        for side in [&mut a, &mut b] {
            let mut node = &mut side["engine"];
            for k in path {
                node = &mut node[*k];
            }
            *node = Value::Null;
            side["engine"]["weight_format_name"] = Value::Null;
            side["engine"]["engine_config_digest"] = Value::Null;
            side["compute_config_digest"] = Value::Null;
        }
        assert_eq!(a, b, "{sibling}: differs from the base in {path:?} only");
    }
}

/// The 256-byte struct of C1 (commit ce5d9eb, vector
/// `qk-norm-empty-sharing-one-full-chunk` chunk 0). C1b keeps offsets 0 to 255
/// as they were: only values derived from the changed identity differ.
const C1_QK_HEADER_HEX: &str = "4149454e4b56433101000100000100009f3ad00ef5e5a019509358ca17af031787ea605b2125adf9f45b45e68d79e77439892035cb1a28bc34e635299fa3952ce55337971cd47d0b15b2decf435611c40000000000000000000000000000000000000000000000000000000000000000abababababababababababababababababababababababababababababababab00000000080000000800000004000000100000000800000040000000030100000000040000000000937b24a1e2f3d11364ae69717aa6511bcdef9beea2569b40a3f7d3403bd80fe000002a36fe9c9717da902ce1dc4828ae671e228f2cf4439b6320f39c5b3f5143c842a0077f2af6c9";

#[test]
fn header_struct_bytes_0_to_255_unchanged_by_the_region() {
    let doc = golden();
    let v = doc["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == "qk-norm-empty-sharing-one-full-chunk")
        .unwrap();
    let new = unhex(v["expected"]["chunks"][0]["header_hex"].as_str().unwrap());
    let old = unhex(C1_QK_HEADER_HEX);
    assert_eq!((new.len(), old.len()), (256, 256));
    // Re-derived by the identity change: namespace_id (16..48), chunk_key
    // (48..80, its input includes the namespace), payload_sha256 (184..216, now
    // a root over segment digests) and header_sha256 (224..256, covers them).
    for range in [0..16, 80..184, 216..224] {
        // chunk_key sits at 48..80; parent 80..112 is zero for chunk 0.
        assert_eq!(&new[range.clone()], &old[range.clone()], "bytes {range:?}");
    }
    assert_ne!(&new[16..48], &old[16..48], "namespace is re-derived");
    assert_ne!(&new[184..216], &old[184..216], "payload root is re-derived");

    // The old struct still decodes and re-encodes with today's codec, and the
    // region puts the struct unchanged at offset 0.
    let rec = RecordHeader::decode_struct(&old).unwrap();
    assert_eq!(rec.encode().as_slice(), old.as_slice());
    let region = rec.encode_region(&seg_digests("legacy", 1)).unwrap();
    assert_eq!(&region[..256], old.as_slice());
    assert_eq!(&region[..256], rec.encode().as_slice());
}

/// `tests/golden/README.md` records the vectors hash that the aienos
/// `CONTRACT.lock` pins; changing the file without re-pinning fails here.
#[test]
fn readme_pins_the_current_vectors_hash() {
    let text = std::fs::read(GOLDEN).unwrap();
    let want = hex(&Sha256::digest(&text));
    let readme = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/README.md"
    ))
    .unwrap();
    assert!(
        readme.contains(&format!("vectors_sha256 = `{want}`")),
        "README.md vectors_sha256 is stale; expected {want}"
    );
}
