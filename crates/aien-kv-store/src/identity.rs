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

// Prefix-chained chunk keys adapt the rolling-hash idea of LMCache's
// `TokenHasher` (lmcache/v1/multiprocess/token_hasher.py, lines 30-52, 60-70,
// 183-190 at 4d69cdc4). LMCache is Apache-2.0, Copyright LMCache contributors.
// No LMCache code is copied. This file hashes with SHA-256 over a different
// byte layout (ADR 0003 section 1) so host and native share one derivation.

//! Versioned chunk identity: namespace id and prefix-chained chunk keys
//! (ADR 0003 section 1). Pure functions, no I/O, no heap use while hashing.

use crate::error::KvError;
use crate::record::{segment_count, ChunkHeader, FLAG_PARTIAL_TAIL, FORMAT_VERSION, MAX_SEGMENTS};
use aien_platform::ObjectId;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

/// Fingerprint of everything that must be equal for stored bytes to be reusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamespaceId(pub [u8; 32]);

/// Key of one chunk; carries the prefix chain. Wraps the platform `ObjectId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkKey(pub ObjectId);

impl ChunkKey {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0 .0
    }
}

impl PartialOrd for ChunkKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

/// KV element type codes (ADR 1.2 field 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KvDtype {
    Fp4 = 1,
    Fp8 = 2,
    Bf16 = 3,
    Fp16 = 4,
    Fp32 = 5,
}

impl KvDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Fp4),
            2 => Some(Self::Fp8),
            3 => Some(Self::Bf16),
            4 => Some(Self::Fp16),
            5 => Some(Self::Fp32),
            _ => None,
        }
    }
}

/// Payload byte format (ADR 1.2 field 11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BackendFormat {
    /// Host paged pool v1: bf16 blocks in the pool's own layout (ADR 2.2).
    HostPagedPoolV1 = 1,
    /// Native f32 decode state v1: strided per layer (ADR 2.3).
    NativeF32DecodeStateV1 = 2,
}

impl BackendFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::HostPagedPoolV1),
            2 => Some(Self::NativeF32DecodeStateV1),
            _ => None,
        }
    }
}

/// Llama-3 RoPE scaling (ADR 1.2 field 7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: u64,
}

/// Byte strides of the pool layout (ADR 1.2 field 10; `KvLayout` in aien-kv-cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutStrides {
    pub head_stride: u64,
    pub token_stride: u64,
    pub kv_plane_stride: u64,
    pub layer_stride: u64,
    pub block_stride: u64,
    pub element_bytes: u64,
}

/// Every input of the namespace id, in ADR 1.2 order. Strings are borrowed so a
/// caller builds this without copying model metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NamespaceFields<'a> {
    pub model_sha256: [u8; 32],
    /// At most 255 bytes (u8 length prefix).
    pub model_digest_kind: &'a str,
    /// At most 65535 bytes (u16 length prefix).
    pub model_id: &'a str,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub qk_norm: bool,
    pub dtype: KvDtype,
    pub layout: LayoutStrides,
    pub backend_format: BackendFormat,
    pub chunk_tokens: u32,
    /// At most 64 bytes (METRICS_AND_SETTINGS `AIEN_KV_STORE_NAMESPACE`).
    pub sharing_namespace: &'a str,
    /// `ComputeConfig::digest_with_engine()` (ADR 1.2 field 14, sections 1.2.1
    /// and 1.2.2): every other number that changes what the forward pass
    /// computes, and the engine that computed it.
    pub compute_config_digest: [u8; 32],
}

pub const MAX_SHARING_NAMESPACE_BYTES: usize = 64;
/// At most 255 bytes (u8 length prefix).
pub const MAX_ARITH_ID_BYTES: usize = 255;

/// Activation function code (ADR 1.2.1 field 7). The loader refuses everything
/// but silu today; the byte exists so a future activation cannot alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HiddenAct {
    Silu = 1,
}

/// Computation-affecting settings not already in the namespace fields (ADR
/// 1.2.1). Built from the resolved `ModelConfig` by `aien-runtime` (cut C6a),
/// never from raw `config.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeConfig<'a> {
    pub num_heads: u32,
    pub hidden_dim: u64,
    pub intermediate_dim: u64,
    pub vocab_size: u64,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub hidden_act: HiddenAct,
    /// `TensorBackend::name()`; at most 255 bytes.
    pub arith_id: &'a str,
    /// `KV_ARITH_EPOCH`: bumped by any change that alters forward-pass numbers.
    pub arith_epoch: u32,
}

impl ComputeConfig<'_> {
    /// Compute-only digest of ADR 1.2.1: `SHA-256("AIEN-PKV-CFG-V1\0" ||
    /// num_heads u32le || hidden_dim u64le || intermediate_dim u64le ||
    /// vocab_size u64le || rms_norm_eps bits u32le || tie u8 || hidden_act u8 ||
    /// arith_id u8-len + bytes || arith_epoch u32le)`. Since sc#504 this is
    /// **not** namespace field 14: field 14 is
    /// [`ComputeConfig::digest_with_engine`], which also binds the engine.
    pub fn digest(&self) -> Result<[u8; 32], KvError> {
        let mut h = Sha256::new();
        h.update(CFG_DOMAIN);
        self.put_fields(&mut h)?;
        Ok(h.finalize().into())
    }

    /// Namespace field 14 (ADR 1.2.2): `SHA-256("AIEN-PKV-CFG-V2\0" || fields 1
    /// to 9 exactly as in [`ComputeConfig::digest`] || weight_format u8 ||
    /// engine_build_id u8-len + bytes || rounding_class u8-len + bytes ||
    /// engine_config_digest)`. Refused when the build id or the rounding class
    /// is empty (`FieldEmpty`) or longer than 255 bytes (`FieldTooLong`).
    pub fn digest_with_engine(&self, engine: &EngineIdentity<'_>) -> Result<[u8; 32], KvError> {
        let mut h = Sha256::new();
        h.update(CFG_ENGINE_DOMAIN);
        self.put_fields(&mut h)?;
        h.update([engine.weight_format as u8]); // 10
        put_required_u8_len_str(&mut h, "engine_build_id", engine.engine_build_id)?; // 11
        put_required_u8_len_str(&mut h, "rounding_class", engine.rounding_class)?; // 12
        h.update(engine.engine_config_digest); // 13
        Ok(h.finalize().into())
    }

    fn put_fields(&self, h: &mut Sha256) -> Result<(), KvError> {
        h.update(self.num_heads.to_le_bytes()); // 1
        h.update(self.hidden_dim.to_le_bytes()); // 2
        h.update(self.intermediate_dim.to_le_bytes()); // 3
        h.update(self.vocab_size.to_le_bytes()); // 4
        h.update(self.rms_norm_eps.to_bits().to_le_bytes()); // 5
        h.update([u8::from(self.tie_word_embeddings)]); // 6
        h.update([self.hidden_act as u8]); // 7
        if self.arith_id.len() > MAX_ARITH_ID_BYTES {
            return Err(KvError::FieldTooLong("arith_id"));
        }
        put_u8_len_str(h, "arith_id", self.arith_id)?; // 8
        h.update(self.arith_epoch.to_le_bytes()); // 9
        Ok(())
    }
}

/// How the engine holds the weights it computes with (ADR 1.2.2). One
/// checkpoint directory (one file digest) can be loaded several ways, and each
/// gives different K/V bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightFormat {
    /// The weights are used in the element type the file stores them in.
    AsFile = 1,
    /// Converted to bf16 at load.
    Bf16 = 2,
    /// Expert (MoE) weights quantized to fp8 at load, the rest as loaded.
    Fp8Experts = 3,
    /// Every linear weight quantized to fp8 at load.
    Fp8All = 4,
}

impl WeightFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::AsFile),
            2 => Some(Self::Bf16),
            3 => Some(Self::Fp8Experts),
            4 => Some(Self::Fp8All),
            _ => None,
        }
    }

    /// Stable name for logs and settings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AsFile => "as-file",
            Self::Bf16 => "bf16",
            Self::Fp8Experts => "fp8-experts",
            Self::Fp8All => "fp8-all",
        }
    }
}

/// MoE shape and run-time numeric switches of the engine (ADR 1.2.2), built by
/// the engine caller. A dense model has zero experts. Whole-prompt prefill
/// changes activation rounding, so it changes the K/V bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig<'a> {
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub expert_intermediate_dim: u64,
    pub whole_prompt_prefill: bool,
    pub device_router: bool,
    /// Matrix-multiply path label (for example `gemm`, `gemv`); 1 to 255 bytes.
    pub matmul_path: &'a str,
}

impl EngineConfig<'_> {
    /// `SHA-256("AIEN-PKV-ENG-V1\0" || num_experts u32le || experts_per_token
    /// u32le || expert_intermediate_dim u64le || whole_prompt_prefill u8 ||
    /// device_router u8 || matmul_path u8-len + bytes)`.
    pub fn digest(&self) -> Result<[u8; 32], KvError> {
        let mut h = Sha256::new();
        h.update(ENG_DOMAIN);
        h.update(self.num_experts.to_le_bytes());
        h.update(self.experts_per_token.to_le_bytes());
        h.update(self.expert_intermediate_dim.to_le_bytes());
        h.update([u8::from(self.whole_prompt_prefill)]);
        h.update([u8::from(self.device_router)]);
        put_required_u8_len_str(&mut h, "matmul_path", self.matmul_path)?;
        Ok(h.finalize().into())
    }
}

/// The engine that computes the K/V (ADR 1.2.2, sc#504). Required by
/// [`ComputeConfig::digest_with_engine`] and by the runtime's persistence
/// wiring: there is no default, so every caller states each value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineIdentity<'a> {
    /// Weights as loaded, not as stored: one file digest can be loaded as
    /// bf16 or fp8.
    pub weight_format: WeightFormat,
    /// Engine and kernel build (for example engine commit plus the linked
    /// kernel archive digest); 1 to 255 bytes.
    pub engine_build_id: &'a str,
    /// Numeric rounding class of the arithmetic (for example backend name plus
    /// `KV_ARITH_EPOCH`); 1 to 255 bytes.
    pub rounding_class: &'a str,
    /// [`EngineConfig::digest`], or the engine's own 32-byte digest over its
    /// MoE shape and numeric run switches.
    pub engine_config_digest: [u8; 32],
}

/// V2 since sc#504 (field 14 binds the engine), so no V1 namespace can match.
const NS_DOMAIN: &[u8] = b"AIEN-PKV-NS-V2\0";
const CHUNK_DOMAIN: &[u8] = b"AIEN-PKV-CHUNK-V1\0";
const CFG_DOMAIN: &[u8] = b"AIEN-PKV-CFG-V1\0";
const CFG_ENGINE_DOMAIN: &[u8] = b"AIEN-PKV-CFG-V2\0";
const ENG_DOMAIN: &[u8] = b"AIEN-PKV-ENG-V1\0";

/// A required string: empty is refused, so an unknown value cannot alias.
fn put_required_u8_len_str(h: &mut Sha256, name: &'static str, s: &str) -> Result<(), KvError> {
    if s.is_empty() {
        return Err(KvError::FieldEmpty(name));
    }
    put_u8_len_str(h, name, s)
}

fn put_u8_len_str(h: &mut Sha256, name: &'static str, s: &str) -> Result<(), KvError> {
    let len = u8::try_from(s.len()).map_err(|_| KvError::FieldTooLong(name))?;
    h.update([len]);
    h.update(s.as_bytes());
    Ok(())
}

fn put_u16_len_str(h: &mut Sha256, name: &'static str, s: &str) -> Result<(), KvError> {
    let len = u16::try_from(s.len()).map_err(|_| KvError::FieldTooLong(name))?;
    h.update(len.to_le_bytes());
    h.update(s.as_bytes());
    Ok(())
}

impl NamespaceFields<'_> {
    fn check_geometry(&self) -> Result<(), KvError> {
        if self.block_size == 0
            || self.chunk_tokens == 0
            || self.num_layers == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
            || !self.chunk_tokens.is_multiple_of(self.block_size)
        {
            return Err(KvError::InvalidGeometry);
        }
        // A zero byte factor makes every payload zero bytes long.
        let zero_bytes = match self.backend_format {
            BackendFormat::HostPagedPoolV1 => self.layout.block_stride == 0,
            BackendFormat::NativeF32DecodeStateV1 => self.layout.element_bytes == 0,
        };
        if zero_bytes {
            return Err(KvError::InvalidGeometry);
        }
        // A host chunk is one segment per pool block and the digest table holds
        // at most `MAX_SEGMENTS` = 119 (ADR 2.1 item 3, amended by section 20):
        // `chunk_blocks` is capped at 119 there.
        if self.backend_format == BackendFormat::HostPagedPoolV1
            && self.chunk_tokens / self.block_size > MAX_SEGMENTS as u32
        {
            return Err(KvError::TooManySegments);
        }
        Ok(())
    }

    /// `SHA-256("AIEN-PKV-NS-V2\0" || fields)` (ADR 1.2).
    pub fn namespace_id(&self) -> Result<NamespaceId, KvError> {
        self.check_geometry()?;
        if self.sharing_namespace.len() > MAX_SHARING_NAMESPACE_BYTES {
            return Err(KvError::FieldTooLong("sharing_namespace"));
        }
        let mut h = Sha256::new();
        h.update(NS_DOMAIN);
        h.update(FORMAT_VERSION.to_le_bytes()); // 1
        h.update(self.model_sha256); // 2
        put_u8_len_str(&mut h, "model_digest_kind", self.model_digest_kind)?; // 3
        put_u16_len_str(&mut h, "model_id", self.model_id)?; // 4
        for v in [
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        ] {
            h.update(v.to_le_bytes()); // 5
        }
        h.update(self.rope_theta.to_bits().to_le_bytes()); // 6
        match &self.rope_scaling {
            // 7
            None => h.update([0u8]),
            Some(r) => {
                h.update([1u8]);
                h.update(r.factor.to_bits().to_le_bytes());
                h.update(r.low_freq_factor.to_bits().to_le_bytes());
                h.update(r.high_freq_factor.to_bits().to_le_bytes());
                h.update(r.original_max_position_embeddings.to_le_bytes());
            }
        }
        h.update([u8::from(self.qk_norm)]); // 8
        h.update([self.dtype as u8]); // 9
        let l = &self.layout; // 10
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
        h.update([self.backend_format as u8]); // 11
        h.update(self.chunk_tokens.to_le_bytes()); // 12
        put_u16_len_str(&mut h, "sharing_namespace", self.sharing_namespace)?; // 13
        h.update(self.compute_config_digest); // 14
        Ok(NamespaceId(h.finalize().into()))
    }

    /// Payload size the writer must produce for `token_count` positions
    /// (ADR 2.2 and 2.3). Checked arithmetic, no allocation.
    ///
    /// Host: `ceil(token_count / block_size) * block_stride`. Native: per layer,
    /// K rows then V rows of `token_count * kv_dim` elements.
    pub fn expected_payload_len(&self, token_count: u32) -> Result<u64, KvError> {
        self.check_geometry()?;
        if token_count == 0 || token_count > self.chunk_tokens {
            return Err(KvError::TokenCountOutOfRange);
        }
        match self.backend_format {
            BackendFormat::HostPagedPoolV1 => {
                let blocks = u64::from(token_count.div_ceil(self.block_size));
                blocks
                    .checked_mul(self.layout.block_stride)
                    .ok_or(KvError::Overflow)
            }
            BackendFormat::NativeF32DecodeStateV1 => {
                let kv_dim = u64::from(self.num_kv_heads)
                    .checked_mul(u64::from(self.head_dim))
                    .ok_or(KvError::Overflow)?;
                u64::from(self.num_layers)
                    .checked_mul(2)
                    .and_then(|v| v.checked_mul(u64::from(token_count)))
                    .and_then(|v| v.checked_mul(kv_dim))
                    .and_then(|v| v.checked_mul(self.layout.element_bytes))
                    .ok_or(KvError::Overflow)
            }
        }
    }

    /// Bytes per payload segment for a chunk of `token_count` positions (ADR
    /// 12 H8): host = `layout.block_stride` (one pool block); native = the whole
    /// payload (one segment).
    pub fn segment_bytes(&self, token_count: u32) -> Result<u64, KvError> {
        match self.backend_format {
            BackendFormat::HostPagedPoolV1 => {
                self.check_geometry()?;
                if token_count == 0 || token_count > self.chunk_tokens {
                    return Err(KvError::TokenCountOutOfRange);
                }
                Ok(self.layout.block_stride)
            }
            BackendFormat::NativeF32DecodeStateV1 => self.expected_payload_len(token_count),
        }
    }

    /// Number of payload segments for `token_count` positions; at most
    /// `MAX_SEGMENTS`.
    pub fn segment_count(&self, token_count: u32) -> Result<u32, KvError> {
        let payload_len = self.expected_payload_len(token_count)?;
        segment_count(payload_len, self.segment_bytes(token_count)?)
    }

    /// Build the header a writer would emit (or a reader would expect) for one
    /// chunk. Sets `partial_tail` when `token_count < chunk_tokens`; fills
    /// `payload_len` from `expected_payload_len`.
    pub fn chunk_header(
        &self,
        namespace_id: &NamespaceId,
        chunk_key: ChunkKey,
        parent: Option<&ChunkKey>,
        first_position: u32,
        token_count: u32,
        written_unix_ns: u64,
    ) -> Result<ChunkHeader, KvError> {
        let payload_len = self.expected_payload_len(token_count)?;
        if !first_position.is_multiple_of(self.chunk_tokens) {
            return Err(KvError::PositionMismatch);
        }
        Ok(ChunkHeader {
            namespace_id: *namespace_id,
            chunk_key,
            parent_key: parent.map_or([0u8; 32], |p| *p.as_bytes()),
            model_sha256: self.model_sha256,
            first_position,
            token_count,
            chunk_tokens: self.chunk_tokens,
            block_size: self.block_size,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype as u8,
            backend_format: self.backend_format as u8,
            flags: if token_count < self.chunk_tokens {
                FLAG_PARTIAL_TAIL
            } else {
                0
            },
            payload_len,
            written_unix_ns,
        })
    }
}

/// One link of the chain (ADR 1.3). `parent` is `None` for the first chunk
/// (hashed as 32 zero bytes). `first_position` is the absolute position of
/// `tokens[0]`.
pub fn chunk_key(
    namespace: &NamespaceId,
    parent: Option<&ChunkKey>,
    first_position: u32,
    tokens: &[u32],
) -> Result<ChunkKey, KvError> {
    let token_count = u32::try_from(tokens.len()).map_err(|_| KvError::Overflow)?;
    if token_count == 0 {
        return Err(KvError::TokenCountOutOfRange);
    }
    first_position
        .checked_add(token_count)
        .ok_or(KvError::Overflow)?;
    let mut h = Sha256::new();
    h.update(CHUNK_DOMAIN);
    h.update(namespace.0);
    h.update(parent.map_or([0u8; 32], |p| *p.as_bytes()));
    h.update(first_position.to_le_bytes());
    h.update(token_count.to_le_bytes());
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    Ok(ChunkKey(ObjectId(h.finalize().into())))
}

/// Keys for every chunk of `tokens`: full chunks of `chunk_tokens`, then an
/// optional partial tail keyed with its exact token count.
pub fn chunk_chain(
    namespace: &NamespaceId,
    chunk_tokens: u32,
    tokens: &[u32],
) -> Result<Vec<ChunkKey>, KvError> {
    if chunk_tokens == 0 {
        return Err(KvError::InvalidGeometry);
    }
    let per = chunk_tokens as usize;
    let mut keys: Vec<ChunkKey> = Vec::with_capacity(tokens.len().div_ceil(per));
    for (i, chunk) in tokens.chunks(per).enumerate() {
        let first_position = u32::try_from(i)
            .ok()
            .and_then(|i| i.checked_mul(chunk_tokens))
            .ok_or(KvError::Overflow)?;
        let key = chunk_key(namespace, keys.last(), first_position, chunk)?;
        keys.push(key);
    }
    Ok(keys)
}
