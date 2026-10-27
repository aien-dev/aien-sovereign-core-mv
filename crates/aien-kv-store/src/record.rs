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

//! The 256-byte record header, identical on host and native (ADR 0003 section 2.1).

use crate::error::KvError;
use crate::identity::{ChunkKey, NamespaceId};
use aien_platform::ObjectId;
use sha2::{Digest, Sha256};

pub const MAGIC: [u8; 8] = *b"AIENKVC1";
pub const HEADER_LEN: usize = 256;
/// A record file is `header region (4096 B) || payload` (ADR 0003 section 2.1, H2).
pub const HEADER_REGION_LEN: usize = 4096;
/// File offset of the first payload byte.
pub const PAYLOAD_OFFSET: usize = 4096;
/// Segment digests that fit between the struct and the seal slot:
/// (4064 - 256) / 32 = 119 (ADR 0003 section 20; 120 before the seal slot).
pub const MAX_SEGMENTS: usize = 119;
/// Region offset of the segment digest table.
pub const SEGMENT_TABLE_OFFSET: usize = HEADER_LEN;
/// Region offset of the 32-byte seal slot of a host object (ADR 0003 section
/// 19): `"AKS"`, sealed format version, 12-byte nonce, 16-byte tag. Zero in
/// a region `encode_region` returns; `seal::SealKey::seal_file` fills it.
pub const SEAL_SLOT_OFFSET: usize = HEADER_REGION_LEN - 32;
pub const FORMAT_VERSION: u16 = 1;
pub const RECORD_KIND_CHUNK: u16 = 1;
/// `flags` bit 0: the chunk holds fewer than `chunk_tokens` positions.
pub const FLAG_PARTIAL_TAIL: u16 = 1;
/// Header bytes covered by `header_sha256`.
pub const HASHED_PREFIX_LEN: usize = 224;
pub(crate) const PAYLOAD_DOMAIN: &[u8] = b"AIEN-PKV-PAYLOAD-V1\0";

/// The header fields a caller sets: section 2.1 minus the constants (magic,
/// version, kind, header_len) and minus the two hashes. `payload_sha256` is
/// known only after the payload is read, so it lives in `RecordHeader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub namespace_id: NamespaceId,
    pub chunk_key: ChunkKey,
    /// All zero for the first chunk.
    pub parent_key: [u8; 32],
    pub model_sha256: [u8; 32],
    pub first_position: u32,
    pub token_count: u32,
    pub chunk_tokens: u32,
    pub block_size: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: u8,
    pub backend_format: u8,
    pub flags: u16,
    pub payload_len: u64,
    /// Informational, not identity.
    pub written_unix_ns: u64,
}

/// A header plus the payload hash; what sits in the first 256 bytes of a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    pub chunk: ChunkHeader,
    pub payload_sha256: [u8; 32],
}

// Byte offsets of ADR 0003 section 2.1.
const O_MAGIC: usize = 0;
const O_VERSION: usize = 8;
const O_KIND: usize = 10;
const O_HEADER_LEN: usize = 12;
const O_NAMESPACE: usize = 16;
const O_KEY: usize = 48;
const O_PARENT: usize = 80;
const O_MODEL: usize = 112;
const O_FIRST_POS: usize = 144;
const O_TOKEN_COUNT: usize = 148;
const O_CHUNK_TOKENS: usize = 152;
const O_BLOCK_SIZE: usize = 156;
const O_LAYERS: usize = 160;
const O_KV_HEADS: usize = 164;
const O_HEAD_DIM: usize = 168;
const O_DTYPE: usize = 172;
const O_BACKEND: usize = 173;
const O_FLAGS: usize = 174;
const O_PAYLOAD_LEN: usize = 176;
const O_PAYLOAD_SHA: usize = 184;
const O_WRITTEN_NS: usize = 216;
const O_HEADER_SHA: usize = 224;

fn put(buf: &mut [u8; HEADER_LEN], off: usize, bytes: &[u8]) {
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

fn get32(buf: &[u8], off: usize) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&buf[off..off + 32]);
    out
}

fn get_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn get_u64(buf: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    u64::from_le_bytes(b)
}

impl RecordHeader {
    /// Serialize to exactly 256 bytes, little-endian, computing `header_sha256`.
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let c = &self.chunk;
        let mut b = [0u8; HEADER_LEN];
        put(&mut b, O_MAGIC, &MAGIC);
        put(&mut b, O_VERSION, &FORMAT_VERSION.to_le_bytes());
        put(&mut b, O_KIND, &RECORD_KIND_CHUNK.to_le_bytes());
        put(&mut b, O_HEADER_LEN, &(HEADER_LEN as u32).to_le_bytes());
        put(&mut b, O_NAMESPACE, &c.namespace_id.0);
        put(&mut b, O_KEY, c.chunk_key.as_bytes());
        put(&mut b, O_PARENT, &c.parent_key);
        put(&mut b, O_MODEL, &c.model_sha256);
        put(&mut b, O_FIRST_POS, &c.first_position.to_le_bytes());
        put(&mut b, O_TOKEN_COUNT, &c.token_count.to_le_bytes());
        put(&mut b, O_CHUNK_TOKENS, &c.chunk_tokens.to_le_bytes());
        put(&mut b, O_BLOCK_SIZE, &c.block_size.to_le_bytes());
        put(&mut b, O_LAYERS, &c.num_layers.to_le_bytes());
        put(&mut b, O_KV_HEADS, &c.num_kv_heads.to_le_bytes());
        put(&mut b, O_HEAD_DIM, &c.head_dim.to_le_bytes());
        b[O_DTYPE] = c.dtype;
        b[O_BACKEND] = c.backend_format;
        put(&mut b, O_FLAGS, &c.flags.to_le_bytes());
        put(&mut b, O_PAYLOAD_LEN, &c.payload_len.to_le_bytes());
        put(&mut b, O_PAYLOAD_SHA, &self.payload_sha256);
        put(&mut b, O_WRITTEN_NS, &c.written_unix_ns.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&b[..HASHED_PREFIX_LEN]).into();
        put(&mut b, O_HEADER_SHA, &digest);
        b
    }

    /// Structural decode of a record's header region: needs the whole 4096
    /// bytes (`Truncated` below that, ADR 0003 section 12 H6), then decodes the
    /// 256-byte struct. Padding and the segment root are checked by
    /// `validate_record`, which knows the segment count. No allocation.
    pub fn decode(buf: &[u8]) -> Result<RecordHeader, KvError> {
        if buf.len() < HEADER_REGION_LEN {
            return Err(KvError::Truncated);
        }
        Self::decode_struct(buf)
    }

    /// Structural decode of the 256-byte struct only (at least 256 bytes):
    /// length, magic, version, kind, header_len, header checksum, reserved flag
    /// bits. Identity checks are in `validate_record`. No allocation.
    pub fn decode_struct(buf: &[u8]) -> Result<RecordHeader, KvError> {
        if buf.len() < HEADER_LEN {
            return Err(KvError::Truncated);
        }
        if buf[O_MAGIC..O_MAGIC + 8] != MAGIC {
            return Err(KvError::BadMagic);
        }
        if get_u16(buf, O_VERSION) != FORMAT_VERSION {
            return Err(KvError::UnsupportedVersion);
        }
        if get_u16(buf, O_KIND) != RECORD_KIND_CHUNK {
            return Err(KvError::BadRecordKind);
        }
        if get_u32(buf, O_HEADER_LEN) as usize != HEADER_LEN {
            return Err(KvError::BadHeaderLen);
        }
        let digest = Sha256::digest(&buf[..HASHED_PREFIX_LEN]);
        if digest.as_slice() != &buf[O_HEADER_SHA..O_HEADER_SHA + 32] {
            return Err(KvError::HeaderChecksum);
        }
        let flags = get_u16(buf, O_FLAGS);
        if flags & !FLAG_PARTIAL_TAIL != 0 {
            return Err(KvError::BadFlags);
        }
        Ok(RecordHeader {
            chunk: ChunkHeader {
                namespace_id: NamespaceId(get32(buf, O_NAMESPACE)),
                chunk_key: ChunkKey(ObjectId(get32(buf, O_KEY))),
                parent_key: get32(buf, O_PARENT),
                model_sha256: get32(buf, O_MODEL),
                first_position: get_u32(buf, O_FIRST_POS),
                token_count: get_u32(buf, O_TOKEN_COUNT),
                chunk_tokens: get_u32(buf, O_CHUNK_TOKENS),
                block_size: get_u32(buf, O_BLOCK_SIZE),
                num_layers: get_u32(buf, O_LAYERS),
                num_kv_heads: get_u32(buf, O_KV_HEADS),
                head_dim: get_u32(buf, O_HEAD_DIM),
                dtype: buf[O_DTYPE],
                backend_format: buf[O_BACKEND],
                flags,
                payload_len: get_u64(buf, O_PAYLOAD_LEN),
                written_unix_ns: get_u64(buf, O_WRITTEN_NS),
            },
            payload_sha256: get32(buf, O_PAYLOAD_SHA),
        })
    }
}

/// `SHA-256("AIEN-PKV-PAYLOAD-V1\0" || seg_digest_0 || ... || seg_digest_{n-1})`,
/// the value stored at header offset 184 (ADR 0003 section 2.1, H4).
pub fn payload_root(seg_digests: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(PAYLOAD_DOMAIN);
    for d in seg_digests {
        h.update(d);
    }
    h.finalize().into()
}

/// Number of segments of a payload: `ceil(payload_len / segment_bytes)`.
/// `InvalidGeometry` for a zero size, `TooManySegments` above `MAX_SEGMENTS`.
pub fn segment_count(payload_len: u64, segment_bytes: u64) -> Result<u32, KvError> {
    if payload_len == 0 || segment_bytes == 0 {
        return Err(KvError::InvalidGeometry);
    }
    let n = payload_len.div_ceil(segment_bytes);
    if n > MAX_SEGMENTS as u64 {
        return Err(KvError::TooManySegments);
    }
    Ok(n as u32)
}

impl RecordHeader {
    /// Serialize the 4096-byte header region: the 256-byte struct, then the
    /// segment digest table, then zeros (ADR 0003 section 2.1, H5). Does not
    /// check that `payload_sha256` is the root of `seg_digests`; validation does.
    pub fn encode_region(
        &self,
        seg_digests: &[[u8; 32]],
    ) -> Result<[u8; HEADER_REGION_LEN], KvError> {
        if seg_digests.len() > MAX_SEGMENTS {
            return Err(KvError::TooManySegments);
        }
        let mut region = [0u8; HEADER_REGION_LEN];
        region[..HEADER_LEN].copy_from_slice(&self.encode());
        for (i, d) in seg_digests.iter().enumerate() {
            let at = SEGMENT_TABLE_OFFSET + 32 * i;
            region[at..at + 32].copy_from_slice(d);
        }
        Ok(region)
    }
}
