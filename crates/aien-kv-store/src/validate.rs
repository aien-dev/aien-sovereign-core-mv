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

//! Validation before any allocation or copy (ADR 0003 section 1.4).

use crate::error::KvError;
use crate::record::{
    segment_count, ChunkHeader, RecordHeader, FLAG_PARTIAL_TAIL, HEADER_REGION_LEN, PAYLOAD_DOMAIN,
    PAYLOAD_OFFSET, SEAL_SLOT_OFFSET, SEGMENT_TABLE_OFFSET,
};
use sha2::{Digest, Sha256};

/// Check `buf` (the first 4096 bytes of a record, the header region) and the
/// total `file_size` against what the caller expects. `segment_bytes` is the
/// payload segment size the caller derived from its own geometry
/// (`NamespaceFields::segment_bytes`; the file does not store it, ADR 2.1 item
/// 3). Order: length (4096), magic, version, kind, header_len, header
/// checksum, reserved flags (all in `RecordHeader::decode`), then namespace,
/// key, parent, position, token count, geometry, partial flag, payload_len,
/// segment count (`TooManySegments`), region padding (`BadPadding`), segment
/// root (`SegmentRootMismatch`), file size (`4096 + payload_len`). Performs no
/// heap allocation and copies no payload; the first failure is returned.
///
/// `TooManySegments` is checked before the padding and the root because both
/// need the table length; the error enum lists the three in a different order
/// (ADR 12 H7), which only matters for source order.
///
/// `expect` is the header the caller derived for the chunk it asked for
/// (`NamespaceFields::chunk_header`); its `written_unix_ns` is ignored.
pub fn validate_record(
    buf: &[u8],
    file_size: u64,
    expect: &ChunkHeader,
    segment_bytes: u64,
) -> Result<RecordHeader, KvError> {
    let rec = RecordHeader::decode(buf)?;
    let h = &rec.chunk;
    if h.namespace_id != expect.namespace_id {
        return Err(KvError::NamespaceMismatch);
    }
    if h.chunk_key != expect.chunk_key {
        return Err(KvError::KeyMismatch);
    }
    if h.parent_key != expect.parent_key {
        return Err(KvError::ParentMismatch);
    }
    if h.first_position != expect.first_position
        || h.chunk_tokens == 0
        || !h.first_position.is_multiple_of(h.chunk_tokens)
    {
        return Err(KvError::PositionMismatch);
    }
    if h.token_count == 0 || h.token_count > h.chunk_tokens {
        return Err(KvError::TokenCountOutOfRange);
    }
    if h.token_count != expect.token_count {
        return Err(KvError::TokenCountMismatch);
    }
    if h.model_sha256 != expect.model_sha256
        || h.chunk_tokens != expect.chunk_tokens
        || h.block_size != expect.block_size
        || h.num_layers != expect.num_layers
        || h.num_kv_heads != expect.num_kv_heads
        || h.head_dim != expect.head_dim
        || h.dtype != expect.dtype
        || h.backend_format != expect.backend_format
    {
        return Err(KvError::GeometryMismatch);
    }
    let partial = h.token_count < h.chunk_tokens;
    if (h.flags & FLAG_PARTIAL_TAIL != 0) != partial || h.flags != expect.flags {
        return Err(KvError::BadFlags);
    }
    if h.payload_len != expect.payload_len {
        return Err(KvError::PayloadLenMismatch);
    }

    // Header region: the segment digest table, zero padding up to the seal
    // slot, and the root. The seal slot (last 32 bytes) is not padding: the
    // seal module checks and authenticates it (ADR 0003 section 20).
    let n_seg = segment_count(h.payload_len, segment_bytes)? as usize;
    let table_end = SEGMENT_TABLE_OFFSET + 32 * n_seg;
    let region = &buf[..HEADER_REGION_LEN];
    if region[table_end..SEAL_SLOT_OFFSET].iter().any(|b| *b != 0) {
        return Err(KvError::BadPadding);
    }
    let mut root = Sha256::new();
    root.update(PAYLOAD_DOMAIN);
    root.update(&region[SEGMENT_TABLE_OFFSET..table_end]);
    if root.finalize().as_slice() != rec.payload_sha256 {
        return Err(KvError::SegmentRootMismatch);
    }

    match (PAYLOAD_OFFSET as u64).checked_add(h.payload_len) {
        Some(total) if total == file_size => Ok(rec),
        _ => Err(KvError::FileSizeMismatch),
    }
}
