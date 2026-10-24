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

//! Error vocabulary for identity, codec and validation. Every variant is a
//! plain value (no allocation to build or report one).

use aien_platform::PlatformError;

/// Why a key, header or record was refused. Variants are listed in the order
/// `validate_record` checks them (ADR 0003 section 1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvError {
    /// Fewer than 4096 bytes were supplied for the header region (256 for
    /// `RecordHeader::decode_struct`).
    Truncated,
    BadMagic,
    UnsupportedVersion,
    BadRecordKind,
    BadHeaderLen,
    /// `header_sha256` does not match bytes 0..224.
    HeaderChecksum,
    /// A reserved flag bit is set, or `partial_tail` disagrees with `token_count`.
    BadFlags,
    /// A nonzero byte in the header region outside the 256-byte struct and the
    /// segment digest table (ADR 0003 section 2.1, H7).
    BadPadding,
    /// The root recomputed from the segment digest table differs from
    /// `payload_sha256` (offset 184).
    SegmentRootMismatch,
    /// More than `MAX_SEGMENTS` (119) segments: the table would not fit the region
    /// before the seal slot (ADR 0003 section 20).
    TooManySegments,
    NamespaceMismatch,
    KeyMismatch,
    ParentMismatch,
    PositionMismatch,
    /// `token_count` is 0 or larger than `chunk_tokens`.
    TokenCountOutOfRange,
    TokenCountMismatch,
    /// Model digest, chunk size, block size, layers, heads, head dim, dtype or backend differ.
    GeometryMismatch,
    PayloadLenMismatch,
    /// File size is not `4096 + payload_len`.
    FileSizeMismatch,
    // Construction errors (identity), not validation outcomes.
    /// A length-prefixed identity field is longer than its prefix can express or its cap.
    FieldTooLong(&'static str),
    /// A required identity field is empty (ADR 1.2.2: an empty engine build id or
    /// rounding class would let every engine alias).
    FieldEmpty(&'static str),
    /// Zero sizes, or `chunk_tokens` not a multiple of `block_size`.
    InvalidGeometry,
    /// A count, position or length does not fit its on-disk integer width.
    Overflow,
}

impl core::fmt::Display for KvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FieldTooLong(name) => write!(f, "kv identity field {name} is too long"),
            Self::FieldEmpty(name) => write!(f, "kv identity field {name} is empty"),
            other => write!(f, "kv record refused: {other:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KvError {}

impl From<KvError> for PlatformError {
    /// Every validation failure is an `InvalidLayout` at the platform boundary
    /// (ADR 0003 section 3.1).
    fn from(_: KvError) -> Self {
        PlatformError::InvalidLayout
    }
}
