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

mod common;

use aien_kv_store::{
    payload_root, segment_count, KvError, RecordHeader, HEADER_LEN, HEADER_REGION_LEN, MAGIC,
    MAX_SEGMENTS, PAYLOAD_OFFSET,
};
use common::*;
use sha2::{Digest, Sha256};

fn record() -> RecordHeader {
    RecordHeader {
        chunk: sample_chunk_header(),
        payload_sha256: [0x5A; 32],
    }
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

#[test]
fn header_roundtrip_256_bytes() {
    let r = record();
    let bytes = r.encode();
    assert_eq!(bytes.len(), 256);
    assert_eq!(HEADER_LEN, 256);
    let back = RecordHeader::decode_struct(&bytes).unwrap();
    assert_eq!(back, r);
    // Re-encoding is byte for byte identical.
    assert_eq!(back.encode(), bytes);
    // Extra trailing bytes do not matter to the struct decode.
    let mut with_tail = bytes.to_vec();
    with_tail.extend_from_slice(&[1, 2, 3]);
    assert_eq!(RecordHeader::decode_struct(&with_tail).unwrap(), r);
}

/// Every field sits at the offset ADR 0003 section 2.1 gives.
#[test]
fn header_fields_sit_at_adr_offsets() {
    let r = record();
    let c = r.chunk;
    let b = r.encode();
    assert_eq!(&b[0..8], &MAGIC);
    assert_eq!(&b[0..8], b"AIENKVC1");
    assert_eq!(u16::from_le_bytes([b[8], b[9]]), 1); // format_version
    assert_eq!(u16::from_le_bytes([b[10], b[11]]), 1); // record_kind
    assert_eq!(u32_at(&b, 12), 256); // header_len
    assert_eq!(&b[16..48], &c.namespace_id.0);
    assert_eq!(&b[48..80], c.chunk_key.as_bytes());
    assert_eq!(&b[80..112], &c.parent_key);
    assert_eq!(&b[112..144], &c.model_sha256);
    assert_eq!(u32_at(&b, 144), c.first_position);
    assert_eq!(u32_at(&b, 148), c.token_count);
    assert_eq!(u32_at(&b, 152), c.chunk_tokens);
    assert_eq!(u32_at(&b, 156), c.block_size);
    assert_eq!(u32_at(&b, 160), c.num_layers);
    assert_eq!(u32_at(&b, 164), c.num_kv_heads);
    assert_eq!(u32_at(&b, 168), c.head_dim);
    assert_eq!(b[172], c.dtype);
    assert_eq!(b[173], c.backend_format);
    assert_eq!(u16::from_le_bytes([b[174], b[175]]), c.flags);
    assert_eq!(u64_at(&b, 176), c.payload_len);
    assert_eq!(&b[184..216], &r.payload_sha256);
    assert_eq!(u64_at(&b, 216), c.written_unix_ns);
    // header_sha256 covers bytes 0..224.
    let digest = Sha256::digest(&b[0..224]);
    assert_eq!(&b[224..256], digest.as_slice());
}

#[test]
fn decode_refuses_short_input_and_structural_damage() {
    let (rec, digests) = sample_record();
    let region = rec.encode_region(&digests).unwrap();
    // The region decode needs the whole 4096 bytes; the struct decode needs 256.
    assert_eq!(
        RecordHeader::decode(&region[..4095]),
        Err(KvError::Truncated)
    );
    assert_eq!(
        RecordHeader::decode(&region[..256]),
        Err(KvError::Truncated)
    );
    assert_eq!(RecordHeader::decode(&[]), Err(KvError::Truncated));
    assert_eq!(
        RecordHeader::decode_struct(&region[..255]),
        Err(KvError::Truncated)
    );
    let mut bad = region;
    bad[3] ^= 0xFF;
    assert_eq!(RecordHeader::decode(&bad), Err(KvError::BadMagic));
    let mut bad = region;
    bad[100] ^= 1; // inside the hashed prefix, magic untouched
    assert_eq!(RecordHeader::decode(&bad), Err(KvError::HeaderChecksum));
    let mut bad = region;
    bad[240] ^= 1; // inside the stored hash itself
    assert_eq!(RecordHeader::decode(&bad), Err(KvError::HeaderChecksum));
}

#[test]
fn header_region_roundtrip_4096() {
    let (rec, digests) = sample_record();
    assert_eq!(digests.len(), 16);
    assert_eq!(HEADER_REGION_LEN, 4096);
    assert_eq!(PAYLOAD_OFFSET, 4096);
    assert_eq!(MAX_SEGMENTS, 119);
    let region = rec.encode_region(&digests).unwrap();
    assert_eq!(region.len(), 4096);
    // Struct first, then the digest table, then zeros to the end.
    assert_eq!(&region[..256], &rec.encode());
    for (i, d) in digests.iter().enumerate() {
        assert_eq!(&region[256 + 32 * i..256 + 32 * (i + 1)], d, "digest {i}");
    }
    assert!(region[256 + 32 * 16..].iter().all(|b| *b == 0));
    // Decode reads the struct back; the root in the struct is the root of the table.
    let back = RecordHeader::decode(&region).unwrap();
    assert_eq!(back, rec);
    assert_eq!(back.payload_sha256, payload_root(&digests));
    // Re-encoding is byte for byte identical.
    assert_eq!(back.encode_region(&digests).unwrap(), region);

    // The table can fill the region up to the seal slot: 119 digests, no
    // padding left, the 32-byte slot zero until sealed (ADR 0003 section 20).
    let full = seg_digests("full", 119);
    let region = rec.encode_region(&full).unwrap();
    assert_eq!(&region[256 + 32 * 118..4064], full[118].as_slice());
    assert!(region[4064..].iter().all(|b| *b == 0));
    // 120 digests do not fit.
    let over = seg_digests("over", 120);
    assert_eq!(rec.encode_region(&over), Err(KvError::TooManySegments));
}

#[test]
fn payload_root_is_domain_separated_and_order_dependent() {
    let d = seg_digests("root", 3);
    let mut h = Sha256::new();
    h.update(b"AIEN-PKV-PAYLOAD-V1\0");
    for x in &d {
        h.update(x);
    }
    let want: [u8; 32] = h.finalize().into();
    assert_eq!(payload_root(&d), want);
    let mut swapped = d.clone();
    swapped.swap(0, 1);
    assert_ne!(payload_root(&swapped), want);
    assert_ne!(payload_root(&d[..2]), want);
    // Not the bare hash of the concatenation.
    let mut bare = Sha256::new();
    for x in &d {
        bare.update(x);
    }
    let bare: [u8; 32] = bare.finalize().into();
    assert_ne!(bare, want);
}

#[test]
fn segment_count_is_ceil_and_bounded() {
    assert_eq!(segment_count(524_288, 524_288), Ok(1));
    assert_eq!(segment_count(524_289, 524_288), Ok(2));
    assert_eq!(segment_count(119 * 4096, 4096), Ok(119));
    assert_eq!(
        segment_count(119 * 4096 + 1, 4096),
        Err(KvError::TooManySegments)
    );
    assert_eq!(segment_count(4096, 0), Err(KvError::InvalidGeometry));
    assert_eq!(segment_count(0, 4096), Err(KvError::InvalidGeometry));
}
