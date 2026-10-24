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

//! `validate_record` refuses every corruption with the named error and without
//! touching the heap (ADR 0003 section 1.4, CUTS C1 red test
//! `validate_refuses_before_any_allocation`). This file is its own test binary
//! so its counting global allocator sees only this binary.

mod common;

use aien_kv_store::{validate_record, ChunkHeader, KvError, RecordHeader};
use common::*;
use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    ARMED.with(|a| {
        if a.get() {
            COUNT.with(|c| c.set(c.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

/// Run `f` and return its result with the number of heap allocations it made.
fn counted<T>(f: impl FnOnce() -> T) -> (T, usize) {
    COUNT.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    (out, COUNT.with(|c| c.get()))
}

fn reseal(b: &mut [u8; 4096]) {
    let d = Sha256::digest(&b[..224]);
    b[224..256].copy_from_slice(&d);
}

fn put32(b: &mut [u8; 4096], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

struct Fixture {
    expect: ChunkHeader,
    /// The 4096-byte header region.
    bytes: [u8; 4096],
    file_size: u64,
    segment_bytes: u64,
}

fn fixture() -> Fixture {
    let (rec, digests) = sample_record();
    let bytes = rec.encode_region(&digests).unwrap();
    Fixture {
        expect: rec.chunk,
        bytes,
        file_size: 4096 + rec.chunk.payload_len,
        segment_bytes: sample().segment_bytes(rec.chunk.token_count).unwrap(),
    }
}

fn check(f: &Fixture) -> Result<RecordHeader, KvError> {
    validate_record(&f.bytes, f.file_size, &f.expect, f.segment_bytes)
}

#[test]
fn counter_detects_allocation() {
    // Negative control: the harness would notice a heap use.
    let (_, n) = counted(|| std::hint::black_box(Vec::<u8>::with_capacity(64)));
    assert!(n >= 1, "counting allocator is not counting");
}

#[test]
fn valid_record_passes_with_zero_allocations() {
    let f = fixture();
    let (r, n) = counted(|| check(&f));
    assert_eq!(n, 0);
    let r = r.expect("valid record");
    assert_eq!(r.chunk, f.expect);
    assert_eq!(r.payload_sha256, sample_record().0.payload_sha256);
}

#[test]
fn validate_refuses_before_any_allocation() {
    type Corrupt = (&'static str, KvError, fn(&mut Fixture));
    let cases: Vec<Corrupt> = vec![
        ("wrong magic", KvError::BadMagic, |f| f.bytes[0] = b'X'),
        ("wrong version", KvError::UnsupportedVersion, |f| {
            f.bytes[8] = 2;
            reseal(&mut f.bytes);
        }),
        ("wrong header hash", KvError::HeaderChecksum, |f| {
            f.bytes[230] ^= 1;
        }),
        ("wrong namespace", KvError::NamespaceMismatch, |f| {
            f.bytes[16] ^= 1;
            reseal(&mut f.bytes);
        }),
        ("wrong key", KvError::KeyMismatch, |f| {
            f.bytes[48] ^= 1;
            reseal(&mut f.bytes);
        }),
        ("wrong parent", KvError::ParentMismatch, |f| {
            f.bytes[80] ^= 1;
            reseal(&mut f.bytes);
        }),
        ("wrong position", KvError::PositionMismatch, |f| {
            put32(&mut f.bytes, 144, 256);
            reseal(&mut f.bytes);
        }),
        ("token_count 0", KvError::TokenCountOutOfRange, |f| {
            put32(&mut f.bytes, 148, 0);
            reseal(&mut f.bytes);
        }),
        (
            "token_count over chunk_tokens",
            KvError::TokenCountOutOfRange,
            |f| {
                put32(&mut f.bytes, 148, 257);
                reseal(&mut f.bytes);
            },
        ),
        ("wrong geometry", KvError::GeometryMismatch, |f| {
            put32(&mut f.bytes, 164, 9); // num_kv_heads
            reseal(&mut f.bytes);
        }),
        ("wrong payload_len", KvError::PayloadLenMismatch, |f| {
            f.bytes[176] ^= 1;
            reseal(&mut f.bytes);
        }),
        (
            "file shorter than header+payload",
            KvError::FileSizeMismatch,
            |f| {
                f.file_size -= 1;
            },
        ),
        (
            "file longer than header+payload",
            KvError::FileSizeMismatch,
            |f| {
                f.file_size += 1;
            },
        ),
        (
            "nonzero byte at the end of the padding",
            KvError::BadPadding,
            |f| {
                f.bytes[4063] = 1;
            },
        ),
        (
            "nonzero byte right after the digest table",
            KvError::BadPadding,
            |f| {
                f.bytes[256 + 32 * 16] = 1;
            },
        ),
        (
            "segment digest flipped in the table",
            KvError::SegmentRootMismatch,
            |f| {
                f.bytes[256] ^= 1;
            },
        ),
        (
            "root at offset 184 flipped",
            KvError::SegmentRootMismatch,
            |f| {
                f.bytes[184] ^= 1;
                reseal(&mut f.bytes);
            },
        ),
        (
            "121 segments of the derived size",
            KvError::TooManySegments,
            |f| {
                f.segment_bytes = 69_905;
            },
        ),
        (
            "file size of the old 256-byte layout",
            KvError::FileSizeMismatch,
            |f| {
                f.file_size = 256 + f.expect.payload_len;
            },
        ),
        ("region cut short", KvError::Truncated, |_| {}),
        ("truncated header", KvError::Truncated, |_| {}),
    ];
    assert!(cases.len() >= 19);
    for (name, want, corrupt) in cases {
        let mut f = fixture();
        corrupt(&mut f);
        let slice: &[u8] = match name {
            "truncated header" => &f.bytes[..200],
            "region cut short" => &f.bytes[..4095],
            _ => &f.bytes,
        };
        let (got, allocs) =
            counted(|| validate_record(slice, f.file_size, &f.expect, f.segment_bytes));
        assert_eq!(got, Err(want), "{name}");
        assert_eq!(allocs, 0, "{name}: validation allocated");
    }
}

#[test]
fn validation_reports_the_first_failure_in_adr_order() {
    // Magic and namespace both wrong: magic wins.
    let mut f = fixture();
    f.bytes[0] = b'X';
    f.bytes[16] ^= 1;
    reseal(&mut f.bytes);
    assert_eq!(check(&f), Err(KvError::BadMagic));
    // Namespace and key both wrong (hash valid): namespace wins.
    let mut f = fixture();
    f.bytes[16] ^= 1;
    f.bytes[48] ^= 1;
    reseal(&mut f.bytes);
    assert_eq!(check(&f), Err(KvError::NamespaceMismatch));
}

#[test]
fn partial_flag_must_agree_with_token_count() {
    // A header claiming partial_tail on a full chunk is refused.
    let mut f = fixture();
    f.bytes[174] = 1;
    reseal(&mut f.bytes);
    assert_eq!(check(&f), Err(KvError::BadFlags));
    // A reserved flag bit is refused.
    let mut f = fixture();
    f.bytes[175] = 0x80;
    reseal(&mut f.bytes);
    assert_eq!(check(&f), Err(KvError::BadFlags));
}

#[test]
fn file_size_is_4096_plus_payload() {
    let f = fixture();
    assert_eq!(f.file_size, 4096 + f.expect.payload_len);
    assert!(check(&f).is_ok());
    // The C1 layout (256-byte header) is refused.
    let mut old = fixture();
    old.file_size = 256 + old.expect.payload_len;
    assert_eq!(check(&old), Err(KvError::FileSizeMismatch));
}

#[test]
fn validate_refuses_nonzero_padding() {
    // Every single byte outside the struct and the table must be zero.
    let f0 = fixture();
    let table_end = 256 + 32 * 16;
    for off in [table_end, table_end + 1, 1000, 4062, 4063] {
        let mut f = fixture();
        f.bytes[off] = 0x01;
        assert_eq!(check(&f), Err(KvError::BadPadding), "offset {off}");
    }
    // A nonzero digest byte inside the table is not padding.
    let mut f = fixture();
    f.bytes[table_end - 1] ^= 1;
    assert_eq!(check(&f), Err(KvError::SegmentRootMismatch));
    // The seal slot (4064..4096) is not padding: the seal module checks it.
    for off in [4064, 4080, 4095] {
        let mut f = fixture();
        f.bytes[off] = 0x01;
        assert!(check(&f).is_ok(), "seal slot offset {off}");
    }
    assert!(check(&f0).is_ok());
}

#[test]
fn validate_refuses_segment_root_mismatch() {
    // Swapping two digests keeps the multiset but changes the root.
    let mut f = fixture();
    let (a, b) = (256, 288);
    let first: [u8; 32] = f.bytes[a..a + 32].try_into().unwrap();
    let second: [u8; 32] = f.bytes[b..b + 32].try_into().unwrap();
    f.bytes[a..a + 32].copy_from_slice(&second);
    f.bytes[b..b + 32].copy_from_slice(&first);
    assert_eq!(check(&f), Err(KvError::SegmentRootMismatch));
    // A root that does not match the table (struct resealed) is refused too.
    let mut f = fixture();
    f.bytes[200] ^= 1;
    reseal(&mut f.bytes);
    assert_eq!(check(&f), Err(KvError::SegmentRootMismatch));
}

#[test]
fn validate_refuses_more_than_119_segments() {
    let f = fixture();
    let n_at = |sb: u64| f.expect.payload_len.div_ceil(sb);
    // 119 segments is the boundary (the table ends at the seal slot, 4064)
    // and is accepted when the table is whole.
    let sb119 = f.expect.payload_len.div_ceil(119);
    assert_eq!(n_at(sb119), 119);
    let digests = seg_digests("edge", 119);
    let rec = RecordHeader {
        chunk: f.expect,
        payload_sha256: aien_kv_store::payload_root(&digests),
    };
    let edge = Fixture {
        bytes: rec.encode_region(&digests).unwrap(),
        segment_bytes: sb119,
        expect: f.expect,
        file_size: f.file_size,
    };
    assert!(check(&edge).is_ok(), "exactly 119 segments is valid");
    // One more segment is refused before the table is read.
    let sb120 = f.expect.payload_len.div_ceil(120);
    assert_eq!(n_at(sb120), 120);
    let over = Fixture {
        segment_bytes: sb120,
        ..edge
    };
    let (got, allocs) = counted(|| check(&over));
    assert_eq!(got, Err(KvError::TooManySegments));
    assert_eq!(allocs, 0);
    // A zero segment size is a geometry error, not a panic.
    let zero = Fixture {
        segment_bytes: 0,
        ..over
    };
    assert_eq!(check(&zero), Err(KvError::InvalidGeometry));
}
