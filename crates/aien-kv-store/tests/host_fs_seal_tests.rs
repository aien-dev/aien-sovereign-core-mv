#![cfg(feature = "std")]
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

//! Encryption at rest in the host filesystem adapter (ADR 0003 section 20, #506).
//! Every refusal case must end with the object discarded and the destination
//! untouched, so the caller recomputes; nothing is ever restored from it.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::HostFsStore;
use aien_kv_store::record::{HEADER_LEN, HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use aien_kv_store::seal::SealKey;
use aien_kv_store::{ChunkKey, KvStore, RecordHeader, Terminal};
use aien_platform::PlatformError;
use rig::*;
use std::sync::Arc;

/// The load of `key` fails `InvalidLayout`, writes nothing into the
/// destination, and the object is gone from disk and index.
fn assert_discarded(fx: &Fx, s: &HostFsStore, key: ChunkKey, n: u32) {
    let (_, header) = fx.chunk(n, CHUNK_TOKENS);
    let (t, mem) = fx.load(s, key, &header);
    assert_eq!(
        t.unwrap(),
        Terminal::Failed(PlatformError::InvalidLayout),
        "a refused object must fail the load"
    );
    assert!(
        mem.bytes().iter().all(|b| *b == 0xEE),
        "nothing may reach the destination before the object is authenticated"
    );
    assert!(!fx.object_file(&key).exists(), "the object is discarded");
    assert_eq!(s.lookup(&[key]), vec![None]);
}

/// Store object `n`, drop the store, flip `mutate` over the file, reopen
/// (same key unless `reopen_key`), recover.
fn stored_then_reopened(
    fx: &Fx,
    n: u32,
    mutate: impl FnOnce(&mut Vec<u8>),
    reopen_key: Option<Arc<SealKey>>,
) -> (HostFsStore, ChunkKey) {
    let s = fx.open();
    let p = fx.put_full(&s, n);
    assert!(p.job.is_ok());
    drop(s);
    let path = fx.object_file(&p.key);
    let mut bytes = std::fs::read(&path).unwrap();
    mutate(&mut bytes);
    std::fs::write(&path, bytes).unwrap();
    let s = fx.open_with(|c| {
        if let Some(k) = reopen_key {
            c.seal_key = Some(k);
        }
    });
    (s, p.key)
}

#[test]
fn host_fs_sealed_object_round_trips_and_is_not_plaintext_on_disk() {
    let fx = Fx::new();
    let s = fx.open();
    let p = fx.put_full(&s, 1);
    let on_disk = std::fs::read(fx.object_file(&p.key)).unwrap();
    assert_eq!(on_disk.len() as u64, OBJ, "sealing adds no bytes");
    assert_ne!(&on_disk[HEADER_REGION_LEN..], p.expected.as_slice());
    assert!(aien_kv_store::seal::is_sealed_region(
        &on_disk[..HEADER_REGION_LEN]
    ));
    let plain = read_object_plain(&fx.object_file(&p.key));
    assert_eq!(&plain[HEADER_REGION_LEN..], p.expected.as_slice());
    drop(s);
    // Survives a restart with the same key.
    let s = fx.open();
    let (t, mem) = fx.load(&s, p.key, &p.header);
    assert_eq!(
        t.unwrap(),
        Terminal::Done(aien_kv_store::JobStage::Completed)
    );
    assert_eq!(mem.bytes(), p.expected);
}

#[test]
fn host_fs_wrong_key_discards_and_never_restores() {
    let fx = Fx::new();
    let other = Arc::new(SealKey::generate().unwrap());
    let (s, key) = stored_then_reopened(&fx, 2, |_| {}, Some(other));
    assert_discarded(&fx, &s, key, 2);
}

#[test]
fn host_fs_tampered_ciphertext_byte_discards() {
    let fx = Fx::new();
    let (s, key) = stored_then_reopened(&fx, 3, |b| b[HEADER_REGION_LEN + 77] ^= 1, None);
    assert_discarded(&fx, &s, key, 3);
}

#[test]
fn host_fs_tampered_tag_discards() {
    let fx = Fx::new();
    let (s, key) = stored_then_reopened(&fx, 4, |b| b[HEADER_REGION_LEN - 1] ^= 0x40, None);
    assert_discarded(&fx, &s, key, 4);
}

#[test]
fn host_fs_identity_header_swapped_to_another_namespace_discards() {
    // Store A (sharing namespace "default") writes object 5. The file is
    // moved into store B (namespace "other", same store key) with its header
    // rewritten to B's identity for the same tokens: the header checksum and
    // every structural check pass, so only the bound identity can refuse it.
    let a = Fx::new();
    let s = a.open();
    let p = a.put_full(&s, 5);
    drop(s);
    let sealed = std::fs::read(a.object_file(&p.key)).unwrap();

    let mut b = Fx::new();
    b.fields.sharing_namespace = "other";
    b.ns = b.fields.namespace_id().unwrap();
    assert_ne!(a.ns, b.ns);
    b.prime();
    let (key_b, header_b) = b.chunk(5, CHUNK_TOKENS);
    let rec_a = RecordHeader::decode(&sealed).unwrap();
    let rec_b = RecordHeader {
        chunk: header_b,
        payload_sha256: rec_a.payload_sha256,
    };
    let mut moved = sealed.clone();
    moved[..HEADER_LEN].copy_from_slice(&rec_b.encode());
    std::fs::write(b.object_file(&key_b), &moved).unwrap();

    let s = b.open();
    assert_eq!(
        s.lookup(&[key_b]),
        vec![Some(FULL_PAYLOAD)],
        "the moved file passes every header check"
    );
    assert_discarded(&b, &s, key_b, 5);
}

#[test]
fn host_fs_old_plaintext_format_object_is_discarded_at_recovery() {
    let fx = Fx::new();
    fx.prime();
    let (key, header) = fx.chunk(6, CHUNK_TOKENS);
    let plain = plain_object_bytes(&header, &expected_payload(6, CHUNK_TOKENS), BS);
    assert!(plain[SEAL_SLOT_OFFSET..HEADER_REGION_LEN]
        .iter()
        .all(|b| *b == 0));
    std::fs::write(fx.object_file(&key), plain).unwrap();
    let s = fx.open_raw(|_| {});
    let p = recover_all(&s);
    assert_eq!(p.corrupt_deleted, 1, "{p:?}");
    assert_eq!(p.indexed, 0);
    assert!(!fx.object_file(&key).exists());
    assert_eq!(s.lookup(&[key]), vec![None]);
}

#[test]
fn host_fs_open_without_a_key_is_refused_and_touches_nothing() {
    let fx = Fx::new();
    let mut c = fx.config();
    c.seal_key = None;
    let got = HostFsStore::open(c);
    assert_eq!(got.err(), Some(PlatformError::AccessDenied));
    assert!(
        std::fs::read_dir(fx.root()).unwrap().next().is_none(),
        "no directory, STORE_ID or object is created without a key"
    );
}

#[test]
fn host_fs_config_debug_never_prints_the_key() {
    let fx = Fx::new();
    let shown = format!("{:?}", fx.config());
    assert!(!shown.contains("SealKey("), "{shown}");
    let k = format!("{:?}", test_key());
    assert!(k.contains("redacted"), "{k}");
}

#[test]
fn host_fs_identity_header_rewritten_in_place_discards() {
    // The header of a sealed object is rewritten where it lies (same store,
    // same file name) to a different but self-consistent header: a new
    // `written_unix_ns` and a recomputed header checksum. Every structural and
    // identity check still passes, so only the AEAD binding can refuse it.
    let fx = Fx::new();
    let (s, key) = stored_then_reopened(
        &fx,
        7,
        |b| {
            let mut rec = RecordHeader::decode(b).unwrap();
            rec.chunk.written_unix_ns ^= 0xFFFF;
            b[..HEADER_LEN].copy_from_slice(&rec.encode());
        },
        None,
    );
    assert_eq!(
        s.lookup(&[key]),
        vec![Some(FULL_PAYLOAD)],
        "the rewritten header passes every check at recovery"
    );
    assert_discarded(&fx, &s, key, 7);
}
