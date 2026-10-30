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

//! The sealing module (ADR 0003 section 20): AEAD over the payload with the
//! record's header region bound as associated data. Every negative case must
//! refuse; a refusal means the caller discards the object and recomputes.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::record::{HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use aien_kv_store::seal::{self, KeyRefusal, SealError, SealKey, KEY_ENV};
use aien_kv_store::{payload_root, RecordHeader};
use rig::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// `region || payload` for object `n` of the rig geometry, unsealed.
fn plain_file(n: u32) -> (Vec<u8>, Vec<u8>) {
    let fx = Fx::new();
    let (_, header) = fx.chunk(n, CHUNK_TOKENS);
    let payload = expected_payload(n, CHUNK_TOKENS);
    let digests: Vec<[u8; 32]> = payload.chunks(BS).map(sha).collect();
    let rec = RecordHeader {
        chunk: header,
        payload_sha256: payload_root(&digests),
    };
    let mut file = rec.encode_region(&digests).unwrap().to_vec();
    file.extend_from_slice(&payload);
    (file, payload)
}

/// Removed when the value drops, including when a seal assertion panics.
struct UsageDir(std::path::PathBuf);

impl UsageDir {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("aien-kv-seal-tests-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        UsageDir(p)
    }
}

impl Drop for UsageDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sealed_file(key: &SealKey, n: u32) -> (Vec<u8>, Vec<u8>) {
    let dir = UsageDir::new();
    let (mut file, payload) = plain_file(n);
    key.seal_file(&dir.0, &mut file).unwrap();
    (file, payload)
}

/// Open `file` (sealed `region || ciphertext`) in a copy; the plaintext on success.
fn open(key: &SealKey, file: &[u8]) -> Result<Vec<u8>, SealError> {
    let (region, body) = file.split_at(HEADER_REGION_LEN);
    let mut body = body.to_vec();
    key.open_payload(region, &mut body)?;
    Ok(body)
}

fn key_hex(b: u8) -> String {
    format!("{b:02x}").repeat(32)
}

#[test]
fn seal_round_trip_restores_the_payload_and_hides_it_on_disk() {
    let key = SealKey::generate().unwrap();
    let (file, payload) = sealed_file(&key, 1);
    assert_eq!(
        file.len(),
        HEADER_REGION_LEN + payload.len(),
        "sealing adds no bytes"
    );
    assert_ne!(
        &file[HEADER_REGION_LEN..],
        payload.as_slice(),
        "payload is not plaintext"
    );
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(open(&key, &file).unwrap(), payload);
    // The header struct and the digest table stay readable for recovery.
    let (plain, _) = plain_file(1);
    assert_eq!(&file[..SEAL_SLOT_OFFSET], &plain[..SEAL_SLOT_OFFSET]);
}

#[test]
fn seal_uses_a_fresh_nonce_per_file() {
    let key = SealKey::generate().unwrap();
    let mut nonces = HashSet::new();
    let mut bodies = HashSet::new();
    for _ in 0..512 {
        let (file, _) = sealed_file(&key, 2);
        let nonce = file[SEAL_SLOT_OFFSET + 4..SEAL_SLOT_OFFSET + 16].to_vec();
        assert!(nonces.insert(nonce), "nonce repeated");
        assert!(
            bodies.insert(file[HEADER_REGION_LEN..].to_vec()),
            "ciphertext repeated"
        );
    }
}

#[test]
fn seal_refuses_a_wrong_key() {
    let a = SealKey::generate().unwrap();
    let b = SealKey::generate().unwrap();
    let (file, _) = sealed_file(&a, 3);
    assert_eq!(open(&b, &file), Err(SealError::Rejected));
}

#[test]
fn seal_refuses_a_tampered_ciphertext_byte() {
    let key = SealKey::generate().unwrap();
    let (file, _) = sealed_file(&key, 4);
    for at in [HEADER_REGION_LEN, HEADER_REGION_LEN + 5000, file.len() - 1] {
        let mut bad = file.clone();
        bad[at] ^= 0x01;
        assert_eq!(open(&key, &bad), Err(SealError::Rejected), "offset {at}");
    }
}

#[test]
fn seal_refuses_a_tampered_tag_or_nonce() {
    let key = SealKey::generate().unwrap();
    let (file, _) = sealed_file(&key, 5);
    for at in [
        SEAL_SLOT_OFFSET + 4,
        SEAL_SLOT_OFFSET + 16,
        HEADER_REGION_LEN - 1,
    ] {
        let mut bad = file.clone();
        bad[at] ^= 0x80;
        assert_eq!(
            open(&key, &bad),
            Err(SealError::Rejected),
            "slot offset {at}"
        );
    }
}

#[test]
fn seal_refuses_a_swapped_identity_header() {
    // Object 6's ciphertext under object 7's (valid, same length) header region.
    let key = SealKey::generate().unwrap();
    let (six, _) = sealed_file(&key, 6);
    let (seven, _) = sealed_file(&key, 7);
    let mut swapped = seven[..SEAL_SLOT_OFFSET].to_vec();
    swapped.extend_from_slice(&six[SEAL_SLOT_OFFSET..]);
    assert_eq!(open(&key, &swapped), Err(SealError::Rejected));
    // One identity byte of the header struct (namespace id at offset 16).
    let mut ns = six.clone();
    ns[16] ^= 1;
    assert_eq!(open(&key, &ns), Err(SealError::Rejected));
    // One byte of the segment digest table.
    let mut table = six.clone();
    table[300] ^= 1;
    assert_eq!(open(&key, &table), Err(SealError::Rejected));
}

#[test]
fn seal_refuses_an_old_plaintext_format_file() {
    let key = SealKey::generate().unwrap();
    let (plain, _) = plain_file(8);
    assert!(!seal::is_sealed_region(&plain[..HEADER_REGION_LEN]));
    assert_eq!(open(&key, &plain), Err(SealError::NotSealed));
    // A sealed marker with another format version is not read either.
    let (mut other, _) = sealed_file(&key, 8);
    other[SEAL_SLOT_OFFSET + 3] = 1;
    assert!(!seal::is_sealed_region(&other[..HEADER_REGION_LEN]));
    assert_eq!(open(&key, &other), Err(SealError::UnsupportedVersion));
}

#[test]
fn seal_and_open_time_is_counted_per_payload_byte() {
    // Totals are process wide and other tests run in parallel: check growth by at least ours.
    let key = SealKey::generate().unwrap();
    let before = seal::seal_timing();
    let (file, payload) = sealed_file(&key, 3);
    open(&key, &file).unwrap();
    let after = seal::seal_timing();
    let n = payload.len() as u64;
    assert!(after.seal_calls > before.seal_calls && after.open_calls > before.open_calls);
    assert!(
        after.seal_bytes - before.seal_bytes >= n,
        "{before:?} {after:?}"
    );
    assert!(
        after.open_bytes - before.open_bytes >= n,
        "{before:?} {after:?}"
    );
    assert!(after.seal_ns > before.seal_ns && after.open_ns > before.open_ns);
}

#[test]
fn seal_refusals_are_counted() {
    let key = SealKey::generate().unwrap();
    let other = SealKey::generate().unwrap();
    let (file, _) = sealed_file(&key, 9);
    let before = seal::open_rejected_total();
    assert!(open(&other, &file).is_err());
    assert!(seal::open_rejected_total() > before);
}

#[test]
fn key_from_hex_accepts_exactly_64_hex_digits() {
    let k = SealKey::from_hex(&key_hex(0xab)).unwrap();
    assert_eq!(k, SealKey::from_bytes([0xab; 32]));
    assert_eq!(SealKey::from_hex(&key_hex(0xAB).to_uppercase()).unwrap(), k);
}

#[test]
fn key_refuses_malformed_values() {
    let good = key_hex(0x11);
    for bad in [
        String::new(),
        good[..62].to_string(),
        good[..63].to_string(),
        format!("{good}0"),
        format!("{good}00"),
        format!("{}zz", &good[..62]),
        format!(" {}", &good[..63]),
        format!("{good}\n"),
    ] {
        assert_eq!(
            SealKey::from_hex(&bad).unwrap_err(),
            KeyRefusal::Malformed,
            "{} chars",
            bad.len()
        );
    }
}

#[test]
fn key_refuses_a_missing_value() {
    let got = SealKey::from_lookup(|_| None);
    assert_eq!(got.unwrap_err(), KeyRefusal::Missing);
    let got = SealKey::from_lookup(|n| (n == KEY_ENV).then(|| "nothex".to_string()));
    assert_eq!(got.unwrap_err(), KeyRefusal::Malformed);
}

#[test]
fn key_debug_and_refusal_text_never_show_the_key() {
    let hex = key_hex(0xc3);
    let k = SealKey::from_hex(&hex).unwrap();
    let shown = format!("{k:?}");
    assert!(!shown.contains("c3"), "{shown}");
    assert!(!shown.contains("195"), "{shown}");
    let bad = format!("{}x", &hex[..63]);
    let refusal = SealKey::from_hex(&bad).unwrap_err();
    assert!(!format!("{refusal} {refusal:?}").contains("c3c3"));
}

#[test]
fn key_taken_from_the_process_environment_is_removed_from_it() {
    // The only test that touches this variable.
    std::env::set_var(KEY_ENV, key_hex(0x5e));
    let k = SealKey::take_from_process_env().unwrap();
    assert_eq!(k, SealKey::from_bytes([0x5e; 32]));
    assert!(
        std::env::var_os(KEY_ENV).is_none(),
        "key left in the environment"
    );
    assert_eq!(
        SealKey::take_from_process_env().unwrap_err(),
        KeyRefusal::Missing
    );
    // A malformed value is removed too, and refused.
    std::env::set_var(KEY_ENV, "abc");
    assert_eq!(
        SealKey::take_from_process_env().unwrap_err(),
        KeyRefusal::Malformed
    );
    assert!(std::env::var_os(KEY_ENV).is_none());
}
