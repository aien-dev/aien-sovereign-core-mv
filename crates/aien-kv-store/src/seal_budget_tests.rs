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

//! Per-key seal budget (#811). The count is installed through
//! [`testing_install_usage_count`] so a test does not seal `2^20` files.
//!
//! Reproduction:
//! `cargo test -p aien-kv-store --lib seal_budget_tests -- --test-threads=8`

use super::seal_nonce_tests::{
    assert_pattern_absent, kvc_files, open_at, publish, terminal_for, test_fields, Tmp,
    SYNTHETIC_KEY,
};
use crate::host_fs::HostFsStore;
use crate::identity::NamespaceFields;
use crate::pin::{HolderGuard, PinnedSource};
use crate::record::HEADER_REGION_LEN;
use crate::seal::{
    self, testing_install_usage_count, testing_read_key_fingerprints, testing_read_usage_count,
    testing_usage_path, NonceSource, SealError, SealKey, RECOMMENDED_SEALS_PER_KEY, USAGE_DIR_NAME,
};
use crate::{chunk_chain, Incarnation, KvStore, Terminal};
use aien_platform::PlatformError;
use sha2::{Digest, Sha256};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::sync::{Arc, Barrier};

const BS: usize = 8192;

fn blank() -> Vec<u8> {
    vec![0x11u8; HEADER_REGION_LEN + 32]
}

fn hex_fingerprint(key: &[u8; 32]) -> String {
    let dig = Sha256::digest(key);
    let mut hex = String::with_capacity(64);
    const NIBBLE: &[u8; 16] = b"0123456789abcdef";
    for byte in dig {
        hex.push(NIBBLE[(byte >> 4) as usize] as char);
        hex.push(NIBBLE[(byte & 0x0f) as usize] as char);
    }
    hex
}

struct FailNonce;

impl NonceSource for FailNonce {
    fn fill_nonce(&self, _out: &mut [u8; seal::NONCE_LEN]) -> Result<(), SealError> {
        Err(SealError::Random)
    }
}

#[test]
fn seal_budget_refuses_at_the_installed_limit_and_leaves_the_buffer() {
    let dir = Tmp::new("at-limit");
    let key = SealKey::from_bytes([0x11; 32]);
    testing_install_usage_count(&dir.0, &key, RECOMMENDED_SEALS_PER_KEY).unwrap();
    let mut file = blank();
    let before = file.clone();
    assert_eq!(
        key.seal_file(&dir.0, &mut file),
        Err(SealError::Budget),
        "the 2^20 count is already spent"
    );
    assert_eq!(file, before);
    assert!(!seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(
        testing_read_usage_count(&dir.0, &key).unwrap(),
        RECOMMENDED_SEALS_PER_KEY
    );
}

#[test]
fn seal_budget_last_allowed_seal_persists_across_a_new_key() {
    let dir = Tmp::new("reload");
    let key = SealKey::from_bytes([0x12; 32]);
    testing_install_usage_count(&dir.0, &key, RECOMMENDED_SEALS_PER_KEY - 1).unwrap();
    let mut file = blank();
    key.seal_file(&dir.0, &mut file).unwrap();
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    let (region, body) = file.split_at(HEADER_REGION_LEN);
    let mut body = body.to_vec();
    key.open_payload(region, &mut body).unwrap();
    assert_eq!(body, vec![0x11u8; 32]);

    let reloaded = SealKey::from_bytes([0x12; 32]);
    assert_eq!(
        testing_read_usage_count(&dir.0, &reloaded).unwrap(),
        RECOMMENDED_SEALS_PER_KEY,
        "a new SealKey reads the count from the directory"
    );
    let mut next = blank();
    let before = next.clone();
    assert_eq!(
        reloaded.seal_file(&dir.0, &mut next),
        Err(SealError::Budget)
    );
    assert_eq!(next, before);
    assert_eq!(
        testing_read_usage_count(&dir.0, &reloaded).unwrap(),
        RECOMMENDED_SEALS_PER_KEY
    );
}

#[test]
fn seal_budget_persist_fault_stops_the_seal_and_keeps_the_count() {
    let dir = Tmp::new("fault");
    let key = SealKey::from_bytes([0x13; 32]);
    testing_install_usage_count(&dir.0, &key, 4).unwrap();
    key.testing_fail_next_usage_persist();
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&dir.0, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert!(!seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(testing_read_usage_count(&dir.0, &key).unwrap(), 4);
    // The fault is one shot. The next seal commits.
    key.seal_file(&dir.0, &mut file).unwrap();
    assert_eq!(testing_read_usage_count(&dir.0, &key).unwrap(), 5);
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
}

#[test]
fn seal_budget_usage_path_that_is_a_file_does_not_seal() {
    let dir = Tmp::new("not-a-dir");
    let bogus = dir.0.join("not-a-directory");
    std::fs::write(&bogus, b"x").unwrap();
    let key = SealKey::from_bytes([0x14; 32]);
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&bogus, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert_eq!(std::fs::read(&bogus).unwrap(), b"x");
}

#[test]
fn seal_budget_corrupt_count_is_not_reset_and_does_not_seal() {
    let dir = Tmp::new("corrupt");
    let key = SealKey::from_bytes([0x15; 32]);
    testing_install_usage_count(&dir.0, &key, 7).unwrap();
    let path = testing_usage_path(&dir.0, &key);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&dir.0, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn seal_budget_short_buffer_does_not_create_a_count() {
    let dir = Tmp::new("short");
    let key = SealKey::from_bytes([0x16; 32]);
    let mut short = vec![1u8; 8];
    assert_eq!(key.seal_file(&dir.0, &mut short), Err(SealError::Short));
    assert!(!testing_usage_path(&dir.0, &key).exists());
    assert!(
        std::fs::read_dir(&dir.0).unwrap().next().is_none(),
        "a short buffer must not create LOCK or a count file"
    );
    assert_eq!(testing_read_usage_count(&dir.0, &key).unwrap(), 0);
}

#[test]
fn seal_budget_random_failure_does_not_consume_a_count() {
    let dir = Tmp::new("random");
    let key = SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(FailNonce));
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&dir.0, &mut file), Err(SealError::Random));
    assert_eq!(file, before);
    assert_eq!(testing_read_usage_count(&dir.0, &key).unwrap(), 0);
    assert!(!testing_usage_path(&dir.0, &key).exists());
}

#[test]
fn seal_budget_is_per_key_fingerprint() {
    let dir = Tmp::new("two-keys");
    let a = SealKey::from_bytes([0x17; 32]);
    let b = SealKey::from_bytes([0x18; 32]);
    testing_install_usage_count(&dir.0, &a, RECOMMENDED_SEALS_PER_KEY).unwrap();
    let mut file = blank();
    assert_eq!(a.seal_file(&dir.0, &mut file), Err(SealError::Budget));
    b.seal_file(&dir.0, &mut file).unwrap();
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(
        testing_read_usage_count(&dir.0, &a).unwrap(),
        RECOMMENDED_SEALS_PER_KEY
    );
    assert_eq!(testing_read_usage_count(&dir.0, &b).unwrap(), 1);
    let expect = dir.0.join(hex_fingerprint(&[0x18; 32]));
    assert_eq!(testing_usage_path(&dir.0, &b), expect);
    let mode = std::fs::metadata(&expect).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[test]
fn seal_budget_two_threads_each_commit_a_count() {
    let dir = Tmp::new("threads");
    let key = Arc::new(SealKey::from_bytes([0x19; 32]));
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let key = Arc::clone(&key);
        let barrier = Arc::clone(&barrier);
        let path = dir.0.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let mut file = blank();
            key.seal_file(&path, &mut file).unwrap();
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(testing_read_usage_count(&dir.0, &key).unwrap(), 2);
}

#[test]
fn seal_budget_host_count_survives_reopen_and_then_refuses() {
    let root = Tmp::new("host");
    let fields = test_fields();
    let usage = root.0.join(USAGE_DIR_NAME);
    let s = open_at(&root.0, Arc::new(SealKey::from_bytes(SYNTHETIC_KEY)));
    assert!(usage.is_dir(), "open creates seal-usage");
    let reader = SealKey::from_bytes(SYNTHETIC_KEY);
    assert_eq!(testing_read_usage_count(&usage, &reader).unwrap(), 0);
    let first = publish(&s, &fields, 1);
    assert_eq!(testing_read_usage_count(&usage, &reader).unwrap(), 1);
    drop(s);

    let s = open_at(&root.0, Arc::new(SealKey::from_bytes(SYNTHETIC_KEY)));
    assert_eq!(
        testing_read_usage_count(&usage, &SealKey::from_bytes(SYNTHETIC_KEY)).unwrap(),
        1,
        "reopen sees the count from the previous store"
    );
    let _second = publish(&s, &fields, 2);
    assert_eq!(testing_read_usage_count(&usage, &reader).unwrap(), 2);
    drop(s);

    testing_install_usage_count(&usage, &reader, RECOMMENDED_SEALS_PER_KEY).unwrap();
    let s = open_at(&root.0, Arc::new(SealKey::from_bytes(SYNTHETIC_KEY)));
    let kept = kvc_files(&root.0);
    assert_eq!(kept.len(), 2);
    let payload = refused_store(&s, &fields, 3);
    assert_eq!(kvc_files(&root.0), kept);
    assert!(!s.object_path(&payload.0).exists());
    assert_pattern_absent(&root.0, &payload.1[..64]);
    assert_eq!(
        testing_read_usage_count(&usage, &reader).unwrap(),
        RECOMMENDED_SEALS_PER_KEY
    );
    let _ = first;
    drop(s);
}

fn store_usage(tag: &str) -> (Tmp, std::path::PathBuf) {
    let root = Tmp::new(tag);
    let usage = root.0.join(USAGE_DIR_NAME);
    std::fs::create_dir(&usage).unwrap();
    (root, usage)
}

fn plant_kvc(root: &std::path::Path, shard: &str, name: &str, sealed: bool) {
    let dir = root.join("objects").join(shard);
    std::fs::create_dir_all(&dir).unwrap();
    let mut region = vec![0u8; HEADER_REGION_LEN];
    if sealed {
        let at = crate::record::SEAL_SLOT_OFFSET;
        region[at..at + 3].copy_from_slice(&seal::SEAL_MAGIC);
        region[at + 3] = seal::SEALED_FORMAT_VERSION;
    }
    std::fs::write(dir.join(name), region).unwrap();
}

#[test]
fn seal_budget_missing_count_beside_a_sealed_object_does_not_seal() {
    let (root, usage) = store_usage("occupied");
    plant_kvc(
        &root.0,
        "ab",
        "ab00000000000000000000000000000000000000000000000000000000000000.kvc",
        true,
    );
    let key = SealKey::from_bytes([0x31; 32]);
    let mut file = blank();
    let before = file.clone();
    assert_eq!(
        key.seal_file(&usage, &mut file),
        Err(SealError::UsageStore),
        "a missing counter next to a sealed object must not restart at zero"
    );
    assert_eq!(file, before);
    assert!(!seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert!(!testing_usage_path(&usage, &key).exists());
}

#[test]
fn seal_budget_missing_count_on_an_empty_store_starts_at_zero() {
    let (root, usage) = store_usage("empty-store");
    std::fs::create_dir_all(root.0.join("objects/00")).unwrap();
    // An old plaintext object is not a seal, so it does not block the zero start.
    plant_kvc(
        &root.0,
        "cd",
        "cd00000000000000000000000000000000000000000000000000000000000000.kvc",
        false,
    );
    let key = SealKey::from_bytes([0x32; 32]);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 0);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
}

#[test]
fn seal_budget_leftover_part_is_replaced_when_we_own_it_and_refused_when_linked() {
    let (root, usage) = store_usage("part");
    let key = SealKey::from_bytes([0x33; 32]);
    testing_install_usage_count(&usage, &key, 4).unwrap();
    let count_path = testing_usage_path(&usage, &key);
    let part = count_path.with_extension("part");
    std::fs::write(&part, b"not-a-count").unwrap();
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    assert!(
        !part.exists(),
        "an owned leftover part is renamed onto the count file"
    );
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 5);
    assert!(!std::fs::read(&count_path)
        .unwrap()
        .windows(11)
        .any(|w| w == b"not-a-count"));
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    drop(root);

    let (_root, usage) = store_usage("part-link");
    let key = SealKey::from_bytes([0x34; 32]);
    testing_install_usage_count(&usage, &key, 4).unwrap();
    let count_path = testing_usage_path(&usage, &key);
    let part = count_path.with_extension("part");
    std::os::unix::fs::symlink(&count_path, &part).unwrap();
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&usage, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert!(
        std::fs::symlink_metadata(&part)
            .unwrap()
            .file_type()
            .is_symlink(),
        "a linked part is left in place"
    );
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 4);
}

fn fp_bytes(key: &[u8; 32]) -> [u8; 32] {
    Sha256::digest(key).into()
}

#[test]
fn seal_budget_rotation_starts_the_new_key_at_one_and_keeps_the_old_object() {
    let (root, usage) = store_usage("rotate");
    let old = SealKey::from_bytes([0x41; 32]);
    let new_key = SealKey::from_bytes([0x42; 32]);
    let mut file = blank();
    old.seal_file(&usage, &mut file).unwrap();
    assert_eq!(testing_read_usage_count(&usage, &old).unwrap(), 1);
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x41; 32])])
    );
    let sealed = file.clone();
    let obj = root
        .0
        .join("objects/ab/ab00000000000000000000000000000000000000000000000000000000000000.kvc");
    std::fs::create_dir_all(obj.parent().unwrap()).unwrap();
    std::fs::write(&obj, &sealed).unwrap();

    let mut next = blank();
    new_key.seal_file(&usage, &mut next).unwrap();
    assert!(seal::is_sealed_region(&next[..HEADER_REGION_LEN]));
    assert_eq!(testing_read_usage_count(&usage, &new_key).unwrap(), 1);
    assert_eq!(testing_read_usage_count(&usage, &old).unwrap(), 1);
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x41; 32]), fp_bytes(&[0x42; 32])])
    );
    assert_eq!(
        std::fs::read(&obj).unwrap(),
        sealed,
        "an old-key object stays until a read drops it"
    );
}

#[test]
fn seal_budget_same_key_with_the_counter_deleted_fails() {
    let (_root, usage) = store_usage("deleted-counter");
    let key = SealKey::from_bytes([0x43; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    let path = testing_usage_path(&usage, &key);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x43; 32])])
    );
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert!(!path.exists(), "the refusal does not recreate the counter");
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x43; 32])])
    );
}

#[test]
fn seal_budget_legacy_store_refuses_then_rotation_seals() {
    let (root, usage) = store_usage("legacy");
    let name = "ab00000000000000000000000000000000000000000000000000000000000000.kvc";
    plant_kvc(&root.0, "ab", name, true);
    let obj = root.0.join("objects/ab").join(name);
    let old_bytes = std::fs::read(&obj).unwrap();
    assert_eq!(testing_read_key_fingerprints(&usage).unwrap(), None);

    let old = SealKey::from_bytes([0x44; 32]);
    let mut file = blank();
    let before = file.clone();
    assert_eq!(old.seal_file(&usage, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert!(!testing_usage_path(&usage, &old).exists());
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x44; 32])]),
        "the refusal records the key and does not write a counter"
    );
    assert_eq!(old.seal_file(&usage, &mut file), Err(SealError::UsageStore));
    assert!(!testing_usage_path(&usage, &old).exists());

    let new_key = SealKey::from_bytes([0x45; 32]);
    let mut next = blank();
    new_key.seal_file(&usage, &mut next).unwrap();
    assert!(seal::is_sealed_region(&next[..HEADER_REGION_LEN]));
    assert_eq!(testing_read_usage_count(&usage, &new_key).unwrap(), 1);
    assert!(!testing_usage_path(&usage, &old).exists());
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x44; 32]), fp_bytes(&[0x45; 32])])
    );
    assert_eq!(std::fs::read(&obj).unwrap(), old_bytes);
}

#[test]
#[ignore = "needs passwordless sudo to chown a leftover part file to nobody"]
fn seal_budget_part_owned_by_another_user_is_left_in_place() {
    assert!(passwordless_sudo(), "passwordless sudo is not available");
    let (_root, usage) = store_usage("part-other-uid");
    let key = SealKey::from_bytes([0x46; 32]);
    testing_install_usage_count(&usage, &key, 4).unwrap();
    let count_path = testing_usage_path(&usage, &key);
    let part = count_path.with_extension("part");
    std::fs::write(&part, b"other-user").unwrap();
    let chown = std::process::Command::new("sudo")
        .args(["-n", "chown", "nobody:nogroup"])
        .arg(&part)
        .status()
        .expect("sudo chown");
    assert!(chown.success(), "chown nobody of the leftover part");
    let mut file = blank();
    let before = file.clone();
    assert_eq!(key.seal_file(&usage, &mut file), Err(SealError::UsageStore));
    assert_eq!(file, before);
    assert_eq!(std::fs::read(&part).unwrap(), b"other-user");
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 4);
}

fn passwordless_sudo() -> bool {
    match std::process::Command::new("sudo")
        .args(["-n", "true"])
        .status()
    {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}

#[test]
fn seal_budget_returning_to_a_listed_key_with_a_deleted_counter_refuses() {
    let (_root, usage) = store_usage("return");
    let a = SealKey::from_bytes([0x61; 32]);
    let b = SealKey::from_bytes([0x62; 32]);
    let mut file = blank();
    a.seal_file(&usage, &mut file).unwrap();
    b.seal_file(&usage, &mut file).unwrap();
    assert_eq!(testing_read_usage_count(&usage, &a).unwrap(), 1);
    assert_eq!(testing_read_usage_count(&usage, &b).unwrap(), 1);
    let a_path = testing_usage_path(&usage, &a);
    std::fs::remove_file(&a_path).unwrap();
    let mut again = blank();
    let before = again.clone();
    assert_eq!(a.seal_file(&usage, &mut again), Err(SealError::UsageStore));
    assert_eq!(again, before);
    assert!(!a_path.exists(), "a retired key does not start over");
    assert_eq!(testing_read_usage_count(&usage, &b).unwrap(), 1);
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x61; 32]), fp_bytes(&[0x62; 32])])
    );
}

#[test]
fn seal_budget_corrupt_fingerprint_list_does_not_consume_a_slot() {
    let (_root, usage) = store_usage("bad-list");
    let key = SealKey::from_bytes([0x63; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
    std::fs::write(usage.join(seal::KEY_FINGERPRINTS_NAME), b"not-a-list").unwrap();
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert_eq!(
        testing_read_usage_count(&usage, &key).unwrap(),
        1,
        "a corrupt list is refused before the count is incremented"
    );
}

#[test]
fn seal_budget_leftover_list_part_is_replaced_when_owned_and_refused_when_linked() {
    let (root, usage) = store_usage("list-part");
    let a = SealKey::from_bytes([0x64; 32]);
    let b = SealKey::from_bytes([0x65; 32]);
    let mut file = blank();
    a.seal_file(&usage, &mut file).unwrap();
    let part = usage.join(format!("{}.part", seal::KEY_FINGERPRINTS_NAME));
    std::fs::write(&part, b"stale-list-part").unwrap();
    b.seal_file(&usage, &mut file).unwrap();
    assert!(
        !part.exists(),
        "an owned leftover list part is renamed away"
    );
    assert_eq!(testing_read_usage_count(&usage, &b).unwrap(), 1);
    let stored = std::fs::read(usage.join(seal::KEY_FINGERPRINTS_NAME)).unwrap();
    assert!(!stored.windows(15).any(|w| w == b"stale-list-part"));
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x64; 32]), fp_bytes(&[0x65; 32])])
    );
    drop(root);

    let (_root, usage) = store_usage("list-part-link");
    let a = SealKey::from_bytes([0x66; 32]);
    let b = SealKey::from_bytes([0x67; 32]);
    let mut file = blank();
    a.seal_file(&usage, &mut file).unwrap();
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let part = usage.join(format!("{}.part", seal::KEY_FINGERPRINTS_NAME));
    std::os::unix::fs::symlink(&list, &part).unwrap();
    let mut next = blank();
    let before = next.clone();
    assert_eq!(b.seal_file(&usage, &mut next), Err(SealError::UsageStore));
    assert_eq!(next, before);
    assert!(
        std::fs::symlink_metadata(&part)
            .unwrap()
            .file_type()
            .is_symlink(),
        "a linked list part is left in place"
    );
    assert!(!testing_usage_path(&usage, &b).exists());
    assert_eq!(testing_read_usage_count(&usage, &a).unwrap(), 1);
}

#[test]
fn seal_budget_deleted_list_keeps_remaining_counters_and_refuses_the_deleted_one() {
    let (_root, usage) = store_usage("deleted-list");
    let a = SealKey::from_bytes([0x71; 32]);
    let b = SealKey::from_bytes([0x72; 32]);
    let mut file = blank();
    a.seal_file(&usage, &mut file).unwrap();
    b.seal_file(&usage, &mut file).unwrap();
    std::fs::remove_file(usage.join(seal::KEY_FINGERPRINTS_NAME)).unwrap();
    let a_path = testing_usage_path(&usage, &a);
    std::fs::remove_file(&a_path).unwrap();

    let mut again = blank();
    let before = again.clone();
    assert_eq!(a.seal_file(&usage, &mut again), Err(SealError::UsageStore));
    assert_eq!(again, before);
    assert!(
        !a_path.exists(),
        "A does not start over after the list is deleted"
    );
    assert_eq!(testing_read_usage_count(&usage, &b).unwrap(), 1);
    b.seal_file(&usage, &mut again).unwrap();
    assert_eq!(testing_read_usage_count(&usage, &b).unwrap(), 2);
    assert!(!a_path.exists());
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x72; 32]), fp_bytes(&[0x71; 32])]),
        "the rebuilt list keeps B's counter file and records A"
    );
}

#[test]
fn seal_budget_truncated_list_does_not_consume_a_slot() {
    let (_root, usage) = store_usage("trunc-list");
    let key = SealKey::from_bytes([0x73; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let mut bytes = std::fs::read(&list).unwrap();
    bytes.truncate(20);
    std::fs::write(&list, &bytes).unwrap();
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
    assert_eq!(std::fs::read(&list).unwrap(), bytes);
}

#[test]
fn seal_budget_duplicate_list_entries_do_not_consume_a_slot() {
    let (_root, usage) = store_usage("dup-list");
    let key = SealKey::from_bytes([0x74; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    let fp = fp_bytes(&[0x74; 32]);
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let bytes = list_bytes(&[fp, fp]);
    std::fs::write(&list, &bytes).unwrap();
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
    assert_eq!(std::fs::read(&list).unwrap(), bytes);
}

#[test]
fn seal_budget_oversized_list_does_not_consume_a_slot() {
    let (_root, usage) = store_usage("oversize-list");
    let key = SealKey::from_bytes([0x75; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let fps: Vec<[u8; 32]> = (0..257u16).map(filler_fp).collect();
    let bytes = list_bytes(&fps);
    std::fs::write(&list, &bytes).unwrap();
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
    assert_eq!(std::fs::read(&list).unwrap(), bytes);
}

#[test]
fn seal_budget_fingerprint_list_stops_at_256_keys() {
    let (_root, usage) = store_usage("cap");
    let key = SealKey::from_bytes([0x76; 32]);
    let extra = SealKey::from_bytes([0x77; 32]);
    let key_fp = fp_bytes(&[0x76; 32]);
    let extra_fp = fp_bytes(&[0x77; 32]);
    let mut fps = Vec::with_capacity(256);
    for i in 0..255u16 {
        let fp = filler_fp(i);
        assert_ne!(fp, key_fp);
        assert_ne!(fp, extra_fp);
        fps.push(fp);
    }
    fps.push(key_fp);
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let full = list_bytes(&fps);
    std::fs::write(&list, &full).unwrap();
    testing_install_usage_count(&usage, &key, 3).unwrap();
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 4);
    assert_eq!(std::fs::read(&list).unwrap(), full);

    let mut next = blank();
    let before = next.clone();
    assert_eq!(
        extra.seal_file(&usage, &mut next),
        Err(SealError::UsageStore)
    );
    assert_eq!(next, before);
    assert!(
        !testing_usage_path(&usage, &extra).exists(),
        "the 257th key does not get a counter"
    );
    assert_eq!(std::fs::read(&list).unwrap(), full);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 4);
}

#[test]
fn seal_budget_legacy_key_fingerprint_file_is_ignored() {
    let (_root, usage) = store_usage("old-marker");
    let marker = usage.join("key-fingerprint");
    std::fs::write(&marker, b"legacy-marker").unwrap();
    let key = SealKey::from_bytes([0x78; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    assert!(seal::is_sealed_region(&file[..HEADER_REGION_LEN]));
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);
    assert_eq!(std::fs::read(&marker).unwrap(), b"legacy-marker");
    assert_eq!(
        testing_read_key_fingerprints(&usage).unwrap(),
        Some(vec![fp_bytes(&[0x78; 32])])
    );
}

#[test]
#[ignore = "needs passwordless sudo to chown seal-usage/key-fingerprints to nobody"]
fn seal_budget_list_owned_by_another_user_is_left_in_place() {
    assert!(passwordless_sudo(), "passwordless sudo is not available");
    let (_root, usage) = store_usage("list-other-uid");
    let key = SealKey::from_bytes([0x79; 32]);
    let mut file = blank();
    key.seal_file(&usage, &mut file).unwrap();
    let list = usage.join(seal::KEY_FINGERPRINTS_NAME);
    let listed = std::fs::read(&list).unwrap();
    let owner = std::fs::metadata(&list).unwrap().uid();
    chown_nobody(&list);
    assert_ne!(std::fs::metadata(&list).unwrap().uid(), owner);
    let mut again = blank();
    let before = again.clone();
    assert_eq!(
        key.seal_file(&usage, &mut again),
        Err(SealError::UsageStore)
    );
    assert_eq!(again, before);
    assert_eq!(sudo_read(&list), listed);
    assert_eq!(testing_read_usage_count(&usage, &key).unwrap(), 1);

    let (_root, usage) = store_usage("list-part-other-uid");
    let a = SealKey::from_bytes([0x7a; 32]);
    let b = SealKey::from_bytes([0x7b; 32]);
    let mut file = blank();
    a.seal_file(&usage, &mut file).unwrap();
    let part = usage.join(format!("{}.part", seal::KEY_FINGERPRINTS_NAME));
    std::fs::write(&part, b"foreign-list-part").unwrap();
    chown_nobody(&part);
    let mut next = blank();
    let before = next.clone();
    assert_eq!(b.seal_file(&usage, &mut next), Err(SealError::UsageStore));
    assert_eq!(next, before);
    assert_eq!(std::fs::read(&part).unwrap(), b"foreign-list-part");
    assert!(!testing_usage_path(&usage, &b).exists());
    assert_eq!(testing_read_usage_count(&usage, &a).unwrap(), 1);
}

fn chown_nobody(path: &std::path::Path) {
    let chown = std::process::Command::new("sudo")
        .args(["-n", "chown", "nobody:nogroup"])
        .arg(path)
        .status()
        .expect("sudo chown");
    assert!(chown.success(), "chown nobody");
}

fn sudo_read(path: &std::path::Path) -> Vec<u8> {
    let out = std::process::Command::new("sudo")
        .args(["-n", "cat"])
        .arg(path)
        .output()
        .expect("sudo cat");
    assert!(out.status.success(), "sudo cat");
    out.stdout
}

fn list_bytes(fps: &[[u8; 32]]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(12 + fps.len() * 32);
    buf.extend_from_slice(b"AKSL");
    buf.push(1);
    buf.extend_from_slice(&[0, 0, 0]);
    buf.extend_from_slice(&(fps.len() as u32).to_le_bytes());
    for fp in fps {
        buf.extend_from_slice(fp);
    }
    buf
}

fn filler_fp(i: u16) -> [u8; 32] {
    let mut fp = [0xA5u8; 32];
    fp[0] = 0xF0;
    fp[30] = (i >> 8) as u8;
    fp[31] = (i & 0xff) as u8;
    fp
}

fn refused_store(
    s: &HostFsStore,
    fields: &NamespaceFields<'_>,
    n: u32,
) -> (crate::ChunkKey, Vec<u8>) {
    let ns = fields.namespace_id().unwrap();
    let toks: Vec<u32> = (0..4).map(|i| n * 100 + i).collect();
    let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let header = fields.chunk_header(&ns, key, None, 0, 4, 1).unwrap();
    let payload = vec![0xABu8; BS];
    let mut buf = payload.clone().into_boxed_slice();
    // SAFETY: `buf` outlives `run_until_idle` and `poll` inside `terminal_for`.
    let src =
        unsafe { PinnedSource::new(buf.as_mut_ptr(), BS, vec![0], 4, HolderGuard::detached()) };
    let job = s
        .submit_store(header, src, Incarnation::seq(n as u64))
        .unwrap();
    s.run_until_idle();
    assert_eq!(
        terminal_for(s, job),
        Terminal::Failed(PlatformError::HardwareFault("seal budget exhausted".into()))
    );
    drop(buf);
    (key, payload)
}
