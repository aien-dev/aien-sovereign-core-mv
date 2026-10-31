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

//! Recovery, purge and admission hardening of the host filesystem adapter
//! (Grok C4 review P1-3, P2-4, P2-5, P2-6; ADR 0003 section 4.1). I/O errors
//! come from directories made unreadable or read-only with chmod in a temp
//! dir (the suite runs as a normal user; `chmod_works` checks that).

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::HostFsStore;
use aien_kv_store::{ChunkKey, KvStore, NamespaceId, RecoveryProgress, StoreState};
use aien_platform::{ObjectId, PlatformError};
use rig::*;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Sets a directory's mode and puts the saved original back on drop (so the
/// temp dir can be removed even when an assertion fails, and the store still
/// sees a private directory afterwards).
struct Mode(PathBuf, std::fs::Permissions);

impl Mode {
    fn set(path: &Path, mode: u32) -> Mode {
        let saved = std::fs::metadata(path).unwrap().permissions();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        Mode(path.to_path_buf(), saved)
    }
}

impl Drop for Mode {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.0, self.1.clone());
    }
}

/// True when chmod actually denies access (it does not for root).
fn chmod_works(dir: &Path) -> bool {
    let probe = dir.join("probe");
    std::fs::create_dir(&probe).unwrap();
    let denied = {
        let _m = Mode::set(&probe, 0o000);
        std::fs::read_dir(&probe).is_err()
    };
    std::fs::remove_dir(&probe).unwrap();
    denied
}

/// Step recovery until it is done or the store failed; never loops forever.
fn settle(s: &HostFsStore) -> RecoveryProgress {
    let mut last = RecoveryProgress::default();
    for _ in 0..10_000 {
        last = s.recover_step(50);
        if last.done || s.capacity().state == StoreState::Failed {
            return last;
        }
    }
    last
}

fn plant_ours(fx: &Fx, n: u32) -> (ChunkKey, PathBuf) {
    let (key, header) = fx.chunk(n, 8);
    fx.plant(&key, &header, &pattern(n, 2 * BS), BS);
    (key, fx.object_file(&key))
}

fn shard_of(path: &Path) -> PathBuf {
    path.parent().unwrap().to_path_buf()
}

#[test]
fn an_unreadable_shard_directory_fails_recovery_and_admission_never_opens() {
    let fx = Fx::new();
    fx.prime();
    if !chmod_works(fx.root()) {
        eprintln!("skipped: chmod does not deny access here");
        return;
    }
    let (_, path) = plant_ours(&fx, 1);
    let _m = Mode::set(&shard_of(&path), 0o000);
    let s = fx.open_raw(|_| {});
    let p = settle(&s);
    assert_eq!(
        s.capacity().state,
        StoreState::Failed,
        "an EACCES listing is not an empty directory"
    );
    assert!(!p.done, "a failed walk is never reported done");
    assert!(matches!(
        fx.submit(&s, 2, 8).0,
        Err(PlatformError::DeviceUnavailable)
    ));
}

#[test]
fn an_unreadable_tmp_directory_fails_recovery() {
    let fx = Fx::new();
    fx.prime();
    if !chmod_works(fx.root()) {
        return;
    }
    let _m = Mode::set(&fx.root().join("tmp"), 0o000);
    let s = fx.open_raw(|_| {});
    let p = settle(&s);
    assert_eq!(s.capacity().state, StoreState::Failed);
    assert!(!p.done);
}

#[test]
fn recover_step_on_a_failed_store_is_not_done() {
    let fx = Fx::new();
    fx.prime();
    std::fs::write(fx.root().join("STORE_ID"), b"short").unwrap();
    let s = fx.open_raw(|_| {});
    assert_eq!(s.capacity().state, StoreState::Failed);
    let p = s.recover_step(50);
    assert!(!p.done, "Failed is not a finished recovery: {p:?}");
}

#[test]
fn a_corrupt_file_that_cannot_be_unlinked_stays_counted_and_degrades_the_store() {
    let fx = Fx::new();
    fx.prime();
    if !chmod_works(fx.root()) {
        return;
    }
    let (key, _) = fx.chunk(1, 8);
    let path = fx.object_file(&key);
    std::fs::write(&path, vec![0xAB; 5000]).unwrap();
    let phys = phys_of(&path);
    let _m = Mode::set(&shard_of(&path), 0o500);
    let s = fx.open_raw(|c| c.purge_after_recovery = false);
    let p = settle(&s);
    assert!(p.done);
    assert!(path.is_file(), "the unlink really failed");
    let cap = s.capacity();
    assert_eq!(cap.used_bytes, phys, "bytes still on disk stay counted");
    assert_eq!(
        cap.state,
        StoreState::Degraded,
        "a failed recovery unlink degrades"
    );
}

#[test]
fn a_failed_purge_unlink_keeps_the_unindexed_accounting() {
    let fx = Fx::new();
    fx.prime();
    if !chmod_works(fx.root()) {
        return;
    }
    let planted: Vec<_> = (1..=3).map(|n| plant_ours(&fx, n)).collect();
    let s = fx.open_with(|c| {
        c.max_objects = 2;
        c.purge_after_recovery = false;
    });
    assert_eq!(s.capacity().unindexed_objects, 1);
    let modes: Vec<Mode> = planted
        .iter()
        .map(|(_, p)| Mode::set(&shard_of(p), 0o500))
        .collect();
    s.start_purge();
    s.run_until_idle();
    let cap = s.capacity();
    assert_eq!(cap.unindexed_objects, 1, "the file is still there");
    assert_eq!(cap.unindexed_bytes, OBJ);
    assert_eq!(cap.used_bytes, 3 * OBJ);
    assert_eq!(s.stats().purge_unlinked, 0);
    drop(modes);
    // With the directories writable again a new purge finishes the job.
    s.start_purge();
    s.run_until_idle();
    let cap = s.capacity();
    assert_eq!((cap.unindexed_objects, cap.unindexed_bytes), (0, 0));
    assert_eq!(cap.used_bytes, 2 * OBJ);
    assert_eq!(object_files(fx.root()).len(), 2);
}

#[test]
fn stores_are_admitted_into_free_headroom_while_unindexed_files_wait() {
    let fx = Fx::new();
    fx.prime();
    let planted: Vec<_> = (1..=3).map(|n| plant_ours(&fx, n)).collect();
    let s = fx.open_with(|c| {
        c.max_objects = 2;
        c.purge_after_recovery = false;
    });
    assert_eq!(s.capacity().unindexed_objects, 1);
    let indexed: Vec<ChunkKey> = planted
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| s.lookup(&[*k])[0].is_some())
        .collect();
    assert_eq!(s.delete(&indexed[..1]), 1);
    s.run_until_idle();
    let b = fx.put_full(&s, 9);
    assert!(
        b.job.is_ok(),
        "one free index slot and free bytes admit a store: {:?}",
        b.job
    );
    assert_eq!(s.lookup(&[b.key])[0], Some(FULL_PAYLOAD));
    assert_eq!(
        s.lookup(&[indexed[1]])[0],
        Some(FULL_PAYLOAD),
        "no eviction"
    );
}

#[test]
fn while_a_purge_is_pending_admission_uses_free_headroom_only() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=3 {
        plant_ours(&fx, n);
    }
    let s = fx.open_with(|c| {
        c.max_objects = 2;
        c.purge_after_recovery = false;
    });
    s.start_purge();
    // Nothing free: a store would need to evict a good object for bytes the
    // purge is about to free anyway.
    assert_eq!(fx.submit(&s, 9, 8).0, Err(PlatformError::OutOfMemory));
    s.run_until_idle();
    assert!(!s.purge_pending());
    assert_eq!(s.capacity().objects, 2, "nothing was evicted");
}

#[test]
fn a_foreign_object_with_dirty_padding_is_corrupt() {
    let fx = Fx::new();
    fx.prime();
    let (_, mut header) = fx.chunk(3, 8);
    let key = ChunkKey(ObjectId([0xA3; 32]));
    header.namespace_id = NamespaceId([0x77; 32]);
    header.chunk_key = key;
    header.payload_len = 100;
    fx.plant(&key, &header, &pattern(3, 100), 100);
    let path = fx.object_file(&key);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[4000] = 1;
    std::fs::write(&path, bytes).unwrap();
    let s = fx.open_raw(|_| {});
    let p = settle(&s);
    assert_eq!(p.foreign_namespace, 0);
    assert_eq!(p.corrupt_deleted, 1);
    assert!(!path.exists());
    assert_eq!(s.capacity().used_bytes, 0);
}

#[test]
fn a_clean_foreign_object_with_several_segments_is_kept() {
    let fx = Fx::new();
    fx.prime();
    let (_, mut header) = fx.chunk(4, 8);
    let key = ChunkKey(ObjectId([0xA4; 32]));
    header.namespace_id = NamespaceId([0x77; 32]);
    header.chunk_key = key;
    header.payload_len = 3 * 1000;
    fx.plant(&key, &header, &pattern(4, 3000), 1000);
    let s = fx.open_raw(|_| {});
    let p = settle(&s);
    assert_eq!(p.foreign_namespace, 1);
    assert_eq!(p.corrupt_deleted, 0);
}

#[test]
fn a_store_of_a_key_being_evicted_waits_until_the_unlink_returns() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = OBJ);
    let a = fx.put_full(&s, 1);
    let (b, _h) = fx.submit(&s, 2, 8);
    b.unwrap();
    assert!(s.pump(), "the store of 2 runs");
    // 1 is now the victim of a store that published; its unlink may still be
    // queued. A new store of 1 must not be published only to be deleted.
    let r = fx.submit(&s, 1, 8).0;
    if s.lookup(&[a.key])[0].is_none() {
        assert_eq!(r, Err(PlatformError::QueueFull));
    }
    s.run_until_idle();
    let _ = drain(&s);
    let again = fx.put_full(&s, 1);
    assert!(again.job.is_ok());
    assert_eq!(s.lookup(&[a.key])[0], Some(FULL_PAYLOAD));
    assert!(fx.object_file(&a.key).is_file());
    s.audit().unwrap();
}
