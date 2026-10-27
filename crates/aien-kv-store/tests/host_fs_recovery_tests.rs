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

// Recovery follows ADR 0003 section 4.1 (review finding 8); the temp-file and
// foreign-file rules come from the same section. LMCache fs_l2_adapter.py
// (4d69cdc4, Apache-2.0, Copyright LMCache contributors) informed only the
// idea of lazy disk recovery; no code or test text was copied. See
// THIRD_PARTY.md and ADR 0003 context item 24.

//! Resumable recovery and accounting of the host filesystem adapter (cut C4,
//! acceptance A3, A6, A7), against real temporary directories.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::HostFsStore;
use aien_kv_store::{
    ChunkHeader, ChunkKey, Incarnation, JobStage, KvStore, NamespaceId, RecordHeader, StoreState,
    Terminal,
};
use aien_platform::{ObjectId, PlatformError};
use rig::*;
use std::path::PathBuf;

/// Plant a valid object of ours for stream `n` (full chunk, 2 blocks).
fn plant_ours(fx: &Fx, n: u32) -> (ChunkKey, PathBuf) {
    let (key, header) = fx.chunk(n, 8);
    fx.plant(&key, &header, &pattern(n, 2 * BS), BS);
    (key, fx.object_file(&key))
}

/// A valid object of another namespace with a `payload_len`-byte payload.
fn plant_foreign(fx: &Fx, n: u32, payload_len: usize) -> (ChunkKey, PathBuf) {
    let (_, mut header) = fx.chunk(n, 8);
    let key = ChunkKey(ObjectId([0xA0 + n as u8; 32]));
    header.namespace_id = NamespaceId([0x77; 32]);
    header.chunk_key = key;
    header.payload_len = payload_len as u64;
    fx.plant(&key, &header, &pattern(n, payload_len), payload_len);
    (key, fx.object_file(&key))
}

fn sum_phys(paths: &[PathBuf]) -> u64 {
    paths.iter().map(|p| phys_of(p)).sum()
}

#[test]
fn empty_store_recovers_in_one_step() {
    let fx = Fx::new();
    let s = fx.open_raw(|_| {});
    assert_eq!(s.capacity().state, StoreState::Recovering);
    let p = s.recover_step(50);
    assert!(p.done);
    assert_eq!(p.indexed, 0);
    assert_eq!(s.capacity().state, StoreState::Open);
    assert_eq!(s.capacity().used_bytes, 0);
    // Further calls change nothing.
    let q = s.recover_step(50);
    assert!(q.done);
    assert_eq!((q.indexed, q.bytes, q.tmp_deleted), (0, 0, 0));
    assert!(!s.purge_pending());
}

#[test]
fn recovery_counts_foreign_namespace_objects_toward_quota() {
    let fx = Fx::new();
    fx.prime();
    let (_, p1) = plant_ours(&fx, 1);
    let (_, p2) = plant_ours(&fx, 2);
    let foreign: Vec<_> = (3..6).map(|n| plant_foreign(&fx, n, 100)).collect();
    let fpaths: Vec<_> = foreign.iter().map(|(_, p)| p.clone()).collect();
    let foreign_phys = sum_phys(&fpaths);
    let total = sum_phys(&[p1.clone(), p2.clone()]) + foreign_phys;
    let s = fx.open_raw(|c| c.quota_bytes = total);
    let p = recover_all(&s);
    assert_eq!(p.indexed, 2);
    assert_eq!(p.foreign_namespace, 3);
    assert_eq!(p.bytes, total);
    let cap = s.capacity();
    assert_eq!(
        cap.used_bytes, total,
        "foreign objects count whether or not the namespace matches"
    );
    assert_eq!(cap.objects, 5, "persisted objects include the foreign ones");
    assert_eq!(cap.state, StoreState::Open);
    // They are never served.
    let fkeys: Vec<_> = foreign.iter().map(|(k, _)| *k).collect();
    assert_eq!(s.lookup(&fkeys), vec![None, None, None]);
    let (_, h) = fx.chunk(3, 8);
    let (job, _) = fx.submit_load(&s, fkeys[0], &h);
    assert_eq!(job.unwrap_err(), PlatformError::ObjectNotFound(fkeys[0].0));

    // A full store admits a new object by evicting the foreign ones first, not our oldest.
    let put = fx.put_full(&s, 9);
    assert_eq!(
        terminal_of(&put.completions, put.job.unwrap()),
        Terminal::Done(JobStage::Durable)
    );
    for p in &fpaths {
        assert!(!p.exists(), "foreign object evicted first");
    }
    assert!(p1.is_file() && p2.is_file(), "our objects stay");
    assert_eq!(s.stats().evicted_bytes_total, foreign_phys);
    s.audit().unwrap();
}

#[test]
fn recovery_deletes_tmp_and_corrupt_headers_only() {
    use std::os::unix::fs::FileExt;
    let fx = Fx::new();
    fx.prime();
    let root = fx.root().to_path_buf();
    let good: Vec<_> = (1..=3).map(|n| plant_ours(&fx, n).1).collect();

    // A leftover temp from a crashed write, and a non-.part file in tmp/.
    std::fs::write(
        root.join("tmp").join("deadbeef.123.4.part"),
        vec![7u8; 5000],
    )
    .unwrap();
    std::fs::write(root.join("tmp").join("notes.txt"), b"mine").unwrap();

    // Five corrupt objects under valid names.
    let mut corrupt = Vec::new();
    for n in 10..15u32 {
        let (_, path) = plant_ours(&fx, n);
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        match n {
            10 => f.write_all_at(b"XXXX", 0).unwrap(), // magic
            11 => {
                let mut b = [0u8; 1];
                f.read_exact_at(&mut b, 150).unwrap();
                f.write_all_at(&[b[0] ^ 1], 150).unwrap(); // header checksum
            }
            12 => f.set_len(100).unwrap(), // shorter than the header region
            13 => {
                let len = f.metadata().unwrap().len();
                f.write_all_at(&[1], len).unwrap(); // one byte too long
            }
            _ => f.write_all_at(&[1], 4000).unwrap(), // nonzero padding
        }
        drop(f);
        corrupt.push(path);
    }
    let corrupt_bytes = sum_phys(&corrupt);

    // Files we did not write: left alone and not counted.
    let strays = [
        root.join("objects").join("00").join("README.txt"),
        root.join("objects")
            .join("00")
            .join(format!("{}.kvc.bak", "0".repeat(64))),
        root.join("objects")
            .join("00")
            .join(format!("{}.kvc", "A".repeat(64))), // uppercase hex
        // a valid-looking name in the wrong shard
        root.join("objects")
            .join("ff")
            .join(format!("{}.kvc", "00".repeat(32))),
    ];
    for p in &strays {
        std::fs::write(p, b"not ours").unwrap();
    }

    let s = fx.open_raw(|_| {});
    let p = recover_all(&s);
    assert_eq!(p.tmp_deleted, 1);
    assert_eq!(p.corrupt_deleted, 5);
    assert_eq!(p.indexed, 3);
    assert_eq!(p.foreign_namespace, 0);
    for c in &corrupt {
        assert!(!c.exists(), "{c:?}");
    }
    assert!(!root.join("tmp").join("deadbeef.123.4.part").exists());
    assert!(root.join("tmp").join("notes.txt").exists());
    for g in &good {
        assert!(g.is_file());
    }
    for s in &strays {
        assert!(s.is_file(), "{s:?} must be left alone");
    }
    let st = s.stats();
    assert_eq!(
        st.foreign_files, 5,
        "four stray names under objects/ plus tmp/notes.txt"
    );
    assert_eq!(st.tmp_bytes_deleted, 5000);
    assert_eq!(st.corrupt_bytes_deleted, corrupt_bytes);
    // Every byte is accounted: only the three good objects are used.
    assert_eq!(s.capacity().used_bytes, 3 * OBJ);
    assert_eq!(s.capacity().unindexed_bytes, 0);
}

#[test]
fn recovery_applies_the_identity_rules_to_objects_that_claim_our_namespace() {
    let fx = Fx::new();
    fx.prime();
    let (good_key, _) = plant_ours(&fx, 1);
    #[allow(clippy::type_complexity)]
    let bad: Vec<(&str, Box<dyn Fn(&mut ChunkHeader)>, usize)> = vec![
        ("model", Box::new(|h| h.model_sha256[0] ^= 1), 2 * BS),
        ("layers", Box::new(|h| h.num_layers += 1), 2 * BS),
        ("block size", Box::new(|h| h.block_size = 8), 2 * BS),
        ("position", Box::new(|h| h.first_position = 3), 2 * BS),
        ("payload length", Box::new(|_| {}), 100),
    ];
    for (i, (why, mutate, plen)) in bad.iter().enumerate() {
        let n = 20 + i as u32;
        let (key, mut h) = fx.chunk(n, 8);
        mutate(&mut h);
        h.payload_len = *plen as u64;
        fx.plant(&key, &h, &pattern(n, *plen), (*plen).min(BS));
        assert!(fx.object_file(&key).is_file(), "{why}");
    }
    // The file name must equal the key in the header.
    let (name_key, mut h) = fx.chunk(30, 8);
    let (other_key, _) = fx.chunk(31, 8);
    h.chunk_key = other_key;
    fx.plant(&name_key, &h, &pattern(30, 2 * BS), BS);
    // A different namespace is kept, counted, never served.
    let (fkey, _) = plant_foreign(&fx, 4, 100);

    let s = fx.open_raw(|_| {});
    let p = recover_all(&s);
    assert_eq!(p.corrupt_deleted, 6);
    assert_eq!(p.foreign_namespace, 1);
    assert_eq!(p.indexed, 1);
    assert_eq!(s.lookup(&[good_key, fkey]), vec![Some(FULL_PAYLOAD), None]);
    assert_eq!(object_files(fx.root()).len(), 2);
}

#[test]
fn recovery_is_bounded_by_max_objects_and_sets_over_cap() {
    let fx = Fx::new();
    fx.prime();
    let planted: Vec<_> = (1..=10).map(|n| plant_ours(&fx, n)).collect();
    let keys: Vec<ChunkKey> = planted.iter().map(|(k, _)| *k).collect();
    let s = fx.open_raw(|c| {
        c.max_objects = 4;
        c.purge_after_recovery = false;
    });
    let before = s.stats();
    let p = recover_all(&s);
    assert_eq!(p.indexed, 4);
    assert_eq!(p.unindexed_objects, 6);
    assert_eq!(p.unindexed_bytes, 6 * OBJ);
    let cap = s.capacity();
    assert_eq!(cap.state, StoreState::Open);
    assert_eq!(cap.objects, 4, "the index never exceeds max_objects");
    assert_eq!(cap.max_objects, 4);
    assert_eq!(cap.unindexed_objects, 6);
    assert_eq!(cap.unindexed_bytes, 6 * OBJ);
    assert_eq!(
        cap.used_bytes,
        10 * OBJ,
        "used_bytes always includes the unindexed bytes"
    );
    assert!(s.stats().over_cap);
    let hits: Vec<_> = s.lookup(&keys).into_iter().flatten().collect();
    assert_eq!(hits.len(), 4);
    assert_eq!(
        s.stats().reads - before.reads,
        4,
        "no header read past the cap"
    );
    s.audit().unwrap();

    // The purge pass (on the worker) unlinks the unindexed files and nothing else.
    let indexed_keys: Vec<_> = keys
        .iter()
        .zip(s.lookup(&keys))
        .filter(|(_, l)| l.is_some())
        .map(|(k, _)| *k)
        .collect();
    assert!(!s.purge_pending());
    s.start_purge();
    assert!(s.purge_pending());
    s.run_until_idle();
    assert!(!s.purge_pending());
    let cap = s.capacity();
    assert_eq!((cap.unindexed_objects, cap.unindexed_bytes), (0, 0));
    assert_eq!(cap.used_bytes, 4 * OBJ);
    assert_eq!(object_files(fx.root()).len(), 4);
    for k in indexed_keys {
        assert!(fx.object_file(&k).is_file());
    }
    assert!(!s.stats().over_cap);
    assert_eq!(s.stats().purge_unlinked, 6);
    s.audit().unwrap();
}

#[test]
fn recovery_reads_exactly_4096_bytes_per_object() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=5 {
        plant_ours(&fx, n);
    }
    plant_foreign(&fx, 6, 100);
    let s = fx.open_raw(|_| {});
    let before = s.stats();
    recover_all(&s);
    let after = s.stats();
    assert_eq!(
        after.reads - before.reads,
        6,
        "one read per object, foreign included"
    );
    assert_eq!(
        after.read_bytes - before.read_bytes,
        6 * 4096,
        "the header region, aligned, never the payload"
    );
}

#[test]
fn recovery_keeps_admission_closed_until_walk_completes() {
    let fx = Fx::new();
    fx.prime();
    let keys: Vec<ChunkKey> = (1..=6).map(|n| plant_ours(&fx, n).0).collect();
    std::fs::write(fx.root().join("tmp").join("x.1.1.part"), b"junk").unwrap();
    let s = fx.open_raw(|_| {});
    assert_eq!(s.capacity().state, StoreState::Recovering);

    let mut steps = 0;
    let mut last_bytes = 0;
    let done = loop {
        steps += 1;
        assert!(steps < 1000);
        let p = s.recover_step(0); // at least one entry per step, no more than one slice allows
        if p.done {
            break p;
        }
        // Mid-walk: every key misses, every admission is refused, whatever is already indexed.
        let cap = s.capacity();
        assert_eq!(cap.state, StoreState::Recovering, "step {steps}");
        assert!(s.lookup(&keys).iter().all(Option::is_none), "step {steps}");
        let (job, _) = fx.submit(&s, 50, 8);
        assert_eq!(job.unwrap_err(), PlatformError::QueueFull, "step {steps}");
        let (l, _) = fx.submit_load(&s, keys[0], &fx.chunk(1, 8).1);
        assert_eq!(l.unwrap_err(), PlatformError::QueueFull, "step {steps}");
        assert_eq!(s.pin(&keys[..1]).unwrap_err(), PlatformError::QueueFull);
        assert_eq!(s.delete(&keys), 0);
        assert_eq!(
            s.submit_flush(Incarnation::seq(1)).unwrap_err(),
            PlatformError::QueueFull
        );
        // Progress is monotone and the bytes seen so far are already counted.
        assert!(p.bytes >= last_bytes);
        assert_eq!(cap.used_bytes, p.bytes, "step {steps}");
        last_bytes = p.bytes;
    };
    assert!(steps >= 7, "the walk was resumed across {steps} calls");
    assert_eq!(done.indexed, 6);
    assert_eq!(done.tmp_deleted, 1);
    assert_eq!(done.bytes, 6 * OBJ);
    assert_eq!(s.capacity().state, StoreState::Open);
    assert!(s.lookup(&keys).iter().all(Option::is_some));
    let put = fx.put_full(&s, 50);
    assert!(put.job.is_ok());
    // A finished walk stays finished.
    let again = s.recover_step(0);
    assert!(again.done);
    assert_eq!(again.indexed, 6);
}

#[test]
fn unindexed_files_never_refuse_a_store_and_are_never_evicted_for_one() {
    // ADR 4.1 as amended (Grok C4 P2-5): unindexed files count toward bytes,
    // not toward the index cap, and their presence alone never refuses a
    // store. With no purge pending, room comes from evicting indexed objects
    // (LRU); the unindexed remainder is left to the purge.
    let fx = Fx::new();
    fx.prime();
    let keys: Vec<ChunkKey> = (1..=4).map(|n| plant_ours(&fx, n).0).collect();
    // Quota exactly full; room for one entry in the index, three files unindexed.
    let s = fx.open_raw(|c| {
        c.quota_bytes = 4 * OBJ;
        c.max_objects = 1;
        c.purge_after_recovery = false;
    });
    recover_all(&s);
    let cap = s.capacity();
    assert_eq!((cap.objects, cap.unindexed_objects), (1, 3));
    assert_eq!(cap.used_bytes, 4 * OBJ);
    let indexed = keys
        .iter()
        .copied()
        .find(|k| s.lookup(&[*k])[0].is_some())
        .expect("one indexed key");

    let put = fx.put_full(&s, 50);
    assert_eq!(
        terminal_of(&put.completions, put.job.unwrap()),
        Terminal::Done(JobStage::Durable),
        "the indexed object is the victim"
    );
    assert_eq!(
        s.lookup(&[indexed, put.key]),
        vec![None, Some(FULL_PAYLOAD)]
    );
    let cap = s.capacity();
    assert_eq!(cap.used_bytes, 4 * OBJ);
    assert_eq!((cap.objects, cap.unindexed_objects), (1, 3));
    assert_eq!(cap.unindexed_bytes, 3 * OBJ, "the remainder is untouched");

    // While a purge is pending only free headroom admits: none here.
    s.start_purge();
    let (job, _) = fx.submit(&s, 51, 8);
    assert_eq!(job.unwrap_err(), PlatformError::OutOfMemory);

    // The purge pass removes the remainder and nothing else.
    s.run_until_idle();
    let cap = s.capacity();
    assert_eq!((cap.used_bytes, cap.unindexed_objects), (OBJ, 0));
    assert_eq!(s.lookup(&[put.key])[0], Some(FULL_PAYLOAD));
    s.audit().unwrap();
}

#[test]
fn used_bytes_counts_header_region_and_physical_blocks() {
    use std::os::unix::fs::MetadataExt;
    let fx = Fx::new();
    fx.prime();
    // Ours: a whole number of 4 KiB blocks, header region included.
    let (_, ours) = plant_ours(&fx, 1);
    // A 4196-byte file occupies two 4 KiB blocks: blocks * 512 beats the size.
    let (_, small) = plant_foreign(&fx, 2, 100);
    // A sparse file: the size beats the allocated blocks.
    let (_, h) = fx.chunk(3, 8);
    let skey = ChunkKey(ObjectId([0xB3; 32]));
    let mut sh = h;
    sh.namespace_id = NamespaceId([0x55; 32]);
    sh.chunk_key = skey;
    sh.payload_len = 1 << 20;
    let digest = sha(&[0u8; 0]);
    let rec = RecordHeader {
        chunk: sh,
        payload_sha256: aien_kv_store::payload_root(&[digest]),
    };
    let sparse = fx.object_file(&skey);
    // Sealed-format marker only: recovery reads no payload, so a sparse file
    // needs no ciphertext (ADR 0003 section 20).
    let mut region = rec.encode_region(&[digest]).unwrap();
    mark_sealed(&mut region);
    std::fs::write(&sparse, region).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&sparse)
        .unwrap()
        .set_len(4096 + (1 << 20))
        .unwrap();

    let m_small = std::fs::metadata(&small).unwrap();
    let m_sparse = std::fs::metadata(&sparse).unwrap();
    assert!(
        m_small.blocks() * 512 > m_small.len(),
        "test setup: blocks beat size"
    );
    assert!(
        m_sparse.len() > m_sparse.blocks() * 512,
        "test setup: size beats blocks"
    );

    let want = phys_of(&ours) + phys_of(&small) + phys_of(&sparse);
    assert_eq!(phys_of(&ours), OBJ);
    assert_eq!(phys_of(&small), m_small.blocks() * 512);
    assert_eq!(phys_of(&sparse), 4096 + (1 << 20));
    let s = fx.open_raw(|_| {});
    let p = recover_all(&s);
    assert_eq!(p.bytes, want);
    assert_eq!(s.capacity().used_bytes, want);
    assert_eq!(s.capacity().objects, 3);
}

#[test]
fn restart_accounting_matches_before_and_after() {
    // A6 restart row, ADAPTER RECREATION: bytes counted before == bytes counted after.
    let fx = Fx::new();
    let s = fx.open();
    for n in 1..=4 {
        fx.put_full(&s, n);
    }
    fx.put(&s, 5, 3);
    let (fkey, fpath) = plant_foreign(&fx, 6, 100);
    let _ = fkey;
    let used_files = sum_phys(&object_files(fx.root()));
    // The foreign file arrived behind the adapter's back: a reopen counts it.
    let before = s.capacity();
    assert_eq!(before.used_bytes, 4 * OBJ + OBJ1);
    drop(s);
    // A crash left a temp behind; it is deleted, not counted.
    std::fs::write(fx.root().join("tmp").join("k.9.9.part"), vec![0u8; 12345]).unwrap();
    let s2 = fx.open_raw(|_| {});
    let p = recover_all(&s2);
    let after = s2.capacity();
    assert_eq!(after.used_bytes, used_files);
    assert_eq!(after.used_bytes, before.used_bytes + phys_of(&fpath));
    assert_eq!(after.objects, 6);
    assert_eq!(p.tmp_deleted, 1);
    assert_eq!(p.foreign_namespace, 1);
    assert_eq!(after.reserved_bytes, 0);
    assert_eq!(after.evicting_bytes, 0);
    s2.audit().unwrap();
}

#[test]
fn purge_evicts_least_recently_used_down_to_ninety_percent_of_quota() {
    let fx = Fx::new();
    fx.prime();
    let planted: Vec<_> = (1..=10u32).map(|n| plant_ours(&fx, n)).collect();
    for (i, (_, path)) in planted.iter().enumerate() {
        set_mtime(path, 1_000_000 + 100 * i as u64);
    }
    let quota = 6 * OBJ;
    let s = fx.open_raw(|c| {
        c.quota_bytes = quota;
        c.max_objects = 10;
    });
    recover_all(&s);
    assert!(s.purge_pending(), "over quota after the walk");
    let cap = s.capacity();
    assert_eq!(cap.state, StoreState::Open);
    assert_eq!(cap.used_bytes, 10 * OBJ);
    s.run_until_idle();
    assert!(!s.purge_pending());
    let cap = s.capacity();
    assert!(
        cap.used_bytes as u128 * 10 <= quota as u128 * 9,
        "back under 90% of quota"
    );
    assert_eq!(cap.used_bytes, 5 * OBJ);
    let keys: Vec<ChunkKey> = planted.iter().map(|(k, _)| *k).collect();
    let hit: Vec<bool> = s.lookup(&keys).iter().map(Option::is_some).collect();
    assert_eq!(
        hit,
        vec![false, false, false, false, false, true, true, true, true, true],
        "the five oldest went"
    );
    assert_eq!(s.stats().evicted_bytes_total, 5 * OBJ);
    s.audit().unwrap();
}

#[test]
fn purge_unlinks_unindexed_files_first_and_does_not_evict_when_under_quota() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=10 {
        plant_ours(&fx, n);
    }
    let s = fx.open_raw(|c| c.max_objects = 8); // purge_after_recovery stays on (default)
    recover_all(&s);
    assert!(s.purge_pending(), "unindexed files need the purge pass");
    assert_eq!(s.capacity().unindexed_objects, 2);
    s.run_until_idle();
    let cap = s.capacity();
    assert_eq!((cap.unindexed_objects, cap.unindexed_bytes), (0, 0));
    assert_eq!(cap.objects, 8);
    assert_eq!(cap.used_bytes, 8 * OBJ);
    assert_eq!(object_files(fx.root()).len(), 8);
    assert_eq!(s.stats().purge_unlinked, 2);
    assert_eq!(
        s.stats().evicted_bytes_total,
        0,
        "under quota: no indexed object was evicted"
    );
}

#[test]
fn recovery_through_the_worker_thread_reaches_open() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=3 {
        plant_ours(&fx, n);
    }
    let s: HostFsStore = fx.open_raw(|c| c.worker = aien_kv_store::host_fs::WorkerMode::Thread);
    wait_for("open", || s.capacity().state == StoreState::Open);
    assert_eq!(s.capacity().objects, 3);
    assert_eq!(s.capacity().used_bytes, 3 * OBJ);
}
