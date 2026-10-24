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

// The publication protocol, fault hooks and cache behaviour are written from
// ADR 0003 sections 2.4, 3.4 and 7.4. LMCache fs_l2_adapter.py (4d69cdc4,
// Apache-2.0, Copyright LMCache contributors) informed two ideas only: a
// completed object is published by writing a temporary file and renaming it,
// and a store of a key that already exists on disk is skipped. No code or
// test text was copied. See THIRD_PARTY.md and ADR 0003 context item 24.

//! The host filesystem `KvStore` adapter (cut C4, ADR 0003 sections 2.4, 3.4,
//! 7.4): real temporary directories, no mocks of the filesystem.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::{Boundary, FaultAction, HostFsStore, WorkerMode, EIO, ENOSPC};
use aien_kv_store::index::{Index, IndexEntry, FLAG_FOREIGN, INDEX_ENTRY_BYTES};
use aien_kv_store::{
    CancelOutcome, Incarnation, JobStage, KvStore, PublishOutcome, StoreState, Terminal,
};
use aien_platform::PlatformError;
use rig::*;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn failed(t: &Terminal) -> &PlatformError {
    match t {
        Terminal::Failed(e) => e,
        other => panic!("expected a failed terminal, got {other:?}"),
    }
}

fn is_hw(e: &PlatformError) -> bool {
    matches!(e, PlatformError::HardwareFault(_))
}

/// Load `key` and assert it returns exactly `expected`.
fn assert_loads(fx: &Fx, s: &HostFsStore, key: aien_kv_store::ChunkKey, n: u32, tokens: u32) {
    let (_, header) = fx.chunk(n, tokens);
    let (t, mem) = fx.load(s, key, &header);
    assert_eq!(t.unwrap(), Terminal::Done(JobStage::Completed));
    assert_eq!(mem.bytes(), expected_payload(n, tokens));
}

// ---- first open ------------------------------------------------------------

#[test]
fn first_open_precreates_256_shards_and_store_id_with_parent_fsync() {
    let fx = Fx::new();
    let s = fx.open_raw(|_| {});
    let st = s.stats();
    for b in 0..=255u8 {
        assert!(
            fx.root().join("objects").join(format!("{b:02x}")).is_dir(),
            "shard {b:02x}"
        );
    }
    assert!(fx.root().join("tmp").is_dir());
    let id = std::fs::read(fx.root().join("STORE_ID")).unwrap();
    assert_eq!(id.len(), 16);
    assert_eq!(s.store_id(), Some(id.clone().try_into().unwrap()));
    assert!(st.mkdirs >= 256, "mkdir each shard: {}", st.mkdirs);
    assert!(st.dir_fsyncs >= 2, "fsync(objects) and fsync(root)");
    assert!(
        tmp_files(fx.root()).is_empty(),
        "the STORE_ID temp was renamed"
    );
    recover_all(&s);

    // A publication never creates a directory.
    let before = s.stats();
    fx.put_full(&s, 1);
    let after = s.stats();
    assert_eq!(after.mkdirs, before.mkdirs);
    drop(s);

    // Reopen: same STORE_ID, nothing recreated.
    let s2 = fx.open_raw(|_| {});
    assert_eq!(s2.store_id(), Some(id.try_into().unwrap()));
    assert_eq!(s2.stats().mkdirs, 0);
}

#[test]
fn missing_shard_dir_is_recreated_at_open_with_a_parent_fsync() {
    let fx = Fx::new();
    fx.prime();
    std::fs::remove_dir(fx.root().join("objects").join("ab")).unwrap();
    let s = fx.open_raw(|_| {});
    assert!(fx.root().join("objects").join("ab").is_dir());
    let st = s.stats();
    assert_eq!(st.mkdirs, 1);
    assert!(st.dir_fsyncs >= 1);
    assert_ne!(s.capacity().state, StoreState::Failed);
}

#[test]
fn malformed_store_id_opens_failed_and_serves_nothing() {
    let fx = Fx::new();
    fx.prime();
    std::fs::write(fx.root().join("STORE_ID"), b"abc").unwrap();
    let s = fx.open_raw(|_| {});
    assert_eq!(s.capacity().state, StoreState::Failed);
    let (_, h) = fx.chunk(1, 8);
    assert_eq!(s.lookup(&[h.chunk_key]), vec![None]);
    let (job, _) = fx.submit(&s, 1, 8);
    assert_eq!(job.unwrap_err(), PlatformError::DeviceUnavailable);
    // A caller looping on recovery must stop: Failed is reported, never done.
    let p = s.recover_step(1);
    assert!(p.failed && !p.done);
    assert_eq!(s.capacity().state, StoreState::Failed);
}

// ---- A1 / A7: adapter recreation -----------------------------------------

#[test]
fn store_then_recreate_adapter_serves_same_bytes() {
    // A7 row: ADAPTER RECREATION (same process, new adapter on the same directory).
    // It is not a process restart, a clean reboot or a power loss.
    let fx = Fx::new();
    let s = fx.open();
    let id = s.store_id().unwrap();
    let a = fx.put(&s, 1, 8);
    let b = fx.put(&s, 2, 6); // 2 blocks, 2 real rows in the last
    let c = fx.put(&s, 3, 3); // 1 block, 3 real rows
    for p in [&a, &b, &c] {
        assert!(p.job.is_ok());
    }
    let cap = s.capacity();
    assert_eq!(cap.objects, 3);
    assert_eq!(cap.used_bytes, OBJ + OBJ + OBJ1);
    let report = s.shutdown(100);
    assert!(report.unfinished.is_empty());
    drop(s);

    let s2 = fx.open();
    assert_eq!(s2.store_id(), Some(id));
    let cap2 = s2.capacity();
    assert_eq!(cap2.state, StoreState::Open);
    assert_eq!(cap2.used_bytes, cap.used_bytes);
    assert_eq!(cap2.objects, 3);
    assert_eq!(
        s2.lookup(&[a.key, b.key, c.key]),
        vec![Some(FULL_PAYLOAD), Some(FULL_PAYLOAD), Some(BS as u64)]
    );
    assert_loads(&fx, &s2, a.key, 1, 8);
    assert_loads(&fx, &s2, b.key, 2, 6);
    assert_loads(&fx, &s2, c.key, 3, 3);
    s2.audit().unwrap();
}

#[test]
fn publication_order_is_write_fsync_rename_dir_fsync_then_index() {
    let fx = Fx::new();
    let s = fx.open();
    let before = s.stats();
    let p = fx.put_full(&s, 1);
    assert_eq!(
        fx.faults.seen(),
        vec![
            Boundary::P1Write,
            Boundary::P2FsyncFile,
            Boundary::P3Rename,
            Boundary::P4FsyncDir,
            Boundary::P5IndexInsert
        ]
    );
    let st = s.stats();
    assert_eq!(st.file_fsyncs - before.file_fsyncs, 1);
    assert_eq!(st.dir_fsyncs - before.dir_fsyncs, 1);
    assert_eq!(st.renames - before.renames, 1);
    assert!(st.write_bytes - before.write_bytes >= OBJ);
    assert!(tmp_files(fx.root()).is_empty());
    assert!(fx.object_file(&p.key).is_file());
    let job = p.job.unwrap();
    // Completed (progress), then the one terminal: Durable, published and durable together.
    let mine: Vec<_> = p.completions.iter().filter(|c| c.job == job).collect();
    assert_eq!(mine.len(), 2, "{mine:?}");
    assert_eq!(mine[0].stage, JobStage::Completed);
    assert!(mine[0].terminal.is_none());
    assert_eq!(mine[1].terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(mine[1].publish, PublishOutcome::Durable);
    assert_eq!(mine[1].bytes, FULL_PAYLOAD);
}

// ---- lookup and submit do no I/O -------------------------------------------

#[test]
fn lookup_is_index_only_no_io() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let (miss, _) = fx.chunk(99, 8);
    let before = s.stats();
    // Even with the files gone from under it, the index answers.
    std::fs::remove_file(fx.object_file(&a.key)).unwrap();
    for _ in 0..1000 {
        assert_eq!(s.lookup(&[a.key, miss]), vec![Some(FULL_PAYLOAD), None]);
    }
    let _ = s.capacity();
    let _ = s.poll(10);
    assert_eq!(s.stats().io_ops, before.io_ops);
}

#[test]
fn submit_store_performs_no_io() {
    let fx = Fx::new();
    let s = fx.open();
    let before = s.stats();
    let (job, _) = fx.submit(&s, 1, 8);
    job.unwrap();
    assert_eq!(s.stats().io_ops, before.io_ops, "submit touches no file");
    assert!(tmp_files(fx.root()).is_empty());
    s.run_until_idle();
    assert!(s.stats().io_ops > before.io_ops);
}

// ---- reservation (A6) ------------------------------------------------------

#[test]
fn inflight_temp_bytes_are_reserved() {
    let fx = Fx::new();
    let cell: StoreCell = Arc::new(OnceLock::new());
    let seen = log();
    let (c2, s2) = (cell.clone(), seen.clone());
    let faults = fx.faults.clone();
    let mut cfg = fx.config();
    cfg.quota_bytes = OBJ + OBJ / 2;
    cfg.faults = Some(Arc::new(CallbackHook {
        scripted: faults,
        f: Box::new(move |b, _| {
            if b == Boundary::P2FsyncFile {
                // The temp file is fully written, the object is not published yet.
                let store = c2.get().expect("store");
                let cap = store.capacity();
                let tmp = tmp_files(store.root());
                s2.lock().unwrap().push(format!(
                    "reserved={} used={} tmp={} tmp_len={}",
                    cap.reserved_bytes,
                    cap.used_bytes,
                    tmp.len(),
                    tmp.first()
                        .map_or(0, |p| std::fs::metadata(p).unwrap().len())
                ));
            }
            None
        }),
    }));
    let store = Arc::new(HostFsStore::open(cfg).unwrap());
    cell.set(store.clone()).ok();
    recover_all(&store);

    let (j1, _) = fx.submit(&store, 1, 8);
    j1.unwrap();
    assert_eq!(store.capacity().reserved_bytes, OBJ, "reserved at submit");
    assert_eq!(store.capacity().inflight_bytes, OBJ);
    assert_eq!(store.capacity().park_ops, 1);
    // No overcommit: a second store beside the first does not fit the quota.
    let (j2, _) = fx.submit(&store, 2, 8);
    assert_eq!(j2.unwrap_err(), PlatformError::OutOfMemory);
    assert_eq!(
        store.capacity().reserved_bytes,
        OBJ,
        "refusal reserves nothing"
    );
    assert!(tmp_files(store.root()).is_empty(), "submit wrote no temp");

    store.run_until_idle();
    let log = seen.lock().unwrap().clone();
    assert_eq!(log.len(), 1, "{log:?}");
    assert_eq!(
        log[0],
        format!("reserved={OBJ} used=0 tmp=1 tmp_len={OBJ}"),
        "while the temp exists, its bytes are reserved and not yet used"
    );
    let cap = store.capacity();
    assert_eq!(cap.reserved_bytes, 0);
    assert_eq!(cap.used_bytes, OBJ);
    assert_eq!(cap.inflight_bytes, 0);
    store.audit().unwrap();
}

// ---- failed writes keep the last usable object (A4) -------------------------

#[test]
fn enospc_mid_write_leaves_last_object_intact() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let used = s.capacity().used_bytes;
    fx.faults.arm(Boundary::P1Write, FaultAction::Fail(ENOSPC));
    let b = fx.put_full(&s, 2);
    let t = terminal_of(&b.completions, b.job.unwrap());
    assert_eq!(*failed(&t), PlatformError::OutOfMemory);
    assert_eq!(s.lookup(&[b.key]), vec![None]);
    assert!(!fx.object_file(&b.key).exists());
    assert!(tmp_files(fx.root()).is_empty(), "partial temp unlinked");
    let cap = s.capacity();
    assert_eq!(cap.used_bytes, used);
    assert_eq!(cap.reserved_bytes, 0);
    assert_eq!(cap.state, StoreState::Open, "ENOSPC alone does not degrade");
    assert_eq!(s.stats().disk_full_total, 1);
    assert_eq!(s.lookup(&[a.key]), vec![Some(FULL_PAYLOAD)]);
    assert_loads(&fx, &s, a.key, 1, 8);
    // The fault was one-shot: the same store now succeeds.
    let b2 = fx.put_full(&s, 2);
    assert_eq!(
        terminal_of(&b2.completions, b2.job.unwrap()),
        Terminal::Done(JobStage::Durable)
    );
    s.audit().unwrap();
}

#[test]
fn torn_write_leaves_no_partial_object_and_keeps_last() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    fx.faults.arm(
        Boundary::P1Write,
        FaultAction::TornWrite {
            after_bytes: BS as u64 + 10,
            crash: false,
        },
    );
    let b = fx.put_full(&s, 2);
    let t = terminal_of(&b.completions, b.job.unwrap());
    assert!(is_hw(failed(&t)), "{t:?}");
    assert!(tmp_files(fx.root()).is_empty());
    assert_eq!(object_files(fx.root()).len(), 1);
    assert_loads(&fx, &s, a.key, 1, 8);
}

#[test]
fn fsync_failure_fails_job_and_publishes_nothing() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let renames = s.stats().renames;
    fx.faults.arm(Boundary::P2FsyncFile, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    let c = completion_of(&b.completions, b.job.unwrap());
    assert!(is_hw(failed(c.terminal.as_ref().unwrap())));
    assert_eq!(c.publish, PublishOutcome::NotApplicable);
    assert_eq!(s.lookup(&[b.key]), vec![None]);
    assert!(!fx.object_file(&b.key).exists());
    assert!(
        tmp_files(fx.root()).is_empty(),
        "temp unlinked, fd never retried"
    );
    assert_eq!(s.stats().renames, renames, "no rename after a failed fsync");
    // Degraded: loads are served, no new stores.
    assert_eq!(s.capacity().state, StoreState::Degraded);
    assert_loads(&fx, &s, a.key, 1, 8);
    let (job, _) = fx.submit(&s, 3, 8);
    assert_eq!(job.unwrap_err(), PlatformError::DeviceUnavailable);
    assert_eq!(s.capacity().reserved_bytes, 0);
}

#[test]
fn rename_failure_unlinks_temp_and_publishes_nothing() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    fx.faults.arm(Boundary::P3Rename, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    let t = terminal_of(&b.completions, b.job.unwrap());
    assert!(is_hw(failed(&t)));
    assert_eq!(s.lookup(&[b.key]), vec![None]);
    assert!(tmp_files(fx.root()).is_empty());
    assert!(!fx.object_file(&b.key).exists());
    assert_eq!(s.capacity().state, StoreState::Open);
    assert_eq!(s.capacity().reserved_bytes, 0);
    assert_loads(&fx, &s, a.key, 1, 8);
}

#[test]
fn dir_fsync_failure_after_rename_is_published_indeterminate() {
    let fx = Fx::new();
    let s = fx.open();
    fx.faults.arm(Boundary::P4FsyncDir, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    let c = completion_of(&b.completions, b.job.unwrap());
    assert!(is_hw(failed(c.terminal.as_ref().unwrap())));
    assert_eq!(c.publish, PublishOutcome::Indeterminate);
    assert_eq!(c.stage, JobStage::Published);
    // The object is whole, indexed and served; only its power-loss durability is unknown.
    assert_eq!(s.lookup(&[b.key]), vec![Some(FULL_PAYLOAD)]);
    assert!(fx.object_file(&b.key).is_file());
    assert_loads(&fx, &s, b.key, 2, 8);
    assert_eq!(s.capacity().state, StoreState::Degraded);
    assert_eq!(s.capacity().used_bytes, OBJ);
    assert_eq!(s.stats().indeterminate_total, 1);
    let (job, _) = fx.submit(&s, 3, 8);
    assert_eq!(job.unwrap_err(), PlatformError::DeviceUnavailable);
    drop(s);
    // Recovery after a restart indexes it like any other (it passed P1 to P3).
    let s2 = fx.open();
    assert_eq!(s2.lookup(&[b.key]), vec![Some(FULL_PAYLOAD)]);
    assert_loads(&fx, &s2, b.key, 2, 8);
}

#[test]
fn fault_at_each_publication_boundary_leaves_consistent_outcome() {
    // ADR 2.4 table: (boundary, object file exists, indexed, temp left, publish outcome, state).
    struct Case {
        at: Boundary,
        object: bool,
        publish: PublishOutcome,
        state: StoreState,
    }
    let cases = [
        Case {
            at: Boundary::P1Write,
            object: false,
            publish: PublishOutcome::NotApplicable,
            state: StoreState::Open,
        },
        Case {
            at: Boundary::P2FsyncFile,
            object: false,
            publish: PublishOutcome::NotApplicable,
            state: StoreState::Degraded,
        },
        Case {
            at: Boundary::P3Rename,
            object: false,
            publish: PublishOutcome::NotApplicable,
            state: StoreState::Open,
        },
        Case {
            at: Boundary::P4FsyncDir,
            object: true,
            publish: PublishOutcome::Indeterminate,
            state: StoreState::Degraded,
        },
    ];
    for case in cases {
        let name = format!("{:?}", case.at);
        let fx = Fx::new();
        let s = fx.open();
        let a = fx.put_full(&s, 1); // the last usable object
        let used = s.capacity().used_bytes;
        fx.faults.arm(case.at, FaultAction::Fail(EIO));
        let b = fx.put_full(&s, 2);
        let job = b.job.unwrap();
        let c = completion_of(&b.completions, job);
        assert!(is_hw(failed(c.terminal.as_ref().unwrap())), "{name}");
        assert_eq!(c.publish, case.publish, "{name}");
        assert_eq!(fx.object_file(&b.key).exists(), case.object, "{name}");
        assert_eq!(
            s.lookup(&[b.key]),
            vec![case.object.then_some(FULL_PAYLOAD)],
            "{name}"
        );
        assert!(tmp_files(fx.root()).is_empty(), "{name}: temp unlinked");
        let cap = s.capacity();
        assert_eq!(cap.state, case.state, "{name}");
        assert_eq!(cap.reserved_bytes, 0, "{name}");
        assert_eq!(
            cap.used_bytes,
            used + if case.object { OBJ } else { 0 },
            "{name}"
        );
        assert_eq!(fx.faults.armed(case.at), 0, "{name}: the fault fired");
        assert_loads(&fx, &s, a.key, 1, 8);
        s.audit().unwrap();
    }

    // P5 (index insert) cannot fail: capacity was reserved at submit. A fault armed
    // there is not an I/O failure and changes nothing.
    let fx = Fx::new();
    let s = fx.open();
    fx.faults
        .arm(Boundary::P5IndexInsert, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    assert_eq!(
        terminal_of(&b.completions, b.job.unwrap()),
        Terminal::Done(JobStage::Durable)
    );
    assert_eq!(s.lookup(&[b.key]), vec![Some(FULL_PAYLOAD)]);
}

#[test]
fn crash_at_each_publication_boundary_recovers_consistently() {
    // The process dies at the boundary: no cleanup, no completion. A new adapter
    // on the same directory must find the last usable object, never a partial
    // one, and no temp.
    struct Case {
        at: Boundary,
        action: FaultAction,
        survives: bool,
        temp_left: bool,
    }
    let cases = [
        Case {
            at: Boundary::P1Write,
            action: FaultAction::Crash,
            survives: false,
            temp_left: true,
        },
        Case {
            at: Boundary::P1Write,
            action: FaultAction::TornWrite {
                after_bytes: BS as u64 + 1,
                crash: true,
            },
            survives: false,
            temp_left: true,
        },
        Case {
            at: Boundary::P2FsyncFile,
            action: FaultAction::Crash,
            survives: false,
            temp_left: true,
        },
        Case {
            at: Boundary::P3Rename,
            action: FaultAction::Crash,
            survives: false,
            temp_left: true,
        },
        Case {
            at: Boundary::P4FsyncDir,
            action: FaultAction::Crash,
            survives: true,
            temp_left: false,
        },
        Case {
            at: Boundary::P5IndexInsert,
            action: FaultAction::Crash,
            survives: true,
            temp_left: false,
        },
    ];
    for case in cases {
        let name = format!("{:?}/{:?}", case.at, case.action);
        let fx = Fx::new();
        let s = fx.open();
        let a = fx.put_full(&s, 1);
        fx.faults.arm(case.at, case.action);
        let (key_b, _) = fx.chunk(2, 8);
        let (job, _) = fx.submit(&s, 2, 8);
        let job = job.unwrap();
        s.run_until_idle();
        let polled = drain(&s);
        assert!(
            polled.iter().all(|c| c.job != job || c.terminal.is_none()),
            "{name}: a dead process reports no terminal"
        );
        assert_eq!(
            tmp_files(fx.root()).len(),
            usize::from(case.temp_left),
            "{name}"
        );
        drop(s);

        let s2 = fx.open_raw(|_| {});
        let p = recover_all(&s2);
        assert_eq!(p.tmp_deleted, u64::from(case.temp_left), "{name}");
        assert_eq!(p.corrupt_deleted, 0, "{name}");
        assert!(tmp_files(fx.root()).is_empty(), "{name}");
        assert_loads(&fx, &s2, a.key, 1, 8);
        assert_eq!(
            s2.lookup(&[key_b]),
            vec![case.survives.then_some(FULL_PAYLOAD)],
            "{name}"
        );
        if case.survives {
            assert_loads(&fx, &s2, key_b, 2, 8);
        }
        assert_eq!(
            s2.capacity().used_bytes,
            OBJ + if case.survives { OBJ } else { 0 },
            "{name}"
        );
    }
}

// ---- A3: truncated, corrupt, wrong identity --------------------------------

#[test]
fn truncated_payload_fails_on_load_and_is_deleted() {
    for cut in [1u64, 4096 + BS as u64, 4096, 100, 0] {
        let fx = Fx::new();
        let s = fx.open();
        let a = fx.put_full(&s, 1);
        let keep = fx.put_full(&s, 2);
        let path = fx.object_file(&a.key);
        let len = std::fs::metadata(&path).unwrap().len();
        let new_len = if cut == 1 { len - 1 } else { cut };
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(new_len)
            .unwrap();
        let used = s.capacity().used_bytes;
        let (t, mem) = fx.load(&s, a.key, &a.header);
        let t = t.unwrap();
        assert_eq!(*failed(&t), PlatformError::InvalidLayout, "cut {cut}");
        assert!(
            mem.bytes().iter().all(|b| *b == 0xEE),
            "cut {cut}: destination untouched"
        );
        assert!(!path.exists(), "cut {cut}: deleted");
        assert_eq!(s.lookup(&[a.key]), vec![None]);
        assert_eq!(s.capacity().used_bytes, used - OBJ, "cut {cut}");
        assert_eq!(s.stats().corrupt_deleted_total, 1);
        assert_loads(&fx, &s, keep.key, 2, 8);
        s.audit().unwrap();
    }
}

#[test]
fn corrupt_payload_byte_fails_segment_hash_and_deletes() {
    use std::os::unix::fs::FileExt;
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let path = fx.object_file(&a.key);
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    // One byte in the second segment (second pool block).
    f.write_all_at(&[0x00], 4096 + BS as u64 + 17).unwrap();
    drop(f);
    let (t, mem) = fx.load(&s, a.key, &a.header);
    assert_eq!(*failed(&t.unwrap()), PlatformError::InvalidLayout);
    // The whole object is authenticated before any segment lands (ADR 0003
    // section 20), so not even the good first segment reaches the destination.
    assert!(
        mem.bytes().iter().all(|x| *x == 0xEE),
        "nothing reaches the destination from a refused object"
    );
    assert!(!path.exists());
    assert_eq!(s.lookup(&[a.key]), vec![None]);
}

#[test]
fn corrupt_header_region_on_load_is_refused_and_deleted() {
    use std::os::unix::fs::FileExt;
    // (offset, why): magic, a struct byte (header checksum), a segment digest
    // (root mismatch), padding.
    for (off, why) in [
        (0u64, "magic"),
        (150, "struct byte"),
        (256 + 40, "digest table"),
        (4000, "padding"),
    ] {
        let fx = Fx::new();
        let s = fx.open();
        let a = fx.put_full(&s, 1);
        let path = fx.object_file(&a.key);
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut b = [0u8; 1];
        std::os::unix::fs::FileExt::read_exact_at(&f, &mut b, off).unwrap();
        f.write_all_at(&[b[0] ^ 0xFF], off).unwrap();
        drop(f);
        let (t, mem) = fx.load(&s, a.key, &a.header);
        assert_eq!(*failed(&t.unwrap()), PlatformError::InvalidLayout, "{why}");
        assert!(mem.bytes().iter().all(|x| *x == 0xEE), "{why}");
        assert!(
            !path.exists(),
            "{why}: a header that fails validation deletes the object"
        );
    }
}

#[test]
fn load_with_wrong_expectation_is_refused_without_io() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let before = s.stats();
    let mut cases: Vec<(&str, aien_kv_store::ChunkHeader)> = Vec::new();
    let mut h = a.header;
    h.model_sha256[0] ^= 1;
    cases.push(("model", h));
    let mut h = a.header;
    h.namespace_id.0[0] ^= 1;
    cases.push(("namespace", h));
    let mut h = a.header;
    h.num_layers += 1;
    cases.push(("layers", h));
    let mut h = a.header;
    h.block_size = 8;
    cases.push(("block size", h));
    let mut h = a.header;
    h.dtype = 4;
    cases.push(("dtype", h));
    let mut h = a.header;
    h.payload_len = BS as u64;
    cases.push(("payload len", h));
    let mut h = a.header;
    h.chunk_key.0 .0[0] ^= 1;
    cases.push(("chunk key", h));
    for (why, h) in cases {
        let (job, mem) = fx.submit_load(&s, a.key, &h);
        assert_eq!(job.unwrap_err(), PlatformError::InvalidLayout, "{why}");
        assert!(mem.bytes().iter().all(|x| *x == 0xEE), "{why}");
    }
    assert_eq!(s.stats().io_ops, before.io_ops, "refused at submit, no I/O");
    assert!(
        fx.object_file(&a.key).is_file(),
        "a caller's wrong expectation deletes nothing"
    );
    // An absent key.
    let (k, h) = fx.chunk(77, 8);
    let (job, _) = fx.submit_load(&s, k, &h);
    assert_eq!(job.unwrap_err(), PlatformError::ObjectNotFound(k.0));
    // Wrong destination size.
    let mem = Mem::new(1, 0xEE);
    let (dst, _) = mem.dest(1);
    assert_eq!(
        s.submit_load(a.key, &a.header, dst, Incarnation::seq(1))
            .unwrap_err(),
        PlatformError::InvalidLayout
    );
    assert_loads(&fx, &s, a.key, 1, 8);
}

#[test]
fn file_header_disagreeing_with_the_requested_chunk_is_deleted_on_load() {
    // A self-consistent file under the right name whose position, parent or
    // token count differs from what the key's chain says: recovery cannot
    // know, the load's validation does.
    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Box<dyn Fn(&mut aien_kv_store::ChunkHeader)>)> = vec![
        ("position", Box::new(|h| h.first_position = 8)),
        ("parent", Box::new(|h| h.parent_key = [0x42; 32])),
        (
            "token count",
            Box::new(|h| {
                h.token_count = 7;
                h.flags = 1;
            }),
        ),
    ];
    for (why, mutate) in cases {
        let fx = Fx::new();
        fx.prime();
        let (key, header) = fx.chunk(1, 8);
        let mut planted = header;
        mutate(&mut planted);
        fx.plant(&key, &planted, &pattern(1, 2 * BS), BS);
        let s = fx.open();
        assert_eq!(
            s.lookup(&[key]),
            vec![Some(FULL_PAYLOAD)],
            "{why}: indexed at recovery"
        );
        let (t, mem) = fx.load(&s, key, &header);
        assert_eq!(*failed(&t.unwrap()), PlatformError::InvalidLayout, "{why}");
        assert!(mem.bytes().iter().all(|x| *x == 0xEE), "{why}");
        assert!(!fx.object_file(&key).exists(), "{why}");
    }
}

#[test]
fn released_source_fails_the_job_and_writes_nothing() {
    let fx = Fx::new();
    let s = fx.open();
    let (key, _) = fx.chunk(1, 8);
    let (job, handle) = fx.submit(&s, 1, 8);
    let job = job.unwrap();
    handle.revoke(); // the owner freed the blocks while the job was queued
    s.run_until_idle();
    let t = terminal_of(&drain(&s), job);
    assert_eq!(*failed(&t), PlatformError::InvalidLayout);
    assert_eq!(s.stats().released_source_violations, 1);
    assert_eq!(s.lookup(&[key]), vec![None]);
    assert!(tmp_files(fx.root()).is_empty());
    assert_eq!(s.capacity().reserved_bytes, 0);
}

// ---- tail rows (finding 3) ---------------------------------------------------

#[test]
fn tail_rows_above_token_count_are_zero_in_object() {
    for tokens in [6u32, 1, 3, 5, 8, 4] {
        let fx = Fx::new();
        let s = fx.open();
        let (key, header) = fx.chunk(1, tokens);
        let blocks = blocks_for(tokens);
        // The pool holds garbage everywhere, including rows another conversation left.
        let mem = Mem::new(blocks, 0xFF);
        let (src, _) = mem.source(blocks, tail_valid(tokens));
        let job = s.submit_store(header, src, Incarnation::seq(1)).unwrap();
        s.run_until_idle();
        assert_eq!(
            terminal_of(&drain(&s), job),
            Terminal::Done(JobStage::Durable),
            "{tokens}"
        );
        assert!(
            mem.bytes().iter().all(|b| *b == 0xFF),
            "the pool block is never modified"
        );

        let path = fx.object_file(&key);
        let plain = read_object_plain(&path);
        let payload = plain[4096..].to_vec();
        assert_eq!(payload.len(), blocks * BS);
        let mut expect = vec![0xFFu8; blocks * BS];
        let tail = tail_valid(tokens);
        if tail < BLOCK_TOKENS {
            let last = (blocks - 1) * BS;
            zero_tail_rows(&mut expect[last..last + BS], tail);
        }
        assert_eq!(payload, expect, "{tokens} tokens");
        if tail < BLOCK_TOKENS {
            // Spot check: layer 1, V plane, first row at or above the tail.
            let off = (blocks - 1) * BS + LAYER_STRIDE + PLANE_STRIDE + tail as usize * ROW;
            assert!(
                payload[off..off + ROW].iter().all(|b| *b == 0),
                "{tokens} tokens"
            );
            assert_eq!(payload[off - 1], 0xFF);
        }
        // The segment digests cover the zeroed bytes.
        let region = &plain[..4096];
        for (i, seg) in payload.chunks(BS).enumerate() {
            assert_eq!(
                &region[256 + 32 * i..256 + 32 * i + 32],
                &sha(seg),
                "segment {i}"
            );
        }
        // And a load returns the zeroed rows.
        let (t, m) = fx.load(&s, key, &header);
        assert_eq!(t.unwrap(), Terminal::Done(JobStage::Completed));
        assert_eq!(m.bytes(), expect, "{tokens} tokens");
    }
}

#[test]
fn tail_scratch_is_one_block() {
    let fx = Fx::new();
    let s = fx.open();
    assert_eq!(s.stats().tail_scratch_bytes, BS as u64);
}

// ---- accounting and eviction -------------------------------------------------

#[test]
fn used_bytes_counts_header_region_and_physical_blocks_after_publish() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let b = fx.put(&s, 2, 2);
    let want = phys_of(&fx.object_file(&a.key)) + phys_of(&fx.object_file(&b.key));
    assert_eq!(want, OBJ + OBJ1);
    assert_eq!(s.capacity().used_bytes, want);
    assert_eq!(s.capacity().objects, 2);
}

#[test]
fn eviction_runs_on_worker_not_in_submit() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = 2 * OBJ);
    let a = fx.put_full(&s, 1);
    let b = fx.put_full(&s, 2);
    assert_eq!(s.capacity().used_bytes, 2 * OBJ);
    let before = s.stats();
    let (job, _) = fx.submit(&s, 3, 8);
    let job = job.unwrap();
    // Submit reserved and nothing else: no unlink, no file touched, no lookup change.
    let mid = s.stats();
    assert_eq!(mid.io_ops, before.io_ops);
    assert_eq!(mid.unlinks, before.unlinks);
    assert!(fx.object_file(&a.key).is_file());
    assert_eq!(
        s.lookup(&[a.key, b.key]),
        vec![Some(FULL_PAYLOAD), Some(FULL_PAYLOAD)]
    );
    assert_eq!(s.capacity().reserved_bytes, OBJ);

    s.run_until_idle();
    assert_eq!(
        terminal_of(&drain(&s), job),
        Terminal::Done(JobStage::Durable)
    );
    let after = s.stats();
    assert_eq!(after.unlinks - before.unlinks, 1, "one victim");
    assert_eq!(after.evicted_bytes_total, OBJ);
    // The least recently used (a) went; b and the new object stay.
    assert!(!fx.object_file(&a.key).exists());
    assert!(fx.object_file(&b.key).is_file());
    let (k3, _) = fx.chunk(3, 8);
    assert_eq!(
        s.lookup(&[a.key, b.key, k3]),
        vec![None, Some(FULL_PAYLOAD), Some(FULL_PAYLOAD)]
    );
    let cap = s.capacity();
    assert!(cap.used_bytes <= cap.quota_bytes);
    assert_eq!(cap.used_bytes, 2 * OBJ);
    s.audit().unwrap();
}

#[test]
fn eviction_is_least_recently_used_by_load_time_and_skips_pinned() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = 3 * OBJ);
    let a = fx.put_full(&s, 1);
    let b = fx.put_full(&s, 2);
    let c = fx.put_full(&s, 3);
    // Loading `a` makes it the most recently used.
    assert_loads(&fx, &s, a.key, 1, 8);
    let token = s.pin(&[b.key]).unwrap();
    let d = fx.put_full(&s, 4);
    assert_eq!(
        terminal_of(&d.completions, d.job.unwrap()),
        Terminal::Done(JobStage::Durable)
    );
    // b is pinned (skipped); c is the least recently used unpinned object.
    assert_eq!(
        s.lookup(&[a.key, b.key, c.key, d.key]),
        vec![
            Some(FULL_PAYLOAD),
            Some(FULL_PAYLOAD),
            None,
            Some(FULL_PAYLOAD)
        ]
    );
    // Everything else pinned: nothing is evictable, so admission refuses at submit.
    let t2 = s.pin(&[a.key, d.key]).unwrap();
    let (job, _) = fx.submit(&s, 5, 8);
    assert_eq!(job.unwrap_err(), PlatformError::OutOfMemory);
    assert_eq!(s.capacity().reserved_bytes, 0);
    s.unpin(token);
    s.unpin(t2);
    let (job, _) = fx.submit(&s, 5, 8);
    job.unwrap();
    s.run_until_idle();
    s.audit().unwrap();
}

#[test]
fn delete_marks_at_once_and_unlinks_on_the_worker() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let b = fx.put_full(&s, 2);
    let token = s.pin(&[b.key]).unwrap();
    let used = s.capacity().used_bytes;
    let before = s.stats();
    assert_eq!(s.delete(&[a.key, b.key]), 1, "the pinned key is skipped");
    assert_eq!(
        s.lookup(&[a.key, b.key]),
        vec![None, Some(FULL_PAYLOAD)],
        "a misses at once"
    );
    assert_eq!(
        s.stats().unlinks,
        before.unlinks,
        "delete itself unlinks nothing"
    );
    assert!(fx.object_file(&a.key).is_file());
    let cap = s.capacity();
    assert_eq!(
        cap.used_bytes, used,
        "bytes are subtracted only after the unlink returns"
    );
    assert_eq!(cap.evicting_bytes, OBJ);
    s.run_until_idle();
    assert!(!fx.object_file(&a.key).exists());
    let cap = s.capacity();
    assert_eq!(cap.used_bytes, used - OBJ);
    assert_eq!(cap.evicting_bytes, 0);
    s.unpin(token);
    s.audit().unwrap();
}

#[test]
fn unlink_failure_returns_the_victim_to_service_and_the_new_object_stays_published() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = OBJ);
    let a = fx.put_full(&s, 1);
    fx.faults.arm(Boundary::Unlink, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    let t = terminal_of(&b.completions, b.job.unwrap());
    assert_eq!(
        t,
        Terminal::Done(JobStage::Durable),
        "the write already succeeded"
    );
    assert_eq!(
        s.lookup(&[a.key, b.key]),
        vec![Some(FULL_PAYLOAD), Some(FULL_PAYLOAD)]
    );
    assert!(fx.object_file(&a.key).is_file());
    assert_eq!(
        s.capacity().used_bytes,
        2 * OBJ,
        "over quota until a later purge"
    );
    assert_eq!(s.capacity().evicting_bytes, 0);
    assert_eq!(s.capacity().reserved_bytes, 0);
    assert!(tmp_files(fx.root()).is_empty());
    s.audit().unwrap();
}

// ---- jobs ----------------------------------------------------------------------

#[test]
fn store_of_a_published_key_is_a_zero_byte_job_and_in_flight_duplicates_share_one() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let writes = s.stats().writes;
    let (j, _) = fx.submit(&s, 1, 8);
    let j = j.unwrap();
    assert_ne!(Some(j), a.job.ok());
    s.run_until_idle();
    let c = completion_of(&drain(&s), j);
    assert_eq!(c.terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(c.bytes, 0);
    assert_eq!(s.stats().writes, writes, "nothing is rewritten");
    assert_eq!(s.stats().dedup_total, 1);

    let (j1, _) = fx.submit(&s, 2, 8);
    let (j2, _) = fx.submit(&s, 2, 8);
    assert_eq!(
        j1.unwrap(),
        j2.unwrap(),
        "same key in flight: the existing job"
    );
    assert_eq!(s.capacity().reserved_bytes, OBJ, "reserved once");
    s.run_until_idle();
    s.audit().unwrap();
}

#[test]
fn cancel_before_start_yields_one_cancelled_terminal_and_touches_nothing() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let (job, _) = fx.submit(&s, 2, 8);
    let job = job.unwrap();
    let before = s.stats();
    assert_eq!(s.cancel(job), CancelOutcome::Cancelled);
    assert_eq!(s.stats().io_ops, before.io_ops);
    s.run_until_idle();
    let v = drain(&s);
    assert_eq!(terminal_of(&v, job), Terminal::Cancelled);
    assert_eq!(s.capacity().reserved_bytes, 0);
    assert_eq!(s.capacity().park_ops, 0);
    assert_eq!(s.cancel(job), CancelOutcome::TooLate, "already terminal");

    // A queued load: cancelled before it starts, destination never written.
    let (lj, mem) = fx.submit_load(&s, a.key, &a.header);
    let lj = lj.unwrap();
    assert_eq!(s.cancel(lj), CancelOutcome::Cancelled);
    s.run_until_idle();
    assert_eq!(terminal_of(&drain(&s), lj), Terminal::Cancelled);
    assert!(mem.bytes().iter().all(|b| *b == 0xEE));
    assert!(!s.pump(), "nothing left to do");
    s.audit().unwrap();
}

#[test]
fn cancel_during_write_unlinks_the_temp_and_publishes_nothing() {
    let fx = Fx::new();
    let cell: StoreCell = Arc::new(OnceLock::new());
    let c2 = cell.clone();
    let outcome = log();
    let o2 = outcome.clone();
    let mut cfg = fx.config();
    cfg.faults = Some(Arc::new(CallbackHook {
        scripted: fx.faults.clone(),
        f: Box::new(move |b, key| {
            if b == Boundary::P1Write {
                let store = c2.get().unwrap();
                // The job id is 1: the first job of a fresh store.
                let _ = key;
                o2.lock()
                    .unwrap()
                    .push(format!("{:?}", store.cancel(aien_kv_store::JobId(1))));
            }
            None
        }),
    }));
    let store = Arc::new(HostFsStore::open(cfg).unwrap());
    cell.set(store.clone()).ok();
    recover_all(&store);
    let (key, _) = fx.chunk(1, 8);
    let (job, _) = fx.submit(&store, 1, 8);
    let job = job.unwrap();
    assert_eq!(job.0, 1);
    store.run_until_idle();
    assert_eq!(
        outcome.lock().unwrap().clone(),
        vec!["Cancelled".to_string()]
    );
    assert_eq!(terminal_of(&drain(&store), job), Terminal::Cancelled);
    assert!(tmp_files(fx.root()).is_empty());
    assert_eq!(store.lookup(&[key]), vec![None]);
    assert_eq!(store.capacity().reserved_bytes, 0);
}

#[test]
fn cancel_after_rename_is_too_late_and_the_object_publishes() {
    let fx = Fx::new();
    let cell: StoreCell = Arc::new(OnceLock::new());
    let c2 = cell.clone();
    let outcome = log();
    let o2 = outcome.clone();
    let mut cfg = fx.config();
    cfg.faults = Some(Arc::new(CallbackHook {
        scripted: fx.faults.clone(),
        f: Box::new(move |b, _| {
            if b == Boundary::P4FsyncDir {
                let store = c2.get().unwrap();
                o2.lock()
                    .unwrap()
                    .push(format!("{:?}", store.cancel(aien_kv_store::JobId(1))));
            }
            None
        }),
    }));
    let store = Arc::new(HostFsStore::open(cfg).unwrap());
    cell.set(store.clone()).ok();
    recover_all(&store);
    let (key, _) = fx.chunk(1, 8);
    let (job, _) = fx.submit(&store, 1, 8);
    let job = job.unwrap();
    store.run_until_idle();
    assert_eq!(outcome.lock().unwrap().clone(), vec!["TooLate".to_string()]);
    assert_eq!(
        terminal_of(&drain(&store), job),
        Terminal::Done(JobStage::Durable)
    );
    assert_eq!(store.lookup(&[key]), vec![Some(FULL_PAYLOAD)]);
}

#[test]
fn every_job_yields_exactly_one_terminal_completion() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    fx.faults.arm(Boundary::P1Write, FaultAction::Fail(EIO));
    let mut jobs = Vec::new();
    for n in 2..5 {
        jobs.push(fx.submit(&s, n, 8).0.unwrap());
    }
    jobs.push(fx.submit_load(&s, a.key, &a.header).0.unwrap());
    jobs.push(s.submit_flush(Incarnation::seq(5)).unwrap());
    let cancelled = fx.submit(&s, 9, 8).0.unwrap();
    s.cancel(cancelled);
    jobs.push(cancelled);
    s.run_until_idle();
    let v = drain(&s);
    for j in jobs {
        let n = v
            .iter()
            .filter(|c| c.job == j && c.terminal.is_some())
            .count();
        assert_eq!(n, 1, "{j:?}");
    }
    assert!(!s.pump());
}

#[test]
fn per_class_bounds_refuse_with_queue_full_and_release_at_terminal() {
    let fx = Fx::new();
    let s = fx.open_with(|c| {
        c.max_park_ops = 2;
        c.max_restore_ops = 1;
    });
    let a = fx.put_full(&s, 1);
    let j1 = fx.submit(&s, 2, 8).0.unwrap();
    let _j2 = fx.submit(&s, 3, 8).0.unwrap();
    assert_eq!(fx.submit(&s, 4, 8).0.unwrap_err(), PlatformError::QueueFull);
    // The restore class has its own reservation.
    let (l1, _) = fx.submit_load(&s, a.key, &a.header);
    let l1 = l1.unwrap();
    let (l2, _) = fx.submit_load(&s, a.key, &a.header);
    assert_eq!(l2.unwrap_err(), PlatformError::QueueFull);
    let cap = s.capacity();
    assert_eq!((cap.park_ops, cap.restore_ops), (2, 1));
    s.run_until_idle();
    let v = drain(&s);
    assert!(v.iter().any(|c| c.job == j1 && c.terminal.is_some()));
    assert!(v.iter().any(|c| c.job == l1 && c.terminal.is_some()));
    let cap = s.capacity();
    assert_eq!(
        (cap.park_ops, cap.restore_ops, cap.inflight_bytes),
        (0, 0, 0)
    );
    fx.submit(&s, 4, 8).0.unwrap();
    s.run_until_idle();
}

#[test]
fn attach_waiter_is_bounded_per_job() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.max_waiters_per_job = 3);
    let a = fx.put_full(&s, 1);
    let (l, _) = fx.submit_load(&s, a.key, &a.header);
    let l = l.unwrap();
    s.attach_waiter(l, Incarnation::seq(77)).unwrap();
    s.attach_waiter(l, Incarnation::seq(77)).unwrap(); // requester + 2 attached = 3
    assert_eq!(
        s.attach_waiter(l, Incarnation::seq(77)).unwrap_err(),
        PlatformError::QueueFull
    );
    let st = fx.submit(&s, 2, 8).0.unwrap();
    assert!(matches!(
        s.attach_waiter(st, Incarnation::seq(77)),
        Err(PlatformError::TaskRejected(_))
    ));
    s.run_until_idle();
}

#[test]
fn pin_is_all_or_nothing_and_unpin_releases() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let (miss, _) = fx.chunk(50, 8);
    assert_eq!(
        s.pin(&[a.key, miss]).unwrap_err(),
        PlatformError::ObjectNotFound(miss.0)
    );
    let t = s.pin(&[a.key]).unwrap();
    assert_eq!(s.delete(&[a.key]), 0);
    s.unpin(t);
    assert_eq!(s.delete(&[a.key]), 1);
    s.run_until_idle();
    s.audit().unwrap();
}

#[test]
fn flush_job_syncs_directories_and_reports_durable() {
    let fx = Fx::new();
    let s = fx.open();
    fx.put_full(&s, 1);
    let before = s.stats();
    let j = s.submit_flush(Incarnation::seq(7)).unwrap();
    s.run_until_idle();
    let c = completion_of(&drain(&s), j);
    assert_eq!(c.terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(c.publish, PublishOutcome::Durable);
    assert!(s.stats().dir_fsyncs > before.dir_fsyncs);

    fx.faults.arm(Boundary::FlushSync, FaultAction::Fail(EIO));
    let j = s.submit_flush(Incarnation::seq(8)).unwrap();
    s.run_until_idle();
    let t = terminal_of(&drain(&s), j);
    assert!(is_hw(failed(&t)));
    assert_eq!(s.capacity().state, StoreState::Degraded);
    // A degraded store may still run a flush.
    s.submit_flush(Incarnation::seq(9)).unwrap();
    s.run_until_idle();
}

// ---- the worker thread and shutdown -----------------------------------------------

#[test]
fn thread_worker_recovers_and_publishes_without_a_manual_pump() {
    let fx = Fx::new();
    // Something to recover.
    {
        let s = fx.open();
        fx.put_full(&s, 1);
    }
    let s = fx.open_raw(|c| c.worker = WorkerMode::Thread);
    wait_for("recovery", || s.capacity().state == StoreState::Open);
    let (k1, _) = fx.chunk(1, 8);
    assert_eq!(s.lookup(&[k1]), vec![Some(FULL_PAYLOAD)]);
    let (job, _) = fx.submit(&s, 2, 8);
    let job = job.unwrap();
    let mut got = Vec::new();
    wait_for("terminal", || {
        got.extend(s.poll(10));
        got.iter().any(|c| c.job == job && c.terminal.is_some())
    });
    assert_eq!(terminal_of(&got, job), Terminal::Done(JobStage::Durable));
    let (k2, _) = fx.chunk(2, 8);
    assert_eq!(s.lookup(&[k2]), vec![Some(FULL_PAYLOAD)]);
    let r = s.shutdown(1000);
    assert!(r.unfinished.is_empty());
}

#[test]
fn shutdown_with_stalled_fsync_returns_by_deadline_and_reports_unfinished() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.worker = WorkerMode::Thread);
    fx.faults.arm(
        Boundary::P2FsyncFile,
        FaultAction::Stall(Duration::from_millis(1500)),
    );
    let (job, _) = fx.submit(&s, 1, 8);
    let job = job.unwrap();
    // Wait until the job is inside the stalled fsync.
    wait_for("stall", || fx.faults.fired(Boundary::P2FsyncFile) == 1);
    let queued = fx.submit(&s, 2, 8).0.unwrap();
    let t0 = Instant::now();
    let report = s.shutdown(100);
    assert!(
        t0.elapsed() < Duration::from_millis(1200),
        "returned by the deadline"
    );
    assert_eq!(report.unfinished.len(), 1, "{report:?}");
    assert_eq!(report.unfinished[0].job, job);
    assert_eq!(report.unfinished[0].bytes, OBJ);
    assert_eq!(s.stats().shutdown_unfinished_jobs, 1);
    // Intake is closed.
    assert_eq!(
        fx.submit(&s, 3, 8).0.unwrap_err(),
        PlatformError::DeviceUnavailable
    );
    // The stalled job is past nothing cancellable: it finishes its publication later;
    // the queued one was cancelled by the drain.
    let mut got = Vec::new();
    wait_for("both terminals", || {
        got.extend(s.poll(10));
        [job, queued]
            .iter()
            .all(|j| got.iter().any(|c| c.job == *j && c.terminal.is_some()))
    });
    assert_eq!(terminal_of(&got, job), Terminal::Done(JobStage::Durable));
    assert_eq!(terminal_of(&got, queued), Terminal::Cancelled);
}

#[test]
fn manual_shutdown_cancels_queued_jobs_and_finishes_nothing_half_done() {
    let fx = Fx::new();
    let s = fx.open();
    let a = fx.put_full(&s, 1);
    let j1 = fx.submit(&s, 2, 8).0.unwrap();
    let l1 = fx.submit_load(&s, a.key, &a.header).0.unwrap();
    let report = s.shutdown(500);
    assert!(report.unfinished.is_empty());
    assert_eq!(report.cancelled, 2);
    let v = drain(&s);
    assert_eq!(terminal_of(&v, j1), Terminal::Cancelled);
    assert_eq!(terminal_of(&v, l1), Terminal::Cancelled);
    assert!(tmp_files(fx.root()).is_empty());
    assert_eq!(
        fx.submit(&s, 3, 8).0.unwrap_err(),
        PlatformError::DeviceUnavailable
    );
}

// ---- settings -------------------------------------------------------------------------

#[test]
fn apply_env_parses_quota_max_objects_and_slice_and_rejects_garbage() {
    let fx = Fx::new();
    let mut c = fx.config();
    let env = |k: &str| match k {
        "AIEN_KV_STORE_QUOTA_BYTES" => Some("123456".to_string()),
        "AIEN_KV_STORE_MAX_OBJECTS" => Some("77".to_string()),
        "AIEN_KV_STORE_RECOVERY_SLICE_MS" => Some("9".to_string()),
        _ => None,
    };
    c.apply_env(env).unwrap();
    assert_eq!(
        (c.quota_bytes, c.max_objects, c.recovery_slice_ms),
        (123456, 77, 9)
    );
    // Unset keeps the config (defaults: 8 GiB, 65536, 50 ms).
    let mut d = fx.config();
    let (q, m, sl) = (d.quota_bytes, d.max_objects, d.recovery_slice_ms);
    d.apply_env(|_| None).unwrap();
    assert_eq!(
        (d.quota_bytes, d.max_objects, d.recovery_slice_ms),
        (q, m, sl)
    );
    let fresh =
        aien_kv_store::host_fs::HostFsConfig::from_fields(fx.root().to_path_buf(), &fx.fields)
            .unwrap();
    assert_eq!(fresh.quota_bytes, 8 << 30);
    assert_eq!(fresh.max_objects, 65536);
    assert_eq!(fresh.recovery_slice_ms, 50);
    assert_eq!((fresh.max_restore_ops, fresh.max_park_ops), (48, 64));
    assert_eq!(fresh.max_inflight_bytes, 256 << 20);
    // Garbage and zero are refused, and leave the config unchanged.
    for bad in ["abc", "", "-5", "0", "1.5"] {
        let mut e = fx.config();
        let (q0, m0) = (e.quota_bytes, e.max_objects);
        let r = e.apply_env(|k| (k == "AIEN_KV_STORE_QUOTA_BYTES").then(|| bad.to_string()));
        assert!(r.is_err(), "{bad:?}");
        assert_eq!((e.quota_bytes, e.max_objects), (q0, m0));
    }
}

#[test]
fn open_refuses_a_layout_whose_rows_do_not_fit_the_block() {
    let fx = Fx::new();
    let mut c = fx.config();
    c.layout.block_stride = 4096; // the last rows would fall outside the block
    assert!(HostFsStore::open(c).is_err());
    let mut c = fx.config();
    c.quota_bytes = 0;
    assert!(HostFsStore::open(c).is_err());
}

// ---- the index ---------------------------------------------------------------------------

fn entry(k: u8, phys: u64, mtime: u64, flags: u32) -> IndexEntry {
    IndexEntry {
        key: [k; 32],
        phys_bytes: phys,
        mtime_ns: mtime,
        payload_len: 1,
        flags,
    }
}

#[test]
fn index_entry_is_64_bytes_and_the_cap_is_4_mib() {
    assert_eq!(std::mem::size_of::<IndexEntry>(), INDEX_ENTRY_BYTES);
    assert_eq!(INDEX_ENTRY_BYTES * 65536, 4_194_304);
}

#[test]
fn index_aggregates_track_busy_and_evicting_entries() {
    let mut i = Index::new();
    for k in 1..=4u8 {
        i.insert(entry(k, 100 * k as u64, k as u64, 0));
    }
    assert_eq!(i.total_bytes(), 1000);
    assert_eq!(i.evictable(), (1000, 4));
    assert!(i.busy_inc(&[2; 32]));
    assert!(i.busy_inc(&[2; 32]));
    assert_eq!(i.evictable(), (800, 3));
    assert!(!i.mark_evicting(&[2; 32]), "busy entries are not marked");
    assert!(i.mark_evicting(&[3; 32]));
    assert_eq!(i.evicting(), (300, 1));
    assert_eq!(i.evictable(), (500, 2));
    assert!(!i.busy_inc(&[3; 32]), "a marked entry cannot be pinned");
    i.busy_dec(&[2; 32]);
    assert_eq!(i.evictable(), (500, 2), "still one reference");
    i.busy_dec(&[2; 32]);
    assert_eq!(i.evictable(), (700, 3));
    i.clear_evicting(&[3; 32]);
    assert_eq!(i.evictable(), (1000, 4));
    assert!(i.mark_evicting(&[3; 32]));
    let gone = i.remove(&[3; 32]).unwrap();
    assert_eq!(gone.phys_bytes, 300);
    assert_eq!(i.total_bytes(), 700);
    assert_eq!(i.evicting(), (0, 0));
    assert_eq!(i.len(), 3);
    i.audit().unwrap();
    // Replacing an entry keeps the aggregates right.
    assert!(i.insert(entry(1, 50, 9, 0)).is_some());
    assert_eq!(i.total_bytes(), 650);
    i.audit().unwrap();
}

#[test]
fn index_victim_prefers_foreign_then_oldest_and_never_busy_or_marked() {
    let mut i = Index::new();
    i.insert(entry(1, 10, 5, 0));
    i.insert(entry(2, 10, 3, 0));
    i.insert(entry(3, 10, 9, FLAG_FOREIGN));
    i.insert(entry(4, 10, 1, 0));
    assert_eq!(
        i.pick_victim(),
        Some([3; 32]),
        "foreign first, whatever its age"
    );
    assert!(i.mark_evicting(&[3; 32]));
    assert_eq!(i.pick_victim(), Some([4; 32]), "then the oldest");
    assert!(i.busy_inc(&[4; 32]));
    assert_eq!(i.pick_victim(), Some([2; 32]));
    i.touch(&[2; 32], 100);
    assert_eq!(i.pick_victim(), Some([1; 32]), "a load raises recency");
    assert!(i.busy_inc(&[1; 32]));
    assert!(i.busy_inc(&[2; 32]));
    assert_eq!(i.pick_victim(), None);
}

// ---- victims are removed only after the new object is durable (C4 open item) -----

#[test]
fn failed_write_keeps_the_victim_it_would_have_evicted() {
    for boundary in [Boundary::P1Write, Boundary::P2FsyncFile] {
        let fx = Fx::new();
        let s = fx.open_with(|c| c.quota_bytes = OBJ);
        let a = fx.put_full(&s, 1);
        fx.faults.arm(boundary, FaultAction::Fail(EIO));
        let b = fx.put_full(&s, 2);
        let t = terminal_of(&b.completions, b.job.unwrap());
        assert!(matches!(t, Terminal::Failed(_)), "{boundary:?}: {t:?}");
        assert_eq!(
            s.lookup(&[a.key, b.key]),
            vec![Some(FULL_PAYLOAD), None],
            "{boundary:?}: the old object is still served"
        );
        assert!(fx.object_file(&a.key).is_file(), "{boundary:?}");
        let cap = s.capacity();
        assert_eq!(cap.used_bytes, OBJ, "{boundary:?}");
        assert_eq!(cap.evicting_bytes, 0, "{boundary:?}");
        assert_eq!(cap.reserved_bytes, 0, "{boundary:?}");
        assert_loads(&fx, &s, a.key, 1, 8);
        s.audit().unwrap();
    }
}

#[test]
fn indeterminate_publish_keeps_the_victim_until_the_new_object_is_known_durable() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = OBJ);
    let a = fx.put_full(&s, 1);
    fx.faults.arm(Boundary::P4FsyncDir, FaultAction::Fail(EIO));
    let b = fx.put_full(&s, 2);
    let c = completion_of(&b.completions, b.job.unwrap());
    assert_eq!(c.publish, PublishOutcome::Indeterminate);
    assert_eq!(
        s.lookup(&[a.key, b.key]),
        vec![Some(FULL_PAYLOAD), Some(FULL_PAYLOAD)],
        "both survive: the new one may not outlive a power cut"
    );
    assert!(fx.object_file(&a.key).is_file());
    assert_eq!(s.capacity().evicting_bytes, 0);
    s.audit().unwrap();
}

#[test]
fn cancel_before_the_write_ends_leaves_every_victim_in_service() {
    let fx = Fx::new();
    let s = fx.open_with(|c| c.quota_bytes = OBJ);
    let a = fx.put_full(&s, 1);
    let (job, _) = fx.submit(&s, 2, 8);
    let job = job.unwrap();
    assert_eq!(s.cancel(job), CancelOutcome::Cancelled);
    s.run_until_idle();
    assert_eq!(terminal_of(&drain(&s), job), Terminal::Cancelled);
    assert_eq!(s.lookup(&[a.key]), vec![Some(FULL_PAYLOAD)]);
    assert!(fx.object_file(&a.key).is_file());
    assert_eq!(s.capacity().evicting_bytes, 0);
    s.audit().unwrap();
}
