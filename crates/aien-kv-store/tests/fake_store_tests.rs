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

// Test cases adapted (re-expressed in Rust, none copied) from LMCache
// tests/v1/distributed/test_nixl_store_dynamic_l2_adapter.py at 4d69cdc4,
// Apache-2.0, Copyright LMCache contributors:
//   test_delete_skips_pinned_key (line 887), test_store_rejected_when_capacity_exceeded
//   (line 985), test_batched_store_reports_failure_when_capacity_exceeded (line 1034),
//   test_load_partial_failure_marks_only_successful (line 561).
// See THIRD_PARTY.md and ADR 0003 context item 24.

//! The deterministic fake `KvStore` adapter (cut C2, ADR 0003 sections 3 and 13).

mod common;

use aien_kv_store::fake::{DiskEntry, DurabilityMode, Fault, JobScript};
use aien_kv_store::{
    CancelOutcome, Completion, HolderGuard, Incarnation, JobId, JobStage, KvStore, PublishOutcome,
    StoreState, Terminal,
};
use aien_platform::PlatformError;
use common::rig::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn terminal_of(c: &Completion) -> &Terminal {
    c.terminal.as_ref().expect("terminal completion")
}

fn only_terminal(v: &[Completion]) -> &Completion {
    let t: Vec<_> = v.iter().filter(|c| c.terminal.is_some()).collect();
    assert_eq!(t.len(), 1, "exactly one terminal in {v:?}");
    t[0]
}

#[test]
fn fake_stages_are_ordered_submitted_completed_published_durable() {
    // Host-like: publication includes the directory sync, one terminal carries Durable.
    let r = rig();
    r.script(JobScript::latency(100, 50));
    let (k, _) = r.chunk(1);
    let job = r.store(1).unwrap();
    assert_eq!(r.fake.job_stage(job), Some(JobStage::Submitted));
    r.fake.advance(99);
    assert!(r.fake.poll(8).is_empty(), "nothing before the latency");
    r.fake.advance(1); // t = 100: temp written and synced
    assert_eq!(r.fake.job_stage(job), Some(JobStage::Completed));
    let c1 = r.fake.poll(8);
    assert_eq!(c1.len(), 1);
    assert_eq!(c1[0].stage, JobStage::Completed);
    assert_eq!(c1[0].terminal, None, "Completed is progress, not terminal");
    assert_eq!(
        r.fake.lookup(&[k]),
        vec![None],
        "not visible before publication"
    );
    r.fake.advance(50); // t = 150: renamed and directory synced
    let c2 = r.fake.poll(8);
    assert_eq!(c2.len(), 1);
    assert_eq!(c2[0].stage, JobStage::Durable);
    assert_eq!(c2[0].terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(c2[0].publish, PublishOutcome::Durable);
    assert_eq!(c2[0].bytes, PAYLOAD);
    assert_eq!(c2[0].elapsed_us, 150);
    assert!(c1[0].stage < c2[0].stage && JobStage::Submitted < c1[0].stage);
    assert_eq!(r.fake.lookup(&[k]), vec![Some(PAYLOAD)]);
    assert_eq!(r.fake.is_durable(&k), Some(true));
    assert_eq!(r.fake.payload_of(&k), Some(pattern(1, PAYLOAD as usize)));
}

#[test]
fn fake_stages_published_is_not_durable_until_a_flush_on_flush_adapters() {
    let r = rig_with(|c| {
        c.durability = DurabilityMode::OnFlush;
        c.flush_us = 30;
    });
    let (k, _) = r.chunk(1);
    r.store(1).unwrap();
    let v = r.settle();
    assert_eq!(v.len(), 2, "Completed progress, then the terminal");
    assert_eq!(v[0].stage, JobStage::Completed);
    assert_eq!(v[1].stage, JobStage::Published);
    assert_eq!(v[1].terminal, Some(Terminal::Done(JobStage::Published)));
    assert_eq!(v[1].publish, PublishOutcome::NotApplicable);
    assert_eq!(r.fake.lookup(&[k]), vec![Some(PAYLOAD)]);
    assert_eq!(r.fake.is_durable(&k), Some(false), "published, not durable");

    // A flush submitted now covers it, and only it.
    let f = r.fake.submit_flush(Incarnation::seq(9)).unwrap();
    r.store(2).unwrap();
    let v = r.settle();
    let fl = v.iter().find(|c| c.job == f).unwrap();
    assert_eq!(fl.stage, JobStage::Durable);
    assert_eq!(fl.terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(r.fake.is_durable(&k), Some(true));
    let (k2, _) = r.chunk(2);
    assert_eq!(
        r.fake.is_durable(&k2),
        Some(false),
        "published after the flush started"
    );
}

#[test]
fn fake_refuses_over_ops_and_over_bytes() {
    // Ops: the park class holds 2 jobs; the third is refused, loads are not.
    let r = rig_with(|c| {
        c.max_park_ops = 2;
        c.max_restore_ops = 2;
        c.max_inflight_bytes = 64 * OBJ;
    });
    r.inject(50);
    r.store(1).unwrap();
    r.store(2).unwrap();
    assert_eq!(r.store(3), Err(PlatformError::QueueFull));
    assert_eq!(r.fake.capacity().park_ops, 2);
    assert_eq!(r.fake.stats().queue_full_total, 1);
    let (l, _m, _h) = r.load(50);
    assert!(l.is_ok(), "a full park class must not refuse a restore");
    assert_eq!(r.fake.capacity().restore_ops, 1);

    // Bytes: plenty of ops, but max_inflight_bytes is split 3:1 (ADR 7.1), so
    // with 4 objects' worth the park class owns one and restores own three.
    let r = rig_with(|c| {
        c.max_park_ops = 8;
        c.max_restore_ops = 8;
        c.max_inflight_bytes = 4 * OBJ;
    });
    r.store(1).unwrap();
    assert_eq!(r.fake.capacity().inflight_bytes, OBJ);
    assert_eq!(r.store(2), Err(PlatformError::QueueFull), "park bytes full");
    assert_eq!(r.fake.capacity().park_ops, 1);
    for n in 60..64 {
        r.inject(n);
    }
    assert!(r.load(60).0.is_ok());
    assert!(r.load(61).0.is_ok());
    assert!(
        r.load(62).0.is_ok(),
        "restores keep their own three quarters"
    );
    assert_eq!(r.fake.capacity().inflight_bytes, 4 * OBJ);
    assert_eq!(
        r.load(63).0,
        Err(PlatformError::QueueFull),
        "restore bytes are bounded"
    );
    // Credits come back at the terminal completion, polled or not.
    r.fake.run_until_idle();
    assert_eq!(r.fake.capacity().inflight_bytes, 0);
    assert!(r.store(2).is_ok());
}

#[test]
fn fake_failed_store_leaves_index_unchanged() {
    type Case = (&'static str, Fault, fn(&PlatformError) -> bool);
    let cases: Vec<Case> = vec![
        ("disk full", Fault::DiskFull, |e| {
            matches!(e, PlatformError::OutOfMemory)
        }),
        ("torn write", Fault::TornWrite, |e| {
            matches!(e, PlatformError::HardwareFault(_))
        }),
        ("fsync(file) fails", Fault::FsyncFileFail, |e| {
            matches!(e, PlatformError::HardwareFault(_))
        }),
    ];
    for (name, fault, is_expected) in cases {
        let r = rig();
        let a = r.inject(1);
        let (b, _) = r.chunk(2);
        let before = r.fake.capacity();
        let a_bytes = r.fake.payload_of(&a);
        r.script(JobScript::fault(fault));
        let job = r.store(2).unwrap();
        assert_eq!(r.fake.capacity().reserved_bytes, OBJ, "{name}: reserved");
        let v = r.settle();
        let t = only_terminal(&v);
        assert_eq!(t.job, job, "{name}");
        match terminal_of(t) {
            Terminal::Failed(e) => assert!(is_expected(e), "{name}: {e:?}"),
            other => panic!("{name}: {other:?}"),
        }
        assert!(
            t.stage < JobStage::Completed,
            "{name}: failed before Completed"
        );
        assert_eq!(t.publish, PublishOutcome::NotApplicable, "{name}");
        assert_eq!(r.fake.lookup(&[b]), vec![None], "{name}: nothing published");
        assert_eq!(
            r.fake.lookup(&[a]),
            vec![Some(PAYLOAD)],
            "{name}: old object"
        );
        assert_eq!(r.fake.payload_of(&a), a_bytes, "{name}: old bytes");
        let after = r.fake.capacity();
        assert_eq!(after.used_bytes, before.used_bytes, "{name}: used");
        assert_eq!(after.objects, before.objects, "{name}: objects");
        assert_eq!(after.reserved_bytes, 0, "{name}: reservation released");
        assert_eq!(after.inflight_bytes, 0, "{name}");
        match fault {
            Fault::TornWrite => {
                assert_eq!(
                    r.fake.torn_residue(),
                    1,
                    "partial temp stays until recovery"
                );
                assert_eq!(after.state, StoreState::Open);
            }
            Fault::FsyncFileFail => assert_eq!(after.state, StoreState::Degraded),
            _ => {
                assert_eq!(r.fake.torn_residue(), 0);
                assert_eq!(after.state, StoreState::Open);
            }
        }
    }
}

#[test]
fn fake_cancel_before_complete_publishes_nothing() {
    let r = rig();
    r.script(JobScript::latency(100, 100));
    let (k, _) = r.chunk(1);
    let job = r.store(1).unwrap();
    r.fake.advance(40);
    assert_eq!(r.fake.cancel(job), CancelOutcome::Cancelled);
    // Returns at once; the terminal arrives when the worker next runs.
    assert!(r.fake.poll(8).is_empty());
    r.fake.advance(0);
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, job);
    assert_eq!(*terminal_of(t), Terminal::Cancelled);
    assert_eq!(t.stage, JobStage::Submitted);
    assert_eq!(r.fake.lookup(&[k]), vec![None]);
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
    assert_eq!(r.fake.stats().cancelled_total, 1);
    assert_eq!(r.fake.stats().writes, 0, "the job never wrote");
    assert_eq!(r.fake.terminal_count(job), 1);

    // Cancelled after the temp object was written and synced, before the rename:
    // the temp is unlinked, nothing is published (ADR 3.3 table).
    r.script(JobScript::latency(100, 100));
    let (k2, _) = r.chunk(2);
    let job2 = r.store(2).unwrap();
    r.fake.advance(120);
    assert_eq!(r.fake.job_stage(job2), Some(JobStage::Completed));
    assert_eq!(r.fake.cancel(job2), CancelOutcome::Cancelled);
    let v = r.settle();
    assert_eq!(*terminal_of(only_terminal(&v)), Terminal::Cancelled);
    assert_eq!(r.fake.lookup(&[k2]), vec![None]);
    assert_eq!(r.fake.torn_residue(), 0, "the temp was unlinked");

    // A load cancelled before it landed never touches the destination.
    let k3 = r.inject(3);
    r.script(JobScript::latency(100, 0));
    let (l, mem, _h) = r.load(3);
    let l = l.unwrap();
    assert_eq!(r.fake.cancel(l), CancelOutcome::Cancelled);
    let v = r.settle();
    assert_eq!(*terminal_of(only_terminal(&v)), Terminal::Cancelled);
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK], "destination untouched");
    assert_eq!(r.fake.lookup(&[k3]), vec![Some(PAYLOAD)]);
}

#[test]
fn fake_cancel_after_complete_is_too_late() {
    let r = rig();
    let (k, _) = r.chunk(1);
    let job = r.store(1).unwrap();
    let v = r.settle();
    assert_eq!(
        *terminal_of(only_terminal(&v)),
        Terminal::Done(JobStage::Durable)
    );
    assert_eq!(r.fake.cancel(job), CancelOutcome::TooLate);
    r.fake.advance(1_000);
    assert!(r.fake.poll(8).is_empty(), "no second terminal");
    assert_eq!(
        r.fake.lookup(&[k]),
        vec![Some(PAYLOAD)],
        "published object stays"
    );
    assert_eq!(r.fake.stats().cancelled_total, 0);
    // An id the store never issued is also too late.
    assert_eq!(r.fake.cancel(JobId(9_999)), CancelOutcome::TooLate);

    // A finished load is too late as well, and its destination keeps the bytes.
    let (l, mem, _h) = r.load(1);
    let l = l.unwrap();
    r.settle();
    assert_eq!(r.fake.cancel(l), CancelOutcome::TooLate);
    assert_eq!(mem.bytes(), pattern(1, PAYLOAD as usize));
}

#[test]
fn fake_same_key_store_dedups() {
    let r = Arc::new(rig());
    let released = Arc::new(AtomicU64::new(0));
    let free_at_release = Arc::new(AtomicU64::new(0));
    let (k, h) = r.chunk(1);

    let first = r.store(1).unwrap();
    // The second source for the same key is released, not queued.
    let mem = Mem::new(2, 5);
    let guard = counting_guard_for(&r, &released, &free_at_release);
    let (src, _handle) = mem.source_with(2, 4, guard);
    let second = r.fake.submit_store(h, src, Incarnation::seq(2)).unwrap();
    assert_eq!(second, first, "same key in flight returns the existing job");
    assert_eq!(
        released.load(Ordering::SeqCst),
        1,
        "duplicate source released"
    );
    assert_eq!(
        free_at_release.load(Ordering::SeqCst),
        1,
        "released outside the store's lock"
    );
    assert_eq!(r.fake.capacity().park_ops, 1);
    assert_eq!(r.fake.capacity().reserved_bytes, OBJ, "reserved once");
    assert_eq!(r.fake.stats().dedup_total, 1);
    let v = r.settle();
    assert_eq!(v.iter().filter(|c| c.terminal.is_some()).count(), 1);
    assert_eq!(r.fake.stats().writes, 1);

    // Once published, a store of the same key is a no-op job that still ends
    // with its own single terminal completion.
    let third = r.store(1).unwrap();
    assert_ne!(third, first);
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, third);
    assert_eq!(t.bytes, 0, "nothing written");
    assert!(matches!(terminal_of(t), Terminal::Done(_)));
    assert_eq!(r.fake.stats().writes, 1, "still one write");
    assert_eq!(r.fake.stats().dedup_total, 2);
    assert_eq!(r.fake.capacity().objects, 1);
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
    assert_eq!(r.fake.lookup(&[k]), vec![Some(PAYLOAD)]);
}

#[test]
fn fake_delete_skips_pinned() {
    let r = rig_with(|c| c.unlink_us = 10);
    let k1 = r.inject(1);
    let k2 = r.inject(2);
    let k3 = r.inject(3);
    let tok = r.fake.pin(&[k1]).unwrap();

    // A key with a load in flight is skipped as well.
    let (l, _m, _h) = r.load(3);
    l.unwrap();

    assert_eq!(r.fake.delete(&[k1, k2, k3]), 1, "only k2 is deletable");
    // Marked: lookups miss at once, the bytes stay counted until the unlink returns.
    assert_eq!(
        r.fake.lookup(&[k1, k2, k3]),
        vec![Some(PAYLOAD), None, Some(PAYLOAD)]
    );
    let c = r.fake.capacity();
    assert_eq!(c.evicting_bytes, OBJ);
    assert_eq!(c.used_bytes, 3 * OBJ);
    r.fake.advance(9);
    assert_eq!(
        r.fake.capacity().used_bytes,
        3 * OBJ,
        "unlink not returned yet"
    );
    r.fake.advance(1);
    let c = r.fake.capacity();
    assert_eq!(c.used_bytes, 2 * OBJ);
    assert_eq!(c.evicting_bytes, 0);
    assert_eq!(c.objects, 2);
    assert_eq!(r.fake.payload_of(&k1), Some(pattern(1, PAYLOAD as usize)));

    r.fake.unpin(tok);
    assert_eq!(r.fake.delete(&[k1]), 1, "deletable once unpinned");
    r.fake.run_until_idle();
    assert_eq!(r.fake.lookup(&[k1]), vec![None]);

    // Pinning is all-or-nothing and needs published keys.
    let k4 = r.inject(4);
    let (missing, _) = r.chunk(99);
    assert!(matches!(
        r.fake.pin(&[k4, missing]),
        Err(PlatformError::ObjectNotFound(_))
    ));
    assert_eq!(r.fake.delete(&[k4]), 1, "the failed pin left no pin behind");
}

#[test]
fn fake_pin_tokens_are_bounded() {
    let r = rig_with(|c| c.max_restore_ops = 2);
    let k = r.inject(1);
    let t1 = r.fake.pin(&[k]).unwrap();
    let _t2 = r.fake.pin(&[k]).unwrap();
    assert_eq!(r.fake.pin(&[k]), Err(PlatformError::QueueFull));
    r.fake.unpin(t1);
    assert!(r.fake.pin(&[k]).is_ok());
}

#[test]
fn fake_submit_store_performs_no_io_and_no_eviction() {
    // Quota for exactly two objects, both resident; a third needs an eviction.
    let r = rig_with(|c| c.quota_bytes = 2 * OBJ);
    let a = r.inject(1);
    let b = r.inject(2);
    let s0 = r.fake.stats();
    let c0 = r.fake.capacity();

    let (c_key, _) = r.chunk(3);
    let job = r
        .store(3)
        .expect("evictable space makes the store admissible");
    let s1 = r.fake.stats();
    let c1 = r.fake.capacity();
    assert_eq!(s1.io_ops, s0.io_ops, "submit did no I/O");
    assert_eq!(
        (s1.reads, s1.writes, s1.unlinks),
        (s0.reads, s0.writes, s0.unlinks)
    );
    assert_eq!(s1.evicted_bytes_total, 0, "submit evicted nothing");
    assert_eq!(c1.objects, c0.objects);
    assert_eq!(c1.used_bytes, c0.used_bytes);
    assert_eq!(c1.evicting_bytes, 0);
    assert_eq!(c1.reserved_bytes, OBJ, "submit only reserved");
    assert_eq!(r.fake.lookup(&[a, b]), vec![Some(PAYLOAD), Some(PAYLOAD)]);

    // Eviction and the write happen when the worker runs, oldest object first.
    let v = r.settle();
    assert_eq!(only_terminal(&v).job, job);
    assert!(r.fake.stats().io_ops > s1.io_ops);
    assert_eq!(r.fake.stats().evicted_bytes_total, OBJ);
    assert_eq!(
        r.fake.lookup(&[a, b, c_key]),
        vec![None, Some(PAYLOAD), Some(PAYLOAD)]
    );
    let c2 = r.fake.capacity();
    assert_eq!(
        (c2.used_bytes, c2.reserved_bytes, c2.objects),
        (2 * OBJ, 0, 2)
    );

    // Everything pinned and nothing evictable: refused at submit, no reservation.
    let t = r.fake.pin(&[b, c_key]).unwrap();
    assert_eq!(r.store(4), Err(PlatformError::OutOfMemory));
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
    assert_eq!(r.fake.stats().disk_full_total, 1);
    r.fake.unpin(t);

    // Admitted while evictable, then everything gets pinned before the worker
    // runs: the job fails on the worker with OutOfMemory and releases its reservation.
    let job5 = r.store(5).unwrap();
    let t = r.fake.pin(&[b, c_key]).unwrap();
    let v = r.settle();
    let term = only_terminal(&v);
    assert_eq!(term.job, job5);
    assert_eq!(
        *terminal_of(term),
        Terminal::Failed(PlatformError::OutOfMemory)
    );
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
    assert_eq!(r.fake.capacity().objects, 2);
    r.fake.unpin(t);
}

#[test]
fn fake_reservation_prevents_overcommit_by_concurrent_stores() {
    let r = rig_with(|c| c.quota_bytes = 2 * OBJ);
    let j1 = r.store(1).unwrap();
    let _j2 = r.store(2).unwrap();
    assert_eq!(r.fake.capacity().reserved_bytes, 2 * OBJ);
    // Nothing is published yet, so a check on `used` alone would admit this.
    assert_eq!(r.store(3), Err(PlatformError::OutOfMemory));
    assert_eq!(r.fake.capacity().used_bytes, 0);
    // A cancelled job gives its reservation back.
    assert_eq!(r.fake.cancel(j1), CancelOutcome::Cancelled);
    r.fake.advance(0);
    assert_eq!(r.fake.capacity().reserved_bytes, OBJ);
    assert!(r.store(3).is_ok());
    let v = r.settle();
    assert_eq!(v.iter().filter(|c| c.terminal.is_some()).count(), 3);
    let c = r.fake.capacity();
    assert_eq!((c.used_bytes, c.reserved_bytes, c.objects), (2 * OBJ, 0, 2));
}

#[test]
fn fake_every_job_yields_exactly_one_terminal_completion() {
    let r = rig();
    let k_old = r.inject(40);
    let k_bad = r.inject(41);
    assert!(r.fake.corrupt_payload(&k_bad));
    let mut jobs: Vec<(&str, JobId)> = Vec::new();

    jobs.push(("store ok", r.store(1).unwrap()));
    r.script(JobScript::fault(Fault::DiskFull));
    jobs.push(("store disk full", r.store(2).unwrap()));
    r.script(JobScript::fault(Fault::TornWrite));
    jobs.push(("store torn", r.store(3).unwrap()));
    jobs.push(("store cancelled queued", r.store(4).unwrap()));
    let cancelled_queued = jobs.last().unwrap().1;
    r.script(JobScript::latency(100, 100));
    jobs.push(("store cancelled mid", r.store(5).unwrap()));
    let cancelled_mid = jobs.last().unwrap().1;
    jobs.push(("store dedup published", r.store(40).unwrap()));
    let (ok_load, _m1, _h1) = r.load(40);
    jobs.push(("load ok", ok_load.unwrap()));
    let (bad_load, _m2, _h2) = r.load(41);
    jobs.push(("load corrupt", bad_load.unwrap()));
    r.script(JobScript::latency(500, 0));
    let (cl, _m3, _h3) = r.load(40);
    jobs.push(("load cancelled", cl.unwrap()));
    let cancelled_load = jobs.last().unwrap().1;
    jobs.push(("flush", r.fake.submit_flush(Incarnation::seq(5)).unwrap()));

    assert_eq!(r.fake.cancel(cancelled_queued), CancelOutcome::Cancelled);
    assert_eq!(r.fake.cancel(cancelled_load), CancelOutcome::Cancelled);
    r.fake.advance(10);
    assert_eq!(r.fake.cancel(cancelled_mid), CancelOutcome::Cancelled);

    // The last job degrades the store, so it goes last.
    r.script(JobScript::fault(Fault::FsyncFileFail));
    jobs.push(("store fsync fails", r.store(6).unwrap()));

    let all = r.settle();
    assert!(
        r.fake.live_jobs().is_empty(),
        "every job reached a terminal"
    );
    for (name, job) in &jobs {
        let mine: Vec<_> = all.iter().filter(|c| c.job == *job).collect();
        let terminals = mine.iter().filter(|c| c.terminal.is_some()).count();
        assert_eq!(terminals, 1, "{name}: {mine:?}");
        assert!(
            mine.last().unwrap().terminal.is_some(),
            "{name}: nothing follows the terminal"
        );
        assert_eq!(r.fake.terminal_count(*job), 1, "{name}");
        assert!(mine.len() <= 3, "{name}: within the three reserved slots");
    }
    let st = r.fake.stats();
    assert_eq!(
        st.terminal_done + st.terminal_failed + st.terminal_cancelled,
        jobs.len() as u64
    );
    assert_eq!(st.terminal_cancelled, 3);
    // The corrupt object failed its load and was removed (cache semantics).
    let bad = all.iter().find(|c| c.job == jobs[7].1).unwrap();
    assert!(matches!(
        terminal_of(bad),
        Terminal::Failed(PlatformError::InvalidLayout)
    ));
    assert_eq!(r.fake.lookup(&[k_bad, k_old]), vec![None, Some(PAYLOAD)]);
}

#[test]
fn fake_rename_ok_dirsync_fail_reports_indeterminate_and_serves_object() {
    let r = rig();
    let (k, _) = r.chunk(1);
    r.script(JobScript::fault(Fault::RenameOkDirsyncFail));
    let job = r.store(1).unwrap();
    let v = r.settle();
    assert_eq!(v.len(), 2, "Completed progress, then the terminal");
    let t = &v[1];
    assert_eq!(t.job, job);
    assert_eq!(t.stage, JobStage::Published);
    assert_eq!(t.publish, PublishOutcome::Indeterminate);
    // The durability claim fails; the object does not.
    assert!(matches!(
        terminal_of(t),
        Terminal::Failed(PlatformError::HardwareFault(_))
    ));
    assert_eq!(
        r.fake.lookup(&[k]),
        vec![Some(PAYLOAD)],
        "the object is served"
    );
    assert_eq!(r.fake.payload_of(&k), Some(pattern(1, PAYLOAD as usize)));
    assert_eq!(r.fake.is_indeterminate(&k), Some(true));
    assert_eq!(r.fake.is_durable(&k), Some(false));
    assert_eq!(r.fake.stats().indeterminate_total, 1);
    let c = r.fake.capacity();
    assert_eq!(c.state, StoreState::Degraded);
    assert_eq!((c.objects, c.used_bytes, c.reserved_bytes), (1, OBJ, 0));

    // Degraded: loads are still served, new stores are not accepted.
    let (l, mem, _h) = r.load(1);
    l.unwrap();
    let v = r.settle();
    assert_eq!(
        *terminal_of(only_terminal(&v)),
        Terminal::Done(JobStage::Completed)
    );
    assert_eq!(mem.bytes(), pattern(1, PAYLOAD as usize));
    assert_eq!(r.store(2), Err(PlatformError::DeviceUnavailable));
}

#[test]
fn fake_lookup_misses_and_submit_refused_while_recovering() {
    let r = rig_with(|c| {
        c.start_recovering = true;
        c.recovery_entries_per_ms = 2;
        c.max_objects = 100;
    });
    let (k_ours, _) = r.chunk(1);
    r.fake.inject_disk_entry(disk_object(&r, 1, true));
    r.fake.inject_disk_entry(disk_object(&r, 2, false));
    r.fake.inject_disk_entry(DiskEntry::Tmp);
    r.fake.inject_disk_entry(DiskEntry::CorruptHeader {
        physical_bytes: 4096,
    });
    r.fake.inject_disk_entry(disk_object(&r, 3, true));

    assert_eq!(r.fake.capacity().state, StoreState::Recovering);
    assert_eq!(
        r.fake.lookup(&[k_ours]),
        vec![None],
        "a miss while recovering"
    );
    assert_eq!(r.store(9), Err(PlatformError::QueueFull));
    assert_eq!(
        r.fake.submit_flush(Incarnation::seq(1)),
        Err(PlatformError::QueueFull)
    );
    let (l, _m, _h) = r.load(1);
    assert_eq!(l, Err(PlatformError::QueueFull));
    assert_eq!(r.fake.stats().io_ops, 0);

    // A slice accounts for two entries; the pass is not done and admission stays closed.
    let p = r.fake.recover_step(1);
    assert!(!p.done);
    assert_eq!(r.fake.capacity().state, StoreState::Recovering);
    assert_eq!(
        r.fake.lookup(&[k_ours]),
        vec![None],
        "still closed mid-walk"
    );
    assert_eq!(r.store(9), Err(PlatformError::QueueFull));
    let p = r.fake.recover_step(1);
    assert!(!p.done);
    let p = r.fake.recover_step(1);
    assert!(p.done, "five entries, two per millisecond");
    assert_eq!(p.indexed, 2, "only our namespace is indexed");
    assert_eq!(p.foreign_namespace, 1);
    assert_eq!(p.tmp_deleted, 1);
    assert_eq!(p.corrupt_deleted, 1);
    assert_eq!(p.unindexed_objects, 0);
    assert_eq!(p.bytes, 3 * OBJ, "foreign objects count toward the quota");
    let c = r.fake.capacity();
    assert_eq!(c.state, StoreState::Open);
    assert_eq!(c.used_bytes, 3 * OBJ);
    assert_eq!(r.fake.lookup(&[k_ours]), vec![Some(PAYLOAD)]);
    assert!(
        r.store(9).is_ok(),
        "admission opens when the walk is complete"
    );
    // The torn-write residue of an earlier run would have been removed as well.
    assert_eq!(r.fake.torn_residue(), 0);
}

#[test]
fn fake_recovery_counts_objects_past_the_index_cap_without_indexing_them() {
    let r = rig_with(|c| {
        c.start_recovering = true;
        c.recovery_entries_per_ms = 100;
        c.max_objects = 2;
    });
    for n in 1..=4 {
        r.fake.inject_disk_entry(disk_object(&r, n, true));
    }
    let p = r.fake.recover_step(1);
    assert!(p.done);
    assert_eq!((p.indexed, p.unindexed_objects), (2, 2));
    let c = r.fake.capacity();
    assert_eq!(
        c.used_bytes,
        4 * OBJ,
        "used includes the unindexed remainder"
    );
    assert_eq!((c.unindexed_objects, c.unindexed_bytes), (2, 2 * OBJ));
    assert_eq!(c.objects, 2);
}

#[test]
fn fake_recover_step_removes_torn_write_residue() {
    let r = rig();
    r.script(JobScript::fault(Fault::TornWrite));
    r.store(1).unwrap();
    r.settle();
    assert_eq!(r.fake.torn_residue(), 1);
    let p = r.fake.recover_step(10);
    assert!(p.done);
    assert_eq!(p.tmp_deleted, 1);
    assert_eq!(r.fake.torn_residue(), 0);
}

#[test]
fn fake_poll_never_overflows_at_max_inflight() {
    let r = rig_with(|c| {
        c.max_restore_ops = 2;
        c.max_park_ops = 2;
    });
    let cap = r.fake.completion_capacity();
    assert_eq!(cap, 12, "three slots per possible in-flight op");
    r.inject(1);
    // Never poll: terminal completions pile up and slots stay reserved for them.
    let mut accepted = 0usize;
    loop {
        let (l, _m, _h) = r.load(1);
        match l {
            Ok(_) => accepted += 1,
            Err(e) => {
                assert_eq!(e, PlatformError::QueueFull);
                break;
            }
        }
        r.fake.run_until_idle();
        assert!(
            r.fake.pending_completions() <= cap,
            "queue {} over capacity {cap}",
            r.fake.pending_completions()
        );
        assert!(accepted <= cap, "the loop must stop");
    }
    assert_eq!(
        accepted, 10,
        "12 slots: a job needs 3 free beyond what is queued"
    );
    assert_eq!(r.fake.pending_completions(), 10);
    let st = r.fake.stats();
    assert!(st.peak_completion_queue as usize <= cap);
    assert_eq!(st.completion_overflow_total, 0);
    assert_eq!(
        r.fake.capacity().restore_ops,
        0,
        "ops were freed at the terminal"
    );

    // Draining makes room again; `poll(max)` honours max.
    assert_eq!(r.fake.poll(4).len(), 4);
    assert_eq!(r.fake.pending_completions(), 6);
    let (l, _m, _h) = r.load(1);
    assert!(l.is_ok(), "polling frees completion slots");
    r.fake.run_until_idle();
    assert!(r.fake.stats().peak_completion_queue as usize <= cap);
    assert_eq!(r.fake.stats().completion_overflow_total, 0);
}

#[test]
fn fake_writes_to_a_freed_destination_fail_the_test() {
    // Control: a normal load lands its bytes and records no violation.
    {
        let r = rig();
        r.inject(1);
        let (l, mem, _h) = r.load(1);
        l.unwrap();
        r.settle();
        assert_eq!(mem.bytes(), pattern(1, PAYLOAD as usize));
        assert_eq!(r.fake.stats().poison_violations, 0);
    }

    // The owner frees the destination while the job is still running: the
    // fake does not write, fails the job and records the violation.
    let r = rig();
    r.inject(1);
    r.script(JobScript::latency(100, 0));
    let (l, mem, handle) = r.load(1);
    let job = l.unwrap();
    handle.revoke(); // "free_sequence(holder)" ran too early
    mem.fill(0xDD); // the freed blocks are poisoned
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, job);
    assert_eq!(
        *terminal_of(t),
        Terminal::Failed(PlatformError::InvalidLayout)
    );
    assert_eq!(
        mem.bytes(),
        vec![0xDD; 2 * BLOCK],
        "no byte landed in freed memory"
    );
    assert_eq!(r.fake.stats().poison_violations, 1);

    // Unacknowledged, the violation fails the test when the store is dropped.
    let Rig { fake, .. } = r;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(fake)));
    assert!(
        outcome.is_err(),
        "dropping with an unacknowledged violation must panic"
    );

    // Acknowledged (a negative test of the detector itself), it is silent.
    let r = rig();
    r.inject(1);
    r.script(JobScript::latency(100, 0));
    let (l, _mem, handle) = r.load(1);
    l.unwrap();
    handle.revoke();
    r.settle();
    assert_eq!(r.fake.take_poison_violations(), 1);
    drop(r);
}

#[test]
fn fake_store_reading_a_freed_source_is_a_violation_too() {
    let r = rig();
    r.script(JobScript::latency(100, 0));
    let (job, handle, _mem) = r.store_h(1);
    let job = job.unwrap();
    handle.revoke();
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, job);
    assert_eq!(
        *terminal_of(t),
        Terminal::Failed(PlatformError::InvalidLayout)
    );
    assert_eq!(r.fake.take_poison_violations(), 1);
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
}

#[test]
fn fake_load_roundtrip_verifies_every_segment() {
    let r = rig();
    let (k, _) = r.chunk(7);
    r.store(7).unwrap();
    r.settle();
    let (l, mem, _h) = r.load(7);
    let l = l.unwrap();
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, l);
    assert_eq!(t.stage, JobStage::Completed);
    assert_eq!(*terminal_of(t), Terminal::Done(JobStage::Completed));
    assert_eq!(t.bytes, PAYLOAD);
    assert_eq!(mem.bytes(), pattern(7, PAYLOAD as usize));
    assert_eq!(r.fake.stats().reads, 1);

    // One flipped payload byte: the load fails, the object is removed, a
    // second load is a miss (ADR 1.4: nothing is partially published).
    assert!(r.fake.corrupt_payload(&k));
    let (l, _mem, _h) = r.load(7);
    let l = l.unwrap();
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, l);
    assert_eq!(
        *terminal_of(t),
        Terminal::Failed(PlatformError::InvalidLayout)
    );
    assert_eq!(r.fake.lookup(&[k]), vec![None]);
    let (l2, _m, _h) = r.load(7);
    assert!(matches!(l2, Err(PlatformError::ObjectNotFound(_))));
}

#[test]
fn fake_load_and_store_validate_at_submit() {
    let r = rig();
    let k = r.inject(1);
    // Wrong expectation: the stored header disagrees.
    let (_, mut wrong) = r.chunk(1);
    wrong.token_count = 4;
    let mem = Mem::new(2, 0xEE);
    let (d, _h) = mem.dest(2);
    assert_eq!(
        r.fake.submit_load(k, &wrong, d, Incarnation::seq(1)),
        Err(PlatformError::InvalidLayout)
    );
    // Wrong destination size.
    let (_, h) = r.chunk(1);
    let (d, _h) = mem.dest(1);
    assert_eq!(
        r.fake.submit_load(k, &h, d, Incarnation::seq(1)),
        Err(PlatformError::InvalidLayout)
    );
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK], "nothing written");
    // A store whose source does not match the header's payload length.
    let (_, h2) = r.chunk(2);
    let m1 = Mem::new(1, 0);
    let (s, _h) = m1.source(1, 4);
    assert_eq!(
        r.fake.submit_store(h2, s, Incarnation::seq(2)),
        Err(PlatformError::InvalidLayout)
    );
    // A foreign-namespace header.
    let (_, mut foreign) = r.chunk(3);
    foreign.namespace_id = aien_kv_store::NamespaceId([9; 32]);
    let m2 = Mem::new(2, 0);
    let (s, _h) = m2.source(2, 4);
    assert_eq!(
        r.fake.submit_store(foreign, s, Incarnation::seq(3)),
        Err(PlatformError::InvalidLayout)
    );
    assert_eq!(
        r.fake.capacity().reserved_bytes,
        0,
        "refusals reserve nothing"
    );
    assert_eq!(
        r.fake.capacity().restore_ops + r.fake.capacity().park_ops,
        0
    );
}

#[test]
fn fake_tail_valid_tokens_must_match_the_header() {
    let r = rig();
    // 6 tokens in blocks of 4: two blocks, 2 rows valid in the last.
    let (_, h6) = r.chunk_of(1, 6);
    assert_eq!(h6.payload_len, PAYLOAD);
    for (tail, ok) in [(2u32, true), (1, false), (4, false), (0, false)] {
        let m = Mem::new(2, 3);
        let (s, _h) = m.source(2, tail);
        let r2 = rig();
        let got = r2.fake.submit_store(h6, s, Incarnation::seq(1));
        assert_eq!(got.is_ok(), ok, "tail_valid_tokens {tail}");
        if !ok {
            assert_eq!(got, Err(PlatformError::InvalidLayout));
        }
    }
    // A full chunk wants a full last block.
    let (_, h8) = r.chunk(2);
    let m = Mem::new(2, 3);
    let (s, _h) = m.source(2, 3);
    assert_eq!(
        r.fake.submit_store(h8, s, Incarnation::seq(2)),
        Err(PlatformError::InvalidLayout)
    );
}

#[test]
fn fake_queue_full_and_flush_faults() {
    let r = rig();
    r.script(JobScript::fault(Fault::QueueFull));
    assert_eq!(r.store(1), Err(PlatformError::QueueFull));
    assert!(
        r.store(1).is_ok(),
        "the fault was consumed by the first submit"
    );
    assert_eq!(r.fake.stats().queue_full_total, 1);

    // Flush failure on a native-like adapter fail-stops the store.
    let r = rig_with(|c| c.durability = DurabilityMode::OnFlush);
    r.inject(2);
    r.script(JobScript::fault(Fault::FlushFail));
    let f = r.fake.submit_flush(Incarnation::seq(1)).unwrap();
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, f);
    assert!(matches!(
        terminal_of(t),
        Terminal::Failed(PlatformError::HardwareFault(_))
    ));
    assert_eq!(r.fake.capacity().state, StoreState::Failed);
    let (k, _) = r.chunk(2);
    assert_eq!(r.fake.lookup(&[k]), vec![None], "fail-stop serves nothing");
    assert_eq!(r.store(3), Err(PlatformError::DeviceUnavailable));
    let (l, _m, _h) = r.load(2);
    assert_eq!(l, Err(PlatformError::DeviceUnavailable));
}

#[test]
fn fake_duplicate_and_after_cancel_faults_exist_to_test_callers() {
    // The only two faults that break the one-terminal rule, on purpose.
    let r = rig();
    r.inject(1);
    r.script(JobScript::fault(Fault::DuplicateCompletion));
    let (l, _m, _h) = r.load(1);
    let l = l.unwrap();
    let v = r.settle();
    assert_eq!(v.iter().filter(|c| c.job == l).count(), 2);
    assert_eq!(r.fake.terminal_count(l), 2);

    let r = rig();
    r.script(JobScript::latency(100, 100).with(Fault::CompletionAfterCancel));
    let job = r.store(1).unwrap();
    assert_eq!(r.fake.cancel(job), CancelOutcome::Cancelled);
    let v = r.settle();
    assert_eq!(v.len(), 2, "{v:?}");
    assert_eq!(*terminal_of(&v[0]), Terminal::Cancelled);
    assert_eq!(
        v[1].stage,
        JobStage::Completed,
        "stale progress after the terminal"
    );
    assert_eq!(v[1].terminal, None);
    assert_eq!(
        v[1].incarnation, v[0].incarnation,
        "the caller drops it by incarnation"
    );
    assert!(r.fake.stats().completion_overflow_total <= 1);
    let (k, _) = r.chunk(1);
    assert_eq!(r.fake.lookup(&[k]), vec![None], "still nothing published");
}

#[test]
fn fake_stall_eviction_delays_only_the_victim_unlink_after_publish() {
    let r = rig_with(|c| {
        c.quota_bytes = OBJ;
        c.unlink_us = 10;
    });
    let a = r.inject(1);
    r.script(JobScript::latency(100, 0).with(Fault::StallEviction(1_000)));
    let job = r.store(2).unwrap();
    let (b, _) = r.chunk(2);
    r.fake.advance(500);
    // The new object is published first; the victim is unservable but still
    // on disk and counted until its (stalled) unlink returns.
    assert_eq!(r.fake.lookup(&[a, b]), vec![None, Some(PAYLOAD)]);
    let c = r.fake.capacity();
    assert_eq!(c.evicting_bytes, OBJ);
    assert_eq!(c.used_bytes, 2 * OBJ);
    assert_eq!(c.reserved_bytes, 0);
    let v = r.poll_all();
    assert_eq!(only_terminal(&v).job, job);
    r.fake.advance(10_000);
    assert_eq!(r.fake.lookup(&[a, b]), vec![None, Some(PAYLOAD)]);
    let c = r.fake.capacity();
    assert_eq!(
        (c.evicting_bytes, c.used_bytes, c.reserved_bytes),
        (0, OBJ, 0)
    );
    assert_eq!(r.fake.stats().evicted_bytes_total, OBJ);
}

#[test]
fn fake_unlink_failure_keeps_the_victim_and_the_published_store() {
    let r = rig_with(|c| c.quota_bytes = OBJ);
    let a = r.inject(1);
    r.script(JobScript::fault(Fault::UnlinkFail));
    let job = r.store(2).unwrap();
    let (b, _) = r.chunk(2);
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, job);
    assert_eq!(*terminal_of(t), Terminal::Done(JobStage::Durable));
    assert_eq!(
        r.fake.lookup(&[a, b]),
        vec![Some(PAYLOAD), Some(PAYLOAD)],
        "victim restored to service, new object published"
    );
    let c = r.fake.capacity();
    assert_eq!(
        (c.used_bytes, c.reserved_bytes, c.evicting_bytes),
        (2 * OBJ, 0, 0),
        "over quota until a later eviction"
    );
    assert_eq!(c.state, StoreState::Open);
}

#[test]
fn fake_shutdown_returns_at_the_deadline_and_keeps_buffers_of_unfinished_jobs() {
    let r = rig_with(|c| c.max_park_ops = 8);
    let released = Arc::new(AtomicU64::new(0));
    let guard = {
        let rel = Arc::clone(&released);
        HolderGuard::new(move || {
            rel.fetch_add(1, Ordering::SeqCst);
        })
    };

    // s1 is past its file sync and will finish its rename inside the window.
    r.script(JobScript::latency(100, 500));
    let s1 = r.store(1).unwrap();
    // s2's fsync stalls for a second: it cannot be cancelled while it runs.
    let (_, h2) = r.chunk(2);
    let mem2 = Mem::new(2, 0);
    mem2.fill_with(|i| pattern(2, PAYLOAD as usize)[i]);
    let (src2, handle2) = mem2.source_with(2, 4, guard);
    r.script(JobScript::latency(100, 100).with(Fault::StallFsync(1_000_000)));
    let s2 = r.fake.submit_store(h2, src2, Incarnation::seq(2)).unwrap();
    // s3 is held (never started), l1 is a load that has not landed.
    r.script(JobScript::fault(Fault::Hold));
    let s3 = r.store(3).unwrap();
    r.inject(4);
    r.script(JobScript::latency(5_000_000, 0));
    let (l1, _m, _h) = r.load(4);
    let l1 = l1.unwrap();

    r.fake.advance(100); // s1 is Completed, s2 is inside its stalled fsync
    assert_eq!(r.fake.job_stage(s1), Some(JobStage::Completed));
    let t0 = r.fake.now_us();
    let report = r.fake.shutdown(10); // 10 ms = 10_000 us
    assert_eq!(r.fake.now_us(), t0 + 10_000, "returned at the deadline");
    assert_eq!(report.published, 1, "s1 finished its rename: {report:?}");
    assert_eq!(report.cancelled, 2, "s3 and the load: {report:?}");
    assert_eq!(report.unfinished.len(), 1);
    let u = &report.unfinished[0];
    assert_eq!(u.job, s2);
    assert_eq!(u.bytes, OBJ);
    assert_eq!(u.stage, JobStage::Submitted);
    let st = r.fake.stats();
    assert_eq!(st.shutdown_unfinished_jobs, 1);
    assert_eq!(st.shutdown_unfinished_bytes, OBJ);

    // Nothing is freed to meet the deadline: s2's source stays owned by the store.
    assert!(handle2.is_held());
    assert_eq!(released.load(Ordering::SeqCst), 0);
    assert_eq!(r.fake.live_jobs(), vec![s2]);
    // Intake is closed.
    assert_eq!(r.store(9), Err(PlatformError::DeviceUnavailable));
    // The cancelled jobs still produce their one terminal each.
    let v = r.poll_all();
    for j in [s3, l1] {
        let mine: Vec<_> = v.iter().filter(|c| c.job == j).collect();
        assert_eq!(mine.len(), 1, "{j:?}");
        assert_eq!(*terminal_of(mine[0]), Terminal::Cancelled);
    }
    // When the stalled sync finally returns, s2 ends and only then is its buffer released.
    r.fake.advance(2_000_000);
    assert_eq!(released.load(Ordering::SeqCst), 1);
    assert!(r.fake.live_jobs().is_empty());
    let v = r.poll_all();
    assert_eq!(
        v.iter()
            .filter(|c| c.job == s2 && c.terminal.is_some())
            .count(),
        1
    );
    let (k1, _) = r.chunk(1);
    assert_eq!(
        r.fake.lookup(&[k1]),
        vec![Some(PAYLOAD)],
        "s1 stayed published"
    );
}

#[test]
fn fake_attach_waiter_is_bounded_per_job_and_in_total() {
    let r = rig_with(|c| {
        c.max_waiters_per_job = 3;
        c.max_waiters_total = 4;
    });
    r.inject(1);
    r.inject(2);
    r.script(JobScript::latency(1_000, 0));
    let (a, _ma, _ha) = r.load(1);
    let a = a.unwrap();
    r.script(JobScript::latency(1_000, 0));
    let (b, _mb, _hb) = r.load(2);
    let b = b.unwrap();
    assert!(r.fake.attach_waiter(a, Incarnation::seq(77)).is_ok()); // job a: 2 waiters
    assert!(r.fake.attach_waiter(a, Incarnation::seq(77)).is_ok()); // job a: 3 waiters, the per-job limit
    assert_eq!(
        r.fake.attach_waiter(a, Incarnation::seq(77)),
        Err(PlatformError::QueueFull)
    );
    assert!(r.fake.attach_waiter(b, Incarnation::seq(77)).is_ok()); // 3 attached in total
    assert!(r.fake.attach_waiter(b, Incarnation::seq(77)).is_ok()); // 4: the total bound
    assert_eq!(
        r.fake.attach_waiter(b, Incarnation::seq(77)),
        Err(PlatformError::QueueFull)
    );
    // Waiters end with their job; the total is released.
    r.fake.advance(1_000);
    let (c, _mc, _hc) = r.load(1);
    let c = c.unwrap();
    assert!(r.fake.attach_waiter(c, Incarnation::seq(77)).is_ok());
    // Only an in-flight load can take a waiter.
    assert!(matches!(
        r.fake.attach_waiter(a, Incarnation::seq(77)),
        Err(PlatformError::TaskRejected(_))
    ));
    let sj = r.store(5).unwrap();
    assert!(matches!(
        r.fake.attach_waiter(sj, Incarnation::seq(77)),
        Err(PlatformError::TaskRejected(_))
    ));
}

#[test]
fn fake_same_script_gives_the_same_completions_in_the_same_order() {
    fn run() -> Vec<Completion> {
        let r = rig_with(|c| {
            c.quota_bytes = 3 * OBJ;
            c.unlink_us = 7;
        });
        r.inject(1);
        r.inject(2);
        r.script(JobScript::latency(130, 40));
        r.store(3).unwrap();
        r.script(JobScript::latency(90, 90).with(Fault::RenameOkDirsyncFail));
        r.store(4).unwrap();
        r.script(JobScript::latency(10, 0));
        let (l, _m, _h) = r.load(2);
        let l = l.unwrap();
        r.fake.advance(5);
        let _ = r.fake.cancel(l);
        r.fake.advance(1_000);
        r.settle()
    }
    let a = run();
    let b = run();
    assert!(!a.is_empty());
    assert_eq!(a, b);
}

#[test]
fn fake_releases_holders_outside_its_lock() {
    let r = Arc::new(rig_with(|c| c.max_park_ops = 4));
    let released = Arc::new(AtomicU64::new(0));
    let free_at_release = Arc::new(AtomicU64::new(0));
    let mk = |n: u32| {
        let guard = counting_guard_for(&r, &released, &free_at_release);
        let (_, h) = r.chunk(n);
        let mem = Mem::new(2, 1);
        let (s, _handle) = mem.source_with(2, 4, guard);
        r.fake.submit_store(h, s, Incarnation::seq(n as u64))
    };
    let ok = mk(1).unwrap();
    let cancelled = mk(2).unwrap();
    r.fake.cancel(cancelled);
    r.script(JobScript::fault(Fault::TornWrite));
    let torn = mk(3).unwrap();
    r.fake.run_until_idle();
    assert_eq!(r.fake.terminal_count(ok), 1);
    assert_eq!(r.fake.terminal_count(torn), 1);
    assert_eq!(
        released.load(Ordering::SeqCst),
        3,
        "every source released at its terminal"
    );
    assert_eq!(
        free_at_release.load(Ordering::SeqCst),
        3,
        "each release ran with the store's lock free"
    );
}

fn counting_guard_for(
    r: &Arc<Rig>,
    released: &Arc<AtomicU64>,
    free: &Arc<AtomicU64>,
) -> HolderGuard {
    let r = Arc::clone(r);
    let rel = Arc::clone(released);
    let free = Arc::clone(free);
    HolderGuard::new(move || {
        rel.fetch_add(1, Ordering::SeqCst);
        if r.fake.lock_is_free() {
            free.fetch_add(1, Ordering::SeqCst);
        }
    })
}

// ---- fixes after the independent review ------------------------------------

/// A small host object (one 64-byte block, partial tail of 4 tokens): costs
/// `4096 + BLOCK` of quota, less than a full `OBJ`.
fn inject_small(r: &Rig, n: u32) -> aien_kv_store::ChunkKey {
    let (k, h) = r.chunk_of(n, 4);
    r.fake.inject_object(h, pattern(n as u8, BLOCK), BLOCK);
    k
}

#[test]
fn fake_max_objects_is_never_exceeded_after_store() {
    let r = rig_with(|c| c.max_objects = 2);
    let a = r.inject(1);
    let b = r.inject(2);
    let job = r.store(3).unwrap();
    let v = r.settle();
    assert_eq!(only_terminal(&v).job, job);
    assert_eq!(
        *terminal_of(only_terminal(&v)),
        Terminal::Done(JobStage::Durable)
    );
    let (c3, _) = r.chunk(3);
    assert_eq!(r.fake.lookup(&[c3]), vec![Some(PAYLOAD)]);
    let c = r.fake.capacity();
    assert_eq!(
        c.objects, 2,
        "the object cap holds after the store publishes"
    );
    // The oldest object paid for it.
    assert_eq!(r.fake.lookup(&[a, b]), vec![None, Some(PAYLOAD)]);
}

#[test]
fn fake_cancel_before_publish_keeps_victims() {
    // ADR 3.4 as amended: a store that does not become durable removes nothing.
    let r = rig_with(|c| {
        c.quota_bytes = OBJ;
        c.unlink_us = 10;
    });
    let a = r.inject(1);
    r.script(JobScript::latency(1_000, 0));
    let job = r.store(2).unwrap();
    r.fake.advance(100);
    assert_eq!(r.fake.lookup(&[a]), vec![None], "victim is marked");
    assert_eq!(r.fake.cancel(job), CancelOutcome::Cancelled);
    let v = r.settle();
    assert_eq!(*terminal_of(only_terminal(&v)), Terminal::Cancelled);
    assert_eq!(r.fake.lookup(&[a]), vec![Some(PAYLOAD)], "victim kept");
    let c = r.fake.capacity();
    assert_eq!(
        (c.used_bytes, c.reserved_bytes, c.evicting_bytes, c.objects),
        (OBJ, 0, 0, 1)
    );
    assert_eq!(r.fake.stats().unlinks, 0);

    // The shutdown variant: the cancelled job leaves its victim visible and resident.
    let r = rig_with(|c| {
        c.quota_bytes = OBJ;
        c.unlink_us = 10;
    });
    let a = r.inject(1);
    r.script(JobScript::latency(1_000, 0));
    let job = r.store(2).unwrap();
    r.fake.advance(100);
    let report = r.fake.shutdown(0);
    assert!(report.unfinished.is_empty());
    let v = r.settle();
    assert_eq!(only_terminal(&v).job, job);
    assert_eq!(*terminal_of(only_terminal(&v)), Terminal::Cancelled);
    assert_eq!(r.fake.lookup(&[a]), vec![Some(PAYLOAD)]);
    let c = r.fake.capacity();
    assert_eq!((c.used_bytes, c.evicting_bytes, c.objects), (OBJ, 0, 1));
}

#[test]
fn fake_failed_victim_unlink_keeps_only_that_victim() {
    // Two small victims are needed for one full object.
    let small = 4096 + BLOCK as u64;
    let r = rig_with(|c| {
        c.quota_bytes = 2 * small;
        c.unlink_us = 10;
    });
    let a = inject_small(&r, 1);
    let b = inject_small(&r, 2);
    r.script(JobScript::fault(Fault::UnlinkFail));
    let job = r.store(3).unwrap();
    let (k3, _) = r.chunk(3);
    let v = r.settle();
    let t = only_terminal(&v);
    assert_eq!(t.job, job);
    assert_eq!(*terminal_of(t), Terminal::Done(JobStage::Durable));
    assert_eq!(
        r.fake.lookup(&[a, b, k3]),
        vec![Some(BLOCK as u64), None, Some(PAYLOAD)],
        "the first unlink failed, the second ran"
    );
    let c = r.fake.capacity();
    assert_eq!(
        (c.used_bytes, c.reserved_bytes, c.evicting_bytes, c.objects),
        (small + OBJ, 0, 0, 2)
    );
    assert_eq!(r.fake.stats().unlinks, 1);
}

#[test]
fn fake_store_after_cancel_of_same_key_is_published() {
    let r = rig();
    let (k, _) = r.chunk(1);
    let j1 = r.store(1).unwrap();
    assert_eq!(r.fake.cancel(j1), CancelOutcome::Cancelled);
    let j2 = r.store(1).unwrap();
    assert_ne!(j1, j2, "a cancelling job is never a dedup target");
    let v = r.settle();
    assert_eq!(
        v.iter().filter(|c| c.terminal.is_some()).count(),
        2,
        "one terminal per job"
    );
    let t1 = v
        .iter()
        .find(|c| c.job == j1 && c.terminal.is_some())
        .unwrap();
    let t2 = v
        .iter()
        .find(|c| c.job == j2 && c.terminal.is_some())
        .unwrap();
    assert_eq!(*terminal_of(t1), Terminal::Cancelled);
    assert_eq!(*terminal_of(t2), Terminal::Done(JobStage::Durable));
    assert_eq!(r.fake.lookup(&[k]), vec![Some(PAYLOAD)]);
    assert_eq!(r.fake.terminal_count(j1), 1);
    assert_eq!(r.fake.terminal_count(j2), 1);
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
}

#[test]
fn fake_store_refuses_more_than_max_segments_blocks() {
    let r = rig();
    let (_, mut h) = r.chunk(1);
    // 121 blocks of 4 tokens: one more segment than a record can describe.
    h.payload_len = 121 * BLOCK as u64;
    h.token_count = 121 * 4;
    let mem = Mem::new(121, 1);
    let (src, _handle) = mem.source(121, 4);
    assert_eq!(
        r.fake.submit_store(h, src, Incarnation::seq(1)),
        Err(PlatformError::InvalidLayout)
    );
    assert_eq!(r.fake.capacity().reserved_bytes, 0);
    assert!(r.fake.live_jobs().is_empty());
}
