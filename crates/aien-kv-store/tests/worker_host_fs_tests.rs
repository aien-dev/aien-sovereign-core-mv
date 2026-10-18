#![cfg(feature = "std")]
//! Integration: the C5 storage worker running over the real C4 host
//! filesystem adapter in a temporary directory (no fake, no mocked I/O).
//! Covers store, restore, cancel and shutdown through the worker's own
//! `KvStore` surface, and reopens the directory afterwards.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::HostFsStore;
use aien_kv_store::worker::{LoadTicket, StorageWorker, WorkerConfig};
use aien_kv_store::{
    CancelOutcome, Completion, Incarnation, JobId, JobStage, KvStore, StoreState, Terminal,
};
use rig::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Rigged {
    fx: Fx,
    store: Arc<HostFsStore>,
    worker: StorageWorker,
}

fn start(tweak: impl FnOnce(&mut WorkerConfig)) -> Rigged {
    let fx = Fx::new();
    let store = Arc::new(fx.open());
    let mut cfg = WorkerConfig {
        idle_poll: Duration::from_micros(200),
        start_paused: false,
        ..WorkerConfig::default()
    };
    tweak(&mut cfg);
    let drive = {
        let store = store.clone();
        move || {
            store.pump();
        }
    };
    let worker = StorageWorker::spawn(store.clone(), cfg, drive).expect("spawn");
    Rigged { fx, store, worker }
}

fn collect_terminals(w: &StorageWorker, jobs: &[JobId]) -> Vec<Completion> {
    let t0 = Instant::now();
    let mut all = Vec::new();
    loop {
        all.extend(w.poll(64));
        let done = jobs
            .iter()
            .all(|j| all.iter().any(|c| c.job == *j && c.terminal.is_some()));
        if done {
            return all;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "timed out: {all:?}");
        std::thread::sleep(Duration::from_micros(300));
    }
}

fn submit_store(r: &Rigged, n: u32) -> (JobId, aien_kv_store::HolderHandle, Mem) {
    let (_, h) = r.fx.chunk(n, 8);
    let mem = Mem::new(2, 0);
    let data = pattern(n, 2 * BS);
    mem.fill_with(|i| data[i]);
    let (src, handle) = mem.source(2, 4);
    let job = r
        .worker
        .submit_store(h, src, Incarnation::new(1000 + n as u64, 1))
        .expect("submit store");
    (job, handle, mem)
}

#[test]
fn worker_over_host_fs_stores_then_restores_bytes_exactly() {
    let r = start(|_| {});
    let (job, _h, _m) = submit_store(&r, 1);
    let v = collect_terminals(&r.worker, &[job]);
    let c = v.iter().find(|c| c.terminal.is_some()).unwrap();
    assert_eq!(c.terminal, Some(Terminal::Done(JobStage::Durable)));
    assert_eq!(c.incarnation, Incarnation::new(1001, 1));
    let (key, header) = r.fx.chunk(1, 8);
    assert!(r.fx.object_file(&key).is_file(), "a real file on disk");
    assert_eq!(r.worker.lookup(&[key]), vec![Some(FULL_PAYLOAD)]);

    let mem = Mem::new(2, 0xEE);
    let (dst, _hd) = mem.dest(2);
    let lj = r
        .worker
        .submit_load(key, &header, dst, Incarnation::new(2000, 3))
        .expect("submit load");
    let v = collect_terminals(&r.worker, &[lj]);
    let c = v
        .iter()
        .find(|c| c.job == lj && c.terminal.is_some())
        .unwrap();
    assert!(matches!(c.terminal, Some(Terminal::Done(_))), "{c:?}");
    assert_eq!(c.incarnation, Incarnation::new(2000, 3));
    assert_eq!(mem.bytes(), pattern(1, 2 * BS), "restored bytes match");
    r.store.audit().unwrap();
}

#[test]
fn worker_over_host_fs_routes_coalesced_restores_per_waiter() {
    let r = start(|c| c.start_paused = true);
    let (job, _h, _m) = submit_store(&r, 2);
    r.worker.resume_dispatch();
    collect_terminals(&r.worker, &[job]);
    let (key, header) = r.fx.chunk(2, 8);

    r.worker.pause_dispatch();
    let mem = Mem::new(2, 0xEE);
    let a = Incarnation::new(50, 1);
    let b = Incarnation::new(50, 2);
    let first = r
        .worker
        .submit_load_coalesced(key, &header, || mem.dest(2).0, a)
        .unwrap();
    let second = r
        .worker
        .submit_load_coalesced(key, &header, || panic!("no second buffer"), b)
        .unwrap();
    assert!(matches!(first, LoadTicket::Started(_)));
    assert_eq!(second, LoadTicket::Attached(first.job()));
    r.worker.resume_dispatch();
    let v = collect_terminals(&r.worker, &[first.job()]);
    // Wait for the second waiter's terminal too.
    let mut all = v;
    let t0 = Instant::now();
    while all
        .iter()
        .filter(|c| c.terminal.is_some() && c.job == first.job())
        .count()
        < 2
    {
        all.extend(r.worker.poll(64));
        assert!(t0.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_micros(300));
    }
    let mut incs: Vec<Incarnation> = all
        .iter()
        .filter(|c| c.job == first.job() && c.terminal.is_some())
        .map(|c| c.incarnation)
        .collect();
    incs.sort();
    assert_eq!(
        incs,
        vec![a, b],
        "same sequence id, two epochs, two answers"
    );
    assert_eq!(mem.bytes(), pattern(2, 2 * BS));
}

#[test]
fn worker_over_host_fs_cancel_before_dispatch_writes_nothing() {
    let r = start(|c| c.start_paused = true);
    let (job, handle, _m) = submit_store(&r, 3);
    assert_eq!(r.worker.cancel(job), CancelOutcome::Cancelled);
    let v = collect_terminals(&r.worker, &[job]);
    let c = v.iter().find(|c| c.terminal.is_some()).unwrap();
    assert_eq!(c.terminal, Some(Terminal::Cancelled));
    assert!(
        !handle.is_held(),
        "the source is released after the terminal"
    );
    let (key, _) = r.fx.chunk(3, 8);
    assert_eq!(r.worker.lookup(&[key]), vec![None]);
    assert!(object_files(r.fx.root()).is_empty());
    assert!(tmp_files(r.fx.root()).is_empty());
    r.worker.resume_dispatch();
    r.store.audit().unwrap();
}

#[test]
fn worker_over_host_fs_cancel_after_dispatch_still_ends_cleanly() {
    // Whichever side of the race the cancel lands on, there is exactly one
    // terminal, no temp file remains and the store stays consistent.
    for n in 10..18u32 {
        let r = start(|_| {});
        let (job, handle, _m) = submit_store(&r, n);
        let out = r.worker.cancel(job);
        let v = collect_terminals(&r.worker, &[job]);
        let terms: Vec<_> = v
            .iter()
            .filter(|c| c.job == job && c.terminal.is_some())
            .collect();
        assert_eq!(terms.len(), 1, "{out:?}: {v:?}");
        let (key, _) = r.fx.chunk(n, 8);
        match terms[0].terminal.as_ref().unwrap() {
            Terminal::Cancelled => assert_eq!(r.worker.lookup(&[key]), vec![None]),
            Terminal::Done(_) => assert_eq!(r.worker.lookup(&[key]), vec![Some(FULL_PAYLOAD)]),
            other => panic!("unexpected terminal {other:?}"),
        }
        let t0 = Instant::now();
        while handle.is_held() {
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "source never released"
            );
            std::thread::sleep(Duration::from_micros(300));
        }
        assert!(tmp_files(r.fx.root()).is_empty());
        r.store.audit().unwrap();
    }
}

#[test]
fn worker_over_host_fs_shutdown_drains_and_the_directory_reopens() {
    let r = start(|_| {});
    let mut jobs = Vec::new();
    let mut keep = Vec::new();
    for n in 20..26u32 {
        let (j, h, m) = submit_store(&r, n);
        jobs.push(j);
        keep.push((h, m));
    }
    let report = r.worker.shutdown(5_000);
    assert!(report.unfinished.is_empty(), "{report:?}");
    // Every accepted job has exactly one terminal.
    let v = collect_terminals(&r.worker, &jobs);
    for j in &jobs {
        let n = v
            .iter()
            .filter(|c| c.job == *j && c.terminal.is_some())
            .count();
        assert_eq!(n, 1, "{j:?}");
    }
    assert!(keep.iter().all(|(h, _)| !h.is_held()), "sources released");
    // After shutdown, intake is closed.
    let (_, h) = r.fx.chunk(30, 8);
    let mem = Mem::new(2, 0);
    let (src, _hh) = mem.source(2, 4);
    assert!(r.worker.submit_store(h, src, Incarnation::seq(1)).is_err());
    assert!(tmp_files(r.fx.root()).is_empty());

    let published: Vec<_> = (20..26u32)
        .map(|n| r.fx.chunk(n, 8).0)
        .filter(|k| r.fx.object_file(k).is_file())
        .collect();
    let Rigged { fx, store, worker } = r;
    drop(worker);
    // Dropping the handle does not join the worker thread, which still holds two
    // clones of the store until its loop returns. The store keeps the directory's
    // LOCK (#521) while any clone lives, so reopen only once the thread let go.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Arc::strong_count(&store) > 1 {
        assert!(Instant::now() < deadline, "worker thread kept the store");
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(store);
    let reopened = fx.open();
    assert_eq!(reopened.capacity().state, StoreState::Open);
    for k in &published {
        assert_eq!(
            reopened.lookup(&[*k]),
            vec![Some(FULL_PAYLOAD)],
            "survives reopen"
        );
    }
    reopened.audit().unwrap();
}

#[test]
fn worker_over_host_fs_shutdown_with_a_stalled_job_then_restart_recovers() {
    let r = start(|_| {});
    let (done, _h0, _m0) = submit_store(&r, 40);
    collect_terminals(&r.worker, &[done]);
    // The next write stalls inside fsync(file) for longer than the deadline.
    r.fx.faults.arm(
        aien_kv_store::host_fs::Boundary::P2FsyncFile,
        aien_kv_store::host_fs::FaultAction::Stall(Duration::from_millis(1_500)),
    );
    let (stalled, handle, _m) = submit_store(&r, 41);
    wait_for("the write reaches fsync", || {
        r.fx.faults
            .fired(aien_kv_store::host_fs::Boundary::P2FsyncFile)
            == 1
    });
    let t0 = Instant::now();
    let report = r.worker.shutdown(100);
    assert!(
        t0.elapsed() < Duration::from_millis(1_200),
        "shutdown returns by its deadline, not after the stall"
    );
    assert!(
        report.unfinished.iter().any(|u| u.job == stalled),
        "the stalled job is reported unfinished: {report:?}"
    );
    assert!(
        handle.is_held(),
        "its source is not released while I/O may run"
    );
    // Once the stall ends the job ends too, with exactly one terminal, and the
    // source is released.
    let v = collect_terminals(&r.worker, &[stalled]);
    assert_eq!(
        v.iter()
            .filter(|c| c.job == stalled && c.terminal.is_some())
            .count(),
        1
    );
    wait_for("source released", || !handle.is_held());

    // Restart: leave debris a crash would leave, reopen, recover, restore.
    let Rigged { fx, store, worker } = r;
    drop(worker);
    drop(store);
    std::fs::write(fx.root().join("tmp").join("leftover.1.2.3.part"), b"x").unwrap();
    let (junk, _) = fx.chunk(42, 8);
    std::fs::write(fx.object_file(&junk), vec![0xAB; 5000]).unwrap();
    let reopened = Arc::new(fx.open_raw(|_| {}));
    let p = recover_all(&reopened);
    assert!(p.done && !p.failed, "{p:?}");
    assert_eq!((p.tmp_deleted, p.corrupt_deleted), (1, 1));
    assert_eq!(reopened.capacity().state, StoreState::Open);
    assert!(tmp_files(fx.root()).is_empty());
    let drive = {
        let s = reopened.clone();
        move || {
            s.pump();
        }
    };
    let worker = StorageWorker::spawn(
        reopened.clone(),
        WorkerConfig {
            idle_poll: Duration::from_micros(200),
            start_paused: false,
            ..WorkerConfig::default()
        },
        drive,
    )
    .expect("spawn");
    for n in [40u32, 41] {
        let (key, header) = fx.chunk(n, 8);
        let mem = Mem::new(2, 0xEE);
        let (dst, _) = mem.dest(2);
        let job = worker
            .submit_load(key, &header, dst, Incarnation::new(7000 + n as u64, 1))
            .expect("restore after restart");
        let v = collect_terminals(&worker, &[job]);
        assert_eq!(
            v.iter()
                .find(|c| c.job == job && c.terminal.is_some())
                .unwrap()
                .terminal,
            Some(Terminal::Done(JobStage::Completed)),
            "object {n}"
        );
        assert_eq!(mem.bytes(), expected_payload(n, 8), "object {n} bytes");
    }
    reopened.audit().unwrap();
}
