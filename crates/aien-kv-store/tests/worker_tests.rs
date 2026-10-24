//! The storage worker (cut C5, ADR 0003 sections 7 and 8) against the
//! deterministic fake store.
//!
//! The fake does all its I/O when its virtual clock is advanced. The worker
//! thread advances it through the `drive` closure given to `spawn`; the test
//! controls how fast (`W::step`). A `Probe` wraps the fake to make a store
//! slow or stalled in real time (a gate that blocks the calling thread) and to
//! record which thread made which call.

mod common;

use aien_kv_store::coalesce::{AttachRefused, CoalesceLimits, CoalesceTable, Detach, Share};
use aien_kv_store::fake::{FakeConfig, FakeKvStore, Fault, JobScript};
use aien_kv_store::worker::{Class, LoadTicket, StorageWorker, WorkerConfig};
use aien_kv_store::{
    CancelOutcome, Capacity, ChunkHeader, ChunkKey, Completion, DrainReport, HolderGuard,
    HolderHandle, Incarnation, JobId, JobStage, KvStore, NamespaceFields, NamespaceId, PinToken,
    PinnedDest, PinnedSource, PublishOutcome, RecoveryProgress, Terminal,
};
use aien_platform::PlatformError;
use common::rig::{pattern, tiny, Mem, BLOCK, OBJ, PAYLOAD};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- the probe

/// Blocks every thread that passes through it while closed.
struct Gate {
    open: Mutex<bool>,
    cv: Condvar,
    waiting: AtomicU64,
}

impl Gate {
    fn new() -> Gate {
        Gate {
            open: Mutex::new(true),
            cv: Condvar::new(),
            waiting: AtomicU64::new(0),
        }
    }

    fn set(&self, open: bool) {
        *self.open.lock().unwrap() = open;
        self.cv.notify_all();
    }

    fn pass(&self) {
        let mut g = self.open.lock().unwrap();
        if !*g {
            self.waiting.fetch_add(1, Ordering::SeqCst);
            while !*g {
                g = self.cv.wait(g).unwrap();
            }
            self.waiting.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn waiting(&self) -> u64 {
        self.waiting.load(Ordering::SeqCst)
    }
}

/// The fake behind a gate, recording who calls what.
struct Probe {
    fake: Arc<FakeKvStore>,
    gate: Arc<Gate>,
    /// Passed only inside `submit_*`: holds a job in `Dispatching`.
    submit_gate: Arc<Gate>,
    threads: Mutex<BTreeMap<&'static str, HashSet<ThreadId>>>,
    seq: Mutex<Vec<char>>,
    forged: Mutex<Vec<Completion>>,
    slow_us: AtomicU64,
}

impl Probe {
    fn note(&self, what: &'static str) {
        self.threads
            .lock()
            .unwrap()
            .entry(what)
            .or_default()
            .insert(thread::current().id());
    }

    fn enter(&self, what: &'static str) {
        self.note(what);
        self.gate.pass();
        let us = self.slow_us.load(Ordering::SeqCst);
        if us > 0 {
            thread::sleep(Duration::from_micros(us));
        }
    }

    fn threads_of(&self, what: &'static str) -> HashSet<ThreadId> {
        self.threads
            .lock()
            .unwrap()
            .get(what)
            .cloned()
            .unwrap_or_default()
    }

    fn submitted(&self) -> Vec<char> {
        self.seq.lock().unwrap().clone()
    }

    fn forge(&self, c: Completion) {
        self.forged.lock().unwrap().push(c);
    }
}

impl KvStore for Probe {
    fn namespace(&self) -> NamespaceId {
        self.fake.namespace()
    }
    fn capacity(&self) -> Capacity {
        self.note("capacity");
        self.fake.capacity()
    }
    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>> {
        self.note("lookup");
        self.fake.lookup(keys)
    }
    fn submit_store(
        &self,
        header: ChunkHeader,
        src: PinnedSource,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        self.enter("submit");
        self.submit_gate.pass();
        self.seq.lock().unwrap().push('S');
        self.fake.submit_store(header, src, inc)
    }
    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        self.enter("submit");
        self.submit_gate.pass();
        self.seq.lock().unwrap().push('L');
        self.fake.submit_load(key, expect, dst, inc)
    }
    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError> {
        self.enter("submit");
        self.submit_gate.pass();
        self.seq.lock().unwrap().push('F');
        self.fake.submit_flush(inc)
    }
    fn attach_waiter(&self, job: JobId, inc: Incarnation) -> Result<(), PlatformError> {
        self.fake.attach_waiter(job, inc)
    }
    fn poll(&self, max: usize) -> Vec<Completion> {
        self.enter("poll");
        let mut v = self.fake.poll(max);
        v.extend(self.forged.lock().unwrap().drain(..));
        v
    }
    fn cancel(&self, job: JobId) -> CancelOutcome {
        self.note("cancel");
        self.fake.cancel(job)
    }
    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError> {
        self.note("pin");
        self.fake.pin(keys)
    }
    fn unpin(&self, token: PinToken) {
        self.note("unpin");
        self.fake.unpin(token)
    }
    fn delete(&self, keys: &[ChunkKey]) -> u64 {
        self.note("delete");
        self.fake.delete(keys)
    }
    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress {
        self.fake.recover_step(slice_ms)
    }
    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport {
        self.enter("shutdown");
        self.fake.shutdown(intake_deadline_ms)
    }
}

// ------------------------------------------------------------------ the rig

struct W {
    fake: Arc<FakeKvStore>,
    probe: Arc<Probe>,
    worker: StorageWorker,
    /// Virtual microseconds the drive closure advances the fake per pass; 0 freezes it.
    step: Arc<AtomicU64>,
    fields: NamespaceFields<'static>,
    ns: NamespaceId,
}

fn rig() -> W {
    w_with(|_| {}, |_| {})
}

/// Paused at start, spinning (not sleeping) between passes, depth 4.
fn w_with(fake_cfg: impl FnOnce(&mut FakeConfig), worker_cfg: impl FnOnce(&mut WorkerConfig)) -> W {
    let fields = tiny();
    let ns = fields.namespace_id().unwrap();
    let mut fc = FakeConfig::new(ns);
    fake_cfg(&mut fc);
    let fake = Arc::new(FakeKvStore::new(fc));
    let gate = Arc::new(Gate::new());
    let probe = Arc::new(Probe {
        fake: fake.clone(),
        gate: gate.clone(),
        submit_gate: Arc::new(Gate::new()),
        threads: Mutex::new(BTreeMap::new()),
        seq: Mutex::new(Vec::new()),
        forged: Mutex::new(Vec::new()),
        slow_us: AtomicU64::new(0),
    });
    let mut wc = WorkerConfig {
        idle_poll: Duration::ZERO,
        start_paused: true,
        dispatch_depth: 4,
        ..WorkerConfig::default()
    };
    worker_cfg(&mut wc);
    let step = Arc::new(AtomicU64::new(100));
    let drive = {
        let (fake, gate, step, probe) = (fake.clone(), gate, step.clone(), probe.clone());
        move || {
            probe.note("drive");
            gate.pass();
            let s = step.load(Ordering::SeqCst);
            if s > 0 {
                fake.advance(s);
            }
        }
    };
    let worker = StorageWorker::spawn(probe.clone(), wc, drive).expect("spawn");
    W {
        fake,
        probe,
        worker,
        step,
        fields,
        ns,
    }
}

fn released_guard(n: &Arc<AtomicU64>) -> HolderGuard {
    let n = n.clone();
    HolderGuard::new(move || {
        n.fetch_add(1, Ordering::SeqCst);
    })
}

fn wait_for(what: &str, mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        if t0.elapsed() > Duration::from_secs(30) {
            panic!("timed out waiting for {what}");
        }
        thread::sleep(Duration::from_micros(200));
    }
}

fn terminals(v: &[Completion]) -> Vec<&Completion> {
    v.iter().filter(|c| c.terminal.is_some()).collect()
}

impl W {
    fn chunk(&self, n: u32) -> (ChunkKey, ChunkHeader) {
        let tokens: Vec<u32> = (0..8).map(|i| n * 1000 + i).collect();
        let key = aien_kv_store::chunk_chain(&self.ns, 8, &tokens).unwrap()[0];
        let h = self
            .fields
            .chunk_header(&self.ns, key, None, 0, 8, 1)
            .unwrap();
        (key, h)
    }

    fn key(&self, n: u32) -> ChunkKey {
        self.chunk(n).0
    }

    fn inject(&self, n: u32) -> ChunkKey {
        let (k, h) = self.chunk(n);
        self.fake
            .inject_object(h, pattern(n as u8, PAYLOAD as usize), BLOCK);
        k
    }

    fn store_with(
        &self,
        n: u32,
        guard: HolderGuard,
    ) -> (Result<JobId, PlatformError>, HolderHandle, Mem) {
        let (_, h) = self.chunk(n);
        let mem = Mem::new(2, 0);
        let p = pattern(n as u8, PAYLOAD as usize);
        mem.fill_with(|i| p[i]);
        let (src, handle) = mem.source_with(2, 4, guard);
        (
            self.worker.submit_store(h, src, Incarnation::seq(n as u64)),
            handle,
            mem,
        )
    }

    fn store_h(&self, n: u32) -> (Result<JobId, PlatformError>, HolderHandle, Mem) {
        self.store_with(n, HolderGuard::detached())
    }

    fn store(&self, n: u32) -> Result<JobId, PlatformError> {
        self.store_h(n).0
    }

    fn load_with(
        &self,
        n: u32,
        guard: HolderGuard,
    ) -> (Result<JobId, PlatformError>, Mem, HolderHandle) {
        let (k, h) = self.chunk(n);
        let mem = Mem::new(2, 0xEE);
        let (dst, handle) = mem.dest_with(2, guard);
        (
            self.worker
                .submit_load(k, &h, dst, Incarnation::seq(1000 + n as u64)),
            mem,
            handle,
        )
    }

    fn load_h(&self, n: u32) -> (Result<JobId, PlatformError>, Mem, HolderHandle) {
        self.load_with(n, HolderGuard::detached())
    }

    fn load(&self, n: u32) -> Result<JobId, PlatformError> {
        self.load_h(n).0
    }

    fn resume(&self) {
        self.worker.resume_dispatch();
    }

    fn poll_all(&self) -> Vec<Completion> {
        let mut out = Vec::new();
        loop {
            let v = self.worker.poll(64);
            if v.is_empty() {
                return out;
            }
            out.extend(v);
        }
    }

    /// Poll until `n` terminal completions have been seen; returns everything polled.
    fn collect(&self, n: usize) -> Vec<Completion> {
        let mut out = Vec::new();
        let mut seen = 0;
        let t0 = Instant::now();
        while seen < n {
            let v = self.worker.poll(64);
            seen += terminals(&v).len();
            out.extend(v);
            if t0.elapsed() > Duration::from_secs(30) {
                panic!("timed out: {seen} of {n} terminals, got {out:?}");
            }
            thread::yield_now();
        }
        out
    }

    /// Wait until nothing is in flight in the worker (all terminals produced), then poll all.
    fn settle(&self) -> Vec<Completion> {
        wait_for("all jobs terminal", || {
            let c = self.worker.capacity();
            c.restore_ops == 0 && c.park_ops == 0
        });
        self.poll_all()
    }

    fn dispatched(&self, job: JobId) -> bool {
        self.worker.inner_job(job).is_some()
    }

    /// Stop the worker thread inside a gated call of the store.
    fn stall_store(&self) {
        self.probe.gate.set(false);
        let g = self.probe.gate.clone();
        wait_for("worker blocked in the store", move || g.waiting() >= 1);
    }

    fn unstall_store(&self) {
        self.probe.gate.set(true);
    }
}

// ---------------------------------------------------------- the red tests

#[test]
fn worker_bounds_ops_and_bytes_in_flight() {
    // Ops: 2 parks, 3 restores. Bytes: 8 objects' worth, split 3:1.
    let w = w_with(
        |_| {},
        |c| {
            c.max_restore_ops = 3;
            c.max_park_ops = 2;
            c.max_inflight_bytes = 8 * OBJ;
        },
    );
    assert!(w.store(1).is_ok());
    assert!(w.store(2).is_ok());
    assert_eq!(w.store(3), Err(PlatformError::QueueFull), "park ops bound");
    let cap = w.worker.capacity();
    assert_eq!((cap.park_ops, cap.max_park_ops), (2, 2));
    assert_eq!(cap.restore_ops, 0);
    assert_eq!(
        cap.reserved_bytes,
        2 * OBJ,
        "queued stores hold a reservation"
    );
    assert_eq!(cap.inflight_bytes, 2 * OBJ);
    assert_eq!(cap.max_inflight_bytes, 8 * OBJ);
    for n in 50..54 {
        w.inject(n);
    }
    for n in 50..53 {
        assert!(
            w.load(n).is_ok(),
            "a full park class must not refuse a restore"
        );
    }
    assert_eq!(
        w.load(53),
        Err(PlatformError::QueueFull),
        "restore ops bound"
    );
    assert_eq!(w.worker.capacity().inflight_bytes, 5 * OBJ);
    let s = w.worker.stats();
    assert!(s.peak_restore_queue <= 3 && s.peak_park_queue <= 2, "{s:?}");
    assert_eq!(s.queue_full_total, 2);
    assert_eq!(
        w.fake.stats().io_ops,
        0,
        "nothing reaches storage while paused"
    );
    assert!(w.probe.submitted().is_empty());

    w.resume();
    let v = w.settle();
    assert_eq!(terminals(&v).len(), 5);
    let cap = w.worker.capacity();
    assert_eq!(
        (cap.restore_ops, cap.park_ops, cap.inflight_bytes),
        (0, 0, 0)
    );
    assert!(
        w.store(3).is_ok(),
        "credits come back at the terminal completion"
    );

    // Bytes: plenty of ops, 4 objects' worth in flight: one for parks, three for restores.
    let w = w_with(
        |_| {},
        |c| {
            c.max_restore_ops = 8;
            c.max_park_ops = 8;
            c.max_inflight_bytes = 4 * OBJ;
        },
    );
    assert!(w.store(1).is_ok());
    assert_eq!(w.store(2), Err(PlatformError::QueueFull), "park bytes full");
    for n in 60..64 {
        w.inject(n);
    }
    assert!(w.load(60).is_ok() && w.load(61).is_ok() && w.load(62).is_ok());
    assert_eq!(
        w.load(63),
        Err(PlatformError::QueueFull),
        "restore bytes full"
    );
    assert_eq!(w.worker.capacity().inflight_bytes, 4 * OBJ);
}

#[test]
fn worker_three_restores_then_one_park_when_both_waiting() {
    let w = w_with(|_| {}, |c| c.dispatch_depth = 1);
    for n in 1..=8 {
        w.inject(n);
    }
    let mut jobs = Vec::new();
    // Park submissions are interleaved with restores on purpose: arrival order must not matter.
    for n in 1..=8 {
        jobs.push((w.load(n).unwrap(), Class::Restore));
        if n % 3 == 0 {
            jobs.push((w.store(100 + n).unwrap(), Class::Park));
        }
    }
    jobs.push((w.store(200).unwrap(), Class::Park));
    assert_eq!(jobs.iter().filter(|(_, c)| *c == Class::Park).count(), 3);
    w.resume();
    let v = w.collect(11);
    assert!(terminals(&v)
        .iter()
        .all(|c| matches!(c.terminal, Some(Terminal::Done(_)))));
    use Class::{Park as P, Restore as R};
    let classes: Vec<Class> = w.worker.dispatch_order().iter().map(|(_, c)| *c).collect();
    assert_eq!(classes, vec![R, R, R, P, R, R, R, P, R, R, P]);
    // The store saw the same order (independent of the worker's own log).
    assert_eq!(
        w.probe.submitted().iter().collect::<String>(),
        "LLLSLLLSLLS"
    );
    // FIFO within a class.
    let restores: Vec<JobId> = w
        .worker
        .dispatch_order()
        .iter()
        .filter(|(_, c)| *c == R)
        .map(|(j, _)| *j)
        .collect();
    let expect: Vec<JobId> = jobs
        .iter()
        .filter(|(_, c)| *c == R)
        .map(|(j, _)| *j)
        .collect();
    assert_eq!(restores, expect);
    assert!(w.worker.stats().max_restore_run_with_park_waiting <= 3);
}

#[test]
fn worker_coalesces_same_key_loads_into_one_io() {
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let made = Arc::new(AtomicU64::new(0));
    let mut tickets = Vec::new();
    let mut mems = Vec::new();
    for i in 0..5u64 {
        let made = made.clone();
        let mem = Mem::new(2, 0xEE);
        mems.push(mem);
        let t = w
            .worker
            .submit_load_coalesced(
                k,
                &h,
                move || {
                    made.fetch_add(1, Ordering::SeqCst);
                    mem.dest(2).0
                },
                Incarnation::seq(500 + i),
            )
            .unwrap();
        tickets.push(t);
    }
    let LoadTicket::Started(job) = tickets[0] else {
        panic!("first request starts the load: {:?}", tickets[0]);
    };
    assert!(tickets[1..].iter().all(|t| *t == LoadTicket::Attached(job)));
    assert_eq!(
        made.load(Ordering::SeqCst),
        1,
        "followers allocate no landing buffer"
    );
    assert_eq!(w.worker.load_in_flight(&k), Some(job));
    assert_eq!(
        w.worker.capacity().restore_ops,
        1,
        "one job for five requests"
    );
    assert_eq!(w.worker.stats().coalesced_total, 4);
    // The trait-level attach joins the same job.
    assert_eq!(w.worker.attach_waiter(job, Incarnation::seq(77)), Ok(()));

    w.resume();
    let v = w.collect(6);
    let t = terminals(&v);
    assert_eq!(
        t.len(),
        6,
        "one terminal per waiter (5 requests + 1 attach): {v:?}"
    );
    assert!(t.iter().all(|c| c.job == job));
    assert_eq!(t[0].terminal, Some(Terminal::Done(JobStage::Completed)));
    assert_eq!(w.fake.stats().reads, 1, "one read of the object");
    assert_eq!(w.probe.submitted(), vec!['L'], "one submit to the store");
    assert_eq!(
        mems[0].bytes(),
        pattern(7, PAYLOAD as usize),
        "the leader's landing buffer holds the object"
    );
    assert_eq!(
        w.worker.load_in_flight(&k),
        None,
        "the entry ends with the job"
    );
    // A request after completion starts a fresh load.
    let mem = Mem::new(2, 0xEE);
    let t = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, Incarnation::seq(9))
        .unwrap();
    assert!(matches!(t, LoadTicket::Started(_)));
}

/// A waiter that attaches late, after the load is already dispatched to the
/// store (held there), rides the same job: it gets its own terminal with the
/// landing holder, it is counted against `max_waiters_total` while the job is
/// in flight, and the count is released when the job closes.
#[test]
fn worker_late_attach_after_dispatch_is_counted_and_released() {
    let (a, b, c, d) = (
        Incarnation::seq(10),
        Incarnation::seq(11),
        Incarnation::seq(12),
        Incarnation::seq(13),
    );
    let w = w_with(|_| {}, |wc| wc.max_waiters_total = 1);
    let k = w.inject(10);
    let (_, h) = w.chunk(10);
    let mem = Mem::new(2, 0xEE);
    w.fake.script(JobScript::fault(Fault::Hold));
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    assert_eq!(w.probe.submitted(), vec!['L'], "the store holds the load");
    assert_eq!(w.worker.capacity().restore_ops, 1);

    // Late attach: the job is at the store, not in the worker queue.
    assert_eq!(w.worker.attach_waiter(job, b), Ok(()));
    assert_eq!(
        w.worker.attach_waiter(job, c),
        Err(PlatformError::QueueFull),
        "the late waiter is counted: the total bound of one is reached"
    );
    assert_eq!(w.worker.stats().attach_refused_total, 1);
    assert_eq!(w.worker.capacity().restore_ops, 1, "still one job");

    assert!(w.fake.release(job), "the held load starts");
    let v = w.collect(2);
    let t = terminals(&v);
    assert_eq!(t.len(), 2, "one terminal per waiter: {v:?}");
    let mut incs: Vec<Incarnation> = t.iter().map(|x| x.incarnation).collect();
    incs.sort();
    assert_eq!(
        incs,
        vec![a, b],
        "the opener and the late waiter, once each"
    );
    assert!(t.iter().all(|x| x.job == job));
    assert!(t
        .iter()
        .all(|x| x.terminal == Some(Terminal::Done(JobStage::Completed))));
    assert!(
        t.iter().all(|x| x.holder.is_some()),
        "every waiter of a completed load gets the landing holder"
    );
    assert_eq!(w.fake.stats().reads, 1, "one read of the object");
    assert_eq!(mem.bytes(), pattern(10, PAYLOAD as usize));
    assert_eq!(w.worker.load_in_flight(&k), None);

    // The count is released with the job: a fresh load takes one waiter again.
    // Hold the second load at the store so the worker thread cannot finish and
    // reap `job2` before `attach_waiter` runs on a loaded runner.
    let mem2 = Mem::new(2, 0xEE);
    w.fake.script(JobScript::fault(Fault::Hold));
    let job2 = w
        .worker
        .submit_load_coalesced(k, &h, || mem2.dest(2).0, c)
        .unwrap();
    let LoadTicket::Started(job2) = job2 else {
        panic!("a fresh load after the first closed: {job2:?}");
    };
    wait_for("dispatch", || w.dispatched(job2));
    assert_eq!(
        w.worker.attach_waiter(job2, d),
        Ok(()),
        "the late waiter's count was released when its job closed"
    );
    assert!(w.fake.release(job2), "the second held load starts");
    let v = w.collect(2);
    assert_eq!(terminals(&v).len(), 2);
    assert_eq!(w.worker.stats().attach_refused_total, 1);
}

#[test]
fn worker_cancel_removes_only_own_waiter() {
    // Queued, three requests: the opener and one follower withdraw, the third
    // still gets its data. Every request still gets exactly one terminal, and
    // only once the shared read ended.
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let (a, b, c) = (
        Incarnation::seq(1),
        Incarnation::seq(77),
        Incarnation::seq(78),
    );
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.worker.attach_waiter(job, b).unwrap();
    w.worker.attach_waiter(job, c).unwrap();
    assert_eq!(
        w.worker.cancel(job),
        CancelOutcome::TooLate,
        "others still wait"
    );
    assert_eq!(
        w.worker.cancel_waiter(job, b),
        CancelOutcome::TooLate,
        "c still waits"
    );
    assert_eq!(
        w.worker.capacity().restore_ops,
        1,
        "the shared job is alive"
    );
    assert!(w.poll_all().is_empty(), "no terminal before the job ends");
    w.resume();
    let v = w.collect(3);
    let t = terminals(&v);
    assert_eq!(t.len(), 3, "one terminal per request: {v:?}");
    assert!(
        t.iter()
            .all(|x| x.terminal == Some(Terminal::Done(JobStage::Completed))),
        "the read ran for the request that stayed: {v:?}"
    );
    assert_eq!(mem.bytes(), pattern(7, PAYLOAD as usize));

    // Queued, everybody leaves: the real cancel, every terminal is Cancelled, no I/O.
    let w = w_with(|_| {}, |_| {});
    let k = w.inject(8);
    let (_, h) = w.chunk(8);
    let mem = Mem::new(2, 0xEE);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.worker.attach_waiter(job, b).unwrap();
    assert_eq!(w.worker.cancel(job), CancelOutcome::TooLate);
    assert_eq!(w.worker.cancel_waiter(job, b), CancelOutcome::Cancelled);
    let v = w.collect(2);
    assert!(terminals(&v)
        .iter()
        .all(|x| x.terminal == Some(Terminal::Cancelled)));
    assert_eq!(w.worker.load_in_flight(&k), None);
    assert!(w.probe.submitted().is_empty(), "never reached the store");
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK], "destination untouched");

    // Dispatched (held in the device), two requests: same rule, through the store.
    let w = w_with(|_| {}, |_| {});
    let k = w.inject(9);
    let (_, h) = w.chunk(9);
    let mem = Mem::new(2, 0xEE);
    w.fake.script(JobScript::fault(Fault::Hold));
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.worker.attach_waiter(job, b).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    assert_eq!(
        w.worker.cancel(job),
        CancelOutcome::TooLate,
        "the other waiter stays"
    );
    assert_eq!(w.fake.live_jobs().len(), 1);
    assert_eq!(w.worker.cancel_waiter(job, b), CancelOutcome::Cancelled);
    let v = w.collect(2);
    assert!(terminals(&v)
        .iter()
        .all(|x| x.terminal == Some(Terminal::Cancelled)));
    assert_eq!(w.fake.stats().reads, 0);
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK]);
}

#[test]
fn worker_stale_completion_is_dropped_by_incarnation() {
    // (a) a store that reports every completion twice still yields one terminal.
    let w = rig();
    w.fake.script(JobScript::fault(Fault::DuplicateCompletion));
    let job = w.store(1).unwrap();
    w.resume();
    let v = w.collect(1);
    wait_for("duplicates dropped", || {
        w.worker.stats().stale_completions_total >= 2
    });
    let mut v = v;
    v.extend(w.poll_all());
    let mine: Vec<_> = v.iter().filter(|c| c.job == job).collect();
    assert_eq!(terminals(&v).len(), 1, "{v:?}");
    assert_eq!(
        mine.iter().filter(|c| c.terminal.is_none()).count(),
        1,
        "one progress report"
    );
    assert_eq!(
        w.fake
            .terminal_count(w.worker.inner_job(job).unwrap_or(JobId(1))),
        2
    );

    // (b) a progress report that arrives after the cancelled terminal is dropped.
    let w = rig();
    w.fake
        .script(JobScript::fault(Fault::Hold).with(Fault::CompletionAfterCancel));
    let job = w.store(2).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    assert_eq!(w.worker.cancel(job), CancelOutcome::Cancelled);
    let mut v = w.collect(1);
    wait_for("stale progress seen", || {
        w.worker.stats().stale_completions_total >= 1
    });
    v.extend(w.poll_all());
    assert_eq!(
        v.len(),
        1,
        "only the cancelled terminal reaches the caller: {v:?}"
    );
    assert_eq!(v[0].terminal, Some(Terminal::Cancelled));

    // (c) a completion with the wrong incarnation is dropped; the real one still lands.
    let w = rig();
    w.fake.script(JobScript::fault(Fault::Hold));
    let (k, _) = w.chunk(3);
    let job = w.store(3).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    let inner = w.worker.inner_job(job).unwrap();
    w.probe.forge(Completion {
        job: inner,
        incarnation: Incarnation::seq(9999),
        key: k,
        stage: JobStage::Durable,
        terminal: Some(Terminal::Done(JobStage::Durable)),
        publish: PublishOutcome::Durable,
        bytes: PAYLOAD,
        elapsed_us: 1,
        holder: None,
    });
    wait_for("forged completion dropped", || {
        w.worker.stats().stale_completions_total >= 1
    });
    assert!(
        w.poll_all().is_empty(),
        "the forgery never reaches the caller"
    );
    assert_eq!(w.worker.capacity().park_ops, 1, "the job is still live");
    assert!(w.fake.release(inner));
    let v = w.collect(1);
    let t = terminals(&v);
    assert_eq!(t[0].job, job);
    assert_eq!(
        t[0].incarnation,
        Incarnation::seq(3),
        "the caller's own token is echoed"
    );
    assert_eq!(t[0].terminal, Some(Terminal::Done(JobStage::Durable)));
}

#[test]
fn worker_never_touches_dest_after_cancel_ack() {
    // Dispatched (held in the device) load: cancel, then the terminal, then nothing more.
    let w = rig();
    w.inject(5);
    w.fake.script(JobScript::fault(Fault::Hold));
    let released = Arc::new(AtomicU64::new(0));
    let (job, mem, handle) = w.load_with(5, released_guard(&released));
    let job = job.unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    let inner = w.worker.inner_job(job).unwrap();
    assert_eq!(w.worker.cancel(job), CancelOutcome::Cancelled);
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].terminal, Some(Terminal::Cancelled));
    wait_for("destination released after the terminal", || {
        released.load(Ordering::SeqCst) == 1
    });
    assert!(!handle.is_held());
    // Time moves on, the store is poked again: the destination is never written.
    assert!(!w.fake.release(inner));
    w.fake.advance(10_000_000);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK], "never written");
    assert_eq!(w.fake.take_poison_violations(), 0);
    assert_eq!(w.fake.stats().reads, 0);
    assert_eq!(released.load(Ordering::SeqCst), 1, "released exactly once");

    // Queued (never dispatched) load: cancelled before the store ever sees it.
    let w = rig();
    w.inject(6);
    let released = Arc::new(AtomicU64::new(0));
    let (job, mem, _h) = w.load_with(6, released_guard(&released));
    let job = job.unwrap();
    assert_eq!(w.worker.cancel(job), CancelOutcome::Cancelled);
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].terminal, Some(Terminal::Cancelled));
    w.resume();
    thread::sleep(Duration::from_millis(20));
    wait_for("queued destination released", || {
        released.load(Ordering::SeqCst) == 1
    });
    assert!(w.probe.submitted().is_empty());
    assert_eq!(w.fake.stats().io_ops, 0);
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK]);
    assert_eq!(w.fake.take_poison_violations(), 0);
}

#[test]
fn worker_backpressure_refuses_not_grows() {
    let w = rig();
    for n in 0..64 {
        w.inject(n);
    }
    let mut ok = 0;
    let mut full = 0;
    for i in 0..1000u32 {
        match w.load(i % 64) {
            Ok(_) => ok += 1,
            Err(PlatformError::QueueFull) => full += 1,
            Err(e) => panic!("{e:?}"),
        }
    }
    assert_eq!(
        (ok, full),
        (48, 952),
        "exactly the restore bound is accepted"
    );
    let s = w.worker.stats();
    assert_eq!(
        s.peak_restore_queue, 48,
        "the intake queue never exceeds the bound"
    );
    assert_eq!(s.queue_full_total, 952);
    assert_eq!(w.worker.capacity().restore_ops, 48);
    assert!(w.poll_all().is_empty());
    let mut okp = 0;
    for n in 0..100u32 {
        if w.store(n).is_ok() {
            okp += 1;
        }
    }
    let park_bound = WorkerConfig::default().max_park_ops;
    assert_eq!(okp, park_bound, "the park class is bounded on its own");
    assert_eq!(w.worker.stats().peak_park_queue, park_bound as u64);
    // Once everything ends the bound is available again.
    w.resume();
    w.settle();
    assert!(w.load(0).is_ok());
}

#[test]
fn sustained_restores_cannot_starve_parks() {
    // Finding 9 / B4: offer restores at the maximum rate for 10 000 operations;
    // parks are still admitted whenever their class has room and are
    // dispatched at no less than their reserved share.
    let w = w_with(
        |c| {
            c.quota_bytes = 1 << 40;
            c.max_objects = 1 << 20;
        },
        |c| c.dispatch_depth = 2,
    );
    for n in 1..=32 {
        w.inject(n);
    }
    let mut restores = 0u64;
    let mut restore_refusals = 0u64;
    let mut parks = 0u64;
    let mut next_park = 1000u32;
    let mut terminal_count = 0u64;
    let mut seen: HashMap<JobId, u32> = HashMap::new();
    let mut last_park_at = u64::MAX;
    while restores < 10_000 {
        for c in w.worker.poll(256) {
            if c.terminal.is_some() {
                *seen.entry(c.job).or_default() += 1;
                terminal_count += 1;
            }
        }
        match w.load(1 + (restores % 32) as u32) {
            Ok(_) => restores += 1,
            Err(PlatformError::QueueFull) => {
                restore_refusals += 1;
                if restore_refusals == 1 {
                    w.resume();
                }
                thread::yield_now();
            }
            Err(e) => panic!("{e:?}"),
        }
        if restores.is_multiple_of(4) && last_park_at != restores {
            last_park_at = restores;
            // Only this thread adds park ops, so a park class below its bound
            // must admit, however saturated the restore class is.
            let before = w.worker.capacity();
            if before.park_ops < before.max_park_ops {
                for c in w.worker.poll(256) {
                    if c.terminal.is_some() {
                        *seen.entry(c.job).or_default() += 1;
                        terminal_count += 1;
                    }
                }
                let r = w.store(next_park);
                assert!(r.is_ok(), "park refused with room in its class: {r:?}");
                next_park += 1;
                parks += 1;
            }
        }
    }
    assert!(
        restore_refusals > 0,
        "the restore class really was saturated"
    );
    assert!(parks >= 100, "parks were offered throughout ({parks})");
    let t0 = Instant::now();
    while terminal_count < restores + parks {
        for c in w.worker.poll(256) {
            if c.terminal.is_some() {
                *seen.entry(c.job).or_default() += 1;
                terminal_count += 1;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "drain stalled");
        thread::yield_now();
    }
    let s = w.worker.stats();
    assert_eq!(s.dispatched_restore, restores);
    assert_eq!(
        s.dispatched_park, parks,
        "every admitted park was dispatched"
    );
    assert!(
        s.max_restore_run_with_park_waiting <= 3,
        "a waiting park was passed over {} times in a row",
        s.max_restore_run_with_park_waiting
    );
    assert!(seen.values().all(|n| *n == 1), "one terminal per job");
    assert_eq!(seen.len() as u64, restores + parks);
}

#[test]
fn completion_queue_never_overflows_at_max_inflight() {
    // 4 restores + 2 parks: 18 completion slots. The caller never polls.
    let cfg = |c: &mut WorkerConfig| {
        c.max_restore_ops = 4;
        c.max_park_ops = 2;
        c.start_paused = false;
    };
    let w = w_with(
        |c| {
            c.max_restore_ops = 4;
            c.max_park_ops = 2;
        },
        cfg,
    );
    for n in 1..=4 {
        w.inject(n);
    }
    let mut accepted = 0;
    let mut refused_when_quiet = false;
    for i in 0..60u32 {
        // Wait for the worker to finish what it has, still without polling.
        wait_for("quiet", || {
            let c = w.worker.capacity();
            c.restore_ops == 0 && c.park_ops == 0
        });
        let r = if i % 2 == 0 {
            w.load(1 + i % 4)
        } else {
            w.store(100 + i)
        };
        match r {
            Ok(_) => accepted += 1,
            Err(PlatformError::QueueFull) => {
                refused_when_quiet = true;
                break;
            }
            Err(e) => panic!("{e:?}"),
        }
    }
    assert!(
        refused_when_quiet,
        "unpolled completions must close admission ({accepted} accepted)"
    );
    wait_for("quiet", || w.worker.capacity().restore_ops == 0);
    let s = w.worker.stats();
    assert!(s.peak_completion_queue <= 18, "{s:?}");
    assert_eq!(s.completion_overflow_total, 0);
    assert_eq!(w.fake.stats().completion_overflow_total, 0);
    // Polling gives the slots back.
    let v = w.poll_all();
    assert!(!v.is_empty());
    assert!(w.store(500).is_ok());
}

#[test]
fn coalesced_waiters_are_bounded() {
    let w = w_with(
        |_| {},
        |c| {
            c.max_waiters_per_job = 3;
            c.max_waiters_total = 4;
        },
    );
    let mut jobs = Vec::new();
    let made = Arc::new(AtomicU64::new(0));
    for n in 1..=3 {
        let k = w.inject(n);
        let (_, h) = w.chunk(n);
        let mem = Mem::new(2, 0xEE);
        let t = w
            .worker
            .submit_load_coalesced(k, &h, || mem.dest(2).0, Incarnation::seq(n as u64))
            .unwrap();
        jobs.push((k, h, t.job()));
    }
    let (a, b, c) = (jobs[0].2, jobs[1].2, jobs[2].2);
    // Per job: the opener counts as waiter 1, so 3 means two attaches.
    assert_eq!(w.worker.attach_waiter(a, Incarnation::seq(76)), Ok(()));
    assert_eq!(w.worker.attach_waiter(a, Incarnation::seq(77)), Ok(()));
    assert_eq!(
        w.worker.attach_waiter(a, Incarnation::seq(78)),
        Err(PlatformError::QueueFull)
    );
    // Coalesced submit is refused the same way, and allocates nothing.
    {
        let made2 = made.clone();
        let r = w.worker.submit_load_coalesced(
            jobs[0].0,
            &jobs[0].1,
            move || {
                made2.fetch_add(1, Ordering::SeqCst);
                Mem::new(2, 0).dest(2).0
            },
            Incarnation::seq(79),
        );
        assert_eq!(r, Err(PlatformError::QueueFull));
        assert_eq!(
            made.load(Ordering::SeqCst),
            0,
            "no landing buffer for a refused waiter"
        );
    }
    // Total: 4 attached waiters over all jobs.
    assert_eq!(w.worker.attach_waiter(b, Incarnation::seq(76)), Ok(()));
    assert_eq!(w.worker.attach_waiter(b, Incarnation::seq(77)), Ok(()));
    assert_eq!(
        w.worker.attach_waiter(c, Incarnation::seq(77)),
        Err(PlatformError::QueueFull),
        "total bound"
    );
    assert!(w.worker.stats().attach_refused_total >= 3);
    // Only loads accept waiters.
    let st = w.store(40).unwrap();
    assert!(matches!(
        w.worker.attach_waiter(st, Incarnation::seq(77)),
        Err(PlatformError::TaskRejected(_))
    ));
    assert!(matches!(
        w.worker.attach_waiter(JobId(9999), Incarnation::seq(77)),
        Err(PlatformError::TaskRejected(_))
    ));
    // When a job ends its waiters leave the total.
    w.resume();
    w.settle();
    w.worker.pause_dispatch();
    let k = w.inject(60);
    let (_, h) = w.chunk(60);
    let mem = Mem::new(2, 0xEE);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, Incarnation::seq(1))
        .unwrap()
        .job();
    assert_eq!(w.worker.attach_waiter(job, Incarnation::seq(77)), Ok(()));
    w.resume();
    w.settle();
}

#[test]
fn coalesce_table_bounds_counts_and_detach() {
    let key = |b: u8| ChunkKey(aien_platform::ObjectId([b; 32]));
    let mut t = CoalesceTable::new(CoalesceLimits {
        max_waiters_per_job: 3,
        max_waiters_total: 3,
    });
    assert_eq!(t.limits().max_waiters_per_job, 3);
    t.open(Share::Load, &key(1), JobId(1));
    t.open(Share::Load, &key(2), JobId(2));
    assert_eq!(t.waiters(JobId(1)), 1);
    assert_eq!(t.find(Share::Load, &key(1)), Some(JobId(1)));
    assert_eq!(
        t.find(Share::Store, &key(1)),
        None,
        "loads and stores never share"
    );
    assert_eq!(t.attach(JobId(1)), Ok(2));
    assert_eq!(t.attach(JobId(1)), Ok(3));
    assert_eq!(t.attach(JobId(1)), Err(AttachRefused::JobFull));
    assert_eq!(t.attach(JobId(2)), Ok(2));
    assert_eq!(t.attached_total(), 3);
    assert_eq!(t.attach(JobId(2)), Err(AttachRefused::TotalFull));
    assert_eq!(t.attach(JobId(9)), Err(AttachRefused::NoSuchJob));
    // Detach frees one slot at a time and reports the last waiter.
    assert_eq!(t.detach(JobId(1)), Detach::Remaining(2));
    assert_eq!(t.attached_total(), 2);
    assert_eq!(t.attach(JobId(2)), Ok(3));
    assert_eq!(t.detach(JobId(2)), Detach::Remaining(2));
    assert_eq!(t.detach(JobId(2)), Detach::Remaining(1));
    assert_eq!(t.detach(JobId(2)), Detach::Last);
    assert_eq!(t.waiters(JobId(2)), 0);
    assert_eq!(t.find(Share::Load, &key(2)), None, "nobody left to join");
    assert!(
        t.has(Share::Load, &key(2)),
        "but the job still exists until it ends"
    );
    assert_eq!(t.attach(JobId(2)), Err(AttachRefused::NoSuchJob));
    assert_eq!(t.detach(JobId(77)), Detach::Unknown);
    // Close releases the job's attached waiters.
    assert_eq!(t.attached_total(), 1);
    t.close(JobId(1));
    assert_eq!(t.attached_total(), 0);
    t.close(JobId(2));
    assert_eq!(t.jobs(), 0);
    assert!(!t.has(Share::Load, &key(1)));
    // A second job for a key that has waiters does not take over the key.
    t.open(Share::Store, &key(5), JobId(10));
    t.open(Share::Store, &key(5), JobId(11));
    assert_eq!(t.find(Share::Store, &key(5)), Some(JobId(10)));
    t.close(JobId(11));
    assert_eq!(t.find(Share::Store, &key(5)), Some(JobId(10)));
    t.close(JobId(10));
    assert_eq!(t.find(Share::Store, &key(5)), None);
}

#[test]
fn shutdown_with_stalled_fsync_returns_by_deadline_and_reports_unfinished() {
    // Finding 11: fsync has no bounded completion time; shutdown still returns
    // at its deadline and names the job, keeping its buffer.
    let w = w_with(|_| {}, |c| c.dispatch_depth = 2);
    w.fake
        .script(JobScript::latency(100, 100).with(Fault::StallFsync(10_000_000)));
    w.fake.script(JobScript::fault(Fault::Hold));
    let rel1 = Arc::new(AtomicU64::new(0));
    let (j1, h1, _m1) = w.store_with(1, released_guard(&rel1));
    let j1 = j1.unwrap();
    let j2 = w.store(2).unwrap();
    let j3 = w.store(3).unwrap(); // behind the depth: never dispatched
    let (k1, _) = w.chunk(1);
    w.resume();
    wait_for("dispatch", || w.dispatched(j1) && w.dispatched(j2));
    wait_for("fsync under way", || w.fake.now_us() >= 400);
    w.step.store(0, Ordering::SeqCst); // the stall is real: time stops for the store

    let t0 = Instant::now();
    let rep = w.worker.shutdown(60);
    let el = t0.elapsed();
    assert!(
        el >= Duration::from_millis(50),
        "waited for the deadline: {el:?}"
    );
    assert!(el < Duration::from_secs(5), "and returned by it: {el:?}");
    assert_eq!(rep.unfinished.len(), 1, "{rep:?}");
    assert_eq!(rep.unfinished[0].job, j1);
    assert_eq!(rep.unfinished[0].key, k1);
    assert_eq!(rep.unfinished[0].bytes, OBJ);
    assert_eq!(rep.cancelled, 2, "the held job and the queued job: {rep:?}");
    assert_eq!(rep.published, 0);
    let s = w.worker.stats();
    assert_eq!(
        (s.shutdown_unfinished_jobs, s.shutdown_unfinished_bytes),
        (1, OBJ)
    );
    assert!(
        h1.is_held() && rel1.load(Ordering::SeqCst) == 0,
        "buffer kept"
    );
    assert_eq!(
        w.store(9),
        Err(PlatformError::DeviceUnavailable),
        "intake closed"
    );

    // The job keeps running after the deadline and ends normally.
    w.step.store(1_000_000_000, Ordering::SeqCst);
    let mut v = w.poll_all();
    let t0 = Instant::now();
    while terminals(&v).len() < 3 {
        v.extend(w.worker.poll(16));
        assert!(t0.elapsed() < Duration::from_secs(20));
        thread::yield_now();
    }
    let by_job: HashMap<JobId, &Completion> =
        terminals(&v).into_iter().map(|c| (c.job, c)).collect();
    assert_eq!(by_job.len(), 3, "exactly one terminal per job: {v:?}");
    assert_eq!(
        by_job[&j1].terminal,
        Some(Terminal::Done(JobStage::Durable))
    );
    assert_eq!(by_job[&j2].terminal, Some(Terminal::Cancelled));
    assert_eq!(by_job[&j3].terminal, Some(Terminal::Cancelled));
    wait_for("buffer released with the job", || {
        rel1.load(Ordering::SeqCst) == 1
    });
    assert!(!h1.is_held());
}

#[test]
fn worker_shutdown_lets_jobs_past_fsync_publish() {
    let w = rig();
    w.fake.script(JobScript::latency(100, 50_000_000));
    let j = w.store(1).unwrap();
    w.resume();
    wait_for("file synced", || w.fake.now_us() >= 300);
    w.step.store(10_000_000, Ordering::SeqCst);
    let rep = w.worker.shutdown(2000);
    assert_eq!(rep.published, 1, "{rep:?}");
    assert!(rep.unfinished.is_empty());
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].job, j);
    assert_eq!(
        terminals(&v)[0].terminal,
        Some(Terminal::Done(JobStage::Durable))
    );
}

#[test]
fn worker_shutdown_keeps_buffers_until_each_job_terminal() {
    // The store is stalled in real time (the worker thread is stuck inside a
    // call). Shutdown returns at its deadline; the buffers of the jobs that
    // are still in the device are released only when each reaches its terminal.
    let w = rig();
    w.inject(2);
    w.fake.script(JobScript::fault(Fault::Hold));
    w.fake.script(JobScript::fault(Fault::Hold));
    let rel_s = Arc::new(AtomicU64::new(0));
    let rel_l = Arc::new(AtomicU64::new(0));
    let (js, hs, _ms) = w.store_with(1, released_guard(&rel_s));
    let (jl, _ml, hl) = w.load_with(2, released_guard(&rel_l));
    let (js, jl) = (js.unwrap(), jl.unwrap());
    w.resume();
    wait_for("both in the device", || {
        w.dispatched(js) && w.dispatched(jl)
    });
    w.stall_store();

    let t0 = Instant::now();
    let rep = w.worker.shutdown(80);
    let el = t0.elapsed();
    assert!(
        el >= Duration::from_millis(70) && el < Duration::from_secs(5),
        "{el:?}"
    );
    let mut un: Vec<JobId> = rep.unfinished.iter().map(|u| u.job).collect();
    un.sort();
    assert_eq!(un, vec![js, jl], "{rep:?}");
    assert!(
        hs.is_held() && hl.is_held(),
        "buffers kept past the deadline"
    );
    assert_eq!(
        rel_s.load(Ordering::SeqCst) + rel_l.load(Ordering::SeqCst),
        0
    );
    assert!(
        w.worker.state_lock_is_free(),
        "no lock is held while the store stalls"
    );

    w.unstall_store();
    let v = w.collect(2);
    assert!(terminals(&v)
        .iter()
        .all(|c| c.terminal == Some(Terminal::Cancelled)));
    wait_for("released with the terminals", || {
        rel_s.load(Ordering::SeqCst) == 1 && rel_l.load(Ordering::SeqCst) == 1
    });
    assert!(!hs.is_held() && !hl.is_held());
}

#[test]
fn worker_shutdown_cancels_queued_and_closes_intake() {
    let w = rig();
    for n in 1..=3 {
        w.inject(n);
    }
    let rels: Vec<Arc<AtomicU64>> = (0..5).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let mut jobs = Vec::new();
    for n in 1..=3u32 {
        jobs.push(
            w.load_with(n, released_guard(&rels[n as usize - 1]))
                .0
                .unwrap(),
        );
    }
    for n in 10..=11u32 {
        jobs.push(
            w.store_with(n, released_guard(&rels[(n - 10) as usize + 3]))
                .0
                .unwrap(),
        );
    }
    let t0 = Instant::now();
    let rep = w.worker.shutdown(500);
    assert!(
        t0.elapsed() < Duration::from_millis(400),
        "nothing in flight: no waiting"
    );
    assert_eq!(rep.cancelled, 5);
    assert!(rep.unfinished.is_empty());
    assert_eq!(rep.published, 0);
    for r in &rels {
        assert_eq!(r.load(Ordering::SeqCst), 1, "queued buffers released once");
    }
    let v = w.poll_all();
    assert_eq!(terminals(&v).len(), 5);
    assert!(terminals(&v)
        .iter()
        .all(|c| c.terminal == Some(Terminal::Cancelled)));
    assert_eq!(w.load(1), Err(PlatformError::DeviceUnavailable));
    assert_eq!(
        w.worker.submit_flush(Incarnation::seq(1)),
        Err(PlatformError::DeviceUnavailable)
    );
    // A second call reports the same facts.
    let rep2 = w.worker.shutdown(10);
    assert_eq!(rep2.cancelled, 5);
    assert!(w.probe.submitted().is_empty(), "the store never saw them");
}

#[test]
fn restore_txn_holder_survives_sibling_failure_until_all_jobs_terminal() {
    // Findings 5 and B3: chunk A fails while B and C are mid-transfer. The
    // destinations of B and C stay held until each has produced its terminal.
    let w = rig();
    let ka = w.inject(1);
    w.inject(2);
    w.inject(3);
    assert!(w.fake.corrupt_payload(&ka));
    w.fake.script(JobScript::default()); // A: runs
    w.fake.script(JobScript::fault(Fault::Hold)); // B: mid-transfer
    w.fake.script(JobScript::fault(Fault::Hold)); // C: mid-transfer
    let rel: Vec<Arc<AtomicU64>> = (0..3).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let (ja, ma, _ha) = w.load_with(1, released_guard(&rel[0]));
    let (jb, mb, hb) = w.load_with(2, released_guard(&rel[1]));
    let (jc, mc, hc) = w.load_with(3, released_guard(&rel[2]));
    let (ja, jb, jc) = (ja.unwrap(), jb.unwrap(), jc.unwrap());
    w.resume();
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].job, ja);
    assert!(matches!(
        terminals(&v)[0].terminal,
        Some(Terminal::Failed(PlatformError::InvalidLayout))
    ));
    wait_for("A's buffer released", || rel[0].load(Ordering::SeqCst) == 1);
    assert_eq!(
        ma.bytes(),
        vec![0xEE; 2 * BLOCK],
        "a failed chunk wrote nothing"
    );

    // The txn is Draining. Freeze the device so nothing is reaped, then cancel the siblings.
    wait_for("B and C in the device", || {
        w.dispatched(jb) && w.dispatched(jc)
    });
    w.stall_store();
    assert_eq!(w.worker.cancel(jb), CancelOutcome::Cancelled);
    assert_eq!(w.worker.cancel(jc), CancelOutcome::Cancelled);
    assert!(
        hb.is_held() && hc.is_held(),
        "cancel acknowledged, the device is not done"
    );
    assert_eq!(
        rel[1].load(Ordering::SeqCst) + rel[2].load(Ordering::SeqCst),
        0
    );
    assert!(w.poll_all().is_empty(), "no terminal yet for B or C");

    w.unstall_store();
    let v = w.collect(2);
    let mut got: Vec<JobId> = terminals(&v).iter().map(|c| c.job).collect();
    got.sort();
    assert_eq!(got, vec![jb, jc]);
    assert!(terminals(&v)
        .iter()
        .all(|c| c.terminal == Some(Terminal::Cancelled)));
    wait_for("B and C released after their terminals", || {
        rel[1].load(Ordering::SeqCst) == 1 && rel[2].load(Ordering::SeqCst) == 1
    });
    assert_eq!(mb.bytes(), vec![0xEE; 2 * BLOCK]);
    assert_eq!(mc.bytes(), vec![0xEE; 2 * BLOCK]);
    assert_eq!(w.fake.take_poison_violations(), 0);
}

// ----------------------------------------------- the hard requirements

/// Run `f` on its own thread; panic if it does not finish within `secs`.
fn within<R: Send + 'static>(secs: u64, what: &str, f: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("{what} blocked on the stalled store"))
}

#[test]
fn worker_submit_and_poll_never_block_on_a_stalled_store() {
    let w = rig();
    for n in 1..=4 {
        w.inject(n);
    }
    let first = w.load(1).unwrap();
    // The worker thread gets stuck inside the store's submit of the first job.
    w.stall_store();
    w.resume();
    wait_for("worker stuck mid-submit", || w.probe.gate.waiting() >= 1);
    // wait until the job was actually handed over (it is `Dispatching` now)
    wait_for("job taken from the queue", || {
        w.worker.dispatch_order().len() == 1
    });

    let worker = Arc::new(w.worker);
    let probe = w.probe.clone();
    let t0 = Instant::now();
    let outcome = {
        let worker = worker.clone();
        let fields = w.fields;
        let ns = w.ns;
        within(20, "submit/poll/cancel/capacity/lookup", move || {
            let mut accepted = Vec::new();
            for i in 0..200u32 {
                let tokens: Vec<u32> = (0..8).map(|j| (2000 + i) * 1000 + j).collect();
                let key = aien_kv_store::chunk_chain(&ns, 8, &tokens).unwrap()[0];
                let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
                let mem = Mem::new(2, 1);
                let (src, _) = mem.source(2, 4);
                if let Ok(j) = worker.submit_store(h, src, Incarnation::seq(i as u64)) {
                    accepted.push(j);
                }
                let _ = worker.poll(8);
                let _ = worker.capacity();
                let _ = worker.lookup(&[key]);
                let _ = worker.stats();
                if i % 3 == 0 {
                    if let Some(j) = accepted.pop() {
                        let _ = worker.cancel(j);
                    }
                }
            }
            // The job the worker is stuck on can be cancelled and polled too.
            let c = worker.cancel(first);
            assert_eq!(
                c,
                CancelOutcome::TooLate,
                "mid-dispatch: the answer is the safe one"
            );
            (accepted.len(), worker.state_lock_is_free())
        })
    };
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    assert!(
        outcome.0 > 0 && outcome.1,
        "no lock held while the store stalls"
    );
    assert!(probe.gate.waiting() >= 1, "the worker was stuck all along");

    // Let the store go: every accepted job ends exactly once.
    probe.gate.set(true);
    let t0 = Instant::now();
    let mut all = Vec::new();
    loop {
        all.extend(worker.poll(64));
        let c = worker.capacity();
        if c.restore_ops == 0 && c.park_ops == 0 {
            all.extend(worker.poll(64));
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(30));
        thread::yield_now();
    }
    let mut per_job: HashMap<JobId, u32> = HashMap::new();
    for c in terminals(&all) {
        *per_job.entry(c.job).or_default() += 1;
    }
    assert!(per_job.values().all(|n| *n == 1));
    assert!(per_job.contains_key(&first));
    assert_eq!(w.fake.take_poison_violations(), 0);
}

#[test]
fn worker_submit_latency_is_independent_of_store_latency() {
    let w = rig();
    for n in 1..=8 {
        w.inject(n);
    }
    // Deterministic (Grok C5 P2-5): the store is stuck for the whole run, so
    // a caller that touched it would hang here instead of being merely slow.
    w.resume();
    w.stall_store();
    let (tx, rx) = mpsc::channel();
    thread::scope(|sc| {
        sc.spawn(|| {
            for i in 0..300u32 {
                let _ = w.load(1 + i % 8);
                let _ = w.store(100 + i);
                let _ = w.worker.poll(8);
                let _ = w.worker.capacity();
            }
            let _ = tx.send(());
        });
        let finished = rx.recv_timeout(Duration::from_secs(30)).is_ok();
        let stuck_calls = w.probe.submitted().len();
        w.unstall_store();
        assert!(finished, "a caller blocked on the stalled store");
        assert!(
            stuck_calls <= 1,
            "the store really was stuck: {stuck_calls} submits"
        );
    });
    w.settle();
}

#[test]
fn worker_inner_io_runs_only_on_the_worker_thread() {
    let w = rig();
    w.inject(1);
    let main = thread::current().id();
    let _ = w.load(1).unwrap();
    let _ = w.store(2).unwrap();
    let _ = w.worker.submit_flush(Incarnation::seq(3)).unwrap();
    w.resume();
    w.collect(3);
    let _ = w.worker.shutdown(50);
    for what in ["submit", "poll", "drive", "shutdown"] {
        let t = w.probe.threads_of(what);
        assert_eq!(t.len(), 1, "{what}: {t:?}");
        assert!(!t.contains(&main), "{what} ran on the caller's thread");
    }
    let worker_thread = *w.probe.threads_of("submit").iter().next().unwrap();
    assert_eq!(w.probe.threads_of("drive"), HashSet::from([worker_thread]));
}

#[test]
fn worker_eviction_happens_on_worker_thread_not_in_submit() {
    // Finding 7: the quota holds exactly three objects and they are all there.
    let w = w_with(|c| c.quota_bytes = 3 * OBJ, |_| {});
    for n in 1..=3 {
        w.inject(n);
    }
    let before = w.fake.capacity();
    assert_eq!(before.used_bytes, 3 * OBJ);
    let job = w.store(10).unwrap();
    let io = w.fake.stats();
    assert_eq!(
        (io.unlinks, io.io_ops, io.evicted_bytes_total),
        (0, 0, 0),
        "submit evicts nothing"
    );
    assert_eq!(w.fake.capacity().used_bytes, 3 * OBJ);
    assert_eq!(w.worker.capacity().reserved_bytes, OBJ);
    w.resume();
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].job, job);
    assert_eq!(
        terminals(&v)[0].terminal,
        Some(Terminal::Done(JobStage::Durable))
    );
    let io = w.fake.stats();
    assert!(io.unlinks >= 1 && io.evicted_bytes_total >= OBJ, "{io:?}");
    let main = thread::current().id();
    assert!(!w.probe.threads_of("drive").contains(&main));
    assert_eq!(w.worker.capacity().inflight_bytes, 0);
}

#[test]
fn worker_delete_skips_keys_with_jobs_and_unlinks_on_the_worker() {
    let w = rig();
    let k1 = w.inject(1);
    let k2 = w.inject(2);
    let job = w.load(1).unwrap(); // queued: k1 has a job
    assert_eq!(
        w.worker.delete(&[k1]),
        0,
        "a key with a queued load is not deleted"
    );
    assert_eq!(w.worker.delete(&[k2]), 1);
    assert_eq!(w.worker.lookup(&[k2]), vec![None], "marked at once");
    assert_eq!(
        w.fake.stats().unlinks,
        0,
        "the unlink is not done by the caller"
    );
    w.resume();
    w.collect(1);
    let _ = job;
    wait_for("unlink done by the worker", || w.fake.stats().unlinks == 1);
    assert_eq!(w.worker.lookup(&[k1]), vec![Some(PAYLOAD)]);
}

#[test]
fn worker_every_job_yields_exactly_one_terminal_completion() {
    let w = w_with(
        |c| {
            c.quota_bytes = 1 << 40;
            c.max_objects = 1 << 20;
        },
        |c| {
            c.start_paused = false;
            c.dispatch_depth = 3;
        },
    );
    for n in 1..=16 {
        w.inject(n);
    }
    // A mixed bag of misbehaviours, consumed in dispatch order.
    for i in 0..400 {
        let s = match i % 9 {
            0 => JobScript::fault(Fault::DiskFull),
            1 => JobScript::fault(Fault::TornWrite),
            2 => JobScript::fault(Fault::DuplicateCompletion),
            3 => JobScript::fault(Fault::CompletionAfterCancel),
            4 => JobScript::latency(300, 300).with(Fault::StallFsync(500)),
            5 => JobScript::fault(Fault::QueueFull),
            _ => JobScript::default(),
        };
        w.fake.script(s);
    }
    let worker = Arc::new(w.worker);
    let accepted: Arc<Mutex<BTreeSet<(JobId, Incarnation)>>> =
        Arc::new(Mutex::new(BTreeSet::new()));
    let done = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    // Each submitter keeps going until it has ACCEPT_PER_THREAD accepted jobs,
    // so the size of the workload does not depend on how fast the dispatcher
    // drains on this host (sc#575). Bounded by attempts and wall time.
    const ACCEPT_PER_THREAD: usize = 60;
    const MAX_ATTEMPTS: u32 = 999;
    let deadline = Instant::now() + Duration::from_secs(30);
    for t in 0..4u32 {
        let worker = worker.clone();
        let accepted = accepted.clone();
        let fields = w.fields;
        let ns = w.ns;
        handles.push(thread::spawn(move || {
            let mut mine = 0usize;
            for i in 0..MAX_ATTEMPTS {
                if mine >= ACCEPT_PER_THREAD || Instant::now() >= deadline {
                    break;
                }
                let n = t * 1000 + i;
                let tokens: Vec<u32> = (0..8).map(|j| (5000 + n % 40) * 1000 + j).collect();
                let key = aien_kv_store::chunk_chain(&ns, 8, &tokens).unwrap()[0];
                let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
                let r = match i % 5 {
                    0 | 1 => {
                        let mem = Mem::new(2, 3);
                        let (src, _) = mem.source(2, 4);
                        worker.submit_store(h, src, Incarnation::seq(n as u64))
                    }
                    2 | 3 => {
                        let k = 1 + (n % 16);
                        let tokens: Vec<u32> = (0..8).map(|j| k * 1000 + j).collect();
                        let key = aien_kv_store::chunk_chain(&ns, 8, &tokens).unwrap()[0];
                        let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
                        let mem = Mem::new(2, 0xEE);
                        let dst = mem.dest(2).0;
                        worker.submit_load(key, &h, dst, Incarnation::seq(n as u64))
                    }
                    _ => worker.submit_flush(Incarnation::seq(n as u64)),
                };
                match r {
                    Ok(j) => {
                        mine += 1;
                        accepted
                            .lock()
                            .unwrap()
                            .insert((j, Incarnation::seq(n as u64)));
                        if i % 7 == 0 {
                            let _ = worker.cancel(j);
                        }
                    }
                    Err(PlatformError::QueueFull)
                    | Err(PlatformError::DeviceUnavailable)
                    | Err(PlatformError::OutOfMemory) => thread::sleep(Duration::from_micros(200)),
                    Err(e) => panic!("{e:?}"),
                }
            }
        }));
    }
    let collector = {
        let worker = worker.clone();
        let done = done.clone();
        thread::spawn(move || {
            let mut per_job: HashMap<(JobId, Incarnation), u32> = HashMap::new();
            let mut last = Instant::now();
            loop {
                let v = worker.poll(64);
                if v.is_empty() {
                    if done.load(Ordering::SeqCst) {
                        let c = worker.capacity();
                        if c.restore_ops == 0 && c.park_ops == 0 {
                            for c in worker.poll(1024) {
                                if c.terminal.is_some() {
                                    *per_job.entry((c.job, c.incarnation)).or_default() += 1;
                                }
                            }
                            return per_job;
                        }
                    }
                    assert!(last.elapsed() < Duration::from_secs(60), "stalled");
                    thread::yield_now();
                    continue;
                }
                last = Instant::now();
                for c in v {
                    if c.terminal.is_some() {
                        *per_job.entry((c.job, c.incarnation)).or_default() += 1;
                    }
                }
            }
        })
    };
    for h in handles {
        h.join().unwrap();
    }
    done.store(true, Ordering::SeqCst);
    let per_job = collector.join().unwrap();
    let accepted = accepted.lock().unwrap();
    assert!(
        accepted.len() >= 4 * ACCEPT_PER_THREAD,
        "a real workload ran ({} accepted, want {})",
        accepted.len(),
        4 * ACCEPT_PER_THREAD
    );
    let got: BTreeSet<(JobId, Incarnation)> = per_job.keys().copied().collect();
    assert_eq!(
        &got, &*accepted,
        "a terminal for every accepted request (job, incarnation) and for nothing else"
    );
    assert!(
        per_job.values().all(|n| *n == 1),
        "exactly one each: {per_job:?}"
    );
    let s = worker.stats();
    assert_eq!(s.completion_overflow_total, 0);
    assert_eq!(
        s.terminal_done + s.terminal_failed + s.terminal_cancelled,
        accepted
            .iter()
            .map(|(j, _)| *j)
            .collect::<BTreeSet<_>>()
            .len() as u64
    );
    let c = worker.capacity();
    assert_eq!((c.restore_ops, c.park_ops, c.inflight_bytes), (0, 0, 0));
    assert!(w.fake.live_jobs().is_empty());
    assert_eq!(w.fake.take_poison_violations(), 0);
}

#[test]
fn worker_refusal_by_the_store_becomes_one_failed_terminal() {
    let w = rig();
    w.fake.script(JobScript::fault(Fault::QueueFull));
    let rel = Arc::new(AtomicU64::new(0));
    let (job, _h, _m) = w.store_with(1, released_guard(&rel));
    let job = job.unwrap();
    assert_eq!(w.worker.capacity().park_ops, 1);
    w.resume();
    let v = w.collect(1);
    let t = terminals(&v);
    assert_eq!((t.len(), t[0].job), (1, job));
    assert_eq!(
        t[0].terminal,
        Some(Terminal::Failed(PlatformError::QueueFull))
    );
    assert_eq!(w.worker.stats().inner_refusals_total, 1);
    let c = w.worker.capacity();
    assert_eq!((c.park_ops, c.inflight_bytes, c.reserved_bytes), (0, 0, 0));
    wait_for("source released", || rel.load(Ordering::SeqCst) == 1);
}

#[test]
fn worker_quota_refusal_is_a_failed_terminal_not_a_submit_error() {
    // DOCS SILENT on where the quota check lives when a worker fronts a store:
    // the worker cannot know what is evictable, so the store decides at dispatch.
    let w = w_with(|c| c.quota_bytes = OBJ / 2, |_| {});
    let job = w.store(1).expect("accepted by the worker");
    w.resume();
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].job, job);
    assert_eq!(
        terminals(&v)[0].terminal,
        Some(Terminal::Failed(PlatformError::OutOfMemory))
    );
    assert_eq!(w.worker.capacity().reserved_bytes, 0);
}

#[test]
fn worker_store_dedup_makes_every_owner_cancel_safe() {
    let w = rig();
    let a = w.store(1).unwrap();
    // A second owner (another incarnation) storing the same key joins the job.
    let (_, h) = w.chunk(1);
    let mem = Mem::new(2, 0);
    let (src, _hold) = mem.source(2, 4);
    let second = Incarnation::seq(901);
    let b = w.worker.submit_store(h, src, second).unwrap();
    assert_eq!(
        a, b,
        "a store of a key already in flight returns the existing job"
    );
    // The same incarnation resubmitting is idempotent: same job, no new waiter.
    assert_eq!(w.store(1).unwrap(), a);
    assert_eq!(w.worker.capacity().park_ops, 1);
    assert_eq!(w.worker.stats().store_dedup_total, 2);
    assert_eq!(
        w.worker.cancel(a),
        CancelOutcome::TooLate,
        "the other owner still wants it"
    );
    w.resume();
    let v = w.collect(2);
    let t = terminals(&v);
    assert_eq!(t.len(), 2, "one terminal per owner: {v:?}");
    assert!(t
        .iter()
        .all(|x| x.terminal == Some(Terminal::Done(JobStage::Durable))));
    assert!(t.iter().any(|x| x.incarnation == second));
    assert!(w.fake.payload_of(&w.key(1)).is_some());
}

#[test]
fn worker_flush_is_a_zero_byte_park_job() {
    let w = rig();
    let f = w.worker.submit_flush(Incarnation::seq(5)).unwrap();
    let cap = w.worker.capacity();
    assert_eq!((cap.park_ops, cap.inflight_bytes), (1, 0));
    w.resume();
    let v = w.collect(1);
    assert_eq!(terminals(&v)[0].job, f);
    assert_eq!(
        terminals(&v)[0].terminal,
        Some(Terminal::Done(JobStage::Durable))
    );
    assert_eq!(w.worker.stats().dispatched_park, 1);
}

#[test]
fn worker_submit_refused_while_the_store_is_recovering() {
    let w = w_with(|c| c.start_recovering = true, |_| {});
    w.inject(1);
    assert_eq!(w.store(2), Err(PlatformError::QueueFull));
    assert_eq!(w.load(1), Err(PlatformError::QueueFull));
    let p = w.worker.recover_step(10);
    assert!(p.done);
    assert!(w.store(2).is_ok(), "open again once the walk is complete");
}

#[test]
fn worker_load_of_unknown_key_is_object_not_found() {
    let w = rig();
    let (k, _) = w.chunk(77);
    assert_eq!(w.load(77), Err(PlatformError::ObjectNotFound(k.0)));
    assert_eq!(w.worker.capacity().restore_ops, 0);
}

#[test]
fn worker_rejects_header_and_source_that_disagree() {
    let w = rig();
    let (_, mut h) = w.chunk(1);
    h.payload_len += 1;
    let mem = Mem::new(2, 0);
    let (src, _) = mem.source(2, 4);
    assert_eq!(
        w.worker.submit_store(h, src, Incarnation::seq(1)),
        Err(PlatformError::InvalidLayout)
    );
    assert_eq!(w.worker.capacity().park_ops, 0);
}

#[test]
fn dropping_the_worker_does_not_wait_for_a_stalled_store() {
    let w = rig();
    let rel = Arc::new(AtomicU64::new(0));
    w.fake.script(JobScript::fault(Fault::Hold));
    let (j, _h, _m) = w.store_with(1, released_guard(&rel));
    let j = j.unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(j));
    w.stall_store();
    let probe = w.probe.clone();
    let W {
        worker,
        probe: rig_probe,
        ..
    } = w;
    drop(rig_probe);
    within(10, "drop of the worker", move || drop(worker));
    assert_eq!(
        rel.load(Ordering::SeqCst),
        0,
        "the buffer is still the store's"
    );
    // The detached thread finishes when the store lets go, then lets go of the store.
    probe.gate.set(true);
    wait_for("buffer released once the job ended", || {
        rel.load(Ordering::SeqCst) == 1
    });
    wait_for("worker thread gone", || Arc::strong_count(&probe) == 1);
}

#[test]
fn worker_panic_in_the_store_fails_live_jobs_once() {
    let boom = Arc::new(AtomicBool::new(false));
    let fields = tiny();
    let ns = fields.namespace_id().unwrap();
    let fake = Arc::new(FakeKvStore::new(FakeConfig::new(ns)));
    let b = boom.clone();
    let f2 = fake.clone();
    let worker = StorageWorker::spawn(
        fake.clone(),
        WorkerConfig {
            idle_poll: Duration::ZERO,
            ..WorkerConfig::default()
        },
        move || {
            if b.load(Ordering::SeqCst) {
                panic!("injected store failure");
            }
            f2.advance(0);
        },
    )
    .unwrap();
    fake.script(JobScript::fault(Fault::Hold));
    let mem = Mem::new(2, 1);
    let (src, _) = mem.source(2, 4);
    let tokens: Vec<u32> = (0..8).collect();
    let key = aien_kv_store::chunk_chain(&ns, 8, &tokens).unwrap()[0];
    let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
    let job = worker.submit_store(h, src, Incarnation::seq(1)).unwrap();
    wait_for("dispatch", || worker.inner_job(job).is_some());
    let inner = worker.inner_job(job).unwrap();
    boom.store(true, Ordering::SeqCst);
    wait_for("the worker failed", || {
        worker.capacity().state == aien_kv_store::StoreState::Failed
    });
    // The store still holds the source (Hold): the job may fail only once
    // the store lets go (Grok C5 P1-3).
    assert_eq!(fake.cancel(inner), CancelOutcome::Cancelled);
    fake.advance(0);
    let mut v = Vec::new();
    wait_for("terminal after the panic", || {
        v.extend(worker.poll(8));
        !terminals(&v).is_empty()
    });
    assert_eq!(terminals(&v).len(), 1);
    assert!(matches!(
        terminals(&v)[0].terminal,
        Some(Terminal::Failed(PlatformError::HardwareFault(_)))
    ));
    assert_eq!(worker.capacity().park_ops, 0);
    let mem = Mem::new(2, 1);
    let (src, _) = mem.source(2, 4);
    let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
    assert_eq!(
        worker.submit_store(h, src, Incarnation::seq(2)),
        Err(PlatformError::DeviceUnavailable)
    );
}

#[test]
fn worker_config_reads_the_documented_settings() {
    use std::collections::HashMap;
    let env: HashMap<&str, &str> = HashMap::from([
        ("AIEN_KV_STORE_MAX_RESTORE_OPS", "12"),
        ("AIEN_KV_STORE_MAX_PARK_OPS", "5"),
        ("AIEN_KV_STORE_MAX_WAITERS_PER_JOB", "8"),
        ("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", "1048576"),
        ("AIEN_KV_STORE_DRAIN_MS", "250"),
    ]);
    let c = WorkerConfig::from_lookup(|n| env.get(n).map(|s| s.to_string())).unwrap();
    assert_eq!(
        (c.max_restore_ops, c.max_park_ops, c.max_waiters_per_job),
        (12, 5, 8)
    );
    assert_eq!((c.max_inflight_bytes, c.drain_ms), (1 << 20, 250));
    // Unset: the documented defaults.
    let d = WorkerConfig::from_lookup(|_| None).unwrap();
    assert_eq!(
        (
            d.max_restore_ops,
            d.max_park_ops,
            d.max_inflight_bytes,
            d.drain_ms
        ),
        (48, 64, 256 << 20, 2000)
    );
    // Garbage is refused, naming the setting; it is not silently replaced.
    for bad in ["abc", "-1", "0", "", "1.5"] {
        let e = WorkerConfig::from_lookup(|n| {
            (n == "AIEN_KV_STORE_MAX_PARK_OPS").then(|| bad.to_string())
        })
        .unwrap_err();
        assert_eq!(e.name, "AIEN_KV_STORE_MAX_PARK_OPS", "{bad:?}");
    }
    assert!(
        WorkerConfig::from_lookup(|n| { (n == "AIEN_KV_STORE_DRAIN_MS").then(|| "0".to_string()) })
            .is_ok(),
        "a zero drain deadline is allowed"
    );
}

// ------------------------------------------------- incarnation (ADR 5.3, P1)

#[test]
fn incarnation_keeps_the_full_64_bit_sequence_id_and_the_epoch() {
    // The scheduler's sequence id is slot << 32 | generation: all 64 bits are used.
    let id = (7u64 << 32) | 9;
    let a = Incarnation::new(id, 1);
    assert_eq!(a.seq_id, id);
    assert_eq!(a.epoch, 1);
    assert_ne!(a, Incarnation::new(id, 2), "same sequence, later epoch");
    assert_ne!(a, Incarnation::new(id & 0xFFFF_FFFF, 1), "high bits matter");
    assert_ne!(a, Incarnation::new(id ^ (1 << 63), 1));
}

#[test]
fn stale_completion_with_reused_seq_id_but_new_epoch_is_rejected() {
    // The job belongs to (seq 3, epoch 1). A completion for (seq 3, epoch 2) is a
    // different residency of the same sequence id: it must never be committed.
    let w = rig();
    w.fake.script(JobScript::fault(Fault::Hold));
    let (k, h) = w.chunk(3);
    let (src_mem, inc) = (Mem::new(2, 0), Incarnation::new(3, 1));
    let p = pattern(3, PAYLOAD as usize);
    src_mem.fill_with(|i| p[i]);
    let (src, _handle) = src_mem.source(2, 4);
    let job = w.worker.submit_store(h, src, inc).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(job));
    let inner = w.worker.inner_job(job).unwrap();
    w.probe.forge(Completion {
        job: inner,
        incarnation: Incarnation::new(3, 2),
        key: k,
        stage: JobStage::Durable,
        terminal: Some(Terminal::Done(JobStage::Durable)),
        publish: PublishOutcome::Durable,
        bytes: PAYLOAD,
        elapsed_us: 1,
        holder: None,
    });
    wait_for("forged completion dropped", || {
        w.worker.stats().stale_completions_total >= 1
    });
    assert!(
        w.poll_all().is_empty(),
        "the other epoch's completion is dropped"
    );
    assert_eq!(w.worker.capacity().park_ops, 1, "the job is still live");
    assert!(w.fake.release(inner));
    let v = w.collect(1);
    assert_eq!(
        terminals(&v)[0].incarnation,
        inc,
        "the caller's own token comes back"
    );
}

#[test]
fn coalesced_waiters_each_get_a_terminal_with_their_own_incarnation() {
    // Three requests, two of them from the same sequence id in different epochs,
    // plus one trait-level attach. One I/O, four terminals, each carrying its
    // own waiter's token: the completion never exposes one token for all.
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let leader = Incarnation::new(10, 1);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, leader)
        .unwrap()
        .job();
    let followers = [Incarnation::new(10, 2), Incarnation::new(11, 1)];
    for inc in followers {
        let t = w
            .worker
            .submit_load_coalesced(k, &h, || panic!("a follower needs no buffer"), inc)
            .unwrap();
        assert_eq!(t, LoadTicket::Attached(job));
    }
    let extra = Incarnation::new(12, 5);
    assert_eq!(w.worker.attach_waiter(job, extra), Ok(()));
    w.resume();
    let v = w.collect(4);
    let t = terminals(&v);
    assert_eq!(t.len(), 4, "one terminal per waiter: {v:?}");
    assert!(t.iter().all(|c| c.job == job));
    assert!(t
        .iter()
        .all(|c| c.terminal == Some(Terminal::Done(JobStage::Completed))));
    let mut got: Vec<Incarnation> = t.iter().map(|c| c.incarnation).collect();
    got.sort();
    let mut want = vec![leader, followers[0], followers[1], extra];
    want.sort();
    assert_eq!(got, want, "each waiter gets exactly its own token back");
    assert_eq!(w.fake.stats().reads, 1, "still one read");
}

#[test]
fn cancel_waiter_removes_exactly_that_incarnation() {
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let (a, b, c) = (
        Incarnation::new(1, 1),
        Incarnation::new(1, 2),
        Incarnation::new(2, 1),
    );
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.worker.attach_waiter(job, b).unwrap();
    w.worker.attach_waiter(job, c).unwrap();
    assert_eq!(
        w.worker.cancel_waiter(job, b),
        CancelOutcome::TooLate,
        "others remain"
    );
    assert_eq!(
        w.worker.cancel_waiter(job, b),
        CancelOutcome::TooLate,
        "a token that is not waiting changes nothing"
    );
    assert_eq!(w.worker.capacity().restore_ops, 1);
    assert!(w.poll_all().is_empty(), "no terminal before the read ends");
    w.resume();
    let v = w.collect(3);
    let mut got: Vec<Incarnation> = terminals(&v).iter().map(|x| x.incarnation).collect();
    got.sort();
    let mut want = vec![a, b, c];
    want.sort();
    assert_eq!(
        got, want,
        "every waiter, detached or not, gets exactly one terminal after the I/O ended"
    );
    assert!(w.poll_all().is_empty(), "and no second one");

    // Detaching every waiter in turn cancels the job only on the last one.
    let w = w_with(|_| {}, |_| {});
    let k = w.inject(8);
    let (_, h) = w.chunk(8);
    let mem = Mem::new(2, 0xEE);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    w.worker.attach_waiter(job, b).unwrap();
    w.worker.attach_waiter(job, c).unwrap();
    assert_eq!(w.worker.cancel_waiter(job, a), CancelOutcome::TooLate);
    assert_eq!(w.worker.cancel_waiter(job, c), CancelOutcome::TooLate);
    assert_eq!(w.worker.cancel_waiter(job, b), CancelOutcome::Cancelled);
    let v = w.collect(3);
    assert!(terminals(&v)
        .iter()
        .all(|x| x.terminal == Some(Terminal::Cancelled)));
}

#[test]
fn the_same_incarnation_cannot_wait_twice_on_one_job() {
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let a = Incarnation::new(5, 1);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, a)
        .unwrap()
        .job();
    assert!(w.worker.attach_waiter(job, a).is_err(), "duplicate refused");
    assert_eq!(
        w.worker.attach_waiter(job, Incarnation::new(5, 2)),
        Ok(()),
        "a new epoch of the same slot is a different waiter"
    );
    w.resume();
    assert_eq!(terminals(&w.collect(2)).len(), 2);
}

#[test]
fn attach_is_refused_when_completion_slots_are_exhausted() {
    // ADR 7.1 item 9: every waiter will need a terminal completion slot, held
    // until the caller polls it. With room for 6 completions, one job holds 3
    // and three waiters take the rest; the next waiter is refused, not queued.
    let w = w_with(
        |_| {},
        |c| {
            c.max_restore_ops = 1;
            c.max_park_ops = 1;
            c.max_waiters_per_job = 16;
            c.max_waiters_total = 1024;
        },
    );
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || mem.dest(2).0, Incarnation::seq(1))
        .unwrap()
        .job();
    for n in 0..3 {
        assert_eq!(
            w.worker.attach_waiter(job, Incarnation::seq(100 + n)),
            Ok(()),
            "waiter {n} fits"
        );
    }
    assert_eq!(
        w.worker.attach_waiter(job, Incarnation::seq(200)),
        Err(PlatformError::QueueFull)
    );
    w.resume();
    let v = w.collect(4);
    assert_eq!(
        terminals(&v).len(),
        4,
        "every admitted waiter gets its terminal"
    );
}

// ---- Grok C5 review fixes -------------------------------------------------

#[test]
fn a_second_store_of_a_live_key_never_takes_over_the_first_jobs_completions() {
    // P1-1: the 17th owner (here the 2nd, with one waiter per job) must not
    // open a second worker job the inner store would merge into the first.
    let w = w_with(|_| {}, |c| c.max_waiters_per_job = 1);
    w.fake.script(JobScript::fault(Fault::Hold));
    let a = w.store(1).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(a));
    let (_, h) = w.chunk(1);
    let mem = Mem::new(2, 0);
    let (src, _keep) = mem.source(2, 4);
    let second = w.worker.submit_store(h, src, Incarnation::seq(901));
    match second {
        Ok(j) => assert_eq!(j, a, "it may only join the live job"),
        Err(e) => assert_eq!(e, PlatformError::QueueFull),
    }
    assert_eq!(w.worker.cancel(a), CancelOutcome::Cancelled);
    let v = w.settle();
    let t = terminals(&v);
    assert!(t.iter().any(|c| c.job == a), "the first job ends: {v:?}");
    assert_eq!(w.worker.capacity().park_ops, 0, "no park slot leaks");
}

#[test]
fn a_dedup_merge_by_the_store_never_strands_a_job() {
    // P1-1 through the other door: the live job's only waiter cancels too late
    // (the store already holds it), then a new store of the key arrives.
    let w = rig();
    // The worker never moves the clock (step 0); the test moves the fake into
    // its file sync, which it stalls and where cancel is too late.
    w.fake
        .script(JobScript::latency(5_000, 0).with(Fault::StallFsync(1_000_000)));
    w.step.store(0, Ordering::SeqCst);
    let a = w.store(1).unwrap();
    w.resume();
    wait_for("dispatch", || w.dispatched(a));
    w.fake.advance(10_000);
    assert!(w.dispatched(a), "still in the store");
    assert_eq!(w.worker.cancel(a), CancelOutcome::TooLate);
    let (_, h) = w.chunk(1);
    let mem = Mem::new(2, 0);
    let (src, _keep) = mem.source(2, 4);
    let b = w.worker.submit_store(h, src, Incarnation::seq(902));
    w.step.store(1_000_000, Ordering::SeqCst);
    let v = w.settle();
    let jobs: BTreeSet<JobId> = terminals(&v).iter().map(|c| c.job).collect();
    assert!(jobs.contains(&a), "the first job ends: {v:?}");
    if let Ok(b) = b {
        assert!(jobs.contains(&b), "the second job ends: {v:?}");
    }
    assert_eq!(w.worker.capacity().park_ops, 0);
}

#[test]
fn a_restored_landing_holder_outlives_the_terminal_the_caller_sees() {
    // P1-2: the caller adopts the landing blocks after it sees Completed, so
    // the holder must still be alive then; it goes when the caller drops the
    // completion (and every copy of it).
    let w = rig();
    w.inject(7);
    let released = Arc::new(AtomicU64::new(0));
    let (job, _mem, handle) = w.load_with(7, released_guard(&released));
    let job = job.unwrap();
    w.resume();
    let v = w.collect(1);
    let t = terminals(&v)[0].clone();
    assert_eq!(t.job, job);
    assert_eq!(t.terminal, Some(Terminal::Done(JobStage::Completed)));
    assert!(handle.is_held(), "alive when the caller sees Completed");
    assert_eq!(released.load(Ordering::SeqCst), 0);
    assert!(t.holder.is_some(), "handed back with the terminal");
    drop(v);
    assert!(handle.is_held(), "a copy still holds it");
    drop(t);
    assert!(!handle.is_held());
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn coalesced_waiters_share_one_landing_holder_until_the_last_drops_it() {
    let w = rig();
    let k = w.inject(7);
    let (_, h) = w.chunk(7);
    let mem = Mem::new(2, 0xEE);
    let released = Arc::new(AtomicU64::new(0));
    let (dst, handle) = mem.dest_with(2, released_guard(&released));
    let mut dst = Some(dst);
    let job = w
        .worker
        .submit_load_coalesced(k, &h, || dst.take().unwrap(), Incarnation::seq(1))
        .unwrap()
        .job();
    w.worker.attach_waiter(job, Incarnation::seq(2)).unwrap();
    w.resume();
    let v = w.collect(2);
    let mut t: Vec<Completion> = terminals(&v).into_iter().cloned().collect();
    drop(v);
    assert!(t.iter().all(|c| c.holder.is_some()));
    t.pop();
    assert!(handle.is_held(), "one waiter still holds it");
    t.clear();
    assert!(!handle.is_held());
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn a_panic_never_reports_a_job_whose_buffer_the_store_still_owns() {
    // P1-3: after the worker thread panics, a job the store still holds stays
    // unfinished (its source held) until the store lets go; then it fails once.
    let boom = Arc::new(AtomicBool::new(false));
    let fields = tiny();
    let ns = fields.namespace_id().unwrap();
    let fake = Arc::new(FakeKvStore::new(FakeConfig::new(ns)));
    let b = boom.clone();
    let f2 = fake.clone();
    let worker = StorageWorker::spawn(
        fake.clone(),
        WorkerConfig {
            idle_poll: Duration::ZERO,
            ..WorkerConfig::default()
        },
        move || {
            if b.load(Ordering::SeqCst) {
                panic!("injected store failure");
            }
            f2.advance(0);
        },
    )
    .unwrap();
    fake.script(JobScript::fault(Fault::Hold));
    let mem = Mem::new(2, 1);
    let released = Arc::new(AtomicU64::new(0));
    let (src, handle) = mem.source_with(2, 4, released_guard(&released));
    let tokens: Vec<u32> = (0..8).collect();
    let key = aien_kv_store::chunk_chain(&ns, 8, &tokens).unwrap()[0];
    let h = fields.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
    let job = worker.submit_store(h, src, Incarnation::seq(1)).unwrap();
    wait_for("dispatch", || worker.inner_job(job).is_some());
    let inner = worker.inner_job(job).unwrap();
    boom.store(true, Ordering::SeqCst);
    wait_for("the worker failed", || {
        worker.capacity().state == aien_kv_store::StoreState::Failed
    });
    assert!(
        worker.poll(8).is_empty(),
        "no terminal while the store owns the buffer"
    );
    assert!(handle.is_held());
    // The store lets go (its own cancel); only now the job fails, once.
    assert_eq!(fake.cancel(inner), CancelOutcome::Cancelled);
    fake.advance(0);
    let mut v = Vec::new();
    wait_for("terminal after the store let go", || {
        v.extend(worker.poll(8));
        !terminals(&v).is_empty()
    });
    assert_eq!(terminals(&v).len(), 1);
    assert!(matches!(
        terminals(&v)[0].terminal,
        Some(Terminal::Failed(PlatformError::HardwareFault(_)))
    ));
    wait_for("source released", || !handle.is_held());
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn shutdown_cancels_a_job_that_is_still_being_handed_to_the_store() {
    // P2-4: a load in Dispatching when shutdown starts must not go on to write
    // its destination.
    let w = rig();
    w.inject(7);
    w.probe.submit_gate.set(false);
    let (job, mem, _h) = w.load_h(7);
    let job = job.unwrap();
    w.resume();
    let g = w.probe.submit_gate.clone();
    wait_for("blocked inside submit_load", move || g.waiting() >= 1);
    assert!(w.worker.inner_job(job).is_none(), "still Dispatching");
    let _ = w.worker.shutdown(0);
    w.probe.submit_gate.set(true);
    let v = w.settle();
    assert_eq!(
        terminals(&v)[0].terminal,
        Some(Terminal::Cancelled),
        "{v:?}"
    );
    assert_eq!(w.fake.stats().reads, 0, "no read after shutdown");
    assert!(
        !w.probe.threads_of("cancel").is_empty(),
        "the worker forwards the cancel itself, not relying on the store's shutdown"
    );
    assert_eq!(mem.bytes(), vec![0xEE; 2 * BLOCK]);
}

// ---- default park bound (C9 bench finding) ---------------------------------

#[test]
fn default_park_bound_admits_eight_conversations_finishing_together() {
    // The C9 bench: 8 conversations finishing in one step park 3 chunks each
    // (24) and the old default of 16 refused 8 of them. The default must
    // admit 8 conversations of up to 8 parks each with no QueueFull.
    let defaults = WorkerConfig::default();
    let w = w_with(|_| {}, |_| {});
    assert_eq!(w.worker.capacity().max_park_ops, defaults.max_park_ops);
    let mut refused = 0;
    for n in 0..64u32 {
        if w.store(1000 + n).is_err() {
            refused += 1;
        }
    }
    assert_eq!(refused, 0, "parks refused at 8 conversations x 8 chunks");
    w.resume();
    w.settle();
}
