//! Scheduling for ahead-of-Enter prefill (OWN-30B T2).
//!
//! The rule: **generation has priority, background prefill yields.**
//!
//! * Every engine call (a prefill chunk or a whole generation) holds `engine_lock`, so the
//!   engine never sees two callers and a prefill can never corrupt a running generation.
//! * Prefill runs in small chunks ([`PrefillConfig::chunk`] tokens). Between chunks the
//!   worker looks at `gen_active`; a generation sets it *before* it waits for the engine
//!   lock, so the worker never starts another chunk once a generation is waiting. A
//!   generation therefore waits at most one chunk.
//! * Starting a generation cancels the pending prefill job: the generation prefills what it
//!   needs itself, and the KV it leaves behind is the newest truth.
//! * One job slot, latest wins. A new prefill request replaces the pending one (a keystroke
//!   supersedes the previous draft); work already done stays in the KV and the engine's
//!   prefix diff reuses it.
//! * The engine holds ONE resident KV. Session records here are the bounded bookkeeping
//!   (count and byte caps, LRU); switching to another session's different prompt rewinds to
//!   the longest common prefix, which keeps a shared system prompt warm across sessions.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::engine::{Engine, EngineError};
use crate::session::{common_prefix, SessionError, SessionLimits, SessionTable};

#[derive(Debug, Clone)]
pub struct PrefillConfig {
    pub enabled: bool,
    pub limits: SessionLimits,
    /// Tokens computed per engine call; bounds how long a generation can be kept waiting.
    pub chunk: usize,
    /// Trailing draft tokens left out of a prefill (the last word may still change).
    pub hold_back: usize,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        Self { enabled: true, limits: SessionLimits::default(), chunk: 32, hold_back: 2 }
    }
}

#[derive(Debug, Default)]
pub struct Stats {
    pub jobs_submitted: AtomicU64,
    pub jobs_cancelled: AtomicU64,
    pub tokens_computed: AtomicU64,
    pub rewinds: AtomicU64,
    pub yields: AtomicU64,
    pub busy_retries: AtomicU64,
}

#[derive(Clone)]
struct Job {
    session: String,
    tokens: Arc<Vec<u32>>,
    epoch: u64,
}

struct Inner {
    job: Option<Job>,
    epoch: u64,
    worker: bool,
    sessions: SessionTable,
    /// Lower bound on the tokens resident in the engine, used only for reporting.
    shadow: Vec<u32>,
}

pub struct Scheduler {
    cfg: PrefillConfig,
    engine_lock: Mutex<()>,
    gen_active: AtomicBool,
    inner: Mutex<Inner>,
    pub stats: Stats,
}

/// What a prefill submission tells the caller right away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submitted {
    pub target_tokens: usize,
    /// Tokens of the target already resident when the request arrived.
    pub cached_tokens: usize,
    /// Tokens still to compute in the background.
    pub to_compute: usize,
    pub evicted: Vec<String>,
}

/// A running generation's claim on the engine. Dropping it frees the claim.
pub struct GenPermit {
    sched: Arc<Scheduler>,
}

impl GenPermit {
    /// Waits for the engine (at most one prefill chunk) and returns exclusive access.
    pub fn lock_engine(&self) -> MutexGuard<'_, ()> {
        self.sched.engine_lock.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Records what the engine holds after the generation (the prompt is the sure part).
    pub fn finished(&self, prompt: &[u32]) {
        let mut g = self.sched.lock_inner();
        g.shadow = prompt.to_vec();
    }
}

impl Drop for GenPermit {
    fn drop(&mut self) {
        self.sched.gen_active.store(false, Ordering::SeqCst);
    }
}

impl Scheduler {
    pub fn new(cfg: PrefillConfig) -> Arc<Self> {
        let sessions = SessionTable::new(cfg.limits.clone());
        Arc::new(Self {
            cfg,
            engine_lock: Mutex::new(()),
            gen_active: AtomicBool::new(false),
            inner: Mutex::new(Inner { job: None, epoch: 0, worker: false, sessions, shadow: Vec::new() }),
            stats: Stats::default(),
        })
    }

    pub fn config(&self) -> &PrefillConfig {
        &self.cfg
    }

    fn lock_inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn generation_active(&self) -> bool {
        self.gen_active.load(Ordering::SeqCst)
    }

    /// True while a prefill job is pending or running.
    pub fn prefill_pending(&self) -> bool {
        self.lock_inner().job.is_some()
    }

    /// Claims the engine for a generation, or `None` when one is already running (the
    /// server answers 503 as before). Cancels the pending prefill job and records the
    /// session's prompt.
    pub fn begin_generation(self: &Arc<Self>, session: Option<&str>, prompt: &[u32]) -> Option<GenPermit> {
        if self
            .gen_active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }
        let mut g = self.lock_inner();
        if g.job.take().is_some() {
            self.stats.jobs_cancelled.fetch_add(1, Ordering::Relaxed);
        }
        g.epoch += 1;
        if let (true, Some(s)) = (self.cfg.enabled, session) {
            // A prompt larger than the byte cap is simply not tracked.
            let _ = g.sessions.put(s, prompt.to_vec(), prompt.len());
        }
        drop(g);
        Some(GenPermit { sched: Arc::clone(self) })
    }

    /// Queues a background prefill of `tokens` for `session` (latest wins).
    pub fn submit(
        self: &Arc<Self>,
        engine: &Arc<dyn Engine>,
        session: &str,
        tokens: Vec<u32>,
    ) -> Result<Submitted, SessionError> {
        let mut g = self.lock_inner();
        let cached = common_prefix(&g.shadow, &tokens);
        let evicted = g.sessions.put(session, tokens.clone(), cached)?;
        g.epoch += 1;
        let epoch = g.epoch;
        let target = tokens.len();
        self.stats.jobs_submitted.fetch_add(1, Ordering::Relaxed);
        if cached >= target {
            g.job = None;
        } else {
            g.job = Some(Job { session: session.to_string(), tokens: Arc::new(tokens), epoch });
            if !g.worker {
                g.worker = true;
                let me = Arc::clone(self);
                let eng = Arc::clone(engine);
                std::thread::Builder::new()
                    .name("aien-prefill".into())
                    .spawn(move || me.worker(eng))
                    .expect("spawn prefill worker");
            }
        }
        Ok(Submitted { target_tokens: target, cached_tokens: cached.min(target), to_compute: target - cached.min(target), evicted })
    }

    /// Snapshot for `GET /v1/aien/sessions`.
    pub fn snapshot(&self) -> serde_json::Value {
        let g = self.lock_inner();
        let pending = g.job.as_ref().map(|j| j.session.clone());
        let sessions: Vec<_> = g
            .sessions
            .names_by_recency()
            .into_iter()
            .filter_map(|n| {
                g.sessions.get(&n).map(|e| {
                    serde_json::json!({
                        "session": n,
                        "tokens": e.tokens.len(),
                        "prefilled": e.prefilled,
                        "pending": pending.as_deref() == Some(n.as_str()),
                        "error": e.error,
                    })
                })
            })
            .collect();
        let l = g.sessions.limits();
        serde_json::json!({
            "sessions": sessions,
            "pending_session": pending,
            "generation_active": self.generation_active(),
            "limits": {"max_sessions": l.max_sessions, "max_kv_bytes": l.max_kv_bytes, "bytes_per_token": l.bytes_per_token},
            "tracked_kv_bytes": g.sessions.total_bytes(),
            "stats": {
                "jobs_submitted": self.stats.jobs_submitted.load(Ordering::Relaxed),
                "jobs_cancelled": self.stats.jobs_cancelled.load(Ordering::Relaxed),
                "tokens_computed": self.stats.tokens_computed.load(Ordering::Relaxed),
                "rewinds": self.stats.rewinds.load(Ordering::Relaxed),
                "yields": self.stats.yields.load(Ordering::Relaxed),
                "busy_retries": self.stats.busy_retries.load(Ordering::Relaxed),
            },
        })
    }

    /// Blocks until no prefill job is pending (boot warmup and tests).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let t0 = std::time::Instant::now();
        while self.prefill_pending() {
            if t0.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        true
    }

    fn worker(self: Arc<Self>, engine: Arc<dyn Engine>) {
        loop {
            let job = {
                let mut g = self.lock_inner();
                match g.job.clone() {
                    Some(j) => j,
                    None => {
                        g.worker = false;
                        return;
                    }
                }
            };
            self.run_job(&engine, &job);
            let mut g = self.lock_inner();
            // Clear the slot only if it still holds the job we just ran.
            if g.job.as_ref().is_some_and(|j| j.epoch == job.epoch) {
                g.job = None;
            }
        }
    }

    fn still_current(&self, job: &Job) -> bool {
        self.lock_inner().epoch == job.epoch
    }

    fn run_job(&self, engine: &Arc<dyn Engine>, job: &Job) {
        let chunk = self.cfg.chunk.max(1);
        let mut busy_retries = 0u32;
        loop {
            if !self.still_current(job) {
                return;
            }
            if self.gen_active.load(Ordering::SeqCst) {
                self.stats.yields.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            let guard = self.engine_lock.lock().unwrap_or_else(|p| p.into_inner());
            // A generation may have claimed the engine while we waited for the lock.
            if self.gen_active.load(Ordering::SeqCst) || !self.still_current(job) {
                drop(guard);
                continue;
            }
            let res = engine.prefill(&job.tokens, chunk);
            drop(guard);
            match res {
                Ok(o) => {
                    self.stats.tokens_computed.fetch_add(o.computed as u64, Ordering::Relaxed);
                    if o.rewound {
                        self.stats.rewinds.fetch_add(1, Ordering::Relaxed);
                    }
                    let mut g = self.lock_inner();
                    if g.epoch == job.epoch {
                        g.shadow = job.tokens[..o.cached.min(job.tokens.len())].to_vec();
                        if let Some(e) = g.sessions.get_mut(&job.session) {
                            e.prefilled = o.cached;
                        }
                    }
                    drop(g);
                    if o.cached >= job.tokens.len() || (o.computed == 0 && o.reused == 0 && o.cached == 0) {
                        return;
                    }
                    if o.computed == 0 {
                        return; // no progress: do not spin
                    }
                }
                Err(EngineError::Busy(_)) => {
                    self.stats.busy_retries.fetch_add(1, Ordering::Relaxed);
                    busy_retries += 1;
                    if busy_retries > 20 {
                        self.note_error(job, "chip busy for too long, prefill dropped");
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    self.note_error(job, &e.to_string());
                    return;
                }
            }
        }
    }

    fn note_error(&self, job: &Job, msg: &str) {
        let mut g = self.lock_inner();
        if let Some(e) = g.sessions.get_mut(&job.session) {
            e.error = Some(msg.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerateParams, StubEngine};
    use std::time::Instant;

    fn params() -> GenerateParams {
        GenerateParams { max_new_tokens: 4, temperature: 0.0, top_p: 1.0, top_k: 0, seed: 1, stop_token_ids: vec![] }
    }

    fn toks(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn cfg(chunk: usize) -> PrefillConfig {
        PrefillConfig { chunk, hold_back: 0, ..PrefillConfig::default() }
    }

    fn engine_arc(e: StubEngine) -> (Arc<StubEngine>, Arc<dyn Engine>) {
        let a = Arc::new(e);
        let d: Arc<dyn Engine> = a.clone();
        (a, d)
    }

    #[test]
    fn append_only_prefill_computes_just_the_new_suffix() {
        let (stub, eng) = engine_arc(StubEngine::new(vec![1]));
        let s = Scheduler::new(cfg(16));
        let r = s.submit(&eng, "a", toks(0..100)).unwrap();
        assert_eq!((r.cached_tokens, r.to_compute), (0, 100));
        assert!(s.wait_idle(Duration::from_secs(5)));
        assert_eq!(stub.resident_snapshot(), toks(0..100));
        // The draft grows by 7 tokens: only those are computed.
        let before = stub.computed_total();
        let r = s.submit(&eng, "a", toks(0..107)).unwrap();
        assert_eq!((r.cached_tokens, r.to_compute), (100, 7));
        assert!(s.wait_idle(Duration::from_secs(5)));
        assert_eq!(stub.computed_total() - before, 7);
        assert_eq!(stub.resident_snapshot(), toks(0..107));
    }

    #[test]
    fn edit_rewinds_and_recomputes_only_from_the_divergence() {
        let (stub, eng) = engine_arc(StubEngine::new(vec![1]));
        let s = Scheduler::new(cfg(16));
        s.submit(&eng, "a", toks(0..60)).unwrap();
        assert!(s.wait_idle(Duration::from_secs(5)));
        let before = stub.computed_total();
        let mut edited = toks(0..40);
        edited.extend([900, 901, 902, 903, 904]);
        s.submit(&eng, "a", edited.clone()).unwrap();
        assert!(s.wait_idle(Duration::from_secs(5)));
        assert_eq!(stub.resident_snapshot(), edited, "KV must equal the edited draft exactly");
        assert_eq!(stub.computed_total() - before, 5);
        assert!(s.stats.rewinds.load(Ordering::Relaxed) >= 1);
    }

    /// The same edit scenario as a checker, so the mutant below can be run through it.
    fn edit_scenario_ok(stub: StubEngine) -> Result<(), String> {
        let (stub, eng) = engine_arc(stub);
        let s = Scheduler::new(cfg(16));
        s.submit(&eng, "a", toks(0..60)).unwrap();
        s.wait_idle(Duration::from_secs(5));
        let mut edited = toks(0..40);
        edited.extend([900, 901, 902]);
        s.submit(&eng, "a", edited.clone()).unwrap();
        s.wait_idle(Duration::from_secs(5));
        if stub.resident_snapshot() == edited {
            Ok(())
        } else {
            Err(format!("KV holds {} tokens that differ from the edited draft", stub.resident_snapshot().len()))
        }
    }

    #[test]
    fn mutant_that_skips_the_rewind_is_caught() {
        assert_eq!(edit_scenario_ok(StubEngine::new(vec![1])), Ok(()), "the real engine passes");
        let r = edit_scenario_ok(StubEngine::new(vec![1]).with_mutant_skip_rewind());
        assert!(r.is_err(), "the rewind-skipping mutant must be caught, got {r:?}");
    }

    #[test]
    fn prefill_yields_to_generation_within_one_chunk_and_never_corrupts_it() {
        // 4 ms per computed token, chunk 8: one chunk is about 32 ms.
        let (stub, eng) =
            engine_arc(StubEngine::new(vec![7, 8, 9]).with_prefill_delay(Duration::from_millis(4)));
        let s = Scheduler::new(cfg(8));
        s.submit(&eng, "a", toks(0..400)).unwrap(); // would take ~1.6 s alone
        std::thread::sleep(Duration::from_millis(100)); // let the worker get going
        let t0 = Instant::now();
        let permit = s.begin_generation(Some("b"), &[500, 501, 502]).expect("first generation claims the engine");
        let guard = permit.lock_engine();
        let waited = t0.elapsed();
        assert!(waited < Duration::from_millis(250), "generation waited {waited:?}, more than a chunk or two");
        let mut got = Vec::new();
        let out = eng.generate(&[500, 501, 502], &params(), &mut |t| {
            got.push(t);
            true
        });
        drop(guard);
        let out = out.expect("generation runs");
        assert_eq!(got, vec![7, 8, 9], "generation output intact");
        assert_eq!(out.emitted_tokens, 3);
        // The pending job was cancelled by the generation, so the KV is the generation's.
        assert!(!s.prefill_pending());
        let kv = stub.resident_snapshot();
        assert_eq!(&kv[..3], &[500, 501, 502]);
        assert_eq!(&kv[3..], &[7, 8, 9]);
        assert!(s.stats.jobs_cancelled.load(Ordering::Relaxed) >= 1);
        drop(permit);
    }

    #[test]
    fn a_job_submitted_during_generation_waits_then_runs() {
        let (stub, eng) = engine_arc(StubEngine::new(vec![1]));
        let s = Scheduler::new(cfg(16));
        let permit = s.begin_generation(None, &[1, 2]).unwrap();
        s.submit(&eng, "a", toks(0..50)).unwrap();
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(stub.computed_total(), 0, "no prefill while a generation is active");
        assert!(s.stats.yields.load(Ordering::Relaxed) > 0);
        drop(permit);
        assert!(s.wait_idle(Duration::from_secs(5)));
        assert_eq!(stub.resident_snapshot(), toks(0..50));
    }

    #[test]
    fn only_one_generation_at_a_time() {
        let s = Scheduler::new(cfg(16));
        let p = s.begin_generation(None, &[1]).unwrap();
        assert!(s.begin_generation(None, &[2]).is_none());
        drop(p);
        assert!(s.begin_generation(None, &[2]).is_some());
    }

    #[test]
    fn newer_draft_supersedes_the_pending_one() {
        let (stub, eng) = engine_arc(StubEngine::new(vec![1]).with_prefill_delay(Duration::from_millis(2)));
        let s = Scheduler::new(cfg(8));
        s.submit(&eng, "a", toks(0..300)).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let mut newer = toks(0..16);
        newer.extend(toks(1000..1010));
        s.submit(&eng, "a", newer.clone()).unwrap();
        assert!(s.wait_idle(Duration::from_secs(5)));
        assert_eq!(stub.resident_snapshot(), newer);
    }

    #[test]
    fn session_caps_hold_through_the_scheduler() {
        let (_stub, eng) = engine_arc(StubEngine::new(vec![1]));
        let mut c = cfg(16);
        c.limits = SessionLimits { max_sessions: 2, max_kv_bytes: 1 << 20, bytes_per_token: 1 };
        let s = Scheduler::new(c);
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            s.submit(&eng, name, toks(0..(10 + i as u32))).unwrap();
            assert!(s.wait_idle(Duration::from_secs(5)));
        }
        let snap = s.snapshot();
        assert_eq!(snap["sessions"].as_array().unwrap().len(), 2);
        assert_eq!(snap["sessions"][0]["session"], "c");
    }
}
