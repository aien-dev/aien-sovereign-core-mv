//! Qwen3.8-27B behind the door, many conversations at once (cascade/continuous-batching).
//!
//! [`Qwen38Door`](crate::qwen38_engine::Qwen38Door) serves one request at a time and reads the
//! 26.9 GB of weights once per token for that one user. This door keeps `slots` conversation
//! states on the chip ([`Stream`]) and one scheduler thread that, every step, feeds one token
//! of every active conversation through [`Qwen38Omega::step_rows`]: one weight read for all of
//! them. Requests join at the next step (their prompt goes through the chip prefill into a free
//! slot first) and leave when they stop, hit their length, or their client goes away; the
//! others never notice. Each row's tokens are bit for bit what it would get alone (the rows
//! path's exactness contract, `aien_inference_abi::qwen38_omega::batch`).
//!
//! The server hands out `slots` permits ([`Engine::concurrency`]); the `slots + 1`th request
//! gets 503 `server_busy` as before. Chip etiquette: `/tmp/aien-gb10.lock` is taken when the
//! first conversation arrives and released when the last one leaves, never held while idle.
//! Next-turn reuse (the single door's live-state and snapshot resume) is not offered here:
//! every prompt is fed cold into its slot.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::spec::RowStats;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, RowInput, Stream};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::qwen38_engine::{
    rss_bytes, KEEP_FREE, LOAD_LOCK_WAIT, PREFILL_CHUNK, REQUEST_LOCK_WAIT,
};
use crate::sampling::SamplingObserver;

/// Most conversations one door stacks (the multi-row GEMM takes up to 1024 rows; the chip
/// memory for the slots is the practical limit).
pub const MAX_SLOTS: usize = 64;

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// A request handed to the scheduler.
struct Job {
    prompt: Vec<u32>,
    params: GenerateParams,
    out: mpsc::SyncSender<Ev>,
    /// Set by the request thread when `on_token` says stop (client gone, stop string).
    cancel: Arc<AtomicBool>,
    t_queued: Instant,
}

enum Ev {
    Token(u32),
    Done(GenerateOutcome),
    Busy(String),
    Failed(String),
}

/// One conversation in flight on the scheduler thread.
struct Active {
    job: Job,
    stream: Stream,
    observer: SamplingObserver,
    emitted: usize,
    /// Logits after the last fed token (the prefill's, then every step's).
    logits: Vec<f32>,
    /// The token to feed next (the last sampled one), once there is one.
    next: Option<u32>,
    prefill_ms: u64,
    t_decode: Instant,
    /// Steps this row rode and the rows that rode with it (for the mean batch size).
    steps: u64,
    rows_sum: u64,
}

/// The door: the server side of the scheduler thread.
pub struct Qwen38BatchDoor {
    tx: Mutex<Option<mpsc::Sender<Job>>>,
    slots: usize,
    max_context: usize,
    /// Memory the slots took on the box, measured at load (`MemAvailable` before and after).
    pub slot_bytes_measured: u64,
    /// Memory one slot takes on paper (`Qwen38Omega::stream_bytes`).
    pub slot_bytes_on_paper: usize,
}

impl Qwen38BatchDoor {
    /// Uploads the checkpoint at `dir` and makes `slots` conversation states of `max_context`
    /// positions each (rounded up to whole 256-position passes), under the GPU lock.
    pub fn load(dir: &Path, max_context: usize, slots: usize) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        if !aien_omega_q38::has_prefill() {
            return Err("this binary has no qwen38 prefill entries linked".into());
        }
        if slots == 0 || slots > MAX_SLOTS {
            return Err(format!("--batch-slots {slots}: 1..={MAX_SLOTS}"));
        }
        let c = Qwen38Config::from_checkpoint(dir)?;
        let max_pos = max_context.next_multiple_of(256);
        let need = need_bytes(&c, max_pos, PREFILL_CHUNK);
        let avail = mem_available();
        if avail < need + KEEP_FREE {
            return Err(format!(
                "needs {} MiB plus {} MiB kept free, MemAvailable {} MiB (another model may be loaded); refusing",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let lock = GpuLock::acquire_queued_at(Path::new(GPU_LOCK_PATH), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen38-batch: uploading {} layers, {slots} slots of {max_pos} positions (model needs {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let model = Qwen38Omega::open(dir, None, max_pos)?;
        let on_paper = model.stream_bytes(max_pos);
        let slot_need = (on_paper as u64) * slots as u64;
        let avail = mem_available();
        if avail < slot_need + KEEP_FREE {
            return Err(format!(
                "{slots} slots need {} MiB plus {} MiB kept free, MemAvailable {} MiB after the upload; refusing",
                slot_need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let before = mem_available();
        let mut streams = Vec::with_capacity(slots);
        for _ in 0..slots {
            streams.push(model.new_stream(max_pos)?);
        }
        let after = mem_available();
        let measured = before.saturating_sub(after);
        drop(lock);
        eprintln!(
            "aien-serve-http: qwen38-batch engine ready in {:.1} s: context {max_context} tokens, {slots} slots, weights {:.2} GB resident, per-slot state {:.1} MiB on paper, {:.1} MiB measured (MemAvailable {} MiB -> {} MiB), rss {:.2} GB",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            on_paper as f64 / (1u64 << 20) as f64,
            measured as f64 / slots as f64 / (1u64 << 20) as f64,
            before >> 20,
            after >> 20,
            rss_bytes() as f64 / 1e9,
        );
        let (tx, rx) = mpsc::channel::<Job>();
        let sched = Scheduler {
            model,
            free: streams,
            rx,
            lock_path: PathBuf::from(GPU_LOCK_PATH),
            lock_wait: REQUEST_LOCK_WAIT,
            max_context,
        };
        std::thread::Builder::new()
            .name("qwen38-batch".into())
            .spawn(move || sched.run())
            .map_err(|e| format!("scheduler thread: {e}"))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            slots,
            max_context,
            slot_bytes_measured: measured,
            slot_bytes_on_paper: on_paper,
        })
    }
}

impl Engine for Qwen38BatchDoor {
    fn kind(&self) -> &str {
        "omega-qwen38-batch"
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn concurrency(&self) -> usize {
        self.slots
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        if prompt.is_empty() {
            return Err(EngineError::Failed("empty prompt".into()));
        }
        if prompt.len() >= self.max_context {
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max: self.max_context,
            });
        }
        let (otx, orx) = mpsc::sync_channel::<Ev>(256);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            prompt: prompt.to_vec(),
            params: p.clone(),
            out: otx,
            cancel: Arc::clone(&cancel),
            t_queued: Instant::now(),
        };
        {
            let tx = self
                .tx
                .lock()
                .map_err(|_| EngineError::Failed("qwen38-batch door poisoned".into()))?;
            tx.as_ref()
                .ok_or_else(|| EngineError::Failed("qwen38-batch scheduler stopped".into()))?
                .send(job)
                .map_err(|_| EngineError::Failed("qwen38-batch scheduler gone".into()))?;
        }
        let mut delivered = 0usize;
        let mut cancelled = false;
        loop {
            match orx.recv() {
                Ok(Ev::Token(t)) => {
                    if cancelled {
                        continue;
                    }
                    delivered += 1;
                    if !on_token(t) {
                        cancelled = true;
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
                Ok(Ev::Done(mut o)) => {
                    if cancelled {
                        o.finish = FinishReason::Cancelled;
                        o.emitted_tokens = delivered;
                    }
                    return Ok(o);
                }
                Ok(Ev::Busy(m)) => return Err(EngineError::Busy(m)),
                Ok(Ev::Failed(m)) => return Err(EngineError::Failed(m)),
                Err(_) => return Err(EngineError::Failed("qwen38-batch scheduler gone".into())),
            }
        }
    }
}

struct Scheduler {
    model: Qwen38Omega,
    free: Vec<Stream>,
    rx: mpsc::Receiver<Job>,
    lock_path: PathBuf,
    lock_wait: Duration,
    max_context: usize,
}

impl Scheduler {
    fn run(mut self) {
        let mut active: Vec<Active> = Vec::new();
        let mut lock: Option<GpuLock> = None;
        // Totals over one busy spell (first arrival to last departure), logged at the end of it.
        let mut spell = RowStats::default();
        let (mut spell_steps, mut spell_wall_ns, mut spell_t0) = (0u64, 0u64, Instant::now());
        loop {
            // Idle: wait for a request, then take the chip.
            if active.is_empty() {
                if lock.take().is_some() {
                    eprintln!(
                        "aien-serve-http: qwen38-batch: spell over: {spell_steps} steps, {} rows, {:.1} ms/step wall, chip gemm {} ms, gdn {} ms, attn {} ms, head {} ms, host {} ms, spell {:.1} s",
                        spell.rows,
                        spell_wall_ns as f64 / 1e6 / spell_steps.max(1) as f64,
                        spell.gemm_ns / 1_000_000,
                        spell.gdn_ns / 1_000_000,
                        spell.attn_ns / 1_000_000,
                        spell.head_ns / 1_000_000,
                        spell.host_ns / 1_000_000,
                        spell_t0.elapsed().as_secs_f64(),
                    );
                }
                let Ok(job) = self.rx.recv() else {
                    return; // the door is gone
                };
                let t = Instant::now();
                match GpuLock::acquire_queued_at(&self.lock_path, self.lock_wait) {
                    Ok(l) => lock = Some(l),
                    Err(m) => {
                        let _ = job.out.send(Ev::Busy(m));
                        continue;
                    }
                }
                eprintln!(
                    "aien-serve-http: qwen38-batch: chip lock after {} ms",
                    t.elapsed().as_millis()
                );
                spell = RowStats::default();
                spell_steps = 0;
                spell_wall_ns = 0;
                spell_t0 = Instant::now();
                self.admit(job, &mut active);
            }
            // Join: every waiting request that fits.
            while !self.free.is_empty() {
                match self.rx.try_recv() {
                    Ok(job) => self.admit(job, &mut active),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => break,
                }
            }
            // Sample one token for every row from its logits; leave on stop / length / cancel.
            let mut i = 0;
            while i < active.len() {
                let a = &mut active[i];
                let finish = Self::sample_one(a, self.max_context);
                match finish {
                    Some(f) => {
                        let a = active.swap_remove(i);
                        self.finish(a, f, None);
                    }
                    None => i += 1,
                }
            }
            if active.is_empty() {
                continue;
            }
            // One weight read for every row.
            let t = Instant::now();
            let mut st = RowStats::default();
            let r = {
                let mut rows: Vec<RowInput<'_>> = active
                    .iter_mut()
                    .map(|a| RowInput {
                        stream: &mut a.stream,
                        token: a.next.expect("a surviving row has a token to feed"),
                    })
                    .collect();
                self.model.step_rows(&mut rows, &mut st)
            };
            let wall = t.elapsed().as_nanos() as u64;
            let m = active.len() as u64;
            spell_steps += 1;
            spell_wall_ns += wall;
            add(&mut spell, &st);
            match r {
                Ok(outs) => {
                    for (a, o) in active.iter_mut().zip(outs) {
                        a.logits = o.logits;
                        a.steps += 1;
                        a.rows_sum += m;
                    }
                }
                Err(e) => {
                    eprintln!("aien-serve-http: qwen38-batch: step over {m} rows failed: {e}");
                    for a in active.drain(..) {
                        self.finish(a, FinishReason::Length, Some(e.clone()));
                    }
                }
            }
        }
    }

    /// Feeds the prompt into a free slot and seats the request. On a prefill error the
    /// request is answered with it and the slot kept free.
    fn admit(&mut self, job: Job, active: &mut Vec<Active>) {
        let Some(mut stream) = self.free.pop() else {
            let _ = job
                .out
                .send(Ev::Failed("qwen38-batch: no free slot".into()));
            return;
        };
        let t0 = Instant::now();
        let mut st = RowStats::default();
        match self
            .model
            .stream_prefill(&mut stream, &job.prompt, PREFILL_CHUNK, &mut st)
        {
            Ok((logits, pf)) => {
                let prefill_ms = t0.elapsed().as_millis() as u64;
                eprintln!(
                    "aien-serve-http: qwen38-batch: join: prompt {} tok, queued {} ms, prefill {prefill_ms} ms (gemm chip {} ms, gdn chip {} ms, attn chip {} ms), now {} rows",
                    job.prompt.len(),
                    job.t_queued.elapsed().as_millis().saturating_sub(prefill_ms as u128),
                    pf.gemm_chip_ns / 1_000_000,
                    pf.gdn_chip_ns / 1_000_000,
                    pf.attn_chip_ns / 1_000_000,
                    active.len() + 1,
                );
                let observer = SamplingObserver::new(&job.params);
                active.push(Active {
                    job,
                    stream,
                    observer,
                    emitted: 0,
                    logits,
                    next: None,
                    prefill_ms,
                    t_decode: Instant::now(),
                    steps: 0,
                    rows_sum: 0,
                });
            }
            Err(e) => {
                eprintln!("aien-serve-http: qwen38-batch: prefill failed: {e}");
                let _ = job.out.send(Ev::Failed(e));
                self.free.push(stream);
            }
        }
    }

    /// Samples the row's next token from its logits and emits it; `Some(reason)` when the row
    /// is done (the token is not fed). Mirrors the single door's loop exactly.
    fn sample_one(a: &mut Active, max_context: usize) -> Option<FinishReason> {
        let p = &a.job.params;
        let plen = a.job.prompt.len();
        if a.job.cancel.load(Ordering::SeqCst) {
            return Some(FinishReason::Cancelled);
        }
        if a.emitted >= p.max_new_tokens || plen + a.emitted >= max_context {
            return Some(FinishReason::Length);
        }
        let tok = a.observer.sample(&a.logits, a.emitted);
        if p.stop_token_ids.contains(&tok) {
            return Some(FinishReason::Stop);
        }
        a.emitted += 1;
        if a.job.out.send(Ev::Token(tok)).is_err() {
            return Some(FinishReason::Cancelled);
        }
        // The last allowed token is not fed: nothing would read its logits.
        if a.emitted >= p.max_new_tokens || plen + a.emitted >= max_context {
            return Some(FinishReason::Length);
        }
        if a.stream.pos >= a.stream.max_pos() {
            return Some(FinishReason::Length);
        }
        a.next = Some(tok);
        None
    }

    fn finish(&mut self, a: Active, finish: FinishReason, err: Option<String>) {
        let decode_ms = a.t_decode.elapsed().as_millis() as u64;
        let steps = a.steps.max(1) as f64;
        eprintln!(
            "aien-serve-http: qwen38-batch: leave: prompt {} tok, prefill {} ms, decode {} tok in {decode_ms} ms over {} steps, mean rows {:.2}, wall {:.1} ms/token, {:?}",
            a.job.prompt.len(),
            a.prefill_ms,
            a.emitted,
            a.steps,
            a.rows_sum as f64 / steps,
            decode_ms as f64 / steps,
            finish,
        );
        let ev = match err {
            Some(e) => Ev::Failed(e),
            None => Ev::Done(GenerateOutcome {
                emitted_tokens: a.emitted,
                finish,
                prefill_ms: a.prefill_ms,
                decode_ms,
                sampling: Some(a.observer.record()),
            }),
        };
        let _ = a.job.out.send(ev);
        self.free.push(a.stream);
    }
}

fn add(sum: &mut RowStats, s: &RowStats) {
    sum.rows += s.rows;
    sum.passes += s.passes;
    sum.gemm_ns += s.gemm_ns;
    sum.gemm_calls += s.gemm_calls;
    sum.ab_ns += s.ab_ns;
    sum.gdn_ns += s.gdn_ns;
    sum.gdn_calls += s.gdn_calls;
    sum.attn_ns += s.attn_ns;
    sum.attn_calls += s.attn_calls;
    sum.host_ns += s.host_ns;
    sum.head_ns += s.head_ns;
    sum.head_calls += s.head_calls;
}
