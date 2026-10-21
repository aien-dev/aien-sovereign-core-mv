//! Qwen3.6-35B-A3B behind the door, many conversations at once (sc L354, the port of L129's
//! qwen38-batch scheduler to the 35B).
//!
//! [`Qwen38Door`](crate::qwen38_engine::Qwen38Door) serves one request at a time. This door keeps
//! `slots` conversation rows on the chip (each its own DeltaNet state, KV cache and position over
//! the one upload of the weights, `Qwen38Omega::add_rows`) and one scheduler thread that, every
//! beat, feeds one token of every live conversation through `Qwen38Omega::step_rows`: the token it
//! just sampled, or (while its prompt is still going in) its next prompt tokens. Requests join at
//! the next beat and leave when they stop, hit their length, or their client goes away; the others
//! never notice. Each row's tokens are bit for bit what that request gets alone (the step's
//! contract; the gate checks it, and a row-swap mutant must break it).
//!
//! Next turns: a row keeps the tokens it was fed after its conversation leaves. A new request
//! whose prompt starts with all of a free row's tokens continues that row (only the new tail is
//! fed); otherwise it takes the least recently used free row, emptied.
//!
//! Today `step_rows` runs the rows' one-row graphs in turn (stage 1: the same chip work as one
//! request at a time, now with no `server_busy` until every slot is taken); L312's multi-row step
//! (one weight read per layer for all rows) plugs in behind the same call.
//!
//! The server hands out `slots` permits ([`Engine::concurrency`]); the `slots + 1`th request gets
//! 503 `server_busy`. Chip etiquette: `/tmp/aien-gb10.lock` is taken when the first conversation
//! arrives and released when the last one leaves, never held while idle.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{
    need_bytes, Qwen38Omega, Qwen38Snapshot, RowFeed, RowOut, SharedWeights, IM_START,
};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::qwen38_engine::{
    is_moe_dir, rss_bytes, CHIP_GREEDY_ENV, KEEP_FREE, LOAD_LOCK_WAIT, PREFILL_CHUNK,
    REQUEST_LOCK_WAIT,
};
use crate::sampling::SamplingObserver;

/// Most conversations one door stacks.
pub const MAX_SLOTS: usize = 64;

/// Prompt tokens a joining row feeds per beat (`AIEN_Q36_PROMPT_PER_BEAT`, default 16): with the
/// rows run in turn this only moves chip time from the rows decoding to the one joining (its
/// time to first token), the total is the same.
pub const PROMPT_PER_BEAT: usize = 16;

fn prompt_per_beat() -> Result<usize, String> {
    match std::env::var("AIEN_Q36_PROMPT_PER_BEAT") {
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(n),
            _ => Err(format!("AIEN_Q36_PROMPT_PER_BEAT={v}: a whole number >= 1")),
        },
        Err(_) => Ok(PROMPT_PER_BEAT),
    }
}

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// What the scheduler needs from the model: rows of conversation state and one step over a set
/// of them. `Qwen38Omega` (the chip) and the host tests' fake implement it.
pub trait RowStep: Send + 'static {
    fn row_count(&self) -> usize;
    /// Empties row `r` (position 0).
    fn row_reset(&mut self, r: usize) -> Result<(), String>;
    /// One token into each listed row; each row's greedy token and (when asked) logits.
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String>;
    /// A host copy of a row's state.
    type Snap: Default + Send;
    fn row_snapshot(&mut self, r: usize, snap: &mut Self::Snap) -> Result<(), String>;
    /// Puts a copy back into row `r` (any row).
    fn row_restore(&mut self, r: usize, snap: &Self::Snap) -> Result<(), String>;
}

impl RowStep for Qwen38Omega {
    fn row_count(&self) -> usize {
        Qwen38Omega::row_count(self)
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        Qwen38Omega::row_reset(self, r)
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        Qwen38Omega::step_rows(self, feed)
    }
    type Snap = Qwen38Snapshot;
    fn row_snapshot(&mut self, r: usize, snap: &mut Qwen38Snapshot) -> Result<(), String> {
        Qwen38Omega::row_snapshot(self, r, snap)
    }
    fn row_restore(&mut self, r: usize, snap: &Qwen38Snapshot) -> Result<(), String> {
        Qwen38Omega::row_restore(self, r, snap)
    }
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

/// A row between conversations: the tokens its state holds and a host copy of its state at its
/// last prompt's last turn header (the single door's two reuse points, per row).
struct Row<T> {
    history: Vec<u32>,
    last_used: u64,
    /// The tokens `snap` holds (empty: none).
    snap_tokens: Vec<u32>,
    snap: T,
}

/// One conversation in flight on the scheduler thread.
struct Active {
    job: Job,
    row: usize,
    /// Prompt tokens in the row's state (the reused ones included).
    fed: usize,
    reused: usize,
    /// Where to copy the state on the way in: the prompt's last turn header (`<|im_start|>`).
    snap_at: Option<usize>,
    observer: SamplingObserver,
    emitted: usize,
    /// Logits after the last fed token, until sampled.
    logits: Option<Vec<f32>>,
    /// The sampled token to feed next.
    next: Option<u32>,
    /// A greedy reply takes the graph's own token (no logits copy per token, L356's single-door
    /// lever); `chip_next` holds it until sampled.
    chip_greedy: bool,
    chip_next: Option<u32>,
    t_admit: Instant,
    prefill_ms: u64,
    t_decode: Instant,
    /// Beats this row decoded in and the rows that rode with it (for the mean batch size).
    steps: u64,
    rows_sum: u64,
}

impl Active {
    fn prefilling(&self) -> bool {
        self.fed < self.job.prompt.len()
    }
}

/// The door: the server side of the scheduler thread.
pub struct Qwen36BatchDoor {
    tx: Mutex<Option<mpsc::Sender<Job>>>,
    slots: usize,
    max_context: usize,
    /// Memory the rows took on the box, measured at load (`MemAvailable` before and after).
    pub slot_bytes_measured: u64,
    /// Memory one row takes on paper (`Qwen38Omega::row_bytes`).
    pub slot_bytes_on_paper: usize,
}

impl Qwen36BatchDoor {
    /// Uploads the 35B checkpoint at `dir` once and makes `slots` conversation rows of
    /// `max_context` positions each (rounded up to whole 256-position passes), under the GPU lock.
    pub fn load(dir: &Path, max_context: usize, slots: usize) -> Result<Self, String> {
        Self::load_with(dir, max_context, slots, None, None)
    }

    /// As [`Self::load`]; with `shared` the big tensors are imported from another process's upload
    /// (L383 shared weights: read-only mappings, no copy, only the rows and the host-side pieces are
    /// this process's own), with `publish` this process serves its own upload at that unix socket
    /// for other copies (`--weights-publish`).
    pub fn load_with(
        dir: &Path,
        max_context: usize,
        slots: usize,
        shared: Option<&SharedWeights>,
        publish: Option<&Path>,
    ) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        if !is_moe_dir(dir) {
            return Err("the batch door serves the Qwen3.6-35B-A3B (qwen3_5_moe) only".into());
        }
        if !aien_omega_q38::q36::is_native() {
            return Err(
                "this binary has no q36 decode graph linked (build against omega q36/serve)".into(),
            );
        }
        if slots == 0 || slots > MAX_SLOTS {
            return Err(format!("--batch-slots {slots}: 1..={MAX_SLOTS}"));
        }
        let per_beat = prompt_per_beat()?;
        let c = Qwen38Config::from_checkpoint(dir)?;
        let max_pos = max_context.next_multiple_of(256);
        // The main graph is never stepped here: its own state is one 256-position pass.
        let need = if shared.is_some() {
            0
        } else {
            need_bytes(&c, 256, PREFILL_CHUNK)
        };
        let avail = mem_available();
        if shared.is_none() && avail < need + KEEP_FREE {
            return Err(format!(
                "needs {} MiB plus {} MiB kept free, MemAvailable {} MiB (another model may be loaded); refusing",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let lock = GpuLock::acquire_queued_at(&serve_lock_path(), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen36-batch: uploading {} layers, {slots} rows of {max_pos} positions (model needs {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let mut model = match shared {
            Some(s) => Qwen38Omega::open_shared(dir, s, 256, 1)?,
            None => Qwen38Omega::open(dir, None, 256)?,
        };
        let weights_line = match shared {
            Some(s) => format!(
                "weights shared from pid {} (read-only mappings, no copy)",
                s.owner_pid
            ),
            None => format!(
                "weights {:.2} GB resident",
                model.weight_bytes() as f64 / 1e9
            ),
        };
        let on_paper = model.row_bytes(max_pos);
        let rows_need = on_paper as u64 * slots as u64;
        let before = mem_available();
        if before < rows_need + KEEP_FREE {
            return Err(format!(
                "{slots} rows need {} MiB plus {} MiB kept free, MemAvailable {} MiB after the upload; refusing",
                rows_need >> 20,
                KEEP_FREE >> 20,
                before >> 20
            ));
        }
        model.add_rows(slots, max_pos)?;
        let after = mem_available();
        let measured = before.saturating_sub(after);
        drop(lock);
        if let Some(sock) = publish {
            model.share(dir)?.publish(sock)?;
            eprintln!(
                "aien-serve-http: qwen36-batch: weights published at {} (unix socket, this user only; test copies start with --weights-from)",
                sock.display()
            );
        }
        eprintln!(
            "aien-serve-http: qwen36-batch engine ready in {:.1} s: context {max_context} tokens, {slots} rows, {weights_line}, per-row state {:.1} MiB on paper, {:.1} MiB measured (MemAvailable {} MiB -> {} MiB), rss {:.2} GB, prompt tokens per beat {per_beat}",
            t0.elapsed().as_secs_f64(),
            on_paper as f64 / (1u64 << 20) as f64,
            measured as f64 / slots as f64 / (1u64 << 20) as f64,
            before >> 20,
            after >> 20,
            rss_bytes() as f64 / 1e9,
        );
        let mut door = Self::with_stepper(
            model,
            max_context,
            serve_lock_path(),
            REQUEST_LOCK_WAIT,
            per_beat,
            std::env::var(CHIP_GREEDY_ENV).as_deref() != Ok("0"),
        )?;
        door.slot_bytes_measured = measured;
        door.slot_bytes_on_paper = on_paper;
        Ok(door)
    }

    /// The door over any [`RowStep`] (every row it has is a slot): the scheduler thread starts
    /// here. `lock_path` is the chip lock (tests pass a private file).
    pub fn with_stepper<S: RowStep>(
        model: S,
        max_context: usize,
        lock_path: PathBuf,
        lock_wait: Duration,
        prompt_per_beat: usize,
        chip_greedy: bool,
    ) -> Result<Self, String> {
        let slots = model.row_count();
        if slots == 0 {
            return Err("qwen36-batch: no rows".into());
        }
        let (tx, rx) = mpsc::channel::<Job>();
        let sched = Scheduler {
            model,
            rows: (0..slots)
                .map(|_| Row {
                    history: Vec::new(),
                    last_used: 0,
                    snap_tokens: Vec::new(),
                    snap: S::Snap::default(),
                })
                .collect(),
            free: (0..slots).rev().collect(),
            rx,
            lock_path,
            lock_wait,
            max_context,
            per_beat: prompt_per_beat.max(1),
            chip_greedy,
            beat: 0,
        };
        std::thread::Builder::new()
            .name("qwen36-batch".into())
            .spawn(move || sched.run())
            .map_err(|e| format!("scheduler thread: {e}"))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            slots,
            max_context,
            slot_bytes_measured: 0,
            slot_bytes_on_paper: 0,
        })
    }
}

impl Engine for Qwen36BatchDoor {
    fn kind(&self) -> &str {
        "omega-qwen36-batch"
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
                .map_err(|_| EngineError::Failed("qwen36-batch door poisoned".into()))?;
            tx.as_ref()
                .ok_or_else(|| EngineError::Failed("qwen36-batch scheduler stopped".into()))?
                .send(job)
                .map_err(|_| EngineError::Failed("qwen36-batch scheduler gone".into()))?;
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
                Err(_) => return Err(EngineError::Failed("qwen36-batch scheduler gone".into())),
            }
        }
    }
}

struct Scheduler<S: RowStep> {
    model: S,
    rows: Vec<Row<S::Snap>>,
    free: Vec<usize>,
    rx: mpsc::Receiver<Job>,
    lock_path: PathBuf,
    lock_wait: Duration,
    max_context: usize,
    per_beat: usize,
    /// Greedy replies take the graph's token (`AIEN_SERVE_CHIP_GREEDY=0` turns it off).
    chip_greedy: bool,
    beat: u64,
}

/// Totals over one busy spell (first arrival to last departure).
#[derive(Default)]
struct Spell {
    beats: u64,
    steps: u64,
    row_tokens: u64,
    decode_tokens: u64,
    wall_ns: u64,
    chip_ns: u64,
}

impl<S: RowStep> Scheduler<S> {
    fn run(mut self) {
        let mut active: Vec<Active> = Vec::new();
        let mut lock: Option<GpuLock> = None;
        let (mut spell, mut spell_t0) = (Spell::default(), Instant::now());
        loop {
            // Idle: wait for a request, then take the chip.
            if active.is_empty() {
                if lock.take().is_some() {
                    let s = &spell;
                    eprintln!(
                        "aien-serve-http: qwen36-batch: spell over: {} beats, {} steps, {} row tokens ({} decoded), {:.1} ms/step wall, chip {:.1} ms/step, spell {:.1} s",
                        s.beats,
                        s.steps,
                        s.row_tokens,
                        s.decode_tokens,
                        s.wall_ns as f64 / 1e6 / s.steps.max(1) as f64,
                        s.chip_ns as f64 / 1e6 / s.steps.max(1) as f64,
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
                    "aien-serve-http: qwen36-batch: chip lock after {} ms",
                    t.elapsed().as_millis()
                );
                spell = Spell::default();
                spell_t0 = Instant::now();
                self.admit(job, &mut active);
            }
            // Join: every waiting request that fits.
            while !self.free.is_empty() {
                match self.rx.try_recv() {
                    Ok(job) => self.admit(job, &mut active),
                    Err(_) => break,
                }
            }
            // Sample one token for every decoding row from its logits; leave on stop / length /
            // cancel (a row still feeding its prompt leaves on cancel only).
            let mut i = 0;
            while i < active.len() {
                let finish = if active[i].prefilling() {
                    active[i]
                        .job
                        .cancel
                        .load(Ordering::SeqCst)
                        .then_some(FinishReason::Cancelled)
                } else {
                    Self::sample_one(&mut active[i], self.max_context)
                };
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
            self.beat += 1;
            spell.beats += 1;
            // One beat: every decoding row feeds its sampled token in the first step; the rows
            // still feeding their prompt feed up to `per_beat` prompt tokens, one per step.
            for sub in 0..self.per_beat {
                // Each joining prompt's state copy at its last turn header, before that token goes in.
                for a in active.iter_mut() {
                    if a.prefilling() && a.snap_at == Some(a.fed) {
                        a.snap_at = None;
                        let row = &mut self.rows[a.row];
                        match self.model.row_snapshot(a.row, &mut row.snap) {
                            Ok(()) => row.snap_tokens = a.job.prompt[..a.fed].to_vec(),
                            Err(e) => {
                                row.snap_tokens.clear();
                                eprintln!(
                                    "aien-serve-http: qwen36-batch: row {} snapshot failed: {e}",
                                    a.row
                                );
                            }
                        }
                    }
                }
                let mut feed = Vec::with_capacity(active.len());
                let mut who = Vec::with_capacity(active.len());
                for (k, a) in active.iter().enumerate() {
                    if a.prefilling() {
                        let plen = a.job.prompt.len();
                        feed.push(RowFeed {
                            row: a.row,
                            token: a.job.prompt[a.fed],
                            logits: a.fed + 1 == plen,
                        });
                        who.push(k);
                    } else if sub == 0 {
                        if let Some(t) = a.next {
                            feed.push(RowFeed {
                                row: a.row,
                                token: t,
                                logits: !a.chip_greedy,
                            });
                            who.push(k);
                        }
                    }
                }
                if feed.is_empty() {
                    break;
                }
                let t = Instant::now();
                let r = self.model.step_rows(&feed);
                spell.wall_ns += t.elapsed().as_nanos() as u64;
                spell.steps += 1;
                spell.row_tokens += feed.len() as u64;
                let m = feed.len() as u64;
                match r {
                    Ok(outs) if outs.len() == feed.len() => {
                        for ((f, &k), o) in feed.iter().zip(&who).zip(outs) {
                            spell.chip_ns += o.stats.chip_ns;
                            let a = &mut active[k];
                            self.rows[a.row].history.push(f.token);
                            if a.prefilling() {
                                a.fed += 1;
                                if !a.prefilling() {
                                    a.prefill_ms = a.t_admit.elapsed().as_millis() as u64;
                                    a.t_decode = Instant::now();
                                }
                            } else {
                                a.next = None;
                                if a.chip_greedy {
                                    a.chip_next = Some(o.token);
                                }
                                a.steps += 1;
                                a.rows_sum += m;
                                spell.decode_tokens += 1;
                            }
                            if f.logits {
                                a.logits = o.logits;
                            }
                        }
                    }
                    r => {
                        let e = match r {
                            Err(e) => e,
                            Ok(o) => format!("step_rows gave {} rows for {}", o.len(), feed.len()),
                        };
                        eprintln!("aien-serve-http: qwen36-batch: step over {m} rows failed: {e}");
                        for a in active.drain(..) {
                            self.finish(a, FinishReason::Length, Some(e.clone()));
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Seats the request on a free row: the longest state some free row holds that the prompt
    /// continues (its live tokens, or its copy at its last turn header), else the least recently
    /// used row, emptied. The single door's reuse rule ([`aien_inference_abi::qwen38_omega::plan_reuse`]), per row.
    fn admit(&mut self, job: Job, active: &mut Vec<Active>) {
        let plen = job.prompt.len();
        let fits = |s: &[u32]| !s.is_empty() && s.len() < plen && job.prompt.starts_with(s);
        // (index in `free`, tokens reused, from the copy)
        let mut best: Option<(usize, usize, bool)> = None;
        for (i, &r) in self.free.iter().enumerate() {
            let row = &self.rows[r];
            for (s, from_snap) in [(&row.history, false), (&row.snap_tokens, true)] {
                if fits(s) && best.is_none_or(|(_, n, _)| s.len() > n) {
                    best = Some((i, s.len(), from_snap));
                }
            }
        }
        let (i, reused, from_snap) = best.unwrap_or_else(|| {
            let i = (0..self.free.len())
                .min_by_key(|&i| self.rows[self.free[i]].last_used)
                .expect("admit is called with a free row");
            (i, 0, false)
        });
        let row = self.free.swap_remove(i);
        let r = {
            let rr = &mut self.rows[row];
            if from_snap {
                self.model
                    .row_restore(row, &rr.snap)
                    .map(|()| rr.history = rr.snap_tokens.clone())
            } else if reused == 0 {
                rr.history.clear();
                self.model.row_reset(row)
            } else {
                Ok(())
            }
        };
        if let Err(e) = r {
            eprintln!("aien-serve-http: qwen36-batch: row {row} reset/restore failed: {e}");
            let rr = &mut self.rows[row];
            rr.history.clear();
            rr.snap_tokens.clear();
            let _ = job.out.send(Ev::Failed(e));
            self.free.push(row);
            return;
        }
        let snap_at = job
            .prompt
            .iter()
            .rposition(|&t| t == IM_START)
            .filter(|&k| k > 0 && k >= reused && self.rows[row].snap_tokens[..] != job.prompt[..k]);
        eprintln!(
            "aien-serve-http: qwen36-batch: join: prompt {plen} tok, reuse {reused} tok ({}, row {row}), queued {} ms, now {} rows",
            if reused == 0 { "cold" } else if from_snap { "snapshot" } else { "live" },
            job.t_queued.elapsed().as_millis(),
            active.len() + 1,
        );
        let observer = SamplingObserver::new(&job.params);
        let chip_greedy = self.chip_greedy && observer.is_greedy();
        active.push(Active {
            job,
            row,
            fed: reused,
            reused,
            snap_at,
            observer,
            emitted: 0,
            logits: None,
            next: None,
            chip_greedy,
            chip_next: None,
            t_admit: Instant::now(),
            prefill_ms: 0,
            t_decode: Instant::now(),
            steps: 0,
            rows_sum: 0,
        });
    }

    /// Samples the row's next token from its logits and emits it; `Some(reason)` when the row is
    /// done (the token is not fed). Mirrors the single door's loop exactly.
    fn sample_one(a: &mut Active, max_context: usize) -> Option<FinishReason> {
        let p = &a.job.params;
        let plen = a.job.prompt.len();
        if a.job.cancel.load(Ordering::SeqCst) {
            return Some(FinishReason::Cancelled);
        }
        if a.chip_next.is_none() && a.logits.is_none() {
            return None; // its sampled token has not been fed yet
        }
        if a.emitted >= p.max_new_tokens || plen + a.emitted >= max_context {
            return Some(FinishReason::Length);
        }
        let tok = match (a.chip_next.take(), a.logits.take()) {
            (Some(t), _) => {
                a.observer.note_greedy();
                t
            }
            (None, Some(logits)) => a.observer.sample(&logits, a.emitted),
            (None, None) => unreachable!("checked above"),
        };
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
        a.next = Some(tok);
        None
    }

    fn finish(&mut self, a: Active, finish: FinishReason, err: Option<String>) {
        let decode_ms = if a.prefilling() {
            0
        } else {
            a.t_decode.elapsed().as_millis() as u64
        };
        let steps = a.steps.max(1) as f64;
        eprintln!(
            "aien-serve-http: qwen36-batch: leave: row {}, prompt {} tok (reused {}), prefill {} ms, decode {} tok in {decode_ms} ms over {} beats, mean rows {:.2}, wall {:.1} ms/token, {:?}",
            a.row,
            a.job.prompt.len(),
            a.reused,
            a.prefill_ms,
            a.emitted,
            a.steps,
            a.rows_sum as f64 / steps,
            decode_ms as f64 / steps,
            finish,
        );
        let row = &mut self.rows[a.row];
        row.last_used = self.beat;
        if err.is_some() {
            row.history.clear(); // the state is unknown after a failed step
        }
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
        self.free.push(a.row);
    }
}

#[cfg(test)]
mod tests;

/// The chip lock the live server takes per busy spell. `AIEN_SERVE_GPU_LOCK` points it at a private
/// file so a live server never waits behind test jobs on `/tmp/aien-gb10.lock` (coordinator
/// 2026-10-21: Drake tests the motor live from Hermes; serving has priority over chip tests).
fn serve_lock_path() -> PathBuf {
    std::env::var_os("AIEN_SERVE_GPU_LOCK")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(GPU_LOCK_PATH))
}
