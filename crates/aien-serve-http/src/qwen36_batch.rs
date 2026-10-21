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
    need_bytes, Qwen38Omega, Qwen38Snapshot, RowFeed, RowOut, RowTest, IM_START,
};

use crate::engine::{
    Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, TestSwitches,
};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::q36_prefix::{PoolConfig, PoolSnap, PrefixPool};
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

/// L369: prompt tokens per prefill chunk (the chip's most) and the fewest worth a chunk.
pub const Q36_PREFILL_CHUNK: usize = 128;
pub const Q36_PREFILL_MIN: usize = 16;

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("aien-serve-http: {name}={v} is not a whole number, using {default}");
            default
        }),
        Err(_) => default,
    }
}

/// Fewest tokens between two of the server's checkpoint marks the door keeps (each checkpoint
/// holds a full DeltaNet state copy).
pub const MARK_GAP: usize = 512;

fn prompt_per_beat() -> Result<usize, String> {
    match std::env::var("AIEN_Q36_PROMPT_PER_BEAT") {
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(n),
            _ => Err(format!("AIEN_Q36_PROMPT_PER_BEAT={v}: a whole number >= 1")),
        },
        Err(_) => Ok(PROMPT_PER_BEAT),
    }
}

/// Memory kept free beyond the load at load time: [`KEEP_FREE`], or `AIEN_SERVE_KEEP_FREE_MIB` for a test server
/// loaded under q36-extra (L366: the kernel takes that one first on out of memory, never 18091).
fn keep_free() -> Result<u64, String> {
    match std::env::var("AIEN_SERVE_KEEP_FREE_MIB") {
        Ok(v) => v
            .parse::<u64>()
            .map(|m| m << 20)
            .map_err(|_| format!("AIEN_SERVE_KEEP_FREE_MIB={v}: a whole number of MiB")),
        Err(_) => Ok(KEEP_FREE),
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
    /// `step_rows` with the merge train's switches per row (`tests[i]` with `feed[i]`). The
    /// default refuses a non-plain switch: a stepper that cannot apply it must say so, never
    /// step as if it had.
    fn step_rows_test(
        &mut self,
        feed: &[RowFeed],
        tests: &[RowTest],
    ) -> Result<Vec<RowOut>, String> {
        if tests.iter().any(|t| !t.is_plain()) {
            return Err("this row stepper has no test switches".into());
        }
        self.step_rows(feed)
    }
    /// L369: a prompt chunk into row `r` in one call (the chip's prefill); the last token's greedy
    /// token and (when asked) logits. The state must end as `tokens.len()` single steps leave it;
    /// the default is exactly those steps.
    fn prefill_row(&mut self, r: usize, tokens: &[u32], logits: bool) -> Result<RowOut, String> {
        let mut last = None;
        for (i, &token) in tokens.iter().enumerate() {
            let feed = [RowFeed {
                row: r,
                token,
                logits: logits && i + 1 == tokens.len(),
            }];
            last = self.step_rows(&feed)?.pop();
        }
        last.ok_or_else(|| "prefill_row: no tokens".to_string())
    }
    /// A host copy of a row's state.
    type Snap: Default + PoolSnap;
    fn row_snapshot(&mut self, r: usize, snap: &mut Self::Snap) -> Result<(), String>;
    /// Puts a copy back into row `r` (any row; the K/V stays: see `row_kv_write`).
    fn row_restore(&mut self, r: usize, snap: &Self::Snap) -> Result<(), String>;
    /// Values one position of a row's K/V holds (the shared prompt cache's copies).
    fn row_kv_values(&self) -> usize;
    /// Copies row `r`'s K/V at positions `from .. from + count` into `out`.
    fn row_kv_read(
        &mut self,
        r: usize,
        from: usize,
        count: usize,
        out: &mut Vec<u16>,
    ) -> Result<(), String>;
    /// Writes a `row_kv_read` copy into row `r` at positions `from .. from + count`.
    fn row_kv_write(
        &mut self,
        r: usize,
        from: usize,
        count: usize,
        data: &[u16],
    ) -> Result<(), String>;
    /// Moves row `r`'s next position (the prompt cache's position mutant only).
    fn row_set_pos(&mut self, r: usize, pos: usize) -> Result<(), String>;
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
    fn step_rows_test(
        &mut self,
        feed: &[RowFeed],
        tests: &[RowTest],
    ) -> Result<Vec<RowOut>, String> {
        Qwen38Omega::step_rows_test(self, feed, tests)
    }
    fn prefill_row(&mut self, r: usize, tokens: &[u32], logits: bool) -> Result<RowOut, String> {
        Qwen38Omega::prefill_row(self, r, tokens, logits)
    }
    type Snap = Qwen38Snapshot;
    fn row_snapshot(&mut self, r: usize, snap: &mut Qwen38Snapshot) -> Result<(), String> {
        Qwen38Omega::row_snapshot(self, r, snap)
    }
    fn row_restore(&mut self, r: usize, snap: &Qwen38Snapshot) -> Result<(), String> {
        Qwen38Omega::row_restore(self, r, snap)
    }
    fn row_kv_values(&self) -> usize {
        Qwen38Omega::row_kv_values(self)
    }
    fn row_kv_read(
        &mut self,
        r: usize,
        from: usize,
        count: usize,
        out: &mut Vec<u16>,
    ) -> Result<(), String> {
        Qwen38Omega::row_kv_read(self, r, from, count, out)
    }
    fn row_kv_write(
        &mut self,
        r: usize,
        from: usize,
        count: usize,
        data: &[u16],
    ) -> Result<(), String> {
        Qwen38Omega::row_kv_write(self, r, from, count, data)
    }
    fn row_set_pos(&mut self, r: usize, pos: usize) -> Result<(), String> {
        Qwen38Omega::row_set_pos(self, r, pos)
    }
}

/// A request handed to the scheduler.
struct Job {
    prompt: Vec<u32>,
    /// Token positions worth a prompt cache checkpoint (tool boundaries, from the server).
    marks: Vec<usize>,
    params: GenerateParams,
    out: mpsc::SyncSender<Ev>,
    /// Set by the request thread when `on_token` says stop (client gone, stop string).
    cancel: Arc<AtomicBool>,
    t_queued: Instant,
    /// The merge train's switches for this request (empty on the served path).
    test: TestSwitches,
}

/// The merge train's switches the batch door applies per request (`Engine::test_switches_honored`).
#[cfg(feature = "aien-test-switches")]
const TEST_SWITCHES_HONORED: &[&str] = &["stroke", "exact_head", "mutant", "row_swap"];

/// The per-step row switches of a request (`stroke`, `exact_head`, `mutant`).
fn row_test(t: &TestSwitches) -> RowTest {
    RowTest {
        stroke: t.stroke.unwrap_or(true),
        exact_head: t.exact_head.unwrap_or(true),
        mutant: t.mutant.unwrap_or(0),
    }
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
    /// Positions (ascending) where the shared prompt cache saves the row's state on the way in.
    cps: Vec<usize>,
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
    /// The merge train's row swap still owed: at this decode step (from 1), at the first beat
    /// from it on with another conversation decoding, the two trade rows (negative control).
    swap_due: Option<u64>,
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
        let need = need_bytes(&c, 256, PREFILL_CHUNK);
        let keep = keep_free()?;
        let avail = mem_available();
        if avail < need + keep {
            return Err(format!(
                "needs {} MiB plus {} MiB kept free, MemAvailable {} MiB (another model may be loaded); refusing",
                need >> 20,
                keep >> 20,
                avail >> 20
            ));
        }
        // The prompt cache store binds the weights' digest: hashed beside the upload.
        let ck_digest = std::env::var_os("AIEN_Q36_PREFIX_DIR").map(|_| {
            let d = dir.to_path_buf();
            std::thread::spawn(move || crate::kv_persist::checkpoint_digest(&d).map(|(h, _)| h))
        });
        let lock = GpuLock::acquire_queued_at(&serve_lock_path(), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen36-batch: uploading {} layers, {slots} rows of {max_pos} positions (model needs {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let mut model = Qwen38Omega::open(dir, None, 256)?;
        let on_paper = model.row_bytes(max_pos);
        let rows_need = on_paper as u64 * slots as u64;
        let before = mem_available();
        if before < rows_need + keep {
            return Err(format!(
                "{slots} rows need {} MiB plus {} MiB kept free, MemAvailable {} MiB after the upload; refusing",
                rows_need >> 20,
                keep >> 20,
                before >> 20
            ));
        }
        model.add_rows(slots, max_pos)?;
        let after = mem_available();
        let measured = before.saturating_sub(after);
        let pool = open_pool(ck_digest, &mut model)?;
        drop(lock);
        eprintln!(
            "aien-serve-http: qwen36-batch engine ready in {:.1} s: context {max_context} tokens, {slots} rows, weights {:.2} GB resident, per-row state {:.1} MiB on paper, {:.1} MiB measured (MemAvailable {} MiB -> {} MiB), rss {:.2} GB, prompt tokens per beat {per_beat}",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            on_paper as f64 / (1u64 << 20) as f64,
            measured as f64 / slots as f64 / (1u64 << 20) as f64,
            before >> 20,
            after >> 20,
            rss_bytes() as f64 / 1e9,
        );
        let mut door = Self::with_stepper_pool(
            model,
            max_context,
            serve_lock_path(),
            REQUEST_LOCK_WAIT,
            per_beat,
            std::env::var(CHIP_GREEDY_ENV).as_deref() != Ok("0"),
            pool,
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
        let off =
            PoolConfig::from_lookup(|k| (k == "AIEN_Q36_PREFIX_POOL_GB").then(|| "0".into()))?;
        Self::with_stepper_pool(
            model,
            max_context,
            lock_path,
            lock_wait,
            prompt_per_beat,
            chip_greedy,
            PrefixPool::new(&off),
        )
    }

    /// [`Self::with_stepper`] with the shared prompt cache `pool` (sc L375).
    #[allow(clippy::too_many_arguments)]
    pub fn with_stepper_pool<S: RowStep>(
        model: S,
        max_context: usize,
        lock_path: PathBuf,
        lock_wait: Duration,
        prompt_per_beat: usize,
        chip_greedy: bool,
        pool: PrefixPool<S::Snap>,
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
            pf_chunk: env_usize("AIEN_Q36_PREFILL_CHUNK", Q36_PREFILL_CHUNK).min(Q36_PREFILL_CHUNK),
            pf_min: env_usize("AIEN_Q36_PREFILL_MIN", Q36_PREFILL_MIN).max(1),
            chip_greedy,
            beat: 0,
            pool,
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
        self.generate_marked(prompt, &[], p, &TestSwitches::default(), on_token)
    }

    fn generate_marked(
        &self,
        prompt: &[u32],
        marks: &[usize],
        p: &GenerateParams,
        test: &TestSwitches,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        #[cfg(feature = "aien-test-switches")]
        if let Some(bad) = test
            .names()
            .into_iter()
            .find(|n| !TEST_SWITCHES_HONORED.contains(n))
        {
            return Err(EngineError::Failed(format!(
                "aien_test.{bad} is not a switch of the batch door (honored: {TEST_SWITCHES_HONORED:?})"
            )));
        }
        #[cfg(not(feature = "aien-test-switches"))]
        if !test.is_empty() {
            return Err(EngineError::Failed(format!(
                "aien_test switches {:?}: this build has no test switches",
                test.names()
            )));
        }
        self.generate_test(prompt, marks, p, test.clone(), on_token)
    }

    #[cfg(feature = "aien-test-switches")]
    fn test_switches_honored(&self) -> &'static [&'static str] {
        TEST_SWITCHES_HONORED
    }

    #[cfg(feature = "aien-test-switches")]
    fn generate_with_test(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        test: &TestSwitches,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.generate_marked(prompt, &[], p, test, on_token)
    }
}

impl Qwen36BatchDoor {
    /// The request path; `test` is empty on the served path, `marks` are the server's
    /// checkpoint hints (tool boundaries).
    fn generate_test(
        &self,
        prompt: &[u32],
        marks: &[usize],
        p: &GenerateParams,
        test: TestSwitches,
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
            marks: marks.to_vec(),
            params: p.clone(),
            out: otx,
            cancel: Arc::clone(&cancel),
            t_queued: Instant::now(),
            test,
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
    /// L369 prefill: prompt tokens per chunk (`AIEN_Q36_PREFILL_CHUNK`, 0 = off) and the fewest
    /// left before a row reads a chunk instead of single tokens (`AIEN_Q36_PREFILL_MIN`).
    pf_chunk: usize,
    pf_min: usize,
    /// Greedy replies take the graph's token (`AIEN_SERVE_CHIP_GREEDY=0` turns it off).
    chip_greedy: bool,
    beat: u64,
    /// The shared prompt cache (sc L375).
    pool: PrefixPool<S::Snap>,
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
            Self::row_swaps_due(&mut active, &mut self.rows);
            // L369: a row with a long stretch of prompt left reads one chunk of it through the chip's
            // prefill this beat (its state ends as token by token would leave it). A chunk stops at
            // the row's snapshot point, so the copy at the last turn header is taken as before.
            // A row with merge-train switches (L384) feeds its prompt token by token through
            // `step_rows_test`, so every switch applies to every one of its steps.
            let mut chunked = vec![false; active.len()];
            let mut pf_err: Option<String> = None;
            if self.pf_chunk > 0 {
                for (k, a) in active.iter_mut().enumerate() {
                    if !a.prefilling()
                        || a.snap_at == Some(a.fed)
                        || !row_test(&a.job.test).is_plain()
                    {
                        continue;
                    }
                    let plen = a.job.prompt.len();
                    let stop = match a.snap_at {
                        Some(s) if s > a.fed => s,
                        _ => plen,
                    };
                    let n = (stop - a.fed).min(self.pf_chunk);
                    if n < self.pf_min {
                        continue;
                    }
                    let (from, end) = (a.fed, a.fed + n);
                    let logits = end == plen;
                    let t = Instant::now();
                    let r = self.model.prefill_row(a.row, &a.job.prompt[from..end], logits);
                    spell.wall_ns += t.elapsed().as_nanos() as u64;
                    spell.steps += 1;
                    spell.row_tokens += n as u64;
                    match r {
                        Ok(o) => {
                            spell.chip_ns += o.stats.chip_ns;
                            self.rows[a.row]
                                .history
                                .extend_from_slice(&a.job.prompt[from..end]);
                            a.fed = end;
                            if !a.prefilling() {
                                a.prefill_ms = a.t_admit.elapsed().as_millis() as u64;
                                a.t_decode = Instant::now();
                            }
                            if logits {
                                a.logits = o.logits;
                            }
                            chunked[k] = true;
                        }
                        Err(e) => {
                            pf_err = Some(format!("prefill of row {} failed: {e}", a.row));
                            break;
                        }
                    }
                }
            }
            if let Some(e) = pf_err {
                eprintln!("aien-serve-http: qwen36-batch: {e}");
                for a in active.drain(..) {
                    self.finish(a, FinishReason::Length, Some(e.clone()));
                }
                continue;
            }
            // One beat: every decoding row feeds its sampled token in the first step; the rows
            // still feeding their prompt feed up to `per_beat` prompt tokens, one per step.
            for sub in 0..self.per_beat {
                // The shared prompt cache's checkpoints, before the token at that position goes in.
                for a in active.iter_mut() {
                    if a.prefilling() && a.cps.first() == Some(&a.fed) {
                        a.cps.remove(0);
                        self.save_checkpoint(a.row, &a.job.prompt[..a.fed]);
                    }
                }
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
                let mut tests = Vec::with_capacity(active.len());
                for (k, a) in active.iter().enumerate() {
                    if a.prefilling() && !chunked[k] {
                        let plen = a.job.prompt.len();
                        feed.push(RowFeed {
                            row: a.row,
                            token: a.job.prompt[a.fed],
                            logits: a.fed + 1 == plen,
                        });
                        who.push(k);
                        tests.push(row_test(&a.job.test));
                    } else if sub == 0 && !a.prefilling() {
                        if let Some(t) = a.next {
                            feed.push(RowFeed {
                                row: a.row,
                                token: t,
                                logits: !a.chip_greedy,
                            });
                            who.push(k);
                            tests.push(row_test(&a.job.test));
                        }
                    }
                }
                if feed.is_empty() {
                    break;
                }
                let t = Instant::now();
                // The served path calls `step_rows` exactly as before; only a beat with a
                // switched row goes through `step_rows_test`.
                let r = if tests.iter().all(RowTest::is_plain) {
                    self.model.step_rows(&feed)
                } else {
                    self.model.step_rows_test(&feed, &tests)
                };
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

    /// Seats the request on a free row: the longest state the prompt continues among the free
    /// rows (live tokens, or the copy at the last turn header) and the shared prompt cache (sc
    /// L375), else the least recently used row, emptied. The single door's reuse rule
    /// ([`aien_inference_abi::qwen38_omega::plan_reuse`]) per row, plus the pool.
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
        // The pool, when it reaches further than every free row.
        let from_pool = if self.pool.on() {
            let hit = self
                .pool
                .longest(&job.prompt, plen - 1)
                .filter(|&(_, n)| best.is_none_or(|(_, b, _)| n > b));
            self.pool.note_lookup(hit.is_some());
            hit
        } else {
            None
        };
        let lru = (0..self.free.len())
            .min_by_key(|&i| self.rows[self.free[i]].last_used)
            .expect("admit is called with a free row");
        let (i, reused, from_snap) = match (from_pool, best) {
            (Some((_, n)), _) => (lru, n, false),
            (None, Some(b)) => b,
            (None, None) => (lru, 0, false),
        };
        let row = self.free.swap_remove(i);
        let r = if let Some((id, n)) = from_pool {
            self.restore_from_pool(row, id, &job.prompt[..n])
        } else {
            let rr = &mut self.rows[row];
            if from_snap {
                self.model
                    .row_restore(row, &rr.snap)
                    .map(|()| rr.history = rr.snap_tokens.clone())
            } else if reused == 0 {
                // The new prompt rewrites the K/V the row's own copy relies on.
                rr.history.clear();
                rr.snap_tokens.clear();
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
        let cps = self.checkpoints(&job, reused);
        eprintln!(
            "aien-serve-http: qwen36-batch: join: prompt {plen} tok, reuse {reused} tok ({}, row {row}), {} cache checkpoints, queued {} ms, now {} rows",
            if from_pool.is_some() {
                "prompt cache"
            } else if reused == 0 {
                "cold"
            } else if from_snap {
                "snapshot"
            } else {
                "live"
            },
            cps.len(),
            job.t_queued.elapsed().as_millis(),
            active.len() + 1,
        );
        let observer = SamplingObserver::new(&job.params);
        let job_swap = job.test.row_swap.filter(|&k| k >= 1);
        let chip_greedy = self.chip_greedy && observer.is_greedy();
        active.push(Active {
            job,
            row,
            fed: reused,
            reused,
            snap_at,
            cps,
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
            swap_due: job_swap,
        });
    }

    /// The merge train's `row_swap: k` (negative control, per request): a conversation whose
    /// `k`th decode step is up trades rows with the first other conversation that is decoding,
    /// so both continue from each other's state; the per-row identity gate must go red for the
    /// two. Fires once per request; waits while it is the only one decoding. Both rows' kept
    /// tokens are forgotten (their states are no longer what their histories say).
    fn row_swaps_due(active: &mut [Active], rows: &mut [Row<S::Snap>]) {
        for i in 0..active.len() {
            let Some(k) = active[i].swap_due else {
                continue;
            };
            if active[i].prefilling() || active[i].steps + 1 < k {
                continue;
            }
            let Some(j) = (0..active.len()).find(|&j| j != i && !active[j].prefilling()) else {
                continue;
            };
            active[i].swap_due = None;
            let (ri, rj) = (active[i].row, active[j].row);
            active[i].row = rj;
            active[j].row = ri;
            for r in [ri, rj] {
                rows[r].history.clear();
                rows[r].snap_tokens.clear();
            }
            eprintln!(
                "aien-serve-http: qwen36-batch: TEST ROW SWAP (aien_test.row_swap {k}): rows {ri} and {rj} trade conversations (negative control; never serve this)"
            );
        }
    }

    /// Copies pool node `id` (the state after `tokens`) into `row`: the K/V pieces root first,
    /// then the DeltaNet state and position. On an error the row's state is unknown (the caller
    /// empties its bookkeeping; the next cold join resets it).
    fn restore_from_pool(&mut self, row: usize, id: u64, tokens: &[u32]) -> Result<(), String> {
        let t = Instant::now();
        let rr = &mut self.rows[row];
        rr.history.clear();
        rr.snap_tokens.clear(); // the K/V below rewrites what the row's own copy relies on
        let vals = self.model.row_kv_values();
        let rs = self.pool.restore(id).ok_or("prompt cache: node gone")?;
        if rs.pos != tokens.len() {
            return Err(format!(
                "prompt cache: node holds {} tokens, not {}",
                rs.pos,
                tokens.len()
            ));
        }
        let mut bytes = 0usize;
        for &(from, kv) in &rs.kv {
            let count = if vals == 0 { 0 } else { kv.len() / vals };
            if count * vals != kv.len() {
                return Err("prompt cache: K/V piece of another shape".into());
            }
            self.model.row_kv_write(row, from, count, kv)?;
            bytes += kv.len() * 2;
        }
        self.model.row_restore(row, rs.snap)?;
        bytes += rs.snap.host_bytes();
        if self.pool.mutant() {
            // The gate's control: the row continues one position early.
            self.model.row_set_pos(row, tokens.len() - 1)?;
        }
        self.rows[row].history = tokens.to_vec();
        eprintln!(
            "aien-serve-http: qwen36-batch: prompt cache restore: {} tok into row {row}, {:.1} MiB in {} ms{}",
            tokens.len(),
            bytes as f64 / (1u64 << 20) as f64,
            t.elapsed().as_millis(),
            if self.pool.mutant() { " (POSITION MUTANT)" } else { "" },
        );
        Ok(())
    }

    /// Where the pool saves this prompt's state on the way in (ascending, past `reused`, before
    /// the last prompt token): every `every` tokens, the server's marks (tool boundaries, at
    /// least `MARK_GAP` apart), the end of the first turn, the last turn header, and where the
    /// prompt leaves the pool's longest match (the next prompt that changes there restores up to
    /// it: coordinator 12:10Z, diff on arrival).
    fn checkpoints(&self, job: &Job, reused: usize) -> Vec<usize> {
        if !self.pool.on() {
            return Vec::new();
        }
        let (p, plen) = (&job.prompt, job.prompt.len());
        let every = self.pool.every();
        let mut v: Vec<usize> = (1..=plen / every).map(|k| k * every).collect();
        let mut last_mark = 0usize;
        for &m in &job.marks {
            if m >= last_mark + MARK_GAP {
                v.push(m);
                last_mark = m;
            }
        }
        let mut heads = p
            .iter()
            .enumerate()
            .filter(|&(k, &t)| k > 0 && t == IM_START)
            .map(|(k, _)| k);
        v.extend(heads.next()); // the end of the first turn
        v.extend(p.iter().rposition(|&t| t == IM_START).filter(|&k| k > 0));
        v.extend(self.pool.divergence(p));
        v.retain(|&x| x > reused && x < plen);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Saves `row`'s state (it holds exactly `tokens`) into the pool: the K/V since the deepest
    /// node `tokens` continues, the DeltaNet state and position.
    fn save_checkpoint(&mut self, row: usize, tokens: &[u32]) {
        if self.pool.touch(tokens) {
            return;
        }
        let t = Instant::now();
        let (parent, from) = match self.pool.parent_for(tokens) {
            Some((id, n)) => (Some(id), n),
            None => (None, 0),
        };
        let mut kv = Vec::new();
        let mut snap = S::Snap::default();
        let r = self
            .model
            .row_kv_read(row, from, tokens.len() - from, &mut kv)
            .and_then(|()| self.model.row_snapshot(row, &mut snap));
        match r {
            Ok(()) => {
                let bytes = kv.len() * 2 + snap.host_bytes();
                let kept = self.pool.insert(tokens.to_vec(), parent, kv, snap);
                let c = self.pool.counters();
                eprintln!(
                    "aien-serve-http: qwen36-batch: prompt cache save: {} tok (K/V from {from}) from row {row}, {:.1} MiB in {} ms{}; pool {} nodes {:.1} MiB",
                    tokens.len(),
                    bytes as f64 / (1u64 << 20) as f64,
                    t.elapsed().as_millis(),
                    if kept { "" } else { ", over budget, dropped" },
                    c.nodes,
                    c.bytes as f64 / (1u64 << 20) as f64,
                );
            }
            Err(e) => eprintln!(
                "aien-serve-http: qwen36-batch: prompt cache save at {} tok from row {row} failed: {e}",
                tokens.len()
            ),
        }
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

/// The shared prompt cache from the environment (sc L375), with its sealed store loaded (before
/// the door takes requests) when `AIEN_Q36_PREFIX_DIR` is set. A store that cannot open, or any
/// mutant or test switch, leaves the pool in memory only. Runs under the load lock (the state
/// probe reads row 0 on the chip).
fn open_pool(
    ck_digest: Option<std::thread::JoinHandle<Result<[u8; 32], String>>>,
    model: &mut Qwen38Omega,
) -> Result<PrefixPool<Qwen38Snapshot>, String> {
    use crate::q36_prefix::{namespace, numeric_env};
    let cfg = PoolConfig::from_env()?;
    let mut pool = PrefixPool::new(&cfg);
    if !pool.on() {
        eprintln!("aien-serve-http: qwen36-batch: prompt cache off (AIEN_Q36_PREFIX_POOL_GB=0)");
        return Ok(pool);
    }
    let kv_values = model.row_kv_values();
    let mut probe = Qwen38Snapshot::default();
    model.row_snapshot(0, &mut probe)?;
    let snap_bytes = probe.bytes();
    eprintln!(
        "aien-serve-http: qwen36-batch: prompt cache on: budget {:.1} GiB, checkpoint every {} tok, K/V {} B per token, state {:.1} MiB per checkpoint{}",
        cfg.budget as f64 / (1u64 << 30) as f64,
        cfg.every,
        kv_values * 2,
        snap_bytes as f64 / (1u64 << 20) as f64,
        if cfg.mutant { ", POSITION MUTANT" } else { "" },
    );
    let Some(sdir) = cfg.dir.as_ref() else {
        return Ok(pool);
    };
    let off = |why: String| {
        eprintln!(
            "aien-serve-http: qwen36-batch: prompt cache store {} off: {why}; the pool stays in memory",
            sdir.display()
        );
    };
    let env = numeric_env(std::env::vars());
    if cfg.mutant
        || env
            .iter()
            .any(|(k, _)| k.contains("MUTANT") || k.contains("_TEST_"))
    {
        off("a mutant or test switch is set (its states are never stored)".into());
        return Ok(pool);
    }
    let lib = aien_omega_gpu::native_identity().archive_sha256;
    if lib.is_empty() {
        off("the linked libomega_gpu.a has no sha256".into());
        return Ok(pool);
    }
    let key = match aien_kv_store::seal::SealKey::take_from_process_env() {
        Ok(k) => k,
        Err(e) => {
            off(format!("{e}"));
            return Ok(pool);
        }
    };
    let ck = match ck_digest.map(|h| h.join()) {
        Some(Ok(Ok(h))) => h,
        Some(Ok(Err(e))) => {
            off(format!("checkpoint digest: {e}"));
            return Ok(pool);
        }
        _ => {
            off("checkpoint digest thread failed".into());
            return Ok(pool);
        }
    };
    let ns = namespace(&ck, lib, kv_values, snap_bytes, &env);
    if let Err(e) = pool.open_store(sdir, key, ns, kv_values, |s| s.bytes() == snap_bytes) {
        off(e);
    }
    Ok(pool)
}
