//! Qwen3.8-27B behind the door (sc #642 serving): the whole model resident on AIEN's own Omega
//! kernels (64 layers, fused FP8 projections, bf16 lm_head), prompts through omega's prefill,
//! replies through the chip-proven decode graph, next turn resumed from the live state or the
//! snapshot taken before the prompt's last turn header (`aien_inference_abi::qwen38_omega::
//! Qwen38Session`).
//!
//! Document memory (cascade/document-memory, [`crate::conv_store`]): with the store on (the
//! default; `AIEN_CONV_STORE_OFF=1` turns it off), every long prompt's state up to its last turn
//! header is saved (KV rows plus DeltaNet checkpoints every `checkpoint_every` tokens and at the
//! turn header), so a later prompt that starts with the same tokens resumes from the store even
//! after other clients replaced the live conversation. `POST /v1/documents` reads a document in
//! the background, a checkpoint every `checkpoint_every` tokens, giving the chip back between
//! pieces.
//!
//! Chip etiquette as the 30B door ([`crate::omega_engine`]): `/tmp/aien-gb10.lock` is taken for
//! the upload and around each generation or ingest piece, never while idle. Persistent KV parks
//! (`AIEN_KV_STORE_DIR`) are not offered: they hold attention KV only, not the DeltaNet state.
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{
    need_bytes, Qwen38Omega, Qwen38Session, Qwen38Snapshot, Reuse, IM_START,
};

use crate::conv_store::{
    self, common_prefix_len, CkptKind, ConvStore, Entry, EntryStatus, Match, Shape, StoreConfig,
};
use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::sampling::SamplingObserver;

/// Model id the door answers to unless `--model-id` says otherwise.
pub const DEFAULT_MODEL_ID: &str = "qwen3.8-27b-fp8";
/// Prefill chunk length (the 2026-10-18 TTFT receipts ran 4096).
pub const PREFILL_CHUNK: usize = 4096;
/// How long a request waits in the chip lock's queue before answering busy (the hive's chip
/// jobs queue with `flock(1)` and run for seconds to minutes each).
pub const REQUEST_LOCK_WAIT: Duration = Duration::from_secs(300);
/// How long the upload waits for the chip lock (other jobs' chip runs take minutes; a request
/// waits [`REQUEST_LOCK_WAIT`]).
pub const LOAD_LOCK_WAIT: Duration = Duration::from_secs(1800);
/// Memory kept free beyond [`need_bytes`] at load (other jobs share the box).
pub const KEEP_FREE: u64 = 8 << 30;
/// Prompts shorter than this are not saved (short chats are cheap to recompute; the
/// translator's traffic would otherwise fill the store). `AIEN_CONV_MIN_TOKENS`.
pub const DEFAULT_MIN_TOKENS: usize = 2048;

/// `config.json` says `model_type: qwen3_5` (the Qwen3.8 family this door serves).
pub fn is_qwen38_dir(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|v| v["model_type"].as_str() == Some("qwen3_5"))
}

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// This process's resident set in bytes (0 when unreadable).
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The store shape of the loaded model.
pub fn store_shape(m: &Qwen38Omega) -> Shape {
    Shape {
        n_attn: m.attn_layers(),
        n_gdn: m.gdn_layers(),
        tail_floats: aien_omega_q38::GDN_TAIL_FLOATS,
        state_floats: aien_omega_q38::GDN_STATE_FLOATS,
        max_pos: m.max_pos(),
    }
}

/// Everything that can change the meaning of a stored byte.
pub fn store_namespace(dir: &Path, model_id: &str, shape: &Shape) -> Result<String, String> {
    let (digest, kind) = crate::kv_persist::checkpoint_digest(dir)?;
    let b = crate::kv_persist::OmegaBuild::linked();
    let digest: String = digest.iter().map(|x| format!("{x:02x}")).collect();
    let fp8_head = std::env::var("AIEN_Q38_FP8_HEAD").unwrap_or_default();
    let max_pos = shape.max_pos.to_string();
    let chunk = PREFILL_CHUNK.to_string();
    Ok(conv_store::namespace_digest(&[
        ("model_id", model_id),
        ("checkpoint", &digest),
        ("checkpoint_kind", kind),
        ("omega_commit", &b.omega_commit),
        ("gpu_archive_sha256", &b.gpu_archive_sha256),
        ("sc_commit", &b.sc_commit),
        ("max_pos", &max_pos),
        ("prefill_chunk", &chunk),
        ("AIEN_Q38_FP8_HEAD", &fp8_head),
        ("kv", "bf16 rows as written by the kernels"),
    ]))
}

// ---------------------------------------------------------------------------
// Device <-> store copies (shared by the request path and the ingest thread)
// ---------------------------------------------------------------------------

/// Times of one restore from the store.
#[derive(Debug, Default, Clone)]
pub struct RestoreStats {
    pub pos: usize,
    /// Of `pos` rows, already on the device (same token prefix): not written.
    pub kept_rows: usize,
    /// Making the entry resident (loading it from NVMe when spilled).
    pub load_ms: f64,
    pub kv_write_ms: f64,
    pub state_ms: f64,
    pub from_nvme: bool,
}

/// Puts conversation `m.key`'s KV rows `0 .. m.pos` and its checkpoint at `m.pos` on the device
/// and makes them the session's live state. Rows the device already holds for the same token
/// prefix are not rewritten. The caller unpins the entry.
pub fn restore_from_store(
    s: &mut Qwen38Session,
    store: &ConvStore,
    m: &Match,
) -> Result<RestoreStats, String> {
    let t_all = Instant::now();
    let from_nvme = m.residency == conv_store::Residency::Spilled;
    let r = store.with_entry(&m.key, |e: &mut Entry| -> Result<RestoreStats, String> {
        let mut st = RestoreStats {
            pos: m.pos,
            from_nvme,
            load_ms: ms(t_all),
            ..Default::default()
        };
        let kv = e.kv.as_ref().ok_or("restore: entry not resident")?;
        if kv.rows() < m.pos {
            return Err(format!(
                "restore: {} rows stored, {} needed",
                kv.rows(),
                m.pos
            ));
        }
        let ck = e
            .checkpoint_at(m.pos)
            .cloned()
            .ok_or_else(|| format!("restore: no checkpoint at {}", m.pos))?;
        // The device's rows are trusted only while the session's history is (pos == len).
        let keep = if s.model().pos == s.history().len() {
            common_prefix_len(s.history(), &e.tokens).min(m.pos)
        } else {
            0
        };
        st.kept_rows = keep;
        // TEST ONLY (negative controls of the chip gate): AIEN_CONV_TEST_MUTANT=state skips the
        // DeltaNet checkpoint (the state stays zero), =kv skips writing the KV rows.
        let mutant = std::env::var("AIEN_CONV_TEST_MUTANT").unwrap_or_default();
        let t = Instant::now();
        if keep < m.pos && mutant != "kv" {
            let model = s.model_mut();
            for layer in 0..model.attn_layers() {
                for head in 0..conv_store::KV_HEADS {
                    let k = kv.plane_words(layer, head, 0, keep, m.pos);
                    let v = kv.plane_words(layer, head, 1, keep, m.pos);
                    model.kv_write_rows(layer, head, keep, k, v)?;
                }
            }
        }
        st.kv_write_ms = ms(t);
        let t = Instant::now();
        let tokens = e.tokens[..m.pos].to_vec();
        if mutant == "state" {
            let z = Qwen38Snapshot::from_parts(
                m.pos,
                vec![0.0; ck.snap.tails().len()],
                vec![0.0; ck.snap.states().len()],
            );
            s.adopt(&tokens, &z)?;
        } else {
            s.adopt(&tokens, &ck.snap)?;
        }
        st.state_ms = ms(t);
        Ok(st)
    });
    match r {
        Ok(Some(r)) => r,
        Ok(None) => Err(format!("restore: conversation {} is gone", m.key)),
        Err(e) => Err(e),
    }
}

/// Copies device KV rows `e.kv_rows .. to` (the session's history must cover them) into `e`.
fn export_rows(model: &mut Qwen38Omega, e: &mut Entry, to: usize) -> Result<(), String> {
    let from = e.kv_rows;
    if to <= from {
        return Ok(());
    }
    let kv = e.kv.as_mut().ok_or("export: entry not resident")?;
    if kv.rows() != from {
        return Err(format!(
            "export: entry holds {} rows, record says {from}",
            kv.rows()
        ));
    }
    let n = to - from;
    let w = aien_omega_q38::HEAD_DIM;
    let (mut kb, mut vb) = (vec![0u16; n * w], vec![0u16; n * w]);
    kv.append_rows(n, |layer, head, plane, out| {
        if plane == 0 {
            model.kv_read_rows(layer, head, from, to, &mut kb, &mut vb)?;
            out.copy_from_slice(&kb);
        } else {
            out.copy_from_slice(&vb);
        }
        Ok(())
    })?;
    e.kv_rows = to;
    e.dirty = true;
    Ok(())
}

/// What one save cost.
#[derive(Debug, Default, Clone)]
pub struct SaveStats {
    pub key: Option<String>,
    pub rows_exported: usize,
    pub export_ms: f64,
    pub checkpoints: usize,
    pub new_entry: bool,
}

/// Saves the session's state up to `upto` tokens of its history into the store: extends the
/// conversation `target` when its token record is a prefix of the history, else a new
/// conversation. `ckpts` are DeltaNet checkpoints taken during the feed (positions `<= upto`),
/// `end_ckpt` the one at `upto`.
pub fn save_to_store(
    s: &mut Qwen38Session,
    store: &ConvStore,
    target: Option<&str>,
    upto: usize,
    ckpts: Vec<Qwen38Snapshot>,
    end_ckpt: (CkptKind, Qwen38Snapshot),
    status: EntryStatus,
) -> Result<SaveStats, String> {
    let hist = s.history()[..upto].to_vec();
    let mut st = SaveStats::default();
    let extend = match target {
        Some(key) => store
            .with_entry(key, |e| {
                hist.starts_with(&e.tokens) && e.kv_rows == e.tokens.len()
            })?
            .unwrap_or(false),
        None => false,
    };
    let key = match (extend, target) {
        (true, Some(k)) => k.to_string(),
        _ => {
            st.new_entry = true;
            store.create(&[], None, EntryStatus::Ready)
        }
    };
    let t = Instant::now();
    let r = store.with_entry(&key, |e| -> Result<(), String> {
        let before = e.kv_rows;
        export_rows(s.model_mut(), e, upto)?;
        st.rows_exported = upto - before;
        e.tokens = hist.clone();
        for c in ckpts {
            if c.pos <= upto && c.pos > 0 {
                e.put_checkpoint(CkptKind::Chunk, c);
                st.checkpoints += 1;
            }
        }
        let (kind, snap) = end_ckpt;
        if snap.pos == upto && upto > 0 {
            e.put_checkpoint(kind, snap);
            st.checkpoints += 1;
        }
        e.status = status;
        e.dirty = true;
        Ok(())
    });
    st.export_ms = ms(t);
    store.unpin();
    match r {
        Ok(Some(Ok(()))) => {
            st.key = Some(key);
            Ok(st)
        }
        Ok(Some(Err(e))) | Err(e) => {
            if st.new_entry {
                store.remove(&key);
            }
            Err(e)
        }
        Ok(None) => Err("save: entry vanished".into()),
    }
}

/// A checkpointing chunk hook: a DeltaNet snapshot each time the fed history lands on a
/// multiple of `every` (at most `limit` tokens).
fn checkpointer(
    every: usize,
    limit: usize,
    out: &mut Vec<Qwen38Snapshot>,
) -> impl FnMut(&mut Qwen38Omega, &[u32]) -> Result<(), String> + '_ {
    move |m: &mut Qwen38Omega, h: &[u32]| {
        let n = h.len();
        if every > 0 && n > 0 && n <= limit && n.is_multiple_of(every) {
            let mut snap = Qwen38Snapshot::default();
            m.snapshot(&mut snap)?;
            out.push(snap);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The door
// ---------------------------------------------------------------------------

struct IngestJob {
    key: String,
    tokens: Vec<u32>,
}

struct DocMem {
    store: Arc<ConvStore>,
    min_tokens: usize,
    ingest: Mutex<mpsc::Sender<IngestJob>>,
}

pub struct Qwen38Door {
    session: Arc<Mutex<Qwen38Session>>,
    max_context: usize,
    lock_path: PathBuf,
    lock_wait: Duration,
    mem: Option<DocMem>,
}

impl Qwen38Door {
    /// Uploads the checkpoint at `dir` with a KV cache for `max_context` positions (rounded up
    /// to whole 256-position passes), under the GPU lock. The document memory is set up from
    /// the `AIEN_CONV_*` environment.
    pub fn load(dir: &Path, max_context: usize) -> Result<Self, String> {
        Self::load_as(dir, max_context, DEFAULT_MODEL_ID)
    }

    pub fn load_as(dir: &Path, max_context: usize, model_id: &str) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        if !aien_omega_q38::has_prefill() {
            return Err("this binary has no qwen38 prefill entries linked (build against omega qwen38/prefill-graph)".into());
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
        let _lock = GpuLock::acquire_queued_at(Path::new(GPU_LOCK_PATH), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen38: uploading {} layers, KV {max_pos} positions (need {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let model = Qwen38Omega::open(dir, None, max_pos)?;
        eprintln!(
            "aien-serve-http: qwen38 engine ready in {:.1} s: context {max_context} tokens, weights {:.2} GB resident, KV {:.2} GB, rss {:.2} GB, prefill chunk {PREFILL_CHUNK}, spin_us {:?}",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            model.kv_bytes() as f64 / 1e9,
            rss_bytes() as f64 / 1e9,
            aien_omega_gpu::spin_us(),
        );
        let shape = store_shape(&model);
        let session = Arc::new(Mutex::new(Qwen38Session::new(model, PREFILL_CHUNK)));
        let lock_path = PathBuf::from(GPU_LOCK_PATH);
        let mem = match conv_store::settings_from_env(|k| std::env::var(k).ok()) {
            None => {
                eprintln!("aien-serve-http: conv store off ({})", conv_store::OFF_ENV);
                None
            }
            Some((sdir, ram, disk, every)) => {
                let namespace = store_namespace(dir, model_id, &shape)?;
                let store = ConvStore::open(StoreConfig {
                    dir: sdir,
                    ram_budget: ram,
                    disk_budget: disk,
                    checkpoint_every: every,
                    namespace,
                    shape,
                    io_threads: conv_store::DEFAULT_IO_THREADS,
                    direct_io: std::env::var("AIEN_CONV_DIRECT_IO").as_deref() != Ok("0"),
                })?;
                let min_tokens = std::env::var("AIEN_CONV_MIN_TOKENS")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(DEFAULT_MIN_TOKENS);
                let (tx, rx) = mpsc::channel::<IngestJob>();
                let (s2, st2, lp) = (Arc::clone(&session), Arc::clone(&store), lock_path.clone());
                std::thread::Builder::new()
                    .name("qwen38-ingest".into())
                    .spawn(move || ingest_loop(rx, s2, st2, lp))
                    .map_err(|e| format!("ingest thread: {e}"))?;
                Some(DocMem {
                    store,
                    min_tokens,
                    ingest: Mutex::new(tx),
                })
            }
        };
        Ok(Self {
            session,
            max_context,
            lock_path,
            lock_wait: REQUEST_LOCK_WAIT,
            mem,
        })
    }
}

/// The background reader: one piece of `checkpoint_every` tokens per chip lock hold.
fn ingest_loop(
    rx: mpsc::Receiver<IngestJob>,
    session: Arc<Mutex<Qwen38Session>>,
    store: Arc<ConvStore>,
    lock_path: PathBuf,
) {
    while let Ok(job) = rx.recv() {
        let t0 = Instant::now();
        match ingest_one(&job, &session, &store, &lock_path) {
            Ok(()) => {
                store.set_status(&job.key, EntryStatus::Ready);
                eprintln!(
                    "aien-serve-http: conv store: ingest {} done: {} tokens in {:.1} s",
                    job.key,
                    job.tokens.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!(
                    "aien-serve-http: conv store: ingest {} failed: {e}",
                    job.key
                );
                store.set_status(&job.key, EntryStatus::Failed(e));
            }
        }
    }
}

fn ingest_one(
    job: &IngestJob,
    session: &Mutex<Qwen38Session>,
    store: &ConvStore,
    lock_path: &Path,
) -> Result<(), String> {
    let every = store.checkpoint_every().max(1);
    let total = job.tokens.len();
    let mut done = 0usize;
    while done < total {
        let t_lock = Instant::now();
        let _lock = GpuLock::acquire_queued_at(lock_path, LOAD_LOCK_WAIT)?;
        let lock_ms = ms(t_lock);
        let mut s = session.lock().map_err(|_| "session poisoned".to_string())?;
        let t0 = Instant::now();
        // Continue the live state when it is exactly this document's first `done` tokens, else
        // restore our own last piece from the store.
        let live_ok = s.history().len() == done
            && job.tokens.starts_with(s.history())
            && s.model().pos == done;
        let mut restore_ms = 0.0;
        if !live_ok {
            if done == 0 {
                s.forget();
            } else {
                let t = Instant::now();
                let m = Match {
                    key: job.key.clone(),
                    pos: done,
                    kind: CkptKind::Chunk,
                    shared: done,
                    entry_len: done,
                    residency: conv_store::Residency::Resident,
                };
                let r = restore_from_store(&mut s, store, &m);
                store.unpin();
                r?;
                restore_ms = ms(t);
            }
        }
        let end = (done / every + 1).saturating_mul(every).min(total);
        let t = Instant::now();
        s.feed_prefix(&job.tokens[..end], &mut |_, _| Ok(()))?;
        let feed_ms = ms(t);
        let mut snap = Qwen38Snapshot::default();
        s.model_mut().snapshot(&mut snap)?;
        let last = end == total;
        let kind = if last {
            CkptKind::DocumentEnd
        } else {
            CkptKind::Chunk
        };
        let status = if last {
            EntryStatus::Ready
        } else {
            EntryStatus::Reading { done: end, total }
        };
        let sv = save_to_store(
            &mut s,
            store,
            Some(&job.key),
            end,
            Vec::new(),
            (kind, snap),
            status,
        )?;
        if sv.new_entry {
            if let Some(k) = sv.key {
                store.remove(&k);
            }
            return Err("ingest: the saved conversation no longer matches the document".into());
        }
        eprintln!(
            "aien-serve-http: conv store: ingest {} piece {done}..{end} of {total}: lock wait {lock_ms:.0} ms, restore {restore_ms:.0} ms, feed {feed_ms:.0} ms, save {:.0} ms ({} rows), wall {:.0} ms",
            job.key,
            sv.export_ms,
            sv.rows_exported,
            ms(t0)
        );
        done = end;
    }
    Ok(())
}

/// The prompt's last turn header (where the next turn of this chat resumes), if any.
fn turn_header(prompt: &[u32]) -> Option<usize> {
    prompt
        .iter()
        .rposition(|&t| t == IM_START)
        .filter(|&k| k > 0)
}

impl Engine for Qwen38Door {
    fn kind(&self) -> &str {
        "omega-qwen38"
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.generate_mode(prompt, p, false, on_token)
    }

    fn generate_mode(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        cold: bool,
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
        let t_lock = Instant::now();
        let lock = GpuLock::acquire_queued_at(&self.lock_path, self.lock_wait)
            .map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        let mut s = self
            .session
            .lock()
            .map_err(|_| EngineError::Failed("qwen38 session poisoned".into()))?;
        let t0 = Instant::now();
        // `cold`: the recomputed reference: no live state, no store, nothing saved.
        let mem = self.mem.as_ref().filter(|_| !cold);
        if cold {
            s.forget();
        }
        let mut store_line = String::new();
        let mut target: Option<String> = None;
        let mut matched_pos = 0usize;
        if let Some(dm) = mem {
            let own = aien_inference_abi::qwen38_omega::plan_reuse(
                s.history(),
                s.snapshot_tokens(),
                prompt,
            )
            .tokens();
            let matched = dm.store.best_match(prompt);
            if let Some(m) = matched.as_ref() {
                target = Some(m.key.clone());
                matched_pos = m.pos;
                if m.pos > own {
                    match restore_from_store(&mut s, &dm.store, m) {
                        Ok(r) => {
                            store_line = format!(
                                ", store restore {} tok from {} ({} rows kept; load {:.1} ms, kv write {:.1} ms, state {:.1} ms, total {:.1} ms)",
                                r.pos,
                                if r.from_nvme { "NVMe" } else { "RAM" },
                                r.kept_rows,
                                r.load_ms,
                                r.kv_write_ms,
                                r.state_ms,
                                ms(t0)
                            );
                        }
                        Err(e) => {
                            eprintln!("aien-serve-http: conv store: restore failed ({e}); feeding from the session's own state");
                            s.forget();
                        }
                    }
                    dm.store.unpin();
                }
            }
        }
        let every = mem.map_or(0, |dm| dm.store.checkpoint_every());
        let k = turn_header(prompt).unwrap_or(0);
        let mut ckpts = Vec::new();
        let r = {
            let mut hook = checkpointer(every, k, &mut ckpts);
            s.start_with(prompt, &mut hook)
        };
        let (mut logits, fs) = r.map_err(EngineError::Failed)?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut observer = SamplingObserver::new(p);
        let (mut emitted, mut chip_ns) = (0usize, 0u64);
        let finish = loop {
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let tok = observer.sample(&logits, emitted);
            if p.stop_token_ids.contains(&tok) {
                break FinishReason::Stop;
            }
            emitted += 1;
            if !on_token(tok) {
                break FinishReason::Cancelled;
            }
            // The last allowed token is not fed: nothing would read its logits.
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let (next, ns) = s.advance(tok).map_err(EngineError::Failed)?;
            logits = next;
            chip_ns += ns;
        };
        let decode_ms = t1.elapsed().as_millis() as u64;
        // Save the state up to the turn header (the snapshot the session took there).
        let mut save_line = String::new();
        if let Some(dm) = mem {
            let snap_ok = s
                .snapshot_tokens()
                .is_some_and(|t| t.len() == k && prompt.starts_with(t));
            if k >= dm.min_tokens && snap_ok && s.history().len() >= k {
                if target
                    .as_deref()
                    .is_some_and(|t| dm.store.holds(t, &prompt[..k]))
                {
                    save_line = ", store already holds this turn".into();
                } else if target.as_deref().is_some_and(|t| {
                    !dm.store.is_prefix_of(t, &prompt[..k])
                        && k.saturating_sub(matched_pos) < every.max(1)
                }) {
                    // A short new turn on a saved conversation that went another way (another
                    // question on the same document): not worth a second copy of the document.
                    save_line = ", store not saved (short branch of a saved conversation)".into();
                } else {
                    let end = s.snapshot().clone();
                    let t = Instant::now();
                    match save_to_store(
                        &mut s,
                        &dm.store,
                        target.as_deref(),
                        k,
                        ckpts,
                        (CkptKind::TurnHeader, end),
                        EntryStatus::Ready,
                    ) {
                        Ok(sv) => {
                            save_line = format!(
                                ", store save {} ({}{} rows exported, {} checkpoints, {:.1} ms)",
                                sv.key.as_deref().unwrap_or("?"),
                                if sv.new_entry { "new, " } else { "" },
                                sv.rows_exported,
                                sv.checkpoints,
                                ms(t)
                            )
                        }
                        Err(e) => save_line = format!(", store save failed: {e}"),
                    }
                }
            }
        }
        let steps = emitted.saturating_sub(1).max(1) as f64;
        let reused = fs.reuse.map_or(0, Reuse::tokens);
        drop(s);
        drop(lock);
        let pf = &fs.prefill;
        eprintln!(
            "aien-serve-http: qwen38: lock wait {lock_wait_ms} ms, prompt {} tok{}, reuse {:?} ({reused} tok){store_line}, fed {} ({} stepped), prefill {prefill_ms} ms (gemm chip {} ms, gdn chip {} ms, attn chip {} ms, host {} ms, restore {} ms, snapshot {} ms, hook {} ms), decode {emitted} tok in {decode_ms} ms, graph chip {:.1} ms/token, wall {:.1} ms/token{save_line}",
            prompt.len(),
            if cold { " (cold requested)" } else { "" },
            fs.reuse.unwrap_or(Reuse::Cold),
            fs.fed,
            fs.stepped,
            pf.gemm_chip_ns / 1_000_000,
            pf.gdn_chip_ns / 1_000_000,
            pf.attn_chip_ns / 1_000_000,
            pf.host_ns / 1_000_000,
            fs.restore_ns / 1_000_000,
            fs.snapshot_ns / 1_000_000,
            fs.hook_ns / 1_000_000,
            chip_ns as f64 / 1e6 / steps,
            decode_ms as f64 / steps,
        );
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms,
            sampling: Some(observer.record()),
        })
    }

    fn ingest(
        &self,
        prompt: Vec<u32>,
        label: Option<String>,
    ) -> Result<serde_json::Value, EngineError> {
        let dm = self
            .mem
            .as_ref()
            .ok_or_else(|| EngineError::Failed("document memory is off".into()))?;
        let k = turn_header(&prompt).unwrap_or(prompt.len());
        if k < 8 || k >= self.max_context {
            return Err(EngineError::Failed(format!(
                "document of {k} tokens (8 ..< {})",
                self.max_context
            )));
        }
        let tokens = prompt[..k].to_vec();
        let key = dm
            .store
            .create(&[], label, EntryStatus::Reading { done: 0, total: k });
        dm.store.unpin();
        dm.ingest
            .lock()
            .map_err(|_| EngineError::Failed("ingest queue poisoned".into()))?
            .send(IngestJob {
                key: key.clone(),
                tokens,
            })
            .map_err(|_| EngineError::Failed("ingest thread gone".into()))?;
        Ok(serde_json::json!({"id": key, "tokens": k, "status": "queued"}))
    }

    fn documents(&self) -> Option<serde_json::Value> {
        self.mem.as_ref().map(|dm| dm.store.status())
    }

    fn document_action(&self, id: &str, action: &str) -> Result<serde_json::Value, EngineError> {
        let dm = self
            .mem
            .as_ref()
            .ok_or_else(|| EngineError::Failed("document memory is off".into()))?;
        let ok = match action {
            "spill" => dm.store.spill(id).map_err(EngineError::Failed)?,
            "remove" => dm.store.remove(id),
            other => {
                return Err(EngineError::Failed(format!(
                    "unknown action {other} (spill, remove)"
                )))
            }
        };
        Ok(serde_json::json!({"id": id, "action": action, "done": ok}))
    }
}
