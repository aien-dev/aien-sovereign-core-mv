//! Document memory for the Qwen3.8-27B door (cascade/document-memory): many saved conversations,
//! each a token record plus the attention KV rows and DeltaNet checkpoints the resident model
//! needs to continue it, kept in host memory up to a budget and spilled to NVMe beyond it.
//!
//! The device holds one conversation at a time ([`aien_inference_abi::qwen38_omega::
//! Qwen38Session`]: its KV cache rows and DeltaNet state). This store is the second and third
//! tier behind it:
//!
//! | tier | what | who moves it |
//! |---|---|---|
//! | device | KV rows + DeltaNet state of the live conversation | the session |
//! | RAM | per conversation: tokens, KV rows `0 .. kv_rows` (one host copy), DeltaNet checkpoints at several positions | [`ConvStore`] on the door's thread |
//! | NVMe | the same, one file per conversation (`<dir>/<key>.conv`, `O_DIRECT`, page aligned) | [`ConvStore`] when the RAM budget is exceeded |
//!
//! Why checkpoints: a KV prefix alone cannot resume a DeltaNet layer (the recurrent state is not
//! stored per position), so a conversation can only be continued from a position whose DeltaNet
//! state was copied out: every `checkpoint_every` tokens during a long read (so a user can ask
//! about a document before it is fully read), before the prompt's last turn header (the point
//! the next turn of the same chat starts from), at the end of a reply (the live point), and at
//! the end of an ingested document (the point a question on that document starts from).
//!
//! Prefix matching ([`ConvStore::best_match`]): among every conversation, the checkpoint at the
//! largest position `p` such that the conversation's first `p` tokens are a prefix of the new
//! prompt and `p < prompt.len()` (at least one token is always fed: its logits give the first
//! reply token). The KV rows `0 .. p` are then written into the device cache (only those the
//! device does not already hold for this exact token prefix), the DeltaNet checkpoint is set, and
//! the rest of the prompt is fed. Rows are moved bit for bit, so the continued conversation
//! computes what the uninterrupted one would.
//!
//! Interface for other parts of the server (the continuous-batching work uses the same store):
//! [`ConvStore::open`], [`ConvStore::best_match`], [`ConvStore::create`], [`ConvStore::with_entry`]
//! (loads a spilled conversation first), [`ConvStore::touch`], [`ConvStore::status`]. The store
//! never touches the device; the door does the device copies through
//! [`aien_inference_abi::qwen38_omega::Qwen38Omega::kv_read_rows`] / `kv_write_rows`.
//!
//! Safety of stored bytes: the namespace in every file header binds the model (config digest and
//! model id), the linked `libomega_gpu.a`, this crate's commit, the cache shape and the numeric
//! switches; a file from another namespace is ignored at open. Files are plain (not sealed): the
//! directory must be private to the server's user (created `0700`). Sealing as `kv_persist`
//! does for the 30B is the follow-up.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::{FileExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use aien_inference_abi::qwen38_omega::Qwen38Snapshot;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bytes of one K or V row (256 bf16).
pub const ROW_BYTES: usize = 512;
/// KV heads per attention layer.
pub const KV_HEADS: usize = 4;
/// Planes per (layer, head): K and V.
pub const PLANES: usize = 2;
/// Alignment of every buffer and file section (`O_DIRECT`).
pub const ALIGN: usize = 4096;
/// File header size (JSON, zero padded).
const HEADER_BYTES: usize = 16384;
const MAGIC: &str = "aien-conv-v1";
const FILE_EXT: &str = "conv";

/// Default host memory budget for resident conversations (8 GiB).
pub const DEFAULT_RAM_BUDGET: u64 = 8 << 30;
/// Default NVMe budget (64 GiB).
pub const DEFAULT_DISK_BUDGET: u64 = 64 << 30;
/// Default checkpoint spacing during a long read.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 4096;
/// Default reader threads for a file (planes are read in parallel).
pub const DEFAULT_IO_THREADS: usize = 8;

// ---------------------------------------------------------------------------
// Aligned buffers
// ---------------------------------------------------------------------------

/// A page aligned byte buffer with a capacity and a length (`O_DIRECT` reads and writes need
/// both the address and the length aligned; `len` is the useful part).
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    cap: usize,
    len: usize,
}

// SAFETY: a plain owned allocation.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Capacity rounded up to [`ALIGN`]; length 0.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.div_ceil(ALIGN).max(1) * ALIGN;
        let layout = Layout::from_size_align(cap, ALIGN).expect("aligned layout");
        // SAFETY: non-zero size, valid alignment.
        let p = unsafe { alloc(layout) };
        let ptr = NonNull::new(p).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, cap, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Sets the useful length (`<= capacity`); bytes past the old length are whatever the
    /// allocation holds until written.
    pub fn set_len(&mut self, len: usize) {
        assert!(
            len <= self.cap,
            "AlignedBuf::set_len {len} > capacity {}",
            self.cap
        );
        self.len = len;
    }

    /// Grows the capacity to at least `cap` (rounded up), keeping the first `len` bytes.
    pub fn reserve_total(&mut self, cap: usize) {
        if cap <= self.cap {
            return;
        }
        let mut n = Self::with_capacity(cap);
        n.as_cap_mut()[..self.len].copy_from_slice(self.as_slice());
        n.len = self.len;
        *self = n;
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for cap bytes, len <= cap.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, unique access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The whole capacity (for reads that fill it).
    pub fn as_cap_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for cap bytes, unique access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }

    pub fn extend_from_slice(&mut self, s: &[u8]) {
        let end = self.len + s.len();
        self.reserve_total(end);
        let start = self.len;
        self.as_cap_mut()[start..end].copy_from_slice(s);
        self.len = end;
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.cap, ALIGN).expect("aligned layout");
        // SAFETY: allocated with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), layout) }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf {{ len: {}, cap: {} }}", self.len, self.cap)
    }
}

fn pad(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

// ---------------------------------------------------------------------------
// Shape, config
// ---------------------------------------------------------------------------

/// The cache shape the store serves: fixed by the model and the server's `--max-context`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    /// Attention layers (16).
    pub n_attn: usize,
    /// DeltaNet layers (48).
    pub n_gdn: usize,
    /// Floats of one layer's conv tail.
    pub tail_floats: usize,
    /// Floats of one layer's recurrent state.
    pub state_floats: usize,
    /// Positions the device cache holds.
    pub max_pos: usize,
}

impl Shape {
    /// KV bytes of `rows` positions (all layers, heads, K and V).
    pub fn kv_bytes(&self, rows: usize) -> u64 {
        (self.n_attn * KV_HEADS * PLANES * rows * ROW_BYTES) as u64
    }

    /// Host bytes of one DeltaNet checkpoint.
    pub fn checkpoint_bytes(&self) -> u64 {
        (4 * self.n_gdn * (self.tail_floats + self.state_floats)) as u64
    }

    fn planes(&self) -> usize {
        self.n_attn * KV_HEADS * PLANES
    }
}

/// How the store is set up (`settings_from_env` reads the `AIEN_CONV_*` variables).
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// Spill directory on NVMe; `None` keeps conversations in RAM only (evicted ones are lost).
    pub dir: Option<PathBuf>,
    pub ram_budget: u64,
    pub disk_budget: u64,
    /// Checkpoint spacing during a long read (a multiple of the prefill chunk is exact).
    pub checkpoint_every: usize,
    /// Binds model, build, shape and switches (hex digest).
    pub namespace: String,
    pub shape: Shape,
    pub io_threads: usize,
    /// `O_DIRECT` file I/O (off only for tests on file systems without it).
    pub direct_io: bool,
}

/// Environment variables the door reads.
pub const DIR_ENV: &str = "AIEN_CONV_STORE_DIR";
pub const RAM_ENV: &str = "AIEN_CONV_RAM_BYTES";
pub const DISK_ENV: &str = "AIEN_CONV_DISK_BYTES";
pub const EVERY_ENV: &str = "AIEN_CONV_CHECKPOINT_EVERY";
pub const OFF_ENV: &str = "AIEN_CONV_STORE_OFF";

/// Settings from the environment: `None` when `AIEN_CONV_STORE_OFF=1`. The directory is
/// optional (RAM only without it); budgets in bytes (suffix `G`/`M` accepted).
pub fn settings_from_env(
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<(Option<PathBuf>, u64, u64, usize)> {
    if lookup(OFF_ENV).as_deref() == Some("1") {
        return None;
    }
    let bytes = |v: Option<String>, d: u64| -> u64 {
        let Some(v) = v else { return d };
        let v = v.trim();
        let (num, mul) = match v.chars().last() {
            Some('G') | Some('g') => (&v[..v.len() - 1], 1u64 << 30),
            Some('M') | Some('m') => (&v[..v.len() - 1], 1u64 << 20),
            _ => (v, 1),
        };
        num.parse::<u64>().map_or(d, |n| n * mul)
    };
    let dir = lookup(DIR_ENV)
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from);
    let ram = bytes(lookup(RAM_ENV), DEFAULT_RAM_BUDGET);
    let disk = bytes(lookup(DISK_ENV), DEFAULT_DISK_BUDGET);
    let every = lookup(EVERY_ENV)
        .and_then(|s| s.trim().parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_CHECKPOINT_EVERY);
    Some((dir, ram, disk, every))
}

/// The namespace digest: everything that can change a stored byte's meaning.
pub fn namespace_digest(parts: &[(&str, &str)]) -> String {
    let mut h = Sha256::new();
    h.update(b"AIEN-CONV-NS-V1\0");
    for (k, v) in parts {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update(b"\n");
    }
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

/// Why a checkpoint was taken (informational; every kind resumes the same way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CkptKind {
    /// Every `checkpoint_every` tokens of a long read.
    Chunk,
    /// Before the prompt's last turn header (the next turn of this chat starts here).
    TurnHeader,
    /// After the reply (the live point: a client that returns the reply verbatim starts here).
    Live,
    /// The end of an ingested document's content (a question on the document starts here) and
    /// the token before it (the document's last token may merge with what follows it).
    DocumentEnd,
}

/// One DeltaNet checkpoint: the state after `pos` tokens.
#[derive(Debug)]
pub struct Checkpoint {
    pub pos: usize,
    pub kind: CkptKind,
    pub snap: Qwen38Snapshot,
}

/// The host copy of one conversation's attention KV rows `0 .. rows`, one page aligned buffer
/// per (layer, head, plane), each `rows * 512` bytes long.
#[derive(Debug)]
pub struct KvRows {
    rows: usize,
    planes: Vec<AlignedBuf>,
}

impl KvRows {
    pub fn new(shape: &Shape, cap_rows: usize) -> Self {
        let planes = (0..shape.planes())
            .map(|_| AlignedBuf::with_capacity(cap_rows.max(1) * ROW_BYTES))
            .collect();
        Self { rows: 0, planes }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bytes(&self) -> u64 {
        (self.planes.len() * self.rows * ROW_BYTES) as u64
    }

    /// Allocated bytes (capacity).
    pub fn capacity_bytes(&self) -> u64 {
        self.planes.iter().map(|p| p.capacity() as u64).sum()
    }

    pub fn plane_index(layer: usize, head: usize, plane: usize) -> usize {
        (layer * KV_HEADS + head) * PLANES + plane
    }

    /// Rows `from .. to` of one plane as bf16 words.
    pub fn plane_words(
        &self,
        layer: usize,
        head: usize,
        plane: usize,
        from: usize,
        to: usize,
    ) -> &[u16] {
        let b = &self.planes[Self::plane_index(layer, head, plane)].as_slice()
            [from * ROW_BYTES..to * ROW_BYTES];
        // SAFETY: the buffer is 4096 aligned and the row size is a multiple of 2; u16 has no invalid bit patterns.
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u16>(), b.len() / 2) }
    }

    /// Makes room for `rows` rows in every plane (keeps what is there).
    pub fn reserve_rows(&mut self, rows: usize) {
        for p in &mut self.planes {
            p.reserve_total(rows * ROW_BYTES);
        }
    }

    /// Appends rows: `fill(layer, head, plane, k_or_v_words)` writes `n` rows (`n * 256` words)
    /// into the given slice for every plane; the row count grows by `n` only if every fill
    /// succeeds.
    pub fn append_rows(
        &mut self,
        n: usize,
        mut fill: impl FnMut(usize, usize, usize, &mut [u16]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (from, to) = (self.rows, self.rows + n);
        self.reserve_rows(to);
        let n_attn = self.planes.len() / (KV_HEADS * PLANES);
        for layer in 0..n_attn {
            for head in 0..KV_HEADS {
                for plane in 0..PLANES {
                    let buf = &mut self.planes[Self::plane_index(layer, head, plane)];
                    buf.set_len(to * ROW_BYTES);
                    let b = &mut buf.as_mut_slice()[from * ROW_BYTES..to * ROW_BYTES];
                    // SAFETY: aligned, even length, u16 has no invalid bit patterns.
                    let w = unsafe {
                        std::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<u16>(), b.len() / 2)
                    };
                    fill(layer, head, plane, w)?;
                }
            }
        }
        self.rows = to;
        Ok(())
    }

    /// Copies rows `0 .. n` of `src` (its first `n` rows) into a fresh copy (a fork).
    pub fn prefix_copy(&self, shape: &Shape, n: usize) -> Self {
        let n = n.min(self.rows);
        let mut out = Self::new(shape, n);
        for (dst, src) in out.planes.iter_mut().zip(&self.planes) {
            dst.extend_from_slice(&src.as_slice()[..n * ROW_BYTES]);
        }
        out.rows = n;
        out
    }

    /// Drops rows past `n`.
    pub fn truncate(&mut self, n: usize) {
        if n < self.rows {
            for p in &mut self.planes {
                p.set_len(n * ROW_BYTES);
            }
            self.rows = n;
        }
    }
}

/// Where an entry's bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Residency {
    /// KV rows and checkpoints in RAM.
    Resident,
    /// Only on NVMe (tokens and the checkpoint table stay in RAM).
    Spilled,
}

/// What an entry is doing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryStatus {
    /// A background read in progress: `done` of `total` tokens fed and checkpointed.
    Reading {
        done: usize,
        total: usize,
    },
    Ready,
    Failed(String),
}

/// One saved conversation.
#[derive(Debug)]
pub struct Entry {
    pub key: String,
    pub label: Option<String>,
    /// The token record (the KV rows cover `0 .. kv_rows` of these).
    pub tokens: Vec<u32>,
    /// Rows the host copy holds (RAM or NVMe).
    pub kv_rows: usize,
    pub kv: Option<KvRows>,
    /// Resident checkpoints, sorted by position.
    pub checkpoints: Vec<Arc<Checkpoint>>,
    /// Every checkpoint's position and kind (resident or on NVMe), sorted.
    pub ckpt_table: Vec<(usize, CkptKind)>,
    pub residency: Residency,
    /// The RAM copy is newer than the file (or there is no file).
    pub dirty: bool,
    pub file: Option<PathBuf>,
    pub file_bytes: u64,
    pub status: EntryStatus,
    pub created_unix: u64,
    pub last_used: Instant,
    pub uses: u64,
}

impl Entry {
    /// Host bytes the resident copy holds (0 when spilled).
    pub fn ram_bytes(&self) -> u64 {
        self.kv.as_ref().map_or(0, KvRows::bytes)
            + self
                .checkpoints
                .iter()
                .map(|c| c.snap.bytes() as u64)
                .sum::<u64>()
    }

    /// The resident checkpoint at `pos`.
    pub fn checkpoint_at(&self, pos: usize) -> Option<&Arc<Checkpoint>> {
        self.checkpoints.iter().find(|c| c.pos == pos)
    }

    /// Adds (or replaces) the checkpoint at `snap.pos`.
    pub fn put_checkpoint(&mut self, kind: CkptKind, snap: Qwen38Snapshot) {
        let pos = snap.pos;
        self.checkpoints.retain(|c| c.pos != pos);
        self.ckpt_table.retain(|c| c.0 != pos);
        self.checkpoints
            .push(Arc::new(Checkpoint { pos, kind, snap }));
        self.checkpoints.sort_by_key(|c| c.pos);
        self.ckpt_table.push((pos, kind));
        self.ckpt_table.sort_by_key(|c| c.0);
        self.dirty = true;
    }

    /// Drops every checkpoint past `pos` and the token record past it (a conversation cut back,
    /// e.g. a fork point) and the KV rows past it.
    pub fn truncate(&mut self, pos: usize) {
        self.checkpoints.retain(|c| c.pos <= pos);
        self.ckpt_table.retain(|c| c.0 <= pos);
        self.tokens.truncate(pos);
        if self.kv_rows > pos {
            self.kv_rows = pos;
            if let Some(kv) = self.kv.as_mut() {
                kv.truncate(pos);
            }
        }
        self.dirty = true;
    }

    /// Drops chunk checkpoints below `keep_from` (after a read finishes, the dense checkpoints
    /// of the read are no longer needed when `keep_all` is off).
    pub fn thin_chunk_checkpoints(&mut self) {
        self.checkpoints.retain(|c| c.kind != CkptKind::Chunk);
        self.ckpt_table.retain(|c| c.1 != CkptKind::Chunk);
        self.dirty = true;
    }
}

/// What [`ConvStore::best_match`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub key: String,
    /// The checkpoint position (tokens `0 .. pos` of the entry are a prefix of the prompt).
    pub pos: usize,
    pub kind: CkptKind,
    /// Tokens the entry and the prompt share (`>= pos`).
    pub shared: usize,
    /// The entry's whole token record length.
    pub entry_len: usize,
    pub residency: Residency,
}

/// Length of the common prefix of two token slices.
pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// Counters for `/health` and the handoff.
#[derive(Debug, Default)]
pub struct Counters {
    pub spills: AtomicU64,
    pub spill_bytes: AtomicU64,
    pub spill_ns: AtomicU64,
    pub loads: AtomicU64,
    pub load_bytes: AtomicU64,
    pub load_ns: AtomicU64,
    pub drops: AtomicU64,
    pub matches: AtomicU64,
    pub misses: AtomicU64,
}

struct Inner {
    entries: HashMap<String, Entry>,
    ram_bytes: u64,
    disk_bytes: u64,
    /// The entry the door is working on: never spilled.
    pinned: Option<String>,
}

pub struct ConvStore {
    cfg: StoreConfig,
    inner: Mutex<Inner>,
    pub counters: Counters,
}

impl ConvStore {
    /// Opens the store: creates the directory (`0700`) and indexes every file of this namespace
    /// in it (tokens and checkpoint tables are read, the bytes stay on disk). Files of other
    /// namespaces are left alone and reported.
    pub fn open(cfg: StoreConfig) -> Result<Arc<Self>, String> {
        let mut inner = Inner {
            entries: HashMap::new(),
            ram_bytes: 0,
            disk_bytes: 0,
            pinned: None,
        };
        if let Some(dir) = &cfg.dir {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
            let mut other = 0usize;
            let mut bad = 0usize;
            for de in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
                let Ok(de) = de else { continue };
                let p = de.path();
                if p.extension().and_then(|e| e.to_str()) != Some(FILE_EXT) {
                    continue;
                }
                match read_header(&p) {
                    Ok(h) if h.namespace == cfg.namespace && h.shape == cfg.shape => {
                        let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                        let tokens = match read_tokens(&p, &h) {
                            Ok(t) => t,
                            Err(_) => {
                                bad += 1;
                                continue;
                            }
                        };
                        inner.disk_bytes += bytes;
                        inner.entries.insert(
                            h.key.clone(),
                            Entry {
                                key: h.key.clone(),
                                label: h.label.clone(),
                                tokens,
                                kv_rows: h.kv_rows,
                                kv: None,
                                checkpoints: Vec::new(),
                                ckpt_table: h.checkpoints.clone(),
                                residency: Residency::Spilled,
                                dirty: false,
                                file: Some(p),
                                file_bytes: bytes,
                                status: EntryStatus::Ready,
                                created_unix: h.created_unix,
                                last_used: Instant::now(),
                                uses: 0,
                            },
                        );
                    }
                    Ok(_) => other += 1,
                    Err(_) => bad += 1,
                }
            }
            eprintln!(
                "aien-serve-http: conv store: {} ({} conversations on disk, {:.2} GB; {other} of other namespaces, {bad} unreadable), RAM budget {:.2} GB, disk budget {:.2} GB, checkpoint every {} tokens",
                dir.display(),
                inner.entries.len(),
                inner.disk_bytes as f64 / 1e9,
                cfg.ram_budget as f64 / 1e9,
                cfg.disk_budget as f64 / 1e9,
                cfg.checkpoint_every
            );
        } else {
            eprintln!(
                "aien-serve-http: conv store: RAM only (no {DIR_ENV}), budget {:.2} GB, checkpoint every {} tokens",
                cfg.ram_budget as f64 / 1e9,
                cfg.checkpoint_every
            );
        }
        Ok(Arc::new(Self {
            cfg,
            inner: Mutex::new(inner),
            counters: Counters::default(),
        }))
    }

    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }

    pub fn shape(&self) -> &Shape {
        &self.cfg.shape
    }

    pub fn checkpoint_every(&self) -> usize {
        self.cfg.checkpoint_every
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The checkpoint, over every conversation, at the largest position that is a strict prefix
    /// of `prompt` (see the module notes). Ties: a resident entry, then the most recently used.
    pub fn best_match(&self, prompt: &[u32]) -> Option<Match> {
        let g = self.lock();
        let mut best: Option<(Match, Instant)> = None;
        for e in g.entries.values() {
            if matches!(e.status, EntryStatus::Failed(_)) {
                continue;
            }
            let shared = common_prefix_len(&e.tokens, prompt);
            let limit = shared.min(prompt.len().saturating_sub(1)).min(e.kv_rows);
            let Some(&(pos, kind)) = e.ckpt_table.iter().rev().find(|c| c.0 <= limit && c.0 > 0)
            else {
                continue;
            };
            let m = Match {
                key: e.key.clone(),
                pos,
                kind,
                shared,
                entry_len: e.tokens.len(),
                residency: e.residency,
            };
            let better = match &best {
                None => true,
                Some((b, t)) => {
                    m.pos > b.pos
                        || (m.pos == b.pos
                            && ((m.residency == Residency::Resident
                                && b.residency != Residency::Resident)
                                || (m.residency == b.residency && e.last_used > *t)))
                }
            };
            if better {
                best = Some((m, e.last_used));
            }
        }
        drop(g);
        match best {
            Some((m, _)) => {
                self.counters.matches.fetch_add(1, Ordering::Relaxed);
                Some(m)
            }
            None => {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// A new, empty, resident entry for `tokens` (no KV rows yet, no checkpoints); returns its
    /// key (`sha256` of the tokens and the time, hex). Pinned until [`Self::unpin`].
    pub fn create(&self, tokens: &[u32], label: Option<String>, status: EntryStatus) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut h = Sha256::new();
        h.update(b"AIEN-CONV-KEY\0");
        for t in tokens {
            h.update(t.to_le_bytes());
        }
        h.update(now.to_le_bytes());
        h.update(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0u128, |d| d.as_nanos())
                .to_le_bytes(),
        );
        let key = hex(&h.finalize())[..32].to_string();
        let mut g = self.lock();
        g.entries.insert(
            key.clone(),
            Entry {
                key: key.clone(),
                label,
                tokens: tokens.to_vec(),
                kv_rows: 0,
                kv: Some(KvRows::new(&self.cfg.shape, tokens.len().max(1))),
                checkpoints: Vec::new(),
                ckpt_table: Vec::new(),
                residency: Residency::Resident,
                dirty: true,
                file: None,
                file_bytes: 0,
                status,
                created_unix: now,
                last_used: Instant::now(),
                uses: 0,
            },
        );
        g.pinned = Some(key.clone());
        key
    }

    /// Pins `key` (never spilled while pinned) and makes it resident, loading it from NVMe when
    /// spilled (evicting others to fit the budget first). Then runs `f` on it. `None` when
    /// there is no such entry. Time in `f` counts as use.
    pub fn with_entry<R>(
        &self,
        key: &str,
        f: impl FnOnce(&mut Entry) -> R,
    ) -> Result<Option<R>, String> {
        let mut g = self.lock();
        if !g.entries.contains_key(key) {
            return Ok(None);
        }
        g.pinned = Some(key.to_string());
        if g.entries[key].residency == Residency::Spilled {
            let need = {
                let e = &g.entries[key];
                self.cfg.shape.kv_bytes(e.kv_rows)
                    + self.cfg.shape.checkpoint_bytes() * e.ckpt_table.len() as u64
            };
            self.make_room(&mut g, need);
            let t = Instant::now();
            let e = g.entries.get_mut(key).expect("present");
            let path = e.file.clone().ok_or("spilled entry without a file")?;
            let loaded = load_file(&path, &self.cfg)?;
            if loaded.tokens != e.tokens {
                return Err(format!("{}: token record changed on disk", path.display()));
            }
            e.kv = Some(loaded.kv);
            e.kv_rows = loaded.kv_rows;
            e.checkpoints = loaded.checkpoints;
            e.ckpt_table = e.checkpoints.iter().map(|c| (c.pos, c.kind)).collect();
            e.residency = Residency::Resident;
            e.dirty = false;
            let bytes = e.ram_bytes();
            let n_ckpt = e.checkpoints.len();
            g.ram_bytes += bytes;
            self.counters.loads.fetch_add(1, Ordering::Relaxed);
            self.counters.load_bytes.fetch_add(bytes, Ordering::Relaxed);
            self.counters
                .load_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            eprintln!(
                "aien-serve-http: conv store: loaded {key} from NVMe: {} rows, {} checkpoints, {:.3} GB in {:.3} s ({:.2} GB/s)",
                loaded.kv_rows,
                n_ckpt,
                bytes as f64 / 1e9,
                t.elapsed().as_secs_f64(),
                bytes as f64 / 1e9 / t.elapsed().as_secs_f64().max(1e-9)
            );
        }
        let e = g.entries.get_mut(key).expect("present");
        let before = e.ram_bytes();
        e.last_used = Instant::now();
        e.uses += 1;
        let r = f(e);
        let after = g.entries[key].ram_bytes();
        g.ram_bytes = g.ram_bytes + after - before;
        Ok(Some(r))
    }

    /// Releases the pin and brings RAM within budget (spilling the least recently used).
    pub fn unpin(&self) {
        let mut g = self.lock();
        g.pinned = None;
        self.make_room(&mut g, 0);
    }

    /// Whether `key` holds a checkpoint at `pos` covering exactly `tokens` (resident or not).
    pub fn holds(&self, key: &str, tokens: &[u32]) -> bool {
        let g = self.lock();
        g.entries.get(key).is_some_and(|e| {
            e.kv_rows >= tokens.len()
                && e.tokens.starts_with(tokens)
                && e.ckpt_table.iter().any(|c| c.0 == tokens.len())
        })
    }

    /// Whether `key`'s token record is a prefix of `tokens` (saving `tokens` would extend it).
    pub fn is_prefix_of(&self, key: &str, tokens: &[u32]) -> bool {
        let g = self.lock();
        g.entries
            .get(key)
            .is_some_and(|e| tokens.starts_with(&e.tokens))
    }

    /// Marks `key` used now.
    pub fn touch(&self, key: &str) {
        let mut g = self.lock();
        if let Some(e) = g.entries.get_mut(key) {
            e.last_used = Instant::now();
        }
    }

    /// Entry status (for the documents endpoint).
    pub fn entry_status(&self, key: &str) -> Option<serde_json::Value> {
        let g = self.lock();
        g.entries.get(key).map(|e| entry_json(e, &self.cfg.shape))
    }

    pub fn set_status(&self, key: &str, status: EntryStatus) {
        let mut g = self.lock();
        if let Some(e) = g.entries.get_mut(key) {
            e.status = status;
        }
    }

    /// Removes an entry (RAM and file).
    pub fn remove(&self, key: &str) -> bool {
        let mut g = self.lock();
        let Some(e) = g.entries.remove(key) else {
            return false;
        };
        g.ram_bytes = g.ram_bytes.saturating_sub(e.ram_bytes());
        if let Some(f) = &e.file {
            let _ = std::fs::remove_file(f);
            g.disk_bytes = g.disk_bytes.saturating_sub(e.file_bytes);
        }
        if g.pinned.as_deref() == Some(key) {
            g.pinned = None;
        }
        true
    }

    /// Writes every dirty resident entry to NVMe (shutdown, or a test wanting the file).
    pub fn flush_all(&self) -> Result<usize, String> {
        let Some(_) = &self.cfg.dir else { return Ok(0) };
        let mut g = self.lock();
        let keys: Vec<String> = g
            .entries
            .values()
            .filter(|e| {
                e.residency == Residency::Resident && e.dirty && e.status == EntryStatus::Ready
            })
            .map(|e| e.key.clone())
            .collect();
        let mut n = 0;
        for k in keys {
            self.write_entry(&mut g, &k)?;
            n += 1;
        }
        Ok(n)
    }

    /// Spills `key` now (writes if dirty, frees RAM). For tests and explicit eviction.
    pub fn spill(&self, key: &str) -> Result<bool, String> {
        let mut g = self.lock();
        if !g.entries.contains_key(key) {
            return Ok(false);
        }
        self.evict(&mut g, key)?;
        Ok(true)
    }

    /// Spills (or drops, without a directory) least recently used resident entries until
    /// `ram_bytes + need <= budget`, never the pinned one or one being read.
    fn make_room(&self, g: &mut Inner, need: u64) {
        while g.ram_bytes + need > self.cfg.ram_budget {
            let victim = g
                .entries
                .values()
                .filter(|e| {
                    e.residency == Residency::Resident
                        && g.pinned.as_deref() != Some(e.key.as_str())
                        && !matches!(e.status, EntryStatus::Reading { .. })
                        && e.ram_bytes() > 0
                })
                .min_by_key(|e| e.last_used)
                .map(|e| e.key.clone());
            let Some(k) = victim else { break };
            if let Err(e) = self.evict(g, &k) {
                eprintln!("aien-serve-http: conv store: could not spill {k}: {e}; dropping it");
                let _ = self.drop_entry(g, &k);
            }
        }
    }

    fn evict(&self, g: &mut Inner, key: &str) -> Result<(), String> {
        if self.cfg.dir.is_none() {
            self.drop_entry(g, key);
            return Ok(());
        }
        if g.entries[key].dirty {
            self.write_entry(g, key)?;
        }
        let e = g.entries.get_mut(key).expect("present");
        let bytes = e.ram_bytes();
        e.kv = None;
        e.checkpoints.clear();
        e.residency = Residency::Spilled;
        g.ram_bytes = g.ram_bytes.saturating_sub(bytes);
        eprintln!(
            "aien-serve-http: conv store: spilled {key} ({:.3} GB freed)",
            bytes as f64 / 1e9
        );
        Ok(())
    }

    fn drop_entry(&self, g: &mut Inner, key: &str) -> bool {
        let Some(e) = g.entries.remove(key) else {
            return false;
        };
        g.ram_bytes = g.ram_bytes.saturating_sub(e.ram_bytes());
        if let Some(f) = &e.file {
            let _ = std::fs::remove_file(f);
            g.disk_bytes = g.disk_bytes.saturating_sub(e.file_bytes);
        }
        self.counters.drops.fetch_add(1, Ordering::Relaxed);
        eprintln!("aien-serve-http: conv store: dropped {key} (no spill directory or over the disk budget)");
        true
    }

    /// Writes `key`'s resident bytes to its file (replacing an old one), within the disk budget
    /// (oldest files of other entries are removed to make room).
    fn write_entry(&self, g: &mut Inner, key: &str) -> Result<(), String> {
        let dir = self.cfg.dir.as_ref().ok_or("no spill directory")?;
        let t = Instant::now();
        let (bytes, path) = {
            let e = g.entries.get(key).expect("present");
            let kv = e.kv.as_ref().ok_or("write_entry: not resident")?;
            let planned = file_bytes(
                &self.cfg.shape,
                e.tokens.len(),
                kv.rows(),
                e.checkpoints.len(),
            );
            (planned, dir.join(format!("{key}.{FILE_EXT}")))
        };
        // Disk budget: remove the oldest other files first.
        let own_old = g.entries[key].file_bytes;
        while g.disk_bytes - own_old + bytes > self.cfg.disk_budget {
            let victim = g
                .entries
                .values()
                .filter(|e| e.key != key && e.file.is_some() && e.residency == Residency::Spilled)
                .min_by_key(|e| e.last_used)
                .map(|e| e.key.clone());
            let Some(v) = victim else { break };
            self.drop_entry(g, &v);
        }
        let tmp = dir.join(format!("{key}.{FILE_EXT}.tmp"));
        {
            let e = g.entries.get(key).expect("present");
            write_file(&tmp, e, &self.cfg)?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let written = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(bytes);
        let e = g.entries.get_mut(key).expect("present");
        g.disk_bytes = g.disk_bytes - e.file_bytes + written;
        e.file = Some(path);
        e.file_bytes = written;
        e.dirty = false;
        self.counters.spills.fetch_add(1, Ordering::Relaxed);
        self.counters
            .spill_bytes
            .fetch_add(written, Ordering::Relaxed);
        self.counters
            .spill_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        eprintln!(
            "aien-serve-http: conv store: wrote {key} to NVMe: {:.3} GB in {:.3} s ({:.2} GB/s)",
            written as f64 / 1e9,
            t.elapsed().as_secs_f64(),
            written as f64 / 1e9 / t.elapsed().as_secs_f64().max(1e-9)
        );
        Ok(())
    }

    /// Store state for `/health` and `/v1/documents`.
    pub fn status(&self) -> serde_json::Value {
        let Ok(g) = self.inner.try_lock() else {
            return serde_json::json!({"busy": true});
        };
        let c = &self.counters;
        let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut entries: Vec<_> = g.entries.values().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.last_used));
        serde_json::json!({
            "namespace": self.cfg.namespace,
            "dir": self.cfg.dir.as_ref().map(|d| d.display().to_string()),
            "ram_bytes": g.ram_bytes,
            "ram_budget": self.cfg.ram_budget,
            "disk_bytes": g.disk_bytes,
            "disk_budget": self.cfg.disk_budget,
            "checkpoint_every": self.cfg.checkpoint_every,
            "conversations": g.entries.len(),
            "resident": g.entries.values().filter(|e| e.residency == Residency::Resident).count(),
            "pinned": g.pinned,
            "counters": {
                "spills": ld(&c.spills), "spill_bytes": ld(&c.spill_bytes), "spill_ns": ld(&c.spill_ns),
                "loads": ld(&c.loads), "load_bytes": ld(&c.load_bytes), "load_ns": ld(&c.load_ns),
                "drops": ld(&c.drops), "matches": ld(&c.matches), "misses": ld(&c.misses),
            },
            "entries": entries.iter().map(|e| entry_json(e, &self.cfg.shape)).collect::<Vec<_>>(),
        })
    }
}

fn entry_json(e: &Entry, shape: &Shape) -> serde_json::Value {
    serde_json::json!({
        "id": e.key,
        "label": e.label,
        "tokens": e.tokens.len(),
        "kv_rows": e.kv_rows,
        "checkpoints": e.ckpt_table.iter().map(|(p, k)| serde_json::json!({"pos": p, "kind": k})).collect::<Vec<_>>(),
        "residency": e.residency,
        "dirty": e.dirty,
        "ram_bytes": e.ram_bytes(),
        "file": e.file.as_ref().map(|f| f.display().to_string()),
        "file_bytes": e.file_bytes,
        "bytes_per_conversation": shape.kv_bytes(e.kv_rows) + shape.checkpoint_bytes() * e.ckpt_table.len() as u64 + 4 * e.tokens.len() as u64,
        "status": e.status,
        "created_unix": e.created_unix,
        "idle_s": e.last_used.elapsed().as_secs(),
        "uses": e.uses,
    })
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Header {
    magic: String,
    namespace: String,
    shape: Shape,
    key: String,
    label: Option<String>,
    tokens: usize,
    kv_rows: usize,
    checkpoints: Vec<(usize, CkptKind)>,
    created_unix: u64,
}

/// Section offsets of a file: header, tokens, then checkpoints (tails, states per checkpoint),
/// then planes.
struct FileLayout {
    tokens_off: u64,
    ckpt_off: u64,
    ckpt_stride: u64,
    tails_bytes: usize,
    states_bytes: usize,
    planes_off: u64,
    plane_stride: u64,
    plane_bytes: usize,
    total: u64,
}

fn layout(shape: &Shape, tokens: usize, kv_rows: usize, n_ckpt: usize) -> FileLayout {
    let tokens_off = HEADER_BYTES as u64;
    let ckpt_off = tokens_off + pad(tokens * 4) as u64;
    let tails_bytes = shape.n_gdn * shape.tail_floats * 4;
    let states_bytes = shape.n_gdn * shape.state_floats * 4;
    let ckpt_stride = (pad(tails_bytes) + pad(states_bytes)) as u64;
    let planes_off = ckpt_off + ckpt_stride * n_ckpt as u64;
    let plane_bytes = kv_rows * ROW_BYTES;
    let plane_stride = pad(plane_bytes) as u64;
    let total = planes_off + plane_stride * shape.planes() as u64;
    FileLayout {
        tokens_off,
        ckpt_off,
        ckpt_stride,
        tails_bytes,
        states_bytes,
        planes_off,
        plane_stride,
        plane_bytes,
        total,
    }
}

/// Bytes a file for this entry takes.
pub fn file_bytes(shape: &Shape, tokens: usize, kv_rows: usize, n_ckpt: usize) -> u64 {
    layout(shape, tokens, kv_rows, n_ckpt).total
}

fn open_rw(path: &Path, cfg: &StoreConfig, write: bool) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    if write {
        o.write(true).create(true).truncate(true).mode(0o600);
    }
    if cfg.direct_io {
        o.custom_flags(libc::O_DIRECT);
    }
    o.open(path)
}

fn read_header(path: &Path) -> Result<Header, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; HEADER_BYTES];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let h: Header = serde_json::from_slice(&buf[..end])
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    if h.magic != MAGIC {
        return Err(format!("{}: not a conversation file", path.display()));
    }
    Ok(h)
}

fn read_tokens(path: &Path, h: &Header) -> Result<Vec<u32>, String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let l = layout(&h.shape, h.tokens, h.kv_rows, h.checkpoints.len());
    let mut raw = vec![0u8; h.tokens * 4];
    f.read_exact_at(&mut raw, l.tokens_off)
        .map_err(|e| format!("{}: tokens: {e}", path.display()))?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

struct Loaded {
    tokens: Vec<u32>,
    kv_rows: usize,
    kv: KvRows,
    checkpoints: Vec<Arc<Checkpoint>>,
}

/// One I/O job: a file range and a buffer.
struct Job<'a> {
    off: u64,
    buf: &'a mut [u8],
}

fn run_jobs(f: &File, jobs: Vec<Job<'_>>, threads: usize, write: bool) -> io::Result<()> {
    let threads = threads.max(1).min(jobs.len().max(1));
    let jobs = Mutex::new(jobs.into_iter());
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let jobs = &jobs;
                s.spawn(move || -> io::Result<()> {
                    loop {
                        let job = jobs.lock().unwrap_or_else(|p| p.into_inner()).next();
                        let Some(job) = job else { return Ok(()) };
                        if write {
                            f.write_all_at(job.buf, job.off)?;
                        } else {
                            f.read_exact_at(job.buf, job.off)?;
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join()
                .map_err(|_| io::Error::other("I/O thread panicked"))??;
        }
        Ok(())
    })
}

fn load_file(path: &Path, cfg: &StoreConfig) -> Result<Loaded, String> {
    let h = read_header(path)?;
    if h.namespace != cfg.namespace || h.shape != cfg.shape {
        return Err(format!("{}: another namespace or shape", path.display()));
    }
    let f = open_rw(path, cfg, false).map_err(|e| format!("{}: {e}", path.display()))?;
    let l = layout(&cfg.shape, h.tokens, h.kv_rows, h.checkpoints.len());
    // Tokens: one aligned read.
    let mut tok = AlignedBuf::with_capacity(pad(h.tokens * 4));
    tok.set_len(pad(h.tokens * 4));
    if !tok.is_empty() {
        f.read_exact_at(tok.as_mut_slice(), l.tokens_off)
            .map_err(|e| format!("{}: tokens: {e}", path.display()))?;
    }
    let tokens: Vec<u32> = tok.as_slice()[..h.tokens * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    // Checkpoints: staged through aligned buffers.
    let n_ckpt = h.checkpoints.len();
    let mut stage: Vec<(AlignedBuf, AlignedBuf)> = (0..n_ckpt)
        .map(|_| {
            let mut t = AlignedBuf::with_capacity(pad(l.tails_bytes));
            t.set_len(pad(l.tails_bytes));
            let mut s = AlignedBuf::with_capacity(pad(l.states_bytes));
            s.set_len(pad(l.states_bytes));
            (t, s)
        })
        .collect();
    let mut kv = KvRows::new(&cfg.shape, h.kv_rows.max(1));
    for p in &mut kv.planes {
        p.reserve_total(pad(l.plane_bytes));
        p.set_len(pad(l.plane_bytes));
    }
    let mut jobs = Vec::with_capacity(2 * n_ckpt + kv.planes.len());
    for (i, (t, s)) in stage.iter_mut().enumerate() {
        let off = l.ckpt_off + l.ckpt_stride * i as u64;
        jobs.push(Job {
            off,
            buf: t.as_mut_slice(),
        });
        jobs.push(Job {
            off: off + pad(l.tails_bytes) as u64,
            buf: s.as_mut_slice(),
        });
    }
    if l.plane_bytes > 0 {
        for (i, p) in kv.planes.iter_mut().enumerate() {
            jobs.push(Job {
                off: l.planes_off + l.plane_stride * i as u64,
                buf: p.as_mut_slice(),
            });
        }
    }
    run_jobs(&f, jobs, cfg.io_threads, false)
        .map_err(|e| format!("{}: read: {e}", path.display()))?;
    for p in &mut kv.planes {
        p.set_len(l.plane_bytes);
    }
    kv.rows = h.kv_rows;
    let checkpoints = stage
        .iter()
        .zip(&h.checkpoints)
        .map(|((t, s), &(pos, kind))| {
            let tails = bytes_to_f32(&t.as_slice()[..l.tails_bytes]);
            let states = bytes_to_f32(&s.as_slice()[..l.states_bytes]);
            Arc::new(Checkpoint {
                pos,
                kind,
                snap: Qwen38Snapshot::from_parts(pos, tails, states),
            })
        })
        .collect();
    Ok(Loaded {
        tokens,
        kv_rows: h.kv_rows,
        kv,
        checkpoints,
    })
}

fn write_file(path: &Path, e: &Entry, cfg: &StoreConfig) -> Result<(), String> {
    let kv = e.kv.as_ref().ok_or("write_file: not resident")?;
    let l = layout(&cfg.shape, e.tokens.len(), kv.rows(), e.checkpoints.len());
    let h = Header {
        magic: MAGIC.into(),
        namespace: cfg.namespace.clone(),
        shape: cfg.shape.clone(),
        key: e.key.clone(),
        label: e.label.clone(),
        tokens: e.tokens.len(),
        kv_rows: kv.rows(),
        checkpoints: e.checkpoints.iter().map(|c| (c.pos, c.kind)).collect(),
        created_unix: e.created_unix,
    };
    let hj = serde_json::to_vec(&h).map_err(|e| e.to_string())?;
    if hj.len() >= HEADER_BYTES {
        return Err(format!(
            "conversation header of {} bytes (limit {HEADER_BYTES})",
            hj.len()
        ));
    }
    let f = open_rw(path, cfg, true).map_err(|e| format!("{}: {e}", path.display()))?;
    f.set_len(l.total)
        .map_err(|e| format!("{}: set_len: {e}", path.display()))?;
    // Header and tokens in one aligned buffer.
    let mut head = AlignedBuf::with_capacity(l.ckpt_off as usize);
    head.set_len(l.ckpt_off as usize);
    head.as_mut_slice().fill(0);
    head.as_mut_slice()[..hj.len()].copy_from_slice(&hj);
    for (i, t) in e.tokens.iter().enumerate() {
        head.as_mut_slice()[HEADER_BYTES + i * 4..HEADER_BYTES + i * 4 + 4]
            .copy_from_slice(&t.to_le_bytes());
    }
    // Checkpoints staged.
    let mut stage: Vec<(AlignedBuf, AlignedBuf)> = e
        .checkpoints
        .iter()
        .map(|c| {
            let mut t = AlignedBuf::with_capacity(pad(l.tails_bytes));
            t.set_len(pad(l.tails_bytes));
            t.as_mut_slice().fill(0);
            f32_to_bytes(c.snap.tails(), &mut t.as_mut_slice()[..l.tails_bytes]);
            let mut s = AlignedBuf::with_capacity(pad(l.states_bytes));
            s.set_len(pad(l.states_bytes));
            s.as_mut_slice().fill(0);
            f32_to_bytes(c.snap.states(), &mut s.as_mut_slice()[..l.states_bytes]);
            (t, s)
        })
        .collect();
    // Planes: written from their own buffers (padded tail needs the capacity to be aligned).
    let mut planes: Vec<AlignedBuf> = Vec::with_capacity(kv.planes.len());
    let mut jobs: Vec<Job<'_>> = Vec::new();
    jobs.push(Job {
        off: 0,
        buf: head.as_mut_slice(),
    });
    for (i, (t, s)) in stage.iter_mut().enumerate() {
        let off = l.ckpt_off + l.ckpt_stride * i as u64;
        jobs.push(Job {
            off,
            buf: t.as_mut_slice(),
        });
        jobs.push(Job {
            off: off + pad(l.tails_bytes) as u64,
            buf: s.as_mut_slice(),
        });
    }
    if l.plane_bytes > 0 {
        for p in &kv.planes {
            let mut b = AlignedBuf::with_capacity(pad(l.plane_bytes));
            b.set_len(pad(l.plane_bytes));
            b.as_mut_slice()[..l.plane_bytes].copy_from_slice(&p.as_slice()[..l.plane_bytes]);
            b.as_mut_slice()[l.plane_bytes..].fill(0);
            planes.push(b);
        }
        for (i, p) in planes.iter_mut().enumerate() {
            jobs.push(Job {
                off: l.planes_off + l.plane_stride * i as u64,
                buf: p.as_mut_slice(),
            });
        }
    }
    run_jobs(&f, jobs, cfg.io_threads, true)
        .map_err(|e| format!("{}: write: {e}", path.display()))?;
    f.sync_all()
        .map_err(|e| format!("{}: sync: {e}", path.display()))?;
    Ok(())
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn f32_to_bytes(v: &[f32], out: &mut [u8]) {
    for (i, x) in v.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}

/// The header of a file (for tools).
pub fn describe_file(path: &Path) -> Result<serde_json::Value, String> {
    let h = read_header(path)?;
    serde_json::to_value(h).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            n_attn: 2,
            n_gdn: 3,
            tail_floats: 8,
            state_floats: 16,
            max_pos: 64,
        }
    }

    fn config(dir: Option<PathBuf>, ram: u64) -> StoreConfig {
        StoreConfig {
            dir,
            ram_budget: ram,
            disk_budget: 1 << 30,
            checkpoint_every: 4,
            namespace: "test-ns".into(),
            shape: shape(),
            io_threads: 2,
            direct_io: std::env::var("AIEN_CONV_TEST_DIRECT").as_deref() == Ok("1"),
        }
    }

    fn snap(pos: usize, seed: f32) -> Qwen38Snapshot {
        let s = shape();
        let tails: Vec<f32> = (0..s.n_gdn * s.tail_floats)
            .map(|i| seed + i as f32)
            .collect();
        let states: Vec<f32> = (0..s.n_gdn * s.state_floats)
            .map(|i| seed * 2.0 - i as f32)
            .collect();
        Qwen38Snapshot::from_parts(pos, tails, states)
    }

    fn fill_rows(e: &mut Entry, n: usize, seed: u16) {
        let kv = e.kv.as_mut().unwrap();
        kv.append_rows(n, |l, h, p, w| {
            for (i, x) in w.iter_mut().enumerate() {
                *x = seed.wrapping_add((l * 1000 + h * 100 + p * 10 + i) as u16);
            }
            Ok(())
        })
        .unwrap();
        e.kv_rows = kv.rows();
        e.dirty = true;
    }

    fn tmpdir(name: &str) -> PathBuf {
        let base = std::env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let d = base.join(format!("conv-store-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn match_takes_the_longest_checkpoint_that_is_a_strict_prefix() {
        let st = ConvStore::open(config(None, 1 << 30)).unwrap();
        let doc: Vec<u32> = (1..=12).collect();
        let k = st.create(&doc, None, EntryStatus::Ready);
        st.with_entry(&k, |e| {
            fill_rows(e, 12, 1);
            e.put_checkpoint(CkptKind::Chunk, snap(4, 1.0));
            e.put_checkpoint(CkptKind::Chunk, snap(8, 2.0));
            e.put_checkpoint(CkptKind::DocumentEnd, snap(12, 3.0));
        })
        .unwrap();
        st.unpin();
        // Same document plus a question: the end checkpoint.
        let mut p = doc.clone();
        p.extend([100, 101]);
        let m = st.best_match(&p).unwrap();
        assert_eq!((m.pos, m.kind, m.shared), (12, CkptKind::DocumentEnd, 12));
        // Diverges after 9 tokens: the checkpoint at 8.
        let mut p = doc[..9].to_vec();
        p.extend([7, 7, 7]);
        assert_eq!(st.best_match(&p).unwrap().pos, 8);
        // Exactly the document: the last token must be fed, so the checkpoint at 8.
        assert_eq!(st.best_match(&doc).unwrap().pos, 8);
        // Shares fewer than 4 tokens: nothing.
        assert!(st.best_match(&[1, 2, 3, 9, 9]).is_none());
        assert!(st.best_match(&[5, 6]).is_none());
    }

    #[test]
    fn budget_spills_least_recently_used_and_loads_back_bit_for_bit() {
        let dir = tmpdir("spill");
        let s = shape();
        let one = s.kv_bytes(8) + s.checkpoint_bytes() * 2;
        let st = ConvStore::open(config(Some(dir.clone()), one + one / 2)).unwrap();
        let a = st.create(
            &(1..=8).collect::<Vec<u32>>(),
            Some("a".into()),
            EntryStatus::Ready,
        );
        st.with_entry(&a, |e| {
            fill_rows(e, 8, 10);
            e.put_checkpoint(CkptKind::Chunk, snap(4, 1.5));
            e.put_checkpoint(CkptKind::Live, snap(8, 2.5));
        })
        .unwrap();
        st.unpin();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = st.create(
            &(11..=18).collect::<Vec<u32>>(),
            Some("b".into()),
            EntryStatus::Ready,
        );
        st.with_entry(&b, |e| {
            fill_rows(e, 8, 20);
            e.put_checkpoint(CkptKind::Chunk, snap(4, 3.5));
            e.put_checkpoint(CkptKind::Live, snap(8, 4.5));
        })
        .unwrap();
        st.unpin();
        // a (older) is spilled, b resident.
        let status = st.status();
        assert_eq!(status["resident"], 1);
        let m = st.best_match(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(
            (m.key.as_str(), m.pos, m.residency),
            (a.as_str(), 8, Residency::Spilled)
        );
        // Loading a back spills b and restores a's bytes exactly.
        let expect_k: Vec<u16> = (0..256)
            .map(|i| 10u16.wrapping_add((1000 + 300 + 10 + i) as u16))
            .collect();
        st.with_entry(&a, |e| {
            assert_eq!(e.residency, Residency::Resident);
            let kv = e.kv.as_ref().unwrap();
            assert_eq!(kv.rows(), 8);
            assert_eq!(kv.plane_words(1, 3, 1, 0, 1), &expect_k[..]);
            let c = e.checkpoint_at(8).unwrap();
            assert_eq!(c.snap.tails(), snap(8, 2.5).tails());
            assert_eq!(c.snap.states(), snap(8, 2.5).states());
            assert_eq!(c.kind, CkptKind::Live);
        })
        .unwrap();
        st.unpin();
        assert_eq!(st.status()["resident"], 1);
        // A second open indexes the files.
        drop(st);
        let st2 = ConvStore::open(config(Some(dir.clone()), one * 4)).unwrap();
        assert_eq!(st2.status()["conversations"], 2);
        let m = st2
            .best_match(&[11, 12, 13, 14, 15, 16, 17, 18, 1])
            .unwrap();
        assert_eq!((m.key.as_str(), m.pos), (b.as_str(), 8));
        st2.with_entry(&b, |e| {
            assert_eq!(e.kv.as_ref().unwrap().rows(), 8);
            assert_eq!(
                e.checkpoint_at(4).unwrap().snap.states(),
                snap(4, 3.5).states()
            );
        })
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_namespace_files_are_left_alone() {
        let dir = tmpdir("ns");
        let st = ConvStore::open(config(Some(dir.clone()), 1 << 30)).unwrap();
        let a = st.create(&[1, 2, 3, 4, 5], None, EntryStatus::Ready);
        st.with_entry(&a, |e| {
            fill_rows(e, 5, 1);
            e.put_checkpoint(CkptKind::Live, snap(4, 1.0));
        })
        .unwrap();
        st.unpin();
        assert_eq!(st.flush_all().unwrap(), 1);
        drop(st);
        let mut other = config(Some(dir.clone()), 1 << 30);
        other.namespace = "another".into();
        let st2 = ConvStore::open(other).unwrap();
        assert_eq!(st2.status()["conversations"], 0);
        assert!(dir.join(format!("{a}.conv")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_parse_sizes() {
        let s = settings_from_env(|k| match k {
            RAM_ENV => Some("12G".into()),
            DISK_ENV => Some("300M".into()),
            EVERY_ENV => Some("2048".into()),
            DIR_ENV => Some("/x/y".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(s, (Some(PathBuf::from("/x/y")), 12 << 30, 300 << 20, 2048));
        assert!(settings_from_env(|k| (k == OFF_ENV).then(|| "1".to_string())).is_none());
    }
}
