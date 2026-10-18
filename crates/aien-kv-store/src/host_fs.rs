// The publication protocol (temp write, file fsync, atomic rename, directory
// fsync, index insert) and the startup walk are written from ADR 0003 sections
// 2.4, 3.4, 4.1 and 7.4. LMCache fs_l2_adapter.py (4d69cdc4, Apache-2.0,
// Copyright LMCache contributors) informed two ideas only: a completed object
// is published by writing a temporary file and renaming it, and a store of a
// key that already exists on disk is skipped. No code was copied. See
// THIRD_PARTY.md and ADR 0003 context item 24.

//! Host filesystem `KvStore` adapter (cut C4, `sovereign-core#484`).
//!
//! Layout under `<root>` (ADR 0003 section 2.4):
//!
//! ```text
//! <root>/
//!   STORE_ID
//!   objects/00/ .. objects/ff/
//!   tmp/
//! ```
//!
//! All 256 shard directories and `tmp/` are created at [`HostFsStore::open`]
//! with a parent-directory `fsync`; the hot path never creates a directory.

use crate::identity::{BackendFormat, ChunkKey, LayoutStrides, NamespaceFields, NamespaceId};
use crate::index::{Index, IndexEntry, FLAG_FOREIGN, FLAG_INDETERMINATE};
use crate::pin::{HolderGuard, PinnedDest, PinnedSource};
use crate::record::{
    payload_root, ChunkHeader, RecordHeader, FLAG_PARTIAL_TAIL, HEADER_REGION_LEN, MAX_SEGMENTS,
    PAYLOAD_DOMAIN, SEGMENT_TABLE_OFFSET,
};
use crate::store::{
    CancelOutcome, Capacity, Completion, DrainReport, Incarnation, JobId, JobStage, KvStore,
    PinToken, PublishOutcome, RecoveryProgress, StoreState, Terminal, UnfinishedJob,
};
use crate::validate::validate_record;
use aien_platform::{ObjectId, PlatformError};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Raw `ENOSPC` for [`FaultAction::Fail`].
pub const ENOSPC: i32 = 28;
/// Raw `EIO` for [`FaultAction::Fail`].
pub const EIO: i32 = 5;

const DEFAULT_QUOTA_BYTES: u64 = 8 << 30;
const DEFAULT_MAX_OBJECTS: u64 = 65_536;
const DEFAULT_MAX_RESTORE_OPS: u32 = 48;
const DEFAULT_MAX_PARK_OPS: u32 = 64;
const DEFAULT_MAX_INFLIGHT_BYTES: u64 = 256 << 20;
const DEFAULT_MAX_WAITERS_PER_JOB: u32 = 16;
const DEFAULT_MAX_WAITERS_TOTAL: u32 = 1024;
const DEFAULT_RECOVERY_SLICE_MS: u64 = 50;

/// Who runs the worker steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMode {
    Manual,
    Thread,
}

/// Where a fault can be injected (ADR 2.4 P1 to P5, plus the other I/O).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Boundary {
    P1Write,
    P2FsyncFile,
    P3Rename,
    P4FsyncDir,
    P5IndexInsert,
    Unlink,
    LoadRead,
    FlushSync,
    /// Once per directory entry of the recovery walk, before its stat, with
    /// the index lock NOT held. `Fail(errno)` stands for a stat I/O error.
    RecoverEntry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultAction {
    Fail(i32),
    TornWrite { after_bytes: u64, crash: bool },
    Crash,
    Stall(Duration),
}

pub trait FaultHook: Send + Sync {
    fn at(&self, boundary: Boundary, key: &ChunkKey) -> Option<FaultAction>;
}

#[derive(Default)]
struct Script {
    armed: BTreeMap<Boundary, Vec<FaultAction>>,
    fired: BTreeMap<Boundary, u64>,
    seen: Vec<Boundary>,
}

/// One-shot faults per boundary, consumed in the order they were armed.
#[derive(Default)]
pub struct ScriptedFaults {
    inner: Mutex<Script>,
}

impl ScriptedFaults {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn arm(&self, boundary: Boundary, action: FaultAction) {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.armed.entry(boundary).or_default().push(action);
    }

    pub fn fired(&self, boundary: Boundary) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .fired
            .get(&boundary)
            .copied()
            .unwrap_or(0)
    }

    /// Every boundary the adapter consulted, in order.
    pub fn seen(&self) -> Vec<Boundary> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .seen
            .clone()
    }

    pub fn armed(&self, boundary: Boundary) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .armed
            .get(&boundary)
            .map_or(0, Vec::len)
    }
}

impl FaultHook for ScriptedFaults {
    fn at(&self, boundary: Boundary, _key: &ChunkKey) -> Option<FaultAction> {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.seen.push(boundary);
        let list = s.armed.get_mut(&boundary)?;
        if list.is_empty() {
            return None;
        }
        let a = list.remove(0);
        *s.fired.entry(boundary).or_insert(0) += 1;
        Some(a)
    }
}

#[derive(Clone)]
pub struct HostFsConfig {
    pub root: PathBuf,
    pub namespace: NamespaceId,
    pub model_sha256: [u8; 32],
    pub layout: LayoutStrides,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub chunk_tokens: u32,
    pub dtype: u8,
    pub backend_format: u8,
    pub quota_bytes: u64,
    pub max_objects: u64,
    pub max_restore_ops: u32,
    pub max_park_ops: u32,
    pub max_inflight_bytes: u64,
    pub max_waiters_per_job: u32,
    pub max_waiters_total: u32,
    pub recovery_slice_ms: u64,
    pub purge_after_recovery: bool,
    pub worker: WorkerMode,
    pub faults: Option<Arc<dyn FaultHook>>,
    /// The key every object is sealed with (ADR 0003 section 20). `None`:
    /// `open` refuses (fail closed); there is no plaintext mode.
    pub seal_key: Option<Arc<crate::seal::SealKey>>,
}

impl std::fmt::Debug for HostFsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostFsConfig")
            .field("root", &self.root)
            .field("namespace", &self.namespace)
            .field("quota_bytes", &self.quota_bytes)
            .field("max_objects", &self.max_objects)
            .field("worker", &self.worker)
            .finish()
    }
}

impl HostFsConfig {
    pub fn from_fields(root: PathBuf, f: &NamespaceFields<'_>) -> Result<Self, PlatformError> {
        if f.backend_format != BackendFormat::HostPagedPoolV1 {
            return Err(PlatformError::InvalidLayout);
        }
        let namespace = f.namespace_id().map_err(|_| PlatformError::InvalidLayout)?;
        let cfg = Self {
            root,
            namespace,
            model_sha256: f.model_sha256,
            layout: f.layout,
            num_layers: f.num_layers,
            num_kv_heads: f.num_kv_heads,
            head_dim: f.head_dim,
            block_size: f.block_size,
            chunk_tokens: f.chunk_tokens,
            dtype: f.dtype as u8,
            backend_format: f.backend_format as u8,
            quota_bytes: DEFAULT_QUOTA_BYTES,
            max_objects: DEFAULT_MAX_OBJECTS,
            max_restore_ops: DEFAULT_MAX_RESTORE_OPS,
            max_park_ops: DEFAULT_MAX_PARK_OPS,
            max_inflight_bytes: DEFAULT_MAX_INFLIGHT_BYTES,
            max_waiters_per_job: DEFAULT_MAX_WAITERS_PER_JOB,
            max_waiters_total: DEFAULT_MAX_WAITERS_TOTAL,
            recovery_slice_ms: DEFAULT_RECOVERY_SLICE_MS,
            purge_after_recovery: true,
            worker: WorkerMode::Thread,
            faults: None,
            seal_key: None,
        };
        validate_config(&cfg)?;
        Ok(cfg)
    }

    pub fn apply_env(&mut self, get: impl Fn(&str) -> Option<String>) -> Result<(), PlatformError> {
        fn parse_pos(val: Option<String>) -> Result<Option<u64>, PlatformError> {
            let Some(s) = val else {
                return Ok(None);
            };
            let v: u64 = s.parse().map_err(|_| PlatformError::InvalidLayout)?;
            if v == 0 {
                return Err(PlatformError::InvalidLayout);
            }
            Ok(Some(v))
        }

        let q = parse_pos(get("AIEN_KV_STORE_QUOTA_BYTES"))?;
        let m = parse_pos(get("AIEN_KV_STORE_MAX_OBJECTS"))?;
        let sl = parse_pos(get("AIEN_KV_STORE_RECOVERY_SLICE_MS"))?;

        if let Some(v) = q {
            self.quota_bytes = v;
        }
        if let Some(v) = m {
            self.max_objects = v;
        }
        if let Some(v) = sl {
            self.recovery_slice_ms = v;
        }
        Ok(())
    }
}

fn validate_config(c: &HostFsConfig) -> Result<(), PlatformError> {
    if c.quota_bytes == 0
        || c.max_objects == 0
        || c.max_restore_ops == 0
        || c.max_park_ops == 0
        || c.max_inflight_bytes == 0
        || c.max_waiters_per_job == 0
        || c.max_waiters_total == 0
        || c.block_size == 0
        || c.chunk_tokens == 0
        || !c.chunk_tokens.is_multiple_of(c.block_size)
        || (c.chunk_tokens / c.block_size) as usize > MAX_SEGMENTS
        || c.num_layers == 0
        || c.num_kv_heads == 0
        || c.head_dim == 0
        || c.layout.block_stride == 0
        || c.layout.element_bytes == 0
    {
        return Err(PlatformError::InvalidLayout);
    }

    let head_bytes = (c.head_dim as u64)
        .checked_mul(c.layout.element_bytes)
        .ok_or(PlatformError::InvalidLayout)?;
    if head_bytes > c.layout.head_stride {
        return Err(PlatformError::InvalidLayout);
    }
    let row_span = (c.num_kv_heads as u64 - 1)
        .checked_mul(c.layout.head_stride)
        .and_then(|x| x.checked_add(head_bytes))
        .ok_or(PlatformError::InvalidLayout)?;
    if row_span > c.layout.token_stride {
        return Err(PlatformError::InvalidLayout);
    }
    let plane_span = (c.block_size as u64 - 1)
        .checked_mul(c.layout.token_stride)
        .and_then(|x| x.checked_add(row_span))
        .ok_or(PlatformError::InvalidLayout)?;
    if plane_span > c.layout.kv_plane_stride {
        return Err(PlatformError::InvalidLayout);
    }
    let two_planes = c
        .layout
        .kv_plane_stride
        .checked_add(plane_span)
        .ok_or(PlatformError::InvalidLayout)?;
    if two_planes > c.layout.layer_stride {
        return Err(PlatformError::InvalidLayout);
    }
    let total_span = (c.num_layers as u64 - 1)
        .checked_mul(c.layout.layer_stride)
        .and_then(|x| x.checked_add(two_planes))
        .ok_or(PlatformError::InvalidLayout)?;
    if total_span > c.layout.block_stride {
        return Err(PlatformError::InvalidLayout);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostFsStats {
    pub io_ops: u64,
    pub reads: u64,
    pub read_bytes: u64,
    pub writes: u64,
    pub write_bytes: u64,
    pub file_fsyncs: u64,
    pub dir_fsyncs: u64,
    pub renames: u64,
    pub unlinks: u64,
    pub metadata_calls: u64,
    pub mkdirs: u64,
    pub dir_entries: u64,
    pub queue_full_total: u64,
    pub disk_full_total: u64,
    pub cancelled_total: u64,
    pub evicted_bytes_total: u64,
    pub dedup_total: u64,
    pub indeterminate_total: u64,
    pub published_total: u64,
    pub terminal_done: u64,
    pub terminal_failed: u64,
    pub terminal_cancelled: u64,
    pub peak_completion_queue: u64,
    pub completion_overflow_total: u64,
    pub released_source_violations: u64,
    pub corrupt_deleted_total: u64,
    pub foreign_files: u64,
    pub foreign_file_bytes: u64,
    pub tmp_bytes_deleted: u64,
    pub corrupt_bytes_deleted: u64,
    pub purge_unlinked: u64,
    pub tail_scratch_bytes: u64,
    pub over_cap: bool,
    pub shutdown_unfinished_jobs: u64,
    pub shutdown_unfinished_bytes: u64,
    /// Most directory entries the recovery or purge walk held in memory at
    /// once. The walk streams one entry at a time (ADR 4.1), so this is 0 or 1.
    pub walk_peak_buffered: u64,
    /// Recovery unlinks (temp or corrupt files) that failed; the file's bytes
    /// stay counted and recovery ends `Degraded`.
    pub recovery_unlink_failures: u64,
    /// Purge unlinks that failed; the purge stopped with its accounting intact.
    pub purge_unlink_failures: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Store,
    Load,
    Flush,
    Noop,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Writing,
    Syncing,
    Publishing,
    Reading,
    Flushing,
}

struct Job {
    id: u64,
    kind: Kind,
    inc: Incarnation,
    key: ChunkKey,
    park: bool,
    bytes: u64,
    payload_len: u64,
    reserved: u64,
    submitted_at: Instant,
    stage: JobStage,
    phase: Phase,
    in_step: bool,
    cancelling: bool,
    crashed: bool,
    header: Option<ChunkHeader>,
    src: Option<PinnedSource>,
    dst: Option<PinnedDest>,
    attached: u32,
    busy_held: bool,
    /// Objects this store marked evicting to make room. They stay on disk and
    /// in the index until the new object is durable (P5); then their unlinks
    /// are queued. Any other ending clears the marks (Grok C4 P1-1).
    victims: Vec<[u8; 32]>,
}

#[derive(Clone)]
struct Unlink {
    key: [u8; 32],
    evict: bool,
}

/// Which directory a walk is listing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WalkStage {
    Tmp,
    Shard(u16),
    Done,
}

/// A streaming directory walk: one open `ReadDir` and the stage it lists.
/// Holds no listing, so memory is constant however many files exist (ADR
/// 4.1, Grok C4 P1-2). Lives behind its own mutex, never the index lock.
struct Walk {
    stage: WalkStage,
    dir: Option<std::fs::ReadDir>,
}

impl Walk {
    fn new(stage: WalkStage) -> Self {
        Self { stage, dir: None }
    }

    fn advance(&mut self) {
        self.dir = None;
        self.stage = match self.stage {
            WalkStage::Tmp => WalkStage::Shard(0),
            WalkStage::Shard(s) if s < 255 => WalkStage::Shard(s + 1),
            _ => WalkStage::Done,
        };
    }

    fn dir_path(&self, root: &Path) -> Option<PathBuf> {
        match self.stage {
            WalkStage::Tmp => Some(root.join("tmp")),
            WalkStage::Shard(s) => Some(root.join("objects").join(format!("{s:02x}"))),
            WalkStage::Done => None,
        }
    }

    /// The next entry, or `None` when every directory has been listed. A
    /// directory that does not exist is empty; any other listing error is
    /// returned (never read as an empty directory, Grok C4 P1-3).
    fn next(&mut self, root: &Path) -> std::io::Result<Option<(WalkStage, PathBuf)>> {
        loop {
            let Some(path) = self.dir_path(root) else {
                return Ok(None);
            };
            if self.dir.is_none() {
                match std::fs::read_dir(&path) {
                    Ok(rd) => self.dir = Some(rd),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        self.advance();
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            match self.dir.as_mut().and_then(Iterator::next) {
                None => self.advance(),
                Some(Err(e)) => return Err(e),
                Some(Ok(entry)) => return Ok(Some((self.stage, entry.path()))),
            }
        }
    }
}

struct Inner {
    cfg: HostFsConfig,
    store_id: Option<[u8; 16]>,
    state: StoreState,
    closed: bool,
    stop_worker: bool,
    next_job: u64,
    next_pin: u64,
    last_clock_ns: u64,
    index: Index,
    unindexed_bytes: u64,
    unindexed_objects: u64,
    /// Bytes of temp files recovery could not unlink: still on disk, so
    /// counted in `used_bytes` until a restart removes them.
    stuck_bytes: u64,
    /// A recovery unlink failed: recovery ends `Degraded`, not `Open`.
    recovery_stuck: bool,
    /// The purge is walking the shards for unindexed files.
    purge_walking: bool,
    /// The one key the purge is unlinking right now: stores of it are refused
    /// until the unlink returns, so the purge never deletes a fresh object.
    purging: Option<[u8; 32]>,
    reserved: u64,
    jobs: BTreeMap<u64, Job>,
    inflight_store: BTreeMap<[u8; 32], u64>,
    unlinks: VecDeque<Unlink>,
    pins: BTreeMap<u64, Vec<[u8; 32]>>,
    restore_ops: u32,
    park_ops: u32,
    restore_bytes: u64,
    park_bytes: u64,
    restore_streak: u32,
    attached_total: u32,
    queue: VecDeque<Completion>,
    stats: HostFsStats,
    recovery: RecoveryProgress,
    purge_pending: bool,
    purge_evict_quota: bool,
    /// Reservations are rounded up to this many bytes (the objects
    /// directory's `st_blksize`, at least 512), so a reservation is not
    /// smaller than what publication later charges from `st_blocks`
    /// (Codex re-review 2, partial 8).
    alloc_unit: u64,
    graveyard: Vec<HolderGuard>,
}

struct Shared {
    root: PathBuf,
    inner: Mutex<Inner>,
    cv: Condvar,
    tail_scratch: Mutex<Vec<u8>>,
    /// The recovery walk. Taken before (never while holding) `inner`.
    walk: Mutex<Walk>,
    /// The purge walk over the shards. Taken before (never while holding) `inner`.
    purge_walk: Mutex<Walk>,
    /// SEAL SEAM (#506): the key every object is sealed and opened with.
    seal_key: Arc<crate::seal::SealKey>,
    /// The root's `LOCK`, held for the store's lifetime (#521).
    _dir_lock: File,
}

/// Host filesystem `KvStore` implementation.
pub struct HostFsStore {
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

static STORE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn generate_store_id() -> [u8; 16] {
    let mut buf = [0u8; 16];
    if let Ok(mut f) = File::open("/dev/urandom") {
        if f.read_exact(&mut buf).is_ok() && buf != [0u8; 16] {
            return buf;
        }
    }
    let mut h = Sha256::new();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    h.update(now.to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    h.update(
        STORE_ID_COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .to_le_bytes(),
    );
    let digest = h.finalize();
    buf.copy_from_slice(&digest[..16]);
    buf
}

fn fsync_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(OPEN_FLAGS | libc::O_DIRECTORY)
        .open(path)?;
    f.sync_all()
}

// ---- at-rest privacy (ADR 0003 section 19) -------------------------------
//
// The objects hold K/V derived from user prompts. Every directory of the store
// is 0700 and every file 0600, owned by the effective uid; existing
// directories are checked (never silently chmodded) and refused with
// `AccessDenied` when others could reach them. Once the root is a verified
// private directory under a trusted parent, only our own uid (or root) can
// change anything inside it, so per-operation dirfd/openat is not needed
// (same-uid attackers are out of scope, section 19.1).

const PRIVATE_DIR_MODE: u32 = 0o700;
const PRIVATE_FILE_MODE: u32 = 0o600;
/// Every open of a store file: never follow a symlink in the last component,
/// never leak the descriptor into a child process.
const OPEN_FLAGS: i32 = libc::O_NOFOLLOW | libc::O_CLOEXEC;

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

/// `Ok(true)`: a private directory of ours. `Ok(false)`: nothing there.
/// `AccessDenied`: a symlink, not a directory, another owner, or any group or
/// other permission bit.
fn check_private_dir(path: &Path) -> Result<bool, PlatformError> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if is_not_found(&e) => return Ok(false),
        Err(e) => return Err(io_to_error(&e, "stat store directory")),
    };
    if !meta.file_type().is_dir() || meta.uid() != euid() || meta.mode() & 0o077 != 0 {
        return Err(PlatformError::AccessDenied);
    }
    Ok(true)
}

/// The directory that will hold the root (the root's parent, or the nearest
/// existing ancestor when the root is still to be created) must be owned by
/// us or by root and, if group or others can write it, carry the sticky bit
/// (like /tmp): otherwise another user could rename the root away and put
/// their own directory in its place.
fn check_trusted_parent(root: &Path) -> Result<(), PlatformError> {
    let mut cur = root.parent();
    while let Some(dir) = cur {
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        match std::fs::metadata(dir) {
            Ok(m) => {
                let shared_write = m.mode() & 0o022 != 0;
                let sticky = m.mode() & 0o1000 != 0;
                let owner_ok = m.uid() == euid() || m.uid() == 0;
                if !m.is_dir() || !owner_ok || (shared_write && !sticky) {
                    return Err(PlatformError::AccessDenied);
                }
                return Ok(());
            }
            Err(e) if is_not_found(&e) => cur = dir.parent(),
            Err(e) => return Err(io_to_error(&e, "stat store parent")),
        }
    }
    Ok(())
}

/// Create one store directory 0700 (the explicit chmod undoes a umask that
/// narrowed it), or verify an existing one. Returns true when it was created.
fn make_private_dir(path: &Path, recursive: bool) -> Result<bool, PlatformError> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    // An existing directory is verified, never chmodded (a recursive
    // DirBuilder would report success for it).
    if check_private_dir(path)? {
        return Ok(false);
    }
    match std::fs::DirBuilder::new()
        .recursive(recursive)
        .mode(PRIVATE_DIR_MODE)
        .create(path)
    {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            check_private_dir(path)?;
            return Ok(false);
        }
        Err(e) => return Err(io_to_error(&e, "create store directory")),
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(PRIVATE_DIR_MODE))
        .map_err(|e| io_to_error(&e, "chmod store directory"))?;
    check_private_dir(path)?;
    Ok(true)
}

/// The one way the adapter creates a file: 0600, no symlink followed, no
/// descriptor leaked. Truncates a leftover of ours (a crash temp in the
/// verified tmp/ directory) instead of failing on it, so a crash never blocks
/// the next attempt; the mode is forced to 0600 through the descriptor so a
/// leftover with an old mode, or a narrowing umask, cannot survive.
fn create_private_file(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(PRIVATE_FILE_MODE)
        .custom_flags(OPEN_FLAGS)
        .open(path)?;
    let meta = f.metadata()?;
    if !meta.is_file() || meta.uid() != euid() {
        return Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
    }
    if meta.mode() & 0o777 != PRIVATE_FILE_MODE {
        f.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    Ok(f)
}

/// One open store per store directory (#521; ADR 0003 supports one runtime
/// per store directory). An exclusive `flock` on `LOCK` in the root, taken at
/// open before anything else in the root is touched and released when the
/// store drops. A second opener, in this process or another, is refused:
/// otherwise its recovery would delete the first store's in-flight temps and
/// the two in-memory indexes would drift apart. A planted `LOCK` (a symlink,
/// another owner) is `AccessDenied` like every other planted name.
fn lock_store_dir(root: &Path) -> Result<File, PlatformError> {
    use std::os::unix::io::AsRawFd;
    let path = root.join("LOCK");
    let f = create_private_file(&path).map_err(|e| {
        if e.raw_os_error() == Some(libc::ELOOP) || e.kind() == std::io::ErrorKind::PermissionDenied
        {
            PlatformError::AccessDenied
        } else {
            io_to_error(&e, "create LOCK")
        }
    })?;
    // SAFETY: flock only takes an advisory lock on a descriptor we own; the
    // lock lives as long as `f`.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Err(PlatformError::TaskRejected(format!(
                "store directory {} is in use by another open store ({} is held)",
                root.display(),
                path.display()
            )));
        }
        return Err(io_to_error(&e, "lock LOCK"));
    }
    Ok(f)
}

/// `create_private_file` plus the bytes (fault injection, temps).
fn write_private_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    create_private_file(path)?.write_all(bytes)
}

/// What a no-follow open of a store file found.
enum PrivateOpen {
    /// A regular file of ours, with its descriptor's metadata.
    Ours(File, std::fs::Metadata),
    /// A symlink, a non-regular file, or another owner's file: not ours,
    /// never read or served, left alone (ADR 0003 section 4.1 `foreign_file`).
    NotOurs(u64),
}

/// Open a store file for reading without following a symlink and check it
/// through the descriptor (no window between check and use). A file of ours
/// that group or others can read (written before section 19) is tightened to
/// 0600 with fchmod.
fn open_private_read(path: &Path) -> std::io::Result<PrivateOpen> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let f = match OpenOptions::new()
        .read(true)
        // O_NONBLOCK: a planted FIFO must not hang the worker; no effect on
        // regular files.
        .custom_flags(OPEN_FLAGS | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
            let len = std::fs::symlink_metadata(path).map_or(0, |m| m.len());
            return Ok(PrivateOpen::NotOurs(len));
        }
        Err(e) => return Err(e),
    };
    let meta = f.metadata()?;
    if !meta.is_file() || meta.uid() != euid() {
        return Ok(PrivateOpen::NotOurs(meta.len()));
    }
    if meta.mode() & 0o077 != 0 {
        f.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    Ok(PrivateOpen::Ours(f, meta))
}

fn phys_bytes_of(meta: &std::fs::Metadata) -> u64 {
    meta.len().max(meta.blocks().saturating_mul(512))
}

fn meta_mtime_ns(meta: &std::fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64)
}

fn system_now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64)
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn object_path_for(root: &Path, key: &[u8; 32]) -> PathBuf {
    let hex = hex_encode(key);
    root.join("objects")
        .join(&hex[..2])
        .join(format!("{hex}.kvc"))
}

fn tmp_path_for(root: &Path, key: &[u8; 32], job_id: u64, inc: Incarnation) -> PathBuf {
    let hex = hex_encode(key);
    root.join("tmp")
        .join(format!("{hex}.{job_id}.{}.{}.part", inc.seq_id, inc.epoch))
}

fn parse_kvc_filename(name: &str, expected_shard: u8) -> Option<[u8; 32]> {
    let hex = name.strip_suffix(".kvc")?;
    if hex.len() != 64 {
        return None;
    }
    if !hex
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let hi = match chunk[0] {
            b'0'..=b'9' => chunk[0] - b'0',
            b'a'..=b'f' => chunk[0] - b'a' + 10,
            _ => return None,
        };
        let lo = match chunk[1] {
            b'0'..=b'9' => chunk[1] - b'0',
            b'a'..=b'f' => chunk[1] - b'a' + 10,
            _ => return None,
        };
        out[i] = (hi << 4) | lo;
    }
    if out[0] != expected_shard {
        return None;
    }
    Some(out)
}

fn errno_to_error(errno: i32, context: &str) -> PlatformError {
    if errno == ENOSPC {
        PlatformError::OutOfMemory
    } else {
        PlatformError::HardwareFault(format!("{context}: errno {errno}"))
    }
}

fn io_to_error(err: &std::io::Error, context: &str) -> PlatformError {
    if err.raw_os_error() == Some(ENOSPC) {
        PlatformError::OutOfMemory
    } else {
        PlatformError::HardwareFault(format!("{context}: {err}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn zero_tail_block(
    scratch: &mut [u8],
    src_block: &[u8],
    layout: &LayoutStrides,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    valid_tokens: u32,
) {
    scratch.copy_from_slice(src_block);
    let head_bytes = (head_dim as usize) * (layout.element_bytes as usize);
    let row_bytes = (num_kv_heads as usize - 1) * (layout.head_stride as usize) + head_bytes;
    for layer in 0..(num_layers as usize) {
        for plane in 0..2usize {
            for tok in (valid_tokens as usize)..(block_size as usize) {
                let off = layer * (layout.layer_stride as usize)
                    + plane * (layout.kv_plane_stride as usize)
                    + tok * (layout.token_stride as usize);
                scratch[off..off + row_bytes].fill(0);
            }
        }
    }
}

enum RecoveredHeader {
    Ours { payload_len: u64 },
    Foreign { payload_len: u64 },
    Corrupt,
}

fn classify_recovered_header(
    buf: &[u8],
    file_size: u64,
    file_key: &[u8; 32],
    cfg: &HostFsConfig,
) -> RecoveredHeader {
    // SEAL SEAM (#506): an object without the sealed-format marker (an old
    // plaintext file) is discarded, never read.
    if !crate::seal::is_sealed_region(buf) {
        return RecoveredHeader::Corrupt;
    }
    let Ok(rec) = RecordHeader::decode(buf) else {
        return RecoveredHeader::Corrupt;
    };
    let h = &rec.chunk;
    if &h.chunk_key.0 .0 != file_key {
        return RecoveredHeader::Corrupt;
    }
    if h.namespace_id == cfg.namespace {
        if h.model_sha256 != cfg.model_sha256
            || h.chunk_tokens != cfg.chunk_tokens
            || h.block_size != cfg.block_size
            || h.num_layers != cfg.num_layers
            || h.num_kv_heads != cfg.num_kv_heads
            || h.head_dim != cfg.head_dim
            || h.dtype != cfg.dtype
            || h.backend_format != cfg.backend_format
            || !h.first_position.is_multiple_of(cfg.chunk_tokens)
            || h.token_count == 0
            || h.token_count > cfg.chunk_tokens
        {
            return RecoveredHeader::Corrupt;
        }
        let want_flags = if h.token_count < cfg.chunk_tokens {
            FLAG_PARTIAL_TAIL
        } else {
            0
        };
        if h.flags != want_flags {
            return RecoveredHeader::Corrupt;
        }
        let expected_blocks = h.token_count.div_ceil(cfg.block_size) as u64;
        if h.payload_len != expected_blocks.saturating_mul(cfg.layout.block_stride) {
            return RecoveredHeader::Corrupt;
        }
        if validate_record(buf, file_size, h, cfg.layout.block_stride).is_err() {
            return RecoveredHeader::Corrupt;
        }
        RecoveredHeader::Ours {
            payload_len: h.payload_len,
        }
    } else {
        if h.payload_len == 0
            || (HEADER_REGION_LEN as u64).checked_add(h.payload_len) != Some(file_size)
            || !foreign_region_is_clean(buf, &rec)
        {
            return RecoveredHeader::Corrupt;
        }
        RecoveredHeader::Foreign {
            payload_len: h.payload_len,
        }
    }
}

/// Padding and segment-root check for an object of another namespace (Grok
/// C4 P2-4). Its segment size is not ours to know (the file does not store
/// it, ADR 2.1 item 3), but the region does: the digest table ends at the
/// last nonzero byte after the 256-byte header, everything after it must be
/// zero, and the root over exactly that table must match. A table whose last
/// digest is 32 zero bytes is the only false negative (probability 2^-256).
fn foreign_region_is_clean(buf: &[u8], rec: &RecordHeader) -> bool {
    let region = &buf[..HEADER_REGION_LEN];
    // SEAL SEAM (#506): the table and padding end where the seal slot starts.
    let Some(last) = region[SEGMENT_TABLE_OFFSET..crate::record::SEAL_SLOT_OFFSET]
        .iter()
        .rposition(|b| *b != 0)
    else {
        return false;
    };
    let n_seg = (last + 1).div_ceil(32);
    if n_seg == 0 || n_seg > MAX_SEGMENTS || n_seg as u64 > rec.chunk.payload_len {
        return false;
    }
    let table_end = SEGMENT_TABLE_OFFSET + 32 * n_seg;
    let mut root = Sha256::new();
    root.update(PAYLOAD_DOMAIN);
    root.update(&region[SEGMENT_TABLE_OFFSET..table_end]);
    root.finalize().as_slice() == rec.payload_sha256
}

impl Inner {
    fn used_bytes(&self) -> u64 {
        self.index.total_bytes() + self.unindexed_bytes + self.stuck_bytes
    }

    fn evicting_bytes(&self) -> u64 {
        self.index.evicting().0
    }

    fn store_reservations(&self) -> u64 {
        self.jobs
            .values()
            .filter(|j| j.kind == Kind::Store && j.reserved > 0 && !j.crashed)
            .count() as u64
    }

    fn bytes_after(&self) -> u64 {
        self.used_bytes()
            .saturating_sub(self.evicting_bytes())
            .saturating_add(self.reserved)
    }

    fn count_after(&self) -> u64 {
        let (_, evicting_count) = self.index.evicting();
        let live = (self.index.len() as u64).saturating_sub(evicting_count);
        live + self.store_reservations()
    }

    fn completion_capacity(&self) -> usize {
        3 * (self.cfg.max_restore_ops as usize + self.cfg.max_park_ops as usize)
    }

    fn park_byte_limit(&self) -> u64 {
        self.cfg.max_inflight_bytes / 4
    }

    fn restore_byte_limit(&self) -> u64 {
        self.cfg.max_inflight_bytes - self.park_byte_limit()
    }

    fn capacity(&self) -> Capacity {
        Capacity {
            state: self.state,
            quota_bytes: self.cfg.quota_bytes,
            used_bytes: self.used_bytes(),
            reserved_bytes: self.reserved,
            evicting_bytes: self.evicting_bytes(),
            unindexed_bytes: self.unindexed_bytes,
            unindexed_objects: self.unindexed_objects,
            objects: self.index.len() as u64,
            max_objects: self.cfg.max_objects,
            restore_ops: self.restore_ops,
            max_restore_ops: self.cfg.max_restore_ops,
            park_ops: self.park_ops,
            max_park_ops: self.cfg.max_park_ops,
            inflight_bytes: self.restore_bytes + self.park_bytes,
            max_inflight_bytes: self.cfg.max_inflight_bytes,
            max_waiters_per_job: self.cfg.max_waiters_per_job,
        }
    }

    fn set_degraded(&mut self) {
        if self.state == StoreState::Open {
            self.state = StoreState::Degraded;
        }
    }

    fn next_timestamp_ns(&mut self, floor_ns: u64) -> u64 {
        let now = system_now_ns()
            .max(floor_ns)
            .max(self.last_clock_ns.saturating_add(1));
        self.last_clock_ns = now;
        now
    }

    fn queue_full(&mut self) -> PlatformError {
        self.stats.queue_full_total += 1;
        PlatformError::QueueFull
    }

    fn gate(&mut self, needs_open: bool) -> Result<(), PlatformError> {
        if self.state == StoreState::Recovering {
            return Err(self.queue_full());
        }
        if self.closed
            || self.state == StoreState::Failed
            || (needs_open && self.state == StoreState::Degraded)
        {
            return Err(PlatformError::DeviceUnavailable);
        }
        Ok(())
    }

    fn active_job_count(&self) -> usize {
        self.jobs.values().filter(|j| !j.crashed).count()
    }

    fn credit_check(&mut self, park: bool, bytes: u64) -> Result<(), PlatformError> {
        let (ops, max_ops, used_bytes, limit) = if park {
            (
                self.park_ops,
                self.cfg.max_park_ops,
                self.park_bytes,
                self.park_byte_limit(),
            )
        } else {
            (
                self.restore_ops,
                self.cfg.max_restore_ops,
                self.restore_bytes,
                self.restore_byte_limit(),
            )
        };
        let slots = self.queue.len() + 3 * self.active_job_count() + 3;
        if ops >= max_ops || used_bytes + bytes > limit || slots > self.completion_capacity() {
            return Err(self.queue_full());
        }
        Ok(())
    }

    fn push_completion(&mut self, c: Completion) {
        self.queue.push_back(c);
        self.stats.peak_completion_queue = self
            .stats
            .peak_completion_queue
            .max(self.queue.len() as u64);
    }

    /// Settle a store's victims: queue their unlinks only when the new object
    /// is durable; otherwise (failure, cancel, crash, indeterminate publish)
    /// return them to service. The store therefore never loses an old object
    /// unless the new one is known to survive (ADR 3.4, Grok C4 P1-1).
    fn settle_victims(&mut self, job: &mut Job, durable: bool) {
        for v in std::mem::take(&mut job.victims) {
            if durable {
                self.unlinks.push_back(Unlink {
                    key: v,
                    evict: true,
                });
            } else {
                self.index.clear_evicting(&v);
            }
        }
    }

    fn finish(
        &mut self,
        mut job: Job,
        terminal: Terminal,
        stage: JobStage,
        publish: PublishOutcome,
    ) {
        if job.park {
            self.park_ops = self.park_ops.saturating_sub(1);
            self.park_bytes = self.park_bytes.saturating_sub(job.bytes);
        } else {
            self.restore_ops = self.restore_ops.saturating_sub(1);
            self.restore_bytes = self.restore_bytes.saturating_sub(job.bytes);
        }
        self.reserved = self.reserved.saturating_sub(job.reserved);
        job.reserved = 0;
        if job.kind == Kind::Store {
            let durable =
                publish == PublishOutcome::Durable && matches!(terminal, Terminal::Done(_));
            self.settle_victims(&mut job, durable);
            if self.inflight_store.get(&job.key.0 .0) == Some(&job.id) {
                self.inflight_store.remove(&job.key.0 .0);
            }
        }
        if job.busy_held {
            self.index.busy_dec(&job.key.0 .0);
            job.busy_held = false;
        }
        self.attached_total = self.attached_total.saturating_sub(job.attached);
        match &terminal {
            Terminal::Done(_) => self.stats.terminal_done += 1,
            Terminal::Failed(_) => self.stats.terminal_failed += 1,
            Terminal::Cancelled => {
                self.stats.terminal_cancelled += 1;
                self.stats.cancelled_total += 1;
            }
        }
        let c = Completion {
            job: JobId(job.id),
            incarnation: job.inc,
            key: job.key,
            stage,
            terminal: Some(terminal),
            publish,
            bytes: job.payload_len,
            elapsed_us: job.submitted_at.elapsed().as_micros() as u64,
            holder: None,
        };
        self.push_completion(c);
        if let Some(s) = job.src.take() {
            self.graveyard.push(s.into_guard());
        }
        if let Some(d) = job.dst.take() {
            self.graveyard.push(d.into_guard());
        }
    }

    fn fail_job(&mut self, job: Job, e: PlatformError) {
        let stage = job.stage;
        self.finish(
            job,
            Terminal::Failed(e),
            stage,
            PublishOutcome::NotApplicable,
        );
    }

    fn mark_crashed(&mut self, mut job: Job) {
        if job.park {
            self.park_ops = self.park_ops.saturating_sub(1);
            self.park_bytes = self.park_bytes.saturating_sub(job.bytes);
        } else {
            self.restore_ops = self.restore_ops.saturating_sub(1);
            self.restore_bytes = self.restore_bytes.saturating_sub(job.bytes);
        }
        self.reserved = self.reserved.saturating_sub(job.reserved);
        job.reserved = 0;
        if job.kind == Kind::Store {
            self.settle_victims(&mut job, false);
            if self.inflight_store.get(&job.key.0 .0) == Some(&job.id) {
                self.inflight_store.remove(&job.key.0 .0);
            }
        }
        if job.busy_held {
            self.index.busy_dec(&job.key.0 .0);
            job.busy_held = false;
        }
        self.attached_total = self.attached_total.saturating_sub(job.attached);
        if let Some(s) = job.src.take() {
            self.graveyard.push(s.into_guard());
        }
        if let Some(d) = job.dst.take() {
            self.graveyard.push(d.into_guard());
        }
        job.crashed = true;
        job.in_step = false;
    }

    fn validate_chunk_header_against_config(&self, h: &ChunkHeader) -> Result<(), PlatformError> {
        if h.namespace_id != self.cfg.namespace
            || h.model_sha256 != self.cfg.model_sha256
            || h.chunk_tokens != self.cfg.chunk_tokens
            || h.block_size != self.cfg.block_size
            || h.num_layers != self.cfg.num_layers
            || h.num_kv_heads != self.cfg.num_kv_heads
            || h.head_dim != self.cfg.head_dim
            || h.dtype != self.cfg.dtype
            || h.backend_format != self.cfg.backend_format
            || !h.first_position.is_multiple_of(self.cfg.chunk_tokens)
            || h.token_count == 0
            || h.token_count > self.cfg.chunk_tokens
            || h.payload_len == 0
        {
            return Err(PlatformError::InvalidLayout);
        }
        let want_flags = if h.token_count < self.cfg.chunk_tokens {
            FLAG_PARTIAL_TAIL
        } else {
            0
        };
        if h.flags != want_flags {
            return Err(PlatformError::InvalidLayout);
        }
        let expected_blocks = h.token_count.div_ceil(self.cfg.block_size) as u64;
        if expected_blocks as usize > MAX_SEGMENTS
            || h.payload_len != expected_blocks.saturating_mul(self.cfg.layout.block_stride)
        {
            return Err(PlatformError::InvalidLayout);
        }
        Ok(())
    }

    fn submit_store(
        &mut self,
        header: ChunkHeader,
        slot: &mut Option<PinnedSource>,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        self.gate(true)?;
        let src = slot.as_ref().ok_or(PlatformError::InvalidLayout)?;
        self.validate_chunk_header_against_config(&header)?;
        if src.total_bytes() != Some(header.payload_len)
            || src.block_count() == 0
            || src.block_count() > MAX_SEGMENTS
            || src.block_bytes() as u64 != self.cfg.layout.block_stride
        {
            return Err(PlatformError::InvalidLayout);
        }
        let full = (src.block_count() as u64 - 1) * header.block_size as u64;
        let tail = (header.token_count as u64).checked_sub(full);
        match tail {
            Some(t)
                if t >= 1
                    && t <= header.block_size as u64
                    && t == src.tail_valid_tokens() as u64 => {}
            _ => return Err(PlatformError::InvalidLayout),
        }

        let key = header.chunk_key;
        let k = key.0 .0;
        if let Some(&jid) = self.inflight_store.get(&k).filter(|jid| {
            self.jobs
                .get(jid)
                .is_some_and(|j| !j.cancelling && !j.crashed)
        }) {
            self.stats.dedup_total += 1;
            return Ok(JobId(jid));
        }
        // A key whose file is being unlinked (a victim, a delete, a purge) is
        // refused until the unlink returns: publishing it now would let that
        // unlink delete the new object.
        if self.purging == Some(k) || self.index.get(&k).is_some_and(|e| e.is_evicting()) {
            return Err(self.queue_full());
        }
        if let Some(e) = self
            .index
            .get(&k)
            .filter(|e| !e.is_evicting() && !e.is_foreign())
        {
            let _ = e;
            self.credit_check(true, 0)?;
            let id = self.next_job;
            self.next_job += 1;
            self.park_ops += 1;
            let job = Job {
                id,
                kind: Kind::Noop,
                inc,
                key,
                park: true,
                bytes: 0,
                payload_len: 0,
                reserved: 0,
                submitted_at: Instant::now(),
                stage: JobStage::Submitted,
                phase: Phase::Queued,
                in_step: false,
                cancelling: false,
                crashed: false,
                header: None,
                src: None,
                dst: None,
                attached: 0,
                busy_held: false,
                victims: Vec::new(),
            };
            self.stats.dedup_total += 1;
            self.jobs.insert(id, job);
            return Ok(JobId(id));
        }

        let bytes =
            (header.payload_len + HEADER_REGION_LEN as u64).next_multiple_of(self.alloc_unit);
        self.credit_check(true, bytes)?;
        // While a purge is pending only free headroom admits a store: evicting
        // a good object for bytes the purge is about to free would lose data
        // for nothing (ADR 4.1, Grok C4 P2-5).
        let (eb, en) = if self.purge_pending {
            (0, 0)
        } else {
            self.index.evictable()
        };
        if self.bytes_after() + bytes > self.cfg.quota_bytes + eb
            || self.count_after() + 1 > self.cfg.max_objects + en
        {
            self.stats.disk_full_total += 1;
            return Err(PlatformError::OutOfMemory);
        }

        let id = self.next_job;
        self.next_job += 1;
        self.park_ops += 1;
        self.park_bytes += bytes;
        self.reserved += bytes;
        let job = Job {
            id,
            kind: Kind::Store,
            inc,
            key,
            park: true,
            bytes,
            payload_len: header.payload_len,
            reserved: bytes,
            submitted_at: Instant::now(),
            stage: JobStage::Submitted,
            phase: Phase::Queued,
            in_step: false,
            cancelling: false,
            crashed: false,
            header: Some(header),
            src: slot.take(),
            dst: None,
            attached: 0,
            busy_held: false,
            victims: Vec::new(),
        };
        self.inflight_store.insert(k, id);
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    fn submit_load(
        &mut self,
        key: ChunkKey,
        expect: &ChunkHeader,
        slot: &mut Option<PinnedDest>,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        self.gate(false)?;
        let dst = slot.as_ref().ok_or(PlatformError::InvalidLayout)?;
        let k = key.0 .0;
        let entry = self
            .index
            .get(&k)
            .filter(|e| !e.is_evicting() && !e.is_foreign())
            .ok_or(PlatformError::ObjectNotFound(key.0))?;
        if expect.chunk_key != key {
            return Err(PlatformError::InvalidLayout);
        }
        self.validate_chunk_header_against_config(expect)?;
        if entry.payload_len != expect.payload_len
            || dst.total_bytes() != Some(expect.payload_len)
            || dst.block_bytes() as u64 != self.cfg.layout.block_stride
        {
            return Err(PlatformError::InvalidLayout);
        }
        let bytes = entry.phys_bytes;
        let payload_len = expect.payload_len;
        self.credit_check(false, bytes)?;
        let ok = self.index.busy_inc(&k);
        debug_assert!(ok);
        let id = self.next_job;
        self.next_job += 1;
        self.restore_ops += 1;
        self.restore_bytes += bytes;
        let job = Job {
            id,
            kind: Kind::Load,
            inc,
            key,
            park: false,
            bytes,
            payload_len,
            reserved: 0,
            submitted_at: Instant::now(),
            stage: JobStage::Submitted,
            phase: Phase::Queued,
            in_step: false,
            cancelling: false,
            crashed: false,
            header: Some(*expect),
            src: None,
            dst: slot.take(),
            attached: 0,
            busy_held: true,
            victims: Vec::new(),
        };
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    fn submit_flush(&mut self, inc: Incarnation) -> Result<JobId, PlatformError> {
        self.gate(false)?;
        self.credit_check(true, 0)?;
        let id = self.next_job;
        self.next_job += 1;
        self.park_ops += 1;
        let job = Job {
            id,
            kind: Kind::Flush,
            inc,
            key: ChunkKey(ObjectId([0; 32])),
            park: true,
            bytes: 0,
            payload_len: 0,
            reserved: 0,
            submitted_at: Instant::now(),
            stage: JobStage::Submitted,
            phase: Phase::Queued,
            in_step: false,
            cancelling: false,
            crashed: false,
            header: None,
            src: None,
            dst: None,
            attached: 0,
            busy_held: false,
            victims: Vec::new(),
        };
        self.jobs.insert(id, job);
        Ok(JobId(id))
    }

    fn reap_cancels(&mut self) -> bool {
        let ids: Vec<u64> = self
            .jobs
            .iter()
            .filter(|(_, j)| j.cancelling && !j.in_step && !j.crashed)
            .map(|(id, _)| *id)
            .collect();
        let any = !ids.is_empty();
        for id in ids {
            let job = self.jobs.remove(&id).expect("cancelling job");
            let stage = job.stage;
            self.finish(
                job,
                Terminal::Cancelled,
                stage,
                PublishOutcome::NotApplicable,
            );
        }
        any
    }
}

fn check_fault(
    hook: &Option<Arc<dyn FaultHook>>,
    boundary: Boundary,
    key: &ChunkKey,
) -> Option<FaultAction> {
    hook.as_ref().and_then(|h| h.at(boundary, key))
}

/// What one recovery entry came to, applied under a short index lock.
enum EntryOutcome {
    Skip,
    TmpDeleted(u64),
    TmpStuck(u64),
    Foreign(u64),
    Unindexed(u64),
    CorruptDeleted(u64),
    CorruptStuck(u64),
    Unreadable(u64),
    Index(IndexEntry, bool),
}

fn is_not_found(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::NotFound
}

/// Unlink with the outcome recovery and purge need: `Ok(true)` removed,
/// `Ok(false)` already gone, `Err` still there.
fn unlink_counted(path: &Path) -> std::io::Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if is_not_found(&e) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Examine one directory entry of the recovery walk. Runs with the index lock
/// NOT held (stat, header read, unlink); only the cap check takes it briefly.
fn recover_entry(
    shared: &Shared,
    cfg: &HostFsConfig,
    stage: WalkStage,
    path: &Path,
) -> std::io::Result<EntryOutcome> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parsed = match stage {
        WalkStage::Shard(s) => parse_kvc_filename(name, s as u8),
        _ => None,
    };
    let hook_key = ChunkKey(ObjectId(parsed.unwrap_or([0; 32])));
    match check_fault(&cfg.faults, Boundary::RecoverEntry, &hook_key) {
        Some(FaultAction::Stall(d)) => std::thread::sleep(d),
        Some(FaultAction::Fail(errno)) => return Err(std::io::Error::from_raw_os_error(errno)),
        _ => {}
    }
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if is_not_found(&e) => return Ok(EntryOutcome::Skip),
        Err(e) => return Err(e),
    };
    let phys = phys_bytes_of(&meta);
    if stage == WalkStage::Tmp {
        if !(meta.is_file() && name.ends_with(".part")) {
            return Ok(EntryOutcome::Foreign(meta.len()));
        }
        return Ok(match unlink_counted(path) {
            Ok(_) => EntryOutcome::TmpDeleted(meta.len()),
            Err(_) => EntryOutcome::TmpStuck(phys),
        });
    }
    // A symlink, a non-regular file or another owner's file is not ours:
    // `foreign_file`, left alone, never read (ADR 0003 sections 4.1 and 19).
    let Some(file_key) = parsed.filter(|_| meta.is_file() && meta.uid() == euid()) else {
        return Ok(EntryOutcome::Foreign(meta.len()));
    };
    let full = {
        let g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        (g.index.len() as u64) >= g.cfg.max_objects
    };
    if full {
        return Ok(EntryOutcome::Unindexed(phys));
    }
    // Re-check through the descriptor: what is read is what was checked.
    let (file, meta) = match open_private_read(path) {
        Ok(PrivateOpen::Ours(f, m)) => (f, m),
        Ok(PrivateOpen::NotOurs(len)) => return Ok(EntryOutcome::Foreign(len)),
        Err(e) if is_not_found(&e) => return Ok(EntryOutcome::Skip),
        // Cannot open or tighten it: do not delete what might be good data;
        // count it.
        Err(_) => return Ok(EntryOutcome::Unreadable(phys)),
    };
    let phys = phys_bytes_of(&meta);
    let file_size = meta.len();
    let kind = if file_size < HEADER_REGION_LEN as u64 {
        RecoveredHeader::Corrupt
    } else {
        let mut buf = [0u8; HEADER_REGION_LEN];
        match file.read_exact_at(&mut buf, 0) {
            Ok(()) => classify_recovered_header(&buf, file_size, &file_key, cfg),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => RecoveredHeader::Corrupt,
            // Cannot read it: do not delete what might be good data; count it.
            Err(_) => return Ok(EntryOutcome::Unreadable(phys)),
        }
    };
    drop(file);
    let mtime_ns = meta_mtime_ns(&meta);
    Ok(match kind {
        RecoveredHeader::Ours { payload_len } => EntryOutcome::Index(
            IndexEntry {
                key: file_key,
                phys_bytes: phys,
                mtime_ns,
                payload_len,
                flags: 0,
            },
            false,
        ),
        RecoveredHeader::Foreign { payload_len } => EntryOutcome::Index(
            IndexEntry {
                key: file_key,
                phys_bytes: phys,
                mtime_ns,
                payload_len,
                flags: FLAG_FOREIGN,
            },
            true,
        ),
        RecoveredHeader::Corrupt => match unlink_counted(path) {
            Ok(_) => EntryOutcome::CorruptDeleted(phys),
            Err(_) => EntryOutcome::CorruptStuck(phys),
        },
    })
}

impl Inner {
    fn apply_recovered(&mut self, out: EntryOutcome) {
        self.stats.dir_entries += 1;
        self.stats.metadata_calls += 1;
        self.stats.io_ops += 1;
        self.stats.walk_peak_buffered = self.stats.walk_peak_buffered.max(1);
        let unindexed = |g: &mut Inner, phys: u64| {
            g.unindexed_objects += 1;
            g.unindexed_bytes += phys;
            g.recovery.unindexed_objects += 1;
            g.recovery.unindexed_bytes += phys;
            g.recovery.bytes += phys;
        };
        match out {
            EntryOutcome::Skip => {}
            EntryOutcome::TmpDeleted(len) => {
                self.stats.unlinks += 1;
                self.stats.io_ops += 1;
                self.recovery.tmp_deleted += 1;
                self.stats.tmp_bytes_deleted += len;
            }
            EntryOutcome::TmpStuck(phys) => {
                self.stats.recovery_unlink_failures += 1;
                self.stuck_bytes += phys;
                self.recovery_stuck = true;
            }
            EntryOutcome::Foreign(len) => {
                self.stats.foreign_files += 1;
                self.stats.foreign_file_bytes += len;
            }
            EntryOutcome::Unindexed(phys) => {
                unindexed(self, phys);
                self.stats.over_cap = true;
            }
            EntryOutcome::CorruptDeleted(phys) => {
                self.stats.unlinks += 1;
                self.stats.io_ops += 1;
                self.recovery.corrupt_deleted += 1;
                self.stats.corrupt_deleted_total += 1;
                self.stats.corrupt_bytes_deleted += phys;
            }
            EntryOutcome::CorruptStuck(phys) | EntryOutcome::Unreadable(phys) => {
                if matches!(out, EntryOutcome::CorruptStuck(_)) {
                    self.stats.recovery_unlink_failures += 1;
                }
                unindexed(self, phys);
                self.recovery_stuck = true;
            }
            EntryOutcome::Index(e, foreign) => {
                self.stats.reads += 1;
                self.stats.read_bytes += HEADER_REGION_LEN as u64;
                self.stats.io_ops += 1;
                self.last_clock_ns = self.last_clock_ns.max(e.mtime_ns);
                self.index.insert(e);
                if foreign {
                    self.recovery.foreign_namespace += 1;
                } else {
                    self.recovery.indexed += 1;
                }
                self.recovery.bytes += e.phys_bytes;
            }
        }
        if matches!(
            out,
            EntryOutcome::CorruptDeleted(_) | EntryOutcome::CorruptStuck(_)
        ) {
            self.stats.reads += 1;
            self.stats.read_bytes += HEADER_REGION_LEN as u64;
        }
    }

    fn fail_recovery(&mut self) {
        self.state = StoreState::Failed;
        self.recovery.failed = true;
        self.recovery.done = false;
    }

    fn progress(&self) -> RecoveryProgress {
        let mut p = self.recovery;
        if self.state == StoreState::Failed {
            p.failed = true;
            p.done = false;
        }
        p
    }
}

/// One slice of the resumable recovery walk (ADR 4.1). The walk streams one
/// directory entry at a time from a `ReadDir` held under its own mutex; the
/// index lock is taken only to apply each entry's outcome, never across a
/// listing, stat, header read or unlink. A listing or stat error fails the
/// store (`Failed`, never done, admission never opens on a partial walk);
/// an unlink that fails leaves the file's bytes counted and the store
/// `Degraded` when the walk ends (Grok C4 P1-2, P1-3, P2-6).
fn recover_step_on(shared: &Shared, slice_ms: u64) -> RecoveryProgress {
    let start = Instant::now();
    let budget = Duration::from_millis(slice_ms);
    let mut walk = shared.walk.lock().unwrap_or_else(|e| e.into_inner());
    let (root, cfg) = {
        let g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.state == StoreState::Failed || g.recovery.done {
            return g.progress();
        }
        (g.cfg.root.clone(), g.cfg.clone())
    };
    let mut processed = 0usize;
    loop {
        if processed >= 1 && start.elapsed() >= budget {
            break;
        }
        let next = walk.next(&root);
        let outcome = match next {
            Ok(Some((stage, path))) => {
                processed += 1;
                recover_entry(shared, &cfg, stage, &path).map(Some)
            }
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        };
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        match outcome {
            Err(_) => {
                g.fail_recovery();
                break;
            }
            Ok(Some(out)) => g.apply_recovered(out),
            Ok(None) => {
                g.recovery.done = true;
                if g.state == StoreState::Recovering {
                    g.state = if g.recovery_stuck {
                        StoreState::Degraded
                    } else {
                        StoreState::Open
                    };
                }
                if g.cfg.purge_after_recovery
                    && (g.unindexed_objects > 0 || g.used_bytes() > g.cfg.quota_bytes)
                {
                    g.purge_pending = true;
                    g.purge_walking = g.unindexed_objects > 0;
                    g.purge_evict_quota = g.used_bytes() > g.cfg.quota_bytes;
                }
                break;
            }
        }
    }
    drop(walk);
    let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
    g.recovery.elapsed_us += start.elapsed().as_micros() as u64;
    let res = g.progress();
    let garbage = std::mem::take(&mut g.graveyard);
    drop(g);
    drop(garbage);
    shared.cv.notify_all();
    res
}

fn run_one_unlink(shared: &Shared, u: Unlink) {
    let (root, hook) = {
        let g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        (g.cfg.root.clone(), g.cfg.faults.clone())
    };
    let k = ChunkKey(ObjectId(u.key));
    let action = check_fault(&hook, Boundary::Unlink, &k);
    if let Some(FaultAction::Stall(d)) = action {
        std::thread::sleep(d);
    }
    let path = object_path_for(&root, &u.key);
    let outcome: std::io::Result<bool> = match action {
        Some(FaultAction::Fail(errno)) => Err(std::io::Error::from_raw_os_error(errno)),
        Some(FaultAction::Crash) => return,
        _ => unlink_counted(&path),
    };

    let garbage = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.io_ops += 1;
        match outcome {
            // Still on disk: back in service, still counted. No job waits on
            // an unlink, so nothing else changes (the store that chose this
            // victim already published; quota is over until a later eviction).
            Err(_) => g.index.clear_evicting(&u.key),
            Ok(_) => {
                if let Some(removed) = g.index.remove(&u.key) {
                    g.stats.unlinks += 1;
                    if u.evict {
                        g.stats.evicted_bytes_total += removed.phys_bytes;
                    }
                }
            }
        }
        std::mem::take(&mut g.graveyard)
    };
    drop(garbage);
    shared.cv.notify_all();
}

impl Inner {
    fn end_purge_walk(&mut self) {
        self.purge_walking = false;
        if !self.purge_evict_quota {
            self.purge_pending = false;
        }
    }
}

/// One step of the unindexed-file purge: stream the shards, unlink one `.kvc`
/// file that is not indexed and not being stored. Accounting is subtracted
/// only after the unlink succeeds; a failed unlink stops the purge with the
/// counts intact (Grok C4 P1-3). A clean pass leaves nothing unindexed.
fn purge_walk_step(shared: &Shared, root: &Path) -> bool {
    let mut walk = shared.purge_walk.lock().unwrap_or_else(|e| e.into_inner());
    let next = walk.next(root);
    let (stage, path) = match next {
        Ok(Some(x)) => x,
        Ok(None) => {
            *walk = Walk::new(WalkStage::Shard(0));
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.unindexed_objects = 0;
            g.unindexed_bytes = 0;
            g.stats.over_cap = false;
            g.end_purge_walk();
            return true;
        }
        Err(_) => {
            *walk = Walk::new(WalkStage::Shard(0));
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.purge_pending = false;
            g.purge_walking = false;
            g.purge_evict_quota = false;
            return true;
        }
    };
    let WalkStage::Shard(s) = stage else {
        return true;
    };
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(key) = parse_kvc_filename(name, s as u8) else {
        return true;
    };
    {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.dir_entries += 1;
        g.stats.walk_peak_buffered = g.stats.walk_peak_buffered.max(1);
        if g.index.get(&key).is_some() || g.inflight_store.contains_key(&key) {
            return true;
        }
        g.purging = Some(key);
    }
    let (phys, res) = match std::fs::symlink_metadata(&path) {
        Ok(m) => (phys_bytes_of(&m), unlink_counted(&path)),
        Err(e) if is_not_found(&e) => (0, Ok(false)),
        Err(e) => (0, Err(e)),
    };
    let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
    g.purging = None;
    g.stats.io_ops += 2;
    g.stats.metadata_calls += 1;
    match res {
        Ok(true) => {
            g.unindexed_objects = g.unindexed_objects.saturating_sub(1);
            g.unindexed_bytes = g.unindexed_bytes.saturating_sub(phys);
            g.stats.unlinks += 1;
            g.stats.purge_unlinked += 1;
        }
        Ok(false) => {}
        Err(_) => {
            g.stats.purge_unlink_failures += 1;
            g.purge_pending = false;
            g.purge_walking = false;
            g.purge_evict_quota = false;
            drop(g);
            *walk = Walk::new(WalkStage::Shard(0));
        }
    }
    true
}

fn run_one_purge_step(shared: &Shared) -> bool {
    let (root, walking) = {
        let g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.purge_pending || !g.recovery.done || g.state == StoreState::Failed {
            return false;
        }
        (g.cfg.root.clone(), g.purge_walking)
    };
    if walking {
        let did = purge_walk_step(shared, &root);
        shared.cv.notify_all();
        return did;
    }

    let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
    if g.purge_evict_quota && (g.used_bytes() as u128) * 10 > (g.cfg.quota_bytes as u128) * 9 {
        if let Some(victim) = g.index.pick_victim() {
            g.index.mark_evicting(&victim);
            let path = object_path_for(&root, &victim);
            drop(g);
            let ok = unlink_counted(&path).is_ok();
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.io_ops += 1;
            if ok {
                if let Some(removed) = g.index.remove(&victim) {
                    g.stats.unlinks += 1;
                    g.stats.evicted_bytes_total += removed.phys_bytes;
                }
            } else {
                g.index.clear_evicting(&victim);
                g.purge_pending = false;
                g.purge_evict_quota = false;
            }
            if (g.used_bytes() as u128) * 10 <= (g.cfg.quota_bytes as u128) * 9 {
                g.purge_pending = false;
                g.purge_evict_quota = false;
            }
            drop(g);
            shared.cv.notify_all();
            return true;
        }
    }

    g.purge_pending = false;
    g.purge_evict_quota = false;
    drop(g);
    shared.cv.notify_all();
    false
}

fn execute_store_job(shared: &Shared, job_id: u64) {
    let (root, hook, key, inc, mut file_bytes) = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = g.jobs.get(&job_id) else {
            return;
        };
        if job.cancelling {
            let job = g.jobs.remove(&job_id).unwrap();
            let stage = job.stage;
            g.finish(
                job,
                Terminal::Cancelled,
                stage,
                PublishOutcome::NotApplicable,
            );
            let garbage = std::mem::take(&mut g.graveyard);
            drop(g);
            drop(garbage);
            shared.cv.notify_all();
            return;
        }

        // Make room by marking victims. They are NOT unlinked yet: the write
        // goes ahead at once, and `finish` queues their unlinks only when the
        // new object is durable (or returns them to service on any other
        // ending). While a purge is pending nothing is evicted for room.
        while g.bytes_after() > g.cfg.quota_bytes || g.count_after() > g.cfg.max_objects {
            let victim = if g.purge_pending {
                None
            } else {
                g.index.pick_victim()
            };
            let Some(victim) = victim else {
                let job = g.jobs.remove(&job_id).unwrap();
                g.stats.disk_full_total += 1;
                g.fail_job(job, PlatformError::OutOfMemory);
                let garbage = std::mem::take(&mut g.graveyard);
                drop(g);
                drop(garbage);
                shared.cv.notify_all();
                return;
            };
            g.index.mark_evicting(&victim);
            g.jobs.get_mut(&job_id).unwrap().victims.push(victim);
        }

        let cfg_layout = g.cfg.layout;
        let num_layers = g.cfg.num_layers;
        let num_kv_heads = g.cfg.num_kv_heads;
        let head_dim = g.cfg.head_dim;
        let block_size = g.cfg.block_size;

        let job = g.jobs.get(&job_id).unwrap();
        let header = job.header.expect("store header");
        let key = job.key;
        let inc = job.inc;
        let src = job.src.as_ref().expect("store source");
        let n_blocks = src.block_count();
        let tail_valid = src.tail_valid_tokens();
        let mut payload = Vec::with_capacity(header.payload_len as usize);
        let mut digests = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            match src.block(i) {
                Ok(b) => {
                    if i + 1 == n_blocks && tail_valid < block_size {
                        let mut scratch = shared
                            .tail_scratch
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        zero_tail_block(
                            &mut scratch,
                            b,
                            &cfg_layout,
                            num_layers,
                            num_kv_heads,
                            head_dim,
                            block_size,
                            tail_valid,
                        );
                        let mut h = Sha256::new();
                        h.update(&*scratch);
                        digests.push(h.finalize().into());
                        payload.extend_from_slice(&scratch);
                    } else {
                        let mut h = Sha256::new();
                        h.update(b);
                        digests.push(h.finalize().into());
                        payload.extend_from_slice(b);
                    }
                }
                Err(e) => {
                    g.stats.released_source_violations += 1;
                    let job = g.jobs.remove(&job_id).unwrap();
                    g.fail_job(job, e);
                    let garbage = std::mem::take(&mut g.graveyard);
                    drop(g);
                    drop(garbage);
                    shared.cv.notify_all();
                    return;
                }
            }
        }

        let rec = RecordHeader {
            chunk: header,
            payload_sha256: payload_root(&digests),
        };
        let region = match rec.encode_region(&digests) {
            Ok(r) => r,
            Err(_) => {
                let job = g.jobs.remove(&job_id).unwrap();
                g.fail_job(job, PlatformError::InvalidLayout);
                let garbage = std::mem::take(&mut g.graveyard);
                drop(g);
                drop(garbage);
                shared.cv.notify_all();
                return;
            }
        };
        let mut file_bytes = Vec::with_capacity(HEADER_REGION_LEN + payload.len());
        file_bytes.extend_from_slice(&region);
        file_bytes.extend_from_slice(&payload);

        let job = g.jobs.get_mut(&job_id).unwrap();
        job.phase = Phase::Writing;
        job.in_step = true;
        (
            g.cfg.root.clone(),
            g.cfg.faults.clone(),
            key,
            inc,
            file_bytes,
        )
    };

    // SEAL SEAM (#506): encrypt the payload in place and fill the seal slot,
    // off the lock. Nothing reaches the filesystem unsealed.
    if shared.seal_key.seal_file(&mut file_bytes).is_err() {
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(job) = g.jobs.remove(&job_id) {
                g.fail_job(job, PlatformError::HardwareFault("seal failed".into()));
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    let tpath = tmp_path_for(&root, &key.0 .0, job_id, inc);
    let opath = object_path_for(&root, &key.0 .0);
    let shard_dir = opath.parent().expect("shard dir").to_path_buf();

    // P1: Write temporary file.
    let p1 = check_fault(&hook, Boundary::P1Write, &key);
    if let Some(FaultAction::Stall(d)) = p1 {
        std::thread::sleep(d);
    }
    match p1 {
        Some(FaultAction::Crash) => {
            let _ = write_private_file(&tpath, &file_bytes);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::TornWrite { after_bytes, crash }) => {
            let cut = (after_bytes as usize).min(file_bytes.len());
            let wrote = write_private_file(&tpath, &file_bytes[..cut]).is_ok();
            if crash {
                let garbage = {
                    let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(job) = g.jobs.remove(&job_id) {
                        g.mark_crashed(job);
                    }
                    std::mem::take(&mut g.graveyard)
                };
                drop(garbage);
                shared.cv.notify_all();
                return;
            }
            if wrote {
                let _ = std::fs::remove_file(&tpath);
            }
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.writes += 1;
                g.stats.write_bytes += cut as u64;
                g.stats.unlinks += 1;
                g.stats.io_ops += 2;
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, PlatformError::HardwareFault("torn write".into()));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => {
            // Only remove what this call created: a failed create (say, a
            // planted symlink) leaves the path alone.
            if write_private_file(&tpath, &file_bytes[..HEADER_REGION_LEN]).is_ok() {
                let _ = std::fs::remove_file(&tpath);
            }
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.unlinks += 1;
                g.stats.io_ops += 2;
                if errno == ENOSPC {
                    g.stats.disk_full_total += 1;
                }
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, errno_to_error(errno, "write temp failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        _ => {}
    }

    // `created`: the open succeeded, so a failed write leaves a temp of ours
    // to remove. A failed open (a planted symlink: O_NOFOLLOW gives ELOOP)
    // created nothing and the path is left alone.
    let mut created = false;
    let write_res = create_private_file(&tpath).and_then(|mut f| {
        created = true;
        f.write_all(&file_bytes)?;
        Ok(f)
    });
    let tmp_file = match write_res {
        Ok(f) => f,
        Err(e) => {
            if created {
                let _ = std::fs::remove_file(&tpath);
            }
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.io_ops += 1;
                if e.raw_os_error() == Some(ENOSPC) {
                    g.stats.disk_full_total += 1;
                }
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, io_to_error(&e, "write temp failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
    };

    {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.writes += 1;
        g.stats.write_bytes += file_bytes.len() as u64;
        g.stats.io_ops += 1;
        let cancelled = g.jobs.get(&job_id).is_none_or(|j| j.cancelling);
        if cancelled {
            drop(tmp_file);
            let _ = std::fs::remove_file(&tpath);
            g.stats.unlinks += 1;
            g.stats.io_ops += 1;
            if let Some(job) = g.jobs.remove(&job_id) {
                let stage = job.stage;
                g.finish(
                    job,
                    Terminal::Cancelled,
                    stage,
                    PublishOutcome::NotApplicable,
                );
            }
            let garbage = std::mem::take(&mut g.graveyard);
            drop(g);
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        if let Some(job) = g.jobs.get_mut(&job_id) {
            job.phase = Phase::Syncing;
        }
    }

    // P2: fsync temporary file.
    let p2 = check_fault(&hook, Boundary::P2FsyncFile, &key);
    if let Some(FaultAction::Stall(d)) = p2 {
        std::thread::sleep(d);
    }
    match p2 {
        Some(FaultAction::Crash) => {
            drop(tmp_file);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => {
            drop(tmp_file);
            let _ = std::fs::remove_file(&tpath);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.file_fsyncs += 1;
                g.stats.unlinks += 1;
                g.stats.io_ops += 2;
                g.set_degraded();
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, errno_to_error(errno, "fsync(file) failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        _ => {}
    }

    if let Err(e) = tmp_file.sync_all() {
        drop(tmp_file);
        let _ = std::fs::remove_file(&tpath);
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.file_fsyncs += 1;
            g.stats.unlinks += 1;
            g.stats.io_ops += 2;
            g.set_degraded();
            if let Some(job) = g.jobs.remove(&job_id) {
                g.fail_job(job, io_to_error(&e, "fsync(file) failed"));
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }
    drop(tmp_file);

    {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.file_fsyncs += 1;
        g.stats.io_ops += 1;
        if let Some(job) = g.jobs.get_mut(&job_id) {
            job.stage = JobStage::Completed;
            job.phase = Phase::Publishing;
            let snap = Completion {
                job: JobId(job.id),
                incarnation: job.inc,
                key: job.key,
                stage: JobStage::Completed,
                terminal: None,
                publish: PublishOutcome::NotApplicable,
                bytes: job.payload_len,
                elapsed_us: job.submitted_at.elapsed().as_micros() as u64,
                holder: None,
            };
            g.push_completion(snap);
        }
    }

    // P3: atomic rename of temporary file into shard directory.
    let p3 = check_fault(&hook, Boundary::P3Rename, &key);
    if let Some(FaultAction::Stall(d)) = p3 {
        std::thread::sleep(d);
    }
    match p3 {
        Some(FaultAction::Crash) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => {
            let _ = std::fs::remove_file(&tpath);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.unlinks += 1;
                g.stats.io_ops += 2;
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, errno_to_error(errno, "rename failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        _ => {}
    }

    if let Err(e) = std::fs::rename(&tpath, &opath) {
        let _ = std::fs::remove_file(&tpath);
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.unlinks += 1;
            g.stats.io_ops += 2;
            if let Some(job) = g.jobs.remove(&job_id) {
                g.fail_job(job, io_to_error(&e, "rename failed"));
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    let (phys_bytes, file_mtime_ns) = match std::fs::symlink_metadata(&opath) {
        Ok(m) => (phys_bytes_of(&m), meta_mtime_ns(&m)),
        Err(_) => (file_bytes.len() as u64, system_now_ns()),
    };

    {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.renames += 1;
        g.stats.metadata_calls += 1;
        g.stats.io_ops += 2;
        if let Some(job) = g.jobs.get_mut(&job_id) {
            job.stage = JobStage::Published;
        }
    }

    // P4: fsync parent shard directory.
    let p4 = check_fault(&hook, Boundary::P4FsyncDir, &key);
    if let Some(FaultAction::Stall(d)) = p4 {
        std::thread::sleep(d);
    }
    let dir_sync_res: Result<(), PlatformError> = match p4 {
        Some(FaultAction::Crash) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => Err(errno_to_error(errno, "fsync(dir) failed")),
        _ => fsync_dir(&shard_dir).map_err(|e| io_to_error(&e, "fsync(dir) failed")),
    };

    if let Err(err) = dir_sync_res {
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.dir_fsyncs += 1;
            g.stats.io_ops += 1;
            g.stats.published_total += 1;
            g.stats.indeterminate_total += 1;
            g.set_degraded();
            let mtime_ns = g.next_timestamp_ns(file_mtime_ns);
            if let Some(job) = g.jobs.remove(&job_id) {
                g.index.insert(IndexEntry {
                    key: key.0 .0,
                    phys_bytes,
                    mtime_ns,
                    payload_len: job.payload_len,
                    flags: FLAG_INDETERMINATE,
                });
                g.finish(
                    job,
                    Terminal::Failed(err),
                    JobStage::Published,
                    PublishOutcome::Indeterminate,
                );
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.dir_fsyncs += 1;
        g.stats.io_ops += 1;
    }

    // P5: insert entry into in-memory index.
    let p5 = check_fault(&hook, Boundary::P5IndexInsert, &key);
    if let Some(FaultAction::Stall(d)) = p5 {
        std::thread::sleep(d);
    }
    if matches!(p5, Some(FaultAction::Crash)) {
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(job) = g.jobs.remove(&job_id) {
                g.mark_crashed(job);
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    let garbage = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mtime_ns = g.next_timestamp_ns(file_mtime_ns);
        g.stats.published_total += 1;
        if let Some(mut job) = g.jobs.remove(&job_id) {
            g.index.insert(IndexEntry {
                key: key.0 .0,
                phys_bytes,
                mtime_ns,
                payload_len: job.payload_len,
                flags: 0,
            });
            job.stage = JobStage::Durable;
            g.finish(
                job,
                Terminal::Done(JobStage::Durable),
                JobStage::Durable,
                PublishOutcome::Durable,
            );
        }
        std::mem::take(&mut g.graveyard)
    };
    drop(garbage);
    shared.cv.notify_all();
}

fn execute_load_job(shared: &Shared, job_id: u64) {
    let (root, hook, key, expect, block_stride) = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = g.jobs.get_mut(&job_id) else {
            return;
        };
        if job.cancelling {
            let job = g.jobs.remove(&job_id).unwrap();
            let stage = job.stage;
            g.finish(
                job,
                Terminal::Cancelled,
                stage,
                PublishOutcome::NotApplicable,
            );
            let garbage = std::mem::take(&mut g.graveyard);
            drop(g);
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        job.phase = Phase::Reading;
        job.in_step = true;
        let key = job.key;
        let expect = job.header.expect("load expect");
        (
            g.cfg.root.clone(),
            g.cfg.faults.clone(),
            key,
            expect,
            g.cfg.layout.block_stride,
        )
    };

    let lr = check_fault(&hook, Boundary::LoadRead, &key);
    if let Some(FaultAction::Stall(d)) = lr {
        std::thread::sleep(d);
    }
    match lr {
        Some(FaultAction::Crash) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.io_ops += 1;
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, errno_to_error(errno, "load read failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        _ => {}
    }

    let opath = object_path_for(&root, &key.0 .0);
    // No symlink followed, checked through the descriptor: a link, a
    // non-regular file or another owner's file is never served; it leaves
    // the index as not found and the file is left alone (ADR 0003 section 19).
    let (file, meta) = match open_private_read(&opath) {
        Ok(PrivateOpen::Ours(f, m)) => (f, m),
        Err(e) if !is_not_found(&e) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.metadata_calls += 1;
                g.stats.io_ops += 1;
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, io_to_error(&e, "open object failed"));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Ok(PrivateOpen::NotOurs(_)) | Err(_) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.metadata_calls += 1;
                g.stats.io_ops += 1;
                if let Some(mut job) = g.jobs.remove(&job_id) {
                    if job.busy_held {
                        g.index.busy_dec(&key.0 .0);
                        job.busy_held = false;
                    }
                    g.index.remove(&key.0 .0);
                    g.fail_job(job, PlatformError::ObjectNotFound(key.0));
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
    };

    let file_size = meta.len();
    if file_size < HEADER_REGION_LEN as u64 {
        let _ = std::fs::remove_file(&opath);
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.metadata_calls += 1;
            g.stats.unlinks += 1;
            g.stats.io_ops += 2;
            g.stats.corrupt_deleted_total += 1;
            if let Some(mut job) = g.jobs.remove(&job_id) {
                if job.busy_held {
                    g.index.busy_dec(&key.0 .0);
                    job.busy_held = false;
                }
                if let Some(rem) = g.index.remove(&key.0 .0) {
                    g.stats.corrupt_bytes_deleted += rem.phys_bytes;
                }
                g.fail_job(job, PlatformError::InvalidLayout);
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    let mut header_buf = [0u8; HEADER_REGION_LEN];
    if file.read_exact_at(&mut header_buf, 0).is_err()
        || validate_record(&header_buf, file_size, &expect, block_stride).is_err()
    {
        drop(file);
        let _ = std::fs::remove_file(&opath);
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.metadata_calls += 1;
            g.stats.reads += 1;
            g.stats.read_bytes += HEADER_REGION_LEN as u64;
            g.stats.unlinks += 1;
            g.stats.io_ops += 3;
            g.stats.corrupt_deleted_total += 1;
            if let Some(mut job) = g.jobs.remove(&job_id) {
                if job.busy_held {
                    g.index.busy_dec(&key.0 .0);
                    job.busy_held = false;
                }
                if let Some(rem) = g.index.remove(&key.0 .0) {
                    g.stats.corrupt_bytes_deleted += rem.phys_bytes;
                }
                g.fail_job(job, PlatformError::InvalidLayout);
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    let n_seg = (expect.payload_len / block_stride) as usize;
    let seg_len = block_stride as usize;
    let mut reads_done = 1u64;
    let mut bytes_read = HEADER_REGION_LEN as u64;

    // SEAL SEAM (#506): read the whole ciphertext, then authenticate and
    // decrypt it before any segment reaches the destination. A refusal (wrong
    // key, tampered bytes, swapped header, no seal) discards the object like
    // any corrupt one; the caller recomputes.
    let mut payload = vec![0u8; expect.payload_len as usize];
    let read_ok = file
        .read_exact_at(&mut payload, crate::record::PAYLOAD_OFFSET as u64)
        .is_ok();
    if read_ok {
        reads_done += 1;
        bytes_read += expect.payload_len;
    }
    if !read_ok
        || shared
            .seal_key
            .open_payload(&header_buf, &mut payload)
            .is_err()
    {
        drop(file);
        let _ = std::fs::remove_file(&opath);
        let garbage = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.stats.metadata_calls += 1;
            g.stats.reads += reads_done;
            g.stats.read_bytes += bytes_read;
            g.stats.unlinks += 1;
            g.stats.io_ops += 2 + reads_done;
            g.stats.corrupt_deleted_total += 1;
            if let Some(mut job) = g.jobs.remove(&job_id) {
                if job.busy_held {
                    g.index.busy_dec(&key.0 .0);
                    job.busy_held = false;
                }
                if let Some(rem) = g.index.remove(&key.0 .0) {
                    g.stats.corrupt_bytes_deleted += rem.phys_bytes;
                }
                g.fail_job(job, PlatformError::InvalidLayout);
            }
            std::mem::take(&mut g.graveyard)
        };
        drop(garbage);
        shared.cv.notify_all();
        return;
    }

    for i in 0..n_seg {
        let seg_buf = &payload[i * seg_len..(i + 1) * seg_len];
        let mut h = Sha256::new();
        h.update(seg_buf);
        let digest: [u8; 32] = h.finalize().into();
        let want = &header_buf[256 + 32 * i..256 + 32 * (i + 1)];
        if digest != want {
            drop(file);
            let _ = std::fs::remove_file(&opath);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.metadata_calls += 1;
                g.stats.reads += reads_done;
                g.stats.read_bytes += bytes_read;
                g.stats.unlinks += 1;
                g.stats.io_ops += 2 + reads_done;
                g.stats.corrupt_deleted_total += 1;
                if let Some(mut job) = g.jobs.remove(&job_id) {
                    if job.busy_held {
                        g.index.busy_dec(&key.0 .0);
                        job.busy_held = false;
                    }
                    if let Some(rem) = g.index.remove(&key.0 .0) {
                        g.stats.corrupt_bytes_deleted += rem.phys_bytes;
                    }
                    g.fail_job(job, PlatformError::InvalidLayout);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }

        let write_err = {
            let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            let dst = g
                .jobs
                .get_mut(&job_id)
                .and_then(|j| j.dst.as_mut())
                .expect("load dst");
            dst.write_block(i, seg_buf).err()
        };
        if let Some(err) = write_err {
            drop(file);
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.stats.metadata_calls += 1;
                g.stats.reads += reads_done;
                g.stats.read_bytes += bytes_read;
                g.stats.io_ops += 1 + reads_done;
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, err);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
    }
    drop(file);

    let garbage = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.metadata_calls += 1;
        g.stats.reads += reads_done;
        g.stats.read_bytes += bytes_read;
        g.stats.io_ops += 1 + reads_done;
        let ts = g.next_timestamp_ns(0);
        g.index.touch(&key.0 .0, ts);
        if let Some(mut job) = g.jobs.remove(&job_id) {
            job.stage = JobStage::Completed;
            g.finish(
                job,
                Terminal::Done(JobStage::Completed),
                JobStage::Completed,
                PublishOutcome::NotApplicable,
            );
        }
        std::mem::take(&mut g.graveyard)
    };
    drop(garbage);
    shared.cv.notify_all();
}

fn execute_flush_job(shared: &Shared, job_id: u64) {
    let (root, hook, key) = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = g.jobs.get_mut(&job_id) else {
            return;
        };
        job.phase = Phase::Flushing;
        job.in_step = true;
        let key = job.key;
        (g.cfg.root.clone(), g.cfg.faults.clone(), key)
    };

    let fs = check_fault(&hook, Boundary::FlushSync, &key);
    if let Some(FaultAction::Stall(d)) = fs {
        std::thread::sleep(d);
    }
    let res: Result<(), PlatformError> = match fs {
        Some(FaultAction::Crash) => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.mark_crashed(job);
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
            return;
        }
        Some(FaultAction::Fail(errno)) => Err(errno_to_error(errno, "flush failed")),
        _ => fsync_dir(&root.join("objects"))
            .and_then(|()| fsync_dir(&root))
            .map_err(|e| io_to_error(&e, "flush failed")),
    };

    let garbage = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.dir_fsyncs += 1;
        g.stats.io_ops += 1;
        match res {
            Err(e) => {
                g.set_degraded();
                if let Some(job) = g.jobs.remove(&job_id) {
                    g.fail_job(job, e);
                }
            }
            Ok(()) => {
                if let Some(mut job) = g.jobs.remove(&job_id) {
                    job.stage = JobStage::Durable;
                    g.finish(
                        job,
                        Terminal::Done(JobStage::Durable),
                        JobStage::Durable,
                        PublishOutcome::Durable,
                    );
                }
            }
        }
        std::mem::take(&mut g.graveyard)
    };
    drop(garbage);
    shared.cv.notify_all();
}

fn pump_one(shared: &Shared) -> bool {
    let reaped = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let r = g.reap_cancels();
        let garbage = std::mem::take(&mut g.graveyard);
        drop(g);
        drop(garbage);
        r
    };
    if reaped {
        shared.cv.notify_all();
        return true;
    }

    let unlink = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.unlinks.pop_front()
    };
    if let Some(u) = unlink {
        run_one_unlink(shared, u);
        return true;
    }

    if run_one_purge_step(shared) {
        return true;
    }

    let picked: Option<(u64, Kind)> = {
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let next_restore = g
            .jobs
            .iter()
            .find(|(_, j)| {
                !j.park && !j.cancelling && !j.in_step && !j.crashed && j.phase == Phase::Queued
            })
            .map(|(id, j)| (*id, j.kind));
        let next_park = g
            .jobs
            .iter()
            .find(|(_, j)| {
                j.park && !j.cancelling && !j.in_step && !j.crashed && j.phase == Phase::Queued
            })
            .map(|(id, j)| (*id, j.kind));
        match (next_restore, next_park) {
            (Some(r), Some(p)) => {
                if g.restore_streak < 3 {
                    g.restore_streak += 1;
                    Some(r)
                } else {
                    g.restore_streak = 0;
                    Some(p)
                }
            }
            (Some(r), None) => {
                g.restore_streak = g.restore_streak.saturating_add(1);
                Some(r)
            }
            (None, Some(p)) => {
                g.restore_streak = 0;
                Some(p)
            }
            (None, None) => None,
        }
    };

    let Some((job_id, kind)) = picked else {
        return false;
    };

    match kind {
        Kind::Noop => {
            let garbage = {
                let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(mut job) = g.jobs.remove(&job_id) {
                    job.stage = JobStage::Durable;
                    g.finish(
                        job,
                        Terminal::Done(JobStage::Durable),
                        JobStage::Durable,
                        PublishOutcome::NotApplicable,
                    );
                }
                std::mem::take(&mut g.graveyard)
            };
            drop(garbage);
            shared.cv.notify_all();
        }
        Kind::Store => execute_store_job(shared, job_id),
        Kind::Load => execute_load_job(shared, job_id),
        Kind::Flush => execute_flush_job(shared, job_id),
    }
    true
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let (need_recover, slice_ms, stop) = {
            let g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            (
                g.state == StoreState::Recovering && !g.recovery.done,
                g.cfg.recovery_slice_ms,
                g.stop_worker,
            )
        };
        if stop {
            break;
        }
        if need_recover {
            recover_step_on(&shared, slice_ms);
            continue;
        }
        if pump_one(&shared) {
            continue;
        }
        let mut g = shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.stop_worker {
            break;
        }
        let has_work = (g.state == StoreState::Recovering && !g.recovery.done)
            || !g.unlinks.is_empty()
            || (g.purge_pending && g.recovery.done)
            || g.jobs
                .values()
                .any(|j| !j.in_step && !j.crashed && (j.cancelling || j.phase == Phase::Queued));
        if !has_work {
            g = shared.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        drop(g);
    }
}

impl HostFsStore {
    pub fn open(config: HostFsConfig) -> Result<Self, PlatformError> {
        validate_config(&config)?;
        // SEAL SEAM (#506): no key, no store (fail closed, nothing touched).
        let seal_key = config.seal_key.clone().ok_or(PlatformError::AccessDenied)?;
        let mut stats = HostFsStats {
            tail_scratch_bytes: config.layout.block_stride,
            ..HostFsStats::default()
        };

        let root = &config.root;
        let objects_dir = root.join("objects");
        let tmp_dir = root.join("tmp");
        let shard_dirs: Vec<PathBuf> = (0..=255u8)
            .map(|b| objects_dir.join(format!("{b:02x}")))
            .collect();

        // Check everything that already exists before creating anything, so
        // a refused store is left exactly as it was (ADR 0003 section 19.3).
        check_trusted_parent(root)?;
        if check_private_dir(root)? {
            check_private_dir(&tmp_dir)?;
            if check_private_dir(&objects_dir)? {
                for shard in &shard_dirs {
                    check_private_dir(shard)?;
                }
            }
        }

        if make_private_dir(root, true)? {
            stats.mkdirs += 1;
            stats.io_ops += 1;
        }
        let dir_lock = lock_store_dir(root)?;
        stats.io_ops += 1;
        for dir in [&objects_dir, &tmp_dir] {
            if make_private_dir(dir, false)? {
                stats.mkdirs += 1;
                stats.io_ops += 1;
            }
        }

        let alloc_unit = std::fs::metadata(&objects_dir)
            .map(|m| m.blksize())
            .unwrap_or(4096)
            .max(512);

        let mut created_shard = false;
        for shard in &shard_dirs {
            if make_private_dir(shard, false)? {
                stats.mkdirs += 1;
                stats.io_ops += 1;
                created_shard = true;
            }
        }
        if created_shard {
            fsync_dir(&objects_dir).map_err(|e| io_to_error(&e, "fsync objects"))?;
            stats.dir_fsyncs += 1;
            stats.io_ops += 1;
        }

        let store_id_path = root.join("STORE_ID");
        let (store_id, state, recovery_done) = if std::fs::symlink_metadata(&store_id_path).is_ok()
        {
            stats.reads += 1;
            stats.io_ops += 1;
            // No symlink followed, a regular file of ours, exactly 16 bytes;
            // anything else fails the store (as an unreadable STORE_ID did).
            let mut arr = [0u8; 16];
            let read = match open_private_read(&store_id_path) {
                Ok(PrivateOpen::Ours(mut f, meta)) if meta.len() == 16 => {
                    f.read_exact(&mut arr).is_ok()
                }
                _ => false,
            };
            if read {
                stats.read_bytes += 16;
                (Some(arr), StoreState::Recovering, false)
            } else {
                (None, StoreState::Failed, false)
            }
        } else {
            let id = generate_store_id();
            let part = tmp_dir.join("STORE_ID.part");
            {
                // A STORE_ID.part left by a crash is ours (tmp/ is verified
                // private) and is truncated, so it never blocks reopening.
                let mut f = create_private_file(&part)
                    .map_err(|e| io_to_error(&e, "create STORE_ID.part"))?;
                f.write_all(&id)
                    .map_err(|e| io_to_error(&e, "write STORE_ID.part"))?;
                stats.writes += 1;
                stats.write_bytes += 16;
                stats.io_ops += 1;
                f.sync_all()
                    .map_err(|e| io_to_error(&e, "fsync STORE_ID.part"))?;
                stats.file_fsyncs += 1;
                stats.io_ops += 1;
            }
            std::fs::rename(&part, &store_id_path)
                .map_err(|e| io_to_error(&e, "rename STORE_ID"))?;
            stats.renames += 1;
            stats.io_ops += 1;
            fsync_dir(root).map_err(|e| io_to_error(&e, "fsync root"))?;
            stats.dir_fsyncs += 1;
            stats.io_ops += 1;
            (Some(id), StoreState::Recovering, false)
        };

        let scratch_len = config.layout.block_stride as usize;
        let worker_mode = config.worker;
        let root_buf = config.root.clone();
        let recovery = RecoveryProgress {
            done: recovery_done,
            failed: state == StoreState::Failed,
            ..RecoveryProgress::default()
        };
        let inner = Inner {
            cfg: config,
            store_id,
            state,
            closed: false,
            stop_worker: false,
            next_job: 1,
            next_pin: 1,
            last_clock_ns: 0,
            index: Index::new(),
            unindexed_bytes: 0,
            unindexed_objects: 0,
            stuck_bytes: 0,
            recovery_stuck: false,
            purge_walking: false,
            purging: None,
            reserved: 0,
            jobs: BTreeMap::new(),
            inflight_store: BTreeMap::new(),
            unlinks: VecDeque::new(),
            pins: BTreeMap::new(),
            restore_ops: 0,
            park_ops: 0,
            restore_bytes: 0,
            park_bytes: 0,
            restore_streak: 0,
            attached_total: 0,
            queue: VecDeque::new(),
            stats,
            recovery,
            purge_pending: false,
            purge_evict_quota: false,
            alloc_unit,
            graveyard: Vec::new(),
        };

        let shared = Arc::new(Shared {
            root: root_buf,
            inner: Mutex::new(inner),
            cv: Condvar::new(),
            tail_scratch: Mutex::new(vec![0u8; scratch_len]),
            walk: Mutex::new(Walk::new(WalkStage::Tmp)),
            purge_walk: Mutex::new(Walk::new(WalkStage::Shard(0))),
            seal_key,
            _dir_lock: dir_lock,
        });

        let handle = if worker_mode == WorkerMode::Thread {
            let s2 = Arc::clone(&shared);
            Some(std::thread::spawn(move || worker_loop(s2)))
        } else {
            None
        };

        Ok(Self {
            shared,
            worker: Mutex::new(handle),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.shared.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let (r, garbage) = {
            let mut g = self.lock();
            let r = f(&mut g);
            (r, std::mem::take(&mut g.graveyard))
        };
        drop(garbage);
        r
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    pub fn object_path(&self, key: &ChunkKey) -> PathBuf {
        object_path_for(&self.shared.root, &key.0 .0)
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.shared.root.join("tmp")
    }

    pub fn store_id(&self) -> Option<[u8; 16]> {
        self.with(|i| i.store_id)
    }

    pub fn stats(&self) -> HostFsStats {
        self.with(|i| i.stats)
    }

    pub fn pump(&self) -> bool {
        pump_one(&self.shared)
    }

    pub fn run_until_idle(&self) {
        while self.pump() {}
    }

    pub fn start_purge(&self) {
        *self
            .shared
            .purge_walk
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Walk::new(WalkStage::Shard(0));
        self.with(|i| {
            i.purge_pending = true;
            i.purge_walking = i.unindexed_objects > 0;
            i.purge_evict_quota = i.used_bytes() > i.cfg.quota_bytes;
        });
        self.shared.cv.notify_all();
    }

    pub fn purge_pending(&self) -> bool {
        self.with(|i| i.purge_pending)
    }

    pub fn audit(&self) -> Result<(), String> {
        self.with(|i| {
            i.index.audit()?;
            let expected_res: u64 = i
                .jobs
                .values()
                .filter(|j| !j.crashed)
                .map(|j| j.reserved)
                .sum();
            if i.reserved != expected_res {
                return Err(format!(
                    "reserved_bytes {} != job sum {}",
                    i.reserved, expected_res
                ));
            }
            Ok(())
        })
    }
}

impl Drop for HostFsStore {
    fn drop(&mut self) {
        {
            let mut g = self.lock();
            g.stop_worker = true;
        }
        self.shared.cv.notify_all();
        if let Some(h) = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
        let garbage = {
            let mut g = self.lock();
            let mut guards = std::mem::take(&mut g.graveyard);
            for (_, mut j) in std::mem::take(&mut g.jobs) {
                if let Some(s) = j.src.take() {
                    guards.push(s.into_guard());
                }
                if let Some(d) = j.dst.take() {
                    guards.push(d.into_guard());
                }
            }
            guards
        };
        drop(garbage);
    }
}

impl KvStore for HostFsStore {
    fn namespace(&self) -> NamespaceId {
        self.with(|i| i.cfg.namespace)
    }

    fn capacity(&self) -> Capacity {
        self.with(|i| i.capacity())
    }

    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>> {
        self.with(|i| {
            let serving = !matches!(i.state, StoreState::Recovering | StoreState::Failed);
            keys.iter()
                .map(|k| {
                    i.index
                        .get(&k.0 .0)
                        .filter(|e| serving && !e.is_evicting() && !e.is_foreign())
                        .map(|e| e.payload_len)
                })
                .collect()
        })
    }

    fn submit_store(
        &self,
        header: ChunkHeader,
        src: PinnedSource,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let mut slot = Some(src);
        let res = self.with(|i| i.submit_store(header, &mut slot, inc));
        drop(slot);
        if res.is_ok() {
            self.shared.cv.notify_all();
        }
        res
    }

    fn submit_load(
        &self,
        key: ChunkKey,
        expect: &ChunkHeader,
        dst: PinnedDest,
        inc: Incarnation,
    ) -> Result<JobId, PlatformError> {
        let mut slot = Some(dst);
        let res = self.with(|i| i.submit_load(key, expect, &mut slot, inc));
        drop(slot);
        if res.is_ok() {
            self.shared.cv.notify_all();
        }
        res
    }

    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError> {
        let res = self.with(|i| i.submit_flush(inc));
        if res.is_ok() {
            self.shared.cv.notify_all();
        }
        res
    }

    fn attach_waiter(&self, job: JobId, _inc: Incarnation) -> Result<(), PlatformError> {
        self.with(|i| {
            let per_job = i.cfg.max_waiters_per_job;
            let total = i.cfg.max_waiters_total;
            let attached_total = i.attached_total;
            let j = match i.jobs.get_mut(&job.0) {
                Some(j) if j.kind == Kind::Load && !j.cancelling && !j.crashed => j,
                _ => {
                    return Err(PlatformError::TaskRejected(String::from(
                        "not an in-flight load",
                    )))
                }
            };
            if 1 + j.attached >= per_job || attached_total >= total {
                i.stats.queue_full_total += 1;
                return Err(PlatformError::QueueFull);
            }
            j.attached += 1;
            i.attached_total += 1;
            Ok(())
        })
    }

    fn poll(&self, max: usize) -> Vec<Completion> {
        self.with(|i| {
            let n = max.min(i.queue.len());
            i.queue.drain(..n).collect()
        })
    }

    fn cancel(&self, job: JobId) -> CancelOutcome {
        let out = self.with(|i| match i.jobs.get_mut(&job.0) {
            None => CancelOutcome::TooLate,
            Some(j) if j.crashed => CancelOutcome::TooLate,
            Some(j) if j.cancelling => CancelOutcome::Cancelled,
            Some(j) => {
                let cancellable = !matches!(
                    (j.kind, j.phase),
                    (Kind::Flush | Kind::Noop, _)
                        | (Kind::Store, Phase::Syncing | Phase::Publishing)
                );
                if cancellable {
                    j.cancelling = true;
                    CancelOutcome::Cancelled
                } else {
                    CancelOutcome::TooLate
                }
            }
        });
        if out == CancelOutcome::Cancelled {
            self.shared.cv.notify_all();
        }
        out
    }

    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError> {
        self.with(|i| {
            if i.state == StoreState::Recovering {
                return Err(i.queue_full());
            }
            if i.state == StoreState::Failed || i.closed {
                return Err(PlatformError::DeviceUnavailable);
            }
            if i.pins.len() as u32 >= i.cfg.max_restore_ops {
                return Err(i.queue_full());
            }
            for k in keys {
                if !i
                    .index
                    .get(&k.0 .0)
                    .is_some_and(|e| !e.is_evicting() && !e.is_foreign())
                {
                    return Err(PlatformError::ObjectNotFound(k.0));
                }
            }
            let id = i.next_pin;
            i.next_pin += 1;
            let list: Vec<[u8; 32]> = keys.iter().map(|k| k.0 .0).collect();
            for k in &list {
                let ok = i.index.busy_inc(k);
                debug_assert!(ok);
            }
            i.pins.insert(id, list);
            Ok(PinToken(id))
        })
    }

    fn unpin(&self, token: PinToken) {
        self.with(|i| {
            if let Some(list) = i.pins.remove(&token.0) {
                for k in list {
                    i.index.busy_dec(&k);
                }
            }
        });
    }

    fn delete(&self, keys: &[ChunkKey]) -> u64 {
        let marked = self.with(|i| {
            if matches!(i.state, StoreState::Recovering | StoreState::Failed) {
                return 0;
            }
            let mut marked = 0;
            for k in keys {
                let kb = k.0 .0;
                if i.index.is_busy(&kb) || i.inflight_store.contains_key(&kb) {
                    continue;
                }
                if i.index.mark_evicting(&kb) {
                    i.unlinks.push_back(Unlink {
                        key: kb,
                        evict: false,
                    });
                    marked += 1;
                }
            }
            marked
        });
        if marked > 0 {
            self.shared.cv.notify_all();
        }
        marked
    }

    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress {
        recover_step_on(&self.shared, slice_ms)
    }

    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport {
        let deadline = Instant::now() + Duration::from_millis(intake_deadline_ms);
        let (published_before, cancelled, worker_mode) = self.with(|i| {
            i.closed = true;
            let published_before = i.stats.published_total;
            let mut cancelled = 0;
            for j in i.jobs.values_mut() {
                if j.crashed {
                    continue;
                }
                let cancellable = !matches!(
                    (j.kind, j.phase),
                    (Kind::Flush | Kind::Noop, _)
                        | (Kind::Store, Phase::Syncing | Phase::Publishing)
                );
                if cancellable {
                    if !j.cancelling {
                        j.cancelling = true;
                    }
                    cancelled += 1;
                }
            }
            i.reap_cancels();
            (published_before, cancelled, i.cfg.worker)
        });
        self.shared.cv.notify_all();

        if worker_mode == WorkerMode::Manual {
            while Instant::now() < deadline {
                if !self.pump() {
                    break;
                }
            }
        } else {
            let mut g = self.lock();
            loop {
                g.reap_cancels();
                let busy = g.jobs.values().any(|j| !j.crashed) || !g.unlinks.is_empty();
                if !busy {
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let wait = deadline - now;
                let (g2, _) = self
                    .shared
                    .cv
                    .wait_timeout(g, wait)
                    .unwrap_or_else(|e| e.into_inner());
                g = g2;
            }
            let garbage = std::mem::take(&mut g.graveyard);
            drop(g);
            drop(garbage);
        }

        self.with(|i| {
            i.reap_cancels();
            let unfinished: Vec<UnfinishedJob> = i
                .jobs
                .values()
                .filter(|j| !j.crashed)
                .map(|j| UnfinishedJob {
                    job: JobId(j.id),
                    key: j.key,
                    bytes: j.bytes,
                    stage: j.stage,
                })
                .collect();
            i.stats.shutdown_unfinished_jobs = unfinished.len() as u64;
            i.stats.shutdown_unfinished_bytes = unfinished.iter().map(|u| u.bytes).sum();
            DrainReport {
                published: i.stats.published_total - published_before,
                cancelled,
                unfinished,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{chunk_chain, ComputeConfig, HiddenAct, KvDtype};
    use crate::pin::HolderGuard;

    const BS: usize = 8192;

    fn test_fields() -> NamespaceFields<'static> {
        let compute = ComputeConfig {
            num_heads: 32,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            vocab_size: 128_256,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            hidden_act: HiddenAct::Silu,
            arith_id: "ReferenceCpuBackend",
            arith_epoch: 1,
        };
        NamespaceFields {
            model_sha256: [0xAB; 32],
            model_digest_kind: "sha256-file",
            model_id: "llama-3.2-1b-instruct",
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            block_size: 4,
            rope_theta: 500000.0,
            rope_scaling: None,
            qk_norm: false,
            dtype: KvDtype::Bf16,
            layout: LayoutStrides {
                head_stride: 128,
                token_stride: 512,
                kv_plane_stride: 2048,
                layer_stride: 4096,
                block_stride: BS as u64,
                element_bytes: 2,
            },
            backend_format: BackendFormat::HostPagedPoolV1,
            chunk_tokens: 8,
            sharing_namespace: "default",
            compute_config_digest: compute.digest().unwrap(),
        }
    }

    fn make_src(blocks: usize, tail_valid: u32) -> (PinnedSource, Box<[u8]>) {
        let mut buf = vec![0x33u8; blocks * BS].into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        let guard = HolderGuard::detached();
        let src = unsafe { PinnedSource::new(ptr, BS, (0..blocks).collect(), tail_valid, guard) };
        (src, buf)
    }

    fn test_key() -> Arc<crate::seal::SealKey> {
        Arc::new(crate::seal::SealKey::generate().unwrap())
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "aien-kv-hostfs-unit-{tag}-{}-{nanos}",
            std::process::id()
        ));
        // The store refuses a root others can reach (ADR 0003 section 19).
        std::os::unix::fs::DirBuilderExt::mode(std::fs::DirBuilder::new().recursive(true), 0o700)
            .create(&p)
            .unwrap();
        p
    }

    #[test]
    fn host_fs_reservation_covers_the_physical_bytes_publication_charges() {
        // Codex re-review 2, partial 8: three KV heads give 6144-byte blocks,
        // so payload plus header (10240) is not a filesystem-block multiple.
        // The reservation taken at submit must not be smaller than what the
        // published file is charged (st_blocks).
        const BS3: usize = 6144;
        let root = temp_root("alloc-round");
        let mut f = test_fields();
        f.num_kv_heads = 3;
        f.layout = LayoutStrides {
            head_stride: 128,
            token_stride: 384,
            kv_plane_stride: 1536,
            layer_stride: 3072,
            block_stride: BS3 as u64,
            element_bytes: 2,
        };
        let ns = f.namespace_id().unwrap();
        let mut cfg = HostFsConfig::from_fields(root.clone(), &f).unwrap();
        cfg.seal_key = Some(test_key());
        cfg.worker = WorkerMode::Manual;
        let s = HostFsStore::open(cfg).unwrap();
        while !s.recover_step(50).done {}
        let toks: Vec<u32> = (0..4).collect();
        let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
        let h = f.chunk_header(&ns, key, None, 0, 4, 1).unwrap();
        let mut buf = vec![0x33u8; BS3].into_boxed_slice();
        let src = unsafe {
            PinnedSource::new(buf.as_mut_ptr(), BS3, vec![0], 4, HolderGuard::detached())
        };
        s.submit_store(h, src, Incarnation::seq(1)).unwrap();
        let reserved = s.capacity().reserved_bytes;
        s.run_until_idle();
        let comps = s.poll(16);
        assert!(comps
            .iter()
            .any(|c| c.terminal == Some(Terminal::Done(JobStage::Durable))));
        let used = s.capacity().used_bytes;
        assert!(
            reserved >= used,
            "reserved {reserved} < charged {used} at publication"
        );
        s.audit().unwrap();
        drop(buf);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn host_fs_max_objects_includes_starting_job_reservation() {
        let root = temp_root("max-obj");
        let f = test_fields();
        let ns = f.namespace_id().unwrap();
        let mut cfg = HostFsConfig::from_fields(root.clone(), &f).unwrap();
        cfg.seal_key = Some(test_key());
        cfg.max_objects = 2;
        cfg.worker = WorkerMode::Manual;
        let s = HostFsStore::open(cfg).unwrap();
        while !s.recover_step(50).done {}

        for n in 1..=3u32 {
            let toks: Vec<u32> = (0..8).map(|i| n * 100 + i).collect();
            let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
            let h = f.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
            let (src, _buf) = make_src(2, 4);
            s.submit_store(h, src, Incarnation::seq(n as u64)).unwrap();
            s.run_until_idle();
            let _ = s.poll(16);
        }
        assert_eq!(s.capacity().objects, 2);
        s.audit().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn host_fs_victims_are_unlinked_after_publish_and_a_failed_unlink_keeps_one() {
        let root = temp_root("abort-evict");
        let f = test_fields();
        let ns = f.namespace_id().unwrap();
        let small_phys = HEADER_REGION_LEN as u64 + BS as u64;
        let faults = ScriptedFaults::new();
        let mut cfg = HostFsConfig::from_fields(root.clone(), &f).unwrap();
        cfg.seal_key = Some(test_key());
        cfg.quota_bytes = 2 * small_phys;
        cfg.worker = WorkerMode::Manual;
        cfg.faults = Some(faults.clone());
        let s = HostFsStore::open(cfg).unwrap();
        while !s.recover_step(50).done {}

        // Store two 1-block objects.
        let mut small_keys = Vec::new();
        for n in 1..=2u32 {
            let toks: Vec<u32> = (0..4).map(|i| n * 100 + i).collect();
            let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
            let h = f.chunk_header(&ns, key, None, 0, 4, 1).unwrap();
            let (src, _buf) = make_src(1, 4);
            s.submit_store(h, src, Incarnation::seq(n as u64)).unwrap();
            s.run_until_idle();
            let _ = s.poll(16);
            small_keys.push(key);
        }

        // A 2-block object needs both small ones evicted. The victims are
        // unlinked only after it is durable; the first unlink fails, so that
        // victim returns to service and stays counted, and the store is Done.
        faults.arm(Boundary::Unlink, FaultAction::Fail(EIO));
        let toks: Vec<u32> = (0..8).map(|i| 300 + i).collect();
        let key3 = chunk_chain(&ns, 8, &toks).unwrap()[0];
        let h3 = f.chunk_header(&ns, key3, None, 0, 8, 1).unwrap();
        let (src3, _buf3) = make_src(2, 4);
        let j3 = s.submit_store(h3, src3, Incarnation::seq(3)).unwrap();
        s.run_until_idle();
        let comps = s.poll(16);
        assert!(comps
            .iter()
            .any(|c| c.job == j3 && c.terminal == Some(Terminal::Done(JobStage::Durable))));
        assert_eq!(s.lookup(&small_keys), vec![Some(BS as u64), None]);
        assert_eq!(s.lookup(&[key3]), vec![Some(2 * BS as u64)]);
        let cap = s.capacity();
        assert_eq!(cap.evicting_bytes, 0);
        assert_eq!(
            cap.used_bytes,
            small_phys + HEADER_REGION_LEN as u64 + 2 * BS as u64
        );
        s.audit().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn host_fs_never_dedups_into_cancelling_job_and_checks_segments_and_strides() {
        let root = temp_root("dedup-cancel");
        let f = test_fields();
        let ns = f.namespace_id().unwrap();
        let mut cfg = HostFsConfig::from_fields(root.clone(), &f).unwrap();
        cfg.seal_key = Some(test_key());
        cfg.worker = WorkerMode::Manual;
        let s = HostFsStore::open(cfg.clone()).unwrap();
        while !s.recover_step(50).done {}

        let toks: Vec<u32> = (0..8).collect();
        let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
        let h = f.chunk_header(&ns, key, None, 0, 8, 1).unwrap();
        let (src1, _b1) = make_src(2, 4);
        let j1 = s.submit_store(h, src1, Incarnation::seq(1)).unwrap();
        assert_eq!(s.cancel(j1), CancelOutcome::Cancelled);
        let (src2, _b2) = make_src(2, 4);
        let j2 = s.submit_store(h, src2, Incarnation::seq(2)).unwrap();
        assert_ne!(j1, j2);
        s.run_until_idle();
        assert_eq!(s.lookup(&[key]), vec![Some(2 * BS as u64)]);

        // Refuse > MAX_SEGMENTS blocks at submit time.
        let mut h_many = h;
        h_many.payload_len = 121 * BS as u64;
        h_many.token_count = 121 * 4;
        let (src_many, _bm) = make_src(121, 4);
        assert_eq!(
            s.submit_store(h_many, src_many, Incarnation::seq(3)),
            Err(PlatformError::InvalidLayout)
        );

        // Refuse zero block_stride and zero element_bytes in config validation.
        let mut bad1 = cfg.clone();
        bad1.layout.block_stride = 0;
        assert!(matches!(
            HostFsStore::open(bad1),
            Err(PlatformError::InvalidLayout)
        ));
        let mut bad2 = cfg;
        bad2.layout.element_bytes = 0;
        assert!(matches!(
            HostFsStore::open(bad2),
            Err(PlatformError::InvalidLayout)
        ));

        let _ = std::fs::remove_dir_all(&root);
    }
}
