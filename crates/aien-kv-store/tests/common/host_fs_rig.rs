//! Shared fixtures for the host filesystem adapter tests (cut C4): a geometry
//! whose objects are whole multiples of 4096 bytes (so physical bytes equal
//! file size on every filesystem), real temporary directories, pool memory
//! with pinned views, and helpers that plant and inspect object files.
//!
//! Included with `#[path = "common/host_fs_rig.rs"]` by the C4 test files, so
//! it does not touch `tests/common/mod.rs`.
#![allow(dead_code)]

use aien_kv_store::host_fs::{
    Boundary, FaultAction, FaultHook, HostFsConfig, HostFsStore, ScriptedFaults, WorkerMode,
};
use aien_kv_store::seal::SealKey;
use aien_kv_store::{
    chunk_chain, payload_root, ChunkHeader, ChunkKey, Completion, HolderGuard, HolderHandle,
    Incarnation, JobId, KvStore, LayoutStrides, NamespaceFields, NamespaceId, PinnedDest,
    PinnedSource, RecordHeader, Terminal,
};
use aien_platform::PlatformError;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard};
use std::time::{Duration, Instant, SystemTime};

/// Bytes of one pool block: 2 layers x 2 planes x 4 tokens x 512-byte rows.
pub const BS: usize = 8192;
pub const BLOCK_TOKENS: u32 = 4;
pub const CHUNK_TOKENS: u32 = 8;
pub const LAYERS: usize = 2;
pub const LAYER_STRIDE: usize = 4096;
pub const PLANE_STRIDE: usize = 2048;
pub const ROW: usize = 512;
/// Payload of a full chunk (2 blocks) and its file size / quota cost.
pub const FULL_PAYLOAD: u64 = 2 * BS as u64;
pub const OBJ: u64 = FULL_PAYLOAD + 4096;
/// Header region plus one block (a chunk of 1 to 4 tokens).
pub const OBJ1: u64 = BS as u64 + 4096;

pub fn geometry() -> NamespaceFields<'static> {
    let mut f = crate::common::sample();
    f.num_layers = LAYERS as u32;
    f.num_kv_heads = 4;
    f.head_dim = 64;
    f.block_size = BLOCK_TOKENS;
    f.chunk_tokens = CHUNK_TOKENS;
    f.layout = LayoutStrides {
        head_stride: 128,
        token_stride: ROW as u64,
        kv_plane_stride: PLANE_STRIDE as u64,
        layer_stride: LAYER_STRIDE as u64,
        block_stride: BS as u64,
        element_bytes: 2,
    };
    f
}

// ---- child processes -----------------------------------------------------

/// Tests run on parallel threads of one process. While a test spawns a child
/// process, the child holds a copy of every descriptor open in this process
/// until its exec closes them (they are close-on-exec), and `flock` belongs to
/// the open file, not the descriptor: a store another test drops in that
/// window stays locked, and that test's reopen is refused (`LOCK` is held).
/// A spawning test takes this for writing and every other test that opens a
/// store for reading, so no store opens or drops while a child is spawned.
pub static SPAWN: RwLock<()> = RwLock::new(());

pub fn no_spawn() -> RwLockReadGuard<'static, ()> {
    SPAWN.read().unwrap_or_else(|e| e.into_inner())
}

// ---- temporary directories -----------------------------------------------

pub struct TempDir(pub PathBuf);

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        let n = DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "aien-kv-c4-{tag}-{}-{n}-{nanos}",
            std::process::id()
        ));
        // The store refuses a root others can reach (ADR 0003 section 19).
        std::os::unix::fs::DirBuilderExt::mode(std::fs::DirBuilder::new().recursive(true), 0o700)
            .create(&p)
            .unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- pool memory ---------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Mem {
    ptr: *mut u8,
    pub blocks: usize,
}

unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

impl Mem {
    pub fn new(blocks: usize, fill: u8) -> Mem {
        let b: Box<[u8]> = vec![fill; blocks * BS].into_boxed_slice();
        Mem {
            ptr: Box::leak(b).as_mut_ptr(),
            blocks,
        }
    }

    pub fn fill_with(&self, f: impl Fn(usize) -> u8) {
        for i in 0..self.blocks * BS {
            unsafe { *self.ptr.add(i) = f(i) };
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        (0..self.blocks * BS)
            .map(|i| unsafe { *self.ptr.add(i) })
            .collect()
    }

    pub fn source(&self, n: usize, tail_valid_tokens: u32) -> (PinnedSource, HolderHandle) {
        let guard = HolderGuard::detached();
        let h = guard.handle();
        let s = unsafe {
            PinnedSource::new(
                self.ptr as *const u8,
                BS,
                (0..n).collect(),
                tail_valid_tokens,
                guard,
            )
        };
        (s, h)
    }

    pub fn dest(&self, n: usize) -> (PinnedDest, HolderHandle) {
        let guard = HolderGuard::detached();
        let h = guard.handle();
        let d = unsafe { PinnedDest::new(self.ptr, BS, (0..n).collect(), guard) };
        (d, h)
    }
}

/// Deterministic bytes, never zero (so a zeroed row proves the writer zeroed it).
pub fn pattern(n: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(31).wrapping_add(n.wrapping_mul(17)) % 251 + 1) as u8)
        .collect()
}

/// Zero the token rows `valid..BLOCK_TOKENS` of every layer and plane of one block.
pub fn zero_tail_rows(block: &mut [u8], valid: u32) {
    for l in 0..LAYERS {
        for p in 0..2 {
            for t in valid as usize..BLOCK_TOKENS as usize {
                let off = l * LAYER_STRIDE + p * PLANE_STRIDE + t * ROW;
                block[off..off + ROW].fill(0);
            }
        }
    }
}

pub fn blocks_for(tokens: u32) -> usize {
    tokens.div_ceil(BLOCK_TOKENS) as usize
}

pub fn tail_valid(tokens: u32) -> u32 {
    tokens - (blocks_for(tokens) as u32 - 1) * BLOCK_TOKENS
}

/// What the object's payload must hold for pool bytes `pattern(n, ..)` and `tokens` positions.
pub fn expected_payload(n: u32, tokens: u32) -> Vec<u8> {
    let blocks = blocks_for(tokens);
    let mut data = pattern(n, blocks * BS);
    let tail = tail_valid(tokens);
    if tail < BLOCK_TOKENS {
        let last = (blocks - 1) * BS;
        zero_tail_rows(&mut data[last..last + BS], tail);
    }
    data
}

// ---- hooks ---------------------------------------------------------------

type Callback = dyn Fn(Boundary, &ChunkKey) -> Option<FaultAction> + Send + Sync;

/// Runs a closure at every boundary, then falls through to the scripted faults.
pub struct CallbackHook {
    pub scripted: Arc<ScriptedFaults>,
    pub f: Box<Callback>,
}

impl FaultHook for CallbackHook {
    fn at(&self, boundary: Boundary, key: &ChunkKey) -> Option<FaultAction> {
        if let Some(a) = (self.f)(boundary, key) {
            return Some(a);
        }
        self.scripted.at(boundary, key)
    }
}

/// A store handle a hook closure can reach once the store exists.
pub type StoreCell = Arc<OnceLock<Arc<HostFsStore>>>;

// ---- the fixture -----------------------------------------------------------

pub struct Fx {
    pub dir: TempDir,
    pub fields: NamespaceFields<'static>,
    pub ns: NamespaceId,
    pub faults: Arc<ScriptedFaults>,
}

pub struct Put {
    pub key: ChunkKey,
    pub header: ChunkHeader,
    pub job: Result<JobId, PlatformError>,
    pub completions: Vec<Completion>,
    /// What the object's payload must contain.
    pub expected: Vec<u8>,
}

impl Fx {
    pub fn new() -> Fx {
        let fields = geometry();
        let ns = fields.namespace_id().unwrap();
        Fx {
            dir: TempDir::new("host"),
            fields,
            ns,
            faults: ScriptedFaults::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.dir.0
    }

    pub fn config(&self) -> HostFsConfig {
        let mut c = HostFsConfig::from_fields(self.dir.0.clone(), &self.fields).unwrap();
        c.quota_bytes = 64 << 20;
        c.max_objects = 1024;
        c.worker = WorkerMode::Manual;
        c.faults = Some(self.faults.clone());
        c.seal_key = Some(test_key());
        c
    }

    /// Open without recovering: the store is `Recovering`.
    pub fn open_raw(&self, tweak: impl FnOnce(&mut HostFsConfig)) -> HostFsStore {
        let mut c = self.config();
        tweak(&mut c);
        HostFsStore::open(c).expect("open")
    }

    /// Open and run recovery to the end.
    pub fn open_with(&self, tweak: impl FnOnce(&mut HostFsConfig)) -> HostFsStore {
        let s = self.open_raw(tweak);
        recover_all(&s);
        s
    }

    pub fn open(&self) -> HostFsStore {
        self.open_with(|_| {})
    }

    /// Key and header of the first chunk of token stream `n` with `tokens` tokens.
    pub fn chunk(&self, n: u32, tokens: u32) -> (ChunkKey, ChunkHeader) {
        let toks: Vec<u32> = (0..tokens).map(|i| n * 1000 + i).collect();
        let key = chunk_chain(&self.ns, CHUNK_TOKENS, &toks).unwrap()[0];
        let h = self
            .fields
            .chunk_header(&self.ns, key, None, 0, tokens, 1)
            .unwrap();
        (key, h)
    }

    /// Second chunk of a 16-token stream `n`: parent is the first chunk, position 8.
    pub fn second_chunk(&self, n: u32) -> (ChunkKey, ChunkHeader) {
        let toks: Vec<u32> = (0..16).map(|i| n * 1000 + i).collect();
        let keys = chunk_chain(&self.ns, CHUNK_TOKENS, &toks).unwrap();
        let h = self
            .fields
            .chunk_header(&self.ns, keys[1], Some(&keys[0]), 8, 8, 1)
            .unwrap();
        (keys[1], h)
    }

    /// Submit a store of object `n` (`tokens` positions) from pool memory filled
    /// with `pattern(n, ..)`; do not run the worker.
    pub fn submit(
        &self,
        s: &HostFsStore,
        n: u32,
        tokens: u32,
    ) -> (Result<JobId, PlatformError>, HolderHandle) {
        let (_, h) = self.chunk(n, tokens);
        let blocks = blocks_for(tokens);
        let mem = Mem::new(blocks, 0);
        let data = pattern(n, blocks * BS);
        mem.fill_with(|i| data[i]);
        let (src, handle) = mem.source(blocks, tail_valid(tokens));
        (s.submit_store(h, src, Incarnation::seq(n as u64)), handle)
    }

    /// Store object `n` and run the worker to idle.
    pub fn put(&self, s: &HostFsStore, n: u32, tokens: u32) -> Put {
        let (key, header) = self.chunk(n, tokens);
        let (job, _) = self.submit(s, n, tokens);
        s.run_until_idle();
        // Distinct mtimes for the LRU order.
        std::thread::sleep(Duration::from_millis(2));
        Put {
            key,
            header,
            job,
            completions: drain(s),
            expected: expected_payload(n, tokens),
        }
    }

    /// Store a full 8-token chunk.
    pub fn put_full(&self, s: &HostFsStore, n: u32) -> Put {
        self.put(s, n, 8)
    }

    /// Submit a load into poisoned memory (0xEE).
    pub fn submit_load(
        &self,
        s: &HostFsStore,
        key: ChunkKey,
        expect: &ChunkHeader,
    ) -> (Result<JobId, PlatformError>, Mem) {
        let blocks = blocks_for(expect.token_count);
        let mem = Mem::new(blocks, 0xEE);
        let (dst, _) = mem.dest(blocks);
        (s.submit_load(key, expect, dst, Incarnation::seq(9000)), mem)
    }

    /// Load and run to idle; returns the terminal completion and the memory.
    pub fn load(
        &self,
        s: &HostFsStore,
        key: ChunkKey,
        expect: &ChunkHeader,
    ) -> (Result<Terminal, PlatformError>, Mem) {
        let (job, mem) = self.submit_load(s, key, expect);
        let job = match job {
            Ok(j) => j,
            Err(e) => return (Err(e), mem),
        };
        s.run_until_idle();
        let t = terminal_of(&drain(s), job);
        (Ok(t), mem)
    }

    pub fn object_file(&self, key: &ChunkKey) -> PathBuf {
        object_path(self.root(), key)
    }

    /// Plant a valid object file by hand (the store must have been opened once
    /// so the shard directories exist). `mtime_secs` sets the file mtime.
    pub fn plant(&self, key: &ChunkKey, header: &ChunkHeader, payload: &[u8], seg: usize) {
        write_object_file(&self.object_file(key), header, payload, seg);
    }

    /// Create the directory layout (and `STORE_ID`) by opening and dropping a store.
    pub fn prime(&self) {
        let s = self.open_raw(|_| {});
        drop(s);
    }
}

pub fn hex(b: &[u8; 32]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for x in b {
        s.push(D[(x >> 4) as usize] as char);
        s.push(D[(x & 15) as usize] as char);
    }
    s
}

pub fn object_path(root: &Path, key: &ChunkKey) -> PathBuf {
    let h = hex(key.as_bytes());
    root.join("objects").join(&h[..2]).join(format!("{h}.kvc"))
}

pub fn tmp_files(root: &Path) -> Vec<PathBuf> {
    match std::fs::read_dir(root.join("tmp")) {
        Ok(rd) => rd.map(|e| e.unwrap().path()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Every `*.kvc` file under `objects/`.
pub fn object_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root.join("objects")) else {
        return out;
    };
    for shard in rd {
        let shard = shard.unwrap().path();
        for f in std::fs::read_dir(&shard).unwrap() {
            let p = f.unwrap().path();
            if p.extension().is_some_and(|e| e == "kvc") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `max(st_size, st_blocks * 512)` computed independently of the adapter.
pub fn phys_of(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(path).unwrap();
    m.len().max(m.blocks() * 512)
}

pub fn sha(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// The store key every rig store and planted object uses: generated once per
/// test process (ADR 0003 section 20), never a vault secret.
pub fn test_key() -> Arc<SealKey> {
    static KEY: OnceLock<Arc<SealKey>> = OnceLock::new();
    KEY.get_or_init(|| Arc::new(SealKey::generate().expect("test key")))
        .clone()
}

/// `region || payload` unsealed: the old (format 1) plaintext object bytes.
pub fn plain_object_bytes(header: &ChunkHeader, payload: &[u8], seg: usize) -> Vec<u8> {
    let digests: Vec<[u8; 32]> = payload.chunks(seg).map(sha).collect();
    let rec = RecordHeader {
        chunk: *header,
        payload_sha256: payload_root(&digests),
    };
    let mut bytes = rec.encode_region(&digests).unwrap().to_vec();
    bytes.extend_from_slice(payload);
    bytes
}

/// Write a sealed object the way the adapter does (`region || ciphertext`,
/// sealed with `test_key`); segment digests over `seg`-byte pieces.
pub fn write_object_file(path: &Path, header: &ChunkHeader, payload: &[u8], seg: usize) {
    let mut bytes = plain_object_bytes(header, payload, seg);
    test_key().seal_file(&mut bytes).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// Write only the sealed-format marker (magic and version) into a region, for
/// files recovery classifies without reading the payload.
pub fn mark_sealed(region: &mut [u8]) {
    let at = aien_kv_store::record::SEAL_SLOT_OFFSET;
    region[at..at + 3].copy_from_slice(&aien_kv_store::seal::SEAL_MAGIC);
    region[at + 3] = aien_kv_store::seal::SEALED_FORMAT_VERSION;
}

/// Decrypt an object file sealed with `test_key`: `region || payload`.
pub fn read_object_plain(path: &Path) -> Vec<u8> {
    let mut bytes = std::fs::read(path).unwrap();
    let (region, payload) = bytes.split_at_mut(4096);
    test_key()
        .open_payload(region, payload)
        .expect("object opens");
    bytes
}

pub fn set_mtime(path: &Path, secs_after_epoch: u64) {
    let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs_after_epoch))
        .unwrap();
}

pub fn drain(s: &HostFsStore) -> Vec<Completion> {
    let mut out = Vec::new();
    loop {
        let v = s.poll(7);
        if v.is_empty() {
            return out;
        }
        out.extend(v);
    }
}

pub fn terminal_of(v: &[Completion], job: JobId) -> Terminal {
    let t: Vec<_> = v
        .iter()
        .filter(|c| c.job == job && c.terminal.is_some())
        .collect();
    assert_eq!(t.len(), 1, "exactly one terminal for {job:?} in {v:?}");
    t[0].terminal.clone().unwrap()
}

pub fn completion_of(v: &[Completion], job: JobId) -> Completion {
    v.iter()
        .find(|c| c.job == job && c.terminal.is_some())
        .cloned()
        .expect("terminal completion")
}

/// Run recovery to the end with a generous slice; returns the final progress.
pub fn recover_all(s: &HostFsStore) -> aien_kv_store::RecoveryProgress {
    for _ in 0..1_000_000 {
        let p = s.recover_step(50);
        if p.done || p.failed {
            return p;
        }
    }
    panic!("recovery did not finish");
}

pub fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !cond() {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Shared log for hook closures.
pub fn log() -> Arc<Mutex<Vec<String>>> {
    Arc::new(Mutex::new(Vec::new()))
}
