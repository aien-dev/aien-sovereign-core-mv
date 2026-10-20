//! Resource pools and the GPU locks (ADR 0028 Decision 6).
//!
//! Three counting semaphores inside one runner process: `host` (default size
//! is the CPU count), `qemu` (default `min(4, host / 2)`, a guess that is
//! overridable with `--jobs`), and `gb10` (fixed at 1). A job holds exactly
//! one pool slot. Semaphores are plain `Mutex` plus `Condvar`; a `Permit` is
//! an RAII guard that returns its slot when dropped.
//!
//! Deadlock rule: whenever more than one resource is taken, they are taken in
//! one fixed order. Pools are ordered by `Pool`'s derived `Ord` (gb10, then
//! qemu, then host). The full order for one gate is: (1) the in-process pool
//! slot, (2) the target-dir lock (`lock_target_dir`, every gate that runs,
//! on its `CARGO_TARGET_DIR` or else the worktree's `target/`), (3)
//! `/tmp/aien-gb10.lock`, (4) aien-proof's `slots/gpu.lock` when the
//! aien-proof board directory exists. A holder of (2) already has its slot
//! and never waits for one, so a gate blocked on (2) with a slot in hand
//! cannot close a cycle; the worker's target priming takes (2) alone and
//! never (3) or (4). Known limit: a gb10 gate is never killed on timeout,
//! so a hung one holds (2) and (3) until it exits on its own; every other
//! gate of that target dir waits behind it (the receipt says
//! `timeout_exceeded`).
//!
//! GPU job protocol (CR-028, CR-029, CR-033, CR-052, CR-053): refuse at once
//! (never wait, never steal) if the quiet flag file exists or an `est_load`
//! process is running, then take the `flock(2)` locks. A held lock is waited
//! for up to a bound (`GpuWait`, default `DEFAULT_GPU_WAIT`), then refused:
//! the live model server takes `/tmp/aien-gb10.lock` around every request
//! (sc `aien-serve-http`, `GpuLock::acquire_queued_at`), so a single try
//! loses nearly every time. The wait is a blocking `flock` in the kernel's
//! queue on a helper thread, the same queue the server's requests and
//! `flock(1)` jobs wait in, so neither side can starve the other the way a
//! poller with `LOCK_NB` is starved by queued waiters. `--gpu-wait 0` refuses
//! at once; `--wait-gpu` waits without a bound. The lock files are created if
//! absent and never deleted. Dropping a `GpuGuard` closes the descriptors,
//! and the kernel releases the locks (a dead holder frees them too).
//!
//! Fork window: a `flock` lives as long as any copy of the open file
//! description does, and a child that another thread forks while a lock
//! descriptor is open holds a copy until it execs (the descriptor is
//! close-on-exec, so it never survives the exec). The guard therefore closes
//! its descriptors under `process::spawn_gate`, the mutex every child spawn
//! in this crate also takes, so no child is in that window at the moment the
//! lock is released and the next `acquire_gpu` is not refused by a lock
//! nobody holds.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub const DEFAULT_GPU_LOCK: &str = "/tmp/aien-gb10.lock";

/// How long a gb10 gate waits for a held GPU lock before NOT_RUN `gpu_lock`.
/// The server's own upload waits this long for the same lock (sc
/// `qwen38_engine::LOAD_LOCK_WAIT`, "other jobs' chip runs take minutes");
/// one server request held it 108 s median and 222 s at most over 323
/// requests on 2026-10-19 (serve.log), so the bound covers many turns.
pub const DEFAULT_GPU_WAIT: Duration = Duration::from_secs(30 * 60);

/// MemAvailable (MB) a gb10 gate needs before it starts, checked with the GPU
/// locks held. Measured on the Spark, 2026-10-20 (omega rx_r14_recovery_silicon
/// under the GB10 lock, MemAvailable sampled every 0.5 s): baseline drift
/// 0.8 GB, drop to 1.6 GB below the pre-run baseline during the run. The
/// omega silicon gates that failed NO_MEMORY at channel create had 3-4 GB
/// free, so the floor sits about twice that failure point. Replace it with
/// per-gate peaks from the `volatile.gpu_wait` fields as they accrue.
pub const DEFAULT_GPU_MEM_MB: u64 = 6 * 1024;

/// How long a gb10 gate waits, holding the GPU locks, for `DEFAULT_GPU_MEM_MB`
/// to show as free. Short on purpose: a 27B server holds its weights for hours,
/// and a memory-short gate is NOT_RUN `gpu_mem` (retryable), not FAIL.
pub const DEFAULT_GPU_MEM_WAIT: Duration = Duration::from_secs(15 * 60);

/// Poll interval of the memory wait.
const GPU_MEM_POLL: Duration = Duration::from_secs(5);

/// The memory reading (procfs).
pub const DEFAULT_MEMINFO: &str = "/proc/meminfo";

/// What a gb10 gate does when another holder has a GPU lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWait {
    /// Refuse at once (`--gpu-wait 0`, and the scratch configs of tests).
    Refuse,
    /// Wait in the kernel's queue up to this long, then refuse.
    Bounded(Duration),
    /// `--wait-gpu`: wait as long as it takes.
    Forever,
}

impl GpuWait {
    /// The receipt's `volatile.gpu_wait.mode`.
    pub fn mode(self) -> &'static str {
        match self {
            GpuWait::Refuse => "refuse",
            GpuWait::Bounded(_) => "bounded",
            GpuWait::Forever => "forever",
        }
    }

    /// The bound in ms (`Refuse` is 0, `Forever` has none).
    pub fn bound_ms(self) -> Option<u64> {
        match self {
            GpuWait::Refuse => Some(0),
            GpuWait::Bounded(d) => Some(d.as_millis() as u64),
            GpuWait::Forever => None,
        }
    }
}

/// Set to `1` in a gb10 gate's environment while the runner holds the GPU
/// locks for it (ADR 0033: gb10 is serialized by the trusted side). A gate
/// script that also takes `/tmp/aien-gb10.lock` by hand must skip that when
/// it sees this: under `--sandbox user` it cannot open the worker's lock file
/// (exit 66, `flock: Permission denied`), and as the same user it would
/// block forever behind the runner's own exclusive lock.
pub const GPU_LOCK_HELD_MARKER: &str = "AIEN_GB10_LOCK_HELD";

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A counting semaphore.
#[derive(Debug)]
pub struct Semaphore {
    size: usize,
    free: Mutex<usize>,
    cv: Condvar,
}

impl Semaphore {
    /// A semaphore with `size` slots (at least 1).
    pub fn new(size: usize) -> Semaphore {
        let size = size.max(1);
        Semaphore {
            size,
            free: Mutex::new(size),
            cv: Condvar::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn available(&self) -> usize {
        *lock(&self.free)
    }

    /// Block until a slot is free.
    pub fn acquire(&self) -> Permit<'_> {
        let mut free = lock(&self.free);
        while *free == 0 {
            free = self.cv.wait(free).unwrap_or_else(|e| e.into_inner());
        }
        *free -= 1;
        Permit { sem: self }
    }

    /// Take a slot if one is free right now.
    pub fn try_acquire(&self) -> Option<Permit<'_>> {
        let mut free = lock(&self.free);
        if *free == 0 {
            return None;
        }
        *free -= 1;
        Some(Permit { sem: self })
    }
}

/// One held slot. Dropping it frees the slot and wakes one waiter.
#[derive(Debug)]
pub struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut free = lock(&self.sem.free);
        *free += 1;
        self.sem.cv.notify_one();
    }
}

/// The pool a job is scheduled in. The derived `Ord` is the fixed acquire
/// order: scarcest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pool {
    Gb10,
    Qemu,
    Host,
}

impl Pool {
    pub fn name(self) -> &'static str {
        match self {
            Pool::Gb10 => "gb10",
            Pool::Qemu => "qemu",
            Pool::Host => "host",
        }
    }

    pub fn from_name(s: &str) -> Option<Pool> {
        match s {
            "gb10" => Some(Pool::Gb10),
            "qemu" => Some(Pool::Qemu),
            "host" => Some(Pool::Host),
            _ => None,
        }
    }
}

/// Sizes of the two resizable pools. The gb10 pool is fixed at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSizes {
    pub host: usize,
    pub qemu: usize,
}

/// `--jobs host=N,qemu=M`: only the pools that were named.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JobsSpec {
    pub host: Option<usize>,
    pub qemu: Option<usize>,
}

impl PoolSizes {
    /// ADR defaults for a machine with `cpus` CPUs: host = cpus, qemu =
    /// min(4, host / 2), never below 1.
    pub fn defaults_for(cpus: usize) -> PoolSizes {
        let host = cpus.max(1);
        PoolSizes {
            host,
            qemu: (host / 2).clamp(1, 4),
        }
    }

    pub fn detect() -> PoolSizes {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        PoolSizes::defaults_for(cpus)
    }

    pub fn with(self, jobs: JobsSpec) -> PoolSizes {
        PoolSizes {
            host: jobs.host.unwrap_or(self.host),
            qemu: jobs.qemu.unwrap_or(self.qemu),
        }
    }
}

/// Parse `host=N,qemu=M` (either part may be left out; N, M at least 1).
pub fn parse_jobs(spec: &str) -> Result<JobsSpec, String> {
    let mut out = JobsSpec::default();
    for part in spec.split(',') {
        let (name, value) = part
            .split_once('=')
            .ok_or_else(|| format!("--jobs entry {part:?} must look like host=N or qemu=M"))?;
        let n: usize = value
            .parse()
            .map_err(|_| format!("--jobs {name} needs a whole number, got {value:?}"))?;
        if n == 0 {
            return Err(format!("--jobs {name} must be at least 1"));
        }
        match name {
            "host" => out.host = Some(n),
            "qemu" => out.qemu = Some(n),
            "gb10" => return Err("the gb10 pool is fixed at 1".to_string()),
            other => return Err(format!("unknown pool {other:?} in --jobs")),
        }
    }
    Ok(out)
}

/// The three semaphores of one runner process.
#[derive(Debug)]
pub struct Pools {
    host: Semaphore,
    qemu: Semaphore,
    gb10: Semaphore,
}

impl Pools {
    pub fn new(sizes: PoolSizes) -> Pools {
        Pools {
            host: Semaphore::new(sizes.host),
            qemu: Semaphore::new(sizes.qemu),
            gb10: Semaphore::new(1),
        }
    }

    pub fn semaphore(&self, pool: Pool) -> &Semaphore {
        match pool {
            Pool::Host => &self.host,
            Pool::Qemu => &self.qemu,
            Pool::Gb10 => &self.gb10,
        }
    }

    pub fn acquire(&self, pool: Pool) -> Permit<'_> {
        self.semaphore(pool).acquire()
    }

    pub fn try_acquire(&self, pool: Pool) -> Option<Permit<'_>> {
        self.semaphore(pool).try_acquire()
    }

    /// Take several pools in the fixed order (duplicates collapse), so two
    /// callers asking for the same set can never deadlock each other.
    pub fn acquire_many(&self, pools: &[Pool]) -> Vec<Permit<'_>> {
        let mut order: Vec<Pool> = pools.to_vec();
        order.sort();
        order.dedup();
        order.into_iter().map(|p| self.acquire(p)).collect()
    }
}

/// Which special hardware this machine has. A heuristic, recorded in the
/// receipt's `machine` object (UNVERIFIED: `/dev/nvidiactl` for the GB10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hardware {
    pub gb10_present: bool,
    pub qemu_available: bool,
}

fn which(name: &str) -> bool {
    match std::env::var_os("PATH") {
        Some(p) => std::env::split_paths(&p).any(|d| d.join(name).is_file()),
        None => false,
    }
}

impl Hardware {
    pub fn detect() -> Hardware {
        Hardware {
            gb10_present: Path::new("/dev/nvidiactl").exists(),
            qemu_available: which("qemu-system-aarch64") || which("qemu-system-x86_64"),
        }
    }
}

/// Where the GPU lock files and the refusal checks live. Passed in explicitly
/// (never read from the environment by the lock code), so a test can point
/// every path at a scratch directory and never touch the real locks.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// `/tmp/aien-gb10.lock`, taken first.
    pub lock_path: PathBuf,
    /// aien-proof's `slots/gpu.lock`, taken second, only when its board
    /// directory exists (ADR Decision 6, stopgap until the locks are unified).
    pub proof_lock_path: Option<PathBuf>,
    /// `~/workspace/.spark-quiet`; None when HOME is unknown.
    pub quiet_flag: Option<PathBuf>,
    /// Where to look for an `est_load` process (`/proc`).
    pub proc_dir: PathBuf,
    /// How long to wait for a held lock (`--gpu-wait`, `--wait-gpu`).
    pub wait: GpuWait,
    /// The memory reading (`/proc/meminfo`, MemAvailable); a file so a test
    /// can fake it.
    pub meminfo: PathBuf,
    /// MB of MemAvailable a gate needs once it holds the locks; 0 = not checked.
    pub mem_need_mb: u64,
    /// How long to wait, holding the locks, for `mem_need_mb` to show as free.
    pub mem_wait: Duration,
}

impl GpuConfig {
    /// The real machine: default lock, HOME-relative quiet flag, aien-proof
    /// board from `AIEN_PROOF_DIR` or `~/.local/state/aien-proof`.
    pub fn detect() -> GpuConfig {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let board = match std::env::var_os("AIEN_PROOF_DIR") {
            Some(d) => Some(PathBuf::from(d)),
            None => home.as_ref().map(|h| h.join(".local/state/aien-proof")),
        };
        GpuConfig {
            lock_path: PathBuf::from(DEFAULT_GPU_LOCK),
            proof_lock_path: board
                .filter(|b| b.is_dir())
                .map(|b| b.join("slots/gpu.lock")),
            quiet_flag: home.map(|h| h.join("workspace/.spark-quiet")),
            proc_dir: PathBuf::from("/proc"),
            wait: GpuWait::Bounded(DEFAULT_GPU_WAIT),
            meminfo: PathBuf::from(DEFAULT_MEMINFO),
            mem_need_mb: DEFAULT_GPU_MEM_MB,
            mem_wait: DEFAULT_GPU_MEM_WAIT,
        }
    }

    /// Every path inside `dir` (a scratch directory): own lock, own quiet
    /// flag, no aien-proof lock, no `est_load`; a held lock is refused at once.
    pub fn rooted_at(dir: &Path) -> GpuConfig {
        GpuConfig {
            lock_path: dir.join("gb10.lock"),
            proof_lock_path: None,
            quiet_flag: Some(dir.join("quiet-flag")),
            proc_dir: dir.join("proc"),
            wait: GpuWait::Refuse,
            meminfo: dir.join("meminfo"),
            // Not checked unless a test asks: the tests of the lock protocol
            // do not depend on the machine's memory.
            mem_need_mb: 0,
            mem_wait: Duration::ZERO,
        }
    }
}

/// Why a GPU job was refused. `reason()` is the stable code for the receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuRefusal {
    /// The quiet flag exists; carries its text (CR-028).
    QuietFlag(String),
    /// An `est_load` process is running (CR-029).
    EstLoad,
    /// A lock file could not be opened (CR-033, `gpu_open`).
    LockOpen(String),
    /// Another holder has `/tmp/aien-gb10.lock` (CR-033, `gpu_lock`).
    LockHeld(PathBuf),
    /// Another holder has aien-proof's `slots/gpu.lock` (`gpu_lock_proof`).
    ProofLockHeld(PathBuf),
    /// The locks were taken but MemAvailable stayed below what the gate needs
    /// (`gpu_mem`, retryable: nothing ran). `None` = the reading was unreadable.
    MemShort { free_mb: Option<u64>, need_mb: u64 },
}

impl GpuRefusal {
    pub fn reason(&self) -> &'static str {
        match self {
            GpuRefusal::QuietFlag(_) => "quiet_flag",
            GpuRefusal::EstLoad => "est_load",
            GpuRefusal::LockOpen(_) => "gpu_open",
            GpuRefusal::LockHeld(_) => "gpu_lock",
            GpuRefusal::ProofLockHeld(_) => "gpu_lock_proof",
            GpuRefusal::MemShort { .. } => "gpu_mem",
        }
    }
}

impl fmt::Display for GpuRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuRefusal::QuietFlag(t) => write!(f, "quiet flag is up: {t}"),
            GpuRefusal::EstLoad => write!(f, "an est_load process is running"),
            GpuRefusal::LockOpen(m) => write!(f, "cannot open GPU lock: {m}"),
            GpuRefusal::LockHeld(p) => write!(f, "GPU lock {} is held", p.display()),
            GpuRefusal::ProofLockHeld(p) => {
                write!(f, "aien-proof GPU lock {} is held", p.display())
            }
            GpuRefusal::MemShort { free_mb, need_mb } => match free_mb {
                Some(free) => write!(
                    f,
                    "free memory {free} MB, the gate needs {need_mb} MB (MemAvailable)"
                ),
                None => write!(
                    f,
                    "free memory unreadable, the gate needs {need_mb} MB (MemAvailable)"
                ),
            },
        }
    }
}

/// MemAvailable in MB from a meminfo file; None when it cannot be read.
pub fn mem_available_mb(meminfo: &Path) -> Option<u64> {
    let text = fs::read_to_string(meminfo).ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024)
}

/// Wait until MemAvailable reaches `need_mb`, polling every `GPU_MEM_POLL`
/// and giving up after `bound` (0 = one reading). Ok(free) when it did;
/// Err(last reading) when it did not.
pub fn wait_for_memory(meminfo: &Path, need_mb: u64, bound: Duration) -> Result<u64, Option<u64>> {
    let t0 = Instant::now();
    loop {
        let free = mem_available_mb(meminfo);
        if let Some(f) = free {
            if f >= need_mb {
                return Ok(f);
            }
        }
        let left = bound.saturating_sub(t0.elapsed());
        if left.is_zero() {
            return Err(free);
        }
        std::thread::sleep(GPU_MEM_POLL.min(left));
    }
}

/// True if any process under `proc_dir` has a name containing `est_load`
/// (what `pgrep est_load` matches, CR-029).
pub fn est_load_running(proc_dir: &Path) -> bool {
    let rd = match fs::read_dir(proc_dir) {
        Ok(r) => r,
        Err(_) => return false,
    };
    for e in rd.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if let Ok(comm) = fs::read_to_string(e.path().join("comm")) {
            if comm.trim().contains("est_load") {
                return true;
            }
        }
    }
    false
}

/// The cheap refusals that need no lock: quiet flag, then `est_load`. The
/// flag is never created, changed or removed here (CR-050).
pub fn preflight(cfg: &GpuConfig) -> Option<GpuRefusal> {
    if let Some(flag) = &cfg.quiet_flag {
        if fs::symlink_metadata(flag).is_ok() {
            let text = fs::read_to_string(flag).unwrap_or_default();
            return Some(GpuRefusal::QuietFlag(text.trim().to_string()));
        }
    }
    if est_load_running(&cfg.proc_dir) {
        return Some(GpuRefusal::EstLoad);
    }
    None
}

/// Descriptors holding the GPU locks. Drop releases them, under the spawn
/// gate (see the module comment, "Fork window").
#[derive(Debug)]
pub struct GpuGuard {
    files: Vec<File>,
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        let _gate = crate::process::spawn_gate();
        self.files.clear();
    }
}

/// The default directory of the target-dir lock files for a bare `run`
/// without a worker state dir (the worker uses `<state>/locks/target`).
pub const DEFAULT_TARGET_LOCK_DIR: &str = "/tmp/aien-test-target-locks";

/// The lock file for one cargo target dir: `<lock_dir>/<sanitised>.lock`,
/// where every character of the target dir path outside `[A-Za-z0-9._-]`
/// becomes `_`. The path is taken as given (no canonicalising), so every
/// holder that names the same target dir names the same lock file. Not
/// inside the target dir itself: under the user sandbox that belongs to
/// `aien-run`, and the lock is the worker's.
pub fn target_lock_path(lock_dir: &Path, target_dir: &Path) -> PathBuf {
    let name: String = target_dir
        .display()
        .to_string()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    lock_dir.join(format!("{name}.lock"))
}

/// Exclusive hold of one cargo target dir (`flock(LOCK_EX)`, waited for).
/// Held by the runner for a whole cargo gate, build and run, so two gates of
/// one head never build in, or relink a binary under, the same target dir
/// at once (sc fix/521: SC-AGENT-PREFLIGHT relinked a test binary while
/// SC-TEST was executing it). Dropping it closes the descriptor under the
/// spawn gate, like `GpuGuard`.
#[derive(Debug)]
pub struct TargetLock {
    file: Option<File>,
    path: PathBuf,
}

impl TargetLock {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TargetLock {
    fn drop(&mut self) {
        let _gate = crate::process::spawn_gate();
        self.file.take();
    }
}

/// Take the target-dir lock at `path` (see `target_lock_path`), waiting for
/// the holder to finish. The file is created if absent and never deleted.
pub fn lock_target_dir(path: &Path) -> io::Result<TargetLock> {
    let file = open_lock(path)?;
    flock_exclusive(&file, true)?;
    Ok(TargetLock {
        file: Some(file),
        path: path.to_path_buf(),
    })
}

fn open_lock(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// `flock(LOCK_EX)`: Ok(true) when taken, Ok(false) when held by someone else
/// and `wait` is off.
fn flock_exclusive(file: &File, wait: bool) -> io::Result<bool> {
    let op = if wait {
        libc::LOCK_EX
    } else {
        libc::LOCK_EX | libc::LOCK_NB
    };
    loop {
        // SAFETY: flock on a descriptor that `file` keeps open for this call.
        let rc = unsafe { libc::flock(file.as_raw_fd(), op) };
        if rc == 0 {
            return Ok(true);
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EINTR) => {}
            Some(libc::EWOULDBLOCK) => return Ok(false),
            _ => return Err(err),
        }
    }
}

/// Wait for `file`'s lock in the kernel's queue until `deadline`. The
/// blocking `flock` runs on a helper thread; after the deadline the helper is
/// abandoned and drops the lock (under the spawn gate) the moment the kernel
/// grants it, so a refused gate never keeps the GPU. An abandoned helper
/// stays queued until the holder lets go: one per refused gate at most.
fn flock_queued(file: File, deadline: Instant) -> io::Result<Option<File>> {
    let (tx, rx) = mpsc::channel();
    let gave_up = Arc::new(AtomicBool::new(false));
    let gave_up_t = Arc::clone(&gave_up);
    std::thread::spawn(move || {
        let r = flock_exclusive(&file, true).map(|_| file);
        let lost = |r: io::Result<File>| {
            if let Ok(f) = r {
                let _gate = crate::process::spawn_gate();
                drop(f);
            }
        };
        if gave_up_t.load(Ordering::SeqCst) {
            return lost(r);
        }
        if let Err(mpsc::SendError(r)) = tx.send(r) {
            lost(r);
        }
    });
    match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(r) => r.map(Some),
        Err(_) => {
            gave_up.store(true, Ordering::SeqCst);
            // The helper may have sent just before the flag: take and drop it.
            if let Ok(Ok(f)) = rx.try_recv() {
                let _gate = crate::process::spawn_gate();
                drop(f);
            }
            Ok(None)
        }
    }
}

/// Take the lock at `path`: at once if it is free, else as `wait` says, with
/// `deadline` the end of a bounded wait. Ok(None) = still held.
fn take_lock(
    path: &Path,
    wait: GpuWait,
    deadline: Option<Instant>,
) -> Result<Option<File>, GpuRefusal> {
    let open_err = |e: io::Error| GpuRefusal::LockOpen(format!("{}: {e}", path.display()));
    let file = open_lock(path).map_err(open_err)?;
    if flock_exclusive(&file, false).map_err(open_err)? {
        return Ok(Some(file));
    }
    match (wait, deadline) {
        (GpuWait::Refuse, _) => Ok(None),
        (GpuWait::Forever, _) => flock_exclusive(&file, true)
            .map(|_| Some(file))
            .map_err(open_err),
        (GpuWait::Bounded(_), Some(d)) => flock_queued(file, d).map_err(open_err),
        (GpuWait::Bounded(_), None) => Ok(None),
    }
}

/// One `acquire_gpu_timed` call: the result and how long it waited for the
/// locks (0 when they were free or a preflight refusal came first).
#[derive(Debug)]
pub struct GpuAttempt {
    pub result: Result<GpuGuard, GpuRefusal>,
    pub waited: Duration,
    /// MemAvailable (MB) read after the locks were taken; None when the
    /// memory check is off or the locks were refused first.
    pub mem_free_mb: Option<u64>,
    /// `GpuConfig::mem_need_mb` at the time of the attempt (0 = not checked).
    pub mem_need_mb: u64,
}

/// The GPU job protocol up to the point of running: refusals first, then the
/// locks in the fixed order. Call it after taking the in-process gb10 slot.
pub fn acquire_gpu(cfg: &GpuConfig) -> Result<GpuGuard, GpuRefusal> {
    acquire_gpu_timed(cfg).result
}

/// `acquire_gpu`, also reporting the wait. One bound covers both locks. With
/// a memory need set, the memory is checked once the locks are held; a short
/// reading waits up to `mem_wait` for it, then refuses as `gpu_mem` and gives
/// the locks back (the guard drops here, through the spawn gate).
pub fn acquire_gpu_timed(cfg: &GpuConfig) -> GpuAttempt {
    let t0 = Instant::now();
    let (result, mem_free_mb) = match acquire_locks(cfg, t0) {
        Err(r) => (Err(r), None),
        Ok(guard) if cfg.mem_need_mb == 0 => (Ok(guard), None),
        Ok(guard) => match wait_for_memory(&cfg.meminfo, cfg.mem_need_mb, cfg.mem_wait) {
            Ok(free) => (Ok(guard), Some(free)),
            Err(free) => {
                drop(guard);
                (
                    Err(GpuRefusal::MemShort {
                        free_mb: free,
                        need_mb: cfg.mem_need_mb,
                    }),
                    free,
                )
            }
        },
    };
    GpuAttempt {
        result,
        waited: t0.elapsed(),
        mem_free_mb,
        mem_need_mb: cfg.mem_need_mb,
    }
}

fn acquire_locks(cfg: &GpuConfig, t0: Instant) -> Result<GpuGuard, GpuRefusal> {
    if let Some(r) = preflight(cfg) {
        return Err(r);
    }
    let deadline = match cfg.wait {
        GpuWait::Bounded(d) => t0.checked_add(d),
        _ => None,
    };
    let primary = match take_lock(&cfg.lock_path, cfg.wait, deadline)? {
        Some(f) => f,
        None => return Err(GpuRefusal::LockHeld(cfg.lock_path.clone())),
    };
    let mut guard = GpuGuard {
        files: vec![primary],
    };
    if let Some(p) = &cfg.proof_lock_path {
        match take_lock(p, cfg.wait, deadline)? {
            Some(f) => guard.files.push(f),
            // Returning drops `guard`, so the first lock is not left held,
            // and it is released through the spawn gate like any other.
            None => return Err(GpuRefusal::ProofLockHeld(p.clone())),
        }
    }
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{queued_on, write_file, TempDir};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn semaphore_counts_slots() {
        let s = Semaphore::new(2);
        assert_eq!((s.size(), s.available()), (2, 2));
        let a = s.try_acquire();
        let b = s.try_acquire();
        assert!(a.is_some() && b.is_some());
        assert_eq!(s.available(), 0);
        assert!(s.try_acquire().is_none());
        drop(a);
        assert_eq!(s.available(), 1);
        assert!(s.try_acquire().is_some());
    }

    #[test]
    fn zero_size_is_one_slot() {
        assert_eq!(Semaphore::new(0).size(), 1);
    }

    #[test]
    fn acquire_waits_for_a_release() {
        let sem = Semaphore::new(1);
        let held = sem.acquire();
        let (tx, rx) = mpsc::channel();
        thread::scope(|s| {
            s.spawn(|| {
                let _p = sem.acquire();
                tx.send(()).unwrap();
            });
            thread::sleep(Duration::from_millis(150));
            assert!(rx.try_recv().is_err(), "the second acquire must wait");
            drop(held);
            rx.recv_timeout(Duration::from_secs(20))
                .expect("the waiter gets the slot after the release");
        });
    }

    #[test]
    fn target_lock_path_is_one_file_per_target_dir_under_the_lock_dir() {
        let p = target_lock_path(
            Path::new("/state/locks/target"),
            Path::new("/var/lib/aien-run/target/aien-sovereign-core/pr"),
        );
        assert_eq!(
            p,
            PathBuf::from(
                "/state/locks/target/_var_lib_aien-run_target_aien-sovereign-core_pr.lock"
            )
        );
        let main = target_lock_path(Path::new("/l"), Path::new("/t/main"));
        let pr = target_lock_path(Path::new("/l"), Path::new("/t/pr"));
        assert_ne!(main, pr, "main and pr are different target dirs");
    }

    /// Two holders in one process (two `open`s, two file descriptions) and
    /// so two threads of the runner exclude each other; the second waits
    /// until the first is dropped.
    #[test]
    fn target_lock_is_exclusive_and_waited_for() {
        let dir = TempDir::new("tlock");
        let path = target_lock_path(dir.path(), Path::new("/t/pr"));
        let held = lock_target_dir(&path).unwrap();
        assert!(held.path().exists());
        let (tx, rx) = mpsc::channel();
        thread::scope(|s| {
            s.spawn(|| {
                let _second = lock_target_dir(&path).unwrap();
                tx.send(()).unwrap();
            });
            thread::sleep(Duration::from_millis(200));
            assert!(rx.try_recv().is_err(), "the second holder must wait");
            drop(held);
            rx.recv_timeout(Duration::from_secs(20))
                .expect("the waiter takes the lock after the release");
        });
    }

    #[test]
    fn gb10_pool_is_serialized() {
        let p = Pools::new(PoolSizes { host: 4, qemu: 2 });
        let first = p.try_acquire(Pool::Gb10);
        assert!(first.is_some());
        assert!(p.try_acquire(Pool::Gb10).is_none(), "gb10 has one slot");
        assert_eq!(p.semaphore(Pool::Gb10).size(), 1);
        assert_eq!(p.semaphore(Pool::Host).size(), 4);
        assert_eq!(p.semaphore(Pool::Qemu).size(), 2);
        drop(first);
        assert!(p.try_acquire(Pool::Gb10).is_some());
    }

    #[test]
    fn pool_names_round_trip() {
        for pool in [Pool::Gb10, Pool::Qemu, Pool::Host] {
            assert_eq!(Pool::from_name(pool.name()), Some(pool));
        }
        assert_eq!(Pool::from_name("gpu"), None);
        assert!(Pool::Gb10 < Pool::Qemu && Pool::Qemu < Pool::Host);
    }

    #[test]
    fn default_sizes_follow_the_adr() {
        assert_eq!(PoolSizes::defaults_for(8), PoolSizes { host: 8, qemu: 4 });
        assert_eq!(PoolSizes::defaults_for(20), PoolSizes { host: 20, qemu: 4 });
        assert_eq!(PoolSizes::defaults_for(4), PoolSizes { host: 4, qemu: 2 });
        assert_eq!(PoolSizes::defaults_for(1), PoolSizes { host: 1, qemu: 1 });
        assert_eq!(PoolSizes::defaults_for(0), PoolSizes { host: 1, qemu: 1 });
        assert!(PoolSizes::detect().host >= 1);
    }

    #[test]
    fn jobs_option_parses_and_overrides() {
        let j = parse_jobs("host=3,qemu=2").unwrap();
        assert_eq!(
            j,
            JobsSpec {
                host: Some(3),
                qemu: Some(2)
            }
        );
        let j = parse_jobs("qemu=5").unwrap();
        assert_eq!(j.host, None);
        let base = PoolSizes { host: 8, qemu: 4 };
        assert_eq!(base.with(j), PoolSizes { host: 8, qemu: 5 });
        for bad in [
            "", "host", "host=", "host=0", "host=x", "gb10=2", "disk=1", "host=1,",
        ] {
            assert!(parse_jobs(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn acquire_many_uses_one_order() {
        let p = Pools::new(PoolSizes { host: 1, qemu: 1 });
        // Opposite orders and repeats: the fixed order keeps them deadlock free.
        thread::scope(|s| {
            for flip in [false, true] {
                let p = &p;
                s.spawn(move || {
                    for _ in 0..100 {
                        let want = if flip {
                            [Pool::Host, Pool::Qemu, Pool::Gb10, Pool::Host]
                        } else {
                            [Pool::Gb10, Pool::Qemu, Pool::Host, Pool::Gb10]
                        };
                        let permits = p.acquire_many(&want);
                        assert_eq!(permits.len(), 3);
                    }
                });
            }
        });
        for pool in [Pool::Gb10, Pool::Qemu, Pool::Host] {
            assert_eq!(p.semaphore(pool).available(), 1, "{pool:?} slot leaked");
        }
    }

    #[test]
    fn gpu_lock_is_exclusive_and_released_on_drop() {
        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        let g = acquire_gpu(&cfg).expect("free lock");
        match acquire_gpu(&cfg) {
            Err(GpuRefusal::LockHeld(p)) => assert_eq!(p, cfg.lock_path),
            other => panic!("expected LockHeld, got {other:?}"),
        }
        drop(g);
        let again = acquire_gpu(&cfg);
        assert!(again.is_ok(), "released by drop: {again:?}");
        assert!(cfg.lock_path.exists(), "the lock file is never deleted");
    }

    /// Regression for the forge flake at e703501c: a child that is between
    /// fork and exec holds a copy of the lock's open file description, so the
    /// lock looked held after the guard was dropped. The child below stays in
    /// that window for 150 ms (its `pre_exec` hook sleeps), the guard is
    /// dropped meanwhile, and the next acquire must succeed. The drop waits for
    /// the spawn to finish (the spawn gate), so this passes whatever the
    /// scheduling; without the gate it fails because the child still holds the
    /// lock when the re-acquire runs.
    #[test]
    fn gpu_lock_release_waits_for_a_child_between_fork_and_exec() {
        use std::os::unix::process::CommandExt;
        use std::process::Command;

        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        let mut fds: [libc::c_int; 2] = [0, 0];
        // SAFETY: `fds` has room for the two descriptors pipe2 writes.
        let rc = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(rc, 0, "pipe2");
        let (ready_r, ready_w) = (fds[0], fds[1]);

        for round in 0..3 {
            let g = acquire_gpu(&cfg).unwrap_or_else(|e| panic!("round {round}: {e}"));
            thread::scope(|s| {
                let spawner = s.spawn(move || {
                    let mut cmd = Command::new("/bin/true");
                    // SAFETY: the hook runs between fork and exec and only
                    // calls write(2) and poll(2), which are async-signal-safe;
                    // it allocates nothing and touches no lock.
                    unsafe {
                        cmd.pre_exec(move || {
                            let byte = [1u8];
                            libc::write(ready_w, byte.as_ptr().cast(), 1);
                            // poll with no descriptors is a sleep: 150 ms.
                            libc::poll(std::ptr::null_mut(), 0, 150);
                            Ok(())
                        });
                    }
                    let mut child = crate::process::spawn_gated(&mut cmd).expect("spawn");
                    child.wait().expect("wait")
                });
                // The child is now alive and between fork and exec.
                let mut pfd = libc::pollfd {
                    fd: ready_r,
                    events: libc::POLLIN,
                    revents: 0,
                };
                // SAFETY: one valid pollfd.
                let n = unsafe { libc::poll(&mut pfd, 1, 20_000) };
                assert_eq!(n, 1, "round {round}: the child never reached pre_exec");
                let mut byte = [0u8; 1];
                // SAFETY: reads at most one byte into a one byte buffer.
                let got = unsafe { libc::read(ready_r, byte.as_mut_ptr().cast(), 1) };
                assert_eq!(got, 1, "round {round}: ready byte");

                drop(g);
                let again = acquire_gpu(&cfg);
                assert!(
                    again.is_ok(),
                    "round {round}: a child between fork and exec kept the lock: {again:?}"
                );
                drop(again);
                let status = spawner.join().expect("spawner thread");
                assert!(status.success(), "round {round}: /bin/true failed");
            });
        }
        // SAFETY: both descriptors are ours and no thread is using them now.
        unsafe {
            libc::close(ready_r);
            libc::close(ready_w);
        }
    }

    #[test]
    fn held_lock_reason_is_gpu_lock() {
        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        let _g = acquire_gpu(&cfg).unwrap();
        let e = acquire_gpu(&cfg).unwrap_err();
        assert_eq!(e.reason(), "gpu_lock");
        assert!(e.to_string().contains("is held"));
    }

    #[test]
    fn bounded_wait_takes_the_lock_when_the_holder_lets_go() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        let held = acquire_gpu(&cfg).unwrap();
        cfg.wait = GpuWait::Bounded(Duration::from_secs(20));
        // Let go 300 ms after the waiter shows up in the kernel queue.
        let lock = cfg.lock_path.clone();
        let holder = thread::spawn(move || {
            let t0 = Instant::now();
            while !queued_on(&lock) && t0.elapsed() < Duration::from_secs(15) {
                thread::sleep(Duration::from_millis(5));
            }
            let seen = queued_on(&lock);
            thread::sleep(Duration::from_millis(300));
            drop(held);
            seen
        });
        let a = acquire_gpu_timed(&cfg);
        assert!(
            holder.join().unwrap(),
            "the waiter never queued in the kernel"
        );
        assert!(a.result.is_ok(), "{:?}", a.result);
        assert!(
            (Duration::from_millis(250)..Duration::from_secs(20)).contains(&a.waited),
            "waited {:?}",
            a.waited
        );
    }

    #[test]
    fn bounded_wait_gives_up_and_the_abandoned_waiter_hands_the_lock_back() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        let held = acquire_gpu(&cfg).unwrap();
        cfg.wait = GpuWait::Bounded(Duration::from_millis(300));
        let a = acquire_gpu_timed(&cfg);
        assert_eq!(a.result.unwrap_err().reason(), "gpu_lock");
        assert!(a.waited >= Duration::from_millis(300), "{:?}", a.waited);
        assert!(a.waited < Duration::from_secs(10), "{:?}", a.waited);
        drop(held);
        // The helper still queued in the kernel gets the lock and drops it at
        // once; a refusing caller then finds it free within moments.
        cfg.wait = GpuWait::Refuse;
        let t0 = Instant::now();
        loop {
            if acquire_gpu(&cfg).is_ok() {
                break;
            }
            assert!(
                t0.elapsed() < Duration::from_secs(10),
                "lock never handed back"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn bounded_wait_on_a_held_proof_lock_refuses_with_its_reason() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        cfg.proof_lock_path = Some(d.path().join("slots/gpu.lock"));
        let proof_only = GpuConfig {
            lock_path: d.path().join("other.lock"),
            ..cfg.clone()
        };
        let held = acquire_gpu(&proof_only).unwrap();
        cfg.wait = GpuWait::Bounded(Duration::from_millis(300));
        let a = acquire_gpu_timed(&cfg);
        assert_eq!(a.result.unwrap_err().reason(), "gpu_lock_proof");
        assert!(a.waited >= Duration::from_millis(300), "{:?}", a.waited);
        assert!(a.waited < Duration::from_secs(10), "{:?}", a.waited);
        drop(held);
    }

    fn fake_meminfo(dir: &Path, mb: u64) {
        write_file(
            dir,
            "meminfo",
            &format!(
                "MemTotal:       127598528 kB\nMemAvailable:   {} kB\n",
                mb * 1024
            ),
            false,
        );
    }

    #[test]
    fn short_memory_is_refused_as_gpu_mem_and_gives_the_locks_back() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        cfg.mem_need_mb = 100;
        cfg.mem_wait = Duration::from_millis(300);
        fake_meminfo(d.path(), 40);
        let a = acquire_gpu_timed(&cfg);
        assert!(
            matches!(
                a.result,
                Err(GpuRefusal::MemShort {
                    free_mb: Some(40),
                    need_mb: 100
                })
            ),
            "{:?}",
            a.result.as_ref().err()
        );
        assert_eq!(a.result.as_ref().err().unwrap().reason(), "gpu_mem");
        assert_eq!((a.mem_free_mb, a.mem_need_mb), (Some(40), 100));
        assert!(a.waited >= Duration::from_millis(300), "{:?}", a.waited);
        // The refused gate released the lock: the next job takes it.
        cfg.mem_need_mb = 0;
        assert!(acquire_gpu(&cfg).is_ok());
    }

    #[test]
    fn enough_memory_runs_and_records_the_reading() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        cfg.mem_need_mb = 100;
        cfg.mem_wait = Duration::from_millis(300);
        fake_meminfo(d.path(), 200);
        let a = acquire_gpu_timed(&cfg);
        assert!(a.result.is_ok(), "{:?}", a.result.as_ref().err());
        assert_eq!((a.mem_free_mb, a.mem_need_mb), (Some(200), 100));
    }

    #[test]
    fn memory_that_comes_back_within_the_bound_runs() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        cfg.mem_need_mb = 100;
        cfg.mem_wait = Duration::from_secs(20);
        fake_meminfo(d.path(), 40);
        let p = d.path().join("meminfo");
        let freer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(700));
            fs::write(p, "MemTotal: 127598528 kB\nMemAvailable:   204800 kB\n").unwrap();
        });
        let a = acquire_gpu_timed(&cfg);
        freer.join().unwrap();
        assert!(a.result.is_ok(), "{:?}", a.result.as_ref().err());
        assert_eq!(a.mem_free_mb, Some(200));
        assert!(a.waited < Duration::from_secs(20), "{:?}", a.waited);
    }

    #[test]
    fn unreadable_memory_is_refused_not_run() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        cfg.mem_need_mb = 100;
        // No meminfo file in the scratch dir.
        let a = acquire_gpu_timed(&cfg);
        assert!(matches!(
            a.result,
            Err(GpuRefusal::MemShort {
                free_mb: None,
                need_mb: 100
            })
        ));
        assert_eq!(a.mem_free_mb, None);
    }

    #[test]
    fn the_machine_default_checks_the_measured_floor() {
        let cfg = GpuConfig::detect();
        assert_eq!(cfg.mem_need_mb, DEFAULT_GPU_MEM_MB);
        assert_eq!(cfg.meminfo, PathBuf::from(DEFAULT_MEMINFO));
        assert_eq!(GpuConfig::rooted_at(Path::new("/x")).mem_need_mb, 0);
    }

    #[test]
    fn open_handle_without_flock_does_not_block() {
        // CR-052: only a flock holder blocks; a plain open handle does not.
        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        write_file(d.path(), "gb10.lock", "", false);
        let _plain = File::open(&cfg.lock_path).unwrap();
        assert!(acquire_gpu(&cfg).is_ok());
    }

    #[test]
    fn quiet_flag_refuses_and_is_left_alone() {
        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        write_file(d.path(), "quiet-flag", "drake start=now\n", false);
        match acquire_gpu(&cfg) {
            Err(GpuRefusal::QuietFlag(t)) => assert_eq!(t, "drake start=now"),
            other => panic!("expected QuietFlag, got {other:?}"),
        }
        assert_eq!(
            fs::read_to_string(d.path().join("quiet-flag")).unwrap(),
            "drake start=now\n",
            "the flag must never be changed"
        );
        assert!(!cfg.lock_path.exists(), "refused before touching the lock");
        assert_eq!(preflight(&cfg).unwrap().reason(), "quiet_flag");
    }

    #[test]
    fn est_load_process_refuses() {
        let d = TempDir::new("gpu");
        let cfg = GpuConfig::rooted_at(d.path());
        assert!(preflight(&cfg).is_none(), "no proc dir means no est_load");
        write_file(d.path(), "proc/101/comm", "bash\n", false);
        write_file(d.path(), "proc/self/comm", "est_load\n", false);
        assert!(
            !est_load_running(&cfg.proc_dir),
            "non-numeric entries are ignored"
        );
        write_file(d.path(), "proc/202/comm", "est_load\n", false);
        assert!(est_load_running(&cfg.proc_dir));
        match acquire_gpu(&cfg) {
            Err(GpuRefusal::EstLoad) => {}
            other => panic!("expected EstLoad, got {other:?}"),
        }
    }

    #[test]
    fn proof_lock_is_second_and_blocks_the_whole_acquire() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        let proof = d.path().join("board/slots/gpu.lock");
        cfg.proof_lock_path = Some(proof.clone());
        // Someone else (aien-proof) holds its lock. A `GpuGuard` stands in for
        // that holder so its release goes through the spawn gate too.
        let other = open_lock(&proof).unwrap();
        assert!(flock_exclusive(&other, false).unwrap());
        let other = GpuGuard { files: vec![other] };
        match acquire_gpu(&cfg) {
            Err(GpuRefusal::ProofLockHeld(p)) => assert_eq!(p, proof),
            got => panic!("expected ProofLockHeld, got {got:?}"),
        }
        // The failed attempt must not leave the first lock held.
        let mut only_first = cfg.clone();
        only_first.proof_lock_path = None;
        assert!(acquire_gpu(&only_first).is_ok());
        drop(other);
        let both = acquire_gpu(&cfg).expect("both free now");
        // While both are held, each is refused to a second holder.
        assert!(matches!(acquire_gpu(&cfg), Err(GpuRefusal::LockHeld(_))));
        drop(both);
    }

    #[test]
    fn wait_mode_blocks_until_the_holder_releases() {
        let d = TempDir::new("gpu");
        let mut cfg = GpuConfig::rooted_at(d.path());
        let held = acquire_gpu(&cfg).unwrap();
        cfg.wait = GpuWait::Forever;
        let (tx, rx) = mpsc::channel();
        thread::scope(|s| {
            s.spawn(|| {
                let g = acquire_gpu(&cfg);
                tx.send(g.is_ok()).unwrap();
            });
            thread::sleep(Duration::from_millis(150));
            assert!(rx.try_recv().is_err(), "wait mode must block");
            drop(held);
            assert!(rx.recv_timeout(Duration::from_secs(20)).unwrap());
        });
    }

    #[test]
    fn detect_builds_a_config_without_touching_anything() {
        let c = GpuConfig::detect();
        assert_eq!(c.lock_path, PathBuf::from(DEFAULT_GPU_LOCK));
        assert_eq!(c.wait, GpuWait::Bounded(DEFAULT_GPU_WAIT));
        assert_eq!(c.proc_dir, PathBuf::from("/proc"));
        let _ = Hardware::detect();
    }
}
