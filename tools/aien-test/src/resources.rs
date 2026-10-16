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
//! slot, (2) the target-dir lock (`lock_target_dir`, cargo gates only, when
//! a `CARGO_TARGET_DIR` is set), (3) `/tmp/aien-gb10.lock`, (4) aien-proof's
//! `slots/gpu.lock` when the aien-proof board directory exists. A holder of
//! (2) already has its slot and never waits for one, so a gate blocked on
//! (2) with a slot in hand cannot close a cycle; the worker's target priming
//! takes (2) alone and never (3) or (4).
//!
//! GPU job protocol (CR-028, CR-029, CR-033, CR-052, CR-053): refuse at once
//! (never wait, never steal) if the quiet flag file exists or an `est_load`
//! process is running, then take `flock(2)` locks without blocking. A held lock
//! is a refusal, not a wait, unless the operator asked for `--wait-gpu`. The
//! lock files are created if absent and never deleted. Dropping a
//! `GpuGuard` closes the descriptors, and the kernel releases the locks (a
//! dead holder frees them too).
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
use std::sync::{Condvar, Mutex, MutexGuard};

pub const DEFAULT_GPU_LOCK: &str = "/tmp/aien-gb10.lock";

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
    /// `--wait-gpu`: block on a held lock instead of refusing.
    pub wait: bool,
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
            wait: false,
        }
    }

    /// Every path inside `dir` (a scratch directory): own lock, own quiet
    /// flag, no aien-proof lock, no `est_load`.
    pub fn rooted_at(dir: &Path) -> GpuConfig {
        GpuConfig {
            lock_path: dir.join("gb10.lock"),
            proof_lock_path: None,
            quiet_flag: Some(dir.join("quiet-flag")),
            proc_dir: dir.join("proc"),
            wait: false,
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
}

impl GpuRefusal {
    pub fn reason(&self) -> &'static str {
        match self {
            GpuRefusal::QuietFlag(_) => "quiet_flag",
            GpuRefusal::EstLoad => "est_load",
            GpuRefusal::LockOpen(_) => "gpu_open",
            GpuRefusal::LockHeld(_) => "gpu_lock",
            GpuRefusal::ProofLockHeld(_) => "gpu_lock_proof",
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
        }
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

fn take_lock(path: &Path, wait: bool) -> Result<Option<File>, GpuRefusal> {
    let file =
        open_lock(path).map_err(|e| GpuRefusal::LockOpen(format!("{}: {e}", path.display())))?;
    match flock_exclusive(&file, wait) {
        Ok(true) => Ok(Some(file)),
        Ok(false) => Ok(None),
        Err(e) => Err(GpuRefusal::LockOpen(format!("{}: {e}", path.display()))),
    }
}

/// The GPU job protocol up to the point of running: refusals first, then the
/// locks in the fixed order. Call it after taking the in-process gb10 slot.
pub fn acquire_gpu(cfg: &GpuConfig) -> Result<GpuGuard, GpuRefusal> {
    if let Some(r) = preflight(cfg) {
        return Err(r);
    }
    let primary = match take_lock(&cfg.lock_path, cfg.wait)? {
        Some(f) => f,
        None => return Err(GpuRefusal::LockHeld(cfg.lock_path.clone())),
    };
    let mut guard = GpuGuard {
        files: vec![primary],
    };
    if let Some(p) = &cfg.proof_lock_path {
        match take_lock(p, cfg.wait)? {
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
    use crate::testutil::{write_file, TempDir};
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
        cfg.wait = true;
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
        assert!(!c.wait);
        assert_eq!(c.proc_dir, PathBuf::from("/proc"));
        let _ = Hardware::detect();
    }
}
