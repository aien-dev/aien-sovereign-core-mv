//! Spawn a command without a shell, capture stdout and stderr, enforce a
//! wall-clock timeout by killing the whole process group (SIGTERM, then
//! SIGKILL after a grace period). No async: one waiter thread and two reader
//! threads per child.

use std::io::{self, Read};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::{mpsc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

pub struct RunSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub timeout: Duration,
    /// Time between SIGTERM and SIGKILL (ADR 0028 Decision 6 says 10 s).
    pub kill_grace: Duration,
    /// false: the job is never signalled and never timed out (ADR 0028
    /// Decision 6, gb10 jobs). `timeout` is then advisory and only sets
    /// `timeout_exceeded`.
    pub enforce_timeout: bool,
    /// Extra `K=V` pairs for the program's environment, set by the runner
    /// (never by a manifest). The user sandbox forwards them as `--env`.
    pub env: Vec<(String, String)>,
}

#[derive(Debug)]
pub struct RunOutput {
    /// Exit code; None if the child was killed by a signal.
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// The runner killed the job because `timeout` passed.
    pub timed_out: bool,
    /// The job ran longer than `timeout`, whether or not it was killed.
    pub timeout_exceeded: bool,
    pub duration_ms: u64,
}

type Waited = io::Result<ExitStatus>;

fn signal_group(pid: i32, sig: i32) {
    if pid > 0 {
        // SAFETY: plain syscall; a stale group id yields ESRCH, ignored.
        unsafe {
            libc::kill(-pid, sig);
        }
    }
}

fn lost() -> io::Error {
    io::Error::other("waiter thread lost")
}

/// Held around every child spawn in this crate and around the release of the
/// GPU lock descriptors (`resources::GpuGuard`), so the two never overlap.
///
/// Why: `flock(2)` locks belong to the open file description, and a child
/// created by fork (or clone) gets its own copy of the descriptor table. The
/// lock descriptor is close-on-exec (std opens every file with `O_CLOEXEC`,
/// `library/std/src/sys/fs/unix.rs` `open_c`), so a child drops its copy at
/// exec. Between the fork and the exec the child still holds the file
/// description, and the lock looks held even after the parent closed its own
/// descriptor. With several threads (parallel gates in the runner, parallel
/// unit tests) another thread's child can sit in that window exactly when a
/// guard is dropped, and the next `acquire_gpu` is refused with `LockHeld`
/// although nobody holds the lock (an intermittent failure of
/// `gpu_lock_is_exclusive_and_released_on_drop` was seen once on the forge).
///
/// `Command::spawn` returns only after the child has exec'd or failed (std
/// reads the close-on-exec pipe until it closes, `process/unix/unix.rs`
/// `spawn`). So a spawn made under this mutex has no child left in the window
/// when the mutex is released, and a guard released under it can only be
/// copied by spawns that start after the descriptors are already closed.
/// A genuinely held lock is still refused at once (ADR 0028 Decision 6).
static SPAWN_GATE: Mutex<()> = Mutex::new(());

/// Worktrees the operator's git may open although `aien-run` owns their
/// files. The worker adds one only after checking that it was made by INFRA's
/// `wt.sh`: its `.git` file is root-owned, not writable by anyone else, and
/// points into the operator's mirror, so every git setting and hook that git
/// reads there is the operator's, never the checked-out code's.
static TRUSTED_WORKTREES: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Trust `dir` for git (git's `safe.directory`) until the guard is dropped.
pub fn trust_worktree(dir: &Path) -> TrustGuard {
    let mut v = TRUSTED_WORKTREES.lock().unwrap_or_else(|p| p.into_inner());
    v.push(dir.to_path_buf());
    TrustGuard(dir.to_path_buf())
}

/// Removes its directory from the trusted list when dropped.
pub struct TrustGuard(PathBuf);

impl Drop for TrustGuard {
    fn drop(&mut self) {
        let mut v = TRUSTED_WORKTREES.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(i) = v.iter().position(|d| d == &self.0) {
            v.remove(i);
        }
    }
}

/// Give `cmd` the trusted worktrees as `safe.directory` entries through
/// git's command-line configuration (`GIT_CONFIG_COUNT`), which git honours
/// for that setting. Children that run git themselves (crumb) inherit it. With
/// no trusted worktree the command is left exactly as it was.
pub fn apply_git_trust(cmd: &mut Command) {
    let v = TRUSTED_WORKTREES.lock().unwrap_or_else(|p| p.into_inner());
    if v.is_empty() {
        return;
    }
    cmd.env("GIT_CONFIG_COUNT", v.len().to_string());
    for (i, d) in v.iter().enumerate() {
        cmd.env(format!("GIT_CONFIG_KEY_{i}"), "safe.directory");
        cmd.env(format!("GIT_CONFIG_VALUE_{i}"), d);
    }
}

/// `git` with the trusted worktrees applied (see [`apply_git_trust`]).
pub fn git_command() -> Command {
    let mut cmd = Command::new("git");
    apply_git_trust(&mut cmd);
    cmd
}

/// Take the spawn gate. It is a leaf lock: nothing else is taken while it is
/// held, and a poisoned gate is still usable (it guards no data).
pub(crate) fn spawn_gate() -> MutexGuard<'static, ()> {
    SPAWN_GATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// `cmd.spawn()` under the spawn gate. Every child this crate starts goes
/// through here (or `output_gated`), never through a bare `Command::spawn`.
pub(crate) fn spawn_gated(cmd: &mut Command) -> io::Result<Child> {
    let _gate = spawn_gate();
    cmd.spawn()
}

/// `cmd.output()` with the spawn under the spawn gate. Waiting for the output
/// happens after the gate is released, so a slow child never blocks others.
pub(crate) fn output_gated(cmd: &mut Command) -> io::Result<Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    spawn_gated(cmd)?.wait_with_output()
}

fn spawn(spec: &RunSpec) -> io::Result<Child> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .current_dir(&spec.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    let mut attempt = 0;
    loop {
        // One attempt per gate hold: the retry sleep below is outside it.
        match spawn_gated(&mut cmd) {
            Ok(c) => return Ok(c),
            // A script written moments ago can be briefly busy when another
            // thread forks while its write descriptor is still open.
            Err(e) if e.raw_os_error() == Some(libc::ETXTBSY) && attempt < 40 => {
                attempt += 1;
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e),
        }
    }
}

fn terminate(pid: i32, rx: &mpsc::Receiver<Waited>, grace: Duration) -> io::Result<Waited> {
    signal_group(pid, libc::SIGTERM);
    match rx.recv_timeout(grace) {
        Ok(r) => Ok(r),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            signal_group(pid, libc::SIGKILL);
            rx.recv().map_err(|_| lost())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(lost()),
    }
}

fn reader<R: Read + Send + 'static>(src: Option<R>) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut s) = src {
            let _ = s.read_to_end(&mut buf);
        }
        buf
    })
}

pub fn run(spec: &RunSpec) -> io::Result<RunOutput> {
    let start = Instant::now();
    let mut child = spawn(spec)?;
    let pid = child.id() as i32;
    let h_out = reader(child.stdout.take());
    let h_err = reader(child.stderr.take());
    let (tx, rx) = mpsc::channel::<Waited>();
    thread::spawn(move || {
        let r = child.wait();
        let _ = tx.send(r);
    });

    let mut timed_out = false;
    let waited: Waited = if !spec.enforce_timeout {
        // Advisory timeout: wait for the job however long it takes.
        rx.recv().map_err(|_| lost())?
    } else {
        match rx.recv_timeout(spec.timeout) {
            Ok(r) => r,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                timed_out = true;
                terminate(pid, &rx, spec.kill_grace)?
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(lost()),
        }
    };
    let waited_for = start.elapsed();
    if spec.enforce_timeout {
        // Whatever the leader did, nothing it started may outlive the run.
        signal_group(pid, libc::SIGKILL);
    }
    let status = waited?;
    let stdout = h_out.join().unwrap_or_default();
    let stderr = h_err.join().unwrap_or_default();
    Ok(RunOutput {
        exit_code: status.code(),
        signal: status.signal(),
        stdout,
        stderr,
        timed_out,
        timeout_exceeded: timed_out || (!spec.enforce_timeout && waited_for > spec.timeout),
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str, timeout_ms: u64) -> RunSpec {
        RunSpec {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), script.to_string()],
            cwd: PathBuf::from("/"),
            timeout: Duration::from_millis(timeout_ms),
            kill_grace: Duration::from_millis(500),
            enforce_timeout: true,
            env: Vec::new(),
        }
    }

    #[test]
    fn captures_output_and_exit_code() {
        let o = run(&sh("echo out; echo err 1>&2; exit 3", 10_000)).unwrap();
        assert_eq!(o.exit_code, Some(3));
        assert_eq!(o.stdout, b"out\n");
        assert_eq!(o.stderr, b"err\n");
        assert!(!o.timed_out);
    }

    #[test]
    fn timeout_kills_and_flags() {
        let o = run(&sh("sleep 20", 300)).unwrap();
        assert!(o.timed_out);
        assert_eq!(o.exit_code, None);
        assert!(o.signal.is_some());
        assert!(o.duration_ms < 10_000, "took {} ms", o.duration_ms);
    }

    #[test]
    fn timeout_kills_the_whole_group() {
        // The grandchild inherits stdout; if only the leader were killed the
        // reader threads would block until the grandchild's sleep ends.
        let o = run(&sh("sleep 20 & wait", 300)).unwrap();
        assert!(o.timed_out);
        assert!(o.duration_ms < 10_000, "took {} ms", o.duration_ms);
    }

    #[test]
    fn term_ignored_gets_killed_after_grace() {
        let o = run(&sh("trap '' TERM; sleep 20", 200)).unwrap();
        assert!(o.timed_out);
        assert!(o.duration_ms < 10_000, "took {} ms", o.duration_ms);
    }

    #[test]
    fn orphans_do_not_outlive_a_normal_exit() {
        let o = run(&sh("sleep 20 & echo done", 10_000)).unwrap();
        assert_eq!(o.exit_code, Some(0));
        assert!(!o.timed_out);
        assert!(o.duration_ms < 10_000, "took {} ms", o.duration_ms);
    }

    #[test]
    fn advisory_timeout_never_signals_the_job() {
        let mut s = sh("sleep 1; echo done", 100);
        s.enforce_timeout = false;
        let o = run(&s).unwrap();
        assert_eq!(o.exit_code, Some(0));
        assert_eq!(o.stdout, b"done\n");
        assert!(!o.timed_out, "an advisory timeout must not kill");
        assert!(o.timeout_exceeded, "but it must be reported");
        assert!(o.signal.is_none());
    }

    #[test]
    fn within_the_limit_is_not_exceeded() {
        let mut s = sh("echo quick", 20_000);
        s.enforce_timeout = false;
        let o = run(&s).unwrap();
        assert!(!o.timeout_exceeded);
        let o = run(&sh("echo quick", 20_000)).unwrap();
        assert!(!o.timeout_exceeded && !o.timed_out);
    }

    #[test]
    fn missing_program_is_an_error() {
        let mut s = sh("true", 1000);
        s.program = PathBuf::from("/nonexistent/aien-test-nope");
        assert!(run(&s).is_err());
    }
}
