// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The board: look up the stamp, join an identical run in flight, or claim the
//! job and run it behind the CPU slot, memory, and GPU gates.
//!
//! Board layout (default `~/.local/state/aien-proof`, override `AIEN_PROOF_DIR`):
//!   stamps/<key>.json   passing runs, reused by every agent and worktree
//!   fails/<key>.json    latest failing run, shared only with runs that joined it
//!   locks/<key>.lock    held while a job runs; identical requests block on it
//!   slots/cpu-N.lock    CPU slots (`AIEN_PROOF_SLOTS`)
//!   slots/gpu.lock      the GPU key, one GPU job at a time
//!   slots/res-<name>.lock  other exclusive keys, e.g. `machine-1`
//!   slots/<name>.holder    who holds a key right now (agent, pid, job, since)
//!   holds/<key>.json    record of each `hold` run (never reused)
//!   logs/<key>.log      full output of the latest real run
//!   ledger.jsonl        hash-chained record of every real run

use crate::{fingerprint, ledger, lock::FileLock};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GIB: u64 = 1024 * 1024 * 1024;

pub struct Board {
    pub root: PathBuf,
    pub cpu_slots: usize,
    pub min_free_bytes: u64,
    /// Longest wait for free memory before running anyway (a gate that never
    /// opens would hang the hive).
    pub mem_wait: Duration,
    pub quiet: bool,
}

pub struct Job {
    pub name: String,
    pub agent: String,
    /// Directory the command runs in; inputs are relative to it.
    pub base: PathBuf,
    pub inputs: Vec<PathBuf>,
    pub cmd: Vec<String>,
    pub gpu: bool,
    /// Exclusive keys held for the whole run, e.g. `machine-1`.
    pub resources: Vec<String>,
    pub toolchain: String,
}

impl Job {
    /// Every exclusive key this job needs, including `gpu` when set.
    pub fn resource_names(&self) -> Vec<String> {
        let mut names = self.resources.clone();
        if self.gpu {
            names.push("gpu".into());
        }
        names.sort();
        names.dedup();
        names
    }
}

/// Key names are path components: lowercase letters, digits and dashes only.
pub fn valid_resource(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// An exclusive key. The holder file is removed before the lock is released
/// (`Drop` runs before fields are dropped), so a named holder is never stale
/// while the key is free.
pub struct ResourceGuard {
    _lock: FileLock,
    holder: PathBuf,
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.holder);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub key: String,
    pub job: String,
    pub agent: String,
    pub cmd: Vec<String>,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub finished_at: u64,
    pub ledger_index: u64,
    pub ledger_hash: String,
}

#[derive(Debug)]
pub enum Outcome {
    /// A passing stamp already existed; nothing ran.
    Stamped(Record),
    /// An identical run was in flight; its result was shared, nothing ran.
    Joined(Record),
    /// This request ran the job.
    Ran(Record),
}

impl Outcome {
    pub fn record(&self) -> &Record {
        match self {
            Outcome::Stamped(r) | Outcome::Joined(r) | Outcome::Ran(r) => r,
        }
    }

    pub fn passed(&self) -> bool {
        self.record().exit_code == 0
    }

    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Stamped(_) => "stamped",
            Outcome::Joined(_) => "joined",
            Outcome::Ran(_) => "ran",
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env_num(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// `MemAvailable` from /proc/meminfo, in bytes.
pub fn mem_available() -> Option<u64> {
    let info = fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line
        .trim_start_matches("MemAvailable:")
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb * 1024)
}

fn tee<R: Read + Send + 'static>(
    mut src: R,
    log: Arc<Mutex<fs::File>>,
    echo: bool,
    to_stderr: bool,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            let n = match src.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            if let Ok(mut f) = log.lock() {
                let _ = f.write_all(&buf[..n]);
            }
            if echo {
                if to_stderr {
                    let _ = io::stderr().write_all(&buf[..n]);
                } else {
                    let _ = io::stdout().write_all(&buf[..n]);
                }
            }
        }
    })
}

impl Board {
    pub fn from_env() -> Self {
        let root = std::env::var_os("AIEN_PROOF_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."));
                home.join(".local/state/aien-proof")
            });
        let cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            root,
            cpu_slots: env_num("AIEN_PROOF_SLOTS")
                .map(|n| n as usize)
                .unwrap_or((cores / 8).max(2)),
            min_free_bytes: env_num("AIEN_PROOF_MIN_FREE_GB").unwrap_or(4) * GIB,
            mem_wait: Duration::from_secs(env_num("AIEN_PROOF_MEM_WAIT_SECS").unwrap_or(600)),
            quiet: false,
        }
    }

    fn note(&self, msg: &str) {
        if !self.quiet {
            eprintln!("[aien-proof] {msg}");
        }
    }

    fn record_path(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join(kind).join(format!("{key}.json"))
    }

    fn read_record(&self, kind: &str, key: &str) -> Option<Record> {
        serde_json::from_slice(&fs::read(self.record_path(kind, key)).ok()?).ok()
    }

    fn write_record(&self, kind: &str, rec: &Record) -> io::Result<()> {
        let path = self.record_path(kind, &rec.key);
        fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        fs::write(&tmp, serde_json::to_vec_pretty(rec)?)?;
        fs::rename(tmp, path)
    }

    fn replay(&self, how: &str, rec: &Record) {
        if self.quiet {
            return;
        }
        let verdict = if rec.exit_code == 0 { "pass" } else { "FAIL" };
        eprintln!(
            "[aien-proof] {}: {how} {verdict} (ran by {} in {} ms, ledger #{}), replaying output",
            rec.job, rec.agent, rec.duration_ms, rec.ledger_index
        );
        if let Ok(log) = fs::read(self.root.join("logs").join(format!("{}.log", rec.key))) {
            let _ = io::stdout().write_all(&log);
        }
    }

    fn acquire_cpu_slot(&self) -> io::Result<FileLock> {
        let mut noted = false;
        loop {
            for i in 0..self.cpu_slots.max(1) {
                if let Some(slot) =
                    FileLock::try_acquire(&self.root.join("slots").join(format!("cpu-{i}.lock")))?
                {
                    return Ok(slot);
                }
            }
            if !noted {
                self.note(&format!("all {} CPU slots busy, waiting", self.cpu_slots));
                noted = true;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn resource_lock_path(&self, name: &str) -> PathBuf {
        // `gpu` keeps its original path so older binaries still exclude each other.
        if name == "gpu" {
            self.root.join("slots/gpu.lock")
        } else {
            self.root.join("slots").join(format!("res-{name}.lock"))
        }
    }

    /// Acquire keys in sorted order so two jobs can never deadlock.
    pub fn acquire_resources(
        &self,
        names: &[String],
        agent: &str,
        job: &str,
    ) -> io::Result<Vec<ResourceGuard>> {
        let mut names: Vec<&String> = names.iter().collect();
        names.sort();
        names.dedup();
        let mut guards = Vec::with_capacity(names.len());
        for name in names {
            if !valid_resource(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "invalid resource name {name:?}: use lowercase letters, digits, dashes"
                    ),
                ));
            }
            let lock_path = self.resource_lock_path(name);
            let holder = self.root.join("slots").join(format!("{name}.holder"));
            let lock = match FileLock::try_acquire(&lock_path)? {
                Some(lock) => lock,
                None => {
                    let who = fs::read_to_string(&holder).unwrap_or_default();
                    let who = who.trim();
                    self.note(&format!(
                        "{name} key held by {}, waiting",
                        if who.is_empty() { "another run" } else { who }
                    ));
                    FileLock::acquire(&lock_path)?
                }
            };
            fs::write(
                &holder,
                format!(
                    "agent={agent} pid={} job={job} since={}\n",
                    std::process::id(),
                    now()
                ),
            )?;
            guards.push(ResourceGuard {
                _lock: lock,
                holder,
            });
        }
        Ok(guards)
    }

    /// Run `cmd` in `base`, teeing output to the terminal and `log_path`.
    /// Returns the exit code and wall time in milliseconds.
    fn execute(
        &self,
        cmd: &[String],
        base: &std::path::Path,
        log_path: &std::path::Path,
    ) -> io::Result<(i32, u64)> {
        fs::create_dir_all(log_path.parent().unwrap())?;
        let log = Arc::new(Mutex::new(fs::File::create(log_path)?));
        let started = Instant::now();
        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .current_dir(base)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let out = tee(
            child.stdout.take().unwrap(),
            log.clone(),
            !self.quiet,
            false,
        );
        let err = tee(child.stderr.take().unwrap(), log.clone(), !self.quiet, true);
        let status = child.wait()?;
        let _ = out.join();
        let _ = err.join();
        Ok((
            status.code().unwrap_or(-1),
            started.elapsed().as_millis() as u64,
        ))
    }

    /// Run `job.cmd` while holding its exclusive keys, with no fingerprint,
    /// stamp or reuse: hardware actions such as booting Machine 1 must happen
    /// every time they are asked for. Recorded in the ledger as an `audit`
    /// event whose payload is the full output.
    pub fn hold(&self, job: &Job) -> io::Result<Record> {
        if job.cmd.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty command"));
        }
        let names = job.resource_names();
        if names.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hold needs at least one --resource",
            ));
        }
        let _keys = self.acquire_resources(&names, &job.agent, &job.name)?;
        let mut id = blake3::Hasher::new();
        id.update(b"AIEN_PROOF_HOLD_V1");
        id.update(job.name.as_bytes());
        id.update(job.agent.as_bytes());
        id.update(&std::process::id().to_be_bytes());
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        id.update(&nanos.to_be_bytes());
        let key = id.finalize().to_hex().to_string();

        let log_path = self.root.join("logs").join(format!("{key}.log"));
        let (exit_code, duration_ms) = self.execute(&job.cmd, &job.base, &log_path)?;
        let payload = fs::read(&log_path)?;
        let verdict = if exit_code == 0 { "pass" } else { "fail" };
        let intent = format!("{verdict} exit={exit_code} resources={}", names.join(","));
        let event = ledger::append_action(
            &self.root, "audit", &job.agent, &job.name, &intent, &payload,
        )?;
        let rec = Record {
            key,
            job: job.name.clone(),
            agent: job.agent.clone(),
            cmd: job.cmd.clone(),
            exit_code,
            duration_ms,
            finished_at: now(),
            ledger_index: event.index,
            ledger_hash: ledger::hex(&event.hash),
        };
        self.write_record("holds", &rec)?;
        Ok(rec)
    }

    /// Current key holders as `(name, holder line)`.
    pub fn holders(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = fs::read_dir(self.root.join("slots"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let name = name.strip_suffix(".holder")?.to_string();
                let who = fs::read_to_string(e.path()).ok()?;
                Some((name, who.trim().to_string()))
            })
            .collect();
        out.sort();
        out
    }

    fn wait_for_memory(&self) {
        let start = Instant::now();
        let mut noted = false;
        while let Some(free) = mem_available() {
            if free >= self.min_free_bytes {
                return;
            }
            if start.elapsed() >= self.mem_wait {
                self.note("memory still low after the wait limit, running anyway");
                return;
            }
            if !noted {
                self.note(&format!(
                    "{} GiB free, below the {} GiB gate, waiting",
                    free / GIB,
                    self.min_free_bytes / GIB
                ));
                noted = true;
            }
            thread::sleep(Duration::from_millis(500));
        }
    }

    pub fn run(&self, job: &Job) -> io::Result<Outcome> {
        if job.cmd.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty command"));
        }
        let key =
            fingerprint::job_key(&job.base, &job.name, &job.cmd, &job.toolchain, &job.inputs)?;
        if let Some(rec) = self.read_record("stamps", &key) {
            self.replay("stamped", &rec);
            return Ok(Outcome::Stamped(rec));
        }

        let lock_path = self.root.join("locks").join(format!("{key}.lock"));
        let _job_lock = match FileLock::try_acquire(&lock_path)? {
            Some(lock) => lock,
            None => {
                let waited_from = now();
                self.note(&format!(
                    "{}: identical run already in flight, joining it",
                    job.name
                ));
                let lock = FileLock::acquire(&lock_path)?;
                if let Some(rec) = self.read_record("stamps", &key) {
                    self.replay("joined", &rec);
                    return Ok(Outcome::Joined(rec));
                }
                if let Some(rec) = self.read_record("fails", &key) {
                    if rec.finished_at >= waited_from {
                        self.replay("joined", &rec);
                        return Ok(Outcome::Joined(rec));
                    }
                }
                lock
            }
        };
        // A run may have finished between the first lookup and taking the lock.
        if let Some(rec) = self.read_record("stamps", &key) {
            self.replay("stamped", &rec);
            return Ok(Outcome::Stamped(rec));
        }

        let _cpu = self.acquire_cpu_slot()?;
        self.wait_for_memory();
        let _keys = self.acquire_resources(&job.resource_names(), &job.agent, &job.name)?;

        let log_path = self.root.join("logs").join(format!("{key}.log"));
        let (exit_code, duration_ms) = self.execute(&job.cmd, &job.base, &log_path)?;

        let payload = fs::read(&log_path)?;
        let verdict = if exit_code == 0 { "pass" } else { "fail" };
        let intent = format!("{verdict} exit={exit_code} key={key}");
        let event = ledger::append(&self.root, &job.agent, &job.name, &intent, &payload)?;
        let rec = Record {
            key,
            job: job.name.clone(),
            agent: job.agent.clone(),
            cmd: job.cmd.clone(),
            exit_code,
            duration_ms,
            finished_at: now(),
            ledger_index: event.index,
            ledger_hash: ledger::hex(&event.hash),
        };
        self.write_record(if exit_code == 0 { "stamps" } else { "fails" }, &rec)?;
        Ok(Outcome::Ran(rec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_dir;

    fn board(tag: &str) -> Board {
        Board {
            root: temp_dir(tag),
            cpu_slots: 4,
            min_free_bytes: 0,
            mem_wait: Duration::from_secs(1),
            quiet: true,
        }
    }

    /// A job whose command appends a line to `runs` (outside its inputs), so the
    /// number of lines is the number of real executions.
    fn job(base: &std::path::Path, name: &str, script: &str) -> Job {
        fs::create_dir_all(base.join("src")).unwrap();
        fs::write(base.join("src/lib.rs"), "pub fn a() {}").unwrap();
        Job {
            name: name.into(),
            agent: "tester".into(),
            base: base.to_path_buf(),
            inputs: vec![PathBuf::from("src")],
            cmd: vec![
                "sh".into(),
                "-c".into(),
                format!("echo x >> runs; {script}"),
            ],
            gpu: false,
            resources: vec![],
            toolchain: "test".into(),
        }
    }

    fn runs(base: &std::path::Path) -> usize {
        fs::read_to_string(base.join("runs"))
            .map(|s| s.lines().count())
            .unwrap_or(0)
    }

    #[test]
    fn pass_is_stamped_then_replayed_without_running() {
        let b = board("board-stamp");
        let base = temp_dir("base-stamp");
        let j = job(&base, "unit", "true");
        assert!(matches!(b.run(&j).unwrap(), Outcome::Ran(_)));
        let second = b.run(&j).unwrap();
        assert!(matches!(second, Outcome::Stamped(_)));
        assert!(second.passed());
        assert_eq!(runs(&base), 1);
        assert_eq!(
            ledger::verify(&b.root.join(ledger::LEDGER_FILE)).unwrap().0,
            1
        );
    }

    #[test]
    fn a_second_worktree_with_the_same_code_reuses_the_stamp() {
        let b = board("board-worktree");
        let (wt1, wt2) = (temp_dir("wt1"), temp_dir("wt2"));
        assert!(matches!(
            b.run(&job(&wt1, "unit", "true")).unwrap(),
            Outcome::Ran(_)
        ));
        assert!(matches!(
            b.run(&job(&wt2, "unit", "true")).unwrap(),
            Outcome::Stamped(_)
        ));
        assert_eq!(runs(&wt1) + runs(&wt2), 1);
    }

    #[test]
    fn identical_concurrent_requests_run_once() {
        let b = Arc::new(board("board-join"));
        let base = temp_dir("base-join");
        let j = Arc::new(job(&base, "slow", "sleep 1"));
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let (b, j) = (b.clone(), j.clone());
                thread::spawn(move || b.run(&j).unwrap())
            })
            .collect();
        let outcomes: Vec<Outcome> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(runs(&base), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|o| matches!(o, Outcome::Ran(_)))
                .count(),
            1
        );
        assert!(outcomes.iter().all(Outcome::passed));
    }

    #[test]
    fn failures_are_not_stamped_but_are_shared_with_joiners() {
        let b = Arc::new(board("board-fail"));
        let base = temp_dir("base-fail");
        let j = Arc::new(job(&base, "broken", "sleep 1; exit 3"));
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let (b, j) = (b.clone(), j.clone());
                thread::spawn(move || b.run(&j).unwrap())
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().record().exit_code, 3);
        }
        assert_eq!(runs(&base), 1);
        // A fresh request after the failure runs again rather than trusting it.
        assert!(matches!(b.run(&j).unwrap(), Outcome::Ran(_)));
        assert_eq!(runs(&base), 2);
    }

    #[test]
    fn gpu_jobs_never_overlap() {
        let b = Arc::new(board("board-gpu"));
        let base = temp_dir("base-gpu");
        let started = Instant::now();
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let b = b.clone();
                let mut j = job(&base, &format!("gpu-{i}"), "sleep 0.5");
                j.gpu = true;
                thread::spawn(move || b.run(&j).unwrap())
            })
            .collect();
        for h in handles {
            assert!(h.join().unwrap().passed());
        }
        assert!(started.elapsed() >= Duration::from_millis(1000));
    }

    fn hold_job(base: &std::path::Path, resource: &str, script: &str) -> Job {
        let mut j = job(base, "boot-machine-1", script);
        j.resources = vec![resource.into()];
        j
    }

    #[test]
    fn holds_on_the_same_key_never_overlap_and_are_never_reused() {
        let b = Arc::new(board("hold-same"));
        let base = temp_dir("base-hold-same");
        let started = Instant::now();
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let (b, j) = (b.clone(), hold_job(&base, "machine-1", "sleep 0.5"));
                thread::spawn(move || b.hold(&j).unwrap())
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().exit_code, 0);
        }
        assert!(started.elapsed() >= Duration::from_millis(1000));
        // Identical commands both ran: hardware actions are not stamped.
        assert_eq!(runs(&base), 2);
    }

    #[test]
    fn different_keys_run_in_parallel() {
        let b = Arc::new(board("hold-diff"));
        let base = temp_dir("base-hold-diff");
        let started = Instant::now();
        let handles: Vec<_> = ["machine-1", "machine-2"]
            .into_iter()
            .map(|r| {
                let (b, j) = (b.clone(), hold_job(&base, r, "sleep 0.6"));
                thread::spawn(move || b.hold(&j).unwrap())
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(started.elapsed() < Duration::from_millis(1100));
    }

    #[test]
    fn holder_is_visible_during_the_run_and_the_run_is_audited() {
        let b = board("hold-audit");
        let base = temp_dir("base-hold-audit");
        let holder = b.root.join("slots/machine-1.holder");
        let rec = b
            .hold(&hold_job(
                &base,
                "machine-1",
                &format!("cat {}", holder.display()),
            ))
            .unwrap();
        assert_eq!(rec.exit_code, 0);
        let log = fs::read_to_string(b.root.join("logs").join(format!("{}.log", rec.key))).unwrap();
        assert!(
            log.contains("agent=tester") && log.contains("job=boot-machine-1"),
            "{log}"
        );
        assert!(!holder.exists());
        assert!(b.holders().is_empty());

        let ledger_text = fs::read_to_string(b.root.join(ledger::LEDGER_FILE)).unwrap();
        let event: ledger::LedgerEvent =
            serde_json::from_str(ledger_text.lines().last().unwrap()).unwrap();
        assert_eq!(event.action, "audit");
        assert!(event.intent.contains("resources=machine-1"));
        assert_eq!(
            ledger::verify(&b.root.join(ledger::LEDGER_FILE)).unwrap().0,
            1
        );
    }

    #[test]
    fn run_honors_named_keys() {
        let b = Arc::new(board("run-keys"));
        let base = temp_dir("base-run-keys");
        let started = Instant::now();
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let mut j = job(&base, &format!("job-{i}"), "sleep 0.5");
                j.resources = vec!["machine-1".into()];
                let b = b.clone();
                thread::spawn(move || b.run(&j).unwrap())
            })
            .collect();
        for h in handles {
            assert!(h.join().unwrap().passed());
        }
        assert!(started.elapsed() >= Duration::from_millis(1000));
    }

    #[test]
    fn rejects_unsafe_key_names_and_holds_without_a_key() {
        let b = board("hold-names");
        let base = temp_dir("base-hold-names");
        for bad in ["", "../etc", "Machine1", "a b", &"x".repeat(65)] {
            assert!(!valid_resource(bad), "{bad:?}");
            assert!(b.hold(&hold_job(&base, bad, "true")).is_err());
        }
        assert!(valid_resource("machine-1"));
        assert!(b.hold(&job(&base, "no-key", "true")).is_err());
        assert_eq!(runs(&base), 0);
    }
}
