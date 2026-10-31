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

//! File checkpoints and the agent-write ledger (sc#667).
//!
//! One shared content-addressed store under `~/basecamp/checkpoints`:
//!
//! - `objects/ab/cdef...`: file contents named by their SHA-256, shared by
//!   every session (the same bytes are stored once).
//! - `snapshots/<id>.json`: one snapshot per directory per turn, taken just
//!   before the agent's first write into that directory in that turn.
//! - `ledger/<session>.jsonl`: every write the agent's file tools made, with
//!   the content hash before and after, hash-chained line to line.
//!
//! A rollback restores only paths in the ledger, and only where the file
//! still holds what the agent left. A file the operator edited after the
//! agent is a conflict and is kept; files the operator edited that the agent
//! never wrote are reported as spared and never touched.
//!
//! The pure part (hashing, the chain, the rollback planner) takes values and
//! returns values, so it can move to Omega (ADR 0024). The store is the edge.
//!
//! Prior art: Hermes Agent, Nous Research, MIT licence, commit 5480918
//! (read 2026-10-18), `tools/checkpoint_manager.py`. Hermes keeps a shared
//! bare git store; this is our own content-addressed store with the same
//! shape (shared objects, per-directory per-turn snapshots, agent ledger).
//! Mechanism re-implemented from reading; no code copied.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub type Hash = String;

/// Directory snapshot bounds: files bigger than this, or past this count,
/// are listed as skipped. The written file's own prior content is always
/// kept, whatever its size.
pub const SNAPSHOT_MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
pub const SNAPSHOT_MAX_FILES: usize = 512;

const GENESIS: &str = "genesis";

// ---------------------------------------------------------------------------
// Pure part
// ---------------------------------------------------------------------------

pub fn hash_bytes(bytes: &[u8]) -> Hash {
    hex::encode(Sha256::digest(bytes))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub session: String,
    pub turn: u64,
    pub dir: String,
    /// File name (not path) to content hash, regular files only.
    pub files: BTreeMap<String, Hash>,
    pub skipped: Vec<String>,
}

impl Snapshot {
    /// Content address of the snapshot itself.
    pub fn id(&self) -> Hash {
        hash_bytes(&serde_json::to_vec(self).unwrap_or_default())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub seq: u64,
    pub turn: u64,
    pub tool: String,
    pub path: String,
    /// `None` = the file did not exist.
    pub before: Option<Hash>,
    pub after: Option<Hash>,
    /// Snapshot of the directory for this turn.
    pub snapshot: Hash,
    /// Hash of the previous ledger line, `genesis` for the first.
    pub prev: Hash,
}

/// Serialize an entry as one ledger line (without the newline) and return
/// the hash the next entry must carry as `prev`.
pub fn ledger_line(entry: &LedgerEntry) -> (String, Hash) {
    let line = serde_json::to_string(entry).unwrap_or_default();
    let next = hash_bytes(line.as_bytes());
    (line, next)
}

/// Parse and verify a ledger file. A torn last line (crash mid-append) is
/// dropped; a broken chain anywhere else is refused, so a rollback never runs
/// on a ledger someone edited. Returns the entries and the next `prev`.
pub fn read_ledger(text: &str) -> Result<(Vec<LedgerEntry>, Hash), String> {
    let mut entries = Vec::new();
    let mut prev = GENESIS.to_string();
    let lines: Vec<&str> = text.split('\n').collect();
    // `text` ends with '\n' when the last append finished, so the last piece
    // is empty; anything else there is a torn line.
    let complete = lines.len().saturating_sub(1);
    for (i, line) in lines.iter().take(complete).enumerate() {
        let entry: LedgerEntry = serde_json::from_str(line)
            .map_err(|e| format!("ledger line {} is damaged: {}", i + 1, e))?;
        if entry.prev != prev {
            return Err(format!("ledger chain broken at line {}", i + 1));
        }
        if entry.seq != i as u64 {
            return Err(format!("ledger sequence broken at line {}", i + 1));
        }
        prev = hash_bytes(line.as_bytes());
        entries.push(entry);
    }
    Ok((entries, prev))
}

#[derive(Debug, Clone, PartialEq)]
pub enum RollbackStep {
    /// Put the agent's starting content back (`None` = remove the file the
    /// agent created).
    Restore { path: String, to: Option<Hash> },
    /// The file already holds its starting content.
    AlreadyThere { path: String },
    /// Someone changed the file after the agent; it is kept as it is.
    Conflict {
        path: String,
        agent_left: Option<Hash>,
        found: Option<Hash>,
    },
}

/// Plan the rollback of every agent write in turns `>= from_turn`.
/// `current` holds the present hash of every ledger path (`None` = absent).
/// A path missing from `current` is treated as absent.
pub fn plan_rollback(
    entries: &[LedgerEntry],
    from_turn: u64,
    current: &BTreeMap<String, Option<Hash>>,
) -> Vec<RollbackStep> {
    let mut by_path: BTreeMap<&str, Vec<&LedgerEntry>> = BTreeMap::new();
    for e in entries.iter().filter(|e| e.turn >= from_turn) {
        by_path.entry(e.path.as_str()).or_default().push(e);
    }
    let mut steps = Vec::new();
    for (path, mut writes) in by_path {
        writes.sort_by_key(|e| e.seq);
        let start = writes[0].before.clone();
        let left = writes[writes.len() - 1].after.clone();
        let found = current.get(path).cloned().flatten();
        let path = path.to_string();
        if found == start {
            steps.push(RollbackStep::AlreadyThere { path });
        } else if found == left {
            steps.push(RollbackStep::Restore { path, to: start });
        } else {
            steps.push(RollbackStep::Conflict {
                path,
                agent_left: left,
                found,
            });
        }
    }
    steps
}

/// Files in the snapshotted directories that changed since the snapshot and
/// that the agent never wrote: the operator's edits, left alone. `current`
/// maps full paths in those directories to their present hash.
pub fn spared_operator_edits(
    snapshots: &[Snapshot],
    agent_paths: &BTreeSet<String>,
    current: &BTreeMap<String, Option<Hash>>,
) -> Vec<String> {
    let mut spared = BTreeSet::new();
    for snap in snapshots {
        let dir = Path::new(&snap.dir);
        for (name, hash) in &snap.files {
            let full = dir.join(name).to_string_lossy().into_owned();
            if agent_paths.contains(&full) {
                continue;
            }
            if current.get(&full).cloned().flatten().as_ref() != Some(hash) {
                spared.insert(full);
            }
        }
        for full in current.keys() {
            let p = Path::new(full);
            if p.parent() == Some(dir)
                && !agent_paths.contains(full)
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    !snap.files.contains_key(n) && !snap.skipped.iter().any(|s| s == n)
                })
            {
                spared.insert(full.clone());
            }
        }
    }
    spared.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Edge: the on-disk store
// ---------------------------------------------------------------------------

pub fn store_root(home: &Path) -> PathBuf {
    home.join("basecamp/checkpoints")
}

#[derive(Debug, Clone)]
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn object_path(&self, hash: &str) -> PathBuf {
        let (a, b) = hash.split_at(2.min(hash.len()));
        self.root.join("objects").join(a).join(b)
    }

    pub fn put_blob(&self, bytes: &[u8]) -> Result<Hash, String> {
        let hash = hash_bytes(bytes);
        let path = self.object_path(&hash);
        if !path.exists() {
            write_atomic(&path, bytes)?;
        }
        Ok(hash)
    }

    /// Read a blob and check it still hashes to its name.
    pub fn get_blob(&self, hash: &str) -> Result<Vec<u8>, String> {
        let bytes = fs::read(self.object_path(hash))
            .map_err(|e| format!("checkpoint object {hash} unreadable: {e}"))?;
        if hash_bytes(&bytes) != hash {
            return Err(format!("checkpoint object {hash} is damaged"));
        }
        Ok(bytes)
    }

    fn ledger_path(&self, session: &str) -> PathBuf {
        self.root.join("ledger").join(format!("{session}.jsonl"))
    }

    pub fn load_ledger(&self, session: &str) -> Result<(Vec<LedgerEntry>, Hash), String> {
        match fs::read_to_string(self.ledger_path(session)) {
            Ok(text) => read_ledger(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok((Vec::new(), GENESIS.into())),
            Err(e) => Err(format!("ledger unreadable: {e}")),
        }
    }

    fn append_ledger(&self, session: &str, line: &str) -> Result<(), String> {
        let path = self.ledger_path(session);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("ledger open failed: {e}"))?;
        f.write_all(format!("{line}\n").as_bytes())
            .and_then(|_| f.sync_all())
            .map_err(|e| format!("ledger append failed: {e}"))
    }

    fn snapshot_path(&self, id: &str) -> PathBuf {
        self.root.join("snapshots").join(format!("{id}.json"))
    }

    pub fn load_snapshot(&self, id: &str) -> Result<Snapshot, String> {
        let bytes = fs::read(self.snapshot_path(id)).map_err(|e| e.to_string())?;
        let snap: Snapshot = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if snap.id() != id {
            return Err(format!("snapshot {id} is damaged"));
        }
        Ok(snap)
    }

    /// Snapshot the regular files directly inside `dir`.
    ///
    /// The directory's final component is opened `O_NOFOLLOW`, and each child
    /// is read with `openat(O_NOFOLLOW)`. A symlink where the directory name
    /// should be refuses the snapshot. A symlink child is omitted, not followed.
    fn take_snapshot(&self, session: &str, turn: u64, dir: &Path) -> Result<Hash, String> {
        let mut files = BTreeMap::new();
        let mut skipped = Vec::new();
        match open_snapshot_dir(dir) {
            Ok(dirfd) => {
                let mut names = snapshot_entries(dirfd.as_raw_fd()).unwrap_or_default();
                names.sort();
                for (name, size) in names {
                    if files.len() >= SNAPSHOT_MAX_FILES || size > SNAPSHOT_MAX_FILE_BYTES {
                        skipped.push(name);
                        continue;
                    }
                    match read_snapshot_file(dirfd.as_raw_fd(), &name) {
                        Ok(Some(bytes)) if bytes.len() as u64 <= SNAPSHOT_MAX_FILE_BYTES => {
                            files.insert(name, self.put_blob(&bytes)?);
                        }
                        _ => skipped.push(name),
                    }
                }
            }
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
                return Err(format!(
                    "cannot snapshot {}: directory entry is a symlink",
                    dir.display()
                ));
            }
            Err(_) => {}
        }
        let snap = Snapshot {
            session: session.to_string(),
            turn,
            dir: dir.to_string_lossy().into_owned(),
            files,
            skipped,
        };
        let id = snap.id();
        let path = self.snapshot_path(&id);
        if !path.exists() {
            write_atomic(&path, &serde_json::to_vec(&snap).unwrap_or_default())?;
        }
        Ok(id)
    }

    /// Plan and apply the rollback of `session`'s agent writes in turns
    /// `>= from_turn`.
    pub fn rollback(&self, session: &str, from_turn: u64) -> Result<RollbackReport, String> {
        let (entries, _) = self.load_ledger(session)?;
        let agent_paths: BTreeSet<String> = entries
            .iter()
            .filter(|e| e.turn >= from_turn)
            .map(|e| e.path.clone())
            .collect();
        let mut current = BTreeMap::new();
        for p in &agent_paths {
            current.insert(p.clone(), current_hash(Path::new(p)));
        }
        let steps = plan_rollback(&entries, from_turn, &current);

        // Spared files are worked out before anything is restored.
        let mut snap_ids = BTreeSet::new();
        for e in entries.iter().filter(|e| e.turn >= from_turn) {
            snap_ids.insert(e.snapshot.clone());
        }
        let mut snapshots = Vec::new();
        for id in &snap_ids {
            snapshots.push(self.load_snapshot(id)?);
        }
        let mut dir_state = BTreeMap::new();
        for snap in &snapshots {
            if let Ok(rd) = fs::read_dir(&snap.dir) {
                for e in rd.flatten() {
                    if e.file_type().map(|t| t.is_file()).unwrap_or(false) {
                        let p = e.path();
                        dir_state.insert(p.to_string_lossy().into_owned(), current_hash(&p));
                    }
                }
            }
        }
        let spared = spared_operator_edits(&snapshots, &agent_paths, &dir_state);

        let mut report = RollbackReport {
            from_turn,
            spared,
            ..Default::default()
        };
        for step in steps {
            match step {
                RollbackStep::Restore { path, to: Some(h) } => {
                    let bytes = self.get_blob(&h)?;
                    write_atomic(Path::new(&path), &bytes)?;
                    report.restored.push(path);
                }
                RollbackStep::Restore { path, to: None } => {
                    fs::remove_file(&path).map_err(|e| format!("remove {path}: {e}"))?;
                    report.removed.push(path);
                }
                RollbackStep::AlreadyThere { path } => report.unchanged.push(path),
                RollbackStep::Conflict { path, .. } => report.conflicts.push(path),
            }
        }
        Ok(report)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RollbackReport {
    pub from_turn: u64,
    /// Agent-written files put back to their starting content.
    pub restored: Vec<String>,
    /// Files the agent created, removed again.
    pub removed: Vec<String>,
    /// Agent-written files already at their starting content.
    pub unchanged: Vec<String>,
    /// Agent-written files someone changed afterwards: kept as found.
    pub conflicts: Vec<String>,
    /// Files in the same directories the operator changed: never touched.
    pub spared: Vec<String>,
}

fn current_hash(path: &Path) -> Option<Hash> {
    fs::read(path).ok().map(|b| hash_bytes(&b))
}

fn open_snapshot_dir(dir: &Path) -> std::io::Result<OwnedFd> {
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))?;
    // SAFETY: `c` is a valid C string. A non-negative fd is owned here.
    // `O_NOFOLLOW` refuses a symlink in the final component.
    let fd = unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Regular files directly inside `dirfd`, as `(name, size)`. Symlinks are omitted.
fn snapshot_entries(dirfd: libc::c_int) -> std::io::Result<Vec<(String, u64)>> {
    let raw = dup_snapshot_fd(dirfd)?.into_raw_fd();
    // SAFETY: `raw` is a live fd. fdopendir owns it on success.
    let dir = unsafe { libc::fdopendir(raw) };
    if dir.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(raw) };
        return Err(err);
    }
    let mut out = Vec::new();
    let walked = loop {
        unsafe { *libc::__errno_location() = 0 };
        let ent = unsafe { libc::readdir(dir) };
        if ent.is_null() {
            let errno = unsafe { *libc::__errno_location() };
            break if errno == 0 {
                Ok(())
            } else {
                Err(std::io::Error::from_raw_os_error(errno))
            };
        }
        let name_c = unsafe { std::ffi::CStr::from_ptr((*ent).d_name.as_ptr()) };
        let name = String::from_utf8_lossy(name_c.to_bytes()).into_owned();
        if name == "." || name == ".." {
            continue;
        }
        let Ok(c_name) = std::ffi::CString::new(name.as_str()) else {
            continue;
        };
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `dirfd` is the caller's directory. `c_name` is one component.
        let rc = unsafe {
            libc::fstatat(
                dirfd,
                c_name.as_ptr(),
                st.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc != 0 {
            continue;
        }
        let st = unsafe { st.assume_init() };
        if st.st_mode as u32 & libc::S_IFMT != libc::S_IFREG {
            continue;
        }
        let size = if st.st_size < 0 { 0 } else { st.st_size as u64 };
        out.push((name, size));
    };
    unsafe { libc::closedir(dir) };
    walked?;
    Ok(out)
}

fn dup_snapshot_fd(fd: libc::c_int) -> std::io::Result<OwnedFd> {
    let n = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(n) })
}

fn read_snapshot_file(dirfd: libc::c_int, name: &str) -> std::io::Result<Option<Vec<u8>>> {
    let c = std::ffi::CString::new(name)
        .map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))?;
    // SAFETY: `dirfd` is a directory and `c` is a single component.
    let fd = unsafe {
        libc::openat(
            dirfd,
            c.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
        )
    };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ELOOP) {
            return Ok(None);
        }
        return Err(err);
    }
    let mut file = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
    if !file.metadata()?.file_type().is_file() {
        return Ok(None);
    }
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = file.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        if buf.len() as u64 + n as u64 > SNAPSHOT_MAX_FILE_BYTES {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    Ok(Some(buf))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    let tmp = parent.join(format!(
        ".{}.aien-tmp-{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("f"),
        std::process::id()
    ));
    let mut f = fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    f.write_all(bytes)
        .and_then(|_| f.sync_all())
        .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("rename onto {}: {e}", path.display()))
}

/// One chat session's view of the store: turn counter, which directories
/// are already snapshotted this turn, and the ledger chain head.
#[derive(Debug)]
pub struct Session {
    pub store: CheckpointStore,
    pub id: String,
    pub turn: u64,
    /// Record only writes under this directory (`None` = every write).
    pub scope: Option<PathBuf>,
    snapshotted: HashSet<(u64, PathBuf)>,
    turn_snapshots: BTreeMap<PathBuf, Hash>,
    next_seq: u64,
    prev: Hash,
}

/// What `before_write` captured; handed back to `after_write`.
#[derive(Debug, Clone)]
pub struct Pending {
    path: PathBuf,
    before: Option<Hash>,
    snapshot: Hash,
}

impl Session {
    pub fn open(store: CheckpointStore, id: &str) -> Result<Self, String> {
        let (entries, prev) = store.load_ledger(id)?;
        let turn = entries.last().map(|e| e.turn).unwrap_or(0);
        Ok(Self {
            store,
            id: id.to_string(),
            turn,
            scope: None,
            snapshotted: HashSet::new(),
            turn_snapshots: BTreeMap::new(),
            next_seq: entries.len() as u64,
            prev,
        })
    }

    pub fn begin_turn(&mut self) -> u64 {
        self.turn += 1;
        self.turn_snapshots.clear();
        self.turn
    }

    /// Call before an agent write to `path` (absolute). Snapshots the
    /// directory once this turn and keeps the file's prior content.
    ///
    /// Reads `path` by name. Session tests use that. Tool dispatch passes
    /// bytes from the held descriptor (`before_write_with_bytes`).
    #[allow(dead_code)]
    pub fn before_write(&mut self, path: &Path) -> Result<Pending, String> {
        let bytes = match fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(format!(
                    "cannot read {} before writing: {e}",
                    path.display()
                ))
            }
        };
        self.before_write_with_bytes(path, bytes)
    }

    fn before_write_with_bytes(
        &mut self,
        path: &Path,
        bytes: Option<Vec<u8>>,
    ) -> Result<Pending, String> {
        let dir = path.parent().unwrap_or(Path::new("/")).to_path_buf();
        let key = (self.turn, dir.clone());
        if !self.snapshotted.contains(&key) {
            let id = self.store.take_snapshot(&self.id, self.turn, &dir)?;
            self.snapshotted.insert(key);
            self.turn_snapshots.insert(dir.clone(), id);
        }
        let snapshot = self.turn_snapshots.get(&dir).cloned().unwrap_or_default();
        let before = match bytes {
            Some(b) => Some(self.store.put_blob(&b)?),
            None => None,
        };
        Ok(Pending {
            path: path.to_path_buf(),
            before,
            snapshot,
        })
    }

    /// Call after the write; appends one ledger line.
    ///
    /// Reads the path by name. Tool dispatch passes bytes from the held
    /// descriptor (`after_write_with_bytes`).
    #[allow(dead_code)]
    pub fn after_write(&mut self, pending: Pending, tool: &str) -> Result<(), String> {
        let after_bytes = match fs::read(&pending.path) {
            Ok(bytes) => Some(bytes),
            Err(_) => None,
        };
        self.after_write_with_bytes(pending, tool, after_bytes)
    }

    fn after_write_with_bytes(
        &mut self,
        pending: Pending,
        tool: &str,
        bytes: Option<Vec<u8>>,
    ) -> Result<(), String> {
        let after = match bytes {
            Some(b) => Some(self.store.put_blob(&b)?),
            None => None,
        };
        if after == pending.before {
            return Ok(());
        }
        let entry = LedgerEntry {
            seq: self.next_seq,
            turn: self.turn,
            tool: tool.to_string(),
            path: pending.path.to_string_lossy().into_owned(),
            before: pending.before,
            after,
            snapshot: pending.snapshot,
            prev: self.prev.clone(),
        };
        let (line, next) = ledger_line(&entry);
        self.store.append_ledger(&self.id, &line)?;
        self.next_seq += 1;
        self.prev = next;
        Ok(())
    }

    /// Turns with agent writes, newest last, with their file counts.
    pub fn turns(&self) -> Result<Vec<(u64, usize)>, String> {
        let (entries, _) = self.store.load_ledger(&self.id)?;
        let mut by_turn: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
        for e in &entries {
            by_turn.entry(e.turn).or_default().insert(e.path.as_str());
        }
        Ok(by_turn.into_iter().map(|(t, p)| (t, p.len())).collect())
    }
}

// ---------------------------------------------------------------------------
// Edge: the process's active session (set by the chat loop)
// ---------------------------------------------------------------------------

fn active() -> &'static Mutex<Option<Session>> {
    static ACTIVE: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(None))
}

/// Start recording agent writes for this session. Without this call (tests,
/// other entry points) the file tools record nothing.
pub fn start_session(home: &Path, session_id: &str) -> Result<(), String> {
    let session = Session::open(CheckpointStore::new(store_root(home)), session_id)?;
    *active().lock().map_err(|_| "checkpoint lock poisoned")? = Some(session);
    Ok(())
}

/// Stop recording (tests and shutdown).
#[allow(dead_code)]
pub fn stop_session() {
    if let Ok(mut guard) = active().lock() {
        *guard = None;
    }
}

/// Start a session that records only writes under `scope`.
#[allow(dead_code)]
pub fn start_scoped_session(home: &Path, session_id: &str, scope: PathBuf) -> Result<(), String> {
    let mut session = Session::open(CheckpointStore::new(store_root(home)), session_id)?;
    session.scope = Some(scope);
    *active().lock().map_err(|_| "checkpoint lock poisoned")? = Some(session);
    Ok(())
}

pub fn begin_turn() {
    if let Ok(mut guard) = active().lock() {
        if let Some(s) = guard.as_mut() {
            s.begin_turn();
        }
    }
}

pub fn active_ledger_name() -> Option<String> {
    active().lock().ok()?.as_ref().map(|s| s.id.clone())
}

/// Like [`record_agent_write`], but the target's before and after bytes come
/// from `read_target` (a descriptor the caller is holding) instead of a fresh
/// pathname read.
pub fn record_agent_write_with_reader<T>(
    path: &Path,
    tool: &str,
    mut read_target: impl FnMut() -> Result<Option<Vec<u8>>, String>,
    write: impl FnOnce() -> T,
) -> Result<(T, Option<String>), String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "checkpoint lock poisoned".to_string())?;
    let Some(session) = guard
        .as_mut()
        .filter(|s| s.scope.as_ref().is_none_or(|d| path.starts_with(d)))
    else {
        return Ok((write(), None));
    };
    let before_bytes = read_target().map_err(|e| format!("CHECKPOINT REFUSAL: {e}"))?;
    let pending = session
        .before_write_with_bytes(path, before_bytes)
        .map_err(|e| format!("CHECKPOINT REFUSAL: {e}"))?;
    let out = write();
    let warn = match read_target() {
        Ok(bytes) => session.after_write_with_bytes(pending, tool, bytes).err(),
        Err(e) => Some(e),
    };
    Ok((out, warn))
}

/// Run `write` as a recorded agent write of `path`. If the prior content
/// cannot be kept the write is refused, because it could not be rolled back.
/// If the ledger append fails after the write, the error is returned beside
/// the result so the caller can show it.
///
/// This reads the target by pathname. Tool dispatch uses
/// [`record_agent_write_with_reader`] so the bytes are the descriptor that
/// was authorized.
#[allow(dead_code)]
pub fn record_agent_write<T>(
    path: &Path,
    tool: &str,
    write: impl FnOnce() -> T,
) -> Result<(T, Option<String>), String> {
    let mut guard = active()
        .lock()
        .map_err(|_| "checkpoint lock poisoned".to_string())?;
    let Some(session) = guard
        .as_mut()
        .filter(|s| s.scope.as_ref().is_none_or(|d| path.starts_with(d)))
    else {
        return Ok((write(), None));
    };
    let pending = session
        .before_write(path)
        .map_err(|e| format!("CHECKPOINT REFUSAL: {e}"))?;
    let out = write();
    let warn = session.after_write(pending, tool).err();
    Ok((out, warn))
}

/// The session `session` names, or else the active one.
fn session_or_active(session: Option<&str>) -> Result<String, String> {
    match session {
        Some(s) => Ok(s.to_string()),
        None => active_ledger_name()
            .ok_or_else(|| "no active session; name one: /checkpoints <session-id>".to_string()),
    }
}

/// Turns (turn number, file writes) of an active session or of `session`.
/// The session id is deliberately not in the result: callers that print the
/// turns do not print the id (sc#689, cleartext logging of a session id).
pub fn list_turns(home: &Path, session: Option<&str>) -> Result<Vec<(u64, usize)>, String> {
    let id = session_or_active(session)?;
    let s = Session::open(CheckpointStore::new(store_root(home)), &id)?;
    s.turns()
}

/// Roll back `session` (default: the active one) from `turn` (default: its
/// last turn with agent writes). Returns the report only; see `list_turns`.
pub fn rollback(
    home: &Path,
    turn: Option<u64>,
    session: Option<&str>,
) -> Result<RollbackReport, String> {
    let id = session_or_active(session)?;
    let turns = list_turns(home, Some(id.as_str()))?;
    let Some(last) = turns.last().map(|(t, _)| *t) else {
        return Err(format!("session {id} has no agent writes to roll back"));
    };
    let from = turn.unwrap_or(last);
    let store = CheckpointStore::new(store_root(home));
    // Hold the active session lock so no agent write lands mid-rollback.
    let _guard = active().lock();
    store.rollback(&id, from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        seq: u64,
        turn: u64,
        path: &str,
        before: Option<&str>,
        after: Option<&str>,
    ) -> LedgerEntry {
        LedgerEntry {
            seq,
            turn,
            tool: "write_to_file".into(),
            path: path.into(),
            before: before.map(String::from),
            after: after.map(String::from),
            snapshot: "s".into(),
            prev: String::new(),
        }
    }

    #[test]
    fn planner_restores_only_what_still_holds_the_agents_bytes() {
        let entries = vec![
            entry(0, 1, "/w/a", Some("a0"), Some("a1")),
            entry(1, 2, "/w/a", Some("a1"), Some("a2")),
            entry(2, 2, "/w/new", None, Some("n1")),
            entry(3, 2, "/w/b", Some("b0"), Some("b1")),
            entry(4, 2, "/w/c", Some("c0"), Some("c1")),
        ];
        let current: BTreeMap<String, Option<Hash>> = [
            ("/w/a", Some("a2")),
            ("/w/new", Some("n1")),
            ("/w/b", Some("operator")), // operator edited after the agent
            ("/w/c", Some("c0")),       // already back
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.map(String::from)))
        .collect();
        let steps = plan_rollback(&entries, 2, &current);
        assert_eq!(
            steps,
            vec![
                RollbackStep::Restore {
                    path: "/w/a".into(),
                    to: Some("a1".into())
                },
                RollbackStep::Conflict {
                    path: "/w/b".into(),
                    agent_left: Some("b1".into()),
                    found: Some("operator".into())
                },
                RollbackStep::AlreadyThere {
                    path: "/w/c".into()
                },
                RollbackStep::Restore {
                    path: "/w/new".into(),
                    to: None
                },
            ]
        );
        // From turn 1, /w/a goes back to its turn-1 start.
        let steps = plan_rollback(&entries, 1, &current);
        assert_eq!(
            steps[0],
            RollbackStep::Restore {
                path: "/w/a".into(),
                to: Some("a0".into())
            }
        );
    }

    #[test]
    fn ledger_chain_detects_edits_and_drops_a_torn_tail() {
        let mut text = String::new();
        let mut prev = GENESIS.to_string();
        for seq in 0..3 {
            let mut e = entry(seq, 1, "/w/a", Some("x"), Some("y"));
            e.prev = prev.clone();
            let (line, next) = ledger_line(&e);
            text.push_str(&line);
            text.push('\n');
            prev = next;
        }
        let (entries, head) = read_ledger(&text).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(head, prev);

        let torn = format!("{text}{{\"seq\":3,\"tu");
        assert_eq!(read_ledger(&torn).unwrap().0.len(), 3);

        let edited = text.replacen("\"/w/a\"", "\"/w/z\"", 1);
        assert!(read_ledger(&edited).is_err());
        let dropped: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
        assert!(read_ledger(&dropped).is_err());
    }

    /// Acceptance: after mixed operator and agent edits, a rollback restores
    /// only the agent's files.
    #[test]
    fn rollback_after_mixed_edits_restores_only_the_agents_files() {
        let tmp = tempfile::tempdir().unwrap();
        let work = tmp.path().join("work");
        fs::create_dir_all(&work).unwrap();
        let store = CheckpointStore::new(tmp.path().join("checkpoints"));
        let mut s = Session::open(store.clone(), "s1").unwrap();

        let agent_file = work.join("agent.rs");
        let shared = work.join("shared.rs");
        let operator_file = work.join("operator.md");
        let created = work.join("created.txt");
        fs::write(&agent_file, "agent v0").unwrap();
        fs::write(&shared, "shared v0").unwrap();
        fs::write(&operator_file, "operator v0").unwrap();

        // Turn 1: the agent writes two files and creates one.
        s.begin_turn();
        for (p, text) in [
            (&agent_file, "agent v1"),
            (&shared, "shared v1"),
            (&created, "new"),
        ] {
            let pending = s.before_write(p).unwrap();
            fs::write(p, text).unwrap();
            s.after_write(pending, "write_to_file").unwrap();
        }
        // The operator edits a file the agent never touched, edits a file the
        // agent wrote, and adds a new file of their own.
        fs::write(&operator_file, "operator v1").unwrap();
        fs::write(&shared, "shared operator fix").unwrap();
        let operator_new = work.join("operator-new.md");
        fs::write(&operator_new, "mine").unwrap();

        let report = store.rollback("s1", 1).unwrap();
        assert_eq!(fs::read_to_string(&agent_file).unwrap(), "agent v0");
        assert!(!created.exists());
        assert_eq!(fs::read_to_string(&operator_file).unwrap(), "operator v1");
        assert_eq!(fs::read_to_string(&shared).unwrap(), "shared operator fix");
        assert_eq!(fs::read_to_string(&operator_new).unwrap(), "mine");
        let s = |p: &Path| p.to_string_lossy().into_owned();
        assert_eq!(report.restored, vec![s(&agent_file)]);
        assert_eq!(report.removed, vec![s(&created)]);
        assert_eq!(report.conflicts, vec![s(&shared)]);
        assert_eq!(report.spared, vec![s(&operator_new), s(&operator_file)]);

        // A second rollback changes nothing.
        let again = store.rollback("s1", 1).unwrap();
        assert!(again.restored.is_empty() && again.removed.is_empty());
        assert_eq!(fs::read_to_string(&operator_file).unwrap(), "operator v1");
    }

    /// Negative control: restoring the whole directory snapshot (what a
    /// plain checkpoint does) would destroy the operator's edit; the ledger
    /// is what prevents it.
    #[test]
    fn negative_control_snapshot_alone_would_revert_operator_edits() {
        let tmp = tempfile::tempdir().unwrap();
        let work = tmp.path().join("work");
        fs::create_dir_all(&work).unwrap();
        let store = CheckpointStore::new(tmp.path().join("checkpoints"));
        let mut s = Session::open(store.clone(), "s2").unwrap();
        let agent_file = work.join("a.txt");
        let operator_file = work.join("o.txt");
        fs::write(&agent_file, "a0").unwrap();
        fs::write(&operator_file, "o0").unwrap();
        s.begin_turn();
        let pending = s.before_write(&agent_file).unwrap();
        fs::write(&agent_file, "a1").unwrap();
        s.after_write(pending, "write_to_file").unwrap();
        fs::write(&operator_file, "o1").unwrap();

        let (entries, _) = store.load_ledger("s2").unwrap();
        let snap = store.load_snapshot(&entries[0].snapshot).unwrap();
        // The snapshot holds the operator file's old content ...
        assert_eq!(snap.files.get("o.txt"), Some(&hash_bytes(b"o0")));
        // ... but the ledger never names it, so rollback leaves it.
        assert!(entries
            .iter()
            .all(|e| e.path != operator_file.to_string_lossy()));
        store.rollback("s2", 1).unwrap();
        assert_eq!(fs::read_to_string(&operator_file).unwrap(), "o1");
        assert_eq!(fs::read_to_string(&agent_file).unwrap(), "a0");
    }

    #[test]
    fn one_snapshot_per_directory_per_turn_and_shared_objects() {
        let tmp = tempfile::tempdir().unwrap();
        let work = tmp.path().join("work");
        fs::create_dir_all(&work).unwrap();
        let store = CheckpointStore::new(tmp.path().join("checkpoints"));
        let mut s = Session::open(store.clone(), "s3").unwrap();
        s.begin_turn();
        for i in 0..3 {
            let p = work.join(format!("f{i}"));
            let pending = s.before_write(&p).unwrap();
            fs::write(&p, "same bytes").unwrap();
            s.after_write(pending, "write_to_file").unwrap();
        }
        let snaps = fs::read_dir(tmp.path().join("checkpoints/snapshots"))
            .unwrap()
            .count();
        assert_eq!(snaps, 1);
        s.begin_turn();
        let p = work.join("f0");
        let pending = s.before_write(&p).unwrap();
        fs::write(&p, "turn two").unwrap();
        s.after_write(pending, "write_to_file").unwrap();
        let snaps = fs::read_dir(tmp.path().join("checkpoints/snapshots"))
            .unwrap()
            .count();
        assert_eq!(snaps, 2);
        // "same bytes" is stored once.
        assert!(store.get_blob(&hash_bytes(b"same bytes")).is_ok());

        // A reopened session continues the chain and the turn count.
        let reopened = Session::open(store.clone(), "s3").unwrap();
        assert_eq!(reopened.turn, 2);
        assert_eq!(reopened.turns().unwrap(), vec![(1, 3), (2, 1)]);
    }

    #[test]
    fn a_damaged_object_stops_the_restore() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(tmp.path().join("checkpoints"));
        let h = store.put_blob(b"good").unwrap();
        fs::write(store.object_path(&h), b"evil").unwrap();
        assert!(store.get_blob(&h).is_err());
    }
}
