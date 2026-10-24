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

//! Append-only JSON-lines journal (Headstart M2, issue #535).
//!
//! Why this and not an existing store: the repo's durable stores are the
//! aien-proof receipt store (immutable receipts, content addressed) and the
//! M1 consumed records (one atomically replaced file per task). Neither fits
//! an ordered event history, so the scheduler keeps its own journal. It
//! persists only scheduler events; proofs stay in the harness store and are
//! re-verified, never copied into the journal as authority.
//!
//! Format: line 1 is a header (format name, version, scheduler config); every
//! further line is one JSON event. Each append is a single `write_all` of
//! `json + "\n"` on an `O_APPEND` file followed by `sync_all`.
//!
//! Crash handling:
//! - A final segment with no trailing newline is a torn write. It is ignored
//!   and truncated away on open (then fsynced), so the next append starts on
//!   a clean line. Truncation happens only after the header and every
//!   complete line validated; a refused journal is not modified.
//! - A newline-terminated line that does not parse is corruption, not a torn
//!   write (a torn write is always a prefix of one line), and open fails
//!   closed.
//! - If an append fails, the file is truncated back to its previous length.
//!   If that also fails the journal refuses every further append.
//! - One process at a time: the journal holds an exclusive `flock` on
//!   `<path>.lock` for its lifetime.

use aien_proof::lock::FileLock;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct Journal {
    file: File,
    path: PathBuf,
    len: u64,
    poisoned: bool,
    _lock: FileLock,
}

/// Result of opening a journal.
pub struct Opened<T> {
    pub journal: Journal,
    pub entries: Vec<T>,
    /// Bytes of a torn final line that were dropped (0 if none).
    pub torn_bytes: usize,
}

fn sync_dir(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => File::open(dir)?.sync_all(),
        _ => Ok(()),
    }
}

impl Journal {
    /// Open `path`, creating it with `header` when missing, and replay its
    /// entries. An existing journal must carry exactly `header`.
    pub fn open<T: DeserializeOwned>(path: &Path, header: &Value) -> Result<Opened<T>, String> {
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let lock = FileLock::try_acquire(&lock_path)
            .map_err(|e| format!("cannot lock journal: {e}"))?
            .ok_or_else(|| format!("journal {} is held by another process", path.display()))?;
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(format!("cannot read journal {}: {e}", path.display())),
        };
        let complete = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let torn_bytes = bytes.len() - complete;
        // Validate everything before touching the file: a journal that is
        // refused (wrong header, corrupt line) is left exactly as found.
        let text = std::str::from_utf8(&bytes[..complete])
            .map_err(|e| format!("journal {} is not UTF-8: {e}", path.display()))?;
        let mut lines = text.lines();
        let fresh = match lines.next() {
            None => true,
            Some(first) => {
                let found: Value = serde_json::from_str(first)
                    .map_err(|e| format!("journal header unreadable: {e}"))?;
                if &found != header {
                    return Err(format!(
                        "journal header mismatch: journal has {found}, expected {header}"
                    ));
                }
                false
            }
        };
        let mut entries = Vec::new();
        for (i, line) in lines.enumerate() {
            let entry = serde_json::from_str(line)
                .map_err(|e| format!("journal line {} is corrupt, refusing: {e}", i + 2))?;
            entries.push(entry);
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("cannot open journal {}: {e}", path.display()))?;
        if torn_bytes > 0 {
            file.set_len(complete as u64)
                .and_then(|_| file.sync_all())
                .map_err(|e| format!("cannot truncate torn journal tail: {e}"))?;
        }
        let mut journal = Journal {
            file,
            path: path.to_path_buf(),
            len: complete as u64,
            poisoned: false,
            _lock: lock,
        };
        if fresh {
            journal.append(header)?;
            sync_dir(path).map_err(|e| format!("cannot fsync journal dir: {e}"))?;
        }
        Ok(Opened {
            journal,
            entries,
            torn_bytes,
        })
    }

    /// Append one entry as a single line and fsync it.
    pub fn append<T: Serialize>(&mut self, entry: &T) -> Result<(), String> {
        if self.poisoned {
            return Err("journal is poisoned after a failed rollback".to_string());
        }
        let mut line = serde_json::to_vec(entry).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let written = self
            .file
            .write_all(&line)
            .and_then(|_| self.file.sync_all());
        match written {
            Ok(()) => {
                self.len += line.len() as u64;
                Ok(())
            }
            Err(e) => {
                if self
                    .file
                    .set_len(self.len)
                    .and_then(|_| self.file.sync_all())
                    .is_err()
                {
                    self.poisoned = true;
                }
                Err(format!(
                    "journal append to {} failed: {e}",
                    self.path.display()
                ))
            }
        }
    }
}
