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

//! The quality ledger: one append-only JSON line per real run, sealed with the
//! Crumb ledger event hash (RFC-0001 section 8.3, `CRUMB_LEDGER_EVENT_V1`), so
//! each line deserializes as a `crumb_spec::ledger::LedgerEvent` and editing any
//! past line breaks every hash after it.

use crate::lock::FileLock;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const LEDGER_FILE: &str = "ledger.jsonl";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub index: u64,
    pub timestamp: u64,
    pub agent: String,
    pub action: String,
    pub target: String,
    pub intent: String,
    pub payload_hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub hash: [u8; 32],
}

#[allow(clippy::too_many_arguments)]
pub fn compute_hash(
    index: u64,
    timestamp: u64,
    agent: &str,
    action: &str,
    target: &str,
    intent: &str,
    payload_hash: &[u8; 32],
    parent_hash: &[u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"CRUMB_LEDGER_EVENT_V1");
    h.update(&index.to_be_bytes());
    h.update(&timestamp.to_be_bytes());
    for s in [agent, action, target, intent] {
        h.update(&(s.len() as u32).to_be_bytes());
        h.update(s.as_bytes());
    }
    h.update(payload_hash);
    h.update(parent_hash);
    *h.finalize().as_bytes()
}

impl LedgerEvent {
    fn recompute(&self) -> [u8; 32] {
        compute_hash(
            self.index,
            self.timestamp,
            &self.agent,
            &self.action,
            &self.target,
            &self.intent,
            &self.payload_hash,
            &self.parent_hash,
        )
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Message for a ledger whose last line has no closing newline: an append was
/// cut off (for example by ENOSPC) and the bytes after `intact` are not a
/// sealed event. Callers fail closed on it instead of skipping the torn line.
fn torn_tail(path: &Path, len: u64, intact: u64) -> String {
    let newline = match intact {
        0 => "the file has no newline".to_string(),
        n => format!("last newline at byte offset {}", n - 1),
    };
    format!(
        "{}: torn tail: an append was cut off and left {} byte(s) without a closing newline \
         ({newline}); the chain is intact up to byte offset {intact}",
        path.display(),
        len - intact
    )
}

/// Length of the prefix of `file` (of length `len`) that ends with a newline,
/// scanning backwards from the end in fixed chunks. 0 when there is none.
fn intact_len(file: &mut File, len: u64) -> io::Result<u64> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut end = len;
    while end > 0 {
        let from = end.saturating_sub(buf.len() as u64);
        let chunk = &mut buf[..(end - from) as usize];
        file.seek(SeekFrom::Start(from))?;
        file.read_exact(chunk)?;
        if let Some(p) = chunk.iter().rposition(|&b| b == b'\n') {
            return Ok(from + p as u64 + 1);
        }
        end = from;
    }
    Ok(0)
}

/// Last event, read from the file tail so appends stay O(1) as the ledger grows.
/// A tail without a closing newline is a cut-off append and fails closed.
fn last_event(path: &Path) -> io::Result<Option<LedgerEvent>> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(None);
    }
    let start = len.saturating_sub(64 * 1024);
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)?;
    if tail.last() != Some(&b'\n') {
        let intact = intact_len(&mut file, len)?;
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            torn_tail(path, len, intact),
        ));
    }
    match tail
        .split(|&b| b == b'\n')
        .rev()
        .find(|l| !l.trim_ascii().is_empty())
    {
        Some(line) => Ok(Some(serde_json::from_slice(line)?)),
        None => Ok(None),
    }
}

/// Whole ledger text, `None` when the file does not exist. A non-empty ledger
/// that does not end with a newline is a torn tail and is an error.
fn read_ledger(path: &Path) -> Result<Option<String>, String> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
        let intact = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
        return Err(torn_tail(path, bytes.len() as u64, intact as u64));
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Seal and append one test event. `payload` is the run's full output.
pub fn append(
    root: &Path,
    agent: &str,
    target: &str,
    intent: &str,
    payload: &[u8],
) -> io::Result<LedgerEvent> {
    append_action(root, "test", agent, target, intent, payload)
}

/// Seal and append one event with a Crumb action such as `test` or `audit`.
pub fn append_action(
    root: &Path,
    action: &str,
    agent: &str,
    target: &str,
    intent: &str,
    payload: &[u8],
) -> io::Result<LedgerEvent> {
    append_with(
        root,
        action,
        agent,
        target,
        intent,
        payload,
        |file, line| {
            file.write_all(line)?;
            file.sync_data()
        },
    )
}

/// `append_action` with the write-and-sync step passed in, so tests can make it
/// fail part way. On failure the ledger is truncated back to its length before
/// the append, leaving it byte-identical, and the original error is returned.
fn append_with(
    root: &Path,
    action: &str,
    agent: &str,
    target: &str,
    intent: &str,
    payload: &[u8],
    write: impl FnOnce(&mut File, &[u8]) -> io::Result<()>,
) -> io::Result<LedgerEvent> {
    fs::create_dir_all(root)?;
    let _guard = FileLock::acquire(&root.join("ledger.lock"))?;
    let path = root.join(LEDGER_FILE);
    let (index, parent_hash, last_ts) = match last_event(&path)? {
        Some(e) => (e.index + 1, e.hash, e.timestamp),
        None => (0, [0u8; 32], 0),
    };
    let timestamp = now().max(last_ts);
    let payload_hash = *blake3::hash(payload).as_bytes();
    let hash = compute_hash(
        index,
        timestamp,
        agent,
        action,
        target,
        intent,
        &payload_hash,
        &parent_hash,
    );
    let event = LedgerEvent {
        index,
        timestamp,
        agent: agent.to_string(),
        action: action.to_string(),
        target: target.to_string(),
        intent: intent.to_string(),
        payload_hash,
        parent_hash,
        hash,
    };
    let mut line = serde_json::to_string(&event)?;
    line.push('\n');
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    // Under the ledger lock, so no other writer moves the end while we hold it.
    let before = file.metadata()?.len();
    if let Err(err) = write(&mut file, line.as_bytes()) {
        return Err(match file.set_len(before).and_then(|()| file.sync_data()) {
            Ok(()) => err,
            Err(rollback) => io::Error::new(
                err.kind(),
                format!(
                    "{}: append of event {index} failed: {err}; rollback to {before} bytes \
                     also failed: {rollback}",
                    path.display()
                ),
            ),
        });
    }
    Ok(event)
}

/// Verify the whole chain. Returns the event count and head hash.
pub fn verify(path: &Path) -> Result<(u64, [u8; 32]), String> {
    let Some(text) = read_ledger(path)? else {
        return Ok((0, [0u8; 32]));
    };
    let (mut expected, mut parent, mut last_ts) = (0u64, [0u8; 32], 0u64);
    for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let e: LedgerEvent =
            serde_json::from_str(line).map_err(|err| format!("line {}: {err}", n + 1))?;
        if e.index != expected {
            return Err(format!("event {}: index gap, expected {expected}", e.index));
        }
        if e.parent_hash != parent {
            return Err(format!(
                "event {}: parent hash does not match previous event",
                e.index
            ));
        }
        if e.timestamp < last_ts {
            return Err(format!("event {}: timestamp goes backwards", e.index));
        }
        if e.recompute() != e.hash {
            return Err(format!(
                "event {}: hash mismatch, event was altered",
                e.index
            ));
        }
        expected += 1;
        parent = e.hash;
        last_ts = e.timestamp;
    }
    Ok((expected, parent))
}

pub fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read the event at `index`, or `Ok(None)` when the ledger is shorter.
pub fn read_event(path: &Path, index: u64) -> Result<Option<LedgerEvent>, String> {
    let Some(text) = read_ledger(path)? else {
        return Ok(None);
    };
    for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let e: LedgerEvent =
            serde_json::from_str(line).map_err(|err| format!("line {}: {err}", n + 1))?;
        if e.index == index {
            return Ok(Some(e));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_dir;

    #[test]
    fn chain_verifies_and_tampering_is_caught() {
        let root = temp_dir("ledger");
        for i in 0..3 {
            append(&root, "tester", &format!("job-{i}"), "pass exit=0", b"ok").unwrap();
        }
        let path = root.join(LEDGER_FILE);
        let (count, _) = verify(&path).unwrap();
        assert_eq!(count, 3);

        let tampered = fs::read_to_string(&path)
            .unwrap()
            .replacen("job-1", "job-X", 1);
        fs::write(&path, tampered).unwrap();
        assert!(verify(&path).unwrap_err().contains("event 1"));
    }

    #[test]
    fn genesis_parent_is_zero_and_events_link() {
        let root = temp_dir("ledger-link");
        let a = append(&root, "t", "a", "pass", b"1").unwrap();
        let b = append(&root, "t", "b", "pass", b"2").unwrap();
        assert_eq!(a.parent_hash, [0u8; 32]);
        assert_eq!(b.parent_hash, a.hash);
        assert_eq!(b.index, 1);
    }

    fn enospc() -> io::Error {
        io::Error::from_raw_os_error(libc::ENOSPC)
    }

    #[test]
    fn failed_append_rolls_back_and_next_append_links() {
        let root = temp_dir("ledger-rollback");
        let path = root.join(LEDGER_FILE);
        // A failed genesis append must leave an empty ledger.
        let err = append_with(&root, "test", "t", "g", "pass", b"0", |f, line| {
            f.write_all(&line[..line.len() / 2])?;
            Err(enospc())
        })
        .unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ENOSPC));
        assert_eq!(fs::read(&path).unwrap(), b"");

        let a = append(&root, "t", "a", "pass", b"1").unwrap();
        let b = append(&root, "t", "b", "pass", b"2").unwrap();
        assert_eq!(a.index, 0);
        let before = fs::read(&path).unwrap();

        // Half the line written then ENOSPC, and the whole line written then a
        // failed sync: both must leave the ledger byte-identical.
        let half = |f: &mut File, line: &[u8]| {
            f.write_all(&line[..line.len() / 2])?;
            Err(enospc())
        };
        let synced = |f: &mut File, line: &[u8]| {
            f.write_all(line)?;
            Err(enospc())
        };
        for step in [
            &half as &dyn Fn(&mut File, &[u8]) -> io::Result<()>,
            &synced,
        ] {
            let err = append_with(&root, "test", "t", "c", "pass", b"3", step).unwrap_err();
            assert_eq!(err.raw_os_error(), Some(libc::ENOSPC));
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(verify(&path).unwrap(), (2, b.hash));
        }

        let c = append(&root, "t", "c", "pass", b"3").unwrap();
        assert_eq!(c.index, 2);
        assert_eq!(c.parent_hash, b.hash);
        assert_eq!(verify(&path).unwrap(), (3, c.hash));
    }

    #[test]
    fn torn_tail_fails_closed_with_clear_error() {
        let root = temp_dir("ledger-torn");
        let path = root.join(LEDGER_FILE);
        append(&root, "t", "a", "pass", b"1").unwrap();
        append(&root, "t", "b", "pass", b"2").unwrap();
        let intact = fs::metadata(&path).unwrap().len();
        let torn = b"{\"index\":2,\"timestamp\":1,\"agent\":\"t";
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(torn).unwrap();
        drop(f);

        let expected = format!(
            "torn tail: an append was cut off and left {} byte(s) without a closing newline \
             (last newline at byte offset {}); the chain is intact up to byte offset {intact}",
            torn.len(),
            intact - 1
        );
        let err = append(&root, "t", "c", "pass", b"3").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains(&expected), "{err}");
        let err = verify(&path).unwrap_err();
        assert!(err.contains(&expected), "{err}");
        assert!(!err.contains("EOF"), "{err}");
        let err = read_event(&path, 0).unwrap_err();
        assert!(err.contains(&expected), "{err}");
        // The failed append must not have touched the torn file either.
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            intact + torn.len() as u64
        );

        // A ledger that is only a torn line names offset 0.
        fs::write(&path, &torn[..10]).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(
            err.contains("left 10 byte(s) without a closing newline (the file has no newline); the chain is intact up to byte offset 0"),
            "{err}"
        );
    }
}
