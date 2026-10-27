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

/// Last event, read from the file tail so appends stay O(1) as the ledger grows.
fn last_event(path: &Path) -> io::Result<Option<LedgerEvent>> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let len = file.metadata()?.len();
    let start = len.saturating_sub(64 * 1024);
    file.seek(SeekFrom::Start(start))?;
    let mut tail = String::new();
    file.read_to_string(&mut tail)?;
    match tail.lines().rev().find(|l| !l.trim().is_empty()) {
        Some(line) => Ok(Some(serde_json::from_str(line)?)),
        None => Ok(None),
    }
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
    file.write_all(line.as_bytes())?;
    file.sync_data()?;
    Ok(event)
}

/// Verify the whole chain. Returns the event count and head hash.
pub fn verify(path: &Path) -> Result<(u64, [u8; 32]), String> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((0, [0u8; 32])),
        Err(e) => return Err(e.to_string()),
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
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
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
}
