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

//! The in-memory object index of the host filesystem adapter (ADR 0003
//! section 4.1, finding 8).
//!
//! One entry per published or recovered object file: 64 bytes of state
//! (`IndexEntry`), capped by `max_objects` (65536 by default, 4 194 304 bytes
//! of entries). The index also keeps the aggregates admission needs in O(1):
//! total physical bytes, bytes marked for removal, and bytes that are busy
//! (pinned or being loaded), so `evictable()` never scans.
//!
//! Pure data structure: no I/O, `no_std` + `alloc`. All locking is the
//! adapter's (the index lives behind its mutex).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;

/// The object belongs to another namespace: counted toward quota, never
/// served, evicted first.
pub const FLAG_FOREIGN: u32 = 1;
/// Marked for removal: lookups miss, the unlink has not returned yet.
pub const FLAG_EVICTING: u32 = 2;
/// Renamed into place but the directory sync failed (ADR 2.4 P4).
pub const FLAG_INDETERMINATE: u32 = 4;

/// Bytes of one index entry (ADR 4.1 as amended: 64 bytes per entry, payload length 64-bit).
pub const INDEX_ENTRY_BYTES: usize = 64;

/// What the index knows about one object file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct IndexEntry {
    pub key: [u8; 32],
    /// `max(st_size, st_blocks * 512)`: includes the 4096-byte header region.
    pub phys_bytes: u64,
    /// Nanoseconds since the Unix epoch: file mtime at recovery or publish
    /// time, raised to the load time by a successful load (in memory only).
    pub mtime_ns: u64,
    /// Payload bytes (`lookup`'s answer), the record header's u64 `payload_len`.
    pub payload_len: u64,
    pub flags: u32,
}

impl IndexEntry {
    pub fn is_foreign(&self) -> bool {
        self.flags & FLAG_FOREIGN != 0
    }

    pub fn is_evicting(&self) -> bool {
        self.flags & FLAG_EVICTING != 0
    }

    pub fn is_indeterminate(&self) -> bool {
        self.flags & FLAG_INDETERMINATE != 0
    }
}

/// An entry without its key (the map holds the key).
#[derive(Debug, Clone, Copy)]
struct Body {
    phys_bytes: u64,
    mtime_ns: u64,
    payload_len: u64,
    flags: u32,
}

impl Body {
    fn entry(&self, key: [u8; 32]) -> IndexEntry {
        IndexEntry {
            key,
            phys_bytes: self.phys_bytes,
            mtime_ns: self.mtime_ns,
            payload_len: self.payload_len,
            flags: self.flags,
        }
    }
}

/// Keyed object index with O(1) admission aggregates.
#[derive(Debug, Default)]
pub struct Index {
    map: BTreeMap<[u8; 32], Body>,
    total_bytes: u64,
    evicting_bytes: u64,
    evicting_count: u64,
    /// Pin and in-flight-load reference counts, by key.
    busy: BTreeMap<[u8; 32], u32>,
    /// Bytes and count of non-evicting entries with a busy count above zero.
    busy_bytes: u64,
    busy_count: u64,
}

impl Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, key: &[u8; 32]) -> Option<IndexEntry> {
        self.map.get(key).map(|b| b.entry(*key))
    }

    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.map.contains_key(key)
    }

    /// Sum of `phys_bytes` over every entry, including those being removed.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// (bytes, count) of entries marked for removal whose unlink has not returned.
    pub fn evicting(&self) -> (u64, u64) {
        (self.evicting_bytes, self.evicting_count)
    }

    /// (bytes, count) of entries eviction could reclaim now: not marked, not
    /// pinned, not being loaded.
    pub fn evictable(&self) -> (u64, u64) {
        (
            self.total_bytes - self.evicting_bytes - self.busy_bytes,
            self.map.len() as u64 - self.evicting_count - self.busy_count,
        )
    }

    pub fn is_busy(&self, key: &[u8; 32]) -> bool {
        self.busy.get(key).copied().unwrap_or(0) > 0
    }

    /// Insert or replace. Returns the entry it replaced.
    pub fn insert(&mut self, e: IndexEntry) -> Option<IndexEntry> {
        let old = self.remove(&e.key);
        self.total_bytes += e.phys_bytes;
        if e.is_evicting() {
            self.evicting_bytes += e.phys_bytes;
            self.evicting_count += 1;
        } else if self.is_busy(&e.key) {
            self.busy_bytes += e.phys_bytes;
            self.busy_count += 1;
        }
        self.map.insert(
            e.key,
            Body {
                phys_bytes: e.phys_bytes,
                mtime_ns: e.mtime_ns,
                payload_len: e.payload_len,
                flags: e.flags,
            },
        );
        old
    }

    /// Remove an entry. Busy reference counts for the key are kept: the
    /// pins and loads that hold them release them later.
    pub fn remove(&mut self, key: &[u8; 32]) -> Option<IndexEntry> {
        let b = self.map.remove(key)?;
        self.total_bytes -= b.phys_bytes;
        if b.flags & FLAG_EVICTING != 0 {
            self.evicting_bytes -= b.phys_bytes;
            self.evicting_count -= 1;
        } else if self.is_busy(key) {
            self.busy_bytes -= b.phys_bytes;
            self.busy_count -= 1;
        }
        Some(b.entry(*key))
    }

    /// Mark for removal. Refused (false) for an absent, already marked or busy entry.
    pub fn mark_evicting(&mut self, key: &[u8; 32]) -> bool {
        if self.is_busy(key) {
            return false;
        }
        match self.map.get_mut(key) {
            Some(b) if b.flags & FLAG_EVICTING == 0 => {
                b.flags |= FLAG_EVICTING;
                self.evicting_bytes += b.phys_bytes;
                self.evicting_count += 1;
                true
            }
            _ => false,
        }
    }

    /// Return a marked entry to service (the unlink failed).
    pub fn clear_evicting(&mut self, key: &[u8; 32]) {
        let busy = self.is_busy(key);
        if let Some(b) = self.map.get_mut(key) {
            if b.flags & FLAG_EVICTING != 0 {
                b.flags &= !FLAG_EVICTING;
                self.evicting_bytes -= b.phys_bytes;
                self.evicting_count -= 1;
                if busy {
                    self.busy_bytes += b.phys_bytes;
                    self.busy_count += 1;
                }
            }
        }
    }

    /// Take one busy reference. False (nothing taken) for an absent or marked entry.
    pub fn busy_inc(&mut self, key: &[u8; 32]) -> bool {
        let Some(b) = self.map.get(key) else {
            return false;
        };
        if b.flags & FLAG_EVICTING != 0 {
            return false;
        }
        let c = self.busy.entry(*key).or_insert(0);
        *c += 1;
        if *c == 1 {
            self.busy_bytes += b.phys_bytes;
            self.busy_count += 1;
        }
        true
    }

    /// Release one busy reference (no-op when there is none).
    pub fn busy_dec(&mut self, key: &[u8; 32]) {
        let Some(c) = self.busy.get_mut(key) else {
            return;
        };
        *c -= 1;
        if *c > 0 {
            return;
        }
        self.busy.remove(key);
        if let Some(b) = self.map.get(key) {
            if b.flags & FLAG_EVICTING == 0 {
                self.busy_bytes -= b.phys_bytes;
                self.busy_count -= 1;
            }
        }
    }

    /// Raise the entry's recency to `mtime_ns` (a successful load).
    pub fn touch(&mut self, key: &[u8; 32], mtime_ns: u64) {
        if let Some(b) = self.map.get_mut(key) {
            b.mtime_ns = b.mtime_ns.max(mtime_ns);
        }
    }

    /// Set the indeterminate flag (P4 failed) or clear it.
    pub fn set_indeterminate(&mut self, key: &[u8; 32], on: bool) {
        if let Some(b) = self.map.get_mut(key) {
            if on {
                b.flags |= FLAG_INDETERMINATE;
            } else {
                b.flags &= !FLAG_INDETERMINATE;
            }
        }
    }

    /// ADR 4.1 step 6: foreign namespaces first, then least recently used
    /// (smallest `mtime_ns`, then key); never a busy or marked entry. O(n),
    /// called only on the worker.
    pub fn pick_victim(&self) -> Option<[u8; 32]> {
        self.map
            .iter()
            .filter(|(k, b)| b.flags & FLAG_EVICTING == 0 && !self.is_busy(k))
            .min_by_key(|(k, b)| (b.flags & FLAG_FOREIGN == 0, b.mtime_ns, **k))
            .map(|(k, _)| *k)
    }

    /// Recompute every aggregate from scratch and compare. For tests and
    /// debug assertions.
    pub fn audit(&self) -> Result<(), String> {
        let mut total = 0u64;
        let (mut eb, mut ec, mut bb, mut bc) = (0u64, 0u64, 0u64, 0u64);
        for (k, b) in &self.map {
            total += b.phys_bytes;
            if b.flags & FLAG_EVICTING != 0 {
                eb += b.phys_bytes;
                ec += 1;
            } else if self.is_busy(k) {
                bb += b.phys_bytes;
                bc += 1;
            }
        }
        let got = (
            self.total_bytes,
            self.evicting_bytes,
            self.evicting_count,
            self.busy_bytes,
            self.busy_count,
        );
        if got != (total, eb, ec, bb, bc) {
            return Err(format!(
                "index aggregates {got:?} differ from recomputed {:?}",
                (total, eb, ec, bb, bc)
            ));
        }
        Ok(())
    }
}
