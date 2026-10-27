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

//! Landing-buffer coalescing table (ADR 0003 section 7.3, review finding 6).
//!
//! Same-key loads coalesce at the **job and landing-buffer level**: one
//! in-flight load per chunk key lands in that chunk's holder, later requests
//! register as waiters on the job. The table is pure bookkeeping (no I/O, no
//! locks); the worker keeps it inside its own critical section.
//!
//! Counting rules (consistent with the deterministic fake, C2):
//!
//! * The request that opened the job is waiter 1, so `max_waiters_per_job`
//!   counts it; attaching is refused once `waiters >= max_waiters_per_job`.
//! * `max_waiters_total` bounds the **attached** waiters (waiters beyond the
//!   first) over all jobs.
//! * Stores share the table to deduplicate a store of a key that is already
//!   in flight: the duplicate is an additional owner of the same job, so one
//!   owner cancelling cannot cancel the others' store.

use crate::identity::ChunkKey;
use crate::store::JobId;
use alloc::collections::BTreeMap;

/// What kind of work a key is shared for. Loads and stores of one key never
/// share a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Share {
    Load,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoalesceLimits {
    /// ADR 7.1: default 16 (the settings table lists 8, see the C5 report).
    pub max_waiters_per_job: u32,
    /// ADR 7.1: default 1024.
    pub max_waiters_total: u32,
}

impl Default for CoalesceLimits {
    fn default() -> Self {
        Self {
            max_waiters_per_job: 16,
            max_waiters_total: 1024,
        }
    }
}

/// Why `attach` refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachRefused {
    /// No such job, or every waiter already left (the job is being torn down).
    NoSuchJob,
    /// The job already has `max_waiters_per_job` waiters.
    JobFull,
    /// `max_waiters_total` attached waiters exist over all jobs.
    TotalFull,
}

/// What `detach` left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detach {
    /// The job is not in the table (a flush, or already closed).
    Unknown,
    /// Other waiters remain: the shared work must go on.
    Remaining(u32),
    /// That was the last waiter: nobody wants the job any more.
    Last,
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    share: Share,
    key: [u8; 32],
    waiters: u32,
}

/// Key to job bookkeeping with per-job and total waiter bounds.
#[derive(Debug, Clone)]
pub struct CoalesceTable {
    limits: CoalesceLimits,
    by_key: BTreeMap<(Share, [u8; 32]), JobId>,
    jobs: BTreeMap<JobId, Entry>,
    attached_total: u32,
}

impl CoalesceTable {
    pub fn new(limits: CoalesceLimits) -> Self {
        Self {
            limits,
            by_key: BTreeMap::new(),
            jobs: BTreeMap::new(),
            attached_total: 0,
        }
    }

    pub fn limits(&self) -> CoalesceLimits {
        self.limits
    }

    /// Register `job` as the owner of `(share, key)` with its opener as waiter 1.
    /// If another job with waiters already owns the key, that job stays the
    /// coalescing target; `job` is still tracked for `attach`/`detach`/`close`.
    pub fn open(&mut self, share: Share, key: &ChunkKey, job: JobId) {
        let k = (share, key.0 .0);
        let taken = self
            .by_key
            .get(&k)
            .and_then(|j| self.jobs.get(j))
            .is_some_and(|e| e.waiters > 0);
        if !taken {
            self.by_key.insert(k, job);
        }
        self.jobs.insert(
            job,
            Entry {
                share,
                key: k.1,
                waiters: 1,
            },
        );
    }

    /// The job a new request for `(share, key)` may join: it exists and still
    /// has at least one waiter.
    pub fn find(&self, share: Share, key: &ChunkKey) -> Option<JobId> {
        let job = *self.by_key.get(&(share, key.0 .0))?;
        (self.jobs.get(&job)?.waiters > 0).then_some(job)
    }

    /// True while any job (even one whose waiters all left) is registered for the key.
    pub fn has(&self, share: Share, key: &ChunkKey) -> bool {
        self.by_key.contains_key(&(share, key.0 .0))
    }

    /// Add one waiter; returns the waiter count now. Bounded.
    pub fn attach(&mut self, job: JobId) -> Result<u32, AttachRefused> {
        let max_job = self.limits.max_waiters_per_job;
        let max_total = self.limits.max_waiters_total;
        let attached = self.attached_total;
        let e = match self.jobs.get_mut(&job) {
            Some(e) if e.waiters > 0 => e,
            _ => return Err(AttachRefused::NoSuchJob),
        };
        if e.waiters >= max_job {
            return Err(AttachRefused::JobFull);
        }
        if attached >= max_total {
            return Err(AttachRefused::TotalFull);
        }
        e.waiters += 1;
        self.attached_total += 1;
        Ok(e.waiters)
    }

    /// Remove one waiter (the caller cancelled its own interest).
    pub fn detach(&mut self, job: JobId) -> Detach {
        let Some(e) = self.jobs.get_mut(&job) else {
            return Detach::Unknown;
        };
        match e.waiters {
            0 | 1 => {
                e.waiters = 0;
                Detach::Last
            }
            n => {
                e.waiters = n - 1;
                self.attached_total -= 1;
                Detach::Remaining(n - 1)
            }
        }
    }

    /// The job reached its terminal completion: forget it and release its
    /// attached waiters from the total.
    pub fn close(&mut self, job: JobId) {
        let Some(e) = self.jobs.remove(&job) else {
            return;
        };
        self.attached_total -= e.waiters.saturating_sub(1);
        let k = (e.share, e.key);
        if self.by_key.get(&k) == Some(&job) {
            self.by_key.remove(&k);
        }
    }

    pub fn waiters(&self, job: JobId) -> u32 {
        self.jobs.get(&job).map_or(0, |e| e.waiters)
    }

    /// Attached waiters (beyond each job's first) over all jobs.
    pub fn attached_total(&self) -> u32 {
        self.attached_total
    }

    /// Jobs currently tracked.
    pub fn jobs(&self) -> usize {
        self.jobs.len()
    }
}
