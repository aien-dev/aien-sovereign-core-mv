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

//! Holder generation allocation (ADR 0003 section 5.2, Codex re-review 2).
//!
//! A chunk holder is `SequenceId { slot: u32::MAX, generation }`. The old rule
//! `generation = job_id as u32` silently wraps after 2^32 jobs and could hand
//! one generation to two live holders. The generation now comes from an
//! allocator that never returns zero and never returns a live value.

use aien_kv_store::holder::{generation_for_job, HolderError, HolderGenerations};
use aien_kv_store::JobId;

#[test]
fn generations_are_nonzero_and_distinct_while_live() {
    let mut h = HolderGenerations::with_max(8);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..8 {
        let g = h.alloc().unwrap();
        assert_ne!(g, 0);
        assert!(seen.insert(g), "{g} handed out twice");
    }
    assert_eq!(h.live(), 8);
}

#[test]
fn exhaustion_is_refused_not_wrapped_onto_a_live_generation() {
    let mut h = HolderGenerations::with_max(3);
    let a = h.alloc().unwrap();
    let _b = h.alloc().unwrap();
    let _c = h.alloc().unwrap();
    assert_eq!(h.alloc(), Err(HolderError::Exhausted));
    assert!(h.release(a));
    assert_eq!(h.alloc().unwrap(), a, "the freed value is reusable");
}

#[test]
fn the_counter_wraps_past_zero_and_skips_live_generations() {
    let mut h = HolderGenerations::starting_at(3, 4);
    let first = h.alloc().unwrap();
    assert_eq!(first, 3);
    let second = h.alloc().unwrap();
    assert_eq!(second, 4);
    // Wrap: next candidate is 1 (never 0).
    let third = h.alloc().unwrap();
    assert_eq!(third, 1);
    // 3 and 4 are live and 1 is live: only 2 is left.
    assert_eq!(h.alloc().unwrap(), 2);
    assert_eq!(h.alloc(), Err(HolderError::Exhausted));
    h.release(4);
    assert_eq!(h.alloc().unwrap(), 4);
}

#[test]
fn releasing_something_not_live_is_reported() {
    let mut h = HolderGenerations::with_max(4);
    assert!(!h.release(2));
    let g = h.alloc().unwrap();
    assert!(h.release(g));
    assert!(!h.release(g), "double release");
}

#[test]
fn job_id_mapping_refuses_to_wrap() {
    assert_eq!(generation_for_job(JobId(1)), Ok(1));
    assert_eq!(generation_for_job(JobId(u32::MAX as u64)), Ok(u32::MAX));
    assert_eq!(
        generation_for_job(JobId(0)),
        Err(HolderError::JobIdOutOfRange)
    );
    assert_eq!(
        generation_for_job(JobId(u32::MAX as u64 + 1)),
        Err(HolderError::JobIdOutOfRange),
        "a plain `as u32` would alias job 2^32 onto holder 0"
    );
    assert_eq!(
        generation_for_job(JobId((1 << 32) | 7)),
        Err(HolderError::JobIdOutOfRange),
        "a plain `as u32` would alias this onto job 7"
    );
}
