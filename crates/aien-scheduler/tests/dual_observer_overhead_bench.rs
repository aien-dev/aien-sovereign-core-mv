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

//! Overhead of the `serve.admit_preempt` read-only tap (DUAL-3a).
//!
//! Manual benchmark, ignored by default (no timing assertion lives in CI):
//!
//! ```text
//! cargo test -p aien-scheduler --release --test dual_observer_overhead_bench -- --ignored --nocapture
//! ```
//!
//! Three arms run the whole pre-registered workload set on fresh state:
//!
//! - `absent`: no observer installed (the production configuration);
//! - `counting`: `CountingObserver`, which serializes each record to count
//!   bytes and keeps nothing (the cost of producing records);
//! - `recording`: `RecordingObserver`, which clones and keeps every record.
//!
//! Each arm runs `ROUNDS` rounds and reports the median wall time, decision
//! count, records emitted, serialized bytes emitted and heap allocations
//! (counted by a wrapping global allocator). The acceptance threshold is
//! pre-registered in the evidence receipt, not here.

mod dual_common;

use aien_scheduler::dual_observer::{CountingObserver, RecordingObserver};
use dual_common::{build, run, scenarios};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

struct CountingAlloc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

const ROUNDS: usize = 7;

#[derive(Debug, Clone, Copy)]
struct Arm {
    name: &'static str,
    wall_ns_median: u128,
    wall_ns_min: u128,
    decisions: usize,
    records: u64,
    bytes: u64,
    allocs: u64,
    alloc_bytes: u64,
}

fn median(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn arm(name: &'static str, mode: u8) -> Arm {
    let set = scenarios();
    let mut walls = Vec::with_capacity(ROUNDS);
    let mut decisions = 0;
    let mut records = 0u64;
    let mut bytes = 0u64;
    let mut allocs = 0u64;
    let mut alloc_bytes = 0u64;
    for _ in 0..ROUNDS {
        let mut round_decisions = 0;
        let mut round_records = 0u64;
        let mut round_bytes = 0u64;
        let a0 = ALLOCS.load(Relaxed);
        let b0 = ALLOC_BYTES.load(Relaxed);
        let t0 = Instant::now();
        for s in &set {
            let (mut scheduler, kv) = build(s);
            let counting = Arc::new(CountingObserver::new());
            let recording = Arc::new(RecordingObserver::new());
            match mode {
                1 => scheduler.set_decision_observer(Some(counting.clone())),
                2 => scheduler.set_decision_observer(Some(recording.clone())),
                _ => {}
            }
            let trace = run(s, &mut scheduler, &kv);
            round_decisions += trace.decisions;
            match mode {
                1 => {
                    round_records += counting.records();
                    round_bytes += counting.bytes();
                }
                2 => {
                    let recs = recording.take();
                    round_records += recs.len() as u64;
                    round_bytes += recs
                        .iter()
                        .map(|r| serde_json::to_vec(r).unwrap().len() as u64)
                        .sum::<u64>();
                }
                _ => {}
            }
        }
        let wall = t0.elapsed().as_nanos();
        let a1 = ALLOCS.load(Relaxed);
        let b1 = ALLOC_BYTES.load(Relaxed);
        walls.push(wall);
        decisions = round_decisions;
        records = round_records;
        bytes = round_bytes;
        allocs = a1 - a0;
        alloc_bytes = b1 - b0;
    }
    Arm {
        name,
        wall_ns_median: median(&mut walls),
        wall_ns_min: *walls.iter().min().unwrap(),
        decisions,
        records,
        bytes,
        allocs,
        alloc_bytes,
    }
}

#[test]
#[ignore = "manual overhead benchmark; run with --ignored --nocapture in release"]
fn dual_observer_overhead() {
    // Warm up once so the first arm does not pay for page faults alone.
    let _ = arm("warmup", 0);
    let absent = arm("absent", 0);
    let counting = arm("counting", 1);
    let recording = arm("recording", 2);
    let absent2 = arm("absent_repeat", 0);

    println!("DUAL serve.admit_preempt observer overhead, {ROUNDS} rounds per arm, whole workload set per round");
    println!(
        "{:<14} {:>14} {:>14} {:>10} {:>9} {:>12} {:>10} {:>12} {:>8}",
        "arm",
        "median_ms",
        "min_ms",
        "decisions",
        "records",
        "bytes",
        "allocs",
        "alloc_bytes",
        "ratio"
    );
    for a in [absent, counting, recording, absent2] {
        println!(
            "{:<14} {:>14.3} {:>14.3} {:>10} {:>9} {:>12} {:>10} {:>12} {:>8.3}",
            a.name,
            a.wall_ns_median as f64 / 1e6,
            a.wall_ns_min as f64 / 1e6,
            a.decisions,
            a.records,
            a.bytes,
            a.allocs,
            a.alloc_bytes,
            a.wall_ns_median as f64 / absent.wall_ns_median as f64
        );
    }
    // Deterministic facts (not timing): records equal decisions when a tap is on.
    assert_eq!(counting.records, counting.decisions as u64);
    assert_eq!(recording.records, recording.decisions as u64);
    assert_eq!(absent.records, 0);
    assert_eq!(absent.decisions, counting.decisions);
}

/// Exploratory (NOT pre-registered) protocol added after the first run: arms
/// interleaved round by round so clock scaling and cache warmth hit every arm
/// alike, 50 rounds, medians. Reported in the receipt as post-hoc only; it
/// does not replace the pre-registered verdict above.
#[test]
#[ignore = "manual overhead benchmark (interleaved, exploratory); --ignored --nocapture in release"]
fn dual_observer_overhead_interleaved() {
    const N: usize = 50;
    let set = scenarios();
    let mut absent_w = Vec::with_capacity(N);
    let mut counting_w = Vec::with_capacity(N);
    let mut recording_w = Vec::with_capacity(N);
    let mut absent_a = Vec::with_capacity(N);
    let mut counting_a = Vec::with_capacity(N);
    let mut recording_a = Vec::with_capacity(N);
    let mut decisions = 0usize;
    let mut records = 0u64;
    let mut bytes = 0u64;

    let run_arm = |mode: u8, walls: &mut Vec<u128>, allocs: &mut Vec<u64>| {
        let a0 = ALLOCS.load(Relaxed);
        let t0 = Instant::now();
        let mut d = 0usize;
        let mut r = 0u64;
        let mut b = 0u64;
        for s in &set {
            let (mut scheduler, kv) = build(s);
            let counting = Arc::new(CountingObserver::new());
            let recording = Arc::new(RecordingObserver::new());
            match mode {
                1 => scheduler.set_decision_observer(Some(counting.clone())),
                2 => scheduler.set_decision_observer(Some(recording.clone())),
                _ => {}
            }
            let trace = run(s, &mut scheduler, &kv);
            d += trace.decisions;
            match mode {
                1 => {
                    r += counting.records();
                    b += counting.bytes();
                }
                2 => {
                    let recs = recording.take();
                    r += recs.len() as u64;
                    b += recs
                        .iter()
                        .map(|x| serde_json::to_vec(x).unwrap().len() as u64)
                        .sum::<u64>();
                }
                _ => {}
            }
        }
        walls.push(t0.elapsed().as_nanos());
        allocs.push(ALLOCS.load(Relaxed) - a0);
        (d, r, b)
    };

    // Warm-up round for every arm, discarded.
    let (mut w, mut a) = (Vec::new(), Vec::new());
    for mode in 0..3u8 {
        run_arm(mode, &mut w, &mut a);
    }
    for _ in 0..N {
        let (d, _, _) = run_arm(0, &mut absent_w, &mut absent_a);
        decisions = d;
        let (_, r, b) = run_arm(1, &mut counting_w, &mut counting_a);
        records = r;
        bytes = b;
        run_arm(2, &mut recording_w, &mut recording_a);
    }
    let med = |v: &mut Vec<u128>| median(v);
    let (ma, mc, mr) = (
        med(&mut absent_w),
        med(&mut counting_w),
        med(&mut recording_w),
    );
    println!("DUAL serve.admit_preempt observer overhead, INTERLEAVED exploratory protocol, {N} rounds, median of per-round wall");
    println!(
        "{:<10} {:>12} {:>12} {:>10} {:>9} {:>12} {:>10} {:>8}",
        "arm", "median_ms", "min_ms", "decisions", "records", "bytes", "allocs", "ratio"
    );
    for (name, w, al, rec, by) in [
        ("absent", &absent_w, &absent_a, 0u64, 0u64),
        ("counting", &counting_w, &counting_a, records, bytes),
        ("recording", &recording_w, &recording_a, records, bytes),
    ] {
        let m = {
            let mut c = w.clone();
            median(&mut c)
        };
        println!(
            "{:<10} {:>12.3} {:>12.3} {:>10} {:>9} {:>12} {:>10} {:>8.3}",
            name,
            m as f64 / 1e6,
            *w.iter().min().unwrap() as f64 / 1e6,
            decisions,
            rec,
            by,
            al[al.len() / 2],
            m as f64 / ma as f64
        );
    }
    let _ = (mc, mr);
}
