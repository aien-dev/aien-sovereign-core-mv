#![cfg(feature = "std")]
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

//! Streaming recovery (Grok C4 review P1-2, P2-6, P2-7; ADR 0003 section
//! 4.1): the walk holds one directory cursor, buffers no listing, never holds
//! the index lock across a listing or a header read, reports a failed walk,
//! and keeps 64-bit payload lengths.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::{Boundary, HostFsStore};
use aien_kv_store::index::{Index, IndexEntry, INDEX_ENTRY_BYTES};
use aien_kv_store::{ChunkKey, KvStore, StoreState};
use rig::*;
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

#[test]
fn recovery_reads_headers_without_holding_the_index_lock() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=3 {
        let (key, header) = fx.chunk(n, 8);
        fx.plant(&key, &header, &pattern(n, 2 * BS), BS);
    }
    let cell: StoreCell = Arc::new(OnceLock::new());
    let seen = log();
    let hook = {
        let cell = cell.clone();
        let seen = seen.clone();
        Arc::new(CallbackHook {
            scripted: fx.faults.clone(),
            f: Box::new(move |b, _k: &ChunkKey| {
                if b == Boundary::RecoverEntry {
                    let s = cell.get().unwrap().clone();
                    let (tx, rx) = mpsc::channel();
                    std::thread::spawn(move || {
                        let _ = s.capacity();
                        let _ = tx.send(());
                    });
                    let free = rx.recv_timeout(Duration::from_secs(2)).is_ok();
                    seen.lock().unwrap().push(format!("{free}"));
                }
                None
            }),
        })
    };
    let s = Arc::new(fx.open_raw(|c| c.faults = Some(hook)));
    let _ = cell.set(s.clone());
    let p = recover_all(&s);
    assert_eq!(p.indexed, 3);
    let seen = seen.lock().unwrap().clone();
    assert!(seen.len() >= 3, "the hook ran per entry: {seen:?}");
    assert!(
        seen.iter().all(|x| x == "true"),
        "the index lock was free during every entry: {seen:?}"
    );
}

#[test]
fn recovery_buffers_no_directory_listing_past_max_objects() {
    let fx = Fx::new();
    fx.prime();
    for n in 1..=40 {
        let (key, header) = fx.chunk(n, 8);
        fx.plant(&key, &header, &pattern(n, 2 * BS), BS);
    }
    let s = fx.open_raw(|c| {
        c.max_objects = 4;
        c.purge_after_recovery = false;
    });
    let p = recover_all(&s);
    assert_eq!((p.indexed, p.unindexed_objects), (4, 36));
    let st = s.stats();
    assert!(
        st.walk_peak_buffered <= 1,
        "one entry at a time, not a listing ({})",
        st.walk_peak_buffered
    );
    s.start_purge();
    s.run_until_idle();
    assert_eq!(object_files(fx.root()).len(), 4);
    assert!(s.stats().walk_peak_buffered <= 1);
}

#[test]
fn a_failed_store_reports_failed_and_not_done() {
    let fx = Fx::new();
    fx.prime();
    std::fs::write(fx.root().join("STORE_ID"), b"short").unwrap();
    let s: HostFsStore = fx.open_raw(|_| {});
    assert_eq!(s.capacity().state, StoreState::Failed);
    let p = s.recover_step(50);
    assert!(p.failed && !p.done, "{p:?}");
}

#[test]
fn index_entries_keep_a_64_bit_payload_length_in_64_bytes() {
    assert_eq!(std::mem::size_of::<IndexEntry>(), INDEX_ENTRY_BYTES);
    assert_eq!(INDEX_ENTRY_BYTES, 64);
    assert_eq!(INDEX_ENTRY_BYTES * 65536, 4 << 20);
    let mut i = Index::new();
    i.insert(IndexEntry {
        key: [1; 32],
        phys_bytes: (1 << 33) + 4096,
        mtime_ns: 1,
        payload_len: 1 << 33,
        flags: 0,
    });
    assert_eq!(i.get(&[1; 32]).unwrap().payload_len, 1u64 << 33);
}
