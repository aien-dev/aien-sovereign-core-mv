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

//! Shared fixtures for the fake-adapter and (later) worker tests: a tiny
//! geometry whose objects cost `OBJ` bytes of quota, leaked "pool" memory with
//! pinned views over it, and a rig that wires them to a `FakeKvStore`.
#![allow(dead_code)]

use aien_kv_store::fake::{DiskEntry, FakeConfig, FakeKvStore, JobScript};
use aien_kv_store::{
    chunk_chain, BackendFormat, ChunkHeader, ChunkKey, Completion, HolderGuard, HolderHandle,
    Incarnation, JobId, KvDtype, KvStore, LayoutStrides, NamespaceFields, NamespaceId, PinnedDest,
    PinnedSource,
};
use aien_platform::PlatformError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Bytes of one pool block in the tiny geometry.
pub const BLOCK: usize = 64;
/// Payload of a full chunk: 2 blocks.
pub const PAYLOAD: u64 = 2 * BLOCK as u64;
/// Quota cost of one full-chunk object: payload plus the 4096-byte header region.
pub const OBJ: u64 = PAYLOAD + 4096;

/// 4-token blocks of 64 bytes, 8-token chunks (2 blocks per full chunk).
pub fn tiny() -> NamespaceFields<'static> {
    let mut f = super::sample();
    f.block_size = 4;
    f.chunk_tokens = 8;
    f.layout = LayoutStrides {
        head_stride: 2,
        token_stride: 4,
        kv_plane_stride: 16,
        layer_stride: 32,
        block_stride: BLOCK as u64,
        element_bytes: 2,
    };
    f.dtype = KvDtype::Bf16;
    f.backend_format = BackendFormat::HostPagedPoolV1;
    f
}

/// Leaked memory standing in for the KV pool (it must outlive any raw pointer
/// handed to a pinned view; tests are short, the leak is fine).
#[derive(Clone, Copy)]
pub struct Mem {
    ptr: *mut u8,
    pub blocks: usize,
}

unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

impl Mem {
    pub fn new(blocks: usize, fill: u8) -> Mem {
        let b: Box<[u8]> = vec![fill; blocks * BLOCK].into_boxed_slice();
        Mem {
            ptr: Box::leak(b).as_mut_ptr(),
            blocks,
        }
    }

    pub fn fill_with(&self, f: impl Fn(usize) -> u8) {
        for i in 0..self.blocks * BLOCK {
            unsafe { *self.ptr.add(i) = f(i) };
        }
    }

    pub fn fill(&self, v: u8) {
        self.fill_with(|_| v);
    }

    pub fn bytes(&self) -> Vec<u8> {
        (0..self.blocks * BLOCK)
            .map(|i| unsafe { *self.ptr.add(i) })
            .collect()
    }

    /// A read-only view of blocks `0..n` with a detached holder. The handle
    /// stands for the owner: `revoke()` is "the owner freed the blocks".
    pub fn source(&self, n: usize, tail_valid_tokens: u32) -> (PinnedSource, HolderHandle) {
        self.source_with(n, tail_valid_tokens, HolderGuard::detached())
    }

    pub fn source_with(
        &self,
        n: usize,
        tail_valid_tokens: u32,
        guard: HolderGuard,
    ) -> (PinnedSource, HolderHandle) {
        let h = guard.handle();
        let s = unsafe {
            PinnedSource::new(
                self.ptr as *const u8,
                BLOCK,
                (0..n).collect(),
                tail_valid_tokens,
                guard,
            )
        };
        (s, h)
    }

    pub fn dest(&self, n: usize) -> (PinnedDest, HolderHandle) {
        self.dest_with(n, HolderGuard::detached())
    }

    pub fn dest_with(&self, n: usize, guard: HolderGuard) -> (PinnedDest, HolderHandle) {
        let h = guard.handle();
        let d = unsafe { PinnedDest::new(self.ptr, BLOCK, (0..n).collect(), guard) };
        (d, h)
    }
}

/// Deterministic payload bytes for object number `n`.
pub fn pattern(n: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| n.wrapping_mul(31).wrapping_add(i as u8))
        .collect()
}

pub struct Rig {
    pub fake: FakeKvStore,
    pub fields: NamespaceFields<'static>,
    pub ns: NamespaceId,
}

pub fn rig() -> Rig {
    rig_with(|_| {})
}

pub fn rig_with(f: impl FnOnce(&mut FakeConfig)) -> Rig {
    let fields = tiny();
    let ns = fields.namespace_id().unwrap();
    let mut cfg = FakeConfig::new(ns);
    f(&mut cfg);
    Rig {
        fake: FakeKvStore::new(cfg),
        fields,
        ns,
    }
}

impl Rig {
    /// Key and header of the first (full, 8-token) chunk of a token stream
    /// seeded by `n`; distinct `n` give distinct keys.
    pub fn chunk(&self, n: u32) -> (ChunkKey, ChunkHeader) {
        self.chunk_of(n, 8)
    }

    /// Same with `token_count` tokens (a partial chunk when below 8).
    pub fn chunk_of(&self, n: u32, token_count: u32) -> (ChunkKey, ChunkHeader) {
        let tokens: Vec<u32> = (0..token_count).map(|i| n * 1000 + i).collect();
        let key = chunk_chain(&self.ns, 8, &tokens).unwrap()[0];
        let h = self
            .fields
            .chunk_header(&self.ns, key, None, 0, token_count, 1)
            .unwrap();
        (key, h)
    }

    /// Put object `n` in place without a job.
    pub fn inject(&self, n: u32) -> ChunkKey {
        let (k, h) = self.chunk(n);
        self.fake
            .inject_object(h, pattern(n as u8, PAYLOAD as usize), BLOCK);
        k
    }

    /// Submit a store of object `n` from fresh memory; returns the holder handle.
    pub fn store(&self, n: u32) -> Result<JobId, PlatformError> {
        self.store_h(n).0
    }

    pub fn store_h(&self, n: u32) -> (Result<JobId, PlatformError>, HolderHandle, Mem) {
        let (_, h) = self.chunk(n);
        let mem = Mem::new(2, 0);
        mem.fill_with(|i| pattern(n as u8, PAYLOAD as usize)[i]);
        let (src, handle) = mem.source(2, 4);
        (
            self.fake.submit_store(h, src, Incarnation::seq(n as u64)),
            handle,
            mem,
        )
    }

    /// Submit a load of object `n` into fresh poisoned memory.
    pub fn load(&self, n: u32) -> (Result<JobId, PlatformError>, Mem, HolderHandle) {
        let (k, h) = self.chunk(n);
        let mem = Mem::new(2, 0xEE);
        let (dst, handle) = mem.dest(2);
        (
            self.fake
                .submit_load(k, &h, dst, Incarnation::seq(1000 + n as u64)),
            mem,
            handle,
        )
    }

    pub fn script(&self, s: JobScript) {
        self.fake.script(s);
    }

    pub fn poll_all(&self) -> Vec<Completion> {
        let mut out = Vec::new();
        loop {
            let v = self.fake.poll(7);
            if v.is_empty() {
                return out;
            }
            out.extend(v);
        }
    }

    /// Run the clock to idle and return every completion.
    pub fn settle(&self) -> Vec<Completion> {
        self.fake.run_until_idle();
        self.poll_all()
    }
}

/// A holder guard that counts its release and records whether the store's
/// lock was free at that moment (it must be, ADR 8).
pub fn counting_guard(
    fake: Arc<FakeKvStore>,
    released: Arc<AtomicU64>,
    lock_free_at_release: Arc<AtomicU64>,
) -> HolderGuard {
    HolderGuard::new(move || {
        released.fetch_add(1, Ordering::SeqCst);
        if fake.lock_is_free() {
            lock_free_at_release.fetch_add(1, Ordering::SeqCst);
        }
    })
}

pub fn disk_object(rig: &Rig, n: u32, ours: bool) -> DiskEntry {
    let (_, mut h) = rig.chunk(n);
    if !ours {
        h.namespace_id = NamespaceId([0x77; 32]);
    }
    DiskEntry::Object {
        header: h,
        payload: pattern(n as u8, PAYLOAD as usize),
        segment_bytes: BLOCK,
    }
}
