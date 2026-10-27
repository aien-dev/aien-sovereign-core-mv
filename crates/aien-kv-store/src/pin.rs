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

//! Pinned views of pool blocks handed to a store job (ADR 0003 section 3.2).
//!
//! A `PinnedSource` / `PinnedDest` names blocks of one pool allocation and
//! carries a `HolderGuard`: the storage reference (a holder block table,
//! ADR section 5.2) that keeps those blocks alive. The store reads or writes
//! blocks only through the checked accessors below, which refuse once the
//! holder has been released, so a use-after-release becomes a failed job
//! (`PlatformError::InvalidLayout`) instead of a stray memory access.

use aien_platform::PlatformError;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

/// Keeps a holder alive. Dropping it runs the release callback exactly once
/// (in the runtime: `free_sequence(holder)`, ADR critical section S4).
pub struct HolderGuard {
    held: Arc<AtomicBool>,
    /// Set on a lease: the guard it was lent from. A lease reports held only
    /// while that guard is held too, so a revoke of the owner reaches the
    /// borrower.
    parent: Option<Arc<AtomicBool>>,
    release: Option<Box<dyn FnOnce() + Send>>,
}

/// A cloneable view of a guard's liveness, for the owner and for tests.
#[derive(Clone)]
pub struct HolderHandle {
    held: Arc<AtomicBool>,
    parent: Option<Arc<AtomicBool>>,
}

impl HolderGuard {
    pub fn new(release: impl FnOnce() + Send + 'static) -> Self {
        Self {
            held: Arc::new(AtomicBool::new(true)),
            parent: None,
            release: Some(Box::new(release)),
        }
    }

    /// A guard with nothing to release (tests, and sources the caller owns outright).
    pub fn detached() -> Self {
        Self {
            held: Arc::new(AtomicBool::new(true)),
            parent: None,
            release: None,
        }
    }

    /// A lease on the same blocks for a borrower (the storage worker lends
    /// the caller's blocks to the inner store this way). Dropping the lease
    /// runs `on_return`, never this guard's release: the owner keeps the
    /// holder until it decides. The lease is held only while both it and this
    /// guard are held.
    pub fn lease(&self, on_return: impl FnOnce() + Send + 'static) -> HolderGuard {
        HolderGuard {
            held: Arc::new(AtomicBool::new(true)),
            parent: Some(Arc::clone(&self.held)),
            release: Some(Box::new(on_return)),
        }
    }

    pub fn handle(&self) -> HolderHandle {
        HolderHandle {
            held: Arc::clone(&self.held),
            parent: self.parent.clone(),
        }
    }
}

// SAFETY: `release` is touched only through `&mut self` (in `drop`); every
// `&self` method reads the atomics behind `Arc`s. Sharing `&HolderGuard`
// across threads therefore exposes nothing that is not already `Sync`.
unsafe impl Sync for HolderGuard {}

impl Drop for HolderGuard {
    fn drop(&mut self) {
        self.held.store(false, Ordering::SeqCst);
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

impl HolderHandle {
    pub fn is_held(&self) -> bool {
        self.held.load(Ordering::SeqCst)
            && self
                .parent
                .as_ref()
                .is_none_or(|p| p.load(Ordering::SeqCst))
    }

    /// Mark the holder as released out of band, the bug case of ADR 3.2: the
    /// owner freed the blocks while a job still referenced them. Every later
    /// block access through the pinned view fails.
    pub fn revoke(&self) {
        self.held.store(false, Ordering::SeqCst);
    }
}

fn block_range(
    blocks: &[usize],
    block_bytes: usize,
    i: usize,
) -> Result<(usize, usize), PlatformError> {
    let block = *blocks.get(i).ok_or(PlatformError::InvalidLayout)?;
    let start = block
        .checked_mul(block_bytes)
        .ok_or(PlatformError::InvalidLayout)?;
    Ok((start, block_bytes))
}

/// Read-only view of pinned pool blocks (store jobs).
pub struct PinnedSource {
    base: *const u8,
    block_bytes: usize,
    blocks: Vec<usize>,
    tail_valid_tokens: u32,
    guard: HolderGuard,
}

// SAFETY: the constructor's contract makes the memory behind `base` valid and
// immutable for as long as the guard is held, from any thread.
unsafe impl Send for PinnedSource {}

impl PinnedSource {
    /// `tail_valid_tokens` is the number of token rows of the last block that
    /// hold real positions; the writer zeroes the rows at or above it in the
    /// stored object (ADR 2.2, 13 T11). Use the block size for a full block.
    ///
    /// # Safety
    /// For every `b` in `blocks`, `base + b * block_bytes .. + block_bytes`
    /// must be readable memory that is not written by anyone while `guard`
    /// (or the handle derived from it) reports held.
    pub unsafe fn new(
        base: *const u8,
        block_bytes: usize,
        blocks: Vec<usize>,
        tail_valid_tokens: u32,
        guard: HolderGuard,
    ) -> Self {
        Self {
            base,
            block_bytes,
            blocks,
            tail_valid_tokens,
            guard,
        }
    }

    /// Token rows of the last block that are real (ADR 13 T11).
    pub fn tail_valid_tokens(&self) -> u32 {
        self.tail_valid_tokens
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Total bytes, or `None` on overflow.
    pub fn total_bytes(&self) -> Option<u64> {
        (self.blocks.len() as u64).checked_mul(self.block_bytes as u64)
    }

    pub fn handle(&self) -> HolderHandle {
        self.guard.handle()
    }

    /// The `i`-th block's bytes; refused after the holder was released.
    pub fn block(&self, i: usize) -> Result<&[u8], PlatformError> {
        if !self.guard.handle().is_held() {
            return Err(PlatformError::InvalidLayout);
        }
        let (start, len) = block_range(&self.blocks, self.block_bytes, i)?;
        // SAFETY: `new`'s contract covers this range while the holder is held,
        // which was checked just above.
        Ok(unsafe { core::slice::from_raw_parts(self.base.add(start), len) })
    }

    /// Give up the view and keep only the guard (the store drops it outside its lock).
    pub fn into_guard(self) -> HolderGuard {
        self.guard
    }

    /// Lend the view: the returned view carries a lease (see
    /// [`HolderGuard::lease`]) and the owner keeps the real guard.
    pub fn lend(mut self, on_return: impl FnOnce() + Send + 'static) -> (Self, HolderGuard) {
        let lease = self.guard.lease(on_return);
        let real = core::mem::replace(&mut self.guard, lease);
        (self, real)
    }
}

/// Writable view of pinned pool blocks (load jobs).
pub struct PinnedDest {
    base: *mut u8,
    block_bytes: usize,
    blocks: Vec<usize>,
    guard: HolderGuard,
}

// SAFETY: as for `PinnedSource`; additionally the holder gives the job
// exclusive write access to these blocks until it is released.
unsafe impl Send for PinnedDest {}

impl PinnedDest {
    /// # Safety
    /// For every `b` in `blocks`, `base + b * block_bytes .. + block_bytes`
    /// must be writable memory that nobody else reads or writes while `guard`
    /// (or the handle derived from it) reports held.
    pub unsafe fn new(
        base: *mut u8,
        block_bytes: usize,
        blocks: Vec<usize>,
        guard: HolderGuard,
    ) -> Self {
        Self {
            base,
            block_bytes,
            blocks,
            guard,
        }
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    pub fn total_bytes(&self) -> Option<u64> {
        (self.blocks.len() as u64).checked_mul(self.block_bytes as u64)
    }

    pub fn handle(&self) -> HolderHandle {
        self.guard.handle()
    }

    /// Copy `data` (exactly `block_bytes` long) into the `i`-th block; refused
    /// after the holder was released.
    pub fn write_block(&mut self, i: usize, data: &[u8]) -> Result<(), PlatformError> {
        if !self.guard.handle().is_held() || data.len() != self.block_bytes {
            return Err(PlatformError::InvalidLayout);
        }
        let (start, len) = block_range(&self.blocks, self.block_bytes, i)?;
        // SAFETY: `new`'s contract covers this range while the holder is held,
        // which was checked just above; `data` is a distinct borrowed slice.
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(start), len) };
        Ok(())
    }

    pub fn into_guard(self) -> HolderGuard {
        self.guard
    }

    /// Lend the view: the returned view carries a lease (see
    /// [`HolderGuard::lease`]) and the owner keeps the real guard.
    pub fn lend(mut self, on_return: impl FnOnce() + Send + 'static) -> (Self, HolderGuard) {
        let lease = self.guard.lease(on_return);
        let real = core::mem::replace(&mut self.guard, lease);
        (self, real)
    }
}
