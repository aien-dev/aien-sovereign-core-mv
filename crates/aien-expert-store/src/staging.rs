//! Caller-provided staging buffers: 4096-aligned (so the direct read mode can
//! use them), zeroed at allocation and scrubbed on drop.
//!
//! A buffer handed to the adapter is owned by the job until its terminal
//! completion, so the caller cannot observe bytes that have not been verified.
//! On failure or cancellation the adapter scrubs it before handing it back.

use crate::format::ALIGN;
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;
use zeroize::Zeroize;

pub struct StagingBuffer {
    ptr: NonNull<u8>,
    cap: usize,
}

// SAFETY: the buffer is a uniquely owned heap allocation (like `Box<[u8]>`);
// no interior aliasing is shared across threads.
unsafe impl Send for StagingBuffer {}
// SAFETY: shared references only give `&[u8]`.
unsafe impl Sync for StagingBuffer {}

impl StagingBuffer {
    /// A zeroed buffer of at least `cap` bytes, rounded up to a 4096 multiple.
    /// `None` when `cap` is zero or the rounded size overflows.
    pub fn new(cap: usize) -> Option<Self> {
        if cap == 0 {
            return None;
        }
        let align = ALIGN as usize;
        let cap = cap.checked_add(align - 1)? / align * align;
        let layout = Layout::from_size_align(cap, align).ok()?;
        // SAFETY: layout has a nonzero size.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        Some(Self { ptr, cap })
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for cap initialised bytes for the life of self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: unique access through &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }

    /// Overwrites every byte with zero (not elided by the optimiser).
    pub fn scrub(&mut self) {
        self.as_mut_slice().zeroize();
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        self.scrub();
        // SAFETY: allocated in `new` with this exact layout.
        unsafe {
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.cap, ALIGN as usize),
            )
        };
    }
}

impl core::fmt::Debug for StagingBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "StagingBuffer({} bytes)", self.cap)
    }
}
