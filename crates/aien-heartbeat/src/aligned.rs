use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

use crate::IO_ALIGN;

/// Heap buffer aligned to [`IO_ALIGN`] with a length rounded up to it, as
/// O_DIRECT requires.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: plain owned heap memory; aliasing is controlled by the owner.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = round_up(len.max(1), IO_ALIGN);
        let layout = Layout::from_size_align(len, IO_ALIGN).expect("valid layout");
        // SAFETY: layout has non-zero size.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, layout }
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for layout.size() initialised bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and &mut self gives exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

pub fn round_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

pub fn round_down(v: u64, a: u64) -> u64 {
    v / a * a
}
