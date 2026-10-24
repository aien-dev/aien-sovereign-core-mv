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

//! Linux Unified Buffer implementation for Grace Blackwell GB10 and Linux host.
//! Allocates anonymous page-aligned mmap; GB10 shares one coherent system memory pool (ATS).

use aien_platform::{BufferLayout, DeviceAddress, PlatformError, UnifiedBuffer};
use core::mem::align_of;
use core::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxMemoryKind {
    AtsSystem,
    HmmSystem,
    /// Memory another allocator owns (the GPU engine's uncached, host-visible pool): released
    /// through its callback when the buffer drops, never unmapped here.
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyPolicy {
    Pageable,
    FaultIn,
    Locked,
}

pub struct LinuxUnifiedBuffer {
    ptr: NonNull<u8>,
    mapping_base: NonNull<u8>,
    mapping_len: usize,
    len: usize,
    align: usize,
    kind: LinuxMemoryKind,
    device_address: DeviceAddress,
    /// `External` memory: runs once on drop instead of the unmap.
    release: Option<Box<dyn FnOnce() + Send + Sync>>,
}

unsafe impl Send for LinuxUnifiedBuffer {}
unsafe impl Sync for LinuxUnifiedBuffer {}

impl LinuxUnifiedBuffer {
    pub fn allocate(
        layout: BufferLayout,
        kind: LinuxMemoryKind,
        policy: ResidencyPolicy,
    ) -> Result<Self, PlatformError> {
        if layout.size == 0 {
            return Err(PlatformError::InvalidLayout);
        }

        let align = layout.align.max(align_of::<usize>()).max(64);
        let mapping_len = layout
            .size
            .checked_add(align)
            .ok_or(PlatformError::OutOfMemory)?;

        let mapping_base_ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                mapping_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapping_base_ptr == libc::MAP_FAILED || mapping_base_ptr.is_null() {
            return Err(PlatformError::OutOfMemory);
        }
        let mapping_base_ptr = mapping_base_ptr as *mut u8;
        let actual_kind = kind;

        let mapping_base = NonNull::new(mapping_base_ptr).ok_or(PlatformError::OutOfMemory)?;
        let base_addr = mapping_base_ptr as usize;
        let aligned_addr = (base_addr + align - 1) & !(align - 1);
        let ptr = NonNull::new(aligned_addr as *mut u8).ok_or(PlatformError::OutOfMemory)?;

        if policy == ResidencyPolicy::FaultIn {
            let page_size = 4096;
            for offset in (0..layout.size).step_by(page_size) {
                unsafe {
                    core::ptr::write_volatile(ptr.as_ptr().add(offset), 0);
                }
            }
        }

        let device_address = DeviceAddress(aligned_addr as u64);

        Ok(Self {
            ptr,
            mapping_base,
            mapping_len,
            len: layout.size,
            align,
            kind: actual_kind,
            device_address,
            release: None,
        })
    }

    /// A buffer over memory another allocator owns (`LinuxMemoryKind::External`): `len` bytes at
    /// `ptr`, reachable by the device at `device_address`. `release` runs exactly once, when the
    /// buffer drops; nothing is unmapped here.
    ///
    /// # Safety
    /// `ptr` must be non-null and stay valid, readable and writable for `len` bytes until
    /// `release` runs, and nothing else may free it.
    pub unsafe fn from_external(
        ptr: *mut u8,
        len: usize,
        device_address: DeviceAddress,
        release: Box<dyn FnOnce() + Send + Sync>,
    ) -> Result<Self, PlatformError> {
        if len == 0 {
            return Err(PlatformError::InvalidLayout);
        }
        let ptr = NonNull::new(ptr).ok_or(PlatformError::InvalidLayout)?;
        let align = 1usize << (ptr.as_ptr() as usize).trailing_zeros().min(12);
        Ok(Self {
            ptr,
            mapping_base: ptr,
            mapping_len: len,
            len,
            align,
            kind: LinuxMemoryKind::External,
            device_address,
            release: Some(release),
        })
    }

    pub fn kind(&self) -> LinuxMemoryKind {
        self.kind
    }

    pub fn alignment(&self) -> usize {
        self.align
    }
}

impl UnifiedBuffer for LinuxUnifiedBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn device_address(&self) -> DeviceAddress {
        self.device_address
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for LinuxUnifiedBuffer {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
            return;
        }
        unsafe {
            libc::munmap(
                self.mapping_base.as_ptr() as *mut libc::c_void,
                self.mapping_len,
            );
        }
    }
}
