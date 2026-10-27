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

//! Linux Compute Device implementation for Grace Blackwell GB10.
//!
//! No CUDA (FB-1 cut 6): GB10 memory is one unified LPDDR5x pool and, on ATS/HMM systems,
//! "all system memory is implicitly managed memory, requiring no special allocation"
//! (CUDA Programming Guide, Understanding Memory > Unified Memory). Buffers are plain
//! anonymous mappings; the Omega GPU engine stages what the chip reads into its own
//! allocations, so copies, zeroing and fences here are synchronous host operations.

use crate::buffer::{LinuxMemoryKind, LinuxUnifiedBuffer, ResidencyPolicy};
use aien_platform::{
    BufferLayout, BufferRegion, ComputeDevice, ComputeWork, Fence, MemoryDevice, PlatformError,
};
use core::sync::atomic::{AtomicU64, Ordering};

pub struct LinuxComputeDevice {
    device_id: u32,
    fence_counter: AtomicU64,
}

impl LinuxComputeDevice {
    pub fn new() -> Self {
        Self::with_device_id(0)
    }

    pub fn with_device_id(device_id: u32) -> Self {
        Self {
            device_id,
            fence_counter: AtomicU64::new(1),
        }
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn preferred_memory_kind(&self) -> LinuxMemoryKind {
        LinuxMemoryKind::AtsSystem
    }

    fn next_fence(&self) -> Fence {
        Fence(self.fence_counter.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for LinuxComputeDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for LinuxComputeDevice {
    type Buffer = LinuxUnifiedBuffer;

    fn alloc(&self, layout: BufferLayout) -> Result<Self::Buffer, PlatformError> {
        let kind = self.preferred_memory_kind();
        LinuxUnifiedBuffer::allocate(layout, kind, ResidencyPolicy::FaultIn)
    }

    fn submit(&self, _work: ComputeWork<'_>) -> Result<Fence, PlatformError> {
        Ok(self.next_fence())
    }

    fn synchronize(&self, _fence: Fence) -> Result<(), PlatformError> {
        Ok(())
    }
}

impl MemoryDevice for LinuxComputeDevice {
    fn copy(&self, src: BufferRegion, dst: BufferRegion) -> Result<Fence, PlatformError> {
        if src.len != dst.len {
            return Err(PlatformError::InvalidLayout);
        }
        if src.len != 0 {
            unsafe {
                core::ptr::copy(
                    src.address.0 as *const u8,
                    dst.address.0 as *mut u8,
                    src.len,
                );
            }
        }
        Ok(self.next_fence())
    }

    fn zero(&self, dst: BufferRegion) -> Result<Fence, PlatformError> {
        if dst.len != 0 {
            unsafe {
                core::ptr::write_bytes(dst.address.0 as *mut u8, 0, dst.len);
            }
        }
        Ok(self.next_fence())
    }
}
