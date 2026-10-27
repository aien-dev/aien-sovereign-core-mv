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

//! `LinuxUnifiedBuffer::from_external`: a buffer over memory another allocator owns releases
//! through its callback exactly once and never unmaps.

use aien_platform::{DeviceAddress, PlatformError, UnifiedBuffer};
use aien_platform_linux::{LinuxMemoryKind, LinuxUnifiedBuffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn external_buffer_releases_once_through_its_callback() {
    let mut backing = vec![7u8; 4096];
    let released = Arc::new(AtomicUsize::new(0));
    let r = released.clone();
    let ptr = backing.as_mut_ptr();
    // SAFETY: `backing` outlives the buffer; nothing else frees it.
    let mut buf = unsafe {
        LinuxUnifiedBuffer::from_external(
            ptr,
            backing.len(),
            DeviceAddress(ptr as u64),
            Box::new(move || {
                r.fetch_add(1, Ordering::SeqCst);
            }),
        )
    }
    .expect("external buffer");
    assert_eq!(buf.kind(), LinuxMemoryKind::External);
    assert_eq!(buf.len(), 4096);
    assert_eq!(buf.as_ptr(), ptr as *const u8);
    assert_eq!(buf.device_address(), DeviceAddress(ptr as u64));
    // SAFETY: inside the 4096 bytes of `backing`.
    unsafe { *buf.as_mut_ptr().add(10) = 9 };
    assert_eq!(
        released.load(Ordering::SeqCst),
        0,
        "not released while alive"
    );
    drop(buf);
    assert_eq!(
        released.load(Ordering::SeqCst),
        1,
        "released exactly once on drop"
    );
    assert_eq!(backing[10], 9, "the writes went to the external memory");
}

#[test]
fn external_buffer_refuses_null_and_empty() {
    let mut backing = [0u8; 16];
    // SAFETY: a null pointer and a zero length are refused before any use.
    let null = unsafe {
        LinuxUnifiedBuffer::from_external(
            std::ptr::null_mut(),
            16,
            DeviceAddress(0),
            Box::new(|| {}),
        )
    };
    assert!(matches!(null, Err(PlatformError::InvalidLayout)));
    let empty = unsafe {
        LinuxUnifiedBuffer::from_external(
            backing.as_mut_ptr(),
            0,
            DeviceAddress(0),
            Box::new(|| {}),
        )
    };
    assert!(matches!(empty, Err(PlatformError::InvalidLayout)));
}
