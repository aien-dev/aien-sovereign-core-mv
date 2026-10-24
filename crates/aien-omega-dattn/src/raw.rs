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

//! Raw cache handle for sibling FFI crates (OWN-30B whole-prompt prefill, W-L1G-R, 2026-10-14). New file.
//!
//! omega's prefill attention (`omega_gpu_prefill_attn`, bound in `aien-omega-pf`) appends to and
//! reads the SAME KV cache the decode attention owns, so it needs the C handle of a
//! [`DecodeAttnCache`]. This child module can read the private field; nothing else changes.
use super::DecodeAttnCache;

impl DecodeAttnCache {
    /// The `OmegaGpuDecodeAttnCache *` behind this cache, or null in the stub build.
    ///
    /// Taking `&mut self` ties the handle's use to an exclusive borrow: the caller must not keep
    /// the pointer past that borrow, and must not free it (the cache's `Drop` does).
    pub fn raw_handle(&mut self) -> *mut std::ffi::c_void {
        #[cfg(has_omega_dattn)]
        {
            self.ptr.cast()
        }
        #[cfg(not(has_omega_dattn))]
        {
            std::ptr::null_mut()
        }
    }
}
