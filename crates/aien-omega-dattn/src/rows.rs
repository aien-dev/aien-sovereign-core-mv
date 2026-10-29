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

//! Row access to the decode attention cache (persistent KV park/restore, sc#505). New file.
//!
//! omega keeps each layer's K and V planes as bf16 rows of [`HEAD_DIM`] values, row
//! `(kv_head, pos)` at `(kv_head * max_pos + pos) * 256` bytes into the plane
//! (`omega_gpu_decode_attn_api.h`, `OmegaGpuDecodeAttnCacheView`). The two calls here are the
//! header's test helpers: `cache_read` returns one K row and one V row, `cache_write` stores
//! `count` consecutive rows of one layer and KV head in one sync. Both take omega's session lock
//! and sync the layer between host and chip, so they must not run while a launch on this cache
//! is in flight (the engine calls them between generations, under the hive's GPU lock).
//! Without the native library both return `Unavailable`.
use super::{bad, check_len, DecodeAttnCache, HEAD_DIM, KV_HEADS};
use aien_omega_gpu::OmegaGpuError;

impl DecodeAttnCache {
    fn check_row(
        &self,
        layer: usize,
        kv_head: usize,
        pos: usize,
        count: usize,
    ) -> Result<(), OmegaGpuError> {
        if layer >= self.layers
            || kv_head >= KV_HEADS
            || count == 0
            || pos >= self.max_pos
            || count > self.max_pos - pos
        {
            return Err(bad(format!(
                "decode attn rows layer {layer} head {kv_head} pos {pos} count {count} outside {} layers x {KV_HEADS} heads x {} positions",
                self.layers, self.max_pos
            )));
        }
        Ok(())
    }

    /// Copies the bf16 K and V rows of (`layer`, `kv_head`, `pos`) into `k` and `v`
    /// ([`HEAD_DIM`] values each, bf16 bit patterns).
    pub fn read_row(
        &mut self,
        layer: usize,
        kv_head: usize,
        pos: usize,
        k: &mut [u16],
        v: &mut [u16],
    ) -> Result<(), OmegaGpuError> {
        check_len("decode attn read k", HEAD_DIM, k.len())?;
        check_len("decode attn read v", HEAD_DIM, v.len())?;
        self.check_row(layer, kv_head, pos, 1)?;
        #[cfg(has_omega_dattn)]
        {
            // SAFETY: ranges and lengths checked above; the cache is live for &mut self.
            let rc = unsafe {
                super::ffi::omega_gpu_decode_attn_cache_read(
                    self.ptr,
                    layer as u32,
                    kv_head as u32,
                    pos as u32,
                    k.as_mut_ptr(),
                    v.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return Err(super::rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dattn))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Stores `k.len() / HEAD_DIM` consecutive bf16 K rows and as many V rows of (`layer`,
    /// `kv_head`) starting at position `pos`, in one sync.
    pub fn write_rows(
        &mut self,
        layer: usize,
        kv_head: usize,
        pos: usize,
        k: &[u16],
        v: &[u16],
    ) -> Result<(), OmegaGpuError> {
        if k.is_empty() || !k.len().is_multiple_of(HEAD_DIM) {
            return Err(bad(format!(
                "decode attn write: {} K values is not a whole number of {HEAD_DIM}-wide rows",
                k.len()
            )));
        }
        check_len("decode attn write v", k.len(), v.len())?;
        let count = k.len() / HEAD_DIM;
        self.check_row(layer, kv_head, pos, count)?;
        #[cfg(has_omega_dattn)]
        {
            // SAFETY: ranges and lengths checked above; the cache is live for &mut self.
            let rc = unsafe {
                super::ffi::omega_gpu_decode_attn_cache_write(
                    self.ptr,
                    layer as u32,
                    kv_head as u32,
                    pos as u32,
                    count as u32,
                    k.as_ptr(),
                    v.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(super::rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dattn))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }
}
