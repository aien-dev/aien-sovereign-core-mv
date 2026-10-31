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

//! The fused attention path of the Omega engine (OWN-30B cut 9): omega's one-launch decode
//! attention (`aien-omega-dattn`) as a [`KvAttention`]. One launch per layer per token does the
//! q/k per-head norms, RoPE, the bf16 K/V append and the attention over the whole context on
//! the chip, from the raw fused q|k|v projection row, so the host work between the q/k/v GEMV
//! and the o GEMV disappears.
//!
//! The cache owns the KV of every layer by position, so prefix reuse works as before:
//! `truncate(len)` forgets positions at or past `len` and the next step rewrites them.
//! `append` / `attend` (the split path) are refused: the engine must take [`KvAttention::fused_attend`].
use crate::qwen3_moe_omega::{kvio, KvAttention, KvGeometry, KvStore, OmegaLayer, KV_BLOCK_SIZE};

/// Omega's one-launch decode attention with its GPU-resident KV, one sequence.
pub struct OmegaFusedKv {
    cache: aien_omega_dattn::DecodeAttnCache,
    capacity: usize,
    geometry: KvGeometry,
    filled: Vec<usize>,
}

impl OmegaFusedKv {
    /// Whether the kernel's fixed shape matches `geometry` (32 query heads, 4 KV heads, 128 wide).
    pub fn fits(geometry: &KvGeometry) -> bool {
        geometry.q_heads == aien_omega_dattn::Q_HEADS
            && geometry.kv_heads == aien_omega_dattn::KV_HEADS
            && geometry.head_dim == aien_omega_dattn::HEAD_DIM
    }

    /// Allocates the cache for `max_context` tokens (rounded up to 256) and uploads every
    /// layer's q_norm / k_norm weights.
    pub fn new(
        geometry: &KvGeometry,
        max_context: usize,
        layers: &[OmegaLayer],
    ) -> Result<Self, String> {
        if !aien_omega_dattn::is_native() {
            return Err("this binary has no decode attention linked (omega before cut 8)".into());
        }
        if !Self::fits(geometry) {
            return Err(format!(
                "fused attention needs 32 q heads x 4 kv heads x 128, config has {} x {} x {}",
                geometry.q_heads, geometry.kv_heads, geometry.head_dim
            ));
        }
        if geometry.block_size != KV_BLOCK_SIZE {
            return Err(format!(
                "fused attention: block size {} is not the payload's ({KV_BLOCK_SIZE})",
                geometry.block_size
            ));
        }
        if layers.len() != geometry.layers {
            return Err(format!(
                "fused attention: {} layers of weights, geometry says {}",
                layers.len(),
                geometry.layers
            ));
        }
        let max_pos = aien_omega_dattn::round_max_pos(max_context);
        let mut cache = aien_omega_dattn::DecodeAttnCache::alloc(
            geometry.layers,
            max_pos,
            geometry.rope_theta as f64,
        )
        .map_err(|e| {
            format!(
                "decode attn cache alloc ({} layers x {max_pos}): {e}",
                geometry.layers
            )
        })?;
        for (l, layer) in layers.iter().enumerate() {
            cache
                .set_norms(l, &layer.q_norm, &layer.k_norm)
                .map_err(|e| format!("decode attn norms layer {l}: {e}"))?;
        }
        Ok(Self {
            cache,
            capacity: max_pos,
            geometry: *geometry,
            filled: vec![0; geometry.layers],
        })
    }

    /// Bytes the chip holds for the K and V planes.
    pub fn kv_bytes(&self) -> usize {
        aien_omega_dattn::kv_bytes(self.cache.layers(), self.cache.max_pos())
    }

    /// omega's prefill attention over this cache (whole-prompt prefill appends K/V for every layer).
    pub fn prefill_attention(&mut self) -> aien_omega_pf::OmegaPrefillAttention<'_> {
        aien_omega_pf::OmegaPrefillAttention::new(&mut self.cache)
    }

    /// Marks positions `0..upto` stored in every layer (after a whole-prompt prefill).
    pub fn mark_filled(&mut self, upto: usize) {
        self.filled.iter_mut().for_each(|f| *f = upto);
    }

    /// The payload geometry of this cache (sc#505): every layer, omega's 4 KV heads x 128.
    pub fn pool_geometry(&self) -> kvio::PoolBlockGeometry {
        kvio::PoolBlockGeometry {
            layers: self.filled.len(),
            kv_heads: aien_omega_dattn::KV_HEADS,
            head_dim: aien_omega_dattn::HEAD_DIM,
            block_size: KV_BLOCK_SIZE,
        }
    }
}

impl KvStore for OmegaFusedKv {
    fn geometry(&self) -> KvGeometry {
        self.geometry
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn len(&self) -> usize {
        self.filled.iter().copied().min().unwrap_or(0)
    }
    fn path_name(&self) -> &'static str {
        "fused"
    }
    fn truncate(&mut self, len: usize) {
        // The cache keeps its bytes; positions past `len` are rewritten by later steps. Those
        // steps write the cache only, so omega's prefill-attention mirror must drop its copies
        // of the forgotten rows first, or a later whole-prompt pass reads the old sequence's
        // K/V there (mirror contract, omega_gpu_prefill_attn_api.h). Every cache write is at a
        // layer's `filled` position, so this one call covers decode, the step_batch fallback
        // and imports after a rewind.
        if let Some(from) = rewritten_from(&self.filled, len) {
            aien_omega_pf::kv_mirror_invalidate(&mut self.cache, from);
        }
        for f in self.filled.iter_mut() {
            *f = (*f).min(len);
        }
    }
    // Persistent KV (sc#505): omega's row helpers move whole bf16 rows between this cache and the
    // `HostPagedPoolV1` payload (`kvio`), no value is converted. Called between generations only.
    fn export_blocks(&mut self, start: usize, len: usize) -> Option<Result<Vec<u8>, String>> {
        if let Some((l, &f)) = self
            .filled
            .iter()
            .enumerate()
            .find(|(_, &f)| f < start + len)
        {
            return Some(Err(format!(
                "fused kv export {start}+{len}: layer {l} holds only {f} positions"
            )));
        }
        let g = self.pool_geometry();
        let cache = &mut self.cache;
        Some(kvio::rows_to_blocks(&g, start, len, |l, h, p, k, v| {
            cache
                .read_row(l, h, p, k, v)
                .map_err(|e| format!("fused kv export layer {l} head {h} pos {p}: {e}"))
        }))
    }
    fn import_blocks(
        &mut self,
        start: usize,
        len: usize,
        bytes: &[u8],
    ) -> Option<Result<(), String>> {
        if let Some((l, &f)) = self.filled.iter().enumerate().find(|(_, &f)| f != start) {
            return Some(Err(format!(
                "fused kv import at {start}: layer {l} holds {f} positions"
            )));
        }
        if start + len > self.capacity {
            return Some(Err(format!(
                "fused kv import {start}+{len}: capacity {}",
                self.capacity
            )));
        }
        let g = self.pool_geometry();
        let cache = &mut self.cache;
        let r = kvio::blocks_to_rows(&g, start, len, bytes, |l, h, k, v| {
            cache
                .write_rows(l, h, start, k, v)
                .map_err(|e| format!("fused kv import layer {l} head {h}: {e}"))
        });
        if r.is_ok() {
            self.filled.iter_mut().for_each(|f| *f = start + len);
        }
        Some(r)
    }
}

impl KvAttention for OmegaFusedKv {
    fn append(&mut self, _layer: usize, pos: usize, _k: &[f32], _v: &[f32]) -> Result<(), String> {
        Err(format!(
            "fused kv: split append at {pos} is not a path here (use fused_attend)"
        ))
    }
    fn attend(
        &mut self,
        _layer: usize,
        _q: &[f32],
        context_len: usize,
        _out: &mut [f32],
    ) -> Result<u64, String> {
        Err(format!(
            "fused kv: split attend over {context_len} is not a path here (use fused_attend)"
        ))
    }
    fn fused_attend(
        &mut self,
        layer: usize,
        pos: usize,
        qkv: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Option<Result<u64, String>> {
        if pos != self.filled[layer] || pos >= self.capacity {
            return Some(Err(format!(
                "fused kv: step at {pos}, have {} (capacity {})",
                self.filled[layer], self.capacity
            )));
        }
        // out_scale 1.0: the linked FP8 GEMV takes a plain f32 row (no x_prescaled entry in
        // omega_gpu_fp8_api.h at own30b/integration 831001cc).
        let r = self
            .cache
            .attend(layer, pos, qkv, eps, 1.0, out)
            .map(|info| info.elapsed_ns)
            .map_err(|e| format!("decode attn layer {layer} pos {pos}: {e}"));
        if r.is_ok() {
            self.filled[layer] = pos + 1;
        }
        Some(r)
    }
    fn resident(&self) -> bool {
        true
    }
    fn fused_kv(&mut self) -> Option<&mut OmegaFusedKv> {
        Some(self)
    }
    fn dgraph_cache(&mut self, pos: usize) -> Option<Result<*mut std::ffi::c_void, String>> {
        if pos >= self.capacity {
            return Some(Err(format!(
                "fused kv (decode graph): step at {pos}, capacity {}",
                self.capacity
            )));
        }
        if let Some((l, &f)) = self.filled.iter().enumerate().find(|(_, &f)| f != pos) {
            return Some(Err(format!(
                "fused kv (decode graph): step at {pos}, layer {l} holds {f} positions"
            )));
        }
        Some(Ok(self.cache.as_raw()))
    }
    fn dgraph_commit(&mut self, pos: usize) {
        self.filled.iter_mut().for_each(|f| *f = pos + 1);
    }
}

impl Drop for OmegaFusedKv {
    fn drop(&mut self) {
        // Mirrors are keyed by the layer's K plane address: a later cache allocated there would
        // inherit stale rows. Runs before the cache field frees the planes.
        aien_omega_pf::kv_mirror_forget(&mut self.cache);
    }
}

/// The first position a truncate to `len` forgets in any layer, if it forgets one: rows from
/// there on will be rewritten, so mirrored copies of them go stale.
fn rewritten_from(filled: &[usize], len: usize) -> Option<usize> {
    filled.iter().any(|&f| f > len).then_some(len)
}

#[cfg(test)]
mod tests {
    use super::rewritten_from;

    #[test]
    fn a_rewind_below_the_stored_rows_invalidates_from_the_kept_prefix() {
        // A: whole-prompt prefill of 40 tokens + 8 decoded; B keeps a 12-token prefix.
        assert_eq!(rewritten_from(&[48; 4], 12), Some(12));
        // One layer ahead of the others (mid-step) still forgets rows.
        assert_eq!(rewritten_from(&[30, 30, 31, 30], 30), Some(30));
        assert_eq!(rewritten_from(&[48; 4], 0), Some(0));
    }

    #[test]
    fn keeping_every_stored_row_invalidates_nothing() {
        assert_eq!(rewritten_from(&[48; 4], 48), None);
        assert_eq!(rewritten_from(&[48; 4], 100), None);
        assert_eq!(rewritten_from(&[], 0), None);
    }
}
