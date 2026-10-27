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

//! Persistent KV export and import for the Omega engine (sc#505). New file.
//!
//! The payload is the persistent KV store's `HostPagedPoolV1` format (ADR 0003 section 2.2): for
//! positions `start..start + len`, `ceil(len / KV_BLOCK_SIZE)` blocks, each exactly
//! [`PoolBlockGeometry::block_stride`] bytes, laid out `[layer][K plane | V plane][token][kv
//! head][dim]` with every element a little-endian bf16 bit pattern. Rows at or past `len` in the
//! last block are zero. This is byte for byte the layout of `aien_kv_cache::UnifiedKvTensorPool`
//! with dtype bf16 and block size [`KV_BLOCK_SIZE`](super::KV_BLOCK_SIZE) (the pool
//! [`super::OmegaPagedKv`] attends over), so the split path copies its blocks as they are.
//!
//! The fused path ([`crate::qwen3_moe_omega_dattn::OmegaFusedKv`], what the live server runs) keeps
//! its K/V differently on the chip: per layer one K plane and one V plane of bf16 rows, row
//! `(kv_head, pos)` at `(kv_head * max_pos + pos) * head_dim * 2` bytes. [`rows_to_blocks`] and
//! [`blocks_to_rows`] move whole bf16 rows between that layout and the payload without touching a
//! value (no rounding, no conversion), so the payload is the same `HostPagedPoolV1` bytes either
//! way. Which attention path computed them is part of the engine identity the namespace binds
//! (aien-serve-http `kv_persist.rs`), so a fused payload is never offered to the split path or
//! the other way round.
//!
//! Both rows of a position are the cached K (after the k norm and RoPE) and V, exactly what the
//! attention reads; positions are absolute, so a restored prefix lines up with the next step.

use super::{KvGeometry, Qwen3MoeConfig};

/// The byte geometry of one `HostPagedPoolV1` block for a model (bf16 elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolBlockGeometry {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
}

/// Bytes of one bf16 element.
pub const ELEMENT_BYTES: usize = 2;

impl PoolBlockGeometry {
    pub fn for_geometry(g: &KvGeometry) -> Self {
        Self {
            layers: g.layers,
            kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            block_size: g.block_size,
        }
    }
    /// Shim until ADR 0004 step 5: the geometry of a 30B-family config.
    pub fn for_config(c: &Qwen3MoeConfig) -> Self {
        Self::for_geometry(&KvGeometry::from(c))
    }
    pub fn head_stride(&self) -> usize {
        self.head_dim * ELEMENT_BYTES
    }
    pub fn token_stride(&self) -> usize {
        self.kv_heads * self.head_stride()
    }
    pub fn kv_plane_stride(&self) -> usize {
        self.block_size * self.token_stride()
    }
    pub fn layer_stride(&self) -> usize {
        2 * self.kv_plane_stride()
    }
    pub fn block_stride(&self) -> usize {
        self.layers * self.layer_stride()
    }
    /// Payload bytes for `len` positions.
    pub fn payload_len(&self, len: usize) -> usize {
        len.div_ceil(self.block_size) * self.block_stride()
    }
    /// Offset of the row of (`pos` relative to the payload start, `layer`, K or V, `head`).
    pub fn row_offset(&self, rel_pos: usize, layer: usize, value: bool, head: usize) -> usize {
        (rel_pos / self.block_size) * self.block_stride()
            + layer * self.layer_stride()
            + usize::from(value) * self.kv_plane_stride()
            + (rel_pos % self.block_size) * self.token_stride()
            + head * self.head_stride()
    }
    fn check(&self, start: usize, len: usize) -> Result<(), String> {
        if self.layers == 0 || self.kv_heads == 0 || self.head_dim == 0 || self.block_size == 0 {
            return Err(format!("kv export geometry {self:?} has a zero dimension"));
        }
        if len == 0 {
            return Err("kv export of zero positions".into());
        }
        if !start.is_multiple_of(self.block_size) {
            return Err(format!(
                "kv export start {start} is not a multiple of the block size {}",
                self.block_size
            ));
        }
        Ok(())
    }
}

/// Builds the payload for positions `start..start + len` from a row reader:
/// `read_row(layer, kv_head, pos, k, v)` fills `head_dim` bf16 bit patterns of K and of V.
pub fn rows_to_blocks(
    g: &PoolBlockGeometry,
    start: usize,
    len: usize,
    mut read_row: impl FnMut(usize, usize, usize, &mut [u16], &mut [u16]) -> Result<(), String>,
) -> Result<Vec<u8>, String> {
    g.check(start, len)?;
    let mut out = vec![0u8; g.payload_len(len)];
    let (mut k, mut v) = (vec![0u16; g.head_dim], vec![0u16; g.head_dim]);
    for layer in 0..g.layers {
        for head in 0..g.kv_heads {
            for rel in 0..len {
                read_row(layer, head, start + rel, &mut k, &mut v)?;
                put_row(&mut out, g.row_offset(rel, layer, false, head), &k);
                put_row(&mut out, g.row_offset(rel, layer, true, head), &v);
            }
        }
    }
    Ok(out)
}

/// Splits a payload of `len` positions into contiguous per-(layer, kv head) rows and hands them
/// to `write_rows(layer, kv_head, k_rows, v_rows)` (`len * head_dim` bf16 bit patterns each,
/// position order). Refuses a payload of the wrong size, or with a non-zero byte in the padding
/// rows of its last block, before writing anything.
pub fn blocks_to_rows(
    g: &PoolBlockGeometry,
    start: usize,
    len: usize,
    bytes: &[u8],
    mut write_rows: impl FnMut(usize, usize, &[u16], &[u16]) -> Result<(), String>,
) -> Result<(), String> {
    g.check(start, len)?;
    check_payload(g, len, bytes)?;
    let (mut k, mut v) = (vec![0u16; len * g.head_dim], vec![0u16; len * g.head_dim]);
    for layer in 0..g.layers {
        for head in 0..g.kv_heads {
            for rel in 0..len {
                let row = rel * g.head_dim..(rel + 1) * g.head_dim;
                get_row(
                    bytes,
                    g.row_offset(rel, layer, false, head),
                    &mut k[row.clone()],
                );
                get_row(bytes, g.row_offset(rel, layer, true, head), &mut v[row]);
            }
            write_rows(layer, head, &k, &v)?;
        }
    }
    Ok(())
}

/// The payload must be exactly `payload_len(len)` bytes, and every row past `len` in the last
/// block must be zero (ADR 0003 section 2.2: the writer zeroes them, so anything else is damage).
pub fn check_payload(g: &PoolBlockGeometry, len: usize, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() != g.payload_len(len) {
        return Err(format!(
            "kv payload of {} bytes, {len} positions need {}",
            bytes.len(),
            g.payload_len(len)
        ));
    }
    let tail = len % g.block_size;
    if tail != 0 {
        let last = len - tail;
        for layer in 0..g.layers {
            for value in [false, true] {
                let from = g.row_offset(last + tail, layer, value, 0);
                let to = g.row_offset(last, layer, value, 0) + g.kv_plane_stride();
                if bytes[from..to].iter().any(|&b| b != 0) {
                    return Err(format!("kv payload: padding rows past position {len} of layer {layer} are not zero"));
                }
            }
        }
    }
    Ok(())
}

/// Zeroes the rows at or past `len` in the last block of a payload (in place).
pub fn zero_padding(g: &PoolBlockGeometry, len: usize, bytes: &mut [u8]) {
    let tail = len % g.block_size;
    if tail == 0 {
        return;
    }
    let last = len - tail;
    for layer in 0..g.layers {
        for value in [false, true] {
            let from = g.row_offset(last + tail, layer, value, 0);
            let to = g.row_offset(last, layer, value, 0) + g.kv_plane_stride();
            bytes[from..to].fill(0);
        }
    }
}

fn put_row(out: &mut [u8], at: usize, row: &[u16]) {
    for (i, w) in row.iter().enumerate() {
        out[at + 2 * i..at + 2 * i + 2].copy_from_slice(&w.to_le_bytes());
    }
}

fn get_row(bytes: &[u8], at: usize, row: &mut [u16]) {
    for (i, w) in row.iter_mut().enumerate() {
        *w = u16::from_le_bytes([bytes[at + 2 * i], bytes[at + 2 * i + 1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> PoolBlockGeometry {
        PoolBlockGeometry {
            layers: 3,
            kv_heads: 2,
            head_dim: 4,
            block_size: 16,
        }
    }

    /// A fake fused cache: rows keyed by (layer, head, pos, value).
    fn row_word(layer: usize, head: usize, pos: usize, value: bool, i: usize) -> u16 {
        (((layer * 7 + head * 3 + pos * 11 + usize::from(value) * 5 + i) % 65_521) as u16) | 1
    }

    #[test]
    fn geometry_equals_the_kv_cache_pool_layout() {
        // The payload must be the aien_kv_cache pool layout (HostPagedPoolV1), not a look-alike.
        let g = geom();
        let cfg = aien_kv_cache::KvPoolConfig::for_model(
            4,
            g.block_size,
            g.layers,
            g.kv_heads,
            g.head_dim,
            aien_kv_cache::KvDType::Bf16,
        );
        let d = aien_kv_cache::UnifiedKvTensorPool::allocate(cfg)
            .unwrap()
            .layout_desc();
        assert_eq!(d.block_stride_bytes as usize, g.block_stride());
        assert_eq!(d.layer_stride_bytes as usize, g.layer_stride());
        assert_eq!(d.kv_plane_stride_bytes as usize, g.kv_plane_stride());
        assert_eq!(d.token_stride_bytes as usize, g.token_stride());
        assert_eq!(d.head_stride_bytes as usize, g.head_stride());
    }

    #[test]
    fn fused_rows_round_trip_through_the_payload_byte_for_byte() {
        let g = geom();
        for (start, len) in [(0usize, 1usize), (0, 16), (0, 37), (32, 5), (48, 48)] {
            let bytes = rows_to_blocks(&g, start, len, |l, h, p, k, v| {
                for i in 0..g.head_dim {
                    k[i] = row_word(l, h, p, false, i);
                    v[i] = row_word(l, h, p, true, i);
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(bytes.len(), g.payload_len(len));
            check_payload(&g, len, &bytes).unwrap();
            let mut seen = 0usize;
            blocks_to_rows(&g, start, len, &bytes, |l, h, k, v| {
                for rel in 0..len {
                    for i in 0..g.head_dim {
                        assert_eq!(
                            k[rel * g.head_dim + i],
                            row_word(l, h, start + rel, false, i)
                        );
                        assert_eq!(
                            v[rel * g.head_dim + i],
                            row_word(l, h, start + rel, true, i)
                        );
                    }
                }
                seen += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, g.layers * g.kv_heads);
            // Export again from what was imported: identical bytes.
            let again = rows_to_blocks(&g, start, len, |l, h, p, k, v| {
                let rel = p - start;
                get_row(&bytes, g.row_offset(rel, l, false, h), k);
                get_row(&bytes, g.row_offset(rel, l, true, h), v);
                Ok(())
            })
            .unwrap();
            assert_eq!(again, bytes);
        }
    }

    #[test]
    fn damaged_payloads_are_refused_before_any_write() {
        let g = geom();
        let mut bytes = rows_to_blocks(&g, 0, 20, |_, _, _, k, v| {
            k.fill(0x3f80);
            v.fill(0x4000);
            Ok(())
        })
        .unwrap();
        let mut writes = 0;
        // Wrong size.
        assert!(
            blocks_to_rows(&g, 0, 20, &bytes[..bytes.len() - 1], |_, _, _, _| {
                writes += 1;
                Ok(())
            })
            .is_err()
        );
        // A non-zero padding byte (position 20 of layer 1, V plane).
        let at = g.row_offset(20, 1, true, 1) + 3;
        bytes[at] = 9;
        assert!(blocks_to_rows(&g, 0, 20, &bytes, |_, _, _, _| {
            writes += 1;
            Ok(())
        })
        .is_err());
        assert_eq!(writes, 0);
        zero_padding(&g, 20, &mut bytes);
        assert!(check_payload(&g, 20, &bytes).is_ok());
        // Unaligned start and zero length are refused.
        assert!(rows_to_blocks(&g, 3, 4, |_, _, _, _, _| Ok(())).is_err());
        assert!(rows_to_blocks(&g, 0, 0, |_, _, _, _, _| Ok(())).is_err());
    }
}
