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

//! `OmegaGb10Backend` (FB-1 cuts 3b, 3c, 3d): all ten `TensorBackend` ops run on our own
//! native GPU engine (canonical CUDApp, no CUDA) through the `aien-omega-gpu` crate.
//!
//! Native mask: `matmul_vec`, `matmul_batch`, `compute_logits`, `rmsnorm`, `rmsnorm_heads`,
//! `apply_rope`, `swiglu`, `gqa_attention`, `paged_attention`, `paged_attention_batch`.
//! Paged attention over a bf16 pool uses omega's paged kernel; over any other pool dtype
//! (the model builds Fp32 pools) the sequence's K/V are gathered on the host into
//! contiguous f32 exactly as the reference reads them, then run on the f32 gqa kernel.
//! omega attention needs head_dim 64 (TinyLlama, Llama-3.2-1B) or 128 (Qwen3-4B, since omega
//! b564bf4). Qwen3 at any other head_dim is always refused; at 128 it is refused when the
//! switch is off (see `omega_model_refusal`).
//!
//! A chip error in one of those ops is a fallback of a claimed-native op: it is counted, goes through
//! `OpAccounting::reference_path` (fatal in a production build, see `strict.rs`)
//! and, in a dev build only, the reference result is computed so the run goes on.
//!
//! Weights: the trait hands weights as `&[f32]` slices per call (row-major
//! `[out_dim, in_dim]`, computing `x * W^T`). The first call for a slice uploads
//! `W^T` (k = in_dim, n = out_dim, rounded to bf16 by omega) once and keeps it
//! resident; later calls move only activations. The cache key is
//! (pointer, length, in_dim, out_dim) plus a sampled content fingerprint.
//!
//! INVARIANT: weight slices are immutable and outlive the backend's use of them.
//! Checked for the model path: `TransformerWeights` (weights.rs:35) owns plain
//! `Vec<f32>` fields; the forward passes borrow it as `&TransformerWeights`
//! (transformer_backend.rs:287, :399) and nothing in the crate mutates a loaded
//! checkpoint after construction (UNVERIFIED for external callers: `weights` is a
//! `pub` field, so a caller that mutates it in place gets stale device weights;
//! the fingerprint catches a freed-and-reused address or a changed sample point,
//! not every in-place edit). Use one backend per weight set, or call
//! [`OmegaGb10Backend::clear_resident`] after changing weights.
use crate::backend::{ReferenceCpuBackend, TensorBackend};
use crate::native_ops::{NativeOpMask, OpAccounting, OpReport, TensorOp};
use aien_abi_core::{ModelConfig, RopeParams};
use aien_omega_gpu::ResidentTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// `AIEN_GPU_BACKEND=omega` demands this backend: a build without the Omega engine linked
/// refuses to start instead of falling back to the CPU reference. It is the only GPU backend.
pub const GPU_BACKEND_ENV: &str = "AIEN_GPU_BACKEND";

/// True when the environment asks for the Omega backend.
pub fn omega_backend_selected() -> bool {
    std::env::var(GPU_BACKEND_ENV)
        .map(|v| v.trim().eq_ignore_ascii_case("omega"))
        .unwrap_or(false)
}

const BACKEND_NAME: &str = "OmegaGb10Backend";

type Key = (usize, usize, usize, usize);

/// Rope cos/sin table for one position: `head_dim / 2` values each.
type RopeTable = Arc<(Vec<f32>, Vec<f32>)>;

/// Cache key: (head_dim, rope parameter bits, position). Drop everything past this many entries.
type RopeKey = (usize, [u64; 5], usize);
const ROPE_CACHE_MAX: usize = 16384;

/// Single-token elementwise ops run on the host (sc#384 decode speed): a one-row rmsnorm, one
/// token's q/k head norms and its rope are 1 to 4 us of arithmetic on the host (measured on the
/// X925) against 11 to 34 us as a chip launch with its copies, and their data already lives in
/// host memory. At or under this many f32 values the op runs on the reference implementation
/// (the golden maths), tallied as `<op>_host` in the timing table and counted in `host_ops`,
/// not a fallback. Prefill rows (128 x 4096 and more) stay on the chip. swiglu stays on the chip
/// at every size (host f64 exp over 12288 values 26 us against 28 us on the chip: no gain).
pub const HOST_ELEMENTWISE_MAX: usize = 16384;

struct Resident {
    fingerprint: u64,
    tensor: ResidentTensor,
}

/// Native backend on the Omega GPU engine (matmuls and elementwise ops).
pub struct OmegaGb10Backend {
    reference: ReferenceCpuBackend,
    acct: OpAccounting,
    /// One lock serializes every chip call and guards the resident cache
    /// (omega's resident-call thread safety is not documented).
    chip: Mutex<HashMap<Key, Resident>>,
    rope_tables: Mutex<HashMap<RopeKey, RopeTable>>,
    chip_errors: AtomicU64,
    /// Elementwise ops routed to the host by size (`HOST_ELEMENTWISE_MAX`), not fallbacks.
    host_ops: AtomicU64,
    chip_calls: AtomicU64,
    chip_ns: AtomicU64,
    /// OM-2 staging counters summed over every chip attention call.
    kv_bytes_staged: AtomicU64,
    kv_bytes_naive: AtomicU64,
    /// Chip attention calls that read the resident KV pool in place (cut E, omega#411).
    kv_resident_calls: AtomicU64,
    last_error: Mutex<String>,
    /// Wall and chip time per operation kind (resident matmul keyed by shape), for the
    /// decode-speed receipts (sc#384); read with `op_timing_lines`, cleared with `reset_op_times`.
    op_times: Mutex<HashMap<String, OpTime>>,
}

/// One row of the per-operation timing table.
#[derive(Default, Clone, Copy, Debug)]
pub struct OpTime {
    pub calls: u64,
    pub wall_ns: u64,
    pub chip_ns: u64,
}

impl Default for OmegaGb10Backend {
    fn default() -> Self {
        Self::new()
    }
}

/// Cheap content check: FNV-1a over up to 64 evenly spaced elements plus the last one.
fn fingerprint(w: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let step = (w.len() / 64).max(1);
    let mut i = 0;
    while i < w.len() {
        h = (h ^ w[i].to_bits() as u64).wrapping_mul(0x100000001b3);
        i += step;
    }
    if let Some(last) = w.last() {
        h = (h ^ last.to_bits() as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl OmegaGb10Backend {
    pub fn new() -> Self {
        Self {
            reference: ReferenceCpuBackend::new(),
            acct: OpAccounting::new(NativeOpMask::from_ops(&[
                TensorOp::MatmulVec,
                TensorOp::MatmulBatch,
                TensorOp::ComputeLogits,
                TensorOp::Rmsnorm,
                TensorOp::RmsnormHeads,
                TensorOp::ApplyRope,
                TensorOp::Swiglu,
                TensorOp::GqaAttention,
                TensorOp::PagedAttention,
                TensorOp::PagedAttentionBatch,
            ])),
            chip: Mutex::new(HashMap::new()),
            rope_tables: Mutex::new(HashMap::new()),
            chip_errors: AtomicU64::new(0),
            host_ops: AtomicU64::new(0),
            chip_calls: AtomicU64::new(0),
            chip_ns: AtomicU64::new(0),
            kv_bytes_staged: AtomicU64::new(0),
            kv_bytes_naive: AtomicU64::new(0),
            kv_resident_calls: AtomicU64::new(0),
            last_error: Mutex::new(String::new()),
            op_times: Mutex::new(HashMap::new()),
        }
    }

    fn tally(&self, key: String, wall_ns: u64, chip_ns: u64) {
        if let Ok(mut m) = self.op_times.lock() {
            let e = m.entry(key).or_default();
            e.calls += 1;
            e.wall_ns += wall_ns;
            e.chip_ns += chip_ns;
        }
    }

    /// One host-routed elementwise op: run it, tally its wall time under `key` (chip time 0).
    fn host_op(&self, key: &str, op: impl FnOnce()) {
        let t0 = Instant::now();
        op();
        self.host_ops.fetch_add(1, Ordering::Relaxed);
        self.tally(key.to_string(), t0.elapsed().as_nanos() as u64, 0);
    }

    /// The per-operation timing table as text, heaviest wall time first.
    pub fn op_timing_lines(&self) -> String {
        let mut rows: Vec<(String, OpTime)> = self
            .op_times
            .lock()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();
        rows.sort_by_key(|(_, t)| std::cmp::Reverse(t.wall_ns));
        if rows.is_empty() {
            return String::new();
        }
        rows.iter()
            .map(|(k, t)| {
                format!(
                    "{k} calls={} wall_ms={:.1} chip_ms={:.1}",
                    t.calls,
                    t.wall_ns as f64 / 1e6,
                    t.chip_ns as f64 / 1e6
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
            + &format!(
                "; kv_staged_bytes_total={} kv_resident_calls_total={}",
                self.kv_bytes_staged(),
                self.kv_resident_calls()
            )
    }

    pub fn reset_op_times(&self) {
        if let Ok(mut m) = self.op_times.lock() {
            m.clear();
        }
    }

    /// True when the real native library is linked (not the CI stub).
    pub fn is_available(&self) -> bool {
        aien_omega_gpu::is_native()
    }

    pub fn chip_calls(&self) -> u64 {
        self.chip_calls.load(Ordering::Relaxed)
    }

    pub fn chip_errors(&self) -> u64 {
        self.chip_errors.load(Ordering::Relaxed)
    }

    /// Sum of chip launch-to-completion time reported by omega, in nanoseconds.
    pub fn chip_ns(&self) -> u64 {
        self.chip_ns.load(Ordering::Relaxed)
    }

    /// KV bytes omega copied into its staging buffer over every chip attention call (OM-2:
    /// one copy per physical block per launch) and what per-reference staging would have
    /// copied. Equal when no block is shared inside a launch.
    pub fn kv_bytes_staged(&self) -> u64 {
        self.kv_bytes_staged.load(Ordering::Relaxed)
    }

    pub fn kv_bytes_naive(&self) -> u64 {
        self.kv_bytes_naive.load(Ordering::Relaxed)
    }

    /// Chip attention calls whose KV came from the resident pool (no staging copy).
    pub fn kv_resident_calls(&self) -> u64 {
        self.kv_resident_calls.load(Ordering::Relaxed)
    }

    /// Elementwise ops routed to the host by size (`HOST_ELEMENTWISE_MAX`), never fallbacks.
    pub fn host_ops(&self) -> u64 {
        self.host_ops.load(Ordering::Relaxed)
    }

    pub fn last_error(&self) -> String {
        self.last_error
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Drop every resident weight (call after changing weights in place).
    pub fn clear_resident(&self) {
        if let Ok(mut c) = self.chip.lock() {
            c.clear();
        }
    }

    /// Number of resident weight matrices (for tests and receipts).
    pub fn resident_count(&self) -> usize {
        self.chip.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// `out[m x out_dim] = x[m x in_dim] * W^T`, W row-major `[out_dim, in_dim]`.
    /// Returns false on any chip error (already recorded).
    fn chip_matmul(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        m: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> bool {
        if x.len() != m * in_dim || out.len() != m * out_dim || weight.len() != out_dim * in_dim {
            return self.fail(format!(
                "shape mismatch: x={} out={} w={} for m={m} in={in_dim} out_dim={out_dim}",
                x.len(),
                out.len(),
                weight.len()
            ));
        }
        let key: Key = (weight.as_ptr() as usize, weight.len(), in_dim, out_dim);
        let fp = fingerprint(weight);
        let mut cache = match self.chip.lock() {
            Ok(g) => g,
            Err(_) => return self.fail("chip lock poisoned".into()),
        };
        if cache.get(&key).is_some_and(|r| r.fingerprint != fp) {
            cache.remove(&key);
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(key) {
            // W is [out, in]; the engine wants B = W^T as [k = in, n = out].
            let mut wt = vec![0.0f32; weight.len()];
            for o in 0..out_dim {
                let row = &weight[o * in_dim..(o + 1) * in_dim];
                for (i, v) in row.iter().enumerate() {
                    wt[i * out_dim + o] = *v;
                }
            }
            match ResidentTensor::upload_f32(in_dim, out_dim, &wt) {
                Ok(tensor) => {
                    slot.insert(Resident {
                        fingerprint: fp,
                        tensor,
                    });
                }
                Err(e) => {
                    let msg = format!("upload {in_dim}x{out_dim}: {e}");
                    return self.fail(msg);
                }
            }
        }
        let res = &cache[&key];
        match res.tensor.matmul_f32(m, x, out) {
            Ok(info) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns
                    .fetch_add(info.raw.elapsed_ns, Ordering::Relaxed);
                true
            }
            Err(e) => {
                drop(cache);
                self.fail(format!("matmul m={m} k={in_dim} n={out_dim}: {e}"))
            }
        }
    }

    /// `out[m x out_dim] = x[m x in_dim] * W^T` for a device-resident weight. The weight has no
    /// host values, so a chip failure cannot fall back to the CPU: it is recorded and then
    /// refused by name (panic), in every build.
    fn chip_resident_matmul(
        &self,
        out: &mut [f32],
        x: &[f32],
        w: &crate::resident::ResidentWeight,
        m: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let refuse = |this: &Self, why: String| -> ! {
            this.fail(why.clone());
            crate::resident::refuse_resident(&why)
        };
        if w.k() != in_dim || w.n() != out_dim || x.len() != m * in_dim || out.len() != m * out_dim
        {
            refuse(
                self,
                format!(
                    "resident weight is {}x{} but the call is m={m} in={in_dim} out={out_dim} (x={} out={})",
                    w.k(),
                    w.n(),
                    x.len(),
                    out.len()
                ),
            );
        }
        // One lock serializes every chip call (same lock as `chip_matmul`).
        let guard = match self.chip.lock() {
            Ok(g) => g,
            Err(_) => refuse(self, "chip lock poisoned".into()),
        };
        let t0 = Instant::now();
        let res = w.matmul_f32(m, x, out);
        drop(guard);
        match res {
            Ok(ns) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns.fetch_add(ns, Ordering::Relaxed);
                self.tally(
                    format!("matmul m={m} k={in_dim} n={out_dim}"),
                    t0.elapsed().as_nanos() as u64,
                    ns,
                );
            }
            Err(e) => refuse(
                self,
                format!("resident matmul m={m} k={in_dim} n={out_dim}: {e}"),
            ),
        }
    }

    /// Rope cos/sin table for `pos`, built exactly as the reference does
    /// (`tensor::apply_rope`, tensor.rs:101): f64 frequency and angle, then cast to f32.
    /// Cached per (head_dim, rope parameters, pos).
    fn rope_table(&self, head_dim: usize, rope: &RopeParams, pos: usize) -> RopeTable {
        let key: RopeKey = (head_dim, rope.cache_key(), pos);
        let mut cache = self
            .rope_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
        let (cos, sin) = crate::tensor::rope_cos_sin(rope, head_dim, pos);
        if cache.len() >= ROPE_CACHE_MAX {
            cache.clear();
        }
        let t: RopeTable = Arc::new((cos, sin));
        cache.insert(key, Arc::clone(&t));
        t
    }

    /// Run one elementwise chip call under the chip lock. False on any error (recorded).
    fn chip_elementwise(
        &self,
        what: &str,
        call: impl FnOnce() -> Result<aien_omega_gpu::OmegaGpuEwInfo, aien_omega_gpu::OmegaGpuError>,
    ) -> bool {
        let guard = match self.chip.lock() {
            Ok(g) => g,
            Err(_) => return self.fail("chip lock poisoned".into()),
        };
        let t0 = Instant::now();
        let res = call();
        drop(guard);
        match res {
            Ok(info) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns.fetch_add(info.elapsed_ns, Ordering::Relaxed);
                self.tally(
                    what.split_whitespace().next().unwrap_or(what).to_string(),
                    t0.elapsed().as_nanos() as u64,
                    info.elapsed_ns,
                );
                true
            }
            Err(e) => self.fail(format!("{what}: {e}")),
        }
    }

    fn chip_rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) -> bool {
        let dim = x.len();
        self.chip_elementwise(&format!("rmsnorm dim={dim}"), || {
            aien_omega_gpu::rmsnorm_f32(1, dim, x, weight, eps, out)
        })
    }

    /// `rows` rows of `weight.len()` in one launch (omega's rmsnorm is row-wise already).
    fn chip_rmsnorm_rows(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        rows: usize,
        eps: f32,
    ) -> bool {
        let dim = weight.len();
        if dim == 0 || x.len() != rows * dim || out.len() != rows * dim {
            return self.fail(format!(
                "rmsnorm_rows shape mismatch: x={} out={} rows={rows} dim={dim}",
                x.len(),
                out.len()
            ));
        }
        self.chip_elementwise(&format!("rmsnorm rows={rows} dim={dim}"), || {
            aien_omega_gpu::rmsnorm_f32(rows, dim, x, weight, eps, out)
        })
    }

    /// Every head is one row of omega's row-wise rmsnorm (same weight for each row), one launch.
    fn chip_rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) -> bool {
        if head_dim == 0 || !x.len().is_multiple_of(head_dim) || weight.len() != head_dim {
            return self.fail(format!(
                "rmsnorm_heads shape mismatch: x={} weight={} head_dim={head_dim}",
                x.len(),
                weight.len()
            ));
        }
        let rows = x.len() / head_dim;
        let input = x.to_vec();
        self.chip_elementwise(&format!("rmsnorm_heads rows={rows} dim={head_dim}"), || {
            aien_omega_gpu::rmsnorm_f32(rows, head_dim, &input, weight, eps, x)
        })
    }

    /// q then k go to the chip as one `heads` run (the table is the same for every head).
    #[allow(clippy::too_many_arguments)]
    fn chip_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope: &RopeParams,
    ) -> bool {
        if head_dim == 0
            || !head_dim.is_multiple_of(2)
            || q.len() != num_q_heads * head_dim
            || k.len() != num_kv_heads * head_dim
        {
            return self.fail(format!(
                "rope shape mismatch: q={} k={} head_dim={head_dim} q_heads={num_q_heads} kv_heads={num_kv_heads}",
                q.len(),
                k.len()
            ));
        }
        let table = self.rope_table(head_dim, rope, pos);
        let mut both = Vec::with_capacity(q.len() + k.len());
        both.extend_from_slice(q);
        both.extend_from_slice(k);
        let mut rotated = vec![0.0f32; both.len()];
        let heads = num_q_heads + num_kv_heads;
        let ok = self.chip_elementwise(&format!("rope heads={heads} head_dim={head_dim}"), || {
            aien_omega_gpu::rope_f32(heads, head_dim, &both, &table.0, &table.1, &mut rotated)
        });
        if ok {
            let (rq, rk) = rotated.split_at(q.len());
            q.copy_from_slice(rq);
            k.copy_from_slice(rk);
        }
        ok
    }

    fn chip_swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) -> bool {
        // The reference uses min(len) of the three; the chip path demands equal lengths.
        if gate.len() != up.len() || gate.len() != out.len() {
            return self.fail(format!(
                "swiglu shape mismatch: gate={} up={} out={}",
                gate.len(),
                up.len(),
                out.len()
            ));
        }
        self.chip_elementwise(&format!("swiglu n={}", gate.len()), || {
            aien_omega_gpu::swiglu_f32(gate, up, out)
        })
    }

    fn chip_attention(
        &self,
        what: &str,
        call: impl FnOnce() -> Result<aien_omega_gpu::OmegaGpuAttnInfo, aien_omega_gpu::OmegaGpuError>,
    ) -> bool {
        let guard = match self.chip.lock() {
            Ok(g) => g,
            Err(_) => return self.fail("chip lock poisoned".into()),
        };
        let t0 = Instant::now();
        let res = call();
        drop(guard);
        match res {
            Ok(info) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns.fetch_add(info.elapsed_ns, Ordering::Relaxed);
                self.tally(
                    what.split_whitespace().next().unwrap_or(what).to_string(),
                    t0.elapsed().as_nanos() as u64,
                    info.elapsed_ns,
                );
                self.kv_bytes_staged
                    .fetch_add(info.kv_bytes_staged, Ordering::Relaxed);
                self.kv_bytes_naive
                    .fetch_add(info.kv_bytes_naive, Ordering::Relaxed);
                if info.kv_source == aien_omega_gpu::KV_SOURCE_RESIDENT {
                    self.kv_resident_calls.fetch_add(1, Ordering::Relaxed);
                }
                true
            }
            Err(e) => self.fail(format!("{what}: {e}")),
        }
    }

    /// One sequence of paged attention on the chip. A bf16 pool goes straight to the
    /// paged kernel; any other dtype (the model builds Fp32 pools) is gathered on the host
    /// into contiguous `[t][kv_head][head_dim]` f32 K and V exactly as the reference reads
    /// them (tokens past the block table are dropped), then runs the f32 gqa kernel.
    #[allow(clippy::too_many_arguments)]
    fn chip_paged_one(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_ids: &[usize],
        context_len: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> bool {
        // Geometry and pool layout are checked before any native code sees the pool. A refusal
        // is counted like a chip error and the reference then refuses the same call loudly.
        let geom = match aien_abi_core::AttentionGeometry::new(num_q_heads, num_kv_heads, head_dim)
        {
            Ok(g) => g,
            Err(e) => return self.fail(format!("paged_attention geometry refused: {e}")),
        };
        let cfg = pool.config();
        if let Err(e) = geom.check_kv_pool(cfg.num_kv_heads, cfg.head_dim) {
            return self.fail(format!("paged_attention refused: {e}"));
        }
        if q.len() != geom.q_dim() || out.len() != geom.q_dim() {
            return self.fail(format!(
                "paged_attention refused: q={} out={} values, geometry needs {}",
                q.len(),
                out.len(),
                geom.q_dim()
            ));
        }
        if context_len == 0 || block_ids.is_empty() {
            out.fill(0.0);
            return true;
        }
        if pool.config().dtype == aien_kv_cache::KvDType::Bf16 {
            let ids: Option<Vec<u32>> = block_ids.iter().map(|&b| u32::try_from(b).ok()).collect();
            let Some(ids) = ids else {
                return self.fail("paged_attention: block id past u32".into());
            };
            let d = pool.layout_desc();
            let layout = aien_omega_gpu::OmegaGpuKvLayout {
                block_stride_bytes: d.block_stride_bytes,
                layer_stride_bytes: d.layer_stride_bytes,
                kv_plane_stride_bytes: d.kv_plane_stride_bytes,
                token_stride_bytes: d.token_stride_bytes,
                head_stride_bytes: d.head_stride_bytes,
                pool_bytes: d.pool_bytes,
                num_blocks: d.num_blocks,
                num_layers: d.num_layers,
                block_size: d.block_size,
            };
            // SAFETY: the pool owns total_bytes() readable bytes at base_ptr() for its lifetime,
            // and we hold &pool for the whole call.
            let bytes = unsafe { std::slice::from_raw_parts(pool.base_ptr(), pool.total_bytes()) };
            return self.chip_attention(&format!("paged_attention_bf16 ctx={context_len}"), || {
                aien_omega_gpu::paged_attention_bf16(
                    q,
                    bytes,
                    &layout,
                    &ids,
                    context_len,
                    layer_idx,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                )
            });
        }
        let block_size = pool.config().block_size;
        let n = context_len.min(block_ids.len().saturating_mul(block_size));
        let row = num_kv_heads * head_dim;
        let mut k = vec![0.0f32; n * row];
        let mut v = vec![0.0f32; n * row];
        for t in 0..n {
            let blk = block_ids[t / block_size];
            let slot = t % block_size;
            for kh in 0..num_kv_heads {
                let at = t * row + kh * head_dim;
                k[at..at + head_dim].copy_from_slice(&crate::backend::read_kv_head(
                    pool, blk, layer_idx, false, slot, kh, head_dim,
                ));
                v[at..at + head_dim].copy_from_slice(&crate::backend::read_kv_head(
                    pool, blk, layer_idx, true, slot, kh, head_dim,
                ));
            }
        }
        self.chip_attention(&format!("paged_attention(gathered) ctx={n}"), || {
            aien_omega_gpu::gqa_attention_f32(
                q,
                &k,
                &v,
                n,
                num_q_heads,
                num_kv_heads,
                head_dim,
                out,
            )
        })
    }

    fn fail(&self, msg: String) -> bool {
        self.chip_errors.fetch_add(1, Ordering::Relaxed);
        let stage = aien_omega_gpu::last_error();
        let full = if stage.is_empty() {
            msg
        } else {
            format!("{msg} (stage: {stage})")
        };
        eprintln!("OMEGA_BACKEND chip error: {full}");
        if let Ok(mut l) = self.last_error.lock() {
            *l = full;
        }
        false
    }

    fn reference_for(&self, op: TensorOp) {
        self.acct.reference_path(BACKEND_NAME, op);
    }
}

/// Why the GB10 engine refuses `config`, if it does. Qwen3 (`qk_norm`) runs its per-head q/k
/// norm as `rmsnorm_heads`, one row per head on omega's row-wise rmsnorm, which needs
/// `head_dim % 128 == 0` (omega `omega_gpu_elementwise_api.h`); attention takes `head_dim` 64 or
/// 128 (omega `omega_gpu_attention_api.h`, since b564bf4). So a Qwen3 model runs on the engine
/// only with `head_dim` 128 (Qwen3-4B) and is refused up front otherwise, before any chip work.
/// Llama-architecture models are not touched by this check.
///
/// Evidence limits: the #277 attempt 3 pass (issue comment 6059774846) covers a Linux-hosted
/// GB10 only (not native AIENOS), one prompt, context 4096, no endurance run and no failing
/// baseline. So the default (switch unset) enables Qwen3 only on a production-strict run; under
/// `AIEN_DEV_FALLBACK=1` it is refused unless the switch is explicitly `1`.
///
/// Qwen3 at `head_dim` 128 runs on the GB10 by default since the qualified path passed
/// (sovereign-core #277, declared attempt 3): the daemon streams the weights to the device
/// and reserves, prepares and seals the serving buffers once, and refuses by name with no
/// fallback if that fails. Before that, the driver's channel and weight allocations failed
/// with NV_ERR_NO_MEMORY when MemFree was low after the f32 weight load (omega#327). The
/// explicit refusal is [`GB10_QWEN3_OPT_IN_ENV`]` = 0`; see [`omega_model_refusal_with`].
pub fn omega_model_refusal(config: &ModelConfig) -> Option<String> {
    omega_model_refusal_with(config, gb10_qwen3_enabled())
}

/// Environment switch for the GB10 Qwen3 path (rule, one place):
/// - exactly `1`: enabled (the old declared-attempt opt-in, any build);
/// - unset: enabled only on the path the evidence covers, a production-strict run
///   ([`crate::strict::production_strict`], so no `AIEN_DEV_FALLBACK=1`); the native engine and
///   the streamed resident load that go with it are checked by `check_model` and
///   [`crate::resident::resident_load_wanted`];
/// - `0`, or any other value including non-UTF8: refused by name (fail closed).
pub const GB10_QWEN3_OPT_IN_ENV: &str = "AIEN_GB10_QWEN3_DECLARED_ATTEMPT";

/// Whether the GB10 Qwen3 path is enabled, read from [`GB10_QWEN3_OPT_IN_ENV`] and the
/// process strictness.
pub fn gb10_qwen3_enabled() -> bool {
    gb10_qwen3_enabled_from_os(
        std::env::var_os(GB10_QWEN3_OPT_IN_ENV).as_deref(),
        crate::strict::production_strict(),
    )
}

/// [`gb10_qwen3_enabled`] on an explicit value and strictness (`strict` is true when
/// fallbacks are not allowed).
pub fn gb10_qwen3_enabled_from(value: Option<&str>, strict: bool) -> bool {
    match value {
        None => strict,
        Some(v) => v == "1",
    }
}

/// Same rule on a raw OS value: non-UTF8 refuses.
pub fn gb10_qwen3_enabled_from_os(value: Option<&std::ffi::OsStr>, strict: bool) -> bool {
    match value {
        None => gb10_qwen3_enabled_from(None, strict),
        Some(v) => gb10_qwen3_enabled_from(Some(v.to_str().unwrap_or("<non-utf8>")), strict),
    }
}

/// [`omega_model_refusal`] with the switch passed explicitly instead of read from the
/// environment (`qwen3_enabled` is true when the Qwen3 path is enabled).
pub fn omega_model_refusal_with(config: &ModelConfig, qwen3_enabled: bool) -> Option<String> {
    if !config.qk_norm {
        return None;
    }
    if config.head_dim != 128 {
        return Some(format!(
            "OmegaGb10Backend refuses model {:?}: Qwen3 (per-head q/k norm) with head_dim {} is \
             unsupported on the GB10 engine, which runs Qwen3 only at head_dim 128 (rmsnorm_heads \
             needs head_dim % 128 == 0); run it on the CPU reference backend",
            config.model_id, config.head_dim
        ));
    }
    (!qwen3_enabled).then(|| {
        format!(
            "OmegaGb10Backend refuses model {:?}: the GB10 Qwen3 path is switched off by \
             {GB10_QWEN3_OPT_IN_ENV} (only 1, or unset on a production-strict run, enables it); run it on the CPU \
             reference backend, or unset {GB10_QWEN3_OPT_IN_ENV} on a strict run (no \
             AIEN_DEV_FALLBACK) to use the qualified GB10 path (sovereign-core#277)",
            config.model_id
        )
    })
}

impl OmegaGb10Backend {
    /// [`TensorBackend::check_model`] with the Qwen3 switch passed explicitly (the real call reads
    /// [`gb10_qwen3_enabled`]); refuses Qwen3 on a stub build whatever the switch says.
    pub fn check_model_with(
        &self,
        config: &ModelConfig,
        qwen3_enabled: bool,
    ) -> Result<(), String> {
        omega_model_refusal_with(config, qwen3_enabled).map_or(Ok(()), Err)?;
        if config.qk_norm && !self.is_available() {
            return Err(format!(
                "OmegaGb10Backend refuses model {:?}: Qwen3 needs the linked GB10 engine for its \
                 streamed weights and serving reservation, and this is a stub build (no chip); \
                 run it on the CPU reference backend",
                config.model_id
            ));
        }
        if config.qk_norm {
            eprintln!(
                "OMEGA_BACKEND NOTE: running Qwen3 {:?} on GB10 via the qualified path \
                 (streamed resident weights, reserve+prepare+seal once, no on-demand fallback; \
                 sovereign-core#277). Set {GB10_QWEN3_OPT_IN_ENV}=0 to refuse it",
                config.model_id
            );
        }
        Ok(())
    }
}

impl TensorBackend for OmegaGb10Backend {
    fn check_model(&self, config: &ModelConfig) -> Result<(), String> {
        self.check_model_with(config, gb10_qwen3_enabled())
    }

    fn name(&self) -> &'static str {
        if self.is_available() {
            "OmegaGb10Backend (native Omega engine, no CUDA, NVIDIA GB10 sm_121)"
        } else {
            "OmegaGb10Backend (stub, no chip)"
        }
    }

    fn fallback_count(&self) -> u64 {
        self.acct.fallback_count()
    }

    fn native_ops(&self) -> NativeOpMask {
        self.acct.mask()
    }

    fn op_report(&self) -> OpReport {
        self.acct.report()
    }

    /// sc#402: the chip calls so far, only when the real engine is linked (the stub has
    /// no chip, so it makes no claim).
    fn gpu_executions(&self) -> Option<u64> {
        self.is_available().then(|| self.chip_calls())
    }

    fn op_timing_report(&self) -> Option<String> {
        Some(self.op_timing_lines())
    }

    fn reset_op_timing(&self) {
        self.reset_op_times()
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        if x.len() <= HOST_ELEMENTWISE_MAX {
            self.host_op("rmsnorm_host", || {
                self.reference.rmsnorm(out, x, weight, eps)
            });
            return;
        }
        if !self.chip_rmsnorm(out, x, weight, eps) {
            self.reference_for(TensorOp::Rmsnorm);
            self.reference.rmsnorm(out, x, weight, eps);
        }
    }

    fn rmsnorm_rows(&self, out: &mut [f32], x: &[f32], weight: &[f32], rows: usize, eps: f32) {
        if rows == 0 {
            return;
        }
        if x.len() <= HOST_ELEMENTWISE_MAX {
            self.host_op("rmsnorm_host", || {
                self.reference.rmsnorm_rows(out, x, weight, rows, eps)
            });
            return;
        }
        if !self.chip_rmsnorm_rows(out, x, weight, rows, eps) {
            self.reference_for(TensorOp::Rmsnorm);
            self.reference.rmsnorm_rows(out, x, weight, rows, eps);
        }
    }

    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        if x.len() <= HOST_ELEMENTWISE_MAX {
            self.host_op("rmsnorm_heads_host", || {
                self.reference.rmsnorm_heads(x, weight, head_dim, eps)
            });
            return;
        }
        if !self.chip_rmsnorm_heads(x, weight, head_dim, eps) {
            self.reference_for(TensorOp::RmsnormHeads);
            self.reference.rmsnorm_heads(x, weight, head_dim, eps);
        }
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope: &RopeParams,
    ) {
        if q.len() + k.len() <= HOST_ELEMENTWISE_MAX {
            self.host_op("rope_host", || {
                self.reference
                    .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, rope)
            });
            return;
        }
        if !self.chip_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, rope) {
            self.reference_for(TensorOp::ApplyRope);
            self.reference
                .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, rope);
        }
    }

    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        if !self.chip_matmul(out, x, weight, 1, in_dim, out_dim) {
            self.reference_for(TensorOp::MatmulVec);
            self.reference.matmul_vec(out, x, weight, out_dim, in_dim);
        }
    }

    fn matmul_batch(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        if !self.chip_matmul(out, x, weight, batch_size, in_dim, out_dim) {
            self.reference_for(TensorOp::MatmulBatch);
            self.reference
                .matmul_batch(out, x, weight, batch_size, in_dim, out_dim);
        }
    }

    fn matmul_vec_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: crate::resident::MatrixRef<'_>,
        out_dim: usize,
        in_dim: usize,
    ) {
        match weight {
            crate::resident::MatrixRef::Host(w) => self.matmul_vec(out, x, w, out_dim, in_dim),
            crate::resident::MatrixRef::Resident(r) => {
                self.chip_resident_matmul(out, x, r, 1, in_dim, out_dim)
            }
        }
    }

    fn matmul_batch_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: crate::resident::MatrixRef<'_>,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        match weight {
            crate::resident::MatrixRef::Host(w) => {
                self.matmul_batch(out, x, w, batch_size, in_dim, out_dim)
            }
            crate::resident::MatrixRef::Resident(r) => {
                self.chip_resident_matmul(out, x, r, batch_size, in_dim, out_dim)
            }
        }
    }

    fn compute_logits_w(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        head: crate::resident::MatrixRef<'_>,
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        match head {
            crate::resident::MatrixRef::Host(w) => {
                self.compute_logits(logits, hidden, w, vocab_size, hidden_dim)
            }
            crate::resident::MatrixRef::Resident(r) => {
                self.chip_resident_matmul(logits, hidden, r, 1, hidden_dim, vocab_size)
            }
        }
    }

    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        if !self.chip_swiglu(out, gate, up) {
            self.reference_for(TensorOp::Swiglu);
            self.reference.swiglu(out, gate, up);
        }
    }

    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // The reference reads only the first seq_len rows; causal prefill passes the whole
        // prompt's cache with a shorter seq_len, so hand omega exactly those rows.
        let geom = aien_abi_core::AttentionGeometry::new(num_q_heads, num_kv_heads, head_dim);
        let need = geom.as_ref().map_or(0, |g| seq_len * g.kv_dim());
        let shapes_ok = match &geom {
            Ok(g) => {
                q.len() == g.q_dim()
                    && out.len() == g.q_dim()
                    && k_cache.len() >= need
                    && v_cache.len() >= need
            }
            Err(e) => self.fail(format!("gqa_attention geometry refused: {e}")),
        };
        let k_rows = &k_cache[..need.min(k_cache.len())];
        let v_rows = &v_cache[..need.min(v_cache.len())];
        let native = shapes_ok
            && self.chip_attention(&format!("gqa_attention seq={seq_len}"), || {
                aien_omega_gpu::gqa_attention_f32(
                    q,
                    k_rows,
                    v_rows,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                )
            });
        if native {
            return;
        }
        self.reference_for(TensorOp::GqaAttention);
        self.reference.gqa_attention(
            out,
            q,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn gqa_attention_rows(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        base_len: usize,
        num_rows: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        if num_rows == 0 {
            return;
        }
        // omega wants exactly base_len + num_rows tokens of cache; the prompt read hands the
        // sequence's whole cache, which is that long right after the chunk's KV writes.
        let geom = aien_abi_core::AttentionGeometry::new(num_q_heads, num_kv_heads, head_dim);
        let total = base_len + num_rows;
        let need = geom.as_ref().map_or(0, |g| total * g.kv_dim());
        let shapes_ok = match &geom {
            Ok(g) => {
                q.len() == num_rows * g.q_dim()
                    && out.len() == num_rows * g.q_dim()
                    && k_cache.len() >= need
                    && v_cache.len() >= need
            }
            Err(e) => self.fail(format!("gqa_attention_rows geometry refused: {e}")),
        };
        let k_rows = &k_cache[..need.min(k_cache.len())];
        let v_rows = &v_cache[..need.min(v_cache.len())];
        let native = shapes_ok
            && self.chip_attention(
                &format!("gqa_attention_rows base={base_len} rows={num_rows}"),
                || {
                    aien_omega_gpu::gqa_attention_rows_f32(
                        q,
                        k_rows,
                        v_rows,
                        base_len,
                        num_rows,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        out,
                    )
                },
            );
        if native {
            return;
        }
        self.reference_for(TensorOp::GqaAttention);
        self.reference.gqa_attention_rows(
            out,
            q,
            k_cache,
            v_cache,
            base_len,
            num_rows,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn alloc_kv_pool(&self, bytes: usize) -> Option<aien_kv_cache::DefaultUnifiedBuffer> {
        // omega's resident KV pool: the paged attention reads it in place (no per-call staging of
        // the layer slices). A refusal falls back to the host-mapped pool, which stays correct.
        let (ptr, len) = match aien_omega_gpu::resident_kv_pool_alloc(bytes) {
            Ok(p) => p,
            Err(e) => {
                self.fail(format!(
                    "resident KV pool of {bytes} bytes: {e:?}; host-mapped pool, staged attention"
                ));
                return None;
            }
        };
        // SAFETY: omega owns `len` bytes at `ptr` until `resident_kv_pool_free`, which the release
        // callback runs exactly once; omega's allocations have chip address == host address.
        let wrapped = unsafe {
            aien_kv_cache::external_default_buffer(
                ptr,
                len,
                ptr as u64,
                Box::new(aien_omega_gpu::resident_kv_pool_free),
            )
        };
        match wrapped {
            Ok(b) => Some(b),
            Err(e) => {
                aien_omega_gpu::resident_kv_pool_free();
                self.fail(e);
                None
            }
        }
    }

    fn paged_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_ids: &[usize],
        context_len: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        if self.chip_paged_one(
            out,
            q,
            pool,
            block_ids,
            context_len,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        ) {
            return;
        }
        self.reference_for(TensorOp::PagedAttention);
        self.reference.paged_attention(
            out,
            q,
            pool,
            block_ids,
            context_len,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn paged_attention_batch(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_tables: &[i32],
        context_lens: &[i32],
        max_blocks_per_seq: usize,
        num_seqs: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // Per sequence, the same row handling as the TensorBackend default (context_lens <= 0
        // means 0, a row past the array is empty, negative entries are removed), each row on
        // the chip. Any failed row sends the whole batch to the reference, counted once as a
        // batch op (the reference batch calls the reference single path, not ours).
        let q_stride = num_q_heads * head_dim;
        let mut all_native = q.len() >= num_seqs * q_stride && out.len() >= num_seqs * q_stride;
        for s in 0..num_seqs {
            if !all_native {
                break;
            }
            let ctx = context_lens
                .get(s)
                .copied()
                .filter(|&c| c > 0)
                .map_or(0, |c| c as usize);
            let (b0, b1) = (s * max_blocks_per_seq, (s + 1) * max_blocks_per_seq);
            let row: Vec<usize> = if b1 <= block_tables.len() {
                block_tables[b0..b1]
                    .iter()
                    .filter(|&&b| b >= 0)
                    .map(|&b| b as usize)
                    .collect()
            } else {
                Vec::new()
            };
            all_native = self.chip_paged_one(
                &mut out[s * q_stride..(s + 1) * q_stride],
                &q[s * q_stride..(s + 1) * q_stride],
                pool,
                &row,
                ctx,
                layer_idx,
                num_q_heads,
                num_kv_heads,
                head_dim,
            );
        }
        if all_native {
            return;
        }
        self.reference_for(TensorOp::PagedAttentionBatch);
        self.reference.paged_attention_batch(
            out,
            q,
            pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_seqs,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );
    }

    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        if !self.chip_matmul(logits, hidden, embed_weight, 1, hidden_dim, vocab_size) {
            self.reference_for(TensorOp::ComputeLogits);
            self.reference
                .compute_logits(logits, hidden, embed_weight, vocab_size, hidden_dim);
        }
    }
}

// ---- bounded GPU session open (issue #239, diagnostics for #236) ----

/// Attempts the daemon makes to open the GPU session before it refuses to start.
pub const GPU_SESSION_OPEN_ATTEMPTS: u32 = 3;

/// Pause between two session-open attempts.
pub const GPU_SESSION_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

/// Wall-clock budget for the whole GPU session open (probe plus retries). It is checked
/// only BETWEEN attempts: no new attempt starts once it has passed. An attempt already
/// running is never interrupted, aborted or killed (standing rule: a GPU job is not
/// killed mid-launch), so a single probe that hangs forever still blocks startup; the
/// bound guarantees only that a slow-failing sequence of attempts cannot go on past
/// the deadline plus one attempt.
pub const GPU_SESSION_OPEN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Why one session-open attempt failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAttemptError {
    /// Return code and omega stage, as the chip-error log prints them.
    pub message: String,
    /// True once omega has latched the process (every later chip call fails), so a
    /// further attempt cannot succeed and is not made.
    pub latched: bool,
}

/// Runs `attempt(n)` for `n = 1..=attempts` (at least once) until one succeeds and
/// returns its value with the attempt number. Each failure goes to `on_failure`
/// (the caller logs it); `pause(n)` runs between failed attempt `n` and the next. A
/// latched failure stops at once. Never loops past `attempts`.
pub fn retry_bounded<T>(
    attempts: u32,
    attempt: impl FnMut(u32) -> Result<T, SessionAttemptError>,
    on_failure: impl FnMut(u32, u32, &SessionAttemptError),
    pause: impl FnMut(u32),
) -> Result<(T, u32), String> {
    retry_bounded_within(
        attempts,
        None,
        || std::time::Duration::ZERO,
        attempt,
        on_failure,
        pause,
    )
}

/// [`retry_bounded`] with a wall-clock bound. `elapsed()` is the time since the whole
/// startup began (the caller's injected clock). `deadline` is checked only BETWEEN
/// attempts, after the pause: once `elapsed() >= deadline` no further attempt starts
/// and the error names the attempts made and the elapsed time. Attempt 1 always runs.
/// An attempt already in flight is never interrupted: `attempt` is called to
/// completion, because a GPU launch must not be cancelled or killed mid-way. So the
/// deadline is a bound on starting attempts, not a timeout on a single hung attempt.
pub fn retry_bounded_within<T>(
    attempts: u32,
    deadline: Option<std::time::Duration>,
    mut elapsed: impl FnMut() -> std::time::Duration,
    mut attempt: impl FnMut(u32) -> Result<T, SessionAttemptError>,
    mut on_failure: impl FnMut(u32, u32, &SessionAttemptError),
    mut pause: impl FnMut(u32),
) -> Result<(T, u32), String> {
    let attempts = attempts.max(1);
    let mut last = String::new();
    for n in 1..=attempts {
        if n > 1 {
            if let Some(limit) = deadline {
                let now = elapsed();
                if now >= limit {
                    return Err(format!(
                        "GPU session open timed out: deadline {} ms passed after {} of {attempts} attempts, elapsed {} ms (an attempt in flight is never interrupted; the deadline applies between attempts); last error: {last}",
                        limit.as_millis(),
                        n - 1,
                        now.as_millis()
                    ));
                }
            }
        }
        match attempt(n) {
            Ok(v) => return Ok((v, n)),
            Err(e) => {
                on_failure(n, attempts, &e);
                if e.latched {
                    return Err(format!(
                        "GPU session open failed on attempt {n}/{attempts} and omega latched the process (not retried): {}",
                        e.message
                    ));
                }
                last = e.message;
                if n < attempts {
                    pause(n);
                }
            }
        }
    }
    Err(format!(
        "GPU session did not open after {attempts} attempts; last error: {last}"
    ))
}

fn fmt_mem(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{b} bytes ({:.2} GiB)", b as f64 / (1u64 << 30) as f64),
        None => "unknown".to_string(),
    }
}

/// Opens the GPU session before serving, with up to `attempts` tries `delay` apart.
///
/// omega opens its device and channel lazily on the first chip call, and a failed
/// open leaves the session closed but not latched (omega c0369e6
/// src/omega_gpu_session.c:101-109: `m16_native_create_channel` failure closes and
/// returns NULL without setting `g_blocked`), so the next call opens again. The
/// probe is one 1 x 128 rmsnorm called on `aien_omega_gpu` directly, outside the
/// `TensorBackend` accounting, so a failure here is a clear start-up refusal
/// instead of the strict-fallback panic at warm-up (#236). Every attempt logs its
/// number, the omega status and stage, and `MemAvailable` from `mem_available`.
/// `deadline` bounds the whole open between attempts only; see [`retry_bounded_within`]
/// (an in-flight probe is never interrupted).
/// Returns the attempt number that opened the session.
pub fn open_gpu_session_with_retry(
    attempts: u32,
    delay: std::time::Duration,
    deadline: std::time::Duration,
    mem_available: &dyn Fn() -> Option<u64>,
) -> Result<u32, String> {
    if !aien_omega_gpu::is_native() {
        return Err(
            "GPU session open: the Omega GPU engine is not linked (stub build)".to_string(),
        );
    }
    const PROBE_DIM: usize = 128; // omega rmsnorm needs dim % 128 == 0
    let ones = [1.0f32; PROBE_DIM];
    let started = std::time::Instant::now();
    let (_, n) = retry_bounded_within(
        attempts,
        Some(deadline),
        || started.elapsed(),
        |_| {
            let mut out = [0.0f32; PROBE_DIM];
            aien_omega_gpu::rmsnorm_f32(1, PROBE_DIM, &ones, &ones, 1e-5, &mut out)
                .map(|_| ())
                .map_err(|e| {
                    let stage = aien_omega_gpu::last_error();
                    SessionAttemptError {
                        message: if stage.is_empty() {
                            e.to_string()
                        } else {
                            format!("{e} (stage: {stage})")
                        },
                        latched: aien_omega_gpu::is_blocked(),
                    }
                })
        },
        |n, of, e| {
            eprintln!(
                "OMEGA_BACKEND GPU session open attempt {n}/{of} failed: {}; latched={}; MemAvailable {}",
                e.message,
                e.latched,
                fmt_mem(mem_available())
            )
        },
        |_| std::thread::sleep(delay),
    )?;
    println!(
        "  GPU session: open on attempt {n}/{} (MemAvailable {})",
        attempts.max(1),
        fmt_mem(mem_available())
    );
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_4b_shape() -> ModelConfig {
        let mut c = ModelConfig::tinyllama_1_1b();
        c.model_id = "qwen3-shape-test".to_string();
        c.qk_norm = true;
        c.head_dim = 128;
        c
    }

    #[test]
    fn stub_backend_refuses_qwen3_with_switch_unset_or_one() {
        let b = OmegaGb10Backend::new();
        if b.is_available() {
            return; // chip build: the stub refusal does not apply
        }
        let q = qwen3_4b_shape();
        // Unset on a strict run is "enabled"; `1` is enabled. A stub build still refuses.
        for enabled in [
            gb10_qwen3_enabled_from(None, true),
            gb10_qwen3_enabled_from(Some("1"), true),
        ] {
            assert!(enabled);
            let err = b.check_model_with(&q, enabled).unwrap_err();
            assert!(err.contains("stub build (no chip)"), "{err}");
        }
        // Llama-architecture models are unchanged.
        assert!(b
            .check_model_with(&ModelConfig::tinyllama_1_1b(), false)
            .is_ok());
    }

    #[test]
    fn non_utf8_switch_value_refuses() {
        use std::os::unix::ffi::OsStrExt;
        let bad = std::ffi::OsStr::from_bytes(&[0x31, 0xff]);
        for strict in [true, false] {
            assert!(!gb10_qwen3_enabled_from_os(Some(bad), strict));
        }
        assert!(gb10_qwen3_enabled_from_os(
            Some(std::ffi::OsStr::new("1")),
            false
        ));
        assert!(!gb10_qwen3_enabled_from_os(
            Some(std::ffi::OsStr::new("0")),
            true
        ));
    }

    #[test]
    fn unset_switch_needs_a_strict_run_but_one_always_enables() {
        // Rule: AIEN_DEV_FALLBACK=1 (non-strict) with the switch unset is refused by name.
        assert!(!gb10_qwen3_enabled_from(None, false));
        let q = qwen3_4b_shape();
        let msg = omega_model_refusal_with(&q, gb10_qwen3_enabled_from(None, false))
            .expect("unset + non-strict must refuse");
        assert!(msg.contains(GB10_QWEN3_OPT_IN_ENV), "{msg}");
        let b = OmegaGb10Backend::new();
        assert!(b
            .check_model_with(&q, gb10_qwen3_enabled_from(None, false))
            .unwrap_err()
            .contains(GB10_QWEN3_OPT_IN_ENV));
        // Explicit 1 keeps the declared-attempt opt-in; no resident load without strict.
        assert!(omega_model_refusal_with(&q, gb10_qwen3_enabled_from(Some("1"), false)).is_none());
        assert!(!crate::resident::resident_load_wanted(
            true,
            false,
            gb10_qwen3_enabled_from(Some("1"), false),
            &q
        ));
        assert!(!crate::resident::resident_load_wanted(
            true,
            false,
            gb10_qwen3_enabled_from(None, false),
            &q
        ));
        assert!(crate::resident::resident_load_wanted(
            true,
            true,
            gb10_qwen3_enabled_from(None, true),
            &q
        ));
    }

    fn lcg(s: &mut u32) -> f32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    }

    #[test]
    fn mask_is_all_ten_ops() {
        let b = OmegaGb10Backend::new();
        let names: Vec<_> = b
            .native_ops()
            .native_ops()
            .iter()
            .map(|o| o.name())
            .collect();
        assert_eq!(
            names,
            [
                "rmsnorm",
                "apply_rope",
                "matmul_vec",
                "matmul_batch",
                "swiglu",
                "gqa_attention",
                "paged_attention",
                "paged_attention_batch",
                "compute_logits",
                "rmsnorm_heads"
            ]
        );
    }

    #[test]
    fn attention_chip_error_is_a_counted_fallback() {
        if aien_omega_gpu::is_native() || !crate::strict::dev_fallback_active() {
            return; // native: nothing fails here; production: it would (correctly) panic
        }
        let b = OmegaGb10Backend::new();
        let (q, k, v) = (
            [0.5f32, -1.0],
            [1.0f32, 2.0, -1.0, 0.25],
            [3.0f32, 4.0, 5.0, 6.0],
        );
        let mut got = [0.0f32; 2];
        let mut want = [0.0f32; 2];
        b.gqa_attention(&mut got, &q, &k, &v, 2, 1, 1, 2);
        ReferenceCpuBackend::new().gqa_attention(&mut want, &q, &k, &v, 2, 1, 1, 2);
        assert_eq!(got, want);
        assert_eq!(b.fallback_count(), 1);
        assert!(b
            .op_report()
            .line()
            .contains("native_fallbacks=[gqa_attention:1]"));
    }

    /// Stub build only: a chip error on a claimed-native op is a counted fallback
    /// (dev build) with the reference result, never silent.
    #[test]
    fn chip_error_is_a_counted_fallback() {
        if aien_omega_gpu::is_native() || !crate::strict::dev_fallback_active() {
            return; // native: nothing fails here; production: it would (correctly) panic
        }
        let b = OmegaGb10Backend::new();
        let mut s = 7u32;
        let (m, k, n) = (3usize, 32usize, 16usize);
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let mut got = vec![0.0f32; m * n];
        let mut want = vec![0.0f32; m * n];
        b.matmul_batch(&mut got, &x, &w, m, k, n);
        ReferenceCpuBackend::new().matmul_batch(&mut want, &x, &w, m, k, n);
        assert_eq!(got, want);
        assert_eq!(b.fallback_count(), 1);
        assert_eq!(b.chip_errors(), 1);
        assert!(b
            .op_report()
            .line()
            .contains("native_fallbacks=[matmul_batch:1]"));
    }

    /// Negative control for stub and native builds alike. The engine refuses a
    /// claimed-native op before any launch (swiglu with unequal gate/up lengths; the
    /// reference path accepts it and uses the shorter length). In a strict process that
    /// refusal must be fatal; in a dev process it must be a counted fallback with the
    /// reference result. This replaces a `should_panic` test whose skip branch panicked
    /// with the expected text, so it passed without testing anything in dev and native runs.
    #[test]
    fn chip_refusal_is_fatal_in_strict_and_counted_in_dev() {
        let b = OmegaGb10Backend::new();
        let gate = [0.5f32, 2.0, -1.0];
        let up = [3.0f32, -0.25];
        let mut out = [0.0f32; 2];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.swiglu(&mut out, &gate, &up)
        }));
        assert_eq!(b.chip_errors(), 1, "{}", b.last_error());
        assert_eq!(b.fallback_count(), 1);
        assert!(
            b.op_report().line().contains("native_fallbacks=[swiglu:1]"),
            "{}",
            b.op_report().line()
        );
        if crate::strict::production_strict() {
            let err = r.expect_err("a strict process must refuse a claimed-native fallback");
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            assert!(
                msg.contains(crate::strict::STRICT_VIOLATION_PREFIX) && msg.contains("swiglu"),
                "{msg}"
            );
        } else {
            r.expect("a dev process continues on the reference path");
            let mut want = [0.0f32; 2];
            ReferenceCpuBackend::new().swiglu(&mut want, &gate, &up);
            assert_eq!(out, want);
        }
    }

    /// The cached cos/sin table, applied on the host with the same f32 formulas as the
    /// chip kernel (FMUL, FMUL, FADD), must equal the reference rope bit for bit.
    #[test]
    fn rope_table_reproduces_reference_rope() {
        let b = OmegaGb10Backend::new();
        let (head_dim, nq, nkv, pos) = (64usize, 4usize, 2usize, 37usize);
        let theta = &RopeParams::plain(10000.0);
        let mut s = 99u32;
        let q0: Vec<f32> = (0..nq * head_dim).map(|_| lcg(&mut s)).collect();
        let k0: Vec<f32> = (0..nkv * head_dim).map(|_| lcg(&mut s)).collect();
        let (mut qr, mut kr) = (q0.clone(), k0.clone());
        ReferenceCpuBackend::new().apply_rope(&mut qr, &mut kr, pos, head_dim, nq, nkv, theta);
        let t = b.rope_table(head_dim, theta, pos);
        let half = head_dim / 2;
        let rot = |v: &[f32]| -> Vec<f32> {
            let mut o = v.to_vec();
            for h in 0..v.len() / head_dim {
                for i in 0..half {
                    let (a, c) = (v[h * head_dim + i], v[h * head_dim + i + half]);
                    o[h * head_dim + i] = a * t.0[i] - c * t.1[i];
                    o[h * head_dim + i + half] = a * t.1[i] + c * t.0[i];
                }
            }
            o
        };
        assert_eq!(rot(&q0), qr);
        assert_eq!(rot(&k0), kr);
        // cached: same allocation on the second ask
        assert!(Arc::ptr_eq(&t, &b.rope_table(head_dim, theta, pos)));
    }

    /// sc#402: the chip-call total is reported only when the real engine is linked; the
    /// stub has no chip and makes no claim (None, never a zero that reads as "ran, no calls").
    #[test]
    fn op_timing_table_tallies_sorts_and_resets() {
        let b = OmegaGb10Backend::new();
        assert_eq!(b.op_timing_lines(), "");
        assert_eq!(b.op_timing_report(), Some(String::new()));
        b.tally("rmsnorm".into(), 2_000_000, 1_500_000);
        b.tally("matmul m=1 k=4096 n=4096".into(), 10_000_000, 9_000_000);
        b.tally("rmsnorm".into(), 1_000_000, 500_000);
        assert_eq!(
            b.op_timing_lines(),
            "matmul m=1 k=4096 n=4096 calls=1 wall_ms=10.0 chip_ms=9.0; rmsnorm calls=2 wall_ms=3.0 chip_ms=2.0; kv_staged_bytes_total=0 kv_resident_calls_total=0"
        );
        b.reset_op_timing();
        assert_eq!(b.op_timing_lines(), "");
    }

    #[test]
    fn gpu_executions_is_the_chip_call_total_or_absent() {
        let b = OmegaGb10Backend::new();
        let got = TensorBackend::gpu_executions(&b);
        if b.is_available() {
            assert_eq!(got, Some(b.chip_calls()));
        } else {
            assert_eq!(got, None);
        }
    }

    /// Stub build only: each newly native elementwise op is a counted fallback with the
    /// reference result when the chip is unavailable, never silent.
    #[test]
    fn single_token_norms_and_rope_run_on_the_host_without_a_fallback() {
        // Any build: the host route never touches the chip. One row of 4096 (the model width),
        // 32 + 8 heads of 128, one token of rope: reference results, host_ops counts, no fallback.
        let b = OmegaGb10Backend::new();
        let r = ReferenceCpuBackend::new();
        let mut s = 9u32;
        let x: Vec<f32> = (0..4096).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..4096).map(|_| lcg(&mut s)).collect();
        let (mut got, mut want) = (vec![0.0f32; 4096], vec![0.0f32; 4096]);
        b.rmsnorm(&mut got, &x, &w, 1e-6);
        r.rmsnorm(&mut want, &x, &w, 1e-6);
        assert_eq!(got, want);
        b.rmsnorm_rows(&mut got, &x, &w, 1, 1e-6);
        assert_eq!(got, want);
        let wh: Vec<f32> = (0..128).map(|_| lcg(&mut s)).collect();
        let (mut hq, mut hw) = (x.clone(), x.clone());
        b.rmsnorm_heads(&mut hq, &wh, 128, 1e-6);
        r.rmsnorm_heads(&mut hw, &wh, 128, 1e-6);
        assert_eq!(hq, hw);
        let (mut q, mut k) = (x.clone(), x[..1024].to_vec());
        let (mut qw, mut kw) = (q.clone(), k.clone());
        let rope = RopeParams::plain(1_000_000.0);
        b.apply_rope(&mut q, &mut k, 1268, 128, 32, 8, &rope);
        r.apply_rope(&mut qw, &mut kw, 1268, 128, 32, 8, &rope);
        assert_eq!((q, k), (qw, kw));
        assert_eq!(b.host_ops(), 4);
        assert_eq!(b.fallback_count(), 0);
        assert_eq!(b.gpu_executions().unwrap_or(0), 0);
        let table = b.op_timing_lines();
        assert!(
            table.contains("rmsnorm_host calls=2") && table.contains("rope_host calls=1"),
            "{table}"
        );
    }

    #[test]
    fn elementwise_chip_errors_are_counted_fallbacks() {
        if aien_omega_gpu::is_native() || !crate::strict::dev_fallback_active() {
            return;
        }
        let b = OmegaGb10Backend::new();
        let r = ReferenceCpuBackend::new();
        let mut s = 5u32;
        let x: Vec<f32> = (0..128).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..128).map(|_| lcg(&mut s)).collect();
        let (mut got, mut want) = (vec![0.0f32; 128], vec![0.0f32; 128]);
        b.rmsnorm(&mut got, &x, &w, 1e-5);
        r.rmsnorm(&mut want, &x, &w, 1e-5);
        assert_eq!(got, want);
        b.swiglu(&mut got, &x, &w);
        r.swiglu(&mut want, &x, &w);
        assert_eq!(got, want);
        let (mut q, mut k) = (x[..64].to_vec(), x[64..].to_vec());
        let (mut qw, mut kw) = (q.clone(), k.clone());
        b.apply_rope(&mut q, &mut k, 3, 32, 2, 2, &RopeParams::plain(10000.0));
        r.apply_rope(&mut qw, &mut kw, 3, 32, 2, 2, &RopeParams::plain(10000.0));
        assert_eq!((q, k), (qw, kw));
        // the one-row rmsnorm and the one-token rope run on the host by size (HOST_ELEMENTWISE_MAX):
        // the golden maths, not fallbacks; swiglu stays a chip op and is the one counted fallback
        assert_eq!(b.host_ops(), 2);
        assert_eq!(b.fallback_count(), 1);
        let line = b.op_report().line();
        assert!(line.contains("native_fallbacks=[swiglu:1]"), "{line}");
    }

    #[test]
    fn fingerprint_changes_with_content() {
        let a = vec![1.0f32; 1000];
        let mut b = a.clone();
        b[999] = 2.0;
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }

    fn fail_msg(n: u32, latched: bool) -> SessionAttemptError {
        SessionAttemptError {
            message: format!("omega_gpu rc=-4 (CHIP_FAIL) try {n}"),
            latched,
        }
    }

    #[test]
    fn retry_stops_at_the_bound_and_logs_every_attempt() {
        let (mut calls, mut logged, mut pauses) = (0u32, Vec::new(), Vec::new());
        let err = retry_bounded::<()>(
            3,
            |n| {
                calls += 1;
                Err(fail_msg(n, false))
            },
            |n, of, e| logged.push(format!("{n}/{of}: {}", e.message)),
            |n| pauses.push(n),
        )
        .unwrap_err();
        assert_eq!(calls, 3);
        assert_eq!(pauses, vec![1, 2]);
        assert_eq!(logged.len(), 3);
        assert!(logged[2].starts_with("3/3"));
        assert!(
            err.contains("after 3 attempts") && err.contains("try 3"),
            "{err}"
        );
    }

    #[test]
    fn retry_returns_the_attempt_that_succeeded() {
        let mut pauses = 0;
        let r = retry_bounded(
            3,
            |n| {
                if n < 2 {
                    Err(fail_msg(n, false))
                } else {
                    Ok(n * 10)
                }
            },
            |_, _, _| {},
            |_| pauses += 1,
        );
        assert_eq!(r, Ok((20, 2)));
        assert_eq!(pauses, 1);
    }

    #[test]
    fn retry_does_not_repeat_a_latched_failure_and_runs_at_least_once() {
        let mut calls = 0;
        let err = retry_bounded::<()>(
            3,
            |n| {
                calls += 1;
                Err(fail_msg(n, true))
            },
            |_, _, _| {},
            |_| panic!("no pause after a latched failure"),
        )
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(err.contains("latched") && err.contains("1/3"), "{err}");

        let mut zero_calls = 0;
        let _ = retry_bounded::<()>(
            0,
            |n| {
                zero_calls += 1;
                Err(fail_msg(n, false))
            },
            |_, _, _| {},
            |_| {},
        );
        assert_eq!(zero_calls, 1);
    }

    #[test]
    fn deadline_passed_between_attempts_stops_with_a_timeout_error() {
        // Injected clock: each attempt "takes" 400 ms; deadline 1000 ms.
        let clock = std::cell::Cell::new(0u64);
        let mut calls = 0;
        let err = retry_bounded_within::<()>(
            10,
            Some(std::time::Duration::from_millis(1000)),
            || std::time::Duration::from_millis(clock.get()),
            |n| {
                calls += 1;
                clock.set(clock.get() + 400);
                Err(fail_msg(n, false))
            },
            |_, _, _| {},
            |_| {},
        )
        .unwrap_err();
        // Attempts at t=0, 400, 800 start; the check at t=1200 stops the fourth.
        assert_eq!(calls, 3, "{err}");
        assert!(err.contains("timed out"), "{err}");
        assert!(err.contains("after 3 of 10 attempts"), "{err}");
        assert!(err.contains("elapsed 1200 ms"), "{err}");
        assert!(err.contains("never interrupted"), "{err}");
    }

    #[test]
    fn attempts_stay_bounded_when_the_deadline_never_passes() {
        let mut calls = 0;
        let err = retry_bounded_within::<()>(
            3,
            Some(std::time::Duration::from_secs(3600)),
            || std::time::Duration::ZERO,
            |n| {
                calls += 1;
                Err(fail_msg(n, false))
            },
            |_, _, _| {},
            |_| {},
        )
        .unwrap_err();
        assert_eq!(calls, 3);
        assert!(err.contains("did not open after 3 attempts"), "{err}");
        assert!(!err.contains("timed out"), "{err}");
    }

    #[test]
    fn first_attempt_runs_even_if_the_deadline_is_already_zero() {
        let mut calls = 0;
        let _ = retry_bounded_within::<()>(
            3,
            Some(std::time::Duration::ZERO),
            || std::time::Duration::from_secs(1),
            |n| {
                calls += 1;
                Err(fail_msg(n, false))
            },
            |_, _, _| {},
            |_| {},
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn session_open_refuses_cleanly_in_a_stub_build() {
        if aien_omega_gpu::is_native() {
            return; // the native path is verified on the chip, not here
        }
        let err = open_gpu_session_with_retry(
            3,
            std::time::Duration::ZERO,
            GPU_SESSION_OPEN_DEADLINE,
            &|| None,
        )
        .unwrap_err();
        assert!(err.contains("not linked"), "{err}");
    }
}
