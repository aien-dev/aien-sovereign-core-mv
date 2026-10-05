//! `OmegaGb10Backend` (FB-1 cuts 3b, 3c, 3d): all nine `TensorBackend` ops run on our own
//! native GPU engine (`libomega_gpu.a`, no CUDA) through the `aien-omega-gpu` crate.
//!
//! Native mask: `matmul_vec`, `matmul_batch`, `compute_logits`, `rmsnorm`,
//! `apply_rope`, `swiglu`, `gqa_attention`, `paged_attention`, `paged_attention_batch`.
//! Paged attention over a bf16 pool uses omega's paged kernel; over any other pool dtype
//! (the model builds Fp32 pools) the sequence's K/V are gathered on the host into
//! contiguous f32 exactly as the reference reads them, then run on the f32 gqa kernel.
//! omega attention needs head_dim 64 (TinyLlama, Llama-3.2-1B). A chip error in
//! one of those is a fallback of a claimed-native op: it is counted, goes through
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
use aien_omega_gpu::ResidentTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

/// Cache key: (head_dim, theta bits, position). Drop everything past this many entries.
type RopeKey = (usize, u32, usize);
const ROPE_CACHE_MAX: usize = 16384;

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
    chip_calls: AtomicU64,
    chip_ns: AtomicU64,
    /// OM-2 staging counters summed over every chip attention call.
    kv_bytes_staged: AtomicU64,
    kv_bytes_naive: AtomicU64,
    last_error: Mutex<String>,
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
                TensorOp::ApplyRope,
                TensorOp::Swiglu,
                TensorOp::GqaAttention,
                TensorOp::PagedAttention,
                TensorOp::PagedAttentionBatch,
            ])),
            chip: Mutex::new(HashMap::new()),
            rope_tables: Mutex::new(HashMap::new()),
            chip_errors: AtomicU64::new(0),
            chip_calls: AtomicU64::new(0),
            chip_ns: AtomicU64::new(0),
            kv_bytes_staged: AtomicU64::new(0),
            kv_bytes_naive: AtomicU64::new(0),
            last_error: Mutex::new(String::new()),
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

    /// Rope cos/sin table for `pos`, built exactly as the reference does
    /// (`tensor::apply_rope`, tensor.rs:101): f64 frequency and angle, then cast to f32.
    /// Cached per (head_dim, theta, pos).
    fn rope_table(&self, head_dim: usize, theta: f32, pos: usize) -> RopeTable {
        let key: RopeKey = (head_dim, theta.to_bits(), pos);
        let mut cache = self
            .rope_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(half);
        let mut sin = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = (2 * i) as f64 / (head_dim as f64);
            let freq = 1.0 / (theta as f64).powf(exponent);
            let rot = (pos as f64) * freq;
            sin.push(rot.sin() as f32);
            cos.push(rot.cos() as f32);
        }
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
        let res = call();
        drop(guard);
        match res {
            Ok(info) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns.fetch_add(info.elapsed_ns, Ordering::Relaxed);
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
        theta: f32,
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
        let table = self.rope_table(head_dim, theta, pos);
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
        let res = call();
        drop(guard);
        match res {
            Ok(info) => {
                self.chip_calls.fetch_add(1, Ordering::Relaxed);
                self.chip_ns.fetch_add(info.elapsed_ns, Ordering::Relaxed);
                self.kv_bytes_staged
                    .fetch_add(info.kv_bytes_staged, Ordering::Relaxed);
                self.kv_bytes_naive
                    .fetch_add(info.kv_bytes_naive, Ordering::Relaxed);
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

impl TensorBackend for OmegaGb10Backend {
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

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        if !self.chip_rmsnorm(out, x, weight, eps) {
            self.reference_for(TensorOp::Rmsnorm);
            self.reference.rmsnorm(out, x, weight, eps);
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
        theta: f32,
    ) {
        if !self.chip_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, theta) {
            self.reference_for(TensorOp::ApplyRope);
            self.reference
                .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, theta);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u32) -> f32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    }

    #[test]
    fn mask_is_all_nine_ops() {
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
                "compute_logits"
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
        let (head_dim, nq, nkv, pos, theta) = (64usize, 4usize, 2usize, 37usize, 10000.0f32);
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

    /// Stub build only: each newly native elementwise op is a counted fallback with the
    /// reference result when the chip is unavailable, never silent.
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
        b.apply_rope(&mut q, &mut k, 3, 32, 2, 2, 10000.0);
        r.apply_rope(&mut qw, &mut kw, 3, 32, 2, 2, 10000.0);
        assert_eq!((q, k), (qw, kw));
        assert_eq!(b.fallback_count(), 3);
        let line = b.op_report().line();
        assert!(
            line.contains("native_fallbacks=[rmsnorm:1,apply_rope:1,swiglu:1]"),
            "{line}"
        );
    }

    #[test]
    fn fingerprint_changes_with_content() {
        let a = vec![1.0f32; 1000];
        let mut b = a.clone();
        b[999] = 2.0;
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }
}
