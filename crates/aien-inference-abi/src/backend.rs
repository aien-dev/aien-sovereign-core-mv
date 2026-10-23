//! Decoupled TensorBackend trait and ReferenceCpuBackend correctness oracle.
//! Provides hardware-neutral abstraction for transformer mathematical operations.

use crate::tensor::{
    apply_rope_params as tensor_rope, matmul_vec as tensor_matmul, rmsnorm as tensor_rmsnorm,
};
use aien_abi_core::{AttentionGeometry, ModelConfig, RopeParams};

/// The trait's attention methods return `()`, so a wrong head layout cannot be reported as an
/// error value; it must never turn into numbers either. These helpers refuse loudly (panic with
/// the exact geometry) before any element is read. Callers that can report errors check the
/// geometry up front (`ModelConfig::attention_geometry`), so a panic here means a caller bypassed
/// that check.
pub(crate) fn checked_geometry(
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    op: &str,
) -> AttentionGeometry {
    match AttentionGeometry::new(num_q_heads, num_kv_heads, head_dim) {
        Ok(g) => g,
        Err(e) => panic!("{op} refused: {e}"),
    }
}

/// q and out must be exactly one flattened `[num_q_heads][head_dim]` row each.
pub(crate) fn check_q_out(geom: &AttentionGeometry, q_len: usize, out_len: usize, op: &str) {
    let want = geom.q_dim();
    assert!(
        q_len == want,
        "{op} refused: q has {q_len} values, geometry {}x{} needs {want}",
        geom.num_q_heads(),
        geom.head_dim()
    );
    assert!(
        out_len == want,
        "{op} refused: out has {out_len} values, geometry {}x{} needs {want}",
        geom.num_q_heads(),
        geom.head_dim()
    );
}

/// The pool must be laid out for this geometry's kv heads and head width.
pub(crate) fn check_pool(
    geom: &AttentionGeometry,
    pool: &aien_kv_cache::UnifiedKvTensorPool,
    op: &str,
) {
    let c = pool.config();
    if let Err(e) = geom.check_kv_pool(c.num_kv_heads, c.head_dim) {
        panic!("{op} refused: {e}");
    }
}

/// Reads one KV head from the pool using the pool's dtype. Bf16 bytes are decoded.
/// A short or unknown dtype returns zeros instead of reading past the allocation.
pub(crate) fn read_kv_head(
    pool: &aien_kv_cache::UnifiedKvTensorPool,
    block_id: usize,
    layer_idx: usize,
    is_value: bool,
    token_in_block: usize,
    kv_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dtype = pool.config().dtype;
    let element_bytes = match dtype {
        aien_kv_cache::KvDType::Fp32 => 4,
        aien_kv_cache::KvDType::Bf16 | aien_kv_cache::KvDType::Fp16 => 2,
        aien_kv_cache::KvDType::Fp8 => 1,
        aien_kv_cache::KvDType::Fp4 => return vec![0.0; head_dim],
    };
    let offset = pool.element_offset(block_id, layer_idx, is_value, token_in_block)
        + kv_head * head_dim * element_bytes;
    let nbytes = head_dim * element_bytes;
    if offset.saturating_add(nbytes) > pool.layout().total_bytes {
        return vec![0.0; head_dim];
    }
    let ptr = unsafe { pool.base_ptr().add(offset) };
    match dtype {
        aien_kv_cache::KvDType::Fp32 => unsafe {
            std::slice::from_raw_parts(ptr as *const f32, head_dim).to_vec()
        },
        aien_kv_cache::KvDType::Bf16 => {
            let bits = unsafe { std::slice::from_raw_parts(ptr as *const u16, head_dim) };
            bits.iter()
                .copied()
                .map(aien_kv_cache::bf16_bits_to_f32)
                .collect()
        }
        aien_kv_cache::KvDType::Fp16 => {
            let bits = unsafe { std::slice::from_raw_parts(ptr as *const u16, head_dim) };
            bits.iter().copied().map(fp16_bits_to_f32).collect()
        }
        aien_kv_cache::KvDType::Fp8 => {
            let bytes = unsafe { std::slice::from_raw_parts(ptr, head_dim) };
            bytes.iter().map(|byte| *byte as f32).collect()
        }
        aien_kv_cache::KvDType::Fp4 => vec![0.0; head_dim],
    }
}

fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32) & 0x8000;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;
    let out_bits = if exp == 0 {
        if frac == 0 {
            sign << 16
        } else {
            let mut mantissa = frac;
            let mut exponent = -14;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x3ff;
            (sign << 16) | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exp == 31 {
        (sign << 16) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 16) | (((exp as u32) + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(out_bits)
}

/// Decoupled mathematical abstraction for transformer tensor operations.
pub trait TensorBackend: Send + Sync {
    /// Human-readable name of the backend implementation.
    fn name(&self) -> &'static str;

    /// Refuses a model this backend cannot run correctly, before any forward pass. Default:
    /// accepts everything. Called at the start of every forward pass.
    fn check_model(&self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    /// How many operations this backend ran on a software fallback instead of
    /// the device it was bound to. Zero for pure CPU backends; a production
    /// build treats a non-zero count as a strict real-model violation.
    fn fallback_count(&self) -> u64 {
        0
    }

    /// Operations this backend claims to run natively on its device. Strict mode
    /// fails only on fallbacks of ops in this mask; ops outside it run on the
    /// reference CPU path by design. Default: all ops native (original semantics).
    fn native_ops(&self) -> crate::native_ops::NativeOpMask {
        crate::native_ops::NativeOpMask::ALL
    }

    /// Native-versus-reference listing and per-op counters for receipts.
    /// Default: all native, no counters.
    fn op_report(&self) -> crate::native_ops::OpReport {
        crate::native_ops::OpReport::all_native()
    }

    /// Operations this backend has run on its chip so far (process total), for
    /// receipts (sc#402). Default: None, for a backend with no chip or that does
    /// not count chip calls: an absent claim, never zero.
    fn gpu_executions(&self) -> Option<u64> {
        None
    }

    /// Per-operation wall and chip time since the last reset, one line per kind
    /// ("key calls=N wall_ms=W chip_ms=C", sorted by wall time), for timing receipts.
    /// Default: None, for a backend that does not keep the table.
    fn op_timing_report(&self) -> Option<String> {
        None
    }

    /// Clear the per-operation timing table (no-op without one).
    fn reset_op_timing(&self) {}

    /// In-place Root Mean Square Normalization:
    /// out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32);

    /// `rows` rows of `weight.len()` values normalized independently with the same `weight`
    /// (`x` and `out` are `[rows][dim]`). The prompt read calls this once per chunk and layer
    /// instead of `rmsnorm` once per token; the default is that per-row loop, a chip backend
    /// makes it one launch. `rows == 0` is a no-op.
    fn rmsnorm_rows(&self, out: &mut [f32], x: &[f32], weight: &[f32], rows: usize, eps: f32) {
        let dim = weight.len();
        for r in 0..rows {
            let (lo, hi) = (r * dim, (r + 1) * dim);
            self.rmsnorm(&mut out[lo..hi], &x[lo..hi], weight, eps);
        }
    }

    /// Per-head RMS normalization in place (Qwen3 q/k norm): `x` is `[heads][head_dim]` and every
    /// head is normalized on its own with the same `weight` (`head_dim` values).
    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32);

    /// In-place Rotary Positional Embeddings (RoPE) following canonical rotate_half convention:
    /// rot(v) = [-v[half_dim..], v[..half_dim]]
    /// out = v * cos(pos * f_i) + rot(v) * sin(pos * f_i), with f_i = `rope.inv_freq(i, head_dim)`
    /// (theta^(-2i/d), smoothed by the llama3 scaling when the model has one).
    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope: &RopeParams,
    );

    /// Vector-matrix multiplication for row-major weights: out = x * W^T
    /// where x is [in_dim], weight is [out_dim, in_dim], and out is [out_dim].
    fn matmul_vec(&self, out: &mut [f32], x: &[f32], weight: &[f32], out_dim: usize, in_dim: usize);

    /// Batched vector-matrix multiplication for row-major weights: out = X * W^T
    /// where X is [batch_size, in_dim], weight is [out_dim, in_dim], and out is [batch_size, out_dim].
    fn matmul_batch(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        for b in 0..batch_size {
            let x_b = &x[b * in_dim..(b + 1) * in_dim];
            let out_b = &mut out[b * out_dim..(b + 1) * out_dim];
            self.matmul_vec(out_b, x_b, weight, out_dim, in_dim);
        }
    }

    /// SwiGLU activation: out = (gate * silu) * up = silu(gate) * up
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]);

    /// Grouped-Query Attention (GQA) with multi-head queries and shared key-value heads.
    /// Grouped-Query Attention (GQA) with multi-head queries and shared key-value heads.
    /// Operates over flattened key and value cache slices with causal history.
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
    );

    /// Grouped-Query Paged Attention directly over a physical Copy-on-Write page pool.
    /// Default implementation evaluates online softmax token-by-token across physical blocks
    /// with zero intermediate array allocation or contiguous gathering.
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
        let geom = checked_geometry(num_q_heads, num_kv_heads, head_dim, "paged_attention");
        check_q_out(&geom, q.len(), out.len(), "paged_attention");
        check_pool(&geom, pool, "paged_attention");
        if context_len == 0 || block_ids.is_empty() {
            out.fill(0.0);
            return;
        }

        let gqa_ratio = geom.gqa_ratio();
        let inv_sqrt_d = 1.0 / (head_dim as f64).sqrt();
        let block_size = pool.config().block_size;

        for h in 0..num_q_heads {
            let kv_head = h / gqa_ratio;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            let mut m_prev = f64::NEG_INFINITY;
            let mut l_prev = 0.0f64;
            let mut acc = vec![0.0f64; head_dim];

            for t in 0..context_len {
                let blk_idx = t / block_size;
                if blk_idx >= block_ids.len() {
                    break;
                }
                let blk_id = block_ids[blk_idx];
                let slot = t % block_size;

                let k_head = read_kv_head(pool, blk_id, layer_idx, false, slot, kv_head, head_dim);
                let v_head = read_kv_head(pool, blk_id, layer_idx, true, slot, kv_head, head_dim);

                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (q_head[d] as f64) * (k_head[d] as f64);
                }
                let score = dot * inv_sqrt_d;

                let m_new = m_prev.max(score);
                let alpha = (m_prev - m_new).exp();
                let beta = (score - m_new).exp();
                let l_new = l_prev * alpha + beta;

                for d in 0..head_dim {
                    acc[d] = acc[d] * alpha + beta * (v_head[d] as f64);
                }

                m_prev = m_new;
                l_prev = l_new;
            }

            let inv_l = if l_prev > 0.0 { 1.0 / l_prev } else { 0.0 };
            let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
            for d in 0..head_dim {
                out_head[d] = (acc[d] * inv_l) as f32;
            }
        }
    }

    /// Batched Grouped-Query Paged Attention across multiple sequences concurrently.
    /// Default implementation evaluates each sequence sequentially via paged_attention.
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
        let geom = checked_geometry(num_q_heads, num_kv_heads, head_dim, "paged_attention_batch");
        check_pool(&geom, pool, "paged_attention_batch");
        let q_stride = geom.q_dim();
        assert!(
            q.len() == num_seqs * q_stride && out.len() == num_seqs * q_stride,
            "paged_attention_batch refused: q has {} and out has {} values, {num_seqs} sequences x q_dim {q_stride} need {}",
            q.len(),
            out.len(),
            num_seqs * q_stride
        );
        for s in 0..num_seqs {
            let q_s = &q[s * q_stride..(s + 1) * q_stride];
            let out_s = &mut out[s * q_stride..(s + 1) * q_stride];
            let ctx_len = if s < context_lens.len() && context_lens[s] > 0 {
                context_lens[s] as usize
            } else {
                0
            };
            let bt_start = s * max_blocks_per_seq;
            let bt_end = bt_start + max_blocks_per_seq;
            let bt_slice = if bt_end <= block_tables.len() {
                &block_tables[bt_start..bt_end]
            } else {
                &[]
            };
            let usize_blocks: Vec<usize> = bt_slice
                .iter()
                .filter(|&&b| b >= 0)
                .map(|&b| b as usize)
                .collect();

            self.paged_attention(
                out_s,
                q_s,
                pool,
                &usize_blocks,
                ctx_len,
                layer_idx,
                num_q_heads,
                num_kv_heads,
                head_dim,
            );
        }
    }

    /// Projects hidden states to vocabulary logits: logits = hidden * W^T
    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    );

    /// [`Self::matmul_vec`] for a weight that may be device-resident. Host weights go to
    /// `matmul_vec`; a resident weight is refused by name unless the backend overrides this (only
    /// the Omega GB10 backend can use one). Never a silent CPU fallback.
    fn matmul_vec_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: crate::resident::MatrixRef<'_>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let w = self.resident_refusal_or_host(weight, "matmul_vec");
        self.matmul_vec(out, x, w, out_dim, in_dim);
    }

    /// [`Self::matmul_batch`] for a weight that may be device-resident (see [`Self::matmul_vec_w`]).
    fn matmul_batch_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: crate::resident::MatrixRef<'_>,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let w = self.resident_refusal_or_host(weight, "matmul_batch");
        self.matmul_batch(out, x, w, batch_size, in_dim, out_dim);
    }

    /// [`Self::compute_logits`] for a head that may be device-resident (see [`Self::matmul_vec_w`]).
    fn compute_logits_w(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        head: crate::resident::MatrixRef<'_>,
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        let w = self.resident_refusal_or_host(head, "compute_logits");
        self.compute_logits(logits, hidden, w, vocab_size, hidden_dim);
    }

    /// Host values of `weight`, or the named refusal when it is resident and this backend
    /// cannot use a device copy.
    #[doc(hidden)]
    fn resident_refusal_or_host<'a>(
        &self,
        weight: crate::resident::MatrixRef<'a>,
        op: &str,
    ) -> &'a [f32] {
        match weight {
            crate::resident::MatrixRef::Host(v) => v,
            crate::resident::MatrixRef::Resident(_) => crate::resident::refuse_resident(&format!(
                "backend {} cannot run {op} on a device-resident weight",
                self.name()
            )),
        }
    }
}

/// Pure Rust FP32 reference implementation satisfying TensorBackend.
/// Serves as the golden correctness oracle on both development workstations and CI runners.
#[derive(Debug, Clone, Default)]
pub struct ReferenceCpuBackend;

impl ReferenceCpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl TensorBackend for ReferenceCpuBackend {
    fn name(&self) -> &'static str {
        "ReferenceCpuBackend"
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        tensor_rmsnorm(x, weight, eps, out);
    }

    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        crate::tensor::rmsnorm_heads_in_place(x, weight, head_dim, eps);
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
        tensor_rope(q, k, pos, num_q_heads, num_kv_heads, head_dim, rope);
    }

    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        tensor_matmul(x, weight, out, in_dim, out_dim);
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
        crate::tensor::matmul_batch(x, weight, out, batch_size, in_dim, out_dim);
    }

    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        let n = gate.len().min(up.len()).min(out.len());
        for i in 0..n {
            let g = gate[i] as f64;
            let silu = g / (1.0 + (-g).exp());
            out[i] = (silu * (up[i] as f64)) as f32;
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
        let geom = checked_geometry(num_q_heads, num_kv_heads, head_dim, "gqa_attention");
        check_q_out(&geom, q.len(), out.len(), "gqa_attention");
        let need = seq_len * geom.kv_dim();
        assert!(
            k_cache.len() >= need && v_cache.len() >= need,
            "gqa_attention refused: k/v caches hold {}/{} values, {seq_len} tokens x kv_dim {} need {need}",
            k_cache.len(),
            v_cache.len(),
            geom.kv_dim()
        );
        if seq_len == 0 {
            out.fill(0.0);
            return;
        }

        let gqa_ratio = geom.gqa_ratio();
        let inv_sqrt_d = 1.0 / (head_dim as f64).sqrt();
        let mut scores = vec![0.0f64; seq_len];

        for h in 0..num_q_heads {
            let kv_head = h / gqa_ratio;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            for t in 0..seq_len {
                let k_offset = (t * num_kv_heads + kv_head) * head_dim;
                let k_t = &k_cache[k_offset..k_offset + head_dim];
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (q_head[d] as f64) * (k_t[d] as f64);
                }
                scores[t] = dot * inv_sqrt_d;
            }

            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }

            let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
            for d in 0..head_dim {
                let mut sum = 0.0f64;
                for t in 0..seq_len {
                    let v_offset = (t * num_kv_heads + kv_head) * head_dim;
                    sum += scores[t] * (v_cache[v_offset + d] as f64);
                }
                out_head[d] = sum as f32;
            }
        }
    }

    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        self.matmul_vec(logits, hidden, embed_weight, vocab_size, hidden_dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default `rmsnorm_rows` is the per-row loop: 3 rows in one call equal 3 single calls,
    /// and 0 rows touches nothing.
    #[test]
    fn test_rmsnorm_rows_default_equals_per_row() {
        let b = ReferenceCpuBackend::new();
        let dim = 8;
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.25).collect();
        let x: Vec<f32> = (0..3 * dim).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut got = vec![0.0f32; 3 * dim];
        b.rmsnorm_rows(&mut got, &x, &w, 3, 1e-5);
        for r in 0..3 {
            let mut want = vec![0.0f32; dim];
            b.rmsnorm(&mut want, &x[r * dim..(r + 1) * dim], &w, 1e-5);
            assert_eq!(&got[r * dim..(r + 1) * dim], &want[..], "row {r}");
        }
        let mut untouched = vec![7.0f32; dim];
        b.rmsnorm_rows(&mut untouched, &x, &w, 0, 1e-5);
        assert!(untouched.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_reference_backend_name() {
        let backend = ReferenceCpuBackend::new();
        assert_eq!(backend.name(), "ReferenceCpuBackend");
    }

    #[test]
    fn test_reference_rmsnorm_matches_tensor() {
        let backend = ReferenceCpuBackend::new();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        backend.rmsnorm(&mut out, &x, &w, 1e-5);

        let mut expected = vec![0.0f32; 4];
        tensor_rmsnorm(&x, &w, 1e-5, &mut expected);

        assert_eq!(out, expected);
    }

    #[test]
    fn test_reference_rope_matches_tensor() {
        let backend = ReferenceCpuBackend::new();
        let mut q1 = vec![1.0f32; 128];
        let mut k1 = vec![1.0f32; 64];
        let mut q2 = q1.clone();
        let mut k2 = k1.clone();

        backend.apply_rope(&mut q1, &mut k1, 5, 64, 2, 1, &RopeParams::plain(10000.0));
        crate::tensor::apply_rope(&mut q2, &mut k2, 5, 2, 1, 64, 10000.0);

        assert_eq!(q1, q2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_reference_matmul_matches_tensor() {
        let backend = ReferenceCpuBackend::new();
        let x = vec![1.0f32, 2.0, 3.0];
        let w = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 2];
        backend.matmul_vec(&mut out, &x, &w, 2, 3);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn test_reference_swiglu() {
        let backend = ReferenceCpuBackend::new();
        let gate = vec![0.0f32, 2.0];
        let up = vec![1.0f32, 3.0];
        let mut out = vec![0.0f32; 2];
        backend.swiglu(&mut out, &gate, &up);

        // silu(0) = 0 * 0.5 = 0, so out[0] = 0
        assert_eq!(out[0], 0.0);
        // silu(2) = 2 / (1 + exp(-2)) ~= 1.761594, out[1] = 1.761594 * 3 ~= 5.28478
        assert!((out[1] - 5.28478).abs() < 1e-4);
    }

    #[test]
    fn test_reference_gqa_attention() {
        let backend = ReferenceCpuBackend::new();
        let head_dim = 4;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let seq_len = 2;

        let q = vec![1.0f32; num_q_heads * head_dim];
        let k_cache = vec![1.0f32; seq_len * num_kv_heads * head_dim];
        let v_cache = vec![2.0f32; seq_len * num_kv_heads * head_dim];
        let mut out = vec![0.0f32; num_q_heads * head_dim];

        backend.gqa_attention(
            &mut out,
            &q,
            &k_cache,
            &v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );

        // All values are 2.0, so weighted sum should be 2.0 for all outputs
        for v in &out {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reference_compute_logits() {
        let backend = ReferenceCpuBackend::new();
        let hidden = vec![1.0f32, 0.5];
        let embed = vec![2.0f32, 0.0, 0.0, 4.0, 1.0, 1.0];
        let mut logits = vec![0.0f32; 3];
        backend.compute_logits(&mut logits, &hidden, &embed, 3, 2);

        assert_eq!(logits, vec![2.0, 2.0, 1.5]);
    }
}
