//! OWN-8B stage 2: the dense Qwen3 forward pass on Omega (design
//! `~/handoffs/2026-10-15-own8b-design.md`, sections 2.2 to 2.4, stage table row 2).
//!
//! [`Qwen3DenseOmegaEngine`] runs a dense Qwen3 checkpoint (Qwen3-8B FP8, Qwen3-4B bf16 with tied
//! embeddings, any shape the loader accepts) one sequence at a time:
//!
//! - decode [`Qwen3DenseOmegaEngine::step`]: host RMSNorm, one projection launch for q|k|v, host
//!   q/k per-head norms and RoPE, the K/V row into the paged bf16 pool, one paged attention launch,
//!   one launch each for o, gate|up and down, host SwiGLU and residuals; final norm on the host.
//!   FP8 projections run omega's block-scaled FP8 GEMV, bf16 projections omega's bf16 streaming
//!   GEMV (the design's kernel map, section 1.4); the lm_head is the bf16 GEMV in both formats
//!   ([`Qwen3DenseOmegaEngine::logits`]).
//! - prefill [`Qwen3DenseOmegaEngine::step_batch`]: the same layer over `m` rows. An FP8 projection
//!   runs omega's FP8 GEMM in place (`m <= 1024`, "bit-identical to calling the GEMV once per
//!   row", `omega_gpu_fp8gemm_api.h:19-21`), a bf16 projection the GEMV once per row; attention is
//!   the split path, one launch per row.
//! - [`Qwen3DenseOmegaEngine::generate_with`]: prefix reuse, prefill, sampling through a caller's
//!   sampler, stop ids, cancel; [`Qwen3DenseOmegaEngine::rewind`] and KV export/restore as the 30B.
//!
//! The same engine runs on the host ([`HostDenseUploader`] + [`HostPoolKv`]): that is oracle A of
//! section 2.4 (same code, host GEMVs, same bf16 KV rounding), compared layer by layer through
//! [`DenseLayerTrace`]. Negative controls: [`MutantUploader`] (k|v row swap, FP8 scale-grid
//! transpose), the attention mutant ([`Qwen3DenseOmegaEngine::attention_mutant`]) and omega's
//! kernel mutants (`aien_omega_fp8::test_set_mutant`, `aien_omega_bf16::test_set_mutant`).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use super::{
    load_qwen3_dense_omega, DenseGemv, DenseLoadReport, DenseTensor, DenseUploader, HostFp8Gemv,
    Qwen3DenseConfig, Qwen3DenseOmegaWeights,
};
use crate::qwen3_coder::rmsnorm_per_head;
use crate::qwen3_moe::{bf16_to_f32, FP8_BLOCK};
use crate::qwen3_moe_omega::{
    sample_next, FinishReason, GenerateOutcome, GenerateParams, HostPoolKv, KvAttention,
    KvGeometry, OmegaPagedKv, StepTiming,
};
use crate::tensor::{apply_rope, rmsnorm};

// ---------------------------------------------------------------------------
// Device and host GEMVs.
// ---------------------------------------------------------------------------

/// omega's bf16 streaming GEMV over one `[n][k]` matrix (projections of a bf16 checkpoint and
/// every lm_head), behind a lock (`Bf16Tensor` is not `Sync`). The kernel stages x in shared memory
/// (k * 4 bytes, at most 16384: omega_blackwell_qmd.h `OMEGA_BW_QMD_MAX_SHARED_BYTES`), so a wider
/// matrix is held as column slices of at most `BF16_MAX_SLICE_K` and their partial sums are added
/// in slice order (Qwen3-4B down: k 9728 = 4096 + 4096 + 1536).
#[derive(Debug)]
pub struct OmegaBf16Gemv(Mutex<Vec<aien_omega_bf16::Bf16Tensor>>, usize, usize);

/// Widest x one bf16 launch stages: 16384 shared bytes / 4.
pub const BF16_MAX_SLICE_K: usize = 4096;

impl DenseGemv for OmegaBf16Gemv {
    fn n(&self) -> usize {
        self.1
    }
    fn k(&self) -> usize {
        self.2
    }
    fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String> {
        let slices = self.0.lock().map_err(|_| "bf16 tensor lock poisoned")?;
        if let [t] = slices.as_slice() {
            return Ok(t.gemv_f32(x, y).map_err(|e| e.to_string())?.elapsed_ns);
        }
        let mut part = vec![0f32; y.len()];
        let mut ns = 0;
        for (i, t) in slices.iter().enumerate() {
            let c0 = i * BF16_MAX_SLICE_K;
            let c1 = (c0 + BF16_MAX_SLICE_K).min(self.2);
            let out = if i == 0 { &mut *y } else { &mut part[..] };
            ns += t.gemv_f32(&x[c0..c1], out).map_err(|e| e.to_string())?.elapsed_ns;
            if i > 0 {
                y.iter_mut().zip(&part).for_each(|(a, b)| *a += b);
            }
        }
        Ok(ns)
    }
}

/// omega's FP8 block-scaled GEMV (decode) and FP8 GEMM (prefill) over one `[n][k]` projection
/// uploaded as a one-expert tensor, behind a lock.
#[derive(Debug)]
pub struct OmegaFp8DenseGemv(Mutex<aien_omega_fp8::Fp8Tensor>, usize, usize);

impl DenseGemv for OmegaFp8DenseGemv {
    fn n(&self) -> usize {
        self.1
    }
    fn k(&self) -> usize {
        self.2
    }
    fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String> {
        let t = self.0.lock().map_err(|_| "fp8 tensor lock poisoned")?;
        let info = t.gemv_f32(0, x, y).map_err(|e| e.to_string())?;
        Ok(info.elapsed_ns)
    }
    fn rows_f32(&self, m: usize, x: &[f32], y: &mut [f32]) -> Result<(u64, u64), String> {
        if m <= 1 || !aien_omega_fp8::has_gemm() {
            return rows_by_gemv(self, m, x, y);
        }
        let (n, k) = (self.1, self.2);
        check_rows(n, k, m, x, y)?;
        let t = self.0.lock().map_err(|_| "fp8 tensor lock poisoned")?;
        let (mut ns, mut launches) = (0u64, 0u64);
        let mut r0 = 0usize;
        while r0 < m {
            let rows = (m - r0).min(aien_omega_fp8::GEMM_MAX_ROWS);
            let ids = vec![0u32; rows];
            let token_rows: Vec<u32> = (0..rows as u32).collect();
            let info = t
                .gemm_f32(
                    &ids,
                    &token_rows,
                    &x[r0 * k..(r0 + rows) * k],
                    &mut y[r0 * n..(r0 + rows) * n],
                )
                .map_err(|e| e.to_string())?;
            ns += info.elapsed_ns;
            launches += 1;
            r0 += rows;
        }
        Ok((ns, launches))
    }
}

fn check_rows(n: usize, k: usize, m: usize, x: &[f32], y: &[f32]) -> Result<(), String> {
    if x.len() != m * k || y.len() != m * n {
        return Err(format!(
            "rows [{n}][{k}] x {m}: given x {} y {}",
            x.len(),
            y.len()
        ));
    }
    Ok(())
}

/// The default multi-row path: the GEMV once per row, in row order.
pub fn rows_by_gemv<G: DenseGemv + ?Sized>(
    g: &G,
    m: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(u64, u64), String> {
    let (n, k) = (g.n(), g.k());
    check_rows(n, k, m, x, y)?;
    let mut ns = 0u64;
    for (xr, yr) in x.chunks_exact(k).zip(y.chunks_exact_mut(n)) {
        ns += g.gemv_f32(xr, yr)?;
    }
    Ok((ns, m as u64))
}

/// One `[n][k]` bf16 matrix on the host: `y[r] = sum_c bf16(W[r][c]) * x[c]` in f32, columns in
/// order per row (rows in parallel; each row's sum is sequential, so the result does not depend
/// on the thread count). The bf16 half of oracle A.
#[derive(Debug)]
pub struct HostBf16Gemv {
    n: usize,
    k: usize,
    bits: Vec<u16>,
}

impl HostBf16Gemv {
    pub fn new(n: usize, k: usize, bits: Vec<u16>) -> Result<Self, String> {
        if bits.len() != n * k {
            return Err(format!("{} words for [{n}][{k}]", bits.len()));
        }
        Ok(Self { n, k, bits })
    }
    /// The stored `[n][k]` bf16 words.
    pub fn bits(&self) -> &[u16] {
        &self.bits
    }
}

impl DenseGemv for HostBf16Gemv {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<u64, String> {
        let k = self.k;
        if x.len() != k || y.len() != self.n {
            return Err(format!(
                "gemv [{}][{k}] given x {} y {}",
                self.n,
                x.len(),
                y.len()
            ));
        }
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            let row = &self.bits[r * k..(r + 1) * k];
            *out = row.iter().zip(x).map(|(&w, &v)| bf16_to_f32(w) * v).sum();
        });
        Ok(0)
    }
}

/// Keeps every matrix on the host (oracle A): bf16 as [`HostBf16Gemv`], FP8 as [`HostFp8Gemv`]
/// (the chip's formula).
#[derive(Debug, Default)]
pub struct HostDenseUploader;

impl DenseUploader for HostDenseUploader {
    fn upload_bf16(
        &self,
        _layer: Option<usize>,
        _what: DenseTensor,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        Ok(Arc::new(HostBf16Gemv::new(n, k, bits.to_vec())?))
    }
    fn upload_fp8(
        &self,
        _layer: Option<usize>,
        _what: DenseTensor,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        Ok(Arc::new(HostFp8Gemv::new(n, k, w.to_vec(), scales.to_vec())?))
    }
}

/// Makes every matrix resident on the GB10: bf16 through omega's `Bf16Tensor` (NaN/Inf refused at
/// upload), FP8 through omega's `Fp8Tensor` with one expert (NaN codes refused at upload).
#[derive(Debug, Default)]
pub struct OmegaDenseUploader;

impl OmegaDenseUploader {
    /// Whether this binary links omega's bf16 GEMV and FP8 GEMV (else every upload is refused).
    pub fn native() -> (bool, bool) {
        (aien_omega_bf16::is_native(), aien_omega_fp8::is_native())
    }
}

impl DenseUploader for OmegaDenseUploader {
    fn upload_bf16(
        &self,
        _layer: Option<usize>,
        _what: DenseTensor,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        if !aien_omega_bf16::is_native() {
            return Err("this binary carries no omega bf16 GEMV (built without AIEN_OMEGA_DIR)".into());
        }
        let mut slices = Vec::new();
        for c0 in (0..k).step_by(BF16_MAX_SLICE_K) {
            let w = BF16_MAX_SLICE_K.min(k - c0);
            let t = if w == k {
                aien_omega_bf16::Bf16Tensor::upload(n, k, bits)
            } else {
                let cols: Vec<u16> = bits.chunks_exact(k).flat_map(|row| &row[c0..c0 + w]).copied().collect();
                aien_omega_bf16::Bf16Tensor::upload(n, w, &cols)
            };
            slices.push(t.map_err(|e| e.to_string())?);
        }
        Ok(Arc::new(OmegaBf16Gemv(Mutex::new(slices), n, k)))
    }
    fn upload_fp8(
        &self,
        _layer: Option<usize>,
        _what: DenseTensor,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        if !aien_omega_fp8::is_native() {
            return Err("this binary carries no omega FP8 GEMV (built without AIEN_OMEGA_DIR)".into());
        }
        let t = aien_omega_fp8::Fp8Tensor::upload(1, n, k, w, scales).map_err(|e| e.to_string())?;
        Ok(Arc::new(OmegaFp8DenseGemv(Mutex::new(t), n, k)))
    }
}

/// A deliberately wrong upload (negative controls of stage 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseUploadMutant {
    /// q|k|v stored with the k rows and the v rows exchanged (FP8: the bytes and the scale rows).
    SwapKv,
    /// FP8 only: every scale grid `[a][b]` stored as the transpose of the grid read as `[b][a]`.
    TransposeScales,
}

impl DenseUploadMutant {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "swap-kv" => Ok(Self::SwapKv),
            "transpose-scales" => Ok(Self::TransposeScales),
            other => Err(format!(
                "unknown upload mutant {other:?} (swap-kv, transpose-scales)"
            )),
        }
    }
}

/// Wraps an uploader and applies a [`DenseUploadMutant`] to what it uploads.
pub struct MutantUploader<'a> {
    pub inner: &'a dyn DenseUploader,
    pub mutant: DenseUploadMutant,
    pub q_rows: usize,
    pub kv_rows: usize,
    /// The checkpoint is FP8 (its lm_head is still bf16).
    pub fp8: bool,
}

impl<'a> MutantUploader<'a> {
    pub fn new(inner: &'a dyn DenseUploader, mutant: DenseUploadMutant, c: &Qwen3DenseConfig) -> Self {
        Self {
            inner,
            mutant,
            q_rows: c.q_dim(),
            kv_rows: c.kv_dim(),
            fp8: c.format == super::DenseWeightFormat::Fp8Block128,
        }
    }
}

/// Exchanges rows `a..a+rows` and `b..b+rows` (`a + rows <= b`) of a row-major `width` matrix.
fn swap_rows<T>(m: &mut [T], width: usize, a: usize, b: usize, rows: usize) {
    let (lo, hi) = m.split_at_mut(b * width);
    lo[a * width..(a + rows) * width].swap_with_slice(&mut hi[..rows * width]);
}

impl DenseUploader for MutantUploader<'_> {
    fn upload_bf16(
        &self,
        layer: Option<usize>,
        what: DenseTensor,
        n: usize,
        k: usize,
        bits: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        if self.mutant == DenseUploadMutant::SwapKv && what == DenseTensor::Qkv {
            let mut b = bits.to_vec();
            swap_rows(&mut b, k, self.q_rows, self.q_rows + self.kv_rows, self.kv_rows);
            return self.inner.upload_bf16(layer, what, n, k, &b);
        }
        if self.mutant == DenseUploadMutant::TransposeScales && !self.fp8 {
            return Err("the transpose-scales mutant needs an FP8 checkpoint".into());
        }
        self.inner.upload_bf16(layer, what, n, k, bits)
    }
    fn upload_fp8(
        &self,
        layer: Option<usize>,
        what: DenseTensor,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        let (mut w2, mut s2) = (w.to_vec(), scales.to_vec());
        let (a, b) = (n / FP8_BLOCK, k / FP8_BLOCK);
        match self.mutant {
            DenseUploadMutant::SwapKv if what == DenseTensor::Qkv => {
                let (q, kv) = (self.q_rows, self.kv_rows);
                swap_rows(&mut w2, k, q, q + kv, kv);
                swap_rows(&mut s2, b, q / FP8_BLOCK, (q + kv) / FP8_BLOCK, kv / FP8_BLOCK);
            }
            DenseUploadMutant::SwapKv => {}
            DenseUploadMutant::TransposeScales => {
                for r in 0..a {
                    for c in 0..b {
                        s2[r * b + c] = scales[c * a + r];
                    }
                }
            }
        }
        self.inner.upload_fp8(layer, what, n, k, &w2, &s2)
    }
}

// ---------------------------------------------------------------------------
// The engine.
// ---------------------------------------------------------------------------

/// What one traced layer did for one position (oracle A compares these between the chip engine
/// and the host engine).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayerTrace {
    pub layer: usize,
    pub pos: usize,
    pub x_in: Vec<f32>,
    /// Attention output before o (`[q_heads x head_dim]`).
    pub attn: Vec<f32>,
    /// o projection of `attn`.
    pub attn_proj: Vec<f32>,
    /// Post-attention RMSNorm of the residual (the MLP input).
    pub post: Vec<f32>,
    /// down(SwiGLU(gate|up(post))).
    pub mlp_out: Vec<f32>,
    pub x_out: Vec<f32>,
}

/// Every run setting of a [`Qwen3DenseOmegaEngine`] that can change a K/V byte (persistent KV
/// identity, as `OmegaKvSwitches` for the 30B; fail closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseKvSwitches {
    pub attention_path: &'static str,
    /// Rows per prefill launch (the FP8 GEMM is bit-equal to the GEMV, listed anyway).
    pub prefill_chunk: usize,
    pub attention_mutant: bool,
    pub trace_layers: bool,
}

/// Most prompt tokens one [`Qwen3DenseOmegaEngine::step_batch`] layer pass takes by default (one
/// FP8 GEMM launch per projection, `GEMM_MAX_ROWS`).
pub const DENSE_PREFILL_CHUNK: usize = aien_omega_fp8::GEMM_MAX_ROWS;

/// The dense Qwen3 engine: one sequence, decode by `step`, prefill by `step_batch`.
pub struct Qwen3DenseOmegaEngine {
    weights: Qwen3DenseOmegaWeights,
    kv: Box<dyn KvAttention>,
    /// Layers whose activations are recorded into `traces`.
    pub trace_layers: Vec<usize>,
    pub traces: Vec<DenseLayerTrace>,
    /// Negative control: attention over `0..pos` instead of `0..=pos` (the newest position dropped,
    /// off by one; position 0 still sees itself). Layer parity and tokens must then fail.
    pub attention_mutant: bool,
    pub last_timing: StepTiming,
    /// Prompt tokens per `step_batch` pass inside `generate_with` (1 = token at a time).
    pub prefill_chunk: usize,
    pos: usize,
    history: Vec<u32>,
}

impl Qwen3DenseOmegaEngine {
    /// The cache must have the config's geometry.
    pub fn new(weights: Qwen3DenseOmegaWeights, kv: Box<dyn KvAttention>) -> Result<Self, String> {
        let want = KvGeometry::from(&weights.config);
        let have = kv.geometry();
        if have != want {
            return Err(format!(
                "kv cache geometry {have:?} is not the model's {want:?}"
            ));
        }
        Ok(Self {
            weights,
            kv,
            trace_layers: Vec::new(),
            traces: Vec::new(),
            attention_mutant: false,
            last_timing: StepTiming::default(),
            prefill_chunk: DENSE_PREFILL_CHUNK,
            pos: 0,
            history: Vec::new(),
        })
    }

    /// Loads `dir` through `uploader` with the chip's paged bf16 KV of `max_context` tokens.
    pub fn load_omega(
        dir: &Path,
        uploader: &dyn DenseUploader,
        max_context: usize,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, DenseLoadReport), String> {
        let (weights, report) = load_qwen3_dense_omega(dir, uploader, progress)?;
        let kv = OmegaPagedKv::new(&KvGeometry::from(&weights.config), max_context)?;
        Ok((Self::new(weights, Box::new(kv))?, report))
    }

    /// Loads `dir` on the host (oracle A): host GEMVs, the host bf16 pool and its attention.
    pub fn load_host(
        dir: &Path,
        uploader: &dyn DenseUploader,
        max_context: usize,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(Self, DenseLoadReport), String> {
        let (weights, report) = load_qwen3_dense_omega(dir, uploader, progress)?;
        let kv = HostPoolKv::new(&KvGeometry::from(&weights.config), max_context)?;
        Ok((Self::new(weights, Box::new(kv))?, report))
    }

    pub fn config(&self) -> &Qwen3DenseConfig {
        &self.weights.config
    }
    pub fn weights(&self) -> &Qwen3DenseOmegaWeights {
        &self.weights
    }
    /// "split" (chip paged attention), "host-pool" (oracle) or a test cache's name.
    pub fn attention_path(&self) -> &'static str {
        self.kv.path_name()
    }
    pub fn max_context_tokens(&self) -> usize {
        self.kv.capacity()
    }
    pub fn kv_resident(&self) -> bool {
        self.kv.resident()
    }
    /// Tokens already in the KV.
    pub fn position(&self) -> usize {
        self.pos
    }
    /// Tokens whose KV is stored (positions `0..position()`).
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.kv.truncate(0);
        self.pos = 0;
        self.history.clear();
        self.traces.clear();
    }

    /// Keeps the first `len` stored tokens; the next step is at `len`.
    pub fn rewind(&mut self, len: usize) {
        let len = len.min(self.pos);
        self.kv.truncate(len);
        self.history.truncate(len);
        self.pos = len;
    }

    /// The longest common prefix of the stored history and `prompt`, capped at
    /// `prompt.len() - 1` (the last prompt token is always stepped).
    pub fn reusable_prefix(&self, prompt: &[u32]) -> usize {
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        common.min(prompt.len().saturating_sub(1))
    }

    /// The K/V of stored positions `start..start + len` as the `kvio` payload.
    pub fn export_kv(&mut self, start: usize, len: usize) -> Result<Vec<u8>, String> {
        if start + len > self.pos {
            return Err(format!(
                "kv export {start}+{len}: only {} positions stored",
                self.pos
            ));
        }
        self.kv.export_blocks(start, len).unwrap_or_else(|| {
            Err(format!(
                "the {} attention cache cannot export its KV",
                self.kv.path_name()
            ))
        })
    }

    /// Replaces the stored sequence by `tokens` with the K/V of `payload`; on error the engine
    /// is reset.
    pub fn restore_kv(&mut self, tokens: &[u32], payload: &[u8]) -> Result<(), String> {
        self.reset();
        if tokens.is_empty() || tokens.len() >= self.kv.capacity() {
            return Err(format!(
                "kv restore of {} tokens, context is {}",
                tokens.len(),
                self.kv.capacity()
            ));
        }
        let r = self
            .kv
            .import_blocks(0, tokens.len(), payload)
            .unwrap_or_else(|| {
                Err(format!(
                    "the {} attention cache cannot import KV",
                    self.kv.path_name()
                ))
            });
        match r {
            Ok(()) => {
                self.history = tokens.to_vec();
                self.pos = tokens.len();
                Ok(())
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    pub fn kv_switches(&self) -> DenseKvSwitches {
        DenseKvSwitches {
            attention_path: self.kv.path_name(),
            prefill_chunk: self.prefill_chunk,
            attention_mutant: self.attention_mutant,
            trace_layers: !self.trace_layers.is_empty(),
        }
    }

    /// Decode: runs `token` at the next position through every layer and returns the final-normed
    /// hidden state (feed it to [`Self::logits`]). Records `last_timing`.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.forward(&[token])
    }

    /// Prefill: runs `tokens` at the next positions, every layer over all rows at once (FP8: one
    /// GEMM launch per projection per 1024 rows), and returns the final-normed hidden state of
    /// the last token. Bit-equal to `step` once per token wherever the multi-row kernel is
    /// bit-equal to the GEMV (omega's FP8 GEMM, every host path).
    pub fn step_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("step_batch of no tokens".into());
        }
        self.forward(tokens)
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let (inter, qkv_n) = (c.intermediate, c.qkv_n());
        let (qh, kvh, theta) = (c.q_heads, c.kv_heads, c.rope_theta);
        let m = tokens.len();
        let p0 = self.pos;
        if p0 + m > self.kv.capacity() {
            return Err(format!(
                "context full: {p0} + {m} of {}",
                self.kv.capacity()
            ));
        }
        let mut x = Vec::with_capacity(m * h);
        for &t in tokens {
            x.extend(self.weights.embed_row(t)?);
        }
        let mut xn = vec![0.0f32; m * h];
        let mut qkv = vec![0.0f32; m * qkv_n];
        let mut qn = vec![0.0f32; m * qd];
        let mut kn = vec![0.0f32; kvd];
        let mut attn = vec![0.0f32; m * qd];
        let mut attn_proj = vec![0.0f32; m * h];
        let mut gu = vec![0.0f32; m * 2 * inter];
        let mut act = vec![0.0f32; m * inter];
        let mut mlp = vec![0.0f32; m * h];

        let rows = |g: &Arc<dyn DenseGemv>,
                    x: &[f32],
                    y: &mut [f32],
                    timing: &mut StepTiming|
         -> Result<(), String> {
            let (ns, launches) = if m == 1 {
                (g.gemv_f32(x, y)?, 1)
            } else {
                g.rows_f32(m, x, y)?
            };
            timing.chip_ns += ns;
            timing.launches += launches;
            Ok(())
        };

        for (li, layer) in self.weights.layers.iter().enumerate() {
            let traced = self.trace_layers.contains(&li);
            let x_in = if traced { x.clone() } else { Vec::new() };

            // --- attention ---
            let t0 = Instant::now();
            for (xr, or) in x.chunks_exact(h).zip(xn.chunks_exact_mut(h)) {
                rmsnorm(xr, &layer.input_norm, eps, or);
            }
            timing.host_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            rows(&layer.qkv, &xn, &mut qkv, &mut timing)
                .map_err(|e| format!("layer {li} qkv: {e}"))?;
            for r in 0..m {
                let th = Instant::now();
                let row = &qkv[r * qkv_n..(r + 1) * qkv_n];
                let q = &mut qn[r * qd..(r + 1) * qd];
                rmsnorm_per_head(&row[..qd], &layer.q_norm, eps, qh, hd, q);
                rmsnorm_per_head(&row[qd..qd + kvd], &layer.k_norm, eps, kvh, hd, &mut kn);
                let pos = p0 + r;
                apply_rope(q, &mut kn, pos, qh, kvh, hd, theta);
                self.kv.append(li, pos, &kn, &row[qd + kvd..])?;
                timing.host_ns += th.elapsed().as_nanos() as u64;
            }
            for r in 0..m {
                // The mutant drops the newest position (off by one), except at position 0.
                let ctx = if self.attention_mutant {
                    (p0 + r).max(1)
                } else {
                    p0 + r + 1
                };
                timing.chip_ns += self.kv.attend(
                    li,
                    &qn[r * qd..(r + 1) * qd],
                    ctx,
                    &mut attn[r * qd..(r + 1) * qd],
                )?;
                timing.launches += 1;
            }
            rows(&layer.o, &attn, &mut attn_proj, &mut timing)
                .map_err(|e| format!("layer {li} o: {e}"))?;
            timing.attention_ns += t1.elapsed().as_nanos() as u64;

            // --- residual, post norm, MLP ---
            let t2 = Instant::now();
            for (xi, p) in x.iter_mut().zip(&attn_proj) {
                *xi += p;
            }
            for (xr, or) in x.chunks_exact(h).zip(xn.chunks_exact_mut(h)) {
                rmsnorm(xr, &layer.post_norm, eps, or);
            }
            timing.host_ns += t2.elapsed().as_nanos() as u64;
            let t3 = Instant::now();
            rows(&layer.gate_up, &xn, &mut gu, &mut timing)
                .map_err(|e| format!("layer {li} gate_up: {e}"))?;
            let th = Instant::now();
            for (g, a) in gu.chunks_exact(2 * inter).zip(act.chunks_exact_mut(inter)) {
                swiglu(g, inter, a);
            }
            timing.host_ns += th.elapsed().as_nanos() as u64;
            rows(&layer.down, &act, &mut mlp, &mut timing)
                .map_err(|e| format!("layer {li} down: {e}"))?;
            for (xi, d) in x.iter_mut().zip(&mlp) {
                *xi += d;
            }
            timing.experts_ns += t3.elapsed().as_nanos() as u64;

            if traced {
                for r in 0..m {
                    let s = |v: &[f32], w: usize| v[r * w..(r + 1) * w].to_vec();
                    self.traces.push(DenseLayerTrace {
                        layer: li,
                        pos: p0 + r,
                        x_in: s(&x_in, h),
                        attn: s(&attn, qd),
                        attn_proj: s(&attn_proj, h),
                        post: s(&xn, h),
                        mlp_out: s(&mlp, h),
                        x_out: s(&x, h),
                    });
                }
            }
        }
        let t4 = Instant::now();
        let mut hidden = vec![0.0f32; h];
        rmsnorm(&x[(m - 1) * h..], &self.weights.final_norm, eps, &mut hidden);
        timing.host_ns += t4.elapsed().as_nanos() as u64;
        self.pos = p0 + m;
        self.history.extend_from_slice(tokens);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hidden)
    }

    /// Vocabulary logits of a final hidden state (one lm_head launch), added to `last_timing`.
    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let mut logits = vec![0.0f32; self.weights.config.vocab];
        let ns = self
            .weights
            .lm_head
            .gemv_f32(hidden, &mut logits)
            .map_err(|e| format!("lm_head: {e}"))?;
        let dt = t0.elapsed().as_nanos() as u64;
        self.last_timing.lm_head_ns += dt;
        self.last_timing.total_ns += dt;
        self.last_timing.chip_ns += ns;
        self.last_timing.launches += 1;
        Ok(logits)
    }

    /// Prefill `prompt` (reusing the stored prefix) then decode with `sampler(logits, step)`;
    /// `on_token` gets every emitted token and may return `false` to cancel. Stop ids are
    /// consumed, never emitted.
    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() >= self.kv.capacity() {
            return Err(format!(
                "prompt of {} tokens, context is {}",
                prompt.len(),
                self.kv.capacity()
            ));
        }
        let reused = self.reusable_prefix(prompt);
        if reused > 0 {
            self.rewind(reused);
        } else {
            self.reset();
        }
        let t0 = Instant::now();
        let mut hidden = Vec::new();
        let mut prefill_timing = StepTiming::default();
        for chunk in prompt[reused..].chunks(self.prefill_chunk.max(1)) {
            hidden = self.step_batch(chunk)?;
            prefill_timing.add(&self.last_timing);
        }
        let prefill_ms = t0.elapsed().as_millis() as u64;
        self.last_timing = StepTiming::default();
        let t1 = Instant::now();
        let mut decode_timing = StepTiming::default();
        let (mut emitted, mut step) = (0usize, 0usize);
        let mut finish = FinishReason::Length;
        while emitted < params.max_new_tokens {
            let logits = self.logits(&hidden)?;
            decode_timing.add(&self.last_timing);
            let next = sampler(&logits, step);
            step += 1;
            if params.stop_token_ids.contains(&next) {
                finish = FinishReason::Stop;
                break;
            }
            emitted += 1;
            if !on_token(next) {
                finish = FinishReason::Cancelled;
                break;
            }
            if emitted >= params.max_new_tokens || self.pos >= self.kv.capacity() {
                break;
            }
            hidden = self.step(next)?;
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms: t1.elapsed().as_millis() as u64,
            prompt_tokens: prompt.len(),
            reused_prompt_tokens: reused,
            decode_timing,
            prefill_timing,
        })
    }

    /// `generate_with` using [`sample_next`] (the door's sampling rule).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        let p = params.clone();
        self.generate_with(
            prompt,
            params,
            &mut |logits, step| sample_next(logits, &p, step),
            on_token,
        )
    }
}

/// `act[j] = silu(gu[j]) * gu[i + j]` (gate rows first, then up rows).
fn swiglu(gu: &[f32], i: usize, act: &mut [f32]) {
    for j in 0..i {
        let g = gu[j];
        act[j] = g / (1.0 + (-g).exp()) * gu[i + j];
    }
}

#[cfg(test)]
#[path = "qwen3_dense_omega_engine_tests.rs"]
mod tests;
