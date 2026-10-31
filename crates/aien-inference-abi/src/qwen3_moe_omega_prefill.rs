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

//! Whole-prompt prefill for OWN-30B (Qwen3-30B-A3B FP8) on omega's own prefill kernels
//! (W-L1G-R, 2026-10-14). New file; not wired into the engine or the server yet.
//!
//! One call takes the whole prompt (`m` tokens at consecutive positions `pos0 ..`) through every
//! layer with a fixed number of launches per layer, whatever `m` is:
//!
//! | step | where | launches |
//! |---|---|---|
//! | input RMSNorm | host | |
//! | quantise x (f32, k = hidden) | chip | 1 |
//! | q\|k\|v QMMA GEMM (PLAIN, 1 expert) | chip | 1 |
//! | attention for all m tokens ([`PrefillAttention`]) | chip | 1 call |
//! | quantise attention out (f32, k = 4096) | chip | 1 |
//! | o QMMA GEMM (PLAIN) | chip | 1 |
//! | residual, post RMSNorm, router logits, top-k ([`MoeBatchPlan`]) | host | |
//! | counting sort of the (token, expert) rows by expert ([`MoeBatchPlan`]) | host | |
//! | quantise post (f32, k = hidden) | chip | 1 |
//! | gate\|up QMMA GEMM, SWIGLU_NAT epilogue, rows gathered by token index | chip | 1 |
//! | quantise act (bf16, k = moe_intermediate) | chip | 1 |
//! | down QMMA GEMM (PLAIN, sorted rows) | chip | 1 |
//! | weighted combine + residual | host | |
//!
//! It returns the last token's final-normed hidden state (the lm_head input).
//!
//! Weights: the FP8 tensors the loader made resident ([`AttnWeights::Fp8`], [`ExpertWeights::Fp8`])
//! are read IN PLACE: the tile tables carry each expert's `(w_va, s_va)` from
//! [`Fp8Gemv::tensor_entry`] (omega_gpu_fp8_tensor_entry), collected once at construction.
//!
//! Reused / reimplemented: routing and the expert sort are [`MoeBatchPlan::from_logits`] (public:
//! softmax, top-k, renormalise, then a stable counting sort by expert). The router logits and the
//! combine are REIMPLEMENTED here (the engine's `router_logits` / `combine_rows` are private), in
//! the same summation order. The tile table is `aien_omega_pf::build_tiles` (a Rust mirror of
//! `omega_gpu_pf_qgemm_tiles_va`, checked equal to the C builder in that crate's tests).
//!
//! Host <-> device traffic. The driver makes NO staging copies of its own:
//!  * HOST buffers (GPU-uncached session memory, `omega_gpu_pf_alloc_host`) are written or read by
//!    the CPU in place: the quantiser inputs (normed x, post; m x hidden f32 each), the tile tables
//!    and the row index (small), and the two GEMM outputs the host needs (o projection m x hidden
//!    f32, down rows g x hidden f32 with g = m * top_k). Rule nvos.h:1117: anything the host reads
//!    or writes stays GPU-uncached.
//!  * DEVICE buffers (GPU-cached, CPU never touches them): quantised activations and scales, the
//!    SwiGLU act rows, and on the device-attention path the q|k|v rows and the attention output.
//!  * Attention, host entry (`omega_gpu_prefill_attn`): q|k|v is read from and the output written
//!    to HOST buffers; omega itself copies them through its staging buffers (inside the call).
//!    The device entry (`omega_gpu_prefill_attn_dev`, taken when [`PrefillAttention::has_dev`])
//!    keeps both on the device: no copies.
//!
//! Per layer the CPU therefore touches about (3m + g) x hidden x 4 bytes of uncached memory; the
//! down readback (g x hidden) dominates. A device combine kernel would remove it (not in omega yet).
//!
//! Wiring into `Qwen3MoeOmegaEngine` (NOT done here; owner's steps):
//!  1. `OmegaFusedKv` (qwen3_moe_omega_dattn.rs): add `prefill_attention(&mut self) ->
//!     aien_omega_pf::OmegaPrefillAttention<'_>` over its private `cache`, and `mark_filled(&mut
//!     self, upto)` setting every layer's private `filled[l] = upto` (the prefill appends K/V for
//!     `pos0 .. upto` in all layers, the fused decode step checks `pos == filled[layer]`).
//!  2. `KvAttention`: a way to reach that (e.g. `fn fused(&mut self) -> Option<&mut OmegaFusedKv>`,
//!     default `None`), since the engine holds `kv: Box<dyn KvAttention>`.
//!  3. Engine: a field `whole_prompt: Option<WholePromptPrefill<aien_omega_pf::OmegaPfDevice>>`,
//!     built in `new()` when `aien_omega_pf::is_native() && aien_omega_pf::has_attention()` and
//!     [`layer_entries`] succeeds (FP8 everywhere, `Fp8Gemv::tensor_entry` linked).
//!  4. In `generate_with`, replace the `prompt[reused..].chunks(prefill_chunk)` loop when that
//!     field is set and `trace_layers` is empty and `routing_mutant` is off: `pos0 = self.pos`
//!     (after `rewind(reused)` / `reset()`), `hidden = wp.run(&self.weights, &mut
//!     fused.prefill_attention(), &prompt[reused..], pos0)?`, then `fused.mark_filled(pos0 + m)`,
//!     `self.pos = pos0 + m`, `self.history.extend_from_slice(&prompt[reused..])`, and map
//!     `wp.stats` into `last_timing`. `pos0 + m < kv.capacity()` is already checked there.
//!  5. Routing here is host routing (no per-token device router launches).
use std::time::Instant;

use aien_omega_pf::{
    build_tiles, Layout, PfBuffer, PrefillAttention, PrefillDevice, QgemmCall, QuantIn, TILE_BYTES,
};
use rayon::prelude::*;

use crate::moe_plan::MoeBatchPlan;
use crate::qwen3_moe_omega::{
    AttnWeights, ExpertWeights, Fp8Gemv, Qwen3MoeConfig, Qwen3MoeOmegaWeights,
};
use crate::tensor::rmsnorm;

/// Scale of the attention output (1.0: the quantiser picks its own power-of-two scales).
pub const ATTN_OUT_SCALE: f32 = 1.0;

/// Device addresses of one layer's FP8 tensors, per expert `(w_va, s_va)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerEntries {
    pub qkv: (u64, u64),
    pub o: (u64, u64),
    pub gu_w: Vec<u64>,
    pub gu_s: Vec<u64>,
    pub dn_w: Vec<u64>,
    pub dn_s: Vec<u64>,
}

fn check_tensor(
    what: &str,
    t: &dyn Fp8Gemv,
    experts: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    if t.experts() != experts || t.n() != n || t.k() != k {
        return Err(format!(
            "{what}: tensor is {} x [{}][{}], the whole-prompt path needs {experts} x [{n}][{k}]",
            t.experts(),
            t.n(),
            t.k()
        ));
    }
    Ok(())
}

/// Collects every layer's in-place weight addresses, checking the shapes the GEMMs assume.
pub fn layer_entries(weights: &Qwen3MoeOmegaWeights) -> Result<Vec<LayerEntries>, String> {
    let c = &weights.config;
    let (h, qd, e, i) = (c.hidden, c.q_dim(), c.experts, c.moe_intermediate);
    if weights.layers.len() != c.layers {
        return Err(format!(
            "{} layers of weights, config says {}",
            weights.layers.len(),
            c.layers
        ));
    }
    weights
        .layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) = (&layer.attn, &layer.experts) else {
                return Err(format!("layer {l}: the whole-prompt path needs FP8 attention and experts (WeightFormat::Fp8All)"));
            };
            check_tensor(&format!("layer {l} q|k|v"), qkv.as_ref(), 1, c.qkv_n(), h)?;
            check_tensor(&format!("layer {l} o"), o.as_ref(), 1, h, qd)?;
            check_tensor(&format!("layer {l} gate|up"), gate_up.as_ref(), e, 2 * i, h)?;
            check_tensor(&format!("layer {l} down"), down.as_ref(), e, h, i)?;
            let entry = |t: &dyn Fp8Gemv, ex: usize, what: &str| t.tensor_entry(ex).map_err(|err| format!("layer {l} {what} expert {ex} address: {err}"));
            let (mut gu_w, mut gu_s, mut dn_w, mut dn_s) = (Vec::with_capacity(e), Vec::with_capacity(e), Vec::with_capacity(e), Vec::with_capacity(e));
            for ex in 0..e {
                let (w, s) = entry(gate_up.as_ref(), ex, "gate|up")?;
                gu_w.push(w);
                gu_s.push(s);
                let (w, s) = entry(down.as_ref(), ex, "down")?;
                dn_w.push(w);
                dn_s.push(s);
            }
            Ok(LayerEntries { qkv: entry(qkv.as_ref(), 0, "q|k|v")?, o: entry(o.as_ref(), 0, "o")?, gu_w, gu_s, dn_w, dn_s })
        })
        .collect()
}

/// The (token, expert) rows grouped by expert, from a routing plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSort {
    /// Rows of each expert, in expert order (the GEMM tile builder's `counts`).
    pub counts: Vec<u32>,
    /// Token of each sorted row (the gate|up GEMM's row index).
    pub rows: Vec<u32>,
    /// Sorted row of each token-major assignment `token * top_k + slot`.
    pub inv: Vec<u32>,
}

impl ExpertSort {
    pub fn from_plan(plan: &MoeBatchPlan) -> Self {
        Self {
            counts: plan
                .expert_offsets
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect(),
            rows: plan.grouped_token_ids.clone(),
            inv: plan.assignment_to_grouped.clone(),
        }
    }
}

/// What a whole-prompt call did. Launch counters count calls into the device / attention.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefillStats {
    pub tokens: usize,
    pub layers: usize,
    pub quant: usize,
    pub qkv: usize,
    pub attn: usize,
    pub o: usize,
    pub gate_up: usize,
    pub down: usize,
    /// Attention ran through the device-pointer entry.
    pub attn_dev: bool,
    pub chip_ns: u64,
    pub attn_ns: u64,
    pub host_ns: u64,
    /// Host-time breakdown by phase (W-L1G-D profile), summed over layers.
    pub prof: HostProf,
    /// Device router-logits and combine launches (W-L1G-D device path; not in [`Self::launches`]).
    pub router: usize,
    pub combine: usize,
}

/// Where the whole-prompt wall time goes (W-L1G-D, 2026-10-14): ns per phase, summed over layers.
/// `dev_ns` is the wall time of the device calls (quantiser, GEMMs, attention) including launch
/// overhead; the other fields are host work. `wall_ns` is the whole `run`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostProf {
    pub embed_ns: u64,
    pub norm_in_ns: u64,
    pub resid_ns: u64,
    pub norm_post_ns: u64,
    pub router_ns: u64,
    pub plan_ns: u64,
    pub tiles_ns: u64,
    pub combine_ns: u64,
    pub final_ns: u64,
    pub dev_ns: u64,
    pub wall_ns: u64,
}

impl PrefillStats {
    pub fn launches(&self) -> usize {
        self.quant + self.qkv + self.attn + self.o + self.gate_up + self.down
    }

    /// Adds the counters of one more pass over the same prompt (chunked prefill): counts and
    /// times sum, `layers` is the per-pass depth, `attn_dev` holds only if every pass used it.
    pub fn absorb(&mut self, p: &PrefillStats) {
        let first = self.tokens == 0 && self.layers == 0;
        self.tokens += p.tokens;
        self.layers = self.layers.max(p.layers);
        self.quant += p.quant;
        self.qkv += p.qkv;
        self.attn += p.attn;
        self.o += p.o;
        self.gate_up += p.gate_up;
        self.down += p.down;
        self.attn_dev = if first {
            p.attn_dev
        } else {
            self.attn_dev && p.attn_dev
        };
        self.chip_ns += p.chip_ns;
        self.attn_ns += p.attn_ns;
        self.host_ns += p.host_ns;
        let (a, b) = (&mut self.prof, &p.prof);
        a.embed_ns += b.embed_ns;
        a.norm_in_ns += b.norm_in_ns;
        a.resid_ns += b.resid_ns;
        a.norm_post_ns += b.norm_post_ns;
        a.router_ns += b.router_ns;
        a.plan_ns += b.plan_ns;
        a.tiles_ns += b.tiles_ns;
        a.combine_ns += b.combine_ns;
        a.final_ns += b.final_ns;
        a.dev_ns += b.dev_ns;
        a.wall_ns += b.wall_ns;
        self.router += p.router;
        self.combine += p.combine;
    }
}

fn as_f32(b: &[u8]) -> Result<&[f32], String> {
    // SAFETY: every bit pattern is a valid f32; align_to only yields the aligned middle.
    let (pre, mid, post) = unsafe { b.align_to::<f32>() };
    if !pre.is_empty() || !post.is_empty() {
        return Err("host buffer not f32-aligned".into());
    }
    Ok(mid)
}

fn as_f32_mut(b: &mut [u8]) -> Result<&mut [f32], String> {
    // SAFETY: as as_f32.
    let (pre, mid, post) = unsafe { b.align_to_mut::<f32>() };
    if !pre.is_empty() || !post.is_empty() {
        return Err("host buffer not f32-aligned".into());
    }
    Ok(mid)
}

fn as_u32_mut(b: &mut [u8]) -> Result<&mut [u32], String> {
    // SAFETY: every bit pattern is a valid u32.
    let (pre, mid, post) = unsafe { b.align_to_mut::<u32>() };
    if !pre.is_empty() || !post.is_empty() {
        return Err("host buffer not u32-aligned".into());
    }
    Ok(mid)
}

fn host<'a, B: PfBuffer>(b: &'a B, what: &str) -> Result<&'a [f32], String> {
    as_f32(
        b.host()
            .ok_or_else(|| format!("{what} is not a host buffer"))?,
    )
}

fn host_mut<'a, B: PfBuffer>(b: &'a mut B, what: &str) -> Result<&'a mut [f32], String> {
    as_f32_mut(
        b.host_mut()
            .ok_or_else(|| format!("{what} is not a host buffer"))?,
    )
}

/// Tile-table regions of the shared HOST tile buffer (one per GEMM of a layer).
#[derive(Debug, Clone, Copy)]
enum Region {
    Qkv = 0,
    O = 1,
    GateUp = 2,
    Down = 3,
}

/// The buffers of one sizing (`m` tokens, `bm` rows per tile). Sizes in the field comments.
struct Bufs<B> {
    m: usize,
    bm: usize,
    max_tiles: usize,
    /// HOST f32 [m][max(hidden, q_dim)]: quantiser input (normed x, host-path attention out, post).
    xin: B,
    /// DEV e4m3 [m + bm][max(hidden, q_dim)] and f32 scales [m + bm][max / 128] (GEMM reads tiles to row0 + bm - 1).
    xq: B,
    xs: B,
    /// HOST f32 [m][qkv_n] (host attention entry) or None.
    qkv_host: Option<B>,
    /// DEV f32 [m][qkv_n] and [m][q_dim] (device attention entry) or None.
    qkv_dev: Option<B>,
    att_dev: Option<B>,
    /// HOST f32 [m][hidden]: o projection.
    oproj: B,
    /// HOST u32 [g + bm]: token of each sorted row (padding rows read token 0).
    rowidx: B,
    /// DEV bf16 [g][moe_intermediate]: SwiGLU act rows.
    act: B,
    /// DEV e4m3 [g + bm][I] and f32 scales [g + bm][I / 128].
    aq: B,
    a_s: B,
    /// HOST f32 [g][hidden]: down rows (sorted order).
    down: B,
    /// HOST 4 regions x max_tiles x 32 bytes.
    tiles: B,
}

/// The whole-prompt prefill driver over a [`PrefillDevice`] (real: `aien_omega_pf::OmegaPfDevice`).
pub struct WholePromptPrefill<D: PrefillDevice> {
    dev: D,
    entries: Vec<LayerEntries>,
    bufs: Option<Bufs<D::Buf>>,
    pub stats: PrefillStats,
    /// GPU-cached copies of the FP8 weights ([`Self::cache_weights`]); empty = read in place.
    cached: Vec<D::Buf>,
    /// Correctness-gate recorder (W-L1G-C, 2026-10-14): `None` = off. See [`WpTrace`].
    pub trace: Option<WpTrace>,
    /// Negative-control mutant (W-L1G-C): `WpMutant::None` in service.
    pub mutant: WpMutant,
    /// Device router logits + combine (W-L1G-D, [`devhost`]): `None` = host path (the default).
    devhost: Option<devhost::DevHost<D::Buf>>,
}

impl<D: PrefillDevice> std::fmt::Debug for WholePromptPrefill<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WholePromptPrefill {{ layers: {}, sized_for: {:?} }}",
            self.entries.len(),
            self.bufs.as_ref().map(|b| b.m)
        )
    }
}

impl<D: PrefillDevice> WholePromptPrefill<D> {
    /// Collects the in-place weight addresses of every layer (no weight copy).
    pub fn new(dev: D, weights: &Qwen3MoeOmegaWeights) -> Result<Self, String> {
        Ok(Self {
            dev,
            entries: layer_entries(weights)?,
            bufs: None,
            stats: PrefillStats::default(),
            cached: Vec::new(),
            trace: None,
            mutant: WpMutant::None,
            devhost: None,
        })
    }

    pub fn device(&self) -> &D {
        &self.dev
    }

    /// Copies every FP8 weight tensor into GPU-cached device memory (one copy kernel per tensor)
    /// and points the tile tables at the copies (the in-place GEMV tensors are GPU-uncached; the
    /// GEMM reuses each weight tile across row tiles, so it wants L2). Returns the bytes copied.
    /// Per expert the omega_gpu_fp8 layout is w `n * k` e4m3 and s `(n / 128) * (k / 128)` f32,
    /// experts contiguous (omega_gpu_fp8_tensor_entry); that contiguity is checked here.
    pub fn cache_weights(&mut self, c: &Qwen3MoeConfig) -> Result<usize, String> {
        if !self.cached.is_empty() {
            return Ok(self.cached.iter().map(|b| b.bytes()).sum());
        }
        let (h, qd, qkv_n, i) = (c.hidden, c.q_dim(), c.qkv_n(), c.moe_intermediate);
        let blk = aien_omega_pf::BLOCK;
        let mut total = 0usize;
        let mut new_entries = self.entries.clone();
        let mut bufs = Vec::new();
        for (l, ent) in new_entries.iter_mut().enumerate() {
            let mut one = |w: &mut [u64],
                           s: &mut [u64],
                           n: usize,
                           k: usize|
             -> Result<(), String> {
                let (wb, sb) = (n * k, (n / blk) * (k / blk) * 4);
                for ex in 0..w.len() {
                    if w[ex] != w[0] + (ex * wb) as u64 || s[ex] != s[0] + (ex * sb) as u64 {
                        return Err(format!("layer {l}: expert {ex} weights are not contiguous; cannot copy in one piece"));
                    }
                }
                for (v, per) in [(w, wb), (s, sb)] {
                    let bytes = per * v.len();
                    let dst = self.dev.alloc_dev(bytes)?;
                    self.dev
                        .copy_va(dst.va(), v[0], bytes)
                        .map_err(|e| format!("layer {l}: weight copy of {bytes} bytes: {e}"))?;
                    for (ex, x) in v.iter_mut().enumerate() {
                        *x = dst.va() + (ex * per) as u64;
                    }
                    total += bytes;
                    bufs.push(dst);
                }
                Ok(())
            };
            let (mut qw, mut qs) = ([ent.qkv.0], [ent.qkv.1]);
            one(&mut qw, &mut qs, qkv_n, h)?;
            ent.qkv = (qw[0], qs[0]);
            let (mut ow, mut os) = ([ent.o.0], [ent.o.1]);
            one(&mut ow, &mut os, h, qd)?;
            ent.o = (ow[0], os[0]);
            one(&mut ent.gu_w, &mut ent.gu_s, 2 * i, h)?;
            one(&mut ent.dn_w, &mut ent.dn_s, h, i)?;
        }
        self.entries = new_entries;
        self.cached = bufs;
        Ok(total)
    }

    /// Tokens the prompt-sized scratch is sized for (0 before the first pass). The scratch is
    /// kept between passes and only regrown, so this is its high-water mark.
    pub fn scratch_tokens(&self) -> usize {
        self.bufs.as_ref().map_or(0, |b| b.m)
    }

    fn ensure(&mut self, c: &Qwen3MoeConfig, m: usize, dev_attn: bool) -> Result<(), String> {
        let bm = self.dev.qgemm_bm();
        if let Some(b) = &self.bufs {
            let mode_ok = if dev_attn {
                b.qkv_dev.is_some()
            } else {
                b.qkv_host.is_some()
            };
            if b.m >= m && b.bm == bm && mode_ok {
                return Ok(());
            }
        }
        self.bufs = None; // free the old sizing before allocating the new one
        let (h, qd, qkv_n, i, e) = (
            c.hidden,
            c.q_dim(),
            c.qkv_n(),
            c.moe_intermediate,
            c.experts,
        );
        let g = m * c.top_k;
        let kmax = h.max(qd);
        let max_tiles = g.div_ceil(bm) + e;
        let d = &mut self.dev;
        let (qkv_host, qkv_dev, att_dev) = if dev_attn {
            (
                None,
                Some(d.alloc_dev(m * qkv_n * 4)?),
                Some(d.alloc_dev(m * qd * 4)?),
            )
        } else {
            (Some(d.alloc_host(m * qkv_n * 4)?), None, None)
        };
        let mut b = Bufs {
            m,
            bm,
            max_tiles,
            xin: d.alloc_host(m * kmax * 4)?,
            xq: d.alloc_dev((m + bm) * kmax)?,
            xs: d.alloc_dev((m + bm) * (kmax / aien_omega_pf::BLOCK) * 4)?,
            qkv_host,
            qkv_dev,
            att_dev,
            oproj: d.alloc_host(m * h * 4)?,
            rowidx: d.alloc_host((g + bm) * 4)?,
            act: d.alloc_dev(g * i * 2)?,
            aq: d.alloc_dev((g + bm) * i)?,
            a_s: d.alloc_dev((g + bm) * (i / aien_omega_pf::BLOCK) * 4)?,
            down: d.alloc_host(g * h * 4)?,
            tiles: d.alloc_host(4 * max_tiles * TILE_BYTES)?,
        };
        // padding rows of the row index read token 0 (a valid row); written once per sizing
        as_u32_mut(
            b.rowidx
                .host_mut()
                .ok_or("row index is not a host buffer")?,
        )?
        .fill(0);
        self.bufs = Some(b);
        Ok(())
    }

    /// Runs the whole prompt `tokens` at positions `pos0 .. pos0 + m - 1` through every layer.
    /// Positions `0 .. pos0 - 1` must already be in `attn`'s KV cache. Returns the last token's
    /// final-normed hidden state.
    pub fn run(
        &mut self,
        weights: &Qwen3MoeOmegaWeights,
        attn: &mut dyn PrefillAttention,
        tokens: &[u32],
        pos0: usize,
    ) -> Result<Vec<f32>, String> {
        if self.devhost.is_some() {
            return self.run_dev(weights, attn, tokens, pos0);
        }
        let c = &weights.config;
        let m = tokens.len();
        if m == 0 {
            return Err("whole-prompt prefill of 0 tokens".into());
        }
        if pos0 + m > u32::MAX as usize {
            return Err(format!("positions {pos0} + {m} do not fit u32"));
        }
        if self.entries.len() != weights.layers.len() {
            return Err(format!(
                "driver built for {} layers, weights have {}",
                self.entries.len(),
                weights.layers.len()
            ));
        }
        let (h, qd, qkv_n, e, top_k, i, eps) = (
            c.hidden,
            c.q_dim(),
            c.qkv_n(),
            c.experts,
            c.top_k,
            c.moe_intermediate,
            c.rms_eps,
        );
        let g = m * top_k;
        let dev_attn = attn.has_dev();
        self.ensure(c, m, dev_attn)?;
        let mut st = PrefillStats {
            tokens: m,
            layers: weights.layers.len(),
            attn_dev: dev_attn,
            ..Default::default()
        };
        let (dev, b) = (&mut self.dev, self.bufs.as_mut().expect("sized"));
        let bm = b.bm;
        let t_run = Instant::now();
        let mut lap = Instant::now();
        // W-L1G-D profile: adds the time since the last lap to one HostProf field.
        macro_rules! lap {
            ($f:ident) => {{
                st.prof.$f += lap.elapsed().as_nanos() as u64;
                lap = Instant::now();
            }};
        }

        let t = Instant::now();
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&tok| weights.embed_row(tok))
            .collect();
        let mut logits = vec![0.0f32; m * e];
        lap!(embed_ns);
        st.host_ns += t.elapsed().as_nanos() as u64;

        // Writes a tile table into its region and returns (tiles_va, ntiles).
        let put_tiles = |tiles: &mut D::Buf,
                         max: usize,
                         region: Region,
                         w: &[u64],
                         s: &[u64],
                         counts: &[u32]|
         -> Result<(u64, usize), String> {
            let tab = build_tiles(bm, w, s, counts).map_err(|err| err.to_string())?;
            if tab.len() > max {
                return Err(format!(
                    "tile table of {} entries > region of {max}",
                    tab.len()
                ));
            }
            let off = region as usize * max * TILE_BYTES;
            let bytes = tiles.host_mut().ok_or("tile table is not a host buffer")?;
            for (j, tl) in tab.iter().enumerate() {
                bytes[off + j * TILE_BYTES..off + (j + 1) * TILE_BYTES]
                    .copy_from_slice(&tl.to_bytes());
            }
            Ok((tiles.va() + off as u64, tab.len()))
        };
        let dense = [m as u32];
        // W-L1G-C gate hooks: the recorder (a field disjoint from dev / bufs) and the mutant switch.
        let mutant = self.mutant;
        let mut tr = self.trace.as_mut();
        if let Some(t) = tr.as_deref_mut() {
            t.begin(m, weights.layers.len(), h, top_k);
        }

        for (l, layer) in weights.layers.iter().enumerate() {
            let ent = &self.entries[l];
            let lu = l as u32;
            // --- input RMSNorm straight into the quantiser's HOST input ---
            let t = Instant::now();
            {
                let xin = host_mut(&mut b.xin, "xin")?;
                xin[..m * h]
                    .par_chunks_mut(h)
                    .zip(x.par_chunks(h))
                    .for_each(|(o, xr)| rmsnorm(xr, &layer.input_norm, eps, o));
                lap!(norm_in_ns);
            }
            st.host_ns += t.elapsed().as_nanos() as u64;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.x_norm = host(&b.xin, "xin")?[..m * h].to_vec();
            }
            st.chip_ns += dev
                .quant(h, QuantIn::F32, m, b.xin.va(), b.xq.va(), b.xs.va())
                .map_err(|err| format!("layer {l} quant x: {err}"))?;
            st.quant += 1;

            // --- q|k|v: one GEMM for all m tokens ---
            lap!(dev_ns);
            let (tv, nt) = put_tiles(
                &mut b.tiles,
                b.max_tiles,
                Region::Qkv,
                &[ent.qkv.0],
                &[ent.qkv.1],
                &dense,
            )?;
            lap!(tiles_ns);
            let qkv_y = match (&b.qkv_dev, &b.qkv_host) {
                (Some(d), _) => d.va(),
                (None, Some(hb)) => hb.va(),
                (None, None) => unreachable!("ensure allocates one q|k|v buffer"),
            };
            let call = QgemmCall {
                n: qkv_n,
                k: h,
                layout: Layout::Plain,
                tiles_va: tv,
                ntiles: nt,
                rowidx_va: 0,
                xq_va: b.xq.va(),
                xs_va: b.xs.va(),
                y_va: qkv_y,
            };
            st.chip_ns += dev
                .qgemm(&call)
                .map_err(|err| format!("layer {l} q|k|v gemm x{m}: {err}"))?;
            st.qkv += 1;

            // --- attention: ONE call for all m tokens ---
            let att_va = if let (Some(qd_buf), Some(ad_buf)) = (&b.qkv_dev, &b.att_dev) {
                let ns = attn
                    .attend_dev(
                        lu,
                        pos0 as u32,
                        m as u32,
                        qd_buf.va(),
                        eps,
                        ATTN_OUT_SCALE,
                        ad_buf.va(),
                    )
                    .ok_or_else(|| {
                        format!("layer {l}: attention offered the device entry, then refused it")
                    })?
                    .map_err(|err| format!("layer {l} attention (device) x{m}: {err}"))?;
                st.attn_ns += ns;
                ad_buf.va()
            } else {
                let qkv = host(b.qkv_host.as_ref().expect("host q|k|v"), "q|k|v")?;
                let out = host_mut(&mut b.xin, "xin")?;
                let ns = attn
                    .attend(
                        lu,
                        pos0 as u32,
                        m as u32,
                        &qkv[..m * qkv_n],
                        eps,
                        ATTN_OUT_SCALE,
                        &mut out[..m * qd],
                    )
                    .map_err(|err| format!("layer {l} attention x{m}: {err}"))?;
                st.attn_ns += ns;
                b.xin.va()
            };
            st.attn += 1;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.qkv = match (&b.qkv_dev, &b.qkv_host) {
                    (Some(d), _) => read_dev(&mut *dev, d.va(), m * qkv_n)?,
                    (None, Some(hb)) => host(hb, "q|k|v")?[..m * qkv_n].to_vec(),
                    (None, None) => Vec::new(),
                };
                t.ops.attn = if b.att_dev.is_some() {
                    read_dev(&mut *dev, att_va, m * qd)?
                } else {
                    host(&b.xin, "xin")?[..m * qd].to_vec()
                };
            }
            st.chip_ns += dev
                .quant(qd, QuantIn::F32, m, att_va, b.xq.va(), b.xs.va())
                .map_err(|err| format!("layer {l} quant attn: {err}"))?;
            st.quant += 1;

            // --- o: one GEMM into HOST rows ---
            lap!(dev_ns);
            let (tv, nt) = put_tiles(
                &mut b.tiles,
                b.max_tiles,
                Region::O,
                &[ent.o.0],
                &[ent.o.1],
                &dense,
            )?;
            lap!(tiles_ns);
            let call = QgemmCall {
                n: h,
                k: qd,
                layout: Layout::Plain,
                tiles_va: tv,
                ntiles: nt,
                rowidx_va: 0,
                xq_va: b.xq.va(),
                xs_va: b.xs.va(),
                y_va: b.oproj.va(),
            };
            st.chip_ns += dev
                .qgemm(&call)
                .map_err(|err| format!("layer {l} o gemm x{m}: {err}"))?;
            st.o += 1;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.oproj = host(&b.oproj, "o projection")?[..m * h].to_vec();
            }

            // --- residual, post norm (into the quantiser input), routing, sort ---
            let t = Instant::now();
            let sort = {
                lap!(dev_ns);
                let oproj = host(&b.oproj, "o projection")?;
                x.par_chunks_mut(h)
                    .zip(oproj[..m * h].par_chunks(h))
                    .for_each(|(xr, pr)| {
                        for (xi, p) in xr.iter_mut().zip(pr) {
                            *xi += p;
                        }
                    });
                if let Some(t) = tr.as_deref_mut() {
                    t.x_mid[l].copy_from_slice(&x);
                }
                lap!(resid_ns);
                let post = host_mut(&mut b.xin, "xin")?;
                post[..m * h]
                    .par_chunks_mut(h)
                    .zip(x.par_chunks(h))
                    .for_each(|(o, xr)| rmsnorm(xr, &layer.post_norm, eps, o));
                lap!(norm_post_ns);
                router_logits(&layer.router, &post[..m * h], h, e, &mut logits);
                lap!(router_ns);
                if mutant == WpMutant::Routing {
                    rotate_logits(&mut logits, e);
                }
                if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                    t.ops.post = post[..m * h].to_vec();
                    t.ops.logits = logits.clone();
                }
                let plan = MoeBatchPlan::from_logits(&logits, m, e, top_k)
                    .map_err(|err| format!("layer {l} routing: {err}"))?;
                let sort = ExpertSort::from_plan(&plan);
                if let Some(t) = tr.as_deref_mut() {
                    t.experts[l].copy_from_slice(&plan.topk_experts);
                }
                as_u32_mut(
                    b.rowidx
                        .host_mut()
                        .ok_or("row index is not a host buffer")?,
                )?[..g]
                    .copy_from_slice(&sort.rows);
                (sort, plan.topk_weights)
            };
            let (sort, wts) = sort;
            lap!(plan_ns);
            st.host_ns += t.elapsed().as_nanos() as u64;
            st.chip_ns += dev
                .quant(h, QuantIn::F32, m, b.xin.va(), b.xq.va(), b.xs.va())
                .map_err(|err| format!("layer {l} quant post: {err}"))?;
            st.quant += 1;

            // --- experts: gate|up with the SwiGLU epilogue (rows gathered by token), act quantised, down ---
            lap!(dev_ns);
            let (tv, nt) = put_tiles(
                &mut b.tiles,
                b.max_tiles,
                Region::GateUp,
                &ent.gu_w,
                &ent.gu_s,
                &sort.counts,
            )?;
            if mutant == WpMutant::GateUpScales {
                // GEMM mutant: every expert's weights with the NEXT expert's scale blocks (same table size, same region).
                let mut s = ent.gu_s.clone();
                s.rotate_left(1);
                put_tiles(
                    &mut b.tiles,
                    b.max_tiles,
                    Region::GateUp,
                    &ent.gu_w,
                    &s,
                    &sort.counts,
                )?;
            }
            lap!(tiles_ns);
            let call = QgemmCall {
                n: 2 * i,
                k: h,
                layout: Layout::SwigluNat,
                tiles_va: tv,
                ntiles: nt,
                rowidx_va: b.rowidx.va(),
                xq_va: b.xq.va(),
                xs_va: b.xs.va(),
                y_va: b.act.va(),
            };
            st.chip_ns += dev
                .qgemm(&call)
                .map_err(|err| format!("layer {l} gate|up gemm x{g}: {err}"))?;
            st.gate_up += 1;
            st.chip_ns += dev
                .quant(i, QuantIn::Bf16, g, b.act.va(), b.aq.va(), b.a_s.va())
                .map_err(|err| format!("layer {l} quant act: {err}"))?;
            st.quant += 1;
            lap!(dev_ns);
            let (tv, nt) = put_tiles(
                &mut b.tiles,
                b.max_tiles,
                Region::Down,
                &ent.dn_w,
                &ent.dn_s,
                &sort.counts,
            )?;
            lap!(tiles_ns);
            let call = QgemmCall {
                n: h,
                k: i,
                layout: Layout::Plain,
                tiles_va: tv,
                ntiles: nt,
                rowidx_va: 0,
                xq_va: b.aq.va(),
                xs_va: b.a_s.va(),
                y_va: b.down.va(),
            };
            st.chip_ns += dev
                .qgemm(&call)
                .map_err(|err| format!("layer {l} down gemm x{g}: {err}"))?;
            st.down += 1;

            let t = Instant::now();
            lap!(dev_ns);
            combine_rows(
                &mut x,
                &host(&b.down, "down rows")?[..g * h],
                &wts,
                &sort.inv,
                h,
                top_k,
            );
            lap!(combine_ns);
            if let Some(t) = tr.as_deref_mut() {
                t.x_out[l].copy_from_slice(&x);
                if t.ops_layer == Some(l) {
                    let moe: Vec<f32> = x.iter().zip(&t.x_mid[l]).map(|(a, b)| a - b).collect();
                    t.ops.moe_out = moe;
                }
            }
            st.host_ns += t.elapsed().as_nanos() as u64;
        }
        let t = Instant::now();
        let mut hidden = vec![0.0f32; h];
        rmsnorm(&x[(m - 1) * h..], &weights.final_norm, eps, &mut hidden);
        lap!(final_ns);
        let _ = lap;
        st.prof.wall_ns = t_run.elapsed().as_nanos() as u64;
        st.host_ns += t.elapsed().as_nanos() as u64;
        self.stats = st;
        Ok(hidden)
    }
}

/// Router logits for every row of `post` (`m x h`): `logits[r][ex] = dot(router[ex], post[r])`,
/// each dot summed in index order (reimplements the engine's private `router_logits`).
pub fn router_logits(router: &[f32], post: &[f32], h: usize, e: usize, logits: &mut [f32]) {
    logits
        .par_chunks_mut(e)
        .zip(post.par_chunks(h))
        .for_each(|(lr, pr)| {
            for (ex, lv) in lr.iter_mut().enumerate() {
                *lv = router[ex * h..(ex + 1) * h]
                    .iter()
                    .zip(pr)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        });
}

/// `x[r] += sum_slot wts[r * top_k + slot] * down[inv[r * top_k + slot]]`, accumulated in slot
/// order from zero (reimplements the engine's private `combine_rows`).
pub fn combine_rows(
    x: &mut [f32],
    down_sorted: &[f32],
    wts: &[f32],
    inv: &[u32],
    h: usize,
    top_k: usize,
) {
    x.par_chunks_mut(h).enumerate().for_each(|(r, xr)| {
        let mut acc = vec![0.0f32; h];
        for slot in 0..top_k {
            let gi = r * top_k + slot;
            let row = inv[gi] as usize;
            for (a, d) in acc.iter_mut().zip(&down_sorted[row * h..(row + 1) * h]) {
                *a += wts[gi] * d;
            }
        }
        for (xi, a) in xr.iter_mut().zip(&acc) {
            *xi += a;
        }
    });
}

pub mod devhost;
/// Host mirror of the activation quantiser (W-L1G-C).
pub mod fake_quant;

/// Negative-control mutants of the whole-prompt path (W-L1G-C). Each must be caught by the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WpMutant {
    #[default]
    None,
    /// Every chosen expert id becomes its neighbour `(ex + 1) % e`, weights unchanged (the
    /// engine's `routing_mutant`, done by rotating each token's router logits by one).
    Routing,
    /// The gate|up GEMM reads every expert's weights with the next expert's scale blocks.
    GateUpScales,
}

/// One layer's intermediate values for every token (recorded at [`WpTrace::ops_layer`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WpOps {
    /// `m x hidden` input RMSNorm output (the q|k|v GEMM input before quantisation).
    pub x_norm: Vec<f32>,
    /// `m x qkv_n` fused projection rows.
    pub qkv: Vec<f32>,
    /// `m x q_dim` attention output (the o GEMM input before quantisation).
    pub attn: Vec<f32>,
    /// `m x hidden` o projection.
    pub oproj: Vec<f32>,
    /// `m x hidden` post RMSNorm output (the gate|up GEMM input before quantisation).
    pub post: Vec<f32>,
    /// `m x experts` router logits.
    pub logits: Vec<f32>,
    /// `m x hidden` weighted expert sum (what the experts add to the residual).
    pub moe_out: Vec<f32>,
}

/// Correctness-gate recorder (W-L1G-C): per layer, every token's residual after the attention
/// block (`x_mid`), after the experts (`x_out`) and the chosen experts (token-major, slot
/// order); plus every op of one layer (`ops_layer`). The same shape is filled by the OFF path
/// and the W8A8 control (`qwen3_moe_omega::w8a8_control`), so the three can be compared per
/// layer and per op.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WpTrace {
    pub m: usize,
    pub hidden: usize,
    pub top_k: usize,
    /// Per layer `m x hidden`.
    pub x_mid: Vec<Vec<f32>>,
    pub x_out: Vec<Vec<f32>>,
    /// Per layer `m x top_k` expert ids.
    pub experts: Vec<Vec<u32>>,
    pub ops_layer: Option<usize>,
    pub ops: WpOps,
}

impl WpTrace {
    /// A recorder of every layer's residuals and the ops of `ops_layer`.
    pub fn new(ops_layer: Option<usize>) -> Self {
        Self {
            ops_layer,
            ..Default::default()
        }
    }

    /// Sizes the per-layer stores for `m` tokens (called by the run being recorded).
    pub fn begin(&mut self, m: usize, layers: usize, hidden: usize, top_k: usize) {
        self.m = m;
        self.hidden = hidden;
        self.top_k = top_k;
        self.x_mid = vec![vec![0.0f32; m * hidden]; layers];
        self.x_out = vec![vec![0.0f32; m * hidden]; layers];
        self.experts = vec![vec![0u32; m * top_k]; layers];
        self.ops = WpOps::default();
    }

    pub fn layers(&self) -> usize {
        self.x_out.len()
    }
}

/// Rotates every token's router logits right by one, so the top-k ids all move to `(ex + 1) % e`
/// with the same softmax weights (the routing mutant on a whole-prompt batch).
pub fn rotate_logits(logits: &mut [f32], e: usize) {
    for row in logits.chunks_mut(e) {
        row.rotate_right(1);
    }
}

/// Reads `n` f32 of a device-private buffer through a temporary HOST buffer (gate recorder only).
fn read_dev<D: PrefillDevice>(dev: &mut D, va: u64, n: usize) -> Result<Vec<f32>, String> {
    let tmp = dev.alloc_host(n * 4)?;
    dev.copy_va(tmp.va(), va, n * 4)
        .map_err(|e| format!("recorder copy of {n} f32: {e}"))?;
    Ok(host(&tmp, "recorder scratch")?[..n].to_vec())
}

#[cfg(test)]
mod tests;
