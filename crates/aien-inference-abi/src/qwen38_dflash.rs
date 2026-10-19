//! DFlash 2 block drafter for Qwen3.8-27B: the host (CPU) reference.
//!
//! The architecture, every tensor and the exact order of the arithmetic are written down in
//! `docs/qwen38/DFLASH2.md`. This module is the reference the chip path is checked against: plain
//! f32 arithmetic on the host, the bf16 weights widened on the fly, every dot product summed in a
//! fixed order whatever the number of threads (eight interleaved partial sums, then added in
//! index order), so a run is bit for bit repeatable.
//!
//! One round (greedy):
//! 1. [`Drafter::add_context`]: the target's residual rows after the tap layers (5 of them,
//!    concatenated) for the newly committed positions go through `fc` and `hidden_norm`; every
//!    layer's keys and values of those rows enter the [`DraftCache`]. They stay.
//! 2. [`Drafter::block`]: the anchor word and `block - 1` mask words at positions `s ..` through
//!    the five layers (both-ways attention over the cached context and the block, inside the
//!    sliding window), final norm. The block's own keys and values are never kept.
//! 3. A word-scoring head (the target's lm_head, [`Head`]) scores rows `1 ..`, and
//!    [`Drafter::select`] walks the top-k candidates with the pairwise selector.
//!
//! Sizes come from [`DflashConfig`], so the tests run the same code on a toy model.
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use rayon::prelude::*;
use serde_json::Value;

use crate::qwen3_moe::bf16_to_f32;

/// The drafter's shape (`config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct DflashConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub rope_theta: f64,
    /// Sliding window: a query at `p` sees keys `k` with `|p - k| < window`.
    pub window: usize,
    /// Anchor plus guesses.
    pub block: usize,
    pub mask_token: u32,
    /// Target layers (from 0) whose output feeds the context.
    pub taps: Vec<usize>,
    pub conv_group: usize,
    pub sel_rank: usize,
    pub sel_top_k: usize,
    pub vocab: usize,
}

fn uget(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("dflash config: no integer {k}"))
}

impl DflashConfig {
    /// From the checkpoint's `config.json`. Refuses what this implementation does not do
    /// (causal drafters, other conv sizes, full-attention layers, biases).
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let d = v
            .get("dflash_config")
            .ok_or("dflash config: no dflash_config")?;
        if v.get("is_causal").and_then(Value::as_bool) != Some(false) {
            return Err("dflash config: only is_causal false is implemented".into());
        }
        if v.get("attention_bias").and_then(Value::as_bool) != Some(false) {
            return Err("dflash config: attention_bias must be false".into());
        }
        if uget(d, "conv_kernel_size")? != 2 {
            return Err("dflash config: conv_kernel_size must be 2".into());
        }
        let layers = uget(v, "num_hidden_layers")?;
        let types = v
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("dflash config: no layer_types")?;
        if types.len() != layers
            || types
                .iter()
                .any(|t| t.as_str() != Some("sliding_attention"))
        {
            return Err("dflash config: every layer must be sliding_attention".into());
        }
        let rope = v
            .get("rope_parameters")
            .ok_or("dflash config: no rope_parameters")?;
        if rope.get("rope_type").and_then(Value::as_str) != Some("default") {
            return Err("dflash config: rope_type must be default".into());
        }
        let taps = d
            .get("target_layer_ids")
            .and_then(Value::as_array)
            .ok_or("dflash config: no target_layer_ids")?
            .iter()
            .map(|x| {
                x.as_u64()
                    .map(|n| n as usize)
                    .ok_or("dflash config: bad tap id")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let c = Self {
            hidden: uget(v, "hidden_size")?,
            layers,
            heads: uget(v, "num_attention_heads")?,
            kv_heads: uget(v, "num_key_value_heads")?,
            head_dim: uget(v, "head_dim")?,
            inter: uget(v, "intermediate_size")?,
            eps: v
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .ok_or("dflash config: no rms_norm_eps")? as f32,
            rope_theta: rope
                .get("rope_theta")
                .and_then(Value::as_f64)
                .ok_or("dflash config: no rope_theta")?,
            window: uget(v, "sliding_window")?,
            block: uget(d, "block_size")?,
            mask_token: uget(d, "mask_token_id")? as u32,
            taps,
            conv_group: uget(d, "conv_group_size")?,
            sel_rank: uget(d, "selector_rank")?,
            sel_top_k: uget(d, "selector_top_k")?,
            vocab: uget(v, "vocab_size")?,
        };
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), String> {
        let ok = self.hidden > 0
            && self.layers > 0
            && self.kv_heads > 0
            && self.heads.is_multiple_of(self.kv_heads)
            && self.head_dim.is_multiple_of(2)
            && self.conv_group > 0
            && self.hidden.is_multiple_of(self.conv_group)
            && self.block >= 2
            && !self.taps.is_empty()
            && self.sel_top_k >= 1
            && self.sel_top_k <= self.vocab
            && (self.mask_token as usize) < self.vocab
            && self.window >= 1;
        if ok {
            Ok(())
        } else {
            Err(format!("dflash config: inconsistent shape {self:?}"))
        }
    }

    pub fn groups(&self) -> usize {
        self.hidden / self.conv_group
    }
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// Width of one context row before `fc`.
    pub fn tap_width(&self) -> usize {
        self.taps.len() * self.hidden
    }
}

/// A bf16 `[n][k]` matrix (row-major, as stored).
#[derive(Clone)]
pub struct Mat {
    pub n: usize,
    pub k: usize,
    pub w: Vec<u16>,
}

impl std::fmt::Debug for Mat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat[{}x{}]", self.n, self.k)
    }
}

/// `sum_i a[i] * b[i]` in the fixed order: eight interleaved partial sums (`i mod 8`), then
/// `((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for j in 0..8 {
            s[j] += x[j] * y[j];
        }
    }
    for (j, (x, y)) in ca.remainder().iter().zip(cb.remainder()).enumerate() {
        s[j] += x * y;
    }
    ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))
}

impl Mat {
    pub fn new(n: usize, k: usize, w: Vec<u16>) -> Result<Self, String> {
        if w.len() != n * k {
            return Err(format!("Mat {n}x{k}: {} values", w.len()));
        }
        Ok(Self { n, k, w })
    }

    /// `y[r][j] = W[j] . x[r]` for `m` rows of `x` (`[m][k]`), out `[m][n]`.
    pub fn matmul(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "matmul input");
        let (n, k) = (self.n, self.k);
        let mut y = vec![0f32; m * n];
        // Columns of y in parallel chunks; each weight row widened once for all m rows.
        const CH: usize = 64;
        let cols: Vec<(usize, Vec<f32>)> = (0..n.div_ceil(CH))
            .into_par_iter()
            .map(|c| {
                let j0 = c * CH;
                let j1 = (j0 + CH).min(n);
                let mut wr = vec![0f32; k];
                let mut out = vec![0f32; (j1 - j0) * m];
                for j in j0..j1 {
                    for (o, &b) in wr.iter_mut().zip(&self.w[j * k..(j + 1) * k]) {
                        *o = bf16_to_f32(b);
                    }
                    for r in 0..m {
                        out[(j - j0) * m + r] = dot(&wr, &x[r * k..(r + 1) * k]);
                    }
                }
                (j0, out)
            })
            .collect();
        for (j0, out) in cols {
            for (jj, chunk) in out.chunks_exact(m).enumerate() {
                for (r, &v) in chunk.iter().enumerate() {
                    y[r * n + j0 + jj] = v;
                }
            }
        }
        y
    }

    /// Row `j` widened.
    pub fn row(&self, j: usize) -> Vec<f32> {
        self.w[j * self.k..(j + 1) * self.k]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }
}

/// Qwen3 RMSNorm with a plain weight: `x * (1 / sqrt(mean(x^2) + eps)) * w`, the mean summed in
/// f32 with [`dot`].
pub fn rmsnorm_plain(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = dot(x, x) / x.len() as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = (v * scale) * g;
    }
}

/// The two-tap grouped dynamic causal conv over `rows` rows of `hidden` (`docs/qwen38/DFLASH2.md`,
/// "Conv"): `base` is `[2 taps][hidden]`, `dynk` is `[rows][2 taps][groups]` (one stage's half of
/// the kernel projection). Every product is rounded before its add, in the order
/// `(((0 + b0 v0) + d0 v0) + b1 v1) + d1 v1`, with `v[-1] = 0`. The Omega chip kernel
/// (`omega_gpu_dynconv2_f32`) does exactly this.
pub fn dynconv2(
    v: &[f32],
    dynk: &[f32],
    base: &[f32],
    rows: usize,
    hidden: usize,
    group: usize,
) -> Vec<f32> {
    let groups = hidden / group;
    assert_eq!(v.len(), rows * hidden);
    assert_eq!(dynk.len(), rows * 2 * groups);
    assert_eq!(base.len(), 2 * hidden);
    let mut out = vec![0f32; rows * hidden];
    for p in 0..rows {
        for c in 0..hidden {
            let g = c / group;
            let v0 = v[p * hidden + c];
            let v1 = if p == 0 { 0.0 } else { v[(p - 1) * hidden + c] };
            let d0 = dynk[p * 2 * groups + g];
            let d1 = dynk[p * 2 * groups + groups + g];
            let mut o = 0.0f32;
            o += base[c] * v0;
            o += d0 * v0;
            o += base[hidden + c] * v1;
            o += d1 * v1;
            out[p * hidden + c] = o;
        }
    }
    out
}

/// `cos(pos * f_i)`, `sin(pos * f_i)` for `i < head_dim / 2`, `f_i = theta^(-2i / head_dim)`,
/// computed in f64 and rounded once.
pub fn rope_table(pos: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut c = Vec::with_capacity(half);
    let mut s = Vec::with_capacity(half);
    for i in 0..half {
        let f = theta.powf(-(2.0 * i as f64) / head_dim as f64);
        let a = pos as f64 * f;
        c.push(a.cos() as f32);
        s.push(a.sin() as f32);
    }
    (c, s)
}

/// rotate_half rope on every head of `v` (`heads * head_dim`): pairs `(i, i + half)`,
/// `o0 = x0 c - x1 s`, `o1 = x1 c + x0 s`.
pub fn rope_apply(v: &mut [f32], head_dim: usize, c: &[f32], s: &[f32]) {
    let half = head_dim / 2;
    for hd in v.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (x0, x1) = (hd[i], hd[i + half]);
            hd[i] = x0 * c[i] - x1 * s[i];
            hd[i + half] = x1 * c[i] + x0 * s[i];
        }
    }
}

/// One drafter layer.
#[derive(Debug, Clone)]
pub struct DflashLayer {
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub o: Mat,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
    /// `[2 stages][2 taps][hidden]`.
    pub att_base: Vec<f32>,
    /// `[2 stages * 2 taps * groups][hidden]`.
    pub att_proj: Mat,
    pub mlp_base: Vec<f32>,
    pub mlp_proj: Mat,
}

/// The whole drafter on the host.
#[derive(Debug, Clone)]
pub struct Drafter {
    pub cfg: DflashConfig,
    pub fc: Mat,
    pub hidden_norm: Vec<f32>,
    pub norm: Vec<f32>,
    pub layers: Vec<DflashLayer>,
    /// `[rank][hidden]`.
    pub sel_proj: Mat,
    /// `[vocab][rank]` each.
    pub pred: Mat,
    pub succ: Mat,
}

/// The drafter's KV cache: context rows only, keys after `k_norm` and rope.
#[derive(Debug, Clone, Default)]
pub struct DraftCache {
    /// Per layer `[pos][kv_dim]`.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// Positions held (`0 .. len`).
    pub len: usize,
}

impl DraftCache {
    pub fn new(layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
            len: 0,
        }
    }
    pub fn clear(&mut self) {
        for x in self.k.iter_mut().chain(self.v.iter_mut()) {
            x.clear();
        }
        self.len = 0;
    }
}

/// A word-scoring head: logits for one normed draft row (the target's lm_head).
pub trait Head {
    fn logits(&mut self, h: &[f32]) -> Result<Vec<f32>, String>;
}

/// The lm_head on the host.
impl Head for Mat {
    fn logits(&mut self, h: &[f32]) -> Result<Vec<f32>, String> {
        if h.len() != self.k {
            return Err(format!("head: {} values, want {}", h.len(), self.k));
        }
        Ok(self.matmul(h, 1))
    }
}

/// The `k` largest logits: values and ids, descending value, ascending id on a tie.
pub fn top_k(logits: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| {
        let (x, y) = (logits[*a as usize], logits[*b as usize]);
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_by(cmp);
    (idx.iter().map(|&i| logits[i as usize]).collect(), idx)
}

/// Test-only faults the parity and acceptance checks must catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftMutant {
    #[default]
    None,
    /// The conv's two taps swapped (the previous row weighted by the current-row kernel).
    ConvTapSwap,
    /// The selector's pairwise term dropped (plain argmax of the top-k logits).
    NoPairwise,
    /// The context rows are not written to the cache (NaN keys: the block is poisoned).
    NoContext,
}

impl DraftMutant {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "none" => Self::None,
            "convswap" => Self::ConvTapSwap,
            "nopair" => Self::NoPairwise,
            "nocontext" => Self::NoContext,
            o => return Err(format!("draft mutant {o}: none|convswap|nopair|nocontext")),
        })
    }
}

/// Which drafter matrix a product uses (the chip backend keys its resident copies by it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatId {
    Fc,
    SelProj,
    Q(usize),
    K(usize),
    V(usize),
    O(usize),
    Gate(usize),
    Up(usize),
    Down(usize),
    AttProj(usize),
    MlpProj(usize),
}

/// Where the drafter's matrix products and convs run. Everything else (norms, rope, attention,
/// selector walk) is the same host code whatever the backend, so a backend is checked by
/// comparing its block rows and drafts with [`HostLinear`]'s.
pub trait Linear {
    /// `y[m][host.n] = host . x[m][host.k]` for the matrix `id` (`host` is its host copy).
    fn mm(&self, id: MatId, host: &Mat, x: &[f32], m: usize) -> Result<Vec<f32>, String>;
    /// [`dynconv2`].
    fn conv(
        &self,
        v: &[f32],
        dynk: &[f32],
        base: &[f32],
        rows: usize,
        hidden: usize,
        group: usize,
    ) -> Result<Vec<f32>, String>;
}

/// The host reference backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostLinear;

impl Linear for HostLinear {
    fn mm(&self, _id: MatId, host: &Mat, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        Ok(host.matmul(x, m))
    }
    fn conv(
        &self,
        v: &[f32],
        dynk: &[f32],
        base: &[f32],
        rows: usize,
        hidden: usize,
        group: usize,
    ) -> Result<Vec<f32>, String> {
        Ok(dynconv2(v, dynk, base, rows, hidden, group))
    }
}

impl Drafter {
    /// [`Self::add_context_on`] on the host.
    pub fn add_context(
        &self,
        cache: &mut DraftCache,
        taps: &[f32],
        mutant: DraftMutant,
    ) -> Result<(), String> {
        self.add_context_on(&HostLinear, cache, taps, mutant)
    }

    /// [`Self::block_on`] on the host.
    pub fn block(
        &self,
        cache: &DraftCache,
        x: &[f32],
        mutant: DraftMutant,
    ) -> Result<Vec<f32>, String> {
        self.block_on(&HostLinear, cache, x, mutant)
    }

    /// [`Self::select_on`] on the host.
    pub fn select(
        &self,
        h: &[f32],
        logits: &[Vec<f32>],
        anchor: u32,
        mutant: DraftMutant,
    ) -> Vec<u32> {
        self.select_on(&HostLinear, h, logits, anchor, mutant)
            .expect("the host backend does not fail")
    }

    /// [`Self::draft_on`] on the host.
    pub fn draft(
        &self,
        cache: &DraftCache,
        anchor: u32,
        anchor_embed: &[f32],
        mask_embed: &[f32],
        head: &mut dyn Head,
        mutant: DraftMutant,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        self.draft_on(
            &HostLinear,
            cache,
            anchor,
            anchor_embed,
            mask_embed,
            head,
            mutant,
        )
    }

    /// Context rows `taps` (`[m][tap_width]`, positions `cache.len ..`) into the cache.
    pub fn add_context_on(
        &self,
        lin: &dyn Linear,
        cache: &mut DraftCache,
        taps: &[f32],
        mutant: DraftMutant,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let tw = c.tap_width();
        if taps.is_empty() || !taps.len().is_multiple_of(tw) {
            return Err(format!("add_context: {} values, rows of {tw}", taps.len()));
        }
        let m = taps.len() / tw;
        if cache.k.len() != c.layers {
            *cache = DraftCache::new(c.layers);
        }
        if mutant == DraftMutant::NoContext {
            cache.len += m;
            for l in 0..c.layers {
                cache.k[l].resize(cache.len * c.kv_dim(), f32::NAN);
                cache.v[l].resize(cache.len * c.kv_dim(), f32::NAN);
            }
            return Ok(());
        }
        let f = lin.mm(MatId::Fc, &self.fc, taps, m)?;
        let mut ctx = vec![0f32; m * c.hidden];
        for (o, r) in ctx.chunks_exact_mut(c.hidden).zip(f.chunks_exact(c.hidden)) {
            rmsnorm_plain(r, &self.hidden_norm, c.eps, o);
        }
        let pos0 = cache.len;
        for (l, ly) in self.layers.iter().enumerate() {
            let mut k = lin.mm(MatId::K(l), &ly.k, &ctx, m)?;
            let v = lin.mm(MatId::V(l), &ly.v, &ctx, m)?;
            for (r, kr) in k.chunks_exact_mut(c.kv_dim()).enumerate() {
                self.key_norm_rope(ly, kr, pos0 + r);
            }
            cache.k[l].extend_from_slice(&k);
            cache.v[l].extend_from_slice(&v);
        }
        cache.len += m;
        Ok(())
    }

    fn key_norm_rope(&self, ly: &DflashLayer, kr: &mut [f32], pos: usize) {
        let c = &self.cfg;
        let mut t = vec![0f32; c.head_dim];
        for hd in kr.chunks_exact_mut(c.head_dim) {
            rmsnorm_plain(hd, &ly.k_norm, c.eps, &mut t);
            hd.copy_from_slice(&t);
        }
        let (co, si) = rope_table(pos, c.head_dim, c.rope_theta);
        rope_apply(kr, c.head_dim, &co, &si);
    }

    /// The block (`x`, `[block][hidden]`: the anchor's and the mask word's embedding rows) at
    /// positions `cache.len ..` through every layer and the final norm: `[block][hidden]`.
    /// The cache is read, never written.
    pub fn block_on(
        &self,
        lin: &dyn Linear,
        cache: &DraftCache,
        x: &[f32],
        mutant: DraftMutant,
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (h, b) = (c.hidden, c.block);
        if x.len() != b * h {
            return Err(format!("block: {} values, want {b} x {h}", x.len()));
        }
        let s = cache.len;
        let mut x = x.to_vec();
        let mut a = vec![0f32; b * h];
        for (l, ly) in self.layers.iter().enumerate() {
            for (o, r) in a.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
                rmsnorm_plain(r, &ly.in_norm, c.eps, o);
            }
            let (pre, fin) = self.conv_kernels(lin, MatId::AttProj(l), &ly.att_proj, &a)?;
            let a2 = self.conv(lin, &a, &pre, &ly.att_base[..2 * h], mutant)?;
            let att = self.attention(lin, ly, l, cache, &a2, s)?;
            let o = lin.mm(MatId::O(l), &ly.o, &att, b)?;
            let o = self.conv(lin, &o, &fin, &ly.att_base[2 * h..], mutant)?;
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }
            for (o, r) in a.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
                rmsnorm_plain(r, &ly.post_norm, c.eps, o);
            }
            let (pre, fin) = self.conv_kernels(lin, MatId::MlpProj(l), &ly.mlp_proj, &a)?;
            let a2 = self.conv(lin, &a, &pre, &ly.mlp_base[..2 * h], mutant)?;
            let g = lin.mm(MatId::Gate(l), &ly.gate, &a2, b)?;
            let u = lin.mm(MatId::Up(l), &ly.up, &a2, b)?;
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let d = lin.mm(MatId::Down(l), &ly.down, &act, b)?;
            let d = self.conv(lin, &d, &fin, &ly.mlp_base[2 * h..], mutant)?;
            for (xv, dv) in x.iter_mut().zip(&d) {
                *xv += dv;
            }
        }
        let mut out = vec![0f32; b * h];
        for (o, r) in out.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
            rmsnorm_plain(r, &self.norm, c.eps, o);
        }
        Ok(out)
    }

    /// The kernel projection of the normed rows, split into the prepare and the finish halves,
    /// each `[rows][2 taps][groups]`.
    fn conv_kernels(
        &self,
        lin: &dyn Linear,
        id: MatId,
        proj: &Mat,
        a: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let g2 = 2 * self.cfg.groups();
        let rows = a.len() / self.cfg.hidden;
        let d = lin.mm(id, proj, a, rows)?;
        let mut pre = Vec::with_capacity(rows * g2);
        let mut fin = Vec::with_capacity(rows * g2);
        for r in d.chunks_exact(2 * g2) {
            pre.extend_from_slice(&r[..g2]);
            fin.extend_from_slice(&r[g2..]);
        }
        Ok((pre, fin))
    }

    fn conv(
        &self,
        lin: &dyn Linear,
        v: &[f32],
        dynk: &[f32],
        base: &[f32],
        mutant: DraftMutant,
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let rows = v.len() / c.hidden;
        if mutant == DraftMutant::ConvTapSwap {
            let mut sw = base[c.hidden..].to_vec();
            sw.extend_from_slice(&base[..c.hidden]);
            return lin.conv(v, dynk, &sw, rows, c.hidden, c.conv_group);
        }
        lin.conv(v, dynk, base, rows, c.hidden, c.conv_group)
    }

    /// Both-ways windowed attention of the block rows over the cached context and the block.
    fn attention(
        &self,
        lin: &dyn Linear,
        ly: &DflashLayer,
        l: usize,
        cache: &DraftCache,
        a: &[f32],
        s: usize,
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (b, hd, kvd, qd) = (c.block, c.head_dim, c.kv_dim(), c.q_dim());
        let mut q = lin.mm(MatId::Q(l), &ly.q, a, b)?;
        let mut k = lin.mm(MatId::K(l), &ly.k, a, b)?;
        let v = lin.mm(MatId::V(l), &ly.v, a, b)?;
        let mut t = vec![0f32; hd];
        for (r, qr) in q.chunks_exact_mut(qd).enumerate() {
            for hh in qr.chunks_exact_mut(hd) {
                rmsnorm_plain(hh, &ly.q_norm, c.eps, &mut t);
                hh.copy_from_slice(&t);
            }
            let (co, si) = rope_table(s + r, hd, c.rope_theta);
            rope_apply(qr, hd, &co, &si);
        }
        for (r, kr) in k.chunks_exact_mut(kvd).enumerate() {
            self.key_norm_rope(ly, kr, s + r);
        }
        let ck = cache.k.get(l).map(Vec::as_slice).unwrap_or(&[]);
        let cv = cache.v.get(l).map(Vec::as_slice).unwrap_or(&[]);
        if ck.len() != s * kvd || cv.len() != s * kvd {
            return Err(format!(
                "attention layer {l}: cache holds {} values for {s} positions",
                ck.len()
            ));
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let group = c.heads / c.kv_heads;
        let mut out = vec![0f32; b * qd];
        out.par_chunks_exact_mut(qd)
            .enumerate()
            .for_each(|(r, orow)| {
                let qp = s + r;
                // Visible keys: context positions within the window, then the block keys in it.
                let lo = qp.saturating_sub(c.window - 1);
                for hq in 0..c.heads {
                    let kh = hq / group;
                    let qv = &q[r * qd + hq * hd..r * qd + (hq + 1) * hd];
                    let mut sc: Vec<(f32, &[f32])> = Vec::with_capacity(s + b);
                    for p in lo..s {
                        let o = p * kvd + kh * hd;
                        sc.push((dot(qv, &ck[o..o + hd]) * scale, &cv[o..o + hd]));
                    }
                    for j in 0..b {
                        if (s + j).abs_diff(qp) >= c.window {
                            continue;
                        }
                        let o = j * kvd + kh * hd;
                        sc.push((dot(qv, &k[o..o + hd]) * scale, &v[o..o + hd]));
                    }
                    let mx = sc.iter().fold(f32::NEG_INFINITY, |m, x| m.max(x.0));
                    let mut sum = 0f32;
                    let w: Vec<f32> = sc
                        .iter()
                        .map(|x| {
                            let e = (x.0 - mx).exp();
                            sum += e;
                            e
                        })
                        .collect();
                    let o = &mut orow[hq * hd..(hq + 1) * hd];
                    for (wi, (_, vv)) in w.iter().zip(&sc) {
                        let p = wi / sum;
                        for (oo, &x) in o.iter_mut().zip(vv.iter()) {
                            *oo += p * x;
                        }
                    }
                }
            });
        Ok(out)
    }

    /// The selector over the draft rows: `h` is `[rows][hidden]` (block rows 1 ..), `logits`
    /// their scores. Greedy path from `anchor`.
    pub fn select_on(
        &self,
        lin: &dyn Linear,
        h: &[f32],
        logits: &[Vec<f32>],
        anchor: u32,
        mutant: DraftMutant,
    ) -> Result<Vec<u32>, String> {
        let c = &self.cfg;
        let rows = logits.len();
        let p = lin.mm(MatId::SelProj, &self.sel_proj, h, rows)?;
        let mut prev = anchor;
        let mut path = Vec::with_capacity(rows);
        for r in 0..rows {
            let (u, cand) = top_k(&logits[r], c.sel_top_k);
            let pr = self.pred.row(prev as usize);
            let hp: Vec<f32> = pr
                .iter()
                .zip(&p[r * c.sel_rank..(r + 1) * c.sel_rank])
                .map(|(a, b)| a * b)
                .collect();
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (j, (&uj, &cj)) in u.iter().zip(&cand).enumerate() {
                let s = if mutant == DraftMutant::NoPairwise {
                    uj
                } else {
                    uj + dot(&hp, &self.succ.row(cj as usize))
                };
                if s > best_s {
                    best_s = s;
                    best = j;
                }
            }
            prev = cand[best];
            path.push(prev);
        }
        Ok(path)
    }

    /// One whole draft: block rows from the anchor's embedding row and the mask word's, scores
    /// through `head`, the selector path (`block - 1` words). Also returns the block rows.
    pub fn draft_on(
        &self,
        lin: &dyn Linear,
        cache: &DraftCache,
        anchor: u32,
        anchor_embed: &[f32],
        mask_embed: &[f32],
        head: &mut dyn Head,
        mutant: DraftMutant,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let c = &self.cfg;
        let mut x = anchor_embed.to_vec();
        for _ in 1..c.block {
            x.extend_from_slice(mask_embed);
        }
        let hb = self.block_on(lin, cache, &x, mutant)?;
        let rows = &hb[c.hidden..];
        let mut logits = Vec::with_capacity(c.block - 1);
        for r in rows.chunks_exact(c.hidden) {
            logits.push(head.logits(r)?);
        }
        Ok((self.select_on(lin, rows, &logits, anchor, mutant)?, hb))
    }

    /// Bytes of bf16 matrix weights held.
    pub fn weight_bytes(&self) -> usize {
        let mats = |l: &DflashLayer| {
            [
                &l.q,
                &l.k,
                &l.v,
                &l.o,
                &l.gate,
                &l.up,
                &l.down,
                &l.att_proj,
                &l.mlp_proj,
            ]
            .iter()
            .map(|m| m.w.len() * 2)
            .sum::<usize>()
        };
        self.layers.iter().map(mats).sum::<usize>()
            + (self.fc.w.len() + self.sel_proj.w.len() + self.pred.w.len() + self.succ.w.len()) * 2
    }

    /// Loads `config.json` and `model.safetensors` from `dir`.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cj = dir.join("config.json");
        let v: Value = serde_json::from_slice(
            &std::fs::read(&cj).map_err(|e| format!("{}: {e}", cj.display()))?,
        )
        .map_err(|e| format!("{}: {e}", cj.display()))?;
        let cfg = DflashConfig::from_json(&v)?;
        let mut st = SafeTensors::open(&dir.join("model.safetensors"))?;
        Self::load(cfg, &mut st)
    }

    /// Every tensor by name from `st`, shapes checked against `cfg`.
    pub fn load(cfg: DflashConfig, st: &mut SafeTensors) -> Result<Self, String> {
        let (h, kd, qd, it, g4) = (
            cfg.hidden,
            cfg.kv_dim(),
            cfg.q_dim(),
            cfg.inter,
            4 * cfg.groups(),
        );
        let fc = Mat::new(
            h,
            cfg.tap_width(),
            st.bf16("fc.weight", &[h, cfg.tap_width()])?,
        )?;
        let sel_proj = Mat::new(
            cfg.sel_rank,
            h,
            st.bf16(
                "candidate_selector.hidden_projection.weight",
                &[cfg.sel_rank, h],
            )?,
        )?;
        let pred = Mat::new(
            cfg.vocab,
            cfg.sel_rank,
            st.bf16(
                "candidate_selector.predecessor_codebook",
                &[cfg.vocab, cfg.sel_rank],
            )?,
        )?;
        let succ = Mat::new(
            cfg.vocab,
            cfg.sel_rank,
            st.bf16(
                "candidate_selector.successor_codebook",
                &[cfg.vocab, cfg.sel_rank],
            )?,
        )?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let mut m = |s: &str, n: usize, k: usize| -> Result<Mat, String> {
                Mat::new(n, k, st.bf16(&p(s), &[n, k])?)
            };
            let q = m("self_attn.q_proj.weight", qd, h)?;
            let k = m("self_attn.k_proj.weight", kd, h)?;
            let v = m("self_attn.v_proj.weight", kd, h)?;
            let o = m("self_attn.o_proj.weight", h, qd)?;
            let gate = m("mlp.gate_proj.weight", it, h)?;
            let up = m("mlp.up_proj.weight", it, h)?;
            let down = m("mlp.down_proj.weight", h, it)?;
            let att_proj = m("attention_conv.kernel_projection.weight", g4, h)?;
            let mlp_proj = m("mlp_conv.kernel_projection.weight", g4, h)?;
            let widen = |w: Vec<u16>| w.into_iter().map(bf16_to_f32).collect::<Vec<f32>>();
            layers.push(DflashLayer {
                in_norm: st.vec_f32(&p("input_layernorm.weight"), h)?,
                post_norm: st.vec_f32(&p("post_attention_layernorm.weight"), h)?,
                q,
                k,
                v,
                o,
                q_norm: st.vec_f32(&p("self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: st.vec_f32(&p("self_attn.k_norm.weight"), cfg.head_dim)?,
                gate,
                up,
                down,
                att_base: widen(st.bf16(&p("attention_conv.base_kernel"), &[2, 2, h])?),
                att_proj,
                mlp_base: widen(st.bf16(&p("mlp_conv.base_kernel"), &[2, 2, h])?),
                mlp_proj,
            });
        }
        Ok(Self {
            hidden_norm: st.vec_f32("hidden_norm.weight", h)?,
            norm: st.vec_f32("norm.weight", h)?,
            cfg,
            fc,
            layers,
            sel_proj,
            pred,
            succ,
        })
    }
}

/// A single-file safetensors reader: the JSON header, then each tensor's bytes read on demand
/// (dtype, shape and byte range checked).
pub struct SafeTensors {
    file: File,
    data_start: u64,
    file_len: u64,
    header: serde_json::Map<String, Value>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let e = |x: std::io::Error| format!("{}: {x}", path.display());
        let mut file = File::open(path).map_err(e)?;
        let file_len = file.metadata().map_err(e)?.len();
        let mut n8 = [0u8; 8];
        file.read_exact(&mut n8).map_err(e)?;
        let n = u64::from_le_bytes(n8);
        if n > 100 << 20 || 8 + n > file_len {
            return Err(format!("{}: header of {n} bytes", path.display()));
        }
        let mut hb = vec![0u8; n as usize];
        file.read_exact(&mut hb).map_err(e)?;
        let v: Value =
            serde_json::from_slice(&hb).map_err(|x| format!("{}: header: {x}", path.display()))?;
        let header = v
            .as_object()
            .cloned()
            .ok_or("safetensors: header is not an object")?;
        Ok(Self {
            file,
            data_start: 8 + n,
            file_len,
            header,
        })
    }

    /// Names in the header (`__metadata__` left out).
    pub fn names(&self) -> Vec<String> {
        self.header
            .keys()
            .filter(|k| *k != "__metadata__")
            .cloned()
            .collect()
    }

    /// A BF16 tensor of exactly `shape`, as raw bf16 words.
    pub fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
        let t = self
            .header
            .get(name)
            .ok_or_else(|| format!("safetensors: no tensor {name}"))?;
        if t.get("dtype").and_then(Value::as_str) != Some("BF16") {
            return Err(format!("{name}: dtype {:?}, want BF16", t.get("dtype")));
        }
        let got: Vec<usize> = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: no shape"))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(u64::MAX) as usize)
            .collect();
        if got != shape {
            return Err(format!("{name}: shape {got:?}, want {shape:?}"));
        }
        let off = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let (a, b) = (
            off[0].as_u64().unwrap_or(u64::MAX),
            off[1].as_u64().unwrap_or(0),
        );
        let count: usize = shape.iter().product();
        if b < a || b - a != 2 * count as u64 || self.data_start + b > self.file_len {
            return Err(format!("{name}: byte range {a}..{b} for {count} values"));
        }
        let mut raw = vec![0u8; 2 * count];
        self.file
            .seek(SeekFrom::Start(self.data_start + a))
            .and_then(|_| self.file.read_exact(&mut raw))
            .map_err(|x| format!("{name}: {x}"))?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    pub fn vec_f32(&mut self, name: &str, n: usize) -> Result<Vec<f32>, String> {
        Ok(self
            .bf16(name, &[n])?
            .into_iter()
            .map(bf16_to_f32)
            .collect())
    }
}

/// Writes a safetensors file of BF16 tensors (tests build toy checkpoints with it).
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, Vec<u16>)],
) -> Result<(), String> {
    let mut header = serde_json::Map::new();
    let mut off = 0u64;
    for (name, shape, w) in tensors {
        let n = 2 * w.len() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}),
        );
        off += n;
    }
    let hb = serde_json::to_vec(&Value::Object(header)).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(8 + hb.len() + off as usize);
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    for (_, _, w) in tensors {
        for x in w {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// The greedy acceptance of one draft against the target's own tokens: how many of `drafts`
/// equal `truth` from the start.
pub fn accepted(drafts: &[u32], truth: &[u32]) -> usize {
    drafts.iter().zip(truth).take_while(|(a, b)| a == b).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f2b(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    /// A deterministic pseudo-random stream (xorshift), values in [-a, a].
    struct Rng(u64);
    impl Rng {
        fn f(&mut self, a: f32) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
        }
        fn mat(&mut self, n: usize, k: usize, a: f32) -> Mat {
            Mat::new(n, k, (0..n * k).map(|_| f2b(self.f(a))).collect()).unwrap()
        }
        fn vecn(&mut self, n: usize, a: f32, c: f32) -> Vec<f32> {
            (0..n).map(|_| bf16_to_f32(f2b(c + self.f(a)))).collect()
        }
        fn vals(&mut self, n: usize, a: f32) -> Vec<f32> {
            (0..n).map(|_| self.f(a)).collect()
        }
    }

    fn toy_cfg() -> DflashConfig {
        DflashConfig {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            inter: 96,
            eps: 1e-6,
            rope_theta: 10_000.0,
            window: 2048,
            block: 4,
            mask_token: 3,
            taps: vec![0, 2],
            conv_group: 16,
            sel_rank: 8,
            sel_top_k: 4,
            vocab: 50,
        }
    }

    fn toy(cfg: DflashConfig, seed: u64) -> Drafter {
        let mut r = Rng(seed);
        let (h, g4) = (cfg.hidden, 4 * cfg.groups());
        let layers = (0..cfg.layers)
            .map(|_| DflashLayer {
                in_norm: r.vecn(h, 0.2, 1.0),
                post_norm: r.vecn(h, 0.2, 1.0),
                q: r.mat(cfg.q_dim(), h, 0.2),
                k: r.mat(cfg.kv_dim(), h, 0.2),
                v: r.mat(cfg.kv_dim(), h, 0.2),
                o: r.mat(h, cfg.q_dim(), 0.2),
                q_norm: r.vecn(cfg.head_dim, 0.2, 1.0),
                k_norm: r.vecn(cfg.head_dim, 0.2, 1.0),
                gate: r.mat(cfg.inter, h, 0.2),
                up: r.mat(cfg.inter, h, 0.2),
                down: r.mat(h, cfg.inter, 0.2),
                att_base: r.vecn(4 * h, 0.5, 0.5),
                att_proj: r.mat(g4, h, 0.1),
                mlp_base: r.vecn(4 * h, 0.5, 0.5),
                mlp_proj: r.mat(g4, h, 0.1),
            })
            .collect();
        Drafter {
            fc: r.mat(h, cfg.tap_width(), 0.2),
            hidden_norm: r.vecn(h, 0.2, 1.0),
            norm: r.vecn(h, 0.2, 1.0),
            layers,
            sel_proj: r.mat(cfg.sel_rank, h, 0.3),
            pred: r.mat(cfg.vocab, cfg.sel_rank, 1.0),
            succ: r.mat(cfg.vocab, cfg.sel_rank, 1.0),
            cfg,
        }
    }

    fn bits(a: &[f32]) -> Vec<u32> {
        a.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn dynconv2_hand_example() {
        // hidden 2, group 1 (two groups), rows 2.
        let v = [1.0, 2.0, 3.0, 4.0];
        // [rows][2 taps][2 groups]
        let d = [0.5, 0.25, 9.0, 9.0, 0.125, 1.0, 2.0, 4.0];
        let base = [1.0, 10.0, 100.0, 1000.0]; // tap 0 [1, 10], tap 1 [100, 1000]
        let o = dynconv2(&v, &d, &base, 2, 2, 1);
        // p0: c0 = 1*1 + .5*1 = 1.5; c1 = 10*2 + .25*2 = 20.5 (v[-1] = 0 silences tap 1)
        // p1: c0 = 1*3 + .125*3 + 100*1 + 2*1 = 105.375; c1 = 10*4 + 1*4 + 1000*2 + 4*2 = 2052
        assert_eq!(o, vec![1.5, 20.5, 105.375, 2052.0]);
    }

    #[test]
    fn dot_is_fixed_order_and_handles_tails() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        assert_eq!(dot(&a, &[1f32; 13]), 78.0);
    }

    #[test]
    fn top_k_descending_ascending_id_on_ties() {
        let (u, c) = top_k(&[0.0, 3.0, 1.0, 3.0, 2.0], 3);
        assert_eq!(c, vec![1, 3, 4]);
        assert_eq!(u, vec![3.0, 3.0, 2.0]);
    }

    #[test]
    fn accepted_counts_the_matching_prefix() {
        assert_eq!(accepted(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(accepted(&[9, 2], &[1, 2]), 0);
        assert_eq!(accepted(&[1, 2], &[1, 2, 3]), 2);
    }

    #[test]
    fn context_in_pieces_equals_context_at_once() {
        let d = toy(toy_cfg(), 7);
        let mut r = Rng(99);
        let tw = d.cfg.tap_width();
        let taps = r.vals(5 * tw, 1.0);
        let x = r.vals(d.cfg.block * d.cfg.hidden, 1.0);
        let mut a = DraftCache::new(d.cfg.layers);
        d.add_context(&mut a, &taps, DraftMutant::None).unwrap();
        let mut b = DraftCache::new(d.cfg.layers);
        d.add_context(&mut b, &taps[..2 * tw], DraftMutant::None)
            .unwrap();
        // A block in between must leave nothing behind.
        let _ = d.block(&b, &x, DraftMutant::None).unwrap();
        d.add_context(&mut b, &taps[2 * tw..], DraftMutant::None)
            .unwrap();
        assert_eq!((a.len, b.len), (5, 5));
        for l in 0..d.cfg.layers {
            assert_eq!(bits(&a.k[l]), bits(&b.k[l]));
            assert_eq!(bits(&a.v[l]), bits(&b.v[l]));
        }
        let ha = d.block(&a, &x, DraftMutant::None).unwrap();
        let hb = d.block(&b, &x, DraftMutant::None).unwrap();
        assert_eq!(bits(&ha), bits(&hb));
    }

    #[test]
    fn context_matters_and_the_no_context_mutant_is_seen() {
        let d = toy(toy_cfg(), 8);
        let mut r = Rng(5);
        let taps = r.vals(3 * d.cfg.tap_width(), 1.0);
        let x = r.vals(d.cfg.block * d.cfg.hidden, 1.0);
        let mut a = DraftCache::new(d.cfg.layers);
        d.add_context(&mut a, &taps, DraftMutant::None).unwrap();
        let mut taps2 = taps.clone();
        taps2[0] += 0.5;
        let mut b = DraftCache::new(d.cfg.layers);
        d.add_context(&mut b, &taps2, DraftMutant::None).unwrap();
        let ha = d.block(&a, &x, DraftMutant::None).unwrap();
        let hb = d.block(&b, &x, DraftMutant::None).unwrap();
        assert_ne!(bits(&ha), bits(&hb), "a context row must change the block");
        let mut m = DraftCache::new(d.cfg.layers);
        d.add_context(&mut m, &taps, DraftMutant::NoContext)
            .unwrap();
        let hm = d.block(&m, &x, DraftMutant::None).unwrap();
        assert_ne!(bits(&ha), bits(&hm));
    }

    #[test]
    fn window_hides_far_context() {
        let mut cfg = toy_cfg();
        cfg.window = 4;
        let d = toy(cfg, 9);
        let mut r = Rng(6);
        let tw = d.cfg.tap_width();
        let taps = r.vals(6 * tw, 1.0);
        let x = r.vals(d.cfg.block * d.cfg.hidden, 1.0);
        // Block at positions 6..=9: row 6 sees keys 3..=9, row 9 sees 6..=9 (and up to 12).
        // Position 2 is outside every row's window; position 3 is inside row 6's.
        let run = |row: usize, delta: f32| {
            let mut t = taps.clone();
            t[row * tw] += delta;
            let mut c = DraftCache::new(d.cfg.layers);
            d.add_context(&mut c, &t, DraftMutant::None).unwrap();
            d.block(&c, &x, DraftMutant::None).unwrap()
        };
        let base = run(0, 0.0);
        assert_eq!(bits(&base), bits(&run(2, 0.7)), "position 2 is outside");
        assert_ne!(bits(&base), bits(&run(3, 0.7)), "position 3 is inside");
    }

    #[test]
    fn block_attends_both_ways() {
        let d = toy(toy_cfg(), 10);
        let mut r = Rng(11);
        let taps = r.vals(2 * d.cfg.tap_width(), 1.0);
        let mut c = DraftCache::new(d.cfg.layers);
        d.add_context(&mut c, &taps, DraftMutant::None).unwrap();
        let h = d.cfg.hidden;
        let x = r.vals(d.cfg.block * h, 1.0);
        let mut x2 = x.clone();
        x2[3 * h] += 0.5;
        let a = d.block(&c, &x, DraftMutant::None).unwrap();
        let b = d.block(&c, &x2, DraftMutant::None).unwrap();
        // Row 1 must see row 3 (no causal mask).
        assert_ne!(bits(&a[h..2 * h]), bits(&b[h..2 * h]));
    }

    #[test]
    fn conv_tap_swap_mutant_changes_the_block() {
        let d = toy(toy_cfg(), 12);
        let mut r = Rng(13);
        let taps = r.vals(2 * d.cfg.tap_width(), 1.0);
        let mut c = DraftCache::new(d.cfg.layers);
        d.add_context(&mut c, &taps, DraftMutant::None).unwrap();
        let x = r.vals(d.cfg.block * d.cfg.hidden, 1.0);
        let a = d.block(&c, &x, DraftMutant::None).unwrap();
        let b = d.block(&c, &x, DraftMutant::ConvTapSwap).unwrap();
        assert_ne!(bits(&a), bits(&b));
    }

    #[test]
    fn selector_pairwise_term_can_overturn_the_logit_order() {
        let mut cfg = toy_cfg();
        cfg.hidden = 16;
        cfg.sel_rank = 2;
        cfg.sel_top_k = 2;
        cfg.vocab = 4;
        let mut d = toy(cfg, 14);
        // p = [h0, h1]; pred[0] = [1, 0]; succ[1] = [10, 0].
        let mut pw = vec![0u16; 2 * 16];
        pw[0] = f2b(1.0);
        pw[16 + 1] = f2b(1.0);
        d.sel_proj = Mat::new(2, 16, pw).unwrap();
        let m = |v: [f32; 8]| Mat::new(4, 2, v.iter().map(|&x| f2b(x)).collect()).unwrap();
        d.pred = m([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        d.succ = m([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut h = vec![0f32; 16];
        h[0] = 1.0;
        // Token 2 leads token 1 by 5 logits; the pairwise term gives token 1 +10 after token 0.
        let logits = vec![vec![0.0, 1.0, 6.0, -1.0]];
        assert_eq!(d.select(&h, &logits, 0, DraftMutant::None), vec![1]);
        assert_eq!(d.select(&h, &logits, 0, DraftMutant::NoPairwise), vec![2]);
        // After token 1 (pred[1] = 0) the pairwise term is gone: plain logit order.
        assert_eq!(d.select(&h, &logits, 1, DraftMutant::None), vec![2]);
    }

    #[test]
    fn matmul_rows_match_single_rows_bit_for_bit() {
        let mut r = Rng(15);
        let m = r.mat(130, 40, 0.5);
        let x = r.vals(3 * 40, 1.0);
        let y = m.matmul(&x, 3);
        for row in 0..3 {
            let y1 = m.matmul(&x[row * 40..(row + 1) * 40], 1);
            assert_eq!(bits(&y[row * 130..(row + 1) * 130]), bits(&y1));
        }
    }

    #[test]
    fn safetensors_roundtrip_and_shape_refusal() {
        let cfg = toy_cfg();
        let d = toy(cfg.clone(), 16);
        let dir = std::env::temp_dir().join(format!("dflash-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.safetensors");
        let b = |v: &[f32]| v.iter().map(|&x| f2b(x)).collect::<Vec<u16>>();
        let h = cfg.hidden;
        let g4 = 4 * cfg.groups();
        let mut t: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("fc.weight".into(), vec![h, cfg.tap_width()], d.fc.w.clone()),
            ("hidden_norm.weight".into(), vec![h], b(&d.hidden_norm)),
            ("norm.weight".into(), vec![h], b(&d.norm)),
            (
                "candidate_selector.hidden_projection.weight".into(),
                vec![cfg.sel_rank, h],
                d.sel_proj.w.clone(),
            ),
            (
                "candidate_selector.predecessor_codebook".into(),
                vec![cfg.vocab, cfg.sel_rank],
                d.pred.w.clone(),
            ),
            (
                "candidate_selector.successor_codebook".into(),
                vec![cfg.vocab, cfg.sel_rank],
                d.succ.w.clone(),
            ),
        ];
        for (i, l) in d.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{i}.{s}");
            t.push((p("input_layernorm.weight"), vec![h], b(&l.in_norm)));
            t.push((
                p("post_attention_layernorm.weight"),
                vec![h],
                b(&l.post_norm),
            ));
            t.push((
                p("self_attn.q_proj.weight"),
                vec![cfg.q_dim(), h],
                l.q.w.clone(),
            ));
            t.push((
                p("self_attn.k_proj.weight"),
                vec![cfg.kv_dim(), h],
                l.k.w.clone(),
            ));
            t.push((
                p("self_attn.v_proj.weight"),
                vec![cfg.kv_dim(), h],
                l.v.w.clone(),
            ));
            t.push((
                p("self_attn.o_proj.weight"),
                vec![h, cfg.q_dim()],
                l.o.w.clone(),
            ));
            t.push((
                p("self_attn.q_norm.weight"),
                vec![cfg.head_dim],
                b(&l.q_norm),
            ));
            t.push((
                p("self_attn.k_norm.weight"),
                vec![cfg.head_dim],
                b(&l.k_norm),
            ));
            t.push((
                p("mlp.gate_proj.weight"),
                vec![cfg.inter, h],
                l.gate.w.clone(),
            ));
            t.push((p("mlp.up_proj.weight"), vec![cfg.inter, h], l.up.w.clone()));
            t.push((
                p("mlp.down_proj.weight"),
                vec![h, cfg.inter],
                l.down.w.clone(),
            ));
            t.push((
                p("attention_conv.base_kernel"),
                vec![2, 2, h],
                b(&l.att_base),
            ));
            t.push((
                p("attention_conv.kernel_projection.weight"),
                vec![g4, h],
                l.att_proj.w.clone(),
            ));
            t.push((p("mlp_conv.base_kernel"), vec![2, 2, h], b(&l.mlp_base)));
            t.push((
                p("mlp_conv.kernel_projection.weight"),
                vec![g4, h],
                l.mlp_proj.w.clone(),
            ));
        }
        write_safetensors(&p, &t).unwrap();
        let mut st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.names().len(), 6 + 15 * cfg.layers);
        let e = Drafter::load(cfg.clone(), &mut st).unwrap();
        let mut r = Rng(17);
        let taps = r.vals(2 * cfg.tap_width(), 1.0);
        let x = r.vals(cfg.block * h, 1.0);
        let (mut ca, mut cb) = (DraftCache::new(cfg.layers), DraftCache::new(cfg.layers));
        d.add_context(&mut ca, &taps, DraftMutant::None).unwrap();
        e.add_context(&mut cb, &taps, DraftMutant::None).unwrap();
        assert_eq!(
            bits(&d.block(&ca, &x, DraftMutant::None).unwrap()),
            bits(&e.block(&cb, &x, DraftMutant::None).unwrap())
        );
        assert!(st.bf16("fc.weight", &[h, 7]).is_err());
        assert!(st.bf16("nope", &[1]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_reads_the_shipped_one_and_refuses_causal() {
        let shipped = include_str!("../../../evidence/qwen38-dflash-weights/drafter-config.json");
        let v: Value = serde_json::from_str(shipped).unwrap();
        let c = DflashConfig::from_json(&v).unwrap();
        assert_eq!(c.taps, vec![5, 19, 33, 47, 61]);
        assert_eq!(
            (c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim),
            (5120, 5, 32, 8, 128)
        );
        assert_eq!(
            (c.block, c.mask_token, c.window, c.groups()),
            (8, 248070, 2048, 320)
        );
        let mut bad = v.clone();
        bad["is_causal"] = Value::Bool(true);
        assert!(DflashConfig::from_json(&bad).is_err());
    }

    /// The shipped checkpoint, when `AIEN_DFLASH_DIR` names it: every tensor loads with its
    /// shape, and a block from a made-up context is finite.
    #[test]
    #[ignore]
    fn real_checkpoint_loads_and_drafts() {
        let Ok(dir) = std::env::var("AIEN_DFLASH_DIR") else {
            return;
        };
        let d = Drafter::open(Path::new(&dir)).unwrap();
        assert_eq!(d.weight_bytes(), 3_848_273_920);
        let mut r = Rng(18);
        let taps = r.vals(3 * d.cfg.tap_width(), 1.0);
        let mut c = DraftCache::new(d.cfg.layers);
        d.add_context(&mut c, &taps, DraftMutant::None).unwrap();
        let x = r.vals(d.cfg.block * d.cfg.hidden, 0.05);
        let hb = d.block(&c, &x, DraftMutant::None).unwrap();
        assert!(hb.iter().all(|v| v.is_finite()));
    }
}
