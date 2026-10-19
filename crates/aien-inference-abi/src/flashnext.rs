//! Qwen3.8-Flash-Next (`qwen4_exp`, 125B total, 6B active per token) host reference.
//!
//! Plan and model facts: `~/handoffs/2026-10-19-flashnext-on-omega.md`. Host only: no launch, no chip. This is the
//! golden model every Omega kernel of the port is checked against, so it is written for clarity and f32 accuracy,
//! not speed. Ground truth for the maths is the model's own published code (transformers `modular_qwen4_exp.py`,
//! read as the specification; nothing is copied) and its `config.json` (`text_config`).
//!
//! The residual stream is four lanes of `hidden` (10240 values for hidden 2560): the embedding is repeated four
//! times. Every sub-block (attention or DeltaNet, then the MoE) goes through a [`GatedResidual`]:
//! `normed = rmsnorm_zc` over each lane (weight `[4 * hidden]`), `mix = sigmoid(up(silu(down(normed) / 4)))`,
//! block input = mean over the lanes of `mix * normed`, injection `2 * sigmoid(inject(normed) / 4)` (one per lane),
//! and the lanes become `lanes + block_out * injection[lane]`. The final mixer (no injection) gives the vector that
//! goes straight to the lm_head; there is no final norm.
//!
//! - DeltaNet layers: the Qwen3.8-27B recurrence (see `qwen38.rs`) at hidden 2560, with the gated norm's gate
//!   activation from `output_gate_type` (`sigmoid` for this checkpoint, `silu` in the 27B).
//! - Indexed attention layers (every 4th): the 27B gated attention (24 q heads, here 2 kv heads, head 256, partial
//!   RoPE 64) restricted to the tokens the QSA indexer selects: 4 indexer query heads and 1 key head of 128 from
//!   `index_qk_proj`; q normed (zero-centred) then RoPE; raw keys cached unnormed; per query, the visible tokens are
//!   cut into complete blocks of `compress_ratio` (4), each block's keys averaged, normed (`k_layernorm`) and
//!   turned at the block's first position; block score = sum over heads of relu(q . k) / sqrt(128); the top
//!   `budget / ratio` (512) blocks plus the incomplete tail are attended, everything else is masked. Below
//!   `budget + ratio - 1` visible tokens every token is selected.
//! - MoE (every layer): softmax over 512 router logits (f32), top 10, renormalised (`norm_topk_prob`, default on);
//!   routed out = sum w_i down_i(silu(gate_i x) * up_i x); plus the shared expert times sigmoid(shared_gate . x).
//! - PLE (layer index `ple_layer_ids - 1`, one-indexed in the config): hashed bigram and trigram ids of the token
//!   history (EOS resets the context), 16 table rows of 160 values (FP8 with one bf16 scale), key/value
//!   projections, a signed-sqrt gate per lane against the normed lanes, and a dilated depthwise conv (kernel 4,
//!   dilation 3) of the normed gated value; the result is added to the lanes before the layer's attention residual.
//!
//! Expert weights may be stored bf16 (sliced from the fused bf16 tensors, alias `experts.E.gate_up_proj` /
//! `experts.E.down_proj`), FP8 with 128x128 block scales (Qwen's FP8 checkpoint), or NVFP4 (NVIDIA's checkpoint:
//! e2m1 nibbles, e4m3 scale per 16, f32 tensor scale). [`ExpertFormat::Mxfp4`] converts every expert on load to the
//! Omega W4 tile format (e2m1 + E8M0 per 32, `omega docs/w4-tile-format.md`), so the reference can mirror the chip.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::qwen38::{read_bf16_words, read_matrix, read_vec, Matrix};
use crate::qwen3_moe::{bf16_to_f32, fp8_e4m3fn_to_f32};
use crate::qwen3_moe_omega::{ShardSource, TensorSource};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    Sigmoid,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashNextConfig {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub full_attention: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub gdn_gate: GateAct,
    pub eps: f32,
    pub eos: u32,
    pub hc: usize,
    pub hc_rank: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub norm_topk: bool,
    pub idx_heads: usize,
    pub idx_head_dim: usize,
    pub idx_budget: usize,
    pub idx_ratio: usize,
    /// Zero-based layer indices that carry a PLE block.
    pub ple_layers: Vec<usize>,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub ngram_base: u64,
    pub ngram_divisor: u64,
    pub ple_conv_kernel: usize,
    pub seed: u64,
}

fn tc_usize(v: &serde_json::Value, k: &str) -> Result<usize, String> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("config: missing integer {k}"))
}

impl FlashNextConfig {
    pub fn from_checkpoint(dir: &Path) -> Result<Self, String> {
        let p = dir.join("config.json");
        Self::from_json(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let root: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("config: {e}"))?;
        if root.get("tie_word_embeddings").and_then(|v| v.as_bool()) == Some(true) {
            return Err("config: tied embeddings are not this model".into());
        }
        let t = root.get("text_config").unwrap_or(&root);
        let mt = t.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        if mt != "qwen4_exp_text" && mt != "qwen4_exp" {
            return Err(format!(
                "config: model_type {mt:?}, expected qwen4_exp_text"
            ));
        }
        let layer_types: Vec<String> = t
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or("config: missing layer_types")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let layers = tc_usize(t, "num_hidden_layers")?;
        if layer_types.len() != layers {
            return Err("config: layer_types length differs from num_hidden_layers".into());
        }
        let mut full_attention = Vec::with_capacity(layers);
        for lt in &layer_types {
            full_attention.push(match lt.as_str() {
                "full_attention" | "indexed_attention" => true,
                "linear_attention" => false,
                o => return Err(format!("config: layer type {o:?}")),
            });
        }
        let rope = t.get("rope_parameters");
        let partial = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| t.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| t.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .ok_or("config: missing rope_theta")?;
        let head_dim = tc_usize(t, "head_dim")?;
        let eos = match t.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => a.first().and_then(|v| v.as_u64()),
            Some(v) => v.as_u64(),
            None => None,
        }
        .ok_or("config: missing eos_token_id")? as u32;
        let gate = t
            .get("output_gate_type")
            .and_then(|v| v.as_str())
            .or_else(|| t.get("hidden_act").and_then(|v| v.as_str()))
            .unwrap_or("silu");
        let gdn_gate = match gate {
            "sigmoid" => GateAct::Sigmoid,
            "silu" | "swish" => GateAct::Silu,
            o => return Err(format!("config: gate activation {o:?}")),
        };
        if t.get("hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            != "silu"
        {
            return Err("config: expert activation is not silu".into());
        }
        let ple_layers = t
            .get("ple_layer_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|x| x as usize - 1)
                    .collect()
            })
            .unwrap_or_default();
        let c = Self {
            hidden: tc_usize(t, "hidden_size")?,
            layers,
            vocab: tc_usize(t, "vocab_size")?,
            full_attention,
            heads: tc_usize(t, "num_attention_heads")?,
            kv_heads: tc_usize(t, "num_key_value_heads")?,
            head_dim,
            rotary_dim: (head_dim as f64 * partial) as usize,
            rope_theta,
            lin_k_heads: tc_usize(t, "linear_num_key_heads")?,
            lin_v_heads: tc_usize(t, "linear_num_value_heads")?,
            lin_k_dim: tc_usize(t, "linear_key_head_dim")?,
            lin_v_dim: tc_usize(t, "linear_value_head_dim")?,
            conv_kernel: tc_usize(t, "linear_conv_kernel_dim")?,
            gdn_gate,
            eps: t
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            eos,
            hc: tc_usize(t, "hc_count")?,
            hc_rank: tc_usize(t, "hc_lowrank")?,
            experts: tc_usize(t, "num_experts")?,
            top_k: tc_usize(t, "num_experts_per_tok")?,
            moe_inter: tc_usize(t, "moe_intermediate_size")?,
            shared_inter: tc_usize(t, "shared_expert_intermediate_size")?,
            norm_topk: t
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            idx_heads: tc_usize(t, "indexer_n_heads")?,
            idx_head_dim: tc_usize(t, "indexer_head_dim")?,
            idx_budget: tc_usize(t, "indexer_budget")?,
            idx_ratio: tc_usize(t, "indexer_compress_ratio")?,
            ple_layers,
            ngram: tc_usize(t, "ngram_size")?,
            heads_per_ngram: tc_usize(t, "heads_per_ngram")?,
            ngram_base: tc_usize(t, "ngram_vocab_size_base")? as u64,
            ngram_divisor: tc_usize(t, "make_ngram_vocab_size_divisible_by")? as u64,
            ple_conv_kernel: tc_usize(t, "ple_conv_kernel_size")?,
            seed: t.get("seed").and_then(|v| v.as_u64()).unwrap_or(1234),
        };
        if tc_usize(t, "indexer_kv_heads")? != 1 {
            return Err("config: indexer_kv_heads other than 1".into());
        }
        if t.get("ple_embed_dim").and_then(|v| v.as_u64()) != Some(c.hidden as u64)
            && !c.ple_layers.is_empty()
        {
            return Err("config: ple_embed_dim differs from hidden_size".into());
        }
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        let ok = self.heads.is_multiple_of(self.kv_heads)
            && self.lin_v_heads.is_multiple_of(self.lin_k_heads)
            && self.rotary_dim.is_multiple_of(2)
            && self.rotary_dim <= self.idx_head_dim
            && self.top_k <= self.experts
            && self.idx_budget.is_multiple_of(self.idx_ratio)
            && self.ngram >= 2
            && self.hidden.is_multiple_of(self.ngram_heads())
            && self.ple_layers.iter().all(|&l| l < self.layers);
        if ok {
            Ok(())
        } else {
            Err(format!("config: inconsistent shape {self:?}"))
        }
    }

    pub fn lanes(&self) -> usize {
        self.hc * self.hidden
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }
    pub fn lin_value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub fn ngram_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    pub fn ngram_row(&self) -> usize {
        self.hidden / self.ngram_heads()
    }
    /// The PLE short conv's state length: (kernel - 1) * dilation, dilation = ngram size.
    pub fn ple_state_len(&self) -> usize {
        (self.ple_conv_kernel - 1) * self.ngram
    }
    /// The first `n` layers only (tests and partial runs on fetched tensors).
    pub fn prefix(&self, n: usize) -> Option<Self> {
        if n == 0 || n > self.layers {
            return None;
        }
        let mut c = self.clone();
        c.layers = n;
        c.full_attention.truncate(n);
        c.ple_layers.retain(|&l| l < n);
        Some(c)
    }
}

// ---------------------------------------------------------------------------
// The n-gram hash (PLE ids): splitmix64 multipliers and prime head vocabularies.
// ---------------------------------------------------------------------------

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn splitmix64(v: u64) -> u64 {
    let mut v = v.wrapping_add(GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// The model's per-layer hash multipliers (odd, below `i64::MAX / vocab`).
pub fn layer_multipliers(vocab: u64, ngram: usize, ple_index: u64, seed: u64) -> Vec<i64> {
    let max_long = i64::MAX as u64;
    let mult_max = max_long / vocab.max(1);
    let half = (mult_max / 2).max(1);
    let base = seed.wrapping_add(10007u64.wrapping_mul(ple_index));
    (0..ngram)
        .map(|i| {
            let v = base.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1));
            (2 * (splitmix64(v) % half) + 1) as i64
        })
        .collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Head vocabularies (the k-th prime after base - 1, k = global head + 1) and offsets.
pub fn ngram_heads(c: &FlashNextConfig, ple_index: usize) -> (Vec<u64>, Vec<u64>) {
    let nh = c.ngram_heads();
    let (mut sizes, mut offs, mut total) = (Vec::new(), Vec::new(), 0u64);
    let mut p = c.ngram_base - 1;
    let first = ple_index * nh;
    for k in 0..first + nh {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
        if k >= first {
            sizes.push(p);
            offs.push(total);
            total += p;
        }
    }
    (sizes, offs)
}

/// Table rows (16 per token, global row ids) for `tokens` after `history` (the previous `ngram - 1` ids, EOS
/// padded at a sequence start). Ports `_shift_right_ignore_eos` + the xor hash: a shifted id that crosses an EOS
/// (or the sequence start) reads as EOS.
pub fn ngram_ids(
    c: &FlashNextConfig,
    mult: &[i64],
    sizes: &[u64],
    offs: &[u64],
    history: &[u32],
    tokens: &[u32],
) -> Vec<u64> {
    let ctx = c.ngram - 1;
    assert_eq!(history.len(), ctx);
    let all: Vec<u32> = history.iter().chain(tokens).copied().collect();
    let eos = c.eos;
    // segment start of every position: the position after the latest EOS strictly before it
    let mut seg = vec![0usize; all.len()];
    let mut last_eos: isize = -1;
    for (p, &tok) in all.iter().enumerate() {
        seg[p] = (last_eos + 1) as usize;
        if tok == eos {
            last_eos = p as isize;
        }
    }
    let shifted = |p: usize, s: usize| -> i64 {
        if s == 0 {
            return all[p] as i64;
        }
        if p >= s && p - seg[p] >= s {
            all[p - s] as i64
        } else {
            eos as i64
        }
    };
    let mut out = Vec::with_capacity(tokens.len() * c.ngram_heads());
    for p in ctx..all.len() {
        for n in 2..=c.ngram {
            let mut mixed = shifted(p, 0).wrapping_mul(mult[0]);
            for s in 1..n {
                mixed ^= shifted(p, s).wrapping_mul(mult[s]);
            }
            let h0 = (n - 2) * c.heads_per_ngram;
            for h in h0..h0 + c.heads_per_ngram {
                // torch.remainder: non-negative for a positive divisor
                let r = (mixed as i128).rem_euclid(sizes[h] as i128) as u64;
                out.push(r + offs[h]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Projections: dense (qwen38 Matrix), NVFP4, MXFP4 (the Omega W4 tile format).
// ---------------------------------------------------------------------------

pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Weight `c` of a packed row: low nibble of byte c/2 for even c (Omega's and modelopt's order).
#[inline]
pub fn nibble(row: &[u8], c: usize) -> u8 {
    (row[c >> 1] >> ((c & 1) * 4)) & 15
}

#[derive(Debug)]
pub enum Proj {
    Dense(Matrix),
    /// NVFP4: `packed [n][k/2]`, `scales [n][k/16]` e4m3, `global` = weight_scale_2.
    Nvfp4 {
        n: usize,
        k: usize,
        packed: Vec<u8>,
        scales: Vec<u8>,
        global: f32,
    },
    /// MXFP4 (Omega W4): `nib [n][k/2]`, `e8 [n][k/32]`, value = e2m1 * 2^(E - 127).
    Mxfp4 {
        n: usize,
        k: usize,
        nib: Vec<u8>,
        e8: Vec<u8>,
    },
}

impl Proj {
    pub fn n(&self) -> usize {
        match self {
            Self::Dense(m) => m.n(),
            Self::Nvfp4 { n, .. } | Self::Mxfp4 { n, .. } => *n,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            Self::Dense(m) => m.k(),
            Self::Nvfp4 { k, .. } | Self::Mxfp4 { k, .. } => *k,
        }
    }
    /// Row `r` as f32.
    pub fn row(&self, r: usize, dst: &mut [f32]) {
        match self {
            Self::Dense(m) => {
                // one-row matmul against unit vectors would be wasteful; widen through a tiny matmul of the row
                let k = m.k();
                dense_row(m, r, &mut dst[..k]);
            }
            Self::Nvfp4 {
                k,
                packed,
                scales,
                global,
                ..
            } => {
                let row = &packed[r * k / 2..(r + 1) * k / 2];
                let sc = &scales[r * k / 16..(r + 1) * k / 16];
                for c in 0..*k {
                    dst[c] = E2M1[nibble(row, c) as usize] * fp8_e4m3fn_to_f32(sc[c / 16]) * global;
                }
            }
            Self::Mxfp4 { k, nib, e8, .. } => {
                let row = &nib[r * k / 2..(r + 1) * k / 2];
                let sc = &e8[r * k / 32..(r + 1) * k / 32];
                for c in 0..*k {
                    dst[c] = E2M1[nibble(row, c) as usize] * pow2(sc[c / 32] as i32 - 127);
                }
            }
        }
    }
    /// `y[t][n] = W x[t]`.
    pub fn matmul(&self, x: &[f32], t: usize) -> Vec<f32> {
        if let Self::Dense(m) = self {
            return m.matmul(x, t);
        }
        let (n, k) = (self.n(), self.k());
        assert_eq!(x.len(), t * k);
        let mut by_row = vec![0f32; n * t];
        by_row.par_chunks_mut(t).enumerate().for_each_init(
            || vec![0f32; k],
            |buf, (r, out)| {
                self.row(r, buf);
                for (tt, o) in out.iter_mut().enumerate() {
                    *o = dot(buf, &x[tt * k..(tt + 1) * k]);
                }
            },
        );
        let mut y = vec![0f32; t * n];
        for r in 0..n {
            for tt in 0..t {
                y[tt * n + r] = by_row[r * t + tt];
            }
        }
        y
    }
    /// The whole matrix as f32 `[n][k]`.
    pub fn dense(&self) -> Vec<f32> {
        let (n, k) = (self.n(), self.k());
        let mut out = vec![0f32; n * k];
        out.par_chunks_mut(k)
            .enumerate()
            .for_each(|(r, o)| self.row(r, o));
        out
    }
    pub fn stored_bytes(&self) -> usize {
        match self {
            Self::Dense(Matrix::Fp8 { w, scales, .. }) => w.len() + 2 * scales.len(),
            Self::Dense(Matrix::Bf16 { w, .. }) => 2 * w.len(),
            Self::Nvfp4 { packed, scales, .. } => packed.len() + scales.len() + 4,
            Self::Mxfp4 { nib, e8, .. } => nib.len() + e8.len(),
        }
    }
}

fn dense_row(m: &Matrix, r: usize, dst: &mut [f32]) {
    match m {
        Matrix::Bf16 { k, w, .. } => {
            for (o, &b) in dst.iter_mut().zip(&w[r * k..(r + 1) * k]) {
                *o = bf16_to_f32(b);
            }
        }
        Matrix::Fp8 { k, w, scales, .. } => {
            let kb = k / 128;
            for c in 0..*k {
                dst[c] =
                    fp8_e4m3fn_to_f32(w[r * k + c]) * bf16_to_f32(scales[(r / 128) * kb + c / 128]);
            }
        }
    }
}

fn pow2(e: i32) -> f32 {
    (e as f64).exp2() as f32
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

/// e2m1 magnitude code (0..7) nearest to `a >= 0`, ties to the even code, saturating at 6 (Omega's
/// `omega_w4_e2m1_rne`).
pub fn e2m1_rne(a: f64) -> u8 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if a >= 6.0 {
        return 7;
    }
    let mut best = 0u8;
    for c in 1..8u8 {
        let (d, db) = ((a - MAG[c as usize]).abs(), (a - MAG[best as usize]).abs());
        if d < db || (d == db && c % 2 == 0) {
            best = c;
        }
    }
    best
}

/// Quantiser statistics of one conversion.
#[derive(Debug, Default, Clone, Copy)]
pub struct QuantStats {
    pub weights: u64,
    pub groups: u64,
    pub zero_groups: u64,
    pub saturated: u64,
    pub sq_err: f64,
    pub sq_ref: f64,
}

/// Omega's MXFP4 rule (`omega_gpu_w4_from_fp8`, the f7 rule) over f32 values `[n][k]`: per 32 weights,
/// X = 2^(floor(log2 max|v|) - 2), q = e2m1(|v| / X) round to nearest even, saturating; E = log2 X + 127;
/// an all-zero group gets E 127 and zero nibbles.
pub fn mxfp4_from_f32(n: usize, k: usize, v: &[f32], st: &mut QuantStats) -> Proj {
    assert!(k.is_multiple_of(32) && v.len() == n * k);
    let mut nib = vec![0u8; n * k / 2];
    let mut e8 = vec![0u8; n * k / 32];
    for r in 0..n {
        for g in 0..k / 32 {
            let grp = &v[r * k + 32 * g..r * k + 32 * g + 32];
            let amax = grp.iter().fold(0f32, |m, x| m.max(x.abs())) as f64;
            st.groups += 1;
            st.weights += 32;
            if amax == 0.0 {
                e8[r * k / 32 + g] = 127;
                st.zero_groups += 1;
                continue;
            }
            let e = amax.log2().floor() as i32 - 2;
            let x = (e as f64).exp2();
            e8[r * k / 32 + g] = (e + 127).clamp(0, 254) as u8;
            for (j, &w) in grp.iter().enumerate() {
                let q = w.abs() as f64 / x;
                if q > 6.0 {
                    st.saturated += 1;
                }
                let mut code = e2m1_rne(q);
                if w < 0.0 && code != 0 {
                    code |= 8;
                }
                let c = 32 * g + j;
                nib[r * k / 2 + c / 2] |= code << ((c & 1) * 4);
                let back = E2M1[code as usize] as f64 * x;
                st.sq_err += (back - w as f64).powi(2);
                st.sq_ref += (w as f64).powi(2);
            }
        }
    }
    Proj::Mxfp4 { n, k, nib, e8 }
}

/// MXFP4 with the per-group exponent chosen to minimise the squared error among E0 - 1, E0, E0 + 1 (E0 the
/// rule above). Same storage; measured against the plain rule in `flashnext_check formats`.
pub fn mxfp4_from_f32_mse(n: usize, k: usize, v: &[f32], st: &mut QuantStats) -> Proj {
    assert!(k.is_multiple_of(32) && v.len() == n * k);
    let mut nib = vec![0u8; n * k / 2];
    let mut e8 = vec![0u8; n * k / 32];
    let enc = |w: f32, x: f64| -> u8 {
        let mut code = e2m1_rne(w.abs() as f64 / x);
        if w < 0.0 && code != 0 {
            code |= 8;
        }
        code
    };
    for r in 0..n {
        for g in 0..k / 32 {
            let grp = &v[r * k + 32 * g..r * k + 32 * g + 32];
            let amax = grp.iter().fold(0f32, |m, x| m.max(x.abs())) as f64;
            st.groups += 1;
            st.weights += 32;
            if amax == 0.0 {
                e8[r * k / 32 + g] = 127;
                st.zero_groups += 1;
                continue;
            }
            let e0 = amax.log2().floor() as i32 - 2;
            let mut best = (f64::INFINITY, e0);
            for e in [e0 - 1, e0, e0 + 1] {
                let x = (e as f64).exp2();
                let sse: f64 = grp
                    .iter()
                    .map(|&w| (E2M1[enc(w, x) as usize] as f64 * x - w as f64).powi(2))
                    .sum();
                if sse < best.0 {
                    best = (sse, e);
                }
            }
            let x = (best.1 as f64).exp2();
            e8[r * k / 32 + g] = (best.1 + 127).clamp(0, 254) as u8;
            for (j, &w) in grp.iter().enumerate() {
                let code = enc(w, x);
                let c = 32 * g + j;
                nib[r * k / 2 + c / 2] |= code << ((c & 1) * 4);
                st.sq_ref += (w as f64).powi(2);
            }
            st.sq_err += best.0;
        }
    }
    Proj::Mxfp4 { n, k, nib, e8 }
}

/// Reads a projection as stored: FP8 + scale_inv, bf16, or NVFP4 (U8 + weight_scale + weight_scale_2).
pub fn read_proj(
    src: &mut dyn TensorSource,
    base: &str,
    n: usize,
    k: usize,
) -> Result<Proj, String> {
    let name = format!("{base}.weight");
    if src.dtype_of(&name)?.as_deref() == Some("U8") {
        let mut packed = vec![0u8; n * k / 2];
        src.read_into(&name, "U8", &[n, k / 2], &mut packed)?;
        let mut scales = vec![0u8; n * k / 16];
        src.read_into(
            &format!("{base}.weight_scale"),
            "F8_E4M3",
            &[n, k / 16],
            &mut scales,
        )?;
        let mut g = [0u8; 4];
        src.read_into(&format!("{base}.weight_scale_2"), "F32", &[], &mut g)?;
        return Ok(Proj::Nvfp4 {
            n,
            k,
            packed,
            scales,
            global: f32::from_le_bytes(g),
        });
    }
    Ok(Proj::Dense(read_matrix(src, &name, n, k)?))
}

// ---------------------------------------------------------------------------
// Small maths
// ---------------------------------------------------------------------------

/// Zero-centred RMSNorm, normalising each `group` run and multiplying by `(1 + w)` (`w` covers the whole row).
pub fn rms_norm_zc_groups(x: &mut [f32], w: &[f32], group: usize, eps: f32) {
    for row in x.chunks_exact_mut(w.len()) {
        for (gi, g) in row.chunks_exact_mut(group).enumerate() {
            let ms = g.iter().map(|v| v * v).sum::<f32>() / group as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (j, v) in g.iter_mut().enumerate() {
                *v = *v * inv * (1.0 + w[gi * group + j]);
            }
        }
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn l2_normalize(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}
/// Rotate-half RoPE on the first `rotary` dims at `pos`.
pub fn rope(x: &mut [f32], pos: usize, rotary: usize, theta: f64) {
    let half = rotary / 2;
    for j in 0..half {
        let inv = 1.0 / theta.powf((2 * j) as f64 / rotary as f64);
        let (s, c) = (pos as f64 * inv).sin_cos();
        let (s, c) = (s as f32, c as f32);
        let (a, b) = (x[j], x[j + half]);
        x[j] = a * c - b * s;
        x[j + half] = b * c + a * s;
    }
}

// ---------------------------------------------------------------------------
// Gated residual (4 lanes)
// ---------------------------------------------------------------------------

pub struct GatedResidual {
    pub norm: Vec<f32>,
    pub down: Matrix,
    pub up: Matrix,
    pub inject: Option<Matrix>,
}

impl GatedResidual {
    pub fn read(
        src: &mut dyn TensorSource,
        p: &str,
        c: &FlashNextConfig,
        inject: bool,
    ) -> Result<Self, String> {
        let l = c.lanes();
        Ok(Self {
            norm: read_vec(src, &format!("{p}.hc_norm.weight"), l)?,
            down: read_matrix(
                src,
                &format!("{p}.input_mix_weight_down.weight"),
                c.hc_rank,
                l,
            )?,
            up: read_matrix(
                src,
                &format!("{p}.input_mix_weight_up.weight"),
                l,
                c.hc_rank,
            )?,
            inject: if inject {
                Some(read_matrix(
                    src,
                    &format!("{p}.block_inject_weight.weight"),
                    c.hc,
                    l,
                )?)
            } else {
                None
            },
        })
    }

    /// `(block input [t][hidden], injection [t][hc])` for `t` rows of lanes `[t][hc * hidden]`.
    pub fn forward(&self, c: &FlashNextConfig, lanes: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let (h, l, hc) = (c.hidden, c.lanes(), c.hc);
        let mut normed = lanes.to_vec();
        rms_norm_zc_groups(&mut normed, &self.norm, h, c.eps);
        let mut d = self.down.matmul(&normed, t);
        d.iter_mut().for_each(|v| *v = silu(*v / hc as f32));
        let mut mix = self.up.matmul(&d, t);
        mix.iter_mut().for_each(|v| *v = sigmoid(*v));
        let mut out = vec![0f32; t * h];
        for i in 0..t {
            for lane in 0..hc {
                for j in 0..h {
                    let at = i * l + lane * h + j;
                    out[i * h + j] += mix[at] * normed[at];
                }
            }
            out[i * h..(i + 1) * h]
                .iter_mut()
                .for_each(|v| *v /= hc as f32);
        }
        let inj = match &self.inject {
            Some(m) => m
                .matmul(&normed, t)
                .into_iter()
                .map(|v| 2.0 * sigmoid(v / hc as f32))
                .collect(),
            None => Vec::new(),
        };
        (out, inj)
    }
}

/// lanes[i][lane][j] += block[i][j] * inj[i][lane]
pub fn inject(c: &FlashNextConfig, lanes: &mut [f32], block: &[f32], inj: &[f32], t: usize) {
    let (h, l, hc) = (c.hidden, c.lanes(), c.hc);
    for i in 0..t {
        for lane in 0..hc {
            let w = inj[i * hc + lane];
            for j in 0..h {
                lanes[i * l + lane * h + j] += block[i * h + j] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

/// How expert weights are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFormat {
    /// As the checkpoint stores them (bf16, FP8 or NVFP4).
    AsStored,
    /// Converted on load to Omega's MXFP4 (the chip's numbers).
    Mxfp4,
}

pub struct Expert {
    pub gate: Proj,
    pub up: Proj,
    pub down: Proj,
}

impl Expert {
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let g = self.gate.matmul(x, t);
        let u = self.up.matmul(x, t);
        let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
        self.down.matmul(&a, t)
    }
}

/// Reads routed expert `e` of the MoE at `mlp` (`...layers.L.mlp`): per-expert tensors (FP8 / NVFP4 / bf16
/// aliases `experts.E.gate_proj.weight`), or the bf16 slices `experts.E.gate_up_proj` / `experts.E.down_proj`.
pub fn read_expert(
    src: &mut dyn TensorSource,
    mlp: &str,
    e: usize,
    c: &FlashNextConfig,
    fmt: ExpertFormat,
    stats: &mut QuantStats,
) -> Result<Expert, String> {
    let (h, m) = (c.hidden, c.moe_inter);
    let p = format!("{mlp}.experts.{e}");
    let ex = if src.dtype_of(&format!("{p}.gate_proj.weight"))?.is_some() {
        Expert {
            gate: read_proj(src, &format!("{p}.gate_proj"), m, h)?,
            up: read_proj(src, &format!("{p}.up_proj"), m, h)?,
            down: read_proj(src, &format!("{p}.down_proj"), h, m)?,
        }
    } else if src.dtype_of(&format!("{p}.gate_up_proj"))?.is_some() {
        let gu = read_bf16_words(src, &format!("{p}.gate_up_proj"), &[2 * m, h])?;
        let dn = read_bf16_words(src, &format!("{p}.down_proj"), &[h, m])?;
        Expert {
            gate: Proj::Dense(Matrix::Bf16 {
                n: m,
                k: h,
                w: gu[..m * h].to_vec(),
            }),
            up: Proj::Dense(Matrix::Bf16 {
                n: m,
                k: h,
                w: gu[m * h..].to_vec(),
            }),
            down: Proj::Dense(Matrix::Bf16 { n: h, k: m, w: dn }),
        }
    } else {
        return Err(format!("missing expert: {p}"));
    };
    Ok(match fmt {
        ExpertFormat::AsStored => ex,
        ExpertFormat::Mxfp4 => Expert {
            gate: mxfp4_from_f32(m, h, &ex.gate.dense(), stats),
            up: mxfp4_from_f32(m, h, &ex.up.dense(), stats),
            down: mxfp4_from_f32(h, m, &ex.down.dense(), stats),
        },
    })
}

/// Router decision for one row: ids (descending probability, lower id first on a tie) and weights.
pub fn route(c: &FlashNextConfig, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = ex.iter().sum();
    let probs: Vec<f32> = ex.iter().map(|v| v / sum).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(c.top_k);
    let mut w: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
    if c.norm_topk {
        let s: f32 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= s);
    }
    (idx, w)
}

pub struct MoeWeights {
    pub router: Matrix,
    pub shared: Expert,
    pub shared_gate: Matrix,
}

/// Everything one MoE call decided and produced (parity fixtures read these).
#[derive(Debug, Clone)]
pub struct MoeTrace {
    pub ids: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f32>>,
    pub shared_gate: Vec<f32>,
}

// ---------------------------------------------------------------------------
// DeltaNet and indexed attention
// ---------------------------------------------------------------------------

pub struct DeltaNetWeights {
    pub in_qkv: Matrix,
    pub in_z: Matrix,
    pub in_a: Matrix,
    pub in_b: Matrix,
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: Matrix,
}

pub struct AttentionWeights {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub idx_qk: Matrix,
    pub idx_q_norm: Vec<f32>,
    pub idx_k_norm: Vec<f32>,
}

pub struct PleWeights {
    pub mult: Vec<i64>,
    pub sizes: Vec<u64>,
    pub offs: Vec<u64>,
    pub table_scale: f32,
    pub key: Matrix,
    pub value: Matrix,
    pub norm_key: Vec<f32>,
    pub norm_query: Vec<f32>,
    pub norm_conv: Vec<f32>,
    /// `[lanes][kernel]`
    pub conv: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum MixerState {
    DeltaNet {
        conv: Vec<f32>,
        s: Vec<f32>,
    },
    /// keys after norm + RoPE, values, raw indexer keys, each `[pos][..]`
    Attention {
        k: Vec<f32>,
        v: Vec<f32>,
        ik: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct FlashNextState {
    pub pos: usize,
    pub mixers: Vec<MixerState>,
    /// the last `ngram - 1` token ids (EOS at the start)
    pub ple_history: Vec<u32>,
    /// PLE conv inputs, `[state_len][lanes]`, oldest first, per PLE layer
    pub ple_conv: Vec<Vec<f32>>,
}

impl FlashNextState {
    pub fn new(c: &FlashNextConfig) -> Self {
        let mixers = c
            .full_attention
            .iter()
            .map(|&full| {
                if full {
                    MixerState::Attention {
                        k: Vec::new(),
                        v: Vec::new(),
                        ik: Vec::new(),
                    }
                } else {
                    MixerState::DeltaNet {
                        conv: vec![0.0; (c.conv_kernel - 1) * c.conv_dim()],
                        s: vec![0.0; c.lin_v_heads * c.lin_k_dim * c.lin_v_dim],
                    }
                }
            })
            .collect();
        Self {
            pos: 0,
            mixers,
            ple_history: vec![c.eos; c.ngram - 1],
            ple_conv: c
                .ple_layers
                .iter()
                .map(|_| vec![0.0; c.ple_state_len() * c.lanes()])
                .collect(),
        }
    }
}

pub fn delta_net(
    c: &FlashNextConfig,
    w: &DeltaNetWeights,
    xn: &[f32],
    t: usize,
    conv_tail: &mut [f32],
    state: &mut [f32],
) -> Vec<f32> {
    let (cd, kern) = (c.conv_dim(), c.conv_kernel);
    let qkv = w.in_qkv.matmul(xn, t);
    let z = w.in_z.matmul(xn, t);
    let a = w.in_a.matmul(xn, t);
    let b = w.in_b.matmul(xn, t);
    let hist = kern - 1;
    let mut seq = conv_tail.to_vec();
    seq.extend_from_slice(&qkv);
    let mut mixed = vec![0f32; t * cd];
    for i in 0..t {
        for ch in 0..cd {
            let taps = &w.conv[ch * kern..(ch + 1) * kern];
            let acc: f32 = (0..kern).map(|j| taps[j] * seq[(i + j) * cd + ch]).sum();
            mixed[i * cd + ch] = silu(acc);
        }
    }
    conv_tail.copy_from_slice(&seq[t * cd..(t + hist) * cd]);
    let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
    let ratio = vh / kh;
    let key_dim = kh * kd;
    let mut q = vec![0f32; t * key_dim];
    let mut k = vec![0f32; t * key_dim];
    let qs = 1.0 / (kd as f32).sqrt();
    for i in 0..t {
        let row = &mixed[i * cd..(i + 1) * cd];
        q[i * key_dim..(i + 1) * key_dim].copy_from_slice(&row[..key_dim]);
        k[i * key_dim..(i + 1) * key_dim].copy_from_slice(&row[key_dim..2 * key_dim]);
        for hh in 0..kh {
            let r = i * key_dim + hh * kd..i * key_dim + (hh + 1) * kd;
            l2_normalize(&mut q[r.clone()], 1e-6);
            l2_normalize(&mut k[r.clone()], 1e-6);
            q[r].iter_mut().for_each(|v| *v *= qs);
        }
    }
    let mut per_head: Vec<Vec<f32>> = vec![Vec::new(); vh];
    state
        .par_chunks_mut(kd * vd)
        .zip(per_head.par_iter_mut())
        .enumerate()
        .for_each(|(hv, (s, out))| {
            let hk = hv / ratio;
            let rate = -w.a_log[hv].exp();
            out.resize(t * vd, 0.0);
            let mut kv = vec![0f32; vd];
            for i in 0..t {
                let kt = &k[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
                let qt = &q[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
                let vt =
                    &mixed[i * cd + 2 * key_dim + hv * vd..i * cd + 2 * key_dim + (hv + 1) * vd];
                let g = rate * softplus(a[i * vh + hv] + w.dt_bias[hv]);
                let beta = sigmoid(b[i * vh + hv]);
                let decay = g.exp();
                s.iter_mut().for_each(|v| *v *= decay);
                kv.iter_mut().for_each(|v| *v = 0.0);
                for (r, &kr) in kt.iter().enumerate() {
                    for (o, &sv) in kv.iter_mut().zip(&s[r * vd..(r + 1) * vd]) {
                        *o += sv * kr;
                    }
                }
                let delta: Vec<f32> = vt.iter().zip(&kv).map(|(&v, &m)| (v - m) * beta).collect();
                for (r, &kr) in kt.iter().enumerate() {
                    for (sv, &d) in s[r * vd..(r + 1) * vd].iter_mut().zip(&delta) {
                        *sv += kr * d;
                    }
                }
                let o = &mut out[i * vd..(i + 1) * vd];
                o.iter_mut().for_each(|v| *v = 0.0);
                for (r, &qr) in qt.iter().enumerate() {
                    for (ov, &sv) in o.iter_mut().zip(&s[r * vd..(r + 1) * vd]) {
                        *ov += sv * qr;
                    }
                }
            }
        });
    let mut core = vec![0f32; t * vh * vd];
    for (hv, out) in per_head.iter().enumerate() {
        for i in 0..t {
            core[i * vh * vd + hv * vd..i * vh * vd + (hv + 1) * vd]
                .copy_from_slice(&out[i * vd..(i + 1) * vd]);
        }
    }
    for (o, zr) in core.chunks_exact_mut(vd).zip(z.chunks_exact(vd)) {
        let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
        let inv = 1.0 / (ms + c.eps).sqrt();
        for ((v, &g), &zz) in o.iter_mut().zip(&w.norm).zip(zr) {
            let act = match c.gdn_gate {
                GateAct::Sigmoid => sigmoid(zz),
                GateAct::Silu => silu(zz),
            };
            *v = *v * inv * g * act;
        }
    }
    w.out.matmul(&core, t)
}

/// The QSA indexer for one query: positions (ascending) this query attends to, given its normed + turned
/// indexer query `[idx_heads][idx_dim]` and the raw keys of positions `0..=p` (`[p+1][idx_dim]`).
pub fn indexer_select(
    c: &FlashNextConfig,
    w_k_norm: &[f32],
    q: &[f32],
    raw_keys: &[f32],
    p: usize,
) -> Vec<usize> {
    let (d, r) = (c.idx_head_dim, c.idx_ratio);
    let visible = p + 1;
    let nb = visible / r;
    let topk = c.idx_budget / r;
    let mut sel: Vec<usize> = Vec::with_capacity(c.idx_budget + r);
    if nb > 0 {
        let scale = 1.0 / (d as f32).sqrt();
        let scores: Vec<f32> = (0..nb)
            .into_par_iter()
            .map(|b| {
                let mut kb = vec![0f32; d];
                for tok in b * r..(b + 1) * r {
                    for (o, &v) in kb.iter_mut().zip(&raw_keys[tok * d..(tok + 1) * d]) {
                        *o += v;
                    }
                }
                kb.iter_mut().for_each(|v| *v /= r as f32);
                rms_norm_zc_groups(&mut kb, w_k_norm, d, c.eps);
                rope(&mut kb, b * r, c.rotary_dim, c.rope_theta);
                (0..c.idx_heads)
                    .map(|h| dot(&q[h * d..(h + 1) * d], &kb).max(0.0))
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let mut order: Vec<usize> = (0..nb).collect();
        if nb > topk {
            order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            order.truncate(topk);
            order.sort_unstable();
        }
        for b in order {
            sel.extend(b * r..(b + 1) * r);
        }
    }
    sel.extend(nb * r..visible);
    sel
}

/// Indexed (QSA) attention over `t` new rows starting at `pos0`; appends to the caches. `selected` (if given)
/// receives each row's selected positions.
#[allow(clippy::too_many_arguments)]
pub fn indexed_attention(
    c: &FlashNextConfig,
    w: &AttentionWeights,
    xn: &[f32],
    t: usize,
    pos0: usize,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    ik_cache: &mut Vec<f32>,
    selected: Option<&mut Vec<Vec<usize>>>,
) -> Vec<f32> {
    let (nh, hd, kvd, id) = (c.heads, c.head_dim, c.kv_dim(), c.idx_head_dim);
    let group = nh / c.kv_heads;
    let qg = w.q.matmul(xn, t);
    let mut k = w.k.matmul(xn, t);
    let v = w.v.matmul(xn, t);
    let iqk = w.idx_qk.matmul(xn, t);
    let iw = (c.idx_heads + 1) * id;
    let mut q = vec![0f32; t * nh * hd];
    let mut gate = vec![0f32; t * nh * hd];
    let mut iq = vec![0f32; t * c.idx_heads * id];
    for i in 0..t {
        for hh in 0..nh {
            let s = &qg[(i * nh + hh) * 2 * hd..(i * nh + hh + 1) * 2 * hd];
            let dq = &mut q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            dq.copy_from_slice(&s[..hd]);
            rms_norm_zc_groups(dq, &w.q_norm, hd, c.eps);
            rope(dq, pos0 + i, c.rotary_dim, c.rope_theta);
            gate[(i * nh + hh) * hd..(i * nh + hh + 1) * hd].copy_from_slice(&s[hd..]);
        }
        for hh in 0..c.kv_heads {
            let dk = &mut k[i * kvd + hh * hd..i * kvd + (hh + 1) * hd];
            rms_norm_zc_groups(dk, &w.k_norm, hd, c.eps);
            rope(dk, pos0 + i, c.rotary_dim, c.rope_theta);
        }
        for hh in 0..c.idx_heads {
            let dq = &mut iq[(i * c.idx_heads + hh) * id..(i * c.idx_heads + hh + 1) * id];
            dq.copy_from_slice(&iqk[i * iw + hh * id..i * iw + (hh + 1) * id]);
            rms_norm_zc_groups(dq, &w.idx_q_norm, id, c.eps);
            rope(dq, pos0 + i, c.rotary_dim, c.rope_theta);
        }
        ik_cache.extend_from_slice(&iqk[i * iw + c.idx_heads * id..(i + 1) * iw]);
    }
    k_cache.extend_from_slice(&k);
    v_cache.extend_from_slice(&v);
    let sels: Vec<Vec<usize>> = (0..t)
        .map(|i| {
            let p = pos0 + i;
            indexer_select(
                c,
                &w.idx_k_norm,
                &iq[i * c.idx_heads * id..(i + 1) * c.idx_heads * id],
                ik_cache,
                p,
            )
        })
        .collect();
    let (kc, vc) = (&k_cache[..], &v_cache[..]);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; t * nh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
        let (i, hh) = (idx / nh, idx % nh);
        let kh = hh / group;
        let qv = &q[idx * hd..(idx + 1) * hd];
        let sel = &sels[i];
        let scores: Vec<f32> = sel
            .iter()
            .map(|&p| dot(qv, &kc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd]) * scale)
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let wts: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = wts.iter().sum();
        for (&p, &wt) in sel.iter().zip(&wts) {
            for (ov, &vv) in o
                .iter_mut()
                .zip(&vc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd])
            {
                *ov += wt * vv;
            }
        }
        for (ov, &gv) in o.iter_mut().zip(&gate[idx * hd..(idx + 1) * hd]) {
            *ov = *ov / sum * sigmoid(gv);
        }
    });
    if let Some(s) = selected {
        s.extend(sels);
    }
    w.o.matmul(&out, t)
}

// ---------------------------------------------------------------------------
// The reference model
// ---------------------------------------------------------------------------

/// Reads the 160-byte FP8 rows of the PLE n-gram table by global row id.
pub trait NgramRows {
    fn row(&mut self, id: u64, dst: &mut [u8]) -> Result<(), String>;
}

/// Rows from a tensor source holding aliases `ngram_row.<id>` (fetched rows) or the full shards.
pub struct SourceRows<'a> {
    pub src: &'a mut dyn TensorSource,
    pub prefix: String,
    pub shard_rows: u64,
    pub row_len: usize,
}

impl NgramRows for SourceRows<'_> {
    fn row(&mut self, id: u64, dst: &mut [u8]) -> Result<(), String> {
        let alias = format!("ngram_row.{id}");
        if self.src.dtype_of(&alias)?.is_some() {
            return self.src.read_into(&alias, "F8_E4M3", &[self.row_len], dst);
        }
        Err(format!(
            "missing ngram row: {id} (shard {} row {})",
            id / self.shard_rows,
            id % self.shard_rows
        ))
    }
}

pub struct FlashNextReference<S: TensorSource> {
    pub config: FlashNextConfig,
    pub source: S,
    pub prefix: String,
    pub expert_format: ExpertFormat,
    pub quant: QuantStats,
    /// per-layer cache of experts already read
    experts: HashMap<(usize, usize), Expert>,
    /// whether to skip the lm_head (partial-layer runs on fetched tensors)
    pub with_head: bool,
    pub traces: Vec<MoeTrace>,
    pub selections: Vec<Vec<Vec<usize>>>,
    /// per layer of the last forward: the MoE input `[t][hidden]` and output
    pub moe_inputs: Vec<Vec<f32>>,
    pub moe_outputs: Vec<Vec<f32>>,
}

impl FlashNextReference<ShardSource> {
    pub fn open(dir: &Path, layers: Option<usize>) -> Result<Self, String> {
        let mut c = FlashNextConfig::from_checkpoint(dir)?;
        if let Some(n) = layers {
            c = c.prefix(n).ok_or("layers out of range")?;
        }
        Ok(Self::new(c, ShardSource::open(dir)?))
    }
}

impl<S: TensorSource> FlashNextReference<S> {
    pub fn new(config: FlashNextConfig, source: S) -> Self {
        Self {
            config,
            source,
            prefix: "model.language_model.".into(),
            expert_format: ExpertFormat::AsStored,
            quant: QuantStats::default(),
            experts: HashMap::new(),
            with_head: true,
            traces: Vec::new(),
            selections: Vec::new(),
            moe_inputs: Vec::new(),
            moe_outputs: Vec::new(),
        }
    }

    /// A routed expert the last forward read (layer, id).
    pub fn expert(&self, layer: usize, e: usize) -> Option<&Expert> {
        self.experts.get(&(layer, e))
    }

    pub fn new_state(&self) -> FlashNextState {
        FlashNextState::new(&self.config)
    }

    fn embed(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        let h = self.config.hidden;
        let alias = format!("embed_row.{tok}");
        let words = if self.source.dtype_of(&alias)?.is_some() {
            read_bf16_words(&mut self.source, &alias, &[h])?
        } else {
            let name = format!("{}embed_tokens.weight", self.prefix);
            let all = read_bf16_words(&mut self.source, &name, &[self.config.vocab, h])?;
            all[tok as usize * h..(tok as usize + 1) * h].to_vec()
        };
        Ok(words.into_iter().map(bf16_to_f32).collect())
    }

    fn read_dn(&mut self, p: &str) -> Result<DeltaNetWeights, String> {
        let c = &self.config;
        let (h, src) = (c.hidden, &mut self.source as &mut dyn TensorSource);
        let a = format!("{p}.linear_attn");
        Ok(DeltaNetWeights {
            in_qkv: read_matrix(src, &format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
            in_z: read_matrix(src, &format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h)?,
            in_a: read_matrix(src, &format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?,
            in_b: read_matrix(src, &format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?,
            conv: read_bf16_words(
                src,
                &format!("{a}.conv1d.weight"),
                &[c.conv_dim(), 1, c.conv_kernel],
            )?
            .into_iter()
            .map(bf16_to_f32)
            .collect(),
            a_log: read_vec(src, &format!("{a}.A_log"), c.lin_v_heads)?,
            dt_bias: read_vec(src, &format!("{a}.dt_bias"), c.lin_v_heads)?,
            norm: read_vec(src, &format!("{a}.norm.weight"), c.lin_v_dim)?,
            out: read_matrix(src, &format!("{a}.out_proj.weight"), h, c.lin_value_dim())?,
        })
    }

    fn read_attn(&mut self, p: &str) -> Result<AttentionWeights, String> {
        let c = &self.config;
        let (h, src) = (c.hidden, &mut self.source as &mut dyn TensorSource);
        let a = format!("{p}.self_attn");
        Ok(AttentionWeights {
            q: read_matrix(
                src,
                &format!("{a}.q_proj.weight"),
                2 * c.heads * c.head_dim,
                h,
            )?,
            k: read_matrix(src, &format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
            v: read_matrix(src, &format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
            o: read_matrix(src, &format!("{a}.o_proj.weight"), h, c.heads * c.head_dim)?,
            q_norm: read_vec(src, &format!("{a}.q_norm.weight"), c.head_dim)?,
            k_norm: read_vec(src, &format!("{a}.k_norm.weight"), c.head_dim)?,
            idx_qk: read_matrix(
                src,
                &format!("{a}.indexer.index_qk_proj.weight"),
                (c.idx_heads + 1) * c.idx_head_dim,
                h,
            )?,
            idx_q_norm: read_vec(
                src,
                &format!("{a}.indexer.q_layernorm.weight"),
                c.idx_head_dim,
            )?,
            idx_k_norm: read_vec(
                src,
                &format!("{a}.indexer.k_layernorm.weight"),
                c.idx_head_dim,
            )?,
        })
    }

    fn read_ple(&mut self, p: &str) -> Result<PleWeights, String> {
        let c = self.config.clone();
        let (h, l, src) = (
            c.hidden,
            c.lanes(),
            &mut self.source as &mut dyn TensorSource,
        );
        let e = format!("{p}.ple.ple_embedding");
        let nh = c.ngram_heads();
        let i64s = |src: &mut dyn TensorSource, name: &str, n: usize| -> Result<Vec<i64>, String> {
            let mut b = vec![0u8; 8 * n];
            src.read_into(name, "I64", &[n], &mut b)?;
            Ok(b.chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
                .collect())
        };
        let mult = i64s(src, &format!("{e}.layer_multipliers"), c.ngram)?;
        let sizes = i64s(src, &format!("{e}.ngram_heads_vocab_sizes"), nh)?
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let offs = i64s(src, &format!("{e}.ngram_heads_offsets"), nh)?
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let table_scale = read_vec(src, &format!("{e}.ngram_embedding.weight_scale"), 1)?[0];
        Ok(PleWeights {
            mult,
            sizes,
            offs,
            table_scale,
            key: read_matrix(src, &format!("{p}.ple.key_proj.weight"), l, h)?,
            value: read_matrix(src, &format!("{p}.ple.value_proj.weight"), h, h)?,
            norm_key: read_vec(src, &format!("{p}.ple.norm_key.weight"), l)?,
            norm_query: read_vec(src, &format!("{p}.ple.norm_query.weight"), l)?,
            norm_conv: read_vec(src, &format!("{p}.ple.norm_conv.weight"), l)?,
            conv: read_bf16_words(
                src,
                &format!("{p}.ple.conv1d.weight"),
                &[l, 1, c.ple_conv_kernel],
            )?
            .into_iter()
            .map(bf16_to_f32)
            .collect(),
        })
    }

    /// The PLE block's output `[t][lanes]` for `tokens` (updates history and conv state).
    fn ple(
        &mut self,
        w: &PleWeights,
        lanes: &[f32],
        tokens: &[u32],
        hist: &mut Vec<u32>,
        conv_state: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let c = self.config.clone();
        let (h, l, hc, t) = (c.hidden, c.lanes(), c.hc, tokens.len());
        let ids = ngram_ids(&c, &w.mult, &w.sizes, &w.offs, hist, tokens);
        let mut all: Vec<u32> = hist.clone();
        all.extend_from_slice(tokens);
        *hist = all[all.len() - (c.ngram - 1)..].to_vec();
        let rl = c.ngram_row();
        let mut emb = vec![0f32; t * h];
        let mut buf = vec![0u8; rl];
        {
            let mut rows = SourceRows {
                src: &mut self.source,
                prefix: self.prefix.clone(),
                shard_rows: 2_500_012,
                row_len: rl,
            };
            for (j, &id) in ids.iter().enumerate() {
                rows.row(id, &mut buf)?;
                for (o, &b) in emb[j * rl..(j + 1) * rl].iter_mut().zip(&buf) {
                    *o = fp8_e4m3fn_to_f32(b) * w.table_scale;
                }
            }
        }
        let mut key = w.key.matmul(&emb, t);
        rms_norm_zc_groups(&mut key, &w.norm_key, h, c.eps);
        let value = w.value.matmul(&emb, t);
        let mut query = lanes.to_vec();
        rms_norm_zc_groups(&mut query, &w.norm_query, h, c.eps);
        let mut gv = vec![0f32; t * l];
        for i in 0..t {
            for lane in 0..hc {
                let r = i * l + lane * h..i * l + (lane + 1) * h;
                let g = dot(&key[r.clone()], &query[r.clone()]) / (h as f32).sqrt();
                let g = g.abs().max(1e-6).sqrt() * g.signum_or_zero();
                let s = sigmoid(g);
                for j in 0..h {
                    gv[i * l + lane * h + j] = s * value[i * h + j];
                }
            }
        }
        let mut gvn = gv.clone();
        rms_norm_zc_groups(&mut gvn, &w.norm_conv, h, c.eps);
        // dilated causal depthwise conv over [state | new], kernel K, dilation ngram; output for new rows
        let (kern, dil, sl) = (c.ple_conv_kernel, c.ngram, c.ple_state_len());
        let mut seq = conv_state.clone();
        seq.extend_from_slice(&gvn);
        let mut out = gv;
        for i in 0..t {
            for ch in 0..l {
                let taps = &w.conv[ch * kern..(ch + 1) * kern];
                let acc: f32 = (0..kern)
                    .map(|j| taps[j] * seq[(i + j * dil) * l + ch])
                    .sum();
                out[i * l + ch] += silu(acc);
            }
        }
        *conv_state = seq[t * l..(t + sl) * l].to_vec();
        Ok(out)
    }

    fn moe(
        &mut self,
        layer: usize,
        mw: &MoeWeights,
        x: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let c = self.config.clone();
        let h = c.hidden;
        let logits = mw.router.matmul(x, t);
        let sg = mw.shared_gate.matmul(x, t);
        let mut out = mw.shared.forward(x, t);
        let mut trace = MoeTrace {
            ids: Vec::new(),
            weights: Vec::new(),
            shared_gate: Vec::new(),
        };
        for i in 0..t {
            let g = sigmoid(sg[i]);
            out[i * h..(i + 1) * h].iter_mut().for_each(|v| *v *= g);
            trace.shared_gate.push(g);
            let (ids, ws) = route(&c, &logits[i * c.experts..(i + 1) * c.experts]);
            trace.ids.push(ids);
            trace.weights.push(ws);
        }
        // read every expert this call needs (report all missing at once)
        let mlp = format!("{}layers.{layer}.mlp", self.prefix);
        let mut missing = Vec::new();
        let mut need: Vec<usize> = trace.ids.iter().flatten().copied().collect();
        need.sort_unstable();
        need.dedup();
        for &e in &need {
            if self.experts.contains_key(&(layer, e)) {
                continue;
            }
            match read_expert(
                &mut self.source,
                &mlp,
                e,
                &c,
                self.expert_format,
                &mut self.quant,
            ) {
                Ok(ex) => {
                    self.experts.insert((layer, e), ex);
                }
                Err(err) if err.starts_with("missing expert") => missing.push(e),
                Err(err) => return Err(err),
            }
        }
        if !missing.is_empty() {
            let list: Vec<String> = missing.iter().map(|e| e.to_string()).collect();
            return Err(format!(
                "missing experts: layer {layer}: {}",
                list.join(" ")
            ));
        }
        for i in 0..t {
            let xi = &x[i * h..(i + 1) * h];
            for (&e, &w) in trace.ids[i].iter().zip(&trace.weights[i]) {
                let y = self.experts[&(layer, e)].forward(xi, 1);
                for (o, v) in out[i * h..(i + 1) * h].iter_mut().zip(&y) {
                    *o += w * v;
                }
            }
        }
        self.traces.push(trace);
        Ok(out)
    }

    /// Runs `tokens` after `state`; returns the mixer output `[t][hidden]` (and logits `[t][vocab]` when
    /// `with_head`). `taps` receives the lanes entering every layer (`[t][lanes]`) when given.
    pub fn forward(
        &mut self,
        state: &mut FlashNextState,
        tokens: &[u32],
        mut taps: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let c = self.config.clone();
        let (t, h, l) = (tokens.len(), c.hidden, c.lanes());
        if t == 0 {
            return Err("forward: no tokens".into());
        }
        let mut lanes = vec![0f32; t * l];
        for (i, &tok) in tokens.iter().enumerate() {
            if tok as usize >= c.vocab {
                return Err(format!("forward: token {tok} outside the vocab"));
            }
            let e = self.embed(tok)?;
            for lane in 0..c.hc {
                lanes[i * l + lane * h..i * l + (lane + 1) * h].copy_from_slice(&e);
            }
        }
        self.traces.clear();
        self.selections.clear();
        self.moe_inputs.clear();
        self.moe_outputs.clear();
        for layer in 0..c.layers {
            if let Some(tp) = taps.as_deref_mut() {
                tp.push(lanes.clone());
            }
            let p = format!("{}layers.{layer}", self.prefix);
            if let Some(pi) = c.ple_layers.iter().position(|&x| x == layer) {
                let w = self.read_ple(&p)?;
                let mut hist = std::mem::take(&mut state.ple_history);
                let mut cs = std::mem::take(&mut state.ple_conv[pi]);
                let add = self.ple(&w, &lanes, tokens, &mut hist, &mut cs);
                state.ple_history = hist;
                state.ple_conv[pi] = cs;
                lanes.iter_mut().zip(&add?).for_each(|(a, b)| *a += b);
            }
            let gr = GatedResidual::read(
                &mut self.source,
                &format!("{p}.attn_hyper_connection"),
                &c,
                true,
            )?;
            let (xin, inj) = gr.forward(&c, &lanes, t);
            let mixed = if c.full_attention[layer] {
                let w = self.read_attn(&p)?;
                let MixerState::Attention { k, v, ik } = &mut state.mixers[layer] else {
                    return Err(format!("layer {layer}: state kind"));
                };
                let mut sel = Vec::new();
                let o = indexed_attention(&c, &w, &xin, t, state.pos, k, v, ik, Some(&mut sel));
                self.selections.push(sel);
                o
            } else {
                let w = self.read_dn(&p)?;
                let MixerState::DeltaNet { conv, s } = &mut state.mixers[layer] else {
                    return Err(format!("layer {layer}: state kind"));
                };
                delta_net(&c, &w, &xin, t, conv, s)
            };
            inject(&c, &mut lanes, &mixed, &inj, t);
            let gr = GatedResidual::read(
                &mut self.source,
                &format!("{p}.mlp_hyper_connection"),
                &c,
                true,
            )?;
            let (xin, inj) = gr.forward(&c, &lanes, t);
            let mw = self.read_moe(&p)?;
            let y = self.moe(layer, &mw, &xin, t)?;
            self.moe_inputs.push(xin);
            self.moe_outputs.push(y.clone());
            inject(&c, &mut lanes, &y, &inj, t);
        }
        if let Some(tp) = taps {
            tp.push(lanes.clone());
        }
        state.pos += t;
        let mixer = GatedResidual::read(
            &mut self.source,
            &format!("{}hyper_connection_mixer", self.prefix),
            &c,
            false,
        )?;
        let (out, _) = mixer.forward(&c, &lanes, t);
        let logits = if self.with_head {
            read_matrix(&mut self.source, "lm_head.weight", c.vocab, h)?.matmul(&out, t)
        } else {
            Vec::new()
        };
        Ok((out, logits))
    }

    pub fn read_moe(&mut self, p: &str) -> Result<MoeWeights, String> {
        let c = &self.config;
        let (h, m, src) = (
            c.hidden,
            c.shared_inter,
            &mut self.source as &mut dyn TensorSource,
        );
        let s = format!("{p}.mlp.shared_expert");
        let mut shared = Expert {
            gate: read_proj(src, &format!("{s}.gate_proj"), m, h)?,
            up: read_proj(src, &format!("{s}.up_proj"), m, h)?,
            down: read_proj(src, &format!("{s}.down_proj"), h, m)?,
        };
        if self.expert_format == ExpertFormat::Mxfp4 {
            // the chip runs the shared expert as the 11th W4 group
            let mut q = QuantStats::default();
            shared = Expert {
                gate: mxfp4_from_f32(m, h, &shared.gate.dense(), &mut q),
                up: mxfp4_from_f32(m, h, &shared.up.dense(), &mut q),
                down: mxfp4_from_f32(h, m, &shared.down.dense(), &mut q),
            };
        }
        Ok(MoeWeights {
            router: read_matrix(src, &format!("{p}.mlp.gate.weight"), c.experts, h)?,
            shared,
            shared_gate: read_matrix(src, &format!("{p}.mlp.shared_expert_gate.weight"), 1, h)?,
        })
    }
}

trait SignumOrZero {
    fn signum_or_zero(self) -> Self;
}
impl SignumOrZero for f32 {
    /// torch.sign: 0 for 0.
    fn signum_or_zero(self) -> f32 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests;
