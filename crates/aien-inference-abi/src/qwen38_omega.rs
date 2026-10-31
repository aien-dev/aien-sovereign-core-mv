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

//! Qwen3.8-27B on the Omega GPU engine (sc #642): the checkpoint uploaded once, then one command
//! list per decoded token (`aien_omega_q38::Q38Graph`, omega `src/omega_gpu_q38_graph.h`).
//!
//! What the upload does to the checkpoint (the graph's contract, checked against
//! [`crate::qwen38`], the host reference):
//!
//! - DeltaNet layer: `in_proj_qkv | in_proj_z` -> one FP8 tensor `[16384][5120]` (rows
//!   concatenated; both are whole 128-row blocks, so the scale grids concatenate too).
//!   `in_proj_a` / `in_proj_b` (bf16 `[48][5120]`) -> one bf16 tensor `[96][5120]`, row `2h` =
//!   a_h, `2h + 1` = b_h, every value times 2^-64 (an exact exponent shift: the graph feeds that
//!   GEMV the same 2^64-scaled row as the FP8 ones, so its output comes out plain).
//!   conv1d, A_log, dt_bias and the gated norm weight go to [`aien_omega_q38::Gdn`] as stored.
//! - Attention layer: `q_proj | k_proj | v_proj` -> one FP8 tensor `[14336][5120]`; q_norm /
//!   k_norm to [`aien_omega_q38::Attn256Cache`] (folded to `1 + w` and permuted there).
//! - Every layer: `gate_proj | up_proj` -> `[34816][5120]`; o_proj / out_proj and down_proj as
//!   stored; input / post-attention norms folded to `1 + w` (the model's zero-centred RMSNorm).
//! - The final norm folded to `1 + w`; lm_head bf16 `[vocab][5120]` as stored. The embedding
//!   stays on the host (one row per token is widened and written before the token).
//!
//! The decode graph covers one token at a time; prompts run through it token by token until the
//! prefill path lands (sc #642 step 9).
use std::path::Path;

use aien_omega_bf16::Bf16Tensor;
use aien_omega_fp8::Fp8Tensor;
use aien_omega_gpu::ResidentTensor;
use aien_omega_q38::q36::Q36Graph;
use aien_omega_q38::{Attn256Cache, Gdn, GraphConfig, GraphStats, Q38Graph};
use aien_omega_router::Router;

mod moe;
pub use moe::router_probe;
mod prefill;
mod rows;
pub use rows::{RowFeed, RowOut, RowTest};
mod session;
pub use prefill::{q36_prefill_chunk, PrefillStats};
pub use session::{plan_reuse, FeedStats, Qwen38Session, Reuse, IM_START};

use crate::qwen38::{read_bf16_words, read_matrix, read_vec, text_prefix, Matrix, Qwen38Config};
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::{ShardSource, TensorSource};

/// l2norm eps of the DeltaNet q/k (the reference's constant).
const EPS_L2: f32 = 1e-6;

/// One layer's device tensors (kept alive for the graph).
struct LayerTensors {
    mix: Fp8Tensor,
    ab: Option<Bf16Tensor>,
    o: Fp8Tensor,
    gate_up: Fp8Tensor,
    down: Fp8Tensor,
}

/// The decode graph: the 27B's dense one or the 35B's sparse one (sc L306).
enum Graph {
    Dense(Q38Graph),
    Moe(Q36Graph),
}

impl Graph {
    fn token(
        &mut self,
        x: &[f32],
        pos: usize,
    ) -> Result<(u32, GraphStats), aien_omega_gpu::OmegaGpuError> {
        match self {
            Graph::Dense(g) => g.token(x, pos),
            Graph::Moe(g) => g.token(x, pos),
        }
    }

    fn read(
        &mut self,
        x: Option<&mut [f32]>,
        hidden: Option<&mut [f32]>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), aien_omega_gpu::OmegaGpuError> {
        match self {
            Graph::Dense(g) => g.read(x, hidden, logits),
            Graph::Moe(g) => g.read(x, hidden, logits),
        }
    }

    /// One stroke of `pos.len()` heads (the 35B graph; the 27B dense graph strokes one).
    fn stroke(&mut self, x: &[f32], pos: &[usize]) -> Result<(Vec<u32>, GraphStats), String> {
        match self {
            Graph::Dense(g) if pos.len() == 1 => g
                .token(x, pos[0])
                .map(|(t, s)| (vec![t], s))
                .map_err(gpu("q38 token")),
            Graph::Dense(_) => Err("the 27B dense graph strokes one conversation".into()),
            Graph::Moe(g) => g.stroke(x, pos).map_err(gpu("q36 stroke")),
        }
    }

    fn read_head(
        &mut self,
        head: usize,
        x: Option<&mut [f32]>,
        hidden: Option<&mut [f32]>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), String> {
        match self {
            Graph::Dense(g) if head == 0 => g.read(x, hidden, logits).map_err(gpu("q38 read")),
            Graph::Dense(_) => Err("the 27B dense graph has one head".into()),
            Graph::Moe(g) => g
                .read_head(head, x, hidden, logits)
                .map_err(gpu("q36 read head")),
        }
    }
}

/// One conversation's state on the chip: what a head slot of the graph points at.
#[derive(Default)]
pub(crate) struct HeadState {
    pub(crate) gdn: Option<Gdn>,
    pub(crate) attn: Option<Attn256Cache>,
    pub(crate) pos: usize,
}

/// The whole model resident on the device plus the host embedding.
///
/// Field order is drop order: the graph goes first, then the tensors and state it points at.
pub struct Qwen38Omega {
    graph: Graph,
    /// The rows' multi-conversation graph (L312's strokes): its head slots are pointed at the rows
    /// of each [`Self::step_rows`] chunk; it points at their state, so it drops before them.
    row_stroke: Option<Q36Graph>,
    /// Extra conversations over the same weights (the 35B only, [`Self::add_rows`]); they point at
    /// the router, lm_head and layers below, so they drop right after the main graph.
    rows: Vec<rows::Q36Row>,
    /// The 35B's router (sparse MLP only); the graph points at it, so it drops after the graph.
    #[allow(dead_code)] // owned only
    router: Option<Router>,
    layers: Vec<LayerTensors>,
    lm_head: Bf16Tensor,
    /// FP8 copy of lm_head (env `AIEN_Q38_FP8_HEAD=1`, qwen38/decode-speed): the decode graph reads it instead.
    #[allow(dead_code)] // owned only: the graph points at it until it is dropped first
    lm_head8: Option<Fp8Tensor>,
    /// The 35B's exact two-pass head screen (HUM L314, on unless `AIEN_Q36_EXACT_HEAD=0`): the fp8b copy
    /// of lm_head the graph screens with, for greedy tokens only ([`Self::set_exact_head_on`]).
    #[allow(dead_code)] // owned only: the graph points at it until it is dropped first
    head_screen: Option<Fp8Tensor>,
    gdn: Option<Gdn>,
    attn: Option<Attn256Cache>,
    /// Heads 1.. of a multi-conversation graph ([`Self::open_window_heads`]): each one's own
    /// DeltaNet state and attention cache (the graph points at them: they drop after it) and
    /// position. Head 0 is `gdn`, `attn`, `pos` above.
    extra_heads: Vec<HeadState>,
    pub config: Qwen38Config,
    /// bf16 `[vocab][hidden]`.
    embed: Vec<u16>,
    /// Tokens the state holds (the next token's position).
    pub pos: usize,
    max_pos: usize,
    /// Every layer's (input, post-attention) norm weight and the final norm, folded to `1 + w`:
    /// the prefill normalises on the host.
    norms: Vec<(Vec<f32>, Vec<f32>)>,
    final_norm: Vec<f32>,
    /// in_proj_a|b of every DeltaNet layer for the prefill's batched projection, `[5120][96]`
    /// (the graph's interleaved rows transposed, plain values) on omega's resident bf16 matmul.
    ab_mm: Vec<ResidentTensor>,
    /// The prefill's buffers, made by the first [`Self::prefill`].
    pf: Option<prefill::PfState>,
    /// What a new row needs (the 35B only).
    row_src: Option<rows::RowSource>,
    /// Multi-row steps so far (the row-swap negative control counts them).
    row_steps: u64,
}

impl std::fmt::Debug for Qwen38Omega {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qwen38Omega {{ layers: {}, pos: {}, max_pos: {} }}",
            self.config.layers, self.pos, self.max_pos
        )
    }
}

/// The DeltaNet tails and states at a position ([`Qwen38Omega::snapshot`]), in host memory.
#[derive(Debug, Default, Clone)]
pub struct Qwen38Snapshot {
    pub pos: usize,
    tails: Vec<f32>,
    states: Vec<f32>,
}

impl Qwen38Snapshot {
    /// The first f32 (tails, then states) whose bits differ from `o`'s, or the length when one
    /// holds more; `None`: the same position and the same bits throughout.
    pub fn first_diff(&self, o: &Self) -> Option<usize> {
        let a = self.tails.iter().chain(&self.states);
        let b = o.tails.iter().chain(&o.states);
        if let Some(i) = a.zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
            return Some(i);
        }
        let (na, nb) = (
            self.tails.len() + self.states.len(),
            o.tails.len() + o.states.len(),
        );
        (na != nb || self.pos != o.pos).then_some(na.min(nb))
    }
}

/// Memory the whole model needs with a KV cache of `max_pos` positions and prefill buffers for
/// `chunk` tokens (device and host: the box's memory is one pool). Counted from the config:
/// weights, KV cache, prefill buffers, the host embedding table and one state snapshot, plus
/// `RUNTIME_SLACK` for what the count misses (allocator rounding, the tokenizer, host staging).
/// The 2026-10-18 128k TTFT run measured RSS 39 GiB where the old count said 35.5 GiB; the
/// embedding (2.37 GiB) and the slack close that gap.
pub fn need_bytes(c: &Qwen38Config, max_pos: usize, chunk: usize) -> u64 {
    const RUNTIME_SLACK: u64 = 3 << 30;
    let h = c.hidden as u64;
    if let Some(m) = &c.moe {
        // Qwen3.6-35B-A3B (sc L306): FP8 mixers and experts (routed + shared, one byte per weight
        // plus a bf16 scale per 128 x 128 block), bf16 a|b, router, lm_head and host embedding; no
        // prefill buffers (the prompt runs through the decode graph).
        let experts = (m.experts + 1) as u64 * 3 * m.expert_inter as u64 * h;
        let o = h * (c.heads * c.head_dim) as u64;
        let fp8: u64 = c
            .full_attention
            .iter()
            .map(|&attn| {
                let mix = if attn {
                    (2 * c.heads * c.head_dim + 2 * c.kv_dim()) as u64 * h
                } else {
                    (c.conv_dim() + c.lin_value_dim()) as u64 * h
                };
                mix + o + experts
            })
            .sum();
        let n_attn = c.full_attention.iter().filter(|&&a| a).count() as u64;
        let n_gdn = c.layers as u64 - n_attn;
        let ab = n_gdn * 2 * 2 * c.lin_v_heads as u64 * h;
        let router = c.layers as u64 * 2 * (m.experts + 1) as u64 * h;
        let weights = fp8 + fp8 / 8192 + ab + router + 2 * c.vocab as u64 * h;
        let kv = n_attn * 2 * c.kv_heads as u64 * max_pos as u64 * 2 * c.head_dim as u64
            + max_pos as u64 * 1024;
        let embed = 2 * c.vocab as u64 * h;
        let tail = (c.conv_kernel as u64 - 1) * c.conv_dim() as u64;
        let st = (c.lin_v_heads * c.lin_v_dim * c.lin_v_dim) as u64;
        let state = n_gdn * 4 * (tail + st);
        let _ = chunk;
        return weights + kv + embed + 2 * state + RUNTIME_SLACK;
    }
    let mlp = 3 * c.intermediate as u64 * h;
    let o = h * (c.heads * c.head_dim) as u64;
    let layers: u64 = c
        .full_attention
        .iter()
        .map(|&attn| {
            if attn {
                (2 * c.heads * c.head_dim + 2 * c.kv_dim()) as u64 * h + o + mlp
            } else {
                (c.conv_dim() + c.lin_value_dim()) as u64 * h + o + mlp + 4 * 2 * 2 * 48 * h
            }
        })
        .sum();
    let weights = layers + layers / 4096 + 2 * c.vocab as u64 * h;
    let n_attn = c.full_attention.iter().filter(|&&a| a).count() as u64;
    let n_gdn = c.layers as u64 - n_attn;
    let kv = n_attn * 2 * 4 * max_pos as u64 * 512 + max_pos as u64 * 1024;
    // xin, xq, xs, mix, act, out, plus the DeltaNet prefill scratch (omega_gpu_gdn_api.h).
    let m = chunk as u64;
    let bufs = m * (6144 * 4 + 17408 + 136 * 4 + 16384 * 4 + 17408 * 2 + 5120 * 4)
        + m * (40960 + 3 * 24576 + 384)
        + m.div_ceil(8) * 8 * 74496;
    let embed = 2 * c.vocab as u64 * h;
    let state =
        n_gdn * 4 * (aien_omega_q38::GDN_TAIL_FLOATS + aien_omega_q38::GDN_STATE_FLOATS) as u64;
    // Device state plus one host snapshot.
    weights + kv + bufs + embed + 2 * state + RUNTIME_SLACK
}

fn gpu(stage: &str) -> impl Fn(aien_omega_gpu::OmegaGpuError) -> String + '_ {
    move |e| format!("{stage}: {e}")
}

/// `1 + w`, the zero-centred RMSNorm weight the graph's norms multiply by.
fn fold(w: Vec<f32>) -> Vec<f32> {
    w.into_iter().map(|v| 1.0 + v).collect()
}

/// FP8 projections stacked by rows into one tensor (every part `[n_i][k]`, whole 128 blocks).
fn fuse_fp8(name: &str, parts: Vec<Matrix>) -> Result<Fp8Tensor, String> {
    let (mut n, mut k0) = (0usize, None);
    let (mut w, mut scales) = (Vec::new(), Vec::new());
    for m in parts {
        match m {
            Matrix::Fp8 {
                n: pn,
                k,
                w: pw,
                scales: ps,
            } => {
                if *k0.get_or_insert(k) != k {
                    return Err(format!("{name}: parts with different k"));
                }
                n += pn;
                w.extend_from_slice(&pw);
                scales.extend_from_slice(&ps);
            }
            Matrix::Bf16 { .. } => {
                return Err(format!(
                    "{name}: a bf16 part (the graph takes FP8 projections)"
                ))
            }
        }
    }
    let k = k0.ok_or_else(|| format!("{name}: no parts"))?;
    Fp8Tensor::upload(1, n, k, &w, &scales).map_err(gpu(name))
}

fn upload_fp8(name: &str, m: Matrix) -> Result<Fp8Tensor, String> {
    fuse_fp8(name, vec![m])
}

/// `in_proj_a` / `in_proj_b` interleaved by rows and scaled by 2^-64 (exact for every normal
/// bf16: only the exponent moves; the checkpoint's values sit far above 2^-62).
fn ab_rows(a: &Matrix, b: &Matrix) -> Result<(usize, usize, Vec<u16>), String> {
    let (
        Matrix::Bf16 { n, k, w: wa },
        Matrix::Bf16 {
            n: nb,
            k: kb,
            w: wb,
        },
    ) = (a, b)
    else {
        return Err("in_proj_a / in_proj_b: expected bf16".into());
    };
    if (n, k) != (nb, kb) {
        return Err("in_proj_a / in_proj_b: shapes differ".into());
    }
    let scale = (2.0f32).powi(-64);
    let shift = |v: u16| -> Result<u16, String> {
        let x = bf16_to_f32(v) * scale;
        let bits = x.to_bits();
        if x != 0.0 && (bits & 0xffff) != 0 {
            return Err(format!(
                "in_proj_a/b: {} * 2^-64 is not exact in bf16",
                bf16_to_f32(v)
            ));
        }
        Ok((bits >> 16) as u16)
    };
    let mut out = Vec::with_capacity(2 * n * k);
    for h in 0..*n {
        for row in [&wa[h * k..(h + 1) * k], &wb[h * k..(h + 1) * k]] {
            for &v in row {
                out.push(shift(v)?);
            }
        }
    }
    Ok((2 * n, *k, out))
}

/// [`ab_rows`]' `[n][k]` words back at the checkpoint's values (times 2^64, exact: the reverse
/// exponent shift) and transposed to `[k][n]`, the resident matmul's B layout.
fn ab_plain_transposed(n: usize, k: usize, w: &[u16]) -> Vec<u16> {
    let unscale = |v: u16| ((bf16_to_f32(v) * aien_omega_q38::ROW_SCALE).to_bits() >> 16) as u16;
    let mut t = vec![0u16; n * k];
    for (j, row) in w.chunks_exact(k).enumerate() {
        for (i, &v) in row.iter().enumerate() {
            t[i * n + j] = unscale(v);
        }
    }
    t
}

impl Qwen38Omega {
    /// Opens a checkpoint directory and uploads its first `layers` layers (all when `None`) with
    /// a KV cache of `max_pos` positions per attention layer.
    pub fn open(dir: &Path, layers: Option<usize>, max_pos: usize) -> Result<Self, String> {
        let full = Qwen38Config::from_checkpoint(dir)?;
        let config = match layers {
            None => full,
            Some(n) => full
                .prefix(n)
                .ok_or_else(|| format!("layers {n}: outside 1..={}", full.layers))?,
        };
        Self::load(config, &mut ShardSource::open(dir)?, max_pos)
    }

    /// Opens a checkpoint directory and uploads model layers `first .. first + count` as a stack
    /// of their own (final norm and lm_head on top): a window, fed and read through
    /// [`Self::step_residual`] / [`Self::residual`] so a model larger than free device memory is
    /// checked one window at a time.
    pub fn open_window(
        dir: &Path,
        first: usize,
        count: usize,
        max_pos: usize,
    ) -> Result<Self, String> {
        let full = Qwen38Config::from_checkpoint(dir)?;
        let config = full
            .window(first, count)
            .ok_or_else(|| format!("window {first}+{count}: outside the {} layers", full.layers))?;
        Self::load_at(config, first, &mut ShardSource::open(dir)?, max_pos)
    }

    /// As [`Self::open_window`] with `heads` conversations resident at once (the 35B graph only,
    /// at most [`aien_omega_q38::q36::MAX_HEADS`]): one [`Self::stroke`] advances every head in
    /// one command list. Each head has its own DeltaNet state, attention cache and position.
    pub fn open_window_heads(
        dir: &Path,
        first: usize,
        count: usize,
        max_pos: usize,
        heads: usize,
    ) -> Result<Self, String> {
        let full = Qwen38Config::from_checkpoint(dir)?;
        let config = full
            .window(first, count)
            .ok_or_else(|| format!("window {first}+{count}: outside the {} layers", full.layers))?;
        Self::load_at_heads(config, first, &mut ShardSource::open(dir)?, max_pos, heads)
    }

    /// Uploads `config.layers` layers from `source`.
    pub fn load(
        config: Qwen38Config,
        source: &mut dyn TensorSource,
        max_pos: usize,
    ) -> Result<Self, String> {
        Self::load_at(config, 0, source, max_pos)
    }

    /// Uploads `config.layers` layers from `source`, reading model layer `first + l` for layer `l`.
    fn load_at(
        config: Qwen38Config,
        first: usize,
        source: &mut dyn TensorSource,
        max_pos: usize,
    ) -> Result<Self, String> {
        Self::load_at_heads(config, first, source, max_pos, 1)
    }

    /// [`Self::load_at`] with `heads` resident conversations (1 for the 27B dense graph).
    fn load_at_heads(
        config: Qwen38Config,
        first: usize,
        source: &mut dyn TensorSource,
        max_pos: usize,
        heads: usize,
    ) -> Result<Self, String> {
        if config.moe.is_some() {
            return Self::load_moe(config, first, source, max_pos, heads);
        }
        if heads != 1 {
            return Err(format!(
                "qwen38 omega: {heads} heads (the 27B dense graph strokes one conversation)"
            ));
        }
        use aien_omega_q38 as q;
        let c = &config;
        let shape_ok = c.hidden == q::HIDDEN
            && c.intermediate == q::INTER
            && c.heads * c.head_dim == q::MIXER_OUT
            && c.head_dim == q::HEAD_DIM
            && c.kv_heads * c.head_dim * 2 + 2 * c.heads * c.head_dim == q::ATT_MIX
            && c.conv_dim() == q::GDN_QKV
            && c.conv_dim() + c.lin_value_dim() == q::GDN_MIX
            && c.lin_v_heads == q::GDN_HEADS
            && c.lin_v_dim == q::GDN_HEAD_DIM
            && c.conv_kernel == q::GDN_CONV_KERNEL;
        if !shape_ok {
            return Err(
                "qwen38 omega: the checkpoint is not the Qwen3.8-27B shape the graph is built for"
                    .into(),
            );
        }
        if c.layers > q::MAX_LAYERS {
            return Err(format!(
                "qwen38 omega: {} layers (at most {})",
                c.layers,
                q::MAX_LAYERS
            ));
        }
        let prefix = text_prefix(source)?;
        let (v, h) = (c.vocab, c.hidden);
        let embed = read_bf16_words(source, &format!("{prefix}embed_tokens.weight"), &[v, h])?;
        let final_norm = fold(read_vec(source, &format!("{prefix}norm.weight"), h)?);
        let (lm_head, lm_head8) = match read_matrix(source, "lm_head.weight", v, h)? {
            Matrix::Bf16 { n, k, w } => {
                let head8 = if std::env::var("AIEN_Q38_FP8_HEAD").as_deref() == Ok("1") {
                    let (w8, s) = aien_omega_q38::quantize_fp8_bf16(n, k, &w, 16)
                        .map_err(gpu("lm_head fp8 quantise"))?;
                    Some(Fp8Tensor::upload(1, n, k, &w8, &s).map_err(gpu("lm_head fp8"))?)
                } else {
                    None
                };
                (Bf16Tensor::upload(n, k, &w).map_err(gpu("lm_head"))?, head8)
            }
            Matrix::Fp8 { .. } => return Err("lm_head: FP8 (the graph takes a bf16 head)".into()),
        };

        let n_attn = c.full_attention.iter().filter(|&&a| a).count();
        let n_gdn = c.layers - n_attn;
        let mut gdn = if n_gdn > 0 {
            Some(Gdn::alloc(n_gdn).map_err(gpu("gdn alloc"))?)
        } else {
            None
        };
        let mut attn = if n_attn > 0 {
            Some(
                Attn256Cache::alloc(n_attn, max_pos, c.rope_theta, c.rotary_dim)
                    .map_err(gpu("attn256 alloc"))?,
            )
        } else {
            None
        };
        let attn_mask = c
            .full_attention
            .iter()
            .enumerate()
            .fold(0u64, |m, (l, &a)| if a { m | 1 << l } else { m });

        let mut layers = Vec::with_capacity(c.layers);
        let mut norms = Vec::with_capacity(c.layers);
        let mut ab_mm = Vec::with_capacity(n_gdn);
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..c.layers {
            let p = format!("{prefix}layers.{}", first + l);
            let rm = |s: &mut dyn TensorSource, name: String, n, k| read_matrix(s, &name, n, k);
            let (mix, ab, o) = if c.full_attention[l] {
                let a = format!("{p}.self_attn");
                let mix = fuse_fp8(
                    &format!("{a} q|k|v"),
                    vec![
                        rm(
                            source,
                            format!("{a}.q_proj.weight"),
                            2 * c.heads * c.head_dim,
                            h,
                        )?,
                        rm(source, format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
                        rm(source, format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
                    ],
                )?;
                let o = upload_fp8(
                    &format!("{a}.o_proj"),
                    rm(
                        source,
                        format!("{a}.o_proj.weight"),
                        h,
                        c.heads * c.head_dim,
                    )?,
                )?;
                let q_norm = read_vec(source, &format!("{a}.q_norm.weight"), c.head_dim)?;
                let k_norm = read_vec(source, &format!("{a}.k_norm.weight"), c.head_dim)?;
                attn.as_mut()
                    .expect("an attention layer has a cache")
                    .set_norms(ai, &q_norm, &k_norm)
                    .map_err(gpu("attn256 set_norms"))?;
                ai += 1;
                (mix, None, o)
            } else {
                let a = format!("{p}.linear_attn");
                let mix = fuse_fp8(
                    &format!("{a} qkv|z"),
                    vec![
                        rm(source, format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
                        rm(
                            source,
                            format!("{a}.in_proj_z.weight"),
                            c.lin_value_dim(),
                            h,
                        )?,
                    ],
                )?;
                let ia = rm(source, format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?;
                let ib = rm(source, format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?;
                let (an, ak, aw) = ab_rows(&ia, &ib).map_err(|e| format!("{a}: {e}"))?;
                let ab = Bf16Tensor::upload(an, ak, &aw).map_err(gpu("in_proj_a|b"))?;
                ab_mm.push(
                    ResidentTensor::upload_bf16(ak, an, &ab_plain_transposed(an, ak, &aw))
                        .map_err(gpu("in_proj_a|b matmul"))?,
                );
                let o = upload_fp8(
                    &format!("{a}.out_proj"),
                    rm(source, format!("{a}.out_proj.weight"), h, c.lin_value_dim())?,
                )?;
                let conv: Vec<f32> = read_bf16_words(
                    source,
                    &format!("{a}.conv1d.weight"),
                    &[c.conv_dim(), 1, c.conv_kernel],
                )?
                .into_iter()
                .map(bf16_to_f32)
                .collect();
                let a_log = read_vec(source, &format!("{a}.A_log"), c.lin_v_heads)?;
                let dt_bias = read_vec(source, &format!("{a}.dt_bias"), c.lin_v_heads)?;
                let norm = read_vec(source, &format!("{a}.norm.weight"), c.lin_v_dim)?;
                gdn.as_mut()
                    .expect("a DeltaNet layer has state")
                    .set_layer(gi, &conv, &a_log, &dt_bias, &norm)
                    .map_err(gpu("gdn set_layer"))?;
                gi += 1;
                (mix, Some(ab), o)
            };
            let m = format!("{p}.mlp");
            let gate_up = fuse_fp8(
                &format!("{m} gate|up"),
                vec![
                    rm(source, format!("{m}.gate_proj.weight"), c.intermediate, h)?,
                    rm(source, format!("{m}.up_proj.weight"), c.intermediate, h)?,
                ],
            )?;
            let down = upload_fp8(
                &format!("{m}.down_proj"),
                rm(source, format!("{m}.down_proj.weight"), h, c.intermediate)?,
            )?;
            let in_norm = fold(read_vec(source, &format!("{p}.input_layernorm.weight"), h)?);
            let post_norm = fold(read_vec(
                source,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
            )?);
            layers.push(LayerTensors {
                mix,
                ab,
                o,
                gate_up,
                down,
            });
            norms.push((in_norm, post_norm));
        }

        let cfg = GraphConfig {
            layers: c.layers as u32,
            attn_mask,
            vocab: v as u32,
            eps: c.eps,
            eps_l2: EPS_L2,
            gdn: gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
            attn: attn
                .as_ref()
                .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
            lm_head: lm_head.as_raw(),
        };
        // SAFETY: gdn, attn and lm_head are live omega objects owned by the returned value, which
        // drops the graph before them (field order).
        let mut graph = unsafe { Q38Graph::create(&cfg) }.map_err(gpu("q38 graph create"))?;
        for (l, (t, (in_norm, post_norm))) in layers.iter().zip(&norms).enumerate() {
            let ab =
                t.ab.as_ref()
                    .map_or(std::ptr::null_mut(), Bf16Tensor::as_raw);
            // SAFETY: the tensors live in `layers`, owned by the returned value and dropped after
            // the graph.
            unsafe {
                graph.set_layer(
                    l,
                    t.mix.as_raw(),
                    ab,
                    t.o.as_raw(),
                    t.gate_up.as_raw(),
                    t.down.as_raw(),
                    in_norm,
                    post_norm,
                )
            }
            .map_err(gpu("q38 graph set_layer"))?;
        }
        graph
            .set_final_norm(&final_norm)
            .map_err(gpu("q38 graph set_final_norm"))?;
        if let Some(h8) = &lm_head8 {
            // SAFETY: lm_head8 is owned by the returned value and dropped after the graph (field order).
            unsafe { graph.set_lm_head_fp8(h8.as_raw()) }
                .map_err(gpu("q38 graph set_lm_head_fp8"))?;
        }
        Ok(Self {
            graph: Graph::Dense(graph),
            router: None,
            layers,
            lm_head,
            lm_head8,
            head_screen: None,
            gdn,
            attn,
            extra_heads: Vec::new(),
            config,
            embed,
            pos: 0,
            max_pos,
            norms,
            final_norm,
            ab_mm,
            pf: None,
            row_stroke: None,
            rows: Vec::new(),
            row_src: None,
            row_steps: 0,
        })
    }

    /// Bytes resident on the device for the weights (tensors as uploaded, not the KV cache).
    pub fn weight_bytes(&self) -> usize {
        let fp8 = |t: &Fp8Tensor| t.resident_bytes();
        self.layers
            .iter()
            .map(|t| {
                fp8(&t.mix)
                    + fp8(&t.o)
                    + fp8(&t.gate_up)
                    + fp8(&t.down)
                    + t.ab.as_ref().map_or(0, Bf16Tensor::resident_bytes)
            })
            .sum::<usize>()
            + self.lm_head.resident_bytes()
    }

    /// Bytes the KV cache holds on the device (every attention layer at `max_pos`).
    pub fn kv_bytes(&self) -> usize {
        self.attn
            .as_ref()
            .map_or(0, |a| a.layers() * Attn256Cache::layer_bytes(a.max_pos()))
    }

    /// A new sequence: DeltaNet state to zero, position 0 (attention rows are overwritten as the
    /// new sequence reaches them).
    pub fn reset(&mut self) -> Result<(), String> {
        for h in 0..self.heads() {
            self.reset_head(h)?;
        }
        Ok(())
    }

    /// Feeds `token` at the next position: the greedy next token and what the step cost.
    pub fn step(&mut self, token: u32) -> Result<(u32, GraphStats), String> {
        let (t, h) = (token as usize, self.config.hidden);
        if t >= self.config.vocab {
            return Err(format!(
                "step: token {t} outside the vocab {}",
                self.config.vocab
            ));
        }
        if self.pos >= self.max_pos {
            return Err(format!(
                "step: the KV cache holds {} positions",
                self.max_pos
            ));
        }
        let x: Vec<f32> = self.embed[t * h..(t + 1) * h]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();
        let out = self.graph.token(&x, self.pos).map_err(gpu("q38 token"))?;
        self.pos += 1;
        Ok(out)
    }

    /// Feeds the residual stream `x` (`[hidden]`, what enters this stack's first layer: the
    /// previous window's [`Self::residual`]) at the next position instead of a token's embedding.
    pub fn step_residual(&mut self, x: &[f32]) -> Result<(u32, GraphStats), String> {
        if x.len() != self.config.hidden {
            return Err(format!(
                "step_residual: {} values (hidden {})",
                x.len(),
                self.config.hidden
            ));
        }
        if self.pos >= self.max_pos {
            return Err(format!(
                "step: the KV cache holds {} positions",
                self.max_pos
            ));
        }
        let out = self.graph.token(x, self.pos).map_err(gpu("q38 token"))?;
        self.pos += 1;
        Ok(out)
    }

    /// The last step's residual stream after this stack's last layer (`[hidden]`, before the
    /// final norm).
    pub fn residual(&mut self) -> Result<Vec<f32>, String> {
        let mut x = vec![0f32; self.config.hidden];
        self.graph
            .read(Some(&mut x), None, None)
            .map_err(gpu("q38 read residual"))?;
        Ok(x)
    }

    /// The last step's logits (`[vocab]`).
    pub fn logits(&mut self) -> Result<Vec<f32>, String> {
        let mut l = vec![0f32; self.config.vocab];
        self.graph
            .read(None, None, Some(&mut l))
            .map_err(gpu("q38 read logits"))?;
        Ok(l)
    }

    /// Positions the KV cache holds (the longest sequence).
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Conversations resident at once (head slots of the graph).
    pub fn heads(&self) -> usize {
        self.extra_heads.len() + 1
    }

    /// Tokens head `head` holds (its next position).
    pub fn head_pos(&self, head: usize) -> Result<usize, String> {
        match head {
            0 => Ok(self.pos),
            h => self
                .extra_heads
                .get(h - 1)
                .map(|s| s.pos)
                .ok_or_else(|| format!("head {h} of {}", self.heads())),
        }
    }

    pub(super) fn head_pos_mut(&mut self, head: usize) -> Result<&mut usize, String> {
        let heads = self.heads();
        match head {
            0 => Ok(&mut self.pos),
            h => self
                .extra_heads
                .get_mut(h - 1)
                .map(|s| &mut s.pos)
                .ok_or_else(|| format!("head {h} of {heads}")),
        }
    }

    /// One stroke: feeds `tokens[h]` to head `h` (the first `tokens.len()` heads, at most
    /// [`Self::heads`]) at each head's next position in ONE command list, one submit. The greedy
    /// next token of every head and what the stroke cost (the chip time is the whole stroke).
    pub fn stroke(&mut self, tokens: &[u32]) -> Result<(Vec<u32>, GraphStats), String> {
        let (heads, h) = (tokens.len(), self.config.hidden);
        if heads == 0 || heads > self.heads() {
            return Err(format!("stroke: {heads} tokens on {} heads", self.heads()));
        }
        let mut x = Vec::with_capacity(heads * h);
        let mut pos = Vec::with_capacity(heads);
        for (head, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            if t >= self.config.vocab {
                return Err(format!(
                    "stroke: head {head} token {t} outside the vocab {}",
                    self.config.vocab
                ));
            }
            let p = self.head_pos(head)?;
            if p >= self.max_pos {
                return Err(format!(
                    "stroke: head {head} at {p}, the KV cache holds {} positions",
                    self.max_pos
                ));
            }
            x.extend(
                self.embed[t * h..(t + 1) * h]
                    .iter()
                    .map(|&b| bf16_to_f32(b)),
            );
            pos.push(p);
        }
        let out = self.graph.stroke(&x, &pos)?;
        for head in 0..heads {
            *self.head_pos_mut(head)? += 1;
        }
        Ok(out)
    }

    /// The last stroke's residual stream of head `head` (as [`Self::residual`]).
    pub fn residual_head(&mut self, head: usize) -> Result<Vec<f32>, String> {
        let mut x = vec![0f32; self.config.hidden];
        self.graph.read_head(head, Some(&mut x), None, None)?;
        Ok(x)
    }

    /// The last stroke's logits of head `head` (as [`Self::logits`]).
    pub fn logits_head(&mut self, head: usize) -> Result<Vec<f32>, String> {
        let mut l = vec![0f32; self.config.vocab];
        self.graph.read_head(head, None, None, Some(&mut l))?;
        Ok(l)
    }

    /// Clears head `head`'s conversation (DeltaNet state and position; the attention cache is
    /// overwritten from position 0).
    pub fn reset_head(&mut self, head: usize) -> Result<(), String> {
        let gdn = match head {
            0 => self.gdn.as_mut(),
            h => self
                .extra_heads
                .get_mut(h - 1)
                .ok_or_else(|| format!("head {h}"))?
                .gdn
                .as_mut(),
        };
        if let Some(g) = gdn {
            g.reset().map_err(gpu("gdn reset"))?;
        }
        *self.head_pos_mut(head)? = 0;
        Ok(())
    }

    fn take_head(&mut self, head: usize) -> HeadState {
        match head {
            0 => HeadState {
                gdn: self.gdn.take(),
                attn: self.attn.take(),
                pos: std::mem::take(&mut self.pos),
            },
            h => std::mem::take(&mut self.extra_heads[h - 1]),
        }
    }

    fn put_head(&mut self, head: usize, s: HeadState) {
        match head {
            0 => {
                self.gdn = s.gdn;
                self.attn = s.attn;
                self.pos = s.pos;
            }
            h => self.extra_heads[h - 1] = s,
        }
    }

    /// The batch (L312 v2): each layer recorded for every conversation at once, one weight read per
    /// layer for all of them. 35B graph only; pins the GEMM kernels on the session.
    pub fn set_batch(&mut self, on: bool) -> Result<(), String> {
        match &mut self.graph {
            Graph::Moe(g) => g
                .set_batch(on)
                .map_err(|rc| format!("q36 set_batch: rc {rc}")),
            Graph::Dense(_) => Err("set_batch: the 27B dense graph has no batch".into()),
        }
    }
    /// Whether the 35B graph records in the batch (also `AIEN_OMEGA_Q36_BATCH=1` at open).
    pub fn batch(&self) -> bool {
        match &self.graph {
            Graph::Moe(g) => g.batch(),
            Graph::Dense(_) => false,
        }
    }
    /// Chip timestamps around every launch of the next strokes (35B graph only): a timeline per token.
    pub fn set_stamps(&mut self, on: bool) -> Result<(), String> {
        match &mut self.graph {
            Graph::Moe(g) => {
                g.set_stamps(on);
                Ok(())
            }
            Graph::Dense(_) => Err("set_stamps: the 27B dense graph has no stroke timeline".into()),
        }
    }
    /// The last stroke's timeline: (launch kind name, chip ns) per launch in launch order; needs
    /// [`Self::set_stamps`] before that stroke.
    pub fn timeline(&self) -> Result<Vec<(String, u64)>, String> {
        match &self.graph {
            Graph::Moe(g) => g
                .timeline()
                .map(|v| {
                    v.into_iter()
                        .map(|(k, ns)| (Q36Graph::kind_name(k), ns))
                        .collect()
                })
                .map_err(|rc| format!("q36 timeline: rc {rc}")),
            Graph::Dense(_) => Err("timeline: the 27B dense graph has no stroke timeline".into()),
        }
    }

    /// Head swap: slots `a` and `b` exchange conversations (state objects and positions); the
    /// graph is re-pointed, nothing is copied. After it, the stroke feeding slot `a` continues
    /// the conversation that was in `b`.
    pub fn swap_heads(&mut self, a: usize, b: usize) -> Result<(), String> {
        let heads = self.heads();
        if a >= heads || b >= heads {
            return Err(format!("swap_heads: {a} and {b} of {heads} heads"));
        }
        if a == b {
            return Ok(());
        }
        if matches!(self.graph, Graph::Dense(_)) {
            return Err("swap_heads: the 27B dense graph has one head".into());
        }
        let (sa, sb) = (self.take_head(a), self.take_head(b));
        let raw = |s: &HeadState| {
            (
                s.gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                s.attn
                    .as_ref()
                    .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
            )
        };
        let ((ga, aa), (gb, ab)) = (raw(&sa), raw(&sb));
        self.put_head(a, sb);
        self.put_head(b, sa);
        if let Graph::Moe(g) = &mut self.graph {
            // SAFETY: both states are owned by self (just put back) and outlive the graph (field order).
            unsafe {
                g.set_head_state(a, gb, ab)
                    .map_err(gpu("q36 set_head_state"))?;
                g.set_head_state(b, ga, aa)
                    .map_err(gpu("q36 set_head_state"))?;
            }
        }
        Ok(())
    }

    /// Copies every DeltaNet layer's conv tail and state and the position into `snap` (host
    /// memory, reused between calls): [`Self::restore`] continues the sequence from here. The
    /// attention KV needs no copy: rows past the restored position are rewritten before they
    /// are read.
    pub fn snapshot(&mut self, snap: &mut Qwen38Snapshot) -> Result<(), String> {
        let Some(g) = self.gdn.as_mut() else {
            snap.pos = self.pos;
            return Ok(());
        };
        let n = g.layers();
        let (tf, sf) = (g.tail_floats(), g.state_floats());
        snap.tails.resize(n * tf, 0.0);
        snap.states.resize(n * sf, 0.0);
        for (l, (t, st)) in snap
            .tails
            .chunks_exact_mut(tf)
            .zip(snap.states.chunks_exact_mut(sf))
            .enumerate()
        {
            g.get_state(l, t, st).map_err(gpu("gdn get_state"))?;
        }
        snap.pos = self.pos;
        Ok(())
    }

    /// Puts a [`Self::snapshot`] back: the next token goes at `snap.pos`.
    pub fn restore(&mut self, snap: &Qwen38Snapshot) -> Result<(), String> {
        if snap.pos > self.max_pos {
            return Err(format!("restore: position {} past the cache", snap.pos));
        }
        if let Some(g) = self.gdn.as_mut() {
            let (tf, sf) = (g.tail_floats(), g.state_floats());
            if snap.tails.len() != g.layers() * tf || snap.states.len() != g.layers() * sf {
                return Err("restore: the snapshot is not of this model".into());
            }
            for (l, (t, st)) in snap
                .tails
                .chunks_exact(tf)
                .zip(snap.states.chunks_exact(sf))
                .enumerate()
            {
                g.set_state(l, t, st).map_err(gpu("gdn set_state"))?;
            }
        }
        self.pos = snap.pos;
        Ok(())
    }

    /// The graph, for traces and timelines.
    pub fn graph_mut(&mut self) -> Option<&mut Q38Graph> {
        match &mut self.graph {
            Graph::Dense(g) => Some(g),
            Graph::Moe(_) => None,
        }
    }

    /// Greedy tokens may use the exact two-pass head (35B, when loaded with it): the token is the
    /// full head's, but the logits read back are -inf off the candidates. Sampled requests turn
    /// it off and read every logit.
    pub fn set_exact_head_on(&mut self, on: bool) {
        if let Graph::Moe(g) = &mut self.graph {
            g.set_exact_head_on(on);
        }
    }

    /// True when the next token runs the exact two-pass head.
    pub fn exact_head_active(&self) -> bool {
        matches!(&self.graph, Graph::Moe(g) if g.exact_head_active())
    }

    /// The 35B graph, for timelines.
    pub fn q36_graph_mut(&mut self) -> Option<&mut Q36Graph> {
        match &mut self.graph {
            Graph::Moe(g) => Some(g),
            Graph::Dense(_) => None,
        }
    }

    /// True for the sparse (Qwen3.6-35B-A3B) model.
    pub fn is_moe(&self) -> bool {
        matches!(self.graph, Graph::Moe(_))
    }
}

/// Free memory kept for everything else on the box (the resident server shares this pool:
/// 2026-10-18 16:43:58Z a 32-layer window pushed the kernel into the OOM killer, which took the
/// server). The gate tools upload a window only when MemAvailable covers it plus this margin.
pub const KEEP_FREE: u64 = 12 << 30;

/// Device bytes a window takes: its FP8 weights (1 byte each; the 128x128 block scales add 1/4096),
/// the bf16 a|b rows of its DeltaNet layers, and the bf16 lm_head; from the config, not measured.
pub fn window_bytes(c: &Qwen38Config, first: usize, count: usize) -> u64 {
    let h = c.hidden as u64;
    // Qwen3.6-35B-A3B (sc L306): every layer holds 257 FP8 experts (256 routed + the shared one)
    // and a bf16 router of 257 rows.
    let mlp = match &c.moe {
        Some(m) => (m.experts as u64 + 1) * (3 * m.expert_inter as u64 * h + 2 * h),
        None => 3 * c.intermediate as u64 * h,
    };
    let o = h * (c.heads * c.head_dim) as u64;
    let layers: u64 = c.full_attention[first..first + count]
        .iter()
        .map(|&attn| {
            if attn {
                (2 * c.heads * c.head_dim + 2 * c.kv_dim()) as u64 * h + o + mlp
            } else {
                (c.conv_dim() + c.lin_value_dim()) as u64 * h
                    + o
                    + mlp
                    + 2 * 2 * c.lin_v_heads as u64 * h
            }
        })
        .sum();
    layers + layers / 4096 + 2 * c.vocab as u64 * h
}

/// MemAvailable from /proc/meminfo, in bytes.
pub fn mem_available() -> Result<u64, String> {
    let m = std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    m.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb << 10)
        .ok_or_else(|| "/proc/meminfo: no MemAvailable".into())
}
