//! Speculative decoding for Qwen3.8-27B with the checkpoint's own MTP head (sc #679 task A).
//!
//! The 27B reads 26.9 GB of weights per token; at 273 GB/s that caps plain decode at 10.15
//! tok/s. A round here drafts `k` tokens with the one-layer MTP head (`mtp.safetensors`), then
//! checks the last token and all `k` drafts in ONE pass of the full model (the weights stream
//! once for the `k + 1` rows), and keeps the longest prefix of drafts the model itself would
//! have produced, plus the model's own next token.
//!
//! Exactness contract (the own30b/spec-exact rule): the check pass, [`Qwen38Omega::exact_rows`],
//! computes every row with the same chip calls whatever the number of rows, so row `r` of an
//! `m`-row pass is bit for bit the single-row pass over the same history:
//!
//! | step | call | per row independent of `m` because |
//! |---|---|---|
//! | norms, residual adds, SwiGLU, output gate, gathers | host f32, row by row | row loops |
//! | mixer, o_proj, gate\|up, down | `omega_gpu_fp8gemm_f32` | header: bit-identical to the GEMV once per row |
//! | DeltaNet a\|b | bf16 GEMV per row | one call per row |
//! | DeltaNet conv + delta rule + gated norm | `omega_gpu_gdn_step` per row, in order | the decode kernel |
//! | attention | `omega_gpu_decode_attn256` per row, in order | the decode kernel |
//! | final norm, lm_head, argmax (strict, first maximum) | host norm, bf16 GEMV per row | one call per row |
//!
//! So greedy output with speculation on equals greedy output of the same path one token at a
//! time (`k = 0`), whatever the drafts were: a draft only decides how many rows one pass checks.
//! The decode graph ([`Qwen38Omega::step`]) runs its glue on the chip with its own maths, so
//! its tokens are a separate measurement, never assumed equal.
//!
//! DeltaNet state cannot be cut back like the KV cache (attention rows past the position are
//! rewritten before they are read). Before a pass the tails and states are copied to the host;
//! when a draft is refused they are put back and only the accepted rows' `gdn_step` calls are
//! replayed from the inputs the pass kept (no weights are read again).
//!
//! The MTP head (Qwen3.5 layout): `x = fc([norm_e(embed(t_next)) ; norm_h(h_t)])`, one
//! full-attention layer with its own KV cache, `mtp.norm`, and the model's lm_head. Which half of
//! the `fc` input comes first, whether `h_t` is the residual stream or the final-normed hidden,
//! and the RoPE position are [`MtpOptions`]; they only change how many drafts are accepted.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use aien_omega_bf16::Bf16Tensor;
use aien_omega_q38 as q;
use aien_omega_q38::{Attn256Cache, Gdn};

use super::{fold, fuse_fp8, gpu, upload_fp8, LayerTensors, Qwen38Omega, Qwen38Snapshot, EPS_L2};
use crate::qwen38::{read_matrix, read_vec, Matrix};
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::{ShardSource, TensorSource};
use crate::tensor::rmsnorm;

/// The dense GEMM shape of the rows path: 2 rows per pass, 192 CTAs. Chip sweep 2026-10-19
/// (omega evidence/fp8gemm-mt-sweep-20261019T0327Z): gate|up 34816 x 5120 at 2 rows 0.98 ms
/// (one row by GEMV 0.87 ms), at 5 rows 3.04 ms against 5.19 ms (4 rows per pass) and 5.12 ms
/// (the API's 8). Every output is bit-identical to the per-row GEMV at every shape, so this is a
/// speed choice only.
const VERIFY_GEMM_MT: u32 = 2;
const VERIFY_GEMM_CTAS: u32 = 192;
static VERIFY_GEMM_SHAPE: std::sync::Once = std::sync::Once::new();
/// Set for the duration of a pass that runs its dense projections (mixer, o_proj, gate|up, down)
/// on omega's tensor-core verify GEMM (`omega_gpu_tv_f32`, branch cascade/tree-verify-gemm)
/// instead of the row-pair GEMM: the batch door (speed5/batching) and the tree verify
/// (cascade/tree-verify-wire) both route through [`gemm`]. Read by [`gemm`] only.
pub(super) static TREE_GEMM: AtomicBool = AtomicBool::new(false);

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

/// Index of the largest logit, the first one on a tie (the decode graph's ARGMAX rule).
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Wall time of the exact rows path by kind (every chip call is synchronous, so a call's wall
/// time includes its launch, wait and copies).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowStats {
    pub rows: u64,
    pub passes: u64,
    pub gemm_ns: u64,
    pub gemm_calls: u64,
    /// Chip time of the tree-verify GEMM launches alone (`TvInfo::chip_ns`); 0 on the row-pair path.
    pub gemm_chip_ns: u64,
    pub ab_ns: u64,
    pub gdn_ns: u64,
    pub gdn_calls: u64,
    pub attn_ns: u64,
    pub attn_calls: u64,
    pub host_ns: u64,
    pub head_ns: u64,
    pub head_calls: u64,
}

/// One DeltaNet layer's inputs for the rows of a pass (`[m][10240]`, `[m][6144]`, `[m][96]`),
/// kept so a refused draft costs a `gdn_step` replay instead of a second pass.
#[derive(Debug, Clone, Default)]
pub struct GdnRows {
    qkv: Vec<f32>,
    z: Vec<f32>,
    ab: Vec<f32>,
}

/// The mixer of one layer of a rows pass (which kernel, which layer index, its weights).
pub(super) enum Mixer<'a> {
    Attn {
        idx: usize,
        gather: &'a [u32],
    },
    Gdn {
        idx: usize,
        ab: &'a Bf16Tensor,
        keep: Option<&'a mut GdnRows>,
    },
}

/// Where row `r` of a rows pass keeps its state: one sequence's rows share a cache and a
/// DeltaNet state at consecutive positions ([`OneSequence`]); independent conversations each
/// bring their own ([`super::batch::Streams`]). Either way every row's mixer is the same
/// `attn256 decode` or `gdn_step` call on its own state, so a row's maths do not depend on
/// how many rows ride the pass or whose they are.
pub(super) trait RowTargets {
    /// The attention cache and position of row `r`.
    fn attn(&mut self, r: usize) -> (&mut Attn256Cache, usize);
    /// The DeltaNet state of row `r`.
    fn gdn(&mut self, r: usize) -> &mut Gdn;
}

/// Rows of one sequence at positions `pos0 ..`.
pub(super) struct OneSequence<'a> {
    pub attn: Option<&'a mut Attn256Cache>,
    pub gdn: Option<&'a mut Gdn>,
    pub pos0: usize,
}

impl RowTargets for OneSequence<'_> {
    fn attn(&mut self, r: usize) -> (&mut Attn256Cache, usize) {
        (
            self.attn
                .as_deref_mut()
                .expect("an attention layer has a cache"),
            self.pos0 + r,
        )
    }
    fn gdn(&mut self, _r: usize) -> &mut Gdn {
        self.gdn.as_deref_mut().expect("a DeltaNet layer has state")
    }
}

/// Row-table GEMM over one dense FP8 tensor: `y[m][n] = W . x[m][k]`.
pub(super) fn gemm(
    w: &aien_omega_fp8::Fp8Tensor,
    x: &[f32],
    y: &mut [f32],
    m: usize,
    what: &str,
    st: &mut RowStats,
) -> Result<(), String> {
    if TREE_GEMM.load(Ordering::Relaxed) {
        let t = Instant::now();
        let info = w
            .tv_f32(m, x, y)
            .map_err(|e| format!("{what} tree-verify gemm x{m}: {e}"))?;
        st.gemm_ns += ns(t);
        st.gemm_chip_ns += info.chip_ns;
        st.gemm_calls += 1;
        return Ok(());
    }
    VERIFY_GEMM_SHAPE.call_once(|| {
        aien_omega_fp8::gemm_set_shape(VERIFY_GEMM_MT, 0);
        aien_omega_fp8::gemm_set_cta_budget(VERIFY_GEMM_CTAS);
    });
    let ids = vec![0u32; m];
    let rows: Vec<u32> = (0..m as u32).collect();
    let t = Instant::now();
    w.gemm_f32(&ids, &rows, x, y)
        .map_err(|e| format!("{what} gemm x{m}: {e}"))?;
    st.gemm_ns += ns(t);
    st.gemm_calls += 1;
    Ok(())
}

/// Host threads for the per-row SiLU of a rows pass (`AIEN_Q38_HOST_THREADS`, default 1 = on the caller's
/// thread). Each row is computed by the same code whichever thread runs it, so the bits do not change.
fn host_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("AIEN_Q38_HOST_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize)
            .clamp(1, 64)
    })
}

/// `f(r, row)` for every row of `out` (rows of `width`), split over [`host_threads`] threads when
/// there are at least 8 rows (below that the thread start costs more than it saves).
fn par_rows(out: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    par_rows_n(out, width, host_threads(), f)
}

fn par_rows_n(out: &mut [f32], width: usize, threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let m = out.len() / width;
    let threads = threads.min(m);
    if threads <= 1 || m < 8 {
        for (r, o) in out.chunks_exact_mut(width).enumerate() {
            f(r, o);
        }
        return;
    }
    let per = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, o) in chunk.chunks_exact_mut(width).enumerate() {
                    f(c * per + j, o);
                }
            });
        }
    });
}

/// `m` rows of the residual stream `x` (`[m][hidden]`, positions `pos0 ..`) through one layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_rows(
    lt: &LayerTensors,
    in_norm: &[f32],
    post_norm: &[f32],
    mixer: Mixer<'_>,
    targets: &mut dyn RowTargets,
    m: usize,
    eps: f32,
    x: &mut [f32],
    st: &mut RowStats,
) -> Result<(), String> {
    let (h, mo, inter) = (q::HIDDEN, q::MIXER_OUT, q::INTER);
    let t = Instant::now();
    let mut xn = vec![0f32; m * h];
    for (o, r) in xn.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
        rmsnorm(r, in_norm, eps, o);
    }
    st.host_ns += ns(t);
    let mix_n = lt.mix.n();
    let mut mix = vec![0f32; m * mix_n];
    gemm(&lt.mix, &xn, &mut mix, m, "mixer", st)?;
    let mut att = vec![0f32; m * mo];
    match mixer {
        Mixer::Attn { idx, gather } => {
            let (gq, gg) = gather.split_at(q::ATT_QKV);
            let (mut qkv, mut gate) = (vec![0f32; q::ATT_QKV], vec![0f32; mo]);
            for (r, out) in att.chunks_exact_mut(mo).enumerate() {
                let t = Instant::now();
                let row = &mix[r * mix_n..(r + 1) * mix_n];
                for (o, &s) in qkv.iter_mut().zip(gq) {
                    *o = row[s as usize];
                }
                for (o, &s) in gate.iter_mut().zip(gg) {
                    *o = row[s as usize];
                }
                st.host_ns += ns(t);
                let t = Instant::now();
                let (cache, pos) = targets.attn(r);
                cache
                    .decode(idx, pos, &qkv, eps, out)
                    .map_err(|e| format!("attention row {r}: {e}"))?;
                st.attn_ns += ns(t);
                st.attn_calls += 1;
                let t = Instant::now();
                for (v, &g) in out.iter_mut().zip(&gate) {
                    *v /= 1.0 + (-g).exp();
                }
                st.host_ns += ns(t);
            }
        }
        Mixer::Gdn { idx, ab, mut keep } => {
            let t = Instant::now();
            let mut abr = vec![0f32; m * q::AB];
            let mut xs = vec![0f32; h];
            for (r, out) in abr.chunks_exact_mut(q::AB).enumerate() {
                // The graph's a|b tensor is uploaded * 2^-64 and fed the 2^64-scaled row: both
                // scalings are exact exponent shifts.
                for (o, &v) in xs.iter_mut().zip(&xn[r * h..(r + 1) * h]) {
                    *o = v * q::ROW_SCALE;
                }
                ab.gemv_f32(&xs, out)
                    .map_err(|e| format!("a|b row {r}: {e}"))?;
            }
            st.ab_ns += ns(t);
            if let Some(k) = keep.as_deref_mut() {
                k.qkv.clear();
                k.z.clear();
                k.ab.clone_from(&abr);
            }
            for (r, out) in att.chunks_exact_mut(mo).enumerate() {
                let row = &mix[r * mix_n..(r + 1) * mix_n];
                let (qkv, z) = row[..q::GDN_MIX].split_at(q::GDN_QKV);
                let t = Instant::now();
                targets
                    .gdn(r)
                    .step(
                        idx,
                        qkv,
                        z,
                        &abr[r * q::AB..(r + 1) * q::AB],
                        EPS_L2,
                        eps,
                        out,
                    )
                    .map_err(|e| format!("gdn row {r}: {e}"))?;
                st.gdn_ns += ns(t);
                st.gdn_calls += 1;
                if let Some(k) = keep.as_deref_mut() {
                    k.qkv.extend_from_slice(qkv);
                    k.z.extend_from_slice(z);
                }
            }
        }
    }
    let mut o = vec![0f32; m * h];
    gemm(&lt.o, &att, &mut o, m, "o_proj", st)?;
    let t = Instant::now();
    for (a, b) in x.iter_mut().zip(&o) {
        *a += b;
    }
    for (o, r) in xn.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
        rmsnorm(r, post_norm, eps, o);
    }
    st.host_ns += ns(t);
    let mut gu = vec![0f32; m * 2 * inter];
    gemm(&lt.gate_up, &xn, &mut gu, m, "gate|up", st)?;
    let t = Instant::now();
    let mut act = vec![0f32; m * inter];
    par_rows(&mut act, inter, |r, a| {
        let (g, u) = gu[r * 2 * inter..(r + 1) * 2 * inter].split_at(inter);
        for ((a, &g), &u) in a.iter_mut().zip(g).zip(u) {
            *a = g / (1.0 + (-g).exp()) * u;
        }
    });
    st.host_ns += ns(t);
    gemm(&lt.down, &act, &mut o, m, "down", st)?;
    let t = Instant::now();
    for (a, b) in x.iter_mut().zip(&o) {
        *a += b;
    }
    st.host_ns += ns(t);
    Ok(())
}

impl Qwen38Omega {
    /// `tokens` at positions `pos ..` through every layer on the exact rows path (module doc):
    /// the residual stream after the last layer, `[m][hidden]`. Advances `pos` and the DeltaNet
    /// state by every row. `keep`, when given, receives every DeltaNet layer's row inputs for
    /// [`Self::replay_gdn`].
    pub fn exact_rows(
        &mut self,
        tokens: &[u32],
        mut keep: Option<&mut Vec<GdnRows>>,
        st: &mut RowStats,
    ) -> Result<Vec<f32>, String> {
        let m = tokens.len();
        let (h, v) = (self.config.hidden, self.config.vocab);
        if m == 0 || m > aien_omega_fp8::GEMM_MAX_ROWS {
            return Err(format!(
                "exact_rows: {m} rows (1..={})",
                aien_omega_fp8::GEMM_MAX_ROWS
            ));
        }
        if self.pos + m > self.max_pos {
            return Err(format!(
                "exact_rows: positions {} + {m} > KV cache of {}",
                self.pos, self.max_pos
            ));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= v) {
            return Err(format!("exact_rows: token {t} outside the vocab {v}"));
        }
        let t = Instant::now();
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&tk| {
                let r = tk as usize * h;
                self.embed[r..r + h].iter().map(|&b| bf16_to_f32(b))
            })
            .collect();
        let gather = q::gather_table(self.config.rotary_dim);
        st.host_ns += ns(t);
        let n_gdn = self.ab_mm.len();
        if let Some(k) = keep.as_deref_mut() {
            k.resize_with(n_gdn, GdnRows::default);
        }
        let (pos0, eps) = (self.pos, self.config.eps);
        let mut targets = OneSequence {
            attn: self.attn.as_mut(),
            gdn: self.gdn.as_mut(),
            pos0,
        };
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..self.config.layers {
            let lt = &self.layers[l];
            let (in_norm, post_norm) = &self.norms[l];
            let mixer = if self.config.full_attention[l] {
                ai += 1;
                Mixer::Attn {
                    idx: ai - 1,
                    gather: &gather,
                }
            } else {
                gi += 1;
                Mixer::Gdn {
                    idx: gi - 1,
                    ab: lt.ab.as_ref().expect("a DeltaNet layer has a|b"),
                    keep: keep.as_deref_mut().map(|k| &mut k[gi - 1]),
                }
            };
            layer_rows(
                lt,
                in_norm,
                post_norm,
                mixer,
                &mut targets,
                m,
                eps,
                &mut x,
                st,
            )
            .map_err(|e| format!("layer {l}: {e}"))?;
        }
        self.pos += m;
        st.rows += m as u64;
        st.passes += 1;
        Ok(x)
    }

    /// The greedy token of one residual row: host final norm, bf16 lm_head GEMV, argmax.
    pub fn head_token(&mut self, residual: &[f32], st: &mut RowStats) -> Result<u32, String> {
        let t = Instant::now();
        let logits = self.head_logits(residual)?;
        let tok = argmax(&logits);
        st.head_ns += ns(t);
        st.head_calls += 1;
        Ok(tok)
    }

    /// Replays rows `0 .. rows` of the kept DeltaNet inputs of a pass (after [`Self::restore`]
    /// put the state back to where the pass began): the state and the position as if only those
    /// rows had run.
    pub fn replay_gdn(
        &mut self,
        keep: &[GdnRows],
        rows: usize,
        st: &mut RowStats,
    ) -> Result<(), String> {
        let Some(g) = self.gdn.as_mut() else {
            self.pos += rows;
            return Ok(());
        };
        let eps = self.config.eps;
        let mut out = vec![0f32; q::MIXER_OUT];
        for (gi, k) in keep.iter().enumerate() {
            for r in 0..rows {
                let t = Instant::now();
                g.step(
                    gi,
                    &k.qkv[r * q::GDN_QKV..(r + 1) * q::GDN_QKV],
                    &k.z[r * q::MIXER_OUT..(r + 1) * q::MIXER_OUT],
                    &k.ab[r * q::AB..(r + 1) * q::AB],
                    EPS_L2,
                    eps,
                    &mut out,
                )
                .map_err(|e| format!("gdn replay layer {gi} row {r}: {e}"))?;
                st.gdn_ns += ns(t);
                st.gdn_calls += 1;
            }
        }
        self.pos += rows;
        Ok(())
    }

    /// The host embedding row of `token`, widened.
    pub(super) fn embed_row(&self, token: u32) -> Vec<f32> {
        let h = self.config.hidden;
        let r = token as usize * h;
        self.embed[r..r + h]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }
}

/// Which half of the MTP `fc` input comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FcOrder {
    /// `[norm_e(embed) ; norm_h(hidden)]`
    EmbedFirst,
    /// `[norm_h(hidden) ; norm_e(embed)]`
    HiddenFirst,
}

/// The MTP head's free choices (draft quality only, never output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpOptions {
    pub order: FcOrder,
    /// Feed the final-normed hidden (true) or the residual stream (false) as `h_t`.
    pub normed_hidden: bool,
    /// RoPE / KV position of the MTP step that pairs `h_t` with `embed(t + 1)`: `t + offset`.
    pub pos_offset: usize,
}

impl Default for MtpOptions {
    fn default() -> Self {
        Self {
            order: FcOrder::EmbedFirst,
            normed_hidden: false,
            pos_offset: 0,
        }
    }
}

/// The checkpoint's MTP head on the device.
pub struct Mtp {
    /// `mtp.fc` `[5120][10240]` as its two column halves `[5120][5120]` (the bf16 GEMV takes k up to
    /// 5120 here: k 10240 needs more registers than the file holds): `fc . [a ; b] = fc0 . a + fc1 . b`.
    fc: [Bf16Tensor; 2],
    norm_e: Vec<f32>,
    norm_h: Vec<f32>,
    layer: LayerTensors,
    norms: (Vec<f32>, Vec<f32>),
    final_norm: Vec<f32>,
    attn: Attn256Cache,
    gather: Vec<u32>,
    pub opts: MtpOptions,
    /// Mean of the raw norm weights (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `mtp.norm`),
    /// printed by the tools: near 0 says the weights are zero-centred like the main model's.
    pub raw_norm_means: [f32; 3],
    eps: f32,
    /// The last MTP output (residual stream after its layer) and the position of the next step.
    last: Vec<f32>,
    next_pos: usize,
}

impl std::fmt::Debug for Mtp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mtp {{ opts: {:?}, next_pos: {} }}",
            self.opts, self.next_pos
        )
    }
}

impl Mtp {
    /// Loads `mtp.*` from the checkpoint `dir` with a KV cache of `max_pos` positions.
    pub fn open(dir: &Path, model: &Qwen38Omega, opts: MtpOptions) -> Result<Self, String> {
        let mut shards = ShardSource::open(dir)?;
        let src: &mut dyn TensorSource = &mut shards;
        let c = &model.config;
        let h = c.hidden;
        let rm = |s: &mut dyn TensorSource, name: &str, n, k| read_matrix(s, name, n, k);
        let fc = match rm(src, "mtp.fc.weight", h, 2 * h)? {
            Matrix::Bf16 { n, k, w } => {
                let half = |i: usize| -> Vec<u16> {
                    w.chunks_exact(k)
                        .flat_map(|r| r[i * k / 2..(i + 1) * k / 2].iter().copied())
                        .collect()
                };
                [
                    Bf16Tensor::upload(n, k / 2, &half(0)).map_err(gpu("mtp.fc[:, :h]"))?,
                    Bf16Tensor::upload(n, k / 2, &half(1)).map_err(gpu("mtp.fc[:, h:]"))?,
                ]
            }
            Matrix::Fp8 { .. } => return Err("mtp.fc: expected bf16".into()),
        };
        let raw_e = read_vec(src, "mtp.pre_fc_norm_embedding.weight", h)?;
        let raw_h = read_vec(src, "mtp.pre_fc_norm_hidden.weight", h)?;
        let raw_f = read_vec(src, "mtp.norm.weight", h)?;
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let raw_norm_means = [mean(&raw_e), mean(&raw_h), mean(&raw_f)];
        let a = "mtp.layers.0.self_attn";
        let mix = fuse_fp8(
            &format!("{a} q|k|v"),
            vec![
                rm(
                    src,
                    &format!("{a}.q_proj.weight"),
                    2 * c.heads * c.head_dim,
                    h,
                )?,
                rm(src, &format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
                rm(src, &format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
            ],
        )?;
        let o = upload_fp8(
            &format!("{a}.o_proj"),
            rm(src, &format!("{a}.o_proj.weight"), h, c.heads * c.head_dim)?,
        )?;
        let q_norm = read_vec(src, &format!("{a}.q_norm.weight"), c.head_dim)?;
        let k_norm = read_vec(src, &format!("{a}.k_norm.weight"), c.head_dim)?;
        let mut attn = Attn256Cache::alloc(1, model.max_pos(), c.rope_theta, c.rotary_dim)
            .map_err(gpu("mtp attn256 alloc"))?;
        attn.set_norms(0, &q_norm, &k_norm)
            .map_err(gpu("mtp attn256 set_norms"))?;
        let m = "mtp.layers.0.mlp";
        let gate_up = fuse_fp8(
            &format!("{m} gate|up"),
            vec![
                rm(src, &format!("{m}.gate_proj.weight"), c.intermediate, h)?,
                rm(src, &format!("{m}.up_proj.weight"), c.intermediate, h)?,
            ],
        )?;
        let down = upload_fp8(
            &format!("{m}.down_proj"),
            rm(src, &format!("{m}.down_proj.weight"), h, c.intermediate)?,
        )?;
        let in_norm = fold(read_vec(src, "mtp.layers.0.input_layernorm.weight", h)?);
        let post_norm = fold(read_vec(
            src,
            "mtp.layers.0.post_attention_layernorm.weight",
            h,
        )?);
        Ok(Self {
            fc,
            norm_e: fold(raw_e),
            norm_h: fold(raw_h),
            layer: LayerTensors {
                mix,
                ab: None,
                o,
                gate_up,
                down,
            },
            norms: (in_norm, post_norm),
            final_norm: fold(raw_f),
            attn,
            gather: q::gather_table(c.rotary_dim),
            opts,
            raw_norm_means,
            eps: c.eps,
            last: Vec::new(),
            next_pos: 0,
        })
    }

    /// Bytes the head holds on the device (fc, the layer; the KV cache not counted).
    pub fn weight_bytes(&self) -> usize {
        self.fc[0].resident_bytes()
            + self.fc[1].resident_bytes()
            + self.layer.mix.resident_bytes()
            + self.layer.o.resident_bytes()
            + self.layer.gate_up.resident_bytes()
            + self.layer.down.resident_bytes()
    }

    /// Runs the head on `m` pairs (`hidden[r]` of the main model at position `pos0 + r`, the
    /// token that followed it) at MTP positions `pos0 + r + offset`, and returns the draft of the
    /// last pair (its greedy guess for the token after that token). Rewrites the head's KV at
    /// those positions with these pairs.
    pub fn feed(
        &mut self,
        model: &mut Qwen38Omega,
        hidden: &[f32],
        next: &[u32],
        pos0: usize,
        st: &mut RowStats,
    ) -> Result<u32, String> {
        let h = q::HIDDEN;
        let m = next.len();
        if hidden.len() != m * h || m == 0 {
            return Err(format!(
                "mtp feed: {} hidden values for {m} tokens",
                hidden.len()
            ));
        }
        let t = Instant::now();
        let mut cat = vec![0f32; 2 * h];
        let mut x = vec![0f32; m * h];
        let mut hn = vec![0f32; h];
        st.host_ns += ns(t);
        for (r, xr) in x.chunks_exact_mut(h).enumerate() {
            let t = Instant::now();
            let hid = &hidden[r * h..(r + 1) * h];
            let hsrc: &[f32] = if self.opts.normed_hidden {
                rmsnorm(hid, &model.final_norm, self.eps, &mut hn);
                &hn
            } else {
                hid
            };
            let e = model.embed_row(next[r]);
            let (first, second) = cat.split_at_mut(h);
            let (ed, hd) = match self.opts.order {
                FcOrder::EmbedFirst => (first, second),
                FcOrder::HiddenFirst => (second, first),
            };
            rmsnorm(&e, &self.norm_e, self.eps, ed);
            rmsnorm(hsrc, &self.norm_h, self.eps, hd);
            st.host_ns += ns(t);
            let t = Instant::now();
            let mut y1 = vec![0f32; h];
            self.fc[0]
                .gemv_f32(&cat[..h], xr)
                .map_err(gpu("mtp.fc[:, :h]"))?;
            self.fc[1]
                .gemv_f32(&cat[h..], &mut y1)
                .map_err(gpu("mtp.fc[:, h:]"))?;
            for (a, b) in xr.iter_mut().zip(&y1) {
                *a += b;
            }
            st.ab_ns += ns(t);
        }
        let pos = pos0 + self.opts.pos_offset;
        if pos + m > self.attn.max_pos() {
            return Err(format!("mtp feed: position {pos} + {m} past the cache"));
        }
        let mixer = Mixer::Attn {
            idx: 0,
            gather: &self.gather,
        };
        let mut targets = OneSequence {
            attn: Some(&mut self.attn),
            gdn: None,
            pos0: pos,
        };
        let (in_norm, post_norm) = &self.norms;
        layer_rows(
            &self.layer,
            in_norm,
            post_norm,
            mixer,
            &mut targets,
            m,
            self.eps,
            &mut x,
            st,
        )
        .map_err(|e| format!("mtp layer: {e}"))?;
        self.last = x[(m - 1) * h..].to_vec();
        self.next_pos = pos0 + m;
        self.draft_head(model, st)
    }

    /// One more draft after [`Self::feed`] or [`Self::chain`]: pairs the head's own last output
    /// with `token` (its last draft) at the next position.
    pub fn chain(
        &mut self,
        model: &mut Qwen38Omega,
        token: u32,
        st: &mut RowStats,
    ) -> Result<u32, String> {
        let hidden = std::mem::take(&mut self.last);
        // The head's own output stands in for the main model's hidden: never final-normed twice.
        let normed = std::mem::replace(&mut self.opts.normed_hidden, false);
        let r = self.feed(model, &hidden, &[token], self.next_pos, st);
        self.opts.normed_hidden = normed;
        r
    }

    /// `mtp.norm`, the model's lm_head, argmax over the last output.
    fn draft_head(&mut self, model: &mut Qwen38Omega, st: &mut RowStats) -> Result<u32, String> {
        let t = Instant::now();
        let mut hn = vec![0f32; q::HIDDEN];
        rmsnorm(&self.last, &self.final_norm, self.eps, &mut hn);
        let mut logits = vec![0f32; model.config.vocab];
        model
            .lm_head
            .gemv_f32(&hn, &mut logits)
            .map_err(gpu("mtp lm_head"))?;
        st.head_ns += ns(t);
        st.head_calls += 1;
        Ok(argmax(&logits))
    }
}

/// Test-only faults that must turn the exactness check red (negative controls).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecMutant {
    #[default]
    None,
    /// The first refused draft of a round is accepted anyway.
    ExtraAccept,
    /// A refused draft leaves the DeltaNet state of every checked row in place.
    NoRollback,
    /// Row `r`'s draft is checked against row `r + 1`'s token.
    RowShift,
}

/// What one generation cost and did.
#[derive(Debug, Clone, Default)]
pub struct SpecRun {
    pub tokens: Vec<u32>,
    pub rounds: u64,
    /// Drafts made and drafts accepted.
    pub drafted: u64,
    pub accepted: u64,
    /// Rounds that refused a draft (state put back).
    pub rollbacks: u64,
    /// Decode wall time (prompt excluded) and its parts.
    pub decode_ns: u64,
    pub prompt_ns: u64,
    pub draft_ns: u64,
    pub verify_ns: u64,
    pub snapshot_ns: u64,
    pub rollback_ns: u64,
    pub main: RowStats,
    pub mtp: RowStats,
}

/// Greedy generation on the exact rows path: `k = 0` is the plain path one token at a time;
/// `k >= 1` drafts `k` tokens per round with `mtp` and checks them in one pass.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut Qwen38Omega,
    mut mtp: Option<&mut Mtp>,
    prompt: &[u32],
    n: usize,
    k: usize,
    stop: &[u32],
    chunk: usize,
    mutant: SpecMutant,
) -> Result<SpecRun, String> {
    if prompt.is_empty() || n == 0 {
        return Err("generate: empty prompt or n = 0".into());
    }
    if k > 0 && mtp.is_none() {
        return Err("generate: k > 0 needs the MTP head".into());
    }
    let h = q::HIDDEN;
    let mut run = SpecRun::default();
    model.reset()?;
    let t0 = Instant::now();
    let mut rows = Vec::with_capacity(prompt.len() * h);
    for c in prompt.chunks(chunk.max(1)) {
        rows.extend(model.exact_rows(c, None, &mut run.main)?);
    }
    let last = rows[(prompt.len() - 1) * h..].to_vec();
    let mut t = model.head_token(&last, &mut run.main)?;
    run.tokens.push(t);
    let mut draft = None;
    if k > 0 {
        let m = mtp.as_deref_mut().expect("checked");
        let mut next = prompt[1..].to_vec();
        next.push(t);
        let td = Instant::now();
        draft = Some(m.feed(model, &rows, &next, 0, &mut run.mtp)?);
        run.draft_ns += ns(td);
    }
    run.prompt_ns = ns(t0);
    drop(rows);

    let t1 = Instant::now();
    let mut snap = Qwen38Snapshot::default();
    let mut keep: Vec<GdnRows> = Vec::new();
    while run.tokens.len() < n && !stop.contains(&t) {
        run.rounds += 1;
        let p = model.pos;
        // Drafts: the first came from the last feed; chain the rest.
        let td = Instant::now();
        let mut feed = vec![t];
        if let Some(m) = mtp.as_deref_mut() {
            let mut d = draft.take().expect("a feed made the first draft");
            feed.push(d);
            while feed.len() < k + 1 && p + feed.len() < model.max_pos() {
                d = m.chain(model, d, &mut run.mtp)?;
                feed.push(d);
            }
        }
        run.draft_ns += ns(td);
        let drafts = feed.len() - 1;
        run.drafted += drafts as u64;

        let ts = Instant::now();
        if drafts > 0 {
            model.snapshot(&mut snap)?;
        }
        run.snapshot_ns += ns(ts);
        let tv = Instant::now();
        let out = model.exact_rows(&feed, (drafts > 0).then_some(&mut keep), &mut run.main)?;
        // Row r's greedy token, row by row until a draft is refused.
        let mut j = 0;
        let mut extra = mutant == SpecMutant::ExtraAccept;
        let next = loop {
            let row = match mutant {
                SpecMutant::RowShift if j < drafts => j + 1,
                _ => j,
            };
            let pj = model.head_token(&out[row * h..(row + 1) * h], &mut run.main)?;
            if j < drafts && (pj == feed[j + 1] || std::mem::take(&mut extra)) {
                j += 1;
                continue;
            }
            break pj;
        };
        run.verify_ns += ns(tv);
        run.accepted += j as u64;
        let emitted = &feed[1..=j];
        let room = n - run.tokens.len();
        run.tokens.extend(emitted.iter().take(room));
        if run.tokens.len() < n {
            run.tokens.push(next);
        }
        // An accepted draft can be the stop token; the round's later rows are past the end.
        if let Some(s) = run.tokens.iter().position(|x| stop.contains(x)) {
            run.tokens.truncate(s + 1);
            break;
        }
        if j < drafts {
            run.rollbacks += 1;
            if mutant != SpecMutant::NoRollback {
                let tr = Instant::now();
                model.restore(&snap)?;
                model.replay_gdn(&keep, j + 1, &mut run.main)?;
                if model.pos != p + j + 1 {
                    return Err(format!(
                        "rollback: position {} after replay, want {}",
                        model.pos,
                        p + j + 1
                    ));
                }
                run.rollback_ns += ns(tr);
            } else {
                model.pos = p + j + 1;
            }
        }
        t = next;
        if let Some(m) = mtp.as_deref_mut() {
            if run.tokens.len() < n && model.pos < model.max_pos() {
                let mut nexts = feed[1..=j].to_vec();
                nexts.push(next);
                let td = Instant::now();
                draft = Some(m.feed(model, &out[..(j + 1) * h], &nexts, p, &mut run.mtp)?);
                run.draft_ns += ns(td);
            }
        }
    }
    run.decode_ns = ns(t1);
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_is_strict_first_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 1.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0]), 0);
    }

    #[test]
    fn options_default_is_embed_first_residual_same_position() {
        let o = MtpOptions::default();
        assert_eq!(o.order, FcOrder::EmbedFirst);
        assert!(!o.normed_hidden);
        assert_eq!(o.pos_offset, 0);
    }

    #[test]
    fn par_rows_same_bits_as_one_thread() {
        let (m, w) = (13usize, 37usize);
        let src: Vec<f32> = (0..m * w).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
        let silu = |r: usize, o: &mut [f32]| {
            for (c, v) in o.iter_mut().enumerate() {
                let g = src[r * w + c];
                *v = g / (1.0 + (-g).exp()) * (c as f32 + 0.5);
            }
        };
        let mut one = vec![0f32; m * w];
        par_rows_n(&mut one, w, 1, silu);
        for threads in [2, 3, 4, 8, 64] {
            let mut many = vec![f32::NAN; m * w];
            par_rows_n(&mut many, w, threads, silu);
            let same = one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{threads} threads changed the bits");
        }
    }
}
