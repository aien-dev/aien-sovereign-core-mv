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
//!
//! The cascade (2026-10-19, handoff moonshot-e-cascade): a round is one [`verify_pass`] over the
//! last token and a chain of up to [`MAX_CHAIN`] drafts. Inside that pass the MTP head runs as
//! "layer 65" on every row, paired with the row's own greedy token, so the accepted tip's next
//! draft is a by-product of the pass (no model call between passes except the chain's further
//! levels). Drafts come from the MTP chain, from the prompt-lookup n-gram drafter
//! ([`NgramDraft`], no model call), or a mix ([`DraftSource`]). The DeltaNet state is one
//! committed state rebuilt from the accepted rows only (one copy per pass, replay of the accepted
//! rows, never a snapshot per draft). [`verify_pass`] is the one boundary the fast tree-verify
//! GEMM (omega branch cascade/tree-verify-gemm) drops into: [`VerifyKernel`].
use std::path::Path;
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

/// The `w` largest logits' indices, largest first, the first index on a tie, so `topk(v, w)[0]`
/// is [`argmax`]`(v)`.
pub fn topk(v: &[f32], w: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(w);
    for _ in 0..w.min(v.len()) {
        let mut best: Option<usize> = None;
        for (i, &x) in v.iter().enumerate() {
            if out.contains(&(i as u32)) {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => x > v[b],
            };
            if better {
                best = Some(i);
            }
        }
        out.push(best.expect("w <= len") as u32);
    }
    out
}

/// Wall time of the exact rows path by kind (every chip call is synchronous, so a call's wall
/// time includes its launch, wait and copies).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowStats {
    pub rows: u64,
    pub passes: u64,
    pub gemm_ns: u64,
    pub gemm_calls: u64,
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

/// The mixer of one layer of a rows pass.
pub(super) enum Mixer<'a> {
    Attn {
        cache: &'a mut Attn256Cache,
        idx: usize,
        gather: &'a [u32],
        /// A tree pass: row `r` writes its K and V at `pos0 + depth[r]` and its raw qkv row is
        /// kept (see [`TreeShape`]).
        tree: Option<(&'a TreeShape, &'a mut Vec<f32>)>,
    },
    Gdn {
        gdn: &'a mut Gdn,
        idx: usize,
        ab: &'a Bf16Tensor,
        keep: Option<&'a mut GdnRows>,
        /// A tree pass: the shape and this layer's committed (tail, state) to rebuild a branch
        /// from.
        tree: Option<(&'a TreeShape, &'a [f32], &'a [f32])>,
    },
}

/// The shape of a tree verify pass: rows in depth-first preorder, `parent[0] = 0` for the
/// root (the last committed token), `parent[r] < r`, `depth[r] = depth[parent[r]] + 1`.
///
/// Why preorder makes the per-row exact path a tree path with no mask and no state copy per
/// node:
/// - attention: row `r` writes its K and V at slot `pos0 + depth[r]` and reads slots
///   `0 ..= pos0 + depth[r]`. In preorder every row between an ancestor and `r` is deeper
///   than that ancestor, so the slots `pos0 .. pos0 + depth[r]` hold exactly `r`'s ancestors.
/// - DeltaNet: rows run in order on one state. When `parent[r] != r - 1` the layer's state is
///   rebuilt: the committed state put back and the kept inputs of `r`'s ancestors replayed
///   (rebuild only the path, never a snapshot per node).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TreeShape {
    pub parent: Vec<usize>,
    pub depth: Vec<usize>,
    /// Test fault: skip the branch rebuild (a sibling sees the previous branch's state).
    pub mutant_no_rebuild: bool,
}

impl TreeShape {
    /// A shape from parents (preorder, `parent[0] = 0`).
    pub fn from_parents(parent: Vec<usize>) -> Result<Self, String> {
        if parent.first() != Some(&0) {
            return Err("tree: row 0 is the root".into());
        }
        let mut depth = vec![0usize; parent.len()];
        for r in 1..parent.len() {
            if parent[r] >= r {
                return Err(format!(
                    "tree: row {r} has parent {} (not before it)",
                    parent[r]
                ));
            }
            depth[r] = depth[parent[r]] + 1;
        }
        // Preorder: the parent of r is r - 1 or an ancestor of r - 1.
        for r in 1..parent.len() {
            let mut a = r - 1;
            while a != parent[r] && a != 0 {
                a = parent[a];
            }
            if a != parent[r] {
                return Err(format!("tree: row {r} breaks depth-first preorder"));
            }
        }
        Ok(Self {
            parent,
            depth,
            mutant_no_rebuild: false,
        })
    }

    /// Rows from the root to `r`, root first.
    pub fn path(&self, r: usize) -> Vec<usize> {
        let mut p = vec![r];
        let mut a = r;
        while a != 0 {
            a = self.parent[a];
            p.push(a);
        }
        p.reverse();
        p
    }

    /// The children of `r` in row order.
    pub fn children(&self, r: usize) -> impl Iterator<Item = usize> + '_ {
        (r + 1..self.parent.len()).filter(move |&c| self.parent[c] == r)
    }

    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0)
    }
}

/// Row-table GEMM over one dense FP8 tensor: `y[m][n] = W . x[m][k]`.
fn gemm(
    w: &aien_omega_fp8::Fp8Tensor,
    x: &[f32],
    y: &mut [f32],
    m: usize,
    what: &str,
    st: &mut RowStats,
) -> Result<(), String> {
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

/// `m` rows of the residual stream `x` (`[m][hidden]`, positions `pos0 ..`) through one layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_rows(
    lt: &LayerTensors,
    in_norm: &[f32],
    post_norm: &[f32],
    mixer: Mixer<'_>,
    pos0: usize,
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
        Mixer::Attn {
            cache,
            idx,
            gather,
            mut tree,
        } => {
            let (gq, gg) = gather.split_at(q::ATT_QKV);
            if let Some((_, kq)) = tree.as_mut() {
                kq.clear();
            }
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
                let slot = match tree.as_mut() {
                    Some((shape, kq)) => {
                        kq.extend_from_slice(&qkv);
                        pos0 + shape.depth[r]
                    }
                    None => pos0 + r,
                };
                st.host_ns += ns(t);
                let t = Instant::now();
                cache
                    .decode(idx, slot, &qkv, eps, out)
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
        Mixer::Gdn {
            gdn,
            idx,
            ab,
            mut keep,
            tree,
        } => {
            if tree.is_some() && keep.is_none() {
                return Err("a tree pass keeps the DeltaNet row inputs".into());
            }
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
                if let Some((shape, tail0, state0)) = tree {
                    if r > 0 && shape.parent[r] != r - 1 && !shape.mutant_no_rebuild {
                        // A new branch: the committed state, then the path down to the parent.
                        gdn.set_state(idx, tail0, state0)
                            .map_err(|e| format!("gdn branch row {r}: {e}"))?;
                        let k = keep.as_deref().expect("checked");
                        let mut scratch = vec![0f32; mo];
                        for a in shape.path(shape.parent[r]) {
                            gdn.step(
                                idx,
                                &k.qkv[a * q::GDN_QKV..(a + 1) * q::GDN_QKV],
                                &k.z[a * mo..(a + 1) * mo],
                                &abr[a * q::AB..(a + 1) * q::AB],
                                EPS_L2,
                                eps,
                                &mut scratch,
                            )
                            .map_err(|e| format!("gdn branch replay row {a}: {e}"))?;
                            st.gdn_calls += 1;
                        }
                    }
                }
                gdn.step(
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
    for (a, r) in act.chunks_exact_mut(inter).zip(gu.chunks_exact(2 * inter)) {
        let (g, u) = r.split_at(inter);
        for ((a, &g), &u) in a.iter_mut().zip(g).zip(u) {
            *a = g / (1.0 + (-g).exp()) * u;
        }
    }
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
        keep: Option<&mut Vec<GdnRows>>,
        st: &mut RowStats,
    ) -> Result<Vec<f32>, String> {
        self.rows_pass(tokens, keep, None, st)
    }

    /// A tree verify pass on the exact rows path ([`TreeShape`]): row `r` is what the one-row
    /// path gives after the committed history plus `r`'s ancestors, bit for bit. `committed`
    /// is the [`Self::snapshot`] taken at the pass start (branches rebuild from it); `kv` gets
    /// every attention layer's raw qkv rows for [`Self::tree_commit`]. Leaves the position at
    /// the start (the caller commits the accepted path).
    pub fn tree_rows(
        &mut self,
        tokens: &[u32],
        shape: &TreeShape,
        committed: &Qwen38Snapshot,
        keep: &mut Vec<GdnRows>,
        kv: &mut Vec<Vec<f32>>,
        st: &mut RowStats,
    ) -> Result<Vec<f32>, String> {
        if shape.parent.len() != tokens.len() {
            return Err("tree_rows: shape and tokens differ in length".into());
        }
        if committed.pos != self.pos {
            return Err("tree_rows: the snapshot is not of this position".into());
        }
        let pos0 = self.pos;
        let x = self.rows_pass(tokens, Some(keep), Some((shape, committed, kv)), st)?;
        self.pos = pos0;
        Ok(x)
    }

    /// Commits the accepted path `path` (rows of the last [`Self::tree_rows`], root first):
    /// the committed DeltaNet state put back and the path's kept inputs replayed, and every
    /// attention layer's K and V for the path rewritten at `pos0 + depth` from the kept qkv (a
    /// later branch may have overwritten them). Afterwards the state and the cache are those
    /// of the path's tokens fed one at a time.
    pub fn tree_commit(
        &mut self,
        committed: &Qwen38Snapshot,
        keep: &[GdnRows],
        kv: &[Vec<f32>],
        path: &[usize],
        fix_kv: bool,
        st: &mut RowStats,
    ) -> Result<(), String> {
        self.restore(committed)?;
        let pos0 = self.pos;
        let eps = self.config.eps;
        if let Some(g) = self.gdn.as_mut() {
            let mut out = vec![0f32; q::MIXER_OUT];
            for (gi, k) in keep.iter().enumerate() {
                for &r in path {
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
                    .map_err(|e| format!("tree commit gdn layer {gi} row {r}: {e}"))?;
                    st.gdn_ns += ns(t);
                    st.gdn_calls += 1;
                }
            }
        }
        if fix_kv {
            if let Some(c) = self.attn.as_mut() {
                let mut out = vec![0f32; q::MIXER_OUT];
                for (ai, rows) in kv.iter().enumerate() {
                    for (d, &r) in path.iter().enumerate() {
                        let t = Instant::now();
                        c.decode(
                            ai,
                            pos0 + d,
                            &rows[r * q::ATT_QKV..(r + 1) * q::ATT_QKV],
                            eps,
                            &mut out,
                        )
                        .map_err(|e| format!("tree commit kv layer {ai} row {r}: {e}"))?;
                        st.attn_ns += ns(t);
                        st.attn_calls += 1;
                    }
                }
            }
        }
        self.pos = pos0 + path.len();
        Ok(())
    }

    fn rows_pass(
        &mut self,
        tokens: &[u32],
        mut keep: Option<&mut Vec<GdnRows>>,
        mut tree: Option<(&TreeShape, &Qwen38Snapshot, &mut Vec<Vec<f32>>)>,
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
        let span = match &tree {
            Some((shape, _, _)) => shape.max_depth() + 1,
            None => m,
        };
        if self.pos + span > self.max_pos {
            return Err(format!(
                "exact_rows: positions {} + {span} > KV cache of {}",
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
        let (mut gi, mut ai) = (0usize, 0usize);
        if let Some((_, _, kv)) = tree.as_mut() {
            kv.resize_with(
                self.config.full_attention.iter().filter(|&&f| f).count(),
                Vec::new,
            );
        }
        let (tf, sf) = (q::GDN_TAIL_FLOATS, q::GDN_STATE_FLOATS);
        for l in 0..self.config.layers {
            let lt = &self.layers[l];
            let (in_norm, post_norm) = &self.norms[l];
            let mixer = if self.config.full_attention[l] {
                ai += 1;
                Mixer::Attn {
                    cache: self.attn.as_mut().expect("an attention layer has a cache"),
                    idx: ai - 1,
                    gather: &gather,
                    tree: tree
                        .as_mut()
                        .map(|(shape, _, kv)| (&**shape, &mut kv[ai - 1])),
                }
            } else {
                gi += 1;
                Mixer::Gdn {
                    gdn: self.gdn.as_mut().expect("a DeltaNet layer has state"),
                    idx: gi - 1,
                    ab: lt.ab.as_ref().expect("a DeltaNet layer has a|b"),
                    keep: keep.as_deref_mut().map(|k| &mut k[gi - 1]),
                    tree: tree.as_ref().map(|(shape, snap, _)| {
                        let l = gi - 1;
                        (
                            &**shape,
                            &snap.tails[l * tf..(l + 1) * tf],
                            &snap.states[l * sf..(l + 1) * sf],
                        )
                    }),
                }
            };
            layer_rows(lt, in_norm, post_norm, mixer, pos0, m, eps, &mut x, st)
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
    /// Every row of the last [`Self::layer65`], `[m][hidden]`.
    rows: Vec<f32>,
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
            rows: Vec::new(),
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
        let m = next.len();
        self.layer65(model, hidden, next, pos0, st)?;
        self.set_tip(m - 1, pos0 + m);
        self.draft_head(model, st)
    }

    /// "Layer 65": the head's fc and attention layer on `m` pairs (`hidden[r]`, `next[r]`) at
    /// MTP positions `pos0 + r + offset`, every row kept in [`Self::rows`] (`[m][hidden]`). The
    /// cascade runs this on every row of a verify pass, so whichever row turns out to be the
    /// accepted tip, its MTP output (and with [`Self::head_rows`], its draft) is already there.
    /// Row `r` reads only the head's KV at positions `<= pos0 + r + offset`, so rows `0 .. j`
    /// are bit for bit what feeding the accepted rows alone would give.
    pub fn layer65(
        &mut self,
        model: &mut Qwen38Omega,
        hidden: &[f32],
        next: &[u32],
        pos0: usize,
        st: &mut RowStats,
    ) -> Result<(), String> {
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
            cache: &mut self.attn,
            idx: 0,
            gather: &self.gather,
            tree: None,
        };
        let (in_norm, post_norm) = &self.norms;
        layer_rows(
            &self.layer,
            in_norm,
            post_norm,
            mixer,
            pos,
            m,
            self.eps,
            &mut x,
            st,
        )
        .map_err(|e| format!("mtp layer: {e}"))?;
        self.rows = x;
        Ok(())
    }

    /// Makes row `row` of the last [`Self::layer65`] the tip the next [`Self::chain`] continues
    /// from, at MTP position `next_pos`.
    pub fn set_tip(&mut self, row: usize, next_pos: usize) {
        let h = q::HIDDEN;
        self.last = self.rows[row * h..(row + 1) * h].to_vec();
        self.next_pos = next_pos;
    }

    /// The MTP output rows of the last [`Self::layer65`], `[m][hidden]`.
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    /// `mtp.norm`, the lm_head and argmax over every row of the last [`Self::layer65`]: the
    /// draft that follows each row's pair.
    pub fn head_rows(
        &mut self,
        model: &mut Qwen38Omega,
        st: &mut RowStats,
    ) -> Result<Vec<u32>, String> {
        let h = q::HIDDEN;
        let rows = std::mem::take(&mut self.rows);
        let mut out = Vec::with_capacity(rows.len() / h);
        for r in rows.chunks_exact(h) {
            out.push(self.head_of(model, r, st)?);
        }
        self.rows = rows;
        Ok(out)
    }

    /// One more draft after [`Self::feed`] or [`Self::chain`]: pairs the head's own last output
    /// with `token` (its last draft) at the next position.
    pub fn chain(
        &mut self,
        model: &mut Qwen38Omega,
        token: u32,
        st: &mut RowStats,
    ) -> Result<u32, String> {
        self.chain_layer(model, token, st)?;
        self.draft_head(model, st)
    }

    /// [`Self::chain`] without the word-scoring head: the pair (`last`, `token`) through the
    /// layer only, the output the new tip. For walking the head through tokens drafted by
    /// another source (the n-gram lookup) without paying the lm_head for each.
    pub fn chain_layer(
        &mut self,
        model: &mut Qwen38Omega,
        token: u32,
        st: &mut RowStats,
    ) -> Result<(), String> {
        let hidden = std::mem::take(&mut self.last);
        let pos0 = self.next_pos;
        // The head's own output stands in for the main model's hidden: never final-normed twice.
        let normed = std::mem::replace(&mut self.opts.normed_hidden, false);
        let r = self.layer65(model, &hidden, &[token], pos0, st);
        self.opts.normed_hidden = normed;
        r?;
        self.set_tip(0, pos0 + 1);
        Ok(())
    }

    /// `mtp.norm`, the model's lm_head, argmax over the last output.
    pub fn draft_head(
        &mut self,
        model: &mut Qwen38Omega,
        st: &mut RowStats,
    ) -> Result<u32, String> {
        let last = std::mem::take(&mut self.last);
        let r = self.head_of(model, &last, st);
        self.last = last;
        r
    }

    fn head_of(
        &self,
        model: &mut Qwen38Omega,
        row: &[f32],
        st: &mut RowStats,
    ) -> Result<u32, String> {
        Ok(argmax(&self.logits_of(model, row, st)?))
    }

    /// The `w` best drafts after the tip (largest first; the first is [`Self::draft_head`]'s).
    pub fn draft_topk(
        &mut self,
        model: &mut Qwen38Omega,
        w: usize,
        st: &mut RowStats,
    ) -> Result<Vec<u32>, String> {
        let last = std::mem::take(&mut self.last);
        let r = self.logits_of(model, &last, st);
        self.last = last;
        Ok(topk(&r?, w.max(1)))
    }

    /// The tip (`last`, `next_pos`), to branch the chain from it more than once.
    pub fn tip(&self) -> (Vec<f32>, usize) {
        (self.last.clone(), self.next_pos)
    }

    /// Puts a [`Self::tip`] back.
    pub fn set_tip_state(&mut self, tip: &(Vec<f32>, usize)) {
        self.last.clone_from(&tip.0);
        self.next_pos = tip.1;
    }

    fn logits_of(
        &self,
        model: &mut Qwen38Omega,
        row: &[f32],
        st: &mut RowStats,
    ) -> Result<Vec<f32>, String> {
        let t = Instant::now();
        let mut hn = vec![0f32; q::HIDDEN];
        rmsnorm(row, &self.final_norm, self.eps, &mut hn);
        let mut logits = vec![0f32; model.config.vocab];
        model
            .lm_head
            .gemv_f32(&hn, &mut logits)
            .map_err(gpu("mtp lm_head"))?;
        st.head_ns += ns(t);
        st.head_calls += 1;
        Ok(logits)
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
    /// Tree pass: a side branch runs on the previous branch's DeltaNet state (no rebuild).
    TreeNoRebuild,
    /// Tree commit: the accepted path's K and V are not rewritten (a later branch's stay).
    TreeNoKvFix,
}

/// Which routine runs the verify pass (the one function boundary the fast tree-verify GEMM
/// drops into: [`verify_pass`]). Everything else in the cascade (draft sources, acceptance,
/// state rebuild, the chain) is the same whichever kernel runs the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerifyKernel {
    /// Today's exact rows path: `omega_gpu_fp8gemm_f32` in row pairs (compute bound past one
    /// pair), bf16 lm_head GEMV per row, per-row glue on the host.
    #[default]
    ExactRows,
}

/// Which rows the verify pass scores with the lm_head and runs the MTP head on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadRows {
    /// The cascade shape: every row of the pass (main head, MTP layer 65, MTP head), so the
    /// tip's draft is a by-product of the pass whichever row the tip turns out to be.
    #[default]
    All,
    /// The pre-cascade shape: main head row by row until a draft is refused, then the MTP head
    /// over the accepted rows only. Same tokens, same acceptance; fewer lm_head calls while the
    /// head is a per-row GEMV. Kept so the cost of `All` can be measured against it.
    Accepted,
}

/// What one verify pass returns: the acceptance decision and the next round's seed.
#[derive(Debug, Clone, Default)]
pub struct VerifyOut {
    /// Residual rows after the last layer, `[m][hidden]`.
    pub hidden: Vec<f32>,
    /// Row `r`'s greedy token for every scored row (`accepted + 1` rows at least).
    pub tokens: Vec<u32>,
    /// Drafts `feed[1..=accepted]` were the model's own tokens.
    pub accepted: usize,
    /// The model's own token after the accepted rows (`tokens[accepted]`).
    pub next: u32,
    /// With an MTP head: its draft for the token after `next` (the seed of the next chain).
    pub draft: Option<u32>,
}

/// ONE verify pass: `feed` (the last token and its drafts) through the whole model as `m`
/// rows, the lm_head on the rows, the acceptance decision, and the MTP head ("layer 65") on
/// the rows so the next round's first draft comes out of the same pass.
///
/// Contract for any [`VerifyKernel`]: row `r` of `hidden` and `tokens[r]` equal what the
/// one-row path gives over the same history (bit for bit for `ExactRows`), `keep` receives the
/// DeltaNet row inputs for [`Qwen38Omega::replay_gdn`], and the model's position advances by
/// `m`. `mtp`, when given, is left with its tip on the accepted row (`Mtp::chain` continues the
/// draft from there) and its KV rows `0 ..= accepted` at `pos0 ..` as a feed of the accepted
/// rows would leave them.
#[allow(clippy::too_many_arguments)]
pub fn verify_pass(
    kernel: VerifyKernel,
    model: &mut Qwen38Omega,
    mtp: Option<&mut Mtp>,
    feed: &[u32],
    keep: Option<&mut Vec<GdnRows>>,
    rows: HeadRows,
    mutant: SpecMutant,
    st: &mut RowStats,
    st_mtp: &mut RowStats,
) -> Result<VerifyOut, String> {
    let VerifyKernel::ExactRows = kernel;
    let h = q::HIDDEN;
    let m = feed.len();
    let drafts = m - 1;
    let pos0 = model.pos;
    let hidden = model.exact_rows(feed, keep, st)?;
    let mut tokens = Vec::with_capacity(m);
    // Rows in order until a draft is refused (the acceptance rule); `All` scores the rest too.
    let mut j = 0;
    let mut extra = mutant == SpecMutant::ExtraAccept;
    let mut deciding = true;
    for r in 0..m {
        let row = match mutant {
            SpecMutant::RowShift if r < drafts => r + 1,
            _ => r,
        };
        tokens.push(model.head_token(&hidden[row * h..(row + 1) * h], st)?);
        if deciding {
            if j < drafts && (tokens[r] == feed[r + 1] || std::mem::take(&mut extra)) {
                j += 1;
            } else {
                deciding = false;
                if rows == HeadRows::Accepted {
                    break;
                }
            }
        }
    }
    let next = tokens[j];
    let draft = match mtp {
        None => None,
        Some(mt) => {
            let n = match rows {
                HeadRows::All => m,
                HeadRows::Accepted => j + 1,
            };
            // Pair row r with the model's own token for row r: for r < j that is the accepted
            // draft feed[r + 1], for r = j it is `next`, so the pairs equal a feed of the
            // accepted rows and the tip's pair is already among them.
            mt.layer65(model, &hidden[..n * h], &tokens[..n], pos0, st_mtp)?;
            mt.set_tip(j, pos0 + j + 1);
            let d = match rows {
                HeadRows::All => mt.head_rows(model, st_mtp)?[j],
                HeadRows::Accepted => mt.draft_head(model, st_mtp)?,
            };
            Some(d)
        }
    };
    Ok(VerifyOut {
        hidden,
        tokens,
        accepted: j,
        next,
        draft,
    })
}

/// Prompt-lookup drafting: when the last `n` tokens of the history (prompt plus the answer so
/// far) appeared earlier, propose the tokens that followed that earlier span. No model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramDraft {
    /// Longest and shortest span tried, longest first.
    pub max_n: usize,
    pub min_n: usize,
}

impl Default for NgramDraft {
    fn default() -> Self {
        Self { max_n: 6, min_n: 2 }
    }
}

/// What the n-gram lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHit {
    /// Span length matched.
    pub n: usize,
    /// Where the earlier copy of the span starts.
    pub at: usize,
    /// The tokens that followed it, at most `max` of them.
    pub tokens: Vec<u32>,
}

impl NgramDraft {
    /// The latest earlier occurrence of the longest matching suffix (length `min_n ..= max_n`)
    /// of `history`, and up to `max` tokens after it (never fewer than one).
    pub fn draft(&self, history: &[u32], max: usize) -> Option<NgramHit> {
        let len = history.len();
        if max == 0 || self.min_n == 0 || self.max_n < self.min_n {
            return None;
        }
        for n in (self.min_n..=self.max_n.min(len.saturating_sub(1))).rev() {
            let suffix = &history[len - n..];
            // Rightmost start i with history[i .. i + n] == suffix and at least one token after
            // the span before the suffix itself begins.
            let mut i = len - n;
            while i > 0 {
                i -= 1;
                if &history[i..i + n] == suffix {
                    let from = i + n;
                    let to = (from + max).min(len);
                    if to > from {
                        return Some(NgramHit {
                            n,
                            at: i,
                            tokens: history[from..to].to_vec(),
                        });
                    }
                }
            }
        }
        None
    }
}

/// Where a round's drafts come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftSource {
    /// The MTP chain only.
    #[default]
    Mtp,
    /// The n-gram lookup when it matches, else the one free MTP draft (no chain).
    Ngram,
    /// The n-gram lookup when it matches, else the MTP chain.
    NgramFirst,
    /// The n-gram lookup when it matches and its first token is the MTP's free draft (the two
    /// sources agree on the next word), else the MTP chain.
    Agree,
    /// `NgramFirst`, and an n-gram chain shorter than `k` is extended by the MTP chain walked
    /// through the n-gram tokens (layer only, no lm_head, until the last).
    Merge,
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
    /// Rounds drafted by the n-gram lookup, drafts and accepted of those.
    pub ngram_rounds: u64,
    pub ngram_drafted: u64,
    pub ngram_accepted: u64,
    /// Accepted drafts per round, the histogram `[0 ..= k]`.
    pub accepted_hist: Vec<u64>,
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

/// The cascade's knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpecOptions {
    pub kernel: VerifyKernel,
    pub rows: HeadRows,
    pub source: DraftSource,
    pub ngram: NgramDraft,
    pub mutant: SpecMutant,
    /// A small draft tree instead of the chain ([`generate_tree`]).
    pub tree: Option<TreeSpec>,
}

/// The small tree: the MTP chain of `k` as the trunk, plus the next `w1 - 1` best MTP guesses
/// for the first word and the next `w2 - 1` best for the second word (under the trunk's first
/// word), each side branch continued by an MTP chain of `side` more words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeSpec {
    pub w1: usize,
    pub w2: usize,
    pub side: usize,
}

/// What one tree verify pass returns.
#[derive(Debug, Clone, Default)]
pub struct TreeOut {
    /// Accepted rows, root first (`path.len() - 1` drafts accepted).
    pub path: Vec<usize>,
    /// The model's own token after the path.
    pub next: u32,
    /// The MTP head's best `w1` guesses for the word after `next` (the next tree's first level).
    pub draft_top: Vec<u32>,
    /// Commit time (state rebuild of the path and the K/V rewrite).
    pub commit_ns: u64,
}

/// ONE tree verify pass on the [`VerifyKernel`] boundary: the tree's rows through the model
/// ([`Qwen38Omega::tree_rows`]), the walk from the root (at each node the model's greedy token
/// picks the child, if any), the commit of the accepted path only (rebuild-only-accepted
/// DeltaNet state, K/V rewrite), and the MTP head as layer 65 over the path rows so the next
/// tree's first level comes out of this pass. `committed` is the snapshot at the pass start.
#[allow(clippy::too_many_arguments)]
pub fn verify_tree(
    kernel: VerifyKernel,
    model: &mut Qwen38Omega,
    mtp: &mut Mtp,
    tokens: &[u32],
    shape: &TreeShape,
    committed: &Qwen38Snapshot,
    keep: &mut Vec<GdnRows>,
    kv: &mut Vec<Vec<f32>>,
    w1: usize,
    mutant: SpecMutant,
    st: &mut RowStats,
    st_mtp: &mut RowStats,
) -> Result<TreeOut, String> {
    let VerifyKernel::ExactRows = kernel;
    let h = q::HIDDEN;
    let pos0 = model.pos;
    let hidden = model.tree_rows(tokens, shape, committed, keep, kv, st)?;
    let mut path = vec![0usize];
    let mut node = 0usize;
    let mut extra = mutant == SpecMutant::ExtraAccept;
    let next = loop {
        let g = model.head_token(&hidden[node * h..(node + 1) * h], st)?;
        let child = shape.children(node).find(|&c| tokens[c] == g).or_else(|| {
            if std::mem::take(&mut extra) {
                shape.children(node).find(|&c| tokens[c] != g)
            } else {
                None
            }
        });
        match child {
            Some(c) => {
                path.push(c);
                node = c;
            }
            None => break g,
        }
    };
    let tc = Instant::now();
    if mutant == SpecMutant::NoRollback {
        model.pos = pos0 + path.len();
    } else {
        model.tree_commit(
            committed,
            keep,
            kv,
            &path,
            mutant != SpecMutant::TreeNoKvFix,
            st,
        )?;
    }
    let commit_ns = ns(tc);
    let hid: Vec<f32> = path
        .iter()
        .flat_map(|&r| hidden[r * h..(r + 1) * h].iter().copied())
        .collect();
    let mut own: Vec<u32> = path[1..].iter().map(|&r| tokens[r]).collect();
    own.push(next);
    mtp.layer65(model, &hid, &own, pos0, st_mtp)?;
    mtp.set_tip(path.len() - 1, pos0 + path.len());
    let draft_top = mtp.draft_topk(model, w1, st_mtp)?;
    Ok(TreeOut {
        path,
        next,
        draft_top,
        commit_ns,
    })
}

/// A side branch to draft: its parent row, first token, depth, and the MTP tip it grows from.
type SideBranch = (usize, u32, usize, (Vec<f32>, usize));

/// Greedy generation with a small draft tree per round ([`TreeSpec`], trunk = the MTP chain of
/// `k`), checked in one [`verify_tree`] pass. Same contract as [`generate`]: the tokens equal
/// the plain path's one at a time.
#[allow(clippy::too_many_arguments)]
pub fn generate_tree(
    model: &mut Qwen38Omega,
    mtp: &mut Mtp,
    prompt: &[u32],
    n: usize,
    k: usize,
    stop: &[u32],
    chunk: usize,
    opts: SpecOptions,
    spec: TreeSpec,
) -> Result<SpecRun, String> {
    if prompt.is_empty() || n == 0 {
        return Err("generate_tree: empty prompt or n = 0".into());
    }
    if k == 0 || k > MAX_CHAIN || spec.side > MAX_CHAIN {
        return Err(format!(
            "generate_tree: k {k}, side {} (1..={MAX_CHAIN})",
            spec.side
        ));
    }
    if !(1..=8).contains(&spec.w1) || !(1..=8).contains(&spec.w2) {
        return Err("generate_tree: widths 1..=8".into());
    }
    let h = q::HIDDEN;
    let mutant = opts.mutant;
    let max_depth = k.max(spec.side + 2);
    let mut run = SpecRun {
        accepted_hist: vec![0; max_depth + 1],
        ..SpecRun::default()
    };
    model.reset()?;
    let t0 = Instant::now();
    let mut rows = Vec::with_capacity(prompt.len() * h);
    for c in prompt.chunks(chunk.max(1)) {
        rows.extend(model.exact_rows(c, None, &mut run.main)?);
    }
    let last = rows[(prompt.len() - 1) * h..].to_vec();
    let mut t = model.head_token(&last, &mut run.main)?;
    run.tokens.push(t);
    let mut next_ids = prompt[1..].to_vec();
    next_ids.push(t);
    let td = Instant::now();
    mtp.layer65(model, &rows, &next_ids, 0, &mut run.mtp)?;
    mtp.set_tip(prompt.len() - 1, prompt.len());
    let mut top = mtp.draft_topk(model, spec.w1, &mut run.mtp)?;
    run.draft_ns += ns(td);
    run.prompt_ns = ns(t0);
    drop(rows);

    let t1 = Instant::now();
    let mut snap = Qwen38Snapshot::default();
    let mut keep: Vec<GdnRows> = Vec::new();
    let mut kv: Vec<Vec<f32>> = Vec::new();
    while run.tokens.len() < n && !stop.contains(&t) {
        run.rounds += 1;
        let p = model.pos;
        let room = (model.max_pos() - p).saturating_sub(1).min(max_depth);
        let trunk = room.min(k);
        let td = Instant::now();
        let tip = mtp.tip();
        let mut toks = vec![t];
        let mut parent = vec![0usize];
        let mut l2: Vec<u32> = Vec::new();
        let mut tip1 = None;
        if trunk >= 1 {
            let mut d = top[0];
            toks.push(d);
            parent.push(0);
            for depth in 2..=trunk {
                d = if depth == 2 && spec.w2 > 1 {
                    mtp.chain_layer(model, d, &mut run.mtp)?;
                    tip1 = Some(mtp.tip());
                    let tk = mtp.draft_topk(model, spec.w2, &mut run.mtp)?;
                    l2 = tk[1..].to_vec();
                    tk[0]
                } else {
                    mtp.chain(model, d, &mut run.mtp)?
                };
                parent.push(toks.len() - 1);
                toks.push(d);
            }
        }
        // Side branches: (start row's parent, first token, its depth), each chained `side` on.
        let mut sides: Vec<SideBranch> = Vec::new();
        if let Some(t1s) = &tip1 {
            for &a in &l2 {
                sides.push((1, a, 2, t1s.clone()));
            }
        }
        if room >= 1 {
            for &a in top.iter().skip(1) {
                sides.push((0, a, 1, tip.clone()));
            }
        }
        for (par, a, depth0, from) in sides {
            mtp.set_tip_state(&from);
            parent.push(par);
            toks.push(a);
            let mut prev = a;
            let mut depth = depth0;
            while depth < depth0 + spec.side && depth < room {
                let x = mtp.chain(model, prev, &mut run.mtp)?;
                parent.push(toks.len() - 1);
                toks.push(x);
                prev = x;
                depth += 1;
            }
        }
        run.draft_ns += ns(td);
        let mut shape = TreeShape::from_parents(parent)?;
        shape.mutant_no_rebuild = mutant == SpecMutant::TreeNoRebuild;
        let drafts = toks.len() - 1;
        run.drafted += drafts as u64;

        let ts = Instant::now();
        model.snapshot(&mut snap)?;
        run.snapshot_ns += ns(ts);
        let tv = Instant::now();
        let out = verify_tree(
            opts.kernel,
            model,
            mtp,
            &toks,
            &shape,
            &snap,
            &mut keep,
            &mut kv,
            spec.w1,
            mutant,
            &mut run.main,
            &mut run.mtp,
        )?;
        run.verify_ns += ns(tv) - out.commit_ns;
        run.rollback_ns += out.commit_ns;
        let j = out.path.len() - 1;
        run.accepted += j as u64;
        if j < shape.max_depth() {
            run.rollbacks += 1;
        }
        if let Some(c) = run.accepted_hist.get_mut(j) {
            *c += 1;
        }
        let room_out = n - run.tokens.len();
        run.tokens
            .extend(out.path[1..].iter().map(|&r| toks[r]).take(room_out));
        if run.tokens.len() < n {
            run.tokens.push(out.next);
        }
        if let Some(s) = run.tokens.iter().position(|x| stop.contains(x)) {
            run.tokens.truncate(s + 1);
            break;
        }
        t = out.next;
        top = out.draft_top;
    }
    run.decode_ns = ns(t1);
    Ok(run)
}

/// Greedy generation on the exact rows path: `k = 0` is the plain path one token at a time;
/// `k >= 1` drafts up to `k` tokens per round and checks them in one [`verify_pass`].
///
/// State: one committed DeltaNet state. Before a pass with drafts its tails and states are
/// copied once (`snapshot`); the pass runs every row through the recurrence; a refused draft
/// puts the copy back and replays only the accepted rows from the kept inputs (`replay_gdn`),
/// so the state after a round is bit for bit the state of the accepted tokens one at a time.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut Qwen38Omega,
    mut mtp: Option<&mut Mtp>,
    prompt: &[u32],
    n: usize,
    k: usize,
    stop: &[u32],
    chunk: usize,
    opts: SpecOptions,
) -> Result<SpecRun, String> {
    if prompt.is_empty() || n == 0 {
        return Err("generate: empty prompt or n = 0".into());
    }
    if k > 0 && mtp.is_none() {
        return Err("generate: k > 0 needs the MTP head".into());
    }
    if k > MAX_CHAIN {
        return Err(format!("generate: k {k} > {MAX_CHAIN}"));
    }
    let h = q::HIDDEN;
    let mutant = opts.mutant;
    let mut run = SpecRun {
        accepted_hist: vec![0; k + 1],
        ..SpecRun::default()
    };
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
    let mut history: Vec<u32> = prompt.to_vec();
    history.push(t);

    let t1 = Instant::now();
    let mut snap = Qwen38Snapshot::default();
    let mut keep: Vec<GdnRows> = Vec::new();
    while run.tokens.len() < n && !stop.contains(&t) {
        run.rounds += 1;
        let p = model.pos;
        let room = (model.max_pos() - p).saturating_sub(1).min(k);
        // Drafts: the first MTP draft came out of the last pass; chain the rest, or look the
        // continuation up in the history.
        let td = Instant::now();
        let mut feed = vec![t];
        let mut from_ngram = false;
        if let Some(m) = mtp.as_deref_mut() {
            let Some(d0) = draft.take() else {
                return Err("generate: out of positions for a draft".into());
            };
            let hit = match opts.source {
                DraftSource::Mtp => None,
                _ => opts.ngram.draft(&history, room),
            };
            let use_hit = match (&hit, opts.source) {
                (Some(_), DraftSource::Ngram | DraftSource::NgramFirst | DraftSource::Merge) => {
                    true
                }
                (Some(hit), DraftSource::Agree) => hit.tokens[0] == d0,
                _ => false,
            };
            if use_hit {
                let hit = hit.expect("checked");
                from_ngram = true;
                feed.extend(hit.tokens.iter().copied().take(room));
                if opts.source == DraftSource::Merge && feed.len() < room + 1 {
                    // Walk the head through the n-gram tokens (layer only), then chain.
                    for &tk in &feed[1..] {
                        m.chain_layer(model, tk, &mut run.mtp)?;
                    }
                    let mut d = m.draft_head(model, &mut run.mtp)?;
                    feed.push(d);
                    while feed.len() < room + 1 {
                        d = m.chain(model, d, &mut run.mtp)?;
                        feed.push(d);
                    }
                }
            } else {
                let mut d = d0;
                feed.push(d);
                let want = match opts.source {
                    DraftSource::Ngram => 1,
                    _ => room,
                };
                while feed.len() < want + 1 {
                    d = m.chain(model, d, &mut run.mtp)?;
                    feed.push(d);
                }
            }
            feed.truncate(room + 1);
        }
        run.draft_ns += ns(td);
        let drafts = feed.len() - 1;
        run.drafted += drafts as u64;
        if from_ngram {
            run.ngram_rounds += 1;
            run.ngram_drafted += drafts as u64;
        }

        let ts = Instant::now();
        if drafts > 0 {
            model.snapshot(&mut snap)?;
        }
        run.snapshot_ns += ns(ts);
        let tv = Instant::now();
        let out = verify_pass(
            opts.kernel,
            model,
            if drafts > 0 { mtp.as_deref_mut() } else { None },
            &feed,
            (drafts > 0).then_some(&mut keep),
            opts.rows,
            mutant,
            &mut run.main,
            &mut run.mtp,
        )?;
        run.verify_ns += ns(tv);
        let j = out.accepted;
        let next = out.next;
        run.accepted += j as u64;
        if from_ngram {
            run.ngram_accepted += j as u64;
        }
        if let Some(c) = run.accepted_hist.get_mut(j) {
            *c += 1;
        }
        let emitted = &feed[1..=j];
        let room_out = n - run.tokens.len();
        run.tokens.extend(emitted.iter().take(room_out));
        if run.tokens.len() < n {
            run.tokens.push(next);
        }
        history.extend_from_slice(emitted);
        history.push(next);
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
                draft = match out.draft {
                    Some(d) => Some(d),
                    // A round without drafts (no room) ran no MTP: feed the one row.
                    None => {
                        let td = Instant::now();
                        let d = m.feed(model, &out.hidden[..h], &[next], p, &mut run.mtp)?;
                        run.draft_ns += ns(td);
                        Some(d)
                    }
                };
            }
        }
    }
    run.decode_ns = ns(t1);
    Ok(run)
}

/// Longest chain of drafts one pass checks (`k`).
pub const MAX_CHAIN: usize = 8;

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
    fn tree_shape_preorder_paths_and_children() {
        // root 0; trunk 1-2-3; side under 1: 4-5; side under 0: 6-7.
        let s = TreeShape::from_parents(vec![0, 0, 1, 2, 1, 4, 0, 6]).unwrap();
        assert_eq!(s.depth, vec![0, 1, 2, 3, 2, 3, 1, 2]);
        assert_eq!(s.path(5), vec![0, 1, 4, 5]);
        assert_eq!(s.path(0), vec![0]);
        assert_eq!(s.children(0).collect::<Vec<_>>(), vec![1, 6]);
        assert_eq!(s.children(1).collect::<Vec<_>>(), vec![2, 4]);
        assert_eq!(s.max_depth(), 3);
        // Not preorder: row 3's parent 1 is not an ancestor of row 2 (a child of 0).
        assert!(TreeShape::from_parents(vec![0, 0, 0, 1]).is_err());
        assert!(TreeShape::from_parents(vec![0, 2, 0]).is_err());
        assert!(TreeShape::from_parents(vec![1]).is_err());
        // A chain is a tree.
        let c = TreeShape::from_parents(vec![0, 0, 1, 2]).unwrap();
        assert_eq!(c.depth, vec![0, 1, 2, 3]);
    }

    #[test]
    fn topk_leads_with_argmax_and_breaks_ties_first() {
        let v = [1.0, 5.0, 3.0, 5.0, 4.0];
        assert_eq!(topk(&v, 1), vec![argmax(&v)]);
        assert_eq!(topk(&v, 4), vec![1, 3, 4, 2]);
        assert_eq!(topk(&v, 9).len(), 5);
    }

    #[test]
    fn ngram_finds_the_latest_longest_span() {
        let g = NgramDraft { max_n: 3, min_n: 2 };
        // history: a b c d | a b c e | a b
        let hist = [1, 2, 3, 4, 1, 2, 3, 5, 1, 2];
        let hit = g.draft(&hist, 4).unwrap();
        assert_eq!(hit.n, 2);
        assert_eq!(hit.at, 4);
        assert_eq!(hit.tokens, vec![3, 5, 1, 2]);
        // A 3-span wins over the latest 2-span.
        let hist = [1, 2, 3, 9, 7, 2, 3, 8, 1, 2, 3];
        let hit = g.draft(&hist, 2).unwrap();
        assert_eq!((hit.n, hit.at, hit.tokens), (3, 0, vec![9, 7]));
        // No earlier copy: nothing.
        assert!(g.draft(&[1, 2, 3, 4], 4).is_none());
        // A match whose continuation runs into the suffix is cut at the end of the history.
        let hit = g.draft(&[5, 6, 5, 6], 8).unwrap();
        assert_eq!((hit.n, hit.at, hit.tokens), (2, 0, vec![5, 6]));
        assert!(g.draft(&[1, 2], 0).is_none());
    }
}
