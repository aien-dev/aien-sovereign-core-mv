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
use aien_omega_q38::{
    Attn256Cache, Gdn, GraphConfig, GraphStats, KvForm, Q38Graph, W4Stats, W4Tensor,
};

mod prefill;
pub use prefill::PrefillStats;

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
    /// MXFP4 copies of mix, o, gate_up, down (env `AIEN_Q38_W4=1`, qwen38/w4-decode): the decode graph reads these;
    /// the prefill keeps the FP8 ones.
    w4: Option<[W4Tensor; 4]>,
}

/// The whole model resident on the device plus the host embedding.
///
/// Field order is drop order: the graph goes first, then the tensors and state it points at.
pub struct Qwen38Omega {
    graph: Q38Graph,
    layers: Vec<LayerTensors>,
    lm_head: Bf16Tensor,
    /// FP8 copy of lm_head (env `AIEN_Q38_FP8_HEAD=1`, qwen38/decode-speed): the decode graph reads it instead.
    #[allow(dead_code)] // owned only: the graph points at it until it is dropped first
    lm_head8: Option<Fp8Tensor>,
    /// MXFP4 lm_head (env `AIEN_Q38_W4_HEAD=1`: the FP8-quantised head converted again), read ahead of `lm_head8`.
    #[allow(dead_code)] // owned only: the graph points at it until it is dropped first
    lm_head4: Option<W4Tensor>,
    /// The MXFP4 converter tally over every converted tensor (projections and head), when any.
    pub w4_stats: Option<W4Stats>,
    gdn: Option<Gdn>,
    /// The attention KV cache: bf16, or 8-bit rows with a scale each for K and V (env
    /// `AIEN_Q38_KV8=1`), V only (`2`) or K only (`3`) (SPEED-7 L131; decode only, the prompt
    /// reader refuses any 8-bit side).
    attn: Option<Attn256Cache>,
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

fn gpu(stage: &str) -> impl Fn(aien_omega_gpu::OmegaGpuError) -> String + '_ {
    move |e| format!("{stage}: {e}")
}

/// `1 + w`, the zero-centred RMSNorm weight the graph's norms multiply by.
fn fold(w: Vec<f32>) -> Vec<f32> {
    w.into_iter().map(|v| 1.0 + v).collect()
}

/// Converter threads for the MXFP4 copies (128-row stripes).
const W4_THREADS: usize = 16;

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// The KV cache form from `AIEN_Q38_KV8` (unset or 0 bf16, 1 K and V 8-bit, 2 V only, 3 K only).
fn kv_form_env() -> Result<KvForm, String> {
    let v = std::env::var("AIEN_Q38_KV8").unwrap_or_default();
    KvForm::from_code(&v).ok_or_else(|| {
        format!("AIEN_Q38_KV8={v}: 0 (bf16), 1 (K and V 8-bit), 2 (V only) or 3 (K only)")
    })
}

/// FP8 -> MXFP4 (`stats` Some) of an FP8 projection already on the host, uploaded; the tally adds into `stats`.
fn w4_copy(
    name: &str,
    n: usize,
    k: usize,
    w: &[u8],
    s: &[u16],
    stats: &mut Option<W4Stats>,
) -> Result<Option<W4Tensor>, String> {
    let Some(total) = stats.as_mut() else {
        return Ok(None);
    };
    let (t, st) = W4Tensor::from_fp8(n, k, w, s, W4_THREADS, w4_rule()?).map_err(gpu(name))?;
    total.add(&st);
    Ok(Some(t))
}
/// FP8 projections stacked by rows into one tensor (every part `[n_i][k]`, whole 128 blocks).
fn fuse_fp8(
    name: &str,
    parts: Vec<Matrix>,
    w4: &mut Option<W4Stats>,
) -> Result<(Fp8Tensor, Option<W4Tensor>), String> {
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
    let f8 = Fp8Tensor::upload(1, n, k, &w, &scales).map_err(gpu(name))?;
    Ok((f8, w4_copy(name, n, k, &w, &scales, w4)?))
}

fn upload_fp8(
    name: &str,
    m: Matrix,
    w4: &mut Option<W4Stats>,
) -> Result<(Fp8Tensor, Option<W4Tensor>), String> {
    fuse_fp8(name, vec![m], w4)
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
        // Some: every projection also as MXFP4 (AIEN_Q38_W4=1); the tally of the conversions so far.
        let mut w4l = env_on("AIEN_Q38_W4").then(W4Stats::default);
        let mut w4h = env_on("AIEN_Q38_W4_HEAD").then(W4Stats::default);
        let (lm_head, lm_head8, lm_head4) = match read_matrix(source, "lm_head.weight", v, h)? {
            Matrix::Bf16 { n, k, w } => {
                let (mut head8, mut head4) = (None, None);
                if env_on("AIEN_Q38_FP8_HEAD") || w4h.is_some() {
                    let (w8, s) = aien_omega_q38::quantize_fp8_bf16(n, k, &w, 16)
                        .map_err(gpu("lm_head fp8 quantise"))?;
                    if env_on("AIEN_Q38_FP8_HEAD") {
                        head8 =
                            Some(Fp8Tensor::upload(1, n, k, &w8, &s).map_err(gpu("lm_head fp8"))?);
                    }
                    head4 = w4_copy("lm_head w4", n, k, &w8, &s, &mut w4h)?;
                }
                (
                    Bf16Tensor::upload(n, k, &w).map_err(gpu("lm_head"))?,
                    head8,
                    head4,
                )
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
                Attn256Cache::alloc_kv(n_attn, max_pos, c.rope_theta, c.rotary_dim, kv_form_env()?)
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
            let (mix, ab, o, mix4, o4) = if c.full_attention[l] {
                let a = format!("{p}.self_attn");
                let (mix, mix4) = fuse_fp8(
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
                    &mut w4l,
                )?;
                let (o, o4) = upload_fp8(
                    &format!("{a}.o_proj"),
                    rm(
                        source,
                        format!("{a}.o_proj.weight"),
                        h,
                        c.heads * c.head_dim,
                    )?,
                    &mut w4l,
                )?;
                let q_norm = read_vec(source, &format!("{a}.q_norm.weight"), c.head_dim)?;
                let k_norm = read_vec(source, &format!("{a}.k_norm.weight"), c.head_dim)?;
                attn.as_mut()
                    .expect("an attention layer has a cache")
                    .set_norms(ai, &q_norm, &k_norm)
                    .map_err(gpu("attn256 set_norms"))?;
                ai += 1;
                (mix, None, o, mix4, o4)
            } else {
                let a = format!("{p}.linear_attn");
                let (mix, mix4) = fuse_fp8(
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
                    &mut w4l,
                )?;
                let ia = rm(source, format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?;
                let ib = rm(source, format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?;
                let (an, ak, aw) = ab_rows(&ia, &ib).map_err(|e| format!("{a}: {e}"))?;
                let ab = Bf16Tensor::upload(an, ak, &aw).map_err(gpu("in_proj_a|b"))?;
                ab_mm.push(
                    ResidentTensor::upload_bf16(ak, an, &ab_plain_transposed(an, ak, &aw))
                        .map_err(gpu("in_proj_a|b matmul"))?,
                );
                let (o, o4) = upload_fp8(
                    &format!("{a}.out_proj"),
                    rm(source, format!("{a}.out_proj.weight"), h, c.lin_value_dim())?,
                    &mut w4l,
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
                (mix, Some(ab), o, mix4, o4)
            };
            let m = format!("{p}.mlp");
            let (gate_up, gu4) = fuse_fp8(
                &format!("{m} gate|up"),
                vec![
                    rm(source, format!("{m}.gate_proj.weight"), c.intermediate, h)?,
                    rm(source, format!("{m}.up_proj.weight"), c.intermediate, h)?,
                ],
                &mut w4l,
            )?;
            let (down, dn4) = upload_fp8(
                &format!("{m}.down_proj"),
                rm(source, format!("{m}.down_proj.weight"), h, c.intermediate)?,
                &mut w4l,
            )?;
            let w4 = match (mix4, o4, gu4, dn4) {
                (Some(a), Some(b), Some(g), Some(d)) => Some([a, b, g, d]),
                (None, None, None, None) => None,
                _ => return Err(format!("{p}: only some projections have an MXFP4 copy")),
            };
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
                w4,
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
            if let Some([m4, o4, g4, d4]) = &t.w4 {
                // SAFETY: the MXFP4 tensors live in `layers` too.
                unsafe {
                    graph.set_layer_w4(l, m4.as_raw(), o4.as_raw(), g4.as_raw(), d4.as_raw())
                }
                .map_err(gpu("q38 graph set_layer_w4"))?;
            }
        }
        graph
            .set_final_norm(&final_norm)
            .map_err(gpu("q38 graph set_final_norm"))?;
        if let Some(h8) = &lm_head8 {
            // SAFETY: lm_head8 is owned by the returned value and dropped after the graph (field order).
            unsafe { graph.set_lm_head_fp8(h8.as_raw()) }
                .map_err(gpu("q38 graph set_lm_head_fp8"))?;
        }
        if let Some(h4) = &lm_head4 {
            // SAFETY: lm_head4 is owned by the returned value and dropped after the graph (field order).
            unsafe { graph.set_lm_head_w4(h4.as_raw()) }
                .map_err(gpu("q38 graph set_lm_head_w4"))?;
        }
        let w4_stats = match (w4l, w4h) {
            (Some(mut a), Some(b)) => {
                a.add(&b);
                Some(a)
            }
            (a, b) => a.or(b),
        };
        Ok(Self {
            graph,
            layers,
            lm_head,
            lm_head8,
            lm_head4,
            w4_stats,
            gdn,
            attn,
            config,
            embed,
            pos: 0,
            max_pos,
            norms,
            final_norm,
            ab_mm,
            pf: None,
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
        self.attn.as_ref().map_or(0, |a| {
            a.layers() * Attn256Cache::layer_bytes_form(a.max_pos(), a.kv_form())
        })
    }

    /// True when the attention KV cache has an 8-bit side (`AIEN_Q38_KV8` 1, 2 or 3).
    pub fn kv8(&self) -> bool {
        self.attn.as_ref().is_some_and(Attn256Cache::kv8)
    }

    /// The attention KV cache form (`AIEN_Q38_KV8`); bf16 when the model has no attention layers.
    pub fn kv_form(&self) -> KvForm {
        self.attn
            .as_ref()
            .map_or(KvForm::Bf16, Attn256Cache::kv_form)
    }

    /// BENCHMARK ONLY: fills every attention layer's positions `0..pos` with a bounded pattern
    /// (not a prompt) and moves the next position to `pos`, so decode speed at long context can
    /// be timed without reading a long prompt. The DeltaNet state is left as it is (its size does
    /// not depend on the position). Tokens produced after this are meaningless.
    pub fn bench_seek(&mut self, pos: usize, seed: u32) -> Result<(), String> {
        if pos >= self.max_pos {
            return Err(format!(
                "bench_seek: position {pos}, the KV cache holds {}",
                self.max_pos
            ));
        }
        if pos > 0 {
            if let Some(a) = self.attn.as_mut() {
                a.bench_fill(pos, seed).map_err(gpu("attn256 bench_fill"))?;
            }
        }
        self.pos = pos;
        Ok(())
    }

    /// A new sequence: DeltaNet state to zero, position 0 (attention rows are overwritten as the
    /// new sequence reaches them).
    pub fn reset(&mut self) -> Result<(), String> {
        if let Some(g) = self.gdn.as_mut() {
            g.reset().map_err(gpu("gdn reset"))?;
        }
        self.pos = 0;
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

    /// The graph, for traces and timelines.
    pub fn graph_mut(&mut self) -> &mut Q38Graph {
        &mut self.graph
    }
}

/// The MXFP4 scale rule the loader uses (env `AIEN_Q38_W4_RULE`: `ocp` default, `fit`, `best`; omega
/// `omega_gpu_w4_from_fp8_rule`).
pub fn w4_rule() -> Result<i32, String> {
    match std::env::var("AIEN_Q38_W4_RULE").as_deref() {
        Err(_) | Ok("ocp") => Ok(aien_omega_q38::W4_RULE_OCP),
        Ok("fit") => Ok(aien_omega_q38::W4_RULE_FIT),
        Ok("best") => Ok(aien_omega_q38::W4_RULE_BEST),
        Ok(o) => Err(format!("AIEN_Q38_W4_RULE={o}: ocp, fit or best")),
    }
}

/// Host only (no device): every FP8 projection of the checkpoint converted to MXFP4 under each rule, the converter's tally
/// summed per rule (relative RMS error, clipped weights). Projections are converted one checkpoint tensor at a time.
pub fn w4_error_survey(dir: &Path, rules: &[i32], threads: usize) -> Result<Vec<W4Stats>, String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let h = c.hidden;
    let mut out = vec![W4Stats::default(); rules.len()];
    for l in 0..c.layers {
        let p = format!("{prefix}layers.{l}");
        let names: Vec<(String, usize, usize)> = if c.full_attention[l] {
            let a = format!("{p}.self_attn");
            vec![
                (format!("{a}.q_proj.weight"), 2 * c.heads * c.head_dim, h),
                (format!("{a}.k_proj.weight"), c.kv_dim(), h),
                (format!("{a}.v_proj.weight"), c.kv_dim(), h),
                (format!("{a}.o_proj.weight"), h, c.heads * c.head_dim),
            ]
        } else {
            let a = format!("{p}.linear_attn");
            vec![
                (format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h),
                (format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h),
                (format!("{a}.out_proj.weight"), h, c.lin_value_dim()),
            ]
        };
        let m = format!("{p}.mlp");
        let mlp = [
            (format!("{m}.gate_proj.weight"), c.intermediate, h),
            (format!("{m}.up_proj.weight"), c.intermediate, h),
            (format!("{m}.down_proj.weight"), h, c.intermediate),
        ];
        for (name, n, k) in names.into_iter().chain(mlp) {
            let Matrix::Fp8 { n, k, w, scales } = read_matrix(&mut source, &name, n, k)? else {
                return Err(format!("{name}: not FP8"));
            };
            for (i, &rule) in rules.iter().enumerate() {
                let (_, _, st) = aien_omega_q38::w4_from_fp8(n, k, &w, &scales, threads, rule)
                    .map_err(gpu(&name))?;
                out[i].add(&st);
            }
        }
        eprintln!("w4 survey: layer {l} done");
    }
    Ok(out)
}
