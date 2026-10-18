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
use aien_omega_q38::{Attn256Cache, Gdn, GraphConfig, GraphStats, Q38Graph};

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

/// The whole model resident on the device plus the host embedding.
///
/// Field order is drop order: the graph goes first, then the tensors and state it points at.
pub struct Qwen38Omega {
    graph: Q38Graph,
    layers: Vec<LayerTensors>,
    lm_head: Bf16Tensor,
    gdn: Option<Gdn>,
    attn: Option<Attn256Cache>,
    pub config: Qwen38Config,
    /// bf16 `[vocab][hidden]`.
    embed: Vec<u16>,
    /// Tokens the state holds (the next token's position).
    pub pos: usize,
    max_pos: usize,
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

    /// Uploads `config.layers` layers from `source`.
    pub fn load(
        config: Qwen38Config,
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
        let lm_head = match read_matrix(source, "lm_head.weight", v, h)? {
            Matrix::Bf16 { n, k, w } => Bf16Tensor::upload(n, k, &w).map_err(gpu("lm_head"))?,
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
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..c.layers {
            let p = format!("{prefix}layers.{l}");
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
        Ok(Self {
            graph,
            layers,
            lm_head,
            gdn,
            attn,
            config,
            embed,
            pos: 0,
            max_pos,
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
