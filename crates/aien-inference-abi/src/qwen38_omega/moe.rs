//! Qwen3.6-35B-A3B (`qwen3_5_moe`) onto the Omega decode graph `aien_omega_q38::q36::Q36Graph`
//! (sc L306). The mixers upload as for the 27B (see the parent module) at the 35B's shapes; the
//! MLP of every layer is the sparse block:
//!
//! - `mlp.experts.{e}.gate_proj | up_proj` for e in 0..256, then `mlp.shared_expert.gate_proj |
//!   up_proj` -> one FP8 tensor of 257 experts `[1024][2048]` (each expert's gate rows, then its up
//!   rows; the 128-blocks concatenate with them). `down_proj` likewise -> 257 x `[2048][512]`.
//! - `mlp.gate.weight` (bf16 `[256][2048]`) with `mlp.shared_expert_gate.weight` (`[1][2048]`) as
//!   row 256 -> the router (`Router::alloc_shape(layers, 256, 1)`), with the post-attention norm
//!   folded to `1 + w`; its directories point at the two expert tensors (entry 256 = the shared
//!   expert, the graph's ninth group, weight sigmoid(shared gate . xn)).
//!
//! The host reference this must match is [`crate::qwen38`] (the `moe` branch).
use std::path::Path;

use aien_omega_bf16::Bf16Tensor;
use aien_omega_fp8::Fp8Tensor;
use aien_omega_q38::q36::{self as q, Q36Config, Q36Graph};
use aien_omega_q38::{Attn256Cache, Gdn};
use aien_omega_router::Router;

use super::rows::{GdnParams, RowSource};
use super::{
    ab_rows, fold, fuse_fp8, gpu, upload_fp8, Graph, HeadState, LayerTensors, Qwen38Omega, EPS_L2,
};
use crate::qwen38::{read_bf16_words, read_matrix, read_vec, text_prefix, Matrix, Qwen38Config};
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::TensorSource;

/// The checkpoint shape the 35B graph is built for (config.json, checked before any upload).
fn check_shape(c: &Qwen38Config) -> Result<(), String> {
    let m = c
        .moe
        .as_ref()
        .ok_or("qwen36: not a qwen3_5_moe checkpoint")?;
    let ok = c.hidden == q::HIDDEN
        && m.experts == 256
        && m.top_k == 8
        && m.expert_inter == q::EXPERT_INTER
        && m.shared_inter == q::EXPERT_INTER
        && c.heads == q::Q_HEADS
        && c.kv_heads == q::KV_HEADS
        && c.head_dim == aien_omega_q38::HEAD_DIM
        && c.heads * c.head_dim == q::MIXER_OUT
        && 2 * c.heads * c.head_dim + 2 * c.kv_dim() == q::ATT_MIX
        && c.conv_dim() == q::GDN_QKV
        && c.conv_dim() + c.lin_value_dim() == q::GDN_MIX
        && c.lin_v_heads == 16 * q::GDN_RATIO
        && c.lin_k_heads == 16
        && c.lin_v_dim == aien_omega_q38::GDN_HEAD_DIM
        && c.conv_kernel == aien_omega_q38::GDN_CONV_KERNEL
        && c.rotary_dim == 64;
    if !ok {
        return Err(
            "qwen36 omega: the checkpoint is not the Qwen3.6-35B-A3B shape the graph is built for"
                .into(),
        );
    }
    if c.layers > aien_omega_q38::MAX_LAYERS {
        return Err(format!(
            "qwen36 omega: {} layers (at most {})",
            c.layers,
            aien_omega_q38::MAX_LAYERS
        ));
    }
    Ok(())
}

/// Appends one FP8 matrix (whole 128-blocks, `[n][k]`) to an expert tensor's bytes and scales.
fn push_fp8(
    name: &str,
    m: Matrix,
    n: usize,
    k: usize,
    w: &mut Vec<u8>,
    s: &mut Vec<u16>,
) -> Result<(), String> {
    match m {
        Matrix::Fp8 {
            n: mn,
            k: mk,
            w: mw,
            scales,
        } if (mn, mk) == (n, k) => {
            w.extend_from_slice(&mw);
            s.extend_from_slice(&scales);
            Ok(())
        }
        Matrix::Fp8 { n: mn, k: mk, .. } => {
            Err(format!("{name}: [{mn}][{mk}], expected [{n}][{k}]"))
        }
        Matrix::Bf16 { .. } => Err(format!("{name}: bf16 (the expert tensors take FP8)")),
    }
}

/// The layer's 257-expert gate|up and down tensors (routed 0..256, then the shared expert).
fn upload_experts(
    source: &mut dyn TensorSource,
    m: &str,
    h: usize,
) -> Result<(Fp8Tensor, Fp8Tensor), String> {
    let i = q::EXPERT_INTER;
    let rows = q::EXPERT_ROWS;
    let (mut gw, mut gs) = (Vec::with_capacity(rows * 2 * i * h), Vec::new());
    let (mut dw, mut ds) = (Vec::with_capacity(rows * h * i), Vec::new());
    for e in 0..rows {
        let p = if e + 1 == rows {
            format!("{m}.shared_expert")
        } else {
            format!("{m}.experts.{e}")
        };
        for part in ["gate_proj", "up_proj"] {
            let name = format!("{p}.{part}.weight");
            push_fp8(
                &name,
                read_matrix(source, &name, i, h)?,
                i,
                h,
                &mut gw,
                &mut gs,
            )?;
        }
        let name = format!("{p}.down_proj.weight");
        push_fp8(
            &name,
            read_matrix(source, &name, h, i)?,
            h,
            i,
            &mut dw,
            &mut ds,
        )?;
    }
    let gu = Fp8Tensor::upload(rows, 2 * i, h, &gw, &gs).map_err(gpu("experts gate|up"))?;
    let dn = Fp8Tensor::upload(rows, h, i, &dw, &ds).map_err(gpu("experts down"))?;
    Ok((gu, dn))
}

/// Router gate rows as bf16 bits: `mlp.gate.weight` `[256][h]` then `shared_expert_gate` `[1][h]`.
fn router_rows(source: &mut dyn TensorSource, m: &str, h: usize) -> Result<Vec<u16>, String> {
    let mut rows = match read_matrix(source, &format!("{m}.gate.weight"), 256, h)? {
        Matrix::Bf16 { w, .. } => w,
        Matrix::Fp8 { .. } => return Err(format!("{m}.gate.weight: FP8 (the router takes bf16)")),
    };
    rows.extend(read_bf16_words(
        source,
        &format!("{m}.shared_expert_gate.weight"),
        &[1, h],
    )?);
    Ok(rows)
}

impl Qwen38Omega {
    /// Opens a Qwen3.6-35B-A3B checkpoint and uploads it (the first `layers` layers when `Some`).
    pub fn open_moe(dir: &Path, layers: Option<usize>, max_pos: usize) -> Result<Self, String> {
        Self::open(dir, layers, max_pos)
    }

    pub(super) fn load_moe(
        config: Qwen38Config,
        first: usize,
        source: &mut dyn TensorSource,
        max_pos: usize,
        heads: usize,
    ) -> Result<Self, String> {
        let c = &config;
        check_shape(c)?;
        if heads == 0 || heads > q::MAX_HEADS {
            return Err(format!(
                "qwen36 omega: {heads} heads (the graph strokes 1..={})",
                q::MAX_HEADS
            ));
        }
        let prefix = text_prefix(source)?;
        let (v, h) = (c.vocab, c.hidden);
        let embed = read_bf16_words(source, &format!("{prefix}embed_tokens.weight"), &[v, h])?;
        let final_norm = fold(read_vec(source, &format!("{prefix}norm.weight"), h)?);
        let head = read_matrix(source, "lm_head.weight", v, h)?;
        let lm_head = match &head {
            Matrix::Bf16 { n, k, w } => Bf16Tensor::upload(*n, *k, w).map_err(gpu("lm_head"))?,
            Matrix::Fp8 { .. } => return Err("lm_head: FP8 (the graph takes a bf16 head)".into()),
        };
        let n_attn = c.full_attention.iter().filter(|&&a| a).count();
        let n_gdn = c.layers - n_attn;
        // One DeltaNet state and one attention cache per head (every head carries the same
        // per-layer parameters, so a head swap can point any slot at any conversation).
        let mut gdns: Vec<Option<Gdn>> = (0..heads)
            .map(|_| {
                (n_gdn > 0)
                    .then(|| Gdn::alloc_ratio(n_gdn, q::GDN_RATIO).map_err(gpu("gdn alloc")))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;
        let mut attns: Vec<Option<Attn256Cache>> = (0..heads)
            .map(|_| {
                (n_attn > 0)
                    .then(|| {
                        Attn256Cache::alloc_shape(
                            n_attn,
                            max_pos,
                            c.rope_theta,
                            c.rotary_dim,
                            q::Q_HEADS,
                            q::KV_HEADS,
                            q::Q_PER_CTA,
                        )
                        .map_err(gpu("attn256 alloc"))
                    })
                    .transpose()
            })
            .collect::<Result<_, _>>()?;
        let mut router = Router::alloc_shape(c.layers, 256, 1).map_err(gpu("router alloc"))?;
        let attn_mask = c
            .full_attention
            .iter()
            .enumerate()
            .fold(0u64, |m, (l, &a)| if a { m | 1 << l } else { m });

        let (mut gdn_params, mut attn_norms) = (Vec::new(), Vec::new());
        let mut layers = Vec::with_capacity(c.layers);
        let mut norms = Vec::with_capacity(c.layers);
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
                for attn in attns.iter_mut() {
                    attn.as_mut()
                        .expect("an attention layer has a cache")
                        .set_norms(ai, &q_norm, &k_norm)
                        .map_err(gpu("attn256 set_norms"))?;
                }
                attn_norms.push((q_norm, k_norm));
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
                for gdn in gdns.iter_mut() {
                    gdn.as_mut()
                        .expect("a DeltaNet layer has state")
                        .set_layer(gi, &conv, &a_log, &dt_bias, &norm)
                        .map_err(gpu("gdn set_layer"))?;
                }
                gdn_params.push(GdnParams {
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                });
                gi += 1;
                (mix, Some(ab), o)
            };
            let m = format!("{p}.mlp");
            let (gate_up, down) = upload_experts(source, &m, h)?;
            let in_norm = fold(read_vec(source, &format!("{p}.input_layernorm.weight"), h)?);
            let post_norm = fold(read_vec(
                source,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
            )?);
            router
                .set_layer(l, &router_rows(source, &m, h)?, &post_norm)
                .map_err(gpu("router set_layer"))?;
            let (mut gu_dir, mut dn_dir) = (
                Vec::with_capacity(2 * q::EXPERT_ROWS),
                Vec::with_capacity(2 * q::EXPERT_ROWS),
            );
            for e in 0..q::EXPERT_ROWS {
                let (w, s) = gate_up.entry(e).map_err(gpu("experts gate|up entry"))?;
                gu_dir.extend([w, s]);
                let (w, s) = down.entry(e).map_err(gpu("experts down entry"))?;
                dn_dir.extend([w, s]);
            }
            router
                .set_directory(l, &gu_dir, &dn_dir)
                .map_err(gpu("router set_directory"))?;
            layers.push(LayerTensors {
                mix,
                ab,
                o,
                gate_up,
                down,
            });
            norms.push((in_norm, post_norm));
        }

        let cfg = Q36Config {
            layers: c.layers as u32,
            attn_mask,
            vocab: v as u32,
            eps: c.eps,
            eps_l2: EPS_L2,
            gdn: gdns[0].as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
            attn: attns[0]
                .as_ref()
                .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
            router: router.as_raw(),
            lm_head: lm_head.as_raw(),
        };
        // SAFETY: every handle is live and owned by the returned struct, which drops the graph first.
        let mut graph =
            unsafe { Q36Graph::create_heads(&cfg, heads) }.map_err(gpu("q36 graph create"))?;
        // L310 merge-train proof line. The omega graph (omega_gpu_q36_graph.c) keeps its chip-only intermediates
        // GPU-L2-cached unless AIEN_OMEGA_Q36_BUF_CACHED is exactly "0"; it reads the switch itself at alloc.
        // Mirror the switch here so the server log states which interior policy this process asked for.
        let interior = match std::env::var("AIEN_OMEGA_Q36_BUF_CACHED") {
            Ok(v) if v == "0" => "GPU-uncached (AIEN_OMEGA_Q36_BUF_CACHED=0)",
            _ => "GPU-cached (default)",
        };
        eprintln!("qwen36 omega: interior buffers {interior}");
        // L310 merge-train proof line (one-wave lever). omega picks the CTAs per group of the grouped expert launch
        // (9 groups: 8 routed + the shared expert) at the first launch: the compiled 6 (measured on 8 groups), an exact
        // AIEN_OMEGA_FP8_GROUPED_CTAS, or AIEN_OMEGA_FP8_GROUPED_WAVE=1 = min(6, 48 SMs / groups), one wave. Asking omega
        // here (same env, same process) logs the number the chip gets; a library without the rule says so.
        let how = if std::env::var("AIEN_OMEGA_FP8_GROUPED_CTAS").is_ok_and(|v| !v.is_empty()) {
            "AIEN_OMEGA_FP8_GROUPED_CTAS"
        } else if std::env::var("AIEN_OMEGA_FP8_GROUPED_WAVE").as_deref() == Ok("1") {
            "wave rule"
        } else {
            "compiled default"
        };
        match aien_omega_fp8::grouped_ctas_auto(q::GROUPS as u32) {
            Some(n) => eprintln!("qwen36 omega: expert grouped launch {n} CTAs per group ({how})"),
            None => eprintln!(
                "qwen36 omega: expert grouped launch: library predates the wave rule ({how} asked)"
            ),
        }
        for h in 1..heads {
            // SAFETY: the states live in `gdns` / `attns`, owned beside the graph (dropped after it).
            unsafe {
                graph.set_head_state(
                    h,
                    gdns[h].as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                    attns[h]
                        .as_ref()
                        .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                )
            }
            .map_err(gpu("q36 graph set_head_state"))?;
        }
        for (l, (t, (in_norm, _))) in layers.iter().zip(&norms).enumerate() {
            let ab =
                t.ab.as_ref()
                    .map_or(std::ptr::null_mut(), Bf16Tensor::as_raw);
            // SAFETY: the tensors live in `layers`, owned beside the graph.
            unsafe {
                graph.set_layer(
                    l,
                    t.mix.as_raw(),
                    ab,
                    t.o.as_raw(),
                    t.gate_up.as_raw(),
                    t.down.as_raw(),
                    in_norm,
                )
            }
            .map_err(gpu("q36 graph set_layer"))?;
        }
        graph
            .set_final_norm(&final_norm)
            .map_err(gpu("q36 graph set_final_norm"))?;
        // HUM L314 exact two-pass head, on by default (AIEN_Q36_EXACT_HEAD=0 keeps the plain head):
        // greedy tokens identical to the full bf16 head by construction (exact_head.rs), half the
        // head's bytes. The engine turns it off for sampled requests.
        let (head_screen, head_coef) = if std::env::var("AIEN_Q36_EXACT_HEAD").as_deref() == Ok("0") {
            eprintln!("qwen36 omega: exact two-pass head OFF (AIEN_Q36_EXACT_HEAD=0)");
            (None, None)
        } else {
            let t = std::time::Instant::now();
            let (codes, scales, coef) = crate::exact_head::chip_screen(&head)?;
            let screen = Fp8Tensor::upload(1, v, h, &codes, &scales).map_err(gpu("exact head screen"))?;
            // SAFETY: the screen is owned by the returned value and dropped after the graph (field order).
            unsafe { graph.set_exact_head(screen.as_raw(), &coef) }.map_err(gpu("q36 graph set_exact_head"))?;
            eprintln!("qwen36 omega: exact two-pass head ON (fp8b screen + Row bound, built in {:.1} s)", t.elapsed().as_secs_f64());
            (Some(screen), Some(coef))
        };
        drop(head);
        // Negative controls for the gates (TEST ONLY): AIEN_Q36_TEST_MUTANT=4 drops the shared expert
        // from COMBINE, 1 shifts the SWIGLU up half, 2 the attention GATHER, 3 drops the attention gate,
        // 5 inverts the exact head's MASK, 6 drops its bound.
        if let Some(m) = graph_mutant() {
            eprintln!("qwen36 omega: TEST MUTANT {m} set (negative control; never serve this)");
            graph.test_set_mutant(m);
        }
        let row_src = RowSource {
            layers: c.layers,
            attn_mask,
            vocab: v,
            eps: c.eps,
            rope_theta: c.rope_theta,
            rotary_dim: c.rotary_dim,
            gdn_params,
            attn_norms,
            mutant: graph_mutant(),
            head_coef,
        };
        let gdn = gdns.remove(0);
        let attn = attns.remove(0);
        let extra_heads = gdns
            .into_iter()
            .zip(attns)
            .map(|(gdn, attn)| HeadState { gdn, attn, pos: 0 })
            .collect();
        Ok(Self {
            graph: Graph::Moe(graph),
            router: Some(router),
            layers,
            lm_head,
            lm_head8: None,
            head_screen,
            gdn,
            attn,
            extra_heads,
            config,
            embed,
            pos: 0,
            max_pos,
            norms,
            final_norm,
            ab_mm: Vec::new(),
            pf: None,
            row_stroke: None,
            rows: Vec::new(),
            row_src: Some(row_src),
            row_steps: 0,
        })
    }
}

/// `AIEN_Q36_TEST_MUTANT` (the gates' negative controls), read once per graph made.
fn graph_mutant() -> Option<i32> {
    std::env::var("AIEN_Q36_TEST_MUTANT")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
}

/// TEST ONLY (L306 router probe): the 35B's router alone on the chip, every layer's gate rows and
/// folded post-attention norm uploaded exactly as [`Qwen38Omega::load_moe`] does, with
/// placeholder directories (the probe reads ids and weights, never the group tables' targets).
pub fn router_probe(dir: &Path) -> Result<(Router, Qwen38Config), String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    check_shape(&c)?;
    let mut source = crate::qwen3_moe_omega::ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let h = c.hidden;
    let mut router = Router::alloc_shape(c.layers, 256, 1).map_err(gpu("router alloc"))?;
    let dir_words: Vec<u64> = (0..2 * q::EXPERT_ROWS as u64)
        .map(|i| 0x1000 + i * 0x100)
        .collect();
    for l in 0..c.layers {
        let p = format!("{prefix}layers.{l}");
        let post_norm = fold(read_vec(
            &mut source,
            &format!("{p}.post_attention_layernorm.weight"),
            h,
        )?);
        router
            .set_layer(
                l,
                &router_rows(&mut source, &format!("{p}.mlp"), h)?,
                &post_norm,
            )
            .map_err(gpu("router set_layer"))?;
        router
            .set_directory(l, &dir_words, &dir_words)
            .map_err(gpu("router set_directory"))?;
    }
    Ok((router, c))
}
