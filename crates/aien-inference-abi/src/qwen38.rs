//! Qwen3.8-27B (`qwen3_5`, the hybrid Gated DeltaNet + attention family) host reference.
//!
//! Plan: `~/handoffs/2026-10-18-qwen38-27b-on-omega.md` (step 2). Host only: no launch, no chip.
//! This is the golden model every later Omega kernel for the 27B is checked against, so it is
//! written for clarity and f32 accuracy, not speed. Ground truth for the maths is the model's own
//! published code (transformers `modeling_qwen3_5.py`, text path) and its `config.json`:
//!
//! - 64 layers; `layer_types` marks every 4th (3, 7, ..., 63) `full_attention`, the rest
//!   `linear_attention` (Gated DeltaNet). Each layer: zero-centred RMSNorm, mixer, residual, then
//!   zero-centred RMSNorm, dense SiLU MLP, residual. Zero-centred: `x * rsqrt(mean(x^2) + eps) *
//!   (1 + w)`.
//! - Attention: q_proj gives per head `[q | gate]` (2 * head_dim), q_norm / k_norm (zero-centred)
//!   over head_dim, RoPE on the first `rotary_dim` dims only (rotate-half within them), causal
//!   softmax with scale `1/sqrt(head_dim)`, grouped kv (`q head h` reads `kv head h / group`),
//!   output times `sigmoid(gate)`, o_proj. Text positions make the three mRoPE sections equal, so
//!   they collapse to plain positions.
//! - Gated DeltaNet: in_proj_qkv -> depthwise causal conv1d (kernel 4, no bias) -> SiLU, split
//!   q | k | v; q and k l2-normalised per head (eps 1e-6), q scaled by `1/sqrt(k_dim)`, q and k
//!   heads repeated to the value heads (`v head h` reads `k head h / ratio`). `beta =
//!   sigmoid(in_proj_b x)`, `g = -exp(A_log) * softplus(in_proj_a x + dt_bias)`. Per head the f32
//!   state `S[k_dim][v_dim]`: `S *= exp(g)`, `kv = S^T k`, `delta = (v - kv) * beta`, `S += k
//!   delta^T`, `o = S^T q`. Then a gated RMSNorm per head (`norm(o) * w * silu(z)`, w used as is)
//!   and out_proj. Prefill here runs the same recurrence token by token (the chunked form is equal
//!   to it by algebra).
//! - FP8 checkpoint: every projection is `F8_E4M3` with a bf16 `weight_scale_inv` grid
//!   `[n/128][k/128]`; embedding, lm_head, norms, conv1d, in_proj_a / in_proj_b, A_log and dt_bias
//!   are bf16. Weights are widened to f32 as `e4m3(w) * scale` and every product accumulates in
//!   f32.
//!
//! Memory: the weights are streamed. [`Qwen38Reference::forward`] reads one layer from the shards,
//! pushes every new token through it, drops it and reads the next, so the host holds the bf16
//! embedding and lm_head plus one layer. A pass costs one read of the checkpoint whatever the
//! number of tokens, which is why prompts are scored in one call and not token by token.
//! [`Qwen38State`] carries the attention KV (f32) and the DeltaNet conv tails and states between
//! calls, so a later call continues the same sequence.

use std::path::Path;

use rayon::prelude::*;

use crate::qwen3_moe::{bf16_to_f32, fp8_table, FP8_BLOCK};
use crate::qwen3_moe_omega::{ShardSource, TensorSource};

/// The text model's shape, read from `config.json` (`text_config` of a `qwen3_5` checkpoint).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen38Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub intermediate: usize,
    /// One entry per layer: `true` for `full_attention`, `false` for `linear_attention`.
    pub full_attention: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `head_dim * partial_rotary_factor`: the leading dims RoPE turns.
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub eos_token_ids: Vec<u32>,
    /// `Some` for the MoE variant (`qwen3_5_moe`, Qwen3.6-35B-A3B): every layer's MLP is a routed
    /// sparse block plus a gated shared expert, and `intermediate` is unused (0).
    pub moe: Option<MoeShape>,
}

/// The sparse MLP of a `qwen3_5_moe` layer (transformers `Qwen3_5MoeSparseMoeBlock`): router
/// logits `W_gate x` (bf16 `[experts][hidden]`), softmax over all experts, the `top_k` largest,
/// their probabilities renormalised to sum 1 (always: the router has no `norm_topk_prob` switch);
/// each selected expert is a SiLU MLP of width `expert_inter`. Plus a shared expert (width
/// `shared_inter`) scaled by `sigmoid(shared_expert_gate x)` (bf16 `[1][hidden]`).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeShape {
    pub experts: usize,
    pub top_k: usize,
    pub expert_inter: usize,
    pub shared_inter: usize,
}

fn field<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, String> {
    v.get(key)
        .ok_or_else(|| format!("config.json: missing {key}"))
}

fn usize_field(v: &serde_json::Value, key: &str) -> Result<usize, String> {
    field(v, key)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json: {key} is not an unsigned integer"))
}

fn f64_field(v: &serde_json::Value, key: &str) -> Result<f64, String> {
    field(v, key)?
        .as_f64()
        .ok_or_else(|| format!("config.json: {key} is not a number"))
}

impl Qwen38Config {
    /// Reads `config.json` (and `generation_config.json` for the stop tokens, when present).
    pub fn from_checkpoint(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| format!("{}: {e}", dir.join(name).display()))
        };
        let generation = read("generation_config.json").ok();
        Self::from_json(&read("config.json")?, generation.as_deref())
    }

    pub fn from_json(config_json: &str, generation_json: Option<&str>) -> Result<Self, String> {
        let raw: serde_json::Value =
            serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        let (text, sparse) = match raw.get("model_type").and_then(|v| v.as_str()) {
            Some("qwen3_5") => (field(&raw, "text_config")?, false),
            Some("qwen3_5_text") => (&raw, false),
            Some("qwen3_5_moe") => (field(&raw, "text_config")?, true),
            Some("qwen3_5_moe_text") => (&raw, true),
            other => {
                return Err(format!(
                    "config.json: model_type {other:?}, this reference runs qwen3_5 and qwen3_5_moe only"
                ))
            }
        };
        let moe = if sparse {
            Some(MoeShape {
                experts: usize_field(text, "num_experts")?,
                top_k: usize_field(text, "num_experts_per_tok")?,
                expert_inter: usize_field(text, "moe_intermediate_size")?,
                shared_inter: usize_field(text, "shared_expert_intermediate_size")?,
            })
        } else {
            None
        };
        let tied = text
            .get("tie_word_embeddings")
            .or_else(|| raw.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if tied {
            return Err("config.json: tied embeddings are not supported (27B is untied)".into());
        }
        if text.get("attn_output_gate").and_then(|v| v.as_bool()) != Some(true) {
            return Err("config.json: attn_output_gate must be true".into());
        }
        if text.get("hidden_act").and_then(|v| v.as_str()) != Some("silu") {
            return Err("config.json: hidden_act must be silu".into());
        }
        if let Some(q) = raw.get("quantization_config") {
            let s = |k: &str| q.get(k).and_then(|v| v.as_str());
            let block = q.get("weight_block_size").and_then(|v| v.as_array());
            let block_ok = block.is_none_or(|b| {
                b.len() == 2 && b.iter().all(|x| x.as_u64() == Some(FP8_BLOCK as u64))
            });
            if s("quant_method") != Some("fp8") || s("fmt") != Some("e4m3") || !block_ok {
                return Err(format!(
                    "config.json: quantization_config {q}: only fp8 e4m3 with 128x128 blocks"
                ));
            }
        }
        let rope = field(text, "rope_parameters")?;
        let head_dim = usize_field(text, "head_dim")?;
        let partial = rope
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rotary_dim = (head_dim as f64 * partial) as usize;
        let layers = usize_field(text, "num_hidden_layers")?;
        let full_attention = field(text, "layer_types")?
            .as_array()
            .ok_or("config.json: layer_types is not a list")?
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("linear_attention") => Ok(false),
                other => Err(format!("config.json: layer type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut eos_token_ids = Vec::new();
        if let Some(g) = generation_json {
            let g: serde_json::Value =
                serde_json::from_str(g).map_err(|e| format!("generation_config.json: {e}"))?;
            if let Some(ids) = crate::tokenizer::eos_token_ids_from_json(&g) {
                eos_token_ids = ids;
            }
        }
        if eos_token_ids.is_empty() {
            eos_token_ids.push(usize_field(text, "eos_token_id")? as u32);
        }
        let c = Self {
            hidden: usize_field(text, "hidden_size")?,
            layers,
            vocab: usize_field(text, "vocab_size")?,
            intermediate: if sparse {
                0
            } else {
                usize_field(text, "intermediate_size")?
            },
            full_attention,
            heads: usize_field(text, "num_attention_heads")?,
            kv_heads: usize_field(text, "num_key_value_heads")?,
            head_dim,
            rotary_dim,
            rope_theta: f64_field(rope, "rope_theta")?,
            lin_k_heads: usize_field(text, "linear_num_key_heads")?,
            lin_v_heads: usize_field(text, "linear_num_value_heads")?,
            lin_k_dim: usize_field(text, "linear_key_head_dim")?,
            lin_v_dim: usize_field(text, "linear_value_head_dim")?,
            conv_kernel: usize_field(text, "linear_conv_kernel_dim")?,
            eps: f64_field(text, "rms_norm_eps")? as f32,
            eos_token_ids,
            moe,
        };
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| Err(format!("qwen3_5 config: {what}"));
        if self.full_attention.len() != self.layers {
            return bad("layer_types length differs from num_hidden_layers");
        }
        if !self.heads.is_multiple_of(self.kv_heads)
            || !self.lin_v_heads.is_multiple_of(self.lin_k_heads)
        {
            return bad("head counts do not group evenly");
        }
        if !self.rotary_dim.is_multiple_of(2) || self.rotary_dim > self.head_dim {
            return bad("rotary_dim must be even and at most head_dim");
        }
        if self.conv_kernel < 1 {
            return bad("conv kernel must be at least 1");
        }
        if let Some(m) = &self.moe {
            if m.top_k == 0 || m.top_k > m.experts {
                return bad("num_experts_per_tok must be 1..=num_experts");
            }
        }
        Ok(())
    }

    /// The first `layers` layers of the model (a truncated stack for parity checks: the final norm
    /// and lm_head then read layer `layers - 1`'s output). `None` when `layers` is 0 or too many.
    pub fn prefix(&self, layers: usize) -> Option<Self> {
        if layers == 0 || layers > self.layers {
            return None;
        }
        let mut c = self.clone();
        c.layers = layers;
        c.full_attention.truncate(layers);
        Some(c)
    }

    /// Layers `first .. first + count` of the model as a stack of their own (a window for parity
    /// checks that fit the device one window at a time). `None` when empty or past the end.
    pub fn window(&self, first: usize, count: usize) -> Option<Self> {
        if count == 0 || first + count > self.layers {
            return None;
        }
        let mut c = self.clone();
        c.layers = count;
        c.full_attention = self.full_attention[first..first + count].to_vec();
        Some(c)
    }

    /// Width of the DeltaNet conv input: q | k | v.
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }

    pub fn lin_value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

// ---------------------------------------------------------------------------
// Matrices: the checkpoint's `[n][k]` rows, widened to f32 a block of rows at a time.
// ---------------------------------------------------------------------------

/// One projection as stored: e4m3 with its block scales, or bf16.
#[derive(Debug)]
pub enum Matrix {
    Fp8 {
        n: usize,
        k: usize,
        w: Vec<u8>,
        scales: Vec<u16>,
    },
    Bf16 {
        n: usize,
        k: usize,
        w: Vec<u16>,
    },
    /// 4-bit (L355): e2m1 nibbles `[n][k/2]` and scale bytes: `global` None = MXFP4, E8M0 per 32 `[n][k/32]` (the bytes the
    /// chip uploads); Some = nv16, e4m3 per 16 `[n][k/16]`, and one f32 global per row (its linear's), applied to each
    /// output once after the dot, as the chip's epilogue would (q36_w4_nv, host arm).
    W4 {
        n: usize,
        k: usize,
        w4: Vec<u8>,
        e8: Vec<u8>,
        global: Option<Vec<f32>>,
    },
}

/// Rows per parallel task: one task widens this many rows and runs every token through them.
const ROWS_PER_TASK: usize = 16;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent lanes so the compiler can keep it in vector registers.
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

impl Matrix {
    pub fn n(&self) -> usize {
        match self {
            Self::Fp8 { n, .. } | Self::Bf16 { n, .. } | Self::W4 { n, .. } => *n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Self::Fp8 { k, .. } | Self::Bf16 { k, .. } | Self::W4 { k, .. } => *k,
        }
    }

    /// Rows `r0..r0 + rows` as f32 into `dst` (`[rows][k]`).
    fn widen_rows(&self, r0: usize, rows: usize, lut: &[f32; 256], dst: &mut [f32]) {
        match self {
            Self::Fp8 { k, w, scales, .. } => {
                let kb = k / FP8_BLOCK;
                for r in 0..rows {
                    let row = r0 + r;
                    let src = &w[row * k..(row + 1) * k];
                    let srow = &scales[(row / FP8_BLOCK) * kb..(row / FP8_BLOCK + 1) * kb];
                    let out = &mut dst[r * k..(r + 1) * k];
                    for (b, (o, s)) in out
                        .chunks_exact_mut(FP8_BLOCK)
                        .zip(src.chunks_exact(FP8_BLOCK))
                        .enumerate()
                    {
                        let scale = bf16_to_f32(srow[b]);
                        for (o, &q) in o.iter_mut().zip(s) {
                            *o = lut[q as usize] * scale;
                        }
                    }
                }
            }
            Self::Bf16 { k, w, .. } => {
                for (o, &b) in dst[..rows * k].iter_mut().zip(&w[r0 * k..(r0 + rows) * k]) {
                    *o = bf16_to_f32(b);
                }
            }
            Self::W4 { k, w4, e8, global, .. } => {
                for r in 0..rows {
                    let o = &mut dst[r * k..(r + 1) * k];
                    match global {
                        None => aien_omega_q38::w4::widen_row(w4, e8, *k, r0 + r, o),
                        // without the global: matmul scales each output (exact e2m1 x e4m3 products here)
                        Some(_) => crate::q36_w4_nv::widen_row(w4, e8, 1.0, *k, r0 + r, o),
                    }
                }
            }
        }
    }

    /// `y[t][r] = sum_c W[r][c] * x[t][c]` for `t` rows of `x` (`[t][k]`), returned `[t][n]`.
    pub fn matmul(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (n, k) = (self.n(), self.k());
        assert_eq!(
            x.len(),
            t * k,
            "matmul [{n}][{k}] given {} values for {t} rows",
            x.len()
        );
        let row_scale = |r: usize| match self {
            Self::W4 { global: Some(g), .. } => g[r],
            _ => 1.0,
        };
        let lut = fp8_table();
        // Worked as `[n][t]` so each task owns a contiguous run of output rows.
        let mut by_row = vec![0f32; n * t];
        by_row
            .par_chunks_mut(ROWS_PER_TASK * t)
            .enumerate()
            .for_each_init(Vec::new, |buf: &mut Vec<f32>, (task, out)| {
                let r0 = task * ROWS_PER_TASK;
                let rows = out.len() / t;
                buf.resize(rows * k, 0.0);
                self.widen_rows(r0, rows, &lut, buf);
                for tt in 0..t {
                    let xv = &x[tt * k..(tt + 1) * k];
                    for r in 0..rows {
                        out[r * t + tt] = dot(&buf[r * k..(r + 1) * k], xv) * row_scale(r0 + r);
                    }
                }
            });
        if t == 1 {
            return by_row;
        }
        let mut y = vec![0f32; t * n];
        for r in 0..n {
            for tt in 0..t {
                y[tt * n + r] = by_row[r * t + tt];
            }
        }
        y
    }
}

fn bytes_of_words(words: &mut [u16]) -> &mut [u8] {
    // SAFETY: u8 has no alignment need and every byte pattern is a valid u16 half; the length is
    // exactly the slice's size in bytes. The checkpoint's little-endian words match this host
    // (aarch64 and x86_64 are little-endian).
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 2) }
}

pub(crate) fn read_bf16_words(
    source: &mut dyn TensorSource,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u16>, String> {
    let mut words = vec![0u16; shape.iter().product()];
    source.read_into(name, "BF16", shape, bytes_of_words(&mut words))?;
    Ok(words)
}

pub(crate) fn read_vec(
    source: &mut dyn TensorSource,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    Ok(read_bf16_words(source, name, &[len])?
        .into_iter()
        .map(bf16_to_f32)
        .collect())
}

/// Reads `name` as `[n][k]`: e4m3 with `name_scale_inv` when the header says `F8_E4M3`, else
/// bf16. An e4m3 weight without its scale grid is refused.
pub(crate) fn read_matrix(
    source: &mut dyn TensorSource,
    name: &str,
    n: usize,
    k: usize,
) -> Result<Matrix, String> {
    match source.dtype_of(name)?.as_deref() {
        Some("F8_E4M3") => {
            if !n.is_multiple_of(FP8_BLOCK) || !k.is_multiple_of(FP8_BLOCK) {
                return Err(format!(
                    "{name}: [{n}][{k}] is not a whole number of 128 blocks"
                ));
            }
            let scale_name = format!("{name}_scale_inv");
            if source.dtype_of(&scale_name)?.is_none() {
                return Err(format!("{name}: F8_E4M3 without {scale_name}"));
            }
            let mut w = vec![0u8; n * k];
            source.read_into(name, "F8_E4M3", &[n, k], &mut w)?;
            let scales = read_bf16_words(source, &scale_name, &[n / FP8_BLOCK, k / FP8_BLOCK])?;
            Ok(Matrix::Fp8 { n, k, w, scales })
        }
        Some("BF16") => Ok(Matrix::Bf16 {
            n,
            k,
            w: read_bf16_words(source, name, &[n, k])?,
        }),
        Some(other) => Err(format!("{name}: dtype {other} (expected F8_E4M3 or BF16)")),
        None => Err(format!("{name}: not in the checkpoint")),
    }
}

/// `{p}.gate_proj` / `up_proj` (`[inter][h]`) and `down_proj` (`[h][inter]`).
fn read_ffn(source: &mut dyn TensorSource, p: &str, inter: usize, h: usize) -> Result<Ffn, String> {
    Ok(Ffn {
        gate: read_matrix(source, &format!("{p}.gate_proj.weight"), inter, h)?,
        up: read_matrix(source, &format!("{p}.up_proj.weight"), inter, h)?,
        down: read_matrix(source, &format!("{p}.down_proj.weight"), h, inter)?,
        corr: None,
    })
}

/// Expert `e` of the layer whose mlp prefix is `m` from the 4-bit slabs (block e: gate rows then up rows; down block e).
fn w4_ffn(w: &W4Host, m: &str, e: usize, inter: usize, h: usize) -> Result<Ffn, String> {
    let gu = format!("{m}.experts gate|up");
    Ok(Ffn {
        gate: w.rows(&gu, e * 2 * inter, inter)?,
        up: w.rows(&gu, e * 2 * inter + inter, inter)?,
        down: w.rows(&format!("{m}.experts down"), e * h, h)?,
        corr: None,
    })
}

/// [`w4_ffn`] with [`W4Host::fp8_part`] applied: the kept matrices come from the FP8 checkpoint at `p` (`{p}.gate_proj`, ..).
/// With [`W4Host::lowrank`] (arm m-dn only) the down projection of layer `l`, slot `e` gets its rank-r correction.
fn w4_mixed_ffn(w: &W4Host, src: &mut dyn TensorSource, m: &str, p: &str, l: usize, e: usize, inter: usize, h: usize) -> Result<Ffn, String> {
    let gu = format!("{m}.experts gate|up");
    let down = format!("{m}.experts down");
    let mut f = match w.fp8_part {
        1 => Ffn {
            gate: read_matrix(src, &format!("{p}.gate_proj.weight"), inter, h)?,
            up: read_matrix(src, &format!("{p}.up_proj.weight"), inter, h)?,
            down: w.rows(&down, e * h, h)?,
            corr: None,
        },
        2 => Ffn {
            gate: w.rows(&gu, e * 2 * inter, inter)?,
            up: w.rows(&gu, e * 2 * inter + inter, inter)?,
            down: read_matrix(src, &format!("{p}.down_proj.weight"), h, inter)?,
            corr: None,
        },
        _ => w4_ffn(w, m, e, inter, h)?,
    };
    if let Some(lr) = &w.lowrank {
        if w.fp8_part != 1 {
            return Err("lowrank correction is defined for arm m-dn only (fp8_part 1)".into());
        }
        let s = lr.shape;
        if s.h != h || s.k != inter {
            return Err(format!("lowrank sidecar {s:?} does not match down [{h}][{inter}]"));
        }
        let (u, v) = lr.get(l, e)?;
        f.corr = Some(LowRankTerm { r: s.r, u, v });
    }
    Ok(f)
}

/// The router's choice for one row of logits: softmax over every expert, the `k` largest
/// probabilities (larger logit first, the lower id on ties), renormalised to sum 1. `mutant`
/// (negative control): 1 swaps the k-th pick for the (k+1)-th; 2 keeps the weights but sends
/// every pick to the next expert id (a wrong expert).
pub fn route(logits: &[f32], k: usize, mutant: u8) -> Vec<(usize, f32)> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let mut pick: Vec<usize> = order[..k].to_vec();
    if mutant == 1 && k < order.len() {
        pick[k - 1] = order[k];
    }
    let sel: f32 = pick.iter().map(|&e| probs[e] / sum).sum();
    let n = logits.len();
    pick.into_iter()
        .map(|e| {
            (
                if mutant == 2 { (e + 1) % n } else { e },
                probs[e] / sum / sel,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One layer's weights, read on demand.
// ---------------------------------------------------------------------------

struct DeltaNetWeights {
    in_qkv: Matrix,
    in_z: Matrix,
    in_a: Matrix,
    in_b: Matrix,
    /// `[conv_dim][conv_kernel]`.
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    out: Matrix,
}

struct AttentionWeights {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

enum Mixer {
    DeltaNet(DeltaNetWeights),
    Attention(AttentionWeights),
}

struct LayerWeights {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    mlp: Mlp,
}

/// One SiLU MLP: `down(silu(gate x) * up x)`, plus `U (V a)` on the down input when a low-rank correction is attached.
struct Ffn {
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    corr: Option<LowRankTerm>,
}

/// U `[h][r]`, V `[r][inter]` (q36_w4_lowrank sidecar record, bf16 values).
struct LowRankTerm {
    r: usize,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl Ffn {
    fn run(&self, xn: &[f32], t: usize) -> Vec<f32> {
        self.down_out(&self.act(xn, t), t)
    }

    /// The down projection of `act` (`[t][inter]`), with the correction if any.
    fn down_out(&self, act: &[f32], t: usize) -> Vec<f32> {
        let mut y = self.down.matmul(act, t);
        if let Some(c) = &self.corr {
            let k = act.len() / t;
            let h = y.len() / t;
            for (a, yr) in act.chunks_exact(k).zip(y.chunks_exact_mut(h)) {
                let z: Vec<f32> = c.v.chunks_exact(k).map(|vr| dot(vr, a)).collect();
                for (o, ur) in yr.iter_mut().zip(c.u.chunks_exact(c.r)) {
                    *o += dot(ur, &z);
                }
            }
        }
        y
    }

    /// `silu(gate x) * up x`, the down projection's input (`[t][inter]`).
    fn act(&self, xn: &[f32], t: usize) -> Vec<f32> {
        let gate = self.gate.matmul(xn, t);
        let up = self.up.matmul(xn, t);
        gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect()
    }
}

/// The sparse block's always-resident part (the experts are read per call, only those routed).
struct MoeWeights {
    /// bf16 `[experts][hidden]`.
    router: Matrix,
    /// `shared_expert_gate`, `[hidden]`.
    shared_gate: Vec<f32>,
    shared: Ffn,
}

enum Mlp {
    Dense(Ffn),
    Moe(MoeWeights),
}

/// Per layer state between calls.
#[derive(Debug, Clone)]
enum LayerState {
    /// `conv`: the last `conv_kernel - 1` conv inputs, oldest first (`[kernel-1][conv_dim]`);
    /// `s`: the state, `[v_heads][k_dim][v_dim]`.
    DeltaNet { conv: Vec<f32>, s: Vec<f32> },
    /// Keys after norm and RoPE, values as projected, `[pos][kv_dim]` each.
    Attention { k: Vec<f32>, v: Vec<f32> },
}

/// Everything a later call needs to continue the sequence.
#[derive(Debug, Clone)]
pub struct Qwen38State {
    pub pos: usize,
    layers: Vec<LayerState>,
}

impl Qwen38State {
    pub fn new(c: &Qwen38Config) -> Self {
        let layers = c
            .full_attention
            .iter()
            .map(|&full| {
                if full {
                    LayerState::Attention {
                        k: Vec::new(),
                        v: Vec::new(),
                    }
                } else {
                    LayerState::DeltaNet {
                        conv: vec![0.0; (c.conv_kernel - 1) * c.conv_dim()],
                        s: vec![0.0; c.lin_v_heads * c.lin_k_dim * c.lin_v_dim],
                    }
                }
            })
            .collect();
        Self { pos: 0, layers }
    }
}

// ---------------------------------------------------------------------------
// Small pieces of maths.
// ---------------------------------------------------------------------------

/// Zero-centred RMSNorm over each `w.len()` run of `x`: `x * rsqrt(mean(x^2) + eps) * (1 + w)`.
fn rms_norm_zc(x: &mut [f32], w: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(w.len()) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(w) {
            *v = *v * inv * (1.0 + g);
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// PyTorch's `softplus` (beta 1, threshold 20).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `x / sqrt(sum(x^2) + eps)` in place.
fn l2_normalize(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

/// Rotate-half RoPE on the first `rotary` dims of `x` at position `pos`.
fn rope(x: &mut [f32], pos: usize, rotary: usize, theta: f64) {
    let half = rotary / 2;
    for j in 0..half {
        let inv_freq = 1.0 / theta.powf((2 * j) as f64 / rotary as f64);
        let (sin, cos) = (pos as f64 * inv_freq).sin_cos();
        let (sin, cos) = (sin as f32, cos as f32);
        let (x1, x2) = (x[j], x[j + half]);
        x[j] = x1 * cos - x2 * sin;
        x[j + half] = x2 * cos + x1 * sin;
    }
}

// ---------------------------------------------------------------------------
// The reference.
// ---------------------------------------------------------------------------

/// Where the logits come from in a [`Qwen38Reference::forward`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    /// Only the last new token (generation).
    Last,
    /// Every new token (scoring a text).
    All,
}

/// Where the text model's tensors live: the multimodal checkpoint nests it under
/// `model.language_model.`, a text-only one under `model.`.
pub fn text_prefix(source: &mut dyn TensorSource) -> Result<String, String> {
    ["model.language_model.", "model."]
        .into_iter()
        .find(|p| {
            matches!(
                source.dtype_of(&format!("{p}embed_tokens.weight")),
                Ok(Some(_))
            )
        })
        .map(str::to_string)
        .ok_or_else(|| {
            "checkpoint: no embed_tokens.weight under model.language_model. or model.".into()
        })
}

pub struct Qwen38Reference<S: TensorSource> {
    pub config: Qwen38Config,
    source: S,
    prefix: String,
    /// bf16 `[vocab][hidden]`, widened per row at lookup.
    embed: Vec<u16>,
    final_norm: Vec<f32>,
    lm_head: Matrix,
    /// TEST ONLY negative control for the MoE router, see [`route`]: 0 off (default), 1 the
    /// `top_k + 1`-th expert in place of the `top_k`-th, 2 every pick sent to the next expert id.
    pub routing_mutant: u8,
    /// MoE only: per layer, the routed expert ids of the last call, `[t][top_k]` flattened.
    pub routes: Vec<Vec<u16>>,
    /// MoE only: per layer, the router logit gap between the `top_k`-th and `top_k + 1`-th
    /// expert of each row of the last call (how close the pick was to flipping).
    pub route_gaps: Vec<Vec<f32>>,
    /// TEST ONLY noise-floor probe (L306): `(where, rel)` multiplies each element of the MoE input
    /// row by `1 + rel * u` (u in [-1, 1], fixed per layer and index). `where` 0 off (default),
    /// 1 only the router's input, 2 only the experts' and shared expert's input.
    pub jitter: (u8, f32),
    /// TEST ONLY (L306 router probe): when set, [`Self::moe_x`] keeps every MoE layer's input
    /// residual (before the post-attention norm, `[t][hidden]`) of the last call.
    pub moe_capture: bool,
    pub moe_x: Vec<Vec<f32>>,
    /// MoE only: per layer, the routed weights `[t][top_k]` and the shared expert gate
    /// sigmoid(shared_gate . xn) `[t]` of the last call (beside [`Self::routes`]).
    pub route_wts: Vec<Vec<f32>>,
    pub shared_g: Vec<Vec<f32>>,
    /// L355: 4-bit tensors read from a store in place of the FP8 ones ([`Self::set_w4`]).
    w4: Option<W4Host>,
    /// Calibration statistics for the error-feedback 4-bit rounding (L355): when set, every MoE call adds its inputs.
    pub calib: Option<crate::q36_w4_gptq::Calib>,
    /// L355: the MTP head's own weights, read on first use ([`Self::mtp_step`]).
    mtp: Option<MtpWeights>,
}

struct MtpWeights {
    fc: Matrix,
    norm_e: Vec<f32>,
    norm_h: Vec<f32>,
    norm: Vec<f32>,
}

/// The MTP head's attention cache (its one layer); `truncate` drops speculative rows.
#[derive(Default, Clone)]
pub struct MtpState {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl MtpState {
    /// Rows held (positions written).
    pub fn rows(&self, c: &Qwen38Config) -> usize {
        self.k.len() / c.kv_dim()
    }
    pub fn truncate(&mut self, c: &Qwen38Config, rows: usize) {
        self.k.truncate(rows * c.kv_dim());
        self.v.truncate(rows * c.kv_dim());
    }
}

/// Which tensors the host reference reads from the 4-bit store (L355), and the calibrated mutant (every e2m1 mantissa bit
/// cleared: a 3-bit grid), the negative control of docs/inference/q36/w4-gate.md.
pub struct W4Host {
    pub store: std::sync::Arc<crate::q36_w4_store::W4Store>,
    pub experts: bool,
    pub trunk: bool,
    pub mutant: bool,
    /// Layers kept at FP8 (bit l), q36_w4_store::fp8_layers_from_env (AIEN_Q36_W4_FP8_LAYERS): the mixed arm.
    pub fp8_layers: u64,
    /// The shared expert stays FP8 (AIEN_Q36_W4_SHARED_FP8=1, host only: the chip slab holds it as group 256).
    pub shared_fp8: bool,
    /// Diagnostic control for nv16 (w4-gate.md "NV16 arm"): every e4m3 scale byte's mantissa cleared, so the scales are powers of
    /// two at group 16: tells whether the gain comes from the scale mantissa or from the group size.
    pub scale_mutant: bool,
    /// Mixed inside the expert (w4-gate.md "NV16 arm" item 4): 1 keeps gate and up at FP8 (arm m-dn), 2 keeps down at FP8
    /// (arm m-gu); 0 takes every expert matrix from the store.
    pub fp8_part: u8,
    /// Rank-r correction of every down projection (w4-gate.md "Low-rank arm", AIEN_Q36_W4_LOWRANK): arm m-dn only.
    pub lowrank: Option<std::sync::Arc<crate::q36_w4_lowrank::LowRank>>,
}

impl W4Host {
    fn on(&self, l: usize) -> bool {
        l >= 64 || self.fp8_layers >> l & 1 == 0
    }
    fn rows(&self, name: &str, row0: usize, rows: usize) -> Result<Matrix, String> {
        let (_, k) = self.store.shape(name).ok_or_else(|| format!("w4 store: no tensor {name}"))?;
        let (w4, e8) = self.store.read_rows(name, row0, rows)?;
        let w4 = if self.mutant { w4.into_iter().map(|b| b & 0xEE).collect() } else { w4 };
        let e8 = if self.scale_mutant { e8.into_iter().map(|b| if b >= 8 { b & 0xF8 } else { b }).collect() } else { e8 };
        let global = self.store.globals(name).map(|(g, per)| (row0..row0 + rows).map(|r| g[r / per]).collect());
        Ok(Matrix::W4 { n: rows, k, w4, e8, global })
    }
}

impl Qwen38Reference<ShardSource> {
    /// Opens a checkpoint directory (`config.json`, `model.safetensors.index.json` and shards).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let config = Qwen38Config::from_checkpoint(dir)?;
        Self::new(config, ShardSource::open(dir)?)
    }
}

impl<S: TensorSource> Qwen38Reference<S> {
    /// Reads the embedding, final norm and lm_head; layers are read per call.
    pub fn new(config: Qwen38Config, mut source: S) -> Result<Self, String> {
        let prefix = text_prefix(&mut source)?;
        let (v, h) = (config.vocab, config.hidden);
        let embed = read_bf16_words(
            &mut source,
            &format!("{prefix}embed_tokens.weight"),
            &[v, h],
        )?;
        let final_norm = read_vec(&mut source, &format!("{prefix}norm.weight"), h)?;
        let lm_head = read_matrix(&mut source, "lm_head.weight", v, h)?;
        Ok(Self {
            config,
            source,
            prefix,
            embed,
            final_norm,
            lm_head,
            routing_mutant: 0,
            routes: Vec::new(),
            route_gaps: Vec::new(),
            jitter: (0, 0.0),
            moe_capture: false,
            moe_x: Vec::new(),
            route_wts: Vec::new(),
            shared_g: Vec::new(),
            w4: None,
            calib: None,
            mtp: None,
        })
    }

    /// L355: read the experts (routed + shared) and/or the trunk (mix + o) from a 4-bit store from now on.
    pub fn set_w4(&mut self, w4: Option<W4Host>) {
        self.w4 = w4;
    }

    /// L355: one step of the checkpoint's MTP head (Qwen3.5 layout, the 27B's spec decode defaults: embedding half
    /// first, `h_prev` the residual stream before the final norm, position `pos` = the position of `h_prev`):
    /// `x = fc([norm_e(embed(next)) ; norm_h(h_prev)])`, the MTP decoder layer (full attention with its own cache in
    /// `st`, then the sparse block), logits = lm_head(mtp.norm(x)). Returns `(logits, x)`; `x` feeds a chained draft.
    pub fn mtp_step(&mut self, st: &mut MtpState, h_prev: &[f32], next: u32, pos: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let c = self.config.clone();
        let h = c.hidden;
        if h_prev.len() != h || next as usize >= c.vocab {
            return Err("mtp_step: hidden width or token out of range".into());
        }
        if self.mtp.is_none() {
            let src: &mut dyn TensorSource = &mut self.source;
            self.mtp = Some(MtpWeights {
                fc: read_matrix(src, "mtp.fc.weight", h, 2 * h)?,
                norm_e: read_vec(src, "mtp.pre_fc_norm_embedding.weight", h)?,
                norm_h: read_vec(src, "mtp.pre_fc_norm_hidden.weight", h)?,
                norm: read_vec(src, "mtp.norm.weight", h)?,
            });
        }
        let m = self.mtp.as_ref().expect("loaded above");
        let mut e: Vec<f32> = self.embed[next as usize * h..(next as usize + 1) * h].iter().map(|&b| bf16_to_f32(b)).collect();
        rms_norm_zc(&mut e, &m.norm_e, c.eps);
        let mut hn = h_prev.to_vec();
        rms_norm_zc(&mut hn, &m.norm_h, c.eps);
        e.extend_from_slice(&hn);
        let mut x = m.fc.matmul(&e, 1);
        let norm = m.norm.clone();
        let w = self.read_layer(c.layers)?;
        let mut xn = x.clone();
        rms_norm_zc(&mut xn, &w.input_norm, c.eps);
        let mixed = match &w.mixer {
            Mixer::Attention(a) => attention(&c, a, &xn, 1, pos, &mut st.k, &mut st.v),
            Mixer::DeltaNet(_) => return Err("mtp layer: not full attention".into()),
        };
        x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
        let mut xn = x.clone();
        rms_norm_zc(&mut xn, &w.post_norm, c.eps);
        let down = match &w.mlp {
            Mlp::Moe(mw) => self.moe(c.layers, mw, &xn, 1)?,
            Mlp::Dense(f) => f.run(&xn, 1),
        };
        x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
        let mut rows = x.clone();
        rms_norm_zc(&mut rows, &norm, c.eps);
        Ok((self.lm_head.matmul(&rows, 1), x))
    }

    pub fn new_state(&self) -> Qwen38State {
        Qwen38State::new(&self.config)
    }

    fn read_layer(&mut self, l: usize) -> Result<LayerWeights, String> {
        let c = self.config.clone();
        // L355: layer index `layers` is the MTP head's one decoder layer (full attention + the sparse block, `mtp.layers.0`)
        let mtp = l >= c.layers;
        let p = if mtp { "mtp.layers.0".to_string() } else { format!("{}layers.{l}", self.prefix) };
        let src: &mut dyn TensorSource = &mut self.source;
        let h = c.hidden;
        let mixer = if mtp || c.full_attention[l] {
            let a = format!("{p}.self_attn");
            Mixer::Attention(AttentionWeights {
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
            })
        } else {
            let a = format!("{p}.linear_attn");
            let conv_shape = [c.conv_dim(), 1, c.conv_kernel];
            Mixer::DeltaNet(DeltaNetWeights {
                in_qkv: read_matrix(src, &format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
                in_z: read_matrix(src, &format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h)?,
                in_a: read_matrix(src, &format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?,
                in_b: read_matrix(src, &format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?,
                conv: read_bf16_words(src, &format!("{a}.conv1d.weight"), &conv_shape)?
                    .into_iter()
                    .map(bf16_to_f32)
                    .collect(),
                a_log: read_vec(src, &format!("{a}.A_log"), c.lin_v_heads)?,
                dt_bias: read_vec(src, &format!("{a}.dt_bias"), c.lin_v_heads)?,
                norm: read_vec(src, &format!("{a}.norm.weight"), c.lin_v_dim)?,
                out: read_matrix(src, &format!("{a}.out_proj.weight"), h, c.lin_value_dim())?,
            })
        };
        let mut mixer = mixer;
        if let Some(w) = self.w4.as_ref().filter(|w| w.trunk && !mtp && w.on(l)) {
            // the store's fused tensors stack the checkpoint parts by rows (q36_w4_store::layer_projs)
            match &mut mixer {
                Mixer::Attention(a) => {
                    let (nm, qn, kv) = (format!("{p}.self_attn q|k|v"), 2 * c.heads * c.head_dim, c.kv_dim());
                    a.q = w.rows(&nm, 0, qn)?;
                    a.k = w.rows(&nm, qn, kv)?;
                    a.v = w.rows(&nm, qn + kv, kv)?;
                    a.o = w.rows(&format!("{p}.self_attn.o_proj"), 0, h)?;
                }
                Mixer::DeltaNet(d) => {
                    let nm = format!("{p}.linear_attn qkv|z");
                    d.in_qkv = w.rows(&nm, 0, c.conv_dim())?;
                    d.in_z = w.rows(&nm, c.conv_dim(), c.lin_value_dim())?;
                    d.out = w.rows(&format!("{p}.linear_attn.out_proj"), 0, h)?;
                }
            }
        }
        let m = format!("{p}.mlp");
        let mlp = match &c.moe {
            None => Mlp::Dense(read_ffn(src, &m, c.intermediate, h)?),
            Some(s) => Mlp::Moe(MoeWeights {
                router: read_matrix(src, &format!("{m}.gate.weight"), s.experts, h)?,
                shared_gate: read_bf16_words(
                    src,
                    &format!("{m}.shared_expert_gate.weight"),
                    &[1, h],
                )?
                .into_iter()
                .map(bf16_to_f32)
                .collect(),
                shared: match self.w4.as_ref().filter(|w| w.experts && !mtp && w.on(l) && !w.shared_fp8) {
                    Some(w) => w4_mixed_ffn(w, src, &m, &format!("{m}.shared_expert"), l, s.experts, s.shared_inter, h)?,
                    None => read_ffn(src, &format!("{m}.shared_expert"), s.shared_inter, h)?,
                },
            }),
        };
        Ok(LayerWeights {
            input_norm: read_vec(src, &format!("{p}.input_layernorm.weight"), h)?,
            post_norm: read_vec(src, &format!("{p}.post_attention_layernorm.weight"), h)?,
            mixer,
            mlp,
        })
    }

    /// The sparse block over `t` rows of `xn` (post-attention normed): routes every row, reads
    /// each routed expert of layer `l` once and adds `weight * expert(x)` plus the gated shared
    /// expert. Records the routed ids in [`Self::routes`].
    fn moe(&mut self, l: usize, w: &MoeWeights, xn: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let s = self.config.moe.clone().ok_or("moe: dense config")?;
        let h = self.config.hidden;
        let (jw, rel) = self.jitter;
        let jittered: Vec<f32> = if jw == 0 {
            Vec::new()
        } else {
            xn.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let mut z =
                        ((l as u64) << 32 | (i % h) as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    z ^= z >> 31;
                    let u = (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                    v * (1.0 + rel * u)
                })
                .collect()
        };
        let (xr, xn) = match jw {
            1 => (&jittered[..], xn),
            2 => (xn, &jittered[..]),
            _ => (xn, xn),
        };
        let logits = w.router.matmul(xr, t);
        let mut picks: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t);
        let mut gaps = Vec::with_capacity(t);
        for row in logits.chunks_exact(s.experts) {
            picks.push(route(row, s.top_k, self.routing_mutant));
            let mut v = row.to_vec();
            v.sort_by(|a, b| b.total_cmp(a));
            gaps.push(if s.top_k < v.len() {
                v[s.top_k - 1] - v[s.top_k]
            } else {
                f32::INFINITY
            });
        }
        if self.routes.len() <= l {
            self.routes.resize(l + 1, Vec::new());
            self.route_gaps.resize(l + 1, Vec::new());
            self.route_wts.resize(l + 1, Vec::new());
            self.shared_g.resize(l + 1, Vec::new());
        }
        self.route_wts[l] = picks
            .iter()
            .flat_map(|p| p.iter().map(|&(_, wt)| wt))
            .collect();
        self.route_gaps[l] = gaps;
        self.routes[l] = picks
            .iter()
            .flat_map(|p| p.iter().map(|&(e, _)| e as u16))
            .collect();
        let mut users: Vec<Vec<(usize, f32)>> = vec![Vec::new(); s.experts];
        for (i, p) in picks.iter().enumerate() {
            for &(e, wt) in p {
                users[e].push((i, wt));
            }
        }
        let mut out = match self.calib.as_mut() {
            Some(c) => {
                c.add_gu(l, xn, t);
                let act = w.shared.act(xn, t);
                let mut wa = act.clone();
                for (i, r) in wa.chunks_exact_mut(s.shared_inter).enumerate() {
                    let g = sigmoid(dot(&w.shared_gate, &xn[i * h..(i + 1) * h]));
                    r.iter_mut().for_each(|v| *v *= g);
                }
                c.add_dn(l, s.experts, &wa, t);
                w.shared.down_out(&act, t)
            }
            None => w.shared.run(xn, t),
        };
        let mut gates = Vec::with_capacity(t);
        for (i, row) in out.chunks_exact_mut(h).enumerate() {
            let g = sigmoid(dot(&w.shared_gate, &xn[i * h..(i + 1) * h]));
            row.iter_mut().for_each(|v| *v *= g);
            gates.push(g);
        }
        self.shared_g[l] = gates;
        let mtp = l >= self.config.layers; // the MTP head's layer (L355): its experts are not in the 4-bit store
        let p = if mtp { "mtp.layers.0.mlp.experts".to_string() } else { format!("{}layers.{l}.mlp.experts", self.prefix) };
        for (e, rows) in users.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let f = match self.w4.as_ref().filter(|w| w.experts && !mtp && w.on(l)) {
                Some(w) => w4_mixed_ffn(w, &mut self.source, &format!("{}layers.{l}.mlp", self.prefix), &format!("{p}.{e}"), l, e, s.expert_inter, h)?,
                None => read_ffn(&mut self.source, &format!("{p}.{e}"), s.expert_inter, h)?,
            };
            let mut xs = Vec::with_capacity(rows.len() * h);
            for &(i, _) in rows {
                xs.extend_from_slice(&xn[i * h..(i + 1) * h]);
            }
            let y = match self.calib.as_mut() {
                Some(c) => {
                    let act = f.act(&xs, rows.len());
                    let mut wa = act.clone();
                    for (r, &(_, wt)) in wa.chunks_exact_mut(s.expert_inter).zip(rows) {
                        r.iter_mut().for_each(|v| *v *= wt);
                    }
                    c.add_dn(l, e, &wa, rows.len());
                    f.down_out(&act, rows.len())
                }
                None => f.run(&xs, rows.len()),
            };
            for (yr, &(i, wt)) in y.chunks_exact(h).zip(rows) {
                for (o, &v) in out[i * h..(i + 1) * h].iter_mut().zip(yr) {
                    *o += wt * v;
                }
            }
        }
        Ok(out)
    }

    /// Runs `tokens` after what `state` already holds and returns logits, `[rows][vocab]` with
    /// `rows` 1 for [`Logits::Last`] and `tokens.len()` for [`Logits::All`].
    pub fn forward(
        &mut self,
        state: &mut Qwen38State,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<Vec<f32>, String> {
        self.forward_tapped(state, tokens, logits, &[], &mut Vec::new())
    }

    /// [`Self::forward`] that also copies the residual stream entering layer `b` (`[t][hidden]`,
    /// before its input norm) into `taps` for every `b` in `at` (ascending; `b == layers` is the
    /// residual before the final norm). The chip checks a model window by window against these.
    pub fn forward_tapped(
        &mut self,
        state: &mut Qwen38State,
        tokens: &[u32],
        logits: Logits,
        at: &[usize],
        taps: &mut Vec<Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let c = self.config.clone();
        let (t, h) = (tokens.len(), c.hidden);
        if t == 0 {
            return Err("forward: no tokens".into());
        }
        let mut x = vec![0f32; t * h];
        for (row, &tok) in x.chunks_exact_mut(h).zip(tokens) {
            let tok = tok as usize;
            if tok >= c.vocab {
                return Err(format!(
                    "forward: token {tok} outside the vocab {}",
                    c.vocab
                ));
            }
            for (o, &b) in row.iter_mut().zip(&self.embed[tok * h..(tok + 1) * h]) {
                *o = bf16_to_f32(b);
            }
        }
        if at.windows(2).any(|p| p[0] >= p[1]) || at.iter().any(|&b| b > c.layers) {
            return Err(format!(
                "forward: taps {at:?} not ascending within 0..={}",
                c.layers
            ));
        }
        for l in 0..c.layers {
            if at.contains(&l) {
                taps.push(x.clone());
            }
            let w = self.read_layer(l)?;
            let mut xn = x.clone();
            rms_norm_zc(&mut xn, &w.input_norm, c.eps);
            let mixed = match (&w.mixer, &mut state.layers[l]) {
                (Mixer::DeltaNet(d), LayerState::DeltaNet { conv, s }) => {
                    delta_net(&c, d, &xn, t, conv, s)
                }
                (Mixer::Attention(a), LayerState::Attention { k, v }) => {
                    attention(&c, a, &xn, t, state.pos, k, v)
                }
                _ => return Err(format!("layer {l}: state kind differs from the weights")),
            };
            x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
            if self.moe_capture && matches!(w.mlp, Mlp::Moe(_)) {
                if self.moe_x.len() <= l {
                    self.moe_x.resize(l + 1, Vec::new());
                }
                self.moe_x[l] = x.clone();
            }
            let mut xn = x.clone();
            rms_norm_zc(&mut xn, &w.post_norm, c.eps);
            let down = match &w.mlp {
                Mlp::Dense(f) => f.run(&xn, t),
                Mlp::Moe(m) => self.moe(l, m, &xn, t)?,
            };
            x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
        }
        if at.contains(&c.layers) {
            taps.push(x.clone());
        }
        state.pos += t;
        let rows = match logits {
            Logits::Last => &x[(t - 1) * h..],
            Logits::All => &x[..],
        };
        let mut rows = rows.to_vec();
        rms_norm_zc(&mut rows, &self.final_norm, c.eps);
        Ok(self.lm_head.matmul(&rows, rows.len() / h))
    }
}

/// One DeltaNet mixer over `t` new tokens; updates the conv tail and the state.
fn delta_net(
    c: &Qwen38Config,
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

    // Causal depthwise conv over [tail | new], then SiLU.
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
    // Normalised q and k per token and k head.
    let mut q = vec![0f32; t * key_dim];
    let mut k = vec![0f32; t * key_dim];
    let q_scale = 1.0 / (kd as f32).sqrt();
    for i in 0..t {
        let row = &mixed[i * cd..(i + 1) * cd];
        q[i * key_dim..(i + 1) * key_dim].copy_from_slice(&row[..key_dim]);
        k[i * key_dim..(i + 1) * key_dim].copy_from_slice(&row[key_dim..2 * key_dim]);
        for hh in 0..kh {
            let r = i * key_dim + hh * kd..i * key_dim + (hh + 1) * kd;
            l2_normalize(&mut q[r.clone()], 1e-6);
            l2_normalize(&mut k[r.clone()], 1e-6);
            q[r].iter_mut().for_each(|v| *v *= q_scale);
        }
    }

    // The recurrence, one value head per task.
    let mut core = vec![0f32; t * vh * vd];
    let mut per_head: Vec<Vec<f32>> = vec![Vec::new(); vh];
    state
        .par_chunks_mut(kd * vd)
        .zip(per_head.par_iter_mut())
        .enumerate()
        .for_each(|(hv, (s, out))| {
            let hk = hv / ratio;
            let decay_rate = -w.a_log[hv].exp();
            out.resize(t * vd, 0.0);
            let mut kv = vec![0f32; vd];
            for i in 0..t {
                let kt = &k[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
                let qt = &q[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
                let vt =
                    &mixed[i * cd + 2 * key_dim + hv * vd..i * cd + 2 * key_dim + (hv + 1) * vd];
                let g = decay_rate * softplus(a[i * vh + hv] + w.dt_bias[hv]);
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
    for (hv, out) in per_head.iter().enumerate() {
        for i in 0..t {
            core[i * vh * vd + hv * vd..i * vh * vd + (hv + 1) * vd]
                .copy_from_slice(&out[i * vd..(i + 1) * vd]);
        }
    }

    // Gated RMSNorm per head (weight used as is), times silu(z).
    for (o, zr) in core.chunks_exact_mut(vd).zip(z.chunks_exact(vd)) {
        let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
        let inv = 1.0 / (ms + c.eps).sqrt();
        for ((v, &g), &zz) in o.iter_mut().zip(&w.norm).zip(zr) {
            *v = *v * inv * g * silu(zz);
        }
    }
    w.out.matmul(&core, t)
}

/// One gated full-attention mixer over `t` new tokens starting at position `pos0`; appends their
/// keys and values to the cache.
fn attention(
    c: &Qwen38Config,
    w: &AttentionWeights,
    xn: &[f32],
    t: usize,
    pos0: usize,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
) -> Vec<f32> {
    let (nh, hd, kvd) = (c.heads, c.head_dim, c.kv_dim());
    let group = nh / c.kv_heads;
    let qg = w.q.matmul(xn, t);
    let mut k = w.k.matmul(xn, t);
    let v = w.v.matmul(xn, t);
    // Split per head [q | gate]; norm and turn q and k.
    let mut q = vec![0f32; t * nh * hd];
    let mut gate = vec![0f32; t * nh * hd];
    for i in 0..t {
        for hh in 0..nh {
            let src = &qg[(i * nh + hh) * 2 * hd..(i * nh + hh + 1) * 2 * hd];
            let dq = &mut q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            dq.copy_from_slice(&src[..hd]);
            rms_norm_zc(dq, &w.q_norm, c.eps);
            rope(dq, pos0 + i, c.rotary_dim, c.rope_theta);
            gate[(i * nh + hh) * hd..(i * nh + hh + 1) * hd].copy_from_slice(&src[hd..]);
        }
        for hh in 0..c.kv_heads {
            let dk = &mut k[i * kvd + hh * hd..i * kvd + (hh + 1) * hd];
            rms_norm_zc(dk, &w.k_norm, c.eps);
            rope(dk, pos0 + i, c.rotary_dim, c.rope_theta);
        }
    }
    k_cache.extend_from_slice(&k);
    v_cache.extend_from_slice(&v);
    let (kc, vc) = (&k_cache[..], &v_cache[..]);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; t * nh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
        let (i, hh) = (idx / nh, idx % nh);
        let kh = hh / group;
        let qv = &q[idx * hd..(idx + 1) * hd];
        let span = pos0 + i + 1;
        let scores: Vec<f32> = (0..span)
            .map(|p| dot(qv, &kc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd]) * scale)
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = weights.iter().sum();
        for (p, &wt) in weights.iter().enumerate() {
            let vrow = &vc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd];
            for (ov, &vv) in o.iter_mut().zip(vrow) {
                *ov += wt * vv;
            }
        }
        let g = &gate[idx * hd..(idx + 1) * hd];
        for (ov, &gv) in o.iter_mut().zip(g) {
            *ov = *ov / sum * sigmoid(gv);
        }
    });
    w.o.matmul(&out, t)
}

/// `-log softmax(logits)[target]` for one row.
pub fn negative_log_likelihood(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    max + sum.ln() - logits[target] as f64
}

/// Index of the largest logit (first one on ties).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_moe::{f32_to_bf16, f32_to_fp8_e4m3fn};
    use std::collections::HashMap;

    /// Deterministic pseudo-random values in [-1, 1).
    fn prand(seed: u64, idx: usize) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(idx as u64 + 1);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        ((x & 0xFFFF) as f32 / 32768.0) - 1.0
    }

    #[derive(Default)]
    struct Mem {
        tensors: HashMap<String, (String, Vec<usize>, Vec<u8>)>,
        seed: u64,
    }

    impl TensorSource for Mem {
        fn read_into(
            &mut self,
            name: &str,
            dtype: &str,
            shape: &[usize],
            dst: &mut [u8],
        ) -> Result<(), String> {
            let (d, s, b) = self.tensors.get(name).ok_or(format!("{name}: missing"))?;
            if d != dtype || s != shape || b.len() != dst.len() {
                return Err(format!("{name}: {d} {s:?} asked {dtype} {shape:?}"));
            }
            dst.copy_from_slice(b);
            Ok(())
        }
        fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
            Ok(self.tensors.get(name).map(|t| t.0.clone()))
        }
    }

    impl Mem {
        fn bf16(&mut self, name: &str, shape: &[usize], scale: f32, offset: f32) {
            self.seed += 1;
            let n: usize = shape.iter().product();
            let bytes = (0..n)
                .flat_map(|i| f32_to_bf16(offset + prand(self.seed, i) * scale).to_le_bytes())
                .collect();
            self.tensors
                .insert(name.into(), ("BF16".into(), shape.to_vec(), bytes));
        }
        fn fp8(&mut self, name: &str, n: usize, k: usize) {
            self.seed += 1;
            let w = (0..n * k)
                .map(|i| f32_to_fp8_e4m3fn(prand(self.seed, i) * 8.0))
                .collect();
            self.tensors
                .insert(name.into(), ("F8_E4M3".into(), vec![n, k], w));
            let grid = [n / FP8_BLOCK, k / FP8_BLOCK];
            let s = (0..grid[0] * grid[1])
                .flat_map(|i| f32_to_bf16(0.01 + 0.005 * prand(self.seed ^ 7, i)).to_le_bytes())
                .collect();
            self.tensors.insert(
                format!("{name}_scale_inv"),
                ("BF16".into(), grid.to_vec(), s),
            );
        }
    }

    fn tiny() -> Qwen38Config {
        Qwen38Config {
            hidden: 256,
            layers: 4,
            vocab: 64,
            intermediate: 384,
            full_attention: vec![false, false, false, true],
            heads: 4,
            kv_heads: 2,
            head_dim: 128,
            rotary_dim: 32,
            rope_theta: 1e7,
            lin_k_heads: 2,
            lin_v_heads: 4,
            lin_k_dim: 64,
            lin_v_dim: 64,
            conv_kernel: 4,
            eps: 1e-6,
            eos_token_ids: vec![1],
            moe: None,
        }
    }

    fn tiny_moe() -> Qwen38Config {
        Qwen38Config {
            intermediate: 0,
            moe: Some(MoeShape {
                experts: 8,
                top_k: 3,
                expert_inter: 128,
                shared_inter: 256,
            }),
            ..tiny()
        }
    }

    fn tiny_source(c: &Qwen38Config) -> Mem {
        let mut m = Mem::default();
        let (h, p) = (c.hidden, "model.language_model.");
        m.bf16(&format!("{p}embed_tokens.weight"), &[c.vocab, h], 1.0, 0.0);
        m.bf16(&format!("{p}norm.weight"), &[h], 0.1, 0.0);
        m.bf16("lm_head.weight", &[c.vocab, h], 0.1, 0.0);
        for l in 0..c.layers {
            let lp = format!("{p}layers.{l}");
            m.bf16(&format!("{lp}.input_layernorm.weight"), &[h], 0.1, 0.0);
            m.bf16(
                &format!("{lp}.post_attention_layernorm.weight"),
                &[h],
                0.1,
                0.0,
            );
            if c.full_attention[l] {
                let a = format!("{lp}.self_attn");
                m.fp8(&format!("{a}.q_proj.weight"), 2 * c.heads * c.head_dim, h);
                m.fp8(&format!("{a}.k_proj.weight"), c.kv_dim(), h);
                m.fp8(&format!("{a}.v_proj.weight"), c.kv_dim(), h);
                m.fp8(&format!("{a}.o_proj.weight"), h, c.heads * c.head_dim);
                m.bf16(&format!("{a}.q_norm.weight"), &[c.head_dim], 0.1, 0.0);
                m.bf16(&format!("{a}.k_norm.weight"), &[c.head_dim], 0.1, 0.0);
            } else {
                let a = format!("{lp}.linear_attn");
                m.fp8(&format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h);
                m.fp8(&format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h);
                m.bf16(
                    &format!("{a}.in_proj_a.weight"),
                    &[c.lin_v_heads, h],
                    0.05,
                    0.0,
                );
                m.bf16(
                    &format!("{a}.in_proj_b.weight"),
                    &[c.lin_v_heads, h],
                    0.05,
                    0.0,
                );
                m.bf16(
                    &format!("{a}.conv1d.weight"),
                    &[c.conv_dim(), 1, c.conv_kernel],
                    0.5,
                    0.0,
                );
                m.bf16(&format!("{a}.A_log"), &[c.lin_v_heads], 0.5, 0.0);
                m.bf16(&format!("{a}.dt_bias"), &[c.lin_v_heads], 0.5, 0.0);
                m.bf16(&format!("{a}.norm.weight"), &[c.lin_v_dim], 0.1, 1.0);
                m.fp8(&format!("{a}.out_proj.weight"), h, c.lin_value_dim());
            }
            let mp = format!("{lp}.mlp");
            let mut ffn = |m: &mut Mem, p: &str, inter: usize| {
                m.fp8(&format!("{p}.gate_proj.weight"), inter, h);
                m.fp8(&format!("{p}.up_proj.weight"), inter, h);
                m.fp8(&format!("{p}.down_proj.weight"), h, inter);
            };
            match &c.moe {
                None => ffn(&mut m, &mp, c.intermediate),
                Some(s) => {
                    m.bf16(&format!("{mp}.gate.weight"), &[s.experts, h], 0.5, 0.0);
                    m.bf16(
                        &format!("{mp}.shared_expert_gate.weight"),
                        &[1, h],
                        0.2,
                        0.0,
                    );
                    ffn(&mut m, &format!("{mp}.shared_expert"), s.shared_inter);
                    for e in 0..s.experts {
                        ffn(&mut m, &format!("{mp}.experts.{e}"), s.expert_inter);
                    }
                }
            }
        }
        m
    }

    #[test]
    fn reads_the_27b_config() {
        let json = r#"{"model_type":"qwen3_5","tie_word_embeddings":false,
          "quantization_config":{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]},
          "text_config":{"attn_output_gate":true,"hidden_act":"silu","hidden_size":5120,
          "num_hidden_layers":4,"vocab_size":248320,"intermediate_size":17408,
          "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
          "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25},
          "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
          "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,
          "eos_token_id":248044}}"#;
        let c = Qwen38Config::from_json(json, Some(r#"{"eos_token_id":[248046,248044]}"#)).unwrap();
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.conv_dim(), 10240);
        assert_eq!(c.full_attention, vec![false, false, false, true]);
        assert_eq!(c.eos_token_ids, vec![248046, 248044]);
        let tied = json.replace(
            "\"tie_word_embeddings\":false",
            "\"tie_word_embeddings\":true",
        );
        assert!(Qwen38Config::from_json(&tied, None).is_err());
        let int8 = json.replace("\"fp8\"", "\"int8\"");
        assert!(Qwen38Config::from_json(&int8, None).is_err());
    }

    #[test]
    fn fp8_matmul_matches_the_scalar_formula() {
        let mut m = Mem::default();
        m.fp8("w", 256, 384);
        let w = read_matrix(&mut m, "w", 256, 384).unwrap();
        let x: Vec<f32> = (0..3 * 384).map(|i| prand(9, i)).collect();
        let y = w.matmul(&x, 3);
        let Matrix::Fp8 { w: q, scales, .. } = &w else {
            panic!()
        };
        let lut = fp8_table();
        for t in 0..3 {
            for r in 0..256 {
                let want: f64 = (0..384)
                    .map(|c| {
                        let s = bf16_to_f32(scales[(r / 128) * 3 + c / 128]) as f64;
                        lut[q[r * 384 + c] as usize] as f64 * s * x[t * 384 + c] as f64
                    })
                    .sum();
                let got = y[t * 256 + r] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{t} {r} {got} {want}"
                );
            }
        }
    }

    #[test]
    fn rope_turns_only_the_leading_dims() {
        let mut x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        rope(&mut x, 0, 4, 1e4);
        assert_eq!(x, (0..8).map(|i| i as f32 + 1.0).collect::<Vec<_>>());
        rope(&mut x, 3, 4, 1e4);
        // j = 0: frequency 1, angle 3; pairs (x0, x2).
        let (s, c) = 3f32.sin_cos();
        assert!((x[0] - (1.0 * c - 3.0 * s)).abs() < 1e-6);
        assert!((x[2] - (3.0 * c + 1.0 * s)).abs() < 1e-6);
        assert_eq!(&x[4..], &[5.0, 6.0, 7.0, 8.0]);
    }

    /// Scoring a sequence in one call and running it token by token through the carried state
    /// must give the same logits: the conv tails, DeltaNet states, KV cache and positions all
    /// carry over correctly.
    #[test]
    fn one_call_equals_token_by_token() {
        let c = tiny();
        let mut whole = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        let mut steps = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        let tokens = [5u32, 17, 3, 42, 8, 8, 60];
        let mut s1 = whole.new_state();
        let all = whole.forward(&mut s1, &tokens, Logits::All).unwrap();
        let mut s2 = steps.new_state();
        // A two-token prefix, then single tokens.
        let mut rows = steps.forward(&mut s2, &tokens[..2], Logits::All).unwrap();
        for &tok in &tokens[2..] {
            rows.extend(steps.forward(&mut s2, &[tok], Logits::Last).unwrap());
        }
        assert_eq!(s1.pos, tokens.len());
        assert_eq!(s2.pos, tokens.len());
        let max_diff = all
            .iter()
            .zip(&rows)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "max logit difference {max_diff}");
        // And the logits are not degenerate.
        let spread = all.iter().cloned().fold(f32::MIN, f32::max)
            - all.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread > 1e-3, "logit spread {spread}");
    }

    #[test]
    fn refuses_an_fp8_weight_without_scales() {
        let mut m = Mem::default();
        m.fp8("w", 128, 128);
        m.tensors.remove("w_scale_inv");
        assert!(read_matrix(&mut m, "w", 128, 128)
            .unwrap_err()
            .contains("without"));
    }

    #[test]
    fn nll_and_argmax() {
        let l = [0.0f32, 2.0, 1.0];
        assert_eq!(argmax(&l), 1);
        let z: f64 = l.iter().map(|&v| (v as f64).exp()).sum();
        assert!((negative_log_likelihood(&l, 2) - (z.ln() - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn reads_the_35b_config() {
        let json = r#"{"model_type":"qwen3_5_moe","tie_word_embeddings":false,
          "quantization_config":{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]},
          "text_config":{"attn_output_gate":true,"hidden_act":"silu","hidden_size":2048,
          "num_hidden_layers":4,"vocab_size":248320,"moe_intermediate_size":512,
          "shared_expert_intermediate_size":512,"num_experts":256,"num_experts_per_tok":8,
          "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
          "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25},
          "linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,
          "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,
          "eos_token_id":248044}}"#;
        let c = Qwen38Config::from_json(json, None).unwrap();
        assert_eq!(c.conv_dim(), 8192);
        assert_eq!(c.lin_value_dim(), 4096);
        assert_eq!(c.kv_dim(), 512);
        assert_eq!(c.intermediate, 0);
        assert_eq!(
            c.moe,
            Some(MoeShape {
                experts: 256,
                top_k: 8,
                expert_inter: 512,
                shared_inter: 512
            })
        );
        let bad = json.replace("\"num_experts_per_tok\":8", "\"num_experts_per_tok\":300");
        assert!(Qwen38Config::from_json(&bad, None).is_err());
    }

    #[test]
    fn route_renormalises_and_breaks_ties_low() {
        let l = [0.5f32, 2.0, 2.0, -1.0, 1.0];
        let r = route(&l, 3, 0);
        assert_eq!(r.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 4]);
        let s: f32 = r.iter().map(|p| p.1).sum();
        assert!((s - 1.0).abs() < 1e-6);
        let (e2, e1) = (2f32.exp(), 1f32.exp());
        assert!((r[0].1 - e2 / (2.0 * e2 + e1)).abs() < 1e-6);
        let m = route(&l, 3, 1);
        assert_eq!(m.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 0]);
    }

    #[test]
    fn moe_one_call_equals_token_by_token() {
        let c = tiny_moe();
        let mut whole = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        let mut steps = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        let tokens = [5u32, 17, 3, 42, 8, 8, 60];
        let mut s1 = whole.new_state();
        let all = whole.forward(&mut s1, &tokens, Logits::All).unwrap();
        let mut s2 = steps.new_state();
        let mut rows = steps.forward(&mut s2, &tokens[..2], Logits::All).unwrap();
        for &tok in &tokens[2..] {
            rows.extend(steps.forward(&mut s2, &[tok], Logits::Last).unwrap());
        }
        let max_diff = all
            .iter()
            .zip(&rows)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-4, "max logit difference {max_diff}");
        assert_eq!(whole.routes.len(), c.layers);
        assert_eq!(whole.routes[0].len(), tokens.len() * 3);
        // More than one expert set in use (the router is not degenerate).
        let mut sets: Vec<Vec<u16>> = whole.routes[0].chunks(3).map(|s| s.to_vec()).collect();
        sets.dedup();
        assert!(sets.len() > 1, "{sets:?}");
        // The routing mutant moves the logits.
        let mut mutant = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        mutant.routing_mutant = 1;
        let mut s3 = mutant.new_state();
        let bad = mutant.forward(&mut s3, &tokens, Logits::All).unwrap();
        let moved = all
            .iter()
            .zip(&bad)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(moved > 1e-3, "mutant moved the logits by only {moved}");
    }

    /// The sparse block against a brute-force f64 evaluation from the raw tensors.
    #[test]
    fn moe_block_matches_brute_force() {
        let c = tiny_moe();
        let s = c.moe.clone().unwrap();
        let mut src = tiny_source(&c);
        let mut r = Qwen38Reference::new(c.clone(), tiny_source(&c)).unwrap();
        let h = c.hidden;
        let t = 4;
        let xn: Vec<f32> = (0..t * h).map(|i| prand(77, i)).collect();
        let w = r.read_layer(0).unwrap();
        let Mlp::Moe(m) = &w.mlp else { panic!() };
        let got = r.moe(0, m, &xn, t).unwrap();
        let p = "model.language_model.layers.0.mlp";
        let lut = fp8_table();
        let wide = |src: &mut Mem, name: &str, n: usize, k: usize| -> Vec<f64> {
            match read_matrix(src, name, n, k).unwrap() {
                Matrix::Fp8 { w, scales, .. } => (0..n * k)
                    .map(|i| {
                        let (r, c) = (i / k, i % k);
                        lut[w[i] as usize] as f64
                            * bf16_to_f32(scales[(r / 128) * (k / 128) + c / 128]) as f64
                    })
                    .collect(),
                Matrix::Bf16 { w, .. } => w.iter().map(|&b| bf16_to_f32(b) as f64).collect(),
                Matrix::W4 { .. } => unreachable!("read_matrix never yields a 4-bit matrix"),
            }
        };
        let mv =|w: &[f64], x: &[f64], n: usize| -> Vec<f64> {
            let k = x.len();
            (0..n)
                .map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum())
                .collect()
        };
        let ffn = |src: &mut Mem, q: &str, inter: usize, x: &[f64]| -> Vec<f64> {
            let g = mv(
                &wide(src, &format!("{q}.gate_proj.weight"), inter, h),
                x,
                inter,
            );
            let u = mv(
                &wide(src, &format!("{q}.up_proj.weight"), inter, h),
                x,
                inter,
            );
            let a: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            mv(
                &wide(src, &format!("{q}.down_proj.weight"), h, inter),
                &a,
                h,
            )
        };
        let router = wide(&mut src, &format!("{p}.gate.weight"), s.experts, h);
        let sg = wide(&mut src, &format!("{p}.shared_expert_gate.weight"), 1, h);
        let mut worst = 0f64;
        for i in 0..t {
            let x: Vec<f64> = xn[i * h..(i + 1) * h].iter().map(|&v| v as f64).collect();
            let logits = mv(&router, &x, s.experts);
            let mut order: Vec<usize> = (0..s.experts).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top = &order[..s.top_k];
            let z: f64 = top.iter().map(|&e| logits[e].exp()).sum();
            let gate = 1.0 / (1.0 + (-mv(&sg, &x, 1)[0]).exp());
            let mut want: Vec<f64> =
                ffn(&mut src, &format!("{p}.shared_expert"), s.shared_inter, &x)
                    .into_iter()
                    .map(|v| v * gate)
                    .collect();
            for &e in top {
                let y = ffn(&mut src, &format!("{p}.experts.{e}"), s.expert_inter, &x);
                let wt = logits[e].exp() / z;
                want.iter_mut().zip(&y).for_each(|(o, v)| *o += wt * v);
            }
            let ids: Vec<u16> = r.routes[0][i * s.top_k..(i + 1) * s.top_k].to_vec();
            assert_eq!(
                ids,
                top.iter().map(|&e| e as u16).collect::<Vec<_>>(),
                "row {i}"
            );
            for (j, w) in want.iter().enumerate() {
                let d = (got[i * h + j] as f64 - w).abs() / (1.0 + w.abs());
                worst = worst.max(d);
            }
        }
        assert!(worst < 1e-4, "worst relative error {worst}");
    }
}
