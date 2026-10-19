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
        let text = match raw.get("model_type").and_then(|v| v.as_str()) {
            Some("qwen3_5") => field(&raw, "text_config")?,
            Some("qwen3_5_text") => &raw,
            other => {
                return Err(format!(
                    "config.json: model_type {other:?}, this reference runs qwen3_5 only"
                ))
            }
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
            intermediate: usize_field(text, "intermediate_size")?,
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
            Self::Fp8 { n, .. } | Self::Bf16 { n, .. } => *n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Self::Fp8 { k, .. } | Self::Bf16 { k, .. } => *k,
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
                        out[r * t + tt] = dot(&buf[r * k..(r + 1) * k], xv);
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
    gate: Matrix,
    up: Matrix,
    down: Matrix,
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
pub(crate) fn text_prefix(source: &mut dyn TensorSource) -> Result<String, String> {
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
        })
    }

    pub fn new_state(&self) -> Qwen38State {
        Qwen38State::new(&self.config)
    }

    fn read_layer(&mut self, l: usize) -> Result<LayerWeights, String> {
        let c = self.config.clone();
        let p = format!("{}layers.{l}", self.prefix);
        let src: &mut dyn TensorSource = &mut self.source;
        let h = c.hidden;
        let mixer = if c.full_attention[l] {
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
        let m = format!("{p}.mlp");
        Ok(LayerWeights {
            input_norm: read_vec(src, &format!("{p}.input_layernorm.weight"), h)?,
            post_norm: read_vec(src, &format!("{p}.post_attention_layernorm.weight"), h)?,
            mixer,
            gate: read_matrix(src, &format!("{m}.gate_proj.weight"), c.intermediate, h)?,
            up: read_matrix(src, &format!("{m}.up_proj.weight"), c.intermediate, h)?,
            down: read_matrix(src, &format!("{m}.down_proj.weight"), h, c.intermediate)?,
        })
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
            let mut xn = x.clone();
            rms_norm_zc(&mut xn, &w.post_norm, c.eps);
            let gate = w.gate.matmul(&xn, t);
            let up = w.up.matmul(&xn, t);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let down = w.down.matmul(&act, t);
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
            m.fp8(&format!("{mp}.gate_proj.weight"), c.intermediate, h);
            m.fp8(&format!("{mp}.up_proj.weight"), c.intermediate, h);
            m.fp8(&format!("{mp}.down_proj.weight"), h, c.intermediate);
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
}
