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
//!
//! Both doors (the checkpoint and a pinned layout pack) hand the same ordered images
//! ([`layout::plan`]) to one assembler, which fingerprints every image right before its upload and
//! records the manifest of what it uploaded ([`Qwen38Omega::layout`]): a checkpoint load and a pack
//! load that print the same layout id uploaded the same bytes (sc #676).
use std::path::Path;
use std::time::Instant;

use aien_omega_bf16::Bf16Tensor;
use aien_omega_fp8::Fp8Tensor;
use aien_omega_gpu::ResidentTensor;
use aien_omega_q38::{Attn256Cache, Gdn, GraphConfig, GraphStats, Q38Graph};

pub mod layout;
mod prefill;
pub use prefill::PrefillStats;

use crate::qwen38::Qwen38Config;
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::{ShardSource, TensorSource};
use layout::{entry_sha256, Image, Manifest, Pack, ShapeFacts, StageMutant};

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

/// Where the load's seconds went.
#[derive(Clone, Debug, Default)]
pub struct LoadStats {
    /// `checkpoint` or `pack <id>`.
    pub source: String,
    pub total_s: f64,
    /// Reading safetensors (checkpoint door only).
    pub read_s: f64,
    /// Fingerprinting the images (the pack door's first touch of each mapped page is here).
    pub hash_s: f64,
    /// Inside the device upload calls.
    pub upload_s: f64,
    pub images: usize,
    pub image_bytes: u64,
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
    /// bf16 `[vocab][hidden]` (heap from the checkpoint door, the shared pack mapping otherwise).
    embed: Image,
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
    /// What was uploaded, image by image, with fingerprints.
    layout: Manifest,
    load_stats: LoadStats,
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

/// Checks the shape the graph is built for.
fn check_shape(c: &Qwen38Config) -> Result<(), String> {
    use aien_omega_q38 as q;
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
    Ok(())
}

/// The layer being assembled.
#[derive(Default)]
struct LayerParts {
    mix: Option<Fp8Tensor>,
    ab: Option<Bf16Tensor>,
    o: Option<Fp8Tensor>,
    gate_up: Option<Fp8Tensor>,
    down: Option<Fp8Tensor>,
}

/// Takes the images in plan order and uploads each as it arrives. The plan order is enforced
/// image by image, so a piece is always the one the dispatch below expects.
struct Assembler {
    config: Qwen38Config,
    plan: std::vec::IntoIter<layout::EntrySpec>,
    max_pos: usize,
    embed: Option<Image>,
    final_norm: Option<Vec<f32>>,
    lm_head: Option<Bf16Tensor>,
    gdn: Option<Gdn>,
    attn: Option<Attn256Cache>,
    layers: Vec<LayerTensors>,
    norms: Vec<(Vec<f32>, Vec<f32>)>,
    ab_mm: Vec<ResidentTensor>,
    /// Images waiting for the rest of their piece (FP8 weights for their scales, norms for
    /// their pair, DeltaNet constants for the gated norm).
    held: Vec<Image>,
    cur: LayerParts,
    gi: usize,
    ai: usize,
    record: Manifest,
    stats: LoadStats,
}

fn take_held(held: &mut Vec<Image>, what: &str) -> Result<Image, String> {
    held.pop()
        .ok_or_else(|| format!("assemble: {what} without its first half"))
}

fn upload_fp8(name: &str, w: &Image, s: &Image) -> Result<Fp8Tensor, String> {
    let (n, k) = (w.spec.dims[0], w.spec.dims[1]);
    Fp8Tensor::upload(1, n, k, w.u8(), s.u16()).map_err(gpu(name))
}

impl Assembler {
    fn new(config: Qwen38Config, first: usize, max_pos: usize) -> Result<Self, String> {
        check_shape(&config)?;
        let c = &config;
        let n_attn = c.full_attention.iter().filter(|&&a| a).count();
        let n_gdn = c.layers - n_attn;
        let gdn = if n_gdn > 0 {
            Some(Gdn::alloc(n_gdn).map_err(gpu("gdn alloc"))?)
        } else {
            None
        };
        let attn = if n_attn > 0 {
            Some(
                Attn256Cache::alloc(n_attn, max_pos, c.rope_theta, c.rotary_dim)
                    .map_err(gpu("attn256 alloc"))?,
            )
        } else {
            None
        };
        Ok(Self {
            plan: layout::plan(c, first).into_iter(),
            record: Manifest::new(ShapeFacts::of(c, first)),
            layers: Vec::with_capacity(c.layers),
            norms: Vec::with_capacity(c.layers),
            ab_mm: Vec::with_capacity(n_gdn),
            config,
            max_pos,
            embed: None,
            final_norm: None,
            lm_head: None,
            gdn,
            attn,
            held: Vec::new(),
            cur: LayerParts::default(),
            gi: 0,
            ai: 0,
            stats: LoadStats::default(),
        })
    }

    /// Fingerprints `img` (unless a pack already did), records it, uploads it.
    fn push(&mut self, img: Image) -> Result<(), String> {
        let spec = self
            .plan
            .next()
            .ok_or_else(|| format!("assemble: {} beyond the plan", img.spec.name))?;
        if img.spec != spec {
            return Err(format!(
                "assemble: got {:?}, the plan says {:?}",
                img.spec, spec
            ));
        }
        let sha = match img.checked_sha256() {
            Some(s) => s.to_string(),
            None => {
                let t = Instant::now();
                let s = entry_sha256(img.bytes());
                self.stats.hash_s += t.elapsed().as_secs_f64();
                s
            }
        };
        self.record.push(&img.spec, sha);
        self.stats.images += 1;
        self.stats.image_bytes += img.bytes().len() as u64;
        let t = Instant::now();
        let r = self.dispatch(img);
        self.stats.upload_s += t.elapsed().as_secs_f64();
        r
    }

    fn dispatch(&mut self, img: Image) -> Result<(), String> {
        let name = img.spec.name.clone();
        let piece = match name.strip_prefix("layers.") {
            Some(rest) => rest.split_once('.').map_or(rest, |(_, p)| p),
            None => name.as_str(),
        };
        match piece {
            "embed" => self.embed = Some(img),
            "final_norm" => self.final_norm = Some(img.f32().to_vec()),
            "lm_head" => {
                let (n, k) = (img.spec.dims[0], img.spec.dims[1]);
                self.lm_head = Some(Bf16Tensor::upload(n, k, img.u16()).map_err(gpu("lm_head"))?);
                img.release();
            }
            "mix.w" | "o.w" | "gate_up.w" | "down.w" | "q_norm" | "conv" | "a_log" | "dt_bias"
            | "in_norm" => self.held.push(img),
            "mix.s" | "o.s" | "gate_up.s" | "down.s" => {
                let w = take_held(&mut self.held, &name)?;
                let t = upload_fp8(&w.spec.name, &w, &img)?;
                w.release();
                img.release();
                let slot = match piece {
                    "mix.s" => &mut self.cur.mix,
                    "o.s" => &mut self.cur.o,
                    "gate_up.s" => &mut self.cur.gate_up,
                    _ => &mut self.cur.down,
                };
                *slot = Some(t);
            }
            "ab" => {
                let (n, k) = (img.spec.dims[0], img.spec.dims[1]);
                self.cur.ab =
                    Some(Bf16Tensor::upload(n, k, img.u16()).map_err(gpu("in_proj_a|b"))?);
                img.release();
            }
            "ab_mm" => {
                let (k, n) = (img.spec.dims[0], img.spec.dims[1]);
                self.ab_mm.push(
                    ResidentTensor::upload_bf16(k, n, img.u16())
                        .map_err(gpu("in_proj_a|b matmul"))?,
                );
                img.release();
            }
            "k_norm" => {
                let q = take_held(&mut self.held, &name)?;
                self.attn
                    .as_mut()
                    .ok_or("assemble: an attention norm without a cache")?
                    .set_norms(self.ai, q.f32(), img.f32())
                    .map_err(gpu("attn256 set_norms"))?;
                self.ai += 1;
            }
            "gnorm" => {
                let dt_bias = take_held(&mut self.held, &name)?;
                let a_log = take_held(&mut self.held, &name)?;
                let conv = take_held(&mut self.held, &name)?;
                self.gdn
                    .as_mut()
                    .ok_or("assemble: DeltaNet constants without state")?
                    .set_layer(self.gi, conv.f32(), a_log.f32(), dt_bias.f32(), img.f32())
                    .map_err(gpu("gdn set_layer"))?;
                self.gi += 1;
            }
            "post_norm" => {
                let in_norm = take_held(&mut self.held, &name)?;
                let p = std::mem::take(&mut self.cur);
                let missing =
                    |what: &str| format!("assemble: {name} closes a layer without {what}");
                self.layers.push(LayerTensors {
                    mix: p.mix.ok_or_else(|| missing("mix"))?,
                    ab: p.ab,
                    o: p.o.ok_or_else(|| missing("o"))?,
                    gate_up: p.gate_up.ok_or_else(|| missing("gate_up"))?,
                    down: p.down.ok_or_else(|| missing("down"))?,
                });
                self.norms
                    .push((in_norm.f32().to_vec(), img.f32().to_vec()));
            }
            other => return Err(format!("assemble: unknown piece {other} ({name})")),
        }
        Ok(())
    }

    fn finish(self, total: Instant) -> Result<Qwen38Omega, String> {
        let Self {
            config,
            mut plan,
            max_pos,
            embed,
            final_norm,
            lm_head,
            gdn,
            attn,
            layers,
            norms,
            ab_mm,
            held,
            record,
            mut stats,
            ..
        } = self;
        if let Some(s) = plan.next() {
            return Err(format!("assemble: ended before {}", s.name));
        }
        if !held.is_empty() {
            return Err("assemble: images left over".into());
        }
        let (embed, final_norm, lm_head) = match (embed, final_norm, lm_head) {
            (Some(e), Some(f), Some(l)) => (e, f, l),
            _ => return Err("assemble: embed, final norm or lm_head missing".into()),
        };
        let c = &config;
        let attn_mask = c
            .full_attention
            .iter()
            .enumerate()
            .fold(0u64, |m, (l, &a)| if a { m | 1 << l } else { m });
        let cfg = GraphConfig {
            layers: c.layers as u32,
            attn_mask,
            vocab: c.vocab as u32,
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
        stats.total_s = total.elapsed().as_secs_f64();
        Ok(Qwen38Omega {
            graph,
            layers,
            lm_head,
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
            layout: record,
            load_stats: stats,
        })
    }
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
        let total = Instant::now();
        let c = config.clone();
        let mut asm = Assembler::new(config, first, max_pos)?;
        let read_s = layout::stage(&c, first, source, &StageMutant::default(), &mut |img| {
            asm.push(img)
        })?;
        asm.stats.source = "checkpoint".into();
        asm.stats.read_s = read_s;
        asm.finish(total)
    }

    /// Uploads model layers `first .. first + count` (the whole model when `window` is `None`)
    /// from a layout pack opened against the id the caller expects. The checkpoint directory
    /// gives the config; the pack must have been built from a checkpoint of that shape. `flip`
    /// names one image to corrupt in memory first (the negative control: the load must refuse).
    pub fn open_pack(
        dir: &Path,
        pack: &Pack,
        window: Option<(usize, usize)>,
        max_pos: usize,
        flip: Option<&str>,
    ) -> Result<Self, String> {
        let total = Instant::now();
        let full = Qwen38Config::from_checkpoint(dir)?;
        let (first, count) = window.unwrap_or((0, full.layers));
        let config = full
            .window(first, count)
            .ok_or_else(|| format!("window {first}+{count}: outside the {} layers", full.layers))?;
        let have = &pack.manifest.shape;
        let want = ShapeFacts::of(&full, 0);
        if (have.hidden, have.vocab, have.intermediate, have.first_layer)
            != (want.hidden, want.vocab, want.intermediate, 0)
            || !want.layer_kinds.starts_with(&have.layer_kinds)
            || first + count > have.layers
        {
            return Err(format!(
                "{}: pack {} holds {:?}, the checkpoint's window {first}+{count} needs {:?}",
                layout::LAYOUT_REFUSAL_PREFIX,
                pack.path.display(),
                have,
                want
            ));
        }
        let mut asm = Assembler::new(config.clone(), first, max_pos)?;
        for spec in layout::plan(&config, first) {
            let t = Instant::now();
            let img = pack.take(&spec, flip == Some(spec.name.as_str()))?;
            asm.stats.hash_s += t.elapsed().as_secs_f64();
            asm.push(img)?;
        }
        asm.stats.source = format!("pack {}", pack.id);
        asm.finish(total)
    }

    /// The manifest of what this engine uploaded: one fingerprinted entry per image. For a whole
    /// model its [`Manifest::id`] is the layout id.
    pub fn layout(&self) -> &Manifest {
        &self.layout
    }

    /// Where the load's seconds went.
    pub fn load_stats(&self) -> &LoadStats {
        &self.load_stats
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
        let x: Vec<f32> = self.embed.u16()[t * h..(t + 1) * h]
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
