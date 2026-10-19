//! Skipped-ideas lab (branch `research/skipped-ideas`, handoff
//! `~/handoffs/2026-10-19-skipped-ideas-lab.md`).
//!
//! Host-only experiments on the f32 reference ([`super::Qwen38Reference`]): the same maths, the
//! same streamed FP8 checkpoint, with one change per variant. A pass reads the checkpoint once;
//! every variant of an experiment rides that pass as its own stream (residual + DeltaNet state +
//! KV), and streams whose layer actions are equal so far share one computation (a stream is
//! cloned the first layer its variants differ). Variant 0 is always the unchanged model; every
//! number is a comparison against it on the same tokens.
//!
//! - Layer skipping: whole layers skipped for every token (static), or for every other decoded
//!   token (dynamic) with the skipped token either invisible to the layer's state (`nostate`:
//!   no DeltaNet state update, no KV entry) or still written into it (`cheap`: the state update
//!   or the k/v rows run, the output is dropped).
//! - TEAL: the input of every projection keeps its largest `(1 - s)` share by magnitude per
//!   decoded row (`row`), or one column set per group of `N` consecutive decoded rows, the
//!   union of the rows' own sets (`union`) or the top columns by summed normalised magnitude
//!   (`shared`). The per-row runs also count how many columns a union over 1, 2, 4, 8, 16 rows
//!   keeps, weighted by the bytes of the matrices reading that input.
//! - Low-bit weights: every FP8 projection widened to f32 and fake-quantised in groups of 32
//!   along k (our MXFP4 converter's rule, its 3-bit analogues, and bf16-scaled int4 / int3).
//! - MTP traces and online drafter learning: see [`mtp_trace`] and [`learn`].
//!
//! Prompt rows (before `score_start`) run the unchanged model for the decode-time ideas
//! (dynamic skip, TEAL), as a server would prefill them; low-bit weights and static skips
//! apply to every row (they are a different model).

use std::collections::HashMap;

use super::*;

/// Weights per quantisation group along k.
pub const GROUP: usize = 32;

// ---------------------------------------------------------------------------
// Low-bit weights.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quant {
    /// Our MXFP4 converter (`qwen38_omega::w4`): e2m1, `X = 2^(floor(log2 amax) - 2)`.
    Mxfp4,
    /// 3-bit sign-magnitude integer 0..3 under a power-of-two scale `2^(floor(log2 amax) - 1)`.
    Mxint3,
    /// 3-bit float e2m0 (0, 1, 2, 4) under `2^(floor(log2 amax) - 1)`.
    Mxfp3,
    /// Symmetric int4 (-7..7), bf16 scale `amax / 7` per group (4.5 bits).
    Int4G32,
    /// Symmetric int3 (-3..3), bf16 scale `amax / 3` per group (3.5 bits).
    Int3G32,
}

impl Quant {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "mxfp4" => Self::Mxfp4,
            "mxint3" => Self::Mxint3,
            "mxfp3" => Self::Mxfp3,
            "int4g32" => Self::Int4G32,
            "int3g32" => Self::Int3G32,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Mxfp4 => "mxfp4",
            Self::Mxint3 => "mxint3",
            Self::Mxfp3 => "mxfp3",
            Self::Int4G32 => "int4g32",
            Self::Int3G32 => "int3g32",
        }
    }
    /// Stored bits per weight including the group scale.
    pub fn bits(self) -> f64 {
        match self {
            Self::Mxfp4 => 4.25,
            Self::Mxint3 | Self::Mxfp3 => 3.25,
            Self::Int4G32 => 4.5,
            Self::Int3G32 => 3.5,
        }
    }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const INT3P: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
const E2M0: [f32; 4] = [0.0, 1.0, 2.0, 4.0];

/// Nearest grid value to `a >= 0`, ties to the even index, saturating at the top.
fn nearest(grid: &[f32], a: f32) -> f32 {
    let top = grid[grid.len() - 1];
    if a >= top {
        return top;
    }
    let i = grid.iter().rposition(|&g| g <= a).expect("a >= 0");
    let (lo, hi) = (grid[i], grid[i + 1]);
    let mid = 0.5 * (lo + hi);
    if a < mid || (a == mid && i % 2 == 0) {
        lo
    } else {
        hi
    }
}

/// `floor(log2 a)` for a finite `a > 0`.
fn floor_log2(a: f32) -> i32 {
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32;
    if e == 0 {
        a.log2().floor() as i32
    } else {
        e - 127
    }
}

/// One group in place.
pub fn quant_group(q: Quant, g: &mut [f32]) {
    let amax = g.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        return;
    }
    match q {
        Quant::Mxfp4 | Quant::Mxint3 | Quant::Mxfp3 => {
            let (grid, off): (&[f32], i32) = match q {
                Quant::Mxfp4 => (&E2M1, 2),
                Quant::Mxint3 => (&INT3P, 1),
                _ => (&E2M0, 1),
            };
            let x = 2f32.powi(floor_log2(amax) - off);
            for v in g.iter_mut() {
                *v = (nearest(grid, v.abs() / x) * x).copysign(*v);
            }
        }
        Quant::Int4G32 | Quant::Int3G32 => {
            let qmax = if q == Quant::Int4G32 { 7.0 } else { 3.0 };
            let s = bf16_to_f32(crate::qwen3_moe::f32_to_bf16(amax / qmax));
            for v in g.iter_mut() {
                *v = ((v.abs() / s).round().min(qmax) * s).copysign(*v);
            }
        }
    }
}

/// A projection held as f32 rows `[n][k]` (fake-quantised).
pub struct F32Mat {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

fn by_row_to_tokens(by_row: Vec<f32>, n: usize, t: usize) -> Vec<f32> {
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

/// `out[r * t + tt] = W[r] . x[tt]` for `rows` rows of `w` (`[rows][k]`) and `t` rows of `x`,
/// two weight rows by four tokens at a time (eight lanes each) so a weight load feeds four
/// products. Sums in another order than [`dot`]: a lab pass compares within one binary only.
fn kernel(w: &[f32], rows: usize, k: usize, x: &[f32], t: usize, out: &mut [f32]) {
    let lanes = |a: &[f32; 8]| a.iter().sum::<f32>();
    let mut tt = 0;
    while tt + 4 <= t && k % 8 == 0 {
        let xs = [
            &x[tt * k..(tt + 1) * k],
            &x[(tt + 1) * k..(tt + 2) * k],
            &x[(tt + 2) * k..(tt + 3) * k],
            &x[(tt + 3) * k..(tt + 4) * k],
        ];
        let mut r = 0;
        while r < rows {
            let w0 = &w[r * k..(r + 1) * k];
            let mut acc = [[0f32; 8]; 4];
            for (c, a) in w0.chunks_exact(8).enumerate() {
                let s = c * 8..c * 8 + 8;
                let x0 = &xs[0][s.clone()];
                let x1 = &xs[1][s.clone()];
                let x2 = &xs[2][s.clone()];
                let x3 = &xs[3][s];
                for l in 0..8 {
                    acc[0][l] += a[l] * x0[l];
                    acc[1][l] += a[l] * x1[l];
                    acc[2][l] += a[l] * x2[l];
                    acc[3][l] += a[l] * x3[l];
                }
            }
            for (j, a) in acc.iter().enumerate() {
                out[r * t + tt + j] = lanes(a);
            }
            r += 1;
        }
        tt += 4;
    }
    while tt < t {
        let xv = &x[tt * k..(tt + 1) * k];
        for r in 0..rows {
            out[r * t + tt] = dot(&w[r * k..(r + 1) * k], xv);
        }
        tt += 1;
    }
}

/// [`Matrix::matmul`] through [`kernel`].
pub fn fast_matmul(m: &Matrix, x: &[f32], t: usize) -> Vec<f32> {
    let (n, k) = (m.n(), m.k());
    assert_eq!(x.len(), t * k);
    let lut = fp8_table();
    let mut by_row = vec![0f32; n * t];
    by_row
        .par_chunks_mut(ROWS_PER_TASK * t)
        .enumerate()
        .for_each_init(Vec::new, |buf: &mut Vec<f32>, (task, out)| {
            let r0 = task * ROWS_PER_TASK;
            let rows = out.len() / t;
            buf.resize(rows * k, 0.0);
            m.widen_rows(r0, rows, &lut, buf);
            kernel(buf, rows, k, x, t, out);
        });
    by_row_to_tokens(by_row, n, t)
}

impl F32Mat {
    pub fn matmul(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        assert_eq!(x.len(), t * k);
        let mut by_row = vec![0f32; n * t];
        by_row
            .par_chunks_mut(ROWS_PER_TASK * t)
            .enumerate()
            .for_each(|(task, out)| {
                let r0 = task * ROWS_PER_TASK;
                let rows = out.len() / t;
                kernel(&self.w[r0 * k..(r0 + rows) * k], rows, k, x, t, out);
            });
        by_row_to_tokens(by_row, n, t)
    }
}

/// `m` widened and fake-quantised; returns the matrix and (sum sq error, sum sq weight).
pub fn quantize(m: &Matrix, q: Quant) -> (F32Mat, f64, f64) {
    let (n, k) = (m.n(), m.k());
    let lut = fp8_table();
    let mut w = vec![0f32; n * k];
    let errs: Vec<(f64, f64)> = w
        .par_chunks_mut(k * ROWS_PER_TASK)
        .enumerate()
        .map(|(task, dst)| {
            let rows = dst.len() / k;
            m.widen_rows(task * ROWS_PER_TASK, rows, &lut, dst);
            let mut e = (0f64, 0f64);
            let mut orig = [0f32; GROUP];
            for g in dst.chunks_exact_mut(GROUP) {
                orig.copy_from_slice(g);
                quant_group(q, g);
                for (a, b) in orig.iter().zip(g.iter()) {
                    e.0 += ((a - b) as f64).powi(2);
                    e.1 += (*a as f64).powi(2);
                }
            }
            e
        })
        .collect();
    let (se, sr) = errs
        .iter()
        .fold((0.0, 0.0), |acc, e| (acc.0 + e.0, acc.1 + e.1));
    (F32Mat { n, k, w }, se, sr)
}

// ---------------------------------------------------------------------------
// Sites, bytes, layer views.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Site {
    Qkv,
    Z,
    A,
    B,
    Out,
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

fn mat_bytes(m: &Matrix) -> f64 {
    match m {
        Matrix::Fp8 { n, k, .. } => (n * k) as f64,
        Matrix::Bf16 { n, k, .. } => (2 * n * k) as f64,
    }
}

/// Bytes one decoded token reads per layer part, from the config (FP8 = 1 byte, bf16 = 2).
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerBytes {
    /// DeltaNet in_proj_qkv + in_proj_a + in_proj_b (what a cheap state update reads), or
    /// attention k + v (what a cheap KV fill reads).
    pub state: f64,
    /// The rest of the mixer (z, out_proj / q, o_proj).
    pub mixer_rest: f64,
    pub mlp: f64,
}

impl LayerBytes {
    pub fn total(&self) -> f64 {
        self.state + self.mixer_rest + self.mlp
    }
}

pub fn layer_bytes(c: &Qwen38Config, l: usize) -> LayerBytes {
    let h = c.hidden as f64;
    let mlp = 3.0 * h * c.intermediate as f64;
    if c.full_attention[l] {
        let q = (2 * c.heads * c.head_dim) as f64 * h;
        let kv = 2.0 * c.kv_dim() as f64 * h;
        let o = h * (c.heads * c.head_dim) as f64;
        LayerBytes {
            state: kv,
            mixer_rest: q + o,
            mlp,
        }
    } else {
        let qkv = c.conv_dim() as f64 * h;
        let ab = 2.0 * 2.0 * c.lin_v_heads as f64 * h;
        let z = c.lin_value_dim() as f64 * h;
        let out = h * c.lin_value_dim() as f64;
        LayerBytes {
            state: qkv + ab,
            mixer_rest: z + out,
            mlp,
        }
    }
}

/// Body bytes per token plus the bf16 lm_head.
pub fn model_bytes(c: &Qwen38Config) -> (f64, f64) {
    let body: f64 = (0..c.layers).map(|l| layer_bytes(c, l).total()).sum();
    (body, 2.0 * c.vocab as f64 * c.hidden as f64)
}

struct View<'a> {
    w: &'a LayerWeights,
    q: Option<&'a HashMap<Site, F32Mat>>,
}

impl View<'_> {
    fn raw(&self, s: Site) -> &Matrix {
        match (&self.w.mixer, s) {
            (_, Site::Gate) => &self.w.gate,
            (_, Site::Up) => &self.w.up,
            (_, Site::Down) => &self.w.down,
            (Mixer::DeltaNet(d), Site::Qkv) => &d.in_qkv,
            (Mixer::DeltaNet(d), Site::Z) => &d.in_z,
            (Mixer::DeltaNet(d), Site::A) => &d.in_a,
            (Mixer::DeltaNet(d), Site::B) => &d.in_b,
            (Mixer::DeltaNet(d), Site::Out) => &d.out,
            (Mixer::Attention(a), Site::Q) => &a.q,
            (Mixer::Attention(a), Site::K) => &a.k,
            (Mixer::Attention(a), Site::V) => &a.v,
            (Mixer::Attention(a), Site::O) => &a.o,
            _ => panic!("site {s:?} not in this layer"),
        }
    }
    fn mm(&self, s: Site, x: &[f32], t: usize) -> Vec<f32> {
        if let Some(m) = self.q.and_then(|q| q.get(&s)) {
            return m.matmul(x, t);
        }
        fast_matmul(self.raw(s), x, t)
    }
    fn bytes(&self, sites: &[Site]) -> f64 {
        sites.iter().map(|&s| mat_bytes(self.raw(s))).sum()
    }
}

fn quantize_layer(w: &LayerWeights, q: Quant, err: &mut (f64, f64)) -> HashMap<Site, F32Mat> {
    let sites: &[Site] = match &w.mixer {
        Mixer::DeltaNet(_) => &[
            Site::Qkv,
            Site::Z,
            Site::Out,
            Site::Gate,
            Site::Up,
            Site::Down,
        ],
        Mixer::Attention(_) => &[
            Site::Q,
            Site::K,
            Site::V,
            Site::O,
            Site::Gate,
            Site::Up,
            Site::Down,
        ],
    };
    let v = View { w, q: None };
    let mut out = HashMap::new();
    for &s in sites {
        let m = v.raw(s);
        if matches!(m, Matrix::Fp8 { .. }) {
            let (f, se, sr) = quantize(m, q);
            err.0 += se;
            err.1 += sr;
            out.insert(s, f);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// TEAL masks.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TealMode {
    Row,
    Union,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Teal {
    pub s: f32,
    pub group: usize,
    pub mode: TealMode,
}

/// Union widths measured on per-row runs.
pub const UNION_N: [usize; 5] = [1, 2, 4, 8, 16];

/// Bytes-weighted kept-column fractions for unions over `UNION_N` rows.
#[derive(Debug, Clone, Default)]
pub struct UnionStats {
    pub kept_bytes: [f64; 5],
    pub total_bytes: f64,
}

fn row_keep_mask(row: &[f32], s: f32) -> Vec<bool> {
    let k = row.len();
    let drop = ((s as f64) * k as f64).round() as usize;
    if drop == 0 {
        return vec![true; k];
    }
    let mut a: Vec<f32> = row.iter().map(|v| v.abs()).collect();
    let (_, thr, _) = a.select_nth_unstable_by(drop.min(k - 1), |x, y| x.total_cmp(y));
    let thr = *thr;
    row.iter().map(|v| v.abs() >= thr).collect()
}

impl Teal {
    /// Sparsifies rows `from..t` of `x` (`[t][k]`) in place.
    fn apply(&self, x: &mut [f32], k: usize, from: usize, bytes: f64, us: Option<&mut UnionStats>) {
        let t = x.len() / k;
        if from >= t {
            return;
        }
        match self.mode {
            TealMode::Row => {
                let masks: Vec<Vec<bool>> = x[from * k..]
                    .par_chunks(k)
                    .map(|r| row_keep_mask(r, self.s))
                    .collect();
                if let Some(us) = us {
                    us.total_bytes += bytes;
                    for (ni, &n) in UNION_N.iter().enumerate() {
                        let (mut kept, mut groups) = (0usize, 0usize);
                        for g in masks.chunks(n) {
                            if g.len() < n {
                                continue;
                            }
                            groups += 1;
                            kept += (0..k).filter(|&c| g.iter().any(|m| m[c])).count();
                        }
                        if groups > 0 {
                            us.kept_bytes[ni] += bytes * kept as f64 / (groups * k) as f64;
                        }
                    }
                }
                for (r, m) in x[from * k..].chunks_mut(k).zip(&masks) {
                    for (v, &keep) in r.iter_mut().zip(m) {
                        if !keep {
                            *v = 0.0;
                        }
                    }
                }
            }
            TealMode::Union | TealMode::Shared => {
                for g in x[from * k..].chunks_mut(self.group * k) {
                    let rows = g.len() / k;
                    let keep: Vec<bool> = if self.mode == TealMode::Union {
                        let masks: Vec<Vec<bool>> =
                            g.chunks(k).map(|r| row_keep_mask(r, self.s)).collect();
                        (0..k).map(|c| masks.iter().any(|m| m[c])).collect()
                    } else {
                        let mut score = vec![0f32; k];
                        for r in g.chunks(k) {
                            let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
                            for (s, v) in score.iter_mut().zip(r) {
                                *s += v.abs() / norm;
                            }
                        }
                        row_keep_mask(&score, self.s)
                    };
                    let _ = rows;
                    for r in g.chunks_mut(k) {
                        for (v, &kp) in r.iter_mut().zip(&keep) {
                            if !kp {
                                *v = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Layer actions and one layer of one stream.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Act {
    Full,
    /// The layer is gone for every token.
    Skip,
    /// Decoded rows marked in the context's `dyn_skip` skip the layer's output; `cheap` keeps
    /// them in the mixer's state (DeltaNet update / KV row).
    DynSkip {
        cheap: bool,
    },
    Teal(Teal),
    Quant(Quant),
}

struct Ctx<'a> {
    c: &'a Qwen38Config,
    /// Rows from here on are decoded rows.
    score_start: usize,
    dyn_skip: &'a [bool],
}

fn gather_rows(x: &[f32], rows: &[usize], h: usize) -> Vec<f32> {
    let mut o = Vec::with_capacity(rows.len() * h);
    for &r in rows {
        o.extend_from_slice(&x[r * h..(r + 1) * h]);
    }
    o
}

/// One layer for one stream; `x` is `[t][hidden]`, positions `pos0 ..`.
#[allow(clippy::too_many_arguments)]
fn layer_fwd(
    ctx: &Ctx,
    v: &View,
    st: &mut LayerState,
    x: &mut [f32],
    t: usize,
    pos0: usize,
    act: &Act,
    us: &mut Option<UnionStats>,
) -> Result<(), String> {
    let c = ctx.c;
    let h = c.hidden;
    if *act == Act::Skip {
        return Ok(());
    }
    let teal = match act {
        Act::Teal(tl) => Some(*tl),
        _ => None,
    };
    let from = ctx.score_start;
    let skip_rows: Option<(&[bool], bool)> = match act {
        Act::DynSkip { cheap } => Some((ctx.dyn_skip, *cheap)),
        _ => None,
    };
    let mut xn = x.to_vec();
    rms_norm_zc(&mut xn, &v.w.input_norm, c.eps);
    let positions: Vec<usize> = (pos0..pos0 + t).collect();
    let mixed = match (&v.w.mixer, st) {
        (Mixer::DeltaNet(d), LayerState::DeltaNet { conv, s }) => match skip_rows {
            Some((skip, false)) => {
                let keep: Vec<usize> = (0..t).filter(|&i| !skip[pos0 + i]).collect();
                let sub = gather_rows(&xn, &keep, h);
                let o = dn_lab(c, v, d, &sub, keep.len(), conv, s, None, 0, us);
                let mut full = vec![0f32; t * h];
                for (j, &i) in keep.iter().enumerate() {
                    full[i * h..(i + 1) * h].copy_from_slice(&o[j * h..(j + 1) * h]);
                }
                full
            }
            _ => dn_lab(
                c,
                v,
                d,
                &xn,
                t,
                conv,
                s,
                teal,
                from.saturating_sub(pos0),
                us,
            ),
        },
        (Mixer::Attention(a), LayerState::Attention { k, v: vc }) => match skip_rows {
            Some((skip, false)) => {
                let keep: Vec<usize> = (0..t).filter(|&i| !skip[pos0 + i]).collect();
                let sub = gather_rows(&xn, &keep, h);
                let pos: Vec<usize> = keep.iter().map(|&i| pos0 + i).collect();
                let o = attn_lab(c, v, a, &sub, &pos, k, vc, None, 0, us);
                let mut full = vec![0f32; t * h];
                for (j, &i) in keep.iter().enumerate() {
                    full[i * h..(i + 1) * h].copy_from_slice(&o[j * h..(j + 1) * h]);
                }
                full
            }
            _ => attn_lab(
                c,
                v,
                a,
                &xn,
                &positions,
                k,
                vc,
                teal,
                from.saturating_sub(pos0),
                us,
            ),
        },
        _ => return Err("state kind differs from the weights".into()),
    };
    let skipped = |i: usize| skip_rows.is_some_and(|(s, _)| s[pos0 + i]);
    for i in 0..t {
        if !skipped(i) {
            for (a, b) in x[i * h..(i + 1) * h]
                .iter_mut()
                .zip(&mixed[i * h..(i + 1) * h])
            {
                *a += b;
            }
        }
    }
    let mut xn = x.to_vec();
    rms_norm_zc(&mut xn, &v.w.post_norm, c.eps);
    let tf = from.saturating_sub(pos0);
    if let Some(tl) = teal {
        tl.apply(
            &mut xn,
            h,
            tf,
            v.bytes(&[Site::Gate, Site::Up]),
            us.as_mut(),
        );
    }
    let gate = v.mm(Site::Gate, &xn, t);
    let up = v.mm(Site::Up, &xn, t);
    let mut act_v: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    if let Some(tl) = teal {
        tl.apply(
            &mut act_v,
            c.intermediate,
            tf,
            v.bytes(&[Site::Down]),
            us.as_mut(),
        );
    }
    let down = v.mm(Site::Down, &act_v, t);
    for i in 0..t {
        if !skipped(i) {
            for (a, b) in x[i * h..(i + 1) * h]
                .iter_mut()
                .zip(&down[i * h..(i + 1) * h])
            {
                *a += b;
            }
        }
    }
    Ok(())
}

/// [`delta_net`] with the lab's matrices and TEAL hook.
#[allow(clippy::too_many_arguments)]
fn dn_lab(
    c: &Qwen38Config,
    v: &View,
    w: &DeltaNetWeights,
    xn: &[f32],
    t: usize,
    conv_tail: &mut [f32],
    state: &mut [f32],
    teal: Option<Teal>,
    from: usize,
    us: &mut Option<UnionStats>,
) -> Vec<f32> {
    let h = c.hidden;
    if t == 0 {
        return Vec::new();
    }
    let mut xs = xn.to_vec();
    if let Some(tl) = teal {
        tl.apply(
            &mut xs,
            h,
            from,
            v.bytes(&[Site::Qkv, Site::Z, Site::A, Site::B]),
            us.as_mut(),
        );
    }
    let (cd, kern) = (c.conv_dim(), c.conv_kernel);
    let qkv = v.mm(Site::Qkv, &xs, t);
    let z = v.mm(Site::Z, &xs, t);
    let a = v.mm(Site::A, &xs, t);
    let b = v.mm(Site::B, &xs, t);
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
    for (o, zr) in core.chunks_exact_mut(vd).zip(z.chunks_exact(vd)) {
        let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
        let inv = 1.0 / (ms + c.eps).sqrt();
        for ((v, &g), &zz) in o.iter_mut().zip(&w.norm).zip(zr) {
            *v = *v * inv * g * silu(zz);
        }
    }
    if let Some(tl) = teal {
        tl.apply(
            &mut core,
            c.lin_value_dim(),
            from,
            v.bytes(&[Site::Out]),
            us.as_mut(),
        );
    }
    v.mm(Site::Out, &core, t)
}

/// [`attention`] at explicit positions (rows may skip positions: a token missing from the
/// cache is simply not attended to).
#[allow(clippy::too_many_arguments)]
fn attn_lab(
    c: &Qwen38Config,
    v: &View,
    _w: &AttentionWeights,
    xn: &[f32],
    positions: &[usize],
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    teal: Option<Teal>,
    from: usize,
    us: &mut Option<UnionStats>,
) -> Vec<f32> {
    let w = match &v.w.mixer {
        Mixer::Attention(a) => a,
        _ => unreachable!(),
    };
    let t = positions.len();
    let h = c.hidden;
    if t == 0 {
        return Vec::new();
    }
    let (nh, hd, kvd) = (c.heads, c.head_dim, c.kv_dim());
    let group = nh / c.kv_heads;
    let mut xs = xn.to_vec();
    if let Some(tl) = teal {
        tl.apply(
            &mut xs,
            h,
            from,
            v.bytes(&[Site::Q, Site::K, Site::V]),
            us.as_mut(),
        );
    }
    let qg = v.mm(Site::Q, &xs, t);
    let mut k = v.mm(Site::K, &xs, t);
    let vv = v.mm(Site::V, &xs, t);
    let mut q = vec![0f32; t * nh * hd];
    let mut gate = vec![0f32; t * nh * hd];
    for i in 0..t {
        for hh in 0..nh {
            let src = &qg[(i * nh + hh) * 2 * hd..(i * nh + hh + 1) * 2 * hd];
            let dq = &mut q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            dq.copy_from_slice(&src[..hd]);
            rms_norm_zc(dq, &w.q_norm, c.eps);
            rope(dq, positions[i], c.rotary_dim, c.rope_theta);
            gate[(i * nh + hh) * hd..(i * nh + hh + 1) * hd].copy_from_slice(&src[hd..]);
        }
        for hh in 0..c.kv_heads {
            let dk = &mut k[i * kvd + hh * hd..i * kvd + (hh + 1) * hd];
            rms_norm_zc(dk, &w.k_norm, c.eps);
            rope(dk, positions[i], c.rotary_dim, c.rope_theta);
        }
    }
    let len0 = k_cache.len() / kvd;
    k_cache.extend_from_slice(&k);
    v_cache.extend_from_slice(&vv);
    let (kc, vc) = (&k_cache[..], &v_cache[..]);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; t * nh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
        let (i, hh) = (idx / nh, idx % nh);
        let kh = hh / group;
        let qv = &q[idx * hd..(idx + 1) * hd];
        let span = len0 + i + 1;
        let scores: Vec<f32> = (0..span)
            .map(|p| dot(qv, &kc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd]) * scale)
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = weights.iter().sum();
        for (p, &wt) in weights.iter().enumerate() {
            let vrow = &vc[p * kvd + kh * hd..p * kvd + (kh + 1) * hd];
            for (ov, &x) in o.iter_mut().zip(vrow) {
                *ov += wt * x;
            }
        }
        let g = &gate[idx * hd..(idx + 1) * hd];
        for (ov, &gv) in o.iter_mut().zip(g) {
            *ov = *ov / sum * sigmoid(gv);
        }
    });
    if let Some(tl) = teal {
        tl.apply(&mut out, nh * hd, from, v.bytes(&[Site::O]), us.as_mut());
    }
    v.mm(Site::O, &out, t)
}

// ---------------------------------------------------------------------------
// A multi-variant pass.
// ---------------------------------------------------------------------------

pub struct Variant {
    pub name: String,
    /// One action per layer.
    pub acts: Vec<Act>,
}

struct Stream {
    x: Vec<f32>,
    st: Qwen38State,
    vars: Vec<usize>,
}

/// Per-variant outcome of a pass.
#[derive(Debug, Clone, Default)]
pub struct VarOut {
    /// Cosine with variant 0's residual after each layer (mean, min over scored rows).
    pub cos_layer: Vec<(f64, f64)>,
    /// Final residual (pre-norm) of the scored rows, `[rows][hidden]`.
    pub final_rows: Vec<f32>,
    pub union: Option<UnionStats>,
    /// Logits of the scored rows, filled for variant 0 only by [`Lab::pass`] (others are
    /// scored on the fly by the caller through [`Lab::logits`]).
    pub quant_err: (f64, f64),
}

pub struct PassOut {
    pub vars: Vec<VarOut>,
    /// Block influence of each layer for variant 0: 1 - mean cos(x_in, x_out) over scored rows.
    pub bi: Vec<f64>,
    /// Variant 0's residual entering layer `l` for every `l` in `taps`, scored rows only.
    pub taps: Vec<(usize, Vec<f32>)>,
    pub seconds: f64,
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-300)
}

fn cos_rows(a: &[f32], b: &[f32], h: usize, r0: usize, r1: usize) -> (f64, f64) {
    let v: Vec<f64> = (r0..r1)
        .map(|r| cos(&a[r * h..(r + 1) * h], &b[r * h..(r + 1) * h]))
        .collect();
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, min)
}

pub type Lab = Qwen38Reference<crate::qwen3_moe_omega::ShardSource>;

impl<S: TensorSource> Qwen38Reference<S> {
    /// Runs every variant over `tokens` from an empty state. Rows `score_start - 1 .. t - 1`
    /// are the scored rows (each predicts a decoded token). `dyn_skip[i]` marks positions
    /// whose `DynSkip` layers are skipped.
    pub fn lab_pass(
        &mut self,
        tokens: &[u32],
        score_start: usize,
        variants: &[Variant],
        dyn_skip: &[bool],
        tap_at: &[usize],
    ) -> Result<PassOut, String> {
        let c = self.config.clone();
        let (t, h) = (tokens.len(), c.hidden);
        if score_start == 0 || score_start >= t || dyn_skip.len() != t {
            return Err("lab_pass: bad score_start or dyn_skip".into());
        }
        if variants.is_empty() || variants[0].acts.iter().any(|a| *a != Act::Full) {
            return Err("lab_pass: variant 0 must be the unchanged model".into());
        }
        let (r0, r1) = (score_start - 1, t - 1);
        let t0 = std::time::Instant::now();
        let mut x = vec![0f32; t * h];
        for (row, &tok) in x.chunks_exact_mut(h).zip(tokens) {
            let tok = tok as usize;
            for (o, &b) in row.iter_mut().zip(&self.embed[tok * h..(tok + 1) * h]) {
                *o = bf16_to_f32(b);
            }
        }
        let mut streams = vec![Stream {
            x,
            st: Qwen38State::new(&c),
            vars: (0..variants.len()).collect(),
        }];
        let nv = variants.len();
        let mut outs: Vec<VarOut> = (0..nv).map(|_| VarOut::default()).collect();
        let mut unions: Vec<Option<UnionStats>> = variants
            .iter()
            .map(|v| {
                v.acts
                    .iter()
                    .any(|a| matches!(a, Act::Teal(tl) if tl.mode == TealMode::Row))
                    .then(UnionStats::default)
            })
            .collect();
        let mut bi = Vec::with_capacity(c.layers);
        let mut taps = Vec::new();
        let ctx = Ctx {
            c: &c,
            score_start,
            dyn_skip,
        };
        for l in 0..c.layers {
            // Split streams whose variants now act differently.
            let mut next = Vec::new();
            for s in streams.drain(..) {
                let mut groups: Vec<(Act, Vec<usize>)> = Vec::new();
                for &vi in &s.vars {
                    let a = variants[vi].acts[l].clone();
                    match groups.iter_mut().find(|g| g.0 == a) {
                        Some(g) => g.1.push(vi),
                        None => groups.push((a, vec![vi])),
                    }
                }
                let n = groups.len();
                let mut s = Some(s);
                for (gi, (a, vars)) in groups.into_iter().enumerate() {
                    let base = if gi + 1 == n {
                        s.take().expect("last group takes the stream")
                    } else {
                        let r = s.as_ref().expect("stream");
                        Stream {
                            x: r.x.clone(),
                            st: r.st.clone(),
                            vars: Vec::new(),
                        }
                    };
                    next.push((a, Stream { vars, ..base }));
                }
            }
            // Variant 0's stream first; quantised streams grouped by scheme.
            next.sort_by_key(|(a, s)| {
                (
                    !s.vars.contains(&0),
                    match a {
                        Act::Quant(q) => *q as usize + 1,
                        _ => 0,
                    },
                )
            });
            if tap_at.contains(&l) {
                taps.push((l, next[0].1.x[r0 * h..r1 * h].to_vec()));
            }
            let w = self.read_layer(l)?;
            let mut qcache: Option<(Quant, HashMap<Site, F32Mat>, (f64, f64))> = None;
            let mut base_x: Vec<f32> = Vec::new();
            for (a, s) in next.iter_mut() {
                let qmap = if let Act::Quant(q) = a {
                    if qcache.as_ref().map(|c| c.0) != Some(*q) {
                        let mut err = (0.0, 0.0);
                        let m = quantize_layer(&w, *q, &mut err);
                        qcache = Some((*q, m, err));
                    }
                    let (_, _, err) = qcache.as_ref().expect("built");
                    for &vi in &s.vars {
                        outs[vi].quant_err.0 += err.0;
                        outs[vi].quant_err.1 += err.1;
                    }
                    Some(&qcache.as_ref().expect("built").1)
                } else {
                    None
                };
                let view = View { w: &w, q: qmap };
                let x_in = if s.vars.contains(&0) {
                    Some(s.x.clone())
                } else {
                    None
                };
                // Union stats: one accumulator per stream (its variants share it).
                let mut us = s.vars.iter().find_map(|&vi| unions[vi].clone());
                let pos = s.st.pos;
                layer_fwd(
                    &ctx,
                    &view,
                    &mut s.st.layers[l],
                    &mut s.x,
                    t,
                    pos,
                    a,
                    &mut us,
                )
                .map_err(|e| format!("layer {l}: {e}"))?;
                if let Some(u) = us {
                    for &vi in &s.vars {
                        if unions[vi].is_some() {
                            unions[vi] = Some(u.clone());
                        }
                    }
                }
                if let Some(xi) = x_in {
                    bi.push(1.0 - cos_rows(&xi, &s.x, h, r0, r1).0);
                    base_x = s.x.clone();
                }
                let cl = cos_rows(&base_x, &s.x, h, r0, r1);
                for &vi in &s.vars {
                    outs[vi].cos_layer.push(cl);
                }
            }
            streams = next.into_iter().map(|(_, s)| s).collect();
            if qcache.is_some() {
                drop(qcache);
            }
            let alive = streams.len();
            eprintln!(
                "layer {l:2} done, {alive} streams, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
        for s in streams.iter_mut() {
            s.st.pos += t;
        }
        if tap_at.contains(&c.layers) {
            let b = streams.iter().find(|s| s.vars.contains(&0)).expect("base");
            taps.push((c.layers, b.x[r0 * h..r1 * h].to_vec()));
        }
        for s in &streams {
            for &vi in &s.vars {
                outs[vi].final_rows = s.x[r0 * h..r1 * h].to_vec();
                outs[vi].union = unions[vi].clone();
            }
        }
        Ok(PassOut {
            vars: outs,
            bi,
            taps,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }

    /// Final norm + lm_head of residual rows `[rows][hidden]`.
    pub fn lab_logits(&self, rows: &[f32]) -> Vec<f32> {
        let mut r = rows.to_vec();
        rms_norm_zc(&mut r, &self.final_norm, self.config.eps);
        fast_matmul(&self.lm_head, &r, rows.len() / self.config.hidden)
    }

    /// The lm_head (bf16) for the drafter-learning simulation.
    pub fn lab_lm_head(&self) -> &Matrix {
        &self.lm_head
    }

    pub fn lab_embed_row(&self, tok: u32) -> Vec<f32> {
        let h = self.config.hidden;
        let t = tok as usize;
        self.embed[t * h..(t + 1) * h]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }

    /// The checkpoint's MTP head over pairs (`hidden[i]` = main residual after the last layer
    /// at position i, `next[i]` = token i + 1) at MTP positions 0.. (embedding half first,
    /// residual stream, offset 0: the wiring the chip sweep chose). Returns the head's output
    /// after `mtp.norm`, `[m][hidden]`, ready for the lm_head.
    pub fn lab_mtp(&mut self, hidden: &[f32], next: &[u32]) -> Result<Vec<f32>, String> {
        let c = self.config.clone();
        let h = c.hidden;
        let m = next.len();
        if hidden.len() != m * h {
            return Err("lab_mtp: hidden / next lengths differ".into());
        }
        let src: &mut dyn TensorSource = &mut self.source;
        let fc = read_matrix(src, "mtp.fc.weight", h, 2 * h)?;
        let norm_e = read_vec(src, "mtp.pre_fc_norm_embedding.weight", h)?;
        let norm_h = read_vec(src, "mtp.pre_fc_norm_hidden.weight", h)?;
        let norm_f = read_vec(src, "mtp.norm.weight", h)?;
        let a = "mtp.layers.0.self_attn";
        let attn = AttentionWeights {
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
        };
        let mm = "mtp.layers.0.mlp";
        let layer = LayerWeights {
            input_norm: read_vec(src, "mtp.layers.0.input_layernorm.weight", h)?,
            post_norm: read_vec(src, "mtp.layers.0.post_attention_layernorm.weight", h)?,
            mixer: Mixer::Attention(attn),
            gate: read_matrix(src, &format!("{mm}.gate_proj.weight"), c.intermediate, h)?,
            up: read_matrix(src, &format!("{mm}.up_proj.weight"), c.intermediate, h)?,
            down: read_matrix(src, &format!("{mm}.down_proj.weight"), h, c.intermediate)?,
        };
        let mut cat = vec![0f32; m * 2 * h];
        for i in 0..m {
            let mut e = self.lab_embed_row(next[i]);
            rms_norm_zc(&mut e, &norm_e, c.eps);
            let mut hh = hidden[i * h..(i + 1) * h].to_vec();
            rms_norm_zc(&mut hh, &norm_h, c.eps);
            cat[i * 2 * h..i * 2 * h + h].copy_from_slice(&e);
            cat[i * 2 * h + h..(i + 1) * 2 * h].copy_from_slice(&hh);
        }
        let mut x = fast_matmul(&fc, &cat, m);
        let mut st = LayerState::Attention {
            k: Vec::new(),
            v: Vec::new(),
        };
        let one = Qwen38Config {
            full_attention: vec![true],
            layers: 1,
            ..c.clone()
        };
        let ctx = Ctx {
            c: &one,
            score_start: m,
            dyn_skip: &vec![false; m],
        };
        let view = View { w: &layer, q: None };
        layer_fwd(&ctx, &view, &mut st, &mut x, m, 0, &Act::Full, &mut None)?;
        rms_norm_zc(&mut x, &norm_f, c.eps);
        Ok(x)
    }
}

/// Argmax and top-two margin per row of `[rows][vocab]` logits.
pub fn top2(logits: &[f32], vocab: usize) -> Vec<(u32, f32)> {
    logits
        .chunks_exact(vocab)
        .map(|r| {
            let a = argmax(r);
            let mut second = f32::NEG_INFINITY;
            for (i, &v) in r.iter().enumerate() {
                if i != a && v > second {
                    second = v;
                }
            }
            (a as u32, r[a] - second)
        })
        .collect()
}

/// KL(p || q) of two logit rows, nats.
pub fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |r: &[f32]| {
        let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let la = a as f64 - lp;
            la.exp() * (la - (b as f64 - lq))
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_matches_dot() {
        let (rows, k) = (5usize, 40usize);
        for t in [1usize, 3, 4, 7, 9] {
            let w: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let x: Vec<f32> = (0..t * k)
                .map(|i| ((i * 53 % 89) as f32 - 44.0) / 44.0)
                .collect();
            let mut out = vec![0f32; rows * t];
            kernel(&w, rows, k, &x, t, &mut out);
            for r in 0..rows {
                for tt in 0..t {
                    let want = dot(&w[r * k..(r + 1) * k], &x[tt * k..(tt + 1) * k]);
                    assert!((out[r * t + tt] - want).abs() < 1e-4, "r {r} t {tt}");
                }
            }
        }
    }

    #[test]
    fn mxfp4_group_matches_rule() {
        // amax 3.0 -> X = 2^(1 - 2) = 0.5, codes up to 6 * 0.5 = 3.0.
        let mut g = [0f32; GROUP];
        g[0] = 3.0;
        g[1] = -0.3; // 0.6 -> 0.5 -> -0.25
        g[2] = 1.1; // 2.2 -> 2.0 -> 1.0
        quant_group(Quant::Mxfp4, &mut g);
        assert_eq!((g[0], g[1], g[2]), (3.0, -0.25, 1.0));
        // int3 pow2: amax 3.0 -> X = 2^(1 - 1) = 1, codes 0..3.
        let mut h = [0f32; GROUP];
        h[0] = 3.0;
        h[1] = 1.4;
        h[2] = -2.6;
        quant_group(Quant::Mxint3, &mut h);
        assert_eq!((h[0], h[1], h[2]), (3.0, 1.0, -3.0));
    }

    #[test]
    fn teal_row_keeps_largest_share() {
        let tl = Teal {
            s: 0.5,
            group: 1,
            mode: TealMode::Row,
        };
        let mut x = vec![1.0f32, -4.0, 2.0, 0.5, 3.0, -0.1, 0.2, 5.0];
        tl.apply(&mut x, 8, 0, 1.0, None);
        assert_eq!(x, vec![0.0, -4.0, 2.0, 0.0, 3.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    #[ignore]
    fn bench_kernel_vs_reference() {
        let (n, k, t) = (2048usize, 5120usize, 288usize);
        let w: Vec<u16> = (0..n * k)
            .map(|i| crate::qwen3_moe::f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 500.0))
            .collect();
        let m = Matrix::Bf16 { n, k, w };
        let x: Vec<f32> = (0..t * k)
            .map(|i| ((i * 53 % 89) as f32 - 44.0) / 44.0)
            .collect();
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            let a = m.matmul(&x, t);
            let ta = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let b = fast_matmul(&m, &x, t);
            let tb = t0.elapsed().as_secs_f64();
            let md = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            eprintln!("reference {ta:.3} s, kernel {tb:.3} s, max diff {md:e}");
        }
    }
}
