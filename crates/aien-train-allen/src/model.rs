//! The Qwen3.5 dense text model (Gated DeltaNet + gated attention) with its backward pass.
//!
//! The forward is the maths of `aien_inference_abi::qwen38` (the host reference the student is
//! held to), written over whole sequences that start from an empty state, with every value the
//! backward needs kept. The backward is written out by hand per op. The base weights are frozen;
//! the trainable part is a low-rank adapter (LoRA, `y += s * (x A^T) B^T`) on the large linear
//! maps, merged into the weights when the run is saved.

use std::sync::atomic::{AtomicU8, Ordering};

use aien_inference_abi::qwen38::Qwen38Config;
use rayon::prelude::*;

use crate::gemm::{gemm, matmul, matmul_p, transpose, Packed};

/// TEST ONLY negative control for the gradient check: 0 off; 1 drops the `S' dkv` term of the
/// DeltaNet key gradient; 2 turns RoPE forwards instead of back in the attention backward.
pub static MUTANT: AtomicU8 = AtomicU8::new(0);

fn mutant() -> u8 {
    MUTANT.load(Ordering::Relaxed)
}

/// One trainable tensor with its gradient and AdamW moments.
pub struct Param {
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub m: Vec<f32>,
    pub s: Vec<f32>,
}

impl Param {
    pub fn new(v: Vec<f32>) -> Self {
        let n = v.len();
        Self {
            v,
            g: vec![0.0; n],
            m: vec![0.0; n],
            s: vec![0.0; n],
        }
    }
}

/// `y += scale * (x A^T) B^T`: A `[r][k]`, B `[n][r]`.
pub struct Lora {
    pub r: usize,
    pub scale: f32,
    pub a: Param,
    pub b: Param,
}

/// A frozen `[n][k]` map, packed for the forward (`W^T`) and backward (`W`) products, and its
/// adapter.
pub struct Lin {
    pub n: usize,
    pub k: usize,
    pub wt_p: Packed,
    pub w_p: Packed,
    pub lora: Option<Lora>,
}

impl Lin {
    pub fn new(w: Vec<f32>, n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        let wt_p = Packed::new(&transpose(&w, n, k), k, n);
        let w_p = Packed::new(&w, n, k);
        Self {
            n,
            k,
            wt_p,
            w_p,
            lora: None,
        }
    }

    /// The frozen map, `[n][k]`.
    pub fn w(&self) -> Vec<f32> {
        self.w_p.unpack()
    }

    /// `y` `[t][n]` and the adapter's `x A^T` `[t][r]` (empty without an adapter).
    pub fn fwd(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut y = matmul_p(x, &self.wt_p, t);
        let Some(l) = &self.lora else {
            return (y, Vec::new());
        };
        let u = matmul(x, &transpose(&l.a.v, l.r, self.k), t, self.k, l.r);
        let us: Vec<f32> = u.iter().map(|v| v * l.scale).collect();
        gemm(
            &us,
            &transpose(&l.b.v, self.n, l.r),
            &mut y,
            t,
            l.r,
            self.n,
            true,
        );
        (y, u)
    }

    /// `dx` for `dy`; adds the adapter's gradients.
    pub fn bwd(&mut self, x: &[f32], u: &[f32], dy: &[f32], t: usize) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        let mut dx = matmul_p(dy, &self.w_p, t);
        if let Some(l) = &mut self.lora {
            let r = l.r;
            let mut du = matmul(dy, &l.b.v, t, n, r);
            du.iter_mut().for_each(|v| *v *= l.scale);
            let us: Vec<f32> = u.iter().map(|v| v * l.scale).collect();
            gemm(&transpose(dy, t, n), &us, &mut l.b.g, n, t, r, true);
            gemm(&transpose(&du, t, r), x, &mut l.a.g, r, t, k, true);
            gemm(&du, &l.a.v, &mut dx, t, r, k, true);
        }
        dx
    }

    /// The frozen map with the adapter folded in, `[n][k]`.
    pub fn merged(&self) -> Vec<f32> {
        let mut w = self.w();
        if let Some(l) = &self.lora {
            let ba = matmul(&l.b.v, &l.a.v, self.n, l.r, self.k);
            w.iter_mut().zip(&ba).for_each(|(w, d)| *w += l.scale * d);
        }
        w
    }
}

pub struct Gdn {
    pub qkv: Lin,
    pub z: Lin,
    pub a: Lin,
    pub b: Lin,
    /// `[conv_dim][kernel]`.
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: Lin,
}

pub struct Attn {
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

pub struct Mlp {
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
}

pub struct Layer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

pub struct Model {
    pub c: Qwen38Config,
    /// `[vocab][hidden]`, also the (tied) lm_head.
    pub embed: Vec<f32>,
    /// The embedding packed as the head's `[hidden][vocab]` and as its backward's `[vocab][hidden]`.
    pub head_p: Packed,
    pub head_bwd_p: Packed,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
}

// ---------------------------------------------------------------------------
// Small pieces and their derivatives.
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn dsilu(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// PyTorch's softplus (beta 1, threshold 20) and its derivative.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn dsoftplus(x: f32) -> f32 {
    if x > 20.0 {
        1.0
    } else {
        sigmoid(x)
    }
}

/// Zero-centred RMSNorm over each `w.len()` run: `x * rsqrt(mean(x^2) + eps) * (1 + w)`.
pub fn rms_zc(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    y.par_chunks_mut(w.len()).for_each(|row| {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(w) {
            *v = *v * inv * (1.0 + g);
        }
    });
    y
}

/// Backward of [`rms_zc`]: `dx = inv * (g - x * inv^2 * mean(g x))` with `g = dy (1 + w)`.
pub fn rms_zc_bwd(x: &[f32], w: &[f32], eps: f32, dy: &[f32]) -> Vec<f32> {
    let n = w.len();
    let mut dx = vec![0f32; x.len()];
    dx.par_chunks_mut(n)
        .zip(x.par_chunks(n).zip(dy.par_chunks(n)))
        .for_each(|(dx, (x, dy))| {
            let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            let s: f32 = (0..n).map(|i| dy[i] * (1.0 + w[i]) * x[i]).sum::<f32>() / n as f32;
            for i in 0..n {
                dx[i] = inv * (dy[i] * (1.0 + w[i]) - x[i] * inv * inv * s);
            }
        });
    dx
}

/// Plain RMSNorm backward with gradient `g` on the normalised value: `inv (g - x inv^2 mean(g x))`.
fn rms_plain_bwd(x: &[f32], g: &[f32], eps: f32, dx: &mut [f32]) {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    let s: f32 = x.iter().zip(g).map(|(a, b)| a * b).sum::<f32>() / n as f32;
    for i in 0..n {
        dx[i] = inv * (g[i] - x[i] * inv * inv * s);
    }
}

fn l2_inv(x: &[f32]) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt()
}

/// Backward of `y = x / sqrt(sum x^2 + eps)`: `dx = (dy - y (y . dy)) / n`.
fn l2_bwd(x: &[f32], dy: &[f32], dx: &mut [f32]) {
    let inv = l2_inv(x);
    let yd: f32 = x.iter().zip(dy).map(|(a, b)| a * inv * b).sum();
    for i in 0..x.len() {
        dx[i] = (dy[i] - x[i] * inv * yd) * inv;
    }
}

/// `cos`, `sin` per position and pair, `[t][rotary/2]`, from f64 like the reference.
pub struct Rope {
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    pub fn new(t: usize, rotary: usize, theta: f64) -> Self {
        let half = rotary / 2;
        let (mut cos, mut sin) = (Vec::with_capacity(t * half), Vec::with_capacity(t * half));
        for pos in 0..t {
            for j in 0..half {
                let inv_freq = 1.0 / theta.powf((2 * j) as f64 / rotary as f64);
                let (s, c) = (pos as f64 * inv_freq).sin_cos();
                cos.push(c as f32);
                sin.push(s as f32);
            }
        }
        Self { half, cos, sin }
    }

    /// Turns `x` (one head) at `pos`; `back` applies the transpose (the backward).
    fn apply(&self, x: &mut [f32], pos: usize, back: bool) {
        let h = self.half;
        for j in 0..h {
            let (c, s) = (self.cos[pos * h + j], self.sin[pos * h + j]);
            let s = if back { -s } else { s };
            let (x1, x2) = (x[j], x[j + h]);
            x[j] = x1 * c - x2 * s;
            x[j + h] = x2 * c + x1 * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Caches.
// ---------------------------------------------------------------------------

pub struct MlpCache {
    ug: Vec<f32>,
    uu: Vec<f32>,
    ud: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    act: Vec<f32>,
}

pub struct AttnCache {
    uq: Vec<f32>,
    uk: Vec<f32>,
    uv: Vec<f32>,
    uo: Vec<f32>,
    /// q and k before their norm, `[t][heads][hd]` and `[t][kv_heads][hd]`.
    qraw: Vec<f32>,
    kraw: Vec<f32>,
    gate: Vec<f32>,
    v: Vec<f32>,
    /// After norm and RoPE.
    qr: Vec<f32>,
    kr: Vec<f32>,
    /// `[heads][t][t]`, zero above the diagonal.
    p: Vec<f32>,
    oraw: Vec<f32>,
    out: Vec<f32>,
}

pub struct GdnCache {
    uqkv: Vec<f32>,
    uz: Vec<f32>,
    ua: Vec<f32>,
    ub: Vec<f32>,
    uo: Vec<f32>,
    /// Conv output before SiLU and after, `[t][conv_dim]`.
    pre: Vec<f32>,
    mixed: Vec<f32>,
    /// Normalised (q scaled), `[t][k_heads * k_dim]`.
    q: Vec<f32>,
    k: Vec<f32>,
    /// `a + dt_bias`, `g` and `beta`, `[t][v_heads]`.
    ax: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    z: Vec<f32>,
    /// Recurrence output before and after the gated norm, `[t][v_heads * v_dim]`.
    core_raw: Vec<f32>,
    core: Vec<f32>,
}

pub enum MixCache {
    Gdn(GdnCache),
    Attn(AttnCache),
}

pub struct LayerCache {
    x: Vec<f32>,
    xn1: Vec<f32>,
    mix: MixCache,
    x_mid: Vec<f32>,
    xn2: Vec<f32>,
    mlp: MlpCache,
}

// ---------------------------------------------------------------------------
// MLP.
// ---------------------------------------------------------------------------

impl Mlp {
    fn fwd(&self, xn: &[f32], t: usize) -> (Vec<f32>, MlpCache) {
        let (g, ug) = self.gate.fwd(xn, t);
        let (u, uu) = self.up.fwd(xn, t);
        let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
        let (y, ud) = self.down.fwd(&act, t);
        (
            y,
            MlpCache {
                ug,
                uu,
                ud,
                g,
                u,
                act,
            },
        )
    }

    fn bwd(&mut self, xn: &[f32], c: &MlpCache, dy: &[f32], t: usize) -> Vec<f32> {
        let dact = self.down.bwd(&c.act, &c.ud, dy, t);
        let dg: Vec<f32> = (0..dact.len())
            .map(|i| dact[i] * c.u[i] * dsilu(c.g[i]))
            .collect();
        let du: Vec<f32> = (0..dact.len()).map(|i| dact[i] * silu(c.g[i])).collect();
        let mut dx = self.gate.bwd(xn, &c.ug, &dg, t);
        let d2 = self.up.bwd(xn, &c.uu, &du, t);
        dx.iter_mut().zip(&d2).for_each(|(a, b)| *a += b);
        dx
    }
}

// ---------------------------------------------------------------------------
// Gated attention.
// ---------------------------------------------------------------------------

impl Attn {
    fn fwd(&self, c: &Qwen38Config, rope: &Rope, xn: &[f32], t: usize) -> (Vec<f32>, AttnCache) {
        let (nh, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
        let kvd = kvh * hd;
        let group = nh / kvh;
        let (qg, uq) = self.q.fwd(xn, t);
        let (kraw, uk) = self.k.fwd(xn, t);
        let (v, uv) = self.v.fwd(xn, t);
        let mut qraw = vec![0f32; t * nh * hd];
        let mut gate = vec![0f32; t * nh * hd];
        for i in 0..t {
            for hh in 0..nh {
                let src = &qg[(i * nh + hh) * 2 * hd..(i * nh + hh + 1) * 2 * hd];
                qraw[(i * nh + hh) * hd..(i * nh + hh + 1) * hd].copy_from_slice(&src[..hd]);
                gate[(i * nh + hh) * hd..(i * nh + hh + 1) * hd].copy_from_slice(&src[hd..]);
            }
        }
        let mut qr = rms_zc(&qraw, &self.q_norm, c.eps);
        let mut kr = rms_zc(&kraw, &self.k_norm, c.eps);
        for i in 0..t {
            for hh in 0..nh {
                rope.apply(
                    &mut qr[(i * nh + hh) * hd..(i * nh + hh + 1) * hd],
                    i,
                    false,
                );
            }
            for hh in 0..kvh {
                rope.apply(
                    &mut kr[(i * kvh + hh) * hd..(i * kvh + hh + 1) * hd],
                    i,
                    false,
                );
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut p = vec![0f32; nh * t * t];
        p.par_chunks_mut(t).enumerate().for_each(|(row, pr)| {
            let (hh, i) = (row / t, row % t);
            let kh = hh / group;
            let qv = &qr[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let kv = &kr[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                pr[j] = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                max = max.max(pr[j]);
            }
            let mut sum = 0f32;
            for v in pr[..=i].iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            pr[..=i].iter_mut().for_each(|v| *v /= sum);
        });
        let mut oraw = vec![0f32; t * nh * hd];
        oraw.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
            let (i, hh) = (idx / nh, idx % nh);
            let kh = hh / group;
            let pr = &p[(hh * t + i) * t..(hh * t + i + 1) * t];
            for j in 0..=i {
                let vr = &v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                for (ov, &vv) in o.iter_mut().zip(vr) {
                    *ov += pr[j] * vv;
                }
            }
        });
        let out: Vec<f32> = oraw
            .iter()
            .zip(&gate)
            .map(|(&o, &g)| o * sigmoid(g))
            .collect();
        let (y, uo) = self.o.fwd(&out, t);
        (
            y,
            AttnCache {
                uq,
                uk,
                uv,
                uo,
                qraw,
                kraw,
                gate,
                v,
                qr,
                kr,
                p,
                oraw,
                out,
            },
        )
    }

    fn bwd(
        &mut self,
        c: &Qwen38Config,
        rope: &Rope,
        xn: &[f32],
        a: &AttnCache,
        dy: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let (nh, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
        let kvd = kvh * hd;
        let group = nh / kvh;
        let scale = 1.0 / (hd as f32).sqrt();
        let dout = self.o.bwd(&a.out, &a.uo, dy, t);
        let mut doraw = vec![0f32; t * nh * hd];
        let mut dgate = vec![0f32; t * nh * hd];
        for i in 0..dout.len() {
            let s = sigmoid(a.gate[i]);
            doraw[i] = dout[i] * s;
            dgate[i] = dout[i] * a.oraw[i] * s * (1.0 - s);
        }
        // dS = P (dP - sum_j P dP), dP_ij = do_i . v_j.
        let mut ds = vec![0f32; nh * t * t];
        ds.par_chunks_mut(t).enumerate().for_each(|(row, dsr)| {
            let (hh, i) = (row / t, row % t);
            let kh = hh / group;
            let pr = &a.p[row * t..(row + 1) * t];
            let dor = &doraw[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            let mut dot = 0f32;
            for j in 0..=i {
                let vr = &a.v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                dsr[j] = dor.iter().zip(vr).map(|(x, y)| x * y).sum();
                dot += pr[j] * dsr[j];
            }
            for j in 0..=i {
                dsr[j] = pr[j] * (dsr[j] - dot);
            }
        });
        let mut dq = vec![0f32; t * nh * hd];
        dq.par_chunks_mut(hd).enumerate().for_each(|(idx, dqr)| {
            let (i, hh) = (idx / nh, idx % nh);
            let kh = hh / group;
            let dsr = &ds[(hh * t + i) * t..(hh * t + i + 1) * t];
            for j in 0..=i {
                let kv = &a.kr[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                for (o, &kk) in dqr.iter_mut().zip(kv) {
                    *o += scale * dsr[j] * kk;
                }
            }
        });
        // dk_j and dv_j per kv head: over the heads of its group and the rows i >= j.
        let mut dk = vec![0f32; t * kvd];
        let mut dv = vec![0f32; t * kvd];
        dk.par_chunks_mut(hd)
            .zip(dv.par_chunks_mut(hd))
            .enumerate()
            .for_each(|(idx, (dkr, dvr))| {
                let (j, kh) = (idx / kvh, idx % kvh);
                for hh in kh * group..(kh + 1) * group {
                    for i in j..t {
                        let row = hh * t + i;
                        let (s, pp) = (ds[row * t + j], a.p[row * t + j]);
                        let qv = &a.qr[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
                        let dor = &doraw[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
                        for e in 0..hd {
                            dkr[e] += scale * s * qv[e];
                            dvr[e] += pp * dor[e];
                        }
                    }
                }
            });
        let back = mutant() != 2;
        for i in 0..t {
            for hh in 0..nh {
                rope.apply(&mut dq[(i * nh + hh) * hd..(i * nh + hh + 1) * hd], i, back);
            }
            for hh in 0..kvh {
                rope.apply(
                    &mut dk[(i * kvh + hh) * hd..(i * kvh + hh + 1) * hd],
                    i,
                    back,
                );
            }
        }
        let dqraw = rms_zc_bwd(&a.qraw, &self.q_norm, c.eps, &dq);
        let dkraw = rms_zc_bwd(&a.kraw, &self.k_norm, c.eps, &dk);
        let mut dqg = vec![0f32; t * nh * 2 * hd];
        for i in 0..t {
            for hh in 0..nh {
                let d = &mut dqg[(i * nh + hh) * 2 * hd..(i * nh + hh + 1) * 2 * hd];
                d[..hd].copy_from_slice(&dqraw[(i * nh + hh) * hd..(i * nh + hh + 1) * hd]);
                d[hd..].copy_from_slice(&dgate[(i * nh + hh) * hd..(i * nh + hh + 1) * hd]);
            }
        }
        let mut dx = self.q.bwd(xn, &a.uq, &dqg, t);
        for d in [
            self.k.bwd(xn, &a.uk, &dkraw, t),
            self.v.bwd(xn, &a.uv, &dv, t),
        ] {
            dx.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        }
        dx
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet.
// ---------------------------------------------------------------------------

/// Per value head: the states after each step, `[t][k_dim][v_dim]` (recomputed in the backward).
fn gdn_states(
    c: &Qwen38Config,
    g: &GdnCache,
    hv: usize,
    t: usize,
    out: Option<&mut Vec<f32>>,
) -> Vec<f32> {
    let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
    let (key_dim, cd) = (kh * kd, c.conv_dim());
    let hk = hv / (vh / kh);
    let mut s = vec![0f32; kd * vd];
    let mut all = Vec::new();
    let keep = out.is_none();
    let mut o = vec![0f32; t * vd];
    let mut kv = vec![0f32; vd];
    for i in 0..t {
        let kt = &g.k[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
        let qt = &g.q[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
        let vt = &g.mixed[i * cd + 2 * key_dim + hv * vd..i * cd + 2 * key_dim + (hv + 1) * vd];
        let (decay, beta) = (g.g[i * vh + hv].exp(), g.beta[i * vh + hv]);
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
        let oi = &mut o[i * vd..(i + 1) * vd];
        for (r, &qr) in qt.iter().enumerate() {
            for (ov, &sv) in oi.iter_mut().zip(&s[r * vd..(r + 1) * vd]) {
                *ov += sv * qr;
            }
        }
        if keep {
            all.extend_from_slice(&s);
        }
    }
    if let Some(out) = out {
        *out = o;
    }
    all
}

/// Gradients of one value head's recurrence: `(dq, dk, dv, dg, dbeta)`, `[t][k_dim]` twice,
/// `[t][v_dim]`, `[t]` twice.
#[allow(clippy::type_complexity)]
fn gdn_head_bwd(
    c: &Qwen38Config,
    g: &GdnCache,
    hv: usize,
    t: usize,
    dcore_raw: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
    let (key_dim, cd) = (kh * kd, c.conv_dim());
    let hk = hv / (vh / kh);
    let states = gdn_states(c, g, hv, t, None);
    let zero = vec![0f32; kd * vd];
    let (mut dq, mut dk) = (vec![0f32; t * kd], vec![0f32; t * kd]);
    let (mut dv, mut dg, mut dbeta) = (vec![0f32; t * vd], vec![0f32; t], vec![0f32; t]);
    let mut ds = vec![0f32; kd * vd];
    let (mut sp, mut kv, mut dd, mut dkv) = (
        vec![0f32; kd * vd],
        vec![0f32; vd],
        vec![0f32; vd],
        vec![0f32; vd],
    );
    for i in (0..t).rev() {
        let s = &states[i * kd * vd..(i + 1) * kd * vd];
        let prev = if i == 0 {
            &zero[..]
        } else {
            &states[(i - 1) * kd * vd..i * kd * vd]
        };
        let kt = &g.k[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
        let qt = &g.q[i * key_dim + hk * kd..i * key_dim + (hk + 1) * kd];
        let vt = &g.mixed[i * cd + 2 * key_dim + hv * vd..i * cd + 2 * key_dim + (hv + 1) * vd];
        let (decay, beta) = (g.g[i * vh + hv].exp(), g.beta[i * vh + hv]);
        let dout = &dcore_raw[i * vh * vd + hv * vd..i * vh * vd + (hv + 1) * vd];
        // o = S^T q
        for r in 0..kd {
            let row = &s[r * vd..(r + 1) * vd];
            dq[i * kd + r] = row.iter().zip(dout).map(|(a, b)| a * b).sum();
            for (d, &o) in ds[r * vd..(r + 1) * vd].iter_mut().zip(dout) {
                *d += qt[r] * o;
            }
        }
        // S' = decay * prev; kv = S'^T k; delta = beta (v - kv); S = S' + k delta^T
        for (a, &b) in sp.iter_mut().zip(prev) {
            *a = decay * b;
        }
        kv.iter_mut().for_each(|v| *v = 0.0);
        for r in 0..kd {
            for (o, &sv) in kv.iter_mut().zip(&sp[r * vd..(r + 1) * vd]) {
                *o += sv * kt[r];
            }
        }
        dd.iter_mut().for_each(|v| *v = 0.0);
        for r in 0..kd {
            let row = &ds[r * vd..(r + 1) * vd];
            let delta_dot: f32 = row
                .iter()
                .zip(vt.iter().zip(&kv))
                .map(|(d, (&v, &m))| d * (v - m) * beta)
                .sum();
            dk[i * kd + r] += delta_dot;
            for (o, &d) in dd.iter_mut().zip(row) {
                *o += d * kt[r];
            }
        }
        dbeta[i] = dd
            .iter()
            .zip(vt.iter().zip(&kv))
            .map(|(d, (&v, &m))| d * (v - m))
            .sum();
        for c_ in 0..vd {
            dv[i * vd + c_] = beta * dd[c_];
            dkv[c_] = -beta * dd[c_];
        }
        // kv = S'^T k: dS' = dS + k dkv^T, dk += S' dkv
        for r in 0..kd {
            if mutant() != 1 {
                dk[i * kd + r] += sp[r * vd..(r + 1) * vd]
                    .iter()
                    .zip(&dkv)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
            for (d, &x) in ds[r * vd..(r + 1) * vd].iter_mut().zip(&dkv) {
                *d += kt[r] * x;
            }
        }
        // S' = decay * prev
        let ddecay: f32 = ds.iter().zip(prev).map(|(a, b)| a * b).sum();
        dg[i] = ddecay * decay;
        ds.iter_mut().for_each(|v| *v *= decay);
    }
    (dq, dk, dv, dg, dbeta)
}

impl Gdn {
    fn fwd(&self, c: &Qwen38Config, xn: &[f32], t: usize) -> (Vec<f32>, GdnCache) {
        let (cd, kern) = (c.conv_dim(), c.conv_kernel);
        let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
        let key_dim = kh * kd;
        let (qkv, uqkv) = self.qkv.fwd(xn, t);
        let (z, uz) = self.z.fwd(xn, t);
        let (a, ua) = self.a.fwd(xn, t);
        let (b, ub) = self.b.fwd(xn, t);
        let hist = kern - 1;
        let mut pre = vec![0f32; t * cd];
        pre.par_chunks_mut(cd).enumerate().for_each(|(i, row)| {
            for (ch, o) in row.iter_mut().enumerate() {
                let taps = &self.conv[ch * kern..(ch + 1) * kern];
                *o = (0..kern)
                    .map(|j| {
                        let p = i + j;
                        if p < hist {
                            0.0
                        } else {
                            taps[j] * qkv[(p - hist) * cd + ch]
                        }
                    })
                    .sum();
            }
        });
        let mixed: Vec<f32> = pre.iter().map(|&v| silu(v)).collect();
        let q_scale = 1.0 / (kd as f32).sqrt();
        let mut q = vec![0f32; t * key_dim];
        let mut k = vec![0f32; t * key_dim];
        for i in 0..t {
            let row = &mixed[i * cd..(i + 1) * cd];
            for hh in 0..kh {
                let (qs, ks) = (
                    &row[hh * kd..(hh + 1) * kd],
                    &row[key_dim + hh * kd..key_dim + (hh + 1) * kd],
                );
                let (qi, ki) = (l2_inv(qs), l2_inv(ks));
                let o = i * key_dim + hh * kd;
                for e in 0..kd {
                    q[o + e] = qs[e] * qi * q_scale;
                    k[o + e] = ks[e] * ki;
                }
            }
        }
        let mut ax = vec![0f32; t * vh];
        let mut gg = vec![0f32; t * vh];
        let mut beta = vec![0f32; t * vh];
        for i in 0..t {
            for hv in 0..vh {
                let x = a[i * vh + hv] + self.dt_bias[hv];
                ax[i * vh + hv] = x;
                gg[i * vh + hv] = -self.a_log[hv].exp() * softplus(x);
                beta[i * vh + hv] = sigmoid(b[i * vh + hv]);
            }
        }
        let mut cache = GdnCache {
            uqkv,
            uz,
            ua,
            ub,
            uo: Vec::new(),
            pre,
            mixed,
            q,
            k,
            ax,
            g: gg,
            beta,
            z,
            core_raw: Vec::new(),
            core: Vec::new(),
        };
        let per_head: Vec<Vec<f32>> = (0..vh)
            .into_par_iter()
            .map(|hv| {
                let mut o = Vec::new();
                gdn_states(c, &cache, hv, t, Some(&mut o));
                o
            })
            .collect();
        let mut core_raw = vec![0f32; t * vh * vd];
        for (hv, o) in per_head.iter().enumerate() {
            for i in 0..t {
                core_raw[i * vh * vd + hv * vd..i * vh * vd + (hv + 1) * vd]
                    .copy_from_slice(&o[i * vd..(i + 1) * vd]);
            }
        }
        let mut core = core_raw.clone();
        for (o, zr) in core.chunks_exact_mut(vd).zip(cache.z.chunks_exact(vd)) {
            let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (ms + c.eps).sqrt();
            for ((v, &g), &zz) in o.iter_mut().zip(&self.norm).zip(zr) {
                *v = *v * inv * g * silu(zz);
            }
        }
        let (y, uo) = self.out.fwd(&core, t);
        cache.uo = uo;
        cache.core_raw = core_raw;
        cache.core = core;
        (y, cache)
    }

    fn bwd(
        &mut self,
        c: &Qwen38Config,
        xn: &[f32],
        g: &GdnCache,
        dy: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let (cd, kern) = (c.conv_dim(), c.conv_kernel);
        let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
        let key_dim = kh * kd;
        let ratio = vh / kh;
        let dcore = self.out.bwd(&g.core, &g.uo, dy, t);
        // Gated norm: y = o * inv * w * silu(z).
        let mut dz = vec![0f32; t * vh * vd];
        let mut dcore_raw = vec![0f32; t * vh * vd];
        for r in 0..t * vh {
            let o = &g.core_raw[r * vd..(r + 1) * vd];
            let zr = &g.z[r * vd..(r + 1) * vd];
            let dyr = &dcore[r * vd..(r + 1) * vd];
            let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (ms + c.eps).sqrt();
            let mut gn = vec![0f32; vd];
            for e in 0..vd {
                dz[r * vd + e] = dyr[e] * o[e] * inv * self.norm[e] * dsilu(zr[e]);
                gn[e] = dyr[e] * self.norm[e] * silu(zr[e]);
            }
            rms_plain_bwd(o, &gn, c.eps, &mut dcore_raw[r * vd..(r + 1) * vd]);
        }
        let heads: Vec<_> = (0..vh)
            .into_par_iter()
            .map(|hv| gdn_head_bwd(c, g, hv, t, &dcore_raw))
            .collect();
        let mut dq = vec![0f32; t * key_dim];
        let mut dk = vec![0f32; t * key_dim];
        let mut dmixed = vec![0f32; t * cd];
        let mut da = vec![0f32; t * vh];
        let mut db = vec![0f32; t * vh];
        for (hv, (hq, hk_, hvv, hg, hb)) in heads.iter().enumerate() {
            let hk = hv / ratio;
            for i in 0..t {
                for e in 0..kd {
                    dq[i * key_dim + hk * kd + e] += hq[i * kd + e];
                    dk[i * key_dim + hk * kd + e] += hk_[i * kd + e];
                }
                dmixed[i * cd + 2 * key_dim + hv * vd..i * cd + 2 * key_dim + (hv + 1) * vd]
                    .copy_from_slice(&hvv[i * vd..(i + 1) * vd]);
                let x = g.ax[i * vh + hv];
                da[i * vh + hv] = hg[i] * -self.a_log[hv].exp() * dsoftplus(x);
                let be = g.beta[i * vh + hv];
                db[i * vh + hv] = hb[i] * be * (1.0 - be);
            }
        }
        let q_scale = 1.0 / (kd as f32).sqrt();
        for i in 0..t {
            let row = &g.mixed[i * cd..(i + 1) * cd];
            for hh in 0..kh {
                let o = i * key_dim + hh * kd;
                let dqs: Vec<f32> = dq[o..o + kd].iter().map(|v| v * q_scale).collect();
                l2_bwd(
                    &row[hh * kd..(hh + 1) * kd],
                    &dqs,
                    &mut dmixed[i * cd + hh * kd..i * cd + (hh + 1) * kd],
                );
                l2_bwd(
                    &row[key_dim + hh * kd..key_dim + (hh + 1) * kd],
                    &dk[o..o + kd],
                    &mut dmixed[i * cd + key_dim + hh * kd..i * cd + key_dim + (hh + 1) * kd],
                );
            }
        }
        let dpre: Vec<f32> = dmixed
            .iter()
            .zip(&g.pre)
            .map(|(&d, &p)| d * dsilu(p))
            .collect();
        // pre[i] = sum_j w[j] seq[i + j], seq[p] = qkv[p - hist] for p >= hist.
        let hist = kern - 1;
        let mut dqkv = vec![0f32; t * cd];
        dqkv.par_chunks_mut(cd).enumerate().for_each(|(p, row)| {
            for (ch, o) in row.iter_mut().enumerate() {
                let taps = &self.conv[ch * kern..(ch + 1) * kern];
                let mut acc = 0f32;
                for (j, &w) in taps.iter().enumerate() {
                    let i = p + hist;
                    if i >= j && i - j < t {
                        acc += w * dpre[(i - j) * cd + ch];
                    }
                }
                *o = acc;
            }
        });
        let mut dx = self.qkv.bwd(xn, &g.uqkv, &dqkv, t);
        for d in [
            self.z.bwd(xn, &g.uz, &dz, t),
            self.a.bwd(xn, &g.ua, &da, t),
            self.b.bwd(xn, &g.ub, &db, t),
        ] {
            dx.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        }
        dx
    }
}

// ---------------------------------------------------------------------------
// The whole model.
// ---------------------------------------------------------------------------

/// What one sequence gave: the weighted loss sum, the weight sum and the weighted hits.
#[derive(Debug, Default, Clone, Copy)]
pub struct SeqLoss {
    pub loss: f64,
    pub weight: f64,
    pub hits: f64,
}

impl Model {
    fn layer_fwd(&self, l: usize, rope: &Rope, x: Vec<f32>, t: usize) -> (Vec<f32>, LayerCache) {
        let c = &self.c;
        let lw = &self.layers[l];
        let xn1 = rms_zc(&x, &lw.input_norm, c.eps);
        let (mixed, mix) = match &lw.mixer {
            Mixer::Gdn(d) => {
                let (y, cache) = d.fwd(c, &xn1, t);
                (y, MixCache::Gdn(cache))
            }
            Mixer::Attn(a) => {
                let (y, cache) = a.fwd(c, rope, &xn1, t);
                (y, MixCache::Attn(cache))
            }
        };
        let x_mid: Vec<f32> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let xn2 = rms_zc(&x_mid, &lw.post_norm, c.eps);
        let (down, mlp) = lw.mlp.fwd(&xn2, t);
        let next: Vec<f32> = x_mid.iter().zip(&down).map(|(a, b)| a + b).collect();
        (
            next,
            LayerCache {
                x,
                xn1,
                mix,
                x_mid,
                xn2,
                mlp,
            },
        )
    }

    fn layer_bwd(
        &mut self,
        l: usize,
        rope: &Rope,
        lc: &LayerCache,
        dnext: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let c = self.c.clone();
        let lw = &mut self.layers[l];
        let dxn2 = lw.mlp.bwd(&lc.xn2, &lc.mlp, dnext, t);
        let mut dmid = rms_zc_bwd(&lc.x_mid, &lw.post_norm, c.eps, &dxn2);
        dmid.iter_mut().zip(dnext).for_each(|(a, b)| *a += b);
        let dxn1 = match (&mut lw.mixer, &lc.mix) {
            (Mixer::Gdn(d), MixCache::Gdn(g)) => d.bwd(&c, &lc.xn1, g, &dmid, t),
            (Mixer::Attn(a), MixCache::Attn(ac)) => a.bwd(&c, rope, &lc.xn1, ac, &dmid, t),
            _ => unreachable!("layer cache kind differs from the layer"),
        };
        let mut dx = rms_zc_bwd(&lc.x, &lw.input_norm, c.eps, &dxn1);
        dx.iter_mut().zip(&dmid).for_each(|(a, b)| *a += b);
        dx
    }

    fn embed_rows(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.c.hidden;
        let mut x = Vec::with_capacity(tokens.len() * h);
        for &tok in tokens {
            let tok = tok as usize;
            assert!(tok < self.c.vocab, "token {tok} outside the vocab");
            x.extend_from_slice(&self.embed[tok * h..(tok + 1) * h]);
        }
        x
    }

    /// Logits for every position, `[t][vocab]` (scoring and the parity check).
    pub fn logits(&self, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let rope = Rope::new(t, self.c.rotary_dim, self.c.rope_theta);
        let mut x = self.embed_rows(tokens);
        for l in 0..self.c.layers {
            x = self.layer_fwd(l, &rope, x, t).0;
        }
        let hn = rms_zc(&x, &self.final_norm, self.c.eps);
        matmul_p(&hn, &self.head_p, t)
    }

    /// `sum_i w_i * CE_i` without gradients (finite differences and validation).
    pub fn loss(&self, tokens: &[u32], weights: &[f32]) -> SeqLoss {
        let vocab = self.c.vocab;
        let lg = self.logits(tokens);
        let mut out = SeqLoss::default();
        for i in 0..tokens.len().saturating_sub(1) {
            let w = weights[i];
            if w <= 0.0 {
                continue;
            }
            let row = &lg[i * vocab..(i + 1) * vocab];
            let nll =
                aien_inference_abi::qwen38::negative_log_likelihood(row, tokens[i + 1] as usize);
            out.loss += nll * w as f64;
            out.weight += w as f64;
            if aien_inference_abi::qwen38::argmax(row) == tokens[i + 1] as usize {
                out.hits += w as f64;
            }
        }
        out
    }

    /// Forward and backward over one sequence. `weights[i]` weighs predicting `tokens[i + 1]` at
    /// position `i` (the last entry is unused). Adds `sum_i w_i * CE_i`'s gradient into the
    /// adapters and returns the sums.
    pub fn train_seq(&mut self, tokens: &[u32], weights: &[f32]) -> SeqLoss {
        let (t, h, vocab) = (tokens.len(), self.c.hidden, self.c.vocab);
        assert_eq!(weights.len(), t);
        let rope = Rope::new(t, self.c.rotary_dim, self.c.rope_theta);
        let mut x = self.embed_rows(tokens);
        let mut caches = Vec::with_capacity(self.c.layers);
        for l in 0..self.c.layers {
            let (next, lc) = self.layer_fwd(l, &rope, x, t);
            caches.push(lc);
            x = next;
        }
        let rows: Vec<usize> = (0..t.saturating_sub(1))
            .filter(|&i| weights[i] > 0.0)
            .collect();
        let mut out = SeqLoss::default();
        let mut dx = vec![0f32; t * h];
        const CHUNK: usize = 64;
        for chunk in rows.chunks(CHUNK) {
            let n = chunk.len();
            let mut xs = Vec::with_capacity(n * h);
            for &i in chunk {
                xs.extend_from_slice(&x[i * h..(i + 1) * h]);
            }
            let hn = rms_zc(&xs, &self.final_norm, self.c.eps);
            let mut lg = matmul_p(&hn, &self.head_p, n);
            let stats: Vec<(f64, f64, bool)> = lg
                .par_chunks_mut(vocab)
                .zip(chunk.par_iter())
                .map(|(row, &i)| {
                    let target = tokens[i + 1] as usize;
                    let w = weights[i];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut best = 0;
                    for (j, &v) in row.iter().enumerate() {
                        if v > row[best] {
                            best = j;
                        }
                    }
                    let sum: f64 = row.iter().map(|&v| ((v - max) as f64).exp()).sum();
                    let nll = max as f64 + sum.ln() - row[target] as f64;
                    for v in row.iter_mut() {
                        *v = (((*v - max) as f64).exp() / sum) as f32 * w;
                    }
                    row[target] -= w;
                    (nll * w as f64, w as f64, best == target)
                })
                .collect();
            for (nll, w, hit) in stats {
                out.loss += nll;
                out.weight += w;
                out.hits += if hit { w } else { 0.0 };
            }
            let dh = matmul_p(&lg, &self.head_bwd_p, n);
            let dxs = rms_zc_bwd(&xs, &self.final_norm, self.c.eps, &dh);
            for (r, &i) in chunk.iter().enumerate() {
                dx[i * h..(i + 1) * h].copy_from_slice(&dxs[r * h..(r + 1) * h]);
            }
        }
        for l in (0..self.c.layers).rev() {
            let lc = caches.pop().expect("one cache per layer");
            dx = self.layer_bwd(l, &rope, &lc, &dx, t);
        }
        out
    }

    /// Every adapter tensor, in a fixed order (layer, then map), with a name.
    pub fn params_mut(&mut self) -> Vec<(String, &mut Param)> {
        let mut out = Vec::new();
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let mut lins: Vec<(&str, &mut Lin)> = Vec::new();
            match &mut layer.mixer {
                Mixer::Gdn(d) => {
                    lins.push(("linear_attn.in_proj_qkv", &mut d.qkv));
                    lins.push(("linear_attn.in_proj_z", &mut d.z));
                    lins.push(("linear_attn.in_proj_a", &mut d.a));
                    lins.push(("linear_attn.in_proj_b", &mut d.b));
                    lins.push(("linear_attn.out_proj", &mut d.out));
                }
                Mixer::Attn(a) => {
                    lins.push(("self_attn.q_proj", &mut a.q));
                    lins.push(("self_attn.k_proj", &mut a.k));
                    lins.push(("self_attn.v_proj", &mut a.v));
                    lins.push(("self_attn.o_proj", &mut a.o));
                }
            }
            lins.push(("mlp.gate_proj", &mut layer.mlp.gate));
            lins.push(("mlp.up_proj", &mut layer.mlp.up));
            lins.push(("mlp.down_proj", &mut layer.mlp.down));
            for (name, lin) in lins {
                if let Some(lo) = &mut lin.lora {
                    out.push((format!("layers.{l}.{name}.lora_A"), &mut lo.a));
                    out.push((format!("layers.{l}.{name}.lora_B"), &mut lo.b));
                }
            }
        }
        out
    }

    /// Every adapted map with its checkpoint name (under the text prefix), for the merge.
    pub fn lins(&self) -> Vec<(String, &Lin)> {
        let mut out = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            let p = format!("layers.{l}");
            match &layer.mixer {
                Mixer::Gdn(d) => {
                    out.push((format!("{p}.linear_attn.in_proj_qkv.weight"), &d.qkv));
                    out.push((format!("{p}.linear_attn.in_proj_z.weight"), &d.z));
                    out.push((format!("{p}.linear_attn.in_proj_a.weight"), &d.a));
                    out.push((format!("{p}.linear_attn.in_proj_b.weight"), &d.b));
                    out.push((format!("{p}.linear_attn.out_proj.weight"), &d.out));
                }
                Mixer::Attn(a) => {
                    out.push((format!("{p}.self_attn.q_proj.weight"), &a.q));
                    out.push((format!("{p}.self_attn.k_proj.weight"), &a.k));
                    out.push((format!("{p}.self_attn.v_proj.weight"), &a.v));
                    out.push((format!("{p}.self_attn.o_proj.weight"), &a.o));
                }
            }
            out.push((format!("{p}.mlp.gate_proj.weight"), &layer.mlp.gate));
            out.push((format!("{p}.mlp.up_proj.weight"), &layer.mlp.up));
            out.push((format!("{p}.mlp.down_proj.weight"), &layer.mlp.down));
        }
        out
    }

    /// Puts a rank-`r` adapter on every large map (not the tiny `in_proj_a`/`in_proj_b`):
    /// A uniform in `+-1/sqrt(k)` from `seed`, B zero (so the start equals the base model).
    pub fn add_lora(&mut self, r: usize, alpha: f32, seed: u64, b_scale: f32) {
        let mut rng = SplitMix(seed);
        let mut put = |lin: &mut Lin| {
            let bound = 1.0 / (lin.k as f32).sqrt();
            let a = (0..r * lin.k).map(|_| rng.uniform() * bound).collect();
            let b = (0..lin.n * r).map(|_| rng.uniform() * b_scale).collect();
            lin.lora = Some(Lora {
                r,
                scale: alpha / r as f32,
                a: Param::new(a),
                b: Param::new(b),
            });
        };
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Gdn(d) => {
                    put(&mut d.qkv);
                    put(&mut d.z);
                    put(&mut d.out);
                }
                Mixer::Attn(a) => {
                    put(&mut a.q);
                    put(&mut a.k);
                    put(&mut a.v);
                    put(&mut a.o);
                }
            }
            put(&mut layer.mlp.gate);
            put(&mut layer.mlp.up);
            put(&mut layer.mlp.down);
        }
    }

    /// A model of shape `c` with seeded random weights (tests).
    pub fn random(c: &Qwen38Config, seed: u64) -> Self {
        let mut rng = SplitMix(seed);
        let mut mat = |n: usize, k: usize| {
            let bound = 1.0 / (k as f32).sqrt();
            Lin::new((0..n * k).map(|_| rng.uniform() * bound).collect(), n, k)
        };
        let h = c.hidden;
        let mut layers = Vec::new();
        for l in 0..c.layers {
            let mixer = if c.full_attention[l] {
                Mixer::Attn(Attn {
                    q: mat(2 * c.heads * c.head_dim, h),
                    k: mat(c.kv_dim(), h),
                    v: mat(c.kv_dim(), h),
                    o: mat(h, c.heads * c.head_dim),
                    q_norm: Vec::new(),
                    k_norm: Vec::new(),
                })
            } else {
                Mixer::Gdn(Gdn {
                    qkv: mat(c.conv_dim(), h),
                    z: mat(c.lin_value_dim(), h),
                    a: mat(c.lin_v_heads, h),
                    b: mat(c.lin_v_heads, h),
                    conv: Vec::new(),
                    a_log: Vec::new(),
                    dt_bias: Vec::new(),
                    norm: Vec::new(),
                    out: mat(h, c.lin_value_dim()),
                })
            };
            layers.push(Layer {
                input_norm: Vec::new(),
                post_norm: Vec::new(),
                mixer,
                mlp: Mlp {
                    gate: mat(c.intermediate, h),
                    up: mat(c.intermediate, h),
                    down: mat(h, c.intermediate),
                },
            });
        }
        let mut vecs = |n: usize, lo: f32, hi: f32| -> Vec<f32> {
            (0..n)
                .map(|_| lo + (hi - lo) * (rng.uniform() + 1.0) / 2.0)
                .collect()
        };
        for (l, layer) in layers.iter_mut().enumerate() {
            layer.input_norm = vecs(h, -0.3, 0.3);
            layer.post_norm = vecs(h, -0.3, 0.3);
            match &mut layer.mixer {
                Mixer::Attn(a) => {
                    a.q_norm = vecs(c.head_dim, -0.3, 0.3);
                    a.k_norm = vecs(c.head_dim, -0.3, 0.3);
                }
                Mixer::Gdn(d) => {
                    d.conv = vecs(c.conv_dim() * c.conv_kernel, -0.6, 0.6);
                    d.a_log = vecs(c.lin_v_heads, -1.0, 1.0);
                    d.dt_bias = vecs(c.lin_v_heads, -1.0, 1.0);
                    d.norm = vecs(c.lin_v_dim, 0.5, 1.5);
                }
            }
            let _ = l;
        }
        let embed = vecs(c.vocab * h, -1.0, 1.0);
        let final_norm = vecs(h, -0.3, 0.3);
        Self {
            c: c.clone(),
            head_p: Packed::new(&transpose(&embed, c.vocab, h), h, c.vocab),
            head_bwd_p: Packed::new(&embed, c.vocab, h),
            embed,
            final_norm,
            layers,
        }
    }
}

/// SplitMix64: a fixed, seedable source for initial values.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}
