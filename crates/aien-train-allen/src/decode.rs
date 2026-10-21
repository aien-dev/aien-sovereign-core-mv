//! Token-by-token generation over the trainer's model (evaluation of a trained student).
//!
//! The same maths as [`crate::model`] carried one position at a time with the DeltaNet state,
//! the conv tail and the attention cache kept between steps. Checked against the whole-sequence
//! forward in the tests.

use rayon::prelude::*;

use crate::model::{rms_zc, Attn, Gdn, Lin, Mixer, Model, Rope};

enum LayerState {
    Gdn { tail: Vec<f32>, s: Vec<f32> },
    Attn { k: Vec<f32>, v: Vec<f32> },
}

pub struct State {
    pub pos: usize,
    layers: Vec<LayerState>,
    rope: Rope,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `y = W x` for one row, adapter included (`[n]`).
fn lin1(l: &Lin, x: &[f32]) -> Vec<f32> {
    l.fwd(x, 1).0
}

impl State {
    pub fn new(m: &Model, max_len: usize) -> Self {
        let c = &m.c;
        let layers = m
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Gdn(_) => LayerState::Gdn {
                    tail: vec![0.0; (c.conv_kernel - 1) * c.conv_dim()],
                    s: vec![0.0; c.lin_v_heads * c.lin_k_dim * c.lin_v_dim],
                },
                Mixer::Attn(_) => LayerState::Attn {
                    k: Vec::new(),
                    v: Vec::new(),
                },
            })
            .collect();
        Self {
            pos: 0,
            layers,
            rope: Rope::new(max_len, c.rotary_dim, c.rope_theta),
        }
    }
}

fn gdn_step(m: &Model, d: &Gdn, xn: &[f32], tail: &mut [f32], s: &mut [f32]) -> Vec<f32> {
    let c = &m.c;
    let (cd, kern) = (c.conv_dim(), c.conv_kernel);
    let (kh, vh, kd, vd) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
    let key_dim = kh * kd;
    let qkv = lin1(&d.qkv, xn);
    let z = lin1(&d.z, xn);
    let a = lin1(&d.a, xn);
    let b = lin1(&d.b, xn);
    let hist = kern - 1;
    let mut mixed = vec![0f32; cd];
    for ch in 0..cd {
        let taps = &d.conv[ch * kern..(ch + 1) * kern];
        let acc: f32 = (0..kern)
            .map(|j| taps[j] * if j < hist { tail[j * cd + ch] } else { qkv[ch] })
            .sum();
        mixed[ch] = silu(acc);
    }
    tail.copy_within(cd.., 0);
    tail[(hist - 1) * cd..].copy_from_slice(&qkv);
    let q_scale = 1.0 / (kd as f32).sqrt();
    let mut q = vec![0f32; key_dim];
    let mut k = vec![0f32; key_dim];
    for hh in 0..kh {
        let (qs, ks) = (
            &mixed[hh * kd..(hh + 1) * kd],
            &mixed[key_dim + hh * kd..key_dim + (hh + 1) * kd],
        );
        let qi = 1.0 / (qs.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        let ki = 1.0 / (ks.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        for e in 0..kd {
            q[hh * kd + e] = qs[e] * qi * q_scale;
            k[hh * kd + e] = ks[e] * ki;
        }
    }
    let ratio = vh / kh;
    let mut core = vec![0f32; vh * vd];
    s.par_chunks_mut(kd * vd)
        .zip(core.par_chunks_mut(vd))
        .enumerate()
        .for_each(|(hv, (s, o))| {
            let hk = hv / ratio;
            let (kt, qt) = (&k[hk * kd..(hk + 1) * kd], &q[hk * kd..(hk + 1) * kd]);
            let vt = &mixed[2 * key_dim + hv * vd..2 * key_dim + (hv + 1) * vd];
            let g = -d.a_log[hv].exp() * softplus(a[hv] + d.dt_bias[hv]);
            let beta = sigmoid(b[hv]);
            let decay = g.exp();
            s.iter_mut().for_each(|v| *v *= decay);
            let mut kv = vec![0f32; vd];
            for (r, &kr) in kt.iter().enumerate() {
                for (o, &sv) in kv.iter_mut().zip(&s[r * vd..(r + 1) * vd]) {
                    *o += sv * kr;
                }
            }
            let delta: Vec<f32> = vt.iter().zip(&kv).map(|(&v, &m)| (v - m) * beta).collect();
            for (r, &kr) in kt.iter().enumerate() {
                for (sv, &dd) in s[r * vd..(r + 1) * vd].iter_mut().zip(&delta) {
                    *sv += kr * dd;
                }
            }
            for (r, &qr) in qt.iter().enumerate() {
                for (ov, &sv) in o.iter_mut().zip(&s[r * vd..(r + 1) * vd]) {
                    *ov += sv * qr;
                }
            }
        });
    for (o, zr) in core.chunks_exact_mut(vd).zip(z.chunks_exact(vd)) {
        let ms = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
        let inv = 1.0 / (ms + c.eps).sqrt();
        for ((v, &g), &zz) in o.iter_mut().zip(&d.norm).zip(zr) {
            *v = *v * inv * g * silu(zz);
        }
    }
    lin1(&d.out, &core)
}

fn attn_step(
    m: &Model,
    a: &Attn,
    rope: &Rope,
    pos: usize,
    xn: &[f32],
    kc: &mut Vec<f32>,
    vc: &mut Vec<f32>,
) -> Vec<f32> {
    let c = &m.c;
    let (nh, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
    let kvd = kvh * hd;
    let group = nh / kvh;
    let qg = lin1(&a.q, xn);
    let kraw = lin1(&a.k, xn);
    let v = lin1(&a.v, xn);
    let mut qraw = vec![0f32; nh * hd];
    let mut gate = vec![0f32; nh * hd];
    for hh in 0..nh {
        qraw[hh * hd..(hh + 1) * hd].copy_from_slice(&qg[hh * 2 * hd..hh * 2 * hd + hd]);
        gate[hh * hd..(hh + 1) * hd].copy_from_slice(&qg[hh * 2 * hd + hd..(hh + 1) * 2 * hd]);
    }
    let mut q = rms_zc(&qraw, &a.q_norm, c.eps);
    let mut k = rms_zc(&kraw, &a.k_norm, c.eps);
    for hh in 0..nh {
        rope_at(rope, &mut q[hh * hd..(hh + 1) * hd], pos);
    }
    for hh in 0..kvh {
        rope_at(rope, &mut k[hh * hd..(hh + 1) * hd], pos);
    }
    kc.extend_from_slice(&k);
    vc.extend_from_slice(&v);
    let span = pos + 1;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; nh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(hh, o)| {
        let kh = hh / group;
        let qv = &q[hh * hd..(hh + 1) * hd];
        let mut p: Vec<f32> = (0..span)
            .map(|j| {
                qv.iter()
                    .zip(&kc[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for x in p.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        p.iter_mut().for_each(|x| *x /= sum);
        for (j, &w) in p.iter().enumerate() {
            for (ov, &vv) in o
                .iter_mut()
                .zip(&vc[j * kvd + kh * hd..j * kvd + (kh + 1) * hd])
            {
                *ov += w * vv;
            }
        }
        for (ov, &g) in o.iter_mut().zip(&gate[hh * hd..(hh + 1) * hd]) {
            *ov *= sigmoid(g);
        }
    });
    lin1(&a.o, &out)
}

fn rope_at(rope: &Rope, x: &mut [f32], pos: usize) {
    let h = rope.half;
    for j in 0..h {
        let (c, s) = (rope.cos[pos * h + j], rope.sin[pos * h + j]);
        let (x1, x2) = (x[j], x[j + h]);
        x[j] = x1 * c - x2 * s;
        x[j + h] = x2 * c + x1 * s;
    }
}

impl Model {
    /// Runs one token at `state.pos` and returns its logits (`[vocab]`).
    pub fn step(&self, state: &mut State, token: u32) -> Vec<f32> {
        let (h, eps) = (self.c.hidden, self.c.eps);
        let tok = token as usize;
        assert!(tok < self.c.vocab, "token {tok} outside the vocab");
        let mut x = self.embed[tok * h..(tok + 1) * h].to_vec();
        let pos = state.pos;
        for (l, ls) in self.layers.iter().zip(state.layers.iter_mut()) {
            let xn = rms_zc(&x, &l.input_norm, eps);
            let mixed = match (&l.mixer, ls) {
                (Mixer::Gdn(d), LayerState::Gdn { tail, s }) => gdn_step(self, d, &xn, tail, s),
                (Mixer::Attn(a), LayerState::Attn { k, v }) => {
                    attn_step(self, a, &state.rope, pos, &xn, k, v)
                }
                _ => unreachable!("state kind differs from the layer"),
            };
            x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
            let xn2 = rms_zc(&x, &l.post_norm, eps);
            let g = lin1(&l.mlp.gate, &xn2);
            let u = lin1(&l.mlp.up, &xn2);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
            let down = lin1(&l.mlp.down, &act);
            x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
        }
        state.pos += 1;
        let hn = rms_zc(&x, &self.final_norm, eps);
        let mut logits = vec![0f32; self.c.vocab];
        logits
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(c, out)| {
                for (r, o) in out.iter_mut().enumerate() {
                    let v = c * 1024 + r;
                    *o = self.embed[v * h..(v + 1) * h]
                        .iter()
                        .zip(&hn)
                        .map(|(a, b)| a * b)
                        .sum();
                }
            });
        logits
    }

    /// Greedy continuation of `prompt` until a stop id or `max_new` tokens.
    pub fn generate(&self, prompt: &[u32], max_new: usize, stop: &[u32]) -> Vec<u32> {
        let mut state = State::new(self, prompt.len() + max_new + 1);
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.step(&mut state, t);
        }
        let mut out = Vec::new();
        for _ in 0..max_new {
            let next = aien_inference_abi::qwen38::argmax(&logits) as u32;
            out.push(next);
            if stop.contains(&next) {
                break;
            }
            logits = self.step(&mut state, next);
        }
        out
    }
}
