// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The checkpoint's multi-token-prediction (MTP) layer on the host reference (DRA-70, sc#866).
//!
//! Qwen3.6-35B-A3B ships one MTP layer (`mtp_num_hidden_layers` 1) under `mtp.*`
//! (`mtp.safetensors`). Ground truth for the maths is vLLM `qwen3_next_mtp.py`
//! (`Qwen3NextMultiTokenPredictor.forward`):
//!
//! ```text
//! e = pre_fc_norm_embedding(embed_tokens(tok[i + 1]))     zero-centred RMSNorm
//! g = pre_fc_norm_hidden(h[i])                            h = the target's post-final-norm row
//! x = fc(cat[e, g])                                       bf16 [hidden][2 * hidden], no bias
//! x = decoder layer mtp.layers.0 (full attention + MoE, residual starts at x), position i
//! m = mtp.norm(x)                                         the draft row; lm_head(m) drafts tok[i + 2]
//! ```
//!
//! The layer has no own embedding or head (`embed_tokens` and `lm_head` are the target's). A chain
//! of k drafts feeds `m` back as the next `h` with the drafted token, one position further. The MTP
//! keeps its own attention cache ([`MtpState`]), filled teacher-forced over the prompt and the
//! accepted tokens. Its MoE routes are recorded at index `config.layers` (one past the target's).

use super::*;

/// Tensor prefix of the MTP decoder layer.
pub const MTP_LAYER: &str = "mtp.layers.0";

/// The MTP block as read: the fc projection, its two input norms, the output norm and one
/// full-attention MoE decoder layer.
pub struct MtpWeights {
    fc: Matrix,
    pre_embed: Vec<f32>,
    pre_hidden: Vec<f32>,
    norm: Vec<f32>,
    layer: LayerWeights,
}

impl MtpWeights {
    /// Bytes the layer streams per drafted token when every routed expert is distinct:
    /// fc + attention + router + shared expert + `top_k` experts (the head is counted apart).
    pub fn draft_bytes(&self, top_k: usize, expert_bytes: usize) -> usize {
        let m = |x: &Matrix| match x {
            Matrix::Fp8 { w, scales, .. } => w.len() + 2 * scales.len(),
            Matrix::Bf16 { w, .. } => 2 * w.len(),
        };
        let mut b = m(&self.fc);
        if let Mixer::Attention(a) = &self.layer.mixer {
            b += m(&a.q) + m(&a.k) + m(&a.v) + m(&a.o);
        }
        if let Mlp::Moe(w) = &self.layer.mlp {
            b += m(&w.router) + m(&w.shared.gate) + m(&w.shared.up) + m(&w.shared.down);
        }
        b + top_k * expert_bytes
    }
}

/// The MTP layer's attention cache between calls.
#[derive(Debug, Clone, Default)]
pub struct MtpState {
    /// Next position (the MTP row for target position i sits at position i).
    pub pos: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl MtpState {
    /// Drops every cached position at or past `pos` (rollback after a rejected draft).
    pub fn truncate(&mut self, pos: usize, kv_dim: usize) {
        if pos < self.pos {
            self.k.truncate(pos * kv_dim);
            self.v.truncate(pos * kv_dim);
            self.pos = pos;
        }
    }
}

impl<S: TensorSource> Qwen38Reference<S> {
    /// Reads the MTP block (`mtp.*`). The 35B layer is a full-attention MoE layer.
    pub fn read_mtp(&mut self) -> Result<MtpWeights, String> {
        let h = self.config.hidden;
        if self.config.moe.is_none() {
            return Err("mtp: the config is dense (the 35B MTP layer is MoE)".into());
        }
        let src: &mut dyn TensorSource = &mut self.source;
        let fc = read_matrix(src, "mtp.fc.weight", h, 2 * h)?;
        let pre_embed = read_vec(src, "mtp.pre_fc_norm_embedding.weight", h)?;
        let pre_hidden = read_vec(src, "mtp.pre_fc_norm_hidden.weight", h)?;
        let norm = read_vec(src, "mtp.norm.weight", h)?;
        let layer = self.read_layer_at(MTP_LAYER, true)?;
        Ok(MtpWeights {
            fc,
            pre_embed,
            pre_hidden,
            norm,
            layer,
        })
    }

    /// Embedding rows of `tokens`, `[t][hidden]`.
    pub fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
        let mut x = Vec::with_capacity(tokens.len() * h);
        for &tok in tokens {
            let tok = tok as usize;
            if tok >= v {
                return Err(format!("embed: token {tok} outside the vocab {v}"));
            }
            x.extend(self.embed[tok * h..(tok + 1) * h].iter().map(|&b| bf16_to_f32(b)));
        }
        Ok(x)
    }

    /// The MTP block over `t` rows: row i pairs the target's post-final-norm `hidden[i]` with
    /// `tokens[i]` (the token after it) at position `state.pos + i`. Returns the rows after
    /// `mtp.norm`, `[t][hidden]` (multiply by the head for the draft logits).
    pub fn mtp_forward(
        &mut self,
        w: &MtpWeights,
        state: &mut MtpState,
        hidden: &[f32],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let c = self.config.clone();
        let (t, h) = (tokens.len(), c.hidden);
        if t == 0 || hidden.len() != t * h {
            return Err(format!(
                "mtp: {t} tokens with {} hidden values (want {})",
                hidden.len(),
                t * h
            ));
        }
        let mut e = self.embed_rows(tokens)?;
        rms_norm_zc(&mut e, &w.pre_embed, c.eps);
        let mut g = hidden.to_vec();
        rms_norm_zc(&mut g, &w.pre_hidden, c.eps);
        let mut cat = Vec::with_capacity(2 * t * h);
        for (er, gr) in e.chunks_exact(h).zip(g.chunks_exact(h)) {
            cat.extend_from_slice(er);
            cat.extend_from_slice(gr);
        }
        let mut x = w.fc.matmul(&cat, t);
        let mut xn = x.clone();
        rms_norm_zc(&mut xn, &w.layer.input_norm, c.eps);
        let Mixer::Attention(a) = &w.layer.mixer else {
            return Err("mtp: the layer is not full attention".into());
        };
        let mixed = attention(&c, a, &xn, t, state.pos, &mut state.k, &mut state.v);
        x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
        let mut xn = x.clone();
        rms_norm_zc(&mut xn, &w.layer.post_norm, c.eps);
        let Mlp::Moe(m) = &w.layer.mlp else {
            return Err("mtp: the layer is not MoE".into());
        };
        let down = self.moe(c.layers, &format!("{MTP_LAYER}.mlp.experts"), m, &xn, t)?;
        x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
        state.pos += t;
        rms_norm_zc(&mut x, &w.norm, c.eps);
        Ok(x)
    }

    /// The lm_head rows `ids` (ascending, unique) as their own bf16 matrix: the truncated draft
    /// head. Logit j of it is the full head's logit `ids[j]`, bit for bit.
    pub fn head_rows(&self, ids: &[u32]) -> Result<Matrix, String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
        if ids.windows(2).any(|p| p[0] >= p[1]) || ids.last().is_some_and(|&l| l as usize >= v) {
            return Err(format!("head_rows: ids not ascending within 0..{v}"));
        }
        let mut w = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            let row = self
                .lm_head
                .bf16_row(id as usize)
                .ok_or("head_rows: the lm_head is not bf16")?;
            w.extend_from_slice(row);
        }
        Ok(Matrix::Bf16 {
            n: ids.len(),
            k: h,
            w,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{tiny_moe, tiny_source, Mem};
    use super::*;
    use crate::qwen3_moe::f32_to_bf16;

    /// A one-layer full-attention MoE target whose MTP layer is a copy of layer 0, `mtp.norm` a
    /// copy of the final norm, and `fc` the given `[hidden][2 * hidden]` matrix.
    fn mirrored(fc: impl Fn(usize, usize) -> f32) -> (Qwen38Config, Mem) {
        let c = Qwen38Config {
            layers: 1,
            full_attention: vec![true],
            ..tiny_moe()
        };
        let mut m = tiny_source(&c);
        let h = c.hidden;
        let p = "model.language_model.";
        // Unit-RMS embedding rows, so the MTP's pre-fc norm (w = 0) leaves them as the target
        // embeds them (to bf16 rounding).
        let (d, s, b) = m.tensors[&format!("{p}embed_tokens.weight")].clone();
        let mut rows: Vec<f32> = b
            .chunks_exact(2)
            .map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))
            .collect();
        for r in rows.chunks_exact_mut(h) {
            let rms = (r.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
            r.iter_mut().for_each(|v| *v /= rms);
        }
        let bytes = rows
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        m.tensors
            .insert(format!("{p}embed_tokens.weight"), (d, s, bytes));
        let names: Vec<String> = m
            .tensors
            .keys()
            .filter(|k| k.starts_with(&format!("{p}layers.0.")))
            .cloned()
            .collect();
        for k in names {
            let t = m.tensors[&k].clone();
            m.tensors
                .insert(k.replace(&format!("{p}layers.0"), MTP_LAYER), t);
        }
        let norm = m.tensors[&format!("{p}norm.weight")].clone();
        m.tensors.insert("mtp.norm.weight".into(), norm);
        let zeros = |n: usize| ("BF16".to_string(), vec![n], vec![0u8; 2 * n]);
        m.tensors
            .insert("mtp.pre_fc_norm_embedding.weight".into(), zeros(h));
        m.tensors
            .insert("mtp.pre_fc_norm_hidden.weight".into(), zeros(h));
        let fcw = (0..h * 2 * h)
            .flat_map(|i| f32_to_bf16(fc(i / (2 * h), i % (2 * h))).to_le_bytes())
            .collect();
        m.tensors
            .insert("mtp.fc.weight".into(), ("BF16".into(), vec![h, 2 * h], fcw));
        (c, m)
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// With fc = [I | 0] the MTP block is the one-layer target fed the (normed) embedding, so its
    /// head logits match the target's: proves the concat order (embedding first), the layer
    /// wiring, the positions and `mtp.norm`.
    #[test]
    fn identity_fc_reproduces_the_target() {
        let (c, m) = mirrored(|r, col| if r == col { 1.0 } else { 0.0 });
        let h = c.hidden;
        let mut r = Qwen38Reference::new(c.clone(), m).unwrap();
        let toks = [3u32, 17, 5, 60, 9];
        let mut st = r.new_state();
        let want = r.forward(&mut st, &toks, Logits::All).unwrap();
        let w = r.read_mtp().unwrap();
        let mut ms = MtpState::default();
        let junk: Vec<f32> = (0..toks.len() * h).map(|i| (i % 7) as f32 - 3.0).collect();
        let rows = r.mtp_forward(&w, &mut ms, &junk, &toks).unwrap();
        let got = r.lm_head().matmul(&rows, toks.len());
        let e = max_rel(&got, &want);
        assert!(e < 2e-2, "mtp vs target logits: max rel {e}");
        for (g, t) in got.chunks_exact(c.vocab).zip(want.chunks_exact(c.vocab)) {
            assert_eq!(argmax(g), argmax(t));
        }
        assert_eq!(ms.pos, toks.len());
        assert_eq!(r.routes.len(), c.layers + 1, "MTP routes recorded past the target's");
    }

    /// Negative control: fc = [0 | I] reads the hidden half instead; the logits must move.
    #[test]
    fn swapped_concat_does_not_reproduce_the_target() {
        let (c, m) = mirrored(|r, col| if col == r + 256 { 1.0 } else { 0.0 });
        let h = c.hidden;
        let mut r = Qwen38Reference::new(c.clone(), m).unwrap();
        let toks = [3u32, 17, 5, 60, 9];
        let mut st = r.new_state();
        let want = r.forward(&mut st, &toks, Logits::All).unwrap();
        let w = r.read_mtp().unwrap();
        let junk: Vec<f32> = (0..toks.len() * h).map(|i| (i % 7) as f32 - 3.0).collect();
        let rows = r
            .mtp_forward(&w, &mut MtpState::default(), &junk, &toks)
            .unwrap();
        let got = r.lm_head().matmul(&rows, toks.len());
        assert!(max_rel(&got, &want) > 0.1);
    }

    /// One call over t rows equals t calls of one row (the attention cache carries positions),
    /// and truncate + replay restores the same rows (draft rollback).
    #[test]
    fn rows_one_call_equals_row_by_row_and_rollback_replays() {
        let (c, m) = mirrored(|r, col| ((r * 31 + col * 7) % 13) as f32 / 64.0 - 0.09);
        let h = c.hidden;
        let mut r = Qwen38Reference::new(c.clone(), m).unwrap();
        let w = r.read_mtp().unwrap();
        let toks = [1u32, 2, 40, 33];
        let hid: Vec<f32> = (0..toks.len() * h)
            .map(|i| ((i * 13) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let all = r
            .mtp_forward(&w, &mut MtpState::default(), &hid, &toks)
            .unwrap();
        let mut st = MtpState::default();
        let mut one = Vec::new();
        for i in 0..toks.len() {
            one.extend(
                r.mtp_forward(&w, &mut st, &hid[i * h..(i + 1) * h], &toks[i..i + 1])
                    .unwrap(),
            );
        }
        assert!(max_rel(&one, &all) < 1e-5);
        st.truncate(2, c.kv_dim());
        let again = r
            .mtp_forward(&w, &mut st, &hid[2 * h..], &toks[2..])
            .unwrap();
        assert_eq!(&again[..], &one[2 * h..], "replay after rollback is bit-identical");
    }

    #[test]
    fn head_rows_are_the_full_head_logits() {
        let (c, m) = mirrored(|r, col| if r == col { 1.0 } else { 0.0 });
        let r = Qwen38Reference::new(c.clone(), m).unwrap();
        let ids = [0u32, 5, 9, 63];
        let sub = r.head_rows(&ids).unwrap();
        let x: Vec<f32> = (0..c.hidden).map(|i| (i % 5) as f32 - 2.0).collect();
        let full = r.lm_head().matmul(&x, 1);
        let part = sub.matmul(&x, 1);
        for (j, &id) in ids.iter().enumerate() {
            assert_eq!(part[j].to_bits(), full[id as usize].to_bits());
        }
        assert!(r.head_rows(&[5, 5]).is_err());
        assert!(r.head_rows(&[64]).is_err());
    }

    #[test]
    fn refuses_a_misshapen_fc() {
        let (c, mut m) = mirrored(|_, _| 0.0);
        let h = c.hidden;
        m.tensors.insert(
            "mtp.fc.weight".into(),
            ("BF16".into(), vec![h, h], vec![0u8; 2 * h * h]),
        );
        let mut r = Qwen38Reference::new(c, m).unwrap();
        assert!(r.read_mtp().is_err());
    }
}
