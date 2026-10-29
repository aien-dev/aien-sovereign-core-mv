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

//! Teacher-forced logit comparison shared by the strict gate and the drift diagnostic.
//!
//! Both backends are fed the reference's own greedy tokens, so every step compares
//! logits for the same context. A greedy text comparison cannot tell a wrong backend
//! from a near-tie: TinyLlama on the gate prompt has a step where "processing" leads
//! "memory" by 0.0005 logits while the f32 backends differ by ~0.03 everywhere
//! (evidence-out/FB1-CUT3D-02dfa2e-drift, 2026-10-04).
#![allow(dead_code)]
use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::SequenceState;

/// Largest allowed |logit difference| at any step. The cut 3d run measured 0.059 worst
/// over 32 steps (32000 logits each); the bound leaves 2.5x headroom.
pub const MAX_ABS_DLOGIT: f32 = 0.15;

pub fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            (i2, v2) = (i1, v1);
            (i1, v1) = (i, v);
        } else if v > v2 {
            (i2, v2) = (i, v);
        }
    }
    (i1, v1, i2, v2)
}

/// Logits for every step: step 0 from prefill, step s from decoding `forced[s-1]`.
/// With `forced` empty the model's own argmax is fed back; the fed tokens are returned.
pub fn logits_per_step(
    t: &mut NativeTransformerBackend,
    prompt: &[u32],
    steps: usize,
    forced: &[u32],
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let seq_id = 0x10_9175_u64;
    let mut out = vec![t.prefill_sequence(seq_id, prompt).expect("prefill")];
    let mut toks = Vec::new();
    for s in 0..steps {
        let tok = if forced.is_empty() {
            top2(&out[s]).0 as u32
        } else {
            forced[s]
        };
        toks.push(tok);
        if s + 1 == steps {
            break;
        }
        let seq: &mut SequenceState = t.sequences.get_mut(&seq_id).expect("sequence");
        seq.tokens.push(tok);
        let pos = seq.tokens.len() - 1;
        let hidden = NativeTransformerBackend::try_forward_token_impl_paged(
            &t.weights,
            &*t.tensor_backend,
            tok,
            pos,
            seq,
            seq_id,
            t.kv_manager.as_ref(),
        )
        .expect("decode");
        out.push(NativeTransformerBackend::compute_logits_impl(
            &t.weights,
            &*t.tensor_backend,
            &hidden,
        ));
    }
    t.release_sequence(seq_id);
    (out, toks)
}

pub struct StepCmp {
    pub step: usize,
    pub ref_tok: usize,
    pub ref_margin: f32,
    pub cand_tok: usize,
    pub max_abs_dlogit: f32,
}

pub fn compare(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> Vec<StepCmp> {
    reference
        .iter()
        .zip(candidate)
        .enumerate()
        .map(|(step, (r, c))| {
            let (r1, rv1, _, rv2) = top2(r);
            let d = r
                .iter()
                .zip(c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |m, x| {
                    if m.is_nan() || x.is_nan() {
                        f32::NAN
                    } else {
                        m.max(x)
                    }
                }); // a NaN logit must fail the judge, f32::max would drop it
            StepCmp {
                step,
                ref_tok: r1,
                ref_margin: rv1 - rv2,
                cand_tok: top2(c).0,
                max_abs_dlogit: d,
            }
        })
        .collect()
}

/// Pass when every step stays within `MAX_ABS_DLOGIT` and every argmax disagreement is a
/// near-tie the measured difference can explain: a flip needs the top-two gap to move by
/// at least the reference margin, and that gap moves by at most 2 * max_abs_dlogit.
pub fn judge(steps: &[StepCmp]) -> Result<usize, String> {
    let mut flips = 0;
    for s in steps {
        if s.max_abs_dlogit.is_nan() || s.max_abs_dlogit > MAX_ABS_DLOGIT {
            return Err(format!(
                "step {}: max |dlogit| {} exceeds {MAX_ABS_DLOGIT}",
                s.step, s.max_abs_dlogit
            ));
        }
        if s.cand_tok != s.ref_tok {
            if s.ref_margin > 2.0 * s.max_abs_dlogit {
                return Err(format!(
                    "step {}: argmax {} vs reference {} with reference margin {} > 2 * max |dlogit| {}",
                    s.step, s.cand_tok, s.ref_tok, s.ref_margin, s.max_abs_dlogit
                ));
            }
            flips += 1;
        }
    }
    Ok(flips)
}
