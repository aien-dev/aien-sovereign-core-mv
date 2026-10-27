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

use super::*;
use crate::bitnet_omega::test_support::{memory_source, tiny_config};
use crate::bitnet_omega::{load_bitnet_omega_from, pack_trits, HostTernMatrix};
use crate::qwen3_moe::f32_to_bf16;
use crate::qwen3_moe_omega::test_support::prand;

fn tiny() -> BitNetOmegaWeights {
    let c = tiny_config();
    load_bitnet_omega_from(&c, &mut memory_source(&c), &HostRefUploader, &mut |_| {})
        .unwrap()
        .0
}

fn tokens() -> Vec<u32> {
    vec![1, 17, 200, 45, 45, 3, 99, 128, 7]
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn batched_ternary_product_equals_the_packed_gemv() {
    let (n, k, t) = (70, 96, 5);
    let trits: Vec<i8> = (0..n * k)
        .map(|i| (prand(3, i) * 3.0).floor().clamp(-1.0, 1.0) as i8)
        .collect();
    let mut words = vec![0u32; n * k / TRITS_PER_WORD];
    pack_trits(&trits, k, &mut words).unwrap();
    let seg = [TernSegment {
        part: TernPart::O,
        rows: n,
        scale_bf16: f32_to_bf16(1.0),
    }];
    let packed = HostTernMatrix::new(n, k, words.clone(), seg.to_vec()).unwrap();
    let ours = I8TernMatrix::from_words(n, k, &words, &seg).unwrap();
    let x: Vec<i8> = (0..t * k)
        .map(|i| (prand(5, i) * 256.0).clamp(-128.0, 127.0) as i8)
        .collect();
    let mut acc = vec![0i32; t * n];
    ours.gemm(&x, t, &mut acc).unwrap();
    for i in 0..t {
        let mut want = vec![0i32; n];
        packed.gemv_i8(&x[i * k..(i + 1) * k], &mut want).unwrap();
        assert_eq!(acc[i * n..(i + 1) * n], want, "token {i}");
    }
    assert!(ours.gemm(&x, t + 1, &mut acc).is_err());
}

#[test]
fn activation_quant_rounds_half_to_even_per_token() {
    let mut q = [0i8; 6];
    let s = quant_act(&[127.0, 0.5, 1.5, -0.5, 2.5, -127.0], &mut q);
    assert_eq!(s, 1.0);
    assert_eq!(q, [127, 0, 2, 0, 2, -127]);
    let s = quant_act(&[0.0; 6], &mut q);
    assert_eq!((s, q), (127.0 / 1e-5, [0; 6]));
}

#[test]
fn rope_is_half_split_and_starts_at_position_zero() {
    let mut v = vec![0f32; 128];
    v[0] = 1.0;
    let at0 = v.clone();
    rope_half_split(&mut v, &rope_table(0, 128, 500000.0));
    assert_eq!(v, at0);
    rope_half_split(&mut v, &rope_table(1, 128, 500000.0));
    // Dim 0 pairs with dim 64 at frequency 1: (cos 1, sin 1); nothing else moves.
    assert!((v[0] - 1f32.cos()).abs() < 1e-7 && (v[64] - 1f32.sin()).abs() < 1e-7);
    assert!(v
        .iter()
        .enumerate()
        .all(|(i, &x)| i == 0 || i == 64 || x == 0.0));
    // The last pair turns at 500000^(-126/128) radians per position.
    let t = rope_table(1000, 128, 500000.0);
    let want = 1000.0 * 500000f64.powf(-126.0 / 128.0);
    assert!((t[63].1 as f64 - want.sin()).abs() < 1e-7);
}

#[test]
fn rms_norm_is_weighted_x_over_root_mean_square() {
    let mut out = [0f32; 4];
    rms_norm(&[3.0, -4.0, 0.0, 0.0], &[1.0, 2.0, 1.0, 1.0], 0.0, &mut out);
    // mean square 25/4, root 2.5.
    assert_eq!(out, [1.2, -3.2, 0.0, 0.0]);
}

#[test]
fn kv_cache_decode_equals_the_whole_prefill_bit_for_bit() {
    let w = tiny();
    let f = HostBitNetForward::new(&w);
    let toks = tokens();
    let mut whole = KvCache::new(&w.config);
    let all = f.forward(&toks, &mut whole, &[], &mut Vec::new()).unwrap();
    let mut step = KvCache::new(&w.config);
    let mut rows = f
        .forward(&toks[..4], &mut step, &[], &mut Vec::new())
        .unwrap();
    for &t in &toks[4..] {
        rows.extend(f.forward(&[t], &mut step, &[], &mut Vec::new()).unwrap());
    }
    assert_eq!(step.len, toks.len());
    assert_eq!(all, rows);
}

#[test]
fn taps_are_the_residual_stream_after_each_listed_layer() {
    let w = tiny();
    let f = HostBitNetForward::new(&w);
    let mut taps = Vec::new();
    let fin = f
        .forward(&tokens(), &mut KvCache::new(&w.config), &[0, 1], &mut taps)
        .unwrap();
    assert_eq!(taps.iter().map(|t| t.0).collect::<Vec<_>>(), [0, 1]);
    let h = w.config.hidden;
    assert!(taps.iter().all(|t| t.1.len() == tokens().len() * h));
    let mut normed = vec![0f32; fin.len()];
    for (o, x) in normed.chunks_mut(h).zip(taps[1].1.chunks(h)) {
        rms_norm(x, &w.final_norm, w.config.rms_eps, o);
    }
    assert_eq!(normed, fin);
}

#[test]
fn variants_change_the_output_except_the_vacuous_q_k_swap() {
    let w = tiny();
    let run = |v: Variant| {
        let mut f = HostBitNetForward::new(&w);
        f.variant = v;
        f.forward(
            &tokens(),
            &mut KvCache::new(&w.config),
            &[],
            &mut Vec::new(),
        )
        .unwrap()
    };
    let r = run(Variant::Reference);
    for v in [
        Variant::SubNormAfterO,
        Variant::SkipSubNorms,
        Variant::SwapQvScales,
    ] {
        assert!(max_rel(&r, &run(v)) > 1e-2, "{} left the output", v.name());
    }
    assert!(max_rel(&r, &run(Variant::SwapQkScales)) < 1e-4);
}

#[test]
fn nll_is_log_softmax_of_the_tied_logits() {
    let w = tiny();
    let f = HostBitNetForward::new(&w);
    let toks = tokens();
    let fin = f
        .forward(&toks, &mut KvCache::new(&w.config), &[], &mut Vec::new())
        .unwrap();
    let h = w.config.hidden;
    let targets = &toks[1..];
    let nll = f.nll(&fin[..targets.len() * h], targets).unwrap();
    let lg = f.logits(&fin);
    let vocab = w.config.vocab;
    for (i, &tg) in targets.iter().enumerate() {
        let row = &lg[i * vocab..(i + 1) * vocab];
        let e = &w.embed_row(tg).unwrap();
        let direct: f64 = e
            .iter()
            .zip(&fin[i * h..(i + 1) * h])
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        assert!((row[tg as usize] as f64 - direct).abs() < 1e-4);
        let want = log_sum_exp(row) - row[tg as usize] as f64;
        assert_eq!(nll[i], want);
        assert!(nll[i] > 0.0);
    }
    assert!(f.nll(&fin, targets).is_err());
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
}

#[test]
fn attention_equals_a_naive_masked_loop() {
    // tiny_config: 4 query heads over 2 kv heads of 64, so heads 0,1 read kv 0 and 2,3 read kv 1.
    let c = tiny_config();
    let (qd, kvd, hd) = (c.q_dim(), c.kv_dim(), c.head_dim);
    let (pos0, t) = (3, 4);
    let q: Vec<f32> = (0..t * qd).map(|i| prand(11, i) * 4.0 - 2.0).collect();
    let k: Vec<f32> = (0..(pos0 + t) * kvd)
        .map(|i| prand(12, i) * 4.0 - 2.0)
        .collect();
    let v: Vec<f32> = (0..(pos0 + t) * kvd)
        .map(|i| prand(13, i) * 4.0 - 2.0)
        .collect();
    let got = attend(&c, &q, &k, &v, pos0);
    for i in 0..t {
        for head in 0..c.q_heads {
            let kvh = [0, 0, 1, 1][head];
            let mut w = vec![0f64; pos0 + t];
            for (j, wj) in w.iter_mut().enumerate() {
                if j > pos0 + i {
                    *wj = f64::NEG_INFINITY; // causal mask
                    continue;
                }
                let s: f64 = (0..hd)
                    .map(|d| q[i * qd + head * hd + d] as f64 * k[j * kvd + kvh * hd + d] as f64)
                    .sum();
                *wj = s / 8.0; // sqrt(64)
            }
            let m = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = w.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..hd {
                let want: f64 = (0..pos0 + t)
                    .map(|j| (w[j] - m).exp() / z * v[j * kvd + kvh * hd + d] as f64)
                    .sum();
                let g = got[i * qd + head * hd + d] as f64;
                assert!(
                    (g - want).abs() < 1e-5,
                    "token {i} head {head} dim {d}: {g} vs {want}"
                );
            }
        }
    }
}

#[test]
fn bit_linear_scales_each_segment_by_its_own_scale() {
    let w = tiny();
    let f = HostBitNetForward::new(&w);
    let m = w.layers[0].qkv.as_ref();
    let (n, k) = (m.n(), m.k());
    let x: Vec<f32> = (0..2 * k).map(|i| prand(21, i) * 2.0 - 1.0).collect();
    let y = f.bit_linear(m, &x, 2).unwrap();
    let mut row_scale = Vec::new();
    for s in m.segments() {
        row_scale.extend(std::iter::repeat_n(bf16_to_f32(s.scale_bf16), s.rows));
    }
    let scales: Vec<f32> = m
        .segments()
        .iter()
        .map(|s| bf16_to_f32(s.scale_bf16))
        .collect();
    assert!(
        scales.windows(2).any(|p| p[0] != p[1]),
        "tiny q/k/v scales all equal"
    );
    for i in 0..2 {
        let mut q = vec![0i8; k];
        let s_a = quant_act(&x[i * k..(i + 1) * k], &mut q);
        let mut acc = vec![0i32; n];
        m.gemv_i8(&q, &mut acc).unwrap();
        for r in 0..n {
            assert_eq!(
                y[i * n + r],
                acc[r] as f32 / s_a * row_scale[r],
                "token {i} row {r}"
            );
        }
    }
}

#[test]
fn a_failed_forward_leaves_the_cache_where_it_was() {
    let w = tiny();
    let f = HostBitNetForward::new(&w);
    let mut cache = KvCache::new(&w.config);
    f.forward(&tokens()[..4], &mut cache, &[], &mut Vec::new())
        .unwrap();
    cache.truncate(3, w.config.kv_dim());
    let before = cache.clone();
    // Token 999 is outside the 256-token vocabulary.
    assert!(f
        .forward(&[5, 999], &mut cache, &[], &mut Vec::new())
        .is_err());
    assert_eq!(
        (cache.len, &cache.k, &cache.v),
        (before.len, &before.k, &before.v)
    );
    let mut whole = KvCache::new(&w.config);
    let all = f
        .forward(&tokens()[..4], &mut whole, &[], &mut Vec::new())
        .unwrap();
    let last = f
        .forward(&tokens()[3..4], &mut cache, &[], &mut Vec::new())
        .unwrap();
    assert_eq!(all[3 * w.config.hidden..], last[..]);
    assert!(f.logits(&[]).is_empty());
}
