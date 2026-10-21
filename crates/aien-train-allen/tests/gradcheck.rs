//! Finite-difference check of the hand-written backward on a tiny random model, with two
//! mutant controls that must fail it (L389, step 5).

use std::sync::atomic::Ordering;

use aien_inference_abi::qwen38::Qwen38Config;
use aien_train_allen::model::{Model, SplitMix, MUTANT};

fn tiny() -> Qwen38Config {
    Qwen38Config {
        hidden: 32,
        layers: 4,
        vocab: 40,
        intermediate: 48,
        full_attention: vec![false, true, false, true],
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
        rotary_dim: 4,
        rope_theta: 1e4,
        lin_k_heads: 2,
        lin_v_heads: 4,
        lin_k_dim: 8,
        lin_v_dim: 8,
        conv_kernel: 4,
        eps: 1e-6,
        eos_token_ids: vec![1],
        moe: None,
        tied_embeddings: true,
    }
}

/// Per adapter tensor: (name, analytic directional derivative, central difference).
fn check(model: &mut Model, tokens: &[u32], weights: &[f32]) -> Vec<(String, f64, f64)> {
    let _ = model.train_seq(tokens, weights);
    let mut rng = SplitMix(7);
    let n = model.params_mut().len();
    let mut out = Vec::new();
    let eps: f32 = std::env::var("AIEN_FD_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-2);
    for idx in 0..n {
        let (name, dir, an) = {
            let mut ps = model.params_mut();
            let (name, p) = &mut ps[idx];
            let mut dir: Vec<f32> = (0..p.v.len()).map(|_| rng.uniform()).collect();
            let norm = dir.iter().map(|v| v * v).sum::<f32>().sqrt();
            dir.iter_mut().for_each(|v| *v /= norm);
            let an: f64 =
                p.g.iter()
                    .zip(&dir)
                    .map(|(g, d)| *g as f64 * *d as f64)
                    .sum();
            (name.clone(), dir, an)
        };
        let at = |sign: f32, model: &mut Model| {
            {
                let mut ps = model.params_mut();
                let p = &mut ps[idx].1;
                p.v.iter_mut()
                    .zip(&dir)
                    .for_each(|(v, d)| *v += sign * eps * d);
            }
            let l = model.loss(tokens, weights).loss;
            let mut ps = model.params_mut();
            let p = &mut ps[idx].1;
            p.v.iter_mut()
                .zip(&dir)
                .for_each(|(v, d)| *v -= sign * eps * d);
            l
        };
        let fd = (at(1.0, model) - at(-1.0, model)) / (2.0 * eps as f64);
        out.push((name, an, fd));
    }
    out
}

fn passes(an: f64, fd: f64) -> bool {
    (an - fd).abs() <= 2e-3 + 2e-2 * an.abs().max(fd.abs())
}

#[test]
fn backward_matches_finite_differences_and_mutants_fail() {
    let c = tiny();
    let tokens: Vec<u32> = (0..14u32).map(|i| (i * 17 + 3) % 40).collect();
    let weights: Vec<f32> = (0..14)
        .map(|i| {
            if i < 3 {
                0.0
            } else if i < 6 {
                0.25
            } else {
                1.0
            }
        })
        .collect();

    MUTANT.store(0, Ordering::Relaxed);
    let mut model = Model::random(&c, 1);
    model.add_lora(4, 8.0, 2, 0.3);
    let base = model.loss(&tokens, &weights);
    let rows = check(&mut model, &tokens, &weights);
    let mut bad = Vec::new();
    for (name, an, fd) in &rows {
        eprintln!("{name:44} analytic {an:+.6e} fd {fd:+.6e}");
        if !passes(*an, *fd) {
            bad.push(name.clone());
        }
        assert!(an.abs() > 1e-4, "{name}: gradient too small to test ({an})");
    }
    assert!(base.weight > 0.0 && base.loss.is_finite());
    assert!(bad.is_empty(), "failed: {bad:?}");
    assert_eq!(
        rows.len(),
        2 * (2 * 3 + 2 * 4 + 4 * 3),
        "every adapter tensor checked"
    );

    for m in [1u8, 2] {
        MUTANT.store(m, Ordering::Relaxed);
        let mut model = Model::random(&c, 1);
        model.add_lora(4, 8.0, 2, 0.3);
        let failed = check(&mut model, &tokens, &weights)
            .iter()
            .filter(|(_, an, fd)| !passes(*an, *fd))
            .count();
        eprintln!("mutant {m}: {failed} tensors fail");
        assert!(
            failed > 0,
            "mutant {m} passed the check: the check is blind"
        );
    }
    MUTANT.store(0, Ordering::Relaxed);
}

#[test]
fn step_by_step_decode_equals_the_whole_sequence_forward() {
    let c = tiny();
    let mut model = Model::random(&c, 3);
    model.add_lora(4, 8.0, 5, 0.3);
    let tokens: Vec<u32> = (0..11u32).map(|i| (i * 7 + 1) % 40).collect();
    let whole = model.logits(&tokens);
    let mut st = aien_train_allen::decode::State::new(&model, 16);
    for (i, &t) in tokens.iter().enumerate() {
        let row = model.step(&mut st, t);
        for (a, b) in row.iter().zip(&whole[i * c.vocab..(i + 1) * c.vocab]) {
            assert!((a - b).abs() < 1e-4, "position {i}: {a} vs {b}");
        }
    }
}
