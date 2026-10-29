//! Host tests of the dense engine (no chip): the maths against the independent f32 reference
//! (`TransformerWeights::forward_token`), prefill equal to decode bit for bit, the FP8 path equal
//! to the bf16 path on the twin model, the negative controls, and the generation contract.

use super::*;
use crate::qwen3_dense_omega::test_support::{
    memory_source_of, tensors, tensors_with, tiny_config, tiny_fp8_config, write_checkpoint,
    TinyWeights,
};
use crate::qwen3_dense_omega::{load_qwen3_dense_omega_from, DenseWeightFormat};
use crate::qwen3_moe::parity_stats;
use crate::qwen3_moe_omega::HostKv;
use crate::tensor::sample_argmax;
use crate::weights::{LayerKvCache, SequenceState, TransformerWeights};

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("aien-own8b-eng-{name}-{}", std::process::id()))
}

/// The engine over the tiny model on the host, with the bf16 host pool (the chip's KV rounding)
/// or the f32 host cache.
fn engine_of(
    c: &Qwen3DenseConfig,
    mode: TinyWeights,
    uploader: &dyn DenseUploader,
    f32_kv: bool,
) -> Qwen3DenseOmegaEngine {
    let (w, _) = load_qwen3_dense_omega_from(
        c,
        &mut memory_source_of(tensors_with(c, mode)),
        uploader,
        &mut |_| {},
    )
    .unwrap();
    let g = KvGeometry::from(c);
    let kv: Box<dyn KvAttention> = if f32_kv {
        Box::new(HostKv::new(&g, 256))
    } else {
        Box::new(HostPoolKv::new(&g, 256).unwrap())
    };
    Qwen3DenseOmegaEngine::new(w, kv).unwrap()
}

fn host_engine(c: &Qwen3DenseConfig) -> Qwen3DenseOmegaEngine {
    engine_of(c, TinyWeights::Random, &HostDenseUploader, false)
}

const PROMPT: [u32; 7] = [5, 17, 2, 40, 63, 9, 9];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Logits at every prompt position, token at a time.
fn logits_per_position(e: &mut Qwen3DenseOmegaEngine, tokens: &[u32]) -> Vec<Vec<f32>> {
    e.reset();
    tokens
        .iter()
        .map(|&t| {
            let h = e.step(t).unwrap();
            e.logits(&h).unwrap()
        })
        .collect()
}

/// The engine (f32 KV) against `TransformerWeights::forward_token` on the same checkpoint, for
/// untied and tied lm_heads: the same maths written independently (fused vs separate
/// projections, rope, q/k norms, GQA, SwiGLU, final norm, lm_head or embedding).
#[test]
fn engine_matches_the_independent_f32_reference() {
    for tied in [false, true] {
        let c = Qwen3DenseConfig {
            tie_word_embeddings: tied,
            ..tiny_config()
        };
        let dir = scratch(if tied { "ref-tied" } else { "ref" });
        let _ = std::fs::remove_dir_all(&dir);
        write_checkpoint(&dir, &c).unwrap();
        let mc = crate::model_dir::load_model_config(&dir).unwrap();
        let tw = TransformerWeights::load_from_safetensors(
            dir.join("model.safetensors.index.json"),
            &mc,
        )
        .unwrap();
        assert_eq!(tw.lm_head.is_none(), tied);
        let (w, _) =
            crate::qwen3_dense_omega::load_qwen3_dense_omega(&dir, &HostDenseUploader, &mut |_| {})
                .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let mut e = Qwen3DenseOmegaEngine::new(w, Box::new(HostKv::new(&KvGeometry::from(&c), 64)))
            .unwrap();
        let ours = logits_per_position(&mut e, &PROMPT);
        let mut st = SequenceState {
            tokens: Vec::new(),
            layers: vec![LayerKvCache::default(); c.layers],
            kv_dtype: aien_kv_cache::KvDType::Fp32,
        };
        for (pos, &t) in PROMPT.iter().enumerate() {
            st.tokens.push(t);
            let h = tw.forward_token(t, pos, &mut st);
            let theirs = tw.compute_logits(&h);
            let s = parity_stats(&ours[pos], &theirs);
            assert!(
                s.cosine > 0.999_999 && s.max_abs < 1e-4 * s.max_abs_expected.max(1.0),
                "tied={tied} pos {pos}: {s:?}"
            );
            assert_eq!(
                sample_argmax(&ours[pos]).0,
                sample_argmax(&theirs).0,
                "tied={tied} pos {pos}"
            );
        }
    }
}

/// Prefill over all rows equals decode once per token bit for bit (hidden, logits, KV, traces),
/// and a chunked prefill equals both.
#[test]
fn step_batch_equals_step_bit_for_bit() {
    for c in [tiny_config(), tiny_fp8_config()] {
        let mut a = host_engine(&c);
        a.trace_layers = vec![0, 1];
        let mut ha = Vec::new();
        for &t in &PROMPT {
            ha = a.step(t).unwrap();
        }
        let mut b = host_engine(&c);
        b.trace_layers = vec![0, 1];
        let hb = b.step_batch(&PROMPT).unwrap();
        assert_eq!(bits(&ha), bits(&hb), "{:?}", c.format);
        assert_eq!(b.position(), PROMPT.len());
        assert_eq!(b.history(), &PROMPT);
        let key = |t: &DenseLayerTrace| (t.layer, t.pos);
        let (mut ta, mut tb) = (a.traces.clone(), b.traces.clone());
        ta.sort_by_key(key);
        tb.sort_by_key(key);
        assert_eq!(ta, tb);
        let mut c3 = host_engine(&c);
        c3.step_batch(&PROMPT[..3]).unwrap();
        let hc = c3.step_batch(&PROMPT[3..]).unwrap();
        assert_eq!(bits(&ha), bits(&hc));
        // The next decode step sees the same KV.
        let (na, nb) = (a.step(11).unwrap(), b.step(11).unwrap());
        assert_eq!(bits(&na), bits(&nb));
        let ka = a.export_kv(0, 8).unwrap();
        assert_eq!(ka, b.export_kv(0, 8).unwrap());
    }
}

/// The twin model (every projection value e4m3-exact, scales 1.0): the FP8 engine equals the bf16
/// engine bit for bit, so the FP8 path's layout is the bf16 path's layout end to end.
#[test]
fn fp8_twin_equals_bf16_engine_bit_for_bit() {
    let mut f = engine_of(
        &tiny_fp8_config(),
        TinyWeights::Twin,
        &HostDenseUploader,
        false,
    );
    let mut b = engine_of(&tiny_config(), TinyWeights::Twin, &HostDenseUploader, false);
    let lf = logits_per_position(&mut f, &PROMPT);
    let lb = logits_per_position(&mut b, &PROMPT);
    for (x, y) in lf.iter().zip(&lb) {
        assert_eq!(bits(x), bits(y));
    }
}

/// The smallest cosine, over the traced layer 1 at every prompt position, of the attention
/// output and the MLP output of `bad` against `good` (what a mutant corrupts first; the tiny
/// model's residual stream, embedding scale 1.0, dilutes it in the final hidden state).
fn worst_layer_cosine(good: &mut Qwen3DenseOmegaEngine, bad: &mut Qwen3DenseOmegaEngine) -> f64 {
    good.trace_layers = vec![1];
    bad.trace_layers = vec![1];
    good.step_batch(&PROMPT).unwrap();
    bad.step_batch(&PROMPT).unwrap();
    assert_eq!(good.traces.len(), PROMPT.len());
    let mut worst = 1.0f64;
    for (g, b) in good.traces.iter().zip(&bad.traces) {
        assert_eq!((g.layer, g.pos), (b.layer, b.pos));
        worst = worst
            .min(parity_stats(&b.attn, &g.attn).cosine)
            .min(parity_stats(&b.mlp_out, &g.mlp_out).cosine);
    }
    worst
}

/// Every negative control breaks a layer: the k|v row swap (bf16 and FP8), the scale-grid
/// transpose (FP8) and the attention mutant; the transpose is refused on a bf16 checkpoint.
#[test]
fn mutants_break_the_forward_pass() {
    for c in [tiny_config(), tiny_fp8_config()] {
        let swap = MutantUploader::new(&HostDenseUploader, DenseUploadMutant::SwapKv, &c);
        let mut bad = engine_of(&c, TinyWeights::Random, &swap, false);
        let cos = worst_layer_cosine(&mut host_engine(&c), &mut bad);
        assert!(cos < 0.99, "{:?} swap-kv cosine {cos}", c.format);

        let mut bad = host_engine(&c);
        bad.attention_mutant = true;
        let cos = worst_layer_cosine(&mut host_engine(&c), &mut bad);
        assert!(cos < 0.99, "{:?} attention mutant cosine {cos}", c.format);

        // The control of the control: two clean engines agree exactly.
        let (mut a, mut b) = (host_engine(&c), host_engine(&c));
        worst_layer_cosine(&mut a, &mut b);
        assert_eq!(a.traces, b.traces);
    }
    let c = tiny_fp8_config();
    let tr = MutantUploader::new(&HostDenseUploader, DenseUploadMutant::TransposeScales, &c);
    let mut bad = engine_of(&c, TinyWeights::Random, &tr, false);
    let cos = worst_layer_cosine(&mut host_engine(&c), &mut bad);
    assert!(cos < 0.99, "transpose-scales cosine {cos}");

    let c = tiny_config();
    let tr = MutantUploader::new(&HostDenseUploader, DenseUploadMutant::TransposeScales, &c);
    let e = load_qwen3_dense_omega_from(&c, &mut memory_source_of(tensors(&c)), &tr, &mut |_| {})
        .unwrap_err();
    assert!(e.contains("FP8"), "{e}");
}
#[test]
fn kv_geometry_mismatch_is_refused() {
    let c = tiny_config();
    let (w, _) = load_qwen3_dense_omega_from(
        &c,
        &mut memory_source_of(tensors(&c)),
        &HostDenseUploader,
        &mut |_| {},
    )
    .unwrap();
    let wrong = KvGeometry {
        kv_heads: 4,
        ..KvGeometry::from(&c)
    };
    let e = Qwen3DenseOmegaEngine::new(w, Box::new(HostKv::new(&wrong, 16)))
        .err()
        .unwrap();
    assert!(e.contains("geometry"), "{e}");
}

/// Greedy generation equals the argmax chain of `step`/`logits`; stop ids are consumed; the
/// length cap and cancel hold; a second prompt sharing a prefix reuses it and answers like a
/// cold engine; the same seed replays and the token stream is deterministic.
#[test]
fn generate_contract() {
    let c = tiny_config();
    let mut e = host_engine(&c);
    let mut out = Vec::new();
    let o = e
        .generate(&PROMPT, &GenerateParams::greedy(6, vec![]), &mut |t| {
            out.push(t);
            true
        })
        .unwrap();
    assert_eq!(o.emitted_tokens, 6);
    assert_eq!(o.finish, FinishReason::Length);
    // The manual argmax chain.
    let mut m = host_engine(&c);
    let mut h = m.step_batch(&PROMPT).unwrap();
    let mut chain = Vec::new();
    for _ in 0..6 {
        let t = sample_argmax(&m.logits(&h).unwrap()).0;
        chain.push(t);
        h = m.step(t).unwrap();
    }
    assert_eq!(out, chain);

    // Stop on the third token: two emitted, finish Stop.
    let mut got = Vec::new();
    let o = e
        .generate(
            &PROMPT,
            &GenerateParams::greedy(6, vec![chain[2]]),
            &mut |t| {
                got.push(t);
                true
            },
        )
        .unwrap();
    assert_eq!(o.finish, FinishReason::Stop);
    assert_eq!(got, chain[..2]);
    // Cancel after one.
    let o = e
        .generate(&PROMPT, &GenerateParams::greedy(6, vec![]), &mut |_| false)
        .unwrap();
    assert_eq!((o.emitted_tokens, o.finish), (1, FinishReason::Cancelled));

    // Prefix reuse: the engine holds PROMPT + 1 token; a longer prompt reuses PROMPT.
    let mut longer = PROMPT.to_vec();
    longer.extend([30, 31]);
    let mut warm_out = Vec::new();
    let o = e
        .generate(&longer, &GenerateParams::greedy(4, vec![]), &mut |t| {
            warm_out.push(t);
            true
        })
        .unwrap();
    assert_eq!(o.reused_prompt_tokens, PROMPT.len());
    let mut cold = host_engine(&c);
    let mut cold_out = Vec::new();
    let o = cold
        .generate(&longer, &GenerateParams::greedy(4, vec![]), &mut |t| {
            cold_out.push(t);
            true
        })
        .unwrap();
    assert_eq!(o.reused_prompt_tokens, 0);
    assert_eq!(warm_out, cold_out);

    // Seeded sampling replays.
    let sampled = |e: &mut Qwen3DenseOmegaEngine, seed: u64| {
        let p = GenerateParams {
            max_new_tokens: 8,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed,
            stop_token_ids: vec![],
        };
        let mut v = Vec::new();
        e.generate(&PROMPT, &p, &mut |t| {
            v.push(t);
            true
        })
        .unwrap();
        v
    };
    let a = sampled(&mut e, 7);
    assert_eq!(a, sampled(&mut cold, 7));
}

/// Rewind forgets positions; export then restore into a fresh engine continues identically.
#[test]
fn rewind_and_kv_restore() {
    let c = tiny_fp8_config();
    let mut e = host_engine(&c);
    e.step_batch(&PROMPT).unwrap();
    let after = e.step(12).unwrap();
    e.rewind(PROMPT.len());
    assert_eq!(e.position(), PROMPT.len());
    assert_eq!(bits(&e.step(12).unwrap()), bits(&after));

    let payload = e.export_kv(0, PROMPT.len()).unwrap();
    let mut f = host_engine(&c);
    f.restore_kv(&PROMPT, &payload).unwrap();
    assert_eq!(f.history(), &PROMPT);
    assert_eq!(bits(&f.step(12).unwrap()), bits(&after));
    assert!(f.restore_kv(&[], &payload).is_err());
    assert_eq!(f.position(), 0);
    assert!(e.export_kv(0, 99).is_err());
}

#[test]
fn prefill_chunk_does_not_change_the_answer() {
    let c = tiny_config();
    let run = |chunk: usize| {
        let mut e = host_engine(&c);
        e.prefill_chunk = chunk;
        let mut v = Vec::new();
        e.generate(&PROMPT, &GenerateParams::greedy(5, vec![]), &mut |t| {
            v.push(t);
            true
        })
        .unwrap();
        v
    };
    let one = run(1);
    assert_eq!(one, run(3));
    assert_eq!(one, run(DENSE_PREFILL_CHUNK));
    assert_eq!(c.format, DenseWeightFormat::Bf16);
}
