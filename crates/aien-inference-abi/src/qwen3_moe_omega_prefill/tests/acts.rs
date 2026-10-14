//! W-L1H: the W8A16 activation mode ([`super::super::acts`]) on the fake device. New file.
//! The fake's numerics are exact in both modes (the chip's bf16 rounding is tested in omega's gpu_pfa16_test),
//! so these tests check the call sequence, the buffers and that W8A16 computes the same function as W8A8.
use super::*;
use crate::qwen3_moe_omega_prefill::PrefillActs;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn setup_a16(bm: usize, dev_attn: bool) -> (TestModel, WholePromptPrefill<FakeDevice>, FakeAttn, Log) {
    let (tm, mut drv, attn, log) = setup(bm, dev_attn);
    drv.dev.a16 = true;
    drv.set_acts(PrefillActs::W8A16).unwrap();
    (tm, drv, attn, log)
}

#[test]
fn parse_and_default() {
    assert_eq!(PrefillActs::default(), PrefillActs::W8A8);
    assert_eq!(PrefillActs::parse("a16").unwrap(), PrefillActs::W8A16);
    assert_eq!(PrefillActs::parse("w8a8").unwrap(), PrefillActs::W8A8);
    assert!(PrefillActs::parse("a4").is_err());
    let (_, drv, _, _) = setup(4, false);
    assert_eq!(drv.acts(), PrefillActs::W8A8, "the default stays W8A8");
}

#[test]
fn refused_without_the_kernels() {
    let (_, mut drv, _, _) = setup(4, false);
    assert!(drv.set_acts(PrefillActs::W8A16).is_err());
    assert_eq!(drv.acts(), PrefillActs::W8A8);
    drv.set_acts(PrefillActs::W8A8).unwrap();
}

#[test]
fn call_order_per_layer() {
    let (tm, mut drv, mut attn, log) = setup_a16(4, false);
    let c = tm.weights.config.clone();
    let m = 5;
    drv.run(&tm.weights, &mut attn, &TOKENS[..m], 0).unwrap();
    let (h, qd, i, g) = (c.hidden, c.q_dim(), c.moe_intermediate, m * c.top_k);
    let log = log.borrow();
    assert_eq!(log.len(), 9 * c.layers, "{log:?}");
    for (l, ev) in log.chunks(9).enumerate() {
        let conv = |k| Ev::Hconv { k, input: QuantIn::F32, rows: m };
        let dense = |n, k| Ev::Hgemm { n, k, ntiles: m.div_ceil(4) };
        assert_eq!(ev[0], conv(h), "layer {l}: convert normed x");
        assert_eq!(ev[1], dense(c.qkv_n(), h), "layer {l}: q|k|v");
        assert_eq!(ev[2], Ev::Attn { layer: l as u32, pos0: 0, m: m as u32, dev: false });
        assert_eq!(ev[3], conv(qd), "layer {l}: convert attention out");
        assert_eq!(ev[4], dense(h, qd), "layer {l}: o");
        assert_eq!(ev[5], conv(h), "layer {l}: convert post");
        assert!(matches!(ev[6], Ev::Hgemm { n, k, .. } if n == 2 * i && k == h), "layer {l}: gate|up {:?}", ev[6]);
        assert_eq!(ev[7], Ev::Hswiglu { i, rows: g }, "layer {l}: SwiGLU straight into the down input");
        assert!(matches!(ev[8], Ev::Hgemm { n, k, .. } if n == h && k == i), "layer {l}: down {:?}", ev[8]);
        let (Ev::Hgemm { ntiles: gu, .. }, Ev::Hgemm { ntiles: dn, .. }) = (&ev[6], &ev[8]) else { unreachable!() };
        assert_eq!(gu, dn);
    }
    assert!(!log.iter().any(|e| matches!(e, Ev::Quant { .. } | Ev::Gemm { .. })), "no W8A8 kernel in W8A16 mode");
}

#[test]
fn same_function_as_w8a8_and_the_reference() {
    for dev_attn in [false, true] {
        let (tm, mut a8, mut at8, _) = setup(4, dev_attn);
        let want = a8.run(&tm.weights, &mut at8, &TOKENS, 0).unwrap();
        let (tm2, mut a16, mut at16, _) = setup_a16(4, dev_attn);
        let got = a16.run(&tm2.weights, &mut at16, &TOKENS, 0).unwrap();
        assert_eq!(bits(&got), bits(&want), "dev_attn {dev_attn}: W8A16 vs W8A8 on exact fakes");
        assert!(max_rel_diff(&got, &reference(&tm2, &TOKENS)) <= TOL);
    }
}

#[test]
fn prefix_then_continuation_and_switching_back() {
    let (tm, mut drv, mut attn, log) = setup_a16(4, false);
    drv.run(&tm.weights, &mut attn, &TOKENS[..3], 0).unwrap();
    let got = drv.run(&tm.weights, &mut attn, &TOKENS[3..], 3).unwrap();
    assert!(max_rel_diff(&got, &reference(&tm, &TOKENS)) <= TOL);
    // back to W8A8 on the same driver: buffers re-sized, W8A8 kernels again, same numbers
    drv.set_acts(PrefillActs::W8A8).unwrap();
    log.borrow_mut().clear();
    let c = tm.weights.config.clone();
    let mut fresh = FakeAttn { core: AttnCore::new(&c), log: log.clone(), dev: None, qkv_n: c.qkv_n(), qd: c.q_dim() };
    let a8 = drv.run(&tm.weights, &mut fresh, &TOKENS, 0).unwrap();
    assert!(log.borrow().iter().any(|e| matches!(e, Ev::Quant { .. })) && !log.borrow().iter().any(|e| matches!(e, Ev::Hconv { .. })));
    assert!(max_rel_diff(&a8, &reference(&tm, &TOKENS)) <= TOL);
}

#[test]
fn tile_size_does_not_change_the_result() {
    let outs: Vec<Vec<u32>> = [1usize, 2, 4, 16]
        .iter()
        .map(|&bm| {
            let (tm, mut drv, mut attn, _) = setup_a16(bm, false);
            bits(&drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap())
        })
        .collect();
    assert!(outs.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn negative_control_ignoring_the_row_index_is_caught() {
    let (tm, mut drv, mut attn, _) = setup_a16(4, false);
    drv.dev.mutant_ignore_rowidx = true;
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    let d = max_rel_diff(&got, &reference(&tm, &TOKENS));
    assert!(d > 1e-2, "W8A16 gate|up that ignores the token gather must disagree, diff {d}");
}
