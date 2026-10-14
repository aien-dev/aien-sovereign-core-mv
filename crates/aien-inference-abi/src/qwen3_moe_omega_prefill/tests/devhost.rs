//! W-L1G-D: the device router logits + combine path ([`super::super::devhost`]) on a fake device.
//! New file. [`DhDev`] wraps the driver tests' [`FakeDevice`] and adds the three device entries,
//! computed by `aien_omega_pf`'s reference loops (the loops the chip kernels are tested against).
use super::*;
use crate::moe_plan::{MoeBatchPlan, MoePlanError};
use crate::qwen3_moe_omega_prefill::devhost::plan_par;

struct DhDev {
    inner: FakeDevice,
    /// Rows per router launch block; `None` = the device has no router / combine (stub build).
    t: Option<usize>,
    /// Negative control: the combine drops each token's last slot.
    mutant_combine_drop_last: bool,
}

fn read_u32(reg: &Reg, va: u64, n: usize, what: &str) -> Vec<u32> {
    check_range(reg, va, n * 4, what);
    // SAFETY: range checked against a live allocation.
    unsafe { std::slice::from_raw_parts(va as *const u32, n) }.to_vec()
}

impl PrefillDevice for DhDev {
    type Buf = FakeBuf;
    fn alloc_dev(&mut self, bytes: usize) -> Result<FakeBuf, String> {
        self.inner.alloc_dev(bytes)
    }
    fn alloc_host(&mut self, bytes: usize) -> Result<FakeBuf, String> {
        self.inner.alloc_host(bytes)
    }
    fn qgemm_bm(&self) -> usize {
        self.inner.qgemm_bm()
    }
    fn quant(&mut self, k: usize, input: QuantIn, rows: usize, x_va: u64, q_va: u64, s_va: u64) -> Result<u64, String> {
        self.inner.quant(k, input, rows, x_va, q_va, s_va)
    }
    fn qgemm(&mut self, c: &QgemmCall) -> Result<u64, String> {
        self.inner.qgemm(c)
    }
    fn copy_va(&mut self, dst_va: u64, src_va: u64, bytes: usize) -> Result<(), String> {
        let v = read_f32(&self.inner.reg, src_va, bytes / 4, "copy src");
        write_f32(&self.inner.reg, dst_va, &v, "copy dst");
        Ok(())
    }
    fn router_rows(&self) -> Option<usize> {
        self.t
    }
    fn router_logits(&mut self, k: usize, e: usize, rows: usize, x_va: u64, wq_va: u64, logits_va: u64) -> Result<u64, String> {
        let t = self.t.ok_or("no router")?;
        self.inner.log.borrow_mut().push(Ev::Router { k, e, rows });
        // The chip contract: whole blocks of t rows are read and written.
        let rp = rows.div_ceil(t) * t;
        let x = read_f32(&self.inner.reg, x_va, rp * k, "router x (padded rows)");
        let wq = read_f32(&self.inner.reg, wq_va, aien_omega_pf::router_packed_len(e, k), "packed router");
        let mut out = vec![0.0f32; rp * e];
        aien_omega_pf::router_logits_ref(k, e, rp, &x, &wq, &mut out);
        write_f32(&self.inner.reg, logits_va, &out, "router logits (padded rows)");
        Ok(5)
    }
    fn moe_combine(&mut self, h: usize, top_k: usize, rows: usize, down_va: u64, inv_va: u64, wt_va: u64, out_va: u64) -> Result<u64, String> {
        self.t.ok_or("no combine")?;
        self.inner.log.borrow_mut().push(Ev::Combine { h, top_k, rows });
        let g = rows * top_k;
        let down = read_f32(&self.inner.reg, down_va, g * h, "combine down rows");
        let inv = read_u32(&self.inner.reg, inv_va, g, "combine inv");
        let mut wt = read_f32(&self.inner.reg, wt_va, g, "combine weights");
        if self.mutant_combine_drop_last {
            wt.chunks_mut(top_k).for_each(|w| w[top_k - 1] = 0.0);
        }
        let mut out = vec![0.0f32; rows * h];
        aien_omega_pf::moe_combine_ref(h, top_k, rows, &down, &inv, &wt, &mut out);
        write_f32(&self.inner.reg, out_va, &out, "combine out");
        Ok(5)
    }
}

fn setup_dh(bm: usize, dev_attn: bool, t: Option<usize>) -> (TestModel, WholePromptPrefill<DhDev>, FakeAttn, Log) {
    let c = config();
    let tm = model(&c, 7);
    let log: Log = Rc::new(RefCell::new(Vec::new()));
    let fd = FakeDevice::new(bm, &tm, log.clone());
    let reg = fd.reg.clone();
    let drv = WholePromptPrefill::new(DhDev { inner: fd, t, mutant_combine_drop_last: false }, &tm.weights).unwrap();
    let attn = FakeAttn { core: AttnCore::new(&c), log: log.clone(), dev: dev_attn.then_some(reg), qkv_n: c.qkv_n(), qd: c.q_dim() };
    (tm, drv, attn, log)
}

/// Runs the whole prompt (as a prefix of 3 then the rest when `split`) with a full trace.
fn traced(drv: &mut WholePromptPrefill<DhDev>, tm: &TestModel, attn: &mut FakeAttn, split: bool) -> (Vec<f32>, WpTrace) {
    drv.trace = Some(WpTrace::new(Some(1)));
    let out = if split {
        drv.run(&tm.weights, attn, &TOKENS[..3], 0).unwrap();
        drv.run(&tm.weights, attn, &TOKENS[3..], 3).unwrap()
    } else {
        drv.run(&tm.weights, attn, &TOKENS, 0).unwrap()
    };
    (out, drv.trace.take().unwrap())
}

#[test]
fn device_router_and_combine_equal_the_host_path_bit_for_bit() {
    for (bm, dev_attn, t, split) in [(4, false, 3, false), (4, true, 16, false), (1, false, 1, true), (64, true, 2, true), (3, false, 7, false)] {
        let (tm, mut host_drv, mut a1, _) = setup_dh(bm, dev_attn, Some(t));
        let (want, want_tr) = traced(&mut host_drv, &tm, &mut a1, split);
        let want_whole = {
            let (tm4, mut d4, mut a4, _) = setup_dh(bm, false, Some(t));
            d4.run(&tm4.weights, &mut a4, &TOKENS, 0).unwrap()
        };
        let (tm2, mut dev_drv, mut a2, _) = setup_dh(bm, dev_attn, Some(t));
        assert!(!dev_drv.device_host_work(), "OFF by default");
        dev_drv.set_device_host_work(&tm2.weights, true).unwrap();
        assert!(dev_drv.device_host_work());
        let (got, got_tr) = traced(&mut dev_drv, &tm2, &mut a2, split);
        let case = format!("bm {bm} dev_attn {dev_attn} t {t} split {split}");
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{case}: final hidden");
        assert_eq!(got_tr.experts, want_tr.experts, "{case}: routing");
        assert_eq!(got_tr, want_tr, "{case}: every layer's residual stream and the ops of layer 1");
        assert!(max_rel_diff(&got, &reference(&tm, &TOKENS)) <= TOL, "{case}: vs the token-by-token reference");
        // and back OFF: the host path again (fresh attention cache, whole prompt)
        dev_drv.set_device_host_work(&tm2.weights, false).unwrap();
        assert!(!dev_drv.device_host_work());
        let c = config();
        let mut fresh = FakeAttn { core: AttnCore::new(&c), log: Rc::new(RefCell::new(Vec::new())), dev: None, qkv_n: c.qkv_n(), qd: c.q_dim() };
        let off = dev_drv.run(&tm2.weights, &mut fresh, &TOKENS, 0).unwrap();
        assert_eq!(off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want_whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{case}: OFF again");
    }
}

#[test]
fn device_path_call_order_per_layer() {
    let (tm, mut drv, mut attn, log) = setup_dh(4, false, Some(3));
    drv.set_device_host_work(&tm.weights, true).unwrap();
    let c = tm.weights.config.clone();
    let m = 5;
    drv.run(&tm.weights, &mut attn, &TOKENS[..m], 0).unwrap();
    let (h, qd, i, g, e, top_k) = (c.hidden, c.q_dim(), c.moe_intermediate, m * c.top_k, c.experts, c.top_k);
    let log = log.borrow();
    assert_eq!(log.len(), 11 * c.layers, "{log:?}");
    for (l, ev) in log.chunks(11).enumerate() {
        let f32q = |k| Ev::Quant { k, input: QuantIn::F32, rows: m };
        assert_eq!(ev[0], f32q(h), "layer {l}");
        assert!(matches!(ev[1], Ev::Gemm { layout: Layout::Plain, gather: false, .. }), "layer {l}: q|k|v");
        assert!(matches!(ev[2], Ev::Attn { .. }), "layer {l}");
        assert_eq!(ev[3], f32q(qd), "layer {l}");
        assert!(matches!(ev[4], Ev::Gemm { layout: Layout::Plain, .. }), "layer {l}: o");
        assert_eq!(ev[5], Ev::Router { k: h, e, rows: m }, "layer {l}: ONE router launch for all tokens");
        assert_eq!(ev[6], f32q(h), "layer {l}: quantise post (the same buffer the router read)");
        assert!(matches!(ev[7], Ev::Gemm { layout: Layout::SwigluNat, gather: true, .. }), "layer {l}: gate|up");
        assert_eq!(ev[8], Ev::Quant { k: i, input: QuantIn::Bf16, rows: g }, "layer {l}");
        assert!(matches!(ev[9], Ev::Gemm { n, k, layout: Layout::Plain, gather: false, .. } if n == h && k == i), "layer {l}: down");
        assert_eq!(ev[10], Ev::Combine { h, top_k, rows: m }, "layer {l}: ONE combine launch");
    }
    let s = &drv.stats;
    assert_eq!((s.router, s.combine, s.launches()), (2, 2, 18));
}

#[test]
fn routing_mutant_is_still_caught_on_the_device_path() {
    let (tm, mut drv, mut attn, _) = setup_dh(4, false, Some(3));
    drv.set_device_host_work(&tm.weights, true).unwrap();
    let (_, good) = traced(&mut drv, &tm, &mut attn, false);
    let (tm2, mut bad, mut attn2, _) = setup_dh(4, false, Some(3));
    bad.set_device_host_work(&tm2.weights, true).unwrap();
    bad.mutant = WpMutant::Routing;
    let (out, bad_tr) = traced(&mut bad, &tm2, &mut attn2, false);
    assert_ne!(bad_tr.experts, good.experts, "rotated logits must change the routing");
    assert!(max_rel_diff(&out, &reference(&tm, &TOKENS)) > TOL);
}

#[test]
fn combine_negative_control_is_caught() {
    let (tm, mut drv, mut attn, _) = setup_dh(4, false, Some(3));
    drv.set_device_host_work(&tm.weights, true).unwrap();
    let (tm2, mut host_drv, mut attn2, _) = setup_dh(4, false, Some(3));
    let want = host_drv.run(&tm2.weights, &mut attn2, &TOKENS, 0).unwrap();
    // a driver whose fake combine drops each token's last slot
    let c = config();
    let log: Log = Rc::new(RefCell::new(Vec::new()));
    let fd = FakeDevice::new(4, &tm, log.clone());
    let mut mutant = WholePromptPrefill::new(DhDev { inner: fd, t: Some(3), mutant_combine_drop_last: true }, &tm.weights).unwrap();
    mutant.set_device_host_work(&tm.weights, true).unwrap();
    let mut attn3 = FakeAttn { core: AttnCore::new(&c), log, dev: None, qkv_n: c.qkv_n(), qd: c.q_dim() };
    let bad = mutant.run(&tm.weights, &mut attn3, &TOKENS, 0).unwrap();
    assert_ne!(bad, want, "dropping a slot in the combine must change the output");
    let good = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    assert_eq!(good, want);
}

#[test]
fn without_device_kernels_the_switch_refuses_and_the_host_path_runs() {
    let (tm, mut drv, mut attn, log) = setup_dh(4, false, None);
    let err = drv.set_device_host_work(&tm.weights, true).unwrap_err();
    assert!(err.contains("not linked"), "{err}");
    assert!(!drv.device_host_work());
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    assert!(max_rel_diff(&got, &reference(&tm, &TOKENS)) <= TOL);
    assert!(!log.borrow().iter().any(|e| matches!(e, Ev::Router { .. } | Ev::Combine { .. })));
}

fn plans_equal(a: &MoeBatchPlan, b: &MoeBatchPlan) -> bool {
    a.tokens == b.tokens
        && a.experts == b.experts
        && a.top_k == b.top_k
        && a.topk_experts == b.topk_experts
        && a.topk_weights.iter().map(|v| v.to_bits()).eq(b.topk_weights.iter().map(|v| v.to_bits()))
        && a.expert_offsets == b.expert_offsets
        && a.grouped_to_assignment == b.grouped_to_assignment
        && a.assignment_to_grouped == b.assignment_to_grouped
        && a.grouped_token_ids == b.grouped_token_ids
}

#[test]
fn parallel_plan_equals_from_logits_exactly() {
    let mut r = Lcg(11);
    for (tokens, experts, top_k) in [(1, 1, 1), (7, 4, 2), (33, 128, 8), (5, 16, 16), (300, 128, 8), (9, 10, 3), (4, 64, 12)] {
        let mut logits = r.vec(tokens * experts, 3.0);
        // ties: whole equal rows, pairs of equal values, +0.0 vs -0.0
        if tokens > 2 {
            logits[..experts].fill(0.25);
            logits[experts] = logits[experts + experts - 1];
            logits[2 * experts] = 0.0;
            if experts > 1 {
                logits[2 * experts + 1] = -0.0;
            }
        }
        // coarse values: many ties at every rank
        if tokens > 3 {
            for v in &mut logits[3 * experts..4 * experts] {
                *v = (*v * 2.0).round();
            }
        }
        let want = MoeBatchPlan::from_logits(&logits, tokens, experts, top_k).unwrap();
        let got = plan_par(&logits, tokens, experts, top_k).unwrap();
        assert!(plans_equal(&got, &want), "tokens {tokens} experts {experts} top_k {top_k}");
    }
}

#[test]
fn parallel_plan_errors_equal_from_logits() {
    let mut logits = Lcg(5).vec(6 * 8, 1.0);
    logits[4 * 8 + 6] = f32::INFINITY;
    logits[2 * 8 + 3] = f32::NAN;
    let want = MoeBatchPlan::from_logits(&logits, 6, 8, 2).unwrap_err();
    assert_eq!(want, MoePlanError::InvalidLogit { token: 2, expert: 3 });
    assert_eq!(plan_par(&logits, 6, 8, 2).unwrap_err(), want, "the FIRST bad token, as the serial loop");
    for (t, e, k, n) in [(6, 8, 0, 48), (6, 8, 9, 48), (6, 0, 1, 0), (6, 8, 2, 47)] {
        assert_eq!(plan_par(&logits[..n], t, e, k).unwrap_err(), MoeBatchPlan::from_logits(&logits[..n], t, e, k).unwrap_err());
    }
}
