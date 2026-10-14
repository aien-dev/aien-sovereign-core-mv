//! Host-only tests of the whole-prompt driver with fakes (no chip, no model load).
//!
//! The fake device mimics the GPU through raw addresses: a buffer's `va` is its host address, and
//! every read or write the real kernel would make is bounds-checked against the live allocations
//! (so a missing `bm` row of padding fails loudly). The fake quantiser is LOSSLESS (it keeps the
//! f32 rows beside the e4m3 buffer): these tests check the plumbing (order, shapes, row tables,
//! sort, combine), not FP8 numerics, which omega's own host/sim/chip gates cover.
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use aien_omega_pf::{Layout, PfBuffer, PrefillAttention, PrefillDevice, QgemmCall, QuantIn, Tile, TILE_BYTES};

use super::*;
use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};
use crate::qwen3_moe_omega::{DenseMatmul, LmHead, OmegaLayer};

// ---------------------------------------------------------------------------
// Fakes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    Quant { k: usize, input: QuantIn, rows: usize },
    Gemm { n: usize, k: usize, layout: Layout, ntiles: usize, gather: bool },
    Attn { layer: u32, pos0: u32, m: u32, dev: bool },
    /// W-L1G-D device router logits / combine (tests/devhost.rs).
    Router { k: usize, e: usize, rows: usize },
    Combine { h: usize, top_k: usize, rows: usize },
}

type Log = Rc<RefCell<Vec<Ev>>>;

/// Live allocations: start address -> bytes.
type Reg = Rc<RefCell<HashMap<u64, usize>>>;

fn check_range(reg: &Reg, va: u64, bytes: usize, what: &str) {
    let ok = reg.borrow().iter().any(|(&s, &n)| va >= s && va + bytes as u64 <= s + n as u64);
    assert!(ok, "{what}: {bytes} bytes at {va:#x} outside every live buffer");
}

fn read_f32(reg: &Reg, va: u64, n: usize, what: &str) -> Vec<f32> {
    check_range(reg, va, n * 4, what);
    // SAFETY: range checked against a live allocation (the address is a host pointer in the fake).
    unsafe { std::slice::from_raw_parts(va as *const f32, n) }.to_vec()
}

fn write_f32(reg: &Reg, va: u64, v: &[f32], what: &str) {
    check_range(reg, va, v.len() * 4, what);
    // SAFETY: as read_f32.
    unsafe { std::slice::from_raw_parts_mut(va as *mut f32, v.len()) }.copy_from_slice(v);
}

struct FakeBuf {
    mem: Vec<u64>,
    bytes: usize,
    host: bool,
    reg: Reg,
}

impl FakeBuf {
    fn bytes_ref(&self) -> &[u8] {
        // SAFETY: mem holds at least `bytes` bytes; u8 has no alignment needs.
        unsafe { std::slice::from_raw_parts(self.mem.as_ptr().cast(), self.bytes) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as bytes_ref.
        unsafe { std::slice::from_raw_parts_mut(self.mem.as_mut_ptr().cast(), self.bytes) }
    }
}

impl PfBuffer for FakeBuf {
    fn va(&self) -> u64 {
        self.mem.as_ptr() as u64
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn host(&self) -> Option<&[u8]> {
        self.host.then(|| self.bytes_ref())
    }
    fn host_mut(&mut self) -> Option<&mut [u8]> {
        if self.host {
            Some(self.bytes_mut())
        } else {
            None
        }
    }
}

impl Drop for FakeBuf {
    fn drop(&mut self) {
        self.reg.borrow_mut().remove(&self.va());
    }
}

struct QEntry {
    s_va: u64,
    k: usize,
    rows: usize,
    data: Vec<f32>,
}

struct FakeDevice {
    bm: usize,
    reg: Reg,
    log: Log,
    /// w_va -> (s_va, rows n, cols k) of every expert of every FP8 tensor.
    weights: HashMap<u64, (u64, usize, usize)>,
    quant: HashMap<u64, QEntry>,
    /// Negative control: the gate|up GEMM ignores the row index (reads row i % rows: valid, wrong token).
    mutant_ignore_rowidx: bool,
}

impl FakeDevice {
    fn new(bm: usize, model: &TestModel, log: Log) -> Self {
        let reg: Reg = Rc::new(RefCell::new(HashMap::new()));
        let mut weights = HashMap::new();
        for t in model.tensors() {
            for ex in 0..t.experts {
                let (w, s) = t.tensor_entry(ex).unwrap();
                reg.borrow_mut().insert(w, t.n * t.k * 4);
                weights.insert(w, (s, t.n, t.k));
            }
        }
        Self { bm, reg, log, weights, quant: HashMap::new(), mutant_ignore_rowidx: false }
    }
    fn alloc(&mut self, bytes: usize, host: bool) -> FakeBuf {
        let b = FakeBuf { mem: vec![0u64; bytes.div_ceil(8)], bytes, host, reg: self.reg.clone() };
        self.reg.borrow_mut().insert(b.va(), bytes);
        b
    }
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

fn dot(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

impl PrefillDevice for FakeDevice {
    type Buf = FakeBuf;
    fn alloc_dev(&mut self, bytes: usize) -> Result<FakeBuf, String> {
        Ok(self.alloc(bytes, false))
    }
    fn alloc_host(&mut self, bytes: usize) -> Result<FakeBuf, String> {
        Ok(self.alloc(bytes, true))
    }
    fn qgemm_bm(&self) -> usize {
        self.bm
    }
    fn quant(&mut self, k: usize, input: QuantIn, rows: usize, x_va: u64, q_va: u64, s_va: u64) -> Result<u64, String> {
        self.log.borrow_mut().push(Ev::Quant { k, input, rows });
        check_range(&self.reg, q_va, rows * k, "quant q");
        check_range(&self.reg, s_va, rows * (k / 128) * 4, "quant s");
        let data = match input {
            QuantIn::F32 => read_f32(&self.reg, x_va, rows * k, "quant x f32"),
            QuantIn::Bf16 => {
                check_range(&self.reg, x_va, rows * k * 2, "quant x bf16");
                // SAFETY: range checked.
                unsafe { std::slice::from_raw_parts(x_va as *const u16, rows * k) }.iter().map(|&b| bf16_to_f32(b)).collect()
            }
        };
        self.quant.insert(q_va, QEntry { s_va, k, rows, data });
        Ok(10)
    }
    fn qgemm(&mut self, c: &QgemmCall) -> Result<u64, String> {
        let gather = c.rowidx_va != 0 && !self.mutant_ignore_rowidx;
        self.log.borrow_mut().push(Ev::Gemm { n: c.n, k: c.k, layout: c.layout, ntiles: c.ntiles, gather: c.rowidx_va != 0 });
        check_range(&self.reg, c.tiles_va, c.ntiles * TILE_BYTES, "tile table");
        let q = self.quant.get(&c.xq_va).expect("GEMM activations were quantised first");
        assert_eq!(q.s_va, c.xs_va, "GEMM scales are the quantiser's");
        assert_eq!(q.k, c.k, "GEMM k equals the quantised row width");
        let kb = c.k / 128;
        for ti in 0..c.ntiles {
            // SAFETY: tile table range checked above.
            let tb = unsafe { std::slice::from_raw_parts((c.tiles_va + (ti * TILE_BYTES) as u64) as *const u8, TILE_BYTES) };
            let t = Tile::from_bytes(tb);
            assert!(t.cnt >= 1 && t.cnt as usize <= self.bm, "tile row count");
            let &(s_va, n, k) = self.weights.get(&t.w_va).expect("tile weight address is an FP8 tensor expert");
            assert_eq!((s_va, n, k), (t.s_va, c.n, c.k), "tile scale address and tensor shape");
            // SAFETY: registered weight range of n * k f32.
            let w = unsafe { std::slice::from_raw_parts(t.w_va as *const f32, n * k) };
            for r in 0..self.bm {
                let idx = t.row0 as usize + r;
                // the kernel reads every row of the tile (to row0 + bm - 1), stored or not
                let src = if gather {
                    check_range(&self.reg, c.rowidx_va + idx as u64 * 4, 4, "row index (tile padding)");
                    // SAFETY: range checked.
                    unsafe { *((c.rowidx_va + idx as u64 * 4) as *const u32) as usize }
                } else {
                    if c.rowidx_va != 0 { idx % q.rows } else { idx } // the mutant reads sorted row i % rows instead of its token
                };
                check_range(&self.reg, c.xq_va + (src * c.k) as u64, c.k, "activation row (tile padding)");
                check_range(&self.reg, c.xs_va + (src * kb * 4) as u64, kb * 4, "activation scales (tile padding)");
                if r >= t.cnt as usize {
                    continue;
                }
                assert!(src < q.rows, "row {src} was quantised");
                let x = &q.data[src * c.k..(src + 1) * c.k];
                match c.layout {
                    Layout::Plain => {
                        let y: Vec<f32> = (0..n).map(|col| dot(&w[col * k..(col + 1) * k], x)).collect();
                        write_f32(&self.reg, c.y_va + (idx * n * 4) as u64, &y, "GEMM output row");
                    }
                    Layout::SwigluNat => {
                        let i = n / 2;
                        let va = c.y_va + (idx * i * 2) as u64;
                        check_range(&self.reg, va, i * 2, "act row");
                        // SAFETY: range checked.
                        let out = unsafe { std::slice::from_raw_parts_mut(va as *mut u16, i) };
                        for (j, o) in out.iter_mut().enumerate() {
                            let g = dot(&w[j * k..(j + 1) * k], x);
                            let u = dot(&w[(i + j) * k..(i + j + 1) * k], x);
                            *o = f32_to_bf16(silu(g) * u);
                        }
                    }
                }
            }
        }
        Ok(100)
    }
}

/// A deterministic causal stand-in for attention: per layer it keeps every position's k and v and
/// writes `0.5 q + mean(v[0..=pos]) + 0.1 mean(k[0..=pos])` per query head (GQA head mapping).
#[derive(Clone)]
struct AttnCore {
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

impl AttnCore {
    fn new(c: &Qwen3MoeConfig) -> Self {
        Self { q_heads: c.q_heads, kv_heads: c.kv_heads, hd: c.head_dim, k: vec![Vec::new(); c.layers], v: vec![Vec::new(); c.layers] }
    }
    fn step(&mut self, layer: usize, pos: usize, qkv: &[f32], out: &mut [f32]) {
        let (qd, kvd) = (self.q_heads * self.hd, self.kv_heads * self.hd);
        assert_eq!(pos, self.v[layer].len(), "positions are consecutive and the prefix is cached");
        self.k[layer].push(qkv[qd..qd + kvd].to_vec());
        self.v[layer].push(qkv[qd + kvd..qd + 2 * kvd].to_vec());
        let n = (pos + 1) as f32;
        let group = self.q_heads / self.kv_heads;
        for hh in 0..self.q_heads {
            let kvh = hh / group;
            for d in 0..self.hd {
                let c = kvh * self.hd + d;
                let mv: f32 = self.v[layer].iter().map(|r| r[c]).sum::<f32>() / n;
                let mk: f32 = self.k[layer].iter().map(|r| r[c]).sum::<f32>() / n;
                out[hh * self.hd + d] = 0.5 * qkv[hh * self.hd + d] + mv + 0.1 * mk;
            }
        }
    }
}

struct FakeAttn {
    core: AttnCore,
    log: Log,
    dev: Option<Reg>,
    qkv_n: usize,
    qd: usize,
}

impl PrefillAttention for FakeAttn {
    fn attend(&mut self, layer: u32, pos0: u32, m: u32, qkv: &[f32], _eps: f32, out_scale: f32, out: &mut [f32]) -> Result<u64, String> {
        self.log.borrow_mut().push(Ev::Attn { layer, pos0, m, dev: false });
        assert_eq!(out_scale, 1.0);
        assert_eq!((qkv.len(), out.len()), (m as usize * self.qkv_n, m as usize * self.qd));
        for t in 0..m as usize {
            self.core.step(layer as usize, pos0 as usize + t, &qkv[t * self.qkv_n..(t + 1) * self.qkv_n], &mut out[t * self.qd..(t + 1) * self.qd]);
        }
        Ok(1000)
    }
    fn has_dev(&self) -> bool {
        self.dev.is_some()
    }
    fn attend_dev(&mut self, layer: u32, pos0: u32, m: u32, qkv_va: u64, _eps: f32, _out_scale: f32, out_va: u64) -> Option<Result<u64, String>> {
        let reg = self.dev.clone()?;
        self.log.borrow_mut().push(Ev::Attn { layer, pos0, m, dev: true });
        let mu = m as usize;
        let qkv = read_f32(&reg, qkv_va, mu * self.qkv_n, "attention q|k|v");
        let mut out = vec![0.0f32; mu * self.qd];
        for t in 0..mu {
            self.core.step(layer as usize, pos0 as usize + t, &qkv[t * self.qkv_n..(t + 1) * self.qkv_n], &mut out[t * self.qd..(t + 1) * self.qd]);
        }
        write_f32(&reg, out_va, &out, "attention out");
        Some(Ok(1000))
    }
}

/// FP8 tensor stand-in: f32 rows `[n][k]` per expert at stable addresses.
#[derive(Debug)]
struct FakeFp8 {
    experts: usize,
    n: usize,
    k: usize,
    w: Vec<Vec<f32>>,
    s: Vec<Vec<u64>>,
}

impl Fp8Gemv for FakeFp8 {
    fn experts(&self) -> usize {
        self.experts
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn grouped_f32(&self, _: &[u32], _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("fake: the whole-prompt path never calls the GEMV".into())
    }
    fn tensor_entry(&self, expert: usize) -> Result<(u64, u64), String> {
        Ok((self.w[expert].as_ptr() as u64, self.s[expert].as_ptr() as u64))
    }
}

#[derive(Debug)]
struct NoDense;
impl DenseMatmul for NoDense {
    fn k(&self) -> usize {
        1
    }
    fn n(&self) -> usize {
        1
    }
    fn matmul_f32(&self, _: usize, _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("fake".into())
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(scale)).collect()
    }
}

struct TestLayer {
    qkv: Arc<FakeFp8>,
    o: Arc<FakeFp8>,
    gu: Arc<FakeFp8>,
    dn: Arc<FakeFp8>,
}

struct TestModel {
    weights: Qwen3MoeOmegaWeights,
    layers: Vec<TestLayer>,
}

impl TestModel {
    fn tensors(&self) -> impl Iterator<Item = &FakeFp8> {
        self.layers.iter().flat_map(|l| [l.qkv.as_ref(), l.o.as_ref(), l.gu.as_ref(), l.dn.as_ref()])
    }
}

fn config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        layers: 2,
        hidden: 128,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        experts: 4,
        top_k: 2,
        moe_intermediate: 128,
        vocab: 16,
        rms_eps: 1e-6,
        rope_theta: 1e7,
        norm_topk_prob: true,
        tie_word_embeddings: false,
        eos_token_ids: vec![0],
    }
}

fn model(c: &Qwen3MoeConfig, seed: u64) -> TestModel {
    let mut r = Lcg(seed);
    let (h, qd, e, i) = (c.hidden, c.q_dim(), c.experts, c.moe_intermediate);
    let tensor = |experts: usize, n: usize, k: usize, r: &mut Lcg| {
        let scale = 1.0 / (k as f32).sqrt();
        Arc::new(FakeFp8 { experts, n, k, w: (0..experts).map(|_| r.vec(n * k, scale)).collect(), s: (0..experts).map(|_| vec![0u64; 2]).collect() })
    };
    let mut layers = Vec::new();
    let mut omega_layers = Vec::new();
    for _ in 0..c.layers {
        let tl = TestLayer { qkv: tensor(1, c.qkv_n(), h, &mut r), o: tensor(1, h, qd, &mut r), gu: tensor(e, 2 * i, h, &mut r), dn: tensor(e, h, i, &mut r) };
        omega_layers.push(OmegaLayer {
            input_norm: r.vec(h, 0.2).iter().map(|v| 1.0 + v).collect(),
            post_norm: r.vec(h, 0.2).iter().map(|v| 1.0 + v).collect(),
            q_norm: vec![1.0; c.head_dim],
            k_norm: vec![1.0; c.head_dim],
            router: r.vec(e * h, 0.3),
            attn: AttnWeights::Fp8 { qkv: tl.qkv.clone(), o: tl.o.clone() },
            experts: ExpertWeights::Fp8 { gate_up: tl.gu.clone(), down: tl.dn.clone() },
        });
        layers.push(tl);
    }
    let weights = Qwen3MoeOmegaWeights {
        config: c.clone(),
        embed_bf16: r.vec(c.vocab * h, 1.0).into_iter().map(f32_to_bf16).collect(),
        layers: omega_layers,
        final_norm: r.vec(h, 0.2).iter().map(|v| 1.0 + v).collect(),
        lm_head: LmHead::Dense(Arc::new(NoDense)),
        device_router: None,
    };
    TestModel { weights, layers }
}

/// The naive host reference: one token at a time through every layer (decode order), routing by
/// its own sort (ties to the lower id), act rounded to bf16 as the SwiGLU epilogue stores it.
fn reference(tm: &TestModel, tokens: &[u32]) -> Vec<f32> {
    let w = &tm.weights;
    let c = &w.config;
    let (h, qd, e, k, i, eps) = (c.hidden, c.q_dim(), c.experts, c.top_k, c.moe_intermediate, c.rms_eps);
    let mut core = AttnCore::new(c);
    let matvec = |t: &FakeFp8, ex: usize, x: &[f32]| -> Vec<f32> { (0..t.n).map(|r| dot(&t.w[ex][r * t.k..(r + 1) * t.k], x)).collect() };
    let mut last = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut x = w.embed_row(tok);
        for (l, (layer, tl)) in w.layers.iter().zip(&tm.layers).enumerate() {
            let mut xn = vec![0.0f32; h];
            rmsnorm(&x, &layer.input_norm, eps, &mut xn);
            let qkv = matvec(&tl.qkv, 0, &xn);
            let mut att = vec![0.0f32; qd];
            core.step(l, pos, &qkv, &mut att);
            for (xi, p) in x.iter_mut().zip(matvec(&tl.o, 0, &att)) {
                *xi += p;
            }
            let mut post = vec![0.0f32; h];
            rmsnorm(&x, &layer.post_norm, eps, &mut post);
            let logits: Vec<f32> = (0..e).map(|ex| dot(&layer.router[ex * h..(ex + 1) * h], &post)).collect();
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let probs: Vec<f32> = order[..k].iter().map(|&ex| (logits[ex] - logits[order[0]]).exp()).collect();
            let sum: f32 = probs.iter().sum();
            let mut acc = vec![0.0f32; h];
            for (slot, &ex) in order[..k].iter().enumerate() {
                let gu = matvec(&tl.gu, ex, &post);
                let act: Vec<f32> = (0..i).map(|j| bf16_to_f32(f32_to_bf16(silu(gu[j]) * gu[i + j]))).collect();
                let wt = probs[slot] / sum;
                for (a, d) in acc.iter_mut().zip(matvec(&tl.dn, ex, &act)) {
                    *a += wt * d;
                }
            }
            for (xi, a) in x.iter_mut().zip(&acc) {
                *xi += a;
            }
        }
        last = vec![0.0f32; h];
        rmsnorm(&x, &w.final_norm, eps, &mut last);
    }
    last
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / (1.0 + y.abs())).fold(0.0, f32::max)
}

const TOKENS: [u32; 7] = [3, 9, 1, 14, 3, 7, 11];
const TOL: f32 = 1e-5;

fn setup(bm: usize, dev_attn: bool) -> (TestModel, WholePromptPrefill<FakeDevice>, FakeAttn, Log) {
    let c = config();
    let tm = model(&c, 7);
    let log: Log = Rc::new(RefCell::new(Vec::new()));
    let fd = FakeDevice::new(bm, &tm, log.clone());
    let reg = fd.reg.clone();
    let drv = WholePromptPrefill::new(fd, &tm.weights).unwrap();
    let attn = FakeAttn { core: AttnCore::new(&c), log: log.clone(), dev: dev_attn.then_some(reg), qkv_n: c.qkv_n(), qd: c.q_dim() };
    (tm, drv, attn, log)
}

// ---------------------------------------------------------------------------
// Tests (host only).
// ---------------------------------------------------------------------------

#[test]
fn call_order_and_launch_counts_per_layer() {
    let (tm, mut drv, mut attn, log) = setup(4, false);
    let c = tm.weights.config.clone();
    let m = 5;
    drv.run(&tm.weights, &mut attn, &TOKENS[..m], 0).unwrap();
    let (h, qd, i, g) = (c.hidden, c.q_dim(), c.moe_intermediate, m * c.top_k);
    let log = log.borrow();
    assert_eq!(log.len(), 9 * c.layers);
    for (l, ev) in log.chunks(9).enumerate() {
        let f32q = |k| Ev::Quant { k, input: QuantIn::F32, rows: m };
        let dense = |n, k| Ev::Gemm { n, k, layout: Layout::Plain, ntiles: m.div_ceil(4), gather: false };
        assert_eq!(ev[0], f32q(h), "layer {l}: quantise normed x");
        assert_eq!(ev[1], dense(c.qkv_n(), h), "layer {l}: one q|k|v GEMM");
        assert_eq!(ev[2], Ev::Attn { layer: l as u32, pos0: 0, m: m as u32, dev: false }, "layer {l}: ONE attention call");
        assert_eq!(ev[3], f32q(qd), "layer {l}: quantise attention out");
        assert_eq!(ev[4], dense(h, qd), "layer {l}: one o GEMM");
        assert_eq!(ev[5], f32q(h), "layer {l}: quantise post");
        assert!(matches!(ev[6], Ev::Gemm { n, k, layout: Layout::SwigluNat, gather: true, .. } if n == 2 * i && k == h), "layer {l}: gate|up {:?}", ev[6]);
        assert_eq!(ev[7], Ev::Quant { k: i, input: QuantIn::Bf16, rows: g }, "layer {l}: quantise bf16 act");
        assert!(matches!(ev[8], Ev::Gemm { n, k, layout: Layout::Plain, gather: false, .. } if n == h && k == i), "layer {l}: down {:?}", ev[8]);
        // grouped tiles: ceil per expert, at least ceil(g / bm)
        let (Ev::Gemm { ntiles: gu, .. }, Ev::Gemm { ntiles: dn, .. }) = (&ev[6], &ev[8]) else { unreachable!() };
        assert_eq!(gu, dn);
        assert!(*gu >= g.div_ceil(4) && *gu <= g.div_ceil(4) + c.experts);
    }
    let s = &drv.stats;
    assert_eq!((s.qkv, s.attn, s.o, s.gate_up, s.down, s.quant), (2, 2, 2, 2, 2, 8));
    assert_eq!(s.launches(), 18);
    assert!(!s.attn_dev);
    assert_eq!((s.tokens, s.layers), (m, 2));
}

#[test]
fn whole_prompt_matches_token_by_token_reference() {
    let (tm, mut drv, mut attn, _) = setup(4, false);
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    let want = reference(&tm, &TOKENS);
    let d = max_rel_diff(&got, &want);
    assert!(d <= TOL, "whole-prompt vs token-by-token reference: max rel diff {d}");
}

#[test]
fn device_attention_entry_gives_the_same_numbers() {
    let (tm, mut drv, mut attn, log) = setup(4, true);
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    assert!(drv.stats.attn_dev);
    assert!(log.borrow().iter().filter(|e| matches!(e, Ev::Attn { .. })).all(|e| matches!(e, Ev::Attn { dev: true, .. })));
    let (tm2, mut drv2, mut attn2, _) = setup(4, false);
    let host_path = drv2.run(&tm2.weights, &mut attn2, &TOKENS, 0).unwrap();
    assert_eq!(got, host_path, "device and host attention entries: same plumbing, same numbers");
    assert!(max_rel_diff(&got, &reference(&tm, &TOKENS)) <= TOL);
}

#[test]
fn prefix_then_continuation_matches_reference() {
    let (tm, mut drv, mut attn, log) = setup(4, false);
    drv.run(&tm.weights, &mut attn, &TOKENS[..3], 0).unwrap();
    let got = drv.run(&tm.weights, &mut attn, &TOKENS[3..], 3).unwrap();
    assert!(log.borrow().contains(&Ev::Attn { layer: 1, pos0: 3, m: 4, dev: false }));
    let d = max_rel_diff(&got, &reference(&tm, &TOKENS));
    assert!(d <= TOL, "prefix 3 + continuation 4 vs reference over 7: {d}");
}

#[test]
fn tile_size_does_not_change_the_result() {
    let mut outs = Vec::new();
    for bm in [1, 3, 64] {
        let (tm, mut drv, mut attn, _) = setup(bm, false);
        outs.push(drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap());
    }
    assert!(outs.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn negative_control_ignoring_the_row_index_is_caught() {
    let (tm, mut drv, mut attn, _) = setup(4, false);
    drv.dev.mutant_ignore_rowidx = true;
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    let d = max_rel_diff(&got, &reference(&tm, &TOKENS));
    assert!(d > 1e-2, "a gate|up GEMM that ignores the token gather must disagree, diff {d}");
}

#[test]
fn expert_sort_groups_rows_by_expert_stably() {
    // 4 tokens, 5 experts, top 2: token-major ids [2,0], [0,4], [2,1], [4,0]
    let logits: Vec<f32> = [
        [2.0, 0.0, 3.0, -1.0, -2.0],
        [3.0, -1.0, 0.0, -2.0, 2.0],
        [0.0, 2.0, 3.0, -1.0, -2.0],
        [2.0, -1.0, 0.0, -2.0, 3.0],
    ]
    .concat();
    let plan = MoeBatchPlan::from_logits(&logits, 4, 5, 2).unwrap();
    assert_eq!(plan.topk_experts, vec![2, 0, 0, 4, 2, 1, 4, 0]);
    let s = ExpertSort::from_plan(&plan);
    assert_eq!(s.counts, vec![3, 1, 2, 0, 2]);
    // expert 0: tokens 0, 1, 3; expert 1: token 2; expert 2: tokens 0, 2; expert 4: tokens 1, 3
    assert_eq!(s.rows, vec![0, 1, 3, 2, 0, 2, 1, 3]);
    for (a, &row) in s.inv.iter().enumerate() {
        assert_eq!(s.rows[row as usize] as usize, a / 2, "assignment {a} lands on its token's row");
        let ex = plan.topk_experts[a] as usize;
        let start: u32 = s.counts[..ex].iter().sum();
        assert!(row >= start && row < start + s.counts[ex], "assignment {a} lands in expert {ex}'s block");
    }
    let tiles = build_tiles(2, &[10, 11, 12, 13, 14], &[20, 21, 22, 23, 24], &s.counts).unwrap();
    let t: Vec<(u64, u32, u32)> = tiles.iter().map(|t| (t.w_va, t.row0, t.cnt)).collect();
    assert_eq!(t, vec![(10, 0, 2), (10, 2, 1), (11, 3, 1), (12, 4, 2), (14, 6, 2)]);
}

#[test]
fn refuses_an_empty_prompt_and_non_fp8_weights() {
    let (tm, mut drv, mut attn, _) = setup(4, false);
    assert!(drv.run(&tm.weights, &mut attn, &[], 0).is_err());
    let mut bad = model(&config(), 3);
    let qkv = Arc::new(NoDense) as Arc<dyn DenseMatmul>;
    bad.weights.layers[1].attn = AttnWeights::Bf16 { qkv: qkv.clone(), o: qkv };
    let err = layer_entries(&bad.weights).unwrap_err();
    assert!(err.contains("layer 1") && err.contains("FP8"), "{err}");
}

#[test]
fn cache_weights_refusal_leaves_the_in_place_path_working() {
    let (tm, mut drv, mut attn, _) = setup(4, false);
    assert!(drv.cache_weights(&tm.weights.config).is_err(), "the fake offers no device copy and its experts are not contiguous");
    let got = drv.run(&tm.weights, &mut attn, &TOKENS, 0).unwrap();
    assert!(max_rel_diff(&got, &reference(&tm, &TOKENS)) <= TOL);
}

#[test]
fn run_rows_gives_every_token_its_own_hidden_and_the_same_last_row_as_run() {
    // W-L4b: the speculative verify pass reads every position's hidden state from one call.
    let (tm, mut drv, mut attn, _) = setup(4, false);
    let h = tm.weights.config.hidden;
    drv.run(&tm.weights, &mut attn, &TOKENS[..3], 0).unwrap();
    let rows = drv.run_rows(&tm.weights, &mut attn, &TOKENS[3..], 3).unwrap();
    assert_eq!(rows.len(), 4 * h);
    for r in 0..4 {
        let want = reference(&tm, &TOKENS[..4 + r]);
        let d = max_rel_diff(&rows[r * h..(r + 1) * h], &want);
        assert!(d <= TOL, "row {r} (position {}) vs reference: {d}", 3 + r);
    }
    let (tm2, mut drv2, mut attn2, _) = setup(4, false);
    drv2.run(&tm2.weights, &mut attn2, &TOKENS[..3], 0).unwrap();
    let last = drv2.run(&tm2.weights, &mut attn2, &TOKENS[3..], 3).unwrap();
    assert_eq!(&rows[3 * h..], &last[..], "run_rows' last row is bit-equal to run");
}

/// W-L1G-D: the device router logits + combine path (fake device in tests/devhost.rs).
mod devhost;

/// W-L4c: the transposed, experts-side-by-side router gives the same bits as `router_logits`
/// (index-order sums), including all-zero and sign-mixed rows; and the serial / parallel row
/// loops give the same bits as each other.
#[test]
fn router_logits_t_and_serial_rows_are_bit_identical() {
    let (h, e) = (96usize, 24usize);
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
    };
    let router: Vec<f32> = (0..e * h).map(|_| rnd()).collect();
    let rt = transpose_router(&router, h, e);
    for m in [1usize, 2, 5, 9, SERIAL_ROWS, SERIAL_ROWS + 3] {
        let mut post: Vec<f32> = (0..m * h).map(|_| rnd()).collect();
        post[..h].fill(0.0);
        let (mut a, mut b) = (vec![0.0f32; m * e], vec![1.0f32; m * e]);
        router_logits(&router, &post, h, e, &mut a);
        router_logits_t(&rt, &post, h, e, &mut b);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b), "m = {m}");
        // combine_rows: serial (<= SERIAL_ROWS rows) and parallel agree with a plain reference.
        let top_k = 3;
        let down: Vec<f32> = (0..m * top_k * h).map(|_| rnd()).collect();
        let wts: Vec<f32> = (0..m * top_k).map(|_| rnd()).collect();
        let inv: Vec<u32> = (0..(m * top_k) as u32).rev().collect();
        let mut x: Vec<f32> = (0..m * h).map(|_| rnd()).collect();
        let mut want = x.clone();
        for r in 0..m {
            let mut acc = vec![0.0f32; h];
            for slot in 0..top_k {
                let gi = r * top_k + slot;
                for (j, a) in acc.iter_mut().enumerate() {
                    *a += wts[gi] * down[inv[gi] as usize * h + j];
                }
            }
            for j in 0..h {
                want[r * h + j] += acc[j];
            }
        }
        combine_rows(&mut x, &down, &wts, &inv, h, top_k);
        assert_eq!(bits(&x), bits(&want), "combine m = {m}");
    }
}
