//! The Qwen3-30B-A3B drafter for the 27B + 30B wormhole (L190), host side: everything but the routed
//! experts resident (`load_qwen3_coder_dense`, about 6 GB), the 8 routed experts of every layer read
//! through the Heartbeat ring (`ExpertCache`: fixed slots, LRU, BLAKE3-checked before use). Same arithmetic
//! as the resident CPU reference (`forward_token` + `compute_logits`), so the same tokens and logits.

use std::path::Path;
use std::time::Instant;

use aien_heartbeat::manifest::Manifest;
use aien_inference_abi::moe_plan::MoeBatchPlan;
use aien_inference_abi::qwen3_coder::{
    compute_logits, forward_token_with, load_qwen3_coder_dense, Qwen3CoderState, Qwen3CoderWeights,
};
use aien_inference_abi::qwen3_moe::{
    reference_moe_forward_experts, reference_router_logits_bf16, ExpertView, OracleActivations,
};

use crate::align::Lm;
use crate::cache::ExpertCache;

const H: usize = 2048;
const I: usize = 768;
const BLOCK: usize = 128;

/// One expert's four reference inputs, owned (gate rows then up rows, as the packed layer holds them).
struct Owned {
    gate_up: Vec<u8>,
    gate_up_scales: Vec<u16>,
    down: Vec<u8>,
    down_scales: Vec<u16>,
}

fn words(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

pub struct Moe30Ring {
    w: Qwen3CoderWeights,
    routers: Vec<Vec<u16>>,
    pub cache: ExpertCache,
    st: Qwen3CoderState,
    len: usize,
    pub steps: usize,
    pub ns: u128,
    /// L312: routed expert ids per layer of the last `step` (the model's own routing, recorded for the engine's schedule).
    pub last_routes: Vec<Vec<u16>>,
}

impl Moe30Ring {
    /// `manifest` from `Manifest::build` (or a saved one); `slots` experts stay resident in the ring.
    pub fn open(dir: &Path, manifest: Manifest, slots: usize, workers: usize, direct: bool) -> Result<Self, String> {
        let (w, routers) = load_qwen3_coder_dense(dir).map_err(|e| format!("{e:?}"))?;
        let cache = ExpertCache::new(manifest, slots, workers, direct)?;
        Ok(Self { w, routers, cache, st: Qwen3CoderState::new(), len: 0, steps: 0, ns: 0, last_routes: Vec::new() })
    }

    fn expert(cache: &mut ExpertCache, layer: usize, e: usize) -> Result<Owned, String> {
        let p = format!("model.layers.{layer}.mlp.experts.{e}");
        let (l, x) = (layer as u16, e as u16);
        let mut gate_up = vec![0u8; 2 * I * H];
        let (g, u) = gate_up.split_at_mut(I * H);
        cache.read_tensor(l, x, &format!("{p}.gate_proj.weight"), "F8_E4M3", &[I, H], g)?;
        cache.read_tensor(l, x, &format!("{p}.up_proj.weight"), "F8_E4M3", &[I, H], u)?;
        let gs = (I / BLOCK) * (H / BLOCK);
        let mut gus = vec![0u8; 2 * gs * 2];
        let (g, u) = gus.split_at_mut(gs * 2);
        cache.read_tensor(l, x, &format!("{p}.gate_proj.weight_scale_inv"), "BF16", &[I / BLOCK, H / BLOCK], g)?;
        cache.read_tensor(l, x, &format!("{p}.up_proj.weight_scale_inv"), "BF16", &[I / BLOCK, H / BLOCK], u)?;
        let mut down = vec![0u8; H * I];
        cache.read_tensor(l, x, &format!("{p}.down_proj.weight"), "F8_E4M3", &[H, I], &mut down)?;
        let mut ds = vec![0u8; gs * 2];
        cache.read_tensor(l, x, &format!("{p}.down_proj.weight_scale_inv"), "BF16", &[H / BLOCK, I / BLOCK], &mut ds)?;
        Ok(Owned { gate_up, gate_up_scales: words(&gus), down, down_scales: words(&ds) })
    }

    /// One token: the final logits.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let mut err = None;
        let (cache, routers) = (&mut self.cache, &self.routers);
        self.last_routes.clear();
        let last_routes = &mut self.last_routes;
        let h = forward_token_with(&self.w, token, self.len, &mut self.st, &mut |layer, post| {
            let logits = reference_router_logits_bf16(&routers[layer], H, post, 1);
            let plan = match MoeBatchPlan::qwen3_coder_a3b(&logits, 1) {
                Ok(p) => p,
                Err(e) => {
                    err.get_or_insert(format!("layer {layer} plan: {e:?}"));
                    return vec![0.0; H];
                }
            };
            last_routes.push(plan.topk_experts[..plan.top_k].iter().map(|&e| e as u16).collect());
            let mut owned: Vec<(usize, Owned)> = Vec::with_capacity(plan.top_k);
            for &e in &plan.topk_experts[..plan.top_k] {
                match Self::expert(cache, layer, e as usize) {
                    Ok(o) => owned.push((e as usize, o)),
                    Err(m) => {
                        err.get_or_insert(format!("layer {layer} expert {e}: {m}"));
                        return vec![0.0; H];
                    }
                }
            }
            let view = |e: usize| {
                let o = &owned.iter().find(|(x, _)| *x == e).expect("routed expert fetched").1;
                ExpertView { gate_up: &o.gate_up, gate_up_scales: &o.gate_up_scales, down: &o.down, down_scales: &o.down_scales }
            };
            reference_moe_forward_experts(H, I, post, &plan, OracleActivations::Fp32, &view)
        });
        if let Some(e) = err {
            return Err(e);
        }
        let logits = compute_logits(&self.w, &h);
        self.len += 1;
        self.steps += 1;
        self.ns += t0.elapsed().as_nanos();
        Ok(logits)
    }

    /// Host bytes of the resident part (dense weights and routers; the ring's slots are counted apart).
    pub fn dense_bytes(&self) -> usize {
        let a: usize = self
            .w
            .attns
            .iter()
            .map(|a| {
                4 * (a.q_proj.len() + a.k_proj.len() + a.v_proj.len() + a.o_proj.len() + a.q_norm.len() + a.k_norm.len()
                    + a.input_norm.len() + a.post_norm.len())
            })
            .sum();
        a + 4 * (self.w.embed.len() + self.w.lm_head.len() + self.w.final_norm.len())
            + 2 * self.routers.iter().map(Vec::len).sum::<usize>()
    }
}

/// First index of the largest logit (the greedy rule every L080 / L190 tool uses).
pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

impl Lm for Moe30Ring {
    fn feed(&mut self, t: &[u32]) -> Result<Vec<u32>, String> {
        t.iter().map(|&x| self.step(x).map(|l| argmax(&l))).collect()
    }
    fn truncate(&mut self, len: usize) -> Result<(), String> {
        for kv in &mut self.st.layers {
            kv.keys.truncate(len);
            kv.values.truncate(len);
        }
        self.len = len.min(self.len);
        Ok(())
    }
    fn len(&self) -> usize {
        self.len
    }
}
