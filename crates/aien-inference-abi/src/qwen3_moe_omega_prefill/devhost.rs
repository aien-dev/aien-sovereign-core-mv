//! The whole-prompt prefill with its two biggest host steps moved to the chip (W-L1G-D,
//! 2026-10-14). New file, a child module of the driver so it can use the driver's buffers.
//!
//! Profile at 2048 tokens before this step (hardware, sc ce59355): wall 2075 ms, of which device
//! calls 802 ms and host 1273 ms: router logits 943, routing plan 148, combine 94, residual 29,
//! norms 52, embedding 7. This path changes three of them:
//!  * router logits: omega_gpu_pf_router_logits, one launch per layer for all m tokens (reads the
//!    post-norm rows from the HOST buffer the quantiser reads too, the router weights from a
//!    GPU-cached packed copy, 1 MB per layer, made once by [`WholePromptPrefill::set_device_host_work`]).
//!    BIT-IDENTICAL to [`super::router_logits`] (FMUL then FADD in index order from -0.0), so the
//!    routing is exactly the host routing (no ties or flips to report).
//!  * MoE combine: the down GEMM writes its rows to a DEVICE buffer (no 134 MB uncached readback
//!    at 2048 tokens) and omega_gpu_pf_moe_combine forms each token's weighted expert sum
//!    (bit-identical to [`super::combine_rows`]'s accumulator) into a HOST buffer; the host adds it
//!    to the residual, which is exactly what `combine_rows` does last.
//!  * routing plan: still on the host, but [`plan_par`] (an exact, token-parallel replica of
//!    [`MoeBatchPlan::from_logits`]: top-k by partial selection, same tie order, same weights).
//!
//! Everything else is [`WholePromptPrefill::run`] unchanged: the same kernels in the same order,
//! so the output equals the host path's bit for bit (checked on the fake device in the driver's
//! tests and on the chip by own30b_ttft's `--devhost` phase).
//!
//! Left on the host (and why): RMSNorms (the host sums squares in f64; omega's IR has no f64 ops,
//! so a device norm would not be bit-identical), the residual add of the o projection (needs the
//! host residual), the expert sort (O(m * top_k)) and the embedding gather.
use std::time::Instant;

use aien_omega_pf::{build_tiles, Layout, PfBuffer, PrefillAttention, PrefillDevice, QgemmCall, QuantIn, TILE_BYTES};
use rayon::prelude::*;

use super::{as_u32_mut, for_rows, host, host_mut, read_dev, rotate_logits, ExpertSort, PrefillStats, Region, WholePromptPrefill, WpMutant, ATTN_OUT_SCALE};
use crate::moe_plan::{MoeBatchPlan, MoePlanError};
use crate::qwen3_moe_omega::{Qwen3MoeConfig, Qwen3MoeOmegaWeights};
use crate::tensor::rmsnorm;

/// Device state of the moved host work: the packed routers and the per-sizing buffers.
pub struct DevHost<B> {
    /// Per layer: the router in omega's packed layout R4 (`aien_omega_pf::pack_router`), DEVICE.
    routers: Vec<B>,
    /// Rows per router CTA (the router reads / writes whole tiles of `t` rows).
    t: usize,
    bufs: Option<DhBufs<B>>,
}

/// Buffers of one sizing (`m` tokens, `g = m * top_k` expert rows).
struct DhBufs<B> {
    m: usize,
    /// HOST f32 [ceil(m / t) * t][hidden]: post-norm rows (quantiser AND router input; pad rows zero).
    post: B,
    /// HOST f32 [ceil(m / t) * t][experts]: router logits.
    logits: B,
    /// DEVICE f32 [g][hidden]: down GEMM rows (sorted order).
    down: B,
    /// HOST u32 / f32 [g]: sorted row and weight of each token-major assignment.
    inv: B,
    wts: B,
    /// HOST f32 [m][hidden]: each token's weighted expert sum.
    moe: B,
}

impl<D: PrefillDevice> WholePromptPrefill<D> {
    /// Turns the device router logits + combine on (packs and uploads every layer's router once,
    /// about `layers x (hidden / 4 + 1) x experts x 16` bytes) or off. OFF is the default; with it
    /// off [`Self::run`] is exactly the host path.
    pub fn set_device_host_work(&mut self, weights: &Qwen3MoeOmegaWeights, on: bool) -> Result<(), String> {
        if !on {
            self.devhost = None;
            return Ok(());
        }
        if self.devhost.is_some() {
            return Ok(());
        }
        let t = self.dev.router_rows().ok_or("device router logits / combine are not linked in this build")?;
        let c = &weights.config;
        let (h, e) = (c.hidden, c.experts);
        if t == 0 {
            return Err("device router: 0 rows per launch block".into());
        }
        // Shapes the chip kernels cannot take (omega_rl_spec_ok / omega_mc_spec_ok) are refused by the
        // device at the first launch; Qwen3-Coder-30B-A3B (hidden 2048, 128 experts, top 8) is in range.
        let bytes = aien_omega_pf::router_packed_len(e, h) * 4;
        let mut staging = self.dev.alloc_host(bytes)?;
        let mut routers = Vec::with_capacity(weights.layers.len());
        for (l, layer) in weights.layers.iter().enumerate() {
            if layer.router.len() != e * h {
                return Err(format!("layer {l}: router has {} values, want {}", layer.router.len(), e * h));
            }
            let wq = aien_omega_pf::pack_router(e, h, &layer.router);
            host_mut(&mut staging, "router staging")?[..wq.len()].copy_from_slice(&wq);
            let dst = self.dev.alloc_dev(bytes)?;
            self.dev.copy_va(dst.va(), staging.va(), bytes).map_err(|err| format!("layer {l}: router upload: {err}"))?;
            routers.push(dst);
        }
        self.devhost = Some(DevHost { routers, t, bufs: None });
        Ok(())
    }

    /// Whether [`Self::run`] takes the device router / combine path.
    pub fn device_host_work(&self) -> bool {
        self.devhost.is_some()
    }

    fn dh_ensure(&mut self, c: &Qwen3MoeConfig, m: usize) -> Result<(), String> {
        let dev = &mut self.dev;
        let dh = self.devhost.as_mut().expect("device host work on");
        if dh.bufs.as_ref().is_some_and(|b| b.m >= m) {
            return Ok(());
        }
        dh.bufs = None;
        let (h, e, g) = (c.hidden, c.experts, m * c.top_k);
        let rp = m.div_ceil(dh.t) * dh.t;
        let mut post = dev.alloc_host(rp * h * 4)?;
        host_mut(&mut post, "post")?.fill(0.0); // pad rows: read by the router, never used
        dh.bufs = Some(DhBufs {
            m,
            post,
            logits: dev.alloc_host(rp * e * 4)?,
            down: dev.alloc_dev(g * h * 4)?,
            inv: dev.alloc_host(g * 4)?,
            wts: dev.alloc_host(g * 4)?,
            moe: dev.alloc_host(m * h * 4)?,
        });
        Ok(())
    }

    /// [`Self::run`] with the router logits and the combine on the chip (see the module docs).
    pub(super) fn run_dev(&mut self, weights: &Qwen3MoeOmegaWeights, attn: &mut dyn PrefillAttention, tokens: &[u32], pos0: usize, all_rows: bool) -> Result<Vec<f32>, String> {
        let c = &weights.config;
        let m = tokens.len();
        if m == 0 {
            return Err("whole-prompt prefill of 0 tokens".into());
        }
        if pos0 + m > u32::MAX as usize {
            return Err(format!("positions {pos0} + {m} do not fit u32"));
        }
        if self.entries.len() != weights.layers.len() {
            return Err(format!("driver built for {} layers, weights have {}", self.entries.len(), weights.layers.len()));
        }
        let (h, qd, qkv_n, e, top_k, i, eps) = (c.hidden, c.q_dim(), c.qkv_n(), c.experts, c.top_k, c.moe_intermediate, c.rms_eps);
        let g = m * top_k;
        let dev_attn = attn.has_dev();
        self.ensure(c, m, dev_attn)?;
        self.dh_ensure(c, m)?;
        let mut st = PrefillStats { tokens: m, layers: weights.layers.len(), attn_dev: dev_attn, ..Default::default() };
        let (dev, b) = (&mut self.dev, self.bufs.as_mut().expect("sized"));
        let dh = self.devhost.as_mut().expect("device host work on");
        let (routers, d) = (&dh.routers, dh.bufs.as_mut().expect("sized"));
        let bm = b.bm;
        let t_run = Instant::now();
        let mut lap = Instant::now();
        macro_rules! lap {
            ($f:ident) => {{
                st.prof.$f += lap.elapsed().as_nanos() as u64;
                lap = Instant::now();
            }};
        }

        let t = Instant::now();
        let mut x: Vec<f32> = tokens.iter().flat_map(|&tok| weights.embed_row(tok)).collect();
        let mut logits = vec![0.0f32; m * e];
        st.host_ns += t.elapsed().as_nanos() as u64;
        lap!(embed_ns);

        let put_tiles = |tiles: &mut D::Buf, max: usize, region: Region, w: &[u64], s: &[u64], counts: &[u32]| -> Result<(u64, usize), String> {
            let tab = build_tiles(bm, w, s, counts).map_err(|err| err.to_string())?;
            if tab.len() > max {
                return Err(format!("tile table of {} entries > region of {max}", tab.len()));
            }
            let off = region as usize * max * TILE_BYTES;
            let bytes = tiles.host_mut().ok_or("tile table is not a host buffer")?;
            for (j, tl) in tab.iter().enumerate() {
                bytes[off + j * TILE_BYTES..off + (j + 1) * TILE_BYTES].copy_from_slice(&tl.to_bytes());
            }
            Ok((tiles.va() + off as u64, tab.len()))
        };
        let dense = [m as u32];
        let mutant = self.mutant;
        let mut tr = self.trace.as_mut();
        if let Some(t) = tr.as_deref_mut() {
            t.begin(m, weights.layers.len(), h, top_k);
        }

        for (l, layer) in weights.layers.iter().enumerate() {
            let ent = &self.entries[l];
            let lu = l as u32;
            // --- input RMSNorm (host) straight into the quantiser's HOST input ---
            let t = Instant::now();
            {
                let xin = host_mut(&mut b.xin, "xin")?;
                for_rows(&mut xin[..m * h], h, &x, h, |o, xr| rmsnorm(xr, &layer.input_norm, eps, o));
                lap!(norm_in_ns);
            }
            st.host_ns += t.elapsed().as_nanos() as u64;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.x_norm = host(&b.xin, "xin")?[..m * h].to_vec();
            }
            st.chip_ns += b.act_in(dev, h, QuantIn::F32, m, b.xin.va()).map_err(|err| format!("layer {l} quant x: {err}"))?;
            st.quant += 1;

            // --- q|k|v, attention, o: exactly as `run` ---
            lap!(dev_ns);
            let (tv, nt) = put_tiles(&mut b.tiles, b.max_tiles, Region::Qkv, &[ent.qkv.0], &[ent.qkv.1], &dense)?;
            lap!(tiles_ns);
            let qkv_y = match (&b.qkv_dev, &b.qkv_host) {
                (Some(d), _) => d.va(),
                (None, Some(hb)) => hb.va(),
                (None, None) => unreachable!("ensure allocates one q|k|v buffer"),
            };
            let call = QgemmCall { n: qkv_n, k: h, layout: Layout::Plain, tiles_va: tv, ntiles: nt, rowidx_va: 0, xq_va: b.xq.va(), xs_va: b.xs.va(), y_va: qkv_y };
            st.chip_ns += b.gemm(dev, &call).map_err(|err| format!("layer {l} q|k|v gemm x{m}: {err}"))?;
            st.qkv += 1;
            let att_va = if let (Some(qd_buf), Some(ad_buf)) = (&b.qkv_dev, &b.att_dev) {
                let ns = attn
                    .attend_dev(lu, pos0 as u32, m as u32, qd_buf.va(), eps, ATTN_OUT_SCALE, ad_buf.va())
                    .ok_or_else(|| format!("layer {l}: attention offered the device entry, then refused it"))?
                    .map_err(|err| format!("layer {l} attention (device) x{m}: {err}"))?;
                st.attn_ns += ns;
                ad_buf.va()
            } else {
                let qkv = host(b.qkv_host.as_ref().expect("host q|k|v"), "q|k|v")?;
                let out = host_mut(&mut b.xin, "xin")?;
                let ns = attn
                    .attend(lu, pos0 as u32, m as u32, &qkv[..m * qkv_n], eps, ATTN_OUT_SCALE, &mut out[..m * qd])
                    .map_err(|err| format!("layer {l} attention x{m}: {err}"))?;
                st.attn_ns += ns;
                b.xin.va()
            };
            st.attn += 1;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.qkv = match (&b.qkv_dev, &b.qkv_host) {
                    (Some(d), _) => read_dev(&mut *dev, d.va(), m * qkv_n)?,
                    (None, Some(hb)) => host(hb, "q|k|v")?[..m * qkv_n].to_vec(),
                    (None, None) => Vec::new(),
                };
                t.ops.attn = if b.att_dev.is_some() { read_dev(&mut *dev, att_va, m * qd)? } else { host(&b.xin, "xin")?[..m * qd].to_vec() };
            }
            st.chip_ns += b.act_in(dev, qd, QuantIn::F32, m, att_va).map_err(|err| format!("layer {l} quant attn: {err}"))?;
            st.quant += 1;
            lap!(dev_ns);
            let (tv, nt) = put_tiles(&mut b.tiles, b.max_tiles, Region::O, &[ent.o.0], &[ent.o.1], &dense)?;
            lap!(tiles_ns);
            let call = QgemmCall { n: h, k: qd, layout: Layout::Plain, tiles_va: tv, ntiles: nt, rowidx_va: 0, xq_va: b.xq.va(), xs_va: b.xs.va(), y_va: b.oproj.va() };
            st.chip_ns += b.gemm(dev, &call).map_err(|err| format!("layer {l} o gemm x{m}: {err}"))?;
            st.o += 1;
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.oproj = host(&b.oproj, "o projection")?[..m * h].to_vec();
            }
            lap!(dev_ns);

            // --- residual + post norm (host) into the padded post buffer ---
            let t = Instant::now();
            {
                let oproj = host(&b.oproj, "o projection")?;
                for_rows(&mut x, h, &oproj[..m * h], h, |xr, pr| {
                    for (xi, v) in xr.iter_mut().zip(pr) {
                        *xi += v;
                    }
                });
            }
            if let Some(t) = tr.as_deref_mut() {
                t.x_mid[l].copy_from_slice(&x);
            }
            lap!(resid_ns);
            {
                let post = host_mut(&mut d.post, "post")?;
                for_rows(&mut post[..m * h], h, &x, h, |o, xr| rmsnorm(xr, &layer.post_norm, eps, o));
            }
            lap!(norm_post_ns);
            st.host_ns += t.elapsed().as_nanos() as u64;

            // --- router logits on the chip (all m tokens, one launch) ---
            st.chip_ns += dev.router_logits(h, e, m, d.post.va(), routers[l].va(), d.logits.va()).map_err(|err| format!("layer {l} router logits x{m}: {err}"))?;
            st.router += 1;
            let t = Instant::now();
            logits.copy_from_slice(&host(&d.logits, "logits")?[..m * e]);
            if mutant == WpMutant::Routing {
                rotate_logits(&mut logits, e);
            }
            if let Some(t) = tr.as_deref_mut().filter(|t| t.ops_layer == Some(l)) {
                t.ops.post = host(&d.post, "post")?[..m * h].to_vec();
                t.ops.logits = logits.clone();
            }
            lap!(router_ns);
            // --- routing plan and sort (host, token-parallel exact replica) ---
            let plan = plan_par(&logits, m, e, top_k).map_err(|err| format!("layer {l} routing: {err}"))?;
            let sort = ExpertSort::from_plan(&plan);
            if let Some(t) = tr.as_deref_mut() {
                t.experts[l].copy_from_slice(&plan.topk_experts);
            }
            as_u32_mut(b.rowidx.host_mut().ok_or("row index is not a host buffer")?)?[..g].copy_from_slice(&sort.rows);
            as_u32_mut(d.inv.host_mut().ok_or("inv is not a host buffer")?)?[..g].copy_from_slice(&sort.inv);
            host_mut(&mut d.wts, "weights")?[..g].copy_from_slice(&plan.topk_weights);
            st.host_ns += t.elapsed().as_nanos() as u64;
            lap!(plan_ns);
            st.chip_ns += b.act_in(dev, h, QuantIn::F32, m, d.post.va()).map_err(|err| format!("layer {l} quant post: {err}"))?;
            st.quant += 1;

            // --- experts: gate|up (rows gathered by token), act quantised, down into DEVICE rows ---
            lap!(dev_ns);
            let (tv, nt) = put_tiles(&mut b.tiles, b.max_tiles, Region::GateUp, &ent.gu_w, &ent.gu_s, &sort.counts)?;
            if mutant == WpMutant::GateUpScales {
                let mut s = ent.gu_s.clone();
                s.rotate_left(1);
                put_tiles(&mut b.tiles, b.max_tiles, Region::GateUp, &ent.gu_w, &s, &sort.counts)?;
            }
            lap!(tiles_ns);
            let call = QgemmCall { n: 2 * i, k: h, layout: Layout::SwigluNat, tiles_va: tv, ntiles: nt, rowidx_va: b.rowidx.va(), xq_va: b.xq.va(), xs_va: b.xs.va(), y_va: b.act.va() };
            st.chip_ns += b.gemm(dev, &call).map_err(|err| format!("layer {l} gate|up gemm x{g}: {err}"))?;
            st.gate_up += 1;
            st.chip_ns += b.act_mid(dev, i, g).map_err(|err| format!("layer {l} quant act: {err}"))?;
            st.quant += 1;
            lap!(dev_ns);
            let (tv, nt) = put_tiles(&mut b.tiles, b.max_tiles, Region::Down, &ent.dn_w, &ent.dn_s, &sort.counts)?;
            lap!(tiles_ns);
            let call = QgemmCall { n: h, k: i, layout: Layout::Plain, tiles_va: tv, ntiles: nt, rowidx_va: 0, xq_va: b.aq.va(), xs_va: b.a_s.va(), y_va: d.down.va() };
            st.chip_ns += b.gemm(dev, &call).map_err(|err| format!("layer {l} down gemm x{g}: {err}"))?;
            st.down += 1;
            lap!(dev_ns);

            // --- combine on the chip, then the residual add on the host ---
            st.chip_ns += dev.moe_combine(h, top_k, m, d.down.va(), d.inv.va(), d.wts.va(), d.moe.va()).map_err(|err| format!("layer {l} combine x{m}: {err}"))?;
            st.combine += 1;
            let t = Instant::now();
            {
                let moe = host(&d.moe, "combine out")?;
                for_rows(&mut x, h, &moe[..m * h], h, |xr, ar| {
                    for (xi, v) in xr.iter_mut().zip(ar) {
                        *xi += v;
                    }
                });
            }
            if let Some(t) = tr.as_deref_mut() {
                t.x_out[l].copy_from_slice(&x);
                if t.ops_layer == Some(l) {
                    t.ops.moe_out = x.iter().zip(&t.x_mid[l]).map(|(a, b)| a - b).collect();
                }
            }
            st.host_ns += t.elapsed().as_nanos() as u64;
            lap!(combine_ns);
        }
        let t = Instant::now();
        let hidden = if all_rows {
            let mut rows = vec![0.0f32; m * h];
            for_rows(&mut rows, h, &x, h, |o, xr| rmsnorm(xr, &weights.final_norm, eps, o));
            rows
        } else {
            let mut last = vec![0.0f32; h];
            rmsnorm(&x[(m - 1) * h..], &weights.final_norm, eps, &mut last);
            last
        };
        st.host_ns += t.elapsed().as_nanos() as u64;
        lap!(final_ns);
        let _ = lap;
        st.prof.wall_ns = t_run.elapsed().as_nanos() as u64;
        self.stats = st;
        Ok(hidden)
    }
}

/// One token's selected experts and weights, in slot order.
type TokenTopK = (Vec<u32>, Vec<f32>);

/// [`MoeBatchPlan::from_logits`] exactly (same fields, same errors), with the per-token top-k
/// done in parallel by partial selection instead of a full sort of every row. The order is the
/// same total order (`total_cmp` descending, then expert index ascending), so the selected
/// experts, their order and the weights (`exp(l - max)` over the selection, summed in slot order)
/// are the same values; the counting sort that follows is the same serial loop.
pub fn plan_par(logits: &[f32], tokens: usize, experts: usize, top_k: usize) -> Result<MoeBatchPlan, MoePlanError> {
    if experts == 0 || top_k == 0 || top_k > experts || tokens > u32::MAX as usize / top_k || logits.len() != tokens.checked_mul(experts).ok_or(MoePlanError::InvalidShape)? {
        return Err(MoePlanError::InvalidShape);
    }
    let assignments = tokens * top_k;
    let one = |token: usize, row: &[f32]| -> Result<TokenTopK, MoePlanError> {
        if let Some(expert) = row.iter().position(|v| !v.is_finite()) {
            return Err(MoePlanError::InvalidLogit { token, expert });
        }
        let cmp = |a: &usize, b: &usize| row[*b].total_cmp(&row[*a]).then(a.cmp(b));
        let mut ranking: Vec<usize> = (0..experts).collect();
        if top_k < experts {
            ranking.select_nth_unstable_by(top_k - 1, cmp);
        }
        ranking.truncate(top_k);
        ranking.sort_unstable_by(cmp);
        let max_logit = row[ranking[0]];
        let probs: Vec<f32> = ranking.iter().map(|&ex| (row[ex] - max_logit).exp()).collect();
        let mut sum = 0.0f32;
        for p in &probs {
            sum += p;
        }
        Ok((ranking.iter().map(|&ex| ex as u32).collect(), probs.iter().map(|p| p / sum).collect()))
    };
    // W-L4c: small batches (the speculative verify) on the calling thread; same values.
    let per_token: Vec<Result<TokenTopK, MoePlanError>> = if tokens <= super::SERIAL_ROWS {
        logits.chunks_exact(experts).enumerate().map(|(t, r)| one(t, r)).collect()
    } else {
        logits.par_chunks_exact(experts).enumerate().map(|(t, r)| one(t, r)).collect()
    };
    let mut topk_experts = Vec::with_capacity(assignments);
    let mut topk_weights = Vec::with_capacity(assignments);
    let mut counts = vec![0u32; experts];
    for r in per_token {
        let (ids, wts) = r?;
        for (&ex, &w) in ids.iter().zip(&wts) {
            topk_experts.push(ex);
            topk_weights.push(w);
            counts[ex as usize] += 1;
        }
    }
    let mut expert_offsets = Vec::with_capacity(experts + 1);
    expert_offsets.push(0u32);
    for count in counts {
        expert_offsets.push(expert_offsets.last().unwrap() + count);
    }
    let mut cursor = expert_offsets[..experts].to_vec();
    let mut grouped_to_assignment = vec![0u32; assignments];
    let mut assignment_to_grouped = vec![0u32; assignments];
    let mut grouped_token_ids = vec![0u32; assignments];
    for assignment in 0..assignments {
        let expert = topk_experts[assignment] as usize;
        let grouped = cursor[expert] as usize;
        cursor[expert] += 1;
        grouped_to_assignment[grouped] = assignment as u32;
        assignment_to_grouped[assignment] = grouped as u32;
        grouped_token_ids[grouped] = (assignment / top_k) as u32;
    }
    Ok(MoeBatchPlan { tokens, experts, top_k, topk_experts, topk_weights, expert_offsets, grouped_to_assignment, assignment_to_grouped, grouped_token_ids })
}
