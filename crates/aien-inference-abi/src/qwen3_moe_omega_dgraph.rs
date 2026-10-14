//! One command list per decode token (OWN-30B L2 cut 2), behind `OWN30B_DECODE_GRAPH=1`.
//!
//! Off (the default) nothing changes: every decode step is [`Qwen3MoeOmegaEngine::step`], about
//! 289 synchronous launches with host work between them. On, a decode step records the whole
//! token into one omega command list (`aien-omega-dgraph`): all 48 layers, the final norm, the
//! bf16 lm_head and the argmax, one submission. The residual row never leaves the chip; the host
//! writes the token's embedding row before and reads the greedy token after.
//!
//! The graph reuses the objects the engine already uploaded (FP8 q|k|v, o, gate|up, down; the
//! device router; the fused-attention KV cache; the lm_head GEMV) by their omega handles. It
//! needs `--weights fp8 --attn fused --router device --lm-head gemv`; any other load with the
//! switch on is refused with a plain message, never silently run on the old path.
//!
//! Prefill stays as it is. Traced steps (the parity harness) and the host routing mutant take
//! the old path: they need the host-side intermediates the graph does not produce.
//!
//! Token choice: with a greedy request through [`Qwen3MoeOmegaEngine::generate`] the graph's own
//! argmax is used (the same rule as `sample_argmax`: lowest index of the max logit). Any other
//! sampler gets the graph's logits read back from the chip (no second lm_head launch).
//!
//! Test hook: `OWN30B_DECODE_GRAPH_MUTANT` (only read when the switch is on) wires the graph
//! wrongly on purpose so a token check must fail: `norm-shift` (layer l gets layer l+1's input
//! norm), `skip-final-norm`, or `1`/`2`/`3` (omega's own kernel mutants).
use super::*;
use aien_omega_dgraph::{DecodeGraph, GraphConfig, GraphStats};

/// The switch: `1` turns the one-list decode on.
pub const ENV: &str = "OWN30B_DECODE_GRAPH";
/// TEST ONLY: deliberate wrong wiring (see the module doc).
pub const MUTANT_ENV: &str = "OWN30B_DECODE_GRAPH_MUTANT";

/// Measurement: `1` (with [`ENV`]`=1`) records a GPU timestamp after every launch of every graph
/// token and sums them per launch kind ([`Timeline`]). Costs a timestamp per launch plus a read.
pub const STAMPS_ENV: &str = "OWN30B_DECODE_GRAPH_STAMPS";

/// Whether the environment asks for the per-launch timeline.
pub fn stamps_requested() -> bool {
    std::env::var(STAMPS_ENV).as_deref() == Ok("1")
}

/// Per-kind GPU time of the graph tokens since the last [`Qwen3MoeOmegaEngine::take_decode_timeline`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Timeline {
    /// Graph tokens summed.
    pub steps: u64,
    /// Launches and GPU ns per kind (`aien_omega_dgraph::KIND_NAMES` order).
    pub launches: [u64; aien_omega_dgraph::KIND_NAMES.len()],
    pub ns: [u64; aien_omega_dgraph::KIND_NAMES.len()],
    /// Sum of each token's first-to-last stamp span, and of the graph's own chip_ns.
    pub span_ns: u64,
    pub chip_ns: u64,
}

/// Whether the environment asks for the one-list decode.
pub fn requested() -> bool {
    std::env::var(ENV).as_deref() == Ok("1")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutant {
    None,
    NormShift,
    SkipFinalNorm,
    Omega(i32),
}

fn mutant() -> Result<Mutant, String> {
    match std::env::var(MUTANT_ENV).as_deref() {
        Err(_) | Ok("") | Ok("0") => Ok(Mutant::None),
        Ok("norm-shift") => Ok(Mutant::NormShift),
        Ok("skip-final-norm") => Ok(Mutant::SkipFinalNorm),
        Ok(v @ ("1" | "2" | "3")) => Ok(Mutant::Omega(v.parse().unwrap_or(0))),
        Ok(v) => Err(format!("{MUTANT_ENV}={v} is not one of norm-shift, skip-final-norm, 1, 2, 3")),
    }
}

/// The engine's graph and the token it produced but nobody consumed yet.
pub(super) struct GraphState {
    graph: DecodeGraph,
    pending: bool,
    pending_token: u32,
    mutant: String,
    stamps: bool,
    timeline: Timeline,
}

impl std::fmt::Debug for GraphState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphState {{ {:?}, pending: {}, mutant: {} }}", self.graph, self.pending, self.mutant)
    }
}

const NEEDS: &str = "OWN30B_DECODE_GRAPH=1 needs the load --weights fp8 --attn fused --router device --lm-head gemv";

/// Every handle the graph takes, or the plain reason this load cannot run it.
fn handles(w: &Qwen3MoeOmegaWeights) -> Result<(GraphConfig, Vec<[*mut std::ffi::c_void; 4]>), String> {
    let c = &w.config;
    if !aien_omega_dgraph::is_native() {
        return Err(format!("{ENV}=1 but this binary has no one-list decode linked (omega before d3f465dc, or a CPU stub build)"));
    }
    if c.hidden != aien_omega_dgraph::HIDDEN {
        return Err(format!("{NEEDS}; and hidden {} (the graph is built for {})", c.hidden, aien_omega_dgraph::HIDDEN));
    }
    let router = match &w.device_router {
        Some(r) => r.omega_raw(),
        None => return Err(format!("{NEEDS}; this load routes on the host (no device router)")),
    };
    let lm_head = match &w.lm_head {
        LmHead::Gemv(g) => g.omega_raw(),
        LmHead::Dense(_) => return Err(format!("{NEEDS}; this load has the dense lm_head")),
    };
    if router.is_null() || lm_head.is_null() {
        return Err(format!("{NEEDS}; the router or lm_head is not an omega object"));
    }
    let mut layers = Vec::with_capacity(w.layers.len());
    for (l, layer) in w.layers.iter().enumerate() {
        let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) = (&layer.attn, &layer.experts) else {
            return Err(format!("{NEEDS}; layer {l} has bf16 attention or experts"));
        };
        let h = [qkv.omega_raw(), o.omega_raw(), gate_up.omega_raw(), down.omega_raw()];
        if h.iter().any(|p| p.is_null()) {
            return Err(format!("{NEEDS}; layer {l} has a projection that is not an omega FP8 tensor"));
        }
        layers.push(h);
    }
    let cfg = GraphConfig {
        layers: c.layers as u32,
        vocab: c.vocab as u32,
        eps: c.rms_eps,
        norm_topk: c.norm_topk_prob as i32,
        router,
        attn: std::ptr::null_mut(),
        lm_head,
    };
    Ok((cfg, layers))
}

/// W-L4f NEGATIVE CONTROL: the act-fp8 numerics class inside the batched verify. Every row's expert input x' (the first
/// HIDDEN of each VERIFY_XR row, x' * 2^64) goes through the prefill quantiser (`fake_quant_row`: e4m3, power-of-two
/// scale per 128 block, so the 2^64 factor passes through exactly) before the experts read it.
unsafe extern "C" fn act_fp8_hook(_ctx: *mut std::ffi::c_void, _layer: u32, rows: u32, xr: *mut f32) {
    use crate::qwen3_moe_omega_prefill::fake_quant::fake_quant_row;
    for r in 0..rows as usize {
        // SAFETY: omega passes rows x VERIFY_XR floats.
        let row = unsafe { std::slice::from_raw_parts_mut(xr.add(r * aien_omega_dgraph::VERIFY_XR), aien_omega_dgraph::HIDDEN) };
        fake_quant_row(row);
    }
}

impl Qwen3MoeOmegaEngine {
    /// Whether decode steps run as one command list (`OWN30B_DECODE_GRAPH=1` at engine creation).
    pub fn decode_graph_on(&self) -> bool {
        self.decode_graph
    }

    /// Turns the one-list decode on or off after creation (OWN-30B RC2: the HTTP door's default is on),
    /// overriding the `OWN30B_DECODE_GRAPH` value read at creation. On builds the graph now, so a load that
    /// cannot run it is refused here with the plain reason (and the switch stays off). Off drops the graph.
    /// Call between generations.
    pub fn set_decode_graph(&mut self, on: bool) -> Result<(), String> {
        self.dgraph_clear();
        if !on {
            self.dgraph = None;
            self.decode_graph = false;
            return Ok(());
        }
        self.decode_graph = true;
        if let Err(e) = self.prepare_decode_graph() {
            self.dgraph = None;
            self.decode_graph = false;
            return Err(e);
        }
        Ok(())
    }

    /// Builds the graph now (it is otherwise built at the first decode step), so a load that
    /// cannot run it fails before any generation. No-op when the switch is off.
    pub fn prepare_decode_graph(&mut self) -> Result<(), String> {
        if !self.decode_graph || self.dgraph.is_some() {
            return Ok(());
        }
        let (mut cfg, layers) = handles(&self.weights)?;
        cfg.attn = match self.kv.dgraph_cache(self.pos) {
            Some(r) => r?,
            None => return Err(format!("{NEEDS}; this load has the {} attention path", self.kv.path_name())),
        };
        if cfg.attn.is_null() {
            return Err(format!("{NEEDS}; the KV cache is not an omega object"));
        }
        let m = mutant()?;
        // SAFETY: every handle is an omega object owned by self.weights / self.kv, which live as
        // long as the engine; `dgraph` is declared before them, so it is dropped first.
        let mut graph = unsafe { DecodeGraph::create(&cfg) }.map_err(|e| format!("decode graph create: {e}"))?;
        let n = layers.len();
        for (l, h) in layers.iter().enumerate() {
            let norm_from = if m == Mutant::NormShift { (l + 1) % n } else { l };
            // SAFETY: as above.
            unsafe { graph.set_layer(l, h[0], h[1], h[2], h[3], &self.weights.layers[norm_from].input_norm) }
                .map_err(|e| format!("decode graph layer {l}: {e}"))?;
        }
        if m != Mutant::SkipFinalNorm {
            graph.set_final_norm(&self.weights.final_norm).map_err(|e| format!("decode graph final norm: {e}"))?;
        }
        if let Mutant::Omega(k) = m {
            graph.test_set_mutant(k);
        }
        let stamps = stamps_requested();
        graph.set_stamps(stamps);
        let mutant = match m {
            Mutant::None => "none".to_string(),
            other => format!("{other:?} (TEST ONLY: wrong on purpose)"),
        };
        self.dgraph = Some(GraphState { graph, pending: false, pending_token: 0, mutant, stamps, timeline: Timeline::default() });
        Ok(())
    }

    /// The per-kind timeline summed since the last call (stamps on), reset to empty.
    pub fn take_decode_timeline(&mut self) -> Option<Timeline> {
        let g = self.dgraph.as_mut()?;
        g.stamps.then(|| std::mem::take(&mut g.timeline))
    }

    /// The deliberate wrong wiring in force ("none" normally), for receipts.
    pub fn decode_graph_mutant(&self) -> &str {
        self.dgraph.as_ref().map_or("none", |g| g.mutant.as_str())
    }

    /// A decode step: the one-list token when the switch is on (and the step is not traced),
    /// else [`Self::step`]. On the graph path the returned hidden state is empty and the token is
    /// held for [`Self::dgraph_take_greedy`] / [`Self::logits`].
    pub(super) fn decode_step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        if !self.decode_graph || !self.trace_layers.is_empty() || self.routing_mutant {
            return self.step(token);
        }
        let t0 = Instant::now();
        let pos = self.pos;
        if pos >= self.kv.capacity() {
            return Err(format!("context full: {pos} of {}", self.kv.capacity()));
        }
        self.prepare_decode_graph()?;
        match self.kv.dgraph_cache(pos) {
            Some(r) => {
                r?;
            }
            None => return Err(format!("{NEEDS}; this load has the {} attention path", self.kv.path_name())),
        }
        let x = self.weights.embed_row(token);
        let g = self.dgraph.as_mut().ok_or("decode graph missing after prepare")?;
        let (next, st): (u32, GraphStats) = g.graph.token(&x, pos).map_err(|e| format!("decode graph token at {pos}: {e}"))?;
        g.pending = true;
        g.pending_token = next;
        // The token ran and wrote this position's KV: book it before reporting a timeline error.
        let mut timeline_err = None;
        if g.stamps {
            match g.graph.stamps() {
                Err(e) => timeline_err = Some(format!("decode graph timeline at {pos}: {e}")),
                Ok((stamps, kinds)) => {
                    let t = &mut g.timeline;
                    t.steps += 1;
                    for (i, &k) in kinds.iter().enumerate() {
                        let k = (k as usize).min(aien_omega_dgraph::KIND_NAMES.len() - 1);
                        t.launches[k] += 1;
                        t.ns[k] += stamps[i + 1].saturating_sub(stamps[i]);
                    }
                    t.span_ns += stamps.last().copied().unwrap_or(0).saturating_sub(stamps.first().copied().unwrap_or(0));
                    t.chip_ns += st.chip_ns;
                }
            }
        }
        self.kv.dgraph_commit(pos);
        self.pos = pos + 1;
        self.history.push(token);
        let total_ns = t0.elapsed().as_nanos() as u64;
        self.last_timing = StepTiming {
            total_ns,
            chip_ns: st.chip_ns,
            launches: st.launches as u64,
            host_ns: total_ns.saturating_sub(st.chip_ns),
            graph_tokens: 1,
            graph_launches: st.launches as u64,
            graph_submits: st.submits as u64,
            graph_record_ns: st.record_ns,
            ..StepTiming::default()
        };
        match timeline_err {
            Some(e) => Err(e),
            None => Ok(Vec::new()),
        }
    }

    /// W-L4f probe: one one-list decode token at `pos` ([`Self::decode_step`]) and its final-normed hidden state read
    /// back from the chip (the reference the batched verify must equal bit for bit), with the graph's own logits when
    /// `logits` (the in-list lm_head). Consumes the pending token. Needs the one-list decode switch on.
    pub fn decode_step_hidden(&mut self, token: u32, logits: bool) -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        if !self.batch_verify_possible() {
            return Err(format!("decode_step_hidden: needs {ENV}=1, no tracing and no routing mutant"));
        }
        self.decode_step(token)?;
        let g = self.dgraph.as_mut().ok_or("decode graph missing after a decode step")?;
        g.pending = false;
        let mut hidden = vec![0.0f32; aien_omega_dgraph::HIDDEN];
        let mut lg = logits.then(|| vec![0.0f32; g.graph.vocab()]);
        g.graph.read(None, Some(&mut hidden), lg.as_deref_mut()).map_err(|e| format!("decode graph hidden: {e}"))?;
        Ok((hidden, lg))
    }

    /// W-L4f: whether [`Self::step_verify_batch`] can run: the one-list decode switch on (the batch verify is that
    /// graph's own kernels), no tracing, no host routing mutant, the graph linked.
    pub fn batch_verify_possible(&self) -> bool {
        self.decode_graph && self.trace_layers.is_empty() && !self.routing_mutant && aien_omega_dgraph::is_native()
    }

    /// W-L4f: the batched exact verify. `tokens` (consecutive positions from `pos`, at most
    /// [`aien_omega_dgraph::VERIFY_MAX_ROWS`]) run through every layer in one omega call (one submission per layer),
    /// and every row's final-normed hidden state comes back. Each row is bit-identical to the one-list decode token
    /// ([`Self::decode_step`], launched or resident) at that position after the rows before it, and so is the KV it
    /// writes (omega `omega_gpu_decode_graph_verify`). Advances `pos` and `history`; the caller rewinds what it
    /// rejects. `mutant`: NEGATIVE CONTROL ONLY; `ActFp8` quantizes every row's expert input (x' after the post
    /// norm) to FP8 with the prefill quantiser inside the call, the rest are refused.
    pub fn step_verify_batch(&mut self, tokens: &[u32], mutant: super::spec::NumericsMutant) -> Result<Vec<Vec<f32>>, String> {
        use super::spec::NumericsMutant;
        let m = tokens.len();
        if m == 0 || m > aien_omega_dgraph::VERIFY_MAX_ROWS {
            return Err(format!("step_verify_batch: {m} tokens (1..={})", aien_omega_dgraph::VERIFY_MAX_ROWS));
        }
        if !self.batch_verify_possible() {
            return Err(format!("step_verify_batch: needs {ENV}=1 with the one-list decode linked, no tracing and no routing mutant"));
        }
        if !matches!(mutant, NumericsMutant::None | NumericsMutant::ActFp8) {
            return Err(format!("step_verify_batch: numerics mutant {mutant:?} is not wired here (none or act-fp8)"));
        }
        let t0 = Instant::now();
        let pos0 = self.pos;
        if pos0 + m > self.kv.capacity() {
            return Err(format!("context full: {pos0} + {m} of {}", self.kv.capacity()));
        }
        self.prepare_decode_graph()?;
        match self.kv.dgraph_cache(pos0) {
            Some(r) => {
                r?;
            }
            None => return Err(format!("{NEEDS}; this load has the {} attention path", self.kv.path_name())),
        }
        let h = self.weights.config.hidden;
        let x: Vec<f32> = tokens.iter().flat_map(|&t| self.weights.embed_row(t)).collect();
        let mut hidden = vec![0.0f32; m * h];
        let g = self.dgraph.as_mut().ok_or("decode graph missing after prepare")?;
        // A pending one-list token belongs to the step before this round; its logits are not these rows'.
        g.pending = false;
        if mutant == NumericsMutant::ActFp8 {
            // SAFETY: act_fp8_hook touches only the rows omega passes; no context.
            unsafe { g.graph.verify_set_hook(Some(act_fp8_hook), std::ptr::null_mut()) };
        }
        let r = g.graph.verify(m, &x, pos0, &mut hidden);
        if mutant == NumericsMutant::ActFp8 {
            // SAFETY: clearing the hook.
            unsafe { g.graph.verify_set_hook(None, std::ptr::null_mut()) };
        }
        let st: GraphStats = r.map_err(|e| format!("decode graph verify at {pos0} ({m} rows): {e}"))?;
        // The call wrote positions pos0 ..= pos0 + m - 1 of every layer.
        self.kv.dgraph_commit(pos0 + m - 1);
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        let total_ns = t0.elapsed().as_nanos() as u64;
        self.last_timing = StepTiming {
            total_ns,
            chip_ns: st.chip_ns,
            launches: st.launches as u64,
            host_ns: total_ns.saturating_sub(st.chip_ns),
            graph_launches: st.launches as u64,
            graph_submits: st.submits as u64,
            graph_record_ns: st.record_ns,
            ..StepTiming::default()
        };
        Ok(hidden.chunks(h).map(<[f32]>::to_vec).collect())
    }

    /// The graph's greedy token of the last decode step, when the caller's sampler is the
    /// argmax (the graph used the same rule). Consumes it.
    pub(super) fn dgraph_take_greedy(&mut self, argmax_sampler: bool) -> Option<u32> {
        let g = self.dgraph.as_mut()?;
        if !argmax_sampler || !g.pending {
            return None;
        }
        g.pending = false;
        Some(g.pending_token)
    }

    /// The graph's logits of the last decode step, read back from the chip (any sampler).
    /// Consumes the pending token. `None`: no graph token is pending.
    pub(super) fn dgraph_take_logits(&mut self) -> Option<Result<Vec<f32>, String>> {
        let g = self.dgraph.as_mut()?;
        if !g.pending {
            return None;
        }
        g.pending = false;
        let mut logits = vec![0.0f32; g.graph.vocab()];
        Some(g.graph.read(None, None, Some(&mut logits)).map(|_| logits).map_err(|e| format!("decode graph logits: {e}")))
    }

    /// Forgets a pending graph token (reset, rewind, a new generation).
    pub(super) fn dgraph_clear(&mut self) {
        if let Some(g) = self.dgraph.as_mut() {
            g.pending = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutant_names_parse_and_unknown_ones_are_refused() {
        // The env is process-wide: this test only reads it through `mutant()` with values it sets.
        std::env::set_var(MUTANT_ENV, "norm-shift");
        assert_eq!(mutant(), Ok(Mutant::NormShift));
        std::env::set_var(MUTANT_ENV, "skip-final-norm");
        assert_eq!(mutant(), Ok(Mutant::SkipFinalNorm));
        std::env::set_var(MUTANT_ENV, "3");
        assert_eq!(mutant(), Ok(Mutant::Omega(3)));
        std::env::set_var(MUTANT_ENV, "eps2");
        assert!(mutant().is_err());
        std::env::remove_var(MUTANT_ENV);
        assert_eq!(mutant(), Ok(Mutant::None));
    }
}
