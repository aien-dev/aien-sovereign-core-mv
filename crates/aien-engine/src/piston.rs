//! The power engine's host piston and the crankshaft that gears both engines to one beat.
//!
//! The host piston is the exact per-expert reference forward (`reference_moe_forward_experts`, the same
//! arithmetic as the resident reference; L190 proved it bit-equal 64/64 through the ring). It fires on the leases
//! of the buffer the schedule names for the beat and never asks the ring for anything else: an expert the model
//! routes to that is not in the buffer is a misfire and fails the run closed.
use crate::movement::FillStamp;
use crate::plan::RouteTrace;
use crate::schedule::{Fill, Schedule};
use crate::timing::BeatTrace;
use crate::{BLOCK, HIDDEN, INTER};
use aien_heartbeat::ring::Ring;
use aien_inference_abi::moe_plan::MoeBatchPlan;
use aien_inference_abi::qwen3_coder::{compute_logits, forward_token_with, load_qwen3_coder_dense, Qwen3CoderState, Qwen3CoderWeights};
use aien_inference_abi::qwen3_moe::{reference_moe_forward_experts, reference_router_logits_bf16, ExpertView, OracleActivations};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

/// The crankshaft: issues fills to the movement engine in schedule order and hands the power engine the stamp
/// of the fill that feeds the beat it is about to fire.
pub struct Crank {
    pub sched: Arc<Schedule>,
    pub ring: Arc<Ring>,
    jobs: Sender<Fill>,
    stamps: Receiver<Result<FillStamp, String>>,
    pending: HashMap<u32, FillStamp>,
    pub t0: Instant,
}

impl Crank {
    pub fn new(sched: Arc<Schedule>, ring: Arc<Ring>, jobs: Sender<Fill>, stamps: Receiver<Result<FillStamp, String>>, t0: Instant) -> Self {
        Self { sched, ring, jobs, stamps, pending: HashMap::new(), t0 }
    }

    pub fn now_ns(&self) -> i64 {
        self.t0.elapsed().as_nanos() as i64
    }

    /// A second hand on the job channel (reservoir coupling: an issuer thread feeds the movement engines).
    pub fn issuer(&self) -> Sender<Fill> {
        self.jobs.clone()
    }

    pub fn issue(&self, f: &Fill) -> Result<(), String> {
        self.jobs.send(f.clone()).map_err(|_| "movement engine gone".to_string())
    }

    /// Prime: issue the fills that must land before beat 0 (the schedule's prime list).
    pub fn prime(&self) -> Result<(), String> {
        for f in &self.sched.prime.clone() {
            self.issue(f)?;
        }
        Ok(())
    }

    /// Block until the fill for `beat` is published. Stamps arrive in issue order.
    pub fn wait_fill(&mut self, beat: u32) -> Result<FillStamp, String> {
        loop {
            if let Some(s) = self.pending.remove(&beat) {
                return Ok(s);
            }
            let s = self.stamps.recv().map_err(|_| "movement engine stopped before the beat was fed".to_string())??;
            self.pending.insert(s.for_beat, s);
        }
    }
}

struct Owned {
    gate_up: Vec<u8>,
    gate_up_scales: Vec<u16>,
    down: Vec<u8>,
    down_scales: Vec<u16>,
}

fn words(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// Copy one tensor out of a published slot (the lease's segments concatenated are the expert's payload).
fn tensor_from_lease(ring: &Ring, sched: &Schedule, part: u32, name: &str, dst: &mut [u8]) -> Result<(), String> {
    let p = sched.part(part);
    let lease = ring.lease(p.layer, p.expert).map_err(|e| e.to_string())?.ok_or_else(|| format!("layer {} expert {}: not in any buffer (misfire)", p.layer, p.expert))?;
    let (start, len) = (p.tensor_offset(name)?, dst.len() as u64);
    let end = start + len;
    let (mut pos, mut out) = (0u64, 0usize);
    for seg in lease.segments() {
        let (s0, s1) = (pos, pos + seg.len() as u64);
        let (a, b) = (start.max(s0), end.min(s1));
        if a < b {
            let n = (b - a) as usize;
            dst[out..out + n].copy_from_slice(&seg[(a - s0) as usize..(b - s0) as usize]);
            out += n;
        }
        pos = s1;
    }
    if out != dst.len() {
        return Err(format!("{name}: copied {out} of {} bytes", dst.len()));
    }
    Ok(())
}

fn owned_expert(ring: &Ring, sched: &Schedule, part: u32, layer: usize, e: usize) -> Result<Owned, String> {
    let p = format!("model.layers.{layer}.mlp.experts.{e}");
    let (h, i) = (HIDDEN, INTER);
    let mut gate_up = vec![0u8; 2 * i * h];
    let (g, u) = gate_up.split_at_mut(i * h);
    tensor_from_lease(ring, sched, part, &format!("{p}.gate_proj.weight"), g)?;
    tensor_from_lease(ring, sched, part, &format!("{p}.up_proj.weight"), u)?;
    let mut sc = vec![0u8; 2 * (i / BLOCK) * (h / BLOCK) * 2];
    let (g, u) = sc.split_at_mut((i / BLOCK) * (h / BLOCK) * 2);
    tensor_from_lease(ring, sched, part, &format!("{p}.gate_proj.weight_scale_inv"), g)?;
    tensor_from_lease(ring, sched, part, &format!("{p}.up_proj.weight_scale_inv"), u)?;
    let mut down = vec![0u8; h * i];
    tensor_from_lease(ring, sched, part, &format!("{p}.down_proj.weight"), &mut down)?;
    let mut ds = vec![0u8; (h / BLOCK) * (i / BLOCK) * 2];
    tensor_from_lease(ring, sched, part, &format!("{p}.down_proj.weight_scale_inv"), &mut ds)?;
    Ok(Owned { gate_up, gate_up_scales: words(&sc), down, down_scales: words(&ds) })
}

pub struct HostPiston {
    w: Qwen3CoderWeights,
    routers: Vec<Vec<u16>>,
    st: Qwen3CoderState,
    pub pos: usize,
}

impl HostPiston {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let (w, routers) = load_qwen3_coder_dense(model_dir).map_err(|e| format!("{e:?}"))?;
        Ok(Self { w, routers, st: Qwen3CoderState::new(), pos: 0 })
    }

    /// One token = one pass over the schedule's layers, one beat per layer starting at `beat0`.
    /// Returns the logits and the beat traces. `routes_out` receives the routed experts per layer when given.
    pub fn run_token(&mut self, crank: &mut Crank, token: u32, beat0: u32, mut routes_out: Option<&mut Vec<Vec<u16>>>) -> Result<(Vec<f32>, Vec<BeatTrace>), String> {
        let sched = crank.sched.clone();
        let ring = crank.ring.clone();
        let mut traces = Vec::with_capacity(sched.layers.len());
        let mut err: Option<String> = None;
        let routers = &self.routers;
        let h = forward_token_with(&self.w, token, self.pos, &mut self.st, &mut |layer, post| {
            if err.is_some() {
                return vec![0.0; HIDDEN];
            }
            let b = beat0 + layer as u32;
            let beat = &sched.beats[b as usize];
            debug_assert_eq!(beat.layer as usize, layer);
            // 1. intake: put the next fill in flight before this beat fires
            if let Some(f) = &beat.fill {
                if let Err(e) = crank.issue(f) {
                    err = Some(e);
                    return vec![0.0; HIDDEN];
                }
            }
            // 2. wait for this beat's fuel, then fire
            let stamp = match crank.wait_fill(b) {
                Ok(s) => s,
                Err(e) => {
                    err = Some(format!("beat {b}: {e}"));
                    return vec![0.0; HIDDEN];
                }
            };
            let fire_start = crank.now_ns();
            let logits = reference_router_logits_bf16(&routers[layer], HIDDEN, post, 1);
            let plan = match MoeBatchPlan::qwen3_coder_a3b(&logits, 1) {
                Ok(p) => p,
                Err(e) => {
                    err = Some(format!("beat {b}: plan {e:?}"));
                    return vec![0.0; HIDDEN];
                }
            };
            let routed: Vec<u16> = plan.topk_experts[..plan.top_k].iter().map(|&x| x as u16).collect();
            if let Some(r) = routes_out.as_deref_mut() {
                r.push(routed.clone());
            }
            let mut owned: Vec<(usize, Owned)> = Vec::with_capacity(plan.top_k);
            let mut misfire = false;
            for &e in &routed {
                let part = beat.fire_parts.iter().copied().find(|&p| sched.part(p).expert == e);
                let Some(part) = part else {
                    misfire = true;
                    err = Some(format!("beat {b} (token {} layer {layer}): routed expert {e} is not in buffer {} (misfire; the schedule does not cover this routing)", beat.token, beat.fire_buffer));
                    break;
                };
                match owned_expert(&ring, &sched, part, layer, e as usize) {
                    Ok(o) => owned.push((e as usize, o)),
                    Err(m) => {
                        misfire = true;
                        err = Some(format!("beat {b}: {m}"));
                        break;
                    }
                }
            }
            let out = if misfire {
                vec![0.0; HIDDEN]
            } else {
                let view = |e: usize| {
                    let o = &owned.iter().find(|(x, _)| *x == e).expect("routed expert fetched").1;
                    ExpertView { gate_up: &o.gate_up, gate_up_scales: &o.gate_up_scales, down: &o.down, down_scales: &o.down_scales }
                };
                reference_moe_forward_experts(HIDDEN, INTER, post, &plan, OracleActivations::Fp32, &view)
            };
            let fire_done = crank.now_ns();
            traces.push(BeatTrace {
                b,
                token: beat.token,
                layer: layer as u16,
                buffer: beat.fire_buffer,
                bytes: stamp.bytes,
                fill_issue_ns: stamp.issue_ns,
                fill_done_ns: stamp.done_ns,
                fire_start_ns: fire_start,
                fire_done_ns: fire_done,
                misfire,
            });
            out
        });
        if let Some(e) = err {
            return Err(e);
        }
        let logits = compute_logits(&self.w, &h);
        self.pos += 1;
        Ok((logits, traces))
    }
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best as u32
}

/// Identity gate: the loop's argmax at every position equals the trace's; counts agreements.
pub fn gate_identical(trace: &RouteTrace, got: &[u32]) -> (usize, usize) {
    let n = trace.argmax.len().min(got.len());
    (trace.argmax.iter().zip(got).filter(|(a, b)| a == b).count(), n)
}

/// H6, the pump stroke: N sequences (heads) on one weight read per beat. Every beat's expert set is copied out of the
/// cylinder ONCE and applied to all N rows; the model's own routing per row is untouched, so each head's tokens must
/// equal its own N = 1 run (the gate). Same statements as `forward_token_with` per row (`forward_layer_attn` is that
/// function's layer body), so N = 1 is bit-identical to the single piston by construction.
pub struct HostPistonBank {
    w: Qwen3CoderWeights,
    routers: Vec<Vec<u16>>,
    pub states: Vec<Qwen3CoderState>,
    pub pos: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct StrokeBeat {
    pub b: u32,
    pub layer: u16,
    pub heads: usize,
    /// Distinct routed experts across the N heads this beat (the expert union).
    pub union: usize,
    /// Bytes copied out of the cylinder for the union (each expert once).
    pub union_bytes: u64,
    pub attn_ns: i64,
    pub moe_ns: i64,
}

impl HostPistonBank {
    pub fn open(model_dir: &Path, n: usize) -> Result<Self, String> {
        let (w, routers) = load_qwen3_coder_dense(model_dir).map_err(|e| format!("{e:?}"))?;
        Ok(Self { w, routers, states: (0..n).map(|_| Qwen3CoderState::new()).collect(), pos: vec![0; n] })
    }

    /// One stroke: token `tokens[s]` for head `s`, all layers, one beat per layer from `beat0`. Returns the logits
    /// per head, the beat traces (timing) and the stroke beats (union per beat).
    pub fn run_stroke(&mut self, crank: &mut Crank, tokens: &[u32], beat0: u32, mut routes_out: Option<&mut Vec<Vec<Vec<u16>>>>) -> Result<(Vec<Vec<f32>>, Vec<BeatTrace>, Vec<StrokeBeat>), String> {
        use aien_inference_abi::qwen3_coder::{embed_token, final_hidden, forward_layer_attn};
        let n = tokens.len();
        if n == 0 || n > self.states.len() {
            return Err(format!("stroke of {n} heads on a bank of {}", self.states.len()));
        }
        let sched = crank.sched.clone();
        let ring = crank.ring.clone();
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| embed_token(&self.w, t)).collect();
        let mut traces = Vec::with_capacity(sched.layers.len());
        let mut strokes = Vec::with_capacity(sched.layers.len());
        let mut routes_this: Vec<Vec<Vec<u16>>> = vec![Vec::new(); n];
        for layer in 0..sched.layers.len() {
            let b = beat0 + layer as u32;
            let beat = &sched.beats[b as usize];
            if beat.layer as usize != layer {
                return Err(format!("beat {b}: schedule layer {} but the piston is at layer {layer}", beat.layer));
            }
            if let Some(f) = &beat.fill {
                crank.issue(f)?;
            }
            let stamp = crank.wait_fill(b)?;
            let fire_start = crank.now_ns();
            let mut posts = Vec::with_capacity(n * HIDDEN);
            for s in 0..n {
                let post = forward_layer_attn(&self.w, layer, &mut xs[s], self.pos[s], &mut self.states[s].layers[layer]);
                posts.extend_from_slice(&post);
            }
            let attn_done = crank.now_ns();
            let logits = reference_router_logits_bf16(&self.routers[layer], HIDDEN, &posts, n);
            let plan = MoeBatchPlan::qwen3_coder_a3b(&logits, n).map_err(|e| format!("beat {b}: plan {e:?}"))?;
            let mut union: Vec<u16> = Vec::new();
            for s in 0..n {
                let routed: Vec<u16> = plan.topk_experts[s * plan.top_k..(s + 1) * plan.top_k].iter().map(|&x| x as u16).collect();
                for &e in &routed {
                    if !union.contains(&e) {
                        union.push(e);
                    }
                }
                routes_this[s].push(routed);
            }
            let mut owned: Vec<(usize, Owned)> = Vec::with_capacity(union.len());
            let mut union_bytes = 0u64;
            for &e in &union {
                let part = beat.fire_parts.iter().copied().find(|&p| sched.part(p).expert == e).ok_or_else(|| format!("beat {b} (stroke {} layer {layer}): routed expert {e} is not in buffer {} (misfire; the schedule does not cover this routing)", beat.token, beat.fire_buffer))?;
                union_bytes += sched.part(part).bytes;
                owned.push((e as usize, owned_expert(&ring, &sched, part, layer, e as usize).map_err(|m| format!("beat {b}: {m}"))?));
            }
            let view = |e: usize| {
                let o = &owned.iter().find(|(x, _)| *x == e).expect("routed expert fetched").1;
                ExpertView { gate_up: &o.gate_up, gate_up_scales: &o.gate_up_scales, down: &o.down, down_scales: &o.down_scales }
            };
            let out = reference_moe_forward_experts(HIDDEN, INTER, &posts, &plan, OracleActivations::Fp32, &view);
            if out.len() != n * HIDDEN {
                return Err(format!("beat {b}: moe output {} for {n} heads", out.len()));
            }
            for s in 0..n {
                for i in 0..HIDDEN {
                    xs[s][i] += out[s * HIDDEN + i];
                }
            }
            let fire_done = crank.now_ns();
            traces.push(BeatTrace { b, token: beat.token, layer: layer as u16, buffer: beat.fire_buffer, bytes: stamp.bytes, fill_issue_ns: stamp.issue_ns, fill_done_ns: stamp.done_ns, fire_start_ns: fire_start, fire_done_ns: fire_done, misfire: false });
            strokes.push(StrokeBeat { b, layer: layer as u16, heads: n, union: union.len(), union_bytes, attn_ns: attn_done - fire_start, moe_ns: fire_done - attn_done });
        }
        let logits: Vec<Vec<f32>> = xs.iter().map(|x| compute_logits(&self.w, &final_hidden(&self.w, x))).collect();
        for s in 0..n {
            self.pos[s] += 1;
        }
        if let Some(r) = routes_out.as_deref_mut() {
            *r = routes_this;
        }
        Ok((logits, traces, strokes))
    }
}
