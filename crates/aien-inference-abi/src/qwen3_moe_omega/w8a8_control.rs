//! W8A8 control for the whole-prompt correctness gate (W-L1G-C, 2026-10-14). New file, a child
//! module of `qwen3_moe_omega` so it can reach the engine's private `weights`, `kv`, `pos`.
//!
//! [`Qwen3MoeOmegaEngine::control_step`] is the OFF path's `step` (FP8 GEMV with f32 activations,
//! omega's fused per-token attention, host routing) restricted to the configuration the ON path
//! replaces (FP8 attention and experts, fused KV), with two additions:
//!  * `ControlMode::W8A8`: every GEMV input is put through the SAME quantiser the ON path's GEMMs
//!    see (`fake_quant::fake_quant_row`: e4m3, power-of-two scale per token per 128 block), at the
//!    same four points: the input-normed row (q|k|v), the attention output (o), the post-normed
//!    row (gate|up) and the SwiGLU activation (down; rounded to bf16 first, as the ON path's GEMM
//!    epilogue writes bf16). Everything else stays the OFF path (f32 GEMV accumulation, the
//!    decode attention's f32 Q and P, host SwiGLU in f32).
//!  * the [`WpTrace`] recorder, filled in the same shape as the ON path fills it.
//!
//! What the control therefore isolates: ON ~ control => the drift is activation rounding;
//! ON differing from the control beyond what the control differs from OFF points at an ON-only
//! ingredient (QMMA accumulation, bf16 Q/P attention, bf16 SwiGLU, or a bug).
use std::time::Instant;

use crate::moe_plan::MoeBatchPlan;
use crate::qwen3_moe_omega::host_attn::HostAttn;
use crate::qwen3_moe_omega::{swiglu, AttnWeights, ExpertWeights, Qwen3MoeOmegaEngine, StepTiming};
use crate::qwen3_moe_omega_prefill::fake_quant::{bf16_round_row, fake_quant_row};
use crate::qwen3_moe_omega_prefill::{rotate_logits, WpMutant, WpTrace};
use crate::tensor::rmsnorm;

/// What the control does to the GEMV inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    /// The OFF path as is (f32 activations).
    Off,
    /// The OFF path with the ON path's activation rounding at every GEMM input.
    W8A8,
}

/// Rounding applied to a GEMV input row (W-L1I drift split).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActRound {
    #[default]
    F32,
    /// e4m3, power-of-two scale per token per 128 block (the W8A8 prefill).
    E4m3,
    /// bf16 round-to-nearest-even (the W8A16 prefill's hconv / hswiglu).
    Bf16,
}

/// GEMM input points, for [`Emul::act_points`].
pub const PT_QKV: u8 = 1;
pub const PT_O: u8 = 2;
pub const PT_GU: u8 = 4;
pub const PT_DOWN: u8 = 8;
pub const PT_ALL: u8 = 15;

/// One-variable-at-a-time emulation of the ON path's roundings in the OFF path (W-L1I).
/// The default is the OFF path itself.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Emul {
    /// Rounding at the GEMV inputs selected by `act_points`.
    pub acts: ActRound,
    /// Bitmask of PT_* points `acts` applies to.
    pub act_points: u8,
    /// Attention on the host emulator (the caller passes the [`HostAttn`]) instead of the chip's
    /// decode kernel; the chip kernel still runs on the same inputs for the floor comparison.
    pub host_attn: bool,
    /// FLOOR CONTROL: flip the last mantissa bit of every embedding element of the first token
    /// (a 6e-8 relative perturbation): how far does a 1-ulp difference travel?
    pub ulp_nudge: bool,
}

impl Emul {
    fn round(&self, pt: u8, row: &mut [f32]) {
        if self.act_points & pt == 0 {
            return;
        }
        match self.acts {
            ActRound::F32 => {}
            ActRound::E4m3 => {
                if pt == PT_DOWN {
                    bf16_round_row(row);
                }
                fake_quant_row(row);
            }
            ActRound::Bf16 => bf16_round_row(row),
        }
    }
}

impl Qwen3MoeOmegaEngine {
    /// Runs `token` at the next position through every layer on the OFF kernels, in `mode`,
    /// recording token row `row` of `trace` (which must have been sized by [`WpTrace::begin`]).
    /// `mutant` applies the routing mutant to the host routing (the GEMM mutant has no OFF-path
    /// meaning and is ignored). Returns the final-normed hidden state.
    pub fn control_step(&mut self, token: u32, mode: ControlMode, mutant: WpMutant, trace: &mut WpTrace, row: usize) -> Result<Vec<f32>, String> {
        let emul = match mode {
            ControlMode::Off => Emul::default(),
            ControlMode::W8A8 => Emul { acts: ActRound::E4m3, act_points: PT_ALL, ..Emul::default() },
        };
        self.control_step_emul(token, &emul, None, mutant, trace, row)
    }

    /// [`Self::control_step`] with the W-L1I emulation switches. `host` must be `Some` when
    /// `emul.host_attn` is set.
    pub fn control_step_emul(&mut self, token: u32, emul: &Emul, mut host: Option<&mut HostAttn>, mutant: WpMutant, trace: &mut WpTrace, row: usize) -> Result<Vec<f32>, String> {
        if emul.host_attn != host.is_some() {
            return Err("control_step_emul: host_attn and the HostAttn argument disagree".into());
        }
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let (h, qd, eps) = (c.hidden, c.q_dim(), c.rms_eps);
        let (e, top_k, i) = (c.experts, c.top_k, c.moe_intermediate);
        let pos = self.pos;
        if pos >= self.kv.capacity() {
            return Err(format!("context full: {pos} of {}", self.kv.capacity()));
        }
        if row >= trace.m || trace.layers() != self.weights.layers.len() {
            return Err(format!("control_step: trace sized for {} rows x {} layers, asked row {row}", trace.m, trace.layers()));
        }
        let mut chip_out = if emul.host_attn { vec![0.0f32; qd] } else { Vec::new() };
        let mut x = self.weights.embed_row(token);
        if emul.ulp_nudge && pos == 0 {
            x.iter_mut().for_each(|v| *v = f32::from_bits(v.to_bits() ^ 1));
        }
        let mut x_norm = vec![0.0f32; h];
        let mut qkv_out = vec![0.0f32; c.qkv_n()];
        let mut attn_out = vec![0.0f32; qd];
        let mut attn_proj = vec![0.0f32; h];
        let mut post = vec![0.0f32; h];
        let mut logits = vec![0.0f32; e];
        let mut gu_out = vec![0.0f32; top_k * 2 * i];
        let mut moe_out = vec![0.0f32; h];
        let mut act_all = vec![0.0f32; top_k * i];
        let mut down_all = vec![0.0f32; top_k * h];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let ops = trace.ops_layer == Some(layer_idx);
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) = (&layer.attn, &layer.experts) else {
                return Err(format!("layer {layer_idx}: the control needs FP8 attention and experts"));
            };

            // --- attention block ---
            rmsnorm(&x, &layer.input_norm, eps, &mut x_norm);
            if ops {
                put(&mut trace.ops.x_norm, trace.m, row, &x_norm);
            }
            emul.round(PT_QKV, &mut x_norm);
            timing.chip_ns += qkv.grouped_f32(&[0], &x_norm, &mut qkv_out).map_err(|e| format!("layer {layer_idx} qkv: {e}"))?;
            if ops {
                put(&mut trace.ops.qkv, trace.m, row, &qkv_out);
            }
            let chip_dst = if emul.host_attn { &mut chip_out } else { &mut attn_out };
            match self.kv.fused_attend(layer_idx, pos, &qkv_out, eps, chip_dst) {
                Some(r) => timing.chip_ns += r?,
                None => return Err("the control needs the fused KV attention".into()),
            }
            if let Some(ha) = host.as_deref_mut() {
                ha.attend(layer_idx, pos, &qkv_out, &layer.q_norm, &layer.k_norm, eps, &mut attn_out).map_err(|e| format!("layer {layer_idx}: {e}"))?;
                ha.record_chip(layer_idx, &attn_out, &chip_out);
            }
            if ops {
                put(&mut trace.ops.attn, trace.m, row, &attn_out);
            }
            emul.round(PT_O, &mut attn_out);
            timing.chip_ns += o.grouped_f32(&[0], &attn_out, &mut attn_proj).map_err(|e| format!("layer {layer_idx} o: {e}"))?;
            if ops {
                put(&mut trace.ops.oproj, trace.m, row, &attn_proj);
            }
            for (xi, p) in x.iter_mut().zip(&attn_proj) {
                *xi += p;
            }
            trace.x_mid[layer_idx][row * h..(row + 1) * h].copy_from_slice(&x);

            // --- routing (host), experts ---
            rmsnorm(&x, &layer.post_norm, eps, &mut post);
            for (ex, l) in logits.iter_mut().enumerate() {
                *l = layer.router[ex * h..(ex + 1) * h].iter().zip(&post).map(|(a, b)| a * b).sum();
            }
            if mutant == WpMutant::Routing {
                rotate_logits(&mut logits, e);
            }
            if ops {
                put(&mut trace.ops.post, trace.m, row, &post);
                put(&mut trace.ops.logits, trace.m, row, &logits);
            }
            let plan = MoeBatchPlan::from_logits(&logits, 1, e, top_k).map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
            let experts: Vec<u32> = plan.topk_experts.clone();
            trace.experts[layer_idx][row * top_k..(row + 1) * top_k].copy_from_slice(&experts);
            emul.round(PT_GU, &mut post);
            timing.chip_ns += gate_up.grouped_f32(&experts, &post, &mut gu_out).map_err(|e| format!("layer {layer_idx} gate_up fp8: {e}"))?;
            for slot in 0..top_k {
                let act = &mut act_all[slot * i..(slot + 1) * i];
                swiglu(&gu_out[slot * 2 * i..(slot + 1) * 2 * i], i, act);
                emul.round(PT_DOWN, act);
            }
            timing.chip_ns += down.grouped_f32(&experts, &act_all, &mut down_all).map_err(|e| format!("layer {layer_idx} down fp8: {e}"))?;
            moe_out.iter_mut().for_each(|v| *v = 0.0);
            for (slot, &w) in plan.topk_weights.iter().enumerate() {
                for (m, d) in moe_out.iter_mut().zip(&down_all[slot * h..(slot + 1) * h]) {
                    *m += w * d;
                }
            }
            if ops {
                put(&mut trace.ops.moe_out, trace.m, row, &moe_out);
            }
            for (xi, m) in x.iter_mut().zip(&moe_out) {
                *xi += m;
            }
            trace.x_out[layer_idx][row * h..(row + 1) * h].copy_from_slice(&x);
        }
        let mut hidden = vec![0.0f32; h];
        rmsnorm(&x, &self.weights.final_norm, eps, &mut hidden);
        self.pos = pos + 1;
        self.history.push(token);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hidden)
    }
}

/// Stores `v` as row `row` of an `m x v.len()` op store (sized on first use).
fn put(store: &mut Vec<f32>, m: usize, row: usize, v: &[f32]) {
    if store.len() != m * v.len() {
        *store = vec![0.0f32; m * v.len()];
    }
    store[row * v.len()..(row + 1) * v.len()].copy_from_slice(v);
}
