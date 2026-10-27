// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

impl Qwen3MoeOmegaEngine {
    /// Runs `token` at the next position through every layer on the OFF kernels, in `mode`,
    /// recording token row `row` of `trace` (which must have been sized by [`WpTrace::begin`]).
    /// `mutant` applies the routing mutant to the host routing (the GEMM mutant has no OFF-path
    /// meaning and is ignored). Returns the final-normed hidden state.
    pub fn control_step(
        &mut self,
        token: u32,
        mode: ControlMode,
        mutant: WpMutant,
        trace: &mut WpTrace,
        row: usize,
    ) -> Result<Vec<f32>, String> {
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
            return Err(format!(
                "control_step: trace sized for {} rows x {} layers, asked row {row}",
                trace.m,
                trace.layers()
            ));
        }
        let fq = mode == ControlMode::W8A8;
        let mut x = self.weights.embed_row(token);
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
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) =
                (&layer.attn, &layer.experts)
            else {
                return Err(format!(
                    "layer {layer_idx}: the control needs FP8 attention and experts"
                ));
            };

            // --- attention block ---
            rmsnorm(&x, &layer.input_norm, eps, &mut x_norm);
            if ops {
                put(&mut trace.ops.x_norm, trace.m, row, &x_norm);
            }
            if fq {
                fake_quant_row(&mut x_norm);
            }
            timing.chip_ns += qkv
                .grouped_f32(&[0], &x_norm, &mut qkv_out)
                .map_err(|e| format!("layer {layer_idx} qkv: {e}"))?;
            if ops {
                put(&mut trace.ops.qkv, trace.m, row, &qkv_out);
            }
            match self
                .kv
                .fused_attend(layer_idx, pos, &qkv_out, eps, &mut attn_out)
            {
                Some(r) => timing.chip_ns += r?,
                None => return Err("the control needs the fused KV attention".into()),
            }
            if ops {
                put(&mut trace.ops.attn, trace.m, row, &attn_out);
            }
            if fq {
                fake_quant_row(&mut attn_out);
            }
            timing.chip_ns += o
                .grouped_f32(&[0], &attn_out, &mut attn_proj)
                .map_err(|e| format!("layer {layer_idx} o: {e}"))?;
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
                *l = layer.router[ex * h..(ex + 1) * h]
                    .iter()
                    .zip(&post)
                    .map(|(a, b)| a * b)
                    .sum();
            }
            if mutant == WpMutant::Routing {
                rotate_logits(&mut logits, e);
            }
            if ops {
                put(&mut trace.ops.post, trace.m, row, &post);
                put(&mut trace.ops.logits, trace.m, row, &logits);
            }
            let plan = MoeBatchPlan::from_logits(&logits, 1, e, top_k)
                .map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
            let experts: Vec<u32> = plan.topk_experts.clone();
            trace.experts[layer_idx][row * top_k..(row + 1) * top_k].copy_from_slice(&experts);
            if fq {
                fake_quant_row(&mut post);
            }
            timing.chip_ns += gate_up
                .grouped_f32(&experts, &post, &mut gu_out)
                .map_err(|e| format!("layer {layer_idx} gate_up fp8: {e}"))?;
            for slot in 0..top_k {
                let act = &mut act_all[slot * i..(slot + 1) * i];
                swiglu(&gu_out[slot * 2 * i..(slot + 1) * 2 * i], i, act);
                if fq {
                    bf16_round_row(act);
                    fake_quant_row(act);
                }
            }
            timing.chip_ns += down
                .grouped_f32(&experts, &act_all, &mut down_all)
                .map_err(|e| format!("layer {layer_idx} down fp8: {e}"))?;
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
