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

//! Branch-agnostic MoE routing plan for Qwen3-Coder-30B-A3B.
//!
//! This is the host-side correctness oracle and batch metadata contract for
//! the routed expert layer (`qwen3_moe` CPU oracle, Omega GB10 path). Expert GEMMs consume assignments in grouped order;
//! the inverse map restores each token's original top-k slots.

use std::fmt;

pub const QWEN3_CODER_A3B_EXPERTS: usize = 128;
pub const QWEN3_CODER_A3B_TOP_K: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoePlanError {
    InvalidShape,
    InvalidLogit { token: usize, expert: usize },
    InvalidExpertOutput,
}

impl fmt::Display for MoePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(f, "invalid MoE routing shape"),
            Self::InvalidLogit { token, expert } => {
                write!(
                    f,
                    "non-finite router logit at token {token}, expert {expert}"
                )
            }
            Self::InvalidExpertOutput => write!(f, "invalid grouped expert output shape"),
        }
    }
}

impl std::error::Error for MoePlanError {}

/// Assignment offsets and maps use u32, the index width of the device routing ABI.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeBatchPlan {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    /// Expert ID for each token-major assignment, length `tokens * top_k`.
    pub topk_experts: Vec<u32>,
    /// Normalized probability for each token-major assignment.
    pub topk_weights: Vec<f32>,
    /// Prefix sum of assignment counts by expert, length `experts + 1`.
    pub expert_offsets: Vec<u32>,
    /// Token-major assignment index at each grouped position.
    pub grouped_to_assignment: Vec<u32>,
    /// Grouped position of each token-major assignment.
    pub assignment_to_grouped: Vec<u32>,
    /// Token ID at each grouped position for gathering activations.
    pub grouped_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeRoutingTelemetry {
    pub tokens_per_expert: Vec<u32>,
    pub experts_touched: usize,
    pub expert_batch_p50: u32,
    pub expert_batch_p95: u32,
    pub max_expert_batch: u32,
    /// Shannon entropy of expert assignment frequency, in bits.
    pub routing_entropy_bits: f64,
}

impl MoeBatchPlan {
    /// Softmax in FP32, top-k, then renormalize selected probabilities.
    /// This matches Qwen3-MoE's `norm_topk_prob=true` routing semantics.
    /// Ties use the lower expert ID so the host plan is deterministic.
    pub fn from_logits(
        logits: &[f32],
        tokens: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<Self, MoePlanError> {
        if experts == 0
            || top_k == 0
            || top_k > experts
            || tokens > u32::MAX as usize / top_k
            || logits.len()
                != tokens
                    .checked_mul(experts)
                    .ok_or(MoePlanError::InvalidShape)?
        {
            return Err(MoePlanError::InvalidShape);
        }

        let assignments = tokens * top_k;
        let mut topk_experts = Vec::with_capacity(assignments);
        let mut topk_weights = Vec::with_capacity(assignments);
        let mut counts = vec![0u32; experts];
        let mut ranking: Vec<usize> = (0..experts).collect();

        for (token, row) in logits.chunks_exact(experts).enumerate() {
            for (expert, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(MoePlanError::InvalidLogit { token, expert });
                }
            }
            ranking.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let max_logit = row[ranking[0]];
            // The common full-softmax denominator cancels during top-k
            // renormalization, so only the selected exponentials are needed.
            let mut selected = [0.0f32; QWEN3_CODER_A3B_TOP_K];
            let mut sum = 0.0f32;
            for slot in 0..top_k {
                let probability = (row[ranking[slot]] - max_logit).exp();
                if top_k <= selected.len() {
                    selected[slot] = probability;
                }
                sum += probability;
            }
            for slot in 0..top_k {
                let expert = ranking[slot];
                topk_experts.push(expert as u32);
                let probability = if top_k <= selected.len() {
                    selected[slot]
                } else {
                    (row[expert] - max_logit).exp()
                };
                topk_weights.push(probability / sum);
                counts[expert] += 1;
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

        Ok(Self {
            tokens,
            experts,
            top_k,
            topk_experts,
            topk_weights,
            expert_offsets,
            grouped_to_assignment,
            assignment_to_grouped,
            grouped_token_ids,
        })
    }

    pub fn qwen3_coder_a3b(logits: &[f32], tokens: usize) -> Result<Self, MoePlanError> {
        Self::from_logits(
            logits,
            tokens,
            QWEN3_CODER_A3B_EXPERTS,
            QWEN3_CODER_A3B_TOP_K,
        )
    }

    pub fn telemetry(&self) -> MoeRoutingTelemetry {
        let tokens_per_expert: Vec<u32> = self
            .expert_offsets
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();
        let mut active: Vec<u32> = tokens_per_expert
            .iter()
            .copied()
            .filter(|&count| count > 0)
            .collect();
        active.sort_unstable();
        let total = (self.tokens * self.top_k) as f64;
        let entropy = if total == 0.0 {
            0.0
        } else {
            active.iter().fold(0.0, |acc, count| {
                let p = *count as f64 / total;
                acc - p * p.log2()
            })
        };
        let percentile = |p: usize| -> u32 {
            if active.is_empty() {
                0
            } else {
                active[((active.len() - 1) * p).div_ceil(100)]
            }
        };
        MoeRoutingTelemetry {
            tokens_per_expert,
            experts_touched: active.len(),
            expert_batch_p50: percentile(50),
            expert_batch_p95: percentile(95),
            max_expert_batch: active.last().copied().unwrap_or(0),
            routing_entropy_bits: entropy,
        }
    }

    /// Reference weighted inverse permutation for device parity checks.
    pub fn reduce_grouped(
        &self,
        grouped_output: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, MoePlanError> {
        if hidden == 0 || grouped_output.len() != self.tokens * self.top_k * hidden {
            return Err(MoePlanError::InvalidExpertOutput);
        }
        let mut output = vec![0.0f32; self.tokens * hidden];
        for assignment in 0..self.tokens * self.top_k {
            let grouped = self.assignment_to_grouped[assignment] as usize;
            let token = assignment / self.top_k;
            let weight = self.topk_weights[assignment];
            for channel in 0..hidden {
                output[token * hidden + channel] +=
                    grouped_output[grouped * hidden + channel] * weight;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_routing_groups_and_restores_all_assignments() {
        let mut logits = vec![-20.0f32; 3 * QWEN3_CODER_A3B_EXPERTS];
        for token in 0..3 {
            for slot in 0..QWEN3_CODER_A3B_TOP_K {
                logits[token * QWEN3_CODER_A3B_EXPERTS + (token * 11 + slot)] = (slot + 1) as f32;
            }
        }
        let plan = MoeBatchPlan::qwen3_coder_a3b(&logits, 3).unwrap();
        assert_eq!(plan.expert_offsets[QWEN3_CODER_A3B_EXPERTS], 24);
        for assignment in 0..24 {
            let grouped = plan.assignment_to_grouped[assignment] as usize;
            assert_eq!(plan.grouped_to_assignment[grouped], assignment as u32);
            assert_eq!(plan.grouped_token_ids[grouped], (assignment / 8) as u32);
            let expert = plan.topk_experts[assignment] as usize;
            assert!(grouped >= plan.expert_offsets[expert] as usize);
            assert!(grouped < plan.expert_offsets[expert + 1] as usize);
        }
        for weights in plan.topk_weights.chunks_exact(8) {
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn weighted_inverse_permutation_preserves_token_values() {
        let plan = MoeBatchPlan::from_logits(&[4.0, 2.0, 1.0, 2.0, 4.0, 1.0], 2, 3, 2).unwrap();
        let mut grouped = vec![0.0; 4 * 2];
        for (position, token) in plan.grouped_token_ids.iter().enumerate() {
            grouped[position * 2] = (*token + 1) as f32;
            grouped[position * 2 + 1] = (*token + 1) as f32 * 3.0;
        }
        let output = plan.reduce_grouped(&grouped, 2).unwrap();
        for (actual, expected) in output.iter().zip([1.0, 3.0, 2.0, 6.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn five_hundred_branch_batch_exposes_expert_histogram() {
        let logits: Vec<f32> = (0..500 * QWEN3_CODER_A3B_EXPERTS)
            .map(|index| ((index * 37 + index / 128 * 19) % 257) as f32 / 32.0)
            .collect();
        let plan = MoeBatchPlan::qwen3_coder_a3b(&logits, 500).unwrap();
        let telemetry = plan.telemetry();
        assert_eq!(telemetry.tokens_per_expert.iter().sum::<u32>(), 4_000);
        assert!(telemetry.experts_touched > 100);
        assert!(telemetry.expert_batch_p95 >= telemetry.expert_batch_p50);
        assert!(telemetry.routing_entropy_bits > 6.0);
    }

    #[test]
    fn rejects_corrupt_router_values_and_shape() {
        assert!(matches!(
            MoeBatchPlan::from_logits(&[f32::NAN], 1, 1, 1),
            Err(MoePlanError::InvalidLogit { .. })
        ));
        assert!(matches!(
            MoeBatchPlan::from_logits(&[1.0], 1, 2, 1),
            Err(MoePlanError::InvalidShape)
        ));
    }
}
