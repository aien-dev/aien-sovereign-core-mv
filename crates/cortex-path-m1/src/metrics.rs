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

/// Binary nDCG@k. Discount at rank `i` (1-based) is `1 / log2(i + 1)`.
pub fn ndcg_at_k(ranked_ids: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let gain = |rank: usize, hit: bool| -> f64 {
        if !hit {
            return 0.0;
        }
        1.0 / ((rank + 1) as f64).log2()
    };
    let dcg: f64 = ranked_ids
        .iter()
        .take(k)
        .enumerate()
        .fold(0.0, |sum, (idx, id)| {
            sum + gain(idx + 1, relevant.iter().any(|item| item == id))
        });
    let ideal_hits = relevant.len().min(k);
    let idcg: f64 = (0..ideal_hits).map(|offset| gain(offset + 1, true)).sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

pub fn recall_at_k(ranked_ids: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranked_ids
        .iter()
        .take(k)
        .filter(|id| relevant.iter().any(|item| item == *id))
        .count();
    hits as f64 / relevant.len() as f64
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Paired bootstrap of the mean. Returns `(low, high)` as the 2.5 and 97.5 percentiles.
pub fn bootstrap_mean_ci(values: &[f64], samples: u32, seed: u64) -> Option<(f64, f64)> {
    if values.is_empty() || samples == 0 {
        return None;
    }
    let mut state = seed;
    let mut means = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let mut total = 0.0;
        for _ in 0..values.len() {
            let index = splitmix(&mut state) as usize % values.len();
            total += values[index];
        }
        means.push(total / values.len() as f64);
    }
    means.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let low_index = ((samples as f64) * 0.025).floor() as usize;
    let high_index = (((samples as f64) * 0.975).ceil() as usize).saturating_sub(1);
    let high_index = high_index.min(means.len() - 1);
    Some((means[low_index.min(means.len() - 1)], means[high_index]))
}

pub fn percentile_95(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let index = (0.95 * values.len() as f64).ceil() as usize - 1;
    values[index.min(values.len() - 1)]
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

pub fn ci_excludes_zero(interval: (f64, f64)) -> bool {
    interval.0 > 0.0 || interval.1 < 0.0
}
