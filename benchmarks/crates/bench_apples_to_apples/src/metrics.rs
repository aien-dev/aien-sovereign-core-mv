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

//! Statistical calculations for percentiles, throughput, and energy metrics.

/// Computes the p-th percentile from a slice of floats.
pub fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let rank = (percentile / 100.0) * (sorted.len() - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = rank.ceil() as usize;

    if lower_idx == upper_idx {
        sorted[lower_idx]
    } else {
        let weight = rank - lower_idx as f64;
        sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
    }
}

/// Computes energy efficiency in Joules per token.
/// Formula: (Average Power in Watts * Total Elapsed Time in Seconds) / Total Tokens Generated.
pub fn calculate_joules_per_token(
    avg_power_watts: f64,
    elapsed_secs: f64,
    total_tokens: usize,
) -> f64 {
    if total_tokens == 0 {
        return 0.0;
    }
    (avg_power_watts * elapsed_secs) / (total_tokens as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_percentiles() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(calculate_percentile(&data, 50.0), 30.0);
        assert!((calculate_percentile(&data, 95.0) - 48.0).abs() < 1e-4);
    }
}
