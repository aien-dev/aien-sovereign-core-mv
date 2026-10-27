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

//! Device placement: AIEN's per-op dispatch record must agree with the external device witness
//! and with `placement.toml` (SPEC.md Section 6).

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

/// Allowlist entries with this prefix are not yet frozen and never match.
pub const PENDING_PREFIX: &str = "PENDING";
pub const CPU: &str = "cpu";

#[derive(Debug, Clone, Deserialize)]
pub struct OpPolicy {
    pub allowed: Vec<String>,
    #[serde(default)]
    pub kernels: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlacementSpec {
    pub ops: BTreeMap<String, OpPolicy>,
}

/// One line of `op_records.jsonl`, written by AIEN.
#[derive(Debug, Clone, Deserialize)]
pub struct OpRecord {
    pub op_call_id: u64,
    pub op: String,
    pub reported_device: String,
    #[serde(default)]
    pub fallback: bool,
}

/// One line of `witness.jsonl`, written by the external device witness.
#[derive(Debug, Clone, Deserialize)]
pub struct WitnessEvent {
    pub op_call_id: u64,
    pub kernel: String,
    pub completed: bool,
}

/// Glob match supporting `*` only.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let first = parts[0];
    let last = parts[parts.len() - 1];
    if text.len() < first.len() + last.len() || !text.starts_with(first) || !text.ends_with(last) {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    true
}

pub fn check_placement(
    spec: &PlacementSpec,
    records: &[OpRecord],
    witness: &[WitnessEvent],
) -> Vec<String> {
    let mut v = Vec::new();

    let mut kernels_by_call: BTreeMap<u64, Vec<&WitnessEvent>> = BTreeMap::new();
    for w in witness {
        kernels_by_call.entry(w.op_call_id).or_default().push(w);
    }

    let mut seen = BTreeSet::new();
    for r in records {
        let id = r.op_call_id;
        if !seen.insert(id) {
            v.push(format!("op_call {id}: duplicate op_call_id"));
        }
        let Some(policy) = spec.ops.get(&r.op) else {
            v.push(format!("op_call {id}: op {} not in placement spec", r.op));
            continue;
        };
        if !policy.allowed.iter().any(|d| d == &r.reported_device) {
            v.push(format!(
                "op_call {id}: {} ran on {}, allowed {:?}",
                r.op, r.reported_device, policy.allowed
            ));
        }
        if r.fallback {
            v.push(format!("op_call {id}: {} fell back", r.op));
        }

        let kernels = kernels_by_call.get(&id).map(Vec::as_slice).unwrap_or(&[]);
        if r.reported_device == CPU {
            if !kernels.is_empty() {
                v.push(format!(
                    "op_call {id}: {} reported cpu but witness saw {} kernel(s)",
                    r.op,
                    kernels.len()
                ));
            }
            continue;
        }
        if kernels.is_empty() {
            v.push(format!(
                "op_call {id}: {} reported {} but witness saw no kernel",
                r.op, r.reported_device
            ));
        }
        for k in kernels {
            if !k.completed {
                v.push(format!(
                    "op_call {id}: kernel {} did not complete",
                    k.kernel
                ));
            }
            let allowed = policy
                .kernels
                .iter()
                .any(|p| !p.starts_with(PENDING_PREFIX) && glob_match(p, &k.kernel));
            if !allowed {
                v.push(format!(
                    "op_call {id}: kernel {} not in the frozen allowlist for {}",
                    k.kernel, r.op
                ));
            }
        }
    }

    for id in kernels_by_call.keys() {
        if !seen.contains(id) {
            v.push(format!(
                "witness saw kernel(s) for op_call {id} with no dispatch record"
            ));
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PlacementSpec {
        toml::from_str(
            r#"
            [ops.rmsnorm]
            allowed = ["cpu"]

            [ops.paged_attention_batch]
            allowed = ["gb10"]
            kernels = ["*paged_attention_bf16*"]

            [ops.compute_logits]
            allowed = ["gb10"]
            kernels = ["PENDING_GATE0_CUBLAS_GEMV"]
            "#,
        )
        .unwrap()
    }

    fn rec(id: u64, op: &str, device: &str) -> OpRecord {
        OpRecord {
            op_call_id: id,
            op: op.into(),
            reported_device: device.into(),
            fallback: false,
        }
    }

    fn kernel(id: u64, name: &str) -> WitnessEvent {
        WitnessEvent {
            op_call_id: id,
            kernel: name.into(),
            completed: true,
        }
    }

    #[test]
    fn glob() {
        assert!(glob_match(
            "*paged_attention_bf16*",
            "void paged_attention_bf16_kernel<128>"
        ));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exactly"));
        assert!(glob_match("a*c", "abc"));
        assert!(!glob_match("ab*bc", "abc"));
    }

    #[test]
    fn honest_hybrid_run_passes() {
        let records = [
            rec(1, "rmsnorm", "cpu"),
            rec(2, "paged_attention_batch", "gb10"),
        ];
        let witness = [kernel(2, "paged_attention_bf16_kernel")];
        assert!(check_placement(&spec(), &records, &witness).is_empty());
    }

    #[test]
    fn required_gpu_op_on_cpu_fails() {
        let records = [rec(1, "paged_attention_batch", "cpu")];
        assert!(!check_placement(&spec(), &records, &[]).is_empty());
    }

    #[test]
    fn backend_claiming_gpu_without_kernel_fails() {
        let records = [rec(1, "paged_attention_batch", "gb10")];
        let v = check_placement(&spec(), &records, &[]);
        assert!(v.iter().any(|m| m.contains("witness saw no kernel")));
    }

    #[test]
    fn kernel_under_cpu_report_fails() {
        let records = [rec(1, "rmsnorm", "cpu")];
        let witness = [kernel(1, "rmsnorm_kernel")];
        let v = check_placement(&spec(), &records, &witness);
        assert!(v.iter().any(|m| m.contains("reported cpu")));
    }

    #[test]
    fn wrong_kernel_symbol_fails() {
        let records = [rec(1, "paged_attention_batch", "gb10")];
        let witness = [kernel(1, "some_other_kernel")];
        let v = check_placement(&spec(), &records, &witness);
        assert!(v.iter().any(|m| m.contains("not in the frozen allowlist")));
    }

    #[test]
    fn pending_allowlist_never_passes() {
        let records = [rec(1, "compute_logits", "gb10")];
        let witness = [kernel(1, "PENDING_GATE0_CUBLAS_GEMV")];
        assert!(!check_placement(&spec(), &records, &witness).is_empty());
    }

    #[test]
    fn fallback_fails() {
        let mut r = rec(1, "rmsnorm", "cpu");
        r.fallback = true;
        assert!(!check_placement(&spec(), &[r], &[]).is_empty());
    }

    #[test]
    fn orphan_witness_kernel_fails() {
        let witness = [kernel(9, "paged_attention_bf16_kernel")];
        let v = check_placement(&spec(), &[], &witness);
        assert!(v.iter().any(|m| m.contains("no dispatch record")));
    }
}
