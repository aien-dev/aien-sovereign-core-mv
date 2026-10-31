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

//! Per-operation native accounting for `TensorBackend` (FB-1 cut 3a).
//!
//! A backend that runs only some operations on the device (for example
//! matmuls first, attention last) declares a [`NativeOpMask`]: the set of
//! operations it CLAIMS to run natively. Operations outside the mask run on
//! the reference CPU path by design and never trip strict mode. An operation
//! inside the mask that runs on CPU is a fallback: it is counted, and in a
//! production build it is fatal. Nothing is hidden: [`OpReport`] lists
//! native and reference operations and the per-op counters for every run.
//!
//! Default for every existing backend: all operations native, which is the
//! pre-existing strict semantics.

use std::sync::atomic::{AtomicU64, Ordering};

/// The ten operations of `TensorBackend` (fallback_count is not an op). New ops go last so
/// existing mask bits keep their positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorOp {
    Rmsnorm,
    ApplyRope,
    MatmulVec,
    MatmulBatch,
    Swiglu,
    GqaAttention,
    PagedAttention,
    PagedAttentionBatch,
    ComputeLogits,
    RmsnormHeads,
}

impl TensorOp {
    pub const COUNT: usize = 10;
    pub const ALL: [TensorOp; Self::COUNT] = [
        TensorOp::Rmsnorm,
        TensorOp::ApplyRope,
        TensorOp::MatmulVec,
        TensorOp::MatmulBatch,
        TensorOp::Swiglu,
        TensorOp::GqaAttention,
        TensorOp::PagedAttention,
        TensorOp::PagedAttentionBatch,
        TensorOp::ComputeLogits,
        TensorOp::RmsnormHeads,
    ];

    fn index(self) -> usize {
        self as usize
    }

    /// Stable name, equal to the trait method name.
    pub fn name(self) -> &'static str {
        match self {
            TensorOp::Rmsnorm => "rmsnorm",
            TensorOp::ApplyRope => "apply_rope",
            TensorOp::MatmulVec => "matmul_vec",
            TensorOp::MatmulBatch => "matmul_batch",
            TensorOp::Swiglu => "swiglu",
            TensorOp::GqaAttention => "gqa_attention",
            TensorOp::PagedAttention => "paged_attention",
            TensorOp::PagedAttentionBatch => "paged_attention_batch",
            TensorOp::ComputeLogits => "compute_logits",
            TensorOp::RmsnormHeads => "rmsnorm_heads",
        }
    }
}

/// The set of operations a backend claims to run natively on its device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOpMask(u16);

impl NativeOpMask {
    /// Every op claimed native (today's semantics, the trait default).
    pub const ALL: NativeOpMask = NativeOpMask((1 << TensorOp::COUNT) - 1);
    /// No op claimed native (pure reference backend that wants no strict trips).
    pub const NONE: NativeOpMask = NativeOpMask(0);

    pub fn from_ops(ops: &[TensorOp]) -> Self {
        ops.iter().fold(Self::NONE, |m, op| m.with(*op))
    }

    pub fn with(self, op: TensorOp) -> Self {
        NativeOpMask(self.0 | (1 << op.index()))
    }

    pub fn without(self, op: TensorOp) -> Self {
        NativeOpMask(self.0 & !(1 << op.index()))
    }

    pub fn contains(self, op: TensorOp) -> bool {
        self.0 & (1 << op.index()) != 0
    }

    pub fn native_ops(self) -> Vec<TensorOp> {
        TensorOp::ALL
            .into_iter()
            .filter(|op| self.contains(*op))
            .collect()
    }

    pub fn reference_ops(self) -> Vec<TensorOp> {
        TensorOp::ALL
            .into_iter()
            .filter(|op| !self.contains(*op))
            .collect()
    }
}

impl Default for NativeOpMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// Per-run record of which ops were native versus reference, with counters.
/// Goes into the strict receipt and the run log so nothing is hidden.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpReport {
    pub native_ops: Vec<String>,
    pub reference_ops: Vec<String>,
    /// Fallbacks of ops in the native mask (these trip strict mode), by op name.
    pub native_fallbacks: Vec<(String, u64)>,
    /// Runs of ops outside the mask on the reference CPU path (by design), by op name.
    pub reference_runs: Vec<(String, u64)>,
}

impl OpReport {
    /// The evidence a run keeps (sc#337): `backend`, the summed fallback and
    /// reference-run counts, and this report's [`OpReport::line`].
    pub fn evidence(&self, backend: &str) -> aien_abi_core::OpEvidence {
        let total = |v: &[(String, u64)]| v.iter().map(|(_, c)| c).sum();
        aien_abi_core::OpEvidence {
            backend: backend.to_string(),
            native_fallbacks: total(&self.native_fallbacks),
            reference_runs: total(&self.reference_runs),
            report: self.line(),
            gpu_executions: None,
        }
    }

    /// One greppable line: `OP_REPORT native=[..] reference=[..] native_fallbacks=[..] reference_runs=[..]`.
    pub fn line(&self) -> String {
        let counts = |v: &[(String, u64)]| {
            v.iter()
                .map(|(n, c)| format!("{n}:{c}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "OP_REPORT native=[{}] reference=[{}] native_fallbacks=[{}] reference_runs=[{}]",
            self.native_ops.join(","),
            self.reference_ops.join(","),
            counts(&self.native_fallbacks),
            counts(&self.reference_runs),
        )
    }

    /// Report for a backend with every op native and no fallbacks (the default).
    pub fn all_native() -> Self {
        OpAccounting::new(NativeOpMask::ALL).report()
    }
}

/// Counters a backend embeds. The backend calls [`OpAccounting::reference_path`]
/// every time an op runs on the reference CPU path; the mask decides whether
/// that is by design or a fallback.
#[derive(Debug)]
pub struct OpAccounting {
    mask: NativeOpMask,
    native_fallbacks: [AtomicU64; TensorOp::COUNT],
    reference_runs: [AtomicU64; TensorOp::COUNT],
}

impl OpAccounting {
    pub fn new(mask: NativeOpMask) -> Self {
        Self {
            mask,
            native_fallbacks: Default::default(),
            reference_runs: Default::default(),
        }
    }

    pub fn mask(&self) -> NativeOpMask {
        self.mask
    }

    /// Record that `op` ran on the reference CPU path. Returns true when this
    /// was a fallback (the op is claimed native). Does not panic; see
    /// [`OpAccounting::reference_path`] for the strict-aware call.
    pub fn record_reference(&self, op: TensorOp) -> bool {
        if self.mask.contains(op) {
            self.native_fallbacks[op.index()].fetch_add(1, Ordering::Relaxed);
            true
        } else {
            self.reference_runs[op.index()].fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Record a reference-path run and, if the op is claimed native, enter
    /// strict mode (fatal in a production build).
    pub fn reference_path(&self, backend: &str, op: TensorOp) {
        if self.record_reference(op) {
            crate::strict::fallback_taken(backend, op.name());
        }
    }

    /// What strict mode compares against zero: fallbacks of native-claimed ops only.
    pub fn fallback_count(&self) -> u64 {
        TensorOp::ALL
            .into_iter()
            .map(|op| self.native_fallbacks[op.index()].load(Ordering::Relaxed))
            .sum()
    }

    pub fn fallbacks_for(&self, op: TensorOp) -> u64 {
        self.native_fallbacks[op.index()].load(Ordering::Relaxed)
    }

    pub fn reference_runs_for(&self, op: TensorOp) -> u64 {
        self.reference_runs[op.index()].load(Ordering::Relaxed)
    }

    pub fn report(&self) -> OpReport {
        let names = |ops: Vec<TensorOp>| ops.into_iter().map(|o| o.name().to_string()).collect();
        let nonzero = |ctr: &[AtomicU64; TensorOp::COUNT]| {
            TensorOp::ALL
                .into_iter()
                .filter_map(|op| {
                    let c = ctr[op.index()].load(Ordering::Relaxed);
                    (c > 0).then(|| (op.name().to_string(), c))
                })
                .collect()
        };
        OpReport {
            native_ops: names(self.mask.native_ops()),
            reference_ops: names(self.mask.reference_ops()),
            native_fallbacks: nonzero(&self.native_fallbacks),
            reference_runs: nonzero(&self.reference_runs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc#337: the evidence a run keeps sums the counters by kind and keeps
    /// the full line; a forced fallback (counted without the strict check)
    /// shows as a non-zero `native_fallbacks`.
    #[test]
    fn evidence_sums_the_counters_and_keeps_the_line() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
        assert_eq!(acct.report().evidence("omega-gb10").native_fallbacks, 0);
        assert!(acct.record_reference(TensorOp::MatmulVec));
        acct.record_reference(TensorOp::Rmsnorm);
        acct.record_reference(TensorOp::Rmsnorm);
        acct.record_reference(TensorOp::Swiglu);
        let e = acct.report().evidence("omega-gb10");
        assert_eq!(e.backend, "omega-gb10");
        assert_eq!((e.native_fallbacks, e.reference_runs), (1, 3));
        assert_eq!(e.report, acct.report().line());
        assert!(
            e.report.contains("native_fallbacks=[matmul_vec:1]"),
            "{}",
            e.report
        );
    }

    #[test]
    fn default_mask_is_all_native() {
        let m = NativeOpMask::default();
        assert_eq!(m, NativeOpMask::ALL);
        assert!(TensorOp::ALL.iter().all(|op| m.contains(*op)));
        assert!(m.reference_ops().is_empty());
        assert_eq!(m.native_ops().len(), TensorOp::COUNT);
    }

    #[test]
    fn op_bits_are_stable() {
        // The nine original ops keep bits 0..8; rmsnorm_heads (Qwen3 q/k norm) is bit 9.
        assert_eq!(NativeOpMask::from_ops(&[TensorOp::Rmsnorm]).0, 1 << 0);
        assert_eq!(NativeOpMask::from_ops(&[TensorOp::ComputeLogits]).0, 1 << 8);
        assert_eq!(NativeOpMask::from_ops(&[TensorOp::RmsnormHeads]).0, 1 << 9);
        assert_eq!(TensorOp::RmsnormHeads.name(), "rmsnorm_heads");
    }

    #[test]
    fn mask_with_without_and_lists() {
        let m = NativeOpMask::from_ops(&[TensorOp::MatmulVec, TensorOp::MatmulBatch]);
        assert!(m.contains(TensorOp::MatmulVec));
        assert!(!m.contains(TensorOp::Rmsnorm));
        assert_eq!(m.native_ops().len(), 2);
        assert_eq!(m.reference_ops().len(), TensorOp::COUNT - 2);
        let m2 = m.with(TensorOp::ComputeLogits).without(TensorOp::MatmulVec);
        assert!(m2.contains(TensorOp::ComputeLogits));
        assert!(!m2.contains(TensorOp::MatmulVec));
        assert_eq!(NativeOpMask::NONE.native_ops().len(), 0);
    }

    #[test]
    fn op_names_unique_and_match_indices() {
        let mut names: Vec<_> = TensorOp::ALL.iter().map(|o| o.name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), TensorOp::COUNT);
        for (i, op) in TensorOp::ALL.iter().enumerate() {
            assert_eq!(op.index(), i);
        }
    }

    #[test]
    fn reference_op_outside_mask_does_not_count_as_fallback() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
        assert!(!acct.record_reference(TensorOp::Rmsnorm));
        assert!(!acct.record_reference(TensorOp::Rmsnorm));
        assert_eq!(acct.fallback_count(), 0);
        assert_eq!(acct.reference_runs_for(TensorOp::Rmsnorm), 2);
        // strict-aware call must not panic for a reference-by-design op.
        acct.reference_path("T", TensorOp::Swiglu);
        assert_eq!(acct.fallback_count(), 0);
    }

    /// Negative control: a claimed-native op that runs on CPU is counted
    /// and trips strict mode (fatal in a production build).
    #[test]
    fn claimed_native_op_falling_back_is_counted_and_trips_strict() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
        assert!(acct.record_reference(TensorOp::MatmulVec));
        assert_eq!(acct.fallback_count(), 1);
        assert_eq!(acct.fallbacks_for(TensorOp::MatmulVec), 1);
        assert_eq!(acct.fallbacks_for(TensorOp::Rmsnorm), 0);
        if crate::strict::production_strict() {
            let r = std::panic::catch_unwind(|| {
                let a = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
                a.reference_path("T", TensorOp::MatmulVec);
            });
            let err = r.expect_err("strict mode must panic on claimed-native fallback");
            let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
            assert!(
                msg.contains(crate::strict::STRICT_VIOLATION_PREFIX),
                "{msg}"
            );
            assert!(msg.contains("matmul_vec"), "{msg}");
        }
    }

    #[test]
    fn report_lists_native_and_reference_and_counters() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[
            TensorOp::MatmulVec,
            TensorOp::MatmulBatch,
        ]));
        acct.record_reference(TensorOp::Rmsnorm);
        acct.record_reference(TensorOp::MatmulBatch);
        let rep = acct.report();
        assert_eq!(rep.native_ops, vec!["matmul_vec", "matmul_batch"]);
        assert_eq!(rep.reference_ops.len(), TensorOp::COUNT - 2);
        assert_eq!(rep.native_fallbacks, vec![("matmul_batch".to_string(), 1)]);
        assert_eq!(rep.reference_runs, vec![("rmsnorm".to_string(), 1)]);
        let line = rep.line();
        assert!(line.starts_with("OP_REPORT native=[matmul_vec,matmul_batch]"));
        assert!(line.contains("native_fallbacks=[matmul_batch:1]"));
        assert!(line.contains("reference_runs=[rmsnorm:1]"));
    }

    #[test]
    fn default_report_is_all_native() {
        let rep = OpReport::all_native();
        assert_eq!(rep.native_ops.len(), TensorOp::COUNT);
        assert!(rep.reference_ops.is_empty());
        assert!(rep.native_fallbacks.is_empty());
    }
}
