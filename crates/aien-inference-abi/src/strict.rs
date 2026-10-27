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

//! Strict real-model execution (AIEN 0.1 commissioning gate GB10-4).
//!
//! A production build (the default: cargo feature `dev-fallback` off) fails
//! closed: no reference weights stand in for a missing checkpoint, no silent
//! CPU execution stands in for a GPU backend that was asked for, and a GPU
//! call that falls back to CPU is a hard failure, not a counter tick. A
//! dev/test build (`--features dev-fallback`) keeps the old permissive
//! behaviour and says so in every receipt.

/// True when this build allows reference weights and software fallbacks.
pub const DEV_FALLBACK: bool = cfg!(feature = "dev-fallback");

/// Environment opt-in for dev and CI runs on machines without the GPU:
/// `AIEN_DEV_FALLBACK=1`. Read once per process, recorded in every receipt,
/// and rejected by the acceptance gate, so a permissive run can never pass
/// as a strict one.
pub const DEV_FALLBACK_ENV: &str = "AIEN_DEV_FALLBACK";

fn env_opt_in() -> bool {
    static OPT_IN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OPT_IN.get_or_init(|| {
        std::env::var(DEV_FALLBACK_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// True when fallbacks are allowed in this process: the `dev-fallback`
/// feature or the `AIEN_DEV_FALLBACK=1` environment opt-in.
pub fn dev_fallback_active() -> bool {
    DEV_FALLBACK || env_opt_in()
}

/// True when the process must fail closed (the production default).
pub fn production_strict() -> bool {
    !dev_fallback_active()
}

/// Tag every strict refusal carries, so logs and receipts can be grepped.
pub const STRICT_VIOLATION_PREFIX: &str = "STRICT_REAL_MODEL_VIOLATION";

/// Build the refusal message for a strict violation.
pub fn violation(what: &str) -> String {
    format!("{STRICT_VIOLATION_PREFIX}: {what}; this production build refuses to fall back (a dev/test run sets AIEN_DEV_FALLBACK=1 or builds with --features dev-fallback)")
}

/// Called by a backend at the moment it would silently run an operation on
/// CPU instead of the GPU it was bound to. In a production build this is
/// fatal: a wrong-or-slow result is worse than no result for the gate.
pub fn fallback_taken(backend: &str, op: &str) {
    if production_strict() {
        panic!(
            "{}",
            violation(&format!("{backend} fell back to CPU in {op}"))
        );
    }
}

/// What a strict acceptance receipt must record (GB10-4).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StrictModelReceipt {
    pub checkpoint_path: String,
    pub checkpoint_sha256: String,
    pub tokenizer_path: String,
    pub tokenizer_sha256: String,
    pub backend_identity: String,
    pub model_id: String,
    pub model_config: serde_json::Value,
    pub fallback_count: u64,
    /// Native-versus-reference op listing for the run (None for receipts that predate cut 3a).
    #[serde(default)]
    pub op_report: Option<crate::native_ops::OpReport>,
    pub dev_fallback_build: bool,
    pub verdict: String,
}

impl StrictModelReceipt {
    /// The gate: a production build, a native GPU backend and zero fallbacks.
    pub fn verify(&mut self) -> Result<(), String> {
        let mut problems = Vec::new();
        if self.dev_fallback_build {
            problems.push("built with dev-fallback".to_string());
        }
        if self.fallback_count != 0 {
            problems.push(format!("fallback_count = {}", self.fallback_count));
        }
        if !self.backend_identity.contains("sm_121") {
            problems.push(format!(
                "backend is not native GB10 sm_121: {}",
                self.backend_identity
            ));
        }
        if self.checkpoint_sha256.len() != 64 || self.tokenizer_sha256.len() != 64 {
            problems.push("checkpoint or tokenizer digest missing".to_string());
        }
        if problems.is_empty() {
            self.verdict = "PASS".to_string();
            Ok(())
        } else {
            self.verdict = format!("FAIL: {}", problems.join("; "));
            Err(violation(&problems.join("; ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_ops::{NativeOpMask, OpAccounting, TensorOp};

    fn receipt(fallback_count: u64, report: crate::native_ops::OpReport) -> StrictModelReceipt {
        StrictModelReceipt {
            checkpoint_path: "x".into(),
            checkpoint_sha256: "0".repeat(64),
            tokenizer_path: "y".into(),
            tokenizer_sha256: "0".repeat(64),
            backend_identity: "OmegaGb10Backend (sm_121)".into(),
            model_id: "m".into(),
            model_config: serde_json::json!({}),
            fallback_count,
            op_report: Some(report),
            dev_fallback_build: false,
            verdict: String::new(),
        }
    }

    #[test]
    fn reference_ops_by_design_pass_the_receipt_gate() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
        acct.record_reference(TensorOp::Rmsnorm);
        let mut r = receipt(acct.fallback_count(), acct.report());
        assert!(r.verify().is_ok(), "{}", r.verdict);
        assert!(r.op_report.as_ref().unwrap().line().contains("rmsnorm:1"));
    }

    #[test]
    fn claimed_native_fallback_fails_the_receipt_gate() {
        let acct = OpAccounting::new(NativeOpMask::from_ops(&[TensorOp::MatmulVec]));
        acct.record_reference(TensorOp::MatmulVec);
        let mut r = receipt(acct.fallback_count(), acct.report());
        assert!(r.verify().is_err());
    }
}
