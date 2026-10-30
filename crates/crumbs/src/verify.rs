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

//! The generic verifier contract (sealed side).
//!
//! A learner's candidate arrives as a CPG1 `Program`. The verifier executes it
//! with this crate's own interpreter against the visible examples and the
//! three sealed held-out tiers, and produces a `SealedEvaluation`. The learner
//! receives exactly one byte derived from it (`learner_verdict`).
//!
//! Visible interpolation is never verification: a candidate is accepted only
//! if it also predicts every sealed held-out.

use crate::canon::Enc;
use crate::digest::{digest, Digest, Domain};
use crate::program::Program;
use crate::sealed::{AdversarialClass, DecoyStatus, HeldoutTier, SealedCrumb};
use crate::visible::{lane_mask, Encoding, VisibleCrumb};
use serde::{Deserialize, Serialize};

pub const VERIFIER_VERSION: &str = "crumbs-verifier/1.0.0";

/// Identity of the verification rules and interpreter.
pub fn verifier_digest() -> Digest {
    let mut e = Enc::new();
    e.str(VERIFIER_VERSION).str(crate::program::std_interpreter_id()).str("accept=visible_ok&&all_heldout_tiers_pass;noise=mismatch_subset_of_sealed_noisy_indices");
    digest(Domain::Verifier, &e.finish())
}

/// The single byte a learner is told.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Verdict {
    Accept = 1,
    Reject = 2,
    BudgetExhausted = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalClass {
    /// Fits the visible examples and predicts every held-out.
    Verified,
    /// Does not even explain the visible examples.
    FalsifiedVisible,
    /// Explains the visible examples, fails held-outs: a recorded useful failure.
    FalsifiedHidden,
    /// Predicts the held-outs, but the visible evidence could not have
    /// established it (insufficient-evidence crumb).
    AmbiguousUnderdetermined,
    /// The crumb's shape is outside CPG1 v1 (output arity != 1).
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierResult {
    pub passed: u32,
    pub total: u32,
}

impl TierResult {
    pub fn ok(&self) -> bool {
        self.passed == self.total
    }
}

/// Machine-readable evidence for one evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedEvaluation {
    pub class: EvalClass,
    pub verdict: Verdict,
    pub visible_matches: u32,
    pub visible_total: u32,
    pub interpolation: TierResult,
    pub extrapolation: TierResult,
    pub adversarial: TierResult,
    /// The visible evidence admitted rival explanations (sealed alternatives exist).
    pub underdetermined: bool,
    /// Insufficient-evidence crumbs only: did the learner report >= 2 rival fits?
    pub ambiguity_recognized: Option<bool>,
    /// Index of the sealed alternative this candidate behaves like on the held-outs.
    pub matches_alternative: Option<u32>,
    pub learner_hypotheses: u8,
    pub robust_submission: bool,
    pub crumb_digest: Digest,
    pub sealed_digest: Digest,
    pub generator_instance_digest: Digest,
    pub program_digest: Digest,
    pub visible_set_digest: Digest,
    pub heldout_set_digest: Digest,
    pub verifier_digest: Digest,
    pub program_steps: u32,
    pub evaluation_ops: u64,
}

impl SealedEvaluation {
    pub fn tier(&self, t: HeldoutTier) -> TierResult {
        match t {
            HeldoutTier::Interpolation => self.interpolation,
            HeldoutTier::Extrapolation => self.extrapolation,
            HeldoutTier::Adversarial => self.adversarial,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.bytes(b"CEV1").u8(self.class as u8).u8(self.verdict as u8);
        e.u32(self.visible_matches).u32(self.visible_total);
        for t in [self.interpolation, self.extrapolation, self.adversarial] {
            e.u32(t.passed).u32(t.total);
        }
        e.u8(self.underdetermined as u8);
        e.u8(match self.ambiguity_recognized {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
        e.u32(self.matches_alternative.unwrap_or(u32::MAX));
        e.u8(self.learner_hypotheses)
            .u8(self.robust_submission as u8);
        for d in [
            &self.crumb_digest,
            &self.sealed_digest,
            &self.generator_instance_digest,
            &self.program_digest,
            &self.visible_set_digest,
            &self.heldout_set_digest,
            &self.verifier_digest,
        ] {
            e.digest(d);
        }
        e.u32(self.program_steps).u64(self.evaluation_ops);
        e.finish()
    }

    /// Evidence receipt id of this evaluation.
    pub fn digest(&self) -> Digest {
        digest(Domain::Evaluation, &self.to_bytes())
    }
}

fn out_mask(v: &VisibleCrumb) -> u64 {
    match v.encoding {
        Encoding::DecimalUtf8 => u64::MAX,
        Encoding::RawLe => lane_mask(v.out_lane_bytes),
    }
}

pub fn evaluate(
    visible: &VisibleCrumb,
    sealed: &SealedCrumb,
    cand: &Program,
    learner_hypotheses: u8,
    robust_submission: bool,
) -> SealedEvaluation {
    let mask = out_mask(visible);
    let mut ops: u64 = 0;
    let steps = cand.len() as u64;
    let mut run = |x: u64| {
        ops += steps;
        cand.run(x) & mask
    };

    let supported = visible.out_arity == 1;
    let values = visible.values();
    let mut visible_matches = 0u32;
    let mut mismatched = Vec::new();
    if supported {
        for (i, (inp, out)) in values.iter().enumerate() {
            if run(inp[0]) == out[0] {
                visible_matches += 1;
            } else {
                mismatched.push(i as u32);
            }
        }
    }
    let visible_ok = supported && mismatched.iter().all(|i| sealed.noisy_visible.contains(i));

    let mut tiers = [TierResult {
        passed: 0,
        total: 0,
    }; 3];
    for (k, t) in HeldoutTier::ALL.iter().enumerate() {
        for ex in sealed.heldouts.tier(*t) {
            tiers[k].total += 1;
            if supported && run(ex.input[0]) == ex.output[0] {
                tiers[k].passed += 1;
            }
        }
    }
    let hidden_ok = tiers.iter().all(|t| t.ok());

    let insufficient = matches!(
        sealed.decoy_status,
        DecoyStatus::RabbitHole(AdversarialClass::InsufficientEvidence)
    );
    let class = if !supported {
        EvalClass::Unsupported
    } else if !visible_ok {
        EvalClass::FalsifiedVisible
    } else if !hidden_ok {
        EvalClass::FalsifiedHidden
    } else if insufficient {
        EvalClass::AmbiguousUnderdetermined
    } else {
        EvalClass::Verified
    };
    let verdict = if matches!(
        class,
        EvalClass::Verified | EvalClass::AmbiguousUnderdetermined
    ) {
        Verdict::Accept
    } else {
        Verdict::Reject
    };

    let matches_alternative = if supported {
        sealed.alternatives.iter().position(|alt| {
            HeldoutTier::ALL
                .iter()
                .flat_map(|t| sealed.heldouts.tier(*t))
                .all(|ex| alt.run(ex.input[0]) & mask == cand.run(ex.input[0]) & mask)
        })
    } else {
        None
    };

    SealedEvaluation {
        class,
        verdict,
        visible_matches,
        visible_total: values.len() as u32,
        interpolation: tiers[0],
        extrapolation: tiers[1],
        adversarial: tiers[2],
        underdetermined: !sealed.alternatives.is_empty(),
        ambiguity_recognized: if insufficient {
            Some(learner_hypotheses >= 2)
        } else {
            None
        },
        matches_alternative: matches_alternative.map(|i| i as u32),
        learner_hypotheses,
        robust_submission,
        crumb_digest: visible.digest(),
        sealed_digest: sealed.digest(),
        generator_instance_digest: sealed.generator_instance_digest,
        program_digest: cand.digest(),
        visible_set_digest: visible.visible_set_digest(),
        heldout_set_digest: sealed.heldouts.digest(),
        verifier_digest: verifier_digest(),
        program_steps: cand.len() as u32,
        evaluation_ops: ops,
    }
}
