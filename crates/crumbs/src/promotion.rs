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

//! The promotion ladder (sealed side).
//!
//! One solved crumb never makes an operation universal. An operation climbs:
//!
//! ```text
//! candidate operation            sub-chain of an accepted solution
//!   -> held-out verified          part of a solution the sealed verifier accepted
//!   -> corroborated               supports >= min_support distinct crumbs
//!   -> survived adversarial       every support passed a non-empty adversarial tier,
//!                                 and rejected-submission exposures stay bounded
//!   -> generalized                every support passed a non-empty extrapolation tier
//!   -> admitted                   sent to the learner's library with an explicit scope
//! ```
//!
//! Scope is recorded, not assumed: `scope_bits` is the widest input magnitude
//! (in bits) over which a supporting solution was verified. A learner may only
//! apply an admitted operation to crumbs inside that scope.

use crate::canon::Enc;
use crate::digest::{digest, Digest, Domain};
use crate::program::Program;
use crate::sealed::{HeldoutTier, SealedCrumb};
use crate::verify::{EvalClass, SealedEvaluation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Stage {
    Candidate = 1,
    HeldoutVerified = 3,
    Corroborated = 4,
    SurvivedAdversarial = 5,
    Generalized = 6,
    Admitted = 7,
    Suspended = 9,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionConfig {
    pub min_len: usize,
    pub max_len: usize,
    pub min_support: usize,
    /// Suspend when exposures in rejected submissions exceed this percent of supports.
    pub max_exposure_pct: u32,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        PromotionConfig {
            min_len: 2,
            max_len: 6,
            min_support: 2,
            max_exposure_pct: 200,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Support {
    pub seq: u64,
    pub crumb: Digest,
    pub family_id: u32,
    pub evaluation: Digest,
    pub adversarial_total: u32,
    pub extrapolation_total: u32,
    pub max_input_bits: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpRecord {
    pub behavior: Digest,
    pub program: Program,
    pub stage: Stage,
    pub supports: Vec<Support>,
    pub exposures: Vec<Digest>,
    pub scope_bits: u8,
    pub first_seen_seq: u64,
    pub op_ref: Option<u32>,
    pub history: Vec<(Stage, u64)>,
}

impl OpRecord {
    pub fn distinct_crumbs(&self) -> usize {
        let mut v: Vec<_> = self.supports.iter().map(|s| s.crumb).collect();
        v.sort();
        v.dedup();
        v.len()
    }

    /// Commitment to the operation, its evidence and its scope.
    pub fn promotion_digest(&self) -> Digest {
        let mut e = Enc::new();
        e.blob(&self.program.to_bytes())
            .u8(self.stage as u8)
            .u8(self.scope_bits);
        e.u32(self.supports.len() as u32);
        for s in &self.supports {
            e.digest(&s.crumb).digest(&s.evaluation);
        }
        e.u32(self.exposures.len() as u32);
        for x in &self.exposures {
            e.digest(x);
        }
        digest(Domain::Promotion, &e.finish())
    }
}

/// A newly admitted operation, to be sent to a learner that may reuse it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Admission {
    pub op_ref: u32,
    pub program: Program,
    pub scope_bits: u8,
    pub promotion_digest: Digest,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Ladder {
    pub cfg: PromotionConfig,
    pub ops: BTreeMap<Digest, OpRecord>,
    next_ref: u32,
}

fn bits(x: u64) -> u8 {
    (64 - x.leading_zeros()) as u8
}

impl Ladder {
    pub fn new(cfg: PromotionConfig) -> Self {
        Ladder {
            cfg,
            ops: BTreeMap::new(),
            next_ref: 1,
        }
    }

    fn candidates_of(&self, program: &Program) -> Vec<Program> {
        let mut out = program.sub_chains(self.cfg.min_len, self.cfg.max_len);
        out.sort();
        out.dedup();
        out
    }

    /// An accepted solution: every sub-chain is (evidence for) a candidate operation.
    pub fn observe_accept(
        &mut self,
        seq: u64,
        sealed: &SealedCrumb,
        eval: &SealedEvaluation,
        program: &Program,
    ) {
        if !matches!(eval.class, EvalClass::Verified) {
            return; // ambiguous acceptances are not evidence for reuse
        }
        let max_in = HeldoutTier::ALL
            .iter()
            .flat_map(|t| sealed.heldouts.tier(*t))
            .map(|ex| ex.input[0])
            .max()
            .unwrap_or(0);
        for cand in self.candidates_of(program) {
            let b = cand.behavior();
            let rec = self.ops.entry(b).or_insert_with(|| OpRecord {
                behavior: b,
                program: cand.clone(),
                stage: Stage::Candidate,
                supports: vec![],
                exposures: vec![],
                scope_bits: 0,
                first_seen_seq: seq,
                op_ref: None,
                history: vec![(Stage::Candidate, seq)],
            });
            if cand.len() < rec.program.len() {
                rec.program = cand.clone();
            }
            if rec.supports.iter().any(|s| s.crumb == eval.crumb_digest) {
                continue;
            }
            rec.supports.push(Support {
                seq,
                crumb: eval.crumb_digest,
                family_id: sealed.family_id,
                evaluation: eval.digest(),
                adversarial_total: eval.adversarial.total,
                extrapolation_total: eval.extrapolation.total,
                max_input_bits: bits(max_in),
            });
        }
    }

    /// A submission that fit the visible examples but failed held-outs.
    pub fn observe_reject(&mut self, eval: &SealedEvaluation, program: &Program) {
        if !matches!(eval.class, EvalClass::FalsifiedHidden) {
            return;
        }
        for cand in self.candidates_of(program) {
            if let Some(rec) = self.ops.get_mut(&cand.behavior()) {
                rec.exposures.push(eval.digest());
            }
        }
    }

    /// Climb every operation as far as its evidence allows. Returns operations
    /// that reached `Admitted` in this call.
    pub fn advance(&mut self, seq: u64) -> Vec<Admission> {
        let mut admitted = Vec::new();
        let cfg = self.cfg.clone();
        for rec in self.ops.values_mut() {
            if rec.stage == Stage::Admitted || rec.stage == Stage::Suspended {
                continue;
            }
            let supports = rec.distinct_crumbs();
            if supports > 0 && rec.exposures.len() * 100 > supports * cfg.max_exposure_pct as usize
            {
                rec.stage = Stage::Suspended;
                rec.history.push((Stage::Suspended, seq));
                continue;
            }
            let mut next = Stage::Candidate;
            if supports >= 1 {
                next = Stage::HeldoutVerified;
            }
            if next == Stage::HeldoutVerified && supports >= cfg.min_support {
                next = Stage::Corroborated;
            }
            if next == Stage::Corroborated && rec.supports.iter().all(|s| s.adversarial_total > 0) {
                next = Stage::SurvivedAdversarial;
            }
            if next == Stage::SurvivedAdversarial
                && rec.supports.iter().all(|s| s.extrapolation_total > 0)
            {
                next = Stage::Generalized;
            }
            if next == Stage::Generalized {
                next = Stage::Admitted;
            }
            // Record every rung crossed, in order.
            for st in [
                Stage::HeldoutVerified,
                Stage::Corroborated,
                Stage::SurvivedAdversarial,
                Stage::Generalized,
                Stage::Admitted,
            ] {
                if st > rec.stage && st <= next {
                    rec.history.push((st, seq));
                }
            }
            if next > rec.stage {
                rec.stage = next;
            }
            rec.scope_bits = rec
                .supports
                .iter()
                .map(|s| s.max_input_bits)
                .max()
                .unwrap_or(0);
            if rec.stage == Stage::Admitted && rec.op_ref.is_none() {
                let r = self.next_ref;
                self.next_ref += 1;
                rec.op_ref = Some(r);
                admitted.push(Admission {
                    op_ref: r,
                    program: rec.program.clone(),
                    scope_bits: rec.scope_bits,
                    promotion_digest: rec.promotion_digest(),
                });
            }
        }
        admitted
    }

    pub fn by_ref(&self, op_ref: u32) -> Option<&OpRecord> {
        self.ops.values().find(|r| r.op_ref == Some(op_ref))
    }

    pub fn count(&self, stage: Stage) -> usize {
        self.ops.values().filter(|r| r.stage == stage).count()
    }
}
