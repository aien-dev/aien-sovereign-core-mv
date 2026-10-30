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

//! Generator framework: deterministic families built from reusable mechanisms.
//!
//! `generate(family, seed)` is a pure function of (family_id,
//! generator_version, seed, canonical parameters). It returns the
//! learner-visible crumb and its sealed record together; the two are separated
//! by type from that point on.
//!
//! Streams: the hidden rule is drawn from `rule`, visible inputs from
//! `visible`, and held-outs from `heldout` keyed by a separate heldout seed,
//! so the visible set can be regenerated without touching the held-outs.

pub mod mechanisms;
pub mod registry;
pub mod rng;

use crate::canon::Enc;
use crate::digest::{digest, Digest, Domain};
use crate::program::{Op, Program, Step};
use crate::provenance::Provenance;
use crate::sealed::{
    generator_instance_digest, CapabilityId, DecoyStatus, Heldouts, KnownSolution, LaneExample,
    Rung, SealedCrumb, SourceFamily, TruthStatus, VerificationMethod, SEALED_SCHEMA_VERSION,
};
use crate::visible::{lane_mask, Encoding, VisibleCrumb};
use rng::Rng;
use serde::Serialize;
use std::collections::HashSet;

pub use mechanisms::Mechanism;
pub use registry::{capability_pool, registry, Registry};

/// Version of every v1 generator. A behaviour change requires a bump; the
/// golden vectors in the test suite fail otherwise.
pub const GENERATOR_VERSION: &str = "1.0.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Vocab {
    /// add/sub/mul/and/or with small constants.
    Basic = 1,
    /// Basic plus xor/shift/rotate.
    Extended = 2,
}

/// Static, canonical family parameters (part of GeneratorInstanceDigest).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Params {
    pub encoding: Encoding,
    /// Input lane width in bits (lane 0 range for unary families).
    pub width: u8,
    /// Chain length, arity, repetition count or state count, per mechanism.
    pub len: u8,
    /// Extra integer parameter, per mechanism.
    pub k: u64,
    pub vocab: Vocab,
    /// Capability pool indices (capability and composition families).
    pub caps: Vec<u16>,
    /// Visible example count.
    pub visible: u8,
}

impl Params {
    pub fn new(width: u8, len: u8) -> Params {
        Params {
            encoding: Encoding::DecimalUtf8,
            width,
            len,
            k: 0,
            vocab: Vocab::Basic,
            caps: vec![],
            visible: 8,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.u8(self.encoding as u8)
            .u8(self.width)
            .u8(self.len)
            .u64(self.k)
            .u8(self.vocab as u8);
        e.u32(self.caps.len() as u32);
        for c in &self.caps {
            e.u16(*c);
        }
        e.u8(self.visible);
        e.finish()
    }
}

/// One registered family: a mechanism plus parameters plus sealed labels.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    pub id: u32,
    pub key: String,
    pub mechanism: Mechanism,
    pub params: Params,
    pub rung: Rung,
    pub source_family: SourceFamily,
    pub provenance: Provenance,
    pub decoy: DecoyStatus,
    pub difficulty: u16,
    pub topic: &'static str,
}

/// A generated crumb: the visible half and the sealed half.
pub struct Generated {
    pub visible: VisibleCrumb,
    pub sealed: SealedCrumb,
}

/// What a mechanism produces before the framework seals it.
pub struct Draft {
    pub in_lane_bytes: u8,
    pub out_lane_bytes: u8,
    pub visible: Vec<(Vec<u64>, Vec<u64>)>,
    pub heldouts: Heldouts,
    pub known: KnownSolution,
    pub alternatives: Vec<Program>,
    pub noisy_visible: Vec<u32>,
    pub domain: (u64, u64),
    pub requires: Vec<CapabilityId>,
    pub provides: Option<CapabilityId>,
    pub verification: VerificationMethod,
}

/// Random streams handed to a mechanism.
pub struct Ctx {
    pub rule: Rng,
    pub vis: Rng,
    pub hold: Rng,
}

/// Which sample a sampler is being asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Visible,
    Interpolation,
    Extrapolation,
    Adversarial,
}

pub fn heldout_seed(family_id: u32, seed: u64) -> u64 {
    let mut e = Enc::new();
    e.bytes(b"heldout-seed").u32(family_id).u64(seed);
    let d = digest(Domain::Rng, &e.finish());
    u64::from_le_bytes(d.0[..8].try_into().unwrap())
}

/// Code identity of a mechanism at this generator version.
pub fn generator_code_digest(m: Mechanism) -> Digest {
    let mut e = Enc::new();
    e.str(m.key()).u32(m.version()).str(GENERATOR_VERSION);
    digest(Domain::GeneratorCode, &e.finish())
}

pub fn generate(spec: &FamilySpec, seed: u64) -> Generated {
    let hseed = heldout_seed(spec.id, seed);
    let mut ctx = Ctx {
        rule: Rng::named("rule", &[spec.id as u64, seed]),
        vis: Rng::named("visible", &[spec.id as u64, seed]),
        hold: Rng::named("heldout", &[spec.id as u64, hseed]),
    };
    let draft = mechanisms::run(spec, &mut ctx);
    let visible = VisibleCrumb::from_values(
        spec.params.encoding,
        draft.in_lane_bytes,
        draft.out_lane_bytes,
        &draft.visible,
        None,
    )
    .expect("mechanism produced a well-formed visible set");
    let params = spec.params.to_bytes();
    let sealed = SealedCrumb {
        schema_version: SEALED_SCHEMA_VERSION,
        crumb_digest: visible.digest(),
        family_id: spec.id,
        family_key: spec.key.clone(),
        mechanism: spec.mechanism.key().to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        generator_code_digest: generator_code_digest(spec.mechanism),
        seed,
        heldout_seed: hseed,
        generator_instance_digest: generator_instance_digest(
            spec.id,
            GENERATOR_VERSION,
            seed,
            &params,
        ),
        generation_parameters: params,
        verification_method: draft.verification,
        difficulty: spec.difficulty,
        rung: spec.rung,
        required_capabilities: draft.requires,
        provides_capability: draft.provides,
        provenance: spec.provenance,
        source_family: spec.source_family,
        truth_status: TruthStatus::Verified,
        decoy_status: spec.decoy,
        known_solution: draft.known,
        alternatives: draft.alternatives,
        noisy_visible: draft.noisy_visible,
        expected_domain: draft.domain,
        topic_internal: spec.topic.to_string(),
        heldouts: draft.heldouts,
    };
    Generated { visible, sealed }
}

// ---------------------------------------------------------------------------
// Combinators shared by mechanisms.
// ---------------------------------------------------------------------------

pub fn width_max(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// The default unary input sampler for a lane of `width` bits.
pub fn default_sample(rng: &mut Rng, stage: Stage, width: u8) -> u64 {
    let max = width_max(width);
    let common = max.min(4095);
    match stage {
        Stage::Visible | Stage::Interpolation => match rng.below(10) {
            0..=5 => rng.range(0, common),
            6..=7 => rng.range(0, max.min(31)),
            _ => rng.range(0, max),
        },
        Stage::Extrapolation => {
            if common < max {
                rng.range(common + 1, max)
            } else {
                rng.range(0, max)
            }
        }
        Stage::Adversarial => {
            let k = rng.below(width as u64) as u32;
            let p = 1u64.checked_shl(k).unwrap_or(0) & max;
            match rng.below(8) {
                0 => 0,
                1 => 1,
                2 => max,
                3 => max.wrapping_sub(1) & max,
                4 => p,
                5 => p.wrapping_sub(1) & max,
                6 => p.wrapping_add(1) & max,
                _ => rng.range(0, max),
            }
        }
    }
}

/// Sizes of the sealed held-out tiers.
pub const HELDOUT_PER_TIER: usize = 16;

/// Build visible + three held-out tiers from a truth function and a sampler.
/// Held-out inputs never repeat a visible input, and tiers never repeat each other.
pub type Pairs = Vec<(Vec<u64>, Vec<u64>)>;

pub fn build_sets(
    ctx: &mut Ctx,
    n_visible: usize,
    out_mask: u64,
    sample: &mut dyn FnMut(&mut Rng, Stage) -> Vec<u64>,
    truth: &dyn Fn(&[u64]) -> Vec<u64>,
) -> (Pairs, Heldouts) {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut visible = Vec::new();
    draw_unique(
        &mut ctx.vis,
        Stage::Visible,
        n_visible,
        &mut seen,
        sample,
        &mut |x| visible.push((x.clone(), mask_all(truth(&x), out_mask))),
    );
    let mut h = Heldouts::default();
    for (stage, dst) in [
        (Stage::Interpolation, &mut h.interpolation),
        (Stage::Extrapolation, &mut h.extrapolation),
        (Stage::Adversarial, &mut h.adversarial),
    ] {
        draw_unique(
            &mut ctx.hold,
            stage,
            HELDOUT_PER_TIER,
            &mut seen,
            sample,
            &mut |x| {
                dst.push(LaneExample {
                    output: mask_all(truth(&x), out_mask),
                    input: x,
                })
            },
        );
    }
    (visible, h)
}

fn mask_all(mut v: Vec<u64>, m: u64) -> Vec<u64> {
    for x in v.iter_mut() {
        *x &= m;
    }
    v
}

fn draw_unique(
    rng: &mut Rng,
    stage: Stage,
    n: usize,
    seen: &mut HashSet<Vec<u64>>,
    sample: &mut dyn FnMut(&mut Rng, Stage) -> Vec<u64>,
    sink: &mut dyn FnMut(Vec<u64>),
) {
    let mut got = 0;
    let mut tries = 0;
    while got < n && tries < n * 200 {
        tries += 1;
        let x = sample(rng, stage);
        if seen.insert(x.clone()) {
            sink(x);
            got += 1;
        }
    }
}

pub fn out_mask_for(encoding: Encoding, out_lane_bytes: u8) -> u64 {
    match encoding {
        Encoding::DecimalUtf8 => u64::MAX,
        Encoding::RawLe => lane_mask(out_lane_bytes),
    }
}

pub fn lane_bytes_for(encoding: Encoding, width_bits: u8) -> u8 {
    match encoding {
        Encoding::DecimalUtf8 => 0,
        Encoding::RawLe => match width_bits {
            0..=8 => 1,
            9..=16 => 2,
            17..=32 => 4,
            _ => 8,
        },
    }
}

/// Random step from a vocabulary.
pub fn random_step(rng: &mut Rng, vocab: Vocab) -> Step {
    let n_ops = match vocab {
        Vocab::Basic => 5,
        Vocab::Extended => 9,
    };
    let op = Op::from_u8(1 + rng.below(n_ops) as u8).unwrap();
    let imm = match op {
        Op::Add | Op::Sub => rng.range(1, 9),
        Op::Mul => rng.range(2, 7),
        Op::And => *rng.pick(&[0xFF, 0x0F, 0x3F, 0xFFF, 0xFFFF]),
        Op::Or => *rng.pick(&[1, 2, 4, 8, 16]),
        Op::Xor => rng.range(1, 255),
        Op::Shl | Op::Shr => rng.range(1, 4),
        Op::Rotl => rng.range(1, 15),
    };
    Step { op, imm }
}

/// Probe-behaviour is constant (the chain ignores its input).
pub fn is_constant(p: &Program) -> bool {
    let first = p.run(crate::program::PROBES[0]);
    crate::program::PROBES.iter().all(|&x| p.run(x) == first)
}

/// A chain of `len` steps with no dead step: removing any one step changes
/// its behaviour, and it is not constant.
pub fn random_chain(rng: &mut Rng, vocab: Vocab, len: usize) -> Program {
    for _ in 0..10_000 {
        let p = Program::new((0..len).map(|_| random_step(rng, vocab)).collect());
        if is_constant(&p) {
            continue;
        }
        let b = p.behavior();
        let dead = (0..len).any(|i| {
            let mut q = p.clone();
            q.steps.remove(i);
            q.behavior() == b
        });
        if !dead {
            return p;
        }
    }
    Program::of(&[(Op::Add, 1)])
}

/// Every Basic chain of length 1..=max_len (the generator's own enumeration,
/// used to find rival explanations for ambiguity and multi-fit families).
pub fn enumerate_basic(max_len: usize) -> Vec<Program> {
    let mut steps = Vec::new();
    for imm in 1..=9 {
        steps.push(Step { op: Op::Add, imm });
        steps.push(Step { op: Op::Sub, imm });
    }
    for imm in 2..=7 {
        steps.push(Step { op: Op::Mul, imm });
    }
    for imm in [0xFF, 0x0F, 0x3F, 0xFFF, 0xFFFF] {
        steps.push(Step { op: Op::And, imm });
    }
    for imm in [1, 2, 4, 8, 16] {
        steps.push(Step { op: Op::Or, imm });
    }
    let mut out: Vec<Program> = steps.iter().map(|s| Program::new(vec![*s])).collect();
    let mut frontier = out.clone();
    for _ in 1..max_len {
        let mut next = Vec::new();
        for p in &frontier {
            for s in &steps {
                let mut q = p.clone();
                q.steps.push(*s);
                next.push(q);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}
