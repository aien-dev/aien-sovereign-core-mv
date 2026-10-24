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

//! Generator mechanisms. Each is a genuinely distinct way of producing
//! observations; registered families are parameterizations of these.
//!
//! Every mechanism returns honest observations: outputs are always the exact
//! value of the hidden rule (noise families corrupt only visible outputs, and
//! say so in the sealed record). Rabbit holes deceive through *which* inputs
//! are shown, never by labeling a wrong output correct.

use super::{
    build_sets, default_sample, enumerate_basic, lane_bytes_for, out_mask_for, random_chain,
    width_max, Ctx, Draft, FamilySpec, Stage, Vocab,
};
use crate::gen::registry::capability_pool;
use crate::gen::rng::Rng;
use crate::program::{Op, Program, Step};
use crate::sealed::{Heldouts, KnownSolution, LaneExample, VerificationMethod};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mechanism {
    OpChain,
    Repeat,
    Threshold,
    ParityBranch,
    MultiLinear,
    AbsDifference,
    Recurrence1,
    Recurrence2,
    BitAutomaton,
    LaneFold,
    ByteReverse,
    BitMirror,
    SymmetricLanes,
    ModResidue,
    ModAffine,
    PopCount,
    NibblePermute,
    GraphComponents,
    HiddenGraphReach,
    PolySequence,
    Euclid,
    DigitSum,
    LatentOffset,
    LatentXorMask,
    NoisyChain,
    PartialObservation,
    CounterexampleSensitive,
    TheoryFamily,
    DecoySimpleRule,
    MultiFit,
    SpuriousField,
    PrefixDiverges,
    ElegantArbitrary,
    InsufficientEvidence,
    NoStructure,
    CapabilityPrimitive,
    CapabilityComposition,
    WrapBoundary,
}

impl Mechanism {
    pub const ALL: [Mechanism; 38] = [
        Mechanism::OpChain,
        Mechanism::Repeat,
        Mechanism::Threshold,
        Mechanism::ParityBranch,
        Mechanism::MultiLinear,
        Mechanism::AbsDifference,
        Mechanism::Recurrence1,
        Mechanism::Recurrence2,
        Mechanism::BitAutomaton,
        Mechanism::LaneFold,
        Mechanism::ByteReverse,
        Mechanism::BitMirror,
        Mechanism::SymmetricLanes,
        Mechanism::ModResidue,
        Mechanism::ModAffine,
        Mechanism::PopCount,
        Mechanism::NibblePermute,
        Mechanism::GraphComponents,
        Mechanism::HiddenGraphReach,
        Mechanism::PolySequence,
        Mechanism::Euclid,
        Mechanism::DigitSum,
        Mechanism::LatentOffset,
        Mechanism::LatentXorMask,
        Mechanism::NoisyChain,
        Mechanism::PartialObservation,
        Mechanism::CounterexampleSensitive,
        Mechanism::TheoryFamily,
        Mechanism::DecoySimpleRule,
        Mechanism::MultiFit,
        Mechanism::SpuriousField,
        Mechanism::PrefixDiverges,
        Mechanism::ElegantArbitrary,
        Mechanism::InsufficientEvidence,
        Mechanism::NoStructure,
        Mechanism::CapabilityPrimitive,
        Mechanism::CapabilityComposition,
        Mechanism::WrapBoundary,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Mechanism::OpChain => "op-chain",
            Mechanism::Repeat => "repeat",
            Mechanism::Threshold => "threshold",
            Mechanism::ParityBranch => "parity-branch",
            Mechanism::MultiLinear => "multi-linear",
            Mechanism::AbsDifference => "abs-difference",
            Mechanism::Recurrence1 => "recurrence-1",
            Mechanism::Recurrence2 => "recurrence-2",
            Mechanism::BitAutomaton => "bit-automaton",
            Mechanism::LaneFold => "lane-fold",
            Mechanism::ByteReverse => "byte-reverse",
            Mechanism::BitMirror => "bit-mirror",
            Mechanism::SymmetricLanes => "symmetric-lanes",
            Mechanism::ModResidue => "mod-residue",
            Mechanism::ModAffine => "mod-affine",
            Mechanism::PopCount => "pop-count",
            Mechanism::NibblePermute => "nibble-permute",
            Mechanism::GraphComponents => "graph-components",
            Mechanism::HiddenGraphReach => "hidden-graph-reach",
            Mechanism::PolySequence => "poly-sequence",
            Mechanism::Euclid => "euclid",
            Mechanism::DigitSum => "digit-sum",
            Mechanism::LatentOffset => "latent-offset",
            Mechanism::LatentXorMask => "latent-xor-mask",
            Mechanism::NoisyChain => "noisy-chain",
            Mechanism::PartialObservation => "partial-observation",
            Mechanism::CounterexampleSensitive => "counterexample-sensitive",
            Mechanism::TheoryFamily => "theory-family",
            Mechanism::DecoySimpleRule => "decoy-simple-rule",
            Mechanism::MultiFit => "multi-fit",
            Mechanism::SpuriousField => "spurious-field",
            Mechanism::PrefixDiverges => "prefix-diverges",
            Mechanism::ElegantArbitrary => "elegant-arbitrary",
            Mechanism::InsufficientEvidence => "insufficient-evidence",
            Mechanism::NoStructure => "no-structure",
            Mechanism::CapabilityPrimitive => "capability-primitive",
            Mechanism::CapabilityComposition => "capability-composition",
            Mechanism::WrapBoundary => "wrap-boundary",
        }
    }

    /// Per-mechanism code version (part of generator_code_digest).
    pub fn version(self) -> u32 {
        1
    }
}

pub fn run(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    match spec.mechanism {
        Mechanism::OpChain => op_chain(spec, ctx),
        Mechanism::Repeat => repeat(spec, ctx),
        Mechanism::Threshold => threshold(spec, ctx),
        Mechanism::ParityBranch => parity_branch(spec, ctx),
        Mechanism::MultiLinear => multi_linear(spec, ctx),
        Mechanism::AbsDifference => abs_difference(spec, ctx),
        Mechanism::Recurrence1 => recurrence1(spec, ctx),
        Mechanism::Recurrence2 => recurrence2(spec, ctx),
        Mechanism::BitAutomaton => bit_automaton(spec, ctx),
        Mechanism::LaneFold => lane_fold(spec, ctx),
        Mechanism::ByteReverse => byte_reverse(spec, ctx),
        Mechanism::BitMirror => bit_mirror(spec, ctx),
        Mechanism::SymmetricLanes => symmetric_lanes(spec, ctx),
        Mechanism::ModResidue => mod_residue(spec, ctx),
        Mechanism::ModAffine => mod_affine(spec, ctx),
        Mechanism::PopCount => pop_count(spec, ctx),
        Mechanism::NibblePermute => nibble_permute(spec, ctx),
        Mechanism::GraphComponents => graph_components(spec, ctx),
        Mechanism::HiddenGraphReach => hidden_graph_reach(spec, ctx),
        Mechanism::PolySequence => poly_sequence(spec, ctx),
        Mechanism::Euclid => euclid(spec, ctx),
        Mechanism::DigitSum => digit_sum(spec, ctx),
        Mechanism::LatentOffset => latent_offset(spec, ctx),
        Mechanism::LatentXorMask => latent_xor_mask(spec, ctx),
        Mechanism::NoisyChain => noisy_chain(spec, ctx),
        Mechanism::PartialObservation => partial_observation(spec, ctx),
        Mechanism::CounterexampleSensitive => counterexample_sensitive(spec, ctx),
        Mechanism::TheoryFamily => theory_family(spec, ctx),
        Mechanism::DecoySimpleRule => decoy_simple_rule(spec, ctx),
        Mechanism::MultiFit => multi_fit(spec, ctx),
        Mechanism::SpuriousField => spurious_field(spec, ctx),
        Mechanism::PrefixDiverges => prefix_diverges(spec, ctx),
        Mechanism::ElegantArbitrary => elegant_arbitrary(spec, ctx),
        Mechanism::InsufficientEvidence => insufficient_evidence(spec, ctx),
        Mechanism::NoStructure => no_structure(spec, ctx),
        Mechanism::CapabilityPrimitive => capability_primitive(spec, ctx),
        Mechanism::CapabilityComposition => capability_composition(spec, ctx),
        Mechanism::WrapBoundary => wrap_boundary(spec, ctx),
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn unary_draft(
    spec: &FamilySpec,
    ctx: &mut Ctx,
    out_width: u8,
    known: KnownSolution,
    sample: &mut dyn FnMut(&mut Rng, Stage) -> u64,
    truth: &dyn Fn(u64) -> u64,
) -> Draft {
    let p = &spec.params;
    let in_lb = lane_bytes_for(p.encoding, p.width);
    let out_lb = lane_bytes_for(p.encoding, out_width);
    let mask = out_mask_for(p.encoding, out_lb);
    let (visible, heldouts) = build_sets(
        ctx,
        p.visible as usize,
        mask,
        &mut |r, s| vec![sample(r, s)],
        &|x: &[u64]| vec![truth(x[0])],
    );
    Draft {
        in_lane_bytes: in_lb,
        out_lane_bytes: out_lb,
        visible,
        heldouts,
        known,
        alternatives: vec![],
        noisy_visible: vec![],
        domain: (0, width_max(p.width)),
        requires: vec![],
        provides: None,
        verification: VerificationMethod::ExactGeneratorEvaluation,
    }
}

fn multi_draft(
    spec: &FamilySpec,
    ctx: &mut Ctx,
    arity: usize,
    out_width: u8,
    sample: &mut dyn FnMut(&mut Rng, Stage) -> Vec<u64>,
    truth: &dyn Fn(&[u64]) -> Vec<u64>,
) -> Draft {
    let p = &spec.params;
    let in_lb = lane_bytes_for(p.encoding, p.width);
    let out_lb = lane_bytes_for(p.encoding, out_width);
    let mask = out_mask_for(p.encoding, out_lb);
    let (visible, heldouts) = build_sets(ctx, p.visible as usize, mask, sample, truth);
    debug_assert!(visible.iter().all(|(i, _)| i.len() == arity));
    Draft {
        in_lane_bytes: in_lb,
        out_lane_bytes: out_lb,
        visible,
        heldouts,
        known: KnownSolution::KnownOutsideProgramFormat,
        alternatives: vec![],
        noisy_visible: vec![],
        domain: (0, width_max(p.width)),
        requires: vec![],
        provides: None,
        verification: VerificationMethod::ExactGeneratorEvaluation,
    }
}

fn dflt(width: u8) -> impl FnMut(&mut Rng, Stage) -> u64 {
    move |r, s| default_sample(r, s, width)
}

fn lanes_sample(arity: usize, width: u8) -> impl FnMut(&mut Rng, Stage) -> Vec<u64> {
    move |r, s| (0..arity).map(|_| default_sample(r, s, width)).collect()
}

/// Re-draw every tier from a fixed input predicate (used by rabbit holes).
fn sets_from_pools(
    ctx: &mut Ctx,
    n_visible: usize,
    visible_pool: &[u64],
    interp_pool: &[u64],
    extrap_pool: &[u64],
    adversarial_pool: &[u64],
    truth: &dyn Fn(u64) -> u64,
) -> (super::Pairs, Heldouts) {
    let mut used = HashSet::new();
    let take = |rng: &mut Rng, pool: &[u64], n: usize, used: &mut HashSet<u64>| -> Vec<u64> {
        let mut idx: Vec<usize> = (0..pool.len()).collect();
        rng.shuffle(&mut idx);
        let mut out = Vec::new();
        for i in idx {
            if out.len() == n {
                break;
            }
            if used.insert(pool[i]) {
                out.push(pool[i]);
            }
        }
        out
    };
    let vis = take(&mut ctx.vis, visible_pool, n_visible, &mut used);
    let mk = |xs: Vec<u64>| {
        xs.into_iter()
            .map(|x| LaneExample {
                input: vec![x],
                output: vec![truth(x)],
            })
            .collect()
    };
    let visible = vis.iter().map(|&x| (vec![x], vec![truth(x)])).collect();
    let h = Heldouts {
        interpolation: mk(take(
            &mut ctx.hold,
            interp_pool,
            super::HELDOUT_PER_TIER,
            &mut used,
        )),
        extrapolation: mk(take(
            &mut ctx.hold,
            extrap_pool,
            super::HELDOUT_PER_TIER,
            &mut used,
        )),
        adversarial: mk(take(
            &mut ctx.hold,
            adversarial_pool,
            super::HELDOUT_PER_TIER,
            &mut used,
        )),
    };
    (visible, h)
}

fn pool(rng: &mut Rng, n: usize, stage: Stage, width: u8, keep: &dyn Fn(u64) -> bool) -> Vec<u64> {
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < n && tries < n * 400 {
        tries += 1;
        let x = default_sample(rng, stage, width);
        if keep(x) {
            out.push(x);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// single / composed transformations
// ---------------------------------------------------------------------------

fn op_chain(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let prog = random_chain(&mut ctx.rule, spec.params.vocab, spec.params.len as usize);
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    )
}

fn repeat(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let glen = 1 + ctx.rule.below(2) as usize;
    let g = random_chain(&mut ctx.rule, spec.params.vocab, glen);
    let mut prog = Program::default();
    for _ in 0..spec.params.len {
        prog = prog.then(&g);
    }
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    )
}

fn wrap_boundary(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // A chain whose behaviour is dominated by wrap-around at the lane width.
    let w = spec.params.width;
    let mut prog = random_chain(&mut ctx.rule, Vocab::Basic, spec.params.len as usize);
    prog.steps.push(Step {
        op: Op::And,
        imm: width_max(w),
    });
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        w,
        KnownSolution::Known(prog),
        &mut dflt(w),
        &move |x| t.run(x),
    )
}

// ---------------------------------------------------------------------------
// conditional
// ---------------------------------------------------------------------------

fn threshold(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let w = spec.params.width;
    let t = ctx.rule.range(8, width_max(w).min(4095));
    let a = random_chain(&mut ctx.rule, Vocab::Basic, 1);
    let b = random_chain(&mut ctx.rule, Vocab::Basic, 1);
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(w),
        &move |x| {
            if x < t {
                a.run(x)
            } else {
                b.run(x)
            }
        },
    )
}

fn parity_branch(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let m = 2 + ctx.rule.below(spec.params.k.max(1));
    let a = random_chain(&mut ctx.rule, Vocab::Basic, 1);
    let b = random_chain(&mut ctx.rule, Vocab::Basic, 1);
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(spec.params.width),
        &move |x| {
            if x % m == 0 {
                a.run(x)
            } else {
                b.run(x)
            }
        },
    )
}

fn counterexample_sensitive(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // Almost always rule R, but a sparse residue class is shifted. Exactly one
    // visible example sits in the exceptional class: ignoring it is an error.
    let w = spec.params.width;
    let base = random_chain(&mut ctx.rule, Vocab::Basic, 2);
    let m = ctx.rule.range(11, 29);
    let r = ctx.rule.below(m);
    let delta = ctx.rule.range(1, 7);
    let truth = {
        let base = base.clone();
        move |x: u64| {
            if x % m == r {
                base.run(x).wrapping_add(delta)
            } else {
                base.run(x)
            }
        }
    };
    let n = spec.params.visible as usize;
    let normal = pool(&mut ctx.vis, 400, Stage::Visible, w, &|x| x % m != r);
    let special = pool(&mut ctx.vis, 50, Stage::Visible, w, &|x| x % m == r);
    let mut vis_pool: Vec<u64> = normal.iter().take(n - 1).cloned().collect();
    if let Some(s) = special.first() {
        vis_pool.push(*s);
    }
    let interp = pool(&mut ctx.hold, 200, Stage::Interpolation, w, &|_| true);
    let extrap = pool(&mut ctx.hold, 200, Stage::Extrapolation, w, &|_| true);
    let adv = pool(&mut ctx.hold, 200, Stage::Interpolation, w, &|x| x % m == r);
    let (visible, heldouts) = sets_from_pools(ctx, n, &vis_pool, &interp, &extrap, &adv, &truth);
    Draft {
        in_lane_bytes: 0,
        out_lane_bytes: 0,
        visible,
        heldouts,
        known: KnownSolution::KnownOutsideProgramFormat,
        alternatives: vec![base],
        noisy_visible: vec![],
        domain: (0, width_max(w)),
        requires: vec![],
        provides: None,
        verification: VerificationMethod::ExactGeneratorEvaluation,
    }
}

// ---------------------------------------------------------------------------
// multi-input relations
// ---------------------------------------------------------------------------

fn multi_linear(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let arity = spec.params.len.clamp(2, 4) as usize;
    let coef: Vec<u64> = (0..arity).map(|_| ctx.rule.range(1, 9)).collect();
    let d = ctx.rule.range(0, 20);
    multi_draft(
        spec,
        ctx,
        arity,
        64,
        &mut lanes_sample(arity, spec.params.width),
        &move |x| {
            vec![x
                .iter()
                .zip(&coef)
                .fold(d, |acc, (v, c)| acc.wrapping_add(v.wrapping_mul(*c)))]
        },
    )
}

fn abs_difference(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let s = ctx.rule.range(0, 5);
    multi_draft(
        spec,
        ctx,
        2,
        64,
        &mut lanes_sample(2, spec.params.width),
        &move |x| vec![x[0].abs_diff(x[1]) << s],
    )
}

fn symmetric_lanes(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // Invariant under any permutation of the input lanes.
    let arity = spec.params.len.clamp(2, 4) as usize;
    let c = ctx.rule.range(1, 9);
    multi_draft(
        spec,
        ctx,
        arity,
        64,
        &mut lanes_sample(arity, spec.params.width),
        &move |x| {
            let s: u64 = x.iter().fold(0u64, |a, v| a.wrapping_add(*v));
            let q: u64 = x
                .iter()
                .fold(0u64, |a, v| a.wrapping_add(v.wrapping_mul(*v)));
            vec![q.wrapping_add(s.wrapping_mul(c))]
        },
    )
}

fn theory_family(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // One law, many parameterizations: lane 0 is a parameter p, lane 1 is x.
    let a = ctx.rule.range(1, 5);
    let b = ctx.rule.range(0, 9);
    let w = spec.params.width;
    multi_draft(
        spec,
        ctx,
        2,
        64,
        &mut move |r, s| vec![r.range(1, 12), default_sample(r, s, w)],
        &move |x| {
            vec![x[0]
                .wrapping_mul(x[1])
                .wrapping_add(a.wrapping_mul(x[0]))
                .wrapping_add(b)]
        },
    )
}

fn euclid(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let _ = ctx.rule.next_u64();
    multi_draft(
        spec,
        ctx,
        2,
        64,
        &mut lanes_sample(2, spec.params.width),
        &|x| {
            let (mut a, mut b) = (x[0], x[1]);
            while b != 0 {
                let t = a % b;
                a = b;
                b = t;
            }
            vec![a]
        },
    )
}

// ---------------------------------------------------------------------------
// recurrence / sequences
// ---------------------------------------------------------------------------

fn seq_sample(max_n: u64) -> impl FnMut(&mut Rng, Stage) -> u64 {
    move |r, s| match s {
        Stage::Visible | Stage::Interpolation => r.range(0, max_n),
        Stage::Extrapolation | Stage::Adversarial => r.range(max_n + 1, max_n * 4),
    }
}

fn recurrence1(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let (a0, p, q) = (
        ctx.rule.range(0, 9),
        ctx.rule.range(2, 5),
        ctx.rule.range(0, 9),
    );
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::KnownOutsideProgramFormat,
        &mut seq_sample(24),
        &move |n| (0..n).fold(a0, |a, _| a.wrapping_mul(p).wrapping_add(q)),
    )
}

fn recurrence2(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let (s0, s1, p, q) = (
        ctx.rule.range(0, 5),
        ctx.rule.range(1, 5),
        ctx.rule.range(1, 3),
        ctx.rule.range(1, 3),
    );
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::KnownOutsideProgramFormat,
        &mut seq_sample(24),
        &move |n| {
            let (mut a, mut b) = (s0, s1);
            for _ in 0..n {
                let c = b.wrapping_mul(p).wrapping_add(a.wrapping_mul(q));
                a = b;
                b = c;
            }
            a
        },
    )
}

fn poly_sequence(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let (a, b, c) = (
        ctx.rule.range(1, 5),
        ctx.rule.range(0, 9),
        ctx.rule.range(0, 9),
    );
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::KnownOutsideProgramFormat,
        &mut seq_sample(40),
        &move |n| {
            a.wrapping_mul(n)
                .wrapping_mul(n)
                .wrapping_add(b.wrapping_mul(n))
                .wrapping_add(c)
        },
    )
}

fn prefix_diverges(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // f(n) = s(n) + c * prod_{i<K}(n - i): identical to the simple rule s on
    // the visible prefix n < K, different from n = K on.
    let s = random_chain(&mut ctx.rule, Vocab::Basic, 2);
    let k = spec.params.visible as u64;
    let c = ctx.rule.range(1, 3);
    let truth = {
        let s = s.clone();
        move |n: u64| {
            let prod = (0..k).fold(1u64, |acc, i| acc.wrapping_mul(n.wrapping_sub(i)));
            s.run(n).wrapping_add(c.wrapping_mul(prod))
        }
    };
    let vis_pool: Vec<u64> = (0..k).collect();
    let interp: Vec<u64> = (k..k + 8).collect();
    let extrap: Vec<u64> = (k + 8..k + 64).collect();
    let adv: Vec<u64> = (k..k + 3).chain([k * 4, k * 16]).collect();
    let (visible, heldouts) =
        sets_from_pools(ctx, k as usize, &vis_pool, &interp, &extrap, &adv, &truth);
    Draft {
        in_lane_bytes: 0,
        out_lane_bytes: 0,
        visible,
        heldouts,
        known: KnownSolution::KnownOutsideProgramFormat,
        alternatives: vec![s],
        noisy_visible: vec![],
        domain: (0, k + 64),
        requires: vec![],
        provides: None,
        verification: VerificationMethod::ExactGeneratorEvaluation,
    }
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

fn bit_automaton(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let states = spec.params.len.clamp(2, 8) as usize;
    let table: Vec<[u8; 2]> = (0..states)
        .map(|_| {
            [
                ctx.rule.below(states as u64) as u8,
                ctx.rule.below(states as u64) as u8,
            ]
        })
        .collect();
    let w = spec.params.width;
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(w),
        &move |x| {
            let mut s = 0usize;
            for i in 0..w {
                s = table[s][((x >> i) & 1) as usize] as usize;
            }
            s as u64
        },
    )
}

fn lane_fold(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let arity = spec.params.len.clamp(2, 8) as usize;
    let m = ctx.rule.range(2, 5);
    let a = ctx.rule.range(0, 9);
    multi_draft(
        spec,
        ctx,
        arity,
        64,
        &mut lanes_sample(arity, spec.params.width),
        &move |x| {
            vec![x
                .iter()
                .fold(a, |acc, v| acc.wrapping_mul(m).wrapping_add(*v))]
        },
    )
}

// ---------------------------------------------------------------------------
// symmetry / permutation / bits
// ---------------------------------------------------------------------------

fn byte_reverse(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let _ = ctx.rule.next_u64();
    let w = spec.params.width;
    let bytes = (w as u32).div_ceil(8);
    unary_draft(
        spec,
        ctx,
        w,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(w),
        &move |x| x.swap_bytes() >> (64 - 8 * bytes),
    )
}

fn bit_mirror(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let _ = ctx.rule.next_u64();
    let w = spec.params.width as u32;
    unary_draft(
        spec,
        ctx,
        w as u8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(w as u8),
        &move |x| x.reverse_bits() >> (64 - w),
    )
}

fn nibble_permute(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let mut perm: Vec<u32> = (0..4).collect();
    loop {
        ctx.rule.shuffle(&mut perm);
        if perm != [0, 1, 2, 3] {
            break;
        }
    }
    unary_draft(
        spec,
        ctx,
        16,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(16),
        &move |x| {
            (0..4).fold(0u64, |acc, i| {
                acc | (((x >> (4 * i)) & 0xF) << (4 * perm[i as usize]))
            })
        },
    )
}

fn pop_count(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let c = ctx.rule.range(0, 5);
    let w = spec.params.width;
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(w),
        &move |x| x.count_ones() as u64 + c,
    )
}

// ---------------------------------------------------------------------------
// modularity
// ---------------------------------------------------------------------------

fn mod_residue(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let m = ctx.rule.range(3, 97) | 1;
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(spec.params.width),
        &move |x| x % m,
    )
}

fn mod_affine(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let bits = ctx.rule.range(4, 12);
    let prog = Program::of(&[
        (Op::Mul, ctx.rule.range(2, 7)),
        (Op::Add, ctx.rule.range(1, 9)),
        (Op::And, (1u64 << bits) - 1),
    ]);
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        bits as u8,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    )
}

// ---------------------------------------------------------------------------
// graphs and algorithms
// ---------------------------------------------------------------------------

fn graph_components(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let _ = ctx.rule.next_u64();
    let v = spec.params.len.clamp(3, 6) as usize;
    let edges: Vec<(usize, usize)> = (0..v)
        .flat_map(|a| (a + 1..v).map(move |b| (a, b)))
        .collect();
    let e = edges.len() as u8;
    let mut sample = move |r: &mut Rng, _s: Stage| r.below(1u64 << e);
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut sample,
        &move |mask| {
            let mut parent: Vec<usize> = (0..v).collect();
            fn find(p: &mut [usize], x: usize) -> usize {
                let mut x = x;
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for (i, &(a, b)) in edges.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent[ra] = rb;
                }
            }
            (0..v).filter(|&x| find(&mut parent, x) == x).count() as u64
        },
    )
}

fn hidden_graph_reach(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // A hidden directed graph on N nodes; input = start node, output = reachable-set bitmask.
    let n = spec.params.len.clamp(4, 16) as u64;
    let adj: Vec<u64> = (0..n)
        .map(|_| ctx.rule.below(1 << n) & ctx.rule.below(1 << n))
        .collect();
    let mut sample = move |r: &mut Rng, _s: Stage| r.below(n);
    let mut d = unary_draft(
        spec,
        ctx,
        16,
        KnownSolution::KnownOutsideProgramFormat,
        &mut sample,
        &move |s| {
            let mut seen = 1u64 << s;
            let mut frontier = seen;
            while frontier != 0 {
                let mut next = 0;
                for i in 0..n {
                    if frontier >> i & 1 == 1 {
                        next |= adj[i as usize];
                    }
                }
                frontier = next & !seen;
                seen |= next;
            }
            seen
        },
    );
    d.domain = (0, n - 1);
    d
}

fn digit_sum(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let _ = ctx.rule.next_u64();
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::KnownOutsideProgramFormat,
        &mut dflt(spec.params.width),
        &|mut x| {
            let mut s = 0;
            while x > 0 {
                s += x % 10;
                x /= 10;
            }
            s
        },
    )
}

// ---------------------------------------------------------------------------
// latent variables, noise, partial observation
// ---------------------------------------------------------------------------

fn latent_offset(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let prog = Program::of(&[
        (Op::Mul, ctx.rule.range(2, 5)),
        (Op::Add, ctx.rule.range(6, 40)),
    ]);
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    )
}

fn latent_xor_mask(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let w = spec.params.width;
    let prog = Program::of(&[
        (Op::Xor, ctx.rule.below(width_max(w))),
        (Op::And, width_max(w)),
    ]);
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        w,
        KnownSolution::Known(prog),
        &mut dflt(w),
        &move |x| t.run(x),
    )
}

fn noisy_chain(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // Held-outs are exact; a minority of visible outputs carry a one-bit flip.
    let prog = random_chain(&mut ctx.rule, Vocab::Basic, spec.params.len as usize);
    let t = prog.clone();
    let mut d = unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    );
    let n = d.visible.len();
    let flips = (n / 8).max(1);
    let mut idx: Vec<usize> = (0..n).collect();
    ctx.rule.shuffle(&mut idx);
    let mut noisy: Vec<u32> = idx[..flips].iter().map(|&i| i as u32).collect();
    noisy.sort_unstable();
    for &i in &noisy {
        let bit = ctx.rule.below(6);
        d.visible[i as usize].1[0] ^= 1u64 << bit;
    }
    d.noisy_visible = noisy;
    d
}

fn partial_observation(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // Only the low part of a hidden computation is observed.
    let inner = random_chain(&mut ctx.rule, Vocab::Basic, 2);
    let keep = *ctx.rule.pick(&[0x0Fu64, 0xFF]);
    let prog = inner.then(&Program::of(&[(Op::And, keep)]));
    let t = prog.clone();
    unary_draft(
        spec,
        ctx,
        8,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    )
}

// ---------------------------------------------------------------------------
// rabbit holes
// ---------------------------------------------------------------------------

fn decoy_simple_rule(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // A: truth = (a*x + b) & M. Visible inputs keep a*x+b <= M, so the
    // simpler rule a*x + b fits every visible example.
    let (a, b) = (ctx.rule.range(2, 7), ctx.rule.range(1, 9));
    let m = *ctx.rule.pick(&[0xFFu64, 0xFFF]);
    let truth_p = Program::of(&[(Op::Mul, a), (Op::Add, b), (Op::And, m)]);
    let decoy = Program::of(&[(Op::Mul, a), (Op::Add, b)]);
    let w = spec.params.width;
    let fits = |x: u64| a.wrapping_mul(x).wrapping_add(b) <= m;
    let vis_pool = pool(&mut ctx.vis, 400, Stage::Visible, w, &|x| fits(x));
    let interp = pool(&mut ctx.hold, 400, Stage::Visible, w, &|x| fits(x));
    let extrap = pool(&mut ctx.hold, 400, Stage::Extrapolation, w, &|x| !fits(x));
    let adv = pool(&mut ctx.hold, 400, Stage::Visible, w, &|x| !fits(x));
    let t = truth_p.clone();
    let (visible, heldouts) = sets_from_pools(
        ctx,
        spec.params.visible as usize,
        &vis_pool,
        &interp,
        &extrap,
        &adv,
        &move |x| t.run(x),
    );
    rabbit(
        visible,
        heldouts,
        KnownSolution::Known(truth_p),
        vec![decoy],
        w,
    )
}

fn multi_fit(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // B: truth T and rival R agree on the visible inputs, disagree on held-outs.
    let w = spec.params.width;
    let n = spec.params.visible as usize;
    let cands = enumerate_basic(2);
    for _ in 0..200 {
        let t = random_chain(&mut ctx.rule, Vocab::Basic, 2);
        let tb = t.behavior();
        let probe = pool(&mut ctx.rule, 256, Stage::Visible, w, &|_| true);
        let mut best: Option<(usize, &Program)> = None;
        for r in &cands {
            if r.behavior() == tb {
                continue;
            }
            let agree = probe.iter().filter(|&&x| r.run(x) == t.run(x)).count();
            if agree >= n * 3 && agree < probe.len() && best.is_none_or(|(b, _)| agree > b) {
                best = Some((agree, r));
            }
        }
        if let Some((_, r)) = best {
            let r = r.clone();
            let (tc, rc) = (t.clone(), r.clone());
            let agree = move |x: u64| tc.run(x) == rc.run(x);
            let vis_pool = pool(&mut ctx.vis, 600, Stage::Visible, w, &agree);
            let interp = pool(&mut ctx.hold, 600, Stage::Interpolation, w, &|_| true);
            let extrap = pool(&mut ctx.hold, 600, Stage::Extrapolation, w, &|_| true);
            let adv = pool(&mut ctx.hold, 600, Stage::Interpolation, w, &|x| !agree(x));
            if vis_pool.len() < n || adv.is_empty() {
                continue;
            }
            let tt = t.clone();
            let (visible, heldouts) =
                sets_from_pools(ctx, n, &vis_pool, &interp, &extrap, &adv, &move |x| {
                    tt.run(x)
                });
            return rabbit(visible, heldouts, KnownSolution::Known(t), vec![r], w);
        }
    }
    op_chain(spec, ctx)
}

fn spurious_field(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // C: x = a | (z << 16). The answer depends only on a, but in the visible
    // sample the high field z equals the answer, so reading z also fits.
    let f = random_chain(&mut ctx.rule, Vocab::Basic, 2);
    let truth_p = Program::of(&[(Op::And, 0xFFFF)]).then(&f);
    let decoy = Program::of(&[(Op::Shr, 16)]);
    let mk_vis = |rng: &mut Rng, n: usize| -> Vec<u64> {
        let mut out = Vec::new();
        let mut tries = 0;
        while out.len() < n && tries < 10_000 {
            tries += 1;
            let a = rng.range(0, 0xFFFF);
            let y = f.run(a);
            if y <= 0xFFFF {
                out.push(a | (y << 16));
            }
        }
        out
    };
    let vis_pool = mk_vis(&mut ctx.vis, 200);
    let interp = mk_vis(&mut ctx.hold, 200);
    let extrap: Vec<u64> = (0..200)
        .map(|_| ctx.hold.range(0, 0xFFFF) | (ctx.hold.range(0, 0xFFFF) << 16))
        .collect();
    let adv: Vec<u64> = (0..200)
        .map(|_| ctx.hold.range(0, 0xFFFF) | (ctx.hold.range(0, 0xFFFF) << 32))
        .collect();
    let t = truth_p.clone();
    let (visible, heldouts) = sets_from_pools(
        ctx,
        spec.params.visible as usize,
        &vis_pool,
        &interp,
        &extrap,
        &adv,
        &move |x| t.run(x),
    );
    rabbit(
        visible,
        heldouts,
        KnownSolution::Known(truth_p),
        vec![decoy],
        48,
    )
}

fn elegant_arbitrary(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // E: an arbitrary 256-entry table whose first entries follow a neat chain.
    let s = random_chain(&mut ctx.rule, Vocab::Basic, 2);
    let k = spec.params.visible as u64;
    let table: Vec<u64> = (0..256u64)
        .map(|i| {
            if i < k {
                s.run(i)
            } else {
                ctx.rule.below(1 << 20)
            }
        })
        .collect();
    let vis_pool: Vec<u64> = (0..k).collect();
    let rest: Vec<u64> = (k..256).collect();
    let t2 = table.clone();
    let (visible, heldouts) =
        sets_from_pools(ctx, k as usize, &vis_pool, &rest, &rest, &rest, &move |x| {
            t2[x as usize]
        });
    let mut d = rabbit(
        visible,
        heldouts,
        KnownSolution::NoCompactStructure,
        vec![s],
        8,
    );
    d.domain = (0, 255);
    d
}

fn insufficient_evidence(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    // F: too few examples to separate several short explanations.
    let w = spec.params.width;
    let n = spec.params.visible.clamp(2, 4) as usize;
    let cands = enumerate_basic(2);
    for _ in 0..200 {
        let t = random_chain(&mut ctx.rule, Vocab::Basic, 2);
        let xs = pool(&mut ctx.vis, n, Stage::Visible, w, &|x| x < 64);
        if xs.len() < n {
            continue;
        }
        let mut rivals: Vec<Program> = Vec::new();
        let mut seen = HashSet::new();
        seen.insert(t.behavior());
        for c in &cands {
            if xs.iter().all(|&x| c.run(x) == t.run(x)) && seen.insert(c.behavior()) {
                rivals.push(c.clone());
            }
        }
        if rivals.len() >= 2 {
            rivals.truncate(8);
            let tt = t.clone();
            let interp = pool(&mut ctx.hold, 200, Stage::Interpolation, w, &|_| true);
            let extrap = pool(&mut ctx.hold, 200, Stage::Extrapolation, w, &|_| true);
            // Every rival gets at least one held-out that separates it from the truth.
            let mut adv: Vec<u64> = Vec::new();
            for r in &rivals {
                let sep = pool(&mut ctx.hold, 1, Stage::Interpolation, w, &|x| {
                    r.run(x) != tt.run(x) && !xs.contains(&x)
                });
                adv.extend(sep);
            }
            if adv.len() < rivals.len() {
                continue;
            }
            let t3 = t.clone();
            let (visible, heldouts) =
                sets_from_pools(ctx, n, &xs, &interp, &extrap, &adv, &move |x| t3.run(x));
            return rabbit(visible, heldouts, KnownSolution::Known(t), rivals, w);
        }
    }
    op_chain(spec, ctx)
}

fn no_structure(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let table: Vec<u64> = (0..256).map(|_| ctx.rule.below(1 << 24)).collect();
    let mut sample = |r: &mut Rng, _s: Stage| r.below(256);
    let mut d = unary_draft(
        spec,
        ctx,
        24,
        KnownSolution::NoCompactStructure,
        &mut sample,
        &move |x| table[x as usize],
    );
    d.domain = (0, 255);
    d
}

fn rabbit(
    visible: Vec<(Vec<u64>, Vec<u64>)>,
    heldouts: Heldouts,
    known: KnownSolution,
    alternatives: Vec<Program>,
    width: u8,
) -> Draft {
    Draft {
        in_lane_bytes: 0,
        out_lane_bytes: 0,
        visible,
        heldouts,
        known,
        alternatives,
        noisy_visible: vec![],
        domain: (0, width_max(width)),
        requires: vec![],
        provides: None,
        verification: VerificationMethod::ExactGeneratorEvaluation,
    }
}

// ---------------------------------------------------------------------------
// capabilities and their compositions (the breadcrumb trail)
// ---------------------------------------------------------------------------

fn capability_primitive(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let pool = capability_pool();
    let cap = &pool[spec.params.caps[0] as usize];
    let prog = cap.program.clone();
    let t = prog.clone();
    let mut d = unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    );
    d.provides = Some(cap.id);
    d
}

fn capability_composition(spec: &FamilySpec, ctx: &mut Ctx) -> Draft {
    let pool = capability_pool();
    let mut prog = Program::default();
    let mut requires = Vec::new();
    for &c in &spec.params.caps {
        prog = prog.then(&pool[c as usize].program);
        requires.push(pool[c as usize].id);
    }
    let t = prog.clone();
    let mut d = unary_draft(
        spec,
        ctx,
        64,
        KnownSolution::Known(prog),
        &mut dflt(spec.params.width),
        &move |x| t.run(x),
    );
    d.requires = requires;
    d
}

/// Count how many visible examples a program explains (used by tests).
pub fn visible_agreement(p: &Program, visible: &[(Vec<u64>, Vec<u64>)]) -> usize {
    visible.iter().filter(|(i, o)| p.run(i[0]) == o[0]).count()
}
