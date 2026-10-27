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

//! The v1 family registry: 38 mechanisms parameterized into 100+ registered
//! family ids, plus the capability pool the breadcrumb trail is built from.
//!
//! Family ids are assigned in registration order and are part of every
//! GeneratorInstanceDigest; the order below is frozen for registry v1.

use super::mechanisms::Mechanism;
use super::rng::Rng;
use super::{random_chain, FamilySpec, Params, Vocab};
use crate::canon::Enc;
use crate::digest::{digest, Domain};
use crate::program::Program;
use crate::provenance::Provenance;
use crate::sealed::{AdversarialClass, CapabilityId, DecoyStatus, Rung, SourceFamily};
use crate::visible::Encoding;
use std::collections::HashSet;
use std::sync::OnceLock;

pub const REGISTRY_VERSION: &str = "crumbs-registry/1.0.0";
pub const CAPABILITY_POOL_SIZE: usize = 12;

/// A hidden capability: a short transformation that later crumbs quietly require.
#[derive(Clone, Debug)]
pub struct Capability {
    pub index: u16,
    pub id: CapabilityId,
    pub program: Program,
}

/// Twelve distinct two-step capabilities drawn deterministically from the
/// Basic vocabulary. Deliberately not tuned to any learner.
pub fn capability_pool() -> &'static [Capability] {
    static POOL: OnceLock<Vec<Capability>> = OnceLock::new();
    POOL.get_or_init(|| {
        let mut rng = Rng::named("capability-pool", &[1]);
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        while out.len() < CAPABILITY_POOL_SIZE {
            let p = random_chain(&mut rng, Vocab::Basic, 2);
            if !seen.insert(p.behavior()) {
                continue;
            }
            let mut e = Enc::new();
            e.str(REGISTRY_VERSION).blob(&p.to_bytes());
            out.push(Capability {
                index: out.len() as u16,
                id: digest(Domain::Capability, &e.finish()),
                program: p,
            });
        }
        out
    })
}

pub struct Registry {
    pub families: Vec<FamilySpec>,
}

impl Registry {
    pub fn get(&self, id: u32) -> Option<&FamilySpec> {
        self.families.iter().find(|f| f.id == id)
    }
    pub fn by_key(&self, key: &str) -> Option<&FamilySpec> {
        self.families.iter().find(|f| f.key == key)
    }
}

/// Capability pairs used by composition families (ordered: first applied first).
pub const COMPOSITION_PAIRS: [(u16, u16); 24] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 0),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (10, 0),
    (11, 1),
];
pub const COMPOSITION_TRIPLES: [(u16, u16, u16); 8] = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (9, 0, 4),
];

pub fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(build)
}

struct B {
    fams: Vec<FamilySpec>,
}

impl B {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        key: String,
        mechanism: Mechanism,
        params: Params,
        rung: Rung,
        source: SourceFamily,
        provenance: Provenance,
        decoy: DecoyStatus,
        difficulty: u16,
        topic: &'static str,
    ) {
        let id = self.fams.len() as u32 + 1;
        self.fams.push(FamilySpec {
            id,
            key,
            mechanism,
            params,
            rung,
            source_family: source,
            provenance,
            decoy,
            difficulty,
            topic,
        });
    }
}

fn p(width: u8, len: u8) -> Params {
    Params::new(width, len)
}

fn build() -> Registry {
    use Mechanism as M;
    use SourceFamily as S;
    let syn = Provenance::synthetic();
    let adv = Provenance::adversarial();
    let hum = Provenance::human_theory();
    let clean = DecoyStatus::Clean;
    let mut b = B { fams: Vec::new() };

    // Single and chained transformations across widths and vocabularies.
    for (len, rung) in [
        (1u8, Rung::SingleTransformation),
        (2, Rung::TwoTransformations),
        (3, Rung::MultiStepAlgorithm),
        (4, Rung::Composition),
    ] {
        for width in [8u8, 16, 32] {
            b.add(
                format!("chain-basic-l{len}-w{width}"),
                M::OpChain,
                p(width, len),
                rung,
                S::Capability,
                syn,
                clean,
                len as u16 * 10,
                "wrapping arithmetic chain",
            );
        }
    }
    for (len, rung) in [
        (1u8, Rung::SingleTransformation),
        (2, Rung::TwoTransformations),
        (3, Rung::MultiStepAlgorithm),
    ] {
        let mut prm = p(32, len);
        prm.vocab = Vocab::Extended;
        b.add(
            format!("chain-ext-l{len}-w32"),
            M::OpChain,
            prm.clone(),
            rung,
            S::Capability,
            syn,
            clean,
            15 + len as u16 * 10,
            "bitwise chain",
        );
        prm.encoding = Encoding::RawLe;
        b.add(
            format!("chain-ext-l{len}-w32-raw"),
            M::OpChain,
            prm,
            rung,
            S::Capability,
            syn,
            clean,
            15 + len as u16 * 10,
            "bitwise chain, raw bytes",
        );
    }
    for k in [2u8, 3, 4] {
        b.add(
            format!("repeat-k{k}"),
            M::Repeat,
            p(16, k),
            Rung::RepeatedTransformation,
            S::Capability,
            syn,
            clean,
            20 + k as u16 * 5,
            "repeated transformation",
        );
    }
    for w in [8u8, 16] {
        b.add(
            format!("wrap-w{w}"),
            M::WrapBoundary,
            p(w, 2),
            Rung::Modularity,
            S::Capability,
            syn,
            clean,
            30,
            "wrap at lane width",
        );
    }
    b.add(
        "threshold-w12".into(),
        M::Threshold,
        p(12, 1),
        Rung::ConditionalTransformation,
        S::Capability,
        syn,
        clean,
        35,
        "piecewise rule",
    );
    b.add(
        "threshold-w16".into(),
        M::Threshold,
        p(16, 1),
        Rung::ConditionalTransformation,
        S::Capability,
        syn,
        clean,
        40,
        "piecewise rule",
    );
    for k in [1u64, 4] {
        let mut prm = p(16, 1);
        prm.k = k;
        b.add(
            format!("parity-branch-k{k}"),
            M::ParityBranch,
            prm,
            Rung::ConditionalTransformation,
            S::Capability,
            syn,
            clean,
            35,
            "residue-class branch",
        );
    }
    for a in [2u8, 3, 4] {
        b.add(
            format!("multi-linear-a{a}"),
            M::MultiLinear,
            p(12, a),
            Rung::MultiInputRelation,
            S::Capability,
            syn,
            clean,
            30 + a as u16 * 5,
            "weighted lane sum",
        );
    }
    b.add(
        "abs-difference".into(),
        M::AbsDifference,
        p(16, 2),
        Rung::MultiInputRelation,
        S::Capability,
        syn,
        clean,
        30,
        "lane distance",
    );
    for a in [2u8, 3] {
        b.add(
            format!("symmetric-lanes-a{a}"),
            M::SymmetricLanes,
            p(10, a),
            Rung::Invariance,
            S::Capability,
            syn,
            clean,
            45,
            "permutation-invariant lanes",
        );
    }
    b.add(
        "theory-family".into(),
        M::TheoryFamily,
        p(12, 2),
        Rung::TheoryFormation,
        S::Capability,
        syn,
        clean,
        55,
        "one law, many parameters",
    );
    b.add(
        "euclid".into(),
        M::Euclid,
        p(16, 2),
        Rung::MultiStepAlgorithm,
        S::Capability,
        hum,
        clean,
        60,
        "greatest common divisor",
    );
    b.add(
        "recurrence-1".into(),
        M::Recurrence1,
        p(8, 1),
        Rung::Recurrence,
        S::Capability,
        syn,
        clean,
        45,
        "first-order recurrence",
    );
    b.add(
        "recurrence-2".into(),
        M::Recurrence2,
        p(8, 2),
        Rung::Recurrence,
        S::Capability,
        syn,
        clean,
        55,
        "second-order recurrence",
    );
    b.add(
        "poly-sequence".into(),
        M::PolySequence,
        p(8, 2),
        Rung::Recurrence,
        S::Capability,
        syn,
        clean,
        45,
        "quadratic sequence",
    );
    for (s, w) in [(2u8, 8u8), (3, 12), (4, 16)] {
        b.add(
            format!("bit-automaton-s{s}-w{w}"),
            M::BitAutomaton,
            p(w, s),
            Rung::State,
            S::Capability,
            syn,
            clean,
            50 + s as u16 * 5,
            "finite automaton over bits",
        );
    }
    for a in [3u8, 5] {
        b.add(
            format!("lane-fold-a{a}"),
            M::LaneFold,
            p(8, a),
            Rung::State,
            S::Capability,
            syn,
            clean,
            50,
            "running fold over lanes",
        );
    }
    for w in [16u8, 32] {
        b.add(
            format!("byte-reverse-w{w}"),
            M::ByteReverse,
            p(w, 1),
            Rung::Symmetry,
            S::Capability,
            syn,
            clean,
            45,
            "byte order reversal",
        );
    }
    for w in [8u8, 16] {
        b.add(
            format!("bit-mirror-w{w}"),
            M::BitMirror,
            p(w, 1),
            Rung::Symmetry,
            S::Capability,
            syn,
            clean,
            45,
            "bit order reversal",
        );
    }
    b.add(
        "nibble-permute".into(),
        M::NibblePermute,
        p(16, 1),
        Rung::Symmetry,
        S::Capability,
        syn,
        clean,
        50,
        "fixed nibble permutation",
    );
    for w in [16u8, 32] {
        b.add(
            format!("pop-count-w{w}"),
            M::PopCount,
            p(w, 1),
            Rung::MultiStepAlgorithm,
            S::Capability,
            syn,
            clean,
            45,
            "set-bit count",
        );
    }
    for w in [12u8, 24] {
        b.add(
            format!("mod-residue-w{w}"),
            M::ModResidue,
            p(w, 1),
            Rung::Modularity,
            S::Capability,
            syn,
            clean,
            45,
            "residue",
        );
        b.add(
            format!("mod-affine-w{w}"),
            M::ModAffine,
            p(w, 3),
            Rung::Modularity,
            S::Capability,
            syn,
            clean,
            40,
            "affine map with wrap",
        );
    }
    for v in [4u8, 5] {
        b.add(
            format!("graph-components-v{v}"),
            M::GraphComponents,
            p(8, v),
            Rung::MultiStepAlgorithm,
            S::Capability,
            hum,
            clean,
            65,
            "connected components of an edge mask",
        );
    }
    for n in [12u8, 16] {
        let mut prm = p(8, n);
        prm.visible = 6;
        b.add(
            format!("hidden-graph-reach-n{n}"),
            M::HiddenGraphReach,
            prm,
            Rung::LatentVariables,
            S::Capability,
            syn,
            clean,
            60,
            "reachability in a hidden graph",
        );
    }
    b.add(
        "digit-sum".into(),
        M::DigitSum,
        p(20, 1),
        Rung::MultiStepAlgorithm,
        S::Capability,
        hum,
        clean,
        50,
        "base-ten digit sum",
    );
    for w in [16u8, 32] {
        b.add(
            format!("latent-offset-w{w}"),
            M::LatentOffset,
            p(w, 2),
            Rung::LatentVariables,
            S::Capability,
            syn,
            clean,
            40,
            "hidden constant offset",
        );
    }
    b.add(
        "latent-xor-w16".into(),
        M::LatentXorMask,
        p(16, 2),
        Rung::LatentVariables,
        S::Capability,
        syn,
        clean,
        45,
        "hidden xor key",
    );
    for w in [16u8, 32] {
        b.add(
            format!("partial-observation-w{w}"),
            M::PartialObservation,
            p(w, 2),
            Rung::PartialObservation,
            S::Capability,
            syn,
            clean,
            45,
            "only low bits observed",
        );
    }
    b.add(
        "counterexample-sensitive".into(),
        M::CounterexampleSensitive,
        p(12, 2),
        Rung::CounterexampleSearch,
        S::Capability,
        syn,
        clean,
        60,
        "rule with a sparse exception class",
    );

    // Rabbit holes (population: rabbit hole), classes A to E and G.
    let rh = |c| DecoyStatus::RabbitHole(c);
    for w in [12u8, 16] {
        b.add(
            format!("decoy-simple-rule-w{w}"),
            M::DecoySimpleRule,
            p(w, 3),
            Rung::CounterexampleSearch,
            S::RabbitHole,
            adv,
            rh(AdversarialClass::SimpleWrongRule),
            55,
            "masked affine shown only below the mask",
        );
        b.add(
            format!("multi-fit-w{w}"),
            M::MultiFit,
            p(w, 2),
            Rung::CounterexampleSearch,
            S::RabbitHole,
            adv,
            rh(AdversarialClass::MultiFit),
            55,
            "two chains agree on the shown inputs",
        );
    }
    b.add(
        "spurious-field".into(),
        M::SpuriousField,
        p(48, 2),
        Rung::PartialObservation,
        S::RabbitHole,
        adv,
        rh(AdversarialClass::SpuriousDimension),
        60,
        "high field copies the answer when shown",
    );
    for k in [6u8, 8] {
        let mut prm = p(8, 2);
        prm.visible = k;
        b.add(
            format!("prefix-diverges-k{k}"),
            M::PrefixDiverges,
            prm,
            Rung::Recurrence,
            S::RabbitHole,
            adv,
            rh(AdversarialClass::PrefixDiverges),
            60,
            "prefix imitates a simple rule",
        );
    }
    b.add(
        "elegant-arbitrary".into(),
        M::ElegantArbitrary,
        p(8, 2),
        Rung::TheoryFormation,
        S::RabbitHole,
        adv,
        rh(AdversarialClass::ElegantArbitrary),
        60,
        "arbitrary table with a neat prefix",
    );
    for len in [2u8, 3] {
        let mut prm = p(16, len);
        prm.visible = 12;
        b.add(
            format!("noisy-chain-l{len}"),
            M::NoisyChain,
            prm,
            Rung::NoisyObservation,
            S::RabbitHole,
            adv,
            rh(AdversarialClass::Noisy),
            55,
            "noisy visible outputs",
        );
    }

    // Ambiguous / insufficient evidence (class F) and no structure.
    for n in [2u8, 3] {
        let mut prm = p(16, 2);
        prm.visible = n;
        b.add(
            format!("insufficient-evidence-n{n}"),
            M::InsufficientEvidence,
            prm,
            Rung::CounterexampleSearch,
            S::Ambiguous,
            adv,
            rh(AdversarialClass::InsufficientEvidence),
            50,
            "too few observations",
        );
    }
    b.add(
        "no-structure".into(),
        M::NoStructure,
        p(8, 1),
        Rung::TheoryFormation,
        S::Ambiguous,
        syn,
        DecoyStatus::NoStructure,
        50,
        "random table",
    );

    // The breadcrumb trail: capability primitives and their compositions.
    for c in 0..CAPABILITY_POOL_SIZE as u16 {
        let mut prm = p(16, 2);
        prm.caps = vec![c];
        b.add(
            format!("capability-{c:02}"),
            M::CapabilityPrimitive,
            prm,
            Rung::TwoTransformations,
            S::Capability,
            syn,
            clean,
            20,
            "hidden capability",
        );
    }
    for (a, c) in COMPOSITION_PAIRS {
        let mut prm = p(16, 4);
        prm.caps = vec![a, c];
        b.add(
            format!("compose-{a:02}-{c:02}"),
            M::CapabilityComposition,
            prm,
            Rung::Composition,
            S::Composition,
            syn,
            clean,
            60,
            "composition of two capabilities",
        );
    }
    for (a, c, d) in COMPOSITION_TRIPLES {
        let mut prm = p(16, 6);
        prm.caps = vec![a, c, d];
        b.add(
            format!("compose-{a:02}-{c:02}-{d:02}"),
            M::CapabilityComposition,
            prm,
            Rung::Composition,
            S::Composition,
            syn,
            clean,
            80,
            "composition of three capabilities",
        );
    }

    Registry { families: b.fams }
}
