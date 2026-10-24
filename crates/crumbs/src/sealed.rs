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

//! Crumb v1 sealed contract: everything the learner never sees.
//!
//! A `SealedCrumb` is a pure function of (family, generator version, seed,
//! generation parameters): regenerating it reproduces the identical bytes and
//! digests. Occurrence-level facts (which population the mixer drew it into,
//! when it was shown) live in the ledger record, not here.
//!
//! `Debug` is redacted: logging a sealed record by accident prints only its
//! digest, never labels or held-out data.

use crate::canon::Enc;
use crate::digest::{
    digest, CrumbDigest, Digest, Domain, GeneratorInstanceDigest, SealedRecordDigest,
};
use crate::program::Program;
use crate::provenance::Provenance;
use serde::{Deserialize, Serialize};

pub const SEALED_SCHEMA_VERSION: u16 = 1;

/// Capability rungs of the progression. Builder vocabulary; never sent to a learner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Rung {
    SingleTransformation = 1,
    TwoTransformations = 2,
    RepeatedTransformation = 3,
    ConditionalTransformation = 4,
    MultiInputRelation = 5,
    Recurrence = 6,
    State = 7,
    Composition = 8,
    Symmetry = 9,
    Modularity = 10,
    Invariance = 11,
    MultiStepAlgorithm = 12,
    LatentVariables = 13,
    NoisyObservation = 14,
    PartialObservation = 15,
    CounterexampleSearch = 16,
    TheoryFormation = 17,
}

impl Rung {
    pub const ALL: [Rung; 17] = [
        Rung::SingleTransformation,
        Rung::TwoTransformations,
        Rung::RepeatedTransformation,
        Rung::ConditionalTransformation,
        Rung::MultiInputRelation,
        Rung::Recurrence,
        Rung::State,
        Rung::Composition,
        Rung::Symmetry,
        Rung::Modularity,
        Rung::Invariance,
        Rung::MultiStepAlgorithm,
        Rung::LatentVariables,
        Rung::NoisyObservation,
        Rung::PartialObservation,
        Rung::CounterexampleSearch,
        Rung::TheoryFormation,
    ];
}

/// What kind of generator family produced the crumb.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceFamily {
    Capability = 1,
    Composition = 2,
    Ambiguous = 3,
    RabbitHole = 4,
    Frontier = 5,
    CleanRoomPhysics = 6,
}

/// The five secret populations the mixer draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Population {
    Fundamental = 1,
    Composition = 2,
    Ambiguous = 3,
    RabbitHole = 4,
    Frontier = 5,
}

impl Population {
    pub const ALL: [Population; 5] = [
        Population::Fundamental,
        Population::Composition,
        Population::Ambiguous,
        Population::RabbitHole,
        Population::Frontier,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruthStatus {
    /// Every labeled output is established by exact computation.
    Verified = 1,
    /// Outputs rest on an unproven claim. Never used for labeled examples.
    Conjectural = 2,
    /// No truth claim.
    None = 3,
}

/// Rabbit-hole mechanisms. Rabbit holes deceive; they never lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdversarialClass {
    /// A: visible examples favour a simple wrong rule; held-outs break it.
    SimpleWrongRule = 1,
    /// B: several programs fit the visible examples exactly; held-outs distinguish.
    MultiFit = 2,
    /// C: an irrelevant input dimension correlates with the answer only in the visible sample.
    SpuriousDimension = 3,
    /// D: a finite prefix imitates a simpler process, then diverges.
    PrefixDiverges = 4,
    /// E: an arbitrary complex generator produces an elegant-looking prefix.
    ElegantArbitrary = 5,
    /// F: the visible evidence cannot separate several hypotheses.
    InsufficientEvidence = 6,
    /// G: noisy observations; exact interpolation is the wrong objective.
    Noisy = 7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoyStatus {
    Clean,
    RabbitHole(AdversarialClass),
    NoStructure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnownSolution {
    /// A generating program in the CPG1 format is known.
    Known(Program),
    /// The generator is known, but not as a CPG1 chain.
    KnownOutsideProgramFormat,
    /// Nobody knows (frontier material).
    UnknownToHumanity,
    /// The outputs have no compact structure by construction.
    NoCompactStructure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Outputs are the generator's own exact evaluation.
    ExactGeneratorEvaluation = 1,
    /// Outputs recomputed by an independent routine and cross-checked.
    IndependentRecomputation = 2,
    /// Bounded exact computation of a finite instance.
    FiniteExactComputation = 3,
}

/// Held-out tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HeldoutTier {
    Interpolation = 0,
    Extrapolation = 1,
    Adversarial = 2,
}

impl HeldoutTier {
    pub const ALL: [HeldoutTier; 3] = [
        HeldoutTier::Interpolation,
        HeldoutTier::Extrapolation,
        HeldoutTier::Adversarial,
    ];
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneExample {
    pub input: Vec<u64>,
    pub output: Vec<u64>,
}

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Heldouts {
    pub interpolation: Vec<LaneExample>,
    pub extrapolation: Vec<LaneExample>,
    pub adversarial: Vec<LaneExample>,
}

impl Heldouts {
    pub fn tier(&self, t: HeldoutTier) -> &[LaneExample] {
        match t {
            HeldoutTier::Interpolation => &self.interpolation,
            HeldoutTier::Extrapolation => &self.extrapolation,
            HeldoutTier::Adversarial => &self.adversarial,
        }
    }

    pub fn encode(&self, e: &mut Enc) {
        for t in HeldoutTier::ALL {
            let set = self.tier(t);
            e.u32(set.len() as u32);
            for ex in set {
                e.u64s(&ex.input).u64s(&ex.output);
            }
        }
    }

    pub fn digest(&self) -> Digest {
        let mut e = Enc::new();
        self.encode(&mut e);
        digest(Domain::HeldoutSet, &e.finish())
    }

    pub fn len(&self) -> usize {
        self.interpolation.len() + self.extrapolation.len() + self.adversarial.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Opaque capability identifier (BLAKE3 of the capability's defining bytes).
pub type CapabilityId = Digest;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedCrumb {
    pub schema_version: u16,
    pub crumb_digest: CrumbDigest,

    pub family_id: u32,
    pub family_key: String,
    pub mechanism: String,
    pub generator_version: String,
    pub generator_code_digest: Digest,
    pub seed: u64,
    pub heldout_seed: u64,
    pub generation_parameters: Vec<u8>,
    pub generator_instance_digest: GeneratorInstanceDigest,

    pub verification_method: VerificationMethod,
    pub difficulty: u16,
    pub rung: Rung,
    pub required_capabilities: Vec<CapabilityId>,
    pub provides_capability: Option<CapabilityId>,

    pub provenance: Provenance,
    pub source_family: SourceFamily,
    pub truth_status: TruthStatus,
    pub decoy_status: DecoyStatus,
    pub known_solution: KnownSolution,
    /// Programs that fit the visible examples as well as the truth does.
    pub alternatives: Vec<Program>,
    /// Indices of visible examples carrying observation noise.
    pub noisy_visible: Vec<u32>,
    /// Inclusive input range the family draws from (lane 0).
    pub expected_domain: (u64, u64),
    pub topic_internal: String,

    pub heldouts: Heldouts,
}

impl std::fmt::Debug for SealedCrumb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SealedCrumb(<redacted {}>)", self.digest().short())
    }
}

impl std::fmt::Debug for Heldouts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Heldouts(<redacted {} examples>)", self.len())
    }
}

impl std::fmt::Debug for LaneExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LaneExample(<redacted>)")
    }
}

impl SealedCrumb {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.bytes(b"CRS1")
            .u16(self.schema_version)
            .digest(&self.crumb_digest);
        e.u32(self.family_id)
            .str(&self.family_key)
            .str(&self.mechanism)
            .str(&self.generator_version);
        e.digest(&self.generator_code_digest)
            .u64(self.seed)
            .u64(self.heldout_seed);
        e.blob(&self.generation_parameters)
            .digest(&self.generator_instance_digest);
        e.u8(self.verification_method as u8)
            .u16(self.difficulty)
            .u8(self.rung as u8);
        e.u32(self.required_capabilities.len() as u32);
        for c in &self.required_capabilities {
            e.digest(c);
        }
        match &self.provides_capability {
            Some(c) => e.u8(1).digest(c),
            None => e.u8(0),
        };
        e.u8(self.provenance.class as u8)
            .u8(self.provenance.uses_human_constants as u8)
            .u8(self.provenance.uses_human_ontology as u8);
        e.u8(self.source_family as u8).u8(self.truth_status as u8);
        match self.decoy_status {
            DecoyStatus::Clean => e.u8(0),
            DecoyStatus::RabbitHole(c) => e.u8(1).u8(c as u8),
            DecoyStatus::NoStructure => e.u8(2),
        };
        match &self.known_solution {
            KnownSolution::Known(p) => e.u8(1).blob(&p.to_bytes()),
            KnownSolution::KnownOutsideProgramFormat => e.u8(2),
            KnownSolution::UnknownToHumanity => e.u8(3),
            KnownSolution::NoCompactStructure => e.u8(4),
        };
        e.u32(self.alternatives.len() as u32);
        for p in &self.alternatives {
            e.blob(&p.to_bytes());
        }
        e.u32(self.noisy_visible.len() as u32);
        for i in &self.noisy_visible {
            e.u32(*i);
        }
        e.u64(self.expected_domain.0)
            .u64(self.expected_domain.1)
            .str(&self.topic_internal);
        self.heldouts.encode(&mut e);
        e.finish()
    }

    /// Commits to metadata, held-outs and provenance. Never enters learner space.
    pub fn digest(&self) -> SealedRecordDigest {
        digest(Domain::SealedRecord, &self.to_bytes())
    }
}

/// GeneratorInstanceDigest = BLAKE3(family_id || generator_version || seed || canonical parameters).
pub fn generator_instance_digest(
    family_id: u32,
    generator_version: &str,
    seed: u64,
    params: &[u8],
) -> GeneratorInstanceDigest {
    let mut e = Enc::new();
    e.u32(family_id)
        .str(generator_version)
        .u64(seed)
        .blob(params);
    digest(Domain::GeneratorInstance, &e.finish())
}
