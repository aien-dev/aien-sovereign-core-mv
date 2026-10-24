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

//! Domain-separated BLAKE3 digests and the Crumb v1 identity model.
//!
//! | identity                  | derivation                                              | learner sees it |
//! |---------------------------|---------------------------------------------------------|-----------------|
//! | `CrumbOccurrenceId`       | UUID v7, one per presentation/run                        | no              |
//! | `CrumbDigest`             | BLAKE3(canonical learner-visible Crumb v1 bytes)         | no (derivable)  |
//! | `GeneratorInstanceDigest` | BLAKE3(family_id, generator_version, seed, parameters)   | no              |
//! | `SealedRecordDigest`      | BLAKE3(canonical sealed record)                          | no              |
//!
//! Every digest uses a distinct BLAKE3 `derive_key` context, so a byte string
//! hashed for one purpose can never collide with the same bytes hashed for
//! another.

use std::fmt;

/// A 32-byte BLAKE3 digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub const ZERO: Digest = Digest([0u8; 32]);

    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn short(&self) -> String {
        self.hex()[..16].to_string()
    }

    pub fn from_hex(s: &str) -> Option<Digest> {
        if s.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = (chunk[0] as char).to_digit(16)?;
            let lo = (chunk[1] as char).to_digit(16)?;
            out[i] = (hi * 16 + lo) as u8;
        }
        Some(Digest(out))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hex())
    }
}

impl serde::Serialize for Digest {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}

impl<'de> serde::Deserialize<'de> for Digest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Digest::from_hex(&s).ok_or_else(|| serde::de::Error::custom("bad digest"))
    }
}

/// Every digest purpose in the system. Adding one is a schema change.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    CrumbDigest,
    GeneratorInstance,
    GeneratorCode,
    SealedRecord,
    HeldoutSet,
    VisibleSet,
    Program,
    Behavior,
    Verifier,
    Evaluation,
    TraceRecord,
    TrainingSample,
    TraceStream,
    Capability,
    Promotion,
    LedgerRecord,
    LedgerGenesis,
    Rng,
    Config,
    LearnerBank,
    SearchState,
}

impl Domain {
    fn context(self) -> &'static str {
        match self {
            Domain::CrumbDigest => "crumbs v1 2026-09-27 CrumbDigest",
            Domain::GeneratorInstance => "crumbs v1 2026-09-27 GeneratorInstanceDigest",
            Domain::GeneratorCode => "crumbs v1 2026-09-27 GeneratorCodeDigest",
            Domain::SealedRecord => "crumbs v1 2026-09-27 SealedRecordDigest",
            Domain::HeldoutSet => "crumbs v1 2026-09-27 HeldoutSetDigest",
            Domain::VisibleSet => "crumbs v1 2026-09-27 VisibleSetDigest",
            Domain::Program => "crumbs v1 2026-09-27 CandidateProgramDigest",
            Domain::Behavior => "crumbs v1 2026-09-27 BehaviorSignature",
            Domain::Verifier => "crumbs v1 2026-09-27 VerifierDigest",
            Domain::Evaluation => "crumbs v1 2026-09-27 EvaluationDigest",
            Domain::TraceRecord => "crumbs v1 2026-09-27 TraceRecordChain",
            Domain::TrainingSample => "crumbs v1 2026-09-27 TrainingSampleChain",
            Domain::TraceStream => "crumbs v1 2026-09-27 TraceStreamDigest",
            Domain::Capability => "crumbs v1 2026-09-27 CapabilityId",
            Domain::Promotion => "crumbs v1 2026-09-27 PromotionRecord",
            Domain::LedgerRecord => "crumbs v1 2026-09-27 CrumblineRecord",
            Domain::LedgerGenesis => "crumbs v1 2026-09-27 CrumblineGenesis",
            Domain::Rng => "crumbs v1 2026-09-27 DeterministicStream",
            Domain::Config => "crumbs v1 2026-09-27 ConfigDigest",
            Domain::LearnerBank => "crumbs v1 2026-09-27 LearnerBankDigest",
            Domain::SearchState => "crumbs v1 2026-09-27 SearchStateDigest",
        }
    }
}

pub fn digest(domain: Domain, bytes: &[u8]) -> Digest {
    Digest(blake3::derive_key(domain.context(), bytes))
}

/// Incremental hashing under a domain.
pub struct Hasher(blake3::Hasher);

impl Hasher {
    pub fn new(domain: Domain) -> Self {
        Hasher(blake3::Hasher::new_derive_key(domain.context()))
    }
    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.update(bytes);
        self
    }
    pub fn finish(&self) -> Digest {
        Digest(*self.0.finalize().as_bytes())
    }
}

/// UUID v7 occurrence id: identifies one presentation of a crumb in one run.
/// Time-ordered and deliberately NOT content-derived; it never identifies the problem.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct CrumbOccurrenceId(pub uuid::Uuid);

impl CrumbOccurrenceId {
    pub fn new() -> Self {
        CrumbOccurrenceId(uuid::Uuid::now_v7())
    }
}

impl Default for CrumbOccurrenceId {
    fn default() -> Self {
        Self::new()
    }
}

/// BLAKE3 of the canonical learner-visible bytes (which include the schema version).
pub type CrumbDigest = Digest;
/// BLAKE3(family_id || generator_version || seed || canonical generation parameters).
pub type GeneratorInstanceDigest = Digest;
/// BLAKE3 of the canonical sealed record (metadata + held-outs + provenance).
pub type SealedRecordDigest = Digest;
