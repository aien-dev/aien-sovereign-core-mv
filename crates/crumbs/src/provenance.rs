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

//! Provenance, contamination classes and the clean-room boundary (sealed side).
//!
//! The clean-room lane exists so that a later rediscovery can be checked: a
//! crumb enters it only if nothing about its construction came from human
//! theory, human constants or human physical ontology. The lane has its own
//! ledger partition, and the partition refuses forbidden records outright.

use serde::{Deserialize, Serialize};

/// Where the structure behind a crumb came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContaminationClass {
    /// Invented by a generator with no reference to human theory.
    SyntheticNovel = 1,
    /// Invented to mislead (rabbit holes); still no human theory.
    AdversarialSynthetic = 2,
    /// Encodes a relationship known from human mathematics or science.
    HumanTheoryDerived = 3,
    /// Finite instances of structures around open problems.
    FrontierDerived = 4,
    /// Measurements of the real world.
    RealWorldMeasured = 5,
}

/// Everything the clean-room policy needs to judge a crumb.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub class: ContaminationClass,
    pub uses_human_constants: bool,
    pub uses_human_ontology: bool,
}

impl Provenance {
    pub const fn synthetic() -> Self {
        Provenance {
            class: ContaminationClass::SyntheticNovel,
            uses_human_constants: false,
            uses_human_ontology: false,
        }
    }
    pub const fn adversarial() -> Self {
        Provenance {
            class: ContaminationClass::AdversarialSynthetic,
            uses_human_constants: false,
            uses_human_ontology: false,
        }
    }
    pub const fn human_theory() -> Self {
        Provenance {
            class: ContaminationClass::HumanTheoryDerived,
            uses_human_constants: true,
            uses_human_ontology: false,
        }
    }
}

/// Ledger lanes. Each lane is a separate hash chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lane {
    Main = 1,
    CleanRoom = 2,
}

/// Why the clean-room lane refused a crumb.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanRoomViolation {
    HumanTheoryDerived,
    FrontierDerived,
    RealWorldMeasured,
    HumanConstants,
    HumanOntology,
}

/// The clean-room forbidden list. Permitted: synthetic and adversarial
/// synthetic structure with no human constants and no human ontology.
pub fn clean_room_check(p: &Provenance) -> Result<(), CleanRoomViolation> {
    match p.class {
        ContaminationClass::HumanTheoryDerived => {
            return Err(CleanRoomViolation::HumanTheoryDerived)
        }
        ContaminationClass::FrontierDerived => return Err(CleanRoomViolation::FrontierDerived),
        ContaminationClass::RealWorldMeasured => return Err(CleanRoomViolation::RealWorldMeasured),
        ContaminationClass::SyntheticNovel | ContaminationClass::AdversarialSynthetic => {}
    }
    if p.uses_human_constants {
        return Err(CleanRoomViolation::HumanConstants);
    }
    if p.uses_human_ontology {
        return Err(CleanRoomViolation::HumanOntology);
    }
    Ok(())
}
