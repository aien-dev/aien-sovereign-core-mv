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

//! The Crumbline ledger: a mechanically checkable provenance chain.
//!
//! Every record binds the generator instance digest, the sealed record digest,
//! the verifier digest, the learner-visible digest, the learner build digest,
//! the evaluation receipts, the trace stream digest and the previous root. The
//! claim "every training record is firsthand" is then a property anyone can
//! re-verify from the ledger file, not a statement of doctrine.
//!
//! Lanes are separate chains. The clean-room lane refuses any record whose
//! provenance violates the clean-room policy.

use crate::canon::Enc;
use crate::digest::{digest, CrumbOccurrenceId, Digest, Domain};
use crate::provenance::{clean_room_check, CleanRoomViolation, Lane, Provenance};
use crate::sealed::Population;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrumblineRecord {
    pub seq: u64,
    pub lane: Lane,
    pub occurrence: CrumbOccurrenceId,
    pub condition: String,
    pub population: Population,
    pub crumb_digest: Digest,
    pub generator_instance_digest: Digest,
    pub sealed_digest: Digest,
    pub verifier_digest: Digest,
    pub learner_digest: Digest,
    pub evaluations: Vec<Digest>,
    pub trace_stream_digest: Digest,
    pub promotions: Vec<Digest>,
    pub prev_root: Digest,
    pub root: Digest,
}

impl CrumblineRecord {
    fn body(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.bytes(b"CLR1")
            .u64(self.seq)
            .u8(self.lane as u8)
            .bytes(self.occurrence.0.as_bytes());
        e.str(&self.condition).u8(self.population as u8);
        for d in [
            &self.crumb_digest,
            &self.generator_instance_digest,
            &self.sealed_digest,
            &self.verifier_digest,
            &self.learner_digest,
        ] {
            e.digest(d);
        }
        e.u32(self.evaluations.len() as u32);
        for d in &self.evaluations {
            e.digest(d);
        }
        e.digest(&self.trace_stream_digest);
        e.u32(self.promotions.len() as u32);
        for d in &self.promotions {
            e.digest(d);
        }
        e.digest(&self.prev_root);
        e.finish()
    }

    pub fn compute_root(&self) -> Digest {
        digest(Domain::LedgerRecord, &self.body())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LedgerError {
    CleanRoom(CleanRoomViolation),
    Chain,
}

pub struct Ledger {
    pub lane: Lane,
    pub genesis: Digest,
    pub head: Digest,
    pub records: Vec<CrumblineRecord>,
}

pub fn genesis(lane: Lane, run_id: &str) -> Digest {
    let mut e = Enc::new();
    e.u8(lane as u8).str(run_id);
    digest(Domain::LedgerGenesis, &e.finish())
}

impl Ledger {
    pub fn new(lane: Lane, run_id: &str) -> Self {
        let g = genesis(lane, run_id);
        Ledger {
            lane,
            genesis: g,
            head: g,
            records: vec![],
        }
    }

    /// Append a record. The clean-room lane checks provenance first.
    pub fn append(
        &mut self,
        mut rec: CrumblineRecord,
        provenance: &Provenance,
    ) -> Result<Digest, LedgerError> {
        if self.lane == Lane::CleanRoom {
            clean_room_check(provenance).map_err(LedgerError::CleanRoom)?;
        }
        rec.lane = self.lane;
        rec.prev_root = self.head;
        rec.root = rec.compute_root();
        self.head = rec.root;
        self.records.push(rec);
        Ok(self.head)
    }

    /// Re-verify the whole chain from genesis.
    pub fn verify(genesis: Digest, records: &[CrumblineRecord]) -> Result<Digest, LedgerError> {
        let mut head = genesis;
        for r in records {
            if r.prev_root != head || r.compute_root() != r.root {
                return Err(LedgerError::Chain);
            }
            head = r.root;
        }
        Ok(head)
    }

    pub fn to_jsonl(&self) -> String {
        let mut s = String::new();
        for r in &self.records {
            s.push_str(&serde_json::to_string(r).expect("serializable"));
            s.push('\n');
        }
        s
    }
}
