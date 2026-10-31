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

//! The ONLY file that knows the kind-24 SubjectState byte layout.
//!
//! Format: aienos ADR 0018 s2.2 (PROPOSED, not frozen). Reference decoder:
//! aienos `native/kernel/svc/continuity_subject.c` (`cs_subject_decode`,
//! `cs_subject_validate`, `cs_intent_id`); object id: `continuity_codec.c`
//! `cc_object_id`. If the format changes, change this file only.
//! Decode and id only: there is deliberately no encoder here.
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"AIENSUBJ";
pub const HEADER: usize = 16;
pub const MAX_OBJECT_BYTES: usize = 16384;
pub const KIND: u16 = 24;
const MAX_INTENTS: usize = 64;
const MAX_KNOWLEDGE: usize = 64;
pub const INTENT_GOAL_LATENCY: u32 = 1;
pub const ACTIVE: u8 = 1;
const INACTIVE: u8 = 2;
const SUPERSEDED: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    pub id: [u8; 32],
    pub supersedes: [u8; 32],
    pub kind: u32,
    pub state: u8,
    pub since: u64,
    pub payload: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subject {
    pub root: [u8; 32],
    pub agent: [u8; 32],
    pub previous: [u8; 32],
    pub sequence: u64,
    pub cortex: [u8; 32],
    pub provenance: [u8; 32],
    pub origin: u8,
    pub intents: Vec<Intent>,
    pub knowledge: Vec<([u8; 32], u64, u64)>,
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.b.len() - self.at {
            return Err("truncated object".into());
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn id(&mut self) -> Result<[u8; 32], String> {
        Ok(self.take(32)?.try_into().expect("32"))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2")))
    }
    fn zeros(&mut self, n: usize, what: &str) -> Result<(), String> {
        if self.take(n)?.iter().any(|&x| x != 0) {
            return Err(format!("nonzero {what}"));
        }
        Ok(())
    }
}

fn zero(x: &[u8; 32]) -> bool {
    x.iter().all(|&b| b == 0)
}

/// `cs_intent_id`: SHA-256("AIENOS_INTENT_v0:" || agent || kind || since || p0 || p1).
fn intent_id(agent: &[u8; 32], kind: u32, since: u64, p: [u64; 2]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIENOS_INTENT_v0:");
    h.update(agent);
    h.update(kind.to_le_bytes());
    h.update(since.to_le_bytes());
    h.update(p[0].to_le_bytes());
    h.update(p[1].to_le_bytes());
    h.finalize().into()
}

/// `cc_object_id(24, bytes, len)`: SHA-256("AIENOS-STORE-OBJECT-V1\0" ||
/// kind u16 || store object version u16 (1) || len u64 || bytes).
pub fn object_id(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIENOS-STORE-OBJECT-V1\0");
    h.update(KIND.to_le_bytes());
    h.update(1u16.to_le_bytes());
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize().into()
}

/// Strict decode with every rule of `cs_subject_decode` + `cs_subject_validate`.
pub fn decode(bytes: &[u8]) -> Result<Subject, String> {
    if bytes.len() > MAX_OBJECT_BYTES {
        return Err("object exceeds the 16384-byte cap".into());
    }
    let mut r = Rd { b: bytes, at: 0 };
    if r.take(8)? != MAGIC {
        return Err("bad magic".into());
    }
    if r.u16()? != 0 {
        return Err("unsupported continuity format version".into());
    }
    r.zeros(2, "header reserved")?;
    if r.u32()? as usize != bytes.len() - HEADER {
        return Err("body length mismatch".into());
    }
    let root = r.id()?;
    let agent = r.id()?;
    let previous = r.id()?;
    let sequence = r.u64()?;
    let cortex = r.id()?;
    let provenance = r.id()?;
    let origin = r.take(1)?[0];
    r.zeros(7, "reserved")?;
    let n_in = r.u16()? as usize;
    if n_in > MAX_INTENTS {
        return Err("too many standing intents".into());
    }
    let n_kn = r.u16()? as usize;
    if n_kn > MAX_KNOWLEDGE {
        return Err("too many knowledge references".into());
    }
    r.zeros(4, "reserved")?;
    let mut intents = Vec::with_capacity(n_in);
    for _ in 0..n_in {
        let id = r.id()?;
        let supersedes = r.id()?;
        let kind = r.u32()?;
        let state = r.take(1)?[0];
        r.zeros(3, "intent reserved")?;
        let since = r.u64()?;
        let payload = [r.u64()?, r.u64()?];
        intents.push(Intent {
            id,
            supersedes,
            kind,
            state,
            since,
            payload,
        });
    }
    let mut knowledge = Vec::with_capacity(n_kn);
    for _ in 0..n_kn {
        let d = r.id()?;
        knowledge.push((d, r.u64()?, r.u64()?));
    }
    if r.at != bytes.len() {
        return Err("trailing bytes".into());
    }
    let s = Subject {
        root,
        agent,
        previous,
        sequence,
        cortex,
        provenance,
        origin,
        intents,
        knowledge,
    };
    validate(&s)?;
    Ok(s)
}

fn validate(s: &Subject) -> Result<(), String> {
    if zero(&s.root) {
        return Err("zero root id".into());
    }
    if zero(&s.agent) {
        return Err("zero agent id".into());
    }
    if s.sequence == 0 {
        return Err("zero sequence".into());
    }
    if s.sequence == 1 && !zero(&s.previous) {
        return Err("genesis subject names a previous object".into());
    }
    if s.sequence > 1 && zero(&s.previous) {
        return Err("subject object after the first has no previous".into());
    }
    if zero(&s.provenance) {
        return Err("zero provenance".into());
    }
    if s.origin != 1 && s.origin != 2 {
        return Err("unknown subject origin".into());
    }
    for (i, a) in s.intents.iter().enumerate() {
        if i > 0 && s.intents[i - 1].id >= a.id {
            return Err("intents not strictly ascending by id".into());
        }
        if ![ACTIVE, INACTIVE, SUPERSEDED].contains(&a.state) {
            return Err("unknown intent state".into());
        }
        if a.kind != INTENT_GOAL_LATENCY {
            return Err("unknown intent kind".into());
        }
        if a.since == 0 || a.since > s.sequence {
            return Err("intent created outside the subject chain".into());
        }
        if intent_id(&s.agent, a.kind, a.since, a.payload) != a.id {
            return Err("intent id does not derive from its content".into());
        }
        if !zero(&a.supersedes) {
            if a.supersedes == a.id {
                return Err("intent supersedes itself".into());
            }
            let Some(j) = s.intents.iter().find(|x| x.id == a.supersedes) else {
                return Err("superseded intent is absent".into());
            };
            if j.state != SUPERSEDED {
                return Err("superseded intent is not marked superseded".into());
            }
            if j.since > a.since {
                return Err("intent supersedes a newer intent".into());
            }
        }
        if a.state == ACTIVE
            && s.intents[..i]
                .iter()
                .any(|j| j.state == ACTIVE && j.kind == a.kind && j.payload[0] == a.payload[0])
        {
            return Err("two active intents in one slot".into());
        }
    }
    for a in s.intents.iter().filter(|a| a.state == SUPERSEDED) {
        if s.intents.iter().filter(|j| j.supersedes == a.id).count() != 1 {
            return Err("superseded intent not named by exactly one successor".into());
        }
    }
    for (i, k) in s.knowledge.iter().enumerate() {
        if zero(&k.0) {
            return Err("zero knowledge digest".into());
        }
        if i > 0 && s.knowledge[i - 1].0 >= k.0 {
            return Err("knowledge not strictly ascending by digest".into());
        }
        if k.2 == 0 || k.2 > s.sequence {
            return Err("knowledge held outside the subject chain".into());
        }
    }
    Ok(())
}
