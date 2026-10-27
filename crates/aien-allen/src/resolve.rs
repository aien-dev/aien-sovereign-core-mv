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

//! Resolve-only gate. Never provisions: no function in this crate creates a
//! subject object (a test scans the source for that). Missing or damaged
//! state is a [`Refusal`].
use crate::binding::{self, Pin};
use crate::subject_v0::{self, Subject};
use crate::{hex, unhex32, ENV_ADOPT, ENV_SUBJECT};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum SubjectSource {
    /// One exported head object file. Chain claims are NOT made.
    Head(PathBuf),
    /// A directory of raw objects `<object-id-hex>.bin`: the chain is checked.
    Dir(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// No subject at the named path: never created.
    Absent(String),
    Corrupt(String),
    /// An object names another agent or root than the rest of the chain / the pin.
    ForeignIdentity(String),
    Forked(u64),
    Gap(u64),
    NoActiveIntent,
    LineageUnbound,
    /// The Cortex journal has no record 1.
    LineageMissing,
    LineageMismatch,
    /// Subject valid, no pin, and no explicit operator adoption.
    PinAbsent,
    PinDamaged(String),
    AdoptMismatch,
    AdoptRefusedPinExists,
    AdoptBadValue,
    MachineChanged,
    RolledBack {
        pinned: u64,
        found: u64,
    },
    SameSequenceDifferentId,
    ForkedFromPin,
    Io(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Refusal::*;
        match self {
            Absent(w) => write!(f, "Absent: no subject ({w}); ALLEN is never created implicitly"),
            Corrupt(w) => write!(f, "Corrupt: {w}"),
            ForeignIdentity(w) => write!(f, "ForeignIdentity: {w}"),
            Forked(s) => write!(f, "Forked: two subject objects at sequence {s}"),
            Gap(s) => write!(f, "Gap: no subject object at sequence {s}"),
            NoActiveIntent => write!(f, "NoActiveIntent: no ACTIVE GOAL_LATENCY intent"),
            LineageUnbound => write!(f, "LineageUnbound: subject names no Cortex lineage"),
            LineageMissing => write!(f, "LineageMissing: the Cortex journal has no record 1"),
            LineageMismatch => write!(
                f,
                "LineageMismatch: this Cortex journal is not the subject's memory"
            ),
            PinAbsent => write!(
                f,
                "PinAbsent: a valid subject but no pin; set {ENV_ADOPT}=<expected agent id hex> to adopt once"
            ),
            PinDamaged(w) => write!(f, "PinDamaged: {w}"),
            AdoptMismatch => write!(
                f,
                "AdoptMismatch: {ENV_ADOPT} is not this subject's agent id"
            ),
            AdoptRefusedPinExists => write!(
                f,
                "AdoptRefusedPinExists: a pin exists; {ENV_ADOPT} is refused"
            ),
            AdoptBadValue => write!(f, "AdoptBadValue: {ENV_ADOPT} must be 64 hex characters"),
            MachineChanged => write!(
                f,
                "MachineChanged: the machine id differs from the pin; never rebound (ruling pending)"
            ),
            RolledBack { pinned, found } => write!(
                f,
                "RolledBack: head sequence {found} is below the pinned {pinned}"
            ),
            SameSequenceDifferentId => write!(
                f,
                "SameSequenceDifferentId: same sequence as the pin, different object"
            ),
            ForkedFromPin => write!(
                f,
                "ForkedFromPin: the chain does not contain the pinned head"
            ),
            Io(w) => write!(f, "Io: {w}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub agent: [u8; 32],
    pub root: [u8; 32],
    pub head_id: [u8; 32],
    pub head_seq: u64,
    pub lineage: [u8; 32],
    /// True only in directory mode (gap-free, fork-free, link-correct to sequence 1).
    pub chain_verified: bool,
    /// True when this call wrote the pin for the first time (adoption).
    pub adopted: bool,
}

/// What the host knows about itself and the journal at open time.
#[derive(Debug, Clone, Copy)]
pub struct Context {
    pub machine_id: [u8; 32],
    /// Digest of the Cortex journal's record 1 (`None` = journal has none).
    pub lineage: Option<[u8; 32]>,
}

#[derive(Debug, Clone)]
pub struct Engagement {
    pub source: SubjectSource,
    pub adopt: Option<String>,
}

pub enum Gate {
    NotEngaged,
    Engaged(Resolved),
    Refused(Refusal),
}

impl Engagement {
    /// `None` = not engaged (variable unset or empty).
    pub fn from_env(get: &dyn Fn(&str) -> Option<String>) -> Option<Engagement> {
        let p = get(ENV_SUBJECT).filter(|s| !s.trim().is_empty())?;
        let p = PathBuf::from(p.trim());
        let source = if p.is_dir() {
            SubjectSource::Dir(p)
        } else {
            SubjectSource::Head(p)
        };
        Some(Engagement {
            source,
            adopt: get(ENV_ADOPT).filter(|s| !s.trim().is_empty()),
        })
    }
}

/// The one entry point the daemon calls when a compose home is opened.
pub fn gate(home: &Path, ctx: &Context, get: &dyn Fn(&str) -> Option<String>) -> Gate {
    let Some(e) = Engagement::from_env(get) else {
        return Gate::NotEngaged;
    };
    match resolve(&e.source, &binding::pin_path(home), ctx, e.adopt.as_deref()) {
        Ok(r) => Gate::Engaged(r),
        Err(r) => Gate::Refused(r),
    }
}

fn read_object(path: &Path) -> Result<Vec<u8>, Refusal> {
    match std::fs::read(path) {
        Ok(b) if b.is_empty() => Err(Refusal::Absent(format!("{} is empty", path.display()))),
        Ok(b) => Ok(b),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(Refusal::Absent(format!("{} not found", path.display())))
        }
        Err(e) => Err(Refusal::Io(format!("{}: {e}", path.display()))),
    }
}

struct Loaded {
    head: Subject,
    head_id: [u8; 32],
    /// (sequence, object id) of every verified chain object; empty in head mode.
    chain: Vec<(u64, [u8; 32])>,
    chain_verified: bool,
}

fn load(src: &SubjectSource) -> Result<Loaded, Refusal> {
    match src {
        SubjectSource::Head(p) => {
            let b = read_object(p)?;
            let head = subject_v0::decode(&b).map_err(Refusal::Corrupt)?;
            Ok(Loaded {
                head,
                head_id: subject_v0::object_id(&b),
                chain: Vec::new(),
                chain_verified: false,
            })
        }
        SubjectSource::Dir(d) => load_dir(d),
    }
}

fn load_dir(d: &Path) -> Result<Loaded, Refusal> {
    let rd = std::fs::read_dir(d).map_err(|e| Refusal::Absent(format!("{}: {e}", d.display())))?;
    let mut objs: Vec<([u8; 32], Subject)> = Vec::new();
    let mut names: Vec<_> = rd.filter_map(|x| x.ok()).map(|x| x.path()).collect();
    names.sort();
    for p in names {
        if p.extension().and_then(|x| x.to_str()) != Some("bin") {
            continue;
        }
        let b = read_object(&p)?;
        let id = subject_v0::object_id(&b);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem != hex(&id) {
            return Err(Refusal::Corrupt(format!(
                "{} does not hash to its name (changed bytes)",
                p.display()
            )));
        }
        let s = subject_v0::decode(&b)
            .map_err(|w| Refusal::Corrupt(format!("{}: {w}", p.display())))?;
        objs.push((id, s));
    }
    if objs.is_empty() {
        return Err(Refusal::Absent(format!("{} holds no objects", d.display())));
    }
    let max = objs.iter().map(|o| o.1.sequence).max().expect("non-empty");
    let (g_agent, g_root) = match objs.iter().find(|o| o.1.sequence == 1) {
        Some(g) => (g.1.agent, g.1.root),
        None => return Err(Refusal::Gap(1)),
    };
    let mut chain = Vec::new();
    let mut prev: Option<[u8; 32]> = None;
    let mut head = None;
    for s in 1..=max {
        let at: Vec<_> = objs.iter().filter(|o| o.1.sequence == s).collect();
        match at.len() {
            0 => return Err(Refusal::Gap(s)),
            1 => {}
            _ => return Err(Refusal::Forked(s)),
        }
        let (id, o) = (&at[0].0, &at[0].1);
        if o.agent != g_agent || o.root != g_root {
            return Err(Refusal::ForeignIdentity(format!(
                "object at sequence {s} names another agent or root"
            )));
        }
        if let Some(p) = prev {
            if o.previous != p {
                return Err(Refusal::Corrupt(format!("link broken at sequence {s}")));
            }
        }
        prev = Some(*id);
        chain.push((s, *id));
        head = Some((*id, o.clone()));
    }
    let (head_id, head) = head.expect("max >= 1");
    Ok(Loaded {
        head,
        head_id,
        chain,
        chain_verified: true,
    })
}

pub fn resolve(
    src: &SubjectSource,
    pin_path: &Path,
    ctx: &Context,
    adopt: Option<&str>,
) -> Result<Resolved, Refusal> {
    let l = load(src)?;
    let h = &l.head;
    if !h
        .intents
        .iter()
        .any(|i| i.state == subject_v0::ACTIVE && i.kind == subject_v0::INTENT_GOAL_LATENCY)
    {
        return Err(Refusal::NoActiveIntent);
    }
    if h.cortex.iter().all(|&b| b == 0) {
        return Err(Refusal::LineageUnbound);
    }
    match ctx.lineage {
        None => return Err(Refusal::LineageMissing),
        Some(j) if j != h.cortex => return Err(Refusal::LineageMismatch),
        Some(_) => {}
    }
    let pin = binding::read(pin_path).map_err(Refusal::PinDamaged)?;
    let mut adopted = false;
    match (&pin, adopt) {
        (None, None) => return Err(Refusal::PinAbsent),
        (None, Some(a)) => {
            let want = unhex32(a.trim()).ok_or(Refusal::AdoptBadValue)?;
            if want != h.agent {
                return Err(Refusal::AdoptMismatch);
            }
            adopted = true;
        }
        (Some(_), Some(_)) => return Err(Refusal::AdoptRefusedPinExists),
        (Some(p), None) => {
            if p.agent != h.agent || p.root != h.root {
                return Err(Refusal::ForeignIdentity(
                    "subject agent or root differs from the pin".into(),
                ));
            }
            if p.machine_id != ctx.machine_id {
                return Err(Refusal::MachineChanged);
            }
            if p.lineage != h.cortex {
                return Err(Refusal::LineageMismatch);
            }
            if h.sequence < p.head_seq {
                return Err(Refusal::RolledBack {
                    pinned: p.head_seq,
                    found: h.sequence,
                });
            }
            if h.sequence == p.head_seq && l.head_id != p.head_id {
                return Err(Refusal::SameSequenceDifferentId);
            }
            if l.chain_verified
                && !l
                    .chain
                    .iter()
                    .any(|(s, id)| *s == p.head_seq && *id == p.head_id)
            {
                return Err(Refusal::ForkedFromPin);
            }
        }
    }
    if adopted || pin.as_ref().is_some_and(|p| p.head_seq < h.sequence) {
        let np = Pin {
            agent: h.agent,
            root: h.root,
            head_id: l.head_id,
            head_seq: h.sequence,
            lineage: h.cortex,
            machine_id: ctx.machine_id,
            provenance: h.provenance,
        };
        binding::write_atomic(pin_path, &np).map_err(Refusal::Io)?;
    }
    Ok(Resolved {
        agent: h.agent,
        root: h.root,
        head_id: l.head_id,
        head_seq: h.sequence,
        lineage: h.cortex,
        chain_verified: l.chain_verified,
        adopted,
    })
}
