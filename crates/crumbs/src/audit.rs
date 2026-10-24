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

//! Leak audits: the vocabulary that must never reach a learner, and scanners
//! for byte streams and for a live learner process's memory.

use crate::digest::Digest;
use crate::gen::{registry, Mechanism};
use crate::sealed::{AdversarialClass, Population, Rung, SealedCrumb, SourceFamily};

/// Every human-readable label the sealed side knows (>= 5 chars, so short
/// coincidences in binary data are not mistaken for leaks).
pub fn label_vocabulary() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for f in &registry().families {
        v.push(f.key.clone());
        v.push(f.topic.to_string());
    }
    for m in Mechanism::ALL {
        v.push(m.key().to_string());
    }
    for r in Rung::ALL {
        v.push(format!("{r:?}"));
    }
    for p in Population::ALL {
        v.push(format!("{p:?}"));
    }
    for s in [
        SourceFamily::Capability,
        SourceFamily::Composition,
        SourceFamily::Ambiguous,
        SourceFamily::RabbitHole,
        SourceFamily::Frontier,
        SourceFamily::CleanRoomPhysics,
    ] {
        v.push(format!("{s:?}"));
    }
    for c in [
        AdversarialClass::SimpleWrongRule,
        AdversarialClass::MultiFit,
        AdversarialClass::SpuriousDimension,
        AdversarialClass::PrefixDiverges,
        AdversarialClass::ElegantArbitrary,
        AdversarialClass::InsufficientEvidence,
        AdversarialClass::Noisy,
    ] {
        v.push(format!("{c:?}"));
    }
    for w in [
        "SyntheticNovel",
        "AdversarialSynthetic",
        "HumanTheoryDerived",
        "FrontierDerived",
        "RealWorldMeasured",
        "decoy",
        "rabbit",
        "heldout",
        "held-out",
        "difficulty",
        "prime",
        "gravity",
        "conjecture",
        "algebra",
        "velocity",
        "energy",
        "force",
        "charge",
        "crumbs-registry",
    ] {
        v.push(w.to_string());
    }
    v.retain(|s| s.len() >= 5);
    v.sort();
    v.dedup();
    v
}

/// Labels that exist only in the sealed registry (family keys, mechanism keys,
/// topic phrases). Generic English words are excluded, so this list can be
/// used to scan a learner binary's own code and read-only data.
pub fn sealed_specific_labels() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for f in &registry().families {
        v.push(f.key.clone());
        v.push(f.topic.to_string());
    }
    for m in Mechanism::ALL {
        v.push(m.key().to_string());
    }
    v.push("crumbs-registry".into());
    v.retain(|s| s.len() >= 6);
    v.sort();
    v.dedup();
    v
}

pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Labels found in `bytes` (case-sensitive and lowercase variants).
pub fn find_labels(bytes: &[u8]) -> Vec<String> {
    let mut hits = Vec::new();
    for l in label_vocabulary() {
        if contains(bytes, l.as_bytes()) || contains(bytes, l.to_lowercase().as_bytes()) {
            hits.push(l);
        }
    }
    hits
}

/// Sealed secrets of one crumb that could identify a leak: its digests and
/// every held-out output (and input) value that does not also occur visibly.
pub fn sealed_needles(
    sealed: &SealedCrumb,
    visible_values: &[(Vec<u64>, Vec<u64>)],
    min_value: u64,
) -> Vec<Vec<u8>> {
    let mut vis: Vec<u64> = Vec::new();
    for (i, o) in visible_values {
        vis.extend(i);
        vis.extend(o);
    }
    let mut out: Vec<Vec<u8>> = Vec::new();
    for d in [
        sealed.digest(),
        sealed.generator_instance_digest,
        sealed.heldouts.digest(),
        sealed.generator_code_digest,
    ] {
        out.push(d.0.to_vec());
        out.push(d.hex().into_bytes());
    }
    for t in crate::sealed::HeldoutTier::ALL {
        for ex in sealed.heldouts.tier(t) {
            for v in ex.output.iter().chain(ex.input.iter()) {
                if *v >= min_value && !vis.contains(v) {
                    // Only distinctive encodings count as evidence of a leak: a
                    // mostly-zero u64 matches ordinary length fields by chance.
                    let le = v.to_le_bytes();
                    if le.iter().filter(|b| **b != 0).count() >= 5 {
                        out.push(le.to_vec());
                    }
                    let dec = v.to_string();
                    if dec.len() >= 6 {
                        out.push(dec.into_bytes());
                    }
                }
            }
        }
    }
    out
}

pub fn find_needles(bytes: &[u8], needles: &[Vec<u8>]) -> usize {
    needles.iter().filter(|n| contains(bytes, n)).count()
}

/// Read every writable mapping of a live process (heap, stack, anonymous and
/// data segments). Requires permission to trace the process (a child of ours).
pub fn read_process_writable_memory(pid: u32) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let mut mem = std::fs::File::open(format!("/proc/{pid}/mem"))?;
    let mut out = Vec::new();
    for line in maps.lines() {
        let mut parts = line.split_whitespace();
        let (Some(range), Some(perms)) = (parts.next(), parts.next()) else {
            continue;
        };
        if !perms.starts_with("rw") {
            continue;
        }
        let name = line.split_whitespace().nth(5).unwrap_or("");
        if name.starts_with("[vvar") || name.starts_with("[vsyscall") {
            continue;
        }
        let Some((a, b)) = range.split_once('-') else {
            continue;
        };
        let (a, b) = (
            u64::from_str_radix(a, 16).unwrap_or(0),
            u64::from_str_radix(b, 16).unwrap_or(0),
        );
        if b <= a || b - a > (512 << 20) {
            continue;
        }
        let mut buf = vec![0u8; (b - a) as usize];
        if mem.seek(SeekFrom::Start(a)).is_ok() && mem.read_exact(&mut buf).is_ok() {
            out.extend_from_slice(&buf);
        }
    }
    Ok(out)
}

pub fn digest_needle(d: &Digest) -> Vec<Vec<u8>> {
    vec![d.0.to_vec(), d.hex().into_bytes()]
}
