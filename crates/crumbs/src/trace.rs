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

//! Search-trace corpus (the M23-style training data for AIEN_0).
//!
//! The learner reports bare search events (which operation it applied to which
//! node, how it pruned, what it submitted). The sealed side reconstructs every
//! candidate program from those events and the declared operation bank, and
//! derives everything else itself from the *visible* examples: state digests,
//! remaining-mismatch vectors, improvement flags. Held-out results enter a
//! record only on SUBMIT events, and only as the verdict the learner received.
//!
//! No record has a free-text field. Nothing resembling reasoning prose can be
//! stored; records hold operations, states, measurements and outcomes.
//!
//! `CTR1` trace record (fixed 247 bytes incl. chain digest) and `CTS1`
//! training sample (fixed 268 bytes incl. chain digest) are hash-chained with
//! BLAKE3 so a corpus file cannot be edited without detection.

use crate::canon::Enc;
use crate::digest::{digest, Digest, Domain, Hasher};
use crate::program::Program;
use crate::protocol::{ek, CrumbSession};
use crate::verify::Verdict;
use crate::visible::{lane_mask, Encoding, VisibleCrumb};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const TRACE_RECORD_BYTES: usize = 247;
pub const SAMPLE_RECORD_BYTES: usize = 268;
pub const SAMPLE_MAX_OPS: usize = 128;
pub const RESIDUALS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ResultClass {
    /// Generated but pruned (equivalent, over cost, frontier cap).
    Pruned = 1,
    /// Evaluated; mismatch did not shrink.
    Failed = 2,
    /// Evaluated; some examples matched but no improvement on the parent.
    Partial = 3,
    /// Evaluated; mismatch shrank relative to the parent.
    Improved = 4,
    /// Submitted and accepted by the sealed verifier.
    Verified = 5,
    /// Submitted, explained the visible examples, rejected by held-outs.
    RejectedHidden = 6,
    /// A robust (best partial fit) submission that was rejected.
    RejectedRobust = 7,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub event_index: u32,
    pub kind: u8,
    pub result_class: ResultClass,
    pub parent: u32,
    pub child: u32,
    pub op_index: u16,
    pub op_id: Digest,
    pub op_origin: u8,
    pub depth: u16,
    pub prune: u8,
    pub verify: u8,
    pub fit: u8,
    /// 0 unless this is a SUBMIT record; then the verdict byte the learner got.
    pub hidden: u8,
    pub improved: bool,
    pub contributed: bool,
    pub state_digest: Digest,
    pub child_state_digest: Digest,
    pub program_digest: Digest,
    pub mismatch_before: u16,
    pub mismatch_after: u16,
    pub hamming_before: u32,
    pub hamming_after: u32,
    pub exec_cost: u32,
    pub search_cost: u32,
    pub program_steps: u16,
    pub oracle_index: u16,
    pub residual: [u64; RESIDUALS],
}

/// One decision state on a verified solution path, with the operations that
/// were tried there and which of them led onward (equivalence-aware).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingSample {
    pub bank_digest: Digest,
    pub state_digest: Digest,
    pub depth: u8,
    pub solution_depth: u8,
    pub n_examples: u8,
    pub n_ops: u16,
    pub values: [u64; 8],
    pub targets: [u64; 8],
    pub positive_bits: [u8; SAMPLE_MAX_OPS / 8],
    pub tried_bits: [u8; SAMPLE_MAX_OPS / 8],
}

/// Per-crumb summary (the `final:` block of the trace).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSummary {
    pub bank_digest: Digest,
    pub bank_op_ids: Vec<Digest>,
    pub solved: bool,
    pub solution_digest: Option<Digest>,
    pub verified_on_heldout: bool,
    pub records: u32,
    pub samples: u32,
    pub stream_digest: Digest,
    /// Library operations (by op_ref) used in the accepted solution.
    pub library_refs_in_solution: Vec<u32>,
    /// A submitted program disagreed with the program reconstructed from events.
    pub integrity_violations: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CrumbTrace {
    pub records: Vec<TraceRecord>,
    pub samples: Vec<TrainingSample>,
    pub summary: Option<TraceSummary>,
}

struct Node {
    parent: u32,
    op: u16,
    depth: u16,
    vals: Vec<u64>,
    program: Program,
}

fn state_digest(vals: &[u64]) -> Digest {
    let mut e = Enc::new();
    for v in vals {
        e.u64(*v);
    }
    digest(Domain::SearchState, &e.finish())
}

pub fn bank_digest(bank: &[crate::protocol::BankEntry]) -> Digest {
    let mut h = Hasher::new(Domain::LearnerBank);
    for b in bank {
        h.update(&[b.origin]).update(&b.program.digest().0);
    }
    h.finish()
}

/// Derive the canonical trace for one crumb session.
pub fn derive(visible: &VisibleCrumb, session: &CrumbSession) -> CrumbTrace {
    let mut out = CrumbTrace::default();
    let vals_in: Vec<(u64, u64)> = visible
        .values()
        .into_iter()
        .map(|(i, o)| (i[0], o[0]))
        .collect();
    let mask = match visible.encoding {
        Encoding::DecimalUtf8 => u64::MAX,
        Encoding::RawLe => lane_mask(visible.out_lane_bytes),
    };
    let targets: Vec<u64> = vals_in.iter().map(|p| p.1).collect();
    let bank_ids: Vec<Digest> = session.bank.iter().map(|b| b.program.digest()).collect();
    let bdig = bank_digest(&session.bank);

    let measure = |vals: &[u64]| -> (u16, u32, [u64; RESIDUALS]) {
        let mut mism = 0u16;
        let mut ham = 0u32;
        let mut res = [0u64; RESIDUALS];
        for (i, (v, t)) in vals.iter().zip(&targets).enumerate() {
            let v = v & mask;
            if v != *t {
                mism += 1;
            }
            ham += (v ^ t).count_ones();
            if i < RESIDUALS {
                res[i] = t.wrapping_sub(v);
            }
        }
        (mism, ham, res)
    };

    let mut nodes: HashMap<u32, Node> = HashMap::new();
    nodes.insert(
        0,
        Node {
            parent: u32::MAX,
            op: u16::MAX,
            depth: 0,
            vals: vals_in.iter().map(|p| p.0).collect(),
            program: Program::default(),
        },
    );

    // Oracle verdicts by node, from the actual submissions.
    let mut verdict_of: HashMap<u32, Verdict> = HashMap::new();
    let mut integrity_violations = 0;
    for s in &session.submissions {
        verdict_of.insert(s.node, s.verdict);
    }

    for (idx, ev) in session.events.iter().enumerate() {
        let mut r = TraceRecord {
            event_index: idx as u32,
            kind: ev.kind,
            result_class: ResultClass::Failed,
            parent: ev.parent,
            child: ev.child,
            op_index: ev.op_index,
            op_id: Digest::ZERO,
            op_origin: 0,
            depth: 0,
            prune: ev.prune,
            verify: ev.verify,
            fit: ev.fit,
            hidden: 0,
            improved: false,
            contributed: false,
            state_digest: Digest::ZERO,
            child_state_digest: Digest::ZERO,
            program_digest: Digest::ZERO,
            mismatch_before: 0,
            mismatch_after: 0,
            hamming_before: 0,
            hamming_after: 0,
            exec_cost: ev.exec_cost,
            search_cost: idx as u32,
            program_steps: 0,
            oracle_index: ev.oracle_index,
            residual: [0; RESIDUALS],
        };
        match ev.kind {
            ek::EXPAND => {
                let Some(parent) = nodes.get(&ev.parent) else {
                    continue;
                };
                let Some(op) = session.bank.get(ev.op_index as usize) else {
                    continue;
                };
                let vals: Vec<u64> = parent.vals.iter().map(|&x| op.program.run(x)).collect();
                let program = parent.program.then(&op.program);
                let (mb, hb, _) = measure(&parent.vals);
                let (ma, ha, res) = measure(&vals);
                r.op_id = bank_ids[ev.op_index as usize];
                r.op_origin = op.origin;
                r.depth = parent.depth + 1;
                r.state_digest = state_digest(&parent.vals);
                r.child_state_digest = state_digest(&vals);
                r.program_digest = program.digest();
                r.mismatch_before = mb;
                r.mismatch_after = ma;
                r.hamming_before = hb;
                r.hamming_after = ha;
                r.program_steps = program.len() as u16;
                r.residual = res;
                r.improved = ev.prune == 0 && (ma < mb || (ma == mb && ha < hb));
                r.result_class = if ev.prune != 0 {
                    ResultClass::Pruned
                } else if r.improved {
                    ResultClass::Improved
                } else if (ma as usize) < targets.len() {
                    ResultClass::Partial
                } else {
                    ResultClass::Failed
                };
                let depth = r.depth;
                nodes.insert(
                    ev.child,
                    Node {
                        parent: ev.parent,
                        op: ev.op_index,
                        depth,
                        vals,
                        program,
                    },
                );
            }
            ek::SUBMIT => {
                let Some(node) = nodes.get(&ev.parent) else {
                    continue;
                };
                let (m, h, res) = measure(&node.vals);
                r.depth = node.depth;
                r.state_digest = state_digest(&node.vals);
                r.child_state_digest = r.state_digest;
                r.program_digest = node.program.digest();
                r.mismatch_before = m;
                r.mismatch_after = m;
                r.hamming_before = h;
                r.hamming_after = h;
                r.program_steps = node.program.len() as u16;
                r.residual = res;
                let v = verdict_of.get(&ev.parent).copied();
                r.hidden = v.map_or(0, |v| v as u8);
                r.result_class = match v {
                    Some(Verdict::Accept) => ResultClass::Verified,
                    _ if ev.fit == 2 => ResultClass::RejectedRobust,
                    _ => ResultClass::RejectedHidden,
                };
                if let Some(sub) = session.submissions.iter().find(|s| s.node == ev.parent) {
                    if sub.program.run_all_eq(&node.program) == Some(false) {
                        integrity_violations += 1;
                    }
                }
            }
            _ => continue,
        }
        out.records.push(r);
    }

    // Backfill: which transitions contributed to the accepted solution?
    let accepted = session
        .submissions
        .iter()
        .find(|s| s.verdict == Verdict::Accept);
    let mut library_refs = Vec::new();
    let mut path_states: HashSet<(u16, Digest)> = HashSet::new();
    let mut path_nodes: Vec<u32> = Vec::new();
    if let Some(acc) = accepted {
        let mut n = acc.node;
        while let Some(node) = nodes.get(&n) {
            path_nodes.push(n);
            path_states.insert((node.depth, state_digest(&node.vals)));
            if node.op != u16::MAX {
                let b = &session.bank[node.op as usize];
                if b.origin == 1 {
                    library_refs.push(b.op_ref);
                }
            }
            if node.parent == u32::MAX {
                break;
            }
            n = node.parent;
        }
        library_refs.sort_unstable();
        for r in out.records.iter_mut() {
            if r.kind == ek::EXPAND && path_states.contains(&(r.depth, r.child_state_digest)) {
                r.contributed = true;
            }
            if r.kind == ek::SUBMIT && r.parent == acc.node {
                r.contributed = true;
            }
        }
        // Training samples: one per decision state on the path. A submission
        // whose node was never reported in the event stream yields none.
        path_nodes.reverse(); // root first
        let solution_depth = path_nodes.len().saturating_sub(1) as u8;
        for &pn in path_nodes.iter().take(path_nodes.len().saturating_sub(1)) {
            let node = &nodes[&pn];
            let sd = state_digest(&node.vals);
            let mut s = TrainingSample {
                bank_digest: bdig,
                state_digest: sd,
                depth: node.depth as u8,
                solution_depth,
                n_examples: targets.len().min(255) as u8,
                n_ops: session.bank.len() as u16,
                values: [0; 8],
                targets: [0; 8],
                positive_bits: [0; SAMPLE_MAX_OPS / 8],
                tried_bits: [0; SAMPLE_MAX_OPS / 8],
            };
            let k = 8.min(targets.len());
            s.values[..k].copy_from_slice(&node.vals[..k]);
            s.targets[..k].copy_from_slice(&targets[..k]);
            for r in out
                .records
                .iter()
                .filter(|r| r.kind == ek::EXPAND && r.parent == pn)
            {
                let i = r.op_index as usize;
                if i < SAMPLE_MAX_OPS {
                    s.tried_bits[i / 8] |= 1 << (i % 8);
                    if r.contributed {
                        s.positive_bits[i / 8] |= 1 << (i % 8);
                    }
                }
            }
            out.samples.push(s);
        }
    }

    let stream = stream_digest(&out.records, &out.samples);
    out.summary = Some(TraceSummary {
        bank_digest: bdig,
        bank_op_ids: bank_ids,
        solved: accepted.is_some(),
        solution_digest: accepted.map(|a| a.program.digest()),
        verified_on_heldout: accepted.is_some(),
        records: out.records.len() as u32,
        samples: out.samples.len() as u32,
        stream_digest: stream,
        library_refs_in_solution: library_refs,
        integrity_violations,
    });
    out
}

impl Program {
    /// Compare behaviour on the public probes (None if either is empty).
    pub fn run_all_eq(&self, other: &Program) -> Option<bool> {
        if self.is_empty() || other.is_empty() {
            return None;
        }
        Some(self.behavior() == other.behavior())
    }
}

impl TraceRecord {
    pub fn encode_body(&self, e: &mut Enc) {
        e.bytes(b"CTR1")
            .u16(1)
            .u8(self.kind)
            .u8(self.result_class as u8);
        e.u32(self.event_index)
            .u32(self.parent)
            .u32(self.child)
            .u16(self.op_index)
            .u8(self.op_origin);
        e.u8(self.prune)
            .u8(self.verify)
            .u8(self.fit)
            .u8(self.hidden);
        e.u8(self.improved as u8)
            .u8(self.contributed as u8)
            .u16(self.depth);
        e.digest(&self.op_id)
            .digest(&self.state_digest)
            .digest(&self.child_state_digest)
            .digest(&self.program_digest);
        e.u16(self.mismatch_before)
            .u16(self.mismatch_after)
            .u32(self.hamming_before)
            .u32(self.hamming_after);
        e.u32(self.exec_cost)
            .u32(self.search_cost)
            .u16(self.program_steps)
            .u16(self.oracle_index);
        for r in self.residual {
            e.u64(r);
        }
    }
}

impl TrainingSample {
    pub fn encode_body(&self, e: &mut Enc) {
        e.bytes(b"CTS1").u16(1).u16(self.n_ops);
        e.digest(&self.bank_digest).digest(&self.state_digest);
        e.u8(self.depth)
            .u8(self.solution_depth)
            .u8(self.n_examples)
            .u8(0);
        for v in self.values {
            e.u64(v);
        }
        for v in self.targets {
            e.u64(v);
        }
        e.bytes(&self.positive_bits).bytes(&self.tried_bits);
    }
}

/// Encode records with a BLAKE3 chain; `chain` carries the previous digest.
pub fn encode_records(records: &[TraceRecord], chain: &mut Digest) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * TRACE_RECORD_BYTES);
    for r in records {
        let mut e = Enc::new();
        r.encode_body(&mut e);
        let body = e.finish();
        debug_assert_eq!(body.len(), TRACE_RECORD_BYTES - 32);
        let d = Hasher::new(Domain::TraceRecord)
            .update(&chain.0)
            .update(&body)
            .finish();
        out.extend_from_slice(&body);
        out.extend_from_slice(&d.0);
        *chain = d;
    }
    out
}

pub fn encode_samples(samples: &[TrainingSample], chain: &mut Digest) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * SAMPLE_RECORD_BYTES);
    for s in samples {
        let mut e = Enc::new();
        s.encode_body(&mut e);
        let body = e.finish();
        debug_assert_eq!(body.len(), SAMPLE_RECORD_BYTES - 32);
        let d = Hasher::new(Domain::TrainingSample)
            .update(&chain.0)
            .update(&body)
            .finish();
        out.extend_from_slice(&body);
        out.extend_from_slice(&d.0);
        *chain = d;
    }
    out
}

fn stream_digest(records: &[TraceRecord], samples: &[TrainingSample]) -> Digest {
    let mut c1 = Digest::ZERO;
    let a = encode_records(records, &mut c1);
    let mut c2 = Digest::ZERO;
    let b = encode_samples(samples, &mut c2);
    let _ = (a, b);
    let mut e = Enc::new();
    e.digest(&c1).digest(&c2);
    digest(Domain::TraceStream, &e.finish())
}

/// Re-verify a chained record file. Returns the number of records or None.
pub fn verify_chain(bytes: &[u8], record_len: usize, domain: Domain) -> Option<usize> {
    if !bytes.len().is_multiple_of(record_len) {
        return None;
    }
    let mut chain = Digest::ZERO;
    for rec in bytes.chunks(record_len) {
        let (body, d) = rec.split_at(record_len - 32);
        let want = Hasher::new(domain).update(&chain.0).update(body).finish();
        if want.0 != d {
            return None;
        }
        chain = want;
    }
    Some(bytes.len() / record_len)
}
