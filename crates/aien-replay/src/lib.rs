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

//! aien-replay: Rust reference reader/verifier for the TRN1 execution
//! transcript (aien-protocols `specs/execution-transcript/`, contract 0.1.0,
//! wire schema 1).
//!
//! ADR 0024: Rust here is scaffolding. Omega is the destination. This crate
//! only reads bytes, applies the spec's check order (section 7) and prints the
//! spec's outcome lines (section 8). It writes nothing and imports no runtime.

use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

pub const HEADER_LEN: usize = 48;
pub const RECORD_HEADER_LEN: usize = 56;
pub const MAX_RECORDS: u64 = 4096;
pub const MAX_IDENT: u32 = 65536;
pub const MAX_ANNOT: u32 = 65536;
pub const MAX_INPUT_CONTENT: u32 = 4096;
const T_END: u16 = 0xFFFF;

/// Refusal codes, spec section 6.
pub mod code {
    pub const MAGIC: i32 = -1;
    pub const VERSION: i32 = -2;
    pub const LENGTH: i32 = -3;
    pub const NONCANONICAL: i32 = -4;
    pub const UNKNOWN: i32 = -5;
    pub const SHAPE: i32 = -6;
    pub const GAP: i32 = -7;
    pub const CHAIN: i32 = -8;
    pub const DIGEST: i32 = -9;
    pub const TRUNCATED: i32 = -10;
}

/// Mutation hook. `mutations/run_mutants.sh trn1` rewrites this constant to
/// the name of one check, which switches that check off, and requires the
/// shared corpus to notice. In the real build it names no check.
const MUTANT: &str = "";
#[inline(always)]
fn on(check: &str) -> bool {
    MUTANT != check
}

/// Subsystem registry, spec section 5.2.
pub fn subsystem_name(id: u16) -> Option<&'static str> {
    Some(match id {
        0 => "transcript",
        1 => "omega-world",
        2 => "omega-cortex",
        3 => "omega-jspace",
        4 => "aienos-kernel",
        5 => "aienos-argus",
        6 => "aienos-store",
        7 => "sovcore-runtime",
        8 => "sovcore-scheduler",
        9 => "sovcore-kv",
        10 => "external",
        _ => return None,
    })
}

enum IdentRule {
    Exact(u32),
    ExternalInput,
    Crumb,
}

/// Record type registry, spec section 5.1. None = unknown type.
fn ident_rule(t: u16) -> Option<IdentRule> {
    use IdentRule::*;
    Some(match t {
        1 => Exact(32),
        2 | 3 | 5 => Exact(24),
        4 | 14 => Exact(40),
        6 | 10 | 13 => Exact(48),
        7 => Exact(32),
        8 | 9 => Exact(56),
        11 => ExternalInput,
        12 => Exact(16),
        15 => Crumb,
        16 => Exact(200),
        T_END => Exact(16),
        _ => return None,
    })
}

/// An accepted transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub records: u64,
    pub run_id: Hash,
    pub final_digest: Hash,
    /// Per record (index 0 = record 1): compared digest and subsystem id.
    pub compared: Vec<(Hash, u16)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Accept(Accepted),
    Refuse { code: i32, event: u64 },
}

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().expect("4 bytes"))
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
}
fn zero(b: &[u8]) -> bool {
    b.iter().all(|&x| x == 0)
}
fn sha(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

/// RX_CRUMB canonical bytes (spec 5.3): true if they parse to exactly their length.
fn crumb_shape(c: &[u8]) -> bool {
    let mut o = 0usize;
    let need = |o: usize, k: usize| c.len() - o >= k;
    if !need(o, 36) {
        return false;
    }
    let id = u64le(c);
    o += 36;
    for (limit, each) in [(8u32, 24usize), (8, 16), (8, 24)] {
        if !need(o, 4) {
            return false;
        }
        let k = u32le(&c[o..]);
        o += 4;
        if k > limit || !need(o, k as usize * each) {
            return false;
        }
        o += k as usize * each;
    }
    if !need(o, 8) {
        return false; // reason i32 + n_parents u32
    }
    let parents = u32le(&c[o + 4..]);
    o += 8;
    if parents > 65 {
        return false;
    }
    for _ in 0..parents {
        if !need(o, 8) {
            return false;
        }
        let p = u64le(&c[o..]);
        o += 8;
        if p >= 1 && p < id {
            if !need(o, 32) {
                return false;
            }
            o += 32;
        }
    }
    o == c.len()
}

/// Reserved fields inside an identity (step 9). True if all are zero.
fn ident_reserved_ok(t: u16, id: &[u8]) -> bool {
    match t {
        3 => u32le(&id[20..]) == 0,
        6 | 11 | 12 | 14 => u32le(&id[4..]) == 0,
        7 => zero(&id[5..8]),
        10 => u32le(&id[12..]) == 0,
        16 => zero(&id[129..136]),
        T_END => u64le(&id[8..]) == 0,
        _ => true,
    }
}

/// Verify one transcript buffer, spec section 7 check order.
pub fn verify(b: &[u8]) -> Verdict {
    let refuse = |code, event| Verdict::Refuse { code, event };
    let n = b.len();

    // Header.
    if n < 4 {
        return refuse(code::LENGTH, 0);
    }
    if on("magic") && &b[..4] != b"TRN1" {
        return refuse(code::MAGIC, 0);
    }
    if n < 6 {
        return refuse(code::LENGTH, 0);
    }
    if on("version") && u16le(&b[4..]) != 1 {
        return refuse(code::VERSION, 0);
    }
    if n < HEADER_LEN {
        return refuse(code::LENGTH, 0);
    }
    if on("hdr_flags") && u16le(&b[6..]) != 0 {
        return refuse(code::NONCANONICAL, 0);
    }
    let producer = u16le(&b[40..]);
    if on("producer") && (producer == 0 || subsystem_name(producer).is_none()) {
        return refuse(code::UNKNOWN, 0);
    }
    if on("hdr_reserved") && !zero(&b[42..48]) {
        return refuse(code::NONCANONICAL, 0);
    }
    let run_id: Hash = b[8..40].try_into().expect("32 bytes");
    let mut prev = sha(&[&b[..HEADER_LEN]]);
    let mut o = HEADER_LEN;
    let mut compared = Vec::new();
    let mut argus_after: Option<Hash> = None;

    for k in 1u64.. {
        // 1. record header present
        if o == n {
            return if on("truncated") {
                refuse(code::TRUNCATED, k)
            } else {
                refuse(0, k)
            };
        }
        if n - o < RECORD_HEADER_LEN || k > MAX_RECORDS {
            return refuse(code::LENGTH, k);
        }
        let r = &b[o..];
        let (t, sub) = (u16le(r), u16le(&r[2..]));
        let (ilen, alen) = (u32le(&r[4..]), u32le(&r[8..]));
        // 2. known type and subsystem
        let rule = ident_rule(t);
        if on("type") && rule.is_none() {
            return refuse(code::UNKNOWN, k);
        }
        if on("subsystem") && subsystem_name(sub).is_none() {
            return refuse(code::UNKNOWN, k);
        }
        // 3. reserved
        if on("rec_reserved") && u32le(&r[12..]) != 0 {
            return refuse(code::NONCANONICAL, k);
        }
        // 4. length limits
        if on("limit") && (ilen > MAX_IDENT || alen > MAX_ANNOT) {
            return refuse(code::LENGTH, k);
        }
        // 5. sequence
        if on("seq") && u64le(&r[16..]) != k {
            return refuse(code::GAP, k);
        }
        // 6. chain
        if on("prev") && r[24..56] != prev {
            return refuse(code::CHAIN, k);
        }
        // 7. body present
        let body = ilen as u64 + alen as u64;
        if ((n - o - RECORD_HEADER_LEN) as u64) < body {
            return refuse(code::LENGTH, k);
        }
        let rec_len = RECORD_HEADER_LEN + body as usize;
        let id = &r[RECORD_HEADER_LEN..RECORD_HEADER_LEN + ilen as usize];

        // 8. shape
        let len_ok = match rule {
            Some(IdentRule::Exact(x)) => ilen == x,
            Some(IdentRule::ExternalInput) => (48..=48 + MAX_INPUT_CONTENT).contains(&ilen),
            Some(IdentRule::Crumb) => ilen >= 32,
            None => true, // only reachable with the "type" check mutated off
        };
        if on("ident_len") && !len_ok {
            return refuse(code::SHAPE, k);
        }
        // Body fields below are read only when the length is right, so a
        // mutated-off length check cannot index out of range.
        if on("end_subsystem") && (t == T_END) != (sub == 0) {
            return refuse(code::SHAPE, k);
        }
        if on("cap_op") && t == 7 && len_ok && !(1..=6).contains(&id[4]) {
            return refuse(code::SHAPE, k);
        }
        if on("crash_seq") && t == 13 && len_ok && u64le(&id[8..]) >= k {
            return refuse(code::SHAPE, k);
        }
        if on("crumb_shape") && t == 15 && ilen >= 32 && !crumb_shape(&id[32..]) {
            return refuse(code::SHAPE, k);
        }
        if on("argus_flag") && t == 16 && len_ok && id[128] > 1 {
            return refuse(code::SHAPE, k);
        }
        if on("end_annot") && t == T_END && alen != 0 {
            return refuse(code::SHAPE, k);
        }
        // 9. reserved fields inside the identity
        if on("body_reserved") && len_ok && !ident_reserved_ok(t, id) {
            return refuse(code::NONCANONICAL, k);
        }
        // 10. embedded digests
        if t == 15
            && ilen >= 32
            && on("crumb_digest")
            && sha(&[b"AIEN_RX_CAUSAL_V1", &id[32..]]) != id[..32]
        {
            return refuse(code::DIGEST, k);
        }
        if t == 11 && ilen > 48 && on("input_digest") && sha(&[&id[48..]]) != id[16..48] {
            return refuse(code::DIGEST, k);
        }
        if t == 16 && len_ok {
            let before: Hash = id[136..168].try_into().expect("32 bytes");
            let after: Hash = id[168..200].try_into().expect("32 bytes");
            if on("argus_cont") && argus_after.is_some_and(|a| a != before) {
                return refuse(code::DIGEST, k);
            }
            let want = if id[128] != 0 {
                sha(&[&before, &id[..128]])
            } else {
                before
            };
            if on("argus_link") && want != after {
                return refuse(code::DIGEST, k);
            }
            argus_after = Some(after);
        }

        // record digest (chain) and compared digest (replay)
        prev = sha(&[&r[..rec_len]]);
        let cmp = sha(&[b"AIEN_TRN1_CMP", &r[0..4], &r[16..24], &r[4..8], id]);
        compared.push((cmp, sub));
        o += rec_len;

        // 11. END
        if t == T_END {
            if on("end_count") && (ilen < 8 || u64le(id) != k) {
                return refuse(code::GAP, k);
            }
            if on("trailing") && o != n {
                return refuse(code::LENGTH, k + 1);
            }
            return Verdict::Accept(Accepted {
                records: k,
                run_id,
                final_digest: prev,
                compared,
            });
        }
    }
    unreachable!("record loop only exits by return")
}

fn hex(d: &Hash) -> String {
    d.iter().map(|x| format!("{x:02x}")).collect()
}

/// Verify outcome line, spec section 8.
pub fn verdict_line(v: &Verdict) -> String {
    match v {
        Verdict::Accept(a) => format!(
            "ok records={} run={} final={}",
            a.records,
            hex(&a.run_id),
            hex(&a.final_digest)
        ),
        Verdict::Refuse { code, event } => format!("refuse {code} event={event}"),
    }
}

/// Compare outcome line, spec section 8: expected first, actual second.
pub fn compare_line(expected: &Verdict, actual: &Verdict) -> String {
    let (e, a) = match (expected, actual) {
        (Verdict::Refuse { code, event }, _) => {
            return format!("refuse expected {code} event={event}")
        }
        (_, Verdict::Refuse { code, event }) => {
            return format!("refuse actual {code} event={event}")
        }
        (Verdict::Accept(e), Verdict::Accept(a)) => (e, a),
    };
    for (i, (x, y)) in e.compared.iter().zip(&a.compared).enumerate() {
        if x.0 != y.0 {
            return format!(
                "DIVERGENCE event={} expected={} actual={} subsystem={}",
                i + 1,
                hex(&x.0),
                hex(&y.0),
                subsystem_name(x.1).unwrap_or("?")
            );
        }
    }
    format!("MATCH through {}", e.records.min(a.records))
}
