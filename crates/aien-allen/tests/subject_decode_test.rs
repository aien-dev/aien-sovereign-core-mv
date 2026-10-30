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

//! Decoder refusals (unsupported dialect, damage) and the id/derivation rules.
mod support;
use aien_allen::subject_v0::{decode, object_id};
use support::*;

fn good() -> Vec<u8> {
    encode(&fx("a"), 1, [0; 32], [7, 1000])
}

#[test]
fn good_object_decodes() {
    let s = decode(&good()).unwrap();
    assert_eq!(s.sequence, 1);
    assert_eq!(s.intents.len(), 1);
    assert_eq!(s.intents[0].payload, [7, 1000]);
}

#[test]
fn unsupported_dialect_is_refused() {
    let mut b = good();
    b[8] = 1; // format version 1
    assert!(decode(&b).unwrap_err().contains("version"));
    let mut b = good();
    b[..8].copy_from_slice(b"AIENSUBX");
    assert!(decode(&b).unwrap_err().contains("magic"));
    // unknown intent kind (offset: header 16 + fixed body 184 = 200; id 32, supersedes 32, kind at +64)
    let mut b = good();
    b[200 + 64] = 2;
    assert!(decode(&b).is_err());
    // unknown origin
    let mut b = good();
    b[16 + 32 * 5 + 8] = 9;
    assert!(decode(&b).unwrap_err().contains("origin"));
}

#[test]
fn damage_is_refused() {
    let g = good();
    for n in 0..g.len() {
        assert!(decode(&g[..n]).is_err(), "prefix {n}");
    }
    let mut b = g.clone();
    b.push(0);
    assert!(decode(&b).is_err());
    // intent id must derive from content
    let mut b = g.clone();
    b[200 + 64 + 4 + 8 + 8] ^= 1; // payload[0]
    assert!(decode(&b).unwrap_err().contains("derive"));
}

#[test]
fn zero_provenance_and_zero_ids_refused() {
    let mut f = fx("a");
    f.provenance = [0; 32];
    assert!(decode(&encode(&f, 1, [0; 32], [7, 1000]))
        .unwrap_err()
        .contains("provenance"));
    let mut f = fx("a");
    f.agent = [0; 32];
    assert!(decode(&encode(&f, 1, [0; 32], [7, 1000])).is_err());
    // sequence 2 with zero previous
    assert!(decode(&encode(&fx("a"), 2, [0; 32], [7, 1000])).is_err());
    // genesis naming a previous
    assert!(decode(&encode(&fx("a"), 1, [9; 32], [7, 1000])).is_err());
}

#[test]
fn object_id_depends_on_every_byte() {
    let g = good();
    let id = object_id(&g);
    for i in (0..g.len()).step_by(7) {
        let mut b = g.clone();
        b[i] ^= 1;
        assert_ne!(object_id(&b), id);
    }
}
