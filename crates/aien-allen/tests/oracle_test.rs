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

//! Known-answer vectors produced by the REFERENCE C implementation, aienos
//! 61bd76a `native/kernel/svc/continuity_subject.c` (cs_subject_genesis,
//! cs_subject_intend, cs_subject_advance, cs_subject_hold,
//! cs_subject_bind_cortex, cs_subject_encode, cs_subject_id), compiled with the
//! host sources of `make -C native/kernel test-continuity-subject-restart`
//! (CPU only; that target also passed: CK_CONTINUITY_SUBJECT_RESTART PASS).
//! Vector A: genesis, root 0x11.., agent 0x22.., one intent (7, 1000).
//! Vector B: successor of A, bound to Cortex lineage 0x77.., a second intent
//! (9, 2000) and one held knowledge digest 0x5c.. record 17.
//! The format is PROPOSED (ADR 0018): if the C encoder changes, regenerate.
mod support;
use aien_allen::subject_v0::{decode, object_id};
use support::*;

const A_HEX: &str = "4149454e5355424a000000001801000011111111111111111111111111111111111111111111111111111111111111112222222222222222222222222222222222222222222222222222222222222222000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000aabbcc0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d010000000000000001000000000000000e1a1094faedf6000d0a1b24a8b379b5ade90e0258fd34a86f98d37c624c94020000000000000000000000000000000000000000000000000000000000000000010000000100000001000000000000000700000000000000e803000000000000";
const A_ID: &str = "0c1ef058eeba1e58ba0f3a4da741f10c69f91ada838f187d5e8d5626b1b446c0";
const B_HEX: &str = "4149454e5355424a00000000a8010000111111111111111111111111111111111111111111111111111111111111111122222222222222222222222222222222222222222222222222222222222222220c1ef058eeba1e58ba0f3a4da741f10c69f91ada838f187d5e8d5626b1b446c002000000000000007777777777777777777777777777777777777777777777777777777777777777aabbcc0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d0100000000000000020001000000000003aec9f97a530f279c35bb8d99025c2feb0f2448aff977510f6012d90b5206510000000000000000000000000000000000000000000000000000000000000000010000000100000002000000000000000900000000000000d0070000000000000e1a1094faedf6000d0a1b24a8b379b5ade90e0258fd34a86f98d37c624c94020000000000000000000000000000000000000000000000000000000000000000010000000100000001000000000000000700000000000000e8030000000000005c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c11000000000000000200000000000000";
const B_ID: &str = "9b579318dccfe570234c66bd82e8cc6656e9aae68907f469751ea84016f9bb6d";

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

#[test]
fn reference_object_ids_match() {
    assert_eq!(aien_allen::hex(&object_id(&unhex(A_HEX))), A_ID);
    assert_eq!(aien_allen::hex(&object_id(&unhex(B_HEX))), B_ID);
}

#[test]
fn reference_objects_decode() {
    let a = decode(&unhex(A_HEX)).unwrap();
    assert_eq!((a.sequence, a.root, a.agent), (1, [0x11; 32], [0x22; 32]));
    assert_eq!(a.intents.len(), 1);
    assert_eq!(a.intents[0].payload, [7, 1000]);
    let b = decode(&unhex(B_HEX)).unwrap();
    assert_eq!(b.sequence, 2);
    assert_eq!(b.previous.to_vec(), unhex(A_ID));
    assert_eq!(b.cortex, [0x77; 32]);
    assert_eq!(b.intents.len(), 2);
    assert_eq!(b.knowledge.len(), 1);
    assert_eq!(b.knowledge[0].0, [0x5c; 32]);
}

#[test]
fn fixture_encoder_matches_the_reference_bytes() {
    let mut f = fx("ignored");
    f.root = [0x11; 32];
    f.agent = [0x22; 32];
    f.cortex = [0; 32];
    f.provenance = unhex("aabbcc0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d")
        .try_into()
        .unwrap();
    assert_eq!(encode(&f, 1, [0; 32], [7, 1000]), unhex(A_HEX));
}

#[test]
fn flipping_any_reference_byte_is_caught() {
    // Negative control: every single-byte change of B either fails to decode
    // or changes the object id (so a pin on the id would notice).
    let b = unhex(B_HEX);
    for i in 0..b.len() {
        let mut m = b.clone();
        m[i] ^= 1;
        assert!(
            decode(&m).is_err() || object_id(&m) != object_id(&b),
            "byte {i}"
        );
    }
}
