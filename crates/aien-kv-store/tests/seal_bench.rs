#![cfg(feature = "std")]
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

//! Host micro-benchmark behind the cipher choice of ADR 0003 section 20:
//! AES-256-GCM against ChaCha20-Poly1305 (both from `ring`), sealing and
//! opening 1 MiB and 16 MiB buffers in place, as the store does.
//!
//! Ignored by default (timing, not correctness). Run on the host with:
//! `cargo test -p aien-kv-store --release --test seal_bench -- --ignored --nocapture`
//! Numbers are host-only; the restore cost on the Spark is measured there.

use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM, CHACHA20_POLY1305};
use std::time::Instant;

fn rate(alg: &'static ring::aead::Algorithm, len: usize, reps: u32) -> (f64, f64) {
    let key = LessSafeKey::new(UnboundKey::new(alg, &[0x42; 32]).unwrap());
    let mut buf = vec![0x5Au8; len];
    let aad = [7u8; 64];
    // Warm up once (page faults, CPU feature detection).
    let tag = key
        .seal_in_place_separate_tag(
            Nonce::assume_unique_for_key([0; 12]),
            Aad::from(&aad),
            &mut buf,
        )
        .unwrap();
    key.open_in_place_separate_tag(
        Nonce::assume_unique_for_key([0; 12]),
        Aad::from(&aad),
        tag,
        &mut buf,
        0..,
    )
    .unwrap();

    let mut tags = Vec::with_capacity(reps as usize);
    let t0 = Instant::now();
    for i in 0..reps {
        let mut n = [0u8; 12];
        n[..4].copy_from_slice(&i.to_le_bytes());
        tags.push(
            key.seal_in_place_separate_tag(
                Nonce::assume_unique_for_key(n),
                Aad::from(&aad),
                &mut buf,
            )
            .unwrap(),
        );
        // Reopen at once so every pass works on the same ciphertext state.
        key.open_in_place_separate_tag(
            Nonce::assume_unique_for_key(n),
            Aad::from(&aad),
            tags[i as usize],
            &mut buf,
            0..,
        )
        .unwrap();
    }
    let both = t0.elapsed().as_secs_f64();

    let mut n = [0u8; 12];
    n[11] = 1;
    let t1 = Instant::now();
    for _ in 0..reps {
        let tag = key
            .seal_in_place_separate_tag(Nonce::assume_unique_for_key(n), Aad::from(&aad), &mut buf)
            .unwrap();
        let _ = std::hint::black_box(tag);
    }
    let seal = t1.elapsed().as_secs_f64();
    let open = (both - seal).max(f64::MIN_POSITIVE);
    let mib = (len as f64 * reps as f64) / (1024.0 * 1024.0);
    (mib / seal, mib / open)
}

#[test]
#[ignore = "host timing benchmark; run with --release -- --ignored --nocapture"]
fn seal_cipher_throughput_aes256gcm_vs_chacha20poly1305() {
    for (len, reps) in [(1usize << 20, 200u32), (16 << 20, 20)] {
        for (name, alg) in [
            ("AES-256-GCM", &AES_256_GCM),
            ("ChaCha20-Poly1305", &CHACHA20_POLY1305),
        ] {
            let (s, o) = rate(alg, len, reps);
            println!(
                "{name:18} {:>3} MiB  seal {s:8.0} MiB/s  open {o:8.0} MiB/s",
                len >> 20
            );
        }
    }
}
