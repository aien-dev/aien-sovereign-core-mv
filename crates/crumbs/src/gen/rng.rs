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

//! Deterministic, named random streams (xoshiro256** seeded from BLAKE3).
//! A stream is fully determined by its name and integer parts, so every
//! generator draw can be replayed exactly years later.

use crate::digest::{digest, Domain};

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn named(name: &str, parts: &[u64]) -> Rng {
        let mut bytes = Vec::with_capacity(name.len() + 1 + parts.len() * 8);
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        for p in parts {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        let d = digest(Domain::Rng, &bytes).0;
        let mut s = [0u64; 4];
        for (i, w) in s.iter_mut().enumerate() {
            *w = u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap());
        }
        if s == [0; 4] {
            s[0] = 1;
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, n); n == 0 means the full u64 range.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return self.next_u64();
        }
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < limit {
                return v % n;
            }
        }
    }

    /// Uniform in [lo, hi] inclusive.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo {
            return lo;
        }
        let span = hi - lo;
        if span == u64::MAX {
            return self.next_u64();
        }
        lo + self.below(span + 1)
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }

    pub fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}
