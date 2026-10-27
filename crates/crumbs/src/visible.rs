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

//! Crumb v1 learner-visible contract.
//!
//! The visible side is an ordered list of opaque `INPUT_BYTES -> OUTPUT_BYTES`
//! examples plus the minimum structure needed to parse them. It has no field
//! that could carry a name, topic, family, difficulty, population, decoy flag,
//! provenance, explanation, solution or held-out example.
//!
//! Canonical bytes (little-endian):
//!
//! ```text
//! "CRB1"                    magic
//! schema_version  u16       = 1
//! encoding        u8        1 = decimal UTF-8 lanes, 2 = raw little-endian lanes
//! flags           u8        bit0 = budget present; other bits must be 0
//! in_arity        u8        lanes per input  (1..=8)
//! out_arity       u8        lanes per output (1..=8)
//! in_lane_bytes   u8        raw: 1,2,4,8; decimal: 0
//! out_lane_bytes  u8        raw: 1,2,4,8; decimal: 0
//! n               u32       examples (1..=64)
//! n x (u32 len, input bytes, u32 len, output bytes)
//! [max_candidates u32, max_depth u32, max_oracle_queries u32, max_program_ops u32]
//! ```
//!
//! Decimal lanes are canonical base-10 u64 values (no sign, no leading zeros)
//! separated by one ASCII space. `CrumbDigest` = BLAKE3 over exactly these
//! bytes, so it covers the schema version and nothing else.

use crate::canon::{Dec, DecodeError, Enc};
use crate::digest::{digest, CrumbDigest, Digest, Domain};

pub const MAGIC: &[u8; 4] = b"CRB1";
pub const SCHEMA_VERSION: u16 = 1;
pub const MAX_ARITY: u8 = 8;
pub const MAX_EXAMPLES: usize = 64;
const FLAG_BUDGET: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Encoding {
    DecimalUtf8 = 1,
    RawLe = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Budget {
    pub max_candidates: u32,
    pub max_depth: u32,
    pub max_oracle_queries: u32,
    pub max_program_ops: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub input: Vec<u8>,
    pub output: Vec<u8>,
}

/// The only problem representation a learner may receive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleCrumb {
    pub encoding: Encoding,
    pub in_arity: u8,
    pub out_arity: u8,
    pub in_lane_bytes: u8,
    pub out_lane_bytes: u8,
    pub examples: Vec<Example>,
    pub budget: Option<Budget>,
}

/// Visible-side errors are bare codes: they cannot carry crumb content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibleError {
    Magic,
    Version,
    Shape,
    Lane,
    Length,
    NonCanonical,
}

impl std::fmt::Display for VisibleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "visible crumb rejected: {self:?}")
    }
}

impl std::error::Error for VisibleError {}

impl From<DecodeError> for VisibleError {
    fn from(_: DecodeError) -> Self {
        VisibleError::Length
    }
}

fn lane_bytes_ok(encoding: Encoding, b: u8) -> bool {
    match encoding {
        Encoding::DecimalUtf8 => b == 0,
        Encoding::RawLe => matches!(b, 1 | 2 | 4 | 8),
    }
}

pub fn lane_mask(lane_bytes: u8) -> u64 {
    if lane_bytes == 0 || lane_bytes >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * lane_bytes as u32)) - 1
    }
}

/// Encode lane values into example bytes.
pub fn encode_lanes(encoding: Encoding, lane_bytes: u8, lanes: &[u64]) -> Vec<u8> {
    match encoding {
        Encoding::DecimalUtf8 => lanes
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
            .into_bytes(),
        Encoding::RawLe => {
            let mut out = Vec::with_capacity(lanes.len() * lane_bytes as usize);
            for v in lanes {
                out.extend_from_slice(&v.to_le_bytes()[..lane_bytes as usize]);
            }
            out
        }
    }
}

/// Strictly parse example bytes into lane values.
pub fn decode_lanes(
    encoding: Encoding,
    lane_bytes: u8,
    arity: u8,
    bytes: &[u8],
) -> Result<Vec<u64>, VisibleError> {
    match encoding {
        Encoding::DecimalUtf8 => {
            let text = std::str::from_utf8(bytes).map_err(|_| VisibleError::Lane)?;
            let parts: Vec<&str> = text.split(' ').collect();
            if parts.len() != arity as usize {
                return Err(VisibleError::Lane);
            }
            let mut out = Vec::with_capacity(parts.len());
            for p in parts {
                if p.is_empty()
                    || !p.bytes().all(|c| c.is_ascii_digit())
                    || (p.len() > 1 && p.starts_with('0'))
                {
                    return Err(VisibleError::NonCanonical);
                }
                out.push(p.parse::<u64>().map_err(|_| VisibleError::Lane)?);
            }
            Ok(out)
        }
        Encoding::RawLe => {
            let lb = lane_bytes as usize;
            if bytes.len() != lb * arity as usize {
                return Err(VisibleError::Lane);
            }
            Ok(bytes
                .chunks(lb)
                .map(|c| {
                    let mut b = [0u8; 8];
                    b[..lb].copy_from_slice(c);
                    u64::from_le_bytes(b)
                })
                .collect())
        }
    }
}

impl VisibleCrumb {
    /// Build a crumb from lane values; values are masked to the lane width.
    pub fn from_values(
        encoding: Encoding,
        in_lane_bytes: u8,
        out_lane_bytes: u8,
        pairs: &[(Vec<u64>, Vec<u64>)],
        budget: Option<Budget>,
    ) -> Result<Self, VisibleError> {
        if pairs.is_empty() || pairs.len() > MAX_EXAMPLES {
            return Err(VisibleError::Length);
        }
        let in_arity = pairs[0].0.len();
        let out_arity = pairs[0].1.len();
        if in_arity == 0
            || out_arity == 0
            || in_arity > MAX_ARITY as usize
            || out_arity > MAX_ARITY as usize
        {
            return Err(VisibleError::Shape);
        }
        if !lane_bytes_ok(encoding, in_lane_bytes) || !lane_bytes_ok(encoding, out_lane_bytes) {
            return Err(VisibleError::Shape);
        }
        let (im, om) = (lane_mask(in_lane_bytes), lane_mask(out_lane_bytes));
        let mut examples = Vec::with_capacity(pairs.len());
        for (i, o) in pairs {
            if i.len() != in_arity || o.len() != out_arity {
                return Err(VisibleError::Shape);
            }
            let i: Vec<u64> = i.iter().map(|v| v & im).collect();
            let o: Vec<u64> = o.iter().map(|v| v & om).collect();
            examples.push(Example {
                input: encode_lanes(encoding, in_lane_bytes, &i),
                output: encode_lanes(encoding, out_lane_bytes, &o),
            });
        }
        Ok(VisibleCrumb {
            encoding,
            in_arity: in_arity as u8,
            out_arity: out_arity as u8,
            in_lane_bytes,
            out_lane_bytes,
            examples,
            budget,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.bytes(MAGIC).u16(SCHEMA_VERSION).u8(self.encoding as u8);
        e.u8(if self.budget.is_some() {
            FLAG_BUDGET
        } else {
            0
        });
        e.u8(self.in_arity)
            .u8(self.out_arity)
            .u8(self.in_lane_bytes)
            .u8(self.out_lane_bytes);
        e.u32(self.examples.len() as u32);
        for ex in &self.examples {
            e.blob(&ex.input).blob(&ex.output);
        }
        if let Some(b) = &self.budget {
            e.u32(b.max_candidates)
                .u32(b.max_depth)
                .u32(b.max_oracle_queries)
                .u32(b.max_program_ops);
        }
        e.finish()
    }

    /// Strict decode: the bytes must be exactly what `to_bytes` would produce.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VisibleError> {
        let mut d = Dec::new(bytes);
        if d.take(4)? != MAGIC {
            return Err(VisibleError::Magic);
        }
        if d.u16()? != SCHEMA_VERSION {
            return Err(VisibleError::Version);
        }
        let encoding = match d.u8()? {
            1 => Encoding::DecimalUtf8,
            2 => Encoding::RawLe,
            _ => return Err(VisibleError::Shape),
        };
        let flags = d.u8()?;
        if flags & !FLAG_BUDGET != 0 {
            return Err(VisibleError::NonCanonical);
        }
        let (in_arity, out_arity, in_lb, out_lb) = (d.u8()?, d.u8()?, d.u8()?, d.u8()?);
        if !(1..=MAX_ARITY).contains(&in_arity) || !(1..=MAX_ARITY).contains(&out_arity) {
            return Err(VisibleError::Shape);
        }
        if !lane_bytes_ok(encoding, in_lb) || !lane_bytes_ok(encoding, out_lb) {
            return Err(VisibleError::Shape);
        }
        let n = d.u32()? as usize;
        if n == 0 || n > MAX_EXAMPLES {
            return Err(VisibleError::Length);
        }
        let mut examples = Vec::with_capacity(n);
        for _ in 0..n {
            let input = d.blob(8 * 21)?.to_vec();
            let output = d.blob(8 * 21)?.to_vec();
            decode_lanes(encoding, in_lb, in_arity, &input)?;
            decode_lanes(encoding, out_lb, out_arity, &output)?;
            examples.push(Example { input, output });
        }
        let budget = if flags & FLAG_BUDGET != 0 {
            Some(Budget {
                max_candidates: d.u32()?,
                max_depth: d.u32()?,
                max_oracle_queries: d.u32()?,
                max_program_ops: d.u32()?,
            })
        } else {
            None
        };
        d.finish()?;
        let c = VisibleCrumb {
            encoding,
            in_arity,
            out_arity,
            in_lane_bytes: in_lb,
            out_lane_bytes: out_lb,
            examples,
            budget,
        };
        if c.to_bytes() != bytes {
            return Err(VisibleError::NonCanonical);
        }
        Ok(c)
    }

    pub fn digest(&self) -> CrumbDigest {
        digest(Domain::CrumbDigest, &self.to_bytes())
    }

    /// Digest of the example list alone (the "visible-set digest" in evidence).
    pub fn visible_set_digest(&self) -> Digest {
        let mut e = Enc::new();
        e.u32(self.examples.len() as u32);
        for ex in &self.examples {
            e.blob(&ex.input).blob(&ex.output);
        }
        digest(Domain::VisibleSet, &e.finish())
    }

    /// Parsed lane values of every example.
    pub fn values(&self) -> Vec<(Vec<u64>, Vec<u64>)> {
        self.examples
            .iter()
            .map(|ex| {
                (
                    decode_lanes(self.encoding, self.in_lane_bytes, self.in_arity, &ex.input)
                        .expect("validated"),
                    decode_lanes(
                        self.encoding,
                        self.out_lane_bytes,
                        self.out_arity,
                        &ex.output,
                    )
                    .expect("validated"),
                )
            })
            .collect()
    }
}
