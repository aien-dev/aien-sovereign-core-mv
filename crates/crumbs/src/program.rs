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

//! Candidate Program v1 ("CPG1"): the learner-neutral program format of the
//! generic verifier contract.
//!
//! A learner submits candidates in this format; the sealed verifier executes
//! them with its own interpreter (it never trusts the learner's execution and
//! never needs to know the learner's internals). v1 programs are unary: they
//! map lane 0 of the input to lane 0 of the output as a chain of wrapping u64
//! operations.
//!
//! ```text
//! "CPG1" | version u16 (=1) | n u8 (<= 64) | n x (op u8, imm u64)
//! ```

use crate::canon::{Dec, DecodeError, Enc};
use crate::digest::{digest, Digest, Domain};

pub const MAGIC: &[u8; 4] = b"CPG1";
pub const VERSION: u16 = 1;
pub const MAX_STEPS: usize = 64;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum Op {
    Add = 1,
    Sub = 2,
    Mul = 3,
    And = 4,
    Or = 5,
    Xor = 6,
    Shl = 7,
    Shr = 8,
    Rotl = 9,
}

impl Op {
    pub fn from_u8(v: u8) -> Option<Op> {
        Some(match v {
            1 => Op::Add,
            2 => Op::Sub,
            3 => Op::Mul,
            4 => Op::And,
            5 => Op::Or,
            6 => Op::Xor,
            7 => Op::Shl,
            8 => Op::Shr,
            9 => Op::Rotl,
            _ => return None,
        })
    }

    #[inline]
    pub fn apply(self, x: u64, imm: u64) -> u64 {
        match self {
            Op::Add => x.wrapping_add(imm),
            Op::Sub => x.wrapping_sub(imm),
            Op::Mul => x.wrapping_mul(imm),
            Op::And => x & imm,
            Op::Or => x | imm,
            Op::Xor => x ^ imm,
            Op::Shl => x.wrapping_shl((imm & 63) as u32),
            Op::Shr => x.wrapping_shr((imm & 63) as u32),
            Op::Rotl => x.rotate_left((imm & 63) as u32),
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Step {
    pub op: Op,
    pub imm: u64,
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Default,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Program {
    pub steps: Vec<Step>,
}

impl Program {
    pub fn new(steps: Vec<Step>) -> Self {
        Program { steps }
    }

    pub fn of(steps: &[(Op, u64)]) -> Self {
        Program {
            steps: steps.iter().map(|&(op, imm)| Step { op, imm }).collect(),
        }
    }

    #[inline]
    pub fn run(&self, mut x: u64) -> u64 {
        for s in &self.steps {
            x = s.op.apply(x, s.imm);
        }
        x
    }

    pub fn then(&self, next: &Program) -> Program {
        let mut steps = self.steps.clone();
        steps.extend_from_slice(&next.steps);
        Program { steps }
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.bytes(MAGIC).u16(VERSION).u8(self.steps.len() as u8);
        for s in &self.steps {
            e.u8(s.op as u8).u64(s.imm);
        }
        e.finish()
    }

    pub fn decode(d: &mut Dec<'_>) -> Result<Program, DecodeError> {
        if d.take(4)? != MAGIC || d.u16()? != VERSION {
            return Err(DecodeError);
        }
        let n = d.u8()? as usize;
        if n > MAX_STEPS {
            return Err(DecodeError);
        }
        let mut steps = Vec::with_capacity(n);
        for _ in 0..n {
            let op = Op::from_u8(d.u8()?).ok_or(DecodeError)?;
            steps.push(Step { op, imm: d.u64()? });
        }
        Ok(Program { steps })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Program, DecodeError> {
        let mut d = Dec::new(bytes);
        let p = Program::decode(&mut d)?;
        d.finish()?;
        Ok(p)
    }

    /// Candidate program digest (operation semantic id).
    pub fn digest(&self) -> Digest {
        digest(Domain::Program, &self.to_bytes())
    }

    /// Behaviour signature over the public probe inputs.
    pub fn behavior(&self) -> Digest {
        let mut e = Enc::new();
        for &x in PROBES.iter() {
            e.u64(self.run(x));
        }
        digest(Domain::Behavior, &e.finish())
    }

    /// Every contiguous sub-chain of length in [min_len, max_len].
    pub fn sub_chains(&self, min_len: usize, max_len: usize) -> Vec<Program> {
        let mut out = Vec::new();
        for len in min_len..=max_len.min(self.steps.len()) {
            for start in 0..=self.steps.len() - len {
                out.push(Program {
                    steps: self.steps[start..start + len].to_vec(),
                });
            }
        }
        out
    }
}

/// Identity of this interpreter's semantics (part of the verifier digest).
pub fn std_interpreter_id() -> &'static str {
    "cpg1-interp/1: u64 wrapping add sub mul; and or xor; shl shr rotl by imm&63"
}

/// Public probe inputs used for behaviour signatures (not secret).
pub const PROBES: [u64; 32] = [
    0,
    1,
    2,
    3,
    5,
    7,
    13,
    42,
    100,
    127,
    128,
    255,
    256,
    1000,
    4095,
    65535,
    65536,
    1_000_003,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    0x0123_4567_89AB_CDEF,
    0x0F0F_0F0F_0F0F_0F0F,
    0x5555_5555_5555_5555,
    0xAAAA_AAAA_AAAA_AAAA,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF_FFFF_FFFE,
    u64::MAX,
    0x9E37_79B9_7F4A_7C15,
    0xD1B5_4A32_D192_ED03,
];
