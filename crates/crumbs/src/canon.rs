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

//! Canonical little-endian encoding primitives. All identities are computed
//! over bytes produced here, never over serde/JSON output.

use crate::digest::Digest;

#[derive(Default, Clone)]
pub struct Enc {
    pub buf: Vec<u8>,
}

impl Enc {
    pub fn new() -> Self {
        Enc { buf: Vec::new() }
    }
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(b);
        self
    }
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }
    pub fn digest(&mut self, d: &Digest) -> &mut Self {
        self.bytes(&d.0)
    }
    /// u32 length prefix + bytes.
    pub fn blob(&mut self, b: &[u8]) -> &mut Self {
        self.u32(b.len() as u32).bytes(b)
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.blob(s.as_bytes())
    }
    pub fn u64s(&mut self, v: &[u64]) -> &mut Self {
        self.u32(v.len() as u32);
        for x in v {
            self.u64(*x);
        }
        self
    }
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// Decoding error: a bare code, never content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeError;

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("malformed canonical bytes")
    }
}

impl std::error::Error for DecodeError {}

pub struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Dec { buf, pos: 0 }
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if n > self.buf.len() - self.pos {
            return Err(DecodeError);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn digest(&mut self) -> Result<Digest, DecodeError> {
        Ok(Digest(self.take(32)?.try_into().unwrap()))
    }
    pub fn blob(&mut self, max: usize) -> Result<&'a [u8], DecodeError> {
        let n = self.u32()? as usize;
        if n > max {
            return Err(DecodeError);
        }
        self.take(n)
    }
    pub fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
    pub fn finish(&self) -> Result<(), DecodeError> {
        if self.done() {
            Ok(())
        } else {
            Err(DecodeError)
        }
    }
}
