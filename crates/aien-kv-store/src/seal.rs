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

//! Encryption at rest of host record files (ADR 0003 section 20, #506).
//!
//! A sealed object is the same `header region (4096 B) || payload` file as
//! before, with two differences: the payload bytes are AES-256-GCM
//! ciphertext (same length: the cipher is a stream mode), and the last 32
//! bytes of the header region, the **seal slot** at `SEAL_SLOT_OFFSET`, hold:
//!
//! | region offset | size | field |
//! |---|---|---|
//! | 4064 | 3 | magic `AKS` |
//! | 4067 | 1 | sealed format version = 2 (1 was the plaintext layout) |
//! | 4068 | 12 | nonce, fresh from the OS random source for every file |
//! | 4080 | 16 | AEAD tag |
//!
//! Associated data: region bytes `0..4068`, i.e. the 256-byte identity
//! header (namespace id, chunk key, parent, geometry, payload root, header
//! checksum), the segment digest table, the padding, and the magic and
//! version. Any change to any of them, a wrong key, or one flipped
//! ciphertext, nonce or tag bit makes `open_payload` refuse. The caller then
//! discards the object and recomputes; nothing is restored from it.
//!
//! The key: 32 bytes, given to the process as 64 hex digits in
//! `AIEN_KV_STORE_KEY` (`atlas-vault run AIEN_KV_STORE_KEY -- <daemon>`). It is
//! read once, removed from the environment, held in a buffer that is zeroed
//! on drop, and never printed: `Debug` is redacted and refusals carry no key
//! text. Key rotation: objects sealed under an old key fail to open and are
//! discarded like any other refused object.

use crate::record::{HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use core::sync::atomic::{AtomicU64, Ordering};
use ring::aead::{Aad, LessSafeKey, Nonce, Tag, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use zeroize::{Zeroize, Zeroizing};

/// The environment variable the key arrives in.
pub const KEY_ENV: &str = "AIEN_KV_STORE_KEY";
pub const KEY_LEN: usize = 32;
pub const SEAL_MAGIC: [u8; 3] = *b"AKS";
/// Host object format: 1 = plaintext payload (no longer read), 2 = sealed.
pub const SEALED_FORMAT_VERSION: u8 = 2;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;
const O_MAGIC: usize = SEAL_SLOT_OFFSET;
const O_VERSION: usize = SEAL_SLOT_OFFSET + 3;
const O_NONCE: usize = SEAL_SLOT_OFFSET + 4;
const O_TAG: usize = O_NONCE + NONCE_LEN;
/// Region bytes bound as associated data: everything before the nonce.
pub const AAD_LEN: usize = O_NONCE;
const _: () = assert!(O_TAG + TAG_LEN == HEADER_REGION_LEN);

/// Why no key could be loaded. Carries no key text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRefusal {
    /// `AIEN_KV_STORE_KEY` is not set.
    Missing,
    /// Set, but not exactly 64 hex digits.
    Malformed,
}

impl KeyRefusal {
    /// The status reason (`kv_store_refused_reason`).
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Missing => "key_missing",
            Self::Malformed => "key_malformed",
        }
    }
}

impl core::fmt::Display for KeyRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Missing => write!(f, "key_missing: {KEY_ENV} is not set"),
            Self::Malformed => write!(
                f,
                "key_malformed: {KEY_ENV} is not exactly 64 hex digits (32 bytes)"
            ),
        }
    }
}

/// Why an object was not sealed or not opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealError {
    /// Fewer bytes than a header region.
    Short,
    /// No seal magic: an old plaintext-format file (or not ours).
    NotSealed,
    /// Seal magic with another sealed format version.
    UnsupportedVersion,
    /// Authentication failed: wrong key, tampered bytes or swapped header.
    Rejected,
    /// The OS random source failed (sealing only).
    Random,
}

/// The store key. Zeroed on drop; never printed.
pub struct SealKey {
    bytes: Zeroizing<[u8; KEY_LEN]>,
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl SealKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        let mut b = bytes;
        let key = Self {
            bytes: Zeroizing::new(b),
        };
        b.zeroize();
        key
    }

    /// A fresh random key (tests, benchmarks).
    pub fn generate() -> Result<Self, SealError> {
        let mut key = Self {
            bytes: Zeroizing::new([0u8; KEY_LEN]),
        };
        SystemRandom::new()
            .fill(&mut key.bytes[..])
            .map_err(|_| SealError::Random)?;
        Ok(key)
    }

    /// Exactly 64 hex digits, either case; anything else is `Malformed`
    /// (no trimming: a stray newline is a refusal, not a guess).
    pub fn from_hex(hex: &str) -> Result<Self, KeyRefusal> {
        let h = hex.as_bytes();
        if h.len() != 2 * KEY_LEN {
            return Err(KeyRefusal::Malformed);
        }
        let mut key = Self {
            bytes: Zeroizing::new([0u8; KEY_LEN]),
        };
        for (i, [hi, lo]) in h.as_chunks::<2>().0.iter().enumerate() {
            match (hex_val(*hi), hex_val(*lo)) {
                (Some(a), Some(b)) => key.bytes[i] = (a << 4) | b,
                _ => return Err(KeyRefusal::Malformed),
            }
        }
        Ok(key)
    }

    /// The key from `KEY_ENV` through any lookup (tests pass a closure). The
    /// looked-up text is zeroed after parsing.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, KeyRefusal> {
        let Some(text) = get(KEY_ENV).map(Zeroizing::new) else {
            return Err(KeyRefusal::Missing);
        };
        Self::from_hex(&text)
    }

    /// The key from the process environment, which no longer holds it
    /// afterwards (whether or not it parsed), so child processes do not
    /// inherit it. Call before other threads read the environment.
    pub fn take_from_process_env() -> Result<Self, KeyRefusal> {
        let got = Self::from_lookup(|n| std::env::var(n).ok());
        let malformed_os = std::env::var_os(KEY_ENV).is_some() && got.is_err();
        std::env::remove_var(KEY_ENV);
        match got {
            // Present but not valid UTF-8 reads as missing above.
            Err(KeyRefusal::Missing) if malformed_os => Err(KeyRefusal::Malformed),
            other => other,
        }
    }

    fn aead(&self) -> LessSafeKey {
        LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &self.bytes[..]).expect("32-byte key"))
    }

    /// Seal `file` = `region || payload` in place: writes the seal slot (magic,
    /// version, fresh nonce, tag) and encrypts the payload. The file length is
    /// unchanged.
    pub fn seal_file(&self, file: &mut [u8]) -> Result<(), SealError> {
        let t0 = std::time::Instant::now();
        let got = self.seal_inner(file);
        note_timing(
            &SEAL_TIMING,
            file.len().saturating_sub(HEADER_REGION_LEN),
            t0,
        );
        got
    }

    fn seal_inner(&self, file: &mut [u8]) -> Result<(), SealError> {
        if file.len() < HEADER_REGION_LEN {
            return Err(SealError::Short);
        }
        let (region, payload) = file.split_at_mut(HEADER_REGION_LEN);
        region[O_MAGIC..O_MAGIC + 3].copy_from_slice(&SEAL_MAGIC);
        region[O_VERSION] = SEALED_FORMAT_VERSION;
        let mut nonce = [0u8; NONCE_LEN];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| SealError::Random)?;
        region[O_NONCE..O_TAG].copy_from_slice(&nonce);
        let tag = self
            .aead()
            .seal_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(&region[..AAD_LEN]),
                payload,
            )
            .map_err(|_| SealError::Rejected)?;
        region[O_TAG..].copy_from_slice(tag.as_ref());
        Ok(())
    }

    /// Authenticate and decrypt `payload` (the bytes after `region` in a
    /// sealed file) in place. On any error the payload buffer is zeroed and
    /// must be discarded; nothing in it may be used.
    pub fn open_payload(&self, region: &[u8], payload: &mut [u8]) -> Result<(), SealError> {
        let t0 = std::time::Instant::now();
        let got = self.open_inner(region, payload);
        note_timing(&OPEN_TIMING, payload.len(), t0);
        if got.is_err() {
            payload.fill(0);
            if got == Err(SealError::Rejected) {
                OPEN_REJECTED.fetch_add(1, Ordering::Relaxed);
            }
        }
        got
    }

    fn open_inner(&self, region: &[u8], payload: &mut [u8]) -> Result<(), SealError> {
        if region.len() < HEADER_REGION_LEN {
            return Err(SealError::Short);
        }
        check_marker(region)?;
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&region[O_NONCE..O_TAG]);
        let mut tag = [0u8; TAG_LEN];
        tag.copy_from_slice(&region[O_TAG..HEADER_REGION_LEN]);
        self.aead()
            .open_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(&region[..AAD_LEN]),
                Tag::from(tag),
                payload,
                0..,
            )
            .map(|_| ())
            .map_err(|_| SealError::Rejected)
    }
}

fn check_marker(region: &[u8]) -> Result<(), SealError> {
    if region[O_MAGIC..O_MAGIC + 3] != SEAL_MAGIC {
        return Err(SealError::NotSealed);
    }
    if region[O_VERSION] != SEALED_FORMAT_VERSION {
        return Err(SealError::UnsupportedVersion);
    }
    Ok(())
}

/// The region carries the sealed-format marker (magic and version 2). Checks
/// no key: recovery uses it to discard old plaintext-format files without
/// reading their payload.
pub fn is_sealed_region(region: &[u8]) -> bool {
    region.len() >= HEADER_REGION_LEN && check_marker(region).is_ok()
}

impl core::fmt::Debug for SealKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SealKey(redacted)")
    }
}

impl PartialEq for SealKey {
    /// Constant time over the 32 bytes.
    fn eq(&self, other: &Self) -> bool {
        self.bytes
            .iter()
            .zip(other.bytes.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

impl Eq for SealKey {}

static KEY_REFUSALS: AtomicU64 = AtomicU64::new(0);
static OPEN_REJECTED: AtomicU64 = AtomicU64::new(0);

/// Count one store-key refusal (the runtime calls it when it refuses persistence).
pub fn note_key_refusal() {
    KEY_REFUSALS.fetch_add(1, Ordering::Relaxed);
}

/// Store-key refusals in this process.
pub fn key_refusals_total() -> u64 {
    KEY_REFUSALS.load(Ordering::Relaxed)
}

/// Objects whose authentication failed in this process (each was discarded).
pub fn open_rejected_total() -> u64 {
    OPEN_REJECTED.load(Ordering::Relaxed)
}

/// Calls, payload bytes and wall nanoseconds of one seal operation (#508 encryption cost).
struct Timing {
    calls: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
}

static SEAL_TIMING: Timing = Timing {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    ns: AtomicU64::new(0),
};
static OPEN_TIMING: Timing = Timing {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    ns: AtomicU64::new(0),
};

fn note_timing(t: &Timing, bytes: usize, t0: std::time::Instant) {
    let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
    t.calls.fetch_add(1, Ordering::Relaxed);
    t.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    t.ns.fetch_add(ns, Ordering::Relaxed);
}

/// Totals of [`SealKey::seal_file`] and [`SealKey::open_payload`] in this process: every call
/// (refused ones included), the payload bytes it covered and the wall time it took. The
/// encryption cost of persistence is `seal_ns` per parked byte and `open_ns` per restored byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SealTiming {
    pub seal_calls: u64,
    pub seal_bytes: u64,
    pub seal_ns: u64,
    pub open_calls: u64,
    pub open_bytes: u64,
    pub open_ns: u64,
}

pub fn seal_timing() -> SealTiming {
    let r = |a: &AtomicU64| a.load(Ordering::Relaxed);
    SealTiming {
        seal_calls: r(&SEAL_TIMING.calls),
        seal_bytes: r(&SEAL_TIMING.bytes),
        seal_ns: r(&SEAL_TIMING.ns),
        open_calls: r(&OPEN_TIMING.calls),
        open_bytes: r(&OPEN_TIMING.bytes),
        open_ns: r(&OPEN_TIMING.ns),
    }
}
