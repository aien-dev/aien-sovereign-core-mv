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
//!
//! # Nonce source (#708)
//!
//! Production [`SealKey::seal_file`] fills the 96-bit nonce from
//! [`ring::rand::SystemRandom`] and from no other source. A failed fill
//! returns [`SealError::Random`] and leaves the caller's buffer unchanged.
//! There is no environment variable, no config field, and no fallback nonce.
//!
//! Crate unit tests may attach a source with `with_nonce_source`. That method
//! exists only when the crate is built for its own unit tests (`cfg(test)`).
//! Integration tests and production builds do not compile it.
//!
//! # Usage budget (#811)
//!
//! Nonces are independent uniform 96-bit draws. For `q` draws under one key
//! the probability of a collision is at most `q(q-1)/2^97` (NIST SP 800-38D,
//! section 8.3). [`RECOMMENDED_SEALS_PER_KEY`] is `2^20` (bound `2^-57`).
//! [`NIST_SP800_38D_RANDOM_NONCE_LIMIT`] is `2^32` (bound about `2^-33`).
//! A finite sample of distinct nonces is not a uniqueness proof.
//!
//! [`SealKey::seal_file`] enforces the `2^20` budget for one store directory.
//! The caller passes that directory's [`USAGE_DIR_NAME`] directory. The count
//! is a file named by the SHA-256 of the key bytes (the key itself is not
//! written). The file is read, and `count + 1` is made durable, before the
//! caller's buffer is modified. A stored count that is already `2^20` returns
//! [`SealError::Budget`] and leaves the buffer unchanged. A count that cannot
//! be read or persisted returns [`SealError::UsageStore`] and does not seal.
//!
//! Order, under an exclusive `flock` on `seal-usage/LOCK`:
//!
//! 1. A buffer shorter than a header region returns [`SealError::Short`]. The
//!    count file is not touched. The nonce is not drawn.
//! 2. The nonce is drawn into a stack buffer. [`SealError::Random`] returns
//!    here. The count file and the caller's buffer stay as they were.
//! 3. The count file is read (a missing file is zero; a short, long, or
//!    unrecognized file is [`SealError::UsageStore`]). At `2^20` the call
//!    returns [`SealError::Budget`].
//! 4. `count + 1` is written to a temporary file in the same directory, that
//!    file is `fsync`ed, the temporary name is renamed onto the count file,
//!    and the directory is `fsync`ed. Any failure is [`SealError::UsageStore`].
//! 5. Only then is the seal slot written and the payload encrypted.
//!
//! A crash before the directory `fsync` keeps the previous count when the
//! rename has not reached disk. A crash after that `fsync` and before the
//! sealed object is published has counted a seal that is not on disk. The
//! count is never decremented: the budget is a ceiling. A failed directory
//! `fsync` after a rename that did reach disk is the same direction (the
//! seal does not run, and a later open may observe the higher count).
//!
//! The file is per store directory and per key fingerprint. Restarting that
//! directory sees it. Sealers that share the directory serialize on the lock.
//! The host adapter already refuses a second open of the same directory.
//! Another directory, including a copy on another host, has its own file and
//! is not included. A missing file starts at zero and does not backfill
//! objects already in the directory. `2^32` is not a second refusal; the
//! `2^20` check comes first. Rotation still discards objects sealed under the
//! previous key. This crate does not migrate them.
//!
//! `Nonce::assume_unique_for_key` does not search for an earlier nonce, and
//! neither does this module. A repeat under one key makes the CTR keystream
//! reusable: the XOR of the two ciphertexts equals the XOR of the two
//! plaintexts. Each tag still authenticates its own file.
//!
//! The nonce that was used is stored in the seal slot. Process exit does not
//! rewind the kernel CSPRNG, and copying the store directory back into place
//! does not rewind it either. A restored file opens because its nonce is in
//! the file. A snapshot of a running VM can include kernel entropy; this
//! crate does not detect that case and does not reseed.

use crate::record::{HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use core::sync::atomic::{AtomicU64, Ordering};
use ring::aead::{Aad, LessSafeKey, Nonce, Tag, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use zeroize::{Zeroize, Zeroizing};

/// The environment variable the key arrives in.
pub const KEY_ENV: &str = "AIEN_KV_STORE_KEY";
pub const KEY_LEN: usize = 32;
pub const SEAL_MAGIC: [u8; 3] = *b"AKS";
/// Host object format: 1 = plaintext payload (no longer read), 2 = sealed.
pub const SEALED_FORMAT_VERSION: u8 = 2;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;
/// Operator rekey budget. [`SealKey::seal_file`] returns [`SealError::Budget`]
/// when the persisted count for this key in the given usage directory is
/// already this value. At this count the NIST SP 800-38D collision bound is
/// `2^-57`.
pub const RECOMMENDED_SEALS_PER_KEY: u64 = 1 << 20;
/// NIST SP 800-38D guidance for random 96-bit nonces: stay below this many
/// invocations per key. The enforced ceiling is [`RECOMMENDED_SEALS_PER_KEY`],
/// which is below this value, so `seal_file` does not consult this constant.
pub const NIST_SP800_38D_RANDOM_NONCE_LIMIT: u64 = 1 << 32;
/// Directory under the store root that holds one count file per key fingerprint.
pub const USAGE_DIR_NAME: &str = "seal-usage";
const USAGE_MAGIC: [u8; 4] = *b"AKSU";
const USAGE_VERSION: u8 = 1;
const USAGE_LEN: usize = 16;
const O_MAGIC: usize = SEAL_SLOT_OFFSET;
const O_VERSION: usize = SEAL_SLOT_OFFSET + 3;
const O_NONCE: usize = SEAL_SLOT_OFFSET + 4;
const O_TAG: usize = O_NONCE + NONCE_LEN;
/// Region bytes bound as associated data: everything before the nonce.
pub const AAD_LEN: usize = O_NONCE;
const _: () = assert!(O_TAG + TAG_LEN == HEADER_REGION_LEN);

/// Fills the 96-bit nonce for one seal.
///
/// [`OsNonceSource`] is the production implementation. A failed fill is
/// [`SealError::Random`]. Callers must not replace a failure with a fixed
/// nonce.
pub(crate) trait NonceSource {
    fn fill_nonce(&self, out: &mut [u8; NONCE_LEN]) -> Result<(), SealError>;
}

struct OsNonceSource;

impl NonceSource for OsNonceSource {
    fn fill_nonce(&self, out: &mut [u8; NONCE_LEN]) -> Result<(), SealError> {
        SystemRandom::new()
            .fill(&mut out[..])
            .map_err(|_| SealError::Random)
    }
}

static OS_NONCE: OsNonceSource = OsNonceSource;

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
    /// The OS random source failed (sealing only). The buffer is unchanged.
    Random,
    /// The persisted seal count for this key in the usage directory is already
    /// [`RECOMMENDED_SEALS_PER_KEY`]. The buffer is unchanged.
    Budget,
    /// The usage count could not be read or made durable. The buffer is unchanged
    /// and the payload was not encrypted.
    UsageStore,
}

/// The store key. Zeroed on drop; never printed.
pub struct SealKey {
    bytes: Zeroizing<[u8; KEY_LEN]>,
    /// Unit-test override. Absent from non-test builds. `None` selects
    /// [`OsNonceSource`].
    #[cfg(test)]
    nonce_source: Option<std::sync::Arc<dyn NonceSource + Send + Sync>>,
    /// Next `commit_usage` fails before it writes. The count file stays put.
    #[cfg(test)]
    fail_usage_persist: core::sync::atomic::AtomicBool,
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
    fn from_parts(bytes: Zeroizing<[u8; KEY_LEN]>) -> Self {
        Self {
            bytes,
            #[cfg(test)]
            nonce_source: None,
            #[cfg(test)]
            fail_usage_persist: core::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        let mut b = bytes;
        let key = Self::from_parts(Zeroizing::new(b));
        b.zeroize();
        key
    }

    /// Attach a nonce source. Compiled only for this crate's unit tests.
    /// Production `seal_file` has no equivalent and reads no nonce setting.
    #[cfg(test)]
    pub(crate) fn with_nonce_source(
        mut self,
        source: std::sync::Arc<dyn NonceSource + Send + Sync>,
    ) -> Self {
        self.nonce_source = Some(source);
        self
    }

    /// A fresh random key (tests, benchmarks). The key bytes come from the OS
    /// source. This is separate from the per-file nonce.
    pub fn generate() -> Result<Self, SealError> {
        let mut raw = [0u8; KEY_LEN];
        SystemRandom::new()
            .fill(&mut raw)
            .map_err(|_| SealError::Random)?;
        let key = Self::from_bytes(raw);
        raw.zeroize();
        Ok(key)
    }

    /// Exactly 64 hex digits, either case; anything else is `Malformed`
    /// (no trimming: a stray newline is a refusal, not a guess).
    pub fn from_hex(hex: &str) -> Result<Self, KeyRefusal> {
        let h = hex.as_bytes();
        if h.len() != 2 * KEY_LEN {
            return Err(KeyRefusal::Malformed);
        }
        let mut raw = [0u8; KEY_LEN];
        for (i, [hi, lo]) in h.as_chunks::<2>().0.iter().enumerate() {
            match (hex_val(*hi), hex_val(*lo)) {
                (Some(a), Some(b)) => raw[i] = (a << 4) | b,
                _ => {
                    raw.zeroize();
                    return Err(KeyRefusal::Malformed);
                }
            }
        }
        let key = Self::from_bytes(raw);
        raw.zeroize();
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

    /// The next persisted count update fails before it writes. The following
    /// `seal_file` returns [`SealError::UsageStore`] and leaves the buffer
    /// unchanged. Compiled only for this crate's unit tests.
    #[cfg(test)]
    pub(crate) fn testing_fail_next_usage_persist(&self) {
        self.fail_usage_persist
            .store(true, core::sync::atomic::Ordering::SeqCst);
    }

    /// Seal `file` = `region || payload` in place: writes the seal slot (magic,
    /// version, fresh nonce, tag) and encrypts the payload. The file length is
    /// unchanged.
    ///
    /// `usage_dir` is the store's [`USAGE_DIR_NAME`] directory. One use of this
    /// key is recorded there, and that record is durable, before this modifies
    /// `file`. [`SealError::Random`], [`SealError::Budget`] and
    /// [`SealError::UsageStore`] leave the buffer as the caller passed it.
    pub fn seal_file(&self, usage_dir: &Path, file: &mut [u8]) -> Result<(), SealError> {
        let t0 = std::time::Instant::now();
        let got = self.seal_counted(usage_dir, file);
        note_timing(
            &SEAL_TIMING,
            file.len().saturating_sub(HEADER_REGION_LEN),
            t0,
        );
        got
    }

    fn active_source(&self) -> &dyn NonceSource {
        #[cfg(test)]
        if let Some(source) = self.nonce_source.as_deref() {
            return source;
        }
        &OS_NONCE
    }

    fn seal_counted(&self, usage_dir: &Path, file: &mut [u8]) -> Result<(), SealError> {
        if file.len() < HEADER_REGION_LEN {
            return Err(SealError::Short);
        }
        // Draw before the count is committed and before any write to `file`.
        // A failed source must not mark the buffer as sealed, must not encrypt,
        // and must not consume a budget slot. The AEAD call stays in this
        // function: `fill_nonce` overwrites the stack buffer, and a helper
        // parameter named `nonce` is a hard-coded-nonce sink for that zero
        // fill (the query does not treat the trait write as a barrier).
        let source = self.active_source();
        let mut nonce = [0u8; NONCE_LEN];
        source.fill_nonce(&mut nonce)?;
        // The count is durable before `file` changes. A persist error returns
        // with the plaintext still in the caller's buffer.
        self.commit_usage(usage_dir)?;
        let (region, payload) = file.split_at_mut(HEADER_REGION_LEN);
        region[O_MAGIC..O_MAGIC + 3].copy_from_slice(&SEAL_MAGIC);
        region[O_VERSION] = SEALED_FORMAT_VERSION;
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

    fn commit_usage(&self, usage_dir: &Path) -> Result<(), SealError> {
        require_usage_dir(usage_dir)?;
        let _lock = lock_usage_dir(usage_dir)?;
        let name = fingerprint_name(&self.bytes);
        let count = read_count_file(&usage_dir.join(&name))?;
        if count >= RECOMMENDED_SEALS_PER_KEY {
            return Err(SealError::Budget);
        }
        #[cfg(test)]
        if self
            .fail_usage_persist
            .swap(false, core::sync::atomic::Ordering::SeqCst)
        {
            return Err(SealError::UsageStore);
        }
        write_count_file(usage_dir, &name, count + 1)
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

/// Write `count` as this key's persisted usage, without the budget check.
/// Unit tests use it to stand at `2^20` without sealing that many files.
/// The bytes are the same encoding `commit_usage` writes.
#[cfg(test)]
pub(crate) fn testing_install_usage_count(
    usage_dir: &Path,
    key: &SealKey,
    count: u64,
) -> Result<(), SealError> {
    require_usage_dir(usage_dir)?;
    let _lock = lock_usage_dir(usage_dir)?;
    write_count_file(usage_dir, &fingerprint_name(&key.bytes), count)
}

/// The persisted count for `key` in `usage_dir`. A missing file is zero.
#[cfg(test)]
pub(crate) fn testing_read_usage_count(usage_dir: &Path, key: &SealKey) -> Result<u64, SealError> {
    require_usage_dir(usage_dir)?;
    let _lock = lock_usage_dir(usage_dir)?;
    read_count_file(&usage_dir.join(fingerprint_name(&key.bytes)))
}

/// Path of the count file. Tests check the fingerprint name independently
/// by hashing the key with SHA-256 themselves.
#[cfg(test)]
pub(crate) fn testing_usage_path(usage_dir: &Path, key: &SealKey) -> PathBuf {
    usage_dir.join(fingerprint_name(&key.bytes))
}

fn fingerprint_name(key: &[u8; KEY_LEN]) -> String {
    let dig: [u8; 32] = Sha256::digest(key).into();
    let mut hex = String::with_capacity(64);
    const NIBBLE: &[u8; 16] = b"0123456789abcdef";
    for byte in dig {
        hex.push(NIBBLE[(byte >> 4) as usize] as char);
        hex.push(NIBBLE[(byte & 0x0f) as usize] as char);
    }
    hex
}

fn require_usage_dir(dir: &Path) -> Result<(), SealError> {
    match std::fs::symlink_metadata(dir) {
        Ok(meta) if meta.file_type().is_dir() => Ok(()),
        _ => Err(SealError::UsageStore),
    }
}

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

fn lock_usage_dir(dir: &Path) -> Result<File, SealError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::os::unix::io::AsRawFd;
    let path = dir.join("LOCK");
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&path)
        .map_err(|_| SealError::UsageStore)?;
    let meta = f.metadata().map_err(|_| SealError::UsageStore)?;
    if !meta.is_file() || meta.uid() != euid() {
        return Err(SealError::UsageStore);
    }
    if meta.mode() & 0o777 != 0o600 {
        f.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|_| SealError::UsageStore)?;
    }
    // SAFETY: flock takes an advisory lock on a descriptor we own. The lock
    // ends when `f` is dropped, which is the end of the read-modify-write.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(SealError::UsageStore);
    }
    Ok(f)
}

fn create_usage_file(path: &Path) -> Result<File, SealError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| SealError::UsageStore)?;
    let meta = f.metadata().map_err(|_| SealError::UsageStore)?;
    // Truncate only after the owner check, so a planted file we do not own
    // is left as it was.
    if !meta.is_file() || meta.uid() != euid() {
        return Err(SealError::UsageStore);
    }
    f.set_len(0).map_err(|_| SealError::UsageStore)?;
    if meta.mode() & 0o777 != 0o600 {
        f.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|_| SealError::UsageStore)?;
    }
    Ok(f)
}

fn read_count_file(path: &Path) -> Result<u64, SealError> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(_) => return Err(SealError::UsageStore),
    };
    let meta = f.metadata().map_err(|_| SealError::UsageStore)?;
    if !meta.is_file() || meta.uid() != euid() || meta.len() != USAGE_LEN as u64 {
        return Err(SealError::UsageStore);
    }
    let mut buf = [0u8; USAGE_LEN];
    f.read_exact(&mut buf).map_err(|_| SealError::UsageStore)?;
    if buf[0..4] != USAGE_MAGIC || buf[4] != USAGE_VERSION || buf[5..8] != [0, 0, 0] {
        return Err(SealError::UsageStore);
    }
    let mut le = [0u8; 8];
    le.copy_from_slice(&buf[8..16]);
    Ok(u64::from_le_bytes(le))
}

fn write_count_file(dir: &Path, name: &str, count: u64) -> Result<(), SealError> {
    let final_path = dir.join(name);
    let part_path = dir.join(format!("{name}.part"));
    {
        let mut f = create_usage_file(&part_path)?;
        f.write_all(&encode_count(count))
            .map_err(|_| SealError::UsageStore)?;
        f.sync_all().map_err(|_| SealError::UsageStore)?;
    }
    std::fs::rename(&part_path, &final_path).map_err(|_| SealError::UsageStore)?;
    fsync_usage_dir(dir)
}

fn encode_count(count: u64) -> [u8; USAGE_LEN] {
    let mut buf = [0u8; USAGE_LEN];
    buf[0..4].copy_from_slice(&USAGE_MAGIC);
    buf[4] = USAGE_VERSION;
    buf[8..16].copy_from_slice(&count.to_le_bytes());
    buf
}

fn fsync_usage_dir(dir: &Path) -> Result<(), SealError> {
    use std::os::unix::fs::OpenOptionsExt;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_DIRECTORY)
        .open(dir)
        .map_err(|_| SealError::UsageStore)?;
    f.sync_all().map_err(|_| SealError::UsageStore)
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
