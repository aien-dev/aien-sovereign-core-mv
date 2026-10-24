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

//! EvidenceReceiptV1 (ADR 0028 Decision 4): canonical JSON, sha256 digest,
//! content-addressed stdout/stderr blobs, immutable write (never overwrite),
//! and read-back of the receipts already in a store.

use crate::verdict::{DepFact, Verdict};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

pub const SCHEMA: &str = "aien-test/EvidenceReceiptV1";

/// Makes every blob temp file name unique inside one process, so gates
/// running on different threads can store the same bytes (an empty stderr,
/// say) at the same moment.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// A float or other value that canonical encoding forbids (C3).
    Canon(String),
    Io(String),
    /// A file already exists at the content address with different bytes.
    Collision(String),
    /// Bytes read back after writing did not match their digest.
    Verify(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Canon(s) => write!(f, "canonical encoding error: {s}"),
            StoreError::Io(s) => write!(f, "io error: {s}"),
            StoreError::Collision(s) => write!(f, "evidence collision: {s}"),
            StoreError::Verify(s) => write!(f, "evidence verify failed: {s}"),
        }
    }
}

fn io_err(what: &str, e: std::io::Error) -> StoreError {
    StoreError::Io(format!("{what}: {e}"))
}

fn write_str(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes())
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

fn write_value(v: &Value, out: &mut Vec<u8>) -> Result<(), StoreError> {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.extend_from_slice(i.to_string().as_bytes());
            } else if let Some(u) = n.as_u64() {
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                return Err(StoreError::Canon(format!(
                    "floating point not allowed: {n}"
                )));
            }
        }
        Value::String(s) => write_str(s, out),
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(x, out)?;
            }
            out.push(b']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(k, out);
                out.push(b':');
                write_value(&m[k.as_str()], out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// Canonical bytes (C1 to C5): sorted keys, integers only, minimal escapes.
pub fn canonical(v: &Value) -> Result<Vec<u8>, StoreError> {
    let mut out = Vec::new();
    write_value(v, &mut out)?;
    Ok(out)
}

pub fn canonical_digest(v: &Value) -> Result<String, StoreError> {
    Ok(sha256_hex(&canonical(v)?))
}

pub fn build_receipt(core: Value, volatile: Value) -> Value {
    json!({ "schema": SCHEMA, "core": core, "volatile": volatile })
}

pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn new(dir: &Path) -> Store {
        Store {
            dir: dir.to_path_buf(),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Store a blob at `blobs/<sha256>.log` via temp file and rename, then
    /// re-verify. An existing blob with the same digest is success; an
    /// existing file with other bytes is a collision.
    pub fn put_blob(&self, bytes: &[u8]) -> Result<String, StoreError> {
        let sha = sha256_hex(bytes);
        let blobs = self.dir.join("blobs");
        fs::create_dir_all(&blobs).map_err(|e| io_err("create blobs dir", e))?;
        let path = blobs.join(format!("{sha}.log"));
        if path.exists() {
            let existing = fs::read(&path).map_err(|e| io_err("read blob", e))?;
            if sha256_hex(&existing) != sha {
                return Err(StoreError::Collision(format!(
                    "blob {sha} exists with other bytes"
                )));
            }
            return Ok(sha);
        }
        let n = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let tmp = blobs.join(format!(".tmp-{sha}-{}-{n}", std::process::id()));
        {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o444)
                .open(&tmp)
                .map_err(|e| io_err("create blob temp", e))?;
            f.write_all(bytes).map_err(|e| io_err("write blob", e))?;
            f.sync_all().map_err(|e| io_err("sync blob", e))?;
        }
        fs::rename(&tmp, &path).map_err(|e| io_err("rename blob", e))?;
        let back = fs::read(&path).map_err(|e| io_err("read back blob", e))?;
        if sha256_hex(&back) != sha {
            return Err(StoreError::Verify(format!(
                "blob {sha} digest mismatch after write"
            )));
        }
        Ok(sha)
    }

    /// True if the stdout/stderr blob `sha` is present in this store (it is
    /// not read or re-hashed here).
    pub fn blob_exists(&self, sha: &str) -> bool {
        sha.len() == 64
            && sha
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            && self.dir.join("blobs").join(format!("{sha}.log")).is_file()
    }

    /// Write `<digest>.json` exclusively, mode 0444, canonical bytes plus one
    /// LF. Identical existing content is success; different content is a
    /// fatal collision. Never overwrites.
    pub fn put_receipt(&self, receipt: &Value) -> Result<(String, PathBuf), StoreError> {
        let bytes = canonical(receipt)?;
        let digest = sha256_hex(&bytes);
        fs::create_dir_all(&self.dir).map_err(|e| io_err("create evidence dir", e))?;
        let path = self.dir.join(format!("{digest}.json"));
        let mut want = bytes;
        want.push(b'\n');
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true).mode(0o444);
        match opts.open(&path) {
            Ok(mut f) => {
                f.write_all(&want).map_err(|e| io_err("write receipt", e))?;
                f.sync_all().map_err(|e| io_err("sync receipt", e))?;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                let existing = fs::read(&path).map_err(|e| io_err("read existing receipt", e))?;
                if existing != want {
                    return Err(StoreError::Collision(format!(
                        "receipt {digest} exists with different bytes"
                    )));
                }
            }
            Err(e) => return Err(io_err("open receipt", e)),
        }
        let back = fs::read(&path).map_err(|e| io_err("read back receipt", e))?;
        if back != want {
            return Err(StoreError::Verify(format!(
                "receipt {digest} differs after write"
            )));
        }
        Ok((digest, path))
    }

    /// The bytes of blob `sha`, only if they still hash to `sha`.
    pub fn read_blob(&self, sha: &str) -> Option<Vec<u8>> {
        if !self.blob_exists(sha) {
            return None;
        }
        let bytes = fs::read(self.dir.join("blobs").join(format!("{sha}.log"))).ok()?;
        (sha256_hex(&bytes) == sha).then_some(bytes)
    }

    /// The one receipt `<digest>.json`, with the same checks `Index::load`
    /// makes (bytes hash to the name, JSON, receipt schema). None otherwise.
    pub fn get_receipt(&self, digest: &str) -> Option<Stored> {
        if !is_sha256_hex(digest) {
            return None;
        }
        let path = self.dir.join(format!("{digest}.json"));
        let bytes = fs::read(&path).ok()?;
        let body = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
        if sha256_hex(body) != digest {
            return None;
        }
        let value: Value = serde_json::from_slice(body).ok()?;
        if value["schema"] != SCHEMA {
            return None;
        }
        let modified = fs::metadata(&path).and_then(|m| m.modified()).ok();
        Some(Stored {
            digest: digest.to_string(),
            value,
            modified,
        })
    }
}

/// A receipt read back from a store. `digest` comes from the file name and
/// has been checked against the file's bytes.
#[derive(Debug, Clone)]
pub struct Stored {
    pub digest: String,
    pub value: Value,
    modified: Option<SystemTime>,
}

impl Stored {
    fn text(&self, section: &str, key: &str) -> &str {
        self.value[section][key].as_str().unwrap_or("")
    }

    pub fn gate(&self) -> &str {
        self.text("core", "gate")
    }

    pub fn commit(&self) -> &str {
        self.text("core", "commit")
    }

    pub fn reason(&self) -> &str {
        self.text("core", "reason")
    }

    pub fn finished_utc(&self) -> &str {
        self.text("volatile", "finished_utc")
    }

    pub fn verdict(&self) -> Option<Verdict> {
        Verdict::parse(self.text("core", "derived_verdict"))
    }
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// All verified receipts of one store, oldest first.
#[derive(Debug, Clone, Default)]
pub struct Index {
    receipts: Vec<Stored>,
}

impl Index {
    /// Read every `<digest>.json` in the store. A file is skipped, never
    /// trusted, if its bytes do not hash to its name, it is not JSON, or it
    /// does not carry the receipt schema. Order is by recorded finish time,
    /// then by file modification time (receipts finished in the same second),
    /// then by digest.
    pub fn load(store: &Store) -> Index {
        let mut receipts = Vec::new();
        let rd = match fs::read_dir(store.dir()) {
            Ok(r) => r,
            Err(_) => return Index::default(),
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let digest = match name.strip_suffix(".json") {
                Some(d) if is_sha256_hex(d) => d.to_string(),
                _ => continue,
            };
            let bytes = match fs::read(e.path()) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let body = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
            if sha256_hex(body) != digest {
                continue;
            }
            let value: Value = match serde_json::from_slice(body) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if value["schema"] != SCHEMA {
                continue;
            }
            let modified = e.metadata().and_then(|m| m.modified()).ok();
            receipts.push(Stored {
                digest,
                value,
                modified,
            });
        }
        receipts.sort_by(|a, b| {
            a.finished_utc()
                .cmp(b.finished_utc())
                .then_with(|| a.modified.cmp(&b.modified))
                .then_with(|| a.digest.cmp(&b.digest))
        });
        Index { receipts }
    }

    /// Every verified receipt, oldest first.
    pub fn receipts(&self) -> &[Stored] {
        &self.receipts
    }

    /// The receipt with exactly this digest, if the store has it.
    pub fn get(&self, digest: &str) -> Option<&Stored> {
        self.receipts.iter().find(|r| r.digest == digest)
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    /// The newest receipt of `gate`, for any commit.
    pub fn latest(&self, gate: &str) -> Option<&Stored> {
        self.receipts.iter().rev().find(|r| r.gate() == gate)
    }

    /// The newest receipt of `gate` made at exactly `commit`.
    pub fn latest_at(&self, gate: &str, commit: &str) -> Option<&Stored> {
        self.receipts
            .iter()
            .rev()
            .find(|r| r.gate() == gate && r.commit() == commit)
    }

    /// What is known about each of `deps` at `commit`, in the given order. A
    /// dependency without a receipt for this commit has no verdict.
    pub fn dep_facts(&self, deps: &[String], commit: &str) -> Vec<DepFact> {
        deps.iter()
            .map(|d| match self.latest_at(d, commit) {
                Some(r) => DepFact {
                    gate: d.clone(),
                    verdict: r.verdict(),
                    receipt: Some(r.digest.clone()),
                },
                None => DepFact {
                    gate: d.clone(),
                    verdict: None,
                    receipt: None,
                },
            })
            .collect()
    }
}

/// Format seconds since the Unix epoch as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_string(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's civil-from-days algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// The inverse of `utc_string`: seconds since the Unix epoch for a
/// `YYYY-MM-DDTHH:MM:SSZ` string; None for anything else.
pub fn utc_secs(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return None;
    }
    let num = |from: usize, to: usize| -> Option<i64> {
        let t = &s[from..to];
        if !t.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        t.parse::<i64>().ok()
    };
    let (y, m, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (hh, mm, ss) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    // Howard Hinnant's days-from-civil algorithm.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + hh * 3600 + mm * 60 + ss;
    u64::try_from(secs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_sorts_keys_and_is_compact() {
        let v = json!({"b": [1, true, null], "a": {"z": "x", "y": -3}});
        assert_eq!(
            String::from_utf8(canonical(&v).unwrap()).unwrap(),
            r#"{"a":{"y":-3,"z":"x"},"b":[1,true,null]}"#
        );
    }

    #[test]
    fn canonical_escapes_per_c4() {
        let v = json!("a\"b\\c\nd\r\te\u{1f}f\u{7f}\u{e9}");
        let s = String::from_utf8(canonical(&v).unwrap()).unwrap();
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\\r\\te\\u001ff\u{7f}\u{e9}\"");
    }

    #[test]
    fn canonical_rejects_floats() {
        assert!(canonical(&json!({"x": 1.5})).is_err());
        assert!(canonical(&json!([0.0])).is_err());
    }

    #[test]
    fn canonical_digest_is_stable_across_key_order() {
        let a = json!({"x": 1, "y": 2});
        let b = json!({"y": 2, "x": 1});
        assert_eq!(canonical_digest(&a).unwrap(), canonical_digest(&b).unwrap());
    }

    #[test]
    fn utc_formatting() {
        assert_eq!(utc_string(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_string(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(utc_string(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn receipt_roundtrip_mode_and_name() {
        let t = TempDir::new("ev1");
        let store = Store::new(t.path());
        let r = build_receipt(json!({"gate": "G"}), json!({"duration_ms": 5}));
        let (digest, path) = store.put_receipt(&r).unwrap();
        assert_eq!(path, t.path().join(format!("{digest}.json")));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(*bytes.last().unwrap(), b'\n');
        assert_eq!(sha256_hex(&bytes[..bytes.len() - 1]), digest);
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o444);
        let back: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back["schema"], SCHEMA);
    }

    #[test]
    fn identical_receipt_twice_is_success() {
        let t = TempDir::new("ev2");
        let store = Store::new(t.path());
        let r = build_receipt(json!({"gate": "G"}), json!({}));
        let a = store.put_receipt(&r).unwrap();
        let b = store.put_receipt(&r).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn receipt_overwrite_is_refused() {
        let t = TempDir::new("ev3");
        let store = Store::new(t.path());
        let r = build_receipt(json!({"gate": "G"}), json!({}));
        let (_, path) = store.put_receipt(&r).unwrap();
        // Tamper with the stored file, then try to write the same receipt again.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        fs::write(&path, b"tampered").unwrap();
        match store.put_receipt(&r) {
            Err(StoreError::Collision(_)) => {}
            other => panic!("expected collision, got {other:?}"),
        }
        assert_eq!(fs::read(&path).unwrap(), b"tampered");
    }

    #[test]
    fn blob_is_content_addressed_and_idempotent() {
        let t = TempDir::new("ev4");
        let store = Store::new(t.path());
        let sha = store.put_blob(b"hello\n").unwrap();
        assert_eq!(sha, sha256_hex(b"hello\n"));
        let p = t.path().join("blobs").join(format!("{sha}.log"));
        assert_eq!(fs::read(&p).unwrap(), b"hello\n");
        assert_eq!(
            fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o444
        );
        assert_eq!(store.put_blob(b"hello\n").unwrap(), sha);
    }

    #[test]
    fn blob_collision_detected() {
        let t = TempDir::new("ev5");
        let store = Store::new(t.path());
        let sha = store.put_blob(b"hello\n").unwrap();
        let p = t.path().join("blobs").join(format!("{sha}.log"));
        fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
        fs::write(&p, b"evil").unwrap();
        match store.put_blob(b"hello\n") {
            Err(StoreError::Collision(_)) => {}
            other => panic!("expected collision, got {other:?}"),
        }
    }

    fn receipt(gate: &str, commit: &str, verdict: &str, finished: &str) -> Value {
        build_receipt(
            json!({"gate": gate, "commit": commit, "derived_verdict": verdict, "reason": ""}),
            json!({"finished_utc": finished}),
        )
    }

    #[test]
    fn index_finds_the_newest_receipt_per_gate_and_commit() {
        let t = TempDir::new("ev6");
        let store = Store::new(t.path());
        let c1 = "1".repeat(40);
        let c2 = "2".repeat(40);
        store
            .put_receipt(&receipt("G", &c1, "FAIL", "2026-10-02T01:00:00Z"))
            .unwrap();
        let (newer, _) = store
            .put_receipt(&receipt("G", &c1, "PASS", "2026-10-02T02:00:00Z"))
            .unwrap();
        let (other_commit, _) = store
            .put_receipt(&receipt("G", &c2, "FAIL", "2026-10-02T03:00:00Z"))
            .unwrap();
        store
            .put_receipt(&receipt(
                "H",
                &c1,
                "BLOCKED_HARDWARE",
                "2026-10-02T01:30:00Z",
            ))
            .unwrap();
        let ix = Index::load(&store);
        assert_eq!(ix.len(), 4);
        assert!(!ix.is_empty());
        assert_eq!(ix.latest("G").unwrap().digest, other_commit);
        let at = ix.latest_at("G", &c1).unwrap();
        assert_eq!(at.digest, newer);
        assert_eq!(at.verdict(), Some(Verdict::Pass));
        assert_eq!((at.gate(), at.commit()), ("G", c1.as_str()));
        assert!(ix.latest_at("G", &"3".repeat(40)).is_none());
        assert!(ix.latest("nope").is_none());
        let want = vec!["H".to_string(), "G".to_string(), "Z".to_string()];
        let facts = ix.dep_facts(&want, &c1);
        assert_eq!(facts.len(), 3);
        assert_eq!(facts[0].verdict, Some(Verdict::BlockedHardware));
        assert_eq!(facts[1].verdict, Some(Verdict::Pass));
        assert_eq!(facts[1].receipt.as_deref(), Some(newer.as_str()));
        assert_eq!(
            facts[2],
            DepFact {
                gate: "Z".to_string(),
                verdict: None,
                receipt: None
            }
        );
    }

    #[test]
    fn index_skips_files_that_are_not_trustworthy_receipts() {
        let t = TempDir::new("ev7");
        let store = Store::new(t.path());
        let c = "1".repeat(40);
        store
            .put_receipt(&receipt("G", &c, "PASS", "2026-10-02T01:00:00Z"))
            .unwrap();
        // The file name does not match the bytes.
        let forged = canonical(&receipt("G", &c, "FAIL", "2026-10-02T09:00:00Z")).unwrap();
        fs::write(t.path().join(format!("{}.json", "0".repeat(64))), forged).unwrap();
        // The right name, but not a receipt of this schema.
        let alien = canonical(&json!({"schema": "something/else", "core": {"gate": "G"}})).unwrap();
        fs::write(
            t.path().join(format!("{}.json", sha256_hex(&alien))),
            &alien,
        )
        .unwrap();
        // Not JSON at all, with the right name.
        let junk = b"not json".to_vec();
        fs::write(t.path().join(format!("{}.json", sha256_hex(&junk))), &junk).unwrap();
        fs::write(t.path().join("notes.txt"), b"hello").unwrap();
        let ix = Index::load(&store);
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.latest("G").unwrap().verdict(), Some(Verdict::Pass));
        let missing = Store::new(&t.path().join("no-such-dir"));
        assert!(Index::load(&missing).is_empty());
    }

    #[test]
    fn identical_blobs_stored_at_the_same_moment_do_not_collide() {
        let t = TempDir::new("ev8");
        let store = Store::new(t.path());
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    barrier.wait();
                    for _ in 0..20 {
                        store.put_blob(b"same bytes").expect("concurrent put_blob");
                    }
                });
            }
        });
        let sha = sha256_hex(b"same bytes");
        let blob = t.path().join("blobs").join(format!("{sha}.log"));
        assert_eq!(fs::read(blob).unwrap(), b"same bytes");
        let leftovers = fs::read_dir(t.path().join("blobs"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .count();
        assert_eq!(leftovers, 0, "no temp file may be left behind");
    }
}

#[cfg(test)]
mod utc_tests {
    use super::*;

    #[test]
    fn utc_secs_inverts_utc_string() {
        for secs in [
            0u64,
            1,
            86_399,
            86_400,
            951_782_400,
            1_700_000_000,
            1_791_000_000,
            4_102_444_800,
        ] {
            assert_eq!(utc_secs(&utc_string(secs)), Some(secs), "{secs}");
        }
        assert_eq!(utc_secs("2026-10-11T00:00:00Z"), Some(1_791_676_800));
        for bad in [
            "",
            "2026-10-11",
            "2026-10-11T00:00:00",
            "2026-13-11T00:00:00Z",
            "x026-10-11T00:00:00Z",
            "2026-10-11T24:00:00Z",
        ] {
            assert_eq!(utc_secs(bad), None, "{bad}");
        }
    }
}
