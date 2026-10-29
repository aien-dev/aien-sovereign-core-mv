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

//! Native archive identity (used from `build.rs`).
//!
//! `native-artifacts.lock` (repository root) records the expected sha256 of each archive built
//! from its pinned component source, per target triple and C compiler, one line each:
//!
//! ```text
//! <archive> <source-sha> <target> <sha256> <cc --version, first line>
//! ```
//!
//! omega's archives are byte-reproducible for a given source, compiler and target (the build
//! writes no paths or times into them), so an entry is a real expectation: a built archive that
//! differs from its entry fails the build. A combination with no entry builds, and is reported
//! as `digest unrecorded`, which the qualification entry point does not accept as PASS.
//! A prebuilt archive (`*_LIB`) must come with its expected digest (`*_LIB_SHA256`).

use std::path::Path;
use std::process::Command;

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("cannot run sha256sum on {}: {e}", path.display()))?;
    if !out.status.success() {
        return Err(format!("sha256sum failed on {}", path.display()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let d = text.split_whitespace().next().unwrap_or("").to_string();
    if is_sha256(&d) {
        Ok(d)
    } else {
        Err(format!(
            "sha256sum printed no digest for {}",
            path.display()
        ))
    }
}

pub fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// First line of `<cc> --version` (the compiler identity recorded with each digest).
pub fn cc_version(cc: &str) -> String {
    Command::new(cc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// One `native-artifacts.lock` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockEntry {
    pub archive: String,
    pub source_sha: String,
    pub target: String,
    pub sha256: String,
    pub cc: String,
}

pub fn parse_lock(text: &str) -> Result<Vec<LockEntry>, String> {
    let mut v = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let w: Vec<&str> = line.splitn(5, ' ').collect();
        if w.len() != 5 || !is_sha256(w[3]) || w[1].len() != 40 {
            return Err(format!(
                "native-artifacts.lock line {}: want `<archive> <source-sha> <target> <sha256> <cc version>`: {raw}",
                i + 1
            ));
        }
        v.push(LockEntry {
            archive: w[0].into(),
            source_sha: w[1].into(),
            target: w[2].into(),
            sha256: w[3].into(),
            cc: w[4].trim().into(),
        });
    }
    Ok(v)
}

/// The recorded digest for this archive, source commit, target and compiler, if any.
pub fn expected<'a>(
    lock: &'a [LockEntry],
    archive: &str,
    source_sha: &str,
    target: &str,
    cc: &str,
) -> Option<&'a LockEntry> {
    lock.iter()
        .find(|e| e.archive == archive && e.source_sha == source_sha && e.target == target && e.cc == cc)
}

/// Outcome of the digest check for an archive built from source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Digest {
    /// matches its `native-artifacts.lock` entry
    Locked,
    /// no entry for this (archive, source commit, target, compiler)
    Unrecorded,
}

/// Compare a built archive's digest with the lock. `Err` = a recorded digest that differs.
pub fn check_built(
    lock: &[LockEntry],
    archive: &str,
    source_sha: &str,
    target: &str,
    cc: &str,
    actual: &str,
) -> Result<Digest, String> {
    match expected(lock, archive, source_sha, target, cc) {
        Some(e) if e.sha256 == actual => Ok(Digest::Locked),
        Some(e) => Err(format!(
            "{archive} built from source {source_sha} for {target} with `{cc}` has sha256 {actual}, native-artifacts.lock expects {}",
            e.sha256
        )),
        None => Ok(Digest::Unrecorded),
    }
}

/// A prebuilt archive must name its expected digest, and match it.
pub fn check_prebuilt(path: &Path, expected: Option<&str>, var: &str) -> Result<String, String> {
    let Some(want) = expected else {
        return Err(format!(
            "{} is a prebuilt archive with no expected digest: set {var}=<sha256> (sha256sum {})",
            path.display(),
            path.display()
        ));
    };
    let want = want.trim().to_ascii_lowercase();
    if !is_sha256(&want) {
        return Err(format!("{var} must be 64 lowercase hex characters"));
    }
    let got = sha256_file(path)?;
    if got != want {
        return Err(format!(
            "{} has sha256 {got}, {var} expects {want} (wrong, rebuilt, truncated or corrupted archive)",
            path.display()
        ));
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "3eb20f5ec60b465413320000000000000000000000000000000000000000abcd";

    #[test]
    fn lock_lookup_and_mismatch() {
        let o = "6c6180cf378075b61291f4565d226eba38b4decd";
        let text = format!("# c\nlibx.a {o} aarch64-unknown-linux-gnu {A} cc (X) 1.2\n");
        let l = parse_lock(&text).unwrap();
        let t = "aarch64-unknown-linux-gnu";
        assert_eq!(
            check_built(&l, "libx.a", o, t, "cc (X) 1.2", A),
            Ok(Digest::Locked)
        );
        assert_eq!(
            check_built(&l, "libx.a", o, t, "cc (X) 1.3", A),
            Ok(Digest::Unrecorded)
        );
        let bad = A.replace('a', "b");
        assert!(check_built(&l, "libx.a", o, t, "cc (X) 1.2", &bad).is_err());
        assert!(parse_lock("libx.a short aarch64 abc cc\n").is_err());
    }

    #[test]
    fn canonical_and_companion_records_are_independent_source_keys() {
        let cudapp = "1111111111111111111111111111111111111111";
        let omega = "2222222222222222222222222222222222222222";
        let companion_digest = A.replace('a', "b");
        let target = "aarch64-unknown-linux-gnu";
        let cc = "cc fixture";
        let lock = parse_lock(&format!(
            "libcudapp.a {cudapp} {target} {A} {cc}\nlibomega_gpu.a {omega} {target} {companion_digest} {cc}\n"
        )).unwrap();
        assert_eq!(check_built(&lock, "libcudapp.a", cudapp, target, cc, A), Ok(Digest::Locked));
        assert_eq!(check_built(&lock, "libomega_gpu.a", omega, target, cc, &companion_digest), Ok(Digest::Locked));
        assert_eq!(check_built(&lock, "libcudapp.a", omega, target, cc, A), Ok(Digest::Unrecorded));
        assert_eq!(check_built(&lock, "libomega_gpu.a", cudapp, target, cc, &companion_digest), Ok(Digest::Unrecorded));
        assert_eq!(check_built(&lock, "libcudapp.a", cudapp, "other-target", cc, A), Ok(Digest::Unrecorded));
        assert_eq!(check_built(&lock, "libcudapp.a", cudapp, target, "other compiler", A), Ok(Digest::Unrecorded));
        assert!(check_built(&lock, "libcudapp.a", cudapp, target, cc, &companion_digest).is_err());
        assert!(check_built(&lock, "libomega_gpu.a", omega, target, cc, A).is_err());
    }

    #[test]
    fn prebuilt_needs_and_checks_a_digest() {
        let dir = std::env::temp_dir().join(format!("abi-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("lib.a");
        std::fs::write(&f, b"!<arch>\n").unwrap();
        let real = sha256_file(&f).unwrap();
        assert!(check_prebuilt(&f, None, "X_SHA256").is_err());
        assert_eq!(check_prebuilt(&f, Some(&real), "X_SHA256").unwrap(), real);
        // truncated / corrupted: same name, different bytes
        std::fs::write(&f, b"!<arch").unwrap();
        assert!(check_prebuilt(&f, Some(&real), "X_SHA256").is_err());
        assert!(check_prebuilt(&f, Some("nothex"), "X_SHA256").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
