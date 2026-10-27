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

//! Shared `build.rs` steps for the crates that link an omega archive: the C-side ABI check,
//! the archive identity check, and the `cargo:rustc-env` identity record the crate exposes at
//! run time (`native_identity()`), so a binary can say what it linked.

use crate::identity::{self, Digest};
use crate::probe::{check_c, defined_symbols, CCheck};
use crate::Manifest;
use std::path::{Path, PathBuf};

/// How the archive was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// no native library (every native call returns Unavailable)
    Stub,
    /// built from a source checkout (omega or CUDApp) at the commit its lock file pins
    PinnedSource,
    /// built from a source checkout at a developer-override commit (not qualifiable)
    DevOverride,
    /// a prebuilt archive given by path and expected digest (not qualifiable)
    Prebuilt,
}

impl Link {
    pub fn as_str(self) -> &'static str {
        match self {
            Link::Stub => "stub",
            Link::PinnedSource => "pinned-source",
            Link::DevOverride => "dev-override",
            Link::Prebuilt => "prebuilt",
        }
    }
}

/// What `build.rs` established; [`Identity::emit`] records it for the crate.
#[derive(Debug, Clone)]
pub struct Identity {
    pub link: Link,
    /// "pass" (headers + probe + symbols), "symbols-only" (prebuilt: no headers), "none" (stub)
    pub abi: &'static str,
    /// "locked" | "unrecorded" | "prebuilt-match" | "none"
    pub digest: &'static str,
    pub archive_sha256: String,
    pub omega_sha: String,
    pub aienos_lock: String,
    pub physics_lock: String,
    pub target: String,
    pub cc: String,
}

impl Identity {
    pub fn stub(omega_sha: &str) -> Identity {
        Identity {
            link: Link::Stub,
            abi: "none",
            digest: "none",
            archive_sha256: String::new(),
            omega_sha: omega_sha.into(),
            aienos_lock: String::new(),
            physics_lock: String::new(),
            target: env_or("TARGET", ""),
            cc: String::new(),
        }
    }

    /// `cargo:rustc-env={prefix}_{KEY}=value` for every field.
    pub fn emit(&self, prefix: &str) {
        let kv = [
            ("LINK", self.link.as_str().to_string()),
            ("ABI", self.abi.to_string()),
            ("DIGEST", self.digest.to_string()),
            ("ARCHIVE_SHA256", self.archive_sha256.clone()),
            ("SOURCE_SHA", self.omega_sha.clone()),
            ("AIENOS_LOCK", self.aienos_lock.clone()),
            ("PHYSICS_LOCK", self.physics_lock.clone()),
            ("TARGET", self.target.clone()),
            ("CC", self.cc.replace(['\n', '\r'], " ")),
        ];
        for (k, v) in kv {
            println!("cargo:rustc-env={prefix}_{k}={v}");
        }
    }
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Makes `out` a build directory for the omega commit `pinned`: when the directory was last
/// used for another commit (or carries no stamp) it is removed first, so `make` never reuses
/// objects compiled from the other commit's sources. The omega Makefile tracks header
/// dependencies per object, but a reused directory also survives Makefile fixes and rule gaps
/// (2026-10-12: `librx_compose.a` mixed objects of two layouts of a Cortex struct after a
/// repin, and the host seal refused with RX_ERR_AUTHORITY). Returns `true` when it wiped.
pub fn fresh_out_dir(out: &Path, pinned: &str) -> bool {
    let stamp = out.join(".omega-commit");
    let same = std::fs::read_to_string(&stamp)
        .map(|s| s.trim() == pinned)
        .unwrap_or(false);
    let wiped = !same && out.exists();
    if !same {
        if wiped {
            std::fs::remove_dir_all(out)
                .unwrap_or_else(|e| panic!("remove stale build dir {}: {e}", out.display()));
        }
        std::fs::create_dir_all(out)
            .unwrap_or_else(|e| panic!("create build dir {}: {e}", out.display()));
        std::fs::write(&stamp, format!("{pinned}\n"))
            .unwrap_or_else(|e| panic!("write {}: {e}", stamp.display()));
    }
    wiped
}

/// `true` when `AIEN_REQUIRE_NATIVE_LIBS=1` (release and qualification builds).
pub fn require_native() -> bool {
    std::env::var("AIEN_REQUIRE_NATIVE_LIBS").as_deref() == Ok("1")
}

/// Refuse `AIEN_FORCE_CPU_STUB=1` together with `AIEN_REQUIRE_NATIVE_LIBS=1`: a build that
/// requires the native library must never quietly become a stub.
pub fn refuse_stub_if_required(krate: &str) {
    if require_native() {
        panic!(
            "{krate}: AIEN_FORCE_CPU_STUB=1 and AIEN_REQUIRE_NATIVE_LIBS=1 contradict each other: \
             a required-native build cannot be a stub (unset one of them)"
        );
    }
}

fn read_trim(p: &Path) -> String {
    std::fs::read_to_string(p)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn load_manifest(path: &Path) -> Manifest {
    println!("cargo:rerun-if-changed={}", path.display());
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read ABI manifest {}: {e}", path.display()));
    Manifest::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn load_lock(path: &Path) -> Vec<identity::LockEntry> {
    println!("cargo:rerun-if-changed={}", path.display());
    match std::fs::read_to_string(path) {
        Ok(t) => identity::parse_lock(&t).unwrap_or_else(|e| panic!("{e}")),
        Err(_) => Vec::new(),
    }
}

/// Checks for an archive built from an omega checkout. Panics (fails the build) on any ABI
/// mismatch, missing symbol or digest that differs from `native-artifacts.lock`.
#[allow(clippy::too_many_arguments)]
pub fn qualify_built(
    krate: &str,
    manifest: &Path,
    lock: &Path,
    omega: &Path,
    physics: &Path,
    archive: &Path,
    omega_sha: &str,
    pinned: bool,
) -> Identity {
    let m = load_manifest(manifest);
    for h in &m.headers {
        println!("cargo:rerun-if-changed={}", omega.join(h).display());
    }
    let cc = env_or("CC", "cc");
    let target = env_or("TARGET", "");
    let host = env_or("HOST", "");
    let abi = if target == host {
        let work = PathBuf::from(env_or("OUT_DIR", ".")).join("abi-probe");
        let includes = vec![
            omega.join("src"),
            omega.join("src/runtime"),
            physics.join("m16"),
            physics.join("nvrm"),
        ];
        match check_c(&CCheck {
            manifest: &m,
            root: omega,
            includes,
            archive: Some(archive),
            work: &work,
            cc: cc.clone(),
        }) {
            Ok(report) => {
                let _ = std::fs::write(work.join("report.txt"), &report);
                "pass"
            }
            Err(errs) => panic!(
                "{krate}: omega {omega_sha} does not match {} ({} problem(s)); nothing was linked:\n  - {}",
                manifest.display(),
                errs.len(),
                errs.join("\n  - ")
            ),
        }
    } else if require_native() {
        panic!("{krate}: cross build ({host} -> {target}): the ABI layout probe cannot run, and AIEN_REQUIRE_NATIVE_LIBS=1 needs it");
    } else {
        println!("cargo:warning={krate}: cross build ({host} -> {target}): C ABI layout probe skipped (not qualifiable)");
        "unchecked"
    };
    let sha = identity::sha256_file(archive).unwrap_or_else(|e| panic!("{krate}: {e}"));
    let ccv = identity::cc_version(&cc);
    let name = archive
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let digest = match identity::check_built(
        &load_lock(lock),
        &name,
        omega_sha,
        &target,
        &ccv,
        &sha,
    ) {
        Ok(Digest::Locked) => "locked",
        Ok(Digest::Unrecorded) => {
            println!(
                "cargo:warning={krate}: {name} sha256 {sha} (omega {omega_sha}, {target}, `{ccv}`) is not recorded in native-artifacts.lock: builds, but does not qualify"
            );
            "unrecorded"
        }
        Err(e) => panic!("{krate}: {e}"),
    };
    Identity {
        link: if pinned {
            Link::PinnedSource
        } else {
            Link::DevOverride
        },
        abi,
        digest,
        archive_sha256: sha,
        omega_sha: omega_sha.into(),
        aienos_lock: read_trim(&omega.join("aienos.lock")),
        physics_lock: read_trim(&omega.join("physics.lock")),
        target,
        cc: ccv,
    }
}

/// Checks for a prebuilt archive: its digest must equal `$sha_var`, and it must define every
/// function the manifest names (no headers are available, so layouts are not re-probed; the
/// identity says so and qualification does not accept it).
pub fn qualify_prebuilt(
    krate: &str,
    manifest: &Path,
    archive: &Path,
    sha_var: &str,
    omega_sha: &str,
) -> Identity {
    println!("cargo:rerun-if-env-changed={sha_var}");
    let want = std::env::var(sha_var).ok();
    let sha = identity::check_prebuilt(archive, want.as_deref(), sha_var)
        .unwrap_or_else(|e| panic!("{krate}: {e}"));
    let m = load_manifest(manifest);
    let syms = defined_symbols(archive).unwrap_or_else(|e| panic!("{krate}: {e}"));
    let missing: Vec<&str> = m
        .fns
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| !syms.iter().any(|s| s == n))
        .collect();
    assert!(
        missing.is_empty(),
        "{krate}: prebuilt {} does not define {missing:?}",
        archive.display()
    );
    println!(
        "cargo:warning={krate}: linking prebuilt {} (sha256 {sha}): a development override, not a qualified build",
        archive.display()
    );
    Identity {
        link: Link::Prebuilt,
        abi: "symbols-only",
        digest: "prebuilt-match",
        archive_sha256: sha,
        omega_sha: omega_sha.into(),
        aienos_lock: String::new(),
        physics_lock: String::new(),
        target: env_or("TARGET", ""),
        cc: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::fresh_out_dir;

    #[test]
    fn fresh_out_dir_wipes_only_on_another_commit() {
        let out = std::env::temp_dir().join(format!("aien-fresh-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        assert!(!fresh_out_dir(&out, "aaaa"), "nothing to wipe on first use");
        std::fs::write(out.join("obj.o"), b"x").unwrap();
        assert!(
            !fresh_out_dir(&out, "aaaa"),
            "same commit keeps the directory"
        );
        assert!(out.join("obj.o").exists());
        assert!(fresh_out_dir(&out, "bbbb"), "another commit wipes it");
        assert!(!out.join("obj.o").exists());
        assert_eq!(
            std::fs::read_to_string(out.join(".omega-commit"))
                .unwrap()
                .trim(),
            "bbbb"
        );
        std::fs::remove_file(out.join(".omega-commit")).unwrap();
        std::fs::write(out.join("obj.o"), b"x").unwrap();
        assert!(
            fresh_out_dir(&out, "bbbb"),
            "no stamp: the directory is of unknown origin, wiped"
        );
        assert!(!out.join("obj.o").exists());
        let _ = std::fs::remove_dir_all(&out);
    }
}
