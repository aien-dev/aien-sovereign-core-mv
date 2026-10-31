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

//! Build identity of the omega door for the persistent KV engine identity (sc#505, ADR 0003
//! section 1.2.2). Embedded at build time, read by `kv_persist::OmegaBuild::linked()`:
//!
//! - `AIEN_SERVE_HTTP_GIT_COMMIT`: the commit this crate was built from (`AIEN_BUILD_COMMIT`
//!   when set, else `git rev-parse HEAD`, else `unknown`).
//! - `AIEN_SERVE_HTTP_GIT_DIRTY`: `1` when `git status` showed modified tracked files at build
//!   time, `0` when clean (or `AIEN_BUILD_COMMIT` given), `unknown` without git. Re-checked
//!   when HEAD or the index moves, not on every unstaged edit.
//! - `AIEN_SERVE_HTTP_OMEGA_LOCK`: the omega commit `omega.lock` pins.
//! - `AIEN_SERVE_HTTP_GPU_RECORDED_SHA256`: the `libomega_gpu.a` sha256 `native-artifacts.lock`
//!   records for that commit and this target, or `unrecorded`.
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..");
    let target = std::env::var("TARGET").unwrap_or_default();

    println!("cargo:rerun-if-env-changed=AIEN_BUILD_COMMIT");
    let forced = std::env::var("AIEN_BUILD_COMMIT")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let (commit, dirty) = match forced {
        Some(c) => (c.trim().to_string(), "0".to_string()),
        None => {
            for p in ["HEAD", "index"] {
                if let Some(path) = git(&root, &["rev-parse", "--git-path", p]) {
                    let path = if Path::new(&path).is_absolute() {
                        PathBuf::from(path)
                    } else {
                        root.join(path)
                    };
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
            if let Some(name) = git(&root, &["symbolic-ref", "-q", "HEAD"]) {
                if let Some(path) = git(&root, &["rev-parse", "--git-path", &name]) {
                    let path = if Path::new(&path).is_absolute() {
                        PathBuf::from(path)
                    } else {
                        root.join(path)
                    };
                    if path.exists() {
                        println!("cargo:rerun-if-changed={}", path.display());
                    }
                }
            }
            let commit = git(&root, &["rev-parse", "HEAD"])
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".into());
            let dirty = match git(&root, &["status", "--porcelain", "--untracked-files=no"]) {
                Some(s) if s.is_empty() => "0",
                Some(_) => "1",
                None => "unknown",
            };
            (commit, dirty.to_string())
        }
    };
    println!("cargo:rustc-env=AIEN_SERVE_HTTP_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=AIEN_SERVE_HTTP_GIT_DIRTY={dirty}");

    let lock_path = root.join("omega.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let omega = std::fs::read_to_string(&lock_path)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let omega = if omega.is_empty() {
        "unknown".to_string()
    } else {
        omega
    };
    println!("cargo:rustc-env=AIEN_SERVE_HTTP_OMEGA_LOCK={omega}");

    let artifacts = root.join("native-artifacts.lock");
    println!("cargo:rerun-if-changed={}", artifacts.display());
    let recorded = std::fs::read_to_string(&artifacts)
        .ok()
        .and_then(|text| {
            text.lines()
                .filter(|l| !l.trim_start().starts_with('#'))
                .find_map(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    (f.len() >= 4 && f[0] == "libomega_gpu.a" && f[1] == omega && f[2] == target)
                        .then(|| f[3].to_string())
                })
        })
        .unwrap_or_else(|| "unrecorded".to_string());
    println!("cargo:rustc-env=AIEN_SERVE_HTTP_GPU_RECORDED_SHA256={recorded}");
}
