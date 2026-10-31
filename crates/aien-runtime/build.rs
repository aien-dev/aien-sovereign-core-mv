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

//! `cfg(compose_linked)` when aien-omega-compose linked the real
//! librx_compose.a (its build script's `links` metadata), so tests that need
//! the Cortex journal are reported IGNORED in a stub build instead of passing
//! vacuously.
//!
//! `AIEN_RUNTIME_GIT_COMMIT` (sc#504, ADR 0003 section 1.2.2): the commit this
//! crate was built from, part of the host engine build id that the persistent
//! KV namespace binds. Taken from the `AIEN_BUILD_COMMIT` environment variable
//! when set (builds without a git checkout), else from `git rev-parse HEAD`,
//! else `unknown`.
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(compose_linked)");
    println!("cargo:rerun-if-env-changed=DEP_RX_COMPOSE_LINKED");
    if std::env::var("DEP_RX_COMPOSE_LINKED").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=compose_linked");
    }

    println!("cargo:rerun-if-env-changed=AIEN_BUILD_COMMIT");
    let commit = std::env::var("AIEN_BUILD_COMMIT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            // Rebuild when HEAD moves: the HEAD file and the branch ref it names.
            if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
                println!("cargo:rerun-if-changed={head}");
            }
            if let Some(name) = git(&["symbolic-ref", "-q", "HEAD"]) {
                if let Some(path) = git(&["rev-parse", "--git-path", &name]) {
                    if std::path::Path::new(&path).exists() {
                        println!("cargo:rerun-if-changed={path}");
                    }
                }
            }
            git(&["rev-parse", "HEAD"])
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=AIEN_RUNTIME_GIT_COMMIT={commit}");
}
