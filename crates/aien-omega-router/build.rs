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

//! Decides whether omega's one-launch MoE router (`src/omega_gpu_router_api.h`,
//! OWN-30B router cut on the omega side) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds `omega_gpu_router` defined
//!     in that archive.
//!   * `AIEN_CUDAPP_DIR=<checkout>` (CUDApp at `cudapp.lock`): native when the checkout has
//!     `src/omega_gpu_router_api.h`.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_router)` is set only when native. A stub never fails the build: the engine
//! keeps its host routing, and `is_native()` says which
//! one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_router)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_GPU_LIB",
        "AIEN_CUDAPP_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        if archive_defines(Path::new(&lib), "omega_gpu_router") {
            println!("cargo:rustc-cfg=has_omega_router");
        } else {
            println!("cargo:warning=aien-omega-router: {lib} does not define the device router API; the device router is stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_CUDAPP_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_router_api.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_router");
        } else {
            println!("cargo:warning=aien-omega-router: {dir} has no src/omega_gpu_router_api.h; the device router is stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-router: nm not available, assuming the archive lacks the device router API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
