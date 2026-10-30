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

//! Decides whether omega's Qwen3.8-27B decode pieces (`src/omega_gpu_q38_graph.h`, sc #642
//! on the omega side) are linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds `omega_gpu_q38_graph_create`
//!     defined in that archive.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout has `src/omega_gpu_q38_graph.h`.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_q38_prefill)` is set when, in addition, the prefill entries are linked
//! (`omega_gpu_gdn_prefill` and `omega_gpu_prefill_attn256`, omega#543 / #545; with
//! `AIEN_OMEGA_DIR`, the header `src/omega_gpu_prefill_attn256_api.h`).
//!
//! `cfg(has_omega_q38)` is set only when native. A stub never fails the build: the Qwen3.8
//! engine reports itself unavailable, and `is_native()` says which one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_q38)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_q38_prefill)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_q36)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_GPU_LIB",
        "AIEN_OMEGA_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        println!("cargo:rerun-if-changed={lib}");
        if archive_defines(Path::new(&lib), "omega_gpu_q38_graph_create") {
            println!("cargo:rustc-cfg=has_omega_q38");
            if archive_defines(Path::new(&lib), "omega_gpu_q36_graph_create") {
                println!("cargo:rustc-cfg=has_omega_q36");
            }
            if archive_defines(Path::new(&lib), "omega_gpu_gdn_prefill")
                && archive_defines(Path::new(&lib), "omega_gpu_prefill_attn256")
            {
                println!("cargo:rustc-cfg=has_omega_q38_prefill");
            }
        } else {
            println!("cargo:warning=aien-omega-q38: {lib} does not define the q38 graph API; the Qwen3.8 decode is stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_q38_graph.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_q38");
            if Path::new(&dir).join("src/omega_gpu_q36_graph.h").is_file() {
                println!("cargo:rustc-cfg=has_omega_q36");
            }
            if Path::new(&dir)
                .join("src/omega_gpu_prefill_attn256_api.h")
                .is_file()
            {
                println!("cargo:rustc-cfg=has_omega_q38_prefill");
            }
        } else {
            println!("cargo:warning=aien-omega-q38: {dir} has no src/omega_gpu_q38_graph.h (omega before sc #642); the Qwen3.8 decode is stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-q38: nm not available, assuming the archive lacks the q38 graph API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
