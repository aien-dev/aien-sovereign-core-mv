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

//! Decides whether omega's FP8 GEMV API (`src/omega_gpu_fp8_api.h`, OWN-30B cuts 5-6) is linked.
//!
//! The archive itself is linked by `aien-omega-gpu`'s build script (this crate only declares
//! the extra symbols), so the decision mirrors its inputs:
//!   * `AIEN_FORCE_CPU_STUB=1`: stub.
//!   * `AIEN_OMEGA_GPU_LIB=<archive>`: native when `nm` finds
//!     `omega_gpu_fp8_gemv_grouped_f32` defined in that archive.
//!   * `AIEN_OMEGA_DIR=<checkout>`: native when the checkout has `src/omega_gpu_fp8_api.h`.
//!
//! `cfg(has_omega_fp8gemm)` (the prefill grouped GEMM) is set when the archive defines
//! `omega_gpu_fp8gemm_f32` or the checkout has `src/omega_gpu_fp8gemm_api.h`.
//!   * otherwise: stub (every call returns `Unavailable`).
//!
//! `cfg(has_omega_fp8)` is set only when native. A stub never fails the build: the bf16
//! path in the model code stays usable, and `is_native()` says which one a binary carries.
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_dev)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8gemm)");
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
        if archive_defines(Path::new(&lib), "omega_gpu_fp8_gemv_grouped_f32") {
            println!("cargo:rustc-cfg=has_omega_fp8");
            if archive_defines(Path::new(&lib), "omega_gpu_fp8_gemv_grouped_dev")
                && archive_defines(Path::new(&lib), "omega_gpu_fp8_tensor_entry")
            {
                println!("cargo:rustc-cfg=has_omega_fp8_dev");
            }
            if archive_defines(Path::new(&lib), "omega_gpu_fp8gemm_f32") {
                println!("cargo:rustc-cfg=has_omega_fp8gemm");
            }
        } else {
            println!("cargo:warning=aien-omega-fp8: {lib} does not define the FP8 GEMV API; FP8 calls are stubbed");
        }
        return;
    }
    if let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") {
        let header = Path::new(&dir).join("src/omega_gpu_fp8_api.h");
        println!("cargo:rerun-if-changed={}", header.display());
        if header.is_file() {
            println!("cargo:rustc-cfg=has_omega_fp8");
            if std::fs::read_to_string(&header).is_ok_and(|t| {
                t.contains("omega_gpu_fp8_gemv_grouped_dev")
                    && t.contains("omega_gpu_fp8_tensor_entry")
            }) {
                println!("cargo:rustc-cfg=has_omega_fp8_dev");
            }
            if Path::new(&dir)
                .join("src/omega_gpu_fp8gemm_api.h")
                .is_file()
            {
                println!("cargo:rustc-cfg=has_omega_fp8gemm");
            }
        } else {
            println!("cargo:warning=aien-omega-fp8: {dir} has no src/omega_gpu_fp8_api.h (omega before OWN-30B cut 5); FP8 calls are stubbed");
        }
    }
}

fn archive_defines(lib: &Path, symbol: &str) -> bool {
    let Ok(out) = Command::new("nm").arg(lib).output() else {
        println!("cargo:warning=aien-omega-fp8: nm not available, assuming the archive lacks the FP8 GEMV API");
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.ends_with(&format!(" T {symbol}")))
}
