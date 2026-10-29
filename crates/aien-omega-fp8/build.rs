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
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//! `cfg(has_omega_fp8_grouped_rule)` (L310 one-wave lever, `omega_gpu_fp8_grouped_ctas_auto`) is set
//! when that admitted archive defines the symbol.
//!
//! `cfg(has_omega_fp8)` is set only when native. A stub never fails the build: the bf16
//! path in the model code stays usable, and `is_native()` says which one a binary carries.
//! `cfg(has_omega_fp8_dev)` is set when the device-pointer entry is present.
//! `cfg(has_omega_fp8gemm)` (the prefill grouped GEMM) is set when the archive defines `omega_gpu_fp8gemm_f32`.
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_dev)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8gemm)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_fp8_grouped_rule)");
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-fp8") else {
        return;
    };
    if cf.has_symbol("omega_gpu_fp8_gemv_grouped_f32") {
        println!("cargo:rustc-cfg=has_omega_fp8");
        if cf.has_symbol("omega_gpu_fp8_gemv_grouped_dev")
            && cf.has_symbol("omega_gpu_fp8_tensor_entry")
        {
            println!("cargo:rustc-cfg=has_omega_fp8_dev");
        }
        if cf.has_symbol("omega_gpu_fp8gemm_f32") {
            println!("cargo:rustc-cfg=has_omega_fp8gemm");
        }
        if cf.has_symbol("omega_gpu_fp8_grouped_ctas_auto") {
            println!("cargo:rustc-cfg=has_omega_fp8_grouped_rule");
        }
    } else {
        println!(
            "cargo:warning=aien-omega-fp8: {} does not define the FP8 GEMV API; FP8 calls are stubbed",
            cf.archive().display()
        );
    }
}
