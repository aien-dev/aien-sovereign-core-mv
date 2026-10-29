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
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//!
//! `cfg(has_omega_q38)` is set only when native. A stub never fails the build: the Qwen3.8
//! engine reports itself unavailable, and `is_native()` says which one a binary carries.
//! `cfg(has_omega_q36)` is set when Qwen3.6-35B-A3B decode graph entries are defined.
//! `cfg(has_omega_q38_prefill)` is set when, in addition, the prefill entries are linked
//! (`omega_gpu_gdn_prefill` and `omega_gpu_prefill_attn256`).
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_q38)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_q38_prefill)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_q36)");
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-q38") else {
        return;
    };
    if cf.has_symbol("omega_gpu_q38_graph_create") {
        println!("cargo:rustc-cfg=has_omega_q38");
        if cf.has_symbol("omega_gpu_q36_graph_create") {
            println!("cargo:rustc-cfg=has_omega_q36");
        }
        if cf.has_symbol("omega_gpu_gdn_prefill")
            && cf.has_symbol("omega_gpu_prefill_attn256")
        {
            println!("cargo:rustc-cfg=has_omega_q38_prefill");
        }
    } else {
        println!(
            "cargo:warning=aien-omega-q38: {} does not define the q38 graph API; the Qwen3.8 decode is stubbed",
            cf.archive().display()
        );
    }
}
