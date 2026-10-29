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
//! Canonical capabilities are inspected from the admitted canonical archive exported by
//! `aien-omega-gpu` via Cargo DEP metadata.
//!
//! `cfg(has_omega_router)` is set only when native. A stub never fails the build: the engine
//! keeps its host routing, and `is_native()` says which
//! one a binary carries.
use aien_abi_conformance::CanonicalFeatures;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_router)");
    let Some(cf) = CanonicalFeatures::from_dep("aien-omega-router") else {
        return;
    };
    if cf.has_symbol("omega_gpu_router") {
        println!("cargo:rustc-cfg=has_omega_router");
    } else {
        println!(
            "cargo:warning=aien-omega-router: {} does not define the device router API; the device router is stubbed",
            cf.archive().display()
        );
    }
}
