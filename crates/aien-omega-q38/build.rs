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

//! Native capabilities come only from the archive-owning aien-omega-gpu build.
fn main() {
    const KEY: &str = "DEP_OMEGA_GPU_CAPABILITIES";
    println!("cargo:rerun-if-env-changed={KEY}");
    let capabilities = std::env::var(KEY).expect("GPU owner must publish capabilities");
    for cfg in ["has_omega_q38", "has_omega_q38_prefill", "has_omega_q36"] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        if capabilities.split_whitespace().any(|c| c == cfg) {
            println!("cargo:rustc-cfg={cfg}");
        }
    }
}
