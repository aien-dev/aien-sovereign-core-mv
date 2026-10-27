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
//! librx_compose.a, so tests that need the Cortex journal are reported IGNORED
//! in a stub build instead of passing vacuously.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(compose_linked)");
    println!("cargo:rerun-if-env-changed=DEP_RX_COMPOSE_LINKED");
    if std::env::var("DEP_RX_COMPOSE_LINKED").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=compose_linked");
    }
}
