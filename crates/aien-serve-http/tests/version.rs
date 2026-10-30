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

//! `--version` prints the build identity without loading a model or touching the GPU, so
//! install.sh and tools/serve-27b.sh can refuse a binary that linked only the CPU stubs.
//! Host only: the engine lines say `native` or `STUB` depending on how this test was built.

use std::process::Command;

#[test]
fn version_prints_build_identity_and_exits_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_aien-serve-http"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success(), "exit {:?}", out.status);
    let text = String::from_utf8(out.stdout).unwrap();
    for want in [
        "aien-serve-http ",
        "cudapp.lock: ",
        "omega gpu engine: ",
        "qwen38 engine: ",
        "default model dir: $HOME/models/qwen3.8-27b-fp8-017b9c7a",
    ] {
        assert!(text.contains(want), "missing {want:?} in:\n{text}");
    }
    let pin =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../cudapp.lock")).unwrap();
    assert!(
        text.contains(&format!("cudapp.lock: {}", pin.trim())),
        "cudapp.lock line does not match the repository pin:\n{text}"
    );
}

#[test]
fn version_wins_over_other_flags_and_loads_nothing() {
    // A missing model dir would fail any real start; --version must not look at it.
    let out = Command::new(env!("CARGO_BIN_EXE_aien-serve-http"))
        .args([
            "--engine",
            "omega",
            "--model-dir",
            "/nonexistent-version-test",
            "--version",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "exit {:?}", out.status);
    assert!(String::from_utf8_lossy(&out.stdout).contains("qwen38 engine: "));
}
