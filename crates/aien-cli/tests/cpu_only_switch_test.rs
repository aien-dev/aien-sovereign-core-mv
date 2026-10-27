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

//! sc#384 preflight: `AIEN_BACKEND=cpu` on a real `aien daemon` process.
//! Whatever the build, the daemon never reports the GB10 backend:
//!   - GPU-capable build: refuses to start with the named error `CPU_ONLY_BUT_GPU_BUILD`
//!     (before any model load or GPU session);
//!   - CPU-stub build: the switch is accepted (the CPU backend is the only one possible);
//!     here the start is stopped one step later by the approval-desk requirement, with no
//!     desk key, so no model is needed and the test never touches the chip.
use std::process::{Command, Stdio};

fn run(extra: &[(&str, &str)]) -> (bool, String) {
    let tmp = tempfile::Builder::new()
        .prefix("co-")
        .tempdir_in("/tmp")
        .unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    for d in ["state", "home", "prov"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    let mut c = Command::new(env!("CARGO_BIN_EXE_aien-cli"));
    c.current_dir(&root)
        .env("HOME", root.join("home"))
        .env("NO_COLOR", "1")
        .env("AIEN_RUNTIME_SOCK", root.join("s"))
        .env("AIEN_RUNTIME_STATE_DIR", root.join("state"))
        .env("AIEN_COMPOSE_DIR", root.join("compose"))
        .env("AIEN_PROVENANCE_DIR", root.join("prov"))
        .env("AIEN_REQUIRE_CHECKPOINT", "0");
    for k in [
        "AIEN_BACKEND",
        "AIEN_REQUIRE_BLACKWELL",
        "AIEN_GPU_BACKEND",
        "AIEN_MODEL_PATH",
        "AIEN_TOKENIZER_PATH",
        "AIEN_MODEL_DIR",
        "AIEN_DEV_FALLBACK",
        "AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK",
    ] {
        c.env_remove(k);
    }
    for (k, v) in extra {
        c.env(k, v);
    }
    let out = c
        .arg("daemon")
        .stdin(Stdio::null())
        .output()
        .expect("run daemon");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn cpu_only_daemon_never_reports_gb10() {
    let (ok, text) = run(&[("AIEN_BACKEND", "cpu")]);
    assert!(
        !ok,
        "the start must be refused (named error or desk): {text}"
    );
    assert!(
        !text.contains("OmegaGb10Backend") && !text.contains("sm_121"),
        "a CPU-only run reported the GB10 backend: {text}"
    );
    if aien_omega_gpu::is_native() {
        assert!(text.contains("CPU_ONLY_BUT_GPU_BUILD"), "{text}");
    } else {
        assert!(!text.contains("CPU_ONLY_BUT_GPU_BUILD"), "{text}");
    }
}

#[test]
fn force_cpu_stub_alone_is_not_the_runtime_switch() {
    let (_ok, text) = run(&[("AIEN_FORCE_CPU_STUB", "1")]);
    assert!(!text.contains("CPU_ONLY_BUT_GPU_BUILD"), "{text}");
}
