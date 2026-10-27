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

//! Links `libomega_gpu.a` from the omega commit pinned in `omega.lock`.
//!
//! Offline by design: nothing is downloaded and no CUDA/nvcc is used. Order:
//!   1. `AIEN_FORCE_CPU_STUB=1`      -> stub (CI without the chip). BUILD-TIME ONLY: a binary already
//!      built with the GPU library ignores it at run time. For a CPU-only RUN set `AIEN_BACKEND=cpu`:
//!      the daemon refuses to start (`CPU_ONLY_BUT_GPU_BUILD`) if the binary would select the GB10 backend.
//!      Refused when `AIEN_REQUIRE_NATIVE_LIBS=1` (a required-native build is never a stub).
//!   2. `AIEN_OMEGA_GPU_LIB=<file>`  -> link that prebuilt archive; `AIEN_OMEGA_GPU_LIB_SHA256` must
//!      name its sha256 (checked) and it must define every function in the ABI manifest. A
//!      development override: its identity says `prebuilt` and it never qualifies.
//!   3. `AIEN_OMEGA_DIR=<checkout>`  -> its HEAD must equal omega.lock; built with
//!      `make libomega_gpu` into OUT_DIR. `AIEN_PHYSICS_DIR` (default `<omega>/../physics`)
//!      is passed as PHYSICS_DIR; omega's own physics.lock check still runs.
//!   4. otherwise                    -> stub, with a build warning.
//!
//! `cfg(has_omega_gpu)` is set only when the real library is linked.
//!
//! Before linking a checkout build, `aien_abi_conformance::build::qualify_built` checks the
//! pinned headers, a compiled layout probe and the archive symbols against
//! `abi/omega_gpu.manifest`, and the archive sha256 against `native-artifacts.lock`; any mismatch
//! fails the build. The result is recorded for `native_identity()`.
use aien_abi_conformance::build;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREFIX: &str = "AIEN_OMEGA_GPU";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_gpu)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_GPU_LIB",
        "AIEN_OMEGA_DIR",
        "AIEN_PHYSICS_DIR",
        "AIEN_REQUIRE_NATIVE_LIBS",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let lock_path = Path::new(&env("CARGO_MANIFEST_DIR")).join("../../omega.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let pinned = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", lock_path.display()))
        .trim()
        .to_string();
    assert!(
        pinned.len() == 40 && pinned.bytes().all(|b| b.is_ascii_hexdigit()),
        "omega.lock must hold one full 40-hex sha"
    );
    println!("cargo:rustc-env=AIEN_OMEGA_PINNED_SHA={pinned}");

    let root = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let manifest = root.join("abi/omega_gpu.manifest");
    let artifacts_lock = root.join("../../native-artifacts.lock");
    println!("cargo:rerun-if-env-changed=AIEN_OMEGA_GPU_LIB_SHA256");
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        build::refuse_stub_if_required("aien-omega-gpu");
        build::Identity::stub(&pinned).emit(PREFIX);
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_GPU_LIB") {
        let lib = PathBuf::from(lib);
        assert!(
            lib.is_file(),
            "AIEN_OMEGA_GPU_LIB {} is not a file",
            lib.display()
        );
        build::qualify_prebuilt(
            "aien-omega-gpu",
            &manifest,
            &lib,
            "AIEN_OMEGA_GPU_LIB_SHA256",
            "unverified",
        )
        .emit(PREFIX);
        link(&lib);
        return;
    }
    let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") else {
        // AIEN_REQUIRE_NATIVE_LIBS=1 (set by scripts/pins.sh env.sh) turns the silent
        // stub fallback into a build error: a release or receipt build must never
        // discover at run time that it carries the CPU stub (sovereign-core #92).
        assert!(
            std::env::var("AIEN_REQUIRE_NATIVE_LIBS").as_deref() != Ok("1"),
            "aien-omega-gpu: AIEN_REQUIRE_NATIVE_LIBS=1 but neither AIEN_OMEGA_DIR nor AIEN_OMEGA_GPU_LIB is set (run scripts/pins.sh --gpu and source target/pins/env.sh)"
        );
        println!("cargo:warning=aien-omega-gpu: no AIEN_OMEGA_DIR or AIEN_OMEGA_GPU_LIB, building the CPU stub");
        build::Identity::stub(&pinned).emit(PREFIX);
        return;
    };
    let dir = PathBuf::from(dir);
    let head = Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    assert!(
        head == pinned,
        "omega checkout {} is at '{head}', omega.lock pins {pinned}",
        dir.display()
    );
    let physics = std::env::var("AIEN_PHYSICS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("../physics"));
    let out = PathBuf::from(env("OUT_DIR")).join("omega-build");
    if build::fresh_out_dir(&out, &pinned) {
        println!("cargo:warning=aien-omega-gpu: build dir was for another omega commit, rebuilt from clean");
    }
    let status = Command::new("make")
        .current_dir(&dir)
        .arg(format!("OUT_DIR={}", out.display()))
        .arg(format!("PHYSICS_DIR={}", physics.display()))
        .arg(out.join("libomega_gpu.a"))
        .status()
        .expect("run make");
    assert!(
        status.success(),
        "make libomega_gpu failed in {}",
        dir.display()
    );
    let archive = out.join("libomega_gpu.a");
    build::qualify_built(
        "aien-omega-gpu",
        &manifest,
        &artifacts_lock,
        &dir,
        &physics,
        &archive,
        &pinned,
        true,
    )
    .emit(PREFIX);
    link(&archive);
}

fn link(lib: &Path) {
    let dir = lib.parent().expect("lib has a parent dir");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=omega_gpu");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rustc-cfg=has_omega_gpu");
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"))
}
