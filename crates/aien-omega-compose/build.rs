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

//! Links omega `librx_compose.a` (COMPOSITION-2 + the rxc_host ABI facade).
//!
//! Offline: nothing is downloaded. Order:
//!   1. `AIEN_FORCE_CPU_STUB=1`          -> stub (no library; every call returns
//!      `ComposeError::Unavailable`). Refused when `AIEN_REQUIRE_NATIVE_LIBS=1`.
//!   2. `AIEN_OMEGA_COMPOSE_LIB=<file>`  -> link that prebuilt archive; `AIEN_OMEGA_COMPOSE_LIB_SHA256`
//!      must name its sha256 (checked) and it must define every function in the ABI manifest.
//!      A development override: its identity says `prebuilt` and it never qualifies.
//!   3. `AIEN_OMEGA_COMPOSE_DIR=<checkout>`, else `AIEN_OMEGA_DIR` (the checkout
//!      aien-omega-gpu builds from) -> its HEAD must equal omega.lock, then
//!      `make <OUT_DIR>/librx_compose.a` runs in it. `AIEN_OMEGA_COMPOSE_SHA=<40-hex>`
//!      replaces the expected sha: a documented developer override for building
//!      against an unmerged omega branch, never used for a release or a receipt. `AIEN_PHYSICS_DIR` (default `<omega>/../physics`) is passed
//!      as PHYSICS_DIR and `AIEN_AIENOS_LOCK_REPO` (if set) as AIENOS_LOCK_REPO;
//!      omega's own lock checks still run.
//!   4. otherwise                        -> stub, with a build warning.
//!
//! `AIEN_OMEGA_COMPOSE_DIR` exists so a developer override never touches the
//! checkout aien-omega-gpu checks against omega.lock.
//! `cfg(has_omega_compose)` is set only when the real library is linked.
//!
//! Before linking a checkout build, `aien_abi_conformance::build::qualify_built` checks the
//! pinned `rxc_host_abi.h`, a compiled layout probe and the archive symbols against
//! `abi/rxc_host_abi.manifest`, and the archive sha256 against `native-artifacts.lock`; any
//! mismatch fails the build. The result is recorded for `native_identity()`.
use aien_abi_conformance::build;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREFIX: &str = "AIEN_OMEGA_COMPOSE";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_compose)");
    println!("cargo:rustc-check-cfg=cfg(has_omega_wait_ms)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_COMPOSE_LIB",
        "AIEN_OMEGA_COMPOSE_DIR",
        "AIEN_OMEGA_DIR",
        "AIEN_OMEGA_COMPOSE_SHA",
        "AIEN_PHYSICS_DIR",
        "AIEN_AIENOS_LOCK_REPO",
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
    assert!(is_sha(&pinned), "omega.lock must hold one full 40-hex sha");
    let expected = match std::env::var("AIEN_OMEGA_COMPOSE_SHA") {
        Ok(s) => {
            let s = s.trim().to_string();
            assert!(
                is_sha(&s),
                "AIEN_OMEGA_COMPOSE_SHA must be a full 40-hex sha"
            );
            println!("cargo:warning=aien-omega-compose: AIEN_OMEGA_COMPOSE_SHA={s} overrides omega.lock {pinned} (developer override, not omega.lock)");
            s
        }
        Err(_) => pinned.clone(),
    };
    println!("cargo:rustc-env=AIEN_OMEGA_COMPOSE_EXPECTED_SHA={expected}");

    let root = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let manifest = root.join("abi/rxc_host_abi.manifest");
    let artifacts_lock = root.join("../../native-artifacts.lock");
    println!("cargo:rerun-if-env-changed=AIEN_OMEGA_COMPOSE_LIB_SHA256");

    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        build::refuse_stub_if_required("aien-omega-compose");
        build::Identity::stub(&expected).emit(PREFIX);
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_COMPOSE_LIB") {
        let lib = PathBuf::from(lib);
        assert!(
            lib.is_file(),
            "AIEN_OMEGA_COMPOSE_LIB {} is not a file",
            lib.display()
        );
        build::qualify_prebuilt(
            "aien-omega-compose",
            &manifest,
            &lib,
            "AIEN_OMEGA_COMPOSE_LIB_SHA256",
            "unverified",
        )
        .emit(PREFIX);
        require_wait_ms_symbol(&lib);
        link(&lib);
        return;
    }
    let Ok(dir) =
        std::env::var("AIEN_OMEGA_COMPOSE_DIR").or_else(|_| std::env::var("AIEN_OMEGA_DIR"))
    else {
        // See aien-omega-gpu/build.rs: AIEN_REQUIRE_NATIVE_LIBS=1 refuses the stub.
        assert!(
            !build::require_native(),
            "aien-omega-compose: AIEN_REQUIRE_NATIVE_LIBS=1 but no AIEN_OMEGA_COMPOSE_DIR, AIEN_OMEGA_DIR or AIEN_OMEGA_COMPOSE_LIB is set (run scripts/pins.sh and source target/pins/env.sh)"
        );
        println!("cargo:warning=aien-omega-compose: no AIEN_OMEGA_COMPOSE_DIR, AIEN_OMEGA_DIR or AIEN_OMEGA_COMPOSE_LIB, building the stub");
        build::Identity::stub(&expected).emit(PREFIX);
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
        head == expected,
        "omega checkout {} is at '{head}', expected {expected}",
        dir.display()
    );
    let physics = std::env::var("AIEN_PHYSICS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("../physics"));
    let out = PathBuf::from(env("OUT_DIR")).join("omega-compose-build");
    if build::fresh_out_dir(&out, &expected) {
        println!("cargo:warning=aien-omega-compose: build dir was for another omega commit, rebuilt from clean");
    }
    let mut make = Command::new("make");
    make.current_dir(&dir)
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .arg(format!("OUT_DIR={}", out.display()))
        .arg(format!("PHYSICS_DIR={}", physics.display()));
    if let Ok(repo) = std::env::var("AIEN_AIENOS_LOCK_REPO") {
        make.arg(format!("AIENOS_LOCK_REPO={repo}"));
    }
    let status = make
        .arg(out.join("librx_compose.a"))
        .status()
        .expect("run make");
    assert!(
        status.success(),
        "make librx_compose.a failed in {}",
        dir.display()
    );
    detect_wait_ms(&dir);
    let archive = out.join("librx_compose.a");
    build::qualify_built(
        "aien-omega-compose",
        &manifest,
        &artifacts_lock,
        &dir,
        &physics,
        &archive,
        &expected,
        expected == pinned,
    )
    .emit(PREFIX);
    require_wait_ms_symbol(&archive);
    link(&archive);
}

/// Check the omega checkout's host header. omega.lock pins a commit that
/// declares `rxc_host_set_wait_ms`, so a real link without it is a build error, never a silent
/// downgrade (the 120 s document budget would be refused at runtime).
fn detect_wait_ms(dir: &Path) {
    let h = dir.join("src/runtime/rxc_host_abi.h");
    let text =
        std::fs::read_to_string(&h).unwrap_or_else(|e| panic!("cannot read {}: {e}", h.display()));
    assert!(
        header_declares_wait_ms(&text),
        "omega header {} does not declare rxc_host_set_wait_ms; omega.lock pins a commit that has it, so this checkout is not usable. Use AIEN_FORCE_CPU_STUB=1 for a stub build.",
        h.display()
    );
}

/// A real declaration, not a mention in a comment: a line that starts with
/// `int rxc_host_set_wait_ms(`.
fn header_declares_wait_ms(header: &str) -> bool {
    header
        .lines()
        .any(|l| l.trim_start().starts_with("int rxc_host_set_wait_ms("))
}

/// Fail the build unless the linked archive defines `rxc_host_set_wait_ms` (checked with `nm`
/// on both the make path and a prebuilt `AIEN_OMEGA_COMPOSE_LIB`), then set `has_omega_wait_ms`.
fn require_wait_ms_symbol(lib: &Path) {
    let out = Command::new("nm")
        .arg("-g")
        .arg("--defined-only")
        .arg(lib)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "cannot run `nm` to check {} for rxc_host_set_wait_ms ({e}); install binutils or use AIEN_FORCE_CPU_STUB=1",
                lib.display()
            )
        });
    assert!(
        out.status.success(),
        "`nm` failed on {}: {}",
        lib.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        archive_defines_wait_ms(&text),
        "{} does not define rxc_host_set_wait_ms; omega.lock pins a commit that has it. Rebuild the archive from that omega, or use AIEN_FORCE_CPU_STUB=1 for a stub build.",
        lib.display()
    );
    println!("cargo:rustc-cfg=has_omega_wait_ms");
}

fn archive_defines_wait_ms(nm_out: &str) -> bool {
    nm_out
        .lines()
        .any(|l| l.split_whitespace().last() == Some("rxc_host_set_wait_ms"))
}

fn link(lib: &Path) {
    let dir = lib.parent().expect("lib has a parent dir");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=rx_compose");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rustc-cfg=has_omega_compose");
    println!("cargo:linked=1");
}

fn is_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"))
}
