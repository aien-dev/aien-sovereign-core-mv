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

//! Links the authenticated Omega CPU/evidence companion first and canonical CUDApp last.
//! The Rust GPU ABI is checked only against the pinned CUDApp source and archive.
//! Both digests remain independently subject to native-artifacts.lock; unrecorded
//! development builds do not qualify. No GPU or model is opened by this build script.
use aien_abi_conformance::build;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREFIX: &str = "AIEN_OMEGA_GPU";
const COMPANION: &str = "AIEN_OMEGA_COMPANION";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_gpu)");
    for v in ["AIEN_FORCE_CPU_STUB", "AIEN_OMEGA_GPU_LIB", "AIEN_OMEGA_GPU_LIB_SHA256",
        "AIEN_OMEGA_DIR", "AIEN_CUDAPP_DIR", "AIEN_PHYSICS_DIR", "AIEN_REQUIRE_NATIVE_LIBS",
        "OMEGA_OMC", "OMEGA_BRIDGE"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    assert!(std::env::var_os("AIEN_OMEGA_GPU_LIB").is_none(),
        "AIEN_OMEGA_GPU_LIB is obsolete: the canonical GPU boundary requires authenticated source and both archives");
    let root = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let lock_path = root.join("../../omega.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let pinned = read_pin(&lock_path);
    println!("cargo:rustc-env=AIEN_OMEGA_PINNED_SHA={pinned}");
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        build::refuse_stub_if_required("aien-omega-gpu");
        emit_stub(&pinned);
        return;
    }
    let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") else {
        assert!(!build::require_native(),
            "aien-omega-gpu: AIEN_REQUIRE_NATIVE_LIBS=1 but AIEN_OMEGA_DIR is not set (run scripts/pins.sh --gpu)");
        println!("cargo:warning=aien-omega-gpu: no AIEN_OMEGA_DIR, building the CPU stub");
        emit_stub(&pinned);
        return;
    };
    let dir = PathBuf::from(dir).canonicalize().expect("resolve Omega checkout");
    let head = git(&dir, &["rev-parse", "HEAD"]);
    assert_eq!(head.trim(), pinned, "Omega checkout does not match omega.lock");
    let physics = std::env::var("AIEN_PHYSICS_DIR").map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("../physics")).canonicalize().expect("resolve physics checkout");
    let cudapp = PathBuf::from(env("AIEN_CUDAPP_DIR")).canonicalize().expect("resolve CUDApp checkout");
    let out = PathBuf::from(env("OUT_DIR")).join("omega-build");
    build::fresh_out_dir(&out, &pinned);
    // Build only authenticated commit bytes, not dirty or ignored worktree overlays.
    let source = materialize_source(&dir, &pinned, &lock_path, &out);
    let cudapp_lock = source.join("cudapp.lock");
    let canonical_pin = read_pin(&cudapp_lock);
    println!("cargo:rustc-env=AIEN_OMEGA_CUDAPP_PINNED_SHA={canonical_pin}");
    let status = Command::new("make").current_dir(&source)
        .env_remove("BASH_ENV")
        .arg(format!("OUT_DIR={}", out.display()))
        .arg(format!("PHYSICS_DIR={}", physics.display()))
        .arg(format!("CUDAPP_DIR={}", cudapp.display()))
        .arg("libomega_gpu").status().expect("run canonical native build");
    assert!(status.success(), "canonical native archive build failed");
    // Resolve the publication symlink once, then authenticate and link that same stage.
    let stage = out.join("cudapp").canonicalize().expect("resolve generated CUDApp stage");
    let status = Command::new("sh").arg(source.join("tools/cudapp-stage.sh"))
        .env_remove("BASH_ENV")
        .arg("--verify").arg(&cudapp_lock).arg(&stage).status().expect("authenticate CUDApp stage");
    assert!(status.success(), "canonical CUDApp stage authentication failed");
    for input in [cudapp_lock.clone(), stage.join("pin"), stage.join("observed"),
        stage.join("archive.sha256"), stage.join("source.sha256"), stage.join("guard.sha256")] {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    // Cargo recursively watches this pristine source; a modified header or an extra file
    // invalidates the cached build and is refused by the next authentication.
    println!("cargo:rerun-if-changed={}", stage.join("source").display());
    let canonical_source = stage.join("source");
    let canonical_archive = stage.join("build/libcudapp.a");
    let companion_archive = out.join("libomega_gpu.a");
    let manifest = root.join("abi/omega_gpu.manifest");
    let artifacts = root.join("../../native-artifacts.lock");
    let includes = ["src", "m16", "nvrm",
        "third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc",
        "third_party/nvidia-open-580.173.02/kernel-open/common/inc",
        "third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm",
        "third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include"]
        .iter().map(|p| canonical_source.join(p)).collect();
    let canonical = build::qualify_built_with_includes("aien-omega-gpu", &manifest, &artifacts,
        &canonical_source, &canonical_archive, &canonical_pin, true, includes);
    canonical.emit(PREFIX);
    let companion = build::qualify_companion("aien-omega-gpu", &manifest, &artifacts,
        &source, &companion_archive, &pinned);
    companion.emit(COMPANION);
    // CUDApp also has an archive named libomega_gpu.a. Only this private directory
    // is searched for the companion, so the checked archive cannot be substituted.
    let link_dir = out.join("admitted-link");
    std::fs::create_dir_all(&link_dir).expect("create admitted archive directory");
    copy_archive(&companion_archive, &link_dir.join("libomega_gpu.a"), &companion.archive_sha256);
    copy_archive(&canonical_archive, &link_dir.join("libcudapp.a"), &canonical.archive_sha256);
    println!("cargo:canonical_archive={}", link_dir.join("libcudapp.a").display());
    println!("cargo:canonical_sha256={}", canonical.archive_sha256);
    println!("cargo:canonical_source={}", canonical_source.display());
    println!("cargo:rustc-link-search=native={}", link_dir.display());
    println!("cargo:rustc-link-lib=static=omega_gpu");
    println!("cargo:rustc-link-lib=static=cudapp");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-cfg=has_omega_gpu");
}

fn copy_archive(source: &Path, destination: &Path, admitted_sha: &str) {
    if aien_abi_conformance::identity::sha256_file(destination).ok().as_deref() != Some(admitted_sha) {
        std::fs::copy(source, destination).expect("snapshot admitted archive");
    }
    assert_eq!(admitted_sha,
        aien_abi_conformance::identity::sha256_file(destination).expect("hash admitted link input"),
        "admitted archive changed before linking");
    println!("cargo:rerun-if-changed={}", destination.display());
}

fn emit_stub(pinned: &str) {
    println!("cargo:rustc-env=AIEN_OMEGA_CUDAPP_PINNED_SHA=");
    build::Identity::stub("").emit(PREFIX);
    build::Identity::stub(pinned).emit(COMPANION);
}

fn materialize_source(repo: &Path, pinned: &str, lock: &Path, out: &Path) -> PathBuf {
    // The checker cannot establish its own trust after execution. Authenticate the
    // bounded commit/tree/blob chain first, then execute only these private bytes.
    let helper = out.join("authenticated-source-checker.sh");
    let checker = aien_abi_conformance::source::authenticated_source_checker(repo, pinned);
    if std::fs::read(&helper).ok().as_deref() != Some(checker.as_slice()) {
        std::fs::write(&helper, checker).expect("snapshot authenticated source checker");
    }
    let roots = git(repo, &["ls-tree", "-z", "--name-only", pinned]);
    let expected_roots: BTreeSet<String> = roots.split('\0').filter(|s| !s.is_empty()).map(str::to_owned).collect();
    let source = out.join("source");
    let run = |mode: &str, names: &BTreeSet<String>| {
        let status = Command::new("bash").arg(&helper).arg(mode).arg(&source).args(names)
            .env_remove("BASH_ENV")
            .env("AIENOS_LOCK", "").env("AIENOS_LOCK_FILE", lock).env("AIENOS_LOCK_REPO", repo)
            .status().expect("run raw-object source authentication");
        assert!(status.success(), "Omega source snapshot authentication failed");
    };
    run("materialize", &expected_roots);
    let mut roots = expected_roots;
    for entry in std::fs::read_dir(&source).expect("read source snapshot entries") {
        roots.insert(entry.expect("read source entry").file_name().into_string().expect("UTF-8 source entry"));
    }
    run("verify-dir", &roots);
    println!("cargo:rerun-if-changed={}", helper.display());
    source
}

fn git(repo: &Path, args: &[&str]) -> String {
    let result = Command::new("git").arg("-C").arg(repo)
        .env("GIT_NO_REPLACE_OBJECTS", "1").env("GIT_GRAFT_FILE", "/nonexistent/core-native-grafts")
        .args(args).output().expect("read native source identity");
    assert!(result.status.success(), "git source identity failed: {args:?}");
    String::from_utf8(result.stdout).expect("UTF-8 git identity")
}

fn read_pin(path: &Path) -> String {
    let pin = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let pin = pin.trim();
    assert!(pin.len() == 40 && pin.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "{} must hold one lowercase full 40-hex source revision", path.display());
    pin.into()
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} not set"))
}
