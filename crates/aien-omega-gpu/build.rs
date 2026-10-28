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

//! Links the pinned Omega runtime companion first, then canonical CUDApp last.
//!
//! Offline source builds use AIEN_OMEGA_DIR at omega.lock and AIEN_CUDAPP_DIR
//! at that Omega revision's cudapp.lock. Omega's canonical staging entry freezes
//! and builds that CUDApp revision with the pinned compiler and bridge.
//! Both archives must match native-artifacts.lock; headers alone grant nothing.
//! AIEN_FORCE_CPU_STUB=1 preserves the explicit CPU-only build, unless native
//! libraries are required. No source selection builds the existing CPU stub.
//! Arbitrary prebuilt GPU archives are refused, even with a caller-selected hash.
//! Native capabilities are published to dependent build scripts only after ABI,
//! source and both archive identity checks succeed.
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
        "AIEN_CUDAPP_DIR",
        "OMEGA_OMC",
        "OMEGA_BRIDGE",
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
    println!("cargo:capabilities=");
    build::Identity::stub("").emit("AIEN_CUDAPP");

    let root = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let manifest = root.join("abi/omega_gpu.manifest");
    let artifacts_lock = root.join("../../native-artifacts.lock");
    println!("cargo:rerun-if-env-changed=AIEN_OMEGA_GPU_LIB_SHA256");
    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        build::refuse_stub_if_required("aien-omega-gpu");
        build::Identity::stub(&pinned).emit(PREFIX);
        return;
    }
    assert!(
        std::env::var_os("AIEN_OMEGA_GPU_LIB").is_none(),
        "AIEN_OMEGA_GPU_LIB is no longer supported: use pinned AIEN_OMEGA_DIR and AIEN_CUDAPP_DIR source builds"
    );
    let Ok(dir) = std::env::var("AIEN_OMEGA_DIR") else {
        // AIEN_REQUIRE_NATIVE_LIBS=1 (set by scripts/pins.sh env.sh) turns the silent
        // stub fallback into a build error: a release or receipt build must never
        // discover at run time that it carries the CPU stub (sovereign-core #92).
        assert!(
            std::env::var("AIEN_REQUIRE_NATIVE_LIBS").as_deref() != Ok("1"),
            "aien-omega-gpu: AIEN_REQUIRE_NATIVE_LIBS=1 but AIEN_OMEGA_DIR is not set (run scripts/pins.sh --gpu and source target/pins/env.sh)"
        );
        println!("cargo:warning=aien-omega-gpu: no AIEN_OMEGA_DIR, building the CPU stub");
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
    let cudapp = std::env::var("AIEN_CUDAPP_DIR")
        .expect("native GPU source build requires AIEN_CUDAPP_DIR at Omega's cudapp.lock");
    println!("cargo:rerun-if-changed={}", dir.join("cudapp.lock").display());
    let committed_pin = Command::new("git").arg("-C").arg(&dir)
        .args(["show", &format!("{pinned}:cudapp.lock")]).output()
        .expect("read committed canonical CUDApp pin");
    assert!(committed_pin.status.success(), "pinned Omega revision has no cudapp.lock");
    let locked_pin = std::fs::read(dir.join("cudapp.lock")).expect("read cudapp.lock");
    assert_eq!(locked_pin, committed_pin.stdout, "Omega cudapp.lock differs from its pinned revision");
    watch_checkout(&dir);
    watch_checkout(Path::new(&cudapp));
    for path in ["src", "Makefile", "tools/cudapp-stage.sh", "om/consumer.om"] {
        println!("cargo:rerun-if-changed={}", dir.join(path).display());
    }
    for key in ["OMEGA_OMC", "OMEGA_BRIDGE"] {
        if let Ok(path) = std::env::var(key) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
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
        .arg(format!("CUDAPP_DIR={cudapp}"))
        .arg(out.join("libomega_gpu.a"))
        .status()
        .expect("run make");
    assert!(
        status.success(),
        "make libomega_gpu failed in {}",
        dir.display()
    );
    let archive = out.join("libomega_gpu.a");
    // Resolve once: pin, headers and archive must belong to the same immutable
    // stage even if another make atomically publishes a newer stage.
    let stage = std::fs::canonicalize(out.join("cudapp"))
        .expect("Omega did not publish its canonical CUDApp stage");
    let canonical = stage.join("source");
    let canonical_pin = std::fs::read_to_string(stage.join("pin"))
        .expect("canonical stage is missing its frozen CUDApp pin");
    assert_eq!(canonical_pin.as_bytes(), locked_pin, "canonical stage pin differs from pinned Omega cudapp.lock");
    let canonical_archive = canonical.join("build/libcudapp.a");
    let (companion_identity, canonical_identity) = build::qualify_gpu_pair(
        &manifest,
        &artifacts_lock,
        &dir,
        &physics,
        &archive,
        &pinned,
        &canonical,
        canonical_pin.trim(),
    );
    companion_identity.emit(PREFIX);
    canonical_identity.emit("AIEN_CUDAPP");
    link(&archive, &canonical_archive);
}

fn link(companion: &Path, canonical: &Path) {
    for (lib, name) in [(companion, "omega_gpu"), (canonical, "cudapp")] {
        let dir = lib.parent().expect("lib has a parent dir");
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
        println!("cargo:rerun-if-changed={}", lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    emit_capabilities(companion, canonical);
    println!("cargo:rustc-cfg=has_omega_gpu");
}

fn emit_capabilities(companion: &Path, canonical: &Path) {
    use aien_abi_conformance::probe::defined_symbols;
    let mut symbols = defined_symbols(companion).expect("cannot inspect Omega GPU companion");
    symbols.extend(defined_symbols(canonical).expect("cannot inspect canonical CUDApp archive"));
    let has = |name: &str| symbols.iter().any(|s| s == name);
    let fp8 = has("omega_gpu_fp8_gemv_grouped_f32");
    let q38 = has("omega_gpu_q38_graph_create");
    let prefill = has("omega_gpu_pf_qgemm")
        && has("omega_gpu_pf_quant") && has("omega_gpu_pf_alloc_dev");
    let pattn = has("omega_gpu_prefill_attn");
    let checks = [
        ("has_omega_bf16", has("omega_gpu_bf16_gemv_f32")),
        ("has_omega_dattn", has("omega_gpu_decode_attn")),
        ("has_omega_dgraph", has("omega_gpu_decode_graph_create")),
        ("has_omega_moe", has("omega_gpu_matmul_resident_windows_f32")),
        ("has_omega_router", has("omega_gpu_router")),
        ("has_omega_fp8", fp8),
        ("has_omega_fp8_dev", fp8 && has("omega_gpu_fp8_gemv_grouped_dev") && has("omega_gpu_fp8_tensor_entry")),
        ("has_omega_fp8gemm", fp8 && has("omega_gpu_fp8gemm_f32")),
        ("has_omega_fp8_grouped_rule", fp8 && has("omega_gpu_fp8_grouped_ctas_auto")),
        ("has_omega_pf", prefill),
        ("has_omega_pattn", pattn),
        ("has_omega_pattn_dev", pattn && has("omega_gpu_prefill_attn_dev")),
        ("has_omega_pfhost", has("omega_gpu_pf_qgemm") && has("omega_gpu_pf_router_logits")
            && has("omega_gpu_pf_moe_combine") && has("omega_gpu_pf_router_pack") && has("omega_gpu_pf_router_rows")),
        ("has_omega_q38", q38),
        ("has_omega_q36", q38 && has("omega_gpu_q36_graph_create")),
        ("has_omega_q38_prefill", q38 && has("omega_gpu_gdn_prefill") && has("omega_gpu_prefill_attn256")),
    ];
    let capabilities = checks.iter().filter_map(|(cfg, present)| present.then_some(*cfg))
        .collect::<Vec<_>>().join(" ");
    println!("cargo:capabilities={capabilities}");
}

fn watch_checkout(dir: &Path) {
    let branch = Command::new("git").arg("-C").arg(dir)
        .args(["symbolic-ref", "-q", "HEAD"]).output()
        .expect("read checkout branch");
    let mut refs = vec!["HEAD".to_string()];
    if branch.status.success() {
        refs.push(String::from_utf8(branch.stdout).expect("git ref UTF-8").trim().into());
    }
    for reference in refs {
        let path = Command::new("git").arg("-C").arg(dir)
            .args(["rev-parse", "--path-format=absolute", "--git-path", &reference])
            .output().expect("read checkout ref path");
        assert!(path.status.success(), "cannot watch native checkout refs");
        println!("cargo:rerun-if-changed={}", String::from_utf8_lossy(&path.stdout).trim());
    }
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"))
}
