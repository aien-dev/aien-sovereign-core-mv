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

//! The Rust mirror of the omega GPU ABI (src/ffi.rs) must equal abi/omega_gpu.manifest,
//! field by field (offset, size, kind, order), constant by constant and signature by
//! signature. Runs in every build, the stub included.

#[test]
fn rust_ffi_matches_the_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("abi/omega_gpu.manifest");
    aien_abi_conformance::assert_manifest(&aien_omega_gpu::ffi::rust_abi(), &path);
}

/// What this test binary linked, as `build.rs` recorded it. Prints one
/// `AIEN_NATIVE_IDENTITY {json}` line (scripts/abi-qualify.sh collects it). A linked build
/// must never report the stub, a stub build must never qualify, and with
/// `AIEN_ABI_QUALIFY=1` (the qualification run) the identity must qualify.
#[test]
fn native_identity_matches_the_link() {
    let id = aien_omega_gpu::native_identity();
    println!("AIEN_NATIVE_IDENTITY {}", id.to_json());
    let canonical = aien_omega_gpu::canonical_identity();
    println!("AIEN_NATIVE_IDENTITY {}", canonical.to_json());
    if cfg!(has_omega_gpu) {
        assert_ne!(id.link, "stub", "{id:?}");
        assert!(id.digest != "none", "{id:?}");
        assert!(canonical.qualifies(), "{canonical:?}");
    } else {
        assert_eq!(id.link, "stub", "{id:?}");
        assert!(!id.qualifies());
        assert_eq!(canonical.link, "stub", "{canonical:?}");
        assert!(!canonical.qualifies());
    }
    if std::env::var("AIEN_ABI_QUALIFY").as_deref() == Ok("1") {
        assert!(
            id.qualifies(),
            "{} does not qualify: {:?}",
            id.component,
            id.gaps()
        );
    }
}

/// Execute a real linked archive symbol without opening a GPU device/session.
#[cfg(has_omega_gpu)]
#[test]
fn canonical_archive_host_link_smoke() {
    use std::ffi::CStr;
    unsafe {
        assert_eq!(
            CStr::from_ptr(aien_omega_gpu::ffi::omega_gpu_matmul_rc_name(0)).to_bytes(),
            b"OK",
        );
        assert_eq!(
            CStr::from_ptr(aien_omega_gpu::ffi::omega_gpu_matmul_rc_name(-1)).to_bytes(),
            b"BAD_ARGS",
        );
    }
}

/// Challenge the existing qualifier using copies of the actual generated archives.
/// Run explicitly with AIEN_GPU_STAGE and AIEN_GPU_COMPANION from the native build.
#[cfg(all(has_omega_gpu, unix))]
#[test]
#[ignore = "requires the exact generated native stage and companion paths"]
fn canonical_pair_artifact_controls() {
    use aien_abi_conformance::build::qualify_gpu_pair;
    use std::{fs, path::PathBuf};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let stage = fs::canonicalize(std::env::var("AIEN_GPU_STAGE").expect("AIEN_GPU_STAGE")).unwrap();
    let companion = PathBuf::from(std::env::var("AIEN_GPU_COMPANION").expect("AIEN_GPU_COMPANION"));
    let omega = PathBuf::from(std::env::var("AIEN_OMEGA_DIR").expect("AIEN_OMEGA_DIR"));
    let physics = PathBuf::from(std::env::var("AIEN_PHYSICS_DIR").expect("AIEN_PHYSICS_DIR"));
    let canonical = stage.join("source");
    let pin = fs::read_to_string(stage.join("pin")).unwrap();
    let lock = root.join("../../native-artifacts.lock");
    let manifest = root.join("abi/omega_gpu.manifest");
    let scratch = std::env::temp_dir().join(format!("core-canonical-controls-{}", std::process::id()));
    fs::create_dir(&scratch).unwrap();
    std::env::set_var("OUT_DIR", &scratch);
    std::env::set_var("TARGET", aien_omega_gpu::native_identity().target);
    std::env::set_var("HOST", aien_omega_gpu::native_identity().target);
    let qualify = |source: &std::path::Path, runtime: &std::path::Path, records: &std::path::Path| {
        qualify_gpu_pair(
            &manifest, records, &omega, &physics, runtime,
            aien_omega_gpu::PINNED_OMEGA_SHA, source, pin.trim(),
        )
    };
    let original = qualify(&canonical, &companion, &lock);
    assert_eq!(original.0.archive_sha256, aien_omega_gpu::native_identity().archive_sha256);
    assert_eq!(original.1.archive_sha256, aien_omega_gpu::canonical_identity().archive_sha256);
    let copied = scratch.join("canonical");
    fs::create_dir_all(copied.join("build")).unwrap();
    std::os::unix::fs::symlink(canonical.join("src"), copied.join("src")).unwrap();
    let archive = copied.join("build/libcudapp.a");
    let refuses = |source: &std::path::Path, runtime: &std::path::Path, records: &std::path::Path| {
        assert!(std::panic::catch_unwind(|| qualify(source, runtime, records)).is_err());
    };
    refuses(&copied, &companion, &lock); // missing canonical artifact
    fs::copy(canonical.join("build/libcudapp.a"), &archive).unwrap();
    let mut stale = fs::read(&archive).unwrap();
    stale.push(0);
    fs::write(&archive, stale).unwrap();
    refuses(&copied, &companion, &lock); // stale/corrupt canonical artifact
    let stale_companion = scratch.join("libomega_gpu.a");
    let mut stale = fs::read(&companion).unwrap();
    stale.push(0);
    fs::write(&stale_companion, stale).unwrap();
    refuses(&canonical, &stale_companion, &lock);
    let records = fs::read_to_string(&lock).unwrap();
    for missing in ["libomega_gpu.a", "libcudapp.a"] {
        let without = records.lines().filter(|line| !line.starts_with(&format!("{missing} ")))
            .collect::<Vec<_>>().join("\n");
        let incomplete = scratch.join("missing.lock");
        fs::write(&incomplete, without).unwrap();
        refuses(&canonical, &companion, &incomplete);
    }
    fs::remove_dir_all(scratch).unwrap();
}
