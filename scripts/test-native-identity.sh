#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Native identity checks in the build scripts of aien-omega-compose and aien-omega-gpu.
# Runs without omega: every case below must be decided by build.rs before any make.
#   - a stub together with AIEN_REQUIRE_NATIVE_LIBS=1 is refused
#   - AIEN_REQUIRE_NATIVE_LIBS=1 with no library source is refused
#   - a prebuilt archive with no expected sha256, a wrong one, or a truncated archive is refused
#   - a prebuilt archive that misses one ABI function is refused
#   - an omega checkout at a commit other than omega.lock is refused
#   - a correct prebuilt archive builds, and its identity says "prebuilt" (never qualifies)
# Exit 0 when every case behaves, 1 otherwise.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export CARGO_TARGET_DIR="${AIEN_IDENTITY_TARGET_DIR:-$ROOT/target/native-identity}"
STAGE_INPUT="${AIEN_CANONICAL_STAGE:-}"
OMEGA_INPUT="${AIEN_OMEGA_DIR:-}"
OMC_INPUT="${OMEGA_OMC:-}"
BRIDGE_INPUT="${OMEGA_BRIDGE:-}"
for v in $(env | sed -n 's/^\(AIEN_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
HOST_TARGET=$(rustc -vV | sed -n 's/^host: //p')
[ -n "$HOST_TARGET" ] || { echo "cannot determine rustc host target" >&2; exit 1; }

fails=0
pass() { echo "ok   $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }

# expect_refused <crate> <case> <pattern> [VAR=value ...]
expect_refused() {
    local crate=$1 name=$2 pat=$3
    shift 3
    local log="$WORK/log"
    if env "$@" cargo check -q -p "$crate" >"$log" 2>&1; then
        fail "$crate: $name (build was accepted)"
    elif grep -qF -- "$pat" "$log"; then
        pass "$crate: $name"
    else
        fail "$crate: $name (refused, but not with '$pat')"
        tail -5 "$log"
    fi
}

# fake_archive <manifest> <out.a> [function to leave out]
fake_archive() {
    local src="$WORK/fake-$$-$RANDOM.c"
    sed -n 's/^fn \([A-Za-z0-9_]*\) .*/\1/p' "$1" | { grep -vx "${3:-}" || true; } |
        sed 's/.*/int &(void) { return 0; }/' >"$src"
    cc -c -o "$src.o" "$src"
    rm -f "$2"
    ar rcs "$2" "$src.o"
}

# A git checkout at a commit that is not omega.lock.
WRONG="$WORK/wrong-omega"
git init -q "$WRONG"
git -C "$WRONG" -c user.name=t -c user.email=t@t commit -q --allow-empty -m wrong

check_crate() {
    local crate=$1 manifest=$2 libvar=$3 dirvar=$4 drop=$5
    local good="$WORK/$crate-good.a" short="$WORK/$crate-short.a" miss="$WORK/$crate-miss.a"
    fake_archive "$manifest" "$good"
    fake_archive "$manifest" "$miss" "$drop"
    local sha miss_sha
    sha=$(sha256sum "$good" | cut -d' ' -f1)
    miss_sha=$(sha256sum "$miss" | cut -d' ' -f1)
    head -c 100 "$good" >"$short"

    expect_refused "$crate" "stub with required native" "contradict each other" \
        AIEN_FORCE_CPU_STUB=1 AIEN_REQUIRE_NATIVE_LIBS=1
    expect_refused "$crate" "required native with no library" "AIEN_REQUIRE_NATIVE_LIBS=1 but" \
        AIEN_REQUIRE_NATIVE_LIBS=1
    expect_refused "$crate" "prebuilt with no sha256" "no expected digest" \
        "$libvar=$good"
    expect_refused "$crate" "prebuilt with a wrong sha256" "expects" \
        "$libvar=$good" "${libvar}_SHA256=$(printf '0%.0s' {1..64})"
    expect_refused "$crate" "truncated prebuilt" "truncated or corrupted" \
        "$libvar=$short" "${libvar}_SHA256=$sha"
    expect_refused "$crate" "prebuilt missing $drop" "does not define" \
        "$libvar=$miss" "${libvar}_SHA256=$miss_sha"
    expect_refused "$crate" "omega checkout at the wrong commit" "is at '" \
        "$dirvar=$WRONG"

    if env "$libvar=$good" "${libvar}_SHA256=$sha" cargo check -q -p "$crate" >"$WORK/log" 2>&1 &&
        grep -qsx "cargo:rustc-env=${6}_LINK=prebuilt" "$(ls -t "$CARGO_TARGET_DIR"/debug/build/"$crate"-*/output | head -1)"; then
        pass "$crate: matching prebuilt builds and says prebuilt"
    else
        fail "$crate: matching prebuilt builds and says prebuilt"
        tail -5 "$WORK/log"
    fi
}

cd "$ROOT"
check_crate aien-omega-compose crates/aien-omega-compose/abi/rxc_host_abi.manifest \
    AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_DIR rxc_host_recover AIEN_OMEGA_COMPOSE
check_gpu() {
    local crate=aien-omega-gpu
    local fake="$WORK/$crate-fake.a"
    touch "$fake"

    expect_refused "$crate" "obsolete prebuilt override refused" "AIEN_OMEGA_GPU_LIB is obsolete" \
        AIEN_OMEGA_GPU_LIB="$fake"
    expect_refused "$crate" "obsolete prebuilt override refused even with stub" "AIEN_OMEGA_GPU_LIB is obsolete" \
        AIEN_FORCE_CPU_STUB=1 AIEN_OMEGA_GPU_LIB="$fake"
    expect_refused "$crate" "stub with required native" "contradict each other" \
        AIEN_FORCE_CPU_STUB=1 AIEN_REQUIRE_NATIVE_LIBS=1
    expect_refused "$crate" "required native with no library" "AIEN_REQUIRE_NATIVE_LIBS=1 but" \
        AIEN_REQUIRE_NATIVE_LIBS=1
    expect_refused "$crate" "omega checkout at the wrong commit" "does not match omega.lock" \
        AIEN_OMEGA_DIR="$WRONG"

    # Two-archive negative controls via aien-abi-conformance:
    local probe_dir="$WORK/native-identity-probe"
    mkdir -p "$probe_dir/src"
    cat <<'EOF' >"$probe_dir/Cargo.toml"
[package]
name = "native-identity-probe"
version = "0.1.0"
edition = "2021"

[dependencies]
aien-abi-conformance = { path = "CRATES_ABI_CONFORMANCE" }
EOF
    sed -i "s|CRATES_ABI_CONFORMANCE|$ROOT/crates/aien-abi-conformance|" "$probe_dir/Cargo.toml"

    cat <<'EOF' >"$probe_dir/src/main.rs"
use aien_abi_conformance::build::{qualify_built_with_includes, qualify_companion};
use aien_abi_conformance::identity;
use std::path::PathBuf;
use std::process::Command;
fn env_or(k: &str, def: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| def.to_string())
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        String::new()
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("AIEN_ROOT").expect("AIEN_ROOT"));
    let manifest = root.join("crates/aien-omega-gpu/abi/omega_gpu.manifest");
    let artifacts_lock = root.join("native-artifacts.lock");
    let temp = PathBuf::from(std::env::var("AIEN_WORK").expect("AIEN_WORK"));

    // Case 1: Monolithic companion shadow.
    // Companion defining a GPU manifest function (e.g. omega_gpu_matmul_f32) must be rejected.
    {
        let shadow_dir = temp.join("shadow-companion");
        std::fs::create_dir_all(&shadow_dir).unwrap();
        let src = shadow_dir.join("shadow.c");
        std::fs::write(&src, b"int omega_gpu_matmul_f32(void) { return 0; }\n").unwrap();
        assert!(Command::new("cc").current_dir(&shadow_dir).args(["-c", "-o", "shadow.o", "shadow.c"]).status().unwrap().success());
        let archive = shadow_dir.join("libomega_gpu.a");
        let _ = std::fs::remove_file(&archive);
        assert!(Command::new("ar").current_dir(&shadow_dir).args(["rcs", "libomega_gpu.a", "shadow.o"]).status().unwrap().success());
        
        let res = std::panic::catch_unwind(|| {
            qualify_companion("aien-omega-gpu", &manifest, &artifacts_lock, &shadow_dir, &archive, "dummy_source");
        });
        let err = res.expect_err("companion shadowing canonical GPU ABI must be rejected");
        let msg = panic_message(err);
        assert!(msg.contains("companion shadows canonical GPU ABI functions"), "unexpected panic: {msg}");
        println!("ok aien-omega-gpu: companion shadowing canonical GPU ABI rejected");
    }

    // Case 2: Companion unrecorded digest.
    {
        let comp_dir = temp.join("unrecorded-companion");
        std::fs::create_dir_all(&comp_dir).unwrap();
        let src = comp_dir.join("comp.c");
        std::fs::write(&src, b"int omega_cpu_helper(void) { return 42; }\n").unwrap();
        assert!(Command::new("cc").current_dir(&comp_dir).args(["-c", "-o", "comp.o", "comp.c"]).status().unwrap().success());
        let archive = comp_dir.join("libomega_gpu.a");
        let _ = std::fs::remove_file(&archive);
        assert!(Command::new("ar").current_dir(&comp_dir).args(["rcs", "libomega_gpu.a", "comp.o"]).status().unwrap().success());
        
        let id = qualify_companion("aien-omega-gpu", &manifest, &artifacts_lock, &comp_dir, &archive, "dummy_source");
        assert_eq!(id.digest, "unrecorded");
        assert_eq!(id.abi, "none");
        println!("ok aien-omega-gpu: unrecorded companion digest is unrecorded with abi none");
    }

    // Case 3: Mandatory fixture with TARGET: positive lock passes and wrong digest triggers exact recorded mismatch.
    {
        let target = env_or("TARGET", "");
        let fixture_source = "1111111111111111111111111111111111111111";
        assert!(!target.is_empty(), "TARGET must not be empty");

        let fixture_dir = temp.join("locked-fixture");
        std::fs::create_dir_all(&fixture_dir).unwrap();
        let mini_manifest = fixture_dir.join("mini.manifest");
        std::fs::write(&mini_manifest, b"abi mini 1\nheader mini.h\nstruct Mini 8 8\nfield Mini val 0 8 u64 1\nconst MINI_OK 0\nfn mini_fn fn(u64) -> i32\n").unwrap();
        let mini_h = fixture_dir.join("mini.h");
        std::fs::write(&mini_h, b"#include <stdint.h>\n#define MINI_OK 0\ntypedef struct { uint64_t val; } Mini;\nint mini_fn(uint64_t);\n").unwrap();
        let mini_c = fixture_dir.join("mini.c");
        std::fs::write(&mini_c, b"#include \"mini.h\"\nint mini_fn(uint64_t x) { return (int)x; }\n").unwrap();
        assert!(Command::new("cc").current_dir(&fixture_dir).args(["-c", "-o", "mini.o", "mini.c"]).status().unwrap().success());
        let archive = fixture_dir.join("libmini.a");
        let _ = std::fs::remove_file(&archive);
        assert!(Command::new("ar").current_dir(&fixture_dir).args(["rcs", "libmini.a", "mini.o"]).status().unwrap().success());

        let actual_sha = identity::sha256_file(&archive).expect("sha256 of libmini.a");
        let cc = env_or("CC", "cc");
        let ccv = identity::cc_version(&cc);

        // Positive check: correct lock passes and reports locked
        let good_lock = fixture_dir.join("good.lock");
        std::fs::write(&good_lock, format!("libmini.a {fixture_source} {target} {actual_sha} {ccv}\n")).unwrap();
        let good_id = qualify_built_with_includes("test-mini", &mini_manifest, &good_lock, &fixture_dir, &archive, fixture_source, true, vec![]);
        assert_eq!(good_id.digest, "locked");
        assert_eq!(good_id.abi, "pass");

        // Negative check: wrong sha in lock must panic with exact recorded sha256 mismatch wording
        let wrong_lock = fixture_dir.join("wrong.lock");
        let fake_sha = "0000000000000000000000000000000000000000000000000000000000000000";
        std::fs::write(&wrong_lock, format!("libmini.a {fixture_source} {target} {fake_sha} {ccv}\n")).unwrap();

        let res = std::panic::catch_unwind(|| {
            qualify_built_with_includes("test-mini", &mini_manifest, &wrong_lock, &fixture_dir, &archive, fixture_source, true, vec![]);
        });
        let err = res.expect_err("qualify_built_with_includes must panic when recorded sha256 mismatches");
        let msg = panic_message(err);
        let expected_snippet = format!("has sha256 {actual_sha}, native-artifacts.lock expects {fake_sha}");
        assert!(
            msg.contains(&expected_snippet),
            "panic message must specify exact recorded sha256 mismatch, expected '{expected_snippet}', got: '{msg}'"
        );
        println!("ok aien-omega-gpu: positive lock passes and wrong digest triggers exact recorded mismatch");
    }

    // Case 4: Optional test with REAL canonical stage when AIEN_CANONICAL_STAGE is supplied.
    if let Ok(stage_str) = std::env::var("AIEN_CANONICAL_STAGE") {
        if !stage_str.is_empty() {
            let orig_stage = PathBuf::from(&stage_str);
            assert!(orig_stage.is_dir(), "AIEN_CANONICAL_STAGE must name an existing real stage");
                let omega_source = PathBuf::from(std::env::var("AIEN_OMEGA_AUTH_DIR").unwrap_or_default());
                let cudapp_lock = if omega_source.join("cudapp.lock").is_file() {
                    omega_source.join("cudapp.lock")
                } else {
                    orig_stage.join("pin")
                };
                let canonical_pin = std::fs::read_to_string(&cudapp_lock)
                    .unwrap_or_else(|_| std::fs::read_to_string(orig_stage.join("observed")).unwrap())
                    .trim()
                    .to_string();

                let orig_canonical_source = orig_stage.join("source");
                let orig_archive = orig_stage.join("build/libcudapp.a");
                let includes: Vec<PathBuf> = [
                    "src", "m16", "nvrm",
                    "third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc",
                    "third_party/nvidia-open-580.173.02/kernel-open/common/inc",
                    "third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm",
                    "third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include",
                ]
                .iter()
                .map(|p| orig_canonical_source.join(p))
                .collect();

                // 1. Original good real stage qualifies first
                let good_id = qualify_built_with_includes(
                    "aien-omega-gpu",
                    &manifest,
                    &artifacts_lock,
                    &orig_canonical_source,
                    &orig_archive,
                    &canonical_pin,
                    true,
                    includes.clone(),
                );
                assert_eq!(good_id.digest, "locked", "original real stage must have digest locked");
                assert_eq!(good_id.abi, "pass", "original real stage must have abi pass");
                println!("ok aien-omega-gpu: original good real stage qualifies");

                // 2. Copy private stage to temp
                let priv_stage = temp.join("priv-stage");
                let _ = std::fs::remove_dir_all(&priv_stage);
                std::fs::create_dir_all(&priv_stage).unwrap();
                let status = Command::new("cp")
                    .args(["-a"])
                    .arg(format!("{}/.", orig_stage.display()))
                    .arg(&priv_stage)
                    .status()
                    .expect("copy private stage");
                assert!(status.success(), "copy private stage failed");

                // 3. Append unused symbol object keeping real ABI
                let mut_c = priv_stage.join("extra_unused.c");
                std::fs::write(&mut_c, b"int unused_dummy_extra_symbol(void) { return 0; }\n").unwrap();
                assert!(Command::new("cc")
                    .current_dir(&priv_stage)
                    .args(["-c", "-o", "extra_unused.o", "extra_unused.c"])
                    .status()
                    .unwrap()
                    .success());
                let priv_archive = priv_stage.join("build/libcudapp.a");
                assert!(Command::new("ar")
                    .current_dir(&priv_stage)
                    .args(["q", "build/libcudapp.a", "extra_unused.o"])
                    .status()
                    .unwrap()
                    .success());

                // 4. Update all 4 checksum files to private paths
                let priv_source_tar = priv_stage.join("source.tar");
                let priv_pincheck = priv_stage.join("pincheck");
                let sha_source = identity::sha256_file(&priv_source_tar).expect("sha source.tar");
                let sha_guard = identity::sha256_file(&priv_pincheck).expect("sha pincheck");
                let sha_archive = identity::sha256_file(&priv_archive).expect("sha archive");

                let line_source = format!("{sha_source}  {}\n", priv_source_tar.display());
                let line_guard = format!("{sha_guard}  {}\n", priv_pincheck.display());
                let line_archive = format!("{sha_archive}  {}\n", priv_archive.display());

                std::fs::write(priv_stage.join("source.sha256"), &line_source).unwrap();
                std::fs::write(priv_stage.join("guard.sha256"), &line_guard).unwrap();
                std::fs::write(priv_stage.join("archive.sha256"), &line_archive).unwrap();
                std::fs::write(priv_stage.join("hashes"), format!("{line_source}{line_guard}{line_archive}")).unwrap();

                // 5. canonical stage --verify MUST pass (consistency)
                if omega_source.join("tools/cudapp-stage.sh").is_file() {
                    let mut cmd = Command::new("sh");
                    cmd.arg(omega_source.join("tools/cudapp-stage.sh"))
                        .arg("--verify")
                        .arg(&cudapp_lock)
                        .arg(&priv_stage);
                    if let Ok(omc) = std::env::var("OMEGA_OMC") {
                        cmd.env("OMEGA_OMC", omc);
                    }
                    if let Ok(bridge) = std::env::var("OMEGA_BRIDGE") {
                        cmd.env("OMEGA_BRIDGE", bridge);
                    }
                    let st = cmd.status().expect("run cudapp-stage.sh --verify");
                    assert!(st.success(), "mutated private stage with consistent checksums must pass stage --verify");
                    println!("ok aien-omega-gpu: mutated private stage passes stage --verify");
                }

                // 6. qualify_built_with_includes using REAL Core manifest + native-artifacts.lock MUST recorded hash mismatch
                let priv_canonical_source = priv_stage.join("source");
                let priv_includes: Vec<PathBuf> = [
                    "src", "m16", "nvrm",
                    "third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc",
                    "third_party/nvidia-open-580.173.02/kernel-open/common/inc",
                    "third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm",
                    "third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include",
                ]
                .iter()
                .map(|p| priv_canonical_source.join(p))
                .collect();

                let res = std::panic::catch_unwind(|| {
                    qualify_built_with_includes(
                        "aien-omega-gpu",
                        &manifest,
                        &artifacts_lock,
                        &priv_canonical_source,
                        &priv_archive,
                        &canonical_pin,
                        true,
                        priv_includes,
                    );
                });
                let err = res.expect_err("qualify_built_with_includes must panic on mutated archive even with consistent stage checksums");
                let msg = panic_message(err);
                assert!(
                    msg.contains("native-artifacts.lock expects"),
                    "panic message must specify recorded mismatch against native-artifacts.lock, got: {msg}"
                );
                assert!(
                    msg.contains(&format!("has sha256 {sha_archive}")),
                    "panic message must name the actual mutated sha256 {sha_archive}, got: {msg}"
                );
                println!("ok aien-omega-gpu: real stage mutated archive rejected by native-artifacts.lock");
        }
    }
}
EOF

    if AIEN_ROOT="$ROOT" AIEN_WORK="$WORK" TARGET="$HOST_TARGET" HOST="$HOST_TARGET" OUT_DIR="$WORK/probe-output" \
        AIEN_CANONICAL_STAGE="$STAGE_INPUT" AIEN_OMEGA_AUTH_DIR="$OMEGA_INPUT" \
        OMEGA_OMC="$OMC_INPUT" OMEGA_BRIDGE="$BRIDGE_INPUT" \
        cargo run -q --manifest-path "$probe_dir/Cargo.toml" >"$WORK/probe.log" 2>&1; then
        pass "$crate: two-archive negative controls (shadow, unrecorded companion, recorded mismatch)"
        if [ -n "$STAGE_INPUT" ]; then
            pass "$crate: real canonical stage qualifies; forged consistent stage refused by independent archive lock"
        fi
    else
        fail "$crate: two-archive negative controls"
        cat "$WORK/probe.log"
    fi
}
check_gpu
if [ "$fails" -ne 0 ]; then
    echo "native identity: $fails case(s) failed"
    exit 1
fi
echo "native identity: all cases behave"
