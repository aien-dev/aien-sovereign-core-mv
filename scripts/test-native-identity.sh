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
# Runs without omega or CUDApp: every case below must be decided by build.rs before any make.
#   - a stub together with AIEN_REQUIRE_NATIVE_LIBS=1 is refused
#   - AIEN_REQUIRE_NATIVE_LIBS=1 with no library source is refused
#   - a prebuilt archive with no expected sha256, a wrong one, or a truncated archive is refused
#   - a prebuilt archive that misses one ABI function is refused
#   - a checkout at a commit other than its lock (omega.lock for compose, cudapp.lock for gpu) is refused
#   - a correct prebuilt archive builds, and its identity says "prebuilt" (never qualifies)
# Exit 0 when every case behaves, 1 otherwise.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export CARGO_TARGET_DIR="${AIEN_IDENTITY_TARGET_DIR:-$ROOT/target/native-identity}"
for v in $(env | sed -n 's/^\(AIEN_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done

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

# A git checkout at a commit that is neither omega.lock nor cudapp.lock.
WRONG="$WORK/wrong-checkout"
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
    expect_refused "$crate" "checkout at the wrong commit" "is at '" \
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
check_crate aien-omega-gpu crates/aien-omega-gpu/abi/omega_gpu.manifest \
    AIEN_OMEGA_GPU_LIB AIEN_CUDAPP_DIR omega_gpu_matmul_f32 AIEN_OMEGA_GPU

if [ "$fails" -ne 0 ]; then
    echo "native identity: $fails case(s) failed"
    exit 1
fi
echo "native identity: all cases behave"
