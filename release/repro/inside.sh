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

# Runs inside the release/repro/Dockerfile image, started by scripts/repro-container-build.sh (DRA-43).
# WORK holds core/, omega/ and physics/ checkouts; everything the build writes stays under WORK, so two runs from
# two WORK paths differ only in their paths. Mode `remap` builds with the core tree's own scripts/repro-build.sh;
# mode `plain` is the negative control (same build, no path remap).
set -euo pipefail

work=${1:?usage: inside.sh WORK remap|plain PACKAGE}
mode=${2:?usage: inside.sh WORK remap|plain PACKAGE}
pkg=${3:?usage: inside.sh WORK remap|plain PACKAGE}
case "$mode" in remap|plain) ;; *) echo "inside.sh: mode must be remap or plain, got $mode" >&2; exit 2 ;; esac

out=$work/out
mkdir -p "$out" "$work/home" "$work/lib"
export HOME=$work/home CARGO_HOME=$work/cargo-home CARGO_TARGET_DIR=$work/target
export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH=$(git -C "$work/core" log -1 --format=%ct)

# The GPU library, built the way omega's Makefile builds it (its CFLAGS, its physics.lock check). The Makefile
# assigns CFLAGS with ?=, so an inherited CFLAGS would replace its flags: build with CFLAGS unset.
env -u CFLAGS -u CXXFLAGS make -C "$work/omega" -j"${CARGO_BUILD_JOBS:-1}" \
	OUT_DIR="$work/lib" PHYSICS_DIR="$work/physics" "$work/lib/libomega_gpu.a" >"$out/make.log" 2>&1
lib_sha=$(sha256sum "$work/lib/libomega_gpu.a" | cut -d' ' -f1)
cp "$work/lib/libomega_gpu.a" "$out/libomega_gpu.a"

# Link that library the way the shipped engine was linked (crates/aien-omega-gpu/build.rs prebuilt mode).
export AIEN_OMEGA_GPU_LIB=$work/lib/libomega_gpu.a AIEN_OMEGA_GPU_LIB_SHA256=$lib_sha
cd "$work/core"
cargo fetch --locked >"$out/fetch.log" 2>&1
if [ "$mode" = remap ]; then
	bash scripts/repro-build.sh -p "$pkg" >"$out/cargo.log" 2>&1
else
	cargo build --release --locked -p "$pkg" >"$out/cargo.log" 2>&1
fi
cp "$CARGO_TARGET_DIR/release/$pkg" "$out/$pkg"

{
	echo "work=$work"
	echo "mode=$mode"
	echo "core=$(git -C "$work/core" rev-parse HEAD)"
	echo "omega=$(git -C "$work/omega" rev-parse HEAD)"
	echo "physics=$(git -C "$work/physics" rev-parse HEAD)"
	echo "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
	echo "CARGO_HOME=$CARGO_HOME"
	echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
	echo "rustc: $(rustc -vV | tr '\n' ' ')"
	echo "gcc: $(gcc --version | head -n1)"
	echo "ar: $(ar --version | head -n1)"
	echo "ld: $(ld --version | head -n1)"
	echo "make: $(make --version | head -n1)"
	echo "ar members:"
	ar tv "$out/libomega_gpu.a"
	echo "sha256:"
	(cd "$out" && sha256sum libomega_gpu.a "$pkg")
} >"$out/BUILD-INFO.txt"
cat "$out/BUILD-INFO.txt"
