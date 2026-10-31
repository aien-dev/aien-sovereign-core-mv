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

# Path-independent build of the serving engine in a pinned container (DRA-43).
#
#   scripts/repro-container-build.sh [--plain] WORK
#
# WORK is a new absolute directory. The script checks out core, omega and physics under WORK, starts the
# release/repro/Dockerfile image with WORK mounted at the same path, builds libomega_gpu.a with omega's Makefile and
# links it into the engine the way the shipped engine is linked (AIEN_OMEGA_GPU_LIB + AIEN_OMEGA_GPU_LIB_SHA256).
# CARGO_HOME, CARGO_TARGET_DIR and HOME live under WORK, so two runs from two WORK paths differ only in their paths:
# equal sha256 in WORK/out/BUILD-INFO.txt means the build does not depend on where it ran. --plain builds without
# scripts/repro-build.sh (no path remap), the negative control. CPU only: no --gpus, no device mounts.
#
# Environment (40-hex commits; defaults in brackets):
#   CORE_REV     core commit to build [HEAD of this checkout]
#   OMEGA_REV    omega commit for libomega_gpu.a [omega.lock at CORE_REV]
#   PHYSICS_REV  physics commit [physics.lock at OMEGA_REV]
#   EXPECT_LIB_SHA256, EXPECT_ENGINE_SHA256   fail unless the built file has this digest [unchecked]
#   PACKAGE      cargo package [aien-serve-http]
#   CORE_URL OMEGA_URL PHYSICS_URL   git remotes [github.com/aien-dev/{aien-sovereign-core,omega,physics}-mv]
#   REPRO_CPUS   docker --cpuset-cpus [2-4]   REPRO_JOBS  build jobs [3]
#   REPRO_MEMORY docker --memory and --memory-swap, e.g. 16g [no limit]
#
# The SI alpha-20261029.1 engine (sovereign-intelligence installer/build.sh):
#   CORE_REV=59b05a71a33e4a2f1324db2f2b6407ee82b8c2c3 OMEGA_REV=6de999da86efc6edd9c2554a87ee31f7fa9215ac \
#   EXPECT_LIB_SHA256=110295ef23dc7561eea371636347c005835f37b82c22f92401aa8a887a3ca990 \
#   scripts/repro-container-build.sh /abs/new/dir
set -euo pipefail

usage() { echo "usage: $0 [--plain] WORK" >&2; exit 2; }
mode=remap
if [ "${1:-}" = --plain ]; then mode=plain; shift; fi
[ $# -eq 1 ] || usage
work=$1
case "$work" in /*) ;; *) echo "$0: WORK must be an absolute path" >&2; exit 2 ;; esac
if [ -e "$work" ] && [ -n "$(ls -A "$work")" ]; then echo "$0: $work exists and is not empty" >&2; exit 2; fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
pkg=${PACKAGE:-aien-serve-http}
core_url=${CORE_URL:-https://github.com/aien-dev/aien-sovereign-core-mv.git}
omega_url=${OMEGA_URL:-https://github.com/aien-dev/omega-mv.git}
physics_url=${PHYSICS_URL:-https://github.com/aien-dev/physics-mv.git}
core_rev=${CORE_REV:-$(git -C "$here" rev-parse HEAD)}

is_sha() { [[ $1 =~ ^[0-9a-f]{40}$ ]]; }
# fetch_rev DIR URL REV: a one-commit checkout of REV at DIR.
fetch_rev() {
	is_sha "$3" || { echo "$0: $3 is not a 40-hex commit" >&2; exit 2; }
	git init -q "$1"
	git -C "$1" -c uploadpack.allowAnySHA1InWant=true fetch -q --depth 1 "$2" "$3"
	git -C "$1" checkout -q --detach FETCH_HEAD
	[ "$(git -C "$1" rev-parse HEAD)" = "$3" ] || { echo "$0: $1 is not at $3" >&2; exit 1; }
}

mkdir -p "$work"
fetch_rev "$work/core" "$core_url" "$core_rev"
omega_rev=${OMEGA_REV:-$(tr -d '[:space:]' <"$work/core/omega.lock")}
fetch_rev "$work/omega" "$omega_url" "$omega_rev"
physics_rev=${PHYSICS_REV:-$(tr -d '[:space:]' <"$work/omega/physics.lock")}
fetch_rev "$work/physics" "$physics_url" "$physics_rev"

image=aien-repro-build:$(sha256sum "$here/release/repro/Dockerfile" | cut -c1-12)
if ! docker image inspect "$image" >/dev/null 2>&1; then
	docker build -q -t "$image" "$here/release/repro" >/dev/null
fi

mem=()
[ -n "${REPRO_MEMORY:-}" ] && mem=(--memory "$REPRO_MEMORY" --memory-swap "$REPRO_MEMORY")
docker run --rm --cpuset-cpus "${REPRO_CPUS:-2-4}" "${mem[@]}" --user "$(id -u):$(id -g)" \
	-e CARGO_BUILD_JOBS="${REPRO_JOBS:-3}" \
	-v "$work:$work" -v "$here/release/repro/inside.sh:/repro/inside.sh:ro" \
	"$image" nice -n 19 ionice -c3 bash /repro/inside.sh "$work" "$mode" "$pkg"

check() { # check FILE EXPECTED
	[ -z "$2" ] && return 0
	got=$(sha256sum "$1" | cut -d' ' -f1)
	[ "$got" = "$2" ] || { echo "$0: $1 sha256 $got, expected $2" >&2; exit 1; }
}
check "$work/out/libomega_gpu.a" "${EXPECT_LIB_SHA256:-}"
check "$work/out/$pkg" "${EXPECT_ENGINE_SHA256:-}"
