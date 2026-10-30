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

# pins.sh: materialize the three pinned checkouts a native (no CUDA) build needs,
# exactly as CI does (ci.yml "Linked composition tests"), so a clean clone plus one
# command yields the real omega libraries instead of the CPU stubs (sovereign-core #92).
#
#   omega    at omega.lock (this repository)
#   physics  at omega's physics.lock
#   aienos   at omega's aienos.lock (authority source for the composition library)
#
# Usage: scripts/pins.sh [--gpu] [--check] [DIR]
#   DIR      where the checkouts live (default target/pins). Existing clones are reused;
#            the network is touched only when a clone is missing or not at its pin.
#   --gpu    env.sh also sets AIEN_OMEGA_DIR, which links the GB10 engine (aien-omega-gpu).
#            Without it only the composition library (CPU) is linked, as the CI gate does.
#   --check  verify only: every clone present and at its pin, no network, no writes.
# Writes DIR/env.sh. Then:  . target/pins/env.sh && cargo build --release
# Env: AIEN_PINS_REMOTE_BASE (default https://github.com/aien-dev; a local mirror directory
#      works too), AIEN_PINS_REMOTE_SUFFIX (appended to each repository name; default "-mv"
#      for the default GitHub base, where the repositories are omega-mv, physics-mv and
#      aienos-mv, and "" when AIEN_PINS_REMOTE_BASE is set), AIEN_PINS_ROOT (repository root
#      override, used by scripts/test-pins.sh).
# Exit: 0 ok, 2 usage, 3 a checkout is not at its pin (or missing under --check), 4 git failed.
set -euo pipefail
ROOT="${AIEN_PINS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -n "${AIEN_PINS_REMOTE_BASE:-}" ]; then
    BASE="$AIEN_PINS_REMOTE_BASE"; SUFFIX="${AIEN_PINS_REMOTE_SUFFIX:-}"
else
    BASE="https://github.com/aien-dev"; SUFFIX="${AIEN_PINS_REMOTE_SUFFIX--mv}"
fi
GPU=0; CHECK=0; DIR=""
for a in "$@"; do
    case "$a" in
        --gpu) GPU=1 ;;
        --check) CHECK=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        -*) echo "pins.sh: unknown flag $a" >&2; exit 2 ;;
        *) [ -z "$DIR" ] || { echo "pins.sh: one DIR only" >&2; exit 2; }; DIR="$a" ;;
    esac
done
DIR="${DIR:-$ROOT/target/pins}"
mkdir -p "$DIR" 2>/dev/null || true; DIR="$(cd "$DIR" 2>/dev/null && pwd || echo "$DIR")"
die() { echo "pins.sh: $*" >&2; exit 3; }
sha_of() { tr -d '[:space:]' < "$1"; }
is_sha() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }
head_of() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
# place NAME SHA: make DIR/NAME a clone at SHA (detached), fetching only when needed.
place() {
    local name="$1" sha="$2"
    local d="$DIR/$name"
    is_sha "$sha" || die "$name pin is not a full 40-hex sha: '$sha'"
    if [ "$(head_of "$d")" = "$sha" ]; then echo "pins: $name at $sha (present)"; return 0; fi
    [ "$CHECK" -eq 0 ] || die "$name is not at $sha under --check (have '$(head_of "$d")', dir $d)"
    mkdir -p "$DIR"
    if [ ! -d "$d/.git" ]; then
        echo "pins: cloning $name from $BASE/$name$SUFFIX"
        git clone -q "$BASE/$name$SUFFIX" "$d" || { echo "pins.sh: clone of $name ($BASE/$name$SUFFIX) failed" >&2; exit 4; }
    fi
    if ! git -C "$d" cat-file -e "$sha^{commit}" 2>/dev/null; then
        git -C "$d" fetch -q origin "$sha" || git -C "$d" fetch -q origin || { echo "pins.sh: fetch of $name failed" >&2; exit 4; }
    fi
    git -C "$d" checkout -q --detach "$sha" || { echo "pins.sh: checkout of $name@$sha failed" >&2; exit 4; }
    [ "$(head_of "$d")" = "$sha" ] || die "$name HEAD is not $sha after checkout"
    echo "pins: $name at $sha"
}
[ -f "$ROOT/omega.lock" ] || die "no omega.lock under $ROOT"
place omega "$(sha_of "$ROOT/omega.lock")"
[ -f "$DIR/omega/physics.lock" ] || die "omega checkout has no physics.lock"
[ -f "$DIR/omega/aienos.lock" ] || die "omega checkout has no aienos.lock"
place physics "$(sha_of "$DIR/omega/physics.lock")"
place aienos "$(sha_of "$DIR/omega/aienos.lock")"
if [ "$CHECK" -eq 0 ]; then
    {
        echo "# written by scripts/pins.sh; source it before cargo build/test"
        echo "export AIEN_OMEGA_COMPOSE_DIR='$DIR/omega'"
        echo "export AIEN_PHYSICS_DIR='$DIR/physics'"
        echo "export AIEN_AIENOS_LOCK_REPO='$DIR/aienos'"
        if [ "$GPU" -eq 1 ]; then echo "export AIEN_OMEGA_DIR='$DIR/omega'"; else echo "unset AIEN_OMEGA_DIR"; fi
        echo "unset AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_SHA AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK"
        echo "export AIEN_REQUIRE_NATIVE_LIBS=1"
    } > "$DIR/env.sh"
    echo "pins: wrote $DIR/env.sh ($([ "$GPU" -eq 1 ] && echo 'GPU engine + composition' || echo 'composition only, CPU'))"
    echo "next:  . '$DIR/env.sh' && cargo build --release"
fi
