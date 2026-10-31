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

# Release build with machine-local paths removed from the binaries (CAND-0 reproducibility gap G13).
#
#   scripts/repro-build.sh [cargo build arguments]      e.g.  scripts/repro-build.sh -p aien-cli
#
# rustc writes the source path of every dependency into panic messages. Dependencies come from
# $CARGO_HOME/registry/src/..., so without this script the aien-cli digest depends on the home
# directory and on CARGO_HOME of whoever built it. The script maps those prefixes to fixed names
# (--remap-path-prefix) and builds with --release --locked. It does not set AIEN_OMEGA_DIR,
# AIEN_PHYSICS_DIR or CARGO_TARGET_DIR: pass them as for any other build.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cargo_home=${CARGO_HOME:-$HOME/.cargo}
remap="--remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$root=/aien"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$remap"
exec cargo build --release --locked "$@"
