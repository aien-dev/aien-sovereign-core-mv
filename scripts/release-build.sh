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

# The one release build recipe (the CAND-3 [build] recipe, per package). Used by .github/workflows/release.yml
# and the thing to run by hand to reproduce a candidate digest. See docs/release/BUILD_RECIPE.md.
#   scripts/release-build.sh [extra cargo args, e.g. --offline]
# Needs a pinned toolchain (rust-toolchain.toml), `cargo fetch --locked` done into the CARGO_HOME in use,
# and for the native engine AIEN_OMEGA_DIR and AIEN_PHYSICS_DIR at omega.lock / omega's physics.lock.
# Cargo unifies features across every -p in one command, so a digest depends on the group it was built in
# (docs/release/F1-binary-digest-difference.md). The groups are therefore fixed: the packaged helpers as one
# group first, then aien-cli alone and last, each through scripts/repro-build.sh (--locked, path remap).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/repro-build.sh" -p spark-cockpit-rs -p spark-inquisitor -p cortex-encoder-rs -p cortex-rs -p spark-supervisor -p spark-debugger "$@"
"$here/repro-build.sh" -p aien-cli "$@"
