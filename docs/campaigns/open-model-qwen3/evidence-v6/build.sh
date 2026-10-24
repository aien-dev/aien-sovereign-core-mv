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

# OPEN-MODEL-QWEN3 v6 build (QUALIFICATION-v6.md Section 6): clean clone of sovereign-core at the frozen commit,
# pinned omega/physics/aienos via scripts/pins.sh --gpu, release build of aien-cli and the two campaign examples.
set -u
SC_COMMIT=${1:?sovereign-core commit}
B=/home/drakestapleton/workspace/oq3-v6-build-${SC_COMMIT:0:7}
[ ! -e "$B" ] || { echo "build: $B exists" >&2; exit 2; }
mkdir -p "$B"
git clone -q https://github.com/aien-dev/aien-sovereign-core "$B/sc" && git -C "$B/sc" checkout -q "$SC_COMMIT" || exit 4
cd "$B/sc"
bash scripts/pins.sh --gpu "$B/pins" > "$B/pins.out" 2>&1; echo "pins rc=$?"
. "$B/pins/env.sh"
export CARGO_TARGET_DIR=$B/target
unset AIEN_FORCE_CPU_STUB AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_SHA AIEN_DEV_FALLBACK
env | grep '^AIEN_' | sort
echo "cargo build -vv --release -p aien-cli"
nice -n 19 cargo build -vv --release -p aien-cli > "$B/build-cli.log" 2>&1; echo "aien-cli rc=$?"
echo "cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge"
nice -n 19 cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge > "$B/build-examples.log" 2>&1; echo "examples rc=$?"
grep -o 'has_omega_[a-z_]*' "$B/build-cli.log" | sort -u
echo "sc $(git -C "$B/sc" rev-parse HEAD) dirty=$(git -C "$B/sc" status --porcelain | wc -l)"
for r in omega physics aienos; do [ -d "$B/pins/$r" ] && echo "$r $(git -C "$B/pins/$r" rev-parse HEAD) dirty=$(git -C "$B/pins/$r" status --porcelain | wc -l)"; done
cat "$B/pins/omega/physics.lock" "$B/pins/omega/aienos.lock" 2>/dev/null
sha256sum "$B/sc/Cargo.lock" "$B/target/release/aien-cli" "$B/target/release/examples/np1_reference" "$B/target/release/examples/np1_edit_merge"
