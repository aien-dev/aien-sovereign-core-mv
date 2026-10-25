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

# OPEN-MODEL-QWEN3 v5 freeze-candidate build (DRAFT, NOT FROZEN): sovereign-core 30258fc (main, holds sc#353, sc#354, sc#357, sc#358, sc#359 and sc#360),
# omega 6c6180c (= omega.lock at 30258fc), physics 6d7cf0d (physics.lock), aienos b84c0a6 (aienos.lock). Clean clones,
# release, real compose + GPU libraries, never the stub, without the dev-fallback feature. Derived from evidence-v4/build.sh.
set -u
B=/home/drakestapleton/workspace/oq3-v5-build-30258fc
cd $B/sc
export AIEN_AIENOS_LOCK_REPO=/home/drakestapleton/workspace/aienos-repo AIEN_OMEGA_DIR=$B/omega AIEN_PHYSICS_DIR=$B/physics CARGO_TARGET_DIR=$B/target
unset AIEN_FORCE_CPU_STUB AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_SHA AIEN_DEV_FALLBACK
echo "cargo build -vv --release -p aien-cli"
cargo build -vv --release -p aien-cli > $B/build-cli.log 2>&1; echo "aien-cli rc=$?"
echo "cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge"
cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge > $B/build-examples.log 2>&1; echo "examples rc=$?"
git -C $B/omega rev-parse HEAD; git -C $B/omega status --short | wc -l; git -C $B/physics rev-parse HEAD; git -C $B/sc rev-parse HEAD; git -C $B/sc status --short | wc -l
sha256sum $B/sc/Cargo.lock $B/target/release/aien-cli $B/target/release/examples/np1_reference $B/target/release/examples/np1_edit_merge
