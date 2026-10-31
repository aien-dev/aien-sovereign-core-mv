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

# OPEN-MODEL-QWEN3 v3 frozen build: sovereign-core 8f3e8c8 (main with sc#285, #284, #287), omega 01f6a74 (= omega.lock,
# omega#331), physics 6d7cf0d (physics.lock), aienos b84c0a6 (aienos.lock). Real compose + GPU libraries, never the stub.
set -u
B=/home/drakestapleton/workspace/oq3-v3-build
cd $B/sc
export AIEN_AIENOS_LOCK_REPO=/home/drakestapleton/workspace/aienos-repo AIEN_OMEGA_DIR=$B/omega AIEN_PHYSICS_DIR=$B/physics CARGO_TARGET_DIR=$B/target
unset AIEN_FORCE_CPU_STUB AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_SHA
cargo build -vv --release -p aien-cli > $B/build-cli.log 2>&1; echo "aien-cli rc=$?"
cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge > $B/build-examples.log 2>&1; echo "examples rc=$?"
git -C $B/omega rev-parse HEAD; git -C $B/omega status --short | wc -l; git -C $B/physics rev-parse HEAD; git -C $B/sc rev-parse HEAD; git -C $B/sc status --short | wc -l
sha256sum $B/target/release/aien-cli $B/target/release/examples/np1_reference $B/target/release/examples/np1_edit_merge
