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

# OPEN-MODEL-QWEN3 v2 frozen build: sovereign-core 905fdfc (main with sc#280), omega 459b461 (= omega.lock),
# physics 6d7cf0d beside omega, aienos lock repo. Real compose + GPU libraries, never the stub.
set -u
cd /home/drakestapleton/workspace/oq3-v2-build/sc
export AIEN_AIENOS_LOCK_REPO=/home/drakestapleton/workspace/aienos-repo AIEN_OMEGA_DIR=/home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/omega-main AIEN_PHYSICS_DIR=/home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/physics CARGO_TARGET_DIR=/home/drakestapleton/workspace/oq3-v2-build/target
unset AIEN_FORCE_CPU_STUB AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_SHA
cargo build -vv --release -p aien-cli > /home/drakestapleton/workspace/oq3-v2-build/build-cli.log 2>&1; echo "aien-cli rc=$?"
cargo build -vv --release -p aien-runtime --example np1_reference --example np1_edit_merge > /home/drakestapleton/workspace/oq3-v2-build/build-examples.log 2>&1; echo "examples rc=$?"
git -C /home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/omega-main rev-parse HEAD; git -C /home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/omega-main status --short | wc -l; git -C /home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/physics rev-parse HEAD; git -C /home/drakestapleton/workspace/oq3-v2-build/sc rev-parse HEAD; git -C /home/drakestapleton/workspace/oq3-v2-build/sc status --short | wc -l
sha256sum /home/drakestapleton/workspace/oq3-v2-build/target/release/aien-cli /home/drakestapleton/workspace/oq3-v2-build/target/release/examples/np1_reference /home/drakestapleton/workspace/oq3-v2-build/target/release/examples/np1_edit_merge
