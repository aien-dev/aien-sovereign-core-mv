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

# Pre-run gate on sovereign-core 905fdfc (ACCEPTANCE-v2 Section 2).
set -u
cd /home/drakestapleton/workspace/oq3-v2-build/sc
export AIEN_AIENOS_LOCK_REPO=/home/drakestapleton/workspace/aienos-repo AIEN_OMEGA_DIR=/home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/omega-main AIEN_PHYSICS_DIR=/home/drakestapleton/workspace/investigations/2026-10-07-qwen3-decode-matmul/physics CARGO_TARGET_DIR=/home/drakestapleton/workspace/oq3-v2-build/target-gate
cargo fmt --all --check > /home/drakestapleton/workspace/oq3-v2-build/gate-fmt.log 2>&1; echo "fmt rc=$?"
cargo clippy --workspace --all-targets -- -D warnings > /home/drakestapleton/workspace/oq3-v2-build/gate-clippy.log 2>&1; echo "clippy rc=$?"
AIEN_FORCE_CPU_STUB=1 cargo test -p aien-runtime -p aien-cli > /home/drakestapleton/workspace/oq3-v2-build/gate-test.log 2>&1; echo "test rc=$?"
grep -E '^test result' /home/drakestapleton/workspace/oq3-v2-build/gate-test.log | awk '{p+=$4; f+=$6} END {print "tests passed="p" failed="f}'
