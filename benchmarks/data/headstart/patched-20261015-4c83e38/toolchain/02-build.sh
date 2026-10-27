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

# Stage 2: build the patched stage-1 rustc and patched cargo, capped at 12 jobs, niced.
# Mirrors upstream scripts/setup.sh build lines; only -j and nice are added.
set -euo pipefail
H=~/workspace/headstart-exp/headstart
J=${J:-12}
cd "$H/rustc"
t0=$(date +%s)
nice -n 10 ./x build --stage 1 compiler library -j "$J"
t1=$(date +%s)
echo "RUSTC_BUILD_S $((t1 - t0))"
cd "$H/cargo"
nice -n 10 cargo +1.98.0 build --release -j "$J"
t2=$(date +%s)
echo "CARGO_BUILD_S $((t2 - t1))"
ls -la "$H/rustc/build/host/stage1/bin/rustc" "$H/cargo/target/release/cargo"
echo BUILD_OK
