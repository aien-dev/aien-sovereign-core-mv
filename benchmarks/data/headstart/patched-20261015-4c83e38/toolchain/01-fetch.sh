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

# Stage 1: fetch pinned submodules and apply headstart patches (mirrors upstream scripts/setup.sh).
set -euo pipefail
H=~/workspace/headstart-exp/headstart
cd "$H"
git submodule update --init --depth 1 rustc cargo
echo "rustc HEAD $(cd rustc && git rev-parse HEAD)"
echo "cargo HEAD $(cd cargo && git rev-parse HEAD)"
for p in patches/rustc/*.patch; do echo "applying $p"; (cd rustc && git apply "$H/$p"); done
for p in patches/cargo/*.patch; do echo "applying $p"; (cd cargo && git apply "$H/$p"); done
cp config/bootstrap.toml rustc/bootstrap.toml
(cd rustc && git diff --stat | tail -1)
(cd cargo && git diff --stat | tail -1)
echo FETCH_OK
