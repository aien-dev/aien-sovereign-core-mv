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

# Report whether the measured AIEN source tree is clean and who holds the quiet flag.
S=~/workspace/headstart-exp/aien-work/src
cd "$S" || exit 1
echo "head: $(git log --oneline -1)"
echo "dirty files: $(git status --short | wc -l)"
git status --short | head -5
quietlock check; echo "quietlock check rc=$?"
ls -la ~/workspace/.spark-quiet* 2>/dev/null | head -3
