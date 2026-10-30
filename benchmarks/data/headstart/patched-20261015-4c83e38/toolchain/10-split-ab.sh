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

# The first ab stage was stopped after round 1 (host contaminated, before the
# per-sample hold existed). Move its samples out of samples.jsonl into a
# separate raw file so the summary only uses the full held rerun.
set -euo pipefail
OUT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a/benchmarks/data/headstart/patched-20261015-4c83e38
S="$OUT/samples.jsonl"
X="$OUT/samples-ab-run1-interrupted-unheld.jsonl"
[ -e "$X" ] && { echo "already split: $X"; exit 1; }
jq -c 'select(.kind == "ab")' "$S" > "$X"
jq -c 'select(.kind != "ab")' "$S" > "$S.tmp"
mv "$S.tmp" "$S"
echo "moved $(wc -l < "$X") ab samples; kept $(wc -l < "$S")"
