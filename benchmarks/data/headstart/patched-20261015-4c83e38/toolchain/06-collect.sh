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

# Stage 6: copy the toolchain pin, digests and small upstream-test logs into
# the evidence dir. No upstream source or binaries are copied.
set -euo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
L=$E/logs
OUT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a/benchmarks/data/headstart/patched-20261015-4c83e38
mkdir -p "$OUT/toolchain" "$OUT/upstream-tests"
clean() { LC_ALL=C tr -cd '\11\12\15\40-\176' | sed "s#$HOME#~#g"; }
clean < "$L/03-digests.txt" > "$OUT/toolchain/digests.txt"
{ grep -E 'downloading|Building stage|Build completed|Finished|RUSTC_BUILD_S|CARGO_BUILD_S|BUILD_OK' "$L/02-build.log" | head -40; } | clean > "$OUT/toolchain/build-log-excerpt.txt"
clean < "$L/01-fetch.log" | tail -20 > "$OUT/toolchain/fetch-log-tail.txt"
for f in "$E"/bin/[01]*.sh; do clean < "$f" > "$OUT/toolchain/$(basename "$f")"; done
for f in "$L"/03-*.log "$L"/04-active.log; do
  b=$(basename "$f" .log)
  tail -n 60 "$f" | clean > "$OUT/upstream-tests/$b.tail.txt"
done
clean < "$L/03-results.txt" > "$OUT/upstream-tests/results.txt"
for f in "$L"/07-*.log; do
  [ -e "$f" ] || continue
  tail -n 60 "$f" | clean > "$OUT/upstream-tests/$(basename "$f" .log).tail.txt"
done
for f in "$OUT"/diag-once/*; do
  clean < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
echo COLLECT_OK
