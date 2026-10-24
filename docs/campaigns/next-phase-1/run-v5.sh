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

# NEXT-PHASE-1 v5 campaign: the three tasks of tasks-v5.json (ACCEPTANCE-v5
# Section 2), one run-campaign.sh launch each, in order, each scored by
# make-receipt.sh with its v5 rows. Shell + jq only. No task is repeated.
# Usage (inside the caller's quietlock hold, ACCEPTANCE-v5 Section 6):
#   AIEN_BIN=... AIEN_MODEL_PATH=... AIEN_TOKENIZER_PATH=... AIEN_COMPOSE_MAX_TOKENS=<frozen> \
#   run-v5.sh RUN_BASE OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT
# Writes RUN_BASE/<task>/ (run roots) and one receipt per task in OUT_DIR.
# Prints one line per task and the campaign verdict (PASS only if all PASS).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?OMEGA_COMPOSE} OMG=${5:?OMEGA_GPU}
mkdir -p "$BASE" "$OUT"
verdict=PASS
for id in $(jq -r '.tasks[].id' "$HERE/tasks-v5.json"); do
  goal=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | .goal' "$HERE/tasks-v5.json")
  NP1_GOAL=$goal bash "$HERE/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "task $id driver exit $?"
  if [ -f "$BASE/$id/run.json" ]; then
    TASK_ID=$id SPEC="ACCEPTANCE-v5.md spec_version 5" bash "$HERE/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "NEXT-PHASE-1 v5 task $id" >"$BASE/$id.receipt.out" 2>&1
    v=$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.* verdict \([A-Z]*\)$/\1/p')
  else
    v=FAIL   # the launch failed (ACCEPTANCE-v5 Section 4)
  fi
  echo "task $id verdict ${v:-FAIL} ($(head -1 "$BASE/$id.receipt.out" 2>/dev/null))"
  [ "${v:-FAIL}" = PASS ] || verdict=FAIL
done
echo "NEXT-PHASE-1 v5 campaign verdict (rows only; VERDICT-v5.md is written by the reviewer): $verdict"
