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

# OPEN-MODEL-SMOLLM2 v1 campaign (ACCEPTANCE-v1.md): the three NEXT-PHASE-1 v5 tasks,
# one run-campaign.sh launch each, scored by next-phase-1/make-receipt.sh with its v5 rows.
# Shell + jq only. No task is repeated. Run inside the caller's quietlock hold.
#   AIEN_BIN=... AIEN_MODEL_PATH=... AIEN_TOKENIZER_PATH=... AIEN_COMPOSE_MAX_TOKENS=64 \
#   run-smollm2.sh RUN_BASE OUT_DIR SC_COMMIT COMPOSE_LIB_COMMIT GPU_ENGINE_COMMIT
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NP1=$(cd "$HERE/../next-phase-1" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?COMPOSE_LIB_COMMIT} OMG=${5:?GPU_ENGINE_COMMIT}
: "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}"
[ "${AIEN_COMPOSE_MAX_TOKENS:-}" = 64 ] || { echo "AIEN_COMPOSE_MAX_TOKENS must be 64 (ACCEPTANCE-v1 Section 2)"; exit 3; }
d=$(dirname "$AIEN_MODEL_PATH")
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "digest mismatch: $1"; exit 3; }; }
chk "$AIEN_MODEL_PATH" f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8
chk "$AIEN_TOKENIZER_PATH" 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
chk "$d/config.json" 994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea
chk "$d/generation_config.json" 87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013
mkdir -p "$BASE" "$OUT"
verdict=PASS
for id in $(jq -r '.tasks[].id' "$NP1/tasks-v5.json"); do
  goal=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | .goal' "$NP1/tasks-v5.json")
  NP1_GOAL=$goal bash "$NP1/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "task $id driver exit $?"
  if [ -f "$BASE/$id/run.json" ]; then
    TASK_ID=$id SPEC="open-model-smollm2/ACCEPTANCE-v1.md spec_version 1" bash "$NP1/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "OPEN-MODEL-SMOLLM2 v1 task $id" >"$BASE/$id.receipt.out" 2>&1
    v=$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.* verdict \([A-Z]*\)$/\1/p')
  else
    v=FAIL
  fi
  echo "task $id verdict ${v:-FAIL} ($(head -1 "$BASE/$id.receipt.out" 2>/dev/null))"
  [ "${v:-FAIL}" = PASS ] || verdict=FAIL
done
echo "OPEN-MODEL-SMOLLM2 v1 campaign verdict (rows only; VERDICT-v1.md is written by the reviewer): $verdict"
