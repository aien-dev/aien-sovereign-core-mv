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

# OPEN-MODEL-QWEN3 v1 campaign (ACCEPTANCE-v1.md): the three NEXT-PHASE-1 v5 tasks,
# one run-campaign.sh launch each, scored by next-phase-1/make-receipt.sh with its v5 rows.
# Shell + jq only. No task is repeated. Run inside the caller's quietlock hold.
#   AIEN_BIN=... AIEN_MODEL_PATH=.../model.safetensors.index.json AIEN_TOKENIZER_PATH=... \
#   AIEN_COMPOSE_MAX_TOKENS=64 AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 \
#   AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
#   run-qwen3.sh RUN_BASE OUT_DIR SC_COMMIT COMPOSE_LIB_COMMIT GPU_ENGINE_COMMIT
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NP1=$(cd "$HERE/../next-phase-1" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?COMPOSE_LIB_COMMIT} OMG=${5:?GPU_ENGINE_COMMIT}
: "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}"
need() { [ "$(printenv "$1")" = "$2" ] || { echo "$1 must be $2 (ACCEPTANCE-v1 Section 3)"; exit 3; }; }
need AIEN_COMPOSE_MAX_TOKENS 64
need AIEN_KV_CONTEXT_TOKENS 4096
need AIEN_REQUIRE_BLACKWELL 1
need AIEN_GB10_QWEN3_DECLARED_ATTEMPT 1
[ -z "${AIEN_FORCE_CPU_STUB:-}" ] || { echo "AIEN_FORCE_CPU_STUB must be unset"; exit 3; }
d=$(dirname "$AIEN_MODEL_PATH")
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "digest mismatch: $1"; exit 3; }; }
chk "$AIEN_MODEL_PATH" d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c
chk "$d/model-00001-of-00003.safetensors" 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6
chk "$d/model-00002-of-00003.safetensors" 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1
chk "$d/model-00003-of-00003.safetensors" 7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d
chk "$AIEN_TOKENIZER_PATH" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
chk "$d/config.json" 5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba
chk "$d/generation_config.json" 835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48
mkdir -p "$BASE" "$OUT"
sha256sum "${AIEN_BIN:?}" > "$BASE/binary.sha256"
verdict=PASS
for id in $(jq -r '.tasks[].id' "$NP1/tasks-v5.json"); do
  goal=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | .goal' "$NP1/tasks-v5.json")
  NP1_GOAL=$goal bash "$NP1/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "task $id driver exit $?"
  if [ -f "$BASE/$id/run.json" ]; then
    TASK_ID=$id SPEC="open-model-qwen3/ACCEPTANCE-v1.md spec_version 1" bash "$NP1/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "OPEN-MODEL-QWEN3 v1 task $id" >"$BASE/$id.receipt.out" 2>&1
    v=$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.* verdict \([A-Z]*\)$/\1/p')
  else
    v=FAIL
  fi
  echo "task $id verdict ${v:-FAIL} ($(head -1 "$BASE/$id.receipt.out" 2>/dev/null))"
  [ "${v:-FAIL}" = PASS ] || verdict=FAIL
done
echo "OPEN-MODEL-QWEN3 v1 campaign verdict (rows only; VERDICT-v1.md is written by the reviewer): $verdict"
