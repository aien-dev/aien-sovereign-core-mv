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

# NEXT-PHASE-1 v7 campaign (ACCEPTANCE-v7.md): run-v6.sh with the v7 wiring of
# ACCEPTANCE-v7 Section 2 and nothing else. Same five launches of tasks-v6.json
# (ACCEPTANCE-v6 Section 2, kept unchanged by ACCEPTANCE-v7 Section 1), same
# goals, max_tokens, T5 seed and CPU reference run; each receipt is built by
# make-receipt.sh with V6_ROWS=rows-v7.jq (the S3 committed-flag fix) and records
# that module and its sha256 as v6_rows_module; the verdict is score-rows.sh
# (scoring-v5) over np1-v7.decl.json. Usage (inside the caller's quietlock hold,
# ACCEPTANCE-v6 Section 8):
#   AIEN_BIN=... AIEN_MODEL_PATH=... AIEN_TOKENIZER_PATH=... \
#   run-v7.sh RUN_BASE OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT REFERENCE_BIN
# Writes RUN_BASE/<launch>/ (run roots), RUN_BASE/R1.reference.json, one
# receipt per launch in OUT_DIR, OUT_DIR/v7-results.jsonl and OUT_DIR/v7-score.json.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?OMEGA_COMPOSE} OMG=${5:?OMEGA_GPU} REFBIN=${6:?REFERENCE_BIN}
TASKS=$HERE/tasks-v6.json
DECL=$HERE/../scoring/declarations/np1-v7.decl.json
SCORER=$HERE/../scoring/score-rows.sh
[ -f "$DECL" ] && [ -x "$SCORER" ] && [ -x "$REFBIN" ] || { echo "run-v7: declaration, scorer or reference binary missing" >&2; exit 2; }
MODEL_DIR=$(dirname "${AIEN_MODEL_PATH:?}")
mkdir -p "$BASE" "$OUT"
RES=$OUT/v7-results.jsonl
: >"$RES"
for id in $(jq -r '.tasks[].id' "$TASKS"); do
  t=$(jq -c --arg id "$id" '.tasks[] | select(.id == $id)' "$TASKS")
  goal=$(jq -r .goal <<<"$t"); max=$(jq -r .max_tokens <<<"$t"); seed=$(jq -r '.seed // empty' <<<"$t")
  seed_dir=; [ -n "$seed" ] && seed_dir=$HERE/$seed
  NP1_GOAL=$goal AIEN_COMPOSE_MAX_TOKENS=$max NP1_SEED_DIR=$seed_dir \
    bash "$HERE/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "launch $id driver exit $? (max_tokens $max${seed:+, seed $seed})"
  ref=
  if [ "$(jq -r .kind <<<"$t")" = identity ] && [ -d "$BASE/$id/ws" ]; then
    ws=$(cd "$BASE/$id/ws" && pwd -P)
    "$REFBIN" "$MODEL_DIR" "$goal" "$ws" "$max" new >"$BASE/$id.reference.json" 2>"$BASE/$id.reference.err"
    echo "launch $id CPU reference exit $?"
    ref=$BASE/$id.reference.json
  fi
  if [ -f "$BASE/$id/run.json" ]; then
    v5env=(); [ "$(jq -r '.destination // empty' <<<"$t")" != "" ] && [ "${id#N}" = "$id" ] && \
      v5env=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v6.md")
    env "${v5env[@]}" V6_ROWS=rows-v7.jq V6_TASK="$id" V6_MAX_TOKENS="$max" V6_REFERENCE="$ref" \
      SPEC="ACCEPTANCE-v7.md spec_version 7" bash "$HERE/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "NEXT-PHASE-1 v7 launch $id" >"$BASE/$id.receipt.out" 2>&1
    rec=$OUT/$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.*receipt \([0-9a-f]*\.json\).*/\1/p')
    if [ -f "$rec" ]; then bash "$HERE/v6-results.sh" "$id" "$rec" >>"$RES"
    else bash "$HERE/v6-results.sh" "$id" - "$DECL" "receipt not written" >>"$RES"; fi
    echo "launch $id receipt $(basename "$rec")"
  else
    bash "$HERE/v6-results.sh" "$id" - "$DECL" "launch failed: no run.json" >>"$RES"   # ACCEPTANCE-v6 Section 7
    echo "launch $id FAILED to launch"
  fi
  jq -r --arg id "$id" 'select(.row | startswith($id + "/")) | select(.verdict != "PASS") | "  \(.verdict) \(.row)"' "$RES"
done
bash "$SCORER" "$DECL" "$RES" >"$OUT/v7-score.json"; rc=$?
echo "NEXT-PHASE-1 v7 campaign verdict (scoring-v5, score-rows.sh exit $rc): $(jq -r .verdict "$OUT/v7-score.json" 2>/dev/null || echo ERROR)"
exit $rc
