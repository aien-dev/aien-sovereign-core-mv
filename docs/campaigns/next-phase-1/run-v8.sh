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

# NEXT-PHASE-1 v8 campaign (ACCEPTANCE-v8.md): run-v7.sh with the v8 wiring of
# ACCEPTANCE-v8 Section 6 and nothing else. The seven launches of tasks-v8.json
# (T4 T5 T6 T7 N1 N2 R1, ACCEPTANCE-v8 Section 3), same
# goals, max_tokens, T5 seed and CPU reference run; each receipt is built by
# make-receipt.sh with V6_ROWS=rows-v8.jq (row <id>-A, V8_MERGE) and records
# that module and its sha256 as v6_rows_module; the verdict is score-rows.sh
# (scoring-v5) over np1-v8.decl.json. Usage (inside the caller's quietlock hold,
# ACCEPTANCE-v6 Section 8):
#   AIEN_BIN=... AIEN_MODEL_PATH=... AIEN_TOKENIZER_PATH=... \
#   run-v8.sh RUN_BASE OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT REFERENCE_BIN MERGE_BIN
# MERGE_BIN is the built np1_edit_merge example; for an edit launch it re-derives the proposal
# from the accepted reply and the pre-seed and its JSON is passed to make-receipt.sh as V8_MERGE.
# Writes RUN_BASE/<launch>/ (run roots), RUN_BASE/R1.reference.json, one
# receipt per launch in OUT_DIR, OUT_DIR/v8-results.jsonl and OUT_DIR/v8-score.json.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?OMEGA_COMPOSE} OMG=${5:?OMEGA_GPU} REFBIN=${6:?REFERENCE_BIN} MERGEBIN=${7:?MERGE_BIN}
# ACCEPTANCE-v8 Section 6 (B1): the run path is pinned, so every prompt's "Authorized workspace" text
# is known before the run, and the run is fresh (one run, nothing reused).
PINNED_BASE=/home/drakestapleton/workspace/np1-v8-runs
[ "$BASE" = "$PINNED_BASE" ] || { echo "run-v8: RUN_BASE must be $PINNED_BASE" >&2; exit 2; }
[ ! -e "$BASE" ] || { echo "run-v8: $BASE already exists (a v8 run is fresh and happens once)" >&2; exit 2; }
TASKS=$HERE/tasks-v8.json
DECL=$HERE/../scoring/declarations/np1-v8.decl.json
SCORER=$HERE/../scoring/score-rows.sh
[ -f "$DECL" ] && [ -x "$SCORER" ] && [ -x "$REFBIN" ] && [ -x "$MERGEBIN" ] || { echo "run-v8: declaration, scorer, reference binary or merge binary missing" >&2; exit 2; }
MODEL_DIR=$(dirname "${AIEN_MODEL_PATH:?}")
mkdir -p "$BASE" "$OUT"
RES=$OUT/v8-results.jsonl
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
    # ACCEPTANCE-v8 Section 2, row <id>-A: for an edit launch, re-derive the proposal from the
    # accepted attempt's reply (the last attempt with outcome "parsed") and the pre-seed.
    v8merge=null
    if [ "$(jq -r .kind <<<"$t")" = edit ] && [ -f "$BASE/$id/s3-report.json" ]; then
      rf=$BASE/$id.accepted-reply.txt
      if jq -e '[.proposal_attempts // [] | .[] | select(.outcome == "parsed")] | length > 0' "$BASE/$id/s3-report.json" >/dev/null 2>&1; then
        jq -j '[.proposal_attempts[] | select(.outcome == "parsed")] | last | .text' "$BASE/$id/s3-report.json" >"$rf"
        m=$("$MERGEBIN" "$(jq -r .destination <<<"$t")" "$HERE/$seed/$(jq -r .destination <<<"$t")" "$rf" 2>"$BASE/$id.merge.err")
        if jq . <<<"$m" >/dev/null 2>&1; then v8merge=$(jq -c . <<<"$m"); fi
      fi
      echo "launch $id edit re-derivation: $(jq -r 'if . == null then "none" else "ok=\(.ok)" end' <<<"$v8merge")"
    fi
    v5env=(); [ "$(jq -r '.destination // empty' <<<"$t")" != "" ] && [ "${id#N}" = "$id" ] && \
      v5env=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v8.md")
    env "${v5env[@]}" V6_ROWS=rows-v8.jq V8_MERGE="$v8merge" V6_TASK="$id" V6_MAX_TOKENS="$max" V6_REFERENCE="$ref" \
      SPEC="ACCEPTANCE-v8.md spec_version 8" bash "$HERE/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "NEXT-PHASE-1 v8 launch $id" >"$BASE/$id.receipt.out" 2>&1
    rec=$OUT/$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.*receipt \([0-9a-f]*\.json\).*/\1/p')
    if [ -f "$rec" ]; then bash "$HERE/v8-results.sh" "$id" "$rec" >>"$RES"
    else bash "$HERE/v8-results.sh" "$id" - "$DECL" "receipt not written" >>"$RES"; fi
    echo "launch $id receipt $(basename "$rec")"
  else
    bash "$HERE/v8-results.sh" "$id" - "$DECL" "launch failed: no run.json" >>"$RES"   # ACCEPTANCE-v6 Section 7
    echo "launch $id FAILED to launch"
  fi
  jq -r --arg id "$id" 'select(.row | startswith($id + "/")) | select(.verdict != "PASS") | "  \(.verdict) \(.row)"' "$RES"
done
bash "$SCORER" "$DECL" "$RES" >"$OUT/v8-score.json"; rc=$?
echo "NEXT-PHASE-1 v8 campaign verdict (scoring-v5, score-rows.sh exit $rc): $(jq -r .verdict "$OUT/v8-score.json" 2>/dev/null || echo ERROR)"
exit $rc
