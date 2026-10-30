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

# OPEN-MODEL-QWEN3 v2 campaign (ACCEPTANCE-v2.md): the seven NEXT-PHASE-1 v8 launches
# (next-phase-1/tasks-v8.json: T4 T5 T6 T7 N1 N2 R1) with Qwen3-4B, wired exactly as
# next-phase-1/run-v8.sh (same driver, rows-v8.jq receipts, np1_edit_merge re-derivation,
# np1_reference CPU run for R1, scoring-v5). Differences, all from ACCEPTANCE-v2 Section 2:
# the model and its digests, the Qwen3 environment, the pinned run path and the declaration
# (oq3-v2.decl.json, the np1-v8 rows unchanged). Shell + jq only. Run inside the caller's
# quietlock hold.
#   AIEN_BIN=... AIEN_MODEL_PATH=.../model.safetensors.index.json AIEN_TOKENIZER_PATH=... \
#   AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
#   OQ3_PART=1|2 run-qwen3-v2.sh RUN_BASE OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT REFERENCE_BIN MERGE_BIN
# Two parts, one quietlock hold each (ACCEPTANCE-v2 Section 3; a Qwen3 daemon start takes about 64 s,
# two per launch, so seven launches do not fit one 20-minute hold): part 1 = T4 T5 T6 T7 on a fresh
# RUN_BASE, part 2 = N1 N2 R1 on the RUN_BASE part 1 completed, then the score over all seven.
# Dry run (ACCEPTANCE-v2 Section 5): OQ3_DRY_TASKS=<tasks file with the same ids and kinds and
# made-up goals> lifts the pinned path and writes oq3-v2-DRYRUN-* outputs. A dry run is never a verdict.
set -u
PART=${OQ3_PART:?OQ3_PART must be 1 or 2}
case $PART in 1) LAUNCHES="T4 T5 T6 T7" ;; 2) LAUNCHES="N1 N2 R1" ;; *) echo "run-qwen3-v2: OQ3_PART must be 1 or 2" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
NP1=$(cd "$HERE/../next-phase-1" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?OMEGA_COMPOSE} OMG=${5:?OMEGA_GPU} REFBIN=${6:?REFERENCE_BIN} MERGEBIN=${7:?MERGE_BIN}
TASKS=$NP1/tasks-v8.json TAG=oq3-v2
if [ -n "${OQ3_DRY_TASKS:-}" ]; then
  TASKS=$OQ3_DRY_TASKS TAG=oq3-v2-DRYRUN
  [ "$(jq -c '[.tasks[] | [.id, .kind, .max_tokens, .destination, .seed]]' "$TASKS")" = \
    "$(jq -c '[.tasks[] | [.id, .kind, .max_tokens, .destination, .seed]]' "$NP1/tasks-v8.json")" ] ||
    { echo "run-qwen3-v2: dry-run tasks must keep the v8 ids, kinds, max_tokens, destinations and seeds" >&2; exit 2; }
else
  # ACCEPTANCE-v2 Section 3: the run path is pinned (the prompt embeds the workspace path) and the run is fresh.
  PINNED_BASE=/home/drakestapleton/workspace/oq3-v2-runs
  [ "$BASE" = "$PINNED_BASE" ] || { echo "run-qwen3-v2: RUN_BASE must be $PINNED_BASE" >&2; exit 2; }
  if [ $PART = 1 ]; then [ ! -e "$BASE" ] || { echo "run-qwen3-v2: $BASE already exists (the v2 run is fresh and happens once)" >&2; exit 2; }; fi
fi
DECL=$HERE/../scoring/declarations/oq3-v2.decl.json
SCORER=$HERE/../scoring/score-rows.sh
[ -f "$DECL" ] && [ -x "$SCORER" ] && [ -x "$REFBIN" ] && [ -x "$MERGEBIN" ] || { echo "run-qwen3-v2: declaration, scorer, reference binary or merge binary missing" >&2; exit 2; }
: "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}" "${AIEN_BIN:?}"
need() { [ "$(printenv "$1")" = "$2" ] || { echo "$1 must be $2 (ACCEPTANCE-v2 Section 2)" >&2; exit 3; }; }
need AIEN_KV_CONTEXT_TOKENS 4096
need AIEN_REQUIRE_BLACKWELL 1
need AIEN_GB10_QWEN3_DECLARED_ATTEMPT 1
for v in AIEN_FORCE_CPU_STUB AIEN_COMPOSE_MAX_TOKENS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET; do
  [ -z "$(printenv $v)" ] || { echo "$v must be unset (ACCEPTANCE-v2 Section 2)" >&2; exit 3; }
done
MODEL_DIR=$(dirname "$AIEN_MODEL_PATH")
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "digest mismatch: $1" >&2; exit 3; }; }
chk "$AIEN_MODEL_PATH" d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c
chk "$MODEL_DIR/model-00001-of-00003.safetensors" 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6
chk "$MODEL_DIR/model-00002-of-00003.safetensors" 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1
chk "$MODEL_DIR/model-00003-of-00003.safetensors" 7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d
chk "$AIEN_TOKENIZER_PATH" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
chk "$MODEL_DIR/config.json" 5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba
chk "$MODEL_DIR/generation_config.json" 835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48
RES=$OUT/$TAG-results.jsonl
if [ $PART = 1 ]; then
  [ ! -e "$BASE/part1.done" ] && [ ! -e "$RES" ] || { echo "run-qwen3-v2: part 1 already ran here" >&2; exit 2; }
  mkdir -p "$BASE" "$OUT"; : >"$RES"
else
  [ -f "$BASE/part1.done" ] && [ ! -e "$BASE/part2.started" ] || { echo "run-qwen3-v2: part 2 needs a completed part 1 and runs once" >&2; exit 2; }
  : >"$BASE/part2.started"
fi
sha256sum "$AIEN_BIN" "$REFBIN" "$MERGEBIN" "$TASKS" "$DECL" "$0" > "$BASE/identity-part$PART.sha256"
if [ $PART = 2 ] && ! cmp -s <(cut -d" " -f1 "$BASE/identity-part1.sha256") <(cut -d" " -f1 "$BASE/identity-part2.sha256"); then
  echo "run-qwen3-v2: part 2 binaries, tasks, declaration or wrapper differ from part 1 (one build per run)" >&2; exit 2
fi
for id in $LAUNCHES; do
  t=$(jq -c --arg id "$id" '.tasks[] | select(.id == $id)' "$TASKS")
  goal=$(jq -r .goal <<<"$t"); max=$(jq -r .max_tokens <<<"$t"); seed=$(jq -r '.seed // empty' <<<"$t")
  seed_dir=; [ -n "$seed" ] && seed_dir=$NP1/$seed
  NP1_GOAL=$goal AIEN_COMPOSE_MAX_TOKENS=$max NP1_SEED_DIR=$seed_dir \
    bash "$NP1/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "launch $id driver exit $? (max_tokens $max${seed:+, seed $seed})"
  ref=
  if [ "$(jq -r .kind <<<"$t")" = identity ] && [ -d "$BASE/$id/ws" ]; then
    ws=$(cd "$BASE/$id/ws" && pwd -P)
    "$REFBIN" "$MODEL_DIR" "$goal" "$ws" "$max" new >"$BASE/$id.reference.json" 2>"$BASE/$id.reference.err"
    echo "launch $id CPU reference exit $?"
    ref=$BASE/$id.reference.json
  fi
  if [ -f "$BASE/$id/run.json" ]; then
    v8merge=null
    if [ "$(jq -r .kind <<<"$t")" = edit ] && [ -f "$BASE/$id/s3-report.json" ]; then
      rf=$BASE/$id.accepted-reply.txt
      if jq -e '[.proposal_attempts // [] | .[] | select(.outcome == "parsed")] | length > 0' "$BASE/$id/s3-report.json" >/dev/null 2>&1; then
        jq -j '[.proposal_attempts[] | select(.outcome == "parsed")] | last | .text' "$BASE/$id/s3-report.json" >"$rf"
        m=$("$MERGEBIN" "$(jq -r .destination <<<"$t")" "$NP1/$seed/$(jq -r .destination <<<"$t")" "$rf" 2>"$BASE/$id.merge.err")
        if jq . <<<"$m" >/dev/null 2>&1; then v8merge=$(jq -c . <<<"$m"); fi
      fi
      echo "launch $id edit re-derivation: $(jq -r 'if . == null then "none" else "ok=\(.ok)" end' <<<"$v8merge")"
    fi
    v5env=(); [ "$(jq -r '.destination // empty' <<<"$t")" != "" ] && [ "${id#N}" = "$id" ] && \
      v5env=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE="$NP1/ACCEPTANCE-v8.md")
    env "${v5env[@]}" V6_ROWS=rows-v8.jq V8_MERGE="$v8merge" V6_TASK="$id" V6_MAX_TOKENS="$max" V6_REFERENCE="$ref" \
      SPEC="open-model-qwen3/ACCEPTANCE-v2.md spec_version 2${OQ3_DRY_TASKS:+ DRY RUN}" bash "$NP1/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "OPEN-MODEL-QWEN3 v2${OQ3_DRY_TASKS:+ DRY RUN} launch $id" >"$BASE/$id.receipt.out" 2>&1
    rec=$OUT/$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.*receipt \([0-9a-f]*\.json\).*/\1/p')
    if [ -f "$rec" ]; then bash "$NP1/v8-results.sh" "$id" "$rec" >>"$RES"
    else bash "$NP1/v8-results.sh" "$id" - "$DECL" "receipt not written" >>"$RES"; fi
    echo "launch $id receipt $(basename "$rec")"
  else
    bash "$NP1/v8-results.sh" "$id" - "$DECL" "launch failed: no run.json" >>"$RES"
    echo "launch $id FAILED to launch"
  fi
  jq -r --arg id "$id" 'select(.row | startswith($id + "/")) | select(.verdict != "PASS") | "  \(.verdict) \(.row)"' "$RES"
done
if [ $PART = 1 ]; then : >"$BASE/part1.done"; echo "part 1 done (T4 T5 T6 T7); run part 2 in a new hold"; exit 0; fi
bash "$SCORER" "$DECL" "$RES" >"$OUT/$TAG-score.json"; rc=$?
echo "OPEN-MODEL-QWEN3 v2${OQ3_DRY_TASKS:+ DRY RUN (not a verdict)} (scoring-v5, score-rows.sh exit $rc): $(jq -r .verdict "$OUT/$TAG-score.json" 2>/dev/null || echo ERROR)"
exit $rc
