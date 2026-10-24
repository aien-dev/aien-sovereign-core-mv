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

# OPEN-MODEL-QWEN3 v3 campaign (ACCEPTANCE-v3.md; FROZEN when this file is on main: it refuses to run unless the
# commits and the binaries sha256 below match). Eight launches in three parts, Qwen3-4B, adapted from
# run-qwen3-v2.sh: same driver (next-phase-1/run-campaign.sh), same receipt builder and v8 rows, same
# scorer; new tasks (tasks-oq3-v3.json), declaration (oq3-v3.decl.json) and extra rows (v3-rows.sh).
# Shell + jq only. Run inside the caller's quietlock hold.
#   AIEN_BIN=... AIEN_MODEL_PATH=.../model.safetensors.index.json AIEN_TOKENIZER_PATH=... \
#   AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 \
#   OQ3_PART=1|2|3 run-qwen3-v3.sh RUN_BASE OUT_DIR SC_COMMIT OMEGA_COMPOSE_COMMIT OMEGA_GPU_COMMIT REFERENCE_BIN MERGE_BIN
# Parts (ACCEPTANCE-v3 Section 8): 1 = G1 G2 G3, 2 = E1 E2 N1 N2, 3 = R1 and the score. Part k needs part k-1
# completed in the same RUN_BASE with the same binaries, tasks file, declaration and wrapper.
# Dry run (ACCEPTANCE-v3 Section 9): OQ3_DRY_TASKS=<tasks file with the same ids, kinds, max_tokens,
# budgets, destinations and seeds and made-up goals> lifts the pinned paths and writes oq3-v3-DRYRUN-*
# outputs. A dry run is never a verdict.
set -u
PART=${OQ3_PART:?OQ3_PART must be 1, 2 or 3}
case $PART in 1) LAUNCHES="G1 G2 G3" ;; 2) LAUNCHES="E1 E2 N1 N2" ;; 3) LAUNCHES="R1" ;; *) echo "run-qwen3-v3: OQ3_PART must be 1, 2 or 3" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
NP1=$(cd "$HERE/../next-phase-1" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SC=${3:?SC_COMMIT} OMC=${4:?OMEGA_COMPOSE} OMG=${5:?OMEGA_GPU} REFBIN=${6:?REFERENCE_BIN} MERGEBIN=${7:?MERGE_BIN}

# ---- Build identity (ACCEPTANCE-v3 Section 7), filled at the freeze: sovereign-core main 8f3e8c8 (sc#285, #284, #287 merged),
# omega.lock 01f6a74 (omega#331). A real run refuses unless the commit arguments and the three binaries sha256 equal these.
PLACEHOLDER="TO BE FILLED at freeze, after all required changes merge and the combined build passes review and checks"
FROZEN_SC_COMMIT=8f3e8c8b879e10dd83883cee150f16508284d643
FROZEN_OMEGA_COMMIT=01f6a74636b8383b010cdb95597839582c415c27
FROZEN_AIEN_CLI_SHA256=689027ea9ac09f52ec130a2d0b0310995b7539113ccd3a63b06f1037ad2cf4de
FROZEN_NP1_REFERENCE_SHA256=df91eede1223f454b7c6afd47589aac6406c8e69e393018fdbe3377b2103f686
FROZEN_NP1_EDIT_MERGE_SHA256=abc345a47e872728e8b59d050ccf41794c78266e19da882ad6e3d1db681c6377

TASKS=$HERE/tasks-oq3-v3.json TAG=oq3-v3
if [ -n "${OQ3_DRY_TASKS:-}" ]; then
  TASKS=$OQ3_DRY_TASKS TAG=oq3-v3-DRYRUN
  shape='[.tasks[] | [.id, .kind, .max_tokens, .budget_ms, .destination, .seed]]'
  [ "$(jq -c "$shape" "$TASKS")" = "$(jq -c "$shape" "$HERE/tasks-oq3-v3.json")" ] ||
    { echo "run-qwen3-v3: dry-run tasks must keep the v3 ids, kinds, max_tokens, budgets, destinations and seeds" >&2; exit 2; }
else
  # Real run: the run path is pinned (the prompt embeds the workspace path), the output directory is fresh,
  # and the build identity must be filled in and must match.
  PINNED_BASE=/home/drakestapleton/workspace/oq3-v3-runs
  [ "$BASE" = "$PINNED_BASE" ] || { echo "run-qwen3-v3: RUN_BASE must be $PINNED_BASE" >&2; exit 2; }
  if [ $PART = 1 ]; then
    [ ! -e "$BASE" ] || { echo "run-qwen3-v3: $BASE already exists (the v3 run is fresh and happens once)" >&2; exit 2; }
    [ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT" 2>/dev/null)" ] || { echo "run-qwen3-v3: OUT_DIR must be new or empty" >&2; exit 2; }
  fi
  for v in "$FROZEN_SC_COMMIT" "$FROZEN_OMEGA_COMMIT" "$FROZEN_AIEN_CLI_SHA256" "$FROZEN_NP1_REFERENCE_SHA256" "$FROZEN_NP1_EDIT_MERGE_SHA256"; do
    [ "$v" != "$PLACEHOLDER" ] || { echo "run-qwen3-v3: the build identity is not frozen yet (ACCEPTANCE-v3 Section 7)" >&2; exit 2; }
  done
  [ "$SC" = "$FROZEN_SC_COMMIT" ] && [ "$OMC" = "$FROZEN_OMEGA_COMMIT" ] && [ "$OMG" = "$FROZEN_OMEGA_COMMIT" ] ||
    { echo "run-qwen3-v3: SC_COMMIT, OMEGA_COMPOSE_COMMIT and OMEGA_GPU_COMMIT must equal the frozen commits" >&2; exit 2; }
fi
DECL=$HERE/../scoring/declarations/oq3-v3.decl.json
SCORER=$HERE/../scoring/score-rows.sh
[ -f "$DECL" ] && [ -x "$SCORER" ] && [ -x "$REFBIN" ] && [ -x "$MERGEBIN" ] || { echo "run-qwen3-v3: declaration, scorer, reference binary or merge binary missing" >&2; exit 2; }
: "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}" "${AIEN_BIN:?}"
need() { [ "$(printenv "$1")" = "$2" ] || { echo "$1 must be $2 (ACCEPTANCE-v3 Section 7)" >&2; exit 3; }; }
need AIEN_KV_CONTEXT_TOKENS 4096
need AIEN_REQUIRE_BLACKWELL 1
need AIEN_GB10_QWEN3_DECLARED_ATTEMPT 1
# The limits are set per launch below; a stale value from the caller is refused, never overridden silently.
# AIEN_COMPOSE_BUDGET_MS is the retired single budget (sc#284 refuses it too).
for v in AIEN_FORCE_CPU_STUB AIEN_COMPOSE_MAX_TOKENS AIEN_COMPOSE_DOC_MAX_TOKENS AIEN_COMPOSE_EDIT_BUDGET_MS \
         AIEN_COMPOSE_DOC_BUDGET_MS AIEN_COMPOSE_BUDGET_MS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET; do
  [ -z "$(printenv $v)" ] || { echo "$v must be unset (ACCEPTANCE-v3 Section 7)" >&2; exit 3; }
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
if [ -z "${OQ3_DRY_TASKS:-}" ]; then
  chk "$AIEN_BIN" "$FROZEN_AIEN_CLI_SHA256"
  chk "$REFBIN" "$FROZEN_NP1_REFERENCE_SHA256"
  chk "$MERGEBIN" "$FROZEN_NP1_EDIT_MERGE_SHA256"
fi
RES=$OUT/$TAG-results.jsonl
if [ $PART = 1 ]; then
  [ ! -e "$BASE/part1.done" ] && [ ! -e "$RES" ] || { echo "run-qwen3-v3: part 1 already ran here" >&2; exit 2; }
  mkdir -p "$BASE" "$OUT"; : >"$RES"
else
  prev=$((PART - 1))
  [ -f "$BASE/part$prev.done" ] && [ ! -e "$BASE/part$PART.started" ] || { echo "run-qwen3-v3: part $PART needs a completed part $prev and runs once" >&2; exit 2; }
  : >"$BASE/part$PART.started"
fi
sha256sum "$AIEN_BIN" "$REFBIN" "$MERGEBIN" "$TASKS" "$DECL" "$HERE/rows-oq3-v3.jq" "$HERE/v3-rows.sh" "$0" > "$BASE/identity-part$PART.sha256"
if [ $PART != 1 ] && ! cmp -s <(cut -d" " -f1 "$BASE/identity-part1.sha256") <(cut -d" " -f1 "$BASE/identity-part$PART.sha256"); then
  echo "run-qwen3-v3: part $PART binaries, tasks, declaration, rows or wrapper differ from part 1 (one build per run)" >&2; exit 2
fi
# The receipt builder looks for tasks-v8.json, rows-v5/v8.jq and the seed folders next to itself. A staging folder of
# links to the unchanged next-phase-1 files, with tasks-v8.json replaced by this campaign's tasks file and seed-v3
# linked in, lets the unchanged make-receipt.sh and rows-v8.jq be used (ACCEPTANCE-v3 Section 6).
STAGE=$BASE/stage
if [ $PART = 1 ]; then
  mkdir -p "$STAGE"
  for f in "$NP1"/*; do
    n=$(basename "$f"); case $n in tasks-v8.json) continue ;; esac
    ln -s "$f" "$STAGE/$n"
  done
  cp "$TASKS" "$STAGE/tasks-v8.json"
  ln -s "$HERE/seed-v3" "$STAGE/seed-v3"
fi
cmp -s "$TASKS" "$STAGE/tasks-v8.json" || { echo "run-qwen3-v3: staged tasks file differs from the tasks file" >&2; exit 2; }
# Product limits (ACCEPTANCE-v3 Section 1), the same for every launch.
export AIEN_COMPOSE_EDIT_BUDGET_MS=$(jq -r .limits.edit_budget_ms "$TASKS") AIEN_COMPOSE_DOC_BUDGET_MS=$(jq -r .limits.doc_budget_ms "$TASKS")
for id in $LAUNCHES; do
  t=$(jq -c --arg id "$id" '.tasks[] | select(.id == $id)' "$TASKS")
  goal=$(jq -r .goal <<<"$t"); max=$(jq -r .max_tokens <<<"$t"); kind=$(jq -r .kind <<<"$t"); seed=$(jq -r '.seed // empty' <<<"$t")
  seed_dir=; [ -n "$seed" ] && seed_dir=$HERE/$seed
  # An edit goal (names an existing file) uses the edit cap AIEN_COMPOSE_MAX_TOKENS; every other goal creates a
  # new file and uses the document cap AIEN_COMPOSE_DOC_MAX_TOKENS. The other cap stays at its product value.
  if [ "$kind" = edit ]; then ecap=$max dcap=$(jq -r .limits.doc_max_tokens "$TASKS"); else ecap=$(jq -r .limits.edit_max_tokens "$TASKS") dcap=$max; fi
  NP1_GOAL=$goal AIEN_COMPOSE_MAX_TOKENS=$ecap AIEN_COMPOSE_DOC_MAX_TOKENS=$dcap NP1_SEED_DIR=$seed_dir \
    bash "$NP1/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  echo "launch $id driver exit $? (edit cap $ecap, document cap $dcap, budgets $AIEN_COMPOSE_EDIT_BUDGET_MS/$AIEN_COMPOSE_DOC_BUDGET_MS ms${seed:+, seed $seed})"
  ref=
  if [ "$kind" = identity ] && [ -d "$BASE/$id/ws" ]; then
    ws=$(cd "$BASE/$id/ws" && pwd -P)
    "$REFBIN" "$MODEL_DIR" "$goal" "$ws" "$max" new >"$BASE/$id.reference.json" 2>"$BASE/$id.reference.err"
    echo "launch $id CPU reference exit $?"
    ref=$BASE/$id.reference.json
  fi
  if [ -f "$BASE/$id/run.json" ]; then
    v8merge=null
    if [ "$kind" = edit ] && [ -f "$BASE/$id/s3-report.json" ]; then
      rf=$BASE/$id.accepted-reply.txt
      if jq -e '[.proposal_attempts // [] | .[] | select(.outcome == "parsed")] | length > 0' "$BASE/$id/s3-report.json" >/dev/null 2>&1; then
        jq -j '[.proposal_attempts[] | select(.outcome == "parsed")] | last | .text' "$BASE/$id/s3-report.json" >"$rf"
        m=$("$MERGEBIN" "$(jq -r .destination <<<"$t")" "$HERE/$seed/$(jq -r .destination <<<"$t")" "$rf" 2>"$BASE/$id.merge.err")
        if jq . <<<"$m" >/dev/null 2>&1; then v8merge=$(jq -c . <<<"$m"); fi
      fi
      echo "launch $id edit re-derivation: $(jq -r 'if . == null then "none" else "ok=\(.ok)" end' <<<"$v8merge")"
    fi
    v5env=(); [ "$(jq -r '.destination // empty' <<<"$t")" != "" ] && [ "${id#N}" = "$id" ] && \
      v5env=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v3.md")
    env "${v5env[@]}" V6_ROWS=rows-v8.jq V8_MERGE="$v8merge" V6_TASK="$id" V6_MAX_TOKENS="$max" V6_REFERENCE="$ref" \
      SPEC="open-model-qwen3/ACCEPTANCE-v3.md spec_version 3${OQ3_DRY_TASKS:+ DRY RUN}" bash "$STAGE/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "OPEN-MODEL-QWEN3 v3${OQ3_DRY_TASKS:+ DRY RUN} launch $id" >"$BASE/$id.receipt.out" 2>&1
    rec=$OUT/$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.*receipt \([0-9a-f]*\.json\).*/\1/p')
    if [ -f "$rec" ]; then
      bash "$NP1/v8-results.sh" "$id" "$rec" >>"$RES"
      bash "$HERE/v3-rows.sh" "$BASE/$id" "$id" "$TASKS" "$OUT" "$(basename "$rec")" >>"$RES"
    else bash "$NP1/v8-results.sh" "$id" - "$DECL" "receipt not written" >>"$RES"; fi
    echo "launch $id receipt $(basename "$rec")"
  else
    bash "$NP1/v8-results.sh" "$id" - "$DECL" "launch failed: no run.json" >>"$RES"
    echo "launch $id FAILED to launch"
  fi
  jq -r --arg id "$id" 'select(.row | startswith($id + "/")) | select(.verdict != "PASS") | "  \(.verdict) \(.row)"' "$RES"
done
: >"$BASE/part$PART.done"
if [ $PART != 3 ]; then echo "part $PART done ($LAUNCHES); run part $((PART + 1)) in a new hold"; exit 0; fi
bash "$SCORER" "$DECL" "$RES" >"$OUT/$TAG-score.json"; rc=$?
echo "OPEN-MODEL-QWEN3 v3${OQ3_DRY_TASKS:+ DRY RUN (not a verdict)} (scoring-v5, score-rows.sh exit $rc): $(jq -r .verdict "$OUT/$TAG-score.json" 2>/dev/null || echo ERROR)"
exit $rc
