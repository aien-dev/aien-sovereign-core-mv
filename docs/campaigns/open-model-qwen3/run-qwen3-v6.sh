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

# OPEN-MODEL-QWEN3 v6 qualification (QUALIFICATION-v6.md). Derived from run-qwen3-v5.sh with names only changed: the frozen
# values file (frozen-v6.json), the pinned run base (oq3-v6-runs), the output tag (oq3-v6) and labels. Tasks (tasks-oq3-v5.json),
# seeds (seed-v5), declaration (oq3-v5.decl.json), row modules, v5-rows.sh, driver, receipt builder and scorer are the v5 files,
# byte-identical. It refuses a real run while any pin in frozen-v6.json holds the placeholder or its status does not say FROZEN,
# and afterwards unless every pin read from the checkouts and binaries equals it. The v5 header follows unchanged:
# OPEN-MODEL-QWEN3 v5 campaign (ACCEPTANCE-v5.md). It refuses a real run while any pin
# in frozen-v5.json holds the placeholder or its status does not say FROZEN, and afterwards unless every pin read
# from the checkouts and binaries equals it. Qwen3-4B, derived from run-qwen3-v4.sh: same driver (next-phase-1/run-campaign.sh), receipt builder, v8 rows and
# scorer; new tasks (tasks-oq3-v5.json), one declaration (qualification) and extra rows (v5-rows.sh, rows-oq3-v5.jq).
# Shell + jq only. Run inside the caller's hold, one part per hold. Environment as in ACCEPTANCE-v5 Section 7
# (model, tokenizer, binary, KV context, hardware requirement, checkpoint requirement, declared attempt flag), then:
#   OQ3_PART=1|2|3|4 run-qwen3-v5.sh RUN_BASE OUT_DIR SC_DIR OMEGA_DIR REFERENCE_BIN MERGE_BIN
# SC_DIR is the sovereign-core checkout the binaries were built from, OMEGA_DIR the omega checkout at its omega.lock;
# the commits and lock files are read from them (row <id>-PIN). Parts (ACCEPTANCE-v5 Section 8): 1 = D1 D2 D3,
# 2 = D4 D5 D6, 3 = E1 E2 N1, 4 = N2 R1 and the QUALIFICATION score. Part k needs part k-1 completed in the same
# RUN_BASE with the same binaries, tasks file, declaration, frozen values, row modules and wrapper.
# Dry run (gate G6): OQ3_DRY_TASKS=<tasks file with the same ids, kinds, max_tokens, budgets, destinations and seeds>
# lifts the pinned paths and the frozen-pin checks and writes oq3-v5-DRYRUN-* outputs. A dry run is never a verdict.
# A dry run still starts the real driver and daemon on the GPU.
set -u
PART=${OQ3_PART:-}
case $PART in 1) LAUNCHES="D1 D2 D3" ;; 2) LAUNCHES="D4 D5 D6" ;; 3) LAUNCHES="E1 E2 N1" ;; 4) LAUNCHES="N2 R1" ;; *) echo "run-qwen3-v6: OQ3_PART must be 1, 2, 3 or 4" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
NP1=$(cd "$HERE/../next-phase-1" && pwd)
BASE=${1:?RUN_BASE} OUT=${2:?OUT_DIR} SCD=${3:?SC_DIR} OMD=${4:?OMEGA_DIR} REFBIN=${5:?REFERENCE_BIN} MERGEBIN=${6:?MERGE_BIN}
FROZEN=$HERE/frozen-v6.json
PLACEHOLDER=$(jq -r .placeholder "$FROZEN")

TASKS=$HERE/tasks-oq3-v5.json TAG=oq3-v6
if [ -n "${OQ3_DRY_TASKS:-}" ]; then
  TASKS=$OQ3_DRY_TASKS TAG=oq3-v6-DRYRUN
  shape='[.tasks[] | [.id, .kind, .max_tokens, .budget_ms, .destination, .seed]]'
  [ "$(jq -c "$shape" "$TASKS")" = "$(jq -c "$shape" "$HERE/tasks-oq3-v5.json")" ] ||
    { echo "run-qwen3-v6: dry-run tasks must keep the v5 ids, kinds, max_tokens, budgets, destinations and seeds" >&2; exit 2; }
else
  # Real run: the run path is pinned (the prompt embeds the workspace path), the output directory is fresh,
  # and every frozen pin must be filled in (the comparison with the recorded pins follows below).
  PINNED_BASE=/home/drakestapleton/workspace/oq3-v6-runs
  [ "$BASE" = "$PINNED_BASE" ] || { echo "run-qwen3-v6: RUN_BASE must be $PINNED_BASE" >&2; exit 2; }
  if [ $PART = 1 ]; then
    [ ! -e "$BASE" ] || { echo "run-qwen3-v6: $BASE already exists (the v6 run is fresh and happens once)" >&2; exit 2; }
    [ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT" 2>/dev/null)" ] || { echo "run-qwen3-v6: OUT_DIR must be new or empty" >&2; exit 2; }
  fi
  [ "$(jq --arg p "$PLACEHOLDER" '[.pins[] | select(. == $p)] | length' "$FROZEN")" = 0 ] ||
    { echo "run-qwen3-v6: the build pins are not frozen yet (frozen-v6.json, QUALIFICATION-v6 Section 6)" >&2; exit 2; }
  # Filled pins are not a freeze: a draft freeze-fill change fills them first. The status line changes only at G8.
  case "$(jq -r .status "$FROZEN")" in FROZEN*) ;; *)
    echo "run-qwen3-v6: frozen-v6.json status is not FROZEN (QUALIFICATION-v6 Section 7)" >&2; exit 2 ;; esac
fi
DECL=$HERE/../scoring/declarations/oq3-v5.decl.json
SCORER=$HERE/../scoring/score-rows.sh
[ -f "$DECL" ] && [ -x "$SCORER" ] && [ -x "$REFBIN" ] && [ -x "$MERGEBIN" ] || { echo "run-qwen3-v6: declaration, scorer, reference binary or merge binary missing" >&2; exit 2; }
: "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}" "${AIEN_BIN:?}"
need() { [ "$(printenv "$1")" = "$2" ] || { echo "$1 must be $2 (ACCEPTANCE-v5 Section 7)" >&2; exit 3; }; }
need AIEN_KV_CONTEXT_TOKENS 4096
need AIEN_REQUIRE_BLACKWELL 1
need AIEN_REQUIRE_CHECKPOINT 1
need AIEN_GB10_QWEN3_DECLARED_ATTEMPT 1
# The limits are set per launch below; a stale value from the caller is refused, never overridden silently.
# AIEN_COMPOSE_BUDGET_MS is the retired single budget (sc#284 refuses it too). AIEN_DEV_FALLBACK is new in v5, and so is
# AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK (the dev-only desk opt-out of sc#342, valid only with AIEN_DEV_FALLBACK=1).
for v in AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK AIEN_COMPOSE_MAX_TOKENS AIEN_COMPOSE_DOC_MAX_TOKENS AIEN_COMPOSE_EDIT_BUDGET_MS \
         AIEN_COMPOSE_DOC_BUDGET_MS AIEN_COMPOSE_BUDGET_MS AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET; do
  [ -z "$(printenv $v)" ] || { echo "$v must be unset (ACCEPTANCE-v5 Section 7)" >&2; exit 3; }
done
MODEL_DIR=$(dirname "$AIEN_MODEL_PATH")
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "digest mismatch: $1" >&2; exit 3; }; }
# The model the generation record must bind (row <id>-GR): the index and every shard, re-hashed before each part.
chk "$AIEN_MODEL_PATH" "$(jq -r .model.index_sha256 "$FROZEN")"
while read -r n h; do chk "$MODEL_DIR/$n" "$h"; done < <(jq -r '.model.shards[] | "\(.[0]) \(.[1])"' "$FROZEN")
chk "$AIEN_TOKENIZER_PATH" "$(jq -r .model.tokenizer_sha256 "$FROZEN")"
chk "$MODEL_DIR/config.json" 5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba
chk "$MODEL_DIR/generation_config.json" 835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48
chk "$MODEL_DIR/tokenizer_config.json" a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3
# Build pins, read from the checkouts and the binaries (row <id>-PIN). A real run refuses any difference.
SC=$(git -C "$SCD" rev-parse HEAD 2>/dev/null) && OMC=$(git -C "$OMD" rev-parse HEAD 2>/dev/null) ||
  { echo "run-qwen3-v6: SC_DIR and OMEGA_DIR must be git checkouts" >&2; exit 2; }
OMG=$OMC
[ "$OMC" = "$(tr -d '[:space:]' <"$SCD/omega.lock")" ] || { echo "run-qwen3-v6: OMEGA_DIR HEAD differs from SC_DIR/omega.lock" >&2; exit 2; }
PINS=$(jq -n --arg sc "$SC" --arg om "$OMC" --arg ph "$(tr -d '[:space:]' <"$OMD/physics.lock" 2>/dev/null)" \
  --arg ao "$(tr -d '[:space:]' <"$OMD/aienos.lock" 2>/dev/null)" --arg cl "$(sha256sum "$SCD/Cargo.lock" | cut -d' ' -f1)" \
  --arg cli "$(sha256sum "$AIEN_BIN" | cut -d' ' -f1)" --arg ref "$(sha256sum "$REFBIN" | cut -d' ' -f1)" --arg mrg "$(sha256sum "$MERGEBIN" | cut -d' ' -f1)" \
  '{sovereign_core_commit: $sc, omega_lock_commit: $om, physics_lock_commit: $ph, aienos_lock_commit: $ao, cargo_lock_sha256: $cl,
    aien_cli_sha256: $cli, np1_reference_sha256: $ref, np1_edit_merge_sha256: $mrg}')
if [ -z "${OQ3_DRY_TASKS:-}" ]; then
  [ -z "$(git -C "$SCD" status --porcelain)" ] || { echo "run-qwen3-v6: SC_DIR has uncommitted changes" >&2; exit 2; }
  [ "$(jq -c --argjson p "$PINS" '.pins == $p' "$FROZEN")" = true ] ||
    { echo "run-qwen3-v6: the recorded build pins differ from frozen-v6.json: $(jq -c --argjson p "$PINS" '.pins as $f | [$f | keys[] | select($f[.] != $p[.])]' "$FROZEN")" >&2; exit 2; }
fi
RES=$OUT/$TAG-results.jsonl
if [ $PART = 1 ]; then
  [ ! -e "$BASE/part1.done" ] && [ ! -e "$RES" ] || { echo "run-qwen3-v6: part 1 already ran here" >&2; exit 2; }
  mkdir -p "$BASE" "$OUT"; : >"$RES"
else
  prev=$((PART - 1))
  [ -f "$BASE/part$prev.done" ] && [ ! -e "$BASE/part$PART.started" ] || { echo "run-qwen3-v6: part $PART needs a completed part $prev and runs once" >&2; exit 2; }
  : >"$BASE/part$PART.started"
fi
sha256sum "$AIEN_BIN" "$REFBIN" "$MERGEBIN" "$TASKS" "$DECL" "$FROZEN" "$HERE/rows-oq3-v3.jq" "$HERE/rows-oq3-v4.jq" "$HERE/rows-oq3-v5.jq" \
  "$HERE/v5-rows.sh" "$0" > "$BASE/identity-part$PART.sha256"
if [ $PART != 1 ] && ! cmp -s <(cut -d" " -f1 "$BASE/identity-part1.sha256") <(cut -d" " -f1 "$BASE/identity-part$PART.sha256"); then
  echo "run-qwen3-v6: part $PART binaries, tasks, declaration, frozen values, rows or wrapper differ from part 1 (one build per run)" >&2; exit 2
fi
printf '%s\n' "$PINS" >"$BASE/pins-part$PART.json"
# Memory before the part (row <id>-MEAS), from /proc/meminfo, in kB.
MEM=$(awk '/^MemFree:/ {f=$2} /^Cached:/ {c=$2} END {printf "{\"mem_free_kb\":%d,\"cached_kb\":%d}", f, c}' /proc/meminfo)
printf '%s\n' "$MEM" >"$BASE/meminfo-part$PART.json"
# Launch wall time on CLOCK_BOOTTIME (monotonic, 10 ms ticks), in ms.
up_ms() { echo $(( $(tr -d . </proc/uptime | cut -d' ' -f1) * 10 )); }
# The receipt builder looks for tasks-v8.json, rows-v5/v8.jq and the seed folders next to itself. A staging folder of
# links to the unchanged next-phase-1 files, with tasks-v8.json replaced by this campaign's tasks file and seed-v5
# linked in, lets make-receipt.sh and rows-v9.jq be used (rows-v9.jq is rows-v8.jq with the harness approval-desk key allowed in the
# compose dir, sc#342; rows-v8.jq itself stays byte-identical because completed runs pin it).
STAGE=$BASE/stage
# The unchanged make-receipt.sh reads seed/<destination> for any task that has a seed, which only fits edits. A non-edit
# task whose seed holds other files (D4: its source file; R1: the notes/ folder its destination needs, Section 2) is
# staged with seed null for the receipt builder only; the driver still gets the real seed from the tasks file
# (NP1_SEED_DIR below), and v5-rows.sh checks the D4 source file itself (row SRC).
STAGE_FILTER='(.tasks[] | select(.kind != "edit") | .seed) = null'
if [ $PART = 1 ]; then
  mkdir -p "$STAGE"
  for f in "$NP1"/*; do
    n=$(basename "$f"); case $n in tasks-v8.json) continue ;; esac
    ln -s "$f" "$STAGE/$n"
  done
  jq "$STAGE_FILTER" "$TASKS" >"$STAGE/tasks-v8.json"
  ln -s "$HERE/seed-v5" "$STAGE/seed-v5"
fi
[ "$(jq "$STAGE_FILTER" "$TASKS")" = "$(cat "$STAGE/tasks-v8.json")" ] || { echo "run-qwen3-v6: staged tasks file differs from the tasks file" >&2; exit 2; }
# Product limits (ACCEPTANCE-v5 Section 2) come from the task's own max_tokens and budget_ms (row <id>-D scores the
# same budget_ms); only D6 differs from the product values (Drake 2026-10-09, ACCEPTANCE-v5 Section 2.1).
for id in $LAUNCHES; do
  t=$(jq -c --arg id "$id" '.tasks[] | select(.id == $id)' "$TASKS")
  goal=$(jq -r .goal <<<"$t"); max=$(jq -r .max_tokens <<<"$t"); bud=$(jq -r .budget_ms <<<"$t"); kind=$(jq -r .kind <<<"$t"); seed=$(jq -r '.seed // empty' <<<"$t")
  seed_dir=; [ -n "$seed" ] && seed_dir=$HERE/$seed
  # An edit goal (names an existing file) uses the edit cap AIEN_COMPOSE_MAX_TOKENS and budget AIEN_COMPOSE_EDIT_BUDGET_MS;
  # every other goal creates a new file and uses the document cap and budget. The other cap and budget stay at their product values.
  if [ "$kind" = edit ]; then ecap=$max dcap=$(jq -r .limits.doc_max_tokens "$TASKS") ebud=$bud dbud=$(jq -r .limits.doc_budget_ms "$TASKS")
  else ecap=$(jq -r .limits.edit_max_tokens "$TASKS") dcap=$max ebud=$(jq -r .limits.edit_budget_ms "$TASKS") dbud=$bud; fi
  export AIEN_COMPOSE_EDIT_BUDGET_MS=$ebud AIEN_COMPOSE_DOC_BUDGET_MS=$dbud
  t0=$(up_ms)
  NP1_GOAL=$goal AIEN_COMPOSE_MAX_TOKENS=$ecap AIEN_COMPOSE_DOC_MAX_TOKENS=$dcap NP1_SEED_DIR=$seed_dir \
    bash "$NP1/run-campaign.sh" "$BASE/$id" >"$BASE/$id.driver.out" 2>&1
  drc=$?; t1=$(up_ms)
  jq -n --arg id "$id" --argjson w $((t1 - t0)) --argjson m "$MEM" --argjson p "$PINS" --argjson rc $drc \
    '{launch: $id, wall_ms: $w, wall_clock: "CLOCK_BOOTTIME, 10 ms ticks, around the driver call", driver_exit: $rc, meminfo_before: $m, pins: $p}' >"$BASE/$id.launch-v5.json"
  echo "launch $id driver exit $drc in $((t1 - t0)) ms (edit cap $ecap, document cap $dcap, budgets $AIEN_COMPOSE_EDIT_BUDGET_MS/$AIEN_COMPOSE_DOC_BUDGET_MS ms${seed:+, seed $seed})"
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
      v5env=(TASK_ID="$id" TASK_SPEC="$TASKS" TASK_ACCEPTANCE="$HERE/ACCEPTANCE-v5.md")
    env "${v5env[@]}" V6_ROWS=rows-v9.jq V8_MERGE="$v8merge" V6_TASK="$id" V6_MAX_TOKENS="$max" V6_REFERENCE="$ref" \
      SPEC="open-model-qwen3/QUALIFICATION-v6.md (rows of ACCEPTANCE-v5.md) spec_version 6${OQ3_DRY_TASKS:+ DRY RUN}" bash "$STAGE/make-receipt.sh" \
      "$BASE/$id" "$OUT" "$SC" "$OMC" "$OMG" 0 "OPEN-MODEL-QWEN3 v6${OQ3_DRY_TASKS:+ DRY RUN} launch $id" >"$BASE/$id.receipt.out" 2>&1
    rec=$OUT/$(head -1 "$BASE/$id.receipt.out" | sed -n 's/.*receipt \([0-9a-f]*\.json\).*/\1/p')
    if [ -f "$rec" ]; then
      bash "$NP1/v8-results.sh" "$id" "$rec" >>"$RES"
      bash "$HERE/v5-rows.sh" "$BASE/$id" "$id" "$TASKS" "$OUT" "$(basename "$rec")" "$BASE/$id.launch-v5.json" "$FROZEN" >>"$RES"
    else bash "$NP1/v8-results.sh" "$id" - "$DECL" "receipt not written" >>"$RES"; fi
    echo "launch $id receipt $(basename "$rec")"
  else
    bash "$NP1/v8-results.sh" "$id" - "$DECL" "launch failed: no run.json" >>"$RES"
    echo "launch $id FAILED to launch"
  fi
  jq -r --arg id "$id" 'select(.row | startswith($id + "/")) | select(.verdict != "PASS") | "  \(.verdict) \(.row)"' "$RES"
done
: >"$BASE/part$PART.done"
if [ $PART -le 3 ]; then echo "part $PART done ($LAUNCHES); run part $((PART + 1)) in a new hold"; exit 0; fi
# Qualification verdict: the declaration and the result file only (ACCEPTANCE-v5 Section 5). Rows a receipt computes
# but the declaration does not hold (for N1: N1-B, N1-C, N1-H) are listed by the scorer as extras, never counted.
bash "$SCORER" "$DECL" "$RES" >"$OUT/$TAG-score.json"; rc=$?
echo "OPEN-MODEL-QWEN3 v6${OQ3_DRY_TASKS:+ DRY RUN (not a verdict)} QUALIFICATION (scoring-v5, score-rows.sh exit $rc): $(jq -r .verdict "$OUT/$TAG-score.json" 2>/dev/null || echo ERROR)"
exit $rc
