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

# OPEN-MODEL-QWEN3 v5 (ACCEPTANCE-v5.md Section 3, DRAFT, NOT FROZEN): the extra rows of one launch, computed
# from the run directory beside the v8 receipt. Derived from v4-rows.sh; shell + jq only. Usage:
#   v5-rows.sh RUN_ROOT LAUNCH_ID TASKS.json OUT_DIR RECEIPT_BASENAME LAUNCH_RECORD|- [FROZEN.json]
# LAUNCH_RECORD is the wrapper's <id>.launch-v5.json (wall time, memory before the part, build pins); "-" = none.
# FROZEN defaults to frozen-v5.json beside this script.
# Writes OUT_DIR/<sha256 of content>.v5rows.json (never edited afterwards) and prints one scoring-v5 result line
# per row: {row: "<id>/<row>", rep: 1, verdict, receipt}. The saved file, the source file and the seed are hashed
# or read here, after the run, not taken from the runtime's own report.
set -eu
R=${1:?RUN_ROOT} ID=${2:?LAUNCH_ID} TASKS=${3:?TASKS} OUT=${4:?OUT_DIR} REC=${5:?RECEIPT} LREC=${6:?LAUNCH_RECORD}
HERE=$(cd "$(dirname "$0")" && pwd)
FROZEN=${7:-$HERE/frozen-v5.json}
# Every path is made absolute, because the row module is read with the working directory set to HERE (below).
abs() { case $1 in /* | -) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
R=$(abs "$R") TASKS=$(abs "$TASKS") OUT=$(abs "$OUT") LREC=$(abs "$LREC") FROZEN=$(abs "$FROZEN")
S=$R/steps
task=$(jq -c --arg id "$ID" '.tasks[] | select(.id == $id)' "$TASKS")
[ -n "$task" ] || { echo "v5-rows: launch $ID not in $TASKS" >&2; exit 4; }
j() { if [ -s "$1" ]; then jq -c . "$1" 2>/dev/null || echo null; else echo null; fi; }
ws=$(cd "$R/ws" && pwd -P)
committed='{"path_rel":null,"text":null,"file_sha256":null}'
s5p=$(jq -r '.path // empty' "$S/S5.json" 2>/dev/null || true)
if [ -n "$s5p" ] && [ -f "$s5p" ]; then
  real=$(realpath -e "$s5p")
  case "$real" in "$ws"/*)
    committed=$(jq -n --arg p "${real#"$ws"/}" --rawfile t "$real" --arg h "$(sha256sum "$real" | cut -d' ' -f1)" \
      '{path_rel: $p, text: $t, file_sha256: $h}') ;;
  esac
fi
seed_text=null
if [ "$(jq -r '.kind' <<<"$task")" = edit ]; then
  sd=$(jq -r '.seed // empty' <<<"$task"); dest=$(jq -r '.destination // empty' <<<"$task")
  [ -n "$sd" ] && [ -f "$HERE/$sd/$dest" ] && seed_text=$(jq -n --rawfile t "$HERE/$sd/$dest" '$t')
fi
source_now=null
sp=$(jq -r '.source.path // empty' <<<"$task")
if [ -n "$sp" ] && [ -f "$ws/$sp" ]; then
  source_now=$(jq -n --arg p "$sp" --arg h "$(sha256sum "$ws/$sp" | cut -d' ' -f1)" '{path: $p, sha256: $h}')
fi
# N1: is there anything at the requested destination, resolved against the workspace (it points outside it)?
dest_exists=null
if [ "$(jq -r '.kind' <<<"$task")" = negative-boundary ]; then
  dp="$ws/$(jq -r '.destination' <<<"$task")"
  if [ -e "$dp" ] || [ -L "$dp" ]; then dest_exists=true; else dest_exists=false; fi
fi
# Daemon logs, one per daemon start recorded in run.json, in that order: only the lines the rows read.
tmp=$(mktemp "$OUT/.v5rows.XXXXXX")
logs='[]'
nd=$(jq '(.daemon // []) | length' "$R/run.json" 2>/dev/null || echo 0)
for k in $(seq 1 "$nd"); do
  f=$R/daemon-$k.log
  if [ -f "$f" ]; then
    sed 's/\x1b\[[0-9;]*m//g; s/^[[:space:]]*//' "$f" >"$tmp"
    l=$(jq -n --arg n "daemon-$k.log" --rawfile c "$tmp" '($c | split("\n")) as $L
      | {name: $n, strict: [$L[] | select(startswith("STRICT "))], op_reports: [$L[] | select(startswith("OP_REPORT "))],
         shards: [$L[] | select(startswith("CHECKPOINT_SHARDS "))], violations: ([$L[] | select(contains("STRICT_REAL_MODEL_VIOLATION"))] | length)}')
  else
    l=$(jq -n --arg n "daemon-$k.log" '{name: $n, missing: true, strict: [], op_reports: [], shards: [], violations: 0}')
  fi
  logs=$(jq -c --argjson l "$l" '. + [$l]' <<<"$logs")
done
launch=null; [ "$LREC" != - ] && launch=$(j "$LREC")
mods=$(cat "$HERE/rows-oq3-v3.jq" "$HERE/rows-oq3-v4.jq" "$HERE/rows-oq3-v5.jq" | sha256sum | cut -d' ' -f1)
run=$(j "$R/run.json")
# jq looks for a top-level include in the working directory before -L, so a stray rows-oq3-v5.jq where the caller stands
# would be used instead of this one. Read the module from HERE only.
cd "$HERE"
jq -L "$HERE" -n --argjson task "$task" --argjson report "$(j "$R/s3-report.json")" --argjson s5 "$(j "$S/S5.json")" \
  --argjson s8 "$(j "$S/S8.json")" --argjson pre "$(j "$S/pre-restart-recall.json")" --argjson s3 "$(j "$S/S3.json")" \
  --argjson committed "$committed" --argjson seed_text "$seed_text" --argjson source_now "$source_now" \
  --argjson run "$run" --argjson logs "$logs" --argjson dest_exists "$dest_exists" --argjson launch "$launch" \
  --argjson frozen "$(j "$FROZEN")" --arg rec "$REC" --arg mod "$mods" \
  'include "rows-oq3-v5";
  def j2: (try fromjson catch null);
  {task: $task, daemon: ($run.daemon // null), report: $report, s5: ($s5 // {}), committed: $committed, seed_text: $seed_text, source_now: $source_now,
   auths: (($s8.authorizations // []) | map((.text // "" | j2) + {id, verified})),
   run_steps: ($run.steps // null), containment: ($run.containment // null), constraint_text: ($run.constraint_text // null),
   s3_step: (if $s3 == null then null else {ok: $s3.ok, error: $s3.error} end),
   recall_pre: $pre, recall_s8: $s8, host: (($s8.recall.host // []) | map({id, note, verified, text})),
   logs: $logs, dest_exists: $dest_exists, launch: $launch, frozen: $frozen} as $ev
  | {launch: $task.id, receipt: $rec, rows_modules_sha256: $mod, rows: ($ev | v5_rows)}' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1)
mv "$tmp" "$OUT/$h.v5rows.json"
jq -c --arg id "$ID" --arg rec "$REC" --arg f "$h.v5rows.json" \
  '.rows[] | {row: ($id + "/" + .row), rep: 1, verdict: .result, receipt: $rec, extra_rows_file: $f}' "$OUT/$h.v5rows.json"
