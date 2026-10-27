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

# OPEN-MODEL-QWEN3 v4 (ACCEPTANCE-v4.md Section 3, PREPARED, NOT FROZEN): the extra rows of one launch, computed
# from the run directory beside the v8 receipt. Shell + jq only. Usage:
#   v4-rows.sh RUN_ROOT LAUNCH_ID TASKS.json OUT_DIR RECEIPT_BASENAME
# Writes OUT_DIR/<sha256 of content>.v4rows.json (never edited afterwards) and prints one scoring-v5 result line
# per row: {row: "<id>/<row>", rep: 1, verdict, receipt}. The saved file, the source file and the seed are hashed
# or read here, after the run, not taken from the runtime's own report.
set -eu
R=${1:?RUN_ROOT} ID=${2:?LAUNCH_ID} TASKS=${3:?TASKS} OUT=${4:?OUT_DIR} REC=${5:?RECEIPT}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$R/steps
task=$(jq -c --arg id "$ID" '.tasks[] | select(.id == $id)' "$TASKS")
[ -n "$task" ] || { echo "v4-rows: launch $ID not in $TASKS" >&2; exit 4; }
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
tmp=$(mktemp "$OUT/.v4rows.XXXXXX")
mods=$(cat "$HERE/rows-oq3-v3.jq" "$HERE/rows-oq3-v4.jq" | sha256sum | cut -d' ' -f1)
jq -L "$HERE" -n --argjson task "$task" --argjson report "$(j "$R/s3-report.json")" --argjson s5 "$(j "$S/S5.json")" \
  --argjson s8 "$(j "$S/S8.json")" --argjson committed "$committed" --argjson seed_text "$seed_text" --argjson source_now "$source_now" \
  --argjson run "$(j "$R/run.json")" --arg rec "$REC" --arg mod "$mods" \
  'include "rows-oq3-v4";
  def j2: (try fromjson catch null);
  {task: $task, daemon: ($run.daemon // null), report: $report, s5: ($s5 // {}), committed: $committed, seed_text: $seed_text, source_now: $source_now,
   auths: (($s8.authorizations // []) | map((.text // "" | j2) + {id, verified}))} as $ev
  | {launch: $task.id, receipt: $rec, rows_modules_sha256: $mod, rows: ($ev | v4_rows)}' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1)
mv "$tmp" "$OUT/$h.v4rows.json"
jq -c --arg id "$ID" --arg rec "$REC" --arg f "$h.v4rows.json" \
  '.rows[] | {row: ($id + "/" + .row), rep: 1, verdict: .result, receipt: $rec, extra_rows_file: $f}' "$OUT/$h.v4rows.json"
