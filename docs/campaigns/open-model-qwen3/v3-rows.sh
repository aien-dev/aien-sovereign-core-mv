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

# OPEN-MODEL-QWEN3 v3 (ACCEPTANCE-v3.md Section 4, FROZEN): the extra rows of one launch,
# computed from the run directory beside the v8 receipt. Shell + jq only. Usage:
#   v3-rows.sh RUN_ROOT LAUNCH_ID TASKS.json OUT_DIR RECEIPT_BASENAME
# Writes OUT_DIR/<sha256 of content>.v3rows.json (never edited afterwards) and prints one scoring-v5
# result line per row: {row: "<id>/<row>", rep: 1, verdict, receipt}.
set -eu
R=${1:?RUN_ROOT} ID=${2:?LAUNCH_ID} TASKS=${3:?TASKS} OUT=${4:?OUT_DIR} REC=${5:?RECEIPT}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$R/steps
task=$(jq -c --arg id "$ID" '.tasks[] | select(.id == $id)' "$TASKS")
[ -n "$task" ] || { echo "v3-rows: launch $ID not in $TASKS" >&2; exit 4; }
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
tmp=$(mktemp "$OUT/.v3rows.XXXXXX")
jq -n --argjson task "$task" --argjson report "$(j "$R/s3-report.json")" --argjson s5 "$(j "$S/S5.json")" \
  --argjson s8 "$(j "$S/S8.json")" --argjson committed "$committed" --arg rec "$REC" --arg mod "$(sha256sum "$HERE/rows-oq3-v3.jq" | cut -d' ' -f1)" \
  "$(cat "$HERE/rows-oq3-v3.jq")"'
  def j2: (try fromjson catch null);
  {task: $task, report: $report, s5: ($s5 // {}), committed: $committed,
   auths: (($s8.authorizations // []) | map((.text // "" | j2) + {id, verified}))} as $ev
  | {launch: $task.id, receipt: $rec, rows_module_sha256: $mod, rows: ($ev | v3_rows)}' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1)
mv "$tmp" "$OUT/$h.v3rows.json"
jq -c --arg id "$ID" --arg rec "$REC" --arg f "$h.v3rows.json" \
  '.rows[] | {row: ($id + "/" + .row), rep: 1, verdict: .result, receipt: $rec, extra_rows_file: $f}' "$OUT/$h.v3rows.json"
