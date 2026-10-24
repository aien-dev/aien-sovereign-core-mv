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

# NEXT-PHASE-1 v6: one launch's scoring-v5 result lines (ACCEPTANCE-v6
# Section 7). Shell + jq only. Usage:
#   v6-results.sh LAUNCH_ID RECEIPT.json   -> the launch's rows, one JSON line each
#   v6-results.sh LAUNCH_ID - DECL.json WHY -> every declared row of the launch as
#                                              FAIL (the launch produced no run.json)
# Row ids: "<launch>/v1 <criterion>" (v1..v4 rows, REPORTED rows left out),
# "<launch>/<Q1..Q4|A1..A6>" (v5 rows), "<launch>/<v6 row>". Declared negative
# launches (N1, N2) contribute only their v6 rows (ACCEPTANCE-v6 Section 4). For the
# positive launches the v1 row "Containment: workspace" and the v5 row A1 are
# replaced by the v6 row <id>-CM (record-mark rule, ACCEPTANCE-v6 Section 6.1).
set -u
ID=${1:?LAUNCH_ID} REC=${2:?RECEIPT}
if [ "$REC" = - ]; then
  jq -c --arg id "$ID" --arg why "${4:-launch failed}" \
    '.rows[] | select(.row | startswith($id + "/")) | {row, rep: 1, verdict: "FAIL", outcome: $why}' "${3:?DECL}"
  exit 0
fi
jq -c --arg id "$ID" --arg rec "$(basename "$REC")" '
  (if (.acceptance_v6 // null) == null then error("receipt has no acceptance_v6") else . end)
  | (.task_v6 == $id) as $same
  | (if $same | not then error("receipt is launch \(.task_v6), not \($id)") else . end)
  | ($id | test("^N")) as $neg
  | ((if $neg then [] else
       [.acceptance[] | select(.result != "REPORTED" and .criterion != "Containment: workspace") | {row: ($id + "/v1 " + .criterion), verdict: .result}]
       + [((.acceptance_v5.task_quality // []) + (.acceptance_v5.authority // []))[] | select(.row != "A1") | {row: ($id + "/" + .row), verdict: .result}]
     end)
     + [.acceptance_v6[] | {row: ($id + "/" + .row), verdict: .result}])[]
  | . + {rep: 1, receipt: $rec}' "$REC"
