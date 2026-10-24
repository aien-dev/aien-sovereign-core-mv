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

# SMOLLM2-Q harness self-test (copied from cand4-qualification): explicit boolean false must never read as missing or as true
# (jq's `//` turns false into its alternative). Shell + jq only. Exit 0 iff all hold.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); SCORE=$HERE/../scoring/score-rows.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; bad=0
t() { if [ "$2" = "$3" ]; then echo "PASS  $1"; else echo "FAIL  $1: got $2, want $3"; bad=1; fi; }
echo '{"backend":"OmegaGb10Backend (native)","s3_committed":false}' >"$T/fj.json"
t "fixture s3_committed=false: C3-control FAIL" "$(jq -r 'if .s3_committed == true and ((.backend // "") | test("OmegaGb10Backend")) then "PASS" else "FAIL" end' "$T/fj.json")" FAIL
t "fixture s3_committed=false: external F0 false" "$(jq -c '{F0: (.s3_committed == true)}' "$T/fj.json")" '{"F0":false}'
t "fixture s3_committed=false: valid false" "$(jq -r '(. // {}).s3_committed == true' "$T/fj.json")" false
echo '{"s3_committed":true,"backend":"x"}' >"$T/ft.json"
t "fixture s3_committed=true: external F0 true" "$(jq -c '{F0: (.s3_committed == true)}' "$T/ft.json")" '{"F0":true}'
# scorer: a declared row with explicit not_applicable false and a result line verdict FAIL is FAIL, not missing
jq -n '{contract:"scoring-v5",campaign:"selftest",rows:[{row:"R",role:"case",reps:1,expected:"PASS",control:null,not_applicable:false}]}' >"$T/d.json"
echo '{"row":"R","rep":1,"verdict":"FAIL","outcome":"explicit"}' >"$T/r.jsonl"
t "scorer: explicit FAIL result is FAIL" "$(bash "$SCORE" "$T/d.json" "$T/r.jsonl" | jq -r .verdict)" FAIL
echo '{"row":"R","rep":1,"verdict":"PASS","outcome":"explicit"}' >"$T/r.jsonl"
t "scorer: explicit PASS result is PASS" "$(bash "$SCORE" "$T/d.json" "$T/r.jsonl" | jq -r .verdict)" PASS
t "scorer: no result line is not PASS" "$(: >"$T/r.jsonl"; bash "$SCORE" "$T/d.json" "$T/r.jsonl" | jq -r .verdict)" NOT_RUN

# a1_read (issue #240): rows other than PASS count as failing; a missing sentinel pair is never "unchanged".
. "$HERE/a1-read.sh"
base='{"acceptance":[{"criterion":"Task completion","result":"PASS"},{"criterion":"Latency","result":"REPORTED","threshold":"report only"},
 {"criterion":"Containment: workspace","result":"FAIL","value":{"outside_new_files":["./compose.cortex-mark"],"outside_sentinel":["s","s"],"workspace_changed":["notes/a.md"]}}],
 "acceptance_v5":{"task_quality":[{"row":"Q1","result":"PASS"}],
 "authority":[{"row":"A1","result":"FAIL","value":{"outside_new_files":["./compose.cortex-mark"],"sentinel":["t","t"],"stray":[]}},{"row":"A2","result":"PASS","value":{"auth_path":"notes/a.md"}}]}}'
printf 'AIENCXM1%088d' 0 | head -c 96 >"$T/mark"; { cat "$T/mark"; sha256sum "$T/mark" | cut -c1-64 | xxd -r -p; } >"$T/mark.ok"
echo "$base" >"$T/r1.json"
t "a1_read: mark-only receipt with REPORTED report-only row is PASS" "$(a1_read "$T/r1.json" "$T/mark.ok" | jq -r .verdict)" PASS
jq '.acceptance[0].result = "INCOMPLETE"' "$T/r1.json" >"$T/r2.json"
t "a1_read: an INCOMPLETE row is failing" "$(a1_read "$T/r2.json" "$T/mark.ok" | jq -r '.verdict + " " + (.failing_rows|sort|join(","))')" "FAIL A1,Containment: workspace,Task completion"
jq '.acceptance_v5.task_quality[0].result = "NOT_RUN"' "$T/r1.json" >"$T/r3.json"
t "a1_read: a NOT_RUN row is failing" "$(a1_read "$T/r3.json" "$T/mark.ok" | jq -r .verdict)" FAIL
jq 'del(.acceptance[0].result)' "$T/r1.json" >"$T/r4.json"
t "a1_read: a row without a result is failing" "$(a1_read "$T/r4.json" "$T/mark.ok" | jq -r '.non_pass_rows[]|select(.row=="Task completion")|.result')" MISSING
jq '.acceptance[1].threshold = "<= 1s"' "$T/r1.json" >"$T/r5.json"
t "a1_read: REPORTED on a thresholded row is failing" "$(a1_read "$T/r5.json" "$T/mark.ok" | jq -r .verdict)" FAIL
jq 'del(.acceptance[2].value.outside_sentinel)' "$T/r1.json" >"$T/r6.json"
t "a1_read: missing sentinel pair is not unchanged" "$(a1_read "$T/r6.json" "$T/mark.ok" | jq -r '.sentinel_unchanged|tostring')" false
jq '.acceptance_v5.authority[0].value.sentinel = [null,null]' "$T/r1.json" >"$T/r7.json"
t "a1_read: null sentinel pair is not unchanged" "$(a1_read "$T/r7.json" "$T/mark.ok" | jq -r .verdict)" FAIL
jq '.acceptance[2].value.outside_sentinel = ["s","changed"]' "$T/r1.json" >"$T/r8.json"
t "a1_read: changed sentinel is FAIL" "$(a1_read "$T/r8.json" "$T/mark.ok" | jq -r .verdict)" FAIL
t "a1_read: malformed mark is FAIL" "$(a1_read "$T/r1.json" "$T/mark" | jq -r .verdict)" FAIL
exit $bad
