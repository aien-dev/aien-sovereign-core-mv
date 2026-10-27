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

# Negative tests for scoring contract v5 (SCORING-v5.md). Shell + jq only.
# Usage: test-scoring-v5.sh   Exit 0 only if every fixture gives its expected verdict.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
S=$HERE/score-rows.sh T=$HERE/tests-v5
pass=0 fail=0
# Scratch files go to a private temp dir, never into the tree: aien-test runs gates in parallel in one
# checkout, and a file here, even for a moment, fails another gate's clean-tree check.
W=$(mktemp -d) || exit 1
trap 'rm -rf "$W"' EXIT
# expect NAME FIXTURE CAMPAIGN_VERDICT EXPECTED_ROW_VERDICTS(ctl,A,B) [EXTRA_JQ_TEST]
expect() {
  local name=$1 fx=$2 want=$3 rows=$4 out got grows
  out=$("$S" "$T/decl.json" "$T/$fx.jsonl" 2>/dev/null); local rc=$?
  got=$(jq -r .verdict <<<"$out"); grows=$(jq -r '[.rows[].verdict] | join(",")' <<<"$out")
  local wantrc=1; [ "$want" = PASS ] && wantrc=0
  if [ "$got" = "$want" ] && [ "$grows" = "$rows" ] && [ "$rc" = "$wantrc" ]; then pass=$((pass + 1)); echo "ok   $name: $got ($grows) exit $rc"
  else fail=$((fail + 1)); echo "FAIL $name: want $want ($rows) exit $wantrc, got $got ($grows) exit $rc"; fi
}
expect "positive: every rep present, NA declared with reason"  pass              PASS       "PASS,PASS,NOT_APPLICABLE"
expect "missing required run (A has 2 of 3)"                  missing-run       INCOMPLETE "PASS,INCOMPLETE,PASS"
expect "duplicate run id (A rep 2 twice)"                     duplicate-run     ERROR      "PASS,ERROR,NOT_APPLICABLE"
expect "failed repetition (A rep 2)"                          failed-run        FAIL       "PASS,FAIL,PASS"
expect "incomplete run record (B rep 2 has no verdict)"       incomplete-run    INCOMPLETE "PASS,PASS,INCOMPLETE"
expect "NOT_APPLICABLE on undeclared case (A)"                na-undeclared     FAIL       "PASS,FAIL,PASS"
expect "control mismatch (ctl FAIL)"                          control-mismatch  FAIL       "FAIL,NOT_RUN,NOT_RUN"
expect "NOT_APPLICABLE without a reason"                      na-no-reason      INCOMPLETE "PASS,PASS,INCOMPLETE"
expect "NOT_APPLICABLE mixed with PASS never satisfies a rep" na-mixed          INCOMPLETE "PASS,PASS,INCOMPLETE"
# extra runs are reported, not counted, and do not change the verdict
out=$("$S" "$T/decl.json" "$T/extra-runs.jsonl" 2>/dev/null); rc=$?
if [ "$rc" = 0 ] && [ "$(jq -r .verdict <<<"$out")" = PASS ] && [ "$(jq '.extras | length' <<<"$out")" = 2 ] \
   && [ "$(jq '.rows[] | select(.row == "A") | .counted_reps' <<<"$out")" = 3 ]; then
  pass=$((pass + 1)); echo "ok   extra unrequested runs reported (2), not counted"
else fail=$((fail + 1)); echo "FAIL extra runs: rc $rc"; fi
# external control F0 absent counts as false (fail closed)
printf '%s\n' '{"contract":"scoring-v5","campaign":"x","rows":[{"row":"Q","role":"case","reps":1,"expected":"PASS","control":"F0","not_applicable":false}]}' >"$W/.f0.decl"
printf '%s\n' '{"row":"Q","rep":1,"verdict":"PASS","outcome":"ok"}' >"$W/.f0.res"
echo '{"F0":true}' >"$W/.f0.ext"
a=$("$S" "$W/.f0.decl" "$W/.f0.res" 2>/dev/null | jq -r .verdict); b=$("$S" "$W/.f0.decl" "$W/.f0.res" "$W/.f0.ext" 2>/dev/null | jq -r .verdict)
rm -f "$W"/.f0.decl "$W"/.f0.res "$W"/.f0.ext
if [ "$a" = NOT_RUN ] && [ "$b" = PASS ]; then pass=$((pass + 1)); echo "ok   external control F0 absent NOT_RUN, present PASS"
else fail=$((fail + 1)); echo "FAIL external F0: absent=$a present=$b"; fi
# a bad declaration is an error, never a PASS
echo '{"contract":"scoring-v5","campaign":"x","rows":[{"row":"Q","reps":0,"expected":"PASS"}]}' >"$W/.bad.decl"
v=$("$S" "$W/.bad.decl" "$T/pass.jsonl" 2>/dev/null | jq -r .verdict); rm -f "$W/.bad.decl"
if [ "$v" = ERROR ]; then pass=$((pass + 1)); echo "ok   bad declaration (reps 0) is ERROR"; else fail=$((fail + 1)); echo "FAIL bad declaration: $v"; fi
# the real NEXT-PHASE-2 v4 results (receipt e401fffe) scored with the v4 declaration
R=$HERE/../next-phase-2/e401fffe189506ecc0152f82bfe895c1374842746ae7cf4b5ac83462ccafe13c.json
tmp=$(mktemp -d); jq -c '(.controls + .rows)[]' "$R" >"$tmp/res.jsonl"
jq -c '{F0: .fixture.s3_committed}' "$R" >"$tmp/ext.json"
out=$("$S" "$HERE/declarations/np2-v4.decl.json" "$tmp/res.jsonl" "$tmp/ext.json" 2>/dev/null); rc=$?
if [ "$rc" = 0 ] && [ "$(jq -r .verdict <<<"$out")" = PASS ] && [ "$(jq -c .not_applicable <<<"$out")" = '["C6d"]' ]; then
  pass=$((pass + 1)); echo "ok   v4 receipt e401fffe inputs: PASS (C6d NOT_APPLICABLE, 33 rows PASS)"
else fail=$((fail + 1)); echo "FAIL v4 inputs: rc $rc $(jq -c '{verdict}' <<<"$out")"; fi
# the same v4 runs with one repetition dropped must not PASS
jq -c 'select(.row != "C7c" or .rep != 3)' "$tmp/res.jsonl" >"$tmp/short.jsonl"
v=$("$S" "$HERE/declarations/np2-v4.decl.json" "$tmp/short.jsonl" "$tmp/ext.json" 2>/dev/null | jq -r .verdict); rm -rf "$tmp"
if [ "$v" = INCOMPLETE ]; then pass=$((pass + 1)); echo "ok   v4 inputs minus C7c rep 3: INCOMPLETE (old scorer would PASS)"
else fail=$((fail + 1)); echo "FAIL v4 short: $v"; fi
echo "scoring-v5 tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
