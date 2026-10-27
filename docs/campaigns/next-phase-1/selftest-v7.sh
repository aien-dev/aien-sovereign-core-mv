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

# NEXT-PHASE-1 v7 self-test of rows-v7.jq: the S3 committed flag (VERDICT-v6.md Section 2).
# Shell + jq only. Exit 0 only if every case gives the expected result.
# Runs v6_evidence on synthetic S3 reports, splices the resulting s3.committed into the
# frozen good evidence (tests-v6/good-*.evidence.json, read only) and scores the rows.
# Case 1 also runs the v6 module to show the defect (red) before the v7 fix (green).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$HERE/tests-v6
V7=$(cat "$HERE/rows-v7.jq"); V6=$(cat "$HERE/rows-v6.jq")
pass=0 fail=0
check() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1 (got: ${3:-none})"
  else fail=$((fail + 1)); echo "FAIL $1: expected [${2}] got [${3}]"; fi; }
# committed_of MODULE REPORT_JSON -> the s3.committed value v6_evidence produces
committed_of() { jq -nc "$1"'
v6_evidence(null; '"$2"'; null; null; null; null; null; null; []; {}; "/ws"; null; null; null; "") | .s3.committed'; }
# fails_with MODULE LAUNCH COMMITTED_JSON -> failing rows of the good evidence with s3.committed set
fails_with() { jq -c --argjson c "$3" '.s3.committed = $c' "$T/good-$2.evidence.json" \
  | jq -r "$1"'
v6_rows | [.[] | select(.result != "PASS") | "\(.row)\(if .result == "NOT_RUN" then ":NOT_RUN" else "" end)"] | join(",")'; }

# 1. refused, uncommitted run: report says committed=false
check "v6 module turns false into null (the defect)" "null" "$(committed_of "$V6" '{committed: false}')"
check "v7 keeps false as false" "false" "$(committed_of "$V7" '{committed: false}')"
check "v6 module: N1 uncommitted FAILs N1-C,N1-B (defect shown)" "N1-C,N1-B" "$(fails_with "$V6" N1 "$(committed_of "$V6" '{committed: false}')")"
check "v6 module: N2 uncommitted FAILs N2-F,N2-C (defect shown)" "N2-F,N2-C" "$(fails_with "$V6" N2 "$(committed_of "$V6" '{committed: false}')")"
check "v7: N1 refused uncommitted run scores PASS" "" "$(fails_with "$V7" N1 "$(committed_of "$V7" '{committed: false}')")"
check "v7: N2 refused uncommitted run scores PASS" "" "$(fails_with "$V7" N2 "$(committed_of "$V7" '{committed: false}')")"
# 2. committed=true scores as before
check "v7 keeps true as true" "true" "$(committed_of "$V7" '{committed: true}')"
check "v7: N1 committed run is not REFUSED" "N1-C,N1-B" "$(fails_with "$V7" N1 "$(committed_of "$V7" '{committed: true}')")"
check "v7: N2 committed run is not REFUSED" "N2-F,N2-C" "$(fails_with "$V7" N2 "$(committed_of "$V7" '{committed: true}')")"
for k in T4 T5 R1; do
  check "v7: $k good evidence with committed=true still all PASS" "" "$(fails_with "$V7" $k "$(committed_of "$V7" '{committed: true}')")"
  check "v6 and v7 agree on $k with committed=true" "$(fails_with "$V6" $k true)" "$(fails_with "$V7" $k true)"
done
# 3. missing field is null, not false, and never makes a negative launch pass
check "v7: report without committed gives null" "null" "$(committed_of "$V7" '{}')"
check "v7: null report gives null" "null" "$(committed_of "$V7" 'null')"
check "v7: N1 with missing committed is not PASS" "N1-C,N1-B" "$(fails_with "$V7" N1 "$(committed_of "$V7" '{}')")"
check "v7: N2 with missing committed is not PASS" "N2-F,N2-C" "$(fails_with "$V7" N2 "$(committed_of "$V7" '{}')")"
# 4. the module has no other boolean read through // (explicit false must survive)
check "no .committed read through //" "0" "$(grep -c 'committed // ' "$HERE/rows-v7.jq")"

echo "selftest-v7: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
