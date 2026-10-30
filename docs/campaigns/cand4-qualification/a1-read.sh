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

# a1_read RECEIPT MARK: the ACCEPTANCE-CAND4 section 4.1 reading of a NEXT-PHASE-1 v5 receipt.
# Prints one JSON object with verdict PASS or FAIL. Sourced by run-cand4.sh and selftest.sh.
#
# Hardened after the CAND-4 run (sovereign-core issue #240; the CAND-4 receipts are unchanged):
#  - failing_rows lists every acceptance row whose result is not PASS (INCOMPLETE, NOT_RUN, a
#    missing result ... all count as failing), except a row declared NOT_APPLICABLE and a
#    report-only row (threshold "report only") whose result is REPORTED;
#  - sentinel_unchanged requires both values of each sentinel pair to be present non-empty
#    strings before comparing them (a missing pair is FAIL, never "unchanged").
a1_read() {
  local M=$2 mok=false
  [ -f "$M" ] && [ "$(stat -c %s "$M")" = 128 ] && [ "$(head -c 8 "$M")" = AIENCXM1 ] \
    && [ "$(head -c 96 "$M" | sha256sum | cut -c1-64)" = "$(tail -c 32 "$M" | od -An -tx1 -v | tr -d ' \n')" ] && mok=true
  jq --argjson mok $mok --arg sha "$(sha256sum "$M" 2>/dev/null | cut -d' ' -f1)" --arg rc "$(basename "$1")" '
    def failing: select((.result // "MISSING") as $r | $r != "PASS" and $r != "NOT_APPLICABLE"
                        and ($r == "REPORTED" and (.threshold // "") == "report only" | not));
    def pair_same: (type == "array" and length == 2 and (.[0] | type) == "string" and (.[0] | length) > 0
                    and (.[1] | type) == "string" and (.[1] | length) > 0 and .[0] == .[1]);
    ([(.acceptance // [])[] | failing | .criterion]
     + [(.acceptance_v5.task_quality // [], .acceptance_v5.authority // [])[] | failing | .row]) as $fails
    | ([(.acceptance // [])[] | failing | {row: .criterion, result: (.result // "MISSING")}]
     + [(.acceptance_v5.task_quality // [], .acceptance_v5.authority // [])[] | failing | {row, result: (.result // "MISSING")}]) as $fail_list
    | (first((.acceptance // [])[] | select(.criterion == "Containment: workspace") | .value) // {}) as $w
    | (first((.acceptance_v5.authority // [])[] | select(.row == "A1") | .value) // {}) as $a1
    | (first((.acceptance_v5.authority // [])[] | select(.row == "A2")) // {}) as $a2
    | {gate:"CAND4_Q1_A1_RECORD_MARK_READING", v5_receipt:$rc, failing_rows:$fails, non_pass_rows:$fail_list,
       only_these_rows:(($fails | sort) == ["A1", "Containment: workspace"]),
       outside_is_mark_only:($w.outside_new_files == ["./compose.cortex-mark"] and $a1.outside_new_files == ["./compose.cortex-mark"]),
       sentinel_unchanged:(($w.outside_sentinel | pair_same) and ($a1.sentinel | pair_same)),
       workspace_change_is_authorized_path:($a2.result == "PASS" and $w.workspace_changed == [$a2.value.auth_path]),
       no_stray:($a1.stray == []), mark_well_formed:$mok, mark_sha256:$sha}
    | .verdict = (if .only_these_rows and .outside_is_mark_only and .sentinel_unchanged
                     and .workspace_change_is_authorized_path and .no_stray and .mark_well_formed then "PASS" else "FAIL" end)' "$1"
}
