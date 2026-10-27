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

# OPEN-MODEL-QWEN3 v8 feature checks (SPEC-v8.md Section 3). CPU only, shell + jq, reads run records only. Required for a v8
# campaign PASS in addition to the 334 declared rows (they add requirements; no row or threshold changes).
#   v8-checks.sh RUN_BASE V7_S3_DIR LAUNCH...
#   RUN_BASE   the v8 run base (<L>/s3-report.json per launch, R1.reference.json)
#   V7_S3_DIR  evidence-v7/run/s3-reports (the v7 GB10 records, s3-report-<L>.json)
# C1 shown:     attempt 1 shows nothing; attempt k>1 shows exactly attempt k-1's reply (shown_refused_text_sha256 == its
#               text_sha256) when attempt k-1 was refused with a reply, and nothing after a timeout or an error.
# C2 never kept: a launch has at most one parsed attempt, it is the last one, and its reply equals no refused reply of the launch.
# C3 limits:    at most 3 attempts, numbered 1..n.
# C4 replay:    R1.reference.json says allen "default" and its prompt_tokens and prompt_ids_sha256 equal the accepted R1 attempt's
#               (exact token identity; the declared row R1-P checks the same pair).
# C5 unchanged: attempt 1's prompt_tokens and prompt_ids_sha256 equal the v7 record of the same launch (only retries changed).
# N1 has no model attempt (refused before any model call): C1 to C3 hold vacuously, C5 requires both records to have no attempt.
# Prints one line per failure and "V8 CHECKS PASS" or "V8 CHECKS FAIL"; exit 0 or 1; exit 2 on usage.
set -u
BASE=${1:?RUN_BASE}; V7=${2:?V7_S3_DIR}; shift 2; [ $# -gt 0 ] || { echo "usage: v8-checks.sh RUN_BASE V7_S3_DIR LAUNCH..." >&2; exit 2; }
fail=0; bad() { echo "FAIL $*"; fail=1; }
for L in "$@"; do
  S=$BASE/$L/s3-report.json; O=$V7/s3-report-$L.json
  [ -f "$S" ] || { bad "$L: no s3-report.json"; continue; }
  A=$(jq -c '.proposal_attempts // []' "$S")
  # C1
  jq -r '. as $a | range(0; length) as $k | $a[$k] as $x
    | (if $k == 0 then null
       elif ($a[$k-1].outcome == "refused" and ($a[$k-1].text_sha256 // null) != null) then $a[$k-1].text_sha256
       else null end) as $want
    | select(($x.shown_refused_text_sha256 // null) != $want)
    | "attempt \($x.attempt) shown=\($x.shown_refused_text_sha256 // "none") expected=\($want // "none")"' <<<"$A" |
    while read -r m; do echo "FAIL $L C1 $m"; done | grep . && fail=1
  # C2
  jq -e '([.[] | select(.outcome == "parsed")] | length) as $p | (last // {}) as $z
    | ($p == 0) or ($p == 1 and ($z.outcome == "parsed")
       and ([.[] | select(.outcome == "refused") | .text_sha256] | index($z.text_sha256) | not))' <<<"$A" >/dev/null ||
    bad "$L C2: more than one parsed attempt, a parsed attempt that is not last, or an accepted reply equal to a refused one"
  # C3
  jq -e 'length <= 3 and ([.[].attempt] == [range(1; length + 1)])' <<<"$A" >/dev/null || bad "$L C3: attempts $(jq -c '[.[].attempt]' <<<"$A")"
  # C5
  if [ -f "$O" ]; then
    n8=$(jq -c '[.[0] // {} | .prompt_tokens, .prompt_ids_sha256]' <<<"$A"); n7=$(jq -c '[.proposal_attempts[0]? // {} | .prompt_tokens, .prompt_ids_sha256]' "$O")
    [ "$n8" = "$n7" ] || bad "$L C5: attempt 1 prompt $n8, v7 $n7"
  else bad "$L C5: no v7 record $O"; fi
  # C4
  if [ "$L" = R1 ]; then
    R=$BASE/R1.reference.json
    if [ -f "$R" ]; then
      acc=$(jq -c '[.[] | select(.outcome == "parsed")] | last // {} | [.prompt_tokens, .prompt_ids_sha256]' <<<"$A")
      ref=$(jq -c '[.prompt_tokens, .prompt_ids_sha256]' "$R")
      [ "$(jq -r '.allen // "missing"' "$R")" = default ] || bad "R1 C4: reference allen=$(jq -r '.allen // "missing"' "$R"), expected default"
      [ "$acc" = "$ref" ] || bad "R1 C4: accepted attempt prompt $acc, CPU reference $ref"
    else bad "R1 C4: no R1.reference.json"; fi
  fi
done
[ $fail = 0 ] && echo "V8 CHECKS PASS" || echo "V8 CHECKS FAIL"
exit $fail
