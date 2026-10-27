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

# OPEN-MODEL-QWEN3 v7 gate G6 (SPEC-v7-DRAFT.md Section 2): the dry run must prove the harness, not the model.
#   g6-gate.sh DRY_RESULTS.jsonl DECLARATION.json DRY_RUN_BASE LAUNCH...      exit 0 = G6 PASS, 1 = G6 FAIL (reasons on stdout)
# For every launch named on the command line (D1 D2 D3 for v7):
#  1. $BASE/$L/run.json exists; steps S1, S2, S7, S8 are ok; the daemon logged "ALLEN: engaged".
#  2. Always required (product rows; independent of the model's text): every declared row of the launch whose id ends in
#     -PRE -OPR -BE -PIN, and "v1 Identity", "v1 Human interventions: rescues", "v1 Containment: speculation",
#     "v1 Correctness: recall".
#  2a. -DEC and -MEAS are product rows too, but by the frozen v5 rules an attempt the product ended by timeout has no decoding
#     and no token ids, so both rows FAIL whenever the model was too slow for the task budget (v7 dry attempt 2, D3). They are
#     required unless $BASE/$L/s3-report.json shows an attempt with outcome "timeout"; then they are listed NOT CHECKED (model
#     timeout) with a NOTE line. The timeout itself is a model finding for the campaign, not a harness fault.
#  3. If the launch committed (S4 and S5 ok), every other declared row of the launch must be PASS too, except the four
#     model-behaviour rows -RQ<k>, -CUT, -F, -SB (v6 failure class (a)), which are listed but may FAIL. "v1 Memory" is in this
#     group: a refused reply writes no memory record, so the row FAILs on every uncommitted launch (v6 class (a); v7 dry attempt 2
#     D1 and D3) and only a committed launch can show it. If the launch did not commit, only 1, 2 and 2a apply and the rest is
#     listed as NOT CHECKED (model).
#  4. A required row missing from the results is a FAIL.
set -u
RES=${1:?DRY_RESULTS.jsonl} DECL=${2:?DECLARATION.json} BASE=${3:?DRY_RUN_BASE}; shift 3
[ $# -gt 0 ] || { echo "g6-gate: give the launch ids" >&2; exit 2; }
fail=0
say() { echo "$*"; }
for L in "$@"; do
  rj=$BASE/$L/run.json
  if [ ! -f "$rj" ]; then say "FAIL $L no run.json"; fail=1; continue; fi
  for s in S1 S2 S7 S8; do
    [ "$(jq -r --arg s $s '[.steps[] | select(.step == $s)][0].ok // false' "$rj")" = true ] || { say "FAIL $L step $s not ok"; fail=1; }
  done
  jq -e '(.allen.engaged_log_line // "") | startswith("ALLEN: engaged")' "$rj" >/dev/null || { say "FAIL $L daemon did not log ALLEN: engaged"; fail=1; }
  committed=$(jq -r '([.steps[] | select(.step == "S4" or .step == "S5") | .ok] | length == 2 and all)' "$rj")
  timeouts=0
  [ -f "$BASE/$L/s3-report.json" ] && timeouts=$(jq -r '[.proposal_attempts[]? | select(.outcome == "timeout") | .attempt] | join(",")' "$BASE/$L/s3-report.json")
  [ "$timeouts" = 0 ] || [ -z "$timeouts" ] && timeouts=0
  [ "$timeouts" = 0 ] || say "NOTE $L attempt $timeouts ended by timeout (model too slow for the task budget): -DEC -MEAS not checked, campaign finding"
  nreq=0 nok=0 nskip=0 ntimeout=0
  while IFS= read -r row; do
    id=${row#"$L/"}
    case $id in
      "$L"-PRE|"$L"-OPR|"$L"-BE|"$L"-PIN|"v1 Identity"|"v1 Human interventions: rescues"|"v1 Containment: speculation"|"v1 Correctness: recall") need=always ;;
      "$L"-DEC|"$L"-MEAS) need=notimeout ;;
      "$L"-RQ*|"$L"-CUT|"$L"-F|"$L"-SB) need=model ;;
      "v1 Memory") need=committed ;;
      *) need=committed ;;
    esac
    if [ $need = notimeout ] && [ "$timeouts" != 0 ]; then ntimeout=$((ntimeout + 1)); continue; fi
    if [ $need = model ] || { [ $need = committed ] && [ "$committed" != true ]; }; then nskip=$((nskip + 1)); continue; fi
    nreq=$((nreq + 1))
    v=$(jq -r --arg r "$row" 'select(.row == $r) | .verdict' "$RES" | head -1)
    if [ "$v" = PASS ]; then nok=$((nok + 1)); else say "FAIL $L row '$id' = ${v:-missing}"; fail=1; fi
  done < <(jq -r --arg l "$L" '.rows[] | select(.row | startswith($l + "/")) | .row' "$DECL")
  say "$L committed=$committed required=$nreq pass=$nok not_checked_model=$nskip not_checked_timeout=$ntimeout"
done
[ $fail = 0 ] && say "G6 PASS" || say "G6 FAIL"
exit $fail
