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

# CAND-4 production hygiene (ACCEPTANCE-CAND4.md section 3, rows H1..H6). Shell + jq only.
# No GPU use: the omega host program's ARGUS probe never starts the graphics processor
# (omega tools/r16_prod_hygiene.sh step 4); the silicon programs are checked statically.
# Usage: hygiene.sh INPUTS OUT_DIR
#   INPUTS  ACCEPTANCE-CAND4.md (qualification) or a TRIAL inputs file with the same block
# Writes OUT_DIR/hygiene-<sha256>.json (named by its content) and prints it. Exit 0 iff PASS.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
INPUTS=${1:?INPUTS} OUT=${2:?OUT_DIR}
. "$HERE/inputs.sh"
load_inputs "$INPUTS" || exit 3
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/cand4-hygiene.XXXXXX"); trap 'rm -rf "$W"' EXIT
CLI=$(inp aien_cli_path) CPU=$(inp cpu_fault_path) EVD=$(inp build_evidence) OB=$(inp omega_build_dir)
verify_digests || exit 3
ROWS='[]'
row() { ROWS=$(jq -c --arg r "$1" --arg c "$2" --arg v "$3" --argjson e "$4" '. + [{row:$r, check:$c, verdict:$v, evidence:$e}]' <<<"$ROWS"); }
pv() { if [ "$1" = 1 ]; then echo PASS; else echo FAIL; fi; }

# H1 / H1c: fault hook strings (the patterns of NEXT-PHASE-2 link_proof, plus the hook names)
PAT='AIEN_FAULT_HOLD|fault hold|reconcile_panic|reconcile_error|forced start-up reconcile'
strings -a "$CLI" >"$W/cli.str"; strings -a "$CPU" >"$W/cpu.str"
n1=$(grep -Ec "$PAT" "$W/cli.str"); n1c=$(grep -Ec "$PAT" "$W/cpu.str")
row H1 "production aien-cli: 0 lines matching /$PAT/" "$(pv $([ "$n1" = 0 ] && echo 1))" \
  "$(jq -nc --argjson n "$n1" --arg s "$(grep -E "$PAT" "$W/cli.str" | head -3)" '{matching_lines:$n, first:$s}')"
row H1c "positive control: cpu-fault binary carries the same patterns" "$(pv $([ "$n1c" -gt 0 ] && echo 1))" \
  "$(jq -nc --argjson n "$n1c" '{matching_lines:$n}')"

# H2 / H2c: no test feature on any rustc line of the production -vv release log
L=$EVD/sc-rel-2.log
if [ -f "$L" ]; then
  f1=$(grep -c 'feature=\\\?"fault-hold\\\?"' "$L"); f2=$(grep -c 'feature=\\\?"dev-fallback\\\?"' "$L")
  row H2 "production -vv build log: no feature fault-hold or dev-fallback" "$(pv $([ "$f1" = 0 ] && [ "$f2" = 0 ] && echo 1))" \
    "$(jq -nc --arg l "$L" --arg sha "$(sha256sum "$L" | cut -d' ' -f1)" --argjson a "$f1" --argjson b "$f2" '{log:$l, sha256:$sha, fault_hold_lines:$a, dev_fallback_lines:$b}')"
else
  row H2 "production -vv build log present" FAIL "$(jq -nc --arg l "$L" '{missing:$l}')"
fi
CL=$(dirname "$CPU")/build.log
if [ -f "$CL" ]; then
  c1=$(grep -c 'feature=\\\?"fault-hold\\\?"' "$CL")
  row H2c "positive control: the cpu-fault -vv log shows feature fault-hold" "$(pv $([ "$c1" -gt 0 ] && echo 1))" "$(jq -nc --argjson a "$c1" '{fault_hold_lines:$a}')"
else
  row H2c "positive control: cpu-fault build log present" FAIL "$(jq -nc --arg l "$CL" '{missing:$l}')"
fi

# H3: no omega test-build piece in aien-cli (stripped: strings) nor in librx_compose.a (nm)
t1=$(grep -c 'AIEN_TEST_BUILD piece:' "$W/cli.str")
LIB=$OB/librx_compose.a
if [ -f "$LIB" ]; then
  t2=$(nm "$LIB" 2>/dev/null | awk '{print $NF}' | grep -c '^aien_test_build_')
  t3=$(strings -a "$LIB" | grep -c 'AIEN_TEST_BUILD piece:')
  row H3 "no AIEN_TEST_BUILD piece in aien-cli or librx_compose.a" "$(pv $([ "$t1" = 0 ] && [ "$t2" = 0 ] && [ "$t3" = 0 ] && echo 1))" \
    "$(jq -nc --argjson a "$t1" --argjson b "$t2" --argjson c "$t3" --arg s "$(sha256sum "$LIB" | cut -d' ' -f1)" '{aien_cli_marker_strings:$a, lib_test_symbols:$b, lib_marker_strings:$c, lib_sha256:$s}')"
else
  row H3 "librx_compose.a present in omega_build_dir" FAIL "$(jq -nc --arg l "$LIB" '{missing:$l}')"
fi

# H4 / H4c / H5: omega's own tools at the frozen omega commit, from a scratch copy
OSRC=$(inp omega_src_dir); OC=$(inp omega_commit); AO=$(inp aienos_repo); PH=$(inp physics_dir)
mkdir -p "$W/omega"
git -C "$OSRC" archive "$OC" | tar -x -C "$W/omega" || { row H4 "omega source at omega_commit" FAIL '{"error":"git archive failed"}'; }
if [ -f "$W/omega/tools/r16_prod_hygiene.sh" ]; then
  (cd "$W/omega" && sh tools/r16_prod_hygiene.sh host "$OB/rx_r13_living_host") >"$W/h4a.out" 2>&1; a=$?
  (cd "$W/omega" && R16_PROD_NO_RUN=1 sh tools/r16_prod_hygiene.sh silicon "$OB/rx_r13_living_silicon") >"$W/h4b.out" 2>&1; b=$?
  row H4 "omega production programs: host (full, ARGUS probe) and silicon (static)" "$(pv $([ $a = 0 ] && [ $b = 0 ] && echo 1))" \
    "$(jq -nc --argjson a $a --argjson b $b --arg x "$(tail -2 "$W/h4a.out")" --arg y "$(tail -2 "$W/h4b.out")" '{host_rc:$a, host:$x, silicon_rc:$b, silicon:$y}')"
  (cd "$W/omega" && R16_PROD_NO_RUN=1 sh tools/r16_prod_hygiene.sh silicon "$OB/rx_r13_living_testbuild_silicon") >"$W/h4c.out" 2>&1; c=$?
  row H4c "negative control: the test-build silicon program fails the same check" "$(pv $([ $c != 0 ] && grep -q 'PROD_HYGIENE=FAIL' "$W/h4c.out" && echo 1))" \
    "$(jq -nc --argjson c $c --arg x "$(head -1 "$W/h4c.out")" '{rc:$c, first_line:$x}')"
  wait_quiet "omega make (H5)"
  ARGUS_REPO=$AO make -C "$W/omega" OUT_DIR="$W/out" PHYSICS_DIR="$PH" ARGUS_REPO="$AO" AIENOS_LOCK_REPO="$AO" \
    test-prod-refuses-test-pieces >"$W/h5.out" 2>&1; d=$?
  row H5 "make test-prod-refuses-test-pieces at omega_commit (scratch copy)" "$(pv $([ $d = 0 ] && echo 1))" \
    "$(jq -nc --argjson d $d --arg x "$(grep -E 'R16 refuse' "$W/h5.out" | tail -12)" '{rc:$d, lines:$x}')"
fi

# H6: report only
names=$(grep -Eo 'AIEN_[A-Z0-9_]*(TEST|STUB|FAULT|FALLBACK|DEV)[A-Z0-9_]*' "$W/cli.str" | sort -u | jq -R . | jq -sc .)
row H6 "report only: AIEN_* names with TEST, STUB, FAULT, FALLBACK or DEV in aien-cli" REPORTED "$(jq -nc --argjson n "$names" '{names:$n}')"

tmp=$(mktemp "$OUT/.hygiene.XXXXXX")
jq -n --argjson rows "$ROWS" --arg kind "$(inp kind)" --arg cand "$(inp candidate_id)" --arg sc "$(inp sc_commit)" \
  --arg oc "$OC" --arg cli "$(inp aien_cli_sha256)" --arg cpu "$(inp cpu_fault_sha256)" \
  '{gate:"CAND4_HYGIENE", kind:$kind, candidate:$cand, sc_commit:$sc, omega_commit:$oc,
    aien_cli_sha256:$cli, cpu_fault_sha256:$cpu, acceptance_spec:"ACCEPTANCE-CAND4.md section 3",
    verdict:(if ($rows | map(select(.verdict != "PASS" and .verdict != "REPORTED")) | length) == 0 then "PASS" else "FAIL" end),
    rows:$rows}' >"$tmp"
h=$(sha256sum "$tmp" | cut -d' ' -f1); mv "$tmp" "$OUT/hygiene-$h.json"
jq -r '"hygiene-'"$h"'.json verdict \(.verdict)", (.rows[] | "  \(.verdict)  \(.row)  \(.check)")' "$OUT/hygiene-$h.json"
[ "$(jq -r .verdict "$OUT/hygiene-$h.json")" = PASS ]
