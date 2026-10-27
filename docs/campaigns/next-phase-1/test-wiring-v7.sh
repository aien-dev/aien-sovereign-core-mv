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

# NEXT-PHASE-1 v7: the v7 harness wiring (ACCEPTANCE-v7 Section 2). Shell + jq
# only. Usage: test-wiring-v7.sh. Exit 0 only if every case holds.
#
# 1. The real v4 run (tests-v5/v4-run) with its S3 report set to an explicit
#    "committed": false, through make-receipt.sh as N1: the default module
#    (rows-v6.jq, every existing caller) still turns false into null, and
#    V6_ROWS=rows-v7.jq keeps it false and records the module and its sha256.
#    Any other V6_ROWS is refused.
# 2. run-v7.sh differs from run-v6.sh only in the v7 wiring lines.
# 3. np1-v7.decl.json declares the same 73 rows as np1-v6.decl.json.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
pass=0 fail=0
check() {   # check NAME EXPECTED ACTUAL
  if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1 (got: ${3:-none})"
  else fail=$((fail + 1)); echo "FAIL $1: expected [${2}] got [${3}]"; fi
}

# ---- 1. make-receipt.sh with and without V6_ROWS --------------------------------
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
tmp=$(cd "$tmp" && pwd -P)
cp -R "$HERE/tests-v5/v4-run" "$tmp/run"
for f in "$tmp"/run/run.json "$tmp"/run/steps/*.json; do sed -i "s#@RUN@#$tmp/run#g" "$f"; done
jq '.report.committed = false' "$tmp/run/steps/S3.json" >"$tmp/s3" && cp "$tmp/s3" "$tmp/run/steps/S3.json"
rec() {   # rec [extra env...] -> receipt path
  local o; o=$(mktemp -d "$tmp/out.XXXX")
  env "$@" V6_TASK=N1 V6_MAX_TOKENS=48 bash "$HERE/make-receipt.sh" "$tmp/run" "$o" t t t 0 >"$o.sum" 2>"$o.err" \
    || { echo "make-receipt.sh failed:"; cat "$o.err"; exit 1; }
  ls "$o"/*.json
}
c3() { jq -r '[.acceptance_v6[] | select(.row == "N1-C")][0].value.s3_committed' "$1"; }
r6=$(rec); r7=$(rec V6_ROWS=rows-v7.jq)
check "default module: explicit false read as null (the v6 defect, unchanged)" "null" "$(c3 "$r6")"
check "default module: no v6_rows_module field" "null" "$(jq -r .v6_rows_module "$r6")"
check "V6_ROWS=rows-v7.jq: explicit false stays false" "false" "$(c3 "$r7")"
check "V6_ROWS=rows-v7.jq: receipt records the module" \
  "rows-v7.jq $(sha256sum "$HERE/rows-v7.jq" | cut -d' ' -f1)" "$(jq -r '"\(.v6_rows_module.file) \(.v6_rows_module.sha256)"' "$r7")"
check "V6_ROWS=rows-v7.jq: legacy verdict and other v6 rows as the default module" \
  "$(jq -c '[.verdict, (.acceptance_v6 | map(select(.row != "N1-C")) | map({row, result}))]' "$r6")" \
  "$(jq -c '[.verdict, (.acceptance_v6 | map(select(.row != "N1-C")) | map({row, result}))]' "$r7")"
check "V6_ROWS=anything else: refused" "refused" \
  "$(V6_ROWS=../rows-v6.jq V6_TASK=N1 bash "$HERE/make-receipt.sh" "$tmp/run" "$tmp/out-bad" t t t 0 >/dev/null 2>&1 && echo accepted || echo refused)"

# ---- 2. run-v7.sh is run-v6.sh plus the v7 wiring ------------------------------
check "run-v7.sh: only the v7 wiring lines differ from run-v6.sh" "5c5 7c7 10c10 29,31c29,31 42,43c42,43" \
  "$(diff <(sed -n '12,$p' "$HERE/run-v6.sh") <(sed -n '14,$p' "$HERE/run-v7.sh") | grep -E '^[0-9]' | paste -sd' ')"
check "run-v7.sh: v7 module, declaration and spec" "1 1 1" \
  "$(for p in 'V6_ROWS=rows-v7.jq V6_TASK=' 'declarations/np1-v7.decl.json' 'SPEC="ACCEPTANCE-v7.md spec_version 7"'; do grep -cF -- "$p" "$HERE/run-v7.sh"; done | paste -sd' ')"

# ---- 3. the v7 declaration ------------------------------------------------------
D=$HERE/../scoring/declarations
check "np1-v7.decl.json: same 73 rows as np1-v6" "73 true" \
  "$(jq -r --slurpfile v6 "$D/np1-v6.decl.json" '"\(.rows | length) \([.rows[].row] == [$v6[0].rows[].row])"' "$D/np1-v7.decl.json")"

echo "test-wiring-v7: $pass passed, $fail failed"
[ "$fail" = 0 ]
