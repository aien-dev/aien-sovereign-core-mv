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

# test-crumb-real.sh <shim> <old-tool>: red/green on real Spark trees and on the real failing heads.
# GREEN needs the shim to agree with each repo's own gate where the old fixed tool did not.
set -u
SHIM=$(readlink -f "$1"); OLD=$(readlink -f "$2")
C=~/workspace/pipeline/clones; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
D0=~/.local/state/aien-local-ci/crumb-tools/d0f0b95d344b4c6628c3d868346932fd247ce168/crumb
GATE_SANDBOX=/usr/local/bin/crumb
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1: $2"; }
tree() {  # resolve the rev in the SOURCE clone (its origin/main), never the copy's idea of it
  local sha; sha=$(git -C "$C/$1" rev-parse --verify "$2^{commit}") || { echo "no $2 in $1"; return 1; }
  rm -rf "$T/$1"; git clone -q --shared --no-checkout "$C/$1" "$T/$1" && git -C "$T/$1" -c advice.detachedHead=false checkout -q --detach "$sha"; }
v() { (cd "$1" && "$2" verify . 2>&1 | tail -1); }
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }

# 1. readiness on mains that pin d0f0b95d: old tool says stale, shim says what the gate says
for r in omega aienos physics; do
  tree $r origin/main
  o=$(v "$T/$r" "$OLD"); s=$(v "$T/$r" "$SHIM"); gt=$(v "$T/$r" "$D0")
  [[ $o == *FAIL* ]] && [[ $s == *OK* ]] && [[ $gt == *OK* ]] && ok "$r main: old=FAIL shim=OK gate=OK" || bad "$r main" "old='$o' shim='$s' gate='$gt'"
done
# 2. migrated trees (RFC-0003 Am1 PR heads): old tool calls every crumb stale, shim agrees with the gate
for x in aegis-runtime:aeefa8dad09954129a06ee429128ae640cb5d0bd aien-protocols:083601ab spark-rsi:d272ecb7 aien-architecture:00034398; do
  r=${x%%:*}; tree $r "${x#*:}"
  o=$(v "$T/$r" "$OLD"); s=$(v "$T/$r" "$SHIM")
  [[ $o == *FAIL* ]] && [[ $s == *"OK"*"on demand"* ]] && ok "$r ${x#*:} (migrated): old=FAIL shim=OK" || bad "$r migrated" "old='$o' shim='$s'"
done
# 3. legacy repos without a lock: shim (sandbox tool) and old tool give the same verdict on main
for r in aien-sovereign-core aegis-runtime aien-protocols spark-rsi aien-architecture synapse; do
  tree $r origin/main
  o=$(v "$T/$r" "$OLD"); s=$(v "$T/$r" "$SHIM")
  [ "$o" = "$s" ] && ok "$r main legacy verdict unchanged by shim ($s)" || bad "$r main legacy" "old='$o' shim='$s'"
done
# 4. author replay on real failing heads: recompile the head's crumbs, then ask the GATE's tool
#    sc 2d16cb94: author compiled with d0f0b95d, sc gate (sandbox tool) said 1 stale (2026-10-16)
#    omega 05d0d8e9: author compiled with 94fffbcf, omega gate (crumb.lock d0f0b95d) said 10 stale
replay() {  # replay <repo> <sha> <gate-tool> <author-tool> <label>
  tree "$1" "$2"; (cd "$T/$1" && "$4" compile . >/dev/null 2>&1)
  v "$T/$1" "$3"
}
s=$(replay aien-sovereign-core 2d16cb94 "$GATE_SANDBOX" "$D0"); [[ $s == *FAIL* ]] && ok "sc 2d16cb94 RED: compiled with d0f0b95d, gate says '$s'" || bad "sc replay red" "$s"
s=$(replay aien-sovereign-core 2d16cb94 "$GATE_SANDBOX" "$SHIM"); [[ $s == *OK* ]] && ok "sc 2d16cb94 GREEN: compiled with shim, gate says '$s'" || bad "sc replay green" "$s"
s=$(replay omega 05d0d8e9 "$D0" "$OLD"); [[ $s == *FAIL* ]] && ok "omega 05d0d8e9 RED: compiled with old tool, gate says '$s'" || bad "omega replay red" "$s"
s=$(replay omega 05d0d8e9 "$D0" "$SHIM"); [[ $s == *OK* ]] && ok "omega 05d0d8e9 GREEN: compiled with shim, gate says '$s'" || bad "omega replay green" "$s"
# 5. after migration an old-tool compile is refused by the gate (the protocol's own guard): proves the
#    remaining failure mode is loud, not silent
tree aegis-runtime aeefa8dad09954129a06ee429128ae640cb5d0bd; (cd "$T/aegis-runtime" && "$OLD" compile . >/dev/null 2>&1)
s=$(v "$T/aegis-runtime" "$GATE_SANDBOX"); [[ $s == *FAIL* ]] && ok "old-tool compile on a migrated tree is caught by the gate ('$s')" || bad "migrated old compile" "$s"
tree aegis-runtime aeefa8dad09954129a06ee429128ae640cb5d0bd; (cd "$T/aegis-runtime" && "$SHIM" compile . >/dev/null 2>&1)
d=$(git -C "$T/aegis-runtime" status --porcelain | wc -l); s=$(v "$T/aegis-runtime" "$GATE_SANDBOX")
(( d == 0 )) && [[ $s == *OK* ]] && ok "shim compile on a migrated tree changes nothing and verifies" || bad "migrated shim compile" "dirty=$d '$s'"

echo "real-tree tests: $pass passed, $fail failed"; (( fail == 0 ))
