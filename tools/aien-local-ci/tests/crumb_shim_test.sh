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

# test-crumb-shim.sh <shim>: unit tests for the operator crumb shim, on throwaway fixtures only.
# Each case asserts which tool ran (the fake tools print their name), so a wrong pick is RED.
set -u
SHIM=$(readlink -f "${1:?usage: test-crumb-shim.sh <shim>}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "ok   $1"; }
bad()  { fail=$((fail+1)); echo "FAIL $1: $2"; }
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }

# fake sandbox tool
printf '#!/bin/sh\necho SANDBOX "$@"\n' > "$T/sandbox"; chmod +x "$T/sandbox"
# fake aien-architecture clone: two tool commits that print their own name
mkdir -p "$T/arch/tools/crumb"; cd "$T/arch"; g init -q
mk() { printf 'crumb: crumb.c\n\t$(CC) -o $@ crumb.c\n' > tools/crumb/Makefile
       printf '#include <stdio.h>\nint main(int c,char**v){printf("%s");for(int i=1;i<c;i++)printf(" %%s",v[i]);printf("\\n");return 0;}\n' "$1" > tools/crumb/crumb.c; }
mk PIN-A; g add -A; g commit -qm A; A=$(git rev-parse HEAD)
mk PIN-B; g add -A; g commit -qm B; B=$(git rev-parse HEAD)
export CRUMB_SANDBOX_TOOL=$T/sandbox CRUMB_TOOL_CACHE=$T/cache CRUMB_ARCH_CLONE=$T/arch
repo() { mkdir -p "$T/$1"; (cd "$T/$1" && g init -q && echo '{}' > .crumb && g add -A && g commit -qm init); }

repo locked; echo "$A" > "$T/locked/crumb.lock"
repo lockedB; echo "  ${B^^}  " > "$T/lockedB/crumb.lock"          # whitespace + upper case tolerated
repo plain
repo badlock; echo "not-a-sha" > "$T/badlock/crumb.lock"
repo nopin; echo "0123456789012345678901234567890123456789" > "$T/nopin/crumb.lock"
repo treerepo; mkdir -p "$T/treerepo/tools/crumb"; (cd "$T/treerepo" && cp "$T/arch/tools/crumb/Makefile" tools/crumb/ \
  && sed 's/PIN-B/TREE-T/' "$T/arch/tools/crumb/crumb.c" > tools/crumb/crumb.c)
mkdir -p "$T/locked/sub/dir" "$T/notgit"

out() { (cd "$1" && shift && "$SHIM" "$@" 2>&1); }
expect() {  # expect <name> <expected-prefix> <dir> <args...>
  local name=$1 want=$2 dir=$3; shift 3
  local got; got=$(out "$dir" "$@"); [[ $got == "$want"* ]] && ok "$name" || bad "$name" "want '$want...', got '$got'"
}
expect "crumb.lock picks the pinned build"            "PIN-A verify ."        "$T/locked" verify .
expect "pin with spaces/upper case still resolves"   "PIN-B verify ."        "$T/lockedB" verify .
expect "subdirectory of a locked repo"               "PIN-A verify ."        "$T/locked/sub/dir" verify .
expect "path argument decides the repo, not cwd"     "PIN-A verify $T/locked" "$T/plain" verify "$T/locked"
expect "file argument resolves via its directory"    "PIN-A context"          "$T/notgit" context "$T/locked/.crumb"
expect "no lock, no tool source: sandbox tool"       "SANDBOX verify ."       "$T/plain" verify .
expect "not a git tree: sandbox tool"                "SANDBOX verify ."       "$T/notgit" verify .
expect "aien-architecture tree: its own tools/crumb" "TREE-T verify ."        "$T/treerepo" verify .
expect "worker's close call (non-path words skipped)" "PIN-A close aien-worker ." "$T/locked" close aien-worker . worker test "local verification finished"

# errors are loud and never fall back to another tool
r=$(out "$T/badlock" verify .); rc=$?
(( rc == 97 )) && [[ $r == *"not one 40-hex commit"* ]] && [[ $r != *SANDBOX* ]] && ok "malformed crumb.lock: exit 97, no fallback" || bad "malformed lock" "rc=$rc out=$r"
r=$(out "$T/nopin" verify .); rc=$?
(( rc == 97 )) && [[ $r == *"is not in"* ]] && [[ $r != *SANDBOX* ]] && ok "pin missing from the arch clone: exit 97, no fallback" || bad "missing pin" "rc=$rc out=$r"
r=$(CRUMB_SANDBOX_TOOL=$T/none out "$T/plain" verify .); rc=$?
(( rc == 97 )) && ok "missing sandbox tool: exit 97" || bad "missing sandbox" "rc=$rc out=$r"

# exit status and stdout pass through untouched
printf '#!/bin/sh\necho out; echo err >&2; exit 5\n' > "$T/sandbox"
o=$(cd "$T/plain" && "$SHIM" verify . 2>/dev/null); rc=$?
(( rc == 5 )) && [ "$o" = out ] && ok "exit code and stdout pass through" || bad "pass-through" "rc=$rc out=$o"

# cache: built once, reused; parallel first builds of one pin all succeed (atomic rename)
m1=$(stat -c %Y "$T/cache/$A/crumb"); sleep 1; out "$T/locked" verify . >/dev/null; m2=$(stat -c %Y "$T/cache/$A/crumb")
[ "$m1" = "$m2" ] && ok "cached build is reused" || bad "cache reuse" "$m1 != $m2"
rm -rf "$T/cache"; for i in $(seq 8); do (out "$T/locked" verify . > "$T/par.$i") & done; wait
n=$(cat "$T"/par.* | grep -c '^PIN-A verify \.$'); (( n == 8 )) && ok "8 parallel first runs agree" || bad "parallel" "$n/8: $(cat "$T"/par.* | sort | uniq -c)"
ls "$T/cache/$A"/.crumb.* >/dev/null 2>&1 && bad "no temp files left" "$(ls -a "$T/cache/$A")" || ok "no temp files left"

# WHICH mode for the doctor
w=$(cd "$T/locked" && CRUMB_SHIM_WHICH=1 "$SHIM" verify .)
[[ $w == "$T/cache/$A/crumb crumb.lock:$A $T/locked" ]] && ok "CRUMB_SHIM_WHICH reports tool, rule, root" || bad "which" "$w"

echo "shim tests: $pass passed, $fail failed"; (( fail == 0 ))
