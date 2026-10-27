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

# test-crumb-shim.sh <shim>: no network, candidate Makefiles/locks never execute.
set -u
SHIM=$(readlink -f "${1:?usage: test-crumb-shim.sh <shim>}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "ok   $1"; }
bad()  { fail=$((fail+1)); echo "FAIL $1: $2"; }
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
printf '#!/bin/sh\necho SANDBOX "$@"\n' > "$T/sandbox"; chmod +x "$T/sandbox"

# Architecture main has two approved revisions. An unmerged commit carries a
# malicious Makefile: neither a candidate HEAD nor its lock can select it.
mkdir -p "$T/up-arch/tools/crumb"; g -C "$T/up-arch" init -q -b main
mk() {
  printf 'crumb: crumb.c\n\t$(CC) -o $@ crumb.c\n' > "$T/up-arch/tools/crumb/Makefile"
  printf '#include <stdio.h>\nint main(int c,char**v){printf("%s");for(int i=1;i<c;i++)printf(" %%s",v[i]);printf("\\n");return 0;}\n' "$1" > "$T/up-arch/tools/crumb/crumb.c"
}
mk PIN-A; g -C "$T/up-arch" add -A; g -C "$T/up-arch" commit -qm A; A=$(git -C "$T/up-arch" rev-parse HEAD)
mk PIN-B; g -C "$T/up-arch" add -A; g -C "$T/up-arch" commit -qm B; B=$(git -C "$T/up-arch" rev-parse HEAD)
g -C "$T/up-arch" checkout -qb hostile
printf 'crumb:\n\t@touch "%s"\n\t@false\n' "$T/hostile-make-ran" > "$T/up-arch/tools/crumb/Makefile"
g -C "$T/up-arch" commit -qam hostile; C=$(git -C "$T/up-arch" rev-parse HEAD)
g -C "$T/up-arch" checkout -q main
git clone -q "$T/up-arch" "$T/arch"
export CRUMB_SANDBOX_TOOL=$T/sandbox CRUMB_TOOL_CACHE=$T/cache CRUMB_ARCH_CLONE=$T/arch

repo() { # repo <name> [main lock]
  local name=$1
  mkdir -p "$T/up-$name"
  g -C "$T/up-$name" init -q -b main
  echo '{}' > "$T/up-$name/.crumb"
  [ $# = 1 ] || printf '%s\n' "$2" > "$T/up-$name/crumb.lock"
  g -C "$T/up-$name" add -A; g -C "$T/up-$name" commit -qm main
  git clone -q "$T/up-$name" "$T/$name"
}
repo locked "$A"
repo lockedB "  ${B^^}  "
repo plain
repo badlock invalid
repo nopin "$C"
mkdir -p "$T/locked/sub/dir" "$T/notgit"
# Candidate lock points to unmerged architecture code, and a generic repo
# with no lock pretends to own a tool. Neither source is executable authority.
printf '%s\n' "$C" > "$T/locked/crumb.lock"
mkdir -p "$T/plain/tools/crumb"
cp "$T/up-arch/tools/crumb/Makefile" "$T/plain/tools/crumb/Makefile"
printf 'crumb:\n\t@touch "%s"\n\t@false\n' "$T/hostile-plain-ran" > "$T/plain/tools/crumb/Makefile"
cp "$T/up-arch/tools/crumb/crumb.c" "$T/plain/tools/crumb/crumb.c"
printf 'crumb:\n\t@touch "%s"\n\t@false\n' "$T/hostile-arch-ran" > "$T/arch/tools/crumb/Makefile"
mkdir -p "$T/arch/sub"

out() { (cd "$1" && shift && "$SHIM" "$@" 2>&1); }
expect() { # expect <name> <prefix> <dir> <args...>
  local name=$1 want=$2 dir=$3 got; shift 3
  got=$(out "$dir" "$@")
  [[ $got == "$want"* ]] && ok "$name" || bad "$name" "want '$want...', got '$got'"
}
expect "candidate lock ignored, trusted main pin used" "PIN-A verify ." "$T/locked" verify .
expect "approved ancestor pin B" "PIN-B verify ." "$T/lockedB" verify .
expect "nested candidate uses trusted main" "PIN-A verify ." "$T/locked/sub/dir" verify .
expect "path argument chooses target main" "PIN-A verify $T/locked" "$T/plain" verify "$T/locked"
expect "file argument chooses target main" "PIN-A context" "$T/notgit" context "$T/locked/.crumb"
expect "candidate source without trusted main tool uses sandbox" "SANDBOX verify ." "$T/plain" verify .
expect "outside git uses sandbox" "SANDBOX verify ." "$T/notgit" verify .
expect "architecture uses archived main not dirty Makefile" "PIN-B verify ." "$T/arch" verify .
[ ! -e "$T/hostile-make-ran" ] && [ ! -e "$T/hostile-plain-ran" ] && [ ! -e "$T/hostile-arch-ran" ] \
  && ok "candidate Makefiles never ran" || bad "candidate Makefile" "sentinel created"

r=$(out "$T/badlock" verify .); rc=$?
(( rc == 97 )) && [[ $r == *"not one 40-hex"* ]] && ok "malformed main lock refuses" || bad "malformed main lock" "rc=$rc $r"
r=$(out "$T/nopin" verify .); rc=$?
(( rc == 97 )) && [[ $r == *"not in architecture origin/main ancestry"* ]] && [ ! -e "$T/hostile-make-ran" ] \
  && ok "unmerged main pin refuses before cache or build" || bad "unmerged main pin" "rc=$rc $r"

# Negative control: main pin absent and architecture main missing its own
# source both refuse, never silently accept a candidate source.
git -C "$T/locked" update-ref -d refs/remotes/origin/main
r=$(out "$T/locked" verify .); rc=$?
(( rc == 97 )) && [[ $r == *"no trusted origin/main"* ]] && ok "missing main refuses" || bad "missing main" "rc=$rc $r"

# Status pass-through applies only to a selected approved/sandbox tool.
printf '#!/bin/sh\necho out; echo err >&2; exit 5\n' > "$T/sandbox"
o=$(cd "$T/plain" && "$SHIM" verify . 2>/dev/null); rc=$?
(( rc == 5 )) && [ "$o" = out ] && ok "exit code and stdout pass through" || bad "pass-through" "rc=$rc out=$o"
w=$(cd "$T/lockedB" && CRUMB_SHIM_WHICH=1 "$SHIM" verify .)
[[ $w == "$T/cache/$B/crumb crumb.lock:$B $T/lockedB" ]] && ok "which reports main pin" || bad "which" "$w"

echo "shim tests: $pass passed, $fail failed"; (( fail == 0 ))
