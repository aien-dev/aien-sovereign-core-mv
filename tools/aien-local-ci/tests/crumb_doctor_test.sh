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

# test-crumb-doctor.sh <doctor>: the doctor goes RED on a stale trusted clone (the 2026-10-11..18 synapse
# CRUMB-VERIFY "../aien-architecture is missing" cause), heals it, and raises/clears its aien-watch alert.
set -u
DOC=$(readlink -f "${1:?usage: test-crumb-doctor.sh <doctor>}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1: $2"; }
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
export AIEN_CLONES=$T/clones AIEN_WATCH_STATE=$T/watch AIEN_DOCTOR_NO_PROC=1 AIEN_DOCTOR_NO_READY=1 CRUMB_TOOL_CACHE=$T/cache
mkdir -p "$T/clones" "$T/watch"
echo "OTHER: unrelated alert (must survive)" > "$T/watch/ALERT"

# upstream sc: commit 1 without the synapse sibling line, commit 2 with it
mkdir -p "$T/up/sc/tools/aien-local-ci"; cd "$T/up/sc"; g init -q -b main
echo "omega aien-architecture aien-architecture main" > tools/aien-local-ci/siblings; g add -A; g commit -qm one
c1=$(git rev-parse HEAD)
echo "synapse aien-architecture aien-architecture main" >> tools/aien-local-ci/siblings; g commit -qam two
# upstream synapse: its gate asks for ../aien-architecture
mkdir -p "$T/up/synapse/tests/gates"; cd "$T/up/synapse"; g init -q -b main
printf 'case $1 in\n  crumb-verify)\n    need_sibling aien-architecture\n    ;;\nesac\n' > tests/gates/run.sh; g add -A; g commit -qm s
# trusted clones; sc work tree left on the OLD commit (what the Spark had)
git clone -q "$T/up/sc" "$T/clones/aien-sovereign-core"; git -C "$T/clones/aien-sovereign-core" -c advice.detachedHead=false checkout -q --detach "$c1"
git clone -q "$T/up/synapse" "$T/clones/synapse"

o=$("$DOC" --no-prs --repos synapse --alert 2>&1); rc=$?
(( rc == 1 )) && [[ $o == *"FAIL SIBMAP"*"missing synapse ../aien-architecture"* ]] && ok "stale trusted clone is RED (exit 1, names the missing sibling)" || bad "red" "rc=$rc $o"
grep -q '^CRUMB-DRIFT: ' "$T/watch/ALERT" && grep -q '^OTHER: ' "$T/watch/ALERT" && ok "alert raised, other alerts kept" || bad "alert" "$(cat "$T/watch/ALERT")"

o=$("$DOC" --heal-only 2>&1); rc=$?
h=$(git -C "$T/clones/aien-sovereign-core" rev-parse HEAD); m=$(git -C "$T/clones/aien-sovereign-core" rev-parse origin/main)
(( rc == 0 )) && [ "$h" = "$m" ] && ok "--heal-only moves the work tree to origin/main" || bad "heal" "rc=$rc head=$h main=$m $o"

o=$("$DOC" --no-prs --repos synapse --alert 2>&1); rc=$?
(( rc == 0 )) && [[ $o == *"PASS SIBMAP"* ]] && ok "after heal: GREEN" || bad "green" "rc=$rc $o"
! grep -q '^CRUMB-DRIFT: ' "$T/watch/ALERT" && grep -q '^OTHER: ' "$T/watch/ALERT" && ok "alert cleared when green, other alerts kept" || bad "clear" "$(cat "$T/watch/ALERT")"

# a dirty trusted clone is never moved (someone is using it): reported, not clobbered
git -C "$T/clones/aien-sovereign-core" -c advice.detachedHead=false checkout -q --detach "$c1"
echo "local edit" >> "$T/clones/aien-sovereign-core/tools/aien-local-ci/siblings"
o=$("$DOC" --heal-only 2>&1); rc=$?
(( rc == 1 )) && [[ $o == *"local changes"* ]] && grep -q "local edit" "$T/clones/aien-sovereign-core/tools/aien-local-ci/siblings" && ok "dirty work tree: refused, edit kept" || bad "dirty" "rc=$rc $o"

# map on main missing a sibling a gate needs: RED even when the work tree is current
git -C "$T/clones/aien-sovereign-core" checkout -q -- . ; git -C "$T/clones/aien-sovereign-core" -c advice.detachedHead=false checkout -q --detach origin/main
cd "$T/up/synapse"; sed -i 's/need_sibling aien-architecture/need_sibling aien-architecture\n    need_sibling omega/' tests/gates/run.sh; g commit -qam needs-omega; git -C "$T/clones/synapse" fetch -q
o=$("$DOC" --no-prs --repos synapse 2>&1); rc=$?
(( rc == 1 )) && [[ $o == *"synapse needs ../omega"* ]] && ok "gate needs a sibling the map lacks: RED" || bad "coverage" "rc=$rc $o"


# Main pin/tool is approved; a fork-like PR changes the lock, hosted workflow
# and Makefile to a key-reading recipe. The doctor may diff those bytes but
# must never build the PR's source or execute its selected pin.
mkdir -p "$T/up/arch/tools/crumb"
g -C "$T/up/arch" init -q -b main
printf 'crumb: crumb.c\n\t$(CC) -o $@ crumb.c\n' > "$T/up/arch/tools/crumb/Makefile"
printf '#include <stdio.h>\nint main(void){puts("crumb verify: OK");return 0;}\n' > "$T/up/arch/tools/crumb/crumb.c"
g -C "$T/up/arch" add -A; g -C "$T/up/arch" commit -qm approved
archpin=$(git -C "$T/up/arch" rev-parse HEAD)
g -C "$T/up/arch" checkout -qb unmerged
printf 'crumb:\n\t@cat "%s" > "%s"\n\t@false\n' "$T/trusted-key" "$T/exfiltrated" > "$T/up/arch/tools/crumb/Makefile"
g -C "$T/up/arch" commit -qam unapproved
badpin=$(git -C "$T/up/arch" rev-parse HEAD)
g -C "$T/up/arch" checkout -q main
git clone -q "$T/up/arch" "$T/clones/aien-architecture"
printf 'PRIVATE-TEST-KEY\n' > "$T/trusted-key"
mkdir -p "$T/up/gate/tests/gates" "$T/up/gate/.github/workflows"
g -C "$T/up/gate" init -q -b main
echo '{}' > "$T/up/gate/.crumb"
printf '%s\n' "$archpin" > "$T/up/gate/crumb.lock"
printf 'case $1 in\n  crumb-verify)\n    pin=$(cat crumb.lock)\n    ;;\nesac\n' > "$T/up/gate/tests/gates/run.sh"
printf '# pinned crumb tool source\n# ref: %s\n' "$archpin" > "$T/up/gate/.github/workflows/aien-verification.yml"
g -C "$T/up/gate" add -A; g -C "$T/up/gate" commit -qm approved
git clone -q "$T/up/gate" "$T/clones/gate"
mkdir -p "$T/bin"
printf '#!/bin/sh\nprintf "7 %%s\\n" "$CANDIDATE_SHA"\n' > "$T/bin/gh"; chmod +x "$T/bin/gh"
export PATH=$T/bin:$PATH CRUMB_SANDBOX_TOOL=$T/sandbox
printf '#!/bin/sh\necho "crumb verify: OK"\n' > "$T/sandbox"; chmod +x "$T/sandbox"
export CANDIDATE_SHA=$(git -C "$T/up/gate" rev-parse HEAD)
o=$("$DOC" --repos gate 2>&1); rc=$?
(( rc == 0 )) && [[ $o == *"PASS AGREE"*"trusted gate[crumb.lock"* ]] \
  && [[ $o == *"INFO PROMOTE"*"metadata unchanged"* ]] \
  && ok "approved main pin and unchanged candidate" || bad "approved main" "rc=$rc $o"

g -C "$T/up/gate" checkout -qb hostile
printf '%s\n' "$badpin" > "$T/up/gate/crumb.lock"
printf '# pinned crumb tool source\n# ref: %s\n' "$badpin" > "$T/up/gate/.github/workflows/aien-verification.yml"
mkdir -p "$T/up/gate/tools/crumb"
cp "$T/up/arch/tools/crumb/Makefile" "$T/up/gate/tools/crumb/Makefile"
cp "$T/up/arch/tools/crumb/crumb.c" "$T/up/gate/tools/crumb/crumb.c"
g -C "$T/up/gate" add -A; g -C "$T/up/gate" commit -qm hostile
export CANDIDATE_SHA=$(git -C "$T/up/gate" rev-parse HEAD)
git -C "$T/clones/gate" fetch -q origin hostile
o=$("$DOC" --repos gate --alert 2>&1); rc=$?
(( rc == 1 )) && [[ $o == *"PASS AGREE"*"trusted gate[crumb.lock"* ]] \
  && [[ $o == *"FAIL PROMOTE"*"operator promotion required"* ]] \
  && [[ $o == *"crumb.lock"* ]] && [ ! -e "$T/exfiltrated" ] \
  && grep -q '^CRUMB-DRIFT: ' "$T/watch/ALERT" \
  && ok "candidate pin/Makefile demand promotion, key not read" \
  || bad "hostile candidate" "rc=$rc sentinel=$(test -e "$T/exfiltrated" && echo PRESENT || echo absent) $o"

# Negative control: even trusted main is refused if its pin is not in
# architecture origin/main ancestry; the candidate source still never runs.
g -C "$T/up/gate" checkout -q main
printf '%s\n' "$badpin" > "$T/up/gate/crumb.lock"
g -C "$T/up/gate" commit -qam unapproved-main-pin
git -C "$T/clones/gate" fetch -q origin
o=$("$DOC" --repos gate --no-prs 2>&1); rc=$?
(( rc == 1 )) && [[ $o == *"trusted gate tool unavailable"* ]] && [ ! -e "$T/exfiltrated" ] \
  && ok "unapproved main pin cannot build" || bad "unapproved main pin" "rc=$rc $o"
echo "doctor tests: $pass passed, $fail failed"; (( fail == 0 ))
