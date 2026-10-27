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

# stall_check_test.sh: offline proofs for stall-check.sh and the desk-loop.sh board alert (no network, no gh).
# One desk pass runs against stub rebase-train.sh / pipeline.sh in a throwaway state directory.
#   bash tools/aien-local-ci/tests/stall_check_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SC=$HERE/../stall-check.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

NOW=2026-10-27T02:00:00Z
row() { printf '{"utc":"%s","repo":"%s","pr":"%s","head":"%s","action":"%s","reason":"%s"}\n' "$@"; }
H1=1111111111111111111111111111111111111111 H2=2222222222222222222222222222222222222222
{
  # sc#595: refused on one head for 3 h, still being refused: STALLED.
  row 2026-10-26T23:00:00Z sc 595 $H1 blocked "rebase onto origin/main conflicts: server.rs"
  row 2026-10-27T01:58:00Z sc 595 $H1 comment-skipped "already commented on this head (state)"
  row 2026-10-27T01:58:30Z sc 595 $H1 blocked "rebase onto origin/main conflicts: server.rs"
  # omega#1: refused for 1 h only: not yet.
  row 2026-10-27T01:00:00Z omega 1 $H1 blocked "conflicts with the base"
  row 2026-10-27T01:59:00Z omega 1 $H1 blocked "conflicts with the base"
  # omega#2: old head refused 5 h but refreshed an hour ago; the new head is 10 min old: not stalled.
  row 2026-10-26T20:00:00Z omega 2 $H1 blocked "conflicts with the base"
  row 2026-10-27T01:00:00Z omega 2 $H1 blocked "conflicts with the base"
  row 2026-10-27T01:50:00Z omega 2 $H2 blocked "failed hosted checks: CodeQL"
  # physics#3: refused 4 h, then merged (a pipeline.jsonl line): not stalled.
  row 2026-10-26T21:00:00Z physics 3 $H1 blocked "conflicts with the base"
  row 2026-10-27T01:30:00Z physics 3 $H1 blocked "conflicts with the base"
  echo '{"utc":"2026-10-27T01:40:00Z","action":"merge","repo":"physics","pr":"3","sha":"x","result":"merged","reason":"clean"}'
  # synapse#4: refused 6 h, label removed 3 h ago (no line since): not stalled any more.
  row 2026-10-26T17:00:00Z synapse 4 $H1 blocked "trust surface"
  row 2026-10-26T23:00:00Z synapse 4 $H1 blocked "trust surface"
  echo '{"utc":"2026-10-27T01:59:59Z","repo":"torn'
} > "$T/train.jsonl"

rc=0; out=$(bash "$SC" --now $NOW < "$T/train.jsonl") || rc=$?
check "a stall exits 1" test "$rc" -eq 1
check "exactly one stall reported" test "$(wc -l <<<"$out")" -eq 1
check "the stall is sc#595 on its head" test "$(jq -r '"\(.repo)#\(.pr)@\(.head)"' <<<"$out")" = "sc#595@$H1"
check "since, hours and newest reason" test "$(jq -r '"\(.since) \(.hours) \(.reason)"' <<<"$out")" = "2026-10-26T23:00:00Z 2.9 rebase onto origin/main conflicts: server.rs"

rc=0; bash "$SC" --now $NOW --hours 0.5 < "$T/train.jsonl" > "$T/half" || rc=$?
check "--hours 0.5 also reports omega#1 (rc 1)" test "$rc" -eq 1
check "--hours 0.5 reports sc#595 then omega#1, oldest first" test "$(jq -r '.repo + "#" + .pr' "$T/half" | paste -sd,)" = "sc#595,omega#1"

rc=0; bash "$SC" --now 2026-10-27T03:00:00Z < "$T/train.jsonl" > /dev/null || rc=$?
check "an hour of silence (label gone) clears the stall: exit 0" test "$rc" -eq 0

rc=0; bash "$SC" --now $NOW < /dev/null > /dev/null || rc=$?
check "empty log exits 0" test "$rc" -eq 0
rc=0; bash "$SC" --now yesterday < "$T/train.jsonl" 2> /dev/null || rc=$?
check "malformed --now exits 2" test "$rc" -eq 2
rc=0; bash "$SC" --bogus < /dev/null 2> /dev/null || rc=$?
check "unknown argument exits 2" test "$rc" -eq 2

# desk-loop.sh: one pass (stub pipeline removes KEEP-RUNNING) posts one board line per stalled head, once.
L=$T/lib S=$T/state
mkdir -p "$L" "$S"
cp "$SC" "$HERE/../desk-loop.sh" "$L/"
echo 'true' > "$L/rebase-train.sh"
printf 'rm -f "%s/KEEP-RUNNING"\n' "$S" > "$L/pipeline.sh"
recent() { date -u -d "-$1 min" +%FT%TZ; }
{
  row "$(recent 200)" sc 612 $H1 blocked "rebase onto origin/main conflicts: kv_persist.rs"
  row "$(recent 1)" sc 612 $H1 blocked "rebase onto origin/main conflicts: kv_persist.rs"
  row "$(recent 30)" omega 9 $H2 blocked "conflicts with the base"
  row "$(recent 1)" omega 9 $H2 blocked "conflicts with the base"
} > "$S/train.jsonl"
echo "2026-10-27T00:00:00Z someone: earlier line" > "$T/BOARD.md"
pass_once() { touch "$S/KEEP-RUNNING"; AIEN_STATE=$S AIEN_LIB=$L AIEN_BOARD=$T/BOARD.md AIEN_DESK_INTERVAL=0 bash "$L/desk-loop.sh" > "$T/desk.log" 2>&1; }
pass_once
check "one desk pass adds exactly one board line" test "$(wc -l < "$T/BOARD.md")" -eq 2
check "the board line names the PR, head and reason" grep -q "desk: STALLED sc#612 head 11111111 refused 3.3h since .*: rebase onto origin/main conflicts: kv_persist.rs" "$T/BOARD.md"
check "the board keeps its earlier line" grep -qx "2026-10-27T00:00:00Z someone: earlier line" "$T/BOARD.md"
pass_once
check "a second pass on the same head posts nothing" test "$(wc -l < "$T/BOARD.md")" -eq 2
row "$(recent 300)" sc 612 $H2 blocked "conflicts with the base" >> "$S/train.jsonl"
row "$(recent 1)" sc 612 $H2 blocked "conflicts with the base" >> "$S/train.jsonl"
pass_once
check "a new head that stalls again is alerted again" test "$(grep -c 'desk: STALLED sc#612' "$T/BOARD.md")" -eq 2
printf 'echo "{\\"repo\\":\\"x\\"" ; exit 2\n' > "$L/stall-check.sh"
pass_once
check "a failing stall-check never stops the desk or writes the board" test "$(wc -l < "$T/BOARD.md")" -eq 3 -a ! -e "$S/KEEP-RUNNING"

echo "stall_check_test: $pass passed, $fail failed"
((fail == 0))
