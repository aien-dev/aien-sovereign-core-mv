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

# Persistent KV acceptance check on the chip (sc#505/#507, locked by #499). CHIP-ONLY: never run
# by CI or cargo test. It starts its OWN servers on 127.0.0.1:18098 (PKV_CHECK_PORT; never
# touches :18091, the live server, or its PID/log files) and loads the 30B checkpoint each time
# (~30.5 GB resident, about 18 s load plus the checkpoint digest when persistence is on).
#
#   scripts/own30b-pkv-chip-check.sh [--dry-run] <evidence-dir> <aien-serve-http binary>
#
# --dry-run prints the plan and checks the prerequisites; it starts no server.
# The binary must be a native release build of this branch from a clean checkout (dirty or
# unknown builds refuse persistence by design), e.g.
#   AIEN_OMEGA_DIR=<omega checkout at the omega.lock pin> cargo build --release -p aien-serve-http
#
# Key: atlas-vault entry AIEN_KV_STORE_KEY by default; PKV_CHECK_EPHEMERAL_KEY=1 makes a
# throwaway key (never printed or saved). Stores are private temp dirs. Refuses while :18091
# answers. Helpers and rules: scripts/own30b-pkv-lib.sh.
#
# Phases (same greedy two-turn conversation, temperature 0, seed 1):
#   off        persistence off, one server kept running -> T_off, T2b_off (other turn 2), A_off (other chat)
#   on         persistence on (fresh store), one server running   -> turn2 = T_on
#   park       persistence on (fresh store S): turn 1, wait for the parks, stop
#   restore    restart on S: turn 2 must restore (restores == 1)  -> T_restored
#   neg-switch restart on S with --prefill-device-host off: restores == 0 (another identity)
#   torn       fresh store W, AIEN_KV_TEST_FAULT=p3:600000 (every park stalls before its publish
#              rename): turn 1, wait for a temp file, kill -9, cut every temp file in half, restart
#              plain on W: nothing published, recovery removes the temps, turn 2 == T_off, restores 0
#   late-load  fresh store L parked normally; restart with AIEN_KV_TEST_FAULT=load:15000:1 (the
#              first restore read stalls past the 10 s load timeout): turn 2 == T_off with
#              restores 0 and one timeout miss; after the late completion lands restores is still
#              0 and another chat == A_off; restart plain on L: turn 2 restores 1 and == T_off
#   late-park  fresh store P, AIEN_KV_TEST_FAULT=p3:20000:1 (the first park stalls 20 s): turn 1,
#              turn 2 and another turn 2 run while it is in flight (== T_off, == T2b_off); parks
#              settle with none failed; restart plain on P: turn 2 restores 1 and == T_off
#   neg-corrupt flip one byte in every stored object of S, restart: restores == 0, still answers
# PASS when every answer equals its uninterrupted twin and every negative control restores nothing.
# Not covered: a conversation "deleted" before its park lands; the server has no delete API, and
# chunks are content addressed, so a late park can only serve the exact same token prefix.
# Rough duration: 13 server starts, 14 to 20 minutes.
set -euo pipefail
. "$(dirname "$0")/own30b-pkv-lib.sh"
pkv_usage_args "$(basename "$0")" "$@"

if [ $DRY = 1 ]; then
	echo "PLAN own30b-pkv-chip-check (#507): off, on, park, restore, neg-switch, torn (kill -9 mid-write),"
	echo "     late-load (restore past its timeout), late-park (park lands after the chat moved on), neg-corrupt"
	echo "     13 server starts on :$PORT, estimate 14 to 20 minutes; evidence -> $dir"
	echo "PREREQUISITES"
	if pkv_prereqs 1; then echo "DRY RUN: ready"; exit 0; fi
	echo "DRY RUN: NOT ready"
	exit 1
fi
pkv_begin

Q2B="Which of those is least useful at home?"
QA="Explain in two sentences what a bearing preload is."
mA=$(jq -n --arg s "$SYS" --arg q "$QA" '[{role:"system",content:$s},{role:"user",content:$q}]')

echo "== off: persistence off, uninterrupted"
start_server off -
a1_off=$(chat off-turn1 "$m1")
t_off=$(chat off-turn2 "$(turn2_msgs "$a1_off")")
t2b_off=$(chat off-turn2b "$(turn2_msgs "$a1_off" "$Q2B")")
aA_off=$(chat off-other "$mA")
health off >/dev/null
stop_server

echo "== on: persistence on, uninterrupted"
start_server on "$work/store-on"
a1_on=$(chat on-turn1 "$m1")
t_on=$(chat on-turn2 "$(turn2_msgs "$a1_on")")
health on >/dev/null
stop_server

echo "== park: turn 1, wait for the parks, stop"
S="$work/store"
start_server park "$S"
a1=$(chat park-turn1 "$m1")
wait_parks
health park >/dev/null
stop_server

echo "== restore: restart on the same store, turn 2"
start_server restore "$S"
t_restored=$(chat restore-turn2 "$(turn2_msgs "$a1")")
health restore >/dev/null
stop_server

echo "== neg-switch: restart with another prefill switch (another identity)"
start_server neg-switch "$S" --prefill-device-host off
chat neg-switch-turn2 "$(turn2_msgs "$a1")" >/dev/null
health neg-switch >/dev/null
stop_server

wait_store_open() { # <name>: until store_state is open (recovery done)
	for _ in $(seq 1 120); do
		[ "$(curl -s -m 5 "$base/health" | jq -r '.persistent_kv.store_state')" = open ] && return 0
		sleep 1
	done
	echo "store did not open after recovery ($1)" >&2
	exit 1
}

echo "== torn: kill -9 while every park is stalled before its publish rename"
W="$work/store-torn"
PKV_SERVER_ENV="AIEN_KV_TEST_FAULT=p3:600000" start_server torn-a "$W"
a1_torn=$(chat torn-turn1 "$m1")
for _ in $(seq 1 120); do
	[ -n "$(find "$W/tmp" -type f 2>/dev/null | head -1)" ] && break
	sleep 0.5
done
health torn-a >/dev/null
kill9_server
find "$W" -type f -printf '%P %s\n' | sort >"$dir/torn-files-at-kill.txt"
torn_tmp=$(find "$W/tmp" -type f | wc -l)
torn_obj=$(find "$W/objects" -type f | wc -l)
find "$W/tmp" -type f | while read -r f; do truncate -s $(($(stat -c %s "$f") / 2)) "$f"; done
start_server torn-b "$W"
wait_store_open torn-b
torn_tmp_after=$(find "$W/tmp" -type f | wc -l)
t_torn=$(chat torn-turn2 "$(turn2_msgs "$a1_torn")")
health torn-b >/dev/null
stop_server

echo "== late-load: the first restore read stalls past the load timeout"
L="$work/store-late-load"
start_server late-load-park "$L"
a1_ll=$(chat late-load-turn1 "$m1")
wait_parks
stop_server
PKV_SERVER_ENV="AIEN_KV_TEST_FAULT=load:15000:1" start_server late-load-a "$L"
wait_store_open late-load-a
t_ll=$(chat late-load-turn2 "$(turn2_msgs "$a1_ll")")
health late-load-a1 >/dev/null
sleep 8 # the stalled read completes about 5 s after the 10 s timeout
aA_ll=$(chat late-load-other "$mA")
health late-load-a2 >/dev/null
stop_server
start_server late-load-b "$L"
t_ll_b=$(chat late-load-b-turn2 "$(turn2_msgs "$a1_ll")")
health late-load-b >/dev/null
stop_server

echo "== late-park: the first park stalls 20 s while the chat moves on"
P="$work/store-late-park"
PKV_SERVER_ENV="AIEN_KV_TEST_FAULT=p3:20000:1" start_server late-park-a "$P"
a1_lp=$(chat late-park-turn1 "$m1")
health late-park-inflight >/dev/null
t_lp=$(chat late-park-turn2 "$(turn2_msgs "$a1_lp")")
t2b_lp=$(chat late-park-turn2b "$(turn2_msgs "$a1_lp" "$Q2B")")
wait_parks
health late-park-a >/dev/null
stop_server
start_server late-park-b "$P"
t_lp_b=$(chat late-park-b-turn2 "$(turn2_msgs "$a1_lp")")
health late-park-b >/dev/null
stop_server

echo "== neg-corrupt: flip one byte in every stored object, restart"
find "$S/objects" -type f | while read -r f; do flip_byte "$f"; done
start_server neg-corrupt "$S"
chat neg-corrupt-turn2 "$(turn2_msgs "$a1")" >/dev/null
health neg-corrupt >/dev/null
stop_server

{
	check "turn 1 identical off/on/park/torn/late" "$a1_off|$a1_on|$a1_torn|$a1_ll|$a1_lp" "$a1|$a1|$a1|$a1|$a1"
	check "T_on == T_off (persistence changes no token)" "$t_on" "$t_off"
	check "T_restored == T_off (restored = uninterrupted)" "$t_restored" "$t_off"
	check "restore served once" "$(pk restore restores)" 1
	check "park refused no chunk (whole turn 1 parked)" "$(pk park park_refused)" 0
	p1=$(jq -r .usage.prompt_tokens "$dir/park-turn1.json")
	check "restore covered at least the turn-1 prompt ($p1 tokens)" "$([ "$(pk restore restored_tokens)" -ge "$p1" ] && echo yes)" yes
	check "neg-switch restores nothing" "$(pk neg-switch restores)" 0
	check "torn: a temp file was in flight at the kill" "$([ "$torn_tmp" -gt 0 ] && echo yes || echo "no ($torn_tmp)")" yes
	check "torn: nothing was published before the kill" "$torn_obj" 0
	check "torn: recovery removed every temp file" "$torn_tmp_after" 0
	check "torn: health counts the removed temps" "$(jq -r '.persistent_kv.recovery.tmp_deleted >= 1' "$dir/torn-b-health.json")" true
	check "torn: restores nothing" "$(pk torn-b restores)" 0
	check "torn: T == T_off" "$t_torn" "$t_off"
	check "late-load: timed out once" "$(jq -r '.persistent_kv.restore_misses.timeout // 0' "$dir/late-load-a1-health.json")" 1
	check "late-load: restores nothing" "$(pk late-load-a1 restores)" 0
	check "late-load: T == T_off" "$t_ll" "$t_off"
	check "late-load: the late completion is never served" "$(pk late-load-a2 restores)" 0
	check "late-load: another chat == A_off" "$aA_ll" "$aA_off"
	check "late-load: the store still restores after" "$(pk late-load-b restores)" 1
	check "late-load: restored T == T_off" "$t_ll_b" "$t_off"
	check "late-park: the park was in flight during turn 2" "$([ "$(pk late-park-inflight parked_chunks)" -eq 0 ] && echo yes || echo no)" yes
	check "late-park: T == T_off" "$t_lp" "$t_off"
	check "late-park: other turn 2 == T2b_off" "$t2b_lp" "$t2b_off"
	check "late-park: no park failed" "$(pk late-park-a park_failed)" 0
	check "late-park: restores once after restart" "$(pk late-park-b restores)" 1
	check "late-park: restored T == T_off" "$t_lp_b" "$t_off"
	check "neg-corrupt restores nothing" "$(pk neg-corrupt restores)" 0
	check "off health has no persistent_kv" "$(jq -r 'has("persistent_kv")' "$dir/off-health.json")" false
	[ $fail = 0 ] && echo "PKV CHIP CHECK: PASS" || echo "PKV CHIP CHECK: FAIL"
} | tee "$dir/summary.txt"
grep -q "PKV CHIP CHECK: PASS" "$dir/summary.txt" || fail=1
pkv_seal
exit $fail
