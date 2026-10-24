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

# Persistent KV operations check on the chip (sc#509, contract locked in #500). CHIP-ONLY: never
# run by CI or cargo test. Same safety rules as own30b-pkv-chip-check.sh (see own30b-pkv-lib.sh):
# own server on 127.0.0.1:18098 (PKV_CHECK_PORT), refuses while :18091 answers or the port is
# taken, stops only the PID it started, key via `atlas-vault run AIEN_KV_STORE_KEY --` (never
# printed), evidence sealed with SHA256SUMS.
#
#   scripts/own30b-pkv-ops-check.sh [--dry-run] <evidence-dir> <aien-serve-http binary>
#
# Phases (the #507 greedy two-turn conversation, temperature 0, seed 1):
#   off        AIEN_KV_STORE_DIR unset: no persistent_kv in health, no "persistent KV" log line
#              -> T_off (turn 2 uninterrupted)
#   on         opt in with a store dir D that does not exist yet: health on, store open,
#              recovery done, every counter present; turn 2 == T_off; after the parks every dir
#              under D is 0700 and every file 0600
#   sigterm    on the same server: a 256-token streamed chat, SIGTERM once its first token is
#              out: the chat still ends with [DONE], the server exits 0 within PKV_STOP_WAIT_S
#              (120), logs "shutdown complete", and leaves no temp file in D
#   restart    restart on D: turn 2 restores once and == T_off, recovery indexed the objects
#   corrupt    flip a byte in every object of D, add a foreign file and a stray temp file:
#              starts and serves, restores 0, turn 2 == T_off, recovery removed the temp
#   foreign    restart on D with another valid key (a store written by someone else): restores 0,
#              turn 2 == T_off
#   unsafe     a store dir with mode 0755: persistence refused at startup (refusals 1), the
#              server still serves, turn 2 == T_off, the dir stays empty
#   nokey      AIEN_KV_STORE_DIR set without AIEN_KV_STORE_KEY: refused, still serves == T_off
#   rollback   unset AIEN_KV_STORE_DIR and restart: health has no persistent_kv, turn 2 == T_off,
#              and D is byte-for-byte untouched (SHA-256 of every file before and after)
# Rough duration: 8 server starts, 9 to 15 minutes.
set -euo pipefail
. "$(dirname "$0")/own30b-pkv-lib.sh"
pkv_usage_args "$(basename "$0")" "$@"

if [ "$DRY" = 1 ]; then
	echo "PLAN own30b-pkv-ops-check (#509): off, on (fresh dir, 0700/0600), sigterm (graceful drain),"
	echo "     restart (restores), corrupt + foreign/stray files, foreign key, unsafe dir (0755), no key, rollback"
	echo "     8 server starts on :$PORT, estimate 9 to 15 minutes; evidence -> $dir"
	echo "PREREQUISITES"
	if pkv_prereqs 1; then echo "DRY RUN: ready"; exit 0; fi
	echo "DRY RUN: NOT ready"
	exit 1
fi
pkv_begin

perm_bad() { # <dir>: entries that are not 0700 dirs / 0600 files
	find "$1" \( -type d ! -perm 700 \) -o \( -type f ! -perm 600 \) | wc -l
}
tree_sums() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum); }

echo "== off: AIEN_KV_STORE_DIR unset"
start_server off -
a1_off=$(chat off-turn1 "$m1")
t_off=$(chat off-turn2 "$(turn2_msgs "$a1_off")")
health off >/dev/null
stop_server

echo "== on: opt in with a store dir that does not exist yet"
D="$work/store"
start_server on "$D"
a1=$(chat on-turn1 "$m1")
wait_parks
t_on=$(chat on-turn2 "$(turn2_msgs "$a1")")
wait_parks
health on >/dev/null
find "$D" -printf '%M %u %s %P\n' | sort -k4 >"$dir/on-perms.txt"
on_files=$(find "$D" -type f | wc -l)
on_perm_bad=$(perm_bad "$D")

echo "== sigterm: SIGTERM while a streamed chat is generating"
jq -n --arg model "$SERVED_MODEL" --argjson m "$(turn2_msgs "$a1" "List ten uses of a torque wrench, one line each, with a reason for each.")" \
	'{model:$model, messages:$m, max_tokens:256, temperature:0, seed:1, stream:true}' >"$work/long.body"
curl -sN -m 600 "$base/v1/chat/completions" -H 'content-type: application/json' -d @"$work/long.body" >"$dir/sigterm-stream.sse" 2>&1 &
cpid=$!
for _ in $(seq 1 600); do
	grep -q '"content":"[^"]' "$dir/sigterm-stream.sse" 2>/dev/null && break
	sleep 0.2
done
spid=$(server_pid)
t0=$SECONDS
kill -TERM "$spid"
wait "$cpid" || true
rc=0
wait "$SERVER_JOB" || rc=$?
term_s=$((SECONDS - t0))
rm -f "$PIDF"
sig_done=$(grep -c 'data: \[DONE\]' "$dir/sigterm-stream.sse" || true)
sig_tokens=$(grep -o '"completion_tokens":[0-9]*' "$dir/sigterm-stream.sse" | tail -1 | cut -d: -f2)
sig_tmp=$(find "$D/tmp" -type f 2>/dev/null | wc -l)

echo "== restart: same store"
start_server restart "$D"
t_restart=$(chat restart-turn2 "$(turn2_msgs "$a1")")
health restart >/dev/null
stop_server

echo "== corrupt: flip a byte in every object, add a foreign file and a stray temp"
find "$D/objects" -type f | while read -r f; do flip_byte "$f"; done
head -c 4096 /dev/urandom >"$D/objects/zz-foreign.bin"
head -c 4096 /dev/urandom >"$D/tmp/zz-stray.tmp"
chmod 600 "$D/objects/zz-foreign.bin" "$D/tmp/zz-stray.tmp"
start_server corrupt "$D"
t_corrupt=$(chat corrupt-turn2 "$(turn2_msgs "$a1")")
health corrupt >/dev/null
stop_server
corrupt_stray=$([ -e "$D/tmp/zz-stray.tmp" ] && echo present || echo removed)

echo "== foreign: another valid key on the same store"
start_server foreign "foreignkey:$D"
t_foreign=$(chat foreign-turn2 "$(turn2_msgs "$a1")")
health foreign >/dev/null
stop_server

echo "== unsafe: store dir with mode 0755"
U="$work/unsafe"
mkdir -p "$U"
chmod 755 "$U"
start_server unsafe "$U"
t_unsafe=$(chat unsafe-turn2 "$(turn2_msgs "$a1")")
health unsafe >/dev/null
stop_server
unsafe_entries=$(find "$U" -mindepth 1 | wc -l)

echo "== nokey: AIEN_KV_STORE_DIR set, no AIEN_KV_STORE_KEY"
start_server nokey "nokey:$work/store-nokey"
t_nokey=$(chat nokey-turn2 "$(turn2_msgs "$a1")")
health nokey >/dev/null
stop_server

echo "== rollback: unset AIEN_KV_STORE_DIR and restart"
tree_sums "$D" >"$dir/rollback-store-before.sha256"
start_server rollback -
t_rollback=$(chat rollback-turn2 "$(turn2_msgs "$a1")")
health rollback >/dev/null
stop_server
tree_sums "$D" >"$dir/rollback-store-after.sha256"

{
	check "off: no persistent_kv in health" "$(jq -r 'has("persistent_kv")' "$dir/off-health.json")" false
	check "off: no persistent KV log line" "$(grep -c 'persistent KV' "$dir/off.log" || true)" 0
	check "on: turn 1 == off" "$a1" "$a1_off"
	check "on: T == T_off" "$t_on" "$t_off"
	check "on: state on, store open" "$(pk on state)/$(pk on store_state)" on/open
	check "on: recovery done" "$(pk on recovery.done)" true
	for k in parked_chunks park_failed restores restore_misses refusals seal recovery io; do
		check "on: health has $k" "$(jq -r ".persistent_kv | has(\"$k\")" "$dir/on-health.json")" true
	done
	check "on: no park failed" "$(pk on park_failed)" 0
	check "on: the store holds files" "$([ "$on_files" -gt 0 ] && echo yes || echo "no ($on_files)")" yes
	check "on: every dir 0700, every file 0600" "$on_perm_bad" 0
	check "sigterm: the in-flight chat finished ([DONE])" "$sig_done" 1
	check "sigterm: the in-flight chat got tokens" "$([ "${sig_tokens:-0}" -gt 0 ] && echo yes || echo no)" yes
	check "sigterm: exit status 0" "$rc" 0
	check "sigterm: exited within ${PKV_STOP_WAIT_S:-120} s" "$([ "$term_s" -le "${PKV_STOP_WAIT_S:-120}" ] && echo yes || echo "no (${term_s} s)")" yes
	check "sigterm: log says shutdown complete" "$(grep -c 'shutdown complete' "$dir/on.log" || true)" 1
	check "sigterm: no temp file left" "$sig_tmp" 0
	check "restart: restores once" "$(pk restart restores)" 1
	check "restart: T == T_off" "$t_restart" "$t_off"
	check "restart: recovery indexed objects" "$(jq -r '.persistent_kv.recovery.indexed > 0' "$dir/restart-health.json")" true
	check "corrupt: serves, store not failed" "$(jq -r '.persistent_kv.store_state != "failed"' "$dir/corrupt-health.json")" true
	check "corrupt: restores nothing" "$(pk corrupt restores)" 0
	check "corrupt: T == T_off" "$t_corrupt" "$t_off"
	check "corrupt: recovery removed the stray temp" "$corrupt_stray" removed
	check "corrupt: health counts the removed temp" "$(jq -r '.persistent_kv.recovery.tmp_deleted >= 1' "$dir/corrupt-health.json")" true
	check "foreign: restores nothing" "$(pk foreign restores)" 0
	check "foreign: T == T_off" "$t_foreign" "$t_off"
	check "unsafe: refused at startup" "$(pk unsafe state)/$(pk unsafe refusals)" refused/1
	check "unsafe: T == T_off" "$t_unsafe" "$t_off"
	check "unsafe: the dir stays empty" "$unsafe_entries" 0
	check "nokey: refused at startup" "$(pk nokey state)/$(pk nokey refusals)" refused/1
	check "nokey: T == T_off" "$t_nokey" "$t_off"
	check "rollback: no persistent_kv in health" "$(jq -r 'has("persistent_kv")' "$dir/rollback-health.json")" false
	check "rollback: no persistent KV log line" "$(grep -c 'persistent KV' "$dir/rollback.log" || true)" 0
	check "rollback: T == T_off" "$t_rollback" "$t_off"
	check "rollback: the store is untouched" "$(cmp -s "$dir/rollback-store-before.sha256" "$dir/rollback-store-after.sha256" && echo same || echo changed)" same
	echo "recovery after corrupt: $(jq -c '.persistent_kv.recovery' "$dir/corrupt-health.json")"
	echo "misses after corrupt: $(jq -c '.persistent_kv.restore_misses' "$dir/corrupt-health.json"); after foreign: $(jq -c '.persistent_kv.restore_misses' "$dir/foreign-health.json")"
	echo "unsafe reason: $(pk unsafe reason); nokey reason: $(pk nokey reason)"
	[ $fail = 0 ] && echo "PKV OPS CHECK: PASS" || echo "PKV OPS CHECK: FAIL"
} | tee "$dir/summary.txt"
grep -q "PKV OPS CHECK: PASS" "$dir/summary.txt" || fail=1
pkv_seal
exit $fail
