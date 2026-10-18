#!/usr/bin/env bash
# Persistent KV two-process check on the chip (#514). CHIP-ONLY: never run by CI or cargo test.
# Same safety rules as own30b-pkv-chip-check.sh (see own30b-pkv-lib.sh): refuses while the live
# server on :18091 answers or a port is taken; stops only the PIDs it started.
#
#   scripts/own30b-pkv-two-process.sh [--dry-run] <evidence-dir> <aien-serve-http binary>
#
# Ports: PKV_CHECK_PORT (default here 18098) and PKV_CHECK_PORT2 (18099).
# Phases (the same greedy two-turn conversation as #507, temperature 0, seed 1):
#   off    persistence off, one server: turn 1 and turn 2 -> A_off, T_off (the baseline)
#   proc1  server 1 on store S: turn 1 (== A_off), wait for the parks; server 1 STAYS loaded
#   proc2  server 2 (a second process on the chip at the same time) on a copy of S, same key:
#          turn 2 must restore (restores == 1) and equal T_off
#   ram    server 1 answers turn 2 from its RAM prefix: equals T_off, restores == 0
# Two 30B processes are resident at once (~31 GB each, unified memory): each load refuses below
# 45 GB MemAvailable. One store directory is never opened by two processes at once (ADR 0003:
# one runtime per store directory; the store takes no cross-process lock), hence the copy.
set -euo pipefail
export PKV_CHECK_PORT=${PKV_CHECK_PORT:-18098}
. "$(dirname "$0")/own30b-pkv-lib.sh"
pkv_usage_args "$(basename "$0")" "$@"
PORT1=$PORT
PORT2=${PKV_CHECK_PORT2:-18099}

if [ $DRY = 1 ]; then
	echo "PLAN own30b-pkv-two-process (#514): off, proc1 (park, stays up), proc2 (restore beside it), ram"
	echo "     3 server starts on :$PORT1 and :$PORT2; evidence -> $dir"
	echo "PREREQUISITES"
	bad=0
	pkv_prereqs 1 || bad=1
	PORT=$PORT2 base="http://127.0.0.1:$PORT2"
	if pkv_port_reserved "$PORT2"; then echo "  STOP :$PORT2 belongs to another service (reserved: $PKV_RESERVED_PORTS)"; bad=1; elif pkv_port_busy; then echo "  STOP something listens on :$PORT2"; bad=1; else echo "  ok   :$PORT2 is free"; fi
	[ $bad = 0 ] && { echo "DRY RUN: ready"; exit 0; }
	echo "DRY RUN: NOT ready"
	exit 1
fi

PORT=$PORT2 base="http://127.0.0.1:$PORT2"
if pkv_port_reserved "$PORT2"; then
	echo "refusing: :$PORT2 belongs to another service (reserved: $PKV_RESERVED_PORTS)" >&2
	exit 2
fi
if pkv_port_busy; then
	echo "refusing: something already listens on :$PORT2 (pick a free PKV_CHECK_PORT2)" >&2
	exit 2
fi
PORT=$PORT1 base="http://127.0.0.1:$PORT1"
pkv_begin

use() { PORT=$1 base="http://127.0.0.1:$1" PIDF="$work/server-$1.pid"; }
trap 'use $PORT2; stop_server || PKV_LEAK=1; use $PORT1; stop_server || PKV_LEAK=1; rm -rf "$work"; pkv_trap_status' EXIT
memavail() { awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo; }

use "$PORT1"
echo "== off: persistence off, uninterrupted (baseline)"
start_server off -
a1_off=$(chat off-turn1 "$m1")
t_off=$(chat off-turn2 "$(turn2_msgs "$a1_off")")
stop_server

echo "== proc1: turn 1 on store S, parks, stays loaded"
S="$work/store"
start_server proc1 "$S"
a1=$(chat proc1-turn1 "$m1")
wait_parks
health proc1-after-turn1 >/dev/null
P1=$(server_pid)
cp -a "$S" "$work/store2"
echo "MemAvailable before proc2 load: $(memavail) GB" | tee -a "$dir/memory.txt"

echo "== proc2: a second process on the chip, restores turn 2 from a copy of S"
use "$PORT2"
start_server proc2 "$work/store2"
P2=$(server_pid)
{
	echo "both resident: proc1 pid $P1 alive=$(kill -0 "$P1" 2>/dev/null && echo yes || echo no), proc2 pid $P2"
	echo "MemAvailable with both loaded: $(memavail) GB"
	ps -o pid,rss,comm -p "$P1","$P2"
} | tee -a "$dir/memory.txt"
t2_proc2=$(chat proc2-turn2 "$(turn2_msgs "$a1")")
health proc2 >/dev/null
r2=$(pk proc2 restores)

echo "== ram: process 1 answers turn 2 from its RAM prefix while process 2 is still loaded"
use "$PORT1"
t2_proc1=$(chat proc1-turn2 "$(turn2_msgs "$a1")")
health proc1 >/dev/null
r1=$(pk proc1 restores)
kill -0 "$P2" 2>/dev/null && echo "proc2 still alive during the ram phase" | tee -a "$dir/memory.txt"

{
	check "proc1 turn 1 equals the uninterrupted turn 1" "$a1" "$a1_off"
	check "proc2 restored turn 2 equals the uninterrupted turn 2" "$t2_proc2" "$t_off"
	check "proc2 restored (restores == 1)" "$r2" 1
	check "proc1 park refused no chunk" "$(pk proc1-after-turn1 park_refused)" 0
	p1=$(jq -r .usage.prompt_tokens "$dir/proc1-turn1.json")
	check "proc2 restore covered at least the turn-1 prompt ($p1 tokens)" "$([ "$(pk proc2 restored_tokens)" -ge "$p1" ] && echo yes)" yes
	check "proc1 RAM turn 2 equals the uninterrupted turn 2" "$t2_proc1" "$t_off"
	check "proc1 did not restore (restores == 0)" "$r1" 0
	[ $fail = 0 ] && echo "VERDICT PASS" || echo "VERDICT FAIL"
} | tee "$dir/summary.txt"
use "$PORT2"
stop_server
use "$PORT1"
stop_server
pkv_seal
grep -q "VERDICT PASS" "$dir/summary.txt"
