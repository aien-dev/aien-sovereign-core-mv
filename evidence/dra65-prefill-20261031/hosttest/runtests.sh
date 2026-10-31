#!/usr/bin/env bash
# Host check of window-dra65.sh run against the mock door (fake-serve.sh); scratch lock, never the chip.
# Run as: nice -n 19 taskset -c 0-9 bash evidence/dra65-prefill-20261031/hosttest/runtests.sh  (ports 18978, 18979)
set -u
H=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$H/.." && pwd)/window-dra65.sh  # the canonical path the script re-execs itself under
export REPO; REPO=$(git -C "$H" rev-parse --show-toplevel)
T=${DRA65_HOSTTEST_DIR:-/tmp/dra65-hosttest}; rm -rf "$T"; mkdir -p "$T/win"
export SPARK_WINDOW=w-test-3 SPARK_WINDOW_DIR=$T/win DRA65_TEST_LOCK=$T/t3.lock DRA65_TEST_PORT=18978
: > "$DRA65_TEST_LOCK"
B=$H/fake-serve.sh
win() { rm -f "$T/win/expired"; date -u +%FT%TZ > "$T/win/held"; echo "$1" > "$T/win/lease"; export SPARK_LEASE_UNIT_S=$2; }
lockfree() { flock -n "$DRA65_TEST_LOCK" true && echo lock=free || echo lock=HELD; }
left() { pgrep -f 'mock2.py|_watch' | grep -v $$ | wc -l; }
sr() { jq -c '{step, rc, verdict, why, lease_expired, v: .result.verdict, same: .result.same, no_snap: .result.no_snap}' "$1/step-receipt.json" 2>/dev/null || echo "no receipt $1"; }
t() { echo; echo "== $*"; }

t "T1 L1 steps 1 2 3 4 5 (lease 20 min), under taskset -c 5"
win 20 60; taskset -c 5 bash "$S" run "$B" "$T/o1" 1 2 3 4 5 > "$T/t1.out" 2> "$T/t1.err"; echo "rc=$?"
for s in 1 2 3 4 5; do sr "$T/o1/$s"; done
grep -h fake-serve "$T"/o1/*/server.log | sed 's/pid [0-9]* sid [0-9]*/pid P sid P/' | sort | uniq -c
for s in 1 4; do echo "server pid==sid: $(grep -ho 'pid [0-9]* sid [0-9]*' "$T/o1/$s/server.log" | awk '{print ($2==$4)}')"; done
jq -c 'select(.id=="turn2") | {id, verdict, snapshot_log}' "$T/o1/4/golden.jsonl"
jq -c '{window, bin_sha256: .bin_sha256[0:12]}' "$T/o1/5/golden-receipt.json"; sleep 2; lockfree; echo "leftover=$(left)"; cat "$T/o1/board.log"

t "T2 green 4 alone on o1 (reds from step receipts)"; win 20 60
bash "$S" run "$B" "$T/o1" 4 > /dev/null 2> "$T/t2.err"; echo "rc=$?"; sr "$T/o1/4"
t "T3 order: 4 1 2 on empty o3"; bash "$S" run "$B" "$T/o3" 4 1 2 2>&1 >/dev/null | tail -n 1; echo "rc=${PIPESTATUS[0]}"
t "T4 golden receipts without step receipts (o4): run 4"
mkdir -p "$T/o4/1" "$T/o4/2"; cp "$T/o1/1/golden-receipt.json" "$T/o4/1/"; cp "$T/o1/2/golden-receipt.json" "$T/o4/2/"
bash "$S" run "$B" "$T/o4" 4 2>&1 >/dev/null | tail -n 1
t "T4b probe green directly with those receipts (mock up on 18979)"
M=exact python3 "$H/mock2.py" "$REPO" 18979 2>/dev/null & mp=$!; sleep 1
SHA=$(sha256sum < "$B" | cut -c1-64)
bash "$S" golden 18979 "$T/o4/g" green "$SHA" x "$T/o4/1/golden-receipt.json" "$T/o4/2/golden-receipt.json" > /dev/null 2> "$T/t4b.err"; echo "rc=$?"; cat "$T/t4b.err"
t "T4c same probe with o1's receipts (step receipts present) but no DRA65_SERVER_LOG -> turn2 NO_SNAP -> FAIL"
bash "$S" golden 18979 "$T/o4/g2" green "$SHA" x "$T/o1/1/golden-receipt.json" "$T/o1/2/golden-receipt.json" | jq -c '{verdict, same, no_snap}'; echo "rc=${PIPESTATUS[0]}"
kill $mp

t "T5 nosnap door, step 4 on o1"; win 20 60
MOCK_M=nosnap bash "$S" run "$B" "$T/o1" 4 > /dev/null 2>&1; echo "rc=$?"; sr "$T/o1/4"
jq -c 'select(.id=="turn2") | {id, verdict, snapshot_log}' "$T/o1/4/golden.jsonl"

t "T6 deadline: slow door (30 s/request), lease 1 x 45 s, step 4"; win 1 45
s0=$(date +%s); MOCK_M=slow bash "$S" run "$B" "$T/o1" 4 > /dev/null 2> "$T/t6.err"; echo "rc=$? after $(( $(date +%s) - s0 )) s"
sr "$T/o1/4"; sleep 2; lockfree; echo "leftover=$(left)"; tail -n 2 "$T/t6.err"

t "T7 expired file appears mid-step (slow1 door), step 4"; win 20 60
( sleep 4; touch "$T/win/expired" ) & MOCK_M=slow1 bash "$S" run "$B" "$T/o1" 4 > /dev/null 2>&1; echo "rc=$?"; sr "$T/o1/4"; wait

t "T8 SIGKILL the run shell mid-step (slow door)"; win 20 60
MOCK_M=slow bash "$S" run "$B" "$T/o8" 1 > /dev/null 2>&1 & sleep 6
inner=$(pgrep -f "bash $S run $B $T/o8" | while read -r p; do [ "$(ps -o comm= -p "$p")" = bash ] && [ "$(ps -o comm= -p "$(ps -o ppid= -p "$p" | tr -d ' ')")" = flock ] && echo "$p"; done)
sp=$(cat $T/o8/.server.*.pid 2>/dev/null); echo "inner=$inner server=$sp"; kill -KILL "$inner"; sleep 4
echo "server alive: $(kill -0 -- "-$sp" 2>/dev/null && echo yes || echo no)"; sleep 2; lockfree; echo "leftover=$(left)"; cat "$T/o8/watch.log"; wait

t "T9 TERM the flock holder mid-step (slow door)"; win 20 60
MOCK_M=slow bash "$S" run "$B" "$T/o9" 1 > /dev/null 2>&1 & sleep 6
fl=$(pgrep -x flock | while read -r p; do grep -q "o9" "/proc/$p/cmdline" && echo "$p"; done); echo "flock=$fl"; kill -TERM "$fl"; sleep 25
sleep 2; lockfree; echo "leftover=$(left)"; cat "$T/o9/watch.log"; wait
