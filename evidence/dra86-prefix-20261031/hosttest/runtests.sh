#!/usr/bin/env bash
# Host check of window-dra86.sh against the mock door (fake-serve.sh -> mock86.py); scratch lock, never the chip.
# Run as: nice -n 19 taskset -c 5 bash evidence/dra86-prefix-20261031/hosttest/runtests.sh  (port 18986)
set -u
H=$(cd "$(dirname "$0")" && pwd); S=$(cd "$H/.." && pwd)/window-dra86.sh
T=${DRA86_HOSTTEST_DIR:-/tmp/dra86-hosttest}; rm -rf "$T"; mkdir -p "$T/win"
export DRA86_KEYSEEN=$T/keys.seen SPARK_WINDOW=w-test-86 SPARK_WINDOW_DIR=$T/win DRA86_TEST_LOCK=$T/t.lock DRA86_TEST_PORT=18986
: > "$DRA86_TEST_LOCK"; B=$H/fake-serve.sh; fails=0
win() { rm -f "$T/win/expired"; date -u +%FT%TZ > "$T/win/held"; echo 60 > "$T/win/lease"; }
sr() { jq -c '{step, rc, verdict, why, r: .result.verdict, rwhy: .result.why, same: .result.same, diff: .result.diff, restored: .result.restored, turn2: .result.turn2}' "$1/step-receipt.json" 2>/dev/null || echo "no receipt $1"; }
expect() { # <name> <want rc> <got rc>
  if [ "$2" = "$3" ]; then echo "ok   $1 (rc $3)"; else echo "FAIL $1: want rc $2, got $3"; fails=$((fails + 1)); fi
}
t() { echo; echo "== $*"; }

t "T0 plan"; bash "$S" plan; expect "plan" 0 $?
t "T1 no SPARK_WINDOW -> NOT_RUN"; env -u SPARK_WINDOW bash "$S" run "$B" "$T/o0" 1 2>&1 | tail -n 1; expect "no window" 3 ${PIPESTATUS[0]}
t "T2 every step on the exact mock door"; win
bash "$S" run "$B" "$T/o1" 1 2 3 4a 4b 4c 5 6a 6b > /dev/null 2> "$T/t2.err"; rc=$?; expect "all steps" 0 $rc
for s in 1 2 3 4a 4b 4c 5 6a 6b; do sr "$T/o1/$s"; done
jq -c '{kind, run, ok, http_ttft_s, reused_tok, door_restore_ms, door_prefill_ms}' "$T/o1/6a/time.jsonl" "$T/o1/6b/time.jsonl"
nk=$(sort -u "$DRA86_KEYSEEN" | wc -l); nl=$(wc -l < "$DRA86_KEYSEEN"); leak=$(grep -rlF -f "$DRA86_KEYSEEN" "$T/o1" | wc -l)
echo "store key: $nl server starts got it, $nk distinct, files under the receipts that contain it: $leak"
[ "$nl" = 3 ] && [ "$nk" = 1 ] && [ "$leak" = 0 ]; expect "key reached 4a 4b 4c, one key, never in a receipt" 0 $?
grep -c 'GPU-START\|GPU-END' "$T/o1/board.log"
t "T3 red mutant that changes nothing (MOCK_M=nomut): step 2 FAIL"; win
MOCK_M=nomut bash "$S" run "$B" "$T/o1" 2 > /dev/null 2>&1; expect "red without effect" 1 $?; sr "$T/o1/2"
t "T4 green after that failed red: NOT_RUN (red receipt not passing)"; win
bash "$S" run "$B" "$T/o1" 3 2>&1 >/dev/null | tail -n 1; expect "green without red" 3 ${PIPESTATUS[0]}
t "T5 red again, then a door that never restores (MOCK_M=norestore): red INCONCLUSIVE"; win
MOCK_M=norestore bash "$S" run "$B" "$T/o1" 2 > /dev/null 2>&1; expect "red without restore" 2 $?; sr "$T/o1/2"
t "T6 exact red, then green on a door that never restores: green FAIL"; win
bash "$S" run "$B" "$T/o1" 2 > /dev/null 2>&1; expect "red" 0 $?
MOCK_M=norestore bash "$S" run "$B" "$T/o1" 3 > /dev/null 2>&1; expect "green without restore" 1 $?; sr "$T/o1/3"
t "T7 store that loads another namespace (MOCK_M=ignore_ns): 4c FAIL"; win
MOCK_M=ignore_ns bash "$S" run "$B" "$T/o1" 4a 4b 4c > /dev/null 2>&1; expect "stale store loaded" 1 $?; sr "$T/o1/4c"
t "T8 4b without 4a in the run: NOT_RUN"; win
bash "$S" run "$B" "$T/o1" 4b 2>&1 >/dev/null | tail -n 1; expect "reload alone" 3 ${PIPESTATUS[0]}
t "T9 green on an empty OUT: NOT_RUN"; win
bash "$S" run "$B" "$T/o9" 3 2>&1 >/dev/null | tail -n 1; expect "green on empty OUT" 3 ${PIPESTATUS[0]}
t "T10 budget step with the default budget (step 3 env, probe budget): FAIL (no eviction, not 0.5 GiB)"
AIEN_Q36_PREFIX=1 python3 "$H/mock86.py" 18987 2> "$T/t10.log" & mp=$!; sleep 1
SHA=$(sha256sum < "$B" | cut -c1-64)
DRA86_SERVER_LOG=$T/t10.log bash "$S" conv 18987 "$T/o10" budget "$SHA" x "$T/o1/1/conv.jsonl" | jq -c '{verdict, why}'; expect "budget at 16 GiB" 1 ${PIPESTATUS[0]}
kill $mp; wait $mp 2>/dev/null
echo; leftover=$(pgrep -f 'mock86.py|window-dra86.sh _watch' | wc -l); echo "leftover processes: $leftover"
flock -n "$DRA86_TEST_LOCK" true && echo "lock free" || { echo "lock HELD"; fails=$((fails + 1)); }
echo "runtests: $fails failed"; [ $fails = 0 ]
