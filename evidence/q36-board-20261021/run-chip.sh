#!/bin/bash
# L376 chip run, detached. Usage: run-chip.sh TAG BIN
# 1) test server (BIN, 8 rows x 16384, port $PORT: 18098 per the coordinator map 12:52Z, or PORT=... when that port is taken; never 18091) loads under q36-extra (L366: one extra 35B,
#    memory checked in its lock, oom 1000), its own board dir (never the live server's).
# 2) spark-window --gb10 none (quiet lock: no builds during the timings), gate.sh, close. 3) server stopped.
set -uo pipefail
TAG=$1; BIN=$2
E=$(cd "$(dirname "$0")"; pwd); S=$HOME/.local/state/aien-q36; PORT=${PORT:-18098}
export Q36_EXTRA_LEDGER=L376   # q36-extra v3 priority rank (coordinator 13:54Z)
export Q36_EXTRA_WAIT=${Q36_EXTRA_WAIT:-14400}   # the extra-35B line is long today; wait up to 4 h for the slot
LOG=$E/server-$TAG.log; BD=$E/board-$TAG; OUT=$E/chip-$TAG
cd "$S/$BIN" && sha256sum -c --quiet SHA256SUMS || exit 2
ss -ltn | grep -q "127.0.0.1:$PORT " && { echo "$PORT in use"; exit 2; }
rm -rf "$BD"; mkdir -p "$BD" "$OUT"
echo "=== $(date -u +%FT%TZ) $TAG: $BIN 8 x 16384 on $PORT, board $BD, MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB" | tee -a "$LOG"
AIEN_BOARD_DIR=$BD setsid "$HOME/.local/bin/q36-extra" ./aien-serve-http --engine omega --model-dir "$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0" \
  --port $PORT --max-context 16384 --max-tokens-cap 16384 --thinking off --batch-slots 8 >> "$LOG" 2>&1 < /dev/null &
SPID=$!
stop() { pkill -TERM -s "$SPID" 2>/dev/null; sleep 5; pkill -KILL -s "$SPID" 2>/dev/null; echo "=== $(date -u +%FT%TZ) $TAG: server stopped" | tee -a "$LOG"; }
until grep -q "serving qwen3.6" "$LOG"; do
  kill -0 $SPID 2>/dev/null || { echo "=== $(date -u +%FT%TZ) $TAG: server ended before serving"; ledger note L376 "chip $TAG: test server ended before serving (see $LOG)"; exit 1; }
  sleep 10
done
echo "=== $(date -u +%FT%TZ) $TAG: serving"
spark-window open "l376-$TAG" --ledger L376 --gb10 none --wait || { stop; exit 1; }
spark-window run "$E/gate.sh" $PORT "$OUT" "$LOG" "$BD"
RC=$?
spark-window close
stop
echo "=== $(date -u +%FT%TZ) $TAG: done rc=$RC"
exit $RC
