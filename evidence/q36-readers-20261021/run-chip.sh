#!/bin/bash
# L371 chip run, detached. Usage: run-chip.sh TAG SLOTS CTX doc:mode...
# 1) test server (bin-6aa63303, SLOTS rows x CTX, port 18098, never 18091) loads under q36-extra (L366: one extra 35B,
#    memory checked in its lock, oom 1000); it takes the chip lock itself per busy spell, so 18091 keeps serving.
# 2) spark-window --gb10 none (quiet lock: no builds during the timings), run-docs.sh, close. 3) server stopped.
# 2026-10-21 12:11Z: 8 x 18432 was refused by the door (8 rows need 6262 MiB + 8192 kept free, 11695 after the upload).
set -uo pipefail
TAG=$1; SL=$2; CTX=$3; shift 3
E=$(cd "$(dirname "$0")"; pwd); S=$HOME/.local/state/aien-q36; BIN=bin-6aa63303; PORT=18098
LOG=$E/server-$TAG.log
cd "$S/$BIN" && sha256sum -c --quiet SHA256SUMS || exit 2
ss -ltn | grep -q "127.0.0.1:$PORT " && { echo "$PORT in use"; exit 2; }
echo "=== $(date -u +%FT%TZ) $TAG: $BIN $SL x $CTX on $PORT, MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB" | tee -a "$LOG"
setsid "$HOME/.local/bin/q36-extra" ./aien-serve-http --engine omega --model-dir "$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0" \
  --port $PORT --max-context $CTX --max-tokens-cap 16384 --thinking off --batch-slots $SL >> "$LOG" 2>&1 < /dev/null &
SPID=$!
stop() { pkill -TERM -s "$SPID" 2>/dev/null; sleep 5; pkill -KILL -s "$SPID" 2>/dev/null; echo "=== $(date -u +%FT%TZ) $TAG: server stopped" | tee -a "$LOG"; }
until grep -q "serving qwen3.6" "$LOG"; do
  kill -0 $SPID 2>/dev/null || { echo "=== $(date -u +%FT%TZ) $TAG: server ended before serving"; ledger note L371 "chip $TAG: test server ended before serving (see $LOG)"; exit 1; }
  sleep 10
done
echo "=== $(date -u +%FT%TZ) $TAG: serving"
spark-window open "l371-$TAG" --ledger L371 --gb10 none --lease ${LEASE:-90} --wait || { stop; exit 1; }
spark-window run "$E/run-docs.sh" $PORT "$E/chip-$TAG" "$@"
spark-window close
stop
echo "=== $(date -u +%FT%TZ) $TAG: done"
