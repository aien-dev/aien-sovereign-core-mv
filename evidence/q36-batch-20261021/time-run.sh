#!/bin/bash
# L354 timing run, inside ONE spark-window (chip + quiet locks): for each mode (stroke, then rows in turn =
# AIEN_Q36_STROKE=0) load the batch door (32 slots) on 18089 under q36-extra, time-batch.sh, stop it. The servers start
# inside the window, so they carry AIEN_GB10_LOCK_HELD=1 and do not queue behind it. q36-extra waits at most
# Q36_EXTRA_WAIT=300 s for the extra-35B slot (its holder may be waiting for the chip we hold), else the mode is skipped.
#   time-run.sh <sha8>     (opens and closes the window itself; log time-run.log)
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; S=$HOME/.local/state/aien-q36; SHA=${1:?sha8}; export PORT=18089
log() { echo "$(date -u +%FT%TZ) $*" >> time-run.log; }
if [ -z "${SPARK_WINDOW:-}" ]; then
  log "waiting for a window"; spark-window open q36-batch --ledger L354 --lease 75 --wait >> time-run.log 2>&1 || { log "no window"; exit 3; }
  spark-window run "$D/time-run.sh" "$SHA"; rc=$?; spark-window close >> time-run.log 2>&1; log "window closed (rc $rc)"; exit $rc
fi
up() { for i in $(seq 1 1000); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && return 0; kill -0 $(cat $S/batch-$PORT.pid) 2>/dev/null || return 1; sleep 5; done; return 1; }
down() { local p=$(cat $S/batch-$PORT.pid); ps -o args= -p $p | grep -q "port $PORT" && kill $p; for i in $(seq 1 30); do ps -p $p >/dev/null || break; sleep 2; done; log "stopped $p"; }
log "window $SPARK_WINDOW"
for mode in stroke inturn; do
  st=1; [ $mode = inturn ] && st=0
  log "$mode: load ($SHA, 32 slots)"; Q36_EXTRA_WAIT=300 AIEN_Q36_STROKE=$st ./serve-batch.sh $SHA 32 2048 time-$mode >> time-run.log 2>&1
  up || { log "$mode: server did not come up: $(tail -1 $S/batch-$PORT.log | cut -c1-200)"; down; continue; }
  log "$mode: up"; ./time-batch.sh $PORT time-$mode-$SHA >> time-run.log 2>&1; log "$mode: timing rc $?"; down
done
log done
