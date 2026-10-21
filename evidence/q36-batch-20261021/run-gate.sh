#!/bin/bash
# L354 chip gate runner (detached). Each test server loads under q36-extra (serve-batch.sh: one extra 35B at a
# time beside 18091, it waits in line), then: real batch door on 18094 -> gate-batch.sh real (must PASS); row-swap mutant
# (AIEN_Q36_TEST_ROW_SWAP=40) -> gate-batch.sh swap40 (must FAIL). Log run-gate.log, result run-gate.result.
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; S=$HOME/.local/state/aien-q36; SHA=${1:-8b5e95bb}; SLOTS=${2:-32}; export PORT=18094
log() { echo "$(date -u +%FT%TZ) $*" >> run-gate.log; }
up() { for i in $(seq 1 1000); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && return 0; kill -0 $(cat $S/batch-$PORT.pid) 2>/dev/null || return 1; sleep 5; done; return 1; }
down() { local p=$(cat $S/batch-$PORT.pid); ps -o args= -p $p | grep -q "port $PORT" && kill $p; for i in $(seq 1 30); do ps -p $p >/dev/null || break; sleep 2; done; log "stopped $p"; }
run() { # <label> <want rc> [env]
  local label=$1 want=$2; shift 2
  for t in 1 2 3; do
    log "$label: start ($*)"; env "$@" ./serve-batch.sh $SHA $SLOTS 2048 $label >> run-gate.log 2>&1
    if ! up; then log "$label: server did not come up: $(tail -1 $S/batch-$PORT.log | cut -c1-300)"; down; sleep 60; continue; fi
    ./gate-batch.sh 18091 $PORT $label >> run-gate.log 2>&1; rc=$?; down
    [ $rc = 2 ] && { log "$label: no verdict, retry"; sleep 60; continue; }
    log "$label: rc $rc (want $want)"; echo "$label rc=$rc want=$want" >> run-gate.result; return 0
  done
  echo "$label rc=2 want=$want (no verdict after 3 tries)" >> run-gate.result
}
log "runner start (bin-$SHA, $SLOTS slots)"
run real$SLOTS 0
[ "${SWAP:-1}" = 1 ] && run swap40-$SLOTS 1 AIEN_Q36_TEST_ROW_SWAP=40
log "done"; echo done >> run-gate.result
