#!/bin/bash
# L313 chip gates for the bf16 DeltaNet state (AIEN_OMEGA_GDN_STATE_BF16=1) on the 35B batch door, detached.
# One test server at a time on 127.0.0.1:$PORT (18098), each loaded under q36-extra by L354's serve-batch.sh (L366 rule),
# 8 slots, context 2048, the L354 gate gate-batch.sh (32 prompts at once + turn 2 + golden):
#   f32   switch off, alone = alone/ (the f32 single door's replies): must PASS (the lib's f32 path unchanged end to end)
#   bf16  switch on, alone collected one at a time on the same server (alone-bf16/), then all at once: must PASS
#   swap  switch on + AIEN_Q36_TEST_ROW_SWAP=40 (rows' outputs swapped once), against alone-bf16/: must FAIL
# then drift.sh: alone-bf16 vs alone (f32 ids), first differing token per prompt -> drift.json.
#   run-door.sh <bin-sha8>     log run-door.log, verdicts run-door.result
set -u
D=$(cd "$(dirname "$0")" && pwd); G=$D/../q36-batch-20261021; S=$HOME/.local/state/aien-q36; SHA=${1:?sha8}; export PORT=18098
cd "$D"
log() { echo "$(date -u +%FT%TZ) $*" >> run-door.log; }
up() { for i in $(seq 1 1000); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && return 0; kill -0 $(cat $S/batch-$PORT.pid) 2>/dev/null || return 1; sleep 5; done; return 1; }
down() { local p=$(cat $S/batch-$PORT.pid); ps -o args= -p $p | grep -q "port $PORT" && kill $p; for i in $(seq 1 30); do ps -p $p >/dev/null || break; sleep 2; done; log "stopped $p"; }
mem() { local p=$(pgrep -f "aien-serve-http.*--port $PORT" | head -1); log "$1 memory: $(grep -E 'VmRSS|VmHWM' /proc/$p/status 2>/dev/null | tr -s ' \n' ' '), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"; }
run() { # <label> <want rc> <alone dir> [env...]
  local label=$1 want=$2 al=$3; shift 3
  for t in 1 2 3; do
    log "$label: start ($*)"; (cd $G && env "$@" ./serve-batch.sh $SHA 8 2048 l313-$label) >> run-door.log 2>&1
    if ! up; then log "$label: server did not come up: $(tail -1 $S/batch-$PORT.log | cut -c1-300)"; down; sleep 60; continue; fi
    mem "$label idle"
    (cd $G && ALONE=$al ./gate-batch.sh $PORT $PORT l313-$label) >> run-door.log 2>&1; rc=$?
    mem "$label after gate"; down
    [ $rc = 2 ] && { log "$label: no verdict, retry"; sleep 60; continue; }
    log "$label: rc $rc (want $want)"; echo "$label rc=$rc want=$want" >> run-door.result; return 0
  done
  echo "$label rc=2 want=$want (no verdict after 3 tries)" >> run-door.result
}
log "runner start (bin-$SHA)"
run f32 0 alone AIEN_OMEGA_GDN_STATE_BF16=0
run bf16 0 $D/alone-bf16 AIEN_OMEGA_GDN_STATE_BF16=1
run swap 1 $D/alone-bf16 AIEN_OMEGA_GDN_STATE_BF16=1 AIEN_Q36_TEST_ROW_SWAP=40
./drift.sh >> run-door.log 2>&1
log "done"; echo done >> run-door.result
