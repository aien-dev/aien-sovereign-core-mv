#!/bin/bash
# L369 timing receipt (v3 build): q36-extra slot FIRST (waits its turn, Q36_EXTRA_LEDGER=L369), then ONE spark-window
# (chip + quiet) for the run; chunked (128) vs token by token at 256/2048/8192 in one process, bit for bit, wall times.
set -u; D=$HOME/workspace/targets/l369; E=$D/ev; B=$E/qwen36_prefill_gate.v3; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
log() { echo "$(date -u +%FT%TZ) $*" >> $E/time-v3.log; }
if [ "${1:-}" = inner ]; then
  log "slot held; waiting for a window"
  spark-window open l369-prefill --ledger L369 --lease 45 --wait >> $E/time-v3.log 2>&1 || { log "no window"; exit 3; }
  spark-window run $B $M $D/corpus.txt 256,2048,8192 128 16 $E/full-time-v3.json > $E/full-time-v3.log 2>&1; rc=$?
  spark-window close >> $E/time-v3.log 2>&1; log "run rc $rc; window closed"; exit $rc
fi
log "start: $(sha256sum $B | cut -c1-16)"
Q36_EXTRA_LEDGER=L369 q36-extra $D/time-v3.sh inner; log "done rc $?"
