#!/bin/bash
# L369 full-35B prefill gate (qwen36_prefill_gate, 40 layers): chunked (128) vs token by token, bit for bit, at 256/2048/8192.
#   full-run.sh value   exactness (chip lock only, under q36-extra): real + mutant 1 at 256 (must FAIL)
#   full-run.sh time    receipts: inside ONE spark-window (chip + quiet; opens/closes it itself), q36-extra waits at most
#                       300 s for the extra-35B slot (its holder may be waiting for the chip we hold), real only.
# Logs/JSON in ev/full-<mode>-*. Never run under flock: value takes the chip lock itself, time holds it via the window.
set -u
D=$HOME/workspace/targets/l369; cd "$D"; E=$D/ev; MODE=${1:?value|time}
B=$D/sc/release/qwen36_prefill_gate; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; LENS=${LENS:-256,2048,8192}
log() { echo "$(date -u +%FT%TZ) $*" >> $E/full-run.log; }
cp $B $E/qwen36_prefill_gate.$MODE; B=$E/qwen36_prefill_gate.$MODE   # the binary measured stays fixed while builds go on
if [ $MODE = value ]; then
  log "value: real $LENS"; q36-extra flock /tmp/aien-gb10.lock $B $M corpus.txt $LENS 128 16 $E/full-value-real.json > $E/full-value-real.log 2>&1; log "value real rc $?"
  log "value: mutant 1 at 256"; AIEN_OMEGA_Q36_PF_MUTANT=1 q36-extra flock /tmp/aien-gb10.lock $B $M corpus.txt 256 128 4 $E/full-value-mut1.json > $E/full-value-mut1.log 2>&1; log "value mut1 rc $? (want 1)"
  exit 0
fi
if [ -z "${SPARK_WINDOW:-}" ]; then
  log "time: waiting for a window"; spark-window open l369-prefill --ledger L369 --lease 60 --wait >> $E/full-run.log 2>&1 || { log "no window"; exit 3; }
  spark-window run "$D/full-run.sh" time; rc=$?; spark-window close >> $E/full-run.log 2>&1; log "window closed (rc $rc)"; exit $rc
fi
log "time: window $SPARK_WINDOW"
Q36_EXTRA_WAIT=300 q36-extra $B $M corpus.txt $LENS 128 16 $E/full-time-real.json > $E/full-time-real.log 2>&1; log "time real rc $?"
