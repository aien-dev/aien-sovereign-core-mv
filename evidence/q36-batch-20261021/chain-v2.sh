#!/bin/bash
# L354: after the running run-stroke.sh 16b27643 ends: the v2 build through the same gate (8 real + swap, 32 real),
# then the timing (v2, v1, rows in turn) in one spark-window. Log chain-v2.log.
cd "$(dirname "$0")"; log() { echo "$(date -u +%FT%TZ) $*" >> chain-v2.log; }
while pgrep -f '^/bin/bash ./run-stroke.sh 16b27643' >/dev/null; do sleep 30; done
log "v1 gate done: $(cat run-gate-16b27643-8.result run-gate-16b27643-32.result 2>/dev/null | tr '\n' ' ')"
./run-stroke.sh 8144655d; log "v2 gate: $(cat run-gate-8144655d-8.result run-gate-8144655d-32.result 2>/dev/null | tr '\n' ' ')"
if grep -q 'real8 rc=0' run-gate-8144655d-8.result && grep -q 'swap40-8 rc=1' run-gate-8144655d-8.result; then
  ./time-run.sh 8144655d; log "timing rc $?"
else log "v2 gate not green: timing v1 build only"; MODES="stroke inturn" ./time-run.sh 16b27643; fi
log done
