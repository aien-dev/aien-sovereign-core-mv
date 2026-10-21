#!/bin/bash
# L354: the v2 build's gate runs in parallel (18089: 8 slots real + swap; 18088: 32 slots real), then the timing
# (v2, v1, rows in turn) in one spark-window. Log chain-v2.log.
cd "$(dirname "$0")"; log() { echo "$(date -u +%FT%TZ) $*" >> chain-v2.log; }
until grep -q done run-gate-18089.result 2>/dev/null && grep -q done run-gate-18088.result 2>/dev/null; do sleep 30; done
log "v2 gate: $(cat run-gate-18089.result run-gate-18088.result | tr '\n' ' ')"
if grep -q 'real8 rc=0' run-gate-18089.result && grep -q 'swap40-8 rc=1' run-gate-18089.result && grep -q 'real32 rc=0' run-gate-18088.result; then
  ./time-run.sh 8144655d; log "timing rc $?"
else log "v2 gate not green: no timing"; fi
log done
