#!/bin/bash
# L354: after 8144655d's swap control (18089) ends: gate 16b0e6f8 (v2 + a lone row on its own graph) in parallel
# (18089: 8 slots real + swap; 18088: 32 slots real), then the timing (v2, v1, rows in turn) in one spark-window.
cd "$(dirname "$0")"; log() { echo "$(date -u +%FT%TZ) $*" >> chain-v2.log; }
until grep -q done run-gate-18089.result 2>/dev/null; do sleep 30; done
mv run-gate-18089.result run-gate-8144655d-8.result; mv run-gate-18088.result run-gate-8144655d-32.result
for d in real8 swap40-8 real32; do [ -d $d ] && rm -rf $d-8144655d && cp -r $d $d-8144655d; done
log "8144655d gate: $(cat run-gate-8144655d-8.result run-gate-8144655d-32.result | tr '\n' ' ')"
S=16b0e6f8
{ PORT=18089 ./run-gate.sh $S 8; } & { PORT=18088 SWAP=0 ./run-gate.sh $S 32; } & wait
log "$S gate: $(cat run-gate-18089.result run-gate-18088.result | tr '\n' ' ')"
if grep -q 'real8 rc=0' run-gate-18089.result && grep -q 'swap40-8 rc=1' run-gate-18089.result && grep -q 'real32 rc=0' run-gate-18088.result; then
  mv run-gate-18089.result run-gate-$S-8.result; mv run-gate-18088.result run-gate-$S-32.result
  for d in real8 swap40-8 real32; do rm -rf $d-$S; cp -r $d $d-$S; done
  ./time-run.sh $S; log "timing rc $?"
else log "$S gate not green: no timing"; fi
log done
