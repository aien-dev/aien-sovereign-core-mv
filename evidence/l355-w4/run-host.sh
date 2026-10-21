#!/bin/bash
# L355 host runs, one process at a time (each holds one model), under quiet-build so chip timings can get in between.
# Rule: docs/inference/q36/w4-gate.md (frozen before these runs). Resumable: a complete output is skipped.
set -u
B=$HOME/workspace/targets/l355/sc/release/q36_w4_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
export AIEN_Q36_W4_STORE=$M.mxfp4/aien-w4-q36.store
G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision   # L313 f32 reference arm (drift-*.json)
O=$HOME/workspace/sc-l355/evidence/l355-w4/runs
log() { echo "$(date -u +%FT%TZ) $*" >> $O/run.out; }
done_ok() { [ -s "$1" ] && jq -e '.complete != false' "$1" > /dev/null 2>&1; }
run() { # name args...
  local name=$1; shift
  done_ok "$O/$name.json" && { log "skip $name"; return; }
  log "start $name MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
  quiet-build nice -n 10 "$B" "$@" "$O/$name.json" > "$O/$name.log.txt" 2>&1
  log "end $name exit $?"
}
until grep -q '^exit 0' $HOME/workspace/sc-l355/evidence/l355-w4/store-write.log 2>/dev/null; do
  grep -q '^exit [1-9]' $HOME/workspace/sc-l355/evidence/l355-w4/store-write.log 2>/dev/null && { log "STOP: store write failed"; exit 3; }
  sleep 30; done
# 1. W4 goldens for the chip gate (host 4-bit greedy, 32 steps, the chip gate's length)
for p in p1 p2 p3 p4; do run golden-w4e-$p forced "$M" "$G/gen-$p.json" 32 w4e; done
# 2. exactness control (fp8 arm vs L313 f32 ids/margins), then the lever
run forced-fp8-p1 forced "$M" "$R/drift-p1.json" 512 fp8
for p in p1 p2 p3 p4 chat-max-24; do
  until done_ok "$R/drift-$p.json"; do sleep 60; done
  run forced-w4e-$p forced "$M" "$R/drift-$p.json" 512 w4e
done
run score-fp8 score "$M" "$G/passage-agpl.txt" 400 fp8
run score-w4e score "$M" "$G/passage-agpl.txt" 400 w4e
run score-w4e-mut score "$M" "$G/passage-agpl.txt" 400 w4e-mut
for p in p1 p2; do run forced-w4e-mut-$p forced "$M" "$R/drift-$p.json" 512 w4e-mut; done
# 3. MTP acceptance (fp8 weights), then the trunk lever
for p in p1 p2 p3 p4 chat-max-24; do run mtp-fp8-$p mtp "$M" "$R/drift-$p.json" 512 fp8; done
for p in p1 p2 p3 p4 chat-max-24; do run forced-w4et-$p forced "$M" "$R/drift-$p.json" 512 w4et; done
run score-w4et score "$M" "$G/passage-agpl.txt" 400 w4et
log "ALL DONE"
