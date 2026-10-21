#!/bin/bash
# L355 MTP acceptance runs (host, FP8 main model), started beside run-host.sh (that queue is disk-bound; run-host.sh skips
# outputs already complete). Chat waits until L313's drift-chat-max-24.json is complete.
set -u
B=$HOME/workspace/targets/l355/sc/release/q36_w4_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision
O=$HOME/workspace/sc-l355/evidence/l355-w4/runs
log() { echo "$(date -u +%FT%TZ) [mtp] $*" >> $O/run.out; }
for p in p1 p2 p3 p4 chat-max-24; do
  until [ -s $R/drift-$p.json ] && jq -e '.complete != false' $R/drift-$p.json >/dev/null 2>&1; do sleep 60; done
  log "start mtp-fp8-$p MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
  quiet-build nice -n 10 $B mtp $M $R/drift-$p.json 512 fp8 $O/mtp-fp8-$p.json > $O/mtp-fp8-$p.log.txt 2>&1
  log "end mtp-fp8-$p exit $?"
done
