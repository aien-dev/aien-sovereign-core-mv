#!/bin/bash
# Rerun named prompts of gate-chip.sh; an error exit (2: memory guard refused, channel alloc failed) is not a verdict: wait, retry.
set -u
B=$HOME/workspace/targets/l355/sc/release/qwen38_omega_full
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
E=$HOME/workspace/sc-l355/evidence/l355-w4; cd $E
export AIEN_Q36_W4_STORE=default
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/gate-chip.log.txt; }
for p in "$@"; do
  for try in 1 2 3 4 5 6 7 8; do
    log "$p try $try waiting for the chip lock (memavail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo))"
    flock /tmp/aien-gb10.lock nice -n 5 $B $M ${WL:-5} gold-w4e-$p.json $E/chip-$p.json > $E/chip-$p.log.txt 2>&1; rc=$?
    log "$p exit=$rc: $(grep '^QWEN38_OMEGA_FULL' $E/chip-$p.log.txt | cut -c1-320)$( [ $rc = 2 ] && tail -1 $E/chip-$p.log.txt | cut -c1-160)"
    [ $rc -ne 2 ] && break; sleep 120
  done
done
log "retry done"
