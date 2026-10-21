#!/bin/bash
# L355: q36_levers (L356's identity + timing tool for the served 35B decode graph) on the same build, FP8 then W4 experts.
# Identity (correctness) under the chip lock; the timing it prints is chip-exclusive but NOT a spark-window receipt
# (the window line was deadlocked, L361): reported as indicative; the window timing (timing.sh) stays queued.
set -u
B=$HOME/workspace/targets/l355/sc/release/q36_levers
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
E=$HOME/workspace/sc-l355/evidence/l355-w4; cd $E
G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/levers-ab.log.txt; }
for arm in w4e fp8; do
  for try in $(seq 1 30); do
    if [ $arm = w4e ]; then export AIEN_Q36_W4_STORE=default; GS="gold-w4e-p1.json gold-w4e-p2.json gold-w4e-p3.json gold-w4e-p4.json gold-w4e-chat-max-24.json"
    else unset AIEN_Q36_W4_STORE; GS="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json"; fi
    log "$arm try $try bin $(sha256sum $B | cut -c1-16) memavail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
    flock /tmp/aien-gb10.lock nice -n 5 $B $M $E/levers-$arm.json $GS > $E/levers-$arm.log.txt 2>&1; rc=$?
    log "$arm exit=$rc $(tail -1 $E/levers-$arm.log.txt | cut -c1-240)"
    grep -q 'refusing\|status 0x51' $E/levers-$arm.log.txt || break; sleep 120
  done
done
log "levers-ab done"
