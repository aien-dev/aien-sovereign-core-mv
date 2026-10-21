#!/bin/bash
# L355 chip identity gate, W4 trunk + experts (AIEN_Q36_W4=all): chip vs host reading the same store. Red before green.
set -u
B=$HOME/workspace/targets/l355/sc/release/qwen38_omega_full
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
E=$HOME/workspace/sc-l355/evidence/l355-w4; cd $E
export AIEN_Q36_W4_STORE=default AIEN_Q36_W4=all
log() { echo "$(date -u +%FT%TZ) [w4et] $*" | tee -a $E/gate-chip-w4et.log.txt; }
for p in p1 p2 p3 p4; do until [ -s gold-w4et-$p.json ] && jq -e '.generated_ids|length==32' gold-w4et-$p.json >/dev/null 2>&1; do sleep 30; done; done
run() { # label golden env...
  local label=$1 gold=$2; shift 2
  for try in $(seq 1 10); do
    log "$label try $try waiting for the chip lock (memavail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo))"
    env "$@" flock /tmp/aien-gb10.lock nice -n 5 $B $M 5 $gold $E/chip-w4et-$label.json > $E/chip-w4et-$label.log.txt 2>&1; rc=$?
    log "$label exit=$rc: $(grep '^QWEN38_OMEGA_FULL' $E/chip-w4et-$label.log.txt | cut -c1-320)$( [ $rc = 2 ] && tail -1 $E/chip-w4et-$label.log.txt | cut -c1-160)"
    [ $rc -ne 2 ] && return $rc; sleep 120
  done; return 2
}
run p1-w4mut gold-w4et-p1.json AIEN_Q36_W4_TEST_MUTANT=1; [ $? -eq 1 ] || { log "STOP: W4 mutant did not give a FAIL verdict"; exit 4; }
for p in p1 p2 p3 p4; do run $p gold-w4et-$p.json AIEN_L355=1; done
log "done"
