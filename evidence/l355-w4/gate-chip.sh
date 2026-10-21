#!/bin/bash
# L355 chip identity gate for the 4-bit experts: chip (W4 store) vs host reference reading the SAME store.
# Goldens = host w4e greedy (q36_w4_drift forced, self-reference) converted to {prompt_ids, generated_ids}.
# Red before green: mutant 4 (drops shared expert) and W4 mutant (mantissa bit cleared on chip only) must FAIL.
set -u
B=$HOME/workspace/targets/l355/sc/release/qwen38_omega_full
D=$HOME/workspace/targets/l355/sc/release/q36_w4_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
E=$HOME/workspace/sc-l355/evidence/l355-w4
G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
export AIEN_Q36_W4_STORE=default
WL=${WL:-40}
cd $E || exit 2
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/gate-chip.log.txt; }
for p in p1 p2 p3 p4; do
  until [ -s runs/golden-w4e-$p.json ] && jq -e '.complete==true' runs/golden-w4e-$p.json >/dev/null 2>&1; do sleep 20; done
  jq -c '{prompt_ids, generated_ids: .ref_ids, source: "q36_w4_drift forced w4e greedy self-reference (L355)", store: .store}' runs/golden-w4e-$p.json > gold-w4e-$p.json
done
log "goldens ready; bin $(sha256sum $B | cut -c1-16) lib $(cut -c1-16 $HOME/workspace/targets/l355/omega/libomega_gpu.a.sha256) memavail_kb $(grep MemAvailable /proc/meminfo | awk '{print $2}')"
run() { # label env...
  local label=$1; shift
  log "$label waiting for the chip lock"
  env "$@" flock /tmp/aien-gb10.lock nice -n 5 $B $M $WL $GOLD $E/chip-$label.json > $E/chip-$label.log.txt 2>&1; rc=$?
  log "$label exit=$rc: $(grep '^QWEN38_OMEGA_FULL' $E/chip-$label.log.txt | cut -c1-320)"
  return $rc
}
GOLD=gold-w4e-p1.json run p1-mutant4 AIEN_Q36_TEST_MUTANT=4 && { log "STOP: mutant 4 did not go red"; exit 4; }
GOLD=gold-w4e-p1.json run p1-w4mut AIEN_Q36_W4_TEST_MUTANT=1 && { log "STOP: W4 mutant did not go red"; exit 4; }
for p in p1 p2 p3 p4; do GOLD=gold-w4e-$p.json run $p AIEN_L355=1; done
log "gate-chip done"
